//! Grimoire term dictionaries.
//!
//! FSLX TERMDICT sections use field-namespaced, prefix-compressed blocks. The
//! durable bytes contain a block-first-key index and full restart keys every
//! [`TERM_RESTART_INTERVAL`] entries. Opening a dictionary validates the whole
//! section and builds a bounded in-memory restart directory; exact lookup then
//! decodes at most one restart group while ordered scans reuse one key buffer.

mod codec;
mod cursor;
mod encoder;
mod types;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::cmp::Ordering;
use std::ops::{Bound, Range};

use crate::schema::SchemaDescriptor;

use codec::{
    ReferenceValidator, SliceReader, build_composite_in, clone_bytes, composite_key, decode_entry,
    range_is_reversed, usize_from_vint, validate_block, validate_composite_key,
    validate_positions_section, validate_schema,
};
pub(crate) use codec::{validate_bound_term, validate_query_term};
use cursor::{
    CursorBound, collect_glob_matches, contains_subslice, glob_limit_error, longest_glob_literal,
};
pub use cursor::{TermCursor, TermScratch};
pub(crate) use cursor::{star_glob_matches, trailing_star_prefix};
pub use encoder::EncodedTermDictionary;
pub(crate) use types::ValidatedTermDictionaryMetadata;
use types::{BlockMeta, IndexRecord, RestartMeta};
pub use types::{
    ByteSpan, OwnedTerm, TermDictionaryError, TermDictionaryLimits, TermInput, TermMatch,
    TermMetadata, TermRef, TermSectionLengths,
};

/// Number of entries between full-key restart points within a block.
pub const TERM_RESTART_INTERVAL: usize = 16;
/// Canonical target size for an ordinary TERMDICT block, including its header.
pub const TERM_BLOCK_TARGET_BYTES: usize = 4_096;
/// Language-contract maximum for the term portion of a composite key.
pub const MAX_TERM_BYTES: usize = 65_530;
/// Maximum composite key size: big-endian field ordinal plus term bytes.
pub const MAX_COMPOSITE_KEY_BYTES: usize = MAX_TERM_BYTES + 2;
/// Default cap on a borrowed TERMDICT section.
pub const DEFAULT_MAX_TERM_DICTIONARY_BYTES: usize = 1 << 30;
/// Default cap on block metadata retained by a reader.
pub const DEFAULT_MAX_TERM_BLOCKS: usize = 1 << 20;
/// Default cap on terms eagerly validated by a reader.
pub const DEFAULT_MAX_TERMS: usize = 1 << 24;
/// Default cap on restart metadata retained by a reader.
pub const DEFAULT_MAX_TERM_RESTARTS: usize = 1 << 20;

const BLOCK_COUNT_BYTES: usize = 4;
const BLOCK_ENTRY_COUNT_BYTES: usize = 2;
const MIN_WIRE_BYTES_PER_BLOCK: usize = 14;
const MIN_WIRE_BYTES_PER_ENTRY: usize = 7;

/// Borrowed, eagerly validated TERMDICT view.
#[derive(Clone, Debug)]
pub struct TermDictionary<'a> {
    bytes: &'a [u8],
    schema: SchemaDescriptor,
    sections: TermSectionLengths,
    blocks: Cow<'a, [BlockMeta]>,
    restarts: Cow<'a, [RestartMeta]>,
    term_count: u32,
}

impl<'a> TermDictionary<'a> {
    /// Validate a complete TERMDICT section with default resource caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed, non-canonical, out-of-budget, or
    /// cross-section-inconsistent bytes.
    pub fn parse(
        bytes: &'a [u8],
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
    ) -> Result<Self, TermDictionaryError> {
        Self::parse_with_limits(bytes, schema, sections, TermDictionaryLimits::default())
    }

    /// Validate a complete TERMDICT section with explicit resource caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error before exceeding the supplied byte, block, term,
    /// or restart budgets.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
        limits: TermDictionaryLimits,
    ) -> Result<Self, TermDictionaryError> {
        validate_schema(schema)?;
        validate_positions_section(schema, sections)?;
        if bytes.len() > limits.max_bytes {
            return Err(TermDictionaryError::ByteBudgetExceeded {
                limit: limits.max_bytes,
                actual: bytes.len(),
            });
        }

        let mut reader = SliceReader::new(bytes);
        let block_count = usize::try_from(reader.read_u32()?).map_err(|_| {
            TermDictionaryError::ValueOutOfRange {
                field: "block_count",
                value: u64::MAX,
                offset: 0,
            }
        })?;
        if block_count > limits.max_blocks {
            return Err(TermDictionaryError::BlockBudgetExceeded {
                limit: limits.max_blocks,
                actual: block_count,
            });
        }
        if block_count > limits.max_terms {
            return Err(TermDictionaryError::TermBudgetExceeded {
                limit: limits.max_terms,
                actual: block_count,
            });
        }
        if block_count > limits.max_restarts {
            return Err(TermDictionaryError::RestartBudgetExceeded {
                limit: limits.max_restarts,
                actual: block_count,
            });
        }

        if block_count == 0 {
            let remaining = bytes.len().saturating_sub(reader.position());
            if remaining != 0 {
                return Err(TermDictionaryError::TrailingEmptyBytes { remaining });
            }
            ReferenceValidator::new(schema, sections)?.finish()?;
            return Ok(Self {
                bytes,
                schema,
                sections,
                blocks: Cow::Owned(Vec::new()),
                restarts: Cow::Owned(Vec::new()),
                term_count: 0,
            });
        }

        let feasible = bytes.len().saturating_sub(BLOCK_COUNT_BYTES) / MIN_WIRE_BYTES_PER_BLOCK;
        if block_count > feasible {
            return Err(TermDictionaryError::ImplausibleBlockCount {
                block_count,
                byte_len: bytes.len(),
            });
        }

        let mut index = Vec::new();
        index
            .try_reserve_exact(block_count)
            .map_err(|_| TermDictionaryError::Allocation {
                context: "block index metadata",
                count: block_count,
            })?;
        let mut previous_index_key: Option<Range<usize>> = None;
        for block_index in 0..block_count {
            let key_len_offset = reader.position();
            let key_len = usize_from_vint(reader.read_vint()?, "first_key_len", key_len_offset)?;
            if !(2..=MAX_COMPOSITE_KEY_BYTES).contains(&key_len) {
                return Err(TermDictionaryError::TermTooLong {
                    term_ordinal: block_index,
                    length: key_len.saturating_sub(2),
                });
            }
            let key_range = reader.take_range(key_len)?;
            validate_composite_key(schema, &bytes[key_range.clone()], block_index)?;
            if let Some(previous) = previous_index_key.as_ref() {
                if bytes[key_range.clone()] <= bytes[previous.clone()] {
                    return Err(TermDictionaryError::NonAscendingIndexKey { block_index });
                }
            }
            previous_index_key = Some(key_range.clone());
            let relative_offset = reader.read_vint()?;
            index.push(IndexRecord {
                first_key_range: key_range,
                relative_offset,
            });
        }

        let blocks_start = reader.position();
        let blocks_len = bytes.len().saturating_sub(blocks_start);
        let mut absolute_offsets = Vec::new();
        absolute_offsets
            .try_reserve_exact(block_count)
            .map_err(|_| TermDictionaryError::Allocation {
                context: "absolute block offsets",
                count: block_count,
            })?;
        let mut previous_relative = None;
        for (block_index, record) in index.iter().enumerate() {
            if block_index == 0 && record.relative_offset != 0 {
                return Err(TermDictionaryError::InvalidBlockOffset {
                    block_index,
                    offset: record.relative_offset,
                    detail: "first offset must be zero",
                });
            }
            if previous_relative.is_some_and(|previous| record.relative_offset <= previous) {
                return Err(TermDictionaryError::InvalidBlockOffset {
                    block_index,
                    offset: record.relative_offset,
                    detail: "offsets must be strictly increasing",
                });
            }
            let relative = usize::try_from(record.relative_offset).map_err(|_| {
                TermDictionaryError::InvalidBlockOffset {
                    block_index,
                    offset: record.relative_offset,
                    detail: "offset does not fit host usize",
                }
            })?;
            if relative >= blocks_len {
                return Err(TermDictionaryError::InvalidBlockOffset {
                    block_index,
                    offset: record.relative_offset,
                    detail: "offset lies outside the blocks area",
                });
            }
            absolute_offsets.push(blocks_start.checked_add(relative).ok_or(
                TermDictionaryError::SizeOverflow {
                    field: "absolute block offset",
                },
            )?);
            previous_relative = Some(record.relative_offset);
        }

        let mut blocks = Vec::new();
        let mut restarts = Vec::new();
        let mut references = ReferenceValidator::new(schema, sections)?;
        let mut previous_key = Vec::new();
        let mut decode_key = Vec::new();
        let mut previous_tail = None;
        let mut term_count = 0_usize;

        for block_index in 0..block_count {
            let start = absolute_offsets[block_index];
            let end = absolute_offsets
                .get(block_index + 1)
                .copied()
                .unwrap_or(bytes.len());
            let base_term_ord =
                u32::try_from(term_count).map_err(|_| TermDictionaryError::ValueOutOfRange {
                    field: "term ordinal",
                    value: u64::try_from(term_count).unwrap_or(u64::MAX),
                    offset: start,
                })?;
            let (meta, tail) = validate_block(
                bytes,
                schema,
                sections,
                limits,
                block_index,
                start..end,
                index[block_index].first_key_range.clone(),
                base_term_ord,
                &mut term_count,
                &mut restarts,
                &mut references,
                &mut decode_key,
                &mut previous_key,
                previous_tail.as_ref(),
            )?;
            blocks
                .try_reserve(1)
                .map_err(|_| TermDictionaryError::Allocation {
                    context: "block metadata",
                    count: blocks.len().saturating_add(1),
                })?;
            blocks.push(meta);
            previous_tail = Some(tail);
        }
        references.finish()?;
        let term_count_u32 =
            u32::try_from(term_count).map_err(|_| TermDictionaryError::ValueOutOfRange {
                field: "term_count",
                value: u64::try_from(term_count).unwrap_or(u64::MAX),
                offset: bytes.len(),
            })?;
        Ok(Self {
            bytes,
            schema,
            sections,
            blocks: Cow::Owned(blocks),
            restarts: Cow::Owned(restarts),
            term_count: term_count_u32,
        })
    }

    /// Perform one complete validation and retain only owned bounded metadata.
    ///
    /// The returned value is crate-private and remains bound to the exact live
    /// byte slice by address and length.
    pub(crate) fn validate_metadata(
        bytes: &'a [u8],
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
    ) -> Result<ValidatedTermDictionaryMetadata, TermDictionaryError> {
        let parsed = Self::parse(bytes, schema, sections)?;
        let Self {
            bytes,
            schema,
            sections,
            blocks,
            restarts,
            term_count,
        } = parsed;
        Ok(ValidatedTermDictionaryMetadata {
            source_start: bytes.as_ptr() as usize,
            source_len: bytes.len(),
            schema,
            sections,
            blocks: blocks.into_owned(),
            restarts: restarts.into_owned(),
            term_count,
        })
    }

    /// Construct a cheap borrowed view over metadata validated for these exact
    /// immutable bytes.
    pub(crate) fn from_validated_metadata(
        bytes: &'a [u8],
        metadata: &'a ValidatedTermDictionaryMetadata,
    ) -> Result<Self, TermDictionaryError> {
        if bytes.len() != metadata.source_len || bytes.as_ptr() as usize != metadata.source_start {
            return Err(TermDictionaryError::InvalidSchema {
                detail: "validated TERMDICT metadata is bound to a different byte source"
                    .to_owned(),
            });
        }
        Ok(Self {
            bytes,
            schema: metadata.schema,
            sections: metadata.sections,
            blocks: Cow::Borrowed(&metadata.blocks),
            restarts: Cow::Borrowed(&metadata.restarts),
            term_count: metadata.term_count,
        })
    }

    /// Exact validated durable bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Number of validated terms.
    #[must_use]
    pub const fn term_count(&self) -> u32 {
        self.term_count
    }

    /// Number of validated prefix-compressed blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Number of retained restart metadata entries.
    #[must_use]
    pub fn restart_count(&self) -> usize {
        self.restarts.len()
    }

    /// Exact lookup with temporary scratch allocated by this call.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an unknown/non-term field, an overlong query
    /// term, allocation failure, or an internal validated-decode failure.
    pub fn lookup(
        &self,
        field_ord: u16,
        term: &[u8],
    ) -> Result<Option<TermMatch>, TermDictionaryError> {
        self.lookup_with_scratch(field_ord, term, &mut TermScratch::default())
    }

    /// Exact lookup reusing caller-owned key buffers.
    ///
    /// The block and restart indexes are binary-searched, then no more than 16
    /// entries are decoded.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid query shape, scratch allocation
    /// failure, or an internal validated-decode failure.
    pub fn lookup_with_scratch(
        &self,
        field_ord: u16,
        term: &[u8],
        scratch: &mut TermScratch,
    ) -> Result<Option<TermMatch>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, term)?;
        build_composite_in(&mut scratch.target, field_ord, term, "lookup target")?;
        let target = scratch.target.as_slice();
        let Some(block_index) = self.block_for_key(target) else {
            return Ok(None);
        };
        let block = &self.blocks[block_index];
        let restart_slice = &self.restarts[block.restart_range.clone()];
        let restart_position = restart_slice
            .partition_point(|restart| self.bytes[restart.key_range.clone()] <= *target)
            .saturating_sub(1);
        let restart = &restart_slice[restart_position];
        scratch.decoded.clear();
        let mut reader =
            SliceReader::with_range(self.bytes, restart.entry_offset, block.byte_range.end)?;
        let group_end = restart
            .entry_ordinal
            .saturating_add(TERM_RESTART_INTERVAL)
            .min(block.entry_count);
        for entry_ordinal in restart.entry_ordinal..group_end {
            let restart_delta = entry_ordinal.saturating_sub(restart.entry_ordinal);
            let term_ordinal = restart
                .term_ordinal
                .checked_add(u32::try_from(restart_delta).map_err(|_| {
                    TermDictionaryError::ValueOutOfRange {
                        field: "term ordinal",
                        value: u64::try_from(entry_ordinal).unwrap_or(u64::MAX),
                        offset: reader.position(),
                    }
                })?)
                .ok_or(TermDictionaryError::SizeOverflow {
                    field: "term ordinal",
                })?;
            let decoded = decode_entry(
                &mut reader,
                entry_ordinal,
                &mut scratch.decoded,
                self.schema,
                self.sections,
                usize::try_from(term_ordinal).unwrap_or(usize::MAX),
            )?;
            match scratch.decoded.as_slice().cmp(target) {
                Ordering::Less => {}
                Ordering::Equal => {
                    return Ok(Some(TermMatch {
                        term_ord: term_ordinal,
                        metadata: decoded.metadata,
                    }));
                }
                Ordering::Greater => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Cursor over every term in global field-major order.
    ///
    /// # Errors
    ///
    /// Returns a typed error if scratch allocation or validated decoding fails.
    pub fn cursor(&self) -> Result<TermCursor<'_, 'a>, TermDictionaryError> {
        TermCursor::new(self, None, None, None, None)
    }

    /// Cursor over one schema field in term-byte order.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an unknown/non-term field or cursor setup
    /// failure.
    pub fn field_cursor(&self, field_ord: u16) -> Result<TermCursor<'_, 'a>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, &[])?;
        let lower = CursorBound::included(composite_key(field_ord, &[], 0)?);
        TermCursor::new(self, Some(lower), None, Some(field_ord), None)
    }

    /// Cursor over terms in one field that start with `prefix`.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid field/prefix shape or cursor setup
    /// failure.
    pub fn prefix_cursor(
        &self,
        field_ord: u16,
        prefix: &[u8],
    ) -> Result<TermCursor<'_, 'a>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, prefix)?;
        let lower_key = composite_key(field_ord, prefix, 0)?;
        let prefix_key = clone_bytes(&lower_key, "prefix cursor key")?;
        TermCursor::new(
            self,
            Some(CursorBound::included(lower_key)),
            None,
            Some(field_ord),
            Some(prefix_key),
        )
    }

    /// Cursor over a field-scoped range with standard bound semantics.
    ///
    /// The query contract's canonical half-open form is represented by an
    /// included `start` and excluded `end`; general bounds are accepted for
    /// callers that need an unbounded or inclusive endpoint.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid fields, overlong endpoints, reversed
    /// bounds, or cursor setup failure.
    pub fn range_cursor(
        &self,
        field_ord: u16,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
    ) -> Result<TermCursor<'_, 'a>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, &[])?;
        validate_bound_term(self.schema, field_ord, &start)?;
        validate_bound_term(self.schema, field_ord, &end)?;
        if range_is_reversed(&start, &end) {
            return Err(TermDictionaryError::InvalidRange);
        }
        let lower = match start {
            Bound::Included(term) => {
                Some(CursorBound::included(composite_key(field_ord, term, 0)?))
            }
            Bound::Excluded(term) => {
                Some(CursorBound::excluded(composite_key(field_ord, term, 0)?))
            }
            Bound::Unbounded => Some(CursorBound::included(composite_key(field_ord, &[], 0)?)),
        };
        let upper = match end {
            Bound::Included(term) => {
                Some(CursorBound::included(composite_key(field_ord, term, 0)?))
            }
            Bound::Excluded(term) => {
                Some(CursorBound::excluded(composite_key(field_ord, term, 0)?))
            }
            Bound::Unbounded => None,
        };
        TermCursor::new(self, lower, upper, Some(field_ord), None)
    }

    /// Expand a field-scoped byte glob under an explicit result cap.
    ///
    /// `*` is the only wildcard and matches any byte sequence, including the
    /// empty sequence. Repeated stars are equivalent to one star. Exact terms
    /// use indexed lookup, a sole trailing star uses the prefix cursor, and all
    /// other shapes scan only the requested field. Results retain dictionary
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`TermDictionaryError::GlobExpansionLimitExceeded`] as soon as
    /// one more matching term than `limit` is observed. It also propagates
    /// invalid field, allocation, and validated-decode errors.
    pub fn expand_glob(
        &self,
        field_ord: u16,
        pattern: &[u8],
        limit: usize,
    ) -> Result<Vec<OwnedTerm>, TermDictionaryError> {
        validate_query_term(self.schema, field_ord, &[])?;

        if !pattern.contains(&b'*') {
            return self.expand_exact_glob(field_ord, pattern, limit);
        }

        if let Some(prefix) = trailing_star_prefix(pattern) {
            return collect_glob_matches(
                self.prefix_cursor(field_ord, prefix)?,
                field_ord,
                pattern,
                limit,
                |_| true,
            );
        }

        let longest_literal = longest_glob_literal(pattern);
        collect_glob_matches(
            self.field_cursor(field_ord)?,
            field_ord,
            pattern,
            limit,
            |term| contains_subslice(term, longest_literal) && star_glob_matches(pattern, term),
        )
    }

    fn expand_exact_glob(
        &self,
        field_ord: u16,
        term: &[u8],
        limit: usize,
    ) -> Result<Vec<OwnedTerm>, TermDictionaryError> {
        let Some(term_match) = self.lookup(field_ord, term)? else {
            return Ok(Vec::new());
        };
        if limit == 0 {
            return Err(glob_limit_error(field_ord, term, limit, 1));
        }

        let mut output = Vec::new();
        output
            .try_reserve_exact(1)
            .map_err(|_| TermDictionaryError::Allocation {
                context: "glob expansion rows",
                count: 1,
            })?;
        output.push(OwnedTerm {
            term_ord: term_match.term_ord,
            field_ord,
            term: clone_bytes(term, "exact glob term")?,
            metadata: term_match.metadata,
        });
        Ok(output)
    }

    fn block_for_key(&self, key: &[u8]) -> Option<usize> {
        let insertion = self
            .blocks
            .partition_point(|block| self.bytes[block.first_key_range.clone()] <= *key);
        insertion.checked_sub(1)
    }
}
