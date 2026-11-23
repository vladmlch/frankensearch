use std::ops::Range;

use thiserror::Error;

use crate::schema::SchemaDescriptor;

use super::{
    DEFAULT_MAX_TERM_BLOCKS, DEFAULT_MAX_TERM_DICTIONARY_BYTES, DEFAULT_MAX_TERM_RESTARTS,
    DEFAULT_MAX_TERMS, MAX_TERM_BYTES,
};
/// One byte range inside another FSLX section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteSpan {
    /// Offset from the referenced section's first byte.
    pub offset: u64,
    /// Number of referenced bytes.
    pub len: u64,
}

impl ByteSpan {
    /// Construct a section-relative byte span.
    #[must_use]
    pub const fn new(offset: u64, len: u64) -> Self {
        Self { offset, len }
    }

    pub(in crate::grimoire) fn end(
        self,
        section: &'static str,
    ) -> Result<u64, TermDictionaryError> {
        self.offset
            .checked_add(self.len)
            .ok_or(TermDictionaryError::ReferenceOverflow {
                section,
                offset: self.offset,
                len: self.len,
            })
    }
}

/// Durable metadata associated with one composite term key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermMetadata {
    /// Documents containing the term in this segment, including tombstoned docs.
    pub doc_freq: u32,
    /// Exact term stream inside POSTINGS.
    pub postings: ByteSpan,
    /// Exact term stream inside POSITIONS for position-indexed text fields.
    pub positions: Option<ByteSpan>,
    /// Exact term stream inside BLOCKMAX.
    pub blockmax: ByteSpan,
}

impl TermMetadata {
    /// Construct metadata for a field that does not store positions.
    #[must_use]
    pub const fn without_positions(doc_freq: u32, postings: ByteSpan, blockmax: ByteSpan) -> Self {
        Self {
            doc_freq,
            postings,
            positions: None,
            blockmax,
        }
    }

    /// Construct metadata for a field that stores positions.
    #[must_use]
    pub const fn with_positions(
        doc_freq: u32,
        postings: ByteSpan,
        positions: ByteSpan,
        blockmax: ByteSpan,
    ) -> Self {
        Self {
            doc_freq,
            postings,
            positions: Some(positions),
            blockmax,
        }
    }
}

/// Borrowed, sorted encoder input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermInput<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Opaque term bytes, with a maximum length of [`MAX_TERM_BYTES`].
    pub term: &'a [u8],
    /// Section references and document frequency.
    pub metadata: TermMetadata,
}

impl<'a> TermInput<'a> {
    /// Construct one encoder input row.
    #[must_use]
    pub const fn new(field_ord: u16, term: &'a [u8], metadata: TermMetadata) -> Self {
        Self {
            field_ord,
            term,
            metadata,
        }
    }
}

/// Section lengths required to validate every TERMDICT reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermSectionLengths {
    /// POSTINGS section length.
    pub postings: u64,
    /// POSITIONS section length; presence must match the schema.
    pub positions: Option<u64>,
    /// BLOCKMAX section length.
    pub blockmax: u64,
}

/// Explicit work and allocation budgets for opening a TERMDICT section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermDictionaryLimits {
    /// Maximum accepted section bytes.
    pub max_bytes: usize,
    /// Maximum retained block metadata entries.
    pub max_blocks: usize,
    /// Maximum validated terms.
    pub max_terms: usize,
    /// Maximum retained restart metadata entries.
    pub max_restarts: usize,
}

impl Default for TermDictionaryLimits {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_TERM_DICTIONARY_BYTES,
            max_blocks: DEFAULT_MAX_TERM_BLOCKS,
            max_terms: DEFAULT_MAX_TERMS,
            max_restarts: DEFAULT_MAX_TERM_RESTARTS,
        }
    }
}

/// Result of an exact term lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermMatch {
    /// Dense segment-local ordinal in composite-key order.
    pub term_ord: u32,
    /// Validated section references and document frequency.
    pub metadata: TermMetadata,
}

/// Borrowed view of the cursor's current reconstructed key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermRef<'a> {
    /// Dense segment-local ordinal in composite-key order.
    pub term_ord: u32,
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Opaque term bytes borrowed from cursor scratch storage.
    pub term: &'a [u8],
    /// Validated section references and document frequency.
    pub metadata: TermMetadata,
}

/// Owned row returned by explicitly bounded materialization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedTerm {
    /// Dense segment-local ordinal in composite-key order.
    pub term_ord: u32,
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Opaque term bytes.
    pub term: Vec<u8>,
    /// Validated section references and document frequency.
    pub metadata: TermMetadata,
}

impl TermRef<'_> {
    pub(in crate::grimoire) fn to_owned(self) -> Result<OwnedTerm, TermDictionaryError> {
        let mut term = Vec::new();
        term.try_reserve_exact(self.term.len())
            .map_err(|_| TermDictionaryError::Allocation {
                context: "owned term bytes",
                count: self.term.len(),
            })?;
        term.extend_from_slice(self.term);
        Ok(OwnedTerm {
            term_ord: self.term_ord,
            field_ord: self.field_ord,
            term,
            metadata: self.metadata,
        })
    }
}

/// Typed failures from encoding or validating an FSLX TERMDICT section.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TermDictionaryError {
    /// The supplied compile-time schema violates its own invariants.
    #[error("invalid schema for TERMDICT: {detail}")]
    InvalidSchema {
        /// Schema validation detail.
        detail: String,
    },
    /// The section exceeds a caller-selected byte budget.
    #[error("TERMDICT length is at least {actual} bytes, exceeding byte budget {limit}")]
    ByteBudgetExceeded {
        /// Configured byte cap.
        limit: usize,
        /// Supplied section length.
        actual: usize,
    },
    /// The declared block count exceeds a caller-selected cap.
    #[error("TERMDICT block count {actual} exceeds block budget {limit}")]
    BlockBudgetExceeded {
        /// Configured block cap.
        limit: usize,
        /// Declared block count.
        actual: usize,
    },
    /// The validated term count exceeds a caller-selected cap.
    #[error("TERMDICT term count {actual} exceeds term budget {limit}")]
    TermBudgetExceeded {
        /// Configured term cap.
        limit: usize,
        /// Declared or accumulated term count.
        actual: usize,
    },
    /// The restart count exceeds a caller-selected cap.
    #[error("TERMDICT restart count {actual} exceeds restart budget {limit}")]
    RestartBudgetExceeded {
        /// Configured restart cap.
        limit: usize,
        /// Declared or accumulated restart count.
        actual: usize,
    },
    /// A tiny byte slice declares more nonempty blocks than can fit.
    #[error("TERMDICT declares {block_count} blocks in only {byte_len} bytes")]
    ImplausibleBlockCount {
        /// Declared block count.
        block_count: usize,
        /// Total supplied bytes.
        byte_len: usize,
    },
    /// Durable bytes ended before a declared field did.
    #[error("truncated TERMDICT at offset {offset}: need {needed} bytes, only {remaining} remain")]
    Truncated {
        /// Absolute offset of the failed read.
        offset: usize,
        /// Required byte count.
        needed: usize,
        /// Available byte count inside the current bounded region.
        remaining: usize,
    },
    /// An unsigned LEB128 value used more bytes than its shortest form.
    #[error("non-canonical vint at TERMDICT offset {offset}")]
    NonCanonicalVint {
        /// Vint start offset.
        offset: usize,
    },
    /// An unsigned LEB128 value exceeds u64 or ten bytes.
    #[error("vint overflow at TERMDICT offset {offset}")]
    VintOverflow {
        /// Vint start offset.
        offset: usize,
    },
    /// A decoded scalar does not fit its semantic domain.
    #[error("TERMDICT {field} value {value} is out of range at offset {offset}")]
    ValueOutOfRange {
        /// Stable field name.
        field: &'static str,
        /// Rejected value.
        value: u64,
        /// Field or entry offset.
        offset: usize,
    },
    /// A term exceeds the language contract.
    #[error("term at ordinal {term_ordinal} has {length} bytes; maximum is {MAX_TERM_BYTES}")]
    TermTooLong {
        /// Dense term ordinal or encoder input index.
        term_ordinal: usize,
        /// Rejected term length.
        length: usize,
    },
    /// A composite key references no schema field.
    #[error("TERMDICT key references unknown field ordinal {field_ord}")]
    UnknownField {
        /// Rejected field ordinal.
        field_ord: u16,
    },
    /// Numeric and stored-only fields cannot appear in TERMDICT.
    #[error("schema field ordinal {field_ord} is not a term-indexed field")]
    NonTermField {
        /// Rejected field ordinal.
        field_ord: u16,
    },
    /// Encoder inputs are not strictly sorted and unique.
    #[error("encoder keys are not strictly ascending at input {index}")]
    NonAscendingInput {
        /// Rejected input index.
        index: usize,
    },
    /// Decoded composite keys are not globally strict.
    #[error("decoded TERMDICT keys are not strictly ascending at term ordinal {term_ordinal}")]
    NonAscendingKey {
        /// Rejected dense term ordinal.
        term_ordinal: usize,
    },
    /// A term entry has no posting-bearing documents.
    #[error("TERMDICT term ordinal {term_ordinal} has zero doc_freq")]
    ZeroDocFrequency {
        /// Rejected dense term ordinal.
        term_ordinal: usize,
    },
    /// A required per-term section range is empty.
    #[error("TERMDICT term ordinal {term_ordinal} has an empty {section} span")]
    EmptyReference {
        /// Referenced section.
        section: &'static str,
        /// Rejected dense term ordinal.
        term_ordinal: usize,
    },
    /// Adding a range offset and length overflowed u64.
    #[error("{section} range overflows: offset {offset}, length {len}")]
    ReferenceOverflow {
        /// Referenced section.
        section: &'static str,
        /// Section-relative offset.
        offset: u64,
        /// Range length.
        len: u64,
    },
    /// A per-term range exceeds its referenced section.
    #[error("{section} range ends at {end}, beyond section length {limit}")]
    ReferenceOutOfBounds {
        /// Referenced section.
        section: &'static str,
        /// Checked range end.
        end: u64,
        /// Declared section length.
        limit: u64,
    },
    /// Per-term section ranges are not contiguous in composite-key order.
    #[error("non-contiguous {section} range: expected offset {expected}, got {actual}")]
    NonContiguousReference {
        /// Referenced section.
        section: &'static str,
        /// Required next offset.
        expected: u64,
        /// Encoded next offset.
        actual: u64,
    },
    /// Final per-term ranges do not consume their declared section.
    #[error("{section} ranges consume {actual} bytes, expected {expected}")]
    SectionLengthMismatch {
        /// Referenced section.
        section: &'static str,
        /// Declared section length.
        expected: u64,
        /// Accumulated range end.
        actual: u64,
    },
    /// Entry payload width disagrees with its schema field.
    #[error("field {field_ord} positions presence mismatch: expected {expected}, got {actual}")]
    PositionsPresenceMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Schema-derived positions presence.
        expected: bool,
        /// Entry payload positions presence.
        actual: bool,
    },
    /// POSITIONS section presence disagrees with the schema as a whole.
    #[error("POSITIONS section presence mismatch: expected {expected}, got {actual}")]
    PositionsSectionMismatch {
        /// Whether the schema requires the section.
        expected: bool,
        /// Whether a section length was supplied.
        actual: bool,
    },
    /// A block index key is not strictly ascending.
    #[error("TERMDICT block index keys are not strictly ascending at block {block_index}")]
    NonAscendingIndexKey {
        /// Rejected block index.
        block_index: usize,
    },
    /// A relative block offset violates the contiguous block-area layout.
    #[error("invalid relative block offset {offset} for block {block_index}: {detail}")]
    InvalidBlockOffset {
        /// Rejected block index.
        block_index: usize,
        /// Encoded relative offset.
        offset: u64,
        /// Stable invariant detail.
        detail: &'static str,
    },
    /// Blocks must contain at least one entry.
    #[error("TERMDICT block {block_index} is empty")]
    EmptyBlock {
        /// Rejected block index.
        block_index: usize,
    },
    /// A declared entry count cannot fit in its block bytes.
    #[error("TERMDICT block {block_index} declares {entry_count} entries in {byte_len} bytes")]
    ImplausibleEntryCount {
        /// Rejected block index.
        block_index: usize,
        /// Declared entry count.
        entry_count: usize,
        /// Delimited block length.
        byte_len: usize,
    },
    /// Only an oversized singleton term may exceed the ordinary block target.
    #[error("TERMDICT block {block_index} has {entry_count} entries in {byte_len} bytes")]
    OversizedBlock {
        /// Rejected block index.
        block_index: usize,
        /// Declared entry count.
        entry_count: usize,
        /// Delimited block length.
        byte_len: usize,
    },
    /// A writer split a block even though the next entry still fit.
    #[error("TERMDICT block {block_index} was split before the 4096-byte target")]
    PrematureBlockSplit {
        /// Earlier block index.
        block_index: usize,
    },
    /// A compressed entry references bytes outside its previous key.
    #[error(
        "invalid shared prefix {shared} for previous key length {previous_len} at offset {offset}"
    )]
    InvalidPrefix {
        /// Encoded shared-prefix length.
        shared: usize,
        /// Previous key length.
        previous_len: usize,
        /// Entry start offset.
        offset: usize,
    },
    /// A compressed entry did not use the longest possible shared prefix.
    #[error("non-canonical shared prefix at TERMDICT offset {offset}")]
    NonCanonicalPrefix {
        /// Entry start offset.
        offset: usize,
    },
    /// The block index first key disagrees with its decoded block.
    #[error("TERMDICT block {block_index} first key disagrees with its index key")]
    IndexKeyMismatch {
        /// Rejected block index.
        block_index: usize,
    },
    /// Declared entries did not consume their exact delimited block.
    #[error("TERMDICT block {block_index} has {remaining} trailing bytes")]
    TrailingBlockBytes {
        /// Rejected block index.
        block_index: usize,
        /// Unconsumed bytes.
        remaining: usize,
    },
    /// A zero-block dictionary contains bytes after its count.
    #[error("empty TERMDICT has {remaining} trailing bytes")]
    TrailingEmptyBytes {
        /// Bytes after the zero block count.
        remaining: usize,
    },
    /// A requested term range has reversed endpoints.
    #[error("TERMDICT range start sorts after its end")]
    InvalidRange,
    /// Explicitly bounded result materialization exceeded its cap.
    #[error("term materialization limit {limit} exceeded")]
    MaterializationLimitExceeded {
        /// Caller-selected result cap.
        limit: usize,
    },
    /// A glob matched one more dictionary term than the caller allowed.
    #[error(
        "glob expansion for pattern {pattern} over field {field_ord} observed at least {actual} matching terms, exceeding limit {limit}"
    )]
    GlobExpansionLimitExceeded {
        /// The glob pattern whose expansion crossed the cap, lossily decoded
        /// for the diagnostic. Carried so a typed refusal names the pattern
        /// that exploded instead of only the field and the count.
        pattern: String,
        /// Field whose dictionary terms were scanned.
        field_ord: u16,
        /// Caller-selected expansion cap.
        limit: usize,
        /// Number of matches observed before the scan stopped, always `limit + 1`.
        actual: usize,
    },
    /// A fallible allocation could not reserve the required space.
    #[error("unable to reserve {count} items for {context}")]
    Allocation {
        /// Stable allocation context.
        context: &'static str,
        /// Requested item or byte count.
        count: usize,
    },
    /// Checked host-size arithmetic overflowed.
    #[error("TERMDICT size arithmetic overflow while computing {field}")]
    SizeOverflow {
        /// Stable size field.
        field: &'static str,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::grimoire) struct EncodedBlock {
    pub(in crate::grimoire) first_key: Vec<u8>,
    pub(in crate::grimoire) bytes: Vec<u8>,
    pub(in crate::grimoire) entry_count: usize,
}

#[derive(Clone, Debug)]
pub(in crate::grimoire) struct BlockMeta {
    pub(in crate::grimoire) first_key_range: Range<usize>,
    pub(in crate::grimoire) byte_range: Range<usize>,
    pub(in crate::grimoire) entry_count: usize,
    pub(in crate::grimoire) base_term_ord: u32,
    pub(in crate::grimoire) restart_range: Range<usize>,
}

#[derive(Clone, Debug)]
pub(in crate::grimoire) struct RestartMeta {
    pub(in crate::grimoire) entry_ordinal: usize,
    pub(in crate::grimoire) term_ordinal: u32,
    pub(in crate::grimoire) entry_offset: usize,
    pub(in crate::grimoire) key_range: Range<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::grimoire) struct BlockTail {
    pub(in crate::grimoire) byte_len: usize,
    pub(in crate::grimoire) entry_count: usize,
}

#[derive(Clone, Debug)]
pub(in crate::grimoire) struct IndexRecord {
    pub(in crate::grimoire) first_key_range: Range<usize>,
    pub(in crate::grimoire) relative_offset: u64,
}

/// Owned, allocation-bounded metadata produced by one complete TERMDICT
/// validation.
///
/// This stays crate-private so callers cannot pair validated offsets with an
/// unrelated byte source. [`TermDictionary::from_validated_metadata`] also
/// checks the exact live slice address and length before constructing a cheap
/// borrowed view.
#[derive(Debug)]
pub struct ValidatedTermDictionaryMetadata {
    pub(in crate::grimoire) source_start: usize,
    pub(in crate::grimoire) source_len: usize,
    pub(in crate::grimoire) schema: SchemaDescriptor,
    pub(in crate::grimoire) sections: TermSectionLengths,
    pub(in crate::grimoire) blocks: Vec<BlockMeta>,
    pub(in crate::grimoire) restarts: Vec<RestartMeta>,
    pub(in crate::grimoire) term_count: u32,
}

impl ValidatedTermDictionaryMetadata {
    pub const fn schema(&self) -> SchemaDescriptor {
        self.schema
    }

    /// Estimated payload bytes retained by this validated metadata object.
    ///
    /// Counts the struct itself plus the full capacity of the block and
    /// restart directories ([`BlockMeta`] and [`RestartMeta`] are flat offset
    /// records without nested heap allocations). This is a payload estimate,
    /// not an exact memory claim: it excludes `Arc` control-block overhead,
    /// allocation alignment, and allocator slack. It reports the persistent
    /// per-segment cost of validating TERMDICT metadata once per immutable
    /// backing instead of per query.
    pub fn payload_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            .saturating_add(
                self.blocks
                    .capacity()
                    .saturating_mul(core::mem::size_of::<BlockMeta>()),
            )
            .saturating_add(
                self.restarts
                    .capacity()
                    .saturating_mul(core::mem::size_of::<RestartMeta>()),
            )
    }
}
