use std::cmp::Ordering;
use std::ops::{Bound, Range};

use crate::schema::{FieldKind, SchemaDescriptor};

use super::encoder::encoded_entry_len;
use super::types::{
    BlockMeta, BlockTail, ByteSpan, RestartMeta, TermDictionaryError, TermDictionaryLimits,
    TermMetadata, TermSectionLengths,
};
use super::{
    BLOCK_ENTRY_COUNT_BYTES, MAX_COMPOSITE_KEY_BYTES, MAX_TERM_BYTES, MIN_WIRE_BYTES_PER_ENTRY,
    TERM_BLOCK_TARGET_BYTES, TERM_RESTART_INTERVAL,
};

#[derive(Clone, Copy, Debug)]
pub(in crate::grimoire) struct DecodedEntry {
    pub(in crate::grimoire) field_ord: u16,
    pub(in crate::grimoire) metadata: TermMetadata,
    full_key_range: Option<RangeMarker>,
}

#[derive(Clone, Copy, Debug)]
struct RangeMarker {
    start: usize,
    end: usize,
}

impl RangeMarker {
    const fn into_range(self) -> Range<usize> {
        self.start..self.end
    }
}

pub(in crate::grimoire) fn validate_block(
    bytes: &[u8],
    schema: SchemaDescriptor,
    sections: TermSectionLengths,
    limits: TermDictionaryLimits,
    block_index: usize,
    byte_range: Range<usize>,
    expected_first_key: Range<usize>,
    base_term_ord: u32,
    term_count: &mut usize,
    restarts: &mut Vec<RestartMeta>,
    references: &mut ReferenceValidator,
    block_key: &mut Vec<u8>,
    previous_key: &mut Vec<u8>,
    previous_tail: Option<&BlockTail>,
) -> Result<(BlockMeta, BlockTail), TermDictionaryError> {
    let block_len = byte_range.end.saturating_sub(byte_range.start);
    let mut reader = SliceReader::with_range(bytes, byte_range.start, byte_range.end)?;
    let entry_count = usize::from(reader.read_u16()?);
    if entry_count == 0 {
        return Err(TermDictionaryError::EmptyBlock { block_index });
    }
    if entry_count
        > block_len
            .saturating_sub(BLOCK_ENTRY_COUNT_BYTES)
            .saturating_div(MIN_WIRE_BYTES_PER_ENTRY)
    {
        return Err(TermDictionaryError::ImplausibleEntryCount {
            block_index,
            entry_count,
            byte_len: block_len,
        });
    }
    if entry_count > 1 && block_len > TERM_BLOCK_TARGET_BYTES {
        return Err(TermDictionaryError::OversizedBlock {
            block_index,
            entry_count,
            byte_len: block_len,
        });
    }
    let next_term_count =
        term_count
            .checked_add(entry_count)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "validated term count",
            })?;
    if next_term_count > limits.max_terms {
        return Err(TermDictionaryError::TermBudgetExceeded {
            limit: limits.max_terms,
            actual: next_term_count,
        });
    }
    let block_restarts = entry_count.div_ceil(TERM_RESTART_INTERVAL);
    let next_restart_count =
        restarts
            .len()
            .checked_add(block_restarts)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "restart count",
            })?;
    if next_restart_count > limits.max_restarts {
        return Err(TermDictionaryError::RestartBudgetExceeded {
            limit: limits.max_restarts,
            actual: next_restart_count,
        });
    }
    restarts
        .try_reserve(block_restarts)
        .map_err(|_| TermDictionaryError::Allocation {
            context: "restart metadata",
            count: next_restart_count,
        })?;

    let restart_start = restarts.len();
    block_key.clear();
    for entry_ordinal in 0..entry_count {
        let entry_offset = reader.position();
        let term_ordinal = usize::try_from(base_term_ord)
            .unwrap_or(usize::MAX)
            .saturating_add(entry_ordinal);
        let decoded = decode_entry(
            &mut reader,
            entry_ordinal,
            block_key,
            schema,
            sections,
            term_ordinal,
        )?;
        if entry_ordinal == 0 {
            if block_key.as_slice() != &bytes[expected_first_key.clone()] {
                return Err(TermDictionaryError::IndexKeyMismatch { block_index });
            }
            if let Some(tail) = previous_tail {
                let appended = encoded_entry_len(
                    block_key,
                    previous_key,
                    tail.entry_count,
                    decoded.metadata,
                    field_has_positions(schema, decoded.field_ord)?,
                )?;
                if tail.byte_len <= TERM_BLOCK_TARGET_BYTES
                    && tail.byte_len.checked_add(appended).ok_or(
                        TermDictionaryError::SizeOverflow {
                            field: "canonical block split",
                        },
                    )? <= TERM_BLOCK_TARGET_BYTES
                {
                    return Err(TermDictionaryError::PrematureBlockSplit {
                        block_index: block_index.saturating_sub(1),
                    });
                }
            }
        }
        if !previous_key.is_empty() && block_key.as_slice() <= previous_key.as_slice() {
            return Err(TermDictionaryError::NonAscendingKey { term_ordinal });
        }
        if let Some(marker) = decoded.full_key_range {
            restarts.push(RestartMeta {
                entry_ordinal,
                term_ordinal: u32::try_from(term_ordinal).map_err(|_| {
                    TermDictionaryError::ValueOutOfRange {
                        field: "term ordinal",
                        value: u64::try_from(term_ordinal).unwrap_or(u64::MAX),
                        offset: entry_offset,
                    }
                })?,
                entry_offset,
                key_range: marker.into_range(),
            });
        }
        references.push(decoded.metadata)?;
        copy_bytes(previous_key, block_key, "validated previous key")?;
    }
    let remaining = byte_range.end.saturating_sub(reader.position());
    if remaining != 0 {
        return Err(TermDictionaryError::TrailingBlockBytes {
            block_index,
            remaining,
        });
    }
    *term_count = next_term_count;
    Ok((
        BlockMeta {
            first_key_range: expected_first_key,
            byte_range,
            entry_count,
            base_term_ord,
            restart_range: restart_start..restarts.len(),
        },
        BlockTail {
            byte_len: block_len,
            entry_count,
        },
    ))
}

pub(in crate::grimoire) fn decode_entry(
    reader: &mut SliceReader<'_>,
    entry_ordinal: usize,
    key: &mut Vec<u8>,
    schema: SchemaDescriptor,
    sections: TermSectionLengths,
    term_ordinal: usize,
) -> Result<DecodedEntry, TermDictionaryError> {
    let entry_offset = reader.position();
    let full_key_range = if entry_ordinal.is_multiple_of(TERM_RESTART_INTERVAL) {
        let key_len_offset = reader.position();
        let key_len = usize_from_vint(reader.read_vint()?, "key_len", key_len_offset)?;
        if !(2..=MAX_COMPOSITE_KEY_BYTES).contains(&key_len) {
            return Err(TermDictionaryError::TermTooLong {
                term_ordinal,
                length: key_len.saturating_sub(2),
            });
        }
        let range = reader.take_range(key_len)?;
        copy_bytes(key, &reader.bytes[range.clone()], "restart key")?;
        Some(RangeMarker {
            start: range.start,
            end: range.end,
        })
    } else {
        let shared_offset = reader.position();
        let shared = usize_from_vint(reader.read_vint()?, "shared_prefix_len", shared_offset)?;
        let suffix_offset = reader.position();
        let suffix_len = usize_from_vint(reader.read_vint()?, "suffix_len", suffix_offset)?;
        if shared > key.len() {
            return Err(TermDictionaryError::InvalidPrefix {
                shared,
                previous_len: key.len(),
                offset: entry_offset,
            });
        }
        let suffix = reader.take(suffix_len)?;
        if shared < key.len() && suffix.first().is_some_and(|byte| *byte == key[shared]) {
            return Err(TermDictionaryError::NonCanonicalPrefix {
                offset: entry_offset,
            });
        }
        let new_len = shared
            .checked_add(suffix_len)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "reconstructed key length",
            })?;
        if !(2..=MAX_COMPOSITE_KEY_BYTES).contains(&new_len) {
            return Err(TermDictionaryError::TermTooLong {
                term_ordinal,
                length: new_len.saturating_sub(2),
            });
        }
        if new_len > key.capacity() {
            key.try_reserve(new_len.saturating_sub(key.len()))
                .map_err(|_| TermDictionaryError::Allocation {
                    context: "reconstructed key",
                    count: new_len,
                })?;
        }
        key.truncate(shared);
        key.extend_from_slice(suffix);
        None
    };

    let has_positions = validate_composite_key(schema, key, term_ordinal)?;
    let field_ord = u16::from_be_bytes([key[0], key[1]]);
    let doc_freq_offset = reader.position();
    let doc_freq_raw = reader.read_vint()?;
    let doc_freq =
        u32::try_from(doc_freq_raw).map_err(|_| TermDictionaryError::ValueOutOfRange {
            field: "doc_freq",
            value: doc_freq_raw,
            offset: doc_freq_offset,
        })?;
    let postings = ByteSpan::new(reader.read_vint()?, reader.read_vint()?);
    let positions = if has_positions {
        Some(ByteSpan::new(reader.read_vint()?, reader.read_vint()?))
    } else {
        None
    };
    let blockmax = ByteSpan::new(reader.read_vint()?, reader.read_vint()?);
    let metadata = TermMetadata {
        doc_freq,
        postings,
        positions,
        blockmax,
    };
    validate_metadata_basic(field_ord, has_positions, metadata, sections, term_ordinal)?;
    Ok(DecodedEntry {
        field_ord,
        metadata,
        full_key_range,
    })
}

#[derive(Debug)]
pub(in crate::grimoire) struct ReferenceValidator {
    sections: TermSectionLengths,
    postings_end: u64,
    positions_end: u64,
    blockmax_end: u64,
}

impl ReferenceValidator {
    pub(in crate::grimoire) fn new(
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
    ) -> Result<Self, TermDictionaryError> {
        validate_positions_section(schema, sections)?;
        Ok(Self {
            sections,
            postings_end: 0,
            positions_end: 0,
            blockmax_end: 0,
        })
    }

    pub(in crate::grimoire) fn push(
        &mut self,
        metadata: TermMetadata,
    ) -> Result<(), TermDictionaryError> {
        if metadata.postings.offset != self.postings_end {
            return Err(TermDictionaryError::NonContiguousReference {
                section: "POSTINGS",
                expected: self.postings_end,
                actual: metadata.postings.offset,
            });
        }
        self.postings_end = metadata.postings.end("POSTINGS")?;
        if let Some(positions) = metadata.positions {
            if positions.offset != self.positions_end {
                return Err(TermDictionaryError::NonContiguousReference {
                    section: "POSITIONS",
                    expected: self.positions_end,
                    actual: positions.offset,
                });
            }
            self.positions_end = positions.end("POSITIONS")?;
        }
        if metadata.blockmax.offset != self.blockmax_end {
            return Err(TermDictionaryError::NonContiguousReference {
                section: "BLOCKMAX",
                expected: self.blockmax_end,
                actual: metadata.blockmax.offset,
            });
        }
        self.blockmax_end = metadata.blockmax.end("BLOCKMAX")?;
        Ok(())
    }

    pub(in crate::grimoire) fn finish(self) -> Result<(), TermDictionaryError> {
        validate_final_length("POSTINGS", self.sections.postings, self.postings_end)?;
        validate_final_length("BLOCKMAX", self.sections.blockmax, self.blockmax_end)?;
        if let Some(expected) = self.sections.positions {
            validate_final_length("POSITIONS", expected, self.positions_end)?;
        }
        Ok(())
    }
}

fn validate_final_length(
    section: &'static str,
    expected: u64,
    actual: u64,
) -> Result<(), TermDictionaryError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TermDictionaryError::SectionLengthMismatch {
            section,
            expected,
            actual,
        })
    }
}

pub(in crate::grimoire) fn validate_metadata_basic(
    field_ord: u16,
    has_positions: bool,
    metadata: TermMetadata,
    sections: TermSectionLengths,
    term_ordinal: usize,
) -> Result<(), TermDictionaryError> {
    if metadata.doc_freq == 0 {
        return Err(TermDictionaryError::ZeroDocFrequency { term_ordinal });
    }
    validate_span(
        "POSTINGS",
        metadata.postings,
        sections.postings,
        term_ordinal,
    )?;
    if metadata.positions.is_some() != has_positions {
        return Err(TermDictionaryError::PositionsPresenceMismatch {
            field_ord,
            expected: has_positions,
            actual: metadata.positions.is_some(),
        });
    }
    if let Some(positions) = metadata.positions {
        let limit = sections
            .positions
            .ok_or(TermDictionaryError::PositionsSectionMismatch {
                expected: true,
                actual: false,
            })?;
        validate_span("POSITIONS", positions, limit, term_ordinal)?;
    }
    validate_span(
        "BLOCKMAX",
        metadata.blockmax,
        sections.blockmax,
        term_ordinal,
    )
}

fn validate_span(
    section: &'static str,
    span: ByteSpan,
    limit: u64,
    term_ordinal: usize,
) -> Result<(), TermDictionaryError> {
    if span.len == 0 {
        return Err(TermDictionaryError::EmptyReference {
            section,
            term_ordinal,
        });
    }
    let end = span.end(section)?;
    if end > limit {
        return Err(TermDictionaryError::ReferenceOutOfBounds {
            section,
            end,
            limit,
        });
    }
    Ok(())
}

pub(in crate::grimoire) fn validate_schema(
    schema: SchemaDescriptor,
) -> Result<(), TermDictionaryError> {
    schema
        .validate()
        .map_err(|error| TermDictionaryError::InvalidSchema {
            detail: error.to_string(),
        })
}

pub(in crate::grimoire) fn validate_positions_section(
    schema: SchemaDescriptor,
    sections: TermSectionLengths,
) -> Result<(), TermDictionaryError> {
    let expected = schema.fields.iter().any(|field| {
        matches!(
            field.kind,
            FieldKind::Text {
                positions: true,
                ..
            }
        )
    });
    let actual = sections.positions.is_some();
    if expected == actual {
        Ok(())
    } else {
        Err(TermDictionaryError::PositionsSectionMismatch { expected, actual })
    }
}

pub(in crate::grimoire) fn field_has_positions(
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<bool, TermDictionaryError> {
    let field = schema
        .fields
        .get(usize::from(field_ord))
        .ok_or(TermDictionaryError::UnknownField { field_ord })?;
    if field.id != field_ord {
        return Err(TermDictionaryError::UnknownField { field_ord });
    }
    match field.kind {
        FieldKind::Keyword => Ok(false),
        FieldKind::Text { positions, .. } => Ok(positions),
        FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
            Err(TermDictionaryError::NonTermField { field_ord })
        }
    }
}

pub(in crate::grimoire) fn validate_composite_key(
    schema: SchemaDescriptor,
    key: &[u8],
    term_ordinal: usize,
) -> Result<bool, TermDictionaryError> {
    if key.len() < 2 {
        return Err(TermDictionaryError::Truncated {
            offset: 0,
            needed: 2,
            remaining: key.len(),
        });
    }
    let term_len = key.len().saturating_sub(2);
    if term_len > MAX_TERM_BYTES {
        return Err(TermDictionaryError::TermTooLong {
            term_ordinal,
            length: term_len,
        });
    }
    field_has_positions(schema, u16::from_be_bytes([key[0], key[1]]))
}

pub fn validate_query_term(
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
) -> Result<(), TermDictionaryError> {
    field_has_positions(schema, field_ord)?;
    if term.len() > MAX_TERM_BYTES {
        return Err(TermDictionaryError::TermTooLong {
            term_ordinal: 0,
            length: term.len(),
        });
    }
    Ok(())
}

pub fn validate_bound_term(
    schema: SchemaDescriptor,
    field_ord: u16,
    bound: &Bound<&[u8]>,
) -> Result<(), TermDictionaryError> {
    match bound {
        Bound::Included(term) | Bound::Excluded(term) => {
            validate_query_term(schema, field_ord, term)
        }
        Bound::Unbounded => Ok(()),
    }
}

pub(in crate::grimoire) fn range_is_reversed(start: &Bound<&[u8]>, end: &Bound<&[u8]>) -> bool {
    match (start, end) {
        (
            Bound::Included(start) | Bound::Excluded(start),
            Bound::Included(end) | Bound::Excluded(end),
        ) => start > end,
        _ => false,
    }
}

pub(in crate::grimoire) fn composite_key(
    field_ord: u16,
    term: &[u8],
    term_ordinal: usize,
) -> Result<Vec<u8>, TermDictionaryError> {
    if term.len() > MAX_TERM_BYTES {
        return Err(TermDictionaryError::TermTooLong {
            term_ordinal,
            length: term.len(),
        });
    }
    let mut key = Vec::new();
    build_composite_in(&mut key, field_ord, term, "composite key")?;
    Ok(key)
}

pub(in crate::grimoire) fn build_composite_in(
    output: &mut Vec<u8>,
    field_ord: u16,
    term: &[u8],
    context: &'static str,
) -> Result<(), TermDictionaryError> {
    let length = term
        .len()
        .checked_add(2)
        .ok_or(TermDictionaryError::SizeOverflow {
            field: "composite key length",
        })?;
    output.clear();
    output
        .try_reserve(length)
        .map_err(|_| TermDictionaryError::Allocation {
            context,
            count: length,
        })?;
    output.extend_from_slice(&field_ord.to_be_bytes());
    output.extend_from_slice(term);
    Ok(())
}

pub(in crate::grimoire) fn composite_cmp(
    field_ord: u16,
    term: &[u8],
    composite: &[u8],
) -> Ordering {
    let field = field_ord.to_be_bytes();
    match field.as_slice().cmp(composite.get(..2).unwrap_or_default()) {
        Ordering::Equal => term.cmp(composite.get(2..).unwrap_or_default()),
        ordering => ordering,
    }
}

pub(in crate::grimoire) fn append_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
    context: &'static str,
) -> Result<(), TermDictionaryError> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| TermDictionaryError::Allocation {
            context,
            count: bytes.len(),
        })?;
    output.extend_from_slice(bytes);
    Ok(())
}

pub(in crate::grimoire) fn clone_bytes(
    bytes: &[u8],
    context: &'static str,
) -> Result<Vec<u8>, TermDictionaryError> {
    let mut output = Vec::new();
    copy_bytes(&mut output, bytes, context)?;
    Ok(output)
}

pub(in crate::grimoire) fn copy_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
    context: &'static str,
) -> Result<(), TermDictionaryError> {
    output.clear();
    output
        .try_reserve(bytes.len())
        .map_err(|_| TermDictionaryError::Allocation {
            context,
            count: bytes.len(),
        })?;
    output.extend_from_slice(bytes);
    Ok(())
}

pub(in crate::grimoire) fn checked_add_size(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, TermDictionaryError> {
    left.checked_add(right)
        .ok_or(TermDictionaryError::SizeOverflow { field })
}

pub(in crate::grimoire) fn usize_from_vint(
    value: u64,
    field: &'static str,
    offset: usize,
) -> Result<usize, TermDictionaryError> {
    usize::try_from(value).map_err(|_| TermDictionaryError::ValueOutOfRange {
        field,
        value,
        offset,
    })
}

pub(in crate::grimoire) fn vint_len(mut value: u64) -> usize {
    let mut length = 1;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

#[allow(clippy::cast_possible_truncation)]
pub(in crate::grimoire) fn write_vint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

pub(in crate::grimoire) struct SliceReader<'a> {
    bytes: &'a [u8],
    position: usize,
    end: usize,
}

impl<'a> SliceReader<'a> {
    pub(in crate::grimoire) const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            end: bytes.len(),
        }
    }

    pub(in crate::grimoire) fn with_range(
        bytes: &'a [u8],
        start: usize,
        end: usize,
    ) -> Result<Self, TermDictionaryError> {
        if start > end || end > bytes.len() {
            return Err(TermDictionaryError::Truncated {
                offset: start.min(bytes.len()),
                needed: end.saturating_sub(start),
                remaining: bytes.len().saturating_sub(start.min(bytes.len())),
            });
        }
        Ok(Self {
            bytes,
            position: start,
            end,
        })
    }

    pub(in crate::grimoire) const fn position(&self) -> usize {
        self.position
    }

    pub(in crate::grimoire) fn take(
        &mut self,
        length: usize,
    ) -> Result<&'a [u8], TermDictionaryError> {
        let range = self.take_range(length)?;
        Ok(&self.bytes[range])
    }

    pub(in crate::grimoire) fn take_range(
        &mut self,
        length: usize,
    ) -> Result<Range<usize>, TermDictionaryError> {
        let remaining = self.end.saturating_sub(self.position);
        let end = self
            .position
            .checked_add(length)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "reader cursor",
            })?;
        if end > self.end {
            return Err(TermDictionaryError::Truncated {
                offset: self.position,
                needed: length,
                remaining,
            });
        }
        let range = self.position..end;
        self.position = end;
        Ok(range)
    }

    fn read_u8(&mut self) -> Result<u8, TermDictionaryError> {
        let offset = self.position;
        self.take(1)?
            .first()
            .copied()
            .ok_or(TermDictionaryError::Truncated {
                offset,
                needed: 1,
                remaining: 0,
            })
    }

    fn read_u16(&mut self) -> Result<u16, TermDictionaryError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(in crate::grimoire) fn read_u32(&mut self) -> Result<u32, TermDictionaryError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    // Profile-directed (bd-b6tc): `read_vint` was 16% of query self-time and
    // its hot instructions were the *function prologue* — it was never inlined,
    // so every call in the tight `decode_entry` loops paid a full 6-register
    // spill. The 1-byte case (value < 128) dominates gap-encoded
    // lengths/offsets and is always canonical, so it is inlined into callers
    // here while the rare multi-byte loop is kept out-of-line and `#[cold]` to
    // avoid code-size bloat. Byte-identical to the former single function.
    #[inline]
    pub(in crate::grimoire) fn read_vint(&mut self) -> Result<u64, TermDictionaryError> {
        if self.position < self.end
            && let Some(&byte) = self.bytes.get(self.position)
            && byte & 0x80 == 0
        {
            self.position += 1;
            return Ok(u64::from(byte));
        }
        self.read_vint_multibyte()
    }

    #[cold]
    #[inline(never)]
    fn read_vint_multibyte(&mut self) -> Result<u64, TermDictionaryError> {
        let start = self.position;
        let mut value = 0_u64;
        for byte_index in 0..10 {
            let byte = self.read_u8()?;
            if byte_index == 9 && byte & 0xfe != 0 {
                return Err(TermDictionaryError::VintOverflow { offset: start });
            }
            value |= u64::from(byte & 0x7f) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint_len(value) != byte_index + 1 {
                    return Err(TermDictionaryError::NonCanonicalVint { offset: start });
                }
                return Ok(value);
            }
        }
        Err(TermDictionaryError::VintOverflow { offset: start })
    }
}
