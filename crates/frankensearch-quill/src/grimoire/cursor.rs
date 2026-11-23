use std::cmp::Ordering;

use super::codec::{SliceReader, composite_cmp, decode_entry};
use super::types::{OwnedTerm, TermDictionaryError, TermMatch, TermRef};
use super::{BLOCK_ENTRY_COUNT_BYTES, TermDictionary};

pub(in crate::grimoire) fn collect_glob_matches<F>(
    mut cursor: TermCursor<'_, '_>,
    field_ord: u16,
    pattern: &[u8],
    limit: usize,
    mut matches: F,
) -> Result<Vec<OwnedTerm>, TermDictionaryError>
where
    F: FnMut(&[u8]) -> bool,
{
    let mut output = Vec::new();
    while let Some(current) = cursor.current() {
        if matches(current.term) {
            if output.len() >= limit {
                return Err(glob_limit_error(
                    field_ord,
                    pattern,
                    limit,
                    output.len().saturating_add(1),
                ));
            }
            output
                .try_reserve(1)
                .map_err(|_| TermDictionaryError::Allocation {
                    context: "glob expansion rows",
                    count: output.len().saturating_add(1),
                })?;
            output.push(current.to_owned()?);
        }
        cursor.next()?;
    }
    Ok(output)
}

pub(in crate::grimoire) fn glob_limit_error(
    field_ord: u16,
    pattern: &[u8],
    limit: usize,
    actual: usize,
) -> TermDictionaryError {
    TermDictionaryError::GlobExpansionLimitExceeded {
        pattern: String::from_utf8_lossy(pattern).into_owned(),
        field_ord,
        limit,
        actual,
    }
}

pub fn trailing_star_prefix(pattern: &[u8]) -> Option<&[u8]> {
    let first_star = pattern.iter().position(|byte| *byte == b'*')?;
    pattern[first_star..]
        .iter()
        .all(|byte| *byte == b'*')
        .then_some(&pattern[..first_star])
}

pub(in crate::grimoire) fn longest_glob_literal(pattern: &[u8]) -> &[u8] {
    pattern
        .split(|byte| *byte == b'*')
        .max_by_key(|literal| literal.len())
        .unwrap_or_default()
}

pub(in crate::grimoire) fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }

    // Reject eight impossible starts per word before confirming the complete
    // literal. This is the safe SWAR analogue of memmem's first-byte filter;
    // the final `starts_with` remains the semantic authority.
    const BYTE_ONES: u64 = 0x0101_0101_0101_0101;
    const BYTE_HIGHS: u64 = 0x8080_8080_8080_8080;

    let final_start = haystack.len() - needle.len();
    let candidates = &haystack[..=final_start];
    let repeated_first = u64::from(needle[0]).wrapping_mul(BYTE_ONES);
    let (chunks, remainder) = candidates.as_chunks::<8>();
    let mut base = 0_usize;
    for chunk in chunks {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(chunk);
        let differences = u64::from_ne_bytes(bytes) ^ repeated_first;
        let has_first = differences.wrapping_sub(BYTE_ONES) & !differences & BYTE_HIGHS != 0;
        if has_first
            && chunk.iter().enumerate().any(|(offset, &candidate)| {
                candidate == needle[0] && haystack[base + offset..].starts_with(needle)
            })
        {
            return true;
        }
        base += chunk.len();
    }
    remainder.iter().enumerate().any(|(offset, &candidate)| {
        candidate == needle[0] && haystack[base + offset..].starts_with(needle)
    })
}

pub fn star_glob_matches(pattern: &[u8], term: &[u8]) -> bool {
    let mut pattern_pos = 0;
    let mut term_pos = 0;
    let mut resume_after_star = None;
    let mut star_match_end = 0;

    while term_pos < term.len() {
        match pattern.get(pattern_pos) {
            Some(pattern_byte) if *pattern_byte == term[term_pos] && *pattern_byte != b'*' => {
                pattern_pos = pattern_pos.saturating_add(1);
                term_pos = term_pos.saturating_add(1);
            }
            Some(&b'*') => {
                while pattern.get(pattern_pos) == Some(&b'*') {
                    pattern_pos = pattern_pos.saturating_add(1);
                }
                resume_after_star = Some(pattern_pos);
                star_match_end = term_pos;
            }
            _ => {
                let Some(resume_pos) = resume_after_star else {
                    return false;
                };
                if star_match_end >= term.len() {
                    return false;
                }
                star_match_end = star_match_end.saturating_add(1);
                term_pos = star_match_end;
                pattern_pos = resume_pos;
            }
        }
    }

    pattern[pattern_pos..].iter().all(|byte| *byte == b'*')
}

/// Reusable exact-lookup buffers.
#[derive(Debug, Default)]
pub struct TermScratch {
    pub(in crate::grimoire) target: Vec<u8>,
    pub(in crate::grimoire) decoded: Vec<u8>,
}

impl TermScratch {
    /// Construct empty reusable lookup scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target: Vec::new(),
            decoded: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::grimoire) struct CursorBound {
    key: Vec<u8>,
    inclusive: bool,
}

impl CursorBound {
    pub(in crate::grimoire) const fn included(key: Vec<u8>) -> Self {
        Self {
            key,
            inclusive: true,
        }
    }

    pub(in crate::grimoire) const fn excluded(key: Vec<u8>) -> Self {
        Self {
            key,
            inclusive: false,
        }
    }
}

/// Allocation-reusing ordered TERMDICT cursor.
#[derive(Debug)]
pub struct TermCursor<'dict, 'bytes> {
    dictionary: &'dict TermDictionary<'bytes>,
    block_index: usize,
    next_entry_ordinal: usize,
    next_offset: usize,
    key: Vec<u8>,
    current: Option<TermMatch>,
    upper: Option<CursorBound>,
    field_filter: Option<u16>,
    prefix_filter: Option<Vec<u8>>,
    finished: bool,
}

impl<'dict, 'bytes> TermCursor<'dict, 'bytes> {
    pub(in crate::grimoire) fn new(
        dictionary: &'dict TermDictionary<'bytes>,
        lower: Option<CursorBound>,
        upper: Option<CursorBound>,
        field_filter: Option<u16>,
        prefix_filter: Option<Vec<u8>>,
    ) -> Result<Self, TermDictionaryError> {
        let mut cursor = Self {
            dictionary,
            block_index: 0,
            next_entry_ordinal: 0,
            next_offset: 0,
            key: Vec::new(),
            current: None,
            upper,
            field_filter,
            prefix_filter,
            finished: dictionary.blocks.is_empty(),
        };
        if cursor.finished {
            return Ok(cursor);
        }

        if let Some(lower) = lower {
            let block_index = dictionary.block_for_key(&lower.key).unwrap_or(0);
            let block = &dictionary.blocks[block_index];
            let restart_slice = &dictionary.restarts[block.restart_range.clone()];
            let restart_index = restart_slice
                .partition_point(|restart| {
                    dictionary.bytes[restart.key_range.clone()] <= *lower.key.as_slice()
                })
                .saturating_sub(1);
            let restart = &restart_slice[restart_index];
            cursor.block_index = block_index;
            cursor.next_entry_ordinal = restart.entry_ordinal;
            cursor.next_offset = restart.entry_offset;
            cursor.advance_raw()?;
            while let Some(current) = cursor.current() {
                let ordering = composite_cmp(current.field_ord, current.term, &lower.key);
                if ordering == Ordering::Greater || (ordering == Ordering::Equal && lower.inclusive)
                {
                    break;
                }
                cursor.advance_raw()?;
            }
        } else {
            cursor.start_block(0)?;
            cursor.advance_raw()?;
        }
        cursor.enforce_filters()?;
        Ok(cursor)
    }

    /// Current term, or `None` after the cursor is fused.
    #[must_use]
    pub fn current(&self) -> Option<TermRef<'_>> {
        let current = self.current?;
        let key = self.key.as_slice();
        let field_bytes = key.get(..2)?;
        let field_ord = u16::from_be_bytes([field_bytes[0], field_bytes[1]]);
        Some(TermRef {
            term_ord: current.term_ord,
            field_ord,
            term: key.get(2..).unwrap_or_default(),
            metadata: current.metadata,
        })
    }

    /// Zero-based prefix-compressed block containing the current term.
    ///
    /// Returns `None` after exhaustion. Query execution uses this only for
    /// coarse cancellation and deterministic fuel checkpoints.
    #[must_use]
    pub fn current_block_index(&self) -> Option<usize> {
        self.current.map(|_| self.block_index)
    }

    /// Advance once in composite-key order.
    ///
    /// # Errors
    ///
    /// Returns a typed error if validated bytes cannot be decoded or scratch
    /// growth fails.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<TermRef<'_>>, TermDictionaryError> {
        if self.current.is_none() {
            return Ok(None);
        }
        self.advance_raw()?;
        self.enforce_filters()?;
        Ok(self.current())
    }

    /// Collect the remaining range under an explicit result cap.
    ///
    /// # Errors
    ///
    /// Returns a typed error on limit exhaustion, allocation failure, or
    /// validated-decode failure.
    pub fn collect_bounded(mut self, limit: usize) -> Result<Vec<OwnedTerm>, TermDictionaryError> {
        let mut output = Vec::new();
        while let Some(current) = self.current() {
            if output.len() >= limit {
                return Err(TermDictionaryError::MaterializationLimitExceeded { limit });
            }
            output
                .try_reserve(1)
                .map_err(|_| TermDictionaryError::Allocation {
                    context: "materialized term rows",
                    count: output.len().saturating_add(1),
                })?;
            output.push(current.to_owned()?);
            self.next()?;
        }
        Ok(output)
    }

    fn start_block(&mut self, block_index: usize) -> Result<(), TermDictionaryError> {
        let Some(block) = self.dictionary.blocks.get(block_index) else {
            self.current = None;
            self.finished = true;
            return Ok(());
        };
        self.block_index = block_index;
        self.next_entry_ordinal = 0;
        self.next_offset = block
            .byte_range
            .start
            .checked_add(BLOCK_ENTRY_COUNT_BYTES)
            .ok_or(TermDictionaryError::SizeOverflow {
                field: "block entry start",
            })?;
        self.key.clear();
        Ok(())
    }

    fn advance_raw(&mut self) -> Result<(), TermDictionaryError> {
        if self.finished {
            self.current = None;
            return Ok(());
        }
        loop {
            let block = &self.dictionary.blocks[self.block_index];
            if self.next_entry_ordinal >= block.entry_count {
                self.start_block(self.block_index.saturating_add(1))?;
                if self.finished {
                    return Ok(());
                }
                continue;
            }
            let entry_ordinal = self.next_entry_ordinal;
            let mut reader = SliceReader::with_range(
                self.dictionary.bytes,
                self.next_offset,
                block.byte_range.end,
            )?;
            let term_ordinal = block
                .base_term_ord
                .checked_add(u32::try_from(entry_ordinal).map_err(|_| {
                    TermDictionaryError::ValueOutOfRange {
                        field: "term ordinal",
                        value: u64::try_from(entry_ordinal).unwrap_or(u64::MAX),
                        offset: self.next_offset,
                    }
                })?)
                .ok_or(TermDictionaryError::SizeOverflow {
                    field: "cursor term ordinal",
                })?;
            let decoded = decode_entry(
                &mut reader,
                entry_ordinal,
                &mut self.key,
                self.dictionary.schema,
                self.dictionary.sections,
                usize::try_from(term_ordinal).unwrap_or(usize::MAX),
            )?;
            self.next_offset = reader.position();
            self.next_entry_ordinal = self.next_entry_ordinal.saturating_add(1);
            self.current = Some(TermMatch {
                term_ord: term_ordinal,
                metadata: decoded.metadata,
            });
            return Ok(());
        }
    }

    fn enforce_filters(&mut self) -> Result<(), TermDictionaryError> {
        loop {
            let Some(current) = self.current() else {
                return Ok(());
            };
            if let Some(field_ord) = self.field_filter {
                match current.field_ord.cmp(&field_ord) {
                    Ordering::Less => {
                        self.advance_raw()?;
                        continue;
                    }
                    Ordering::Greater => {
                        self.current = None;
                        self.finished = true;
                        return Ok(());
                    }
                    Ordering::Equal => {}
                }
            }
            if let Some(prefix) = self.prefix_filter.as_ref() {
                if !self.key.starts_with(prefix) {
                    self.current = None;
                    self.finished = true;
                    return Ok(());
                }
            }
            if let Some(upper) = self.upper.as_ref() {
                let ordering = self.key.as_slice().cmp(&upper.key);
                if ordering == Ordering::Greater
                    || (ordering == Ordering::Equal && !upper.inclusive)
                {
                    self.current = None;
                    self.finished = true;
                }
            }
            return Ok(());
        }
    }
}
