use std::ops::Range;

use thiserror::Error;

use super::block_max::write_vint64;
use super::position_cursor::{
    PositionByteReader, PositionCursor, PositionIter, consume_position_run,
    expected_input_position_count, expected_position_count, position_directory_len,
    positions_for_ordinal, validated_position_run_len, write_position_run,
};
use super::postings::{Posting, PostingBlockMeta, PostingCodecError, PostingCursor, PostingList};
use super::postings_codec::{validate_encoder_input, write_vint};

/// Fresh-seal payload target for one FSLX POSITIONS block.
///
/// Complete document runs are never split. A single run wider than the target
/// remains a legal oversized singleton, and Q1 concatenation preserves source
/// block seams without repacking them.
pub const POSITION_BLOCK_TARGET_BYTES: usize = 4_096;
/// Default maximum bytes accepted for one term's complete POSITIONS span.
pub const DEFAULT_MAX_POSITION_BYTES_PER_TERM: usize = 128 * 1024 * 1024;
/// Default maximum skip-directory entries retained for one positional term.
pub const DEFAULT_MAX_POSITION_BLOCKS: usize = 1 << 18;
/// Default maximum frequency-derived position values validated for one term.
pub const DEFAULT_MAX_POSITIONS_PER_TERM: u64 = 1 << 24;

pub(super) const POSITION_DIRECTORY_HEADER_LEN: usize = 4;

/// Explicit resource ceilings for one TERMDICT-referenced POSITIONS span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionListLimits {
    /// Maximum complete term-span bytes, including the block directory.
    pub max_bytes: usize,
    /// Maximum retained block-directory entries.
    pub max_blocks: usize,
    /// Maximum sum of POSTINGS frequencies validated for the term.
    pub max_positions: u64,
}

impl Default for PositionListLimits {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_POSITION_BYTES_PER_TERM,
            max_blocks: DEFAULT_MAX_POSITION_BLOCKS,
            max_positions: DEFAULT_MAX_POSITIONS_PER_TERM,
        }
    }
}

/// Typed failures from encoding or validating an FSLX POSITIONS term stream.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PositionCodecError {
    /// The corresponding POSTINGS list was invalid or could not be decoded.
    #[error(transparent)]
    Posting(#[from] PostingCodecError),
    /// Flat position values did not match the sum of posting frequencies.
    #[error("position count mismatch: POSTINGS frequencies require {expected}, got {actual}")]
    PositionCountMismatch {
        /// Frequency-derived position count.
        expected: u64,
        /// Supplied flat position count.
        actual: usize,
    },
    /// Encoder input positions decreased within one document run.
    #[error(
        "positions decrease for posting {posting_ordinal} at run index {position_index}: {previous} then {position}"
    )]
    NonAscendingPosition {
        /// Zero-based posting ordinal.
        posting_ordinal: u32,
        /// Zero-based position index within the document run.
        position_index: usize,
        /// Previous absolute position.
        previous: u32,
        /// Rejected absolute position.
        position: u32,
    },
    /// A complete term span exceeded the caller-selected byte ceiling.
    #[error("position byte limit {limit} exceeded by {actual} bytes")]
    ByteLimitExceeded {
        /// Configured byte ceiling.
        limit: usize,
        /// Required or supplied bytes.
        actual: usize,
    },
    /// Frequency-derived work exceeded the caller-selected position ceiling.
    #[error("position validation limit {limit} exceeded by {actual} values")]
    PositionLimitExceeded {
        /// Configured value ceiling.
        limit: u64,
        /// Frequency-derived values.
        actual: u64,
    },
    /// A durable or newly produced directory exceeded its block ceiling.
    #[error("position block limit {limit} exceeded by {actual} blocks")]
    BlockLimitExceeded {
        /// Configured block ceiling.
        limit: usize,
        /// Declared or produced blocks.
        actual: usize,
    },
    /// The fixed block count cannot describe the referenced posting list.
    #[error("position block count {block_count} is invalid for doc_freq {doc_freq}")]
    InvalidBlockCount {
        /// Declared directory row count.
        block_count: usize,
        /// Validated posting count.
        doc_freq: u32,
    },
    /// The first directory row was not the canonical `(0, 0)` pair.
    #[error(
        "first position block must start at posting 0 and payload offset 0, got ({first_posting_ordinal}, {block_offset})"
    )]
    InvalidDirectoryStart {
        /// First durable posting ordinal.
        first_posting_ordinal: u32,
        /// First blocks-area-relative byte offset.
        block_offset: u64,
    },
    /// Directory posting ordinals were not strictly increasing.
    #[error(
        "position block {block_index} posting ordinal is not ascending: {previous} then {current}"
    )]
    NonAscendingBlockOrdinal {
        /// Rejected block index.
        block_index: usize,
        /// Previous first posting ordinal.
        previous: u32,
        /// Rejected first posting ordinal.
        current: u32,
    },
    /// Directory payload offsets were not strictly increasing.
    #[error(
        "position block {block_index} payload offset is not ascending: {previous} then {current}"
    )]
    NonAscendingBlockOffset {
        /// Rejected block index.
        block_index: usize,
        /// Previous blocks-area-relative byte offset.
        previous: u64,
        /// Rejected blocks-area-relative byte offset.
        current: u64,
    },
    /// A block began outside the corresponding POSTINGS ordinal domain.
    #[error(
        "position block {block_index} starts at posting {first_posting_ordinal}, outside doc_freq {doc_freq}"
    )]
    BlockOrdinalOutOfRange {
        /// Rejected block index.
        block_index: usize,
        /// Rejected posting ordinal.
        first_posting_ordinal: u32,
        /// Validated posting count.
        doc_freq: u32,
    },
    /// A block offset did not point inside the blocks area.
    #[error(
        "position block {block_index} offset {block_offset} is outside payload length {payload_len}"
    )]
    BlockOffsetOutOfRange {
        /// Rejected block index.
        block_index: usize,
        /// Rejected blocks-area-relative byte offset.
        block_offset: u64,
        /// Complete blocks-area byte length.
        payload_len: usize,
    },
    /// A directory row described no postings or no payload bytes.
    #[error("position block {block_index} is empty")]
    EmptyBlock {
        /// Rejected block index.
        block_index: usize,
    },
    /// Only one complete document run may exceed the 4096-byte target.
    #[error(
        "position block {block_index} has {posting_count} documents in oversized {byte_len}-byte payload"
    )]
    OversizedMultiDocumentBlock {
        /// Rejected block index.
        block_index: usize,
        /// Frequency-derived document runs in the block.
        posting_count: u32,
        /// Block payload bytes.
        byte_len: usize,
    },
    /// Durable bytes ended before a fixed field or vint completed.
    #[error(
        "truncated position stream at offset {offset}: need {needed} bytes, only {remaining} remain"
    )]
    Truncated {
        /// Absolute byte offset in the term span.
        offset: usize,
        /// Minimum additional bytes needed.
        needed: usize,
        /// Available bytes in the bounded range.
        remaining: usize,
    },
    /// A vint used more bytes than its shortest representation.
    #[error("non-canonical {domain} position vint at offset {offset}")]
    NonCanonicalVint {
        /// Absolute vint start offset.
        offset: usize,
        /// Stable integer-domain name.
        domain: &'static str,
    },
    /// A vint exceeded its declared integer domain.
    #[error("{domain} position vint overflow at offset {offset}")]
    VintOverflow {
        /// Absolute vint start offset.
        offset: usize,
        /// Stable integer-domain name.
        domain: &'static str,
    },
    /// Reconstructing a delta-encoded absolute position overflowed u32.
    #[error(
        "position overflow for posting {posting_ordinal}: previous {previous} plus delta {delta}"
    )]
    PositionOverflow {
        /// Zero-based posting ordinal.
        posting_ordinal: u32,
        /// Previous absolute position.
        previous: u32,
        /// Rejected delta.
        delta: u32,
    },
    /// Frequency-derived runs did not exhaust one bounded block exactly.
    #[error("position block {block_index} has {remaining} trailing bytes")]
    TrailingBlockBytes {
        /// Rejected block index.
        block_index: usize,
        /// Unconsumed bytes.
        remaining: usize,
    },
    /// Concatenation inputs were not ordered by disjoint absolute docids.
    #[error("position concat docids are not ascending: {previous} then {current}")]
    NonAscendingConcat {
        /// Last docid in the preceding non-empty input.
        previous: u32,
        /// First docid in the rejected input.
        current: u32,
    },
    /// A requested posting ordinal was outside the paired list.
    #[error("position posting ordinal {posting_ordinal} is outside doc_freq {doc_freq}")]
    PostingOrdinalOutOfRange {
        /// Rejected zero-based posting ordinal.
        posting_ordinal: u32,
        /// Validated posting count.
        doc_freq: u32,
    },
    /// Checked offset, count, or length arithmetic overflowed.
    #[error("position arithmetic overflow for {field}")]
    ArithmeticOverflow {
        /// Stable value name.
        field: &'static str,
    },
    /// A bounded allocation request failed.
    #[error("unable to reserve {count} entries for {resource}")]
    Allocation {
        /// Stable allocation purpose.
        resource: &'static str,
        /// Requested entries or bytes.
        count: usize,
    },
    /// Already validated paired metadata disagreed during cursor use.
    #[error("position cursor invariant failed for {field}")]
    CursorInvariant {
        /// Stable invariant name.
        field: &'static str,
    },
}

/// Validated location and posting interval of one POSITIONS payload block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionBlockMeta {
    pub(super) base_posting_ordinal: u32,
    pub(super) posting_count: u32,
    pub(super) first_doc: u32,
    pub(super) last_doc: u32,
    pub(super) byte_offset: usize,
    pub(super) byte_len: usize,
}

impl PositionBlockMeta {
    /// First zero-based posting ordinal represented by the block.
    #[must_use]
    pub const fn base_posting_ordinal(&self) -> u32 {
        self.base_posting_ordinal
    }

    /// Number of complete document runs represented by the block.
    #[must_use]
    pub const fn posting_count(&self) -> u32 {
        self.posting_count
    }

    /// Absolute first docid, derived from the paired POSTINGS stream.
    #[must_use]
    pub const fn first_doc(&self) -> u32 {
        self.first_doc
    }

    /// Absolute last docid, derived from the paired POSTINGS stream.
    #[must_use]
    pub const fn last_doc(&self) -> u32 {
        self.last_doc
    }

    /// Absolute byte offset inside the complete positional term span.
    #[must_use]
    pub const fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Exact payload byte length.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub(super) fn byte_range(&self) -> Option<Range<usize>> {
        self.byte_offset
            .checked_add(self.byte_len)
            .map(|end| self.byte_offset..end)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RawPositionBlock {
    pub(super) first_posting_ordinal: u32,
    pub(super) block_offset: u64,
}

/// Owned canonical bytes for one term's POSITIONS stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedPositionList {
    bytes: Vec<u8>,
    doc_freq: u32,
    total_positions: u64,
    block_count: usize,
}

impl EncodedPositionList {
    /// Encode flat absolute positions aligned by each posting's `freq`.
    ///
    /// Fresh blocks greedily contain complete document runs up to the 4096-byte
    /// payload target. Positions reset per posting and may repeat.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid postings, count misalignment,
    /// decreasing positions, arithmetic overflow, or resource exhaustion.
    pub fn encode(postings: &[Posting], positions: &[u32]) -> Result<Self, PositionCodecError> {
        Self::encode_with_limits(postings, positions, PositionListLimits::default())
    }

    /// Encode with explicit byte, block, and position ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`], plus a selected
    /// resource ceiling when exceeded.
    pub fn encode_with_limits(
        postings: &[Posting],
        positions: &[u32],
        limits: PositionListLimits,
    ) -> Result<Self, PositionCodecError> {
        validate_encoder_input(postings)?;
        let doc_freq =
            u32::try_from(postings.len()).map_err(|_| PostingCodecError::TooManyPostings {
                count: postings.len(),
            })?;
        let total_positions = expected_input_position_count(postings, limits.max_positions)?;
        if total_positions != u64::try_from(positions.len()).unwrap_or(u64::MAX) {
            return Err(PositionCodecError::PositionCountMismatch {
                expected: total_positions,
                actual: positions.len(),
            });
        }

        let mut payload = Vec::new();
        let mut directory = Vec::new();
        let mut position_index = 0_usize;
        for (posting_index, posting) in postings.iter().enumerate() {
            let count = usize::try_from(posting.freq).map_err(|_| {
                PositionCodecError::ArithmeticOverflow {
                    field: "posting frequency",
                }
            })?;
            let end = position_index.checked_add(count).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "flat position range",
                },
            )?;
            let run = positions.get(position_index..end).ok_or(
                PositionCodecError::PositionCountMismatch {
                    expected: total_positions,
                    actual: positions.len(),
                },
            )?;
            let posting_ordinal = u32::try_from(posting_index).map_err(|_| {
                PositionCodecError::ArithmeticOverflow {
                    field: "posting ordinal",
                }
            })?;
            let run_len = validated_position_run_len(posting_ordinal, run)?;
            let next_payload_len = payload.len().checked_add(run_len).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position payload length",
                },
            )?;
            if next_payload_len > limits.max_bytes {
                return Err(PositionCodecError::ByteLimitExceeded {
                    limit: limits.max_bytes,
                    actual: next_payload_len,
                });
            }

            let start_new_block = directory.last().is_none_or(|_| {
                let current_start = usize::try_from(
                    directory
                        .last()
                        .map_or(0, |block: &RawPositionBlock| block.block_offset),
                )
                .unwrap_or(usize::MAX);
                payload
                    .len()
                    .checked_sub(current_start)
                    .and_then(|current| current.checked_add(run_len))
                    .is_none_or(|combined| combined > POSITION_BLOCK_TARGET_BYTES)
            });
            if start_new_block {
                if directory.len() >= limits.max_blocks {
                    return Err(PositionCodecError::BlockLimitExceeded {
                        limit: limits.max_blocks,
                        actual: directory.len().saturating_add(1),
                    });
                }
                directory
                    .try_reserve(1)
                    .map_err(|_| PositionCodecError::Allocation {
                        resource: "position block directory",
                        count: directory.len().saturating_add(1),
                    })?;
                directory.push(RawPositionBlock {
                    first_posting_ordinal: posting_ordinal,
                    block_offset: u64::try_from(payload.len()).map_err(|_| {
                        PositionCodecError::ArithmeticOverflow {
                            field: "position payload offset",
                        }
                    })?,
                });
            }
            payload
                .try_reserve(run_len)
                .map_err(|_| PositionCodecError::Allocation {
                    resource: "position payload bytes",
                    count: next_payload_len,
                })?;
            write_position_run(run, &mut payload);
            position_index = end;
        }

        Self::finish_encoding(&directory, &payload, doc_freq, total_positions, limits)
    }

    /// Concatenate validated source streams without rewriting payload blocks.
    ///
    /// Only directory ordinals and offsets are rebased. Underfilled source
    /// seams remain visible, so direct and staged Q1 merges create no fragments.
    ///
    /// # Errors
    ///
    /// Returns a typed error for reversed/overlapping source docids, checked
    /// count overflow, or resource exhaustion.
    pub fn concatenate(parts: &[&PositionList<'_>]) -> Result<Self, PositionCodecError> {
        Self::concatenate_with_limits(parts, PositionListLimits::default())
    }

    /// Concatenate with explicit byte, block, and position ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::concatenate`], plus a selected
    /// resource ceiling when exceeded.
    pub fn concatenate_with_limits(
        parts: &[&PositionList<'_>],
        limits: PositionListLimits,
    ) -> Result<Self, PositionCodecError> {
        let mut directory = Vec::new();
        let mut payload_len = 0_usize;
        let mut doc_freq = 0_u32;
        let mut total_positions = 0_u64;
        let mut previous_last_doc = None;

        for part in parts {
            let first_doc = part.posting_blocks.first().map(|block| block.first_doc);
            let last_doc = part.posting_blocks.last().map(|block| block.last_doc);
            if let (Some(previous), Some(current)) = (previous_last_doc, first_doc)
                && current <= previous
            {
                return Err(PositionCodecError::NonAscendingConcat { previous, current });
            }
            if last_doc.is_some() {
                previous_last_doc = last_doc;
            }

            total_positions = total_positions.checked_add(part.total_positions).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "concatenated position count",
                },
            )?;
            if total_positions > limits.max_positions {
                return Err(PositionCodecError::PositionLimitExceeded {
                    limit: limits.max_positions,
                    actual: total_positions,
                });
            }
            for block in &part.blocks {
                if directory.len() >= limits.max_blocks {
                    return Err(PositionCodecError::BlockLimitExceeded {
                        limit: limits.max_blocks,
                        actual: directory.len().saturating_add(1),
                    });
                }
                directory
                    .try_reserve(1)
                    .map_err(|_| PositionCodecError::Allocation {
                        resource: "concatenated position directory",
                        count: directory.len().saturating_add(1),
                    })?;
                directory.push(RawPositionBlock {
                    first_posting_ordinal: doc_freq.checked_add(block.base_posting_ordinal).ok_or(
                        PositionCodecError::ArithmeticOverflow {
                            field: "concatenated posting ordinal",
                        },
                    )?,
                    block_offset: u64::try_from(payload_len).map_err(|_| {
                        PositionCodecError::ArithmeticOverflow {
                            field: "concatenated payload offset",
                        }
                    })?,
                });
                payload_len = payload_len.checked_add(block.byte_len).ok_or(
                    PositionCodecError::ArithmeticOverflow {
                        field: "concatenated payload length",
                    },
                )?;
            }
            doc_freq = doc_freq.checked_add(part.doc_freq).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "concatenated doc_freq",
                },
            )?;
        }

        let directory_len = position_directory_len(&directory)?;
        let complete_len = directory_len.checked_add(payload_len).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "concatenated term length",
            },
        )?;
        if complete_len > limits.max_bytes {
            return Err(PositionCodecError::ByteLimitExceeded {
                limit: limits.max_bytes,
                actual: complete_len,
            });
        }
        let block_count =
            u32::try_from(directory.len()).map_err(|_| PositionCodecError::BlockLimitExceeded {
                limit: usize::try_from(u32::MAX).unwrap_or(usize::MAX),
                actual: directory.len(),
            })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(complete_len)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "concatenated position bytes",
                count: complete_len,
            })?;
        bytes.extend_from_slice(&block_count.to_le_bytes());
        for block in &directory {
            write_vint(block.first_posting_ordinal, &mut bytes);
            write_vint64(block.block_offset, &mut bytes);
        }
        for part in parts {
            for block in &part.blocks {
                let range = block
                    .byte_range()
                    .ok_or(PositionCodecError::ArithmeticOverflow {
                        field: "source position block range",
                    })?;
                let payload = part
                    .bytes
                    .get(range)
                    .ok_or(PositionCodecError::CursorInvariant {
                        field: "validated source position block",
                    })?;
                bytes.extend_from_slice(payload);
            }
        }
        debug_assert_eq!(bytes.len(), complete_len);
        Ok(Self {
            bytes,
            doc_freq,
            total_positions,
            block_count: directory.len(),
        })
    }

    fn finish_encoding(
        directory: &[RawPositionBlock],
        payload: &[u8],
        doc_freq: u32,
        total_positions: u64,
        limits: PositionListLimits,
    ) -> Result<Self, PositionCodecError> {
        let directory_len = position_directory_len(directory)?;
        let complete_len = directory_len.checked_add(payload.len()).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "position term length",
            },
        )?;
        if complete_len > limits.max_bytes {
            return Err(PositionCodecError::ByteLimitExceeded {
                limit: limits.max_bytes,
                actual: complete_len,
            });
        }
        let block_count =
            u32::try_from(directory.len()).map_err(|_| PositionCodecError::BlockLimitExceeded {
                limit: usize::try_from(u32::MAX).unwrap_or(usize::MAX),
                actual: directory.len(),
            })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(complete_len)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "position term bytes",
                count: complete_len,
            })?;
        bytes.extend_from_slice(&block_count.to_le_bytes());
        for block in directory {
            write_vint(block.first_posting_ordinal, &mut bytes);
            write_vint64(block.block_offset, &mut bytes);
        }
        bytes.extend_from_slice(payload);
        debug_assert_eq!(bytes.len(), complete_len);
        Ok(Self {
            bytes,
            doc_freq,
            total_positions,
            block_count: directory.len(),
        })
    }

    /// Exact canonical term bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return exact canonical term bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of aligned postings.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Frequency-derived number of encoded positions.
    #[must_use]
    pub const fn total_positions(&self) -> u64 {
        self.total_positions
    }

    /// Number of preserved payload blocks.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Re-open owned bytes against the corresponding validated POSTINGS list.
    ///
    /// # Errors
    ///
    /// Returns a typed error if an internal encoder invariant regressed.
    pub fn position_list<'a>(
        &'a self,
        postings: &'a PostingList<'_>,
    ) -> Result<PositionList<'a>, PositionCodecError> {
        PositionList::parse_with_limits(
            &self.bytes,
            postings,
            PositionListLimits {
                max_bytes: self.bytes.len(),
                max_blocks: self.block_count,
                max_positions: self.total_positions,
            },
        )
    }
}

/// Borrowed, fully validated POSITIONS span paired with its POSTINGS list.
#[derive(Clone, Debug)]
pub struct PositionList<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) posting_bytes: &'a [u8],
    pub(super) posting_blocks: &'a [PostingBlockMeta],
    pub(super) doc_freq: u32,
    pub(super) total_positions: u64,
    pub(super) blocks: Vec<PositionBlockMeta>,
}

impl<'a> PositionList<'a> {
    /// Validate a complete positional term span against POSTINGS frequencies.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed/corrupt durable bytes, frequency
    /// misalignment, integer overflow, or resource exhaustion.
    pub fn parse(
        bytes: &'a [u8],
        postings: &'a PostingList<'_>,
    ) -> Result<Self, PositionCodecError> {
        Self::parse_with_limits(bytes, postings, PositionListLimits::default())
    }

    /// Validate with explicit byte, block, and position ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::parse`], plus a selected resource
    /// ceiling when exceeded.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        postings: &'a PostingList<'_>,
        limits: PositionListLimits,
    ) -> Result<Self, PositionCodecError> {
        if bytes.len() > limits.max_bytes {
            return Err(PositionCodecError::ByteLimitExceeded {
                limit: limits.max_bytes,
                actual: bytes.len(),
            });
        }
        let total_positions = expected_position_count(postings, limits.max_positions)?;
        let count_bytes =
            bytes
                .get(..POSITION_DIRECTORY_HEADER_LEN)
                .ok_or(PositionCodecError::Truncated {
                    offset: 0,
                    needed: POSITION_DIRECTORY_HEADER_LEN,
                    remaining: bytes.len(),
                })?;
        let block_count = usize::try_from(u32::from_le_bytes([
            count_bytes[0],
            count_bytes[1],
            count_bytes[2],
            count_bytes[3],
        ]))
        .map_err(|_| PositionCodecError::ArithmeticOverflow {
            field: "position block count",
        })?;
        if block_count > limits.max_blocks {
            return Err(PositionCodecError::BlockLimitExceeded {
                limit: limits.max_blocks,
                actual: block_count,
            });
        }
        if (postings.doc_freq == 0 && block_count != 0)
            || (postings.doc_freq != 0
                && (block_count == 0
                    || u64::try_from(block_count).unwrap_or(u64::MAX)
                        > u64::from(postings.doc_freq)))
        {
            return Err(PositionCodecError::InvalidBlockCount {
                block_count,
                doc_freq: postings.doc_freq,
            });
        }
        if block_count == 0 {
            if bytes.len() != POSITION_DIRECTORY_HEADER_LEN {
                return Err(PositionCodecError::TrailingBlockBytes {
                    block_index: 0,
                    remaining: bytes.len() - POSITION_DIRECTORY_HEADER_LEN,
                });
            }
            return Ok(Self {
                bytes,
                posting_bytes: postings.bytes,
                posting_blocks: &postings.blocks,
                doc_freq: 0,
                total_positions: 0,
                blocks: Vec::new(),
            });
        }

        let mut reader =
            PositionByteReader::new(bytes, POSITION_DIRECTORY_HEADER_LEN, bytes.len())?;
        let mut raw_blocks: Vec<RawPositionBlock> = Vec::new();
        raw_blocks
            .try_reserve_exact(block_count)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "position block directory",
                count: block_count,
            })?;
        for block_index in 0..block_count {
            let first_posting_ordinal = reader.read_u32_vint()?;
            let block_offset = reader.read_u64_vint()?;
            if block_index == 0 && (first_posting_ordinal != 0 || block_offset != 0) {
                return Err(PositionCodecError::InvalidDirectoryStart {
                    first_posting_ordinal,
                    block_offset,
                });
            }
            if let Some(previous) = raw_blocks.last() {
                if first_posting_ordinal <= previous.first_posting_ordinal {
                    return Err(PositionCodecError::NonAscendingBlockOrdinal {
                        block_index,
                        previous: previous.first_posting_ordinal,
                        current: first_posting_ordinal,
                    });
                }
                if block_offset <= previous.block_offset {
                    return Err(PositionCodecError::NonAscendingBlockOffset {
                        block_index,
                        previous: previous.block_offset,
                        current: block_offset,
                    });
                }
            }
            if first_posting_ordinal >= postings.doc_freq {
                return Err(PositionCodecError::BlockOrdinalOutOfRange {
                    block_index,
                    first_posting_ordinal,
                    doc_freq: postings.doc_freq,
                });
            }
            raw_blocks.push(RawPositionBlock {
                first_posting_ordinal,
                block_offset,
            });
        }
        let blocks_area_start = reader.position();
        let payload_len = bytes.len().checked_sub(blocks_area_start).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "position blocks-area length",
            },
        )?;

        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(block_count)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "validated position metadata",
                count: block_count,
            })?;
        let mut postings_cursor = postings.cursor()?;
        let mut decoded_positions = 0_u64;
        for (block_index, raw) in raw_blocks.iter().copied().enumerate() {
            let next_ordinal = raw_blocks
                .get(block_index + 1)
                .map_or(postings.doc_freq, |next| next.first_posting_ordinal);
            let posting_count = next_ordinal.checked_sub(raw.first_posting_ordinal).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position block posting count",
                },
            )?;
            if posting_count == 0 {
                return Err(PositionCodecError::EmptyBlock { block_index });
            }
            let relative_start = usize::try_from(raw.block_offset).map_err(|_| {
                PositionCodecError::BlockOffsetOutOfRange {
                    block_index,
                    block_offset: raw.block_offset,
                    payload_len,
                }
            })?;
            let relative_end_u64 = raw_blocks.get(block_index + 1).map_or_else(
                || u64::try_from(payload_len).unwrap_or(u64::MAX),
                |next| next.block_offset,
            );
            let relative_end = usize::try_from(relative_end_u64).map_err(|_| {
                PositionCodecError::BlockOffsetOutOfRange {
                    block_index,
                    block_offset: relative_end_u64,
                    payload_len,
                }
            })?;
            if relative_start >= payload_len || relative_end > payload_len {
                return Err(PositionCodecError::BlockOffsetOutOfRange {
                    block_index,
                    block_offset: if relative_start >= payload_len {
                        raw.block_offset
                    } else {
                        relative_end_u64
                    },
                    payload_len,
                });
            }
            if relative_start >= relative_end {
                return Err(PositionCodecError::EmptyBlock { block_index });
            }
            let byte_offset = blocks_area_start.checked_add(relative_start).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position block start",
                },
            )?;
            let byte_end = blocks_area_start.checked_add(relative_end).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position block end",
                },
            )?;
            let byte_len = byte_end.checked_sub(byte_offset).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position block length",
                },
            )?;
            if byte_len > POSITION_BLOCK_TARGET_BYTES && posting_count != 1 {
                return Err(PositionCodecError::OversizedMultiDocumentBlock {
                    block_index,
                    posting_count,
                    byte_len,
                });
            }

            let actual_ordinal =
                postings_cursor
                    .posting_ordinal()
                    .ok_or(PositionCodecError::CursorInvariant {
                        field: "position block first posting",
                    })?;
            if actual_ordinal != raw.first_posting_ordinal {
                return Err(PositionCodecError::CursorInvariant {
                    field: "position directory posting ordinal",
                });
            }
            let first_doc = postings_cursor
                .doc()
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position block first doc",
                })?;
            let mut last_doc = first_doc;
            let mut block_reader = PositionByteReader::new(bytes, byte_offset, byte_end)?;
            for expected_ordinal in raw.first_posting_ordinal..next_ordinal {
                if postings_cursor.posting_ordinal() != Some(expected_ordinal) {
                    return Err(PositionCodecError::CursorInvariant {
                        field: "position/posting ordinal alignment",
                    });
                }
                let freq = postings_cursor
                    .freq()
                    .ok_or(PositionCodecError::CursorInvariant {
                        field: "position/posting frequency alignment",
                    })?;
                consume_position_run(&mut block_reader, expected_ordinal, freq)?;
                decoded_positions = decoded_positions.checked_add(u64::from(freq)).ok_or(
                    PositionCodecError::ArithmeticOverflow {
                        field: "decoded position count",
                    },
                )?;
                if decoded_positions > limits.max_positions {
                    return Err(PositionCodecError::PositionLimitExceeded {
                        limit: limits.max_positions,
                        actual: decoded_positions,
                    });
                }
                last_doc = postings_cursor
                    .doc()
                    .ok_or(PositionCodecError::CursorInvariant {
                        field: "position block last doc",
                    })?;
                postings_cursor.next()?;
            }
            if !block_reader.is_empty() {
                return Err(PositionCodecError::TrailingBlockBytes {
                    block_index,
                    remaining: block_reader.remaining(),
                });
            }
            blocks.push(PositionBlockMeta {
                base_posting_ordinal: raw.first_posting_ordinal,
                posting_count,
                first_doc,
                last_doc,
                byte_offset,
                byte_len,
            });
        }
        if postings_cursor.current().is_some() || decoded_positions != total_positions {
            return Err(PositionCodecError::CursorInvariant {
                field: "complete position/posting coverage",
            });
        }

        Ok(Self {
            bytes,
            posting_bytes: postings.bytes,
            posting_blocks: &postings.blocks,
            doc_freq: postings.doc_freq,
            total_positions,
            blocks,
        })
    }

    /// Exact borrowed durable term bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Validated aligned posting count.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Frequency-derived position count.
    #[must_use]
    pub const fn total_positions(&self) -> u64 {
        self.total_positions
    }

    /// Number of validated payload blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Validated payload block directory.
    #[must_use]
    pub fn blocks(&self) -> &[PositionBlockMeta] {
        &self.blocks
    }

    /// Exact payload bytes for one validated block.
    #[must_use]
    pub fn block_bytes(&self, block_index: usize) -> Option<&'a [u8]> {
        let range = self.blocks.get(block_index)?.byte_range()?;
        self.bytes.get(range)
    }

    /// Stream one document's absolute positions by zero-based posting ordinal.
    ///
    /// Lookup binary-searches the bounded block directory, then scans at most
    /// that block's complete frequency-derived runs. It allocates no per-doc
    /// metadata or output buffer.
    ///
    /// # Errors
    ///
    /// Returns a typed out-of-range error or an internal validated-cursor
    /// invariant failure.
    pub fn positions_for_ordinal(
        &self,
        posting_ordinal: u32,
    ) -> Result<PositionIter<'_>, PositionCodecError> {
        positions_for_ordinal(
            self.bytes,
            &self.blocks,
            self.posting_bytes,
            self.posting_blocks,
            self.doc_freq,
            posting_ordinal,
        )
    }

    /// Create an allocation-free posting cursor layered with position access.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the paired, already validated POSTINGS bytes no
    /// longer satisfy their cursor invariant.
    pub fn cursor(&self) -> Result<PositionCursor<'_>, PositionCodecError> {
        let postings = PostingCursor::new(self.posting_bytes, self.posting_blocks)?;
        let (position_block_index, position_reader) = if postings.current().is_some() {
            let first_block = self
                .blocks
                .first()
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "initial position block",
                })?;
            (
                Some(0),
                PositionByteReader::from_block(self.bytes, first_block)?,
            )
        } else {
            (
                None,
                PositionByteReader::new(self.bytes, self.bytes.len(), self.bytes.len())?,
            )
        };
        Ok(PositionCursor {
            position_bytes: self.bytes,
            position_blocks: &self.blocks,
            position_block_index,
            current_position_offset: position_reader.position(),
            current_positions_consumed: false,
            position_reader,
            postings,
        })
    }
}
