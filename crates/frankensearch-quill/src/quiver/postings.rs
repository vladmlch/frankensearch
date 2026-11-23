use std::ops::Range;
use std::sync::Arc;

use thiserror::Error;

use super::bitpack::BitpackError;
use super::block_max::{
    BlockMaxEntry, BlockMaxError, BlockMaxList, EncodedBlockMax, allocate_block_max_bytes,
    append_block_max_entry, canonical_block_max_frequency_code, derive_fresh_block_max_entry,
    encode_posting_block,
};
use super::doclen::DocLenField;
use super::postings_codec::{DecodedBlock, decode_block_at, validate_encoder_input};

/// Number of postings in every full FOR or bitmap block.
pub const POSTINGS_PER_BLOCK: usize = 128;
/// Default cap on metadata entries retained for one term's posting stream.
///
/// The default covers more block fragments than an ordinary segment should
/// need while bounding hostile one-posting fragmentation. Callers opening
/// deliberately larger trusted streams can opt into higher caps with
/// [`PostingList::parse_with_limits`].
pub const DEFAULT_MAX_POSTING_BLOCKS: usize = 1 << 18;
/// Default cap on postings eagerly validated for one term.
pub const DEFAULT_MAX_POSTINGS_PER_TERM: usize = 1 << 22;

/// Explicit resource budgets for validating one term's posting stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PostingListLimits {
    /// Maximum validated self-delimiting blocks retained for cursor skips.
    pub max_blocks: usize,
    /// Maximum postings eagerly validated across those blocks.
    pub max_postings: usize,
}

impl Default for PostingListLimits {
    fn default() -> Self {
        Self {
            max_blocks: DEFAULT_MAX_POSTING_BLOCKS,
            max_postings: DEFAULT_MAX_POSTINGS_PER_TERM,
        }
    }
}

pub(super) const POSTINGS_PER_BLOCK_U8: u8 = 128;
pub(super) const BLOCK_HEADER_LEN: usize = 4;
pub(super) const FOR_KIND: u8 = 0;
pub(super) const BITMAP_KIND: u8 = 1;
pub(super) const VINT_KIND: u8 = 2;
pub(super) const FREQ_ALL_ONE: u8 = 0;
pub(super) const FREQ_BITPACKED: u8 = 1;

/// One document occurrence in a term's posting list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Posting {
    /// Absolute global document id.
    pub doc_id: u32,
    /// Term frequency in this document. Always at least one.
    pub freq: u32,
}

impl Posting {
    /// Construct a posting. Validation occurs when a list is encoded.
    #[must_use]
    pub const fn new(doc_id: u32, freq: u32) -> Self {
        Self { doc_id, freq }
    }
}

/// Durable FSLX posting-block codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PostingBlockKind {
    /// Frame-of-reference block: absolute first docid plus bitpacked deltas.
    FrameOfReference,
    /// Dense 512-bit docid bitmap plus aligned frequencies.
    Bitmap,
    /// Partial block encoded as canonical u32 LEB128 pairs.
    Vint,
}

impl PostingBlockKind {
    pub(super) const fn code(self) -> u8 {
        match self {
            Self::FrameOfReference => FOR_KIND,
            Self::Bitmap => BITMAP_KIND,
            Self::Vint => VINT_KIND,
        }
    }

    pub(super) fn from_code(code: u8, offset: usize) -> Result<Self, PostingCodecError> {
        match code {
            FOR_KIND => Ok(Self::FrameOfReference),
            BITMAP_KIND => Ok(Self::Bitmap),
            VINT_KIND => Ok(Self::Vint),
            kind => Err(PostingCodecError::InvalidBlockKind { offset, kind }),
        }
    }
}

/// Typed failures from encoding or validating an FSLX posting stream.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PostingCodecError {
    /// Concatenation requires at least one validated source list.
    #[error("cannot concatenate an empty POSTINGS list")]
    EmptyConcat,
    /// Concat inputs must be ordered by their decoded absolute docids.
    #[error(
        "POSTINGS concat part {part_index} starts at docid {first_doc}, not after prior docid {previous_last_doc}"
    )]
    ConcatRangeOrder {
        /// Rejected source index.
        part_index: usize,
        /// Prior non-empty source's final absolute docid.
        previous_last_doc: u32,
        /// Current non-empty source's first absolute docid.
        first_doc: u32,
    },
    /// Checked concat preflight arithmetic overflowed.
    #[error("POSTINGS concat arithmetic overflow in part {part_index} while computing {field}")]
    ConcatArithmeticOverflow {
        /// Source index whose contribution overflowed.
        part_index: usize,
        /// Accumulator that overflowed.
        field: &'static str,
    },
    /// Reserving the exact concatenated byte stream failed without panicking.
    #[error("unable to reserve {bytes} concatenated POSTINGS bytes")]
    ConcatAllocation {
        /// Exact output byte count requested.
        bytes: usize,
    },
    /// A slice cannot be represented by the u32 TERMDICT `doc_freq` field.
    #[error("posting count {count} exceeds the u32 doc_freq range")]
    TooManyPostings {
        /// Rejected posting count.
        count: usize,
    },
    /// Frequencies are strictly positive.
    #[error("posting {index} for docid {doc_id} has zero frequency")]
    ZeroFrequency {
        /// Posting ordinal.
        index: usize,
        /// Posting docid.
        doc_id: u32,
    },
    /// Encoder input must be strictly ordered by absolute docid.
    #[error(
        "posting docids are not strictly ascending at index {index}: previous {previous}, current {doc_id}"
    )]
    NonAscendingInput {
        /// Posting ordinal.
        index: usize,
        /// Previous docid.
        previous: u32,
        /// Current docid.
        doc_id: u32,
    },
    /// A block payload cannot fit in its u16 length field.
    #[error("posting block at offset {block_offset} has {length} payload bytes, exceeding u16")]
    PayloadTooLarge {
        /// Block start in the term stream.
        block_offset: usize,
        /// Computed payload length.
        length: usize,
    },
    /// Durable bytes ended before a declared field or payload did.
    #[error(
        "truncated posting stream at offset {offset}: need {needed} bytes, only {remaining} remain"
    )]
    Truncated {
        /// Absolute byte offset in the term stream.
        offset: usize,
        /// Required byte count.
        needed: usize,
        /// Available byte count.
        remaining: usize,
    },
    /// The common header named an unknown codec.
    #[error("unknown posting block kind {kind} at offset {offset}")]
    InvalidBlockKind {
        /// Common-header offset.
        offset: usize,
        /// Rejected kind byte.
        kind: u8,
    },
    /// A codec/count pairing violates the FSLX grammar.
    #[error("posting block kind {kind} at offset {offset} has invalid count {count}")]
    InvalidBlockCount {
        /// Common-header offset.
        offset: usize,
        /// Raw kind byte.
        kind: u8,
        /// Raw posting count.
        count: u8,
    },
    /// Frequency metadata named an unknown representation.
    #[error("posting block at offset {block_offset} has invalid frequency kind {kind}")]
    InvalidFrequencyKind {
        /// Common-header offset.
        block_offset: usize,
        /// Rejected frequency kind.
        kind: u8,
    },
    /// Bitpacked frequencies were used for an all-one run.
    #[error("posting block at offset {block_offset} bitpacks an all-one frequency run")]
    NonCanonicalFrequencyEncoding {
        /// Common-header offset.
        block_offset: usize,
    },
    /// A bitmap's inclusive docid span is outside 128..=511.
    #[error("bitmap block at offset {block_offset} has invalid inclusive span {span}")]
    InvalidBitmapSpan {
        /// Common-header offset.
        block_offset: usize,
        /// Rejected span.
        span: u16,
    },
    /// A bitmap violates its boundary, cardinality, or padding invariant.
    #[error("bitmap block at offset {block_offset} violates {invariant}")]
    InvalidBitmap {
        /// Common-header offset.
        block_offset: usize,
        /// Stable invariant name.
        invariant: &'static str,
    },
    /// A bitpacked field did not use its unique minimal width.
    #[error(
        "posting block at offset {block_offset} uses non-canonical {field} width {encoded}; required {required}"
    )]
    NonCanonicalWidth {
        /// Common-header offset.
        block_offset: usize,
        /// Width-bearing field.
        field: &'static str,
        /// Encoded width.
        encoded: u8,
        /// Minimal required width.
        required: u8,
    },
    /// An unsigned LEB128 value used more bytes than its shortest form.
    #[error("non-canonical u32 vint at offset {offset}")]
    NonCanonicalVint {
        /// Vint start offset.
        offset: usize,
    },
    /// An unsigned LEB128 value exceeds u32 or five bytes.
    #[error("u32 vint overflow at offset {offset}")]
    VintOverflow {
        /// Vint start offset.
        offset: usize,
    },
    /// Reconstructing a durable value overflowed its declared integer domain.
    #[error("arithmetic overflow for {field} in posting block at offset {block_offset}")]
    ArithmeticOverflow {
        /// Common-header offset.
        block_offset: usize,
        /// Value being reconstructed.
        field: &'static str,
    },
    /// Decoded docids were not strictly ordered, including across blocks.
    #[error(
        "decoded posting order violation at ordinal {posting_ordinal}: previous {previous}, current {doc_id}"
    )]
    NonAscendingDecoded {
        /// Posting ordinal in the term stream.
        posting_ordinal: u32,
        /// Previous decoded docid.
        previous: u32,
        /// Current decoded docid.
        doc_id: u32,
    },
    /// A declared payload contained bytes beyond its canonical fields.
    #[error("posting block at offset {block_offset} has {remaining} trailing payload bytes")]
    TrailingPayload {
        /// Common-header offset.
        block_offset: usize,
        /// Unconsumed bytes.
        remaining: usize,
    },
    /// TERMDICT `doc_freq` disagrees with the sum of self-delimiting blocks.
    #[error("posting doc_freq mismatch: expected {expected}, decoded {actual}")]
    DocFrequencyMismatch {
        /// TERMDICT value.
        expected: u32,
        /// Sum of block counts.
        actual: u32,
    },
    /// Decoded blocks already exceed TERMDICT `doc_freq`; validation stops
    /// before processing any later bytes.
    #[error("posting doc_freq exceeded: expected {expected}, decoded at least {actual}")]
    DocFrequencyExceeded {
        /// TERMDICT value.
        expected: u32,
        /// Sum including the first block that exceeds the declaration.
        actual: u32,
    },
    /// A caller-selected block metadata budget was exhausted before any later
    /// bytes were inspected.
    #[error("posting block budget {limit} exhausted after {validated} validated blocks")]
    BlockBudgetExhausted {
        /// Configured metadata-entry cap.
        limit: usize,
        /// Blocks validated before validation stopped.
        validated: usize,
    },
    /// TERMDICT `doc_freq` exceeded a caller-selected eager validation budget.
    #[error("posting validation limit {limit} exceeded by declared doc_freq {actual}")]
    PostingLimitExceeded {
        /// Configured posting-work cap.
        limit: usize,
        /// Declared posting count.
        actual: usize,
    },
    /// Reserving validated block metadata failed without panicking.
    #[error("unable to reserve metadata for posting block {block_index}")]
    MetadataAllocation {
        /// Zero-based block index that could not be reserved.
        block_index: usize,
    },
    /// A cached validated block table was paired with a different POSTINGS span.
    #[error(
        "cached POSTINGS metadata expects {expected} bytes, but the selected term span has {actual}"
    )]
    CachedMetadataLengthMismatch {
        /// Byte length validated when the cache entry was created.
        expected: usize,
        /// Byte length of the immutable term span selected by TERMDICT.
        actual: usize,
    },
    /// A bounded collection request exceeded its explicit posting budget.
    #[error("posting materialization limit {limit} exceeded by {actual} postings")]
    MaterializationLimitExceeded {
        /// Caller-selected posting cap.
        limit: usize,
        /// Validated posting count.
        actual: usize,
    },
    /// Reserving the bounded materialized posting vector failed.
    #[error("unable to reserve space for {count} decoded postings")]
    MaterializationAllocation {
        /// Validated posting count requested.
        count: usize,
    },
    /// A canonical bitpacked substream failed validation.
    #[error("invalid bitpacked payload in posting block at offset {block_offset}: {source}")]
    Bitpack {
        /// Common-header offset.
        block_offset: usize,
        /// Bitpacking failure.
        #[source]
        source: BitpackError,
    },
}

/// Validated location and bounds of one self-delimiting posting block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostingBlockMeta {
    /// Durable codec.
    pub kind: PostingBlockKind,
    /// Number of postings in this block.
    pub posting_count: u8,
    /// Common-header offset relative to the term's `postings_offset`.
    pub byte_offset: usize,
    /// Common header plus payload length.
    pub byte_len: usize,
    /// First absolute docid in this block.
    pub first_doc: u32,
    /// Last absolute docid in this block.
    pub last_doc: u32,
    /// Posting ordinal of this block's first entry.
    pub base_posting_ordinal: u32,
    /// Canonical conservative maximum-frequency code derived during validation.
    pub max_frequency_code: u8,
}

impl PostingBlockMeta {
    /// Exact raw-byte range copied by Q1 concat-merge.
    #[must_use]
    pub fn byte_range(&self) -> Option<Range<usize>> {
        self.byte_offset
            .checked_add(self.byte_len)
            .map(|end| self.byte_offset..end)
    }
}

/// Owned bytes produced by the deterministic fresh-seal writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedPostingList {
    pub(super) bytes: Vec<u8>,
    pub(super) doc_freq: u32,
    pub(super) block_count: usize,
}

impl EncodedPostingList {
    pub(super) const fn validation_limits(
        posting_count: usize,
        block_count: usize,
    ) -> PostingListLimits {
        PostingListLimits {
            max_blocks: block_count,
            max_postings: posting_count,
        }
    }

    /// Concatenate validated posting streams without rewriting any block.
    ///
    /// Every block carries an absolute first docid, so an ordered Q1 merge is
    /// exactly the bytewise concatenation of its source streams. Empty source
    /// streams are allowed, but the source list itself must be non-empty.
    ///
    /// # Errors
    ///
    /// Rejects empty input, non-ascending source docids, checked count/length
    /// overflow, aggregate reader-budget exhaustion, or fallible allocation
    /// failure.
    pub fn concatenate(parts: &[&PostingList<'_>]) -> Result<Self, PostingCodecError> {
        Self::concatenate_with_limits(parts, PostingListLimits::default())
    }

    /// Concatenate with explicit aggregate reader budgets.
    ///
    /// This is primarily useful to pin resource-boundary behavior. Shipping
    /// concat merge uses [`PostingListLimits::default`] so an output accepted
    /// for publication is guaranteed to reopen through ordinary query reads.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::concatenate`].
    pub fn concatenate_with_limits(
        parts: &[&PostingList<'_>],
        limits: PostingListLimits,
    ) -> Result<Self, PostingCodecError> {
        if parts.is_empty() {
            return Err(PostingCodecError::EmptyConcat);
        }

        let mut byte_len = 0_usize;
        let mut doc_freq = 0_u32;
        let mut block_count = 0_usize;
        let mut previous_last_doc = None;
        for (part_index, part) in parts.iter().copied().enumerate() {
            if let Some(first) = part.blocks.first()
                && let Some(previous) = previous_last_doc
                && first.first_doc <= previous
            {
                return Err(PostingCodecError::ConcatRangeOrder {
                    part_index,
                    previous_last_doc: previous,
                    first_doc: first.first_doc,
                });
            }
            if let Some(last) = part.blocks.last() {
                previous_last_doc = Some(last.last_doc);
            }
            byte_len = byte_len.checked_add(part.bytes.len()).ok_or(
                PostingCodecError::ConcatArithmeticOverflow {
                    part_index,
                    field: "byte length",
                },
            )?;
            doc_freq = doc_freq.checked_add(part.doc_freq).ok_or(
                PostingCodecError::ConcatArithmeticOverflow {
                    part_index,
                    field: "doc_freq",
                },
            )?;
            block_count = block_count.checked_add(part.blocks.len()).ok_or(
                PostingCodecError::ConcatArithmeticOverflow {
                    part_index,
                    field: "block count",
                },
            )?;
        }
        let posting_count =
            usize::try_from(doc_freq).map_err(|_| PostingCodecError::PostingLimitExceeded {
                limit: limits.max_postings,
                actual: usize::MAX,
            })?;
        if posting_count > limits.max_postings {
            return Err(PostingCodecError::PostingLimitExceeded {
                limit: limits.max_postings,
                actual: posting_count,
            });
        }
        if block_count > limits.max_blocks {
            return Err(PostingCodecError::BlockBudgetExhausted {
                limit: limits.max_blocks,
                validated: block_count,
            });
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| PostingCodecError::ConcatAllocation { bytes: byte_len })?;
        for part in parts {
            bytes.extend_from_slice(part.as_bytes());
        }
        debug_assert_eq!(bytes.len(), byte_len);
        Ok(Self {
            bytes,
            doc_freq,
            block_count,
        })
    }

    /// Encode a strictly docid-ascending posting slice.
    ///
    /// Full blocks use BITMAP iff their inclusive span is below 512 and FOR
    /// otherwise. The final 1..=127 postings use a partial VINT block.
    ///
    /// # Errors
    ///
    /// Returns a typed error for duplicate/descending docids, zero frequency,
    /// an unrepresentable count, or any internal payload-length overflow.
    pub fn encode(postings: &[Posting]) -> Result<Self, PostingCodecError> {
        let doc_freq =
            u32::try_from(postings.len()).map_err(|_| PostingCodecError::TooManyPostings {
                count: postings.len(),
            })?;
        validate_encoder_input(postings)?;

        let mut bytes = Vec::new();
        for block in postings.chunks(POSTINGS_PER_BLOCK) {
            encode_posting_block(block, &mut bytes)?;
        }

        let expected_block_count = postings.len().div_ceil(POSTINGS_PER_BLOCK);
        Self::finish_encoding(bytes, doc_freq, postings.len(), expected_block_count)
    }

    /// Encode POSTINGS and its aligned BLOCKMAX stream in one fresh-seal pass.
    ///
    /// Bound components are derived directly from the source postings before
    /// the newly written POSTINGS bytes undergo their normal self-validation;
    /// fresh seal therefore does not decode the output a second time merely to
    /// compute BLOCKMAX. Compaction may instead call [`EncodedBlockMax::encode`]
    /// against an already validated posting view.
    ///
    /// # Errors
    ///
    /// Returns a typed posting-codec error, a missing DOCLEN value, checked
    /// offset overflow, or a fallible BLOCKMAX allocation failure.
    pub fn encode_with_block_max<F>(
        postings: &[Posting],
        mut fieldnorm_for_doc: F,
    ) -> Result<(Self, EncodedBlockMax), BlockMaxError>
    where
        F: FnMut(u32) -> Option<u8>,
    {
        let doc_freq =
            u32::try_from(postings.len()).map_err(|_| PostingCodecError::TooManyPostings {
                count: postings.len(),
            })?;
        validate_encoder_input(postings)?;

        let entry_count = postings.len().div_ceil(POSTINGS_PER_BLOCK);
        let mut posting_bytes = Vec::new();
        let mut block_max_bytes = allocate_block_max_bytes(entry_count, "fresh-seal bytes")?;
        for (block_index, block) in postings.chunks(POSTINGS_PER_BLOCK).enumerate() {
            let block_offset = u64::try_from(posting_bytes.len()).map_err(|_| {
                BlockMaxError::ArithmeticOverflow {
                    field: "fresh-seal posting block offset",
                }
            })?;
            let entry = derive_fresh_block_max_entry(
                block,
                block_index,
                block_offset,
                &mut fieldnorm_for_doc,
            )?;
            encode_posting_block(block, &mut posting_bytes)?;
            append_block_max_entry(entry, &mut block_max_bytes);
        }

        let postings = Self::finish_encoding(posting_bytes, doc_freq, postings.len(), entry_count)?;
        let block_max = EncodedBlockMax {
            bytes: block_max_bytes,
            entry_count,
        };
        Ok((postings, block_max))
    }

    fn finish_encoding(
        bytes: Vec<u8>,
        doc_freq: u32,
        posting_count: usize,
        expected_block_count: usize,
    ) -> Result<Self, PostingCodecError> {
        let parsed = PostingList::parse_with_limits(
            &bytes,
            doc_freq,
            Self::validation_limits(posting_count, expected_block_count),
        )?;
        let block_count = parsed.block_count();
        Ok(Self {
            bytes,
            doc_freq,
            block_count,
        })
    }

    /// Borrow the exact durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return the exact durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of encoded postings.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Number of self-delimiting blocks.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Re-open the owned bytes through the validating reader.
    ///
    /// # Errors
    ///
    /// Returns a typed error if an internal invariant was violated.
    pub fn posting_list(&self) -> Result<PostingList<'_>, PostingCodecError> {
        let posting_count =
            usize::try_from(self.doc_freq).map_err(|_| PostingCodecError::ArithmeticOverflow {
                block_offset: 0,
                field: "owned doc_freq",
            })?;
        PostingList::parse_with_limits(
            &self.bytes,
            self.doc_freq,
            Self::validation_limits(posting_count, self.block_count),
        )
    }
}

/// Borrowed, fully validated view of one term's POSTINGS byte range.
#[derive(Clone, Debug)]
pub struct PostingList<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) doc_freq: u32,
    pub(super) blocks: Vec<PostingBlockMeta>,
}

impl<'a> PostingList<'a> {
    /// Validate a complete TERMDICT-referenced posting range.
    ///
    /// Validation is eager for block grammar, canonical widths/vints, decoded
    /// ordering, arithmetic, and the expected `doc_freq`. Cursors may therefore
    /// skip by validated block bounds without trusting unbounded durable counts.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed or non-canonical durable bytes.
    pub fn parse(bytes: &'a [u8], expected_doc_freq: u32) -> Result<Self, PostingCodecError> {
        Self::parse_with_limits(bytes, expected_doc_freq, PostingListLimits::default())
    }

    /// Validate a complete posting range with explicit block and posting caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed bytes, a `doc_freq` mismatch, block
    /// budget exhaustion, or fallible metadata reservation failure.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        expected_doc_freq: u32,
        limits: PostingListLimits,
    ) -> Result<Self, PostingCodecError> {
        let expected_postings = usize::try_from(expected_doc_freq).map_err(|_| {
            PostingCodecError::PostingLimitExceeded {
                limit: limits.max_postings,
                actual: usize::MAX,
            }
        })?;
        if expected_postings > limits.max_postings {
            return Err(PostingCodecError::PostingLimitExceeded {
                limit: limits.max_postings,
                actual: expected_postings,
            });
        }
        let mut offset = 0_usize;
        let mut actual_doc_freq = 0_u32;
        let mut previous_last = None;
        let mut blocks = Vec::new();

        while offset < bytes.len() {
            if blocks.len() >= limits.max_blocks {
                return Err(PostingCodecError::BlockBudgetExhausted {
                    limit: limits.max_blocks,
                    validated: blocks.len(),
                });
            }
            let base_posting_ordinal = usize::try_from(actual_doc_freq).map_err(|_| {
                PostingCodecError::ArithmeticOverflow {
                    block_offset: offset,
                    field: "posting ordinal",
                }
            })?;
            let decoded = decode_block_at(bytes, offset, base_posting_ordinal)?;
            if let Some(previous) = previous_last {
                if decoded.first_doc <= previous {
                    return Err(PostingCodecError::NonAscendingDecoded {
                        posting_ordinal: actual_doc_freq,
                        previous,
                        doc_id: decoded.first_doc,
                    });
                }
            }
            let count = u32::from(decoded.posting_count);
            let next_doc_freq = actual_doc_freq.checked_add(count).ok_or(
                PostingCodecError::ArithmeticOverflow {
                    block_offset: offset,
                    field: "doc_freq sum",
                },
            )?;
            if next_doc_freq > expected_doc_freq {
                return Err(PostingCodecError::DocFrequencyExceeded {
                    expected: expected_doc_freq,
                    actual: next_doc_freq,
                });
            }
            blocks
                .try_reserve(1)
                .map_err(|_| PostingCodecError::MetadataAllocation {
                    block_index: blocks.len(),
                })?;
            blocks.push(PostingBlockMeta {
                kind: decoded.kind,
                posting_count: decoded.posting_count,
                byte_offset: offset,
                byte_len: decoded.byte_len,
                first_doc: decoded.first_doc,
                last_doc: decoded.last_doc,
                base_posting_ordinal: actual_doc_freq,
                max_frequency_code: canonical_block_max_frequency_code(&decoded),
            });
            previous_last = Some(decoded.last_doc);
            actual_doc_freq = next_doc_freq;
            offset = offset.checked_add(decoded.byte_len).ok_or(
                PostingCodecError::ArithmeticOverflow {
                    block_offset: offset,
                    field: "block byte offset",
                },
            )?;
        }

        if actual_doc_freq != expected_doc_freq {
            return Err(PostingCodecError::DocFrequencyMismatch {
                expected: expected_doc_freq,
                actual: actual_doc_freq,
            });
        }
        Ok(Self {
            bytes,
            doc_freq: actual_doc_freq,
            blocks,
        })
    }

    /// Exact borrowed durable bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Validated document frequency.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Number of validated blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Validated block index in durable order.
    #[must_use]
    pub fn blocks(&self) -> &[PostingBlockMeta] {
        &self.blocks
    }

    /// Exact raw bytes for one validated block.
    #[must_use]
    pub fn block_bytes(&self, block_index: usize) -> Option<&'a [u8]> {
        let block = self.blocks.get(block_index)?;
        self.bytes.get(block.byte_range()?)
    }

    /// Create a cursor positioned on the first posting, or exhausted if empty.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the already-validated immutable bytes fail to
    /// decode, which indicates an internal reader invariant regression.
    pub fn cursor(&self) -> Result<PostingCursor<'_>, PostingCodecError> {
        PostingCursor::new(self.bytes, &self.blocks)
    }

    /// Consume this validated view into a cursor that owns its block metadata.
    ///
    /// The durable bytes remain borrowed, while the validated block table moves
    /// behind shared ownership. Cloning the resulting cursor therefore does not
    /// copy the term's complete block table.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the already-validated immutable bytes fail to
    /// decode, which indicates an internal reader invariant regression.
    pub fn into_cursor(self) -> Result<PostingCursor<'a>, PostingCodecError> {
        PostingCursor::from_owned(self.bytes, self.blocks)
    }

    /// Consume this posting view into one score-capable cached metadata value.
    ///
    /// BLOCKMAX is validated against this exact POSTINGS value and DOCLEN
    /// reader inside the constructor, so callers cannot combine independently
    /// validated terms.
    ///
    /// # Errors
    ///
    /// Returns a typed BLOCKMAX error for malformed wire data or any
    /// POSTINGS/DOCLEN cross-section mismatch.
    pub(crate) fn into_pruning_metadata(
        self,
        block_max_bytes: &[u8],
        fieldnorms: DocLenField<'_>,
    ) -> Result<ValidatedTermPruningMetadata, BlockMaxError> {
        let block_max = BlockMaxList::parse(block_max_bytes, &self, fieldnorms)?;
        Ok(ValidatedTermPruningMetadata::from_validated(
            self, block_max,
        ))
    }

    /// Test-only convenience; production callers must select an explicit cap.
    ///
    /// # Errors
    ///
    /// Returns a typed error if allocation or the validated cursor invariant
    /// fails while materializing the list.
    #[cfg(test)]
    pub fn decode_all(&self) -> Result<Vec<Posting>, PostingCodecError> {
        self.decode_all_bounded(self.blocks.len().saturating_mul(POSTINGS_PER_BLOCK))
    }

    /// Decode all postings under a caller-selected materialization cap.
    ///
    /// # Errors
    ///
    /// Returns a typed error if `max_postings` is exceeded, allocation fails,
    /// or cursor decoding fails.
    pub fn decode_all_bounded(
        &self,
        max_postings: usize,
    ) -> Result<Vec<Posting>, PostingCodecError> {
        let posting_count = usize::try_from(self.doc_freq).map_err(|_| {
            PostingCodecError::MaterializationLimitExceeded {
                limit: max_postings,
                actual: usize::MAX,
            }
        })?;
        if posting_count > max_postings {
            return Err(PostingCodecError::MaterializationLimitExceeded {
                limit: max_postings,
                actual: posting_count,
            });
        }
        let mut output = Vec::new();
        output.try_reserve_exact(posting_count).map_err(|_| {
            PostingCodecError::MaterializationAllocation {
                count: posting_count,
            }
        })?;
        let mut cursor = self.cursor()?;
        while let Some(posting) = cursor.current() {
            output.push(posting);
            cursor.next()?;
        }
        Ok(output)
    }
}

/// Shared, fully cross-validated metadata for one immutable sealed term.
///
/// The posting block table and score-capable BLOCKMAX rows are constructed as
/// one value only after POSTINGS, BLOCKMAX, and DOCLEN validation succeeds.
/// Keeping them behind one `Arc` prevents production cursors from pairing
/// bounds with another term's block layout.
#[derive(Debug, Eq, PartialEq)]
pub struct ValidatedTermPruningMetadata {
    postings_bytes_len: usize,
    doc_freq: u32,
    posting_blocks: Vec<PostingBlockMeta>,
    block_max: Vec<BlockMaxEntry>,
}

impl ValidatedTermPruningMetadata {
    fn from_validated(postings: PostingList<'_>, block_max: BlockMaxList<'_>) -> Self {
        debug_assert_eq!(postings.block_count(), block_max.entry_count());
        debug_assert_eq!(
            u64::try_from(postings.as_bytes().len()).ok(),
            Some(block_max.posting_bytes_len())
        );
        Self {
            postings_bytes_len: postings.bytes.len(),
            doc_freq: postings.doc_freq,
            posting_blocks: postings.blocks,
            block_max: block_max.entries,
        }
    }

    /// Empty paired metadata used only to exercise cache admission policy.
    #[cfg(test)]
    pub fn empty_for_cache_test() -> Self {
        Self {
            postings_bytes_len: 0,
            doc_freq: 0,
            posting_blocks: Vec::new(),
            block_max: Vec::new(),
        }
    }

    /// Open a lazy cursor over the exact immutable term span validated here.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant error if TERMDICT selected a span with a
    /// different byte length or if the validated first block cannot reopen.
    pub fn cursor<'a>(
        self: &Arc<Self>,
        postings_bytes: &'a [u8],
    ) -> Result<PostingCursor<'a>, PostingCodecError> {
        if postings_bytes.len() != self.postings_bytes_len {
            return Err(PostingCodecError::CachedMetadataLengthMismatch {
                expected: self.postings_bytes_len,
                actual: postings_bytes.len(),
            });
        }
        PostingCursor::from_blocks(
            postings_bytes,
            PostingCursorBlocks::Pruning(Arc::clone(self)),
        )
    }

    /// Validated TERMDICT document frequency.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Score-capable bound rows paired one-for-one with `posting_blocks`.
    #[must_use]
    pub fn block_max(&self) -> &[BlockMaxEntry] {
        &self.block_max
    }

    /// Retained heap payload used by this cache entry, excluding `Arc` and
    /// sparse-cache row overhead.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.posting_blocks
            .len()
            .saturating_mul(std::mem::size_of::<PostingBlockMeta>())
            .saturating_add(
                self.block_max
                    .len()
                    .saturating_mul(std::mem::size_of::<BlockMaxEntry>()),
            )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CursorState {
    Positioned { block: usize, within: usize },
    Exhausted,
}

#[derive(Clone)]
enum PostingCursorBlocks<'a> {
    Borrowed(&'a [PostingBlockMeta]),
    Owned(Arc<[PostingBlockMeta]>),
    Pruning(Arc<ValidatedTermPruningMetadata>),
}

impl PostingCursorBlocks<'_> {
    fn as_slice(&self) -> &[PostingBlockMeta] {
        match self {
            Self::Borrowed(blocks) => blocks,
            Self::Owned(blocks) => blocks,
            Self::Pruning(metadata) => &metadata.posting_blocks,
        }
    }
}

/// Lazy posting cursor with block-level `advance` skipping.
pub struct PostingCursor<'a> {
    bytes: &'a [u8],
    blocks: PostingCursorBlocks<'a>,
    state: CursorState,
    decoded_docs: [u32; POSTINGS_PER_BLOCK],
    decoded_freqs: [u32; POSTINGS_PER_BLOCK],
    decoded_count: usize,
    /// One block retained behind the cursor for backward predecessor reads.
    ///
    /// A doc-major union resolving an equal-document tie walks *backwards*
    /// through a clause's history, and the read that serves it is otherwise
    /// stateless: descending through one 128-posting block decoded that block
    /// once per posting, where a forward walk pays one decode for the whole
    /// block. Retaining the block a backward read lands in collapses that to
    /// one decode, because the descent is monotone and every later probe in the
    /// same block is served from here.
    ///
    /// Documents only. Frequencies are never read backwards — an arrival order
    /// is decided by document identity alone — so caching them would double the
    /// footprint to answer a question nobody asks.
    ///
    /// `None` until a backward read first leaves the parked block, so a cursor
    /// driven only forwards never allocates.
    back: Option<BackwardBlock>,
    /// Blocks actually decoded, counted at the decode itself.
    ///
    /// The admission gate charges a block before it is read, so the predicates
    /// that decide those charges have to be checkable against what the cursor
    /// really did rather than against a second derivation of the same guess.
    /// This counts at the successful decode seam so it sees *every* decode,
    /// including a backward read's; counting at `load_block` alone made the
    /// backward path invisible to exactly the tests meant to bound it. Test
    /// builds only.
    #[cfg(test)]
    decoded_blocks: std::cell::Cell<usize>,
}

/// The one block retained behind a cursor for backward predecessor reads.
///
/// Deliberately not `Clone`. Retaining a block is optional and fallible by
/// policy — the read falls back to decoding when a reservation fails — and a
/// derived clone would smuggle an infallible `Vec` allocation into every
/// pruning fork, which is both a cost the policy refuses and a failure the
/// caller could not handle. [`PostingCursor`] clones cold instead.
struct BackwardBlock {
    index: usize,
    docs: Vec<u32>,
}

/// How one backward predecessor read will be answered.
///
/// Both [`PostingCursor::doc_before`] and
/// [`PostingCursor::doc_before_decodes_block`] are written against this single
/// value. They used to derive the same decision separately, which is the shape
/// that lets a permit predicate drift from the work it is predicting: one
/// charging for a decode that never happens, or worse, a decode running
/// unadmitted.
enum ArrivalPlan {
    /// Answerable from the validated block table alone.
    Metadata(Option<u32>),
    /// Served by the block the cursor is parked in.
    Parked { earlier: Option<u32> },
    /// Served by the retained backward block.
    Retained { earlier: Option<u32> },
    /// Requires decoding `index`, and is the only plan that costs a block.
    Decode { index: usize, earlier: Option<u32> },
}

impl Clone for PostingCursor<'_> {
    /// Clone the cursor cold: position and validated metadata carry over, the
    /// retained backward block does not.
    ///
    /// Every other field is a copy or a shared-handle bump, so this allocates
    /// nothing. Carrying the retained block would make each clone perform an
    /// infallible `Vec` clone — the exact allocation the cache's fallible
    /// `try_reserve_exact` policy exists to avoid, and one a fork could not
    /// refuse. Dropping it is free of semantic consequence: the cache is
    /// disposable, so a cold clone answers every backward read identically and
    /// simply pays one decode to warm itself again if it ever descends.
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes,
            blocks: self.blocks.clone(),
            state: self.state,
            decoded_docs: self.decoded_docs,
            decoded_freqs: self.decoded_freqs,
            decoded_count: self.decoded_count,
            back: None,
            #[cfg(test)]
            decoded_blocks: std::cell::Cell::new(self.decoded_blocks.get()),
        }
    }
}

impl<'a> PostingCursor<'a> {
    pub(super) fn new(
        bytes: &'a [u8],
        blocks: &'a [PostingBlockMeta],
    ) -> Result<Self, PostingCodecError> {
        Self::from_blocks(bytes, PostingCursorBlocks::Borrowed(blocks))
    }

    fn from_owned(
        bytes: &'a [u8],
        blocks: Vec<PostingBlockMeta>,
    ) -> Result<Self, PostingCodecError> {
        Self::from_blocks(bytes, PostingCursorBlocks::Owned(Arc::from(blocks)))
    }

    fn from_blocks(
        bytes: &'a [u8],
        blocks: PostingCursorBlocks<'a>,
    ) -> Result<Self, PostingCodecError> {
        let mut cursor = Self {
            bytes,
            blocks,
            state: CursorState::Exhausted,
            decoded_docs: [0; POSTINGS_PER_BLOCK],
            decoded_freqs: [0; POSTINGS_PER_BLOCK],
            decoded_count: 0,
            back: None,
            #[cfg(test)]
            decoded_blocks: std::cell::Cell::new(0),
        };
        if !cursor.blocks.as_slice().is_empty() {
            cursor.load_block(0)?;
            cursor.state = CursorState::Positioned {
                block: 0,
                within: 0,
            };
        }
        Ok(cursor)
    }

    /// Decode one validated block without disturbing the cursor's position.
    ///
    /// The metadata cross-check is the same one [`Self::load_block`] applies,
    /// because a read-only probe reads the same durable bytes and must refuse
    /// the same drift.
    fn decode_block(&self, block_index: usize) -> Result<DecodedBlock, PostingCodecError> {
        let block = self.blocks.as_slice().get(block_index).ok_or(
            PostingCodecError::ArithmeticOverflow {
                block_offset: self.bytes.len(),
                field: "cursor block index",
            },
        )?;
        let base_posting_ordinal = usize::try_from(block.base_posting_ordinal).map_err(|_| {
            PostingCodecError::ArithmeticOverflow {
                block_offset: block.byte_offset,
                field: "posting ordinal",
            }
        })?;
        let decoded = decode_block_at(self.bytes, block.byte_offset, base_posting_ordinal)?;
        if decoded.byte_len != block.byte_len
            || decoded.kind != block.kind
            || decoded.posting_count != block.posting_count
        {
            return Err(PostingCodecError::ArithmeticOverflow {
                block_offset: block.byte_offset,
                field: "validated block metadata",
            });
        }
        // Counted here, at the decode that actually happened, so a forward load
        // and a backward read are both visible to the same counter.
        #[cfg(test)]
        self.decoded_blocks
            .set(self.decoded_blocks.get().saturating_add(1));
        Ok(decoded)
    }

    fn load_block(&mut self, block_index: usize) -> Result<(), PostingCodecError> {
        let decoded = self.decode_block(block_index)?;
        self.decoded_docs = decoded.docs;
        self.decoded_freqs = decoded.freqs;
        self.decoded_count = usize::from(decoded.posting_count);
        Ok(())
    }

    /// Blocks this cursor has actually decoded.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn decoded_blocks(&self) -> usize {
        self.decoded_blocks.get()
    }

    /// Current posting, including a valid `u32::MAX` docid when present.
    #[must_use]
    pub fn current(&self) -> Option<Posting> {
        let CursorState::Positioned { within, .. } = self.state else {
            return None;
        };
        Some(Posting {
            doc_id: *self.decoded_docs.get(within)?,
            freq: *self.decoded_freqs.get(within)?,
        })
    }

    /// Current absolute docid.
    #[must_use]
    pub fn doc(&self) -> Option<u32> {
        self.current().map(|posting| posting.doc_id)
    }

    /// Current term frequency.
    #[must_use]
    pub fn freq(&self) -> Option<u32> {
        self.current().map(|posting| posting.freq)
    }

    /// Current zero-based posting-block index.
    #[must_use]
    pub const fn block_index(&self) -> Option<usize> {
        match self.state {
            CursorState::Positioned { block, .. } => Some(block),
            CursorState::Exhausted => None,
        }
    }

    /// Validated inclusive last docid of the current posting block.
    #[must_use]
    pub fn block_last_doc(&self) -> Option<u32> {
        self.block_index()
            .and_then(|index| self.blocks.as_slice().get(index))
            .map(|block| block.last_doc)
    }

    /// Whether [`Self::next`] would decode a block, without decoding one.
    ///
    /// This mirrors the branch in [`Self::next`] exactly and must keep
    /// mirroring it: a caller admits the cost of a block *before* the decode
    /// happens, so a predicate that drifts from the move it predicts either
    /// charges for work never done or lets work run unadmitted. It reads only
    /// the already-decoded block state and the validated block table.
    #[must_use]
    pub fn next_decodes_block(&self) -> bool {
        let CursorState::Positioned { block, within } = self.state else {
            return false;
        };
        if within + 1 < self.decoded_count {
            return false;
        }
        block + 1 < self.blocks.as_slice().len()
    }

    /// Total number of validated posting blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.as_slice().len()
    }

    /// Index of the first block whose validated `last_doc` reaches `target`.
    ///
    /// This is the same partition the seek in [`Self::advance`] performs, and
    /// `None` means no block can hold `target` — a seek there exhausts.
    #[must_use]
    pub fn block_containing(&self, target: u32) -> Option<usize> {
        let blocks = self.blocks.as_slice();
        let index = blocks.partition_point(|block| block.last_doc < target);
        (index < blocks.len()).then_some(index)
    }

    /// Whether [`Self::advance`] would decode a block, without decoding one.
    ///
    /// Mirrors [`Self::advance`] under the same obligation as
    /// [`Self::next_decodes_block`]: a satisfied target, a hit inside the
    /// decoded tail, and a target past every remaining block all decode
    /// nothing.
    #[must_use]
    pub fn advance_decodes_block(&self, target: u32) -> bool {
        let CursorState::Positioned {
            block: current_block,
            within: current_within,
        } = self.state
        else {
            return false;
        };
        if self.decoded_docs[current_within] >= target {
            return false;
        }
        let current_tail = &self.decoded_docs[current_within + 1..self.decoded_count];
        if current_tail.partition_point(|doc_id| *doc_id < target) < current_tail.len() {
            return false;
        }
        let later = &self.blocks.as_slice()[current_block + 1..];
        later.partition_point(|block| block.last_doc < target) != later.len()
    }

    /// Current zero-based posting ordinal for later POSITIONS alignment.
    #[must_use]
    pub fn posting_ordinal(&self) -> Option<u32> {
        let CursorState::Positioned { block, within } = self.state else {
            return None;
        };
        let base = self.blocks.as_slice().get(block)?.base_posting_ordinal;
        base.checked_add(u32::try_from(within).ok()?)
    }

    /// Move strictly forward by one posting. Exhaustion is fused.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the next validated block cannot be decoded.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Posting>, PostingCodecError> {
        let CursorState::Positioned { block, within } = self.state else {
            return Ok(None);
        };
        if within + 1 < self.decoded_count {
            self.state = CursorState::Positioned {
                block,
                within: within + 1,
            };
            return Ok(self.current());
        }

        let next_block = block + 1;
        if next_block >= self.blocks.as_slice().len() {
            self.state = CursorState::Exhausted;
            self.decoded_count = 0;
            return Ok(None);
        }
        self.load_block(next_block)?;
        self.state = CursorState::Positioned {
            block: next_block,
            within: 0,
        };
        Ok(self.current())
    }

    /// Land on the first posting whose docid is at least `target`.
    ///
    /// If the current docid already satisfies the target, the cursor does not
    /// move. Exhaustion is fused and `u32::MAX` remains a normal searchable id.
    ///
    /// # Errors
    ///
    /// Returns a typed error if a selected validated block cannot be decoded.
    pub fn advance(&mut self, target: u32) -> Result<Option<Posting>, PostingCodecError> {
        let CursorState::Positioned {
            block: current_block,
            within: current_within,
        } = self.state
        else {
            return Ok(None);
        };
        if self.decoded_docs[current_within] >= target {
            return Ok(self.current());
        }

        let current_tail = &self.decoded_docs[current_within + 1..self.decoded_count];
        let within_tail = current_tail.partition_point(|doc_id| *doc_id < target);
        if within_tail < current_tail.len() {
            self.state = CursorState::Positioned {
                block: current_block,
                within: current_within + 1 + within_tail,
            };
            return Ok(self.current());
        }

        let later = &self.blocks.as_slice()[current_block + 1..];
        let relative_block = later.partition_point(|block| block.last_doc < target);
        if relative_block == later.len() {
            self.state = CursorState::Exhausted;
            self.decoded_count = 0;
            return Ok(None);
        }
        let block = current_block + 1 + relative_block;
        self.load_block(block)?;
        let within =
            self.decoded_docs[..self.decoded_count].partition_point(|doc_id| *doc_id < target);
        if within == self.decoded_count {
            return Err(PostingCodecError::ArithmeticOverflow {
                block_offset: self.blocks.as_slice()[block].byte_offset,
                field: "validated block last_doc",
            });
        }
        self.state = CursorState::Positioned { block, within };
        Ok(self.current())
    }

    /// Inclusive last docid this term holds, read from validated block
    /// metadata without decoding and without moving.
    ///
    /// A doc-major union uses this to decide, before it moves anything,
    /// whether a scorer can survive a whole window: a term that exhausts
    /// mid-window is removed by a positional `swap_remove`, and that removal
    /// permutes the scorer vector in a way no per-term metadata can rebuild.
    #[must_use]
    pub fn final_doc(&self) -> Option<u32> {
        self.blocks.as_slice().last().map(|block| block.last_doc)
    }

    /// Greatest docid this term holds strictly below `target`.
    ///
    /// This is the *arrival predecessor* Tantivy's stable doc-major term union
    /// needs to reconstruct an equal-document order without walking to it.
    /// That traversal sorts by current docid, sums the equal-document prefix,
    /// steps it, and stably re-sorts, so a scorer that steps onto a document
    /// lands ahead of everyone already parked there: equal-document order is
    /// decreasing previous posting. A seek that jumps a block loses the steps
    /// but not this value.
    ///
    /// The probe is decode-free exactly where the union needs it. Right after
    /// `advance(target)` the cursor is parked in the block that holds `target`,
    /// whose docids are already decoded, so the answer is the neighbouring
    /// decoded docid — or, when `target` is that block's first posting, the
    /// previous block's validated `last_doc`.
    ///
    /// Descending *below* the parked block is the case that has to retain
    /// state. A tie walk asks for the predecessor of the predecessor and so on,
    /// so a stateless read decoded the same block again for every posting it
    /// held. The block a backward read lands in is retained, and because the
    /// descent is monotone every later probe inside it is then free. Only the
    /// first read of a block costs one, which is exactly what
    /// [`Self::doc_before_decodes_block`] reports — both are written against the
    /// same [`ArrivalPlan`], so the charge cannot drift from the work.
    ///
    /// # Errors
    ///
    /// Returns a typed error if a selected validated block cannot be decoded.
    pub fn doc_before(&mut self, target: u32) -> Result<Option<u32>, PostingCodecError> {
        match self.arrival_plan(target) {
            ArrivalPlan::Metadata(answer) => Ok(answer),
            ArrivalPlan::Parked { earlier } => {
                let within =
                    self.decoded_docs[..self.decoded_count].partition_point(|doc| *doc < target);
                Ok(match within.checked_sub(1) {
                    Some(previous) => Some(self.decoded_docs[previous]),
                    None => earlier,
                })
            }
            ArrivalPlan::Retained { earlier } => {
                // The plan only names this arm when the retained block matches,
                // so an absent one is a broken invariant. Answering from an
                // empty slice would silently return the wrong predecessor, and a
                // wrong predecessor is a silently wrong equal-document order —
                // precisely the failure this whole path exists to prevent.
                let back = self
                    .back
                    .as_ref()
                    .ok_or(PostingCodecError::ArithmeticOverflow {
                        block_offset: self.bytes.len(),
                        field: "retained backward block",
                    })?;
                let within = back.docs.partition_point(|doc| *doc < target);
                Ok(within
                    .checked_sub(1)
                    .map_or(earlier, |previous| Some(back.docs[previous])))
            }
            ArrivalPlan::Decode { index, earlier } => {
                let decoded = self.decode_block(index)?;
                let count = usize::from(decoded.posting_count);
                let answer = {
                    let within = decoded.docs[..count].partition_point(|doc| *doc < target);
                    within
                        .checked_sub(1)
                        .map_or(earlier, |previous| Some(decoded.docs[previous]))
                };
                // Retaining the block is an optimisation, so it is allowed to
                // fail: a cursor that cannot reserve keeps answering, and simply
                // decodes again next time rather than refusing a read the caller
                // has already been charged for.
                let mut docs = Vec::new();
                if docs.try_reserve_exact(count).is_ok() {
                    docs.extend_from_slice(&decoded.docs[..count]);
                    self.back = Some(BackwardBlock { index, docs });
                }
                Ok(answer)
            }
        }
    }

    /// Whether [`Self::doc_before`] would decode a block, determined without
    /// decoding one.
    ///
    /// This carries the same obligation as [`Self::advance_decodes_block`]: it
    /// mirrors the branch it predicts so a caller can admit the block before the
    /// read performs it. It cannot drift from that branch, because both read the
    /// same [`ArrivalPlan`]. Three answers cost nothing — a target no block can
    /// hold, the block the cursor is parked in, and the block retained behind it.
    #[must_use]
    pub fn doc_before_decodes_block(&self, target: u32) -> bool {
        matches!(self.arrival_plan(target), ArrivalPlan::Decode { .. })
    }

    /// Decide how a backward read for `target` will be answered.
    ///
    /// This is the single decision both the read and its permit are built on.
    fn arrival_plan(&self, target: u32) -> ArrivalPlan {
        let blocks = self.blocks.as_slice();
        // The first block that can hold a docid at or above `target`; every
        // earlier block lies wholly below it and is summarized by `last_doc`.
        let index = blocks.partition_point(|block| block.last_doc < target);
        let earlier = index
            .checked_sub(1)
            .and_then(|previous| blocks.get(previous))
            .map(|block| block.last_doc);
        let Some(block) = blocks.get(index) else {
            // Every posting is below `target`.
            return ArrivalPlan::Metadata(earlier);
        };
        if block.first_doc >= target {
            // The straddling block starts at or above `target`, so it holds
            // nothing below it.
            return ArrivalPlan::Metadata(earlier);
        }
        if self.block_index() == Some(index) {
            return ArrivalPlan::Parked { earlier };
        }
        if self.back.as_ref().is_some_and(|back| back.index == index) {
            return ArrivalPlan::Retained { earlier };
        }
        ArrivalPlan::Decode { index, earlier }
    }
}
