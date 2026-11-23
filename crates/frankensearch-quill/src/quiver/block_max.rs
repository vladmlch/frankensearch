use thiserror::Error;

use super::doclen::DocLenField;
use super::postings::{POSTINGS_PER_BLOCK, Posting, PostingCodecError, PostingList};
use super::postings_codec::{
    DecodedBlock, decode_block_at, encode_bitmap_block, encode_for_block, encode_vint_block,
};

/// Typed failures from encoding or validating an FSLX BLOCKMAX term stream.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BlockMaxError {
    /// The referenced POSTINGS block could not be decoded consistently.
    #[error("posting validation failed while processing BLOCKMAX: {0}")]
    Posting(#[from] PostingCodecError),
    /// Durable bytes ended before one complete BLOCKMAX entry was available.
    #[error(
        "truncated BLOCKMAX stream at offset {offset}: need {needed} bytes, only {remaining} remain"
    )]
    Truncated {
        /// Byte offset relative to the term's BLOCKMAX span.
        offset: usize,
        /// Required byte count.
        needed: usize,
        /// Available byte count.
        remaining: usize,
    },
    /// A u64 LEB128 offset used more bytes than its shortest encoding.
    #[error("non-canonical BLOCKMAX offset vint at byte {offset}")]
    NonCanonicalVint {
        /// Vint start relative to the term's BLOCKMAX span.
        offset: usize,
    },
    /// A u64 LEB128 offset exceeded ten bytes or the u64 domain.
    #[error("BLOCKMAX offset vint overflow at byte {offset}")]
    VintOverflow {
        /// Vint start relative to the term's BLOCKMAX span.
        offset: usize,
    },
    /// A real posting block cannot have maximum frequency zero.
    #[error("BLOCKMAX entry {block_index} uses zero as its maximum-frequency code")]
    ZeroMaximumFrequency {
        /// Zero-based posting-block index.
        block_index: usize,
    },
    /// The redundant first-doc skip key disagreed with POSTINGS.
    #[error("BLOCKMAX entry {block_index} first doc mismatch: expected {expected}, got {actual}")]
    FirstDocMismatch {
        /// Zero-based posting-block index.
        block_index: usize,
        /// POSTINGS block's validated first docid.
        expected: u32,
        /// Durable BLOCKMAX value.
        actual: u32,
    },
    /// The redundant posting-byte skip offset disagreed with POSTINGS.
    #[error(
        "BLOCKMAX entry {block_index} posting offset mismatch: expected {expected}, got {actual}"
    )]
    BlockOffsetMismatch {
        /// Zero-based posting-block index.
        block_index: usize,
        /// POSTINGS block's validated term-relative byte offset.
        expected: u64,
        /// Durable BLOCKMAX value.
        actual: u64,
    },
    /// A frequency code was not the unique conservative encoding of the block.
    #[error(
        "BLOCKMAX entry {block_index} frequency code mismatch: expected {expected}, got {actual}"
    )]
    MaximumFrequencyMismatch {
        /// Zero-based posting-block index.
        block_index: usize,
        /// Canonical code derived from POSTINGS.
        expected: u8,
        /// Durable BLOCKMAX value.
        actual: u8,
    },
    /// A posting docid had no corresponding DOCLEN value.
    #[error("BLOCKMAX entry {block_index} has no fieldnorm for docid {doc_id}")]
    MissingFieldnorm {
        /// Zero-based posting-block index.
        block_index: usize,
        /// Present posting docid missing from DOCLEN.
        doc_id: u32,
    },
    /// The stored componentwise minimum fieldnorm disagreed with DOCLEN.
    #[error(
        "BLOCKMAX entry {block_index} minimum fieldnorm mismatch: expected {expected}, got {actual}"
    )]
    MinimumFieldnormMismatch {
        /// Zero-based posting-block index.
        block_index: usize,
        /// Minimum stored fieldnorm ID over the block's present docs.
        expected: u8,
        /// Durable BLOCKMAX value.
        actual: u8,
    },
    /// Bytes remained after the exact one-entry-per-posting-block grammar.
    #[error("BLOCKMAX stream has {remaining} trailing bytes at offset {offset}")]
    TrailingBytes {
        /// First unconsumed byte.
        offset: usize,
        /// Unconsumed byte count.
        remaining: usize,
    },
    /// Q1 concat inputs were not in strictly increasing docid order.
    #[error(
        "BLOCKMAX concat part {part_index} starts at docid {first_doc}, not after {previous_last_doc}"
    )]
    NonAscendingConcat {
        /// Source part containing the invalid first block.
        part_index: usize,
        /// Last docid in the prior non-empty source part.
        previous_last_doc: u32,
        /// First docid in this source part.
        first_doc: u32,
    },
    /// Checked size, offset, or posting-ordinal arithmetic overflowed.
    #[error("arithmetic overflow while computing BLOCKMAX {field}")]
    ArithmeticOverflow {
        /// Value being computed.
        field: &'static str,
    },
    /// A bounded metadata allocation failed instead of panicking.
    #[error("unable to reserve BLOCKMAX {resource} for {count} entries")]
    Allocation {
        /// Collection being reserved.
        resource: &'static str,
        /// Requested entry count.
        count: usize,
    },
}

/// One canonical BLOCKMAX record, aligned one-for-one with a POSTINGS block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockMaxEntry {
    /// First absolute docid in the posting block.
    pub(super) first_doc: u32,
    /// Byte offset from this term's `postings_offset`.
    pub(super) block_offset: u64,
    /// Conservative maximum-frequency code.
    pub(super) max_frequency_code: u8,
    /// Minimum stored fieldnorm ID among docs in the block.
    pub(super) min_fieldnorm_id: u8,
}

impl BlockMaxEntry {
    /// First absolute docid in the posting block.
    #[must_use]
    pub const fn first_doc(self) -> u32 {
        self.first_doc
    }

    /// Byte offset from this term's `postings_offset`.
    #[must_use]
    pub const fn block_offset(self) -> u64 {
        self.block_offset
    }

    /// Conservative maximum-frequency wire code.
    #[must_use]
    pub const fn max_frequency_code(self) -> u8 {
        self.max_frequency_code
    }

    /// Exact minimum stored fieldnorm ID among docs in the block.
    #[must_use]
    pub const fn min_fieldnorm_id(self) -> u8 {
        self.min_fieldnorm_id
    }

    /// Decode the conservative maximum frequency represented by this entry.
    #[must_use]
    pub const fn max_frequency(self) -> u32 {
        crate::contract::block_max_frequency_from_code(self.max_frequency_code)
    }

    /// Decode the minimum quantized document length represented by this entry.
    #[must_use]
    pub fn min_fieldnorm(self) -> u32 {
        crate::contract::id_to_fieldnorm(self.min_fieldnorm_id)
    }

    /// Compute the conservative live-snapshot BM25 tf-factor.
    #[must_use]
    pub fn tf_factor_upper_bound(self, live_avgdl: f32) -> Option<f32> {
        crate::contract::block_max_tf_factor(
            self.max_frequency_code,
            self.min_fieldnorm_id,
            live_avgdl,
        )
    }

    /// Apply the full non-negative `idf * (1 + k1) * boost` live term weight.
    #[must_use]
    pub fn score_upper_bound(self, live_avgdl: f32, nonnegative_weight: f32) -> Option<f32> {
        crate::contract::block_max_score(
            self.max_frequency_code,
            self.min_fieldnorm_id,
            live_avgdl,
            nonnegative_weight,
        )
    }
}

/// Owned canonical bytes for one term's BLOCKMAX range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedBlockMax {
    pub(super) bytes: Vec<u8>,
    pub(super) entry_count: usize,
}

impl EncodedBlockMax {
    /// Recompute one bound record per validated posting block.
    ///
    /// `fieldnorm_for_doc` must return the stored DOCLEN byte for every
    /// POSTINGS-present docid. Every `u8`, including zero, is valid; `None`
    /// means the column is missing or the docid is out of range.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a missing fieldnorm, inconsistent validated
    /// POSTINGS bytes, checked arithmetic overflow, or allocation failure.
    pub fn encode<F>(
        postings: &PostingList<'_>,
        mut fieldnorm_for_doc: F,
    ) -> Result<Self, BlockMaxError>
    where
        F: FnMut(u32) -> Option<u8>,
    {
        let entry_count = postings.block_count();
        let mut bytes = allocate_block_max_bytes(entry_count, "fresh-seal bytes")?;
        for block_index in 0..postings.block_count() {
            let entry = derive_block_max_entry(postings, block_index, &mut fieldnorm_for_doc)?;
            append_block_max_entry(entry, &mut bytes);
        }
        Ok(Self { bytes, entry_count })
    }

    /// Re-emit Q1 concat inputs without recomputing either bound component.
    ///
    /// Only `block_offset` changes: each source offset is increased by the sum
    /// of prior source term POSTINGS byte lengths. No seam entry is added.
    ///
    /// # Errors
    ///
    /// Returns a typed error for out-of-order source docids, offset/size
    /// overflow, or allocation failure.
    pub fn concatenate(parts: &[&BlockMaxConcatList<'_>]) -> Result<Self, BlockMaxError> {
        let entry_count = parts.iter().try_fold(0_usize, |count, part| {
            count
                .checked_add(part.entry_count())
                .ok_or(BlockMaxError::ArithmeticOverflow {
                    field: "concat entry count",
                })
        })?;
        let mut bytes = allocate_block_max_bytes(entry_count, "concat bytes")?;

        let mut posting_base = 0_u64;
        let mut previous_last_doc = None;
        for (part_index, part) in parts.iter().enumerate() {
            if let (Some(previous), Some(first)) = (previous_last_doc, part.entries.first()) {
                if first.first_doc <= previous {
                    return Err(BlockMaxError::NonAscendingConcat {
                        part_index,
                        previous_last_doc: previous,
                        first_doc: first.first_doc,
                    });
                }
            }
            for entry in &part.entries {
                append_block_max_entry(
                    BlockMaxEntry {
                        first_doc: entry.first_doc,
                        block_offset: posting_base.checked_add(entry.block_offset).ok_or(
                            BlockMaxError::ArithmeticOverflow {
                                field: "rebased block offset",
                            },
                        )?,
                        max_frequency_code: entry.max_frequency_code,
                        min_fieldnorm_id: entry.min_fieldnorm_id,
                    },
                    &mut bytes,
                );
            }
            posting_base = posting_base.checked_add(part.posting_bytes_len).ok_or(
                BlockMaxError::ArithmeticOverflow {
                    field: "concat posting-byte prefix",
                },
            )?;
            if let Some(last_doc) = part.last_doc {
                previous_last_doc = Some(last_doc);
            }
        }

        Ok(Self { bytes, entry_count })
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

    /// Number of one-per-posting-block records.
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Re-open bytes as a merge-only view that cannot calculate score bounds.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the wire or POSTINGS cross-section invariants
    /// fail. The opaque fieldnorm byte is intentionally not trusted for score
    /// pruning until [`Self::block_max_list`] performs DOCLEN validation.
    pub fn concat_list<'a>(
        &'a self,
        postings: &PostingList<'_>,
    ) -> Result<BlockMaxConcatList<'a>, BlockMaxError> {
        BlockMaxConcatList::parse(&self.bytes, postings)
    }

    /// Re-open these owned bytes against the referenced POSTINGS term stream.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the wire or cross-section invariants fail.
    pub fn block_max_list<'a>(
        &'a self,
        postings: &PostingList<'_>,
        fieldnorms: DocLenField<'_>,
    ) -> Result<BlockMaxList<'a>, BlockMaxError> {
        BlockMaxList::parse(&self.bytes, postings, fieldnorms)
    }
}

/// Merge-only BLOCKMAX view with no score-bound API.
///
/// This validates canonical wire shape plus redundant POSTINGS keys and keeps
/// `min_fieldnorm_id` opaque. It is sufficient for Q1 byte-preserving concat;
/// only [`BlockMaxList`] can expose entries and calculate pruning bounds after
/// validation against DOCLEN.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockMaxConcatList<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) entries: Vec<BlockMaxEntry>,
    posting_bytes_len: u64,
    pub(super) last_doc: Option<u32>,
}

impl<'a> BlockMaxConcatList<'a> {
    /// Validate the merge-safe wire and POSTINGS invariants.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed/non-canonical bytes, redundant-key
    /// mismatch, a non-canonical frequency code, or allocation failure.
    pub fn parse(bytes: &'a [u8], postings: &PostingList<'_>) -> Result<Self, BlockMaxError> {
        let block_count = postings.block_count();
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(block_count)
            .map_err(|_| BlockMaxError::Allocation {
                resource: "entries",
                count: block_count,
            })?;

        let mut reader = BlockMaxByteReader::new(bytes);
        for (block_index, posting_block) in postings.blocks().iter().enumerate() {
            let first_doc = reader.read_u32()?;
            let block_offset = reader.read_vint()?;
            let max_frequency_code = reader.read_u8()?;
            let min_fieldnorm_id = reader.read_u8()?;
            if max_frequency_code == 0 {
                return Err(BlockMaxError::ZeroMaximumFrequency { block_index });
            }
            if first_doc != posting_block.first_doc {
                return Err(BlockMaxError::FirstDocMismatch {
                    block_index,
                    expected: posting_block.first_doc,
                    actual: first_doc,
                });
            }
            let expected_offset = u64::try_from(posting_block.byte_offset).map_err(|_| {
                BlockMaxError::ArithmeticOverflow {
                    field: "posting block offset",
                }
            })?;
            if block_offset != expected_offset {
                return Err(BlockMaxError::BlockOffsetMismatch {
                    block_index,
                    expected: expected_offset,
                    actual: block_offset,
                });
            }
            let required_code = posting_block.max_frequency_code;
            if max_frequency_code != required_code {
                return Err(BlockMaxError::MaximumFrequencyMismatch {
                    block_index,
                    expected: required_code,
                    actual: max_frequency_code,
                });
            }
            entries.push(BlockMaxEntry {
                first_doc,
                block_offset,
                max_frequency_code,
                min_fieldnorm_id,
            });
        }
        reader.finish()?;

        let posting_bytes_len = u64::try_from(postings.as_bytes().len()).map_err(|_| {
            BlockMaxError::ArithmeticOverflow {
                field: "posting term byte length",
            }
        })?;
        let last_doc = postings.blocks().last().map(|block| block.last_doc);
        Ok(Self {
            bytes,
            entries,
            posting_bytes_len,
            last_doc,
        })
    }

    /// Exact borrowed durable bytes copied by Q1 concat.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Number of structurally validated merge records.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Borrowed, validated view of one term's BLOCKMAX range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockMaxList<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) entries: Vec<BlockMaxEntry>,
    block_last_docs: Vec<u32>,
    posting_bytes_len: u64,
}

impl<'a> BlockMaxList<'a> {
    /// Validate exact wire grammar and cross-check POSTINGS plus DOCLEN bounds.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed/non-canonical bytes, an under- or
    /// over-stated frequency/fieldnorm bound, a cross-section mismatch, or
    /// allocation failure. No score-capable view exists before both bound
    /// components have been validated.
    pub fn parse(
        bytes: &'a [u8],
        postings: &PostingList<'_>,
        fieldnorms: DocLenField<'_>,
    ) -> Result<Self, BlockMaxError> {
        Self::parse_with_fieldnorms(bytes, postings, |doc_id| {
            fieldnorms.fieldnorm_id(u64::from(doc_id))
        })
    }

    pub(super) fn parse_with_fieldnorms<F>(
        bytes: &'a [u8],
        postings: &PostingList<'_>,
        mut fieldnorm_for_doc: F,
    ) -> Result<Self, BlockMaxError>
    where
        F: FnMut(u32) -> Option<u8>,
    {
        let structural = BlockMaxConcatList::parse(bytes, postings)?;
        let block_count = structural.entry_count();
        let mut block_last_docs = Vec::new();
        block_last_docs
            .try_reserve_exact(block_count)
            .map_err(|_| BlockMaxError::Allocation {
                resource: "last-doc skip keys",
                count: block_count,
            })?;

        for (block_index, (entry, posting_block)) in
            structural.entries.iter().zip(postings.blocks()).enumerate()
        {
            let decoded = decode_block_for_block_max(postings, block_index)?;
            let required_minimum =
                minimum_block_fieldnorm(&decoded, block_index, &mut fieldnorm_for_doc)?;
            if entry.min_fieldnorm_id != required_minimum {
                return Err(BlockMaxError::MinimumFieldnormMismatch {
                    block_index,
                    expected: required_minimum,
                    actual: entry.min_fieldnorm_id,
                });
            }
            block_last_docs.push(posting_block.last_doc);
        }
        Ok(Self {
            bytes,
            entries: structural.entries,
            block_last_docs,
            posting_bytes_len: structural.posting_bytes_len,
        })
    }

    /// Exact borrowed durable bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Validated one-per-posting-block entries in durable order.
    #[must_use]
    pub fn entries(&self) -> &[BlockMaxEntry] {
        &self.entries
    }

    /// Number of validated BLOCKMAX records.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Referenced term POSTINGS byte length used by concat rebasing.
    #[must_use]
    pub const fn posting_bytes_len(&self) -> u64 {
        self.posting_bytes_len
    }

    /// Create a metadata-only block cursor positioned on the first entry.
    ///
    /// Validation is intentionally eager once per immutable term view; callers
    /// should retain this view and create cheap cursors from it per scorer.
    #[must_use]
    pub fn cursor(&self) -> BlockMaxCursor<'_> {
        BlockMaxCursor {
            entries: &self.entries,
            block_last_docs: &self.block_last_docs,
            block_index: (!self.entries.is_empty()).then_some(0),
        }
    }
}

/// Metadata-only cursor that performs no POSTINGS decoding while it moves.
pub struct BlockMaxCursor<'a> {
    entries: &'a [BlockMaxEntry],
    block_last_docs: &'a [u32],
    block_index: Option<usize>,
}

impl BlockMaxCursor<'_> {
    /// Current validated BLOCKMAX entry.
    #[must_use]
    pub fn current(&self) -> Option<BlockMaxEntry> {
        self.block_index
            .and_then(|index| self.entries.get(index).copied())
    }

    /// Current zero-based posting-block index.
    #[must_use]
    pub const fn block_index(&self) -> Option<usize> {
        self.block_index
    }

    /// Validated inclusive last docid of the current posting block.
    #[must_use]
    pub fn last_doc(&self) -> Option<u32> {
        self.block_index
            .and_then(|index| self.block_last_docs.get(index).copied())
    }

    /// Advance by one posting block. Exhaustion is fused.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<BlockMaxEntry> {
        let current = self.block_index?;
        let next = current + 1;
        if next >= self.entries.len() {
            self.block_index = None;
            return None;
        }
        self.block_index = Some(next);
        self.current()
    }

    /// Land on the first posting block whose validated last doc reaches target.
    ///
    /// This never rewinds. It skips complete blocks without decoding their
    /// postings and returns `None` when no later block can contain a matching
    /// docid.
    pub fn advance(&mut self, target: u32) -> Option<BlockMaxEntry> {
        let current = self.block_index?;
        if self.block_last_docs.get(current).copied()? >= target {
            return self.current();
        }
        let later = self.block_last_docs.get(current + 1..)?;
        let relative = later.partition_point(|last_doc| *last_doc < target);
        if relative == later.len() {
            self.block_index = None;
            return None;
        }
        self.block_index = Some(current + 1 + relative);
        self.current()
    }
}

pub(super) fn encode_posting_block(
    block: &[Posting],
    output: &mut Vec<u8>,
) -> Result<(), PostingCodecError> {
    if block.len() == POSTINGS_PER_BLOCK {
        let first = block.first().map_or(0, |posting| posting.doc_id);
        let last = block.last().map_or(first, |posting| posting.doc_id);
        let span = u64::from(last) - u64::from(first) + 1;
        if span < 512 {
            encode_bitmap_block(block, output)
        } else {
            encode_for_block(block, output)
        }
    } else {
        encode_vint_block(block, output)
    }
}

pub(super) fn derive_fresh_block_max_entry<F>(
    postings: &[Posting],
    block_index: usize,
    block_offset: u64,
    fieldnorm_for_doc: &mut F,
) -> Result<BlockMaxEntry, BlockMaxError>
where
    F: FnMut(u32) -> Option<u8>,
{
    let first_doc = postings
        .first()
        .ok_or(BlockMaxError::ArithmeticOverflow {
            field: "empty fresh-seal posting block",
        })?
        .doc_id;
    let maximum_frequency = postings
        .iter()
        .map(|posting| posting.freq)
        .max()
        .unwrap_or(0);
    let mut min_fieldnorm_id = u8::MAX;
    for posting in postings {
        let fieldnorm_id =
            fieldnorm_for_doc(posting.doc_id).ok_or(BlockMaxError::MissingFieldnorm {
                block_index,
                doc_id: posting.doc_id,
            })?;
        min_fieldnorm_id = min_fieldnorm_id.min(fieldnorm_id);
    }
    Ok(BlockMaxEntry {
        first_doc,
        block_offset,
        max_frequency_code: crate::contract::block_max_frequency_to_code(maximum_frequency),
        min_fieldnorm_id,
    })
}

fn derive_block_max_entry<F>(
    postings: &PostingList<'_>,
    block_index: usize,
    fieldnorm_for_doc: &mut F,
) -> Result<BlockMaxEntry, BlockMaxError>
where
    F: FnMut(u32) -> Option<u8>,
{
    let block = postings
        .blocks()
        .get(block_index)
        .ok_or(BlockMaxError::ArithmeticOverflow {
            field: "posting block index",
        })?;
    let decoded = decode_block_for_block_max(postings, block_index)?;
    let min_fieldnorm_id = minimum_block_fieldnorm(&decoded, block_index, fieldnorm_for_doc)?;
    Ok(BlockMaxEntry {
        first_doc: block.first_doc,
        block_offset: u64::try_from(block.byte_offset).map_err(|_| {
            BlockMaxError::ArithmeticOverflow {
                field: "posting block offset",
            }
        })?,
        max_frequency_code: canonical_block_max_frequency_code(&decoded),
        min_fieldnorm_id,
    })
}

pub(super) fn decode_block_for_block_max(
    postings: &PostingList<'_>,
    block_index: usize,
) -> Result<DecodedBlock, BlockMaxError> {
    let block = postings
        .blocks()
        .get(block_index)
        .ok_or(BlockMaxError::ArithmeticOverflow {
            field: "posting block index",
        })?;
    let base_posting_ordinal = usize::try_from(block.base_posting_ordinal).map_err(|_| {
        BlockMaxError::ArithmeticOverflow {
            field: "posting ordinal",
        }
    })?;
    Ok(decode_block_at(
        postings.as_bytes(),
        block.byte_offset,
        base_posting_ordinal,
    )?)
}

pub(super) fn canonical_block_max_frequency_code(decoded: &DecodedBlock) -> u8 {
    let count = usize::from(decoded.posting_count);
    let maximum = decoded.freqs[..count].iter().copied().max().unwrap_or(0);
    crate::contract::block_max_frequency_to_code(maximum)
}

fn minimum_block_fieldnorm<F>(
    decoded: &DecodedBlock,
    block_index: usize,
    fieldnorm_for_doc: &mut F,
) -> Result<u8, BlockMaxError>
where
    F: FnMut(u32) -> Option<u8>,
{
    let mut minimum = u8::MAX;
    for &doc_id in &decoded.docs[..usize::from(decoded.posting_count)] {
        let fieldnorm_id = fieldnorm_for_doc(doc_id).ok_or(BlockMaxError::MissingFieldnorm {
            block_index,
            doc_id,
        })?;
        minimum = minimum.min(fieldnorm_id);
    }
    Ok(minimum)
}

pub(super) fn allocate_block_max_bytes(
    entry_count: usize,
    resource: &'static str,
) -> Result<Vec<u8>, BlockMaxError> {
    const MAX_ENTRY_BYTES: usize = 4 + 10 + 1 + 1;
    let capacity =
        entry_count
            .checked_mul(MAX_ENTRY_BYTES)
            .ok_or(BlockMaxError::ArithmeticOverflow {
                field: "encoded byte capacity",
            })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| BlockMaxError::Allocation {
            resource,
            count: entry_count,
        })?;
    Ok(bytes)
}

pub(super) fn append_block_max_entry(entry: BlockMaxEntry, bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&entry.first_doc.to_le_bytes());
    write_vint64(entry.block_offset, bytes);
    bytes.push(entry.max_frequency_code);
    bytes.push(entry.min_fieldnorm_id);
}

pub(super) fn vint64_length(mut value: u64) -> usize {
    let mut length = 1;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn write_vint64(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

pub(super) struct BlockMaxByteReader<'a> {
    pub(super) bytes: &'a [u8],
    position: usize,
}

impl<'a> BlockMaxByteReader<'a> {
    pub(super) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], BlockMaxError> {
        let offset = self.position;
        let remaining = self.bytes.len().saturating_sub(offset);
        let end = offset
            .checked_add(length)
            .ok_or(BlockMaxError::ArithmeticOverflow {
                field: "reader position",
            })?;
        let bytes = self
            .bytes
            .get(offset..end)
            .ok_or(BlockMaxError::Truncated {
                offset,
                needed: length,
                remaining,
            })?;
        self.position = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, BlockMaxError> {
        let offset = self.position;
        self.take(1)?
            .first()
            .copied()
            .ok_or(BlockMaxError::Truncated {
                offset,
                needed: 1,
                remaining: 0,
            })
    }

    fn read_u32(&mut self) -> Result<u32, BlockMaxError> {
        let bytes = self.take(4)?;
        let mut little_endian = [0_u8; 4];
        little_endian.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(little_endian))
    }

    pub(super) fn read_vint(&mut self) -> Result<u64, BlockMaxError> {
        let start = self.position;
        let mut value = 0_u64;
        for byte_index in 0..10 {
            let byte = self.read_u8()?;
            if byte_index == 9 && byte & 0xfe != 0 {
                return Err(BlockMaxError::VintOverflow { offset: start });
            }
            value |= u64::from(byte & 0x7f) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint64_length(value) != byte_index + 1 {
                    return Err(BlockMaxError::NonCanonicalVint { offset: start });
                }
                return Ok(value);
            }
        }
        Err(BlockMaxError::VintOverflow { offset: start })
    }

    pub(super) fn finish(self) -> Result<(), BlockMaxError> {
        let remaining = self.bytes.len().saturating_sub(self.position);
        if remaining == 0 {
            Ok(())
        } else {
            Err(BlockMaxError::TrailingBytes {
                offset: self.position,
                remaining,
            })
        }
    }
}
