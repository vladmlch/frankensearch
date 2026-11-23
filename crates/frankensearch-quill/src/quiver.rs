//! Quiver postings and positions.
//!
//! FSLX posting lists are streams of self-delimiting FOR, bitmap, and partial
//! VINT blocks. Full blocks contain 128 postings; a fresh seal emits at most one
//! final partial block, while Q1 concat-merge may preserve partial blocks at
//! interior seams. Every full block stores an absolute first docid, so ordinary
//! merge copies POSTINGS bytes without rebasing or seam rewrites.

mod bitpack;
mod block_max;
mod doclen;
mod id_hash;
mod id_map;
mod numeric;
mod position_cursor;
mod positions;
mod postings;
mod postings_codec;
mod stats;
mod stored_meta;
mod stored_meta_section;

pub use bitpack::BitpackError;
pub use block_max::{
    BlockMaxConcatList, BlockMaxCursor, BlockMaxEntry, BlockMaxError, BlockMaxList, EncodedBlockMax,
};
pub use doclen::{
    DOCLEN_ALIGNMENT, DOCLEN_DIRECTORY_ENTRY_LEN, DOCLEN_HOLE_FIELDNORM_ID, DocLenCodecError,
    DocLenField, DocLenFieldInput, DocLenLimits, DocLenSection, EncodedDocLenSection,
    STATS_ENTRY_LEN,
};
pub use id_hash::{EncodedIdHashSection, IdHashCodecError, IdHashLimits, IdHashSection};
pub(crate) use id_hash::{IdHashLookupPlan, IdMapPresence};
pub(crate) use id_map::IdMapConcatPlan;
pub use id_map::{
    EncodedIdMapSection, ID_HASH_ENTRY_LEN, ID_HASH_HEADER_LEN, ID_HASH_SEED, ID_MAP_HASH_LEN,
    ID_MAP_HEADER_LEN, ID_MAP_MAX_SPAN, ID_MAP_OFFSET_LEN, IdMapCodecError, IdMapEntry,
    IdMapEntryInput, IdMapLimits, IdMapSection, id_map_content_hash,
};
pub use numeric::{
    EncodedNumericSection, NUMERIC_FIELD_HEADER_LEN, NUMERIC_PAIR_LEN, NumericCodecError,
    NumericDocIdSet, NumericEntries, NumericEntry, NumericField, NumericFieldInput, NumericLimits,
    NumericSection, NumericValue,
};
pub use position_cursor::{PositionCursor, PositionIter};
#[cfg(test)]
pub(crate) use position_cursor::{position_run_work_for_test, reset_position_run_work_for_test};
pub use positions::{
    DEFAULT_MAX_POSITION_BLOCKS, DEFAULT_MAX_POSITION_BYTES_PER_TERM,
    DEFAULT_MAX_POSITIONS_PER_TERM, EncodedPositionList, POSITION_BLOCK_TARGET_BYTES,
    PositionBlockMeta, PositionCodecError, PositionList, PositionListLimits,
};
pub(crate) use postings::ValidatedTermPruningMetadata;
pub use postings::{
    DEFAULT_MAX_POSTING_BLOCKS, DEFAULT_MAX_POSTINGS_PER_TERM, EncodedPostingList,
    POSTINGS_PER_BLOCK, Posting, PostingBlockKind, PostingBlockMeta, PostingCodecError,
    PostingCursor, PostingList, PostingListLimits,
};
pub use stats::{
    EncodedStatsSection, FieldStats, SnapshotFieldStats, StatsCodecError, StatsLimits,
    StatsSection, aggregate_field_stats,
};
pub use stored_meta::{
    BorrowedStoredMetaSection, EncodedStoredMetaSection, STORED_META_DIRECTORY_ENTRY_LEN,
    StoredMetaCodecError, StoredMetaFieldInput, StoredMetaLimits,
};
pub(crate) use stored_meta::{
    StoredMetaConcatPlan, StoredMetaLookup, ValidatedStoredMetaLookupPlan,
};
#[cfg(test)]
pub(crate) use stored_meta::{reset_stored_meta_materializations, stored_meta_materializations};
pub use stored_meta_section::{StoredMetaField, StoredMetaSection};

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let slice = bytes.get(offset..offset.checked_add(std::mem::size_of::<u16>())?)?;
    Some(u16::from_le_bytes(slice.try_into().ok()?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let bytes = bytes.get(offset..offset.checked_add(std::mem::size_of::<u32>())?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let bytes = bytes.get(offset..offset.checked_add(std::mem::size_of::<u64>())?)?;
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

/// Feature-gated scalar/SIMD oracle surface for same-binary benches and the
/// Quill gauntlet's internal differential suite.
#[cfg(feature = "bench-internals")]
pub mod differential {
    pub use super::bitpack::BitpackError;

    /// Stable fixture identifier reserved for gauntlet registration.
    pub const FIXTURE_ID: &str = "quiver-postings-bitpack-scalar-wide-v1";
    /// Normative source exercised by this differential.
    pub const SPEC_SECTION: &str = "FSLX v1 section 5.2 LSB-first bitpacked payloads";

    /// Pack values with the canonical scalar encoder.
    ///
    /// # Errors
    ///
    /// Returns a typed error for widths above 32, length overflow, or a value
    /// that cannot be represented at `width`.
    pub fn pack_values(values: &[u32], width: u8) -> Result<Vec<u8>, BitpackError> {
        super::bitpack::pack_lsb(values, width)
    }

    /// Decode into `output` with the scalar reference implementation.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid width, non-canonical padding, or a
    /// payload whose length does not exactly match `(output.len(), width)`.
    pub fn unpack_scalar_into(
        input: &[u8],
        width: u8,
        output: &mut [u32],
    ) -> Result<(), BitpackError> {
        super::bitpack::unpack_scalar_into(input, width, output)
    }

    /// Decode into `output` with the portable `wide::u32x8` implementation.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid width, non-canonical padding, or a
    /// payload whose length does not exactly match `(output.len(), width)`.
    pub fn unpack_wide_into(
        input: &[u8],
        width: u8,
        output: &mut [u32],
    ) -> Result<(), BitpackError> {
        super::bitpack::unpack_wide_into(input, width, output)
    }
}

#[cfg(test)]
mod tests;
