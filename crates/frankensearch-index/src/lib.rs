// Raised for the trait solver: proving Send for the nested async blocks here
// overflows the default limit on newer rustc. Not a defect in the code.
#![recursion_limit = "512"]

//! Vector index storage and loading for frankensearch.
//!
//! This crate implements the FSVI binary format reader/writer plus exact
//! brute-force top-k vector search, with optional HNSW ANN acceleration.
//!
//! # FSVI File Layout
//!
//! All multi-byte integers are little-endian. The vector slab is 64-byte
//! aligned for cache-line / SIMD friendliness.
//!
//! ```text
//! ┌───────────────────────────────────────────┐
//! │ Header (variable length)                  │
//! │   magic: b"FSVI"              (4 bytes)   │
//! │   version: u16                (2 bytes)   │
//! │   embedder_id_len: u16        (2 bytes)   │
//! │   embedder_id: [u8]           (variable)  │
//! │   embedder_revision_len: u16  (2 bytes)   │
//! │   embedder_revision: [u8]     (variable)  │
//! │   dimension: u32              (4 bytes)   │
//! │   quantization: u8            (1 byte)    │
//! │   reserved: [u8; 3]           (3 bytes)   │
//! │   record_count: u64           (8 bytes)   │
//! │   vectors_offset: u64         (8 bytes)   │
//! │   header_crc32: u32           (4 bytes)   │
//! ├───────────────────────────────────────────┤
//! │ Record Table                              │
//! │   record_count × 16 bytes each:           │
//! │     doc_id_hash: u64          (8 bytes)   │
//! │     doc_id_offset: u32        (4 bytes)   │
//! │     doc_id_len: u16           (2 bytes)   │
//! │     flags: u16                (2 bytes)   │
//! ├───────────────────────────────────────────┤
//! │ String Table                              │
//! │   Concatenated UTF-8 doc_id strings       │
//! ├───────────────────────────────────────────┤
//! │ Padding (to 64-byte alignment)            │
//! ├───────────────────────────────────────────┤
//! │ Vector Slab                               │
//! │   record_count × dimension × elem_size    │
//! │   (2 bytes/elem for f16, 4 for f32)       │
//! └───────────────────────────────────────────┘
//! ```

pub mod exact_component_adapters;
#[cfg(unix)]
pub mod fd_acl;
pub mod file_identity;
mod fsvi;
pub mod generation_root;
#[cfg(feature = "ann")]
pub mod hnsw;
pub mod in_memory;
pub mod mapped_file;
pub mod mrl;
pub mod native_hnsw;
pub mod quantization;
pub mod recall_certificate;
mod repro_soft_delete_rollback;
mod repro_wal_shadow_bug;
mod repro_wal_truncation;
pub mod search;
pub mod simd;
pub mod two_tier;
pub mod wal;
pub mod warmup;

#[cfg(test)]
mod tests;

pub use frankensearch_core::config::{ZERO_SIGNAL_SCHEMA_VERSION, ZeroSignalReason};
#[cfg(feature = "ann")]
pub use hnsw::{
    AnnFallbackReason, AnnSearchStats, HNSW_DEFAULT_EF_CONSTRUCTION, HNSW_DEFAULT_EF_SEARCH,
    HNSW_DEFAULT_M, HNSW_DEFAULT_MAX_LAYER, HnswConfig, HnswIndex, HnswLoadDisposition,
};
pub use in_memory::{InMemoryTwoTierIndex, InMemoryVectorIndex};
pub use mrl::{MrlConfig, MrlSearchStats};
pub use quantization::ScalarQuantizer;
pub use recall_certificate::{
    CertifiedEf, EfCalibration, calibrate_certified_ef, certified_min_ef, certified_min_ef_mean,
    conformal_recall_lower_bound, mean_recall_lower_bound, mean_recall_lower_bound_bernstein,
};
pub use search::{ClassifiedHits, PARALLEL_CHUNK_SIZE, PARALLEL_THRESHOLD, SearchParams};
pub use simd::{
    PreparedQuery4bit, cosine_similarity_f16, dot_4bit_prepared, dot_4bit_prepared_dynamic,
    dot_4bit_prepared_generic, dot_i8_i8, dot_i8_i8_generic, dot_i8_i8_maddubs,
    dot_i8x4_i8_maddubs, dot_packed_4bit, dot_product_f16_bytes_f32, dot_product_f16_bytes_f32_fma,
    dot_product_f16_bytes_f32_generic, dot_product_f16_f32, dot_product_f16_f32_generic,
    dot_product_f32_bytes_f32, dot_product_f32_bytes_f32_generic, dot_product_f32_f32,
    dot_product_f32_f32_generic, encode_f32_to_f16_extend, encode_f32_to_f16_extend_generic,
    maddubs_query_bias, pack_f16_le_bytes_to_4bit, pack_f16_le_bytes_to_4bit_explicit_round,
    pack_f16_le_bytes_to_4bit_generic, pack_f16_le_bytes_to_4bit_scalar_pack,
    pack_f16_slab_to_4bit, pack_f16_slab_to_4bit_generic, prepare_4bit_query,
    quantize_f16_le_bytes_to_i8, quantize_f16_le_bytes_to_i8_generic, quantize_f16_slab_to_i8,
    quantize_f16_slab_to_i8_generic,
};
pub use two_tier::{
    TwoTierIndex, TwoTierIndexBuilder, TwoTierIndexPaths, VECTOR_INDEX_FALLBACK_FILENAME,
    VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
};
#[cfg(feature = "ann")]
pub use two_tier::{VECTOR_ANN_FAST_FILENAME, VECTOR_ANN_QUALITY_FILENAME};
pub use wal::{
    CompactionStats, StrictWalImage, StrictWalInspection, WalConfig, inspect_wal_strict,
    wal_path_for,
};
pub use warmup::{AdaptiveConfig, HeatMap, WarmUpConfig, WarmUpResult, WarmUpStrategy};

pub use fsvi::{
    FSVI_MAGIC, FSVI_V2_IDENTITY_BINDING_SCHEMA, FSVI_V2_VERSION, FSVI_VERSION, FsviAdmissionError,
    FsviAnnAdmission, FsviInspection, FsviRecordFlags, FsviReindexReason, FsviReindexRequired,
    FsviSnapshotRejected, FsviSnapshotRejectionReason, FsviUpgradeRequired, FsviV2IdentityBinding,
    FsviV2IdentityMetadata, FsviV2Witness, Quantization, VacuumStats, ValidatedFsviBytes,
    ValidatedFsviRow, ValidatedFsviRowSource, VectorIndex, VectorIndexWriter, VectorMetadata,
    next_generation, next_publication_nonce,
};

#[cfg(any(feature = "ann", test))]
pub(crate) use fsvi::SHA256_BYTES;
#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
pub(crate) use fsvi::VectorIndexData;
#[cfg(unix)]
pub(crate) use fsvi::sync_parent_directory;
pub(crate) use fsvi::{
    FSVI_V1_MAX_HEADER_BYTES, ORDERED_DOCSET_DIGEST_DOMAIN, fingerprint_hex, fnv1a_hash,
    index_corrupted, is_tombstoned_flags, open_readonly_noatime_nofollow, parse_header,
    parse_v2_header, read_exact_index_bytes, snapshot_parent_or_current, snapshot_rejected,
    stable_file_identity, temporary_output_path, update_digest_domain,
    validate_single_link_regular_file, validate_v2_header_size,
};
#[cfg(test)]
pub(crate) use fsvi::{
    FSVI_V2_BUNDLE_FINGERPRINT_OFFSET, FSVI_V2_DOCSET_DIGEST_OFFSET, FSVI_V2_FIXED_PREFIX_BYTES,
    FSVI_V2_GENERATION_FINGERPRINT_OFFSET, FSVI_V2_INPUT_FINGERPRINT_OFFSET,
    FSVI_V2_PRODUCER_FINGERPRINT_OFFSET, FSVI_V2_SPACE_FINGERPRINT_OFFSET,
    FSVI_V2_STORAGE_FINGERPRINT_OFFSET, FSVI_V2_VECTOR_DIGEST_OFFSET, RECORD_FLAG_TOMBSTONE,
    RECORD_SIZE_BYTES, VECTOR_ALIGN_BYTES, align_up, crc32,
    f16_destination_reservation_failure_is_pending, fail_next_f16_destination_reservation,
    parse_complete_space_identity, reserve_f16_destination, validate_header_string,
};
#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
pub(crate) use fsvi::{FsviMapLock, PublishedFsviPathSnapshot, StableFileIdentity};
