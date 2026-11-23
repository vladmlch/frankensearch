mod codec;
mod header;
mod index_io;
mod mutations;
mod owner;
mod types;

pub use codec::{next_generation, next_publication_nonce};
pub use mutations::VectorIndexWriter;
pub use owner::{ValidatedFsviBytes, ValidatedFsviRow, ValidatedFsviRowSource};
pub use types::{
    FSVI_MAGIC, FSVI_V2_IDENTITY_BINDING_SCHEMA, FSVI_V2_VERSION, FSVI_VERSION, FsviAdmissionError,
    FsviAnnAdmission, FsviInspection, FsviRecordFlags, FsviReindexReason, FsviReindexRequired,
    FsviSnapshotRejected, FsviSnapshotRejectionReason, FsviUpgradeRequired, FsviV2IdentityBinding,
    FsviV2IdentityMetadata, FsviV2Witness, Quantization, VacuumStats, VectorIndex, VectorMetadata,
};

#[cfg(unix)]
pub use codec::sync_parent_directory;
#[cfg(test)]
pub use codec::{align_up, crc32, reserve_f16_destination, validate_header_string};
pub use codec::{
    fingerprint_hex, fnv1a_hash, index_corrupted, is_tombstoned_flags, temporary_output_path,
    update_digest_domain,
};
#[cfg(test)]
pub use header::parse_complete_space_identity;
pub use header::{parse_header, parse_v2_header, read_exact_index_bytes, validate_v2_header_size};
#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
pub use owner::{PublishedFsviPathSnapshot, StableFileIdentity};
pub use owner::{
    open_readonly_noatime_nofollow, snapshot_parent_or_current, snapshot_rejected,
    stable_file_identity, validate_single_link_regular_file,
};
#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
pub use types::FsviMapLock;
#[cfg(any(feature = "ann", test))]
pub use types::SHA256_BYTES;
#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
pub use types::VectorIndexData;
pub use types::{FSVI_V1_MAX_HEADER_BYTES, ORDERED_DOCSET_DIGEST_DOMAIN};
#[cfg(test)]
pub use types::{
    FSVI_V2_BUNDLE_FINGERPRINT_OFFSET, FSVI_V2_DOCSET_DIGEST_OFFSET, FSVI_V2_FIXED_PREFIX_BYTES,
    FSVI_V2_GENERATION_FINGERPRINT_OFFSET, FSVI_V2_INPUT_FINGERPRINT_OFFSET,
    FSVI_V2_PRODUCER_FINGERPRINT_OFFSET, FSVI_V2_SPACE_FINGERPRINT_OFFSET,
    FSVI_V2_STORAGE_FINGERPRINT_OFFSET, FSVI_V2_VECTOR_DIGEST_OFFSET, RECORD_FLAG_TOMBSTONE,
    RECORD_SIZE_BYTES, VECTOR_ALIGN_BYTES, f16_destination_reservation_failure_is_pending,
    fail_next_f16_destination_reservation,
};
