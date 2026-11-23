use std::fmt;
use std::fs::File;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, FrozenEmbeddingIdentityBundleV1,
};
use frankensearch_core::{SearchError, SearchResult};
use memmap2::{Mmap, MmapMut};
use serde::{Deserialize, Serialize};

use crate::wal::{self, WalConfig};

use super::codec::{
    decode_sha256_fingerprint, fingerprint_hex, fsvi_v2_config_error, index_corrupted,
    quantization_from_identity,
};

/// Magic bytes at the start of every FSVI file.
pub const FSVI_MAGIC: [u8; 4] = *b"FSVI";

/// Supported FSVI format version.
pub const FSVI_VERSION: u16 = 1;

/// Identity-complete immutable FSVI format version.
pub const FSVI_V2_VERSION: u16 = 2;

/// Schema for the identity-binding envelope inside an FSVI v2 header.
pub const FSVI_V2_IDENTITY_BINDING_SCHEMA: u16 = 1;

pub const FSVI_V2_FIXED_PREFIX_BYTES: usize = 332;
pub(super) const FSVI_V2_MIN_HEADER_BYTES: usize = FSVI_V2_FIXED_PREFIX_BYTES + 4;
pub(super) const FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES: usize = 1024 * 1024;
pub(super) const FSVI_V2_MAX_HEADER_BYTES: usize =
    3 * FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES + FSVI_V2_MIN_HEADER_BYTES;
pub const FSVI_V1_MAX_HEADER_BYTES: usize = 4 + 2 + 2 + 65_535 + 2 + 65_535 + 4 + 1 + 3 + 8 + 8 + 4;
pub const SHA256_BYTES: usize = 32;
#[cfg(test)]
pub const FSVI_V2_BUNDLE_FINGERPRINT_OFFSET: usize = 76;
#[cfg(test)]
pub const FSVI_V2_SPACE_FINGERPRINT_OFFSET: usize = 108;
#[cfg(test)]
pub const FSVI_V2_PRODUCER_FINGERPRINT_OFFSET: usize = 140;
#[cfg(test)]
pub const FSVI_V2_INPUT_FINGERPRINT_OFFSET: usize = 172;
#[cfg(test)]
pub const FSVI_V2_STORAGE_FINGERPRINT_OFFSET: usize = 204;
#[cfg(test)]
pub const FSVI_V2_GENERATION_FINGERPRINT_OFFSET: usize = 236;
#[cfg(test)]
pub const FSVI_V2_DOCSET_DIGEST_OFFSET: usize = 268;
#[cfg(test)]
pub const FSVI_V2_VECTOR_DIGEST_OFFSET: usize = 300;
pub(super) const EMBEDDING_BUNDLE_CANONICAL_DOMAIN: &[u8] = b"frankensearch.embedding-bundle.v1";
pub(super) const EMBEDDING_SPACE_CANONICAL_DOMAIN: &[u8] = b"frankensearch.embedding-space.v1";
pub(super) const VECTOR_STORAGE_CANONICAL_DOMAIN: &[u8] = b"frankensearch.vector-storage.v1";
pub const ORDERED_DOCSET_DIGEST_DOMAIN: &[u8] = b"frankensearch.fsvi-v2.ordered-live-docset.v1";
pub(super) const VECTOR_CONTENT_DIGEST_DOMAIN: &[u8] = b"frankensearch.fsvi-v2.vector-content.v1";
pub(super) const FSVI_WITNESS_SCHEMA_V1: u16 = 1;

pub const RECORD_SIZE_BYTES: usize = 16;
pub const VECTOR_ALIGN_BYTES: u64 = 64;
pub const RECORD_FLAG_TOMBSTONE: u16 = 0x0001;
pub(super) const TOMBSTONE_VACUUM_THRESHOLD: f64 = 0.20;

#[cfg(test)]
std::thread_local! {
    /// Forces the next f16 decode-destination reservation to fail without
    /// exhausting the process allocator.
    static FAIL_NEXT_F16_DESTINATION_RESERVATION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub fn fail_next_f16_destination_reservation() {
    FAIL_NEXT_F16_DESTINATION_RESERVATION.with(|failure| failure.set(true));
}

#[cfg(test)]
pub(super) fn take_f16_destination_reservation_failure() -> bool {
    FAIL_NEXT_F16_DESTINATION_RESERVATION.with(|failure| failure.replace(false))
}

#[cfg(test)]
pub fn f16_destination_reservation_failure_is_pending() -> bool {
    FAIL_NEXT_F16_DESTINATION_RESERVATION.with(std::cell::Cell::get)
}

/// Vector element quantization stored in the FSVI slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Quantization {
    /// Full-precision float32.
    F32 = 0,
    /// Half-precision float16.
    F16 = 1,
}

/// Valid record-state bits in an identity-complete FSVI v2 record.
///
/// The all-zero value is a live row. Bit 0 is a retained tombstone. Every
/// other bit is reserved and rejected during admission, so callers never need
/// to guess how an unknown flag should affect search or ANN construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FsviRecordFlags(pub(super) u16);

impl FsviRecordFlags {
    /// A searchable live row.
    pub const LIVE: Self = Self(0);
    /// A retained, non-searchable tombstone row.
    pub const TOMBSTONE: Self = Self(RECORD_FLAG_TOMBSTONE);

    /// Decode a validated raw FSVI v2 flag word.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] if any reserved bit is set.
    pub fn from_bits(bits: u16) -> SearchResult<Self> {
        if bits & !RECORD_FLAG_TOMBSTONE != 0 {
            return Err(index_corrupted(
                Path::new("<owned-fsvi-v2>"),
                format!("unsupported FSVI v2 record flags {bits:#06x}"),
            ));
        }
        Ok(Self(bits))
    }

    /// Exact on-disk bit representation.
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Whether this row is a retained tombstone.
    #[must_use]
    pub const fn is_tombstone(self) -> bool {
        self.0 & RECORD_FLAG_TOMBSTONE != 0
    }

    /// Whether this row participates in exact search and the ordered live
    /// document-set digest.
    #[must_use]
    pub const fn is_live(self) -> bool {
        !self.is_tombstone()
    }
}

impl Quantization {
    pub(crate) fn from_wire(value: u8, path: &Path) -> SearchResult<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            _ => Err(index_corrupted(
                path,
                format!("unsupported quantization byte: {value}"),
            )),
        }
    }

    pub(crate) const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Exact caller-owned identity required to write or admit one FSVI v2 artifact.
///
/// Construction validates the complete frozen embedding identity, requires
/// canonical `fsvi-v2` little-endian storage, and binds a full-width immutable
/// generation. Human-readable model names and vector dimensions are never
/// sufficient to construct this value.
#[derive(Clone, PartialEq, Eq)]
pub struct FsviV2IdentityBinding {
    pub(super) generation: ArtifactGenerationIdentityV1,
    pub(super) frozen_identity: FrozenEmbeddingIdentityBundleV1,
    pub(super) space_canonical_bytes: Vec<u8>,
    pub(super) storage_canonical_bytes: Vec<u8>,
    pub(super) bundle_fingerprint: [u8; SHA256_BYTES],
    pub(super) space_fingerprint: [u8; SHA256_BYTES],
    pub(super) producer_fingerprint: [u8; SHA256_BYTES],
    pub(super) input_fingerprint: [u8; SHA256_BYTES],
    pub(super) storage_fingerprint: [u8; SHA256_BYTES],
    pub(super) generation_fingerprint: [u8; SHA256_BYTES],
    pub(super) dimension: usize,
    pub(super) quantization: Quantization,
}

impl fmt::Debug for FsviV2IdentityBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FsviV2IdentityBinding")
            .field("generation", &self.generation)
            .field("dimension", &self.dimension)
            .field("quantization", &self.quantization)
            .field(
                "bundle_fingerprint",
                &fingerprint_hex(&self.bundle_fingerprint),
            )
            .field(
                "canonical_bundle_bytes",
                &self.frozen_identity.canonical_bytes.len(),
            )
            .field("canonical_space_bytes", &self.space_canonical_bytes.len())
            .field(
                "canonical_storage_bytes",
                &self.storage_canonical_bytes.len(),
            )
            .finish_non_exhaustive()
    }
}

impl FsviV2IdentityBinding {
    /// Validate and freeze an exact FSVI v2 generation/storage binding.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the generation or frozen
    /// identity is invalid, storage is not exactly `fsvi-v2` little-endian,
    /// quantization is not F32/F16, or canonical identity bytes exceed the
    /// bounded header allowance.
    pub fn new(
        generation: ArtifactGenerationIdentityV1,
        frozen_identity: FrozenEmbeddingIdentityBundleV1,
    ) -> SearchResult<Self> {
        generation.validate()?;
        frozen_identity.validate()?;
        let storage = &frozen_identity.identity.storage;
        if storage.format != "fsvi-v2" {
            return Err(fsvi_v2_config_error(
                "storage.format",
                "must be exactly fsvi-v2",
            ));
        }
        if storage.endianness != "little-endian" {
            return Err(fsvi_v2_config_error(
                "storage.endianness",
                "must be exactly little-endian",
            ));
        }
        let quantization = quantization_from_identity(storage.quantization)?;
        let dimension = usize::try_from(storage.dimension)
            .map_err(|_| fsvi_v2_config_error("storage.dimension", "must fit in usize"))?;
        let space_canonical_bytes = frozen_identity.identity.space.canonical_bytes();
        let storage_canonical_bytes = storage.canonical_bytes();
        for (field, bytes) in [
            (
                "identity.canonical_bytes",
                frozen_identity.canonical_bytes.as_slice(),
            ),
            ("space.canonical_bytes", space_canonical_bytes.as_slice()),
            (
                "storage.canonical_bytes",
                storage_canonical_bytes.as_slice(),
            ),
        ] {
            if bytes.is_empty() || bytes.len() > FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES {
                return Err(fsvi_v2_config_error(
                    field,
                    "must be non-empty and no larger than 1 MiB",
                ));
            }
        }

        let bundle_fingerprint =
            decode_sha256_fingerprint("identity.fingerprint", &frozen_identity.fingerprint)?;
        let space_fingerprint = decode_sha256_fingerprint(
            "space.fingerprint",
            &frozen_identity.identity.space.fingerprint(),
        )?;
        let producer_fingerprint = decode_sha256_fingerprint(
            "producer.fingerprint",
            &frozen_identity.identity.producer.fingerprint(),
        )?;
        let input_fingerprint = decode_sha256_fingerprint(
            "input.fingerprint",
            &frozen_identity.identity.input.fingerprint(),
        )?;
        let storage_fingerprint =
            decode_sha256_fingerprint("storage.fingerprint", &storage.fingerprint())?;
        let generation_fingerprint =
            decode_sha256_fingerprint("generation.fingerprint", &generation.fingerprint())?;

        Ok(Self {
            generation,
            frozen_identity,
            space_canonical_bytes,
            storage_canonical_bytes,
            bundle_fingerprint,
            space_fingerprint,
            producer_fingerprint,
            input_fingerprint,
            storage_fingerprint,
            generation_fingerprint,
            dimension,
            quantization,
        })
    }

    /// Immutable artifact generation bound to this writer/reader.
    #[must_use]
    pub const fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.generation
    }

    /// Complete validated frozen embedding identity.
    #[must_use]
    pub const fn frozen_identity(&self) -> &FrozenEmbeddingIdentityBundleV1 {
        &self.frozen_identity
    }

    /// Stored vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Stored vector quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }
}

/// Identity and content bindings decoded from an FSVI v2 header.
#[derive(Clone, PartialEq, Eq)]
pub struct FsviV2IdentityMetadata {
    /// Full-width immutable artifact generation.
    pub generation: ArtifactGenerationIdentityV1,
    /// Exact canonical bytes of the complete embedding identity bundle.
    pub identity_bundle_canonical_bytes: Vec<u8>,
    /// Exact canonical bytes of the mathematical embedding-space identity.
    pub space_identity_canonical_bytes: Vec<u8>,
    /// Exact canonical bytes of the physical storage identity.
    pub storage_identity_canonical_bytes: Vec<u8>,
    /// SHA-256 of the complete identity bundle canonical bytes.
    pub identity_bundle_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the mathematical embedding space.
    pub space_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the embedding producer attestation.
    pub producer_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the outer embedding input contract.
    pub input_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the physical vector storage identity.
    pub storage_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the canonical full-width generation identity.
    pub generation_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the ordered live document identifiers.
    pub ordered_live_docset_digest: [u8; SHA256_BYTES],
    /// SHA-256 of the exact persisted vector slab bytes and shape.
    pub vector_content_digest: [u8; SHA256_BYTES],
    /// Exact byte length of the complete v2 header, including CRC.
    pub header_size: usize,
}

impl fmt::Debug for FsviV2IdentityMetadata {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FsviV2IdentityMetadata")
            .field("generation", &self.generation)
            .field(
                "identity_bundle_fingerprint",
                &fingerprint_hex(&self.identity_bundle_fingerprint),
            )
            .field(
                "space_fingerprint",
                &fingerprint_hex(&self.space_fingerprint),
            )
            .field(
                "storage_fingerprint",
                &fingerprint_hex(&self.storage_fingerprint),
            )
            .field(
                "ordered_live_docset_digest",
                &fingerprint_hex(&self.ordered_live_docset_digest),
            )
            .field(
                "vector_content_digest",
                &fingerprint_hex(&self.vector_content_digest),
            )
            .field("header_size", &self.header_size)
            .finish_non_exhaustive()
    }
}

/// Serializable, redacted proof of one fully validated immutable FSVI v2 byte
/// image.
///
/// The witness contains no document identifiers or pathnames. Its whole-image
/// SHA-256 covers every persisted byte, while the remaining fields make the
/// semantic, storage, generation, membership, and vector-content bindings
/// directly auditable. Equality is exact and is the only supported way to
/// authorize a reopen of an already witnessed generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FsviV2Witness {
    /// Witness schema version.
    pub schema_version: u16,
    /// FSVI format version covered by this witness.
    pub fsvi_version: u16,
    /// Exact byte length of the owned image.
    pub byte_len: u64,
    /// SHA-256 of every byte in the owned image.
    pub whole_image_sha256: [u8; SHA256_BYTES],
    /// Full-width immutable generation.
    pub generation: ArtifactGenerationIdentityV1,
    /// Complete frozen embedding-bundle fingerprint.
    pub identity_bundle_fingerprint: [u8; SHA256_BYTES],
    /// Mathematical embedding-space fingerprint.
    pub space_fingerprint: [u8; SHA256_BYTES],
    /// Producer-attestation fingerprint.
    pub producer_fingerprint: [u8; SHA256_BYTES],
    /// Outer embedding-input-contract fingerprint.
    pub input_fingerprint: [u8; SHA256_BYTES],
    /// Physical vector-storage fingerprint.
    pub storage_fingerprint: [u8; SHA256_BYTES],
    /// Full-width generation fingerprint.
    pub generation_fingerprint: [u8; SHA256_BYTES],
    /// Ordered live-document-set digest from the validated header.
    pub ordered_live_docset_digest: [u8; SHA256_BYTES],
    /// Exact vector-slab digest from the validated header.
    pub vector_content_digest: [u8; SHA256_BYTES],
    /// Persisted vector dimension.
    pub dimension: u32,
    /// Persisted vector quantization.
    pub quantization: Quantization,
    /// Number of physical rows, including tombstones.
    pub record_count: u64,
    /// Number of searchable live rows.
    pub live_count: u64,
    /// Number of retained tombstone rows.
    pub tombstone_count: u64,
}

/// Machine-matchable reason an immutable pathname snapshot was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsviSnapshotRejectionReason {
    /// This target cannot provide a safe no-atime open and therefore cannot
    /// prove literal side-effect freedom.
    NoAtimeUnsupported,
    /// The final index path is a symbolic link.
    SymbolicLink,
    /// The final index path is not a regular file.
    NotRegularFile,
    /// The inode has more than one hard link and can be mutated through an
    /// unobserved alias.
    HardLinked,
    /// The file identity or metadata changed while bytes were being owned.
    PathChangedDuringRead,
    /// The containing directory changed while the publication snapshot was
    /// being owned.
    DirectoryChangedDuringRead,
    /// Any WAL directory entry exists beside a purported published FSVI v2
    /// generation, including an empty or otherwise valid sidecar.
    PublishedWalPresent,
    /// A reopen produced a different complete witness.
    WitnessMismatch,
}

/// Typed pathname/publication rejection without document identifiers or
/// unbounded path-derived diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FsviSnapshotRejected {
    /// Stable reason for programmatic recovery routing.
    pub reason: FsviSnapshotRejectionReason,
    /// Bounded redacted diagnostic.
    pub detail: String,
}

/// Typed ANN disposition for one immutable FSVI owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum FsviAnnAdmission {
    /// [`crate::native_hnsw::ValidatedNativeHnsw`] builds and loads only from
    /// this retained owner and binds the graph receipt to its exact witness.
    Enabled,
}

/// Why a recognized artifact must be rebuilt rather than adopted or relabeled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsviReindexReason {
    /// FSVI v1 carries no complete embedding/storage/generation identity.
    LegacyUnidentified,
    /// Persisted embedding identity differs from the caller's exact identity.
    IdentityMismatch,
    /// Persisted immutable generation differs from the caller's generation.
    GenerationMismatch,
    /// Persisted physical storage contract differs from the caller's contract.
    StorageMismatch,
}

/// Typed, actionable outcome for a recognized artifact that cannot be admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsviReindexRequired {
    /// Machine-matchable reason.
    pub reason: FsviReindexReason,
    /// Format version found in the artifact.
    pub found_version: u16,
    /// Bounded diagnostic that never suggests relabeling/adoption.
    pub detail: String,
}

/// Typed outcome for an artifact produced by a newer FSVI schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsviUpgradeRequired {
    /// Newer format version found in the artifact.
    pub found_version: u16,
    /// Newest format version this build can inspect and admit.
    pub supported_version: u16,
}

/// Non-mutating format inspection result.
///
/// Structural corruption remains [`SearchError::IndexCorrupted`], so callers
/// cannot confuse damaged bytes with either a source reindex or software
/// upgrade action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsviInspection {
    /// Identity-complete v2 header; full content admission still recomputes
    /// docset/vector digests in [`VectorIndex::open_admitted_v2`].
    V2IdentityComplete(Box<VectorMetadata>),
    /// Recognized legacy bytes that must be rebuilt from source.
    ReindexRequired(FsviReindexRequired),
    /// A newer schema that requires newer reader software.
    UpgradeRequired(FsviUpgradeRequired),
}

/// Error returned by exact FSVI v2 admission.
#[derive(Debug)]
pub enum FsviAdmissionError {
    /// Recognized bytes require a source reindex.
    ReindexRequired(FsviReindexRequired),
    /// Newer bytes require newer reader software.
    UpgradeRequired(FsviUpgradeRequired),
    /// The path/publication snapshot could not be proven immutable and
    /// side-effect-free.
    SnapshotRejected(FsviSnapshotRejected),
    /// I/O, not-found, or actual corruption from the index layer.
    Index(SearchError),
}

impl fmt::Display for FsviAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReindexRequired(required) => write!(
                formatter,
                "FSVI source reindex required ({:?}): {}",
                required.reason, required.detail
            ),
            Self::UpgradeRequired(required) => write!(
                formatter,
                "FSVI reader upgrade required: found v{}, supported through v{}",
                required.found_version, required.supported_version
            ),
            Self::SnapshotRejected(rejected) => write!(
                formatter,
                "FSVI immutable snapshot rejected ({:?}): {}",
                rejected.reason, rejected.detail
            ),
            Self::Index(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FsviAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Index(error) => Some(error),
            Self::ReindexRequired(_) | Self::UpgradeRequired(_) | Self::SnapshotRejected(_) => None,
        }
    }
}

impl From<SearchError> for FsviAdmissionError {
    fn from(error: SearchError) -> Self {
        Self::Index(error)
    }
}

/// Parsed metadata from an FSVI file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorMetadata {
    /// On-disk FSVI format version.
    pub fsvi_version: u16,
    /// Stable embedder id used to build the index.
    pub embedder_id: String,
    /// Model revision identifier (e.g. pinned commit hash).
    pub embedder_revision: String,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Stored quantization.
    pub quantization: Quantization,
    /// Compaction generation counter (0-255) used for stale WAL detection.
    pub compaction_gen: u8,
    /// Shared publication identity of a multi-tier install (bd-miio8).
    ///
    /// Every tier published by one `TwoTierIndexBuilder::finish` carries the
    /// same non-zero nonce; a mismatch at open time means the pair is a
    /// mixed-generation survivor of a crash between tier installs. Zero is
    /// the legacy value (pre-nonce files and single-tier writers).
    pub publication_nonce: u16,
    /// Number of records in the index.
    pub record_count: usize,
    /// Byte offset to the aligned vector slab.
    pub vectors_offset: u64,
    /// Complete v2 identity/content bindings, absent for legacy v1 bytes.
    pub identity_v2: Option<FsviV2IdentityMetadata>,
}

/// Statistics returned by [`VectorIndex::vacuum`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VacuumStats {
    /// Records in the main index before vacuum.
    pub records_before: usize,
    /// Records in the main index after vacuum.
    pub records_after: usize,
    /// Tombstoned records removed by vacuum.
    pub tombstones_removed: usize,
    /// Approximate number of bytes reclaimed in the main index file.
    pub bytes_reclaimed: usize,
    /// Time taken by the vacuum operation.
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy)]
pub struct RecordEntry {
    pub doc_id_hash: u64,
    pub doc_id_offset: u32,
    pub doc_id_len: u16,
    pub flags: u16,
}

#[derive(Debug)]
pub enum VectorIndexData {
    Mutable(MmapMut),
    ReadOnly(Mmap),
    Immutable(Arc<[u8]>),
}

impl VectorIndexData {
    pub(super) fn write_and_flush(&mut self, offset: usize, bytes: &[u8]) -> SearchResult<()> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| index_corrupted(Path::new("<fsvi>"), "write range overflow"))?;
        match self {
            Self::Mutable(mapping) => {
                if end > mapping.len() {
                    return Err(index_corrupted(
                        Path::new("<fsvi>"),
                        "write range extends beyond mapped data",
                    ));
                }
                mapping[offset..end].copy_from_slice(bytes);
                mapping
                    .flush_range(offset, bytes.len())
                    .map_err(SearchError::Io)
            }
            Self::ReadOnly(_) => Err(SearchError::InvalidConfig {
                field: "fsvi.writer_lock".to_owned(),
                value: "read-only mapping".to_owned(),
                reason: "mutating a published FSVI requires VectorIndex::open_writer and its retained exclusive writer lock".to_owned(),
            }),
            Self::Immutable(_) => Err(SearchError::InvalidConfig {
                field: "fsvi_v2.mutation".to_owned(),
                value: "immutable-owned-image".to_owned(),
                reason: "FSVI v2 artifacts are immutable; create a new generation instead"
                    .to_owned(),
            }),
        }
    }
}

impl Deref for VectorIndexData {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mutable(mapping) => mapping,
            Self::ReadOnly(mapping) => mapping,
            Self::Immutable(bytes) => bytes,
        }
    }
}

/// The OS lock retained for the complete lifetime of an FSVI mapping.
///
/// A shared lock accompanies every read-only mapping; a mutable mapping is
/// created only after acquiring and retaining the exclusive counterpart. The
/// retained descriptor is load-bearing: dropping it while the mapping remains
/// live would reopen the cross-process mutation race that `memmap2` marks
/// unsafe.
///
/// Creator-process drop explicitly unlocks after the mapping is destroyed, so
/// an incidental descriptor inherited across fork/exec cannot prolong this
/// owner's lock. A child-side drop only closes its inherited descriptor. This
/// does not make continued use of an inherited mapping in a fork child safe;
/// it covers creator-owned mappings and children that close or exec.
#[derive(Debug)]
pub enum FsviMapLock {
    Shared {
        file: File,
        path: PathBuf,
        acquired: Instant,
        creator_pid: u32,
    },
    Exclusive {
        file: File,
        path: PathBuf,
        acquired: Instant,
        creator_pid: u32,
    },
}

impl FsviMapLock {
    pub(super) const fn is_writer(&self) -> bool {
        matches!(self, Self::Exclusive { .. })
    }

    pub(super) const fn mode(&self) -> &'static str {
        if self.is_writer() {
            "exclusive"
        } else {
            "shared"
        }
    }

    fn path(&self) -> &Path {
        match self {
            Self::Shared { path, .. } | Self::Exclusive { path, .. } => path,
        }
    }

    fn acquired(&self) -> Instant {
        match self {
            Self::Shared { acquired, .. } | Self::Exclusive { acquired, .. } => *acquired,
        }
    }
}

/// Lock lifecycle tracing (`RUST_LOG=frankensearch_index::map_lock=debug`):
/// a refused `fsvi.map_lock` is only diagnosable when the holder that is
/// still alive can be named, so every acquisition and release is logged with
/// its mode, hold time, and thread.
impl Drop for FsviMapLock {
    fn drop(&mut self) {
        let (file, creator_pid) = match self {
            Self::Shared {
                file, creator_pid, ..
            }
            | Self::Exclusive {
                file, creator_pid, ..
            } => (file, *creator_pid),
        };
        if creator_pid != std::process::id() {
            // An inherited descriptor shares the creator's lock. Even an
            // explicit unlock from the child would release the creator's
            // still-live mapping, so child drop only closes its descriptor.
            return;
        }
        if let Err(error) = file.unlock() {
            tracing::warn!(
                target: "frankensearch_index::map_lock",
                path = %self.path().display(),
                mode = self.mode(),
                %error,
                "fsvi map lock explicit unlock failed; closing its descriptor"
            );
            return;
        }
        tracing::debug!(
            target: "frankensearch_index::map_lock",
            path = %self.path().display(),
            mode = self.mode(),
            held_ms = self.acquired().elapsed().as_millis(),
            thread = ?std::thread::current().id(),
            "fsvi map lock released"
        );
    }
}

#[derive(Debug)]
pub struct VectorIndex {
    pub(crate) path: PathBuf,
    // Declaration order is load-bearing: destroy the mapping before the
    // following guard explicitly releases its lock. Do not reorder these.
    pub(crate) data: VectorIndexData,
    pub(crate) map_lock: Option<FsviMapLock>,
    pub(crate) metadata: VectorMetadata,
    pub(crate) records_offset: usize,
    pub(crate) strings_offset: usize,
    pub(crate) vectors_offset: usize,
    /// WAL entries for incremental updates (empty if no WAL exists).
    pub(crate) wal_entries: Vec<wal::WalEntry>,
    /// WAL configuration.
    pub(super) wal_config: WalConfig,
    /// Lazily-built int8 quantization of the (contiguous, F16) main vector region,
    /// for the optional int8 two-pass scan (`search_top_k_int8_two_pass`). Built on
    /// first two-pass use, so exact-only callers never pay the quantization cost or
    /// its footprint. One corpus-wide max-abs scale (see `quantize_f16_bytes_to_i8`).
    pub(crate) vectors_i8: OnceLock<Vec<i8>>,
    /// Lazily-built packed signed-4-bit quantization of the (contiguous, F16) main
    /// vector region — 2 dims/byte, `dim.div_ceil(2)` bytes per vector (half the int8
    /// slab) — for the optional 4-bit two-pass scan (`search_top_k_4bit_two_pass`).
    /// 16 levels stay lossless at mult≈5 on realistic data while halving pass-1
    /// bandwidth vs int8. Built on first 4-bit-two-pass use; other callers never pay.
    pub(crate) vectors_nibbles: OnceLock<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FsviPublicationState {
    OwnedBytesOnly,
    PublishedWalAbsent,
}
