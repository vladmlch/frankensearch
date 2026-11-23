use std::fmt;
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use frankensearch_core::generation::ArtifactGenerationIdentityV1;
use frankensearch_core::{SearchError, SearchResult};
use sha2::{Digest, Sha256};

use crate::search::ClassifiedHits;
use crate::wal::{self, WalConfig};
use crate::warmup::{self, WarmUpConfig, WarmUpStrategy};

#[cfg(unix)]
use super::codec::sync_parent_directory;
use super::codec::{
    fnv1a_hash, index_corrupted, is_tombstoned_flags, update_digest_domain,
    vector_content_digest_from_bytes,
};
use super::header::{parse_header, parse_v2_header};
use super::types::{
    FSVI_MAGIC, FSVI_V2_VERSION, FSVI_VERSION, FSVI_WITNESS_SCHEMA_V1, FsviAdmissionError,
    FsviAnnAdmission, FsviPublicationState, FsviRecordFlags, FsviReindexReason,
    FsviReindexRequired, FsviSnapshotRejected, FsviSnapshotRejectionReason, FsviUpgradeRequired,
    FsviV2IdentityBinding, FsviV2IdentityMetadata, FsviV2Witness, ORDERED_DOCSET_DIGEST_DOMAIN,
    Quantization, RECORD_FLAG_TOMBSTONE, RECORD_SIZE_BYTES, SHA256_BYTES, VECTOR_ALIGN_BYTES,
    VectorIndex, VectorIndexData, VectorMetadata,
};

/// Sealed owner of one fully validated immutable FSVI v2 byte image.
///
/// Construction copies pathname input into an owned [`Arc`] and then parses,
/// hashes, validates, searches, and serves rows exclusively from that exact
/// allocation. No file mapping or writable slice is retained. Replacing,
/// renaming, linking, or mutating the source pathname after construction cannot
/// alter this owner's search results.
///
/// The owner is safe for concurrent shared reads. A [`ValidatedFsviRowSource`]
/// borrows the owner, so Rust's lifetime rules prevent the row source from
/// outliving or becoming detached from the byte image it describes. The owner
/// intentionally exposes no conversion into a mutable/path-opened
/// [`VectorIndex`].
///
/// ```compile_fail
/// use frankensearch_index::ValidatedFsviBytes;
///
/// fn mutate(owner: &mut ValidatedFsviBytes) {
///     owner.append("forbidden", &[1.0, 0.0]);
/// }
/// ```
///
/// An owner is reachable only through admission against a validated
/// [`FsviV2IdentityBinding`]. None of the following shortcuts compile, and
/// each block below is a regression guard: if a later change adds the
/// corresponding public constructor, the block starts compiling and the
/// doctest fails.
///
/// Raw bytes alone cannot become an owner:
///
/// ```compile_fail
/// use frankensearch_index::ValidatedFsviBytes;
///
/// fn from_raw(bytes: &[u8]) -> ValidatedFsviBytes {
///     ValidatedFsviBytes::from_raw_bytes(bytes)
/// }
/// ```
///
/// The private fields cannot be assembled by hand:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use frankensearch_index::ValidatedFsviBytes;
///
/// fn assemble(bytes: Arc<[u8]>) -> ValidatedFsviBytes {
///     ValidatedFsviBytes { bytes, ..unreachable!() }
/// }
/// ```
///
/// A bare fingerprint string is not an identity binding:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use frankensearch_index::ValidatedFsviBytes;
///
/// fn from_fingerprint(bytes: Arc<[u8]>, fingerprint: &str) {
///     let _ = ValidatedFsviBytes::from_arc(bytes, fingerprint);
/// }
/// ```
///
/// Dimension-only metadata is not an identity binding either:
///
/// ```compile_fail
/// use frankensearch_index::FsviV2IdentityBinding;
///
/// fn dimension_only(dimension: usize) -> FsviV2IdentityBinding {
///     FsviV2IdentityBinding::new(dimension).expect("must not type-check")
/// }
/// ```
pub struct ValidatedFsviBytes {
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) index: VectorIndex,
    witness: FsviV2Witness,
    publication_state: FsviPublicationState,
}

impl fmt::Debug for ValidatedFsviBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedFsviBytes")
            .field("witness", &self.witness)
            .field("publication_state", &self.publication_state)
            .finish_non_exhaustive()
    }
}

/// Sealed read-only row source whose lifetime is tied to one
/// [`ValidatedFsviBytes`] owner.
///
/// ```compile_fail
/// use frankensearch_index::{ValidatedFsviBytes, ValidatedFsviRowSource};
///
/// fn detach<'a>(owner: ValidatedFsviBytes) -> ValidatedFsviRowSource<'a> {
///     owner.row_source()
/// }
/// ```
#[derive(Clone, Copy)]
pub struct ValidatedFsviRowSource<'owner> {
    owner: &'owner ValidatedFsviBytes,
}

impl fmt::Debug for ValidatedFsviRowSource<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedFsviRowSource")
            .field("witness", self.owner.witness())
            .finish_non_exhaustive()
    }
}

/// One immutable physical FSVI row borrowed from its sealed owner.
pub struct ValidatedFsviRow<'owner> {
    physical_index: usize,
    doc_id: &'owner str,
    vector_bytes: &'owner [u8],
    flags: FsviRecordFlags,
}

impl fmt::Debug for ValidatedFsviRow<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedFsviRow")
            .field("physical_index", &self.physical_index)
            .field("doc_id", &"<redacted>")
            .field("vector_bytes", &self.vector_bytes.len())
            .field("flags", &self.flags)
            .finish()
    }
}

impl ValidatedFsviRow<'_> {
    /// Physical row position in the exact owned image.
    #[must_use]
    pub const fn physical_index(&self) -> usize {
        self.physical_index
    }

    /// Borrow the validated UTF-8 document identifier.
    #[must_use]
    pub const fn doc_id(&self) -> &str {
        self.doc_id
    }

    /// Borrow the exact persisted vector bytes.
    #[must_use]
    pub const fn vector_bytes(&self) -> &[u8] {
        self.vector_bytes
    }

    /// Validated LIVE/TOMBSTONE state.
    #[must_use]
    pub const fn flags(&self) -> FsviRecordFlags {
        self.flags
    }
}

impl ValidatedFsviRowSource<'_> {
    /// Exact witness of the byte owner backing this row source.
    #[must_use]
    pub const fn witness(&self) -> &FsviV2Witness {
        self.owner.witness()
    }

    /// Persisted vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.owner.dimension()
    }

    /// Persisted storage quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.owner.quantization()
    }

    /// Number of physical rows, including tombstones.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.owner.record_count()
    }

    /// Number of searchable live rows.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.owner.live_count()
    }

    /// Number of retained tombstone rows.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        self.owner.tombstone_count()
    }

    /// Full-width immutable generation.
    #[must_use]
    pub const fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.owner.witness.generation
    }

    /// Ordered live-document-set digest.
    #[must_use]
    pub const fn ordered_live_docset_digest(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.ordered_live_docset_digest
    }

    /// Exact vector-content digest.
    #[must_use]
    pub const fn vector_content_digest(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.vector_content_digest
    }

    /// Mathematical embedding-space fingerprint.
    #[must_use]
    pub const fn space_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.space_fingerprint
    }

    /// Complete frozen embedding-bundle fingerprint.
    #[must_use]
    pub const fn identity_bundle_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.identity_bundle_fingerprint
    }

    /// Embedding producer-attestation fingerprint.
    #[must_use]
    pub const fn producer_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.producer_fingerprint
    }

    /// Outer embedding-input-contract fingerprint.
    #[must_use]
    pub const fn input_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.input_fingerprint
    }

    /// Physical storage fingerprint.
    #[must_use]
    pub const fn storage_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.storage_fingerprint
    }

    /// Full-width generation fingerprint.
    #[must_use]
    pub const fn generation_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.generation_fingerprint
    }

    /// Borrow one validated physical row.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an out-of-range row.
    pub fn row(&self, index: usize) -> SearchResult<ValidatedFsviRow<'_>> {
        self.owner.row(index)
    }
}

impl ValidatedFsviBytes {
    /// Admit an already owned immutable byte image.
    ///
    /// This constructor validates content and identity but does not claim that
    /// a pathname publication had no WAL sidecar. Use [`Self::open_published`]
    /// for published-generation admission.
    ///
    /// # Errors
    ///
    /// Returns typed legacy/upgrade outcomes, exact identity mismatch outcomes,
    /// or structural corruption.
    pub fn from_arc(
        bytes: Arc<[u8]>,
        expected: &FsviV2IdentityBinding,
    ) -> Result<Self, FsviAdmissionError> {
        Self::from_arc_with_state(bytes, expected, FsviPublicationState::OwnedBytesOnly)
    }

    /// Open a published pathname into one immutable owned image.
    ///
    /// Linux and Android use `O_NOATIME | O_NOFOLLOW | O_CLOEXEC`; other
    /// targets fail closed because literal timestamp preservation cannot be
    /// established with the available safe standard-library API. The final
    /// file must be a single-link regular inode. File identity, length, mode,
    /// link count, all timestamps, and containing-directory identity are
    /// compared before and after both the read and full semantic validation.
    ///
    /// # Errors
    ///
    /// Returns a typed snapshot rejection for unsafe path topology, mutation,
    /// unsupported no-atime operation, witness drift, or any adjacent WAL
    /// entry. Identity/version/content errors retain their normal typed forms.
    pub fn open_published(
        path: &Path,
        expected: &FsviV2IdentityBinding,
    ) -> Result<Self, FsviAdmissionError> {
        let snapshot = PublishedFsviPathSnapshot::read(path)?;
        let owner = Self::from_arc_with_state(
            Arc::clone(&snapshot.bytes),
            expected,
            FsviPublicationState::PublishedWalAbsent,
        )?;
        snapshot.verify()?;
        Ok(owner)
    }

    /// Reopen a published generation and require byte-for-byte witness equality.
    ///
    /// # Errors
    ///
    /// Returns [`FsviSnapshotRejectionReason::WitnessMismatch`] if the complete
    /// newly validated witness differs in any field.
    pub fn reopen_exact(
        path: &Path,
        expected: &FsviV2IdentityBinding,
        witness: &FsviV2Witness,
    ) -> Result<Self, FsviAdmissionError> {
        let owner = Self::open_published(path, expected)?;
        if owner.witness != *witness {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::WitnessMismatch,
                "reopened bytes differ from the complete expected witness",
            ));
        }
        Ok(owner)
    }

    /// Publish one completed sibling generation from its exact sealed bytes.
    ///
    /// The completed sibling is admitted before the destination is touched and
    /// remains in place as evidence. Publication copies only the admitted
    /// owner's exact bytes into a fresh same-directory file, syncs that file,
    /// atomically replaces the destination, and syncs the directory. Any stale
    /// destination WAL is removed and directory-synced only after the v2 main
    /// file is durable. Therefore a crash cannot leave the old main file with
    /// its committed WAL silently discarded; the only intermediate state is a
    /// v2 main file beside a WAL, which strict admission rejects fail-closed.
    ///
    /// The returned owner is an exact witness-checked reopen of the published
    /// pathname after WAL removal. No mutable [`VectorIndex`] is created or
    /// exposed for the v2 generation.
    ///
    /// # Errors
    ///
    /// Returns a typed admission error when the completed sibling is not an
    /// exact identity-complete v2 generation, the paths are not distinct
    /// siblings, publication or durability fails, a stale WAL cannot be
    /// removed, or the final bytes differ from the pre-publication witness.
    pub fn publish_completed_sibling(
        destination: &Path,
        completed_sibling: &Path,
        expected: &FsviV2IdentityBinding,
    ) -> Result<Self, FsviAdmissionError> {
        Self::publish_completed_sibling_with_hooks(
            destination,
            completed_sibling,
            expected,
            |_| Ok(()),
            || Ok(()),
            || Ok(()),
            || Ok(()),
        )
    }

    pub(crate) fn publish_completed_sibling_with_hooks<B, P, A, R>(
        destination: &Path,
        completed_sibling: &Path,
        expected: &FsviV2IdentityBinding,
        before_temp_admission: B,
        before_replace: P,
        after_main_sync: A,
        before_final_reopen: R,
    ) -> Result<Self, FsviAdmissionError>
    where
        B: FnOnce(&Path) -> SearchResult<()>,
        P: FnOnce() -> SearchResult<()>,
        A: FnOnce() -> SearchResult<()>,
        R: FnOnce() -> SearchResult<()>,
    {
        let destination_parent = snapshot_parent_or_current(destination);
        let completed_parent = snapshot_parent_or_current(completed_sibling);
        let destination_parent_metadata =
            fs::symlink_metadata(destination_parent).map_err(SearchError::Io)?;
        let completed_parent_metadata =
            fs::symlink_metadata(completed_parent).map_err(SearchError::Io)?;
        if !destination_parent_metadata.file_type().is_dir()
            || !completed_parent_metadata.file_type().is_dir()
        {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
                "FSVI v2 publication requires real source and destination parent directories",
            ));
        }
        #[cfg(unix)]
        let same_parent = {
            let destination_parent_identity = stable_file_identity(&destination_parent_metadata);
            let completed_parent_identity = stable_file_identity(&completed_parent_metadata);
            destination_parent_identity.device == completed_parent_identity.device
                && destination_parent_identity.inode == completed_parent_identity.inode
        };
        #[cfg(not(unix))]
        let same_parent = fs::canonicalize(destination_parent)
            .and_then(|destination| {
                fs::canonicalize(completed_parent).map(|completed| destination == completed)
            })
            .map_err(SearchError::Io)?;
        if !same_parent {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.completed_sibling".to_owned(),
                value: completed_sibling.display().to_string(),
                reason: "completed generation and destination must share one parent directory"
                    .to_owned(),
            }
            .into());
        }
        let destination_identity = match fs::symlink_metadata(destination) {
            Ok(metadata) => Some(validate_single_link_regular_file(&metadata)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(SearchError::Io(error).into()),
        };

        let completed_metadata_before =
            fs::symlink_metadata(completed_sibling).map_err(SearchError::Io)?;
        let completed_identity_before =
            validate_single_link_regular_file(&completed_metadata_before)?;
        let completed_owner = Self::open_published(completed_sibling, expected)?;
        let completed_metadata_after =
            fs::symlink_metadata(completed_sibling).map_err(SearchError::Io)?;
        let completed_identity_after =
            validate_single_link_regular_file(&completed_metadata_after)?;
        if completed_identity_before != completed_identity_after {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "completed sibling identity changed across sealed-byte admission",
            ));
        }
        #[cfg(unix)]
        let aliases_destination = destination_identity.is_some_and(|identity| {
            identity.device == completed_identity_after.device
                && identity.inode == completed_identity_after.inode
        });
        #[cfg(not(unix))]
        let aliases_destination = if destination_identity.is_some() {
            fs::canonicalize(destination)
                .and_then(|destination| {
                    fs::canonicalize(completed_sibling).map(|completed| destination == completed)
                })
                .map_err(SearchError::Io)?
        } else {
            false
        };
        if aliases_destination {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.completed_sibling".to_owned(),
                value: completed_sibling.display().to_string(),
                reason: "completed sibling and destination resolve to the same file".to_owned(),
            }
            .into());
        }

        let wal_path = wal::wal_path_for(destination);
        let aliases_destination_wal = match fs::symlink_metadata(&wal_path) {
            Ok(metadata) => {
                #[cfg(unix)]
                {
                    let identity = stable_file_identity(&metadata);
                    identity.device == completed_identity_after.device
                        && identity.inode == completed_identity_after.inode
                }
                #[cfg(not(unix))]
                {
                    !metadata.file_type().is_symlink()
                        && fs::canonicalize(&wal_path)
                            .and_then(|wal| {
                                fs::canonicalize(completed_sibling)
                                    .map(|completed| wal == completed)
                            })
                            .map_err(SearchError::Io)?
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(SearchError::Io(error).into()),
        };
        if aliases_destination_wal {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.completed_sibling".to_owned(),
                value: completed_sibling.display().to_string(),
                reason: "completed sibling resolves to the destination WAL pathname".to_owned(),
            }
            .into());
        }
        let witness = completed_owner.witness.clone();
        let completed_bytes = Arc::clone(&completed_owner.bytes);

        let mut temporary =
            tempfile::NamedTempFile::new_in(destination_parent).map_err(SearchError::Io)?;
        temporary
            .as_file_mut()
            .write_all(completed_bytes.as_ref())
            .map_err(SearchError::Io)?;
        temporary.as_file().sync_all().map_err(SearchError::Io)?;
        before_temp_admission(temporary.path())?;
        let _validated_temporary = Self::reopen_exact(temporary.path(), expected, &witness)?;
        before_replace()?;
        temporary.persist(destination).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.error.kind(),
                format!(
                    "failed to atomically publish sealed FSVI v2 bytes at '{}': {}",
                    destination.display(),
                    error.error
                ),
            ))
        })?;
        #[cfg(unix)]
        sync_parent_directory(destination)?;

        after_main_sync()?;

        let completed_metadata_before_wal =
            fs::symlink_metadata(completed_sibling).map_err(SearchError::Io)?;
        let completed_identity_before_wal =
            validate_single_link_regular_file(&completed_metadata_before_wal)?;
        if completed_identity_before_wal != completed_identity_after {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "completed sibling identity changed before destination WAL removal",
            ));
        }
        match fs::symlink_metadata(&wal_path) {
            Ok(_) => fs::remove_file(&wal_path).map_err(SearchError::Io)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(SearchError::Io(error).into()),
        }
        #[cfg(unix)]
        sync_parent_directory(&wal_path)?;

        before_final_reopen()?;
        Self::reopen_exact(destination, expected, &witness)
    }

    /// Redacted, serializable proof of the exact owned image.
    #[must_use]
    pub const fn witness(&self) -> &FsviV2Witness {
        &self.witness
    }

    /// Exact byte length retained by this owner.
    #[must_use]
    pub fn owned_byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether pathname construction proved WAL absence for this published
    /// generation. An owner created with [`Self::from_arc`] returns `false`.
    #[must_use]
    pub const fn published_wal_absent(&self) -> bool {
        matches!(
            self.publication_state,
            FsviPublicationState::PublishedWalAbsent
        )
    }

    /// Borrow a row source that cannot outlive this owner.
    #[must_use]
    pub const fn row_source(&self) -> ValidatedFsviRowSource<'_> {
        ValidatedFsviRowSource { owner: self }
    }

    /// Owner-bound native HNSW can build or load only against this exact owner.
    #[must_use]
    pub const fn ann_admission(&self) -> FsviAnnAdmission {
        FsviAnnAdmission::Enabled
    }

    /// Exact top-k search over the owned image.
    ///
    /// This delegates to the normal index search implementation while its
    /// backing store is the owner's exact [`Arc`] allocation, preventing search
    /// and witness logic from drifting onto different bytes.
    ///
    /// # Errors
    ///
    /// Returns the normal exact-search errors for a query dimension mismatch,
    /// non-finite query values, or structurally invalid row data.
    pub fn search_top_k(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn frankensearch_core::filter::SearchFilter>,
    ) -> SearchResult<Vec<frankensearch_core::VectorHit>> {
        self.index.search_top_k(query, limit, filter)
    }

    /// Exact top-k with typed zero-signal classification.
    ///
    /// # Errors
    ///
    /// Returns the normal classified-search errors for a query dimension
    /// mismatch, non-finite query values, or structurally invalid row data.
    pub fn search_top_k_classified(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn frankensearch_core::filter::SearchFilter>,
    ) -> SearchResult<ClassifiedHits> {
        self.index.search_top_k_classified(query, limit, filter)
    }

    /// Full parsed metadata.
    #[must_use]
    pub const fn metadata(&self) -> &VectorMetadata {
        self.index.metadata()
    }

    /// Validated logical model id retained for diagnostics only.
    #[must_use]
    pub fn embedder_id(&self) -> &str {
        self.index.embedder_id()
    }

    /// Validated immutable model revision retained for diagnostics only.
    #[must_use]
    pub fn embedder_revision(&self) -> &str {
        self.index.embedder_revision()
    }

    /// Always true for a successfully constructed sealed owner.
    #[must_use]
    pub const fn is_identity_admitted_v2(&self) -> bool {
        true
    }

    /// Complete v2 identity/content metadata.
    #[must_use]
    pub fn identity_v2(&self) -> &FsviV2IdentityMetadata {
        self.index
            .identity_v2()
            .expect("ValidatedFsviBytes construction requires FSVI v2 identity metadata")
    }

    /// Persisted vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.index.dimension()
    }

    /// Persisted storage quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.index.quantization()
    }

    /// Number of physical rows, including tombstones.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.index.record_count()
    }

    /// Number of searchable live rows.
    #[must_use]
    pub fn live_count(&self) -> usize {
        usize::try_from(self.witness.live_count).unwrap_or(usize::MAX)
    }

    /// Number of retained tombstone rows.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        usize::try_from(self.witness.tombstone_count).unwrap_or(usize::MAX)
    }

    /// Resolve a document id from the exact owned image.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an out-of-range row or
    /// [`SearchError::IndexCorrupted`] if its validated string range cannot be
    /// decoded.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.index.doc_id_at(index)
    }

    /// Decode a row vector as f32.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an out-of-range row or a
    /// decoding error if its persisted vector bytes are structurally invalid.
    pub fn vector_at_f32(&self, index: usize) -> SearchResult<Vec<f32>> {
        self.index.vector_at_f32(index)
    }

    /// Find the first physical row with the requested document hash.
    #[must_use]
    pub fn find_index_by_doc_hash(&self, doc_id_hash: u64) -> Option<usize> {
        self.index.find_index_by_doc_hash(doc_id_hash)
    }

    /// Borrow one validated physical row.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an out-of-range row or
    /// [`SearchError::IndexCorrupted`] if its validated document range cannot
    /// be decoded.
    pub fn row(&self, index: usize) -> SearchResult<ValidatedFsviRow<'_>> {
        let entry = self.index.record_at(index)?;
        Ok(ValidatedFsviRow {
            physical_index: index,
            doc_id: self.index.doc_id_at(index)?,
            vector_bytes: self.index.vector_bytes(index)?,
            flags: FsviRecordFlags(entry.flags),
        })
    }

    #[cfg(test)]
    pub(crate) fn owner_and_search_share_allocation(&self) -> bool {
        matches!(
            &self.index.data,
            VectorIndexData::Immutable(search_bytes) if Arc::ptr_eq(&self.bytes, search_bytes)
        )
    }

    fn from_arc_with_state(
        bytes: Arc<[u8]>,
        expected: &FsviV2IdentityBinding,
        publication_state: FsviPublicationState,
    ) -> Result<Self, FsviAdmissionError> {
        const OWNED_PATH: &str = "<owned-fsvi-v2>";
        let path = Path::new(OWNED_PATH);
        if bytes.len() < 6 {
            return Err(index_corrupted(path, "magic and version are truncated").into());
        }
        if bytes[..4] != FSVI_MAGIC {
            return Err(index_corrupted(path, "invalid FSVI magic bytes").into());
        }
        let version = u16::from_le_bytes(
            bytes[4..6]
                .try_into()
                .expect("owned FSVI version field has fixed width"),
        );
        match version {
            FSVI_VERSION => {
                let _ = parse_header(path, &bytes)?;
                return Err(FsviAdmissionError::ReindexRequired(
                    FsviReindexRequired {
                        reason: FsviReindexReason::LegacyUnidentified,
                        found_version: FSVI_VERSION,
                        detail: "FSVI v1 has no complete embedding-space, storage, content, or full-width generation identity; rebuild from source into a separate v2 generation"
                            .to_owned(),
                    },
                ));
            }
            FSVI_V2_VERSION => {}
            found if found > FSVI_V2_VERSION => {
                return Err(FsviAdmissionError::UpgradeRequired(FsviUpgradeRequired {
                    found_version: found,
                    supported_version: FSVI_V2_VERSION,
                }));
            }
            found => {
                return Err(index_corrupted(
                    path,
                    format!("unsupported historical FSVI schema version {found}"),
                )
                .into());
            }
        }

        let (metadata, header_len) = parse_v2_header(path, &bytes)?;
        validate_expected_v2_binding(path, &metadata, expected)?;
        let (records_offset, strings_offset, vectors_offset) =
            validate_v2_layout_len(path, &metadata, header_len, bytes.len())?;
        let content = validate_v2_records_and_content(
            path,
            &bytes,
            &metadata,
            records_offset,
            strings_offset,
            vectors_offset,
        )?;
        let identity = metadata
            .identity_v2
            .as_ref()
            .ok_or_else(|| index_corrupted(path, "v2 metadata omitted identity bindings"))?;
        let byte_len = u64::try_from(bytes.len())
            .map_err(|_| index_corrupted(path, "owned image length does not fit in u64"))?;
        let dimension = u32::try_from(metadata.dimension)
            .map_err(|_| index_corrupted(path, "dimension does not fit in u32"))?;
        let record_count = u64::try_from(metadata.record_count)
            .map_err(|_| index_corrupted(path, "record count does not fit in u64"))?;
        let witness = FsviV2Witness {
            schema_version: FSVI_WITNESS_SCHEMA_V1,
            fsvi_version: FSVI_V2_VERSION,
            byte_len,
            whole_image_sha256: Sha256::digest(&bytes).into(),
            generation: identity.generation,
            identity_bundle_fingerprint: identity.identity_bundle_fingerprint,
            space_fingerprint: identity.space_fingerprint,
            producer_fingerprint: identity.producer_fingerprint,
            input_fingerprint: identity.input_fingerprint,
            storage_fingerprint: identity.storage_fingerprint,
            generation_fingerprint: identity.generation_fingerprint,
            ordered_live_docset_digest: identity.ordered_live_docset_digest,
            vector_content_digest: identity.vector_content_digest,
            dimension,
            quantization: metadata.quantization,
            record_count,
            live_count: content.live_count,
            tombstone_count: content.tombstone_count,
        };

        let warm_up_config = WarmUpConfig::from_env();
        if !matches!(warm_up_config.strategy, WarmUpStrategy::None) {
            let _ = warmup::warm_up_bytes(&bytes, header_len, &warm_up_config, None);
        }

        let index = VectorIndex {
            path: PathBuf::from(OWNED_PATH),
            data: VectorIndexData::Immutable(Arc::clone(&bytes)),
            map_lock: None,
            metadata,
            records_offset,
            strings_offset,
            vectors_offset,
            wal_entries: Vec::new(),
            wal_config: WalConfig::default(),
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
        };
        Ok(Self {
            bytes,
            index,
            witness,
            publication_state,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StableFileIdentity {
    pub device: u64,
    pub inode: u64,
    pub mode: u32,
    pub hard_links: u64,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub modified_seconds: i64,
    pub modified_nanoseconds: i64,
    pub changed_seconds: i64,
    pub changed_nanoseconds: i64,
    pub accessed_seconds: i64,
    pub accessed_nanoseconds: i64,
}

#[cfg(unix)]
pub fn stable_file_identity(metadata: &fs::Metadata) -> StableFileIdentity {
    use std::os::unix::fs::MetadataExt;

    StableFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        mode: metadata.mode(),
        hard_links: metadata.nlink(),
        uid: metadata.uid(),
        gid: metadata.gid(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
        accessed_seconds: metadata.atime(),
        accessed_nanoseconds: metadata.atime_nsec(),
    }
}

#[cfg(not(unix))]
pub fn stable_file_identity(metadata: &fs::Metadata) -> StableFileIdentity {
    StableFileIdentity {
        device: 0,
        inode: 0,
        mode: 0,
        hard_links: 1,
        uid: 0,
        gid: 0,
        size: metadata.len(),
        modified_seconds: 0,
        modified_nanoseconds: 0,
        changed_seconds: 0,
        changed_nanoseconds: 0,
        accessed_seconds: 0,
        accessed_nanoseconds: 0,
    }
}

pub fn snapshot_rejected(
    reason: FsviSnapshotRejectionReason,
    detail: impl Into<String>,
) -> FsviAdmissionError {
    FsviAdmissionError::SnapshotRejected(FsviSnapshotRejected {
        reason,
        detail: detail.into(),
    })
}

pub fn validate_single_link_regular_file(
    metadata: &fs::Metadata,
) -> Result<StableFileIdentity, FsviAdmissionError> {
    if metadata.file_type().is_symlink() {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::SymbolicLink,
            "the final FSVI path must not be a symbolic link",
        ));
    }
    if !metadata.file_type().is_file() {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::NotRegularFile,
            "the final FSVI path must be a regular file",
        ));
    }
    let identity = stable_file_identity(metadata);
    if identity.hard_links != 1 {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::HardLinked,
            "the FSVI inode must have exactly one hard link",
        ));
    }
    Ok(identity)
}

fn ensure_published_wal_absent(path: &Path) -> Result<(), FsviAdmissionError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(snapshot_rejected(
            FsviSnapshotRejectionReason::PublishedWalPresent,
            "published immutable FSVI v2 generations require the WAL path to be absent",
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error).into()),
    }
}

pub fn snapshot_parent_or_current(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn open_readonly_noatime_nofollow(path: &Path) -> Result<File, FsviAdmissionError> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOATIME | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|error| {
            let raw_error = error.raw_os_error();
            if raw_error == Some(libc::EPERM)
                || raw_error == Some(libc::EINVAL)
                || raw_error == Some(libc::EOPNOTSUPP)
            {
                snapshot_rejected(
                    FsviSnapshotRejectionReason::NoAtimeUnsupported,
                    "O_NOATIME was denied or unsupported; admission will not weaken timestamp preservation",
                )
            } else if raw_error == Some(libc::ELOOP) {
                snapshot_rejected(
                    FsviSnapshotRejectionReason::PathChangedDuringRead,
                    "the FSVI path became a symbolic link while it was being opened",
                )
            } else {
                FsviAdmissionError::Index(SearchError::Io(error))
            }
        })
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn open_readonly_noatime_nofollow(_: &Path) -> Result<File, FsviAdmissionError> {
    Err(snapshot_rejected(
        FsviSnapshotRejectionReason::NoAtimeUnsupported,
        "this target has no supported safe O_NOATIME pathname-open implementation",
    ))
}

pub struct PublishedFsviPathSnapshot {
    path: PathBuf,
    wal_path: PathBuf,
    parent: PathBuf,
    opened_file: File,
    file_identity: StableFileIdentity,
    parent_identity: StableFileIdentity,
    bytes: Arc<[u8]>,
}

impl PublishedFsviPathSnapshot {
    pub fn read(path: &Path) -> Result<Self, FsviAdmissionError> {
        let path_metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(SearchError::IndexNotFound {
                    path: path.to_path_buf(),
                }
                .into());
            }
            Err(error) => return Err(SearchError::Io(error).into()),
        };
        let file_identity = validate_single_link_regular_file(&path_metadata)?;
        let parent = snapshot_parent_or_current(path).to_path_buf();
        let parent_metadata = fs::symlink_metadata(&parent).map_err(SearchError::Io)?;
        if !parent_metadata.file_type().is_dir() {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
                "the FSVI parent must be a real directory, not a symlink or special file",
            ));
        }
        let parent_identity = stable_file_identity(&parent_metadata);
        let wal_path = wal::wal_path_for(path);
        ensure_published_wal_absent(&wal_path)?;

        let mut opened_file = open_readonly_noatime_nofollow(path)?;
        let opened_identity =
            validate_single_link_regular_file(&opened_file.metadata().map_err(SearchError::Io)?)?;
        if opened_identity != file_identity {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the FSVI pathname and opened descriptor identify different immutable bytes",
            ));
        }
        let byte_len = usize::try_from(file_identity.size).map_err(|_| {
            index_corrupted(
                path,
                "FSVI byte length does not fit in this process address space",
            )
        })?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(byte_len)
            .map_err(|_| SearchError::InvalidConfig {
                field: "fsvi_snapshot.byte_len".to_owned(),
                value: byte_len.to_string(),
                reason: "unable to reserve the exact immutable byte image".to_owned(),
            })?;
        owned.resize(byte_len, 0);
        opened_file.read_exact(&mut owned).map_err(|error| {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                snapshot_rejected(
                    FsviSnapshotRejectionReason::PathChangedDuringRead,
                    "the FSVI inode was truncated while its byte image was being owned",
                )
            } else {
                FsviAdmissionError::Index(SearchError::Io(error))
            }
        })?;
        let mut trailing = [0_u8; 1];
        if opened_file.read(&mut trailing).map_err(SearchError::Io)? != 0 {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the FSVI inode grew while its byte image was being owned",
            ));
        }
        let snapshot = Self {
            path: path.to_path_buf(),
            wal_path,
            parent,
            opened_file,
            file_identity,
            parent_identity,
            bytes: Arc::from(owned),
        };
        snapshot.verify()?;
        Ok(snapshot)
    }

    pub fn verify(&self) -> Result<(), FsviAdmissionError> {
        let descriptor_metadata = self.opened_file.metadata().map_err(SearchError::Io)?;
        let descriptor_identity = validate_single_link_regular_file(&descriptor_metadata)?;
        let path_metadata = fs::symlink_metadata(&self.path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                snapshot_rejected(
                    FsviSnapshotRejectionReason::PathChangedDuringRead,
                    "the FSVI pathname disappeared while its byte image was being validated",
                )
            } else {
                FsviAdmissionError::Index(SearchError::Io(error))
            }
        })?;
        let path_identity = validate_single_link_regular_file(&path_metadata)?;
        if descriptor_identity != self.file_identity || path_identity != self.file_identity {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the FSVI inode identity, size, mode, links, or timestamps changed during admission",
            ));
        }
        ensure_published_wal_absent(&self.wal_path)?;
        let parent_metadata = fs::symlink_metadata(&self.parent).map_err(SearchError::Io)?;
        if !parent_metadata.file_type().is_dir()
            || stable_file_identity(&parent_metadata) != self.parent_identity
        {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
                "the FSVI containing directory changed during admission",
            ));
        }
        Ok(())
    }
}

fn validate_expected_v2_binding(
    path: &Path,
    metadata: &VectorMetadata,
    expected: &FsviV2IdentityBinding,
) -> Result<(), FsviAdmissionError> {
    let Some(actual) = metadata.identity_v2.as_ref() else {
        return Err(index_corrupted(path, "v2 metadata omitted identity bindings").into());
    };
    if actual.generation != expected.generation
        || actual.generation_fingerprint != expected.generation_fingerprint
    {
        return Err(FsviAdmissionError::ReindexRequired(
            FsviReindexRequired {
                reason: FsviReindexReason::GenerationMismatch,
                found_version: FSVI_V2_VERSION,
                detail: "persisted full-width artifact generation differs from the caller-owned generation; rebuild or select the exact published generation"
                    .to_owned(),
            },
        ));
    }
    let persisted_storage_bytes = &actual.storage_identity_canonical_bytes;
    let expected_storage_bytes = &expected.storage_canonical_bytes;
    let storage_mismatch = metadata.dimension != expected.dimension
        || metadata.quantization != expected.quantization
        || persisted_storage_bytes != expected_storage_bytes
        || actual.storage_fingerprint != expected.storage_fingerprint;
    if storage_mismatch {
        return Err(FsviAdmissionError::ReindexRequired(
            FsviReindexRequired {
                reason: FsviReindexReason::StorageMismatch,
                found_version: FSVI_V2_VERSION,
                detail: "persisted storage, dimension, or quantization differs from the caller-owned storage identity; source reindex is required"
                    .to_owned(),
            },
        ));
    }
    if actual.identity_bundle_canonical_bytes != expected.frozen_identity.canonical_bytes
        || actual.space_identity_canonical_bytes != expected.space_canonical_bytes
        || actual.identity_bundle_fingerprint != expected.bundle_fingerprint
        || actual.space_fingerprint != expected.space_fingerprint
        || actual.producer_fingerprint != expected.producer_fingerprint
        || actual.input_fingerprint != expected.input_fingerprint
    {
        return Err(FsviAdmissionError::ReindexRequired(
            FsviReindexRequired {
                reason: FsviReindexReason::IdentityMismatch,
                found_version: FSVI_V2_VERSION,
                detail: "persisted canonical embedding identity differs from the caller-owned identity; same names or dimensions must never be adopted or relabeled"
                    .to_owned(),
            },
        ));
    }
    Ok(())
}

fn validate_v2_layout_len(
    path: &Path,
    metadata: &VectorMetadata,
    header_len: usize,
    actual_len: usize,
) -> SearchResult<(usize, usize, usize)> {
    let records_bytes = metadata
        .record_count
        .checked_mul(RECORD_SIZE_BYTES)
        .ok_or_else(|| index_corrupted(path, "record table size overflow"))?;
    let records_offset = header_len;
    let strings_offset = records_offset
        .checked_add(records_bytes)
        .ok_or_else(|| index_corrupted(path, "record table offset overflow"))?;
    let vectors_offset = usize::try_from(metadata.vectors_offset)
        .map_err(|_| index_corrupted(path, "vectors_offset does not fit in usize"))?;
    if metadata.vectors_offset % VECTOR_ALIGN_BYTES != 0 {
        return Err(index_corrupted(
            path,
            format!(
                "v2 vectors_offset must be {VECTOR_ALIGN_BYTES}-byte aligned, found {}",
                metadata.vectors_offset
            ),
        ));
    }
    if vectors_offset < strings_offset {
        return Err(index_corrupted(
            path,
            "vectors_offset points inside the record table/string table region",
        ));
    }
    let vector_bytes = metadata
        .record_count
        .checked_mul(metadata.dimension)
        .and_then(|count| count.checked_mul(metadata.quantization.bytes_per_element()))
        .ok_or_else(|| index_corrupted(path, "vector slab size overflow"))?;
    let required_len = vectors_offset
        .checked_add(vector_bytes)
        .ok_or_else(|| index_corrupted(path, "vector slab end overflow"))?;
    if actual_len != required_len {
        return Err(index_corrupted(
            path,
            format!(
                "v2 file length must exactly match the bound layout: found {actual_len}, expected {required_len}"
            ),
        ));
    }
    Ok((records_offset, strings_offset, vectors_offset))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ValidatedV2ContentStats {
    live_count: u64,
    tombstone_count: u64,
}

fn validate_v2_records_and_content(
    path: &Path,
    data: &[u8],
    metadata: &VectorMetadata,
    records_offset: usize,
    strings_offset: usize,
    vectors_offset: usize,
) -> SearchResult<ValidatedV2ContentStats> {
    let identity = metadata
        .identity_v2
        .as_ref()
        .ok_or_else(|| index_corrupted(path, "v2 metadata omitted identity bindings"))?;
    let mut expected_string_offset = 0usize;
    let mut previous: Option<(u64, &str)> = None;
    let mut live_count = 0_u64;
    let mut tombstone_count = 0_u64;
    for index in 0..metadata.record_count {
        let offset =
            records_offset
                .checked_add(index.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
                    index_corrupted(path, "record offset multiplication overflow")
                })?)
                .ok_or_else(|| index_corrupted(path, "record offset overflow"))?;
        let end = offset
            .checked_add(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(path, "record end overflow"))?;
        if end > strings_offset || end > data.len() {
            return Err(index_corrupted(
                path,
                "v2 record table extends beyond its bound region",
            ));
        }
        let record = &data[offset..end];
        let doc_id_hash = u64::from_le_bytes(
            record[..8]
                .try_into()
                .map_err(|_| index_corrupted(path, "v2 record hash is truncated"))?,
        );
        let doc_id_offset = usize::try_from(u32::from_le_bytes(
            record[8..12]
                .try_into()
                .map_err(|_| index_corrupted(path, "v2 doc_id_offset is truncated"))?,
        ))
        .map_err(|_| index_corrupted(path, "v2 doc_id_offset does not fit in usize"))?;
        let doc_id_len = usize::from(u16::from_le_bytes(
            record[12..14]
                .try_into()
                .map_err(|_| index_corrupted(path, "v2 doc_id_len is truncated"))?,
        ));
        let flags = u16::from_le_bytes(
            record[14..16]
                .try_into()
                .map_err(|_| index_corrupted(path, "v2 record flags are truncated"))?,
        );
        if flags & !RECORD_FLAG_TOMBSTONE != 0 {
            return Err(index_corrupted(
                path,
                format!("v2 record {index} uses unsupported flags {flags:#06x}"),
            ));
        }
        if is_tombstoned_flags(flags) {
            tombstone_count = tombstone_count
                .checked_add(1)
                .ok_or_else(|| index_corrupted(path, "tombstone count overflow"))?;
        } else {
            live_count = live_count
                .checked_add(1)
                .ok_or_else(|| index_corrupted(path, "live count overflow"))?;
        }
        if doc_id_offset != expected_string_offset {
            return Err(index_corrupted(
                path,
                "v2 document strings must be contiguous in record order",
            ));
        }
        if doc_id_len == 0 {
            return Err(index_corrupted(path, "v2 document ids must not be empty"));
        }
        let doc_start = strings_offset
            .checked_add(doc_id_offset)
            .ok_or_else(|| index_corrupted(path, "v2 document id start overflow"))?;
        let doc_end = doc_start
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(path, "v2 document id end overflow"))?;
        if doc_end > vectors_offset || doc_end > data.len() {
            return Err(index_corrupted(
                path,
                "v2 document id extends beyond the bound string table",
            ));
        }
        let doc_id = std::str::from_utf8(&data[doc_start..doc_end]).map_err(|error| {
            index_corrupted(
                path,
                format!("v2 document id at record {index} is not UTF-8: {error}"),
            )
        })?;
        if fnv1a_hash(doc_id.as_bytes()) != doc_id_hash {
            return Err(index_corrupted(
                path,
                format!("v2 document hash mismatch at record {index}"),
            ));
        }
        if previous.is_some_and(|prior| prior >= (doc_id_hash, doc_id)) {
            return Err(index_corrupted(
                path,
                "v2 records must be strictly sorted with unique document ids",
            ));
        }
        previous = Some((doc_id_hash, doc_id));
        expected_string_offset = expected_string_offset
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(path, "v2 string table length overflow"))?;
    }
    let string_end = strings_offset
        .checked_add(expected_string_offset)
        .ok_or_else(|| index_corrupted(path, "v2 string table end overflow"))?;
    if string_end > vectors_offset {
        return Err(index_corrupted(
            path,
            "v2 string table overlaps the vector slab",
        ));
    }
    if data[string_end..vectors_offset]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(index_corrupted(
            path,
            "v2 alignment padding must be all zero",
        ));
    }
    let mut docset_hasher = Sha256::new();
    update_digest_domain(&mut docset_hasher, ORDERED_DOCSET_DIGEST_DOMAIN);
    docset_hasher.update(live_count.to_be_bytes());
    for index in 0..metadata.record_count {
        let record_offset = records_offset
            .checked_add(
                index
                    .checked_mul(RECORD_SIZE_BYTES)
                    .ok_or_else(|| index_corrupted(path, "record offset overflow"))?,
            )
            .ok_or_else(|| index_corrupted(path, "record offset overflow"))?;
        let record_end = record_offset
            .checked_add(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(path, "record end overflow"))?;
        let record = &data[record_offset..record_end];
        let flags = u16::from_le_bytes(
            record[14..16]
                .try_into()
                .map_err(|_| index_corrupted(path, "record flags are truncated"))?,
        );
        if is_tombstoned_flags(flags) {
            continue;
        }
        let doc_id_offset = usize::try_from(u32::from_le_bytes(
            record[8..12]
                .try_into()
                .map_err(|_| index_corrupted(path, "doc_id_offset is truncated"))?,
        ))
        .map_err(|_| index_corrupted(path, "doc_id_offset does not fit in usize"))?;
        let doc_id_len = usize::from(u16::from_le_bytes(
            record[12..14]
                .try_into()
                .map_err(|_| index_corrupted(path, "doc_id_len is truncated"))?,
        ));
        let doc_start = strings_offset
            .checked_add(doc_id_offset)
            .ok_or_else(|| index_corrupted(path, "document id start overflow"))?;
        let doc_end = doc_start
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(path, "document id end overflow"))?;
        docset_hasher.update(
            u64::try_from(doc_id_len)
                .map_err(|_| index_corrupted(path, "v2 doc_id length does not fit in u64"))?
                .to_be_bytes(),
        );
        docset_hasher.update(&data[doc_start..doc_end]);
    }
    let observed_docset_digest: [u8; SHA256_BYTES] = docset_hasher.finalize().into();
    if observed_docset_digest != identity.ordered_live_docset_digest {
        return Err(index_corrupted(
            path,
            "v2 ordered live-docset digest mismatch",
        ));
    }
    let observed_vector_digest = vector_content_digest_from_bytes(
        metadata.record_count,
        metadata.dimension,
        metadata.quantization,
        &data[vectors_offset..],
    )?;
    if observed_vector_digest != identity.vector_content_digest {
        return Err(index_corrupted(path, "v2 vector-content digest mismatch"));
    }
    Ok(ValidatedV2ContentStats {
        live_count,
        tombstone_count,
    })
}
