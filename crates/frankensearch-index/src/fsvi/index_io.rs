use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime};

use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use memmap2::{Mmap, MmapMut};
use tracing::debug;

use crate::simd::{dot_product_f16_bytes_f32, dot_product_f32_bytes_f32, dot_product_f32_f32};
use crate::wal::{self, WalConfig};
use crate::warmup::{self, WarmUpConfig, WarmUpStrategy};

#[cfg(unix)]
use super::codec::sync_parent_directory;
use super::codec::{
    fnv1a_hash, index_corrupted, is_tombstoned_flags, map_lock_error, next_generation,
    reserve_f16_destination, temporary_output_path, validate_header_string,
};
use super::header::{parse_header, parse_v2_header, read_header_for_inspection};
use super::mutations::VectorIndexWriter;
use super::owner::ValidatedFsviBytes;
use super::types::{
    FSVI_V2_VERSION, FSVI_VERSION, FsviAdmissionError, FsviInspection, FsviMapLock,
    FsviReindexReason, FsviReindexRequired, FsviUpgradeRequired, FsviV2IdentityBinding,
    FsviV2IdentityMetadata, Quantization, RECORD_SIZE_BYTES, RecordEntry, VectorIndex,
    VectorIndexData, VectorMetadata,
};

impl VectorIndex {
    /// Inspect an FSVI header without adopting, relabeling, or mutating it.
    ///
    /// Valid v1 bytes are always reported as
    /// [`FsviInspection::ReindexRequired`] with
    /// [`FsviReindexReason::LegacyUnidentified`]. A future schema is
    /// [`FsviInspection::UpgradeRequired`]. Bad magic, truncation, malformed
    /// identity material, or CRC drift remains actual
    /// [`SearchError::IndexCorrupted`].
    ///
    /// # Errors
    ///
    /// Returns index-not-found/I/O errors or actual structural corruption.
    pub fn inspect(path: &Path) -> SearchResult<FsviInspection> {
        let (version, header) = read_header_for_inspection(path)?;
        match version {
            FSVI_VERSION => {
                let _ = parse_header(path, &header)?;
                Ok(FsviInspection::ReindexRequired(FsviReindexRequired {
                    reason: FsviReindexReason::LegacyUnidentified,
                    found_version: FSVI_VERSION,
                    detail: "FSVI v1 has no complete embedding-space, storage, content, or full-width generation identity; rebuild from source into a separate v2 generation"
                        .to_owned(),
                }))
            }
            FSVI_V2_VERSION => {
                let (metadata, _) = parse_v2_header(path, &header)?;
                Ok(FsviInspection::V2IdentityComplete(Box::new(metadata)))
            }
            found if found > FSVI_V2_VERSION => {
                Ok(FsviInspection::UpgradeRequired(FsviUpgradeRequired {
                    found_version: found,
                    supported_version: FSVI_V2_VERSION,
                }))
            }
            found => Err(index_corrupted(
                path,
                format!("unsupported historical FSVI schema version {found}"),
            )),
        }
    }

    /// Open and fully admit one immutable identity-complete FSVI v2 artifact
    /// into a sealed byte owner.
    ///
    /// The pathname is opened with no-atime and no-follow semantics, copied
    /// once into an [`std::sync::Arc`], and checked against pre/post inode and directory
    /// identity. Header inspection, complete admission, witness hashing, exact
    /// search, and row access all consume that same allocation. Published
    /// admission rejects every WAL directory entry, including an empty or valid
    /// sidecar.
    ///
    /// # Errors
    ///
    /// Returns typed source-reindex or reader-upgrade outcomes for recognized
    /// incompatible formats. I/O and actual corruption are wrapped in
    /// [`FsviAdmissionError::Index`].
    pub fn open_admitted_v2(
        path: &Path,
        expected: &FsviV2IdentityBinding,
    ) -> Result<ValidatedFsviBytes, FsviAdmissionError> {
        ValidatedFsviBytes::open_published(path, expected)
    }

    /// Open an existing legacy FSVI index through an exclusive writer lock.
    ///
    /// This compatibility entry point retains an exclusive OS lock for as
    /// long as its writable map can expose bytes. Query-only consumers should
    /// use [`Self::open_read_only`] so they can coexist under shared locks.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` if the file does not exist and
    /// `SearchError::IndexCorrupted` when header/layout validation fails.
    pub fn open(path: &Path) -> SearchResult<Self> {
        Self::open_writer(path)
    }

    /// Open an existing FSVI index through a shared lock and read-only map.
    ///
    /// This is the normal published-artifact query path. It retains a shared
    /// OS lock for as long as its mapping can expose bytes, preventing a
    /// cooperating writer from creating a writable map until every reader has
    /// dropped its handle.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` for a missing artifact,
    /// `SearchError::IndexCorrupted` for invalid bytes, or a typed
    /// `fsvi.map_lock` refusal when a cooperating writer is live.
    pub fn open_read_only(path: &Path) -> SearchResult<Self> {
        Self::open_with_lock(path, false)
    }

    /// Open an existing legacy FSVI index for mutation.
    ///
    /// The returned index retains an exclusive OS lock for its complete
    /// lifetime before creating its writable map. If any reader or writer
    /// already holds the artifact lock, this returns a typed configuration
    /// refusal instead of mapping bytes writable.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::open`], plus a typed
    /// `fsvi.map_lock` refusal when the artifact is already mapped by a
    /// cooperating reader or writer.
    pub fn open_writer(path: &Path) -> SearchResult<Self> {
        Self::open_with_lock(path, true)
    }

    #[allow(unsafe_code, clippy::too_many_lines)] // memmap constructors require unsafe for memory-mapped I/O.
    pub(crate) fn open_with_lock(path: &Path, writer: bool) -> SearchResult<Self> {
        if !path.exists() {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }

        let file = if writer {
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(SearchError::Io)?
        } else {
            File::open(path).map_err(SearchError::Io)?
        };
        let acquired = Instant::now();
        let map_lock = if writer {
            file.try_lock().map_err(|error| {
                tracing::debug!(
                    target: "frankensearch_index::map_lock",
                    path = %path.display(),
                    mode = "exclusive",
                    thread = ?std::thread::current().id(),
                    "fsvi map lock refused: {error}"
                );
                map_lock_error(path, "exclusive writer", &error)
            })?;
            FsviMapLock::Exclusive {
                file,
                path: path.to_path_buf(),
                acquired,
                creator_pid: std::process::id(),
            }
        } else {
            file.try_lock_shared().map_err(|error| {
                tracing::debug!(
                    target: "frankensearch_index::map_lock",
                    path = %path.display(),
                    mode = "shared",
                    thread = ?std::thread::current().id(),
                    "fsvi map lock refused: {error}"
                );
                map_lock_error(path, "shared reader", &error)
            })?;
            FsviMapLock::Shared {
                file,
                path: path.to_path_buf(),
                acquired,
                creator_pid: std::process::id(),
            }
        };
        tracing::debug!(
            target: "frankensearch_index::map_lock",
            path = %path.display(),
            mode = map_lock.mode(),
            thread = ?std::thread::current().id(),
            "fsvi map lock acquired"
        );
        let data = match &map_lock {
            FsviMapLock::Exclusive { file, .. } => VectorIndexData::Mutable(unsafe {
                MmapMut::map_mut(file).map_err(SearchError::Io)?
            }),
            FsviMapLock::Shared { file, .. } => {
                VectorIndexData::ReadOnly(unsafe { Mmap::map(file).map_err(SearchError::Io)? })
            }
        };
        let (metadata, header_len) = parse_header(path, &data)?;

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
        if vectors_offset < strings_offset {
            return Err(index_corrupted(
                path,
                "vectors_offset points inside the record table/string table region",
            ));
        }

        let vector_bytes = metadata
            .record_count
            .checked_mul(metadata.dimension)
            .and_then(|v| v.checked_mul(metadata.quantization.bytes_per_element()))
            .ok_or_else(|| index_corrupted(path, "vector slab size overflow"))?;
        let required_len = vectors_offset
            .checked_add(vector_bytes)
            .ok_or_else(|| index_corrupted(path, "vector slab end overflow"))?;
        if data.len() < required_len {
            return Err(index_corrupted(
                path,
                format!(
                    "truncated file: have {} bytes, need at least {} bytes",
                    data.len(),
                    required_len
                ),
            ));
        }

        let warm_up_config = WarmUpConfig::from_env();
        if !matches!(warm_up_config.strategy, WarmUpStrategy::None) {
            let warm_up = warmup::warm_up_bytes(&data, header_len, &warm_up_config, None);
            debug!(
                target: "frankensearch.warmup",
                path = %path.display(),
                strategy = %warm_up.strategy_name,
                pages_touched = warm_up.pages_touched,
                bytes_touched = warm_up.bytes_touched,
                budget_exhausted = warm_up.budget_exhausted,
                "index warm-up complete"
            );
        }

        // Load WAL entries if a sidecar file exists.
        let wal_path = wal::wal_path_for(path);
        // Only a write-capable open may quarantine a torn header; a read-only
        // open stays side-effect-free like every other `writer &&` guard here.
        let (wal_entries_raw, wal_compaction_gen, valid_len) = if writer {
            wal::read_wal_recovering_torn_header(
                &wal_path,
                metadata.dimension,
                metadata.quantization,
            )?
        } else {
            wal::read_wal(&wal_path, metadata.dimension, metadata.quantization)?
        };

        let mut deduped_wal = Vec::with_capacity(wal_entries_raw.len());
        let mut seen_ids = std::collections::HashSet::new();
        for entry in wal_entries_raw.into_iter().rev() {
            if seen_ids.insert(entry.doc_id.clone()) {
                deduped_wal.push(entry);
            }
        }
        deduped_wal.reverse();
        let mut wal_entries = deduped_wal;

        let is_stale = if valid_len > 0 {
            if wal_compaction_gen == 0 {
                metadata.compaction_gen > 0
            } else {
                let expected = next_generation(metadata.compaction_gen);
                wal_compaction_gen != expected
            }
        } else {
            false
        };

        if is_stale {
            tracing::warn!(
                path = %path.display(),
                main_gen = metadata.compaction_gen,
                wal_gen = wal_compaction_gen,
                "discarding stale/mismatched WAL entries"
            );
            wal_entries.clear();
            if writer && wal_path.exists() {
                let _ = std::fs::remove_file(&wal_path);
            }
        } else if writer && wal_path.exists() {
            let actual_len = std::fs::metadata(&wal_path).map_err(SearchError::Io)?.len();
            if actual_len > valid_len {
                tracing::warn!(
                    path = %wal_path.display(),
                    actual_len,
                    valid_len,
                    "truncating corrupted WAL trailer"
                );
                let file = OpenOptions::new()
                    .write(true)
                    .open(&wal_path)
                    .map_err(SearchError::Io)?;
                file.set_len(valid_len).map_err(SearchError::Io)?;
                file.sync_all().map_err(SearchError::Io)?;
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            data,
            map_lock: Some(map_lock),
            metadata,
            records_offset,
            strings_offset,
            vectors_offset,
            wal_entries,
            wal_config: WalConfig::default(),
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
        })
    }

    /// Create a writer that stores vectors as f16 with an empty revision string.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when arguments are invalid
    /// (for example, zero dimension or oversized header fields).
    pub fn create(
        path: &Path,
        embedder_id: &str,
        dimension: usize,
    ) -> SearchResult<VectorIndexWriter> {
        Self::create_with_revision(path, embedder_id, "", dimension, Quantization::F16)
    }

    /// Create an immutable identity-complete FSVI v2 writer.
    ///
    /// Unlike the legacy v1 constructors, this API cannot be called with a
    /// display name, empty revision, or dimension-only compatibility claim.
    /// The exact validated identity and full-width generation are required
    /// before any writer is returned.
    ///
    /// The writer starts at publication nonce 0 (single-tier). Every tier of
    /// one multi-tier v2 publication must be stamped with the same non-zero
    /// nonce via [`VectorIndexWriter::with_publication_nonce`] (bd-miio8), or
    /// the pair reads 0 == 0 and is exempt from the mixed-generation check.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if the binding is invalid.
    pub fn create_v2(
        path: &Path,
        binding: FsviV2IdentityBinding,
    ) -> SearchResult<VectorIndexWriter> {
        let embedder_id = binding
            .frozen_identity
            .identity
            .space
            .logical_model_id
            .clone();
        let embedder_revision = binding
            .frozen_identity
            .identity
            .space
            .immutable_revision
            .clone();
        Ok(VectorIndexWriter {
            path: path.to_path_buf(),
            embedder_id,
            embedder_revision,
            dimension: binding.dimension,
            quantization: binding.quantization,
            compaction_gen: 0,
            publication_nonce: 0,
            records: Vec::new(),
            identity_v2: Some(binding),
        })
    }

    /// Replace any existing index generation with a durable empty F16 index.
    ///
    /// Unlike calling [`Self::create`] directly, this first removes and
    /// directory-syncs the incremental WAL sidecar. That ordering prevents a
    /// WAL from the replaced generation from being accepted by the new empty
    /// generation after a crash or restart.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the stale WAL cannot be removed durably, or the
    /// same validation and corruption errors as [`Self::create`] and
    /// [`Self::open`].
    pub fn replace_with_empty(
        path: &Path,
        embedder_id: &str,
        embedder_revision: &str,
        dimension: usize,
    ) -> SearchResult<Self> {
        let replacement_gen = if path.exists() {
            next_generation(Self::peek_compaction_gen(path)?)
        } else {
            1
        };
        let replacement_path = temporary_output_path(path);
        let writer = Self::create_with_revision(
            &replacement_path,
            embedder_id,
            embedder_revision,
            dimension,
            Quantization::F16,
        )?
        .with_generation(replacement_gen);
        writer.finish()?;
        Self::install_replacement(path, &replacement_path)
    }

    /// Durably install a fully written FSVI generation over `path`.
    ///
    /// The replacement is validated before the destination is touched. The
    /// destination WAL is then removed and directory-synced before the main
    /// file is atomically replaced, preventing records from the prior
    /// generation from being replayed into the replacement after a crash.
    ///
    /// # Errors
    ///
    /// Returns an error when the replacement is invalid, has a live WAL, is
    /// the destination itself, or cannot be atomically installed and synced.
    pub fn install_replacement(path: &Path, replacement_path: &Path) -> SearchResult<Self> {
        if path == replacement_path {
            return Err(SearchError::InvalidConfig {
                field: "replacement_path".to_owned(),
                value: replacement_path.display().to_string(),
                reason: "replacement path must differ from destination".to_owned(),
            });
        }

        let replacement = Self::open(replacement_path)?;
        if replacement.wal_record_count() != 0 {
            return Err(SearchError::InvalidConfig {
                field: "replacement_path".to_owned(),
                value: replacement_path.display().to_string(),
                reason: "replacement generation must not have a live WAL".to_owned(),
            });
        }
        let replacement_gen = replacement.metadata.compaction_gen;
        drop(replacement);

        // A two-resource commit (main file + WAL sidecar) cannot be made safe
        // by rename ordering alone: removing the WAL before the rename opens a
        // crash window that destroys acknowledged records of the SURVIVING
        // generation, and removing it after opens one where the stale sidecar
        // is adopted by the replacement. The generation byte is the authority
        // that closes both: the replacement must carry
        // next_generation(destination), so the rename atomically installs the
        // new data AND invalidates the old sidecar (whose records bind to the
        // superseded generation), letting WAL removal happen safely after.
        if path.exists() {
            let destination_gen = Self::peek_compaction_gen(path)?;
            let required = next_generation(destination_gen);
            if replacement_gen != required {
                return Err(SearchError::InvalidConfig {
                    field: "replacement_path".to_owned(),
                    value: replacement_path.display().to_string(),
                    reason: format!(
                        "replacement compaction generation {replacement_gen} must be \
                         next_generation(destination {destination_gen}) = {required}; build \
                         the replacement with with_generation(next_generation(\
                         VectorIndex::peek_compaction_gen(destination)?)) so the rename \
                         atomically invalidates any surviving destination WAL"
                    ),
                });
            }
        }

        let temporary = tempfile::TempPath::try_from_path(replacement_path.to_path_buf())?;
        if let Err(error) = temporary.persist(path) {
            let tempfile::PathPersistError { error, mut path } = error;
            path.disable_cleanup(true);
            return Err(SearchError::Io(error));
        }
        #[cfg(unix)]
        sync_parent_directory(path)?;

        // The stale sidecar is already invalid (generation mismatch); remove
        // it so a healthy install leaves no residue. On failure, open() below
        // and every future open reject and clean it anyway.
        let wal_path = wal::wal_path_for(path);
        if let Err(error) = wal::remove_wal(&wal_path) {
            tracing::warn!(
                path = %wal_path.display(),
                error = %error,
                "superseded WAL sidecar could not be removed after install; \
                 open() will reject it by generation"
            );
        } else {
            #[cfg(unix)]
            sync_parent_directory(&wal_path)?;
        }
        Self::open(path)
    }

    /// Read the on-disk compaction generation of a published FSVI without
    /// opening it — no WAL adoption, trailer truncation, or sidecar cleanup
    /// side effects. Shares [`Self::open`]'s v1 header domain (the mutable
    /// generation surface [`Self::install_replacement`] operates on).
    ///
    /// # Errors
    ///
    /// Returns the same header corruption/version errors as [`Self::open`].
    pub fn peek_compaction_gen(path: &Path) -> SearchResult<u8> {
        Ok(Self::peek_header_metadata(path)?.compaction_gen)
    }

    /// Read the on-disk publication nonce of a published FSVI without opening
    /// it, with the same side-effect-free contract as
    /// [`Self::peek_compaction_gen`].
    ///
    /// # Errors
    ///
    /// Returns the same header corruption/version errors as [`Self::open`].
    pub fn peek_publication_nonce(path: &Path) -> SearchResult<u16> {
        Ok(Self::peek_header_metadata(path)?.publication_nonce)
    }

    fn peek_header_metadata(path: &Path) -> SearchResult<VectorMetadata> {
        use std::io::Read;
        // The variable-length header (two u16-length strings + fixed fields +
        // CRC) cannot exceed 256 KiB, so this prefix always suffices.
        let file = File::open(path).map_err(SearchError::Io)?;
        let mut data = Vec::new();
        file.take(256 * 1024)
            .read_to_end(&mut data)
            .map_err(SearchError::Io)?;
        let (metadata, _header_len) = parse_header(path, &data)?;
        Ok(metadata)
    }

    /// Create a writer with explicit embedder revision and quantization.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when arguments are invalid
    /// (for example, zero dimension or oversized header fields).
    pub fn create_with_revision(
        path: &Path,
        embedder_id: &str,
        embedder_revision: &str,
        dimension: usize,
        quantization: Quantization,
    ) -> SearchResult<VectorIndexWriter> {
        if dimension == 0 {
            return Err(SearchError::InvalidConfig {
                field: "dimension".to_owned(),
                value: "0".to_owned(),
                reason: "dimension must be greater than zero".to_owned(),
            });
        }
        validate_header_string(embedder_id, "embedder_id")?;
        validate_header_string(embedder_revision, "embedder_revision")?;
        let _ = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
            field: "dimension".to_owned(),
            value: dimension.to_string(),
            reason: "dimension must fit in u32 for FSVI header encoding".to_owned(),
        })?;

        Ok(VectorIndexWriter {
            path: path.to_path_buf(),
            embedder_id: embedder_id.to_owned(),
            embedder_revision: embedder_revision.to_owned(),
            dimension,
            quantization,
            compaction_gen: 1,
            publication_nonce: 0,
            records: Vec::new(),
            identity_v2: None,
        })
    }

    /// Number of vectors in this index.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.metadata.record_count
    }

    /// Embedding dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.metadata.dimension
    }

    /// Path of the main FSVI file this handle is mapped from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Embedder id stored in the index header.
    #[must_use]
    pub fn embedder_id(&self) -> &str {
        &self.metadata.embedder_id
    }

    /// Embedder revision stored in the index header.
    #[must_use]
    pub fn embedder_revision(&self) -> &str {
        &self.metadata.embedder_revision
    }

    /// Stored quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.metadata.quantization
    }

    /// Shared publication identity of a multi-tier install; zero on legacy
    /// and single-tier files (bd-miio8).
    #[must_use]
    pub const fn publication_nonce(&self) -> u16 {
        self.metadata.publication_nonce
    }

    /// Full parsed metadata.
    #[must_use]
    pub const fn metadata(&self) -> &VectorMetadata {
        &self.metadata
    }

    /// Complete v2 identity/content metadata, absent for legacy v1 bytes.
    #[must_use]
    pub const fn identity_v2(&self) -> Option<&FsviV2IdentityMetadata> {
        self.metadata.identity_v2.as_ref()
    }

    /// Whether this index was opened through exact FSVI v2 admission.
    #[must_use]
    pub const fn is_identity_admitted_v2(&self) -> bool {
        self.metadata.fsvi_version == FSVI_V2_VERSION && self.metadata.identity_v2.is_some()
    }

    // ─── WAL / Incremental Update API ───────────────────────────────────

    /// Set the WAL configuration for incremental updates.
    pub const fn set_wal_config(&mut self, config: WalConfig) {
        self.wal_config = config;
    }

    /// Number of entries in the write-ahead log (pending compaction).
    #[must_use]
    pub const fn wal_record_count(&self) -> usize {
        self.wal_entries.len()
    }

    /// Return the document ids with live membership in either the compacted
    /// main index or the resident write-ahead log.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] if a live main-index document
    /// id cannot be decoded.
    pub fn live_doc_ids(&self) -> SearchResult<std::collections::HashSet<String>> {
        let mut ids = std::collections::HashSet::with_capacity(
            self.record_count().saturating_add(self.wal_record_count()),
        );
        for record_index in 0..self.record_count() {
            if !self.is_deleted(record_index) {
                ids.insert(self.doc_id_at(record_index)?.to_owned());
            }
        }
        ids.extend(self.wal_entries.iter().map(|entry| entry.doc_id.clone()));
        Ok(ids)
    }

    /// Iterate the WAL-resident rows as `(doc_id, embedding)` pairs.
    ///
    /// These are acknowledged appends that have not been compacted into the
    /// main slab yet; any full rebuild that merges "the previous index"
    /// must include them or it silently drops durable writes. The resident
    /// set is already deduplicated (last write wins per doc ID).
    pub fn wal_records(&self) -> impl Iterator<Item = (&str, &[f32])> {
        self.wal_entries
            .iter()
            .map(|entry| (entry.doc_id.as_str(), entry.embedding.as_slice()))
    }

    /// Whether the WAL is large enough that compaction is recommended.
    ///
    /// Returns `true` when the WAL exceeds either the absolute threshold
    /// or the ratio threshold relative to the main index size.
    #[must_use]
    pub fn needs_compaction(&self) -> bool {
        if self.wal_entries.is_empty() {
            return false;
        }
        if self.wal_entries.len() >= self.wal_config.compaction_threshold {
            return true;
        }
        if self.record_count() > 0 {
            #[allow(clippy::cast_precision_loss)]
            let ratio = self.wal_entries.len() as f64 / self.record_count() as f64;
            // NaN compaction_ratio makes >= always false, silently disabling
            // ratio-based compaction. Fall back to the default.
            let threshold = if self.wal_config.compaction_ratio.is_finite() {
                self.wal_config.compaction_ratio
            } else {
                0.10
            };
            if ratio >= threshold {
                return true;
            }
        }
        false
    }

    /// Resolve the document id at `index`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed record/string tables.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.ensure_index(index)?;
        let entry = self.record_at(index)?;
        let doc_id_offset = usize::try_from(entry.doc_id_offset).map_err(|_| {
            index_corrupted(
                &self.path,
                format!("doc_id_offset overflow for record at index {index}"),
            )
        })?;
        let doc_id_len = usize::from(entry.doc_id_len);
        let start = self
            .strings_offset
            .checked_add(doc_id_offset)
            .ok_or_else(|| index_corrupted(&self.path, "doc_id start offset overflow"))?;
        let end = start
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(&self.path, "doc_id end offset overflow"))?;
        if end > self.vectors_offset {
            return Err(index_corrupted(
                &self.path,
                format!(
                    "doc_id range [{start}, {end}) exceeds string table end {}",
                    self.vectors_offset
                ),
            ));
        }
        std::str::from_utf8(&self.data[start..end]).map_err(|error| {
            index_corrupted(
                &self.path,
                format!("invalid UTF-8 in doc_id at index {index}: {error}"),
            )
        })
    }

    /// Decode a vector as f32 values.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed vector slab data.
    pub fn vector_at_f32(&self, index: usize) -> SearchResult<Vec<f32>> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let dim = self.dimension();
        match self.quantization() {
            Quantization::F32 => {
                let byte_len = dim.checked_mul(4).ok_or_else(|| {
                    index_corrupted(&self.path, "f32 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f32 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f32 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                for chunk in self.data[start..end].as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(*chunk));
                }
                Ok(out)
            }
            Quantization::F16 => {
                let byte_len = dim.checked_mul(2).ok_or_else(|| {
                    index_corrupted(&self.path, "f16 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f16 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f16 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                // SIMD-widen 8 little-endian f16 per 16-byte block (`widen8_f16_bytes`,
                // the same magic-factor widen the f16 dot kernels use — bit-identical
                // to the scalar `f16::to_f32`), then a scalar tail for the last < 8.
                let (blocks, remainder) = self.data[start..end].as_chunks::<16>();
                for arr in blocks {
                    out.extend_from_slice(&crate::simd::widen8_f16_bytes(arr).to_array());
                }
                for chunk in remainder.as_chunks::<2>().0 {
                    out.push(f16::from_le_bytes(*chunk).to_f32());
                }
                Ok(out)
            }
        }
    }

    /// Raw stored slab bytes of the vector at `index` (`dim * 2` for f16,
    /// `dim * 4` for f32) — the same window [`Self::vector_at_f32`] decodes, but
    /// borrowed without materializing an f32 `Vec`. Feeds the fused-kernel scorer.
    pub(crate) fn vector_bytes(&self, index: usize) -> SearchResult<&[u8]> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let byte_len = self
            .dimension()
            .checked_mul(self.quantization().bytes_per_element())
            .ok_or_else(|| index_corrupted(&self.path, "vector byte length overflow"))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| index_corrupted(&self.path, "vector byte end overflow"))?;
        if end > self.data.len() {
            return Err(index_corrupted(&self.path, "vector extends past file end"));
        }
        Ok(&self.data[start..end])
    }

    /// Exact dot of the stored vector at `index` against an f32 `query`, using the
    /// fused byte-based SIMD kernel (`dot_product_f16_bytes_f32` / `_f32_bytes_f32`)
    /// the brute-force scan already uses — no per-call f32 `Vec` and a hardware
    /// (`vcvtph2ps`) f16 decode instead of the scalar [`Self::vector_at_f32`] loop.
    ///
    /// For `dim % 32 == 0` (every standard embedding width: 128/256/384/512/768…)
    /// the fused kernel's 4-accumulator grouping and reduction coincide exactly
    /// with `vector_at_f32` + `dot_product_f32_f32` and there is no scalar tail, so
    /// the score is **bit-identical** to the former decode-then-dot path. Other
    /// dims fall back to that path to preserve exact scores.
    ///
    /// # Errors
    ///
    /// Propagates the same index/corruption errors as [`Self::vector_at_f32`], and
    /// `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn dot_query_at(&self, index: usize, query: &[f32]) -> SearchResult<f32> {
        if self.dimension() % 32 != 0 {
            let vector = self.vector_at_f32(index)?;
            return dot_product_f32_f32(&vector, query);
        }
        let bytes = self.vector_bytes(index)?;
        match self.quantization() {
            Quantization::F16 => dot_product_f16_bytes_f32(bytes, query),
            Quantization::F32 => dot_product_f32_bytes_f32(bytes, query),
        }
    }

    /// Decode a vector as f16 values.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed vector slab data, or
    /// `SearchError::InvalidConfig` when the output allocation fails.
    pub fn vector_at_f16(&self, index: usize) -> SearchResult<Vec<f16>> {
        let mut output = Vec::new();
        self.extend_vector_at_f16(index, &mut output)?;
        Ok(output)
    }

    /// Decode one vector directly into a caller-owned f16 slab.
    ///
    /// This is the allocation-bounded counterpart to [`Self::vector_at_f16`]:
    /// callers that already reserved a destination for many rows avoid one
    /// transient `Vec` allocation per source row. The destination's required
    /// growth is checked before any values are appended.
    ///
    /// # Errors
    ///
    /// Returns the same indexing and corruption errors as
    /// [`Self::vector_at_f16`], or [`SearchError::InvalidConfig`] if the
    /// destination cannot reserve this row.
    pub fn extend_vector_at_f16(
        &self,
        index: usize,
        destination: &mut Vec<f16>,
    ) -> SearchResult<()> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let dimension = self.dimension();
        match self.quantization() {
            Quantization::F16 => {
                let byte_len = dimension.checked_mul(2).ok_or_else(|| {
                    index_corrupted(&self.path, "f16 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f16 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f16 vector extends past file end",
                    ));
                }
                reserve_f16_destination(destination, dimension)?;
                for chunk in self.data[start..end].as_chunks::<2>().0 {
                    destination.push(f16::from_le_bytes(*chunk));
                }
            }
            Quantization::F32 => {
                let byte_len = dimension.checked_mul(4).ok_or_else(|| {
                    index_corrupted(&self.path, "f32 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f32 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f32 vector extends past file end",
                    ));
                }
                reserve_f16_destination(destination, dimension)?;
                for chunk in self.data[start..end].as_chunks::<4>().0 {
                    destination.push(f16::from_f32(f32::from_le_bytes(*chunk)));
                }
            }
        }
        Ok(())
    }

    /// Binary-search the sorted record table by document hash.
    #[must_use]
    pub fn find_index_by_doc_hash(&self, doc_id_hash: u64) -> Option<usize> {
        let mut low = 0usize;
        let mut high = self.record_count();
        while low < high {
            let mid = low + (high - low) / 2;
            let entry = self.record_at(mid).ok()?;
            match entry.doc_id_hash.cmp(&doc_id_hash) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => {
                    let mut first = mid;
                    while first > 0 {
                        let prev = self.record_at(first - 1).ok()?;
                        if prev.doc_id_hash != doc_id_hash {
                            break;
                        }
                        first -= 1;
                    }
                    for index in first..self.record_count() {
                        let entry = self.record_at(index).ok()?;
                        if entry.doc_id_hash != doc_id_hash {
                            break;
                        }
                        if !is_tombstoned_flags(entry.flags) {
                            return Some(index);
                        }
                    }
                    return None;
                }
            }
        }
        None
    }

    /// Fetch embeddings for hashed doc ids (f16 values).
    ///
    /// Missing hashes return `None` entries at the same position.
    #[must_use]
    pub fn get_embeddings(&self, doc_id_hashes: &[u64]) -> Vec<Option<Vec<f16>>> {
        doc_id_hashes
            .iter()
            .map(|&hash| {
                for entry in self.wal_entries.iter().rev() {
                    if entry.doc_id_hash == hash {
                        // WAL embeddings are f32, we need to convert them to f16
                        return Some(
                            entry
                                .embedding
                                .iter()
                                .map(|&v| half::f16::from_f32(v))
                                .collect(),
                        );
                    }
                }
                if let Some(index) = self.find_index_by_doc_hash(hash) {
                    if let Ok(vec) = self.vector_at_f16(index) {
                        return Some(vec);
                    }
                }
                None
            })
            .collect()
    }

    pub(crate) fn ensure_index(&self, index: usize) -> SearchResult<()> {
        if index >= self.record_count() {
            return Err(SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index out of range for record_count={}",
                    self.record_count()
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn find_index_by_doc_id(&self, doc_id: &str) -> SearchResult<Option<usize>> {
        let doc_id_hash = fnv1a_hash(doc_id.as_bytes());
        let Some(mut index) = self.find_first_hash_match(doc_id_hash)? else {
            return Ok(None);
        };
        while index > 0 {
            let prev = self.record_at(index - 1)?;
            if prev.doc_id_hash != doc_id_hash {
                break;
            }
            index -= 1;
        }

        for candidate in index..self.record_count() {
            let entry = self.record_at(candidate)?;
            if entry.doc_id_hash != doc_id_hash {
                break;
            }
            if !is_tombstoned_flags(entry.flags) {
                let candidate_doc_id = self.doc_id_at(candidate)?;
                if candidate_doc_id == doc_id {
                    return Ok(Some(candidate));
                }
            }
        }
        Ok(None)
    }

    pub(super) fn find_first_hash_match(&self, doc_id_hash: u64) -> SearchResult<Option<usize>> {
        let mut low = 0usize;
        let mut high = self.record_count();
        while low < high {
            let mid = low + (high - low) / 2;
            let entry = self.record_at(mid)?;
            match entry.doc_id_hash.cmp(&doc_id_hash) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }

    fn record_flags_offset(&self, index: usize) -> SearchResult<usize> {
        self.ensure_index(index)?;
        let record_offset = self
            .records_offset
            .checked_add(index.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
                index_corrupted(&self.path, "record offset multiplication overflow")
            })?)
            .ok_or_else(|| index_corrupted(&self.path, "record offset overflow"))?;
        record_offset
            .checked_add(14)
            .ok_or_else(|| index_corrupted(&self.path, "flags offset overflow"))
    }

    pub(crate) fn set_record_flags(&mut self, index: usize, flags: u16) -> SearchResult<()> {
        let flags_offset = self.record_flags_offset(index)?;
        let end = flags_offset
            .checked_add(2)
            .ok_or_else(|| index_corrupted(&self.path, "flags end overflow"))?;
        if end > self.data.len() {
            return Err(index_corrupted(
                &self.path,
                "flags offset points beyond mapped data",
            ));
        }

        self.data
            .write_and_flush(flags_offset, &flags.to_le_bytes())
    }

    /// In-place flag writes go through the shared mapping. On tmpfs a page
    /// that is already dirty never faults again, so the file's mtime/ctime can
    /// stay exactly as they were and every consumer that fingerprints the
    /// generation by metadata (the fsfs search-resource rebind) would keep
    /// serving tombstoned records. Bump the timestamp explicitly after such a
    /// write; ext4 does this on the write fault, tmpfs does not.
    pub(super) fn touch_main_file(&self) -> SearchResult<()> {
        let file = OpenOptions::new()
            .write(true)
            .open(&self.path)
            .map_err(SearchError::Io)?;
        file.set_modified(SystemTime::now())
            .map_err(SearchError::Io)
    }

    pub(super) fn rewrite_wal_sidecar(&self) -> SearchResult<()> {
        self.ensure_legacy_mutation_format("rewrite_wal_sidecar")?;
        let wal_path = wal::wal_path_for(&self.path);
        if self.wal_entries.is_empty() {
            wal::remove_wal(&wal_path)?;
            // Durable removal needs the dirent update persisted: without the
            // parent sync a crash can resurrect the removed sidecar, whose
            // stale entries would replay over the post-delete state.
            #[cfg(unix)]
            sync_parent_directory(&wal_path)?;
            return Ok(());
        }

        let mut tmp = wal_path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp_path = PathBuf::from(tmp);
        let _ = wal::remove_wal(&tmp_path);

        if let Err(e) = wal::append_wal_batch(
            &tmp_path,
            &self.wal_entries,
            self.dimension(),
            self.quantization(),
            next_generation(self.metadata.compaction_gen),
            self.wal_config.fsync_on_write,
        ) {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }

        match fs::rename(&tmp_path, &wal_path) {
            Ok(()) => {
                #[cfg(unix)]
                sync_parent_directory(&wal_path)?;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                wal::remove_wal(&wal_path)?;
                fs::rename(&tmp_path, &wal_path)?;
                #[cfg(unix)]
                sync_parent_directory(&wal_path)?;
                Ok(())
            }
            Err(error) => {
                let _ = wal::remove_wal(&tmp_path);
                Err(error.into())
            }
        }
    }

    pub(crate) fn record_at(&self, index: usize) -> SearchResult<RecordEntry> {
        self.ensure_index(index)?;
        let offset = self
            .records_offset
            .checked_add(index.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
                index_corrupted(&self.path, "record offset multiplication overflow")
            })?)
            .ok_or_else(|| index_corrupted(&self.path, "record offset overflow"))?;
        let end = offset
            .checked_add(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(&self.path, "record end overflow"))?;
        if end > self.data.len() {
            return Err(index_corrupted(
                &self.path,
                "record table extends beyond file size",
            ));
        }
        let chunk = &self.data[offset..end];
        Ok(RecordEntry {
            doc_id_hash: u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            doc_id_offset: u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
            doc_id_len: u16::from_le_bytes([chunk[12], chunk[13]]),
            flags: u16::from_le_bytes([chunk[14], chunk[15]]),
        })
    }

    pub(crate) fn vector_start(&self, index: usize) -> SearchResult<usize> {
        let stride = self
            .dimension()
            .checked_mul(self.quantization().bytes_per_element())
            .ok_or_else(|| index_corrupted(&self.path, "vector stride overflow"))?;
        self.vectors_offset
            .checked_add(
                index
                    .checked_mul(stride)
                    .ok_or_else(|| index_corrupted(&self.path, "vector index overflow"))?,
            )
            .ok_or_else(|| index_corrupted(&self.path, "vector offset overflow"))
    }

    pub(super) fn ensure_legacy_mutation_format(&self, operation: &str) -> SearchResult<()> {
        if self.metadata.fsvi_version == FSVI_V2_VERSION {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.mutation".to_owned(),
                value: operation.to_owned(),
                reason: "identity-complete FSVI v2 artifacts are immutable in this admission slice; rebuild and publish a separate generation rather than mutating content or attaching a legacy WAL"
                    .to_owned(),
            });
        }
        if self.map_lock.as_ref().is_some_and(FsviMapLock::is_writer) {
            return Ok(());
        }
        Err(SearchError::InvalidConfig {
            field: "fsvi.writer_lock".to_owned(),
            value: operation.to_owned(),
            reason: "this handle is a shared read-only mapping; reopen with VectorIndex::open_writer before mutating a published FSVI"
                .to_owned(),
        })
    }
}
