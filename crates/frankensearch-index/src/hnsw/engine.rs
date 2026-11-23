use std::path::Path;
use std::sync::atomic::AtomicBool;

use frankensearch_core::{SearchError, SearchResult};
use hnsw_rs::hnswio::ReloadOptions;
use hnsw_rs::prelude::{AnnT, DistDot, HnswIo};

use crate::VectorIndex;

use super::fingerprint::{
    ann_corrupted, dist_dot_budget, fingerprint_doc_ids, fingerprint_hnsw_sidecar,
    fingerprint_source_map, fingerprint_vector_index_positions, meta_matches_live_doc_ids,
    normalize_for_dist_dot,
};
use super::persistence::{
    acquire_hnsw_save_lock, find_reusable_hnsw_generation, gc_superseded_hnsw_generations,
    hnsw_generation_prefix, hnsw_metadata_file_name, hnsw_sidecar_basename,
    native_sidecar_pair_is_local, persisted_hnsw_sidecar_location, publish_hnsw_metadata,
    serialize_hnsw_metadata, sync_hnsw_directory, sync_hnsw_sidecars,
    validate_hnsw_generation_receipt, validate_hnsw_sidecar_basename,
    write_hnsw_generation_receipt,
};
use super::topology::{ann_topology_error, validate_hnsw_topology};
use super::types::{
    HNSW_GENERATION_RECEIPT_VERSION, HNSW_META_FORMAT_CURRENT, HnswAppendDisposition, HnswConfig,
    HnswGenerationReceipt, HnswIndex, HnswLoadDisposition, HnswMeta, HnswMetadataPublisher,
    HnswSourceIdentityV1,
};

impl HnswIndex {
    /// Append new document IDs to an existing (including natively loaded) graph.
    ///
    /// Both sources must carry explicit generation identities and identical model,
    /// space, and storage identities. Every prior row must remain byte-equivalent
    /// after decoding; deletions, duplicate IDs, and edited old vectors require
    /// a rebuild. Source rows may move when FSVI sorts new IDs: old graph origin
    /// IDs stay unchanged and their physical source mapping is updated.
    /// Refusal returns the unchanged graph. On an insertion
    /// error this consuming API discards the candidate, so a partially modified
    /// graph cannot escape under its old identity. Persisted graphs are untouched.
    ///
    /// Validation and fingerprinting read the full source; native insertion only
    /// processes new IDs. Loading and saving still process the whole graph.
    ///
    /// # Errors
    /// Returns source decoding, invalid vector, or graph topology errors.
    pub fn append_from_vector_index(
        mut self,
        previous: &VectorIndex,
        current: &VectorIndex,
    ) -> SearchResult<(Self, HnswAppendDisposition)> {
        let refuse =
            |graph, reason| Ok((graph, HnswAppendDisposition::FullRebuildRequired { reason }));
        if !self.matches_vector_index(previous)?
            || self.source_record_count != previous.record_count()
        {
            return refuse(self, "prior_source_mismatch");
        }
        if self.matches_vector_index(current)? {
            return Ok((self, HnswAppendDisposition::NoOp));
        }
        let (Some(prior_identity), Some(current_identity)) = (
            HnswSourceIdentityV1::capture(previous),
            HnswSourceIdentityV1::capture(current),
        ) else {
            return refuse(self, "missing_source_identity");
        };
        if previous.dimension() != current.dimension()
            || previous.embedder_id() != current.embedder_id()
            || prior_identity.identity_bundle_fingerprint
                != current_identity.identity_bundle_fingerprint
            || prior_identity.space_fingerprint != current_identity.space_fingerprint
            || prior_identity.storage_fingerprint != current_identity.storage_fingerprint
        {
            return refuse(self, "embedding_identity_changed");
        }
        if current.record_count() <= previous.record_count() {
            return refuse(self, "not_an_append");
        }
        if prior_identity.generation == current_identity.generation {
            return refuse(self, "generation_not_advanced");
        }
        if self.doc_ids.len() != previous.record_count() {
            return refuse(self, "deleted_rows");
        }
        let old_origins: std::collections::HashMap<_, _> = self
            .doc_ids
            .iter()
            .enumerate()
            .map(|(origin, id)| (id.clone(), origin))
            .collect();
        if old_origins.len() != self.doc_ids.len() {
            return refuse(self, "duplicate_document_id");
        }
        let mut positions = vec![0_u32; self.doc_ids.len()];
        let mut matched_old = 0_usize;
        let mut identifiers = std::collections::HashSet::new();
        let mut new_rows = Vec::new();
        for row in 0..current.record_count() {
            if current.is_deleted(row) {
                return refuse(self, "deleted_rows");
            }
            let id = current.doc_id_at(row)?;
            if !identifiers.insert(id.to_owned()) {
                return refuse(self, "duplicate_document_id");
            }
            let Ok(position) = u32::try_from(row) else {
                return refuse(self, "source_position_overflow");
            };
            if let Some(&origin) = old_origins.get(id) {
                let prior_row = self.source_positions[origin] as usize;
                if previous
                    .vector_at_f32(prior_row)?
                    .iter()
                    .map(|v| v.to_bits())
                    .ne(current.vector_at_f32(row)?.iter().map(|v| v.to_bits()))
                {
                    return refuse(self, "old_vector_changed");
                }
                positions[origin] = position;
                matched_old += 1;
            } else {
                new_rows.push((id.to_owned(), position));
            }
        }
        if matched_old != self.doc_ids.len() {
            return refuse(self, "old_document_missing");
        }
        let budget = dist_dot_budget(self.dimension)?;
        let mut suffix = Vec::new();
        let mut doc_ids = self.doc_ids.clone();
        for (id, position) in new_rows {
            let vector = current.vector_at_f32(position as usize)?;
            if vector.len() != self.dimension || vector.iter().any(|value| !value.is_finite()) {
                return refuse(self, "invalid_suffix_vector");
            }
            doc_ids.push(id);
            positions.push(position);
            suffix.push(normalize_for_dist_dot(vector, budget));
        }
        let fingerprint =
            fingerprint_vector_index_positions(current, &positions, &doc_ids, self.dimension)?;
        let inserted = suffix.len();
        self.hnsw.set_searching_mode(false);
        for (offset, vector) in suffix.into_iter().enumerate() {
            self.hnsw
                .insert_slice((&vector, self.doc_ids.len() + offset));
        }
        validate_hnsw_topology(&self.hnsw, doc_ids.len())
            .map_err(|detail| ann_topology_error(&detail))?;
        self.doc_ids = doc_ids;
        self.source_positions = positions;
        self.source_record_count = current.record_count();
        self.vector_fingerprint = fingerprint;
        self.source_identity = Some(current_identity);
        self.underfill_warned = AtomicBool::new(false);
        Ok((self, HnswAppendDisposition::Appended { inserted }))
    }

    /// Build a new HNSW index from an opened `VectorIndex`.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `SearchError::InvalidConfig` for invalid HNSW params
    /// - `SearchError::IndexCorrupted` if `vectors/doc_ids` cannot be decoded
    pub fn build_from_vector_index(index: &VectorIndex, config: HnswConfig) -> SearchResult<Self> {
        let dimension = index.dimension();
        let mut doc_ids = Vec::with_capacity(index.record_count());
        let mut vectors = Vec::with_capacity(index.record_count());
        let mut live_positions = Vec::with_capacity(index.record_count());
        for i in 0..index.record_count() {
            if index.is_deleted(i) {
                continue;
            }
            doc_ids.push(index.doc_id_at(i)?.to_owned());
            vectors.push(index.vector_at_f32(i)?);
            live_positions.push(i);
        }
        let mut ann = Self::build_from_parts(doc_ids, vectors, dimension, config)?;
        // Bind the graph to the exact generation it was built from, so a later
        // load cannot serve it against a different one (bd-r65a).
        ann.source_identity = HnswSourceIdentityV1::capture(index);
        ann.source_record_count = index.record_count();
        ann.source_positions = live_positions
            .into_iter()
            .map(|position| {
                u32::try_from(position).map_err(|_| SearchError::InvalidConfig {
                    field: "source_position".to_owned(),
                    value: position.to_string(),
                    reason: "VectorIndex row exceeds u32".to_owned(),
                })
            })
            .collect::<SearchResult<Vec<_>>>()?;
        Ok(ann)
    }

    /// Load an ANN index from disk, rebuilding the graph from `source_index` when
    /// the native graph/data pair is legacy, stale, missing, or corrupt.
    ///
    /// The source index validates the persisted document sequence and vector
    /// fingerprint. A fallback rebuild reads every live row from that same
    /// source, preserving row-to-vector alignment even when document IDs are
    /// not unique.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexCorrupted` if metadata is missing/malformed or
    /// if live rows cannot be decoded from `source_index`.
    pub fn load(path: &Path, source_index: &VectorIndex) -> SearchResult<Self> {
        Self::load_with_disposition(path, source_index).map(|(index, _)| index)
    }

    /// Try to load only the exact persisted native graph paired with
    /// `source_index`, without ever rebuilding it.
    ///
    /// `Ok(Some(index))` proves the current native graph/data generation,
    /// digest receipt, ordered document IDs, vector fingerprint, dimension,
    /// and topology all match `source_index`. `Ok(None)` means the metadata is
    /// readable but the selected graph is legacy, stale, incomplete, corrupt,
    /// or otherwise not admissible as that native artifact. This method never
    /// scans source rows to construct a replacement graph and never writes,
    /// replaces, renames, or changes permissions or mtimes.
    ///
    /// # Errors
    ///
    /// Returns an error when the metadata file cannot be read or parsed, its
    /// dimension disagrees with `source_index`, or its dimension exceeds the
    /// supported distance-kernel budget.
    pub fn try_load_native(path: &Path, source_index: &VectorIndex) -> SearchResult<Option<Self>> {
        let meta = Self::validated_load_metadata(path, source_index)?;
        if meta.format_version != HNSW_META_FORMAT_CURRENT {
            return Ok(None);
        }
        Ok(Self::try_load_native_graph(path, &meta, source_index))
    }

    /// Load an ANN index and report whether its graph came from native sidecars
    /// or was rebuilt from `source_index`.
    ///
    /// This is the fail-closed inspection surface for consumers that must
    /// distinguish the exact persisted ANN artifact from a compatible
    /// in-memory fallback. Neither outcome writes, replaces, renames, or
    /// changes permissions on `path`, its native sidecars, or `source_index`.
    /// Callers that require the selected on-disk graph must accept only
    /// [`HnswLoadDisposition::Native`].
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexCorrupted` when metadata is unreadable,
    /// malformed, dimensionally incompatible, or the source rows cannot be
    /// decoded. A readable legacy/stale/corrupt native graph is instead
    /// reported as [`HnswLoadDisposition::Rebuilt`] after an in-memory rebuild.
    pub fn load_with_disposition(
        path: &Path,
        source_index: &VectorIndex,
    ) -> SearchResult<(Self, HnswLoadDisposition)> {
        let meta = Self::validated_load_metadata(path, source_index)?;

        // Current format: deserialize the prebuilt native graph directly, skipping the
        // O(n log n) rebuild. Any problem (missing/corrupt sidecars, point-count
        // mismatch, or stale vector fingerprint) returns None and we fall
        // through to the rebuild path, so a bad graph sidecar degrades to
        // "slow load" rather than a hard failure.
        if meta.format_version == HNSW_META_FORMAT_CURRENT
            && let Some(index) = Self::try_load_native_graph(path, &meta, source_index)
        {
            return Ok((index, HnswLoadDisposition::Native));
        } else if meta.format_version != HNSW_META_FORMAT_CURRENT {
            tracing::warn!(
                path = %path.display(),
                format_version = meta.format_version,
                current_format_version = HNSW_META_FORMAT_CURRENT,
                "rebuilding HNSW sidecar written with a different persistence contract; \
                 re-save to skip rebuild on the next cold load"
            );
        }

        // v1/legacy or fallback: rebuild directly from live source rows. Looking
        // vectors up by doc ID would collapse duplicate IDs onto their first
        // occurrence and silently attach the wrong vector to later rows.
        Self::build_from_vector_index(source_index, meta.config)
            .map(|index| (index, HnswLoadDisposition::Rebuilt))
    }

    pub(super) fn validated_load_metadata(
        path: &Path,
        source_index: &VectorIndex,
    ) -> SearchResult<HnswMeta> {
        let metadata_bytes = std::fs::read(path).map_err(SearchError::Io)?;
        let meta: HnswMeta = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| ann_corrupted(path, format!("failed to parse HNSW metadata: {e}")))?;

        if meta.dimension != source_index.dimension() {
            return Err(ann_corrupted(
                path,
                format!(
                    "dimension mismatch: hnsw={} source={}",
                    meta.dimension,
                    source_index.dimension()
                ),
            ));
        }

        // Validate before the native fast path. Otherwise a forged current-format sidecar
        // could bypass the dimension bound enforced by graph construction.
        dist_dot_budget(meta.dimension)?;
        Ok(meta)
    }

    /// Attempt to load the prebuilt native `hnsw_rs` graph for a current-format sidecar.
    ///
    /// Returns `None` if any of the following hold. Permissive callers may
    /// rebuild from the source index; strict callers can reject the artifact:
    /// - the `.hnsw.graph` / `.hnsw.data` sidecars are absent,
    /// - `hnsw_rs` fails to deserialize them,
    /// - the loaded point count disagrees with the metadata `doc_ids`,
    /// - the metadata `doc_ids` disagree with the live `VectorIndex`'s
    ///   doc-id sequence (live tombstones excluded),
    /// - the persisted vector fingerprint disagrees with the live
    ///   `VectorIndex`'s fingerprint (i.e. vectors were swapped behind the
    ///   same doc ids — the case the prompt explicitly calls out).
    pub(super) fn try_load_native_graph(
        path: &Path,
        meta: &HnswMeta,
        source_index: &VectorIndex,
    ) -> Option<Self> {
        let (sidecar_parent, basename) = persisted_hnsw_sidecar_location(path, meta).ok()?;
        let graph = sidecar_parent.join(format!("{basename}.hnsw.graph"));
        let data = sidecar_parent.join(format!("{basename}.hnsw.data"));
        if !native_sidecar_pair_is_local(path, &sidecar_parent, &graph, &data) {
            return None;
        }
        let metadata_file_name = hnsw_metadata_file_name(path).ok()?;
        let validated_generation = match validate_hnsw_generation_receipt(
            path,
            &sidecar_parent,
            &metadata_file_name,
            &meta.doc_ids,
            meta.vector_fingerprint,
            fingerprint_source_map(&meta.source_positions, meta.source_record_count),
            meta.dimension,
            meta.config,
        ) {
            Ok(Some(validated)) => validated,
            Ok(None) => {
                tracing::warn!(
                    path = %path.display(),
                    "HNSW native sidecars lack a matching digest receipt; native load unavailable"
                );
                return None;
            }
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    ?error,
                    "HNSW native sidecar receipt validation failed; native load unavailable"
                );
                return None;
            }
        };
        if validated_generation.basename != basename
            || validated_generation.graph != graph
            || !HnswSourceIdentityV1::admits(
                validated_generation.source_identity.as_ref(),
                meta.source_identity.as_ref(),
            )
        {
            tracing::warn!(
                path = %path.display(),
                "HNSW metadata and digest receipt name different native sidecars; \
                 native load unavailable"
            );
            return None;
        }

        // Validate the SOURCE GENERATION IDENTITY before anything else about
        // the content (bd-r65a). Content equality is not identity equality:
        // two generations can hold byte-identical live vectors under the same
        // doc ids and still be different published artifacts — a
        // re-publication carries a new generation nonce, and a model revision
        // bump moves the space fingerprint without necessarily moving a single
        // vector. Every check below this one would pass in that case, so
        // ordering this first is what stops a stale graph being served.
        let live_identity = HnswSourceIdentityV1::capture(source_index);
        if !HnswSourceIdentityV1::admits(meta.source_identity.as_ref(), live_identity.as_ref()) {
            tracing::warn!(
                path = %path.display(),
                sidecar_bound = meta.source_identity.is_some(),
                source_bound = live_identity.is_some(),
                "HNSW sidecar is bound to a different FSVI generation identity than the live \
                 source; native load unavailable and the graph will be rebuilt"
            );
            return None;
        }

        // Validate doc-id sequence against the live VectorIndex *before*
        // touching the (potentially expensive) hnsw_rs load.
        if !meta_matches_live_doc_ids(meta, source_index).ok()? {
            tracing::warn!(
                path = %path.display(),
                "HNSW sidecar doc_ids disagree with live VectorIndex; native load unavailable"
            );
            return None;
        }

        // Validate the vector-content fingerprint against the live VectorIndex.
        // This is the critical stale-vectors guard: if a caller swaps the FSVI
        // contents while keeping the same doc IDs in the same order, the
        // persisted graph would otherwise silently serve hits against vectors
        // that no longer exist. `try_load_native_graph` is only called for the
        // current format, so a missing fingerprint cannot be treated as a
        // legacy exception: 0 is compared like any other digest value.
        let live_fp = fingerprint_vector_index_positions(
            source_index,
            &meta.source_positions,
            &meta.doc_ids,
            meta.dimension,
        )
        .ok()?;
        if live_fp != meta.vector_fingerprint {
            tracing::warn!(
                path = %path.display(),
                expected = meta.vector_fingerprint,
                actual = live_fp,
                "HNSW sidecar vector fingerprint disagrees with live VectorIndex \
                 (vectors swapped behind matching doc ids); native load unavailable"
            );
            return None;
        }

        // `HnswIo::load_hnsw` returns an `Hnsw` borrowed from the `HnswIo`
        // (`'a: 'b`), so to store it in the `'static` field we must keep the
        // `HnswIo` alive for the program. Load with `ReloadOptions::new(true)`
        // so the immutable `.hnsw.data` sidecar's vector payload stays
        // file-backed (read-only mmap) instead of being copied into a second
        // owned per-point heap: at multi-million-vector scale the owned copy
        // alone is gigabytes per cold process. Graph topology is still parsed
        // into owned state and attested below, and every admission check
        // (receipt, locality, doc-id/fingerprint binding) runs before the map
        // is created. Leaking the reloader (rather than a self-referential
        // struct or unsafe lifetime transmute) is the simplest sound way to
        // obtain a `'static` graph, keeps the mmap alive for exactly the graph
        // lifetime, and happens about once per process.
        let native_io: &'static mut HnswIo = Box::leak(Box::new(HnswIo::new_with_options(
            &sidecar_parent,
            &basename,
            ReloadOptions::new(true),
        )));
        let hnsw = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            // Moving the leaked mutable reference into a closure-local binding
            // makes this closure `FnOnce` and lets the parser's returned graph
            // carry the same `'static` lifetime as its deliberately leaked
            // owner.
            let native_io = native_io;
            native_io.load_hnsw::<'static, 'static, f32, DistDot>()
        })) {
            Ok(Ok(hnsw)) => hnsw,
            Ok(Err(error)) => {
                tracing::warn!(
                    path = %path.display(),
                    ?error,
                    "HNSW native sidecar parser rejected the installed generation; \
                     native load unavailable"
                );
                return None;
            }
            Err(_) => {
                tracing::warn!(
                    path = %path.display(),
                    "HNSW native sidecar parser panicked on the installed generation; \
                     native load unavailable"
                );
                return None;
            }
        };

        // Guard against a graph that doesn't match the metadata it shipped with
        // (e.g. truncated dump, mismatched sidecars). The caller additionally
        // validates doc_ids against the live VectorIndex.
        if hnsw.get_nb_point() != meta.doc_ids.len() {
            return None;
        }
        if let Err(detail) = validate_hnsw_topology(&hnsw, meta.doc_ids.len()) {
            tracing::warn!(
                path = %path.display(),
                %detail,
                "HNSW native sidecar failed topology attestation; native load unavailable"
            );
            return None;
        }

        Some(Self {
            hnsw,
            underfill_warned: AtomicBool::new(false),
            doc_ids: meta.doc_ids.clone(),
            source_positions: meta.source_positions.clone(),
            source_record_count: meta.source_record_count,
            dimension: meta.dimension,
            config: meta.config,
            vector_fingerprint: meta.vector_fingerprint,
            source_identity: meta.source_identity.clone(),
        })
    }

    /// Persist the ANN index to disk.
    ///
    /// Writes the JSON metadata sidecar (`doc_ids`, config, dimension) at
    /// `path`, plus the native `hnsw_rs` graph and data pair next to it. The
    /// metadata records the exact generation directory and basename returned by
    /// `hnsw_rs`. A new graph state dumps into a fresh immutable generation, so
    /// neither a newly built nor mmap-backed loaded graph can truncate the
    /// currently installed pair. Equivalent retries validate and reuse a
    /// durable READY generation left by an uncertain metadata publication.
    /// Vectors are embedded in the native data sidecar; the `VectorIndex` is
    /// only consulted on a legacy or fallback rebuild.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` on write/dump failure.
    pub fn save(&self, path: &Path) -> SearchResult<()> {
        self.save_with_metadata_publisher(path, publish_hnsw_metadata)
    }

    pub(super) fn save_with_metadata_publisher(
        &self,
        path: &Path,
        publish_metadata: HnswMetadataPublisher,
    ) -> SearchResult<()> {
        let parent = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;

        let requested_basename = hnsw_sidecar_basename(path)?;
        let metadata_file_name = hnsw_metadata_file_name(path)?;
        // Serialize writers before any generation is staged. The lock file
        // lives in a reserved sibling namespace and remains persistent because
        // removing or replacing it creates an inode race in which two processes
        // can each hold a different lock.
        let _save_lock = acquire_hnsw_save_lock(path)?;

        if let Some(meta) = find_reusable_hnsw_generation(
            self,
            path,
            parent,
            &requested_basename,
            &metadata_file_name,
        )? {
            // A receipt can survive a crash before the generation's parent
            // directory entry was durable. Re-sync the parent before making
            // metadata point at it.
            sync_hnsw_directory(parent)?;
            let metadata_bytes = serialize_hnsw_metadata(&meta)?;
            publish_metadata(path, parent, &metadata_bytes)?;
            gc_superseded_hnsw_generations(parent, &requested_basename, &meta)?;
            return Ok(());
        }

        // Persist into a unique generation first. `hnsw_rs` truncates an
        // occupied basename for freshly built graphs and uses a racy random
        // suffix for loaded graphs; an atomically created directory avoids both
        // behaviors. Metadata remains the sole commit point.
        let generation_prefix =
            hnsw_generation_prefix(&requested_basename, self.vector_fingerprint);
        let mut generation_builder = tempfile::Builder::new();
        generation_builder.prefix(&generation_prefix);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            // Tempfile defaults to a private 0700 directory. Published ANN
            // generations must instead inherit the caller's umask, just like
            // the FSVI artifact in the same parent directory.
            generation_builder.permissions(std::fs::Permissions::from_mode(0o777));
        }
        let generation = generation_builder
            .tempdir_in(parent)
            .map_err(SearchError::Io)?;
        let dumped_basename = self
            .hnsw
            .file_dump(generation.path(), &requested_basename)
            .map_err(|error| {
                SearchError::Io(std::io::Error::other(format!(
                    "failed to dump HNSW graph: {error}"
                )))
            })?;
        let dumped_basename = validate_hnsw_sidecar_basename(path, &dumped_basename)?;
        sync_hnsw_sidecars(generation.path(), &dumped_basename)?;
        let generation_name = generation
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| ann_corrupted(path, "HNSW generation has no UTF-8 directory name"))?;
        let generation_name = validate_hnsw_sidecar_basename(path, generation_name)?;
        let meta = self.metadata_for_generation(&generation_name, &dumped_basename);
        let graph_path = generation
            .path()
            .join(format!("{dumped_basename}.hnsw.graph"));
        let data_path = generation
            .path()
            .join(format!("{dumped_basename}.hnsw.data"));
        let receipt = HnswGenerationReceipt {
            receipt_version: HNSW_GENERATION_RECEIPT_VERSION,
            metadata_file_name,
            format_version: HNSW_META_FORMAT_CURRENT,
            generation: generation_name,
            sidecar_basename: dumped_basename,
            doc_count: self.doc_ids.len(),
            doc_ids_fingerprint: fingerprint_doc_ids(&self.doc_ids),
            vector_fingerprint: self.vector_fingerprint,
            source_map_fingerprint: fingerprint_source_map(
                &self.source_positions,
                self.source_record_count,
            ),
            dimension: self.dimension,
            config: self.config,
            source_identity: self.source_identity.clone(),
            graph: fingerprint_hnsw_sidecar(&graph_path)?,
            data: fingerprint_hnsw_sidecar(&data_path)?,
        };
        write_hnsw_generation_receipt(generation.path(), &receipt)?;

        // From this point onward every retained complete generation has a
        // durable receipt and is recoverable by a later save. Metadata remains
        // the atomic authority visible to readers.
        let _generation_path = generation.keep();
        sync_hnsw_directory(parent)?;

        let metadata_bytes = serialize_hnsw_metadata(&meta)?;
        publish_metadata(path, parent, &metadata_bytes)?;
        gc_superseded_hnsw_generations(parent, &requested_basename, &meta)?;

        Ok(())
    }

    pub(super) fn metadata_for_generation(&self, generation: &str, basename: &str) -> HnswMeta {
        HnswMeta {
            format_version: HNSW_META_FORMAT_CURRENT,
            doc_ids: self.doc_ids.clone(),
            source_positions: self.source_positions.clone(),
            source_record_count: self.source_record_count,
            config: self.config,
            dimension: self.dimension,
            vector_fingerprint: self.vector_fingerprint,
            sidecar_generation: Some(generation.to_owned()),
            sidecar_basename: Some(basename.to_owned()),
            source_identity: self.source_identity.clone(),
        }
    }
}
