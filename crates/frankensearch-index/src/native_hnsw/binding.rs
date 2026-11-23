use std::path::Path;
use std::sync::Arc;

use crate::ValidatedFsviBytes;
use frankensearch_core::{
    error::{SearchError, SearchResult},
    generation::ArtifactGenerationIdentityV1,
};

use super::NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2;
use super::codec::NativeHnsw;
use super::io_helpers::{
    canonical_graph_basename, encode_lower_hex, native_hnsw_corrupted,
    native_hnsw_receipt_config_error, native_hnsw_receipt_corrupted, persist_native_hnsw_receipt,
    read_regular_file_bytes, reject_non_regular_destination,
    reject_non_regular_receipt_destination, reject_symlink_ancestors, sha256_hex, usize_to_u64,
    validate_sha256_hex,
};
use super::receipt::{NativeHnswGenerationReceiptV2, native_hnsw_generation_receipt_path};
use super::types::{
    GraphDefect, HnswParams, NativeHnswFileMetadata, NativeHnswParamsIdentityV1,
    ValidatedNativeHnsw, ValidatedNativeHnswHit, widen_search_width,
};

/// Internal exact-FSVI trust material for an owner-bound native HNSW sidecar.
///
/// This type is deliberately private. Public callers can create, persist,
/// reopen, and search a receipted graph only through [`ValidatedNativeHnsw`],
/// whose retained owner and private fields keep the graph paired with the
/// admitted bytes that supplied this trust material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NativeHnswGenerationBindingV2 {
    pub(super) artifact_generation: ArtifactGenerationIdentityV1,
    pub(super) artifact_generation_fingerprint: String,
    pub(super) embedding_identity_fingerprint: String,
    pub(super) embedding_space_fingerprint: String,
    pub(super) embedding_producer_fingerprint: String,
    pub(super) embedding_input_fingerprint: String,
    pub(super) vector_storage_fingerprint: String,
    pub(super) vector_content_digest: String,
    pub(super) ordered_live_docset_digest: String,
    pub(super) fsvi_whole_image_sha256: String,
    pub(super) fsvi_physical_row_count: u64,
}

impl NativeHnswGenerationBindingV2 {
    pub(super) fn from_validated_fsvi(owner: &ValidatedFsviBytes) -> SearchResult<Self> {
        let identity = owner.identity_v2();
        let fsvi_physical_row_count =
            usize_to_u64(owner.record_count(), "binding.fsvi_physical_row_count")?;

        let binding = Self {
            artifact_generation: identity.generation,
            artifact_generation_fingerprint: encode_lower_hex(identity.generation_fingerprint),
            embedding_identity_fingerprint: encode_lower_hex(identity.identity_bundle_fingerprint),
            embedding_space_fingerprint: encode_lower_hex(identity.space_fingerprint),
            embedding_producer_fingerprint: encode_lower_hex(identity.producer_fingerprint),
            embedding_input_fingerprint: encode_lower_hex(identity.input_fingerprint),
            vector_storage_fingerprint: encode_lower_hex(identity.storage_fingerprint),
            vector_content_digest: encode_lower_hex(identity.vector_content_digest),
            ordered_live_docset_digest: encode_lower_hex(identity.ordered_live_docset_digest),
            fsvi_whole_image_sha256: encode_lower_hex(owner.witness().whole_image_sha256),
            fsvi_physical_row_count,
        };
        binding.validate_fingerprints()?;
        Ok(binding)
    }

    /// Save `graph` atomically, then publish its canonical adjacent receipt.
    ///
    /// Both destination paths are preflighted before the graph is touched.
    /// A crash between the two atomic renames can leave a new graph with a
    /// missing or stale receipt, but that state fails closed on every load.
    ///
    /// # Errors
    ///
    /// Returns the same graph errors as [`NativeHnsw::save`], plus
    /// [`SearchError::InvalidConfig`] for a noncanonical graph path and
    /// [`SearchError::Io`] for receipt persistence failures.
    pub(super) fn save_bound_graph(
        &self,
        graph: &NativeHnsw,
        graph_path: &Path,
    ) -> SearchResult<NativeHnswGenerationReceiptV2> {
        let graph_point_count = usize_to_u64(graph.len(), "graph.point_count")?;
        if graph_point_count != self.fsvi_physical_row_count {
            return Err(native_hnsw_receipt_config_error(
                "graph.point_count",
                &graph_point_count.to_string(),
                "must equal the admitted FSVI physical row count",
            ));
        }
        let receipt_path = native_hnsw_generation_receipt_path(graph_path)?;
        reject_symlink_ancestors(graph_path)?;
        reject_non_regular_destination(graph_path)?;
        reject_non_regular_receipt_destination(&receipt_path)?;

        let metadata = graph.save(graph_path)?;
        let graph_bytes = read_regular_file_bytes(graph_path, "native HNSW artifact")?;
        if u64::try_from(graph_bytes.len()).ok() != Some(metadata.byte_len()) {
            return Err(native_hnsw_corrupted(
                graph_path,
                "native HNSW file length changed after atomic publication",
            ));
        }
        let mut receipt = NativeHnswGenerationReceiptV2 {
            schema_version: NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2,
            artifact_generation: self.artifact_generation,
            artifact_generation_fingerprint: self.artifact_generation_fingerprint.clone(),
            embedding_identity_fingerprint: self.embedding_identity_fingerprint.clone(),
            embedding_space_fingerprint: self.embedding_space_fingerprint.clone(),
            embedding_producer_fingerprint: self.embedding_producer_fingerprint.clone(),
            embedding_input_fingerprint: self.embedding_input_fingerprint.clone(),
            vector_storage_fingerprint: self.vector_storage_fingerprint.clone(),
            vector_content_digest: self.vector_content_digest.clone(),
            ordered_live_docset_digest: self.ordered_live_docset_digest.clone(),
            fsvi_whole_image_sha256: self.fsvi_whole_image_sha256.clone(),
            fsvi_physical_row_count: self.fsvi_physical_row_count,
            graph_basename: canonical_graph_basename(graph_path)?,
            graph_byte_len: metadata.byte_len(),
            graph_sha256: sha256_hex(&graph_bytes),
            native_format_version: metadata.format_version(),
            params: NativeHnswParamsIdentityV1::from_params(graph.params())?,
            seed: graph.seed(),
            point_count: metadata.point_count(),
            entry_point: graph.entry_point(),
            max_level: usize_to_u64(graph.max_level(), "receipt.max_level")?,
            payload_crc32: metadata.payload_crc32(),
            header_crc32: metadata.header_crc32(),
            topology_sha256: graph.topology_sha256()?,
            receipt_sha256: String::new(),
        };
        receipt.seal()?;
        let encoded = receipt.to_bytes()?;
        persist_native_hnsw_receipt(&receipt_path, &encoded)?;
        Ok(receipt)
    }

    /// Load a graph only after its adjacent receipt proves the exact expected
    /// FSVI generation, embedding identity, file bytes, and topology.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexNotFound`] when either artifact is missing,
    /// [`SearchError::IndexCorrupted`] for malformed, stale, replaced, or
    /// mismatched receipt/graph material, and [`SearchError::Io`] for other
    /// filesystem failures.
    pub(super) fn load_bound_graph(
        &self,
        graph_path: &Path,
        owner: &ValidatedFsviBytes,
    ) -> SearchResult<(NativeHnsw, NativeHnswGenerationReceiptV2)> {
        self.load_bound_graph_with_after_first_observation(graph_path, owner, || Ok(()))
    }

    pub(super) fn load_bound_graph_with_after_first_observation(
        &self,
        graph_path: &Path,
        owner: &ValidatedFsviBytes,
        after_first_observation: impl FnOnce() -> SearchResult<()>,
    ) -> SearchResult<(NativeHnsw, NativeHnswGenerationReceiptV2)> {
        let observed_binding = Self::from_validated_fsvi(owner)?;
        if observed_binding != *self {
            return Err(native_hnsw_receipt_config_error(
                "owner",
                "redacted-witness-mismatch",
                "must be the exact admitted FSVI owner that created this binding",
            ));
        }
        let receipt_path = native_hnsw_generation_receipt_path(graph_path)?;
        reject_symlink_ancestors(graph_path)?;
        let receipt_bytes =
            read_regular_file_bytes(&receipt_path, "native HNSW generation receipt")?;
        let receipt = NativeHnswGenerationReceiptV2::from_bytes(&receipt_bytes, &receipt_path)?;
        self.validate_receipt_identity(&receipt, graph_path, &receipt_path)?;

        let graph_bytes_before = read_regular_file_bytes(graph_path, "native HNSW artifact")?;
        Self::validate_graph_bytes(&receipt, graph_path, &graph_bytes_before)?;
        after_first_observation()?;
        let (graph, metadata) = NativeHnsw::load_with_metadata(graph_path, owner)?;
        Self::validate_graph_identity(&receipt, graph_path, &graph, metadata)?;

        // A second exact byte observation closes the replacement window
        // between the cryptographic check and the owned graph parser.
        let graph_bytes_after = read_regular_file_bytes(graph_path, "native HNSW artifact")?;
        // ubs:ignore — public artifact bytes are integrity material, not a secret.
        if graph_bytes_after != graph_bytes_before {
            return Err(native_hnsw_corrupted(
                graph_path,
                "native HNSW artifact changed during receipt verification",
            ));
        }
        Ok((graph, receipt))
    }

    fn validate_fingerprints(&self) -> SearchResult<()> {
        for (field, fingerprint) in [
            (
                "artifact_generation_fingerprint",
                &self.artifact_generation_fingerprint,
            ),
            (
                "embedding_identity_fingerprint",
                &self.embedding_identity_fingerprint,
            ),
            (
                "embedding_space_fingerprint",
                &self.embedding_space_fingerprint,
            ),
            (
                "embedding_producer_fingerprint",
                &self.embedding_producer_fingerprint,
            ),
            (
                "embedding_input_fingerprint",
                &self.embedding_input_fingerprint,
            ),
            (
                "vector_storage_fingerprint",
                &self.vector_storage_fingerprint,
            ),
            ("vector_content_digest", &self.vector_content_digest),
            (
                "ordered_live_docset_digest",
                &self.ordered_live_docset_digest,
            ),
            ("fsvi_whole_image_sha256", &self.fsvi_whole_image_sha256),
        ] {
            validate_sha256_hex(field, fingerprint)?;
        }
        // ubs:ignore — generation fingerprints are public artifact identities.
        if self.artifact_generation_fingerprint != self.artifact_generation.fingerprint() {
            return Err(native_hnsw_receipt_config_error(
                "artifact_generation_fingerprint",
                "redacted-digest-mismatch",
                "must match the admitted FSVI generation",
            ));
        }
        Ok(())
    }

    fn validate_receipt_identity(
        &self,
        receipt: &NativeHnswGenerationReceiptV2,
        graph_path: &Path,
        receipt_path: &Path,
    ) -> SearchResult<()> {
        let mismatch = |field: &str| {
            native_hnsw_receipt_corrupted(
                receipt_path,
                format!("native HNSW receipt {field} mismatch"),
            )
        };
        // ubs:ignore — generation identity is public artifact metadata, not a credential.
        if receipt.artifact_generation != self.artifact_generation {
            return Err(mismatch("artifact generation"));
        }
        // ubs:ignore — this public SHA-256 is an integrity fingerprint, not a MAC or secret.
        if receipt.artifact_generation_fingerprint != self.artifact_generation_fingerprint {
            return Err(mismatch("artifact generation fingerprint"));
        }
        for (field, actual, expected) in [
            (
                "embedding identity fingerprint",
                &receipt.embedding_identity_fingerprint,
                &self.embedding_identity_fingerprint,
            ),
            (
                "embedding space fingerprint",
                &receipt.embedding_space_fingerprint,
                &self.embedding_space_fingerprint,
            ),
            (
                "embedding producer fingerprint",
                &receipt.embedding_producer_fingerprint,
                &self.embedding_producer_fingerprint,
            ),
            (
                "embedding input fingerprint",
                &receipt.embedding_input_fingerprint,
                &self.embedding_input_fingerprint,
            ),
            (
                "vector storage fingerprint",
                &receipt.vector_storage_fingerprint,
                &self.vector_storage_fingerprint,
            ),
            (
                "vector content digest",
                &receipt.vector_content_digest,
                &self.vector_content_digest,
            ),
            (
                "ordered live-docset digest",
                &receipt.ordered_live_docset_digest,
                &self.ordered_live_docset_digest,
            ),
            (
                "FSVI whole-image SHA-256",
                &receipt.fsvi_whole_image_sha256,
                &self.fsvi_whole_image_sha256,
            ),
        ] {
            // ubs:ignore — frozen identity fingerprints are public compatibility metadata.
            if actual != expected {
                return Err(mismatch(field));
            }
        }
        if receipt.fsvi_physical_row_count != self.fsvi_physical_row_count {
            return Err(mismatch("FSVI physical row count"));
        }
        if receipt.graph_basename != canonical_graph_basename(graph_path)? {
            return Err(mismatch("graph basename"));
        }
        Ok(())
    }

    fn validate_graph_bytes(
        receipt: &NativeHnswGenerationReceiptV2,
        graph_path: &Path,
        graph_bytes: &[u8],
    ) -> SearchResult<()> {
        let actual_len = u64::try_from(graph_bytes.len()).map_err(|_| {
            native_hnsw_corrupted(graph_path, "native HNSW file length does not fit u64")
        })?;
        if actual_len != receipt.graph_byte_len {
            return Err(native_hnsw_corrupted(
                graph_path,
                "native HNSW file length disagrees with generation receipt",
            ));
        }
        // ubs:ignore — the graph SHA-256 is public integrity metadata, not authentication.
        if sha256_hex(graph_bytes) != receipt.graph_sha256 {
            return Err(native_hnsw_corrupted(
                graph_path,
                "native HNSW SHA-256 disagrees with generation receipt",
            ));
        }
        Ok(())
    }

    fn validate_graph_identity(
        receipt: &NativeHnswGenerationReceiptV2,
        graph_path: &Path,
        graph: &NativeHnsw,
        metadata: NativeHnswFileMetadata,
    ) -> SearchResult<()> {
        let mismatch = |field: &str| {
            native_hnsw_corrupted(
                graph_path,
                format!("native HNSW {field} disagrees with generation receipt"),
            )
        };
        if metadata.format_version() != receipt.native_format_version {
            return Err(mismatch("format version"));
        }
        if metadata.byte_len() != receipt.graph_byte_len {
            return Err(mismatch("byte length"));
        }
        // ubs:ignore — point counts are public graph-cardinality evidence, not secrets.
        if metadata.point_count() != receipt.point_count {
            return Err(mismatch("point count"));
        }
        if metadata.payload_crc32() != receipt.payload_crc32 {
            return Err(mismatch("payload CRC"));
        }
        if metadata.header_crc32() != receipt.header_crc32 {
            return Err(mismatch("header CRC"));
        }
        if graph.params() != receipt.params.to_params()? {
            return Err(mismatch("parameter identity"));
        }
        // ubs:ignore — this deterministic topology seed is public receipt metadata.
        if graph.seed() != receipt.seed {
            return Err(mismatch("level-sampling seed"));
        }
        if graph.entry_point() != receipt.entry_point {
            return Err(mismatch("entry point"));
        }
        if usize_to_u64(graph.max_level(), "verified.max_level")? != receipt.max_level {
            return Err(mismatch("maximum level"));
        }
        // ubs:ignore — the topology SHA-256 is public integrity metadata.
        if graph.topology_sha256()? != receipt.topology_sha256 {
            return Err(mismatch("topology SHA-256"));
        }
        Ok(())
    }
}

impl ValidatedNativeHnsw {
    pub(super) fn binding_for_owner(
        owner: &ValidatedFsviBytes,
    ) -> SearchResult<NativeHnswGenerationBindingV2> {
        NativeHnswGenerationBindingV2::from_validated_fsvi(owner)
    }

    pub(super) fn from_verified_graph(
        owner: Arc<ValidatedFsviBytes>,
        binding: NativeHnswGenerationBindingV2,
        graph: NativeHnsw,
    ) -> SearchResult<Self> {
        let owner_binding = Self::binding_for_owner(owner.as_ref())?;
        // ubs:ignore -- these are public artifact identity digests, not credentials.
        if binding != owner_binding {
            return Err(native_hnsw_receipt_config_error(
                "generation_binding",
                "redacted-mismatch",
                "must be derived from the exact retained FSVI owner",
            ));
        }
        graph
            .verify_for_store(owner.as_ref())
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "native-hnsw",
                source: Box::new(source),
            })?;
        Ok(Self {
            owner,
            binding,
            graph,
        })
    }

    /// Build a graph from every physical row of one admitted FSVI owner.
    ///
    /// The supplied [`Arc`] is retained by the returned handle. Callers may
    /// drop every other owner and the original source pathname immediately
    /// after this method returns without detaching the graph from its bytes.
    ///
    /// Tombstoned rows remain in the graph as routing nodes. They retain their
    /// exact physical-row identity in topology and receipt material, but
    /// [`Self::search`] never exposes them as hits.
    ///
    /// # Errors
    ///
    /// Returns identity/cardinality errors from receipt binding construction
    /// or distance-computation errors from the admitted owner, and a
    /// `native-hnsw` [`SearchError::SubsystemError`] if post-build structural
    /// attestation rejects the graph.
    pub fn build(
        owner: Arc<ValidatedFsviBytes>,
        params: HnswParams,
        seed: u64,
    ) -> SearchResult<Self> {
        let binding = Self::binding_for_owner(owner.as_ref())?;
        let graph = NativeHnsw::build(params, seed, owner.as_ref())?;
        Self::from_verified_graph(owner, binding, graph)
    }

    /// Load a graph only after its receipt matches the exact admitted owner.
    ///
    /// The supplied [`Arc`] is retained by the returned handle. Receipt
    /// verification, structural attestation, and every later search therefore
    /// resolve rows from the same allocation even after the caller drops its
    /// own reference.
    ///
    /// Receipt identity is checked before graph bytes are opened, and graph
    /// parsing and structural verification use the same retained owner.
    /// The exact whole-image witness binds the LIVE/TOMBSTONE layout, while
    /// every physical row remains present in the loaded routing topology.
    ///
    /// # Errors
    ///
    /// Returns receipt, graph, filesystem, or owner-distance errors without
    /// falling back to a caller-supplied vector store.
    pub fn load(
        owner: Arc<ValidatedFsviBytes>,
        graph_path: &Path,
    ) -> SearchResult<(Self, NativeHnswGenerationReceiptV2)> {
        let binding = Self::binding_for_owner(owner.as_ref())?;
        let (graph, receipt) = binding.load_bound_graph(graph_path, owner.as_ref())?;
        Ok((
            Self {
                owner,
                binding,
                graph,
            },
            receipt,
        ))
    }

    #[cfg(test)]
    pub(super) fn load_with_after_first_observation(
        owner: Arc<ValidatedFsviBytes>,
        graph_path: &Path,
        after_first_observation: impl FnOnce() -> SearchResult<()>,
    ) -> SearchResult<(Self, NativeHnswGenerationReceiptV2)> {
        let binding = Self::binding_for_owner(owner.as_ref())?;
        let (graph, receipt) = binding.load_bound_graph_with_after_first_observation(
            graph_path,
            owner.as_ref(),
            after_first_observation,
        )?;
        Ok((
            Self {
                owner,
                binding,
                graph,
            },
            receipt,
        ))
    }

    /// Durably publish this owner-built graph, then its adjacent receipt.
    ///
    /// The two renames are not a single transaction: a crash or concurrent
    /// writer between them can leave a mismatched pair, which every load
    /// rejects. Production publication must therefore use a serialized,
    /// generation-specific staging path and select the generation only after
    /// both artifacts are sealed.
    ///
    /// # Errors
    ///
    /// Returns graph, receipt, path-hardening, or filesystem errors. No API
    /// accepts a replacement graph at this boundary.
    pub fn save(&self, graph_path: &Path) -> SearchResult<NativeHnswGenerationReceiptV2> {
        self.binding.save_bound_graph(&self.graph, graph_path)
    }

    /// Search only against the owner used to build or load this graph.
    ///
    /// Every physical row remains eligible for graph traversal, including the
    /// entry point and intermediate tombstones. Returned rows are filtered
    /// through the same owner and are always LIVE. If the initial ANN window
    /// contains too many tombstones, both candidate count and beam width widen
    /// deterministically, never beyond the physical row count, until exactly
    /// `min(k, owner.live_count())` hits are available. A full-width underfill
    /// is a typed structural defect rather than a silent short result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::DimensionMismatch`] when the query does not
    /// match the admitted owner's dimension, [`SearchError::InvalidConfig`]
    /// for non-finite query values, or graph-distance and owner row-resolution
    /// errors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
    ) -> SearchResult<Vec<ValidatedNativeHnswHit<'_>>> {
        if query.len() != self.owner.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.owner.dimension(),
                found: query.len(),
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "query".to_owned(),
                value: "non-finite".to_owned(),
                reason: "all query vector values must be finite".to_owned(),
            });
        }
        let owner = self.owner.as_ref();
        let expected_live_hits = k.min(owner.live_count());
        // ubs:ignore -- cardinalities are public index metadata, not secrets.
        if expected_live_hits == 0 {
            return Ok(Vec::new());
        }

        let physical_rows = owner.record_count();
        let mut candidate_count = expected_live_hits.min(physical_rows);
        let mut search_ef = ef
            .unwrap_or_else(|| self.graph.params().ef_search)
            .max(candidate_count)
            .min(physical_rows);

        loop {
            let candidates = self
                .graph
                .search(query, candidate_count, Some(search_ef), owner)?;
            let mut live_hits = Vec::with_capacity(expected_live_hits);
            for (physical_row, distance) in candidates {
                let physical_index =
                    usize::try_from(physical_row).map_err(|_| SearchError::InvalidConfig {
                        field: "native_hnsw.physical_row".to_owned(),
                        value: physical_row.to_string(),
                        reason: "physical row does not fit usize".to_owned(),
                    })?;
                let flags = owner.row(physical_index)?.flags();
                if flags.is_tombstone() {
                    continue;
                }
                let doc_id = owner.doc_id_at(physical_index)?;
                live_hits.push(ValidatedNativeHnswHit {
                    physical_row,
                    distance,
                    doc_id,
                    flags,
                });
                // ubs:ignore -- result cardinalities are not secret material.
                if live_hits.len() == expected_live_hits {
                    return Ok(live_hits);
                }
            }

            // ubs:ignore -- search widths are public index cardinalities.
            if candidate_count == physical_rows {
                return Err(SearchError::SubsystemError {
                    subsystem: "native-hnsw",
                    source: Box::new(GraphDefect::LiveResultUnderfill {
                        requested_k: k,
                        expected_live_hits,
                        returned_live_hits: live_hits.len(),
                        physical_rows,
                    }),
                });
            }

            candidate_count = widen_search_width(candidate_count, physical_rows);
            search_ef = widen_search_width(search_ef, physical_rows).max(candidate_count);
        }
    }

    /// Number of physical owner rows indexed by this graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Whether the bound graph and owner are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Exact admitted FSVI witness paired with this graph.
    #[must_use]
    pub fn owner_witness(&self) -> &crate::FsviV2Witness {
        self.owner.witness()
    }
}
