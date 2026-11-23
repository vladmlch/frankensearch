use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use hnsw_rs::prelude::{DistDot, Hnsw, Neighbour};

use crate::VectorIndex;
use crate::recall_certificate::{EfCalibration, calibrate_certified_ef};

use super::fingerprint::{
    ann_corrupted, certified_ann_recall_sample, dist_dot_budget, estimate_recall,
    fingerprint_vector_index_positions, fingerprint_vectors, normalize_for_dist_dot,
    restore_dist_dot_score, source_map_matches_live_rows, validate_config,
};
use super::topology::{ann_topology_error, validate_hnsw_topology};
use super::types::{
    AnnFallbackReason, AnnSearchStats, DistDotBudget, HNSW_PARALLEL_INSERT_MIN_POINTS_PER_THREAD,
    HnswConfig, HnswHitResolution, HnswIndex, HnswSourceIdentityV1,
};

impl HnswIndex {
    /// Run ANN query and return hits plus query diagnostics.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs,
    /// and a subsystem error when a malformed/underfilled native graph cannot
    /// be repaired because no authoritative source was supplied.
    pub fn knn_search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        self.knn_search_with_optional_source(
            None,
            query,
            k,
            ef_search,
            HnswHitResolution::CanonicalPublic,
        )
    }

    /// Run ANN query with an authoritative `VectorIndex` available for exact
    /// underfill repair.
    ///
    /// The source is borrowed only for this query. It verifies the physical
    /// main-slab extent and returned row identities, filters post-build
    /// tombstones, and supplies an exact scan when the native graph underfills.
    /// This avoids retaining a second mmap or depending on a path that may be
    /// renamed/replaced after the caller opened the canonical index.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for incompatible vectors and
    /// rejects source-extent or returned-row identity divergence. The caller
    /// must supply the same immutable main slab used to build/load this graph;
    /// a full vector fingerprint is intentionally recomputed only before exact
    /// repair, avoiding an O(n) check on every successful ANN query.
    pub fn knn_search_with_stats_against(
        &self,
        source: &VectorIndex,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        self.knn_search_with_optional_source(
            Some(source),
            query,
            k,
            ef_search,
            HnswHitResolution::CanonicalPublic,
        )
    }

    pub(super) fn knn_search_with_optional_source(
        &self,
        source: Option<&VectorIndex>,
        query: &[f32],
        k: usize,
        ef_search: usize,
        resolution: HnswHitResolution,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        if let Some(source) = source
            && source.dimension() != self.dimension
        {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: source.dimension(),
            });
        }
        if let Some(source) = source
            && source.record_count() != self.source_record_count
        {
            return Err(ann_corrupted(
                &source.path,
                format!(
                    "canonical VectorIndex has {} physical rows, but HNSW was built against {}",
                    source.record_count(),
                    self.source_record_count
                ),
            ));
        }
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
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

        if k == 0 || self.doc_ids.is_empty() {
            // k = 0 and empty-graph are distinct zero-signal states and must
            // not collapse into one indistinguishable empty result (bd-tqhc).
            // k = 0 takes precedence: it is request-scoped.
            let zero_signal = if k == 0 {
                ZeroSignalReason::CallerRequestedZeroK
            } else {
                ZeroSignalReason::NewlyCreatedEmpty
            };
            let stats = AnnSearchStats {
                index_size: self.len(),
                dimension: self.dimension,
                ef_search,
                k_requested: k,
                k_returned: 0,
                search_time_us: 0,
                is_approximate: true,
                fallback_reason: None,
                estimated_recall: 1.0,
                zero_signal: Some(zero_signal),
            };
            return Ok((Vec::new(), stats));
        }

        let effective_k = k.min(self.doc_ids.len());
        let effective_ef = ef_search.max(effective_k).max(1);
        let budget = dist_dot_budget(self.dimension)?;
        let normalized_query = normalize_for_dist_dot(query.to_vec(), budget);

        let start = Instant::now();
        let neighbors = self
            .hnsw
            .search(&normalized_query, effective_k, effective_ef);
        self.finish_search_with_neighbors_resolution(
            source,
            query,
            k,
            effective_k,
            effective_ef,
            neighbors,
            budget,
            start,
            resolution,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish_search_with_neighbors(
        &self,
        source: Option<&VectorIndex>,
        canonical_query: &[f32],
        k_requested: usize,
        effective_k: usize,
        effective_ef: usize,
        neighbors: Vec<Neighbour>,
        budget: DistDotBudget,
        start: Instant,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        self.finish_search_with_neighbors_resolution(
            source,
            canonical_query,
            k_requested,
            effective_k,
            effective_ef,
            neighbors,
            budget,
            start,
            HnswHitResolution::CanonicalPublic,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_search_with_neighbors_resolution(
        &self,
        source: Option<&VectorIndex>,
        canonical_query: &[f32],
        k_requested: usize,
        effective_k: usize,
        effective_ef: usize,
        neighbors: Vec<Neighbour>,
        budget: DistDotBudget,
        start: Instant,
        resolution: HnswHitResolution,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        let (hits, fallback_reason) = self.resolve_neighbors_with_fallback(
            source,
            canonical_query,
            effective_k,
            neighbors,
            budget,
            resolution,
        )?;
        let search_time_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);

        // An empty result here means the exact repair also found nothing, so
        // classify from the authoritative source census. When the census says
        // usable vectors exist yet even exact repair produced nothing, that is
        // the ANN-availability anomaly rather than a benign state.
        let zero_signal = if hits.is_empty() {
            source.map(|source_index| {
                let state = source_index.zero_signal_state();
                state.state_reason().unwrap_or_else(|| {
                    if state.is_wal_only() {
                        ZeroSignalReason::WalOnlyNoLiveRecords
                    } else {
                        ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors
                    }
                })
            })
        } else {
            None
        };

        let stats = AnnSearchStats {
            index_size: self.len(),
            dimension: self.dimension,
            ef_search: effective_ef,
            k_requested,
            k_returned: hits.len(),
            search_time_us,
            is_approximate: fallback_reason.is_none(),
            fallback_reason,
            estimated_recall: if fallback_reason.is_some() {
                1.0
            } else {
                estimate_recall(effective_ef, effective_k)
            },
            zero_signal,
        };
        Ok((hits, stats))
    }

    fn resolve_neighbors_with_fallback(
        &self,
        source: Option<&VectorIndex>,
        canonical_query: &[f32],
        effective_k: usize,
        neighbors: Vec<Neighbour>,
        budget: DistDotBudget,
        resolution: HnswHitResolution,
    ) -> SearchResult<(Vec<VectorHit>, Option<AnnFallbackReason>)> {
        let native_returned = neighbors.len();
        let mut seen_neighbors = HashSet::with_capacity(native_returned);
        let mut eligible_neighbors = Vec::with_capacity(native_returned);
        for neighbor in neighbors {
            if !seen_neighbors.insert(neighbor.d_id) {
                continue;
            }
            if let Some(source) = source {
                let physical_position = self
                    .source_positions
                    .get(neighbor.d_id)
                    .copied()
                    .and_then(|position| usize::try_from(position).ok())
                    .ok_or_else(|| {
                        ann_corrupted(
                            &source.path,
                            format!(
                                "native neighbor {} has no compact-to-physical source row",
                                neighbor.d_id
                            ),
                        )
                    })?;
                if physical_position >= source.record_count() {
                    return Err(ann_corrupted(
                        &source.path,
                        format!(
                            "native neighbor {} maps to source row {physical_position}, \
                             beyond {} rows",
                            neighbor.d_id,
                            source.record_count()
                        ),
                    ));
                }
                let expected_doc_id = self.doc_ids.get(neighbor.d_id).ok_or_else(|| {
                    ann_corrupted(
                        &source.path,
                        format!("native neighbor {} has no document identity", neighbor.d_id),
                    )
                })?;
                if source.doc_id_at(physical_position)? != expected_doc_id {
                    return Err(ann_corrupted(
                        &source.path,
                        format!(
                            "native neighbor {} maps to source row {physical_position} with a \
                             different document identity",
                            neighbor.d_id
                        ),
                    ));
                }
                if source.is_deleted(physical_position) {
                    continue;
                }
            }
            eligible_neighbors.push(neighbor);
        }
        if eligible_neighbors.len() < effective_k {
            // First underfill on this graph instance warns; repeats log at
            // debug. Instances are per-generation, giving the required
            // once-per-generation warning bound (bd-tqhc).
            if self.underfill_warned.swap(true, Ordering::Relaxed) {
                tracing::debug!(
                    index_size = self.len(),
                    effective_k,
                    native_returned,
                    native_unique_live = eligible_neighbors.len(),
                    "HNSW search underfilled; returning exact top-k"
                );
            } else {
                tracing::warn!(
                    index_size = self.len(),
                    effective_k,
                    native_returned,
                    native_unique_live = eligible_neighbors.len(),
                    "HNSW search underfilled; returning exact top-k \
                     (repeat underfills for this graph instance log at debug)"
                );
            }
            let source = source.ok_or_else(|| SearchError::SubsystemError {
                subsystem: "hnsw",
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "HNSW search underfilled; call knn_search_with_stats_against with the \
                     authoritative VectorIndex to permit canonical exact repair",
                )),
            })?;
            let hits = self.exact_scan(source, canonical_query, effective_k, resolution)?;
            // A graph that yields zero candidates despite indexing points is a
            // stronger anomaly than a partial underfill; record it distinctly.
            let reason = if native_returned == 0 && !self.doc_ids.is_empty() {
                AnnFallbackReason::EmptyDespiteIndexedPoints
            } else {
                AnnFallbackReason::Underfilled
            };
            return Ok((hits, Some(reason)));
        }

        self.hits_from_distances(
            eligible_neighbors
                .into_iter()
                .map(|neighbor| (neighbor.d_id, neighbor.distance)),
            budget,
            resolution,
        )
        .map(|hits| (hits, None))
    }

    fn exact_scan(
        &self,
        source: &VectorIndex,
        canonical_query: &[f32],
        effective_k: usize,
        resolution: HnswHitResolution,
    ) -> SearchResult<Vec<VectorHit>> {
        if source.dimension() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: source.dimension(),
            });
        }
        let source_fingerprint = fingerprint_vector_index_positions(
            source,
            &self.source_positions,
            &self.doc_ids,
            self.dimension,
        )?;
        if source_fingerprint != self.vector_fingerprint {
            return Err(ann_corrupted(
                &source.path,
                "canonical VectorIndex source rows changed after HNSW construction",
            ));
        }
        match resolution {
            HnswHitResolution::CanonicalPublic => {
                source.search_main_top_k(canonical_query, effective_k)
            }
            HnswHitResolution::RawPhysical => {
                source.search_main_top_k_raw(canonical_query, effective_k)
            }
        }
    }

    fn hits_from_distances(
        &self,
        distances: impl IntoIterator<Item = (usize, f32)>,
        budget: DistDotBudget,
        resolution: HnswHitResolution,
    ) -> SearchResult<Vec<VectorHit>> {
        let scores = distances
            .into_iter()
            .map(|(neighbor_id, distance)| {
                restore_dist_dot_score(distance, budget).map(|score| (neighbor_id, score))
            })
            .collect::<SearchResult<Vec<_>>>()?;
        self.hits_from_scores(scores, resolution)
    }

    fn hits_from_scores(
        &self,
        scores: impl IntoIterator<Item = (usize, f32)>,
        resolution: HnswHitResolution,
    ) -> SearchResult<Vec<VectorHit>> {
        let mut hits = Vec::new();
        for (neighbor_id, score) in scores {
            let doc_id =
                self.doc_ids
                    .get(neighbor_id)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "neighbor_id".to_owned(),
                        value: neighbor_id.to_string(),
                        reason: "neighbor id exceeds doc_id table".to_owned(),
                    })?;
            let index = self
                .source_positions
                .get(neighbor_id)
                .copied()
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "neighbor_id".to_owned(),
                    value: neighbor_id.to_string(),
                    reason: "neighbor id exceeds compact-to-physical row map".to_owned(),
                })?;
            hits.push(VectorHit {
                index,
                // Graph and query vectors use the same deterministic radius.
                // Undo that uniform scale so callers continue receiving cosine
                // similarity rather than a dimension-dependent proxy score.
                // Clamp only after proving the deviation is inside the derived
                // floating-point envelope; materially invalid distances fail.
                score,
                doc_id: doc_id.as_str().into(),
            });
        }
        hits.sort_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        });
        if resolution == HnswHitResolution::CanonicalPublic {
            let mut seen_doc_ids = HashSet::with_capacity(hits.len());
            hits.retain(|hit| seen_doc_ids.insert(hit.doc_id.clone()));
        }
        Ok(hits)
    }

    /// Run ANN query and return only the hits.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs,
    /// and a subsystem error when a malformed/underfilled native graph cannot
    /// be repaired because no authoritative source was supplied.
    pub fn knn_search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.knn_search_with_stats(query, k, ef_search)
            .map(|(hits, _)| hits)
    }

    /// Run ANN query with an authoritative source available for exact
    /// underfill repair, returning only hits.
    ///
    /// # Errors
    ///
    /// Propagates dimension, source-identity, native-search, or exact-repair
    /// errors from [`Self::knn_search_with_stats_against`].
    pub fn knn_search_against(
        &self,
        source: &VectorIndex,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.knn_search_with_stats_against(source, query, k, ef_search)
            .map(|(hits, _)| hits)
    }

    /// Certify the cheapest `ef_search` whose recall meets `target` — the automated
    /// replacement for the human "recall-budget sign-off" that gated ANN-in-BOLD.
    ///
    /// Measures this ANN index's recall@`k` against exact bruteforce
    /// ([`VectorIndex::search_top_k`]) over `calibration_queries`, and returns the
    /// smallest `ef` in `candidate_efs` whose split-conformal recall **lower bound**
    /// is `≥ target` at confidence `1 − alpha` (distribution-free, finite-sample
    /// valid; see [`crate::recall_certificate`]). If none qualifies, returns the
    /// best-certifiable `ef` with `meets_target = false`.
    ///
    /// The exact top-k for each query is independent of `ef`, so it is computed
    /// **once per query** (not per `ef`); only the ANN search re-runs per candidate,
    /// and the sweep short-circuits at the first certified `ef`, so no ANN search is
    /// run at an `ef` larger than the chosen one.
    ///
    /// # Errors
    ///
    /// Propagates any error from the exact [`VectorIndex::search_top_k`] pass. A
    /// failed ANN search or exact-underfill fallback for a single (query, ef)
    /// is treated as recall `0.0` for that query. This conservative direction
    /// prevents calibration from selecting an `ef` whose apparent recall came
    /// from repeatedly abandoning ANN for a full scan.
    pub fn certify_ef_search(
        &self,
        exact_index: &VectorIndex,
        calibration_queries: &[Vec<f32>],
        candidate_efs: &[usize],
        k: usize,
        target: f64,
        alpha: f64,
    ) -> SearchResult<Option<EfCalibration>> {
        // Exact top-k is ef-independent: compute it once per calibration query.
        let exact: Vec<Vec<VectorHit>> = calibration_queries
            .iter()
            .map(|q| exact_index.search_top_k(q, k, None))
            .collect::<SearchResult<_>>()?;

        Ok(calibrate_certified_ef(
            candidate_efs,
            |ef| {
                calibration_queries
                    .iter()
                    .zip(&exact)
                    .map(|(q, exact_hits)| {
                        certified_ann_recall_sample(
                            self.knn_search_with_stats_against(exact_index, q, k, ef),
                            exact_hits,
                        )
                    })
                    .collect()
            },
            target,
            alpha,
        ))
    }

    /// Returns true when this ANN index maps exactly to a `VectorIndex` generation.
    ///
    /// Since `HnswIndex` no longer stores vectors, this checks:
    /// 1. Dimension match.
    /// 2. Every live physical row is mapped once with the correct `doc_id`.
    /// 3. The fingerprint of every live vector.
    ///
    /// # Errors
    ///
    /// Propagates decoding errors from `VectorIndex::doc_id_at`.
    pub fn matches_vector_index(&self, index: &VectorIndex) -> SearchResult<bool> {
        if self.dimension != index.dimension() {
            return Ok(false);
        }
        // Same generation-identity gate as the native load path (bd-r65a):
        // matching dimension, doc ids, ordering, and vector fingerprint prove
        // the CONTENT is the same, never that the artifact is.
        if !HnswSourceIdentityV1::admits(
            self.source_identity.as_ref(),
            HnswSourceIdentityV1::capture(index).as_ref(),
        ) {
            return Ok(false);
        }
        if !source_map_matches_live_rows(
            index,
            &self.doc_ids,
            &self.source_positions,
            self.source_record_count,
        )? {
            return Ok(false);
        }
        Ok(fingerprint_vector_index_positions(
            index,
            &self.source_positions,
            &self.doc_ids,
            self.dimension,
        )? == self.vector_fingerprint)
    }

    /// Number of indexed vectors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Whether ANN index has zero vectors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Vector dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// ANN configuration used to build this index.
    #[must_use]
    pub const fn config(&self) -> HnswConfig {
        self.config
    }

    pub(super) fn build_from_parts(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
        config: HnswConfig,
    ) -> SearchResult<Self> {
        validate_config(config)?;
        if dimension == 0 {
            return Err(SearchError::InvalidConfig {
                field: "dimension".to_owned(),
                value: "0".to_owned(),
                reason: "dimension must be greater than zero".to_owned(),
            });
        }
        let budget = dist_dot_budget(dimension)?;
        if doc_ids.len() != vectors.len() {
            return Err(SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: vectors.len().to_string(),
                reason: format!("doc_id count {} must match vector count", doc_ids.len()),
            });
        }
        // Fingerprint the raw (un-normalized) input vectors. This is what
        // VectorIndex::vector_at_f32 returns at load time, so a fresh load from
        // the live VectorIndex will produce the same digest iff the underlying
        // bytes are unchanged. Used by the native-graph load path to detect
        // "doc IDs match but vectors were silently swapped" and trigger a
        // rebuild (see try_load_native_graph).
        let vector_fingerprint = fingerprint_vectors(&doc_ids, &vectors);

        let mut normalized_vectors = Vec::with_capacity(vectors.len());
        for (idx, vector) in vectors.into_iter().enumerate() {
            if vector.len() != dimension {
                return Err(SearchError::DimensionMismatch {
                    expected: dimension,
                    found: vector.len(),
                });
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(SearchError::InvalidConfig {
                    field: "vector".to_owned(),
                    value: idx.to_string(),
                    reason: "all vector values must be finite".to_owned(),
                });
            }
            normalized_vectors.push(normalize_for_dist_dot(vector, budget));
        }

        let use_parallel_insert = should_use_parallel_insert(normalized_vectors.len());
        let initial_hnsw = construct_hnsw_graph(&normalized_vectors, config, use_parallel_insert);
        let hnsw = attest_or_rebuild_serial(
            initial_hnsw,
            use_parallel_insert,
            doc_ids.len(),
            |graph| validate_hnsw_topology(graph, doc_ids.len()),
            || construct_hnsw_graph(&normalized_vectors, config, false),
        )
        .map_err(|detail| ann_topology_error(&detail))?;

        Ok(Self {
            hnsw,
            underfill_warned: AtomicBool::new(false),
            doc_ids,
            source_positions: (0..normalized_vectors.len())
                .map(|position| {
                    u32::try_from(position).map_err(|_| SearchError::InvalidConfig {
                        field: "source_position".to_owned(),
                        value: position.to_string(),
                        reason: "HNSW row exceeds u32".to_owned(),
                    })
                })
                .collect::<SearchResult<Vec<_>>>()?,
            source_record_count: normalized_vectors.len(),
            dimension,
            config,
            vector_fingerprint,
            source_identity: None,
        })
    }
}

pub(super) fn should_use_parallel_insert(point_count: usize) -> bool {
    let remaining_points = point_count.saturating_sub(1);
    let minimum_parallel_batch =
        rayon::current_num_threads().saturating_mul(HNSW_PARALLEL_INSERT_MIN_POINTS_PER_THREAD);
    remaining_points >= minimum_parallel_batch
}

fn construct_hnsw_graph(
    vectors: &[Vec<f32>],
    config: HnswConfig,
    parallel: bool,
) -> Hnsw<'static, f32, DistDot> {
    let hnsw = Hnsw::new(
        config.m,
        vectors.len().max(1),
        config.max_layer,
        config.ef_construction,
        DistDot,
    );
    let Some((first, remaining)) = vectors.split_first() else {
        return hnsw;
    };

    // Seed the entry point serially. The upstream implementation admits a new
    // point into shared tables before reverse-edge updates are complete, so an
    // established entry point is required before any concurrent insertion.
    hnsw.insert((first, 0));
    if parallel {
        let vectors_with_ids = remaining
            .iter()
            .enumerate()
            .map(|(index, vector)| (vector, index.saturating_add(1)))
            .collect::<Vec<_>>();
        hnsw.parallel_insert(&vectors_with_ids);
    } else {
        for (index, vector) in remaining.iter().enumerate() {
            hnsw.insert((vector, index.saturating_add(1)));
        }
    }
    hnsw
}

pub(super) fn attest_or_rebuild_serial<T>(
    initial_graph: T,
    attempted_parallel: bool,
    point_count: usize,
    mut validate: impl FnMut(&T) -> Result<(), String>,
    rebuild_serial: impl FnOnce() -> T,
) -> Result<T, String> {
    let Err(initial_detail) = validate(&initial_graph) else {
        return Ok(initial_graph);
    };
    if !attempted_parallel {
        return Err(initial_detail);
    }

    // Never publish a graph whose concurrent construction violated the
    // searchable-topology contract. Drop it before allocating the replacement:
    // parallel mode is reserved for large indexes, where retaining both full
    // graphs could turn a recoverable topology fault into an OOM.
    tracing::warn!(
        point_count,
        parallel_error = %initial_detail,
        "parallel HNSW construction failed topology attestation; rebuilding serially"
    );
    drop(initial_graph);

    let serial_graph = rebuild_serial();
    if let Err(serial_detail) = validate(&serial_graph) {
        return Err(format!(
            "parallel construction failed ({initial_detail}); \
             serial rebuild also failed ({serial_detail})"
        ));
    }
    Ok(serial_graph)
}
