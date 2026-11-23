use std::collections::BinaryHeap;

use ahash::AHashSet;
use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::filter::{ExcludeDocIdsFilter, SearchFilter};
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use rayon::slice::ParallelSliceMut;

use crate::VectorIndex;

use super::{
    ClassifiedHits, HeapEntry, PAR_SORT_THRESHOLD, PARALLEL_CHUNK_SIZE, PARALLEL_THRESHOLD,
    SearchParams, compare_best_first, parallel_search_enabled,
};

impl VectorIndex {
    /// Brute-force cosine-similarity top-k search over all records.
    ///
    /// The query is expected to already be normalized for cosine similarity.
    /// The result is sorted by descending score with NaN-safe semantics.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, and `SearchError::IndexCorrupted` for
    /// malformed vector slab contents.
    pub fn search_top_k(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_internal(
            query,
            limit,
            filter,
            PARALLEL_THRESHOLD,
            PARALLEL_CHUNK_SIZE,
            parallel_search_enabled(),
        )
    }

    /// Brute-force top-k with typed zero-signal classification.
    ///
    /// Behaves like [`Self::search_top_k`] with two fail-closed differences
    /// that align the exact path with the ANN path:
    /// - a query containing NaN or infinite components is rejected with
    ///   [`SearchError::InvalidConfig`] instead of silently scoring garbage
    ///   (parity with `HnswIndex`);
    /// - an empty result always carries a typed
    ///   [`ZeroSignalReason`], so a legitimate empty answer is
    ///   distinguishable from an unusable semantic lane.
    ///
    /// Classification is lazy: the non-empty path costs nothing extra, and
    /// an empty result pays one census pass comparable to the scan that
    /// just ran.
    ///
    /// # Errors
    ///
    /// Everything [`Self::search_top_k`] returns, plus
    /// [`SearchError::InvalidConfig`] for non-finite query vectors.
    pub fn search_top_k_classified(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<ClassifiedHits> {
        self.ensure_query_dimension(query)?;
        if limit == 0 {
            return Ok(ClassifiedHits::empty(
                ZeroSignalReason::CallerRequestedZeroK,
            ));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "query".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "query vector must be finite".to_owned(),
            });
        }
        if query.iter().all(|&value| value == 0.0) {
            return Ok(ClassifiedHits::empty(ZeroSignalReason::ZeroNormQuery));
        }
        let hits = self.search_top_k(query, limit, filter)?;
        if hits.is_empty() {
            let reason = self.classify_empty_result(filter.is_some());
            return Ok(ClassifiedHits {
                hits,
                zero_signal: Some(reason),
            });
        }
        Ok(ClassifiedHits {
            hits,
            zero_signal: None,
        })
    }

    /// Classify why a well-formed search (k > 0, finite non-zero query)
    /// returned nothing, following the precedence documented on
    /// [`ZeroSignalReason`].
    pub(crate) fn classify_empty_result(&self, had_filter: bool) -> ZeroSignalReason {
        self.zero_signal_state().empty_result_reason(had_filter)
    }

    /// Exact top-k over the persisted main slab only.
    ///
    /// HNSW uses this crate-private lane to repair a native underfill with the
    /// same quantization decoder, dot-product implementation, tombstone
    /// semantics, score ordering, physical row identities, and post-top-k
    /// document-ID deduplication as canonical `VectorIndex` search. Resident
    /// WAL entries and their supersession rules are deliberately excluded:
    /// `TwoTierIndex` merges them exactly once after ANN candidate retrieval.
    #[cfg(feature = "ann")]
    pub(crate) fn search_main_top_k(
        &self,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        let mut hits = self.search_main_top_k_raw(query, limit)?;
        let mut seen = AHashSet::with_capacity(hits.len());
        hits.retain(|hit| seen.insert(hit.doc_id.clone()));
        Ok(hits)
    }

    /// Raw physical top-k over the persisted main slab only.
    ///
    /// Unlike [`Self::search_main_top_k`], this does not apply document-ID
    /// deduplication. Neither main-only lane applies resident-WAL supersession.
    /// `TwoTierIndex` uses this raw lane only when an ANN underfill must be
    /// repaired before main and WAL candidates are ranked together through the
    /// canonical result resolver.
    #[cfg(feature = "ann")]
    pub(crate) fn search_main_top_k_raw(
        &self,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        let heap = self.scan_main_top_k_heap(query, limit)?;
        let mut winners = heap.into_vec();
        winners.sort_unstable_by(compare_best_first);
        winners
            .into_iter()
            .map(|winner| {
                let index =
                    u32::try_from(winner.index).map_err(|_| SearchError::InvalidConfig {
                        field: "index".to_owned(),
                        value: winner.index.to_string(),
                        reason: "winner index exceeds u32 range for VectorHit".to_owned(),
                    })?;
                Ok(VectorHit {
                    index,
                    score: winner.score,
                    doc_id: self.doc_id_at(winner.index)?.into(),
                })
            })
            .collect()
    }

    #[cfg(feature = "ann")]
    fn scan_main_top_k_heap(
        &self,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        self.ensure_query_dimension(query)?;
        if limit == 0 || self.record_count() == 0 {
            return Ok(BinaryHeap::new());
        }
        if parallel_search_enabled() && self.record_count() >= PARALLEL_THRESHOLD {
            self.scan_parallel(query, limit, None, PARALLEL_CHUNK_SIZE)
        } else {
            self.scan_sequential(query, limit, None)
        }
    }

    /// Brute-force cosine-similarity top-k search with configurable parallelism.
    ///
    /// Behaves identically to [`search_top_k`](Self::search_top_k) but uses the
    /// caller-supplied [`SearchParams`] instead of the compiled-in defaults.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, and `SearchError::IndexCorrupted` for
    /// malformed vector slab contents.
    pub fn search_top_k_with_params(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_internal(
            query,
            limit,
            filter,
            params.parallel_threshold,
            params.parallel_chunk_size,
            params.parallel_enabled,
        )
    }

    /// Bench-only: force the old per-document filtered scan, bypassing the
    /// selective-filter gather fast-path.
    #[doc(hidden)]
    pub fn bench_scan_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.ensure_query_dimension(query)?;
        let has_main = self.record_count() > 0;
        let has_wal = !self.wal_entries.is_empty();
        if limit == 0 || (!has_main && !has_wal) {
            return Ok(Vec::new());
        }
        let use_parallel = parallel_search_enabled() && self.record_count() >= PARALLEL_THRESHOLD;
        let mut heap = if has_main {
            if use_parallel {
                self.scan_parallel(query, limit, filter, PARALLEL_CHUNK_SIZE)?
            } else {
                self.scan_sequential(query, limit, filter)?
            }
        } else {
            BinaryHeap::with_capacity(limit.min(self.wal_entries.len()).saturating_add(1))
        };
        if has_wal {
            heap = self.merge_wal_top_k(query, heap, limit, filter, SearchParams::default())?;
        }
        self.resolve_hits(heap)
    }

    /// Bench-only: force the selective-filter gather path, ignoring the production
    /// selectivity gate.
    #[doc(hidden)]
    pub fn bench_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: &dyn SearchFilter,
    ) -> SearchResult<Vec<VectorHit>> {
        self.ensure_query_dimension(query)?;
        if limit == 0 || (self.record_count() == 0 && self.wal_entries.is_empty()) {
            return Ok(Vec::new());
        }
        let Some(allowed) = filter.candidate_hashes() else {
            return self.bench_scan_filtered(query, limit, Some(filter));
        };
        let mut heap = if self.record_count() > 0 {
            self.scan_gather_hashes(allowed, query, limit)?
        } else {
            BinaryHeap::with_capacity(limit.min(self.wal_entries.len()).saturating_add(1))
        };
        if !self.wal_entries.is_empty() {
            heap =
                self.merge_wal_top_k(query, heap, limit, Some(filter), SearchParams::default())?;
        }
        self.resolve_hits(heap)
    }

    // The input heap contains main-slab candidates only. Ordinary appends
    // tombstone superseded rows, but a durable WAL can survive a crash before
    // that tombstone reaches the main file. Dropping stale winners only in
    // resolve_sorted_entries is too late: they already displaced live rows.
    fn merge_wal_top_k(
        &self,
        query: &[f32],
        mut heap: BinaryHeap<HeapEntry>,
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let shadowed_winner = {
            let wal_hashes: AHashSet<u64> = self
                .wal_entries
                .iter()
                .map(|entry| entry.doc_id_hash)
                .collect();
            let wal_ids: AHashSet<&str> = self
                .wal_entries
                .iter()
                .map(|entry| entry.doc_id.as_str())
                .collect();
            let mut shadowed = false;
            for winner in &heap {
                let hash = self.record_at(winner.index)?.doc_id_hash;
                if wal_hashes.contains(&hash) && wal_ids.contains(self.doc_id_at(winner.index)?) {
                    shadowed = true;
                    break;
                }
            }
            // Release these temporary sets before allocating the exclusion view.
            shadowed
        };
        if shadowed_winner {
            let visible_main = ExcludeDocIdsFilter::new(
                filter,
                self.wal_entries.iter().map(|entry| entry.doc_id.as_str()),
            );
            // Only an actual superseded winner takes the extra scan. Normal
            // WAL appends preserve the existing selective gather and SIMD path.
            // Do not forward candidate_hashes: hashes alone cannot represent
            // the exact-ID exclusion when a collision occurs.
            heap = if params.parallel_enabled && self.record_count() >= params.parallel_threshold {
                self.scan_parallel(
                    query,
                    limit,
                    Some(&visible_main),
                    params.parallel_chunk_size.max(1),
                )?
            } else {
                self.scan_sequential(query, limit, Some(&visible_main))?
            };
        }
        // Apply the caller's original filter to the live WAL, NOT visible_main,
        // which would exclude the very replacements that must remain searchable.
        self.scan_wal(query, &mut heap, limit, filter)?;
        Ok(heap)
    }

    pub(super) fn search_top_k_internal(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        parallel_threshold: usize,
        parallel_chunk_size: usize,
        parallel_enabled: bool,
    ) -> SearchResult<Vec<VectorHit>> {
        self.ensure_query_dimension(query)?;
        let has_main = self.record_count() > 0;
        let has_wal = !self.wal_entries.is_empty();
        if limit == 0 || (!has_main && !has_wal) {
            return Ok(Vec::new());
        }
        let chunk_size = parallel_chunk_size.max(1);
        let use_parallel = parallel_enabled && self.record_count() >= parallel_threshold;
        let total_candidate_upper_bound =
            self.record_count().saturating_add(self.wal_entries.len());

        // Full-recall requests should avoid top-k heap churn.
        // When the caller asks for all available candidates (`k >= total`),
        // collect-and-sort is measurably faster than maintaining a size-k heap.
        if filter.is_none() && limit >= total_candidate_upper_bound {
            let mut winners = if has_main {
                if use_parallel {
                    self.scan_parallel_collect_all(query, chunk_size)?
                } else {
                    self.scan_range_collect_all(0, self.record_count(), query)?
                }
            } else {
                Vec::new()
            };
            if has_wal {
                self.scan_wal_collect_all(query, &mut winners)?;
            }
            // `limit_all` scan-all path: `winners` can hold every match. Above a
            // threshold the final sort dominates, and a parallel sort pays
            // (measured ~2.81× at 50k winners, `winners_sort` bench); below it the
            // rayon overhead is not worth it, so stay serial. Bit-identical either
            // way — `compare_best_first` is a strict total order.
            if winners.len() >= PAR_SORT_THRESHOLD {
                winners.par_sort_unstable_by(compare_best_first);
            } else {
                winners.sort_unstable_by(compare_best_first);
            }
            return self.resolve_sorted_entries(winners);
        }

        let mut heap = if has_main {
            if let Some(gathered) = self.try_gather_filtered(query, limit, filter)? {
                gathered
            } else if use_parallel {
                self.scan_parallel(query, limit, filter, chunk_size)?
            } else {
                self.scan_sequential(query, limit, filter)?
            }
        } else {
            let max_wal = self.wal_entries.len();
            BinaryHeap::with_capacity(limit.min(max_wal).saturating_add(1))
        };

        // Resolve main/WAL visibility before the final bounded selection.
        if has_wal {
            heap = self.merge_wal_top_k(
                query,
                heap,
                limit,
                filter,
                SearchParams {
                    parallel_threshold,
                    parallel_chunk_size: chunk_size,
                    parallel_enabled,
                },
            )?;
        }

        self.resolve_hits(heap)
    }
}
