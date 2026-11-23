use std::collections::{BinaryHeap, HashMap};

use ahash::AHashMap;
use frankensearch_core::config::{ZeroSignalReason, ZeroSignalState};
use frankensearch_core::filter::{BuildIdentityHasherU64, SearchFilter, fnv1a_hash};
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use half::f16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::search::{ClassifiedHits, PARALLEL_CHUNK_SIZE, SearchParams};
use crate::simd::{dot_4bit_prepared, dot_i8_i8, dot_product_f16_f32, prepare_4bit_query};

use super::constructors::{pack_4bit_query, pack_4bit_slab, quantize_i8_query, quantize_i8_slab};
use super::residual_io::residual_sidecar_error;
use super::residual_sidecar::{
    ExactResidualSidecar, ResidualPruningCensus, ResidualQueryTransform, ResidualScanOutcome,
    outward_f64_norm_from_square_sum, residual_lane_upper_bound,
};
use super::{
    EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS, EXACT_RESIDUAL_LANES, GATHER_SELECTIVITY_DIVISOR,
    HeapEntry, InMemoryVectorIndex, PAR_SORT_THRESHOLD, bounded_heap, compare_best_first,
    insert_candidate, merge_partial_heaps, score_key,
};

impl InMemoryVectorIndex {
    /// Get the document ID at position `index`.
    ///
    /// # Errors
    ///
    /// Returns error if index is out of bounds.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.doc_ids
            .get(index)
            .map(String::as_str)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index {} out of bounds (record_count = {})",
                    index,
                    self.doc_ids.len()
                ),
            })
    }

    /// Get the f16 vector slice at position `index`.
    pub(super) fn vector_slice(&self, index: usize) -> &[f16] {
        let start = index * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Brute-force cosine-similarity top-k search.
    ///
    /// Query must be pre-normalized. Uses f16→f32 SIMD dot product.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_with_params(query, limit, filter, SearchParams::default())
    }

    /// Compute the zero-signal census for this in-memory index.
    ///
    /// The in-memory index has no tombstones and no WAL, so `live_count`
    /// equals `record_count` and both WAL-derived states are structurally
    /// unreachable. O(n·dim); intended for lazy classification of empty
    /// results, not the hot path.
    #[must_use]
    pub fn zero_signal_state(&self) -> ZeroSignalState {
        let record_count = self.record_count();
        let mut usable_vector_count = 0usize;
        for row in 0..record_count {
            let start = row * self.dimension;
            let row_slice = &self.vectors[start..start + self.dimension];
            let mut norm_sq = 0.0_f32;
            let mut finite = true;
            for &value in row_slice {
                let value = value.to_f32();
                if !value.is_finite() {
                    finite = false;
                    break;
                }
                norm_sq += value * value;
            }
            if finite && norm_sq > 0.0 && norm_sq.is_finite() {
                usable_vector_count += 1;
            }
        }
        ZeroSignalState {
            record_count,
            live_count: record_count,
            tombstone_count: 0,
            wal_count: 0,
            usable_vector_count,
        }
    }

    /// Brute-force top-k with typed zero-signal classification.
    ///
    /// Mirrors [`crate::VectorIndex::search_top_k_classified`] exactly (bd-tqhc
    /// requires the in-memory and file-backed paths to classify equivalent
    /// states identically): non-finite queries are rejected fail-closed and
    /// an empty result always carries a typed [`ZeroSignalReason`].
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
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
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
        let zero_signal = hits.is_empty().then(|| {
            self.zero_signal_state()
                .empty_result_reason(filter.is_some())
        });
        Ok(ClassifiedHits { hits, zero_signal })
    }

    /// Brute-force top-k search with configurable parallelism.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_with_params(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<Vec<VectorHit>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }

        // Selective hash-addressable filter → gather the allow-set and exact-scan
        // only those positions (work ∝ |allow-set|, not corpus N). Bit-identical to
        // the per-document scan below.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
        }

        let use_parallel = params.parallel_enabled && count >= params.parallel_threshold;

        // A residual sidecar is optional and is reachable only after complete
        // source-bound admission. It has a matching parallel group route, so
        // the default 10K–100K parallel search uses the structural eliminator
        // while retaining the incumbent f16 scorer and merge order. `k >= N`
        // cannot prune anything, so it deliberately takes the unchanged flat
        // route and never allocates a `usize::MAX`-sized sidecar heap.
        if limit < count
            && let Some(sidecar) = self.exact_residual_sidecar.get()
        {
            let outcome = if use_parallel {
                self.scan_exact_residual_sidecar_parallel(
                    query,
                    limit,
                    filter,
                    sidecar,
                    params.parallel_chunk_size.max(1),
                )?
            } else {
                self.scan_exact_residual_sidecar(query, limit, filter, sidecar)?
            };
            return self.resolve_heap(outcome.heap);
        }

        let chunk_size = params.parallel_chunk_size.max(1);

        let heap = if use_parallel {
            self.scan_parallel(query, limit, filter, chunk_size)?
        } else {
            self.scan_sequential(query, limit, filter)?
        };

        self.resolve_heap(heap)
    }

    /// Approximate top-k via an **int8 ADC two-pass** (`bd-b5wl`): an int8 pass-1
    /// over all vectors keeps the top `limit * candidate_multiplier` candidates,
    /// then an exact f16 rescore with the same deterministic selection as
    /// [`Self::search_top_k`] produces the final ranking.
    ///
    /// Results are **bit-identical** to [`Self::search_top_k`] whenever pass-1
    /// retains the true top-k (recall = 1). Measured ~1.4–1.5× faster than the
    /// parallel exact path across 10k–100k at `candidate_multiplier = 5` (int8 is
    /// half the bytes + an integer `widening_mul` MAC — see `docs/PERF_LEDGER.md`), at
    /// the cost of *approximate* recall.
    ///
    /// **Tuning `candidate_multiplier`:** recall@10 = 1.0 held down to `mult = 2`
    /// for well-separated (random) vectors, so `mult = 5` is a good default; the
    /// candidate budget (`limit * mult`) is the selection-overhead knob — smaller is
    /// faster. Clustered real embeddings have closer neighbours, so re-measure recall
    /// on a representative corpus and raise `mult` if needed.
    ///
    /// Falls back to the exact path when the int8 slab is unavailable.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_int8_two_pass(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_filtered(query, limit, candidate_multiplier, None)
    }

    /// int8 ADC two-pass with an optional [`SearchFilter`]. Pass-1 pre-screens each
    /// vector by its precomputed `doc_id` hash (the same `matches_doc_id_hash` path
    /// the exact scan uses), so **filtered** large-N searches get the int8 speedup
    /// instead of falling back to the exact scan. The result matches the exact
    /// filtered top-k whenever pass-1 retains the true filtered top-k.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_int8_two_pass_filtered(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let candidate_count = limit.saturating_mul(candidate_multiplier.max(1)).min(count);
        // Full-recall short-cut: when the candidate budget already covers every
        // vector (`limit·mult ≥ count`, e.g. `limit_all` or a large limit on a small
        // corpus), pass-1 prunes NOTHING — the int8 scan, the size-N candidate heap,
        // the query quantize, and the slab build are all pure overhead, and pass-2
        // would rescore all N regardless. Delegate to the exact f16 single pass.
        // **Bit-identical by construction:** the two-pass equals `search_top_k`
        // whenever pass-1 retains the true top-k (its own doc-contract), and here it
        // retains *every* candidate. `limit.min(count)` avoids a `usize::MAX`-sized
        // heap and returns the same `min(limit, count)` hits.
        if candidate_count >= count {
            return self.search_top_k(query, limit.min(count), filter);
        }
        // Selective hash-addressable filter → exact gather of the allow-set. The
        // gather scans only `|allow-set|` vectors (vs all N for the int8 pass-1) and
        // is *exact* f16, so it is strictly more accurate than this approximate
        // two-pass while doing far less work. Falls through when the filter is not a
        // selective allow-list.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
        }
        let query_i8 = quantize_i8_query(query);
        // Build the int8 slab once, on first use — exact-only callers never pay the
        // O(N·d) quantization or its `N·d`-byte footprint at construction time.
        let vectors_i8 = self
            .vectors_i8
            .get_or_init(|| quantize_i8_slab(&self.vectors));

        // Pass 1: parallel **bounded-heap** int8 scan — each chunk keeps only its
        // top `candidate_count` (never materializing all N scores, unlike a
        // collect-all + select), then merge. This mirrors the exact path's
        // structure so the 3× int8 dot win is not eaten by selection overhead.
        // int8 dots peak well below 2^24 for realistic dims, so `i32 as f32` is
        // exact and preserves the candidate ranking + deterministic index tie-break.
        // Match the exact path's chunking so pass-1 uses all cores, not ~2.
        let chunk_size = PARALLEL_CHUNK_SIZE;
        let chunk_count = count.div_ceil(chunk_size);
        // When filtering, build the precomputed doc_id-hash slab once here (before
        // the parallel section) so pass-1 pre-screens by a hash lookup instead of
        // re-hashing each doc_id string per vector.
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());
        let partials: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                let mut heap = bounded_heap(candidate_count.min(end - start), "int8_partial_heap")?;
                let mut cutoff = f32::NEG_INFINITY;
                for index in start..end {
                    if let Some(f) = filter {
                        let passed = doc_id_hashes
                            .and_then(|h| f.matches_doc_id_hash(h[index], None))
                            .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
                        if !passed {
                            continue;
                        }
                    }
                    let offset = index * self.dimension;
                    let stored = &vectors_i8[offset..offset + self.dimension];
                    let score = dot_i8_i8(stored, &query_i8) as f32;
                    // Skip the insert_candidate call for scores that cannot enter the
                    // full bounded heap — the same cutoff fast-path scan_range uses,
                    // which the int8 pass-1 previously lacked. Result is unchanged
                    // (a sub-cutoff score never makes the bounded heap anyway).
                    if heap.len() < candidate_count || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), candidate_count);
                        if heap.len() >= candidate_count
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
                Ok(heap)
            })
            .collect();
        let candidate_heap = merge_partial_heaps(partials?, candidate_count)?;

        // Pass 2: exact f16 rescore of the candidates through the SAME bounded-heap
        // selection + tie-break as the exact path, so the final order matches
        // `search_top_k` exactly whenever pass-1 retained the true top-k.
        let mut heap = bounded_heap(limit.min(candidate_heap.len()), "int8_rescore_heap")?;
        for candidate in candidate_heap {
            let score = dot_product_f16_f32(self.vector_slice(candidate.index), query)?;
            insert_candidate(&mut heap, HeapEntry::new(candidate.index, score), limit);
        }
        self.resolve_heap(heap)
    }

    /// 4-bit (16-level) two-pass exact top-k — the in-memory twin of the FSVI
    /// `search_top_k_4bit_two_pass`. A parallel pass-1 over a packed signed-4-bit
    /// slab (`dim/2` bytes/vector — half the int8 slab) via the fused
    /// `dot_packed_4bit` kernel keeps the top `k·mult`, then an exact f16 rescore
    /// selects the final top-k. 16 levels are lossless at mult≈5 on realistic
    /// clustered data; the result matches the exact top-k whenever pass-1 retains it.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_4bit_two_pass(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_4bit_two_pass_filtered(query, limit, candidate_multiplier, None)
    }

    /// 4-bit two-pass with an optional [`SearchFilter`]. Pass-1 pre-screens each
    /// vector by its precomputed `doc_id` hash (the same path the int8 two-pass and
    /// exact scan use), so **filtered** searches keep the 4-bit speedup. Result
    /// matches the exact filtered top-k whenever pass-1 retains the true filtered top-k.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_4bit_two_pass_filtered(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let candidate_count = limit.saturating_mul(candidate_multiplier.max(1)).min(count);
        // Full-recall short-cut (see the int8 two-pass for the full rationale): when
        // `limit·mult ≥ count`, pass-1 prunes nothing, so the 4-bit scan, size-N
        // heap, query nibble-prep, and slab build are pure overhead — delegate to the
        // exact f16 pass. Bit-identical by construction (pass-1 retains every
        // candidate ⇒ two-pass == `search_top_k`).
        if candidate_count >= count {
            return self.search_top_k(query, limit.min(count), filter);
        }
        // Selective hash-addressable filter → exact gather of the allow-set (see the
        // int8 two-pass for the rationale): scans only `|allow-set|` vectors and is
        // exact f16, so it is strictly more accurate than this approximate 4-bit
        // two-pass while doing far less work.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
        }
        // Decode the (loop-invariant) query nibbles once, not per stored vector.
        let query_prepared = prepare_4bit_query(&pack_4bit_query(query));
        let bytes_per_vector = self.dimension.div_ceil(2);
        let nibbles = self
            .vectors_nibbles
            .get_or_init(|| pack_4bit_slab(&self.vectors, self.dimension));

        // Pass 1: parallel bounded-heap 4-bit scan (same chunking + cutoff fast-path
        // + filter pre-screen as the int8 two-pass).
        let chunk_size = PARALLEL_CHUNK_SIZE;
        let chunk_count = count.div_ceil(chunk_size);
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());
        let partials: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                let mut heap = bounded_heap(candidate_count.min(end - start), "4bit_partial_heap")?;
                let mut cutoff = f32::NEG_INFINITY;
                for index in start..end {
                    if let Some(f) = filter {
                        let passed = doc_id_hashes
                            .and_then(|h| f.matches_doc_id_hash(h[index], None))
                            .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
                        if !passed {
                            continue;
                        }
                    }
                    let offset = index * bytes_per_vector;
                    let stored = &nibbles[offset..offset + bytes_per_vector];
                    let score = dot_4bit_prepared(stored, &query_prepared) as f32;
                    if heap.len() < candidate_count || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), candidate_count);
                        if heap.len() >= candidate_count
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
                Ok(heap)
            })
            .collect();
        let candidate_heap = merge_partial_heaps(partials?, candidate_count)?;

        // Pass 2: exact f16 rescore of the candidates (same selection + tie-break).
        let mut heap = bounded_heap(limit.min(candidate_heap.len()), "4bit_rescore_heap")?;
        for candidate in candidate_heap {
            let score = dot_product_f16_f32(self.vector_slice(candidate.index), query)?;
            insert_candidate(&mut heap, HeapEntry::new(candidate.index, score), limit);
        }
        self.resolve_heap(heap)
    }

    #[allow(clippy::cast_precision_loss)]
    pub(super) fn scan_exact_residual_sidecar(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        sidecar: &ExactResidualSidecar,
    ) -> SearchResult<ResidualScanOutcome> {
        // The ordinary flat scan has established behaviour for non-finite
        // queries.  Do not reinterpret it through interval arithmetic: retain
        // that behaviour verbatim by declining the optional route.
        if query.iter().any(|value| !value.is_finite()) {
            return Ok(ResidualScanOutcome {
                heap: self.scan_sequential(query, limit, filter)?,
                census: ResidualPruningCensus::default(),
            });
        }
        let Ok(transformed) = ResidualQueryTransform::from_query(query, sidecar) else {
            // The transform and suffix norms belong exclusively to the
            // optional cache. Any preparation failure must be indistinguishable
            // from no cache: run the incumbent sequential exact scan.
            return Ok(ResidualScanOutcome {
                heap: self.scan_sequential(query, limit, filter)?,
                census: ResidualPruningCensus::default(),
            });
        };
        if !transformed.f32_flat_envelope_is_finite {
            return Ok(ResidualScanOutcome {
                heap: self.scan_sequential(query, limit, filter)?,
                census: ResidualPruningCensus::default(),
            });
        }
        self.scan_exact_residual_sidecar_groups(
            query,
            limit,
            filter,
            sidecar,
            &transformed,
            0,
            sidecar.group_count(),
            true,
        )
    }

    pub(super) fn scan_exact_residual_sidecar_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        sidecar: &ExactResidualSidecar,
        row_chunk_size: usize,
    ) -> SearchResult<ResidualScanOutcome> {
        if query.iter().any(|value| !value.is_finite()) {
            return Ok(ResidualScanOutcome {
                heap: self.scan_parallel(query, limit, filter, row_chunk_size)?,
                census: ResidualPruningCensus::default(),
            });
        }
        let Ok(transformed) = ResidualQueryTransform::from_query(query, sidecar) else {
            // See the sequential route: cache-only preparation can never
            // change the result or error surface of the parallel exact scan.
            return Ok(ResidualScanOutcome {
                heap: self.scan_parallel(query, limit, filter, row_chunk_size)?,
                census: ResidualPruningCensus::default(),
            });
        };
        if !transformed.f32_flat_envelope_is_finite {
            return Ok(ResidualScanOutcome {
                heap: self.scan_parallel(query, limit, filter, row_chunk_size)?,
                census: ResidualPruningCensus::default(),
            });
        }
        let groups = sidecar.group_count();
        if limit == 0 || groups == 0 {
            return Ok(ResidualScanOutcome {
                heap: BinaryHeap::new(),
                census: ResidualPruningCensus::default(),
            });
        }
        // Probe exactly once for this query before partitioning work across
        // Rayon. Per-chunk probes multiply the nonselective penalty by the
        // chunk count; this shared decision either enables pruning globally or
        // takes the incumbent exact parallel scan for the whole query.
        let probe_end = groups.min(EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS);
        let probe = self.scan_exact_residual_sidecar_groups(
            query,
            limit,
            filter,
            sidecar,
            &transformed,
            0,
            probe_end,
            false,
        )?;
        if probe_end < groups && probe.census.lanes_pruned == 0 {
            let mut census = probe.census;
            census.flat_fallback_rows = census.flat_fallback_rows.saturating_add(sidecar.count);
            census.adaptive_fallbacks = census.adaptive_fallbacks.saturating_add(1);
            return Ok(ResidualScanOutcome {
                heap: self.scan_parallel(query, limit, filter, row_chunk_size)?,
                census,
            });
        }
        let groups_per_chunk = row_chunk_size.div_ceil(sidecar.lanes).max(1);
        let remaining_groups = groups.saturating_sub(probe_end);
        let chunk_count = remaining_groups.div_ceil(groups_per_chunk);
        let partials: SearchResult<Vec<ResidualScanOutcome>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let first_group = probe_end + chunk_index * groups_per_chunk;
                let end_group = (first_group + groups_per_chunk).min(groups);
                self.scan_exact_residual_sidecar_groups(
                    query,
                    limit,
                    filter,
                    sidecar,
                    &transformed,
                    first_group,
                    end_group,
                    false,
                )
            })
            .collect();
        let mut census = probe.census;
        let mut heaps = Vec::new();
        heaps
            .try_reserve_exact(
                chunk_count
                    .checked_add(1)
                    .ok_or_else(|| residual_sidecar_error("parallel", "chunk count overflow"))?,
            )
            .map_err(|_| residual_sidecar_error("parallel", "partial heap allocation failed"))?;
        heaps.push(probe.heap);
        for partial in partials? {
            census.merge(partial.census);
            heaps.push(partial.heap);
        }
        Ok(ResidualScanOutcome {
            heap: merge_partial_heaps(heaps, limit)?,
            census,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn scan_exact_residual_sidecar_groups(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        sidecar: &ExactResidualSidecar,
        transformed: &ResidualQueryTransform,
        first_group: usize,
        end_group: usize,
        allow_adaptive_fallback: bool,
    ) -> SearchResult<ResidualScanOutcome> {
        let capped_end_group = end_group.min(sidecar.group_count());
        if limit == 0 || first_group >= capped_end_group {
            return Ok(ResidualScanOutcome {
                heap: BinaryHeap::new(),
                census: ResidualPruningCensus::default(),
            });
        }
        let range_start = first_group.saturating_mul(sidecar.lanes).min(sidecar.count);
        let range_end = capped_end_group
            .saturating_mul(sidecar.lanes)
            .min(sidecar.count);
        let rows_in_range = range_end.saturating_sub(range_start);
        let mut heap = bounded_heap(limit.min(rows_in_range), "residual_heap")?;
        let mut census = ResidualPruningCensus::default();
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());
        for group in first_group..capped_end_group {
            // When the first bounded probe cannot prune even one lane, retain
            // the exact heap accumulated so far and finish this work unit with
            // the incumbent scanner. This avoids paying residual arithmetic for
            // a query/cutoff geometry where this cache is not selective.
            if allow_adaptive_fallback
                && census.groups_scanned >= EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS
                && census.lanes_pruned == 0
            {
                let fallback_start = group * sidecar.lanes;
                let fallback_end = (capped_end_group * sidecar.lanes).min(sidecar.count);
                let fallback =
                    self.scan_range(fallback_start, fallback_end, query, limit, filter)?;
                for candidate in fallback {
                    insert_candidate(&mut heap, candidate, limit);
                }
                census.flat_fallback_rows = census
                    .flat_fallback_rows
                    .saturating_add(fallback_end.saturating_sub(fallback_start));
                census.adaptive_fallbacks = census.adaptive_fallbacks.saturating_add(1);
                break;
            }

            census.groups_scanned = census.groups_scanned.saturating_add(1);
            let group_start = group * sidecar.lanes;
            let active_lanes = (sidecar.count - group_start).min(sidecar.lanes);
            let mut survives = [false; EXACT_RESIDUAL_LANES];
            for (lane, survives_lane) in survives[..active_lanes].iter_mut().enumerate() {
                let index = group_start + lane;
                let passed = filter.is_none_or(|current_filter| {
                    doc_id_hashes
                        .and_then(|hashes| current_filter.matches_doc_id_hash(hashes[index], None))
                        .unwrap_or_else(|| current_filter.matches(&self.doc_ids[index], None))
                });
                *survives_lane = passed;
                if passed {
                    census.eligible_lanes = census.eligible_lanes.saturating_add(1);
                }
            }
            if !survives[..active_lanes].iter().any(|&live| live) {
                continue;
            }

            let centroid =
                &sidecar.centroids[group * sidecar.dimension..(group + 1) * sidecar.dimension];
            let mut centroid_dot = 0.0_f64;
            let mut centroid_norm_sq = 0.0_f64;
            for (&query_value, &centroid_value) in transformed.transformed.iter().zip(centroid) {
                centroid_dot += f64::from(query_value) * f64::from(centroid_value);
                centroid_norm_sq += f64::from(centroid_value) * f64::from(centroid_value);
            }
            // Like the transformed query norm, this enters a multiplicative
            // upper-bound term. Round away from zero before it reaches the
            // pruning predicate.
            let centroid_norm =
                outward_f64_norm_from_square_sum(centroid_norm_sq, sidecar.dimension)
                    .unwrap_or(f64::INFINITY);
            let mut partial = [0.0_f64; EXACT_RESIDUAL_LANES];
            for block_index in 0..sidecar.block_count() {
                if heap.len() >= limit {
                    let cutoff = f64::from(
                        heap.peek()
                            .expect("a full bounded heap has a worst candidate")
                            .score,
                    );
                    for lane in 0..active_lanes {
                        if !survives[lane] {
                            continue;
                        }
                        let suffix_offset = (group * (sidecar.block_count() + 1) + block_index)
                            * sidecar.lanes
                            + lane;
                        let residual_norm = f64::from(sidecar.suffix_norms[suffix_offset]);
                        let correction_norm =
                            f64::from(sidecar.correction_norms[group * sidecar.lanes + lane]);
                        let upper_bound = residual_lane_upper_bound(
                            sidecar,
                            transformed,
                            block_index,
                            centroid_dot,
                            centroid_norm,
                            residual_norm,
                            correction_norm,
                            partial[lane],
                        );
                        if upper_bound.is_finite() && upper_bound < cutoff {
                            survives[lane] = false;
                            census.lanes_pruned = census.lanes_pruned.saturating_add(1);
                        }
                    }
                    if !survives[..active_lanes].iter().any(|&live| live) {
                        break;
                    }
                }
                let start = block_index * sidecar.block;
                let end = (start + sidecar.block).min(sidecar.dimension);
                for transformed_dimension in start..end {
                    let query_value = f64::from(transformed.transformed[transformed_dimension]);
                    let base = group * sidecar.dimension * sidecar.lanes
                        + transformed_dimension * sidecar.lanes;
                    for lane in 0..active_lanes {
                        if survives[lane] {
                            partial[lane] +=
                                query_value * f64::from(sidecar.residuals[base + lane]);
                        }
                    }
                }
            }
            // The sidecar is an eliminator, never a scorer.  Survivors use the
            // exact incumbent operation, which makes the visible score and total
            // order identical to the flat path rather than merely numerically close.
            for (lane, &survived) in survives[..active_lanes].iter().enumerate() {
                if !survived {
                    continue;
                }
                let index = group_start + lane;
                let score = dot_product_f16_f32(self.vector_slice(index), query)?;
                census.exact_sidecar_scores = census.exact_sidecar_scores.saturating_add(1);
                let cutoff = heap
                    .peek()
                    .map_or(f32::NEG_INFINITY, |entry| score_key(entry.score));
                if heap.len() < limit || score_key(score) >= cutoff {
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                }
            }
        }
        Ok(ResidualScanOutcome { heap, census })
    }

    fn scan_sequential(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        self.scan_range(0, self.record_count(), query, limit, filter)
    }

    fn scan_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        chunk_size: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let count = self.record_count();
        let chunk_count = count.div_ceil(chunk_size);
        let partial_heaps: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                self.scan_range(start, end, query, limit, filter)
            })
            .collect();

        merge_partial_heaps(partial_heaps?, limit)
    }

    /// Lazily-built FNV-1a hashes of every `doc_id` (matches the hash
    /// `BitsetFilter` computes), so the filtered scan can pre-screen by hash via
    /// `SearchFilter::matches_doc_id_hash` instead of re-hashing each `doc_id`
    /// string per vector. Built once on first filtered search.
    pub(super) fn doc_id_hashes(&self) -> &[u64] {
        self.doc_id_hashes.get_or_init(|| {
            self.doc_ids
                .iter()
                .map(|id| fnv1a_hash(id.as_bytes()))
                .collect()
        })
    }

    /// Lazily-built `doc_id_hash → position` map for the selective-filter gather
    /// fast-path. Returns `None` when two `doc_ids` hash to the same value (the map
    /// would not be a bijection, so a gather could miss a colliding position the
    /// per-document scan would visit) — callers then fall back to the full scan,
    /// preserving exact results. Built once on first selective-filter search.
    pub(super) fn hash_to_pos(&self) -> Option<&HashMap<u64, usize, BuildIdentityHasherU64>> {
        self.hash_to_pos
            .get_or_init(|| {
                let hashes = self.doc_id_hashes();
                let mut map =
                    HashMap::with_capacity_and_hasher(hashes.len(), BuildIdentityHasherU64);
                for (pos, &h) in hashes.iter().enumerate() {
                    if map.insert(h, pos).is_some() {
                        // Hash collision (or duplicate doc_id): the map can hold
                        // only one position per hash, so disable the fast path.
                        return None;
                    }
                }
                Some(map)
            })
            .as_ref()
    }

    /// Exact f16 top-k over an explicit list of candidate positions (the
    /// selective-filter gather fast-path). Parallel above `PARALLEL_CHUNK_SIZE`
    /// positions, serial below (tiny allow-sets don't amortize rayon overhead). The
    /// parallel path scans disjoint position chunks into per-chunk bounded heaps and
    /// merges by the `(score, index)` total order — order-independent, exactly like
    /// [`Self::scan_parallel`] — so it is bit-identical to the serial gather, to the
    /// per-document scan, and across thread counts.
    fn scan_gather(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        if positions.len() > PARALLEL_CHUNK_SIZE {
            let partials: SearchResult<Vec<BinaryHeap<HeapEntry>>> = positions
                .par_chunks(PARALLEL_CHUNK_SIZE)
                .map(|chunk| self.gather_range(chunk, query, limit))
                .collect();
            return merge_partial_heaps(partials?, limit);
        }
        self.gather_range(positions, query, limit)
    }

    /// Bounded-heap exact f16 scan over one slice of candidate positions. Mirrors
    /// [`Self::scan_range`]'s cutoff fast-path, minus the per-document filter probe
    /// (membership was already decided when the positions were gathered).
    fn gather_range(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let mut heap = bounded_heap(limit.min(positions.len()), "gather_heap")?;
        let mut cutoff = f32::NEG_INFINITY;
        for &index in positions {
            let stored = self.vector_slice(index);
            let score = dot_product_f16_f32(stored, query)?;
            if heap.len() < limit || score_key(score) >= cutoff {
                insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                if heap.len() >= limit
                    && let Some(&worst) = heap.peek()
                {
                    cutoff = score_key(worst.score);
                }
            }
        }
        Ok(heap)
    }

    /// Try the selective-filter gather fast-path: when `filter` is a
    /// hash-addressable allow-list whose size is below
    /// `count / GATHER_SELECTIVITY_DIVISOR`, gather the allowed positions and exact
    /// f16-scan only those. Returns `Some(hits)` when the fast-path applied,
    /// `None` to fall through to the per-document scan. Bit-identical: the gathered
    /// passing set equals `{ pos : doc_id_hash[pos] ∈ allow-set }`, the same set the
    /// scan keeps, and both rank by the `(score, index)` total order.
    fn try_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        count: usize,
    ) -> SearchResult<Option<Vec<VectorHit>>> {
        let Some(f) = filter else {
            return Ok(None);
        };
        let Some(allowed) = f.candidate_hashes() else {
            return Ok(None);
        };
        // Selectivity gate: gather pays a map lookup + sort per allowed hash, so it
        // only wins when the allow-set is a small fraction of the corpus. Above the
        // crossover the per-document scan (which skips the gather's setup) is faster.
        if allowed.len().saturating_mul(GATHER_SELECTIVITY_DIVISOR) >= count {
            return Ok(None);
        }
        let Some(map) = self.hash_to_pos() else {
            return Ok(None);
        };
        let mut positions: Vec<usize> =
            allowed.iter().filter_map(|h| map.get(h).copied()).collect();
        // Ascending position order → sequential slab access (cache-friendly); not
        // required for correctness (the heap's total order is position-independent).
        positions.sort_unstable();
        let heap = self.scan_gather(&positions, query, limit)?;
        Ok(Some(self.resolve_heap(heap)?))
    }

    /// Bench-only: forced per-document filtered scan (the gather-fast-path baseline).
    /// Bypasses [`Self::try_gather_filtered`] so the A/B measures the old behavior.
    #[doc(hidden)]
    pub fn bench_scan_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let params = SearchParams::default();
        let use_parallel = params.parallel_enabled && count >= params.parallel_threshold;
        let heap = if use_parallel {
            self.scan_parallel(query, limit, filter, params.parallel_chunk_size.max(1))?
        } else {
            self.scan_sequential(query, limit, filter)?
        };
        self.resolve_heap(heap)
    }

    /// Bench-only: forced gather over a hash-addressable allow-set, ignoring the
    /// selectivity gate (so the crossover can be measured directly).
    #[doc(hidden)]
    pub fn bench_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: &dyn SearchFilter,
    ) -> SearchResult<Vec<VectorHit>> {
        let allowed = filter
            .candidate_hashes()
            .expect("bench_gather_filtered requires a hash-addressable allow-set");
        let map = self
            .hash_to_pos()
            .expect("bench_gather_filtered requires a bijective hash→pos map");
        let mut positions: Vec<usize> =
            allowed.iter().filter_map(|h| map.get(h).copied()).collect();
        positions.sort_unstable();
        let heap = self.scan_gather(&positions, query, limit)?;
        self.resolve_heap(heap)
    }

    /// O(1) `doc_id → position` lookup via a lazily-built map, replacing the O(N)
    /// `doc_ids.iter().position(...)` linear scan. First-insert-wins, matching
    /// `position`'s first-match semantics for any (non-canonical) duplicate ids.
    /// Built once on first lookup (the per-hit quality-rerank path); search-only
    /// callers never pay the `O(N)` build or its footprint. The map hashes with
    /// `ahash` (not SipHash): every quality query pays one hash of each hit's
    /// `doc_id` here, and `ahash` cuts the rescore loop ~1.27–1.48× at 128–300
    /// candidates (see `quality_rescore_hasher_ab`). Semantics are unchanged.
    pub(super) fn index_of_doc_id(&self, doc_id: &str) -> Option<usize> {
        self.doc_id_index
            .get_or_init(|| {
                let mut map = AHashMap::with_capacity(self.doc_ids.len());
                for (i, id) in self.doc_ids.iter().enumerate() {
                    map.entry(id.clone()).or_insert(i);
                }
                map
            })
            .get(doc_id)
            .copied()
    }

    fn scan_range(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let max_elements = end.saturating_sub(start);
        let mut heap = bounded_heap(limit.min(max_elements), "scan_heap")?;
        let mut cutoff = f32::NEG_INFINITY;

        // When filtering, pre-screen by precomputed doc_id hash (one HashSet lookup)
        // instead of re-hashing the doc_id string per vector; fall back to the
        // string/metadata path only when the filter can't decide by hash.
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());

        for index in start..end {
            if let Some(f) = filter {
                let passed = doc_id_hashes
                    .and_then(|h| f.matches_doc_id_hash(h[index], None))
                    .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
                if !passed {
                    continue;
                }
            }
            let stored = self.vector_slice(index);
            let score = dot_product_f16_f32(stored, query)?;
            if heap.len() < limit || score_key(score) >= cutoff {
                insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                if heap.len() >= limit
                    && let Some(&worst) = heap.peek()
                {
                    cutoff = score_key(worst.score);
                }
            }
        }
        Ok(heap)
    }

    pub(super) fn resolve_heap(&self, heap: BinaryHeap<HeapEntry>) -> SearchResult<Vec<VectorHit>> {
        if heap.is_empty() {
            return Ok(Vec::new());
        }
        let mut winners = heap.into_vec();
        // In-memory limit_all (`limit >= count`) builds a count-sized heap, so
        // `winners` can hold every record. Above a threshold the final sort
        // dominates and a parallel sort pays (~2.81× at 50k, `winners_sort` bench);
        // below it the rayon overhead is not worth it. Bit-identical — the same
        // gated lever as `search.rs:183` (`compare_best_first` is a strict total
        // order, so the parallel sort yields the same unique order).
        if winners.len() >= PAR_SORT_THRESHOLD {
            winners.par_sort_unstable_by(compare_best_first);
        } else {
            winners.sort_unstable_by(compare_best_first);
        }
        let mut hits = Vec::with_capacity(winners.len());
        for winner in winners {
            let index_u32 =
                u32::try_from(winner.index).map_err(|_| SearchError::InvalidConfig {
                    field: "index".to_owned(),
                    value: winner.index.to_string(),
                    reason: "index exceeds u32 range for VectorHit".to_owned(),
                })?;
            hits.push(VectorHit {
                index: index_u32,
                score: winner.score,
                doc_id: self.doc_ids[winner.index].as_str().into(),
            });
        }
        Ok(hits)
    }

    /// Iterate over all document IDs.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = &str> {
        self.doc_ids.iter().map(String::as_str)
    }

    /// Get the f32 vector at position `index`.
    ///
    /// # Errors
    ///
    /// Returns error if index is out of bounds.
    pub fn vector_at_f32(&self, index: usize) -> SearchResult<Vec<f32>> {
        if index >= self.record_count() {
            return Err(SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index {} out of bounds (record_count = {})",
                    index,
                    self.record_count()
                ),
            });
        }
        let stored = self.vector_slice(index);
        // SIMD-widen 8 f16 per block (`widen8_f16_slice`, the same magic-factor widen
        // the f16 dot kernels use — bit-identical to the scalar `f16::to_f32`), then a
        // scalar tail for the last < 8. Mirrors the FSVI `VectorIndex::vector_at_f32`.
        let mut out = Vec::with_capacity(stored.len());
        let (blocks, remainder) = stored.as_chunks::<8>();
        for arr in blocks {
            out.extend_from_slice(&crate::simd::widen8_f16_slice(arr).to_array());
        }
        for v in remainder {
            out.push(v.to_f32());
        }
        Ok(out)
    }

    /// Compute dot products between a query and specific hit positions.
    ///
    /// Used for quality scoring when this index serves as the quality tier.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn scores_for_hits(&self, query: &[f32], hits: &[VectorHit]) -> SearchResult<Vec<f32>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let mut scores = Vec::with_capacity(hits.len());
        for hit in hits {
            // Try to find by doc_id
            let score = self
                .index_of_doc_id(&hit.doc_id)
                .map(|idx| {
                    let stored = self.vector_slice(idx);
                    dot_product_f16_f32(stored, query)
                })
                .transpose()?
                .unwrap_or(0.0);
            scores.push(score);
        }
        Ok(scores)
    }
}
