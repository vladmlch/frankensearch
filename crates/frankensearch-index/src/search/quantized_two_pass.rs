use std::collections::BinaryHeap;

use frankensearch_core::{SearchError, SearchResult, VectorHit};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::simd::dot_i8x4_i8;
use crate::{
    PreparedQuery4bit, Quantization, VectorIndex, dot_4bit_prepared, dot_i8_i8, dot_i8_i8_maddubs,
    dot_product_f16_bytes_f32, maddubs_query_bias, pack_f16_le_bytes_to_4bit, prepare_4bit_query,
    quantize_f16_le_bytes_to_i8,
};

use super::{
    HeapEntry, INT8_PARALLEL_CHUNK_SIZE, Int8HeapKey, MAX_EXACT_I8_DOT_DIM, PARALLEL_CHUNK_SIZE,
    PARALLEL_THRESHOLD, insert_candidate, int8_heap_index, int8_heap_key, int8_heap_key_from_f32,
    merge_int8_partial_heaps, merge_partial_heaps, pack_4bit_query, quantize_i8_query,
    retain_int8_candidate, score_key,
};

impl VectorIndex {
    /// int8 ADC two-pass exact top-k for **standalone** large-N vector search:
    /// a fast parallel int8 pass-1 over all main records keeps the top
    /// `k·candidate_multiplier` by approximate score, then an exact f16 rescore of
    /// just those candidates selects the final top-k. Lossless (recall=1.0) whenever
    /// pass-1 retains the true top-k — validated on the in-memory twin; the int8
    /// dot is monotonic with the true dot under one corpus max-abs scale.
    ///
    /// Covers the contiguous F16 main-vector region only; falls back to the exact
    /// [`VectorIndex::search_top_k`] when a WAL is present or quantization is not F16
    /// (so results are always correct, never silently degraded). Not wired into the
    /// BOLD hybrid (that gap is not vector-bound — see `docs/NEGATIVE_EVIDENCE.md`);
    /// this targets pure vector-search latency at large N.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, and `SearchError::IndexCorrupted` for
    /// malformed slab data.
    pub fn search_top_k_int8_two_pass(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        // Production keeps the EXACT-int8 pass-1. The `vpmaddubs` kernel (bd-b5wl) is 1.23× faster
        // in isolation (decidable) and recall-exact, but its **scan-level** win is only ~1.02–1.11×
        // (Amdahl-shrunk) and is NOT robustly decidable under fleet contention: two null-controlled
        // runs on `hetzner1` disagreed — 0.9023 (median below null p5, a clear win) then 0.9821
        // (inside the null floor). Shipping the approximate kernel as default on a marginal,
        // contention-dependent effect fails the gate, so it stays behind
        // `bench_search_top_k_int8_two_pass_maddubs`. Retry = worker isolation (same as cod's int8
        // micro-opt block). See docs/NEGATIVE_EVIDENCE.md 2026-07-10.
        self.search_top_k_int8_two_pass_impl::<false, false>(query, k, candidate_multiplier)
    }

    /// Exact pre-row-block implementation retained only for same-binary
    /// performance comparisons. Production callers should use
    /// [`Self::search_top_k_int8_two_pass`].
    #[doc(hidden)]
    pub fn bench_search_top_k_int8_two_pass_orig(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_impl::<false, false>(query, k, candidate_multiplier)
    }

    /// Four-row query-decode-reuse candidate retained only so the null-controlled
    /// negative measurement stays reproducible. Production callers use
    /// [`Self::search_top_k_int8_two_pass`].
    #[doc(hidden)]
    pub fn bench_search_top_k_int8_two_pass_row_block_candidate(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_impl::<true, false>(query, k, candidate_multiplier)
    }

    /// `vpmaddubs` pass-1 kernel candidate (bd-b5wl), retained for the same-binary A/B.
    /// Bit-identical *ranking* to [`Self::search_top_k_int8_two_pass`] on realistic quantized data
    /// (proven recall in `simd::tests::maddubs_pass1_preserves_f32_recall_under_real_saturation`);
    /// the pass-1 int8 dot is the approximate `dot_i8_i8_maddubs` (see its saturation caveat).
    #[doc(hidden)]
    pub fn bench_search_top_k_int8_two_pass_maddubs(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_impl::<false, true>(query, k, candidate_multiplier)
    }

    fn search_top_k_int8_two_pass_impl<const ROW_BLOCKED: bool, const MADDUBS: bool>(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        let count = self.record_count();
        // Fall back to the exact scan for anything this fast path does not cover.
        if k == 0
            || count == 0
            || !self.wal_entries.is_empty()
            || self.quantization() != Quantization::F16
        {
            return self.search_top_k(query, k, None);
        }
        if query.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: query.len(),
            });
        }

        let dim = self.dimension();
        let candidate_count = k
            .saturating_mul(candidate_multiplier.max(1))
            .min(count)
            .max(k.min(count));
        let query_i8 = quantize_i8_query(query);
        // Per-query bias `128·Σq` for the `MADDUBS` pass-1 kernel; unused (0) otherwise.
        let q_bias128 = if MADDUBS {
            maddubs_query_bias(&query_i8, dim)
        } else {
            0
        };
        let slab = self.int8_slab();

        // Pass 1: bounded-heap int8 scan keeping the top `candidate_count`.
        // The int8 dot is cheap enough that exact-scan sized chunks overproduce
        // local top-N heaps; larger chunks keep enough Rayon tasks while shrinking
        // the post-scan merge fan-in.
        let candidate_heap = if count < PARALLEL_THRESHOLD {
            if ROW_BLOCKED {
                self.int8_scan_range(slab, &query_i8, 0, count, candidate_count)
            } else {
                self.int8_scan_range_orig::<MADDUBS>(
                    slab,
                    &query_i8,
                    q_bias128,
                    0,
                    count,
                    candidate_count,
                )
            }
        } else {
            let chunk_count = count.div_ceil(INT8_PARALLEL_CHUNK_SIZE);
            let partials: Vec<BinaryHeap<Int8HeapKey>> = (0..chunk_count)
                .into_par_iter()
                .map(|chunk_index| {
                    let start = chunk_index * INT8_PARALLEL_CHUNK_SIZE;
                    let end = (start + INT8_PARALLEL_CHUNK_SIZE).min(count);
                    if ROW_BLOCKED {
                        self.int8_scan_range(slab, &query_i8, start, end, candidate_count)
                    } else {
                        self.int8_scan_range_orig::<MADDUBS>(
                            slab,
                            &query_i8,
                            q_bias128,
                            start,
                            end,
                            candidate_count,
                        )
                    }
                })
                .collect();
            merge_int8_partial_heaps(partials, candidate_count)
        };

        // Pass 2: exact f16 rescore of the candidates through the SAME bounded-heap
        // selection + tie-break as `search_top_k`, so the final order is identical
        // whenever pass-1 retained the true top-k.
        let stride = dim * 2;
        let mut heap = BinaryHeap::with_capacity(k.saturating_add(1));
        for candidate in candidate_heap {
            let index = int8_heap_index(candidate);
            let vector_offset = self.vectors_offset + index * stride;
            let vector_bytes = &self.data[vector_offset..vector_offset + stride];
            let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
            insert_candidate(&mut heap, HeapEntry::new(index, score), k);
        }
        self.resolve_hits(heap)
    }

    fn int8_scan_range_orig<const MADDUBS: bool>(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        q_bias128: i32,
        start: usize,
        end: usize,
        limit: usize,
    ) -> BinaryHeap<Int8HeapKey> {
        if limit == 0 {
            return BinaryHeap::new();
        }
        if self.dimension() <= MAX_EXACT_I8_DOT_DIM {
            self.int8_scan_range_orig_with_key::<MADDUBS, _>(
                slab,
                query_i8,
                q_bias128,
                start,
                end,
                limit,
                int8_heap_key,
            )
        } else {
            self.int8_scan_range_orig_with_key::<MADDUBS, _>(
                slab,
                query_i8,
                q_bias128,
                start,
                end,
                limit,
                int8_heap_key_from_f32,
            )
        }
    }

    /// Exact `1948a65` per-row scan retained for the in-binary ORIGINAL arm. `MADDUBS` swaps the
    /// pass-1 int8 dot to the approximate `vpmaddubs` kernel (bd-b5wl); `q_bias128 = 128·Σq` is then
    /// live, else ignored. Production is `MADDUBS = false` → byte-identical to the shipped scan.
    fn int8_scan_range_orig_with_key<const MADDUBS: bool, F>(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        q_bias128: i32,
        start: usize,
        end: usize,
        limit: usize,
        make_key: F,
    ) -> BinaryHeap<Int8HeapKey>
    where
        F: Fn(usize, i32) -> Int8HeapKey,
    {
        let dim = self.dimension();
        let mut heap = BinaryHeap::with_capacity(limit.min(end - start).saturating_add(1));
        let mut cutoff = Int8HeapKey::MAX;
        let mut flags_offset = self.records_offset + start * 16 + 14;
        let mut slab_offset = start * dim;

        for index in start..end {
            let flags_bytes = &self.data[flags_offset..flags_offset + 2];
            let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);
            if (flags & 0x0001) == 0 {
                let stored = &slab[slab_offset..slab_offset + dim];
                let dot = if MADDUBS {
                    dot_i8_i8_maddubs(stored, query_i8, q_bias128)
                } else {
                    dot_i8_i8(stored, query_i8)
                };
                let candidate = make_key(index, dot);
                if heap.len() < limit {
                    heap.push(candidate);
                    if heap.len() == limit {
                        cutoff = heap.peek().copied().unwrap_or(Int8HeapKey::MAX);
                    }
                } else if candidate < cutoff {
                    let _ = heap.pop();
                    heap.push(candidate);
                    cutoff = heap.peek().copied().unwrap_or(Int8HeapKey::MAX);
                }
            }
            flags_offset += 16;
            slab_offset += dim;
        }
        heap
    }

    /// Bounded-heap int8 scan of records `[start, end)` over the int8 `slab`
    /// (index-aligned with the record table), skipping tombstoned records via the
    /// same flag check + cutoff fast-path as the exact `scan_range_chunk`.
    fn int8_scan_range(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        start: usize,
        end: usize,
        limit: usize,
    ) -> BinaryHeap<Int8HeapKey> {
        if limit == 0 {
            return BinaryHeap::new();
        }
        if self.dimension() <= MAX_EXACT_I8_DOT_DIM {
            self.int8_scan_range_with_key(slab, query_i8, start, end, limit, int8_heap_key)
        } else {
            self.int8_scan_range_with_key(slab, query_i8, start, end, limit, int8_heap_key_from_f32)
        }
    }

    fn int8_scan_range_with_key<F>(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        start: usize,
        end: usize,
        limit: usize,
        make_key: F,
    ) -> BinaryHeap<Int8HeapKey>
    where
        F: Fn(usize, i32) -> Int8HeapKey,
    {
        let dim = self.dimension();
        let mut heap = BinaryHeap::with_capacity(limit.min(end - start).saturating_add(1));
        let mut cutoff = Int8HeapKey::MAX;
        let mut flags_offset = self.records_offset + start * 16 + 14;
        let mut slab_offset = start * dim;

        let mut index = start;
        while index + 4 <= end {
            let flags0 = u16::from_le_bytes([self.data[flags_offset], self.data[flags_offset + 1]]);
            let flags1 =
                u16::from_le_bytes([self.data[flags_offset + 16], self.data[flags_offset + 17]]);
            let flags2 =
                u16::from_le_bytes([self.data[flags_offset + 32], self.data[flags_offset + 33]]);
            let flags3 =
                u16::from_le_bytes([self.data[flags_offset + 48], self.data[flags_offset + 49]]);
            let flags = [flags0, flags1, flags2, flags3];

            if ((flags0 | flags1 | flags2 | flags3) & 0x0001) == 0 {
                let stored_rows = &slab[slab_offset..slab_offset + 4 * dim];
                let scores = dot_i8x4_i8(stored_rows, query_i8);
                for (lane, score) in scores.into_iter().enumerate() {
                    retain_int8_candidate(
                        &mut heap,
                        &mut cutoff,
                        make_key(index + lane, score),
                        limit,
                    );
                }
            } else {
                for (lane, flags) in flags.into_iter().enumerate() {
                    if (flags & 0x0001) == 0 {
                        let row_offset = slab_offset + lane * dim;
                        let stored = &slab[row_offset..row_offset + dim];
                        retain_int8_candidate(
                            &mut heap,
                            &mut cutoff,
                            make_key(index + lane, dot_i8_i8(stored, query_i8)),
                            limit,
                        );
                    }
                }
            }

            index += 4;
            flags_offset += 64;
            slab_offset += 4 * dim;
        }

        while index < end {
            let flags_bytes = &self.data[flags_offset..flags_offset + 2];
            let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);
            if (flags & 0x0001) == 0 {
                let stored = &slab[slab_offset..slab_offset + dim];
                retain_int8_candidate(
                    &mut heap,
                    &mut cutoff,
                    make_key(index, dot_i8_i8(stored, query_i8)),
                    limit,
                );
            }
            index += 1;
            flags_offset += 16;
            slab_offset += dim;
        }
        heap
    }

    /// Lazily build (once) the int8 quantization of the contiguous F16 main-vector
    /// region. Only called after the F16/no-WAL gate in `search_top_k_int8_two_pass`.
    fn int8_slab(&self) -> &[i8] {
        self.vectors_i8.get_or_init(|| {
            let count = self.record_count();
            let dim = self.dimension();
            let byte_len = count * dim * 2;
            quantize_f16_le_bytes_to_i8(
                &self.data[self.vectors_offset..self.vectors_offset + byte_len],
            )
        })
    }

    /// 4-bit (16-level) two-pass exact top-k for standalone large-N vector search.
    /// A fast parallel pass-1 over a packed signed-4-bit slab (`dim/2` bytes/vector —
    /// half the int8 slab, so the bandwidth-bound pass-1 is faster) keeps the top
    /// `k·candidate_multiplier` by approximate score (`dot_4bit_prepared`), then
    /// an exact f16 rescore of just those candidates selects the final top-k. 16
    /// levels stay lossless at mult≈5 on realistic clustered data (see
    /// `fsvi_4bit_two_pass` bench); recall rises with `candidate_multiplier`.
    /// Falls back to the exact `search_top_k` for WAL/non-F16 indexes. Not wired
    /// into the BOLD hybrid.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, and `SearchError::IndexCorrupted` for
    /// malformed slab data.
    pub fn search_top_k_4bit_two_pass(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        let count = self.record_count();
        if k == 0
            || count == 0
            || !self.wal_entries.is_empty()
            || self.quantization() != Quantization::F16
        {
            return self.search_top_k(query, k, None);
        }
        if query.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: query.len(),
            });
        }

        let dim = self.dimension();
        let bytes_per_vector = dim.div_ceil(2);
        let candidate_count = k
            .saturating_mul(candidate_multiplier.max(1))
            .min(count)
            .max(k.min(count));
        let query_packed = pack_4bit_query(query);
        let query_prepared = prepare_4bit_query(&query_packed);
        let slab = self.nibbles_slab();

        let candidate_heap = if count < PARALLEL_THRESHOLD {
            self.nibble_scan_range(
                slab,
                &query_prepared,
                bytes_per_vector,
                0,
                count,
                candidate_count,
            )
        } else {
            let chunk_count = count.div_ceil(PARALLEL_CHUNK_SIZE);
            let partials: Vec<BinaryHeap<HeapEntry>> = (0..chunk_count)
                .into_par_iter()
                .map(|chunk_index| {
                    let start = chunk_index * PARALLEL_CHUNK_SIZE;
                    let end = (start + PARALLEL_CHUNK_SIZE).min(count);
                    self.nibble_scan_range(
                        slab,
                        &query_prepared,
                        bytes_per_vector,
                        start,
                        end,
                        candidate_count,
                    )
                })
                .collect();
            merge_partial_heaps(partials, candidate_count)
        };

        // Pass 2: exact f16 rescore (same bounded-heap selection + tie-break).
        let stride = dim * 2;
        let mut heap = BinaryHeap::with_capacity(k.saturating_add(1));
        for candidate in candidate_heap {
            let vector_offset = self.vectors_offset + candidate.index * stride;
            let vector_bytes = &self.data[vector_offset..vector_offset + stride];
            let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
            insert_candidate(&mut heap, HeapEntry::new(candidate.index, score), k);
        }
        self.resolve_hits(heap)
    }

    /// Bounded-heap 4-bit scan of records `[start, end)` over the packed nibble
    /// `slab` (index-aligned with the record table), skipping tombstoned records,
    /// with the same cutoff fast-path as the exact scan.
    fn nibble_scan_range(
        &self,
        slab: &[u8],
        query_prepared: &PreparedQuery4bit,
        bytes_per_vector: usize,
        start: usize,
        end: usize,
        limit: usize,
    ) -> BinaryHeap<HeapEntry> {
        let mut heap = BinaryHeap::with_capacity(limit.min(end - start).saturating_add(1));
        let mut cutoff = f32::NEG_INFINITY;
        let mut flags_offset = self.records_offset + start * 16 + 14;
        let mut slab_offset = start * bytes_per_vector;

        for index in start..end {
            let flags_bytes = &self.data[flags_offset..flags_offset + 2];
            let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);
            if (flags & 0x0001) == 0 {
                let stored = &slab[slab_offset..slab_offset + bytes_per_vector];
                let score = dot_4bit_prepared(stored, query_prepared) as f32;
                if heap.len() < limit || score_key(score) >= cutoff {
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                    if heap.len() >= limit
                        && let Some(&worst) = heap.peek()
                    {
                        cutoff = score_key(worst.score);
                    }
                }
            }
            flags_offset += 16;
            slab_offset += bytes_per_vector;
        }
        heap
    }

    /// Lazily build (once) the packed signed-4-bit quantization of the contiguous
    /// F16 main-vector region. Only called after the F16/no-WAL gate.
    fn nibbles_slab(&self) -> &[u8] {
        self.vectors_nibbles.get_or_init(|| {
            let count = self.record_count();
            let dim = self.dimension();
            let byte_len = count * dim * 2;
            pack_f16_le_bytes_to_4bit(
                &self.data[self.vectors_offset..self.vectors_offset + byte_len],
                dim,
            )
        })
    }
}
