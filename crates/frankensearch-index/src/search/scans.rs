use std::collections::BinaryHeap;

use ahash::AHashSet;
use frankensearch_core::filter::{DocIdHashSet, SearchFilter};
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

use crate::wal::{from_wal_index, is_wal_index, to_wal_index};
use crate::{
    Quantization, VectorIndex, dot_product_f16_bytes_f32, dot_product_f32_bytes_f32,
    dot_product_f32_f32,
};

use super::{
    GATHER_SELECTIVITY_DIVISOR, HeapEntry, PARALLEL_CHUNK_SIZE, compare_best_first,
    insert_candidate, merge_partial_heaps, score_key,
};

impl VectorIndex {
    pub(super) fn scan_sequential(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        // Re-use the parallel chunk logic for sequential scan to benefit from optimizations.
        filter.map_or_else(
            || self.scan_range_chunk(0, self.record_count(), query, limit),
            |filter| self.scan_range_chunk_filtered(0, self.record_count(), query, limit, filter),
        )
    }

    pub(super) fn scan_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        chunk_size: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let chunk_count = self.record_count().div_ceil(chunk_size);
        let partial_heaps: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(self.record_count());
                filter.map_or_else(
                    || self.scan_range_chunk(start, end, query, limit),
                    |active_filter| {
                        self.scan_range_chunk_filtered(start, end, query, limit, active_filter)
                    },
                )
            })
            .collect();

        Ok(merge_partial_heaps(partial_heaps?, limit))
    }

    pub(super) fn scan_parallel_collect_all(
        &self,
        query: &[f32],
        chunk_size: usize,
    ) -> SearchResult<Vec<HeapEntry>> {
        let chunk_count = self.record_count().div_ceil(chunk_size);
        let partial: SearchResult<Vec<Vec<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(self.record_count());
                self.scan_range_collect_all(start, end, query)
            })
            .collect();

        let partial = partial?;
        let total = partial.iter().map(std::vec::Vec::len).sum();
        let mut merged = Vec::with_capacity(total);
        for mut chunk in partial {
            merged.append(&mut chunk);
        }
        Ok(merged)
    }

    pub(super) fn scan_range_collect_all(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
    ) -> SearchResult<Vec<HeapEntry>> {
        let mut winners = Vec::with_capacity(end.saturating_sub(start));
        let dim = self.dimension();

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    let flags_bytes = &self.data[flags_offset..flags_offset + 2];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) == 0 {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
                        winners.push(HeapEntry::new(index, score));
                    }

                    flags_offset += 16;
                    vector_offset += stride;
                }
            }
            Quantization::F32 => {
                let stride = dim * 4;
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    let flags_bytes = &self.data[flags_offset..flags_offset + 2];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) == 0 {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f32_bytes_f32(vector_bytes, query)?;
                        winners.push(HeapEntry::new(index, score));
                    }

                    flags_offset += 16;
                    vector_offset += stride;
                }
            }
        }
        Ok(winners)
    }

    pub(super) fn try_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Option<BinaryHeap<HeapEntry>>> {
        let Some(active_filter) = filter else {
            return Ok(None);
        };
        let Some(allowed) = active_filter.candidate_hashes() else {
            return Ok(None);
        };
        let count = self.record_count();
        if count == 0 || allowed.len().saturating_mul(GATHER_SELECTIVITY_DIVISOR) >= count {
            return Ok(None);
        }
        self.scan_gather_hashes(allowed, query, limit).map(Some)
    }

    pub(super) fn scan_gather_hashes(
        &self,
        allowed: &DocIdHashSet,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let positions = self.gather_positions_for_hashes(allowed)?;
        self.scan_gather_positions(&positions, query, limit)
    }

    fn gather_positions_for_hashes(&self, allowed: &DocIdHashSet) -> SearchResult<Vec<usize>> {
        let mut positions = Vec::with_capacity(allowed.len());
        for &hash in allowed {
            let Some((start, end)) = self.hash_range(hash)? else {
                continue;
            };
            for index in start..end {
                let entry = self.record_at(index)?;
                if (entry.flags & 0x0001) == 0 {
                    positions.push(index);
                }
            }
        }
        positions.sort_unstable();
        Ok(positions)
    }

    fn hash_range(&self, hash: u64) -> SearchResult<Option<(usize, usize)>> {
        let count = self.record_count();
        let start = self.lower_bound_hash(hash, 0, count)?;
        if start == count || self.record_at(start)?.doc_id_hash != hash {
            return Ok(None);
        }
        let end = self.upper_bound_hash(hash, start + 1, count)?;
        Ok(Some((start, end)))
    }

    fn lower_bound_hash(&self, hash: u64, mut low: usize, mut high: usize) -> SearchResult<usize> {
        while low < high {
            let mid = low + (high - low) / 2;
            if self.record_at(mid)?.doc_id_hash < hash {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        Ok(low)
    }

    fn upper_bound_hash(&self, hash: u64, mut low: usize, mut high: usize) -> SearchResult<usize> {
        while low < high {
            let mid = low + (high - low) / 2;
            if self.record_at(mid)?.doc_id_hash <= hash {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        Ok(low)
    }

    fn scan_gather_positions(
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
            return Ok(merge_partial_heaps(partials?, limit));
        }
        self.gather_range(positions, query, limit)
    }

    fn gather_range(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let mut heap = BinaryHeap::with_capacity(limit.min(positions.len()).saturating_add(1));
        let dim = self.dimension();
        let mut cutoff = f32::NEG_INFINITY;

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                for &index in positions {
                    let vector_offset = self.vectors_offset + index * stride;
                    let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                    let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
                    if heap.len() < limit || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                        if heap.len() >= limit
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
            }
            Quantization::F32 => {
                let stride = dim * 4;
                for &index in positions {
                    let vector_offset = self.vectors_offset + index * stride;
                    let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                    let score = dot_product_f32_bytes_f32(vector_bytes, query)?;
                    if heap.len() < limit || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                        if heap.len() >= limit
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
            }
        }
        Ok(heap)
    }

    fn scan_range_chunk(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let max_elements = end.saturating_sub(start);
        let mut heap = BinaryHeap::with_capacity(limit.min(max_elements).saturating_add(1));
        let dim = self.dimension();
        let mut cutoff = f32::NEG_INFINITY;

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                // Flags are at offset 14 in 16-byte record
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    // Check flags directly from mapped memory
                    // SAFETY: offset arithmetic is bounded by record_count checks in open()
                    let flags_bytes = &self.data[flags_offset..flags_offset + 2];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) == 0 {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
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
                    vector_offset += stride;
                }
            }
            Quantization::F32 => {
                let stride = dim * 4;
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    let flags_bytes = &self.data[flags_offset..flags_offset + 2];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) == 0 {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f32_bytes_f32(vector_bytes, query)?;
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
                    vector_offset += stride;
                }
            }
        }
        Ok(heap)
    }

    fn scan_range_chunk_filtered(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
        filter: &dyn SearchFilter,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let max_elements = end.saturating_sub(start);
        let mut heap = BinaryHeap::with_capacity(limit.min(max_elements).saturating_add(1));
        let dim = self.dimension();
        let mut cutoff = f32::NEG_INFINITY;

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                let mut record_offset = self.records_offset + start * 16;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    let flags_bytes = &self.data[record_offset + 14..record_offset + 16];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) != 0 {
                        record_offset += 16;
                        vector_offset += stride;
                        continue;
                    }

                    let hash_bytes = &self.data[record_offset..record_offset + 8];
                    let hash = u64::from_le_bytes([
                        hash_bytes[0],
                        hash_bytes[1],
                        hash_bytes[2],
                        hash_bytes[3],
                        hash_bytes[4],
                        hash_bytes[5],
                        hash_bytes[6],
                        hash_bytes[7],
                    ]);

                    let passed = if let Some(matches) = filter.matches_doc_id_hash(hash, None) {
                        matches
                    } else {
                        let doc_id = self.doc_id_at(index)?;
                        filter.matches(doc_id, None)
                    };

                    if passed {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
                        if heap.len() < limit || score_key(score) >= cutoff {
                            insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                            if heap.len() >= limit
                                && let Some(&worst) = heap.peek()
                            {
                                cutoff = score_key(worst.score);
                            }
                        }
                    }

                    record_offset += 16;
                    vector_offset += stride;
                }
            }
            Quantization::F32 => {
                let stride = dim * 4;
                let mut record_offset = self.records_offset + start * 16;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    let flags_bytes = &self.data[record_offset + 14..record_offset + 16];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) != 0 {
                        record_offset += 16;
                        vector_offset += stride;
                        continue;
                    }

                    let hash_bytes = &self.data[record_offset..record_offset + 8];
                    let hash = u64::from_le_bytes([
                        hash_bytes[0],
                        hash_bytes[1],
                        hash_bytes[2],
                        hash_bytes[3],
                        hash_bytes[4],
                        hash_bytes[5],
                        hash_bytes[6],
                        hash_bytes[7],
                    ]);

                    let passed = if let Some(matches) = filter.matches_doc_id_hash(hash, None) {
                        matches
                    } else {
                        let doc_id = self.doc_id_at(index)?;
                        filter.matches(doc_id, None)
                    };

                    if passed {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f32_bytes_f32(vector_bytes, query)?;
                        if heap.len() < limit || score_key(score) >= cutoff {
                            insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                            if heap.len() >= limit
                                && let Some(&worst) = heap.peek()
                            {
                                cutoff = score_key(worst.score);
                            }
                        }
                    }

                    record_offset += 16;
                    vector_offset += stride;
                }
            }
        }
        Ok(heap)
    }

    pub(super) fn scan_wal(
        &self,
        query: &[f32],
        heap: &mut BinaryHeap<HeapEntry>,
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<()> {
        for (idx, entry) in self.wal_entries.iter().enumerate() {
            if let Some(f) = filter {
                if let Some(matches) = f.matches_doc_id_hash(entry.doc_id_hash, None) {
                    if !matches {
                        continue;
                    }
                } else if !f.matches(&entry.doc_id, None) {
                    continue;
                }
            }
            let score = dot_product_f32_f32(&entry.embedding, query)?;
            // Guard: corrupt WAL embeddings can produce NaN/Inf scores that
            // poison the top-k sort. Skip them (matches two_tier ANN path).
            if !score.is_finite() {
                continue;
            }
            insert_candidate(heap, HeapEntry::new(to_wal_index(idx), score), limit);
        }
        Ok(())
    }

    pub(super) fn scan_wal_collect_all(
        &self,
        query: &[f32],
        winners: &mut Vec<HeapEntry>,
    ) -> SearchResult<()> {
        winners.reserve(self.wal_entries.len());
        for (idx, entry) in self.wal_entries.iter().enumerate() {
            let score = dot_product_f32_f32(&entry.embedding, query)?;
            if !score.is_finite() {
                continue;
            }
            winners.push(HeapEntry::new(to_wal_index(idx), score));
        }
        Ok(())
    }

    pub(super) fn resolve_hits(&self, heap: BinaryHeap<HeapEntry>) -> SearchResult<Vec<VectorHit>> {
        if heap.is_empty() {
            return Ok(Vec::new());
        }

        let mut winners = heap.into_vec();
        winners.sort_unstable_by(compare_best_first);
        self.resolve_sorted_entries(winners)
    }

    pub(super) fn resolve_sorted_entries(
        &self,
        winners: Vec<HeapEntry>,
    ) -> SearchResult<Vec<VectorHit>> {
        // Pre-build a hash set of WAL doc_id hashes for O(1) pre-screening
        // instead of O(W) linear scan per main-index winner. On hash match,
        // falls back to string verification for correctness.
        let wal_hashes: AHashSet<u64> = self.wal_entries.iter().map(|e| e.doc_id_hash).collect();

        let mut seen: AHashSet<String> = AHashSet::with_capacity(winners.len());
        let mut hits = Vec::with_capacity(winners.len());
        for winner in winners {
            if is_wal_index(winner.index) {
                let wal_idx = from_wal_index(winner.index);
                let doc_id = &self.wal_entries[wal_idx].doc_id;
                // Skip WAL-vs-WAL duplicates (keep the first, i.e. highest-scored).
                if !seen.insert(doc_id.clone()) {
                    continue;
                }
                hits.push(self.resolve_wal_hit(&winner)?);
            } else {
                // Main index entry.
                if self.is_deleted(winner.index) {
                    continue;
                }
                let doc_id = self.doc_id_at(winner.index)?.to_owned();
                // Read pre-computed hash from record table instead of recomputing.
                let record = self.record_at(winner.index)?;
                let doc_id_hash = record.doc_id_hash;
                // O(1) hash pre-screen; only linear-scan on hash match.
                if wal_hashes.contains(&doc_id_hash) {
                    let has_wal_entry = self
                        .wal_entries
                        .iter()
                        .any(|e| e.doc_id_hash == doc_id_hash && e.doc_id == doc_id);
                    if has_wal_entry {
                        continue;
                    }
                }
                // Skip main-vs-main duplicates (String-based for correctness).
                if !seen.insert(doc_id.clone()) {
                    continue;
                }
                let index_u32 =
                    u32::try_from(winner.index).map_err(|_| SearchError::InvalidConfig {
                        field: "index".to_owned(),
                        value: winner.index.to_string(),
                        reason: "winner index exceeds u32 range for VectorHit".to_owned(),
                    })?;
                hits.push(VectorHit {
                    index: index_u32,
                    score: winner.score,
                    doc_id: doc_id.into(),
                });
            }
        }

        Ok(hits)
    }

    pub(super) fn resolve_wal_hit(&self, winner: &HeapEntry) -> SearchResult<VectorHit> {
        if !is_wal_index(winner.index) {
            return Err(SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: winner.index.to_string(),
                reason: "winner index is not WAL-encoded".to_owned(),
            });
        }

        let wal_idx = from_wal_index(winner.index);
        let entry = self
            .wal_entries
            .get(wal_idx)
            .ok_or_else(|| SearchError::IndexCorrupted {
                path: self.path.clone(),
                detail: format!(
                    "WAL index {} out of bounds (wal_entries.len() = {})",
                    wal_idx,
                    self.wal_entries.len()
                ),
            })?;
        let virtual_index =
            self.record_count()
                .checked_add(wal_idx)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "index".to_owned(),
                    value: wal_idx.to_string(),
                    reason: "WAL virtual index overflow".to_owned(),
                })?;
        let index_u32 = u32::try_from(virtual_index).map_err(|_| SearchError::InvalidConfig {
            field: "index".to_owned(),
            value: virtual_index.to_string(),
            reason: "WAL entry index exceeds u32 range".to_owned(),
        })?;
        Ok(VectorHit {
            index: index_u32,
            score: winner.score,
            doc_id: entry.doc_id.as_str().into(),
        })
    }

    #[allow(clippy::missing_const_for_fn)]
    pub(super) fn ensure_query_dimension(&self, query: &[f32]) -> SearchResult<()> {
        if query.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: query.len(),
            });
        }
        Ok(())
    }
}
