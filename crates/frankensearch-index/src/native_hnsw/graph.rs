use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::ValidatedFsviBytes;
use frankensearch_core::error::{SearchError, SearchResult};

use super::codec::{Adjacency, MutationJournal, NativeHnsw};
use super::types::{GraphDefect, HnswParams, MAX_LEVEL};

/// A candidate under consideration, ordered by distance.
///
/// Distances are compared with [`f32::total_cmp`] so a NaN — which a
/// corrupt vector could produce — yields a deterministic order instead of
/// undefined heap behaviour.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    id: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap ordering: `BinaryHeap` is a max-heap, so invert.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Nearest(Candidate);

impl Eq for Nearest {}

impl Ord for Nearest {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

impl PartialOrd for Nearest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reusable "have I seen this point?" set for a beam search.
///
/// A fresh `vec![false; len]` per beam search would make construction
/// quadratic in the corpus size purely in zeroing cost — every insert runs
/// one beam search per layer, so an n-point build writes O(n² · layers)
/// bytes before doing any useful work. Instead the buffer is allocated once
/// and reused: each search bumps an epoch, and a point counts as visited
/// only if its mark equals the current epoch, so clearing is O(1).
#[derive(Debug, Clone, Default)]
pub(super) struct VisitedSet {
    marks: Vec<u32>,
    epoch: u32,
}

impl VisitedSet {
    /// Begin a new search over `len` points, discarding previous marks.
    fn begin(&mut self, len: usize) {
        if self.marks.len() < len {
            self.marks.resize(len, 0);
        }
        // `0` means "never visited", so the epoch must never be 0. On
        // wraparound, clear the marks once rather than let a stale mark
        // from 2^32 searches ago read as visited.
        match self.epoch.checked_add(1) {
            Some(next) => self.epoch = next,
            None => {
                self.marks.fill(0);
                self.epoch = 1;
            }
        }
    }

    /// Mark `id` visited, returning whether it was already seen.
    ///
    /// Ids outside the buffer report as already-visited so callers skip
    /// them; a point that does not exist cannot be explored.
    fn visit(&mut self, id: u32) -> bool {
        let Some(mark) = self.marks.get_mut(id as usize) else {
            return true;
        };
        if *mark == self.epoch {
            return true;
        }
        *mark = self.epoch;
        false
    }
}

/// Deterministic level sampler.
///
/// HNSW assigns each point a level from a geometric distribution. Using a
/// seeded counter-based hash rather than a thread RNG makes graph topology
/// a pure function of insertion order and seed, which is what lets the
/// regression fixtures below pin exact structural outcomes.
#[derive(Debug, Clone, Copy)]
pub(super) struct LevelSampler {
    pub(super) seed: u64,
    level_scale: f64,
}

impl LevelSampler {
    /// Create a sampler for the given degree and seed.
    #[must_use]
    pub(super) fn new(m: usize, seed: u64) -> Self {
        // The standard 1/ln(M) scale; guard M == 1, whose ln is zero.
        let level_scale = if m <= 1 { 1.0 } else { 1.0 / (m as f64).ln() };
        Self { seed, level_scale }
    }

    /// Sample the level for `id`, in `0..MAX_LEVEL`.
    #[must_use]
    pub(super) fn level_for(&self, id: u32) -> usize {
        // SplitMix64 finalizer over (seed, id): deterministic, well
        // distributed, and independent of insertion order.
        let mut z = self
            .seed
            .wrapping_add(u64::from(id).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // Map to (0, 1] and take the geometric level.
        let unit = ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let level = (-unit.ln() * self.level_scale).floor();
        if !level.is_finite() || level <= 0.0 {
            return 0;
        }
        // Walk up to the sampled level rather than casting the float.
        // `MAX_LEVEL` is small, so this is at most a handful of comparisons,
        // and it makes the ceiling structural: there is no `as usize` whose
        // out-of-range behaviour would silently saturate a corrupt or
        // extreme value into a plausible-looking level.
        let mut chosen = 0usize;
        while chosen + 1 < MAX_LEVEL && (chosen + 1) as f64 <= level {
            chosen += 1;
        }
        chosen
    }
}

/// Distance function over stored vectors.
///
/// The graph stores no vectors of its own: it indexes rows of a caller-owned
/// store and asks for distances by row id. That keeps a single copy of the
/// vector data (the FSVI slab) and lets the graph reuse whichever SIMD
/// kernel the store already selected.
pub(super) trait VectorDistance {
    /// Distance between stored row `id` and `query`. Smaller is nearer.
    ///
    /// # Errors
    ///
    /// Returns whatever the underlying store returns for an unreadable row.
    fn distance_to_query(&self, id: u32, query: &[f32]) -> SearchResult<f32>;

    /// Distance between two stored rows.
    ///
    /// # Errors
    ///
    /// Returns whatever the underlying store returns for an unreadable row.
    fn distance_between(&self, a: u32, b: u32) -> SearchResult<f32>;

    /// Number of rows available.
    fn len(&self) -> usize;
}

impl VectorDistance for ValidatedFsviBytes {
    fn distance_to_query(&self, id: u32, query: &[f32]) -> SearchResult<f32> {
        let index = usize::try_from(id).map_err(|_| SearchError::InvalidConfig {
            field: "native_hnsw.physical_row".to_owned(),
            value: id.to_string(),
            reason: "physical row does not fit usize".to_owned(),
        })?;
        let vector = self.vector_at_f32(index)?;
        Ok(1.0 - crate::dot_product_f32_f32(&vector, query)?)
    }

    fn distance_between(&self, a: u32, b: u32) -> SearchResult<f32> {
        let index = usize::try_from(a).map_err(|_| SearchError::InvalidConfig {
            field: "native_hnsw.physical_row".to_owned(),
            value: a.to_string(),
            reason: "physical row does not fit usize".to_owned(),
        })?;
        let left = self.vector_at_f32(index)?;
        self.distance_to_query(b, &left)
    }

    fn len(&self) -> usize {
        self.record_count()
    }
}

impl NativeHnsw {
    /// Build a graph over every row of `store`, in row order.
    ///
    /// Serial by construction: the previous engine's parallel insert
    /// carried an entry-point race, and correctness comes before
    /// parallelism. A parallel builder can be added behind the same
    /// structural attestation once this path is proven.
    ///
    /// # Errors
    ///
    /// Propagates distance-computation failures from `store`.
    pub(super) fn build<D: VectorDistance>(
        params: HnswParams,
        seed: u64,
        store: &D,
    ) -> SearchResult<Self> {
        let mut graph = Self::new(params, seed)?;
        let count = u32::try_from(store.len()).map_err(|_| SearchError::InvalidConfig {
            field: "store.len".to_owned(),
            value: store.len().to_string(),
            reason: "row count must fit in u32".to_owned(),
        })?;
        for id in 0..count {
            graph.insert(id, store)?;
        }
        Ok(graph)
    }

    /// Insert row `id`, which must be the next unindexed row.
    ///
    /// Failure-atomic: every existing neighbour list touched by linking or
    /// pruning is journaled before mutation. If the vector store returns an
    /// error at any distance-computation boundary, those lists, the entry
    /// point, the maximum level, and the point count are restored exactly.
    /// The same row can then be retried without rebuilding the graph.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `id` is not the next row or
    /// the graph is full, and propagates distance-computation failures.
    pub(super) fn insert<D: VectorDistance>(&mut self, id: u32, store: &D) -> SearchResult<()> {
        // Borrow the reusable visited buffer out of `self` so the insert
        // body can hold `&mut self` and the buffer at the same time; it is
        // restored on every path, including the error path.
        let mut visited = std::mem::take(&mut self.scratch);
        let outcome = self.insert_with(id, store, &mut visited);
        self.scratch = visited;
        outcome
    }

    fn insert_with<D: VectorDistance>(
        &mut self,
        id: u32,
        store: &D,
        visited: &mut VisitedSet,
    ) -> SearchResult<()> {
        // A saturating conversion here would let the (4-billionth) insert
        // into a full graph look like a valid in-order one, so report the
        // capacity explicitly instead.
        let expected =
            u32::try_from(self.adjacency.len()).map_err(|_| SearchError::InvalidConfig {
                field: "points".to_owned(),
                value: self.adjacency.len().to_string(),
                reason: "graph is full: point ids must fit in u32".to_owned(),
            })?;
        if id != expected {
            return Err(SearchError::InvalidConfig {
                field: "id".to_owned(),
                value: id.to_string(),
                reason: format!("points must be inserted in row order; expected {expected}"),
            });
        }
        if id as usize >= store.len() {
            return Err(SearchError::InvalidConfig {
                field: "store.len".to_owned(),
                value: store.len().to_string(),
                reason: format!("store does not contain insertion row {id}"),
            });
        }

        let original_point_count = self.adjacency.len();
        let original_entry = self.entry;
        let original_max_level = self.max_level;
        let mut journal = MutationJournal::new(original_point_count);
        let level = self.sampler.level_for(id);
        self.adjacency.push(Adjacency::with_level(level));

        let outcome = (|| {
            // First point: it becomes the entry and has nothing to link to.
            let Some(mut current) = self.entry else {
                self.entry = Some(id);
                self.max_level = level;
                return Ok(());
            };

            let previous_max = self.max_level;

            // Phase 1 — greedy descent through the layers ABOVE this point's
            // own level, purely to find a good entry point.
            //
            // bd-u3wt defect (1): this descent must never link the new point.
            // It runs strictly above `level`, so no edge can be created in a
            // layer the new point does not occupy.
            let mut layer = previous_max;
            while layer > level {
                current = self.greedy_descend(current, id, layer, store)?;
                layer -= 1;
            }

            // Phase 2 — connect, starting at min(new level, previous max).
            //
            // bd-u3wt defect (2): starting above `previous_max` would link at
            // layers no other point occupies; starting below the new point's
            // level would leave its upper layers empty. The floor is exactly
            // this minimum.
            let mut entry_points = vec![current];
            for layer in (0..=level.min(previous_max)).rev() {
                let candidates = self.search_layer(
                    &entry_points,
                    id,
                    layer,
                    self.params.ef_construction,
                    store,
                    visited,
                )?;
                let mut selected =
                    Self::select_neighbours(id, &candidates, self.params.degree_at(layer), store)?;

                // A reciprocal degree-bounded graph needs a spanning path
                // that pruning cannot remove. Equal vectors otherwise make
                // every neighbor evict the same older row when the newest
                // row arrives, disconnecting valid data. Spend an existing
                // layer-zero slot on the physical predecessor; no additional
                // edges or vector storage are allocated beyond the budget.
                if layer == 0 {
                    let predecessor = id - 1; // The first row returned above.
                    if !selected.contains(&predecessor) {
                        if selected.len() == self.params.m0
                            && let Some(farthest) = candidates
                                .iter()
                                .rev()
                                .find(|candidate| selected.contains(&candidate.id))
                                .map(|candidate| candidate.id)
                        {
                            selected.retain(|&neighbor| neighbor != farthest);
                        }
                        selected.push(predecessor);
                    }
                }

                self.link(id, &selected, layer, store, &mut journal)?;

                entry_points = candidates.iter().map(|candidate| candidate.id).collect();
                if entry_points.is_empty() {
                    entry_points.push(current);
                }
            }

            // A point sampled above the previous maximum becomes the new
            // entry only after every fallible operation has succeeded.
            if level > previous_max {
                self.entry = Some(id);
                self.max_level = level;
            }
            Ok(())
        })();

        if let Err(error) = outcome {
            journal.rollback(&mut self.adjacency);
            self.adjacency.truncate(original_point_count);
            self.entry = original_entry;
            self.max_level = original_max_level;
            return Err(error);
        }
        Ok(())
    }

    /// Walk greedily to the nearest point to `target` at `layer`.
    fn greedy_descend<D: VectorDistance>(
        &self,
        start: u32,
        target: u32,
        layer: usize,
        store: &D,
    ) -> SearchResult<u32> {
        let mut current = start;
        let mut current_distance = store.distance_between(current, target)?;
        loop {
            let mut improved = false;
            for &neighbour in self.neighbours_at(current, layer) {
                let distance = store.distance_between(neighbour, target)?;
                if distance < current_distance {
                    current_distance = distance;
                    current = neighbour;
                    improved = true;
                }
            }
            if !improved {
                return Ok(current);
            }
        }
    }

    /// Beam search at `layer`, returning up to `ef` nearest candidates.
    ///
    /// `target_id` names a stored row; distances are row-to-row, which is
    /// what construction needs. [`Self::search_layer_query`] is the
    /// query-vector twin used at search time.
    fn search_layer<D: VectorDistance>(
        &self,
        entry_points: &[u32],
        target_id: u32,
        layer: usize,
        ef: usize,
        store: &D,
        visited: &mut VisitedSet,
    ) -> SearchResult<Vec<Candidate>> {
        self.beam_search(entry_points, layer, ef, store, visited, &mut |id, store| {
            store.distance_between(id, target_id)
        })
    }

    /// Beam search at `layer` against a raw query vector.
    fn search_layer_query<D: VectorDistance>(
        &self,
        entry_points: &[u32],
        query: &[f32],
        layer: usize,
        ef: usize,
        store: &D,
        visited: &mut VisitedSet,
    ) -> SearchResult<Vec<Candidate>> {
        self.beam_search(entry_points, layer, ef, store, visited, &mut |id, store| {
            store.distance_to_query(id, query)
        })
    }

    /// The shared beam search over a layer.
    ///
    /// Maintains a visited set, a min-heap frontier and a max-heap result
    /// set of size `ef`, expanding the nearest unexplored candidate until
    /// nothing closer than the current worst result remains.
    fn beam_search<D: VectorDistance>(
        &self,
        entry_points: &[u32],
        layer: usize,
        ef: usize,
        store: &D,
        visited: &mut VisitedSet,
        distance: &mut dyn FnMut(u32, &D) -> SearchResult<f32>,
    ) -> SearchResult<Vec<Candidate>> {
        let ef = ef.max(1);
        visited.begin(self.adjacency.len());
        let mut frontier: BinaryHeap<Nearest> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &entry in entry_points {
            if visited.visit(entry) {
                continue;
            }
            let candidate = Candidate {
                distance: distance(entry, store)?,
                id: entry,
            };
            frontier.push(Nearest(candidate));
            results.push(candidate);
        }
        while results.len() > ef {
            results.pop();
        }

        while let Some(Nearest(current)) = frontier.pop() {
            // Stop once the frontier's nearest is worse than the worst kept
            // result and the result set is already full.
            if let Some(worst) = results.peek()
                && results.len() >= ef
                && current.distance > worst.distance
            {
                break;
            }
            for &neighbour in self.neighbours_at(current.id, layer) {
                if visited.visit(neighbour) {
                    continue;
                }
                let candidate = Candidate {
                    distance: distance(neighbour, store)?,
                    id: neighbour,
                };
                let keep = results.len() < ef
                    || results
                        .peek()
                        .is_some_and(|worst| candidate.distance < worst.distance);
                if keep {
                    frontier.push(Nearest(candidate));
                    results.push(candidate);
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut ordered = results.into_vec();
        ordered.sort_unstable();
        Ok(ordered)
    }

    /// Choose which candidates to keep as neighbours.
    ///
    /// Uses the relative-neighbourhood heuristic: a candidate is kept only
    /// if it is closer to the new point than to any already-kept
    /// neighbour. This preserves long-range links that plain
    /// nearest-`m` selection prunes away, which is what keeps the graph
    /// navigable rather than clustered.
    fn select_neighbours<D: VectorDistance>(
        new_id: u32,
        candidates: &[Candidate],
        limit: usize,
        store: &D,
    ) -> SearchResult<Vec<u32>> {
        let mut selected: Vec<u32> = Vec::with_capacity(limit.min(candidates.len()));
        for candidate in candidates {
            if selected.len() >= limit {
                break;
            }
            if candidate.id == new_id {
                continue;
            }
            let mut dominated = false;
            for &kept in &selected {
                if store.distance_between(candidate.id, kept)? < candidate.distance {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                selected.push(candidate.id);
            }
        }
        // Top up to the budget with the nearest remaining candidates.
        //
        // Measured, not assumed: leaving the heuristic to under-fill dropped
        // recall@10 from 0.66 to 0.54 and average layer-0 degree from 32 to
        // 11.6 on the same corpus. The heuristic decides *preference* among
        // candidates; starving the neighbourhood costs more recall than the
        // clustering it avoids.
        if selected.len() < limit {
            for candidate in candidates {
                if selected.len() >= limit {
                    break;
                }
                if candidate.id != new_id && !selected.contains(&candidate.id) {
                    selected.push(candidate.id);
                }
            }
        }
        Ok(selected)
    }

    /// Install edges between `id` and `selected` at `layer`.
    ///
    /// bd-u3wt defect (3): the reciprocal edge is stored at `layer` — the
    /// layer being processed — never at the new point's top level. Both
    /// endpoints are checked for participation first, so an edge can never
    /// land in a layer either endpoint does not occupy.
    pub(super) fn link<D: VectorDistance>(
        &mut self,
        id: u32,
        selected: &[u32],
        layer: usize,
        store: &D,
        journal: &mut MutationJournal,
    ) -> SearchResult<()> {
        for &neighbour in selected {
            if neighbour == id {
                continue;
            }
            let both_participate = self
                .adjacency
                .get(id as usize)
                .is_some_and(|point| point.participates_in(layer))
                && self
                    .adjacency
                    .get(neighbour as usize)
                    .is_some_and(|point| point.participates_in(layer));
            if !both_participate {
                continue;
            }

            journal.record(&self.adjacency, id, layer);
            if let Some(point) = self.adjacency.get_mut(id as usize)
                && let Some(list) = point.layers.get_mut(layer)
                && !list.contains(&neighbour)
            {
                list.push(neighbour);
            }
            journal.record(&self.adjacency, neighbour, layer);
            if let Some(point) = self.adjacency.get_mut(neighbour as usize)
                && let Some(list) = point.layers.get_mut(layer)
                && !list.contains(&id)
            {
                list.push(id);
            }

            // The reverse edge may have pushed the neighbour over budget.
            // Retain the newly inserted edge wherever the budget allows,
            // alongside layer zero's permanent spanning path. A degree-two
            // base layer has room only for that path.
            self.prune(neighbour, layer, Some(id), store, journal)?;
        }
        Ok(())
    }

    /// Trim a neighbourhood back to the layer's degree budget.
    fn prune<D: VectorDistance>(
        &mut self,
        id: u32,
        layer: usize,
        protected: Option<u32>,
        store: &D,
        journal: &mut MutationJournal,
    ) -> SearchResult<()> {
        let budget = self.params.degree_at(layer);
        let current = self.neighbours_at(id, layer).to_vec();
        if current.len() <= budget {
            return Ok(());
        }
        let mut scored: Vec<Candidate> = Vec::with_capacity(current.len());
        for neighbour in current {
            scored.push(Candidate {
                distance: store.distance_between(id, neighbour)?,
                id: neighbour,
            });
        }
        scored.sort_unstable();
        let mut kept = Self::select_neighbours(id, &scored, budget, store)?;
        // Reinstate routing edges if the heuristic dropped them, evicting
        // the farthest unprotected neighbour to stay inside the budget.
        // Consecutive physical rows form a permanent layer-zero path. Both
        // incident path edges must survive, including one restored earlier
        // in this loop. With a third slot available, preserve the incoming
        // geometric edge too, as the original builder did. A new outlier
        // should retain its chosen neighbors, not only an arbitrary path link.
        //
        // The eviction must find the genuinely farthest entry rather than
        // popping the last one: `select_neighbours` appends its top-up fill
        // after the heuristic's picks, so the list is not in distance order
        // and popping would usually evict a NEAR neighbour. `scored` is
        // sorted ascending, so the last unprotected kept entry it mentions
        // is the farthest eligible eviction.
        let is_protected = |neighbour: u32| {
            (layer == 0 && id.abs_diff(neighbour) == 1)
                || ((layer > 0 || budget >= 3) && protected == Some(neighbour))
        };
        for candidate in &scored {
            if is_protected(candidate.id) && !kept.contains(&candidate.id) {
                if let Some(farthest) = scored
                    .iter()
                    .rev()
                    .find(|candidate| kept.contains(&candidate.id) && !is_protected(candidate.id))
                    .map(|candidate| candidate.id)
                {
                    kept.retain(|&neighbour| neighbour != farthest);
                }
                kept.push(candidate.id);
            }
        }

        // Dropping an edge must drop BOTH directions. Trimming only this
        // point's list leaves the counterpart pointing back at a node that
        // no longer points to it, so traversal can enter a region it cannot
        // leave — the graph silently becomes directed and fragments. This is
        // the same asymmetry class bd-u3wt found in the previous engine
        // (measured here as 33 of 5000 points reachable before the fix).
        let dropped: Vec<u32> = scored
            .iter()
            .map(|candidate| candidate.id)
            .filter(|candidate| !kept.contains(candidate))
            .collect();
        journal.record(&self.adjacency, id, layer);
        if let Some(point) = self.adjacency.get_mut(id as usize)
            && let Some(list) = point.layers.get_mut(layer)
        {
            *list = kept;
        }
        for neighbour in dropped {
            journal.record(&self.adjacency, neighbour, layer);
            if let Some(point) = self.adjacency.get_mut(neighbour as usize)
                && let Some(list) = point.layers.get_mut(layer)
            {
                list.retain(|&back| back != id);
            }
        }
        Ok(())
    }

    /// Neighbours of `id` at `layer`, empty if it does not participate.
    pub(super) fn neighbours_at(&self, id: u32, layer: usize) -> &[u32] {
        self.adjacency
            .get(id as usize)
            .map_or(&[], |point| point.neighbours(layer))
    }

    /// Search for the `k` nearest rows to `query`.
    ///
    /// Returns `(row id, distance)` pairs, nearest first. `ef` overrides the
    /// configured beam width; it is raised to at least `k` so the beam can
    /// hold the answer.
    ///
    /// # Errors
    ///
    /// Propagates distance-computation failures from `store`.
    pub(super) fn search<D: VectorDistance>(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        store: &D,
    ) -> SearchResult<Vec<(u32, f32)>> {
        self.verify_store_cardinality(store)
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "native-hnsw",
                source: Box::new(source),
            })?;
        if k == 0 || self.adjacency.is_empty() {
            return Ok(Vec::new());
        }
        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };
        let ef = ef.unwrap_or(self.params.ef_search).max(k);
        // One buffer for the whole query rather than one per layer.
        let mut visited = VisitedSet::default();

        // Descend the upper layers greedily, then run one wide beam search
        // at layer 0 where every point participates.
        let mut current = entry;
        let mut layer = self.max_level;
        while layer > 0 {
            let found =
                self.search_layer_query(&[current], query, layer, 1, store, &mut visited)?;
            if let Some(best) = found.first() {
                current = best.id;
            }
            layer -= 1;
        }

        let mut candidates =
            self.search_layer_query(&[current], query, 0, ef, store, &mut visited)?;
        candidates.truncate(k);
        Ok(candidates
            .into_iter()
            .map(|candidate| (candidate.id, candidate.distance))
            .collect())
    }

    /// Verify the graph's structural invariants.
    ///
    /// This is the executable form of the bd-u3wt findings, suitable for
    /// use as a post-build gate: every defect that campaign found would be
    /// caught here.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphDefect`] found.
    pub(super) fn verify(&self) -> Result<(), GraphDefect> {
        let count = self.adjacency.len();

        if count == 0 {
            if let Some(entry) = self.entry {
                return Err(GraphDefect::EntryPointInEmptyGraph { entry });
            }
            if self.max_level != 0 {
                return Err(GraphDefect::MaxLevelInEmptyGraph {
                    max_level: self.max_level,
                });
            }
            return Ok(());
        }

        let entry = self.entry.ok_or(GraphDefect::MissingEntryPoint)?;
        let entry_level = self
            .adjacency
            .get(entry as usize)
            .ok_or(GraphDefect::EntryPointUnknown { entry })?
            .level();
        if entry_level != self.max_level {
            return Err(GraphDefect::EntryPointBelowMaxLevel {
                entry,
                level: entry_level,
                max_level: self.max_level,
            });
        }

        for (id, point) in self.adjacency.iter().enumerate() {
            let id = u32::try_from(id).unwrap_or(u32::MAX);
            let expected_level = self.sampler.level_for(id);
            if point.level() != expected_level {
                return Err(GraphDefect::SampledLevelMismatch {
                    id,
                    expected: expected_level,
                    actual: point.level(),
                });
            }
            if point.level() > self.max_level {
                return Err(GraphDefect::LevelAboveMax {
                    id,
                    level: point.level(),
                    max_level: self.max_level,
                });
            }
            for (layer, neighbours) in point.layers.iter().enumerate() {
                let budget = self.params.degree_at(layer);
                if neighbours.len() > budget {
                    return Err(GraphDefect::DegreeExceeded {
                        id,
                        layer,
                        held: neighbours.len(),
                        budget,
                    });
                }
                let mut seen = neighbours.clone();
                seen.sort_unstable();
                let before = seen.len();
                seen.dedup();
                if seen.len() != before {
                    return Err(GraphDefect::DuplicateNeighbour { id, layer });
                }
                for &neighbour in neighbours {
                    if neighbour == id {
                        return Err(GraphDefect::SelfEdge { id, layer });
                    }
                    let target = self.adjacency.get(neighbour as usize).ok_or(
                        GraphDefect::NeighbourUnknown {
                            id,
                            layer,
                            neighbour,
                        },
                    )?;
                    // The invariant the previous engine violated: an edge
                    // may only exist in a layer BOTH endpoints occupy.
                    if !target.participates_in(layer) {
                        return Err(GraphDefect::EdgeAboveNeighbourLevel {
                            id,
                            layer,
                            neighbour,
                            neighbour_level: target.level(),
                        });
                    }
                    if !target.neighbours(layer).contains(&id) {
                        return Err(GraphDefect::MissingReciprocalEdge {
                            id,
                            layer,
                            neighbour,
                        });
                    }
                }
            }
        }

        // Every point must be reachable from the entry at layer 0, or a
        // query can never find it however wide the beam.
        let mut reached = vec![false; count];
        let mut stack = vec![entry];
        if let Some(slot) = reached.get_mut(entry as usize) {
            *slot = true;
        }
        let mut reached_count = 1usize;
        while let Some(current) = stack.pop() {
            for &neighbour in self.neighbours_at(current, 0) {
                let Some(slot) = reached.get_mut(neighbour as usize) else {
                    continue;
                };
                if !*slot {
                    *slot = true;
                    reached_count += 1;
                    stack.push(neighbour);
                }
            }
        }
        if reached_count != count {
            let first = reached.iter().position(|seen| !seen).unwrap_or(0);
            return Err(GraphDefect::Unreachable {
                reached: reached_count,
                total: count,
                first_unreachable: u32::try_from(first).unwrap_or(u32::MAX),
            });
        }
        Ok(())
    }

    /// Verify graph structure against the exact caller-owned vector store.
    ///
    /// # Errors
    ///
    /// Returns [`GraphDefect::StoreCardinalityMismatch`] before structural
    /// verification when graph ids and store rows do not describe the same
    /// universe; otherwise returns the first defect from [`Self::verify`].
    pub(super) fn verify_for_store<D: VectorDistance>(&self, store: &D) -> Result<(), GraphDefect> {
        self.verify_store_cardinality(store)?;
        self.verify()
    }

    fn verify_store_cardinality<D: VectorDistance>(&self, store: &D) -> Result<(), GraphDefect> {
        let graph_points = self.adjacency.len();
        let store_rows = store.len();
        if graph_points != store_rows {
            return Err(GraphDefect::StoreCardinalityMismatch {
                graph_points,
                store_rows,
            });
        }
        Ok(())
    }
}
