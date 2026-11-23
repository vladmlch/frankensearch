#[cfg(feature = "pruning-conformance")]
use super::ConformanceQueryWorkPhase;
use super::boolean_scorers::{
    conservative_bound_sum, conservative_optional_bound_sum, finite_score,
};
use super::reference_scorer::ReferenceScorer;
use super::term_scorer::CompetitiveTermCursor;
use super::union_scorer::{BufferedUnionScorer, CompetitiveCandidates, UnionPruningStrategy};
use super::{
    ArgusError, BMW_MIN_CLAUSES, BMW_MIN_TOTAL_COST, MAX_SCORE_MAX_CLAUSES, UNION_HORIZON,
};

impl BufferedUnionScorer<'_> {
    pub(in crate::argus) fn competitive_candidates(
        &self,
        window_start: u32,
        horizon_end: u64,
        cutoff: f32,
    ) -> Result<Option<CompetitiveCandidates>, ArgusError> {
        // Grouped MaxScore: a nested union whose active children are all
        // pure-term groups (multi-field terms) rather than direct terms
        // (`bd-quill-e8-perf-doctrine-x4e4.5.1`). Dormant until the query gate
        // opens BLOCKMAX for group terms — without their whole-term ceilings
        // `competitive_is_groupable` is `false`, so this branch is skipped and
        // nested unions keep scoring exhaustively (behavior-neutral today).
        if (2..=MAX_SCORE_MAX_CLAUSES).contains(&self.active.len())
            && !self
                .active
                .iter()
                .all(ReferenceScorer::competitive_is_direct_term)
            && self
                .active
                .iter()
                .all(ReferenceScorer::competitive_is_groupable)
        {
            let Some(candidates) =
                Self::grouped_max_score_candidates(&self.active, horizon_end, cutoff)?
            else {
                return Ok(None);
            };
            if candidates
                .docs
                .iter()
                .any(|doc| *doc < window_start || u64::from(*doc) >= horizon_end)
            {
                return Err(ArgusError::CursorInvariant(
                    "grouped competitive candidate escaped its union window",
                ));
            }
            return Ok(Some(candidates));
        }
        let strategy = match self.active.len() {
            2..=MAX_SCORE_MAX_CLAUSES
                if self
                    .active
                    .iter()
                    .all(ReferenceScorer::competitive_is_direct_term) =>
            {
                UnionPruningStrategy::MaxScore
            }
            2..=MAX_SCORE_MAX_CLAUSES => return Ok(None),
            BMW_MIN_CLAUSES..
                if self
                    .active
                    .iter()
                    .map(ReferenceScorer::cost)
                    .fold(0_u64, u64::saturating_add)
                    >= BMW_MIN_TOTAL_COST =>
            {
                UnionPruningStrategy::BlockMaxWand
            }
            _ => return Ok(None),
        };
        match strategy {
            UnionPruningStrategy::MaxScore => {
                let mut smallest = f32::INFINITY;
                for scorer in &self.active {
                    let Some(bound) = scorer.competitive_score_upper_bound() else {
                        return Ok(None);
                    };
                    smallest = smallest.min(bound);
                }
                if smallest >= cutoff {
                    return Ok(None);
                }
            }
            UnionPruningStrategy::BlockMaxWand => {
                if self
                    .active
                    .iter()
                    .any(|scorer| !scorer.competitive_supports_block_max())
                {
                    return Ok(None);
                }
            }
        }
        let mut cursors = Vec::new();
        cursors
            .try_reserve_exact(self.active.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "competitive term cursors",
                count: self.active.len(),
            })?;
        for scorer in &self.active {
            let Some(cursor) = scorer.competitive_term_cursor() else {
                return Ok(None);
            };
            cursors.push(cursor);
        }
        let candidates = match strategy {
            UnionPruningStrategy::MaxScore => {
                let Some(candidates) =
                    Self::max_score_candidates(&mut cursors, horizon_end, cutoff)?
                else {
                    return Ok(None);
                };
                candidates
            }
            UnionPruningStrategy::BlockMaxWand => {
                Self::block_max_wand_candidates(&mut cursors, horizon_end, cutoff)?
            }
        };
        if candidates
            .docs
            .iter()
            .any(|doc| *doc < window_start || u64::from(*doc) >= horizon_end)
        {
            return Err(ArgusError::CursorInvariant(
                "competitive candidate escaped its union window",
            ));
        }
        Ok(Some(candidates))
    }

    fn max_score_candidates(
        cursors: &mut [CompetitiveTermCursor<'_>],
        horizon_end: u64,
        cutoff: f32,
    ) -> Result<Option<CompetitiveCandidates>, ArgusError> {
        let mut by_upper_bound = Vec::new();
        by_upper_bound
            .try_reserve_exact(cursors.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "MaxScore bound order",
                count: cursors.len(),
            })?;
        by_upper_bound.extend(0..cursors.len());
        by_upper_bound.sort_unstable_by(|left, right| {
            cursors[*left]
                .term_upper_bound()
                .total_cmp(&cursors[*right].term_upper_bound())
                .then_with(|| left.cmp(right))
        });

        let mut nonessential = 0_usize;
        while nonessential < by_upper_bound.len() {
            let prefix = &by_upper_bound[..=nonessential];
            let bound = conservative_bound_sum(
                prefix
                    .iter()
                    .map(|index| cursors[*index].term_upper_bound()),
            )
            .ok_or(ArgusError::CursorInvariant(
                "MaxScore bound sum was not finite and conservative",
            ))?;
            if bound < cutoff {
                nonessential += 1;
            } else {
                break;
            }
        }
        if nonessential == 0 {
            return Ok(None);
        }

        let maximum_candidates = UNION_HORIZON
            .checked_mul(cursors.len().saturating_sub(nonessential))
            .ok_or(ArgusError::Allocation {
                resource: "MaxScore candidates",
                count: usize::MAX,
            })?;
        let mut docs = Vec::new();
        docs.try_reserve_exact(maximum_candidates)
            .map_err(|_| ArgusError::Allocation {
                resource: "MaxScore candidates",
                count: maximum_candidates,
            })?;
        for &index in &by_upper_bound[nonessential..] {
            while let Some(doc) = cursors[index].doc() {
                if u64::from(doc) >= horizon_end {
                    break;
                }
                docs.push(doc);
                cursors[index].next()?;
            }
        }
        docs.sort_unstable();
        docs.dedup();
        Ok(Some(CompetitiveCandidates {
            strategy: UnionPruningStrategy::MaxScore,
            docs,
            blocks_skipped: 0,
        }))
    }

    /// Grouped `MaxScore` over pure-term groups (nested multi-field unions):
    /// the direct analog of [`Self::max_score_candidates`] with each group's
    /// conservative whole-group ceiling as the partition key and each group's
    /// per-term competitive cursors as the doc source. A candidate document
    /// must appear in an *essential* group; documents confined to the
    /// low-ceiling non-essential groups cannot reach `cutoff` and are skipped.
    /// Fails closed (`Ok(None)` → exhaustive) if any group lacks a ceiling or a
    /// competitive cursor. Rank-safety and f32 score-bit parity are gated by
    /// the exhaustive parity harness that lands with the query gate
    /// (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
    fn grouped_max_score_candidates(
        groups: &[ReferenceScorer<'_>],
        horizon_end: u64,
        cutoff: f32,
    ) -> Result<Option<CompetitiveCandidates>, ArgusError> {
        let mut ceilings = Vec::new();
        ceilings
            .try_reserve_exact(groups.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "grouped MaxScore ceilings",
                count: groups.len(),
            })?;
        for group in groups {
            let Some(bound) = group.competitive_score_upper_bound() else {
                return Ok(None);
            };
            ceilings.push(bound);
        }

        let mut by_upper_bound = Vec::new();
        by_upper_bound
            .try_reserve_exact(groups.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "grouped MaxScore bound order",
                count: groups.len(),
            })?;
        by_upper_bound.extend(0..groups.len());
        by_upper_bound.sort_unstable_by(|left, right| {
            ceilings[*left]
                .total_cmp(&ceilings[*right])
                .then_with(|| left.cmp(right))
        });

        let mut nonessential = 0_usize;
        while nonessential < by_upper_bound.len() {
            let bound = conservative_bound_sum(
                by_upper_bound[..=nonessential]
                    .iter()
                    .map(|index| ceilings[*index]),
            )
            .ok_or(ArgusError::CursorInvariant(
                "grouped MaxScore bound sum was not finite and conservative",
            ))?;
            if bound < cutoff {
                nonessential += 1;
            } else {
                break;
            }
        }
        if nonessential == 0 {
            return Ok(None);
        }

        let mut docs = Vec::new();
        for &index in &by_upper_bound[nonessential..] {
            if !groups[index].group_competitive_docs(horizon_end, &mut docs)? {
                return Ok(None);
            }
        }
        docs.sort_unstable();
        docs.dedup();
        Ok(Some(CompetitiveCandidates {
            strategy: UnionPruningStrategy::MaxScore,
            docs,
            blocks_skipped: 0,
        }))
    }

    fn block_max_wand_candidates(
        cursors: &mut [CompetitiveTermCursor<'_>],
        horizon_end: u64,
        cutoff: f32,
    ) -> Result<CompetitiveCandidates, ArgusError> {
        let mut docs = Vec::new();
        docs.try_reserve_exact(UNION_HORIZON)
            .map_err(|_| ArgusError::Allocation {
                resource: "block-max WAND candidates",
                count: UNION_HORIZON,
            })?;
        let mut order = Vec::new();
        order
            .try_reserve_exact(cursors.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "block-max WAND pivot order",
                count: cursors.len(),
            })?;
        let horizon_last = u32::try_from(horizon_end.saturating_sub(1).min(u64::from(u32::MAX)))
            .map_err(|_| ArgusError::CursorInvariant("union horizon last doc exceeds u32"))?;
        let mut blocks_skipped = 0_u64;

        loop {
            order.clear();
            order.extend(cursors.iter().enumerate().filter_map(|(index, cursor)| {
                cursor
                    .doc()
                    .filter(|doc| u64::from(*doc) < horizon_end)
                    .map(|_| index)
            }));
            if order.is_empty() {
                break;
            }
            order.sort_unstable_by(|left, right| {
                cursors[*left]
                    .doc()
                    .cmp(&cursors[*right].doc())
                    .then_with(|| left.cmp(right))
            });

            let interval_end = order
                .iter()
                .filter_map(|index| cursors[*index].current_block_last_doc())
                .min()
                .ok_or(ArgusError::CursorInvariant(
                    "block-max cursor has no current block end",
                ))?
                .min(horizon_last);
            let relevant_count = order
                .iter()
                .take_while(|index| {
                    cursors[**index]
                        .doc()
                        .is_some_and(|doc| doc <= interval_end)
                })
                .count();
            let relevant = &order[..relevant_count];
            let block_bound = conservative_optional_bound_sum(
                relevant
                    .iter()
                    .map(|index| cursors[*index].current_block_upper_bound()),
            )
            .ok_or(ArgusError::CursorInvariant(
                "block-max bound sum was not finite and conservative",
            ))?;
            if block_bound < cutoff {
                for &index in relevant {
                    if interval_end == u32::MAX {
                        while cursors[index].next()?.is_some() {}
                    } else {
                        cursors[index].advance(interval_end + 1)?;
                    }
                    blocks_skipped = blocks_skipped.saturating_add(1);
                }
                continue;
            }

            let mut pivot = None;
            for pivot_index in 0..order.len() {
                let bound = conservative_bound_sum(
                    order[..=pivot_index]
                        .iter()
                        .map(|index| cursors[*index].term_upper_bound()),
                )
                .ok_or(ArgusError::CursorInvariant(
                    "WAND pivot bound sum was not finite and conservative",
                ))?;
                if bound >= cutoff {
                    pivot = Some(pivot_index);
                    break;
                }
            }
            let Some(pivot_index) = pivot else {
                break;
            };
            let pivot_doc = cursors[order[pivot_index]]
                .doc()
                .ok_or(ArgusError::CursorInvariant("WAND pivot is exhausted"))?;
            let minimum_doc = cursors[order[0]]
                .doc()
                .ok_or(ArgusError::CursorInvariant("WAND minimum is exhausted"))?;
            if minimum_doc < pivot_doc {
                for &index in &order[..pivot_index] {
                    if cursors[index].doc().is_some_and(|doc| doc < pivot_doc) {
                        cursors[index].advance(pivot_doc)?;
                    }
                }
                continue;
            }

            docs.push(pivot_doc);
            for &index in &order {
                if cursors[index].doc() == Some(pivot_doc) {
                    cursors[index].next()?;
                } else {
                    break;
                }
            }
        }

        Ok(CompetitiveCandidates {
            strategy: UnionPruningStrategy::BlockMaxWand,
            docs,
            blocks_skipped,
        })
    }

    pub(in crate::argus) fn fill_candidate_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
        candidates: &[u32],
    ) -> Result<(), ArgusError> {
        if self.tantivy_topdocs_term_union {
            return self.fill_tantivy_topdocs_candidate_window(
                window_start,
                horizon_end,
                candidates,
            );
        }

        // Candidate discovery advances only shadow cursors. Real cursors still
        // visit selected documents in their current scorer-vector order, so
        // pruning cannot regroup ordinary exhaustive contributions. That holds
        // for the ordinary buffered union only; a ranked direct-term root pins
        // Tantivy's doc-major order and takes the branch above.
        let mut index = 0;
        while index < self.active.len() {
            loop {
                let Some(doc) = self.active[index].doc() else {
                    self.active.swap_remove(index);
                    break;
                };
                if u64::from(doc) >= horizon_end {
                    index += 1;
                    break;
                }
                let candidate_index = candidates.partition_point(|candidate| *candidate < doc);
                let Some(&candidate) = candidates.get(candidate_index) else {
                    #[cfg(feature = "pruning-conformance")]
                    self.set_conformance_query_work_context(
                        ConformanceQueryWorkPhase::Seek,
                        None,
                        false,
                    );
                    Self::advance_scorer_to_horizon(&mut self.active[index], horizon_end)?;
                    if self.active[index].doc().is_none() {
                        self.active.swap_remove(index);
                    } else {
                        index += 1;
                    }
                    break;
                };
                if doc < candidate {
                    #[cfg(feature = "pruning-conformance")]
                    self.set_conformance_query_work_context(
                        ConformanceQueryWorkPhase::Seek,
                        Some(candidate),
                        false,
                    );
                    self.active[index].advance(candidate)?;
                    continue;
                }
                let offset = usize::try_from(u64::from(doc) - u64::from(window_start))
                    .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
                let contribution = self.active[index].score()?;
                let total = self.score_window[offset].unwrap_or(0.0) + contribution;
                self.score_window[offset] = Some(finite_score(total, doc)?);
                #[cfg(feature = "pruning-conformance")]
                self.set_conformance_query_work_context(
                    ConformanceQueryWorkPhase::Drain,
                    Some(doc),
                    true,
                );
                self.active[index].next()?;
                if self.active[index].doc().is_none() {
                    self.active.swap_remove(index);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Fill one candidate-restricted window in exactly the doc-major order
    /// [`Self::fill_tantivy_topdocs_term_window`] would have produced, scoring
    /// only the documents competitive discovery selected.
    ///
    /// Shipping rank pruning is precisely the ranked direct-term root that sets
    /// `tantivy_topdocs_term_union`, so summing a pruned window in
    /// scorer-vector order made a single query mix two f32 addition orders:
    /// Tantivy's in its first, always-exhaustive window and scorer-major in
    /// every later pruned one. The drift also outlived the window, because the
    /// pinned sort is stable and the next refill inherits whatever permutation
    /// this one leaves (`bd-quill-pruned-topdocs-term-order-9wu3p`).
    ///
    /// What has to be reproduced is the traversal order, not just the
    /// per-document sum — but it does not have to be reproduced by stepping.
    /// Under that stable sort the equal-document order is "decreasing previous
    /// posting": a scorer stepping onto a document sorts ahead of everyone
    /// already parked there. With `X = {5, 8, 10}`, `Y = {7, 10}`, `Z = {10}`
    /// and `10` the only candidate, stepping yields `X + Y + Z` because `X`'s
    /// stop at 8 lets `Y` overtake it, and seeking straight to 10 rebuilds that
    /// same order from the previous postings 8, 7 and none. So the arrival
    /// predecessor is the whole of what a seek loses, and a cursor that
    /// validates a block table already knows it
    /// ([`SealedPostingCursor::sealed_doc_before`](super::segment_cursors::SealedPostingCursor::sealed_doc_before), reached only through
    /// [`ValidatedArrivalCursor::arrival_doc_before`](super::cursor::ValidatedArrivalCursor::arrival_doc_before)).
    ///
    /// Two cases resist that reconstruction and take the walking fill, which
    /// simulates the traversal exactly.
    ///
    /// The first is the exhaustion of a clause: a term that runs out mid-window
    /// is dropped by a positional `swap_remove`, which moves the vector's last
    /// scorer into the hole and can invert an equal-document pair whose
    /// postings say nothing about it. For sealed clauses that is at most one
    /// window per clause for the whole query, since a clause exhausts once.
    ///
    /// The second is a clause whose storage cannot report arrival metadata at
    /// all, which today means every non-sealed cursor — Delta above all. That
    /// is not bounded per clause: one live Delta clause routes *every* pruned
    /// window here. It is still not a lost skip, because
    /// [`DeltaPostingCursor::advance`](super::segment_cursors::DeltaPostingCursor::advance) is itself a per-posting walk with no
    /// block-skip structure and no per-block last document, so the walking fill
    /// costs a Delta clause what its own seek to the horizon would have cost.
    /// The seek repair is scoped to sealed block-max clauses, which is where
    /// discovery can skip blocks that scoring cursors would otherwise re-walk.
    /// A Delta predecessor is not derivable from its live-filtered forward walk
    /// without a second walk, so this deliberately does not fake one.
    pub(in crate::argus) fn fill_tantivy_topdocs_candidate_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
        candidates: &[u32],
    ) -> Result<(), ArgusError> {
        if self.arrival_order_is_reconstructible(horizon_end) {
            return self.fill_tantivy_topdocs_seeking_window(window_start, horizon_end, candidates);
        }
        self.pruning_stats.arrival_walk_windows =
            self.pruning_stats.arrival_walk_windows.saturating_add(1);
        self.fill_tantivy_topdocs_walking_window(window_start, horizon_end, candidates)
    }

    /// Whether every active clause can rebuild this window's arrival order from
    /// validated metadata rather than from steps.
    ///
    /// Two conditions, neither of them a claim taken on trust. The clause must
    /// hold the crate's own validated cursor — [`TermCursor::Validated`](super::cursor::TermCursor::Validated), which
    /// no wrapper can enter — and that cursor's reachable last document must lie
    /// at or beyond this horizon, because a clause exhausting mid-window is
    /// `swap_remove`d and that permutes the vector by position.
    ///
    /// A reachable last document is a `u32`, so a horizon past `u32::MAX` fails
    /// the second condition, which is what keeps the seeking fill's horizon seek
    /// in range.
    fn arrival_order_is_reconstructible(&self, horizon_end: u64) -> bool {
        self.active.iter().all(|scorer| {
            scorer
                .arrival_final_doc()
                .is_some_and(|last| u64::from(last) >= horizon_end)
        })
    }

    /// Fill a candidate-restricted doc-major window with seeks, reconstructing
    /// each pivot's equal-document order from arrival predecessors.
    ///
    /// Real scoring cursors move only to selected candidates and then to the
    /// horizon, so their *moves* are seeks rather than one step per posting.
    ///
    /// That is a statement about moves only. The order reconstruction this fill
    /// depends on also issues predecessor probes; those probes decode at most
    /// one block per block they descend into, and are charged for exactly that
    /// ([`Self::refine_arrival_group`]). Whether the total is *cheaper* than the
    /// walk it replaces is still unmeasured, and no such claim is made here. The
    /// property this fill exists to hold is the *order*: pruned windows
    /// accumulate in the same doc-major sequence as exhaustive ones.
    ///
    /// The order is rebuilt at each pivot by [`Self::sort_active_by_arrival`]
    /// against a `floor` — the most recent document at which the vector order
    /// was known correct. It starts at `window_start`, where the previous fill
    /// left a correct order, and advances to each scored candidate. Two clauses
    /// whose predecessor chains agree all the way down to the floor were parked
    /// on the same document there, so the order the floor left them in is the
    /// order stepping would have kept, and the walk stops instead of descending
    /// into history the window cannot observe.
    fn fill_tantivy_topdocs_seeking_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
        candidates: &[u32],
    ) -> Result<(), ArgusError> {
        let mut floor = window_start;
        let mut arrival_keys = Vec::new();
        for &candidate in candidates {
            if u64::from(candidate) >= horizon_end {
                break;
            }
            #[cfg(feature = "pruning-conformance")]
            self.set_conformance_query_work_context(
                ConformanceQueryWorkPhase::Seek,
                Some(candidate),
                false,
            );
            for scorer in &mut self.active {
                if scorer.doc().is_some_and(|doc| doc < candidate) {
                    scorer.advance(candidate)?;
                }
            }
            // Every clause now sits at or beyond the candidate, so the pivot
            // group — if the candidate is held at all — is the sorted prefix.
            self.sort_active_by_arrival(floor, &mut arrival_keys)?;
            floor = candidate;
            let pivot_len = self
                .active
                .iter()
                .take_while(|scorer| scorer.doc() == Some(candidate))
                .count();
            if pivot_len == 0 {
                continue;
            }
            let offset = usize::try_from(u64::from(candidate) - u64::from(window_start))
                .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
            let mut score = 0.0_f32;
            for scorer in &mut self.active[..pivot_len] {
                score += scorer.score()?;
            }
            self.score_window[offset] = Some(finite_score(score, candidate)?);
            #[cfg(feature = "pruning-conformance")]
            self.set_conformance_query_work_context(
                ConformanceQueryWorkPhase::Drain,
                Some(candidate),
                true,
            );
        }
        #[cfg(feature = "pruning-conformance")]
        self.set_conformance_query_work_context(ConformanceQueryWorkPhase::Seek, None, false);
        // The exhaustive fill leaves every clause on its first posting at or
        // beyond the horizon; nothing exhausts here, so this reaches the same
        // positions with one seek each.
        for scorer in &mut self.active {
            Self::advance_scorer_to_horizon(scorer, horizon_end)?;
        }
        self.sort_active_by_arrival(floor, &mut arrival_keys)
    }

    /// Restore the scorer order the stepping traversal would have left, without
    /// stepping: stable-sort by current document, then order each
    /// equal-document group by descending arrival predecessor.
    ///
    /// The stable sort is what carries the base case. A group the predecessor
    /// refinement cannot separate keeps the order the vector already held,
    /// which is the order the `floor` established.
    ///
    /// `arrival_keys` is the refinement's scratch, borrowed from the caller so
    /// a window that never ties two clauses on one document allocates nothing.
    fn sort_active_by_arrival(
        &mut self,
        floor: u32,
        arrival_keys: &mut Vec<Option<u32>>,
    ) -> Result<(), ArgusError> {
        self.active.sort_by_key(ReferenceScorer::doc);
        let mut start = 0;
        while start < self.active.len() {
            let doc = self.active[start].doc();
            let mut end = start + 1;
            while end < self.active.len() && self.active[end].doc() == doc {
                end += 1;
            }
            if end - start > 1 && doc.is_some() {
                self.refine_arrival_group(start, end, floor, arrival_keys)?;
            }
            start = end;
        }
        Ok(())
    }

    /// Order one equal-document group by descending arrival predecessor,
    /// refining a round at a time rather than comparing pairs.
    ///
    /// The pinned traversal moves an equal-document prefix to the vector's
    /// front and stably re-sorts, so a clause that just stepped onto a document
    /// precedes one already parked there, and two that stepped on together keep
    /// the order they held where they parted. Separating them is therefore a
    /// lexicographic comparison of descending posting histories — previous
    /// postings first, then the postings before those.
    ///
    /// Doing that pairwise would let one shared run be re-descended once per
    /// comparison, so the histories are walked as *cached keys* instead: every
    /// round probes each still-tied clause exactly once, from the value that
    /// clause last answered. Each probe therefore steps that clause past one of
    /// its own postings inside `(floor, group document]`, and the floor
    /// advances to each scored candidate, so those intervals are disjoint
    /// across the window. That bounds the *number of probes* by the postings the
    /// exhaustive fill would have stepped through, and it is one probe per
    /// clause whenever the tied clauses hold different documents in the gap.
    ///
    /// A probe count is still not a cost bound, but the cost it used to hide is
    /// now bounded and measured. A probe is free in the block its cursor is
    /// parked in; a descent below that block used to re-decode the block it read
    /// on *every* probe, so a tie walking a shared run through one 128-posting
    /// block cost up to one decode per posting where a forward walk costs one
    /// for the block. The cursor now retains the block a backward read lands in,
    /// and because the descent is monotone the rest of that block is served from
    /// it: `deep_backward_descent_is_charged_once` pins 32+ probes to one
    /// physical decode and one admitted posting block.
    ///
    /// What is still unproven is the comparison that matters operationally —
    /// whether this path is *cheaper* than the walk it replaces. That needs a
    /// live measurement, not a decode count, so neither this function nor its
    /// callers may claim a skip-cost advantage.
    ///
    /// A key of `None` settles a clause: either its history ended, or it
    /// reached the floor, where the order the vector already holds is
    /// authoritative. `None` sorting below `Some` is exactly right — a clause
    /// with no posting above the floor arrived earlier than one that has one.
    fn refine_arrival_group(
        &mut self,
        start: usize,
        end: usize,
        floor: u32,
        arrival_keys: &mut Vec<Option<u32>>,
    ) -> Result<(), ArgusError> {
        let doc = self.active[start].doc();
        arrival_keys.clear();
        let width = end - start;
        arrival_keys
            .try_reserve(width)
            .map_err(|_| ArgusError::Allocation {
                resource: "union arrival keys",
                count: width,
            })?;
        arrival_keys.resize(width, doc);

        let mut run_start = start;
        while run_start < end {
            let key = arrival_keys[run_start - start];
            let mut run_end = run_start + 1;
            while run_end < end && arrival_keys[run_end - start] == key {
                run_end += 1;
            }
            let Some(probe) = key else {
                // Settled: every clause in this run is ordered by the vector.
                run_start = run_end;
                continue;
            };
            if run_end - run_start == 1 {
                run_start = run_end;
                continue;
            }
            for index in run_start..run_end {
                let earlier = self.active[index].arrival_doc_before(probe)?;
                // Termination rests on this: the refinement only ends because
                // every round moves a probe strictly downward toward the floor.
                // A cursor that answered with its own probe would spin here
                // forever, so the contract is enforced rather than trusted.
                if earlier.is_some_and(|value| value >= probe) {
                    return Err(ArgusError::CursorInvariant(
                        "arrival predecessor did not move strictly below its probe",
                    ));
                }
                arrival_keys[index - start] = earlier.filter(|value| *value > floor);
            }
            // Stable insertion sort on the refreshed keys: descending, so the
            // later arrival leads. No probe happens here — the keys are cached
            // — and the run is re-scanned so its new sub-runs refine next.
            let mut index = run_start + 1;
            while index < run_end {
                let mut position = index;
                while position > run_start
                    && arrival_keys[position - 1 - start] < arrival_keys[position - start]
                {
                    self.active.swap(position - 1, position);
                    arrival_keys.swap(position - 1 - start, position - start);
                    position -= 1;
                }
                index += 1;
            }
        }
        Ok(())
    }

    /// Fill a candidate-restricted doc-major window by simulating the stepping
    /// traversal, eliding only `score` for documents discovery did not select.
    ///
    /// This is the exact fallback for the windows
    /// [`Self::arrival_order_is_reconstructible`] refuses. It costs one step
    /// per posting in the window. For a sealed union that is bounded at one
    /// window per clause per query — the window where that clause exhausts and
    /// a positional `swap_remove` permutes the vector — but for a union holding
    /// any cursor without arrival metadata it is every pruned window, which the
    /// dispatch documents rather than hides.
    fn fill_tantivy_topdocs_walking_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
        candidates: &[u32],
    ) -> Result<(), ArgusError> {
        self.active.sort_by_key(ReferenceScorer::doc);
        while let Some(doc) = self.active.first().and_then(ReferenceScorer::doc) {
            if u64::from(doc) >= horizon_end {
                break;
            }
            let pivot_len = self
                .active
                .iter()
                .take_while(|scorer| scorer.doc() == Some(doc))
                .count();
            // Discovery emits its documents sorted and deduplicated, which is
            // the same ordering the ordinary candidate fill partitions over.
            let selected = candidates.binary_search(&doc).is_ok();
            if selected {
                let offset = usize::try_from(u64::from(doc) - u64::from(window_start))
                    .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
                let mut score = 0.0_f32;
                for scorer in &mut self.active[..pivot_len] {
                    score += scorer.score()?;
                }
                self.score_window[offset] = Some(finite_score(score, doc)?);
            }
            // Same phase vocabulary the ordinary candidate fill installs: a
            // scored pivot drains, an unselected one is passed over.
            #[cfg(feature = "pruning-conformance")]
            self.set_conformance_query_work_context(
                if selected {
                    ConformanceQueryWorkPhase::Drain
                } else {
                    ConformanceQueryWorkPhase::Seek
                },
                selected.then_some(doc),
                selected,
            );

            for scorer in &mut self.active[..pivot_len] {
                scorer.next()?;
            }
            let mut index = 0;
            while index < self.active.len() {
                if self.active[index].doc().is_none() {
                    self.active.swap_remove(index);
                } else {
                    index += 1;
                }
            }
            self.active.sort_by_key(ReferenceScorer::doc);
        }
        Ok(())
    }
}
