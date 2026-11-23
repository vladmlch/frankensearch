#[cfg(feature = "pruning-conformance")]
use std::sync::Arc;

#[cfg(feature = "pruning-conformance")]
use super::QueryWorkCheckpoint;
use super::boolean_scorers::{
    conservative_optional_bound_sum, estimate_union, finite_score, shared_segment_num_docs,
};
use super::collector::{LiveDocs, TopDocsCollector};
use super::reference_scorer::{ReferenceScorer, SeekDangerResult, classify_seek_result};
use super::{ArgusError, UNION_HORIZON, UNION_HORIZON_U64};
#[cfg(feature = "pruning-conformance")]
use super::{
    ConformanceQueryWorkContext, ConformanceQueryWorkPhase, ScorerTraceSink, UnionAddTrace,
    UnionFillPath, UnionTraceEvent, active_scorer_trace_sink,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::argus) enum UnionPruningStrategy {
    MaxScore,
    BlockMaxWand,
}

/// Exact refill strategy captured only by tests and method-bound conformance.
#[cfg(any(test, feature = "pruning-conformance"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConformanceUnionRefillStrategy {
    Exhaustive,
    MaxScore,
    BlockMaxWand,
}

/// Typed, bounded witness for one direct-term union refill.
///
/// Window bounds and cutoff bits are intentionally absent from shipping
/// builds. The conformance harness uses them to prove temporal path coverage
/// without scraping operational logs or exposing terms/document identities.
#[cfg(any(test, feature = "pruning-conformance"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConformanceUnionRefill {
    pub ordinal: u64,
    pub window_start: u32,
    pub horizon_end: u64,
    pub cutoff_bits: Option<u32>,
    pub strategy: ConformanceUnionRefillStrategy,
    pub candidate_docs: u64,
    pub buffer_empty: bool,
    pub live_work_remains: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::argus) struct UnionPruningStats {
    pub(in crate::argus) max_score_windows: u64,
    pub(in crate::argus) block_max_wand_windows: u64,
    /// Posting blocks the *shadow* discovery cursors skipped.
    ///
    /// This has never been a count of scoring work avoided, and reading it as
    /// one is what made the doc-major candidate fill's traversal invisible:
    /// discovery can skip a block that the real scoring cursors then walk. The
    /// doc-major fill now seeks over the same region, so the two agree — except
    /// in the windows counted by [`Self::arrival_walk_windows`], which are the
    /// windows where the scoring cursors did walk it.
    pub(in crate::argus) blocks_skipped: u64,
    pub(in crate::argus) candidate_docs: u64,
    /// Pruned doc-major windows whose scoring cursors walked every posting
    /// instead of seeking: a clause exhausts inside them, or a clause cannot
    /// report arrival metadata at all.
    pub(in crate::argus) arrival_walk_windows: u64,
}

/// Aggregate scorer evidence suitable for stable tracing fields.
///
/// This deliberately excludes terms, document identifiers, and cutoff scores.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PruningTelemetry {
    max_score_windows: u64,
    block_max_wand_windows: u64,
    blocks_skipped: u64,
    candidate_docs: u64,
}

impl From<UnionPruningStats> for PruningTelemetry {
    fn from(stats: UnionPruningStats) -> Self {
        Self {
            max_score_windows: stats.max_score_windows,
            block_max_wand_windows: stats.block_max_wand_windows,
            blocks_skipped: stats.blocks_skipped,
            candidate_docs: stats.candidate_docs,
        }
    }
}

impl PruningTelemetry {
    /// Stable name for the pruning path observed during collection.
    #[must_use]
    pub const fn plan(self) -> &'static str {
        match (
            self.max_score_windows != 0,
            self.block_max_wand_windows != 0,
        ) {
            (false, false) => "exhaustive",
            (true, false) => "max_score",
            (false, true) => "block_max_wand",
            (true, true) => "mixed_pruning",
        }
    }

    #[must_use]
    pub const fn pruning_windows(self) -> u64 {
        self.max_score_windows
            .saturating_add(self.block_max_wand_windows)
    }

    #[must_use]
    pub const fn blocks_skipped(self) -> u64 {
        self.blocks_skipped
    }

    #[must_use]
    pub const fn candidate_docs(self) -> u64 {
        self.candidate_docs
    }
}

pub(in crate::argus) struct CompetitiveCandidates {
    pub(in crate::argus) strategy: UnionPruningStrategy,
    pub(in crate::argus) docs: Vec<u32>,
    pub(in crate::argus) blocks_skipped: u64,
}

pub(in crate::argus) struct BufferedUnionScorer<'a> {
    pub(in crate::argus) active: Vec<ReferenceScorer<'a>>,
    /// Ranked `TopDocs` over a direct frequency-term root follows Tantivy's
    /// specialized term-union traversal. That path repeatedly scores the
    /// scorers on the current minimum document, swap-removes exhausted
    /// scorers, and stably re-sorts survivors by their next document. The
    /// resulting f32 addition order is observable in raw score bits.
    pub(in crate::argus) tantivy_topdocs_term_union: bool,
    pub(in crate::argus) score_window: Vec<Option<f32>>,
    pub(in crate::argus) window_start: Option<u32>,
    pub(in crate::argus) scan_offset: usize,
    pub(in crate::argus) current: Option<u32>,
    pub(in crate::argus) current_score: f32,
    pub(in crate::argus) segment_num_docs: u32,
    pub(in crate::argus) pruning_stats: UnionPruningStats,
    #[cfg(any(test, feature = "pruning-conformance"))]
    pub(in crate::argus) conformance_refills: Vec<ConformanceUnionRefill>,
    #[cfg(feature = "pruning-conformance")]
    pub(in crate::argus) conformance_checkpoint: Option<Arc<dyn QueryWorkCheckpoint + 'a>>,
    /// Invocation-scoped trace sink captured at construction (bd-rtnwu).
    #[cfg(feature = "pruning-conformance")]
    pub(in crate::argus) trace_sink: Option<std::sync::Arc<dyn ScorerTraceSink>>,
    /// 1-based refill ordinal for trace events (bd-rtnwu).
    #[cfg(feature = "pruning-conformance")]
    pub(in crate::argus) trace_refill_ordinal: u64,
    /// Whole-group score ceiling captured once at construction, before the
    /// first refill drains any child into [`Self::score_window`].
    ///
    /// This must not be recomputed from [`Self::active`]: a group whose
    /// postings all land inside one `UNION_HORIZON` window has an empty
    /// `active` immediately after construction, yet its buffered residue can
    /// still realize scores. A sum over *all* original children stays a valid
    /// upper bound for any surviving subset — looser as the group drains,
    /// never lower than a realized score, which is the rank-safety
    /// precondition the grouped candidate path relies on (`bd-bt2t`).
    ///
    /// `None` when any original child was not a bounded direct term, so
    /// callers fail closed for the whole lifetime of the group.
    pub(in crate::argus) group_ceiling: Option<f32>,
}

impl<'a> BufferedUnionScorer<'a> {
    pub(in crate::argus) fn new(
        mut scorers: Vec<ReferenceScorer<'a>>,
        topdocs_root: bool,
        #[cfg(feature = "pruning-conformance")] conformance_checkpoint: Option<
            Arc<dyn QueryWorkCheckpoint + 'a>,
        >,
    ) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        #[cfg(feature = "pruning-conformance")]
        let trace_sink = active_scorer_trace_sink();
        #[cfg(feature = "pruning-conformance")]
        if trace_sink.is_some() {
            // Stable identity predates the empty-child retain below: a child
            // that dies at construction still had an identity in the lowered
            // tree the trace describes.
            for (ordinal, scorer) in scorers.iter_mut().enumerate() {
                scorer.construction_ordinal = u32::try_from(ordinal).ok();
            }
        }
        scorers.retain(|scorer| scorer.doc().is_some());
        let tantivy_topdocs_term_union = topdocs_root
            && scorers
                .iter()
                .all(ReferenceScorer::is_tantivy_topdocs_frequency_term);
        let mut score_window = Vec::new();
        score_window
            .try_reserve_exact(UNION_HORIZON)
            .map_err(|_| ArgusError::Allocation {
                resource: "buffered union score window",
                count: UNION_HORIZON,
            })?;
        score_window.resize(UNION_HORIZON, None);
        // Capture the whole-group ceiling from the original children, before
        // the refill below drains any of them into the score window.
        let group_ceiling = if scorers
            .iter()
            .all(ReferenceScorer::competitive_is_direct_term)
        {
            conservative_optional_bound_sum(
                scorers
                    .iter()
                    .map(ReferenceScorer::competitive_score_upper_bound),
            )
        } else {
            None
        };
        #[cfg(feature = "pruning-conformance")]
        if let Some(sink) = &trace_sink {
            sink.record(UnionTraceEvent::Constructed {
                route_specialized: tantivy_topdocs_term_union,
                topdocs_root,
                child_kinds: scorers
                    .iter()
                    .map(ReferenceScorer::trace_node_kind)
                    .collect(),
                group_ceiling_bits: group_ceiling.map(f32::to_bits),
            });
        }
        let mut scorer = Self {
            active: scorers,
            tantivy_topdocs_term_union,
            score_window,
            window_start: None,
            scan_offset: 0,
            current: None,
            current_score: 0.0,
            segment_num_docs,
            pruning_stats: UnionPruningStats::default(),
            #[cfg(any(test, feature = "pruning-conformance"))]
            conformance_refills: Vec::new(),
            #[cfg(feature = "pruning-conformance")]
            conformance_checkpoint,
            #[cfg(feature = "pruning-conformance")]
            trace_sink,
            #[cfg(feature = "pruning-conformance")]
            trace_refill_ordinal: 0,
            group_ceiling,
        };
        if scorer.refill()? {
            scorer.advance_buffered();
        }
        Ok(scorer)
    }

    pub(in crate::argus) const fn doc(&self) -> Option<u32> {
        self.current
    }

    pub(in crate::argus) fn cost(&self) -> u64 {
        self.active
            .iter()
            .map(ReferenceScorer::cost)
            .fold(0_u64, u64::saturating_add)
    }

    pub(in crate::argus) fn size_hint(&self) -> u32 {
        estimate_union(
            self.active.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    pub(in crate::argus) fn clear_buffer(&mut self) {
        self.score_window.fill(None);
        self.scan_offset = 0;
        self.current = None;
        self.current_score = 0.0;
    }

    #[cfg(feature = "pruning-conformance")]
    pub(in crate::argus) fn set_conformance_query_work_context(
        &self,
        phase: ConformanceQueryWorkPhase,
        candidate_doc: Option<u32>,
        candidate_scored: bool,
    ) {
        let (Some(checkpoint), Some(window_start)) =
            (&self.conformance_checkpoint, self.window_start)
        else {
            return;
        };
        let refill_ordinal = u64::try_from(self.conformance_refills.len())
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        checkpoint.set_conformance_query_work_context(Some(ConformanceQueryWorkContext::new(
            refill_ordinal,
            window_start,
            phase,
            candidate_doc,
            candidate_scored,
        )));
    }

    #[cfg(feature = "pruning-conformance")]
    pub(in crate::argus) fn clear_conformance_query_work_context(&self) {
        if let Some(checkpoint) = &self.conformance_checkpoint {
            checkpoint.set_conformance_query_work_context(None);
        }
    }

    /// Emit one trace event when an invocation-scoped sink is installed
    /// (bd-rtnwu).
    #[cfg(feature = "pruning-conformance")]
    pub(in crate::argus) fn trace(&self, event: UnionTraceEvent) {
        if let Some(sink) = &self.trace_sink {
            sink.record(event);
        }
    }

    /// Construction ordinal of one active child, when identity was assigned.
    #[cfg(feature = "pruning-conformance")]
    pub(in crate::argus) fn trace_ordinal(&self, runtime_index: usize) -> u32 {
        self.active
            .get(runtime_index)
            .and_then(|scorer| scorer.construction_ordinal)
            .unwrap_or(u32::MAX)
    }

    pub(in crate::argus) fn refill(&mut self) -> Result<bool, ArgusError> {
        self.refill_with_cutoff(None)
    }

    /// Whole-group score ceiling for a pure-term group: the conservative sum of
    /// every active child's whole-term ceiling. Returns `None` when the group is
    /// empty or any active child is not a bounded direct term (a nested non-term
    /// child, or a cursor without a ceiling), so callers can fail closed.
    ///
    /// Foundation for the deferred grouped-`MaxScore` lever
    /// (`bd-quill-e8-perf-doctrine-x4e4.5.1`): a nested pure-term union exposes
    /// this ceiling to a root `MaxScore` over groups. Uses the same
    /// [`conservative_optional_bound_sum`] as the term `MaxScore` path so the
    /// ceiling is rounded outward and never falls below a realized group score
    /// — a rank-safety precondition for the follow-up grouped candidate path.
    ///
    /// Reads the construction-time [`Self::group_ceiling`] rather than summing
    /// over `active`. `active` is NOT the group's remaining-work set: a group
    /// whose postings all fall inside one window drains to empty at
    /// construction while its scores live on in `score_window`, so a live sum
    /// would report `None` for a group that can still realize hits — the
    /// `max_score_windows: 0` regression in `bd-bt2t`.
    ///
    /// Safe to consume only because [`ReferenceScorer::group_competitive_docs`]
    /// now enumerates buffered residue alongside live cursors. Before that fix
    /// this returning `Some` for a drained group meant silently dropping its
    /// documents from the top-k; the two changes must stay together.
    pub(in crate::argus) const fn group_upper_bound(&self) -> Option<f32> {
        self.group_ceiling
    }

    pub(in crate::argus) fn refill_with_cutoff(
        &mut self,
        cutoff: Option<f32>,
    ) -> Result<bool, ArgusError> {
        #[cfg(feature = "pruning-conformance")]
        self.clear_conformance_query_work_context();
        self.clear_buffer();
        let Some(window_start) = self.active.iter().filter_map(ReferenceScorer::doc).min() else {
            self.window_start = None;
            return Ok(false);
        };
        self.window_start = Some(window_start);
        let horizon_end = u64::from(window_start) + UNION_HORIZON_U64;
        #[cfg(feature = "pruning-conformance")]
        {
            self.trace_refill_ordinal = self.trace_refill_ordinal.saturating_add(1);
            self.trace(UnionTraceEvent::Refill {
                ordinal: self.trace_refill_ordinal,
                window_start,
                cutoff_bits: cutoff.map(f32::to_bits),
                fill_path: if self.tantivy_topdocs_term_union {
                    UnionFillPath::TantivyTopDocsTerm
                } else {
                    UnionFillPath::Exhaustive
                },
            });
        }
        #[cfg(feature = "pruning-conformance")]
        if cutoff.is_some_and(f32::is_finite) {
            self.set_conformance_query_work_context(
                ConformanceQueryWorkPhase::Discovery,
                None,
                false,
            );
        }
        if let Some(cutoff) = cutoff
            && cutoff.is_finite()
            && let Some(candidates) =
                self.competitive_candidates(window_start, horizon_end, cutoff)?
        {
            match candidates.strategy {
                UnionPruningStrategy::MaxScore => {
                    self.pruning_stats.max_score_windows =
                        self.pruning_stats.max_score_windows.saturating_add(1);
                }
                UnionPruningStrategy::BlockMaxWand => {
                    self.pruning_stats.block_max_wand_windows =
                        self.pruning_stats.block_max_wand_windows.saturating_add(1);
                }
            }
            self.pruning_stats.blocks_skipped = self
                .pruning_stats
                .blocks_skipped
                .saturating_add(candidates.blocks_skipped);
            self.pruning_stats.candidate_docs = self
                .pruning_stats
                .candidate_docs
                .saturating_add(u64::try_from(candidates.docs.len()).unwrap_or(u64::MAX));
            #[cfg(any(test, feature = "pruning-conformance"))]
            let strategy = match candidates.strategy {
                UnionPruningStrategy::MaxScore => ConformanceUnionRefillStrategy::MaxScore,
                UnionPruningStrategy::BlockMaxWand => ConformanceUnionRefillStrategy::BlockMaxWand,
            };
            #[cfg(any(test, feature = "pruning-conformance"))]
            let candidate_docs = u64::try_from(candidates.docs.len()).unwrap_or(u64::MAX);
            self.fill_candidate_window(window_start, horizon_end, &candidates.docs)?;
            #[cfg(feature = "pruning-conformance")]
            self.clear_conformance_query_work_context();
            #[cfg(any(test, feature = "pruning-conformance"))]
            self.record_conformance_refill(
                window_start,
                horizon_end,
                Some(cutoff.to_bits()),
                strategy,
                candidate_docs,
            );
            return Ok(true);
        }
        #[cfg(feature = "pruning-conformance")]
        self.clear_conformance_query_work_context();
        self.fill_exhaustive_window(window_start, horizon_end)?;
        #[cfg(any(test, feature = "pruning-conformance"))]
        self.record_conformance_refill(
            window_start,
            horizon_end,
            cutoff.filter(|value| value.is_finite()).map(f32::to_bits),
            ConformanceUnionRefillStrategy::Exhaustive,
            u64::try_from(
                self.score_window
                    .iter()
                    .filter(|score| score.is_some())
                    .count(),
            )
            .unwrap_or(u64::MAX),
        );
        Ok(true)
    }

    #[cfg(any(test, feature = "pruning-conformance"))]
    fn record_conformance_refill(
        &mut self,
        window_start: u32,
        horizon_end: u64,
        cutoff_bits: Option<u32>,
        strategy: ConformanceUnionRefillStrategy,
        candidate_docs: u64,
    ) {
        let ordinal = u64::try_from(self.conformance_refills.len())
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        self.conformance_refills.push(ConformanceUnionRefill {
            ordinal,
            window_start,
            horizon_end,
            cutoff_bits,
            strategy,
            candidate_docs,
            buffer_empty: self.score_window.iter().all(Option::is_none),
            live_work_remains: !self.active.is_empty(),
        });
    }

    fn fill_exhaustive_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
    ) -> Result<(), ArgusError> {
        if self.tantivy_topdocs_term_union {
            return self.fill_tantivy_topdocs_term_window(window_start, horizon_end);
        }

        let mut index = 0;
        while index < self.active.len() {
            loop {
                let Some(doc) = self.active[index].doc() else {
                    #[cfg(feature = "pruning-conformance")]
                    self.trace(UnionTraceEvent::SwapRemove {
                        runtime_index: index,
                        child_construction_ordinal: self.trace_ordinal(index),
                    });
                    self.active.swap_remove(index);
                    break;
                };
                if u64::from(doc) >= horizon_end {
                    index += 1;
                    break;
                }
                let offset = usize::try_from(u64::from(doc) - u64::from(window_start))
                    .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
                let contribution = self.active[index].score()?;
                let accumulator = self.score_window[offset].unwrap_or(0.0);
                let total = accumulator + contribution;
                self.score_window[offset] = Some(finite_score(total, doc)?);
                #[cfg(feature = "pruning-conformance")]
                self.trace(UnionTraceEvent::Add(UnionAddTrace {
                    refill_ordinal: self.trace_refill_ordinal,
                    window_start,
                    fill_path: UnionFillPath::Exhaustive,
                    child_construction_ordinal: self.trace_ordinal(index),
                    runtime_index: index,
                    doc,
                    accumulator_bits: accumulator.to_bits(),
                    contribution_bits: contribution.to_bits(),
                    result_bits: total.to_bits(),
                }));
                self.active[index].next()?;
                if self.active[index].doc().is_none() {
                    #[cfg(feature = "pruning-conformance")]
                    self.trace(UnionTraceEvent::SwapRemove {
                        runtime_index: index,
                        child_construction_ordinal: self.trace_ordinal(index),
                    });
                    self.active.swap_remove(index);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Fill one exhaustive window in the same evolving scorer order as
    /// Tantivy's direct frequency-term `BlockWAND` specialization while its
    /// threshold is `Score::MIN`.
    ///
    /// Tantivy initially performs a stable sort by current document. At each
    /// pivot it sums the equal-document prefix, advances that prefix,
    /// swap-removes every exhausted scorer, then stably sorts the survivors by
    /// their new document. Quill buffers documents in windows, but preserving
    /// these mutations across refill boundaries retains the same per-document
    /// f32 accumulation order.
    pub(in crate::argus) fn fill_tantivy_topdocs_term_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
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
            let offset = usize::try_from(u64::from(doc) - u64::from(window_start))
                .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
            let mut score = 0.0_f32;
            // Index-addressed on purpose: the pruning-conformance trace below
            // re-borrows `self` (`trace`/`trace_ordinal`) while the loop walks
            // `self.active`, so `iter_mut()` cannot coexist; under default
            // features the index still addresses the pivot prefix.
            #[allow(clippy::needless_range_loop)]
            for index in 0..pivot_len {
                let contribution = self.active[index].score()?;
                #[cfg(feature = "pruning-conformance")]
                let accumulator = score;
                score += contribution;
                #[cfg(feature = "pruning-conformance")]
                self.trace(UnionTraceEvent::Add(UnionAddTrace {
                    refill_ordinal: self.trace_refill_ordinal,
                    window_start,
                    fill_path: UnionFillPath::TantivyTopDocsTerm,
                    child_construction_ordinal: self.trace_ordinal(index),
                    runtime_index: index,
                    doc,
                    accumulator_bits: accumulator.to_bits(),
                    contribution_bits: contribution.to_bits(),
                    result_bits: score.to_bits(),
                }));
            }
            self.score_window[offset] = Some(finite_score(score, doc)?);

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

    pub(in crate::argus) fn advance_scorer_to_horizon(
        scorer: &mut ReferenceScorer<'_>,
        horizon_end: u64,
    ) -> Result<(), ArgusError> {
        if let Ok(target) = u32::try_from(horizon_end) {
            scorer.advance(target)?;
            return Ok(());
        }
        while scorer.next()?.is_some() {}
        Ok(())
    }

    pub(in crate::argus) fn advance_buffered(&mut self) -> Option<u32> {
        let start = self.window_start?;
        while self.scan_offset < self.score_window.len() {
            let offset = self.scan_offset;
            self.scan_offset += 1;
            if let Some(score) = self.score_window[offset].take() {
                let doc = u64::from(start) + offset as u64;
                let doc = u32::try_from(doc).ok()?;
                self.current = Some(doc);
                self.current_score = score;
                return self.current;
            }
        }
        None
    }

    pub(in crate::argus) fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        if self.advance_buffered().is_some() {
            return Ok(self.current);
        }
        if !self.refill()? {
            return Ok(None);
        }
        Ok(self.advance_buffered())
    }

    pub(in crate::argus) fn collect_top_docs<L>(
        &mut self,
        collector: &mut TopDocsCollector,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        while let Some(doc) = self.current {
            let score = self.score()?;
            if live_docs.is_live(doc) {
                collector.record_live(doc, Some(score))?;
            }
            if self.advance_buffered().is_some() {
                continue;
            }
            loop {
                let cutoff = collector.competitive_cutoff_score();
                if !self.refill_with_cutoff(cutoff)? {
                    return Ok(());
                }
                if self.advance_buffered().is_some() {
                    break;
                }
            }
        }
        Ok(())
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        if self.window_start.is_some_and(|start| {
            u64::from(target).saturating_sub(u64::from(start)) < UNION_HORIZON_U64
        }) {
            while self.current.is_some_and(|doc| doc < target) {
                self.next()?;
            }
            return Ok(self.current);
        }

        self.clear_buffer();
        let mut index = 0;
        while index < self.active.len() {
            if self.active[index].doc().is_some_and(|doc| doc < target) {
                self.active[index].advance(target)?;
            }
            if self.active[index].doc().is_none() {
                self.active.swap_remove(index);
            } else {
                index += 1;
            }
        }
        if !self.refill()? {
            return Ok(None);
        }
        Ok(self.advance_buffered())
    }

    pub(in crate::argus) fn seek_danger(
        &mut self,
        target: u32,
    ) -> Result<SeekDangerResult, ArgusError> {
        let Some(current) = self.current else {
            return Ok(SeekDangerResult::SeekLowerBound(None));
        };
        // A target at or before the current document is answered by the
        // buffer, exactly as `seek` answers it. The children have already been
        // drained past this window, so their positions are not lower bounds
        // for the documents still buffered in it: consulting them would skip
        // every buffered match between `target` and the next child document.
        if current >= target {
            return classify_seek_result(target, Some(current));
        }
        if self.window_start.is_some_and(|start| {
            u64::from(target).saturating_sub(u64::from(start)) < UNION_HORIZON_U64
                && target >= start
        }) {
            return classify_seek_result(target, self.seek(target)?);
        }

        let mut minimum_lower_bound = None;
        let mut found = false;
        for scorer in &mut self.active {
            match scorer.seek_danger(target)? {
                SeekDangerResult::Found => {
                    found = true;
                    break;
                }
                SeekDangerResult::SeekLowerBound(Some(lower_bound)) => {
                    minimum_lower_bound = Some(
                        minimum_lower_bound
                            .map_or(lower_bound, |minimum: u32| minimum.min(lower_bound)),
                    );
                }
                SeekDangerResult::SeekLowerBound(None) => {}
            }
        }
        if found {
            let result = self.seek(target)?;
            if result != Some(target) {
                return Err(ArgusError::CursorInvariant(
                    "union danger seek found a child but not the union target",
                ));
            }
            return Ok(SeekDangerResult::Found);
        }
        Ok(SeekDangerResult::SeekLowerBound(minimum_lower_bound))
    }
    pub(in crate::argus) fn score(&self) -> Result<f32, ArgusError> {
        let doc = self
            .current
            .ok_or(ArgusError::CursorInvariant("cannot score exhausted union"))?;
        finite_score(self.current_score, doc)
    }
}
