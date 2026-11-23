use crate::argus::{ArgusError, DocSetCollector, QueryWorkKind, TopDocsCollector};
use crate::query::{
    BooleanClause, Occur, Query, QueryDiagnostic, QueryExplanation, QueryNode, QueryValue,
    canonicalize_query, classify_query,
};
use crate::schema::FieldKind;
use crate::snippet::{SnippetConfig, SnippetGenerator};
use crate::stats::{SegmentStats, SegmentStatsProvider};
use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;

use super::checkpoint::{
    PreparedSnapshotDocFreqs, QueryCheckpointHandle, prepare_snapshot_doc_freqs,
    query_work_upper_bound, sealed_segment_fanout, snippet_tail_work_upper_bound,
};
#[cfg(feature = "conformance-internals")]
use super::conformance::ConformanceCancellationStage;
#[cfg(feature = "pruning-conformance")]
use super::conformance::{ConformancePruningExecutionMode, ConformancePruningTraceSession};
use super::error_results::{
    CONTENT_FIELD, ID_FIELD, QuillClauseKind, QuillHit, QuillIndexError, QuillScoreExplanation,
    QuillSearchResult, QuillSnippetHit, QuillTermScore, check_cancel, invalid_state,
    record_pruning_telemetry,
};
#[cfg(feature = "profile-internals")]
use super::profile::{QuillProfileExecutionMode, QuillProfileSession};
use super::query_cursors::compiled_snippet_terms;
use super::query_lowering::{
    QueryLeaf, lower_query, lower_query_unscored, query_has_prunable_root_union,
    validate_query_lowering,
};
use super::reader::QuillReader;
use super::snapshot::QuillSearchSnapshot;

impl QuillReader {
    #[cfg(any(test, feature = "bench-internals"))]
    pub(super) fn execute_ranked_query(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        limit: usize,
        offset: usize,
        exact_count: bool,
        diagnostics: Vec<QueryDiagnostic>,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.execute_ranked_query_inner(
            cx,
            query,
            snapshot,
            limit,
            offset,
            exact_count,
            false,
            diagnostics,
            None,
            None,
            #[cfg(feature = "pruning-conformance")]
            None,
            #[cfg(feature = "profile-internals")]
            None,
        )
    }

    pub(super) fn execute_ranked_query_inner<'a>(
        &self,
        cx: &'a Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        limit: usize,
        offset: usize,
        exact_count: bool,
        // Test-only escape hatch (bd-1i4j4): lower the ranked direct-term
        // specialization even for an exact-count invocation, reproducing the
        // pre-repair counted route so internal mixed-snapshot tests can build
        // a same-route exhaustive reference. Every shipping caller passes
        // `false`; Tantivy's MultiCollector semantics forbid the
        // specialization on counted paths.
        force_specialized_root: bool,
        diagnostics: Vec<QueryDiagnostic>,
        supplied_checkpoint: Option<&QueryCheckpointHandle<'a>>,
        supplied_checkpoint_metering: Option<bool>,
        #[cfg(feature = "pruning-conformance")] pruning_trace: Option<
            &ConformancePruningTraceSession,
        >,
        #[cfg(feature = "profile-internals")] profile: Option<&QuillProfileSession>,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        validate_query_lowering(query, 1.0, self.schema)?;
        let work_upper_bound = query_work_upper_bound(
            query,
            snapshot,
            self.schema,
            self.config.glob_expansion_limit,
        )?;
        let (checkpoint, metering) = if let Some(checkpoint) = supplied_checkpoint {
            let metering = supplied_checkpoint_metering.ok_or_else(|| {
                invalid_state("supplied ranked-query checkpoint has no metering disposition")
            })?;
            (Arc::clone(checkpoint), metering)
        } else {
            #[cfg(feature = "profile-internals")]
            let concrete_checkpoint = profile.map_or_else(
                || {
                    self.query_checkpoint(
                        cx,
                        "search",
                        self.config.query_fuel_budget,
                        work_upper_bound,
                    )
                },
                |profile| {
                    self.profile_query_checkpoint(
                        cx,
                        "search",
                        self.config.query_fuel_budget,
                        work_upper_bound,
                        profile,
                    )
                },
            );
            #[cfg(not(feature = "profile-internals"))]
            let concrete_checkpoint = self.query_checkpoint(
                cx,
                "search",
                self.config.query_fuel_budget,
                work_upper_bound,
            );
            let metering = concrete_checkpoint.metering();
            let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
            (checkpoint, metering)
        };
        #[cfg(feature = "profile-internals")]
        if let Some(profile) = profile {
            profile.bind_work_plan(work_upper_bound, metering)?;
        }
        let prepared_doc_freqs = prepare_snapshot_doc_freqs(snapshot);
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let mut collector = if exact_count {
            TopDocsCollector::with_exact_count(limit, offset)?
        } else {
            TopDocsCollector::new(limit, offset)?
        };
        // A ranked DIRECT-TERM root may take Tantivy's specialized term-union
        // traversal (doc-major pivot-prefix f32 accumulation) only when the
        // oracle itself does: Tantivy runs that specialization for a lone
        // `TopDocs` collector, but an exact-count request pairs `TopDocs`
        // with `Count` in a MultiCollector, which forces the generic union
        // (scorer-vector accumulation order). Lowering the counted tree with
        // the specialization produced the RANKED association on the counted
        // path — one ULP off Tantivy's counted aggregate
        // (`harvested_14_score_bits_preserve_ranked_and_counted_orders`,
        // counted pin 0x4005_5FC8 vs ranked 0x4005_5FC7; bd-r1-exact-repair-
        // or-residual-1i4j4).
        let topdocs_root = limit != 0 && (force_specialized_root || !exact_count);
        let rank_pruning =
            !exact_count && topdocs_root && query_has_prunable_root_union(query, 1.0);
        let sealed_docs: u64 = keeper
            .segments()
            .iter()
            .map(|segment| u64::from(segment.doc_count()))
            .sum();
        let fanout_eligible = sealed_segment_fanout(keeper.segments().len(), sealed_docs);
        #[cfg(feature = "profile-internals")]
        if let Some(profile) = profile {
            profile.bind_fanout_eligibility(fanout_eligible)?;
        }
        let fan_out = fanout_eligible && !metering;
        #[cfg(feature = "pruning-conformance")]
        let fan_out = fan_out
            && pruning_trace.is_none_or(|trace| trace.cancellation_arm_generation().is_none());
        #[cfg(feature = "profile-internals")]
        if !keeper.segments().is_empty() {
            if let Some(profile) = profile {
                profile.bind_execution(if fan_out {
                    QuillProfileExecutionMode::Rayon
                } else {
                    QuillProfileExecutionMode::Serial
                })?;
            }
        }
        self.collect_sealed_segments(
            cx,
            &checkpoint,
            &mut collector,
            query,
            snapshot,
            prepared_doc_freqs.as_ref(),
            rank_pruning,
            topdocs_root,
            #[cfg(feature = "pruning-conformance")]
            pruning_trace,
            #[cfg(feature = "profile-internals")]
            profile,
            fan_out,
        )?;
        for delta in snapshot.delta_snapshots() {
            checkpoint.admit(QueryWorkKind::Segment, 1)?;
            #[cfg(feature = "profile-internals")]
            if let Some(profile) = profile {
                profile.record_segment_lowered(1);
            }
            let score_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_SCORE,
                phase = "score",
                residency = "delta",
                lease_base = delta.lease_base(),
                doc_count = delta.live_document_count(),
                plan = tracing::field::Empty,
                segments_touched = 1_u64,
                pruning_windows = tracing::field::Empty,
                blocks_skipped = tracing::field::Empty,
                candidate_docs = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
            let _score_entered = score_span.enter();
            let mut scorer = lower_query(
                cx,
                &checkpoint,
                query,
                1.0,
                QueryLeaf::Delta(delta),
                snapshot,
                prepared_doc_freqs.as_ref(),
                self.schema,
                self.config.glob_expansion_limit,
                rank_pruning,
                topdocs_root,
            )?;
            collector.collect(&mut scorer, delta.as_ref())?;
            record_pruning_telemetry(&score_span, scorer.pruning_telemetry());
        }
        let collect_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_COLLECT,
            phase = "collect",
            segment_count,
            segments_touched = segment_count,
            doc_count = snapshot.live_doc_count(),
            result_count = tracing::field::Empty,
            total_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _collect_timer = crate::tracing_conventions::StageTimer::new(&collect_span);
        let _collect_entered = collect_span.enter();
        let collected = collector.finish()?;
        let total_count = collected.total_count;
        let mut hits = Vec::new();
        hits.try_reserve_exact(collected.hits.len())
            .map_err(|_| invalid_state("could not allocate final hit page"))?;
        for hit in collected.hits {
            let document_id = snapshot.materialize_document_id(hit.global_docid).ok_or(
                ArgusError::MissingExternalDocId {
                    global_docid: hit.global_docid,
                },
            )?;
            hits.push(QuillHit {
                document_id: document_id.to_string(),
                global_docid: hit.global_docid,
                score: hit.score,
            });
        }
        let result_count = u64::try_from(hits.len()).unwrap_or(u64::MAX);
        collect_span.record("result_count", result_count);
        if let Some(total_count) = total_count {
            collect_span.record("total_count", total_count);
        }
        Ok(QuillSearchResult {
            hits: hits.into(),
            total_count,
            doc_count: snapshot.live_doc_count(),
            diagnostics,
        })
    }

    /// Score every sealed segment into `collector`, serially or fanned across
    /// rayon (bd-quill-e4-argus-3ycz.9).
    ///
    /// Each fan-out task scores one segment into a private collector built
    /// from `collector`'s shape via [`TopDocsCollector::empty_like`]; the
    /// partials fold back in ascending segment order. The result is exactly
    /// the serial path's: the collector's total order (score `total_cmp`
    /// descending, then ascending global docid) makes the retained top set
    /// unique, so it is independent of both the rayon schedule and the merge
    /// order. Rank pruning stays rank-safe under the weaker per-segment local
    /// cutoffs; a shared cross-task cutoff is an explicit E8 follow-up lever,
    /// not part of this change. Delta snapshots never fan out: the mutable
    /// tail is small and keeps the serial path free of rayon overhead.
    pub(super) fn collect_sealed_segments(
        &self,
        cx: &Cx,
        checkpoint: &QueryCheckpointHandle<'_>,
        collector: &mut TopDocsCollector,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
        rank_pruning: bool,
        topdocs_root: bool,
        #[cfg(feature = "pruning-conformance")] pruning_trace: Option<
            &ConformancePruningTraceSession,
        >,
        #[cfg(feature = "profile-internals")] profile: Option<&QuillProfileSession>,
        fan_out: bool,
    ) -> Result<(), QuillIndexError> {
        let keeper = snapshot.keeper_snapshot();
        #[cfg(feature = "pruning-conformance")]
        if let Some(trace) = pruning_trace {
            trace.bind_execution_mode(if fan_out {
                ConformancePruningExecutionMode::Rayon
            } else {
                ConformancePruningExecutionMode::Serial
            })?;
        }
        if fan_out {
            check_cancel(cx, "search")?;
            let collector_shape = &*collector;
            let schema = self.schema;
            let glob_expansion_limit = self.config.glob_expansion_limit;
            let segment_count = keeper.segments().len();
            let partials = keeper
                .segments()
                .par_iter()
                .enumerate()
                .map(|(segment_ordinal, segment)| {
                    debug_assert!(segment_ordinal < segment_count);
                    checkpoint.admit(QueryWorkKind::Segment, 1)?;
                    #[cfg(feature = "profile-internals")]
                    if let Some(profile) = profile {
                        profile.record_segment_lowered(1);
                    }
                    let mut local = collector_shape.empty_like()?;
                    let score_span = tracing::info_span!(
                        target: crate::tracing_conventions::TARGET,
                        crate::tracing_conventions::ARGUS_SCORE,
                        phase = "score",
                        segment_id = segment.manifest().segment_id,
                        doc_count = segment.doc_count(),
                        plan = tracing::field::Empty,
                        segments_touched = 1_u64,
                        pruning_windows = tracing::field::Empty,
                        blocks_skipped = tracing::field::Empty,
                        candidate_docs = tracing::field::Empty,
                        duration_us = tracing::field::Empty,
                    );
                    let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
                    let _score_entered = score_span.enter();
                    let mut scorer = lower_query(
                        cx,
                        checkpoint,
                        query,
                        1.0,
                        QueryLeaf::Sealed(segment),
                        snapshot,
                        prepared_doc_freqs,
                        schema,
                        glob_expansion_limit,
                        rank_pruning,
                        topdocs_root,
                    )?;
                    local.collect(&mut scorer, segment)?;
                    record_pruning_telemetry(&score_span, scorer.pruning_telemetry());
                    #[cfg(feature = "pruning-conformance")]
                    if let Some(trace) = pruning_trace {
                        trace.record_and_checkpoint(
                            u64::try_from(segment_ordinal).map_err(|_| {
                                invalid_state("Quill segment ordinal does not fit u64")
                            })?,
                            u64::from(segment.doc_count()),
                            scorer.conformance_union_refills(),
                            &self.conformance_controller,
                            cx,
                        )?;
                        check_cancel(cx, "search")?;
                    }
                    Ok(local)
                })
                .collect::<Result<Vec<_>, QuillIndexError>>()?;
            check_cancel(cx, "search")?;
            for partial in partials {
                collector.merge(partial)?;
            }
        } else {
            let segment_count = keeper.segments().len();
            for (segment_ordinal, segment) in keeper.segments().iter().enumerate() {
                debug_assert!(segment_ordinal < segment_count);
                checkpoint.admit(QueryWorkKind::Segment, 1)?;
                #[cfg(feature = "profile-internals")]
                if let Some(profile) = profile {
                    profile.record_segment_lowered(1);
                }
                let score_span = tracing::info_span!(
                    target: crate::tracing_conventions::TARGET,
                    crate::tracing_conventions::ARGUS_SCORE,
                    phase = "score",
                    segment_id = segment.manifest().segment_id,
                    doc_count = segment.doc_count(),
                    plan = tracing::field::Empty,
                    segments_touched = 1_u64,
                    pruning_windows = tracing::field::Empty,
                    blocks_skipped = tracing::field::Empty,
                    candidate_docs = tracing::field::Empty,
                    duration_us = tracing::field::Empty,
                );
                let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
                let _score_entered = score_span.enter();
                let mut scorer = lower_query(
                    cx,
                    checkpoint,
                    query,
                    1.0,
                    QueryLeaf::Sealed(segment),
                    snapshot,
                    prepared_doc_freqs,
                    self.schema,
                    self.config.glob_expansion_limit,
                    rank_pruning,
                    topdocs_root,
                )?;
                collector.collect(&mut scorer, segment)?;
                record_pruning_telemetry(&score_span, scorer.pruning_telemetry());
                #[cfg(feature = "pruning-conformance")]
                if let Some(trace) = pruning_trace {
                    trace.record_and_checkpoint(
                        u64::try_from(segment_ordinal)
                            .map_err(|_| invalid_state("Quill segment ordinal does not fit u64"))?,
                        u64::from(segment.doc_count()),
                        scorer.conformance_union_refills(),
                        &self.conformance_controller,
                        cx,
                    )?;
                    check_cancel(cx, "search")?;
                }
            }
        }
        Ok(())
    }

    /// Walk every sealed segment through the unscored id-set path, serially
    /// or fanned across rayon (bd-quill-e4-argus-3ycz.9 sibling lever).
    ///
    /// Determinism is free here: [`DocSetCollector::finish`] sorts and
    /// dedups, so any partial fold order produces the identical final id
    /// set. Delta snapshots stay on the caller's serial path.
    pub(super) fn collect_docids_sealed(
        &self,
        cx: &Cx,
        checkpoint: &QueryCheckpointHandle<'_>,
        collector: &mut DocSetCollector,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
        fan_out: bool,
    ) -> Result<(), QuillIndexError> {
        let keeper = snapshot.keeper_snapshot();
        if fan_out {
            check_cancel(cx, "collect_docids")?;
            let schema = self.schema;
            let glob_expansion_limit = self.config.glob_expansion_limit;
            let partials = keeper
                .segments()
                .par_iter()
                .map(|segment| {
                    checkpoint.admit(QueryWorkKind::Segment, 1)?;
                    let mut local = DocSetCollector::new();
                    let score_span = tracing::info_span!(
                        target: crate::tracing_conventions::TARGET,
                        crate::tracing_conventions::ARGUS_SCORE,
                        phase = "score",
                        collector = "docset",
                        segment_id = segment.manifest().segment_id,
                        doc_count = segment.doc_count(),
                        plan = "unscored",
                        segments_touched = 1_u64,
                        pruning_windows = 0_u64,
                        blocks_skipped = 0_u64,
                        candidate_docs = 0_u64,
                        duration_us = tracing::field::Empty,
                    );
                    let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
                    let _score_entered = score_span.enter();
                    let mut scorer = lower_query_unscored(
                        cx,
                        checkpoint,
                        query,
                        1.0,
                        QueryLeaf::Sealed(segment),
                        snapshot,
                        prepared_doc_freqs,
                        schema,
                        glob_expansion_limit,
                    )?;
                    local.collect(&mut scorer, segment)?;
                    Ok(local)
                })
                .collect::<Result<Vec<_>, QuillIndexError>>()?;
            check_cancel(cx, "collect_docids")?;
            for partial in partials {
                collector.merge(partial)?;
            }
        } else {
            for segment in keeper.segments() {
                checkpoint.admit(QueryWorkKind::Segment, 1)?;
                let score_span = tracing::info_span!(
                    target: crate::tracing_conventions::TARGET,
                    crate::tracing_conventions::ARGUS_SCORE,
                    phase = "score",
                    collector = "docset",
                    segment_id = segment.manifest().segment_id,
                    doc_count = segment.doc_count(),
                    plan = "unscored",
                    segments_touched = 1_u64,
                    pruning_windows = 0_u64,
                    blocks_skipped = 0_u64,
                    candidate_docs = 0_u64,
                    duration_us = tracing::field::Empty,
                );
                let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
                let _score_entered = score_span.enter();
                let mut scorer = lower_query_unscored(
                    cx,
                    checkpoint,
                    query,
                    1.0,
                    QueryLeaf::Sealed(segment),
                    snapshot,
                    prepared_doc_freqs,
                    self.schema,
                    self.config.glob_expansion_limit,
                )?;
                collector.collect(&mut scorer, segment)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "bench-internals")]
    pub(super) fn bench_collect_docids_forced_on(
        &self,
        cx: &Cx,
        query: &str,
        fan_out: bool,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<u32>, QuillIndexError> {
        let mut parsed = self.default_parser()?.parse_lenient(query);
        let _canonicalization = canonicalize_query(&mut parsed.query);
        validate_query_lowering(&parsed.query, 1.0, self.schema)?;
        let mut collector = DocSetCollector::new();
        let work_upper_bound = query_work_upper_bound(
            &parsed.query,
            snapshot,
            self.schema,
            self.config.glob_expansion_limit,
        )?;
        let concrete_checkpoint = self.query_checkpoint(
            cx,
            "collect_docids",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        let prepared_doc_freqs = prepare_snapshot_doc_freqs(snapshot);
        self.collect_docids_sealed(
            cx,
            &checkpoint,
            &mut collector,
            &parsed.query,
            snapshot,
            prepared_doc_freqs.as_ref(),
            fan_out && !metering,
        )?;
        Ok(collector.finish())
    }

    pub(super) fn execute_docid_query(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<u32>, QuillIndexError> {
        validate_query_lowering(query, 1.0, self.schema)?;
        let work_upper_bound = query_work_upper_bound(
            query,
            snapshot,
            self.schema,
            self.config.glob_expansion_limit,
        )?;
        let concrete_checkpoint = self.query_checkpoint(
            cx,
            "collect_docids",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        let prepared_doc_freqs = prepare_snapshot_doc_freqs(snapshot);
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let mut collector = DocSetCollector::new();
        let sealed_docs: u64 = keeper
            .segments()
            .iter()
            .map(|segment| u64::from(segment.doc_count()))
            .sum();
        let fan_out = sealed_segment_fanout(keeper.segments().len(), sealed_docs) && !metering;
        self.collect_docids_sealed(
            cx,
            &checkpoint,
            &mut collector,
            query,
            snapshot,
            prepared_doc_freqs.as_ref(),
            fan_out,
        )?;
        for delta in snapshot.delta_snapshots() {
            checkpoint.admit(QueryWorkKind::Segment, 1)?;
            let score_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_SCORE,
                phase = "score",
                collector = "docset",
                residency = "delta",
                lease_base = delta.lease_base(),
                doc_count = delta.live_document_count(),
                plan = "unscored",
                segments_touched = 1_u64,
                pruning_windows = 0_u64,
                blocks_skipped = 0_u64,
                candidate_docs = 0_u64,
                duration_us = tracing::field::Empty,
            );
            let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
            let _score_entered = score_span.enter();
            let mut scorer = lower_query_unscored(
                cx,
                &checkpoint,
                query,
                1.0,
                QueryLeaf::Delta(delta),
                snapshot,
                prepared_doc_freqs.as_ref(),
                self.schema,
                self.config.glob_expansion_limit,
            )?;
            collector.collect(&mut scorer, delta.as_ref())?;
        }
        let collect_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_COLLECT,
            phase = "collect",
            collector = "docset",
            segment_count,
            segments_touched = segment_count,
            doc_count = snapshot.live_doc_count(),
            result_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _collect_timer = crate::tracing_conventions::StageTimer::new(&collect_span);
        let _collect_entered = collect_span.enter();
        let docids = collector.finish();
        collect_span.record(
            "result_count",
            u64::try_from(docids.len()).unwrap_or(u64::MAX),
        );
        Ok(docids)
    }

    pub(super) fn search_doc_ids(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Arc<[QuillHit]>, QuillIndexError> {
        Ok(self.search_paginated(cx, query, limit, 0, false)?.hits)
    }

    pub(super) fn search_with_snippets(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
    ) -> Result<Vec<QuillSnippetHit>, QuillIndexError> {
        let snapshot = self.published_snapshot.load();
        self.search_with_snippets_on(cx, query, limit, snippet_config, snapshot.as_ref())
    }

    /// Generate snippets against the exact caller-pinned composite snapshot.
    pub(super) fn search_with_snippets_on(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<QuillSnippetHit>, QuillIndexError> {
        check_cancel(cx, "search")?;
        let query_type = classify_query(query);
        if query_type == QueryExplanation::Empty {
            return Ok(Vec::new());
        }
        let mut parsed = self.default_parser()?.parse_lenient(query);
        let _canonicalization = canonicalize_query(&mut parsed.query);
        let rank_work_upper_bound = query_work_upper_bound(
            &parsed.query,
            snapshot,
            self.schema,
            self.config.glob_expansion_limit,
        )?;
        let tail_work_upper_bound = snippet_tail_work_upper_bound(
            &parsed.query,
            snapshot,
            limit,
            self.config.glob_expansion_limit,
        )?;
        let work_upper_bound = rank_work_upper_bound.saturating_add(tail_work_upper_bound);
        let concrete_checkpoint = self.query_checkpoint(
            cx,
            "search",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        let checkpoint_metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        let search = self.search_paginated_on_with_checkpoint(
            cx,
            query,
            limit,
            0,
            false,
            snapshot,
            &checkpoint,
            checkpoint_metering,
        )?;
        #[cfg(feature = "conformance-internals")]
        self.conformance_controller
            .checkpoint(ConformanceCancellationStage::SnippetTailAdmission, cx);
        checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
        let terms = compiled_snippet_terms(
            &checkpoint,
            &parsed.query,
            snapshot,
            self.schema,
            self.config.glob_expansion_limit,
        )?;
        let analyzer = match self.schema.fields.get(usize::from(CONTENT_FIELD)) {
            Some(field) => match field.kind {
                FieldKind::Text { analyzer, .. } => analyzer,
                _ => return Err(invalid_state("content field is not text")),
            },
            None => return Err(invalid_state("schema has no content field")),
        };
        let mut generator = SnippetGenerator::new(analyzer, terms, snippet_config.clone());
        let mut results = Vec::new();
        results
            .try_reserve_exact(search.hits.len())
            .map_err(|_| invalid_state("could not allocate enriched lexical results"))?;
        for (rank, hit) in search.hits.iter().enumerate() {
            // Stored-content materialisation plus snippet analysis is per hit
            // and is the last span of the tail that admitted nothing at all: a
            // query cancelled after ranking still enriched every hit. Admitted
            // BEFORE the stored value is materialised, and covered by the
            // `limit` term of `snippet_tail_work_upper_bound`, so a budget
            // sized for this query cannot be refused here.
            checkpoint.admit(QueryWorkKind::PositionDocument, 1)?;
            let snippet = snapshot
                .materialize_stored_value(CONTENT_FIELD, hit.global_docid)?
                .map(|content| {
                    String::from_utf8(content)
                        .map_err(|_| invalid_state("stored content contains non-UTF-8 bytes"))
                })
                .transpose()?
                .as_deref()
                .and_then(|content| generator.snippet(content));
            results.push(QuillSnippetHit {
                document_id: hit.document_id.clone(),
                score: hit.score,
                rank,
                snippet,
                query_type,
                metadata: snapshot.materialize_metadata(hit.global_docid)?,
            });
        }
        Ok(results)
    }

    pub(super) fn snippets_for_documents(
        &self,
        cx: &Cx,
        query: &str,
        document_ids: &[&str],
        snippet_config: &SnippetConfig,
        max_source_bytes: usize,
    ) -> Result<Vec<Option<String>>, QuillIndexError> {
        let snapshot = self.published_snapshot.load();
        self.snippets_for_documents_on(
            cx,
            query,
            document_ids,
            snippet_config,
            max_source_bytes,
            snapshot.as_ref(),
        )
    }

    /// Snippets for chosen documents with the query terms, global document
    /// frequencies and analyzer [`Self::search_with_snippets_on`] uses, so a
    /// document it ranks gets the same snippet here. A caller that ranks from
    /// identifiers alone then pays for snippets only on what it returns.
    pub(super) fn snippets_for_documents_on(
        &self,
        cx: &Cx,
        query: &str,
        document_ids: &[&str],
        snippet_config: &SnippetConfig,
        max_source_bytes: usize,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<Option<String>>, QuillIndexError> {
        check_cancel(cx, "snippets")?;
        if document_ids.is_empty() || classify_query(query) == QueryExplanation::Empty {
            return Ok(vec![None; document_ids.len()]);
        }
        let mut parsed = self.default_parser()?.parse_lenient(query);
        let _canonicalization = canonicalize_query(&mut parsed.query);
        let work_upper_bound = snippet_tail_work_upper_bound(
            &parsed.query,
            snapshot,
            document_ids.len(),
            self.config.glob_expansion_limit,
        )?;
        let checkpoint: QueryCheckpointHandle<'_> = self.query_checkpoint(
            cx,
            "snippets",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
        let terms = compiled_snippet_terms(
            &checkpoint,
            &parsed.query,
            snapshot,
            self.schema,
            self.config.glob_expansion_limit,
        )?;
        let analyzer = match self.schema.fields.get(usize::from(CONTENT_FIELD)) {
            Some(field) => match field.kind {
                FieldKind::Text { analyzer, .. } => analyzer,
                _ => return Err(invalid_state("content field is not text")),
            },
            None => return Err(invalid_state("schema has no content field")),
        };
        let mut generator = SnippetGenerator::new(analyzer, terms, snippet_config.clone());
        let mut snippets = Vec::new();
        snippets
            .try_reserve_exact(document_ids.len())
            .map_err(|_| invalid_state("could not allocate document snippets"))?;
        for document_id in document_ids {
            // One stored value per document, as the ranked snippet tail admits.
            checkpoint.admit(QueryWorkKind::PositionDocument, 1)?;
            let Some(global_docid) = snapshot.resolve_document_id(document_id)? else {
                snippets.push(None);
                continue;
            };
            let content = snapshot
                .materialize_stored_value(CONTENT_FIELD, global_docid)?
                .map(|content| {
                    String::from_utf8(content)
                        .map_err(|_| invalid_state("stored content contains non-UTF-8 bytes"))
                })
                .transpose()?;
            snippets.push(content.as_deref().and_then(|content| {
                let mut end = content.len().min(max_source_bytes);
                while !content.is_char_boundary(end) {
                    end -= 1;
                }
                generator.snippet_or_prefix(&content[..end])
            }));
        }
        Ok(snippets)
    }

    /// Split one document's score for `query` by clause and field.
    ///
    /// Every term, phrase and glob leaf of the parsed query is scored alone,
    /// once per field and under the boosts above it, by the same evaluator
    /// the search runs, restricted to the one document by a zero-weight
    /// identifier clause. A Boolean score is the sum of its matching
    /// clauses', so the parts add up to the whole.
    pub(super) fn explain_document_on(
        &self,
        cx: &Cx,
        query: &str,
        document_id: &str,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Option<QuillScoreExplanation>, QuillIndexError> {
        check_cancel(cx, "explain")?;
        if classify_query(query) == QueryExplanation::Empty {
            return Ok(None);
        }
        let mut parsed = self.default_parser()?.parse_lenient(query);
        let _canonicalization = canonicalize_query(&mut parsed.query);
        let score_of = |clause: Query| -> Result<Option<f32>, QuillIndexError> {
            let only_this_document = Query::boost(
                Query::set(ID_FIELD, vec![QueryValue::Str(document_id.to_owned())]),
                0.0,
            );
            let pinned = Query::boolean(
                vec![
                    BooleanClause::new(Occur::Must, clause),
                    BooleanClause::new(Occur::Must, only_this_document),
                ],
                None,
            );
            Ok(self
                .search_preparsed_uncached_on(cx, &pinned, 1, 0, false, snapshot)?
                .hits
                .first()
                .map(|hit| hit.score))
        };
        let Some(score) = score_of(parsed.query.clone())? else {
            return Ok(None);
        };
        let field_name = |field_id: u16| {
            self.schema
                .fields
                .get(usize::from(field_id))
                .map_or("unknown", |field| field.name)
        };
        let doc_count = snapshot.bm25_doc_count();
        let mut terms = Vec::new();
        let mut pending = vec![(parsed.query.root_id(), 1.0_f32)];
        while let Some((node, boost)) = pending.pop() {
            match parsed.query.node(node) {
                QueryNode::Boolean { clauses, .. } => pending.extend(
                    clauses
                        .iter()
                        .rev()
                        .filter(|clause| clause.occur != Occur::MustNot)
                        .map(|clause| (clause.query, boost)),
                ),
                QueryNode::Boost { query, factor } => pending.push((*query, boost * factor)),
                QueryNode::Term { fields, text } => {
                    for field in fields {
                        let doc_freq = snapshot.bm25_doc_freq(field.field_id, text.as_bytes())?;
                        let clause = Query::term(vec![*field], text.clone());
                        terms.push(QuillTermScore {
                            text: text.clone(),
                            kind: QuillClauseKind::Term,
                            field: field_name(field.field_id),
                            score: score_of(Query::boost(clause, boost))?.unwrap_or(0.0),
                            doc_freq: Some(doc_freq),
                            idf: (doc_freq <= doc_count)
                                .then(|| crate::contract::idf(doc_freq, doc_count)),
                        });
                    }
                }
                QueryNode::Phrase {
                    fields,
                    terms: words,
                    slop,
                    prefix,
                } => {
                    let text = words
                        .iter()
                        .map(|word| word.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    for field in fields {
                        let clause = Query::phrase(vec![*field], words.clone(), *slop, *prefix);
                        terms.push(QuillTermScore {
                            text: text.clone(),
                            kind: QuillClauseKind::Phrase,
                            field: field_name(field.field_id),
                            score: score_of(Query::boost(clause, boost))?.unwrap_or(0.0),
                            doc_freq: None,
                            idf: None,
                        });
                    }
                }
                QueryNode::Glob { field_ids, pattern } => {
                    for &field_id in field_ids {
                        let clause = Query::glob(vec![field_id], pattern.clone());
                        terms.push(QuillTermScore {
                            text: pattern.clone(),
                            kind: QuillClauseKind::Glob,
                            field: field_name(field_id),
                            score: score_of(Query::boost(clause, boost))?.unwrap_or(0.0),
                            doc_freq: None,
                            idf: None,
                        });
                    }
                }
                QueryNode::Empty
                | QueryNode::AllocationFailure { .. }
                | QueryNode::All
                | QueryNode::Range { .. }
                | QueryNode::Set { .. } => {}
            }
        }
        Ok(Some(QuillScoreExplanation { score, terms }))
    }

    pub(super) fn segment_stats(&self) -> SearchResult<SegmentStats> {
        let snapshot = self
            .checked_published_snapshot()
            .map_err(SearchError::from)?;
        let mut stats = snapshot.keeper_snapshot().segment_stats()?;
        stats.delta_segments = snapshot.delta_count();
        stats.live_docs = usize::try_from(snapshot.live_doc_count()).map_err(|_| {
            SearchError::SubsystemError {
                subsystem: "quill",
                source: "published Quill live-document count does not fit usize".into(),
            }
        })?;
        stats.delta_memory_bytes =
            snapshot
                .delta_snapshots()
                .iter()
                .try_fold(0_u64, |total, delta| {
                    let bytes = u64::try_from(delta.segment().bytes_used()).map_err(|_| {
                        SearchError::SubsystemError {
                            subsystem: "quill",
                            source: "Quill Delta memory footprint does not fit u64".into(),
                        }
                    })?;
                    total
                        .checked_add(bytes)
                        .ok_or_else(|| SearchError::SubsystemError {
                            subsystem: "quill",
                            source: "Quill Delta memory footprint overflowed u64".into(),
                        })
                })?;
        Ok(stats)
    }
}
