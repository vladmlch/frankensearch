#[cfg(feature = "bench-internals")]
use crate::argus::TopDocsCollector;
use crate::config::QuillConfig;
use crate::delta::DeltaSnapshot;
use crate::keeper::{KeeperSnapshot, PublicationReadState};
use crate::query::{DefaultQueryParser, Query, canonicalize_query};
use crate::schema::SchemaDescriptor;
use asupersync::Cx;
use frankensearch_core::{ScoreSource, ScoredResult};
#[cfg(feature = "conformance-internals")]
use sha2::{Digest, Sha256};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex as StdMutex;
#[cfg(test)]
use std::time::{Duration, Instant};

use super::checkpoint::{QueryCheckpoint, QueryCheckpointHandle};
#[cfg(feature = "bench-internals")]
use super::checkpoint::{prepare_snapshot_doc_freqs, query_work_upper_bound};
#[cfg(feature = "conformance-internals")]
use super::conformance::ConformanceCancellationController;
#[cfg(feature = "pruning-conformance")]
use super::conformance::{
    ConformancePruningTraceGuard, ConformancePruningTraceReceipt, ConformancePruningTraceSession,
};
use super::error_results::{
    QuillIndexError, QuillSearchResult, check_cancel, invalid_state,
    publication_state_reconciliation_required,
};
#[cfg(feature = "profile-internals")]
use super::profile::{
    QuillProfileCacheDisposition, QuillProfileOutcome, QuillProfileSession,
    QuillProfiledSearchOutcome,
};
use super::query_cache::raw_query_cache_fingerprint;
use super::query_lowering::record_query_trace_shape;
#[cfg(feature = "bench-internals")]
use super::query_lowering::{query_has_prunable_root_union, validate_query_lowering};
use super::snapshot::{PreparedSealedPublication, QuillSearchSnapshot, SnapshotPublisher};

#[cfg(test)]
#[derive(Clone, Copy, Default)]
pub(super) enum PublicationReadPauseState {
    #[default]
    Idle,
    Armed,
    Reached,
    ReleasedBeforeReached,
    ReleasedAfterReached,
}

/// Test-only rendezvous immediately after pinning an `Arc` and before the
/// second packed-authority load. It makes the double-read retry parent-red.
#[cfg(test)]
#[derive(Default)]
pub(super) struct PublicationReadPause {
    pub(super) state: StdMutex<PublicationReadPauseState>,
    pub(super) wake: std::sync::Condvar,
}

#[cfg(test)]
pub(super) struct PublicationReadPauseControl {
    pub(super) pause: Arc<PublicationReadPause>,
}

#[derive(Clone)]
pub(super) struct QuillReader {
    pub(super) config: QuillConfig,
    pub(super) schema: SchemaDescriptor,
    pub(super) parser: Option<DefaultQueryParser>,
    pub(super) published_snapshot: Arc<SnapshotPublisher>,
    pub(super) publication_read_state: PublicationReadState,
    #[cfg(test)]
    pub(super) publication_read_pause: Arc<PublicationReadPause>,
    #[cfg(feature = "conformance-internals")]
    pub(super) conformance_controller: Arc<ConformanceCancellationController>,
}

#[cfg(test)]
impl PublicationReadPauseControl {
    pub(super) fn wait_until_reached(&self, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        {
            let mut state = self
                .pause
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while !matches!(
                *state,
                PublicationReadPauseState::Reached
                    | PublicationReadPauseState::ReleasedAfterReached
            ) {
                let now = Instant::now();
                if now >= deadline {
                    return Err("reader did not reach the authority double-read pause".to_owned());
                }
                let (next, _) = self
                    .pause
                    .wake
                    .wait_timeout(state, deadline.saturating_duration_since(now))
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state = next;
            }
            drop(state);
        }
        Ok(())
    }

    pub(super) fn release(&self) {
        {
            let mut state = self
                .pause
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *state = match *state {
                PublicationReadPauseState::Reached
                | PublicationReadPauseState::ReleasedAfterReached => {
                    PublicationReadPauseState::ReleasedAfterReached
                }
                PublicationReadPauseState::Idle
                | PublicationReadPauseState::Armed
                | PublicationReadPauseState::ReleasedBeforeReached => {
                    PublicationReadPauseState::ReleasedBeforeReached
                }
            };
        }
        self.pause.wake.notify_all();
    }
}

#[cfg(test)]
impl Drop for PublicationReadPauseControl {
    fn drop(&mut self) {
        self.release();
    }
}

/// Three same-snapshot observations plus the snapshot's physical digest.
///
/// See [`super::QuillSearchIndex::search_paginated_pinned`].
#[cfg(feature = "conformance-internals")]
#[derive(Debug, Clone)]
pub struct PinnedSearchBundle {
    /// The requested page (`limit` at `offset`), count-free.
    pub observed: QuillSearchResult,
    /// The expanded prefix from rank 0 (`expanded_limit` rows), count-free.
    pub expanded: QuillSearchResult,
    /// The count-only invocation (`limit = 0`, exact count on).
    pub count: QuillSearchResult,
    /// Domain-separated SHA-256 over the pinned publication's physical
    /// identity: epoch, Keeper generation, every sealed segment's manifest
    /// identity (id, seal sequence, length, xxh3, docid range, doc count),
    /// and the Delta segment count.
    pub snapshot_sha256: [u8; 32],
    /// Publication epoch the bundle was taken on.
    pub snapshot_epoch: u64,
    /// Keeper generation the bundle was taken on.
    pub keeper_generation: u64,
}

#[cfg(feature = "conformance-internals")]
pub(super) fn pinned_snapshot_sha256(snapshot: &QuillSearchSnapshot) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch/quill/pinned-search-snapshot/v1\0");
    hasher.update(snapshot.snapshot_epoch().to_be_bytes());
    hasher.update(snapshot.keeper_generation().to_be_bytes());
    let keeper = snapshot.keeper_snapshot();
    hasher.update(keeper.doc_count().to_be_bytes());
    let segments = keeper.segments();
    hasher.update(
        u64::try_from(segments.len())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    for segment in segments {
        let manifest = segment.manifest();
        for word in [
            manifest.segment_id,
            manifest.seal_seq,
            manifest.file_len,
            manifest.file_xxh3,
            manifest.docid_lo,
            manifest.docid_hi,
            u64::from(manifest.doc_count),
        ] {
            hasher.update(word.to_be_bytes());
        }
    }
    hasher.update(
        u64::try_from(snapshot.delta_count())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    hasher.finalize().into()
}

#[cfg(feature = "conformance-internals")]
impl QuillReader {
    /// Ranked page, expanded prefix, and exact count on ONE loaded
    /// publication. See [`super::QuillSearchIndex::search_paginated_pinned`].
    pub(super) fn search_paginated_pinned(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        expanded_limit: usize,
    ) -> Result<PinnedSearchBundle, QuillIndexError> {
        let published = self.checked_published_snapshot()?;
        let observed =
            self.search_paginated_on(cx, query, limit, offset, false, published.as_ref())?;
        let expanded =
            self.search_paginated_on(cx, query, expanded_limit, 0, false, published.as_ref())?;
        let count = self.search_paginated_on(cx, query, 0, 0, true, published.as_ref())?;
        Ok(PinnedSearchBundle {
            observed,
            expanded,
            count,
            snapshot_sha256: pinned_snapshot_sha256(published.as_ref()),
            snapshot_epoch: published.snapshot_epoch(),
            keeper_generation: published.keeper_generation(),
        })
    }
}

impl QuillReader {
    /// Return the only `Arc` that was validated against one stable authority
    /// word. Two bounded attempts cover a publication completing between the
    /// first authority load and the pinned `Arc` load without turning a read
    /// into an unbounded spin.
    pub(super) fn checked_published_snapshot(
        &self,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let mut last_generation = self.published_snapshot.load().keeper_generation();
        for attempt in 0..2 {
            let before = self.publication_read_state.load();
            if !before.is_readable() {
                return Err(publication_state_reconciliation_required(
                    before,
                    last_generation,
                ));
            }
            let snapshot = self.published_snapshot.load();
            #[cfg(test)]
            self.pause_after_pinned_snapshot_for_test();
            let after = self.publication_read_state.load();
            last_generation = snapshot.keeper_generation();
            let stable_authority = before == after;
            let generation_matches = last_generation == before.generation();
            if stable_authority && generation_matches {
                return Ok(snapshot);
            }
            if attempt == 1 {
                return Err(publication_state_reconciliation_required(
                    after,
                    last_generation,
                ));
            }
        }
        unreachable!("two bounded publication-read attempts always return")
    }

    #[cfg(test)]
    pub(super) fn arm_publication_read_pause_for_test(&self) -> PublicationReadPauseControl {
        {
            let mut state = self
                .publication_read_pause
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                !matches!(
                    *state,
                    PublicationReadPauseState::Armed | PublicationReadPauseState::Reached
                ),
                "only one publication-read pause may be armed"
            );
            *state = PublicationReadPauseState::Armed;
        }
        PublicationReadPauseControl {
            pause: Arc::clone(&self.publication_read_pause),
        }
    }

    #[cfg(test)]
    fn pause_after_pinned_snapshot_for_test(&self) {
        {
            let mut state = self
                .publication_read_pause
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !matches!(*state, PublicationReadPauseState::Armed) {
                return;
            }
            *state = PublicationReadPauseState::Reached;
            self.publication_read_pause.wake.notify_all();
            while !matches!(*state, PublicationReadPauseState::ReleasedAfterReached) {
                state = self
                    .publication_read_pause
                    .wake
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            drop(state);
        }
    }

    /// Store a fully validated sealed successor and only then make its
    /// generation readable through the shared authority word.
    pub(super) fn install_validated_prepared_sealed(
        &self,
        keeper: Arc<KeeperSnapshot>,
        prepared: PreparedSealedPublication,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        PublicationReadState::validate_generation(keeper.loaded_manifest().manifest.generation)?;
        let installed = self
            .published_snapshot
            .install_validated_prepared_sealed(keeper, prepared);
        self.publication_read_state
            .stabilize(installed.keeper_generation())?;
        Ok(installed)
    }

    /// Publish one complete process-local composition and then expose that
    /// exact installed `Arc` through the shared authority word.
    pub(super) fn publish_complete(
        &self,
        keeper: Arc<KeeperSnapshot>,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        PublicationReadState::validate_generation(keeper.loaded_manifest().manifest.generation)?;
        let installed = self.published_snapshot.publish_complete(keeper, deltas)?;
        self.publication_read_state
            .stabilize(installed.keeper_generation())?;
        Ok(installed)
    }

    pub(super) fn publish_durable_catch_up(
        &self,
        keeper: &Arc<KeeperSnapshot>,
    ) -> Result<Option<Arc<QuillSearchSnapshot>>, QuillIndexError> {
        PublicationReadState::validate_generation(keeper.loaded_manifest().manifest.generation)?;
        let installed = self.published_snapshot.publish_durable_catch_up(keeper)?;
        if let Some(snapshot) = &installed {
            self.publication_read_state
                .stabilize(snapshot.keeper_generation())?;
        }
        Ok(installed)
    }

    #[cfg(not(feature = "conformance-internals"))]
    #[inline]
    #[allow(
        clippy::unused_self,
        reason = "keeps cache call sites cfg-symmetric with the controller-backed variant"
    )]
    fn ranked_query_cache_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "conformance-internals")]
    #[inline]
    fn ranked_query_cache_enabled(&self) -> bool {
        !self.conformance_controller.is_armed()
    }

    #[cfg(not(feature = "conformance-internals"))]
    #[inline]
    #[allow(
        clippy::unused_self,
        reason = "keeps checkpoint call sites cfg-symmetric with the controller-backed variant"
    )]
    pub(super) fn query_checkpoint<'a>(
        &self,
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
    ) -> Arc<QueryCheckpoint<'a>> {
        QueryCheckpoint::new(cx, phase, budget, upper_bound)
    }

    #[cfg(feature = "conformance-internals")]
    #[inline]
    pub(super) fn query_checkpoint<'a>(
        &self,
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
    ) -> Arc<QueryCheckpoint<'a>> {
        QueryCheckpoint::new_with_controller(
            cx,
            phase,
            budget,
            upper_bound,
            Arc::clone(&self.conformance_controller),
        )
    }

    #[cfg(all(feature = "profile-internals", not(feature = "conformance-internals")))]
    #[inline]
    #[allow(
        clippy::unused_self,
        reason = "keeps profile checkpoint call sites cfg-symmetric with the controller-backed variant"
    )]
    pub(super) fn profile_query_checkpoint<'a>(
        &self,
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
        profile: &'a QuillProfileSession,
    ) -> Arc<QueryCheckpoint<'a>> {
        QueryCheckpoint::new_with_profile(cx, phase, budget, upper_bound, profile)
    }

    #[cfg(all(feature = "profile-internals", feature = "conformance-internals"))]
    #[inline]
    pub(super) fn profile_query_checkpoint<'a>(
        &self,
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
        profile: &'a QuillProfileSession,
    ) -> Arc<QueryCheckpoint<'a>> {
        QueryCheckpoint::new_with_controller_and_profile(
            cx,
            phase,
            budget,
            upper_bound,
            Arc::clone(&self.conformance_controller),
            profile,
        )
    }

    pub(super) fn default_parser(&self) -> Result<&DefaultQueryParser, QuillIndexError> {
        self.parser.as_ref().ok_or_else(|| {
            invalid_state("string query APIs are unavailable for this preparsed-only index")
        })
    }

    /// Parse and exhaustively execute one query over the published composite
    /// Keeper-plus-Delta snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, section validation, unsupported-query, or
    /// scorer/collector failures. Scalar accumulator state and installed but
    /// MANIFEST-unreferenced bytes are intentionally absent; frozen Delta
    /// epochs published in the process-local view are included.
    pub(super) fn search_paginated(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let published = self.published_snapshot.load();
        self.search_paginated_on(cx, query, limit, offset, exact_count, published.as_ref())
    }

    #[cfg(feature = "profile-internals")]
    pub(super) fn search_paginated_with_profile(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillProfiledSearchOutcome, QuillIndexError> {
        let published = self.published_snapshot.load();
        let keeper = published.keeper_snapshot();
        let sealed_segments = u64::try_from(keeper.segments().len()).map_err(|_| {
            invalid_state("profile-internals sealed segment count does not fit u64")
        })?;
        let delta_segments = u64::try_from(published.delta_count())
            .map_err(|_| invalid_state("profile-internals Delta segment count does not fit u64"))?;
        let session = QuillProfileSession::new(
            published.snapshot_epoch(),
            published.keeper_generation(),
            sealed_segments,
            delta_segments,
        );
        let result = self.search_paginated_on_inner(
            cx,
            query,
            limit,
            offset,
            exact_count,
            None,
            None,
            #[cfg(feature = "pruning-conformance")]
            None,
            Some(&session),
            published.as_ref(),
        );
        let outcome = match &result {
            Ok(_) => QuillProfileOutcome::Completed,
            Err(QuillIndexError::Cancelled { .. }) => QuillProfileOutcome::Cancelled,
            Err(QuillIndexError::QueryFuelExhausted { .. }) => QuillProfileOutcome::FuelExhausted,
            Err(_) => QuillProfileOutcome::OtherError,
        };
        let receipt = session.complete(outcome)?;
        Ok(match result {
            Ok(result) => QuillProfiledSearchOutcome::Completed { result, receipt },
            Err(error) => QuillProfiledSearchOutcome::Failed { error, receipt },
        })
    }

    #[cfg(feature = "pruning-conformance")]
    pub(super) fn search_paginated_with_conformance_pruning_trace_on(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<(QuillSearchResult, ConformancePruningTraceReceipt), QuillIndexError> {
        let expected_segments = snapshot.keeper_snapshot().segments().len();
        let guard = ConformancePruningTraceGuard::new(
            expected_segments,
            Arc::clone(&self.conformance_controller),
        );
        let result = self.search_paginated_on_inner(
            cx,
            query,
            limit,
            offset,
            exact_count,
            None,
            None,
            Some(guard.session()),
            #[cfg(feature = "profile-internals")]
            None,
            snapshot,
        )?;
        let receipt = guard.complete()?;
        Ok((result, receipt))
    }

    pub(super) fn search_paginated_on(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.search_paginated_on_inner(
            cx,
            query,
            limit,
            offset,
            exact_count,
            None,
            None,
            #[cfg(feature = "pruning-conformance")]
            None,
            #[cfg(feature = "profile-internals")]
            None,
            snapshot,
        )
    }

    pub(super) fn search_paginated_on_with_checkpoint<'a>(
        &self,
        cx: &'a Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        snapshot: &QuillSearchSnapshot,
        checkpoint: &QueryCheckpointHandle<'a>,
        checkpoint_metering: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.search_paginated_on_inner(
            cx,
            query,
            limit,
            offset,
            exact_count,
            Some(checkpoint),
            Some(checkpoint_metering),
            #[cfg(feature = "pruning-conformance")]
            None,
            #[cfg(feature = "profile-internals")]
            None,
            snapshot,
        )
    }

    fn search_paginated_on_inner<'a>(
        &self,
        cx: &'a Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        checkpoint: Option<&QueryCheckpointHandle<'a>>,
        checkpoint_metering: Option<bool>,
        #[cfg(feature = "pruning-conformance")] pruning_trace: Option<
            &ConformancePruningTraceSession,
        >,
        #[cfg(feature = "profile-internals")] profile: Option<&QuillProfileSession>,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            query_form = "raw",
            cache_lookup = tracing::field::Empty,
            query_len = query.len(),
            segment_count,
            doc_count = snapshot.live_doc_count(),
            limit,
            offset,
            exact_count,
            result_count = tracing::field::Empty,
            total_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _query_timer = crate::tracing_conventions::StageTimer::new(&query_span);
        let _query_entered = query_span.enter();
        if let Err(error) = check_cancel(cx, "search") {
            query_span.record("cache_lookup", "not_checked");
            #[cfg(feature = "profile-internals")]
            if let Some(profile) = profile {
                profile.bind_cache(QuillProfileCacheDisposition::NotChecked)?;
                profile.record_cancellation_observation();
            }
            return Err(error);
        }
        let cache_enabled = self.ranked_query_cache_enabled();
        #[cfg(feature = "pruning-conformance")]
        let cache_enabled = cache_enabled && pruning_trace.is_none();
        let cache_fingerprint = cache_enabled.then(|| {
            raw_query_cache_fingerprint(
                snapshot.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
            )
        });
        if let Some(fingerprint) = cache_fingerprint {
            if let Some(result) = self.published_snapshot.ranked_query_cache.get_raw(
                fingerprint,
                snapshot.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
            ) {
                query_span.record("cache_lookup", "hit");
                #[cfg(feature = "profile-internals")]
                if let Some(profile) = profile {
                    profile.bind_cache(QuillProfileCacheDisposition::Hit)?;
                }
                query_span.record(
                    "result_count",
                    u64::try_from(result.hits.len()).unwrap_or(u64::MAX),
                );
                if let Some(total_count) = result.total_count {
                    query_span.record("total_count", total_count);
                }
                return Ok(result);
            }
            query_span.record("cache_lookup", "miss");
            #[cfg(feature = "profile-internals")]
            if let Some(profile) = profile {
                profile.bind_cache(QuillProfileCacheDisposition::Miss)?;
            }
        } else {
            query_span.record("cache_lookup", "disabled");
            #[cfg(feature = "profile-internals")]
            if let Some(profile) = profile {
                profile.bind_cache(QuillProfileCacheDisposition::Disabled)?;
            }
        }
        let parsed = {
            let parse_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_PARSE,
                phase = "parse",
                query_len = query.len(),
                diagnostic_count = tracing::field::Empty,
                result_count = tracing::field::Empty,
                query_root = tracing::field::Empty,
                query_shape_hash = tracing::field::Empty,
                query_nodes = tracing::field::Empty,
                query_depth = tracing::field::Empty,
                term_nodes = tracing::field::Empty,
                phrase_nodes = tracing::field::Empty,
                boolean_nodes = tracing::field::Empty,
                predicate_nodes = tracing::field::Empty,
                boost_nodes = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _parse_timer = crate::tracing_conventions::StageTimer::new(&parse_span);
            let _parse_entered = parse_span.enter();
            let mut parsed = self.default_parser()?.parse_lenient(query);
            let report = canonicalize_query(&mut parsed.query);
            parse_span.record(
                "diagnostic_count",
                u64::try_from(parsed.diagnostics.len()).unwrap_or(u64::MAX),
            );
            parse_span.record("result_count", 1_u64);
            record_query_trace_shape(&parse_span, &parsed.query);
            tracing::debug!(
                target: crate::tracing_conventions::TARGET,
                must_not_duplicates_removed = report.must_not_duplicates_removed,
                filter_duplicates_removed = report.filter_duplicates_removed,
                duplicate_fields_removed = report.duplicate_fields_removed,
                boolean_levels_sorted = report.boolean_levels_sorted,
                glob_fields_canonicalized = report.glob_fields_canonicalized,
                "canonicalized parsed Quill query"
            );
            parsed
        };
        let result = self.execute_ranked_query_inner(
            cx,
            &parsed.query,
            snapshot,
            limit,
            offset,
            exact_count,
            false,
            parsed.diagnostics,
            checkpoint,
            checkpoint_metering,
            #[cfg(feature = "pruning-conformance")]
            pruning_trace,
            #[cfg(feature = "profile-internals")]
            profile,
        )?;
        let result_count = u64::try_from(result.hits.len()).unwrap_or(u64::MAX);
        query_span.record("result_count", result_count);
        if let Some(total_count) = result.total_count {
            query_span.record("total_count", total_count);
        }
        if let Some(fingerprint) = cache_fingerprint {
            self.published_snapshot.ranked_query_cache.insert_raw(
                fingerprint,
                snapshot.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
                &result,
            );
        }
        Ok(result)
    }

    pub(super) fn scored_results(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        hydrate_metadata: bool,
    ) -> Result<Vec<ScoredResult>, QuillIndexError> {
        let snapshot = self.published_snapshot.load();
        self.scored_results_on(cx, query, limit, hydrate_metadata, snapshot.as_ref())
    }

    /// Score on the currently published snapshot and return that exact
    /// snapshot `Arc` alongside the results, so callers can pin hydration to
    /// the generation that produced the scores (bd-8nqz.1).
    pub(super) fn scored_results_pinned(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        hydrate_metadata: bool,
    ) -> Result<(Vec<ScoredResult>, Arc<QuillSearchSnapshot>), QuillIndexError> {
        let published = self.published_snapshot.load();
        let results =
            self.scored_results_on(cx, query, limit, hydrate_metadata, published.as_ref())?;
        Ok((results, published))
    }

    /// Score against the exact caller-pinned composite snapshot.
    pub(super) fn scored_results_on(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        hydrate_metadata: bool,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<ScoredResult>, QuillIndexError> {
        if limit == 0 {
            check_cancel(cx, "search")?;
            return Ok(Vec::new());
        }
        let search = self.search_paginated_on(cx, query, limit, 0, false, snapshot)?;
        let mut results = Vec::new();
        results
            .try_reserve_exact(search.hits.len())
            .map_err(|_| invalid_state("could not allocate lexical results"))?;
        for hit in search.hits.iter() {
            let metadata = hydrate_metadata
                .then(|| snapshot.materialize_metadata(hit.global_docid))
                .transpose()?
                .flatten();
            results.push(ScoredResult {
                doc_id: hit.document_id.clone().into(),
                score: hit.score,
                source: ScoreSource::Lexical,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: Some(hit.score),
                rerank_score: None,
                explanation: None,
                metadata,
            });
        }
        Ok(results)
    }

    /// Collect the complete deterministic set of matching global document IDs.
    ///
    /// This is the scoreless collector lane: it lowers the same canonical
    /// default-parser tree as [`Self::search_paginated`], traverses every
    /// Keeper and published process-local Delta leaf without ranking, and
    /// returns sorted unique Q1 IDs. Scalar accumulator documents remain
    /// invisible.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, cursor, or allocation
    /// failures.
    pub(super) fn collect_docids(&self, cx: &Cx, query: &str) -> Result<Vec<u32>, QuillIndexError> {
        let published = self.published_snapshot.load();
        self.collect_docids_on(cx, query, published.as_ref())
    }

    /// Collect against the exact caller-pinned composite snapshot.
    pub(super) fn collect_docids_on(
        &self,
        cx: &Cx,
        query: &str,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<u32>, QuillIndexError> {
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            collector = "docset",
            query_len = query.len(),
            segment_count,
            doc_count = snapshot.live_doc_count(),
            result_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _query_timer = crate::tracing_conventions::StageTimer::new(&query_span);
        let _query_entered = query_span.enter();
        check_cancel(cx, "collect_docids")?;
        let parsed = {
            let parse_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_PARSE,
                phase = "parse",
                query_len = query.len(),
                diagnostic_count = tracing::field::Empty,
                result_count = tracing::field::Empty,
                query_root = tracing::field::Empty,
                query_shape_hash = tracing::field::Empty,
                query_nodes = tracing::field::Empty,
                query_depth = tracing::field::Empty,
                term_nodes = tracing::field::Empty,
                phrase_nodes = tracing::field::Empty,
                boolean_nodes = tracing::field::Empty,
                predicate_nodes = tracing::field::Empty,
                boost_nodes = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _parse_timer = crate::tracing_conventions::StageTimer::new(&parse_span);
            let _parse_entered = parse_span.enter();
            let mut parsed = self.default_parser()?.parse_lenient(query);
            let _canonicalization = canonicalize_query(&mut parsed.query);
            parse_span.record(
                "diagnostic_count",
                u64::try_from(parsed.diagnostics.len()).unwrap_or(u64::MAX),
            );
            parse_span.record("result_count", 1_u64);
            record_query_trace_shape(&parse_span, &parsed.query);
            parsed
        };
        let docids = self.execute_docid_query(cx, &parsed.query, snapshot)?;
        let result_count = u64::try_from(docids.len()).unwrap_or(u64::MAX);
        query_span.record("result_count", result_count);
        Ok(docids)
    }

    /// Execute one already-built query tree through the ranked mixed-state path.
    ///
    /// This remains feature-gated because the shipping surface accepts the
    /// language string and owns parser diagnostics. The conformance gauntlet
    /// needs the lower AST boundary to cover query classes that the default
    /// shipping parser cannot construct for every compiled schema.
    ///
    /// A schema with no default field reaches the AST boundary through
    /// [`Self::search_preparsed_uncached_on`] instead, which is un-gated and
    /// deliberately does not touch the ranked-query cache.
    ///
    /// # Errors
    ///
    /// Returns the same typed lowering, collection, and cancellation failures
    /// as [`Self::search_paginated`].
    #[cfg(feature = "bench-internals")]
    pub(super) fn search_preparsed_paginated(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let published = self.published_snapshot.load();
        self.search_preparsed_paginated_on(
            cx,
            query,
            limit,
            offset,
            exact_count,
            published.as_ref(),
        )
    }

    /// Execute one already-built query tree without consulting the ranked
    /// query cache.
    ///
    /// This is the shipping AST entry point, for a compiled schema that
    /// carries its own query language and therefore has no string parser to
    /// go through — the CASS profile parses via
    /// [`crate::query::CassQueryParser`]. It bypasses the ranked-query cache
    /// on purpose: that cache is keyed by the raw query string, which a
    /// preparsed caller does not have, and its gated key space is measured
    /// evidence this path must not perturb.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, lowering, scoring, or collection failures.
    pub(super) fn search_preparsed_uncached_on(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        check_cancel(cx, "search_preparsed")?;
        self.execute_ranked_query_inner(
            cx,
            query,
            snapshot,
            limit,
            offset,
            exact_count,
            false,
            Vec::new(),
            None,
            None,
            #[cfg(feature = "pruning-conformance")]
            None,
            #[cfg(feature = "profile-internals")]
            None,
        )
    }

    #[cfg(feature = "bench-internals")]
    pub(super) fn search_preparsed_paginated_on(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            query_form = "preparsed",
            cache_lookup = tracing::field::Empty,
            segment_count,
            doc_count = snapshot.live_doc_count(),
            limit,
            offset,
            exact_count,
            result_count = tracing::field::Empty,
            total_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _query_timer = crate::tracing_conventions::StageTimer::new(&query_span);
        let _query_entered = query_span.enter();
        if let Err(error) = check_cancel(cx, "search_preparsed") {
            query_span.record("cache_lookup", "not_checked");
            return Err(error);
        }
        let cache_enabled = self.ranked_query_cache_enabled();
        if cache_enabled {
            if let Some(result) = self.published_snapshot.ranked_query_cache.get_preparsed(
                snapshot.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
            ) {
                query_span.record("cache_lookup", "hit");
                query_span.record(
                    "result_count",
                    u64::try_from(result.hits.len()).unwrap_or(u64::MAX),
                );
                if let Some(total_count) = result.total_count {
                    query_span.record("total_count", total_count);
                }
                return Ok(result);
            }
            query_span.record("cache_lookup", "miss");
        } else {
            query_span.record("cache_lookup", "disabled");
        }
        let mut canonical = query.clone();
        let _canonicalization = canonicalize_query(&mut canonical);
        let result = self.execute_ranked_query(
            cx,
            &canonical,
            snapshot,
            limit,
            offset,
            exact_count,
            Vec::new(),
        )?;
        query_span.record(
            "result_count",
            u64::try_from(result.hits.len()).unwrap_or(u64::MAX),
        );
        if let Some(total_count) = result.total_count {
            query_span.record("total_count", total_count);
        }
        if cache_enabled {
            self.published_snapshot.ranked_query_cache.insert_preparsed(
                snapshot.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
                &result,
            );
        }
        Ok(result)
    }

    #[cfg(feature = "bench-internals")]
    pub(super) fn collect_preparsed_docids_on(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<u32>, QuillIndexError> {
        check_cancel(cx, "collect_preparsed_docids")?;
        let mut canonical = query.clone();
        let _canonicalization = canonicalize_query(&mut canonical);
        self.execute_docid_query(cx, &canonical, snapshot)
    }

    #[cfg(feature = "bench-internals")]
    pub(super) fn bench_search_sealed_forced_on(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        fan_out: bool,
        rank_pruning: Option<bool>,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<(u32, u32)>, QuillIndexError> {
        let mut parsed = self.default_parser()?.parse_lenient(query);
        let _canonicalization = canonicalize_query(&mut parsed.query);
        validate_query_lowering(&parsed.query, 1.0, self.schema)?;
        let topdocs_root = limit != 0;
        let rank_pruning = topdocs_root
            && rank_pruning.unwrap_or_else(|| query_has_prunable_root_union(&parsed.query, 1.0));
        let mut collector = TopDocsCollector::new(limit, 0)?;
        let work_upper_bound = query_work_upper_bound(
            &parsed.query,
            snapshot,
            self.schema,
            self.config.glob_expansion_limit,
        )?;
        let concrete_checkpoint = self.query_checkpoint(
            cx,
            "search",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        let prepared_doc_freqs = prepare_snapshot_doc_freqs(snapshot);
        self.collect_sealed_segments(
            cx,
            &checkpoint,
            &mut collector,
            &parsed.query,
            snapshot,
            prepared_doc_freqs.as_ref(),
            rank_pruning,
            topdocs_root,
            #[cfg(feature = "pruning-conformance")]
            None,
            #[cfg(feature = "profile-internals")]
            None,
            fan_out && !metering,
        )?;
        let collected = collector.finish()?;
        Ok(collected
            .hits
            .iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect())
    }
}
