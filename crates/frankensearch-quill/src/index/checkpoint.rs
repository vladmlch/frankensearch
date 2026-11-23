#[cfg(feature = "pruning-conformance")]
use crate::argus::ConformanceQueryWorkContext;
use crate::argus::{ArgusError, QueryWorkCheckpoint, QueryWorkKind};
use crate::keeper::Manifest;
use crate::query::{Occur, Query, QueryNode, QueryValue};
use crate::schema::{FieldKind, SchemaDescriptor};
use crate::segment::{EncodedSegment, SectionKind};
use asupersync::Cx;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "pruning-conformance")]
use std::thread::ThreadId;

#[cfg(feature = "conformance-internals")]
use super::conformance::ConformanceCancellationController;
#[cfg(all(
    feature = "conformance-internals",
    not(feature = "pruning-conformance")
))]
use super::conformance::ConformanceCancellationStage;
#[cfg(feature = "pruning-conformance")]
use super::conformance::ConformanceQueryInterruptionLocation;
use super::error_results::{CONTENT_FIELD, QuillIndexError, invalid_state};
#[cfg(feature = "profile-internals")]
use super::profile::QuillProfileSession;
use super::query_cursors::{
    SnapshotDocFreqDeltaAdmission, checkpointed_snapshot_doc_freq, open_dictionary,
    required_section,
};
use super::query_predicates::query_field_kind;
use super::snapshot::{PreparedSealedPublication, QuillSearchSnapshot};

#[derive(Default)]
pub(super) struct QueryFuelState {
    pub(super) consumed: AtomicU64,
    pub(super) segments_touched: AtomicU64,
    pub(super) dictionary_blocks: AtomicU64,
    pub(super) posting_blocks: AtomicU64,
    pub(super) position_docs: AtomicU64,
}

pub(super) struct QueryCheckpoint<'a> {
    pub(super) cx: &'a Cx,
    pub(super) phase: &'static str,
    pub(super) budget: u64,
    pub(super) metering: bool,
    pub(super) state: QueryFuelState,
    #[cfg(feature = "profile-internals")]
    pub(super) profile: Option<&'a QuillProfileSession>,
    #[cfg(feature = "conformance-internals")]
    pub(super) conformance_controller: Arc<ConformanceCancellationController>,
    #[cfg(feature = "pruning-conformance")]
    pub(super) conformance_query_work_contexts:
        StdMutex<HashMap<ThreadId, ConformanceQueryWorkContext>>,
}

impl<'a> QueryCheckpoint<'a> {
    #[cfg(any(not(feature = "conformance-internals"), test))]
    pub(super) fn new(cx: &'a Cx, phase: &'static str, budget: u64, upper_bound: u64) -> Arc<Self> {
        Arc::new(Self {
            cx,
            phase,
            budget,
            metering: upper_bound > budget,
            state: QueryFuelState::default(),
            #[cfg(feature = "profile-internals")]
            profile: None,
            #[cfg(feature = "conformance-internals")]
            conformance_controller: Arc::default(),
            #[cfg(feature = "pruning-conformance")]
            conformance_query_work_contexts: StdMutex::new(HashMap::new()),
        })
    }

    #[cfg(feature = "conformance-internals")]
    pub(super) fn new_with_controller(
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
        conformance_controller: Arc<ConformanceCancellationController>,
    ) -> Arc<Self> {
        #[cfg(feature = "pruning-conformance")]
        conformance_controller.reset_query_interruption_location();
        Arc::new(Self {
            cx,
            phase,
            budget,
            metering: upper_bound > budget,
            state: QueryFuelState::default(),
            #[cfg(feature = "profile-internals")]
            profile: None,
            conformance_controller,
            #[cfg(feature = "pruning-conformance")]
            conformance_query_work_contexts: StdMutex::new(HashMap::new()),
        })
    }

    #[cfg(all(feature = "profile-internals", not(feature = "conformance-internals")))]
    pub(super) fn new_with_profile(
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
        profile: &'a QuillProfileSession,
    ) -> Arc<Self> {
        Arc::new(Self {
            cx,
            phase,
            budget,
            metering: upper_bound > budget,
            state: QueryFuelState::default(),
            profile: Some(profile),
        })
    }

    #[cfg(all(feature = "profile-internals", feature = "conformance-internals"))]
    pub(super) fn new_with_controller_and_profile(
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
        conformance_controller: Arc<ConformanceCancellationController>,
        profile: &'a QuillProfileSession,
    ) -> Arc<Self> {
        Arc::new(Self {
            cx,
            phase,
            budget,
            metering: upper_bound > budget,
            state: QueryFuelState::default(),
            profile: Some(profile),
            conformance_controller,
            #[cfg(feature = "pruning-conformance")]
            conformance_query_work_contexts: StdMutex::new(HashMap::new()),
        })
    }

    pub(super) const fn metering(&self) -> bool {
        self.metering
    }

    #[cfg(feature = "profile-internals")]
    pub(super) fn record_snapshot_doc_freq(&self) {
        if let Some(profile) = self.profile {
            profile.record_snapshot_doc_freq(1);
        }
    }

    #[cfg(feature = "profile-internals")]
    pub(super) fn record_global_doc_freq(&self) {
        if let Some(profile) = self.profile {
            profile.record_global_doc_freq(1);
        }
    }

    #[cfg(feature = "profile-internals")]
    pub(super) fn record_term_dictionary_view(&self) {
        if let Some(profile) = self.profile {
            profile.record_term_dictionary_view(1);
        }
    }

    pub(super) fn exhausted(&self, consumed: u64) -> ArgusError {
        let segments_touched = self.state.segments_touched.load(Ordering::Acquire);
        let dictionary_blocks = self.state.dictionary_blocks.load(Ordering::Acquire);
        let posting_blocks = self.state.posting_blocks.load(Ordering::Acquire);
        let position_docs = self.state.position_docs.load(Ordering::Acquire);
        tracing::warn!(
            target: crate::tracing_conventions::TARGET,
            event = "quill.argus.query_fuel_exhausted",
            phase = self.phase,
            budget = self.budget,
            consumed,
            segments_touched,
            dictionary_blocks,
            posting_blocks,
            position_docs,
            "Quill query exhausted its deterministic coarse work budget"
        );
        ArgusError::QueryFuelExhausted {
            budget: self.budget,
            consumed,
            segments_touched,
            dictionary_blocks,
            posting_blocks,
            position_docs,
        }
    }

    #[cfg(feature = "pruning-conformance")]
    fn conformance_query_work_context(&self) -> Option<ConformanceQueryWorkContext> {
        self.conformance_query_work_contexts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&std::thread::current().id())
            .copied()
    }

    #[cfg(feature = "pruning-conformance")]
    fn record_query_interruption(
        &self,
        context: Option<ConformanceQueryWorkContext>,
        kind: QueryWorkKind,
        units: u64,
        consumed_before: u64,
    ) {
        if let Some(context) = context {
            self.conformance_controller
                .record_query_interruption_location(ConformanceQueryInterruptionLocation::new(
                    context,
                    kind,
                    units,
                    consumed_before,
                ));
        }
    }
}

impl QueryWorkCheckpoint for QueryCheckpoint<'_> {
    fn admit(&self, kind: QueryWorkKind, units: u64) -> Result<(), ArgusError> {
        #[cfg(feature = "pruning-conformance")]
        {
            // `consumed_before` is an atomic load and stays here: it has to be
            // sampled before the checkpoint, and under Rayon a later read would
            // resample a value another segment worker has already moved.
            //
            // The context read must NOT. It takes a global `StdMutex` and hashes
            // a `ThreadId`, on every admission — that is once per posting move,
            // and twice on a move that enters a block — and the value is
            // discarded unless the checkpoint fires, which happens at most once
            // per query. Reading it inside the fire branch is value-identical:
            // the map is keyed by the current thread and
            // `set_conformance_query_work_context` is the only writer, reached
            // only from the union scorer on this same thread, never from
            // `checkpoint_query_collection` (which touches controller atomics
            // and `cx` cancellation alone). The fuel-exhaustion arm below
            // already reads it exactly this way.
            let consumed_before = self.state.consumed.load(Ordering::Acquire);
            if self
                .conformance_controller
                .checkpoint_query_collection(self.cx)
            {
                self.record_query_interruption(
                    self.conformance_query_work_context(),
                    kind,
                    units,
                    consumed_before,
                );
            }
        }
        #[cfg(all(
            feature = "conformance-internals",
            not(feature = "pruning-conformance")
        ))]
        self.conformance_controller
            .checkpoint(ConformanceCancellationStage::QueryCollection, self.cx);
        if units == 0 {
            // A zero-unit admit still has to SURFACE cancellation. The
            // conformance checkpoints above run before this guard, so they can
            // request cancellation and then have the caller told `Ok(())`,
            // leaving a cancelled query to keep walking its plan. No work is
            // metered here, so only the observation and the typed error apply.
            // bd-tb4c4: per-posting cancellation poll uses the Cx cancellation flag.
            if self.cx.is_cancelled() {
                #[cfg(feature = "profile-internals")]
                if let Some(profile) = self.profile {
                    profile.record_cancellation_observation();
                }
                return Err(ArgusError::QueryCancelled { phase: self.phase });
            }
            return Ok(());
        }
        #[cfg(feature = "profile-internals")]
        if let Some(profile) = self.profile {
            profile.record_work_requested(kind, units);
        }
        // bd-tb4c4: per-posting cancellation poll uses the Cx cancellation flag.
        if self.cx.is_cancelled() {
            #[cfg(feature = "profile-internals")]
            if let Some(profile) = self.profile {
                profile.record_cancellation_observation();
            }
            return Err(ArgusError::QueryCancelled { phase: self.phase });
        }
        if !self.metering {
            #[cfg(feature = "profile-internals")]
            if let Some(profile) = self.profile {
                profile.record_fuel_units(units);
                profile.record_work_admitted(kind, units);
            }
            return Ok(());
        }
        let admitted =
            self.state
                .consumed
                .try_update(Ordering::AcqRel, Ordering::Acquire, |consumed| {
                    consumed
                        .checked_add(units)
                        .filter(|next| *next <= self.budget)
                });
        let previous = match admitted {
            Ok(previous) => previous,
            Err(consumed) => {
                #[cfg(feature = "pruning-conformance")]
                self.record_query_interruption(
                    self.conformance_query_work_context(),
                    kind,
                    units,
                    consumed,
                );
                #[cfg(feature = "profile-internals")]
                if let Some(profile) = self.profile {
                    profile.record_work_refused(kind, units);
                }
                return Err(self.exhausted(consumed));
            }
        };
        debug_assert!(previous.saturating_add(units) <= self.budget);
        let counter = match kind {
            QueryWorkKind::Segment => &self.state.segments_touched,
            QueryWorkKind::DictionaryBlock => &self.state.dictionary_blocks,
            QueryWorkKind::PostingBlock => &self.state.posting_blocks,
            QueryWorkKind::PositionDocument => &self.state.position_docs,
        };
        let _ = counter.fetch_add(units, Ordering::Relaxed);
        #[cfg(feature = "profile-internals")]
        if let Some(profile) = self.profile {
            profile.record_fuel_units(units);
            profile.record_work_admitted(kind, units);
        }
        Ok(())
    }

    #[cfg(feature = "pruning-conformance")]
    fn set_conformance_query_work_context(&self, context: Option<ConformanceQueryWorkContext>) {
        let thread_id = std::thread::current().id();
        let mut contexts = self
            .conformance_query_work_contexts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(context) = context {
            contexts.insert(thread_id, context);
        } else {
            contexts.remove(&thread_id);
        }
    }
}

#[cfg(feature = "profile-internals")]
pub(super) type QueryCheckpointHandle<'a> = Arc<QueryCheckpoint<'a>>;
#[cfg(not(feature = "profile-internals"))]
pub(super) type QueryCheckpointHandle<'a> = Arc<dyn QueryWorkCheckpoint + 'a>;

pub(super) fn clone_query_checkpoint_for_argus<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
) -> Arc<dyn QueryWorkCheckpoint + 'a> {
    checkpoint.clone()
}

#[derive(Default)]
struct QueryWorkShape {
    pub(super) posting_streams: u64,
    pub(super) dictionary_scans: u64,
    pub(super) string_range_expansions: u64,
    pub(super) phrase_streams: u64,
    pub(super) phrase_fields: u64,
}

/// Pinned Tantivy 0.27.0 expands the final phrase-prefix term to at most 50
/// terms independently inside each immutable leaf. This is deliberately not
/// the configurable ordinary-glob ceiling: applying one snapshot-global glob
/// limit here drops valid matches from later leaves and diverges from the
/// oracle's public query parser.
pub(super) const TANTIVY_PHRASE_PREFIX_MAX_EXPANSIONS: usize = 50;

impl QueryWorkShape {
    pub(super) fn visit(
        &mut self,
        query: &Query,
        schema: SchemaDescriptor,
        glob_expansion_limit: usize,
    ) -> Result<(), QuillIndexError> {
        let mut pending = Vec::new();
        if pending.try_reserve_exact(query.node_count()).is_err() {
            return Err(ArgusError::Allocation {
                resource: "query work-shape traversal",
                count: query.node_count(),
            }
            .into());
        }
        pending.push(query.root_id());
        while let Some(node_id) = pending.pop() {
            match query.node(node_id) {
                QueryNode::Empty | QueryNode::All => {}
                QueryNode::AllocationFailure { requested } => {
                    return Err(ArgusError::Allocation {
                        resource: "query arena",
                        count: *requested,
                    }
                    .into());
                }
                QueryNode::Term { fields, .. } => {
                    let fields = u64::try_from(fields.len()).unwrap_or(u64::MAX);
                    self.posting_streams = self.posting_streams.saturating_add(fields);
                }
                QueryNode::Phrase {
                    fields,
                    terms,
                    prefix,
                    ..
                } => {
                    let fields = u64::try_from(fields.len()).unwrap_or(u64::MAX);
                    let terms = u64::try_from(terms.len()).unwrap_or(u64::MAX);
                    let expansions = if *prefix {
                        self.dictionary_scans = self.dictionary_scans.saturating_add(fields);
                        u64::try_from(TANTIVY_PHRASE_PREFIX_MAX_EXPANSIONS).unwrap_or(u64::MAX)
                    } else {
                        1
                    };
                    let phrase_fields = fields.saturating_mul(expansions);
                    let streams = phrase_fields.saturating_mul(terms);
                    self.posting_streams = self.posting_streams.saturating_add(streams);
                    self.phrase_streams = self.phrase_streams.saturating_add(streams);
                    self.phrase_fields = self.phrase_fields.saturating_add(phrase_fields);
                }
                QueryNode::Boolean { clauses, .. } => {
                    pending.extend(clauses.iter().rev().map(|clause| clause.query));
                }
                QueryNode::Boost { query: child, .. } => pending.push(*child),
                QueryNode::Range { field_id, .. } => {
                    if matches!(
                        query_field_kind(schema, *field_id)?,
                        FieldKind::Keyword | FieldKind::Text { .. }
                    ) {
                        self.dictionary_scans = self.dictionary_scans.saturating_add(1);
                        self.string_range_expansions =
                            self.string_range_expansions.saturating_add(1);
                    }
                }
                QueryNode::Glob { field_ids, .. } => {
                    let fields = u64::try_from(field_ids.len()).unwrap_or(u64::MAX);
                    self.dictionary_scans = self.dictionary_scans.saturating_add(fields);
                    self.posting_streams =
                        self.posting_streams.saturating_add(fields.saturating_mul(
                            u64::try_from(glob_expansion_limit).unwrap_or(u64::MAX),
                        ));
                }
                QueryNode::Set { values, .. } => {
                    let strings = values
                        .iter()
                        .filter(|value| matches!(value, QueryValue::Str(_)))
                        .count();
                    let strings = u64::try_from(strings).unwrap_or(u64::MAX);
                    self.posting_streams = self.posting_streams.saturating_add(strings);
                }
            }
        }
        Ok(())
    }
}

/// Snapshot-global BM25 statistics reused by lowered query terms.
///
/// Ranked and identifier-only search lower the same query independently for
/// every sealed segment. Without this query-local table, each lowering opens
/// every sealed dictionary again to recompute an identical global document
/// frequency. Entries are populated lazily at the exact-term lowering boundary
/// so creating the table adds no unmetered query walk. The table lives for one
/// pinned snapshot invocation and cannot cross either a query or publication
/// boundary.
pub(super) struct PreparedSnapshotDocFreqs {
    pub(super) entries: StdMutex<HashMap<u16, HashMap<Vec<u8>, u64>>>,
    #[cfg(test)]
    pub(super) reservation_overrides: SnapshotDocFreqReservationOverrides,
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct SnapshotDocFreqReservationOverrides {
    pub(super) term: Option<usize>,
    pub(super) outer_map: Option<usize>,
    pub(super) new_field_map: Option<usize>,
    pub(super) existing_field_map: Option<usize>,
}

impl PreparedSnapshotDocFreqs {
    pub(super) fn new() -> Self {
        Self {
            entries: StdMutex::new(HashMap::new()),
            #[cfg(test)]
            reservation_overrides: SnapshotDocFreqReservationOverrides::default(),
        }
    }

    pub(super) fn get_or_compute(
        &self,
        checkpoint: &QueryCheckpointHandle<'_>,
        snapshot: &QuillSearchSnapshot,
        field_ord: u16,
        term: &[u8],
    ) -> Result<u64, QuillIndexError> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = entries
            .get(&field_ord)
            .and_then(|field_entries| field_entries.get(term))
            .copied()
        {
            drop(entries);
            // A hit opens no dictionary, so it admits no dictionary block.
            // The per-segment replay that used to stand here left the charge
            // at `streams x segments^2` even though the scan happened once,
            // which made the table a latency optimisation that could not stop
            // a few hundred segments from exhausting the budget on statistics
            // nothing recomputed (#41). The zero-unit admission preserves the
            // cancellation poll those replayed admissions used to provide.
            checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
            return Ok(cached);
        }

        // The linear work ceiling requires this cache. Surface cancellation
        // before allocating, and refuse allocation failure before scanning;
        // an uncached fallback could exceed the ceiling with metering disabled.
        checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
        let mut owned_term = Vec::new();
        let term_capacity = term.len();
        #[cfg(test)]
        let term_capacity = self.reservation_overrides.term.unwrap_or(term_capacity);
        if owned_term.try_reserve_exact(term_capacity).is_err() {
            drop(entries);
            return Err(ArgusError::Allocation {
                resource: "snapshot document-frequency term",
                count: term.len(),
            }
            .into());
        }
        owned_term.extend_from_slice(term);

        checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
        let mut new_field_entries = None;
        let outer_capacity = 1;
        #[cfg(test)]
        let outer_capacity = self
            .reservation_overrides
            .outer_map
            .unwrap_or(outer_capacity);
        let reserve_failed = if let Some(field_entries) = entries.get_mut(&field_ord) {
            let field_capacity = 1;
            #[cfg(test)]
            let field_capacity = self
                .reservation_overrides
                .existing_field_map
                .unwrap_or(field_capacity);
            field_entries.try_reserve(field_capacity).is_err()
        } else if entries.try_reserve(outer_capacity).is_err() {
            true
        } else {
            checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
            let mut field_entries = HashMap::new();
            let field_capacity = 1;
            #[cfg(test)]
            let field_capacity = self
                .reservation_overrides
                .new_field_map
                .unwrap_or(field_capacity);
            if field_entries.try_reserve(field_capacity).is_err() {
                true
            } else {
                new_field_entries = Some(field_entries);
                false
            }
        };
        if reserve_failed {
            drop(entries);
            return Err(ArgusError::Allocation {
                resource: "snapshot document-frequency cache",
                count: 1,
            }
            .into());
        }

        let computed = checkpointed_snapshot_doc_freq(
            checkpoint,
            snapshot,
            field_ord,
            term,
            SnapshotDocFreqDeltaAdmission::Ranking,
        )?;
        if let Some(mut field_entries) = new_field_entries {
            field_entries.insert(owned_term, computed);
            entries.insert(field_ord, field_entries);
        } else if let Some(field_entries) = entries.get_mut(&field_ord) {
            field_entries.insert(owned_term, computed);
        }
        drop(entries);
        Ok(computed)
    }
}

/// A snapshot with fewer sealed segments than this has nothing to reuse: a
/// single lowering already performs the one and only snapshot scan. The
/// query-work ceiling has to charge whichever regime actually runs, so it
/// reads the threshold from here rather than restating it.
const SNAPSHOT_DOC_FREQ_TABLE_MIN_SEGMENTS: usize = 2;

pub(super) fn prepare_snapshot_doc_freqs(
    snapshot: &QuillSearchSnapshot,
) -> Option<PreparedSnapshotDocFreqs> {
    if snapshot.keeper_snapshot().segments().len() < SNAPSHOT_DOC_FREQ_TABLE_MIN_SEGMENTS {
        return None;
    }
    Some(PreparedSnapshotDocFreqs::new())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LifecycleTrigger {
    ArenaBudget,
    LeaseBoundary,
    VisibilityLag,
    ExplicitFlush,
    BulkCadence,
    BulkFinish,
    TierFanout,
}

impl LifecycleTrigger {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::ArenaBudget => "arena_budget",
            Self::LeaseBoundary => "lease_boundary",
            Self::VisibilityLag => "visibility_lag",
            Self::ExplicitFlush => "explicit_flush",
            Self::BulkCadence => "bulk_cadence",
            Self::BulkFinish => "bulk_finish",
            Self::TierFanout => "tier_fanout",
        }
    }
}

pub(super) struct PendingDeltaSeal {
    /// Retained encoded segment shared with the in-flight publication task.
    ///
    /// The outer [`Arc`] shares the whole segment (header plus section
    /// table) with `KeeperWriter::publish_encoded_segment_retryable`, which
    /// needs `'static` ownership on the blocking lane. The segment's byte
    /// payload is additionally backing-shared, so the memory-backend
    /// republication copies stay O(1) in the payload.
    pub(super) encoded: Option<Arc<EncodedSegment>>,
    pub(super) segment_installed: bool,
    pub(super) manifest: Manifest,
    pub(super) prepared: PreparedSealedPublication,
    pub(super) next_seal_seq: u64,
    pub(super) successor_watermark: u64,
}

/// Minimum total sealed live-document count before ranked queries fan
/// per-segment scoring across rayon (bd-quill-e4-argus-3ycz.9). Follows the
/// house `PARALLEL_THRESHOLD` philosophy from
/// `frankensearch-index/src/search.rs`: picked by measurement, starting at
/// the same 10k floor. Below the gate, per-task scorer and collector setup
/// costs more than the fan-out returns.
pub(super) const SEGMENT_FANOUT_THRESHOLD: u64 = 10_000;
/// A fragmented snapshot has enough independent scorer setup and posting work
/// to amortize rayon even below the document-count gate. The 2-segment
/// below-gate control remains serial; watch-mode replacement batches cross
/// this threshold as immutable leaves accumulate.
pub(super) const SEGMENT_COUNT_FANOUT_THRESHOLD: usize = 8;

/// Decide whether sealed-segment scoring fans across rayon. Single-segment
/// snapshots never fan out (there is nothing to overlap). Larger corpora use
/// the document-count gate, while sufficiently fragmented smaller snapshots
/// use the segment-count gate.
pub(super) const fn sealed_segment_fanout(segment_count: usize, total_sealed_docs: u64) -> bool {
    segment_count >= 2
        && (total_sealed_docs >= SEGMENT_FANOUT_THRESHOLD
            || segment_count >= SEGMENT_COUNT_FANOUT_THRESHOLD)
}

#[derive(Default)]
struct SnippetTailWorkShape {
    wildcard_glob_scans: u64,
    exact_glob_scans: u64,
    term_count_ceiling: u64,
}

impl SnippetTailWorkShape {
    fn add_term_count(&mut self, count: usize) {
        self.term_count_ceiling = self
            .term_count_ceiling
            .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
    }

    fn add_glob(&mut self, pattern: &[u8], expansion_limit: usize) {
        if pattern.contains(&b'*') {
            self.add_wildcard_glob(expansion_limit);
        } else {
            self.exact_glob_scans = self.exact_glob_scans.saturating_add(1);
            self.add_term_count(expansion_limit);
        }
    }

    fn add_wildcard_glob(&mut self, expansion_limit: usize) {
        self.wildcard_glob_scans = self.wildcard_glob_scans.saturating_add(1);
        self.add_term_count(expansion_limit);
    }

    pub(super) fn visit(&mut self, query: &Query, expansion_limit: usize) {
        let mut pending = Vec::new();
        if pending.try_reserve_exact(query.node_count()).is_err() {
            self.wildcard_glob_scans = u64::MAX;
            self.exact_glob_scans = u64::MAX;
            self.term_count_ceiling = u64::MAX;
            return;
        }
        pending.push(query.root_id());
        while let Some(node_id) = pending.pop() {
            match query.node(node_id) {
                QueryNode::Term { fields, .. }
                    if fields.iter().any(|field| field.field_id == CONTENT_FIELD) =>
                {
                    self.add_term_count(1);
                }
                QueryNode::Phrase {
                    fields,
                    terms,
                    prefix,
                    ..
                } if fields.iter().any(|field| field.field_id == CONTENT_FIELD) => {
                    if *prefix && !terms.is_empty() {
                        self.add_term_count(terms.len().saturating_sub(1));
                        self.add_wildcard_glob(expansion_limit);
                    } else {
                        self.add_term_count(terms.len());
                    }
                }
                QueryNode::Boolean { clauses, .. } => {
                    pending.extend(
                        clauses
                            .iter()
                            .rev()
                            .filter(|clause| clause.occur != Occur::MustNot)
                            .map(|clause| clause.query),
                    );
                }
                QueryNode::Set { field_id, values } if *field_id == CONTENT_FIELD => {
                    self.add_term_count(
                        values
                            .iter()
                            .filter(|value| matches!(value, QueryValue::Str(_)))
                            .count(),
                    );
                }
                QueryNode::Glob { field_ids, pattern } if field_ids.contains(&CONTENT_FIELD) => {
                    self.add_glob(pattern.as_bytes(), expansion_limit);
                }
                QueryNode::Boost { query: child, .. } => pending.push(*child),
                QueryNode::Empty
                | QueryNode::All
                | QueryNode::AllocationFailure { .. }
                | QueryNode::Term { .. }
                | QueryNode::Phrase { .. }
                | QueryNode::Range { .. }
                | QueryNode::Set { .. }
                | QueryNode::Glob { .. } => {}
            }
        }
    }
}

/// Dictionary-block ceiling for one Delta generation's ordered term view.
///
/// `DeltaSnapshot::ordered_terms` materialises AND radix-sorts every term id
/// on its first view before the first term can be examined, so this ceiling is
/// what the tail must admit BEFORE that cache can be built. The bound and the
/// admission both derive from this one function on purpose: metering is armed
/// by `upper_bound > budget`, so a unit the bound does not cover could refuse
/// a query whose real work fits its budget.
pub(super) fn delta_ordered_view_blocks(term_count: usize) -> u64 {
    u64::try_from(term_count).unwrap_or(u64::MAX).div_ceil(16)
}

pub(super) fn snippet_tail_work_upper_bound(
    query: &Query,
    snapshot: &QuillSearchSnapshot,
    limit: usize,
    expansion_limit: usize,
) -> Result<u64, QuillIndexError> {
    let mut shape = SnippetTailWorkShape::default();
    shape.visit(query, expansion_limit);

    let keeper = snapshot.keeper_snapshot();
    let keeper_segment_count = u64::try_from(keeper.segments().len()).unwrap_or(u64::MAX);
    let delta_count = u64::try_from(snapshot.delta_count()).unwrap_or(u64::MAX);
    let mut wildcard_dictionary_blocks = 0_u64;
    let mut delta_dictionary_blocks = 0_u64;
    for segment in keeper.segments() {
        let bytes = required_section(segment, SectionKind::TERMDICT)?;
        let count_bytes = bytes
            .get(..4)
            .ok_or_else(|| invalid_state("TERMDICT is shorter than its block-count header"))?;
        let block_count = u32::from_le_bytes([
            count_bytes[0],
            count_bytes[1],
            count_bytes[2],
            count_bytes[3],
        ]);
        wildcard_dictionary_blocks =
            wildcard_dictionary_blocks.saturating_add(u64::from(block_count));
    }
    for delta in snapshot.delta_snapshots() {
        delta_dictionary_blocks = delta_dictionary_blocks
            .saturating_add(delta_ordered_view_blocks(delta.segment().term_count()));
    }

    // `snapshot_glob_terms` visits one indexed block per wildcard cursor
    // block. Exact globs use one indexed lookup per Keeper segment, while
    // both forms walk Delta's sorted terms in 16-term blocks. This derives the
    // bound from immutable dictionary headers instead of scanning the tail
    // before its checkpoint can admit the real work.
    let wildcard_glob_work = shape
        .wildcard_glob_scans
        .saturating_mul(wildcard_dictionary_blocks.saturating_add(delta_dictionary_blocks));
    let exact_glob_work = shape
        .exact_glob_scans
        .saturating_mul(keeper_segment_count.saturating_add(delta_dictionary_blocks));
    // Per compiled term the tail does one admitted Keeper dictionary lookup
    // per sealed segment AND one admitted probe per live Delta generation.
    // Charging the Keeper side alone left `term_count_ceiling x delta_count`
    // probes outside every budget, and made this whole bound ZERO for a
    // Delta-only snapshot -- which disarms metering (`upper_bound > budget`)
    // and let the entire tail run unmetered at any budget, including zero.
    // A frozen `live_doc_freq` read is O(1), but the PRODUCT is not, and an
    // unadmitted product is exactly what a fuel budget exists to refuse.
    let document_frequency_work = shape
        .term_count_ceiling
        .saturating_mul(keeper_segment_count.saturating_add(delta_count));
    // Per-hit stored-content materialisation plus snippet analysis is the
    // last unadmitted span of the same tail. Ranking returns at most `limit`
    // hits, so the ceiling is exact.
    let enrichment_work = u64::try_from(limit).unwrap_or(u64::MAX);

    Ok(wildcard_glob_work
        .saturating_add(exact_glob_work)
        .saturating_add(document_frequency_work)
        .saturating_add(enrichment_work))
}

pub(super) fn query_work_upper_bound(
    query: &Query,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    glob_expansion_limit: usize,
) -> Result<u64, QuillIndexError> {
    let mut shape = QueryWorkShape::default();
    shape.visit(query, schema, glob_expansion_limit)?;

    let keeper = snapshot.keeper_snapshot();
    let segment_count = keeper
        .segments()
        .len()
        .saturating_add(snapshot.delta_count());
    let segment_count = u64::try_from(segment_count).unwrap_or(u64::MAX);
    let keeper_count = u64::try_from(keeper.segments().len()).unwrap_or(u64::MAX);

    let mut physical_docs = 0_u64;
    let mut posting_blocks_per_stream = 0_u64;
    let mut dictionary_blocks = 0_u64;
    let mut string_range_term_ceiling = 0_u64;
    for segment in keeper.segments() {
        let docs = u64::from(segment.doc_count());
        physical_docs = physical_docs.saturating_add(docs);
        posting_blocks_per_stream = posting_blocks_per_stream.saturating_add(docs.div_ceil(128));
        let bytes = required_section(segment, SectionKind::TERMDICT)?;
        let count_bytes = bytes
            .get(..4)
            .ok_or_else(|| invalid_state("TERMDICT is shorter than its block-count header"))?;
        let count = u32::from_le_bytes([
            count_bytes[0],
            count_bytes[1],
            count_bytes[2],
            count_bytes[3],
        ]);
        dictionary_blocks = dictionary_blocks.saturating_add(u64::from(count));
        if shape.string_range_expansions != 0 {
            let terms = u64::from(open_dictionary(segment, schema)?.term_count());
            string_range_term_ceiling = string_range_term_ceiling.saturating_add(terms);
        }
    }
    for delta in snapshot.delta_snapshots() {
        // A Delta walk is charged over the physical rows it reads, tombstones
        // included, so the ceiling has to be drawn from the same population.
        // Bounding it by the live count let a tombstone-heavy generation
        // exhaust a budget that was sized for a fraction of the work.
        let docs = u64::try_from(delta.segment().physical_document_count()).unwrap_or(u64::MAX);
        physical_docs = physical_docs.saturating_add(docs);
        posting_blocks_per_stream = posting_blocks_per_stream.saturating_add(docs.div_ceil(128));
        let terms = u64::try_from(delta.segment().term_count()).unwrap_or(u64::MAX);
        dictionary_blocks = dictionary_blocks.saturating_add(terms.div_ceil(16));
        if shape.string_range_expansions != 0 {
            string_range_term_ceiling = string_range_term_ceiling.saturating_add(terms);
        }
    }

    // Unlike glob expansion, string ranges deliberately have no truncation
    // limit: a matching term becomes a real scorer stream. Their ceiling must
    // therefore come from the finite immutable snapshot, including terms that
    // the range's field or endpoints will later exclude. This over-count is
    // intentional; it keeps fuel admission conservative without changing a
    // range's result set.
    let posting_streams = shape.posting_streams.saturating_add(
        shape
            .string_range_expansions
            .saturating_mul(string_range_term_ceiling),
    );

    // A rank-pruning fork can traverse a sealed posting stream independently
    // of the scorer-owned cursor. Two full traversals are therefore a safe
    // ceiling for every syntactic stream.
    let posting_blocks = posting_streams
        .saturating_mul(posting_blocks_per_stream)
        .saturating_mul(2);
    // Snapshot-level expansions run while each segment scorer is lowered.
    // Glob expansion became leaf-local (`leaf_glob_terms`: one dictionary
    // walk per leaf, S walks total), but the scan ceiling keeps the
    // historical snapshot-wide product on purpose: it only decides whether
    // metering arms (`upper_bound > budget`), and over-covering arms metering
    // sooner without ever refusing work that fits the budget.
    let scanned_dictionary_blocks = shape
        .dictionary_scans
        .saturating_mul(dictionary_blocks)
        .saturating_mul(segment_count);
    // Every posting stream is opened through `lower_leaf_term`, including
    // terms materialized by range and glob expansion, so use the complete
    // stream ceiling rather than only the syntactic exact-term count. Each
    // opening costs snapshot statistics plus one probe of the leaf-local
    // dictionary, and only a sealed leaf performs that local probe.
    //
    // The statistics are the part that used to dominate. A snapshot scan
    // reads every sealed dictionary, and without the query-local table each
    // of the `segment_count` lowerings repeated it, so the charge grew as
    // `streams x segments^2` and a few hundred segments exhausted any budget
    // (#41). `PreparedSnapshotDocFreqs` computes it once per term, so above
    // the table threshold the ceiling is one scan per stream; below it the
    // repeat is real and still has to be charged.
    let snapshot_statistic_probes =
        if keeper.segments().len() >= SNAPSHOT_DOC_FREQ_TABLE_MIN_SEGMENTS {
            keeper_count
        } else {
            segment_count.saturating_mul(keeper_count)
        };
    let indexed_dictionary_blocks =
        posting_streams.saturating_mul(snapshot_statistic_probes.saturating_add(keeper_count));
    // Existing phrase lowering materializes positioned rows before candidate
    // verification. Bound both the decode and verification passes.
    let position_docs = shape
        .phrase_streams
        .saturating_add(shape.phrase_fields)
        .saturating_mul(physical_docs);

    Ok(segment_count
        .saturating_add(posting_blocks)
        .saturating_add(scanned_dictionary_blocks)
        .saturating_add(indexed_dictionary_blocks)
        .saturating_add(position_docs))
}
