use crate::grimoire::MAX_TERM_BYTES;
use crate::keeper::ManifestSegment;
use crate::quiver::NumericValue;
use crate::schema::{Analyzer, FieldKind, SchemaDescriptor};
use crate::scribe::{
    AccumulatorError, ArenaSpan, ColumnarAccumulator, DEFAULT_ARENA_CHUNK_BYTES, DocIdSpan,
    FIELD_PREFIX_BYTES, FrankensearchTokenizer, TERM_BUCKET_BYTES_ESTIMATE,
};
use crate::segment::EncodedSegment;
use asupersync::Cx;
use frankensearch_core::IndexableDocument;
use std::collections::BTreeMap;
#[cfg(feature = "conformance-internals")]
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
#[cfg(test)]
use std::time::{Duration, Instant};

#[cfg(feature = "conformance-internals")]
use super::conformance::{ConformanceCancellationController, ConformanceCancellationStage};
use super::error_results::{
    CONTENT_FIELD, ID_FIELD, MAX_GLOBAL_DOCID_EXCLUSIVE, METADATA_FIELD, QuillIndexError,
    TITLE_FIELD, check_cancel, invalid_state,
};
use super::ingest_doc::{
    IngestBatch, canonical_metadata, manifest_segment, write_canonical_metadata,
};
use super::reader::QuillReader;

pub(super) const PARALLEL_INGEST_MIN_DOCS_PER_SHARD: usize = 64;
pub(super) const PARALLEL_INGEST_PLANNER_VERSION: u8 = 1;
pub(super) const PARALLEL_ARENA_BUDGET_DIVISOR: usize = 2;
pub(super) const MIN_ARENA_CHUNK_BYTES: usize = 4096;
pub(super) const INTERNAL_PARALLEL_INGEST_SHARDS: usize = 4;
pub(super) fn ingest_parallelism_at_construction() -> usize {
    if rayon::current_thread_index().is_some() {
        rayon::current_num_threads()
    } else {
        std::thread::available_parallelism().map_or(1, usize::from)
    }
}

#[derive(Debug)]
pub(super) struct PendingIdentity {
    pub(super) doc_ord: u32,
    pub(super) document_id: String,
    /// Domain-separated xxh3-64 witness of this document's canonical content.
    ///
    /// Computed field-wise by `canonical_document_content_hash` at accumulation
    /// time; no canonical-JSON preimage buffer is ever materialized for it.
    /// This `u64` is the only per-document identity state retained until the
    /// seal, where `derive_segment_id` folds it into the batch digest — so a
    /// batch costs 8 bytes per document here, not a second copy of every
    /// document's bytes.
    pub(super) content_hash: u64,
}

pub(super) struct StagedFlush {
    pub(super) shard: usize,
    pub(super) encoded: EncodedSegment,
    pub(super) manifest_segment: ManifestSegment,
    pub(super) pending_field_stats: BTreeMap<u16, (u64, u32)>,
    pub(super) next_seal_seq: u64,
}

pub(super) struct BuiltShardFlush {
    pub(super) shard: usize,
    pub(super) encoded: EncodedSegment,
    pub(super) manifest_segment: ManifestSegment,
    pub(super) next_seal_seq: u64,
}

pub(super) struct ShardFlushPlan<'a> {
    pub(super) shard: usize,
    pub(super) state: &'a ScribeShardState,
    pub(super) segment_id: u64,
    pub(super) lease_docid_base: u64,
    pub(super) created_unix_s: i64,
    pub(super) seal_seq: u64,
}

pub(super) struct ScribeShardState {
    pub(super) accumulator: ColumnarAccumulator,
    pub(super) identities: Vec<PendingIdentity>,
    pub(super) current_lease_base: Option<u64>,
    /// Reused canonicalization buffer. It lives on the shard rather than in
    /// `accumulate_shard_run` so the fan-out's parallel region does not hand
    /// the allocator a fresh buffer per document on every worker thread.
    pub(super) scratch_metadata: Vec<u8>,
}

pub(super) struct ParallelShardWork {
    pub(super) shard: usize,
    pub(super) assignment: ParallelShardAssignment,
    pub(super) state: ScribeShardState,
    /// Canonical metadata moved from the adaptive budget proof into the exact
    /// shard that consumes it. Entries are dropped as documents accumulate,
    /// so cached bytes do not remain resident beside their copied columns.
    pub(super) prepared_metadata: Option<Vec<Option<Vec<u8>>>>,
}

/// Monotonic identity for one shared-nothing ingest batch, so worker ids in a
/// receipt are attributable to a specific run rather than to "some batch".
static PARALLEL_INGEST_RUN_SEQ: AtomicU64 = AtomicU64::new(0);

/// How a shared-nothing batch ended.
///
/// The default is [`Aborted`](ParallelIngestTerminal::Aborted), NOT success:
/// the terminal state is only promoted by an explicit `finish`, so every
/// unwind, `?` early return, and cancelled exit reports honestly without the
/// exit path having to remember to say so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ParallelIngestTerminal {
    /// Default. The whole ingest operation did not reach a successful end: a
    /// worker failed, a later step errored, or the future was dropped.
    #[default]
    Aborted,
    /// A shared-nothing fan-out published a receipt AND the whole operation
    /// then completed. Never set merely because a route returned `Ok(None)`.
    FanoutCompleted,
    /// The shared-nothing routes declined and the whole operation completed on
    /// the legacy `index_batch_fanout` path. That path is ALSO Rayon-parallel,
    /// so it must not be published as serial; its shards are witnessed like
    /// any other.
    FanoutFallbackCompleted,
    /// The routes declined and the whole operation completed on the genuinely
    /// SERIAL path. Any shard counts on this batch belong to a shared-nothing
    /// attempt whose work was discarded.
    SerialCompleted,
    /// Nothing to do: the batch was empty. A success, with no work witnessed.
    NoWork,
    Cancelled,
}

impl ParallelIngestTerminal {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Aborted => "aborted",
            Self::FanoutCompleted => "fanout_completed",
            Self::FanoutFallbackCompleted => "fanout_fallback_completed",
            Self::SerialCompleted => "serial_completed",
            Self::NoWork => "no_work",
            Self::Cancelled => "cancelled",
        }
    }

    pub(super) const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::FanoutCompleted,
            2 => Self::SerialCompleted,
            4 => Self::Cancelled,
            5 => Self::FanoutFallbackCompleted,
            6 => Self::NoWork,
            _ => Self::Aborted,
        }
    }

    pub(super) const fn code(self) -> u8 {
        match self {
            Self::Aborted => 0,
            Self::FanoutCompleted => 1,
            Self::SerialCompleted => 2,
            Self::Cancelled => 4,
            Self::FanoutFallbackCompleted => 5,
            Self::NoWork => 6,
        }
    }

    /// Whether the WHOLE ingest operation reached a successful end.
    ///
    /// True for an empty batch and for a serial-only run: it says the
    /// operation finished, NOT that anything was fanned out.
    pub(super) const fn is_success(self) -> bool {
        matches!(
            self,
            Self::FanoutCompleted
                | Self::FanoutFallbackCompleted
                | Self::SerialCompleted
                | Self::NoWork
        )
    }

    /// Classify a failed operation from its TYPED outcome.
    ///
    /// Deliberately not `cx.is_cancel_requested()`: that flag is read at drop
    /// time, so a cancellation racing an unrelated error or panic would
    /// mislabel the real cause. The error the operation actually returned is
    /// the only non-racy evidence of why it ended.
    pub(super) fn from_error(error: &QuillIndexError) -> Self {
        match error {
            QuillIndexError::Cancelled { .. } => Self::Cancelled,
            _ => Self::Aborted,
        }
    }
}

/// Live, batch-scoped observation of one shared-nothing ingest run.
///
/// Workers write into this through [`ParallelShardGuard`]; the driver reads it
/// through a [`ParallelObservationRecorder`] whose `Drop` publishes it. Because
/// both sides are RAII, a worker error, a panic, or a cancellation that returns
/// long before any receipt is built still leaves a complete, published witness
/// — which is exactly the case the earlier success-path-only counter missed.
///
/// Cost is per SHARD, never per document — but it is NOT free, and an earlier
/// revision of this comment wrongly claimed "two atomic RMWs, non-perturbing".
/// A successful shard performs FIVE atomic read-modify-writes (`started`,
/// `in_flight` up, `peak` `fetch_max`, `completed`, `in_flight` down) plus one
/// acquisition of a mutex CONTENDED by every other shard in the batch. The
/// work is per shard and never per token. Its cost has NOT been measured, so
/// no characterisation of it — cheap, negligible, non-perturbing — is made
/// here or anywhere else.
#[derive(Debug)]
pub(super) struct ParallelBatchObservation {
    pub(super) run_seq: u64,
    pub(super) pool_threads: usize,
    pub(super) started_shards: AtomicUsize,
    pub(super) completed_shards: AtomicUsize,
    pub(super) in_flight: AtomicUsize,
    pub(super) peak_in_flight: AtomicUsize,
    pub(super) unidentified_shards: AtomicUsize,
    pub(super) worker_ids: StdMutex<Vec<usize>>,
    /// Set whenever the worker-id lock could not be taken. Without it a
    /// poisoned lock would publish zero distinct slots, which is
    /// indistinguishable from an honest "no identifiable workers".
    pub(super) identity_degraded: AtomicBool,
    /// The shared-nothing routes declined this batch. Recorded as a FACT, not
    /// as a terminal: work still lies ahead when it is set, so a fallback
    /// panic or a dropped future must still publish `Aborted` rather than a
    /// stale "declined" that reads like an ending.
    pub(super) shared_nothing_declined: AtomicBool,
    pub(super) terminal: AtomicU8,
    /// Test-only rendezvous: when set, each shard waits inside `enter_shard`
    /// until `rendezvous_width` shards are in flight (or a deadline passes),
    /// so a production-path test can force real overlap instead of hoping for
    /// it. Shipping builds do not compile this field.
    #[cfg(test)]
    pub(super) rendezvous_width: AtomicUsize,
    #[cfg(test)]
    pub(super) rendezvous_arrived: AtomicUsize,
    /// Test-only proof that the adaptive budget pass prepared and the actual
    /// workers consumed one canonical metadata image per document.
    #[cfg(test)]
    pub(super) prepared_metadata_documents: AtomicUsize,
    #[cfg(test)]
    pub(super) reused_metadata_documents: AtomicUsize,
}

impl Default for ParallelBatchObservation {
    fn default() -> Self {
        Self {
            run_seq: PARALLEL_INGEST_RUN_SEQ.fetch_add(1, Ordering::Relaxed),
            pool_threads: rayon::current_num_threads(),
            started_shards: AtomicUsize::new(0),
            completed_shards: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            peak_in_flight: AtomicUsize::new(0),
            unidentified_shards: AtomicUsize::new(0),
            worker_ids: StdMutex::new(Vec::new()),
            identity_degraded: AtomicBool::new(false),
            shared_nothing_declined: AtomicBool::new(false),
            terminal: AtomicU8::new(ParallelIngestTerminal::Aborted.code()),
            #[cfg(test)]
            rendezvous_width: AtomicUsize::new(0),
            #[cfg(test)]
            rendezvous_arrived: AtomicUsize::new(0),
            #[cfg(test)]
            prepared_metadata_documents: AtomicUsize::new(0),
            #[cfg(test)]
            reused_metadata_documents: AtomicUsize::new(0),
        }
    }
}

impl ParallelBatchObservation {
    /// Mark one shard started and in flight until the returned guard drops.
    ///
    /// The worker identity is read here, inside the worker, because that is the
    /// only place it exists.
    pub(super) fn enter_shard(&self) -> ParallelShardGuard<'_> {
        self.started_shards.fetch_add(1, Ordering::AcqRel);
        let in_flight = self
            .in_flight
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        self.peak_in_flight.fetch_max(in_flight, Ordering::AcqRel);
        match rayon::current_thread_index() {
            Some(worker) => match self.worker_ids.lock() {
                Ok(mut ids) => ids.push(worker),
                // Fail closed: the identity is lost, so say so rather than
                // letting the count silently under-report.
                Err(_) => self.identity_degraded.store(true, Ordering::Release),
            },
            None => {
                self.unidentified_shards.fetch_add(1, Ordering::AcqRel);
            }
        }
        #[cfg(test)]
        self.rendezvous();
        ParallelShardGuard { observation: self }
    }

    /// Test-only: block until `rendezvous_width` shards are simultaneously
    /// inside `enter_shard`, or until a deadline passes. The deadline means a
    /// pool that serializes cannot deadlock the suite — it fails the overlap
    /// assertion loudly instead.
    #[cfg(test)]
    pub(super) fn rendezvous(&self) {
        let width = self.rendezvous_width.load(Ordering::Acquire);
        if width == 0 {
            return;
        }
        let arrived = self
            .rendezvous_arrived
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        if arrived >= width {
            return;
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        while self.rendezvous_arrived.load(Ordering::Acquire) < width && Instant::now() < deadline {
            std::thread::yield_now();
        }
    }

    /// Test-only: arm the rendezvous for a fan-out of `width` shards.
    ///
    /// Called by the routes from the PLAN, so overlap is forced for any test
    /// that drives a real fan-out — including one that goes through the public
    /// shipping driver and therefore cannot reach the observation itself. No
    /// process-global state is involved, so concurrent tests cannot interfere
    /// with each other.
    #[cfg(test)]
    pub(super) fn arm_rendezvous(&self, width: usize) {
        self.rendezvous_width.store(width, Ordering::Release);
    }

    #[cfg(test)]
    pub(super) fn record_prepared_metadata(&self, documents: usize) {
        self.prepared_metadata_documents
            .fetch_add(documents, Ordering::AcqRel);
    }

    #[cfg(test)]
    pub(super) fn record_reused_metadata(&self) {
        self.reused_metadata_documents
            .fetch_add(1, Ordering::AcqRel);
    }

    /// Set the terminal state. A SUCCESS terminal is set only after the whole
    /// ingest operation has finished, never because the parallel attempt
    /// returned `Ok`.
    pub(super) fn set_terminal(&self, terminal: ParallelIngestTerminal) {
        self.terminal.store(terminal.code(), Ordering::Release);
    }

    pub(super) fn terminal(&self) -> ParallelIngestTerminal {
        ParallelIngestTerminal::from_code(self.terminal.load(Ordering::Acquire))
    }

    pub(super) fn snapshot(&self) -> ParallelWorkerWitness {
        let mut worker_ids = self.worker_ids.lock().map_or_else(
            |_| {
                self.identity_degraded.store(true, Ordering::Release);
                Vec::new()
            },
            |ids| ids.clone(),
        );
        worker_ids.sort_unstable();
        worker_ids.dedup();
        ParallelWorkerWitness {
            run_seq: self.run_seq,
            pool_threads: self.pool_threads,
            started_shards: self.started_shards.load(Ordering::Acquire),
            completed_shards: self.completed_shards.load(Ordering::Acquire),
            distinct_worker_slots: worker_ids.len(),
            unidentified_shards: self.unidentified_shards.load(Ordering::Acquire),
            peak_shards_in_flight: self.peak_in_flight.load(Ordering::Acquire),
            identity_degraded: self.identity_degraded.load(Ordering::Acquire),
            shared_nothing_declined: self.shared_nothing_declined.load(Ordering::Acquire),
            #[cfg(test)]
            prepared_metadata_documents: self.prepared_metadata_documents.load(Ordering::Acquire),
            #[cfg(test)]
            reused_metadata_documents: self.reused_metadata_documents.load(Ordering::Acquire),
        }
    }

    /// Sorted, deduplicated, POOL-LOCAL worker indices for this run.
    ///
    /// These are rayon worker indices, meaningful only inside the pool that
    /// ran this batch. They are not global thread identities.
    pub(super) fn sorted_worker_ids(&self) -> Vec<usize> {
        let mut ids = self.worker_ids.lock().map_or_else(
            |_| {
                self.identity_degraded.store(true, Ordering::Release);
                Vec::new()
            },
            |ids| ids.clone(),
        );
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

/// Per-shard RAII: decrements in-flight on EVERY exit path — normal return,
/// cancelled early return, and unwind — and counts a completion only when the
/// shard actually finished.
pub(super) struct ParallelShardGuard<'observation> {
    observation: &'observation ParallelBatchObservation,
}

impl ParallelShardGuard<'_> {
    pub(super) fn completed(&self) {
        self.observation
            .completed_shards
            .fetch_add(1, Ordering::AcqRel);
    }
}

impl Drop for ParallelShardGuard<'_> {
    fn drop(&mut self) {
        self.observation.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Batch-scoped RAII that publishes the observation onto the ingest span on
/// EVERY exit, so failure, panic, and cancellation are as well-evidenced as
/// success.
pub(super) struct ParallelObservationRecorder<'observation> {
    pub(super) span: tracing::Span,
    pub(super) observation: &'observation ParallelBatchObservation,
}

impl Drop for ParallelObservationRecorder<'_> {
    fn drop(&mut self) {
        // The terminal is read, never inferred: the driver classifies it from
        // the typed operation outcome before this runs.
        let terminal = self.observation.terminal();
        let operation_succeeded = terminal.is_success();
        let witness = self.observation.snapshot();
        let worker_ids = self
            .observation
            .sorted_worker_ids()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        self.span.record("parallel_run_seq", witness.run_seq);
        self.span.record(
            "parallel_pool_threads",
            u64::try_from(witness.pool_threads).unwrap_or(u64::MAX),
        );
        self.span.record(
            "parallel_started_shards",
            u64::try_from(witness.started_shards).unwrap_or(u64::MAX),
        );
        self.span.record(
            "parallel_completed_shards",
            u64::try_from(witness.completed_shards).unwrap_or(u64::MAX),
        );
        self.span.record(
            "parallel_distinct_worker_slots",
            u64::try_from(witness.distinct_worker_slots).unwrap_or(u64::MAX),
        );
        self.span.record(
            "parallel_unidentified_shards",
            u64::try_from(witness.unidentified_shards).unwrap_or(u64::MAX),
        );
        self.span.record(
            "parallel_peak_shards_in_flight",
            u64::try_from(witness.peak_shards_in_flight).unwrap_or(u64::MAX),
        );
        self.span
            .record("parallel_pool_local_worker_ids", worker_ids.as_str());
        self.span
            .record("parallel_identity_degraded", witness.identity_degraded);
        self.span.record(
            "parallel_shared_nothing_declined",
            witness.shared_nothing_declined,
        );
        self.span.record("parallel_terminal", terminal.as_str());
        // Deliberately not `parallel_*`: this is the WHOLE ingest operation's
        // outcome and is true for an empty batch and for a serial-only run,
        // neither of which fanned out at all. Read `parallel_route`,
        // `parallel_terminal` and the shard counts for what the fan-out did.
        self.span
            .record("ingest_operation_succeeded", operation_succeeded);
    }
}

/// Immutable snapshot of a [`ParallelBatchObservation`].
///
/// Every count is over SHARD TASKS, not workers, except
/// `distinct_worker_slots`. Read the names literally:
///
/// - `started_shards` / `completed_shards` — shard tasks entered and finished.
///   A gap between them is the drain witness: shards that started and did not
///   finish because the batch failed, panicked, or was cancelled.
/// - `distinct_worker_slots` — how many different pool worker indices executed
///   at least one shard: worker-thread DIVERSITY. Shards may run sequentially
///   on several workers, so `N` here does NOT mean `N` shards were ever in
///   flight together. It is an upper bound on peak concurrency among
///   identified work, never a lower bound.
/// - `unidentified_shards` — shards rayon ran off-pool. Excluded from
///   `distinct_worker_slots`, which is why that field cannot bound concurrency
///   from below.
/// - `peak_shards_in_flight` — the ONLY concurrency evidence here: the exact
///   high-water mark of shard accumulations running at the same time.
/// - `run_seq` — per-batch run identity (process-monotonic).
/// - `pool_threads` — the WIDTH of the pool that ran the batch. It is not a
///   pool identity: rayon exposes no stable pool id, so none is claimed here.
///   Worker indices are POOL-LOCAL and are only comparable within one
///   `run_seq`; two batches on different pools may reuse index 0 for
///   different threads.
/// - `identity_degraded` — the worker-id lock failed at least once, so
///   `distinct_worker_slots` and the id list are INCOMPLETE. Fail-closed: a
///   lost identity reads as unknown, never as "no workers".
///
/// The terminal state is deliberately NOT here. A receipt is built inside a
/// route, before the driver knows the whole-operation outcome, so any terminal
/// captured at that point would be a stale default. Terminal lives on
/// [`ParallelBatchObservation`] and is published only by the recorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct ParallelWorkerWitness {
    pub(super) run_seq: u64,
    pub(super) pool_threads: usize,
    pub(super) started_shards: usize,
    pub(super) completed_shards: usize,
    pub(super) distinct_worker_slots: usize,
    pub(super) unidentified_shards: usize,
    pub(super) peak_shards_in_flight: usize,
    pub(super) identity_degraded: bool,
    pub(super) shared_nothing_declined: bool,
    #[cfg(test)]
    pub(super) prepared_metadata_documents: usize,
    #[cfg(test)]
    pub(super) reused_metadata_documents: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ParallelShardAssignment {
    pub(super) document_start: usize,
    pub(super) document_end: usize,
    pub(super) span: DocIdSpan,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ParallelIngestReceipt {
    pub(super) planner_version: u8,
    pub(super) route: ParallelIngestRoute,
    pub(super) configured_width: usize,
    pub(super) verified_pool_capacity: usize,
    pub(super) eligible_shards: usize,
    pub(super) active_shards: usize,
    pub(super) logical_budget_bytes: usize,
    pub(super) initial_logical_bytes: usize,
    pub(super) batch_logical_upper_bound: usize,
    pub(super) projected_logical_upper_bound: usize,
    pub(super) arena_chunk_bytes: usize,
    pub(super) arena_bytes_used_high_water: usize,
    pub(super) arena_bytes_reserved_high_water: usize,
    /// Observed shard-completion and worker-diversity counts for the batch,
    /// plus the exact peak overlap. Binding these to the receipt keeps the
    /// planner's *requested* width separable from what was *observed*, and
    /// keeps worker diversity separable from actual concurrency.
    #[cfg(test)]
    pub(super) worker_witness: ParallelWorkerWitness,
}

#[derive(Debug, Clone)]
pub(super) struct ParallelBudgetAdmission {
    pub(super) logical_budget_bytes: usize,
    pub(super) initial_logical_bytes: usize,
    pub(super) initial_reserved_bytes: usize,
    pub(super) batch_logical_upper_bound: usize,
    pub(super) projected_logical_upper_bound: usize,
    pub(super) arena_chunk_bytes: usize,
    /// Canonical metadata prepared by the normal adaptive budget proof.
    /// Conformance-armed admission preserves its original serial semantics
    /// and deliberately leaves this absent.
    pub(super) prepared_metadata: Option<Vec<Option<Vec<u8>>>>,
    #[cfg(test)]
    pub(super) budget_worker_slots: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ParallelDocumentBudgetBound {
    Within {
        logical_upper_bound: usize,
        canonical_metadata: Option<Vec<u8>>,
    },
    ReachesCeiling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ParallelIngestRoute {
    Serial,
    SharedNothing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IngestParallelismPolicy {
    Adaptive,
    #[cfg(feature = "pruning-conformance")]
    ScalarTopologyConformance,
}

/// Proof that the complete ingest batch passed the mutation-free admission
/// check before any route touched writer state.
///
/// Keeping this as a private capability makes the parallel helpers state their
/// real precondition without rescanning every document after the public driver
/// has already established it. Shipping callers obtain the value only from
/// [`super::writer::QuillWriterState::validate_batch_admission`]; route-level unit tests may
/// construct it directly to isolate planner and budget behavior.
#[derive(Debug, Clone, Copy)]
pub(super) struct ValidatedBatchAdmission<'a> {
    pub(super) documents: IngestBatch<'a>,
}

#[derive(Clone)]
pub(super) struct ParallelIngestCheckpoint {
    #[cfg(feature = "conformance-internals")]
    pub(super) controller: Arc<ConformanceCancellationController>,
}

impl ParallelIngestCheckpoint {
    #[cfg(not(feature = "conformance-internals"))]
    pub(super) const fn new(_: &QuillReader) -> Self {
        Self {}
    }

    #[cfg(feature = "conformance-internals")]
    pub(super) fn new(reader: &QuillReader) -> Self {
        Self {
            controller: Arc::clone(&reader.conformance_controller),
        }
    }

    #[allow(
        clippy::unused_self,
        reason = "keeps worker checkpoint call sites cfg-symmetric with the controller-backed variant"
    )]
    pub(super) fn check(&self, cx: &Cx) -> Result<(), QuillIndexError> {
        #[cfg(feature = "conformance-internals")]
        self.controller
            .checkpoint(ConformanceCancellationStage::ParallelIngest, cx);
        // A conformance-only worker FAILURE, which is not a cancellation and
        // must not be reported as one. It returns the same typed class a
        // caught worker panic produces (`catch_parallel_ingest_worker`), so
        // the driver takes the identical non-cancel path to an `aborted`
        // terminal without any panic being injected.
        #[cfg(feature = "conformance-internals")]
        if self
            .controller
            .checkpoint_worker_failure(ConformanceCancellationStage::ParallelIngestWorkerFailure)
        {
            return Err(invalid_state(
                "parallel ingest worker failed before transactional commit",
            ));
        }
        check_cancel(cx, "parallel index worker")
    }
}

impl ParallelIngestRoute {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::SharedNothing => "shared_nothing",
        }
    }
}

pub(super) fn catch_parallel_ingest_worker<T>(
    shard: usize,
    operation: impl FnOnce() -> Result<T, QuillIndexError>,
) -> Result<T, QuillIndexError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).unwrap_or_else(|_| {
        Err(invalid_state(format!(
            "parallel ingest worker {shard} panicked before transactional commit"
        )))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ParallelDocumentRange {
    pub(super) start: usize,
    pub(super) end: usize,
}

impl ParallelDocumentRange {
    pub(super) const fn len(self) -> usize {
        self.end - self.start
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ParallelIngestPlan {
    pub(super) planner_version: u8,
    pub(super) route: ParallelIngestRoute,
    pub(super) configured_width: usize,
    pub(super) verified_pool_capacity: usize,
    pub(super) eligible_shards: usize,
    pub(super) active_shards: usize,
    pub(super) ranges: Vec<ParallelDocumentRange>,
}

pub(super) fn plan_parallel_ingest(
    document_count: usize,
    configured_width: usize,
    verified_pool_capacity: usize,
) -> Result<ParallelIngestPlan, QuillIndexError> {
    let eligible_shards = configured_width.min(verified_pool_capacity);
    let active_shards = eligible_shards.min(document_count / PARALLEL_INGEST_MIN_DOCS_PER_SHARD);
    let route = if active_shards < 2 {
        ParallelIngestRoute::Serial
    } else {
        ParallelIngestRoute::SharedNothing
    };
    let mut ranges = Vec::new();
    if route == ParallelIngestRoute::SharedNothing {
        ranges
            .try_reserve_exact(active_shards)
            .map_err(|_| invalid_state("could not reserve parallel ingest plan ranges"))?;
        let documents_per_shard = document_count / active_shards;
        let remainder = document_count % active_shards;
        let mut start = 0_usize;
        for position in 0..active_shards {
            let count = documents_per_shard + usize::from(position < remainder);
            let end = start
                .checked_add(count)
                .ok_or_else(|| invalid_state("parallel ingest plan range overflow"))?;
            ranges.push(ParallelDocumentRange { start, end });
            start = end;
        }
        if start != document_count {
            return Err(invalid_state(
                "parallel ingest plan did not cover the complete document batch",
            ));
        }
    }
    Ok(ParallelIngestPlan {
        planner_version: PARALLEL_INGEST_PLANNER_VERSION,
        route,
        configured_width,
        verified_pool_capacity,
        eligible_shards,
        active_shards,
        ranges,
    })
}

fn checked_budget_charge(total: &mut usize, additional: usize) -> bool {
    let Some(charged) = total.checked_add(additional) else {
        return false;
    };
    *total = charged;
    true
}

fn charge_parallel_budget_token(total: &mut usize, term_bytes: usize, row_bytes: usize) -> bool {
    let Some(additional) = term_bytes
        .checked_add(FIELD_PREFIX_BYTES)
        .and_then(|value| value.checked_add(std::mem::size_of::<ArenaSpan>()))
        .and_then(|value| value.checked_add(TERM_BUCKET_BYTES_ESTIMATE))
        .and_then(|value| value.checked_add(row_bytes))
    else {
        return false;
    };
    checked_budget_charge(total, additional)
}

pub(super) fn validate_parallel_source_len(
    field_ord: u16,
    bytes: usize,
) -> Result<(), QuillIndexError> {
    if u32::try_from(bytes).is_err() {
        return Err(AccumulatorError::SourceTooLarge { field_ord, bytes }.into());
    }
    Ok(())
}

/// Return a conservative increment to `ColumnarAccumulator::bytes_used` for
/// one shipping-shaped document.
///
/// Every admitted token occurrence is charged as though it were a distinct
/// term. That deliberately overstates interner use for repeated terms while
/// retaining the exact configured analyzer, normalized term bytes, positions
/// width, stored bytes, and per-document column rows. `None` means the schema
/// is not the five-field shipping shape or an arithmetic bound overflowed, so
/// the caller must retain the scalar route. Both Frankensearch analyzers are
/// admitted: the bound runs the field's own analyzer, so every CJK bigram the
/// accumulator will intern is charged here too.
pub(super) fn parallel_document_logical_upper_bound(
    schema: SchemaDescriptor,
    analyzer: &mut FrankensearchTokenizer,
    document: &IndexableDocument,
    exclusive_ceiling: usize,
) -> Result<Option<ParallelDocumentBudgetBound>, QuillIndexError> {
    let [
        id_field,
        content_field,
        title_field,
        metadata_field,
        ord_field,
    ] = schema.fields
    else {
        return Ok(None);
    };
    if !matches!(id_field.kind, FieldKind::Keyword)
        || !id_field.stored
        || !matches!(
            content_field.kind,
            FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault | Analyzer::FrankensearchCjkBigrams,
                ..
            }
        )
        || !content_field.stored
        || !matches!(
            title_field.kind,
            FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault | Analyzer::FrankensearchCjkBigrams,
                ..
            }
        )
        || !title_field.stored
        || !matches!(metadata_field.kind, FieldKind::StoredOnly)
        || !metadata_field.stored
        || !matches!(ord_field.kind, FieldKind::U64 { fast: true, .. })
        || !ord_field.stored
    {
        return Ok(None);
    }

    validate_parallel_source_len(ID_FIELD, document.id.len())?;
    validate_parallel_source_len(CONTENT_FIELD, document.content.len())?;
    validate_parallel_source_len(TITLE_FIELD, document.title.as_deref().unwrap_or("").len())?;

    let mut upper_bound = std::mem::size_of::<u32>();
    for field in schema.fields {
        if matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. })
            && !checked_budget_charge(
                &mut upper_bound,
                std::mem::size_of::<u32>() + std::mem::size_of::<u8>(),
            )
        {
            return Ok(None);
        }
        if field.kind.has_numeric_column()
            && !checked_budget_charge(
                &mut upper_bound,
                std::mem::size_of::<Option<NumericValue>>(),
            )
        {
            return Ok(None);
        }
        if field.stored
            && !checked_budget_charge(
                &mut upper_bound,
                std::mem::size_of::<u32>() + std::mem::size_of::<u8>(),
            )
        {
            return Ok(None);
        }
    }

    let metadata = canonical_metadata(&document.metadata)?;
    if u32::try_from(metadata.len()).is_err() {
        return Err(AccumulatorError::StoredValueTooLarge {
            field_ord: METADATA_FIELD,
            bytes: metadata.len(),
        }
        .into());
    }
    for stored_bytes in [
        document.id.len(),
        document.content.len(),
        document.title.as_deref().unwrap_or("").len(),
        metadata.len(),
        std::mem::size_of::<u64>(),
    ] {
        if !checked_budget_charge(&mut upper_bound, stored_bytes) {
            return Ok(None);
        }
    }

    // Stored bytes and fixed rows are a cheap lower bound. Once they alone
    // reach the caller's remaining admission budget, do not scan either text
    // field merely to prove a rejection already established without
    // tokenization. This also bounds the count-only double scan to source
    // sizes below the configured logical ceiling.
    if upper_bound >= exclusive_ceiling {
        return Ok(Some(ParallelDocumentBudgetBound::ReachesCeiling));
    }

    if document.id.len() <= MAX_TERM_BYTES
        && !charge_parallel_budget_token(
            &mut upper_bound,
            document.id.len(),
            2 * std::mem::size_of::<u32>(),
        )
    {
        return Ok(None);
    }
    if upper_bound >= exclusive_ceiling {
        return Ok(Some(ParallelDocumentBudgetBound::ReachesCeiling));
    }

    for (field, text) in [
        (content_field, document.content.as_str()),
        (title_field, document.title.as_deref().unwrap_or("")),
    ] {
        let FieldKind::Text {
            analyzer: analyzer_kind,
            positions,
        } = field.kind
        else {
            unreachable!("shipping-shape validation accepted a non-text field");
        };
        let row_bytes =
            2 * std::mem::size_of::<u32>() + usize::from(positions) * std::mem::size_of::<u32>();
        let mut overflowed = false;
        let mut reached_ceiling = false;
        analyzer.analyze_borrowed(analyzer_kind, text, &mut |token| {
            if token.text.len() <= MAX_TERM_BYTES
                && !overflowed
                && !reached_ceiling
                && !charge_parallel_budget_token(&mut upper_bound, token.text.len(), row_bytes)
            {
                overflowed = true;
            }
            reached_ceiling = upper_bound >= exclusive_ceiling;
        });
        if overflowed {
            return Ok(None);
        }
        if reached_ceiling {
            return Ok(Some(ParallelDocumentBudgetBound::ReachesCeiling));
        }
    }

    debug_assert!(upper_bound < exclusive_ceiling);
    Ok(Some(ParallelDocumentBudgetBound::Within {
        logical_upper_bound: upper_bound,
        canonical_metadata: Some(metadata),
    }))
}

pub(super) fn retain_prepared_metadata_within_budget(
    metadata: Vec<u8>,
    retained_bytes: &AtomicUsize,
    exclusive_ceiling: usize,
) -> Option<Vec<u8>> {
    retained_bytes
        .try_update(Ordering::AcqRel, Ordering::Acquire, |retained| {
            retained
                .checked_add(metadata.len())
                .filter(|projected| *projected < exclusive_ceiling)
        })
        .ok()
        .map(|_| metadata)
}

pub(super) fn parallel_arena_chunk_bytes(
    batch_logical_upper_bound: usize,
    active_shards: usize,
) -> usize {
    // Size speculative term arenas from the admitted work, not from the
    // configuration ceiling. With the 64 MiB default, deriving this from the
    // ceiling would clamp every supported width back to the scalar 1 MiB
    // chunk and multiply idle reservation by the worker count. The second cap
    // shares one scalar chunk of aggregate slack across workers; above 256
    // workers, ByteArena's 4 KiB hard floor is the irreducible bound.
    let active_shards = active_shards.max(1);
    let per_shard_share = batch_logical_upper_bound / active_shards;
    let per_shard_slack_cap =
        (DEFAULT_ARENA_CHUNK_BYTES / active_shards).max(MIN_ARENA_CHUNK_BYTES);
    per_shard_share
        .saturating_div(PARALLEL_ARENA_BUDGET_DIVISOR)
        .clamp(MIN_ARENA_CHUNK_BYTES, per_shard_slack_cap)
}

/// Minimum documents a shard must receive to be worth enrolling in a fan-out.
///
/// Each participating shard seals its own segment, and every segment carries a
/// full term dictionary, so a wide split of a small batch buys parallelism at
/// the cost of index bytes and search-time segment count. Batches smaller than
/// twice this stay on the single-shard path.
pub(super) const FANOUT_MIN_SHARD_DOCUMENTS: usize = 256;

/// Documents handed to one shard per parallel wave. A seal cannot be awaited
/// from inside the parallel region, so this bounds how far a shard's arena can
/// exceed `scribe_shard_budget_bytes` before the between-wave seal check runs.
pub(super) const FANOUT_WAVE_SHARD_DOCUMENTS: usize = 256;

/// Accumulate one contiguous run of documents into one shard.
///
/// Shared-nothing: touches only `state`, so shards run concurrently. Admission
/// and docid allocation have already happened on the serial path.
pub(super) fn accumulate_shard_run(
    state: &mut ScribeShardState,
    documents: IngestBatch<'_>,
    span: DocIdSpan,
) -> Result<(usize, usize), QuillIndexError> {
    // Disjoint field borrows: the accumulator, the identity log and the
    // metadata scratch buffer are all touched per document, so they are
    // split out once instead of being re-borrowed through `state` each time.
    let ScribeShardState {
        accumulator,
        identities,
        scratch_metadata,
        ..
    } = state;
    let mut bytes_used_high_water = 0_usize;
    let mut bytes_reserved_high_water = 0_usize;
    for (offset, document) in documents.iter().enumerate() {
        let span_offset = u32::try_from(offset)
            .map_err(|_| invalid_state("fan-out document offset does not fit u32"))?;
        let doc_ord = span
            .ord_start
            .checked_add(span_offset)
            .ok_or_else(|| invalid_state("lease-relative document ordinal overflow"))?;
        let global_docid = span
            .lease_base
            .checked_add(u64::from(doc_ord))
            .filter(|docid| *docid < MAX_GLOBAL_DOCID_EXCLUSIVE)
            .ok_or_else(|| invalid_state("global Q1 document-id space exhausted"))?;
        document.write_canonical_metadata(scratch_metadata)?;
        let accumulated =
            document.accumulate(accumulator, doc_ord, global_docid, scratch_metadata)?;
        bytes_used_high_water = bytes_used_high_water.max(accumulated.bytes_used);
        bytes_reserved_high_water = bytes_reserved_high_water.max(accumulated.bytes_reserved);
        let content_hash = document.content_hash(scratch_metadata)?;
        identities.push(PendingIdentity {
            doc_ord,
            document_id: document.id().to_owned(),
            content_hash,
        });
    }
    Ok((bytes_used_high_water, bytes_reserved_high_water))
}
