#[cfg(feature = "pruning-conformance")]
use crate::argus::{
    ConformanceQueryWorkContext, ConformanceQueryWorkPhase, ConformanceUnionRefill,
    ConformanceUnionRefillStrategy, QueryWorkKind,
};
use crate::keeper::{Manifest, ManifestSegment};
use asupersync::Cx;
#[cfg(feature = "pruning-conformance")]
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
#[cfg(feature = "pruning-conformance")]
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use super::checkpoint::PendingDeltaSeal;
use super::error_results::{QuillIndexError, invalid_state};

/// Refill strategy observed by the exact pruning-conformance witness.
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConformancePruningStrategy {
    /// Full window scoring without a competitive cutoff.
    Exhaustive,
    /// Direct-term `MaxScore` candidate selection.
    MaxScore,
    /// Block-Max WAND candidate selection.
    BlockMaxWand,
}

/// One exact direct-term union refill from a conformance-only search.
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformancePruningRefillReceipt {
    ordinal: u64,
    window_start: u32,
    horizon_end: u64,
    cutoff_bits: Option<u32>,
    strategy: ConformancePruningStrategy,
    candidate_docs: u64,
    buffer_empty: bool,
    live_work_remains: bool,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningRefillReceipt {
    /// One-based refill ordinal within the segment scorer.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Inclusive global document identifier that starts this window.
    #[must_use]
    pub const fn window_start(&self) -> u32 {
        self.window_start
    }

    /// Exclusive global document bound for this window.
    #[must_use]
    pub const fn horizon_end(&self) -> u64 {
        self.horizon_end
    }

    /// Exact competitive cutoff bits, or `None` before the heap is full.
    #[must_use]
    pub const fn cutoff_bits(&self) -> Option<u32> {
        self.cutoff_bits
    }

    /// Scoring strategy used for this refill.
    #[must_use]
    pub const fn strategy(&self) -> ConformancePruningStrategy {
        self.strategy
    }

    /// Number of documents admitted by the selected strategy.
    #[must_use]
    pub const fn candidate_docs(&self) -> u64 {
        self.candidate_docs
    }

    /// Whether the refill produced no buffered scored document.
    #[must_use]
    pub const fn buffer_empty(&self) -> bool {
        self.buffer_empty
    }

    /// Whether at least one posting cursor remains beyond this window.
    #[must_use]
    pub const fn live_work_remains(&self) -> bool {
        self.live_work_remains
    }
}

/// Ordered refill receipts for one immutable Quill segment.
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformanceSegmentPruningReceipt {
    segment_ordinal: u64,
    segment_doc_count: u64,
    refills: Vec<ConformancePruningRefillReceipt>,
}

#[cfg(feature = "pruning-conformance")]
impl ConformanceSegmentPruningReceipt {
    /// Stable ordinal in the committed Quill snapshot.
    #[must_use]
    pub const fn segment_ordinal(&self) -> u64 {
        self.segment_ordinal
    }

    /// Physical document cardinality of the scored segment.
    #[must_use]
    pub const fn segment_doc_count(&self) -> u64 {
        self.segment_doc_count
    }

    /// Ordered refill events observed while collecting this segment.
    #[must_use]
    pub fn refills(&self) -> &[ConformancePruningRefillReceipt] {
        &self.refills
    }
}

#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConformancePruningExecutionMode {
    /// The shipping collector scored sealed segments in snapshot order.
    Serial,
    /// The shipping collector scored sealed segments through Rayon fan-out.
    Rayon,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningExecutionMode {
    const SERIAL_CODE: u8 = 1;
    const RAYON_CODE: u8 = 2;

    pub(super) const fn code(self) -> u8 {
        match self {
            Self::Serial => Self::SERIAL_CODE,
            Self::Rayon => Self::RAYON_CODE,
        }
    }

    pub(super) const fn from_code(code: u8) -> Option<Self> {
        match code {
            Self::SERIAL_CODE => Some(Self::Serial),
            Self::RAYON_CODE => Some(Self::Rayon),
            _ => None,
        }
    }
}

#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformancePruningTraceReceipt {
    execution_mode: ConformancePruningExecutionMode,
    segments: Vec<ConformanceSegmentPruningReceipt>,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningTraceReceipt {
    /// Actual shipping collection branch used by this exact search invocation.
    #[must_use]
    pub const fn execution_mode(&self) -> ConformancePruningExecutionMode {
        self.execution_mode
    }

    /// Complete receipts in dense Quill segment order.
    #[must_use]
    pub fn segments(&self) -> &[ConformanceSegmentPruningReceipt] {
        &self.segments
    }
}

#[cfg(feature = "pruning-conformance")]
#[derive(Debug)]
pub(super) enum ConformancePruningTraceState {
    Collecting {
        expected_segments: usize,
        execution_mode: Option<ConformancePruningExecutionMode>,
        receipts: Vec<ConformanceSegmentPruningReceipt>,
    },
    Completed,
    Failed,
}

#[cfg(feature = "pruning-conformance")]
#[derive(Debug)]
pub(super) struct ConformancePruningTraceSession {
    pub(super) state: StdMutex<ConformancePruningTraceState>,
    pub(super) cancellation_arm_generation: Option<u64>,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningTraceSession {
    #[cfg(test)]
    pub(super) fn new(expected_segments: usize) -> Self {
        Self::new_for_cancellation_arm(expected_segments, None)
    }

    pub(super) fn new_for_cancellation_arm(
        expected_segments: usize,
        cancellation_arm_generation: Option<u64>,
    ) -> Self {
        Self {
            state: StdMutex::new(ConformancePruningTraceState::Collecting {
                expected_segments,
                execution_mode: None,
                receipts: Vec::new(),
            }),
            cancellation_arm_generation,
        }
    }

    pub(super) const fn cancellation_arm_generation(&self) -> Option<u64> {
        self.cancellation_arm_generation
    }

    pub(super) fn bind_execution_mode(
        &self,
        execution_mode: ConformancePruningExecutionMode,
    ) -> Result<(), QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("pruning-conformance receipt lock is poisoned"))?;
        let result = match &mut *state {
            ConformancePruningTraceState::Collecting {
                execution_mode: slot,
                ..
            } if slot.is_none() => {
                *slot = Some(execution_mode);
                Ok(())
            }
            ConformancePruningTraceState::Collecting { .. } => {
                *state = ConformancePruningTraceState::Failed;
                Err(invalid_state(
                    "pruning-conformance execution mode was already bound",
                ))
            }
            ConformancePruningTraceState::Completed => Err(invalid_state(
                "pruning-conformance trace session is already completed",
            )),
            ConformancePruningTraceState::Failed => Err(invalid_state(
                "pruning-conformance trace session has failed",
            )),
        };
        drop(state);
        result
    }

    #[cfg(test)]
    pub(super) fn record(
        &self,
        segment_ordinal: u64,
        segment_doc_count: u64,
        refills: &[ConformanceUnionRefill],
    ) -> Result<u64, QuillIndexError> {
        self.record_inner(segment_ordinal, segment_doc_count, refills, None)
    }

    pub(super) fn record_and_checkpoint(
        &self,
        segment_ordinal: u64,
        segment_doc_count: u64,
        refills: &[ConformanceUnionRefill],
        controller: &ConformanceCancellationController,
        cx: &Cx,
    ) -> Result<u64, QuillIndexError> {
        self.record_inner(
            segment_ordinal,
            segment_doc_count,
            refills,
            Some((controller, cx)),
        )
    }

    fn record_inner(
        &self,
        segment_ordinal: u64,
        segment_doc_count: u64,
        refills: &[ConformanceUnionRefill],
        checkpoint: Option<(&ConformanceCancellationController, &Cx)>,
    ) -> Result<u64, QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("pruning-conformance receipt lock is poisoned"))?;
        let ConformancePruningTraceState::Collecting {
            expected_segments,
            execution_mode,
            receipts,
        } = &mut *state
        else {
            return match &*state {
                ConformancePruningTraceState::Completed => Err(invalid_state(
                    "pruning-conformance trace session is already completed",
                )),
                ConformancePruningTraceState::Failed => Err(invalid_state(
                    "pruning-conformance trace session has failed",
                )),
                ConformancePruningTraceState::Collecting { .. } => {
                    Err(invalid_state("pruning-conformance trace state is invalid"))
                }
            };
        };
        let Some(execution_mode) = *execution_mode else {
            *state = ConformancePruningTraceState::Failed;
            return Err(invalid_state(
                "pruning-conformance execution mode was not bound before recording",
            ));
        };
        let expected_segments = u64::try_from(*expected_segments)
            .map_err(|_| invalid_state("pruning-conformance segment count does not fit u64"))?;
        if segment_ordinal >= expected_segments {
            *state = ConformancePruningTraceState::Failed;
            return Err(invalid_state(format!(
                "pruning-conformance segment ordinal {segment_ordinal} exceeds expected count \
                 {expected_segments}",
            )));
        }
        if receipts
            .iter()
            .any(|receipt| receipt.segment_ordinal == segment_ordinal)
        {
            *state = ConformancePruningTraceState::Failed;
            return Err(invalid_state(format!(
                "pruning-conformance segment ordinal {segment_ordinal} was recorded twice",
            )));
        }
        let refills = refills
            .iter()
            .map(|refill| ConformancePruningRefillReceipt {
                ordinal: refill.ordinal,
                window_start: refill.window_start,
                horizon_end: refill.horizon_end,
                cutoff_bits: refill.cutoff_bits,
                strategy: match refill.strategy {
                    ConformanceUnionRefillStrategy::Exhaustive => {
                        ConformancePruningStrategy::Exhaustive
                    }
                    ConformanceUnionRefillStrategy::MaxScore => {
                        ConformancePruningStrategy::MaxScore
                    }
                    ConformanceUnionRefillStrategy::BlockMaxWand => {
                        ConformancePruningStrategy::BlockMaxWand
                    }
                },
                candidate_docs: refill.candidate_docs,
                buffer_empty: refill.buffer_empty,
                live_work_remains: refill.live_work_remains,
            })
            .collect();
        receipts.push(ConformanceSegmentPruningReceipt {
            segment_ordinal,
            segment_doc_count,
            refills,
        });
        let Ok(recorded_receipts) = u64::try_from(receipts.len()) else {
            *state = ConformancePruningTraceState::Failed;
            return Err(invalid_state(
                "pruning-conformance recorded receipt count does not fit u64",
            ));
        };
        if let Some((controller, cx)) = checkpoint
            && let Err(error) = controller.checkpoint_pruning_trace_segment_recorded(
                cx,
                recorded_receipts,
                self.cancellation_arm_generation,
                execution_mode,
            )
        {
            *state = ConformancePruningTraceState::Failed;
            return Err(error);
        }
        drop(state);
        Ok(recorded_receipts)
    }

    #[allow(
        clippy::significant_drop_tightening,
        reason = "the trace state must remain locked from extraction through the terminal Completed transition"
    )]
    pub(super) fn complete(&self) -> Result<ConformancePruningTraceReceipt, QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("pruning-conformance receipt lock is poisoned"))?;
        let collecting = std::mem::replace(&mut *state, ConformancePruningTraceState::Failed);
        let ConformancePruningTraceState::Collecting {
            expected_segments,
            execution_mode,
            mut receipts,
        } = collecting
        else {
            return match collecting {
                ConformancePruningTraceState::Completed => Err(invalid_state(
                    "pruning-conformance trace session is already completed",
                )),
                ConformancePruningTraceState::Failed => Err(invalid_state(
                    "pruning-conformance trace session has failed",
                )),
                ConformancePruningTraceState::Collecting { .. } => {
                    Err(invalid_state("pruning-conformance trace state is invalid"))
                }
            };
        };
        let execution_mode = execution_mode
            .ok_or_else(|| invalid_state("pruning-conformance execution mode was never bound"))?;
        receipts.sort_by_key(|receipt| receipt.segment_ordinal);
        if receipts.len() != expected_segments
            || !receipts.iter().enumerate().all(|(ordinal, receipt)| {
                u64::try_from(ordinal).is_ok_and(|ordinal| ordinal == receipt.segment_ordinal)
            })
        {
            return Err(invalid_state(format!(
                "pruning-conformance trace is incomplete: expected {expected_segments} dense \
                 segment receipts, observed {}",
                receipts.len(),
            )));
        }
        *state = ConformancePruningTraceState::Completed;
        drop(state);
        Ok(ConformancePruningTraceReceipt {
            execution_mode,
            segments: receipts,
        })
    }

    pub(super) fn fail(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = ConformancePruningTraceState::Failed;
        }
    }
}

#[cfg(feature = "pruning-conformance")]
#[derive(Debug)]
pub(super) struct ConformancePruningTraceGuard {
    pub(super) session: Arc<ConformancePruningTraceSession>,
    pub(super) controller: Arc<ConformanceCancellationController>,
    pub(super) finished: bool,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningTraceGuard {
    pub(super) fn new(
        expected_segments: usize,
        controller: Arc<ConformanceCancellationController>,
    ) -> Self {
        let cancellation_arm_generation = controller.active_pruning_trace_arm_generation();
        Self {
            session: Arc::new(ConformancePruningTraceSession::new_for_cancellation_arm(
                expected_segments,
                cancellation_arm_generation,
            )),
            controller,
            finished: false,
        }
    }

    pub(super) fn session(&self) -> &ConformancePruningTraceSession {
        self.session.as_ref()
    }

    pub(super) fn complete(mut self) -> Result<ConformancePruningTraceReceipt, QuillIndexError> {
        let receipt = self.session.complete()?;
        self.finished = true;
        Ok(receipt)
    }
}

#[cfg(feature = "pruning-conformance")]
impl Drop for ConformancePruningTraceGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.session.fail();
            if let Some(arm_generation) = self.session.cancellation_arm_generation() {
                self.controller
                    .note_discarded_pruning_trace_session(arm_generation);
            }
        }
    }
}

/// Deterministic request-cancellation checkpoints exposed only to the
/// conformance harness.
///
/// Synchronous Quill collection cannot be preempted from an external test
/// task at a reproducible work unit. The conformance controller therefore
/// requests cancellation on the real [`Cx`] from an exact engine checkpoint;
/// the public method must still observe that request through its normal
/// cancellation poll and return its own typed outcome.
#[cfg(feature = "conformance-internals")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConformanceCancellationStage {
    /// One admitted collection work unit.
    QueryCollection = 1,
    /// One deferred-metadata candidate before materialization.
    FusionHydration = 2,
    /// The retained commit transaction immediately before publication.
    CommitPublication = 3,
    /// One document boundary inside a shared-nothing ingest worker.
    ParallelIngest = 4,
    /// One document boundary in the read-only parallel budget preflight.
    ParallelBudgetAdmission = 5,
    /// One sealed-segment pruning receipt has been recorded successfully.
    PruningTraceSegmentRecorded = 6,
    /// Ranked hits are complete and the snippet tail is about to begin.
    SnippetTailAdmission = 7,
    /// One document boundary inside a shared-nothing ingest worker, failed
    /// rather than cancelled.
    ///
    /// Distinct from [`Self::ParallelIngest`] because the two prove different
    /// terminals. That stage requests cancellation on the real `Cx` and the
    /// operation ends `cancelled`; this one leaves `Cx` untouched and returns
    /// a typed worker failure, so the operation ends `aborted`. Arming one
    /// never advances the other's ordinal: the stage guard rejects a
    /// non-matching stage before the counter moves.
    ParallelIngestWorkerFailure = 8,
}

/// Fixed-size conformance receipt for the complete retained scalar writer
/// transaction.
///
/// The digest binds canonical pending segment bytes and identities, staged
/// flush state, retained MANIFEST proposals, dirty shard document witnesses,
/// allocator leases, and all publication-relevant flags. Summary counters stay
/// explicit so a receipt cannot hide a missing state class behind an opaque
/// digest. This type and the hashing work do not exist in shipping builds.
#[cfg(feature = "conformance-internals")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformancePendingWriterState {
    pub(super) digest_sha256: [u8; 32],
    pub(super) shard_count: u64,
    pub(super) dirty_shard_count: u64,
    pub(super) pending_identity_count: u64,
    pub(super) uncommitted_id_count: u64,
    pub(super) pending_segment_count: u64,
    pub(super) pending_owned_segment_count: u64,
    pub(super) flags: u8,
}

#[cfg(feature = "conformance-internals")]
impl ConformancePendingWriterState {
    pub(super) const STAGED_FLUSH_PRESENT: u8 = 1 << 0;
    pub(super) const PENDING_MANIFEST_PRESENT: u8 = 1 << 1;
    pub(super) const PENDING_REPLACEMENT_MANIFEST_PRESENT: u8 = 1 << 2;
    pub(super) const PENDING_DELTA_SEAL_PRESENT: u8 = 1 << 3;
    pub(super) const UNPUBLISHED_SINCE_PRESENT: u8 = 1 << 4;
    pub(super) const INGEST_RETRY_REQUIRED: u8 = 1 << 5;

    const fn has_flag(self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    /// SHA-256 over the canonical complete pending writer transaction.
    #[must_use]
    pub const fn digest_sha256(self) -> [u8; 32] {
        self.digest_sha256
    }

    /// Number of configured scalar writer shards represented by the digest.
    #[must_use]
    pub const fn shard_count(self) -> u64 {
        self.shard_count
    }

    /// Number of shards retaining documents, identities, or an active lease.
    #[must_use]
    pub const fn dirty_shard_count(self) -> u64 {
        self.dirty_shard_count
    }

    /// Canonical document identity rows still resident in dirty shards.
    #[must_use]
    pub const fn pending_identity_count(self) -> u64 {
        self.pending_identity_count
    }

    /// Stable document IDs reserved by all unpublished scalar state.
    #[must_use]
    pub const fn uncommitted_id_count(self) -> u64 {
        self.uncommitted_id_count
    }

    /// Installed but unpublished MANIFEST segment rows.
    #[must_use]
    pub const fn pending_segment_count(self) -> u64 {
        self.pending_segment_count
    }

    /// Canonical in-memory FSLX buffers retained for pending segments.
    #[must_use]
    pub const fn pending_owned_segment_count(self) -> u64 {
        self.pending_owned_segment_count
    }

    /// Whether a flush transaction has been encoded but not fully installed.
    #[must_use]
    pub const fn staged_flush_present(self) -> bool {
        self.has_flag(Self::STAGED_FLUSH_PRESENT)
    }

    /// Whether the exact next ordinary MANIFEST proposal is retained.
    #[must_use]
    pub const fn pending_manifest_present(self) -> bool {
        self.has_flag(Self::PENDING_MANIFEST_PRESENT)
    }

    /// Whether a complete replacement MANIFEST proposal is retained.
    #[must_use]
    pub const fn pending_replacement_manifest_present(self) -> bool {
        self.has_flag(Self::PENDING_REPLACEMENT_MANIFEST_PRESENT)
    }

    /// Whether a Delta-seal publication transaction is retained.
    #[must_use]
    pub const fn pending_delta_seal_present(self) -> bool {
        self.has_flag(Self::PENDING_DELTA_SEAL_PRESENT)
    }

    /// Whether the writer is tracking an unpublished visibility interval.
    #[must_use]
    pub const fn unpublished_since_present(self) -> bool {
        self.has_flag(Self::UNPUBLISHED_SINCE_PRESENT)
    }

    /// Whether scalar ingest requires reconciliation before more mutation.
    #[must_use]
    pub const fn ingest_retry_required(self) -> bool {
        self.has_flag(Self::INGEST_RETRY_REQUIRED)
    }
}

#[cfg(feature = "conformance-internals")]
impl ConformanceCancellationStage {
    pub(super) const fn code(self) -> u8 {
        match self {
            Self::QueryCollection => 1,
            Self::FusionHydration => 2,
            Self::CommitPublication => 3,
            Self::ParallelIngest => 4,
            Self::ParallelBudgetAdmission => 5,
            Self::PruningTraceSegmentRecorded => 6,
            Self::SnippetTailAdmission => 7,
            Self::ParallelIngestWorkerFailure => 8,
        }
    }
}

/// Exact competitive-union location where a conformance-only query was
/// interrupted by deterministic cancellation or fuel exhaustion.
#[cfg(feature = "pruning-conformance")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformanceQueryInterruptionLocation {
    refill_ordinal: u64,
    window_start: u32,
    phase: ConformanceQueryWorkPhase,
    candidate_doc: Option<u32>,
    candidate_scored: bool,
    work_kind: QueryWorkKind,
    units: u64,
    consumed_before: u64,
}

#[cfg(feature = "pruning-conformance")]
impl ConformanceQueryInterruptionLocation {
    pub(super) const fn new(
        context: ConformanceQueryWorkContext,
        work_kind: QueryWorkKind,
        units: u64,
        consumed_before: u64,
    ) -> Self {
        Self {
            refill_ordinal: context.refill_ordinal,
            window_start: context.window_start,
            phase: context.phase,
            candidate_doc: context.candidate_doc,
            candidate_scored: context.candidate_scored,
            work_kind,
            units,
            consumed_before,
        }
    }

    /// One-based refill ordinal, including the initial exhaustive refill.
    #[must_use]
    pub const fn refill_ordinal(self) -> u64 {
        self.refill_ordinal
    }

    /// Inclusive document at the start of the interrupted refill window.
    #[must_use]
    pub const fn window_start(self) -> u32 {
        self.window_start
    }

    /// Competitive-union operation surrounding the interrupted admission.
    #[must_use]
    pub const fn phase(self) -> ConformanceQueryWorkPhase {
        self.phase
    }

    /// Selected candidate, if the phase had one.
    #[must_use]
    pub const fn candidate_doc(self) -> Option<u32> {
        self.candidate_doc
    }

    /// Whether that candidate's score contribution completed before the
    /// interrupted admission.
    #[must_use]
    pub const fn candidate_scored(self) -> bool {
        self.candidate_scored
    }

    /// Coarse work class being admitted at the interruption boundary.
    #[must_use]
    pub const fn work_kind(self) -> QueryWorkKind {
        self.work_kind
    }

    /// Work units requested by the interrupted admission.
    #[must_use]
    pub const fn units(self) -> u64 {
        self.units
    }

    /// Successfully admitted query-fuel units before this admission.
    #[must_use]
    pub const fn consumed_before(self) -> u64 {
        self.consumed_before
    }
}

/// Per-index deterministic cancellation requester for live conformance tests.
///
/// This type, its atomics, and every call site are absent unless the dedicated
/// `conformance-internals` feature is enabled. It cannot fabricate a return
/// value: firing only marks the real request [`Cx`] as cancelled.
#[cfg(feature = "conformance-internals")]
#[derive(Debug, Default)]
pub struct ConformanceCancellationController {
    pub(super) stage: AtomicU8,
    pub(super) arm_generation: AtomicU64,
    arm_transition_lock: StdMutex<()>,
    trigger_ordinal: AtomicU64,
    pub(super) observed_checkpoints: AtomicU64,
    pub(super) recorded_pruning_receipts_at_fire: AtomicU64,
    #[cfg(feature = "pruning-conformance")]
    pub(super) pruning_trace_execution_mode_at_fire: AtomicU8,
    #[cfg(feature = "pruning-conformance")]
    pub(super) query_interruption_location: StdMutex<Option<ConformanceQueryInterruptionLocation>>,
    pub(super) discarded_pruning_trace_sessions: AtomicU64,
    pub(super) fired: AtomicBool,
}

#[cfg(feature = "conformance-internals")]
impl ConformanceCancellationController {
    pub(super) const DISARMED: u8 = 0;
    const ARMING: u8 = u8::MAX;

    /// Arm one exact nonzero checkpoint ordinal.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error for ordinal zero or when a prior stage
    /// remains armed.
    pub fn arm(
        &self,
        stage: ConformanceCancellationStage,
        trigger_ordinal: u64,
    ) -> Result<(), QuillIndexError> {
        if trigger_ordinal == 0 {
            return Err(invalid_state(
                "conformance cancellation trigger ordinal must be nonzero",
            ));
        }
        let arm_transition = self.arm_transition_lock.lock().map_err(|_| {
            invalid_state("conformance cancellation arm transition lock is poisoned")
        })?;
        self.stage
            .compare_exchange(
                Self::DISARMED,
                Self::ARMING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| invalid_state("conformance cancellation controller is already armed"))?;
        let Some(arm_generation) = self.arm_generation.load(Ordering::Relaxed).checked_add(1)
        else {
            self.stage.store(Self::DISARMED, Ordering::Release);
            return Err(invalid_state(
                "conformance cancellation arm generation is exhausted",
            ));
        };
        self.arm_generation.store(arm_generation, Ordering::Relaxed);
        self.trigger_ordinal
            .store(trigger_ordinal, Ordering::Relaxed);
        self.observed_checkpoints.store(0, Ordering::Relaxed);
        self.recorded_pruning_receipts_at_fire
            .store(0, Ordering::Relaxed);
        #[cfg(feature = "pruning-conformance")]
        self.pruning_trace_execution_mode_at_fire
            .store(0, Ordering::Relaxed);
        #[cfg(feature = "pruning-conformance")]
        self.reset_query_interruption_location();
        self.discarded_pruning_trace_sessions
            .store(0, Ordering::Relaxed);
        self.fired.store(false, Ordering::Relaxed);
        self.stage.store(stage.code(), Ordering::Release);
        drop(arm_transition);
        Ok(())
    }

    /// Stop injecting checkpoints without changing request cancellation.
    ///
    /// The caller must clear the real [`Cx`] explicitly. Keeping those actions
    /// separate proves the request context remains the cancellation authority.
    pub fn disarm(&self) {
        let _arm_transition = self.arm_transition_lock.lock().ok();
        self.stage.store(Self::DISARMED, Ordering::Release);
    }

    /// Number of matching stage checkpoints observed by the current arm.
    #[must_use]
    pub fn observed_checkpoints(&self) -> u64 {
        self.observed_checkpoints.load(Ordering::Acquire)
    }

    /// Successfully recorded pruning receipts at the exact firing boundary.
    ///
    /// This stays zero for every non-pruning cancellation stage. A nonzero
    /// value proves the pruning checkpoint ran only after receipt mutation
    /// succeeded; it does not infer how much unrecorded scorer work remained.
    #[must_use]
    pub fn recorded_pruning_receipts_at_fire(&self) -> u64 {
        self.recorded_pruning_receipts_at_fire
            .load(Ordering::Acquire)
    }

    /// Shipping sealed-segment execution branch at the pruning fire boundary.
    #[cfg(feature = "pruning-conformance")]
    #[must_use]
    pub fn pruning_trace_execution_mode_at_fire(&self) -> Option<ConformancePruningExecutionMode> {
        ConformancePruningExecutionMode::from_code(
            self.pruning_trace_execution_mode_at_fire
                .load(Ordering::Acquire),
        )
    }

    /// Invocation-bound pruning trace sessions failed and discarded since arm.
    #[must_use]
    pub fn discarded_pruning_trace_sessions(&self) -> u64 {
        self.discarded_pruning_trace_sessions
            .load(Ordering::Acquire)
    }

    /// Whether the current arm requested cancellation on its real [`Cx`].
    #[must_use]
    pub fn fired(&self) -> bool {
        self.fired.load(Ordering::Acquire)
    }

    pub(super) fn is_armed(&self) -> bool {
        self.stage.load(Ordering::Acquire) != Self::DISARMED
    }

    /// Exact competitive-union interruption most recently captured for this
    /// controller, or `None` before a qualifying interruption.
    #[cfg(feature = "pruning-conformance")]
    #[must_use]
    pub fn query_interruption_location(&self) -> Option<ConformanceQueryInterruptionLocation> {
        *self
            .query_interruption_location
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(feature = "pruning-conformance")]
    pub(super) fn reset_query_interruption_location(&self) {
        *self
            .query_interruption_location
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    #[cfg(feature = "pruning-conformance")]
    pub(super) fn record_query_interruption_location(
        &self,
        location: ConformanceQueryInterruptionLocation,
    ) {
        let mut recorded = self
            .query_interruption_location
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if recorded.is_none() {
            *recorded = Some(location);
        }
    }

    pub(super) fn checkpoint(&self, stage: ConformanceCancellationStage, cx: &Cx) {
        let _ = self.checkpoint_with_pruning_receipts(stage, cx, None, None);
    }

    /// Fire the armed ordinal for a stage that must FAIL its caller rather
    /// than cancel the operation.
    ///
    /// Takes no `Cx` on purpose. Requesting cancellation here would make the
    /// driver classify the terminal from a `Cancelled` error and hide the
    /// non-cancel path this stage exists to prove. Arm and ordinal accounting
    /// is otherwise identical to
    /// [`Self::checkpoint_with_pruning_receipts`], including the stage guard
    /// that returns before the ordinal counter moves, so an unrelated armed
    /// stage is never advanced by this call.
    pub(super) fn checkpoint_worker_failure(&self, stage: ConformanceCancellationStage) -> bool {
        if self.stage.load(Ordering::Acquire) != stage.code() {
            return false;
        }
        let ordinal = self
            .observed_checkpoints
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        if ordinal != self.trigger_ordinal.load(Ordering::Acquire) {
            return false;
        }
        self.fired.store(true, Ordering::Release);
        tracing::info!(
            target: crate::tracing_conventions::TARGET,
            event = "quill.conformance.worker_failure_checkpoint",
            ?stage,
            ordinal,
            "deterministic conformance checkpoint failed a parallel ingest worker"
        );
        true
    }

    #[cfg(feature = "pruning-conformance")]
    pub(super) fn checkpoint_query_collection(&self, cx: &Cx) -> bool {
        self.checkpoint_with_pruning_receipts(
            ConformanceCancellationStage::QueryCollection,
            cx,
            None,
            None,
        )
    }

    #[cfg(feature = "pruning-conformance")]
    fn checkpoint_pruning_trace_segment_recorded(
        &self,
        cx: &Cx,
        recorded_receipts: u64,
        expected_arm_generation: Option<u64>,
        execution_mode: ConformancePruningExecutionMode,
    ) -> Result<(), QuillIndexError> {
        if recorded_receipts == 0 {
            return Err(invalid_state(
                "pruning-trace cancellation checkpoint requires a successful receipt",
            ));
        }
        let Some(expected_arm_generation) = expected_arm_generation else {
            return Ok(());
        };
        let _arm_transition = self.arm_transition_lock.lock().map_err(|_| {
            invalid_state("conformance cancellation arm transition lock is poisoned")
        })?;
        if self.stage.load(Ordering::Acquire)
            != ConformanceCancellationStage::PruningTraceSegmentRecorded.code()
            || self.arm_generation.load(Ordering::Acquire) != expected_arm_generation
        {
            return Ok(());
        }
        let _ = self.checkpoint_with_pruning_receipts(
            ConformanceCancellationStage::PruningTraceSegmentRecorded,
            cx,
            Some(recorded_receipts),
            Some(execution_mode.code()),
        );
        Ok(())
    }

    fn checkpoint_with_pruning_receipts(
        &self,
        stage: ConformanceCancellationStage,
        cx: &Cx,
        recorded_receipts: Option<u64>,
        pruning_execution_mode_code: Option<u8>,
    ) -> bool {
        if self.stage.load(Ordering::Acquire) != stage.code() {
            return false;
        }
        let ordinal = self
            .observed_checkpoints
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        if ordinal != self.trigger_ordinal.load(Ordering::Acquire) {
            return false;
        }
        if let Some(recorded_receipts) = recorded_receipts {
            self.recorded_pruning_receipts_at_fire
                .store(recorded_receipts, Ordering::Release);
        }
        #[cfg(feature = "pruning-conformance")]
        if let Some(execution_mode_code) = pruning_execution_mode_code {
            self.pruning_trace_execution_mode_at_fire
                .store(execution_mode_code, Ordering::Release);
        }
        #[cfg(not(feature = "pruning-conformance"))]
        let _ = pruning_execution_mode_code;
        self.fired.store(true, Ordering::Release);
        tracing::info!(
            target: crate::tracing_conventions::TARGET,
            event = "quill.conformance.cancellation_checkpoint",
            ?stage,
            ordinal,
            "deterministic conformance checkpoint requested cancellation on the real Cx"
        );
        cx.set_cancel_requested(true);
        true
    }

    #[cfg(feature = "pruning-conformance")]
    pub(super) fn active_pruning_trace_arm_generation(&self) -> Option<u64> {
        let _arm_transition = self.arm_transition_lock.lock().ok()?;
        (self.stage.load(Ordering::Acquire)
            == ConformanceCancellationStage::PruningTraceSegmentRecorded.code())
        .then(|| self.arm_generation.load(Ordering::Acquire))
    }

    #[cfg(feature = "pruning-conformance")]
    fn note_discarded_pruning_trace_session(&self, expected_arm_generation: u64) {
        let Ok(_arm_transition) = self.arm_transition_lock.lock() else {
            return;
        };
        if self.stage.load(Ordering::Acquire)
            == ConformanceCancellationStage::PruningTraceSegmentRecorded.code()
            && self.arm_generation.load(Ordering::Acquire) == expected_arm_generation
        {
            self.discarded_pruning_trace_sessions
                .fetch_add(1, Ordering::AcqRel);
        }
    }
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_usize_to_u64(
    value: usize,
    label: &'static str,
) -> Result<u64, QuillIndexError> {
    u64::try_from(value).map_err(|_| invalid_state(format!("{label} does not fit u64")))
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_be_bytes());
    hasher.update(bytes);
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_hash_optional_u64(hasher: &mut Sha256, value: Option<u64>) {
    match value {
        None => hasher.update([0]),
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_be_bytes());
        }
    }
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_hash_optional_lease(hasher: &mut Sha256, lease: Option<(u64, u32)>) {
    match lease {
        None => hasher.update([0]),
        Some((base, used)) => {
            hasher.update([1]);
            hasher.update(base.to_be_bytes());
            hasher.update(used.to_be_bytes());
        }
    }
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_manifest_bytes(
    manifest: &Manifest,
    label: &'static str,
) -> Result<Vec<u8>, QuillIndexError> {
    manifest
        .to_bytes()
        .map_err(|error| invalid_state(format!("could not encode {label} receipt: {error}")))
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_hash_manifest_segment(hasher: &mut Sha256, segment: &ManifestSegment) {
    hasher.update(segment.segment_id.to_be_bytes());
    hasher.update(segment.seal_seq.to_be_bytes());
    hasher.update(segment.file_len.to_be_bytes());
    hasher.update(segment.file_xxh3.to_be_bytes());
    hasher.update(segment.docid_lo.to_be_bytes());
    hasher.update(segment.docid_hi.to_be_bytes());
    hasher.update(segment.doc_count.to_be_bytes());
    conformance_hash_bytes(hasher, segment.tombstones.as_bytes());
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_hash_field_stats(
    hasher: &mut Sha256,
    field_stats: &BTreeMap<u16, (u64, u32)>,
) -> Result<(), QuillIndexError> {
    hasher.update(
        conformance_usize_to_u64(field_stats.len(), "pending field-stat count")?.to_be_bytes(),
    );
    for (&field_ord, &(total_tokens, document_count)) in field_stats {
        hasher.update(field_ord.to_be_bytes());
        hasher.update(total_tokens.to_be_bytes());
        hasher.update(document_count.to_be_bytes());
    }
    Ok(())
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_hash_optional_manifest(
    hasher: &mut Sha256,
    manifest: Option<&Manifest>,
    label: &'static str,
) -> Result<(), QuillIndexError> {
    match manifest {
        None => hasher.update([0]),
        Some(manifest) => {
            hasher.update([1]);
            conformance_hash_bytes(hasher, &conformance_manifest_bytes(manifest, label)?);
        }
    }
    Ok(())
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_hash_optional_delta_seal(
    hasher: &mut Sha256,
    pending: Option<&PendingDeltaSeal>,
) -> Result<(), QuillIndexError> {
    let Some(pending) = pending else {
        hasher.update([0]);
        return Ok(());
    };
    hasher.update([1]);
    match &pending.encoded {
        None => hasher.update([0]),
        Some(encoded) => {
            hasher.update([1]);
            conformance_hash_bytes(hasher, encoded.as_bytes());
        }
    }
    hasher.update([u8::from(pending.segment_installed)]);
    conformance_hash_bytes(
        hasher,
        &conformance_manifest_bytes(&pending.manifest, "pending Delta MANIFEST")?,
    );
    hasher.update(pending.next_seal_seq.to_be_bytes());
    hasher.update(pending.successor_watermark.to_be_bytes());

    let prepared = &pending.prepared;
    hasher.update(prepared.snapshot_epoch.to_be_bytes());
    hasher.update(prepared.expected_keeper_generation.to_be_bytes());
    hasher.update(prepared.expected_schema_id.to_be_bytes());
    hasher.update(prepared.expected_docid_high_watermark.to_be_bytes());
    conformance_hash_bytes(hasher, &prepared.schema.canonical_encoding()?);
    hasher.update(
        conformance_usize_to_u64(prepared.field_stats.len(), "prepared field-stat count")?
            .to_be_bytes(),
    );
    for field_stats in &prepared.field_stats {
        hasher.update(field_stats.field_ord.to_be_bytes());
        hasher.update(field_stats.total_tokens.to_be_bytes());
        hasher.update(field_stats.doc_count.to_be_bytes());
    }
    hasher.update(prepared.bm25_doc_count.to_be_bytes());
    hasher.update(prepared.live_doc_count.to_be_bytes());
    hasher.update(prepared.delta_live_doc_count.to_be_bytes());
    hasher.update(
        conformance_usize_to_u64(prepared.deltas.len(), "prepared Delta count")?.to_be_bytes(),
    );
    for delta in &prepared.deltas {
        hasher.update(delta.keeper_generation().to_be_bytes());
        let (lineage_id, generation, lease_base, lease_end) = delta.publication_lineage();
        hasher.update(lineage_id.to_be_bytes());
        hasher.update(generation.to_be_bytes());
        hasher.update(lease_base.to_be_bytes());
        hasher.update(lease_end.to_be_bytes());
        hasher.update(
            conformance_usize_to_u64(delta.live_document_count(), "prepared Delta live rows")?
                .to_be_bytes(),
        );
        conformance_hash_bytes(hasher, &delta.conformance_content_sha256()?);
    }
    Ok(())
}
