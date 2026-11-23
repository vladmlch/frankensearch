use crate::argus::QueryWorkKind;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::error_results::{QuillIndexError, QuillSearchResult, invalid_state};

/// Cache disposition observed by one invocation-local diagnostic profile.
///
/// This type is deliberately unavailable to ordinary and benchmark builds.
/// Profiled observations must be collected by a separately built sidecar ELF,
/// never by the timing executable.
#[cfg(feature = "profile-internals")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuillProfileCacheDisposition {
    /// The invocation returned before a cache lookup could occur.
    NotChecked,
    /// The ranked query cache supplied the ordinary result.
    Hit,
    /// The ranked query cache was checked but had no matching result.
    Miss,
    /// The ranked query cache was intentionally unavailable for this invocation.
    Disabled,
}

/// Shipping sealed-segment branch observed by one diagnostic invocation.
#[cfg(feature = "profile-internals")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuillProfileExecutionMode {
    /// Sealed segments were collected in snapshot order.
    Serial,
    /// Sealed segments were collected through the shipping Rayon fan-out path.
    Rayon,
}

/// Terminal outcome recorded by an invocation-local diagnostic profile.
#[cfg(feature = "profile-internals")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuillProfileOutcome {
    /// The ordinary query returned its normal result.
    Completed,
    /// The ordinary query observed cancellation.
    Cancelled,
    /// The ordinary query exhausted its shipping fuel budget.
    FuelExhausted,
    /// The ordinary query ended with another typed error.
    OtherError,
}

/// Per-kind query-work observations from the ordinary checkpoint path.
///
/// Each tuple is ordered as segment, dictionary block, posting block, and
/// position document. Requested units reached the checkpoint; admitted units
/// were allowed to enter shipping work; refused units exceeded the shipping
/// fuel budget. Cancellation is reported separately on [`QuillProfileReceipt`].
#[cfg(feature = "profile-internals")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuillProfileWorkUnits {
    requested: [u64; 4],
    admitted: [u64; 4],
    refused: [u64; 4],
}

#[cfg(feature = "profile-internals")]
impl QuillProfileWorkUnits {
    /// Requested units by kind: segment, dictionary, posting, position.
    #[must_use]
    pub const fn requested(&self) -> [u64; 4] {
        self.requested
    }

    /// Admitted units by kind: segment, dictionary, posting, position.
    #[must_use]
    pub const fn admitted(&self) -> [u64; 4] {
        self.admitted
    }

    /// Fuel-refused units by kind: segment, dictionary, posting, position.
    #[must_use]
    pub const fn refused(&self) -> [u64; 4] {
        self.refused
    }
}

/// Immutable counters and single-assignment facts from one diagnostic query.
#[cfg(feature = "profile-internals")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuillProfileReceipt {
    snapshot_epoch: u64,
    keeper_generation: u64,
    sealed_segments: u64,
    delta_segments: u64,
    cache: QuillProfileCacheDisposition,
    fanout_eligible: Option<bool>,
    execution: Option<QuillProfileExecutionMode>,
    work_plan: Option<(u64, bool)>,
    snapshot_doc_freq_calls: u64,
    global_doc_freq_probes: u64,
    term_dictionary_views: u64,
    segments_lowered: u64,
    fuel_units: u64,
    work_units: QuillProfileWorkUnits,
    cancellation_observations: u64,
    overflowed: bool,
    outcome: QuillProfileOutcome,
}

/// Ordinary search outcome paired with its invocation-bound profile receipt.
#[cfg(feature = "profile-internals")]
#[doc(hidden)]
#[derive(Debug)]
pub enum QuillProfiledSearchOutcome {
    /// The ordinary search completed normally.
    Completed {
        /// Ordinary result from the profiled invocation.
        result: QuillSearchResult,
        /// Immutable receipt from that same invocation.
        receipt: QuillProfileReceipt,
    },
    /// The ordinary search returned a typed failure after profile admission.
    Failed {
        /// Original ordinary-search error.
        error: QuillIndexError,
        /// Immutable receipt from that same invocation.
        receipt: QuillProfileReceipt,
    },
}

#[cfg(feature = "profile-internals")]
impl QuillProfileReceipt {
    /// Snapshot epoch bound to the profiled ordinary invocation.
    #[must_use]
    pub const fn snapshot_epoch(&self) -> u64 {
        self.snapshot_epoch
    }

    /// Retained Keeper generation bound to the profiled ordinary invocation.
    #[must_use]
    pub const fn keeper_generation(&self) -> u64 {
        self.keeper_generation
    }

    /// Sealed and Delta segment counts observed at query admission.
    #[must_use]
    pub const fn segment_counts(&self) -> (u64, u64) {
        (self.sealed_segments, self.delta_segments)
    }

    /// Cache disposition, including the explicit no-lookup state.
    #[must_use]
    pub const fn cache(&self) -> QuillProfileCacheDisposition {
        self.cache
    }

    /// Whether the sealed snapshot met the shipping Rayon fan-out shape gate.
    ///
    /// Cache-served and pre-planning failures have no fan-out observation.
    /// This records eligibility independently from the selected execution
    /// branch, which can still be serial when shipping fuel metering is active.
    #[must_use]
    pub const fn fanout_eligible(&self) -> Option<bool> {
        self.fanout_eligible
    }

    /// Actual sealed-segment execution branch, when sealed work was reached.
    #[must_use]
    pub const fn execution(&self) -> Option<QuillProfileExecutionMode> {
        self.execution
    }

    /// Computed work upper bound and whether the shipping fuel meter was active.
    ///
    /// Cache-served and pre-planning failures have no work-plan observation.
    #[must_use]
    pub const fn work_plan(&self) -> Option<(u64, bool)> {
        self.work_plan
    }

    /// Counter tuple in the order: snapshot DF, global DF, TERMDICT views,
    /// lowered segments, and admitted fuel units.
    #[must_use]
    pub const fn counters(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.snapshot_doc_freq_calls,
            self.global_doc_freq_probes,
            self.term_dictionary_views,
            self.segments_lowered,
            self.fuel_units,
        )
    }

    /// Requested, admitted, and fuel-refused work units by checkpoint kind.
    #[must_use]
    pub const fn work_units(&self) -> QuillProfileWorkUnits {
        self.work_units
    }

    /// Number of cancellation checks that observed an already-cancelled Cx.
    #[must_use]
    pub const fn cancellation_observations(&self) -> u64 {
        self.cancellation_observations
    }

    /// Whether a counter saturated before the receipt could be finalized.
    #[must_use]
    pub const fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Terminal ordinary-query outcome.
    #[must_use]
    pub const fn outcome(&self) -> QuillProfileOutcome {
        self.outcome
    }
}

/// Invocation-local sidecar session for Quill diagnostic instrumentation.
///
/// Counters use relaxed increments because the caller only finalizes after all
/// worker joins; finalization reads with acquire ordering and rejects any
/// repeated or contradictory single-assignment fact. The session has no global
/// reset path and is therefore safe to keep beside an ordinary invocation.
#[cfg(feature = "profile-internals")]
#[doc(hidden)]
#[derive(Debug)]
pub struct QuillProfileSession {
    pub(super) snapshot_epoch: u64,
    pub(super) keeper_generation: u64,
    pub(super) sealed_segments: u64,
    pub(super) delta_segments: u64,
    pub(super) snapshot_doc_freq_calls: AtomicU64,
    pub(super) global_doc_freq_probes: AtomicU64,
    pub(super) term_dictionary_views: AtomicU64,
    pub(super) segments_lowered: AtomicU64,
    fuel_units: AtomicU64,
    work_requested: [AtomicU64; 4],
    work_admitted: [AtomicU64; 4],
    work_refused: [AtomicU64; 4],
    pub(super) cancellation_observations: AtomicU64,
    pub(super) overflowed: AtomicBool,
    pub(super) state: StdMutex<QuillProfileSessionState>,
}

#[cfg(feature = "profile-internals")]
#[derive(Debug)]
pub(super) struct QuillProfileSessionState {
    pub(super) cache: Option<QuillProfileCacheDisposition>,
    pub(super) fanout_eligible: Option<bool>,
    pub(super) execution: Option<QuillProfileExecutionMode>,
    pub(super) work_plan: Option<(u64, bool)>,
    pub(super) completed: bool,
}

#[cfg(feature = "profile-internals")]
impl QuillProfileSession {
    /// Begin a sidecar receipt bound to one already-admitted snapshot.
    #[must_use]
    pub fn new(
        snapshot_epoch: u64,
        keeper_generation: u64,
        sealed_segments: u64,
        delta_segments: u64,
    ) -> Self {
        Self {
            snapshot_epoch,
            keeper_generation,
            sealed_segments,
            delta_segments,
            snapshot_doc_freq_calls: AtomicU64::new(0),
            global_doc_freq_probes: AtomicU64::new(0),
            term_dictionary_views: AtomicU64::new(0),
            segments_lowered: AtomicU64::new(0),
            fuel_units: AtomicU64::new(0),
            work_requested: std::array::from_fn(|_| AtomicU64::new(0)),
            work_admitted: std::array::from_fn(|_| AtomicU64::new(0)),
            work_refused: std::array::from_fn(|_| AtomicU64::new(0)),
            cancellation_observations: AtomicU64::new(0),
            overflowed: AtomicBool::new(false),
            state: StdMutex::new(QuillProfileSessionState {
                cache: None,
                fanout_eligible: None,
                execution: None,
                work_plan: None,
                completed: false,
            }),
        }
    }

    /// Bind one cache disposition. Conflicting or repeated binding fails closed.
    ///
    /// # Errors
    ///
    /// Returns a typed invalid-state error if the session was finalized, the
    /// cache disposition was already bound, or the session lock is poisoned.
    pub fn bind_cache(&self, cache: QuillProfileCacheDisposition) -> Result<(), QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("profile-internals session lock is poisoned"))?;
        if state.completed || state.cache.is_some() {
            return Err(invalid_state(
                "profile-internals cache disposition was already bound",
            ));
        }
        state.cache = Some(cache);
        drop(state);
        Ok(())
    }

    /// Bind one actual sealed-segment execution branch.
    ///
    /// # Errors
    ///
    /// Returns a typed invalid-state error if the session was finalized, the
    /// execution branch was already bound, or the session lock is poisoned.
    pub fn bind_execution(
        &self,
        execution: QuillProfileExecutionMode,
    ) -> Result<(), QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("profile-internals session lock is poisoned"))?;
        if state.completed || state.execution.is_some() {
            return Err(invalid_state(
                "profile-internals execution mode was already bound",
            ));
        }
        state.execution = Some(execution);
        drop(state);
        Ok(())
    }

    /// Bind whether the ordinary sealed snapshot met the Rayon shape gate.
    ///
    /// # Errors
    ///
    /// Returns a typed invalid-state error if the session was finalized, the
    /// eligibility fact was already bound, or the session lock is poisoned.
    pub fn bind_fanout_eligibility(&self, eligible: bool) -> Result<(), QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("profile-internals session lock is poisoned"))?;
        if state.completed || state.fanout_eligible.is_some() {
            return Err(invalid_state(
                "profile-internals fan-out eligibility was already bound",
            ));
        }
        state.fanout_eligible = Some(eligible);
        drop(state);
        Ok(())
    }

    /// Bind the exact work-plan facts computed by the ordinary query path.
    ///
    /// # Errors
    ///
    /// Returns a typed invalid-state error if planning was already bound, the
    /// session was finalized, or the session lock is poisoned.
    pub fn bind_work_plan(&self, upper_bound: u64, metering: bool) -> Result<(), QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("profile-internals session lock is poisoned"))?;
        if state.completed || state.work_plan.is_some() {
            return Err(invalid_state(
                "profile-internals work plan was already bound",
            ));
        }
        state.work_plan = Some((upper_bound, metering));
        drop(state);
        Ok(())
    }

    /// Record one or more snapshot document-frequency calls.
    pub fn record_snapshot_doc_freq(&self, units: u64) {
        self.record_counter(&self.snapshot_doc_freq_calls, units);
    }

    /// Record one or more global document-frequency probes.
    pub fn record_global_doc_freq(&self, units: u64) {
        self.record_counter(&self.global_doc_freq_probes, units);
    }

    /// Record one or more borrowed TERMDICT views.
    pub fn record_term_dictionary_view(&self, units: u64) {
        self.record_counter(&self.term_dictionary_views, units);
    }

    /// Record one or more lowered sealed or Delta segments.
    pub fn record_segment_lowered(&self, units: u64) {
        self.record_counter(&self.segments_lowered, units);
    }

    /// Record admitted shipping fuel units, including unmetered observations.
    pub fn record_fuel_units(&self, units: u64) {
        self.record_counter(&self.fuel_units, units);
    }

    /// Record one cancellation check that observed cancellation.
    pub fn record_cancellation_observation(&self) {
        self.record_counter(&self.cancellation_observations, 1);
    }

    pub(super) fn record_work_requested(&self, kind: QueryWorkKind, units: u64) {
        self.record_work_counter(&self.work_requested, kind, units);
    }

    pub(super) fn record_work_admitted(&self, kind: QueryWorkKind, units: u64) {
        self.record_work_counter(&self.work_admitted, kind, units);
    }

    pub(super) fn record_work_refused(&self, kind: QueryWorkKind, units: u64) {
        self.record_work_counter(&self.work_refused, kind, units);
    }

    fn record_work_counter(&self, counters: &[AtomicU64; 4], kind: QueryWorkKind, units: u64) {
        let index = match kind {
            QueryWorkKind::Segment => 0,
            QueryWorkKind::DictionaryBlock => 1,
            QueryWorkKind::PostingBlock => 2,
            QueryWorkKind::PositionDocument => 3,
        };
        self.record_counter(&counters[index], units);
    }

    fn record_counter(&self, counter: &AtomicU64, units: u64) {
        if units == 0 {
            return;
        }
        if counter
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(units)
            })
            .is_err()
        {
            self.overflowed.store(true, Ordering::Release);
        }
    }

    /// Finalize the session once all worker joins have completed.
    ///
    /// A profile with an unset cache state, an overflow, or a second finalizer
    /// is rejected rather than becoming a partial timing artifact.
    ///
    /// # Errors
    ///
    /// Returns a typed invalid-state error for a repeated finalizer, missing
    /// cache disposition, poisoned session lock, or counter overflow.
    pub fn complete(
        &self,
        outcome: QuillProfileOutcome,
    ) -> Result<QuillProfileReceipt, QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("profile-internals session lock is poisoned"))?;
        if state.completed {
            return Err(invalid_state(
                "profile-internals session was already completed",
            ));
        }
        state.completed = true;
        let cache = state
            .cache
            .ok_or_else(|| invalid_state("profile-internals cache disposition was never bound"))?;
        let overflowed = self.overflowed.load(Ordering::Acquire);
        if overflowed {
            return Err(invalid_state(
                "profile-internals counter overflowed before finalization",
            ));
        }
        Ok(QuillProfileReceipt {
            snapshot_epoch: self.snapshot_epoch,
            keeper_generation: self.keeper_generation,
            sealed_segments: self.sealed_segments,
            delta_segments: self.delta_segments,
            cache,
            fanout_eligible: state.fanout_eligible,
            execution: state.execution,
            work_plan: state.work_plan,
            snapshot_doc_freq_calls: self.snapshot_doc_freq_calls.load(Ordering::Acquire),
            global_doc_freq_probes: self.global_doc_freq_probes.load(Ordering::Acquire),
            term_dictionary_views: self.term_dictionary_views.load(Ordering::Acquire),
            segments_lowered: self.segments_lowered.load(Ordering::Acquire),
            fuel_units: self.fuel_units.load(Ordering::Acquire),
            work_units: QuillProfileWorkUnits {
                requested: self
                    .work_requested
                    .each_ref()
                    .map(|counter| counter.load(Ordering::Acquire)),
                admitted: self
                    .work_admitted
                    .each_ref()
                    .map(|counter| counter.load(Ordering::Acquire)),
                refused: self
                    .work_refused
                    .each_ref()
                    .map(|counter| counter.load(Ordering::Acquire)),
            },
            cancellation_observations: self.cancellation_observations.load(Ordering::Acquire),
            overflowed,
            outcome,
        })
    }
}
