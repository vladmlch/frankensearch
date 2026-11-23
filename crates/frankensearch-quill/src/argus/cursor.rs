use std::sync::Arc;

use super::segment_cursors::SealedPostingCursor;
use super::term_scorer::TermRecordOption;
use super::{ArgusError, PositionsHandle, QueryWorkCheckpoint, QueryWorkKind};

/// Unforgeable evidence that a cursor's backing codec validates each
/// post-move scorer invariant.
///
/// The private field prevents custom [`PostingCursor`] implementations from
/// claiming this contract. Only the sealed Quiver adapter and wrappers that
/// preserve its exact cursor semantics may return this token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatedPostMoveContract(pub(in crate::argus) ());

/// The concrete, non-virtual path a doc-major union reads arrival predecessors
/// through.
///
/// Every previous attempt to seal this capability sealed the wrong thing. A
/// capability token could be forwarded by a wrapper that did not implement the
/// operation; a sealed *result* could be replayed; binding a result to the
/// document it answered still left a wrapper free to probe one cursor while the
/// engine drove another, producing receipts that were genuine, correctly minted
/// and correctly targeted, and described the wrong posting list.
///
/// The common flaw is that all of them asked a `dyn PostingCursor` — an
/// overridable surface — for evidence about itself. There is no arrival method
/// on that trait any more. The read exists only as an inherent method on this
/// concrete type, and a term scorer either holds one of these or does not: a
/// wrapper cannot become one, cannot interpose on one, and cannot hand the
/// engine a second cursor to probe, because the value the engine probes is the
/// same value it advances and scores.
pub struct ValidatedArrivalCursor<'a> {
    sealed: SealedPostingCursor<'a>,
    checkpoint: PostingCheckpoint<'a>,
    refused: Option<RefusedAdmission>,
    /// Work this cursor really performed, when a test asked to observe it.
    ///
    /// The counter lives here rather than in a wrapper because a wrapper is an
    /// opaque cursor by construction: instrumenting from outside would move the
    /// clause off the validated arm and measure the walk instead of the seek.
    #[cfg(test)]
    work: Option<std::rc::Rc<CursorWork>>,
}

/// Movement a scoring cursor performed, counted at the cursor itself.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct CursorWork {
    pub steps: std::cell::Cell<u64>,
    pub seeks: std::cell::Cell<u64>,
    pub probes: std::cell::Cell<u64>,
}

impl<'a> ValidatedArrivalCursor<'a> {
    /// Bind a validated sealed cursor to the checkpoint that meters it.
    pub fn new(
        sealed: SealedPostingCursor<'a>,
        checkpoint: Arc<dyn QueryWorkCheckpoint + 'a>,
    ) -> Self {
        Self {
            sealed,
            checkpoint: PostingCheckpoint::Shared(checkpoint),
            refused: None,
            #[cfg(test)]
            work: None,
        }
    }

    /// The same cursor, reporting what it does to `work`.
    #[cfg(test)]
    pub fn new_counted(
        sealed: SealedPostingCursor<'a>,
        checkpoint: Arc<dyn QueryWorkCheckpoint + 'a>,
        work: std::rc::Rc<CursorWork>,
    ) -> Self {
        Self {
            sealed,
            checkpoint: PostingCheckpoint::Shared(checkpoint),
            refused: None,
            work: Some(work),
        }
    }

    /// Blocks the underlying sealed cursor has physically decoded.
    #[cfg(test)]
    pub fn decoded_blocks(&self) -> usize {
        self.sealed.sealed_decoded_blocks()
    }

    #[cfg(test)]
    fn record(&self, select: fn(&CursorWork) -> &std::cell::Cell<u64>) {
        if let Some(work) = &self.work {
            let counter = select(work);
            counter.set(counter.get() + 1);
        }
    }

    fn guard_refused(&self) -> Result<(), ArgusError> {
        self.refused.map_or(Ok(()), |refusal| Err(refusal.error()))
    }

    fn admit_or_refuse(&mut self, units: u64) -> Result<(), ArgusError> {
        match self.checkpoint.admit(QueryWorkKind::PostingBlock, units) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.refused = RefusedAdmission::capture(&error);
                Err(error)
            }
        }
    }

    /// Inclusive last document this term can still reach.
    pub fn arrival_final_doc(&self) -> Option<u32> {
        self.sealed.sealed_final_doc()
    }

    /// Greatest document this term holds strictly below `target`.
    ///
    /// # Errors
    ///
    /// Propagates the checkpoint's refusal, which is latched here exactly as a
    /// move latches it, and any typed storage failure.
    pub fn arrival_doc_before(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.guard_refused()?;
        #[cfg(test)]
        self.record(|work| &work.probes);
        let permit = self.sealed.sealed_arrival_permit(target);
        if permit > 0 {
            self.admit_or_refuse(permit)?;
        }
        match self.sealed.sealed_doc_before(target) {
            Ok(predecessor) => Ok(predecessor),
            Err(error) => {
                self.refused = RefusedAdmission::capture(&error);
                Err(error)
            }
        }
    }
}

impl PostingCursor for ValidatedArrivalCursor<'_> {
    fn doc(&self) -> Option<u32> {
        self.sealed.doc()
    }

    fn freq(&self) -> Option<u32> {
        self.sealed.freq()
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.sealed.positions_handle()
    }

    fn size_hint(&self) -> u32 {
        self.sealed.size_hint()
    }

    fn cost(&self) -> u64 {
        self.sealed.cost()
    }

    fn segment_num_docs(&self) -> u32 {
        self.sealed.segment_num_docs()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.guard_refused()?;
        #[cfg(test)]
        self.record(|work| &work.steps);
        self.admit_or_refuse(0)?;
        let permit = self.sealed.block_permit_for_next();
        if permit > 0 {
            self.admit_or_refuse(permit)?;
        }
        match self.sealed.next() {
            Ok(moved) => Ok(moved),
            Err(error) => {
                self.refused = RefusedAdmission::capture(&error);
                Err(error)
            }
        }
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.guard_refused()?;
        #[cfg(test)]
        self.record(|work| &work.seeks);
        self.admit_or_refuse(0)?;
        let permit = self.sealed.block_permit_for_advance(target);
        if permit > 0 {
            self.admit_or_refuse(permit)?;
        }
        match self.sealed.advance(target) {
            Ok(moved) => Ok(moved),
            Err(error) => {
                self.refused = RefusedAdmission::capture(&error);
                Err(error)
            }
        }
    }

    fn validated_post_move_contract(&self) -> Option<ValidatedPostMoveContract> {
        self.sealed.validated_post_move_contract()
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        let inner = self.sealed.fork_for_pruning()?;
        Some(Box::new(CheckpointPostingCursor {
            inner,
            checkpoint: PostingCheckpoint::Borrowed(self.checkpoint.as_borrowed()),
            refused: None,
        }))
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        self.sealed
            .term_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn supports_block_max(&self) -> bool {
        self.sealed.supports_block_max()
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        self.sealed
            .current_block_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        self.sealed.current_block_last_doc()
    }

    fn current_work_block(&self) -> Option<u64> {
        self.sealed.current_work_block()
    }

    fn block_permit_for_next(&self) -> u64 {
        self.sealed.block_permit_for_next()
    }

    fn block_permit_for_advance(&self, target: u32) -> u64 {
        self.sealed.block_permit_for_advance(target)
    }
}

/// The cursor one term scorer drives.
///
/// This enum is the whole of the exact-cursor guarantee. `Validated` owns the
/// crate's own cursor as a concrete type, so movement, scoring and arrival
/// reads are all served by one instance with no virtual call in between;
/// `Opaque` is every other implementation, including every wrapper, and has no
/// arrival read at all. A wrapper cannot move itself into `Validated` because
/// the constructor that builds that arm takes the concrete type.
///
/// The validated arm is boxed. `ValidatedArrivalCursor` embeds a whole decoded
/// posting block, so holding it inline made every move of a `TermCursor` — and
/// therefore of the `TermScorer` and `ScorerNode` that carry it — copy better
/// than a kilobyte to carry an arm that is otherwise one pointer wide. The box
/// is an owning, statically dispatched pointer to the concrete type: dispatch
/// stays direct, and the arm still cannot be reached with anything but the
/// crate's own cursor.
pub(in crate::argus) enum TermCursor<'a> {
    Validated(Box<ValidatedArrivalCursor<'a>>),
    Opaque(Box<dyn PostingCursor + 'a>),
}

impl TermCursor<'_> {
    /// Inclusive last document, on the validated arm only.
    pub(in crate::argus) fn arrival_final_doc(&self) -> Option<u32> {
        match self {
            Self::Validated(cursor) => cursor.arrival_final_doc(),
            Self::Opaque(_) => None,
        }
    }

    /// Greatest document strictly below `target`, on the validated arm only.
    ///
    /// # Errors
    ///
    /// Refuses for an opaque cursor — there is no way for one to answer, and
    /// answering `None` would be indistinguishable from the legitimate "this
    /// term holds nothing below the probe".
    pub(in crate::argus) fn arrival_doc_before(
        &mut self,
        target: u32,
    ) -> Result<Option<u32>, ArgusError> {
        match self {
            Self::Validated(cursor) => cursor.arrival_doc_before(target),
            Self::Opaque(_) => Err(ArgusError::CursorInvariant(
                "arrival predecessor read from a cursor outside the validated path",
            )),
        }
    }
}

impl PostingCursor for TermCursor<'_> {
    fn doc(&self) -> Option<u32> {
        match self {
            Self::Validated(cursor) => cursor.doc(),
            Self::Opaque(cursor) => cursor.doc(),
        }
    }

    fn freq(&self) -> Option<u32> {
        match self {
            Self::Validated(cursor) => cursor.freq(),
            Self::Opaque(cursor) => cursor.freq(),
        }
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        match self {
            Self::Validated(cursor) => cursor.positions_handle(),
            Self::Opaque(cursor) => cursor.positions_handle(),
        }
    }

    fn size_hint(&self) -> u32 {
        match self {
            Self::Validated(cursor) => cursor.size_hint(),
            Self::Opaque(cursor) => cursor.size_hint(),
        }
    }

    fn cost(&self) -> u64 {
        match self {
            Self::Validated(cursor) => cursor.cost(),
            Self::Opaque(cursor) => cursor.cost(),
        }
    }

    fn segment_num_docs(&self) -> u32 {
        match self {
            Self::Validated(cursor) => cursor.segment_num_docs(),
            Self::Opaque(cursor) => cursor.segment_num_docs(),
        }
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match self {
            Self::Validated(cursor) => cursor.next(),
            Self::Opaque(cursor) => cursor.next(),
        }
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        match self {
            Self::Validated(cursor) => cursor.advance(target),
            Self::Opaque(cursor) => cursor.advance(target),
        }
    }

    fn validated_post_move_contract(&self) -> Option<ValidatedPostMoveContract> {
        match self {
            Self::Validated(cursor) => cursor.validated_post_move_contract(),
            Self::Opaque(cursor) => cursor.validated_post_move_contract(),
        }
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        match self {
            Self::Validated(cursor) => cursor.fork_for_pruning(),
            Self::Opaque(cursor) => cursor.fork_for_pruning(),
        }
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        match self {
            Self::Validated(cursor) => {
                cursor.term_score_upper_bound(live_avgdl, weight, record_option)
            }
            Self::Opaque(cursor) => {
                cursor.term_score_upper_bound(live_avgdl, weight, record_option)
            }
        }
    }

    fn supports_block_max(&self) -> bool {
        match self {
            Self::Validated(cursor) => cursor.supports_block_max(),
            Self::Opaque(cursor) => cursor.supports_block_max(),
        }
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        match self {
            Self::Validated(cursor) => {
                cursor.current_block_score_upper_bound(live_avgdl, weight, record_option)
            }
            Self::Opaque(cursor) => {
                cursor.current_block_score_upper_bound(live_avgdl, weight, record_option)
            }
        }
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        match self {
            Self::Validated(cursor) => cursor.current_block_last_doc(),
            Self::Opaque(cursor) => cursor.current_block_last_doc(),
        }
    }

    fn current_work_block(&self) -> Option<u64> {
        match self {
            Self::Validated(cursor) => cursor.current_work_block(),
            Self::Opaque(cursor) => cursor.current_work_block(),
        }
    }

    fn block_permit_for_next(&self) -> u64 {
        match self {
            Self::Validated(cursor) => cursor.block_permit_for_next(),
            Self::Opaque(cursor) => cursor.block_permit_for_next(),
        }
    }

    fn block_permit_for_advance(&self, target: u32) -> u64 {
        match self {
            Self::Validated(cursor) => cursor.block_permit_for_advance(target),
            Self::Opaque(cursor) => cursor.block_permit_for_advance(target),
        }
    }

    fn next_admitted(
        &mut self,
        checkpoint: &dyn QueryWorkCheckpoint,
    ) -> Result<Option<u32>, ArgusError> {
        match self {
            Self::Validated(cursor) => cursor.next_admitted(checkpoint),
            Self::Opaque(cursor) => cursor.next_admitted(checkpoint),
        }
    }

    fn advance_admitted(
        &mut self,
        checkpoint: &dyn QueryWorkCheckpoint,
        target: u32,
    ) -> Result<Option<u32>, ArgusError> {
        match self {
            Self::Validated(cursor) => cursor.advance_admitted(checkpoint, target),
            Self::Opaque(cursor) => cursor.advance_admitted(checkpoint, target),
        }
    }
}

/// Forward-only posting access shared by sealed and future delta segments.
///
/// A cursor starts on its first posting. `advance` is an inclusive lower-bound
/// seek and does not move when the current document already satisfies the
/// target. Exhaustion is fused and represented by `None`.
pub trait PostingCursor {
    /// Return the current global document id, or `None` after exhaustion.
    fn doc(&self) -> Option<u32>;

    /// Return the current term frequency, or `None` after exhaustion.
    fn freq(&self) -> Option<u32>;

    /// Return the lazy POSITIONS locator for the current posting, when present.
    fn positions_handle(&self) -> Option<PositionsHandle<'_>>;

    /// Best-effort number of documents matched by this cursor.
    fn size_hint(&self) -> u32;

    /// Relative cost to drive this cursor to exhaustion.
    fn cost(&self) -> u64 {
        u64::from(self.size_hint())
    }

    /// Live document count used by Tantivy's composite size estimators.
    fn segment_num_docs(&self) -> u32;

    /// Move strictly forward by one posting.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the cursor cannot preserve its validated
    /// storage invariants.
    fn next(&mut self) -> Result<Option<u32>, ArgusError>;

    /// Land on the first posting whose document id is at least `target`.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the cursor cannot preserve its validated
    /// storage invariants.
    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError>;

    /// Return evidence that this cursor's backing storage validates every
    /// post-move invariant required by a term scorer.
    ///
    /// The default keeps defensive post-move validation enabled for custom
    /// and Delta cursors. The sealed Quiver adapter may opt in only because
    /// its validated codec guarantees ordered document ids and positive
    /// frequencies for every decoded posting. The marker is unforgeable by a
    /// custom cursor implementation.
    fn validated_post_move_contract(&self) -> Option<ValidatedPostMoveContract> {
        None
    }

    /// Fork the current position into an independent cursor suitable for
    /// competitive candidate generation.
    ///
    /// The default opts out. A cursor must return `Some` only when the fork
    /// retains sound live-snapshot impact metadata for every remaining row.
    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        None
    }

    /// Conservative score ceiling over every remaining block in this term.
    ///
    /// Implementations must apply `record_option` to the cursor's occurrence
    /// count. In particular, [`TermRecordOption::Basic`] bounds use presence
    /// frequency one even when a sealed POSTINGS row retains a raw count.
    fn term_score_upper_bound(
        &self,
        _live_avgdl: f32,
        _weight: f32,
        _record_option: TermRecordOption,
    ) -> Option<f32> {
        None
    }

    /// Whether this cursor exposes real skip-capable posting blocks.
    ///
    /// A whole-term ceiling is sufficient for `MaxScore` but not for BMW. The
    /// default is deliberately false so a custom cursor with only a term bound
    /// falls back to exhaustive collection instead of surfacing an invariant
    /// error when the clause-count threshold selects BMW.
    fn supports_block_max(&self) -> bool {
        false
    }

    /// Conservative score ceiling for the cursor's current posting block.
    fn current_block_score_upper_bound(
        &self,
        _live_avgdl: f32,
        _weight: f32,
        _record_option: TermRecordOption,
    ) -> Option<f32> {
        None
    }

    /// Inclusive validated last docid of the current posting block.
    fn current_block_last_doc(&self) -> Option<u32> {
        None
    }

    /// Stable zero-based work-block ordinal for coarse query checkpoints.
    ///
    /// Sealed cursors report their validated POSTINGS block. Delta cursors
    /// report their logical 128-posting block. Cursors whose postings were
    /// already charged while materializing may retain the default `None`.
    fn current_work_block(&self) -> Option<u64> {
        None
    }

    /// Blocks [`Self::next`] would decode, determined without decoding.
    ///
    /// This is the permit a caller must admit *before* the move, so it has to
    /// mirror the move exactly: too low lets a decode run unadmitted, too
    /// high charges for work never done. The default reports zero, which is
    /// correct for cursors that decode nothing and for cursors that admit
    /// their own work internally through [`Self::next_admitted`].
    fn block_permit_for_next(&self) -> u64 {
        0
    }

    /// Blocks [`Self::advance`] would decode, determined without decoding.
    fn block_permit_for_advance(&self, _target: u32) -> u64 {
        0
    }

    /// Step one posting, admitting any work this cursor performs internally.
    ///
    /// The default is for cursors whose whole cost is the block permit their
    /// caller already admitted. A cursor that can perform an unbounded amount
    /// of work inside one move — a Delta cursor walking a tombstoned prefix —
    /// overrides this to admit as it goes, so cancellation is observed during
    /// the walk instead of after it.
    ///
    /// # Errors
    ///
    /// Propagates the checkpoint's refusal or a typed cursor failure.
    fn next_admitted(
        &mut self,
        _checkpoint: &dyn QueryWorkCheckpoint,
    ) -> Result<Option<u32>, ArgusError> {
        self.next()
    }

    /// Seek, admitting any work this cursor performs internally.
    ///
    /// # Errors
    ///
    /// Propagates the checkpoint's refusal or a typed cursor failure.
    fn advance_admitted(
        &mut self,
        _checkpoint: &dyn QueryWorkCheckpoint,
        target: u32,
    ) -> Result<Option<u32>, ArgusError> {
        self.advance(target)
    }
}

impl<T> PostingCursor for Box<T>
where
    T: PostingCursor + ?Sized,
{
    fn doc(&self) -> Option<u32> {
        (**self).doc()
    }

    fn freq(&self) -> Option<u32> {
        (**self).freq()
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        (**self).positions_handle()
    }

    fn size_hint(&self) -> u32 {
        (**self).size_hint()
    }

    fn cost(&self) -> u64 {
        (**self).cost()
    }

    fn segment_num_docs(&self) -> u32 {
        (**self).segment_num_docs()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        (**self).next()
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        (**self).advance(target)
    }

    fn validated_post_move_contract(&self) -> Option<ValidatedPostMoveContract> {
        (**self).validated_post_move_contract()
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        (**self).fork_for_pruning()
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        (**self).term_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn supports_block_max(&self) -> bool {
        (**self).supports_block_max()
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        (**self).current_block_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        (**self).current_block_last_doc()
    }

    fn current_work_block(&self) -> Option<u64> {
        (**self).current_work_block()
    }

    fn block_permit_for_next(&self) -> u64 {
        (**self).block_permit_for_next()
    }

    fn block_permit_for_advance(&self, target: u32) -> u64 {
        (**self).block_permit_for_advance(target)
    }

    fn next_admitted(
        &mut self,
        checkpoint: &dyn QueryWorkCheckpoint,
    ) -> Result<Option<u32>, ArgusError> {
        (**self).next_admitted(checkpoint)
    }

    fn advance_admitted(
        &mut self,
        checkpoint: &dyn QueryWorkCheckpoint,
        target: u32,
    ) -> Result<Option<u32>, ArgusError> {
        (**self).advance_admitted(checkpoint, target)
    }
}

enum PostingCheckpoint<'a> {
    Shared(Arc<dyn QueryWorkCheckpoint + 'a>),
    Borrowed(&'a dyn QueryWorkCheckpoint),
}

impl PostingCheckpoint<'_> {
    fn admit(&self, kind: QueryWorkKind, units: u64) -> Result<(), ArgusError> {
        match self {
            Self::Shared(checkpoint) => checkpoint.admit(kind, units),
            Self::Borrowed(checkpoint) => checkpoint.admit(kind, units),
        }
    }

    fn as_borrowed(&self) -> &dyn QueryWorkCheckpoint {
        match self {
            Self::Shared(checkpoint) => checkpoint.as_ref(),
            Self::Borrowed(checkpoint) => *checkpoint,
        }
    }
}

/// Posting cursor wrapper that charges coarse fuel only for entered posting
/// blocks while observing cancellation on every move.
///
/// The two halves of that sentence are separate concerns and must stay
/// separate. Fuel is a cost model, so only a newly entered block is charged.
/// Cancellation is a liveness obligation, and
/// [`QueryWorkCheckpoint::admit`] is the only place a scoring cursor can
/// discharge it, so an in-block step still has to call it — with zero units,
/// which charges nothing.
///
/// A refusal is terminal: every later move repeats it without decoding,
/// without moving, and without admitting again, so a refused cursor charges
/// no further fuel and its reported position stays where the refusal found
/// it. This is a necessary part of the pre-work gate, not the whole of it —
/// the admission ordering itself is being reworked under
/// `bd-quill-cancel-poll-leaf-fastpath-tzztr`.
pub struct CheckpointPostingCursor<'a> {
    inner: Box<dyn PostingCursor + 'a>,
    checkpoint: PostingCheckpoint<'a>,
    refused: Option<RefusedAdmission>,
}

/// One checkpoint refusal, retained so a refused cursor can repeat it exactly
/// without ever touching the cursor it wraps again.
///
/// [`ArgusError`] is not `Clone`, but both refusal variants carry only `Copy`
/// data, so the exact error — variant and every field — is reconstructible.
/// Anything else a checkpoint returns is propagated unchanged and does not
/// make the cursor terminal, because only these two are refusals.
#[derive(Clone, Copy, Debug)]
enum RefusedAdmission {
    Cancelled {
        phase: &'static str,
    },
    FuelExhausted {
        budget: u64,
        consumed: u64,
        segments_touched: u64,
        dictionary_blocks: u64,
        posting_blocks: u64,
        position_docs: u64,
    },
}

impl RefusedAdmission {
    fn capture(error: &ArgusError) -> Option<Self> {
        match *error {
            ArgusError::QueryCancelled { phase } => Some(Self::Cancelled { phase }),
            ArgusError::QueryFuelExhausted {
                budget,
                consumed,
                segments_touched,
                dictionary_blocks,
                posting_blocks,
                position_docs,
            } => Some(Self::FuelExhausted {
                budget,
                consumed,
                segments_touched,
                dictionary_blocks,
                posting_blocks,
                position_docs,
            }),
            _ => None,
        }
    }

    const fn error(self) -> ArgusError {
        match self {
            Self::Cancelled { phase } => ArgusError::QueryCancelled { phase },
            Self::FuelExhausted {
                budget,
                consumed,
                segments_touched,
                dictionary_blocks,
                posting_blocks,
                position_docs,
            } => ArgusError::QueryFuelExhausted {
                budget,
                consumed,
                segments_touched,
                dictionary_blocks,
                posting_blocks,
                position_docs,
            },
        }
    }
}

impl<'a> CheckpointPostingCursor<'a> {
    /// Wrap a freshly opened cursor and admit its already-decoded first block.
    ///
    /// # Errors
    ///
    /// Returns a typed checkpoint error before scorer construction continues.
    pub fn new<C>(
        inner: C,
        checkpoint: Arc<dyn QueryWorkCheckpoint + 'a>,
    ) -> Result<Self, ArgusError>
    where
        C: PostingCursor + 'a,
    {
        Ok(Self {
            inner: Box::new(inner),
            checkpoint: PostingCheckpoint::Shared(checkpoint),
            refused: None,
        })
    }

    /// Admit the block a cursor's construction is about to decode.
    ///
    /// Construction decodes the first block before any wrapper exists, so the
    /// caller admits it here — *before* building the cursor — and this type no
    /// longer charges for it afterwards. `decodes_initial_block` is false for
    /// a term with no blocks, which must not be charged for a decode it never
    /// performs.
    ///
    /// # Errors
    ///
    /// Returns the checkpoint's refusal, in which case the caller must not
    /// construct the cursor at all.
    pub fn admit_initial_block(
        checkpoint: &dyn QueryWorkCheckpoint,
        decodes_initial_block: bool,
    ) -> Result<(), ArgusError> {
        checkpoint.admit(
            QueryWorkKind::PostingBlock,
            u64::from(decodes_initial_block),
        )
    }

    /// Repeat a retained refusal instead of doing any further work.
    ///
    /// A refused cursor is terminal: it decodes nothing, mutates nothing, and
    /// admits nothing again, so its position stays exactly where the refusal
    /// found it and no further fuel can be charged against a query that has
    /// already been told to stop.
    fn guard_refused(&self) -> Result<(), ArgusError> {
        self.refused.map_or(Ok(()), |refusal| Err(refusal.error()))
    }

    /// Admit `units`, retaining any refusal so the cursor becomes terminal.
    ///
    /// Every move admits at least once, with zero units when it enters no
    /// block. A zero-unit admission charges no fuel and touches no work
    /// counter, because [`QueryWorkCheckpoint::admit`] returns right after its
    /// cancellation poll when `units == 0`, so metering still reflects entered
    /// blocks only. Dropping that call for an in-block step is what lets a
    /// cancelled query keep walking its plan: the drain of a scored candidate
    /// and the move that exhausts a cursor both enter zero blocks, so a
    /// request arriving after the last block entry would never be observed and
    /// the query would return a complete result set.
    ///
    /// The cursor leaf itself takes no lock: the whole cost of the per-move
    /// admission is the checkpoint's cancellation poll, and that poll cannot
    /// be elided or amortized here. It has to read state only the canceller
    /// can write, and a mirror refreshed at block granularity would restore
    /// exactly the in-block blind spot described above. Removing the poll's
    /// lock is therefore a checkpoint-side or upstream concern, not a cursor
    /// one (`bd-quill-cancel-poll-leaf-fastpath-tzztr`).
    fn admit_or_refuse(&mut self, units: u64) -> Result<(), ArgusError> {
        match self.checkpoint.admit(QueryWorkKind::PostingBlock, units) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.refused = RefusedAdmission::capture(&error);
                Err(error)
            }
        }
    }
}

impl PostingCursor for CheckpointPostingCursor<'_> {
    fn doc(&self) -> Option<u32> {
        self.inner.doc()
    }

    fn freq(&self) -> Option<u32> {
        self.inner.freq()
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.inner.positions_handle()
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }

    fn cost(&self) -> u64 {
        self.inner.cost()
    }

    fn segment_num_docs(&self) -> u32 {
        self.inner.segment_num_docs()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.guard_refused()?;
        self.admit_or_refuse(0)?;
        let permit = self.inner.block_permit_for_next();
        if permit > 0 {
            self.admit_or_refuse(permit)?;
        }
        let checkpoint = self.checkpoint.as_borrowed();
        match self.inner.next_admitted(checkpoint) {
            Ok(moved) => Ok(moved),
            Err(error) => {
                self.refused = RefusedAdmission::capture(&error);
                Err(error)
            }
        }
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.guard_refused()?;
        self.admit_or_refuse(0)?;
        let permit = self.inner.block_permit_for_advance(target);
        if permit > 0 {
            self.admit_or_refuse(permit)?;
        }
        let checkpoint = self.checkpoint.as_borrowed();
        match self.inner.advance_admitted(checkpoint, target) {
            Ok(moved) => Ok(moved),
            Err(error) => {
                self.refused = RefusedAdmission::capture(&error);
                Err(error)
            }
        }
    }

    fn validated_post_move_contract(&self) -> Option<ValidatedPostMoveContract> {
        self.inner.validated_post_move_contract()
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        let inner = self.inner.fork_for_pruning()?;
        Some(Box::new(CheckpointPostingCursor {
            inner,
            checkpoint: PostingCheckpoint::Borrowed(self.checkpoint.as_borrowed()),
            // A fork inherits the shared checkpoint but not the parent's
            // refusal: it is a fresh cursor over already-charged bytes, and
            // its own first admission observes any standing refusal.
            refused: None,
        }))
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        self.inner
            .term_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn supports_block_max(&self) -> bool {
        self.inner.supports_block_max()
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        self.inner
            .current_block_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        self.inner.current_block_last_doc()
    }

    fn current_work_block(&self) -> Option<u64> {
        self.inner.current_work_block()
    }

    fn block_permit_for_next(&self) -> u64 {
        self.inner.block_permit_for_next()
    }

    fn block_permit_for_advance(&self, target: u32) -> u64 {
        self.inner.block_permit_for_advance(target)
    }
}
