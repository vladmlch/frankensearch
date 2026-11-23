use std::sync::Arc;

use crate::delta::{DeltaBlockMax, DeltaPosting, DeltaPostings, DeltaSnapshot, DeltaTerm};
use crate::quiver::{BlockMaxEntry, ValidatedTermPruningMetadata};

use super::PostingCursor;
use super::cursor::ValidatedPostMoveContract;
use super::term_scorer::{FieldNormReader, TermRecordOption};
use super::{
    ArgusError, DELTA_LOGICAL_BLOCK_POSTINGS, PositionsHandle, PositionsReader,
    QueryWorkCheckpoint, QueryWorkKind,
};

enum SealedCursorInner<'a> {
    Docs(crate::quiver::PostingCursor<'a>),
    Positions(crate::quiver::PositionCursor<'a>),
}

#[derive(Clone)]
enum SealedBlockMax {
    ValidatedTerm(Arc<ValidatedTermPruningMetadata>),
    #[cfg(test)]
    Fixture(Arc<[BlockMaxEntry]>),
}

impl SealedBlockMax {
    fn entries(&self) -> &[BlockMaxEntry] {
        match self {
            Self::ValidatedTerm(metadata) => metadata.block_max(),
            #[cfg(test)]
            Self::Fixture(entries) => entries,
        }
    }
}

/// Owner-safe adapter for one validated sealed posting or position cursor.
pub struct SealedPostingCursor<'a> {
    inner: SealedCursorInner<'a>,
    block_max: Option<SealedBlockMax>,
    size_hint: u32,
    segment_num_docs: u32,
}

impl<'a> SealedPostingCursor<'a> {
    /// Open a position-free sealed posting cursor.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the validated posting list can no longer open
    /// its cursor.
    pub fn new(
        postings: &'a crate::quiver::PostingList<'_>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        Ok(Self {
            inner: SealedCursorInner::Docs(postings.cursor()?),
            block_max: None,
            size_hint: postings.doc_freq(),
            segment_num_docs,
        })
    }

    /// Bind an owner-safe compressed posting cursor without pruning metadata.
    ///
    /// This is the exact-count and scoreless production path: POSTINGS remains
    /// lazy and owner-safe, while BLOCKMAX is neither opened nor represented by
    /// an empty sentinel.
    #[must_use]
    pub(crate) fn from_owned(
        postings: crate::quiver::PostingCursor<'a>,
        size_hint: u32,
        segment_num_docs: u32,
    ) -> Self {
        Self {
            inner: SealedCursorInner::Docs(postings),
            block_max: None,
            size_hint,
            segment_num_docs,
        }
    }

    /// Open an owner-safe cursor from one segment-cached, fully validated term.
    ///
    /// POSTINGS block layout and BLOCKMAX rows remain paired behind the same
    /// shared metadata value, so no production caller can bind bounds from a
    /// different term.
    ///
    /// # Errors
    ///
    /// Returns a typed posting invariant error if the selected immutable term
    /// span no longer matches the cached validation witness.
    pub(crate) fn from_validated_pruning(
        postings_bytes: &'a [u8],
        metadata: Arc<ValidatedTermPruningMetadata>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        let size_hint = metadata.doc_freq();
        let postings = metadata.cursor(postings_bytes)?;
        Ok(Self {
            inner: SealedCursorInner::Docs(postings),
            block_max: Some(SealedBlockMax::ValidatedTerm(metadata)),
            size_hint,
            segment_num_docs,
        })
    }

    /// Bind an owner-safe posting cursor to fully validated BLOCKMAX entries.
    ///
    /// The entries must come from the same [`crate::quiver::PostingList`] that
    /// produced `postings`; the storage opener is responsible for validating
    /// that one-to-one relationship before construction.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn with_block_max(
        postings: crate::quiver::PostingCursor<'a>,
        block_max: Arc<[BlockMaxEntry]>,
        size_hint: u32,
        segment_num_docs: u32,
    ) -> Self {
        Self {
            inner: SealedCursorInner::Docs(postings),
            block_max: Some(SealedBlockMax::Fixture(block_max)),
            size_hint,
            segment_num_docs,
        }
    }

    /// Open a sealed cursor already paired with its validated POSITIONS stream.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the validated paired streams can no longer
    /// open their allocation-free cursor.
    pub fn with_positions(
        positions: &'a crate::quiver::PositionList<'_>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        Ok(Self {
            inner: SealedCursorInner::Positions(positions.cursor()?),
            block_max: None,
            size_hint: positions.doc_freq(),
            segment_num_docs,
        })
    }

    /// Inclusive last document this term can still reach, read from validated
    /// block metadata without moving or decoding.
    ///
    /// Inherent rather than a trait method, and that is the point: there is no
    /// arrival surface on [`PostingCursor`] for a wrapper to override, forward,
    /// or answer on behalf of a different cursor. Only code holding this
    /// concrete type can ask, and the only thing that holds one is the term
    /// scorer that also advances and scores it.
    #[must_use]
    pub(crate) fn sealed_final_doc(&self) -> Option<u32> {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return None;
        };
        cursor.final_doc()
    }

    /// Greatest document this term holds strictly below `target`.
    ///
    /// # Errors
    ///
    /// Returns a typed error if a selected validated block cannot be decoded,
    /// and refuses for a POSITIONS-backed cursor, which has no block table.
    pub(crate) fn sealed_doc_before(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        let SealedCursorInner::Docs(cursor) = &mut self.inner else {
            return Err(ArgusError::CursorInvariant(
                "a POSITIONS-backed sealed cursor has no validated arrival metadata",
            ));
        };
        Ok(cursor.doc_before(target)?)
    }

    /// Blocks this cursor has physically decoded, forward loads and backward
    /// reads alike. Test builds only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn sealed_decoded_blocks(&self) -> usize {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return 0;
        };
        cursor.decoded_blocks()
    }

    /// Blocks [`Self::sealed_doc_before`] would decode, determined without
    /// decoding, so the caller admits them first.
    #[must_use]
    pub(crate) fn sealed_arrival_permit(&self, target: u32) -> u64 {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return 0;
        };
        u64::from(cursor.doc_before_decodes_block(target))
    }
}

impl PostingCursor for SealedPostingCursor<'_> {
    fn doc(&self) -> Option<u32> {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => cursor.doc(),
            SealedCursorInner::Positions(cursor) => cursor.doc(),
        }
    }

    fn freq(&self) -> Option<u32> {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => cursor.freq(),
            SealedCursorInner::Positions(cursor) => cursor.freq(),
        }
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        match &self.inner {
            SealedCursorInner::Docs(_) => None,
            SealedCursorInner::Positions(cursor) => cursor
                .posting_ordinal()
                .map(|ordinal| PositionsHandle::new(self, ordinal)),
        }
    }

    fn size_hint(&self) -> u32 {
        self.size_hint
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match &mut self.inner {
            SealedCursorInner::Docs(cursor) => Ok(cursor.next()?.map(|posting| posting.doc_id)),
            SealedCursorInner::Positions(cursor) => {
                Ok(cursor.next()?.map(|posting| posting.doc_id))
            }
        }
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        match &mut self.inner {
            SealedCursorInner::Docs(cursor) => {
                Ok(cursor.advance(target)?.map(|posting| posting.doc_id))
            }
            SealedCursorInner::Positions(cursor) => {
                Ok(cursor.advance(target)?.map(|posting| posting.doc_id))
            }
        }
    }

    fn validated_post_move_contract(&self) -> Option<ValidatedPostMoveContract> {
        Some(ValidatedPostMoveContract(()))
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return None;
        };
        self.block_max.as_ref()?;
        Some(Box::new(Self {
            inner: SealedCursorInner::Docs(cursor.clone()),
            block_max: self.block_max.clone(),
            size_hint: self.size_hint,
            segment_num_docs: self.segment_num_docs,
        }))
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        let entries = self.block_max.as_ref()?.entries();
        let mut maximum = None::<f32>;
        for entry in entries {
            let bound = record_option.score_upper_bound(
                entry.max_frequency(),
                entry.min_fieldnorm_id(),
                live_avgdl,
                weight,
            )?;
            maximum = Some(maximum.map_or(bound, |current| current.max(bound)));
        }
        maximum
    }

    fn supports_block_max(&self) -> bool {
        matches!(&self.inner, SealedCursorInner::Docs(_))
            && self
                .block_max
                .as_ref()
                .is_some_and(|entries| !entries.entries().is_empty())
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return None;
        };
        let block_index = cursor.block_index()?;
        let entry = self.block_max.as_ref()?.entries().get(block_index)?;
        record_option.score_upper_bound(
            entry.max_frequency(),
            entry.min_fieldnorm_id(),
            live_avgdl,
            weight,
        )
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return None;
        };
        cursor.block_last_doc()
    }

    fn current_work_block(&self) -> Option<u64> {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => cursor.block_index().map(|block| block as u64),
            SealedCursorInner::Positions(cursor) => {
                cursor.posting_block_index().map(|block| block as u64)
            }
        }
    }

    /// Exactly the decodes the next step performs, from the codec's own
    /// mirror of that step. A positional step touches its paired posting
    /// cursor once; a docs step decodes at most its successor block.
    fn block_permit_for_next(&self) -> u64 {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => u64::from(cursor.next_decodes_block()),
            SealedCursorInner::Positions(cursor) => cursor.next_decode_count(),
        }
    }

    /// Exactly the decodes the seek performs. A positional seek is two-stage
    /// and can decode several blocks, so the permit is a count, not a flag.
    fn block_permit_for_advance(&self, target: u32) -> u64 {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => u64::from(cursor.advance_decodes_block(target)),
            SealedCursorInner::Positions(cursor) => cursor.advance_decode_count(target),
        }
    }
}

impl PositionsReader for SealedPostingCursor<'_> {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        let SealedCursorInner::Positions(cursor) = &self.inner else {
            return Err(ArgusError::CursorInvariant(
                "position-free sealed cursor cannot resolve positions",
            ));
        };
        if cursor.posting_ordinal() != Some(posting_ordinal) {
            return Err(ArgusError::CursorInvariant(
                "positions handle no longer identifies the current posting",
            ));
        }
        let positions = cursor.positions()?.ok_or(ArgusError::CursorInvariant(
            "positioned sealed cursor has no current position run",
        ))?;
        output
            .try_reserve_exact(positions.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "decoded positions",
                count: positions.len(),
            })?;
        for position in positions {
            output.push(position?);
        }
        Ok(())
    }
}

/// Allocation-free cursor over one immutable Delta term chain.
///
/// Physical postings remain append-ordered in the frozen generation. The
/// adapter validates that ordering while skipping rows superseded or deleted
/// by the Delta-local identity overlay. A cursor starts on its first live row,
/// exactly like the sealed adapter.
#[derive(Clone)]
pub struct DeltaPostingCursor<'a> {
    term: Option<DeltaTerm<'a>>,
    block_max: Option<DeltaBlockMax>,
    remaining: Option<DeltaPostings<'a>>,
    current: Option<DeltaPosting<'a>>,
    current_ordinal: Option<u32>,
    next_ordinal: u32,
    last_physical_doc: Option<u32>,
    /// Physical rows this term holds, tombstoned ones included.
    ///
    /// A step is only real while an unread row remains, so this bounds the
    /// admissions: a walk that ends exactly on a logical-block boundary must
    /// not buy a permit for a block that does not exist.
    physical_len: u32,
    size_hint: u32,
    cost: u64,
    segment_num_docs: u32,
}

impl<'a> DeltaPostingCursor<'a> {
    /// Open one live-only `(field, term)` cursor in a frozen Delta generation.
    ///
    /// A missing term produces a fused empty cursor with the generation's live
    /// segment cardinality. No posting or position row is materialized.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant failure if a generation exceeds the durable
    /// `u32` cursor domain or its first posting violates ascending order.
    pub fn new(
        delta: &'a DeltaSnapshot,
        field_ord: u16,
        term_bytes: &[u8],
    ) -> Result<Self, ArgusError> {
        Self::open(delta, field_ord, term_bytes, None)
    }

    /// Open the same cursor with its construction walk admitted.
    ///
    /// Construction pulls the first live posting, and that pull can traverse
    /// an arbitrarily long tombstoned prefix. Shipping callers use this so the
    /// walk is admitted as it happens instead of running before any checkpoint
    /// exists.
    ///
    /// # Errors
    ///
    /// Returns the checkpoint's refusal, or the same typed invariant failures
    /// as [`Self::new`].
    pub fn new_admitted(
        delta: &'a DeltaSnapshot,
        field_ord: u16,
        term_bytes: &[u8],
        checkpoint: &dyn QueryWorkCheckpoint,
    ) -> Result<Self, ArgusError> {
        Self::open(delta, field_ord, term_bytes, Some(checkpoint))
    }

    fn open(
        delta: &'a DeltaSnapshot,
        field_ord: u16,
        term_bytes: &[u8],
        checkpoint: Option<&dyn QueryWorkCheckpoint>,
    ) -> Result<Self, ArgusError> {
        let term = delta.find_term(field_ord, term_bytes);
        // Term metadata is answered from the count taken at freeze, so
        // opening a cursor reads no rows and buys no permit for the lookup.
        // It still polls: a term missing from this generation does no work at
        // all, and that is exactly the path on which a cancelled query would
        // otherwise open cursors forever without ever being told to stop.
        if let Some(checkpoint) = checkpoint {
            checkpoint.admit(QueryWorkKind::PostingBlock, 0)?;
        }
        let (live_doc_freq, block_max) =
            term.map_or((0, None), DeltaTerm::live_doc_freq_and_block_max);
        let size_hint = u32::try_from(live_doc_freq).map_err(|_| {
            ArgusError::CursorInvariant("Delta live document frequency exceeds u32")
        })?;
        // Intersection ordering also fixes the f32 score-accumulation order.
        // Use the same live document frequency emitted by a sealed cursor so
        // folding Delta tombstones cannot perturb score bits across a seal.
        let cost = u64::try_from(live_doc_freq).map_err(|_| {
            ArgusError::CursorInvariant("Delta live document frequency exceeds u64")
        })?;
        let segment_num_docs = u32::try_from(delta.live_document_count())
            .map_err(|_| ArgusError::CursorInvariant("Delta live document count exceeds u32"))?;
        let remaining = term.map(DeltaTerm::postings);
        let physical_len = u32::try_from(term.map_or(0, DeltaTerm::physical_doc_freq))
            .map_err(|_| ArgusError::CursorInvariant("Delta physical row count exceeds u32"))?;
        let mut cursor = Self {
            term,
            block_max,
            remaining,
            current: None,
            current_ordinal: None,
            next_ordinal: 0,
            last_physical_doc: None,
            physical_len,
            size_hint,
            cost,
            segment_num_docs,
        };
        cursor.pull_next_live(checkpoint)?;
        Ok(cursor)
    }

    /// Pull the next live posting, admitting each physical step it walks.
    ///
    /// A Delta term can carry an arbitrarily long tombstoned prefix, and one
    /// move walks all of it. Admitting only at the end would leave that walk
    /// uncancellable and charge its blocks after the fact, so the admission
    /// happens *inside* the loop, before each physical step: zero units
    /// within a logical block, one unit when the step opens a new one.
    ///
    /// The charge is over **physical** blocks walked. For a tombstone-free
    /// term that is the same total as before, because every step is live. For
    /// a term with tombstones it is higher, and deliberately so: the blocks a
    /// walk reads to skip superseded postings are work the query performed,
    /// and charging only the live positions understated it while leaving the
    /// walk itself unobservable.
    fn pull_next_live(
        &mut self,
        checkpoint: Option<&dyn QueryWorkCheckpoint>,
    ) -> Result<Option<u32>, ArgusError> {
        let Some(term) = self.term else {
            self.current = None;
            self.current_ordinal = None;
            return Ok(None);
        };
        let Some(remaining) = self.remaining.as_mut() else {
            self.current = None;
            self.current_ordinal = None;
            return Ok(None);
        };
        loop {
            if let Some(checkpoint) = checkpoint {
                // Admitted before the step, never after: a tombstoned run is
                // still work, and the query must be able to stop inside it.
                // A step past the last physical row reads nothing, so a walk
                // that ends exactly on a block boundary buys no phantom
                // permit for a block that does not exist — it still polls.
                let opens_block = self.next_ordinal < self.physical_len
                    && self.next_ordinal % DELTA_LOGICAL_BLOCK_POSTINGS == 0;
                checkpoint.admit(QueryWorkKind::PostingBlock, u64::from(opens_block))?;
            }
            let Some(posting) = remaining.next() else {
                self.current = None;
                self.current_ordinal = None;
                return Ok(None);
            };
            if self
                .last_physical_doc
                .is_some_and(|previous| posting.global_docid <= previous)
            {
                return Err(ArgusError::CursorInvariant(
                    "Delta term postings are not strictly docid-ascending",
                ));
            }
            self.last_physical_doc = Some(posting.global_docid);
            let ordinal = self.next_ordinal;
            self.next_ordinal =
                self.next_ordinal
                    .checked_add(1)
                    .ok_or(ArgusError::CursorInvariant(
                        "Delta posting ordinal exceeds u32",
                    ))?;
            if !term.is_live(posting) {
                continue;
            }
            self.current = Some(posting);
            self.current_ordinal = Some(ordinal);
            return Ok(Some(posting.global_docid));
        }
    }
}

impl PostingCursor for DeltaPostingCursor<'_> {
    fn doc(&self) -> Option<u32> {
        self.current.map(|posting| posting.global_docid)
    }

    fn freq(&self) -> Option<u32> {
        self.current.map(|posting| posting.frequency)
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.current.filter(|posting| posting.has_positions())?;
        self.current_ordinal
            .map(|ordinal| PositionsHandle::new(self, ordinal))
    }

    fn size_hint(&self) -> u32 {
        self.size_hint
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        self.pull_next_live(None)
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_none_or(|doc| doc >= target) {
            return Ok(self.doc());
        }
        loop {
            let moved = self.pull_next_live(None)?;
            if moved.is_none_or(|doc| doc >= target) {
                return Ok(moved);
            }
        }
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        self.block_max?;
        Some(Box::new(self.clone()))
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        let block_max = self.block_max?;
        record_option.score_upper_bound(
            block_max.max_frequency(),
            block_max.min_fieldnorm_id(),
            live_avgdl,
            weight,
        )
    }

    fn current_work_block(&self) -> Option<u64> {
        self.current_ordinal
            .map(|ordinal| u64::from(ordinal / DELTA_LOGICAL_BLOCK_POSTINGS))
    }

    /// Delta admits its own walk inside `pull_next_live`, so the caller must
    /// not pre-charge a permit on top of it.
    fn block_permit_for_next(&self) -> u64 {
        0
    }

    fn block_permit_for_advance(&self, _target: u32) -> u64 {
        0
    }

    fn next_admitted(
        &mut self,
        checkpoint: &dyn QueryWorkCheckpoint,
    ) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        self.pull_next_live(Some(checkpoint))
    }

    fn advance_admitted(
        &mut self,
        checkpoint: &dyn QueryWorkCheckpoint,
        target: u32,
    ) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_none_or(|doc| doc >= target) {
            return Ok(self.doc());
        }
        loop {
            let moved = self.pull_next_live(Some(checkpoint))?;
            if moved.is_none_or(|doc| doc >= target) {
                return Ok(moved);
            }
        }
    }
}

impl PositionsReader for DeltaPostingCursor<'_> {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        if self.current_ordinal != Some(posting_ordinal) {
            return Err(ArgusError::CursorInvariant(
                "positions handle no longer identifies the current Delta posting",
            ));
        }
        let term = self.term.ok_or(ArgusError::CursorInvariant(
            "empty Delta cursor cannot resolve positions",
        ))?;
        let posting = self.current.ok_or(ArgusError::CursorInvariant(
            "exhausted Delta cursor cannot resolve positions",
        ))?;
        let expected = usize::try_from(posting.frequency)
            .map_err(|_| ArgusError::CursorInvariant("Delta position count does not fit usize"))?;
        let positions = term.positions(posting).ok_or(ArgusError::CursorInvariant(
            "positioned Delta cursor has no current position run",
        ))?;
        output
            .try_reserve_exact(expected)
            .map_err(|_| ArgusError::Allocation {
                resource: "decoded Delta positions",
                count: expected,
            })?;
        let initial_len = output.len();
        output.extend(positions);
        if output.len().saturating_sub(initial_len) != expected {
            return Err(ArgusError::CursorInvariant(
                "Delta position run disagrees with posting frequency",
            ));
        }
        Ok(())
    }
}

/// Borrowed fieldnorm view paired with one frozen Delta generation.
#[derive(Clone, Copy)]
pub struct DeltaFieldNorms<'a> {
    delta: &'a DeltaSnapshot,
    field_ord: u16,
}

impl<'a> DeltaFieldNorms<'a> {
    /// Bind one indexed string field in an immutable Delta generation.
    #[must_use]
    pub const fn new(delta: &'a DeltaSnapshot, field_ord: u16) -> Self {
        Self { delta, field_ord }
    }
}

impl FieldNormReader for DeltaFieldNorms<'_> {
    fn field_ord(&self) -> u16 {
        self.field_ord
    }

    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
        self.delta.live_fieldnorm_id(self.field_ord, global_docid)
    }
}
