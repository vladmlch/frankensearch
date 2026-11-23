pub(super) use std::ops::Bound;
pub(super) use std::sync::Arc;

pub(super) use frankensearch_core::DocId;

pub(super) use crate::contract::{BM25_B, BM25_K1, fieldnorm_to_id, id_to_fieldnorm, idf};
pub(super) use crate::delta::DeltaSnapshot;
#[cfg(feature = "bench-internals")]
pub(super) use crate::quiver::ValidatedTermPruningMetadata;
pub(super) use crate::quiver::{
    BlockMaxEntry, DocLenField, DocLenFieldInput, EncodedBlockMax, EncodedDocLenSection,
    EncodedNumericSection, EncodedPositionList, EncodedPostingList, NumericEntry,
    NumericFieldInput, NumericValue, Posting, SnapshotFieldStats,
};
pub(super) use crate::schema::{FieldDescriptor, FieldKind, SchemaDescriptor};

pub(super) use super::boolean_scorers::{estimate_intersection, estimate_union};
pub(super) use super::phrase_scorer::intersect_phrase_positions_carrying_slop;
pub(super) use super::reference_scorer::ScorerNode;
#[cfg(feature = "bench-internals")]
pub(super) use super::union_scorer::UnionPruningStrategy;
pub(super) use super::union_scorer::{BufferedUnionScorer, UnionPruningStats};
pub(super) use super::*;

mod arrival_order_tests;
mod arrival_security_tests;
mod boolean_collector_tests;
mod checkpoint_delta_tests;
mod leaf_scorer_tests;
mod numeric_tombstone_tests;
mod phrase_scorer_tests;
mod pruning_matrix_bench_tests;
mod pruning_refill_tests;

const RANGE_TEST_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "created_at",
    kind: FieldKind::I64 {
        indexed: true,
        fast: true,
    },
    stored: false,
}];
pub(super) const RANGE_TEST_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "argus-range-tests",
    fields: &RANGE_TEST_FIELDS,
};

#[derive(Clone, Debug)]
pub(super) struct VecCursor {
    postings: Vec<Posting>,
    index: usize,
    cost: u64,
    segment_num_docs: u32,
}

impl VecCursor {
    pub(super) const fn new(postings: Vec<Posting>, cost: u64, segment_num_docs: u32) -> Self {
        Self {
            postings,
            index: 0,
            cost,
            segment_num_docs,
        }
    }

    fn current(&self) -> Option<Posting> {
        self.postings.get(self.index).copied()
    }
}

impl PostingCursor for VecCursor {
    fn doc(&self) -> Option<u32> {
        self.current().map(|posting| posting.doc_id)
    }

    fn freq(&self) -> Option<u32> {
        self.current().map(|posting| posting.freq)
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        None
    }

    fn size_hint(&self) -> u32 {
        u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.index < self.postings.len() {
            self.index += 1;
        }
        Ok(self.doc())
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        let tail = self.postings.get(self.index..).unwrap_or_default();
        self.index += tail.partition_point(|posting| posting.doc_id < target);
        Ok(self.doc())
    }
}

#[derive(Default)]
pub(super) struct CountingCheckpoint {
    pub(super) admissions: std::sync::atomic::AtomicUsize,
    pub(super) admitted_units: std::sync::atomic::AtomicU64,
}

impl QueryWorkCheckpoint for CountingCheckpoint {
    fn admit(&self, kind: QueryWorkKind, units: u64) -> Result<(), ArgusError> {
        assert_eq!(kind, QueryWorkKind::PostingBlock);
        self.admissions
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.admitted_units
            .fetch_add(units, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

/// Refuses the `fire_on`-th admission the way a cancelled real
/// [`QueryCheckpoint`] does, and counts what it saw.
pub(super) struct CancelOnNthAdmission {
    fire_on: usize,
    pub(super) admissions: std::sync::atomic::AtomicUsize,
    pub(super) units_at_fire: std::sync::atomic::AtomicU64,
}

impl CancelOnNthAdmission {
    pub(super) fn new(fire_on: usize) -> Self {
        Self {
            fire_on,
            admissions: std::sync::atomic::AtomicUsize::new(0),
            units_at_fire: std::sync::atomic::AtomicU64::new(u64::MAX),
        }
    }
}

impl QueryWorkCheckpoint for CancelOnNthAdmission {
    fn admit(&self, _kind: QueryWorkKind, units: u64) -> Result<(), ArgusError> {
        let ordinal = self
            .admissions
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if ordinal == self.fire_on {
            self.units_at_fire
                .store(units, std::sync::atomic::Ordering::SeqCst);
            return Err(ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test",
            });
        }
        Ok(())
    }
}

/// Records the exact per-admission unit sequence a full scan produces.
#[derive(Default)]
pub(super) struct UnitSequenceCheckpoint {
    pub(super) units: std::sync::Mutex<Vec<u64>>,
}

impl QueryWorkCheckpoint for UnitSequenceCheckpoint {
    fn admit(&self, kind: QueryWorkKind, units: u64) -> Result<(), ArgusError> {
        assert_eq!(kind, QueryWorkKind::PostingBlock);
        self.units
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(units);
        Ok(())
    }
}

/// Refuses with an exact fuel-exhaustion error on the first admission.
#[derive(Default)]
pub(super) struct FuelExhaustedOnFirstAdmission {
    pub(super) admissions: std::sync::atomic::AtomicUsize,
}

impl FuelExhaustedOnFirstAdmission {
    pub(super) const BUDGET: u64 = 7;
    pub(super) const CONSUMED: u64 = 5;
    pub(super) const POSTING_BLOCKS: u64 = 3;
}

impl QueryWorkCheckpoint for FuelExhaustedOnFirstAdmission {
    fn admit(&self, _kind: QueryWorkKind, _units: u64) -> Result<(), ArgusError> {
        let ordinal = self
            .admissions
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if ordinal == 1 {
            return Err(ArgusError::QueryFuelExhausted {
                budget: Self::BUDGET,
                consumed: Self::CONSUMED,
                segments_touched: 1,
                dictionary_blocks: 2,
                posting_blocks: Self::POSTING_BLOCKS,
                position_docs: 0,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct TermBoundOnlyCursor(pub(super) VecCursor);

impl PostingCursor for TermBoundOnlyCursor {
    fn doc(&self) -> Option<u32> {
        self.0.doc()
    }

    fn freq(&self) -> Option<u32> {
        self.0.freq()
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        None
    }

    fn size_hint(&self) -> u32 {
        self.0.size_hint()
    }

    fn cost(&self) -> u64 {
        self.0.cost()
    }

    fn segment_num_docs(&self) -> u32 {
        self.0.segment_num_docs()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.0.next()
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.0.advance(target)
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        Some(Box::new(self.clone()))
    }

    fn term_score_upper_bound(
        &self,
        _live_avgdl: f32,
        weight: f32,
        _record_option: TermRecordOption,
    ) -> Option<f32> {
        (weight.is_finite() && !weight.is_sign_negative()).then_some(weight)
    }
}

#[derive(Clone, Debug)]
pub(super) struct PositionedVecCursor {
    postings: Vec<Posting>,
    positions: Vec<Vec<u32>>,
    index: usize,
    cost: u64,
    segment_num_docs: u32,
}

impl PositionedVecCursor {
    pub(super) fn new(rows: Vec<(u32, Vec<u32>)>, cost: u64, segment_num_docs: u32) -> Self {
        let mut postings = Vec::with_capacity(rows.len());
        let mut positions = Vec::with_capacity(rows.len());
        for (doc, row_positions) in rows {
            let frequency = u32::try_from(row_positions.len()).expect("fixture frequency fits");
            postings.push(Posting::new(doc, frequency));
            positions.push(row_positions);
        }
        Self {
            postings,
            positions,
            index: 0,
            cost,
            segment_num_docs,
        }
    }

    fn current(&self) -> Option<Posting> {
        self.postings.get(self.index).copied()
    }
}

impl PositionsReader for PositionedVecCursor {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        let ordinal = usize::try_from(posting_ordinal)
            .map_err(|_| ArgusError::CursorInvariant("test position ordinal does not fit usize"))?;
        if ordinal != self.index {
            return Err(ArgusError::CursorInvariant(
                "test position handle is no longer current",
            ));
        }
        let positions = self
            .positions
            .get(ordinal)
            .ok_or(ArgusError::CursorInvariant(
                "test position ordinal is out of range",
            ))?;
        output
            .try_reserve_exact(positions.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "test decoded positions",
                count: positions.len(),
            })?;
        output.extend_from_slice(positions);
        Ok(())
    }
}

impl PostingCursor for PositionedVecCursor {
    fn doc(&self) -> Option<u32> {
        self.current().map(|posting| posting.doc_id)
    }

    fn freq(&self) -> Option<u32> {
        self.current().map(|posting| posting.freq)
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.current().and_then(|_| {
            u32::try_from(self.index)
                .ok()
                .map(|ordinal| PositionsHandle::new(self, ordinal))
        })
    }

    fn size_hint(&self) -> u32 {
        u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.index < self.postings.len() {
            self.index += 1;
        }
        Ok(self.doc())
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        let tail = self.postings.get(self.index..).unwrap_or_default();
        self.index += tail.partition_point(|posting| posting.doc_id < target);
        Ok(self.doc())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum CursorFault {
    StickyNext,
    BackwardNext,
    BelowTargetAdvance,
    MovesDespiteSatisfiedTarget,
    ResurrectsAfterExhaustion,
}

#[derive(Clone, Debug)]
pub(super) struct FaultCursor {
    current: Option<Posting>,
    fault: CursorFault,
    moves: u8,
}

impl FaultCursor {
    pub(super) const fn new(fault: CursorFault) -> Self {
        let doc_id = match fault {
            CursorFault::BelowTargetAdvance | CursorFault::ResurrectsAfterExhaustion => 1,
            CursorFault::MovesDespiteSatisfiedTarget => 5,
            CursorFault::StickyNext | CursorFault::BackwardNext => 3,
        };
        Self {
            current: Some(Posting::new(doc_id, 1)),
            fault,
            moves: 0,
        }
    }
}

impl PostingCursor for FaultCursor {
    fn doc(&self) -> Option<u32> {
        self.current.map(|posting| posting.doc_id)
    }

    fn freq(&self) -> Option<u32> {
        self.current.map(|posting| posting.freq)
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        None
    }

    fn size_hint(&self) -> u32 {
        if matches!(self.fault, CursorFault::ResurrectsAfterExhaustion) {
            2
        } else {
            1
        }
    }

    fn segment_num_docs(&self) -> u32 {
        8
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match self.fault {
            CursorFault::StickyNext => {}
            CursorFault::BackwardNext => self.current = Some(Posting::new(2, 1)),
            CursorFault::ResurrectsAfterExhaustion if self.moves == 0 => {
                self.current = None;
            }
            CursorFault::ResurrectsAfterExhaustion => {
                self.current = Some(Posting::new(2, 1));
            }
            CursorFault::BelowTargetAdvance | CursorFault::MovesDespiteSatisfiedTarget => {
                self.current = None;
            }
        }
        self.moves = self.moves.saturating_add(1);
        Ok(self.doc())
    }

    fn advance(&mut self, _target: u32) -> Result<Option<u32>, ArgusError> {
        match self.fault {
            CursorFault::BelowTargetAdvance => self.current = Some(Posting::new(3, 1)),
            CursorFault::MovesDespiteSatisfiedTarget => {
                self.current = Some(Posting::new(6, 1));
            }
            CursorFault::ResurrectsAfterExhaustion if self.current.is_none() => {
                self.current = Some(Posting::new(2, 1));
            }
            CursorFault::StickyNext
            | CursorFault::BackwardNext
            | CursorFault::ResurrectsAfterExhaustion => {}
        }
        self.moves = self.moves.saturating_add(1);
        Ok(self.doc())
    }
}

pub(super) fn snapshot(
    field_ord: u16,
    total_tokens: u64,
    doc_count: u64,
) -> Result<Bm25FieldSnapshot, ArgusError> {
    Bm25FieldSnapshot::new(SnapshotFieldStats {
        field_ord,
        total_tokens,
        doc_count,
    })
}

pub(super) fn term<'a>(
    postings: Vec<Posting>,
    fieldnorms: DocLenField<'a>,
    snapshot: &Bm25FieldSnapshot,
    snapshot_doc_freq: u64,
    cost: u64,
    boost: f32,
) -> Result<ReferenceScorer<'a>, ArgusError> {
    let segment_num_docs = u32::try_from(snapshot.doc_count()).unwrap_or(u32::MAX);
    Ok(ReferenceScorer::term(TermScorer::new(
        VecCursor::new(postings, cost, segment_num_docs),
        fieldnorms,
        snapshot.clone(),
        snapshot_doc_freq,
        TermRecordOption::WithFreqs,
        boost,
    )?))
}

pub(super) fn sealed_union<'a>(
    posting_lists: &'a [crate::quiver::PostingList<'_>],
    block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
    fieldnorms: DocLenField<'a>,
    snapshot: &Bm25FieldSnapshot,
    rows_by_term: &[Vec<Posting>],
    boosts: &[f32],
    segment_num_docs: u32,
) -> Result<ReferenceScorer<'a>, ArgusError> {
    sealed_union_with_record_option(
        posting_lists,
        block_max,
        fieldnorms,
        snapshot,
        rows_by_term,
        boosts,
        segment_num_docs,
        TermRecordOption::WithFreqs,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sealed_union_with_record_option<'a>(
    posting_lists: &'a [crate::quiver::PostingList<'_>],
    block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
    fieldnorms: DocLenField<'a>,
    snapshot: &Bm25FieldSnapshot,
    rows_by_term: &[Vec<Posting>],
    boosts: &[f32],
    segment_num_docs: u32,
    record_option: TermRecordOption,
) -> Result<ReferenceScorer<'a>, ArgusError> {
    if posting_lists.len() != rows_by_term.len() || posting_lists.len() != boosts.len() {
        return Err(ArgusError::CursorInvariant(
            "sealed union fixture cardinalities disagree",
        ));
    }
    if block_max.is_some_and(|bounds| bounds.len() != posting_lists.len()) {
        return Err(ArgusError::CursorInvariant(
            "sealed union BLOCKMAX cardinality disagrees",
        ));
    }
    let mut clauses = Vec::new();
    clauses
        .try_reserve_exact(posting_lists.len())
        .map_err(|_| ArgusError::Allocation {
            resource: "sealed union fixture clauses",
            count: posting_lists.len(),
        })?;
    for (index, postings) in posting_lists.iter().enumerate() {
        let cursor = if let Some(bounds) = block_max {
            SealedPostingCursor::with_block_max(
                postings.cursor()?,
                Arc::clone(&bounds[index]),
                postings.doc_freq(),
                segment_num_docs,
            )
        } else {
            SealedPostingCursor::new(postings, segment_num_docs)?
        };
        let scorer = TermScorer::new(
            cursor,
            fieldnorms,
            snapshot.clone(),
            u64::try_from(rows_by_term[index].len()).map_err(|_| {
                ArgusError::CursorInvariant("fixture document frequency exceeds u64")
            })?,
            record_option,
            boosts[index],
        )?;
        clauses.push(ScorerClause::should(ReferenceScorer::term(scorer)));
    }
    ReferenceScorer::boolean(clauses)
}

/// Ranked-root (`boolean_topdocs`) analogue of [`sealed_union`], with one
/// explicit snapshot document frequency shared by every clause.
///
/// `boolean_topdocs` is the only constructor that sets
/// `tantivy_topdocs_term_union`, and it is what shipping lowers a ranked
/// direct-term root to, so it is the shape rank pruning actually runs in.
///
/// The explicit document frequency is what a real multi-segment snapshot
/// supplies: the cursor is segment-local while `doc_freq` is global. Fixing
/// one value across every clause makes `idf` identical between them, so a
/// planted contribution ratio is exactly the boost ratio no matter how many
/// postings each term carries.
pub(super) fn sealed_topdocs_union<'a>(
    posting_lists: &'a [crate::quiver::PostingList<'_>],
    block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
    fieldnorms: DocLenField<'a>,
    snapshot: &Bm25FieldSnapshot,
    snapshot_doc_freq: u64,
    boosts: &[f32],
    segment_num_docs: u32,
) -> Result<ReferenceScorer<'a>, ArgusError> {
    if posting_lists.len() != boosts.len() {
        return Err(ArgusError::CursorInvariant(
            "ranked-root union fixture cardinalities disagree",
        ));
    }
    if block_max.is_some_and(|bounds| bounds.len() != posting_lists.len()) {
        return Err(ArgusError::CursorInvariant(
            "ranked-root union BLOCKMAX cardinality disagrees",
        ));
    }
    let mut clauses = Vec::new();
    clauses
        .try_reserve_exact(posting_lists.len())
        .map_err(|_| ArgusError::Allocation {
            resource: "ranked-root union fixture clauses",
            count: posting_lists.len(),
        })?;
    for (index, postings) in posting_lists.iter().enumerate() {
        let cursor = if let Some(bounds) = block_max {
            SealedPostingCursor::with_block_max(
                postings.cursor()?,
                Arc::clone(&bounds[index]),
                postings.doc_freq(),
                segment_num_docs,
            )
        } else {
            SealedPostingCursor::new(postings, segment_num_docs)?
        };
        clauses.push(ScorerClause::should(ReferenceScorer::term(
            TermScorer::new(
                cursor,
                fieldnorms,
                snapshot.clone(),
                snapshot_doc_freq,
                TermRecordOption::WithFreqs,
                boosts[index],
            )?,
        )));
    }
    ReferenceScorer::boolean_topdocs(clauses)
}

/// Build a nested pure-term union: `group_sizes` slices the flat term
/// arrays into per-group multi-field unions (each a `sealed_union`), wrapped
/// in an outer `Should` union. With `block_max` present each group's terms
/// carry ceilings, so the outer union is grouped-`MaxScore` eligible.
#[allow(clippy::too_many_arguments)]
pub(super) fn sealed_grouped_union<'a>(
    posting_lists: &'a [crate::quiver::PostingList<'_>],
    block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
    fieldnorms: DocLenField<'a>,
    snapshot: &Bm25FieldSnapshot,
    rows_by_term: &[Vec<Posting>],
    boosts: &[f32],
    segment_num_docs: u32,
    group_sizes: &[usize],
) -> Result<ReferenceScorer<'a>, ArgusError> {
    sealed_grouped_union_with_record_option(
        posting_lists,
        block_max,
        fieldnorms,
        snapshot,
        rows_by_term,
        boosts,
        segment_num_docs,
        group_sizes,
        TermRecordOption::WithFreqs,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sealed_grouped_union_with_record_option<'a>(
    posting_lists: &'a [crate::quiver::PostingList<'_>],
    block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
    fieldnorms: DocLenField<'a>,
    snapshot: &Bm25FieldSnapshot,
    rows_by_term: &[Vec<Posting>],
    boosts: &[f32],
    segment_num_docs: u32,
    group_sizes: &[usize],
    record_option: TermRecordOption,
) -> Result<ReferenceScorer<'a>, ArgusError> {
    let mut outer = Vec::new();
    let mut offset = 0_usize;
    for &size in group_sizes {
        let end = offset + size;
        let group = sealed_union_with_record_option(
            &posting_lists[offset..end],
            block_max.map(|bounds| &bounds[offset..end]),
            fieldnorms,
            snapshot,
            &rows_by_term[offset..end],
            &boosts[offset..end],
            segment_num_docs,
            record_option,
        )?;
        outer.push(ScorerClause::should(group));
        offset = end;
    }
    ReferenceScorer::boolean(outer)
}

pub(super) fn assert_hits_bit_exact(actual: &[ScoredDoc], expected: &[ScoredDoc]) {
    assert_eq!(actual.len(), expected.len(), "hit count differs");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.global_docid, expected.global_docid,
            "docid differs at rank {index}"
        );
        assert_eq!(
            actual.score.to_bits(),
            expected.score.to_bits(),
            "score bits differ at rank {index} for doc {}",
            actual.global_docid
        );
    }
}

/// Bench-only control retaining the sealed cursor's data path while forcing
/// `TermScorer` through its defensive post-move validation branch.
#[cfg(feature = "bench-internals")]
pub(super) struct ForcedPostMoveValidationCursor<'a> {
    pub(super) inner: SealedPostingCursor<'a>,
}

#[cfg(feature = "bench-internals")]
impl PostingCursor for ForcedPostMoveValidationCursor<'_> {
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
        self.inner.next()
    }
    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.inner.advance(target)
    }
    fn term_score_upper_bound(
        &self,
        average: f32,
        weight: f32,
        option: TermRecordOption,
    ) -> Option<f32> {
        self.inner.term_score_upper_bound(average, weight, option)
    }
    fn supports_block_max(&self) -> bool {
        self.inner.supports_block_max()
    }
    fn current_block_score_upper_bound(
        &self,
        average: f32,
        weight: f32,
        option: TermRecordOption,
    ) -> Option<f32> {
        self.inner
            .current_block_score_upper_bound(average, weight, option)
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

pub(super) fn validated_block_max_entries(
    encoded_terms: &[(EncodedPostingList, EncodedBlockMax)],
    posting_lists: &[crate::quiver::PostingList<'_>],
    fieldnorms: DocLenField<'_>,
) -> Result<Vec<Arc<[BlockMaxEntry]>>, crate::quiver::BlockMaxError> {
    encoded_terms
        .iter()
        .zip(posting_lists)
        .map(|((_, block_max), postings)| {
            block_max
                .block_max_list(postings, fieldnorms)
                .map(|bounds| Arc::<[BlockMaxEntry]>::from(bounds.entries()))
        })
        .collect()
}

pub(super) fn postings(docs: &[u32]) -> Vec<Posting> {
    docs.iter().map(|&doc| Posting::new(doc, 1)).collect()
}

pub(super) fn expected_term_score(
    snapshot: &Bm25FieldSnapshot,
    doc_freq: u64,
    fieldnorm_id: u8,
    frequency: u32,
    boost: f32,
) -> f32 {
    let average = snapshot
        .average_field_length()
        .expect("non-empty scored fixture");
    let decoded = id_to_fieldnorm(fieldnorm_id);
    let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * decoded as f32 / average);
    let mut weight = idf(doc_freq, snapshot.doc_count()) * (1.0 + BM25_K1);
    weight *= boost;
    let frequency = frequency as f32;
    weight * (frequency / (frequency + norm))
}

pub(super) fn expected_phrase_score(
    snapshot: &Bm25FieldSnapshot,
    doc_freqs: &[u64],
    fieldnorm_id: u8,
    frequency: u32,
    boost: f32,
) -> f32 {
    let average = snapshot
        .average_field_length()
        .expect("non-empty scored fixture");
    let decoded = id_to_fieldnorm(fieldnorm_id);
    let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * decoded as f32 / average);
    let mut idf_sum = 0.0_f32;
    for &doc_freq in doc_freqs {
        idf_sum += idf(doc_freq, snapshot.doc_count());
    }
    let mut weight = idf_sum * (1.0 + BM25_K1);
    weight *= boost;
    let frequency = frequency as f32;
    weight * (frequency / (frequency + norm))
}

/// [`sealed_topdocs_union`] built on the *validated* arm, with each cursor
/// reporting the work it really performs.
///
/// This is the only builder that exercises the seeking fill. A wrapper
/// cannot be used to instrument it: wrapping moves the clause onto the
/// opaque arm, so an outside counter would measure the walk it was meant to
/// disprove. The counter therefore lives on the validated cursor itself.
pub(super) fn validated_topdocs_union<'a>(
    posting_lists: &'a [crate::quiver::PostingList<'_>],
    block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
    fieldnorms: DocLenField<'a>,
    snapshot: &Bm25FieldSnapshot,
    snapshot_doc_freq: u64,
    boosts: &[f32],
    segment_num_docs: u32,
) -> Result<(ReferenceScorer<'a>, Vec<std::rc::Rc<CursorWork>>), ArgusError> {
    if posting_lists.len() != boosts.len() {
        return Err(ArgusError::CursorInvariant(
            "validated ranked-root fixture cardinalities disagree",
        ));
    }
    let checkpoint: Arc<dyn QueryWorkCheckpoint> = Arc::new(CountingCheckpoint::default());
    let mut clauses = Vec::new();
    let mut work = Vec::new();
    for (index, postings) in posting_lists.iter().enumerate() {
        let sealed = if let Some(bounds) = block_max {
            SealedPostingCursor::with_block_max(
                postings.cursor()?,
                Arc::clone(&bounds[index]),
                postings.doc_freq(),
                segment_num_docs,
            )
        } else {
            SealedPostingCursor::new(postings, segment_num_docs)?
        };
        let counter = std::rc::Rc::new(CursorWork::default());
        work.push(std::rc::Rc::clone(&counter));
        clauses.push(ScorerClause::should(ReferenceScorer::term(
            TermScorer::new_validated(
                ValidatedArrivalCursor::new_counted(sealed, Arc::clone(&checkpoint), counter),
                fieldnorms,
                snapshot.clone(),
                snapshot_doc_freq,
                TermRecordOption::WithFreqs,
                boosts[index],
            )?,
        )));
    }
    Ok((ReferenceScorer::boolean_topdocs(clauses)?, work))
}

pub(super) fn total_steps(work: &[std::rc::Rc<CursorWork>]) -> u64 {
    work.iter().map(|counter| counter.steps.get()).sum()
}

/// A wrapper embedding a real sealed cursor and forwarding every
/// [`PostingCursor`] method to it.
///
/// It is as close to the genuine article as a wrapper can be, and that is
/// the point of the fixture. Two earlier designs handed the seeking fill to
/// something like this: one admitted on a forwardable capability token, the
/// other on a per-result receipt that a wrapper could mint honestly and then
/// replay. Both let a wrapper stand between the union and the cursor whose
/// history it was reading.
///
/// There is nothing left for this type to forward or withhold. Arrival reads
/// are not on [`PostingCursor`] at all — they are inherent methods on
/// [`ValidatedArrivalCursor`], reachable only by the term scorer that owns
/// one — so wrapping is itself the whole of what this can do, and a wrapped
/// clause lands on [`TermCursor::Opaque`] and walks.
pub(super) struct PartialArrivalCursor<'a> {
    pub(super) inner: SealedPostingCursor<'a>,
}

impl PostingCursor for PartialArrivalCursor<'_> {
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
        self.inner.next()
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.inner.advance(target)
    }

    fn validated_post_move_contract(&self) -> Option<ValidatedPostMoveContract> {
        self.inner.validated_post_move_contract()
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        self.inner.fork_for_pruning()
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

/// [`sealed_topdocs_union`] whose sealed cursors are each handed to `wrap`,
/// so one fixture can be driven through every hostile wrapper.
pub(super) fn wrapped_topdocs_union<'a, C, F>(
    posting_lists: &'a [crate::quiver::PostingList<'_>],
    block_max: &'a [Arc<[BlockMaxEntry]>],
    fieldnorms: DocLenField<'a>,
    snapshot: &Bm25FieldSnapshot,
    snapshot_doc_freq: u64,
    boosts: &[f32],
    segment_num_docs: u32,
    wrap: F,
) -> Result<ReferenceScorer<'a>, ArgusError>
where
    C: PostingCursor + 'a,
    F: Fn(SealedPostingCursor<'a>) -> C,
{
    let mut clauses = Vec::new();
    for (index, postings) in posting_lists.iter().enumerate() {
        let inner = SealedPostingCursor::with_block_max(
            postings.cursor()?,
            Arc::clone(&block_max[index]),
            postings.doc_freq(),
            segment_num_docs,
        );
        clauses.push(ScorerClause::should(ReferenceScorer::term(
            TermScorer::new(
                wrap(inner),
                fieldnorms,
                snapshot.clone(),
                snapshot_doc_freq,
                TermRecordOption::WithFreqs,
                boosts[index],
            )?,
        )));
    }
    ReferenceScorer::boolean_topdocs(clauses)
}
