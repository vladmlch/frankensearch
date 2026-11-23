use std::sync::Arc;

use crate::contract::{BM25_K1, idf};

use super::boolean_scorers::{estimate_intersection, estimate_union, finite_score};
use super::cursor::PostingCursor;
use super::term_scorer::{Bm25FieldSnapshot, FieldNormReader};
use super::{ArgusError, QueryWorkCheckpoint, QueryWorkKind};

/// One analyzed phrase term bound to its positioned posting cursor.
///
/// Terms must be supplied in non-decreasing analyzer-position order. Terms at
/// the same position are treated as alternatives for matching while retaining
/// their original order in the fixed phrase-weight calculation. The field
/// ordinal is trusted lowering metadata and must come from the same posting-open
/// operation as the cursor; the opaque cursor contract cannot recover it.
pub struct PhraseTerm<'a> {
    field_ord: u16,
    position: u32,
    cursor: Box<dyn PostingCursor + 'a>,
    snapshot_doc_freq: u64,
}

impl<'a> PhraseTerm<'a> {
    /// Bind one field, query position, and snapshot document frequency to a cursor.
    ///
    /// `field_ord` must be copied from the trusted posting-open path that
    /// produced `cursor`; construction validates it against snapshot STATS.
    #[must_use]
    pub fn new<C>(field_ord: u16, position: u32, cursor: C, snapshot_doc_freq: u64) -> Self
    where
        C: PostingCursor + 'a,
    {
        Self {
            field_ord,
            position,
            cursor: Box::new(cursor),
            snapshot_doc_freq,
        }
    }

    /// Stable schema field ordinal whose postings back this term.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// Analyzer position retained from the parsed phrase.
    #[must_use]
    pub const fn position(&self) -> u32 {
        self.position
    }

    /// Snapshot document frequency used in the phrase's fixed BM25 weight.
    #[must_use]
    pub const fn snapshot_doc_freq(&self) -> u64 {
        self.snapshot_doc_freq
    }
}

struct PhraseSlot<'a> {
    position: u32,
    alternatives: Vec<Box<dyn PostingCursor + 'a>>,
    positions: Vec<u32>,
    pub(in crate::argus) cost: u64,
    pub(in crate::argus) size_hint: u32,
}

impl<'a> PhraseSlot<'a> {
    fn new(position: u32, cursor: Box<dyn PostingCursor + 'a>) -> Self {
        let cost = cursor.cost();
        let size_hint = cursor.size_hint();
        Self {
            position,
            alternatives: vec![cursor],
            positions: Vec::new(),
            cost,
            size_hint,
        }
    }

    fn push_alternative(&mut self, cursor: Box<dyn PostingCursor + 'a>) -> Result<(), ArgusError> {
        self.alternatives
            .try_reserve(1)
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase alternatives",
                count: self.alternatives.len().saturating_add(1),
            })?;
        self.cost = self.cost.saturating_add(cursor.cost());
        self.alternatives.push(cursor);
        Ok(())
    }

    pub(in crate::argus) fn doc(&self) -> Option<u32> {
        self.alternatives
            .iter()
            .filter_map(|cursor| cursor.doc())
            .min()
    }

    fn seek(
        &mut self,
        target: u32,
        fieldnorms: &dyn FieldNormReader,
    ) -> Result<Option<u32>, ArgusError> {
        for cursor in &mut self.alternatives {
            let previous = cursor.doc();
            if previous.is_some_and(|doc| doc < target) {
                let moved = cursor.advance(target)?;
                validate_phrase_cursor_after_move(cursor.as_ref(), fieldnorms, moved)?;
                match (previous, moved) {
                    (None, Some(_)) => {
                        return Err(ArgusError::CursorInvariant(
                            "exhausted phrase cursor resurrected during advance",
                        ));
                    }
                    (_, Some(after)) if after < target => {
                        return Err(ArgusError::CursorInvariant(
                            "phrase advance landed below its requested target",
                        ));
                    }
                    _ => {}
                }
            }
        }
        Ok(self.doc())
    }

    fn next(&mut self, fieldnorms: &dyn FieldNormReader) -> Result<Option<u32>, ArgusError> {
        let Some(previous) = self.doc() else {
            return Ok(None);
        };
        for cursor in &mut self.alternatives {
            if cursor.doc() == Some(previous) {
                let moved = cursor.next()?;
                validate_phrase_cursor_after_move(cursor.as_ref(), fieldnorms, moved)?;
                if moved.is_some_and(|after| after <= previous) {
                    return Err(ArgusError::CursorInvariant(
                        "phrase next did not move to a strictly greater document",
                    ));
                }
            }
        }
        let moved = self.doc();
        if moved.is_some_and(|after| after <= previous) {
            return Err(ArgusError::CursorInvariant(
                "phrase slot did not move to a strictly greater document",
            ));
        }
        Ok(moved)
    }
}

fn validate_phrase_term_binding(
    term: &PhraseTerm<'_>,
    fieldnorms: &dyn FieldNormReader,
    snapshot: &Bm25FieldSnapshot,
    segment_num_docs: &mut Option<u32>,
) -> Result<(), ArgusError> {
    if term.field_ord != snapshot.field_ord() {
        return Err(ArgusError::PhraseTermFieldMismatch {
            term_field: term.field_ord,
            stats_field: snapshot.field_ord(),
        });
    }
    if term.snapshot_doc_freq > snapshot.doc_count() {
        return Err(ArgusError::InvalidDocFrequency {
            doc_freq: term.snapshot_doc_freq,
            doc_count: snapshot.doc_count(),
        });
    }
    let cursor_segment_num_docs = term.cursor.segment_num_docs();
    if u64::from(cursor_segment_num_docs) > snapshot.doc_count() {
        return Err(ArgusError::InvalidSnapshot {
            field_ord: snapshot.field_ord(),
            reason: "phrase segment num_docs exceeds snapshot BM25 document count",
        });
    }
    if segment_num_docs
        .replace(cursor_segment_num_docs)
        .is_some_and(|previous| previous != cursor_segment_num_docs)
    {
        return Err(ArgusError::CursorInvariant(
            "phrase cursors belong to different segment domains",
        ));
    }
    validate_phrase_cursor_state(term.cursor.as_ref(), fieldnorms)?;
    if term.cursor.doc().is_none() && (term.cursor.size_hint() != 0 || term.cursor.cost() != 0) {
        return Err(ArgusError::CursorInvariant(
            "an initially empty phrase cursor must have zero size and runtime cost",
        ));
    }
    if term.cursor.doc().is_some() && term.snapshot_doc_freq == 0 {
        return Err(ArgusError::InvalidSnapshot {
            field_ord: snapshot.field_ord(),
            reason: "a non-empty phrase cursor cannot have zero snapshot doc_freq",
        });
    }
    Ok(())
}

/// Phrase scorer over positioned posting cursors.
///
/// Candidate documents are intersected before positions are decoded. The
/// scorer owns and reuses every positional scratch buffer, so advancing after
/// warm-up performs no per-document allocation.
pub struct PhraseScorer<'a> {
    slots: Vec<PhraseSlot<'a>>,
    prefix_gate: Vec<PhraseSlot<'a>>,
    prefix_constant_score: bool,
    fieldnorms: Box<dyn FieldNormReader + 'a>,
    snapshot: Bm25FieldSnapshot,
    weight: f32,
    lead_slot: usize,
    cost: u64,
    size_hint: u32,
    pub(in crate::argus) segment_num_docs: u32,
    current: Option<u32>,
    current_frequency: u32,
    slop: u32,
    decode_scratch: Vec<u32>,
    position_indices: Vec<usize>,
    slop_positions: Vec<u32>,
    slop_costs: Vec<u32>,
    slop_positions_buffer: Vec<u32>,
    slop_costs_buffer: Vec<u32>,
    checkpoint: Option<Arc<dyn QueryWorkCheckpoint + 'a>>,
}

fn intersect_phrase_positions_with_slop(
    left_positions: &mut Vec<u32>,
    right_positions: &[u32],
    slop: u32,
    update_left: bool,
) -> usize {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut count = 0;
    let left_len = left_positions.len();
    while left_index < left_len && right_index < right_positions.len() {
        let left = left_positions[left_index];
        let right = right_positions[right_index];
        if left.abs_diff(right) <= slop {
            while left_index + 1 < left_len {
                let next_left = left_positions[left_index + 1];
                if next_left > right {
                    break;
                }
                left_index += 1;
            }
            if update_left {
                left_positions[count] = right;
            }
            count += 1;
            left_index += 1;
            right_index += 1;
        } else if left < right {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    if update_left {
        left_positions.truncate(count);
    }
    count
}

#[allow(clippy::too_many_arguments)]
pub(in crate::argus) fn intersect_phrase_positions_carrying_slop(
    left_positions: &mut Vec<u32>,
    left_slops: &mut Vec<u32>,
    right_positions: &[u32],
    max_slop: u32,
    update_left: bool,
    positions_buffer: &mut Vec<u32>,
    slops_buffer: &mut Vec<u32>,
) -> Result<u32, ArgusError> {
    if left_positions.is_empty() || right_positions.is_empty() {
        if update_left {
            left_positions.clear();
            left_slops.clear();
        }
        return Ok(0);
    }

    if update_left {
        positions_buffer.clear();
        slops_buffer.clear();
        let capacity = left_positions.len().saturating_add(right_positions.len());
        positions_buffer
            .try_reserve(capacity)
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase slop positions",
                count: capacity,
            })?;
        slops_buffer
            .try_reserve(capacity)
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase slop costs",
                count: capacity,
            })?;
    }

    let add = |value: (u32, u32),
               positions: &mut Vec<u32>,
               slops: &mut Vec<u32>|
     -> Result<(), ArgusError> {
        if update_left {
            if positions.last() == Some(&value.1) {
                let Some(last_slop) = slops.last_mut() else {
                    return Err(ArgusError::CursorInvariant(
                        "a retained slop position has no cost",
                    ));
                };
                *last_slop = (*last_slop).min(value.0);
            } else {
                positions.push(value.1);
                slops.push(value.0);
            }
        }
        Ok(())
    };

    let mut left_index = 0;
    let mut right_index = 0;
    let mut count = 0_u32;
    loop {
        let left = left_positions[left_index];
        let slop_so_far = left_slops.get(left_index).copied().unwrap_or(0);
        let right = right_positions[right_index];
        let distance = slop_so_far
            .checked_add(left.abs_diff(right))
            .filter(|distance| *distance <= max_slop);
        if let Some(distance) = distance {
            let (smaller, larger, mut smaller_index, smaller_positions) = if left < right {
                (left, right, left_index, left_positions.as_slice())
            } else {
                (right, left, right_index, right_positions)
            };
            let mut new_slop = distance;
            add((new_slop, smaller), positions_buffer, slops_buffer)?;
            while smaller_index + 1 < smaller_positions.len() {
                let next = smaller_positions[smaller_index + 1];
                if next > larger {
                    break;
                }
                smaller_index += 1;
                if let Some(candidate_slop) = slop_so_far.checked_add(next.abs_diff(larger)) {
                    new_slop = candidate_slop;
                    add((new_slop, next), positions_buffer, slops_buffer)?;
                }
            }
            add((new_slop, larger), positions_buffer, slops_buffer)?;
            count = count
                .checked_add(1)
                .ok_or(ArgusError::CursorInvariant("phrase frequency exceeds u32"))?;
            left_index += 1;
            right_index += 1;
        } else if left < right {
            left_index += 1;
        } else {
            right_index += 1;
        }

        if left_index >= left_positions.len() || right_index >= right_positions.len() {
            if left_index >= left_positions.len() {
                let Some(&left) = left_positions.last() else {
                    return Err(ArgusError::CursorInvariant(
                        "phrase slop positions became empty",
                    ));
                };
                let slop_so_far = left_slops.last().copied().unwrap_or(0);
                for &right in &right_positions[right_index..] {
                    if let Some(new_slop) = slop_so_far.checked_add(left.abs_diff(right))
                        && new_slop <= max_slop
                    {
                        add((new_slop, right), positions_buffer, slops_buffer)?;
                    }
                }
            } else {
                let Some(&right) = right_positions.last() else {
                    return Err(ArgusError::CursorInvariant(
                        "phrase slop right positions became empty",
                    ));
                };
                for (offset, &left) in left_positions[left_index..].iter().enumerate() {
                    let slop_index = left_index + offset;
                    let slop_so_far = left_slops.get(slop_index).copied().unwrap_or(0);
                    if let Some(new_slop) = slop_so_far.checked_add(left.abs_diff(right))
                        && new_slop <= max_slop
                    {
                        add((new_slop, left), positions_buffer, slops_buffer)?;
                    }
                }
            }
            break;
        }
    }

    if update_left {
        std::mem::swap(left_positions, positions_buffer);
        std::mem::swap(left_slops, slops_buffer);
        positions_buffer.clear();
        slops_buffer.clear();
    }
    Ok(count)
}

impl<'a> PhraseScorer<'a> {
    /// Build and initially align one exact phrase scorer.
    ///
    /// Query terms must be ordered by non-decreasing analyzer position and
    /// span at least two distinct positions. Equal-position terms form one OR
    /// slot. Each term needs its own forward-only positioned cursor, including
    /// repeated terms such as `a a b`.
    ///
    /// # Errors
    ///
    /// Rejects malformed positions, non-positioned cursors, field/domain
    /// drift, invalid snapshot frequencies or boosts, and bounded allocation
    /// failures.
    pub fn new<F>(
        terms: Vec<PhraseTerm<'a>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        Self::new_with_slop_and_checkpoint(terms, fieldnorms, snapshot, field_boost, 0, None)
    }

    /// Build and initially align one phrase scorer with a shared slop budget.
    ///
    /// Slop is accumulated across every adjacent query-term pair after query
    /// offsets are applied. Reordering is therefore possible but costs the
    /// distance moved in both directions, matching Tantivy's phrase-query
    /// contract. A zero budget retains the exact-adjacency fast path.
    ///
    /// # Errors
    ///
    /// Returns the same construction errors as [`Self::new`].
    pub fn new_with_slop<F>(
        terms: Vec<PhraseTerm<'a>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
        slop: u32,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        Self::new_with_slop_and_checkpoint(terms, fieldnorms, snapshot, field_boost, slop, None)
    }

    /// Build one exact phrase scorer with a coarse position-verification
    /// checkpoint.
    ///
    /// The checkpoint runs once per candidate document before position rows
    /// are inspected by the matcher. It is never called inside the tight
    /// position-merge loop.
    ///
    /// # Errors
    ///
    /// Returns the same construction errors as [`Self::new`], plus typed
    /// cancellation or fuel exhaustion from the initial alignment.
    pub fn new_with_checkpoint<F>(
        terms: Vec<PhraseTerm<'a>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
        checkpoint: Option<Arc<dyn QueryWorkCheckpoint + 'a>>,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        Self::new_with_slop_and_checkpoint(terms, fieldnorms, snapshot, field_boost, 0, checkpoint)
    }

    /// Build one phrase scorer with slop and a coarse position-verification
    /// checkpoint.
    ///
    /// # Errors
    ///
    /// Returns the same construction errors as [`Self::new`], plus typed
    /// cancellation or fuel exhaustion from the initial alignment.
    pub fn new_with_slop_and_checkpoint<F>(
        terms: Vec<PhraseTerm<'a>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
        max_slop: u32,
        checkpoint: Option<Arc<dyn QueryWorkCheckpoint + 'a>>,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        Self::new_internal(
            terms,
            None,
            fieldnorms,
            snapshot,
            field_boost,
            max_slop,
            checkpoint,
        )
    }

    /// Build a phrase-prefix scorer whose suffix alternatives gate membership
    /// without contributing to the fixed phrase's BM25 weight.
    ///
    /// The fixed terms must occupy at least one query position. Every suffix
    /// term must occupy the same final position at or after the fixed phrase.
    /// A one-position fixed phrase follows pinned Tantivy and scores every
    /// matching document as exactly `1.0`; longer fixed phrases retain the
    /// ordinary fixed-phrase BM25 score and frequency.
    ///
    /// # Errors
    ///
    /// Returns the same construction errors as [`Self::new`], plus malformed
    /// suffix-position or empty-suffix failures and typed cancellation/fuel
    /// exhaustion from initial alignment.
    pub fn new_phrase_prefix_with_checkpoint<F>(
        fixed_terms: Vec<PhraseTerm<'a>>,
        suffix_terms: Vec<PhraseTerm<'a>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
        checkpoint: Option<Arc<dyn QueryWorkCheckpoint + 'a>>,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        Self::new_internal(
            fixed_terms,
            Some(suffix_terms),
            fieldnorms,
            snapshot,
            field_boost,
            0,
            checkpoint,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_internal<F>(
        terms: Vec<PhraseTerm<'a>>,
        prefix_terms: Option<Vec<PhraseTerm<'a>>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
        max_slop: u32,
        checkpoint: Option<Arc<dyn QueryWorkCheckpoint + 'a>>,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        let is_phrase_prefix = prefix_terms.is_some();
        if terms.is_empty() {
            return Err(ArgusError::InvalidPhrase {
                reason: if is_phrase_prefix {
                    "a phrase prefix requires fixed positioned terms"
                } else {
                    "an exact phrase requires positioned terms"
                },
            });
        }
        if fieldnorms.field_ord() != snapshot.field_ord() {
            return Err(ArgusError::FieldMismatch {
                fieldnorm_field: fieldnorms.field_ord(),
                stats_field: snapshot.field_ord(),
            });
        }
        if !field_boost.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        let num_terms = u32::try_from(terms.len()).map_err(|_| ArgusError::InvalidPhrase {
            reason: "phrase term count exceeds u32",
        })?;
        let phrase_work_factor = num_terms.checked_mul(10).ok_or(ArgusError::InvalidPhrase {
            reason: "phrase term cost factor exceeds u32",
        })?;

        let mut slots: Vec<PhraseSlot<'a>> = Vec::new();
        slots
            .try_reserve_exact(terms.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase position slots",
                count: terms.len(),
            })?;
        let mut segment_num_docs = None;
        let mut idf_sum = 0.0_f32;
        for term in terms {
            validate_phrase_term_binding(&term, &fieldnorms, &snapshot, &mut segment_num_docs)?;
            idf_sum += idf(term.snapshot_doc_freq, snapshot.doc_count());
            if !idf_sum.is_finite() {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "phrase IDF sum is not finite",
                });
            }

            match slots.last_mut() {
                Some(slot) if term.position < slot.position => {
                    return Err(ArgusError::InvalidPhrase {
                        reason: "phrase positions must be non-decreasing",
                    });
                }
                Some(slot) if term.position == slot.position => {
                    slot.push_alternative(term.cursor)?;
                }
                _ => slots.push(PhraseSlot::new(term.position, term.cursor)),
            }
        }
        let mut prefix_gate: Vec<PhraseSlot<'a>> = Vec::new();
        if let Some(prefix_terms) = prefix_terms {
            if prefix_terms.is_empty() {
                return Err(ArgusError::InvalidPhrase {
                    reason: "a phrase prefix requires suffix alternatives",
                });
            }
            let final_fixed_position =
                slots
                    .last()
                    .map(|slot| slot.position)
                    .ok_or(ArgusError::InvalidPhrase {
                        reason: "a phrase prefix requires fixed positioned terms",
                    })?;
            prefix_gate
                .try_reserve_exact(1)
                .map_err(|_| ArgusError::Allocation {
                    resource: "phrase-prefix gate",
                    count: 1,
                })?;
            for term in prefix_terms {
                validate_phrase_term_binding(&term, &fieldnorms, &snapshot, &mut segment_num_docs)?;
                if term.position < final_fixed_position {
                    return Err(ArgusError::InvalidPhrase {
                        reason: "phrase-prefix suffix position precedes the fixed phrase",
                    });
                }
                match prefix_gate.last_mut() {
                    Some(gate) if term.position != gate.position => {
                        return Err(ArgusError::InvalidPhrase {
                            reason: "phrase-prefix suffix alternatives must share one position",
                        });
                    }
                    Some(gate) => gate.push_alternative(term.cursor)?,
                    None => prefix_gate.push(PhraseSlot::new(term.position, term.cursor)),
                }
            }
        }
        if slots.len() < 2 && prefix_gate.is_empty() {
            return Err(ArgusError::InvalidPhrase {
                reason: "an exact phrase must span at least two positions",
            });
        }
        let prefix_constant_score = !prefix_gate.is_empty() && slots.len() == 1;
        if !prefix_constant_score
            && slots.iter().any(|slot| {
                slot.alternatives
                    .iter()
                    .any(|cursor| cursor.doc().is_some())
            })
            && snapshot
                .average_field_length()
                .is_none_or(|average| average <= 0.0)
        {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "a scored phrase requires a positive raw average field length",
            });
        }

        let segment_num_docs = segment_num_docs.unwrap_or(0);
        for slot in &mut slots {
            if slot.alternatives.len() > 1 {
                slot.size_hint = estimate_union(
                    slot.alternatives.iter().map(|cursor| cursor.size_hint()),
                    segment_num_docs,
                );
            }
        }
        if let Some(gate) = prefix_gate.last_mut()
            && gate.alternatives.len() > 1
        {
            gate.size_hint = estimate_union(
                gate.alternatives.iter().map(|cursor| cursor.size_hint()),
                segment_num_docs,
            );
        }
        let lead_slot = slots
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| slot.cost)
            .map_or(0, |(index, _)| index);
        let mut cost_ordered_estimates = Vec::new();
        cost_ordered_estimates
            .try_reserve_exact(slots.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase cost estimates",
                count: slots.len(),
            })?;
        cost_ordered_estimates.extend(slots.iter().map(|slot| (slot.cost, slot.size_hint)));
        cost_ordered_estimates.sort_by_key(|(cost, _)| *cost);
        let candidate_estimate = estimate_intersection(
            cost_ordered_estimates.into_iter().map(|(_, size)| size),
            segment_num_docs,
        );
        let cost = u64::from(candidate_estimate) * u64::from(phrase_work_factor);
        let size_hint = candidate_estimate / phrase_work_factor;
        let mut position_indices = Vec::new();
        position_indices
            .try_reserve_exact(slots.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase position indices",
                count: slots.len(),
            })?;
        position_indices.resize(slots.len(), 0);

        let mut weight = idf_sum * (1.0 + BM25_K1);
        weight *= field_boost;
        if !weight.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        let mut scorer = Self {
            slots,
            prefix_gate,
            prefix_constant_score,
            fieldnorms: Box::new(fieldnorms),
            snapshot,
            weight,
            lead_slot,
            cost,
            size_hint,
            segment_num_docs,
            current: None,
            current_frequency: 0,
            slop: max_slop,
            decode_scratch: Vec::new(),
            position_indices,
            slop_positions: Vec::new(),
            slop_costs: Vec::new(),
            slop_positions_buffer: Vec::new(),
            slop_costs_buffer: Vec::new(),
            checkpoint,
        };
        scorer.align()?;
        Ok(scorer)
    }

    fn doc(&self) -> Option<u32> {
        self.current
    }

    pub(in crate::argus) fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        let fieldnorms = self.fieldnorms.as_ref();
        self.slots[self.lead_slot].next(fieldnorms)?;
        self.align()
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        let fieldnorms = self.fieldnorms.as_ref();
        self.slots[self.lead_slot].seek(target, fieldnorms)?;
        self.align()
    }

    fn align(&mut self) -> Result<Option<u32>, ArgusError> {
        self.current = None;
        self.current_frequency = 0;
        loop {
            let Some(mut target) = self.slots.iter().filter_map(PhraseSlot::doc).max() else {
                return Ok(None);
            };
            loop {
                let mut raised_target = false;
                for slot in &mut self.slots {
                    let Some(doc) = slot.seek(target, self.fieldnorms.as_ref())? else {
                        return Ok(None);
                    };
                    if doc > target {
                        target = doc;
                        raised_target = true;
                    }
                }
                if !raised_target && self.slots.iter().all(|slot| slot.doc() == Some(target)) {
                    break;
                }
            }

            let frequency = self.phrase_frequency(target)?;
            if frequency != 0 && self.phrase_prefix_matches(target)? {
                self.current = Some(target);
                self.current_frequency = frequency;
                return Ok(self.current);
            }
            let fieldnorms = self.fieldnorms.as_ref();
            self.slots[self.lead_slot].next(fieldnorms)?;
        }
    }

    fn phrase_frequency(&mut self, doc: u32) -> Result<u32, ArgusError> {
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint.admit(QueryWorkKind::PositionDocument, 1)?;
        }
        for slot_index in 0..self.slots.len() {
            self.decode_slot_positions(slot_index, doc)?;
        }
        if self.slop != 0 {
            return self.phrase_frequency_with_slop();
        }
        self.position_indices.fill(0);
        let first_position = self.slots[0].position;
        let (slots, position_indices) = (&self.slots, &mut self.position_indices);
        let mut frequency = 0_u32;
        for &base in &slots[0].positions {
            let mut matched = true;
            for slot_index in 1..slots.len() {
                let offset = slots[slot_index].position - first_position;
                let Some(target) = base.checked_add(offset) else {
                    matched = false;
                    break;
                };
                let positions = &slots[slot_index].positions;
                let position_index = &mut position_indices[slot_index];
                while positions
                    .get(*position_index)
                    .is_some_and(|position| *position < target)
                {
                    *position_index += 1;
                }
                if positions.get(*position_index) != Some(&target) {
                    matched = false;
                    break;
                }
            }
            if matched {
                frequency = frequency
                    .checked_add(1)
                    .ok_or(ArgusError::CursorInvariant("phrase frequency exceeds u32"))?;
            }
        }
        Ok(frequency)
    }

    fn phrase_prefix_matches(&mut self, doc: u32) -> Result<bool, ArgusError> {
        let Some(gate) = self.prefix_gate.first_mut() else {
            return Ok(true);
        };
        if gate.seek(doc, self.fieldnorms.as_ref())? != Some(doc) {
            return Ok(false);
        }
        decode_phrase_slot_positions(
            gate,
            &mut self.decode_scratch,
            self.snapshot.field_ord(),
            doc,
        )?;

        self.position_indices.fill(0);
        let first_position = self.slots[0].position;
        let suffix_offset =
            gate.position
                .checked_sub(first_position)
                .ok_or(ArgusError::InvalidPhrase {
                    reason: "phrase-prefix suffix position precedes the fixed phrase",
                })?;
        for &base in &self.slots[0].positions {
            let mut fixed_phrase_matches = true;
            for slot_index in 1..self.slots.len() {
                let offset = self.slots[slot_index].position - first_position;
                let Some(target) = base.checked_add(offset) else {
                    fixed_phrase_matches = false;
                    break;
                };
                let positions = &self.slots[slot_index].positions;
                let position_index = &mut self.position_indices[slot_index];
                while positions
                    .get(*position_index)
                    .is_some_and(|position| *position < target)
                {
                    *position_index += 1;
                }
                if positions.get(*position_index) != Some(&target) {
                    fixed_phrase_matches = false;
                    break;
                }
            }
            if fixed_phrase_matches
                && base
                    .checked_add(suffix_offset)
                    .is_some_and(|target| gate.positions.binary_search(&target).is_ok())
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn phrase_frequency_with_slop(&mut self) -> Result<u32, ArgusError> {
        let max_query_position = self
            .slots
            .last()
            .map(|slot| slot.position)
            .ok_or(ArgusError::CursorInvariant("phrase has no position slots"))?;
        for slot in &mut self.slots {
            let offset = max_query_position - slot.position;
            for position in &mut slot.positions {
                *position = position
                    .checked_add(offset)
                    .ok_or(ArgusError::CursorInvariant(
                        "phrase position plus query offset exceeds u32",
                    ))?;
            }
        }

        self.slop_positions.clear();
        let first_positions = &self.slots[0].positions;
        self.slop_positions
            .try_reserve(first_positions.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase slop positions",
                count: first_positions.len(),
            })?;
        self.slop_positions.extend_from_slice(first_positions);
        self.slop_costs.clear();

        for slot_index in 1..self.slots.len() - 1 {
            intersect_phrase_positions_carrying_slop(
                &mut self.slop_positions,
                &mut self.slop_costs,
                &self.slots[slot_index].positions,
                self.slop,
                true,
                &mut self.slop_positions_buffer,
                &mut self.slop_costs_buffer,
            )?;
            if self.slop_positions.is_empty() {
                return Ok(0);
            }
        }

        let final_positions = &self
            .slots
            .last()
            .ok_or(ArgusError::CursorInvariant("phrase has no final slot"))?
            .positions;
        if self.slots.len() == 2 {
            return u32::try_from(intersect_phrase_positions_with_slop(
                &mut self.slop_positions,
                final_positions,
                self.slop,
                false,
            ))
            .map_err(|_| ArgusError::CursorInvariant("phrase frequency exceeds u32"));
        }
        intersect_phrase_positions_carrying_slop(
            &mut self.slop_positions,
            &mut self.slop_costs,
            final_positions,
            self.slop,
            false,
            &mut self.slop_positions_buffer,
            &mut self.slop_costs_buffer,
        )
    }

    fn decode_slot_positions(&mut self, slot_index: usize, doc: u32) -> Result<(), ArgusError> {
        let (slots, decode_scratch) = (&mut self.slots, &mut self.decode_scratch);
        decode_phrase_slot_positions(
            &mut slots[slot_index],
            decode_scratch,
            self.snapshot.field_ord(),
            doc,
        )
    }

    pub(in crate::argus) fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.current.ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted phrase scorer",
        ))?;
        if self.current_frequency == 0 {
            return Err(ArgusError::CursorInvariant(
                "current phrase match has zero frequency",
            ));
        }
        if self.prefix_constant_score {
            return Ok(1.0);
        }
        let fieldnorm_id =
            self.fieldnorms
                .fieldnorm_id(doc)
                .ok_or_else(|| ArgusError::MissingFieldnorm {
                    field_ord: self.fieldnorms.field_ord(),
                    global_docid: doc,
                })?;
        let frequency = self.current_frequency as f32;
        let norm = self.snapshot.tf_cache[usize::from(fieldnorm_id)];
        finite_score(self.weight * (frequency / (frequency + norm)), doc)
    }
}

fn decode_phrase_slot_positions(
    slot: &mut PhraseSlot<'_>,
    decode_scratch: &mut Vec<u32>,
    field_ord: u16,
    doc: u32,
) -> Result<(), ArgusError> {
    slot.positions.clear();
    for cursor in &slot.alternatives {
        if cursor.doc() != Some(doc) {
            continue;
        }
        let frequency = cursor.freq().ok_or(ArgusError::CursorInvariant(
            "positioned phrase cursor has no frequency",
        ))?;
        let handle = cursor
            .positions_handle()
            .ok_or(ArgusError::MissingPositions {
                field_ord,
                global_docid: doc,
            })?;
        handle.decode_into(decode_scratch)?;
        let expected = usize::try_from(frequency)
            .map_err(|_| ArgusError::CursorInvariant("phrase frequency does not fit usize"))?;
        if decode_scratch.len() != expected {
            return Err(ArgusError::CursorInvariant(
                "decoded phrase position count differs from term frequency",
            ));
        }
        let required = slot.positions.len().saturating_add(decode_scratch.len());
        slot.positions
            .try_reserve(decode_scratch.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase slot positions",
                count: required,
            })?;
        slot.positions.extend_from_slice(decode_scratch);
    }
    slot.positions.sort_unstable();
    slot.positions.dedup();
    if slot.positions.is_empty() {
        return Err(ArgusError::CursorInvariant(
            "phrase candidate has no decoded positions",
        ));
    }
    Ok(())
}

fn validate_phrase_cursor_after_move(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
    moved_doc: Option<u32>,
) -> Result<(), ArgusError> {
    if cursor.doc() != moved_doc {
        return Err(ArgusError::CursorInvariant(
            "phrase movement result disagrees with current document",
        ));
    }
    validate_phrase_cursor_state(cursor, fieldnorms)
}

fn validate_phrase_cursor_state(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
) -> Result<(), ArgusError> {
    let Some(doc) = cursor.doc() else {
        if cursor.freq().is_some() || cursor.positions_handle().is_some() {
            return Err(ArgusError::CursorInvariant(
                "exhausted phrase cursor retained frequency or positions state",
            ));
        }
        return Ok(());
    };
    if cursor.freq().is_none_or(|frequency| frequency == 0) {
        return Err(ArgusError::CursorInvariant(
            "positioned phrase cursor has no positive frequency",
        ));
    }
    if cursor.positions_handle().is_none() {
        return Err(ArgusError::MissingPositions {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    if cursor.size_hint() == 0 || cursor.cost() == 0 {
        return Err(ArgusError::CursorInvariant(
            "a non-empty phrase cursor must have non-zero size and runtime cost",
        ));
    }
    if fieldnorms.fieldnorm_id(doc).is_none() {
        return Err(ArgusError::MissingFieldnorm {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    Ok(())
}
