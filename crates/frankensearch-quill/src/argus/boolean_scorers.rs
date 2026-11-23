#[cfg(feature = "pruning-conformance")]
use std::sync::Arc;

use crate::quiver::NumericDocIdSet;

#[cfg(feature = "pruning-conformance")]
use super::QueryWorkCheckpoint;
use super::reference_scorer::{
    ReferenceScorer, ScorerNode, SeekDangerResult, classify_seek_result,
};
use super::union_scorer::BufferedUnionScorer;
use super::{ArgusError, MAX_GLOBAL_DOCID_EXCLUSIVE};

pub(in crate::argus) fn reserve_scorers<'a>(
    resource: &'static str,
    count: usize,
) -> Result<Vec<ReferenceScorer<'a>>, ArgusError> {
    let mut scorers = Vec::new();
    scorers
        .try_reserve_exact(count)
        .map_err(|_| ArgusError::Allocation { resource, count })?;
    Ok(scorers)
}

pub(in crate::argus) fn remove_raw_unit_all_scorers<'a>(
    scorers: &mut Vec<ReferenceScorer<'a>>,
) -> Option<ReferenceScorer<'a>> {
    let mut retained_all = None;
    let mut index = 0;
    while index < scorers.len() {
        if scorers[index].is_raw_unit_all() {
            let all = scorers.remove(index);
            if retained_all.is_none() {
                retained_all = Some(all);
            }
        } else {
            index += 1;
        }
    }
    retained_all
}

pub(in crate::argus) fn shared_segment_num_docs(
    scorers: &[ReferenceScorer<'_>],
) -> Result<u32, ArgusError> {
    let segment_num_docs = scorers.first().map_or(0, ReferenceScorer::segment_num_docs);
    if scorers
        .iter()
        .any(|scorer| scorer.segment_num_docs() != segment_num_docs)
    {
        return Err(ArgusError::CursorInvariant(
            "Boolean children belong to different segment domains",
        ));
    }
    Ok(segment_num_docs)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the pinned Tantivy estimator clamps its finite non-negative result to u32 input bounds"
)]
pub(in crate::argus) fn estimate_intersection(
    mut sizes: impl Iterator<Item = u32>,
    segment_num_docs: u32,
) -> u32 {
    if segment_num_docs == 0 {
        return 0;
    }
    let Some(first) = sizes.next() else {
        return 0;
    };
    let mut co_location_factor = 1.3_f64;
    let mut estimate = f64::from(first);
    let mut smallest = estimate;
    for size in sizes {
        co_location_factor = (co_location_factor - 0.1).max(1.0);
        estimate *= (f64::from(size) / f64::from(segment_num_docs)) * co_location_factor;
        smallest = smallest.min(f64::from(size));
    }
    estimate.round().min(smallest) as u32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the pinned Tantivy estimator clamps its finite non-negative result to segment_num_docs"
)]
pub(in crate::argus) fn estimate_union(
    sizes: impl Iterator<Item = u32>,
    segment_num_docs: u32,
) -> u32 {
    if segment_num_docs == 0 {
        return 0;
    }
    let mut not_in_any_set_probability = 1.0_f64;
    for size in sizes {
        let probability = (f64::from(size) / f64::from(segment_num_docs)) * 0.8;
        not_in_any_set_probability *= 1.0 - probability;
    }
    let estimate = (f64::from(segment_num_docs) * (1.0 - not_in_any_set_probability)).round();
    estimate.min(f64::from(segment_num_docs)) as u32
}

pub(in crate::argus) fn scorer_intersection(
    mut scorers: Vec<ReferenceScorer<'_>>,
) -> Result<ReferenceScorer<'_>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "required scorer count changed during lowering",
        )),
        _ => {
            let subtree_nodes = scorers.iter().fold(1_usize, |count, scorer| {
                count.saturating_add(scorer.subtree_nodes)
            });
            let intersection = IntersectionScorer::new(scorers)?;
            if intersection.doc().is_none() {
                Ok(ReferenceScorer::empty())
            } else {
                Ok(ReferenceScorer {
                    node: ScorerNode::Intersection(intersection),
                    subtree_nodes,
                    #[cfg(feature = "pruning-conformance")]
                    construction_ordinal: None,
                })
            }
        }
    }
}

// `'a` is load-bearing under `pruning-conformance`, where the checkpoint
// parameter shares it; elision only type-checks with the feature off.
#[allow(clippy::elidable_lifetime_names)]
pub(in crate::argus) fn scorer_union<'a>(
    mut scorers: Vec<ReferenceScorer<'a>>,
    topdocs_root: bool,
    #[cfg(feature = "pruning-conformance")] conformance_checkpoint: Option<
        Arc<dyn QueryWorkCheckpoint + 'a>,
    >,
) -> Result<ReferenceScorer<'a>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "optional scorer count changed during lowering",
        )),
        _ => {
            let subtree_nodes = scorers.iter().fold(1_usize, |count, scorer| {
                count.saturating_add(scorer.subtree_nodes)
            });
            Ok(ReferenceScorer {
                node: ScorerNode::Union(BufferedUnionScorer::new(
                    scorers,
                    topdocs_root,
                    #[cfg(feature = "pruning-conformance")]
                    conformance_checkpoint,
                )?),
                subtree_nodes,
                #[cfg(feature = "pruning-conformance")]
                construction_ordinal: None,
            })
        }
    }
}

pub(in crate::argus) fn scorer_union_unscored(
    mut scorers: Vec<ReferenceScorer<'_>>,
) -> Result<ReferenceScorer<'_>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "unscored optional scorer count changed during lowering",
        )),
        _ => {
            let subtree_nodes = scorers.iter().fold(1_usize, |count, scorer| {
                count.saturating_add(scorer.subtree_nodes)
            });
            Ok(ReferenceScorer {
                node: ScorerNode::UnscoredUnion(UnscoredUnionScorer::new(scorers)?),
                subtree_nodes,
                #[cfg(feature = "pruning-conformance")]
                construction_ordinal: None,
            })
        }
    }
}

/// Allocation-free cursor over one physical segment document-id domain.
///
/// Unlike Tantivy's local-docid `AllScorer`, this cursor retains global bounds
/// and an explicit exhausted state so `u32::MAX` remains a real match.
pub(in crate::argus) struct AllScorer {
    current: Option<u32>,
    docid_hi: u64,
    span: u64,
    pub(in crate::argus) segment_num_docs: u32,
    score: f32,
}

impl AllScorer {
    pub(in crate::argus) fn new(
        docid_lo: u64,
        docid_hi: u64,
        segment_num_docs: u32,
        boost: f32,
    ) -> Result<Self, ArgusError> {
        if !boost.is_finite() {
            return Err(ArgusError::InvalidAllBoost {
                boost_bits: boost.to_bits(),
            });
        }
        let invalid = |reason| ArgusError::InvalidAllRange {
            docid_lo,
            docid_hi,
            segment_num_docs,
            reason,
        };
        if docid_lo > docid_hi {
            return Err(invalid("the lower bound exceeds the upper bound"));
        }
        if docid_hi > MAX_GLOBAL_DOCID_EXCLUSIVE {
            return Err(invalid("the upper bound exceeds the u32 payload domain"));
        }
        let span = docid_hi - docid_lo;
        if u64::from(segment_num_docs) > span {
            return Err(invalid("the live document count exceeds the range span"));
        }
        let current =
            if span == 0 {
                None
            } else {
                Some(u32::try_from(docid_lo).map_err(|_| {
                    invalid("the non-empty lower bound exceeds the u32 payload domain")
                })?)
            };
        Ok(Self {
            current,
            docid_hi,
            span,
            segment_num_docs,
            score: boost,
        })
    }

    pub(in crate::argus) const fn doc(&self) -> Option<u32> {
        self.current
    }

    pub(in crate::argus) const fn cost(&self) -> u64 {
        self.span
    }

    pub(in crate::argus) fn size_hint(&self) -> u32 {
        u32::try_from(self.span).unwrap_or(u32::MAX)
    }

    fn is_raw_unit_score(&self) -> bool {
        self.score.to_bits() == 1.0_f32.to_bits()
    }

    pub(in crate::argus) fn next(&mut self) -> Option<u32> {
        let current = self.current?;
        let next = u64::from(current) + 1;
        self.current = if next < self.docid_hi {
            u32::try_from(next).ok()
        } else {
            None
        };
        self.current
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Option<u32> {
        let current = self.current?;
        if current >= target {
            return Some(current);
        }
        self.current = (u64::from(target) < self.docid_hi).then_some(target);
        self.current
    }

    pub(in crate::argus) fn score(&self) -> Result<f32, ArgusError> {
        self.current
            .ok_or(ArgusError::CursorInvariant(
                "cannot score an exhausted All scorer",
            ))
            .map(|_| self.score)
    }
}

/// Owned lower-bound cursor over one materialized NUMERIC predicate.
///
/// Movement uses the exact sorted matches, while planning metadata mirrors
/// Tantivy's runtime fast-field range scorer. Keeping those concerns separate
/// preserves both filter-first seeking and the oracle's stable f32 score order.
pub(in crate::argus) struct NumericRangeScorer {
    docids: NumericRangeDocIds,
    index: usize,
    cost: u64,
    size_hint: u32,
    pub(in crate::argus) segment_num_docs: u32,
    covers_every_document: bool,
    score: f32,
}

impl NumericRangeScorer {
    pub(in crate::argus) fn new(
        docids: NumericDocIdSet,
        value_count: usize,
        segment_num_docs: u32,
        score: f32,
    ) -> Self {
        let covers_every_document = value_count
            == usize::try_from(segment_num_docs).unwrap_or(usize::MAX)
            && u32::try_from(docids.len()).unwrap_or(u32::MAX) == segment_num_docs;
        Self::from_docids(
            NumericRangeDocIds::Encoded(docids),
            segment_num_docs,
            covers_every_document,
            score,
        )
    }

    fn new_materialized(
        docids: Vec<u32>,
        segment_num_docs: u32,
        covers_every_document: bool,
        score: f32,
    ) -> Self {
        Self::from_docids(
            NumericRangeDocIds::Materialized(docids),
            segment_num_docs,
            covers_every_document,
            score,
        )
    }

    fn from_docids(
        docids: NumericRangeDocIds,
        segment_num_docs: u32,
        covers_every_document: bool,
        score: f32,
    ) -> Self {
        let (cost, size_hint) = if covers_every_document {
            (u64::from(segment_num_docs), segment_num_docs)
        } else {
            (tantivy_range_cost(segment_num_docs), segment_num_docs / 10)
        };
        Self {
            docids,
            index: 0,
            cost,
            size_hint,
            segment_num_docs,
            covers_every_document,
            score,
        }
    }

    pub(in crate::argus) fn doc(&self) -> Option<u32> {
        self.docids.as_slice().get(self.index).copied()
    }

    pub(in crate::argus) const fn cost(&self) -> u64 {
        self.cost
    }

    pub(in crate::argus) const fn size_hint(&self) -> u32 {
        self.size_hint
    }

    pub(in crate::argus) fn is_raw_unit_all(&self) -> bool {
        self.covers_every_document && self.score.to_bits() == 1.0_f32.to_bits()
    }

    pub(in crate::argus) fn next(&mut self) -> Option<u32> {
        if self.index < self.docids.len() {
            self.index += 1;
        }
        self.doc()
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Option<u32> {
        let tail = self.docids.as_slice().get(self.index..).unwrap_or_default();
        self.index += tail.partition_point(|&docid| docid < target);
        self.doc()
    }

    pub(in crate::argus) fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted numeric range",
        ))?;
        finite_score(self.score, doc)
    }
}

enum NumericRangeDocIds {
    Encoded(NumericDocIdSet),
    Materialized(Vec<u32>),
}

impl NumericRangeDocIds {
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Encoded(docids) => docids.as_slice(),
            Self::Materialized(docids) => docids,
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "contract parity copies Tantivy RangeDocSet::cost's nonnegative f64-to-u64 truncation"
)]
fn tantivy_range_cost(segment_num_docs: u32) -> u64 {
    (f64::from(segment_num_docs) * 0.8) as u64
}

/// Materialized constant-score cursor over one exact unique document set.
pub(in crate::argus) struct ConstantScoreScorer {
    pub(in crate::argus) docids: Vec<u32>,
    pub(in crate::argus) index: usize,
    pub(in crate::argus) segment_num_docs: u32,
    pub(in crate::argus) score: f32,
}

impl ConstantScoreScorer {
    pub(in crate::argus) fn doc(&self) -> Option<u32> {
        self.docids.get(self.index).copied()
    }

    pub(in crate::argus) fn cost(&self) -> u64 {
        u64::try_from(self.docids.len()).unwrap_or(u64::MAX)
    }

    pub(in crate::argus) fn size_hint(&self) -> u32 {
        u32::try_from(self.docids.len()).unwrap_or(u32::MAX)
    }

    pub(in crate::argus) const fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    pub(in crate::argus) fn next(&mut self) -> Option<u32> {
        if self.index < self.docids.len() {
            self.index += 1;
        }
        self.doc()
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Option<u32> {
        let tail = self.docids.get(self.index..).unwrap_or_default();
        self.index += tail.partition_point(|&docid| docid < target);
        self.doc()
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        let result = self.seek(target);
        classify_seek_result(target, result)
    }

    pub(in crate::argus) fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted constant-score predicate",
        ))?;
        finite_score(self.score, doc)
    }
}

/// Lazy scoreless disjunction used only by doc-set collection.
///
/// The scored parity path keeps [`BufferedUnionScorer`] unchanged. This cursor
/// advances all children at the current minimum document together, so it
/// deduplicates matches without evaluating or retaining any BM25 contribution.
pub(in crate::argus) struct UnscoredUnionScorer<'a> {
    pub(in crate::argus) active: Vec<ReferenceScorer<'a>>,
    current: Option<u32>,
    pub(in crate::argus) segment_num_docs: u32,
}

impl<'a> UnscoredUnionScorer<'a> {
    pub(in crate::argus) fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        scorers.retain(|scorer| scorer.doc().is_some());
        let current = scorers.iter().filter_map(ReferenceScorer::doc).min();
        Ok(Self {
            active: scorers,
            current,
            segment_num_docs,
        })
    }

    pub(in crate::argus) const fn doc(&self) -> Option<u32> {
        self.current
    }

    pub(in crate::argus) fn cost(&self) -> u64 {
        self.active
            .iter()
            .map(ReferenceScorer::cost)
            .fold(0_u64, u64::saturating_add)
    }

    pub(in crate::argus) fn size_hint(&self) -> u32 {
        estimate_union(
            self.active.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    fn refresh_current(&mut self) -> Option<u32> {
        self.active.retain(|scorer| scorer.doc().is_some());
        self.current = self.active.iter().filter_map(ReferenceScorer::doc).min();
        self.current
    }

    pub(in crate::argus) fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        for scorer in &mut self.active {
            if scorer.doc() == Some(current) {
                scorer.next()?;
            }
        }
        Ok(self.refresh_current())
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        for scorer in &mut self.active {
            if scorer.doc().is_some_and(|doc| doc < target) {
                scorer.advance(target)?;
            }
        }
        Ok(self.refresh_current())
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        let result = self.seek(target)?;
        classify_seek_result(target, result)
    }
}

pub(in crate::argus) struct IntersectionScorer<'a> {
    pub(in crate::argus) scorers: Vec<ReferenceScorer<'a>>,
    current: Option<u32>,
    pub(in crate::argus) segment_num_docs: u32,
}

impl<'a> IntersectionScorer<'a> {
    pub(in crate::argus) fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        scorers.sort_by_key(ReferenceScorer::cost);
        let mut scorer = Self {
            scorers,
            current: None,
            segment_num_docs,
        };
        scorer.align()?;
        Ok(scorer)
    }

    pub(in crate::argus) const fn doc(&self) -> Option<u32> {
        self.current
    }

    pub(in crate::argus) fn cost(&self) -> u64 {
        self.scorers.first().map_or(0, ReferenceScorer::cost)
    }

    pub(in crate::argus) fn size_hint(&self) -> u32 {
        estimate_intersection(
            self.scorers.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    fn align(&mut self) -> Result<Option<u32>, ArgusError> {
        let mut candidate = match self.scorers.iter().filter_map(ReferenceScorer::doc).max() {
            Some(candidate) if self.scorers.iter().all(|scorer| scorer.doc().is_some()) => {
                candidate
            }
            _ => {
                self.current = None;
                return Ok(None);
            }
        };
        'outer: loop {
            for scorer in &mut self.scorers {
                let Some(doc) = scorer.advance(candidate)? else {
                    self.current = None;
                    return Ok(None);
                };
                if doc > candidate {
                    candidate = doc;
                    continue 'outer;
                }
            }
            self.current = Some(candidate);
            return Ok(self.current);
        }
    }

    pub(in crate::argus) fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        let Some(mut candidate) = current.checked_add(1) else {
            self.scorers[0].next()?;
            self.current = None;
            return Ok(None);
        };

        'outer: loop {
            let Some(driver_doc) = self.scorers[0].advance(candidate)? else {
                self.current = None;
                return Ok(None);
            };
            candidate = driver_doc;
            for scorer in &mut self.scorers[1..] {
                match scorer.seek_danger(candidate)? {
                    SeekDangerResult::Found => {}
                    SeekDangerResult::SeekLowerBound(Some(lower_bound)) => {
                        if lower_bound <= candidate {
                            return Err(ArgusError::CursorInvariant(
                                "danger seek lower bound did not strictly advance",
                            ));
                        }
                        candidate = lower_bound;
                        continue 'outer;
                    }
                    SeekDangerResult::SeekLowerBound(None) => {
                        self.current = None;
                        return Ok(None);
                    }
                }
            }
            self.current = Some(candidate);
            return Ok(self.current);
        }
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        self.scorers[0].advance(target)?;
        self.align()
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        for scorer in &mut self.scorers {
            match scorer.seek_danger(target)? {
                SeekDangerResult::Found => {}
                lower_bound @ SeekDangerResult::SeekLowerBound(_) => return Ok(lower_bound),
            }
        }
        self.current = Some(target);
        Ok(SeekDangerResult::Found)
    }

    pub(in crate::argus) fn score(&mut self) -> Result<f32, ArgusError> {
        let doc = self.current.ok_or(ArgusError::CursorInvariant(
            "cannot score exhausted intersection",
        ))?;
        let (left_slice, rest) = self.scorers.split_at_mut(1);
        let left = left_slice[0].score()?;
        let (right_slice, others) = rest.split_at_mut(1);
        let right = right_slice[0].score()?;
        let mut other_sum = 0.0_f32;
        for scorer in others {
            other_sum += scorer.score()?;
        }
        finite_score(left + right + other_sum, doc)
    }
}

// Rank-safety argument for every pruning comparison:
//
// * each sealed/Delta leaf first recomputes its tf ceiling from conservative
//   `(max_frequency, min_fieldnorm)` metadata using the live snapshot avgdl and
//   the live idf/boost weight;
// * all contributions are non-negative, so the exact f64 sum is inflated by
//   the standard `n * EPSILON` forward-error envelope before rounding outward
//   to f32. The result therefore covers any scorer-vector addition order;
// * callers skip only when this ceiling is strictly below the running kth
//   score. Equality remains competitive because a smaller docid can win a tie.
//
// MaxScore may consequently omit a document that matches only the
// non-essential prefix, while BMW may skip an interval only after summing every
// term whose current doc can occur in that interval. Neither proof depends on
// the avgdl regime under which a sealed block was created.
pub(in crate::argus) fn conservative_bound_sum(
    bounds: impl IntoIterator<Item = f32>,
) -> Option<f32> {
    conservative_optional_bound_sum(bounds.into_iter().map(Some))
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the checked finite f64 envelope is deliberately rounded outward to f32"
)]
pub(in crate::argus) fn conservative_optional_bound_sum(
    bounds: impl IntoIterator<Item = Option<f32>>,
) -> Option<f32> {
    let mut exact = 0.0_f64;
    let mut count = 0_u32;
    for bound in bounds {
        let bound = bound?;
        if !bound.is_finite() || bound.is_sign_negative() {
            return None;
        }
        exact += f64::from(bound);
        count = count.checked_add(1)?;
    }
    if !exact.is_finite() || exact > f64::from(f32::MAX) {
        return None;
    }
    let rounding_budget = f64::from(count) * f64::from(f32::EPSILON);
    if rounding_budget >= 1.0 {
        return None;
    }
    let inflated = exact / (1.0 - rounding_budget);
    if !inflated.is_finite() || inflated > f64::from(f32::MAX) {
        return None;
    }
    let rounded = inflated as f32;
    if f64::from(rounded) >= inflated {
        Some(rounded)
    } else {
        Some(f32::from_bits(rounded.to_bits().checked_add(1)?))
    }
}

pub(in crate::argus) struct RequiredOptionalScorer<'a> {
    pub(in crate::argus) required: Box<ReferenceScorer<'a>>,
    pub(in crate::argus) optional: Box<ReferenceScorer<'a>>,
    score_cache: Option<(u32, f32)>,
    pub(in crate::argus) segment_num_docs: u32,
}

impl<'a> RequiredOptionalScorer<'a> {
    pub(in crate::argus) fn new(
        required: ReferenceScorer<'a>,
        optional: ReferenceScorer<'a>,
    ) -> Result<Self, ArgusError> {
        if required.segment_num_docs() != optional.segment_num_docs() {
            return Err(ArgusError::CursorInvariant(
                "required and optional scorers belong to different segment domains",
            ));
        }
        let segment_num_docs = required.segment_num_docs();
        Ok(Self {
            required: Box::new(required),
            optional: Box::new(optional),
            score_cache: None,
            segment_num_docs,
        })
    }

    pub(in crate::argus) fn doc(&self) -> Option<u32> {
        self.required.doc()
    }

    pub(in crate::argus) fn cost(&self) -> u64 {
        self.required.cost()
    }

    pub(in crate::argus) fn size_hint(&self) -> u32 {
        self.required.size_hint()
    }

    pub(in crate::argus) fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.score_cache = None;
        self.required.next()
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        self.score_cache = None;
        self.required.advance(target)
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        self.score_cache = None;
        self.required.seek_danger(target)
    }

    pub(in crate::argus) fn score(&mut self) -> Result<f32, ArgusError> {
        let doc = self.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score exhausted required-optional scorer",
        ))?;
        if let Some((cached_doc, score)) = self.score_cache
            && cached_doc == doc
        {
            return Ok(score);
        }
        let mut score = 0.0_f32;
        score += self.required.score()?;
        if self
            .optional
            .doc()
            .is_some_and(|optional_doc| optional_doc <= doc)
            && self.optional.advance(doc)? == Some(doc)
        {
            score += self.optional.score()?;
        }
        let score = finite_score(score, doc)?;
        self.score_cache = Some((doc, score));
        Ok(score)
    }
}

pub(in crate::argus) struct ExcludeScorer<'a> {
    pub(in crate::argus) include: Box<ReferenceScorer<'a>>,
    pub(in crate::argus) excluded: Vec<ReferenceScorer<'a>>,
    pub(in crate::argus) segment_num_docs: u32,
}

impl<'a> ExcludeScorer<'a> {
    pub(in crate::argus) fn new(
        include: ReferenceScorer<'a>,
        excluded: Vec<ReferenceScorer<'a>>,
    ) -> Result<Self, ArgusError> {
        let segment_num_docs = if include.is_explicit_empty() {
            shared_segment_num_docs(&excluded)?
        } else {
            include.segment_num_docs()
        };
        if !include.is_explicit_empty()
            && excluded
                .iter()
                .any(|scorer| scorer.segment_num_docs() != segment_num_docs)
        {
            return Err(ArgusError::CursorInvariant(
                "include and exclusion scorers belong to different segment domains",
            ));
        }
        let mut scorer = Self {
            include: Box::new(include),
            excluded,
            segment_num_docs,
        };
        scorer.skip_excluded()?;
        Ok(scorer)
    }

    pub(in crate::argus) fn doc(&self) -> Option<u32> {
        self.include.doc()
    }

    pub(in crate::argus) fn cost(&self) -> u64 {
        u64::from(self.size_hint())
    }

    pub(in crate::argus) fn size_hint(&self) -> u32 {
        self.include.size_hint()
    }

    fn contains(&mut self, doc: u32) -> Result<bool, ArgusError> {
        for scorer in &mut self.excluded {
            if scorer.seek_danger(doc)? == SeekDangerResult::Found {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn skip_excluded(&mut self) -> Result<Option<u32>, ArgusError> {
        while let Some(doc) = self.include.doc() {
            if !self.contains(doc)? {
                return Ok(Some(doc));
            }
            self.include.next()?;
        }
        Ok(None)
    }

    pub(in crate::argus) fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.include.doc().is_none() {
            return Ok(None);
        }
        self.include.next()?;
        self.skip_excluded()
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.include.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.include.doc());
        }
        self.include.advance(target)?;
        self.skip_excluded()
    }

    pub(in crate::argus) fn score(&mut self) -> Result<f32, ArgusError> {
        self.include.score()
    }
}

pub(in crate::argus) fn finite_score(score: f32, global_docid: u32) -> Result<f32, ArgusError> {
    if score.is_finite() {
        Ok(score)
    } else {
        Err(ArgusError::NonFiniteScore { global_docid })
    }
}
