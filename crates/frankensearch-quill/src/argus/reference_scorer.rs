use std::ops::Bound;
#[cfg(feature = "pruning-conformance")]
use std::sync::Arc;

use crate::quiver::{NumericField, NumericValue};

#[cfg(feature = "pruning-conformance")]
use super::QueryWorkCheckpoint;
use super::boolean_scorers::{
    AllScorer, ConstantScoreScorer, ExcludeScorer, IntersectionScorer, NumericRangeScorer,
    RequiredOptionalScorer, UnscoredUnionScorer, remove_raw_unit_all_scorers, reserve_scorers,
    scorer_intersection, scorer_union, scorer_union_unscored,
};
use super::collector::{CollectorState, DocSetCollector, LiveDocs, ScoredDoc, TopDocsCollector};
use super::cursor::PostingCursor;
#[cfg(test)]
use super::cursor::TermCursor;
use super::phrase_scorer::PhraseScorer;
use super::term_scorer::{CompetitiveTermCursor, TermRecordOption, TermScorer};
#[cfg(feature = "pruning-conformance")]
use super::union_scorer::ConformanceUnionRefill;
use super::union_scorer::{BufferedUnionScorer, PruningTelemetry, UnionPruningStats};
use super::{ArgusError, Occur};

/// One lowered scorer child and its canonical query occurrence.
pub struct ScorerClause<'a> {
    pub(in crate::argus) occur: Occur,
    pub(in crate::argus) scorer: ReferenceScorer<'a>,
}

impl<'a> ScorerClause<'a> {
    /// Build a lowered scorer child.
    #[must_use]
    pub const fn new(occur: Occur, scorer: ReferenceScorer<'a>) -> Self {
        Self { occur, scorer }
    }

    /// Build a required positive child.
    #[must_use]
    pub const fn must(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::Must, scorer)
    }

    /// Build an optional positive child.
    #[must_use]
    pub const fn should(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::Should, scorer)
    }

    /// Build a scoreless exclusion child.
    #[must_use]
    pub const fn must_not(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::MustNot, scorer)
    }
}

pub(in crate::argus) enum ScorerNode<'a> {
    Empty,
    All(AllScorer),
    Term(TermScorer<'a>),
    Phrase(PhraseScorer<'a>),
    NumericRange(NumericRangeScorer),
    ConstantScore(ConstantScoreScorer),
    Intersection(IntersectionScorer<'a>),
    Union(BufferedUnionScorer<'a>),
    UnscoredUnion(UnscoredUnionScorer<'a>),
    RequiredOptional(RequiredOptionalScorer<'a>),
    Exclude(ExcludeScorer<'a>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::argus) enum BooleanMode {
    Scored,
    Unscored,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::argus) enum SeekDangerResult {
    Found,
    SeekLowerBound(Option<u32>),
}

pub(in crate::argus) fn classify_seek_result(
    target: u32,
    result: Option<u32>,
) -> Result<SeekDangerResult, ArgusError> {
    match result {
        Some(doc) if doc == target => Ok(SeekDangerResult::Found),
        Some(doc) if doc > target => Ok(SeekDangerResult::SeekLowerBound(Some(doc))),
        None => Ok(SeekDangerResult::SeekLowerBound(None)),
        Some(_) => Err(ArgusError::CursorInvariant(
            "danger seek landed below its requested target",
        )),
    }
}

/// Stateful exhaustive scorer tree.
///
/// Construct one tree per segment and consume it with [`Self::top_k`]. The
/// object intentionally retains the unpruned path for differential testing of
/// future WAND and block-max implementations.
pub struct ReferenceScorer<'a> {
    pub(in crate::argus) node: ScorerNode<'a>,
    pub(in crate::argus) subtree_nodes: usize,
    /// Stable identity assigned when a buffered union adopts this child
    /// (bd-rtnwu); `None` outside a traced union.
    #[cfg(feature = "pruning-conformance")]
    pub(in crate::argus) construction_ordinal: Option<u32>,
}

#[cfg(feature = "pruning-conformance")]
impl ReferenceScorer<'_> {
    /// Cursor-kind label for trace evidence (bd-rtnwu).
    pub(in crate::argus) const fn trace_node_kind(&self) -> &'static str {
        match &self.node {
            ScorerNode::Empty => "empty",
            ScorerNode::All(_) => "all",
            ScorerNode::Term(_) => "term",
            ScorerNode::Phrase(_) => "phrase",
            ScorerNode::NumericRange(_) => "numeric_range",
            ScorerNode::ConstantScore(_) => "constant_score",
            ScorerNode::Intersection(_) => "intersection",
            ScorerNode::Union(_) => "union",
            ScorerNode::UnscoredUnion(_) => "unscored_union",
            ScorerNode::RequiredOptional(_) => "required_optional",
            ScorerNode::Exclude(_) => "exclude",
        }
    }
}

fn take_scorer_node(mut scorer: ReferenceScorer<'_>) -> ScorerNode<'_> {
    std::mem::replace(&mut scorer.node, ScorerNode::Empty)
}

impl Drop for ReferenceScorer<'_> {
    fn drop(&mut self) {
        let root = std::mem::replace(&mut self.node, ScorerNode::Empty);
        if matches!(root, ScorerNode::Empty) {
            return;
        }
        let mut pending = Vec::new();
        if pending.try_reserve_exact(self.subtree_nodes).is_err() {
            // Recursive destruction is not an admissible OOM fallback for an
            // adversarial scorer shape. Leaking during process-wide allocation
            // failure is preferable to overflowing the native stack.
            std::mem::forget(root);
            return;
        }
        pending.push(root);
        while let Some(node) = pending.pop() {
            match node {
                ScorerNode::Intersection(scorer) => {
                    for child in scorer.scorers {
                        pending.push(take_scorer_node(child));
                    }
                }
                ScorerNode::Union(scorer) => {
                    for child in scorer.active {
                        pending.push(take_scorer_node(child));
                    }
                }
                ScorerNode::UnscoredUnion(scorer) => {
                    for child in scorer.active {
                        pending.push(take_scorer_node(child));
                    }
                }
                ScorerNode::RequiredOptional(scorer) => {
                    pending.push(take_scorer_node(*scorer.required));
                    pending.push(take_scorer_node(*scorer.optional));
                }
                ScorerNode::Exclude(scorer) => {
                    pending.push(take_scorer_node(*scorer.include));
                    for child in scorer.excluded {
                        pending.push(take_scorer_node(child));
                    }
                }
                ScorerNode::Empty
                | ScorerNode::All(_)
                | ScorerNode::Term(_)
                | ScorerNode::Phrase(_)
                | ScorerNode::NumericRange(_)
                | ScorerNode::ConstantScore(_) => {}
            }
        }
    }
}

impl<'a> ReferenceScorer<'a> {
    /// Build a scorer that never matches.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            node: ScorerNode::Empty,
            subtree_nodes: 1,
            #[cfg(feature = "pruning-conformance")]
            construction_ordinal: None,
        }
    }

    /// Build a unit-score scorer over one half-open global document-id range.
    ///
    /// The range is the segment's physical document domain. It may include
    /// holes or tombstoned rows; [`LiveDocs`] remains the single visibility
    /// authority during collection. `segment_num_docs` is retained separately
    /// for Tantivy-compatible composite estimates and therefore may be smaller
    /// than the physical range span.
    ///
    /// The bounds use `u64` so `docid_hi == 2^32` can represent a final
    /// searchable `u32::MAX` document. An empty range with zero live documents
    /// lowers to [`Self::empty`].
    ///
    /// # Errors
    ///
    /// Rejects reversed ranges, bounds outside the `u32` payload domain, and a
    /// live count larger than the physical range span.
    pub fn all(docid_lo: u64, docid_hi: u64, segment_num_docs: u32) -> Result<Self, ArgusError> {
        Self::all_with_boost(docid_lo, docid_hi, segment_num_docs, 1.0)
    }

    /// Build a boosted match-all scorer over one half-open global document-id range.
    ///
    /// This is the exact scorer-level lowering for `Boost(All)`: every current
    /// document receives `boost`, while movement and unscored collection never
    /// evaluate it.
    ///
    /// # Errors
    ///
    /// Returns the same range failures as [`Self::all`] and rejects a non-finite
    /// boost before exposing the scorer.
    pub fn all_with_boost(
        docid_lo: u64,
        docid_hi: u64,
        segment_num_docs: u32,
        boost: f32,
    ) -> Result<Self, ArgusError> {
        let all = AllScorer::new(docid_lo, docid_hi, segment_num_docs, boost)?;
        if all.doc().is_none() {
            Ok(Self::empty())
        } else {
            Ok(Self {
                node: ScorerNode::All(all),
                subtree_nodes: 1,
                #[cfg(feature = "pruning-conformance")]
                construction_ordinal: None,
            })
        }
    }

    /// Wrap one BM25 term leaf.
    #[must_use]
    pub fn term(term: TermScorer<'a>) -> Self {
        if term.doc().is_none() {
            Self::empty()
        } else {
            Self {
                node: ScorerNode::Term(term),
                subtree_nodes: 1,
                #[cfg(feature = "pruning-conformance")]
                construction_ordinal: None,
            }
        }
    }

    /// Wrap one exact-adjacency phrase leaf.
    #[must_use]
    pub fn phrase(phrase: PhraseScorer<'a>) -> Self {
        if phrase.doc().is_none() {
            Self::empty()
        } else {
            Self {
                node: ScorerNode::Phrase(phrase),
                subtree_nodes: 1,
                #[cfg(feature = "pruning-conformance")]
                construction_ordinal: None,
            }
        }
    }

    /// Compile one NUMERIC field range into an owned constant-score doc set.
    ///
    /// A sealed segment has two legitimate document counts, and this leaf needs
    /// both:
    ///
    /// - `column_num_docs` is the **physical at-seal** count, including
    ///   tombstoned rows. The persisted column stores one value per row that
    ///   existed at seal time, so this is the only count against which its
    ///   cardinality can be validated.
    /// - `live_num_docs` is the **current snapshot** count, and becomes the
    ///   scorer's segment domain. Term and `All` leaves are built on the live
    ///   count, and `shared_segment_num_docs` requires every Boolean child to
    ///   agree, so a range leaf carrying the at-seal count made an ordinary
    ///   filtered query fail with `Boolean children belong to different segment
    ///   domains` as soon as the segment acquired a tombstone (#49).
    ///
    /// The two are equal until a manifest tombstones a document without
    /// compacting the segment, which is why passing one value for both went
    /// unnoticed on freshly built indexes.
    ///
    /// # Errors
    ///
    /// Returns a typed numeric-codec error for a bound whose signedness does
    /// not match the field or if the bounded docid set cannot be allocated.
    /// It also rejects a field value count larger than `column_num_docs`.
    pub fn numeric_range(
        field: NumericField<'_>,
        lower: Bound<NumericValue>,
        upper: Bound<NumericValue>,
        column_num_docs: u32,
        live_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        Self::numeric_range_with_boost(field, lower, upper, column_num_docs, live_num_docs, 1.0)
    }

    /// Compile one boosted NUMERIC field range into an owned constant-score doc set.
    ///
    /// Movement, cost, and size estimates are identical to [`Self::numeric_range`].
    /// Each current match scores exactly `boost`, preserving `Boost(Range)` without
    /// introducing a synthetic Boolean child or changing score accumulation order.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::numeric_range`] and rejects a
    /// non-finite boost before exposing the scorer.
    pub fn numeric_range_with_boost(
        field: NumericField<'_>,
        lower: Bound<NumericValue>,
        upper: Bound<NumericValue>,
        column_num_docs: u32,
        live_num_docs: u32,
        boost: f32,
    ) -> Result<Self, ArgusError> {
        if !boost.is_finite() {
            return Err(ArgusError::InvalidConstantBoost {
                boost_bits: boost.to_bits(),
            });
        }
        let value_count = field.len();
        if value_count > usize::try_from(column_num_docs).unwrap_or(usize::MAX) {
            return Err(ArgusError::InvalidNumericCardinality {
                field_ord: field.field_ord(),
                value_count,
                segment_num_docs: column_num_docs,
            });
        }
        let docids = field.range_docids(lower, upper)?;
        if docids.is_empty() {
            return Ok(Self::empty());
        }
        Ok(Self {
            node: ScorerNode::NumericRange(NumericRangeScorer::new(
                docids,
                value_count,
                live_num_docs,
                boost,
            )),
            subtree_nodes: 1,
            #[cfg(feature = "pruning-conformance")]
            construction_ordinal: None,
        })
    }

    /// Build one numeric-range scorer from an already materialized sorted doc set.
    ///
    /// The index layer binary-searches the persisted NUMERIC pair run, filters
    /// snapshot-visible documents with cancellation checks, and supplies the
    /// exact matches here. Planning metadata and constant-score behavior remain
    /// identical to the direct NUMERIC range constructor.
    pub(crate) fn materialized_numeric_range(
        field_ord: u16,
        docids: Vec<u32>,
        value_count: usize,
        column_num_docs: u32,
        live_num_docs: u32,
        covers_every_document: bool,
        boost: f32,
    ) -> Result<Self, ArgusError> {
        if !boost.is_finite() {
            return Err(ArgusError::InvalidConstantBoost {
                boost_bits: boost.to_bits(),
            });
        }
        if value_count > usize::try_from(column_num_docs).unwrap_or(usize::MAX) {
            return Err(ArgusError::InvalidNumericCardinality {
                field_ord,
                value_count,
                segment_num_docs: column_num_docs,
            });
        }
        if docids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ArgusError::CursorInvariant(
                "materialized numeric range docids are not strictly ascending",
            ));
        }
        if docids.is_empty() {
            return Ok(Self::empty());
        }
        Ok(Self {
            node: ScorerNode::NumericRange(NumericRangeScorer::new_materialized(
                docids,
                live_num_docs,
                covers_every_document,
                boost,
            )),
            subtree_nodes: 1,
            #[cfg(feature = "pruning-conformance")]
            construction_ordinal: None,
        })
    }

    /// Build the same constant-score node [`Self::constant_score`] produces,
    /// from a caller-collected document-id set instead of a drained child.
    ///
    /// The set must be strictly increasing — the exact invariant the drain
    /// enforces on its child — and every member must be a live global document
    /// id as reported by the validated cursors that produced it. This is the
    /// bulk companion to the drain: the glob lowering collects matching doc
    /// ids directly from the posting cursors (one tight merge walk, no
    /// per-document scorer dispatch), so a common-term glob arm costs its
    /// decode work instead of a full union-driven materialization.
    ///
    /// # Errors
    ///
    /// Rejects a non-finite score or a set that is not strictly increasing.
    pub fn constant_score_from_docids(
        docids: Vec<u32>,
        segment_num_docs: u32,
        score: f32,
    ) -> Result<Self, ArgusError> {
        if !score.is_finite() {
            return Err(ArgusError::InvalidConstantBoost {
                boost_bits: score.to_bits(),
            });
        }
        for pair in docids.windows(2) {
            if pair[0] >= pair[1] {
                return Err(ArgusError::CursorInvariant(
                    "constant-score docids did not advance monotonically",
                ));
            }
        }
        if docids.is_empty() {
            return Ok(Self::empty());
        }
        Ok(Self {
            node: ScorerNode::ConstantScore(ConstantScoreScorer {
                docids,
                index: 0,
                segment_num_docs,
                score,
            }),
            subtree_nodes: 1,
            #[cfg(feature = "pruning-conformance")]
            construction_ordinal: None,
        })
    }

    /// Replace one movement-only scorer tree's score with a finite constant.
    ///
    /// This is the canonical lowering for predicates such as a bounded glob:
    /// the child is materialized into its exact unique document set, so each
    /// document contributes the constant exactly once and the resulting cost
    /// is the true unique-match cardinality. The child must be safe for
    /// scoreless movement so no discarded BM25 arithmetic can affect traversal
    /// state.
    ///
    /// # Errors
    ///
    /// Rejects a non-finite score or a child tree that requires scoring while
    /// advancing.
    pub fn constant_score(mut inner: Self, score: f32) -> Result<Self, ArgusError> {
        if !score.is_finite() {
            return Err(ArgusError::InvalidConstantBoost {
                boost_bits: score.to_bits(),
            });
        }
        if inner.is_explicit_empty() {
            return Ok(Self::empty());
        }
        if !inner.is_unscored_safe() {
            return Err(ArgusError::ScoredTreeForUnscoredCollector);
        }
        let segment_num_docs = inner.segment_num_docs();
        let mut docids = Vec::new();
        while let Some(doc) = inner.doc() {
            if docids.last().is_some_and(|previous| *previous >= doc) {
                return Err(ArgusError::CursorInvariant(
                    "constant-score child did not advance monotonically",
                ));
            }
            if docids.len() == docids.capacity() {
                docids.try_reserve(1).map_err(|_| ArgusError::Allocation {
                    resource: "constant-score docids",
                    count: docids.len().saturating_add(1),
                })?;
            }
            docids.push(doc);
            inner.next()?;
        }
        Ok(Self {
            node: ScorerNode::ConstantScore(ConstantScoreScorer {
                docids,
                index: 0,
                segment_num_docs,
                score,
            }),
            subtree_nodes: 1,
            #[cfg(feature = "pruning-conformance")]
            construction_ordinal: None,
        })
    }

    pub(in crate::argus) fn is_explicit_empty(&self) -> bool {
        matches!(self.node, ScorerNode::Empty)
    }

    pub(in crate::argus) fn is_raw_unit_all(&self) -> bool {
        matches!(
            &self.node,
            ScorerNode::All(scorer) if scorer.is_raw_unit_score()
        ) || matches!(
            &self.node,
            ScorerNode::NumericRange(scorer) if scorer.is_raw_unit_all()
        )
    }

    pub(in crate::argus) fn is_unscored_safe(&self) -> bool {
        match &self.node {
            ScorerNode::Empty
            | ScorerNode::All(_)
            | ScorerNode::Term(_)
            | ScorerNode::Phrase(_)
            | ScorerNode::NumericRange(_) => true,
            ScorerNode::ConstantScore(_) => true,
            ScorerNode::Intersection(scorer) => scorer.scorers.iter().all(Self::is_unscored_safe),
            ScorerNode::Union(_) => false,
            ScorerNode::UnscoredUnion(scorer) => scorer.active.iter().all(Self::is_unscored_safe),
            ScorerNode::RequiredOptional(scorer) => {
                scorer.required.is_unscored_safe() && scorer.optional.is_unscored_safe()
            }
            ScorerNode::Exclude(scorer) => {
                scorer.include.is_unscored_safe()
                    && scorer.excluded.iter().all(Self::is_unscored_safe)
            }
        }
    }

    pub(in crate::argus) fn competitive_score_upper_bound(&self) -> Option<f32> {
        match &self.node {
            ScorerNode::Term(term) => term.term_score_upper_bound,
            // A pure-term group (nested multi-field union) reports its ceiling
            // as the sum of its terms' ceilings. Behavior-neutral today: the
            // sole consumer (`competitive_candidates`) only reads this behind
            // the all-direct-terms gate, which a union child never satisfies;
            // the deferred grouped-MaxScore follow-up will consume it directly.
            ScorerNode::Union(union) => union.group_upper_bound(),
            _ => None,
        }
    }

    pub(in crate::argus) fn competitive_supports_block_max(&self) -> bool {
        matches!(
            &self.node,
            ScorerNode::Term(term)
                if term.term_score_upper_bound.is_some() && term.cursor.supports_block_max()
        )
    }

    pub(in crate::argus) fn competitive_is_direct_term(&self) -> bool {
        matches!(&self.node, ScorerNode::Term(_))
    }

    pub(in crate::argus) fn is_tantivy_topdocs_frequency_term(&self) -> bool {
        matches!(
            &self.node,
            ScorerNode::Term(term) if term.record_option != TermRecordOption::Basic
        )
    }

    pub(in crate::argus) fn competitive_term_cursor(&self) -> Option<CompetitiveTermCursor<'_>> {
        match &self.node {
            ScorerNode::Term(term) => term.competitive_cursor(),
            _ => None,
        }
    }

    /// Inclusive last document this scorer can still reach, when its storage
    /// can answer from validated metadata alone.
    ///
    /// Only a direct term answers: the doc-major arrival order this feeds is
    /// defined over a term union, and every other node would have to derive a
    /// reachable bound from children it does not own.
    pub(in crate::argus) fn arrival_final_doc(&self) -> Option<u32> {
        match &self.node {
            ScorerNode::Term(term) => term.cursor.arrival_final_doc(),
            _ => None,
        }
    }

    /// Greatest document this scorer holds strictly below `target`.
    ///
    /// The receipt is redeemed against the document that was actually probed,
    /// so a cursor replaying an older answer is refused here rather than
    /// believed.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant error if the clause is not a direct term or
    /// answers a document other than `target`, the cursor's refusal when it
    /// does not implement validated arrival reads, and propagates its typed
    /// storage or admission failure.
    pub(in crate::argus) fn arrival_doc_before(
        &mut self,
        target: u32,
    ) -> Result<Option<u32>, ArgusError> {
        match &mut self.node {
            ScorerNode::Term(term) => term.cursor.arrival_doc_before(target),
            _ => Err(ArgusError::CursorInvariant(
                "arrival predecessor read from a clause that is not a direct term",
            )),
        }
    }

    /// A pure-term group eligible for grouped `MaxScore`: a union whose active
    /// children are all bounded direct terms, so it exposes a conservative
    /// whole-group ceiling. Deferred grouped-`MaxScore` foundation
    /// (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
    pub(in crate::argus) fn competitive_is_groupable(&self) -> bool {
        matches!(&self.node, ScorerNode::Union(union) if union.group_upper_bound().is_some())
    }

    /// Append this group's competitive documents in `[_, horizon_end)` to
    /// `docs`, enumerated through fresh per-term competitive cursors so the
    /// group's own scan state is left untouched (mirrors how the term
    /// `MaxScore` path uses [`CompetitiveTermCursor`]). Returns `false`
    /// (fail-closed) when the scorer is not a pure-term group or any child
    /// lacks a competitive cursor, so the caller falls back to exhaustive.
    pub(in crate::argus) fn group_competitive_docs(
        &self,
        horizon_end: u64,
        docs: &mut Vec<u32>,
    ) -> Result<bool, ArgusError> {
        let ScorerNode::Union(union) = &self.node else {
            return Ok(false);
        };
        // A group's remaining documents are NOT `union.active`. The group is
        // itself a buffered union that has already scored a window and drained
        // the corresponding children, so its remaining set is the union of
        // three disjoint parts. Enumerating only the third silently drops the
        // first two from the root's candidate set (`bd-bt2t`).

        // (1) The document the group is currently positioned on.
        if let Some(doc) = union.current
            && u64::from(doc) < horizon_end
        {
            docs.push(doc);
        }

        // (2) Buffered residue: scored but not yet consumed by `next()`.
        // Covers [group window_start, group horizon).
        if let Some(window_start) = union.window_start {
            for (offset, score) in union
                .score_window
                .iter()
                .enumerate()
                .skip(union.scan_offset)
            {
                if score.is_none() {
                    continue;
                }
                let doc = u64::from(window_start).saturating_add(offset as u64);
                if doc >= horizon_end {
                    break;
                }
                docs.push(u32::try_from(doc).map_err(|_| {
                    ArgusError::CursorInvariant("buffered union residue docid does not fit u32")
                })?);
            }
        }

        // (3) Live children, which the group's own fill already advanced to at
        // least its horizon — so this cannot overlap (2).
        for child in &union.active {
            let Some(mut cursor) = child.competitive_term_cursor() else {
                return Ok(false);
            };
            while let Some(doc) = cursor.doc() {
                if u64::from(doc) >= horizon_end {
                    break;
                }
                docs.push(doc);
                cursor.next()?;
            }
        }
        Ok(true)
    }

    pub(in crate::argus) fn union_pruning_stats(&self) -> Option<UnionPruningStats> {
        let ScorerNode::Union(union) = &self.node else {
            return None;
        };
        Some(union.pruning_stats)
    }

    /// Whether this clause reads arrival history from the crate's own cursor.
    ///
    /// Test-only, and deliberately structural rather than behavioural: it is
    /// how a lowering test proves the *shipping* path reaches
    /// [`TermCursor::Validated`], which is the half of the property a fixture
    /// built inside `argus` cannot establish for itself.
    #[cfg(test)]
    pub(crate) fn arrival_path_is_validated(&self) -> bool {
        match &self.node {
            ScorerNode::Term(term) => matches!(term.cursor, TermCursor::Validated(_)),
            _ => false,
        }
    }

    /// Return privacy-safe aggregate evidence for the pruning path that
    /// actually executed while this scorer was collected.
    #[must_use]
    pub(crate) fn pruning_telemetry(&self) -> PruningTelemetry {
        self.union_pruning_stats()
            .map_or_else(PruningTelemetry::default, PruningTelemetry::from)
    }

    /// Return exact refill receipts for method-bound conformance only.
    #[cfg(feature = "pruning-conformance")]
    #[must_use]
    pub(crate) fn conformance_union_refills(&self) -> &[ConformanceUnionRefill] {
        let ScorerNode::Union(union) = &self.node else {
            return &[];
        };
        &union.conformance_refills
    }

    #[cfg(test)]
    pub(crate) fn pruning_window_counts(&self) -> Option<(u64, u64)> {
        self.union_pruning_stats()
            .map(|stats| (stats.max_score_windows, stats.block_max_wand_windows))
    }

    /// Lower Boolean clauses into the pinned Tantivy scorer-tree shape.
    ///
    /// `Should` is required only when no positive `Must` exists. A raw
    /// `MustNot`-only node matches nothing.
    ///
    /// # Errors
    ///
    /// Returns a typed error if bounded scorer buffers cannot be allocated or
    /// initial cursor alignment detects malformed state.
    pub fn boolean(clauses: Vec<ScorerClause<'a>>) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(
            clauses,
            BooleanMode::Scored,
            false,
            #[cfg(feature = "pruning-conformance")]
            None,
        )
    }

    /// Lower a ranked root Boolean through Tantivy's frequency-term `TopDocs`
    /// accumulation order when the final scorer shape permits it.
    ///
    /// This is deliberately root-only: nested unions and counted collectors
    /// keep the ordinary exhaustive scorer order. The union constructor also
    /// fails closed unless every surviving child is a frequency-bearing term.
    ///
    /// # Errors
    ///
    /// Returns the same typed allocation, cursor-alignment, and scorer-shape
    /// failures as [`Self::boolean`].
    pub(crate) fn boolean_topdocs(clauses: Vec<ScorerClause<'a>>) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(
            clauses,
            BooleanMode::Scored,
            true,
            #[cfg(feature = "pruning-conformance")]
            None,
        )
    }

    /// Lower a ranked root Boolean while attaching exact interruption-location
    /// context to conformance-only query-work admissions.
    #[cfg(feature = "pruning-conformance")]
    pub(crate) fn boolean_topdocs_with_checkpoint(
        clauses: Vec<ScorerClause<'a>>,
        checkpoint: Arc<dyn QueryWorkCheckpoint + 'a>,
    ) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(clauses, BooleanMode::Scored, true, Some(checkpoint))
    }

    /// Lower Boolean clauses into a genuinely scoreless matching tree.
    ///
    /// The caller must use this constructor recursively for nested Boolean
    /// children. Unlike the parity-pinned buffered scored union, its `Should`
    /// union advances lazily and never invokes BM25 scoring. Optional `Should`
    /// clauses are omitted when a positive `Must` already determines matching.
    ///
    /// # Errors
    ///
    /// Rejects a child tree containing a scored buffered union, or any ordinary
    /// allocation and cursor-alignment failure.
    pub fn boolean_unscored(clauses: Vec<ScorerClause<'a>>) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(
            clauses,
            BooleanMode::Unscored,
            false,
            #[cfg(feature = "pruning-conformance")]
            None,
        )
    }

    pub(in crate::argus) fn boolean_with_mode(
        clauses: Vec<ScorerClause<'a>>,
        mode: BooleanMode,
        topdocs_root: bool,
        #[cfg(feature = "pruning-conformance")] conformance_checkpoint: Option<
            Arc<dyn QueryWorkCheckpoint + 'a>,
        >,
    ) -> Result<Self, ArgusError> {
        if mode == BooleanMode::Unscored
            && clauses
                .iter()
                .any(|clause| !clause.scorer.is_unscored_safe())
        {
            return Err(ArgusError::ScoredTreeForUnscoredCollector);
        }
        if matches!(
            clauses.as_slice(),
            [clause] if clause.occur == Occur::MustNot
        ) {
            return Ok(Self::empty());
        }
        let must_count = clauses
            .iter()
            .filter(|clause| clause.occur == Occur::Must)
            .count();
        let should_count = clauses
            .iter()
            .filter(|clause| clause.occur == Occur::Should)
            .count();
        let excluded_count = clauses.len() - must_count - should_count;
        let mut must = reserve_scorers("required Boolean clauses", must_count)?;
        let mut should = reserve_scorers("optional Boolean clauses", should_count)?;
        let mut excluded = reserve_scorers("excluded Boolean clauses", excluded_count)?;
        for clause in clauses {
            match clause.occur {
                Occur::Must if clause.scorer.is_explicit_empty() => return Ok(Self::empty()),
                Occur::Must => must.push(clause.scorer),
                Occur::Should if clause.scorer.is_explicit_empty() => {}
                Occur::Should => should.push(clause.scorer),
                Occur::MustNot if clause.scorer.is_explicit_empty() => {}
                Occur::MustNot => excluded.push(clause.scorer),
            }
        }

        // Tantivy removes direct, unboosted AllScorers before composing a
        // complex Boolean tree. A full-cardinality numeric range lowers to
        // that same raw AllScorer, while a boosted one is wrapped and remains
        // score-bearing. Keep one removed scorer per occurrence as the global
        // domain restoration token; duplicate unit contributions collapse.
        let must_all = remove_raw_unit_all_scorers(&mut must);
        let should_all = remove_raw_unit_all_scorers(&mut should);
        if remove_raw_unit_all_scorers(&mut excluded).is_some() {
            return Ok(Self::empty());
        }

        let required = if must.is_empty() {
            must_all
        } else {
            Some(scorer_intersection(must)?)
        };
        let include = match required {
            Some(required) => {
                // A raw optional All is score-neutral in Tantivy once a MUST
                // scorer determines matching. Non-All optional scorers retain
                // their scored RequiredOptional shape.
                if should.is_empty() || mode == BooleanMode::Unscored {
                    required
                } else {
                    let optional = scorer_union(
                        should,
                        false,
                        #[cfg(feature = "pruning-conformance")]
                        None,
                    )?;
                    let subtree_nodes = required
                        .subtree_nodes
                        .saturating_add(optional.subtree_nodes)
                        .saturating_add(1);
                    Self {
                        node: ScorerNode::RequiredOptional(RequiredOptionalScorer::new(
                            required, optional,
                        )?),
                        subtree_nodes,
                        #[cfg(feature = "pruning-conformance")]
                        construction_ordinal: None,
                    }
                }
            }
            None => match (should_all, mode) {
                (Some(all), BooleanMode::Unscored) => all,
                (Some(all), BooleanMode::Scored) if should.is_empty() => all,
                (Some(all), BooleanMode::Scored) => {
                    // Preserve Tantivy's nested score order: first aggregate
                    // the ordinary SHOULD scorers, then union one AllScorer.
                    let ordinary_should = scorer_union(
                        should,
                        false,
                        #[cfg(feature = "pruning-conformance")]
                        None,
                    )?;
                    let mut with_all = reserve_scorers("optional All clauses", 2)?;
                    with_all.push(ordinary_should);
                    with_all.push(all);
                    scorer_union(
                        with_all,
                        false,
                        #[cfg(feature = "pruning-conformance")]
                        None,
                    )?
                }
                (None, BooleanMode::Scored) => scorer_union(
                    should,
                    topdocs_root && excluded.is_empty(),
                    #[cfg(feature = "pruning-conformance")]
                    conformance_checkpoint,
                )?,
                (None, BooleanMode::Unscored) => scorer_union_unscored(should)?,
            },
        };
        if excluded.is_empty() {
            return Ok(include);
        }
        let subtree_nodes = excluded
            .iter()
            .fold(include.subtree_nodes.saturating_add(1), |count, scorer| {
                count.saturating_add(scorer.subtree_nodes)
            });
        Ok(Self {
            node: ScorerNode::Exclude(ExcludeScorer::new(include, excluded)?),
            subtree_nodes,
            #[cfg(feature = "pruning-conformance")]
            construction_ordinal: None,
        })
    }

    /// Current matching global document id.
    #[must_use]
    pub fn doc(&self) -> Option<u32> {
        match &self.node {
            ScorerNode::Empty => None,
            ScorerNode::All(scorer) => scorer.doc(),
            ScorerNode::Term(scorer) => scorer.doc(),
            ScorerNode::Phrase(scorer) => scorer.doc(),
            ScorerNode::NumericRange(scorer) => scorer.doc(),
            ScorerNode::ConstantScore(scorer) => scorer.doc(),
            ScorerNode::Intersection(scorer) => scorer.doc(),
            ScorerNode::Union(scorer) => scorer.doc(),
            ScorerNode::UnscoredUnion(scorer) => scorer.doc(),
            ScorerNode::RequiredOptional(scorer) => scorer.doc(),
            ScorerNode::Exclude(scorer) => scorer.doc(),
        }
    }

    /// Runtime cost used for stable required-clause ordering.
    #[must_use]
    pub fn cost(&self) -> u64 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::All(scorer) => scorer.cost(),
            ScorerNode::Term(scorer) => scorer.cost,
            ScorerNode::Phrase(scorer) => scorer.cost,
            ScorerNode::NumericRange(scorer) => scorer.cost(),
            ScorerNode::ConstantScore(scorer) => scorer.cost(),
            ScorerNode::Intersection(scorer) => scorer.cost(),
            ScorerNode::Union(scorer) => scorer.cost(),
            ScorerNode::UnscoredUnion(scorer) => scorer.cost(),
            ScorerNode::RequiredOptional(scorer) => scorer.cost(),
            ScorerNode::Exclude(scorer) => scorer.cost(),
        }
    }

    /// Tantivy-compatible best-effort match-count estimate.
    #[must_use]
    pub fn size_hint(&self) -> u32 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::All(scorer) => scorer.size_hint(),
            ScorerNode::Term(scorer) => scorer.size_hint,
            ScorerNode::Phrase(scorer) => scorer.size_hint,
            ScorerNode::NumericRange(scorer) => scorer.size_hint(),
            ScorerNode::ConstantScore(scorer) => scorer.size_hint(),
            ScorerNode::Intersection(scorer) => scorer.size_hint(),
            ScorerNode::Union(scorer) => scorer.size_hint(),
            ScorerNode::UnscoredUnion(scorer) => scorer.size_hint(),
            ScorerNode::RequiredOptional(scorer) => scorer.size_hint(),
            ScorerNode::Exclude(scorer) => scorer.size_hint(),
        }
    }

    /// Live document count used for Tantivy-compatible composite estimates.
    #[must_use]
    pub fn segment_num_docs(&self) -> u32 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::All(scorer) => scorer.segment_num_docs,
            ScorerNode::Term(scorer) => scorer.segment_num_docs,
            ScorerNode::Phrase(scorer) => scorer.segment_num_docs,
            ScorerNode::NumericRange(scorer) => scorer.segment_num_docs,
            ScorerNode::ConstantScore(scorer) => scorer.segment_num_docs(),
            ScorerNode::Intersection(scorer) => scorer.segment_num_docs,
            ScorerNode::Union(scorer) => scorer.segment_num_docs,
            ScorerNode::UnscoredUnion(scorer) => scorer.segment_num_docs,
            ScorerNode::RequiredOptional(scorer) => scorer.segment_num_docs,
            ScorerNode::Exclude(scorer) => scorer.segment_num_docs,
        }
    }

    /// Move strictly forward to the next match.
    ///
    /// # Errors
    ///
    /// Propagates typed cursor, fieldnorm, or score-buffer failures.
    #[allow(
        clippy::should_implement_trait,
        reason = "a scorer exposes both next-match and inclusive advance operations"
    )]
    pub fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(None),
            ScorerNode::All(scorer) => Ok(scorer.next()),
            ScorerNode::Term(scorer) => scorer.next(),
            ScorerNode::Phrase(scorer) => scorer.next(),
            ScorerNode::NumericRange(scorer) => Ok(scorer.next()),
            ScorerNode::ConstantScore(scorer) => Ok(scorer.next()),
            ScorerNode::Intersection(scorer) => scorer.next(),
            ScorerNode::Union(scorer) => scorer.next(),
            ScorerNode::UnscoredUnion(scorer) => scorer.next(),
            ScorerNode::RequiredOptional(scorer) => scorer.next(),
            ScorerNode::Exclude(scorer) => scorer.next(),
        }
    }

    /// Seek to the first match at or beyond `target`.
    ///
    /// # Errors
    ///
    /// Propagates typed cursor, fieldnorm, or score-buffer failures.
    pub fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(None),
            ScorerNode::All(scorer) => Ok(scorer.seek(target)),
            ScorerNode::Term(scorer) => scorer.seek(target),
            ScorerNode::Phrase(scorer) => scorer.seek(target),
            ScorerNode::NumericRange(scorer) => Ok(scorer.seek(target)),
            ScorerNode::ConstantScore(scorer) => Ok(scorer.seek(target)),
            ScorerNode::Intersection(scorer) => scorer.seek(target),
            ScorerNode::Union(scorer) => scorer.seek(target),
            ScorerNode::UnscoredUnion(scorer) => scorer.seek(target),
            ScorerNode::RequiredOptional(scorer) => scorer.seek(target),
            ScorerNode::Exclude(scorer) => scorer.seek(target),
        }
    }

    pub(in crate::argus) fn seek_danger(
        &mut self,
        target: u32,
    ) -> Result<SeekDangerResult, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(SeekDangerResult::SeekLowerBound(None)),
            ScorerNode::All(scorer) => classify_seek_result(target, scorer.seek(target)),
            ScorerNode::Term(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
            ScorerNode::Phrase(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
            ScorerNode::NumericRange(scorer) => classify_seek_result(target, scorer.seek(target)),
            ScorerNode::ConstantScore(scorer) => scorer.seek_danger(target),
            ScorerNode::Intersection(scorer) => scorer.seek_danger(target),
            ScorerNode::Union(scorer) => scorer.seek_danger(target),
            ScorerNode::UnscoredUnion(scorer) => scorer.seek_danger(target),
            ScorerNode::RequiredOptional(scorer) => scorer.seek_danger(target),
            ScorerNode::Exclude(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
        }
    }

    /// Score the current match in the pinned f32 scorer-tree order.
    ///
    /// # Errors
    ///
    /// Returns a typed error for exhausted or inconsistent cursors, missing
    /// fieldnorms, and non-finite arithmetic.
    pub fn score(&mut self) -> Result<f32, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Err(ArgusError::CursorInvariant("cannot score an empty scorer")),
            ScorerNode::All(scorer) => scorer.score(),
            ScorerNode::Term(scorer) => scorer.score(),
            ScorerNode::Phrase(scorer) => scorer.score(),
            ScorerNode::NumericRange(scorer) => scorer.score(),
            ScorerNode::ConstantScore(scorer) => scorer.score(),
            ScorerNode::Intersection(scorer) => scorer.score(),
            ScorerNode::Union(scorer) => scorer.score(),
            ScorerNode::UnscoredUnion(_) => Err(ArgusError::ScoreUnavailable),
            ScorerNode::RequiredOptional(scorer) => scorer.score(),
            ScorerNode::Exclude(scorer) => scorer.score(),
        }
    }

    /// Exhaustively evaluate this scorer and retain its best `limit` live hits.
    ///
    /// The phase-one heap stores only `(global_docid, score)`. Deleted documents
    /// are filtered here, after physical posting/statistics construction but
    /// before they can affect the heap threshold.
    ///
    /// # Errors
    ///
    /// Propagates typed scorer failures or a bounded heap allocation failure.
    pub fn top_k<L>(&mut self, limit: usize, live_docs: &L) -> Result<Vec<ScoredDoc>, ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut collector = TopDocsCollector::new(limit, 0)?;
        collector.collect(self, live_docs)?;
        Ok(collector.finish()?.hits)
    }

    /// Exhaust this recursively unscored tree into sorted unique global docids.
    ///
    /// Use [`Self::boolean_unscored`] for every Boolean level. Term and phrase
    /// leaves are intrinsically safe because advancing them never evaluates a
    /// BM25 score.
    ///
    /// # Errors
    ///
    /// Rejects a scored buffered union anywhere in the tree and propagates
    /// cursor or bounded-allocation failures.
    pub fn collect_doc_set<L>(&mut self, live_docs: &L) -> Result<Vec<u32>, ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut collector = DocSetCollector::new();
        collector.collect(self, live_docs)?;
        Ok(collector.finish())
    }

    pub(in crate::argus) fn collect_with<L>(
        &mut self,
        collector: &mut CollectorState<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        if matches!(collector, CollectorState::DocSet(_)) && !self.is_unscored_safe() {
            return Err(ArgusError::ScoredTreeForUnscoredCollector);
        }
        if let CollectorState::TopDocs(top_docs) = collector
            && !top_docs.exact_count
            && let ScorerNode::Union(union) = &mut self.node
        {
            return union.collect_top_docs(top_docs, live_docs);
        }
        while let Some(doc) = self.doc() {
            match collector {
                CollectorState::TopDocs(top_docs) => {
                    // Scoring physical deleted matches before visibility
                    // filtering preserves the buffered union's f32 state.
                    let score = (top_docs.retained != 0).then(|| self.score()).transpose()?;
                    if live_docs.is_live(doc) {
                        top_docs.record_live(doc, score)?;
                    }
                }
                CollectorState::DocSet(doc_set) => {
                    if live_docs.is_live(doc) {
                        doc_set.push(doc)?;
                    }
                }
            }
            self.next()?;
        }
        Ok(())
    }
}
