#[cfg(feature = "pruning-conformance")]
use crate::argus::QueryWorkCheckpoint;
use crate::argus::{
    ArgusError, BMW_MIN_CLAUSES, Bm25FieldSnapshot, CheckpointPostingCursor, DeltaFieldNorms,
    DeltaPostingCursor, MAX_SCORE_MAX_CLAUSES, PhraseScorer, PhraseTerm, ReferenceScorer,
    ScorerClause,
};
use crate::delta::DeltaSnapshot;
use crate::grimoire::{trailing_star_prefix, validate_bound_term, validate_query_term};
use crate::keeper::RecoveredSegment;
use crate::query::{
    BooleanOperator, Occur, Query, QueryClause, QueryNode, QueryNodeId, QueryValue,
    validate_index_capabilities,
};
use crate::schema::{FieldKind, SchemaDescriptor};
use asupersync::Cx;
use std::ops::Bound;
#[cfg(feature = "pruning-conformance")]
use std::sync::Arc;
use xxhash_rust::xxh3::Xxh3;

use super::checkpoint::{
    PreparedSnapshotDocFreqs, QueryCheckpointHandle, clone_query_checkpoint_for_argus,
};
use super::error_results::{QuillIndexError, invalid_state};
use super::query_cursors::{
    SnapshotDocFreqDeltaAdmission, borrowed_fieldnorms, checkpointed_snapshot_doc_freq,
    composite_snapshot_field, leaf_phrase_prefix_terms, lower_leaf_term, open_owned_cursor,
    snapshot_doc_freq,
};
use super::query_predicates::{
    lower_leaf_glob, lower_leaf_range, lower_leaf_set, numeric_query_bounds, numeric_query_values,
    query_field_kind, string_query_bounds, validate_string_query_field,
};
use super::snapshot::QuillSearchSnapshot;

#[derive(Clone, Copy)]
pub(super) enum QueryLeaf<'a> {
    Sealed(&'a RecoveredSegment),
    Delta(&'a DeltaSnapshot),
}

impl QueryLeaf<'_> {
    pub(super) fn docid_range(self) -> (u64, u64) {
        match self {
            Self::Sealed(segment) => (segment.manifest().docid_lo, segment.manifest().docid_hi),
            Self::Delta(delta) => delta.occupied_docid_range().unwrap_or_else(|| {
                let lease_base = delta.lease_base();
                (lease_base, lease_base)
            }),
        }
    }

    pub(super) fn live_document_count(self) -> Result<u32, QuillIndexError> {
        match self {
            Self::Sealed(segment) => Ok(segment.doc_count()),
            Self::Delta(delta) => u32::try_from(delta.live_document_count())
                .map_err(|_| invalid_state("Delta live document count does not fit u32")),
        }
    }

    pub(super) fn is_live_document(self, global_docid: u32) -> bool {
        match self {
            Self::Sealed(segment) => crate::argus::LiveDocs::is_live(segment, global_docid),
            Self::Delta(delta) => delta.is_live_document(global_docid),
        }
    }
}

pub(super) fn validate_query_lowering(
    query: &Query,
    inherited_boost: f32,
    schema: SchemaDescriptor,
) -> Result<(), QuillIndexError> {
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(query.node_count())
        .map_err(|_| ArgusError::Allocation {
            resource: "query validation traversal",
            count: query.node_count(),
        })?;
    pending.push((query.root_id(), inherited_boost));
    while let Some((node_id, inherited_boost)) = pending.pop() {
        match query.node(node_id) {
            QueryNode::Empty | QueryNode::All => {}
            QueryNode::AllocationFailure { requested } => {
                return Err(ArgusError::Allocation {
                    resource: "query arena",
                    count: *requested,
                }
                .into());
            }
            QueryNode::Term { fields, text } => {
                for field in fields {
                    validate_query_term(schema, field.field_id, text.as_bytes())?;
                    validate_cumulative_boost(inherited_boost, field.boost)?;
                }
            }
            QueryNode::Phrase {
                fields,
                terms,
                slop: _,
                prefix,
            } => {
                if terms.is_empty() {
                    return Err(ArgusError::InvalidPhrase {
                        reason: "an exact phrase requires positioned terms",
                    }
                    .into());
                }
                if terms.len() > 1 {
                    // Capability errors take precedence over operator refinements:
                    // slop and phrase-prefix queries also consume positions, so a
                    // positionless schema must receive the same typed failure.
                    validate_index_capabilities(query, schema)?;
                    if terms
                        .windows(2)
                        .any(|pair| pair[0].position > pair[1].position)
                    {
                        return Err(ArgusError::InvalidPhrase {
                            reason: "phrase positions must be non-decreasing",
                        }
                        .into());
                    }
                    if terms.first().map(|term| term.position)
                        == terms.last().map(|term| term.position)
                    {
                        return Err(ArgusError::InvalidPhrase {
                            reason: "an exact phrase must span at least two positions",
                        }
                        .into());
                    }
                }
                if *prefix && terms.len() == 1 {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: "phrase prefix requires at least two positioned terms".to_owned(),
                    });
                }
                for term in terms {
                    for field in fields {
                        validate_query_term(schema, field.field_id, term.text.as_bytes())?;
                    }
                }
                for field in fields {
                    validate_cumulative_boost(inherited_boost, field.boost)?;
                }
            }
            QueryNode::Boolean { clauses, .. } => {
                pending.extend(
                    clauses
                        .iter()
                        .rev()
                        .map(|clause| (clause.query, inherited_boost)),
                );
            }
            QueryNode::Range {
                field_id,
                lower,
                upper,
            } => {
                match query_field_kind(schema, *field_id)? {
                    FieldKind::Keyword | FieldKind::Text { .. } => {
                        validate_string_query_field(schema, *field_id, "range")?;
                        let (lower, upper) = string_query_bounds(*field_id, lower, upper)?;
                        validate_bound_term(schema, *field_id, &lower)?;
                        validate_bound_term(schema, *field_id, &upper)
                            .map_err(QuillIndexError::from)
                    }
                    FieldKind::I64 { indexed: true, .. }
                    | FieldKind::I64 {
                        indexed: false,
                        fast: true,
                    }
                    | FieldKind::U64 { indexed: true, .. }
                    | FieldKind::U64 {
                        indexed: false,
                        fast: true,
                    } => numeric_query_bounds(schema, *field_id, lower, upper).map(|_| ()),
                    FieldKind::I64 {
                        indexed: false,
                        fast: false,
                    }
                    | FieldKind::U64 {
                        indexed: false,
                        fast: false,
                    } => Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("range names non-indexed numeric field {field_id}"),
                    }),
                    FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("range names stored-only field {field_id}"),
                    }),
                }?;
            }
            QueryNode::Set { field_id, values } => {
                match query_field_kind(schema, *field_id)? {
                    FieldKind::Keyword | FieldKind::Text { .. } => {
                        for value in values {
                            let QueryValue::Str(value) = value else {
                                return Err(QuillIndexError::UnsupportedQuery {
                                    detail: format!(
                                        "set value type does not match string field {field_id}"
                                    ),
                                });
                            };
                            validate_query_term(schema, *field_id, value.as_bytes())?;
                        }
                        Ok(())
                    }
                    FieldKind::I64 { indexed: true, .. }
                    | FieldKind::I64 {
                        indexed: false,
                        fast: true,
                    }
                    | FieldKind::U64 { indexed: true, .. }
                    | FieldKind::U64 {
                        indexed: false,
                        fast: true,
                    } => numeric_query_values(schema, *field_id, values).map(|_| ()),
                    FieldKind::I64 {
                        indexed: false,
                        fast: false,
                    }
                    | FieldKind::U64 {
                        indexed: false,
                        fast: false,
                    } => Err(QuillIndexError::UnsupportedQuery {
                        detail: format!(
                            "set names non-indexed and non-fast numeric field {field_id}"
                        ),
                    }),
                    FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("set names stored-only field {field_id}"),
                    }),
                }?;
            }
            QueryNode::Glob { field_ids, .. } => {
                for &field_id in field_ids {
                    validate_string_query_field(schema, field_id, "glob")?;
                }
            }
            QueryNode::Boost {
                query: child,
                factor,
            } => {
                let boost = validate_cumulative_boost(inherited_boost, *factor)?;
                pending.push((*child, boost));
            }
        }
    }
    Ok(())
}

fn validate_cumulative_boost(inherited: f32, factor: f32) -> Result<f32, QuillIndexError> {
    let boost = inherited * factor;
    if boost.is_finite() {
        Ok(boost)
    } else {
        Err(QuillIndexError::UnsupportedQuery {
            detail: format!("non-finite cumulative boost bits 0x{:08x}", boost.to_bits()),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_leaf_phrase_term<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    field_ord: u16,
    position: u32,
    term_bytes: &[u8],
) -> Result<PhraseTerm<'a>, QuillIndexError> {
    let snapshot_doc_freq = match prepared_doc_freqs {
        Some(prepared) => prepared.get_or_compute(checkpoint, snapshot, field_ord, term_bytes)?,
        None => checkpointed_snapshot_doc_freq(
            checkpoint,
            snapshot,
            field_ord,
            term_bytes,
            SnapshotDocFreqDeltaAdmission::Ranking,
        )?,
    };
    Ok(match leaf {
        QueryLeaf::Sealed(segment) => {
            let cursor = open_owned_cursor(
                segment,
                schema,
                field_ord,
                term_bytes,
                true,
                Some(checkpoint),
            )?;
            PhraseTerm::new(
                field_ord,
                position,
                CheckpointPostingCursor::new(cursor, clone_query_checkpoint_for_argus(checkpoint))?,
                snapshot_doc_freq,
            )
        }
        QueryLeaf::Delta(delta) => {
            let cursor = DeltaPostingCursor::new_admitted(
                delta,
                field_ord,
                term_bytes,
                checkpoint.as_ref(),
            )?;
            PhraseTerm::new(
                field_ord,
                position,
                CheckpointPostingCursor::new(cursor, clone_query_checkpoint_for_argus(checkpoint))?,
                snapshot_doc_freq,
            )
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_leaf_exact_phrase<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    field_ord: u16,
    field_boost: f32,
    terms: &[crate::query::PositionedTerm],
    slop: u32,
    inherited_boost: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let stats = composite_snapshot_field(snapshot, field_ord)?;
    let mut phrase_terms = Vec::new();
    phrase_terms
        .try_reserve_exact(terms.len())
        .map_err(|_| invalid_state("could not allocate phrase terms"))?;
    for term in terms {
        phrase_terms.push(lower_leaf_phrase_term(
            checkpoint,
            leaf,
            snapshot,
            prepared_doc_freqs,
            schema,
            field_ord,
            term.position,
            term.text.as_bytes(),
        )?);
    }
    let bm25 = Bm25FieldSnapshot::new(stats)?;
    let boost = inherited_boost * field_boost;
    let scorer = match leaf {
        QueryLeaf::Sealed(segment) => PhraseScorer::new_with_slop_and_checkpoint(
            phrase_terms,
            borrowed_fieldnorms(segment, schema, field_ord)?,
            bm25,
            boost,
            slop,
            Some(clone_query_checkpoint_for_argus(checkpoint)),
        )?,
        QueryLeaf::Delta(delta) => PhraseScorer::new_with_slop_and_checkpoint(
            phrase_terms,
            DeltaFieldNorms::new(delta, field_ord),
            bm25,
            boost,
            slop,
            Some(clone_query_checkpoint_for_argus(checkpoint)),
        )?,
    };
    Ok(ReferenceScorer::phrase(scorer))
}

#[allow(clippy::too_many_arguments)]
fn lower_leaf_phrase_prefix<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    field_ord: u16,
    field_boost: f32,
    terms: &[crate::query::PositionedTerm],
    suffixes: &[Vec<u8>],
    inherited_boost: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    if suffixes.is_empty() {
        return Ok(ReferenceScorer::empty());
    }
    let (prefix_term, fixed_terms) = terms
        .split_last()
        .ok_or_else(|| invalid_state("validated phrase prefix lost its final term"))?;
    if fixed_terms.is_empty() {
        return Err(invalid_state(
            "validated phrase prefix has no fixed phrase term",
        ));
    }

    let mut lowered_fixed = Vec::new();
    lowered_fixed
        .try_reserve_exact(fixed_terms.len())
        .map_err(|_| invalid_state("could not allocate fixed phrase-prefix terms"))?;
    for term in fixed_terms {
        lowered_fixed.push(lower_leaf_phrase_term(
            checkpoint,
            leaf,
            snapshot,
            prepared_doc_freqs,
            schema,
            field_ord,
            term.position,
            term.text.as_bytes(),
        )?);
    }

    let mut lowered_suffixes = Vec::new();
    lowered_suffixes
        .try_reserve_exact(suffixes.len())
        .map_err(|_| invalid_state("could not allocate phrase-prefix suffix terms"))?;
    for suffix in suffixes {
        lowered_suffixes.push(lower_leaf_phrase_term(
            checkpoint,
            leaf,
            snapshot,
            prepared_doc_freqs,
            schema,
            field_ord,
            prefix_term.position,
            suffix,
        )?);
    }

    let stats = composite_snapshot_field(snapshot, field_ord)?;
    let bm25 = Bm25FieldSnapshot::new(stats)?;
    let boost = inherited_boost * field_boost;
    let scorer = match leaf {
        QueryLeaf::Sealed(segment) => PhraseScorer::new_phrase_prefix_with_checkpoint(
            lowered_fixed,
            lowered_suffixes,
            borrowed_fieldnorms(segment, schema, field_ord)?,
            bm25,
            boost,
            Some(clone_query_checkpoint_for_argus(checkpoint)),
        )?,
        QueryLeaf::Delta(delta) => PhraseScorer::new_phrase_prefix_with_checkpoint(
            lowered_fixed,
            lowered_suffixes,
            DeltaFieldNorms::new(delta, field_ord),
            bm25,
            boost,
            Some(clone_query_checkpoint_for_argus(checkpoint)),
        )?,
    };
    Ok(ReferenceScorer::phrase(scorer))
}

pub(super) fn lower_query<'a>(
    cx: &Cx,
    checkpoint: &QueryCheckpointHandle<'a>,
    query: &Query,
    inherited_boost: f32,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    glob_expansion_limit: usize,
    rank_pruning: bool,
    topdocs_root: bool,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    lower_query_with_mode(
        cx,
        checkpoint,
        query,
        inherited_boost,
        leaf,
        snapshot,
        prepared_doc_freqs,
        schema,
        glob_expansion_limit,
        QueryLoweringMode::Scored,
        rank_pruning,
        topdocs_root,
    )
}

pub(super) fn lower_query_unscored<'a>(
    cx: &Cx,
    checkpoint: &QueryCheckpointHandle<'a>,
    query: &Query,
    inherited_boost: f32,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    glob_expansion_limit: usize,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    lower_query_with_mode(
        cx,
        checkpoint,
        query,
        inherited_boost,
        leaf,
        snapshot,
        prepared_doc_freqs,
        schema,
        glob_expansion_limit,
        QueryLoweringMode::Unscored,
        false,
        false,
    )
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct QueryTraceShape {
    pub(super) root_kind: &'static str,
    pub(super) topology_hash: u64,
    pub(super) nodes: u64,
    pub(super) depth: u64,
    pub(super) terms: u64,
    pub(super) phrases: u64,
    pub(super) booleans: u64,
    pub(super) predicates: u64,
    pub(super) boosts: u64,
}

impl QueryTraceShape {
    fn from_query(query: &Query) -> Self {
        let mut shape = Self {
            root_kind: query_trace_kind(query.root()),
            topology_hash: query_topology_hash(query),
            ..Self::default()
        };
        shape.visit(query);
        shape
    }

    pub(super) fn visit(&mut self, query: &Query) {
        let mut pending = Vec::new();
        if pending.try_reserve_exact(query.node_count()).is_err() {
            return;
        }
        pending.push((query.root_id(), 1_u64));
        while let Some((node_id, depth)) = pending.pop() {
            self.nodes = self.nodes.saturating_add(1);
            self.depth = self.depth.max(depth);
            match query.node(node_id) {
                QueryNode::Empty | QueryNode::All => {}
                QueryNode::AllocationFailure { .. } => {}
                QueryNode::Term { .. } => self.terms = self.terms.saturating_add(1),
                QueryNode::Phrase { .. } => self.phrases = self.phrases.saturating_add(1),
                QueryNode::Boolean { clauses, .. } => {
                    self.booleans = self.booleans.saturating_add(1);
                    pending.extend(
                        clauses
                            .iter()
                            .rev()
                            .map(|clause| (clause.query, depth.saturating_add(1))),
                    );
                }
                QueryNode::Range { .. } | QueryNode::Set { .. } | QueryNode::Glob { .. } => {
                    self.predicates = self.predicates.saturating_add(1);
                }
                QueryNode::Boost { query: child, .. } => {
                    self.boosts = self.boosts.saturating_add(1);
                    pending.push((*child, depth.saturating_add(1)));
                }
            }
        }
    }
}

const fn query_trace_kind(query: &QueryNode) -> &'static str {
    match query {
        QueryNode::Empty => "empty",
        QueryNode::All => "all",
        QueryNode::AllocationFailure { .. } => "allocation_failure",
        QueryNode::Term { .. } => "term",
        QueryNode::Phrase { .. } => "phrase",
        QueryNode::Boolean { .. } => "boolean",
        QueryNode::Range { .. } => "range",
        QueryNode::Set { .. } => "set",
        QueryNode::Glob { .. } => "glob",
        QueryNode::Boost { .. } => "boost",
    }
}

pub(super) fn record_query_trace_shape(span: &tracing::Span, query: &Query) {
    let shape = QueryTraceShape::from_query(query);
    span.record("query_root", shape.root_kind);
    span.record("query_shape_hash", shape.topology_hash);
    span.record("query_nodes", shape.nodes);
    span.record("query_depth", shape.depth);
    span.record("term_nodes", shape.terms);
    span.record("phrase_nodes", shape.phrases);
    span.record("boolean_nodes", shape.booleans);
    span.record("predicate_nodes", shape.predicates);
    span.record("boost_nodes", shape.boosts);
}

fn query_topology_hash(query: &Query) -> u64 {
    let mut hasher = Xxh3::new();
    hash_query_topology(&mut hasher, query);
    hasher.digest()
}

fn hash_query_topology(hasher: &mut Xxh3, query: &Query) {
    enum HashTask {
        Node(QueryNodeId),
        Occur(Occur),
    }

    let Some(task_count) = query.node_count().checked_mul(2) else {
        hasher.update(&[u8::MAX]);
        return;
    };
    let mut pending = Vec::new();
    if pending.try_reserve_exact(task_count).is_err() {
        hasher.update(&[u8::MAX]);
        return;
    }
    pending.push(HashTask::Node(query.root_id()));
    while let Some(task) = pending.pop() {
        let node_id = match task {
            HashTask::Node(node_id) => node_id,
            HashTask::Occur(occur) => {
                hasher.update(&[match occur {
                    Occur::Must => 0,
                    Occur::Should => 1,
                    Occur::MustNot => 2,
                }]);
                continue;
            }
        };
        match query.node(node_id) {
            QueryNode::Empty => hasher.update(&[0]),
            QueryNode::All => hasher.update(&[1]),
            QueryNode::AllocationFailure { .. } => hasher.update(&[9]),
            QueryNode::Term { fields, .. } => {
                hasher.update(&[2]);
                hash_topology_len(hasher, fields.len());
            }
            QueryNode::Phrase {
                fields,
                terms,
                slop,
                prefix,
            } => {
                hasher.update(&[3]);
                hash_topology_len(hasher, fields.len());
                hash_topology_len(hasher, terms.len());
                hasher.update(&slop.to_le_bytes());
                hasher.update(&[u8::from(*prefix)]);
            }
            QueryNode::Boolean { clauses, operator } => {
                hasher.update(&[4]);
                hash_topology_len(hasher, clauses.len());
                hasher.update(&[match operator {
                    None => 0,
                    Some(BooleanOperator::And) => 1,
                    Some(BooleanOperator::Or) => 2,
                }]);
                for clause in clauses.iter().rev() {
                    pending.push(HashTask::Node(clause.query));
                    pending.push(HashTask::Occur(clause.occur));
                }
            }
            QueryNode::Range { lower, upper, .. } => {
                hasher.update(&[5, bound_topology_kind(lower), bound_topology_kind(upper)]);
            }
            QueryNode::Set { values, .. } => {
                hasher.update(&[6]);
                hash_topology_len(hasher, values.len());
            }
            QueryNode::Glob { field_ids, .. } => {
                hasher.update(&[7]);
                hash_topology_len(hasher, field_ids.len());
            }
            QueryNode::Boost { query: child, .. } => {
                hasher.update(&[8]);
                pending.push(HashTask::Node(*child));
            }
        }
    }
}

fn hash_topology_len(hasher: &mut Xxh3, len: usize) {
    hasher.update(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

const fn bound_topology_kind(bound: &Bound<QueryValue>) -> u8 {
    match bound {
        Bound::Unbounded => 0,
        Bound::Included(_) => 1,
        Bound::Excluded(_) => 2,
    }
}

/// Gate for deferred grouped `MaxScore` over nested pure-term unions
/// (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
///
/// **Currently `false`: the grouped runtime is rank-safe and demonstrably
/// prunes, but it did not earn activation in the paired release A/B.**
///
/// `argus::tests::grouped_max_score_matches_exhaustive_and_prunes` and the
/// drained-essential-group regression both pass. The immutable 100k,
/// same-binary, 41-round campaign recorded pruned/exhaustive medians of 1.0138
/// for two groups, 1.0023 for three, and 1.0151 for four, all inside their A/A
/// median-CI floors; eight groups was a decision-valid 1.0261 regression.
///
/// Flipping this to `true` would open BLOCKMAX for every nested union's terms
/// — i.e. pay the metadata cost on the most common query shape — without a
/// measured query-time win. Full provenance and the retry predicate live in
/// `docs/NEGATIVE_EVIDENCE.md`.
///
/// With this `false`, nested unions fail the rank-pruning gate exactly as they
/// did before this change: the shape classification below is inert, no BLOCKMAX
/// is opened for their terms, and every such query keeps the pre-existing
/// POSTINGS-only exhaustive path. Direct-term `MaxScore`/BMW is unaffected
/// either way.
///
/// **To activate:** first produce a materially different traversal and a
/// same-binary win for at least the two- and four-group 100k cells, then run
/// the full 1M and parity matrix. The gate flip itself remains the only
/// production edit required.
pub(super) const GROUPED_MAX_SCORE_ENABLED: bool = false;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PrunableScorerShape {
    Empty,
    Term,
    Union {
        children: usize,
        kind: UnionChildKind,
    },
}

/// What a prunable union's children lower to, which decides which runtime
/// pruning strategy (if any) can consume the union.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum UnionChildKind {
    /// Every child lowers to a single direct term cursor, so the union feeds
    /// term-granular `MaxScore` or block-max WAND.
    DirectTerms,
    /// Every child is itself a union of direct terms — a pure-term group, i.e.
    /// the multi-field terms a default query lowers to. Such a union feeds
    /// deferred grouped `MaxScore`, which decides essential vs. non-essential
    /// at *group* granularity (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
    TermGroups,
    /// Mixed children or deeper nesting: no runtime strategy consumes it, so
    /// the union stays POSTINGS-only.
    Mixed,
}

/// Mirror the score-tree topology that lowering will build without opening any
/// segment sections. Only pure non-negative term unions can consume `MaxScore`
/// or BMW metadata; every other shape keeps the pre-E4.4 POSTINGS-only path.
pub(super) fn prunable_scorer_shape(
    query: &Query,
    inherited_boost: f32,
) -> Option<PrunableScorerShape> {
    enum ShapeTask {
        Visit(QueryNodeId, f32),
        ReduceUnion(usize),
    }

    let node_count = query.node_count();
    let task_count = node_count.checked_mul(2)?;
    let mut pending = Vec::new();
    let mut values = Vec::new();
    if pending.try_reserve_exact(task_count).is_err()
        || values.try_reserve_exact(node_count).is_err()
    {
        return None;
    }
    pending.push(ShapeTask::Visit(query.root_id(), inherited_boost));
    while let Some(task) = pending.pop() {
        let ShapeTask::Visit(node_id, inherited_boost) = task else {
            let ShapeTask::ReduceUnion(clause_count) = task else {
                return None;
            };
            let mut children = 0_usize;
            let mut singleton = PrunableScorerShape::Empty;
            let mut direct_terms = true;
            let mut term_groups = true;
            let mut valid = true;
            for _ in 0..clause_count {
                let Some(shape) = values.pop().flatten() else {
                    valid = false;
                    continue;
                };
                if shape == PrunableScorerShape::Empty {
                    continue;
                }
                let Some(next) = children.checked_add(1) else {
                    valid = false;
                    continue;
                };
                children = next;
                direct_terms &= shape == PrunableScorerShape::Term;
                term_groups &= matches!(
                    shape,
                    PrunableScorerShape::Union {
                        kind: UnionChildKind::DirectTerms,
                        ..
                    }
                );
                singleton = shape;
            }
            values.push(valid.then_some(match children {
                0 => PrunableScorerShape::Empty,
                1 => singleton,
                _ => PrunableScorerShape::Union {
                    children,
                    kind: if direct_terms {
                        UnionChildKind::DirectTerms
                    } else if term_groups {
                        UnionChildKind::TermGroups
                    } else {
                        UnionChildKind::Mixed
                    },
                },
            }));
            continue;
        };
        match query.node(node_id) {
            QueryNode::Empty => values.push(Some(PrunableScorerShape::Empty)),
            QueryNode::Term { fields, .. } => {
                let valid = fields.iter().all(|field| {
                    let boost = inherited_boost * field.boost;
                    boost.is_finite() && !boost.is_sign_negative()
                });
                values.push(valid.then_some(match fields.len() {
                    0 => PrunableScorerShape::Empty,
                    1 => PrunableScorerShape::Term,
                    children => PrunableScorerShape::Union {
                        children,
                        kind: UnionChildKind::DirectTerms,
                    },
                }));
            }
            QueryNode::Boolean { clauses, .. }
                if clauses.iter().all(|clause| clause.occur == Occur::Should) =>
            {
                pending.push(ShapeTask::ReduceUnion(clauses.len()));
                pending.extend(
                    clauses
                        .iter()
                        .rev()
                        .map(|clause| ShapeTask::Visit(clause.query, inherited_boost)),
                );
            }
            QueryNode::Boost {
                query: child,
                factor,
            } => {
                let boost = inherited_boost * *factor;
                if boost.is_finite() && !boost.is_sign_negative() {
                    pending.push(ShapeTask::Visit(*child, boost));
                } else {
                    values.push(None);
                }
            }
            QueryNode::All
            | QueryNode::AllocationFailure { .. }
            | QueryNode::Phrase { .. }
            | QueryNode::Range { .. }
            | QueryNode::Set { .. }
            | QueryNode::Glob { .. }
            | QueryNode::Boolean { .. } => values.push(None),
        }
    }
    values.pop().flatten()
}

/// Whether the root scorer can actually consume rank-pruning metadata, and so
/// whether opening BLOCKMAX for its terms is worth the setup cost.
///
/// Mirrors the runtime strategy selection in `argus::competitive_candidates`
/// exactly — admitting a shape no strategy consumes would pay for BLOCKMAX and
/// still score exhaustively.
pub(super) fn query_has_prunable_root_union(query: &Query, inherited_boost: f32) -> bool {
    match prunable_scorer_shape(query, inherited_boost) {
        Some(PrunableScorerShape::Union { children, kind }) => match kind {
            // Direct-term roots take `MaxScore` (2..=8 clauses) or block-max
            // WAND (9+), both term-granular.
            UnionChildKind::DirectTerms => {
                matches!(children, 2..=MAX_SCORE_MAX_CLAUSES | BMW_MIN_CLAUSES..)
            }
            // Grouped roots take deferred grouped `MaxScore` only. Grouped BMW
            // does not exist (a group has no physical block-max list of its
            // own), so wider grouped roots stay POSTINGS-only rather than
            // paying for metadata nothing reads.
            UnionChildKind::TermGroups => {
                GROUPED_MAX_SCORE_ENABLED && matches!(children, 2..=MAX_SCORE_MAX_CLAUSES)
            }
            UnionChildKind::Mixed => false,
        },
        Some(PrunableScorerShape::Empty | PrunableScorerShape::Term) | None => false,
    }
}

#[derive(Clone, Copy)]
pub(super) enum QueryLoweringMode {
    Scored,
    Unscored,
}

// `'a` is load-bearing under `pruning-conformance`, where the checkpoint
// parameter shares it; elision only type-checks with the feature off.
#[allow(clippy::elidable_lifetime_names)]
pub(super) fn lower_boolean<'a>(
    clauses: Vec<ScorerClause<'a>>,
    mode: QueryLoweringMode,
    topdocs_root: bool,
    #[cfg(feature = "pruning-conformance")] checkpoint: Option<&QueryCheckpointHandle<'a>>,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    match mode {
        QueryLoweringMode::Scored if topdocs_root => {
            #[cfg(feature = "pruning-conformance")]
            if let Some(checkpoint) = checkpoint {
                // `QueryCheckpointHandle` is the concrete `Arc<QueryCheckpoint>`
                // under `profile-internals` and already `Arc<dyn ..>` otherwise.
                // The clone must infer its type parameter from the ARGUMENT, so
                // it is bound unannotated first; annotating it directly makes
                // `Arc::clone` unify against `Arc<dyn ..>` and reject the
                // concrete handle before any unsize coercion can apply. The
                // second binding is the coercion site, and is an identity when
                // the alias is already the trait object.
                let handle = Arc::clone(checkpoint);
                let checkpoint: Arc<dyn QueryWorkCheckpoint + 'a> = handle;
                return ReferenceScorer::boolean_topdocs_with_checkpoint(clauses, checkpoint)
                    .map_err(QuillIndexError::from);
            }
            ReferenceScorer::boolean_topdocs(clauses)
        }
        QueryLoweringMode::Scored => ReferenceScorer::boolean(clauses),
        QueryLoweringMode::Unscored => ReferenceScorer::boolean_unscored(clauses),
    }
    .map_err(QuillIndexError::from)
}

/// Detect the two-arm exact-or-glob Boolean that cass's grammar lowers
/// every non-CJK exact and prefix literal into (see `lower_term_part` in
/// `query.rs`): an explicit `Or` of one `Should(Term)` and one
/// `Should(Glob)` over exactly one field, where that field is one of the
/// Term arm's fields and the glob's pattern is the Term's own text plus a
/// single trailing star. Returns the glob child's id.
///
/// Only for that shape — where the exact Term arm is guaranteed to walk the
/// same token over the glob's one field, whatever schema number that field
/// carries (the CASS schema's `content` is field 7, the engine default
/// schema's is 1, so the check is by set membership, never by a hardcoded
/// field constant) — may the glob arm drop its own literal term from the
/// expansion: every document the literal's posting list names is already
/// matched by the Boolean `Or`'s Term arm, so decoding it again only
/// re-derived admitted documents, and it is the single largest posting
/// list of a common-term expansion (on the owner archive, `error`'s
/// literal term dominated its whole arm). The drop changes no matched
/// document; it removes the arm's constant boost from documents whose only
/// globset membership was the literal itself, which keep their Term-arm
/// match and BM25 unchanged.
///
/// A shape this detector does not recognize — wrong operator, wrong
/// occurrence, clause order swapped, a Term arm that skips the glob's
/// field, a multi-field glob, a pattern with interior stars — returns
/// `None` and the glob child lowers through the plain path with its
/// literal intact, so the drop can never fire without its exact-arm
/// guarantee.
pub(super) fn two_arm_variants_glob(
    clauses: &[QueryClause],
    query: &Query,
    operator: Option<BooleanOperator>,
) -> Option<QueryNodeId> {
    if clauses.len() != 2
        || operator != Some(BooleanOperator::Or)
        || clauses.iter().any(|clause| clause.occur != Occur::Should)
    {
        return None;
    }
    let QueryNode::Term { fields, text } = query.node(clauses[0].query) else {
        return None;
    };
    let QueryNode::Glob { field_ids, pattern } = query.node(clauses[1].query) else {
        return None;
    };
    if field_ids.len() != 1 || !fields.iter().any(|field| field.field_id == field_ids[0]) {
        return None;
    }
    (trailing_star_prefix(pattern.as_bytes()) == Some(text.as_bytes())).then_some(clauses[1].query)
}

fn lower_query_with_mode<'a>(
    cx: &Cx,
    checkpoint: &QueryCheckpointHandle<'a>,
    query: &Query,
    inherited_boost: f32,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    glob_expansion_limit: usize,
    mode: QueryLoweringMode,
    rank_pruning: bool,
    topdocs_root: bool,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    enum LowerTask {
        Visit {
            node: QueryNodeId,
            boost: f32,
            topdocs_root: bool,
        },
        /// Visit a Glob child whose own literal term an accompanying exact
        /// Term arm already walks over the glob's one field — the two-arm
        /// shape cass's grammar lowers every non-CJK exact and prefix
        /// literal into (see [`two_arm_variants_glob`]). The glob arm may
        /// then drop that literal from its expansion.
        VisitGlobVariants {
            node: QueryNodeId,
            boost: f32,
            topdocs_root: bool,
        },
        BuildBoolean {
            node: QueryNodeId,
            topdocs_root: bool,
        },
    }

    let node_count = query.node_count();
    let task_count = node_count
        .checked_mul(2)
        .ok_or_else(|| invalid_state("query lowering task count overflowed"))?;
    let mut pending = Vec::new();
    let mut values = Vec::new();
    pending
        .try_reserve_exact(task_count)
        .map_err(|_| ArgusError::Allocation {
            resource: "query lowering tasks",
            count: task_count,
        })?;
    values
        .try_reserve_exact(node_count)
        .map_err(|_| ArgusError::Allocation {
            resource: "query lowering values",
            count: node_count,
        })?;
    pending.push(LowerTask::Visit {
        node: query.root_id(),
        boost: inherited_boost,
        topdocs_root,
    });
    while let Some(task) = pending.pop() {
        let (node, inherited_boost, topdocs_root, exclude_glob_literal) = match task {
            LowerTask::Visit {
                node,
                boost,
                topdocs_root,
            } => (node, boost, topdocs_root, false),
            LowerTask::VisitGlobVariants {
                node,
                boost,
                topdocs_root,
            } => (node, boost, topdocs_root, true),
            LowerTask::BuildBoolean { node, topdocs_root } => {
                let QueryNode::Boolean { clauses, .. } = query.node(node) else {
                    return Err(invalid_state("Boolean lowering task lost its query node"));
                };
                let mut lowered = Vec::new();
                lowered
                    .try_reserve_exact(clauses.len())
                    .map_err(|_| invalid_state("could not allocate Boolean clauses"))?;
                for clause in clauses.iter().rev() {
                    let scorer = values
                        .pop()
                        .ok_or_else(|| invalid_state("Boolean lowering lost a child scorer"))?;
                    lowered.push(ScorerClause::new(clause.occur, scorer));
                }
                lowered.reverse();
                values.push(lower_boolean(
                    lowered,
                    mode,
                    topdocs_root,
                    #[cfg(feature = "pruning-conformance")]
                    Some(checkpoint),
                )?);
                continue;
            }
        };

        let scorer = match query.node(node) {
            QueryNode::Empty => ReferenceScorer::empty(),
            QueryNode::AllocationFailure { requested } => {
                return Err(ArgusError::Allocation {
                    resource: "query arena",
                    count: *requested,
                }
                .into());
            }
            QueryNode::All => {
                let (docid_lo, docid_hi) = leaf.docid_range();
                ReferenceScorer::all_with_boost(
                    docid_lo,
                    docid_hi,
                    leaf.live_document_count()?,
                    inherited_boost,
                )?
            }
            QueryNode::Term { fields, text } => {
                let mut clauses = Vec::new();
                clauses
                    .try_reserve_exact(fields.len())
                    .map_err(|_| invalid_state("could not allocate expanded term clauses"))?;
                for field in fields {
                    clauses.push(ScorerClause::should(lower_leaf_term(
                        leaf,
                        snapshot,
                        schema,
                        field.field_id,
                        text.as_bytes(),
                        prepared_doc_freqs,
                        inherited_boost * field.boost,
                        rank_pruning,
                        checkpoint,
                    )?));
                }
                lower_boolean(
                    clauses,
                    mode,
                    topdocs_root,
                    #[cfg(feature = "pruning-conformance")]
                    Some(checkpoint),
                )?
            }
            QueryNode::Phrase {
                fields,
                terms,
                slop,
                prefix,
            } => {
                if *prefix && terms.len() == 1 {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: "phrase prefix requires at least two positioned terms".to_owned(),
                    });
                }
                if terms.len() == 1 {
                    let term = &terms[0];
                    let mut clauses = Vec::new();
                    clauses
                        .try_reserve_exact(fields.len())
                        .map_err(|_| invalid_state("could not allocate phrase field clauses"))?;
                    for field in fields {
                        clauses.push(ScorerClause::should(lower_leaf_term(
                            leaf,
                            snapshot,
                            schema,
                            field.field_id,
                            term.text.as_bytes(),
                            prepared_doc_freqs,
                            inherited_boost * field.boost,
                            false,
                            checkpoint,
                        )?));
                    }
                    lower_boolean(
                        clauses,
                        mode,
                        topdocs_root,
                        #[cfg(feature = "pruning-conformance")]
                        Some(checkpoint),
                    )?
                } else {
                    let mut clauses = Vec::new();
                    clauses
                        .try_reserve_exact(fields.len())
                        .map_err(|_| invalid_state("could not allocate phrase field clauses"))?;
                    for field in fields {
                        if *prefix {
                            let final_term = terms.last().ok_or_else(|| {
                                invalid_state("validated phrase prefix lost its final term")
                            })?;
                            let final_bytes = final_term.text.as_bytes();
                            let expansions = leaf_phrase_prefix_terms(
                                checkpoint,
                                leaf,
                                schema,
                                field.field_id,
                                final_bytes,
                            )?;
                            // Tantivy scores the fixed phrase once and uses all suffix
                            // expansions only to gate membership. Lowering one exact
                            // scored phrase per suffix would both add suffix IDF and
                            // multiply the fixed score for multi-suffix documents.
                            clauses.push(ScorerClause::should(lower_leaf_phrase_prefix(
                                checkpoint,
                                leaf,
                                snapshot,
                                prepared_doc_freqs,
                                schema,
                                field.field_id,
                                field.boost,
                                terms,
                                &expansions,
                                inherited_boost,
                            )?));
                        } else {
                            clauses.push(ScorerClause::should(lower_leaf_exact_phrase(
                                checkpoint,
                                leaf,
                                snapshot,
                                prepared_doc_freqs,
                                schema,
                                field.field_id,
                                field.boost,
                                terms,
                                *slop,
                                inherited_boost,
                            )?));
                        }
                    }
                    lower_boolean(
                        clauses,
                        mode,
                        topdocs_root,
                        #[cfg(feature = "pruning-conformance")]
                        Some(checkpoint),
                    )?
                }
            }
            QueryNode::Boolean { clauses, operator } => {
                pending.push(LowerTask::BuildBoolean { node, topdocs_root });
                // The two-arm exact-or-glob shape (see `two_arm_variants_glob`
                // below) is the one place a glob arm's literal term is
                // guaranteed covered by an accompanying exact Term walking
                // the same token over the glob's field. Route its glob
                // child through `VisitGlobVariants` so `lower_leaf_glob`
                // can skip re-enumerating that literal — the single largest
                // posting list of a common-term expansion, whose every
                // document the Term arm already matched. A shape the
                // detector does not recognize keeps the plain `Visit`, so
                // the drop can never fire without its exact-arm guarantee.
                let variants_glob = two_arm_variants_glob(clauses, query, *operator);
                pending.extend(clauses.iter().rev().map(|clause| {
                    if variants_glob == Some(clause.query) {
                        LowerTask::VisitGlobVariants {
                            node: clause.query,
                            boost: inherited_boost,
                            topdocs_root: false,
                        }
                    } else {
                        LowerTask::Visit {
                            node: clause.query,
                            boost: inherited_boost,
                            topdocs_root: false,
                        }
                    }
                }));
                continue;
            }
            QueryNode::Boost {
                query: child,
                factor,
            } => {
                let boost = inherited_boost * *factor;
                if !boost.is_finite() {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: format!(
                            "non-finite cumulative boost bits 0x{:08x}",
                            boost.to_bits()
                        ),
                    });
                }
                pending.push(LowerTask::Visit {
                    node: *child,
                    boost,
                    topdocs_root,
                });
                continue;
            }
            QueryNode::Range {
                field_id,
                lower,
                upper,
            } => lower_leaf_range(
                cx,
                checkpoint,
                leaf,
                snapshot,
                prepared_doc_freqs,
                schema,
                *field_id,
                lower,
                upper,
                inherited_boost,
                mode,
            )?,
            QueryNode::Glob { field_ids, pattern } => lower_leaf_glob(
                checkpoint,
                leaf,
                snapshot,
                prepared_doc_freqs,
                schema,
                field_ids,
                pattern.as_bytes(),
                inherited_boost,
                glob_expansion_limit,
                mode,
                exclude_glob_literal,
            )?,
            QueryNode::Set { field_id, values } => lower_leaf_set(
                checkpoint,
                leaf,
                snapshot,
                prepared_doc_freqs,
                schema,
                *field_id,
                values,
                inherited_boost,
                mode,
            )?,
        };
        values.push(scorer);
    }
    if values.len() != 1 {
        return Err(invalid_state(
            "query lowering produced an invalid scorer stack",
        ));
    }
    values
        .pop()
        .ok_or_else(|| invalid_state("query lowering produced no root scorer"))
}
