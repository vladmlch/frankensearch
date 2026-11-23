use std::borrow::Cow;
use std::fmt;
use std::ops::Bound;

use thiserror::Error;

use crate::schema::{FieldKind, SchemaDescriptor};

use super::ast::{BooleanOperator, Occur, Query, QueryField, QueryNode, QueryNodeId, QueryValue};

/// Outcome of one [`canonicalize_query`] pass (bd-quill-duel-ast-dedup-7hsu).
///
/// Every count is score-neutral by construction: scoring clauses
/// (`Term`/`Phrase`/scoring `Must`/`Should`) are never removed, so oracle
/// conformance on duplicate-term queries is preserved exactly.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QueryCanonicalizationReport {
    /// Exact-duplicate `MustNot` clauses removed (exclusion is idempotent).
    pub must_not_duplicates_removed: usize,
    /// Reserved for score-neutral positive-filter cleanup. Positive branches
    /// are currently all retained because ancestor boosts affect their score.
    pub filter_duplicates_removed: usize,
    /// Reserved for score-neutral field expansion cleanup. Duplicate scoring
    /// field branches are intentionally retained, so this is currently zero.
    pub duplicate_fields_removed: usize,
    /// Boolean nodes whose occurrence groups were put into the deterministic
    /// `Must`, `Should`, `MustNot` order while preserving construction order
    /// within every group.
    pub boolean_levels_sorted: usize,
    /// Glob nodes whose field list was sorted while retaining duplicates.
    pub glob_fields_canonicalized: usize,
}

/// Rank used to group canonical Boolean children: required, optional,
/// excluded. Within one group, construction order is preserved.
const fn occur_rank(occur: Occur) -> u8 {
    match occur {
        Occur::Must => 0,
        Occur::Should => 1,
        Occur::MustNot => 2,
    }
}

fn push_framed_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&bytes.len().to_be_bytes());
    out.extend_from_slice(bytes);
}

fn push_query_field_key(out: &mut Vec<u8>, field: QueryField) {
    out.extend_from_slice(&field.field_id.to_be_bytes());
    out.extend_from_slice(&field.boost.to_bits().to_be_bytes());
}

fn push_query_value_key(out: &mut Vec<u8>, value: &QueryValue) {
    match value {
        QueryValue::I64(v) => {
            out.push(0);
            out.extend_from_slice(&v.to_be_bytes());
        }
        QueryValue::U64(v) => {
            out.push(1);
            out.extend_from_slice(&v.to_be_bytes());
        }
        QueryValue::Str(text) => {
            out.push(2);
            push_framed_bytes(out, text.as_bytes());
        }
    }
}

fn push_bound_key(out: &mut Vec<u8>, bound: &Bound<QueryValue>) {
    match bound {
        Bound::Included(value) => {
            out.push(0);
            push_query_value_key(out, value);
        }
        Bound::Excluded(value) => {
            out.push(1);
            push_query_value_key(out, value);
        }
        Bound::Unbounded => out.push(2),
    }
}

pub(super) fn canonical_child_key(keys: &[Vec<u8>], id: QueryNodeId) -> &[u8] {
    if id == QueryNodeId::ALL {
        &[1]
    } else if id == QueryNodeId::EMPTY {
        &[0]
    } else {
        &keys[id.index()]
    }
}

/// Total deterministic structural key for one query subtree. Two subtrees
/// share a key iff they are structurally identical (field ids, boost bits,
/// terms, positions, bounds, values, patterns, and child order-sensitive
/// structure), so the key is safe both for deterministic ordering and for
/// exact-duplicate detection.
pub(super) fn canonical_sort_key_for_node(node: &QueryNode, keys: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    match node {
        QueryNode::Empty => out.push(0),
        QueryNode::All => out.push(1),
        QueryNode::AllocationFailure { requested } => {
            out.push(9);
            out.extend_from_slice(&requested.to_be_bytes());
        }
        QueryNode::Term { fields, text } => {
            out.push(2);
            out.extend_from_slice(&fields.len().to_be_bytes());
            for field in fields {
                push_query_field_key(&mut out, *field);
            }
            push_framed_bytes(&mut out, text.as_bytes());
        }
        QueryNode::Phrase {
            fields,
            terms,
            slop,
            prefix,
        } => {
            out.push(3);
            out.extend_from_slice(&fields.len().to_be_bytes());
            for field in fields {
                push_query_field_key(&mut out, *field);
            }
            out.extend_from_slice(&terms.len().to_be_bytes());
            for term in terms {
                out.extend_from_slice(&term.position.to_be_bytes());
                push_framed_bytes(&mut out, term.text.as_bytes());
            }
            out.extend_from_slice(&slop.to_be_bytes());
            out.push(u8::from(*prefix));
        }
        QueryNode::Boolean { clauses, operator } => {
            out.push(4);
            out.push(match operator {
                None => 0,
                Some(BooleanOperator::And) => 1,
                Some(BooleanOperator::Or) => 2,
            });
            out.extend_from_slice(&clauses.len().to_be_bytes());
            for clause in clauses {
                out.push(occur_rank(clause.occur));
                push_framed_bytes(&mut out, canonical_child_key(keys, clause.query));
            }
        }
        QueryNode::Range {
            field_id,
            lower,
            upper,
        } => {
            out.push(5);
            out.extend_from_slice(&field_id.to_be_bytes());
            push_bound_key(&mut out, lower);
            push_bound_key(&mut out, upper);
        }
        QueryNode::Set { field_id, values } => {
            out.push(6);
            out.extend_from_slice(&field_id.to_be_bytes());
            out.extend_from_slice(&values.len().to_be_bytes());
            for value in values {
                push_query_value_key(&mut out, value);
            }
        }
        QueryNode::Glob { field_ids, pattern } => {
            out.push(7);
            out.extend_from_slice(&field_ids.len().to_be_bytes());
            for field_id in field_ids {
                out.extend_from_slice(&field_id.to_be_bytes());
            }
            push_framed_bytes(&mut out, pattern.as_bytes());
        }
        QueryNode::Boost {
            query: child,
            factor,
        } => {
            out.push(8);
            out.extend_from_slice(&factor.to_bits().to_be_bytes());
            push_framed_bytes(&mut out, canonical_child_key(keys, *child));
        }
    }
    out
}

#[cfg(test)]
pub(super) fn canonical_sort_key(query: &Query) -> Vec<u8> {
    let mut keys = Vec::<Vec<u8>>::new();
    if keys.try_reserve_exact(query.nodes.len()).is_err() {
        return Vec::new();
    }
    for node in &query.nodes {
        keys.push(canonical_sort_key_for_node(node, &keys));
    }
    canonical_sort_key_for_node(query.root(), &keys)
}

/// Canonicalize a parsed query tree for deterministic planning and
/// score-neutral deduplication (bd-quill-duel-ast-dedup-7hsu).
///
/// The pass is iterative (bottom-up) and applies three score-neutral rules:
///
/// 1. **Deterministic occurrence groups**: Boolean children are stably grouped
///    as `Must`, `Should`, `MustNot`. Construction order within each group is
///    retained because `f32` score accumulation is not associative and the
///    oracle exposes that order in exact score bits.
/// 2. **`MustNot` dedup**: exact-duplicate exclusion clauses collapse to
///    one; exclusion is idempotent.
/// 3. **Field-list ordering**: `Glob` field ids are sorted without removing
///    duplicates. `Term`/`Phrase` expansions retain exact duplicate branches
///    because each branch contributes to the score.
///
/// Every positive branch is retained, including `All`: an ancestor boost can
/// make apparently unit-score children contribute independently. The scorer
/// owns the narrower effective-unit-`All` collapse because it sees cumulative
/// boosts during lowering.
///
/// Clause-level repetition caps are unnecessary here: `MAX_QUERY_LENGTH`
/// already bounds the source, and this pass only rewrites structure.
/// Idempotent: a second pass is a no-op and returns a zero report.
#[must_use]
pub fn canonicalize_query(query: &mut Query) -> QueryCanonicalizationReport {
    let mut report = QueryCanonicalizationReport::default();
    let mut keys = Vec::<Vec<u8>>::new();
    if keys.try_reserve_exact(query.nodes.len()).is_err() {
        return report;
    }
    for index in 0..query.nodes.len() {
        if let QueryNode::Boolean { clauses, .. } = &mut query.nodes[index] {
            let mut seen_must_not = Vec::<Vec<u8>>::new();
            if seen_must_not.try_reserve_exact(clauses.len()).is_err() {
                return report;
            }
            clauses.retain(|clause| {
                if clause.occur != Occur::MustNot {
                    return true;
                }
                let key = canonical_child_key(&keys, clause.query);
                if seen_must_not
                    .iter()
                    .any(|candidate| candidate.as_slice() == key)
                {
                    report.must_not_duplicates_removed += 1;
                    return false;
                }
                seen_must_not.push(key.to_vec());
                true
            });
            let before = clauses
                .iter()
                .map(|clause| clause.occur)
                .collect::<Vec<_>>();
            clauses.sort_by_key(|clause| occur_rank(clause.occur));
            if before
                != clauses
                    .iter()
                    .map(|clause| clause.occur)
                    .collect::<Vec<_>>()
            {
                report.boolean_levels_sorted += 1;
            }
        } else if let QueryNode::Glob { field_ids, .. } = &mut query.nodes[index] {
            let original = field_ids.clone();
            field_ids.sort_unstable();
            if *field_ids != original {
                report.glob_fields_canonicalized += 1;
            }
        }
        keys.push(canonical_sort_key_for_node(&query.nodes[index], &keys));
    }
    if let QueryNode::Boolean { clauses, .. } = &mut query.root {
        let mut seen_must_not = Vec::<Vec<u8>>::new();
        if seen_must_not.try_reserve_exact(clauses.len()).is_ok() {
            clauses.retain(|clause| {
                if clause.occur != Occur::MustNot {
                    return true;
                }
                let key = canonical_child_key(&keys, clause.query);
                if seen_must_not
                    .iter()
                    .any(|candidate| candidate.as_slice() == key)
                {
                    report.must_not_duplicates_removed += 1;
                    return false;
                }
                seen_must_not.push(key.to_vec());
                true
            });
        }
        let before = clauses
            .iter()
            .map(|clause| clause.occur)
            .collect::<Vec<_>>();
        clauses.sort_by_key(|clause| occur_rank(clause.occur));
        if before
            != clauses
                .iter()
                .map(|clause| clause.occur)
                .collect::<Vec<_>>()
        {
            report.boolean_levels_sorted += 1;
        }
    } else if let QueryNode::Glob { field_ids, .. } = &mut query.root {
        let original = field_ids.clone();
        field_ids.sort_unstable();
        if *field_ids != original {
            report.glob_fields_canonicalized += 1;
        }
    }
    report
}

/// Stable user-facing classification preserved from the incumbent adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QueryExplanation {
    /// Empty or whitespace-only input.
    Empty,
    /// One whitespace-delimited fragment.
    Simple,
    /// Input wrapped in matching single or double quotes.
    Phrase,
    /// More than one whitespace-delimited fragment.
    Boolean,
}

impl fmt::Display for QueryExplanation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("empty"),
            Self::Simple => formatter.write_str("simple"),
            Self::Phrase => formatter.write_str("phrase"),
            Self::Boolean => formatter.write_str("boolean"),
        }
    }
}

/// Classify a raw query exactly as the shipping explanation surface does.
#[must_use]
pub fn classify_query(query: &str) -> QueryExplanation {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryExplanation::Empty;
    }
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        return QueryExplanation::Phrase;
    }
    if trimmed.split_whitespace().count() <= 1 {
        QueryExplanation::Simple
    } else {
        QueryExplanation::Boolean
    }
}

/// Parser recovery/normalization diagnostic category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryDiagnosticKind {
    /// The query exceeded [`MAX_QUERY_LENGTH`].
    Truncated,
    /// Invalid syntax was repaired or skipped.
    SyntaxRecovery,
    /// A field prefix was not present in the schema.
    UnknownField,
    /// A known field cannot be queried by this parser.
    UnsupportedField,
    /// A fragment emitted no usable query branch.
    DroppedFragment,
    /// A typed range endpoint or set member did not match its schema field.
    InvalidTypedValue,
    /// A boost was malformed or outside the finite non-negative domain.
    InvalidBoost,
    /// Match-all was added to give an all-negative user query complement semantics.
    AllNegativeRepair,
    /// The nesting cap was reached.
    DepthLimit,
    /// A bounded parser or arena allocation could not be reserved.
    AllocationFailure,
    /// An analyzed term exceeded [`MAX_TERM_BYTES`], so the branch that
    /// required it was lowered to match-nothing.
    ///
    /// The analyzer drops the oversized term and the branch cannot be
    /// satisfied without it, so the lowering is a silent change in meaning
    /// rather than a repair: the query still parses, and it can no longer
    /// match. Every other lowering of that severity announces itself here, and
    /// until this variant existed this one did not — a consumer reading the
    /// diagnostics stream could not distinguish "matched nothing" from
    /// "lowered to match nothing".
    ///
    /// [`MAX_QUERY_LENGTH`]: super::MAX_QUERY_LENGTH
    /// [`MAX_TERM_BYTES`]: crate::grimoire::MAX_TERM_BYTES
    TermLengthLimit,
}

/// One best-effort parse diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryDiagnostic {
    /// Stable diagnostic category.
    pub kind: QueryDiagnosticKind,
    /// Human-readable contract explanation.
    pub message: String,
    /// Source byte offset, when tied to a fragment.
    pub byte_offset: Option<usize>,
    /// Source fragment, when useful and bounded by the query cap.
    pub fragment: Option<String>,
}

/// Complete output of a lenient parse.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedQuery {
    /// Recovered engine-neutral tree.
    pub query: Query,
    /// Incumbent-compatible explanation classification.
    pub explanation: QueryExplanation,
    /// Ordered recovery/truncation diagnostics.
    pub diagnostics: Vec<QueryDiagnostic>,
    /// Whether the input was truncated before lexing.
    pub was_truncated: bool,
}

/// Maximum raw-query bytes admitted by the dev-only CASS observation boundary.
///
/// Shipping parsing still admits [`MAX_QUERY_LENGTH`] Unicode scalar values.
/// This separate byte cap prevents a conformance caller from retaining an
/// arbitrarily large pre-truncation witness.
///
/// [`MAX_QUERY_LENGTH`]: super::MAX_QUERY_LENGTH
#[cfg(any(test, feature = "conformance-internals"))]
pub const MAX_CASS_OBSERVATION_RAW_QUERY_BYTES: usize = 1_048_576;

/// Maximum number of exact filter values admitted by one CASS observation.
#[cfg(any(test, feature = "conformance-internals"))]
pub const MAX_CASS_OBSERVATION_FILTER_VALUES: usize = 4_096;

/// Maximum bytes admitted for any individual exact filter value.
#[cfg(any(test, feature = "conformance-internals"))]
pub const MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES: usize = 4_096;

/// Maximum aggregate bytes admitted across all exact filter values.
#[cfg(any(test, feature = "conformance-internals"))]
pub const MAX_CASS_OBSERVATION_FILTER_BYTES: usize = 1_048_576;

/// One token consumed by the native CASS grammar, including its source offset.
///
/// This is an evidence boundary rather than a second lexer: values are projected
/// from the private token stream that [`CassQueryParser`] actually lowers. Text
/// is intentionally retained here only in memory; durable gauntlet artifacts
/// must redact it before persistence.
///
/// [`CassQueryParser`]: super::CassQueryParser
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(test, feature = "conformance-internals"))]
pub enum CassQueryToken {
    /// A term operand and its starting UTF-8 byte offset.
    Term { text: String, byte_offset: usize },
    /// A quoted phrase operand and its opening-quote byte offset.
    Phrase { text: String, byte_offset: usize },
    /// An explicit conjunction.
    And { byte_offset: usize },
    /// An explicit disjunction.
    Or { byte_offset: usize },
    /// An explicit or prefix negation.
    Not { byte_offset: usize },
    /// A group-opening parenthesis.
    LParen { byte_offset: usize },
    /// A group-closing parenthesis.
    RParen { byte_offset: usize },
}

/// Complete in-memory witness emitted by the real native CASS parser.
///
/// `parsed` is exactly the value returned by [`CassQueryParser::parse`]. The
/// additional fields expose the parser's actual token stream and the named
/// CASS sanitization transform so a conformance adapter does not reconstruct
/// either value from the source corpus.
///
/// [`CassQueryParser::parse`]: super::CassQueryParser::parse
#[derive(Debug, Clone, PartialEq)]
#[cfg(any(test, feature = "conformance-internals"))]
pub struct CassParsedQueryObservation {
    /// Lowered query tree, classification, diagnostics, and truncation state.
    pub parsed: ParsedQuery,
    /// Shipping CASS sanitization applied to the admitted query prefix.
    pub sanitized_query: String,
    /// Exact query prefix admitted to the shipping parser.
    pub admitted_query: String,
    /// Ordered tokens consumed by the native grammar.
    pub tokens: Vec<CassQueryToken>,
}

/// A bounded CASS observation request was rejected before witness allocation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[cfg(any(test, feature = "conformance-internals"))]
pub enum CassQueryObservationError {
    /// One request dimension exceeded its fixed conformance-only cap.
    #[error("CASS observation field {field} has size {actual}, exceeding limit {limit}")]
    BoundExceeded {
        /// Stable request dimension.
        field: &'static str,
        /// Observed size, saturated to `usize::MAX` on arithmetic overflow.
        actual: usize,
        /// Maximum admitted size.
        limit: usize,
    },
}

/// Invalid schema configuration for the shipping default parser.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum QueryParserConfigError {
    /// The schema violates the dense-ID descriptor contract.
    #[error("query schema {schema} is invalid: {detail}")]
    InvalidSchema {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Descriptor validation detail.
        detail: String,
    },
    /// A mandatory default field is absent.
    #[error("default query field {field} is absent from schema {schema}")]
    MissingDefaultField {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Missing field name.
        field: &'static str,
    },
    /// A mandatory default field is not default-analyzed text.
    #[error("default query field {field} in schema {schema} is not default-analyzed text")]
    InvalidDefaultField {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Invalid field name.
        field: &'static str,
    },
}

/// A posting-list capability an operator needs from every field it names.
///
/// Centralizing this (`bd-vmpb`) keeps capability knowledge in one table rather
/// than scattering `positions: true` special cases through lowering, so a new
/// operator cannot silently bypass the check: it must declare what it reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexCapability {
    /// The operator reads a position list, so the field must be indexed with
    /// `positions: true`.
    Positions,
}

impl fmt::Display for IndexCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positions => formatter.write_str("positions"),
        }
    }
}

/// A query names an operator the index cannot serve on that field.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum QueryCapabilityError {
    /// The query arena could not reserve its bounded node storage.
    #[error("query arena could not reserve {requested} entries")]
    ArenaAllocation {
        /// Requested arena entry count.
        requested: usize,
    },
    /// A position-dependent operator named a field indexed without positions.
    #[error(
        "operator {operator} requires {capability} on field {field} of schema {schema}, \
         which is indexed without them"
    )]
    PositionsRequired {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Field the operator named, by schema name where resolvable.
        field: String,
        /// Operator that requires the capability.
        operator: QueryExplanation,
        /// Capability the operator requires.
        capability: IndexCapability,
    },
}

/// Capabilities `query` requires that `schema` does not provide.
///
/// Position-independent operators — `Term`, `Range`, `Set`, `Glob`, `Boolean`,
/// `Boost`, `All`, `Empty` — never read a position list and are therefore valid
/// against a positionless index, matching and ranking identically to a
/// positioned one over the same documents. Only `Phrase` consumes positions.
///
/// A single-term phrase is exempt on purpose: it degenerates to a term match
/// and reads no positions, which is the same rule lowering already applies.
///
/// # Errors
///
/// Returns [`QueryCapabilityError::PositionsRequired`] naming the schema, the
/// field, and the operator — never a bare "unsupported query".
pub fn validate_index_capabilities(
    query: &Query,
    schema: SchemaDescriptor,
) -> Result<(), QueryCapabilityError> {
    let mut pending = Vec::new();
    if pending.try_reserve_exact(query.node_count()).is_err() {
        return Err(QueryCapabilityError::ArenaAllocation {
            requested: query.node_count(),
        });
    }
    pending.push(query.root_id());
    while let Some(node_id) = pending.pop() {
        match query.node(node_id) {
            QueryNode::AllocationFailure { requested } => {
                return Err(QueryCapabilityError::ArenaAllocation {
                    requested: *requested,
                });
            }
            QueryNode::Phrase { fields, terms, .. } if terms.len() >= 2 => {
                for field in fields {
                    let descriptor = schema
                        .fields
                        .iter()
                        .find(|candidate| candidate.id == field.field_id);
                    let positioned = matches!(
                        descriptor.map(|descriptor| descriptor.kind),
                        Some(FieldKind::Text {
                            positions: true,
                            ..
                        })
                    );
                    if !positioned {
                        return Err(QueryCapabilityError::PositionsRequired {
                            schema: schema.name,
                            field: descriptor.map_or_else(
                                || format!("#{}", field.field_id),
                                |descriptor| descriptor.name.to_owned(),
                            ),
                            operator: QueryExplanation::Phrase,
                            capability: IndexCapability::Positions,
                        });
                    }
                }
            }
            QueryNode::Boolean { clauses, .. } => {
                pending.extend(clauses.iter().rev().map(|clause| clause.query));
            }
            QueryNode::Boost { query: child, .. } => pending.push(*child),
            QueryNode::Empty
            | QueryNode::All
            | QueryNode::Term { .. }
            | QueryNode::Phrase { .. }
            | QueryNode::Range { .. }
            | QueryNode::Set { .. }
            | QueryNode::Glob { .. } => {}
        }
    }
    Ok(())
}

/// Escape every colon in free text that does not end one of `schema`'s field
/// names, so code identifiers and URLs stay literal.
///
/// The lenient parser reads `HashMap::new` or `https://x` as `field:value`
/// with an unknown field and drops the fragment. A real field prefix keeps
/// its meaning (`title:rust`, also after a group or negation), `name::x` is
/// never a field prefix, quoted phrases are left alone, and an escape the
/// caller already wrote is kept, so applying this twice changes nothing.
/// Search consumers call it on user text; the parser itself is unchanged.
#[must_use]
pub fn escape_unknown_field_colons(query: &str, schema: SchemaDescriptor) -> Cow<'_, str> {
    if !query.contains(':') {
        return Cow::Borrowed(query);
    }
    let mut escaped_query = String::with_capacity(query.len() + 8);
    let mut term_start = 0;
    let mut in_quotes = false;
    let mut escaped = false;
    for (index, ch) in query.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch == ':' && !in_quotes {
            let name = query.get(term_start..index).unwrap_or_default();
            let doubled = query
                .get(index + 1..)
                .is_some_and(|rest| rest.starts_with(':'));
            let field = !doubled && schema.fields.iter().any(|field| field.name == name);
            if !field {
                escaped_query.push('\\');
            }
        }
        escaped_query.push(ch);
        if ch.is_whitespace() || matches!(ch, '(' | ')' | '+' | '-') {
            term_start = index + ch.len_utf8();
        }
    }
    Cow::Owned(escaped_query)
}
