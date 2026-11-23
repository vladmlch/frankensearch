use std::ops::Bound;

use thiserror::Error;

use crate::schema::{Analyzer as AnalyzerKind, FieldDescriptor, FieldKind, SchemaDescriptor};
use crate::scribe::is_cass_cjk;

use super::MAX_QUERY_LENGTH;
use super::ast::{
    BooleanClause, BooleanOperator, Occur, PositionedTerm, Query, QueryField, QueryValue,
};
#[cfg(any(test, feature = "conformance-internals"))]
use super::canonical::{
    CassParsedQueryObservation, CassQueryObservationError, CassQueryToken,
    MAX_CASS_OBSERVATION_FILTER_BYTES, MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES,
    MAX_CASS_OBSERVATION_FILTER_VALUES, MAX_CASS_OBSERVATION_RAW_QUERY_BYTES,
};
use super::canonical::{ParsedQuery, QueryDiagnostic, QueryDiagnosticKind, classify_query};
use super::syntax::{emit_diagnostic, truncated_prefix};

/// Source restriction used by the native CASS query parser.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CassSourceFilter {
    /// Do not restrict the durable source identity.
    #[default]
    All,
    /// Match documents indexed from the local machine.
    Local,
    /// Match documents indexed through SSH.
    Remote,
    /// Match one exact source identifier.
    SourceId(String),
}

/// Structured filters appended to one native CASS query.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CassQueryFilters {
    /// Exact agent names, combined with OR and required by the root query.
    pub agents: Vec<String>,
    /// Exact workspace names, combined with OR and required by the root query.
    pub workspaces: Vec<String>,
    /// Inclusive lower timestamp bound.
    pub created_from: Option<i64>,
    /// Inclusive upper timestamp bound.
    pub created_to: Option<i64>,
    /// Durable source restriction.
    pub source_filter: CassSourceFilter,
}

/// Shipping wildcard classes for one CASS term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CassWildcardPattern {
    /// No wildcard, lowered through exact term queries.
    Exact(String),
    /// One trailing wildcard, accelerated through the prefix fields.
    Prefix(String),
    /// One leading wildcard, lowered through regex expansion.
    Suffix(String),
    /// Leading and trailing wildcards, lowered through regex expansion.
    Substring(String),
    /// One or more interior wildcards, lowered through regex expansion.
    Complex(String),
}

impl CassWildcardPattern {
    /// Classify and lowercase one sanitized CASS query term.
    #[must_use]
    pub fn parse(term: &str) -> Self {
        let starts_with_star = term.starts_with('*');
        let ends_with_star = term.ends_with('*');
        let core = term.trim_matches('*');
        if core.is_empty() {
            return Self::Exact(String::new());
        }
        if core.contains('*') {
            return Self::Complex(term.to_lowercase());
        }
        let core = core.to_lowercase();
        match (starts_with_star, ends_with_star) {
            (true, true) => Self::Substring(core),
            (true, false) => Self::Suffix(core),
            (false, true) => Self::Prefix(core),
            (false, false) => Self::Exact(core),
        }
    }

    /// Return the anchor-free Tantivy FST regex for regex-backed classes.
    ///
    /// The FST engine matches complete terms and rejects `^` / `$` assertions.
    #[must_use]
    pub fn to_regex(&self) -> Option<String> {
        match self {
            Self::Suffix(core) => Some(format!(".*{}", cass_escape_regex(core))),
            Self::Substring(core) => Some(format!(".*{}.*", cass_escape_regex(core))),
            Self::Complex(pattern) => Some(cass_complex_regex(pattern)),
            Self::Exact(_) | Self::Prefix(_) => None,
        }
    }
}

/// Whether `raw` carries an explicit boolean operator.
///
/// This is a lexical question, not a shape question: `QueryExplanation::Boolean`
/// answers "more than one fragment", which is true of `alpha beta` and says
/// nothing about whether the author wrote `AND`, `OR`, or `NOT`. A consumer
/// that branches on operator presence — cass does — needs the lexer's answer,
/// so this runs the same lexer the parser runs and reports whether it produced
/// an operator token.
///
/// Recovery diagnostics from the lex are discarded: a caller asking this
/// question is deciding how to route the query, not reporting on it.
#[must_use]
pub fn cass_has_boolean_operators(raw: &str) -> bool {
    let mut diagnostics = Vec::new();
    cass_lex(raw, &mut diagnostics).iter().any(|token| {
        matches!(
            token,
            CassLexToken::And { .. } | CassLexToken::Or { .. } | CassLexToken::Not { .. }
        )
    })
}

/// Sanitize a CASS query using the shipping hyphen-normalize boundary.
///
/// Alphanumeric scalars, wildcards, quotes, and hyphens survive. Every other
/// scalar becomes a space so later splitting matches the indexed tokenizer.
#[must_use]
pub fn cass_sanitize_query(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_alphanumeric() || matches!(ch, '*' | '"' | '-') {
                ch
            } else {
                ' '
            }
        })
        .collect()
}

fn cass_escape_regex(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len().saturating_mul(2));
    for ch in value.chars() {
        if matches!(
            ch,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
        ) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

fn cass_complex_regex(pattern: &str) -> String {
    let mut regex = String::with_capacity(pattern.len().saturating_mul(2).saturating_add(2));
    if pattern.starts_with('*') {
        regex.push_str(".*");
    }
    let core = pattern.trim_start_matches('*').trim_end_matches('*');
    for ch in core.chars() {
        if ch == '*' {
            regex.push_str(".*");
        } else {
            if matches!(
                ch,
                '\\' | '.' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
            ) {
                regex.push('\\');
            }
            regex.push(ch);
        }
    }
    if pattern.ends_with('*') {
        regex.push_str(".*");
    }
    regex
}

/// Invalid schema configuration for [`CassQueryParser`].
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum CassQueryParserConfigError {
    /// The schema violates the dense descriptor contract.
    #[error("CASS query schema {schema} is invalid: {detail}")]
    InvalidSchema {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Descriptor validation detail.
        detail: String,
    },
    /// A mandatory CASS field is absent.
    #[error("CASS query field {field} is absent from schema {schema}")]
    MissingField {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Missing field name.
        field: &'static str,
    },
    /// A mandatory field has the wrong durable shape.
    #[error("CASS query field {field} has an incompatible shape in schema {schema}")]
    InvalidField {
        /// Schema diagnostic name.
        schema: &'static str,
        /// Invalid field name.
        field: &'static str,
    },
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CassParserFields {
    pub(super) agent: u16,
    pub(super) workspace: u16,
    pub(super) created_at: u16,
    pub(super) title: u16,
    pub(super) content: u16,
    pub(super) title_prefix: u16,
    pub(super) content_prefix: u16,
    pub(super) source_id: u16,
    pub(super) origin_kind: u16,
}

impl CassParserFields {
    pub(super) fn searchable(self) -> Vec<QueryField> {
        [
            self.title,
            self.content,
            self.title_prefix,
            self.content_prefix,
        ]
        .into_iter()
        .map(|field_id| QueryField::new(field_id, 1.0))
        .collect()
    }

    pub(super) fn regex_fields(self) -> Vec<u16> {
        Vec::from([self.content, self.title])
    }
}

/// Native parser for the CASS Boolean grammar.
///
/// Standard precedence: NOT binds tightest, then AND (explicit, `&&`, or
/// implied between adjacent operands), then OR (`OR`, `||`). Parentheses
/// group. A `(` opens a group only at the start of a word and a `)` closes
/// one only while a group is open, so code such as `foo(bar)` stays a term.
/// Negation is parity-based (`NOT NOT x` is `x`), and a negative used as an
/// OR operand or as the complete root is wrapped in `All + MustNot` so it
/// denotes a complement. Malformed input is recovered with a
/// [`QueryDiagnosticKind::SyntaxRecovery`] diagnostic rather than an error.
#[derive(Debug, Clone, Copy)]
pub struct CassQueryParser {
    pub(super) fields: CassParserFields,
}

impl CassQueryParser {
    /// Bind the parser to a CASS-compatible schema.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error for a missing or incompatible
    /// field, or for an invalid schema descriptor.
    pub fn new(schema: SchemaDescriptor) -> Result<Self, CassQueryParserConfigError> {
        schema
            .validate()
            .map_err(|error| CassQueryParserConfigError::InvalidSchema {
                schema: schema.name,
                detail: error.to_string(),
            })?;
        let keyword = |kind| kind == FieldKind::Keyword;
        let cass_text = |kind| {
            matches!(
                kind,
                FieldKind::Text {
                    analyzer: AnalyzerKind::CassHyphenNormalize,
                    positions: true,
                }
            )
        };
        let cass_prefix = |kind| {
            matches!(
                kind,
                FieldKind::Text {
                    analyzer: AnalyzerKind::CassPrefixNormalize,
                    positions: false,
                }
            )
        };
        let created_at = |kind| {
            matches!(
                kind,
                FieldKind::I64 {
                    indexed: true,
                    fast: true,
                }
            )
        };
        Ok(Self {
            fields: CassParserFields {
                agent: required_cass_field(schema, "agent", keyword)?.id,
                workspace: required_cass_field(schema, "workspace", keyword)?.id,
                created_at: required_cass_field(schema, "created_at", created_at)?.id,
                title: required_cass_field(schema, "title", cass_text)?.id,
                content: required_cass_field(schema, "content", cass_text)?.id,
                title_prefix: required_cass_field(schema, "title_prefix", cass_prefix)?.id,
                content_prefix: required_cass_field(schema, "content_prefix", cass_prefix)?.id,
                source_id: required_cass_field(schema, "source_id", keyword)?.id,
                origin_kind: required_cass_field(schema, "origin_kind", keyword)?.id,
            },
        })
    }

    /// Parse one CASS query and append its structured filters.
    #[must_use]
    pub fn parse(&self, raw_query: &str, filters: &CassQueryFilters) -> ParsedQuery {
        let (admitted_query, was_truncated) = truncated_prefix(raw_query);
        self.parse_admitted(admitted_query, was_truncated, filters, |_| {})
    }

    /// Parse one CASS query and retain the exact lexer/lowering witness.
    ///
    /// This calls the same implementation as [`Self::parse`]. It exists for
    /// dev-only conformance adapters that must observe the real parser boundary
    /// without echoing or independently reparsing the input corpus.
    /// # Errors
    ///
    /// Returns a typed bound error before parsing or cloning evidence when the
    /// raw query, filter count, individual filter value, or aggregate filter
    /// bytes exceed the diagnostic contract.
    #[cfg(any(test, feature = "conformance-internals"))]
    pub fn parse_observed(
        &self,
        raw_query: &str,
        filters: &CassQueryFilters,
    ) -> Result<CassParsedQueryObservation, CassQueryObservationError> {
        cass_observation_preflight(raw_query, filters)?;
        let (admitted_query, was_truncated) = truncated_prefix(raw_query);
        let mut observed_tokens = Vec::new();
        let parsed = self.parse_admitted(admitted_query, was_truncated, filters, |tokens| {
            observed_tokens.extend(tokens.iter().map(CassLexToken::observed));
        });
        Ok(CassParsedQueryObservation {
            parsed,
            sanitized_query: cass_sanitize_query(admitted_query),
            admitted_query: admitted_query.to_owned(),
            tokens: observed_tokens,
        })
    }

    fn parse_admitted<Observe>(
        &self,
        admitted_query: &str,
        was_truncated: bool,
        filters: &CassQueryFilters,
        mut observe_tokens: Observe,
    ) -> ParsedQuery
    where
        Observe: FnMut(&[CassLexToken]),
    {
        let mut diagnostics = Vec::new();
        if was_truncated {
            emit_diagnostic(
                &mut diagnostics,
                QueryDiagnostic {
                    kind: QueryDiagnosticKind::Truncated,
                    message: format!(
                        "CASS query truncated to {MAX_QUERY_LENGTH} Unicode scalar values"
                    ),
                    byte_offset: Some(admitted_query.len()),
                    fragment: None,
                },
            );
        }
        let tokens = cass_lex(admitted_query, &mut diagnostics);
        observe_tokens(&tokens);
        let mut grammar = CassGrammar {
            parser: *self,
            tokens,
            position: 0,
            open_groups: 0,
            diagnostics,
        };
        let parsed = grammar.parse();
        let root = parsed.map_or_else(Query::all, |node| {
            if node.negative {
                cass_complement(node.query)
            } else {
                node.query
            }
        });
        let query = self.apply_filters(root, filters);
        if let Some(requested) = query.allocation_failure() {
            emit_diagnostic(
                &mut grammar.diagnostics,
                QueryDiagnostic {
                    kind: QueryDiagnosticKind::AllocationFailure,
                    message: format!("query arena could not reserve {requested} entries"),
                    byte_offset: None,
                    fragment: None,
                },
            );
        }
        ParsedQuery {
            query,
            explanation: classify_query(admitted_query),
            diagnostics: grammar.diagnostics,
            was_truncated,
        }
    }

    fn apply_filters(&self, root: Query, filters: &CassQueryFilters) -> Query {
        if filters.agents.is_empty()
            && filters.workspaces.is_empty()
            && filters.created_from.is_none()
            && filters.created_to.is_none()
            && filters.source_filter == CassSourceFilter::All
        {
            return root;
        }
        let mut clauses = Vec::new();
        if clauses.try_reserve_exact(5).is_err() {
            return Query::arena_allocation_failure(5);
        }
        clauses.push(BooleanClause::new(Occur::Must, root));
        if let Some(filter) = cass_string_filter(self.fields.agent, &filters.agents) {
            clauses.push(BooleanClause::new(Occur::Must, filter));
        }
        if let Some(filter) = cass_string_filter(self.fields.workspace, &filters.workspaces) {
            clauses.push(BooleanClause::new(Occur::Must, filter));
        }
        if filters.created_from.is_some() || filters.created_to.is_some() {
            clauses.push(BooleanClause::new(
                Occur::Must,
                Query::range(
                    self.fields.created_at,
                    filters.created_from.map_or(Bound::Unbounded, |value| {
                        Bound::Included(QueryValue::I64(value))
                    }),
                    filters.created_to.map_or(Bound::Unbounded, |value| {
                        Bound::Included(QueryValue::I64(value))
                    }),
                ),
            ));
        }
        let source = match &filters.source_filter {
            CassSourceFilter::All => None,
            CassSourceFilter::Local => Some((self.fields.origin_kind, "local")),
            CassSourceFilter::Remote => Some((self.fields.origin_kind, "ssh")),
            CassSourceFilter::SourceId(source_id) => {
                Some((self.fields.source_id, source_id.as_str()))
            }
        };
        if let Some((field_id, value)) = source {
            clauses.push(BooleanClause::new(
                Occur::Must,
                cass_exact_term(field_id, value.to_owned()),
            ));
        }
        if clauses.len() == 1 {
            clauses.pop().map_or_else(Query::all, |clause| clause.query)
        } else {
            Query::boolean(clauses, None)
        }
    }

    fn lower_term(&self, raw: &str) -> Query {
        let parts = cass_sanitize_query(raw)
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        self.lower_compound(&parts)
    }

    fn lower_compound(&self, parts: &[String]) -> Query {
        let queries = parts
            .iter()
            .filter_map(|part| self.lower_term_part(part))
            .collect::<Vec<_>>();
        cass_required_query(queries)
    }

    fn lower_term_part(&self, raw: &str) -> Option<Query> {
        let pattern = CassWildcardPattern::parse(raw);
        match pattern {
            CassWildcardPattern::Exact(term) | CassWildcardPattern::Prefix(term) => {
                if term.is_empty() {
                    return None;
                }
                if term.chars().any(is_cass_cjk) {
                    let terms = cass_cjk_terms(&term);
                    return Some(cass_required_query(
                        terms
                            .into_iter()
                            .map(|text| Query::term(self.fields.searchable(), text))
                            .collect(),
                    ));
                }
                // CASS fork (cass-prefix-expansion): exact and prefix
                // literals lower to two OR'd arms instead of the incumbent
                // single term query. The first arm is the incumbent four
                // field `TermQuery` (`title`, `content`, `title_prefix`,
                // `content_prefix`); the second arm expands `term*` as a
                // trailing-star glob over the unwindowed `content` field.
                // The `content_prefix` edge-ngram column is built over only
                // the first 4 KiB of each document, so the incumbent arm
                // alone silently missed every document whose only matching
                // token sits past that window (`docker` matched
                // `dockerfile` inside the window, but not at offset 5 KiB).
                // The glob arm walks each leaf's own `content` term
                // dictionary (`leaf_glob_terms`), so a prefix that expands
                // past the per-leaf cap is a typed
                // `GlobExpansionLimitExceeded` refusal naming the pattern
                // and a query whose expansion work exceeds the fuel budget
                // is a typed `QueryFuelExhausted` refusal - never a silent
                // under-match. Because this exact arm walks the same token
                // over the full `content` field, the lowering recognizes
                // this two-arm shape (`two_arm_variants_glob` in
                // `index.rs`) and drops the glob arm's own literal term
                // from the expansion: every document that literal names is
                // already matched by the Term arm, and decoding its posting
                // list again was the single largest cost of a common-term
                // arm. The arm's constant boost therefore lands on documents
                // a proper extension of the term pulled in (`errors`, not
                // `error`); a document whose only globset membership was the
                // literal keeps its Term-arm match and BM25 and loses only
                // the flat boost. CJK literals keep the bigram lowering
                // above: the decomposition already covers the unwindowed
                // `content` field, and bigram globs would explode the
                // expansion.
                Some(Query::boolean(
                    Vec::from([
                        BooleanClause::new(
                            Occur::Should,
                            Query::term(self.fields.searchable(), term.clone()),
                        ),
                        BooleanClause::new(
                            Occur::Should,
                            Query::glob(Vec::from([self.fields.content]), format!("{term}*")),
                        ),
                    ]),
                    Some(BooleanOperator::Or),
                ))
            }
            CassWildcardPattern::Suffix(_)
            | CassWildcardPattern::Substring(_)
            | CassWildcardPattern::Complex(_) => {
                Some(Query::glob(self.fields.regex_fields(), raw.to_lowercase()))
            }
        }
    }

    fn lower_phrase(&self, raw: &str) -> Query {
        let terms = cass_sanitize_query(raw)
            .split_whitespace()
            .map(|term| term.trim_matches('*').to_lowercase())
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>();
        if terms.len() <= 1 || terms.iter().any(|term| term.chars().any(is_cass_cjk)) {
            return self.lower_compound(&terms);
        }
        Query::phrase(
            Vec::from([
                QueryField::new(self.fields.title, 1.0),
                QueryField::new(self.fields.content, 1.0),
            ]),
            terms
                .into_iter()
                .zip(0_u32..)
                .map(|(text, position)| PositionedTerm::new(position, text))
                .collect(),
            0,
            false,
        )
    }
}

fn required_cass_field<F>(
    schema: SchemaDescriptor,
    name: &'static str,
    accepts: F,
) -> Result<FieldDescriptor, CassQueryParserConfigError>
where
    F: FnOnce(FieldKind) -> bool,
{
    let Some(field) = schema.fields.iter().find(|field| field.name == name) else {
        return Err(CassQueryParserConfigError::MissingField {
            schema: schema.name,
            field: name,
        });
    };
    if !accepts(field.kind) {
        return Err(CassQueryParserConfigError::InvalidField {
            schema: schema.name,
            field: name,
        });
    }
    Ok(*field)
}

fn cass_string_filter(field_id: u16, values: &[String]) -> Option<Query> {
    let clauses = values
        .iter()
        .map(|value| BooleanClause::new(Occur::Should, cass_exact_term(field_id, value.to_owned())))
        .collect::<Vec<_>>();
    (!clauses.is_empty()).then(|| Query::boolean(clauses, None))
}

fn cass_exact_term(field_id: u16, text: String) -> Query {
    Query::term(Vec::from([QueryField::new(field_id, 1.0)]), text)
}

fn cass_required_query(mut queries: Vec<Query>) -> Query {
    queries.retain(|query| !query.is_empty());
    match queries.len() {
        0 => Query::empty(),
        1 => queries.pop().unwrap_or_else(Query::empty),
        _ => Query::boolean(
            queries
                .into_iter()
                .map(|query| BooleanClause::new(Occur::Must, query))
                .collect(),
            Some(BooleanOperator::And),
        ),
    }
}

fn cass_cjk_terms(term: &str) -> Vec<String> {
    let chars = term
        .chars()
        .filter(|ch| is_cass_cjk(*ch))
        .collect::<Vec<_>>();
    if chars.len() <= 1 {
        return chars.into_iter().map(|ch| ch.to_string()).collect();
    }
    chars
        .windows(2)
        .map(|pair| pair.iter().collect::<String>())
        .collect()
}

fn cass_complement(query: Query) -> Query {
    Query::boolean(
        Vec::from([
            BooleanClause::new(Occur::Must, Query::all()),
            BooleanClause::new(Occur::MustNot, query),
        ]),
        None,
    )
}

#[cfg(any(test, feature = "conformance-internals"))]
fn cass_observation_preflight(
    raw_query: &str,
    filters: &CassQueryFilters,
) -> Result<(), CassQueryObservationError> {
    cass_observation_bound(
        "raw_query_bytes",
        raw_query.len(),
        MAX_CASS_OBSERVATION_RAW_QUERY_BYTES,
    )?;

    let source_value = match &filters.source_filter {
        CassSourceFilter::SourceId(value) => Some(value.as_str()),
        CassSourceFilter::All | CassSourceFilter::Local | CassSourceFilter::Remote => None,
    };
    let filter_count = filters
        .agents
        .len()
        .checked_add(filters.workspaces.len())
        .and_then(|count| count.checked_add(usize::from(source_value.is_some())))
        .unwrap_or(usize::MAX);
    cass_observation_bound(
        "filter_value_count",
        filter_count,
        MAX_CASS_OBSERVATION_FILTER_VALUES,
    )?;

    let mut aggregate_bytes = 0_usize;
    for value in filters
        .agents
        .iter()
        .chain(&filters.workspaces)
        .map(String::as_str)
        .chain(source_value)
    {
        cass_observation_bound(
            "filter_value_bytes",
            value.len(),
            MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES,
        )?;
        aggregate_bytes = aggregate_bytes.saturating_add(value.len());
        cass_observation_bound(
            "aggregate_filter_bytes",
            aggregate_bytes,
            MAX_CASS_OBSERVATION_FILTER_BYTES,
        )?;
    }
    Ok(())
}

#[cfg(any(test, feature = "conformance-internals"))]
fn cass_observation_bound(
    field: &'static str,
    actual: usize,
    limit: usize,
) -> Result<(), CassQueryObservationError> {
    if actual > limit {
        return Err(CassQueryObservationError::BoundExceeded {
            field,
            actual,
            limit,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CassLexToken {
    Term { text: String, offset: usize },
    Phrase { text: String, offset: usize },
    And { offset: usize },
    Or { offset: usize },
    Not { offset: usize },
    LParen { offset: usize },
    RParen { offset: usize },
}

impl CassLexToken {
    #[cfg(any(test, feature = "conformance-internals"))]
    fn observed(&self) -> CassQueryToken {
        match self {
            Self::Term { text, offset } => CassQueryToken::Term {
                text: text.clone(),
                byte_offset: *offset,
            },
            Self::Phrase { text, offset } => CassQueryToken::Phrase {
                text: text.clone(),
                byte_offset: *offset,
            },
            Self::And { offset } => CassQueryToken::And {
                byte_offset: *offset,
            },
            Self::Or { offset } => CassQueryToken::Or {
                byte_offset: *offset,
            },
            Self::Not { offset } => CassQueryToken::Not {
                byte_offset: *offset,
            },
            Self::LParen { offset } => CassQueryToken::LParen {
                byte_offset: *offset,
            },
            Self::RParen { offset } => CassQueryToken::RParen {
                byte_offset: *offset,
            },
        }
    }
}

pub(super) fn cass_lex(query: &str, diagnostics: &mut Vec<QueryDiagnostic>) -> Vec<CassLexToken> {
    let mut tokens = Vec::new();
    let mut chars = query.char_indices().peekable();
    let mut word = String::new();
    let mut word_offset = 0_usize;
    // Groups currently open. A `(` inside a word and a `)` with no group
    // open stay word characters, so code terms like `foo(bar)` survive.
    let mut open_groups = 0_usize;
    while let Some((offset, ch)) = chars.next() {
        match ch {
            '(' if word.is_empty() => {
                tokens.push(CassLexToken::LParen { offset });
                open_groups += 1;
            }
            ')' if open_groups > 0 => {
                cass_flush_word(&mut tokens, &mut word, word_offset);
                tokens.push(CassLexToken::RParen { offset });
                open_groups -= 1;
            }
            '"' => {
                cass_flush_word(&mut tokens, &mut word, word_offset);
                let mut phrase = String::new();
                let mut closed = false;
                for (_, next) in chars.by_ref() {
                    if next == '"' {
                        closed = true;
                        break;
                    }
                    phrase.push(next);
                }
                if !phrase.is_empty() {
                    tokens.push(CassLexToken::Phrase {
                        text: phrase,
                        offset,
                    });
                }
                if !closed {
                    emit_diagnostic(
                        diagnostics,
                        QueryDiagnostic {
                            kind: QueryDiagnosticKind::SyntaxRecovery,
                            message: "CASS syntax recovery: unterminated phrase".to_owned(),
                            byte_offset: Some(offset),
                            fragment: None,
                        },
                    );
                }
            }
            '&' if chars.peek().is_some_and(|(_, next)| *next == '&') => {
                chars.next();
                cass_flush_word(&mut tokens, &mut word, word_offset);
                tokens.push(CassLexToken::And { offset });
            }
            '|' if chars.peek().is_some_and(|(_, next)| *next == '|') => {
                chars.next();
                cass_flush_word(&mut tokens, &mut word, word_offset);
                tokens.push(CassLexToken::Or { offset });
            }
            '-' if word.is_empty() => tokens.push(CassLexToken::Not { offset }),
            ' ' | '\t' | '\n' => {
                cass_flush_word(&mut tokens, &mut word, word_offset);
            }
            _ => {
                if word.is_empty() {
                    word_offset = offset;
                }
                word.push(ch);
            }
        }
    }
    cass_flush_word(&mut tokens, &mut word, word_offset);
    tokens
}

fn cass_flush_word(tokens: &mut Vec<CassLexToken>, word: &mut String, offset: usize) {
    if word.is_empty() {
        return;
    }
    let text = std::mem::take(word);
    let token = if text.eq_ignore_ascii_case("AND") {
        CassLexToken::And { offset }
    } else if text.eq_ignore_ascii_case("OR") {
        CassLexToken::Or { offset }
    } else if text.eq_ignore_ascii_case("NOT") {
        CassLexToken::Not { offset }
    } else {
        CassLexToken::Term { text, offset }
    };
    tokens.push(token);
}

#[derive(Debug)]
struct CassNode {
    query: Query,
    negative: bool,
}

/// Recursive-descent parser over the lexed CASS tokens:
///
/// ```text
/// query   := or
/// or      := and (OR and)*
/// and     := unary ([AND] unary)*      adjacent operands are an implied AND
/// unary   := NOT* primary              an odd count of NOTs negates
/// primary := TERM | PHRASE | '(' or ')'
/// ```
struct CassGrammar {
    parser: CassQueryParser,
    tokens: Vec<CassLexToken>,
    position: usize,
    /// Parentheses entered and not yet closed.
    open_groups: usize,
    diagnostics: Vec<QueryDiagnostic>,
}

impl CassGrammar {
    fn parse(&mut self) -> Option<CassNode> {
        self.parse_or()
    }

    fn peek(&self) -> Option<&CassLexToken> {
        self.tokens.get(self.position)
    }

    /// A `)` outside any group cannot reach the parser, because the lexer
    /// emits one only while a group is open; it is recovered, not trusted.
    fn parse_or(&mut self) -> Option<CassNode> {
        let mut operands = Vec::new();
        let mut pending_or = None;
        loop {
            match self.peek() {
                None => break,
                Some(CassLexToken::RParen { .. }) if self.open_groups > 0 => break,
                Some(CassLexToken::RParen { offset }) => {
                    let offset = *offset;
                    self.position += 1;
                    self.syntax_diagnostic("unmatched ')' was ignored", offset);
                    continue;
                }
                Some(CassLexToken::Or { offset }) => {
                    let offset = *offset;
                    self.position += 1;
                    if operands.is_empty() || pending_or.is_some() {
                        self.syntax_diagnostic(
                            "OR without an adjacent operand was recovered",
                            offset,
                        );
                    }
                    pending_or = Some(offset);
                    continue;
                }
                Some(_) => {}
            }
            if let Some(node) = self.parse_and() {
                operands.push(node);
                pending_or = None;
            }
        }
        if let Some(offset) = pending_or {
            self.syntax_diagnostic("dangling binary operator has no operand", offset);
        }
        cass_combine_or(operands)
    }

    fn parse_and(&mut self) -> Option<CassNode> {
        let mut operands = Vec::new();
        let mut pending_and = None;
        loop {
            match self.peek() {
                None | Some(CassLexToken::Or { .. } | CassLexToken::RParen { .. }) => break,
                Some(CassLexToken::And { offset }) => {
                    let offset = *offset;
                    self.position += 1;
                    if operands.is_empty() || pending_and.is_some() {
                        self.syntax_diagnostic(
                            "AND without an adjacent operand was recovered",
                            offset,
                        );
                    }
                    pending_and = Some(offset);
                    continue;
                }
                Some(_) => {}
            }
            if let Some(node) = self.parse_unary() {
                operands.push(node);
            }
            pending_and = None;
        }
        if let Some(offset) = pending_and {
            self.syntax_diagnostic("dangling binary operator has no operand", offset);
        }
        cass_combine_and(operands)
    }

    fn parse_unary(&mut self) -> Option<CassNode> {
        let mut negated = false;
        let mut first_not = None;
        while let Some(CassLexToken::Not { offset }) = self.peek() {
            first_not.get_or_insert(*offset);
            negated = !negated;
            self.position += 1;
        }
        match (self.peek(), first_not) {
            (
                Some(
                    CassLexToken::Term { .. }
                    | CassLexToken::Phrase { .. }
                    | CassLexToken::LParen { .. },
                ),
                _,
            ) => {}
            (Some(CassLexToken::And { .. }), Some(offset)) => {
                self.syntax_diagnostic("NOT has no operand before AND", offset);
                return None;
            }
            (_, Some(offset)) => {
                self.syntax_diagnostic("dangling NOT has no operand", offset);
                return None;
            }
            (_, None) => return None,
        }
        let mut node = self.parse_primary()?;
        if negated {
            node.negative = !node.negative;
        }
        Some(node)
    }

    fn parse_primary(&mut self) -> Option<CassNode> {
        let token = self.tokens.get(self.position)?.clone();
        self.position += 1;
        match token {
            CassLexToken::Term { text, offset } => {
                let query = self.parser.lower_term(&text);
                if query.is_empty() {
                    self.syntax_diagnostic("empty term operand was skipped", offset);
                    return None;
                }
                Some(CassNode {
                    query,
                    negative: false,
                })
            }
            CassLexToken::Phrase { text, offset } => {
                let query = self.parser.lower_phrase(&text);
                if query.is_empty() {
                    self.syntax_diagnostic("empty phrase operand was skipped", offset);
                    return None;
                }
                Some(CassNode {
                    query,
                    negative: false,
                })
            }
            CassLexToken::LParen { offset } => {
                self.open_groups += 1;
                let inner = self.parse_or();
                self.open_groups -= 1;
                if matches!(self.peek(), Some(CassLexToken::RParen { .. })) {
                    self.position += 1;
                } else {
                    self.syntax_diagnostic("unclosed '(' was closed at the end", offset);
                }
                if inner.is_none() {
                    self.syntax_diagnostic("empty group was skipped", offset);
                }
                inner
            }
            // Operators never reach here: parse_unary admits only operands.
            CassLexToken::And { .. }
            | CassLexToken::Or { .. }
            | CassLexToken::Not { .. }
            | CassLexToken::RParen { .. } => None,
        }
    }

    fn syntax_diagnostic(&mut self, message: &str, offset: usize) {
        emit_diagnostic(
            &mut self.diagnostics,
            QueryDiagnostic {
                kind: QueryDiagnosticKind::SyntaxRecovery,
                message: format!("CASS syntax recovery: {message}"),
                byte_offset: Some(offset),
                fragment: None,
            },
        );
    }
}

/// Disjunction of `operands`. A negative operand denotes its complement, so
/// it is wrapped in `All + MustNot` before joining the disjunction.
fn cass_combine_or(mut operands: Vec<CassNode>) -> Option<CassNode> {
    if operands.len() <= 1 {
        return operands.pop();
    }
    let clauses = operands
        .into_iter()
        .map(|node| {
            let query = if node.negative {
                cass_complement(node.query)
            } else {
                node.query
            };
            BooleanClause::new(Occur::Should, query)
        })
        .collect();
    Some(CassNode {
        query: Query::boolean(clauses, Some(BooleanOperator::Or)),
        negative: false,
    })
}

/// Conjunction of `operands`: positives are `Must`, negatives `MustNot`. A
/// conjunction of negatives alone is anchored on `All`, since exclusions
/// need a universe to exclude from.
fn cass_combine_and(mut operands: Vec<CassNode>) -> Option<CassNode> {
    if operands.len() <= 1 {
        return operands.pop();
    }
    let mut clauses: Vec<BooleanClause> = operands
        .into_iter()
        .map(|node| {
            let occur = if node.negative {
                Occur::MustNot
            } else {
                Occur::Must
            };
            BooleanClause::new(occur, node.query)
        })
        .collect();
    if clauses.iter().all(|clause| clause.occur == Occur::MustNot) {
        if clauses.try_reserve_exact(1).is_err() {
            return None;
        }
        clauses.insert(0, BooleanClause::new(Occur::Must, Query::all()));
    }
    Some(CassNode {
        query: Query::boolean(clauses, Some(BooleanOperator::And)),
        negative: false,
    })
}
