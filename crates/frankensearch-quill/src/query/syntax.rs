use crate::schema::{Analyzer as AnalyzerKind, FieldDescriptor, FieldKind, SchemaDescriptor};

use super::ast::{Occur, Query, QueryField};
use super::canonical::{
    ParsedQuery, QueryDiagnostic, QueryDiagnosticKind, QueryParserConfigError, classify_query,
};
use super::grammar::Grammar;
use super::lexer::{lex, peel_redundant_outer_groups};
use super::rewrite::{
    collapse_degenerate_groups, repair_root_all_negative, rewrite_parser_syntax,
    trim_parser_dropped,
};
use super::{CONTENT_FIELD_NAME, MAX_QUERY_LENGTH, TITLE_BOOST, TITLE_FIELD_NAME};

/// Shipping lenient parser for the default `[content, title^2]` expansion.
#[derive(Debug, Clone, Copy)]
pub struct DefaultQueryParser {
    pub(super) schema: SchemaDescriptor,
    pub(super) default_fields: [QueryField; 2],
}

impl DefaultQueryParser {
    /// Bind the parser to a schema containing default-analyzed `content` and
    /// `title` fields.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when either mandatory field is missing or
    /// uses a non-default analyzer.
    pub fn new(schema: SchemaDescriptor) -> Result<Self, QueryParserConfigError> {
        schema
            .validate()
            .map_err(|error| QueryParserConfigError::InvalidSchema {
                schema: schema.name,
                detail: error.to_string(),
            })?;
        let content = required_default_field(schema, CONTENT_FIELD_NAME)?;
        let title = required_default_field(schema, TITLE_FIELD_NAME)?;
        Ok(Self {
            schema,
            default_fields: [
                QueryField::new(content.id, 1.0),
                QueryField::new(title.id, TITLE_BOOST),
            ],
        })
    }

    /// Parse arbitrary user input without returning a syntax error.
    #[must_use]
    pub fn parse(&self, query: &str) -> ParsedQuery {
        self.parse_lenient(query)
    }

    /// Explicitly named alias emphasizing that syntax never becomes a search
    /// failure.
    #[must_use]
    pub fn parse_lenient(&self, query: &str) -> ParsedQuery {
        let (query, was_truncated) = truncated_prefix(query);
        let mut diagnostics = Vec::new();
        if was_truncated {
            emit_diagnostic(
                &mut diagnostics,
                QueryDiagnostic {
                    kind: QueryDiagnosticKind::Truncated,
                    message: format!("query truncated to {MAX_QUERY_LENGTH} Unicode scalar values"),
                    byte_offset: Some(query.len()),
                    fragment: None,
                },
            );
        }

        let explanation = classify_query(query);
        let mut tokens = lex(query, &mut diagnostics);
        peel_redundant_outer_groups(&mut tokens);
        let mut grammar = Grammar {
            parser: *self,
            tokens,
            cursor: 0,
            diagnostics,
            lowered_atoms: Vec::new(),
            field_scopes: Vec::new(),
            allocation_failure: None,
        };
        grammar.lower_primary_tokens();
        grammar.collapse_groups();
        grammar.recover_leading_binary_operators();
        let mut parsed = grammar.parse_expression();
        grammar.recover_trailing_tokens();
        if let Some(requested) = parsed
            .as_ref()
            .and_then(|node| node.dedup_key.allocation_failure())
        {
            grammar.allocation_failure("query syntax arena", requested);
        }
        if grammar.allocation_failure.is_none() {
            if let Some(node) = parsed.as_mut() {
                rewrite_parser_syntax(node);
                trim_parser_dropped(&mut node.query, &node.dedup_key);
                // PROTOTYPE (bd-1i4j4): tantivy preserves Should nesting and
                // evaluates it with generic unions, so full flattening broke
                // f32 association parity on every unfielded multi-term shape.
                // Only the flatten's SEMANTIC duty survives: parser-dropped
                // fragments leave empty groups behind, and refusal surfaces
                // key on the collapsed Empty root.
                collapse_degenerate_groups(&mut node.query);
            }
            repair_root_all_negative(&mut parsed, &mut grammar);
        }

        let query = grammar.allocation_failure.map_or_else(
            || parsed.map_or_else(Query::empty, |node| node.query),
            Query::arena_allocation_failure,
        );
        if let Some(requested) = query.allocation_failure() {
            if !grammar
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::AllocationFailure)
            {
                grammar.push_diagnostic(QueryDiagnostic {
                    kind: QueryDiagnosticKind::AllocationFailure,
                    message: format!("query arena could not reserve {requested} entries"),
                    byte_offset: None,
                    fragment: None,
                });
            }
        }
        ParsedQuery {
            query,
            explanation,
            diagnostics: grammar.diagnostics,
            was_truncated,
        }
    }
}

fn required_default_field(
    schema: SchemaDescriptor,
    name: &'static str,
) -> Result<FieldDescriptor, QueryParserConfigError> {
    let Some(field) = schema.fields.iter().find(|field| field.name == name) else {
        return Err(QueryParserConfigError::MissingDefaultField {
            schema: schema.name,
            field: name,
        });
    };
    // Positions are deliberately NOT required here (`bd-vmpb`). A position-free
    // text index is a valid, smaller, faster configuration: every
    // position-independent operator — bare term, prefix, boolean, range, set,
    // glob — matches and ranks identically with or without them, because none
    // of them reads a position list. Requiring `positions: true` to *construct*
    // the parser made a positionless schema unqueryable end to end, so
    // `DefaultQueryParser::new` failed before any query could be parsed and the
    // QG positions_off fixture aborted the whole matrix run.
    //
    // What a default field genuinely requires is the intended analyzer, so the
    // term text a query produces matches the term text indexing produced.
    // Position-dependent operators are rejected later and precisely, by
    // [`validate_index_capabilities`], which names the offending field and
    // operator instead of disabling the schema wholesale.
    if !matches!(
        field.kind,
        FieldKind::Text {
            analyzer: AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams,
            ..
        }
    ) {
        return Err(QueryParserConfigError::InvalidDefaultField {
            schema: schema.name,
            field: name,
        });
    }
    Ok(*field)
}

/// Return the character-safe prefix accepted by the parser and log truncation.
#[must_use]
pub fn truncate_query(query: &str) -> &str {
    let (prefix, was_truncated) = truncated_prefix(query);
    if was_truncated {
        tracing::warn!(
            target: crate::tracing_conventions::TARGET,
            phase = crate::tracing_conventions::ARGUS_QUERY,
            query_len = query.len(),
            "query truncated to MAX_QUERY_LENGTH"
        );
    }
    prefix
}

pub(super) fn truncated_prefix(query: &str) -> (&str, bool) {
    if query.len() <= MAX_QUERY_LENGTH {
        return (query, false);
    }
    query
        .char_indices()
        .nth(MAX_QUERY_LENGTH)
        .map_or((query, false), |(end, _)| (&query[..end], true))
}

pub(super) fn emit_diagnostic(diagnostics: &mut Vec<QueryDiagnostic>, diagnostic: QueryDiagnostic) {
    tracing::warn!(
        target: crate::tracing_conventions::TARGET,
        phase = crate::tracing_conventions::ARGUS_QUERY,
        "lenient query parse diagnostic"
    );
    diagnostics.push(diagnostic);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AtomDelimiter {
    None,
    SingleQuotes,
    DoubleQuotes,
}

impl AtomDelimiter {
    pub(super) const fn from_quote(delimiter: char) -> Self {
        match delimiter {
            '\'' => Self::SingleQuotes,
            '"' => Self::DoubleQuotes,
            _ => Self::None,
        }
    }

    pub(super) const fn is_quoted(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AtomToken {
    pub(super) raw: String,
    pub(super) field: Option<String>,
    pub(super) delimiter: AtomDelimiter,
    pub(super) slop: u32,
    pub(super) prefix: bool,
    pub(super) boost: Option<String>,
    pub(super) byte_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RangeToken {
    pub(super) field: Option<String>,
    pub(super) lower: String,
    pub(super) upper: String,
    pub(super) lower_inclusive: bool,
    pub(super) upper_inclusive: bool,
    pub(super) boost: Option<String>,
    pub(super) byte_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SetToken {
    pub(super) field: Option<String>,
    pub(super) values: Vec<String>,
    pub(super) boost: Option<String>,
    pub(super) byte_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SyntaxNodeId(usize);

impl SyntaxNodeId {
    pub(super) const ROOT: Self = Self(usize::MAX);

    pub(super) const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SyntaxNode {
    Atom {
        field: Option<String>,
        raw: String,
        delimiter: AtomDelimiter,
        slop: u32,
        prefix: bool,
    },
    Range {
        field: Option<String>,
        lower: String,
        upper: String,
        lower_inclusive: bool,
        upper_inclusive: bool,
    },
    Set {
        field: Option<String>,
        values: Vec<String>,
    },
    Regex {
        field: Option<String>,
        pattern: String,
    },
    LoweredAway(SyntaxNodeId),
    Boolean(Vec<(Option<Occur>, SyntaxNodeId)>),
    Boost {
        query: SyntaxNodeId,
        factor_bits: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SyntaxKey {
    pub(super) nodes: Vec<SyntaxNode>,
    root: SyntaxNode,
    allocation_failure: Option<usize>,
}

impl SyntaxKey {
    pub(super) fn leaf(root: SyntaxNode) -> Self {
        Self {
            nodes: Vec::new(),
            root,
            allocation_failure: None,
        }
    }

    pub(super) fn empty_boolean() -> Self {
        Self::leaf(SyntaxNode::Boolean(Vec::new()))
    }

    pub(super) const fn root_id() -> SyntaxNodeId {
        SyntaxNodeId::ROOT
    }

    pub(super) fn node(&self, id: SyntaxNodeId) -> &SyntaxNode {
        if id == SyntaxNodeId::ROOT {
            &self.root
        } else {
            &self.nodes[id.index()]
        }
    }

    pub(super) fn node_mut(&mut self, id: SyntaxNodeId) -> &mut SyntaxNode {
        if id == SyntaxNodeId::ROOT {
            &mut self.root
        } else {
            &mut self.nodes[id.index()]
        }
    }

    fn push(&mut self, node: SyntaxNode) -> SyntaxNodeId {
        debug_assert!(self.nodes.len() < self.nodes.capacity());
        let id = SyntaxNodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub(super) const fn allocation_failure(&self) -> Option<usize> {
        self.allocation_failure
    }

    fn append(&mut self, other: Self) -> SyntaxNodeId {
        let base = self.nodes.len();
        let child_root = base.saturating_add(other.nodes.len());
        let remap = |id: SyntaxNodeId| {
            if id == SyntaxNodeId::ROOT {
                SyntaxNodeId(child_root)
            } else {
                SyntaxNodeId(base.saturating_add(id.index()))
            }
        };
        for node in other.nodes {
            self.push(remap_syntax_node(node, remap));
        }
        self.push(remap_syntax_node(other.root, remap))
    }

    pub(super) fn boolean(keys: Vec<(Option<Occur>, Self)>) -> Self {
        let mut syntax = Self::empty_boolean();
        if let Some(requested) = keys.iter().find_map(|(_, key)| key.allocation_failure()) {
            syntax.allocation_failure = Some(requested);
            return syntax;
        }
        let descendants = keys.iter().fold(0_usize, |count, (_, key)| {
            count.saturating_add(key.nodes.len().saturating_add(1))
        });
        if syntax.nodes.try_reserve_exact(descendants).is_err() {
            syntax.allocation_failure = Some(descendants);
            return syntax;
        }
        let mut children = Vec::new();
        if children.try_reserve_exact(keys.len()).is_err() {
            syntax.allocation_failure = Some(keys.len());
            return syntax;
        }
        for (occur, key) in keys {
            let child = syntax.append(key);
            children.push((occur, child));
        }
        syntax.root = SyntaxNode::Boolean(children);
        syntax
    }

    pub(super) fn boost(query: Self, factor_bits: u64) -> Self {
        if query.allocation_failure().is_some() {
            return query;
        }
        let mut syntax = Self::empty_boolean();
        if syntax
            .nodes
            .try_reserve_exact(query.nodes.len().saturating_add(1))
            .is_err()
        {
            syntax.allocation_failure = Some(query.nodes.len().saturating_add(1));
            return syntax;
        }
        let child = syntax.append(query);
        syntax.root = SyntaxNode::Boost {
            query: child,
            factor_bits,
        };
        syntax
    }

    pub(super) fn mark_lowered_away(&mut self) {
        let mut current = Self::root_id();
        while let SyntaxNode::Boost { query, .. } = self.node(current) {
            current = *query;
        }
        if self.nodes.try_reserve_exact(1).is_err() {
            self.allocation_failure = Some(1);
            return;
        }
        if current == SyntaxNodeId::ROOT {
            let child = std::mem::replace(&mut self.root, SyntaxNode::Boolean(Vec::new()));
            let child = self.push(child);
            self.root = SyntaxNode::LoweredAway(child);
        } else {
            let child = std::mem::replace(
                &mut self.nodes[current.index()],
                SyntaxNode::Boolean(Vec::new()),
            );
            let child = self.push(child);
            self.nodes[current.index()] = SyntaxNode::LoweredAway(child);
        }
    }
}

fn remap_syntax_node(
    node: SyntaxNode,
    remap: impl Fn(SyntaxNodeId) -> SyntaxNodeId + Copy,
) -> SyntaxNode {
    match node {
        SyntaxNode::LoweredAway(query) => SyntaxNode::LoweredAway(remap(query)),
        SyntaxNode::Boolean(mut keys) => {
            for (_, child) in &mut keys {
                *child = remap(*child);
            }
            SyntaxNode::Boolean(keys)
        }
        SyntaxNode::Boost { query, factor_bits } => SyntaxNode::Boost {
            query: remap(query),
            factor_bits,
        },
        leaf => leaf,
    }
}

pub(super) fn take_syntax_key(slot: &mut SyntaxKey) -> SyntaxKey {
    std::mem::replace(slot, SyntaxKey::empty_boolean())
}

pub(super) fn boost_syntax_key(query: SyntaxKey, raw: Option<&str>) -> SyntaxKey {
    let Some(raw) = raw else {
        return query;
    };
    let Some(factor) = raw.parse::<f64>().ok().filter(|factor| *factor >= 0.0) else {
        return query;
    };
    if parse_boost_factor(raw).is_none() {
        return query;
    }
    if (factor - 1.0).abs() <= f64::EPSILON {
        query
    } else {
        SyntaxKey::boost(query, factor.to_bits())
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn empty_query_for_syntax(syntax: &SyntaxKey) -> Query {
    if let Some(requested) = syntax.allocation_failure() {
        return Query::arena_allocation_failure(requested);
    }
    let mut factors = Vec::new();
    if factors
        .try_reserve_exact(syntax.nodes.len().saturating_add(1))
        .is_err()
    {
        return Query::arena_allocation_failure(syntax.nodes.len().saturating_add(1));
    }
    let mut current = SyntaxKey::root_id();
    loop {
        match syntax.node(current) {
            SyntaxNode::LoweredAway(child) => current = *child,
            SyntaxNode::Boost { query, factor_bits } => {
                factors.push(*factor_bits);
                current = *query;
            }
            SyntaxNode::Atom { .. }
            | SyntaxNode::Range { .. }
            | SyntaxNode::Set { .. }
            | SyntaxNode::Regex { .. }
            | SyntaxNode::Boolean(_) => break,
        }
    }
    let mut query = Query::empty();
    for factor_bits in factors.into_iter().rev() {
        // Tantivy parses boost syntax as f64, then lowers it to the f32 scoring
        // domain. Preserve that intentional narrowing, including overflow to
        // infinity, for differential parity.
        let factor = f64::from_bits(factor_bits) as f32;
        // Bit-exact, not approximate: `1.0_f32` has exactly one encoding, so
        // this is the same predicate as `factor != 1.0` for every input,
        // including NaN and the signed zeroes, without inviting an epsilon.
        if factor.to_bits() != 1.0_f32.to_bits() {
            query = Query::boost(query, factor);
        }
    }
    query
}

pub(super) fn mark_syntax_lowered_away(mut syntax: SyntaxKey) -> SyntaxKey {
    syntax.mark_lowered_away();
    syntax
}

pub(super) fn range_dedup_key(token: &RangeToken) -> SyntaxKey {
    boost_syntax_key(
        SyntaxKey::leaf(SyntaxNode::Range {
            field: token.field.clone(),
            lower: token.lower.clone(),
            upper: token.upper.clone(),
            lower_inclusive: token.lower != "*" && token.lower_inclusive,
            upper_inclusive: token.upper != "*" && token.upper_inclusive,
        }),
        token.boost.as_deref(),
    )
}

pub(super) fn set_dedup_key(token: &SetToken) -> SyntaxKey {
    boost_syntax_key(
        SyntaxKey::leaf(SyntaxNode::Set {
            field: token.field.clone(),
            values: token.values.clone(),
        }),
        token.boost.as_deref(),
    )
}

pub(super) fn atom_dedup_key(atom: &AtomToken) -> SyntaxKey {
    boost_syntax_key(
        SyntaxKey::leaf(SyntaxNode::Atom {
            field: atom.field.clone(),
            raw: atom.raw.clone(),
            delimiter: atom.delimiter,
            slop: atom.slop,
            prefix: atom.prefix,
        }),
        atom.boost.as_deref(),
    )
}

pub(super) fn atom_cache_key(atom: &AtomToken) -> Option<String> {
    if atom
        .boost
        .as_deref()
        .is_some_and(|raw| parse_boost_factor(raw).is_none())
    {
        None
    } else {
        Some(format!("{atom:?}"))
    }
}

pub(super) fn parse_boost_factor(raw: &str) -> Option<f32> {
    let (integer, fraction) = raw
        .split_once('.')
        .map_or((raw, None), |(integer, fraction)| (integer, Some(fraction)));
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.is_some_and(|fraction| {
            fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return None;
    }
    raw.parse::<f32>()
        .ok()
        .filter(|factor| factor.is_finite() && *factor >= 0.0)
}
