use frankensearch_core::error::SearchResult;
use tantivy::Term;
use tantivy::query::{
    AllQuery, BooleanQuery, Occur, PhraseQuery, Query, RangeQuery, RegexQuery, TermQuery,
};
use tantivy::schema::{Field, IndexRecordOption};
use tracing::warn;

#[cfg(any(test, feature = "bench-internals"))]
use super::query::cass_tokens_have_boolean_operators;
use super::query::{
    CassQueryFilters, CassQueryToken, CassSourceFilter, CassWildcardPattern,
    cass_has_boolean_operators, cass_parse_boolean_query, cass_regex_query_cached,
    cass_sanitize_query,
};
use super::schema::CassFields;
use super::tokenizer::is_cjk;

/// Normalize a term into tokenizer-aligned parts (preserving `*` for wildcards).
#[must_use]
fn cass_normalize_term_parts(raw: &str) -> Vec<String> {
    cass_sanitize_query(raw)
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// Normalize phrase text into tokenizer-aligned terms (lowercased, no wildcards).
#[must_use]
fn cass_normalize_phrase_terms(raw: &str) -> Vec<String> {
    cass_sanitize_query(raw)
        .split_whitespace()
        .map(|s| s.trim_matches('*').to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// The complement of `query`: Tantivy reads a query of `MustNot` clauses
/// alone as match-none, so exclusions are anchored on `AllQuery`.
fn cass_wrap_negated_clause(query: Box<dyn Query>) -> Box<dyn Query> {
    Box::new(BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery)),
        (Occur::MustNot, query),
    ]))
}

/// A lowered CASS operand and whether it stands negated.
struct CassOracleNode {
    expr: CassOracleExpr,
    negative: bool,
}

/// Conjunctions keep their clause list, so a conjunction at the root is
/// emitted as flat top-level clauses: the query tree, and so the score, of
/// every query the pre-grouping builder read the same way is unchanged.
enum CassOracleExpr {
    Leaf(Box<dyn Query>),
    /// `Must`/`MustNot` clauses, anchored on `AllQuery` when all are negative.
    And(Vec<(Occur, Box<dyn Query>)>),
    /// Disjunction operands, negative ones already complemented.
    Or(Vec<Box<dyn Query>>),
}

impl CassOracleExpr {
    fn into_query(self) -> Box<dyn Query> {
        match self {
            Self::Leaf(query) => query,
            Self::And(clauses) => Box::new(BooleanQuery::new(clauses)),
            Self::Or(operands) => Box::new(BooleanQuery::new(
                operands
                    .into_iter()
                    .map(|query| (Occur::Should, query))
                    .collect(),
            )),
        }
    }
}

/// Disjunction of `operands`. A negative operand denotes its complement, so
/// it is wrapped in `All + MustNot` before joining the disjunction.
fn cass_combine_or(mut operands: Vec<CassOracleNode>) -> Option<CassOracleNode> {
    if operands.len() <= 1 {
        return operands.pop();
    }
    let operands = operands
        .into_iter()
        .map(|node| {
            let query = node.expr.into_query();
            if node.negative {
                cass_wrap_negated_clause(query)
            } else {
                query
            }
        })
        .collect();
    Some(CassOracleNode {
        expr: CassOracleExpr::Or(operands),
        negative: false,
    })
}

/// Conjunction of `operands`: positives are `Must`, negatives `MustNot`. A
/// conjunction of negatives alone is anchored on `AllQuery`, since
/// exclusions need a universe to exclude from.
fn cass_combine_and(mut operands: Vec<CassOracleNode>) -> Option<CassOracleNode> {
    if operands.len() <= 1 {
        return operands.pop();
    }
    let mut clauses = operands
        .into_iter()
        .map(|node| {
            let occur = if node.negative {
                Occur::MustNot
            } else {
                Occur::Must
            };
            (occur, node.expr.into_query())
        })
        .collect::<Vec<_>>();
    if clauses.iter().all(|(occur, _)| *occur == Occur::MustNot) {
        clauses.insert(0, (Occur::Must, Box::new(AllQuery)));
    }
    Some(CassOracleNode {
        expr: CassOracleExpr::And(clauses),
        negative: false,
    })
}

/// Recursive-descent CASS grammar, the one Quill's `CassQueryParser` parses:
///
/// ```text
/// or      := and (OR and)*
/// and     := unary ([AND] unary)*      adjacent operands are an implied AND
/// unary   := NOT* primary              an odd count of NOTs negates
/// primary := TERM | PHRASE | '(' or ')'
/// ```
///
/// Malformed input is recovered the way Quill recovers it: an unclosed `(`
/// closes at the end, and an empty group, a dangling operator, a NOT with no
/// operand and an operand that lowers to nothing are skipped. This adapter
/// has no diagnostic channel, so the recovery is silent.
struct CassOracleGrammar<'a> {
    tokens: &'a [CassQueryToken],
    position: usize,
    /// Parentheses entered and not yet closed.
    open_groups: usize,
    fields: &'a CassFields,
    regex_query_factory: CassRegexQueryFactory,
}

impl CassOracleGrammar<'_> {
    fn peek(&self) -> Option<&CassQueryToken> {
        self.tokens.get(self.position)
    }

    fn parse_or(&mut self) -> SearchResult<Option<CassOracleNode>> {
        let mut operands = Vec::new();
        loop {
            match self.peek() {
                None => break,
                Some(CassQueryToken::RParen) if self.open_groups > 0 => break,
                // The lexer emits `)` only inside a group; any other is skipped.
                Some(CassQueryToken::Or | CassQueryToken::RParen) => {
                    self.position += 1;
                    continue;
                }
                Some(_) => {}
            }
            if let Some(node) = self.parse_and()? {
                operands.push(node);
            }
        }
        Ok(cass_combine_or(operands))
    }

    fn parse_and(&mut self) -> SearchResult<Option<CassOracleNode>> {
        let mut operands = Vec::new();
        loop {
            match self.peek() {
                None | Some(CassQueryToken::Or | CassQueryToken::RParen) => break,
                Some(CassQueryToken::And) => {
                    self.position += 1;
                    continue;
                }
                Some(_) => {}
            }
            if let Some(node) = self.parse_unary()? {
                operands.push(node);
            }
        }
        Ok(cass_combine_and(operands))
    }

    fn parse_unary(&mut self) -> SearchResult<Option<CassOracleNode>> {
        let mut negated = false;
        while matches!(self.peek(), Some(CassQueryToken::Not)) {
            negated = !negated;
            self.position += 1;
        }
        if !matches!(
            self.peek(),
            Some(CassQueryToken::Term(_) | CassQueryToken::Phrase(_) | CassQueryToken::LParen)
        ) {
            return Ok(None);
        }
        Ok(self.parse_primary()?.map(|mut node| {
            node.negative ^= negated;
            node
        }))
    }

    fn parse_primary(&mut self) -> SearchResult<Option<CassOracleNode>> {
        let Some(token) = self.tokens.get(self.position) else {
            return Ok(None);
        };
        self.position += 1;
        let query = match token {
            CassQueryToken::Term(term) => cass_build_compound_term_query(
                &cass_normalize_term_parts(term),
                self.fields,
                self.regex_query_factory,
            )?,
            CassQueryToken::Phrase(phrase) => cass_build_phrase_query(
                &cass_normalize_phrase_terms(phrase),
                self.fields,
                self.regex_query_factory,
            )?,
            CassQueryToken::LParen => {
                self.open_groups += 1;
                let inner = self.parse_or()?;
                self.open_groups -= 1;
                if matches!(self.peek(), Some(CassQueryToken::RParen)) {
                    self.position += 1;
                }
                return Ok(inner);
            }
            // Operators never reach here: parse_unary admits only operands.
            CassQueryToken::And
            | CassQueryToken::Or
            | CassQueryToken::Not
            | CassQueryToken::RParen => None,
        };
        Ok(query.map(|query| CassOracleNode {
            expr: CassOracleExpr::Leaf(query),
            negative: false,
        }))
    }
}

/// Returns `true` when the string contains at least one CJK character.
#[inline]
fn contains_cjk(s: &str) -> bool {
    s.chars().any(is_cjk)
}

/// Decompose a CJK string into character bigrams (matching `CjkBigramDecompose`).
/// For a single CJK character, returns the character itself as a unigram.
fn cjk_bigrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().filter(|c| is_cjk(*c)).collect();
    if chars.len() <= 1 {
        return chars.iter().map(|c| c.to_string()).collect();
    }
    (0..chars.len() - 1)
        .map(|i| {
            let mut b = String::with_capacity(8);
            b.push(chars[i]);
            b.push(chars[i + 1]);
            b
        })
        .collect()
}

/// Build a query that requires ALL bigrams to match in at least one field.
/// This mirrors how the tokenizer indexes CJK text.
#[inline]
fn cass_term_query_fields(fields: &CassFields) -> [(Field, IndexRecordOption); 4] {
    [
        (fields.title, IndexRecordOption::WithFreqsAndPositions),
        (fields.content, IndexRecordOption::WithFreqsAndPositions),
        (fields.title_prefix, IndexRecordOption::Basic),
        (fields.content_prefix, IndexRecordOption::Basic),
    ]
}

fn cass_build_cjk_term_query(bigrams: &[String], fields: &CassFields) -> Option<Box<dyn Query>> {
    if bigrams.is_empty() {
        return None;
    }

    // For each bigram, require it to appear in at least one searchable field.
    let mut bigram_musts: Vec<(Occur, Box<dyn Query>)> = Vec::new();
    for bigram in bigrams {
        let mut field_shoulds: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for (field, index_record_option) in cass_term_query_fields(fields) {
            field_shoulds.push((
                Occur::Should,
                Box::new(TermQuery::new(
                    Term::from_field_text(field, bigram),
                    index_record_option,
                )),
            ));
        }
        bigram_musts.push((Occur::Must, Box::new(BooleanQuery::new(field_shoulds))));
    }

    match bigram_musts.len() {
        0 => None,
        1 => bigram_musts.pop().map(|(_, q)| q),
        _ => Some(Box::new(BooleanQuery::new(bigram_musts))),
    }
}

pub(super) type CassRegexQueryFactory = fn(Field, &str) -> SearchResult<RegexQuery>;

/// Build query clauses for a single term based on its wildcard pattern.
fn cass_build_term_query_clauses(
    pattern: &CassWildcardPattern,
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Vec<(Occur, Box<dyn Query>)>> {
    let mut shoulds: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    match pattern {
        CassWildcardPattern::Exact(term) | CassWildcardPattern::Prefix(term) => {
            if term.is_empty() {
                return Ok(shoulds);
            }
            // CJK terms must be decomposed into bigrams to match the indexed tokens.
            if contains_cjk(term) {
                let bigrams = cjk_bigrams(term);
                if let Some(q) = cass_build_cjk_term_query(&bigrams, fields) {
                    shoulds.push((Occur::Should, q));
                }
                return Ok(shoulds);
            }
            for (field, index_record_option) in cass_term_query_fields(fields) {
                shoulds.push((
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(field, term),
                        index_record_option,
                    )),
                ));
            }
        }
        CassWildcardPattern::Suffix(_)
        | CassWildcardPattern::Substring(_)
        | CassWildcardPattern::Complex(_) => {
            if let Some(regex_pattern) = pattern.to_regex() {
                let content_query = regex_query_factory(fields.content, &regex_pattern)?;
                let title_query = regex_query_factory(fields.title, &regex_pattern)?;
                shoulds.push((Occur::Should, Box::new(content_query)));
                shoulds.push((Occur::Should, Box::new(title_query)));
            }
        }
    }

    Ok(shoulds)
}

/// Build a compound query that requires all term parts to match (implicit AND).
fn cass_build_compound_term_query(
    parts: &[String],
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Option<Box<dyn Query>>> {
    let mut subqueries: Vec<Box<dyn Query>> = Vec::new();
    for part in parts {
        let pattern = CassWildcardPattern::parse(part);
        let term_shoulds = cass_build_term_query_clauses(&pattern, fields, regex_query_factory)?;
        if !term_shoulds.is_empty() {
            subqueries.push(Box::new(BooleanQuery::new(term_shoulds)));
        }
    }

    Ok(match subqueries.len() {
        0 => None,
        1 => subqueries.pop(),
        _ => {
            let musts = subqueries.into_iter().map(|q| (Occur::Must, q)).collect();
            Some(Box::new(BooleanQuery::new(musts)))
        }
    })
}

/// Build a phrase query (exact order) across title/content fields.
fn cass_build_phrase_query(
    terms: &[String],
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Option<Box<dyn Query>>> {
    if terms.is_empty() {
        return Ok(None);
    }
    if terms.len() == 1 {
        return cass_build_compound_term_query(terms, fields, regex_query_factory);
    }

    // If any term contains CJK, fall back to compound term query (AND of bigrams)
    // because PhraseQuery expects exact indexed tokens and CJK is bigram-indexed.
    if terms.iter().any(|t| contains_cjk(t)) {
        return cass_build_compound_term_query(terms, fields, regex_query_factory);
    }

    let mut shoulds: Vec<(Occur, Box<dyn Query>)> = Vec::new();
    for field in [fields.title, fields.content] {
        let phrase_terms = terms
            .iter()
            .map(|t| Term::from_field_text(field, t))
            .collect::<Vec<_>>();
        shoulds.push((Occur::Should, Box::new(PhraseQuery::new(phrase_terms))));
    }
    Ok(Some(Box::new(BooleanQuery::new(shoulds))))
}

/// Build Tantivy query clauses from boolean tokens.
///
/// Standard precedence, as cass documents it and Quill parses it: NOT binds
/// tightest, then AND (explicit, `&&`, or implied between adjacent operands),
/// then OR (`OR`, `||`); parentheses group; negation is parity-based.
fn cass_build_boolean_query_clauses(
    tokens: &[CassQueryToken],
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Vec<(Occur, Box<dyn Query>)>> {
    let mut grammar = CassOracleGrammar {
        tokens,
        position: 0,
        open_groups: 0,
        fields,
        regex_query_factory,
    };
    let Some(root) = grammar.parse_or()? else {
        return Ok(Vec::new());
    };
    Ok(match root {
        // A negative root denotes the complement of its operand.
        CassOracleNode {
            expr,
            negative: true,
        } => vec![
            (Occur::Must, Box::new(AllQuery)),
            (Occur::MustNot, expr.into_query()),
        ],
        // Flat, so a positive conjunction such as `auth AND NOT deprecated`
        // stays unanchored and gains no AllQuery score.
        CassOracleNode {
            expr: CassOracleExpr::And(clauses),
            negative: false,
        } => clauses,
        CassOracleNode {
            expr,
            negative: false,
        } => vec![(Occur::Must, expr.into_query())],
    })
}

fn cass_match_none_query() -> Box<dyn Query> {
    Box::new(BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery)),
        (Occur::MustNot, Box::new(AllQuery)),
    ]))
}

pub(super) fn cass_fail_closed_tantivy_query(
    raw_query: &str,
    result: SearchResult<Box<dyn Query>>,
) -> Box<dyn Query> {
    result.unwrap_or_else(|error| {
        warn!(
            query_bytes = raw_query.len(),
            error = %error,
            "CASS query construction failed; returning an explicit match-none query"
        );
        cass_match_none_query()
    })
}

/// Build a cass-compatible Tantivy query for `raw_query`, applying filters.
///
/// Returns an `AllQuery` when the query is empty.
/// Regex construction errors are logged and converted to an explicit
/// match-none query because this compatibility API predates fallible returns.
#[must_use]
pub fn cass_build_tantivy_query(
    raw_query: &str,
    filters: &CassQueryFilters,
    fields: &CassFields,
) -> Box<dyn Query> {
    let tokens = cass_parse_boolean_query(raw_query);
    let has_boolean_operators = !tokens.is_empty() && cass_has_boolean_operators(raw_query);
    cass_fail_closed_tantivy_query(
        raw_query,
        cass_try_build_tantivy_query_from_tokens(
            tokens,
            has_boolean_operators,
            filters,
            fields,
            cass_regex_query_cached,
        ),
    )
}

/// Single-parse candidate retained for same-binary performance and query-tree
/// parity checks. Production remains on [`cass_build_tantivy_query`] because
/// the candidate did not clear the paired A/A null floor.
#[cfg(any(test, feature = "bench-internals"))]
#[doc(hidden)]
#[must_use]
pub fn cass_build_tantivy_query_single_parse(
    raw_query: &str,
    filters: &CassQueryFilters,
    fields: &CassFields,
) -> Box<dyn Query> {
    let tokens = cass_parse_boolean_query(raw_query);
    let has_boolean_operators = cass_tokens_have_boolean_operators(&tokens);
    cass_fail_closed_tantivy_query(
        raw_query,
        cass_try_build_tantivy_query_from_tokens(
            tokens,
            has_boolean_operators,
            filters,
            fields,
            cass_regex_query_cached,
        ),
    )
}

pub(super) fn cass_try_build_tantivy_query_from_tokens(
    tokens: Vec<CassQueryToken>,
    has_boolean_operators: bool,
    filters: &CassQueryFilters,
    fields: &CassFields,
    regex_query_factory: CassRegexQueryFactory,
) -> SearchResult<Box<dyn Query>> {
    let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    if tokens.is_empty() {
        clauses.push((Occur::Must, Box::new(AllQuery)));
    } else if has_boolean_operators {
        clauses.extend(cass_build_boolean_query_clauses(
            &tokens,
            fields,
            regex_query_factory,
        )?);
    } else {
        for token in tokens {
            if let CassQueryToken::Term(term_str) = token {
                let parts = cass_normalize_term_parts(&term_str);
                if let Some(term_query) =
                    cass_build_compound_term_query(&parts, fields, regex_query_factory)?
                {
                    clauses.push((Occur::Must, term_query));
                }
            }
        }
    }

    if !filters.agents.is_empty() {
        let terms = filters
            .agents
            .iter()
            .map(|agent| {
                (
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(fields.agent, agent),
                        IndexRecordOption::Basic,
                    )) as Box<dyn Query>,
                )
            })
            .collect();
        clauses.push((Occur::Must, Box::new(BooleanQuery::new(terms))));
    }

    if !filters.workspaces.is_empty() {
        let terms = filters
            .workspaces
            .iter()
            .map(|ws| {
                (
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(fields.workspace, ws),
                        IndexRecordOption::Basic,
                    )) as Box<dyn Query>,
                )
            })
            .collect();
        clauses.push((Occur::Must, Box::new(BooleanQuery::new(terms))));
    }

    if filters.created_from.is_some() || filters.created_to.is_some() {
        use std::ops::Bound::{Included, Unbounded};
        let lower = filters.created_from.map_or(Unbounded, |v| {
            Included(Term::from_field_i64(fields.created_at, v))
        });
        let upper = filters.created_to.map_or(Unbounded, |v| {
            Included(Term::from_field_i64(fields.created_at, v))
        });
        let range = RangeQuery::new(lower, upper);
        clauses.push((Occur::Must, Box::new(range)));
    }

    match &filters.source_filter {
        CassSourceFilter::All => {}
        CassSourceFilter::Local => {
            let term = Term::from_field_text(fields.origin_kind, "local");
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
            ));
        }
        CassSourceFilter::Remote => {
            let term = Term::from_field_text(fields.origin_kind, "ssh");
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
            ));
        }
        CassSourceFilter::SourceId(source_id) => {
            let term = Term::from_field_text(fields.source_id, source_id);
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
            ));
        }
    }

    Ok(match clauses.len() {
        0 => Box::new(AllQuery),
        1 => {
            if let Some((occur, query_box)) = clauses.pop() {
                match occur {
                    Occur::Must => query_box,
                    _ => Box::new(BooleanQuery::new(vec![(occur, query_box)])),
                }
            } else {
                Box::new(AllQuery)
            }
        }
        _ => Box::new(BooleanQuery::new(clauses)),
    })
}
