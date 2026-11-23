use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "tantivy-oracle")]
use frankensearch_core::error::SearchError;
use frankensearch_core::error::SearchResult;
use tantivy::query::RegexQuery;
use tantivy::schema::Field;

use super::tantivy_err;

const CASS_REGEX_QUERY_CACHE_CAP: usize = 128;
static CASS_REGEX_QUERY_CACHE: OnceLock<RwLock<HashMap<Field, HashMap<String, RegexQuery>>>> =
    OnceLock::new();

/// Build a Tantivy [`RegexQuery`] using a small global cache (cass compatibility).
///
/// Motivation: cass can generate many regex queries when handling `*foo` / `*foo*`
/// patterns and boolean combinations. Compiling Tantivy regex queries repeatedly
/// is expensive; caching improves interactive search latency significantly.
///
/// Cache behavior:
/// - Bounded in-memory cache (clears on overflow).
/// - Keyed by `(field, pattern)`.
/// - Thread-safe via `RwLock`.
///
/// # Errors
///
/// Returns [`frankensearch_core::error::SearchError::SubsystemError`] when Tantivy fails to compile the regex.
pub fn cass_regex_query_cached(field: Field, pattern: &str) -> SearchResult<RegexQuery> {
    let cache = CASS_REGEX_QUERY_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    {
        let guard = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(field_cache) = guard.get(&field)
            && let Some(q) = field_cache.get(pattern)
        {
            return Ok(q.clone());
        }
    }

    let query = RegexQuery::from_pattern(pattern, field).map_err(tantivy_err)?;
    let mut guard = cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let total_entries: usize = guard.values().map(HashMap::len).sum();
    if total_entries >= CASS_REGEX_QUERY_CACHE_CAP {
        guard.clear();
    }
    guard
        .entry(field)
        .or_default()
        .insert(pattern.to_string(), query.clone());
    drop(guard);
    Ok(query)
}

/// Build a Tantivy [`RegexQuery`] without caching (baseline for benchmarks/tests).
///
/// # Errors
///
/// Returns [`frankensearch_core::error::SearchError::SubsystemError`] when Tantivy fails to compile the regex.
pub fn cass_regex_query_uncached(field: Field, pattern: &str) -> SearchResult<RegexQuery> {
    RegexQuery::from_pattern(pattern, field).map_err(tantivy_err)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cass Lexical Query Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Source filter options used by cass-compatible lexical queries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CassSourceFilter {
    /// No source filtering.
    #[default]
    All,
    /// Local-only (`origin_kind` == "local").
    Local,
    /// Remote-only (`origin_kind` == "ssh").
    Remote,
    /// Specific source id (`source_id` == <id>).
    SourceId(String),
}

/// Cass-compatible lexical filters applied directly in Tantivy.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CassQueryFilters {
    pub agents: Vec<String>,
    pub workspaces: Vec<String>,
    pub created_from: Option<i64>,
    pub created_to: Option<i64>,
    pub source_filter: CassSourceFilter,
}

/// Token types for cass-style boolean query parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CassQueryToken {
    /// A search term (may include wildcards).
    Term(String),
    /// A quoted phrase for exact-order matching.
    Phrase(String),
    /// Explicit AND operator.
    And,
    /// OR operator.
    Or,
    /// NOT operator (negates the next term, phrase, or group).
    Not,
    /// A group-opening parenthesis.
    LParen,
    /// A group-closing parenthesis.
    RParen,
}

/// In-memory evidence from the real fallible Tantivy CASS query boundary.
///
/// This type deliberately retains plaintext only between the adapter and the
/// gauntlet's redaction step. It must never be serialized directly. The
/// profile-specific `ArtifactStore` v4 DTO owns any durable representation.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug)]
pub struct CassOracleProfileObservation {
    /// Shipping CASS sanitization of the raw query.
    pub sanitized_query: String,
    /// Ordered tokens passed to the actual Tantivy query builder.
    pub tokens: Vec<CassQueryToken>,
    /// Whether the builder selected its explicit Boolean grammar.
    pub has_boolean_operators: bool,
    /// Structured filters consumed by the real query builder.
    pub filters: CassQueryFilters,
    /// Native rank/tie/count result or the original typed failure.
    pub outcome: CassOracleProfileOutcome,
}

/// Total result of one fallible Tantivy CASS profile attempt.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug)]
pub enum CassOracleProfileOutcome {
    /// Query construction and retrieval succeeded.
    Success(crate::OracleQueryObservation),
    /// Query construction or retrieval failed with its original typed error.
    Error(SearchError),
}

/// Maximum raw-query bytes admitted by the dev-only CASS oracle boundary.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_RAW_QUERY_BYTES: usize = 1_048_576;

/// Maximum token count retained by one dev-only CASS oracle observation.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_TOKENS: usize = 20_000;

/// Maximum number of exact filter values admitted by one oracle request.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_FILTER_VALUES: usize = 4_096;

/// Maximum bytes admitted for any individual exact filter value.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_FILTER_VALUE_BYTES: usize = 4_096;

/// Maximum aggregate bytes admitted across all exact filter values.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_FILTER_BYTES: usize = 1_048_576;

/// Maximum requested or expanded fetch cardinality for one oracle call.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_FETCH_HITS: usize = 100_000;

/// Maximum UTF-8 bytes retained for one canonical CASS document identity.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_DOCUMENT_ID_BYTES: usize = 1_024;

/// Maximum aggregate identity bytes retained across fetched and cutoff rows.
#[cfg(feature = "tantivy-oracle")]
pub const MAX_CASS_ORACLE_DOCUMENT_ID_AGGREGATE_BYTES: usize = 16 * 1_024 * 1_024;

#[cfg(feature = "tantivy-oracle")]
pub(super) fn cass_oracle_preflight(
    raw_query: &str,
    filters: &CassQueryFilters,
    limit: usize,
    tie_expansion_limit: usize,
) -> SearchResult<()> {
    cass_oracle_bound(
        "raw_query_bytes",
        raw_query.len(),
        MAX_CASS_ORACLE_RAW_QUERY_BYTES,
    )?;
    cass_oracle_bound("limit", limit, MAX_CASS_ORACLE_FETCH_HITS)?;
    cass_oracle_bound(
        "tie_expansion_limit",
        tie_expansion_limit,
        MAX_CASS_ORACLE_FETCH_HITS,
    )?;
    let expanded_fetch = limit.saturating_add(tie_expansion_limit);
    cass_oracle_bound(
        "expanded_fetch_hits",
        expanded_fetch,
        MAX_CASS_ORACLE_FETCH_HITS,
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
    cass_oracle_bound(
        "filter_value_count",
        filter_count,
        MAX_CASS_ORACLE_FILTER_VALUES,
    )?;

    let mut aggregate_bytes = 0_usize;
    for value in filters
        .agents
        .iter()
        .chain(&filters.workspaces)
        .map(String::as_str)
        .chain(source_value)
    {
        cass_oracle_bound(
            "filter_value_bytes",
            value.len(),
            MAX_CASS_ORACLE_FILTER_VALUE_BYTES,
        )?;
        aggregate_bytes = aggregate_bytes.saturating_add(value.len());
        cass_oracle_bound(
            "aggregate_filter_bytes",
            aggregate_bytes,
            MAX_CASS_ORACLE_FILTER_BYTES,
        )?;
    }
    Ok(())
}

#[cfg(feature = "tantivy-oracle")]
pub(super) fn cass_oracle_bound(
    field: &'static str,
    actual: usize,
    limit: usize,
) -> SearchResult<()> {
    if actual > limit {
        return Err(cass_oracle_invalid_bound(field, actual, limit));
    }
    Ok(())
}

#[cfg(feature = "tantivy-oracle")]
pub(super) fn cass_oracle_invalid_bound(
    field: &'static str,
    actual: usize,
    limit: usize,
) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("cass_oracle.{field}"),
        value: actual.to_string(),
        reason: format!("must be no greater than {limit}"),
    }
}

/// Sanitize query string to match the `hyphen_normalize` tokenizer for cass indexes.
///
/// The tokenizer preserves hyphens inside words (e.g. `bd-q3fy`, `POL-358`).
/// We therefore keep hyphens alongside `*` (wildcards) and `"` (phrases),
/// replacing all other non-alphanumeric characters with spaces so that query
/// terms align with indexed tokens.
#[must_use]
pub fn cass_sanitize_query(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '*' || c == '"' || c == '-' {
                c
            } else {
                ' '
            }
        })
        .collect()
}

#[must_use]
fn cass_escape_regex(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        match c {
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' => {
                escaped.push('\\');
                escaped.push(c);
            }
            _ => escaped.push(c),
        }
    }
    escaped
}

/// Represents different wildcard patterns for a cass lexical search term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CassWildcardPattern {
    Exact(String),
    Prefix(String),
    Suffix(String),
    Substring(String),
    Complex(String),
}

impl CassWildcardPattern {
    #[must_use]
    pub fn parse(term: &str) -> Self {
        let starts_with_star = term.starts_with('*');
        let ends_with_star = term.ends_with('*');

        let core = term.trim_matches('*').to_lowercase();
        if core.is_empty() {
            return Self::Exact(String::new());
        }

        // Internal wildcards (e.g. f*o) -> complex pattern.
        if core.contains('*') {
            return Self::Complex(term.to_lowercase());
        }

        match (starts_with_star, ends_with_star) {
            (true, true) => Self::Substring(core),
            (true, false) => Self::Suffix(core),
            (false, true) => Self::Prefix(core),
            (false, false) => Self::Exact(core),
        }
    }

    #[must_use]
    pub fn to_regex(&self) -> Option<String> {
        match self {
            // tantivy-fst regexes already match the complete term and reject
            // zero-width assertions, so explicit `^` / `$` anchors are both
            // unnecessary and invalid here.
            Self::Suffix(core) => Some(format!(".*{}", cass_escape_regex(core))),
            Self::Substring(core) => Some(format!(".*{}.*", cass_escape_regex(core))),
            Self::Complex(full_term) => {
                let mut regex = String::with_capacity(full_term.len() * 2 + 2);

                if full_term.starts_with('*') {
                    regex.push_str(".*");
                }

                let trimmed_start = full_term.trim_start_matches('*');
                let trimmed = trimmed_start.trim_end_matches('*');
                for c in trimmed.chars() {
                    if c == '*' {
                        regex.push_str(".*");
                    } else {
                        match c {
                            '\\' | '.' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|'
                            | '^' | '$' => {
                                regex.push('\\');
                                regex.push(c);
                            }
                            _ => regex.push(c),
                        }
                    }
                }

                if full_term.ends_with('*') {
                    regex.push_str(".*");
                }

                Some(regex)
            }
            _ => None,
        }
    }
}

/// Parse a query string into boolean tokens.
///
/// Supports:
/// - AND / && (explicit AND; implicit AND between terms is handled by query construction)
/// - OR / || (OR)
/// - NOT / -prefix (negation)
/// - \"quoted phrases\" (phrase match)
/// - ( ) grouping: a `(` opens a group only at the start of a word and a `)`
///   closes one only while a group is open, so `foo(bar)` stays one term
#[must_use]
pub fn cass_parse_boolean_query(query: &str) -> Vec<CassQueryToken> {
    match cass_parse_boolean_query_with_sink(query, |tokens, token| {
        tokens.push(token);
        Ok::<(), std::convert::Infallible>(())
    }) {
        Ok(tokens) => tokens,
        Err(never) => match never {},
    }
}

#[cfg(feature = "tantivy-oracle")]
pub(super) fn cass_parse_boolean_query_bounded(
    query: &str,
    max_tokens: usize,
) -> Result<Vec<CassQueryToken>, usize> {
    cass_parse_boolean_query_with_sink(query, |tokens, token| {
        cass_push_query_token(tokens, token, max_tokens)
    })
}

fn cass_parse_boolean_query_with_sink<Error, PushToken>(
    query: &str,
    mut push_token: PushToken,
) -> Result<Vec<CassQueryToken>, Error>
where
    PushToken: FnMut(&mut Vec<CassQueryToken>, CassQueryToken) -> Result<(), Error>,
{
    let mut tokens = Vec::new();
    let mut chars = query.chars().peekable();
    let mut current_word = String::new();
    // Groups currently open. A `(` inside a word and a `)` with no group
    // open stay word characters, so code terms like `foo(bar)` survive.
    let mut open_groups = 0_usize;

    while let Some(c) = chars.next() {
        match c {
            '(' if current_word.is_empty() => {
                push_token(&mut tokens, CassQueryToken::LParen)?;
                open_groups += 1;
            }
            ')' if open_groups > 0 => {
                if !current_word.is_empty() {
                    let token = cass_classify_query_word(std::mem::take(&mut current_word));
                    push_token(&mut tokens, token)?;
                }
                push_token(&mut tokens, CassQueryToken::RParen)?;
                open_groups -= 1;
            }
            '"' => {
                if !current_word.is_empty() {
                    push_token(
                        &mut tokens,
                        CassQueryToken::Term(std::mem::take(&mut current_word)),
                    )?;
                }
                let mut phrase = String::new();
                while let Some(&next) = chars.peek() {
                    if next == '"' {
                        chars.next();
                        break;
                    }
                    if let Some(c) = chars.next() {
                        phrase.push(c);
                    }
                }
                if !phrase.is_empty() {
                    push_token(&mut tokens, CassQueryToken::Phrase(phrase))?;
                }
            }
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                if !current_word.is_empty() {
                    push_token(
                        &mut tokens,
                        CassQueryToken::Term(std::mem::take(&mut current_word)),
                    )?;
                }
                push_token(&mut tokens, CassQueryToken::And)?;
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                if !current_word.is_empty() {
                    push_token(
                        &mut tokens,
                        CassQueryToken::Term(std::mem::take(&mut current_word)),
                    )?;
                }
                push_token(&mut tokens, CassQueryToken::Or)?;
            }
            '-' if current_word.is_empty() => {
                push_token(&mut tokens, CassQueryToken::Not)?;
            }
            ' ' | '\t' | '\n' => {
                if !current_word.is_empty() {
                    let word = std::mem::take(&mut current_word);
                    let token = cass_classify_query_word(word);
                    push_token(&mut tokens, token)?;
                }
            }
            _ => current_word.push(c),
        }
    }

    if !current_word.is_empty() {
        let token = cass_classify_query_word(current_word);
        push_token(&mut tokens, token)?;
    }

    Ok(tokens)
}

fn cass_classify_query_word(word: String) -> CassQueryToken {
    if word.eq_ignore_ascii_case("AND") {
        CassQueryToken::And
    } else if word.eq_ignore_ascii_case("OR") {
        CassQueryToken::Or
    } else if word.eq_ignore_ascii_case("NOT") {
        CassQueryToken::Not
    } else {
        CassQueryToken::Term(word)
    }
}

#[cfg(feature = "tantivy-oracle")]
fn cass_push_query_token(
    tokens: &mut Vec<CassQueryToken>,
    token: CassQueryToken,
    max_tokens: usize,
) -> Result<(), usize> {
    if tokens.len() >= max_tokens {
        return Err(tokens.len().saturating_add(1));
    }
    tokens.push(token);
    Ok(())
}

#[must_use]
pub fn cass_has_boolean_operators(query: &str) -> bool {
    let tokens = cass_parse_boolean_query(query);
    cass_tokens_have_boolean_operators(&tokens)
}

pub(super) fn cass_tokens_have_boolean_operators(tokens: &[CassQueryToken]) -> bool {
    tokens.iter().any(|t| {
        matches!(
            t,
            CassQueryToken::And
                | CassQueryToken::Or
                | CassQueryToken::Not
                | CassQueryToken::Phrase(_)
        )
    })
}
