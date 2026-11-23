//! Engine-neutral query trees and native default/CASS lenient parsers.
//!
//! Parsing is deliberately separated from scorer construction. The tree uses
//! stable Quill field identifiers, never Tantivy field handles, and malformed
//! user input is recovered or dropped with diagnostics rather than promoted to
//! a search error.

mod ast;
mod canonical;
mod cass;
mod grammar;
mod lexer;
mod rewrite;
mod syntax;

#[cfg(test)]
mod tests;

/// Maximum number of Unicode scalar values admitted to either native parser.
pub const MAX_QUERY_LENGTH: usize = 10_000;

/// Historical parser nesting threshold retained as a public API constant.
///
/// The arena parser no longer rejects queries at this depth. The admitted
/// query-length bound limits total work, while explicit parser and lowering
/// stacks keep nesting independent of the native call stack.
pub const MAX_QUERY_DEPTH: usize = 64;

const CONTENT_FIELD_NAME: &str = "content";
const TITLE_FIELD_NAME: &str = "title";
const TITLE_BOOST: f32 = 2.0;

pub use ast::{
    BooleanClause, BooleanOperator, Occur, PositionedTerm, Query, QueryClause, QueryField,
    QueryNode, QueryNodeId, QueryValue,
};
#[cfg(any(test, feature = "conformance-internals"))]
pub use canonical::{
    CassParsedQueryObservation, CassQueryObservationError, CassQueryToken,
    MAX_CASS_OBSERVATION_FILTER_BYTES, MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES,
    MAX_CASS_OBSERVATION_FILTER_VALUES, MAX_CASS_OBSERVATION_RAW_QUERY_BYTES,
};
pub use canonical::{
    IndexCapability, ParsedQuery, QueryCanonicalizationReport, QueryCapabilityError,
    QueryDiagnostic, QueryDiagnosticKind, QueryExplanation, QueryParserConfigError,
    canonicalize_query, classify_query, escape_unknown_field_colons, validate_index_capabilities,
};
pub use cass::{
    CassQueryFilters, CassQueryParser, CassQueryParserConfigError, CassSourceFilter,
    CassWildcardPattern, cass_has_boolean_operators, cass_sanitize_query,
};
pub use syntax::{DefaultQueryParser, truncate_query};

#[cfg(test)]
use canonical::{canonical_child_key, canonical_sort_key, canonical_sort_key_for_node};
#[cfg(all(test, feature = "tantivy-oracle"))]
use cass::{CassLexToken, cass_lex};
#[cfg(test)]
use grammar::Grammar;
#[cfg(test)]
use lexer::{lex, peel_redundant_outer_groups};
#[cfg(test)]
use rewrite::{
    flatten_should_of_should, repair_root_all_negative, rewrite_parser_syntax, trim_parser_dropped,
};
