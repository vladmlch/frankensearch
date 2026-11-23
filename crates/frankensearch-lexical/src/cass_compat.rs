//! External CASS Tantivy-format interoperability.
//!
//! NOTE: this module is intentionally not a fallback for native Quill. It
//! reads and writes the external CASS tool's schema-v8 Tantivy indexes at
//! `<base>/index/v8/`, so removing it before CASS migrates to FSLX (or drops
//! the integration) would break a foreign-format boundary. The facade exposes
//! it only through `cass-compat`; default builds do not include this crate.

mod index;
mod preview;
mod query;
mod query_builder;
mod schema;
mod tokenizer;

#[cfg(test)]
mod tests;

use frankensearch_core::error::SearchError;

fn tantivy_err<E>(err: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "tantivy",
        source: Box::new(err),
    }
}

pub use index::{CassMergeStatus, CassTantivyIndex};
pub use preview::{
    cass_build_preview, cass_build_preview_slow, cass_generate_edge_ngrams,
    cass_generate_edge_ngrams_slow, cass_prefix_source_fast_bench, cass_prefix_source_slow,
};
#[cfg(feature = "tantivy-oracle")]
pub use query::{
    CassOracleProfileObservation, CassOracleProfileOutcome,
    MAX_CASS_ORACLE_DOCUMENT_ID_AGGREGATE_BYTES, MAX_CASS_ORACLE_DOCUMENT_ID_BYTES,
    MAX_CASS_ORACLE_FETCH_HITS, MAX_CASS_ORACLE_FILTER_BYTES, MAX_CASS_ORACLE_FILTER_VALUE_BYTES,
    MAX_CASS_ORACLE_FILTER_VALUES, MAX_CASS_ORACLE_RAW_QUERY_BYTES, MAX_CASS_ORACLE_TOKENS,
};
pub use query::{
    CassQueryFilters, CassQueryToken, CassSourceFilter, CassWildcardPattern,
    cass_has_boolean_operators, cass_parse_boolean_query, cass_regex_query_cached,
    cass_regex_query_uncached, cass_sanitize_query,
};
pub use query_builder::cass_build_tantivy_query;
#[cfg(any(test, feature = "bench-internals"))]
pub use query_builder::cass_build_tantivy_query_single_parse;
pub use schema::{
    CASS_SCHEMA_HASH, CASS_SCHEMA_VERSION, CassDocument, CassDocumentRef, CassFields,
    cass_build_schema, cass_fields_from_schema, cass_index_dir, cass_open_search_reader,
    cass_schema_hash_matches,
};
#[cfg(feature = "tantivy-oracle")]
pub use schema::{cass_document_identity, cass_document_identity_parts};
pub use tokenizer::{
    CassNormalizeAndLimit, CassNormalizeAndLimitFilter, CassNormalizeAndLimitStream,
    CjkBigramDecompose, CjkBigramDecomposeFilter, CjkBigramDecomposeStream, HyphenDecompose,
    HyphenDecomposeFilter, HyphenDecomposeStream, cass_char_walk_fast, cass_char_walk_slow,
    cass_cjk_collect_fast, cass_cjk_collect_slow, cass_ensure_tokenizer, next_char_from_slow,
};
#[cfg(feature = "bench-internals")]
pub use tokenizer::{cass_cjk_bigrams_direct, cass_cjk_bigrams_staged};
