//! Tantivy BM25 full-text search integration for frankensearch.
//!
//! Provides the [`TantivyIndex`] implementation of the [`LexicalRead`](frankensearch_core::LexicalRead)
//! and [`LexicalWrite`](frankensearch_core::LexicalWrite) traits,
//! including schema creation, document indexing, BM25 query parsing,
//! and search result ranking.
//!
//! The crate version constant is part of gauntlet dependency provenance: it
//! identifies the concrete lexical wrapper compiled around Tantivy.
//!
//! # Schema
//!
//! | Field | Tantivy Options | Source |
//! |-------|-----------------|--------|
//! | `id` | `STRING \| STORED` | `IndexableDocument::id` |
//! | `content` | `TEXT \| STORED` | `IndexableDocument::content` |
//! | `title` | `TEXT \| STORED` | `IndexableDocument::title` (empty if `None`) |
//! | `metadata_json` | `STORED` | Serialized `IndexableDocument::metadata` |
//!
//! The `content` and `title` fields are searched with BM25 scoring.
//! Title matches receive a 2× boost via `QueryParser::set_field_boost`.

/// Exact `frankensearch-lexical` crate version compiled into this adapter.
pub const FRANKENSEARCH_LEXICAL_CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod cass_compat;
pub mod quill_contract;

#[cfg(feature = "bench-internals")]
mod benchmark;
mod execution;
mod index;
#[cfg(feature = "bench-internals")]
mod index_bench;
#[cfg(feature = "tantivy-oracle")]
mod index_oracle;
mod index_search;
mod query_repair;
mod schema;
mod tokenizer;
mod types;

pub use cass_compat::{
    CASS_SCHEMA_HASH, CASS_SCHEMA_VERSION, CassDocument, CassDocumentRef, CassFields,
    CassMergeStatus, CassQueryFilters, CassQueryToken, CassSourceFilter, CassTantivyIndex,
    CassWildcardPattern, cass_build_preview, cass_build_schema, cass_build_tantivy_query,
    cass_ensure_tokenizer, cass_fields_from_schema, cass_generate_edge_ngrams,
    cass_has_boolean_operators, cass_index_dir, cass_open_search_reader, cass_parse_boolean_query,
    cass_regex_query_cached, cass_regex_query_uncached, cass_sanitize_query,
    cass_schema_hash_matches,
};

// Re-export tantivy types that appear in frankensearch-lexical's public API.
// Consumers can import these from `frankensearch::lexical_tantivy::` instead
// of adding a direct Tantivy dependency. The explicit namespace keeps
// foreign-format and oracle callers stable when the facade's generic
// `lexical` feature selects Quill.
pub use tantivy::collector::{Count, TopDocs};
pub use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
pub use tantivy::schema::{Field, IndexRecordOption, Schema, Value};
pub use tantivy::{
    self as tantivy_crate, DocAddress, Index, IndexReader, IndexWriter, ReloadPolicy, Searcher,
    TantivyDocument, Term,
};

#[cfg(feature = "bench-internals")]
pub use benchmark::{
    BenchmarkMaterializedWidth, BenchmarkRetainedTantivyReader, BenchmarkTantivyDirectorySyncArm,
    BenchmarkTantivyDirectorySyncReceipt, BenchmarkWidthUnobservableReason,
    BenchmarkWriterAttestation, BenchmarkWriterJoinReceipt, BenchmarkWriterMode,
    BenchmarkWriterReceipt,
};
pub use execution::{
    execute_query_with_offset, execute_top_k, load_doc, render_snippet_html,
    try_build_snippet_generator,
};
pub use index::TantivyIndex;
#[cfg(feature = "bench-internals")]
pub use tokenizer::default_tokenizer_for_bench;
pub use types::{
    LexicalDocHit, LexicalHit, LexicalIdHit, LexicalSearchResult, QueryExplanation, SnippetConfig,
};
#[cfg(feature = "tantivy-oracle")]
pub use types::{
    OraclePageHit, OraclePageObservation, OracleQueryObservation, OracleRankedHit,
    OracleSegmentLayout,
};

#[cfg(test)]
pub(crate) use schema::build_schema;
#[cfg(test)]
pub(crate) use tokenizer::build_tokenizer;
#[cfg(feature = "tantivy-oracle")]
pub(crate) use types::oracle_searcher_sha256;

#[cfg(test)]
mod tests;
