#[cfg(feature = "tantivy-oracle")]
pub(super) use frankensearch_core::error::SearchError;
pub(super) use frankensearch_core::error::SearchResult;
pub(super) use tantivy::{
    indexer::NoMergePolicy,
    query::RegexQuery,
    schema::{Field, IndexRecordOption, Value as _},
    tokenizer::{TextAnalyzer, TokenStream, Tokenizer},
};

pub(super) use super::index::{
    CassAddPlan, CassWriterConfig, cass_parallel_add_plan, cass_writer_config_for_parallelism,
};
pub(super) use super::preview::{cass_build_content_prefix_and_preview, cass_prefix_source};
pub(super) use super::query::cass_tokens_have_boolean_operators;
pub(super) use super::query_builder::{
    cass_fail_closed_tantivy_query, cass_try_build_tantivy_query_from_tokens,
};
pub(super) use super::tokenizer::{CassTokenizer, is_cjk, next_char_from};
pub(super) use super::*;
pub(super) use support::*;

mod query_and_oracle;
mod support;
mod tokenizer_and_merge;
