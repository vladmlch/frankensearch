pub use std::{borrow::Cow, path::Path};

pub use asupersync::Cx;
pub use frankensearch_core::error::SearchError;
// The split capabilities, so `idx.search(..)` / `idx.doc_count()` resolve
// now that the combined trait is gone. B1 deliberately kept these out of
// module scope to avoid method ambiguity while both surfaces coexisted.
pub use frankensearch_core::traits::{LexicalRead, LexicalWrite};
pub use frankensearch_core::types::{DocId, IndexableDocument, ScoreSource, ScoredResult};

#[cfg(feature = "bench-internals")]
pub use crate::benchmark::positions_from_live_schema;
pub use crate::execution::{collector_invocations, reset_collector_invocations};
pub use crate::index::{MAX_QUERY_LENGTH, WriterCall};
pub use crate::query_repair::repair_negated_conjunction;
#[cfg(feature = "bench-internals")]
pub use crate::schema::TOKENIZER_NAME;
pub use crate::schema::build_schema;
pub use crate::types::{DEFAULT_SNIPPET_MAX_CHARS, classify_query};
#[cfg(feature = "bench-internals")]
pub use crate::{
    BenchmarkMaterializedWidth, BenchmarkWidthUnobservableReason, BenchmarkWriterMode,
};
pub use crate::{
    Count, IndexRecordOption, LexicalDocHit, LexicalHit, LexicalIdHit, QueryExplanation,
    SnippetConfig, TantivyIndex, Term, TermQuery, execute_query_with_offset, execute_top_k,
};

pub use support::{assert_live_lexical_contract, run_with_cx, sample_docs, shared_full120_docs};

#[cfg(feature = "bench-internals")]
mod benchmark_writer_mode;
mod index_lifecycle;
mod query_repair;
mod search_snippets;
mod support;
