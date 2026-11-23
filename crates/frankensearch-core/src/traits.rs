//! Core traits for the frankensearch search pipeline.
//!
//! - [`Embedder`]: Text embedding model interface (hash, model2vec, fastembed).
//! - [`Reranker`]: Cross-encoder reranking model interface.
//! - [`LexicalRead`] / [`LexicalWrite`]: Split full-text backend interface
//!   (Tantivy, FTS5, Quill).
//!
//! Async operations are represented as boxed futures so the traits remain
//! dyn-compatible for runtime polymorphism (`Box<dyn Embedder>`, etc.).

use std::future::Future;
use std::pin::Pin;

use asupersync::Cx;

use crate::error::{SearchError, SearchResult};

mod embed;
mod lexical;
mod metrics;
mod rerank;

pub use embed::{
    Embedder, IdentityBoundEmbedding, ModelCategory, ModelInfo, ModelTier, SyncEmbed,
    SyncEmbedderAdapter, cosine_similarity, l2_normalize, l2_normalize_in_place,
    truncate_embedding,
};
pub use lexical::{LexicalCandidateBatch, LexicalHydrationContext, LexicalRead, LexicalWrite};
pub use metrics::{MetricsExporter, NoOpMetricsExporter, SharedMetricsExporter};
pub use rerank::{RerankDocument, RerankScore, Reranker, SyncRerank, SyncRerankerAdapter};

/// Boxed future carrying a `SearchResult<T>`.
pub type SearchFuture<'a, T> = Pin<Box<dyn Future<Output = SearchResult<T>> + Send + 'a>>;

pub(super) fn embedding_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

#[cfg(test)]
mod tests;
