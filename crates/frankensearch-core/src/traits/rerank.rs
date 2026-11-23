use asupersync::Cx;
use serde::{Deserialize, Serialize};

use super::{SearchFuture, embedding_checkpoint};
use crate::error::SearchResult;

// ─── Reranker Trait ─────────────────────────────────────────────────────────

/// A document for reranking: pairs a document ID with its text content.
///
/// Text must be provided because cross-encoders process query+document
/// pairs through a transformer. `ScoredResult` intentionally does not
/// carry text to avoid memory waste in the common case.
#[derive(Debug, Clone)]
pub struct RerankDocument {
    /// Document identifier.
    pub doc_id: String,
    /// Document text content for cross-encoder input.
    pub text: String,
}

/// A reranking score for a single document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankScore {
    /// Document identifier.
    pub doc_id: String,
    /// Cross-encoder relevance score (typically sigmoid-activated logit).
    pub score: f32,
    /// Position before reranking (for rank-change tracking).
    pub original_rank: usize,
    /// Raw pre-sigmoid logit, when the backend exposes it.
    ///
    /// Some cross-encoder implementations only emit a final score (after sigmoid
    /// activation). When the raw logit is unavailable, this field is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_logit: Option<f32>,
}

/// Core trait for cross-encoder reranking models.
///
/// Cross-encoders process query+document pairs together through a transformer,
/// producing more accurate relevance scores than bi-encoder cosine similarity.
/// This accuracy comes at the cost of not being able to pre-compute anything:
/// every query-document pair requires a full inference pass.
///
/// # Graceful Failure
///
/// The reranking step should never block search results. If the model is
/// unavailable or inference fails, implementations should return
/// `Err(SearchError::RerankFailed { .. })` and callers should fall back
/// to the original RRF scores.
pub trait Reranker: Send + Sync {
    /// Score and re-rank documents against a query.
    ///
    /// Returns documents sorted by descending cross-encoder score.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::RerankFailed` if cross-encoder inference fails.
    fn rerank<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>>;

    /// A unique identifier for this reranker model.
    fn id(&self) -> &str;

    /// Human-friendly reranker model name.
    fn model_name(&self) -> &str;

    /// Maximum supported token length for query+document pair input.
    fn max_length(&self) -> usize {
        512
    }

    /// Whether this reranker is loaded and ready for inference.
    fn is_available(&self) -> bool {
        true
    }
}

// ─── Synchronous Reranker Bridge ────────────────────────────────────────────

/// Synchronous reranking interface for host projects that call rerankers from
/// non-async contexts.
///
/// Implement this trait for rerankers whose `rerank` operations are inherently
/// synchronous (e.g., blocking ONNX inference). The companion
/// [`SyncRerankerAdapter`] wraps any `SyncRerank` implementor into a full
/// async [`Reranker`], suitable for use anywhere frankensearch expects one.
pub trait SyncRerank: Send + Sync {
    /// Synchronously rerank documents against a query.
    ///
    /// Returns documents sorted by descending cross-encoder score.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::SearchError`] when reranking fails (for example model load,
    /// inference, or input validation failures).
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>>;

    /// A unique identifier for this reranker model.
    fn id(&self) -> &str;

    /// Human-friendly reranker model name.
    fn model_name(&self) -> &str;

    /// Maximum supported token length for query+document pair input.
    fn max_length(&self) -> usize {
        512
    }

    /// Whether this reranker is loaded and ready for inference.
    fn is_available(&self) -> bool {
        true
    }
}

/// Adapts a [`SyncRerank`] implementor into a full async [`Reranker`].
///
/// The sync `rerank_sync()` call is wrapped in `Box::pin(async move { ... })`,
/// which is acceptable for blocking ONNX inference when called from a
/// `spawn_blocking` context. Cancellation is checked before inference and after
/// success or failure; already-running synchronous work is not preempted.
pub struct SyncRerankerAdapter<T: SyncRerank>(pub T);

impl<T: SyncRerank + 'static> Reranker for SyncRerankerAdapter<T> {
    fn rerank<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            embedding_checkpoint(cx, "sync_rerank.rerank")?;
            let outcome = self.0.rerank_sync(query, documents);
            embedding_checkpoint(cx, "sync_rerank.rerank")?;
            let mut scores = outcome?;
            scores.sort_by(|lhs, rhs| {
                rhs.score
                    .total_cmp(&lhs.score)
                    .then_with(|| lhs.original_rank.cmp(&rhs.original_rank))
                    .then_with(|| lhs.doc_id.cmp(&rhs.doc_id))
            });
            embedding_checkpoint(cx, "sync_rerank.rerank")?;
            Ok(scores)
        })
    }

    fn id(&self) -> &str {
        self.0.id()
    }

    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    fn max_length(&self) -> usize {
        self.0.max_length()
    }

    fn is_available(&self) -> bool {
        self.0.is_available()
    }
}
