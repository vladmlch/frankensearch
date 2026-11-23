use std::any::Any;
use std::borrow::Cow;
use std::fmt;

use asupersync::Cx;

use super::SearchFuture;
use crate::error::{SearchError, SearchResult};
use crate::types::{IndexableDocument, ScoredResult};

// ─── Lexical Search Trait (REMOVED — bd-8nqz.1) ─────────────────────────────
//
// `LexicalSearch` combined search, metadata hydration, indexing and commit in
// one trait. That forced read-only consumers and `TwoTierSearcher` to hold a
// writer-capable backend, and it let hydration read a newer snapshot than the
// one that scored a candidate batch. It is replaced by `LexicalRead` and
// `LexicalWrite` below; every backend implements them directly, with no
// compatibility shim.

// ─── Split lexical traits (bd-8nqz.1) ───────────────────────────────────────
//
// The retired combined trait mixed read and write concerns, which forced
// read-only consumers to hold writer-capable backends and let hydration read a
// newer snapshot than the one that scored a candidate batch. This is the
// replacement contract: [`LexicalRead`] for search plus generation-pinned
// hydration, [`LexicalWrite`] for mutation.

/// Opaque, backend-owned pin of the immutable snapshot that scored a
/// candidate batch (bd-8nqz.1).
///
/// A backend stores whatever it needs — typically an `Arc` of its published
/// search snapshot — and downcasts it back during hydration, so hydrated
/// metadata always comes from the exact generation that produced the scores.
/// Callers cannot forge or relabel a context: the payload is opaque, the
/// backend tag is read-only, and a context only originates from a backend's
/// own [`LexicalRead::search_candidates`].
pub struct LexicalHydrationContext {
    backend: &'static str,
    inner: Box<dyn Any + Send + Sync>,
}

impl LexicalHydrationContext {
    /// Wrap a backend-owned snapshot pin.
    #[must_use]
    pub fn new(backend: &'static str, inner: Box<dyn Any + Send + Sync>) -> Self {
        Self { backend, inner }
    }

    /// Stable tag of the backend that produced this context.
    #[must_use]
    pub const fn backend(&self) -> &'static str {
        self.backend
    }

    /// Downcast the opaque payload back to the backend's snapshot type.
    ///
    /// Returns `None` for a foreign context (wrong backend or wrong payload
    /// type) — backends must treat that as a typed error, never a silent
    /// no-op, so cross-engine mixing cannot pass unnoticed.
    #[must_use]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

impl fmt::Debug for LexicalHydrationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexicalHydrationContext")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

/// Typed result of a fusion-candidate search: scored candidates plus the
/// snapshot pin required to hydrate them from the exact immutable generation
/// that scored them (bd-8nqz.1).
#[derive(Debug)]
pub struct LexicalCandidateBatch {
    results: Vec<ScoredResult>,
    context: Option<LexicalHydrationContext>,
}

impl LexicalCandidateBatch {
    /// Batch whose results already carry full metadata (no hydration needed).
    #[must_use]
    pub const fn eager(results: Vec<ScoredResult>) -> Self {
        Self {
            results,
            context: None,
        }
    }

    /// Batch with deferred metadata, pinned to the scoring snapshot.
    #[must_use]
    pub const fn deferred(results: Vec<ScoredResult>, context: LexicalHydrationContext) -> Self {
        Self {
            results,
            context: Some(context),
        }
    }

    /// Scored candidates in backend rank order.
    #[must_use]
    pub fn results(&self) -> &[ScoredResult] {
        &self.results
    }

    /// Snapshot pin for hydration; `None` when the batch is eager.
    #[must_use]
    pub const fn context(&self) -> Option<&LexicalHydrationContext> {
        self.context.as_ref()
    }

    /// Whether metadata hydration is required for final winners.
    #[must_use]
    pub const fn is_deferred(&self) -> bool {
        self.context.is_some()
    }

    /// Decompose into candidates and the hydration pin.
    #[must_use]
    pub fn into_parts(self) -> (Vec<ScoredResult>, Option<LexicalHydrationContext>) {
        (self.results, self.context)
    }
}

/// Read-only lexical search surface (bd-8nqz.1).
///
/// Search consumers (`TwoTierSearcher`, the sync searcher, `open_hybrid`
/// callers) depend on this trait alone, so a read-only reader never needs a
/// writer-capable backend or a writer lease.
pub trait LexicalRead: Send + Sync {
    /// The text a search consumer hands to [`Self::search`] or
    /// [`Self::search_candidates`] for a free-text user query.
    ///
    /// Code identifiers and URLs contain colons (`HashMap::new`,
    /// `https://x`) that a field-aware parser reads as `field:value` and
    /// drops when no such field exists. A backend whose parser does that
    /// escapes every colon that does not end one of its own field names.
    /// Consumers apply this once to user text; `search` keeps the parser's raw
    /// syntax. The default returns the query unchanged.
    fn query_text<'q>(&self, query: &'q str) -> Cow<'q, str> {
        Cow::Borrowed(query)
    }

    /// Search for documents matching the query, returning up to `limit`
    /// results sorted by BM25 relevance, with full metadata attached.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the query cannot be parsed or the backend
    /// fails.
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>>;

    /// Search for fusion candidates as a typed batch pinned to the scoring
    /// snapshot.
    ///
    /// The default preserves full-metadata results as an eager batch.
    /// Backends with a cheaper deferred-metadata path override this and
    /// return [`LexicalCandidateBatch::deferred`] with their snapshot pin.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` under the same conditions as [`Self::search`].
    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        Box::pin(async move {
            Ok(LexicalCandidateBatch::eager(
                self.search(cx, query, limit).await?,
            ))
        })
    }

    /// Restore metadata for final winners produced from a deferred candidate
    /// batch, reading from the pinned scoring snapshot — never from a newer
    /// generation.
    ///
    /// Implementations must ignore results without a lexical score (those
    /// did not survive from the lexical candidate pool) and must reject a
    /// foreign context with a typed error.
    ///
    /// # The default fails closed on any context
    ///
    /// A backend that does not override this method also does not override
    /// [`Self::search_candidates`], so it only ever issues *eager* batches,
    /// whose metadata was attached by the scoring search itself and whose
    /// context is therefore `None`. Such a backend can never legitimately be
    /// handed a context — so receiving one means a caller mixed a batch from
    /// another engine or generation into this reader, which is precisely what
    /// the hydration capability exists to prevent.
    ///
    /// This used to be `let _ = context; Ok(())`, which accepted *any*
    /// context — including another engine's snapshot pin — and returned
    /// success. The observable damage was silent: deferred winners came back
    /// with no metadata restored and no error raised, so a cross-engine mix
    /// degraded to missing fields rather than failing. Quill already rejected
    /// a foreign pin with a typed error; every eager-batch backend (Tantivy,
    /// FTS5, the shadow adapter) inherited the permissive default and did not.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if winner metadata cannot be materialized, or if
    /// a context is supplied to a backend that issues only eager batches.
    fn hydrate_candidates<'a>(
        &'a self,
        _cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        _results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        let foreign_backend = context.map(LexicalHydrationContext::backend);
        Box::pin(async move {
            foreign_backend.map_or(Ok(()), |backend| {
                Err(SearchError::SubsystemError {
                    subsystem: "lexical.hydration",
                    source: format!(
                        "hydration context from backend {backend:?} was supplied to a backend \
                         that issues only eager candidate batches; refusing cross-engine or \
                         cross-generation hydration"
                    )
                    .into(),
                })
            })
        })
    }

    /// Number of documents currently searchable.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when the backend cannot establish an authoritative
    /// count for its current readable generation.
    fn doc_count(&self) -> SearchResult<usize>;
}

/// Mutation/indexing surface of a lexical backend (bd-8nqz.1).
pub trait LexicalWrite: Send + Sync {
    /// Index a single document for full-text search.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the document cannot be indexed.
    fn index_document<'a>(&'a self, cx: &'a Cx, doc: &'a IndexableDocument)
    -> SearchFuture<'a, ()>;

    /// Index a batch of documents.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if any document cannot be indexed.
    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            for doc in docs {
                self.index_document(cx, doc).await?;
            }
            Ok(())
        })
    }

    /// Commit any pending writes to the index.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the commit fails (e.g., I/O error).
    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()>;
}
