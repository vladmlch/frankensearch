use frankensearch_core::error::{SearchError, SearchResult};
use tantivy::collector::{Count, TopDocs};
use tantivy::schema::Field;
use tantivy::{DocAddress, Searcher, TantivyDocument};
use tracing::{debug, instrument, warn};

use crate::types::{LexicalDocHit, LexicalSearchResult, SnippetConfig};

#[cfg(test)]
std::thread_local! {
    static COUNTED_COLLECTOR_INVOCATIONS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    static TOP_K_COLLECTOR_INVOCATIONS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub fn reset_collector_invocations() {
    COUNTED_COLLECTOR_INVOCATIONS.set(0);
    TOP_K_COLLECTOR_INVOCATIONS.set(0);
}

#[cfg(test)]
pub fn collector_invocations() -> (u64, u64) {
    (
        COUNTED_COLLECTOR_INVOCATIONS.get(),
        TOP_K_COLLECTOR_INVOCATIONS.get(),
    )
}

/// Execute a pre-built Tantivy query with offset pagination.
/// Execute a Tantivy collector search behind a panic guard.
///
/// The pinned Tantivy 0.26.1 scorer stack contains at least one panic
/// reachable from ordinary user input: a negated phrase whose exact sequence
/// is absent seeks a terminated docset in `PhraseScorer` (bd-nqeb4, found by
/// the bd-bsjw structure-aware campaign). A search engine must degrade, not
/// abort the host process, so the execution boundary converts an engine
/// panic into the same typed error surface as an engine `Err`. The searcher
/// is an immutable snapshot, so the unwind leaves no poisoned state behind.
pub fn search_guarded<C: tantivy::collector::Collector>(
    searcher: &Searcher,
    query: &dyn tantivy::query::Query,
    collector: &C,
) -> SearchResult<C::Fruit> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        searcher.search(query, collector)
    })) {
        Ok(result) => result.map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        }),
        Err(panic) => {
            let detail = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic payload");
            warn!(
                panic = detail,
                "tantivy panicked during query execution; degrading to a typed error (bd-nqeb4)"
            );
            Err(SearchError::SubsystemError {
                subsystem: "tantivy",
                source: format!("tantivy panicked during query execution: {detail}").into(),
            })
        }
    }
}

///
/// This helper centralizes error mapping and result-shape normalization so
/// downstream callers can keep custom query construction while reusing the
/// lexical execution core from this crate.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] when Tantivy search fails.
#[instrument(skip(searcher, query), fields(limit = limit, offset = offset))]
pub fn execute_query_with_offset(
    searcher: &Searcher,
    query: &dyn tantivy::query::Query,
    limit: usize,
    offset: usize,
) -> SearchResult<LexicalSearchResult> {
    #[cfg(test)]
    COUNTED_COLLECTOR_INVOCATIONS.set(COUNTED_COLLECTOR_INVOCATIONS.get().saturating_add(1));

    if limit == 0 {
        let total_count = search_guarded(searcher, query, &Count)?;
        return Ok(LexicalSearchResult {
            hits: Vec::new(),
            total_count,
        });
    }

    let (top_docs, total_count) = search_guarded(
        searcher,
        query,
        &(
            TopDocs::with_limit(limit)
                .and_offset(offset)
                .order_by_score(),
            Count,
        ),
    )?;

    let hits = top_docs
        .into_iter()
        .enumerate()
        .map(|(rank, (bm25_score, doc_address))| LexicalDocHit {
            bm25_score,
            rank,
            doc_address,
        })
        .collect();

    Ok(LexicalSearchResult { hits, total_count })
}

/// Execute a query for the top-`limit` hits by BM25 score without computing the
/// total match count.
///
/// Unlike [`execute_query_with_offset`], this omits the [`Count`] collector.
/// `Count` must visit every matching document, while this ID-only path only
/// needs the ranked page.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] when Tantivy search fails.
pub fn execute_top_k(
    searcher: &Searcher,
    query: &dyn tantivy::query::Query,
    limit: usize,
    offset: usize,
) -> SearchResult<Vec<LexicalDocHit>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    #[cfg(test)]
    TOP_K_COLLECTOR_INVOCATIONS.set(TOP_K_COLLECTOR_INVOCATIONS.get().saturating_add(1));

    let top_docs = search_guarded(
        searcher,
        query,
        &TopDocs::with_limit(limit)
            .and_offset(offset)
            .order_by_score(),
    )?;

    Ok(top_docs
        .into_iter()
        .enumerate()
        .map(|(rank, (bm25_score, doc_address))| LexicalDocHit {
            bm25_score,
            rank,
            doc_address,
        })
        .collect())
}

/// Load a stored Tantivy document by address.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] when document loading fails.
pub fn load_doc(searcher: &Searcher, doc_address: DocAddress) -> SearchResult<TantivyDocument> {
    searcher
        .doc(doc_address)
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })
}

/// Try to build a snippet generator for a query/content field pair.
///
/// Returns `None` if snippet generation cannot be initialized. This mirrors the
/// tolerant behavior used by `search_with_snippets`.
#[must_use]
pub fn try_build_snippet_generator(
    searcher: &Searcher,
    query: &dyn tantivy::query::Query,
    content_field: Field,
    snippet_config: &SnippetConfig,
) -> Option<tantivy::snippet::SnippetGenerator> {
    match tantivy::snippet::SnippetGenerator::create(searcher, query, content_field) {
        Ok(mut generator) => {
            generator.set_max_num_chars(snippet_config.max_chars);
            Some(generator)
        }
        Err(e) => {
            debug!(error = %e, "failed to create snippet generator, snippets will be absent");
            None
        }
    }
}

/// Render snippet HTML for a document with caller-specified highlight tags.
#[must_use]
pub fn render_snippet_html(
    snippet_generator: &tantivy::snippet::SnippetGenerator,
    doc: &TantivyDocument,
    highlight_prefix: &str,
    highlight_postfix: &str,
) -> Option<String> {
    let mut snippet = snippet_generator.snippet_from_doc(doc);
    snippet.set_snippet_prefix_postfix(highlight_prefix, highlight_postfix);
    let html = snippet.to_html();
    if html.is_empty() { None } else { Some(html) }
}
