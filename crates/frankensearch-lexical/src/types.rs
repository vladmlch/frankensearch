use frankensearch_core::types::DocId;
use serde::{Deserialize, Serialize};
use tantivy::DocAddress;

/// Default Tantivy snippet window value. Tantivy names this a character limit,
/// but observably compares UTF-8 byte offsets and cuts at token boundaries.
pub const DEFAULT_SNIPPET_MAX_CHARS: usize = 200;

// ─── Query Explanation ──────────────────────────────────────────────────────

/// Classification of a parsed query for debugging and diagnostics.
///
/// Returned by [`crate::TantivyIndex::search_with_snippets`] to help callers
/// understand how a query was interpreted by Tantivy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryExplanation {
    /// Query was empty or whitespace-only; no search performed.
    Empty,
    /// Single-term query (e.g., `"authentication"`).
    Simple,
    /// Quoted phrase query (e.g., `"error handling"`).
    Phrase,
    /// Multi-term query interpreted as boolean OR (default Tantivy behavior).
    Boolean,
}

impl std::fmt::Display for QueryExplanation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty"),
            Self::Simple => write!(f, "simple"),
            Self::Phrase => write!(f, "phrase"),
            Self::Boolean => write!(f, "boolean"),
        }
    }
}

/// Classify a raw query string into a [`QueryExplanation`].
pub fn classify_query(query: &str) -> QueryExplanation {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryExplanation::Empty;
    }
    // Check for quoted phrase: starts and ends with matching quotes.
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        return QueryExplanation::Phrase;
    }
    // Count whitespace-separated tokens.
    let token_count = trimmed.split_whitespace().count();
    if token_count <= 1 {
        QueryExplanation::Simple
    } else {
        QueryExplanation::Boolean
    }
}

// ─── Snippet Configuration ──────────────────────────────────────────────────

/// Configuration for snippet generation in [`crate::TantivyIndex::search_with_snippets`].
#[derive(Debug, Clone)]
pub struct SnippetConfig {
    /// Tantivy snippet window value. Despite the upstream name, the pinned
    /// implementation enforces this against UTF-8 byte offsets at token
    /// boundaries.
    pub max_chars: usize,
    /// HTML tag prefix for highlighted terms (e.g., `"<b>"`).
    pub highlight_prefix: String,
    /// HTML tag postfix for highlighted terms (e.g., `"</b>"`).
    pub highlight_postfix: String,
}

impl Default for SnippetConfig {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_SNIPPET_MAX_CHARS,
            highlight_prefix: "<b>".to_owned(),
            highlight_postfix: "</b>".to_owned(),
        }
    }
}

// ─── LexicalHit ─────────────────────────────────────────────────────────────

/// An enriched search result from [`crate::TantivyIndex::search_with_snippets`].
///
/// Contains everything in [`frankensearch_core::types::ScoredResult`] plus a snippet and query explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalHit {
    /// Unique document identifier.
    pub doc_id: String,
    /// BM25 relevance score.
    pub bm25_score: f32,
    /// 0-based rank in the result set.
    pub rank: usize,
    /// Highlighted snippet from the content field, if available.
    pub snippet: Option<String>,
    /// How the query was classified.
    pub query_type: QueryExplanation,
    /// Arbitrary document metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Raw lexical hit containing BM25 score and Tantivy doc address.
///
/// This is useful for callers that need custom field extraction from stored
/// documents while still reusing frankensearch's query execution helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalDocHit {
    /// BM25 relevance score returned by Tantivy.
    pub bm25_score: f32,
    /// 0-based rank in the returned page.
    pub rank: usize,
    /// Tantivy document address inside a segment.
    pub doc_address: DocAddress,
}

/// Paginated lexical search result containing both the matched hits and the
/// total number of documents matching the query.
///
/// The `total_count` reflects **all** matching documents in the index, not just
/// the page returned in `hits`. Clients can use this for pagination UI
/// (e.g., `page 2 of ceil(total_count / page_size)`).
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalSearchResult {
    /// The paginated slice of matching documents.
    pub hits: Vec<LexicalDocHit>,
    /// Total number of documents matching the query across the entire index.
    pub total_count: usize,
}

/// Lightweight lexical hit used by hot paths that only need `doc_id` + score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalIdHit {
    /// Unique document identifier. `DocId` (`CompactString`, SSO) so the
    /// per-hit id-materialization clone (`ord_table` lookup / docstore) is an
    /// inline memcpy for short ids and needs no `String→CompactString`
    /// re-conversion when it flows into `ScoredResult` at the fusion boundary.
    pub doc_id: DocId,
    /// BM25 relevance score.
    pub bm25_score: f32,
    /// 0-based rank in the returned page.
    pub rank: usize,
}

/// Dev-only ranked oracle evidence retaining Tantivy's full native tie key.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleRankedHit {
    /// External document identifier used for cross-engine alignment.
    pub doc_id: String,
    /// Exact BM25 score representation.
    pub score_bits: u32,
    /// Native rank in the fetched oracle result list.
    pub rank: usize,
    /// Tantivy segment ordinal (the first half of `DocAddress`).
    pub segment_ord: u32,
    /// Segment-local document ID (the second half of `DocAddress`).
    pub segment_doc_id: u32,
    /// Rendered snippet; `None` remains distinct from an empty snippet.
    pub snippet: Option<String>,
}

/// Dev-only complete observation used by the Quill conformance gauntlet.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleQueryObservation {
    /// Requested top-k rows in native Tantivy order.
    pub hits: Vec<OracleRankedHit>,
    /// Expanded exact-score group at the top-k cutoff.
    pub cutoff_tie_group: Vec<OracleRankedHit>,
    /// False when the configured expansion budget ended inside the tie group.
    pub cutoff_tie_complete: bool,
    /// Exact total number of matches, independent of top-k.
    pub total_count: usize,
    /// Exact live-document count.
    pub doc_count: usize,
    /// Every row of the expanded fetch in native rank order: a prefix
    /// `[0, n)` of the full ordering on the SAME pinned searcher, a superset
    /// of `hits`. Conformance certificates derive whole-group expansion and
    /// boundary witnesses from it instead of inferring completeness from the
    /// fetch capacity (bd-pjvl1).
    pub expanded_hits: Vec<OracleRankedHit>,
    /// Domain-separated SHA-256 of the pinned searcher's physical identity
    /// (every segment id, max doc, deleted count, and the live count).
    pub searcher_sha256: [u8; 32],
}

/// Physical identity of one pinned Tantivy searcher.
#[cfg(feature = "tantivy-oracle")]
pub fn oracle_searcher_sha256(searcher: &tantivy::Searcher) -> [u8; 32] {
    use sha2::Digest as _;

    let mut hasher = sha2::Sha256::new();
    hasher.update(b"frankensearch/lexical/oracle-searcher/v1\0");
    hasher.update(searcher.num_docs().to_be_bytes());
    let readers = searcher.segment_readers();
    hasher.update(
        u64::try_from(readers.len())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    for reader in readers {
        hasher.update(reader.segment_id().uuid_string().as_bytes());
        hasher.update(reader.max_doc().to_be_bytes());
        hasher.update(reader.num_deleted_docs().to_be_bytes());
    }
    hasher.finalize().into()
}

/// Dev-only exact counted page used by the Quill replacement witness.
///
/// This is deliberately separate from [`OracleQueryObservation`]: the latter
/// owns cutoff-tie expansion for rank-parity campaigns, while this DTO proves
/// the incumbent's real offset-pagination contract. Shipping callers should
/// not depend on this evidence-only surface.
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OraclePageObservation {
    /// Ordered page returned by Tantivy's real offset collector.
    pub hits: Vec<OraclePageHit>,
    /// Exact number of matches, independent of `limit` and `offset`.
    pub total_count: usize,
    /// Exact live-document count.
    pub doc_count: usize,
}

/// One ordered hit from [`OraclePageObservation`].
#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OraclePageHit {
    /// Stable external document identifier.
    pub doc_id: String,
    /// Exact BM25 score representation.
    pub score_bits: u32,
    /// Zero-based rank inside this returned page.
    pub page_rank: usize,
    /// Zero-based rank in the complete result stream.
    pub absolute_rank: usize,
}

/// Ordered searchable-segment geometry from the pinned Tantivy oracle.
///
/// This conformance-only receipt preserves Tantivy's native `segment_ord`
/// assignment as well as physical and live document counts. It is absent from
/// normal shipping builds.
#[cfg(feature = "tantivy-oracle")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleSegmentLayout {
    /// Native Tantivy segment ordinal used in `DocAddress` tie-breaking.
    pub segment_ord: u32,
    /// Physical document cardinality, including deleted rows.
    pub max_doc: u32,
    /// Live document cardinality.
    pub num_docs: u32,
}
