use std::sync::Arc;

use ahash::AHashMap;
use unicode_normalization::UnicodeNormalization;

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::explanation::{
    ExplainedSource, ExplanationPhase, HitExplanation, ScoreComponent,
};
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::traits::{Embedder, ModelCategory};
use frankensearch_core::types::{ScoreSource, ScoredResult, VectorHit};

pub(super) struct NormalizedExclusions {
    pub(super) terms: Vec<String>,
    pub(super) phrases: Vec<String>,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(super) fn scaled_budget(base_candidates: usize, multiplier: f32) -> usize {
    if base_candidates == 0 || !multiplier.is_finite() || multiplier <= 0.0 {
        return 0;
    }
    let scaled = (base_candidates as f32 * multiplier).ceil() as usize;
    scaled.max(1)
}

pub(super) fn is_shipped_hash_embedder(embedder: &dyn Embedder) -> bool {
    if embedder.is_semantic() || embedder.category() != ModelCategory::HashEmbedder {
        return false;
    }
    frankensearch_core::is_hash_generation_id(embedder.id())
}

#[must_use]
pub(super) fn vector_exclusion_source(embedder: &dyn Embedder) -> &'static str {
    if is_shipped_hash_embedder(embedder) {
        "hash_control"
    } else {
        "semantic"
    }
}

/// Quality evidence may attach to any hit; it must not relabel retrieval
/// provenance. Lexical, hybrid, and hash-control stays stay put even when a
/// quality cosine is present.
#[must_use]
pub(super) const fn source_can_promote_to_quality(source: ScoreSource) -> bool {
    !matches!(
        source,
        ScoreSource::Lexical | ScoreSource::Hybrid | ScoreSource::HashControl
    )
}

#[must_use]
pub(super) const fn promoted_quality_source(
    source: ScoreSource,
    has_quality_score: bool,
) -> ScoreSource {
    if has_quality_score && source_can_promote_to_quality(source) {
        ScoreSource::SemanticQuality
    } else {
        source
    }
}

/// Convert `FusedHit` results to `ScoredResult`.
pub(super) fn fused_hits_to_scored_results(
    fused: &[frankensearch_core::types::FusedHit],
    lexical_results: &[ScoredResult],
    explain: bool,
    fast_embedder_id: &str,
    rrf_k: f64,
) -> Vec<ScoredResult> {
    // Borrow each candidate's metadata `&Value` into the map instead of cloning
    // it: only the `fused` (top-k) hits below ever read it, so cloning every
    // lexical candidate's (often dropped) metadata here is wasted. The `clone` is
    // deferred to the per-winner lookup, so a deep/large metadata `Value` for a
    // candidate that misses the top-k is never cloned. `lexical_results` outlives
    // this function, so the borrows are valid for the whole materialization.
    let lexical_metadata_by_doc: AHashMap<&str, &Arc<serde_json::Value>> = lexical_results
        .iter()
        .filter_map(|result| {
            result
                .metadata
                .as_ref()
                .map(|metadata| (result.doc_id.as_str(), metadata))
        })
        .collect();

    fused
        .iter()
        .map(|fh| {
            let mut fh = fh.clone();
            if frankensearch_core::is_hash_generation_id(fast_embedder_id) {
                fh.remap_hash_control_ranks();
            }
            #[allow(clippy::cast_possible_truncation)]
            let score = fh.rrf_score as f32;
            let source = crate::rrf::classify_fused_hit_source(&fh, fast_embedder_id);
            let explanation = explain.then(|| {
                let mut components = Vec::new();

                if let (Some(rank), Some(raw_score)) = (fh.lexical_rank, fh.lexical_score) {
                    components.push(ScoreComponent {
                        source: ExplainedSource::LexicalBm25 { terms: Vec::new() },
                        raw_score: f64::from(raw_score),
                        normalized_score: f64::from(raw_score),
                        rrf_contribution: rank_contribution(rrf_k, rank),
                        weight: 1.0,
                    });
                }

                let (vector_rank, vector_score) =
                    if frankensearch_core::is_hash_generation_id(fast_embedder_id) {
                        (fh.hash_rank, fh.hash_score)
                    } else {
                        (fh.semantic_rank, fh.semantic_score)
                    };
                if let (Some(rank), Some(raw_score)) = (vector_rank, vector_score) {
                    components.push(ScoreComponent {
                        source: ExplainedSource::vector_fast(
                            fast_embedder_id,
                            f64::from(raw_score),
                        ),
                        raw_score: f64::from(raw_score),
                        normalized_score: f64::from(raw_score),
                        rrf_contribution: rank_contribution(rrf_k, rank),
                        weight: 1.0,
                    });
                }

                Box::new(HitExplanation {
                    final_score: f64::from(score),
                    components,
                    phase: ExplanationPhase::Initial,
                    rank_movement: None,
                })
            });
            ScoredResult {
                doc_id: fh.doc_id.clone(),
                score,
                source,
                // A quality-only document carries the vector hit's "no fast
                // row" sentinel, which is not a fast-tier position.
                index: fh.semantic_index.filter(|&index| index != u32::MAX),
                fast_score: fh.hash_score.or(fh.semantic_score),
                quality_score: None,
                lexical_score: fh.lexical_score,
                rerank_score: None,
                explanation,
                metadata: lexical_metadata_by_doc
                    .get(fh.doc_id.as_str())
                    .copied()
                    .cloned(),
            }
        })
        .collect()
}

fn rank_contribution(rrf_k: f64, rank: usize) -> f64 {
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0 / (sanitize_rrf_k(rrf_k) + f64::from(rank_u32) + 1.0)
}

fn sanitize_rrf_k(k: f64) -> f64 {
    const DEFAULT_RRF_K: f64 = 60.0;
    if k.is_finite() && k >= 0.0 {
        k
    } else {
        DEFAULT_RRF_K
    }
}

pub(super) fn should_lexical_short_circuit(
    query_class: QueryClass,
    lexical_count: usize,
    k: usize,
    graph_ranking_enabled: bool,
    fast_embedder_is_semantic: bool,
    has_quality_embedder: bool,
) -> bool {
    if graph_ranking_enabled || k == 0 {
        return false;
    }

    if lexical_count >= k
        && matches!(
            query_class,
            QueryClass::Identifier | QueryClass::ShortKeyword
        )
    {
        return true;
    }

    !fast_embedder_is_semantic
        && !has_quality_embedder
        && (lexical_count == 0
            || (lexical_count >= k && matches!(query_class, QueryClass::NaturalLanguage)))
}

/// Convert `VectorHit` results to `ScoredResult` (semantic-only mode).
pub(super) fn vector_hits_to_scored_results(
    hits: &[VectorHit],
    k: usize,
    config: &TwoTierConfig,
    fast_embedder_id: &str,
) -> Vec<ScoredResult> {
    let mut seen = ahash::AHashSet::new();
    hits.iter()
        .filter(|h| seen.insert(&h.doc_id))
        .take(k)
        .map(|h| {
            let explanation = if config.explain {
                Some(Box::new(HitExplanation {
                    final_score: f64::from(h.score),
                    components: vec![ScoreComponent {
                        source: ExplainedSource::vector_fast(fast_embedder_id, f64::from(h.score)),
                        raw_score: f64::from(h.score),
                        normalized_score: f64::from(h.score),
                        rrf_contribution: 0.0,
                        weight: 1.0,
                    }],
                    phase: ExplanationPhase::Initial,
                    rank_movement: None,
                }))
            } else {
                None
            };

            ScoredResult {
                doc_id: h.doc_id.clone(),
                score: h.score,
                source: if frankensearch_core::is_hash_generation_id(fast_embedder_id) {
                    ScoreSource::HashControl
                } else {
                    ScoreSource::SemanticFast
                },
                index: Some(h.index),
                fast_score: Some(h.score),
                quality_score: None,
                lexical_score: None,
                rerank_score: None,
                explanation,
                metadata: None,
            }
        })
        .collect()
}

pub(super) fn filter_scored_results_by_negations(
    mut results: Vec<ScoredResult>,
    exclusions: &NormalizedExclusions,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> Vec<ScoredResult> {
    // In-place retain avoids allocating a new Vec (fff.nvim pattern:
    // never allocate when you can mutate in-place).
    results.retain(|result| !should_exclude_document(&result.doc_id, exclusions, text_fn, source));
    results
}

pub(super) fn filter_vector_hits_by_negations(
    mut hits: Vec<VectorHit>,
    exclusions: &NormalizedExclusions,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> Vec<VectorHit> {
    hits.retain(|hit| !should_exclude_document(&hit.doc_id, exclusions, text_fn, source));
    hits
}

pub(super) fn should_exclude_document(
    doc_id: &str,
    exclusions: &NormalizedExclusions,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> bool {
    let Some(text) = text_fn(doc_id) else {
        return false;
    };
    let Some(matched_clause) = find_negative_match(&text, exclusions) else {
        return false;
    };
    tracing::debug!(
        %doc_id,
        matched_exclusion_term = %matched_clause,
        source,
        "doc_excluded"
    );
    true
}

pub(super) fn find_negative_match(text: &str, exclusions: &NormalizedExclusions) -> Option<String> {
    let normalized_text = normalize_for_negation_match(text);
    for term in &exclusions.terms {
        if !term.is_empty() && contains_negative_term(&normalized_text, term) {
            return Some(term.clone());
        }
    }
    for phrase in &exclusions.phrases {
        if !phrase.is_empty() && normalized_text.contains(phrase) {
            return Some(phrase.clone());
        }
    }
    None
}

pub(super) fn normalize_for_negation_match(value: &str) -> String {
    // Fast path: for pure-ASCII input (the common case for query text and most
    // document bodies), NFC is the identity and Unicode lowercasing equals ASCII
    // lowercasing, so we skip the `unicode_normalization` composing iterator and
    // Unicode case-folding entirely. Byte-identical to the Unicode path for ASCII.
    if value.is_ascii() {
        return value.to_ascii_lowercase();
    }
    value.nfc().collect::<String>().to_lowercase()
}

/// Bench-only accessor for the shipped ASCII-fast-path negation normalizer.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_normalize_for_negation_match_fast(value: &str) -> String {
    normalize_for_negation_match(value)
}

/// Bench-only reference: the original always-Unicode negation normalizer, retained
/// as the exact-parity oracle and the A/B baseline.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_normalize_for_negation_match_reference(value: &str) -> String {
    value.nfc().collect::<String>().to_lowercase()
}

pub(super) fn contains_negative_term(normalized_text: &str, normalized_term: &str) -> bool {
    if term_is_word_like(normalized_term) {
        contains_term_with_word_boundaries(normalized_text, normalized_term)
    } else {
        normalized_text.contains(normalized_term)
    }
}

pub(super) fn term_is_word_like(term: &str) -> bool {
    term.chars().all(is_word_char)
}

pub(super) fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

pub(super) fn contains_term_with_word_boundaries(text: &str, term: &str) -> bool {
    let mut search_from = 0_usize;
    while let Some(relative_index) = text[search_from..].find(term) {
        let start = search_from + relative_index;
        let end = start + term.len();
        let prev = text[..start].chars().next_back();
        let next = text[end..].chars().next();
        let start_boundary = prev.is_none_or(|ch| !is_word_char(ch));
        let end_boundary = next.is_none_or(|ch| !is_word_char(ch));
        if start_boundary && end_boundary {
            return true;
        }
        search_from = end;
    }
    false
}

/// Poll an immediately-ready future to completion without an async runtime.
///
/// All frankensearch `Embedder::embed` and `LexicalRead::search` implementations
/// are synchronous code wrapped in `Box::pin(async move { ... })` — they complete
/// on the first poll. This helper exploits that property to call them from
/// synchronous contexts (e.g., `rayon::join` threads).
///
/// Returns `None` if the future is not immediately ready (i.e., yields `Pending`).
pub(super) fn poll_immediate<T>(
    mut fut: std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + '_>>,
) -> Option<T> {
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    match fut.as_mut().poll(&mut cx) {
        std::task::Poll::Ready(val) => Some(val),
        std::task::Poll::Pending => None,
    }
}
