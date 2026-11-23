//! Reciprocal Rank Fusion (RRF) for combining lexical and semantic search results.
//!
//! RRF is a principled, training-free method for fusing ranked lists from
//! different retrieval systems (Cormack et al., 2009).
//!
//! The score for a document appearing at rank `r` (0-based) in source `i` is:
//!
//! ```text
//! score(doc) = Σ_i  1 / (K + r_i + 1)
//! ```
//!
//! Documents appearing in multiple sources get their contributions summed,
//! which naturally boosts multi-source hits.

use std::collections::hash_map::Entry;

use ahash::AHashMap;
use frankensearch_core::{
    FusedHit, FusionStrategy, ScoreSource, ScoredResult, VectorHit, is_hash_generation_id,
};
use tracing::{Level, debug, instrument};

// ─── Configuration ──────────────────────────────────────────────────────────
const DEFAULT_RRF_K: f64 = 60.0;

/// RRF fusion parameters.
///
/// The `k` constant controls how steeply rank affects score:
/// - Higher K → flatter distribution (high and low ranks scored similarly)
/// - Lower K → sharper distribution (top ranks much more valuable)
///
/// K=60 is the empirically optimal value from the original paper and is
/// used in production at Elastic, Pinecone, and Vespa.
#[derive(Debug, Clone)]
pub struct RrfConfig {
    /// RRF constant K. Default: 60.0.
    pub k: f64,
    /// Multiplier applied to every lexical (BM25) tier RRF contribution. Default `1.0`
    /// (neutral). Up-weighting the *stronger* tier for the workload makes the hybrid
    /// strictly dominate the best single tier on both recall and nDCG
    /// (see `docs/NEGATIVE_EVIDENCE.md`). Non-finite or `≤ 0` values are treated as `1.0`.
    pub lexical_weight: f64,
    /// Multiplier applied to every semantic (vector) tier RRF contribution. Default `1.0`.
    /// See [`RrfConfig::lexical_weight`].
    pub semantic_weight: f64,
    /// How to break exact RRF-score ties. Default [`RrfTiebreak::LexicalThenId`] (legacy).
    pub tiebreak: RrfTiebreak,
}

/// Tiebreak strategy for documents with an identical RRF score *and* the same
/// both-sources status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RrfTiebreak {
    /// Legacy: prefer the higher lexical score, then `doc_id`. This is **asymmetric** —
    /// vector-only docs (no lexical score) always lose the tie, systematically demoting
    /// semantic-only best-answers (diagnosed in `docs/NEGATIVE_EVIDENCE.md`).
    #[default]
    LexicalThenId,
    /// Neutral: break ties by an unbiased hash of `doc_id` (then `doc_id` for
    /// determinism), so neither tier is favored. Measured a small nDCG / MRR gain over
    /// the lexical-favoring default. Note: never fall through to raw `doc_id` alone —
    /// that alphabetical bias is *worse* (see `docs/NEGATIVE_EVIDENCE.md`).
    Hash,
}

/// Deterministic, dependency-free FNV-1a hash of a `doc_id`, for the neutral tiebreak.
#[inline]
fn doc_id_tiebreak_hash(doc_id: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in doc_id.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            k: DEFAULT_RRF_K,
            lexical_weight: 1.0,
            semantic_weight: 1.0,
            tiebreak: RrfTiebreak::LexicalThenId,
        }
    }
}

/// Sanitize a tier weight: non-finite or non-positive values fall back to the neutral
/// `1.0`, so a bad config degrades to standard (unweighted) RRF rather than corrupting
/// scores.
#[inline]
fn sanitize_tier_weight(weight: f64) -> f64 {
    if weight.is_finite() && weight > 0.0 {
        weight
    } else {
        1.0
    }
}

// ─── Candidate Budget ───────────────────────────────────────────────────────

/// Compute how many candidates to fetch from each source.
///
/// Fetches `multiplier × (limit + offset)` to ensure good coverage for
/// documents that may rank differently across sources.
///
/// # Arguments
///
/// * `limit` - Number of final results desired.
/// * `offset` - Pagination offset.
/// * `multiplier` - Candidate multiplier (typically 3).
#[must_use]
pub const fn candidate_count(limit: usize, offset: usize, multiplier: usize) -> usize {
    limit.saturating_add(offset).saturating_mul(multiplier)
}

#[inline]
fn rank_contribution(k: f64, rank: usize) -> f64 {
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0 / (k + f64::from(rank_u32) + 1.0)
}

#[inline]
fn sanitize_rrf_k(k: f64) -> f64 {
    if k.is_finite() && k >= 0.0 {
        k
    } else {
        DEFAULT_RRF_K
    }
}

#[inline]
fn sanitize_graph_weight(weight: f64) -> f64 {
    if weight.is_finite() && weight > 0.0 {
        weight
    } else {
        0.0
    }
}

/// Label a fused hit by the lanes that actually contributed.
///
/// Hash/fnv/jl vector-only hits are [`ScoreSource::HashControl`], never
/// [`ScoreSource::SemanticFast`]. Lexical-only fused rows stay lexical.
#[must_use]
pub(crate) fn classify_fused_hit_source(hit: &FusedHit, fast_embedder_id: &str) -> ScoreSource {
    if hit.in_both_sources {
        ScoreSource::Hybrid
    } else if hit.lexical_rank.is_some() {
        ScoreSource::Lexical
    } else if hit.semantic_rank.is_some() || hit.hash_rank.is_some() {
        if is_hash_generation_id(fast_embedder_id) {
            ScoreSource::HashControl
        } else {
            ScoreSource::SemanticFast
        }
    } else {
        ScoreSource::Hybrid
    }
}

#[derive(Debug)]
struct FusedHitScratch<'a> {
    doc_id: &'a str,
    rrf_score: f64,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    hash_rank: Option<usize>,
    semantic_index: Option<u32>,
    graph_rank: Option<usize>,
    lexical_score: Option<f32>,
    semantic_score: Option<f32>,
    hash_score: Option<f32>,
    graph_score: Option<f32>,
    in_both_sources: bool,
}

impl FusedHitScratch<'_> {
    fn cmp_for_ranking(&self, other: &Self, tiebreak: RrfTiebreak) -> std::cmp::Ordering {
        let base = other
            .rrf_score
            .total_cmp(&self.rrf_score)
            .then(other.in_both_sources.cmp(&self.in_both_sources));
        match tiebreak {
            RrfTiebreak::LexicalThenId => base
                .then_with(|| {
                    let a = self.lexical_score.unwrap_or(f32::NEG_INFINITY);
                    let b = other.lexical_score.unwrap_or(f32::NEG_INFINITY);
                    b.total_cmp(&a)
                })
                .then_with(|| self.doc_id.cmp(other.doc_id)),
            RrfTiebreak::Hash => base
                .then_with(|| {
                    doc_id_tiebreak_hash(self.doc_id).cmp(&doc_id_tiebreak_hash(other.doc_id))
                })
                .then_with(|| self.doc_id.cmp(other.doc_id)),
        }
    }

    fn into_owned(self) -> FusedHit {
        FusedHit {
            doc_id: self.doc_id.into(),
            rrf_score: self.rrf_score,
            lexical_rank: self.lexical_rank,
            semantic_rank: self.semantic_rank,
            hash_rank: self.hash_rank,
            semantic_index: self.semantic_index,
            lexical_score: self.lexical_score,
            semantic_score: self.semantic_score,
            hash_score: self.hash_score,
            in_both_sources: self.in_both_sources,
        }
    }

    const fn vector_fields(
        rank: usize,
        score: f32,
        vector_is_hash: bool,
    ) -> (Option<usize>, Option<f32>, Option<usize>, Option<f32>) {
        if vector_is_hash {
            (None, None, Some(rank), Some(score))
        } else {
            (Some(rank), Some(score), None, None)
        }
    }

    const fn has_vector_rank(&self) -> bool {
        self.semantic_rank.is_some() || self.hash_rank.is_some()
    }

    fn assign_vector(&mut self, rank: usize, score: f32, index: u32, vector_is_hash: bool) {
        let (semantic_rank, semantic_score, hash_rank, hash_score) =
            Self::vector_fields(rank, score, vector_is_hash);
        self.semantic_rank = semantic_rank;
        self.semantic_score = semantic_score;
        self.hash_rank = hash_rank;
        self.hash_score = hash_score;
        self.semantic_index = Some(index);
    }

    fn vector_score(&self) -> Option<f32> {
        self.hash_score.or(self.semantic_score)
    }
}

// ─── RRF Fusion ─────────────────────────────────────────────────────────────

/// Fuse lexical and semantic search results using Reciprocal Rank Fusion.
///
/// # Algorithm
///
/// 1. Assign RRF scores: `1/(K + rank + 1)` for each source (0-based ranks).
/// 2. Sum scores for documents appearing in both sources.
/// 3. Sort by the 4-level deterministic ordering defined on [`FusedHit`]:
///    - RRF score descending
///    - `in_both_sources` (true preferred)
///    - Lexical score descending
///    - `doc_id` ascending (absolute determinism)
/// 4. Apply offset and limit for pagination.
///
/// # Arguments
///
/// * `lexical` - Lexical (BM25) search results, in descending relevance order.
/// * `semantic` - Vector search results, in descending score order.
///   The list may be a semantic generation or a hash-control generation;
///   fusion is rank-based and does not treat the name as a quality claim.
/// * `limit` - Maximum number of results to return.
/// * `offset` - Number of top results to skip (for pagination).
/// * `config` - RRF parameters (K constant).
#[must_use]
#[instrument(
    name = "frankensearch::rrf_fuse",
    skip(lexical, semantic),
    fields(
        lexical_count = lexical.len(),
        vector_count = semantic.len(),
        k = config.k,
        limit,
        offset,
    )
)]
pub fn rrf_fuse(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    // Merge-structured fusion: byte-identical to `rrf_fuse_with_graph`
    // (proven by `merge_matches_map_fusion`) but feeds the final sort a
    // near-sorted (semantic-ordered) input — 1.31-1.46× faster on the limit_all
    // shape, growing with N (`rrf_merge_fuse` bench).
    rrf_fuse_for_vector_lane(lexical, semantic, limit, offset, config, false)
}

/// Fuse lexical and vector results, placing the vector rank on `hash_rank`
/// when the vector lane is a hash-control generation.
#[must_use]
pub fn rrf_fuse_for_vector_lane(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    limit: usize,
    offset: usize,
    config: &RrfConfig,
    vector_is_hash: bool,
) -> Vec<FusedHit> {
    let mut fused = rrf_fuse_merge_inner(
        lexical,
        semantic,
        &[],
        0.0,
        limit,
        offset,
        config,
        true,
        vector_is_hash,
    );
    remap_fused_hits_for_hash_lane(&mut fused, vector_is_hash);
    fused
}

fn remap_fused_hits_for_hash_lane(hits: &mut [FusedHit], vector_is_hash: bool) {
    if !vector_is_hash {
        return;
    }
    for hit in hits {
        hit.remap_hash_control_ranks();
    }
}

/// Fuse lexical, semantic, and optional graph-ranked results with weighted RRF.
#[must_use]
#[allow(clippy::too_many_lines)]
#[instrument(
    name = "frankensearch::rrf_fuse_with_graph",
    skip(lexical, semantic, graph),
    fields(
        lexical_count = lexical.len(),
        vector_count = semantic.len(),
        graph_count = graph.len(),
        graph_weight,
        k = config.k,
        limit,
        offset,
    )
)]
pub fn rrf_fuse_with_graph(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    graph: &[ScoredResult],
    graph_weight: f64,
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    rrf_fuse_with_graph_for_vector_lane(
        lexical,
        semantic,
        graph,
        graph_weight,
        limit,
        offset,
        config,
        false,
    )
}

fn rrf_fuse_with_graph_for_vector_lane(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    graph: &[ScoredResult],
    graph_weight: f64,
    limit: usize,
    offset: usize,
    config: &RrfConfig,
    vector_is_hash: bool,
) -> Vec<FusedHit> {
    let k = sanitize_rrf_k(config.k);
    let lexical_weight = sanitize_tier_weight(config.lexical_weight);
    let semantic_weight = sanitize_tier_weight(config.semantic_weight);
    let graph_weight = sanitize_graph_weight(graph_weight);
    let tiebreak = config.tiebreak;
    // Adjusted for typical ~50% overlap to reduce over-allocation.
    let graph_len = if graph_weight > 0.0 { graph.len() } else { 0 };
    let capacity = (lexical.len() + semantic.len() + graph_len) * 3 / 4 + 1;
    let mut hits: AHashMap<&str, FusedHitScratch<'_>> = AHashMap::with_capacity(capacity);

    // Score lexical results.
    for (rank, result) in lexical.iter().enumerate() {
        let rrf_contribution = rank_contribution(k, rank) * lexical_weight;

        // Single hash lookup via `entry` instead of `get` (dedup probe) + `entry`
        // (update). We iterate in rank order (0, 1, ...), so the first occurrence
        // is the best one: if this doc already has a lexical rank, keep it and skip.
        match hits.entry(result.doc_id.as_str()) {
            Entry::Occupied(mut e) => {
                let hit = e.get_mut();
                if hit.lexical_rank.is_some() {
                    continue;
                }
                hit.rrf_score += rrf_contribution;
                hit.lexical_rank = Some(rank);
                hit.lexical_score = Some(result.score);
                // Compute in_both_sources inline: if the vector lane was already seen.
                if hit.has_vector_rank() {
                    hit.in_both_sources = true;
                }
            }
            Entry::Vacant(e) => {
                e.insert(FusedHitScratch {
                    doc_id: result.doc_id.as_str(),
                    rrf_score: rrf_contribution,
                    lexical_rank: Some(rank),
                    semantic_rank: None,
                    hash_rank: None,
                    semantic_index: None,
                    graph_rank: None,
                    lexical_score: Some(result.score),
                    semantic_score: None,
                    hash_score: None,
                    graph_score: None,
                    in_both_sources: false,
                });
            }
        }
    }

    // Score semantic results.
    for (rank, hit) in semantic.iter().enumerate() {
        let rrf_contribution = rank_contribution(k, rank) * semantic_weight;

        // Single hash lookup (see lexical loop): skip if already seen in the vector lane.
        match hits.entry(hit.doc_id.as_str()) {
            Entry::Occupied(mut e) => {
                let fh = e.get_mut();
                if fh.has_vector_rank() {
                    continue;
                }
                fh.rrf_score += rrf_contribution;
                fh.assign_vector(rank, hit.score, hit.index, vector_is_hash);
                // Compute in_both_sources inline: if lexical was already seen.
                if fh.lexical_rank.is_some() {
                    fh.in_both_sources = true;
                }
            }
            Entry::Vacant(e) => {
                let (semantic_rank, semantic_score, hash_rank, hash_score) =
                    FusedHitScratch::vector_fields(rank, hit.score, vector_is_hash);
                e.insert(FusedHitScratch {
                    doc_id: hit.doc_id.as_str(),
                    rrf_score: rrf_contribution,
                    lexical_rank: None,
                    semantic_rank,
                    hash_rank,
                    semantic_index: Some(hit.index),
                    graph_rank: None,
                    lexical_score: None,
                    semantic_score,
                    hash_score,
                    graph_score: None,
                    in_both_sources: false,
                });
            }
        }
    }

    if graph_weight > 0.0 {
        for (rank, result) in graph.iter().enumerate() {
            let rrf_contribution = rank_contribution(k, rank) * graph_weight;

            // Single hash lookup (see lexical loop): skip if already seen in graph.
            match hits.entry(result.doc_id.as_str()) {
                Entry::Occupied(mut e) => {
                    let hit = e.get_mut();
                    if hit.graph_rank.is_some() {
                        continue;
                    }
                    hit.rrf_score += rrf_contribution;
                    hit.graph_rank = Some(rank);
                    hit.graph_score = Some(result.score);
                }
                Entry::Vacant(e) => {
                    e.insert(FusedHitScratch {
                        doc_id: result.doc_id.as_str(),
                        rrf_score: rrf_contribution,
                        lexical_rank: None,
                        semantic_rank: None,
                        hash_rank: None,
                        semantic_index: None,
                        graph_rank: Some(rank),
                        lexical_score: None,
                        semantic_score: None,
                        hash_score: None,
                        graph_score: Some(result.score),
                        in_both_sources: false,
                    });
                }
            }
        }
    }

    // in_both_sources was computed inline during insertion — no separate pass needed.
    let mut results: Vec<FusedHitScratch<'_>> = hits.into_values().collect();

    let overlap_count = tracing::enabled!(target: "frankensearch.rrf", Level::DEBUG)
        .then(|| results.iter().filter(|h| h.in_both_sources).count());
    let fused_count = results.len();

    // Ranking window needed for pagination. For small windows this avoids
    // sorting every fused hit while preserving deterministic output order.
    let window = limit.saturating_add(offset);
    if window == 0 {
        if let Some(overlap_count) = overlap_count {
            debug!(
                target: "frankensearch.rrf",
                fused_count,
                overlap_count,
                output_count = 0,
                "rrf fusion complete"
            );
        }
        return Vec::new();
    }
    if window < results.len() {
        let nth_index = window.saturating_sub(1);
        results.select_nth_unstable_by(nth_index, |a, b| a.cmp_for_ranking(b, tiebreak));
        results.truncate(window);
    }

    // Deterministic comparator gives a total order, so unstable sort is safe
    // and avoids stable-sort overhead on large candidate sets.
    results.sort_unstable_by(|a, b| a.cmp_for_ranking(b, tiebreak));

    // Apply offset and limit.
    let output: Vec<FusedHit> = results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(FusedHitScratch::into_owned)
        .collect();

    if let Some(overlap_count) = overlap_count {
        debug!(
            target: "frankensearch.rrf",
            fused_count,
            overlap_count,
            output_count = output.len(),
            "rrf fusion complete"
        );
    }

    output
}

// ─── Pool-Local Min-Max Score Fusion ────────────────────────────────────────

/// Min and max of a tier's retrieved pool. Returns `(+inf, -inf)` for an empty
/// pool (so [`minmax_norm`] degenerates every score to `0.0`, contributing nothing).
#[inline]
fn pool_min_max(scores: impl Iterator<Item = f32>) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for s in scores {
        if s < min {
            min = s;
        }
        if s > max {
            max = s;
        }
    }
    (min, max)
}

/// Min-max normalize `score` into `[0, 1]` over its tier pool. A degenerate pool
/// (`max == min`, or empty) maps every score to `0.0` (no divide-by-zero, no
/// spurious ranking signal from a flat tier).
#[inline]
fn minmax_norm(score: f32, min: f32, max: f32) -> f64 {
    let range = max - min;
    if range > 0.0 {
        f64::from((score - min) / range)
    } else {
        0.0
    }
}

/// Fuse lexical and semantic results by **pool-local min-max score fusion** — a
/// drop-in alternative to [`rrf_fuse`] that recovers the score MAGNITUDE the rank
/// transform discards.
///
/// # Algorithm
///
/// 1. For each tier, min-max normalize its raw scores **within its retrieved pool**
///    (the input slice) to `[0, 1]`.
/// 2. A document a tier did *not* retrieve gets that tier's pool **minimum**
///    normalized value — which is `0.0` for min-max.
/// 3. The fused score is the tier-weighted sum of the two normalized scores.
/// 4. Sort by the same deterministic ordering as RRF ([`FusedHitScratch::cmp_for_ranking`],
///    the fused score living in `rrf_score`), then paginate.
///
/// # Why it beats RRF
///
/// RRF flattens a runaway top match (`score ≫ the rest of the pool`) and a marginal
/// one to the same "rank 1". Pool-local min-max keeps that magnitude, and calibrating
/// **over the retrieved pool** (not the zero-swamped full corpus) avoids the outlier
/// crushing that makes naive score fusion lose to RRF. Measured **+0.0038 mean
/// nDCG@10 over RRF across 4 BEIR corpora, never-negative** at pool depths 50 and 100
/// (`docs/NEGATIVE_EVIDENCE.md`, `45530fb`). Tier weights reuse [`RrfConfig`].
///
/// The opt-in *NQC dense down-weight* (see `docs/SEARCH_QUALITY_FINDINGS.md`, 2026-07-12)
/// needs **no change here**: a caller realizes it by scaling `config.semantic_weight`
/// per query by `clip(1 - beta * CDF(nqc_cv(lexical_scores)))` — the same `semantic_weight`
/// path already covered by the `pool_minmax_tier_weights_reweight_the_sum` test. Remaining
/// land work is entirely caller-side (the streaming cv-quantile CDF + that per-query scale).
#[must_use]
#[instrument(
    name = "frankensearch::pool_minmax_fuse",
    skip(lexical, semantic),
    fields(
        lexical_count = lexical.len(),
        vector_count = semantic.len(),
        limit,
        offset,
    )
)]
pub fn pool_minmax_fuse(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    pool_minmax_fuse_for_vector_lane(lexical, semantic, limit, offset, config, false)
}

fn pool_minmax_fuse_for_vector_lane(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    limit: usize,
    offset: usize,
    config: &RrfConfig,
    vector_is_hash: bool,
) -> Vec<FusedHit> {
    let lexical_weight = sanitize_tier_weight(config.lexical_weight);
    let semantic_weight = sanitize_tier_weight(config.semantic_weight);
    let tiebreak = config.tiebreak;

    // Pool statistics over each tier's retrieved pool (the input slice).
    let (lex_min, lex_max) = pool_min_max(lexical.iter().map(|r| r.score));
    let (sem_min, sem_max) = pool_min_max(semantic.iter().map(|h| h.score));

    let capacity = (lexical.len() + semantic.len()) * 3 / 4 + 1;
    let mut hits: AHashMap<&str, FusedHitScratch<'_>> = AHashMap::with_capacity(capacity);

    // Accumulate ranks + raw scores (dedup on first, best, occurrence — same as RRF).
    for (rank, result) in lexical.iter().enumerate() {
        match hits.entry(result.doc_id.as_str()) {
            Entry::Occupied(mut e) => {
                let hit = e.get_mut();
                if hit.lexical_rank.is_some() {
                    continue;
                }
                hit.lexical_rank = Some(rank);
                hit.lexical_score = Some(result.score);
                if hit.has_vector_rank() {
                    hit.in_both_sources = true;
                }
            }
            Entry::Vacant(e) => {
                e.insert(FusedHitScratch {
                    doc_id: result.doc_id.as_str(),
                    rrf_score: 0.0,
                    lexical_rank: Some(rank),
                    semantic_rank: None,
                    hash_rank: None,
                    semantic_index: None,
                    graph_rank: None,
                    lexical_score: Some(result.score),
                    semantic_score: None,
                    hash_score: None,
                    graph_score: None,
                    in_both_sources: false,
                });
            }
        }
    }
    for (rank, hit) in semantic.iter().enumerate() {
        match hits.entry(hit.doc_id.as_str()) {
            Entry::Occupied(mut e) => {
                let fh = e.get_mut();
                if fh.has_vector_rank() {
                    continue;
                }
                fh.assign_vector(rank, hit.score, hit.index, vector_is_hash);
                if fh.lexical_rank.is_some() {
                    fh.in_both_sources = true;
                }
            }
            Entry::Vacant(e) => {
                let (semantic_rank, semantic_score, hash_rank, hash_score) =
                    FusedHitScratch::vector_fields(rank, hit.score, vector_is_hash);
                e.insert(FusedHitScratch {
                    doc_id: hit.doc_id.as_str(),
                    rrf_score: 0.0,
                    lexical_rank: None,
                    semantic_rank,
                    hash_rank,
                    semantic_index: Some(hit.index),
                    graph_rank: None,
                    lexical_score: None,
                    semantic_score,
                    hash_score,
                    graph_score: None,
                    in_both_sources: false,
                });
            }
        }
    }

    let mut results: Vec<FusedHitScratch<'_>> = hits.into_values().collect();

    // Fused score = tier-weighted sum of pool-normalized scores; a tier that did not
    // retrieve a doc contributes its pool minimum (= 0.0 for min-max). Stored in
    // `rrf_score` so the existing deterministic comparator sorts on it.
    for h in &mut results {
        let lex_norm = h
            .lexical_score
            .map_or(0.0_f64, |s| minmax_norm(s, lex_min, lex_max));
        let sem_norm = h
            .vector_score()
            .map_or(0.0_f64, |s| minmax_norm(s, sem_min, sem_max));
        h.rrf_score = lexical_weight * lex_norm + semantic_weight * sem_norm;
    }

    let window = limit.saturating_add(offset);
    if window == 0 {
        return Vec::new();
    }
    if window < results.len() {
        let nth_index = window.saturating_sub(1);
        results.select_nth_unstable_by(nth_index, |a, b| a.cmp_for_ranking(b, tiebreak));
        results.truncate(window);
    }
    results.sort_unstable_by(|a, b| a.cmp_for_ranking(b, tiebreak));

    results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(FusedHitScratch::into_owned)
        .collect()
}

/// Merge-structured [`pool_minmax_fuse`] — **bit-identical** output, built so the final sort
/// receives a **near-sorted** (semantic-ordered) input.
///
/// The value-map [`pool_minmax_fuse`] accumulates every doc into one `N`-entry
/// `HashMap<&str, FusedHitScratch>` and then `into_values()` in **random** hash order, forcing
/// a from-scratch `O(N log N)` sort on the `limit_all` shape (window ≥ N). This variant instead
/// keeps only a small `&str → (rank, score)` **lexical** contribution map (cache-resident) and
/// walks the already-score-sorted `semantic` slice **once in order**, emitting each fused hit
/// directly into `results`. A semantic-only doc's fused score is the monotone
/// `semantic_weight · minmax_norm(score)`, so it lands in fused order and the final sort runs
/// near-`O(N)` (pdqsort is adaptive). Same structure that made [`rrf_fuse_with_graph_merge`]
/// beat the RRF value-map on the `limit_all` shape (`4aeb66b`, 1.31–1.46×).
///
/// **Bit-identical** to [`pool_minmax_fuse`]: f64 addition is commutative (emitting
/// `semantic + lexical` yields the same fused score as `lexical + semantic`), the same
/// first-occurrence dedup on each tier, and the same total-order [`FusedHitScratch::cmp_for_ranking`]
/// comparator. Verified by `pool_minmax_merge_matches_map`. This is the variant a searcher should
/// wire once pool-min-max fusion is enabled (as the searcher already uses
/// [`rrf_fuse_with_graph_merge_unique`] over the map RRF).
#[must_use]
#[allow(clippy::similar_names)] // `lex_map` (contribution map) vs `lex_max` (pool max)
#[instrument(
    name = "frankensearch::pool_minmax_fuse_merge",
    skip(lexical, semantic),
    fields(
        lexical_count = lexical.len(),
        vector_count = semantic.len(),
        limit,
        offset,
    )
)]
pub fn pool_minmax_fuse_merge(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    pool_minmax_fuse_merge_for_vector_lane(lexical, semantic, limit, offset, config, false)
}

#[allow(clippy::similar_names)] // `lex_map` (contribution map) vs `lex_max` (pool max)
fn pool_minmax_fuse_merge_for_vector_lane(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    limit: usize,
    offset: usize,
    config: &RrfConfig,
    vector_is_hash: bool,
) -> Vec<FusedHit> {
    let lexical_weight = sanitize_tier_weight(config.lexical_weight);
    let semantic_weight = sanitize_tier_weight(config.semantic_weight);
    let tiebreak = config.tiebreak;

    let (lex_min, lex_max) = pool_min_max(lexical.iter().map(|r| r.score));
    let (sem_min, sem_max) = pool_min_max(semantic.iter().map(|h| h.score));

    // Small cache-resident lexical contribution map (first occurrence wins, matching the
    // value-map version's `Entry::Occupied … lexical_rank.is_some() … continue`).
    let mut lex_map: AHashMap<&str, (usize, f32)> = AHashMap::with_capacity(lexical.len());
    for (rank, result) in lexical.iter().enumerate() {
        lex_map
            .entry(result.doc_id.as_str())
            .or_insert((rank, result.score));
    }

    let capacity = (lexical.len() + semantic.len()) * 3 / 4 + 1;
    let mut results: Vec<FusedHitScratch<'_>> = Vec::with_capacity(capacity);
    let mut seen_semantic: ahash::AHashSet<&str> = ahash::AHashSet::with_capacity(semantic.len());

    // Walk the score-sorted `semantic` slice once → near-sorted `results`.
    for (rank, hit) in semantic.iter().enumerate() {
        let doc_id = hit.doc_id.as_str();
        if !seen_semantic.insert(doc_id) {
            continue; // first semantic occurrence wins (matches the value-map version)
        }
        let lex = lex_map.remove(doc_id);
        let sem_norm = minmax_norm(hit.score, sem_min, sem_max);
        // `semantic + lexical` == the map version's `lexical + semantic` (f64 add commutes).
        let mut fused = semantic_weight * sem_norm;
        if let Some((_, lex_score)) = lex {
            fused += lexical_weight * minmax_norm(lex_score, lex_min, lex_max);
        }
        let (semantic_rank, semantic_score, hash_rank, hash_score) =
            FusedHitScratch::vector_fields(rank, hit.score, vector_is_hash);
        results.push(FusedHitScratch {
            doc_id,
            rrf_score: fused,
            lexical_rank: lex.map(|(r, _)| r),
            semantic_rank,
            hash_rank,
            semantic_index: Some(hit.index),
            graph_rank: None,
            lexical_score: lex.map(|(_, s)| s),
            semantic_score,
            hash_score,
            graph_score: None,
            in_both_sources: lex.is_some(),
        });
    }
    // Lexical-only docs (never seen in semantic): the semantic tier contributes its pool
    // minimum (= 0.0 for min-max), so the fused score is the lexical term alone — identical to
    // the map version's `lexical_weight · lex_norm + semantic_weight · 0.0`.
    for (doc_id, (lex_rank, lex_score)) in lex_map.drain() {
        results.push(FusedHitScratch {
            doc_id,
            rrf_score: lexical_weight * minmax_norm(lex_score, lex_min, lex_max),
            lexical_rank: Some(lex_rank),
            semantic_rank: None,
            hash_rank: None,
            semantic_index: None,
            graph_rank: None,
            lexical_score: Some(lex_score),
            semantic_score: None,
            hash_score: None,
            graph_score: None,
            in_both_sources: false,
        });
    }

    let window = limit.saturating_add(offset);
    if window == 0 {
        return Vec::new();
    }
    if window < results.len() {
        let nth_index = window.saturating_sub(1);
        results.select_nth_unstable_by(nth_index, |a, b| a.cmp_for_ranking(b, tiebreak));
        results.truncate(window);
    }
    results.sort_unstable_by(|a, b| a.cmp_for_ranking(b, tiebreak));

    results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(FusedHitScratch::into_owned)
        .collect()
}

/// Merge-structured RRF: identical result to [`rrf_fuse_with_graph`], built so the
/// final sort receives a **near-sorted** (semantic-ordered) input.
///
/// Instead of accumulating every doc into one `N`-entry value map (random
/// iteration order → a from-scratch O(N log N) sort), this keeps only small
/// `&str → (rank, score)` contribution maps for the lexical and graph sources
/// (cache-resident), then walks the already-score-sorted `semantic` slice **once
/// in order**, emitting each fused hit directly into `results`. Vector-only docs
/// land in fused order (their score is the monotone `1/(k+sem_rank+1)`), so the
/// sort runs near-O(N) (pdqsort is adaptive).
///
/// **Bit-identical** to the map version: the `rrf_score` is a sum of the same
/// per-source contributions, and f64 addition is commutative, so emitting
/// `semantic + lexical + graph` instead of `lexical + semantic + graph` yields the
/// byte-identical score; all other fields and the `in_both_sources` rule
/// (lexical ∧ semantic) are reproduced exactly. Verified by
/// `merge_matches_map_fusion`.
#[must_use]
pub fn rrf_fuse_with_graph_merge(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    graph: &[ScoredResult],
    graph_weight: f64,
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    rrf_fuse_merge_inner(
        lexical,
        semantic,
        graph,
        graph_weight,
        limit,
        offset,
        config,
        true,
        false,
    )
}

/// The single Phase-1 fusion entry point: dispatch the configured [`FusionStrategy`].
///
/// [`FusionStrategy::Rrf`] (the default) routes to [`rrf_fuse_with_graph_merge_unique`], so a
/// default-configured searcher produces **byte-identical** results to before this dispatch existed.
///
/// [`FusionStrategy::PoolMinMax`] routes to [`pool_minmax_fuse_merge`] — but **only when there is
/// no graph contribution**. That operator has no graph arm, and silently discarding a graph signal
/// the caller explicitly enabled would be worse than ignoring their fusion preference, so a
/// non-empty `graph` falls back to RRF. `graph_ranking_enabled` defaults to `false`, so in the
/// common case the preference is honoured.
#[must_use]
pub fn fuse_by_strategy(
    strategy: FusionStrategy,
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    graph: &[ScoredResult],
    graph_weight: f64,
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    fuse_by_strategy_for_vector_lane(
        strategy,
        lexical,
        semantic,
        graph,
        graph_weight,
        limit,
        offset,
        config,
        false,
    )
}

/// Fuse by strategy, remapping the vector rank onto `hash_rank` when the
/// vector lane is a hash-control generation.
#[must_use]
pub fn fuse_by_strategy_for_vector_lane(
    strategy: FusionStrategy,
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    graph: &[ScoredResult],
    graph_weight: f64,
    limit: usize,
    offset: usize,
    config: &RrfConfig,
    vector_is_hash: bool,
) -> Vec<FusedHit> {
    let mut fused = match strategy {
        FusionStrategy::PoolMinMax if graph.is_empty() => pool_minmax_fuse_merge_for_vector_lane(
            lexical,
            semantic,
            limit,
            offset,
            config,
            vector_is_hash,
        ),
        FusionStrategy::Rrf | FusionStrategy::PoolMinMax => rrf_fuse_merge_inner(
            lexical,
            semantic,
            graph,
            graph_weight,
            limit,
            offset,
            config,
            false,
            vector_is_hash,
        ),
    };
    remap_fused_hits_for_hash_lane(&mut fused, vector_is_hash);
    fused
}

/// Like [`rrf_fuse_with_graph_merge`] but **assumes the `semantic` slice has no
/// duplicate `doc_id`s**.
///
/// That is true for any vector-index `search_top_k` result, so this skips the
/// O(N) `seen_semantic` dedup set — saving N hash-inserts on the hot `limit_all`
/// path. Output is identical to the dedup version whenever `semantic` is in fact
/// unique; it only diverges on never-produced duplicate semantic hits.
#[must_use]
pub fn rrf_fuse_with_graph_merge_unique(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    graph: &[ScoredResult],
    graph_weight: f64,
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    rrf_fuse_merge_inner(
        lexical,
        semantic,
        graph,
        graph_weight,
        limit,
        offset,
        config,
        false,
        false,
    )
}

#[must_use]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn rrf_fuse_merge_inner(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    graph: &[ScoredResult],
    graph_weight: f64,
    limit: usize,
    offset: usize,
    config: &RrfConfig,
    dedup_semantic: bool,
    vector_is_hash: bool,
) -> Vec<FusedHit> {
    let k = sanitize_rrf_k(config.k);
    let lexical_weight = sanitize_tier_weight(config.lexical_weight);
    let semantic_weight = sanitize_tier_weight(config.semantic_weight);
    let graph_weight = sanitize_graph_weight(graph_weight);
    let tiebreak = config.tiebreak;
    let graph_active = graph_weight > 0.0;
    let graph_len = if graph_active { graph.len() } else { 0 };

    // Small, cache-resident contribution maps (first occurrence wins, matching the
    // map version's `Entry::Occupied … continue`).
    let mut lex_map: AHashMap<&str, (usize, f32)> = AHashMap::with_capacity(lexical.len());
    for (rank, result) in lexical.iter().enumerate() {
        lex_map
            .entry(result.doc_id.as_str())
            .or_insert((rank, result.score));
    }
    let mut graph_map: AHashMap<&str, (usize, f32)> = AHashMap::with_capacity(graph_len);
    if graph_active {
        for (rank, result) in graph.iter().enumerate() {
            graph_map
                .entry(result.doc_id.as_str())
                .or_insert((rank, result.score));
        }
    }

    let capacity = (lexical.len() + semantic.len() + graph_len) * 3 / 4 + 1;
    let mut results: Vec<FusedHitScratch<'_>> = Vec::with_capacity(capacity);
    // Defensive dedup of the semantic slice (keep first occurrence), mirroring the
    // map's `semantic_rank.is_some() … continue`. A `&str` set is far smaller than
    // the old value map, so it stays cache-friendlier. Skipped (`None`) when the
    // caller guarantees unique semantic doc_ids (the vector-index hot path).
    let mut seen_semantic: Option<ahash::AHashSet<&str>> =
        dedup_semantic.then(|| ahash::AHashSet::with_capacity(semantic.len()));

    for (rank, hit) in semantic.iter().enumerate() {
        let doc_id = hit.doc_id.as_str();
        if let Some(seen) = seen_semantic.as_mut()
            && !seen.insert(doc_id)
        {
            continue;
        }
        let lex = lex_map.remove(doc_id);
        let gr = if graph_active {
            graph_map.remove(doc_id)
        } else {
            None
        };
        let mut rrf_score = rank_contribution(k, rank) * semantic_weight;
        if let Some((lex_rank, _)) = lex {
            rrf_score += rank_contribution(k, lex_rank) * lexical_weight;
        }
        if let Some((graph_rank, _)) = gr {
            rrf_score += rank_contribution(k, graph_rank) * graph_weight;
        }
        let (semantic_rank, semantic_score, hash_rank, hash_score) =
            FusedHitScratch::vector_fields(rank, hit.score, vector_is_hash);
        results.push(FusedHitScratch {
            doc_id,
            rrf_score,
            lexical_rank: lex.map(|(r, _)| r),
            semantic_rank,
            hash_rank,
            semantic_index: Some(hit.index),
            graph_rank: gr.map(|(r, _)| r),
            lexical_score: lex.map(|(_, s)| s),
            semantic_score,
            hash_score,
            graph_score: gr.map(|(_, s)| s),
            in_both_sources: lex.is_some(),
        });
    }

    // Lexical-only docs (never seen in semantic).
    for (doc_id, (lex_rank, lex_score)) in lex_map.drain() {
        let gr = if graph_active {
            graph_map.remove(doc_id)
        } else {
            None
        };
        let mut rrf_score = rank_contribution(k, lex_rank) * lexical_weight;
        if let Some((graph_rank, _)) = gr {
            rrf_score += rank_contribution(k, graph_rank) * graph_weight;
        }
        results.push(FusedHitScratch {
            doc_id,
            rrf_score,
            lexical_rank: Some(lex_rank),
            semantic_rank: None,
            hash_rank: None,
            semantic_index: None,
            graph_rank: gr.map(|(r, _)| r),
            lexical_score: Some(lex_score),
            semantic_score: None,
            hash_score: None,
            graph_score: gr.map(|(_, s)| s),
            in_both_sources: false,
        });
    }

    // Graph-only docs (never seen in semantic or lexical).
    if graph_active {
        for (doc_id, (graph_rank, graph_score)) in graph_map.drain() {
            results.push(FusedHitScratch {
                doc_id,
                rrf_score: rank_contribution(k, graph_rank) * graph_weight,
                lexical_rank: None,
                semantic_rank: None,
                hash_rank: None,
                semantic_index: None,
                graph_rank: Some(graph_rank),
                lexical_score: None,
                semantic_score: None,
                hash_score: None,
                graph_score: Some(graph_score),
                in_both_sources: false,
            });
        }
    }

    let overlap_count = tracing::enabled!(target: "frankensearch.rrf", Level::DEBUG)
        .then(|| results.iter().filter(|h| h.in_both_sources).count());
    let fused_count = results.len();

    let window = limit.saturating_add(offset);
    if window == 0 {
        if let Some(overlap_count) = overlap_count {
            debug!(
                target: "frankensearch.rrf",
                fused_count,
                overlap_count,
                output_count = 0,
                "rrf fusion complete"
            );
        }
        return Vec::new();
    }
    if window < results.len() {
        let nth_index = window.saturating_sub(1);
        results.select_nth_unstable_by(nth_index, |a, b| a.cmp_for_ranking(b, tiebreak));
        results.truncate(window);
    }
    results.sort_unstable_by(|a, b| a.cmp_for_ranking(b, tiebreak));

    let output: Vec<FusedHit> = results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(FusedHitScratch::into_owned)
        .collect();

    if let Some(overlap_count) = overlap_count {
        debug!(
            target: "frankensearch.rrf",
            fused_count,
            overlap_count,
            output_count = output.len(),
            "rrf fusion complete"
        );
    }

    output
}

#[cfg(test)]
mod tests;
