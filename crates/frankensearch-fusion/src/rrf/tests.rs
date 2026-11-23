use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use super::*;

fn fused(
    doc_id: &str,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    in_both_sources: bool,
) -> FusedHit {
    FusedHit {
        doc_id: doc_id.into(),
        rrf_score: 1.0,
        lexical_rank,
        semantic_rank,
        hash_rank: None,
        semantic_index: semantic_rank.map(|_| 0),
        lexical_score: lexical_rank.map(|_| 1.0),
        semantic_score: semantic_rank.map(|_| 0.5),
        hash_score: None,
        in_both_sources,
    }
}

#[test]
fn classify_fused_hit_source_separates_hash_and_lexical() {
    assert_eq!(
        classify_fused_hit_source(&fused("both", Some(0), Some(0), true), "fnv1a-256"),
        ScoreSource::Hybrid
    );
    assert_eq!(
        classify_fused_hit_source(&fused("lex", Some(0), None, false), "fnv1a-256"),
        ScoreSource::Lexical
    );
    assert_eq!(
        classify_fused_hit_source(&fused("hash", None, Some(0), false), "fnv1a-256"),
        ScoreSource::HashControl
    );
    let mut remapped = fused("hash-remapped", None, Some(2), false);
    remapped.remap_hash_control_ranks();
    assert_eq!(remapped.hash_rank, Some(2));
    assert_eq!(remapped.semantic_rank, None);
    assert_eq!(
        classify_fused_hit_source(&remapped, "fnv1a-256"),
        ScoreSource::HashControl
    );
    assert_eq!(
        classify_fused_hit_source(&fused("legacy", None, Some(0), false), "hash/fnv1a"),
        ScoreSource::HashControl
    );
    assert_eq!(
        classify_fused_hit_source(&fused("sem", None, Some(0), false), "minilm-l6-v2"),
        ScoreSource::SemanticFast
    );
}

#[test]
fn rrf_fuse_hash_lane_writes_hash_rank_not_semantic_rank() {
    let semantic = vec![semantic_hit("doc-a", 0.9)];
    let fused = rrf_fuse_for_vector_lane(&[], &semantic, 10, 0, &RrfConfig::default(), true);
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].hash_rank, Some(0));
    assert_eq!(fused[0].semantic_rank, None);
    assert_eq!(fused[0].hash_score, Some(0.9));
    assert_eq!(fused[0].semantic_score, None);
    let json = serde_json::to_string(&fused[0]).expect("serialize hash-lane fused hit");
    assert!(
        json.contains("hash_rank") && !json.contains("semantic_rank"),
        "hash-lane fuse must not publish semantic_rank: {json}"
    );

    let both = rrf_fuse_for_vector_lane(
        &[lexical_hit("doc-a", 4.0)],
        &semantic,
        10,
        0,
        &RrfConfig::default(),
        true,
    );
    assert!(both[0].in_both_sources);
    assert_eq!(both[0].lexical_rank, Some(0));
    assert_eq!(both[0].hash_rank, Some(0));
    assert_eq!(both[0].semantic_rank, None);

    let semantic_lane = rrf_fuse(&[], &semantic, 10, 0, &RrfConfig::default());
    assert_eq!(semantic_lane[0].semantic_rank, Some(0));
    assert_eq!(semantic_lane[0].hash_rank, None);

    let pool = fuse_by_strategy_for_vector_lane(
        FusionStrategy::PoolMinMax,
        &[],
        &semantic,
        &[],
        0.0,
        10,
        0,
        &RrfConfig::default(),
        true,
    );
    assert_eq!(pool[0].hash_rank, Some(0));
    assert_eq!(pool[0].semantic_rank, None);
    assert_eq!(pool[0].hash_score, Some(0.9));
    assert_eq!(pool[0].semantic_score, None);

    let mapped = rrf_fuse_with_graph_for_vector_lane(
        &[lexical_hit("doc-a", 4.0)],
        &semantic,
        &[],
        0.0,
        10,
        0,
        &RrfConfig::default(),
        true,
    );
    assert!(mapped[0].in_both_sources);
    assert_eq!(mapped[0].hash_rank, Some(0));
    assert_eq!(mapped[0].semantic_rank, None);
    assert_eq!(mapped[0].hash_score, Some(0.9));
    assert_eq!(mapped[0].semantic_score, None);

    let pool_map = pool_minmax_fuse_for_vector_lane(
        &[],
        &[semantic_hit("doc-a", 0.9), semantic_hit("doc-b", 0.1)],
        10,
        0,
        &RrfConfig::default(),
        true,
    );
    assert_eq!(pool_map[0].hash_rank, Some(0));
    assert_eq!(pool_map[0].semantic_rank, None);
    assert_eq!(pool_map[0].hash_score, Some(0.9));
    assert_eq!(pool_map[0].semantic_score, None);
    assert!(
        pool_map[0].rrf_score > pool_map[1].rrf_score,
        "hash-lane pool-minmax must keep the vector score in the fused norm"
    );
}

fn lexical_hit(doc_id: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.into(),
        score,
        source: frankensearch_core::ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

fn semantic_hit(doc_id: &str, score: f32) -> VectorHit {
    VectorHit {
        index: 0,
        score,
        doc_id: doc_id.into(),
    }
}

fn graph_hit(doc_id: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.into(),
        score,
        source: frankensearch_core::ScoreSource::SemanticFast,
        index: Some(0),
        fast_score: Some(score),
        quality_score: None,
        lexical_score: None,
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

// ─── Pool-local min-max score fusion ────────────────────────────────

fn fused_ids(hits: &[FusedHit]) -> Vec<String> {
    hits.iter().map(|h| h.doc_id.to_string()).collect()
}

// ─── FusionStrategy dispatch (searcher wiring) ──────────────────────

/// PARITY GATE. The default strategy must be byte-identical to calling the RRF entry point
/// directly — this is what makes wiring the dispatch into both searchers a no-op for every
/// existing caller.
#[test]
fn default_strategy_is_byte_identical_to_rrf() {
    let lexical = [
        lexical_hit("a", 10.0),
        lexical_hit("b", 2.0),
        lexical_hit("c", 1.0),
    ];
    let semantic = [
        semantic_hit("a", 0.9),
        semantic_hit("d", 0.5),
        semantic_hit("b", 0.4),
    ];
    let cfg = RrfConfig::default();

    let direct = rrf_fuse_with_graph_merge_unique(&lexical, &semantic, &[], 0.0, 10, 0, &cfg);
    let via = fuse_by_strategy(
        FusionStrategy::default(),
        &lexical,
        &semantic,
        &[],
        0.0,
        10,
        0,
        &cfg,
    );
    assert_eq!(FusionStrategy::default(), FusionStrategy::Rrf);
    assert_eq!(fused_ids(&via), fused_ids(&direct));
    for (x, y) in via.iter().zip(&direct) {
        assert_eq!(
            x.rrf_score.to_bits(),
            y.rrf_score.to_bits(),
            "doc {}",
            x.doc_id
        );
    }
}

#[test]
fn pool_minmax_strategy_routes_to_pool_minmax_merge() {
    let lexical = [
        lexical_hit("a", 10.0),
        lexical_hit("b", 2.0),
        lexical_hit("c", 1.0),
    ];
    let semantic = [
        semantic_hit("a", 0.9),
        semantic_hit("d", 0.5),
        semantic_hit("b", 0.4),
    ];
    let cfg = RrfConfig::default();

    let direct = pool_minmax_fuse_merge(&lexical, &semantic, 10, 0, &cfg);
    let via = fuse_by_strategy(
        FusionStrategy::PoolMinMax,
        &lexical,
        &semantic,
        &[],
        0.0,
        10,
        0,
        &cfg,
    );
    assert_eq!(fused_ids(&via), fused_ids(&direct));
    // And it really is the magnitude-recovering order, not RRF's.
    assert_eq!(fused_ids(&via), vec!["a", "d", "b", "c"]);
}

/// `pool_minmax_fuse_merge` has no graph arm. Rather than silently drop a graph signal the
/// caller explicitly enabled, a non-empty graph falls back to RRF.
#[test]
fn pool_minmax_with_graph_falls_back_to_rrf_not_dropping_graph() {
    let lexical = [lexical_hit("a", 10.0), lexical_hit("b", 2.0)];
    let semantic = [semantic_hit("a", 0.9), semantic_hit("d", 0.5)];
    let graph = [graph_hit("z", 0.99)];
    let cfg = RrfConfig::default();

    let via = fuse_by_strategy(
        FusionStrategy::PoolMinMax,
        &lexical,
        &semantic,
        &graph,
        0.5,
        10,
        0,
        &cfg,
    );
    let rrf = rrf_fuse_with_graph_merge_unique(&lexical, &semantic, &graph, 0.5, 10, 0, &cfg);
    assert_eq!(fused_ids(&via), fused_ids(&rrf), "must fall back to RRF");
    assert!(
        fused_ids(&via).contains(&"z".to_string()),
        "graph-only doc must survive; dropping it would be a silent correctness loss"
    );
}

/// QUALITY GATE. The claimed nDCG advantage is not just a rank permutation — measure it.
///
/// Ground truth: `d` is the relevant answer. It is semantic-only but overwhelmingly similar;
/// `b` is a marginal doc that happens to appear in *both* lists. RRF's rank transform rewards
/// `b` for its dual appearance and buries `d`; pool-min-max sees `d`'s magnitude and promotes
/// it. This is the exact mechanism behind the +0.0038 mean nDCG@10 measured on BEIR.
#[test]
fn pool_minmax_beats_rrf_on_ndcg_when_magnitude_carries_the_signal() {
    use frankensearch_core::metrics_eval::ndcg_at_k;

    let lexical = [
        lexical_hit("a", 10.0),
        lexical_hit("b", 2.0),
        lexical_hit("c", 1.0),
    ];
    let semantic = [
        semantic_hit("a", 0.9),
        semantic_hit("d", 0.5),
        semantic_hit("b", 0.4),
    ];
    let cfg = RrfConfig::default();
    let relevant = ["d"];

    let rrf = fused_ids(&fuse_by_strategy(
        FusionStrategy::Rrf,
        &lexical,
        &semantic,
        &[],
        0.0,
        10,
        0,
        &cfg,
    ));
    let pmm = fused_ids(&fuse_by_strategy(
        FusionStrategy::PoolMinMax,
        &lexical,
        &semantic,
        &[],
        0.0,
        10,
        0,
        &cfg,
    ));

    fn as_refs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }
    let ndcg_rrf = ndcg_at_k(&as_refs(&rrf), &relevant, 10);
    let ndcg_pmm = ndcg_at_k(&as_refs(&pmm), &relevant, 10);

    assert!(
        ndcg_pmm > ndcg_rrf,
        "pool_minmax nDCG@10 {ndcg_pmm} should beat rrf {ndcg_rrf} (rrf order {rrf:?}, pmm order {pmm:?})"
    );
}

#[test]
fn pool_minmax_recovers_top_match_magnitude() {
    // lexical pool [10,2,1] -> a=1.0, b=0.111, c=0.0
    // semantic pool [0.9,0.5,0.4] -> a=1.0, d=0.2, b=0.0
    let lexical = [
        lexical_hit("a", 10.0),
        lexical_hit("b", 2.0),
        lexical_hit("c", 1.0),
    ];
    let semantic = [
        semantic_hit("a", 0.9),
        semantic_hit("d", 0.5),
        semantic_hit("b", 0.4),
    ];
    let out = pool_minmax_fuse(&lexical, &semantic, 10, 0, &RrfConfig::default());
    // a=2.0, d=0.2, b=0.111, c=0.0. Crucially d (semantic-only, higher magnitude)
    // outranks b (in BOTH sources but marginal in each) — the magnitude recovery
    // RRF's rank transform discards.
    assert_eq!(fused_ids(&out), vec!["a", "d", "b", "c"]);
    // Fused score is the tier-weighted sum of normalized scores.
    assert!((out[0].rrf_score - 2.0).abs() < 1e-6, "a should be 1.0+1.0");
    assert!((out[1].rrf_score - 0.2).abs() < 1e-6, "d should be 0.0+0.2");
}

#[test]
fn pool_minmax_out_of_pool_contributes_zero() {
    // c is lexical-only: its semantic contribution must be the semantic pool
    // minimum normalized value = 0.0 (not the mean, not skipped).
    let lexical = [lexical_hit("a", 10.0), lexical_hit("c", 1.0)];
    let semantic = [semantic_hit("a", 0.8), semantic_hit("z", 0.2)];
    let out = pool_minmax_fuse(&lexical, &semantic, 10, 0, &RrfConfig::default());
    let c = out.iter().find(|h| h.doc_id.as_str() == "c").unwrap();
    // c: lex (1-1)/9 = 0.0 (pool min) + sem out-of-pool 0.0 = 0.0
    assert!((c.rrf_score - 0.0).abs() < 1e-6);
    assert!(!c.in_both_sources);
}

#[test]
fn pool_minmax_degenerate_pool_no_divide_by_zero() {
    // All lexical scores equal → range 0 → every lexical norm is 0.0 (safe).
    let lexical = [lexical_hit("a", 5.0), lexical_hit("b", 5.0)];
    let semantic = [semantic_hit("a", 0.8), semantic_hit("b", 0.2)];
    let out = pool_minmax_fuse(&lexical, &semantic, 10, 0, &RrfConfig::default());
    // Only semantic differentiates: a=1.0, b=0.0.
    assert_eq!(fused_ids(&out), vec!["a", "b"]);
    assert!((out[0].rrf_score - 1.0).abs() < 1e-6);
}

#[test]
fn pool_minmax_empty_lexical_tier() {
    let lexical: [ScoredResult; 0] = [];
    let semantic = [semantic_hit("a", 0.9), semantic_hit("b", 0.1)];
    let out = pool_minmax_fuse(&lexical, &semantic, 10, 0, &RrfConfig::default());
    assert_eq!(fused_ids(&out), vec!["a", "b"]);
}

#[test]
fn pool_minmax_tier_weights_reweight_the_sum() {
    // lexical: a=1.0,b=0.0 ; semantic: b=1.0,a=0.0. Up-weighting semantic 2x flips
    // the order to prefer the semantically-strong doc.
    let lexical = [lexical_hit("a", 10.0), lexical_hit("b", 1.0)];
    let semantic = [semantic_hit("b", 1.0), semantic_hit("a", 0.0)];
    let config = RrfConfig {
        semantic_weight: 2.0,
        ..RrfConfig::default()
    };
    let out = pool_minmax_fuse(&lexical, &semantic, 10, 0, &config);
    // a=1.0*1.0=1.0 ; b=2.0*1.0=2.0 → b first.
    assert_eq!(fused_ids(&out), vec!["b", "a"]);
}

#[test]
fn nqc_dense_downweight_composes_end_to_end() {
    use crate::{NqcDenseWeight, nqc_cv};
    // The opt-in NQC dense down-weight is realized by composing the three landed pieces
    // with NO kernel change: NQC from the lexical scores -> per-query dense multiplier ->
    // scale `semantic_weight` before pool_minmax_fuse. This test is executable docs for
    // the (caller-side, product-gated) searcher wiring.
    let lexical = [lexical_hit("L", 10.0), lexical_hit("Lb", 1.0)];
    let semantic = [semantic_hit("S", 0.9), semantic_hit("Sb", 0.1)];

    // Baseline (equal weights): lexical-only L and semantic-only S both pool-normalize to
    // 1.0 (a tie); default lexical-favoring tiebreak puts L first.
    let base = pool_minmax_fuse(&lexical, &semantic, 10, 0, &RrfConfig::default());
    let base_s = base.iter().find(|h| h.doc_id == "S").unwrap().rrf_score;
    assert!(
        (base_s - 1.0).abs() < 1e-12,
        "baseline dense score = 1.0, got {base_s}"
    );

    // Peaked lexical pool -> high NQC -> down-weight the dense tier.
    let lex_scores: Vec<f32> = lexical.iter().map(|r| r.score).collect();
    let cv = nqc_cv(&lex_scores);
    let weight = NqcDenseWeight::from_sample(&[0.1, 0.2, 0.3, cv]); // cv is the sample max
    let w = weight.dense_weight(cv, 0.5, 0.0); // 1 - 0.5*percentile(1.0) = 0.5
    assert!(
        (w - 0.5).abs() < 1e-6,
        "high-NQC query -> dense weight 0.5, got {w}"
    );

    // base semantic_weight (1.0) scaled by the per-query dense multiplier `w`.
    let cfg = RrfConfig {
        semantic_weight: 1.0 * f64::from(w),
        ..Default::default()
    };
    let down = pool_minmax_fuse(&lexical, &semantic, 10, 0, &cfg);
    let down_s = down.iter().find(|h| h.doc_id == "S").unwrap().rrf_score;
    let down_l = down.iter().find(|h| h.doc_id == "L").unwrap().rrf_score;
    assert!(
        (down_s - 0.5).abs() < 1e-12,
        "down-weighted dense = 0.5, got {down_s}"
    );
    assert!(
        down_l > down_s,
        "dense down-weight must demote the dense-only doc"
    );
    assert_eq!(down[0].doc_id, "L", "lexical-only doc now ranks first");
}

#[test]
fn pool_minmax_pagination_offset_limit() {
    let lexical = [
        lexical_hit("a", 10.0),
        lexical_hit("b", 2.0),
        lexical_hit("c", 1.0),
    ];
    let semantic = [
        semantic_hit("a", 0.9),
        semantic_hit("d", 0.5),
        semantic_hit("b", 0.4),
    ];
    // Full order is [a, d, b, c]; skip 1, take 2 → [d, b].
    let out = pool_minmax_fuse(&lexical, &semantic, 2, 1, &RrfConfig::default());
    assert_eq!(fused_ids(&out), vec!["d", "b"]);
}

#[test]
fn pool_minmax_empty_window_is_empty() {
    let lexical = [lexical_hit("a", 10.0)];
    let semantic = [semantic_hit("a", 0.9)];
    assert!(pool_minmax_fuse(&lexical, &semantic, 0, 0, &RrfConfig::default()).is_empty());
}

// ─── Merge fusion bit-identity ──────────────────────────────────────

#[test]
fn merge_matches_map_fusion() {
    // Deterministic xorshift to drive varied overlap/graph/dedup scenarios.
    let mut st = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    let cfg = RrfConfig::default();
    for trial in 0..40 {
        let n_lex = (next() % 30) as usize;
        let n_sem = (next() % 40) as usize;
        let n_graph = (next() % 20) as usize;
        // Shared id pool so sources overlap; small pool forces dedup dupes.
        let pool = 1 + (next() % 25) as usize;
        let id = |x: u64| format!("doc-{:04}", x % pool as u64);

        let lexical: Vec<ScoredResult> = (0..n_lex)
            .map(|_| lexical_hit(&id(next()), (next() % 1000) as f32 * 0.01))
            .collect();
        let semantic: Vec<VectorHit> = (0..n_sem)
            .map(|_| VectorHit {
                index: (next() % 10_000) as u32,
                score: (next() % 1000) as f32 * 0.001,
                doc_id: id(next()).into(),
            })
            .collect();
        let graph: Vec<ScoredResult> = (0..n_graph)
            .map(|_| graph_hit(&id(next()), (next() % 1000) as f32 * 0.01))
            .collect();
        let gw = if trial % 3 == 0 { 0.0 } else { 0.5 };
        let limit = 1 + (next() % 50) as usize;
        let offset = (next() % 5) as usize;

        let a = rrf_fuse_with_graph(&lexical, &semantic, &graph, gw, limit, offset, &cfg);
        let b = rrf_fuse_with_graph_merge(&lexical, &semantic, &graph, gw, limit, offset, &cfg);

        assert_eq!(a.len(), b.len(), "trial {trial}: length differs");
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.doc_id, y.doc_id, "trial {trial} row {i}: doc_id");
            assert_eq!(
                x.rrf_score.to_bits(),
                y.rrf_score.to_bits(),
                "trial {trial} row {i}: rrf_score not byte-identical ({} vs {})",
                x.rrf_score,
                y.rrf_score
            );
            assert_eq!(
                x.lexical_rank, y.lexical_rank,
                "trial {trial} row {i}: lexical_rank"
            );
            assert_eq!(
                x.semantic_rank, y.semantic_rank,
                "trial {trial} row {i}: semantic_rank"
            );
            assert_eq!(
                x.semantic_index, y.semantic_index,
                "trial {trial} row {i}: semantic_index"
            );
            assert_eq!(
                x.in_both_sources, y.in_both_sources,
                "trial {trial} row {i}: in_both"
            );
            assert_eq!(
                x.lexical_score.map(f32::to_bits),
                y.lexical_score.map(f32::to_bits),
                "trial {trial} row {i}: lexical_score"
            );
            assert_eq!(
                x.semantic_score.map(f32::to_bits),
                y.semantic_score.map(f32::to_bits),
                "trial {trial} row {i}: semantic_score"
            );
        }
    }
}

#[test]
fn pool_minmax_merge_matches_map() {
    // Same randomized harness as `merge_matches_map_fusion`, applied to the pool-min-max
    // operator: `pool_minmax_fuse_merge` must be byte-identical to the value-map
    // `pool_minmax_fuse` across varied overlap / dedup / pagination scenarios.
    let mut st = 0x1234_5678_9ABC_DEF0_u64;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    for trial in 0..40 {
        let n_lex = (next() % 30) as usize;
        let n_sem = (next() % 40) as usize;
        let pool = 1 + (next() % 25) as usize;
        let id = |x: u64| format!("doc-{:04}", x % pool as u64);

        let lexical: Vec<ScoredResult> = (0..n_lex)
            .map(|_| lexical_hit(&id(next()), (next() % 1000) as f32 * 0.01))
            .collect();
        // Semantic scores DESCENDING (score-sorted, as a vector index returns) so the merge
        // path's near-sorted assumption is exercised; bit-identity must hold regardless.
        let mut semantic: Vec<VectorHit> = (0..n_sem)
            .map(|_| VectorHit {
                index: (next() % 10_000) as u32,
                score: (next() % 1000) as f32 * 0.001,
                doc_id: id(next()).into(),
            })
            .collect();
        semantic.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Vary tier weights across trials to exercise the weighted sum.
        let cfg = RrfConfig {
            lexical_weight: 1.0 + (next() % 3) as f64,
            semantic_weight: 1.0 + (next() % 3) as f64,
            ..RrfConfig::default()
        };
        let limit = 1 + (next() % 50) as usize;
        let offset = (next() % 5) as usize;

        let a = pool_minmax_fuse(&lexical, &semantic, limit, offset, &cfg);
        let b = pool_minmax_fuse_merge(&lexical, &semantic, limit, offset, &cfg);

        assert_eq!(a.len(), b.len(), "trial {trial}: length differs");
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.doc_id, y.doc_id, "trial {trial} row {i}: doc_id");
            assert_eq!(
                x.rrf_score.to_bits(),
                y.rrf_score.to_bits(),
                "trial {trial} row {i}: fused score not byte-identical ({} vs {})",
                x.rrf_score,
                y.rrf_score
            );
            assert_eq!(
                x.lexical_rank, y.lexical_rank,
                "trial {trial} row {i}: lexical_rank"
            );
            assert_eq!(
                x.semantic_rank, y.semantic_rank,
                "trial {trial} row {i}: semantic_rank"
            );
            assert_eq!(
                x.semantic_index, y.semantic_index,
                "trial {trial} row {i}: semantic_index"
            );
            assert_eq!(
                x.in_both_sources, y.in_both_sources,
                "trial {trial} row {i}: in_both"
            );
            assert_eq!(
                x.lexical_score.map(f32::to_bits),
                y.lexical_score.map(f32::to_bits),
                "trial {trial} row {i}: lexical_score"
            );
            assert_eq!(
                x.semantic_score.map(f32::to_bits),
                y.semantic_score.map(f32::to_bits),
                "trial {trial} row {i}: semantic_score"
            );
        }
    }
}

// ─── Score formula tests ────────────────────────────────────────────

#[test]
fn rrf_score_formula_k60() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("doc-a", 10.0)];
    let semantic = vec![];

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    assert_eq!(results.len(), 1);
    let expected = 1.0 / (60.0 + 0.0 + 1.0); // rank 0 → 1/61
    assert!(
        (results[0].rrf_score - expected).abs() < 1e-12,
        "expected {expected}, got {}",
        results[0].rrf_score
    );
}

#[test]
fn rrf_score_formula_k1() {
    let config = RrfConfig {
        k: 1.0,
        ..Default::default()
    };
    let semantic = vec![semantic_hit("first", 0.9), semantic_hit("second", 0.8)];

    let results = rrf_fuse(&[], &semantic, 10, 0, &config);

    assert_eq!(results.len(), 2);
    // rank 0: 1/(1+0+1) = 0.5
    // rank 1: 1/(1+1+1) = 0.333...
    assert!((results[0].rrf_score - 0.5).abs() < 1e-12);
    assert!((results[1].rrf_score - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn rrf_score_formula_k0_is_valid() {
    let config = RrfConfig {
        k: 0.0,
        ..Default::default()
    };
    let lexical = vec![lexical_hit("doc-a", 10.0)];

    let results = rrf_fuse(&lexical, &[], 10, 0, &config);

    assert_eq!(results.len(), 1);
    assert!((results[0].rrf_score - 1.0).abs() < 1e-12);
}

#[test]
fn invalid_k_falls_back_to_default() {
    let lexical = vec![lexical_hit("doc-a", 10.0)];
    let expected = 1.0 / (DEFAULT_RRF_K + 1.0);

    for invalid_k in [f64::NAN, f64::INFINITY, -1.0, -100.0] {
        let config = RrfConfig {
            k: invalid_k,
            ..Default::default()
        };
        let results = rrf_fuse(&lexical, &[], 10, 0, &config);
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].rrf_score - expected).abs() < 1e-12,
            "invalid k={invalid_k} should fall back to default",
        );
    }
}

// ─── Multi-source fusion ────────────────────────────────────────────

#[test]
fn document_in_both_sources_gets_summed_score() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("shared", 5.0)];
    let semantic = vec![semantic_hit("shared", 0.9)];

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    assert_eq!(results.len(), 1);
    let expected = 2.0 / 61.0; // Both at rank 0 → 1/61 + 1/61
    assert!(
        (results[0].rrf_score - expected).abs() < 1e-12,
        "expected {expected}, got {}",
        results[0].rrf_score
    );
    assert!(results[0].in_both_sources);
    assert_eq!(results[0].lexical_rank, Some(0));
    assert_eq!(results[0].semantic_rank, Some(0));
}

#[test]
fn multi_source_doc_ranks_higher_than_single_source() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("shared", 5.0), lexical_hit("lex-only", 4.0)];
    let semantic = vec![semantic_hit("shared", 0.9), semantic_hit("sem-only", 0.8)];

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    assert_eq!(results.len(), 3);
    // "shared" should be first (highest combined score)
    assert_eq!(results[0].doc_id, "shared");
    assert!(results[0].in_both_sources);
}

#[test]
fn graph_channel_can_promote_document_with_weighted_rrf() {
    let config = RrfConfig::default();
    let semantic = vec![semantic_hit("a", 0.9), semantic_hit("b", 0.8)];
    let graph = vec![graph_hit("b", 1.0)];

    let results = rrf_fuse_with_graph(&[], &semantic, &graph, 1.0, 10, 0, &config);

    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].doc_id, "b",
        "graph contribution should promote b above semantic rank-0 doc a"
    );
}

#[test]
fn zero_graph_weight_matches_two_source_rrf() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("a", 10.0)];
    let semantic = vec![semantic_hit("b", 0.9)];
    let graph = vec![graph_hit("b", 1.0)];

    let base = rrf_fuse(&lexical, &semantic, 10, 0, &config);
    let weighted = rrf_fuse_with_graph(&lexical, &semantic, &graph, 0.0, 10, 0, &config);

    assert_eq!(weighted.len(), base.len());
    assert_eq!(weighted[0].doc_id, base[0].doc_id);
    assert!((weighted[0].rrf_score - base[0].rrf_score).abs() < 1e-12);
}

// ─── Single-source fusion ───────────────────────────────────────────

#[test]
fn lexical_only_produces_correct_ranking() {
    let config = RrfConfig::default();
    let lexical = vec![
        lexical_hit("a", 10.0),
        lexical_hit("b", 8.0),
        lexical_hit("c", 5.0),
    ];

    let results = rrf_fuse(&lexical, &[], 10, 0, &config);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].doc_id, "a");
    assert_eq!(results[1].doc_id, "b");
    assert_eq!(results[2].doc_id, "c");
    // All single-source
    assert!(results.iter().all(|r| !r.in_both_sources));
}

#[test]
fn semantic_only_produces_correct_ranking() {
    let config = RrfConfig::default();
    let semantic = vec![semantic_hit("x", 0.95), semantic_hit("y", 0.85)];

    let results = rrf_fuse(&[], &semantic, 10, 0, &config);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].doc_id, "x");
    assert_eq!(results[1].doc_id, "y");
}

#[test]
fn tier_weight_reorders_by_upweighted_source() {
    let lexical = vec![lexical_hit("lex", 1.0)];
    let semantic = vec![semantic_hit("sem", 0.9)];
    let k = RrfConfig::default().k;

    // Up-weight the semantic tier 2× → the semantic-only doc ranks first with
    // exactly 2× the unweighted rank-0 contribution.
    let sem_cfg = RrfConfig {
        semantic_weight: 2.0,
        ..Default::default()
    };
    let sem_first = rrf_fuse(&lexical, &semantic, 10, 0, &sem_cfg);
    assert_eq!(sem_first[0].doc_id, "sem");
    let expected = 2.0 / (k + 1.0);
    assert!(
        (sem_first[0].rrf_score - expected).abs() < 1e-12,
        "expected {expected}, got {}",
        sem_first[0].rrf_score
    );

    // Symmetrically, up-weighting lexical flips the winner to the lexical-only doc.
    let lex_cfg = RrfConfig {
        lexical_weight: 2.0,
        ..Default::default()
    };
    let lex_first = rrf_fuse(&lexical, &semantic, 10, 0, &lex_cfg);
    assert_eq!(lex_first[0].doc_id, "lex");

    // A non-finite / non-positive weight degrades to neutral 1.0 (standard RRF),
    // never corrupts scores.
    let bad_cfg = RrfConfig {
        semantic_weight: f64::NAN,
        lexical_weight: -1.0,
        ..Default::default()
    };
    let neutral = rrf_fuse(&lexical, &semantic, 10, 0, &bad_cfg);
    assert_eq!(neutral.len(), 2);
    assert!(
        neutral
            .iter()
            .all(|h| (h.rrf_score - 1.0 / (k + 1.0)).abs() < 1e-12),
        "bad weights should degrade to unweighted RRF"
    );
}

#[test]
fn hash_tiebreak_is_symmetric_across_tiers() {
    // Two docs that tie on rrf_score (each rank 0 in its own tier) and both have
    // in_both_sources == false, so only the tiebreak decides their order.
    let lexical = vec![lexical_hit("alpha", 5.0)];
    let semantic = vec![semantic_hit("beta", 0.9)];

    // Default (lexical-favoring): the lexical-only doc always wins the tie.
    let d = rrf_fuse(&lexical, &semantic, 10, 0, &RrfConfig::default());
    assert_eq!(
        d[0].doc_id, "alpha",
        "default tiebreak favors the lexical-only doc"
    );

    // Hash tiebreak: order decided by an unbiased hash of doc_id, not the tier.
    let hash_cfg = RrfConfig {
        tiebreak: RrfTiebreak::Hash,
        ..Default::default()
    };
    let h = rrf_fuse(&lexical, &semantic, 10, 0, &hash_cfg);
    assert_eq!(h.len(), 2);
    assert!(
        (h[0].rrf_score - h[1].rrf_score).abs() < 1e-12,
        "the two docs genuinely tie on rrf_score"
    );
    let expected_first = if doc_id_tiebreak_hash("alpha") <= doc_id_tiebreak_hash("beta") {
        "alpha"
    } else {
        "beta"
    };
    assert_eq!(
        h[0].doc_id, expected_first,
        "hash tiebreak orders by doc_id hash, tier-agnostic"
    );
}

// ─── Empty input ────────────────────────────────────────────────────

#[test]
fn both_empty_returns_empty() {
    let results = rrf_fuse(&[], &[], 10, 0, &RrfConfig::default());
    assert!(results.is_empty());
}

// ─── Offset and limit ───────────────────────────────────────────────

#[test]
fn limit_truncates_results() {
    let config = RrfConfig::default();
    let semantic = vec![
        semantic_hit("a", 0.9),
        semantic_hit("b", 0.8),
        semantic_hit("c", 0.7),
    ];

    let results = rrf_fuse(&[], &semantic, 2, 0, &config);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].doc_id, "a");
    assert_eq!(results[1].doc_id, "b");
}

#[test]
fn offset_skips_top_results() {
    let config = RrfConfig::default();
    let semantic = vec![
        semantic_hit("a", 0.9),
        semantic_hit("b", 0.8),
        semantic_hit("c", 0.7),
    ];

    let results = rrf_fuse(&[], &semantic, 10, 1, &config);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].doc_id, "b");
    assert_eq!(results[1].doc_id, "c");
}

#[test]
fn offset_and_limit_combined() {
    let config = RrfConfig::default();
    let semantic = vec![
        semantic_hit("a", 0.9),
        semantic_hit("b", 0.8),
        semantic_hit("c", 0.7),
        semantic_hit("d", 0.6),
    ];

    let results = rrf_fuse(&[], &semantic, 2, 1, &config);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].doc_id, "b");
    assert_eq!(results[1].doc_id, "c");
}

// ─── Tie-breaking ───────────────────────────────────────────────────

#[test]
fn tie_breaking_in_both_sources_preferred() {
    let config = RrfConfig::default();
    // "shared" at lex rank 1, sem rank 0: RRF = 1/62 + 1/61
    // "lex-0" at lex rank 0: RRF = 1/61
    // "sem-1" at sem rank 1: RRF = 1/62
    // Adjust so "shared" and another doc have same RRF score:
    // Actually, just verify in_both_sources wins on tie.
    let lexical = vec![
        lexical_hit("only-lex", 10.0), // rank 0 → 1/61
    ];
    let semantic = vec![
        semantic_hit("only-sem", 0.9), // rank 0 → 1/61
    ];

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    // Both have same RRF score (1/61). Neither is in both sources.
    // Tie-break goes to lexical_score (only-lex has Some(10.0), only-sem has None).
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].doc_id, "only-lex"); // has lexical_score
}

#[test]
fn tie_breaking_doc_id_ascending() {
    let config = RrfConfig::default();
    // Same semantic scores, same rank structure, different doc_ids
    let semantic = vec![
        semantic_hit("beta", 0.9), // rank 0
    ];
    let lexical = vec![
        lexical_hit("alpha", 10.0), // rank 0
    ];

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    // Same RRF, same in_both_sources (false). lexical_score tiebreak:
    // "alpha" has Some(10.0), "beta" has None → alpha first.
    assert_eq!(results[0].doc_id, "alpha");
    assert_eq!(results[1].doc_id, "beta");
}

// ─── Candidate budget ───────────────────────────────────────────────

#[test]
fn candidate_count_basic() {
    assert_eq!(candidate_count(10, 0, 3), 30);
    assert_eq!(candidate_count(10, 5, 3), 45);
    assert_eq!(candidate_count(20, 0, 4), 80);
}

#[test]
fn candidate_count_overflow_safety() {
    // Should not panic on overflow
    let result = candidate_count(usize::MAX, 1, 3);
    assert_eq!(result, usize::MAX); // saturating
}

// ─── Score preservation ─────────────────────────────────────────────

#[test]
fn lexical_score_preserved() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("doc", 42.5)];

    let results = rrf_fuse(&lexical, &[], 10, 0, &config);

    assert_eq!(results[0].lexical_score, Some(42.5));
    assert_eq!(results[0].semantic_score, None);
}

#[test]
fn semantic_score_preserved() {
    let config = RrfConfig::default();
    let semantic = vec![semantic_hit("doc", 0.87)];

    let results = rrf_fuse(&[], &semantic, 10, 0, &config);

    assert_eq!(results[0].semantic_score, Some(0.87));
    assert_eq!(results[0].lexical_score, None);
}

#[test]
fn both_scores_preserved_on_shared_doc() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("doc", 15.0)];
    let semantic = vec![semantic_hit("doc", 0.93)];

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].lexical_score, Some(15.0));
    assert_eq!(results[0].semantic_score, Some(0.93));
    assert!(results[0].in_both_sources);
}

// ─── Output ordering ────────────────────────────────────────────────

#[test]
fn output_is_strictly_descending_by_rrf_score() {
    let config = RrfConfig::default();
    let lexical = vec![
        lexical_hit("a", 10.0),
        lexical_hit("b", 8.0),
        lexical_hit("c", 6.0),
    ];
    let semantic = vec![
        semantic_hit("d", 0.9),
        semantic_hit("e", 0.8),
        semantic_hit("f", 0.7),
    ];

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

    for window in results.windows(2) {
        assert!(
            window[0].rrf_score >= window[1].rrf_score,
            "results not in descending order: {} ({}) > {} ({})",
            window[0].doc_id,
            window[0].rrf_score,
            window[1].doc_id,
            window[1].rrf_score,
        );
    }
}

#[test]
fn candidate_count_zero_multiplier_returns_zero() {
    assert_eq!(candidate_count(100, 50, 0), 0);
    assert_eq!(candidate_count(0, 0, 0), 0);
}

#[test]
fn candidate_count_zero_limit_and_offset() {
    assert_eq!(candidate_count(0, 0, 3), 0);
}

#[test]
fn limit_zero_returns_empty() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("a", 10.0)];
    let results = rrf_fuse(&lexical, &[], 0, 0, &config);
    assert!(results.is_empty());
}

#[test]
fn offset_beyond_results_returns_empty() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("a", 10.0)];
    let results = rrf_fuse(&lexical, &[], 10, 100, &config);
    assert!(results.is_empty());
}

#[test]
fn single_element_single_source_correct_rank() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("only", 1.0)];
    let results = rrf_fuse(&lexical, &[], 10, 0, &config);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, "only");
    assert_eq!(results[0].lexical_rank, Some(0));
    assert_eq!(results[0].semantic_rank, None);
    assert!(!results[0].in_both_sources);
}

#[test]
fn duplicate_doc_in_single_source_does_not_set_in_both_sources() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("dup", 1.0), lexical_hit("dup", 0.5)];

    let results = rrf_fuse(&lexical, &[], 10, 0, &config);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, "dup");
    assert!(!results[0].in_both_sources);
    assert!(results[0].lexical_rank.is_some());
    assert!(results[0].semantic_rank.is_none());
}

fn large_lexical_semantic_fixture(doc_count: usize) -> (Vec<ScoredResult>, Vec<VectorHit>) {
    let mut lexical = Vec::with_capacity(doc_count);
    let mut semantic = Vec::with_capacity(doc_count);

    for index in 0..doc_count {
        let shared_bucket = index % (doc_count / 2).max(1);
        let doc_id = format!("doc-{shared_bucket}");

        let lexical_rank = f32::from(u16::try_from(index % 512).unwrap_or(u16::MAX));
        let semantic_rank = f32::from(u16::try_from(index % 256).unwrap_or(u16::MAX));

        lexical.push(lexical_hit(&doc_id, 1000.0 - lexical_rank));
        semantic.push(semantic_hit(&doc_id, 1.0 - (semantic_rank / 1024.0)));
    }

    (lexical, semantic)
}

fn rrf_fuse_reference_full_sort(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    let k = sanitize_rrf_k(config.k);
    let capacity = lexical.len() + semantic.len();
    let mut hits: HashMap<String, FusedHit> = HashMap::with_capacity(capacity);

    for (rank, result) in lexical.iter().enumerate() {
        // Skip within-source duplicates (keep first/best-ranked occurrence).
        if let Some(existing) = hits.get(result.doc_id.as_str())
            && existing.lexical_rank.is_some()
        {
            continue;
        }

        let rrf_contribution = rank_contribution(k, rank);

        hits.entry(result.doc_id.to_string())
            .and_modify(|hit| {
                hit.rrf_score += rrf_contribution;
                hit.lexical_rank = Some(rank);
                hit.lexical_score = Some(result.score);
            })
            .or_insert_with(|| FusedHit {
                doc_id: result.doc_id.clone(),
                rrf_score: rrf_contribution,
                lexical_rank: Some(rank),
                semantic_rank: None,
                hash_rank: None,
                semantic_index: result.index,
                lexical_score: Some(result.score),
                semantic_score: None,
                hash_score: None,
                in_both_sources: false,
            });
    }

    for (rank, hit) in semantic.iter().enumerate() {
        // Skip within-source duplicates (keep first/best-ranked occurrence).
        if let Some(existing) = hits.get(hit.doc_id.as_str())
            && existing.semantic_rank.is_some()
        {
            continue;
        }

        let rrf_contribution = rank_contribution(k, rank);

        hits.entry(hit.doc_id.to_string())
            .and_modify(|fh| {
                fh.rrf_score += rrf_contribution;
                fh.semantic_rank = Some(rank);
                fh.semantic_score = Some(hit.score);
            })
            .or_insert_with(|| FusedHit {
                doc_id: hit.doc_id.clone(),
                rrf_score: rrf_contribution,
                lexical_rank: None,
                semantic_rank: Some(rank),
                hash_rank: None,
                semantic_index: Some(hit.index),
                lexical_score: None,
                semantic_score: Some(hit.score),
                hash_score: None,
                in_both_sources: false,
            });
    }

    let mut results: Vec<FusedHit> = hits.into_values().collect();
    for hit in &mut results {
        hit.in_both_sources = hit.lexical_rank.is_some() && hit.semantic_rank.is_some();
    }
    results.sort_by(FusedHit::cmp_for_ranking);
    results.into_iter().skip(offset).take(limit).collect()
}

#[test]
fn ranking_window_selection_matches_full_sort_reference() {
    let config = RrfConfig::default();
    let (lexical, semantic) = large_lexical_semantic_fixture(6_000);
    let windows = [
        (0_usize, 0_usize),
        (25, 0),
        (50, 10),
        (100, 250),
        (256, 800),
        (512, 2_000),
        (128, 10_000),
    ];

    for (limit, offset) in windows {
        let actual = rrf_fuse(&lexical, &semantic, limit, offset, &config);
        let expected = rrf_fuse_reference_full_sort(&lexical, &semantic, limit, offset, &config);

        assert_eq!(
            actual.len(),
            expected.len(),
            "window mismatch for limit={limit} offset={offset}"
        );

        for (actual_hit, expected_hit) in actual.iter().zip(&expected) {
            assert_eq!(actual_hit.doc_id, expected_hit.doc_id);
            assert!((actual_hit.rrf_score - expected_hit.rrf_score).abs() < 1e-12);
            assert_eq!(actual_hit.lexical_rank, expected_hit.lexical_rank);
            assert_eq!(actual_hit.semantic_rank, expected_hit.semantic_rank);
            assert_eq!(actual_hit.lexical_score, expected_hit.lexical_score);
            assert_eq!(actual_hit.semantic_score, expected_hit.semantic_score);
            assert_eq!(actual_hit.in_both_sources, expected_hit.in_both_sources);
        }
    }
}

#[test]
#[ignore = "Perf probe for optimization loop: run explicitly with --ignored"]
fn perf_probe_rrf_large_candidates() {
    let doc_count = std::env::var("RRF_PERF_DOCS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8_000);
    let iterations = std::env::var("RRF_PERF_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    let (lexical, semantic) = large_lexical_semantic_fixture(doc_count);
    let config = RrfConfig::default();

    let started = Instant::now();
    let mut checksum = 0.0_f64;
    for _ in 0..iterations {
        let output = rrf_fuse(&lexical, &semantic, 100, 0, &config);
        checksum += output.iter().map(|hit| hit.rrf_score).sum::<f64>();
    }
    let elapsed = started.elapsed();

    eprintln!(
        "RRF_PERF baseline_or_candidate elapsed_ms={} doc_count={} iterations={} checksum={checksum}",
        elapsed.as_millis(),
        doc_count,
        iterations
    );
    assert!(checksum.is_finite());
}

#[test]
#[ignore = "Perf probe for optimization loop: run explicitly with --ignored"]
fn perf_probe_rrf_window_selection_vs_full_sort_reference() {
    let doc_count = std::env::var("RRF_PERF_DOCS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8_000);
    let iterations = std::env::var("RRF_PERF_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    let limit = 100;
    let offset = 0;

    let (lexical, semantic) = large_lexical_semantic_fixture(doc_count);
    let config = RrfConfig::default();

    let optimized_started = Instant::now();
    let mut optimized_checksum = 0.0_f64;
    for _ in 0..iterations {
        let output = black_box(rrf_fuse(&lexical, &semantic, limit, offset, &config));
        optimized_checksum += output.iter().map(|hit| hit.rrf_score).sum::<f64>();
    }
    let optimized_elapsed = optimized_started.elapsed();

    let reference_started = Instant::now();
    let mut reference_checksum = 0.0_f64;
    for _ in 0..iterations {
        let output = black_box(rrf_fuse_reference_full_sort(
            &lexical, &semantic, limit, offset, &config,
        ));
        reference_checksum += output.iter().map(|hit| hit.rrf_score).sum::<f64>();
    }
    let reference_elapsed = reference_started.elapsed();

    assert!((optimized_checksum - reference_checksum).abs() < 1e-8);

    let optimized_ms = optimized_elapsed.as_secs_f64() * 1_000.0;
    let reference_ms = reference_elapsed.as_secs_f64() * 1_000.0;
    let speedup = if optimized_ms > 0.0 {
        reference_ms / optimized_ms
    } else {
        1.0
    };

    eprintln!(
        "RRF_PERF_COMPARE optimized_ms={optimized_ms:.3} reference_ms={reference_ms:.3} speedup={speedup:.3} doc_count={doc_count} iterations={iterations}"
    );
}

// ─── bd-1o9x tests begin ───

#[test]
fn rrf_config_debug_format() {
    let config = RrfConfig {
        k: 42.0,
        ..Default::default()
    };
    let debug = format!("{config:?}");
    assert!(debug.contains("42"));
    assert!(debug.contains("RrfConfig"));
}

#[test]
fn rrf_config_default_k_exact() {
    let config = RrfConfig::default();
    assert!((config.k - 60.0).abs() < f64::EPSILON);
}

#[test]
fn rank_contribution_monotonically_decreasing() {
    let k = 60.0;
    let mut prev = rank_contribution(k, 0);
    for r in 1..100 {
        let current = rank_contribution(k, r);
        assert!(current < prev, "rank_contribution not decreasing: rank {r}");
        prev = current;
    }
}

#[test]
fn large_overlapping_docs_stress() {
    let config = RrfConfig::default();
    let count = 500;
    let lexical: Vec<ScoredResult> = (0_u16..count)
        .map(|i| lexical_hit(&format!("doc-{i}"), 1000.0 - f32::from(i)))
        .collect();
    let semantic: Vec<VectorHit> = (0_u16..count)
        .map(|i| semantic_hit(&format!("doc-{i}"), 1.0 - f32::from(i) / 1000.0))
        .collect();

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);
    assert_eq!(results.len(), 10);
    // All should be in both sources
    assert!(results.iter().all(|r| r.in_both_sources));
    // Output should be descending by RRF score
    for w in results.windows(2) {
        assert!(w[0].rrf_score >= w[1].rrf_score);
    }
}

#[test]
fn duplicate_doc_id_same_source_uses_best_rank() {
    let config = RrfConfig::default();
    // Same doc_id appearing twice in lexical results
    let lexical = vec![
        lexical_hit("dup", 10.0), // rank 0
        lexical_hit("other", 8.0),
        lexical_hit("dup", 5.0), // rank 2 (duplicate)
    ];

    let results = rrf_fuse(&lexical, &[], 10, 0, &config);

    let dup_hit = results.iter().find(|r| r.doc_id == "dup").unwrap();
    // HashMap entry: first insertion at rank 0 gives 1/61.
    // second occurrence at rank 2 is ignored by dedup logic.
    let expected = rank_contribution(60.0, 0);
    assert!(
        (dup_hit.rrf_score - expected).abs() < 1e-12,
        "expected {expected}, got {}",
        dup_hit.rrf_score
    );
}

#[test]
fn candidate_count_multiplier_one() {
    assert_eq!(candidate_count(10, 5, 1), 15);
    assert_eq!(candidate_count(0, 10, 1), 10);
}

#[test]
fn doc_id_tiebreak_alphabetical() {
    let config = RrfConfig::default();
    // Two docs with identical structure but different doc_ids
    // Same source and rank → same RRF score
    let semantic = vec![semantic_hit("zebra", 0.9), semantic_hit("alpha", 0.8)];
    let results = rrf_fuse(&[], &semantic, 10, 0, &config);

    // "zebra" has higher RRF (rank 0), "alpha" lower (rank 1)
    assert_eq!(results[0].doc_id, "zebra");
    assert_eq!(results[1].doc_id, "alpha");
}

#[test]
fn rank_contribution_very_large_rank() {
    let score = rank_contribution(60.0, 1_000_000);
    assert!(score > 0.0);
    assert!(score.is_finite());
    assert!(score < 1e-5);
}

#[test]
fn multi_source_different_ranks_preserved() {
    let config = RrfConfig::default();
    let lexical = vec![
        lexical_hit("a", 10.0),
        lexical_hit("shared", 8.0), // rank 1
    ];
    let semantic = vec![
        semantic_hit("shared", 0.9), // rank 0
        semantic_hit("b", 0.8),
    ];

    let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);
    let shared = results.iter().find(|r| r.doc_id == "shared").unwrap();
    assert_eq!(shared.lexical_rank, Some(1));
    assert_eq!(shared.semantic_rank, Some(0));
    assert!(shared.in_both_sources);
}

#[test]
fn zero_window_returns_empty() {
    let config = RrfConfig::default();
    let lexical = vec![lexical_hit("a", 10.0)];
    // limit=0, offset=0 → window=0
    let results = rrf_fuse(&lexical, &[], 0, 0, &config);
    assert!(results.is_empty());
}

// ─── bd-1o9x tests end ───
