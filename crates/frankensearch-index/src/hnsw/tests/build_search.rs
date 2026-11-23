use std::cell::Cell;
use std::rc::Rc;

use frankensearch_core::SearchError;

use crate::{Quantization, VectorIndex};

use super::super::fingerprint::{validate_config, vector_component_close};
use super::super::search::{attest_or_rebuild_serial, should_use_parallel_insert};
use super::super::topology::validate_hnsw_topology;
use super::super::types::{
    HNSW_DEFAULT_EF_SEARCH, HNSW_DEFAULT_M, HNSW_DEFAULT_MAX_LAYER,
    HNSW_PARALLEL_INSERT_MIN_POINTS_PER_THREAD, HnswConfig, HnswIndex, HnswLoadDisposition,
};
use super::{GraphRepairProbe, normalized_vector, recall_at_k, temp_path, write_index};

#[test]
fn empty_index_returns_no_hits() {
    let path = temp_path("empty", "fsvi");
    let writer = VectorIndex::create_with_revision(&path, "hash", "test", 8, Quantization::F16)
        .expect("create writer");
    writer.finish().expect("finish");

    let index = VectorIndex::open(&path).expect("open index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    let hits = ann
        .knn_search(&normalized_vector(7, 8), 10, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert!(hits.is_empty());
}

#[test]
fn single_vector_round_trip() {
    let path = temp_path("single", "fsvi");
    let index = write_index(&path, &[normalized_vector(1, 32)]).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

    let hits = ann
        .knn_search(&normalized_vector(1, 32), 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-0000");
}

#[test]
fn small_serial_builds_preserve_topology_and_return_every_available_hit() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("four-thread test pool");
    pool.install(|| {
        for round in 0..32 {
            for count in [0_usize, 1, 2, 3, HNSW_DEFAULT_M, HNSW_DEFAULT_M + 1] {
                let doc_ids = (0..count)
                    .map(|index| format!("round-{round}-doc-{index}"))
                    .collect::<Vec<_>>();
                let vectors = (0..count)
                    .map(|index| normalized_vector(round * 100 + index, 16))
                    .collect::<Vec<_>>();
                let ann = HnswIndex::build_from_parts(
                    doc_ids,
                    vectors.clone(),
                    16,
                    HnswConfig::default(),
                )
                .expect("small graph build");
                validate_hnsw_topology(&ann.hnsw, count).expect("topology attestation");

                for query in &vectors {
                    let (hits, stats) = ann
                        .knn_search_with_stats(query, count, HNSW_DEFAULT_EF_SEARCH)
                        .expect("full small-graph search");
                    assert_eq!(hits.len(), count, "round={round}, count={count}");
                    assert_eq!(stats.fallback_reason, None);
                }
            }
        }
    });
}

#[test]
fn parallel_insert_threshold_scales_with_the_active_rayon_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("four-thread test pool");
    pool.install(|| {
        let threshold = 4 * HNSW_PARALLEL_INSERT_MIN_POINTS_PER_THREAD;
        // One point is seeded serially, so the decision is based on the
        // remaining insertion batch rather than total graph size.
        assert!(!should_use_parallel_insert(threshold));
        assert!(should_use_parallel_insert(threshold + 1));
    });
}

#[test]
fn failed_parallel_attestation_drops_graph_then_repairs_once_serially() {
    let live_graphs = Rc::new(Cell::new(0_usize));
    let rebuilds = Cell::new(0_usize);
    let initial = GraphRepairProbe::new("parallel-invalid", false, &live_graphs);

    let repaired = attest_or_rebuild_serial(
        initial,
        true,
        50_000,
        |graph| {
            graph
                .valid
                .then_some(())
                .ok_or_else(|| graph.label.to_owned())
        },
        || {
            assert_eq!(
                live_graphs.get(),
                0,
                "invalid graph must be dropped before allocating its replacement"
            );
            rebuilds.set(rebuilds.get().saturating_add(1));
            GraphRepairProbe::new("serial-valid", true, &live_graphs)
        },
    )
    .expect("serial repair");

    assert_eq!(repaired.label, "serial-valid");
    assert_eq!(rebuilds.get(), 1);
    assert_eq!(live_graphs.get(), 1);
    drop(repaired);
    assert_eq!(live_graphs.get(), 0);
}

#[test]
fn failed_serial_repair_reports_both_attestations_and_fails_closed() {
    let live_graphs = Rc::new(Cell::new(0_usize));
    let rebuilds = Cell::new(0_usize);
    let initial = GraphRepairProbe::new("parallel-invalid", false, &live_graphs);

    let error = attest_or_rebuild_serial(
        initial,
        true,
        50_000,
        |graph| {
            graph
                .valid
                .then_some(())
                .ok_or_else(|| graph.label.to_owned())
        },
        || {
            assert_eq!(live_graphs.get(), 0);
            rebuilds.set(rebuilds.get().saturating_add(1));
            GraphRepairProbe::new("serial-invalid", false, &live_graphs)
        },
    )
    .expect_err("both invalid graphs must fail closed");

    assert!(error.contains("parallel-invalid"), "{error}");
    assert!(error.contains("serial-invalid"), "{error}");
    assert_eq!(rebuilds.get(), 1);
    assert_eq!(live_graphs.get(), 0);

    let serial_only = GraphRepairProbe::new("serial-only-invalid", false, &live_graphs);
    let error = attest_or_rebuild_serial(
        serial_only,
        false,
        3,
        |graph| Err(graph.label.to_owned()),
        || panic!("a serial build must never recursively rebuild"),
    )
    .expect_err("serial construction failure");
    assert_eq!(error, "serial-only-invalid");
    assert_eq!(rebuilds.get(), 1);
    assert_eq!(live_graphs.get(), 0);
}

#[test]
fn two_vector_queries_return_both_directions_without_fallback() {
    let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    for _ in 0..128 {
        let ann = HnswIndex::build_from_parts(
            vec!["east".to_owned(), "north".to_owned()],
            vectors.clone(),
            2,
            HnswConfig::default(),
        )
        .expect("two-vector graph");
        for (expected_first, query) in vectors.iter().enumerate() {
            let (hits, stats) = ann
                .knn_search_with_stats(query, 2, HNSW_DEFAULT_EF_SEARCH)
                .expect("two-vector search");
            assert_eq!(hits.len(), 2);
            assert_eq!(
                usize::try_from(hits[0].index).expect("hit index"),
                expected_first
            );
            assert_eq!(stats.fallback_reason, None);
            assert!(stats.is_approximate);
        }
    }
}

#[test]
fn higher_ef_improves_or_matches_recall() {
    let fsvi_path = temp_path("ef", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..256).map(|i| normalized_vector(i, 96)).collect();
    let index = write_index(&fsvi_path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

    let mut low_total = 0.0_f64;
    let mut high_total = 0.0_f64;
    let mut count = 0_u32;
    for query_seed in (0..128).step_by(16) {
        let query = normalized_vector(query_seed, 96);
        let exact = index.search_top_k(&query, 10, None).expect("exact");
        let low = ann.knn_search(&query, 10, 10).expect("low ef");
        let high = ann.knn_search(&query, 10, 100).expect("high ef");
        low_total += recall_at_k(&low, &exact);
        high_total += recall_at_k(&high, &exact);
        count += 1;
    }

    let count_f = f64::from(count);
    assert!((high_total / count_f) >= (low_total / count_f));
}

#[test]
fn recall_against_bruteforce_is_high() {
    let fsvi_path = temp_path("recall", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..1_000).map(|i| normalized_vector(i, 384)).collect();
    let index = write_index(&fsvi_path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

    let mut total_recall = 0.0_f64;
    let mut query_count = 0_u32;
    for query_seed in (0..1_000).step_by(40) {
        let query = normalized_vector(query_seed, 384);
        let exact = index.search_top_k(&query, 10, None).expect("exact");
        let approx = ann
            .knn_search(&query, 10, HNSW_DEFAULT_EF_SEARCH)
            .expect("approx");
        total_recall += recall_at_k(&approx, &exact);
        query_count += 1;
    }

    let avg_recall = total_recall / f64::from(query_count);
    assert!(
        avg_recall >= 0.95,
        "expected avg recall >= 0.95, got {avg_recall:.4}"
    );
}

#[test]
fn certify_ef_search_wires_conformal_certificate_end_to_end() {
    // Real ANN index + real bruteforce feeding the conformal certificate.
    let fsvi_path = temp_path("certify", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..800).map(|i| normalized_vector(i, 384)).collect();
    let index = write_index(&fsvi_path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    // Calibration queries disjoint from... they're near corpus points, fine for wiring.
    let calibration: Vec<Vec<f32>> = (5..1_600)
        .step_by(64)
        .map(|s| normalized_vector(s, 384))
        .collect();
    let candidate_efs = [10usize, 40, 100, 200];

    // target=0.0 is always certified => cheapest ef, and the sweep must
    // short-circuit immediately (only the smallest ef is ever ANN-searched).
    let trivial = ann
        .certify_ef_search(&index, &calibration, &candidate_efs, 10, 0.0, 0.1)
        .expect("certify")
        .expect("some");
    assert!(trivial.chosen.meets_target);
    assert_eq!(
        trivial.chosen.ef_search, 10,
        "cheapest ef for a trivial target"
    );
    assert_eq!(
        trivial.sweep.len(),
        1,
        "short-circuits at the first certified ef"
    );

    // An unreachable target => no ef meets it, fall back to the best-certifiable
    // (largest ef here, since recall is non-decreasing in ef), full sweep measured,
    // and the certified bound is a real recall in [0, 1].
    let strict = ann
        .certify_ef_search(&index, &calibration, &candidate_efs, 10, 2.0, 0.1)
        .expect("certify")
        .expect("some");
    assert!(!strict.chosen.meets_target);
    assert_eq!(
        strict.sweep.len(),
        candidate_efs.len(),
        "measures all when none certifies"
    );
    assert!((0.0..=1.0).contains(&strict.chosen.certified_recall));
    // The best-certifiable fallback should be a high ef with a strong bound on
    // this tight synthetic corpus (sanity: ANN recovers most exact neighbours).
    assert!(
        strict.chosen.certified_recall > 0.3,
        "expected a meaningful certified recall, got {}",
        strict.chosen.certified_recall
    );
}

// ── Config validation edge cases ──────────────────────────────────

#[test]
fn validate_config_rejects_m_zero() {
    let config = HnswConfig {
        m: 0,
        ..HnswConfig::default()
    };
    let error = validate_config(config).unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_m"),
        "expected InvalidConfig for hnsw_m, got {error:?}"
    );
}

#[test]
fn validate_config_rejects_m_256_that_cannot_round_trip_through_u8() {
    let config = HnswConfig {
        m: 256,
        ..HnswConfig::default()
    };
    let error = validate_config(config).unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_m"),
        "expected InvalidConfig for hnsw_m, got {error:?}"
    );
}

#[test]
fn validate_config_rejects_ef_construction_zero() {
    let config = HnswConfig {
        ef_construction: 0,
        ..HnswConfig::default()
    };
    let error = validate_config(config).unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_ef_construction"),
        "expected InvalidConfig for ef_construction, got {error:?}"
    );
}

#[test]
fn validate_config_rejects_ef_search_zero() {
    let config = HnswConfig {
        ef_search: 0,
        ..HnswConfig::default()
    };
    let error = validate_config(config).unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_ef_search"),
        "expected InvalidConfig for ef_search, got {error:?}"
    );
}

#[test]
fn validate_config_rejects_max_layer_zero() {
    let config = HnswConfig {
        max_layer: 0,
        ..HnswConfig::default()
    };
    let error = validate_config(config).unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_max_layer"),
        "expected InvalidConfig for max_layer, got {error:?}"
    );
}

#[test]
fn validate_config_rejects_max_layer_above_native_limit() {
    let config = HnswConfig {
        max_layer: HNSW_DEFAULT_MAX_LAYER + 1,
        ..HnswConfig::default()
    };
    let error = validate_config(config).unwrap_err();
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "hnsw_max_layer"
        ),
        "expected InvalidConfig for hnsw_max_layer, got {error:?}"
    );
}

#[test]
fn native_storage_boundaries_validate_and_round_trip() {
    let source_path = temp_path("config-boundary-source", "fsvi");
    let source =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source");
    let config = HnswConfig {
        m: usize::from(u8::MAX),
        max_layer: HNSW_DEFAULT_MAX_LAYER,
        ..HnswConfig::default()
    };
    validate_config(config).expect("native storage boundaries must validate");

    let ann = HnswIndex::build_from_vector_index(&source, config)
        .expect("build at native storage boundaries");
    let metadata_path = temp_path("config-boundary-sidecar", "hnsw");
    ann.save(&metadata_path)
        .expect("persist native storage boundaries");
    let (loaded, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source)
        .expect("load native storage boundaries");
    assert_eq!(disposition, HnswLoadDisposition::Native);
    assert_eq!(loaded.config(), config);
}

// ── build_from_parts error paths ────────────────────────────────────

#[test]
fn build_rejects_dimension_zero() {
    let error = HnswIndex::build_from_parts(vec![], vec![], 0, HnswConfig::default()).unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "dimension"),
        "expected InvalidConfig for dimension, got {error:?}"
    );
}

#[test]
fn build_rejects_doc_id_vector_count_mismatch() {
    let error = HnswIndex::build_from_parts(
        vec!["a".to_owned(), "b".to_owned()],
        vec![vec![1.0, 0.0]],
        2,
        HnswConfig::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "vectors"),
        "expected InvalidConfig for vectors, got {error:?}"
    );
}

#[test]
fn build_rejects_vector_dimension_mismatch() {
    let error = HnswIndex::build_from_parts(
        vec!["a".to_owned()],
        vec![vec![1.0, 0.0, 0.0]], // 3D but declared 2D
        2,
        HnswConfig::default(),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            SearchError::DimensionMismatch {
                expected: 2,
                found: 3
            }
        ),
        "expected DimensionMismatch, got {error:?}"
    );
}

#[test]
fn build_rejects_nan_in_vector() {
    let error = HnswIndex::build_from_parts(
        vec!["a".to_owned()],
        vec![vec![1.0, f32::NAN]],
        2,
        HnswConfig::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, ref reason, .. }
                 if field == "vector" && reason.contains("finite")),
        "expected InvalidConfig for non-finite vector, got {error:?}"
    );
}

#[test]
fn build_rejects_infinity_in_vector() {
    let error = HnswIndex::build_from_parts(
        vec!["a".to_owned()],
        vec![vec![f32::INFINITY, 0.0]],
        2,
        HnswConfig::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "vector"),
        "expected InvalidConfig for non-finite vector, got {error:?}"
    );
}

// ── vector_component_close edge cases ────────────────────────────────

#[test]
fn vector_component_close_rejects_infinity_vs_finite() {
    // Regression: Inf <= (EPSILON * 8 * Inf) was true, causing false positive.
    assert!(!vector_component_close(f32::INFINITY, 100.0));
    assert!(!vector_component_close(100.0, f32::INFINITY));
    assert!(!vector_component_close(f32::NEG_INFINITY, 0.0));
    assert!(!vector_component_close(0.0, f32::NEG_INFINITY));
}

#[test]
fn vector_component_close_accepts_identical_infinities() {
    assert!(vector_component_close(f32::INFINITY, f32::INFINITY));
    assert!(vector_component_close(f32::NEG_INFINITY, f32::NEG_INFINITY));
}

#[test]
fn vector_component_close_rejects_opposite_infinities() {
    assert!(!vector_component_close(f32::INFINITY, f32::NEG_INFINITY));
}

#[test]
fn vector_component_close_nan_vs_finite_is_rejected() {
    assert!(!vector_component_close(f32::NAN, 0.0));
    assert!(!vector_component_close(0.0, f32::NAN));
}

#[test]
fn vector_component_close_identical_nan_bits_accepted() {
    // Same NaN bit pattern passes the to_bits() fast path. This is fine:
    // NaN vectors are rejected at construction time by build_from_parts().
    assert!(vector_component_close(f32::NAN, f32::NAN));
}

#[test]
fn vector_component_close_accepts_equal_values() {
    assert!(vector_component_close(0.0, 0.0));
    assert!(vector_component_close(1.0, 1.0));
    assert!(vector_component_close(-42.5, -42.5));
}

// ── knn_search_with_stats boundary conditions ───────────────────────

#[test]
fn search_with_k_zero_returns_empty_with_stats() {
    let path = temp_path("k0", "fsvi");
    let index = write_index(&path, &[normalized_vector(1, 16)]).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    let (hits, stats) = ann
        .knn_search_with_stats(&normalized_vector(1, 16), 0, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert!(hits.is_empty());
    assert_eq!(stats.k_requested, 0);
    assert_eq!(stats.k_returned, 0);
    assert!(stats.is_approximate);
    assert_eq!(stats.fallback_reason, None);
    assert!((stats.estimated_recall - 1.0).abs() < f64::EPSILON);
}

#[test]
fn search_dimension_mismatch_returns_error() {
    let path = temp_path("dimmis", "fsvi");
    let index = write_index(&path, &[normalized_vector(1, 16)]).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    let error = ann
        .knn_search_with_stats(&normalized_vector(1, 8), 5, HNSW_DEFAULT_EF_SEARCH)
        .unwrap_err();
    assert!(
        matches!(
            error,
            SearchError::DimensionMismatch {
                expected: 16,
                found: 8
            }
        ),
        "expected DimensionMismatch, got {error:?}"
    );
}

#[test]
fn search_stats_fields_are_populated() {
    let path = temp_path("stats", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..50).map(|i| normalized_vector(i, 32)).collect();
    let index = write_index(&path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    let (hits, stats) = ann
        .knn_search_with_stats(&normalized_vector(999, 32), 5, 64)
        .expect("search");
    assert_eq!(stats.index_size, 50);
    assert_eq!(stats.dimension, 32);
    assert_eq!(stats.ef_search, 64);
    assert_eq!(stats.k_requested, 5);
    assert_eq!(stats.k_returned, hits.len());
    assert!(stats.is_approximate);
    assert_eq!(stats.fallback_reason, None);
    assert!(stats.estimated_recall > 0.0);
    assert!(stats.estimated_recall <= 1.0);
}
