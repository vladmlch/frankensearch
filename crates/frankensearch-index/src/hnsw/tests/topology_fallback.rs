use std::time::Instant;

use frankensearch_core::{SearchError, VectorHit};
use hnsw_rs::prelude::{Neighbour, PointId};

use crate::{Quantization, VectorIndex};

use super::super::fingerprint::{
    certified_ann_recall_sample, dist_dot_budget, estimate_recall, normalize_for_dist_dot,
};
use super::super::topology::{
    measure_directed_base_coverage, validate_hnsw_topology_observations,
    validate_weak_base_connectivity,
};
use super::super::types::{
    AnnFallbackReason, AnnSearchStats, DIST_DOT_MAX_DIMENSION, HNSW_DEFAULT_EF_SEARCH, HnswConfig,
    HnswIndex, HnswLoadDisposition, HnswTopologyPoint,
};
use super::{normalized_vector, temp_path, write_index, write_index_with_quantization};

#[test]
fn underfilled_native_result_falls_back_to_exact_top_k() {
    let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0], vec![-1.0_f32, 0.0]];
    let source_path = temp_path("underfill-exact", "fsvi");
    let source = write_index(&source_path, &vectors).expect("source index");
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
    let budget = dist_dot_budget(2).expect("distance budget");
    let canonical_query = vec![1.0_f32, 0.0];

    let (hits, stats) = ann
        .finish_search_with_neighbors(
            Some(&source),
            &canonical_query,
            2,
            2,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect("exact underfill fallback");

    // The injected native result is empty over a populated graph, which
    // classifies as the stronger empty-despite-points anomaly (bd-tqhc).
    assert_eq!(
        stats.fallback_reason,
        Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
    );
    assert!(!stats.is_approximate);
    assert_eq!(stats.k_returned, 2);
    assert_eq!(stats.estimated_recall.to_bits(), 1.0_f64.to_bits());
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc-0000");
    assert_eq!(hits[1].doc_id, "doc-0001");
}

#[test]
fn underfilled_native_result_without_source_fails_closed() {
    let source_path = temp_path("underfill-no-source", "fsvi");
    let source =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
    let budget = dist_dot_budget(2).expect("distance budget");
    let error = ann
        .finish_search_with_neighbors(
            None,
            &[1.0_f32, 0.0],
            2,
            2,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect_err("underfill without canonical source must not claim exact repair");
    assert!(
        error.to_string().contains("knn_search_with_stats_against"),
        "{error}"
    );
}

#[test]
fn public_main_only_search_ignores_wal_consistently_before_and_after_underfill() {
    use crate::wal::WalEntry;

    let source_path = temp_path("underfill-public-main-only", "fsvi");
    let mut source =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
    source.wal_entries.push(WalEntry {
        doc_id: "doc-0000".into(),
        doc_id_hash: crate::fnv1a_hash(b"doc-0000"),
        embedding: vec![-1.0, 0.0],
    });
    let query = [1.0_f32, 0.0];

    let (normal, normal_stats) = ann
        .knn_search_with_stats_against(&source, &query, 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("normal public main-only ANN");
    assert_eq!(normal_stats.fallback_reason, None);
    assert_eq!(normal.len(), 1);
    assert_eq!(normal[0].doc_id, "doc-0000");

    let budget = dist_dot_budget(2).expect("distance budget");
    let (fallback, fallback_stats) = ann
        .finish_search_with_neighbors(
            Some(&source),
            &query,
            1,
            1,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect("public main-only exact repair");
    assert_eq!(
        fallback_stats.fallback_reason,
        Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
    );
    assert_eq!(fallback, normal);
}

#[test]
fn bounded_underfill_scan_matches_flat_exact_top_k() {
    let vectors = (0..256)
        .map(|seed| normalized_vector(seed, 32))
        .collect::<Vec<_>>();
    let source_path = temp_path("underfill-flat-oracle", "fsvi");
    let source = write_index(&source_path, &vectors).expect("flat source");
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
    let query = normalized_vector(10_000, 32);
    let exact = source.search_top_k(&query, 7, None).expect("flat exact");
    let budget = dist_dot_budget(32).expect("distance budget");
    let (fallback, stats) = ann
        .finish_search_with_neighbors(
            Some(&source),
            &query,
            7,
            7,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect("bounded exact fallback");

    assert_eq!(
        stats.fallback_reason,
        Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
    );
    assert_eq!(fallback, exact);
}

#[test]
fn f16_non_unit_underfill_fallback_is_bit_exact_with_canonical_scan() {
    let vectors = vec![
        vec![8.0_f32, 0.25, -0.5],
        vec![0.5_f32, 0.5, 0.5],
        vec![1.25_f32, 2.5, -0.75],
        vec![-3.0_f32, 0.125, 4.0],
    ];
    let source_path = temp_path("underfill-f16-non-unit", "fsvi");
    let source = write_index_with_quantization(&source_path, &vectors, Quantization::F16)
        .expect("F16 source");
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
    let canonical_query = vec![2.0_f32, -0.5, 0.25];
    let exact = source
        .search_main_top_k(&canonical_query, 3)
        .expect("canonical persisted scan");
    let budget = dist_dot_budget(3).expect("distance budget");
    let (fallback, stats) = ann
        .finish_search_with_neighbors(
            Some(&source),
            &canonical_query,
            3,
            3,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect("F16 canonical fallback");

    assert_eq!(
        stats.fallback_reason,
        Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
    );
    assert_eq!(fallback, exact);
    assert!(
        fallback
            .iter()
            .zip(&exact)
            .all(|(actual, expected)| actual.score.to_bits() == expected.score.to_bits())
    );

    let metadata_path = temp_path("underfill-f16-native", "hnsw");
    ann.save(&metadata_path).expect("save native graph");
    let (loaded, disposition) =
        HnswIndex::load_with_disposition(&metadata_path, &source).expect("load native graph");
    assert_eq!(disposition, HnswLoadDisposition::Native);
    let (native_fallback, native_stats) = loaded
        .finish_search_with_neighbors(
            Some(&source),
            &canonical_query,
            3,
            3,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect("native F16 canonical fallback");
    assert_eq!(
        native_stats.fallback_reason,
        Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
    );
    assert_eq!(native_fallback, exact);
}

#[test]
fn underfill_fallback_excludes_wal_for_single_merge_in_two_tier() {
    let vectors = vec![vec![0.5_f32, 0.0], vec![0.25_f32, 0.0], vec![-0.5_f32, 0.0]];
    let source_path = temp_path("underfill-wal", "fsvi");
    let mut source = write_index(&source_path, &vectors).expect("source");
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
    source
        .append("wal-top", &[1.0_f32, 0.0])
        .expect("append resident WAL entry");
    let canonical_query = vec![1.0_f32, 0.0];
    let expected_main = source
        .search_main_top_k(&canonical_query, 2)
        .expect("persisted-only oracle");
    let full = source
        .search_top_k(&canonical_query, 2, None)
        .expect("main plus WAL oracle");
    assert_eq!(full[0].doc_id, "wal-top");
    let budget = dist_dot_budget(2).expect("distance budget");
    let (fallback, stats) = ann
        .finish_search_with_neighbors(
            Some(&source),
            &canonical_query,
            2,
            2,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect("main-only fallback");

    assert_eq!(
        stats.fallback_reason,
        Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
    );
    assert_eq!(fallback, expected_main);
    assert!(
        fallback.iter().all(|hit| hit.doc_id != "wal-top"),
        "resident WAL must be merged exactly once by TwoTierIndex"
    );
}

#[test]
fn duplicate_doc_ids_and_tombstone_gaps_keep_distinct_physical_rows() {
    let source_path = temp_path("duplicate-id-gap", "fsvi");
    let mut writer =
        VectorIndex::create_with_revision(&source_path, "hash", "test", 2, Quantization::F32)
            .expect("writer");
    writer
        .write_record("duplicate", &[1.0_f32, 0.0])
        .expect("first duplicate");
    writer.write_record("gap", &[0.0_f32, 1.0]).expect("gap");
    writer
        .write_record("duplicate", &[-1.0_f32, 0.0])
        .expect("second duplicate");
    writer.finish().expect("finish source");
    let mut source = VectorIndex::open(&source_path).expect("open source");
    assert!(source.soft_delete("gap").expect("tombstone gap"));
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
    let expected_duplicate_positions = (0..source.record_count())
        .filter(|&position| {
            !source.is_deleted(position)
                && source.doc_id_at(position).expect("source doc ID") == "duplicate"
        })
        .map(|position| u32::try_from(position).expect("test position fits u32"))
        .collect::<Vec<_>>();
    assert_eq!(ann.source_positions, expected_duplicate_positions);
    assert_eq!(ann.source_record_count, 3);
    let mut selected_positions = Vec::new();
    for query in [[1.0_f32, 0.0], [-1.0_f32, 0.0]] {
        let expected = source
            .search_main_top_k(&query, 1)
            .expect("canonical physical row");
        assert_eq!(expected.len(), 1);
        assert_eq!(expected[0].doc_id, "duplicate");

        let (ann_hits, ann_stats) = ann
            .knn_search_with_stats_against(&source, &query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("normal ANN search");
        assert_eq!(ann_stats.fallback_reason, None);
        assert_eq!(
            ann_hits, expected,
            "ANN must preserve the winning duplicate's physical identity"
        );

        let budget = dist_dot_budget(2).expect("distance budget");
        let (fallback, fallback_stats) = ann
            .finish_search_with_neighbors(
                Some(&source),
                &query,
                1,
                1,
                HNSW_DEFAULT_EF_SEARCH,
                Vec::new(),
                budget,
                Instant::now(),
            )
            .expect("canonical fallback");
        assert_eq!(
            fallback_stats.fallback_reason,
            Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
        );
        assert_eq!(fallback, expected);
        selected_positions.push(expected[0].index);
    }
    assert_ne!(
        selected_positions[0], selected_positions[1],
        "opposing queries must retain the two duplicate-ID physical rows instead of \
         remapping both through the first matching document ID"
    );

    let expected = source
        .search_main_top_k(&[1.0_f32, 0.0], 2)
        .expect("canonical duplicate-ID semantics");
    assert_eq!(
        expected.len(),
        1,
        "canonical search selects two physical winners, then deduplicates public doc IDs"
    );
    let (raw_hits, raw_stats) = ann
        .knn_search_raw_with_stats_against(&source, &[1.0_f32, 0.0], 2, HNSW_DEFAULT_EF_SEARCH)
        .expect("raw duplicate-ID ANN search");
    assert_eq!(raw_stats.fallback_reason, None);
    assert_eq!(raw_hits.len(), 2);
    assert!(
        raw_hits.iter().all(|hit| hit.doc_id == "duplicate"),
        "raw TwoTier candidates must retain both physical duplicate-ID winners"
    );
    assert_ne!(
        raw_hits[0].index, raw_hits[1].index,
        "raw candidates must preserve distinct physical rows"
    );
    let (ann_hits, ann_stats) = ann
        .knn_search_with_stats_against(&source, &[1.0_f32, 0.0], 2, HNSW_DEFAULT_EF_SEARCH)
        .expect("duplicate-ID ANN search");
    assert_eq!(ann_stats.fallback_reason, None);
    assert_eq!(
        ann_hits, expected,
        "ANN must preserve physical identity internally while matching canonical \
         duplicate-ID result semantics"
    );
}

#[test]
fn post_build_tombstone_is_filtered_and_underfill_uses_current_exact_source() {
    let source_path = temp_path("post-build-tombstone", "fsvi");
    let mut source = write_index(
        &source_path,
        &[
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0_f32, 1.0, 0.0],
            vec![0.0_f32, 0.0, 1.0],
        ],
    )
    .expect("source");
    let ann = HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN");
    assert!(source.soft_delete("doc-0000").expect("post-build delete"));
    let query = [1.0_f32, 0.0, 0.0];
    let expected = source
        .search_main_top_k(&query, 3)
        .expect("current exact source");
    assert_eq!(expected.len(), 2);
    assert!(expected.iter().all(|hit| hit.doc_id != "doc-0000"));

    let (hits, stats) = ann
        .knn_search_with_stats_against(&source, &query, 3, HNSW_DEFAULT_EF_SEARCH)
        .expect("tombstone-aware ANN search");
    assert_eq!(stats.fallback_reason, Some(AnnFallbackReason::Underfilled));
    assert_eq!(hits, expected);

    let budget = dist_dot_budget(3).expect("distance budget");
    let (forced, forced_stats) = ann
        .finish_search_with_neighbors(
            Some(&source),
            &query,
            3,
            3,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect("forced exact repair after tombstone");
    assert_eq!(
        forced_stats.fallback_reason,
        Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
    );
    assert_eq!(forced, expected);
}

#[test]
fn borrowed_source_fallback_survives_path_rename_after_open() {
    let source_path = temp_path("borrowed-source-before-rename", "fsvi");
    let renamed_path = temp_path("borrowed-source-after-rename", "fsvi");
    let source =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source");
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("ANN index");
    std::fs::rename(&source_path, &renamed_path).expect("rename open source file");
    let expected = source
        .search_main_top_k(&[1.0_f32, 0.0], 2)
        .expect("open mmap remains authoritative after rename");
    let budget = dist_dot_budget(2).expect("distance budget");
    let (fallback, stats) = ann
        .finish_search_with_neighbors(
            Some(&source),
            &[1.0_f32, 0.0],
            2,
            2,
            HNSW_DEFAULT_EF_SEARCH,
            Vec::new(),
            budget,
            Instant::now(),
        )
        .expect("borrowed-source fallback after rename");
    assert_eq!(
        stats.fallback_reason,
        Some(AnnFallbackReason::EmptyDespiteIndexedPoints)
    );
    assert_eq!(fallback, expected);
}

#[test]
fn topology_validator_accepts_valid_layers_and_rejects_captured_wrong_layer() {
    let point_zero_id = PointId(0, 0);
    let point_one_id = PointId(1, 0);
    let valid = vec![
        HnswTopologyPoint {
            origin_id: 0,
            point_id: point_zero_id,
            neighborhoods: vec![vec![Neighbour::new(1, 1.0, point_one_id)], Vec::new()],
        },
        HnswTopologyPoint {
            origin_id: 1,
            point_id: point_one_id,
            neighborhoods: vec![vec![Neighbour::new(0, 1.0, point_zero_id)], Vec::new()],
        },
    ];
    validate_hnsw_topology_observations(&valid, 2, Some((1, point_one_id)))
        .expect("valid topology");

    let mut wrong_layer = valid;
    wrong_layer[0].neighborhoods[1].push(Neighbour::new(1, 1.0, point_one_id));
    let detail = validate_hnsw_topology_observations(&wrong_layer, 2, Some((1, point_one_id)))
        .expect_err("neighbor above source sampled layer must fail");
    assert!(detail.contains("above sampled layer"), "{detail}");
}

#[test]
fn topology_validator_rejects_duplicate_and_out_of_range_origin_ids() {
    let duplicate = vec![
        HnswTopologyPoint {
            origin_id: 0,
            point_id: PointId(0, 0),
            neighborhoods: vec![Vec::new()],
        },
        HnswTopologyPoint {
            origin_id: 0,
            point_id: PointId(0, 1),
            neighborhoods: vec![Vec::new()],
        },
    ];
    let detail = validate_hnsw_topology_observations(&duplicate, 2, Some((0, PointId(0, 0))))
        .expect_err("duplicate origin must fail");
    assert!(detail.contains("duplicate origin id"), "{detail}");

    let out_of_range = vec![HnswTopologyPoint {
        origin_id: 1,
        point_id: PointId(0, 0),
        neighborhoods: vec![Vec::new()],
    }];
    let detail = validate_hnsw_topology_observations(&out_of_range, 1, Some((1, PointId(0, 0))))
        .expect_err("out-of-range origin must fail");
    assert!(detail.contains("outside 0..1"), "{detail}");
}

#[test]
fn topology_validator_rejects_edgeless_multi_point_graph() {
    let edgeless = vec![
        HnswTopologyPoint {
            origin_id: 0,
            point_id: PointId(0, 0),
            neighborhoods: vec![Vec::new()],
        },
        HnswTopologyPoint {
            origin_id: 1,
            point_id: PointId(0, 1),
            neighborhoods: vec![Vec::new()],
        },
    ];
    let detail = validate_hnsw_topology_observations(&edgeless, 2, Some((0, PointId(0, 0))))
        .expect_err("edgeless multi-point graph must fail");
    assert!(detail.contains("weak components"), "{detail}");
    assert!(detail.contains("only 1/2"), "{detail}");
    assert!(detail.contains("2 weak components total"), "{detail}");
    assert!(
        detail.contains("2 points have zero base out-edges (first: origin 0)"),
        "{detail}"
    );
}

/// #32: bounded neighbour pruning can strip every in-edge from a point,
/// leaving a directed "source" that no walk from the entry can visit even
/// though the graph is one weak component. Real serial and parallel builds
/// of large corpora produce exactly this shape, so attestation must admit
/// it (coverage is reported separately) instead of rejecting the build.
#[test]
fn topology_validator_admits_directed_source_points_in_one_weak_component() {
    let ids = [PointId(0, 0), PointId(0, 1), PointId(0, 2)];
    let points = vec![
        // Entry keeps no out-edges at the base layer; every other point
        // kept only its out-edge toward the entry (in-edges pruned away).
        HnswTopologyPoint {
            origin_id: 0,
            point_id: ids[0],
            neighborhoods: vec![Vec::new()],
        },
        HnswTopologyPoint {
            origin_id: 1,
            point_id: ids[1],
            neighborhoods: vec![vec![Neighbour::new(0, 1.0, ids[0])]],
        },
        HnswTopologyPoint {
            origin_id: 2,
            point_id: ids[2],
            neighborhoods: vec![vec![Neighbour::new(0, 1.0, ids[0])]],
        },
    ];

    validate_hnsw_topology_observations(&points, 3, Some((0, ids[0])))
        .expect("weakly connected graph with directed source points must be admitted");
}

#[test]
fn directed_base_coverage_reports_unreached_points_without_failing() {
    // Same shape as the admission test above: entry 0 has no out-edges.
    let adjacency: Vec<Vec<usize>> = vec![Vec::new(), vec![0], vec![0]];
    let coverage =
        measure_directed_base_coverage(0, 3, |origin_id| Ok(adjacency[origin_id].clone()))
            .expect("coverage walk must succeed");
    assert_eq!(coverage.reached_count, 1);
    assert_eq!(coverage.first_unreachable, Some(1));

    // A directed cycle covers everything.
    let cycle: Vec<Vec<usize>> = vec![vec![1], vec![2], vec![0]];
    let coverage = measure_directed_base_coverage(0, 3, |origin_id| Ok(cycle[origin_id].clone()))
        .expect("coverage walk must succeed");
    assert_eq!(coverage.reached_count, 3);
    assert_eq!(coverage.first_unreachable, None);
}

#[test]
fn weak_connectivity_rejects_out_of_range_neighbor() {
    let adjacency: Vec<Vec<usize>> = vec![vec![1], vec![7]];
    let detail =
        validate_weak_base_connectivity(0, 2, |origin_id| Ok(adjacency[origin_id].clone()))
            .expect_err("out-of-range neighbor must fail");
    assert!(detail.contains("out-of-range base neighbor 7"), "{detail}");
}

#[test]
fn topology_validator_rejects_edge_above_target_sampled_layer() {
    let low_id = PointId(0, 0);
    let high_id = PointId(1, 0);
    let points = vec![
        HnswTopologyPoint {
            origin_id: 0,
            point_id: low_id,
            neighborhoods: vec![vec![Neighbour::new(1, 1.0, high_id)]],
        },
        HnswTopologyPoint {
            origin_id: 1,
            point_id: high_id,
            neighborhoods: vec![
                vec![Neighbour::new(0, 1.0, low_id)],
                vec![Neighbour::new(0, 1.0, low_id)],
            ],
        },
    ];
    let detail = validate_hnsw_topology_observations(&points, 2, Some((1, high_id)))
        .expect_err("edge above target sampled layer must fail");
    assert!(detail.contains("neighbor's sampled layer"), "{detail}");
}

#[test]
fn topology_validator_rejects_disconnected_nonempty_components() {
    let ids = [PointId(0, 0), PointId(0, 1), PointId(0, 2), PointId(0, 3)];
    let points = vec![
        HnswTopologyPoint {
            origin_id: 0,
            point_id: ids[0],
            neighborhoods: vec![vec![Neighbour::new(1, 1.0, ids[1])]],
        },
        HnswTopologyPoint {
            origin_id: 1,
            point_id: ids[1],
            neighborhoods: vec![vec![Neighbour::new(0, 1.0, ids[0])]],
        },
        HnswTopologyPoint {
            origin_id: 2,
            point_id: ids[2],
            neighborhoods: vec![vec![Neighbour::new(3, 1.0, ids[3])]],
        },
        HnswTopologyPoint {
            origin_id: 3,
            point_id: ids[3],
            neighborhoods: vec![vec![Neighbour::new(2, 1.0, ids[2])]],
        },
    ];

    let detail = validate_hnsw_topology_observations(&points, 4, Some((0, ids[0])))
        .expect_err("two locally valid components must fail weak connectivity");
    assert!(detail.contains("only 2/4"), "{detail}");
    assert!(detail.contains("first detached origin is 2"), "{detail}");
    assert!(detail.contains("2 weak components total"), "{detail}");
    assert!(
        detail.contains("0 points have zero base out-edges (first: none)"),
        "{detail}"
    );
}

/// Pin the weak-component union-find at the exact cardinality of the
/// largest real corpus that has exercised this gate (CASS's published
/// quality tier). A forward-attached insertion chain — each non-seed point
/// holding one base edge to its predecessor, the invariant `insert_slice`
/// guarantees — must resolve to a single weak component with no overflow
/// or path-compression fault at this scale.
#[test]
fn weak_connectivity_attests_single_chain_at_production_cardinality() {
    const PRODUCTION_POINT_COUNT: usize = 2_573_003;

    validate_weak_base_connectivity(0, PRODUCTION_POINT_COUNT, |origin_id| {
        Ok(if origin_id == 0 {
            Vec::new()
        } else {
            vec![origin_id - 1]
        })
    })
    .expect("a forward-attached insertion chain is one weak component");
}

#[test]
fn topology_validator_rejects_mismatched_entry_identity() {
    let id = PointId(0, 0);
    let points = vec![HnswTopologyPoint {
        origin_id: 0,
        point_id: id,
        neighborhoods: vec![Vec::new()],
    }];

    let detail = validate_hnsw_topology_observations(&points, 1, Some((1, id)))
        .expect_err("entry origin must agree with its internal point identity");
    assert!(detail.contains("not advertised origin 1"), "{detail}");
}

#[test]
fn calibration_rejects_fallback_tainted_exact_results() {
    let exact = vec![VectorHit {
        index: 0,
        score: 1.0,
        doc_id: "doc".into(),
    }];
    let stats = AnnSearchStats {
        index_size: 1,
        dimension: 2,
        ef_search: 1,
        k_requested: 1,
        k_returned: 1,
        search_time_us: 10,
        is_approximate: false,
        fallback_reason: Some(AnnFallbackReason::Underfilled),
        estimated_recall: 1.0,
        zero_signal: None,
    };
    assert_eq!(
        certified_ann_recall_sample(Ok((exact.clone(), stats)), &exact).to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn search_k_larger_than_index_returns_all() {
    let path = temp_path("klarge", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..5).map(|i| normalized_vector(i, 16)).collect();
    let index = write_index(&path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    let hits = ann
        .knn_search(&normalized_vector(999, 16), 100, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert_eq!(hits.len(), 5);
}

// ── matches_vector_index edge cases ─────────────────────────────────

#[test]
fn matches_returns_false_for_dimension_mismatch() {
    let path_a = temp_path("match-a", "fsvi");
    let path_b = temp_path("match-b", "fsvi");
    let index_a = write_index(&path_a, &[normalized_vector(1, 16)]).expect("index_a");
    let index_b = write_index(&path_b, &[normalized_vector(1, 32)]).expect("index_b");
    let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
    assert!(!ann.matches_vector_index(&index_b).expect("matches"));
}

#[test]
fn matches_returns_false_for_record_count_mismatch() {
    let path_a = temp_path("match-rc-a", "fsvi");
    let path_b = temp_path("match-rc-b", "fsvi");
    let index_a = write_index(
        &path_a,
        &[normalized_vector(1, 16), normalized_vector(2, 16)],
    )
    .expect("index_a");
    let index_b = write_index(&path_b, &[normalized_vector(1, 16)]).expect("index_b");
    let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
    assert!(!ann.matches_vector_index(&index_b).expect("matches"));
}

#[test]
fn matches_returns_false_when_vectors_change_but_doc_ids_match() {
    // Matching document IDs and dimensions are insufficient: serving a
    // graph built from different vectors would silently return stale ANN
    // results. The full live-vector fingerprint must reject that source.
    let path_a = temp_path("match-vec-a", "fsvi");
    let path_b = temp_path("match-vec-b", "fsvi");
    let index_a = write_index(
        &path_a,
        &[normalized_vector(1, 16), normalized_vector(2, 16)],
    )
    .expect("index_a");
    let index_b = write_index(
        &path_b,
        &[normalized_vector(3, 16), normalized_vector(4, 16)],
    )
    .expect("index_b");
    let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
    assert!(!ann.matches_vector_index(&index_b).expect("matches"));
}

#[test]
fn build_from_vector_index_excludes_tombstoned_records() {
    let path = temp_path("tombstone-filter", "fsvi");
    let mut index = write_index(
        &path,
        &[
            normalized_vector(1, 16),
            normalized_vector(2, 16),
            normalized_vector(3, 16),
        ],
    )
    .expect("index");
    let deleted = index
        .soft_delete("doc-0001")
        .expect("soft_delete should succeed");
    assert!(deleted);

    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    assert_eq!(ann.len(), 2, "ANN should only index live vectors");

    let query = normalized_vector(2, 16);
    let hits = ann
        .knn_search(&query, 10, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert!(
        !hits.iter().any(|hit| hit.doc_id == "doc-0001"),
        "ANN should never return tombstoned doc IDs"
    );
}

// ── normalize_for_dist_dot ──────────────────────────────────────────

#[test]
fn normalize_zero_vector_unchanged() {
    let zero = vec![0.0_f32; 8];
    let budget = dist_dot_budget(zero.len()).expect("budget");
    let result = normalize_for_dist_dot(zero.clone(), budget);
    assert_eq!(
        result, zero,
        "zero vector should remain zero after normalize"
    );
}

#[test]
fn dist_dot_roundoff_budget_is_dimension_aware_and_ranking_neutral() {
    let radius_16 = dist_dot_budget(16).expect("budget").radius_squared;
    let radius_384 = dist_dot_budget(384).expect("budget").radius_squared;
    let radius_4096 = dist_dot_budget(4096).expect("budget").radius_squared;

    assert!((0.5..1.0).contains(&radius_16));
    assert!(radius_384 < radius_16);
    assert!(radius_4096 < radius_384);
    assert!(
        radius_384 > 0.999,
        "384-dim safety margin must stay small enough to preserve cosine resolution"
    );

    let error = dist_dot_budget(DIST_DOT_MAX_DIMENSION + 1)
        .expect_err("ill-conditioned f32 bound must fail closed");
    assert!(matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "dimension"));
}

#[test]
fn dist_dot_normalization_stays_below_one_across_reduction_orders() {
    fn reassociated_dot(left: &[f32], right: &[f32], lanes: usize) -> f32 {
        let mut partials = vec![0.0_f32; lanes];
        for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
            partials[index % lanes] += left * right;
        }
        partials.into_iter().fold(0.0_f32, |sum, value| sum + value)
    }

    for dimension in [16_usize, 384, 4_096] {
        let budget = dist_dot_budget(dimension).expect("budget");
        let vectors: Vec<Vec<f32>> = (0..16)
            .map(|seed| normalize_for_dist_dot(normalized_vector(seed, dimension), budget))
            .collect();
        for left in &vectors {
            for right in &vectors {
                for lanes in [1_usize, 2, 4, 8, 16] {
                    let dot = reassociated_dot(left, right, lanes);
                    assert!(
                        dot <= 1.0,
                        "DistDot precondition violated: dim={dimension} lanes={lanes} dot={dot}"
                    );
                }
            }
        }
    }
}

#[test]
fn safe_radius_scaling_preserves_valid_cosine_ranking() {
    fn dot(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    let query = vec![1.0_f32, 0.0, 0.0, 0.0];
    let candidates = [
        vec![1.0_f32, 0.0, 0.0, 0.0],
        vec![0.8_f32, 0.6, 0.0, 0.0],
        vec![0.6_f32, 0.8, 0.0, 0.0],
        vec![-0.2_f32, 0.0, 0.0, 0.98],
    ];
    let budget = dist_dot_budget(query.len()).expect("budget");
    let scaled_query = normalize_for_dist_dot(query.clone(), budget);

    let mut original: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let candidate_norm = dot(candidate, candidate).sqrt();
            (index, dot(&query, candidate) / candidate_norm)
        })
        .collect();
    let mut restored: Vec<(usize, f32)> = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            let scaled = normalize_for_dist_dot(candidate, budget);
            (index, dot(&scaled_query, &scaled) / budget.radius_squared)
        })
        .collect();
    original.sort_by(|left, right| right.1.total_cmp(&left.1));
    restored.sort_by(|left, right| right.1.total_cmp(&left.1));

    assert_eq!(
        original.iter().map(|entry| entry.0).collect::<Vec<_>>(),
        restored.iter().map(|entry| entry.0).collect::<Vec<_>>(),
        "uniform safety scaling must not change valid cosine ranking"
    );
    for ((_, expected), (_, actual)) in original.iter().zip(&restored) {
        assert!((expected - actual).abs() <= budget.score_tolerance);
    }
}

#[test]
fn hnsw_restores_cosine_score_after_safe_radius_scaling() {
    let path = temp_path("safe-radius-score", "fsvi");
    let vector = normalized_vector(41, 384);
    let index = write_index(&path, std::slice::from_ref(&vector)).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

    let hits = ann
        .knn_search(&vector, 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert!(
        (hits[0].score - 1.0).abs() <= 1.0e-5,
        "uniform DistDot safety scaling must not leak into public cosine scores: {}",
        hits[0].score
    );
}

#[test]
fn hnsw_rejects_non_finite_query_before_distance_evaluation() {
    let path = temp_path("non-finite-query", "fsvi");
    let vector = normalized_vector(42, 32);
    let index = write_index(&path, std::slice::from_ref(&vector)).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    let mut query = vector;
    query[7] = f32::NAN;

    let error = ann
        .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
        .expect_err("non-finite queries must return an error rather than panic in DistDot");
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, ref reason, .. }
            if field == "query" && reason.contains("finite")),
        "expected query InvalidConfig, got {error:?}"
    );
}

// ── estimate_recall ─────────────────────────────────────────────────

#[test]
fn estimate_recall_k_zero_returns_one() {
    assert!((estimate_recall(100, 0) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn estimate_recall_clamped_between_zero_and_one() {
    // Very low ef relative to k
    let low = estimate_recall(1, 1000);
    assert!((0.0..=1.0).contains(&low), "low recall: {low}");

    // Very high ef relative to k
    let high = estimate_recall(10_000, 1);
    assert!((0.0..=1.0).contains(&high), "high recall: {high}");
}

#[test]
fn estimate_recall_increases_with_ef() {
    let r_low = estimate_recall(10, 10);
    let r_high = estimate_recall(100, 10);
    assert!(
        r_high >= r_low,
        "recall should increase with ef: {r_low} vs {r_high}"
    );
}

// ── len / is_empty / dimension / config accessors ───────────────────

#[test]
fn accessors_report_correct_values() {
    let path = temp_path("accessors", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..10).map(|i| normalized_vector(i, 24)).collect();
    let index = write_index(&path, &vectors).expect("index");
    let config = HnswConfig {
        m: 8,
        ..HnswConfig::default()
    };
    let ann = HnswIndex::build_from_vector_index(&index, config).expect("ann");
    assert_eq!(ann.len(), 10);
    assert!(!ann.is_empty());
    assert_eq!(ann.dimension(), 24);
    assert_eq!(ann.config().m, 8);
}

// ── Debug impl ──────────────────────────────────────────────────────

#[test]
fn debug_impl_does_not_panic() {
    let path = temp_path("debug", "fsvi");
    let index = write_index(&path, &[normalized_vector(1, 8)]).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
    let debug_str = format!("{ann:?}");
    assert!(debug_str.contains("HnswIndex"));
    assert!(debug_str.contains("dimension: 8"));
}

// ── Original tests ──────────────────────────────────────────────────

#[test]
fn scores_are_consistent_with_exact_top_hit() {
    let fsvi_path = temp_path("score", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..128).map(|i| normalized_vector(i, 64)).collect();
    let index = write_index(&fsvi_path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

    let query = normalized_vector(7, 64);
    let exact = index.search_top_k(&query, 1, None).expect("exact");
    let approx = ann
        .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("approx");

    assert_eq!(exact[0].doc_id, approx[0].doc_id);
    assert!((exact[0].score - approx[0].score).abs() < 1e-3);
}
