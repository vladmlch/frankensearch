use frankensearch_core::error::SearchError;

use super::super::codec::{Adjacency, MutationJournal, NativeHnsw};
use super::super::graph::LevelSampler;
use super::super::io_helpers::{put_u32, put_u64};
use super::super::types::{GraphDefect, HnswParams, MAX_LEVEL};
use super::super::{
    NATIVE_HNSW_FORMAT_VERSION, NATIVE_HNSW_HEADER_CRC_OFFSET, NATIVE_HNSW_HEADER_LEN,
};
use super::{
    FailingStore, TestStore, first_persisted_neighbour, params, reseal_persisted_checksums,
    tempfile, topology_snapshot,
};

// ─── Owned persistence ──────────────────────────────────────────────

#[test]
fn owned_empty_graph_round_trip_preserves_empty_identity() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("empty.fshnsw");
    let store = TestStore::new(Vec::new());
    let graph = NativeHnsw::build(params(), 0x5eed_cafe, &store).expect("build empty");

    let saved = graph.save(&path).expect("save empty");
    let loaded = NativeHnsw::load(&path, &store).expect("load empty");

    assert_eq!(saved.point_count(), 0);
    assert_eq!(
        saved.byte_len(),
        u64::try_from(NATIVE_HNSW_HEADER_LEN).expect("header length fits u64")
    );
    assert_eq!(loaded.params(), graph.params());
    assert_eq!(loaded.seed(), graph.seed());
    assert_eq!(loaded.len(), 0);
    assert!(loaded.entry_point().is_none());
    assert_eq!(loaded.max_level(), 0);
    assert!(
        loaded
            .search(&[1.0], 10, None, &store)
            .expect("search")
            .is_empty()
    );
}

#[test]
fn owned_empty_graph_rejects_resealed_nonzero_max_level() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("empty.fshnsw");
    let store = TestStore::new(Vec::new());
    NativeHnsw::build(params(), 0x5eed_cafe, &store)
        .expect("build empty")
        .save(&path)
        .expect("save empty");

    let mut bytes = std::fs::read(&path).expect("empty graph bytes");
    put_u64(&mut bytes, 72, 1);
    reseal_persisted_checksums(&mut bytes);
    std::fs::write(&path, bytes).expect("write resealed empty-level forgery");

    let error = NativeHnsw::load(&path, &store).expect_err("empty graph maximum must remain zero");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("empty graph claims maximum level 1")),
        "unexpected empty-graph maximum-level error: {error:?}"
    );
}

#[test]
fn owned_graph_round_trip_preserves_topology_identity_and_search() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("graph.fshnsw");
    let store = TestStore::synthetic(96, 8);
    let graph = NativeHnsw::build(params(), 0x5eed_cafe, &store).expect("build");

    let saved = graph.save(&path).expect("save");
    let (loaded, observed) =
        NativeHnsw::load_with_metadata(&path, &store).expect("load and attest");

    assert_eq!(saved, observed);
    assert_eq!(saved.format_version(), NATIVE_HNSW_FORMAT_VERSION);
    assert_eq!(saved.point_count(), 96);
    assert_eq!(
        saved.byte_len(),
        std::fs::metadata(&path).expect("metadata").len()
    );
    assert_ne!(saved.payload_crc32(), 0);
    assert_ne!(saved.header_crc32(), 0);
    assert_eq!(loaded.params(), graph.params());
    assert_eq!(loaded.seed(), graph.seed());
    assert_eq!(topology_snapshot(&loaded), topology_snapshot(&graph));

    for query_id in 0..8 {
        let query = TestStore::query(query_id, 8);
        assert_eq!(
            loaded
                .search(&query, 10, None, &store)
                .expect("loaded search"),
            graph
                .search(&query, 10, None, &store)
                .expect("original search")
        );
    }
}

#[test]
fn owned_graph_serialization_is_byte_deterministic() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let first_path = directory.path().join("first.fshnsw");
    let second_path = directory.path().join("second.fshnsw");
    let store = TestStore::synthetic(64, 6);
    let graph = NativeHnsw::build(params(), 0x1234_5678, &store).expect("build");

    graph.save(&first_path).expect("first save");
    graph.save(&second_path).expect("second save");

    assert_eq!(
        std::fs::read(first_path).expect("first bytes"),
        std::fs::read(second_path).expect("second bytes")
    );
}

#[test]
fn owned_graph_load_rejects_truncation_and_payload_tampering() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("graph.fshnsw");
    let store = TestStore::synthetic(48, 6);
    let graph = NativeHnsw::build(params(), 91, &store).expect("build");
    graph.save(&path).expect("save");
    let original = std::fs::read(&path).expect("graph bytes");

    std::fs::write(&path, &original[..original.len() - 1]).expect("truncate");
    let error = NativeHnsw::load(&path, &store).expect_err("truncation must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { .. }),
        "unexpected truncation error: {error:?}"
    );

    let mut tampered = original;
    let last = tampered.last_mut().expect("non-empty graph file");
    *last ^= 0x80;
    std::fs::write(&path, tampered).expect("tamper payload");
    let error = NativeHnsw::load(&path, &store).expect_err("payload tamper must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("payload CRC mismatch")),
        "payload tamper did not reach the CRC gate: {error:?}"
    );
}

#[test]
fn owned_graph_load_rejects_header_version_and_checksum_tampering() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("graph.fshnsw");
    let store = TestStore::synthetic(24, 4);
    NativeHnsw::build(params(), 17, &store)
        .expect("build")
        .save(&path)
        .expect("save");
    let original = std::fs::read(&path).expect("graph bytes");

    let mut wrong_version = original.clone();
    put_u32(
        &mut wrong_version,
        8,
        NATIVE_HNSW_FORMAT_VERSION.saturating_add(1),
    );
    reseal_persisted_checksums(&mut wrong_version);
    std::fs::write(&path, wrong_version).expect("write version tamper");
    let error = NativeHnsw::load(&path, &store).expect_err("version tamper must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("unsupported native HNSW format version")),
        "unexpected version error: {error:?}"
    );

    let mut bad_header_crc = original;
    bad_header_crc[NATIVE_HNSW_HEADER_CRC_OFFSET] ^= 0x01;
    std::fs::write(&path, bad_header_crc).expect("write CRC tamper");
    let error = NativeHnsw::load(&path, &store).expect_err("header CRC tamper must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("header CRC mismatch")),
        "unexpected header CRC error: {error:?}"
    );
}

#[test]
fn owned_graph_structural_attestation_rejects_resealed_forgery() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("graph.fshnsw");
    let store = TestStore::synthetic(40, 6);
    NativeHnsw::build(params(), 73, &store)
        .expect("build")
        .save(&path)
        .expect("save");

    let mut bytes = std::fs::read(&path).expect("graph bytes");
    let (neighbour_offset, owner) =
        first_persisted_neighbour(&bytes).expect("non-trivial graph contains an edge");
    put_u32(&mut bytes, neighbour_offset, owner);
    reseal_persisted_checksums(&mut bytes);
    std::fs::write(&path, bytes).expect("write resealed forgery");

    let error = NativeHnsw::load(&path, &store).expect_err("resealed structural forgery must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("is its own neighbour")),
        "unexpected structural error: {error:?}"
    );
}

#[test]
fn owned_graph_seed_is_attested_against_every_persisted_level() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("graph.fshnsw");
    let store = TestStore::synthetic(64, 6);
    let graph = NativeHnsw::build(params(), 11, &store).expect("build");
    graph.save(&path).expect("save");

    let mut bytes = std::fs::read(&path).expect("graph bytes");
    let forged_seed = (12_u64..=u64::from(u16::MAX))
        .find(|&candidate| {
            let sampler = LevelSampler::new(params().m, candidate);
            graph.adjacency.iter().enumerate().any(|(id, point)| {
                sampler.level_for(u32::try_from(id).expect("test id")) != point.level()
            })
        })
        .expect("some alternate seed changes a sampled level");
    put_u64(&mut bytes, 48, forged_seed);
    reseal_persisted_checksums(&mut bytes);
    std::fs::write(&path, bytes).expect("write resealed seed forgery");

    let error = NativeHnsw::load(&path, &store).expect_err("seed forgery must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("seed and parameters sample level")),
        "unexpected seed-attestation error: {error:?}"
    );
}

#[test]
fn owned_graph_load_rejects_store_cardinality_mismatch() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("graph.fshnsw");
    let store = TestStore::synthetic(16, 4);
    NativeHnsw::build(params(), 41, &store)
        .expect("build")
        .save(&path)
        .expect("save");
    let wrong_store = TestStore::synthetic(15, 4);

    let error =
        NativeHnsw::load(&path, &wrong_store).expect_err("store cardinality must bind load");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("vector store exposes 15 rows")),
        "unexpected cardinality error: {error:?}"
    );
}

#[cfg(unix)]
#[test]
fn owned_graph_load_and_save_reject_symbolic_links() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("temporary directory");
    let target = directory.path().join("target.fshnsw");
    let alias = directory.path().join("alias.fshnsw");
    let store = TestStore::synthetic(8, 4);
    let graph = NativeHnsw::build(params(), 29, &store).expect("build");
    graph.save(&target).expect("save target");
    symlink(&target, &alias).expect("create symlink");

    let load_error = NativeHnsw::load(&alias, &store).expect_err("load must reject symlink");
    assert!(matches!(load_error, SearchError::Io(_)));
    let save_error = graph.save(&alias).expect_err("save must reject symlink");
    assert!(matches!(save_error, SearchError::Io(_)));
    assert!(
        NativeHnsw::load(&target, &store).is_ok(),
        "rejecting the alias must not alter its target"
    );
}

#[test]
fn malformed_graph_refuses_save_before_touching_existing_artifact() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("graph.fshnsw");
    let sentinel = b"existing artifact must survive";
    std::fs::write(&path, sentinel).expect("write sentinel");
    let store = TestStore::synthetic(32, 4);
    let mut graph = NativeHnsw::build(params(), 17, &store).expect("build");
    let (id, layer, neighbour) = graph
        .adjacency
        .iter()
        .enumerate()
        .find_map(|(id, point)| {
            point
                .layers
                .iter()
                .enumerate()
                .find_map(|(layer, neighbours)| {
                    neighbours
                        .first()
                        .copied()
                        .map(|neighbour| (id, layer, neighbour))
                })
        })
        .expect("non-trivial graph edge");
    let id = u32::try_from(id).expect("test id");
    graph.adjacency[neighbour as usize].layers[layer].retain(|&candidate| candidate != id);

    let error = graph
        .save(&path)
        .expect_err("malformed graph must not publish");
    assert!(
        matches!(error, SearchError::IndexCorrupted { .. }),
        "unexpected save error: {error:?}"
    );
    assert_eq!(std::fs::read(path).expect("surviving bytes"), sentinel);
}

#[test]
fn owned_graph_save_atomically_replaces_complete_prior_generation() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("graph.fshnsw");
    let first_store = TestStore::synthetic(12, 4);
    let first = NativeHnsw::build(params(), 3, &first_store).expect("first build");
    let first_metadata = first.save(&path).expect("first save");

    let second_store = TestStore::synthetic(48, 4);
    let second = NativeHnsw::build(params(), 5, &second_store).expect("second build");
    let second_metadata = second.save(&path).expect("replacement save");
    let loaded = NativeHnsw::load(&path, &second_store).expect("load replacement");

    assert_ne!(first_metadata.point_count(), second_metadata.point_count());
    assert_eq!(second_metadata.point_count(), 48);
    assert_eq!(topology_snapshot(&loaded), topology_snapshot(&second));
}

// ─── bd-u3wt regression fixtures ────────────────────────────────────
//
// Each of these reproduces a defect the previous engine shipped.

/// bd-u3wt: two orthogonal vectors, k=2 — the original reproduction.
/// One direction returned only one result because the reverse edge was
/// stored at the wrong layer, leaving the other point unreachable.
#[test]
fn two_orthogonal_points_are_mutually_reachable() {
    let store = TestStore::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let graph = NativeHnsw::build(params(), 0x5eed, &store).expect("build");
    graph.verify().expect("structure");

    for query in [&[1.0_f32, 0.0], &[0.0_f32, 1.0]] {
        let hits = graph.search(query, 2, None, &store).expect("search");
        assert_eq!(
            hits.len(),
            2,
            "both points must be reachable from query {query:?}, got {hits:?}"
        );
    }
}

/// bd-u3wt defect (4): a `high, high, zero` level sequence left the
/// logical layer looking empty because participation was tested as
/// `level == layer` rather than `level >= layer`.
#[test]
fn high_high_zero_level_sequence_stays_searchable() {
    // Force the exact shape by driving the sampler's output through a
    // constructed graph rather than hoping the seed produces it.
    let store = TestStore::new(vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]]);
    // Seed 3 deterministically samples levels [1, 1, 0] for ids
    // [0, 1, 2] with this parameter set.
    let mut graph = NativeHnsw::new(params(), 3).expect("new");
    let first_level = graph.sampler.level_for(0);
    assert_eq!(first_level, 1);
    graph.adjacency.push(Adjacency::with_level(first_level));
    graph.entry = Some(0);
    graph.max_level = first_level;
    // Points 1 and 2 arrive at level 1 and level 0 respectively.
    for id in [1_u32, 2] {
        let level = graph.sampler.level_for(id);
        graph.adjacency.push(Adjacency::with_level(level));
        let selected = (0..id).collect::<Vec<_>>();
        for layer in 0..=level.min(graph.max_level) {
            let mut journal = MutationJournal::new(id as usize);
            graph
                .link(id, &selected, layer, &store, &mut journal)
                .expect("link");
        }
    }
    graph
        .verify()
        .expect("structure must hold for high,high,zero");

    let hits = graph.search(&[0.0, 1.0], 3, None, &store).expect("search");
    assert_eq!(hits.len(), 3, "every point must remain findable: {hits:?}");
}

/// bd-u3wt defect (3): an edge must never exist in a layer either
/// endpoint does not occupy. `verify` is the executable statement of
/// that invariant, so prove it actually rejects a violation rather
/// than passing vacuously.
#[test]
fn verify_rejects_an_edge_above_a_neighbours_level() {
    let store = TestStore::synthetic(32, 4);
    let mut graph = NativeHnsw::build(params(), 7, &store).expect("build");
    graph.verify().expect("a freshly built graph is sound");

    // Find a point that reaches a layer above some neighbour, then
    // forge exactly the edge the old engine used to create.
    let low = graph
        .adjacency
        .iter()
        .position(|point| point.level() == 0)
        .expect("some point sits at level 0");
    let high = graph
        .adjacency
        .iter()
        .position(|point| point.level() >= 1)
        .expect("some point reaches level 1");
    let low_id = u32::try_from(low).expect("test graph length fits u32");
    graph.adjacency[high].layers[1].push(low_id);

    let defect = graph
        .verify()
        .expect_err("an edge above the neighbour's level must be rejected");
    assert!(
        matches!(
            defect,
            GraphDefect::EdgeAboveNeighbourLevel { neighbour, .. }
                if neighbour == low_id
        ),
        "expected the layer-violation defect naming point {low}, got: {defect:?}"
    );
}

#[test]
fn verify_rejects_a_missing_reciprocal_edge() {
    let store = TestStore::synthetic(32, 4);
    let mut graph = NativeHnsw::build(params(), 17, &store).expect("build");
    graph.verify().expect("a freshly built graph is sound");

    let (id, layer, neighbour) = graph
        .adjacency
        .iter()
        .enumerate()
        .find_map(|(id, point)| {
            point
                .layers
                .iter()
                .enumerate()
                .find_map(|(layer, neighbours)| {
                    neighbours
                        .first()
                        .copied()
                        .map(|neighbour| (id, layer, neighbour))
                })
        })
        .expect("a non-trivial graph has an edge");
    let id = u32::try_from(id).expect("test graph length fits u32");
    graph.adjacency[neighbour as usize].layers[layer].retain(|&candidate| candidate != id);

    assert_eq!(
        graph
            .verify()
            .expect_err("a one-way edge must fail structural attestation"),
        GraphDefect::MissingReciprocalEdge {
            id,
            layer,
            neighbour,
        }
    );
}

#[test]
fn store_cardinality_is_part_of_attestation_and_search_admission() {
    let store = TestStore::synthetic(12, 4);
    let graph = NativeHnsw::build(params(), 23, &store).expect("build");
    graph.verify_for_store(&store).expect("matching store");

    for mismatched_count in [11usize, 13] {
        let mismatched = TestStore::synthetic(mismatched_count, 4);
        assert_eq!(
            graph
                .verify_for_store(&mismatched)
                .expect_err("mismatched store must fail attestation"),
            GraphDefect::StoreCardinalityMismatch {
                graph_points: 12,
                store_rows: mismatched_count,
            }
        );
        let error = graph
            .search(&[1.0, 0.0, 0.0, 0.0], 0, None, &mismatched)
            .expect_err("cardinality admission must precede the zero-k fast path");
        assert!(
            error.to_string().contains("vector store exposes"),
            "typed cardinality defect was not preserved: {error}"
        );
    }

    let empty = NativeHnsw::new(params(), 23).expect("empty graph");
    let error = empty
        .search(&[1.0, 0.0, 0.0, 0.0], 0, None, &store)
        .expect_err("cardinality admission must precede the empty-graph fast path");
    assert!(
        error.to_string().contains("graph indexes 0 points"),
        "empty mismatch was silently accepted: {error}"
    );
}

#[test]
fn insert_is_failure_atomic_at_every_distance_boundary() {
    let store = TestStore::synthetic(48, 8);
    let mut before = NativeHnsw::new(params(), 0x00a7_0b1c).expect("new");
    for id in 0..40u32 {
        before.insert(id, &store).expect("build prefix");
    }
    before.verify().expect("prefix graph");
    let before_topology = topology_snapshot(&before);

    let counting = FailingStore::new(&store, None);
    let mut expected = before.clone();
    expected.insert(40, &counting).expect("reference insertion");
    let distance_boundaries = counting.between_calls();
    assert!(
        distance_boundaries > params().m0,
        "fixture must exercise search, selection, and post-link pruning boundaries"
    );
    let expected_topology = topology_snapshot(&expected);

    for fail_at in 1..=distance_boundaries {
        let failing = FailingStore::new(&store, Some(fail_at));
        let mut trial = before.clone();
        let error = trial
            .insert(40, &failing)
            .expect_err("the selected distance boundary must fail");
        assert!(
            error
                .to_string()
                .contains("injected distance_between failure"),
            "unexpected error at distance boundary {fail_at}: {error}"
        );
        assert_eq!(
            topology_snapshot(&trial),
            before_topology,
            "distance failure {fail_at}/{distance_boundaries} changed graph topology, entry, \
             maximum level, or point count"
        );
        trial.verify().expect("rollback must leave a sound graph");

        trial
            .insert(40, &store)
            .expect("retry after a rolled-back failure must succeed");
        assert_eq!(
            topology_snapshot(&trial),
            expected_topology,
            "retry after distance failure {fail_at} did not reproduce the clean insertion"
        );
    }
}

// ─── Boundaries ─────────────────────────────────────────────────────

#[test]
fn point_count_boundaries_build_and_verify() {
    // n = 0,1,2,3 and around the degree budget m and m+1.
    for count in [0usize, 1, 2, 3, 8, 9, 17] {
        let store = TestStore::synthetic(count, 6);
        let graph =
            NativeHnsw::build(params(), 42, &store).expect("boundary-sized graph must build");
        graph.verify().expect("boundary-sized graph must verify");
        assert_eq!(graph.len(), count);

        let hits = graph
            .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4, None, &store)
            .expect("search");
        assert_eq!(
            hits.len(),
            count.min(4),
            "k is capped by the corpus size, not by graph structure"
        );
    }
}

#[test]
fn repeated_vectors_remain_connected_beyond_the_layer_zero_budget() {
    let config = HnswParams::default();
    let count = config.m0 + 2;
    let store = TestStore::new(vec![vec![1.0, 0.0]; count]);
    let graph = NativeHnsw::build(config, 42, &store).expect("build repeated vectors");
    graph
        .verify()
        .expect("equal embeddings must not disconnect a previously inserted row");
    let actual: Vec<_> = graph
        .search(&[1.0, 0.0], count, Some(count), &store)
        .expect("full-width repeated-vector search")
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(actual, store.exact_top_k(&[1.0, 0.0], count));
}

#[test]
fn tied_vectors_keep_every_insert_reachable_with_small_degree_and_beam_budgets() {
    let count = 65usize;
    let store = TestStore::new(
        (0..count)
            .map(|row| {
                (0..8)
                    .map(|axis| ((row * 7 + axis * 5) % 17) as f32)
                    .collect()
            })
            .collect(),
    );
    for m0 in [2, 3, 16] {
        let config = HnswParams {
            m: 2,
            m0,
            ef_construction: 1,
            ef_search: 1,
        };
        let mut graph = NativeHnsw::new(config, 17).expect("bounded-degree graph");
        for row in 0..u32::try_from(count).unwrap() {
            graph.insert(row, &store).expect("insert tied vector");
            graph.verify().unwrap_or_else(|error| {
                panic!("row {row}, m0={m0}: insertion lost connectivity or degree bound: {error}")
            });
        }
        for seed in 0..4 {
            let query = TestStore::query(seed, 8);
            let actual: Vec<_> = graph
                .search(&query, 7, Some(count), &store)
                .expect("full-width tied-vector search")
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(actual, store.exact_top_k(&query, 7));
        }
    }
}

#[test]
fn empty_graph_and_zero_k_are_benign() {
    let store = TestStore::synthetic(0, 4);
    let graph = NativeHnsw::build(params(), 1, &store).expect("build empty");
    assert!(graph.is_empty());
    assert_eq!(graph.entry_point(), None);
    assert!(
        graph
            .search(&[1.0, 0.0, 0.0, 0.0], 5, None, &store)
            .expect("search")
            .is_empty()
    );

    let store = TestStore::synthetic(10, 4);
    let graph = NativeHnsw::build(params(), 1, &store).expect("build");
    assert!(
        graph
            .search(&[1.0, 0.0, 0.0, 0.0], 0, None, &store)
            .expect("search")
            .is_empty()
    );
}

// ─── Differential recall against exact search ───────────────────────

/// The property that actually matters: the graph must find what an
/// exhaustive scan finds. Exact search is the oracle.
#[test]
fn recall_against_exact_search_is_high() {
    let store = TestStore::synthetic(500, 12);
    let graph = NativeHnsw::build(params(), 0xa11ce, &store).expect("build");
    graph.verify().expect("structure");

    let mut total = 0usize;
    let mut found = 0usize;
    for q in 0..25usize {
        let query = TestStore::query(q, 12);
        let expected = store.exact_top_k(&query, 10);
        let actual: Vec<u32> = graph
            .search(&query, 10, Some(128), &store)
            .expect("search")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        total += expected.len();
        found += expected.iter().filter(|id| actual.contains(id)).count();
    }
    let recall = found as f64 / total as f64;
    // Measured at 1.0 on this corpus; the bound leaves headroom for
    // tuning without letting a real regression through.
    assert!(
        recall >= 0.98,
        "recall@10 against exact search was {recall:.4} ({found}/{total}); the graph must \
         find what a full scan finds"
    );
}

/// A wide beam must reach exact agreement on a small corpus: with ef
/// at or above the corpus size there is nowhere for a true neighbour
/// to hide, so anything missing is a structural defect.
#[test]
fn exhaustive_beam_matches_exact_search_exactly() {
    let store = TestStore::synthetic(64, 8);
    let graph = NativeHnsw::build(params(), 0xb0b, &store).expect("build");
    graph.verify().expect("structure");

    for q in 0..10usize {
        let query = TestStore::query(q, 8);
        let expected = store.exact_top_k(&query, 5);
        let actual: Vec<u32> = graph
            .search(&query, 5, Some(64), &store)
            .expect("search")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            actual, expected,
            "with a full-width beam the graph must agree with exact search on query {q}"
        );
    }
}

// ─── Determinism and parameters ─────────────────────────────────────

#[test]
fn construction_is_deterministic_for_a_seed() {
    let store = TestStore::synthetic(120, 6);
    let first = NativeHnsw::build(params(), 99, &store).expect("build");
    let second = NativeHnsw::build(params(), 99, &store).expect("build");
    assert_eq!(first.max_level(), second.max_level());
    assert_eq!(first.entry_point(), second.entry_point());
    let point_count = u32::try_from(first.len()).expect("test graph length fits u32");
    for id in 0..point_count {
        for layer in 0..=first.max_level() {
            assert_eq!(
                first.neighbours_at(id, layer),
                second.neighbours_at(id, layer),
                "point {id} layer {layer} must be identical across builds"
            );
        }
    }
}

#[test]
fn insert_out_of_order_is_rejected() {
    let store = TestStore::synthetic(4, 4);
    let mut graph = NativeHnsw::new(params(), 1).expect("new");
    graph.insert(0, &store).expect("first row");
    let error = graph
        .insert(2, &store)
        .expect_err("skipping a row must be rejected");
    assert!(error.to_string().contains("row order"), "got: {error}");
}

#[test]
fn degenerate_parameters_are_rejected() {
    for (m, m0, efc, efs) in [
        (0, 16, 64, 64),
        (8, 0, 64, 64),
        (8, 1, 64, 64),
        (8, 16, 0, 64),
        (8, 16, 64, 0),
    ] {
        let params = HnswParams {
            m,
            m0,
            ef_construction: efc,
            ef_search: efs,
        };
        assert!(
            NativeHnsw::new(params, 1).is_err(),
            "degenerate params {params:?} must be rejected"
        );
    }
}

#[test]
fn sampled_levels_are_bounded_and_deterministic() {
    let sampler = LevelSampler::new(16, 12345);
    for id in 0..10_000u32 {
        let level = sampler.level_for(id);
        assert!(level < MAX_LEVEL, "level {level} for {id} exceeds the cap");
        assert_eq!(level, sampler.level_for(id), "sampling must be pure");
    }
}
