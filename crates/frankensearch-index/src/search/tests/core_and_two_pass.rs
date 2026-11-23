use std::collections::HashSet;
use std::fs;

use frankensearch_core::{PredicateFilter, SearchError};
use proptest::prelude::*;

use crate::{Quantization, VectorIndex};

use super::super::*;
use super::{create_rows, hit_ids, temp_index_path, write_index};

#[test]
#[allow(clippy::cast_sign_loss)] // deterministic non-negative fixture mixing
fn int8_two_pass_keep_all_matches_exact() {
    // With a multiplier large enough to retain every record, pass-1 keeps all
    // main vectors, so the exact f16 rescore must reproduce `search_top_k`
    // bit-for-bit — verifying the byte offsets, tombstone flag check, rescore,
    // and resolve are correct, independent of int8 selection quality.
    let path = temp_index_path("int8-two-pass-keepall");
    let dim = 8;
    let count = 300;
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut s =
                        (i as u64).wrapping_mul(2_654_435_761) ^ (j as u64).wrapping_mul(40_503);
                    s ^= s >> 13;
                    ((s & 0xffff) as f32 / 65_535.0) - 0.5
                })
                .collect()
        })
        .collect();
    let rows = create_rows(&vectors);
    let row_refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
        .collect();
    write_index(&path, &row_refs).expect("write index");
    let index = VectorIndex::open(&path).expect("open index");

    for qi in 0..8_usize {
        let query: Vec<f32> = (0..dim)
            .map(|j| (((qi * 7 + j * 3) % 11) as f32 / 11.0) - 0.5)
            .collect();
        let exact = index.search_top_k(&query, 10, None).expect("exact");
        // mult=50 → candidate_count clamps to `count` → pass-1 retains all.
        let approx = index
            .search_top_k_int8_two_pass(&query, 10, 50)
            .expect("int8 two-pass");
        let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
        let approx_ids: Vec<&str> = approx.iter().map(|h| h.doc_id.as_str()).collect();
        assert_eq!(
            exact_ids, approx_ids,
            "int8 two-pass (keep-all) must match exact search_top_k for query {qi}"
        );
    }
}

#[test]
#[allow(clippy::cast_sign_loss)] // deterministic non-negative fixture mixing
fn int8_row_block_matches_orig_with_tombstones_and_tail() {
    // Cross the parallel threshold and leave a three-row tail in the final
    // chunk. Tombstones in two different four-row blocks force the exact
    // per-row fallback while the remaining blocks use the fused x4 kernel.
    let path = temp_index_path("int8-row-block-orig-parity");
    let dim = 33;
    let count = 10_003;
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut s =
                        (i as u64).wrapping_mul(2_654_435_761) ^ (j as u64).wrapping_mul(40_503);
                    s ^= s >> 13;
                    ((s & 0xffff) as f32 / 65_535.0) - 0.5
                })
                .collect()
        })
        .collect();
    let rows = create_rows(&vectors);
    let row_refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
        .collect();
    write_index(&path, &row_refs).expect("write index");
    let mut index = VectorIndex::open(&path).expect("open index");
    index
        .soft_delete_batch(&["doc-001", "doc-004", "doc-4097", "doc-10002"])
        .expect("soft delete mixed rows");

    for qi in 0..3_usize {
        let query: Vec<f32> = (0..dim)
            .map(|j| (((qi * 11 + j * 7) % 29) as f32 / 29.0) - 0.5)
            .collect();
        for mult in [2, 3, 5, 10] {
            let orig = index
                .bench_search_top_k_int8_two_pass_orig(&query, 10, mult)
                .expect("original int8 two-pass");
            let candidate = index
                .bench_search_top_k_int8_two_pass_row_block_candidate(&query, 10, mult)
                .expect("row-blocked int8 two-pass");
            assert_eq!(candidate.len(), orig.len());
            for (candidate_hit, orig_hit) in candidate.iter().zip(&orig) {
                assert_eq!(candidate_hit.index, orig_hit.index, "qi={qi} mult={mult}");
                assert_eq!(candidate_hit.doc_id, orig_hit.doc_id, "qi={qi} mult={mult}");
                assert_eq!(
                    candidate_hit.score.to_bits(),
                    orig_hit.score.to_bits(),
                    "qi={qi} mult={mult}"
                );
            }
        }
    }
}

/// CI RECALL GUARD for the shipped maddubs pass-1 kernel (bd-b5wl). Unlike the row-block
/// candidate, the `vpmaddubs` pass-1 is APPROXIMATE (saturates on the quantizer's ±127 tail), so
/// it is *not* bit-identical to the exact int8 scan. What must hold — and what makes it safe to
/// ship as the default — is that it does not lose recall vs the exact-flat f32 top-k: the pass-1
/// still retains the true top-k into the candidate set, and the f16 rescore then orders them
/// exactly. Asserts maddubs recall@10 ≥ orig recall@10 and both are perfect on a realistic
/// (normalized, clustered) corpus that exercises the saturation.
#[test]
#[allow(clippy::cast_sign_loss)] // deterministic non-negative fixture mixing
fn int8_two_pass_maddubs_preserves_recall_vs_flat() {
    let path = temp_index_path("int8-maddubs-recall");
    let dim = 384;
    let count = 4_000;
    let normalize = |mut v: Vec<f32>| -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        v
    };
    // 16 clusters + jitter → realistic ANN corpus (tight top-k, real quantized magnitudes).
    let centroids: Vec<Vec<f32>> = (0..16)
        .map(|c| {
            normalize(
                (0..dim)
                    .map(|j| {
                        let mut s =
                            (c as u64 + 1).wrapping_mul(0x9e37) ^ (j as u64).wrapping_mul(40_503);
                        s ^= s >> 13;
                        ((s & 0xffff) as f32 / 65_535.0) - 0.5
                    })
                    .collect(),
            )
        })
        .collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            let c = &centroids[i % 16];
            normalize(
                (0..dim)
                    .map(|j| {
                        let mut s =
                            (i as u64 + 1).wrapping_mul(2_654_435_761) ^ (j as u64).wrapping_mul(7);
                        s ^= s >> 13;
                        c[j] + 0.15 * (((s & 0xffff) as f32 / 65_535.0) - 0.5)
                    })
                    .collect(),
            )
        })
        .collect();
    let rows = create_rows(&vectors);
    let row_refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
        .collect();
    write_index(&path, &row_refs).expect("write index");
    let index = VectorIndex::open(&path).expect("open index");

    let ids = |hits: Vec<VectorHit>| -> Vec<String> {
        hits.into_iter().map(|h| h.doc_id.to_string()).collect()
    };
    let recall = |exact: &[String], approx: &[String]| -> f64 {
        let hit = approx.iter().filter(|id| exact.contains(id)).count();
        hit as f64 / exact.len().max(1) as f64
    };

    for (c, centroid) in centroids.iter().take(4).enumerate() {
        let query = normalize(centroid.clone());
        let exact = ids(index.search_top_k(&query, 10, None).expect("flat"));
        for mult in [3usize, 5] {
            let orig = ids(index
                .bench_search_top_k_int8_two_pass_orig(&query, 10, mult)
                .expect("orig"));
            let maddubs = ids(index
                .bench_search_top_k_int8_two_pass_maddubs(&query, 10, mult)
                .expect("maddubs"));
            let orig_r = recall(&exact, &orig);
            let maddubs_r = recall(&exact, &maddubs);
            assert!(
                maddubs_r >= orig_r - 1e-9,
                "c={c} mult={mult}: maddubs recall {maddubs_r} < orig {orig_r}"
            );
            assert!(
                (maddubs_r - 1.0).abs() < 1e-9,
                "c={c} mult={mult}: maddubs recall {maddubs_r} != 1.0"
            );
        }
    }
}

#[test]
#[allow(clippy::cast_sign_loss)] // deterministic non-negative fixture mixing
fn four_bit_two_pass_keep_all_matches_exact() {
    // With a multiplier large enough to retain every record, the exact f16
    // rescore must reproduce `search_top_k` bit-for-bit — verifying the nibble
    // pack/unpack offsets, tombstone flag check, rescore, and resolve.
    let path = temp_index_path("4bit-two-pass-keepall");
    let dim = 70; // odd-ish, > 64, exercises packing + a partial last byte
    let count = 300;
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut s =
                        (i as u64).wrapping_mul(2_654_435_761) ^ (j as u64).wrapping_mul(40_503);
                    s ^= s >> 13;
                    ((s & 0xffff) as f32 / 65_535.0) - 0.5
                })
                .collect()
        })
        .collect();
    let rows = create_rows(&vectors);
    let row_refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
        .collect();
    write_index(&path, &row_refs).expect("write index");
    let index = VectorIndex::open(&path).expect("open index");

    for qi in 0..8_usize {
        let query: Vec<f32> = (0..dim)
            .map(|j| (((qi * 7 + j * 3) % 11) as f32 / 11.0) - 0.5)
            .collect();
        let exact = index.search_top_k(&query, 10, None).expect("exact");
        let approx = index
            .search_top_k_4bit_two_pass(&query, 10, 50)
            .expect("4bit two-pass");
        let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
        let approx_ids: Vec<&str> = approx.iter().map(|h| h.doc_id.as_str()).collect();
        assert_eq!(
            exact_ids, approx_ids,
            "4bit two-pass (keep-all) must match exact search_top_k for query {qi}"
        );
    }
}

proptest! {
    #[test]
    fn property_top_k_invariants_hold(
        vectors in prop::collection::vec(prop::collection::vec(-1.0_f32..1.0_f32, 4), 1..20),
        query in prop::collection::vec(-1.0_f32..1.0_f32, 4),
        limit in 1_usize..20,
    ) {
        let path = temp_index_path("prop-top-k");
        let rows = create_rows(&vectors);
        let row_refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
            .collect();
        // Skip test case if temp dir is unwritable (macOS CI runners can
        // hit PermissionDenied under heavy concurrent test load).
        prop_assume!(write_index(&path, &row_refs).is_ok());

        let index = VectorIndex::open(&path).expect("open index");
        let hits = index.search_top_k(&query, limit, None).expect("search");

        let expected_len = limit.min(vectors.len());
        prop_assert_eq!(hits.len(), expected_len);
        let mut seen_indices = HashSet::new();
        for hit in &hits {
            prop_assert!(seen_indices.insert(hit.index));
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn property_parallel_and_sequential_paths_match(
        vectors in prop::collection::vec(prop::collection::vec(-1.0_f32..1.0_f32, 4), 8..40),
        query in prop::collection::vec(-1.0_f32..1.0_f32, 4),
        limit in 1_usize..20,
    ) {
        let path = temp_index_path("prop-parallel");
        let rows = create_rows(&vectors);
        let row_refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
            .collect();
        prop_assume!(write_index(&path, &row_refs).is_ok());

        let index = VectorIndex::open(&path).expect("open index");
        let sequential = index
            .search_top_k_internal(&query, limit, None, usize::MAX, PARALLEL_CHUNK_SIZE, true)
            .expect("sequential search");
        let parallel = index
            .search_top_k_internal(&query, limit, None, 1, 4, true)
            .expect("parallel search");

        prop_assert_eq!(sequential.len(), parallel.len());
        for (left, right) in sequential.iter().zip(parallel.iter()) {
            prop_assert_eq!(&left.doc_id, &right.doc_id);
            prop_assert_eq!(left.index, right.index);
            prop_assert!((left.score - right.score).abs() <= 1e-6);
        }
        let _ = fs::remove_file(&path);
    }
}

#[test]
fn top_k_orders_by_score_descending() {
    let path = temp_index_path("top-k-order");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
            ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc-a");
    assert_eq!(hits[1].doc_id, "doc-b");
    assert!(hits[0].score >= hits[1].score);
}

#[test]
fn filter_excludes_matching_doc_id() {
    let path = temp_index_path("filter");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
            ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let filter = PredicateFilter::new("exclude-a", |doc_id| doc_id != "doc-a");
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, Some(&filter))
        .expect("search");

    assert_eq!(hits.len(), 2);
    assert!(hits.iter().all(|hit| hit.doc_id != "doc-a"));
}

#[test]
fn tombstoned_records_are_excluded_from_search() {
    let path = temp_index_path("tombstone-excluded");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
            ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let mut index = VectorIndex::open(&path).expect("open index");
    assert!(index.soft_delete("doc-a").expect("delete doc-a"));

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("search");

    assert_eq!(hits.len(), 2);
    assert!(hits.iter().all(|hit| hit.doc_id != "doc-a"));
}

#[test]
fn parallel_and_sequential_ignore_tombstones() {
    let path = temp_index_path("tombstone-parallel");
    let mut rows = Vec::new();
    for i in 0..96 {
        let score = f32::from(u16::try_from(96 - i).expect("test index must fit in u16"));
        rows.push((format!("doc-{i:03}"), vec![score, 0.0, 0.0, 0.0]));
    }
    let refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
        .collect();
    write_index(&path, &refs).expect("write index");

    let mut index = VectorIndex::open(&path).expect("open index");
    let deleted = index
        .soft_delete_batch(&["doc-000", "doc-001", "doc-002", "doc-003"])
        .expect("batch delete");
    assert_eq!(deleted, 4);

    let query = [1.0, 0.0, 0.0, 0.0];
    let sequential = index
        .search_top_k_internal(&query, 10, None, usize::MAX, PARALLEL_CHUNK_SIZE, true)
        .expect("sequential");
    let parallel = index
        .search_top_k_internal(&query, 10, None, 1, 8, true)
        .expect("parallel");

    assert_eq!(sequential.len(), parallel.len());
    let deleted_ids = ["doc-000", "doc-001", "doc-002", "doc-003"];
    assert!(
        sequential
            .iter()
            .all(|hit| !deleted_ids.contains(&hit.doc_id.as_str()))
    );
    assert!(
        parallel
            .iter()
            .all(|hit| !deleted_ids.contains(&hit.doc_id.as_str()))
    );
    for (left, right) in sequential.iter().zip(parallel.iter()) {
        assert_eq!(left.doc_id, right.doc_id);
        assert!((left.score - right.score).abs() < 1e-6);
    }
}

#[test]
fn parallel_and_sequential_paths_match() {
    let path = temp_index_path("parallel-match");
    let mut rows = Vec::new();
    for i in 0..64 {
        // Start at 1: a zero-norm row would be rejected by the writer gate.
        let rank = f32::from(u16::try_from(i + 1).expect("test index must fit in u16"));
        rows.push((
            format!("doc-{i:03}"),
            vec![rank, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ));
    }
    let refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
        .collect();
    write_index(&path, &refs).expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let sequential = index
        .search_top_k_internal(&query, 10, None, usize::MAX, PARALLEL_CHUNK_SIZE, true)
        .expect("sequential search");
    let parallel = index
        .search_top_k_internal(&query, 10, None, 1, 4, true)
        .expect("parallel search");

    assert_eq!(sequential.len(), parallel.len());
    for (left, right) in sequential.iter().zip(parallel.iter()) {
        assert_eq!(left.doc_id, right.doc_id);
        assert!((left.score - right.score).abs() < 1e-6);
    }
}

#[test]
fn parallel_and_sequential_paths_match_with_filter() {
    let path = temp_index_path("parallel-match-filter");
    let mut rows = Vec::new();
    for i in 0..96 {
        // Start at 1: a zero-norm row would be rejected by the writer gate.
        let rank = f32::from(u16::try_from(i + 1).expect("test index must fit in u16"));
        rows.push((
            format!("doc-{i:03}"),
            vec![rank, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ));
    }
    let refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
        .collect();
    write_index(&path, &refs).expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let filter = PredicateFilter::new("even-docs", |doc_id| {
        let suffix = doc_id.strip_prefix("doc-").unwrap_or_default();
        suffix.parse::<u32>().is_ok_and(|v| v % 2 == 0)
    });

    let sequential = index
        .search_top_k_internal(
            &query,
            15,
            Some(&filter),
            usize::MAX,
            PARALLEL_CHUNK_SIZE,
            true,
        )
        .expect("sequential search");
    let parallel = index
        .search_top_k_internal(&query, 15, Some(&filter), 1, 8, true)
        .expect("parallel search");

    assert_eq!(sequential.len(), parallel.len());
    for (left, right) in sequential.iter().zip(parallel.iter()) {
        assert_eq!(left.doc_id, right.doc_id);
        assert!((left.score - right.score).abs() < 1e-6);
    }
}

#[test]
fn resolves_doc_ids_only_for_winners() {
    let path = temp_index_path("two-phase");
    write_index(
        &path,
        &[("winner", vec![1.0, 0.0]), ("loser", vec![0.0, 1.0])],
    )
    .expect("write index");

    let inspect = VectorIndex::open(&path).expect("open index");
    let loser_idx = inspect
        .find_index_by_doc_hash(super::super::fnv1a_hash(b"loser"))
        .expect("loser index");
    let entry = inspect.record_at(loser_idx).expect("record");
    let loser_offset = inspect.strings_offset + usize::try_from(entry.doc_id_offset).unwrap_or(0);
    drop(inspect);

    let mut bytes = fs::read(&path).expect("read bytes");
    bytes[loser_offset] = 0xFF;
    fs::write(&path, bytes).expect("write corrupt bytes");

    let index = VectorIndex::open(&path).expect("open index");
    let hits = index
        .search_top_k(&[1.0, 0.0], 1, None)
        .expect("search should only resolve winner doc_id");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "winner");
}

#[test]
fn bitset_filter_skips_doc_id_decode_for_non_matching_records() {
    let path = temp_index_path("bitset-hash-fast-path");
    write_index(
        &path,
        &[("doc-a", vec![1.0, 0.0]), ("doc-b", vec![0.0, 1.0])],
    )
    .expect("write index");

    let inspect = VectorIndex::open(&path).expect("open index");
    let bad_idx = inspect
        .find_index_by_doc_hash(super::super::fnv1a_hash(b"doc-b"))
        .expect("doc-b index");
    let record = inspect.record_at(bad_idx).expect("record");
    let bad_offset =
        inspect.strings_offset + usize::try_from(record.doc_id_offset).expect("offset");
    drop(inspect);

    let mut bytes = fs::read(&path).expect("read bytes");
    bytes[bad_offset] = 0xFF;
    fs::write(&path, bytes).expect("write corrupt bytes");

    let index = VectorIndex::open(&path).expect("open index");
    let filter = frankensearch_core::BitsetFilter::from_doc_ids(["doc-a"]);
    let hits = index
        .search_top_k(&[1.0, 0.0], 10, Some(&filter))
        .expect("search should ignore corrupted filtered-out doc_id");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-a");
}

#[test]
fn selective_bitset_filter_uses_file_backed_gather() {
    let path = temp_index_path("fsvi-selective-gather");
    let rows: Vec<(String, Vec<f32>)> = (0..256)
        .map(|i| {
            let x = f32::from(u16::try_from(i % 31).expect("small test value")) / 31.0;
            let y = f32::from(u16::try_from(i / 31).expect("small test value")) / 10.0;
            (format!("doc-{i:03}"), vec![x, y, 0.25, 0.5])
        })
        .collect();
    let refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
        .collect();
    write_index(&path, &refs).expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let filter = frankensearch_core::BitsetFilter::from_doc_ids(["doc-003", "doc-097", "doc-203"]);
    let query = [0.7, 0.3, 0.0, 0.0];
    let scan = index
        .bench_scan_filtered(&query, 3, Some(&filter))
        .expect("forced scan");
    let gather = index
        .bench_gather_filtered(&query, 3, &filter)
        .expect("forced gather");
    let public = index
        .search_top_k(&query, 3, Some(&filter))
        .expect("public search");

    assert_eq!(hit_ids(&gather), hit_ids(&scan));
    assert_eq!(hit_ids(&public), hit_ids(&scan));
}

#[test]
fn limit_zero_or_empty_index_returns_no_hits() {
    let path = temp_index_path("limit-zero");
    let writer = VectorIndex::create_with_revision(&path, "hash", "test", 4, Quantization::F16)
        .expect("writer");
    writer.finish().expect("finish");

    let index = VectorIndex::open(&path).expect("open index");
    let zero_limit = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 0, None)
        .expect("search");
    let empty_index = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 5, None)
        .expect("search");

    assert!(zero_limit.is_empty());
    assert!(empty_index.is_empty());
}

// ── search_top_k_classified (bd-tqhc) ───────────────────────────────

#[test]
fn classified_distinguishes_k_zero_from_newly_created_empty() {
    let path = temp_index_path("classified-kzero-vs-empty");
    let writer = VectorIndex::create_with_revision(&path, "hash", "test", 4, Quantization::F16)
        .expect("writer");
    writer.finish().expect("finish");
    let index = VectorIndex::open(&path).expect("open index");

    let k_zero = index
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 0, None)
        .expect("k=0 search");
    assert!(k_zero.hits.is_empty());
    assert_eq!(
        k_zero.zero_signal,
        Some(ZeroSignalReason::CallerRequestedZeroK)
    );

    let empty = index
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
        .expect("empty-index search");
    assert!(empty.hits.is_empty());
    assert_eq!(
        empty.zero_signal,
        Some(ZeroSignalReason::NewlyCreatedEmpty),
        "an index that never held a record must classify as newly created, \
             not collapse into the k=0 shape"
    );
}

#[test]
fn classified_rejects_non_finite_query_like_ann_does() {
    let path = temp_index_path("classified-nonfinite-query");
    write_index(&path, &[("doc-a", vec![0.1, 0.0, 0.0, 0.0])]).expect("write index");
    let index = VectorIndex::open(&path).expect("open index");

    let err = index
        .search_top_k_classified(&[f32::NAN, 0.0, 0.0, 0.0], 5, None)
        .expect_err("non-finite query must fail closed");
    assert!(
        matches!(err, SearchError::InvalidConfig { ref field, .. } if field == "query"),
        "expected InvalidConfig on query, got: {err:?}"
    );

    // The unclassified lane keeps its legacy behavior (no error), which is
    // exactly why production flows through the classified lane.
    index
        .search_top_k(&[f32::NAN, 0.0, 0.0, 0.0], 5, None)
        .expect("legacy lane is unchanged");
}

#[test]
fn classified_zero_norm_query_is_typed_not_silent() {
    let path = temp_index_path("classified-zero-norm");
    write_index(&path, &[("doc-a", vec![0.1, 0.0, 0.0, 0.0])]).expect("write index");
    let index = VectorIndex::open(&path).expect("open index");

    let classified = index
        .search_top_k_classified(&[0.0, 0.0, 0.0, 0.0], 5, None)
        .expect("zero-norm search");
    assert!(classified.hits.is_empty());
    assert_eq!(
        classified.zero_signal,
        Some(ZeroSignalReason::ZeroNormQuery)
    );
}

#[test]
fn classified_filter_eliminating_all_is_distinct_from_empty_index() {
    let path = temp_index_path("classified-filter-all");
    write_index(
        &path,
        &[
            ("doc-a", vec![0.1, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");
    let index = VectorIndex::open(&path).expect("open index");

    let reject_all = PredicateFilter::new("reject-all", |_| false);
    let classified = index
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, Some(&reject_all))
        .expect("filtered search");
    assert!(classified.hits.is_empty());
    assert_eq!(
        classified.zero_signal,
        Some(ZeroSignalReason::FilterEliminatedAll)
    );
}

#[test]
fn classified_all_tombstoned_is_distinct_from_newly_created() {
    let path = temp_index_path("classified-all-tombstoned");
    write_index(
        &path,
        &[
            ("doc-a", vec![0.1, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");
    let mut index = VectorIndex::open(&path).expect("open index");
    index.soft_delete("doc-a").expect("tombstone doc-a");
    index.soft_delete("doc-b").expect("tombstone doc-b");

    let classified = index
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
        .expect("search over tombstoned index");
    assert!(classified.hits.is_empty());
    assert_eq!(
        classified.zero_signal,
        Some(ZeroSignalReason::AllTombstoned)
    );
}

#[test]
fn classified_nonempty_result_carries_no_reason() {
    let path = temp_index_path("classified-nonempty");
    write_index(&path, &[("doc-a", vec![0.1, 0.0, 0.0, 0.0])]).expect("write index");
    let index = VectorIndex::open(&path).expect("open index");

    let classified = index
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
        .expect("search");
    assert_eq!(classified.hits.len(), 1);
    assert_eq!(classified.zero_signal, None);
}

#[test]
fn classified_in_memory_parity_on_request_scoped_states() {
    // The bead requires equivalent states to classify identically across
    // backends. Request-scoped states are representable in both; index
    // states diverge only where a state is structurally unrepresentable
    // in memory (tombstones compact away at load).
    let path = temp_index_path("classified-parity");
    write_index(
        &path,
        &[
            ("doc-a", vec![0.1, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");
    let file_backed = VectorIndex::open_read_only(&path).expect("open index");
    let in_memory = crate::InMemoryVectorIndex::from_fsvi(&path).expect("load in-memory copy");

    // k = 0.
    let file_k0 = file_backed
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 0, None)
        .expect("file k=0");
    let mem_k0 = in_memory
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 0, None)
        .expect("mem k=0");
    assert_eq!(file_k0.zero_signal, mem_k0.zero_signal);

    // Zero-norm query.
    let file_zero = file_backed
        .search_top_k_classified(&[0.0; 4], 5, None)
        .expect("file zero-norm");
    let mem_zero = in_memory
        .search_top_k_classified(&[0.0; 4], 5, None)
        .expect("mem zero-norm");
    assert_eq!(file_zero.zero_signal, mem_zero.zero_signal);

    // Non-finite query errors on both.
    assert!(
        file_backed
            .search_top_k_classified(&[f32::INFINITY, 0.0, 0.0, 0.0], 5, None)
            .is_err()
    );
    assert!(
        in_memory
            .search_top_k_classified(&[f32::INFINITY, 0.0, 0.0, 0.0], 5, None)
            .is_err()
    );

    // Filter excludes all.
    let reject_all = PredicateFilter::new("reject-all", |_| false);
    let file_filtered = file_backed
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, Some(&reject_all))
        .expect("file filter-all");
    let mem_filtered = in_memory
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, Some(&reject_all))
        .expect("mem filter-all");
    assert_eq!(file_filtered.zero_signal, mem_filtered.zero_signal);

    // Non-empty carries no reason on either.
    let file_hits = file_backed
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
        .expect("file search");
    let mem_hits = in_memory
        .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
        .expect("mem search");
    assert_eq!(file_hits.zero_signal, None);
    assert_eq!(mem_hits.zero_signal, None);
}

#[test]
fn k_above_record_count_returns_all_hits() {
    let path = temp_index_path("k-above-count");
    write_index(
        &path,
        &[
            ("doc-a", vec![0.1, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 20, None)
        .expect("search");
    assert_eq!(hits.len(), 2);
}

#[test]
fn full_recall_collect_all_matches_heap_prefix_main_only() {
    let path = temp_index_path("collect-all-main");
    let mut rows = Vec::new();
    for i in 0..80 {
        let score = f32::from(u16::try_from(80 - i).expect("test index must fit in u16"));
        rows.push((format!("doc-{i:03}"), vec![score, 0.0, 0.0, 0.0]));
    }
    let refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
        .collect();
    write_index(&path, &refs).expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let query = [1.0, 0.0, 0.0, 0.0];
    let total = index.record_count();
    let heap_limit = total.saturating_sub(7);

    let collect_all = index
        .search_top_k_internal(&query, total, None, 1, 8, true)
        .expect("collect-all");
    let heap_top = index
        .search_top_k_internal(
            &query,
            heap_limit,
            None,
            usize::MAX,
            PARALLEL_CHUNK_SIZE,
            true,
        )
        .expect("heap-top");

    assert_eq!(collect_all.len(), total);
    assert_eq!(heap_top.len(), heap_limit);
    for (heap_hit, full_hit) in heap_top.iter().zip(collect_all.iter()) {
        assert_eq!(heap_hit.doc_id, full_hit.doc_id);
        assert_eq!(heap_hit.index, full_hit.index);
        assert!((heap_hit.score - full_hit.score).abs() < 1e-6);
    }
}

#[test]
fn full_recall_collect_all_matches_heap_prefix_with_wal() {
    let path = temp_index_path("collect-all-wal");
    let mut rows = Vec::new();
    for i in 0..48 {
        let score = f32::from(u16::try_from(48 - i).expect("test index must fit in u16"));
        rows.push((format!("doc-{i:03}"), vec![score, 0.0, 0.0, 0.0]));
    }
    let refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
        .collect();
    write_index(&path, &refs).expect("write index");

    let mut index = VectorIndex::open(&path).expect("open index");
    index
        .append_batch(&[
            ("wal-top".to_owned(), vec![200.0, 0.0, 0.0, 0.0]),
            ("wal-mid".to_owned(), vec![24.5, 0.0, 0.0, 0.0]),
            ("wal-tail".to_owned(), vec![-1.0, 0.0, 0.0, 0.0]),
        ])
        .expect("append wal batch");

    let query = [1.0, 0.0, 0.0, 0.0];
    let total = index
        .record_count()
        .saturating_add(index.wal_record_count());
    let heap_limit = total.saturating_sub(5);

    let collect_all = index
        .search_top_k_internal(&query, total.saturating_add(10), None, 1, 8, true)
        .expect("collect-all with wal");
    let heap_top = index
        .search_top_k_internal(
            &query,
            heap_limit,
            None,
            usize::MAX,
            PARALLEL_CHUNK_SIZE,
            true,
        )
        .expect("heap-top with wal");

    assert_eq!(collect_all.len(), total);
    assert_eq!(collect_all[0].doc_id, "wal-top");
    assert_eq!(heap_top.len(), heap_limit);
    for (heap_hit, full_hit) in heap_top.iter().zip(collect_all.iter()) {
        assert_eq!(heap_hit.doc_id, full_hit.doc_id);
        assert_eq!(heap_hit.index, full_hit.index);
        assert!((heap_hit.score - full_hit.score).abs() < 1e-6);
    }
}

#[test]
fn ties_are_broken_by_index() {
    let path = temp_index_path("tie-break");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-c", vec![1.0, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 3, None)
        .expect("search");

    let mut indexes: Vec<u32> = hits.iter().map(|hit| hit.index).collect();
    let mut sorted = indexes.clone();
    sorted.sort_unstable();
    assert_eq!(indexes, sorted);
    assert_eq!(hits.len(), 3);
    indexes.clear();
}

#[test]
fn nan_scores_do_not_panic_and_sort_last() {
    let path = temp_index_path("nan-safe");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.5, 0.0, 0.0, 0.0]),
            ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let hits = index
        .search_top_k(&[f32::NAN, 0.0, 0.0, 0.0], 3, None)
        .expect("search");

    assert_eq!(hits.len(), 3);
    assert!(hits.iter().all(|hit| hit.score.is_nan()));
    assert!(hits.windows(2).all(|pair| pair[0].index <= pair[1].index));
}
