use std::fs;

use frankensearch_core::{PredicateFilter, SearchError};

use crate::wal::to_wal_index;
use crate::{Quantization, VectorIndex};

use super::super::*;
use super::{temp_index_path, write_index};

#[test]
fn parse_parallel_search_env_values() {
    assert!(parse_parallel_search_env(None));
    assert!(parse_parallel_search_env(Some("1")));
    assert!(parse_parallel_search_env(Some("true")));
    assert!(parse_parallel_search_env(Some("yes")));
    assert!(!parse_parallel_search_env(Some("0")));
    assert!(!parse_parallel_search_env(Some("false")));
    assert!(!parse_parallel_search_env(Some("no")));
    assert!(!parse_parallel_search_env(Some("off")));
}

// --- SearchFilter integration tests ---

#[test]
fn bitset_filter_during_vector_search() {
    let path = temp_index_path("bitset-filter");
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
    let filter = frankensearch_core::BitsetFilter::from_doc_ids(["doc-a", "doc-c"]);
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter))
        .expect("search");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc-a");
    assert_eq!(hits[1].doc_id, "doc-c");
}

#[test]
fn filter_chain_and_semantics_in_search() {
    let path = temp_index_path("filter-chain-and");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
            ("doc-c", vec![0.6, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let chain = frankensearch_core::FilterChain::new(frankensearch_core::FilterMode::All)
        .with(Box::new(PredicateFilter::new("not-c", |id| id != "doc-c")))
        .with(Box::new(PredicateFilter::new("not-a", |id| id != "doc-a")));

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&chain))
        .expect("search");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-b");
}

#[test]
fn filter_chain_or_semantics_in_search() {
    let path = temp_index_path("filter-chain-or");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
            ("doc-c", vec![0.6, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let chain = frankensearch_core::FilterChain::new(frankensearch_core::FilterMode::Any)
        .with(Box::new(PredicateFilter::new("is-a", |id| id == "doc-a")))
        .with(Box::new(PredicateFilter::new("is-c", |id| id == "doc-c")));

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&chain))
        .expect("search");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc-a");
    assert_eq!(hits[1].doc_id, "doc-c");
}

#[test]
fn filter_rejects_all_returns_empty() {
    let path = temp_index_path("filter-all-rejected");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let filter = PredicateFilter::new("reject-all", |_| false);
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter))
        .expect("search");

    assert!(hits.is_empty());
}

#[test]
fn filter_applies_to_wal_entries() {
    let path = temp_index_path("filter-wal");
    write_index(&path, &[("doc-a", vec![1.0, 0.0, 0.0, 0.0])]).expect("write index");

    let mut index = VectorIndex::open(&path).expect("open index");
    index
        .append("doc-b", &[0.9, 0.0, 0.0, 0.0])
        .expect("append doc-b");
    index
        .append("doc-c", &[0.8, 0.0, 0.0, 0.0])
        .expect("append doc-c");

    // Filter to only include doc-b (WAL entry).
    let filter = PredicateFilter::new("only-b", |id| id == "doc-b");
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter))
        .expect("search");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-b");
}

#[test]
fn filter_works_with_wal_and_main_combined() {
    let path = temp_index_path("filter-wal-main");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.5, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let mut index = VectorIndex::open(&path).expect("open index");
    index
        .append("doc-c", &[0.9, 0.0, 0.0, 0.0])
        .expect("append doc-c");

    // Filter includes doc-a (main) and doc-c (WAL), excludes doc-b (main).
    let filter = frankensearch_core::BitsetFilter::from_doc_ids(["doc-a", "doc-c"]);
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter))
        .expect("search");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc-a");
    assert_eq!(hits[1].doc_id, "doc-c");
}

#[test]
fn all_records_soft_deleted_returns_empty() {
    let path = temp_index_path("all-deleted");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
            ("doc-c", vec![0.5, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let mut index = VectorIndex::open(&path).expect("open index");
    let deleted = index
        .soft_delete_batch(&["doc-a", "doc-b", "doc-c"])
        .expect("batch delete");
    assert_eq!(deleted, 3);

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert!(
        hits.is_empty(),
        "search over fully-deleted index should return empty"
    );
}

#[test]
fn dimension_mismatch_returns_error() {
    let path = temp_index_path("dim-mismatch");
    write_index(&path, &[("doc-a", vec![1.0, 0.0, 0.0, 0.0])]).expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let result = index.search_top_k(&[1.0, 0.0], 5, None);
    assert!(matches!(
        result,
        Err(SearchError::DimensionMismatch {
            expected: 4,
            found: 2
        })
    ));
}

#[test]
fn wal_only_search_returns_wal_entries() {
    let path = temp_index_path("wal-only");
    let writer = VectorIndex::create_with_revision(&path, "hash", "test", 4, Quantization::F16)
        .expect("writer");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open index");
    assert_eq!(index.record_count(), 0);

    index
        .append("wal-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("append wal-a");
    index
        .append("wal-b", &[0.5, 0.0, 0.0, 0.0])
        .expect("append wal-b");

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 5, None)
        .expect("search");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "wal-a");
    assert_eq!(hits[1].doc_id, "wal-b");
    assert!(hits[0].score >= hits[1].score);
}

#[test]
fn wal_entries_can_outrank_main_index() {
    let path = temp_index_path("wal-outranks-main");
    write_index(
        &path,
        &[
            ("main-a", vec![0.3, 0.0, 0.0, 0.0]),
            ("main-b", vec![0.2, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let mut index = VectorIndex::open(&path).expect("open index");
    index
        .append("wal-top", &[1.0, 0.0, 0.0, 0.0])
        .expect("append wal-top");

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 3, None)
        .expect("search");

    assert_eq!(hits.len(), 3);
    assert_eq!(
        hits[0].doc_id, "wal-top",
        "WAL entry with highest score should rank first"
    );
    assert!(hits[0].score >= hits[1].score);
    assert!(hits[1].score >= hits[2].score);
}

#[test]
fn stale_main_entry_shadowed_by_wal() {
    let path = temp_index_path("stale-shadow");
    // Create main index with [1.0, 0.0]
    let mut writer =
        VectorIndex::create_with_revision(&path, "test", "r1", 2, Quantization::F32).unwrap();
    writer.write_record("doc-a", &[1.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    // Append doc-a with [0.0, 1.0] to WAL
    index.append("doc-a", &[0.0, 1.0]).unwrap();

    // Search for [1.0, 0.0]. The WAL entry scores 0.0, the Main entry scores 1.0.
    // If the bug exists, the Main entry will be returned instead of the WAL entry,
    // because the WAL entry (score 0.0) might not make it into the top-K heap
    // if there are other candidates, or it just gets omitted if K is small.
    let hits = index.search_top_k(&[1.0, 0.0], 1, None).unwrap();

    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].score.abs() < f32::EPSILON,
        "Expected score 0.0 from WAL entry, but got leaked score {}",
        hits[0].score
    );
}

#[test]
fn wal_index_marker_out_of_bounds_returns_error() {
    let path = temp_index_path("wal-oob-index-marker");
    write_index(&path, &[("main-a", vec![1.0, 0.0, 0.0, 0.0])]).expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let fabricated = HeapEntry::new(to_wal_index(42), 1.0);
    let err = index
        .resolve_wal_hit(&fabricated)
        .expect_err("fabricated WAL marker should fail bounds check");
    assert!(matches!(err, SearchError::IndexCorrupted { .. }));
}

#[test]
fn heap_entry_nan_sorted_below_finite_scores() {
    let nan_entry = HeapEntry::new(0, f32::NAN);
    let finite_entry = HeapEntry::new(1, 0.5);
    // candidate_is_better: finite should beat NaN
    assert!(candidate_is_better(finite_entry, nan_entry));
    assert!(!candidate_is_better(nan_entry, finite_entry));
}

#[test]
fn heap_entry_equal_scores_tiebreak_by_index() {
    let left = HeapEntry::new(3, 0.5);
    let right = HeapEntry::new(7, 0.5);
    // Lower index wins the tiebreak
    assert!(candidate_is_better(left, right));
    assert!(!candidate_is_better(right, left));
}

#[test]
fn int8_heap_keys_match_legacy_score_and_index_order() {
    let exact_scores = [-16_774_160, -1, 0, 1, 16_774_160];
    let indices = [0, 17, usize::MAX];
    for &left_score in &exact_scores {
        for &left_index in &indices {
            for &right_score in &exact_scores {
                for &right_index in &indices {
                    let packed = int8_heap_key(left_index, left_score)
                        .cmp(&int8_heap_key(right_index, right_score));
                    let legacy = HeapEntry::new(left_index, left_score as f32)
                        .cmp(&HeapEntry::new(right_index, right_score as f32));
                    assert_eq!(packed, legacy);
                }
            }
            assert_eq!(
                int8_heap_index(int8_heap_key(left_index, left_score)),
                left_index
            );
        }
    }

    // Above the consecutive-integer f32 range, distinct i32 dots can collapse
    // to the same float. The fallback must preserve that legacy tie behavior.
    let fallback_scores = [i32::MIN, -16_777_217, 16_777_216, 16_777_217, i32::MAX];
    for &left_score in &fallback_scores {
        for &right_score in &fallback_scores {
            let packed = int8_heap_key_from_f32(11, left_score)
                .cmp(&int8_heap_key_from_f32(29, right_score));
            let legacy =
                HeapEntry::new(11, left_score as f32).cmp(&HeapEntry::new(29, right_score as f32));
            assert_eq!(packed, legacy);
        }
    }
}

#[test]
fn packed_int8_merge_matches_legacy_heap_merge() {
    let partitions = [
        [(0_usize, 90_i32), (1, 40), (2, 40)],
        [(3, 100), (4, -5), (5, 40)],
        [(6, 70), (7, 40), (8, i32::MIN)],
    ];
    for limit in [0, 1, 3, 9] {
        let legacy_parts = partitions
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&(index, score)| HeapEntry::new(index, score as f32))
                    .collect::<BinaryHeap<_>>()
            })
            .collect();
        let packed_parts = partitions
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&(index, score)| int8_heap_key(index, score))
                    .collect::<BinaryHeap<_>>()
            })
            .collect();

        let mut legacy_indices = merge_partial_heaps(legacy_parts, limit)
            .into_iter()
            .map(|entry| entry.index)
            .collect::<Vec<_>>();
        let mut packed_indices = merge_int8_partial_heaps(packed_parts, limit)
            .into_iter()
            .map(int8_heap_index)
            .collect::<Vec<_>>();
        legacy_indices.sort_unstable();
        packed_indices.sort_unstable();
        assert_eq!(packed_indices, legacy_indices);
    }
}

#[test]
fn insert_candidate_with_limit_zero_is_noop() {
    let mut heap = BinaryHeap::new();
    insert_candidate(&mut heap, HeapEntry::new(0, 0.9), 0);
    assert!(heap.is_empty());
}

#[test]
fn insert_candidate_evicts_worst_when_full() {
    let mut heap = BinaryHeap::new();
    insert_candidate(&mut heap, HeapEntry::new(0, 0.1), 2);
    insert_candidate(&mut heap, HeapEntry::new(1, 0.5), 2);
    // Heap is full (limit=2). Insert better candidate.
    insert_candidate(&mut heap, HeapEntry::new(2, 0.9), 2);

    let entries: Vec<HeapEntry> = heap.into_vec();
    assert_eq!(entries.len(), 2);
    let scores: Vec<f32> = entries.iter().map(|e| e.score).collect();
    // Top 3: 0.9, 0.7, 0.5 — the 0.1 should be evicted
    assert!(scores.contains(&0.9));
    assert!(scores.contains(&0.5));
    assert!(!scores.contains(&0.1));
}

#[test]
fn insert_candidate_rejects_worse_when_full() {
    let mut heap = BinaryHeap::new();
    insert_candidate(&mut heap, HeapEntry::new(0, 0.5), 2);
    insert_candidate(&mut heap, HeapEntry::new(1, 0.9), 2);
    // Heap is full. Insert worse candidate — should be rejected.
    insert_candidate(&mut heap, HeapEntry::new(2, 0.1), 2);

    let entries: Vec<HeapEntry> = heap.into_vec();
    assert_eq!(entries.len(), 2);
    let scores: Vec<f32> = entries.iter().map(|e| e.score).collect();
    assert!(scores.contains(&0.9));
    assert!(scores.contains(&0.5));
    assert!(!scores.contains(&0.1));
}

#[test]
fn merge_partial_heaps_preserves_top_k() {
    let mut heap_a = BinaryHeap::new();
    insert_candidate(&mut heap_a, HeapEntry::new(0, 0.9), 3);
    insert_candidate(&mut heap_a, HeapEntry::new(1, 0.1), 3);

    let mut heap_b = BinaryHeap::new();
    insert_candidate(&mut heap_b, HeapEntry::new(2, 0.7), 3);
    insert_candidate(&mut heap_b, HeapEntry::new(3, 0.5), 3);

    let merged = merge_partial_heaps(vec![heap_a, heap_b], 3);
    let entries: Vec<HeapEntry> = merged.into_vec();
    assert_eq!(entries.len(), 3);
    let scores: Vec<f32> = entries.iter().map(|e| e.score).collect();
    // Top 3: 0.9, 0.7, 0.5 — the 0.1 should be evicted
    assert!(scores.contains(&0.9));
    assert!(scores.contains(&0.7));
    assert!(scores.contains(&0.5));
    assert!(!scores.contains(&0.1));
}

#[test]
fn parse_parallel_search_env_case_insensitive() {
    assert!(!parse_parallel_search_env(Some("OFF")));
    assert!(!parse_parallel_search_env(Some("False")));
    assert!(!parse_parallel_search_env(Some("NO")));
    assert!(!parse_parallel_search_env(Some("  off  ")));
}

#[test]
fn parse_parallel_search_env_empty_string_enables() {
    // Empty string is not any of the disable values, so parallel stays enabled.
    assert!(parse_parallel_search_env(Some("")));
    assert!(parse_parallel_search_env(Some("  ")));
}

#[test]
fn parallel_filter_path_propagates_doc_id_errors() {
    let path = temp_index_path("parallel-filter-errors");
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
    let bad_offset = inspect.strings_offset + usize::try_from(record.doc_id_offset).unwrap_or(0);
    drop(inspect);

    let mut bytes = fs::read(&path).expect("read index bytes");
    bytes[bad_offset] = 0xFF;
    fs::write(&path, bytes).expect("write corrupt bytes");

    let index = VectorIndex::open(&path).expect("reopen index");
    let filter = PredicateFilter::new("allow-all", |_| true);
    let query = [1.0, 0.0];

    let sequential = index.search_top_k_internal(&query, 1, Some(&filter), usize::MAX, 2, true);
    let parallel = index.search_top_k_internal(&query, 1, Some(&filter), 1, 2, true);

    assert!(
        sequential.is_err(),
        "sequential path should surface doc_id errors"
    );
    assert!(
        parallel.is_err(),
        "parallel path should surface doc_id errors"
    );
}

// --- SearchParams tests ---

#[test]
fn search_params_default_matches_constants() {
    let params = SearchParams::default();
    assert_eq!(params.parallel_threshold, PARALLEL_THRESHOLD);
    assert_eq!(params.parallel_chunk_size, PARALLEL_CHUNK_SIZE);
}

#[test]
fn search_top_k_with_params_matches_default_search() {
    let path = temp_index_path("with-params-default");
    let mut rows = Vec::new();
    for i in 0..32 {
        // Start at 1: a zero-norm row would be rejected by the writer gate.
        let rank = f32::from(u16::try_from(i + 1).expect("fits u16"));
        rows.push((format!("doc-{i:03}"), vec![rank, 0.0, 0.0, 0.0]));
    }
    let refs: Vec<(&str, Vec<f32>)> = rows
        .iter()
        .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
        .collect();
    write_index(&path, &refs).expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let query = [1.0, 0.0, 0.0, 0.0];

    let default_hits = index.search_top_k(&query, 5, None).expect("default search");
    let params_hits = index
        .search_top_k_with_params(&query, 5, None, SearchParams::default())
        .expect("params search");

    assert_eq!(default_hits.len(), params_hits.len());
    for (left, right) in default_hits.iter().zip(params_hits.iter()) {
        assert_eq!(left.doc_id, right.doc_id);
        assert_eq!(left.index, right.index);
        assert!((left.score - right.score).abs() < 1e-6);
    }
}

#[test]
fn search_top_k_with_params_custom_threshold() {
    let path = temp_index_path("with-params-custom");
    let mut rows = Vec::new();
    for i in 0..64 {
        // Start at 1: a zero-norm row would be rejected by the writer gate.
        let rank = f32::from(u16::try_from(i + 1).expect("fits u16"));
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
            10,
            Some(&filter),
            usize::MAX,
            PARALLEL_CHUNK_SIZE,
            true,
        )
        .expect("sequential search");
    let parallel = index
        .search_top_k_internal(&query, 10, Some(&filter), 1, 8, true)
        .expect("parallel search");

    assert_eq!(sequential.len(), parallel.len());
    for (left, right) in sequential.iter().zip(parallel.iter()) {
        assert_eq!(left.doc_id, right.doc_id);
        assert!((left.score - right.score).abs() < 1e-6);
    }
}

#[test]
fn search_top_k_with_params_disabled_parallel() {
    let path = temp_index_path("with-params-disabled");
    write_index(
        &path,
        &[
            ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
            ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write index");

    let index = VectorIndex::open(&path).expect("open index");
    let query = [1.0, 0.0, 0.0, 0.0];

    let params = SearchParams {
        parallel_threshold: 1, // would trigger parallel...
        parallel_chunk_size: 1,
        parallel_enabled: false, // ...but disabled
    };
    let hits = index
        .search_top_k_with_params(&query, 2, None, params)
        .expect("search with disabled parallel");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc-a");
    assert_eq!(hits[1].doc_id, "doc-b");
}

// ─── bd-2k3d tests begin ──────────────────────────────────────────

#[test]
fn search_params_debug_clone_copy() {
    let params = SearchParams {
        parallel_threshold: 100,
        parallel_chunk_size: 32,
        parallel_enabled: true,
    };
    let debug = format!("{params:?}");
    assert!(debug.contains("SearchParams"));
    assert!(debug.contains("100"));

    let copied: SearchParams = params;
    assert_eq!(copied.parallel_threshold, 100);
    assert_eq!(copied.parallel_chunk_size, 32);

    let cloned = params;
    assert!(cloned.parallel_enabled);
}

#[test]
fn compare_best_first_higher_score_wins() {
    let a = HeapEntry::new(0, 0.9);
    let b = HeapEntry::new(1, 0.5);
    assert_eq!(compare_best_first(&a, &b), Ordering::Less);
    assert_eq!(compare_best_first(&b, &a), Ordering::Greater);
}

#[test]
fn compare_best_first_equal_scores_tiebreak_by_index() {
    let a = HeapEntry::new(2, 0.7);
    let b = HeapEntry::new(5, 0.7);
    assert_eq!(compare_best_first(&a, &b), Ordering::Less);
}

#[test]
fn candidate_is_better_with_nan() {
    let good = HeapEntry::new(0, 0.5);
    let nan_entry = HeapEntry::new(1, f32::NAN);
    assert!(candidate_is_better(good, nan_entry));
    assert!(!candidate_is_better(nan_entry, good));
}

#[test]
fn candidate_is_better_equal_scores_lower_index_wins() {
    let a = HeapEntry::new(3, 0.8);
    let b = HeapEntry::new(7, 0.8);
    assert!(candidate_is_better(a, b));
    assert!(!candidate_is_better(b, a));
}

#[test]
fn score_key_maps_nan_to_neg_infinity() {
    assert_eq!(score_key(f32::NAN).to_bits(), f32::NEG_INFINITY.to_bits());
    assert!((score_key(0.5) - 0.5).abs() < f32::EPSILON);
    assert!((score_key(-0.3) - (-0.3)).abs() < f32::EPSILON);
    assert!((score_key(0.0)).abs() < f32::EPSILON);
}

#[test]
fn merge_partial_heaps_empty_list() {
    let merged = merge_partial_heaps(vec![], 10);
    assert!(merged.is_empty());
}

#[test]
fn merge_partial_heaps_single_heap() {
    let mut h = BinaryHeap::new();
    h.push(HeapEntry::new(0, 0.9));
    h.push(HeapEntry::new(1, 0.5));
    let merged = merge_partial_heaps(vec![h], 10);
    assert_eq!(merged.len(), 2);
}

#[test]
fn search_f32_quantization_index() {
    let path = temp_index_path("f32-quant-search");
    let dim = 4;
    let mut writer =
        VectorIndex::create_with_revision(&path, "test", "r1", dim, Quantization::F32).unwrap();
    writer.write_record("doc-a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.write_record("doc-b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
    writer.write_record("doc-c", &[0.5, 0.5, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    assert_eq!(index.quantization(), Quantization::F32);

    let query = [1.0, 0.0, 0.0, 0.0];
    let hits = index.search_top_k(&query, 2, None).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc-a");

    fs::remove_file(&path).ok();
}

#[test]
fn search_f32_with_filter() {
    let path = temp_index_path("f32-filter");
    let dim = 4;
    let mut writer =
        VectorIndex::create_with_revision(&path, "test", "r1", dim, Quantization::F32).unwrap();
    writer.write_record("doc-a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.write_record("doc-b", &[0.9, 0.1, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let filter = PredicateFilter::new("only-b", |doc_id: &str| doc_id == "doc-b");
    let query = [1.0, 0.0, 0.0, 0.0];
    let hits = index.search_top_k(&query, 10, Some(&filter)).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-b");

    fs::remove_file(&path).ok();
}

// ─── bd-2k3d tests end ────────────────────────────────────────────
