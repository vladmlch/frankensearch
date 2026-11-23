#![allow(clippy::wildcard_imports)]

use super::*;

#[test]
fn round_trip_f16_with_revision_and_lookup() {
    let path = temp_index_path("round-trip");
    let mut writer =
        VectorIndex::create_with_revision(&path, "fnv1a-384", "rev-123", 8, Quantization::F16)
            .expect("writer");
    writer
        .write_record("doc-b", &sample_vector(1.0, 8))
        .expect("write doc-b");
    writer
        .write_record("doc-a", &sample_vector(2.0, 8))
        .expect("write doc-a");
    writer.finish().expect("finish");

    let index = VectorIndex::open(&path).expect("open index");
    assert_eq!(index.record_count(), 2);
    assert_eq!(index.dimension(), 8);
    assert_eq!(index.embedder_id(), "fnv1a-384");
    assert_eq!(index.embedder_revision(), "rev-123");
    assert_eq!(index.quantization(), Quantization::F16);
    assert_eq!(index.metadata().vectors_offset % VECTOR_ALIGN_BYTES, 0);

    let hash_a = fnv1a_hash(b"doc-a");
    let pos_a = index
        .find_index_by_doc_hash(hash_a)
        .expect("hash lookup should find doc-a");
    let doc_id = index.doc_id_at(pos_a).expect("doc id");
    assert_eq!(doc_id, "doc-a");
    let vec_a = index.vector_at_f32(pos_a).expect("vector");
    assert_eq!(vec_a.len(), 8);
    assert!((vec_a[0] - 2.0).abs() < 0.002);
}

#[cfg(feature = "bench-internals")]
#[test]
fn owned_record_handoff_matches_borrowed_writer_bytes() {
    let borrowed_path = temp_index_path("borrowed-record-handoff");
    let owned_path = temp_index_path("owned-record-handoff");
    let records = vec![
        ("doc-z".to_owned(), vec![0.25, -0.5, 0.75, 1.0]),
        ("doc-a".to_owned(), vec![-1.0, 0.5, 0.0, 0.125]),
        ("doc-m".to_owned(), vec![0.0, 0.25, -0.25, 0.5]),
    ];

    let mut borrowed = VectorIndex::create(&borrowed_path, "parity", 4).expect("borrowed");
    for (doc_id, embedding) in &records {
        borrowed
            .write_record(doc_id, embedding)
            .expect("write borrowed record");
    }
    borrowed.finish().expect("finish borrowed");

    let mut owned = VectorIndex::create(&owned_path, "parity", 4).expect("owned");
    for (doc_id, embedding) in records {
        owned
            .write_record_owned_for_benchmark(doc_id, embedding)
            .expect("write owned record");
    }
    owned.finish().expect("finish owned");

    assert_eq!(
        fs::read(&borrowed_path).expect("read borrowed bytes"),
        fs::read(&owned_path).expect("read owned bytes")
    );
}

#[test]
fn detects_header_crc_corruption() {
    let path = temp_index_path("crc");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-1", &sample_vector(0.5, 4))
        .expect("write");
    writer.finish().expect("finish");

    let mut bytes = fs::read(&path).expect("read index");
    // Flip a byte in the header payload before crc.
    bytes[6] ^= 0xAA;
    fs::write(&path, bytes).expect("rewrite corrupt index");

    let error = VectorIndex::open(&path).expect_err("corruption should be detected");
    assert!(matches!(error, SearchError::IndexCorrupted { .. }));
}

#[test]
fn write_record_dimension_mismatch_is_error() {
    let path = temp_index_path("dim-mismatch");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 3).expect("writer");
    let error = writer
        .write_record("doc-1", &[1.0, 2.0])
        .expect_err("must reject wrong dimension");
    assert!(matches!(
        error,
        SearchError::DimensionMismatch {
            expected: 3,
            found: 2
        }
    ));
}

#[test]
fn empty_index_round_trip() {
    let path = temp_index_path("empty");
    let writer = VectorIndex::create(&path, "fnv1a-384", 16).expect("writer");
    writer.finish().expect("finish");

    let index = VectorIndex::open(&path).expect("open");
    assert_eq!(index.record_count(), 0);
    assert_eq!(index.dimension(), 16);
}

#[test]
fn get_embeddings_returns_none_for_missing_hashes() {
    let path = temp_index_path("get-embeddings");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-1", &[0.1, 0.2, 0.3, 0.4])
        .expect("write");
    writer.finish().expect("finish");

    let index = VectorIndex::open(&path).expect("open");
    let existing = fnv1a_hash(b"doc-1");
    let missing = fnv1a_hash(b"missing");
    let embeddings = index.get_embeddings(&[existing, missing]);
    assert!(embeddings[0].is_some());
    assert!(embeddings[1].is_none());
    assert_eq!(embeddings[0].as_ref().expect("existing").len(), 4);
}

#[test]
fn soft_delete_marks_record_and_hides_hash_lookup() {
    let path = temp_index_path("soft-delete-main");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write doc-a");
    writer
        .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
        .expect("write doc-b");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    assert!(index.soft_delete("doc-a").expect("soft delete"));
    assert!(!index.soft_delete("doc-a").expect("idempotent soft delete"));

    let hash_a = fnv1a_hash(b"doc-a");
    let hash_b = fnv1a_hash(b"doc-b");
    assert_eq!(index.find_index_by_doc_hash(hash_a), None);
    assert!(index.find_index_by_doc_hash(hash_b).is_some());
    assert_eq!(index.tombstone_count(), 1);

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-b");

    std::fs::remove_file(&path).ok();
}

#[test]
fn soft_delete_missing_returns_false() {
    let path = temp_index_path("soft-delete-missing");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    assert!(
        !index
            .soft_delete("missing-doc")
            .expect("missing soft delete")
    );
    assert_eq!(index.tombstone_count(), 0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn soft_delete_batch_counts_only_new_tombstones() {
    let path = temp_index_path("soft-delete-batch");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write a");
    writer
        .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
        .expect("write b");
    writer
        .write_record("doc-c", &[0.0, 0.0, 1.0, 0.0])
        .expect("write c");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    let deleted = index
        .soft_delete_batch(&["doc-a", "doc-b", "missing", "doc-a"])
        .expect("batch delete");
    assert_eq!(deleted, 2);
    assert_eq!(index.tombstone_count(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn tombstone_ratio_and_needs_vacuum_threshold() {
    let path = temp_index_path("soft-delete-ratio");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    for i in 0..10 {
        writer
            .write_record(&format!("doc-{i}"), &sample_vector(0.1, 4))
            .expect("write");
    }
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    assert!(index.tombstone_ratio().abs() < f64::EPSILON);
    assert!(!index.needs_vacuum());

    index.soft_delete("doc-0").expect("delete 0");
    index.soft_delete("doc-1").expect("delete 1");
    assert_eq!(index.tombstone_count(), 2);
    assert!((index.tombstone_ratio() - 0.2).abs() < f64::EPSILON);
    assert!(!index.needs_vacuum(), "threshold is strict greater-than");

    index.soft_delete("doc-2").expect("delete 2");
    assert_eq!(index.tombstone_count(), 3);
    assert!(index.needs_vacuum());

    std::fs::remove_file(&path).ok();
}

#[test]
fn vacuum_removes_tombstones_and_preserves_live_results() {
    let path = temp_index_path("soft-delete-vacuum");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write a");
    writer
        .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
        .expect("write b");
    writer
        .write_record("doc-c", &[0.0, 0.0, 1.0, 0.0])
        .expect("write c");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    index.soft_delete("doc-b").expect("delete b");

    let pre_hits = index
        .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
        .expect("pre-vacuum search");
    assert_eq!(pre_hits.len(), 2);
    assert!(pre_hits.iter().all(|hit| hit.doc_id != "doc-b"));

    let stats = index.vacuum().expect("vacuum");
    assert_eq!(stats.records_before, 3);
    assert_eq!(stats.records_after, 2);
    assert_eq!(stats.tombstones_removed, 1);
    assert!(stats.bytes_reclaimed > 0);
    assert!(stats.duration >= Duration::ZERO);

    assert_eq!(index.record_count(), 2);
    assert_eq!(index.tombstone_count(), 0);
    assert_eq!(index.find_index_by_doc_hash(fnv1a_hash(b"doc-b")), None);

    let post_hits = index
        .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
        .expect("post-vacuum search");
    assert_eq!(post_hits.len(), 2);
    assert!(post_hits.iter().all(|hit| hit.doc_id != "doc-b"));

    std::fs::remove_file(&path).ok();
}

/// The lazy int8/4-bit slabs quantize the main-vector region at first
/// two-pass search. `vacuum()`/`compact()` swap that region via
/// `rewrite_index`; a slab surviving the swap scores pass-1 candidates
/// from rows that no longer exist (bd-0ho5p). One-hot vectors on unique
/// axes make the misalignment loud: pass-1 over a stale slab can never
/// surface the probe target after rows shift.
#[test]
fn vacuum_and_compact_invalidate_lazy_quantization_slabs() {
    let dim = 16;
    let path = temp_index_path("quant-slab-invalidate");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
    for i in 0..12 {
        let mut vector = vec![0.0f32; dim];
        vector[i] = 1.0;
        writer
            .write_record(&format!("doc-{i:02}"), &vector)
            .expect("write");
    }
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    let first = index.doc_id_at(0).expect("row 0").to_string();
    let target = index.doc_id_at(11).expect("row 11").to_string();
    let target_axis: usize = target
        .strip_prefix("doc-")
        .expect("doc prefix")
        .parse()
        .expect("axis");
    let mut probe = vec![0.0f32; dim];
    probe[target_axis] = 1.0;

    let pre_i8 = index
        .search_top_k_int8_two_pass(&probe, 1, 1)
        .expect("pre-vacuum int8");
    assert_eq!(pre_i8[0].doc_id, target);
    let pre_4bit = index
        .search_top_k_4bit_two_pass(&probe, 1, 1)
        .expect("pre-vacuum 4bit");
    assert_eq!(pre_4bit[0].doc_id, target);

    index.soft_delete(&first).expect("delete row 0");
    index.vacuum().expect("vacuum");

    let post_i8 = index
        .search_top_k_int8_two_pass(&probe, 1, 1)
        .expect("post-vacuum int8");
    assert_eq!(
        post_i8[0].doc_id, target,
        "int8 slab served stale pre-vacuum rows"
    );
    assert!(post_i8[0].score > 0.9);
    let post_4bit = index
        .search_top_k_4bit_two_pass(&probe, 1, 1)
        .expect("post-vacuum 4bit");
    assert_eq!(
        post_4bit[0].doc_id, target,
        "4-bit slab served stale pre-vacuum rows"
    );
    assert!(post_4bit[0].score > 0.9);

    // Same class through compact(): grow the index by one WAL resident so
    // the rewritten region is larger than the cached slabs.
    let mut extra = vec![0.0f32; dim];
    extra[12] = 1.0;
    index.append("doc-zz", &extra).expect("append doc-zz");
    index.compact().expect("compact");

    let target_after = index
        .search_top_k_int8_two_pass(&probe, 1, 1)
        .expect("post-compact int8");
    assert_eq!(
        target_after[0].doc_id, target,
        "int8 slab survived compact()"
    );
    assert!(target_after[0].score > 0.9);
    let mut zz_probe = vec![0.0f32; dim];
    zz_probe[12] = 1.0;
    let zz_hits = index
        .search_top_k_4bit_two_pass(&zz_probe, 1, 1)
        .expect("post-compact 4bit");
    assert_eq!(
        zz_hits[0].doc_id, "doc-zz",
        "4-bit slab survived compact() and cannot see the merged WAL resident"
    );
    assert!(zz_hits[0].score > 0.9);

    std::fs::remove_file(&path).ok();
}

#[test]
fn soft_delete_and_search_interleaving_has_no_corruption() {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};

    let path = temp_index_path("soft-delete-concurrent");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
    for i in 0..128 {
        writer
            .write_record(&format!("doc-{i:03}"), &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
    }
    writer.finish().expect("finish");

    let shared = Arc::new(Mutex::new(VectorIndex::open(&path).expect("open")));
    let deleter = {
        let index = Arc::clone(&shared);
        std::thread::spawn(move || {
            for i in 0..32 {
                let mut guard = index.lock().expect("lock for delete");
                let doc_id = format!("doc-{i:03}");
                let _ = guard.soft_delete(&doc_id).expect("soft delete");
            }
        })
    };

    let query = [1.0, 0.0, 0.0, 0.0];
    let searchers: Vec<_> = (0..4)
        .map(|_| {
            let index = Arc::clone(&shared);
            std::thread::spawn(move || {
                for _ in 0..32 {
                    let hits = index
                        .lock()
                        .expect("lock for search")
                        .search_top_k(&query, 10, None)
                        .expect("search");
                    assert!(!hits.is_empty());
                }
            })
        })
        .collect();

    deleter.join().expect("join deleter");
    for handle in searchers {
        handle.join().expect("join searcher");
    }

    let hits = shared
        .lock()
        .expect("lock final")
        .search_top_k(&query, 64, None)
        .expect("final search");
    let deleted_ids: HashSet<String> = (0..32).map(|i| format!("doc-{i:03}")).collect();
    assert!(
        hits.iter()
            .all(|hit| !deleted_ids.contains(hit.doc_id.as_str()))
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn soft_delete_preserves_existing_non_tombstone_flags() {
    let path = temp_index_path("soft-delete-flags");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write doc-a");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    let hash_a = fnv1a_hash(b"doc-a");
    let record_index = index
        .find_index_by_doc_hash(hash_a)
        .expect("record index for doc-a");

    let custom_flag: u16 = 0x0004;
    index
        .set_record_flags(record_index, custom_flag)
        .expect("seed custom flag");
    assert_eq!(
        index.record_at(record_index).expect("read flags").flags,
        custom_flag
    );

    assert!(index.soft_delete("doc-a").expect("soft delete doc-a"));
    let flags_after = index.record_at(record_index).expect("read flags").flags;
    assert_eq!(
        flags_after & RECORD_FLAG_TOMBSTONE,
        RECORD_FLAG_TOMBSTONE,
        "tombstone bit must be set",
    );
    assert_eq!(
        flags_after & custom_flag,
        custom_flag,
        "non-tombstone bits must remain untouched",
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn tombstone_flag_persists_after_reopen() {
    let path = temp_index_path("soft-delete-persist");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write a");
    writer
        .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
        .expect("write b");
    writer.finish().expect("finish");

    {
        let mut index = VectorIndex::open(&path).expect("open for delete");
        assert!(index.soft_delete("doc-a").expect("delete doc-a"));
        assert_eq!(index.tombstone_count(), 1);
    }

    let reopened = VectorIndex::open(&path).expect("reopen");
    assert_eq!(reopened.tombstone_count(), 1);
    assert_eq!(reopened.find_index_by_doc_hash(fnv1a_hash(b"doc-a")), None);
    let hits = reopened
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("search after reopen");
    assert!(hits.iter().all(|hit| hit.doc_id != "doc-a"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn delete_vacuum_append_cycle_keeps_expected_live_set() {
    use std::collections::HashSet;

    let path = temp_index_path("soft-delete-reindex-cycle");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
    for i in 0..100 {
        writer
            .write_record(&format!("doc-{i:03}"), &[1.0, 0.0, 0.0, 0.0])
            .expect("write initial doc");
    }
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    let delete_ids: Vec<String> = (0..50).map(|i| format!("doc-{i:03}")).collect();
    let delete_refs: Vec<&str> = delete_ids.iter().map(String::as_str).collect();
    let deleted = index.soft_delete_batch(&delete_refs).expect("batch delete");
    assert_eq!(deleted, 50);
    assert_eq!(index.tombstone_count(), 50);

    let vacuum_stats = index.vacuum().expect("vacuum");
    assert_eq!(vacuum_stats.records_before, 100);
    assert_eq!(vacuum_stats.records_after, 50);
    assert_eq!(index.tombstone_count(), 0);
    assert_eq!(index.record_count(), 50);

    let append_entries: Vec<(String, Vec<f32>)> = (100..150)
        .map(|i| (format!("doc-{i:03}"), vec![1.0, 0.0, 0.0, 0.0]))
        .collect();
    index.append_batch(&append_entries).expect("append batch");
    assert_eq!(index.wal_record_count(), 50);

    let compact_stats = index.compact().expect("compact");
    assert_eq!(compact_stats.total_records_after, 100);
    assert_eq!(index.record_count(), 100);
    assert_eq!(index.wal_record_count(), 0);

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 150, None)
        .expect("search");
    assert_eq!(hits.len(), 100);
    let ids: HashSet<String> = hits.iter().map(|hit| hit.doc_id.to_string()).collect();

    for i in 0..50 {
        assert!(
            !ids.contains(&format!("doc-{i:03}")),
            "deleted id must not be present",
        );
    }
    for i in 50..150 {
        assert!(
            ids.contains(&format!("doc-{i:03}")),
            "live id must be present",
        );
    }

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn tombstones_remain_excluded_with_wal_and_after_compaction() {
    let path = temp_index_path("soft-delete-wal-integration");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write a");
    writer
        .write_record("doc-b", &[1.0, 0.0, 0.0, 0.0])
        .expect("write b");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    assert!(index.soft_delete("doc-a").expect("delete a"));
    index
        .append("doc-c", &[1.0, 0.0, 0.0, 0.0])
        .expect("append c");
    assert_eq!(index.wal_record_count(), 1);

    let pre_compact = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("pre-compact search");
    assert_eq!(pre_compact.len(), 2);
    assert!(pre_compact.iter().all(|hit| hit.doc_id != "doc-a"));
    assert!(pre_compact.iter().any(|hit| hit.doc_id == "doc-b"));
    assert!(pre_compact.iter().any(|hit| hit.doc_id == "doc-c"));

    index.compact().expect("compact");
    let post_compact = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("post-compact search");
    assert_eq!(post_compact.len(), 2);
    assert!(post_compact.iter().all(|hit| hit.doc_id != "doc-a"));
    assert!(post_compact.iter().any(|hit| hit.doc_id == "doc-b"));
    assert!(post_compact.iter().any(|hit| hit.doc_id == "doc-c"));

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn vacuum_noop_when_no_tombstones() {
    let path = temp_index_path("soft-delete-vacuum-noop");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write a");
    writer
        .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
        .expect("write b");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    assert_eq!(index.tombstone_count(), 0);

    let stats = index.vacuum().expect("vacuum with no tombstones");
    assert_eq!(stats.records_before, 2);
    assert_eq!(stats.records_after, 2);
    assert_eq!(stats.tombstones_removed, 0);
    assert_eq!(index.record_count(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn soft_delete_all_records_yields_empty_search() {
    let path = temp_index_path("soft-delete-all");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    for i in 0..5 {
        writer
            .write_record(&format!("doc-{i}"), &sample_vector(0.1, 4))
            .expect("write");
    }
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    for i in 0..5 {
        assert!(index.soft_delete(&format!("doc-{i}")).expect("delete"));
    }
    assert_eq!(index.tombstone_count(), 5);
    assert!((index.tombstone_ratio() - 1.0).abs() < f64::EPSILON);
    assert!(index.needs_vacuum());

    let hits = index
        .search_top_k(&sample_vector(0.1, 4), 10, None)
        .expect("search");
    assert!(
        hits.is_empty(),
        "search with all deleted should return nothing"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn vacuum_after_deleting_all_records_yields_empty_index() {
    let path = temp_index_path("soft-delete-vacuum-all");
    let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
    for i in 0..3 {
        writer
            .write_record(&format!("doc-{i}"), &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
    }
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    for i in 0..3 {
        index.soft_delete(&format!("doc-{i}")).expect("delete");
    }

    let stats = index.vacuum().expect("vacuum all deleted");
    assert_eq!(stats.records_before, 3);
    assert_eq!(stats.records_after, 0);
    assert_eq!(stats.tombstones_removed, 3);
    assert_eq!(index.record_count(), 0);
    assert_eq!(index.tombstone_count(), 0);
    assert!(index.tombstone_ratio().abs() < f64::EPSILON);
    assert!(!index.needs_vacuum());

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert!(hits.is_empty());

    std::fs::remove_file(&path).ok();
}
