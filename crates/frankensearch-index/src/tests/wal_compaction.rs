#![allow(clippy::wildcard_imports)]

use super::*;

// ─── WAL integration tests ─────────────────────────────────────────

/// Fleet-review critical regression pin: an update whose WAL append
/// FAILS must leave the old vector fully alive. The pre-fix ordering
/// ran the durably-destructive soft-delete before logging the
/// replacement, so a failed/interrupted WAL write destroyed the old
/// vector with the new one never written anywhere.
#[cfg(unix)]
#[test]
fn failed_wal_append_leaves_the_old_vector_alive() {
    use std::os::unix::fs::PermissionsExt;

    let path = temp_index_path("wal-append-atomic-update");
    let dim = 4;
    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("doc-x", &[1.0, 0.0, 0.0, 0.0])
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");

    // Make WAL-sidecar creation impossible: the parent directory is
    // read-only, so append_wal_batch's create fails with EACCES.
    let dir = path.parent().expect("fixture dir").to_path_buf();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).expect("seal fixture dir");
    let result = index.append("doc-x", &[0.0, 1.0, 0.0, 0.0]);
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).expect("restore fixture dir");
    if result.is_ok() {
        // Root ignores directory permission bits; the scenario cannot
        // be forced, so the ordering property is unobservable here.
        eprintln!("skipping: running as a principal that bypasses directory permissions");
        return;
    }

    // The failed update must not have destroyed the old vector in the
    // live handle or in a later read-only map. The writer map is dropped
    // before opening that reader: the retained exclusive lock forbids
    // concurrent published readers while a mutation handle is live.
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("live search");
    assert_eq!(
        hits.len(),
        1,
        "live handle: old vector must survive a failed update"
    );
    assert_eq!(hits[0].doc_id, "doc-x", "live handle");
    assert!(
        !index.is_deleted(0),
        "live handle: the main-slab record must not be tombstoned by a failed update"
    );
    drop(index);

    let reader = VectorIndex::open_read_only(&path).expect("reopen");
    let hits = reader
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("reader search");
    assert_eq!(
        hits.len(),
        1,
        "fresh reader: old vector must survive a failed update"
    );
    assert_eq!(hits[0].doc_id, "doc-x", "fresh reader");
    assert!(
        !reader.is_deleted(0),
        "fresh reader: the main-slab record must not be tombstoned by a failed update"
    );
}

#[test]
fn append_single_vector_is_searchable() {
    let path = temp_index_path("wal-append-single");
    let dim = 4;

    // Build initial index.
    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
        .expect("write");
    writer.finish().expect("finish");

    // Append via WAL.
    let mut index = VectorIndex::open(&path).expect("open");
    assert_eq!(index.wal_record_count(), 0);
    index
        .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
        .expect("append");
    assert_eq!(index.wal_record_count(), 1);
    assert_eq!(
        index.live_doc_ids().expect("live document ids"),
        std::collections::HashSet::from(["main-0".to_owned(), "wal-0".to_owned()])
    );

    // Search should find both main and WAL entries.
    let hits = index
        .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "wal-0", "WAL entry should rank first");

    // Cleanup.
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

/// bd-cnby1: a v1 rebuild via bare create+finish over an existing index
/// must be REFUSED while that index's WAL sidecar is present. Before the
/// fix, `finish()` published the fresh generation beside the old sidecar;
/// the fresh `compaction_gen` restarts at 1, `next_generation(1)` matches
/// the old sidecar's binding, and the next `open()` adopted every stale WAL
/// entry: deleted documents resurrected and the WAL shadow suppressed the
/// freshly rebuilt main-slab vectors.
#[test]
fn v1_rebuild_over_existing_wal_sidecar_is_refused_not_adopted() {
    let path = temp_index_path("v1-rebuild-wal-adoption");
    let mut writer = VectorIndex::create(&path, "old", 4).expect("writer");
    writer
        .write_record("kept-doc", &[1.0, 0.0, 0.0, 0.0])
        .expect("write old main");
    writer.finish().expect("finish old main");
    let mut old = VectorIndex::open(&path).expect("open old generation");
    old.append("deleted-doc", &[0.0, 1.0, 0.0, 0.0])
        .expect("append WAL record that the rebuild intends to drop");
    drop(old);

    // Operator intent: rebuild the corpus WITHOUT "deleted-doc".
    let mut rebuild = VectorIndex::create(&path, "old", 4).expect("rebuild writer");
    rebuild
        .write_record("kept-doc", &[0.9, 0.1, 0.0, 0.0])
        .expect("write rebuilt record");
    let refusal = rebuild.finish();
    let Err(SearchError::InvalidConfig { field, reason, .. }) = refusal else {
        panic!(
            "v1 finish() over an existing WAL sidecar must refuse publication, \
             got {refusal:?}"
        );
    };
    assert_eq!(field, "wal_sidecar");
    assert!(
        reason.contains("install_replacement"),
        "refusal must route the caller to the safe replacement path: {reason}"
    );

    // The refused publication must leave the old generation fully intact —
    // main slab AND its acknowledged WAL record.
    let survivor = VectorIndex::open(&path).expect("old generation survives refusal");
    assert_eq!(survivor.record_count(), 1);
    assert_eq!(
        survivor.wal_record_count(),
        1,
        "acknowledged WAL record must survive a refused rebuild"
    );
    drop(survivor);

    // The safe route succeeds and does NOT resurrect the dropped doc.
    let replacement_path = path.with_extension("replacement");
    let destination_gen =
        VectorIndex::peek_compaction_gen(&path).expect("peek destination generation");
    let mut replacement_writer = VectorIndex::create(&replacement_path, "old", 4)
        .expect("replacement writer")
        .with_generation(next_generation(destination_gen));
    replacement_writer
        .write_record("kept-doc", &[0.9, 0.1, 0.0, 0.0])
        .expect("write rebuilt record via safe route");
    replacement_writer
        .finish()
        .expect("replacement path has no sidecar, publication proceeds");
    let installed = VectorIndex::install_replacement(&path, &replacement_path)
        .expect("carried-generation install");
    assert_eq!(installed.record_count(), 1);
    assert_eq!(
        installed.wal_record_count(),
        0,
        "the dropped document's WAL entry must not be adopted by the rebuild"
    );
    let live = installed.live_doc_ids().expect("live ids");
    assert!(live.contains("kept-doc"));
    assert!(
        !live.contains("deleted-doc"),
        "bd-cnby1: the deliberately dropped document must stay dropped"
    );
    drop(installed);

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&replacement_path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

/// A replacement whose compaction generation collides with the
/// destination's must be refused outright: installing it would require
/// destroying the destination's acknowledged WAL BEFORE the rename, and a
/// crash in that window loses acknowledged records of the SURVIVING
/// generation (bd-zhjv8 finding 2).
#[test]
fn install_replacement_refuses_generation_collision_and_preserves_destination_wal() {
    let path = temp_index_path("install-gen-collision");
    let mut writer = VectorIndex::create(&path, "old", 4).expect("writer");
    writer
        .write_record("main-old", &[1.0, 0.0, 0.0, 0.0])
        .expect("write old main");
    writer.finish().expect("finish old main");
    let mut old = VectorIndex::open(&path).expect("open old generation");
    old.append("wal-acknowledged", &[0.0, 1.0, 0.0, 0.0])
        .expect("append acknowledged WAL record");
    drop(old);

    let replacement_path = path.with_extension("replacement");
    let replacement_writer =
        VectorIndex::create(&replacement_path, "new", 4).expect("replacement writer");
    replacement_writer.finish().expect("finish replacement");

    assert!(
        VectorIndex::install_replacement(&path, &replacement_path).is_err(),
        "a generation-colliding replacement must be refused, never installed by \
         destroying the destination's acknowledged WAL first"
    );
    let survivor = VectorIndex::open(&path).expect("destination must survive intact");
    assert_eq!(survivor.record_count(), 1);
    assert_eq!(
        survivor.wal_record_count(),
        1,
        "acknowledged WAL record must survive a refused install"
    );
    drop(survivor);

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&replacement_path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

/// The correct-by-construction path: a replacement built with the
/// carried-forward generation installs cleanly, and the destination's
/// surviving sidecar is superseded — never adopted — because its records
/// bind to the replaced generation.
#[test]
fn install_replacement_with_carried_generation_supersedes_stale_wal() {
    let path = temp_index_path("install-gen-carried");
    let mut writer = VectorIndex::create(&path, "old", 4).expect("writer");
    writer
        .write_record("main-old", &[1.0, 0.0, 0.0, 0.0])
        .expect("write old main");
    writer.finish().expect("finish old main");
    let mut old = VectorIndex::open(&path).expect("open old generation");
    old.append("wal-old", &[0.0, 1.0, 0.0, 0.0])
        .expect("append old WAL");
    drop(old);

    let replacement_path = path.with_extension("replacement");
    let destination_gen =
        VectorIndex::peek_compaction_gen(&path).expect("peek destination generation");
    let mut replacement_writer = VectorIndex::create(&replacement_path, "new", 4)
        .expect("replacement writer")
        .with_generation(next_generation(destination_gen));
    replacement_writer
        .write_record("main-new", &[0.0, 0.0, 1.0, 0.0])
        .expect("write replacement record");
    replacement_writer.finish().expect("finish replacement");

    let installed = VectorIndex::install_replacement(&path, &replacement_path)
        .expect("carried-generation replacement installs");
    assert_eq!(installed.record_count(), 1);
    assert_eq!(
        installed.wal_record_count(),
        0,
        "the superseded sidecar must never be adopted"
    );
    assert!(
        installed
            .live_doc_ids()
            .expect("live ids")
            .contains("main-new")
    );
    drop(installed);
    assert!(
        !wal::wal_path_for(&path).exists(),
        "a healthy install leaves no sidecar residue"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn replace_with_empty_cannot_resurrect_previous_wal_entries() {
    let path = temp_index_path("wal-replace-empty");
    let mut writer = VectorIndex::create(&path, "old", 4).expect("writer");
    writer
        .write_record("main-old", &[1.0, 0.0, 0.0, 0.0])
        .expect("write old main");
    writer.finish().expect("finish old main");

    let mut old = VectorIndex::open(&path).expect("open old generation");
    old.append("wal-old", &[0.0, 1.0, 0.0, 0.0])
        .expect("append old WAL");
    assert!(wal::wal_path_for(&path).exists());
    drop(old);

    assert!(
        VectorIndex::replace_with_empty(&path, "invalid", "", 0).is_err(),
        "replacement validation must fail before touching the old generation"
    );
    let intact = VectorIndex::open(&path).expect("reopen generation after rejected replace");
    assert_eq!(intact.record_count(), 1);
    assert_eq!(intact.wal_record_count(), 1);
    drop(intact);

    let replacement =
        VectorIndex::replace_with_empty(&path, "new", "rev-next", 4).expect("replace generation");
    assert_eq!(replacement.embedder_id(), "new");
    assert_eq!(replacement.embedder_revision(), "rev-next");
    assert_eq!(replacement.record_count(), 0);
    assert_eq!(replacement.wal_record_count(), 0);
    assert!(!wal::wal_path_for(&path).exists());
    drop(replacement);

    let reopened = VectorIndex::open(&path).expect("reopen replacement");
    let hits = reopened
        .search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None)
        .expect("search replacement");
    assert!(hits.is_empty(), "old WAL entries must not be resurrected");

    std::fs::remove_file(&path).ok();
}

#[test]
fn append_batch_all_searchable() {
    let path = temp_index_path("wal-append-batch");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    index
        .append_batch(&[
            ("wal-0".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
            ("wal-1".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
            ("wal-2".to_owned(), vec![0.0, 0.0, 0.0, 1.0]),
        ])
        .expect("append batch");
    assert_eq!(index.wal_record_count(), 3);

    let hits = index
        .search_top_k(&[1.0, 1.0, 1.0, 1.0], 10, None)
        .expect("search");
    assert_eq!(hits.len(), 4, "all 4 vectors should be returned");

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn compaction_merges_wal_into_main() {
    let path = temp_index_path("wal-compact");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    index
        .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
        .expect("append");
    index
        .append("wal-1", &[0.0, 0.0, 1.0, 0.0])
        .expect("append");

    assert_eq!(index.record_count(), 1);
    assert_eq!(index.wal_record_count(), 2);

    let stats = index.compact().expect("compact");
    assert_eq!(stats.main_records_before, 1);
    assert_eq!(stats.wal_records, 2);
    assert_eq!(stats.total_records_after, 3);
    assert_eq!(index.record_count(), 3);
    assert_eq!(index.wal_record_count(), 0);
    assert!(!wal::wal_path_for(&path).exists(), "WAL should be deleted");

    // All records should still be searchable from main index.
    let hits = index
        .search_top_k(&[1.0, 1.0, 1.0, 1.0], 10, None)
        .expect("search");
    assert_eq!(hits.len(), 3);

    std::fs::remove_file(&path).ok();
}

#[test]
fn needs_compaction_threshold() {
    let path = temp_index_path("wal-threshold");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    for i in 0..10 {
        writer
            .write_record(&format!("main-{i}"), &sample_vector(0.1, dim))
            .expect("write");
    }
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    index.set_wal_config(WalConfig {
        compaction_threshold: 5,
        compaction_ratio: 0.10,
        fsync_on_write: false,
    });

    assert!(!index.needs_compaction());

    // Add 1 entry: ratio = 1/10 = 0.10, hits the ratio threshold.
    index
        .append("wal-0", &sample_vector(0.2, dim))
        .expect("append");
    assert!(index.needs_compaction());

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn wal_survives_reopen() {
    let path = temp_index_path("wal-reopen");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
        .expect("write");
    writer.finish().expect("finish");

    // Append and drop.
    {
        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
            .expect("append");
    }

    // Reopen — WAL should be loaded automatically.
    let index = VectorIndex::open(&path).expect("reopen");
    assert_eq!(index.wal_record_count(), 1);

    let hits = index
        .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "wal-0");

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn append_dimension_mismatch_rejected() {
    let path = temp_index_path("wal-dim-mismatch");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    let err = index
        .append("bad", &[1.0, 2.0])
        .expect_err("should reject wrong dimension");
    assert!(matches!(err, SearchError::DimensionMismatch { .. }));
    assert_eq!(
        index.wal_record_count(),
        0,
        "failed append should not persist"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn compact_empty_wal_is_noop() {
    let path = temp_index_path("wal-compact-empty");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    let stats = index.compact().expect("compact empty WAL");
    assert_eq!(stats.wal_records, 0);
    assert_eq!(stats.total_records_after, 1);

    std::fs::remove_file(&path).ok();
}

#[test]
fn wal_entries_rank_correctly_against_main() {
    let path = temp_index_path("wal-ranking");
    let dim = 4;

    // Main index has a mediocre match.
    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-mediocre", &[0.5, 0.5, 0.0, 0.0])
        .expect("write");
    writer.finish().expect("finish");

    // WAL has a perfect match.
    let mut index = VectorIndex::open(&path).expect("open");
    index
        .append("wal-perfect", &[1.0, 0.0, 0.0, 0.0])
        .expect("append");

    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "wal-perfect");
    assert!(hits[0].score > hits[1].score);

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn append_duplicate_doc_id_both_searchable() {
    let path = temp_index_path("wal-dup-docid");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    // Append a second entry with the same doc_id but different vector.
    index
        .append("doc-a", &[0.0, 0.0, 0.0, 1.0])
        .expect("append duplicate");
    assert_eq!(index.wal_record_count(), 1);

    // WAL entry shadows the main-index entry (WAL is newer).
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert_eq!(
        hits.len(),
        1,
        "WAL shadows main — only WAL entry should appear"
    );
    assert_eq!(hits[0].doc_id, "doc-a");

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn append_large_batch_100_vectors() {
    let path = temp_index_path("wal-large-batch");
    let dim = 8;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    let batch: Vec<(String, Vec<f32>)> = (0..100)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            // Offset keeps wal-000 above the zero-norm writer gate while
            // staying below main-0's base of 1.0 for the ranking assert.
            let base = (i as f32).mul_add(0.01, 0.005);
            (format!("wal-{i:03}"), sample_vector(base, dim))
        })
        .collect();
    index.append_batch(&batch).expect("large batch");
    assert_eq!(index.wal_record_count(), 100);

    let hits = index
        .search_top_k(&sample_vector(1.0, dim), 5, None)
        .expect("search");
    assert_eq!(hits.len(), 5);
    // The main-0 (base=1.0) should rank near the top with query [1.0, ...].
    assert!(hits.iter().any(|h| h.doc_id == "main-0"));

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn concurrent_append_and_search() {
    use std::sync::Arc;

    let path = temp_index_path("wal-concurrent");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    for i in 0..10 {
        writer
            .write_record(&format!("main-{i}"), &sample_vector(0.1, dim))
            .expect("write");
    }
    writer.finish().expect("finish");

    // Append sequentially (VectorIndex is not Send+Sync for shared mutation),
    // then search from multiple threads using a snapshot.
    let mut index = VectorIndex::open(&path).expect("open");
    for i in 0..20 {
        index
            .append(&format!("wal-{i}"), &sample_vector(0.5, dim))
            .expect("append");
    }

    let index = Arc::new(index);
    let query = sample_vector(1.0, dim);

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let idx = Arc::clone(&index);
            let q = query.clone();
            std::thread::spawn(move || idx.search_top_k(&q, 10, None).expect("search"))
        })
        .collect();

    for handle in handles {
        let hits = handle.join().expect("thread join");
        assert_eq!(hits.len(), 10);
        // All scores should be positive (dot product of positive vectors).
        assert!(hits.iter().all(|h| h.score > 0.0));
    }

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn wal_record_count_across_append_compact_cycles() {
    let path = temp_index_path("wal-count-cycle");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    assert_eq!(index.wal_record_count(), 0);
    assert_eq!(index.record_count(), 1);

    // Append 3 entries.
    index.append("w1", &sample_vector(0.1, dim)).expect("a1");
    index.append("w2", &sample_vector(0.2, dim)).expect("a2");
    index.append("w3", &sample_vector(0.3, dim)).expect("a3");
    assert_eq!(index.wal_record_count(), 3);
    assert_eq!(index.record_count(), 1);

    // Compact.
    index.compact().expect("compact");
    assert_eq!(index.wal_record_count(), 0);
    assert_eq!(index.record_count(), 4);

    // Append 2 more.
    index.append("w4", &sample_vector(0.4, dim)).expect("a4");
    index.append("w5", &sample_vector(0.5, dim)).expect("a5");
    assert_eq!(index.wal_record_count(), 2);
    assert_eq!(index.record_count(), 4);

    // Total searchable = 4 + 2 = 6.
    let hits = index
        .search_top_k(&sample_vector(1.0, dim), 100, None)
        .expect("search");
    assert_eq!(hits.len(), 6);

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

/// Minimal `tracing` subscriber that records every WARN-or-worse event
/// message. The crate has no subscriber dev-dependency, and the
/// regression below is invisible to every other observable: compaction
/// used to reload through `open` while the just-merged WAL sidecar was
/// still on disk, so the reload classified it as stale and warned about
/// discarding entries it had already absorbed.
struct WarnRecorder {
    messages: std::sync::Mutex<Vec<String>>,
}

impl tracing::Subscriber for WarnRecorder {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::WARN
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        struct MessageVisitor(String);
        impl tracing::field::Visit for MessageVisitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = format!("{value:?}");
                }
            }
        }
        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        self.messages.lock().unwrap().push(visitor.0);
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

#[test]
fn compaction_reload_does_not_warn_about_the_wal_it_just_merged() {
    let path = temp_index_path("wal-compact-reload-quiet");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    index.append("w1", &sample_vector(0.1, dim)).expect("a1");
    index.append("w2", &sample_vector(0.2, dim)).expect("a2");
    assert!(
        wal::wal_path_for(&path).exists(),
        "append must create the sidecar"
    );

    let recorder = std::sync::Arc::new(WarnRecorder {
        messages: std::sync::Mutex::new(Vec::new()),
    });
    let stats = tracing::subscriber::with_default(std::sync::Arc::clone(&recorder), || {
        index.compact().expect("compact")
    });
    assert_eq!(stats.wal_records, 2);
    assert_eq!(stats.total_records_after, 3);

    let warnings = recorder.messages.lock().unwrap().clone();
    assert!(
        warnings.is_empty(),
        "compacting a live WAL must not warn about discarding it: {warnings:?}"
    );
    assert!(
        !wal::wal_path_for(&path).exists(),
        "the merged sidecar must be gone after compaction"
    );

    // The merged entries are durable and a later append still lands in a
    // fresh, valid sidecar for the new generation.
    index.append("w3", &sample_vector(0.3, dim)).expect("a3");
    drop(index);
    let reopened = VectorIndex::open(&path).expect("reopen");
    assert_eq!(reopened.record_count(), 3);
    assert_eq!(reopened.wal_record_count(), 1);

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

/// A main-generation tombstone is an in-place write through the shared
/// mapping; consumers fingerprint the generation by file metadata, so the
/// write must be visible as a timestamp change on every filesystem
/// (tmpfs does not bump mtime for a write to an already-dirty page).
#[test]
fn soft_delete_of_a_main_record_bumps_the_file_timestamp() {
    let path = temp_index_path("soft-delete-touches-main");
    let dim = 4;
    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .expect("write");
    writer.finish().expect("finish");

    let before = std::fs::metadata(&path)
        .expect("metadata before")
        .modified()
        .expect("mtime before");
    // Coarse-clock filesystems stamp at jiffy resolution; step past it.
    std::thread::sleep(Duration::from_millis(20));

    let mut index = VectorIndex::open(&path).expect("open");
    assert!(index.soft_delete("main-0").expect("tombstone main record"));
    drop(index);

    let after = std::fs::metadata(&path)
        .expect("metadata after")
        .modified()
        .expect("mtime after");
    assert!(
        after > before,
        "an in-place tombstone must advance the file's mtime: before={before:?} after={after:?}"
    );

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

/// A `RaptorQ` sidecar encodes exact bytes; every write that changes the
/// main file must drop it, or a later repair restores stale contents.
#[test]
fn main_file_writes_remove_the_durability_sidecar() {
    let path = temp_index_path("sidecar-invalidated-by-writes");
    let dim = 4;
    let sidecar = PathBuf::from(format!("{}.fec", path.display()));

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .expect("write");
    writer
        .write_record("main-1", &sample_vector(0.5, dim))
        .expect("write");
    std::fs::write(&sidecar, b"stale sidecar from before finish").expect("plant sidecar");
    writer.finish().expect("finish");
    assert!(
        !sidecar.exists(),
        "finish must drop a sidecar built for older bytes"
    );

    // Compaction rewrites the main file.
    std::fs::write(&sidecar, b"stale sidecar from before compaction").expect("plant sidecar");
    let mut index = VectorIndex::open(&path).expect("open");
    index
        .append("w1", &sample_vector(0.1, dim))
        .expect("append");
    index.compact().expect("compact");
    assert!(!sidecar.exists(), "compaction must drop the sidecar");

    // An in-place tombstone changes the main file too.
    std::fs::write(&sidecar, b"stale sidecar from before tombstone").expect("plant sidecar");
    assert!(index.soft_delete("main-0").expect("tombstone"));
    assert!(
        !sidecar.exists(),
        "an in-place tombstone must drop the sidecar"
    );

    // A WAL-only tombstone leaves the main file untouched and the sidecar alive.
    std::fs::write(&sidecar, b"sidecar for the current bytes").expect("plant sidecar");
    index
        .append("w2", &sample_vector(0.2, dim))
        .expect("append");
    assert!(index.soft_delete("w2").expect("wal-only tombstone"));
    assert!(
        sidecar.exists(),
        "a WAL-only tombstone does not change the main file"
    );

    drop(index);
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&sidecar).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn soft_delete_removes_wal_only_record_and_persists() {
    let path = temp_index_path("wal-soft-delete-only");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .expect("write");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    index
        .append("wal-only", &[0.0, 1.0, 0.0, 0.0])
        .expect("append wal-only");
    assert_eq!(index.wal_record_count(), 1);

    assert!(index.soft_delete("wal-only").expect("soft delete wal-only"));
    assert_eq!(index.wal_record_count(), 0);
    let hits = index
        .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert!(hits.iter().all(|hit| hit.doc_id != "wal-only"));

    drop(index);
    let reopened = VectorIndex::open(&path).expect("reopen");
    assert_eq!(reopened.wal_record_count(), 0);
    let reopened_hits = reopened
        .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
        .expect("search after reopen");
    assert!(reopened_hits.iter().all(|hit| hit.doc_id != "wal-only"));

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn soft_delete_clears_pending_wal_updates_for_same_doc_id() {
    let path = temp_index_path("wal-soft-delete-main-and-wal");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer
        .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("write doc-a");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    index
        .append("doc-a", &[0.0, 1.0, 0.0, 0.0])
        .expect("append doc-a update");
    index
        .append("doc-b", &[0.0, 0.0, 1.0, 0.0])
        .expect("append doc-b");
    assert_eq!(index.wal_record_count(), 2);

    assert!(index.soft_delete("doc-a").expect("soft delete doc-a"));
    assert_eq!(
        index.wal_record_count(),
        1,
        "doc-a WAL entries should be purged"
    );

    let hits = index
        .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert!(
        hits.iter().all(|hit| hit.doc_id != "doc-a"),
        "doc-a should not be searchable from main or WAL"
    );
    assert!(hits.iter().any(|hit| hit.doc_id == "doc-b"));

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn empty_index_append_only() {
    let path = temp_index_path("wal-empty-append");
    let dim = 4;

    // Create an empty main index.
    let writer = VectorIndex::create(&path, "test", dim).expect("writer");
    writer.finish().expect("finish");

    let mut index = VectorIndex::open(&path).expect("open");
    assert_eq!(index.record_count(), 0);

    // Append to empty index via WAL.
    index
        .append("first", &[1.0, 0.0, 0.0, 0.0])
        .expect("append");
    assert_eq!(index.wal_record_count(), 1);

    // Should still be searchable.
    let hits = index
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "first");

    // Compact from empty main + WAL.
    let stats = index.compact().expect("compact");
    assert_eq!(stats.main_records_before, 0);
    assert_eq!(stats.wal_records, 1);
    assert_eq!(stats.total_records_after, 1);
    assert_eq!(index.record_count(), 1);

    std::fs::remove_file(&path).ok();
}
