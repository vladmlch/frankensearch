use super::*;

#[cfg(feature = "ann")]
#[test]
fn ann_sidecar_is_created_when_threshold_is_met() {
    let dir = temp_index_dir("ann-enabled");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
        ],
    )
    .expect("write fast index");

    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 32,
        ..TwoTierConfig::default()
    };
    let index = TwoTierIndex::open(&dir, config).expect("open with ann");
    assert!(index.has_fast_ann());
    assert!(dir.join(VECTOR_ANN_FAST_FILENAME).exists());

    let hits = index
        .search_fast(&[1.0, 0.0, 0.0, 0.0], 1)
        .expect("ann search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-a");
}

#[cfg(feature = "ann")]
#[test]
fn ann_wal_merge_matches_canonical_resolution_before_and_after_post_build_tombstone() {
    use crate::wal::WalEntry;

    let dir = temp_index_dir("ann-wal-canonical-resolution");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0]),
            ("doc-b", &[0.8, 0.6]),
            ("doc-c", &[0.0, 1.0]),
            ("doc-delete", &[-1.0, 0.0]),
        ],
    )
    .expect("write fast index");

    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 32,
        ..TwoTierConfig::default()
    };
    let mut index = TwoTierIndex::open(&dir, config.clone()).expect("open ANN index");
    assert!(index.has_fast_ann());

    // Model the crash-recovery window in which a durable WAL update exists
    // but its best-effort main-slab tombstone did not land. Include a
    // repeated WAL identity to exercise the canonical post-rank dedup rule.
    let wal_entries = vec![
        WalEntry {
            doc_id: "doc-a".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-a"),
            embedding: vec![-1.0, 0.0],
        },
        WalEntry {
            doc_id: "doc-a".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-a"),
            embedding: vec![0.2, 0.979_795_9],
        },
        WalEntry {
            doc_id: "doc-new".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-new"),
            embedding: vec![0.95, 0.312_249_9],
        },
    ];
    let fast_tier = index
        .fast_tier_mut_for_test()
        .expect("path-opened fast tier");
    fast_tier.wal_entries.extend(wal_entries.clone());

    let query = [1.0_f32, 0.0];
    let assert_canonical = |index: &TwoTierIndex, label: &str| {
        let expected = index
            .fast_tier()
            .search_top_k(&query, 10, None)
            .expect("canonical main plus WAL search");
        let actual = index.search_fast(&query, 10).expect("ANN plus WAL search");
        let expected_identity: Vec<_> = expected
            .iter()
            .map(|hit| (hit.doc_id.clone(), hit.index))
            .collect();
        let actual_identity: Vec<_> = actual
            .iter()
            .map(|hit| (hit.doc_id.clone(), hit.index))
            .collect();
        assert_eq!(
            actual_identity, expected_identity,
            "{label}: ANN and exact search must resolve the same public identities"
        );

        let unique_ids: HashSet<_> = actual.iter().map(|hit| hit.doc_id.as_str()).collect();
        assert_eq!(
            unique_ids.len(),
            actual.len(),
            "{label}: public results must not contain duplicate document IDs"
        );
        let doc_a = actual
            .iter()
            .find(|hit| hit.doc_id == "doc-a")
            .expect("WAL doc-a remains searchable");
        assert!(
            usize::try_from(doc_a.index).expect("u32 fits usize")
                >= index.fast_tier().record_count(),
            "{label}: WAL doc-a must supersede the stale main-slab version"
        );
    };

    assert_canonical(&index, "normal ANN");
    assert_eq!(index.ann_fallback_count(), 0);

    // A post-build tombstone must happen through the writer role between
    // reader-tier opens. Reopening refreshes the ANN sidecar against the
    // live main rows, then the resident WAL must still merge through the
    // same resolver.
    drop(index);
    let mut writer = VectorIndex::open_writer(&fast_path).expect("writer opens after reader drops");
    assert!(
        writer
            .soft_delete("doc-delete")
            .expect("post-build tombstone")
    );
    drop(writer);
    let mut index = TwoTierIndex::open(&dir, config).expect("reopen ANN index after tombstone");
    index
        .fast_tier_mut_for_test()
        .expect("reopened path-opened fast tier")
        .wal_entries
        .extend(wal_entries);
    assert!(
        !index
            .search_fast(&query, 10)
            .expect("refreshed ANN search")
            .iter()
            .any(|hit| hit.doc_id == "doc-delete"),
        "the writer-role tombstone must survive the reader reopen"
    );
    assert_canonical(&index, "post-build tombstone");
    assert_eq!(
        index.ann_fallback_count(),
        0,
        "a sidecar refreshed for the tombstone must not report stale-graph fallback"
    );
}

#[cfg(feature = "ann")]
#[test]
fn ann_two_tier_duplicate_main_ids_consume_physical_top_k_slots() {
    let dir = temp_index_dir("ann-duplicate-main-top-k");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[
            ("duplicate", &[1.0, 0.0]),
            ("duplicate", &[0.95, 0.312_249_9]),
            ("lower-unique", &[0.8, 0.6]),
        ],
    )
    .expect("write duplicate-ID fast index");
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 32,
        ..TwoTierConfig::default()
    };
    let index = TwoTierIndex::open(&dir, config).expect("open ANN index");
    let query = [1.0_f32, 0.0];
    let expected = index
        .fast_tier()
        .search_top_k(&query, 2, None)
        .expect("canonical duplicate-ID search");
    let actual = index
        .search_fast(&query, 2)
        .expect("ANN duplicate-ID search");

    assert_eq!(expected.len(), 1);
    assert_eq!(actual, expected);
    assert_eq!(actual[0].doc_id, "duplicate");
    assert!(
        actual.iter().all(|hit| hit.doc_id != "lower-unique"),
        "requesting k=2 must not overfetch and backfill a third physical row"
    );
}

#[cfg(feature = "ann")]
#[test]
fn ann_is_skipped_below_threshold() {
    let dir = temp_index_dir("ann-disabled");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
    )
    .expect("write fast index");

    let config = TwoTierConfig {
        hnsw_threshold: 10_000,
        ..TwoTierConfig::default()
    };
    let index = TwoTierIndex::open(&dir, config).expect("open");
    assert!(!index.has_fast_ann());
    assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
}

#[cfg(feature = "ann")]
#[test]
fn ann_sidecar_rebuilds_when_config_changes() {
    let dir = temp_index_dir("ann-rebuild-config");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
        ],
    )
    .expect("write fast index");

    let initial = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_m: 8,
        hnsw_ef_construction: 64,
        hnsw_ef_search: 16,
        ..TwoTierConfig::default()
    };
    let first_open = TwoTierIndex::open(&dir, initial).expect("open with initial ann config");
    assert!(first_open.has_fast_ann());

    let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
    let before = load_native_ann_sidecar(&ann_path, first_open.fast_tier());
    let before_config = before.config();
    assert_eq!(before_config.m, 8);
    assert_eq!(before_config.ef_construction, 64);
    assert_eq!(before_config.ef_search, 16);

    let updated = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_m: 24,
        hnsw_ef_construction: 96,
        hnsw_ef_search: 48,
        ..TwoTierConfig::default()
    };
    let second_open = TwoTierIndex::open(&dir, updated).expect("open with updated ann config");
    assert!(second_open.has_fast_ann());

    let after = load_native_ann_sidecar(&ann_path, second_open.fast_tier());
    let after_config = after.config();
    assert_eq!(after_config.m, 24);
    assert_eq!(after_config.ef_construction, 96);
    assert_eq!(after_config.ef_search, 48);
}

#[cfg(feature = "ann")]
#[test]
fn ann_sidecar_rebuilds_when_vectors_change_with_same_doc_ids() {
    let dir = temp_index_dir("ann-rebuild-vectors");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
    )
    .expect("write initial fast index");

    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 64,
        ..TwoTierConfig::default()
    };
    let first = TwoTierIndex::open(&dir, config.clone()).expect("open initial");
    assert!(first.has_fast_ann());
    let before = first
        .search_fast(&[1.0, 0.0, 0.0], 1)
        .expect("search before");
    assert_eq!(before[0].doc_id, "doc-a");

    // Same doc IDs/order, but vectors are swapped. Sidecar must rebuild.
    write_index_file(
        &fast_path,
        &[("doc-a", &[0.0, 1.0, 0.0]), ("doc-b", &[1.0, 0.0, 0.0])],
    )
    .expect("rewrite fast index");

    let reopened = TwoTierIndex::open(&dir, config).expect("reopen");
    assert!(reopened.has_fast_ann());
    let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
    load_native_ann_sidecar(&ann_path, reopened.fast_tier());
    let after = reopened
        .search_fast(&[1.0, 0.0, 0.0], 1)
        .expect("search after");
    assert_eq!(
        after[0].doc_id, "doc-b",
        "ANN sidecar should rebuild when vector content changes"
    );
}

#[cfg(feature = "ann")]
#[test]
fn legacy_ann_fallback_is_persisted_for_native_second_load() {
    let dir = temp_index_dir("ann-self-heal-legacy");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
    )
    .expect("write fast index");

    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };
    let initial = TwoTierIndex::open(&dir, config.clone()).expect("seed ANN sidecar");
    assert!(initial.has_fast_ann());

    let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&ann_path).expect("read ANN metadata"))
            .expect("parse ANN metadata");
    let object = metadata.as_object_mut().expect("ANN metadata object");
    object.remove("format_version");
    object.remove("sidecar_generation");
    object.remove("sidecar_basename");
    fs::write(
        &ann_path,
        serde_json::to_vec(&metadata).expect("serialize legacy metadata"),
    )
    .expect("write legacy metadata");

    let (_, disposition) = HnswIndex::load_with_disposition(&ann_path, initial.fast_tier())
        .expect("legacy fallback rebuild");
    assert_eq!(disposition, HnswLoadDisposition::Rebuilt);

    let reopened = TwoTierIndex::open(&dir, config).expect("self-heal legacy ANN sidecar");
    assert!(reopened.has_fast_ann());
    load_native_ann_sidecar(&ann_path, reopened.fast_tier());
}

#[cfg(feature = "ann")]
#[test]
fn degraded_ann_fallback_is_persisted_for_native_second_load() {
    let dir = temp_index_dir("ann-self-heal-degraded");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
    )
    .expect("write fast index");

    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };
    let initial = TwoTierIndex::open(&dir, config.clone()).expect("seed ANN sidecar");
    assert!(initial.has_fast_ann());

    let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
    let missing_generation = ".missing-hnsw-generation";
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&ann_path).expect("read ANN metadata"))
            .expect("parse ANN metadata");
    metadata["sidecar_generation"] = missing_generation.into();
    fs::write(
        &ann_path,
        serde_json::to_vec(&metadata).expect("serialize degraded metadata"),
    )
    .expect("write degraded metadata");

    let (_, disposition) = HnswIndex::load_with_disposition(&ann_path, initial.fast_tier())
        .expect("degraded fallback rebuild");
    assert_eq!(disposition, HnswLoadDisposition::Rebuilt);

    let reopened = TwoTierIndex::open(&dir, config).expect("self-heal degraded ANN sidecar");
    assert!(reopened.has_fast_ann());
    load_native_ann_sidecar(&ann_path, reopened.fast_tier());
    let repaired: serde_json::Value =
        serde_json::from_slice(&fs::read(&ann_path).expect("read repaired metadata"))
            .expect("parse repaired metadata");
    assert_ne!(
        repaired["sidecar_generation"].as_str(),
        Some(missing_generation),
        "self-heal must publish metadata naming the new native generation"
    );
}

#[cfg(feature = "ann")]
#[test]
fn ann_persistence_failure_keeps_rebuilt_index_in_memory() {
    fn reject_persistence(_: &HnswIndex, _: &Path) -> SearchResult<()> {
        Err(SearchError::Io(std::io::Error::other(
            "injected ANN persistence failure",
        )))
    }

    let dir = temp_index_dir("ann-persist-failure");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
    )
    .expect("write fast index");

    let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 64,
        ..TwoTierConfig::default()
    };
    let initial = TwoTierIndex::open(&dir, config.clone()).expect("seed ANN sidecar");
    assert!(initial.has_fast_ann());

    // Force the successful-load fallback path, then inject a persistence
    // failure at the exact save seam. Startup must retain the rebuilt ANN
    // rather than turn a repair failure into a brute-force fallback.
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&ann_path).expect("read ANN metadata"))
            .expect("parse ANN metadata");
    let object = metadata.as_object_mut().expect("ANN metadata object");
    object.remove("format_version");
    object.remove("sidecar_generation");
    object.remove("sidecar_basename");
    fs::write(
        &ann_path,
        serde_json::to_vec(&metadata).expect("serialize legacy metadata"),
    )
    .expect("write legacy metadata");

    let rebuilt = maybe_load_or_build_ann_with_save(
        initial.fast_tier(),
        &ann_path,
        1,
        &config,
        "fast",
        || Ok(()),
        reject_persistence,
    )
    .expect("retain rebuilt ANN after persistence failure");

    // The injected failure left legacy metadata installed, proving the
    // error occurred after a successful rebuild rather than after a native
    // reload. That means the next startup will retry the repair.
    let (_, disposition) = HnswIndex::load_with_disposition(&ann_path, initial.fast_tier())
        .expect("legacy sidecar still rebuilds after failed persistence");
    assert_eq!(disposition, HnswLoadDisposition::Rebuilt);

    let hits = rebuilt
        .knn_search(&[1.0, 0.0, 0.0], 2, config.hnsw_ef_search)
        .expect("search in-memory rebuilt ANN");
    assert_eq!(
        hits.len(),
        2,
        "a retained rebuilt ANN must keep every live point reachable"
    );
    let mut hit_ids: Vec<&str> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
    hit_ids.sort_unstable();
    assert_eq!(hit_ids, ["doc-a", "doc-b"]);
}

#[cfg(feature = "ann")]
#[test]
fn ann_search_excludes_tombstoned_docs() {
    let dir = temp_index_dir("ann-tombstones");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0]),
            ("doc-c", &[0.0, 0.0, 1.0]),
        ],
    )
    .expect("write fast index");

    let mut fast_index = VectorIndex::open(&fast_path).expect("open fast index");
    let deleted = fast_index
        .soft_delete("doc-b")
        .expect("soft delete should succeed");
    assert!(deleted);
    // Release the writable mapping before reopening: since bd-x06p5 the
    // two-tier open acquires a SHARED READER lock on the same FSVI, which
    // a live writer handle would block.
    drop(fast_index);

    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 64,
        ..TwoTierConfig::default()
    };
    let index = TwoTierIndex::open(&dir, config).expect("open with ann");
    assert!(index.has_fast_ann());

    let hits = index.search_fast(&[0.0, 1.0, 0.0], 10).expect("search");
    assert!(
        !hits.iter().any(|hit| hit.doc_id == "doc-b"),
        "tombstoned document should not be returned by ANN search"
    );
}

/// Regression test for bd-2grj: HNSW `d_id` diverges from `VectorIndex` position
/// after tombstone-aware rebuild. Verifies that live docs survive the tombstone
/// filter even when their HNSW `d_id` differs from their `VectorIndex` position.
#[cfg(feature = "ann")]
#[test]
fn ann_tombstone_filter_uses_doc_id_not_hnsw_position() {
    let dir = temp_index_dir("ann-tombstone-docid");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    // A@0, B@1, C@2, D@3
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ("doc-d", &[0.0, 0.0, 0.0, 1.0]),
        ],
    )
    .expect("write fast index");

    // Soft-delete doc-b (position 1) — creates gap between HNSW d_ids and positions.
    // After rebuild: HNSW d_ids = {0:doc-a, 1:doc-c, 2:doc-d}
    // VectorIndex positions = {0:doc-a, 1:doc-b(deleted), 2:doc-c, 3:doc-d}
    let mut fast_index = VectorIndex::open(&fast_path).expect("open for delete");
    assert!(fast_index.soft_delete("doc-b").expect("soft_delete"));
    // See the note in `ann_search_excludes_tombstoned_docs`: the writable
    // mapping must be released before the two-tier reader lock is taken.
    drop(fast_index);

    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 64,
        ..TwoTierConfig::default()
    };
    let index = TwoTierIndex::open(&dir, config).expect("open with ann");
    assert!(index.has_fast_ann());

    // Search for all docs — should return doc-a, doc-c, doc-d (NOT doc-b)
    let hits = index
        .search_fast(&[0.25, 0.25, 0.25, 0.25], 10)
        .expect("search");

    let hit_ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();

    // Critical: doc-c and doc-d must survive even though their HNSW d_ids (1, 2)
    // differ from their VectorIndex positions (2, 3). Before bd-2grj fix,
    // doc-c was incorrectly filtered because is_deleted(1) checked position 1
    // (doc-b, which IS deleted) instead of position 2 (doc-c, which is live).
    assert!(
        hit_ids.contains(&"doc-a"),
        "doc-a should be returned, got: {hit_ids:?}"
    );
    assert!(
        hit_ids.contains(&"doc-c"),
        "doc-c should be returned (bd-2grj regression), got: {hit_ids:?}"
    );
    assert!(
        hit_ids.contains(&"doc-d"),
        "doc-d should be returned (bd-2grj regression), got: {hit_ids:?}"
    );
    assert!(
        !hit_ids.contains(&"doc-b"),
        "tombstoned doc-b should NOT be returned, got: {hit_ids:?}"
    );
    assert_eq!(
        hits.len(),
        3,
        "expected exactly 3 live docs, got: {hit_ids:?}"
    );
}
