use std::io::Write;

use frankensearch_core::SearchError;

use crate::{Quantization, VectorIndex};

use super::super::fingerprint::{fingerprint_live_vector_index, fingerprint_vectors};
use super::super::persistence::{
    acquire_hnsw_save_lock, gc_superseded_hnsw_generations, hnsw_save_lock_path,
    hnsw_sidecar_basename, persisted_hnsw_sidecar_location,
};
use super::super::types::{
    HNSW_DEFAULT_EF_SEARCH, HNSW_GENERATION_RECEIPT_FILENAME, HNSW_GENERATION_RECEIPT_MAX_BYTES,
    HNSW_META_FORMAT_CURRENT, HNSW_SAVE_LOCK_DIRECTORY, HnswConfig, HnswGenerationReceipt,
    HnswIndex, HnswLoadDisposition, HnswMeta,
};
use super::{
    install_digest_valid_unloadable_data, install_then_report_parent_sync_failure,
    normalized_vector, ready_generation_paths, reject_hnsw_metadata_publish,
    snapshot_immutable_tree, temp_path, write_index,
};

#[test]
fn persistence_round_trip() {
    let fsvi_path = temp_path("persist", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
    let index = write_index(&fsvi_path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

    let save_path = temp_path("persist", "hnsw");
    ann.save(&save_path).expect("save");

    let (loaded, disposition) = HnswIndex::load_with_disposition(&save_path, &index).expect("load");
    assert_eq!(disposition, HnswLoadDisposition::Native);
    assert_eq!(loaded.len(), 64);
    assert_eq!(loaded.dimension(), 32);

    let query = normalized_vector(10, 32);
    let hits = loaded
        .knn_search(&query, 5, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert_eq!(hits[0].doc_id, "doc-0010");
    assert!((hits[0].score - 1.0).abs() < 1e-5);
}

#[test]
fn public_strict_load_reports_native_without_mutating_selected_artifacts() {
    let root = temp_path("public-strict-native", "dir");
    std::fs::create_dir_all(&root).expect("create strict native fixture");
    let fsvi_path = root.join("selected-source.fsvi");
    let vectors: Vec<Vec<f32>> = (0..32).map(|i| normalized_vector(i, 16)).collect();
    let source = write_index(&fsvi_path, &vectors).expect("write source FSVI");
    let ann =
        HnswIndex::build_from_vector_index(&source, HnswConfig::default()).expect("build ANN");
    let metadata_path = root.join("selected-ann.hnsw");
    ann.save(&metadata_path).expect("save selected native ANN");
    let before = snapshot_immutable_tree(&root).expect("snapshot selected artifacts");

    let native_only = HnswIndex::try_load_native(&metadata_path, &source)
        .expect("inspect native graph")
        .expect("selected graph must be native");
    let (loaded, disposition) =
        HnswIndex::load_with_disposition(&metadata_path, &source).expect("strict native load");

    assert_eq!(disposition, HnswLoadDisposition::Native);
    assert_eq!(native_only.len(), source.live_count());
    assert_eq!(loaded.len(), source.live_count());
    assert_eq!(
        snapshot_immutable_tree(&root).expect("snapshot after strict native load"),
        before,
        "native inspection must preserve bytes, mtimes, permissions, and directory inventory"
    );
}

#[test]
fn public_strict_load_reports_rebuilt_for_stale_source_without_mutation() {
    let root = temp_path("public-strict-stale", "dir");
    std::fs::create_dir_all(&root).expect("create strict stale fixture");
    let original_path = root.join("original-source.fsvi");
    let original_vectors: Vec<Vec<f32>> = (0..24).map(|i| normalized_vector(i, 12)).collect();
    let original = write_index(&original_path, &original_vectors).expect("original FSVI");
    let ann = HnswIndex::build_from_vector_index(&original, HnswConfig::default())
        .expect("build original ANN");
    let metadata_path = root.join("selected-ann.hnsw");
    ann.save(&metadata_path).expect("save selected ANN");

    let replacement_path = root.join("replacement-source.fsvi");
    let replacement_vectors: Vec<Vec<f32>> =
        (10_000..10_024).map(|i| normalized_vector(i, 12)).collect();
    let replacement =
        write_index(&replacement_path, &replacement_vectors).expect("replacement FSVI");
    let before = snapshot_immutable_tree(&root).expect("snapshot stale fixture");

    assert!(
        HnswIndex::try_load_native(&metadata_path, &replacement)
            .expect("inspect stale sidecar")
            .is_none(),
        "native-only inspection must reject stale vectors without rebuilding"
    );
    let (loaded, disposition) = HnswIndex::load_with_disposition(&metadata_path, &replacement)
        .expect("stale sidecar should rebuild in memory");

    assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
    assert_eq!(loaded.len(), replacement.live_count());
    assert!(
        loaded
            .matches_vector_index(&replacement)
            .expect("rebuilt graph matches replacement")
    );
    assert_eq!(
        snapshot_immutable_tree(&root).expect("snapshot after stale load"),
        before,
        "stale detection and in-memory rebuild must preserve every selected artifact"
    );
}

#[test]
fn save_retry_reuses_ready_generation_after_pre_publish_error() {
    let source_path = temp_path("ready-pre-publish-source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("ANN index");
    let metadata_path = temp_path("ready-pre-publish", "hnsw");

    let error = ann
        .save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
        .expect_err("injected publication failure");
    assert!(matches!(&error, SearchError::Io(_)));
    assert!(
        !metadata_path.exists(),
        "pre-publication failure must leave metadata absent"
    );
    let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(before.len(), 1, "one complete READY generation retained");
    let retained_generation = before[0]
        .file_name()
        .and_then(|name| name.to_str())
        .expect("retained generation name")
        .to_owned();

    ann.save(&metadata_path).expect("retry READY publication");
    let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(
        after, before,
        "retry must reuse the complete generation instead of dumping another"
    );
    let metadata: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
            .expect("parse metadata");
    assert_eq!(
        metadata.sidecar_generation.as_deref(),
        Some(retained_generation.as_str())
    );
    let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
        .expect("native load after retry");
    assert_eq!(disposition, HnswLoadDisposition::Native);
}

#[test]
fn save_retry_after_metadata_rename_sync_uncertainty_reuses_generation() {
    let source_path = temp_path("ready-post-rename-source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("ANN index");
    let metadata_path = temp_path("ready-post-rename", "hnsw");

    let error = ann
        .save_with_metadata_publisher(&metadata_path, install_then_report_parent_sync_failure)
        .expect_err("injected post-rename sync failure");
    assert!(matches!(error, SearchError::Io(_)));
    assert!(
        metadata_path.is_file(),
        "metadata rename may already be visible when parent sync fails"
    );
    let installed: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
            .expect("parse metadata");
    let installed_generation = installed
        .sidecar_generation
        .as_deref()
        .expect("installed generation")
        .to_owned();
    let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(before.len(), 1);

    ann.save(&metadata_path)
        .expect("retry durability-uncertain publication");
    let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(
        after, before,
        "retry must not strand the installed generation"
    );
    let repaired: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read repaired metadata"))
            .expect("parse repaired metadata");
    assert_eq!(
        repaired.sidecar_generation.as_deref(),
        Some(installed_generation.as_str())
    );
    let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
        .expect("native load after durability retry");
    assert_eq!(disposition, HnswLoadDisposition::Native);
}

#[test]
fn save_lock_contention_fails_before_dump() {
    let source_path = temp_path("save-lock-source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("ANN index");
    let metadata_path = temp_path("save-lock", "hnsw");
    let lock = acquire_hnsw_save_lock(&metadata_path).expect("hold save lock");

    let error = ann
        .save(&metadata_path)
        .expect_err("contending save must fail before staging");
    assert!(matches!(&error, SearchError::Io(_)));
    assert!(
        error.to_string().contains("another writer may be saving"),
        "lock failure should be actionable: {error}"
    );
    assert!(!metadata_path.exists());
    assert!(
        ready_generation_paths(&metadata_path, ann.vector_fingerprint).is_empty(),
        "contending writer must not dump a generation"
    );

    drop(lock);
    ann.save(&metadata_path)
        .expect("persistent lock file remains reusable after holder exits");
    assert_eq!(
        ready_generation_paths(&metadata_path, ann.vector_fingerprint).len(),
        1
    );
}

#[test]
fn save_ignores_mismatched_ready_receipt() {
    let source_path = temp_path("ready-mismatch-source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("ANN index");
    let metadata_path = temp_path("ready-mismatch", "hnsw");
    ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
        .expect_err("retain unpublished READY generation");
    let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(before.len(), 1);
    let rejected_generation = before[0]
        .file_name()
        .and_then(|name| name.to_str())
        .expect("generation name")
        .to_owned();
    let receipt_path = before[0].join(HNSW_GENERATION_RECEIPT_FILENAME);
    let mut receipt: HnswGenerationReceipt =
        serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
            .expect("parse receipt");
    receipt.config.m = receipt.config.m.saturating_add(1);
    std::fs::write(
        &receipt_path,
        serde_json::to_vec(&receipt).expect("serialize mismatched receipt"),
    )
    .expect("write mismatched receipt");

    ann.save(&metadata_path)
        .expect("save past mismatched retained receipt");
    let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    // bd-v03xp made publish reclaim every sibling generation the published
    // metadata does not reference, precisely so corrupt-but-receipt-valid
    // generations stop being re-probed with a full native load on every
    // save. The rejected generation is therefore reclaimed, not retained;
    // what this test pins is that save REFUSED TO REUSE it and published a
    // fresh one instead, which the assert_ne! below carries.
    assert_eq!(
        after.len(),
        1,
        "publish must reclaim the superseded invalid generation"
    );
    let metadata: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
            .expect("parse metadata");
    assert_ne!(
        metadata.sidecar_generation.as_deref(),
        Some(rejected_generation.as_str())
    );
    let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
        .expect("native load after rejecting receipt");
    assert_eq!(disposition, HnswLoadDisposition::Native);
}

#[test]
fn save_ignores_digest_valid_but_unloadable_ready_generation() {
    let source_path = temp_path("ready-unloadable-source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("ANN index");
    let metadata_path = temp_path("ready-unloadable", "hnsw");
    ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
        .expect_err("retain unpublished READY generation");
    let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(before.len(), 1);
    let rejected_generation = before[0]
        .file_name()
        .and_then(|name| name.to_str())
        .expect("generation name")
        .to_owned();
    let receipt_path = before[0].join(HNSW_GENERATION_RECEIPT_FILENAME);
    let receipt: HnswGenerationReceipt =
        serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
            .expect("parse receipt");
    let corrupt_basename = &receipt.sidecar_basename;
    install_digest_valid_unloadable_data(&before[0], corrupt_basename);

    ann.save(&metadata_path)
        .expect("save past unloadable retained generation");
    let after = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(
        after.len(),
        1,
        "an unloadable generation must not trap rebuild-save in a reuse loop, \
         and publish reclaims it once a fresh generation supersedes it (bd-v03xp)"
    );
    let metadata: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
            .expect("parse metadata");
    assert_ne!(
        metadata.sidecar_generation.as_deref(),
        Some(rejected_generation.as_str())
    );
    let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
        .expect("native load after rejecting unloadable generation");
    assert_eq!(disposition, HnswLoadDisposition::Native);
}

#[test]
fn save_ignores_oversized_ready_receipt() {
    let source_path = temp_path("ready-oversized-source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("ANN index");
    let metadata_path = temp_path("ready-oversized", "hnsw");
    ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
        .expect_err("retain unpublished READY generation");
    let before = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(before.len(), 1);
    let rejected_generation = before[0]
        .file_name()
        .and_then(|name| name.to_str())
        .expect("generation name")
        .to_owned();
    let receipt_path = before[0].join(HNSW_GENERATION_RECEIPT_FILENAME);
    std::fs::write(
        receipt_path,
        vec![b' '; HNSW_GENERATION_RECEIPT_MAX_BYTES + 1],
    )
    .expect("write oversized receipt fixture");

    ann.save(&metadata_path)
        .expect("save past oversized retained receipt");
    assert_eq!(
        ready_generation_paths(&metadata_path, ann.vector_fingerprint).len(),
        1,
        "publish must reclaim the superseded oversized-receipt generation"
    );
    // The load-bearing contract, which this test previously left implicit
    // in the directory count: the oversized receipt must not be REUSED.
    // Without this the count alone would pass even if save adopted it.
    let metadata: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
            .expect("parse metadata");
    assert_ne!(
        metadata.sidecar_generation.as_deref(),
        Some(rejected_generation.as_str()),
        "save must publish a fresh generation, never reuse the oversized-receipt one"
    );
    let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
        .expect("native load after rejecting oversized receipt");
    assert_eq!(disposition, HnswLoadDisposition::Native);
}

#[test]
fn save_lock_namespace_prevents_metadata_name_collision() {
    let source_path = temp_path("save-lock-namespace-source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("ANN index");
    let metadata_path = temp_path("save-lock-namespace", "hnsw");
    let held_lock = acquire_hnsw_save_lock(&metadata_path).expect("hold save lock");
    let lock_path = hnsw_save_lock_path(&metadata_path).expect("lock path");

    let mut old_lock_name = metadata_path
        .file_name()
        .expect("metadata file name")
        .to_os_string();
    old_lock_name.push(".lock");
    let formerly_colliding_metadata_path = metadata_path.with_file_name(old_lock_name);
    ann.save(&formerly_colliding_metadata_path)
        .expect("adjacent .lock metadata name is isolated from the lock namespace");
    assert!(lock_path.is_file(), "held lock inode must remain installed");

    let contention = ann
        .save(&metadata_path)
        .expect_err("original metadata path must remain locked");
    assert!(
        contention
            .to_string()
            .contains("another writer may be saving")
    );

    let reserved_path_error = ann
        .save(&lock_path)
        .expect_err("metadata cannot overwrite the reserved lock namespace");
    assert!(
        reserved_path_error
            .to_string()
            .contains(HNSW_SAVE_LOCK_DIRECTORY)
    );

    drop(held_lock);
    ann.save(&metadata_path)
        .expect("original path saves after lock release");
}

#[test]
fn persistence_writes_native_graph_sidecars() {
    let fsvi_path = temp_path("persist_native", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
    let index = write_index(&fsvi_path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

    let save_path = temp_path("persist_native", "hnsw");
    ann.save(&save_path).expect("save");

    let meta: HnswMeta =
        serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse meta");
    assert_eq!(meta.format_version, HNSW_META_FORMAT_CURRENT);

    // A fresh save gets the requested basename inside an owned generation
    // and records that exact pair in current-format metadata.
    let parent = save_path.parent().expect("parent");
    let basename = save_path.file_stem().unwrap().to_str().unwrap();
    let generation = meta
        .sidecar_generation
        .as_deref()
        .expect("current metadata generation");
    let sidecar_parent = parent.join(generation);
    assert_eq!(meta.sidecar_basename.as_deref(), Some(basename));
    assert!(
        sidecar_parent
            .join(format!("{basename}.hnsw.graph"))
            .is_file(),
        "native graph sidecar should exist"
    );
    assert!(
        sidecar_parent
            .join(format!("{basename}.hnsw.data"))
            .is_file(),
        "native data sidecar should exist"
    );

    // Load goes through the native graph path and still answers correctly.
    let (loaded, disposition) =
        HnswIndex::load_with_disposition(&save_path, &index).expect("native load");
    assert_eq!(disposition, HnswLoadDisposition::Native);
    assert_eq!(loaded.len(), 64);
    let query = normalized_vector(7, 32);
    let hits = loaded
        .knn_search(&query, 3, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert_eq!(hits[0].doc_id, "doc-0007");
}

#[test]
fn native_resave_atomically_publishes_new_generation() {
    let original_source_path = temp_path("resave_original_source", "fsvi");
    let original_source = write_index(
        &original_source_path,
        &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]],
    )
    .expect("original source");
    let original_ann = HnswIndex::build_from_vector_index(&original_source, HnswConfig::default())
        .expect("original ann");
    let destination = temp_path("resave_destination", "hnsw");
    original_ann
        .save(&destination)
        .expect("seed occupied destination");
    let original_meta: HnswMeta =
        serde_json::from_slice(&std::fs::read(&destination).expect("original metadata"))
            .expect("parse original metadata");
    let original_generation = original_meta
        .sidecar_generation
        .as_deref()
        .expect("original generation");
    let destination_parent = destination.parent().expect("destination parent");
    let original_sidecar_parent = destination_parent.join(original_generation);
    // Load a graph over the same IDs/count/dimension but different vectors.
    // hnsw_rs marks every loaded graph as mmap-backed. Saving this value
    // over occupied metadata must publish a fresh generation without
    // touching the previously authoritative pair.
    let changed_source_path = temp_path("resave_changed_source", "fsvi");
    let changed_source = write_index(
        &changed_source_path,
        &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]],
    )
    .expect("changed source");
    let changed_seed = HnswIndex::build_from_vector_index(&changed_source, HnswConfig::default())
        .expect("changed ann");
    let changed_seed_path = temp_path("resave_changed_seed", "hnsw");
    changed_seed
        .save(&changed_seed_path)
        .expect("save changed native graph");
    let changed_meta: HnswMeta = serde_json::from_slice(
        &std::fs::read(&changed_seed_path).expect("changed native metadata"),
    )
    .expect("parse changed native metadata");
    let changed_loaded =
        HnswIndex::try_load_native_graph(&changed_seed_path, &changed_meta, &changed_source)
            .expect("force changed graph through the native mmap-backed load path");
    changed_loaded
        .save(&destination)
        .expect("resave loaded graph over occupied destination");

    let meta: HnswMeta = serde_json::from_slice(&std::fs::read(&destination).expect("metadata"))
        .expect("parse metadata");
    let requested = hnsw_sidecar_basename(&destination).expect("requested basename");
    let published_generation = meta
        .sidecar_generation
        .as_deref()
        .expect("current metadata must name its generation");
    let published = meta
        .sidecar_basename
        .as_deref()
        .expect("current metadata must name its native pair");
    assert_ne!(
        published_generation, original_generation,
        "every save must publish a collision-free generation"
    );
    assert_eq!(
        published, requested,
        "metadata must record file_dump's basename"
    );
    let published_parent = destination_parent.join(published_generation);
    assert!(
        published_parent
            .join(format!("{published}.hnsw.graph"))
            .is_file()
    );
    assert!(
        published_parent
            .join(format!("{published}.hnsw.data"))
            .is_file()
    );
    assert!(
        !original_sidecar_parent.exists(),
        "a successfully published replacement must reclaim its superseded generation"
    );
    assert_eq!(
        ready_generation_paths(&destination, changed_loaded.vector_fingerprint),
        vec![published_parent.clone()],
        "the published metadata generation must be the only retained generation for this vector state"
    );

    let native = HnswIndex::try_load_native_graph(&destination, &meta, &changed_source)
        .expect("published metadata must admit the returned native pair");
    let native_hits = native
        .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search published native graph");
    assert_eq!(native_hits[0].doc_id, "doc-0001");

    let reloaded =
        HnswIndex::load(&destination, &changed_source).expect("reload atomically published graph");
    let reloaded_hits = reloaded
        .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search reloaded graph");
    assert_eq!(reloaded_hits[0].doc_id, "doc-0001");
}

#[cfg(unix)]
#[test]
fn generation_gc_never_follows_or_removes_a_generation_named_symlink() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().expect("temporary generation root");
    let parent = root.path().join("ann");
    std::fs::create_dir(&parent).expect("create ANN parent");
    let retained_generation = ".vector.fast.generation-0000000000000001-retained";
    std::fs::create_dir(parent.join(retained_generation)).expect("create retained generation");

    let external = root.path().join("external");
    std::fs::create_dir(&external).expect("create external directory");
    let external_sentinel = external.join("must-survive");
    std::fs::write(&external_sentinel, b"outside generation namespace")
        .expect("write external sentinel");
    let linked_generation = ".vector.fast.generation-0000000000000002-link";
    let linked_path = parent.join(linked_generation);
    symlink(&external, &linked_path).expect("link external directory into generation namespace");

    let metadata = HnswMeta {
        format_version: HNSW_META_FORMAT_CURRENT,
        doc_ids: Vec::new(),
        source_positions: Vec::new(),
        source_record_count: 0,
        config: HnswConfig::default(),
        dimension: 0,
        vector_fingerprint: 0,
        sidecar_generation: Some(retained_generation.to_owned()),
        sidecar_basename: Some("vector.fast".to_owned()),
        source_identity: None,
    };
    gc_superseded_hnsw_generations(&parent, "vector.fast", &metadata)
        .expect("GC must skip generation-named symlinks");

    assert!(parent.join(retained_generation).is_dir());
    assert!(
        std::fs::symlink_metadata(&linked_path)
            .expect("inspect generation-named symlink")
            .file_type()
            .is_symlink(),
        "GC must not remove a symlink merely because its name matches the generation prefix"
    );
    assert!(
        external_sentinel.is_file(),
        "GC must not traverse into the target of a generation-named symlink"
    );
}

#[cfg(unix)]
#[test]
fn native_publication_honors_umask_for_metadata_and_generation_artifacts() {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir().expect("temporary ANN root");
    let source_path = root.path().join("source.fsvi");
    let source = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
        .expect("write source index");
    let ann = HnswIndex::build_from_vector_index(&source, HnswConfig::default())
        .expect("build native ANN");
    let metadata_path = root.path().join("vector.fast.hnsw");
    ann.save(&metadata_path).expect("save native ANN");

    let source_mode = std::fs::metadata(&source_path)
        .expect("inspect source index")
        .permissions()
        .mode()
        & 0o666;
    let metadata_mode = std::fs::metadata(&metadata_path)
        .expect("inspect HNSW metadata")
        .permissions()
        .mode()
        & 0o666;
    assert_eq!(
        metadata_mode, source_mode,
        "persisted HNSW metadata must retain the parent deployment's umask policy"
    );

    let metadata: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read HNSW metadata"))
            .expect("parse HNSW metadata");
    let generation = root.path().join(
        metadata
            .sidecar_generation
            .as_deref()
            .expect("metadata must name its generation"),
    );
    let generation_mode = std::fs::metadata(&generation)
        .expect("inspect HNSW generation")
        .permissions()
        .mode()
        & 0o666;
    assert_eq!(
        generation_mode, source_mode,
        "published generation data permissions must retain the parent deployment's umask policy"
    );
    let basename = metadata
        .sidecar_basename
        .as_deref()
        .expect("metadata must name its native sidecars");
    for path in [
        generation.join(format!("{basename}.hnsw.graph")),
        generation.join(format!("{basename}.hnsw.data")),
        generation.join(HNSW_GENERATION_RECEIPT_FILENAME),
    ] {
        assert_eq!(
            std::fs::metadata(&path)
                .expect("inspect generation artifact")
                .permissions()
                .mode()
                & 0o666,
            source_mode,
            "generation artifact '{}' must retain the parent deployment's umask policy",
            path.display()
        );
    }
}

#[test]
fn load_rebuilds_from_legacy_v1_metadata() {
    let fsvi_path = temp_path("persist_v1", "fsvi");
    let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
    let index = write_index(&fsvi_path, &vectors).expect("index");
    let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

    let save_path = temp_path("persist_v1", "hnsw");
    ann.save(&save_path).expect("save");

    // Fabricate a legacy v1 sidecar at a fresh path: identical metadata but
    // with `format_version` stripped (deserializes to 0) and no graph/data
    // sidecars beside it. load() must transparently rebuild from the
    // VectorIndex instead of failing.
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
    let object = value.as_object_mut().expect("object");
    object.remove("format_version");
    object.remove("sidecar_generation");
    object.remove("sidecar_basename");
    let v1_path = temp_path("persist_v1_legacy", "hnsw");
    std::fs::write(&v1_path, serde_json::to_vec(&value).expect("v1 json")).expect("write v1");

    let (loaded, disposition) =
        HnswIndex::load_with_disposition(&v1_path, &index).expect("v1 rebuild load");
    assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
    assert_eq!(loaded.len(), 64);
    let query = normalized_vector(10, 32);
    let hits = loaded
        .knn_search(&query, 5, HNSW_DEFAULT_EF_SEARCH)
        .expect("search");
    assert_eq!(hits[0].doc_id, "doc-0010");
}

#[test]
fn fallback_rebuild_preserves_duplicate_doc_id_vector_alignment() {
    let source_path = temp_path("duplicate-id-source", "fsvi");
    let mut writer =
        VectorIndex::create_with_revision(&source_path, "hash", "test", 2, Quantization::F32)
            .expect("create source index");
    writer
        .write_record("duplicate", &[1.0_f32, 0.0])
        .expect("write first duplicate");
    writer
        .write_record("duplicate", &[0.0_f32, 1.0])
        .expect("write second duplicate");
    writer.finish().expect("finish source index");
    let source_index = VectorIndex::open(&source_path).expect("open source index");

    let legacy_path = temp_path("duplicate-id-legacy", "hnsw");
    let legacy_meta = HnswMeta {
        format_version: 0,
        doc_ids: vec!["duplicate".to_owned(), "duplicate".to_owned()],
        source_positions: Vec::new(),
        source_record_count: 0,
        config: HnswConfig::default(),
        dimension: 2,
        vector_fingerprint: 0,
        sidecar_generation: None,
        sidecar_basename: None,
        source_identity: None,
    };
    std::fs::write(
        &legacy_path,
        serde_json::to_vec(&legacy_meta).expect("serialize legacy metadata"),
    )
    .expect("write legacy metadata");

    let (rebuilt, disposition) =
        HnswIndex::load_with_disposition(&legacy_path, &source_index).expect("fallback rebuild");
    assert_eq!(disposition, HnswLoadDisposition::Rebuilt);

    let expected_doc_ids = vec!["duplicate".to_owned(), "duplicate".to_owned()];
    let expected_vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    assert_eq!(
        rebuilt.vector_fingerprint,
        fingerprint_vectors(&expected_doc_ids, &expected_vectors),
        "fallback must fingerprint the vector at each source row, not the first row sharing its ID"
    );

    let first = rebuilt
        .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search first duplicate vector");
    let second = rebuilt
        .knn_search(&[0.0, 1.0], 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search second duplicate vector");
    assert_eq!(first[0].index, 0);
    assert_eq!(second[0].index, 1);
}

#[test]
fn load_never_treats_v5_native_graph_as_v6() {
    // The source index says doc-0000 is e0 and doc-0001 is e1.
    let source_path = temp_path("persist_v5_source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");

    // Build a valid native graph with the same document IDs but the two
    // vectors swapped, then mark its metadata as v5 while forging the live
    // source fingerprint. That makes the format boundary the sole guard: if
    // load accepts v5 as current, an e0 query returns doc-0001. Correct
    // v6-only loading rebuilds from source and returns doc-0000.
    let swapped_path = temp_path("persist_v5_swapped", "fsvi");
    let swapped_index = write_index(&swapped_path, &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]])
        .expect("swapped index");
    let swapped_ann = HnswIndex::build_from_vector_index(&swapped_index, HnswConfig::default())
        .expect("swapped ann");
    let save_path = temp_path("persist_v5_native", "hnsw");
    swapped_ann.save(&save_path).expect("save native graph");

    let mut meta: HnswMeta =
        serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
    meta.format_version = 5;
    meta.vector_fingerprint =
        fingerprint_live_vector_index(&source_index, 2, 2).expect("live source fingerprint");
    std::fs::write(&save_path, serde_json::to_vec(&meta).expect("serialize v5"))
        .expect("write v5 metadata");

    let (loaded, disposition) =
        HnswIndex::load_with_disposition(&save_path, &source_index).expect("rebuild v5");
    assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
    let hits = loaded
        .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search rebuilt graph");
    assert_eq!(
        hits[0].doc_id, "doc-0000",
        "v5 graph without topology attestation must rebuild under v6"
    );
}

#[test]
fn current_native_graph_with_stale_digest_receipt_rebuilds_before_load() {
    let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let source_path = temp_path("stale-receipt-source", "fsvi");
    let source_index = write_index(&source_path, &vectors).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("ANN graph");
    let metadata_path = temp_path("stale-receipt", "hnsw");
    ann.save(&metadata_path).expect("save ANN graph");
    let meta: HnswMeta = serde_json::from_slice(&std::fs::read(&metadata_path).expect("metadata"))
        .expect("parse metadata");
    let (generation_path, basename) =
        persisted_hnsw_sidecar_location(&metadata_path, &meta).expect("sidecar location");
    let graph_path = generation_path.join(format!("{basename}.hnsw.graph"));
    std::fs::OpenOptions::new()
        .append(true)
        .open(&graph_path)
        .expect("open graph for corruption")
        .write_all(&[0])
        .expect("append unreceipted byte");

    let (_, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
        .expect("stale digest must rebuild");
    assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
}

#[test]
fn current_metadata_rejects_missing_or_nonlocal_sidecar_locations() {
    let source_path = temp_path("persist_location_validation_source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("build graph");
    let save_path = temp_path("persist_location_validation", "hnsw");
    ann.save(&save_path).expect("save native graph");
    let metadata_bytes = std::fs::read(&save_path).expect("metadata bytes");

    for (field, replacement) in [
        ("sidecar_generation", None),
        ("sidecar_basename", None),
        ("sidecar_generation", Some("../escape")),
        ("sidecar_basename", Some("nested/escape")),
    ] {
        let mut value: serde_json::Value =
            serde_json::from_slice(&metadata_bytes).expect("metadata value");
        let object = value.as_object_mut().expect("metadata object");
        if let Some(replacement) = replacement {
            object.insert(field.to_owned(), replacement.into());
        } else {
            object.remove(field);
        }
        let meta: HnswMeta = serde_json::from_value(value).expect("parse corrupted metadata");
        assert!(
            persisted_hnsw_sidecar_location(&save_path, &meta).is_err(),
            "invalid {field} must fail location validation"
        );
        assert!(
            HnswIndex::try_load_native_graph(&save_path, &meta, &source_index).is_none(),
            "invalid {field} must fail native loading closed"
        );
    }
}

#[cfg(unix)]
#[test]
fn current_metadata_rejects_symlinked_native_sidecars() {
    use std::os::unix::fs::symlink;

    let source_path = temp_path("persist_symlink_source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("build graph");
    let save_path = temp_path("persist_symlink_validation", "hnsw");
    ann.save(&save_path).expect("save native graph");
    let metadata_bytes = std::fs::read(&save_path).expect("metadata bytes");
    let original: HnswMeta =
        serde_json::from_slice(&metadata_bytes).expect("parse native metadata");
    let parent = save_path.parent().expect("metadata parent");
    let generation = original
        .sidecar_generation
        .as_deref()
        .expect("native generation");
    let basename = original
        .sidecar_basename
        .as_deref()
        .expect("native basename");
    let original_generation = parent.join(generation);

    let generation_link_name = format!("{generation}-link");
    symlink(&original_generation, parent.join(&generation_link_name))
        .expect("create generation symlink");
    let mut generation_link_meta: HnswMeta =
        serde_json::from_slice(&metadata_bytes).expect("parse generation-link metadata");
    generation_link_meta.sidecar_generation = Some(generation_link_name);
    assert!(
        HnswIndex::try_load_native_graph(&save_path, &generation_link_meta, &source_index)
            .is_none(),
        "native loading must not follow a generation symlink"
    );

    let sidecar_link_generation_name = format!("{generation}-sidecar-links");
    let sidecar_link_generation = parent.join(&sidecar_link_generation_name);
    std::fs::create_dir(&sidecar_link_generation).expect("create sidecar-link generation");
    for suffix in [".hnsw.graph", ".hnsw.data"] {
        symlink(
            original_generation.join(format!("{basename}{suffix}")),
            sidecar_link_generation.join(format!("{basename}{suffix}")),
        )
        .expect("create native sidecar symlink");
    }
    let mut sidecar_link_meta: HnswMeta =
        serde_json::from_slice(&metadata_bytes).expect("parse sidecar-link metadata");
    sidecar_link_meta.sidecar_generation = Some(sidecar_link_generation_name);
    assert!(
        HnswIndex::try_load_native_graph(&save_path, &sidecar_link_meta, &source_index).is_none(),
        "native loading must not follow graph or data symlinks"
    );
}

#[test]
fn load_rejects_current_sidecar_with_missing_vector_fingerprint() {
    let source_path = temp_path("persist_missing_fp_source", "fsvi");
    let source_index =
        write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]]).expect("source index");

    // Persist a current-format native graph whose vectors are swapped
    // behind the same document IDs, then remove only its fingerprint field.
    // Treating the serde default of 0 as an opt-out would admit this stale
    // graph and make an e0 query return doc-0001.
    let swapped_path = temp_path("persist_missing_fp_swapped", "fsvi");
    let swapped_index = write_index(&swapped_path, &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]])
        .expect("swapped index");
    let swapped_ann = HnswIndex::build_from_vector_index(&swapped_index, HnswConfig::default())
        .expect("swapped ann");
    let save_path = temp_path("persist_missing_fp_native", "hnsw");
    swapped_ann.save(&save_path).expect("save native graph");

    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
    value
        .as_object_mut()
        .expect("metadata object")
        .remove("vector_fingerprint");
    std::fs::write(
        &save_path,
        serde_json::to_vec(&value).expect("serialize metadata without fingerprint"),
    )
    .expect("write metadata without fingerprint");

    let loaded = HnswIndex::load(&save_path, &source_index)
        .expect("missing fingerprint must rebuild from source");
    let hits = loaded
        .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search rebuilt graph");
    assert_eq!(
        hits[0].doc_id, "doc-0000",
        "current-format metadata cannot omit its vector fingerprint to admit a stale graph"
    );
}

#[test]
fn load_rejects_native_sidecar_when_previously_unsampled_vector_changes() {
    // With 300 rows, the v3 fingerprint's ceil(300 / 256) stride was 2.
    // Row 1 was therefore neither an even-stride sample nor the final row.
    const VECTOR_COUNT: usize = 300;
    let source_path = temp_path("persist_unsampled_source", "fsvi");
    let original_vectors: Vec<Vec<f32>> = (0..VECTOR_COUNT)
        .map(|i| normalized_vector(i, 32))
        .collect();
    let source_index = write_index(&source_path, &original_vectors).expect("source index");
    let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
        .expect("build original graph");
    let save_path = temp_path("persist_unsampled_native", "hnsw");
    ann.save(&save_path).expect("save native graph");

    // Keep the document IDs, count, and dimension identical while changing
    // only the old scheme's unsampled row. Negating the original unit
    // vector makes doc-0001 the worst possible match in the stale graph and
    // the exact match after a rebuild, keeping the result proof decisive.
    let changed_vector: Vec<f32> = original_vectors[1].iter().map(|value| -*value).collect();
    let mut changed_vectors = original_vectors;
    changed_vectors[1].clone_from(&changed_vector);
    let changed_path = temp_path("persist_unsampled_changed", "fsvi");
    let changed_index = write_index(&changed_path, &changed_vectors).expect("changed index");

    let loaded = HnswIndex::load(&save_path, &changed_index)
        .expect("fingerprint mismatch must rebuild from changed source");
    let hits = loaded
        .knn_search(&changed_vector, 1, VECTOR_COUNT)
        .expect("search rebuilt graph");
    assert_eq!(
        hits[0].doc_id, "doc-0001",
        "every live vector must affect persisted source identity; otherwise the stale v3 \
         graph survives a change to row 1"
    );
}

/// Stale-vectors validation: the native-graph load path must reject a
/// sidecar whose persisted vector fingerprint disagrees with the live
/// `VectorIndex` (i.e. someone swapped the FSVI contents behind matching
/// doc IDs), and transparently fall back to rebuild rather than silently
/// serving hits against vectors that no longer exist. This is the exact
/// scenario the prompt for `frankensearch#25` calls out.
#[test]
fn load_rejects_native_sidecar_when_vectors_swapped_under_same_doc_ids() {
    // Build + save a native sidecar over an FSVI where doc-i ≈ basis_i.
    let fsvi_path = temp_path("persist_swap", "fsvi");
    let original_vectors: Vec<Vec<f32>> = (0..16).map(|i| normalized_vector(i, 32)).collect();
    let original_index = write_index(&fsvi_path, &original_vectors).expect("index");
    let ann =
        HnswIndex::build_from_vector_index(&original_index, HnswConfig::default()).expect("build");

    let save_path = temp_path("persist_swap", "hnsw");
    ann.save(&save_path).expect("save native graph");

    // Sanity: a load against the *original* FSVI takes the fast path and
    // returns doc-0007 for query≈doc-0007.
    let loaded_original =
        HnswIndex::load(&save_path, &original_index).expect("native load matches");
    let query = normalized_vector(7, 32);
    let original_hits = loaded_original
        .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search original");
    assert_eq!(original_hits[0].doc_id, "doc-0007");
    // Sanity that the fingerprint actually got stamped.
    let mut meta: HnswMeta =
        serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse meta");
    assert_ne!(
        meta.vector_fingerprint, 0,
        "native save must stamp a fingerprint"
    );

    // Now swap vectors behind the same doc IDs (doc-0007 now points at
    // basis_99) while leaving the .hnsw.graph / .hnsw.data sidecars
    // unchanged. A naive native fast path would return doc-0007 against the
    // *old* graph; the fingerprint guard forces a rebuild instead.
    let mut swapped_vectors = original_vectors.clone();
    swapped_vectors[7] = normalized_vector(99, 32);
    let swapped_path = temp_path("persist_swap_after", "fsvi");
    let swapped_index = write_index(&swapped_path, &swapped_vectors).expect("swapped");

    // Copy the metadata + graph + data sidecars next to the swapped FSVI
    // so the load path's directory layout is plausible.
    let swap_save_path = temp_path("persist_swap_after", "hnsw");
    let src_parent = save_path.parent().expect("src parent");
    let dst_parent = swap_save_path.parent().expect("dst parent");
    let src_generation = meta
        .sidecar_generation
        .as_deref()
        .expect("source metadata generation");
    let src_stem = meta
        .sidecar_basename
        .as_deref()
        .expect("source metadata basename");
    let dst_stem = swap_save_path.file_stem().unwrap().to_str().unwrap();
    let dst_generation = format!(".{dst_stem}.relocated");
    let dst_sidecar_parent = dst_parent.join(&dst_generation);
    std::fs::create_dir(&dst_sidecar_parent).expect("create relocated generation");
    for ext in ["hnsw.graph", "hnsw.data"] {
        std::fs::copy(
            src_parent
                .join(src_generation)
                .join(format!("{src_stem}.{ext}")),
            dst_sidecar_parent.join(format!("{dst_stem}.{ext}")),
        )
        .expect("copy sidecar");
    }
    meta.sidecar_generation = Some(dst_generation);
    meta.sidecar_basename = Some(dst_stem.to_owned());
    std::fs::write(
        &swap_save_path,
        serde_json::to_vec(&meta).expect("serialize relocated metadata"),
    )
    .expect("write relocated metadata");

    // Load against the swapped FSVI. The fingerprint mismatch must trigger
    // the rebuild fallback, which sees doc-0007 ≈ basis_99 and therefore
    // returns doc-0007 *only* when querying ≈ basis_99 — not when querying
    // basis_7.
    let loaded_swapped = HnswIndex::load(&swap_save_path, &swapped_index).expect("load after swap");
    let stale_query = normalized_vector(7, 32);
    let stale_hits = loaded_swapped
        .knn_search(&stale_query, 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search stale");
    // The persisted graph would have returned doc-0007 (basis_7) here; the
    // rebuilt graph against the swapped FSVI returns whichever doc now
    // *actually* sits near basis_7, which is **not** doc-0007.
    assert_ne!(
        stale_hits[0].doc_id, "doc-0007",
        "fingerprint guard must reject the persisted graph and rebuild against \
         the swapped FSVI; otherwise we'd return stale ANN hits"
    );

    // And the rebuilt graph *can* find the swapped doc by its new vector.
    let new_query = normalized_vector(99, 32);
    let new_hits = loaded_swapped
        .knn_search(&new_query, 1, HNSW_DEFAULT_EF_SEARCH)
        .expect("search new");
    assert_eq!(
        new_hits[0].doc_id, "doc-0007",
        "rebuild path must have picked up the swapped vector for doc-0007"
    );
}
