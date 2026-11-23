use super::*;

// ── Error paths ──────────────────────────────────────────────────

#[test]
fn open_returns_index_not_found_when_no_fast_or_fallback() {
    let dir = temp_index_dir("missing");
    fs::create_dir_all(&dir).expect("create temp dir");
    let error = TwoTierIndex::open(&dir, TwoTierConfig::default()).unwrap_err();
    let paths = match &error {
        SearchError::IndexCandidatesNotFound { paths } => paths.as_slice(),
        _ => &[],
    };
    let expected_paths = [
        dir.join(VECTOR_INDEX_FAST_FILENAME),
        dir.join(VECTOR_INDEX_FALLBACK_FILENAME),
    ];
    assert_eq!(
        paths,
        expected_paths.as_slice(),
        "unexpected error variant: {error:?}"
    );
    let message = error.to_string();
    assert!(message.contains(VECTOR_INDEX_FAST_FILENAME));
    assert!(message.contains(VECTOR_INDEX_FALLBACK_FILENAME));
}

#[test]
fn open_with_paths_supports_consumer_owned_filenames() {
    let dir = temp_index_dir("explicit-paths");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-384.fsvi");
    let quality_path = dir.join("index-minilm-384.fsvi");
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
    )
    .expect("write custom fast index");
    write_index_file(
        &quality_path,
        &[("doc-a", &[0.5, 0.5]), ("doc-b", &[1.0, 0.0])],
    )
    .expect("write custom quality index");

    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
    let index = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default())
        .expect("open custom two-tier paths");

    assert_eq!(paths.fast_index(), fast_path);
    assert_eq!(paths.quality_index(), Some(quality_path.as_path()));
    assert!(index.has_quality_index());
    assert_eq!(index.doc_count(), 2);
    assert_eq!(index.fast_embedder_id(), "test");
    assert_eq!(index.fast_embedder_revision(), "");
    assert_eq!(index.quality_embedder_id(), Some("test"));
    assert_eq!(index.quality_embedder_revision(), Some(""));
    assert_eq!(index.fast_index_path(), fast_path);
    assert_eq!(index.quality_index_path(), Some(quality_path.as_path()));
    assert!(!dir.join(VECTOR_INDEX_FAST_FILENAME).exists());
    assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());

    let hits = index.search_fast(&[1.0, 0.0], 2).expect("search fast");
    let quality_scores = index
        .quality_scores_for_hits(&[1.0, 0.0], &hits)
        .expect("score quality tier");
    assert_eq!(quality_scores.len(), hits.len());
    assert!(quality_scores.iter().all(Option::is_some));
}

#[test]
fn open_with_paths_supports_custom_fast_only_without_copying() {
    let dir = temp_index_dir("explicit-fast-only");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-384.fsvi");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write custom fast index");

    let paths = TwoTierIndexPaths::new(&fast_path);
    let index = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default())
        .expect("open custom fast-only path");

    assert_eq!(index.fast_index_path(), fast_path);
    assert_eq!(index.quality_index_path(), None);
    assert!(!index.has_quality_index());
    assert!(!dir.join(VECTOR_INDEX_FAST_FILENAME).exists());
    assert!(!dir.join(VECTOR_INDEX_FALLBACK_FILENAME).exists());
    assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());
}

#[test]
fn open_with_paths_reports_the_explicit_missing_path() {
    let dir = temp_index_dir("explicit-missing");
    fs::create_dir_all(&dir).expect("create temp dir");
    let missing_path = dir.join("index-custom.fsvi");
    let paths = TwoTierIndexPaths::new(&missing_path);

    let error = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default()).unwrap_err();
    assert!(
        matches!(
            error,
            SearchError::IndexNotFound { ref path } if path == &missing_path
        ),
        "expected exact explicit path, got {error:?}"
    );
}

#[test]
fn open_with_paths_reports_an_explicit_missing_quality_path() {
    let dir = temp_index_dir("explicit-missing-quality");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-384.fsvi");
    let missing_quality_path = dir.join("index-minilm-384.fsvi");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write custom fast index");
    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&missing_quality_path);

    let error = TwoTierIndex::open_with_paths(&paths, TwoTierConfig::default()).unwrap_err();
    assert!(
        matches!(
            error,
            SearchError::IndexNotFound { ref path } if path == &missing_quality_path
        ),
        "expected exact missing quality path, got {error:?}"
    );
}

#[cfg(unix)]
#[test]
fn explicit_fast_final_symlink_to_distinct_target_is_rejected() {
    use std::os::unix::fs::symlink;

    let dir = temp_index_dir("explicit-fast-distinct-symlink");
    fs::create_dir_all(&dir).expect("create temp dir");
    let target_path = dir.join("target.fsvi");
    let symlink_path = dir.join("index-fast.fsvi");
    write_index_file(&target_path, &[("doc-a", &[1.0, 0.0])]).expect("write distinct fast target");
    symlink(&target_path, &symlink_path).expect("create fast final symlink");

    let error = TwoTierIndex::open_with_paths(
        &TwoTierIndexPaths::new(&symlink_path),
        TwoTierConfig::default(),
    )
    .expect_err("a distinct-target final symlink must be rejected");

    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
        ),
        "unexpected final-symlink error: {error:?}"
    );
    assert_eq!(
        fs::read(&target_path).expect("read preserved target"),
        fs::read(&symlink_path).expect("read through preserved symlink")
    );
}

#[cfg(unix)]
#[test]
fn explicit_dangling_fast_final_symlink_is_rejected_as_configuration() {
    use std::os::unix::fs::symlink;

    let dir = temp_index_dir("explicit-fast-dangling-symlink");
    fs::create_dir_all(&dir).expect("create temp dir");
    let missing_target = dir.join("missing-target.fsvi");
    let symlink_path = dir.join("index-fast.fsvi");
    symlink(&missing_target, &symlink_path).expect("create dangling fast symlink");

    let error = TwoTierIndex::open_with_paths(
        &TwoTierIndexPaths::new(&symlink_path),
        TwoTierConfig::default(),
    )
    .expect_err("a dangling final symlink must be rejected");

    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
        ),
        "unexpected dangling-symlink error: {error:?}"
    );
    assert!(!missing_target.exists());
    assert!(fs::symlink_metadata(&symlink_path).is_ok());
}

#[cfg(feature = "ann")]
#[test]
fn explicit_ann_paths_are_inspectable() {
    let fast_index = PathBuf::from("index-fast.fsvi");
    let quality_index = PathBuf::from("index-quality.fsvi");
    let fast_ann = PathBuf::from("index-fast.hnsw");
    let quality_ann = PathBuf::from("index-quality.hnsw");
    let paths = TwoTierIndexPaths::new(&fast_index)
        .with_quality_index(&quality_index)
        .with_fast_ann(&fast_ann)
        .with_quality_ann(&quality_ann);

    assert_eq!(paths.fast_index(), fast_index);
    assert_eq!(paths.quality_index(), Some(quality_index.as_path()));
    assert_eq!(paths.fast_ann(), Some(fast_ann.as_path()));
    assert_eq!(paths.quality_ann(), Some(quality_ann.as_path()));
}

#[cfg(feature = "ann")]
#[test]
fn explicit_path_roles_must_not_alias() {
    let cases = [
        TwoTierIndexPaths::new("fast.fsvi").with_quality_index("fast.fsvi"),
        TwoTierIndexPaths::new("fast.fsvi").with_fast_ann("fast.fsvi"),
        TwoTierIndexPaths::new("fast.fsvi")
            .with_quality_index("quality.fsvi")
            .with_quality_ann("fast.fsvi"),
        TwoTierIndexPaths::new("fast.fsvi")
            .with_quality_index("quality.fsvi")
            .with_fast_ann("quality.fsvi"),
        TwoTierIndexPaths::new("fast.fsvi")
            .with_quality_index("quality.fsvi")
            .with_quality_ann("quality.fsvi"),
        TwoTierIndexPaths::new("fast.fsvi")
            .with_quality_index("quality.fsvi")
            .with_fast_ann("shared.hnsw")
            .with_quality_ann("shared.hnsw"),
    ];

    for paths in cases {
        let error = validate_index_paths(&paths).expect_err("aliased roles must be rejected");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
            ),
            "unexpected alias error: {error:?}"
        );
    }
}

#[cfg(feature = "ann")]
#[test]
fn quality_ann_requires_a_quality_index() {
    let paths = TwoTierIndexPaths::new("fast.fsvi").with_quality_ann("orphan-quality.hnsw");
    let error = validate_index_paths(&paths).expect_err("orphan quality ANN must be rejected");

    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "quality_ann"
        ),
        "unexpected quality ANN error: {error:?}"
    );
}

#[cfg(feature = "ann")]
#[test]
fn custom_ann_path_builds_without_creating_a_conventional_sidecar() {
    let dir = temp_index_dir("explicit-custom-ann");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let fast_ann_path = dir.join("index-fnv1a-4.hnsw");
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ],
    )
    .expect("write custom fast index");
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let index = TwoTierIndex::open_with_paths(&paths, config).expect("open and build custom ANN");

    assert!(index.has_fast_ann());
    assert!(fast_ann_path.exists());
    assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
}

#[cfg(feature = "ann")]
#[test]
fn native_custom_ann_reopen_does_not_materialize_a_missing_save_lock() {
    let dir = temp_index_dir("explicit-native-ann-read-only-reopen");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let fast_ann_path = dir.join("index-fnv1a-4.hnsw");
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ],
    )
    .expect("write custom fast index");
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };
    let first =
        TwoTierIndex::open_with_paths(&paths, config.clone()).expect("build custom ANN sidecar");
    assert!(first.has_fast_ann());
    load_native_ann_sidecar(&fast_ann_path, first.fast_tier());
    drop(first);

    let lock_path = crate::hnsw::hnsw_save_lock_artifact_path(&fast_ann_path).expect("lock path");
    let retained_lock = lock_path.with_extension("lock.retained-for-native-reopen");
    fs::rename(&lock_path, &retained_lock).expect("retain lock under a non-active name");
    assert!(!lock_path.exists());
    assert!(retained_lock.exists());

    let reopened =
        TwoTierIndex::open_with_paths(&paths, config).expect("native read-only-style reopen");
    assert!(reopened.has_fast_ann());
    assert!(
        !lock_path.exists(),
        "a native-valid reopen must not require or recreate a writable save lock"
    );
    assert!(retained_lock.exists());
}

#[cfg(feature = "ann")]
#[test]
fn custom_ann_save_creates_a_missing_parent_and_reopens_natively() {
    let dir = temp_index_dir("explicit-custom-ann-missing-parent");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let fast_ann_path = dir
        .join("consumer-owned")
        .join("ann")
        .join("index-fnv1a-4.hnsw");
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ],
    )
    .expect("write custom fast index");
    assert!(!fast_ann_path.parent().expect("ANN parent").exists());
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let first = TwoTierIndex::open_with_paths(&paths, config.clone())
        .expect("build ANN below a newly created custom parent");
    assert!(first.has_fast_ann());
    load_native_ann_sidecar(&fast_ann_path, first.fast_tier());
    drop(first);

    let reopened = TwoTierIndex::open_with_paths(&paths, config).expect("native custom ANN reopen");
    assert!(reopened.has_fast_ann());
    load_native_ann_sidecar(&fast_ann_path, reopened.fast_tier());
    assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
}

#[cfg(feature = "ann")]
#[test]
fn custom_fast_and_quality_ann_paths_reopen_through_native_sidecars() {
    let dir = temp_index_dir("explicit-custom-two-tier-ann");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let quality_path = dir.join("index-minilm-4.fsvi");
    let fast_ann_path = dir.join("index-fnv1a-4.hnsw");
    let quality_ann_path = dir.join("index-minilm-4.hnsw");
    let rows = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0][..]),
        ("doc-c", &[0.0, 0.0, 1.0, 0.0][..]),
    ];
    write_index_file(&fast_path, &rows).expect("write custom fast index");
    write_index_file(&quality_path, &rows).expect("write custom quality index");
    let paths = TwoTierIndexPaths::new(&fast_path)
        .with_quality_index(&quality_path)
        .with_fast_ann(&fast_ann_path)
        .with_quality_ann(&quality_ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let first =
        TwoTierIndex::open_with_paths(&paths, config.clone()).expect("build both custom ANN tiers");
    assert!(first.has_fast_ann());
    assert!(first.has_quality_ann());
    load_native_ann_sidecar(&fast_ann_path, first.fast_tier());
    load_native_ann_sidecar(
        &quality_ann_path,
        first.quality_tier().expect("quality index"),
    );
    let fast_metadata = fs::read(&fast_ann_path).expect("read fast ANN metadata");
    let quality_metadata = fs::read(&quality_ann_path).expect("read quality ANN metadata");
    drop(first);

    let reopened =
        TwoTierIndex::open_with_paths(&paths, config).expect("reopen both custom ANN tiers");
    assert!(reopened.has_fast_ann());
    assert!(reopened.has_quality_ann());
    load_native_ann_sidecar(&fast_ann_path, reopened.fast_tier());
    load_native_ann_sidecar(
        &quality_ann_path,
        reopened.quality_tier().expect("quality index"),
    );
    assert_eq!(
        fs::read(&fast_ann_path).expect("reread fast ANN metadata"),
        fast_metadata,
        "native reopen must not republish fast ANN metadata"
    );
    assert_eq!(
        fs::read(&quality_ann_path).expect("reread quality ANN metadata"),
        quality_metadata,
        "native reopen must not republish quality ANN metadata"
    );
    assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
    assert!(!dir.join(VECTOR_ANN_QUALITY_FILENAME).exists());
}

#[cfg(feature = "ann")]
#[test]
fn custom_quality_only_ann_path_does_not_enable_or_create_fast_ann() {
    let dir = temp_index_dir("explicit-custom-quality-only-ann");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let quality_path = dir.join("index-minilm-4.fsvi");
    let quality_ann_path = dir.join("index-minilm-4.hnsw");
    let rows = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0][..]),
    ];
    write_index_file(&fast_path, &rows).expect("write custom fast index");
    write_index_file(&quality_path, &rows).expect("write custom quality index");
    let paths = TwoTierIndexPaths::new(&fast_path)
        .with_quality_index(&quality_path)
        .with_quality_ann(&quality_ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let first = TwoTierIndex::open_with_paths(&paths, config.clone())
        .expect("build only the custom quality ANN tier");
    assert!(!first.has_fast_ann());
    assert!(first.has_quality_ann());
    load_native_ann_sidecar(
        &quality_ann_path,
        first.quality_tier().expect("quality index"),
    );
    drop(first);

    let reopened =
        TwoTierIndex::open_with_paths(&paths, config).expect("reopen quality-only custom ANN");
    assert!(!reopened.has_fast_ann());
    assert!(reopened.has_quality_ann());
    load_native_ann_sidecar(
        &quality_ann_path,
        reopened.quality_tier().expect("quality index"),
    );
    assert!(!dir.join("index-fnv1a-4.hnsw").exists());
    assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
    assert!(!dir.join(VECTOR_ANN_QUALITY_FILENAME).exists());
}

#[cfg(feature = "ann")]
#[test]
fn ann_persistence_revalidation_rejects_new_save_lock_alias() {
    let dir = temp_index_dir("ann-save-lock-alias-race");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fast.fsvi");
    let missing_quality_path = dir.join("quality-will-alias-lock.fsvi");
    let ann_path = dir.join("index-fast.hnsw");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");
    let paths = TwoTierIndexPaths::new(&fast_path)
        .with_quality_index(&missing_quality_path)
        .with_fast_ann(&ann_path);

    validate_ann_persistence_paths(&paths, true, false)
        .expect("initial roles and materialized save lock are distinct");
    let lock_path =
        crate::hnsw::hnsw_save_lock_artifact_path(&ann_path).expect("derive save-lock path");
    fs::hard_link(&lock_path, &missing_quality_path)
        .expect("inject a configured-role alias to the materialized save lock");

    let error = validate_ann_persistence_paths(&paths, true, false)
        .expect_err("complete immediate pre-save validation must reject the new lock alias");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
        ),
        "unexpected save-lock alias error: {error:?}"
    );
    assert!(
        crate::file_identity::is_same_file(&lock_path, &missing_quality_path)
            .expect("compare injected save-lock alias")
    );
}

#[cfg(all(feature = "ann", unix))]
#[test]
fn fresh_ann_save_revalidates_paths_after_an_alias_race() {
    use std::cell::Cell;

    let dir = temp_index_dir("ann-fresh-save-alias-race");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let ann_path = dir.join("index-fnv1a-4.hnsw");
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ],
    )
    .expect("write custom fast index");
    let original = fs::read(&fast_path).expect("read original FSVI");
    let vector_index = VectorIndex::open(&fast_path).expect("open custom fast index");
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
    validate_index_paths(&paths).expect("paths start distinct");
    let save_called = Cell::new(false);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let ann = maybe_load_or_build_ann_with_save(
        &vector_index,
        &ann_path,
        1,
        &config,
        "fast",
        || {
            fs::hard_link(&fast_path, &ann_path).expect("inject hardlink alias before save");
            validate_ann_persistence_paths(&paths, true, false)
        },
        |_, _| {
            save_called.set(true);
            Ok(())
        },
    )
    .expect("keep freshly built ANN in memory after rejected persistence");

    assert!(
        !save_called.get(),
        "save must not run after revalidation fails"
    );
    assert_eq!(ann.len(), 2);
    assert!(crate::file_identity::is_same_file(&fast_path, &ann_path).expect("compare aliases"));
    assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
}

#[cfg(all(feature = "ann", unix))]
#[test]
fn rebuilt_ann_resave_revalidates_every_role_after_an_alias_race() {
    use std::cell::Cell;

    let dir = temp_index_dir("ann-rebuilt-save-alias-race");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let quality_path = dir.join("index-minilm-4.fsvi");
    let fast_ann_path = dir.join("index-fnv1a-4.hnsw");
    let quality_ann_path = dir.join("index-minilm-4.hnsw");
    let rows = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0][..]),
    ];
    write_index_file(&fast_path, &rows).expect("write custom fast index");
    write_index_file(&quality_path, &rows).expect("write custom quality index");
    let paths = TwoTierIndexPaths::new(&fast_path)
        .with_quality_index(&quality_path)
        .with_fast_ann(&fast_ann_path)
        .with_quality_ann(&quality_ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };
    let seeded = TwoTierIndex::open_with_paths(
        &TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_ann_path),
        config.clone(),
    )
    .expect("seed fast ANN sidecar");
    assert!(seeded.has_fast_ann());
    drop(seeded);

    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&fast_ann_path).expect("read ANN metadata"))
            .expect("parse ANN metadata");
    let object = metadata.as_object_mut().expect("ANN metadata object");
    object.remove("format_version");
    object.remove("sidecar_generation");
    object.remove("sidecar_basename");
    fs::write(
        &fast_ann_path,
        serde_json::to_vec(&metadata).expect("serialize legacy metadata"),
    )
    .expect("write legacy ANN metadata");

    let vector_index = VectorIndex::open(&fast_path).expect("reopen custom fast index");
    validate_index_paths(&paths).expect("paths start distinct");
    let save_called = Cell::new(false);
    let quality_original = fs::read(&quality_path).expect("read original quality FSVI");
    let ann = maybe_load_or_build_ann_with_save(
        &vector_index,
        &fast_ann_path,
        1,
        &config,
        "fast",
        || {
            fs::hard_link(&quality_path, &quality_ann_path)
                .expect("inject quality-role hardlink alias before rebuilt resave");
            validate_ann_persistence_paths(&paths, true, true)
        },
        |_, _| {
            save_called.set(true);
            Ok(())
        },
    )
    .expect("keep rebuilt ANN in memory after rejected resave");

    assert!(
        !save_called.get(),
        "resave must not run after revalidation fails"
    );
    assert_eq!(ann.len(), 2);
    assert!(
        crate::file_identity::is_same_file(&quality_path, &quality_ann_path)
            .expect("compare aliases")
    );
    assert_eq!(
        fs::read(&quality_path).expect("read preserved quality FSVI"),
        quality_original
    );
}

#[cfg(feature = "ann")]
#[test]
fn ann_path_equal_to_fsvi_is_rejected_before_any_overwrite() {
    let dir = temp_index_dir("ann-direct-alias");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
        .expect("write custom fast index");
    let original = fs::read(&fast_path).expect("read original FSVI");
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&fast_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let error = TwoTierIndex::open_with_paths(&paths, config).expect_err("alias must be rejected");

    assert!(matches!(error, SearchError::InvalidConfig { .. }));
    assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
}

#[cfg(all(feature = "ann", unix))]
#[test]
fn ann_hardlink_alias_is_rejected_before_any_overwrite() {
    let dir = temp_index_dir("ann-hardlink-alias");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let ann_path = dir.join("index-fnv1a-4.hnsw");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
        .expect("write custom fast index");
    fs::hard_link(&fast_path, &ann_path).expect("create hardlink alias");
    let original = fs::read(&fast_path).expect("read original FSVI");
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let error = TwoTierIndex::open_with_paths(&paths, config).expect_err("alias must be rejected");

    assert!(matches!(error, SearchError::InvalidConfig { .. }));
    assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
}

#[cfg(all(feature = "ann", unix))]
#[test]
fn ann_symlink_alias_is_rejected_before_any_overwrite() {
    use std::os::unix::fs::symlink;

    let dir = temp_index_dir("ann-symlink-alias");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let ann_path = dir.join("index-fnv1a-4.hnsw");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
        .expect("write custom fast index");
    symlink(&fast_path, &ann_path).expect("create symlink alias");
    let original = fs::read(&fast_path).expect("read original FSVI");
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let error = TwoTierIndex::open_with_paths(&paths, config).expect_err("alias must be rejected");

    assert!(matches!(error, SearchError::InvalidConfig { .. }));
    assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
}

#[cfg(all(feature = "ann", unix))]
#[test]
fn ann_final_symlink_to_distinct_target_is_rejected_before_load_or_save() {
    use std::os::unix::fs::symlink;

    let dir = temp_index_dir("ann-distinct-final-symlink");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let ann_target = dir.join("distinct-target.hnsw");
    let ann_path = dir.join("index-fnv1a-4.hnsw");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
        .expect("write custom fast index");
    fs::write(&ann_target, b"preserve distinct ANN target").expect("write distinct ANN target");
    let original_target = fs::read(&ann_target).expect("read original ANN target");
    symlink(&ann_target, &ann_path).expect("create distinct-target ANN symlink");
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let error = TwoTierIndex::open_with_paths(&paths, config)
        .expect_err("a distinct-target ANN final symlink must be rejected");

    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
        ),
        "unexpected final-symlink error: {error:?}"
    );
    assert_eq!(
        fs::read(&ann_target).expect("read preserved ANN target"),
        original_target
    );
}

#[cfg(all(feature = "ann", unix))]
#[test]
fn dangling_ann_final_symlink_is_rejected_before_parent_or_lock_creation() {
    use std::os::unix::fs::symlink;

    let dir = temp_index_dir("ann-dangling-final-symlink");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("index-fnv1a-4.fsvi");
    let missing_target = dir.join("missing-target.hnsw");
    let ann_path = dir.join("index-fnv1a-4.hnsw");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
        .expect("write custom fast index");
    symlink(&missing_target, &ann_path).expect("create dangling ANN symlink");
    let lock_path =
        crate::hnsw::hnsw_save_lock_artifact_path(&ann_path).expect("derive save-lock path");
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };

    let error = TwoTierIndex::open_with_paths(&paths, config)
        .expect_err("a dangling ANN final symlink must be rejected");

    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
        ),
        "unexpected dangling-symlink error: {error:?}"
    );
    assert!(!missing_target.exists());
    assert!(!lock_path.exists());
    assert!(fs::symlink_metadata(&ann_path).is_ok());
}

#[cfg(all(feature = "ann", unix))]
#[test]
fn ann_symlinked_ancestor_with_parent_component_cannot_alias_fsvi() {
    use std::os::unix::fs::symlink;

    let dir = temp_index_dir("ann-symlink-ancestor-parent");
    let artifact_dir = dir.join("artifacts");
    let nested_dir = artifact_dir.join("nested");
    let indirection_dir = dir.join("indirection");
    fs::create_dir_all(&nested_dir).expect("create artifact directories");
    fs::create_dir_all(&indirection_dir).expect("create indirection directory");

    let fast_path = artifact_dir.join("index-fnv1a-4.fsvi");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
        .expect("write custom fast index");
    let original = fs::read(&fast_path).expect("read original FSVI");

    let ancestor_link = indirection_dir.join("link");
    symlink(&nested_dir, &ancestor_link).expect("create symlinked ancestor");
    let ann_path = ancestor_link.join("..").join("index-fnv1a-4.fsvi");
    assert_eq!(
        fs::canonicalize(&ann_path).expect("resolve OS path semantics"),
        fs::canonicalize(&fast_path).expect("resolve fast path")
    );

    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        ..TwoTierConfig::default()
    };
    let error = TwoTierIndex::open_with_paths(&paths, config).expect_err("alias must be rejected");

    assert!(matches!(error, SearchError::InvalidConfig { .. }));
    assert_eq!(fs::read(&fast_path).expect("read preserved FSVI"), original);
}

#[cfg(all(feature = "ann", unix))]
#[test]
fn missing_ann_leaves_cannot_alias_through_a_symlinked_ancestor_and_parent_component() {
    use std::os::unix::fs::symlink;

    let dir = temp_index_dir("ann-missing-symlink-ancestor-parent");
    let artifact_dir = dir.join("artifacts");
    let nested_dir = artifact_dir.join("nested");
    let indirection_dir = dir.join("indirection");
    fs::create_dir_all(&nested_dir).expect("create artifact directories");
    fs::create_dir_all(&indirection_dir).expect("create indirection directory");

    let fast_path = artifact_dir.join("fast.fsvi");
    let quality_path = artifact_dir.join("quality.fsvi");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");
    write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0])]).expect("write quality index");

    let ancestor_link = indirection_dir.join("link");
    symlink(&nested_dir, &ancestor_link).expect("create symlinked ancestor");
    let fast_ann_path = artifact_dir.join("shared.hnsw");
    let quality_ann_path = ancestor_link.join("..").join("shared.hnsw");
    assert!(!fast_ann_path.exists());
    assert!(!quality_ann_path.exists());

    let paths = TwoTierIndexPaths::new(&fast_path)
        .with_quality_index(&quality_path)
        .with_fast_ann(&fast_ann_path)
        .with_quality_ann(&quality_ann_path);
    let error = validate_index_paths(&paths)
        .expect_err("missing leaves with the same OS-resolved identity must be rejected");

    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "index_paths"
        ),
        "unexpected missing-leaf alias error: {error:?}"
    );
    assert!(!fast_ann_path.exists());
    assert!(!quality_ann_path.exists());
}
