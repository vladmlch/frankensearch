use super::*;

#[test]
fn opens_with_fallback_fast_index() {
    let dir = temp_index_dir("fallback");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fallback = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);

    write_index_file(
        &fallback,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ],
    )
    .expect("write fallback index");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
    assert_eq!(index.doc_count(), 2);
    assert!(!index.has_quality_index());
    let ids: Vec<String> = index
        .iter_doc_ids()
        .collect::<SearchResult<_>>()
        .expect("ids");
    assert_eq!(ids, vec!["doc-a".to_owned(), "doc-b".to_owned()]);
    assert_eq!(index.fast_index_for_doc_id("doc-a").unwrap(), Some(0));
    assert_eq!(index.fast_index_for_doc_id("doc-b").unwrap(), Some(1));
    assert_eq!(index.fast_index_for_doc_id("missing").unwrap(), None);

    let hits = index
        .search_fast(&[1.0, 0.0, 0.0, 0.0], 1)
        .expect("fast search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-a");
    #[cfg(feature = "ann")]
    assert_eq!(index.ann_fallback_count(), 0);
}

#[test]
fn search_fast_with_params_matches_default_path() {
    let dir = temp_index_dir("search-params-default-match");
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

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
    let baseline = index
        .search_fast(&[1.0, 0.0, 0.0, 0.0], 2)
        .expect("baseline");
    let overridden = index
        .search_fast_with_params(&[1.0, 0.0, 0.0, 0.0], 2, Some(SearchParams::default()))
        .expect("search with params");
    assert_eq!(baseline, overridden);
}

#[test]
fn search_fast_with_params_accepts_explicit_sequential_override() {
    let dir = temp_index_dir("search-params-seq-override");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
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

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
    let params = SearchParams {
        parallel_enabled: false,
        parallel_threshold: usize::MAX,
        parallel_chunk_size: 2,
    };
    let hits = index
        .search_fast_with_params(&[1.0, 0.0, 0.0, 0.0], 3, Some(params))
        .expect("sequential override search");
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].doc_id, "doc-a");
}

#[test]
fn quality_alignment_handles_partial_coverage() {
    let dir = temp_index_dir("alignment");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
        ],
    )
    .expect("write fast index");

    // Quality tier intentionally omits doc-b and uses different order.
    write_index_file(
        &quality_path,
        &[
            ("doc-c", &[0.0, 1.0, 0.0, 0.0]),
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ],
    )
    .expect("write quality index");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
    assert!(index.has_quality_index());
    assert!(index.has_quality_for_index(0));
    assert!(!index.has_quality_for_index(1));
    assert!(index.has_quality_for_index(2));

    let hits = vec![
        VectorHit {
            index: 0,
            score: 0.0,
            doc_id: "doc-a".into(),
        },
        VectorHit {
            index: 1,
            score: 0.0,
            doc_id: "doc-b".into(),
        },
        VectorHit {
            index: 2,
            score: 0.0,
            doc_id: "doc-c".into(),
        },
    ];
    let scores = index
        .quality_scores_for_hits(&[1.0, 0.0, 0.0, 0.0], &hits)
        .expect("quality scores");
    assert_eq!(scores.len(), 3);
    // doc-a has quality vector [1,0,0,0], query=[1,0,0,0] → dot=1.0
    assert!((scores[0].unwrap() - 1.0).abs() < 1e-6);
    // doc-b has NO quality vector → None
    assert!(scores[1].is_none());
    // doc-c has quality vector [0,1,0,0], query=[1,0,0,0] → dot=0.0
    assert!(scores[2].unwrap().abs() < 1e-6);
}

#[test]
fn quality_scores_are_none_without_quality_index() {
    let dir = temp_index_dir("no-quality");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
    )
    .expect("write fast index");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let hits = vec![
        VectorHit {
            index: 0,
            score: 0.0,
            doc_id: "doc-a".into(),
        },
        VectorHit {
            index: 1,
            score: 0.0,
            doc_id: "doc-b".into(),
        },
        VectorHit {
            index: 99,
            score: 0.0,
            doc_id: "doc-missing".into(),
        },
    ];
    let scores = index
        .quality_scores_for_hits(&[1.0, 0.0], &hits)
        .expect("scores");
    assert_eq!(scores, vec![None, None, None]);
}

/// A crash between the two per-tier installs leaves a mixed-generation
/// pair (new fast + old quality). The pair's publication nonces disagree,
/// and open must degrade to fast-only instead of silently blending tiers
/// from different publications (bd-miio8).
#[test]
fn open_degrades_to_fast_only_when_tier_publication_identities_disagree() {
    let dir = temp_index_dir("mixed-pair-degrade");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[1.0, 0.0]))
        .expect("add doc-a");
    drop(builder.finish().expect("finish pair"));

    // Simulate the crash window: only the fast tier of the NEXT
    // publication lands.
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let fast_gen =
        crate::next_generation(VectorIndex::peek_compaction_gen(&fast_path).expect("peek gen"));
    let fast_nonce = crate::next_publication_nonce(
        VectorIndex::peek_publication_nonce(&fast_path).expect("peek nonce"),
    );
    let staging = crate::temporary_output_path(&fast_path);
    let mut writer = VectorIndex::create(&staging, "fast-tier", 4)
        .expect("staging writer")
        .with_generation(fast_gen)
        .with_publication_nonce(fast_nonce);
    writer
        .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
        .expect("write doc-b");
    writer.finish().expect("finish staging");
    VectorIndex::install_replacement(&fast_path, &staging).expect("install fast tier only");

    let reopened = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open mixed pair");
    assert!(
        !reopened.has_quality_index(),
        "a mixed-publication pair must degrade to fast-only, not blend tiers"
    );
}

#[test]
fn open_degrades_to_fast_only_when_discovered_quality_index_is_corrupt() {
    let dir = temp_index_dir("corrupt-discovered-quality-degrade");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
    )
    .expect("write fast index");
    fs::write(dir.join(VECTOR_INDEX_QUALITY_FILENAME), b"not an FSVI file")
        .expect("write corrupt optional quality index");

    let opened = TwoTierIndex::open(&dir, TwoTierConfig::default())
        .expect("a corrupt discovered optional quality tier must not block fast-only open");

    assert_eq!(opened.doc_count(), 2);
    assert!(
        !opened.has_quality_index(),
        "a corrupt discovered quality tier must degrade to fast-only"
    );
    assert_eq!(
        opened
            .search_fast(&[1.0, 0.0], 1)
            .expect("fast search after quality degradation")[0]
            .doc_id,
        "doc-a"
    );
}

#[test]
fn open_with_paths_rejects_a_corrupt_explicit_quality_index() {
    let dir = temp_index_dir("corrupt-explicit-quality-reject");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("fast.fsvi");
    let quality_path = dir.join("quality.fsvi");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");
    fs::write(&quality_path, b"not an FSVI file").expect("write corrupt quality index");

    let error = TwoTierIndex::open_with_paths(
        &TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path),
        TwoTierConfig::default(),
    )
    .expect_err("an explicit corrupt quality path must remain an error");

    assert!(
        !matches!(error, SearchError::IndexNotFound { .. }),
        "the corrupt explicit path must not be reported as absent: {error:?}"
    );
}

#[test]
fn finish_stamps_both_tiers_with_one_nonzero_publication_nonce() {
    let dir = temp_index_dir("pair-nonce");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_record("doc-a", &[1.0, 0.0], Some(&[0.0, 1.0]))
        .expect("add doc-a");
    drop(builder.finish().expect("finish pair"));

    let fast_nonce = VectorIndex::peek_publication_nonce(&dir.join(VECTOR_INDEX_FAST_FILENAME))
        .expect("fast nonce");
    let quality_nonce =
        VectorIndex::peek_publication_nonce(&dir.join(VECTOR_INDEX_QUALITY_FILENAME))
            .expect("quality nonce");
    assert_ne!(fast_nonce, 0, "a published pair must carry a real identity");
    assert_eq!(
        fast_nonce, quality_nonce,
        "both tiers share one publication"
    );

    // Republishing advances the shared identity.
    let mut second = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    second
        .add_record("doc-b", &[0.0, 1.0], Some(&[1.0, 0.0]))
        .expect("add doc-b");
    drop(second.finish().expect("finish second pair"));
    let second_nonce = VectorIndex::peek_publication_nonce(&dir.join(VECTOR_INDEX_FAST_FILENAME))
        .expect("second fast nonce");
    assert_ne!(second_nonce, fast_nonce);
}

/// Rebuilding a live directory must SUPERSEDE the prior generation's WAL,
/// never adopt it: an acknowledged append from the old generation may not
/// resurrect inside the freshly built one (bd-zhjv8 finding 1).
#[test]
fn finish_over_live_dir_never_adopts_the_prior_generations_wal() {
    let dir = temp_index_dir("rebuild-wal-supersede");
    let mut first = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    first
        .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], None)
        .expect("add doc-a");
    drop(first.finish().expect("finish first generation"));

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let mut live = VectorIndex::open(&fast_path).expect("open first generation");
    live.append("ghost", &[0.0, 1.0, 0.0, 0.0])
        .expect("append acknowledged WAL record");
    drop(live);

    let mut second = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    second
        .add_record("doc-b", &[0.0, 0.0, 1.0, 0.0], None)
        .expect("add doc-b");
    drop(second.finish().expect("finish second generation"));

    let rebuilt = VectorIndex::open(&fast_path).expect("open second generation");
    let live_ids = rebuilt.live_doc_ids().expect("live ids");
    assert!(
        !live_ids.contains("ghost"),
        "prior generation's WAL record resurrected into the rebuilt index: {live_ids:?}"
    );
    assert!(live_ids.contains("doc-b"));
    assert_eq!(
        rebuilt.wal_record_count(),
        0,
        "rebuilt generation must start with no adopted WAL"
    );
}

#[test]
fn builder_round_trips_fast_and_quality_records() {
    let dir = temp_index_dir("builder");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .set_fast_embedder_id("fast-test")
        .set_quality_embedder_id("quality-test");
    builder
        .add_record("doc-a", &[1.0, 0.0, 0.0], Some(&[1.0, 0.0, 0.0]))
        .expect("add doc-a");
    builder
        .add_record("doc-b", &[0.0, 1.0, 0.0], None)
        .expect("add doc-b");

    let index = builder.finish().expect("finish builder");
    assert_eq!(index.doc_count(), 2);
    assert!(index.has_quality_index());
    assert!(index.has_quality_for_index(0));
    assert!(!index.has_quality_for_index(1));
}

#[test]
fn builder_rejects_inconsistent_fast_dimension() {
    let dir = temp_index_dir("bad-dim");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
        .expect("first record");

    let err = builder
        .add_fast_record("doc-b", &[1.0, 0.0])
        .expect_err("must reject dimension mismatch");
    assert!(matches!(
        err,
        SearchError::DimensionMismatch {
            expected: 3,
            found: 2
        }
    ));
}

#[test]
fn quality_scores_dimension_mismatch() {
    let dir = temp_index_dir("quality-dim");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write fast index");
    write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
        .expect("write quality index");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert!(index.has_quality_index());

    // Query dimension (4) doesn't match quality dimension (6)
    let hits = vec![VectorHit {
        index: 0,
        score: 0.0,
        doc_id: "doc-a".into(),
    }];
    let error = index
        .quality_scores_for_hits(&[1.0, 0.0, 0.0, 0.0], &hits)
        .unwrap_err();
    assert!(
        matches!(
            error,
            SearchError::DimensionMismatch {
                expected: 6,
                found: 4
            }
        ),
        "expected DimensionMismatch, got {error:?}"
    );
}

#[test]
fn has_quality_for_index_out_of_bounds_returns_false() {
    let dir = temp_index_dir("oob");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert!(!index.has_quality_for_index(999));
}

// ── Builder error paths ─────────────────────────────────────────

#[test]
fn builder_finish_rejects_empty_fast_records() {
    let dir = temp_index_dir("empty-builder");
    let builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    let error = builder.finish().unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "fast_records"),
        "expected InvalidConfig for fast_records, got {error:?}"
    );
}

#[test]
fn builder_rejects_inconsistent_quality_dimension() {
    let dir = temp_index_dir("bad-quality-dim");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
        .expect("fast record");
    builder
        .add_quality_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
        .expect("first quality");
    let error = builder
        .add_quality_record("doc-b", &[1.0, 0.0])
        .unwrap_err();
    assert!(
        matches!(
            error,
            SearchError::DimensionMismatch {
                expected: 4,
                found: 2
            }
        ),
        "expected DimensionMismatch, got {error:?}"
    );
}

// ── Fast-tier with explicit fast.idx (not fallback) ─────────────

#[test]
fn opens_with_explicit_fast_index() {
    let dir = temp_index_dir("explicit-fast");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-x", &[0.0, 1.0, 0.0]), ("doc-y", &[0.0, 0.0, 1.0])],
    )
    .expect("write fast index");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert_eq!(index.doc_count(), 2);
    let ids: Vec<String> = index
        .iter_doc_ids()
        .collect::<SearchResult<_>>()
        .expect("ids");
    assert_eq!(ids, vec!["doc-x".to_owned(), "doc-y".to_owned()]);

    let hits = index.search_fast(&[0.0, 0.0, 1.0], 1).expect("fast search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "doc-y");
}

// ── Accessors ───────────────────────────────────────────────────

#[test]
fn config_accessor_returns_construction_config() {
    let dir = temp_index_dir("config-acc");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");

    let config = TwoTierConfig {
        hnsw_threshold: 42,
        ..TwoTierConfig::default()
    };
    let index = TwoTierIndex::open(&dir, config).expect("open");
    assert_eq!(index.config().hnsw_threshold, 42);
}

// ── Quality alignment: unmatched doc_ids ────────────────────────

#[test]
fn quality_index_with_extra_doc_ids_still_opens() {
    let dir = temp_index_dir("quality-extra");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");
    // Quality has a doc_id not in fast — should trigger warning but still open
    write_index_file(
        &quality_path,
        &[
            ("doc-a", &[1.0, 0.0]),
            ("doc-z", &[0.0, 1.0]), // extra
        ],
    )
    .expect("write quality");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert!(index.has_quality_index());
    assert!(index.has_quality_for_index(0)); // doc-a matched
    assert_eq!(index.doc_count(), 1); // only fast-tier docs counted
}

// ── Builder: fast+quality via add_record convenience ────────────

#[test]
fn builder_add_record_with_none_quality_skips_quality_tier() {
    let dir = temp_index_dir("add-record-no-quality");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_record("doc-a", &[1.0, 0.0], None)
        .expect("add doc-a");
    builder
        .add_record("doc-b", &[0.0, 1.0], None)
        .expect("add doc-b");

    let index = builder.finish().expect("finish");
    assert_eq!(index.doc_count(), 2);
    assert!(!index.has_quality_index());
}

// ─── bd-3nsq: NaN score in WAL merge ───

#[test]
fn search_fast_skips_nan_score_wal_entries() {
    use crate::wal::WalEntry;

    let dir = temp_index_dir("nan-wal-score");
    fs::create_dir_all(&dir).expect("create temp dir");

    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write fast index");

    let mut index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");

    // Inject a WAL entry with NaN in the embedding. dot_product with NaN
    // produces NaN, which should be filtered out by the is_finite() guard.
    index
        .fast_tier_mut_for_test()
        .expect("path-opened tier")
        .wal_entries
        .push(WalEntry {
            doc_id: "doc-nan".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-nan"),
            embedding: vec![f32::NAN, 0.0, 0.0, 0.0],
        });

    // Also inject a valid WAL entry to confirm it's still returned.
    index
        .fast_tier_mut_for_test()
        .expect("path-opened tier")
        .wal_entries
        .push(WalEntry {
            doc_id: "doc-wal-ok".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-wal-ok"),
            embedding: vec![0.0, 1.0, 0.0, 0.0],
        });

    let hits = index
        .search_fast(&[1.0, 0.0, 0.0, 0.0], 10)
        .expect("search");
    let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();

    assert!(
        !ids.contains(&"doc-nan"),
        "NaN-scored WAL entry must be excluded, got: {ids:?}"
    );
    assert!(
        ids.contains(&"doc-a"),
        "base doc-a should be returned, got: {ids:?}"
    );
    assert!(
        ids.contains(&"doc-wal-ok"),
        "valid WAL entry should be returned, got: {ids:?}"
    );

    // Verify all returned scores are finite.
    for hit in &hits {
        assert!(
            hit.score.is_finite(),
            "hit {} has non-finite score {}",
            hit.doc_id,
            hit.score
        );
    }
}

// ─── bd-3szp tests begin ───

#[test]
fn filename_constants_are_correct() {
    assert_eq!(VECTOR_INDEX_FAST_FILENAME, "vector.fast.idx");
    assert_eq!(VECTOR_INDEX_QUALITY_FILENAME, "vector.quality.idx");
    assert_eq!(VECTOR_INDEX_FALLBACK_FILENAME, "vector.idx");
}

#[test]
fn two_tier_index_implements_debug() {
    let dir = temp_index_dir("debug-index");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let debug_str = format!("{index:?}");
    assert!(debug_str.contains("TwoTierIndex"));
}

#[test]
fn two_tier_index_builder_implements_debug() {
    let dir = temp_index_dir("debug-builder");
    let builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    let debug_str = format!("{builder:?}");
    assert!(debug_str.contains("TwoTierIndexBuilder"));
}

#[test]
fn fast_index_preferred_over_fallback_when_both_exist() {
    let dir = temp_index_dir("prefer-fast");
    fs::create_dir_all(&dir).expect("create temp dir");

    // Write fallback with doc-fallback
    let fallback_path = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
    write_index_file(&fallback_path, &[("doc-fallback", &[0.0, 1.0])]).expect("write fallback");

    // Write explicit fast with doc-fast
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-fast", &[1.0, 0.0])]).expect("write fast");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert_eq!(index.doc_count(), 1);
    let ids: Vec<String> = index
        .iter_doc_ids()
        .collect::<SearchResult<_>>()
        .expect("ids");
    assert_eq!(ids, vec!["doc-fast".to_owned()]);
}

#[test]
fn search_fast_returns_sorted_by_score_descending() {
    let dir = temp_index_dir("sort-order");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[
            ("doc-low", &[0.1, 0.0, 0.0]),
            ("doc-mid", &[0.5, 0.5, 0.0]),
            ("doc-high", &[1.0, 0.0, 0.0]),
        ],
    )
    .expect("write fast");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let hits = index.search_fast(&[1.0, 0.0, 0.0], 3).expect("search");
    assert_eq!(hits.len(), 3);
    // Scores should be in descending order
    assert!(hits[0].score >= hits[1].score);
    assert!(hits[1].score >= hits[2].score);
    assert_eq!(hits[0].doc_id, "doc-high");
}

#[test]
fn search_fast_k_zero_returns_empty() {
    let dir = temp_index_dir("k-zero");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let hits = index.search_fast(&[1.0, 0.0], 0).expect("search k=0");
    assert!(hits.is_empty());
}

#[test]
fn search_fast_k_larger_than_doc_count_returns_all() {
    let dir = temp_index_dir("k-large");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
    )
    .expect("write fast");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let hits = index.search_fast(&[1.0, 0.0], 100).expect("search k=100");
    assert_eq!(hits.len(), 2);
}

#[test]
fn quality_scores_empty_indices_returns_empty() {
    let dir = temp_index_dir("empty-indices");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");
    write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0])]).expect("write quality");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    let scores = index
        .quality_scores_for_hits(&[1.0, 0.0], &[])
        .expect("empty indices");
    assert!(scores.is_empty());
}

#[test]
fn quality_scores_full_coverage() {
    let dir = temp_index_dir("full-quality");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

    write_index_file(
        &fast_path,
        &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
    )
    .expect("write fast");

    write_index_file(
        &quality_path,
        &[("doc-a", &[0.0, 0.0, 1.0]), ("doc-b", &[0.0, 1.0, 0.0])],
    )
    .expect("write quality");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert!(index.has_quality_for_index(0));
    assert!(index.has_quality_for_index(1));

    let hits = vec![
        VectorHit {
            index: 0,
            score: 0.0,
            doc_id: "doc-a".into(),
        },
        VectorHit {
            index: 1,
            score: 0.0,
            doc_id: "doc-b".into(),
        },
    ];
    let scores = index
        .quality_scores_for_hits(&[0.0, 1.0, 0.0], &hits)
        .expect("quality scores");
    assert_eq!(scores.len(), 2);
    // doc-a quality = [0,0,1] dot [0,1,0] = 0.0
    assert!(scores[0].unwrap().abs() < 1e-6);
    // doc-b quality = [0,1,0] dot [0,1,0] = 1.0
    assert!((scores[1].unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn builder_embedder_id_chaining() {
    let dir = temp_index_dir("embedder-chain");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");

    // Verify chaining returns &mut Self
    let _same_ref = builder
        .set_fast_embedder_id("custom-fast")
        .set_quality_embedder_id("custom-quality");

    builder
        .add_record("doc-a", &[1.0, 0.0], Some(&[0.0, 1.0]))
        .expect("add record");
    let index = builder.finish().expect("finish");
    assert_eq!(index.doc_count(), 1);
}

// ─── Embedding-space identity retention (bd-9xuj T2-C2) ─────────────────

#[test]
fn builder_finish_retains_declared_typed_identity() {
    let dir = temp_index_dir("typed-identity-retention");
    let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("builder-fast-model", 4);
    let quality_bundle = EmbeddingIdentityBundleV1::explicit_test_model("builder-quality-model", 4);

    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder.set_fast_embedder_id("op-fast-id");
    builder.set_quality_embedder_id("op-quality-id");
    builder
        .set_fast_identity(&fast_bundle)
        .expect("declare fast identity");
    builder
        .set_quality_identity(&quality_bundle)
        .expect("declare quality identity");
    builder
        .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0, 0.0]))
        .expect("add record");
    let index = builder.finish().expect("finish");

    // The finished index carries the REAL typed identity, per tier.
    assert_eq!(
        index.fast_space_fingerprint_hex(),
        Some(fast_bundle.space.fingerprint().as_str())
    );
    assert_eq!(
        index.quality_space_fingerprint_hex(),
        Some(quality_bundle.space.fingerprint().as_str())
    );
    assert_ne!(
        index.fast_space_fingerprint_hex(),
        index.quality_space_fingerprint_hex(),
        "distinct producing models must stay distinct per tier"
    );
    assert_eq!(
        index
            .fast_declared_identity()
            .map(EmbeddingIdentityBundleV1::fingerprint),
        Some(fast_bundle.fingerprint()),
        "the full producing bundle survives finish for the admission law"
    );
    assert_eq!(
        index
            .quality_declared_identity()
            .map(EmbeddingIdentityBundleV1::fingerprint),
        Some(quality_bundle.fingerprint())
    );

    // The operational id string is untouched by the typed declaration;
    // the v1 header revision carries the complete producer bundle fingerprint.
    assert_eq!(index.fast_embedder_id(), "op-fast-id");
    assert_eq!(index.quality_embedder_id(), Some("op-quality-id"));
    assert_eq!(index.fast_embedder_revision(), fast_bundle.fingerprint());
    assert_eq!(
        index.quality_embedder_revision(),
        Some(quality_bundle.fingerprint().as_str())
    );
    drop(index);

    // Reopen from disk: the v1 artifacts persist id/revision strings but
    // NO space identity — absence stays typed (LegacyUnidentified), and
    // is never re-fabricated from the surviving strings.
    let reopened = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("reopen");
    assert_eq!(reopened.fast_embedder_id(), "op-fast-id");
    assert_eq!(
        reopened.fast_embedder_revision(),
        fast_bundle.fingerprint(),
        "the declared revision must survive persistence in the v1 header"
    );
    assert_eq!(reopened.fast_space_fingerprint_hex(), None);
    assert_eq!(reopened.quality_space_fingerprint_hex(), None);
    assert!(reopened.fast_declared_identity().is_none());
    assert!(reopened.quality_declared_identity().is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn builder_finish_rejects_identity_not_describing_vectors() -> Result<(), String> {
    // A 16-dim space cannot describe 4-dim written vectors: the claim is
    // checked against what was actually written, never trusted alone.
    let dir = temp_index_dir("typed-identity-dim-mismatch");
    let wrong_bundle = EmbeddingIdentityBundleV1::explicit_test_model("wrong-dim-model", 16);

    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .set_fast_identity(&wrong_bundle)
        .expect("bundle itself is coherent; the mismatch is against the records");
    builder
        .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], None)
        .expect("add record");
    let error = builder
        .finish()
        .expect_err("a non-describing identity must be rejected at finish");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig { field, value, .. } = error else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "fast_identity.space.dimension");
    assert_eq!(value, "16");

    // Quality arm: same law, its own field name.
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .set_quality_identity(&wrong_bundle)
        .expect("bundle itself is coherent");
    builder
        .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0, 0.0]))
        .expect("add record");
    let error = builder
        .finish()
        .expect_err("a non-describing quality identity must be rejected at finish");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig { field, value, .. } = error else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "quality_identity.space.dimension");
    assert_eq!(value, "16");

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

#[test]
fn builder_quality_identity_without_quality_tier_is_dropped() {
    // A declared quality identity with no quality records written must
    // NOT be attached: this build produced no quality tier, and the
    // declaration must never end up describing a stale or absent one.
    let dir = temp_index_dir("typed-identity-no-quality");
    let quality_bundle = EmbeddingIdentityBundleV1::explicit_test_model("unused-quality", 4);

    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .set_quality_identity(&quality_bundle)
        .expect("declare quality identity");
    builder
        .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], None)
        .expect("add fast-only record");
    let index = builder.finish().expect("finish fast-only build");
    assert!(!index.has_quality_index());
    assert_eq!(index.quality_space_fingerprint_hex(), None);
    assert!(index.quality_declared_identity().is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn builder_without_identity_stays_legacy_unidentified() {
    // The pre-C2 builder path, byte-compatible: no declaration, empty
    // header revision, and the typed-absent identity state throughout.
    let dir = temp_index_dir("typed-identity-legacy");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_record("doc-a", &[1.0, 0.0], Some(&[0.0, 1.0]))
        .expect("add record");
    let index = builder.finish().expect("finish");
    assert_eq!(index.fast_space_fingerprint_hex(), None);
    assert_eq!(index.quality_space_fingerprint_hex(), None);
    assert!(index.fast_declared_identity().is_none());
    assert!(index.quality_declared_identity().is_none());
    assert_eq!(index.fast_embedder_revision(), "");
    assert_eq!(index.quality_embedder_revision(), Some(""));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn semantic_vector_with_tier_reports_serving_tier() {
    // doc-a exists in both tiers; doc-b is fast-only. The provenance twin
    // must (1) return exactly what semantic_vector_for_doc_id returns,
    // (2) name the tier that served it, and (3) join to that tier's
    // space through space_fingerprint_hex_for_tier.
    let dir = temp_index_dir("semantic-vector-tier");
    let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("svt-fast-model", 4);
    let quality_bundle = EmbeddingIdentityBundleV1::explicit_test_model("svt-quality-model", 4);

    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .set_fast_identity(&fast_bundle)
        .expect("declare fast identity");
    builder
        .set_quality_identity(&quality_bundle)
        .expect("declare quality identity");
    builder
        .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0, 0.0]))
        .expect("add doc-a");
    builder
        .add_record("doc-b", &[0.0, 0.0, 1.0, 0.0], None)
        .expect("add doc-b");
    let index = builder.finish().expect("finish");

    // Quality-covered doc: served from the quality tier.
    let (tier, vector) = index
        .semantic_vector_with_tier_for_doc_id("doc-a")
        .expect("lookup doc-a")
        .expect("doc-a exists");
    assert_eq!(tier, SemanticVectorTier::Quality);
    assert_eq!(
        Some(vector.clone()),
        index
            .semantic_vector_for_doc_id("doc-a")
            .expect("legacy lookup doc-a"),
        "the provenance twin must return exactly the legacy accessor's vector"
    );
    assert_eq!(
        index.space_fingerprint_hex_for_tier(tier),
        Some(quality_bundle.space.fingerprint().as_str()),
        "Quality provenance joins to the quality tier's space"
    );

    // Quality-missing doc: the silent fast fallback, now observable.
    let (tier, vector) = index
        .semantic_vector_with_tier_for_doc_id("doc-b")
        .expect("lookup doc-b")
        .expect("doc-b exists");
    assert_eq!(tier, SemanticVectorTier::Fast);
    assert_eq!(
        Some(vector),
        index
            .semantic_vector_for_doc_id("doc-b")
            .expect("legacy lookup doc-b")
    );
    assert_eq!(
        index.space_fingerprint_hex_for_tier(tier),
        Some(fast_bundle.space.fingerprint().as_str()),
        "Fast provenance joins to the fast tier's space"
    );

    // Missing doc: both accessors agree on absence.
    assert!(
        index
            .semantic_vector_with_tier_for_doc_id("doc-missing")
            .expect("lookup missing")
            .is_none()
    );
    assert!(
        index
            .semantic_vector_for_doc_id("doc-missing")
            .expect("legacy lookup missing")
            .is_none()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn builder_fast_only_no_quality_index_created() {
    let dir = temp_index_dir("fast-only-builder");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
        .expect("fast a");
    builder
        .add_fast_record("doc-b", &[0.0, 1.0, 0.0])
        .expect("fast b");

    let index = builder.finish().expect("finish");
    assert_eq!(index.doc_count(), 2);
    assert!(!index.has_quality_index());
    assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());
}

#[test]
fn builder_add_record_with_quality_creates_both_tiers() {
    let dir = temp_index_dir("both-tiers-builder");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_record("doc-a", &[1.0, 0.0], Some(&[0.5, 0.5, 0.0]))
        .expect("add doc-a");
    builder
        .add_record("doc-b", &[0.0, 1.0], Some(&[0.0, 0.5, 0.5]))
        .expect("add doc-b");

    let index = builder.finish().expect("finish");
    assert_eq!(index.doc_count(), 2);
    assert!(index.has_quality_index());
    assert!(index.has_quality_for_index(0));
    assert!(index.has_quality_for_index(1));
}

#[test]
fn builder_preserves_all_doc_ids() {
    let dir = temp_index_dir("all-docids");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    let names = ["zebra", "apple", "mango", "banana"];
    for name in &names {
        builder
            .add_fast_record(*name, &[1.0, 0.0])
            .expect("add record");
    }
    let index = builder.finish().expect("finish");
    assert_eq!(index.doc_count(), 4);
    let ids: Vec<String> = index
        .iter_doc_ids()
        .collect::<SearchResult<_>>()
        .expect("ids");
    let mut actual: Vec<&str> = ids.iter().map(String::as_str).collect();
    actual.sort_unstable();
    let mut expected = names.to_vec();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

#[test]
fn fast_index_for_doc_id_empty_string_returns_none() {
    let dir = temp_index_dir("empty-docid-lookup");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert_eq!(index.fast_index_for_doc_id("").unwrap(), None);
}

#[test]
fn has_quality_for_index_boundary_last_valid() {
    let dir = temp_index_dir("boundary-last");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[1.0, 0.0]),
            ("doc-b", &[0.0, 1.0]),
            ("doc-c", &[0.5, 0.5]),
        ],
    )
    .expect("write fast");

    // Quality only has last doc
    write_index_file(&quality_path, &[("doc-c", &[0.5, 0.5])]).expect("write quality");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert!(!index.has_quality_for_index(0));
    assert!(!index.has_quality_for_index(1));
    assert!(index.has_quality_for_index(2)); // last valid index
    assert!(!index.has_quality_for_index(3)); // out of bounds
}

#[test]
fn quality_scores_no_quality_index_ignores_query_dimension() {
    // When there's no quality index, any query dimension is accepted (returns all 0.0)
    let dir = temp_index_dir("no-quality-any-dim");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
    assert!(!index.has_quality_index());

    // Use a completely different dimension query — should still return 0s
    let hits = vec![VectorHit {
        index: 0,
        score: 0.0,
        doc_id: "doc-a".into(),
    }];
    let scores = index
        .quality_scores_for_hits(&[1.0, 2.0, 3.0, 4.0, 5.0], &hits)
        .expect("any dim accepted");
    assert_eq!(scores, vec![None]);
}

#[test]
fn search_fast_returns_correct_doc_ids() {
    let dir = temp_index_dir("correct-docids");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[
            ("alpha", &[1.0, 0.0, 0.0]),
            ("beta", &[0.0, 1.0, 0.0]),
            ("gamma", &[0.0, 0.0, 1.0]),
        ],
    )
    .expect("write fast");

    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");

    // Query aligned with beta
    let hits = index
        .search_fast(&[0.0, 1.0, 0.0], 1)
        .expect("search for beta");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "beta");

    // Query aligned with gamma
    let hits = index
        .search_fast(&[0.0, 0.0, 1.0], 1)
        .expect("search for gamma");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "gamma");
}

#[test]
fn builder_add_record_dimension_mismatch_in_quality() {
    let dir = temp_index_dir("record-quality-dim");
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_record("doc-a", &[1.0, 0.0], Some(&[1.0, 0.0, 0.0]))
        .expect("first record ok");

    // Second record has different quality dimension
    let err = builder
        .add_record("doc-b", &[0.0, 1.0], Some(&[1.0, 0.0]))
        .expect_err("quality dim mismatch");
    assert!(matches!(
        err,
        SearchError::DimensionMismatch {
            expected: 3,
            found: 2
        }
    ));
}

#[test]
fn open_nonexistent_directory_returns_error() {
    let dir = temp_index_dir("nonexistent-subdir");
    // Don't create the directory
    let result = TwoTierIndex::open(&dir, TwoTierConfig::default());
    assert!(result.is_err());
}

// ─── bd-3szp tests end ───
