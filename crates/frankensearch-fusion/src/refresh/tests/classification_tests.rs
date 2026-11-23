use super::*;

// ─── C4-write r2: read-only pre-drain classification + retained owners ──

/// Sorted (name, byte length, mtime) manifest of a directory's entries —
/// the invariance witness for "classification touched nothing here".
/// atime is deliberately NOT asserted (relatime makes it flaky and the
/// no-atime open is best-effort by platform).
fn dir_manifest(dir: &Path) -> Vec<(String, u64, std::time::SystemTime)> {
    let mut entries: Vec<(String, u64, std::time::SystemTime)> = std::fs::read_dir(dir)
        .expect("read dir")
        .map(|entry| {
            let entry = entry.expect("dir entry");
            let metadata = entry.metadata().expect("entry metadata");
            (
                entry.file_name().to_string_lossy().into_owned(),
                metadata.len(),
                metadata.modified().expect("entry mtime"),
            )
        })
        .collect();
    entries.sort();
    entries
}

/// Plant a WAL sidecar beside `target_main` whose compaction generation
/// cannot match the target's main slab. Built generation-RELATIVE (the
/// authority-model `publish_tier` bumps the canonical generation on
/// every publish, so absolute values are not stable): the donor main
/// carries `next(target)`, so its WAL carries `next(next(target))`,
/// which can never equal the `next(target)` the target expects.
///
/// A control copy proves the staleness precondition the hazardous way:
/// mutable `VectorIndex::open` on a copy of the pair DELETES the WAL.
fn transplant_stale_wal(label: &str, target_main: &Path, dimension: usize) {
    use frankensearch_index::wal::wal_path_for;

    let target_gen = VectorIndex::peek_compaction_gen(target_main).expect("peek target generation");
    let donor_dir = temp_index_dir(&format!("{label}-wal-donor"));
    let donor_main = donor_dir.join("donor.idx");
    let mut writer = VectorIndex::create(&donor_main, "stub-fast", dimension)
        .expect("create donor")
        .with_generation(frankensearch_index::next_generation(target_gen));
    writer
        .write_record("donor-seed", &normalized(dimension, 0.91))
        .expect("write donor seed");
    writer.finish().expect("finish donor");
    {
        let mut donor = VectorIndex::open(&donor_main).expect("open donor");
        donor
            .append("donor-wal-resident", &normalized(dimension, 0.71))
            .expect("append the donor WAL resident");
    }
    let donor_wal = wal_path_for(&donor_main);
    assert!(donor_wal.exists(), "donor WAL must exist");
    std::fs::copy(&donor_wal, wal_path_for(target_main)).expect("transplant WAL");

    // Control arm: prove the transplanted WAL is stale for a
    // generation-0 main slab by demonstrating the exact hazard on a
    // throwaway copy — the mutable v1 open deletes it.
    let control_dir = temp_index_dir(&format!("{label}-wal-control"));
    let control_main = control_dir.join("control.idx");
    std::fs::copy(target_main, &control_main).expect("copy control main");
    std::fs::copy(&donor_wal, wal_path_for(&control_main)).expect("copy control wal");
    let _ = VectorIndex::open(&control_main).expect("control open");
    assert!(
        !wal_path_for(&control_main).exists(),
        "control precondition: the transplanted WAL must be STALE for the target \
         (mutable VectorIndex::open deletes it)"
    );
}

/// Required test (i), stale-WAL v1 case — RED on 868c0801: the r1
/// pre-drain classification routed v1 tiers through the mutable
/// `VectorIndex::open` (refresh.rs:244-250 there), which DELETES a stale
/// WAL sidecar during what claims to be classification. r2 classifies
/// read-only: refusal fires with the canonical directory byte-identical.
#[test]
fn classification_never_deletes_a_stale_wal_v1() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("r2-stale-wal-invariance");
        let queue = make_queue(100);
        let (worker, _cache, embedder) = make_worker(queue.clone(), &dir, 256);

        submit(&queue, "doc-1", "First document");
        worker.run_cycle(&cx).await.expect("bootstrap cycle");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        transplant_stale_wal("r2-stale", &fast_path, 256);
        let wal_path = frankensearch_index::wal::wal_path_for(&fast_path);
        let fast_bytes = std::fs::read(&fast_path).expect("read fast");
        let wal_bytes = std::fs::read(&wal_path).expect("read wal");
        let manifest_before = dir_manifest(&dir);

        submit(&queue, "doc-2", "Second document");
        let embed_baseline = embedder.embed_invocations();
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("content-retaining v1 must refuse identityless");
        assert_invalid_config(
            &error,
            "refresh.fast_index_identity",
            "identityless-fsvi-v1",
        );
        assert_no_embed_invocations_since(&embedder, embed_baseline);

        assert!(
            wal_path.exists(),
            "read-only classification must NEVER delete a stale WAL sidecar"
        );
        assert_eq!(
            std::fs::read(&wal_path).expect("reread wal"),
            wal_bytes,
            "WAL bytes must be untouched by classification"
        );
        assert_eq!(
            std::fs::read(&fast_path).expect("reread fast"),
            fast_bytes,
            "main artifact bytes must be untouched by classification"
        );
        assert_eq!(
            dir_manifest(&dir),
            manifest_before,
            "no file in the canonical directory may change (names, sizes, mtimes)"
        );
        assert_eq!(queue.pending_count(), 1, "refusal fires before drain");
    });
}

/// Required test (i), corrupt-trailer v1 case — RED on 868c0801: the
/// mutable open TRUNCATES a corrupt WAL trailer during classification.
/// r2 leaves the trailer bytes exactly in place.
#[test]
fn classification_never_truncates_a_corrupt_wal_trailer_v1() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("r2-corrupt-trailer-invariance");
        let queue = make_queue(100);
        let (worker, cache, embedder) = make_worker(queue.clone(), &dir, 256);

        submit(&queue, "doc-1", "First document");
        worker.run_cycle(&cx).await.expect("bootstrap cycle");

        drop(worker);
        drop(cache);
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        {
            let mut index = VectorIndex::open(&fast_path).expect("fixture open");
            index
                .append("doc-wal", &normalized(256, 0.55))
                .expect("append fresh WAL resident");
        }
        let wal_path = frankensearch_index::wal::wal_path_for(&fast_path);
        {
            use std::io::Write as _;
            let mut wal_file = std::fs::OpenOptions::new()
                .append(true)
                .open(&wal_path)
                .expect("open wal for corruption");
            wal_file
                .write_all(&[0xAB; 32])
                .expect("append corrupt trailer");
        }
        let wal_bytes = std::fs::read(&wal_path).expect("read wal");
        let manifest_before = dir_manifest(&dir);
        let cache = Arc::new(
            IndexCache::open(
                &dir,
                TwoTierConfig::default(),
                Box::new(SentinelFileDetector::new()),
            )
            .expect("open corrupt-WAL generation"),
        );
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10)),
            queue.clone(),
            embedder.clone(),
            cache,
        );

        // Control arm: the mutable open truncates this trailer.
        {
            let control_dir = temp_index_dir("r2-corrupt-trailer-control");
            let control_main = control_dir.join("control.idx");
            std::fs::copy(&fast_path, &control_main).expect("copy control main");
            let control_wal = frankensearch_index::wal::wal_path_for(&control_main);
            std::fs::copy(&wal_path, &control_wal).expect("copy control wal");
            let _ = VectorIndex::open(&control_main).expect("control open");
            assert!(
                std::fs::metadata(&control_wal)
                    .expect("stat control wal")
                    .len()
                    < wal_bytes.len() as u64,
                "control precondition: mutable open truncates the corrupt trailer"
            );
        }

        submit(&queue, "doc-2", "Second document");
        let embed_baseline = embedder.embed_invocations();
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("content-retaining v1 must refuse identityless");
        assert_invalid_config(
            &error,
            "refresh.fast_index_identity",
            "identityless-fsvi-v1",
        );
        assert_no_embed_invocations_since(&embedder, embed_baseline);

        assert_eq!(
            std::fs::read(&wal_path).expect("reread wal"),
            wal_bytes,
            "read-only classification must NEVER truncate a corrupt WAL trailer"
        );
        assert_eq!(
            dir_manifest(&dir),
            manifest_before,
            "no file in the canonical directory may change"
        );
        assert_eq!(queue.pending_count(), 1, "refusal fires before drain");
    });
}

/// Required test (ii) — RED on 868c0801: a mixed v2-fast + v1-quality
/// generation must classify (and refuse, on the quality tier) without
/// mutating EITHER tier. Under r1, classifying the v1 quality tier ran
/// the mutable open, which deleted its stale WAL before the refusal.
#[test]
fn mixed_v2_fast_v1_quality_classifies_without_mutating_either() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
        let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
            "r2-mixed-generation",
            embedder.identity_bundle(),
            &[("doc-old", normalized(V2_DIM, 0.5))],
            4,
        );

        // Content-retaining v1 quality tier with a stale WAL beside it.
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let mut writer =
            VectorIndex::create(&quality_path, "stub-quality", 256).expect("create quality");
        writer
            .write_record("q-doc", &normalized(256, 0.4))
            .expect("write quality row");
        writer.finish().expect("finish quality");
        transplant_stale_wal("r2-mixed", &quality_path, 256);
        let quality_wal = frankensearch_index::wal::wal_path_for(&quality_path);

        let fast_bytes = std::fs::read(&fast_path).expect("read v2 fast");
        let quality_bytes = std::fs::read(&quality_path).expect("read v1 quality");
        let wal_bytes = std::fs::read(&quality_wal).expect("read quality wal");
        let manifest_before = dir_manifest(&dir);

        let queue = make_queue(100);
        submit(&queue, "doc-new", "new document");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder.clone(),
            cache,
        );
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("mixed generation with content-retaining v1 quality must refuse");
        assert_invalid_config(
            &error,
            "refresh.quality_index_identity",
            "identityless-fsvi-v1",
        );
        assert_no_embed_invocations_since(&embedder, 0);

        assert_eq!(
            std::fs::read(&fast_path).expect("reread v2 fast"),
            fast_bytes,
            "the v2 fast tier must be untouched"
        );
        assert_eq!(
            std::fs::read(&quality_path).expect("reread v1 quality"),
            quality_bytes,
            "the v1 quality tier must be untouched"
        );
        assert!(
            quality_wal.exists(),
            "classification must not delete the v1 quality tier's stale WAL"
        );
        assert_eq!(
            std::fs::read(&quality_wal).expect("reread quality wal"),
            wal_bytes,
            "the quality WAL must be byte-identical"
        );
        assert_eq!(dir_manifest(&dir), manifest_before);
        assert_eq!(queue.pending_count(), 1, "refusal fires before drain");
        let drained = queue.drain_batch();
        assert_eq!(drained[0].retry_count, 0, "no retry budget consumed");
    });
}

/// Required test (iii) — RED on 868c0801: a header-valid but
/// content-corrupt v2 artifact must fail ADMISSION with its own typed
/// corruption error, not sail past a header-only check to the
/// composite-authority refusal (which would falsely certify the
/// generation as fully admitted). Option-A choice: pre-drain performs
/// full `open_admitted_v2` admission, so the digest recomputation
/// catches the corruption.
#[test]
fn header_valid_content_corrupt_v2_refuses_admission_not_composite() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
        let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
            "r2-content-corrupt-v2",
            embedder.identity_bundle(),
            &[("doc-old", normalized(V2_DIM, 0.5))],
            6,
        );

        // Flip one byte in the vector slab (the file tail): the header —
        // including its CRC — stays valid, so header-only inspection
        // still reports identity-complete v2.
        let mut bytes = std::fs::read(&fast_path).expect("read canonical");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&fast_path, &bytes).expect("plant content corruption");

        let queue = make_queue(100);
        submit(&queue, "doc-new", "new document");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder.clone(),
            cache,
        );
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("content-corrupt v2 must fail admission");

        assert!(
            matches!(error, SearchError::IndexCorrupted { .. }),
            "the refusal must be the admission's own typed corruption error, got {error:?}"
        );
        assert!(
            error.to_string().contains("digest mismatch"),
            "the corruption must be caught by digest recomputation, got: {error}"
        );
        assert!(
            !error.to_string().contains("composite-generation-authority"),
            "a content-corrupt artifact must never reach the composite-authority refusal"
        );
        assert_no_embed_invocations_since(&embedder, 0);
        assert_eq!(queue.pending_count(), 1, "refusal fires before drain");
        let drained = queue.drain_batch();
        assert_eq!(drained[0].retry_count, 0, "no retry budget consumed");
    });
}

/// Required test (i), v2 case: the full pre-drain admission (option A)
/// performs no writes on the canonical directory — nothing is written,
/// truncated, deleted, or opened writable; bytes, directory entries,
/// sizes, and mtimes are all invariant across the refused cycle, and no
/// WAL sidecar or staging directory appears. Access metadata is the one
/// deliberate non-guarantee (r3 claim precision, #8366/#8367):
/// admission prefers the no-atime/no-follow `O_NOATIME | O_NOFOLLOW`
/// opener, but its documented ordinary read-only fallback — and the v1
/// WAL sidecar read — may update atime and follow symlinks, which is
/// why `dir_manifest` deliberately never asserts atime.
#[test]
fn pre_drain_full_admission_leaves_canonical_directory_invariant() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
        let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
            "r2-v2-dir-invariance",
            embedder.identity_bundle(),
            &[("doc-old", normalized(V2_DIM, 0.5))],
            3,
        );
        let fast_bytes = std::fs::read(&fast_path).expect("read canonical");
        let manifest_before = dir_manifest(&dir);

        let queue = make_queue(100);
        submit(&queue, "doc-new", "new document");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder.clone(),
            cache,
        );
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("fully admitted v2 still refuses canonical publication");
        assert_invalid_config(
            &error,
            "refresh.canonical_publication",
            "composite-generation-authority-unavailable",
        );
        assert_no_embed_invocations_since(&embedder, 0);

        assert_eq!(
            std::fs::read(&fast_path).expect("reread canonical"),
            fast_bytes
        );
        assert_eq!(
            dir_manifest(&dir),
            manifest_before,
            "full admission must not change names, sizes, or mtimes"
        );
        assert!(
            !frankensearch_index::wal::wal_path_for(&fast_path).exists(),
            "admission must not materialize a WAL sidecar"
        );
        assert!(!dir.join(STAGED_V2_DIR_NAME).exists());
    });
}

/// Required test (iv), integration form — RED on 868c0801 at compile
/// time: `fast_admitted_owner` does not exist there because the r1
/// two-tier open peeled `validated.index` and dropped the owner. After
/// the r2 owner-retention rework, the staged generation retains its
/// sealed admission owners, and replacing the staged file on disk does
/// not affect what the retained owner serves.
#[test]
fn staged_generation_retains_admission_owners() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
        let (dir, cache, _binding, _fast_path) = v2_canonical_fixture(
            "r2-staged-owner-retention",
            embedder.identity_bundle(),
            &[("old-1", normalized(V2_DIM, 0.25))],
            7,
        );
        let queue = make_queue(100);
        submit(&queue, "doc-2", "a brand new document");
        let jobs = queue.drain_batch();
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder,
            cache,
        );
        let staged = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect("staging must succeed");

        let owner = staged
            .fast_admitted_owner()
            .expect("staged generation must retain its fast admission owner");
        assert!(
            owner.published_wal_absent(),
            "staged pathname admission proves WAL absence into the retained owner"
        );
        assert_eq!(owner.witness().record_count, 2, "old-1 + doc-2");
        let witness_before = owner.witness().clone();
        let hits_before = staged
            .index
            .search_fast(&normalized(V2_DIM, 0.25), 1)
            .expect("search staged");
        assert_eq!(hits_before.len(), 1);

        // Replace the staged file with garbage: the retained owner's
        // Arc'd bytes are the authority, not the pathname.
        std::fs::write(&staged.fast_path, b"garbage-not-an-index").expect("clobber staged file");
        let owner = staged
            .fast_admitted_owner()
            .expect("owner remains retained");
        assert_eq!(
            owner.witness(),
            &witness_before,
            "the witness lives in the sealed owner, not the pathname"
        );
        let hits_after = staged
            .index
            .search_fast(&normalized(V2_DIM, 0.25), 1)
            .expect("search staged after clobber");
        assert_eq!(
            hits_after[0].doc_id, hits_before[0].doc_id,
            "reads must serve the admitted bytes, never the pathname"
        );
    });
}

/// Deliberate fail-closed narrowing (r2, documented on the card): an
/// all-tombstoned v1 tier previously classified as bootstrap-replaceable
/// because the mutable open could read record flags. Read-only
/// classification counts record slots conservatively, so this now takes
/// the identityless refusal instead of silently replacing the artifact.
/// Flag-level precision returns with the read-only record-table
/// inspector (observational-open train, index crate root).
#[test]
fn all_tombstoned_v1_fails_closed_as_retaining_content() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("r2-all-tombstoned-fail-closed");
        let queue = make_queue(100);
        let (worker, cache, embedder) = make_worker(queue.clone(), &dir, 256);

        submit(&queue, "doc-1", "First document");
        worker.run_cycle(&cx).await.expect("bootstrap cycle");

        drop(worker);
        drop(cache);
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        {
            let mut index = VectorIndex::open(&fast_path).expect("fixture open");
            assert!(index.soft_delete("doc-1").expect("tombstone the only row"));
            assert_eq!(index.live_count(), 0);
        }
        let fast_bytes = std::fs::read(&fast_path).expect("read tombstoned artifact");
        let cache = Arc::new(
            IndexCache::open(
                &dir,
                TwoTierConfig::default(),
                Box::new(SentinelFileDetector::new()),
            )
            .expect("open tombstoned generation"),
        );
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10)),
            queue.clone(),
            embedder.clone(),
            cache,
        );

        submit(&queue, "doc-2", "Second document");
        let embed_baseline = embedder.embed_invocations();
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("all-tombstoned v1 fails closed under read-only classification");
        assert_invalid_config(
            &error,
            "refresh.fast_index_identity",
            "identityless-fsvi-v1",
        );
        assert_no_embed_invocations_since(&embedder, embed_baseline);
        assert_eq!(
            std::fs::read(&fast_path).expect("reread artifact"),
            fast_bytes,
            "the refused artifact must not be replaced"
        );
        assert_eq!(queue.pending_count(), 1);
    });
}

/// Guard for the r2 flow change: classification no longer deletes a
/// stale WAL beside an EMPTY v1 seed, so the bootstrap rebuild must not
/// let that leftover sidecar resurrect foreign rows into the new
/// generation (`TwoTierIndexBuilder::finish` removes sidecars of tiers it
/// rewrites).
#[test]
fn bootstrap_over_empty_seed_with_stale_wal_does_not_resurrect_foreign_rows() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("r2-bootstrap-stale-wal");
        let queue = make_queue(100);
        let (worker, cache, _embedder) = make_worker(queue.clone(), &dir, 256);

        // Empty v1 seed (from make_worker) + transplanted stale WAL.
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        transplant_stale_wal("r2-bootstrap", &fast_path, 256);

        submit(&queue, "doc-1", "First document");
        let embedded = worker
            .run_cycle(&cx)
            .await
            .expect("empty seed with stale WAL stays bootstrap-replaceable");
        assert_eq!(embedded, 1);

        let current = cache.current();
        let ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
        assert_eq!(ids, vec!["doc-1".to_owned()]);
        assert!(
            !ids.iter().any(|id| id.starts_with("donor-")),
            "no row from the transplanted WAL may leak into the bootstrap: {ids:?}"
        );
        assert!(
            !frankensearch_index::wal::wal_path_for(&fast_path).exists(),
            "the write path must have cleared the dead sidecar during rebuild"
        );
    });
}

// ─── C4-write r3: crash-state fail-closed reopen at the refresh seam ──

/// r3 fault items (ii)/(iv), refresh-seam half (#8366/#8367): the exact
/// on-disk state a pre-r3 crash-after-main-rename left — a fresh
/// generation-1 legacy main REPLACED over the canonical name with the
/// old generation's generation-2 WAL still adjacent — must fail CLOSED
/// at the observation-driven admission seam: a typed refusal before any
/// drain or embedding, with the crash state left byte-identical. The
/// foreign WAL rows are never merged into a refreshed generation. The
/// index-crate side of this pin (observer classification, plain-reopen
/// replay, unconstructability through `finish()`) lives in
/// `frankensearch-index/src/two_tier.rs`.
#[test]
fn legacy_crash_state_generation_collision_refuses_pre_drain_without_embedding() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("r3-crash-state-refusal");
        let queue = make_queue(100);
        let (worker, cache, embedder) = make_worker(queue.clone(), &dir, 256);

        // Old generation: bootstrap a real legacy v1 main, then append a
        // foreign row out-of-band, which creates the generation-2 WAL.
        submit(&queue, "doc-old", "old generation document");
        worker.run_cycle(&cx).await.expect("bootstrap cycle");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let served_dir = temp_index_dir("r3-crash-state-served-generation");
        let served_fast_path = served_dir.join(VECTOR_INDEX_FAST_FILENAME);
        std::fs::copy(&fast_path, &served_fast_path).expect("copy served generation");
        drop(worker);
        drop(cache);
        let cache = Arc::new(
            IndexCache::open_with_paths(
                TwoTierIndexPaths::new(served_fast_path),
                &dir,
                TwoTierConfig::default(),
                Box::new(SentinelFileDetector::new()),
            )
            .expect("open retained served generation"),
        );
        {
            let mut index = VectorIndex::open(&fast_path).expect("open old generation");
            let dim = index.dimension();
            index
                .append("foreign-doc", &vec![0.25_f32; dim])
                .expect("append the foreign WAL resident");
        }
        let wal_path = frankensearch_index::wal::wal_path_for(&fast_path);
        assert!(wal_path.exists(), "fixture WAL must exist");

        // Simulate the pre-authority crash point: a replacement main that
        // did NOT bump the generation (the old in-place republish shape)
        // is renamed over the canonical name; the WAL removal never ran.
        // Generation-RELATIVE: the replacement carries the OLD main's
        // generation, so the leftover `next(old)` WAL collides exactly.
        let old_gen = VectorIndex::peek_compaction_gen(&fast_path).expect("peek old generation");
        let scratch = dir.join("replacement.scratch");
        {
            let mut writer = VectorIndex::create(&scratch, "stub-fast", 256)
                .expect("create replacement")
                .with_generation(old_gen);
            writer
                .write_record("doc-new", &normalized(256, 0.5))
                .expect("write replacement row");
            writer.finish().expect("finish replacement");
        }
        std::fs::rename(&scratch, &fast_path).expect("simulate crash after rename");

        // Collision precondition, proven read-only: the leftover WAL
        // reads as ACTIVE against the fresh generation-1 main.
        let observation = observe_tier(&fast_path).expect("observe crash state");
        assert!(
            matches!(observation, FsviTierObservation::V1(_)),
            "crash state must observe as V1, got {observation:?}"
        );
        let FsviTierObservation::V1(observed) = observation else {
            return;
        };
        assert_eq!(
            observed.active_wal_records, 1,
            "the generation-2 WAL must read as ACTIVE against the fresh \
             generation-1 main (the exact collision)"
        );
        assert!(observed.wal_sidecar_present);
        assert!(
            observed.retains_content(),
            "the crash state retains content"
        );

        let fast_bytes = std::fs::read(&fast_path).expect("read crash-state main");
        let wal_bytes = std::fs::read(&wal_path).expect("read crash-state wal");
        let manifest_before = dir_manifest(&dir);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10)),
            queue.clone(),
            embedder.clone(),
            cache.clone(),
        );

        // The refusing cycle: fail closed, pre-drain, zero embedding.
        submit(&queue, "doc-2", "second document");
        let embed_baseline = embedder.embed_invocations();
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("the generation-collision crash state must fail closed");
        assert_invalid_config(
            &error,
            "refresh.fast_index_identity",
            "identityless-fsvi-v1",
        );
        assert_no_embed_invocations_since(&embedder, embed_baseline);
        assert_eq!(queue.pending_count(), 1, "refusal fires before drain");

        // The crash state is left exactly as found: no resurrection into
        // a refreshed generation, no deletion, no replacement.
        assert_eq!(
            std::fs::read(&fast_path).expect("reread main"),
            fast_bytes,
            "the crash-state main must be byte-identical after refusal"
        );
        assert_eq!(
            std::fs::read(&wal_path).expect("reread wal"),
            wal_bytes,
            "the crash-state WAL must be byte-identical after refusal"
        );
        assert_eq!(dir_manifest(&dir), manifest_before);

        // The serving cache still holds the pre-crash generation: the
        // foreign row was never merged anywhere.
        let current = cache.current();
        let ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
        assert_eq!(ids, vec!["doc-old".to_owned()]);
        assert!(
            !ids.iter().any(|id| id == "foreign-doc"),
            "no foreign WAL row may reach the served generation: {ids:?}"
        );
    });
}
