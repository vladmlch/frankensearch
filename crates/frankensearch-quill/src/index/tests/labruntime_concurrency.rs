use super::*;

const E6_5_QUERIES: [&str; 5] = ["old", "new", "newcomer", "alpha OR beta OR gamma", "epoch"];

type E6_5QueryArtifact = Vec<(String, QuillSearchResult, Vec<u32>)>;

fn e6_5_query_artifact(index: &QuillIndex, cx: &Cx) -> E6_5QueryArtifact {
    E6_5_QUERIES
        .iter()
        .map(|query| {
            let ranked = index
                .search_paginated(cx, query, 100, 0, true)
                // ubs:ignore -- A failed ranked query must fail this test's observation helper.
                .unwrap_or_else(|error| panic!("E6.5 ranked query {query:?}: {error}"));
            let docids = index
                .collect_docids(cx, query)
                // ubs:ignore -- A failed scoreless query must fail this test's observation helper.
                .unwrap_or_else(|error| panic!("E6.5 scoreless query {query:?}: {error}"));
            ((*query).to_owned(), ranked, docids)
        })
        .collect()
}

async fn e6_5_watch_oracle(cx: &Cx) -> Vec<E6_5QueryArtifact> {
    let index = QuillIndex::in_memory(deterministic_config()).expect("E6.5 watch oracle");
    LexicalWrite::index_documents(
        &index,
        cx,
        &[
            IndexableDocument::new("first", "old alpha epoch"),
            IndexableDocument::new("second", "old beta epoch"),
        ],
    )
    .await
    .expect("seed E6.5 watch oracle");
    LexicalWrite::commit(&index, cx)
        .await
        .expect("publish E6.5 watch oracle");
    let mut artifacts = vec![e6_5_query_artifact(&index, cx)];

    LexicalWrite::index_documents(
        &index,
        cx,
        &[
            IndexableDocument::new("first", "new alpha epoch"),
            IndexableDocument::new("second", "new beta epoch"),
        ],
    )
    .await
    .expect("replace E6.5 watch oracle batch");
    artifacts.push(e6_5_query_artifact(&index, cx));

    assert!(
        index
            .delete_document(cx, "second")
            .await
            .expect("delete E6.5 watch oracle row")
    );
    artifacts.push(e6_5_query_artifact(&index, cx));

    LexicalWrite::index_document(
        &index,
        cx,
        &IndexableDocument::new("third", "newcomer gamma epoch"),
    )
    .await
    .expect("stage E6.5 watch oracle newcomer");
    LexicalWrite::commit(&index, cx)
        .await
        .expect("publish E6.5 watch oracle newcomer");
    artifacts.push(e6_5_query_artifact(&index, cx));
    artifacts
}

#[test]
fn e3_9_labruntime_pinned_commit_delete_interleavings_are_atomic() {
    run_with_cx(|cx| async move {
        for seed in e3_9_seed_corpus() {
            let index =
                Arc::new(QuillIndex::in_memory(deterministic_config()).expect("memory index"));
            let alpha = IndexableDocument::new("alpha", "alpha committed");
            index
                .index_document(&cx, &alpha)
                .await
                .expect("accumulate alpha");
            index.commit(&cx).await.expect("publish alpha");
            let beta = IndexableDocument::new("beta", "beta pending");
            index
                .index_document(&cx, &beta)
                .await
                .expect("accumulate beta");

            let writer = Arc::clone(&index.writer);
            let commit_succeeded = Arc::new(AtomicBool::new(false));
            // 1 = delete followed the commit and succeeded; 2 = delete
            // serialized first and correctly rejected the pending commit.
            let delete_outcome = Arc::new(AtomicU8::new(0));
            let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let holder_writer = Arc::clone(&writer);
            let (holder, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let holder_cx = Cx::for_testing();
                    let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                        .await
                        .expect("acquire interleaving gate");
                    while holder_writer.waiters() < 2 {
                        yield_now().await;
                    }
                    drop(guard);
                })
                .expect("create interleaving gate");

            let commit_index = Arc::clone(&index);
            let commit_result = Arc::clone(&commit_succeeded);
            let (commit_task, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let task_cx = Cx::for_testing();
                    commit_index.commit(&task_cx).await.unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: commit failed: {error}") // ubs:ignore -- Failed commit must fail this seeded test.
                    });
                    commit_result.store(true, Ordering::SeqCst);
                })
                .expect("create commit task");

            let delete_index = Arc::clone(&index);
            let delete_result = Arc::clone(&delete_outcome);
            let (delete_task, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let task_cx = Cx::for_testing();
                    let outcome = match delete_index.delete_document(&task_cx, "alpha").await {
                        Ok(true) => 1,
                        Err(QuillIndexError::InvalidState { detail })
                            if detail.contains("fully committed") =>
                        {
                            2
                        }
                        Ok(false) => {
                            panic!("seed={seed:#018x}: live alpha was not deleted") // ubs:ignore -- Missing deletion must fail this test.
                        }
                        Err(error) => {
                            panic!("seed={seed:#018x}: unexpected delete result: {error}") // ubs:ignore -- Unexpected deletion error must fail this test.
                        }
                    };
                    delete_result.store(outcome, Ordering::SeqCst);
                })
                .expect("create delete task");

            lab.scheduler.lock().schedule(holder, 0);
            lab.step_for_test();
            lab.scheduler.lock().schedule(commit_task, 0);
            lab.scheduler.lock().schedule(delete_task, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("commit-delete", seed, &report);

            assert!(
                commit_succeeded.load(Ordering::SeqCst),
                "seed={seed:#018x}: commit did not complete"
            );
            let outcome = delete_outcome.load(Ordering::SeqCst);
            assert_ne!(
                outcome, 0,
                "seed={seed:#018x}: delete task did not complete"
            );
            let snapshot = index
                .snapshot()
                .expect("commit/delete snapshot is authoritative");
            assert!(
                snapshot
                    .resolve_document_id("beta")
                    .expect("resolve beta")
                    .is_some(),
                "seed={seed:#018x}: committed beta was lost"
            );
            let alpha_is_live = snapshot
                .resolve_document_id("alpha")
                .expect("resolve alpha")
                .is_some();
            if outcome == 1 {
                assert!(
                    !alpha_is_live,
                    "seed={seed:#018x}: successful delete left alpha live"
                );
                assert_eq!(snapshot.doc_count(), 1);
            } else {
                assert!(
                    alpha_is_live,
                    "seed={seed:#018x}: rejected delete changed alpha visibility"
                );
                assert_eq!(snapshot.doc_count(), 2);
            }
        }
    });
}

#[test]
fn e3_9_labruntime_cancelled_ingest_and_merge_waiters_leave_no_partial_state() {
    run_with_cx(|cx| async move {
        for seed in e3_9_seed_corpus() {
            let ingest =
                Arc::new(QuillIndex::in_memory(deterministic_config()).expect("memory index"));
            let before = ingest
                .search_snapshot()
                .expect("ingest cancellation snapshot is authoritative");
            let writer = Arc::clone(&ingest.writer);
            let operation_cx = Arc::new(Cx::for_testing());
            let holder_release = Arc::new(AtomicBool::new(false));
            let operation_cancelled = Arc::new(AtomicBool::new(false));
            let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let holder_writer = Arc::clone(&writer);
            let release = Arc::clone(&holder_release);
            let (holder, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let holder_cx = Cx::for_testing();
                    let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                        .await
                        .expect("acquire ingest cancellation gate");
                    while !release.load(Ordering::SeqCst) {
                        yield_now().await;
                    }
                    drop(guard);
                })
                .expect("create ingest cancellation gate");

            let ingest_index = Arc::clone(&ingest);
            let ingest_cx = Arc::clone(&operation_cx);
            let cancel_observed = Arc::clone(&operation_cancelled);
            let (ingest_task, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let document = IndexableDocument::new("cancelled", "never published");
                    match ingest_index.index_document(&ingest_cx, &document).await {
                        Err(QuillIndexError::Cancelled {
                            phase: "index writer lock",
                        }) => {
                            cancel_observed.store(true, Ordering::SeqCst);
                        }
                        result => {
                            panic!(
                                // ubs:ignore -- Wrong cancellation outcome must fail this seeded test.
                                "seed={seed:#018x}: unexpected cancelled-ingest result: {result:?}"
                            )
                        }
                    }
                })
                .expect("create cancelled ingest");

            let cancel_writer = Arc::clone(&writer);
            let cancel_cx = Arc::clone(&operation_cx);
            let release = Arc::clone(&holder_release);
            let (canceller, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    while cancel_writer.waiters() == 0 {
                        yield_now().await;
                    }
                    cancel_cx.set_cancel_requested(true);
                    release.store(true, Ordering::SeqCst);
                })
                .expect("create ingest canceller");

            lab.scheduler.lock().schedule(holder, 0);
            lab.step_for_test();
            lab.scheduler.lock().schedule(ingest_task, 0);
            lab.scheduler.lock().schedule(canceller, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("cancel-ingest", seed, &report);
            assert!(
                operation_cancelled.load(Ordering::SeqCst),
                "seed={seed:#018x}: ingest waiter did not observe cancellation"
            );
            assert!(Arc::ptr_eq(
                &before,
                &ingest
                    .search_snapshot()
                    .expect("ingest cancellation snapshot is authoritative"),
            ));
            assert!(!ingest.has_uncommitted_changes());
            assert_eq!(
                ingest
                    .doc_count()
                    .expect("cancelled ingest count is authoritative"),
                0
            );

            let merge = Arc::new(concat_merge_fixture_index(&cx).await);
            let before = merge
                .search_snapshot()
                .expect("merge cancellation snapshot is authoritative");
            let before_evidence = q1_ob2a_query_evidence(&merge, &cx);
            let source_ids = committed_segment_ids(&merge);
            let output_segment_id = fresh_merge_segment_id(&merge, seed);
            let writer = Arc::clone(&merge.writer);
            let operation_cx = Arc::new(Cx::for_testing());
            let holder_release = Arc::new(AtomicBool::new(false));
            let operation_cancelled = Arc::new(AtomicBool::new(false));
            let mut lab = LabRuntime::new(LabConfig::new(seed.rotate_left(17)).max_steps(100_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let holder_writer = Arc::clone(&writer);
            let release = Arc::clone(&holder_release);
            let (holder, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let holder_cx = Cx::for_testing();
                    let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                        .await
                        .expect("acquire merge cancellation gate");
                    while !release.load(Ordering::SeqCst) {
                        yield_now().await;
                    }
                    drop(guard);
                })
                .expect("create merge cancellation gate");

            let merge_index = Arc::clone(&merge);
            let merge_cx = Arc::clone(&operation_cx);
            let cancel_observed = Arc::clone(&operation_cancelled);
            let (merge_task, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    match merge_index
                        .concat_merge(&merge_cx, &source_ids, output_segment_id, 1_700_000_000)
                        .await
                    {
                        Err(QuillIndexError::Cancelled {
                            phase: "concat merge writer lock",
                        }) => {
                            cancel_observed.store(true, Ordering::SeqCst);
                        }
                        Ok(_) => {
                            panic!("seed={seed:#018x}: cancelled merge unexpectedly succeeded") // ubs:ignore -- Ignored cancellation must fail this test.
                        }
                        Err(error) => {
                            panic! /* ubs:ignore -- Test rejects the wrong cancellation error. */ (
                            "seed={seed:#018x}: unexpected cancelled-merge error: {error}"
                        )
                        }
                    }
                })
                .expect("create cancelled merge");

            let cancel_writer = Arc::clone(&writer);
            let cancel_cx = Arc::clone(&operation_cx);
            let release = Arc::clone(&holder_release);
            let (canceller, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    while cancel_writer.waiters() == 0 {
                        yield_now().await;
                    }
                    cancel_cx.set_cancel_requested(true);
                    release.store(true, Ordering::SeqCst);
                })
                .expect("create merge canceller");

            lab.scheduler.lock().schedule(holder, 0);
            lab.step_for_test();
            lab.scheduler.lock().schedule(merge_task, 0);
            lab.scheduler.lock().schedule(canceller, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("cancel-merge", seed, &report);
            assert!(
                operation_cancelled.load(Ordering::SeqCst),
                "seed={seed:#018x}: merge waiter did not observe cancellation"
            );
            assert!(Arc::ptr_eq(
                &before,
                &merge
                    .search_snapshot()
                    .expect("merge cancellation snapshot is authoritative"),
            ));
            assert_eq!(q1_ob2a_query_evidence(&merge, &cx), before_evidence);
        }
    });
}

#[test]
fn e3_9_labruntime_compaction_racing_search_keeps_complete_held_snapshots() {
    for seed in e3_9_seed_corpus() {
        let index = Arc::new(q1_ob4_tombstoned_index(20, &[0, 1, 2, 3, 4]));
        let query_cx = Cx::for_testing();
        let expected = q1_ob4_query_evidence(&index, &query_cx);
        let held = index
            .snapshot()
            .expect("compaction held snapshot is authoritative");
        let held_generation = held.loaded_manifest().manifest.generation;
        let weak = Arc::downgrade(&held);
        let reader_started = Arc::new(AtomicBool::new(false));
        let writer_done = Arc::new(AtomicBool::new(false));
        let reader_passes = Arc::new(AtomicUsize::new(0));
        let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);

        let reader_index = Arc::clone(&index);
        let reader_expected = expected.clone();
        let reader_started_flag = Arc::clone(&reader_started);
        let reader_done = Arc::clone(&writer_done);
        let passes = Arc::clone(&reader_passes);
        let (reader, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                reader_started_flag.store(true, Ordering::SeqCst);
                while !reader_done.load(Ordering::SeqCst) {
                    assert_eq!(
                        q1_ob4_query_evidence(&reader_index, &cx),
                        reader_expected,
                        "seed={seed:#018x}: reader observed torn compaction state"
                    );
                    passes.fetch_add(1, Ordering::SeqCst);
                    yield_now().await;
                }
                assert_eq!(
                    q1_ob4_query_evidence(&reader_index, &cx),
                    reader_expected,
                    "seed={seed:#018x}: post-compaction query drift"
                );
                passes.fetch_add(1, Ordering::SeqCst);
            })
            .expect("create compaction reader");

        let writer_index = Arc::clone(&index);
        let writer_started = Arc::clone(&reader_started);
        let writer_finished = Arc::clone(&writer_done);
        let (writer, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !writer_started.load(Ordering::SeqCst) {
                    yield_now().await;
                }
                let cx = Cx::for_testing();
                let report = writer_index
                    .compact(&cx, CompactionPolicy::default())
                    .await
                    .unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: compaction failed: {error}") // ubs:ignore -- Failed compaction must fail this positive test.
                    });
                assert!(report.changed(), "seed={seed:#018x}: expected compaction");
                writer_finished.store(true, Ordering::SeqCst);
            })
            .expect("create compaction writer");

        lab.scheduler.lock().schedule(reader, 0);
        lab.step_for_test();
        assert!(reader_started.load(Ordering::SeqCst));
        lab.scheduler.lock().schedule(writer, 0);
        let report = lab.run_until_quiescent_with_report();
        assert_e3_9_lab_report("compaction-search", seed, &report);
        assert!(
            reader_passes.load(Ordering::SeqCst) >= 2,
            "seed={seed:#018x}: reader did not span the publication"
        );
        assert_eq!(q1_ob4_query_evidence(&index, &query_cx), expected);
        assert_eq!(
            index
                .snapshot()
                .expect("compaction successor snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            held_generation + 1
        );
        assert_eq!(held.loaded_manifest().manifest.generation, held_generation);
        held.segments()[0]
            .verify()
            // ubs:ignore -- Invalid held segment must fail this retention test.
            .unwrap_or_else(|error| panic!("seed={seed:#018x}: held segment failed: {error}"));
        assert!(
            weak.upgrade().is_some(),
            "seed={seed:#018x}: held snapshot retired too early"
        );
        drop(held);
        assert!(
            weak.upgrade().is_none(),
            "seed={seed:#018x}: old snapshot survived its last reader"
        );
    }
}

#[test]
fn e6_5_labruntime_concurrent_ingest_and_search_are_snapshot_atomic() {
    run_with_cx(|cx| async move {
        for seed in e6_5_seed_corpus() {
            let initial = IndexableDocument::new("stable", "old alpha epoch");
            let additions = vec![
                IndexableDocument::new("next-a", "new beta epoch"),
                IndexableDocument::new("next-b", "new gamma epoch"),
            ];
            let index =
                Arc::new(QuillIndex::in_memory(deterministic_config()).expect("E6.5 index"));
            index
                .index_document(&cx, &initial)
                .await
                .expect("seed E6.5 index");
            index.commit(&cx).await.expect("publish E6.5 seed");
            let held_old_snapshot = index
                .search_snapshot()
                .expect("concurrent ingest snapshot is authoritative");
            let old_artifact = e6_5_query_artifact(&index, &cx);

            let oracle =
                QuillIndex::in_memory(deterministic_config()).expect("E6.5 successor oracle");
            oracle
                .index_document(&cx, &initial)
                .await
                .expect("seed E6.5 successor oracle");
            oracle
                .index_documents(&cx, &additions)
                .await
                .expect("extend E6.5 successor oracle");
            oracle
                .commit(&cx)
                .await
                .expect("publish E6.5 successor oracle");
            let new_artifact = e6_5_query_artifact(&oracle, &cx);
            assert_ne!(old_artifact, new_artifact);

            let reader_started = Arc::new(AtomicBool::new(false));
            let writer_done = Arc::new(AtomicBool::new(false));
            let old_seen = Arc::new(AtomicBool::new(false));
            let new_seen = Arc::new(AtomicBool::new(false));
            let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(200_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let reader_index = Arc::clone(&index);
            let reader_started_flag = Arc::clone(&reader_started);
            let reader_done = Arc::clone(&writer_done);
            let reader_old_seen = Arc::clone(&old_seen);
            let reader_new_seen = Arc::clone(&new_seen);
            let reader_old = old_artifact.clone();
            let reader_new = new_artifact.clone();
            let (reader, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let task_cx = Cx::for_testing();
                    reader_started_flag.store(true, Ordering::SeqCst);
                    while !reader_done.load(Ordering::SeqCst) {
                        let observed = e6_5_query_artifact(&reader_index, &task_cx);
                        if observed == reader_old {
                            reader_old_seen.store(true, Ordering::SeqCst);
                        } else if observed == reader_new {
                            reader_new_seen.store(true, Ordering::SeqCst);
                        } else {
                            panic! /* ubs:ignore -- A torn publication must fail this test. */ (
                                "seed={seed:#018x}: reader observed a torn ingest publication"
                            );
                        }
                        yield_now().await;
                    }
                    assert_eq!(
                        e6_5_query_artifact(&reader_index, &task_cx),
                        reader_new,
                        "seed={seed:#018x}: final reader snapshot is not the successor"
                    );
                    reader_new_seen.store(true, Ordering::SeqCst);
                })
                .expect("create E6.5 snapshot reader");

            let writer_index = Arc::clone(&index);
            let writer_started = Arc::clone(&reader_started);
            let writer_finished = Arc::clone(&writer_done);
            let writer_old = old_artifact.clone();
            let (writer, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    while !writer_started.load(Ordering::SeqCst) {
                        yield_now().await;
                    }
                    let task_cx = Cx::for_testing();
                    writer_index
                        .index_documents(&task_cx, &additions)
                        .await
                        .unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: concurrent ingest failed: {error}") // ubs:ignore -- Failed ingest must fail this positive test.
                        });
                    assert_eq!(
                        e6_5_query_artifact(&writer_index, &task_cx),
                        writer_old,
                        "seed={seed:#018x}: uncommitted ingest leaked into search"
                    );
                    yield_now().await;
                    writer_index.commit(&task_cx).await.unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: concurrent commit failed: {error}") // ubs:ignore -- Failed commit must fail this positive test.
                    });
                    writer_finished.store(true, Ordering::SeqCst);
                })
                .expect("create E6.5 ingest writer");

            lab.scheduler.lock().schedule(reader, 0);
            lab.step_for_test();
            lab.scheduler.lock().schedule(writer, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("e6.5-ingest-search", seed, &report);
            let replay = format!(
                "seed={seed:#018x} fingerprint={:#018x}",
                report.trace_fingerprint
            );
            assert!(
                old_seen.load(Ordering::SeqCst),
                "{replay}: reader never observed the old publication"
            );
            assert!(
                new_seen.load(Ordering::SeqCst),
                "{replay}: reader never observed the successor publication"
            );
            assert_eq!(
                held_old_snapshot.live_doc_count(),
                1,
                "{replay}: held reader snapshot changed after publication"
            );
            assert_eq!(
                index
                    .doc_count()
                    .expect("concurrent ingest count is authoritative"),
                3,
                "{replay}: successor document count"
            );
        }
    });
}

#[test]
fn e6_5_labruntime_cancelled_commit_and_seal_waiters_publish_nothing() {
    run_with_cx(|cx| async move {
        for seed in e6_5_seed_corpus() {
            let commit_index =
                Arc::new(QuillIndex::in_memory(deterministic_config()).expect("commit index"));
            commit_index
                .index_document(
                    &cx,
                    &IndexableDocument::new("commit-cancelled", "never partly visible"),
                )
                .await
                .expect("stage cancellable commit");
            let commit_before = commit_index
                .search_snapshot()
                .expect("cancelled commit snapshot is authoritative");
            let commit_writer = Arc::clone(&commit_index.writer);
            let commit_cx = Arc::new(Cx::for_testing());
            let commit_release = Arc::new(AtomicBool::new(false));
            let commit_cancelled = Arc::new(AtomicBool::new(false));
            let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let holder_writer = Arc::clone(&commit_writer);
            let holder_release = Arc::clone(&commit_release);
            let (holder, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let holder_cx = Cx::for_testing();
                    let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                        .await
                        .expect("acquire E6.5 commit cancellation gate");
                    while !holder_release.load(Ordering::SeqCst) {
                        yield_now().await;
                    }
                    drop(guard);
                })
                .expect("create E6.5 commit cancellation gate");

            let operation_index = Arc::clone(&commit_index);
            let operation_cx = Arc::clone(&commit_cx);
            let operation_cancelled = Arc::clone(&commit_cancelled);
            let (operation, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    match operation_index.commit(&operation_cx).await {
                        Err(QuillIndexError::Cancelled {
                            phase: "commit writer lock",
                        }) => {
                            operation_cancelled.store(true, Ordering::SeqCst);
                        }
                        Ok(_) => {
                            panic!("seed={seed:#018x}: cancelled commit unexpectedly succeeded") // ubs:ignore -- Ignored cancellation must fail this test.
                        }
                        Err(error) => {
                            panic! /* ubs:ignore -- Test rejects the wrong cancellation error. */ (
                            "seed={seed:#018x}: unexpected cancelled-commit error: {error}"
                        )
                        }
                    }
                })
                .expect("create E6.5 cancelled commit");

            let cancel_writer = Arc::clone(&commit_writer);
            let cancel_cx = Arc::clone(&commit_cx);
            let cancel_release = Arc::clone(&commit_release);
            let (canceller, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    while cancel_writer.waiters() == 0 {
                        yield_now().await;
                    }
                    cancel_cx.set_cancel_requested(true);
                    cancel_release.store(true, Ordering::SeqCst);
                })
                .expect("create E6.5 commit canceller");

            lab.scheduler.lock().schedule(holder, 0);
            lab.step_for_test();
            lab.scheduler.lock().schedule(operation, 0);
            lab.scheduler.lock().schedule(canceller, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("e6.5-cancel-commit", seed, &report);
            let replay = format!(
                "seed={seed:#018x} fingerprint={:#018x}",
                report.trace_fingerprint
            );
            assert!(
                commit_cancelled.load(Ordering::SeqCst),
                "{replay}: commit waiter missed cancellation"
            );
            assert!(
                Arc::ptr_eq(
                    &commit_before,
                    &commit_index
                        .search_snapshot()
                        .expect("cancelled commit snapshot is authoritative"),
                ),
                "{replay}: cancelled commit changed the published snapshot"
            );
            assert!(
                commit_index.has_uncommitted_changes(),
                "{replay}: cancelled commit discarded retry state"
            );
            commit_index
                .commit(&cx)
                .await
                .expect("retry cancelled E6.5 commit");
            assert_eq!(
                commit_index
                    .doc_count()
                    .expect("retried commit count is authoritative"),
                1,
                "{replay}: commit retry"
            );

            let seal_index =
                Arc::new(QuillIndex::in_memory(deterministic_config()).expect("seal index"));
            let generation = seal_index
                .snapshot()
                .expect("seal fixture snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation;
            let mut delta =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("E6.5 seal Delta");
            apply_alpha_delta(&mut delta, 0, "seal-cancelled", 1);
            let sealed = Arc::new(delta.freeze(generation));
            seal_index
                .publish_delta_table(vec![Arc::clone(&sealed)])
                .expect("publish E6.5 seal Delta");
            let seal_before = seal_index
                .search_snapshot()
                .expect("cancelled seal snapshot is authoritative");
            let seal_artifact_before = e6_5_query_artifact(&seal_index, &cx);
            let seal_writer = Arc::clone(&seal_index.writer);
            let seal_cx = Arc::new(Cx::for_testing());
            let seal_release = Arc::new(AtomicBool::new(false));
            let seal_cancelled = Arc::new(AtomicBool::new(false));
            let mut lab = LabRuntime::new(LabConfig::new(seed.rotate_left(23)).max_steps(100_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let holder_writer = Arc::clone(&seal_writer);
            let holder_release = Arc::clone(&seal_release);
            let (holder, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let holder_cx = Cx::for_testing();
                    let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                        .await
                        .expect("acquire E6.5 seal cancellation gate");
                    while !holder_release.load(Ordering::SeqCst) {
                        yield_now().await;
                    }
                    drop(guard);
                })
                .expect("create E6.5 seal cancellation gate");

            let operation_index = Arc::clone(&seal_index);
            let operation_cx = Arc::clone(&seal_cx);
            let operation_source = Arc::clone(&sealed);
            let operation_cancelled = Arc::clone(&seal_cancelled);
            let (operation, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    match operation_index
                        .seal_delta_snapshot(
                            &operation_cx,
                            operation_source,
                            Vec::new(),
                            DeltaFlushInput {
                                segment_id: seed ^ 0xe6_5000_5ea1_0000,
                                created_unix_s: 0,
                                engine_version: CURRENT_ENGINE_VERSION,
                            },
                        )
                        .await
                    {
                        Err(QuillIndexError::Cancelled {
                            phase: "Delta seal writer lock",
                        }) => {
                            operation_cancelled.store(true, Ordering::SeqCst);
                        }
                        Ok(_) => {
                            panic!("seed={seed:#018x}: cancelled seal unexpectedly succeeded") // ubs:ignore -- Ignored cancellation must fail this test.
                        }
                        Err(error) => {
                            panic! /* ubs:ignore -- Test rejects the wrong cancellation error. */ (
                            "seed={seed:#018x}: unexpected cancelled-seal error: {error}"
                        )
                        }
                    }
                })
                .expect("create E6.5 cancelled seal");

            let cancel_writer = Arc::clone(&seal_writer);
            let cancel_cx = Arc::clone(&seal_cx);
            let cancel_release = Arc::clone(&seal_release);
            let (canceller, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    while cancel_writer.waiters() == 0 {
                        yield_now().await;
                    }
                    cancel_cx.set_cancel_requested(true);
                    cancel_release.store(true, Ordering::SeqCst);
                })
                .expect("create E6.5 seal canceller");

            lab.scheduler.lock().schedule(holder, 0);
            lab.step_for_test();
            lab.scheduler.lock().schedule(operation, 0);
            lab.scheduler.lock().schedule(canceller, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("e6.5-cancel-seal", seed, &report);
            let replay = format!(
                "seed={seed:#018x} fingerprint={:#018x}",
                report.trace_fingerprint
            );
            assert!(
                seal_cancelled.load(Ordering::SeqCst),
                "{replay}: seal waiter missed cancellation"
            );
            assert!(
                Arc::ptr_eq(
                    &seal_before,
                    &seal_index
                        .search_snapshot()
                        .expect("cancelled seal snapshot is authoritative"),
                ),
                "{replay}: cancelled seal changed the published snapshot"
            );
            assert_eq!(
                e6_5_query_artifact(&seal_index, &cx),
                seal_artifact_before,
                "{replay}: cancelled seal created a visibility gap"
            );
            assert_eq!(
                seal_index
                    .search_snapshot()
                    .expect("cancelled seal snapshot is authoritative")
                    .delta_count(),
                1
            );
            seal_index
                .seal_delta_snapshot(
                    &cx,
                    sealed,
                    Vec::new(),
                    DeltaFlushInput {
                        segment_id: seed ^ 0xe6_5000_5ea1_0000,
                        created_unix_s: 0,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .await
                .expect("retry cancelled E6.5 seal");
            assert_eq!(
                seal_index
                    .search_snapshot()
                    .expect("retried seal snapshot is authoritative")
                    .delta_count(),
                0
            );
            assert_eq!(
                e6_5_query_artifact(&seal_index, &cx),
                seal_artifact_before,
                "{replay}: seal retry changed query evidence"
            );
        }
    });
}

#[test]
fn e6_5_labruntime_watch_stream_is_continuous_and_reopens_exactly() {
    run_with_cx(|cx| async move {
        let allowed = Arc::new(e6_5_watch_oracle(&cx).await);
        assert_eq!(allowed.len(), 4);

        for seed in e6_5_seed_corpus() {
            let directory = tempfile::tempdir().expect("E6.5 watch directory");
            let index = Arc::new(
                QuillIndex::create(&cx, directory.path(), deterministic_config())
                    .await
                    .expect("create E6.5 watch index"),
            );
            LexicalWrite::index_documents(
                index.as_ref(),
                &cx,
                &[
                    IndexableDocument::new("first", "old alpha epoch"),
                    IndexableDocument::new("second", "old beta epoch"),
                ],
            )
            .await
            .expect("seed E6.5 watch index");
            LexicalWrite::commit(index.as_ref(), &cx)
                .await
                .expect("publish E6.5 watch seed");
            assert_eq!(e6_5_query_artifact(&index, &cx), allowed[0]);

            let reader_started = Arc::new(AtomicBool::new(false));
            let writer_done = Arc::new(AtomicBool::new(false));
            let observed_states = Arc::new(AtomicU8::new(0));
            let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(300_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let reader_index = Arc::clone(&index);
            let reader_allowed = Arc::clone(&allowed);
            let reader_started_flag = Arc::clone(&reader_started);
            let reader_done = Arc::clone(&writer_done);
            let reader_observed = Arc::clone(&observed_states);
            let (reader, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let task_cx = Cx::for_testing();
                    reader_started_flag.store(true, Ordering::SeqCst);
                    loop {
                        let artifact = e6_5_query_artifact(&reader_index, &task_cx);
                        let Some(state) = reader_allowed
                            .iter()
                            .position(|expected| *expected == artifact)
                        else {
                            panic! /* ubs:ignore -- A torn generation must fail this test. */ (
                                "seed={seed:#018x}: watch reader observed a torn generation"
                            );
                        };
                        reader_observed.fetch_or(1_u8 << state, Ordering::SeqCst);
                        if reader_done.load(Ordering::SeqCst) {
                            break;
                        }
                        yield_now().await;
                    }
                })
                .expect("create E6.5 watch reader");

            let writer_index = Arc::clone(&index);
            let writer_started = Arc::clone(&reader_started);
            let writer_done_flag = Arc::clone(&writer_done);
            let writer_observed = Arc::clone(&observed_states);
            let writer_allowed = Arc::clone(&allowed);
            let (writer, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    while !writer_started.load(Ordering::SeqCst) {
                        yield_now().await;
                    }
                    let task_cx = Cx::for_testing();
                    LexicalWrite::index_documents(
                        writer_index.as_ref(),
                        &task_cx,
                        &[
                            IndexableDocument::new("first", "new alpha epoch"),
                            IndexableDocument::new("second", "new beta epoch"),
                        ],
                    )
                    .await
                    .unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: watch upsert failed: {error}") // ubs:ignore -- Failed watch upsert must fail this test.
                    });
                    while writer_observed.load(Ordering::SeqCst) & (1 << 1) == 0 {
                        yield_now().await;
                    }

                    assert!(
                        writer_index
                            .delete_document(&task_cx, "second")
                            .await
                            .unwrap_or_else(|error| {
                                panic!("seed={seed:#018x}: watch delete failed: {error}") // ubs:ignore -- Failed watch deletion must fail this test.
                            })
                    );
                    while writer_observed.load(Ordering::SeqCst) & (1 << 2) == 0 {
                        yield_now().await;
                    }

                    LexicalWrite::index_document(
                        writer_index.as_ref(),
                        &task_cx,
                        &IndexableDocument::new("third", "newcomer gamma epoch"),
                    )
                    .await
                    .unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: watch newcomer ingest failed: {error}") // ubs:ignore -- Failed newcomer ingest must fail this test.
                    });
                    assert_eq!(
                        e6_5_query_artifact(&writer_index, &task_cx),
                        writer_allowed[2],
                        "seed={seed:#018x}: uncommitted watch row became visible"
                    );
                    LexicalWrite::commit(writer_index.as_ref(), &task_cx)
                        .await
                        .unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: watch commit failed: {error}") // ubs:ignore -- Failed watch commit must fail this test.
                        });
                    writer_done_flag.store(true, Ordering::SeqCst);
                })
                .expect("create E6.5 watch writer");

            lab.scheduler.lock().schedule(reader, 0);
            lab.step_for_test();
            lab.scheduler.lock().schedule(writer, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("e6.5-watch-stream", seed, &report);
            let replay = format!(
                "seed={seed:#018x} fingerprint={:#018x}",
                report.trace_fingerprint
            );
            assert_eq!(
                observed_states.load(Ordering::SeqCst),
                0b1111,
                "{replay}: watch reader did not observe every complete generation"
            );
            assert_eq!(
                e6_5_query_artifact(&index, &cx),
                allowed[3],
                "{replay}: watch writer did not publish the final generation"
            );

            drop(index);
            for recovery_round in 0..2 {
                let reopened = QuillIndex::open(&cx, directory.path(), deterministic_config())
                    .await
                    .unwrap_or_else(|error| {
                        panic!("{replay}: recovery round {recovery_round} failed: {error}") // ubs:ignore -- Failed recovery must fail this replay test.
                    });
                assert_eq!(
                    e6_5_query_artifact(&reopened, &cx),
                    allowed[3],
                    "{replay}: recovery round {recovery_round} changed query artifacts"
                );
                assert_eq!(
                    reopened
                        .doc_count()
                        .expect("recovered watch count is authoritative"),
                    2,
                    "{replay}: recovery round {recovery_round} live count"
                );
                drop(reopened);
            }
        }
    });
}

#[test]
fn e6_5_labruntime_same_seed_replays_identical_segments_and_queries() {
    for seed in e6_5_seed_corpus() {
        let documents = (0..12)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("seed-{seed:016x}-{ordinal:02}"),
                    format!(
                        "epoch replay bucket-{} token-{}",
                        (seed ^ ordinal) % 5,
                        ordinal % 3
                    ),
                )
            })
            .collect::<Vec<_>>();
        let first = Arc::new(QuillIndex::in_memory(deterministic_config()).expect("first replay"));
        let second =
            Arc::new(QuillIndex::in_memory(deterministic_config()).expect("second replay"));
        let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(200_000));
        let region = lab.state.create_root_region(Budget::INFINITE);

        let first_index = Arc::clone(&first);
        let first_documents = documents.clone();
        let (first_task, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let task_cx = Cx::for_testing();
                first_index
                    .index_documents(&task_cx, &first_documents)
                    .await
                    .unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: first replay ingest failed: {error}") // ubs:ignore -- Failed replay ingest must fail this test.
                    });
                yield_now().await;
                first_index.commit(&task_cx).await.unwrap_or_else(|error| {
                    panic!("seed={seed:#018x}: first replay commit failed: {error}") // ubs:ignore -- Failed replay commit must fail this test.
                });
            })
            .expect("create first E6.5 replay");

        let second_index = Arc::clone(&second);
        let (second_task, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let task_cx = Cx::for_testing();
                second_index
                    .index_documents(&task_cx, &documents)
                    .await
                    .unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: second replay ingest failed: {error}") // ubs:ignore -- Failed replay ingest must fail this test.
                    });
                yield_now().await;
                second_index.commit(&task_cx).await.unwrap_or_else(|error| {
                    panic!("seed={seed:#018x}: second replay commit failed: {error}") // ubs:ignore -- Failed replay commit must fail this test.
                });
            })
            .expect("create second E6.5 replay");

        lab.scheduler.lock().schedule(first_task, 0);
        lab.scheduler.lock().schedule(second_task, 0);
        let report = lab.run_until_quiescent_with_report();
        assert_e3_9_lab_report("e6.5-deterministic-replay", seed, &report);
        let replay = format!(
            "seed={seed:#018x} fingerprint={:#018x}",
            report.trace_fingerprint
        );

        let first_snapshot = first
            .snapshot()
            .expect("first replay snapshot is authoritative");
        let second_snapshot = second
            .snapshot()
            .expect("second replay snapshot is authoritative");
        assert_eq!(
            first_snapshot.loaded_manifest().manifest,
            second_snapshot.loaded_manifest().manifest,
            "{replay}: deterministic MANIFEST mismatch"
        );
        assert_eq!(
            first_snapshot.segments().len(),
            second_snapshot.segments().len(),
            "{replay}: deterministic segment-count mismatch"
        );
        for (ordinal, (first_segment, second_segment)) in first_snapshot
            .segments()
            .iter()
            .zip(second_snapshot.segments())
            .enumerate()
        {
            assert_eq!(
                first_segment.source_bytes(),
                second_segment.source_bytes(),
                "{replay}: deterministic segment {ordinal} bytes differ"
            );
        }
        let query_cx = Cx::for_testing();
        assert_eq!(
            e6_5_query_artifact(&first, &query_cx),
            e6_5_query_artifact(&second, &query_cx),
            "{replay}: deterministic query artifacts differ"
        );
    }
}
