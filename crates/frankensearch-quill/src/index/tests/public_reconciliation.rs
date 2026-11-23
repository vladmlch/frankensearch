use super::*;

fn is_reconciliation_required(error: &QuillIndexError) -> bool {
    matches!(
        error,
        QuillIndexError::Keeper(KeeperError::PublicationReconciliationRequired { .. })
            | QuillIndexError::PublicationReconciliationRequired { .. }
    )
}

fn is_search_reconciliation_required(error: &SearchError) -> bool {
    match error {
        SearchError::SubsystemError { subsystem, source } if *subsystem == "quill" => source
            .downcast_ref::<QuillIndexError>()
            .is_some_and(is_reconciliation_required),
        _ => false,
    }
}

async fn assert_public_authority_refuses_stale_publication(
    index: &QuillIndex,
    cx: &Cx,
    query: &str,
    document_id: &str,
) {
    macro_rules! assert_refused {
        ($result:expr, $facade:literal) => {{
            let Err(error) = $result else {
                // ubs:ignore -- This test macro must fail when a negative control is accepted.
                panic!(concat!($facade, " must fail closed"));
            };
            assert!(
                is_reconciliation_required(&error),
                "{} must report typed reconciliation-required failure: {error:?}",
                $facade,
            );
        }};
    }

    assert_refused!(index.snapshot(), "snapshot");
    assert_refused!(index.search_snapshot(), "search_snapshot");
    assert_refused!(index.doc_count(), "doc_count");
    assert_refused!(index.document_witness(document_id), "document_witness");
    assert_refused!(
        index.search_paginated(cx, query, 10, 0, true),
        "search_paginated"
    );
    assert_refused!(index.search_results(cx, query, 10), "search_results");
    assert_refused!(index.search_doc_ids(cx, query, 10), "search_doc_ids");
    assert_refused!(
        index.search_with_snippets(cx, query, 10, &SnippetConfig::default()),
        "search_with_snippets"
    );
    assert_refused!(index.collect_docids(cx, query), "collect_docids");
    let count_error =
        LexicalRead::doc_count(index).expect_err("LexicalRead::doc_count must fail closed");
    assert!(
        is_search_reconciliation_required(&count_error),
        "LexicalRead::doc_count must retain the typed reconciliation failure: {count_error:?}"
    );
    let stats_error = SegmentStatsProvider::segment_stats(index)
        .expect_err("SegmentStatsProvider must fail closed");
    assert!(
        is_search_reconciliation_required(&stats_error),
        "SegmentStatsProvider must retain the typed reconciliation failure: {stats_error:?}"
    );
    assert!(
        LexicalRead::search(index, cx, query, 10).await.is_err(),
        "LexicalRead::search must not return the retained stale view"
    );
    assert!(
        LexicalRead::search_candidates(index, cx, query, 10)
            .await
            .is_err(),
        "LexicalRead::search_candidates must not pin the retained stale view"
    );
}

async fn assert_public_authority_observes_reconciled_single_document(
    index: &QuillIndex,
    cx: &Cx,
    generation: u64,
    query: &str,
    document_id: &str,
    expected_live_docs: u64,
) {
    let snapshot = index
        .snapshot()
        .expect("reconciled snapshot is authoritative");
    assert_eq!(snapshot.loaded_manifest().manifest.generation, generation);
    let search_snapshot = index
        .search_snapshot()
        .expect("reconciled search snapshot is authoritative");
    assert_eq!(search_snapshot.keeper_generation(), generation);
    assert_eq!(search_snapshot.live_doc_count(), expected_live_docs);
    assert_eq!(
        index
            .doc_count()
            .expect("reconciled document count is authoritative"),
        expected_live_docs
    );
    assert_eq!(
        LexicalRead::doc_count(index).expect("reconciled trait count is authoritative"),
        usize::try_from(expected_live_docs).expect("test count fits usize")
    );
    let stats = SegmentStatsProvider::segment_stats(index)
        .expect("reconciled segment stats are authoritative");
    assert_eq!(stats.published_generation, generation);
    assert_eq!(
        stats.live_docs,
        usize::try_from(expected_live_docs).expect("test count fits usize")
    );

    let page = index
        .search_paginated(cx, query, 10, 0, true)
        .expect("reconciled paginated search");
    let results = index
        .search_results(cx, query, 10)
        .expect("reconciled enriched search");
    let doc_ids = index
        .search_doc_ids(cx, query, 10)
        .expect("reconciled ID search");
    let snippets = index
        .search_with_snippets(cx, query, 10, &SnippetConfig::default())
        .expect("reconciled snippet search");
    let collected = index
        .collect_docids(cx, query)
        .expect("reconciled document-ID collection");
    let lexical = LexicalRead::search(index, cx, query, 10)
        .await
        .expect("reconciled LexicalRead search");
    let candidates = LexicalRead::search_candidates(index, cx, query, 10)
        .await
        .expect("reconciled LexicalRead candidate search");
    let (mut candidates, context) = candidates.into_parts();
    LexicalRead::hydrate_candidates(index, cx, context.as_ref(), &mut candidates)
        .await
        .expect("hydrate from the candidate scoring snapshot");

    let expected_hits =
        usize::try_from(expected_live_docs).expect("test live document count fits usize");
    assert_eq!(page.hits.len(), expected_hits);
    assert_eq!(results.len(), expected_hits);
    assert_eq!(doc_ids.len(), expected_hits);
    assert_eq!(snippets.len(), expected_hits);
    assert_eq!(collected.len(), expected_hits);
    assert_eq!(lexical.len(), expected_hits);
    assert_eq!(candidates.len(), expected_hits);
    assert_eq!(
        LexicalRead::doc_count(index).expect("reconciled trait count remains authoritative"),
        expected_hits
    );
    if expected_live_docs == 0 {
        assert!(
            index
                .document_witness(document_id)
                .expect("reconciled absence witness")
                .is_none(),
            "the reconciled tombstone must be authoritative"
        );
    } else {
        assert_eq!(page.hits[0].document_id, document_id);
        assert_eq!(results[0].doc_id.as_str(), document_id);
        assert_eq!(doc_ids[0].document_id, document_id);
        assert_eq!(snippets[0].document_id, document_id);
        assert_eq!(lexical[0].doc_id.as_str(), document_id);
        assert_eq!(candidates[0].doc_id.as_str(), document_id);
        assert!(
            index
                .document_witness(document_id)
                .expect("reconciled live witness")
                .is_some(),
            "the reconciled document must have an authority witness"
        );
    }
}

#[test]
fn dropped_delta_seal_after_manifest_install_reconciles_exact_generation() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    // This fixture intentionally pauses inside real blocking filesystem
    // choreography. A configured blocking pool keeps the executor free to
    // observe and release that checkpoint; the lightweight default test
    // runtime otherwise executes spawn_blocking inline by design.
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary Keeper directory");
        let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable index");
        let generation = index
            .snapshot()
            .expect("fresh durable snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let next_seal_seq = index.writer_mut().next_seal_seq;
        let mut delta =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("drop-test Delta source");
        apply_alpha_delta(&mut delta, 0, "drop-survivor", 1);
        let sealed = Arc::new(delta.freeze(generation));
        index
            .publish_delta_table(vec![Arc::clone(&sealed)])
            .expect("publish drop-test Delta");
        let before = index
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("query drop-test Delta before seal");

        let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::TempMovedToCurrent,
        );
        let mut seal = Box::pin(index.seal_delta_snapshot(
            &cx,
            Arc::clone(&sealed),
            Vec::new(),
            DeltaFlushInput {
                segment_id: 0xe5_4031,
                created_unix_s: 1_700_000_031,
                engine_version: CURRENT_ENGINE_VERSION,
            },
        ));
        crate::keeper::drive_manifest_publish_to_checkpoint_for_test(
            &mut seal,
            &pause,
            PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT,
        )
        .await
        .expect("reach the bounded durable Delta-seal rendezvous");
        drop(seal);

        assert!(
            index.writer_mut().pending_delta_seal.is_some(),
            "the actual dropped future must leave its complete proposal retained"
        );
        assert!(matches!(
            index.snapshot(),
            Err(QuillIndexError::PublicationReconciliationRequired {
                published_generation,
                durable_generation,
            }) if published_generation == generation && durable_generation == generation + 1
        ));
        assert_public_authority_refuses_stale_publication(&index, &cx, "alpha", "drop-survivor")
            .await;
        assert!(
            index.has_uncommitted_changes(),
            "an abandoned durable publication is pending writer work, never a clean epoch"
        );
        let installed = Manifest::from_bytes(
            &std::fs::read(directory.path().join("MANIFEST"))
                .expect("read renamed successor MANIFEST"),
        )
        .expect("decode renamed successor MANIFEST");
        assert_eq!(
            installed.generation,
            generation + 1,
            "the real blocking publisher must have installed generation N+1"
        );

        pause.release();
        let resumed = index
            .reconcile_publication(&cx)
            .await
            .expect("reconcile stale writer snapshot and finish the local swap");
        assert!(index.writer_mut().pending_delta_seal.is_none());
        assert!(
            !index.has_uncommitted_changes(),
            "generic reconciliation must resolve the retained Delta retry state"
        );
        assert_eq!(resumed.keeper_generation(), generation + 1);
        assert_eq!(resumed.delta_count(), 0);
        assert_eq!(index.writer_mut().next_seal_seq, next_seal_seq + 1);
        assert_eq!(index.writer_mut().next_lease_base, sealed.lease_end());
        assert_eq!(
            index
                .snapshot()
                .expect("resumed snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation + 1,
            "resume must reconcile N+1 rather than publishing N+2"
        );
        assert_eq!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .expect("reopen exactly reconciled Keeper")
                .loaded_manifest()
                .manifest
                .generation,
            generation + 1
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("query Keeper after exact resume"),
            before
        );
        assert_public_authority_observes_reconciled_single_document(
            &index,
            &cx,
            generation + 1,
            "alpha",
            "drop-survivor",
            1,
        )
        .await;
    });
}

#[test]
fn live_commit_writer_reads_linearize_to_the_pinned_prepublication_view() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary live-writer Keeper directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable live-writer index");
        let generation = index
            .snapshot()
            .expect("fresh live-writer snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        index
            .index_document(
                &cx,
                &IndexableDocument::new("live-writer", "alpha live writer"),
            )
            .await
            .expect("stage live-writer document");

        let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::TempWritten,
        );
        let mut commit = Box::pin(index.commit(&cx));
        crate::keeper::drive_manifest_publish_to_checkpoint_for_test(
            &mut commit,
            &pause,
            PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT,
        )
        .await
        .expect("reach bounded live-writer rendezvous");

        assert_eq!(
            index
                .snapshot()
                .expect("live writer reads retain the prior authoritative view")
                .loaded_manifest()
                .manifest
                .generation,
            generation,
            "a locked writer linearizes lock-free reads before its not-yet-returned mutation"
        );
        assert_eq!(
            index
                .search_snapshot()
                .expect("live writer retains the prior search snapshot")
                .keeper_generation(),
            generation
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("live writer search linearizes before publication")
                .hits
                .len(),
            0
        );
        assert_eq!(
            index
                .doc_count()
                .expect("live writer public count linearizes before publication"),
            0
        );
        assert_eq!(
            LexicalRead::doc_count(&index)
                .expect("live writer trait count retains readable predecessor"),
            0
        );
        let stats = SegmentStatsProvider::segment_stats(&index)
            .expect("live writer stats retain readable predecessor");
        assert_eq!(stats.published_generation, generation);
        assert_eq!(stats.live_docs, 0);
        assert!(
            LexicalRead::search(&index, &cx, "alpha", 10)
                .await
                .expect("live writer LexicalRead search linearizes before publication")
                .is_empty()
        );

        pause.release();
        let published = commit
            .await
            .expect("finish live-writer commit after release");
        assert_eq!(
            published.loaded_manifest().manifest.generation,
            generation + 1
        );
        assert_public_authority_observes_reconciled_single_document(
            &index,
            &cx,
            generation + 1,
            "alpha",
            "live-writer",
            1,
        )
        .await;
    });
}

#[test]
fn claimed_live_publication_refuses_reads_before_any_manifest_slot_moves() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary claimed-publication directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create claimed-publication fixture");
        let generation = index
            .snapshot()
            .expect("fresh claimed-publication snapshot")
            .loaded_manifest()
            .manifest
            .generation;
        index
            .index_document(&cx, &IndexableDocument::new("claimed", "alpha claimed"))
            .await
            .expect("stage claimed-publication document");
        let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::GenerationClaimed,
        );
        let mut commit = Box::pin(index.commit(&cx));
        crate::keeper::drive_manifest_publish_to_checkpoint_for_test(
            &mut commit,
            &pause,
            PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT,
        )
        .await
        .expect("reach pre-slot claimed-publication rendezvous");
        assert_eq!(
            Manifest::from_bytes(
                &std::fs::read(directory.path().join("MANIFEST")).expect("read pre-slot MANIFEST"),
            )
            .expect("decode pre-slot MANIFEST")
            .generation,
            generation,
            "GenerationClaimed pause must precede every MANIFEST slot rename"
        );
        assert_public_authority_refuses_stale_publication(&index, &cx, "alpha", "claimed").await;
        pause.release();
        commit
            .await
            .expect("finish publication after the pre-slot refusal proof");
    });
}

#[test]
fn public_in_memory_commit_rejects_unrepresentable_successor_without_mutation() {
    run_with_cx(|cx| async move {
        let generation = PublicationReadState::MAX_GENERATION;
        let snapshot =
            KeeperSnapshot::in_memory_with_generation_for_test(DEFAULT_SCHEMA, generation)
                .expect("construct maximum-generation in-memory Keeper fixture");
        let mut index = QuillIndex::from_backend(
            IndexBackend::Memory(snapshot),
            DEFAULT_SCHEMA,
            deterministic_config(),
        )
        .expect("bind maximum-generation in-memory facade");
        let before = index
            .snapshot()
            .expect("maximum-generation snapshot starts authoritative");
        index
            .index_document(
                &cx,
                &IndexableDocument::new("max-generation", "unrepresentable successor"),
            )
            .await
            .expect("stage maximum-generation in-memory document");

        let Err(error) = index.commit(&cx).await else {
            panic!("public in-memory commit must reject an unrepresentable successor"); // ubs:ignore -- Test rejects an overflowing generation.
        };
        assert!(matches!(
            error,
            QuillIndexError::Keeper(KeeperError::InvalidTransition { detail })
                if detail.contains("cannot be represented in the packed publication authority word")
        ));

        let after = index
            .snapshot()
            .expect("failed successor leaves the published Arc authoritative");
        assert!(
            Arc::ptr_eq(&before, &after),
            "failed maximum-generation commit must not replace the public snapshot Arc"
        );
        assert_eq!(
            index
                .writer_mut()
                .backend
                .retained_snapshot_for_bookkeeping()
                .loaded_manifest()
                .manifest
                .generation,
            generation,
            "failed maximum-generation commit must not mutate the in-memory Keeper"
        );
    });
}

#[test]
fn known_pre_rename_publish_failure_restores_the_readable_predecessor() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary pre-rename failure directory");
        let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable pre-rename failure fixture");
        let generation = index
            .snapshot()
            .expect("fresh pre-rename snapshot")
            .loaded_manifest()
            .manifest
            .generation;
        index
            .index_document(&cx, &IndexableDocument::new("pre-rename", "alpha retained"))
            .await
            .expect("stage pre-rename document");
        let fault = crate::keeper::fail_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::TempWritten,
        );
        assert!(matches!(
            index.commit(&cx).await,
            Err(QuillIndexError::Keeper(KeeperError::InvalidTransition { detail }))
                if detail.contains("test-only MANIFEST publisher failure")
        ));
        assert_eq!(
            index
                .snapshot()
                .expect("known pre-rename failure leaves predecessor readable")
                .loaded_manifest()
                .manifest
                .generation,
            generation
        );
        assert!(
            index.writer_mut().pending_manifest.is_some(),
            "known pre-rename failure must retain the exact retry proposal"
        );
        drop(fault);
        assert_eq!(
            index
                .commit(&cx)
                .await
                .expect("retry retained pre-rename proposal")
                .loaded_manifest()
                .manifest
                .generation,
            generation + 1
        );
    });
}

#[test]
fn claimed_publish_failure_requires_proven_same_generation_reconciliation() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary claimed-failure directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable claimed-failure fixture");
        let generation = index
            .snapshot()
            .expect("fresh claimed-failure snapshot")
            .loaded_manifest()
            .manifest
            .generation;
        index
            .index_document(
                &cx,
                &IndexableDocument::new("claimed-failure", "alpha claimed failure"),
            )
            .await
            .expect("stage claimed-failure document");
        let fault = crate::keeper::fail_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::GenerationClaimed,
        );

        assert!(matches!(
            index.commit(&cx).await,
            Err(QuillIndexError::Keeper(KeeperError::InvalidTransition { detail }))
                if detail.contains("test-only MANIFEST publisher failure")
        ));
        assert_eq!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .expect("reopen pre-slot durable generation")
                .loaded_manifest()
                .manifest
                .generation,
            generation,
            "GenerationClaimed failure must precede every MANIFEST slot move"
        );
        assert_public_authority_refuses_stale_publication(&index, &cx, "alpha", "claimed-failure")
            .await;

        let reconciled = index
            .reconcile_publication(&cx)
            .await
            .expect("reopen proves the predecessor remains durable authority");
        assert_eq!(reconciled.keeper_generation(), generation);
        assert!(
            index
                .writer
                .try_lock()
                .expect("inspect retained claimed-failure proposal")
                .pending_manifest
                .is_some(),
            "same-generation reconciliation must retain the exact retry proposal"
        );
        assert_public_authority_observes_reconciled_single_document(
            &index,
            &cx,
            generation,
            "alpha",
            "claimed-failure",
            0,
        )
        .await;

        drop(fault);
        assert_eq!(
            index
                .commit(&cx)
                .await
                .expect("retry retained claimed-failure proposal")
                .loaded_manifest()
                .manifest
                .generation,
            generation + 1
        );
        assert_public_authority_observes_reconciled_single_document(
            &index,
            &cx,
            generation + 1,
            "alpha",
            "claimed-failure",
            1,
        )
        .await;
    });
}

#[test]
fn reader_retries_when_publish_completes_between_authority_loads() {
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary reader-retry directory");
        let index = Arc::new(
            QuillIndex::create(&cx, directory.path(), deterministic_config())
                .await
                .expect("create durable reader-retry fixture"),
        );
        let generation = index
            .snapshot()
            .expect("fresh reader-retry snapshot")
            .loaded_manifest()
            .manifest
            .generation;
        index
            .index_document(
                &cx,
                &IndexableDocument::new("reader-retry", "alpha successor"),
            )
            .await
            .expect("stage reader-retry document");
        let pause = index.reader.arm_publication_read_pause_for_test();
        let reader_index = Arc::clone(&index);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let reader = std::thread::spawn(move || {
            sender
                .send(
                    reader_index
                        .search_snapshot()
                        .map(|snapshot| snapshot.keeper_generation()),
                )
                .expect("return reader authority result");
        });
        pause
            .wait_until_reached(PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT)
            .expect("reader reaches the authority double-read rendezvous");
        index
            .commit(&cx)
            .await
            .expect("publish while reader holds its first authority observation");
        pause.release();
        let observed = receiver
            .recv_timeout(PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT)
            .expect("reader completes after publication")
            .expect("reader must retry to one authoritative Arc");
        reader.join().expect("reader thread does not panic");
        assert_eq!(observed, generation + 1);
    });
}

#[test]
fn dropped_ordinary_commit_requires_public_reconciliation_before_reads() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary dropped-commit Keeper directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable dropped-commit index");
        let generation = index
            .snapshot()
            .expect("fresh dropped-commit snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        index
            .index_document(
                &cx,
                &IndexableDocument::new("dropped-commit", "alpha durable commit"),
            )
            .await
            .expect("stage dropped-commit document");

        let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::TempMovedToCurrent,
        );
        let mut commit = Box::pin(index.commit(&cx));
        crate::keeper::drive_manifest_publish_to_checkpoint_for_test(
            &mut commit,
            &pause,
            PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT,
        )
        .await
        .expect("reach bounded dropped-commit rendezvous");

        assert_public_authority_refuses_stale_publication(&index, &cx, "alpha", "dropped-commit")
            .await;
        drop(commit);

        assert_public_authority_refuses_stale_publication(&index, &cx, "alpha", "dropped-commit")
            .await;
        assert!(
            index.has_uncommitted_changes(),
            "a dropped ordinary commit must remain recoverable writer work"
        );
        let installed = Manifest::from_bytes(
            &std::fs::read(directory.path().join("MANIFEST"))
                .expect("read durable dropped-commit MANIFEST"),
        )
        .expect("decode durable dropped-commit MANIFEST");
        assert_eq!(installed.generation, generation + 1);

        pause.release();
        let reconciled = index
            .reconcile_publication(&cx)
            .await
            .expect("reconcile dropped ordinary commit");
        assert_eq!(reconciled.keeper_generation(), generation + 1);
        assert!(
            !index.has_uncommitted_changes(),
            "generic reconciliation must resolve the retained commit retry state"
        );
        assert_public_authority_observes_reconciled_single_document(
            &index,
            &cx,
            generation + 1,
            "alpha",
            "dropped-commit",
            1,
        )
        .await;
    });
}

#[test]
fn dropped_ordinary_delete_requires_public_reconciliation_before_reads() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary dropped-delete Keeper directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable dropped-delete index");
        index
            .index_document(
                &cx,
                &IndexableDocument::new("dropped-delete", "alpha durable delete"),
            )
            .await
            .expect("stage dropped-delete document");
        index
            .commit(&cx)
            .await
            .expect("publish dropped-delete fixture");
        let generation = index
            .snapshot()
            .expect("dropped-delete fixture snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;

        let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::DirectorySynced,
        );
        let mut delete = Box::pin(index.delete_document(&cx, "dropped-delete"));
        crate::keeper::drive_manifest_publish_to_checkpoint_for_test(
            &mut delete,
            &pause,
            PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT,
        )
        .await
        .expect("reach bounded dropped-delete rendezvous");
        drop(delete);

        assert_public_authority_refuses_stale_publication(&index, &cx, "alpha", "dropped-delete")
            .await;
        assert!(
            index.has_uncommitted_changes(),
            "a dropped ordinary delete must remain recoverable writer work"
        );
        let installed = Manifest::from_bytes(
            &std::fs::read(directory.path().join("MANIFEST"))
                .expect("read durable dropped-delete MANIFEST"),
        )
        .expect("decode durable dropped-delete MANIFEST");
        assert_eq!(installed.generation, generation + 1);

        pause.release();
        let reconciled = index
            .reconcile_publication(&cx)
            .await
            .expect("reconcile dropped ordinary delete");
        assert_eq!(reconciled.keeper_generation(), generation + 1);
        assert!(
            !index.has_uncommitted_changes(),
            "generic reconciliation must resolve the dropped-delete state"
        );
        assert_public_authority_observes_reconciled_single_document(
            &index,
            &cx,
            generation + 1,
            "alpha",
            "dropped-delete",
            0,
        )
        .await;
    });
}

#[test]
fn dropped_ordinary_compaction_requires_public_reconciliation_before_reads() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary dropped-compaction Keeper directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable dropped-compaction index");
        let documents = q1_ob2a_documents();
        index
            .index_documents(&cx, &documents[..20])
            .await
            .expect("stage dropped-compaction documents");
        index
            .commit(&cx)
            .await
            .expect("publish dropped-compaction fixture");
        let deleted_ids = (0_u32..5)
            .map(|ordinal| format!("q1-ob2a-{ordinal:03}"))
            .collect::<Vec<_>>();
        let deleted_refs = deleted_ids.iter().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(
            index
                .delete_documents(&cx, &deleted_refs)
                .await
                .expect("publish dropped-compaction tombstones"),
            deleted_refs.len()
        );
        let generation = index
            .snapshot()
            .expect("dropped-compaction source snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;

        let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::DirectorySynced,
        );
        let mut compact = Box::pin(index.compact(&cx, CompactionPolicy::default()));
        crate::keeper::drive_manifest_publish_to_checkpoint_for_test(
            &mut compact,
            &pause,
            PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT,
        )
        .await
        .expect("reach bounded dropped-compaction rendezvous");
        drop(compact);

        assert_public_authority_refuses_stale_publication(&index, &cx, "shared", "q1-ob2a-005")
            .await;
        assert!(
            index.has_uncommitted_changes(),
            "a dropped ordinary compaction must remain recoverable writer work"
        );
        let installed = Manifest::from_bytes(
            &std::fs::read(directory.path().join("MANIFEST"))
                .expect("read durable dropped-compaction MANIFEST"),
        )
        .expect("decode durable dropped-compaction MANIFEST");
        assert_eq!(installed.generation, generation + 1);

        pause.release();
        let reconciled = index
            .reconcile_publication(&cx)
            .await
            .expect("reconcile dropped ordinary compaction");
        assert_eq!(reconciled.keeper_generation(), generation + 1);
        assert!(
            !index.has_uncommitted_changes(),
            "generic reconciliation must resolve the dropped-compaction state"
        );
        assert_eq!(
            index
                .snapshot()
                .expect("reconciled compaction snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation + 1
        );
        assert_eq!(
            index
                .doc_count()
                .expect("reconciled compaction document count is authoritative"),
            15
        );
        assert_eq!(
            index
                .search_paginated(&cx, "shared", 500, 0, true)
                .expect("reconciled compaction search")
                .total_count,
            Some(15)
        );
    });
}
