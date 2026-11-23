use super::*;

#[test]
fn manifest_watermark_opens_exactly_inside_the_q1_address_space() {
    let lease_size = u64::from(DOC_ORDS_PER_LEASE);
    for watermark in [0, 1, lease_size, lease_size + 1, MAX_GLOBAL_DOCID_EXCLUSIVE] {
        assert_eq!(
            validate_docid_watermark(watermark).expect("in-range watermark"),
            watermark
        );
    }
    assert!(validate_docid_watermark(MAX_GLOBAL_DOCID_EXCLUSIVE + 1).is_err());
}

#[test]
fn field_stat_overflow_fails_before_segment_installation() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        index
            .writer_mut()
            .pending_field_stats
            .insert(ID_FIELD, (u64::MAX, 0));
        index
            .index_documents(&cx, &[IndexableDocument::new("overflow", "one token")])
            .await
            .expect("accumulate document");

        let Err(error) = index.commit(&cx).await else {
            panic!("field stats overflow unexpectedly committed"); // ubs:ignore -- Test rejects overflowing statistics.
        };
        assert!(matches!(error, QuillIndexError::InvalidState { .. }));
        let writer = index.writer_mut();
        assert_eq!(writer.shards[0].accumulator.document_count(), 1);
        assert!(writer.staged_flush.is_none());
        assert!(writer.pending_segments.is_empty());
        assert!(writer.pending_owned_segments.is_empty());
    });
}

#[cfg(unix)]
#[test]
fn filesystem_manifest_retry_reuses_stamped_bytes_after_clock_advances() {
    let directory = tempfile::tempdir().expect("index tempdir").keep();
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::create(&cx, &directory, deterministic_config())
            .await
            .expect("create filesystem index");
        index
            .index_documents(&cx, &fixture_documents())
            .await
            .expect("accumulate fixture");
        index
            .writer_mut()
            .flush_shard(&cx, 0, LifecycleTrigger::ExplicitFlush)
            .await
            .expect("install immutable segment");
        index
            .writer_mut()
            .prepare_pending_manifest()
            .expect("retain exact MANIFEST proposal");
        let proposal = index
            .writer_mut()
            .pending_manifest
            .as_ref()
            .expect("proposal retained")
            .clone();
        assert!(proposal.last_publish_unix_s > 0);
        let proposal_bytes = proposal.to_bytes().expect("encode retained proposal");
        let temp_path = directory.join(format!(".tmp-manifest-{}", proposal.generation));
        std::fs::write(&temp_path, &proposal_bytes).expect("seed synced-attempt image");

        std::thread::sleep(std::time::Duration::from_millis(1_100));
        index
            .commit(&cx)
            .await
            .expect("reuse exact prior-attempt bytes");

        assert_eq!(
            index
                .snapshot()
                .expect("reused-manifest snapshot is authoritative")
                .loaded_manifest()
                .manifest,
            proposal
        );
        assert!(index.writer_mut().pending_manifest.is_none());
    });
}

#[cfg(unix)]
#[test]
fn restarted_ingest_avoids_differing_abandoned_segment_id() {
    let directory = tempfile::tempdir().expect("index tempdir").keep();
    run_with_cx(|cx| async move {
        let abandoned_segment_id = {
            let mut first = QuillIndex::create(&cx, &directory, deterministic_config())
                .await
                .expect("create first writer");
            first
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("abandoned", "old orphan content")],
                )
                .await
                .expect("accumulate abandoned batch");
            first
                .writer_mut()
                .flush_shard(&cx, 0, LifecycleTrigger::ExplicitFlush)
                .await
                .expect("install segment without MANIFEST");
            assert_eq!(
                first
                    .snapshot()
                    .expect("unpublished first snapshot is authoritative")
                    .doc_count(),
                0
            );
            first.writer_mut().pending_segments[0].segment_id
        };
        let abandoned_path = directory.join(format!("seg-{abandoned_segment_id:016x}.fslx"));
        assert!(
            abandoned_path.exists(),
            "orphan must survive writer restart"
        );

        let restarted = QuillIndex::open(&cx, &directory, deterministic_config())
            .await
            .expect("reopen old committed generation");
        restarted
            .index_documents(
                &cx,
                &[IndexableDocument::new(
                    "replacement",
                    "fresh replacement content",
                )],
            )
            .await
            .expect("accumulate replacement batch");
        restarted
            .commit(&cx)
            .await
            .expect("publish replacement batch");

        let restarted_snapshot = restarted
            .snapshot()
            .expect("restarted snapshot is authoritative");
        let committed = &restarted_snapshot.loaded_manifest().manifest;
        assert_eq!(committed.segments.len(), 1);
        assert_ne!(committed.segments[0].segment_id, abandoned_segment_id);
        assert!(
            abandoned_path.exists(),
            "grace-window orphan remains intact"
        );
        let replacement = restarted
            .search_paginated(&cx, "replacement", 10, 0, true)
            .expect("search replacement batch");
        assert_eq!(replacement.total_count, Some(1));
        assert_eq!(replacement.hits[0].document_id, "replacement");
        let abandoned = restarted
            .search_paginated(&cx, "abandoned", 10, 0, true)
            .expect("search committed snapshot only");
        assert_eq!(abandoned.total_count, Some(0));
    });
}

/// bd-8nqz.1 flagship: a retained candidate batch hydrates from the
/// EXACT snapshot that scored it, even after a newer generation commits —
/// while the legacy combined-trait path demonstrably reads the newer
/// generation (the race the split exists to fix). Also pins the typed
/// rejection of missing and foreign hydration contexts.
#[test]
fn lexical_read_hydration_is_pinned_to_the_scoring_generation() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(QuillConfig::default()).expect("in-memory index");
        let doc_v1 =
            IndexableDocument::new("doc-a", "alpha pinned content").with_metadata("rev", "v1");
        index
            .index_documents(&cx, std::slice::from_ref(&doc_v1))
            .await
            .expect("index generation N");
        index.commit(&cx).await.expect("publish generation N");

        // Score on generation N; candidates defer metadata.
        let batch = LexicalRead::search_candidates(&index, &cx, "pinned", 10)
            .await
            .expect("candidates on generation N");
        assert!(batch.is_deferred(), "Quill candidates must carry a pin");
        assert_eq!(batch.results().len(), 1);
        assert!(
            batch.results()[0].metadata.is_none(),
            "candidate metadata is deferred until hydration"
        );
        let mut legacy_path = batch.results().to_vec();

        // Publish generation N+1 with different metadata for the same doc.
        let doc_v2 =
            IndexableDocument::new("doc-a", "alpha pinned content").with_metadata("rev", "v2");
        LexicalWrite::index_document(&index, &cx, &doc_v2)
            .await
            .expect("upsert generation N+1");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("publish generation N+1");

        let rev_of = |result: &ScoredResult| -> Option<String> {
            result.metadata.as_ref().and_then(|metadata| {
                metadata
                    .get("rev")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
            })
        };

        // A fresh search observes N+1.
        let fresh = LexicalRead::search(&index, &cx, "pinned", 10)
            .await
            .expect("fresh search on N+1");
        assert_eq!(rev_of(&fresh[0]).as_deref(), Some("v2"));

        // The retained batch hydrates from its PINNED generation N.
        let (mut winners, context) = batch.into_parts();
        LexicalRead::hydrate_candidates(&index, &cx, context.as_ref(), &mut winners)
            .await
            .expect("pinned hydration");
        assert_eq!(
            rev_of(&winners[0]).as_deref(),
            Some("v1"),
            "hydration must read the scoring snapshot, not the newest one"
        );

        // Differential control. This used to hydrate through the combined
        // trait, which read whatever snapshot was current and so returned
        // v2. That trait is gone, but the demonstration must not go with
        // it — otherwise this test would only assert the happy path and
        // would still pass if pinning silently stopped working.
        //
        // The replacement is stricter, because it varies ONLY the pin: the
        // same retained generation-N candidates are hydrated against a pin
        // taken at N+1. Getting v2 proves the snapshot the context carries
        // is what decides the answer, which is precisely the property the
        // split buys.
        let (_fresh_candidates, newer_pin) =
            LexicalRead::search_candidates(&index, &cx, "pinned", 10)
                .await
                .expect("candidates on N+1")
                .into_parts();
        LexicalRead::hydrate_candidates(&index, &cx, newer_pin.as_ref(), &mut legacy_path)
            .await
            .expect("hydration against a newer pin");
        assert_eq!(
            rev_of(&legacy_path[0]).as_deref(),
            Some("v2"),
            "a newer pin must surface the newer generation — the race the split fixes"
        );

        // Missing context: typed rejection, never a silent no-op.
        let missing = LexicalRead::hydrate_candidates(&index, &cx, None, &mut winners)
            .await
            .expect_err("deferred hydration without a context must fail");
        assert!(
            matches!(missing, SearchError::SubsystemError { subsystem, .. }
                if subsystem == "quill.hydration"),
        );

        // Foreign context: typed rejection of cross-engine mixing.
        let foreign = LexicalHydrationContext::new("not-quill", Box::new(7_u32));
        let mixed = LexicalRead::hydrate_candidates(&index, &cx, Some(&foreign), &mut winners)
            .await
            .expect_err("foreign hydration context must fail");
        assert!(
            matches!(mixed, SearchError::SubsystemError { subsystem, .. }
                if subsystem == "quill.hydration"),
        );
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn qg4_directory_sync_receipt_binds_real_commit_and_fresh_reopen() {
    let directory = tempfile::tempdir().expect("temporary QG-4 directory");
    let root = directory.path().to_path_buf();
    run_with_cx(|cx| async move {
        let index = QuillIndex::create(&cx, &root, deterministic_config())
            .await
            .expect("create QG-4 Quill index");
        index
            .index_document(&cx, &IndexableDocument::new("warm", "warm QG-4 document"))
            .await
            .expect("stage warm QG-4 document");
        index.commit(&cx).await.expect("commit warm QG-4 document");

        index
            .index_document(
                &cx,
                &IndexableDocument::new("qg4-staged", "staged QG-4 document"),
            )
            .await
            .expect("stage QG-4 probe document");
        let mut arm = index
            .benchmark_arm_qg4_directory_sync()
            .expect("arm QG-4 Quill directory sync");
        index
            .benchmark_commit_qg4_directory_sync(&cx, &mut arm)
            .await
            .expect("commit staged QG-4 probe");
        let receipt = arm.finish().expect("one real MANIFEST directory sync");
        assert_eq!(receipt.root, root);
        assert_ne!(receipt.session_nonce, 0);
        assert_eq!(receipt.observed_sync_count, 1);
        assert_eq!(
            receipt.manifest_generation,
            index
                .snapshot()
                .expect("committed QG-4 snapshot")
                .loaded_manifest()
                .manifest
                .generation,
            "receipt generation must be the generation installed by the public commit"
        );

        assert!(matches!(
            arm.finish(),
            Err(QuillIndexError::Config(SearchError::InvalidConfig { value, .. }))
                if value == "foreign_or_disarmed_session"
        ));

        drop(index);
        let reopened = QuillIndex::open(&cx, &root, deterministic_config())
            .await
            .expect("fresh reopen after QG-4 commit");
        assert_eq!(reopened.doc_count().expect("reopened document count"), 2);
        let staged = reopened
            .search_doc_ids(&cx, "id:qg4-staged", 2)
            .expect("exact staged-id query after fresh reopen");
        assert_eq!(staged.len(), 1, "fresh reopen must retain staged ID");
        assert_eq!(staged[0].document_id, "qg4-staged");
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn qg4_directory_sync_receipt_rejects_zero_and_multiple_real_syncs() {
    let zero_directory = tempfile::tempdir().expect("temporary zero-sync QG-4 directory");
    run_with_cx(|cx| async move {
        let index = QuillIndex::create(&cx, zero_directory.path(), deterministic_config())
            .await
            .expect("create zero-sync QG-4 index");
        let mut arm = index
            .benchmark_arm_qg4_directory_sync()
            .expect("arm zero-sync QG-4 observation");
        index
            .benchmark_commit_qg4_directory_sync(&cx, &mut arm)
            .await
            .expect("empty public commit");
        assert!(matches!(
            arm.finish(),
            Err(QuillIndexError::Config(SearchError::InvalidConfig { value, .. }))
                if value == "zero_syncs"
        ));
    });

    let multiple_directory = tempfile::tempdir().expect("temporary multiple-sync QG-4 directory");
    run_with_cx(|cx| async move {
        let index = QuillIndex::create(&cx, multiple_directory.path(), deterministic_config())
            .await
            .expect("create multiple-sync QG-4 index");
        index
            .index_document(&cx, &IndexableDocument::new("first", "first QG-4 document"))
            .await
            .expect("stage first QG-4 document");
        let mut arm = index
            .benchmark_arm_qg4_directory_sync()
            .expect("arm multiple-sync QG-4 observation");
        index
            .benchmark_commit_qg4_directory_sync(&cx, &mut arm)
            .await
            .expect("first public QG-4 commit");
        index
            .index_document(
                &cx,
                &IndexableDocument::new("second", "second QG-4 document"),
            )
            .await
            .expect("stage second QG-4 document");
        index.commit(&cx).await.expect("second real public commit");
        assert!(matches!(
            arm.finish(),
            Err(QuillIndexError::Config(SearchError::InvalidConfig { value, .. }))
                if value == "multiple_syncs"
        ));
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn qg4_directory_sync_receipt_rejects_foreign_and_failed_after_sync_commits() {
    let first_directory = tempfile::tempdir().expect("temporary first QG-4 directory");
    let second_directory = tempfile::tempdir().expect("temporary second QG-4 directory");
    run_with_cx(|cx| async move {
        let first = QuillIndex::create(&cx, first_directory.path(), deterministic_config())
            .await
            .expect("create first QG-4 index");
        let second = QuillIndex::create(&cx, second_directory.path(), deterministic_config())
            .await
            .expect("create second QG-4 index");
        let mut foreign_arm = first
            .benchmark_arm_qg4_directory_sync()
            .expect("arm first QG-4 index");
        assert!(matches!(
            second
                .benchmark_commit_qg4_directory_sync(&cx, &mut foreign_arm)
                .await,
            Err(QuillIndexError::Config(SearchError::InvalidConfig { value, .. }))
                if value == "foreign_arm"
        ));
        assert!(matches!(
            foreign_arm.finish(),
            Err(QuillIndexError::Config(SearchError::InvalidConfig { value, .. }))
                if value == "zero_syncs"
        ));

        first
            .index_document(
                &cx,
                &IndexableDocument::new("failed", "failed-after-sync QG-4 document"),
            )
            .await
            .expect("stage failed-after-sync QG-4 document");
        let mut failed_arm = first
            .benchmark_arm_qg4_directory_sync()
            .expect("arm failed-after-sync QG-4 observation");
        failed_arm.fail_after_commit = true;
        let failed_commit = first
            .benchmark_commit_qg4_directory_sync(&cx, &mut failed_arm)
            .await;
        assert!(
            matches!(
                &failed_commit,
                Err(QuillIndexError::Config(SearchError::InvalidConfig { value, .. }))
                    if value == "test_failure_after_commit"
            ),
            "unexpected post-directory-sync failure: {failed_commit:?}"
        );
        assert!(matches!(
            failed_arm.finish(),
            Err(QuillIndexError::Config(SearchError::InvalidConfig { value, .. }))
                if value == "commit_failed_after_sync"
        ));
    });
}
