use super::*;

#[test]
fn installed_delta_segment_without_manifest_is_not_durable_visibility() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary Keeper directory");
        let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable index");
        let generation = index
            .snapshot()
            .expect("installed-segment snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let mut delta =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("uncommitted Delta");
        apply_alpha_delta(&mut delta, 0, "visible-not-durable", 1);
        let frozen = Arc::new(delta.freeze(generation));
        index
            .publish_delta_table(vec![Arc::clone(&frozen)])
            .expect("publish process-local Delta");
        let encoded = flush_delta_snapshot(
            &frozen,
            DeltaFlushInput {
                segment_id: 0xe5_4020,
                created_unix_s: 1_700_000_020,
                engine_version: CURRENT_ENGINE_VERSION,
            },
        )
        .expect("build uncommitted segment")
        .expect("live Delta emits a segment");
        let writer = match &mut index.writer_mut().backend {
            IndexBackend::Durable(writer) => writer,
            IndexBackend::Memory(_) => panic!("fixture must use a durable Keeper"), // ubs:ignore -- Test fixture requires a durable backend.
        };
        let pending = encoded
            .write_temp_retryable(directory.path())
            .expect("sync retryable segment");
        writer
            .publish_segment(&cx, pending)
            .await
            .expect("install unreferenced segment");

        assert_eq!(
            index
                .search_snapshot()
                .expect("process-local Delta snapshot is authoritative")
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("process-local alpha frequency"),
            1,
            "the writer process sees its published Delta before commit"
        );
        let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
            .expect("read old authoritative MANIFEST");
        assert_eq!(reopened.doc_count(), 0);
        assert!(reopened.segments().is_empty());
        assert_eq!(reopened.materialize_document_id(0), None);
    });
}

#[test]
fn retained_delta_seal_reconciles_an_ambiguously_installed_segment_without_a_new_temp() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary Keeper directory");
        let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable index");
        let generation = index
            .snapshot()
            .expect("retained-seal snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let mut delta =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("retained Delta source");
        apply_alpha_delta(&mut delta, 0, "retained", 1);
        let sealed = Arc::new(delta.freeze(generation));
        index
            .publish_delta_table(vec![Arc::clone(&sealed)])
            .expect("publish retained Delta source");
        let before = index
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("query pre-seal Delta");

        let encoded = Arc::new(
            flush_delta_snapshot(
                &sealed,
                DeltaFlushInput {
                    segment_id: 0xe5_4030,
                    created_unix_s: 1_700_000_030,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .expect("build retained seal")
            .expect("live Delta emits FSLX"),
        );
        let mut manifest = index
            .snapshot()
            .expect("retained-seal snapshot is authoritative")
            .next_manifest()
            .expect("successor MANIFEST");
        manifest.last_publish_unix_s = 0;
        manifest.segments.push(manifest_segment(&encoded, 1));
        manifest.docid_high_watermark = sealed.lease_end();
        let mut pending_field_stats = BTreeMap::new();
        for field in DEFAULT_SCHEMA
            .fields
            .iter()
            .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        {
            pending_field_stats.insert(
                field.id,
                (
                    sealed
                        .live_total_tokens(field.id)
                        .expect("retained Delta field stats"),
                    1,
                ),
            );
        }
        manifest.field_stats = merge_field_stats(&manifest.field_stats, &pending_field_stats)
            .expect("merge retained Delta field stats");
        let prepared = index
            .reader
            .published_snapshot
            .prepare_sealed_manifest(DEFAULT_SCHEMA, &manifest)
            .expect("prepare retained local swap");
        index.writer_mut().pending_delta_seal = Some(PendingDeltaSeal {
            encoded: Some(Arc::clone(&encoded)),
            segment_installed: false,
            manifest,
            prepared,
            next_seal_seq: 2,
            successor_watermark: sealed.lease_end(),
        });

        let writer = match &mut index.writer_mut().backend {
            IndexBackend::Durable(writer) => writer,
            IndexBackend::Memory(_) => panic!("fixture must use a durable Keeper"), // ubs:ignore -- Test fixture requires a durable backend.
        };
        writer
            .publish_encoded_segment_retryable(&cx, Arc::clone(&encoded))
            .await
            .expect("simulate install whose caller future lost completion");
        assert_eq!(
            index
                .snapshot()
                .expect("pre-reconciliation snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation,
            "segment installation alone must not advance durable authority"
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("old Delta remains visible while reconciliation is pending"),
            before
        );

        index
            .resume_pending_delta_seal(&cx)
            .await
            .expect("reconcile exact canonical segment and publish retained MANIFEST");
        assert!(index.writer_mut().pending_delta_seal.is_none());
        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("query reconciled Keeper"),
            before
        );
        assert!(
            index
                .snapshot()
                .expect("reconciled retained-seal snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .last_publish_unix_s
                > 0,
            "Keeper must stamp the retained zero-timestamp proposal exactly once"
        );
        assert!(
            !std::fs::read_dir(directory.path())
                .expect("inspect retained-seal directory")
                .filter_map(Result::ok)
                .any(|entry| entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".tmp-segment-")),
            "exact canonical reconciliation must not manufacture a redundant retry temp"
        );
    });
}

/// Pre-publication cancellation: the cancel gate immediately before
/// MANIFEST authority changes aborts the attempt AFTER the retained
/// proposal was cloned for owned publication but BEFORE
/// `publish_owned_segments` runs. This pins retention across an aborted
/// attempt; it is NOT a publisher failure — see
/// `durable_delta_seal_retains_pending_on_manifest_write_failure_then_retries`
/// for a real typed publisher error.
#[test]
fn memory_delta_seal_retains_pending_on_prepublication_cancellation_then_retries() {
    run_with_cx(|cx| async move {
        let mut index =
            QuillIndex::in_memory(deterministic_config()).expect("create in-memory index");
        let generation = index
            .snapshot()
            .expect("prepublication fixture snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let base_seal_seq = index.writer_mut().next_seal_seq;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
            .expect("in-memory retry-test Delta source");
        apply_alpha_delta(&mut delta, 0, "retry-survivor", 1);
        let sealed = Arc::new(delta.freeze(generation));
        index
            .publish_delta_table(vec![Arc::clone(&sealed)])
            .expect("publish retry-test Delta");
        let before = index
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("query pre-seal Delta");

        let encoded = Arc::new(
            flush_delta_snapshot(
                &sealed,
                DeltaFlushInput {
                    segment_id: 0x51c1_0001,
                    created_unix_s: 1_700_000_050,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .expect("build retained seal")
            .expect("live Delta emits FSLX"),
        );
        let mut manifest = index
            .snapshot()
            .expect("prepublication fixture snapshot is authoritative")
            .next_manifest()
            .expect("successor MANIFEST");
        manifest.last_publish_unix_s = 0;
        manifest
            .segments
            .push(manifest_segment(&encoded, base_seal_seq));
        manifest.docid_high_watermark = sealed.lease_end();
        let mut pending_field_stats = BTreeMap::new();
        for field in DEFAULT_SCHEMA
            .fields
            .iter()
            .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        {
            pending_field_stats.insert(
                field.id,
                (
                    sealed
                        .live_total_tokens(field.id)
                        .expect("retained Delta field stats"),
                    1,
                ),
            );
        }
        manifest.field_stats = merge_field_stats(&manifest.field_stats, &pending_field_stats)
            .expect("merge retained Delta field stats");
        let prepared = index
            .reader
            .published_snapshot
            .prepare_sealed_manifest(DEFAULT_SCHEMA, &manifest)
            .expect("prepare retained local swap");
        index.writer_mut().pending_delta_seal = Some(PendingDeltaSeal {
            encoded: Some(Arc::clone(&encoded)),
            segment_installed: false,
            manifest,
            prepared,
            next_seal_seq: base_seal_seq + 1,
            successor_watermark: sealed.lease_end(),
        });

        // Pre-publication cancellation: `check_cancel(cx, "Delta
        // MANIFEST publish")` fires after the memory-owned clones ran
        // but before the publisher executes.
        let cancelled = Cx::for_testing();
        cancelled.set_cancel_requested(true);
        let failure = index.resume_pending_delta_seal(&cancelled).await;
        assert!(failure.is_err(), "cancelled attempt must fail");
        let retained_encoded = index
            .writer_mut()
            .pending_delta_seal
            .as_ref()
            .expect("aborted attempt must retain the pending seal")
            .encoded
            .as_ref()
            .map(Arc::clone)
            .expect("retained seal keeps its encoded segment");
        assert!(
            Arc::ptr_eq(&retained_encoded, &encoded),
            "the retained seal must still reference the exact encoded bytes"
        );
        drop(retained_encoded);
        assert_eq!(
            Arc::strong_count(&encoded),
            2,
            "only the test handle and the retained seal may hold the segment"
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("old epoch remains visible after the aborted attempt"),
            before
        );

        index
            .resume_pending_delta_seal(&cx)
            .await
            .expect("retried publication succeeds with no bytes lost");
        assert!(index.writer_mut().pending_delta_seal.is_none());
        assert_eq!(
            Arc::strong_count(&encoded),
            1,
            "publication must consume the retained reference"
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("query the published successor"),
            before
        );
    });
}

/// A REAL publisher failure: an exact MANIFEST temp-file collision returns
/// a typed error after the segment was already installed. The retained seal
/// must survive the failed publication attempt, and a retry must publish
/// losslessly once the conflicting artifact is preserved aside.
#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
#[test]
fn durable_delta_seal_retains_pending_on_manifest_write_failure_then_retries() {
    run_with_cx(|cx| async move {
        const INJECTED_MANIFEST_TEMP_BYTES: &[u8] = b"injected conflicting MANIFEST bytes";

        let directory = tempfile::tempdir().expect("temporary Keeper directory");
        let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable index");
        let generation = index
            .snapshot()
            .expect("manifest-failure fixture snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let base_seal_seq = index.writer_mut().next_seal_seq;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
            .expect("manifest-failure Delta source");
        apply_alpha_delta(&mut delta, 0, "manifest-failure-survivor", 1);
        let sealed = Arc::new(delta.freeze(generation));
        index
            .publish_delta_table(vec![Arc::clone(&sealed)])
            .expect("publish manifest-failure Delta");
        let before = index
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("query pre-seal Delta");

        let encoded = Arc::new(
            flush_delta_snapshot(
                &sealed,
                DeltaFlushInput {
                    segment_id: 0x51c1_0002,
                    created_unix_s: 1_700_000_051,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .expect("build retained seal")
            .expect("live Delta emits FSLX"),
        );
        let mut manifest = index
            .snapshot()
            .expect("manifest-failure fixture snapshot is authoritative")
            .next_manifest()
            .expect("successor MANIFEST");
        manifest.last_publish_unix_s = 0;
        manifest
            .segments
            .push(manifest_segment(&encoded, base_seal_seq));
        manifest.docid_high_watermark = sealed.lease_end();
        let mut pending_field_stats = BTreeMap::new();
        for field in DEFAULT_SCHEMA
            .fields
            .iter()
            .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        {
            pending_field_stats.insert(
                field.id,
                (
                    sealed
                        .live_total_tokens(field.id)
                        .expect("retained Delta field stats"),
                    1,
                ),
            );
        }
        manifest.field_stats = merge_field_stats(&manifest.field_stats, &pending_field_stats)
            .expect("merge retained Delta field stats");
        let prepared = index
            .reader
            .published_snapshot
            .prepare_sealed_manifest(DEFAULT_SCHEMA, &manifest)
            .expect("prepare retained local swap");
        index.writer_mut().pending_delta_seal = Some(PendingDeltaSeal {
            encoded: Some(Arc::clone(&encoded)),
            segment_installed: false,
            manifest,
            prepared,
            next_seal_seq: base_seal_seq + 1,
            successor_watermark: sealed.lease_end(),
        });

        // Install the segment for real while the directory is writable,
        // exactly like a publication whose caller future lost completion
        // after the install step.
        {
            let writer = match &mut index.writer_mut().backend {
                IndexBackend::Durable(writer) => Some(writer),
                IndexBackend::Memory(_) => None,
            }
            .expect("fixture must use a durable Keeper");
            writer
                .publish_encoded_segment_retryable(&cx, Arc::clone(&encoded))
                .await
                .expect("install segment before the injected MANIFEST failure");
        }

        // Injected REAL publisher failure: a mismatched file at the exact
        // temp path cannot be reused as the pending generation's MANIFEST.
        // Unlike permission bits, this remains effective when the worker
        // has root or another write-bypassing capability, and unlike a
        // directory collision it has the same typed result cross-platform.
        let pending_generation = generation
            .checked_add(1)
            .expect("fixture generation has a successor");
        let collision_path = directory
            .path()
            .join(format!(".tmp-manifest-{pending_generation}"));
        std::fs::write(&collision_path, INJECTED_MANIFEST_TEMP_BYTES)
            .expect("create exact MANIFEST temp-file collision");
        let failure = index
            .resume_pending_delta_seal(&cx)
            .await
            .err()
            .expect("MANIFEST temp-path collision must fail publication");
        assert!(
            matches!(
                &failure,
                QuillIndexError::Keeper(KeeperError::TempConflict { .. })
            ),
            "failure must be the injected typed temp conflict, got: {failure:?}"
        );
        if let QuillIndexError::Keeper(KeeperError::TempConflict { path }) = &failure {
            // ubs:ignore -- these are public temporary filesystem paths, not secrets.
            assert_eq!(path, &collision_path);
        }
        assert_eq!(
            index
                .snapshot()
                .expect("failed manifest publication snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation,
            "failed MANIFEST publication must not advance durable authority"
        );
        {
            let retained = index
                .writer_mut()
                .pending_delta_seal
                .as_ref()
                .expect("failed publication must retain the pending seal");
            assert!(
                retained.segment_installed,
                "the reconciled install must be recorded before the failed publish"
            );
            assert!(
                Arc::ptr_eq(
                    retained.encoded.as_ref().expect("retained encoded segment"),
                    &encoded
                ),
                "the retained seal must still reference the exact encoded bytes"
            );
        }
        let preserved_collision = directory.path().join(format!(
            ".injected-manifest-temp-conflict-{pending_generation}"
        ));
        std::fs::rename(&collision_path, &preserved_collision)
            .expect("preserve injected collision aside before retry");
        assert!(
            preserved_collision.is_file(),
            "the injected failure artifact must remain preserved for diagnosis"
        );
        // ubs:ignore -- this public test sentinel is not credential material.
        assert_eq!(
            std::fs::read(&preserved_collision)
                .expect("read preserved MANIFEST temp-file collision"),
            INJECTED_MANIFEST_TEMP_BYTES,
            "the failed reusable-temp probe must not overwrite the collision"
        );
        assert_eq!(
            Arc::strong_count(&encoded),
            2,
            "only the test handle and the retained seal may hold the segment"
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("old epoch remains visible after the failed publication"),
            before
        );

        index
            .resume_pending_delta_seal(&cx)
            .await
            .expect("retried publication succeeds with no bytes lost");
        assert!(index.writer_mut().pending_delta_seal.is_none());
        assert_eq!(
            Arc::strong_count(&encoded),
            1,
            "publication must consume the retained reference"
        );
        assert_eq!(
            index
                .snapshot()
                .expect("retried publication snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation + 1,
            "retried publication must advance exactly one generation"
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("query the published successor"),
            before
        );
    });
}

#[test]
fn published_memory_snapshot_bytes_survive_successor_publication() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("create in-memory index");
        index
            .index_document(
                &cx,
                &IndexableDocument::new("doc-a", "aardvark burrows deep"),
            )
            .await
            .expect("index first document");
        index.commit(&cx).await.expect("first commit publishes");
        let old = index
            .snapshot()
            .expect("first published snapshot is authoritative");
        for segment in old.segments() {
            segment.verify().expect("fresh snapshot verifies");
        }
        let old_segment_count = old.segments().len();
        let old_backing: Vec<*const u8> = old
            .segments()
            .iter()
            .map(|segment| segment.source_bytes().as_ptr())
            .collect();
        let before = index
            .search_paginated(&cx, "aardvark", 10, 0, true)
            .expect("query first publication");

        index
            .index_document(
                &cx,
                &IndexableDocument::new("doc-b", "bobcat prowls quietly"),
            )
            .await
            .expect("index second document");
        index.commit(&cx).await.expect("second commit publishes");

        let new = index
            .snapshot()
            .expect("successor published snapshot is authoritative");
        assert!(
            !Arc::ptr_eq(&old, &new),
            "a successor publication must install a new snapshot"
        );
        assert!(new.segments().len() > old_segment_count);
        // The pre-publication snapshot still owns valid, hash-verified
        // bytes: successor publication and writer-side pending-state
        // clearing must not disturb the shared backing.
        for segment in old.segments() {
            segment
                .verify()
                .expect("pre-publication snapshot backing must stay intact");
        }
        assert_eq!(old.segments().len(), old_segment_count);
        let old_backing_after: Vec<*const u8> = old
            .segments()
            .iter()
            .map(|segment| segment.source_bytes().as_ptr())
            .collect();
        assert_eq!(
            old_backing, old_backing_after,
            "the old snapshot must keep its exact byte backing"
        );
        // Corpus statistics legitimately change with the second commit,
        // so compare identity, not scores.
        let after = index
            .search_paginated(&cx, "aardvark", 10, 0, true)
            .expect("first document remains visible");
        assert_eq!(after.total_count, Some(1));
        assert_eq!(after.hits.len(), 1);
        assert_eq!(after.hits[0].document_id, before.hits[0].document_id);
        assert_eq!(
            index
                .search_paginated(&cx, "bobcat", 10, 0, true)
                .expect("second document becomes visible")
                .total_count,
            Some(1)
        );
    });
}

/// The ordinary scalar commit publishes by sharing the pending payload
/// backing, and a publication that fails after the flush sealed that
/// payload must retain it — unmoved — for the retry.
///
/// Both halves are pinned by allocation identity rather than by content,
/// which is what makes this a regression test instead of a restatement of
/// the code: while `EncodedSegment` owned its bytes outright, publication
/// consumed a deep copy of `pending_owned_segments`, so the installed
/// segment's image lived at a different address than the payload the
/// writer retained for retry. Any future design that copies the payload,
/// or that moves the pending inventory out before publication succeeds,
/// fails here.
#[cfg(feature = "conformance-internals")]
#[test]
fn memory_commit_publication_failure_retains_the_exact_pending_payload_then_retries() {
    run_with_cx(|cx| async move {
        let mut index =
            QuillIndex::in_memory(deterministic_config()).expect("create in-memory index");
        index
            .index_document(
                &cx,
                &IndexableDocument::new("doc-a", "aardvark burrows deep"),
            )
            .await
            .expect("stage the commit-retry document");
        let generation = index
            .snapshot()
            .expect("pending-payload fixture snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;

        // Inject the failure at the exact publication boundary: the shard
        // flush has already sealed the encoded segment into the writer's
        // pending inventory and the successor MANIFEST is prepared, but
        // the Keeper has adopted nothing yet.
        let controller = index.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::CommitPublication, 1)
            .expect("arm the commit publication checkpoint");
        // `KeeperSnapshot` is deliberately not `Debug`, so the outcome is
        // matched in place rather than unwrapped.
        assert!(
            matches!(
                index.commit(&cx).await,
                Err(QuillIndexError::Cancelled {
                    phase: "commit publish"
                })
            ),
            "the injected failure must abort the commit at publication",
        );
        assert!(controller.fired());
        assert_eq!(controller.observed_checkpoints(), 1);
        controller.disarm();
        cx.set_cancel_requested(false);

        let (payload_ptr, payload, segment_id) = {
            let pending = &index.writer_mut().pending_owned_segments;
            assert_eq!(
                pending.len(),
                1,
                "the failed publication must retain the sealed segment for retry"
            );
            let retained = &pending[0];
            (
                retained.as_bytes().as_ptr(),
                retained.as_bytes().to_vec(),
                retained.header().segment_id,
            )
        };
        assert_eq!(
            index
                .snapshot()
                .expect("aborted pending-payload snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation,
            "an aborted publication must not advance the published generation"
        );
        assert!(
            index
                .search_paginated(&cx, "aardvark", 10, 0, true)
                .expect("query the unchanged epoch")
                .hits
                .is_empty(),
            "the retained segment must stay invisible until publication succeeds"
        );

        let published = index
            .commit(&cx)
            .await
            .expect("the retry publishes the retained segment");
        assert!(
            index.writer_mut().pending_owned_segments.is_empty(),
            "a successful publication must consume the pending inventory"
        );
        let segment = published
            .segments()
            .iter()
            .find(|segment| segment.header().segment_id == segment_id)
            .expect("the retry installs the exact retained segment");
        assert_eq!(
            segment.source_bytes().as_ptr(),
            payload_ptr,
            "publication must adopt the retained payload allocation, never copy it"
        );
        assert_eq!(
            segment.source_bytes(),
            payload.as_slice(),
            "the retried publication must seal byte-identical FSLX"
        );
        segment.verify().expect("the published segment verifies");
        let hits = index
            .search_paginated(&cx, "aardvark", 10, 0, true)
            .expect("query the retried publication");
        assert_eq!(hits.total_count, Some(1));
        assert_eq!(hits.hits[0].document_id, "doc-a");
    });
}

/// A REAL publisher rejection — raised inside `publish_owned_segments`,
/// after the pending inventory has already been handed to it — must leave
/// the writer's own copy of the payload retained at the exact same
/// allocation, and the repaired retry must publish that allocation.
///
/// The sibling prepublication test cannot see this, which is why both
/// exist: it aborts at the checkpoint before the publisher is ever called,
/// so a refactor that moves `pending_owned_segments` into the call —
/// `mem::take` and friends — would still pass it while silently losing the
/// sealed bytes on any publisher error. The failure injected here is the
/// publisher's own typed duplicate-segment rejection, reachable only after
/// ownership of the inventory has transferred, so an emptied pending
/// inventory fails the very next assertion.
#[test]
fn memory_publisher_error_retains_the_exact_pending_payload_then_retries() {
    run_with_cx(|cx| async move {
        let mut index =
            QuillIndex::in_memory(deterministic_config()).expect("create in-memory index");
        index
            .index_document(
                &cx,
                &IndexableDocument::new("doc-a", "aardvark burrows deep"),
            )
            .await
            .expect("stage the publisher-failure document");
        let generation = index
            .snapshot()
            .expect("publisher-retry fixture snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;

        // Seal without publishing. The encoded segment now sits in the
        // writer's pending inventory exactly as an interrupted commit
        // leaves it, with no MANIFEST proposal prepared yet.
        index
            .writer_mut()
            .flush_all_shards(&cx, LifecycleTrigger::ExplicitFlush)
            .await
            .expect("seal the pending segment without publishing it");

        // Arm the publisher failure: one segment supplied twice is rejected
        // by `publish_owned_segments` itself. Cloning the retained segment
        // shares its backing, so the injected duplicate adds no payload.
        let (payload_ptr, payload, segment_id) = {
            let writer = index.writer_mut();
            assert_eq!(
                writer.pending_owned_segments.len(),
                1,
                "the flush must seal exactly one pending segment"
            );
            let retained = &writer.pending_owned_segments[0];
            let witness = (
                retained.as_bytes().as_ptr(),
                retained.as_bytes().to_vec(),
                retained.header().segment_id,
            );
            let duplicate = writer.pending_owned_segments[0].clone();
            writer.pending_owned_segments.push(duplicate);
            witness
        };

        assert!(
            matches!(
                index.commit(&cx).await,
                Err(QuillIndexError::Keeper(KeeperError::InvalidTransition {
                    detail,
                    ..
                })) if detail.contains("supplied more than once")
            ),
            "the duplicated owned segment must be rejected by the publisher itself",
        );

        {
            let writer = index.writer_mut();
            assert_eq!(
                writer.pending_owned_segments.len(),
                2,
                "a rejected publication must retain the inventory it handed over"
            );
            assert_eq!(
                writer.pending_owned_segments[0].as_bytes().as_ptr(),
                payload_ptr,
                "the retained payload must be the exact same allocation, never a copy"
            );
            assert_eq!(
                writer.pending_owned_segments[0].as_bytes(),
                payload.as_slice(),
                "the failed attempt must not disturb the retained payload bytes"
            );
            // Repair only the injected duplicate. The sealed original stays
            // exactly where the failed publication left it.
            writer.pending_owned_segments.truncate(1);
        }
        assert_eq!(
            index
                .snapshot()
                .expect("rejected publisher snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation,
            "a rejected publication must not advance the published generation"
        );
        assert!(
            index
                .search_paginated(&cx, "aardvark", 10, 0, true)
                .expect("query the unchanged epoch")
                .hits
                .is_empty(),
            "a rejected publication must not make the segment visible"
        );

        let published = index
            .commit(&cx)
            .await
            .expect("the repaired retry publishes the retained segment");
        assert_eq!(
            index
                .snapshot()
                .expect("retried publisher snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation + 1,
            "the retried publication must advance exactly one generation"
        );
        assert!(
            index.writer_mut().pending_owned_segments.is_empty(),
            "a successful publication must consume the pending inventory"
        );
        let segment = published
            .segments()
            .iter()
            .find(|segment| segment.header().segment_id == segment_id)
            .expect("the retry installs the exact retained segment");
        assert_eq!(
            segment.source_bytes().as_ptr(),
            payload_ptr,
            "the retry must adopt the retained allocation, never a copy"
        );
        assert_eq!(
            segment.source_bytes(),
            payload.as_slice(),
            "the retried publication must seal byte-identical FSLX"
        );
        segment.verify().expect("the published segment verifies");
        let hits = index
            .search_paginated(&cx, "aardvark", 10, 0, true)
            .expect("query the retried publication");
        assert_eq!(hits.total_count, Some(1));
        assert_eq!(hits.hits[0].document_id, "doc-a");
    });
}
