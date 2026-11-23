use super::*;

fn concat_merge_query_results(index: &QuillIndex, cx: &Cx) -> Vec<QuillSearchResult> {
    CONCAT_MERGE_QUERIES
        .iter()
        .map(|query| {
            index
                .search_paginated(cx, query, 10, 0, true)
                .expect("concat-merge fixture query")
        })
        .collect()
}

/// Build a batch large enough that every shard clears the per-shard floor
/// and the writer takes the fan-out path.
fn fanout_corpus(prefix: &str, count: usize) -> Vec<IndexableDocument> {
    (0..count)
        .map(|ordinal| {
            IndexableDocument::new(
                format!("{prefix}-{ordinal}"),
                format!("shared corpus term ordinal {ordinal}"),
            )
        })
        .collect()
}

#[test]
fn within_batch_fanout_agrees_with_the_retained_single_shard_path() {
    run_with_cx(|cx| async move {
        const DOCUMENTS: usize = 2_048;
        let documents = fanout_corpus("fanout", DOCUMENTS);

        // Fan-out arm: one batch partitioned across several shards.
        let fanout_config = QuillConfig {
            max_ingest_shards: 4,
            ..QuillConfig::default()
        };
        let detected = std::thread::available_parallelism().map_or(1, usize::from);
        let fanout_shards = fanout_config.resolved_ingest_shards(detected);
        let fanout = QuillIndex::in_memory(fanout_config).expect("fan-out memory index");
        fanout
            .index_documents(&cx, &documents)
            .await
            .expect("fan out one batch");
        fanout.commit(&cx).await.expect("publish fanned-out batch");

        // Control arm: the byte-identical single-shard path, same corpus.
        let serial_config = QuillConfig {
            deterministic_ingest: true,
            ..QuillConfig::default()
        };
        let serial = QuillIndex::in_memory(serial_config).expect("serial memory index");
        serial
            .index_documents(&cx, &documents)
            .await
            .expect("index one serial batch");
        serial.commit(&cx).await.expect("publish serial batch");

        // Every document survives the partition exactly once, and the two
        // arms agree as sets.
        let fanned = fanout
            .search_paginated(&cx, "corpus", DOCUMENTS, 0, true)
            .expect("fan-out query");
        let control = serial
            .search_paginated(&cx, "corpus", DOCUMENTS, 0, true)
            .expect("control query");
        assert_eq!(
            fanned.hits.len(),
            DOCUMENTS,
            "fan-out must not drop or duplicate a document",
        );
        assert_eq!(control.hits.len(), DOCUMENTS);
        assert_eq!(fanned.total_count, control.total_count);

        let mut fanned_ids = fanned
            .hits
            .iter()
            .map(|hit| hit.document_id.clone())
            .collect::<Vec<_>>();
        let mut control_ids = control
            .hits
            .iter()
            .map(|hit| hit.document_id.clone())
            .collect::<Vec<_>>();
        fanned_ids.sort();
        control_ids.sort();
        assert_eq!(fanned_ids, control_ids, "fan-out changed the result set");

        assert_pairwise_disjoint_manifest(
            fanout
                .snapshot()
                .expect("fan-out snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .segments
                .as_slice(),
        );
        if fanout_shards > 1 {
            assert!(
                fanout
                    .snapshot()
                    .expect("fan-out snapshot is authoritative")
                    .segments()
                    .len()
                    > 1,
                "a fanned-out batch must seal a segment per participating shard",
            );
        }
    });
}

#[test]
fn within_batch_fanout_rejects_a_duplicate_id_inside_one_batch() {
    run_with_cx(|cx| async move {
        const DOCUMENTS: usize = 2_048;
        let mut documents = fanout_corpus("fanout-dup", DOCUMENTS);
        // Collide the last document with the first, so the collision spans
        // two different shards of the same batch.
        documents[DOCUMENTS - 1] =
            IndexableDocument::new("fanout-dup-0", "shared corpus term ordinal 0");

        let config = QuillConfig {
            max_ingest_shards: 4,
            ..QuillConfig::default()
        };
        let index = QuillIndex::in_memory(config).expect("fan-out memory index");
        let error = index
            .index_documents(&cx, &documents)
            .await
            .expect_err("a duplicate id inside one fanned-out batch must be rejected");
        assert!(
            error.to_string().contains("duplicate live document id"),
            "unexpected error: {error}",
        );
    });
}

#[test]
fn bulk_cadence_publishes_intermediate_manifests_and_finish_leaves_one_segment() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            scribe_shard_budget_bytes: 1,
            deterministic_ingest: true,
            bulk_load_mode: true,
            bulk_publish_segment_cadence: 3,
            tier_fanout: 2,
            merge_max_hole_ratio: 1.0,
            ..QuillConfig::default()
        };
        let index = QuillIndex::in_memory(config).expect("bulk memory index");
        let documents = (0..7)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("bulk-{ordinal}"),
                    format!("bulk cadence document {ordinal}"),
                )
            })
            .collect::<Vec<_>>();
        index
            .index_documents(&cx, &documents)
            .await
            .expect("accumulate bulk fixture");

        let intermediate = index
            .snapshot()
            .expect("bulk intermediate snapshot is authoritative");
        assert_eq!(intermediate.segments().len(), 6);
        assert_eq!(intermediate.doc_count(), 6);
        assert_ne!(
            intermediate.loaded_manifest().manifest.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
            0,
        );
        assert!(index.has_uncommitted_changes());

        let finished = index
            .finish_bulk_load(&cx)
            .await
            .expect("finish bulk fixture");
        assert_eq!(finished.segments().len(), 1);
        assert_eq!(finished.doc_count(), 7);
        assert_eq!(
            finished.loaded_manifest().manifest.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
            0,
        );
        assert!(!index.has_uncommitted_changes());
        assert_eq!(
            index
                .search_paginated(&cx, "bulk", 10, 0, true)
                .expect("search finished bulk index")
                .total_count,
            Some(7),
        );
    });
}

/// bd-k07zw: fsfs publishes every complete-generation edit, delete, and
/// append through a short single-shard writer session that ends in a
/// bulk-finish concat. Each session used to open a fresh 65,536-docid
/// lease, and the concat stored the skipped slots as dense DOCLEN/IDMAP
/// holes (2.36 MB per one-line edit). The session must continue at the
/// exact docid frontier, so each new row widens the merged hull by one
/// slot.
#[test]
fn reopened_single_shard_sessions_merge_without_lease_holes() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("lexical directory");
        let bulk = || QuillConfig {
            bulk_load_mode: true,
            max_ingest_shards: 1,
            ..QuillConfig::default()
        };
        let documents = (0..5)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("doc-{ordinal}"),
                    format!("generation fixture document {ordinal}"),
                )
            })
            .collect::<Vec<_>>();
        let index = QuillIndex::create(&cx, directory.path(), bulk())
            .await
            .expect("create bulk index");
        index
            .index_documents(&cx, &documents)
            .await
            .expect("index initial build");
        let built = index
            .finish_bulk_load(&cx)
            .await
            .expect("finish initial build");
        assert_eq!(built.loaded_manifest().manifest.docid_high_watermark, 5);
        drop(index);

        for session in 0..3_u64 {
            let index = QuillIndex::create(&cx, directory.path(), bulk())
                .await
                .expect("reopen for one session");
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        format!("added-{session}"),
                        format!("generation fixture addition {session}"),
                    )],
                )
                .await
                .expect("index one added document");
            let snapshot = index.finish_bulk_load(&cx).await.expect("finish session");
            let manifest = &snapshot.loaded_manifest().manifest;
            assert_eq!(manifest.segments.len(), 1);
            let segment = &manifest.segments[0];
            assert_eq!(
                (segment.docid_lo, segment.docid_hi),
                (0, 6 + session),
                "session {session} must widen the hull by one slot, not by a lease"
            );
            assert_eq!(manifest.docid_high_watermark, 6 + session);
            assert_eq!(snapshot.doc_count(), 6 + session);
            drop(index);
        }
    });
}

/// Bounded seal-producing batches compose across commits, and a REJECTED
/// batch composes with them by contributing nothing.
///
/// bd-quill-rejected-ingest-publishes-partial-batch-aihri rewrote the
/// second half of this test. It previously asserted the opposite contract
/// in these words — "duplicate batch must fail after accepting its unique
/// prefix", then an armed retry guard, then `commit()` "reconciling" the
/// partial batch into `doc_count() == 3`. That was the defect, written down
/// as intent: the reconciling commit was publishing rows the caller had
/// been told were rejected. What the bounded-seal half of this test covers
/// — many small batches each producing a sealed segment, composing into one
/// committed generation — is unchanged and still asserted below.
#[test]
fn successful_sealed_batches_compose_and_a_rejected_batch_contributes_nothing() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            scribe_shard_budget_bytes: 1,
            deterministic_ingest: true,
            ..QuillConfig::default()
        };
        let mut index = QuillIndex::in_memory(config).expect("bounded-batch memory index");

        for ordinal in 0..2 {
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        format!("bounded-{ordinal}"),
                        "bounded batch searchable",
                    )],
                )
                .await
                .expect("successful seal-producing batch");
        }
        assert_eq!(
            index
                .doc_count()
                .expect("unpublished bounded-batch count is authoritative"),
            0,
            "segments remain unpublished"
        );
        assert!(index.has_uncommitted_changes());

        let duplicate_batch = [
            IndexableDocument::new("bounded-2", "bounded batch searchable"),
            IndexableDocument::new("bounded-2", "bounded duplicate"),
        ];
        assert!(
            index.index_documents(&cx, &duplicate_batch).await.is_err(),
            "an in-batch duplicate must reject the whole batch"
        );
        // The rejected batch staged nothing, so there is nothing to
        // reconcile and no retry obligation to impose on the caller.
        assert!(!index.writer_mut().ingest_retry_required);
        index
            .index_documents(
                &cx,
                &[IndexableDocument::new(
                    "bounded-3",
                    "bounded batch searchable",
                )],
            )
            .await
            .expect("a rejected batch must not block the next ingest");

        index
            .commit(&cx)
            .await
            .expect("commit the accepted batches");
        // bounded-0, bounded-1, bounded-3. bounded-2 was rejected and
        // contributed nothing: this is the count that was 3-with-bounded-2
        // before the fix, and it is the whole point of the bead.
        assert_eq!(
            index
                .doc_count()
                .expect("bounded-batch count is authoritative"),
            3
        );
        assert!(!index.writer_mut().ingest_retry_required);
        assert!(
            index
                .search_paginated(&cx, "duplicate", 10, 0, true)
                .expect("search for the rejected document's text")
                .hits
                .is_empty(),
            "no text from the rejected batch may be searchable",
        );

        // The rejected id was never claimed, so it still ingests cleanly.
        index
            .index_documents(
                &cx,
                &[IndexableDocument::new(
                    "bounded-2",
                    "bounded batch searchable",
                )],
            )
            .await
            .expect("the rejected id must remain available");
        index.commit(&cx).await.expect("final bounded-batch commit");
        assert_eq!(
            index
                .search_paginated(&cx, "searchable", 10, 0, true)
                .expect("search committed bounded batches")
                .total_count,
            Some(4),
        );
    });
}

#[test]
fn continuous_tier_merges_preserve_safe_live_ingest_leases() {
    run_with_cx(|cx| async move {
        let config = deterministic_config();
        let segment_bound = 3 * (config.tier_fanout - 1);
        let index = QuillIndex::in_memory(config).expect("continuous tier index");
        for ordinal in 0_u32..192 {
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        format!("continuous-{ordinal}"),
                        "shared continuous content",
                    )],
                )
                .await
                .expect("append to the same live writer");
            let snapshot = index
                .commit(&cx)
                .await
                .expect("commit and apply tier policy");
            assert_pairwise_disjoint_manifest(&snapshot.loaded_manifest().manifest.segments);
        }
        let snapshot = index.snapshot().expect("continuous tier snapshot");
        let ids = index
            .collect_docids(&cx, "shared")
            .expect("query all committed rows");
        let expected_ids: Vec<_> = (0_u32..192).collect();
        assert_eq!(snapshot.doc_count(), 192);
        eprintln!(
            "GH41 continuous writer: documents={} segments={} bound={segment_bound} last_docid={:?}",
            snapshot.doc_count(),
            snapshot.segments().len(),
            ids.last()
        );
        assert_eq!(
            ids, expected_ids,
            "a merge must not burn a live lease whose next ID is beyond every published interval"
        );
        for ordinal in 0_u32..192 {
            assert_eq!(
                snapshot.materialize_document_id(ordinal),
                Some(DocId::new(format!("continuous-{ordinal}"))),
                "each retained numeric ID must still name its original document"
            );
        }
        assert!(
            snapshot.segments().len() <= segment_bound,
            "continuous commit/merge cycles must retain the S/M/L segment-count bound"
        );
    });
}

#[test]
fn commit_applies_tier_policy_to_cross_shard_bound_consecutive_run() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            max_ingest_shards: 2,
            tier_fanout: 2,
            merge_max_hole_ratio: 1.0,
            ..QuillConfig::default()
        };
        let index = QuillIndex::in_memory(config).expect("tier policy memory index");
        index
            .index_documents(&cx, &[IndexableDocument::new("tier-a", "shared alpha")])
            .await
            .expect("route first tier batch");
        index
            .index_documents(&cx, &[IndexableDocument::new("tier-b", "shared beta")])
            .await
            .expect("route second tier batch");
        index.commit(&cx).await.expect("publish and tier merge");

        let snapshot = index
            .snapshot()
            .expect("tier-policy snapshot is authoritative");
        assert_eq!(snapshot.segments().len(), 1);
        assert_eq!(snapshot.doc_count(), 2);
        let merged_hi = snapshot.segments()[0].manifest().docid_hi;
        assert_eq!(
            index
                .search_paginated(&cx, "shared", 10, 0, true)
                .expect("query tier-merged snapshot")
                .total_count,
            Some(2),
        );

        index
            .index_documents(&cx, &[IndexableDocument::new("tier-after", "shared gamma")])
            .await
            .expect("append after tier merge");
        index.commit(&cx).await.expect("publish post-merge append");
        let post_merge = index
            .snapshot()
            .expect("post-merge snapshot is authoritative");
        assert_pairwise_disjoint_manifest(
            post_merge.loaded_manifest().manifest.segments.as_slice(),
        );
        assert!(
            post_merge
                .segments()
                .iter()
                .any(|segment| segment.manifest().docid_lo >= merged_hi),
            "lease retirement must place future batches above the merged hull",
        );
    });
}

#[test]
fn visibility_lag_uses_the_same_lifecycle_commit_boundary() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            deterministic_ingest: true,
            max_visibility_lag_ms: 1,
            ..QuillConfig::default()
        };
        let index = QuillIndex::in_memory(config).expect("visibility memory index");
        index
            .index_documents(&cx, &[IndexableDocument::new("lag-a", "visibility alpha")])
            .await
            .expect("accumulate first visibility batch");
        assert_eq!(
            index.doc_count().expect("pre-lag count is authoritative"),
            0,
            "sub-bound changes stay unpublished"
        );

        std::thread::sleep(Duration::from_millis(2));
        index
            .index_documents(&cx, &[IndexableDocument::new("lag-b", "visibility beta")])
            .await
            .expect("lag-bound batch forces publication");
        assert_eq!(
            index
                .doc_count()
                .expect("lag-forced count is authoritative"),
            2
        );
        assert!(!index.has_uncommitted_changes());
    });
}

#[test]
fn scoreless_docset_collector_is_wired_across_committed_segments() {
    run_with_cx(|cx| async move {
        let documents = fixture_documents();
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        index
            .index_documents(&cx, &documents[..2])
            .await
            .expect("first segment documents");
        index.commit(&cx).await.expect("first segment commit");
        index
            .index_documents(&cx, &documents[2..])
            .await
            .expect("second segment documents");

        assert_eq!(
            index
                .collect_docids(&cx, "rust OR python")
                .expect("old committed doc set")
                .len(),
            2,
            "uncommitted second-segment document must remain invisible",
        );

        index.commit(&cx).await.expect("second segment commit");
        let docids = index
            .collect_docids(&cx, "rust OR python")
            .expect("Boolean doc set");
        let ranked = index
            .search_paginated(&cx, "rust OR python", 10, 0, true)
            .expect("ranked Boolean search");
        let mut ranked_docids = ranked
            .hits
            .iter()
            .map(|hit| hit.global_docid)
            .collect::<Vec<_>>();
        ranked_docids.sort_unstable();
        assert_eq!(docids, ranked_docids);
        assert_eq!(docids.len(), 3);
        assert_eq!(ranked.total_count, Some(3));
    });
}

#[test]
fn in_memory_concat_merge_preserves_query_results_scores_and_docids() {
    run_with_cx(|cx| async move {
        let index = concat_merge_fixture_index(&cx).await;
        let source_ids = committed_segment_ids(&index);
        assert_eq!(source_ids.len(), 3);

        let generation_before = index
            .snapshot()
            .expect("concat source snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let results_before = concat_merge_query_results(&index, &cx);
        let docids_before = index
            .collect_docids(&cx, "rust OR python")
            .expect("collect pre-merge docids");
        let output_segment_id = fresh_merge_segment_id(&index, 0xc011_ca7e_0000_0001);
        let source_snapshot = index
            .snapshot()
            .expect("concat source snapshot is authoritative");
        let source_dictionary_counts = source_snapshot
            .segments()
            .iter()
            .map(RecoveredSegment::term_dictionary_cache_counts)
            .collect::<Vec<_>>();

        index
            .concat_merge(&cx, &source_ids, output_segment_id, 17)
            .await
            .expect("merge the complete in-memory leaf run");

        for (segment, (full_validations_before, borrowed_views_before)) in source_snapshot
            .segments()
            .iter()
            .zip(source_dictionary_counts)
        {
            let (full_validations_after, borrowed_views_after) =
                segment.term_dictionary_cache_counts();
            assert_eq!(
                full_validations_after, full_validations_before,
                "concat must not repeat complete TERMDICT validation"
            );
            assert_eq!(
                borrowed_views_after,
                borrowed_views_before + 1,
                "concat must open one cached dictionary view per source segment"
            );
        }
        assert_eq!(
            index
                .snapshot()
                .expect("concat successor snapshot is authoritative")
                .segments()
                .len(),
            1
        );
        assert_eq!(
            index
                .snapshot()
                .expect("concat successor snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation_before + 1,
        );
        assert_eq!(
            concat_merge_query_results(&index, &cx),
            results_before,
            "ranked hits, exact scores, global docids, and counts must survive merge",
        );
        assert_eq!(
            index
                .collect_docids(&cx, "rust OR python")
                .expect("collect post-merge docids"),
            docids_before,
        );
        let merge_seal_seq = index
            .snapshot()
            .expect("concat successor snapshot is authoritative")
            .segments()[0]
            .manifest()
            .seal_seq;

        index
            .index_documents(
                &cx,
                &[IndexableDocument::new(
                    "after-merge",
                    "ordinary sealing remains available after concat merge",
                )],
            )
            .await
            .expect("accumulate after concat merge");
        index
            .commit(&cx)
            .await
            .expect("ordinary commit after concat merge");
        assert_eq!(
            index
                .snapshot()
                .expect("post-concat snapshot is authoritative")
                .segments()
                .iter()
                .map(|segment| segment.manifest().seal_seq)
                .collect::<Vec<_>>(),
            [merge_seal_seq, merge_seal_seq + 1],
            "concat output and the next ordinary seal need distinct monotone sequences",
        );
    });
}

#[test]
fn concat_merge_matches_fresh_monolithic_df_100_plus_300_with_same_docids() {
    run_with_cx(|cx| async move {
        let (leaves, monolithic) = q1_ob2a_fixture_indexes();
        assert_eq!(
            leaves
                .snapshot()
                .expect("leaf source snapshot is authoritative")
                .segments()
                .iter()
                .map(|segment| (segment.manifest().docid_lo, segment.manifest().docid_hi))
                .collect::<Vec<_>>(),
            [(0, 100), (100, 400)],
            "Q1-OB2a leaves must be adjacent inside one lease",
        );
        assert_eq!(
            monolithic
                .snapshot()
                .expect("monolithic source snapshot is authoritative")
                .segments()[0]
                .manifest()
                .docid_lo,
            0
        );
        assert_eq!(
            monolithic
                .snapshot()
                .expect("monolithic source snapshot is authoritative")
                .segments()[0]
                .manifest()
                .docid_hi,
            400
        );

        let source_shared_dfs = leaves
            .snapshot()
            .expect("leaf source snapshot is authoritative")
            .segments()
            .iter()
            .map(|segment| {
                open_dictionary(segment, DEFAULT_SCHEMA)
                    .expect("source Q1-OB2a TERMDICT")
                    .lookup(CONTENT_FIELD, b"shared")
                    .expect("source shared lookup")
                    .expect("source shared term")
                    .metadata
                    .doc_freq
            })
            .collect::<Vec<_>>();
        assert_eq!(source_shared_dfs, [100, 300]);
        assert_eq!(
            snapshot_doc_freq(
                &monolithic
                    .snapshot()
                    .expect("monolithic source snapshot is authoritative"),
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"shared",
            )
            .expect("monolithic shared df"),
            400,
        );

        let source_stats = leaves
            .snapshot()
            .expect("leaf source snapshot is authoritative")
            .segments()
            .iter()
            .map(q1_ob2a_decoded_stats)
            .collect::<Vec<_>>();
        let source_aggregate =
            aggregate_field_stats(source_stats.iter()).expect("aggregate source STATS");
        let monolithic_stats = q1_ob2a_decoded_stats(
            &monolithic
                .snapshot()
                .expect("monolithic source snapshot is authoritative")
                .segments()[0],
        );
        let monolithic_aggregate = monolithic_stats
            .rows()
            .iter()
            .map(|row| SnapshotFieldStats {
                field_ord: row.field_ord,
                total_tokens: row.total_tokens,
                doc_count: u64::from(row.doc_count),
            })
            .collect::<Vec<_>>();
        assert_eq!(source_aggregate, monolithic_aggregate);
        assert_eq!(
            leaves
                .snapshot()
                .expect("leaf source snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats,
            monolithic
                .snapshot()
                .expect("monolithic source snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats,
        );

        let monolithic_terms = q1_ob2a_decoded_terms(&monolithic);
        assert_eq!(
            q1_ob2a_decoded_terms(&leaves),
            monolithic_terms,
            "source terms and decoded postings must match fresh monolithic indexing",
        );
        let monolithic_evidence = q1_ob2a_query_evidence(&monolithic, &cx);
        assert_eq!(
            q1_ob2a_query_evidence(&leaves, &cx),
            monolithic_evidence,
            "source leaves and fresh monolithic seal must have exact scores, ids, docids, and counts",
        );
        assert_eq!(
            leaves
                .collect_docids(&cx, "shared")
                .expect("source shared docids"),
            (0_u32..400).collect::<Vec<_>>(),
        );

        let source_ids = committed_segment_ids(&leaves);
        leaves
            .concat_merge(&cx, &source_ids, 0x0b2a_0004, 0)
            .await
            .expect("Q1-OB2a concat merge");
        assert_eq!(
            leaves
                .snapshot()
                .expect("merged leaf snapshot is authoritative")
                .segments()
                .len(),
            1
        );
        assert_eq!(
            snapshot_doc_freq(
                &leaves
                    .snapshot()
                    .expect("merged leaf snapshot is authoritative"),
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"shared",
            )
            .expect("merged shared df"),
            400,
        );
        assert_eq!(
            q1_ob2a_decoded_terms(&leaves),
            monolithic_terms,
            "every merged term and decoded posting must match fresh monolithic indexing",
        );
        assert_eq!(
            q1_ob2a_decoded_stats(
                &leaves
                    .snapshot()
                    .expect("merged leaf snapshot is authoritative")
                    .segments()[0],
            ),
            monolithic_stats,
            "merged decoded STATS must match fresh monolithic indexing",
        );
        assert_eq!(
            leaves
                .snapshot()
                .expect("merged leaf snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats,
            monolithic
                .snapshot()
                .expect("monolithic source snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats,
        );
        assert_eq!(
            q1_ob2a_query_evidence(&leaves, &cx),
            monolithic_evidence,
            "merged and monolithic query scores, ids, docids, counts, and docsets must match",
        );
    });
}

#[test]
fn concat_merge_rejects_reversed_and_skipped_runs_without_publication() {
    run_with_cx(|cx| async move {
        let index = concat_merge_fixture_index(&cx).await;
        let source_ids = committed_segment_ids(&index);
        assert_eq!(source_ids.len(), 3);
        let manifest_before = index
            .snapshot()
            .expect("concat rejection snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .clone();
        let output_segment_id = fresh_merge_segment_id(&index, 0xc011_ca7e_0000_0010);

        let reversed = [source_ids[1], source_ids[0]];
        let Err(error) = index
            .concat_merge(&cx, &reversed, output_segment_id, 19)
            .await
        else {
            panic!("reversed concat-merge run unexpectedly published"); // ubs:ignore -- Test rejects reversed merge runs.
        };
        assert!(matches!(
            error,
            QuillIndexError::Keeper(KeeperError::ConcatMerge {
                source: ConcatMergeError::NonConsecutiveSources { .. }
            })
        ));
        assert_eq!(
            &index
                .snapshot()
                .expect("reversed rejection snapshot is authoritative")
                .loaded_manifest()
                .manifest,
            &manifest_before,
            "reversed rejection must not advance the snapshot generation",
        );

        let skipped = [source_ids[0], source_ids[2]];
        let Err(error) = index
            .concat_merge(&cx, &skipped, output_segment_id, 23)
            .await
        else {
            panic!("skipped concat-merge run unexpectedly published"); // ubs:ignore -- Test rejects skipped merge runs.
        };
        assert!(matches!(
            error,
            QuillIndexError::Keeper(KeeperError::ConcatMerge {
                source: ConcatMergeError::NonConsecutiveSources { .. }
            })
        ));
        assert_eq!(
            &index
                .snapshot()
                .expect("skipped rejection snapshot is authoritative")
                .loaded_manifest()
                .manifest,
            &manifest_before,
            "skipped-run rejection must leave the exact snapshot unchanged",
        );

        let wrapped = [source_ids[1], source_ids[2], source_ids[0]];
        let Err(error) = index
            .concat_merge(&cx, &wrapped, output_segment_id, 29)
            .await
        else {
            panic!("manifest-wrapping concat-merge run unexpectedly published"); // ubs:ignore -- Test rejects wrapped merge runs.
        };
        assert!(matches!(
            error,
            QuillIndexError::Keeper(KeeperError::ConcatMerge {
                source: ConcatMergeError::SourceRunPastManifest {
                    position: 2,
                    actual,
                }
            }) if actual == source_ids[0]
        ));
        assert_eq!(
            &index
                .snapshot()
                .expect("wrapped rejection snapshot is authoritative")
                .loaded_manifest()
                .manifest,
            &manifest_before,
            "wrapped-run rejection must leave the exact snapshot unchanged",
        );
    });
}

#[test]
fn concat_merge_direct_and_associated_schedules_match() {
    run_with_cx(|cx| async move {
        let direct = concat_merge_fixture_index(&cx).await;
        let left_associated = concat_merge_fixture_index(&cx).await;
        let right_associated = concat_merge_fixture_index(&cx).await;
        let leaf_ids = committed_segment_ids(&direct);
        assert_eq!(leaf_ids, committed_segment_ids(&left_associated));
        assert_eq!(leaf_ids, committed_segment_ids(&right_associated));
        assert_eq!(leaf_ids.len(), 3);

        let results_before = concat_merge_query_results(&direct, &cx);
        assert_eq!(
            concat_merge_query_results(&left_associated, &cx),
            results_before,
        );
        assert_eq!(
            concat_merge_query_results(&right_associated, &cx),
            results_before,
        );
        let final_segment_id = fresh_merge_segment_id(&direct, 0xc011_ca7e_0000_0020);
        let intermediate_segment_id =
            fresh_merge_segment_id(&left_associated, 0xc011_ca7e_0000_0030);
        let right_intermediate_segment_id =
            fresh_merge_segment_id(&right_associated, 0xc011_ca7e_0000_0040);

        direct
            .concat_merge(&cx, &leaf_ids, final_segment_id, 31)
            .await
            .expect("direct three-leaf concat merge");
        left_associated
            .concat_merge(&cx, &leaf_ids[..2], intermediate_segment_id, 29)
            .await
            .expect("left-associated first concat merge");
        let left_final_sources = [intermediate_segment_id, leaf_ids[2]];
        left_associated
            .concat_merge(&cx, &left_final_sources, final_segment_id, 31)
            .await
            .expect("left-associated final concat merge");
        right_associated
            .concat_merge(&cx, &leaf_ids[1..], right_intermediate_segment_id, 29)
            .await
            .expect("right-associated first concat merge");
        let right_final_sources = [leaf_ids[0], right_intermediate_segment_id];
        right_associated
            .concat_merge(&cx, &right_final_sources, final_segment_id, 31)
            .await
            .expect("right-associated final concat merge");

        assert_eq!(
            direct
                .snapshot()
                .expect("direct merge snapshot is authoritative")
                .segments()
                .len(),
            1
        );
        assert_eq!(
            left_associated
                .snapshot()
                .expect("left-associated merge snapshot is authoritative")
                .segments()
                .len(),
            1
        );
        assert_eq!(
            right_associated
                .snapshot()
                .expect("right-associated merge snapshot is authoritative")
                .segments()
                .len(),
            1
        );
        let direct_snapshot = direct
            .snapshot()
            .expect("direct merge snapshot is authoritative");
        let left_snapshot = left_associated
            .snapshot()
            .expect("left-associated merge snapshot is authoritative");
        let right_snapshot = right_associated
            .snapshot()
            .expect("right-associated merge snapshot is authoritative");
        let direct_segment = &direct_snapshot.segments()[0];
        let left_segment = &left_snapshot.segments()[0];
        let right_segment = &right_snapshot.segments()[0];
        assert_eq!(
            direct_segment.source_bytes(),
            left_segment.source_bytes(),
            "the complete merged FSLX image must be schedule-independent",
        );
        assert_eq!(
            direct_segment.source_bytes(),
            right_segment.source_bytes(),
            "right-associated merge must emit the same complete FSLX image",
        );
        for kind in KNOWN_SECTION_KINDS {
            assert_eq!(
                direct_segment.section(kind).expect("direct merged section"),
                left_segment
                    .section(kind)
                    .expect("left-associated merged section"),
                "section {} differs by merge schedule",
                kind.raw(),
            );
            assert_eq!(
                direct_segment.section(kind).expect("direct merged section"),
                right_segment
                    .section(kind)
                    .expect("right-associated merged section"),
                "section {} differs under right association",
                kind.raw(),
            );
        }

        let direct_results = concat_merge_query_results(&direct, &cx);
        let left_results = concat_merge_query_results(&left_associated, &cx);
        let right_results = concat_merge_query_results(&right_associated, &cx);
        assert_eq!(direct_results, results_before);
        assert_eq!(left_results, results_before);
        assert_eq!(right_results, results_before);
    });
}

#[test]
fn deterministic_runs_have_identical_segments_and_results() {
    run_with_cx(|cx| async move {
        let documents = fixture_documents();
        let first = QuillIndex::in_memory(deterministic_config()).expect("first index");
        let second = QuillIndex::in_memory(deterministic_config()).expect("second index");
        first
            .index_documents(&cx, &documents)
            .await
            .expect("first ingest");
        second
            .index_documents(&cx, &documents)
            .await
            .expect("second ingest");
        first.commit(&cx).await.expect("first commit");
        second.commit(&cx).await.expect("second commit");

        assert_eq!(
            first
                .snapshot()
                .expect("first deterministic snapshot is authoritative")
                .loaded_manifest()
                .manifest,
            second
                .snapshot()
                .expect("second deterministic snapshot is authoritative")
                .loaded_manifest()
                .manifest
        );
        let first_snapshot = first
            .snapshot()
            .expect("first deterministic snapshot is authoritative");
        let second_snapshot = second
            .snapshot()
            .expect("second deterministic snapshot is authoritative");
        let first_segment = &first_snapshot.segments()[0];
        let second_segment = &second_snapshot.segments()[0];
        assert_eq!(first_segment.header(), second_segment.header());
        assert_eq!(first_segment.source_bytes(), second_segment.source_bytes());
        for kind in [
            SectionKind::TERMDICT,
            SectionKind::POSTINGS,
            SectionKind::POSITIONS,
            SectionKind::BLOCKMAX,
            SectionKind::DOCLEN,
            SectionKind::IDMAP,
            SectionKind::IDHASH,
            SectionKind::STOREDMETA,
            SectionKind::STATS,
        ] {
            assert_eq!(
                first_segment.section(kind).expect("first section"),
                second_segment.section(kind).expect("second section"),
                "section {}",
                kind.raw()
            );
        }

        let first_result = first
            .search_paginated(&cx, "rust", 10, 0, true)
            .expect("first search");
        let second_result = second
            .search_paginated(&cx, "rust", 10, 0, true)
            .expect("second search");
        assert_eq!(first_result, second_result);
    });
}
