use super::*;

/// cass#453: the inputs a concat merge folds away must leave the disk
/// once every durable slot has dropped them and their retirement receipts
/// have aged, however many further publications happen in between -- and
/// a reader that opened the pre-merge generation keeps answering from its
/// mapped segments after they are unlinked.
#[cfg(unix)]
#[test]
fn retired_merge_inputs_are_reclaimed_while_an_older_reader_keeps_working() {
    use crate::keeper::{
        DEFAULT_GARBAGE_GRACE, GarbageCollectionOptions, KeeperSnapshot, collect_writer_garbage_at,
    };
    use crate::segment::SectionKind;

    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("index directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create on-disk index");
        for (document_id, content) in [("one", "alpha shared"), ("two", "beta shared")] {
            LexicalWrite::index_document(
                &index,
                &cx,
                &IndexableDocument::new(document_id, content),
            )
            .await
            .expect("stage document");
            LexicalWrite::commit(&index, &cx)
                .await
                .expect("publish one segment per round");
        }
        let inputs = committed_segment_ids(&index);
        assert_eq!(inputs.len(), 2, "each round sealed exactly one segment");
        let input_files = live_segment_files(&index);

        // A reader on the pre-merge generation, plus a raw snapshot that
        // pins the input segments' mappings.
        let older_reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open pre-merge reader");
        let older_generation = older_reader.keeper_generation();
        let older_snapshot =
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA).expect("pin inputs");
        assert_eq!(older_snapshot.segments().len(), 2);

        let output_segment_id = fresh_merge_segment_id(&index, 0x0453_0001);
        index
            .concat_merge(&cx, &inputs, output_segment_id, 1_700_000_000)
            .await
            .expect("fold both rounds into one segment");
        assert_eq!(committed_segment_ids(&index), vec![output_segment_id]);
        let (on_disk, receipts) = segment_files_and_receipts(directory.path());
        assert_eq!(
            on_disk.len(),
            3,
            "folded inputs stay on disk after the merge"
        );
        assert!(
            receipts.is_empty(),
            "MANIFEST.prev still references the inputs, so nothing is retired yet"
        );

        // Two further back-to-back publications: the first retires the
        // inputs (they drop out of MANIFEST.prev), the second is the kind
        // of publication that used to restart their grace.
        for (document_id, content) in [("three", "gamma shared"), ("four", "delta shared")] {
            LexicalWrite::index_document(
                &index,
                &cx,
                &IndexableDocument::new(document_id, content),
            )
            .await
            .expect("stage later document");
            LexicalWrite::commit(&index, &cx)
                .await
                .expect("publish later generation");
        }
        let (on_disk, receipts) = segment_files_and_receipts(directory.path());
        assert_eq!(
            receipts,
            input_files
                .iter()
                .map(|name| format!("{name}.retired"))
                .collect::<Vec<_>>(),
            "the retiring publication stamped one receipt per folded input"
        );
        assert_eq!(
            on_disk.len(),
            5,
            "inputs are still within their grace period"
        );

        // Inside the grace period the public sweep is a no-op, both
        // through the index and at the Keeper seam.
        let report = index
            .collect_garbage(&cx)
            .await
            .expect("sweep inside the grace period");
        assert!(report.is_empty(), "nothing is old enough yet: {report:?}");
        assert_eq!(segment_files_and_receipts(directory.path()).0.len(), 5);

        // Once the receipts have aged, the sweep reclaims exactly the
        // folded inputs and their receipts, even though the two later
        // publications are far younger than the grace period.
        let after_grace = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        let report = collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            after_grace,
        )
        .expect("sweep after the grace period");
        let mut removed = report
            .removed
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        removed.sort();
        let mut expected_removed = input_files.clone();
        expected_removed.extend(input_files.iter().map(|name| format!("{name}.retired")));
        expected_removed.sort();
        assert_eq!(removed, expected_removed);
        let (on_disk, receipts) = segment_files_and_receipts(directory.path());
        assert_eq!(on_disk, live_segment_files(&index));
        assert!(receipts.is_empty());

        // The pre-merge reader still answers, and the raw snapshot still
        // reads the unlinked inputs through its mappings.
        for segment in older_snapshot.segments() {
            segment
                .section(SectionKind::TERMDICT)
                .expect("unlinked input segment stays readable through its mapping");
        }
        assert_eq!(older_reader.keeper_generation(), older_generation);
        let hits = LexicalRead::search(&older_reader, &cx, "shared", 10)
            .await
            .expect("pre-merge reader searches after the sweep");
        assert!(
            hits.len() >= 2,
            "pre-merge reader still sees both folded documents: {hits:?}"
        );

        // A fresh reader and a reopened writer see the consolidated index.
        let fresh = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open post-sweep reader");
        assert_eq!(fresh.segment_count().expect("segment count"), 3);
        assert_eq!(
            LexicalRead::search(&fresh, &cx, "shared", 10)
                .await
                .expect("post-sweep search")
                .len(),
            4
        );
        drop(index);
        let reopened = QuillIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("reopen writer after the sweep");
        assert_eq!(committed_segment_ids(&reopened).len(), 3);
    });
}

/// A sweep interrupted after unlinking a segment but before its receipt
/// (the receipt sorts after its segment) leaves a receipt with no segment.
/// The next open recovers: the stale receipt is reclaimed once it is old
/// enough, and it never blocks readers or writers meanwhile.
#[cfg(unix)]
#[test]
fn interrupted_sweep_leaves_only_a_stale_receipt_that_the_next_sweep_reclaims() {
    use crate::keeper::{
        DEFAULT_GARBAGE_GRACE, GarbageCollectionOptions, collect_writer_garbage_at,
    };

    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("index directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create on-disk index");
        for (document_id, content) in [("one", "alpha"), ("two", "beta")] {
            LexicalWrite::index_document(
                &index,
                &cx,
                &IndexableDocument::new(document_id, content),
            )
            .await
            .expect("stage document");
            LexicalWrite::commit(&index, &cx)
                .await
                .expect("publish one segment per round");
        }
        let inputs = committed_segment_ids(&index);
        let input_files = live_segment_files(&index);
        let output_segment_id = fresh_merge_segment_id(&index, 0x0453_0002);
        index
            .concat_merge(&cx, &inputs, output_segment_id, 1_700_000_000)
            .await
            .expect("fold both rounds");
        LexicalWrite::index_document(&index, &cx, &IndexableDocument::new("three", "gamma"))
            .await
            .expect("stage retiring document");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("retiring publication");
        drop(index);

        // Simulate the crash: the first input's segment is gone, its
        // receipt survived.
        std::fs::remove_file(directory.path().join(&input_files[0]))
            .expect("simulate an unlink that completed before the crash");
        let (on_disk, receipts) = segment_files_and_receipts(directory.path());
        assert_eq!(
            on_disk.len(),
            3,
            "merged output, the third round, one input"
        );
        assert_eq!(receipts.len(), 2);

        // Readers and writers open normally over the partial sweep.
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("reader opens over a partially swept directory");
        assert_eq!(
            LexicalRead::search(&reader, &cx, "alpha OR beta OR gamma", 10)
                .await
                .expect("search over a partially swept directory")
                .len(),
            3
        );
        let reopened = QuillIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("writer reopens (and sweeps) over a partially swept directory");
        drop(reopened);

        // The aged sweep finishes the job: the surviving input goes with
        // its receipt, and the orphaned receipt goes on its own.
        let after_grace = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        let report = collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            after_grace,
        )
        .expect("sweep after the grace period");
        let mut removed = report
            .removed
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        removed.sort();
        let mut expected_removed = vec![
            format!("{}.retired", input_files[0]),
            input_files[1].clone(),
            format!("{}.retired", input_files[1]),
        ];
        expected_removed.sort();
        assert_eq!(removed, expected_removed);
        let (on_disk, receipts) = segment_files_and_receipts(directory.path());
        assert_eq!(on_disk.len(), 2);
        assert!(receipts.is_empty());
        assert!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                after_grace,
            )
            .expect("idempotent sweep")
            .is_empty()
        );
    });
}

#[cfg(unix)]
#[test]
fn collect_garbage_refuses_uncommitted_state_and_is_a_no_op_in_memory() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("index directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create on-disk index");
        LexicalWrite::index_document(&index, &cx, &IndexableDocument::new("one", "alpha"))
            .await
            .expect("stage document");
        let error = index
            .collect_garbage(&cx)
            .await
            .expect_err("a staged document blocks the sweep");
        assert!(
            matches!(error, QuillIndexError::InvalidState { ref detail } if detail.contains("committed")),
            "{error}"
        );
        LexicalWrite::commit(&index, &cx).await.expect("publish");
        assert!(
            index
                .collect_garbage(&cx)
                .await
                .expect("sweep a committed index")
                .is_empty()
        );

        let memory = QuillIndex::in_memory(deterministic_config()).expect("in-memory index");
        assert!(
            memory
                .collect_garbage(&cx)
                .await
                .expect("in-memory sweep")
                .is_empty()
        );
    });
}

/// A never-published build directory may reclaim a merge's folded inputs
/// at once. The default window keeps them; a zero window removes exactly
/// the inputs and their receipts, never a segment a MANIFEST slot names.
#[cfg(unix)]
#[test]
fn collect_garbage_with_zero_grace_reclaims_what_the_default_window_keeps() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("index directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create on-disk index");
        for (document_id, content) in [("one", "alpha"), ("two", "beta")] {
            LexicalWrite::index_document(
                &index,
                &cx,
                &IndexableDocument::new(document_id, content),
            )
            .await
            .expect("stage document");
            LexicalWrite::commit(&index, &cx)
                .await
                .expect("publish one segment per round");
        }
        let inputs = committed_segment_ids(&index);
        let input_files = live_segment_files(&index);
        let output_segment_id = fresh_merge_segment_id(&index, 0x0453_0003);
        index
            .concat_merge(&cx, &inputs, output_segment_id, 1_700_000_000)
            .await
            .expect("fold both rounds");
        LexicalWrite::index_document(&index, &cx, &IndexableDocument::new("three", "gamma"))
            .await
            .expect("stage retiring document");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("retiring publication");
        let (on_disk, receipts) = segment_files_and_receipts(directory.path());
        assert_eq!(on_disk.len(), 4, "merged output, third round, two inputs");
        assert_eq!(receipts.len(), 2);

        assert!(
            index
                .collect_garbage(&cx)
                .await
                .expect("default-window sweep")
                .is_empty(),
            "freshly retired inputs are inside the default window"
        );
        let zero = GarbageCollectionOptions {
            grace_period: Duration::ZERO,
        };
        let report = index
            .collect_garbage_with(&cx, zero)
            .await
            .expect("zero-window sweep");
        let mut removed = report
            .removed
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        removed.sort();
        let mut expected_removed = input_files
            .iter()
            .flat_map(|file| [file.clone(), format!("{file}.retired")])
            .collect::<Vec<_>>();
        expected_removed.sort();
        assert_eq!(removed, expected_removed);
        let (on_disk, receipts) = segment_files_and_receipts(directory.path());
        assert_eq!(on_disk.len(), 2);
        assert!(receipts.is_empty());
        assert!(
            index
                .collect_garbage_with(&cx, zero)
                .await
                .expect("idempotent zero-window sweep")
                .is_empty()
        );
        drop(index);

        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("reader opens the swept directory");
        assert_eq!(
            LexicalRead::search(&reader, &cx, "alpha OR beta OR gamma", 10)
                .await
                .expect("search the swept directory")
                .len(),
            3
        );
    });
}

// ==== Segment-parallel query fan-out (bd-quill-e4-argus-3ycz.9) ====

const SEGMENT_FANOUT_QUERIES: [&str; 5] = [
    "shared",
    "alpha",
    "alpha OR beta OR rare",
    "\"shared shared\"",
    "zeta OR argus",
];

/// Run the sealed-segment collection stage on one explicit path and
/// return the finished page.
fn collect_sealed_page(
    index: &QuillIndex,
    cx: &Cx,
    query_text: &str,
    limit: usize,
    offset: usize,
    exact_count: bool,
    fan_out: bool,
) -> CollectedTopDocs {
    let mut parsed = index
        .reader
        .default_parser()
        .expect("fan-out fixture has a default parser")
        .parse_lenient(query_text);
    let _ = canonicalize_query(&mut parsed.query);
    let snapshot = index
        .search_snapshot()
        .expect("fan-out fixture snapshot is authoritative");
    let rank_pruning =
        !exact_count && limit != 0 && query_has_prunable_root_union(&parsed.query, 1.0);
    // DELIBERATELY NOT the shipping executor's route (bd-1i4j4): shipping
    // lowers an exact-count invocation WITHOUT the specialized ranked
    // term-union root, mirroring Tantivy's MultiCollector. This test-only
    // helper keeps the specialized route so internal pruning-safety tests
    // can build a same-route, pruning-free exhaustive REFERENCE page.
    let topdocs_root = limit != 0;
    let mut collector = if exact_count {
        TopDocsCollector::with_exact_count(limit, offset)
    } else {
        TopDocsCollector::new(limit, offset)
    }
    .expect("ranked collector");
    index
        .collect_sealed_segments(
            cx,
            &mut collector,
            &parsed.query,
            &snapshot,
            rank_pruning,
            topdocs_root,
            fan_out,
        )
        .expect("sealed collection");
    collector.finish().expect("finish sealed collection")
}

/// Bit-exact comparison key: global docid plus raw f32 score bits.
fn ranked_page_key(collected: &CollectedTopDocs) -> (Vec<(u32, u32)>, Option<u64>) {
    (
        collected
            .hits
            .iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect(),
        collected.total_count,
    )
}

/// Run the sealed-segment unscored id-set stage on one explicit path.
fn collect_docid_set(index: &QuillIndex, cx: &Cx, query_text: &str, fan_out: bool) -> Vec<u32> {
    let mut parsed = index
        .reader
        .default_parser()
        .expect("fan-out fixture has a default parser")
        .parse_lenient(query_text);
    let _ = canonicalize_query(&mut parsed.query);
    let snapshot = index
        .search_snapshot()
        .expect("fan-out fixture snapshot is authoritative");
    let mut collector = DocSetCollector::new();
    let work_upper_bound = query_work_upper_bound(
        &parsed.query,
        &snapshot,
        index.reader.schema,
        index.reader.config.glob_expansion_limit,
    )
    .expect("bound sealed docid work");
    let checkpoint = QueryCheckpoint::new(
        cx,
        "collect_docids_test",
        index.reader.config.query_fuel_budget,
        work_upper_bound,
    );
    let metering = checkpoint.metering();
    let checkpoint: QueryCheckpointHandle<'_> = checkpoint;
    let prepared_doc_freqs = prepare_snapshot_doc_freqs(&snapshot);
    index
        .reader
        .collect_docids_sealed(
            cx,
            &checkpoint,
            &mut collector,
            &parsed.query,
            &snapshot,
            prepared_doc_freqs.as_ref(),
            fan_out && !metering,
        )
        .expect("sealed docid collection");
    collector.finish()
}

#[test]
fn fanned_docid_collection_matches_serial_exactly() {
    run_with_cx(|cx| async move {
        let concat = concat_merge_fixture_index(&cx).await;
        let seeded = segment_fanout_fixture_index(&cx, 5, 64, 0x0d0c_1d5e_7c01_1ec7).await;
        let fixtures = [
            (&concat, CONCAT_MERGE_QUERIES.as_slice()),
            (&seeded, SEGMENT_FANOUT_QUERIES.as_slice()),
        ];
        for (index, queries) in fixtures {
            for query in queries {
                let serial = collect_docid_set(index, &cx, query, false);
                for round in 0..3 {
                    assert_eq!(
                        collect_docid_set(index, &cx, query, true),
                        serial,
                        "docid fan-out diverged from serial: query={query} round={round}",
                    );
                }
            }
        }
    });
}

#[test]
fn sealed_segment_fanout_gate_thresholds() {
    assert!(!sealed_segment_fanout(0, 0));
    assert!(
        !sealed_segment_fanout(1, SEGMENT_FANOUT_THRESHOLD),
        "single-segment snapshots must never fan out",
    );
    assert!(
        !sealed_segment_fanout(4, SEGMENT_FANOUT_THRESHOLD - 1),
        "small unfragmented corpora stay on the serial path",
    );
    assert!(sealed_segment_fanout(2, SEGMENT_FANOUT_THRESHOLD));
    assert!(sealed_segment_fanout(
        SEGMENT_COUNT_FANOUT_THRESHOLD,
        SEGMENT_FANOUT_THRESHOLD - 1,
    ));
    assert!(!sealed_segment_fanout(
        SEGMENT_COUNT_FANOUT_THRESHOLD - 1,
        SEGMENT_FANOUT_THRESHOLD - 1,
    ));
    assert!(sealed_segment_fanout(16, u64::MAX));
}

#[cfg(feature = "pruning-conformance")]
#[test]
fn pruning_conformance_traced_search_bypasses_warm_cache_without_mutation() {
    run_with_cx(|cx| async move {
        const QUERY: &str = "shared OR alpha";
        const LIMIT: usize = 20;
        const OFFSET: usize = 0;
        const EXACT_COUNT: bool = false;
        let fixtures = [
            (
                1,
                0x5e21_a11c_0000_0001,
                ConformancePruningExecutionMode::Serial,
            ),
            (
                SEGMENT_COUNT_FANOUT_THRESHOLD,
                0x5e21_a11c_0000_0002,
                ConformancePruningExecutionMode::Rayon,
            ),
        ];

        for (segment_count, seed, expected_mode) in fixtures {
            let index = segment_fanout_fixture_index(&cx, segment_count, 64, seed).await;
            let snapshot = index
                .search_snapshot()
                .expect("pruning fixture snapshot is authoritative");
            let snapshot_epoch = snapshot.snapshot_epoch();
            let fingerprint =
                raw_query_cache_fingerprint(snapshot_epoch, QUERY, LIMIT, OFFSET, EXACT_COUNT);
            let expected_doc_counts = index
                .snapshot()
                .expect("pruning fixture snapshot is authoritative")
                .segments()
                .iter()
                .map(|segment| u64::from(segment.doc_count()))
                .collect::<Vec<_>>();
            let cache = &index.reader.published_snapshot.ranked_query_cache;

            assert!(
                cache
                    .get_raw(
                        fingerprint,
                        snapshot_epoch,
                        QUERY,
                        LIMIT,
                        OFFSET,
                        EXACT_COUNT,
                    )
                    .is_none(),
                "fixture must begin cold for {expected_mode:?}",
            );
            let warm = index
                .search_paginated(&cx, QUERY, LIMIT, OFFSET, EXACT_COUNT)
                .expect("ordinary search warms the ranked query cache");
            assert_eq!(
                cache.get_raw(
                    fingerprint,
                    snapshot_epoch,
                    QUERY,
                    LIMIT,
                    OFFSET,
                    EXACT_COUNT,
                ),
                Some(warm.clone()),
                "ordinary search must populate the exact raw-query cache key for {expected_mode:?}",
            );
            let slots_before = cache
                .slots
                .iter()
                .map(ArcSwapOption::load_full)
                .collect::<Vec<_>>();

            let (traced, receipt) = index
                .search_paginated_with_conformance_pruning_trace(
                    &cx,
                    QUERY,
                    LIMIT,
                    OFFSET,
                    EXACT_COUNT,
                )
                .expect("traced search must execute instead of returning from the warm cache");
            assert_eq!(traced, warm);
            assert_eq!(receipt.execution_mode(), expected_mode);
            assert_eq!(receipt.segments().len(), expected_doc_counts.len());
            assert_eq!(
                receipt
                    .segments()
                    .iter()
                    .map(ConformanceSegmentPruningReceipt::segment_ordinal)
                    .collect::<Vec<_>>(),
                (0..u64::try_from(expected_doc_counts.len())
                    .expect("fixture segment count fits u64"))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                receipt
                    .segments()
                    .iter()
                    .map(ConformanceSegmentPruningReceipt::segment_doc_count)
                    .collect::<Vec<_>>(),
                expected_doc_counts,
            );

            let slots_after = cache
                .slots
                .iter()
                .map(ArcSwapOption::load_full)
                .collect::<Vec<_>>();
            assert_eq!(slots_after.len(), slots_before.len());
            for (slot, (before, after)) in slots_before.iter().zip(&slots_after).enumerate() {
                match (before, after) {
                    (Some(before), Some(after)) => assert!(
                        Arc::ptr_eq(before, after),
                        "traced search replaced cache slot {slot} for {expected_mode:?}",
                    ),
                    (None, None) => {}
                    _ => panic! /* ubs:ignore -- Test rejects changed cache occupancy. */ (
                        "traced search changed cache slot occupancy at {slot} for {expected_mode:?}"
                    ),
                }
            }
        }
    });
}

#[cfg(feature = "pruning-conformance")]
#[test]
fn pruning_conformance_invocation_bound_traces_do_not_cross_concurrent_searches() {
    run_with_cx(|cx| async move {
        let index =
            segment_fanout_fixture_index(&cx, SEGMENT_COUNT_FANOUT_THRESHOLD, 64, 0x51A7).await;
        let expected_doc_counts = index
            .snapshot()
            .expect("pruning-conformance fixture snapshot is authoritative")
            .segments()
            .iter()
            .map(|segment| u64::from(segment.doc_count()))
            .collect::<Vec<_>>();
        let query = "shared OR alpha";
        let (first, (second, ordinary)) = rayon::join(
            || {
                index
                    .search_paginated_with_conformance_pruning_trace(&cx, query, 20, 0, false)
                    .expect("first concurrent traced search")
            },
            || {
                rayon::join(
                    || {
                        index
                            .search_paginated_with_conformance_pruning_trace(
                                &cx, query, 20, 0, false,
                            )
                            .expect("second concurrent traced search")
                    },
                    || {
                        index
                            .search_paginated(&cx, query, 20, 0, false)
                            .expect("concurrent ordinary search")
                    },
                )
            },
        );

        assert_eq!(first.0, second.0);
        assert_eq!(first.0, ordinary);
        for receipt in [&first.1, &second.1] {
            assert_eq!(
                receipt.execution_mode(),
                ConformancePruningExecutionMode::Rayon,
            );
            assert_eq!(receipt.segments().len(), expected_doc_counts.len());
            assert_eq!(
                receipt
                    .segments()
                    .iter()
                    .map(ConformanceSegmentPruningReceipt::segment_ordinal)
                    .collect::<Vec<_>>(),
                (0..u64::try_from(expected_doc_counts.len())
                    .expect("fan-out segment count fits u64"))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                receipt
                    .segments()
                    .iter()
                    .map(ConformanceSegmentPruningReceipt::segment_doc_count)
                    .collect::<Vec<_>>(),
                expected_doc_counts,
            );
        }
    });
}

#[cfg(feature = "pruning-conformance")]
#[test]
fn pruning_conformance_cancellation_forces_high_fanout_trace_to_serial() {
    run_with_cx(|cx| async move {
        let index =
            segment_fanout_fixture_index(&cx, SEGMENT_COUNT_FANOUT_THRESHOLD, 64, 0x5A1E).await;
        let controller = index.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::PruningTraceSegmentRecorded, 1)
            .expect("arm high-fanout pruning cancellation");

        let error = index
            .search_paginated_with_conformance_pruning_trace(&cx, "shared OR alpha", 20, 0, false)
            .expect_err("armed high-fanout trace must cancel after its first receipt");
        assert!(matches!(
            error,
            QuillIndexError::Cancelled { phase: "search" }
        ));
        assert_eq!(controller.observed_checkpoints(), 1);
        assert_eq!(controller.recorded_pruning_receipts_at_fire(), 1);
        assert_eq!(
            controller.pruning_trace_execution_mode_at_fire(),
            Some(ConformancePruningExecutionMode::Serial),
            "the injected invocation must replace schedule-dependent Rayon receipt order with serial segment order",
        );
        assert_eq!(controller.discarded_pruning_trace_sessions(), 1);

        controller.disarm();
        cx.set_cancel_requested(false);
        let (_, receipt) = index
            .search_paginated_with_conformance_pruning_trace(&cx, "shared OR alpha", 20, 0, false)
            .expect("unarmed high-fanout replay");
        assert_eq!(
            receipt.execution_mode(),
            ConformancePruningExecutionMode::Rayon,
            "ordinary traced high-fanout searches must retain the shipping Rayon branch",
        );
    });
}

#[cfg(feature = "pruning-conformance")]
#[test]
fn pruning_conformance_stale_session_cannot_checkpoint_rearmed_controller() {
    run_with_cx(|cx| async move {
        let controller = ConformanceCancellationController::default();
        controller
            .arm(ConformanceCancellationStage::PruningTraceSegmentRecorded, 1)
            .expect("arm stale-session generation");
        let stale_generation = controller
            .active_pruning_trace_arm_generation()
            .expect("capture stale pruning generation");
        let stale_session =
            ConformancePruningTraceSession::new_for_cancellation_arm(1, Some(stale_generation));
        stale_session
            .bind_execution_mode(ConformancePruningExecutionMode::Serial)
            .expect("bind stale session");

        controller.disarm();
        controller
            .arm(ConformanceCancellationStage::PruningTraceSegmentRecorded, 1)
            .expect("arm replacement generation");
        assert_eq!(
            stale_session
                .record_and_checkpoint(0, 1, &[], &controller, &cx)
                .expect("record stale receipt without crossing generations"),
            1,
        );
        assert_eq!(controller.observed_checkpoints(), 0);
        assert_eq!(controller.recorded_pruning_receipts_at_fire(), 0);
        assert_eq!(controller.pruning_trace_execution_mode_at_fire(), None);
        assert!(!controller.fired());
        assert!(!cx.is_cancel_requested());
        assert_eq!(
            stale_session
                .complete()
                .expect("complete generation-isolated stale session")
                .execution_mode(),
            ConformancePruningExecutionMode::Serial,
        );
    });
}

#[test]
fn fanned_sealed_collection_matches_serial_exactly() {
    run_with_cx(|cx| async move {
        let concat = concat_merge_fixture_index(&cx).await;
        let seeded = segment_fanout_fixture_index(&cx, 5, 64, 0x9e37_79b9_7f4a_7c15).await;
        let pages = [(10_usize, 0_usize), (3, 4), (1, 0), (0, 0)];
        let fixtures = [
            (&concat, CONCAT_MERGE_QUERIES.as_slice()),
            (&seeded, SEGMENT_FANOUT_QUERIES.as_slice()),
        ];
        for (index, queries) in fixtures {
            for query in queries {
                for (limit, offset) in pages {
                    for exact_count in [false, true] {
                        let serial = ranked_page_key(&collect_sealed_page(
                            index,
                            &cx,
                            query,
                            limit,
                            offset,
                            exact_count,
                            false,
                        ));
                        // Repeat the fanned arm: rayon schedules differ
                        // run to run, and every schedule must reproduce
                        // the serial page bit-exactly.
                        for round in 0..3 {
                            let fanned = ranked_page_key(&collect_sealed_page(
                                index,
                                &cx,
                                query,
                                limit,
                                offset,
                                exact_count,
                                true,
                            ));
                            assert_eq!(
                                fanned, serial,
                                "fan-out diverged from serial: query={query} \
                                 limit={limit} offset={offset} \
                                 exact_count={exact_count} round={round}",
                            );
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn snapshot_publication_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<DeltaSnapshot>();
    assert_send_sync::<QuillSearchSnapshot>();
    assert_send_sync::<SnapshotPublisher>();
    assert_send_sync::<QuillIndex>();
}

#[test]
fn repeated_doc_frequency_cursor_and_search_calls_reuse_validated_termdict_metadata() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        index
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("first", "alpha shared"),
                    IndexableDocument::new("second", "beta shared"),
                    IndexableDocument::new("third", "gamma"),
                ],
            )
            .await
            .expect("index cache fixture");
        index.commit(&cx).await.expect("seal cache fixture");

        let snapshot = index
            .snapshot()
            .expect("cache fixture snapshot is authoritative");
        assert_eq!(snapshot.segments().len(), 1);
        let segment = &snapshot.segments()[0];
        let (full_validations_before, borrowed_views_before) =
            segment.term_dictionary_cache_counts();
        assert_eq!(full_validations_before, 1);

        assert_eq!(
            snapshot_doc_freq(&snapshot, DEFAULT_SCHEMA, CONTENT_FIELD, b"shared")
                .expect("cached doc frequency"),
            2
        );
        let term_count = usize::try_from(
            segment
                .term_dictionary(DEFAULT_SCHEMA)
                .expect("cached cursor dictionary")
                .term_count(),
        )
        .expect("fixture term count fits usize");
        let cursor_rows = segment
            .term_dictionary(DEFAULT_SCHEMA)
            .expect("cached cursor dictionary")
            .cursor()
            .and_then(|cursor| cursor.collect_bounded(term_count))
            .expect("collect cached cursor");
        assert!(!cursor_rows.is_empty());

        for _ in 0..32 {
            let hits = index
                .search_doc_ids(&cx, "shared", 10)
                .expect("repeat cached search");
            assert_eq!(hits.len(), 2);
        }
        let (full_validations_after, borrowed_views_after) = segment.term_dictionary_cache_counts();
        assert_eq!(
            full_validations_after, 1,
            "no query path may repeat complete TERMDICT validation"
        );
        assert_eq!(
            borrowed_views_after,
            borrowed_views_before.saturating_add(7),
            "doc-frequency, two cursor opens, and four field-specific views for the first search must be borrowed; the remaining 31 searches must use the immutable-snapshot result cache"
        );
    });
}

#[test]
fn tombstone_only_delete_commit_reuses_termdict_metadata_and_preserves_results() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        index
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("first", "alpha shared"),
                    IndexableDocument::new("second", "beta shared"),
                    IndexableDocument::new("third", "gamma only"),
                ],
            )
            .await
            .expect("index rebind fixture");
        index.commit(&cx).await.expect("seal rebind fixture");

        let before_snapshot = index
            .snapshot()
            .expect("rebind fixture snapshot is authoritative");
        assert_eq!(before_snapshot.segments().len(), 1);
        let segment = &before_snapshot.segments()[0];
        assert_eq!(segment.term_dictionary_cache_counts().0, 1);
        assert_eq!(segment.term_dictionary_metadata_reuse_count(), 0);
        let metadata_bytes = segment.term_dictionary_metadata_payload_bytes();
        assert!(metadata_bytes > 0, "validated metadata must be accounted");
        let baseline = index
            .search_paginated(&cx, "shared", 10, 0, true)
            .expect("baseline ranked search")
            .hits
            .iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(baseline.len(), 2);

        // Deleting the only non-matching document publishes a
        // tombstone-only successor generation over the same immutable
        // segment backing.
        assert!(
            index
                .delete_document(&cx, "third")
                .await
                .expect("tombstone delete")
        );

        let after_snapshot = index
            .snapshot()
            .expect("tombstone successor snapshot is authoritative");
        assert_eq!(after_snapshot.segments().len(), 1);
        let rebound = &after_snapshot.segments()[0];
        assert_eq!(
            rebound.term_dictionary_cache_counts().0,
            1,
            "a tombstone-only successor generation must not re-validate TERMDICT"
        );
        assert!(
            rebound.term_dictionary_metadata_reuse_count() >= 1,
            "the successor generation must reuse the validated metadata"
        );
        assert_eq!(
            rebound.term_dictionary_metadata_payload_bytes(),
            metadata_bytes,
            "reuse must not grow persistent metadata bytes"
        );
        let after = index
            .search_paginated(&cx, "shared", 10, 0, true)
            .expect("post-rebind ranked search")
            .hits
            .iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(
            after, baseline,
            "ranked hits and exact score bits must be byte-identical across \
             a tombstone-only rebind that does not touch matching documents"
        );
    });
}

/// bd-quill-shipping-conformance-parse-split-w7bsu: Quill answers
/// `A AND NOT B` correctly, and now DIVERGES from the pinned oracle.
///
/// THE TRIPWIRE FIRED AS DESIGNED. This test previously asserted the
/// opposite — that Quill shared tantivy 0.26.1's lowering and returned
/// nothing — and its message said that the day Quill's path was repaired,
/// whoever did it "must register the divergence they have just created
/// rather than discover it in a campaign". That is what happened: removing
/// `wrap_not_for_and` (query.rs) turned this red, and DIV-010 was written
/// before the change landed.
///
/// The assertions are inverted rather than deleted so the arc stays legible:
/// bd-eeq0q measured the agreement and could therefore repair the oracle's
/// SHIPPING path without a register entry; this bead broke the agreement
/// deliberately and paid for it with one.
///
/// Quill is the measured SUBJECT and the gauntlet observes it through
/// `search_paginated`, the same public surface users call — so unlike
/// `frankensearch-lexical`, there is no second role here to keep
/// bit-faithful, and no shipping/conformance split to build. Repairing the
/// one path IS the repair.
#[test]
fn quill_answers_and_not_correctly_and_diverges_from_the_pinned_oracle() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("bd-eeq0q index");
        for (id, content) in [("p1", "alpha beta"), ("p2", "alpha gamma")] {
            LexicalWrite::index_document(&index, &cx, &IndexableDocument::new(id, content))
                .await
                .expect("index bd-eeq0q document");
        }
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit bd-eeq0q documents");

        let ids = |query: &'static str| {
            let index = &index;
            let cx = &cx;
            async move {
                let mut ids = LexicalRead::search(index, cx, query, 10)
                    .await
                    .expect("bd-eeq0q Quill search")
                    .into_iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>();
                ids.sort();
                ids
            }
        };

        // The spellings Quill always got right, unchanged by the repair.
        assert_eq!(ids("alpha NOT beta").await, vec!["p2".to_owned()]);
        assert_eq!(ids("alpha -beta").await, vec!["p2".to_owned()]);

        // Every AND-NOT spelling now means "must alpha, must not beta".
        for query in [
            "alpha AND NOT beta",
            "(alpha AND NOT beta)",
            "(alpha AND NOT beta)^2",
            "alpha AND NOT beta AND NOT delta",
        ] {
            assert_eq!(
                ids(query).await,
                vec!["p2".to_owned()],
                "{query:?} must return A-minus-B on Quill"
            );
        }

        // The leading-NOT form is repaired by the same deletion: it used to
        // go through the same wrapper from `parse_and`'s first operand.
        assert_eq!(
            ids("NOT beta AND alpha").await,
            vec!["p2".to_owned()],
            "a leading NOT under an explicit AND must exclude, not empty the conjunction"
        );

        // A purely negative conjunction still has no positive term to match,
        // which is the same answer the oracle gives and is not part of this
        // repair. Pinned so a later change to the OR/negative handling
        // cannot silently turn exclusion-only queries into match-all.
        assert!(
            ids("NOT alpha AND NOT beta").await.is_empty(),
            "an exclusion-only conjunction must still match nothing"
        );
    });
}
