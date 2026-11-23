use super::*;

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_search_binds_the_ordinary_snapshot_and_cache_disposition() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("profiled index directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create profiled writer");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new("first", "profiled alpha"),
        )
        .await
        .expect("stage profiled document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish profiled document");
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open profiled reader");
        let expected_generation = reader.keeper_generation();
        let outcome = reader
            .search_paginated_with_profile(&cx, "alpha", 10, 0, false)
            .expect("profiled ordinary search");
        assert!(
            matches!(outcome, QuillProfiledSearchOutcome::Completed { .. }),
            "profiled ordinary search unexpectedly failed"
        );
        let Some((result, receipt)) = (match outcome {
            QuillProfiledSearchOutcome::Completed { result, receipt } => Some((result, receipt)),
            QuillProfiledSearchOutcome::Failed { .. } => None,
        }) else {
            return;
        };
        assert_eq!(result.hits.len(), 1);
        assert_eq!(receipt.keeper_generation(), expected_generation);
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Miss);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Serial));
        let Some((work_upper_bound, _metering)) = receipt.work_plan() else {
            panic!("profiled ordinary query did not bind its work plan"); // ubs:ignore -- Test requires its profiling work plan.
        };
        assert!(work_upper_bound > 0);
        assert_eq!(
            receipt.counters().0,
            2,
            "the default parser expands a bare term over both default text fields"
        );
        assert_eq!(receipt.counters().1, 2);
        assert_eq!(receipt.counters().2, 4);
        assert!(
            receipt.counters().4 > 0,
            "profiled ordinary query did not record any accepted work checkpoints"
        );
        assert_eq!(
            receipt.counters().3,
            1,
            "profiled ordinary query must record its one sealed lowering"
        );
        assert_eq!(receipt.outcome(), QuillProfileOutcome::Completed);

        let repeat = reader
            .search_paginated_with_profile(&cx, "alpha", 10, 0, false)
            .expect("repeat profiled ordinary search");
        let repeat_receipt = match repeat {
            QuillProfiledSearchOutcome::Completed { receipt, .. } => receipt,
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("repeat profiled ordinary search unexpectedly failed: {error}") // ubs:ignore -- Failed search must fail this positive test.
            }
        };
        assert_eq!(repeat_receipt.cache(), QuillProfileCacheDisposition::Hit);
        assert_eq!(repeat_receipt.fanout_eligible(), None);
        assert_eq!(repeat_receipt.execution(), None);
        assert_eq!(repeat_receipt.work_plan(), None);
        assert_eq!(repeat_receipt.counters(), (0, 0, 0, 0, 0));
        assert_eq!(
            repeat_receipt.work_units(),
            QuillProfileWorkUnits::default()
        );
        assert_eq!(repeat_receipt.outcome(), QuillProfileOutcome::Completed);
    });
}

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_search_preserves_precheck_cancellation_without_work() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("cancelled profile directory");
        let _writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create cancelled profile writer");
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open cancelled profile reader");
        let cancelled = cx.clone();
        cancelled.set_cancel_requested(true);
        let outcome = reader
            .search_paginated_with_profile(&cancelled, "alpha", 10, 0, false)
            .expect("profile admission must preserve ordinary cancellation");
        let (error, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { .. } => {
                panic!("pre-cancelled profiled search unexpectedly completed") // ubs:ignore -- Ignored cancellation must fail this test.
            }
            QuillProfiledSearchOutcome::Failed { error, receipt } => (error, receipt),
        };
        assert!(matches!(
            error,
            QuillIndexError::Cancelled { phase } if phase == "search"
        ));
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::NotChecked);
        assert_eq!(receipt.fanout_eligible(), None);
        assert_eq!(receipt.execution(), None);
        assert_eq!(receipt.work_plan(), None);
        assert_eq!(receipt.counters(), (0, 0, 0, 0, 0));
        assert_eq!(receipt.work_units(), QuillProfileWorkUnits::default());
        assert_eq!(receipt.cancellation_observations(), 1);
        assert_eq!(receipt.outcome(), QuillProfileOutcome::Cancelled);
    });
}

#[cfg(all(feature = "profile-internals", feature = "conformance-internals"))]
#[test]
fn profiled_search_records_disabled_cache_without_skipping_ordinary_work() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("disabled-cache profile directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create disabled-cache profile writer");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new("first", "profiled alpha"),
        )
        .await
        .expect("stage disabled-cache profile document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish disabled-cache profile segment");
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open disabled-cache profile reader");
        let controller = reader.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::CommitPublication, 1)
            .expect("arm unrelated conformance stage to disable ranked cache");
        let outcome = reader
            .search_paginated_with_profile(&cx, "alpha", 10, 0, false)
            .expect("disabled-cache profiled search");
        controller.disarm();
        let (result, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { result, receipt } => (result, receipt),
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("disabled-cache profiled search unexpectedly failed: {error}") // ubs:ignore -- Failed search must fail this positive test.
            }
        };
        assert_eq!(result.hits.len(), 1);
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Disabled);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Serial));
        assert!(receipt.work_plan().is_some());
        assert_eq!(receipt.counters().0, 2);
        assert_eq!(receipt.counters().1, 2);
        assert_eq!(receipt.counters().2, 4);
        assert_eq!(receipt.counters().3, 1);
        assert!(receipt.counters().4 > 0);
        assert_eq!(receipt.outcome(), QuillProfileOutcome::Completed);
    });
}

#[cfg(all(feature = "profile-internals", feature = "conformance-internals"))]
#[test]
fn profiled_search_reports_exact_prefix_on_checkpoint_cancellation() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("checkpoint-cancel profile directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create checkpoint-cancel profile writer");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new("first", "profiled alpha"),
        )
        .await
        .expect("stage checkpoint-cancel profile document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish checkpoint-cancel profile segment");
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open checkpoint-cancel profile reader");
        let controller = reader.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::QueryCollection, 2)
            .expect("arm cancellation at the second ordinary checkpoint");
        let outcome = reader
            .search_paginated_with_profile(&cx, "alpha", 10, 0, false)
            .expect("checkpoint cancellation must retain profile receipt");
        controller.disarm();
        let (error, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { .. } => {
                panic!("checkpoint-cancelled profiled search unexpectedly completed") // ubs:ignore -- Ignored cancellation must fail this test.
            }
            QuillProfiledSearchOutcome::Failed { error, receipt } => (error, receipt),
        };
        assert!(matches!(error, QuillIndexError::Cancelled { phase } if phase == "search"));
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Disabled);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Serial));
        assert!(receipt.work_plan().is_some());
        assert_eq!(receipt.work_units().requested(), [1, 1, 0, 0]);
        assert_eq!(receipt.work_units().admitted(), [1, 0, 0, 0]);
        assert_eq!(receipt.work_units().refused(), [0, 0, 0, 0]);
        assert_eq!(receipt.counters(), (1, 1, 1, 1, 1));
        assert_eq!(receipt.cancellation_observations(), 1);
        assert_eq!(receipt.outcome(), QuillProfileOutcome::Cancelled);
    });
}

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_search_returns_fuel_exhaustion_with_work_disposition() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("fuel profile directory");
        let config = QuillConfig {
            query_fuel_budget: 1,
            deterministic_ingest: true,
            ..QuillConfig::default()
        };
        let writer = QuillIndex::create(&cx, directory.path(), config.clone())
            .await
            .expect("create fuel profile writer");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new("first", "profiled alpha"),
        )
        .await
        .expect("stage fuel profile document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish fuel profile segment");
        let reader = QuillSearchIndex::open(&cx, directory.path(), config)
            .await
            .expect("open fuel profile reader");
        let outcome = reader
            .search_paginated_with_profile(&cx, "alpha", 10, 0, false)
            .expect("profile admission must preserve fuel exhaustion");
        let (error, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { .. } => {
                panic!("fuel-limited profiled search unexpectedly completed") // ubs:ignore -- Ignored fuel exhaustion must fail this test.
            }
            QuillProfiledSearchOutcome::Failed { error, receipt } => (error, receipt),
        };
        assert!(matches!(
            error,
            QuillIndexError::QueryFuelExhausted {
                budget: 1,
                consumed: 1,
                ..
            }
        ));
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Miss);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Serial));
        let Some((work_upper_bound, metering)) = receipt.work_plan() else {
            panic!("fuel-limited profiled search did not bind work plan") // ubs:ignore -- Test requires its profiling work plan.
        };
        assert!(work_upper_bound > 1);
        assert!(metering);
        assert_eq!(receipt.work_units().requested(), [1, 1, 0, 0]);
        assert_eq!(receipt.work_units().admitted(), [1, 0, 0, 0]);
        assert_eq!(receipt.work_units().refused(), [0, 1, 0, 0]);
        assert_eq!(receipt.counters(), (1, 1, 1, 1, 1));
        assert_eq!(receipt.cancellation_observations(), 0);
        assert_eq!(receipt.outcome(), QuillProfileOutcome::FuelExhausted);
    });
}

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_empty_snapshot_has_no_sealed_execution_branch() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("empty profile directory");
        let _writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create empty profile writer");
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open empty profile reader");
        let outcome = reader
            .search_paginated_with_profile(&cx, "alpha", 10, 0, false)
            .expect("execute empty profiled search");
        let (result, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { result, receipt } => (result, receipt),
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("empty profiled search unexpectedly failed: {error}") // ubs:ignore -- Failed search must fail this positive test.
            }
        };
        assert!(result.hits.is_empty());
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Miss);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(receipt.execution(), None);
        assert!(receipt.work_plan().is_some());
        assert_eq!(receipt.counters(), (0, 0, 0, 0, 0));
        assert_eq!(receipt.work_units(), QuillProfileWorkUnits::default());
        assert_eq!(receipt.outcome(), QuillProfileOutcome::Completed);
    });
}

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_exact_term_counter_algebra_tracks_two_sealed_segments() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("two-segment profile directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create two-segment profile writer");
        for document in [
            IndexableDocument::new("first", "profiled alpha first"),
            IndexableDocument::new("second", "profiled alpha second"),
        ] {
            LexicalWrite::index_document(&writer, &cx, &document)
                .await
                .expect("stage two-segment profile document");
            LexicalWrite::commit(&writer, &cx)
                .await
                .expect("publish one sealed profile segment");
        }
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open two-segment profile reader");
        let outcome = reader
            // Field-qualified on purpose: see the fragmented-snapshot test.
            // A bare term fans out over both parser default fields, which
            // doubles every counter independently of segment fan-out.
            .search_paginated_with_profile(&cx, "content:alpha", 10, 0, false)
            .expect("execute two-segment profiled search");
        let (result, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { result, receipt } => (result, receipt),
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("two-segment profiled search unexpectedly failed: {error}") // ubs:ignore -- Failed search must fail this positive test.
            }
        };
        assert_eq!(result.hits.len(), 2);
        assert_eq!(receipt.segment_counts(), (2, 0));
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Miss);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Serial));
        assert_eq!(receipt.counters().0, 1);
        assert_eq!(receipt.counters().1, 2);
        assert_eq!(receipt.counters().2, 4);
        assert_eq!(receipt.counters().3, 2);
        assert!(receipt.counters().4 > 0);
    });
}

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_fragmented_snapshot_reuses_snapshot_term_statistics() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("fragmented profile directory");
        // The fixture needs SEGMENT_COUNT_FANOUT_THRESHOLD segments to stay
        // LIVE so the search actually fans out. `deterministic_config()`
        // leaves tier_fanout at DEFAULT_TIER_FANOUT, which is the same 8, so
        // the eighth commit tripped the same-tier merge and the receipt
        // truthfully reported one segment. Give the tier policy headroom,
        // exactly as segment_fanout_fixture_index does for the non-profiled
        // fan-out tests.
        let fragmented_config = QuillConfig {
            tier_fanout: SEGMENT_COUNT_FANOUT_THRESHOLD
                .checked_add(2)
                .expect("fan-out threshold leaves room for its tier policy"),
            ..deterministic_config()
        };
        let writer = QuillIndex::create(&cx, directory.path(), fragmented_config.clone())
            .await
            .expect("create fragmented profile writer");
        for ordinal in 0..SEGMENT_COUNT_FANOUT_THRESHOLD {
            let document = IndexableDocument::new(
                format!("fragmented-{ordinal}"),
                format!("profiled alpha fragmented {ordinal}"),
            );
            LexicalWrite::index_document(&writer, &cx, &document)
                .await
                .expect("stage fragmented profile document");
            LexicalWrite::commit(&writer, &cx)
                .await
                .expect("publish fragmented profile segment");
        }
        let reader = QuillSearchIndex::open(&cx, directory.path(), fragmented_config)
            .await
            .expect("open fragmented profile reader");
        let outcome = reader
            // Field-qualified on purpose: a bare term expands across BOTH
            // parser default fields (content and title), so every counter
            // below would be doubled by field fan-out rather than by
            // segment fan-out. The counter algebra this test documents is
            // the single-field model: one snapshot-global scan plus one
            // local cursor opening per segment.
            .search_paginated_with_profile(&cx, "content:alpha", 16, 0, false)
            .expect("execute fragmented profiled search");
        let (result, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { result, receipt } => (result, receipt),
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("fragmented profiled search unexpectedly failed: {error}") // ubs:ignore -- Failed search must fail this positive test.
            }
        };
        let expected_segments = u64::try_from(SEGMENT_COUNT_FANOUT_THRESHOLD)
            .expect("fan-out threshold fits the profile receipt");
        assert_eq!(result.hits.len(), SEGMENT_COUNT_FANOUT_THRESHOLD);
        for (ordinal, hit) in result.hits.iter().enumerate() {
            assert_eq!(hit.document_id, format!("fragmented-{ordinal}"));
        }
        assert!(
            result
                .hits
                .windows(2)
                .all(|hits| hits[0].score.to_bits() == hits[1].score.to_bits()),
            "equal-length exact-term rows must retain identical BM25 score bits"
        );
        assert_eq!(receipt.segment_counts(), (expected_segments, 0));
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Miss);
        assert_eq!(receipt.fanout_eligible(), Some(true));
        assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Rayon));
        assert_eq!(receipt.counters().0, 1);
        assert_eq!(receipt.counters().1, expected_segments);
        assert_eq!(receipt.counters().2, expected_segments * 2);
        assert_eq!(receipt.counters().3, expected_segments);

        let successor_ordinal = SEGMENT_COUNT_FANOUT_THRESHOLD;
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new(
                format!("fragmented-{successor_ordinal}"),
                format!("profiled alpha fragmented {successor_ordinal}"),
            ),
        )
        .await
        .expect("stage successor fragmented profile document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish successor fragmented profile segment");
        assert!(
            reader
                .refresh(&cx)
                .await
                .expect("refresh fragmented profile reader"),
            "the read handle must adopt the successor publication"
        );
        let successor = reader
            .search_paginated_with_profile(&cx, "content:alpha", 16, 0, false)
            .expect("execute successor fragmented profiled search");
        let (successor_result, successor_receipt) = match successor {
            QuillProfiledSearchOutcome::Completed { result, receipt } => (result, receipt),
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("successor fragmented profile search unexpectedly failed: {error}") // ubs:ignore -- Failed successor must fail this positive test.
            }
        };
        let successor_segments = expected_segments + 1;
        assert_eq!(successor_result.hits.len(), successor_ordinal + 1);
        for (ordinal, hit) in successor_result.hits.iter().enumerate() {
            assert_eq!(hit.document_id, format!("fragmented-{ordinal}"));
        }
        assert_eq!(successor_receipt.segment_counts(), (successor_segments, 0));
        assert_eq!(
            successor_receipt.cache(),
            QuillProfileCacheDisposition::Miss
        );
        assert_eq!(successor_receipt.fanout_eligible(), Some(true));
        assert_eq!(
            successor_receipt.execution(),
            Some(QuillProfileExecutionMode::Rayon)
        );
        assert_eq!(successor_receipt.counters().0, 1);
        assert_eq!(successor_receipt.counters().1, successor_segments);
        assert_eq!(successor_receipt.counters().2, successor_segments * 2);
        assert_eq!(successor_receipt.counters().3, successor_segments);
    });
}

#[test]
fn snapshot_term_statistics_allocation_failure_refuses_before_dictionary_work() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("DF allocation fixture directory");
        let config = QuillConfig {
            tier_fanout: 3,
            ..deterministic_config()
        };
        let writer = QuillIndex::create(&cx, directory.path(), config)
            .await
            .expect("create DF allocation fixture");
        for ordinal in 0..2 {
            LexicalWrite::index_document(
                &writer,
                &cx,
                &IndexableDocument::new(format!("allocation-{ordinal}"), "alpha beta"),
            )
            .await
            .expect("stage DF allocation document");
            LexicalWrite::commit(&writer, &cx)
                .await
                .expect("publish DF allocation segment");
        }
        let snapshot = writer.search_snapshot().expect("DF allocation snapshot");
        assert_eq!(snapshot.keeper_snapshot().segments().len(), 2);
        for site in 0..4 {
            for metered in [false, true] {
                let mut prepared = PreparedSnapshotDocFreqs::new();
                if site == 3 {
                    let seed: QueryCheckpointHandle<'_> =
                        QueryCheckpoint::new(&cx, "seed_df", 2, 3);
                    assert_eq!(
                        prepared
                            .get_or_compute(&seed, &snapshot, CONTENT_FIELD, b"beta")
                            .expect("seed the existing field map"),
                        2
                    );
                }
                // Capacity overflow forces the real fallible reservation
                // path without a process-wide allocator or memory pressure.
                match site {
                    0 => prepared.reservation_overrides.term = Some(usize::MAX),
                    1 => prepared.reservation_overrides.outer_map = Some(usize::MAX),
                    2 => prepared.reservation_overrides.new_field_map = Some(usize::MAX),
                    _ => prepared.reservation_overrides.existing_field_map = Some(usize::MAX),
                }
                let checkpoint = if metered {
                    QueryCheckpoint::new(&cx, "failed_df", 0, 1)
                } else {
                    QueryCheckpoint::new(&cx, "failed_df", 2, 2)
                };
                assert_eq!(checkpoint.metering(), metered);
                let checkpoint_handle: QueryCheckpointHandle<'_> = checkpoint.clone();
                let error = prepared
                    .get_or_compute(&checkpoint_handle, &snapshot, CONTENT_FIELD, b"alpha")
                    .expect_err("allocation failure must never fall back to an uncached scan");
                let QuillIndexError::Argus(ArgusError::Allocation { resource, count }) = error
                else {
                    panic!("expected allocation refusal before dictionary admission: {error}"); // ubs:ignore -- Test requires the exact allocation refusal.
                };
                assert_eq!(
                    (resource, count),
                    if site == 0 {
                        ("snapshot document-frequency term", 5)
                    } else {
                        ("snapshot document-frequency cache", 1)
                    }
                );
                assert_eq!(checkpoint.state.consumed.load(Ordering::Acquire), 0);
                assert_eq!(
                    checkpoint.state.dictionary_blocks.load(Ordering::Acquire),
                    0
                );
                {
                    let entries = prepared.entries.lock().expect("unpoisoned cache");
                    let field = entries.get(&CONTENT_FIELD);
                    let alpha = field
                        .and_then(|values| values.get(b"alpha".as_slice()))
                        .copied();
                    let beta = field
                        .and_then(|values| values.get(b"beta".as_slice()))
                        .copied();
                    drop(entries);
                    assert!(alpha.is_none());
                    if site == 3 {
                        assert_eq!(beta, Some(2));
                    }
                }
                prepared.reservation_overrides = SnapshotDocFreqReservationOverrides::default();
                let retry = QueryCheckpoint::new(&cx, "retry_df", 2, 3);
                let retry_handle: QueryCheckpointHandle<'_> = retry.clone();
                for _ in 0..2 {
                    assert_eq!(
                        prepared
                            .get_or_compute(&retry_handle, &snapshot, CONTENT_FIELD, b"alpha")
                            .expect("retry and cache hit retain the correct document frequency"),
                        2
                    );
                }
                assert_eq!(retry.state.dictionary_blocks.load(Ordering::Acquire), 2);
            }
        }
    });
}

#[test]
fn cached_snapshot_term_statistics_charge_no_repeat_dictionary_scan() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("cached DF fuel directory");
        let config = QuillConfig {
            tier_fanout: 3,
            ..deterministic_config()
        };
        let writer = QuillIndex::create(&cx, directory.path(), config)
            .await
            .expect("create cached DF fuel writer");
        for ordinal in 0..2 {
            LexicalWrite::index_document(
                &writer,
                &cx,
                &IndexableDocument::new(format!("fuel-{ordinal}"), "alpha"),
            )
            .await
            .expect("stage cached DF fuel document");
            LexicalWrite::commit(&writer, &cx)
                .await
                .expect("publish cached DF fuel segment");
        }
        let snapshot = writer
            .search_snapshot()
            .expect("cached DF fuel snapshot is authoritative");
        assert_eq!(snapshot.keeper_snapshot().segments().len(), 2);
        let prepared = PreparedSnapshotDocFreqs::new();
        // `upper_bound > budget` keeps metering live, so every assertion
        // below is about admissions that are really being counted. The
        // budget is exactly one snapshot scan: two sealed segments, one
        // dictionary block each.
        let checkpoint: QueryCheckpointHandle<'_> =
            QueryCheckpoint::new(&cx, "cached_df_fuel", 2, 3);

        assert_eq!(
            prepared
                .get_or_compute(&checkpoint, &snapshot, CONTENT_FIELD, b"alpha")
                .expect("first snapshot DF scan is admitted"),
            2
        );
        // Repeats open no dictionary, so they must not spend the budget a
        // second time. Charging them was what kept the per-query table
        // from bounding anything in #41: the table saved the reads while
        // the fuel still grew as `terms x segments^2`.
        for repeat in 0..8 {
            assert_eq!(
                prepared
                    .get_or_compute(&checkpoint, &snapshot, CONTENT_FIELD, b"alpha")
                    .unwrap_or_else(|error| panic! /* ubs:ignore -- Test requires repeat lookup within fuel. */ (
                        "cached repeat {repeat} must not charge a second scan: {error}"
                    )),
                2
            );
        }
        // The budget is genuinely spent, not merely unmetered: one more
        // real dictionary block is refused.
        let error = checkpoint
            .admit(QueryWorkKind::DictionaryBlock, 1)
            .expect_err("the single scan must have consumed the whole budget");
        assert!(matches!(
            error,
            ArgusError::QueryFuelExhausted {
                budget: 2,
                consumed: 2,
                dictionary_blocks: 2,
                ..
            }
        ));
    });
}

/// Publish `commits` single-document segments and return the query-work
/// ceiling of one field-qualified term against them.
///
/// Every term of that ceiling is linear in the segment count for this
/// query shape -- one segment charge, one posting traversal pair per
/// segment, one statistics scan plus one local probe per segment -- so the
/// ceiling itself must be.
async fn fragmented_term_query_ceiling(cx: &Cx, commits: usize) -> u64 {
    let index = QuillIndex::in_memory(QuillConfig {
        // Well clear of `commits`, so each publication keeps its own
        // segment instead of being folded into a same-tier merge.
        tier_fanout: 32,
        ..deterministic_config()
    })
    .expect("create fragmented ceiling index");
    for ordinal in 0..commits {
        index
            .index_documents(
                cx,
                &[IndexableDocument::new(
                    format!("ceiling-{ordinal}"),
                    "alpha",
                )],
            )
            .await
            .expect("stage fragmented ceiling document");
        index
            .commit(cx)
            .await
            .expect("publish fragmented ceiling segment");
    }
    let snapshot = index
        .search_snapshot()
        .expect("fragmented ceiling snapshot is authoritative");
    assert_eq!(
        snapshot.keeper_snapshot().segments().len(),
        commits,
        "the fixture must keep one sealed segment per commit"
    );
    assert_eq!(snapshot.delta_count(), 0, "every commit must seal");
    let mut parsed = index
        .reader
        .default_parser()
        .expect("fragmented ceiling parser")
        // Field-qualified on purpose: a bare term fans out across both
        // default fields and doubles every count below without saying
        // anything about segment scaling.
        .parse_lenient("content:alpha");
    let _ = canonicalize_query(&mut parsed.query);
    query_work_upper_bound(
        &parsed.query,
        &snapshot,
        index.reader.schema,
        index.reader.config.glob_expansion_limit,
    )
    .expect("bound fragmented ceiling query")
}

#[test]
fn the_query_ceiling_stays_linear_in_segment_count_issue_41() {
    run_with_cx(|cx| async move {
        let two = fragmented_term_query_ceiling(&cx, 2).await;
        let four = fragmented_term_query_ceiling(&cx, 4).await;
        let eight = fragmented_term_query_ceiling(&cx, 8).await;

        // Pinned so the ratio below cannot be satisfied by two wrong
        // numbers. For `k` single-document segments and one stream the
        // ceiling is `k` segment charges, `2k` posting blocks (one
        // traversal pair per segment) and `2k` dictionary blocks (one
        // snapshot scan plus one local probe per segment): `5k`. The
        // pre-#41 ceiling was `k^2 + 4k`, so these three would read
        // 12, 32 and 96.
        assert_eq!((two, four, eight), (10, 20, 40));

        // `PreparedSnapshotDocFreqs` computes a term's snapshot statistics
        // once, so the ceiling charges one scan per stream rather than one
        // per lowered segment. Doubling the segment count must therefore
        // double the ceiling exactly. The pre-#41 ceiling carried
        // `streams x segments^2`, which quadrupled it: 2 -> 4 -> 8
        // segments produced 10 -> 24 -> 64 statistics units instead of
        // 4 -> 8 -> 16, and a CASS archive at 473 segments spent a 10M
        // budget on statistics it had already computed.
        assert_eq!(
            four,
            two * 2,
            "doubling 2 -> 4 segments must double the ceiling ({two} -> {four})"
        );
        assert_eq!(
            eight,
            four * 2,
            "doubling 4 -> 8 segments must double the ceiling ({four} -> {eight})"
        );
    });
}

#[test]
fn read_only_search_handle_coexists_with_writer_and_pins_publication() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("index directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable writer");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new("first", "published alpha"),
        )
        .await
        .expect("stage first document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish first document");

        let pinned = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open read-only handle while writer lease is live");
        assert_eq!(pinned.doc_count().expect("read pinned count"), 1);
        let first_alpha_page = pinned
            .search_doc_ids(&cx, "alpha", 10)
            .expect("search pinned publication");
        let first_alpha_identity = first_alpha_page
            .iter()
            .map(|hit| {
                (
                    hit.document_id.clone(),
                    hit.global_docid,
                    hit.score.to_bits(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(first_alpha_identity[0].0, "first");
        let cached_alpha_page = pinned
            .search_doc_ids(&cx, "alpha", 10)
            .expect("repeat pinned search through cache");
        assert_eq!(
            cached_alpha_page
                .iter()
                .map(|hit| {
                    (
                        hit.document_id.clone(),
                        hit.global_docid,
                        hit.score.to_bits(),
                    )
                })
                .collect::<Vec<_>>(),
            first_alpha_identity,
        );
        assert!(
            Arc::ptr_eq(&first_alpha_page, &cached_alpha_page),
            "a repeated public cache hit must share the immutable winner page"
        );
        let cancelled = Cx::for_testing();
        cancelled.set_cancel_requested(true);
        assert!(matches!(
            pinned.search_doc_ids(&cancelled, "alpha", 10),
            Err(QuillIndexError::Cancelled { phase }) if phase == "search"
        ));

        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new("second", "published beta"),
        )
        .await
        .expect("stage second document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish second document");

        let still_pinned_alpha_page = pinned
            .search_doc_ids(&cx, "alpha", 10)
            .expect("old handle remains pinned");
        assert!(Arc::ptr_eq(&first_alpha_page, &still_pinned_alpha_page));
        assert!(
            pinned
                .search_doc_ids(&cx, "beta", 10)
                .expect("old handle excludes successor")
                .is_empty()
        );
        assert!(pinned.refresh(&cx).await.expect("refresh publication"));
        assert_eq!(pinned.doc_count().expect("read refreshed count"), 2);
        let refreshed_alpha_page = pinned
            .search_doc_ids(&cx, "alpha", 10)
            .expect("refreshed handle rebuilds the alpha page");
        assert!(
            !Arc::ptr_eq(&first_alpha_page, &refreshed_alpha_page),
            "publication must invalidate the prior snapshot's cached page"
        );
        assert_eq!(
            first_alpha_page
                .iter()
                .map(|hit| {
                    (
                        hit.document_id.clone(),
                        hit.global_docid,
                        hit.score.to_bits(),
                    )
                })
                .collect::<Vec<_>>(),
            first_alpha_identity,
            "a retained page must remain readable and byte-stable after publication"
        );
        assert_eq!(
            pinned
                .search_doc_ids(&cx, "beta", 10)
                .expect("refreshed handle observes successor")[0]
                .document_id,
            "second"
        );
        assert!(!pinned.refresh(&cx).await.expect("idempotent refresh"));
        assert!(
            pinned
                .segment_stats()
                .expect("read pinned segment stats")
                .live_writer
        );
    });
}
