use super::*;

#[test]
fn parallel_budget_default_bounds_reused_generation_reservations() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker default-budget pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("default-budget index");
            let documents = parallel_budget_fixture_documents(256);
            let writer = index.writer_mut();
            if writer.shard_router.shard_count() < 2 {
                return;
            }
            let initial_reserved = writer
                .shards
                .iter()
                .try_fold(0_usize, |total, shard| {
                    total.checked_add(shard.accumulator.bytes_reserved())
                })
                .expect("initial reserve total fits usize");
            let first = writer
                .try_index_documents_parallel(
                    &cx,
                    ValidatedBatchAdmission {
                        documents: IngestBatch::Default(&documents),
                    },
                    &BTreeSet::new(),
                    false,
                    &ParallelBatchObservation::default(),
                )
                .await
                .expect("run first default-budget generation")
                .expect("first default-budget generation uses shared-nothing");
            assert_eq!(
                first.logical_budget_bytes,
                crate::config::DEFAULT_SCRIBE_SHARD_BUDGET_BYTES,
            );
            assert_eq!(
                first.arena_chunk_bytes,
                parallel_arena_chunk_bytes(first.batch_logical_upper_bound, first.active_shards,),
            );
            assert!(first.arena_chunk_bytes < DEFAULT_ARENA_CHUNK_BYTES);
            let first_installed_reserved = writer
                .shards
                .iter()
                .filter(|shard| shard.accumulator.document_count() != 0)
                .try_fold(0_usize, |total, shard| {
                    total.checked_add(shard.accumulator.bytes_reserved())
                })
                .expect("first installed reserve total fits usize");
            assert_eq!(
                first.arena_bytes_reserved_high_water,
                initial_reserved
                    .checked_add(first_installed_reserved)
                    .expect("first exact reserve peak fits usize"),
            );
            let first_default_chunk_peak = initial_reserved
                .checked_add(
                    first
                        .active_shards
                        .checked_mul(DEFAULT_ARENA_CHUNK_BYTES)
                        .expect("default chunk peak fits usize"),
                )
                .expect("first-generation peak fits usize");
            assert!(first.arena_bytes_reserved_high_water < first_default_chunk_peak);
            assert!(
                writer
                    .shards
                    .iter()
                    .filter(|shard| shard.accumulator.document_count() != 0)
                    .all(|shard| {
                        shard.accumulator.terms().arena_stats().1 < DEFAULT_ARENA_CHUNK_BYTES
                    })
            );

            writer
                .commit_with_trigger(&cx, LifecycleTrigger::ExplicitFlush)
                .await
                .expect("publish first default-budget generation");
            assert!(writer.shards.iter().all(|shard| {
                shard.accumulator.document_count() == 0
                    && shard.identities.is_empty()
                    && shard.current_lease_base.is_none()
            }));
            let retained_reserved = writer
                .shards
                .iter()
                .try_fold(0_usize, |total, shard| {
                    total.checked_add(shard.accumulator.bytes_reserved())
                })
                .expect("retained reserve total fits usize");
            assert!(retained_reserved > initial_reserved);
            let (retained_arena_bytes, retained_arena_chunks) = writer
                .shards
                .iter()
                .try_fold((0_usize, 0_usize), |(bytes, chunks), shard| {
                    let (_, shard_bytes, shard_chunks) = shard.accumulator.terms().arena_stats();
                    Some((
                        bytes.checked_add(shard_bytes)?,
                        chunks.checked_add(shard_chunks)?,
                    ))
                })
                .expect("retained arena totals fit usize");
            assert!(retained_arena_bytes > 0);
            assert!(retained_arena_chunks > 0);

            let mut second_documents = parallel_budget_fixture_documents(256);
            for (ordinal, document) in second_documents.iter_mut().enumerate() {
                document.id = format!("parallel-budget-second-{ordinal:05}");
            }
            let second = writer
                .try_index_documents_parallel(
                    &cx,
                    ValidatedBatchAdmission {
                        documents: IngestBatch::Default(&second_documents),
                    },
                    &BTreeSet::new(),
                    false,
                    &ParallelBatchObservation::default(),
                )
                .await
                .expect("run second default-budget generation")
                .expect("second default-budget generation uses shared-nothing");
            assert!(second.arena_chunk_bytes < DEFAULT_ARENA_CHUNK_BYTES);
            let second_installed_reserved = writer
                .shards
                .iter()
                .filter(|shard| shard.accumulator.document_count() != 0)
                .try_fold(0_usize, |total, shard| {
                    total.checked_add(shard.accumulator.bytes_reserved())
                })
                .expect("second installed reserve total fits usize");
            assert_eq!(
                second.arena_bytes_reserved_high_water,
                retained_reserved
                    .checked_add(second_installed_reserved)
                    .expect("second exact reserve peak fits usize"),
            );
            let second_default_chunk_peak = retained_reserved
                .checked_add(
                    second
                        .active_shards
                        .checked_mul(DEFAULT_ARENA_CHUNK_BYTES)
                        .expect("second default chunk peak fits usize"),
                )
                .expect("second-generation peak fits usize");
            assert!(second.arena_bytes_reserved_high_water < second_default_chunk_peak);

            writer
                .commit_with_trigger(&cx, LifecycleTrigger::ExplicitFlush)
                .await
                .expect("publish second default-budget generation");
            assert_eq!(
                writer
                    .authority_snapshot()
                    .expect("fan-out writer snapshot is authoritative")
                    .doc_count(),
                512
            );
        });
    });
}

#[test]
fn dirty_prior_shard_blocks_parallel_budget_admission() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker dirty-shard pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("dirty-shard index");
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("prior", "retained prior shard")],
                )
                .await
                .expect("seed one dirty serial shard");
            let documents = parallel_budget_fixture_documents(250);
            let writer = index.writer_mut();
            let grants_before = writer.docid_allocator.lease_grants().len();
            let watermark_before = writer.docid_allocator.watermark();
            assert!(
                writer
                    .try_index_documents_parallel(
                        &cx,
                        ValidatedBatchAdmission {
                            documents: IngestBatch::Default(&documents),
                        },
                        &BTreeSet::new(),
                        false,
                        &ParallelBatchObservation::default(),
                    )
                    .await
                    .expect("dirty-shard fallback is not an error")
                    .is_none()
            );
            assert_eq!(writer.docid_allocator.lease_grants().len(), grants_before);
            assert_eq!(writer.docid_allocator.watermark(), watermark_before);
            assert_eq!(
                writer
                    .shards
                    .iter()
                    .map(|shard| shard.accumulator.document_count())
                    .sum::<usize>(),
                1,
            );
        });
    });
}

#[test]
fn scalar_budget_overshoot_flushes_before_the_successor_document() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            scribe_shard_budget_bytes: 10_000,
            deterministic_ingest: true,
            max_visibility_lag_ms: 60_000,
            ..QuillConfig::default()
        };
        let mut index = QuillIndex::in_memory(config).expect("scalar boundary index");
        let documents = [
            IndexableDocument::new("large", "unique ".repeat(5_000)),
            IndexableDocument::new("tiny", "x"),
        ];
        index
            .index_documents(&cx, &documents)
            .await
            .expect("index scalar boundary fixture");
        let writer = index.writer_mut();
        assert_eq!(writer.pending_segments.len(), 1);
        assert_eq!(writer.pending_segments[0].doc_count, 1);
        let dirty = writer
            .shards
            .iter()
            .find(|shard| shard.accumulator.document_count() != 0)
            .expect("successor document remains in a fresh accumulator");
        assert_eq!(dirty.accumulator.document_count(), 1);
        assert_eq!(dirty.identities[0].document_id, "tiny");
    });
}

#[test]
fn parallel_budget_boundary_matches_forced_scalar() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker boundary parity pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let documents = parallel_budget_fixture_documents(250);
            let common = QuillConfig {
                scribe_shard_budget_bytes: 1,
                max_visibility_lag_ms: 60_000,
                tier_fanout: usize::MAX,
                ..QuillConfig::default()
            };
            let mut adaptive = QuillIndex::in_memory(QuillConfig {
                max_ingest_shards: 4,
                ..common.clone()
            })
            .expect("adaptive boundary index");
            let mut scalar = QuillIndex::in_memory(QuillConfig {
                deterministic_ingest: true,
                ..common
            })
            .expect("forced scalar boundary index");

            adaptive
                .index_documents(&cx, &documents)
                .await
                .expect("index adaptive boundary fixture");
            scalar
                .index_documents(&cx, &documents)
                .await
                .expect("index scalar boundary fixture");
            assert_eq!(adaptive.writer_mut().pending_segments.len(), 250);
            assert_eq!(scalar.writer_mut().pending_segments.len(), 250);

            adaptive
                .commit(&cx)
                .await
                .expect("publish adaptive boundary");
            scalar.commit(&cx).await.expect("publish scalar boundary");
            assert_eq!(
                adaptive
                    .snapshot()
                    .expect("adaptive equality snapshot is authoritative")
                    .loaded_manifest()
                    .manifest
                    .field_stats,
                scalar
                    .snapshot()
                    .expect("scalar equality snapshot is authoritative")
                    .loaded_manifest()
                    .manifest
                    .field_stats,
            );
            for document in &documents {
                assert_eq!(
                    adaptive
                        .document_witness(&document.id)
                        .expect("adaptive document witness"),
                    scalar
                        .document_witness(&document.id)
                        .expect("scalar document witness"),
                    "budget fallback witness differs for {}",
                    document.id,
                );
            }
            assert_eq!(
                adaptive
                    .search_doc_ids(&cx, "shared budget", 300)
                    .expect("search adaptive boundary"),
                scalar
                    .search_doc_ids(&cx, "shared budget", 300)
                    .expect("search scalar boundary"),
            );
        });
    });
}

#[test]
fn parallel_budget_equality_wrapper_matches_forced_scalar() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker equality parity pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let documents = parallel_budget_fixture_documents(250);
            let mut adaptive = QuillIndex::in_memory(QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            })
            .expect("adaptive equality index");
            let projected = {
                let writer = adaptive.writer_mut();
                let plan = plan_parallel_ingest(
                    documents.len(),
                    writer.shard_router.shard_count(),
                    rayon::current_num_threads(),
                )
                .expect("plan equality fixture");
                writer
                    .parallel_budget_admission(
                        &cx,
                        IngestBatch::Default(&documents),
                        plan.active_shards,
                    )
                    .expect("estimate equality fixture")
                    .expect("default budget fits equality fixture")
                    .projected_logical_upper_bound
            };
            adaptive
                .writer_mut()
                .reader
                .config
                .scribe_shard_budget_bytes = projected;
            let scalar = QuillIndex::in_memory(QuillConfig {
                scribe_shard_budget_bytes: projected,
                deterministic_ingest: true,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            })
            .expect("scalar equality control");

            adaptive
                .index_documents(&cx, &documents)
                .await
                .expect("index equality fallback");
            scalar
                .index_documents(&cx, &documents)
                .await
                .expect("index equality scalar control");
            let adaptive_writer = adaptive.writer_mut();
            assert!(adaptive_writer.pending_segments.is_empty());
            assert_eq!(
                adaptive_writer
                    .shards
                    .iter()
                    .filter(|shard| shard.accumulator.document_count() != 0)
                    .count(),
                1,
                "budget equality must enter the scalar wrapper route",
            );

            adaptive
                .commit(&cx)
                .await
                .expect("publish equality fallback");
            scalar
                .commit(&cx)
                .await
                .expect("publish equality scalar control");
            assert_eq!(
                adaptive
                    .snapshot()
                    .expect("adaptive equality snapshot is authoritative")
                    .loaded_manifest()
                    .manifest
                    .field_stats,
                scalar
                    .snapshot()
                    .expect("scalar equality snapshot is authoritative")
                    .loaded_manifest()
                    .manifest
                    .field_stats,
            );
            for document in [
                documents.first().expect("first equality document"),
                documents.last().expect("last equality document"),
            ] {
                assert_eq!(
                    adaptive
                        .document_witness(&document.id)
                        .expect("adaptive equality witness"),
                    scalar
                        .document_witness(&document.id)
                        .expect("scalar equality witness"),
                );
            }
            assert_eq!(
                adaptive
                    .search_doc_ids(&cx, "shared budget", 300)
                    .expect("search equality fallback"),
                scalar
                    .search_doc_ids(&cx, "shared budget", 300)
                    .expect("search equality control"),
            );
        });
    });
}

#[test]
fn large_batch_builds_all_available_shard_segments_shared_nothing() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            max_ingest_shards: 4,
            ..QuillConfig::default()
        };
        let mut index = QuillIndex::in_memory(config).expect("multi-shard memory index");
        let shard_count = index.writer_mut().shard_router.shard_count();
        let documents = (0..shard_count * PARALLEL_INGEST_MIN_DOCS_PER_SHARD)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("parallel-shard-{ordinal:05}"),
                    "parallel shared nothing indexing fixture",
                )
            })
            .collect::<Vec<_>>();
        index
            .index_documents(&cx, &documents)
            .await
            .expect("accumulate one large batch");

        let active_shards =
            plan_parallel_ingest(documents.len(), shard_count, rayon::current_num_threads())
                .expect("plan large parallel fixture")
                .active_shards;
        let expected_segments = active_shards.max(1);
        index.commit(&cx).await.expect("publish parallel batch");
        assert_eq!(
            index
                .snapshot()
                .expect("large-batch snapshot is authoritative")
                .segments()
                .len(),
            expected_segments
        );
        assert_eq!(
            index
                .snapshot()
                .expect("large-batch snapshot is authoritative")
                .doc_count(),
            u64::try_from(documents.len()).expect("fixture document count fits u64"),
        );
        assert_pairwise_disjoint_manifest(
            index
                .snapshot()
                .expect("large-batch snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .segments
                .as_slice(),
        );
    });
}

#[test]
fn deterministic_large_batch_builds_four_contiguous_segments() {
    run_with_cx(|cx| async move {
        let mut index =
            QuillIndex::in_memory(deterministic_config()).expect("deterministic memory index");
        let document_count = INTERNAL_PARALLEL_INGEST_SHARDS
            .checked_mul(PARALLEL_INGEST_MIN_DOCS_PER_SHARD)
            .expect("bounded deterministic parallel fixture");
        let documents = (0..document_count)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("internal-parallel-{ordinal:05}"),
                    "deterministic internal parallel indexing fixture",
                )
            })
            .collect::<Vec<_>>();
        index
            .index_documents(&cx, &documents)
            .await
            .expect("build deterministic internal segments");

        assert_eq!(
            index.writer_mut().pending_segments.len(),
            INTERNAL_PARALLEL_INGEST_SHARDS,
        );
        assert!(index.writer_mut().shards.iter().all(|shard| {
            shard.accumulator.document_count() == 0 && shard.identities.is_empty()
        }));
        index
            .commit(&cx)
            .await
            .expect("publish deterministic internal segments");

        let snapshot = index
            .snapshot()
            .expect("deterministic batch snapshot is authoritative");
        let manifest = &snapshot.loaded_manifest().manifest;
        assert_eq!(manifest.segments.len(), INTERNAL_PARALLEL_INGEST_SHARDS);
        assert_eq!(
            index
                .snapshot()
                .expect("deterministic batch snapshot is authoritative")
                .doc_count(),
            u64::try_from(document_count).expect("fixture document count fits u64"),
        );
        assert_eq!(
            manifest.segments.first().map(|segment| segment.docid_lo),
            Some(0)
        );
        assert_eq!(
            manifest.segments.last().map(|segment| segment.docid_hi),
            Some(u64::try_from(document_count).expect("fixture document count fits u64")),
        );
        assert!(
            manifest
                .segments
                .windows(2)
                .all(|pair| pair[0].docid_hi == pair[1].docid_lo),
            "internal segment ranges must be contiguous and gap-free",
        );
        assert_pairwise_disjoint_manifest(&manifest.segments);
    });
}

#[cfg(feature = "pruning-conformance")]
#[test]
fn scalar_topology_conformance_seam_preserves_one_requested_leaf() {
    run_with_cx(|cx| async move {
        let document_count = INTERNAL_PARALLEL_INGEST_SHARDS
            .checked_mul(PARALLEL_INGEST_MIN_DOCS_PER_SHARD)
            .expect("bounded scalar-topology fixture");
        let documents = (0..document_count)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("scalar-topology-{ordinal:05}"),
                    "scalar topology conformance fixture",
                )
            })
            .collect::<Vec<_>>();
        let config = QuillConfig {
            max_visibility_lag_ms: u64::MAX,
            ..deterministic_config()
        };
        let mut adaptive = QuillIndex::in_memory(config.clone()).expect("adaptive control index");
        let mut scalar = QuillIndex::in_memory(config).expect("scalar topology index");

        adaptive
            .index_documents(&cx, &documents)
            .await
            .expect("build adaptive internal segments");
        scalar
            .index_documents_with_scalar_topology_conformance(&cx, &documents)
            .await
            .expect("accumulate one scalar topology leaf");

        assert_eq!(
            adaptive.writer_mut().pending_segments.len(),
            INTERNAL_PARALLEL_INGEST_SHARDS,
            "shipping ingest must retain its fixed-width internal fanout",
        );
        assert!(scalar.writer_mut().pending_segments.is_empty());
        assert_eq!(
            scalar
                .writer_mut()
                .shards
                .iter()
                .map(|shard| shard.accumulator.document_count())
                .sum::<usize>(),
            document_count,
        );

        adaptive
            .commit(&cx)
            .await
            .expect("publish adaptive control");
        scalar
            .commit(&cx)
            .await
            .expect("publish scalar topology leaf");
        assert_eq!(
            adaptive
                .snapshot()
                .expect("adaptive topology snapshot is authoritative")
                .segments()
                .len(),
            INTERNAL_PARALLEL_INGEST_SHARDS,
        );
        assert_eq!(
            scalar
                .snapshot()
                .expect("scalar topology snapshot is authoritative")
                .segments()
                .len(),
            1
        );
        assert_eq!(
            scalar
                .snapshot()
                .expect("scalar topology snapshot is authoritative")
                .segments()[0]
                .doc_count(),
            u32::try_from(document_count).expect("fixture count fits u32"),
        );
        assert_eq!(
            adaptive
                .snapshot()
                .expect("adaptive topology snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats,
            scalar
                .snapshot()
                .expect("scalar topology snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats,
        );
        assert_eq!(
            adaptive
                .search_doc_ids(&cx, "scalar topology", document_count)
                .expect("search adaptive control"),
            scalar
                .search_doc_ids(&cx, "scalar topology", document_count)
                .expect("search scalar topology leaf"),
        );
    });
}

#[test]
fn adaptive_parallel_planner_v1_covers_frozen_matrix() {
    const DOCUMENT_COUNTS: [usize; 10] = [0, 1, 63, 64, 127, 128, 249, 250, 5_000, 8_192];
    const WIDTHS: [usize; 6] = [1, 2, 4, 64, 96, 128];

    let goldens = [
        (250, 4, 4, 3),
        (5_000, 96, 96, 78),
        (5_000, 128, 128, 78),
        (8_192, 128, 128, 128),
    ];
    for (document_count, configured_width, pool_capacity, expected_active) in goldens {
        let plan = plan_parallel_ingest(document_count, configured_width, pool_capacity)
            .expect("plan frozen adaptive-ingest golden");
        assert_eq!(plan.active_shards, expected_active);
    }

    for document_count in DOCUMENT_COUNTS {
        for width in WIDTHS {
            let plan = plan_parallel_ingest(document_count, width, width)
                .expect("plan frozen adaptive-ingest matrix cell");
            let repeated = plan_parallel_ingest(document_count, width, width)
                .expect("repeat frozen adaptive-ingest matrix cell");
            assert_eq!(plan, repeated, "planner must be deterministic");
            assert_eq!(plan.planner_version, PARALLEL_INGEST_PLANNER_VERSION);
            assert_eq!(plan.configured_width, width);
            assert_eq!(plan.verified_pool_capacity, width);
            assert_eq!(plan.eligible_shards, width);
            assert_eq!(
                plan.active_shards,
                width.min(document_count / PARALLEL_INGEST_MIN_DOCS_PER_SHARD),
            );

            if plan.active_shards < 2 {
                assert_eq!(plan.route, ParallelIngestRoute::Serial);
                assert!(plan.ranges.is_empty());
                continue;
            }

            assert_eq!(plan.route, ParallelIngestRoute::SharedNothing);
            assert_eq!(plan.ranges.len(), plan.active_shards);
            assert_eq!(plan.ranges.first().map(|range| range.start), Some(0));
            assert_eq!(
                plan.ranges.last().map(|range| range.end),
                Some(document_count),
            );
            assert!(
                plan.ranges.windows(2).all(|pair| pair
                    .first()
                    .zip(pair.get(1))
                    .is_some_and(|(first, second)| first.end == second.start)),
                "ranges must be contiguous and nonoverlapping",
            );
            assert!(
                plan.ranges.iter().all(|range| {
                    range.len() >= PARALLEL_INGEST_MIN_DOCS_PER_SHARD && range.end <= document_count
                }),
                "tail documents must be folded into nonempty amortized ranges",
            );
            let shortest = plan
                .ranges
                .iter()
                .map(|range| range.len())
                .min()
                .expect("parallel plan has ranges");
            let longest = plan
                .ranges
                .iter()
                .map(|range| range.len())
                .max()
                .expect("parallel plan has ranges");
            assert!(longest - shortest <= 1, "ranges must remain balanced");
            assert_eq!(
                plan.ranges.iter().map(|range| range.len()).sum::<usize>(),
                document_count,
            );
        }
    }

    let capacity_limited =
        plan_parallel_ingest(8_192, 128, 96).expect("plan capacity-limited adaptive ingest");
    assert_eq!(capacity_limited.eligible_shards, 96);
    assert_eq!(capacity_limited.active_shards, 96);
}

#[test]
fn adaptive_parallel_batch_activates_only_amortized_shards() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            max_ingest_shards: 4,
            ..QuillConfig::default()
        };
        let mut index = QuillIndex::in_memory(config).expect("multi-shard memory index");
        let shard_count = index.writer_mut().shard_router.shard_count();
        let documents = (0..250)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("adaptive-shard-{ordinal:05}"),
                    "adaptive shared nothing indexing fixture",
                )
            })
            .collect::<Vec<_>>();
        index
            .index_documents(&cx, &documents)
            .await
            .expect("accumulate one adaptive batch");

        let active_shards =
            plan_parallel_ingest(documents.len(), shard_count, rayon::current_num_threads())
                .expect("plan adaptive parallel fixture")
                .active_shards;
        index.commit(&cx).await.expect("publish adaptive batch");
        assert_eq!(
            index
                .snapshot()
                .expect("adaptive batch snapshot is authoritative")
                .segments()
                .len(),
            active_shards.max(1)
        );
        assert_eq!(
            index
                .snapshot()
                .expect("adaptive batch snapshot is authoritative")
                .doc_count(),
            250
        );
        assert_pairwise_disjoint_manifest(
            index
                .snapshot()
                .expect("adaptive batch snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .segments
                .as_slice(),
        );
    });
}

#[cfg(feature = "conformance-internals")]
#[test]
fn cancelled_parallel_budget_preflight_is_pristine_and_retryable() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker budget cancellation pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            };
            let index = QuillIndex::in_memory(config.clone()).expect("budget cancellation index");
            let documents = parallel_budget_fixture_documents(250);
            let state_before = index
                .conformance_pending_writer_state()
                .expect("capture pristine budget-preflight state");
            let controller = index.conformance_cancellation_controller();
            controller
                .arm(ConformanceCancellationStage::ParallelBudgetAdmission, 3)
                .expect("arm budget-preflight cancellation");

            let cancelled = index
                .index_documents(&cx, &documents)
                .await
                .expect_err("budget preflight must observe injected cancellation");
            assert!(matches!(
                &cancelled,
                QuillIndexError::Cancelled {
                    phase: "parallel budget admission"
                }
            ));
            assert!(controller.fired());
            assert!(controller.observed_checkpoints() >= 3);
            assert_eq!(
                index
                    .conformance_pending_writer_state()
                    .expect("capture cancelled budget-preflight state"),
                state_before,
            );

            controller.disarm();
            cx.set_cancel_requested(false);
            index
                .index_documents(&cx, &documents)
                .await
                .expect("retry cancelled budget preflight");
            let retry_state = index
                .conformance_pending_writer_state()
                .expect("capture retried budget-preflight state");
            let control = QuillIndex::in_memory(config).expect("fresh budget control index");
            control
                .index_documents(&cx, &documents)
                .await
                .expect("run fresh budget control");
            assert_eq!(
                retry_state,
                control
                    .conformance_pending_writer_state()
                    .expect("capture fresh budget-control state"),
            );
        });
    });
}

#[cfg(feature = "conformance-internals")]
#[test]
fn cancelled_parallel_batch_preserves_exact_state_and_retries_identically() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            max_ingest_shards: 4,
            max_visibility_lag_ms: 60_000,
            ..QuillConfig::default()
        };
        let mut index = QuillIndex::in_memory(config.clone()).expect("parallel memory index");
        let shard_count = index.writer_mut().shard_router.shard_count();
        let plan = plan_parallel_ingest(256, shard_count, rayon::current_num_threads())
            .expect("plan cancellation fixture");
        assert_eq!(
            plan.route,
            ParallelIngestRoute::SharedNothing,
            "transaction test requires at least two verified worker shards",
        );
        let documents = (0..256)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("parallel-transaction-{ordinal:05}"),
                    "shared nothing cancellation transaction fixture",
                )
            })
            .collect::<Vec<_>>();
        let state_before = index
            .conformance_pending_writer_state()
            .expect("capture pristine writer state");
        let controller = index.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::ParallelIngest, 3)
            .expect("arm mid-worker cancellation");

        let cancelled = index
            .index_documents(&cx, &documents)
            .await
            .expect_err("parallel worker must observe injected cancellation");
        assert!(
            matches!(
                &cancelled,
                QuillIndexError::Cancelled {
                    phase: "parallel index worker"
                }
            ),
            "unexpected cancellation result: {cancelled:?}",
        );
        assert!(controller.fired());
        assert!(controller.observed_checkpoints() >= 3);
        assert!(cx.is_cancel_requested());
        assert!(!index.has_uncommitted_changes());
        assert_eq!(
            index
                .conformance_pending_writer_state()
                .expect("capture cancelled writer state"),
            state_before,
            "a joined worker cancellation must not commit allocator, router, shard, or retry-guard state",
        );

        controller.disarm();
        cx.set_cancel_requested(false);
        index
            .index_documents(&cx, &documents)
            .await
            .expect("retry cancelled parallel batch");
        let retry_state = index
            .conformance_pending_writer_state()
            .expect("capture retried writer state");

        let control = QuillIndex::in_memory(config).expect("fresh parallel control index");
        control
            .index_documents(&cx, &documents)
            .await
            .expect("accumulate fresh parallel control batch");
        assert_eq!(
            retry_state,
            control
                .conformance_pending_writer_state()
                .expect("capture control writer state"),
            "retry must reproduce the exact fresh allocator, router, shard, and identity state",
        );

        index.commit(&cx).await.expect("publish retried batch");
        control.commit(&cx).await.expect("publish control batch");
        assert_eq!(
            index
                .snapshot()
                .expect("retried parallel snapshot is authoritative")
                .doc_count(),
            256
        );
        assert_eq!(
            control
                .snapshot()
                .expect("parallel control snapshot is authoritative")
                .doc_count(),
            256
        );
        for document in &documents {
            assert_eq!(
                index
                    .document_witness(&document.id)
                    .expect("resolve retried witness"),
                control
                    .document_witness(&document.id)
                    .expect("resolve control witness"),
                "retry witness differs for {}",
                document.id,
            );
        }
    });
}

#[test]
fn parallel_worker_panic_is_a_typed_precommit_failure() {
    let error = catch_parallel_ingest_worker(7, || -> Result<(), QuillIndexError> {
        // ubs:ignore -- Deliberate panic injection tests conversion to the typed worker failure.
        panic!("injected parallel worker panic");
    })
    .expect_err("worker panic must be caught");
    assert!(matches!(
        error,
        QuillIndexError::InvalidState { detail }
            if detail == "parallel ingest worker 7 panicked before transactional commit"
    ));
}

#[test]
fn deterministic_batch_uses_fixed_internal_parallel_plan() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            max_ingest_shards: 4,
            deterministic_ingest: true,
            ..QuillConfig::default()
        };
        let index = QuillIndex::in_memory(config).expect("deterministic memory index");
        let documents = (0..250)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("deterministic-shard-{ordinal:05}"),
                    "deterministic indexing fixture",
                )
            })
            .collect::<Vec<_>>();
        index
            .index_documents(&cx, &documents)
            .await
            .expect("accumulate deterministic batch");
        index
            .commit(&cx)
            .await
            .expect("publish deterministic batch");
        let expected_segments = INTERNAL_PARALLEL_INGEST_SHARDS
            .min(documents.len() / PARALLEL_INGEST_MIN_DOCS_PER_SHARD);
        assert_eq!(
            index
                .snapshot()
                .expect("fixed-plan snapshot is authoritative")
                .segments()
                .len(),
            expected_segments
        );
        assert_eq!(
            index
                .snapshot()
                .expect("fixed-plan snapshot is authoritative")
                .doc_count(),
            250
        );
    });
}
