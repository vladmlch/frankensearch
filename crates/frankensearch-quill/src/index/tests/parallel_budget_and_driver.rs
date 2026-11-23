use super::*;

fn assert_parallel_budget_bound_for_schema(schema: SchemaDescriptor) {
    let exact_maximum = "X".repeat(MAX_TERM_BYTES);
    let dropped_oversized = "Y".repeat(MAX_TERM_BYTES + 1);
    let mut documents = vec![
        IndexableDocument::new("bound-0", "alpha alpha İSTANBUL 東京")
            .with_title("Repeated REPEATED")
            .with_metadata("escaped", "line\nquote\"nul\0"),
        IndexableDocument::new("bound-1", format!("{exact_maximum} {dropped_oversized}"))
            .with_title("z-z-z"),
        IndexableDocument::new(exact_maximum.clone(), "keyword boundary admitted"),
        IndexableDocument::new(dropped_oversized.clone(), "keyword boundary dropped"),
    ];
    documents.extend((0..32).map(|ordinal| {
        IndexableDocument::new(
            format!("bound-random-{ordinal:02}"),
            format!(
                "{} {} shared-{} Ω{}",
                "unique".repeat(ordinal % 7 + 1),
                "repeat ".repeat(ordinal % 5 + 1),
                ordinal % 3,
                ordinal
            ),
        )
        .with_title(format!("title-{}", ordinal % 4))
        .with_metadata("bucket", format!("{}\n{}", ordinal % 9, ordinal))
    }));

    let mut accumulator = ColumnarAccumulator::new(schema).expect("budget-bound accumulator");
    let mut analyzer = FrankensearchTokenizer::default();
    let mut cumulative_upper_bound = accumulator.bytes_used();
    for (ordinal, document) in documents.iter().enumerate() {
        let document_bound =
            parallel_document_logical_upper_bound(schema, &mut analyzer, document, usize::MAX)
                .expect("estimate budget-bound document")
                .expect("shipping-shaped schema has a bound");
        let ParallelDocumentBudgetBound::Within {
            logical_upper_bound: document_upper_bound,
            canonical_metadata: Some(metadata),
        } = document_bound
        else {
            // ubs:ignore -- Missing fixture metadata must fail this budget test.
            panic!("unbounded fixture must produce reusable canonical metadata");
        };
        cumulative_upper_bound = cumulative_upper_bound
            .checked_add(document_upper_bound)
            .expect("fixture upper bound fits usize");

        let title = document.title.as_deref().unwrap_or("");
        let indexed = [
            IndexedFieldValue::new(ID_FIELD, &document.id),
            IndexedFieldValue::new(CONTENT_FIELD, &document.content),
            IndexedFieldValue::new(TITLE_FIELD, title),
        ];
        let numeric = [IndexedNumericValue::u64(
            ORD_FIELD,
            u64::try_from(ordinal).expect("fixture ordinal fits u64"),
        )];
        let stored = [StoredFieldValue::new(METADATA_FIELD, &metadata)];
        let mut fresh = ColumnarAccumulator::new(schema).expect("fresh bound accumulator");
        let fresh_initial = fresh.bytes_used();
        fresh
            .add_document_with_values(0, &indexed, &numeric, &stored)
            .expect("accumulate fresh budget-bound document");
        let fresh_delta = fresh
            .bytes_used()
            .checked_sub(fresh_initial)
            .expect("one document cannot reduce logical use");
        assert!(
            fresh_delta <= document_upper_bound,
            "fresh-document logical increment {fresh_delta} exceeded bound {document_upper_bound} for document {ordinal}",
        );

        accumulator
            .add_document_with_values(
                u32::try_from(ordinal).expect("fixture ordinal fits u32"),
                &indexed,
                &numeric,
                &stored,
            )
            .expect("accumulate budget-bound document");
        assert!(
            accumulator.bytes_used() <= cumulative_upper_bound,
            "logical use {} exceeded conservative bound {cumulative_upper_bound} after document {ordinal}",
            accumulator.bytes_used(),
        );
    }
}

#[test]
fn parallel_budget_fit_bound_is_conservative() {
    assert_parallel_budget_bound_for_schema(DEFAULT_SCHEMA);
    assert_parallel_budget_bound_for_schema(MIXED_BUDGET_SCHEMA);
}

#[test]
fn parallel_budget_source_length_validation_rejects_u32_overflow() {
    let maximum = usize::try_from(u32::MAX).expect("test requires a usize at least u32 wide");
    assert!(validate_parallel_source_len(CONTENT_FIELD, maximum).is_ok());
    let Some(oversized) = maximum.checked_add(1) else {
        return;
    };
    assert!(matches!(
        validate_parallel_source_len(CONTENT_FIELD, oversized),
        Err(QuillIndexError::Accumulator(
            AccumulatorError::SourceTooLarge {
                field_ord: CONTENT_FIELD,
                bytes,
            }
        )) if bytes == oversized
    ));
}

#[test]
fn parallel_budget_fixed_lower_bound_precedes_tokenization() {
    let document = IndexableDocument::new("lower-bound-id", "content must not be tokenized")
        .with_title("title must not be tokenized");
    let mut analyzer = FrankensearchTokenizer::default();
    assert_eq!(analyzer.bytes_reserved(), 0);
    assert_eq!(
        parallel_document_logical_upper_bound(DEFAULT_SCHEMA, &mut analyzer, &document, 1)
            .expect("estimate lower-bound fixture"),
        Some(ParallelDocumentBudgetBound::ReachesCeiling),
    );
    assert_eq!(
        analyzer.bytes_reserved(),
        0,
        "fixed/stored rejection must precede tokenizer scratch allocation",
    );
}

#[test]
fn parallel_budget_token_charge_honors_exclusive_ceiling() {
    let document =
        IndexableDocument::new("token-ceiling-id", "alpha beta").with_title("gamma delta");
    let mut unbounded_analyzer = FrankensearchTokenizer::default();
    let full = parallel_document_logical_upper_bound(
        DEFAULT_SCHEMA,
        &mut unbounded_analyzer,
        &document,
        usize::MAX,
    )
    .expect("estimate token-ceiling fixture")
    .expect("shipping schema has token-ceiling bound");
    let ParallelDocumentBudgetBound::Within {
        logical_upper_bound: full,
        canonical_metadata: Some(full_metadata),
    } = full
    else {
        // ubs:ignore -- Missing fixture metadata must fail this token budget test.
        panic!("unbounded token fixture must produce reusable canonical metadata");
    };

    let mut equality_analyzer = FrankensearchTokenizer::default();
    assert_eq!(
        parallel_document_logical_upper_bound(
            DEFAULT_SCHEMA,
            &mut equality_analyzer,
            &document,
            full,
        )
        .expect("estimate token-bound equality"),
        Some(ParallelDocumentBudgetBound::ReachesCeiling),
    );

    let mut fitting_analyzer = FrankensearchTokenizer::default();
    let fitting = parallel_document_logical_upper_bound(
        DEFAULT_SCHEMA,
        &mut fitting_analyzer,
        &document,
        full.checked_add(1).expect("fixture bound fits usize"),
    )
    .expect("estimate token-bound fit")
    .expect("fitting token bound is present");
    let ParallelDocumentBudgetBound::Within {
        logical_upper_bound: fitting_bound,
        canonical_metadata: Some(fitting_metadata),
    } = fitting
    else {
        // ubs:ignore -- Rejected fitting metadata must fail this boundary test.
        panic!("one byte of slack must admit reusable canonical metadata");
    };
    assert_eq!(fitting_bound, full);
    assert_eq!(fitting_metadata, full_metadata);
    assert_eq!(
        fitting_analyzer.owned_bridge_calls(),
        0,
        "the count-only budget pass must consume borrowed tokens without copying them through the owned compatibility bridge",
    );
}

#[test]
fn prepared_metadata_retention_respects_the_exclusive_batch_budget() {
    let retained_bytes = AtomicUsize::new(0);
    let exclusive_ceiling = 10;

    assert!(
        retain_prepared_metadata_within_budget(vec![0_u8; 4], &retained_bytes, exclusive_ceiling,)
            .is_some()
    );
    assert!(
        retain_prepared_metadata_within_budget(vec![0_u8; 5], &retained_bytes, exclusive_ceiling,)
            .is_some()
    );
    assert!(
        retain_prepared_metadata_within_budget(vec![0_u8; 1], &retained_bytes, exclusive_ceiling,)
            .is_none(),
        "the cache must not reach the exclusive ceiling",
    );
    assert_eq!(retained_bytes.load(Ordering::Acquire), 9);
}

#[cfg(feature = "bench-internals")]
#[test]
fn parallel_budget_fit_bound_is_conservative_without_positions() {
    assert_parallel_budget_bound_for_schema(POSITIONLESS_QG_SCHEMA);
}

#[test]
fn parallel_budget_arena_chunk_slack_is_aggregate_bounded() {
    const WIDTHS: [usize; 10] = [1, 2, 3, 4, 8, 16, 32, 128, 256, 512];
    let near_default_budget = crate::config::DEFAULT_SCRIBE_SHARD_BUDGET_BYTES - 1;
    for active_shards in WIDTHS {
        let chunk_bytes = parallel_arena_chunk_bytes(near_default_budget, active_shards);
        let aggregate_chunk_slack = active_shards
            .checked_mul(chunk_bytes)
            .expect("aggregate chunk slack fits usize");
        let arena_floor = active_shards
            .checked_mul(MIN_ARENA_CHUNK_BYTES)
            .expect("aggregate arena floor fits usize");
        assert!(
            aggregate_chunk_slack <= DEFAULT_ARENA_CHUNK_BYTES.max(arena_floor),
            "width {active_shards} selected {chunk_bytes} bytes per shard ({aggregate_chunk_slack} aggregate)",
        );
    }
}

#[test]
fn parallel_budget_preflight_uses_the_scoped_worker_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker budget preflight pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            })
            .expect("parallel budget preflight index");
            let documents = parallel_budget_fixture_documents(5_000);
            let writer = index.writer_mut();
            writer.force_ingest_overlap_rendezvous = true;
            let plan = plan_parallel_ingest(
                documents.len(),
                writer.shard_router.shard_count(),
                rayon::current_num_threads(),
            )
            .expect("plan parallel budget preflight fixture");
            assert_eq!(plan.active_shards, 4);

            let admission = writer
                .parallel_budget_admission(
                    &cx,
                    IngestBatch::Default(&documents),
                    plan.active_shards,
                )
                .expect("estimate parallel budget preflight fixture")
                .expect("default budget admits preflight fixture");
            assert!(
                admission.budget_worker_slots >= 2,
                "a 5,000-document preflight on a four-worker pool must not collapse to one worker",
            );
            assert!(admission.budget_worker_slots <= plan.active_shards);
        });
    });
}

#[test]
fn qg1_shaped_parallel_budget_refusal_stops_after_the_first_bounded_wave() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("build eight-worker QG-1-shaped budget pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(QuillConfig {
                scribe_shard_budget_bytes: 50_000_000 / 8,
                max_ingest_shards: 8,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            })
            .expect("QG-1-shaped budget-refusal index");
            let documents = (0..5_000)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("qg1-shaped-{ordinal:05}"),
                        "common ranked synthetic token ".repeat(120),
                    )
                })
                .collect::<Vec<_>>();
            let writer = index.writer_mut();
            let plan = plan_parallel_ingest(
                documents.len(),
                writer.shard_router.shard_count(),
                rayon::current_num_threads(),
            )
            .expect("plan QG-1-shaped budget-refusal fixture");
            assert_eq!(plan.active_shards, 8);

            assert!(
                writer
                    .parallel_budget_admission(
                        &cx,
                        IngestBatch::Default(&documents),
                        plan.active_shards
                    )
                    .expect("evaluate QG-1-shaped budget-refusal fixture")
                    .is_none(),
                "the 6.25 MB QG-1 budget must reject the 5,000-document fixture",
            );
            let analyzed = writer
                .parallel_budget_documents_analyzed
                .load(Ordering::Acquire);
            assert_eq!(
                analyzed,
                plan.active_shards
                    .checked_mul(FANOUT_WAVE_SHARD_DOCUMENTS)
                    .expect("bounded analysis wave fits usize"),
                "refusal must finish the first issued wave but launch no successor wave",
            );
            assert!(
                analyzed < documents.len(),
                "a rejected batch must not be analyzed to exhaustion",
            );
        });
    });
}

#[test]
fn writer_construction_uses_scoped_rayon_pool_width() {
    let host_parallelism = std::thread::available_parallelism().map_or(1, usize::from);
    let scoped_width = if host_parallelism == 1 { 2 } else { 1 };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(scoped_width)
        .build()
        .expect("build scoped writer-construction pool");

    pool.install(|| {
        let mut index = QuillIndex::in_memory(QuillConfig {
            max_ingest_shards: host_parallelism.saturating_add(1),
            ..QuillConfig::default()
        })
        .expect("construct index in scoped pool");

        assert_eq!(
            index.writer_mut().shard_router.shard_count(),
            scoped_width,
            "writer construction must use the active scoped pool, not host parallelism"
        );
    });
}

#[test]
fn parallel_budget_threshold_falls_back_before_workers() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker budget test pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("budget threshold index");
            let documents = parallel_budget_fixture_documents(250);
            let writer = index.writer_mut();
            let plan = plan_parallel_ingest(
                documents.len(),
                writer.shard_router.shard_count(),
                rayon::current_num_threads(),
            )
            .expect("plan budget threshold fixture");
            assert_eq!(plan.route, ParallelIngestRoute::SharedNothing);
            let projected = writer
                .parallel_budget_admission(
                    &cx,
                    IngestBatch::Default(&documents),
                    plan.active_shards,
                )
                .expect("admit fixture under default budget")
                .expect("default budget fits fixture")
                .projected_logical_upper_bound;
            let grants_before = writer.docid_allocator.lease_grants().len();
            let watermark_before = writer.docid_allocator.watermark();
            let next_shard_before = writer.shard_router.clone().route_batch();

            for budget in [projected, projected.saturating_sub(1)] {
                writer.reader.config.scribe_shard_budget_bytes = budget;
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
                        .expect("budget fallback is not an error")
                        .is_none(),
                    "projected logical use {projected} must not be admitted at budget {budget}",
                );
                assert_eq!(writer.docid_allocator.lease_grants().len(), grants_before);
                assert_eq!(writer.docid_allocator.watermark(), watermark_before);
                assert_eq!(writer.shard_router.clone().route_batch(), next_shard_before);
                assert!(!writer.ingest_retry_required);
                assert!(writer.shards.iter().all(|shard| {
                    shard.accumulator.document_count() == 0
                        && shard.identities.is_empty()
                        && shard.current_lease_base.is_none()
                }));
            }
        });
    });
}

#[test]
fn parallel_budget_fit_uses_shared_nothing_without_budget_flush() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker budget admission pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("budget fit index");
            let documents = parallel_budget_fixture_documents(250);
            let writer = index.writer_mut();
            let plan = plan_parallel_ingest(
                documents.len(),
                writer.shard_router.shard_count(),
                rayon::current_num_threads(),
            )
            .expect("plan budget fit fixture");
            assert_eq!(plan.route, ParallelIngestRoute::SharedNothing);
            let projected = writer
                .parallel_budget_admission(
                    &cx,
                    IngestBatch::Default(&documents),
                    plan.active_shards,
                )
                .expect("estimate budget fit fixture")
                .expect("default budget fits fixture")
                .projected_logical_upper_bound;
            writer.reader.config.scribe_shard_budget_bytes = projected + 1;

            let receipt = writer
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
                .expect("run budget-fit shared-nothing ingest")
                .expect("budget-fit fixture uses shared-nothing");
            assert_eq!(receipt.route, ParallelIngestRoute::SharedNothing);
            assert_eq!(receipt.logical_budget_bytes, projected + 1);
            assert_eq!(receipt.projected_logical_upper_bound, projected);
            assert!(receipt.arena_bytes_used_high_water <= projected);
            assert!(receipt.arena_bytes_used_high_water < receipt.logical_budget_bytes);
            assert!(receipt.arena_chunk_bytes < DEFAULT_ARENA_CHUNK_BYTES);
            assert!(writer.pending_segments.is_empty());
            assert_eq!(
                writer
                    .shards
                    .iter()
                    .map(|shard| shard.accumulator.document_count())
                    .sum::<usize>(),
                documents.len(),
            );

            writer
                .commit_with_trigger(&cx, LifecycleTrigger::ExplicitFlush)
                .await
                .expect("publish budget-fit shared-nothing batch");
            assert_eq!(
                writer
                    .authority_snapshot()
                    .expect("budget-fit writer snapshot is authoritative")
                    .doc_count(),
                250
            );
        });
    });
}

#[test]
fn production_fanout_overlaps_on_an_explicit_multi_worker_pool() {
    // PRODUCTION PATH, forced overlap. The rendezvous lives inside
    // `enter_shard`, so every shard of the real `try_index_documents_parallel`
    // fan-out must be in flight before any of them may leave. Overlap is
    // therefore forced rather than hoped for, and the rendezvous deadline
    // means a serializing pool fails the assertion instead of hanging.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-thread production overlap pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("overlap index");
            let documents = parallel_budget_fixture_documents(250);
            let writer = index.writer_mut();
            // Opt in explicitly: the rendezvous is per index and off by
            // default, so no other test's scheduling is affected.
            writer.force_ingest_overlap_rendezvous = true;
            let plan = plan_parallel_ingest(
                documents.len(),
                writer.shard_router.shard_count(),
                rayon::current_num_threads(),
            )
            .expect("plan production overlap fixture");
            assert_eq!(plan.route, ParallelIngestRoute::SharedNothing);
            assert!(plan.active_shards >= 2);
            let projected = writer
                .parallel_budget_admission(
                    &cx,
                    IngestBatch::Default(&documents),
                    plan.active_shards,
                )
                .expect("estimate production overlap fixture")
                .expect("default budget fits fixture")
                .projected_logical_upper_bound;
            writer.reader.config.scribe_shard_budget_bytes = projected + 1;

            // The route arms the rendezvous from its own plan.
            let observation = ParallelBatchObservation::default();
            let receipt = writer
                .try_index_documents_parallel(
                    &cx,
                    ValidatedBatchAdmission {
                        documents: IngestBatch::Default(&documents),
                    },
                    &BTreeSet::new(),
                    false,
                    &observation,
                )
                .await
                .expect("run production shared-nothing ingest")
                .expect("fixture takes the shared-nothing route");

            let witness = receipt.worker_witness;
            assert_eq!(witness.started_shards, plan.active_shards);
            assert_eq!(witness.completed_shards, plan.active_shards);
            // The point of the whole witness: real overlap through the
            // production route, not a diversity count and not peak 1.
            assert_eq!(witness.peak_shards_in_flight, plan.active_shards);
            assert!(witness.peak_shards_in_flight >= 2);
            assert_eq!(witness.distinct_worker_slots, plan.active_shards);
            assert_eq!(witness.unidentified_shards, 0);
            assert_eq!(
                observation.sorted_worker_ids().len(),
                witness.distinct_worker_slots,
            );
        });
    });
}

#[test]
fn production_one_worker_pool_declines_the_fanout() {
    // PRODUCTION NEGATIVE. On a one-thread pool the verified capacity is 1,
    // so the planner activates one shard and the route declines outright:
    // no fan-out, no shards started, and nothing that could be mistaken for
    // realized parallelism.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build one-thread production pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("one-worker index");
            let documents = parallel_budget_fixture_documents(250);
            let writer = index.writer_mut();
            assert_eq!(rayon::current_num_threads(), 1);
            let plan = plan_parallel_ingest(
                documents.len(),
                writer.shard_router.shard_count(),
                rayon::current_num_threads(),
            )
            .expect("plan one-worker fixture");
            assert_eq!(plan.route, ParallelIngestRoute::Serial);

            let observation = ParallelBatchObservation::default();
            let declined = writer
                .try_index_documents_parallel(
                    &cx,
                    ValidatedBatchAdmission {
                        documents: IngestBatch::Default(&documents),
                    },
                    &BTreeSet::new(),
                    false,
                    &observation,
                )
                .await
                .expect("one-worker route declines without error");
            assert!(declined.is_none());
            let witness = observation.snapshot();
            assert_eq!(witness.started_shards, 0);
            assert_eq!(witness.completed_shards, 0);
            assert_eq!(witness.peak_shards_in_flight, 0);
            assert_eq!(witness.distinct_worker_slots, 0);
            // A declined route never promotes a success terminal. Read
            // from the observation: the receipt cannot carry a terminal.
            assert!(!observation.terminal().is_success());
            assert_eq!(observation.terminal(), ParallelIngestTerminal::Aborted);
        });
    });
}

#[test]
fn shard_guard_drains_and_never_promotes_success_on_failure_paths() {
    // Guard-level (not production-path): pins the RAII contract the
    // production routes rely on.
    let failed = ParallelBatchObservation::default();
    drop(failed.enter_shard());
    let failed_witness = failed.snapshot();
    assert_eq!(failed_witness.started_shards, 1);
    assert_eq!(failed_witness.completed_shards, 0);
    assert_eq!(failed_witness.peak_shards_in_flight, 1);
    assert_eq!(failed.terminal(), ParallelIngestTerminal::Aborted);
    assert!(!failed.terminal().is_success());

    // Unwinding drains too. Production converts a worker panic to an error
    // inside `catch_parallel_ingest_worker`, so this covers the guard's
    // unwind contract rather than the production panic path.
    let panicked = ParallelBatchObservation::default();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = panicked.enter_shard();
        // ubs:ignore -- Deliberate panic injection tests catch_unwind and shard cleanup below.
        panic!("shard panics mid-accumulation");
    }));
    assert!(outcome.is_err());
    let panic_witness = panicked.snapshot();
    assert_eq!(panic_witness.started_shards, 1);
    assert_eq!(panic_witness.completed_shards, 0);
    assert!(!panicked.terminal().is_success());

    // A shared-nothing decline is a routing FACT and must not move the
    // terminal: work still lies ahead when it is recorded.
    let declined = ParallelBatchObservation::default();
    declined
        .shared_nothing_declined
        .store(true, Ordering::Release);
    assert!(declined.snapshot().shared_nothing_declined);
    assert_eq!(declined.terminal(), ParallelIngestTerminal::Aborted);
    assert!(!declined.terminal().is_success());
    let cancelled = ParallelBatchObservation::default();
    cancelled.set_terminal(ParallelIngestTerminal::Cancelled);
    assert!(!cancelled.terminal().is_success());
    // Only whole-operation terminals count as success.
    for success in [
        ParallelIngestTerminal::FanoutCompleted,
        ParallelIngestTerminal::FanoutFallbackCompleted,
        ParallelIngestTerminal::SerialCompleted,
        ParallelIngestTerminal::NoWork,
    ] {
        let observation = ParallelBatchObservation::default();
        observation.set_terminal(success);
        assert!(observation.terminal().is_success());
    }

    // Classification is TYPED, not inferred from a live cancel flag: a
    // cancellation racing an unrelated error must not relabel that error.
    assert_eq!(
        ParallelIngestTerminal::from_error(&QuillIndexError::Cancelled { phase: "index" }),
        ParallelIngestTerminal::Cancelled,
    );
    assert_eq!(
        ParallelIngestTerminal::from_error(&invalid_state("worker failed")),
        ParallelIngestTerminal::Aborted,
    );
}

/// Capture the closed `scribe.ingest` span so shipping-driver tests can
/// assert what the recorder actually EMITTED, not what a private route
/// returned.
struct IngestSpanCapture {
    buffer: Arc<StdMutex<Vec<u8>>>,
}

impl Write for IngestSpanCapture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.buffer
            .lock()
            .expect("ingest span capture buffer is not poisoned")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Run `body` under a subscriber that emits closed spans, and return the
/// captured text.
fn capture_ingest_spans(body: impl FnOnce()) -> String {
    let buffer = Arc::new(StdMutex::new(Vec::<u8>::new()));
    let writer_buffer = Arc::clone(&buffer);
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_env_filter("off,frankensearch.quill=info")
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_writer(move || IngestSpanCapture {
            buffer: Arc::clone(&writer_buffer),
        })
        .finish();
    tracing::subscriber::with_default(subscriber, body);
    let captured = buffer
        .lock()
        .expect("ingest span capture buffer is not poisoned")
        .clone();
    String::from_utf8(captured).expect("captured spans are utf8")
}

/// Read one `key=value` field out of a captured span line.
///
/// Substring presence is not evidence; every assertion below compares an
/// exact parsed value.
fn ingest_span_field<'a>(captured: &'a str, key: &str) -> &'a str {
    let needle = format!("{key}=");
    let start = captured
        .find(&needle)
        // ubs:ignore -- Missing trace fields must fail this test observation helper.
        .unwrap_or_else(|| panic!("field {key} missing from:\n{captured}"))
        + needle.len();
    let rest = &captured[start..];
    // A QUOTED value runs to its closing quote and may legally contain
    // commas — `parallel_pool_local_worker_ids="0,1,2"` is ONE value, not
    // three fields. Treating a comma as an outer delimiter truncated it to
    // the first id.
    if let Some(body) = rest.strip_prefix('"') {
        let end = body
            .find('"')
            // ubs:ignore -- Malformed trace fields must fail this test observation helper.
            .unwrap_or_else(|| panic!("unterminated quoted {key} in:\n{captured}"));
        return &body[..end];
    }
    let end = rest
        .find(|c: char| [' ', ',', '}', '\n'].contains(&c))
        .unwrap_or(rest.len());
    &rest[..end]
}

fn assert_ingest_span_reports(captured: &str, terminal: &str, succeeded: bool) {
    assert_eq!(
        ingest_span_field(captured, "parallel_terminal"),
        terminal,
        "captured:\n{captured}",
    );
    assert_eq!(
        ingest_span_field(captured, "ingest_operation_succeeded"),
        succeeded.to_string(),
        "captured:\n{captured}",
    );
}

#[test]
fn shipping_driver_emits_a_receipt_on_the_empty_batch_early_exit() {
    // The empty-batch exit returns before any routing. Before the recorder
    // moved above the early exits this emitted NOTHING.
    let captured = capture_ingest_spans(|| {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(QuillConfig::default()).expect("empty-batch index");
            index
                .index_documents(&cx, &[])
                .await
                .expect("an empty batch is not an error");
        });
    });
    assert_ingest_span_reports(&captured, "no_work", true);
}

#[test]
fn shipping_driver_emits_an_aborted_receipt_on_an_admission_error() {
    // Duplicate ids inside one batch are refused by admission, which is
    // another exit that used to precede the recorder.
    let captured = capture_ingest_spans(|| {
        run_with_cx(|cx| async move {
            let index =
                QuillIndex::in_memory(QuillConfig::default()).expect("admission-error index");
            let duplicate = vec![
                IndexableDocument::new("duplicate-id", "alpha"),
                IndexableDocument::new("duplicate-id", "beta"),
            ];
            index
                .index_documents(&cx, &duplicate)
                .await
                .expect_err("a duplicate id in one batch must be refused");
        });
    });
    assert_ingest_span_reports(&captured, "aborted", false);
}

#[test]
fn shipping_driver_emits_a_cancelled_receipt_from_the_typed_error() {
    // Classified from the typed `QuillIndexError::Cancelled`, not from a
    // live flag read at drop time.
    let captured = capture_ingest_spans(|| {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(QuillConfig::default()).expect("cancellation index");
            let mut writer = index
                .lock_writer(&cx, "index writer lock")
                .await
                .expect("a live Cx must acquire the writer before cancellation");
            let cancelled = cx.clone();
            cancelled.set_cancel_requested(true);
            let documents = vec![IndexableDocument::new("cancelled-doc", "alpha beta")];
            let error = writer
                .index_documents(&cancelled, &documents)
                .await
                .expect_err("a cancelled Cx must be refused");
            assert!(matches!(
                error,
                QuillIndexError::Cancelled { phase: "index" }
            ));
        });
    });
    assert_ingest_span_reports(&captured, "cancelled", false);
}

/// The cancelled receipt above refuses on a pre-flight flag, before any
/// shard exists, so its counts are all zero and it cannot witness a drain.
/// This arms the production `ParallelIngest` checkpoint instead, so
/// cancellation lands INSIDE a worker — after the recorder and with shards
/// already in flight — through the fully public driver.
///
/// `parallel_route` is deliberately asserted ABSENT. It is recorded only
/// from a published receipt, and a cancelled batch propagates before that
/// point, so the route is success-only evidence and the terminal plus the
/// shard counts are what carry a failure.
#[cfg(feature = "conformance-internals")]
#[test]
fn shipping_driver_cancelled_mid_ingest_records_the_shard_drain() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-thread mid-ingest cancellation pool");
    let captured = pool.install(|| {
        capture_ingest_spans(|| {
            run_with_cx(|cx| async move {
                let config = QuillConfig {
                    max_ingest_shards: 4,
                    max_visibility_lag_ms: 60_000,
                    // Generous enough that the shared-nothing route is not
                    // declined on budget, so a real fan-out is under way
                    // when the checkpoint fires.
                    scribe_shard_budget_bytes: 512 * 1024 * 1024,
                    ..QuillConfig::default()
                };
                let index = QuillIndex::in_memory(config).expect("mid-ingest cancellation index");
                let documents = (0..250)
                    .map(|ordinal| {
                        IndexableDocument::new(
                            format!("cancel-mid-doc-{ordinal:05}"),
                            "alpha beta gamma delta",
                        )
                    })
                    .collect::<Vec<_>>();
                let controller = index.conformance_cancellation_controller();
                controller
                    .arm(ConformanceCancellationStage::ParallelIngest, 3)
                    .expect("arm mid-worker cancellation");
                let error = index
                    .index_documents(&cx, &documents)
                    .await
                    .expect_err("an armed parallel worker checkpoint must cancel the batch");
                assert!(
                    matches!(
                        error,
                        QuillIndexError::Cancelled {
                            phase: "parallel index worker"
                        }
                    ),
                    "cancellation must be observed by a worker, not earlier: {error:?}",
                );
                assert!(controller.fired());
                controller.disarm();
            });
        })
    });

    assert_ingest_span_reports(&captured, "cancelled", false);
    assert!(
        !captured.contains("parallel_route="),
        "a cancelled batch publishes no route receipt, captured:\n{captured}",
    );
    let started = ingest_span_field(&captured, "parallel_started_shards")
        .parse::<u64>()
        .expect("started shards is a count");
    let completed = ingest_span_field(&captured, "parallel_completed_shards")
        .parse::<u64>()
        .expect("completed shards is a count");
    let peak = ingest_span_field(&captured, "parallel_peak_shards_in_flight")
        .parse::<u64>()
        .expect("peak shards in flight is a count");
    assert!(
        started >= 1,
        "cancellation must land after a shard started, captured:\n{captured}",
    );
    // The drain witness, and the whole reason this case is distinct from
    // the pre-flight refusal above: a cancelled worker returns without
    // counting a completion, so completed must trail started. Both are
    // zero on the pre-flight path, which satisfies no strict inequality.
    assert!(
        completed < started,
        "a cancelled shard must not be counted complete: {completed} of {started}, \
         captured:\n{captured}",
    );
    assert!(
        (1..=started).contains(&peak),
        "peak {peak} must be a real in-flight count no greater than {started}, \
         captured:\n{captured}",
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_identity_degraded"),
        "false",
        "captured:\n{captured}",
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_unidentified_shards"),
        "0",
        "captured:\n{captured}",
    );
}

/// The non-cancel counterpart of the drain test above, and the only case
/// in this file where a shard that really started fails without being
/// cancelled. `parallel_worker_panic_is_a_typed_precommit_failure` proves
/// the same typed class but calls the catch helper directly, so it never
/// reaches the recorder, the span, or the driver's terminal
/// classification.
///
/// The armed stage leaves `Cx` untouched, which is what separates this
/// from cancellation: an operation that ends on a typed non-cancel error
/// must classify `aborted`, not `cancelled`.
#[cfg(feature = "conformance-internals")]
#[test]
fn shipping_driver_worker_failure_mid_ingest_records_an_aborted_drain() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-thread mid-ingest worker-failure pool");
    let captured = pool.install(|| {
        capture_ingest_spans(|| {
            run_with_cx(|cx| async move {
                let config = QuillConfig {
                    max_ingest_shards: 4,
                    max_visibility_lag_ms: 60_000,
                    scribe_shard_budget_bytes: 512 * 1024 * 1024,
                    ..QuillConfig::default()
                };
                let index = QuillIndex::in_memory(config).expect("mid-ingest worker-failure index");
                let documents = (0..250)
                    .map(|ordinal| {
                        IndexableDocument::new(
                            format!("fail-mid-doc-{ordinal:05}"),
                            "alpha beta gamma delta",
                        )
                    })
                    .collect::<Vec<_>>();
                let controller = index.conformance_cancellation_controller();
                controller
                    .arm(ConformanceCancellationStage::ParallelIngestWorkerFailure, 3)
                    .expect("arm mid-worker typed failure");
                let error = index
                    .index_documents(&cx, &documents)
                    .await
                    .expect_err("an armed worker-failure checkpoint must fail the batch");
                assert!(
                    matches!(
                        &error,
                        QuillIndexError::InvalidState { detail }
                            if detail == "parallel ingest worker failed before transactional commit"
                    ),
                    "expected the typed worker failure, got: {error:?}",
                );
                assert!(controller.fired());
                // The whole point of the seam: a failure is not a
                // cancellation, so nothing may have requested one.
                assert!(
                    !cx.is_cancel_requested(),
                    "a worker failure must not request cancellation on the real Cx",
                );
                controller.disarm();
            });
        })
    });

    assert_ingest_span_reports(&captured, "aborted", false);
    assert!(
        !captured.contains("parallel_route="),
        "a failed batch publishes no route receipt, captured:\n{captured}",
    );
    let started = ingest_span_field(&captured, "parallel_started_shards")
        .parse::<u64>()
        .expect("started shards is a count");
    let completed = ingest_span_field(&captured, "parallel_completed_shards")
        .parse::<u64>()
        .expect("completed shards is a count");
    let peak = ingest_span_field(&captured, "parallel_peak_shards_in_flight")
        .parse::<u64>()
        .expect("peak shards in flight is a count");
    assert!(
        started >= 1,
        "the failure must land after a shard started, captured:\n{captured}",
    );
    assert!(
        completed < started,
        "a failed shard must not be counted complete: {completed} of {started}, \
         captured:\n{captured}",
    );
    assert!(
        (1..=started).contains(&peak),
        "peak {peak} must be a real in-flight count no greater than {started}, \
         captured:\n{captured}",
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_identity_degraded"),
        "false",
        "captured:\n{captured}",
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_unidentified_shards"),
        "0",
        "captured:\n{captured}",
    );
}

#[test]
fn shipping_driver_success_receipt_binds_route_terminal_and_real_overlap() {
    // 250 documents at width 4 activate exactly 3 shards, and the route
    // arms the rendezvous from that plan, so every shard must be in flight
    // together. Every assertion below is an exact parsed value, never a
    // substring presence check.
    const EXPECTED_ACTIVE_SHARDS: &str = "3";
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-thread shipping-driver pool");
    let captured = pool.install(|| {
        capture_ingest_spans(|| {
            run_with_cx(|cx| async move {
                let config = QuillConfig {
                    max_ingest_shards: 4,
                    max_visibility_lag_ms: 60_000,
                    // Generous enough that the shared-nothing route is not
                    // declined on budget, so this exercises the fan-out.
                    scribe_shard_budget_bytes: 512 * 1024 * 1024,
                    ..QuillConfig::default()
                };
                let mut index = QuillIndex::in_memory(config).expect("success receipt index");
                index.writer_mut().force_ingest_overlap_rendezvous = true;
                let documents = (0..250)
                    .map(|ordinal| {
                        IndexableDocument::new(
                            format!("receipt-doc-{ordinal:05}"),
                            "alpha beta gamma delta",
                        )
                    })
                    .collect::<Vec<_>>();
                index
                    .index_documents(&cx, &documents)
                    .await
                    .expect("index the success-receipt batch");
            });
        })
    });

    // Exact route and terminal, not "a success of some kind".
    assert_eq!(
        ingest_span_field(&captured, "parallel_route"),
        "shared_nothing",
        "captured:\n{captured}",
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_terminal"),
        "fanout_completed",
        "captured:\n{captured}",
    );
    assert_eq!(
        ingest_span_field(&captured, "ingest_operation_succeeded"),
        "true",
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_active_shards"),
        EXPECTED_ACTIVE_SHARDS,
    );
    // Nonzero started shards, and REAL overlap: peak equals the activated
    // width because the rendezvous held every shard in flight at once.
    assert_eq!(
        ingest_span_field(&captured, "parallel_started_shards"),
        EXPECTED_ACTIVE_SHARDS,
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_completed_shards"),
        EXPECTED_ACTIVE_SHARDS,
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_peak_shards_in_flight"),
        EXPECTED_ACTIVE_SHARDS,
        "the shipping driver must show real overlap, captured:\n{captured}",
    );
    // Worker identities were captured, not merely counted.
    assert_eq!(
        ingest_span_field(&captured, "parallel_distinct_worker_slots"),
        EXPECTED_ACTIVE_SHARDS,
    );
    let worker_ids = ingest_span_field(&captured, "parallel_pool_local_worker_ids");
    assert_eq!(worker_ids.split(',').count(), 3, "ids were {worker_ids}");
    assert!(worker_ids.split(',').all(|id| id.parse::<usize>().is_ok()));
    assert_eq!(
        ingest_span_field(&captured, "parallel_identity_degraded"),
        "false",
    );
    assert_eq!(
        ingest_span_field(&captured, "parallel_unidentified_shards"),
        "0",
    );
}

#[test]
fn shipping_fanout_builds_simultaneous_budget_seals_in_parallel() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-thread budget-seal pool");
    let captured = pool.install(|| {
        capture_ingest_spans(|| {
            run_with_cx(|cx| async move {
                let mut index = QuillIndex::in_memory(QuillConfig {
                    max_ingest_shards: 4,
                    max_visibility_lag_ms: 60_000,
                    scribe_shard_budget_bytes: 512 * 1024 * 1024,
                    ..QuillConfig::default()
                })
                .expect("budget-seal index");
                let first = (0..256)
                    .map(|ordinal| {
                        IndexableDocument::new(
                            format!("budget-seal-first-{ordinal:05}"),
                            "alpha beta gamma delta",
                        )
                    })
                    .collect::<Vec<_>>();
                index
                    .index_documents(&cx, &first)
                    .await
                    .expect("populate every production ingest shard");

                let writer = index.writer_mut();
                let populated_shards = writer
                    .shards
                    .iter()
                    .filter(|shard| shard.accumulator.document_count() != 0)
                    .count();
                assert_eq!(populated_shards, 4);
                let current_maximum = writer
                    .shards
                    .iter()
                    .map(|shard| shard.accumulator.bytes_used())
                    .max()
                    .expect("writer has ingest shards");
                writer.reader.config.scribe_shard_budget_bytes = current_maximum + 1;

                let second = (0..1_024)
                    .map(|ordinal| {
                        IndexableDocument::new(
                            format!("budget-seal-second-{ordinal:05}"),
                            "budget seal ".repeat(1_000),
                        )
                    })
                    .collect::<Vec<_>>();
                index
                    .index_documents(&cx, &second)
                    .await
                    .expect("cross every shard budget through the shipping driver");

                let committed = index.commit(&cx).await.expect("publish budget-seal index");
                assert_eq!(committed.doc_count(), 1_280);
                assert_eq!(committed.segments().len(), 4);
                let seal_sequences = committed
                    .loaded_manifest()
                    .manifest
                    .segments
                    .iter()
                    .map(|segment| segment.seal_seq)
                    .collect::<Vec<_>>();
                assert_eq!(seal_sequences, vec![1, 2, 3, 4]);
                assert_pairwise_disjoint_manifest(&committed.loaded_manifest().manifest.segments);
                drop(committed);
                assert_eq!(
                    index
                        .search_paginated(&cx, "budget", 1_024, 0, true)
                        .expect("search committed budget-seal index")
                        .total_count,
                    Some(1_024),
                );
            });
        })
    });

    let parallel_budget_seals = captured
        .lines()
        .filter(|line| {
            line.contains("action=\"install_parallel_build\"")
                && line.contains("trigger=\"arena_budget\"")
        })
        .count();
    assert_eq!(
        parallel_budget_seals, 4,
        "every simultaneously crossing shard must use the parallel build path:\n{captured}",
    );
}

#[test]
fn poisoned_worker_id_lock_fails_closed_on_every_branch() {
    // Really poison the mutex — do not set the flag by hand, or the
    // PoisonError branches never execute.
    let observation = Arc::new(ParallelBatchObservation::default());
    assert!(!observation.snapshot().identity_degraded);
    let poisoner = Arc::clone(&observation);
    // The panic hook is process-global: silencing it would race every
    // concurrently running test and leak if this thread unwound before
    // restoring it. The expected panic message on stderr is the cheaper
    // price, so the hook is left alone.
    let join_result = std::thread::spawn(move || {
        let _guard = poisoner
            .worker_ids
            .lock()
            .expect("lock is not yet poisoned");
        // ubs:ignore -- Deliberate panic injection poisons the test lock; join failure is asserted.
        panic!("expected: poisoning the worker-id lock for the fail-closed test");
    })
    .join();
    assert!(join_result.is_err());
    assert!(observation.worker_ids.is_poisoned());

    // WRITE branch: `enter_shard` must run inside a pool so a worker index
    // exists and the lock is actually attempted.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build poison-branch pool");
    let writer = Arc::clone(&observation);
    pool.install(|| {
        drop(writer.enter_shard());
    });
    assert!(
        observation.identity_degraded.load(Ordering::Acquire),
        "a failed identity write must fail closed",
    );

    // READ branches: both report degraded and empty rather than a
    // confident zero.
    let witness = observation.snapshot();
    assert!(witness.identity_degraded);
    assert_eq!(witness.distinct_worker_slots, 0);
    assert!(observation.sorted_worker_ids().is_empty());
    // The shard itself was still counted; only identity was lost.
    assert_eq!(witness.started_shards, 1);
}

#[test]
fn observation_worker_indices_are_pool_local_and_collapse() {
    let first = ParallelBatchObservation::default();
    let second = ParallelBatchObservation::default();
    assert_ne!(first.snapshot().run_seq, second.snapshot().run_seq);

    // Indices are pool-local: on a one-thread pool every shard reports
    // index 0, and repeats collapse to a single slot.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build one-thread identity pool");
    // The observation must be built INSIDE the pool, or `pool_threads`
    // captures the ambient global width and the assertion degenerates into
    // global-vs-global while the ids came from the explicit pool.
    let witness = pool.install(|| {
        let observation = ParallelBatchObservation::default();
        (0..3_usize).into_par_iter().for_each(|_| {
            observation.enter_shard().completed();
        });
        assert_eq!(observation.sorted_worker_ids(), vec![0]);
        observation.snapshot()
    });
    assert_eq!(witness.distinct_worker_slots, 1);
    assert_eq!(witness.started_shards, 3);
    // The explicit pool's width, not the ambient global one.
    assert_eq!(witness.pool_threads, 1);
    assert!(!witness.identity_degraded);
}

#[test]
fn shared_nothing_receipt_binds_join_and_realized_worker_counts() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build four-worker witness pool");
    pool.install(|| {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                max_visibility_lag_ms: 60_000,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("worker witness index");
            let documents = parallel_budget_fixture_documents(250);
            let writer = index.writer_mut();
            let plan = plan_parallel_ingest(
                documents.len(),
                writer.shard_router.shard_count(),
                rayon::current_num_threads(),
            )
            .expect("plan worker witness fixture");
            assert_eq!(plan.route, ParallelIngestRoute::SharedNothing);
            let projected = writer
                .parallel_budget_admission(
                    &cx,
                    IngestBatch::Default(&documents),
                    plan.active_shards,
                )
                .expect("estimate worker witness fixture")
                .expect("default budget fits fixture")
                .projected_logical_upper_bound;
            writer.reader.config.scribe_shard_budget_bytes = projected + 1;

            let observation = ParallelBatchObservation::default();
            let receipt = writer
                .try_index_documents_parallel(
                    &cx,
                    ValidatedBatchAdmission {
                        documents: IngestBatch::Default(&documents),
                    },
                    &BTreeSet::new(),
                    false,
                    &observation,
                )
                .await
                .expect("run shared-nothing ingest")
                .expect("250 documents at width 4 take the shared-nothing route");
            let actual_hashes = writer
                .shards
                .iter()
                .flat_map(|shard| shard.identities.iter())
                .map(|identity| (identity.document_id.as_str(), identity.content_hash))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(actual_hashes.len(), documents.len());
            for document in &documents {
                assert_eq!(
                    actual_hashes.get(document.id.as_str()).copied(),
                    Some(
                        indexable_document_content_hash(document)
                            .expect("derive independent canonical document witness"),
                    ),
                    "prepared metadata must remain paired with document {}",
                    document.id,
                );
            }
            // Identities are retained, not just counted.
            assert_eq!(
                observation.sorted_worker_ids().len(),
                receipt.worker_witness.distinct_worker_slots,
            );

            assert_eq!(receipt.route, ParallelIngestRoute::SharedNothing);
            // Derived from the plan, not hard-coded: the fixture's active
            // width is a function of the configured shard count, and this
            // test is about the witness, not the planner arithmetic that
            // `adaptive_parallel_planner_v1_covers_frozen_matrix` pins.
            assert_eq!(receipt.active_shards, plan.active_shards);
            assert!(receipt.active_shards >= 2);
            let witness = receipt.worker_witness;
            // Drain witness: every shard that started also finished, so
            // none was abandoned between fan-out and the commit table.
            // These count SHARDS, not workers.
            assert_eq!(witness.started_shards, receipt.active_shards);
            assert_eq!(witness.completed_shards, receipt.active_shards);
            assert_eq!(witness.pool_threads, rayon::current_num_threads());
            assert!(witness.unidentified_shards <= witness.completed_shards);
            assert!(
                witness.distinct_worker_slots
                    <= witness.completed_shards - witness.unidentified_shards,
            );
            // Overlap is OBSERVED, never assumed. Rayon may legally run
            // every shard on one worker here, so this pins the invariants;
            // forced-overlap and one-thread cases are proven exactly in
            // `overlap_witness_records_real_rayon_overlap_and_one_thread_negative`.
            assert!(witness.peak_shards_in_flight >= 1);
            assert!(witness.peak_shards_in_flight <= witness.completed_shards);
            assert_eq!(
                witness.prepared_metadata_documents,
                documents.len(),
                "the adaptive budget proof must prepare one canonical metadata image per document",
            );
            assert_eq!(
                witness.reused_metadata_documents,
                documents.len(),
                "the real shared-nothing workers must reuse every prepared metadata image",
            );
            // The receipt carries no terminal by construction: a route
            // cannot know the whole-operation outcome. Terminal lives on
            // the observation and is published only by the recorder.
            assert!(!witness.identity_degraded);
        });
    });
}
