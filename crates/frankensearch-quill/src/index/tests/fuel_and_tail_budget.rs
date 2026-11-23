use super::*;

#[cfg(feature = "conformance-internals")]
#[test]
fn public_snippet_search_cancels_at_the_post_ranking_tail_boundary() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config())
            .expect("create public snippet cancellation index");
        index
            .index_document(
                &cx,
                &IndexableDocument::new("snippet-cancel", "alpha alphabet alpine"),
            )
            .await
            .expect("index public snippet cancellation fixture");
        index
            .commit(&cx)
            .await
            .expect("publish public snippet cancellation fixture");

        let baseline = index
            .search_with_snippets(&cx, "alpha*", 10, &crate::SnippetConfig::default())
            .expect("uninterrupted public snippet search");
        assert_eq!(baseline.len(), 1);
        assert!(
            baseline[0]
                .snippet
                .as_deref()
                .is_some_and(|snippet| snippet.contains("<b>alpha</b>")),
            "the fixture must require the real snippet tail"
        );

        let controller = index.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::SnippetTailAdmission, 1)
            .expect("arm the exact post-ranking snippet-tail checkpoint");
        let cancelled =
            index.search_with_snippets(&cx, "alpha*", 10, &crate::SnippetConfig::default());
        assert!(
            matches!(cancelled, Err(QuillIndexError::Cancelled { phase }) if phase == "search"),
            "cancellation at the post-ranking checkpoint must prevent a normal enriched response: {cancelled:?}"
        );
        assert_eq!(
            controller.observed_checkpoints(),
            1,
            "the armed checkpoint must be the single post-ranking tail admission"
        );
        assert!(
            controller.fired(),
            "the real invocation Cx must be cancelled there"
        );
        controller.disarm();
    });
}

#[cfg(feature = "conformance-internals")]
#[test]
fn pinned_bundle_describes_exactly_one_publication() {
    run_with_cx(|cx| async move {
        let index =
            QuillIndex::in_memory(deterministic_config()).expect("create pinned-bundle index");
        for id in ["a", "b", "c"] {
            index
                .index_document(&cx, &IndexableDocument::new(id, "shared token"))
                .await
                .expect("index tie fixture");
        }
        index.commit(&cx).await.expect("publish tie fixture");

        let bundle = index
            .search_paginated_pinned(&cx, "shared", 2, 0, 10)
            .expect("pinned bundle");
        assert_eq!(bundle.observed.hits.len(), 2);
        assert_eq!(bundle.expanded.hits.len(), 3, "expanded prefix from rank 0");
        assert!(
            bundle.count.hits.is_empty(),
            "count is a separate invocation"
        );
        assert_eq!(bundle.count.total_count, Some(3));
        assert_eq!(
            bundle.observed.total_count, None,
            "ranked page is count-free"
        );
        assert_eq!(bundle.expanded.total_count, None);
        assert_eq!(
            &bundle.observed.hits[..],
            &bundle.expanded.hits[..2],
            "the page is exactly the head of the expanded prefix on the same snapshot"
        );
        assert_ne!(bundle.snapshot_sha256, [0; 32]);

        // Same publication: same digest, same epoch/generation.
        let again = index
            .search_paginated_pinned(&cx, "shared", 2, 0, 10)
            .expect("pinned bundle again");
        assert_eq!(again.snapshot_sha256, bundle.snapshot_sha256);
        assert_eq!(again.snapshot_epoch, bundle.snapshot_epoch);
        assert_eq!(again.keeper_generation, bundle.keeper_generation);

        // A new publication is a different physical snapshot.
        index
            .index_document(&cx, &IndexableDocument::new("d", "shared token"))
            .await
            .expect("index fourth document");
        index.commit(&cx).await.expect("publish successor");
        let successor = index
            .search_paginated_pinned(&cx, "shared", 2, 0, 10)
            .expect("pinned bundle on successor");
        assert_ne!(successor.snapshot_sha256, bundle.snapshot_sha256);
        assert_eq!(successor.count.total_count, Some(4));
        assert_eq!(successor.expanded.hits.len(), 4);
    });
}

/// bd-pjvl1's concurrent-refresh law for the pinned bundle, proven with
/// the real publish-pause seam rather than a sleep: while a publication
/// is stopped mid-install (`TempMovedToCurrent`), the pinned read path
/// FAILS CLOSED with the typed reconciliation error — it never blends
/// the old and new publications into one bundle — and after
/// reconciliation a fresh bundle describes exactly the successor under a
/// new physical digest, while a bundle taken before the mutation keeps
/// describing its own publication.
#[cfg(feature = "conformance-internals")]
#[test]
fn pinned_bundle_fails_closed_mid_publication_and_reconciles_to_the_successor() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("pinned-refresh Keeper directory");
        let index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create pinned-refresh index");
        for id in ["a", "b", "c"] {
            index
                .index_document(&cx, &IndexableDocument::new(id, "shared token"))
                .await
                .expect("index pinned-refresh fixture");
        }
        index
            .commit(&cx)
            .await
            .expect("publish pinned-refresh fixture");

        let before = index
            .search_paginated_pinned(&cx, "shared", 2, 0, 10)
            .expect("pinned bundle before the mutation");
        assert_eq!(before.count.total_count, Some(3));
        assert_eq!(before.expanded.hits.len(), 3);

        index
            .index_document(&cx, &IndexableDocument::new("d", "shared token"))
            .await
            .expect("stage the successor document");
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
        .expect("reach the pinned-refresh mid-install rendezvous");

        // Mid-install: the pinned path refuses typed. No partial bundle,
        // no blend of generations, no silent old-snapshot answer.
        let refused = index
            .search_paginated_pinned(&cx, "shared", 2, 0, 10)
            .expect_err("a pinned bundle mid-publication must fail closed");
        assert!(
            matches!(
                refused,
                QuillIndexError::PublicationReconciliationRequired { .. }
            ),
            "unexpected mid-publication refusal: {refused:?}"
        );
        // The earlier bundle is an immutable value describing its own
        // publication, untouched by the paused successor.
        assert_eq!(before.count.total_count, Some(3));
        assert_eq!(before.expanded.hits.len(), 3);
        assert_ne!(before.snapshot_sha256, [0; 32]);

        drop(commit);
        pause.release();
        index
            .reconcile_publication(&cx)
            .await
            .expect("reconcile the dropped mid-install publication");

        let after = index
            .search_paginated_pinned(&cx, "shared", 2, 0, 10)
            .expect("pinned bundle after reconciliation");
        assert_eq!(after.count.total_count, Some(4));
        assert_eq!(after.expanded.hits.len(), 4);
        assert_ne!(
            after.snapshot_sha256, before.snapshot_sha256,
            "the successor is a different physical publication"
        );
        assert!(after.keeper_generation > before.keeper_generation);
    });
}

#[test]
fn public_snippet_search_low_fuel_exact_term_refuses_before_tail_dictionary_work() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config())
            .expect("create public low-fuel exact-term snippet index");
        let content = (0..64)
            .map(|ordinal| format!("tailx{ordinal:03}"))
            .collect::<Vec<_>>()
            .join(" ");
        index
            .index_document(&cx, &IndexableDocument::new("snippet-fuel", content))
            .await
            .expect("index adversarial snippet exact-term fixture");
        index
            .commit(&cx)
            .await
            .expect("publish adversarial snippet exact-term fixture");

        let query = "content:tailx000";
        let ranked = index
            .search_paginated(&cx, query, 10, 0, false)
            .expect("prime the public ranked-search cache");
        assert_eq!(
            ranked.hits.len(),
            1,
            "the public ranked search must complete first"
        );

        // The cache hit makes this invocation start exactly at the snippet
        // tail. A zero budget must therefore refuse the first admitted
        // exact-term dictionary unit rather than returning an unmetered
        // snippet.
        index.reader.config.query_fuel_budget = 0;
        let first = index
            .search_with_snippets(&cx, query, 10, &crate::SnippetConfig::default())
            .expect_err("public low-fuel exact-term must not bypass the snippet-tail checkpoint");
        let second = index
            .search_with_snippets(&cx, query, 10, &crate::SnippetConfig::default())
            .expect_err("repeated public low-fuel exact-term must fail at the same boundary");

        assert_eq!(fuel_diagnostics(&first), fuel_diagnostics(&second));
        assert_eq!(
            fuel_diagnostics(&first),
            (0, 0, 0, 0, 0, 0),
            "the first exact-term tail dictionary admission must refuse before any work is recorded"
        );
    });
}

/// The post-ranking tail must meter DELTA work, not only Keeper work.
///
/// Planted negative for `d656de7d`. On a Delta-only snapshot that parent
/// computed `document_frequency_work = term_count_ceiling *
/// keeper_segment_count = 0`, and its per-term probe loop admitted one
/// unit per Keeper segment and none per Delta generation, so the whole
/// tail ran with no unit ever charged at ANY budget -- including zero.
/// Both refusals below return a full enriched response against it.
///
/// The ranked cache is primed first so this invocation starts exactly at
/// the tail: ranking then admits nothing, and every unit in the
/// diagnostics below belongs to the tail alone.
#[test]
fn public_snippet_tail_meters_delta_probes_and_per_hit_enrichment() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config())
            .expect("create Delta-only snippet tail index");
        let generation = index
            .snapshot()
            .expect("fresh durable snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let lease_base = index
            .snapshot()
            .expect("fresh durable snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .docid_high_watermark;
        let base = u32::try_from(lease_base).expect("Delta docid base fits u32");

        // Two generations whose content terms INTERLEAVE (alpha < beta <
        // gamma) and each of which is MISSING a term the other holds. A
        // probe that stopped at the first generation, or that answered
        // from one generation's ordered view, would report a different
        // document frequency for `alpha` than the two live rows below.
        let mut first =
            DeltaSegment::new(DEFAULT_SCHEMA, lease_base, usize::MAX).expect("first Delta");
        apply_sealable_delta_document(&mut first, base, "delta-alpha-one", "alpha", 1);
        apply_sealable_delta_document(&mut first, base + 1, "delta-gamma", "gamma", 1);
        let second_lease_base = first.lease_end();
        let second_base =
            u32::try_from(second_lease_base).expect("second Delta docid base fits u32");
        let mut second =
            DeltaSegment::new(DEFAULT_SCHEMA, second_lease_base, usize::MAX).expect("second Delta");
        apply_sealable_delta_document(&mut second, second_base, "delta-beta", "beta", 1);
        apply_sealable_delta_document(&mut second, second_base + 1, "delta-alpha-two", "alpha", 1);
        index
            .publish_delta_table(vec![
                Arc::new(first.freeze(generation)),
                Arc::new(second.freeze(generation)),
            ])
            .expect("publish the two live Delta generations");

        let snapshot = index
            .search_snapshot()
            .expect("Delta-only snapshot is authoritative");
        assert_eq!(
            snapshot.keeper_snapshot().segments().len(),
            0,
            "the fixture must be Delta-only, or the Keeper probe would refuse first"
        );
        assert_eq!(snapshot.delta_count(), 2, "both generations must be live");
        assert_eq!(
            snapshot.bm25_doc_freq(CONTENT_FIELD, b"alpha").unwrap(),
            2,
            "`alpha` must span BOTH generations"
        );
        assert_eq!(snapshot.bm25_doc_freq(CONTENT_FIELD, b"gamma").unwrap(), 1);
        assert_eq!(snapshot.bm25_doc_freq(CONTENT_FIELD, b"beta").unwrap(), 1);

        let ranked = index
            .search_paginated(&cx, "content:alpha", 10, 0, false)
            .expect("prime the public ranked-search cache");
        assert_eq!(
            ranked.hits.len(),
            2,
            "both interleaved generations must rank before the tail runs"
        );

        // Unchanged semantics first: a sufficient budget still enriches
        // every hit, so neither refusal below is a fixture that simply
        // cannot produce a result.
        let baseline = index
            .search_with_snippets(&cx, "content:alpha", 10, &crate::SnippetConfig::default())
            .expect("uninterrupted Delta-only snippet search");
        assert_eq!(baseline.len(), 2);
        assert!(
            baseline
                .iter()
                .all(|hit| hit.snippet.as_deref().is_some_and(|s| s.contains("alpha"))),
            "the fixture must require the real per-hit enrichment"
        );

        // One compiled term x two generations = two probe units, so a zero
        // budget must refuse the FIRST of them, before any work is
        // recorded.
        index.reader.config.query_fuel_budget = 0;
        let refused = index
            .search_with_snippets(&cx, "content:alpha", 10, &crate::SnippetConfig::default())
            .expect_err("a zero budget must not buy an unmetered Delta probe");
        assert_eq!(
            fuel_diagnostics(&refused),
            (0, 0, 0, 0, 0, 0),
            "the first Delta probe must refuse before any unit is recorded"
        );

        // Exactly the two probe units and nothing more: the per-hit
        // enrichment admission is then the first thing that cannot be
        // paid for, which pins it as metered work rather than a free tail.
        index.reader.config.query_fuel_budget = 2;
        let enrichment = index
            .search_with_snippets(&cx, "content:alpha", 10, &crate::SnippetConfig::default())
            .expect_err("per-hit enrichment must be admitted, not free");
        assert_eq!(
            fuel_diagnostics(&enrichment),
            (2, 2, 0, 2, 0, 0),
            "both Delta probes must be paid for and the refusal must land on enrichment"
        );
    });
}

/// The supported typed glob path must retain a per-generation boundary in
/// the snippet tail after ranking has completed.
///
/// The two Delta ordered views each consume one unit to expand `alpha*`.
/// With one unit remaining, the first document-frequency lookup is
/// admitted and the second is refused before its lookup. The parent
/// batched both probes, so it refused at two units instead; this exact
/// counter distinction is the planted parent-red.
#[test]
fn delta_only_glob_snippet_tail_admits_each_generation_before_its_lookup() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config())
            .expect("create Delta-only glob-tail index");
        let generation = index
            .snapshot()
            .expect("fresh durable snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let lease_base = index
            .snapshot()
            .expect("fresh durable snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .docid_high_watermark;
        let first_base = u32::try_from(lease_base).expect("first Delta docid base fits u32");
        let mut first =
            DeltaSegment::new(DEFAULT_SCHEMA, lease_base, usize::MAX).expect("first Delta");
        apply_sealable_delta_document(&mut first, first_base, "glob-tail-first", "alpha", 1);
        let second_lease_base = first.lease_end();
        let second_base =
            u32::try_from(second_lease_base).expect("second Delta docid base fits u32");
        let mut second =
            DeltaSegment::new(DEFAULT_SCHEMA, second_lease_base, usize::MAX).expect("second Delta");
        apply_sealable_delta_document(&mut second, second_base, "glob-tail-second", "alpha", 1);
        index
            .publish_delta_table(vec![
                Arc::new(first.freeze(generation)),
                Arc::new(second.freeze(generation)),
            ])
            .expect("publish two live Delta generations");

        let snapshot = index
            .search_snapshot()
            .expect("Delta-only glob-tail snapshot is authoritative");
        assert_eq!(snapshot.keeper_snapshot().segments().len(), 0);
        assert_eq!(snapshot.delta_count(), 2);
        let query = Query::glob(Vec::from([CONTENT_FIELD]), "alpha*".to_owned());
        let ranked = index
            .execute_ranked_query(&cx, &query, &snapshot, 10, 0, false, Vec::new())
            .expect("supported typed glob must rank both Delta documents before the tail");
        assert_eq!(ranked.hits.len(), 2);

        let tail_upper_bound = snippet_tail_work_upper_bound(
            &query,
            &snapshot,
            10,
            index.reader.config.glob_expansion_limit,
        )
        .expect("bound Delta-only glob tail");
        assert!(tail_upper_bound > 3, "the tail checkpoint must meter");
        let exhaust_once = || {
            let checkpoint: QueryCheckpointHandle<'_> =
                QueryCheckpoint::new(&cx, "search", 3, tail_upper_bound);
            checkpoint
                .admit(QueryWorkKind::DictionaryBlock, 0)
                .expect("enter the exact post-ranking snippet tail");
            compiled_snippet_terms(
                &checkpoint,
                &query,
                &snapshot,
                DEFAULT_SCHEMA,
                index.reader.config.glob_expansion_limit,
            )
            .expect_err("the second Delta lookup must exhaust the remaining tail fuel")
        };

        let first = exhaust_once();
        let second = exhaust_once();
        assert_eq!(fuel_diagnostics(&second), fuel_diagnostics(&first));
        assert_eq!(
            fuel_diagnostics(&first),
            (3, 3, 0, 3, 0, 0),
            "two ordered-view admissions plus the first Delta lookup are paid before the next lookup is refused"
        );
    });
}

/// A tombstone-heavy published Delta gets enough query work for every
/// physical block the public scorer can walk.
///
/// This deliberately goes through snapshot composition, the production
/// work-bound function, and the public search API. The two live rows sit
/// in three physical posting blocks, so replacing the physical count with
/// the live count makes the independently calculated bound disagree before
/// the query is allowed to claim that its exact budget is sufficient.
#[test]
fn tombstone_heavy_delta_public_query_accepts_its_physical_work_budget() {
    let cx = Cx::for_testing();
    let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
    let generation = keeper.loaded_manifest().manifest.generation;
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
    apply_alpha_delta(&mut delta, 0, "first-live", 1);
    for global_docid in 1..=300 {
        apply_alpha_delta(&mut delta, global_docid, "churn", 1);
    }
    assert_eq!(delta.physical_document_count(), 301);
    assert_eq!(delta.live_document_count(), 2);

    let frozen = Arc::new(delta.freeze(generation));
    let snapshot = QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::clone(&frozen)])
        .expect("compose tombstone-heavy Delta snapshot");
    let query = Query::term(
        Vec::from([crate::query::QueryField::new(CONTENT_FIELD, 1.0)]),
        "alpha".to_owned(),
    );
    let work_budget = query_work_upper_bound(
        &query,
        &snapshot,
        DEFAULT_SCHEMA,
        QuillConfig::default().glob_expansion_limit,
    )
    .expect("bound tombstone-heavy public query");
    let physical_blocks = u64::try_from(frozen.segment().physical_document_count())
        .expect("physical Delta rows fit u64")
        .div_ceil(128);
    let live_blocks = u64::try_from(frozen.live_document_count())
        .expect("live Delta rows fit u64")
        .div_ceil(128);
    let exact_physical_budget = 1_u64.saturating_add(physical_blocks.saturating_mul(2));
    let stale_live_budget = 1_u64.saturating_add(live_blocks.saturating_mul(2));
    assert_eq!(
        work_budget, exact_physical_budget,
        "a one-term, Delta-only query pays one segment plus both possible physical block walks"
    );
    assert!(
        work_budget > stale_live_budget,
        "the physical Delta span, not its live survivors, determines this query's work ceiling"
    );

    let index = QuillIndex::in_memory(QuillConfig {
        query_fuel_budget: work_budget,
        ..deterministic_config()
    })
    .expect("create exact-budget public index");
    assert_eq!(
        index
            .snapshot()
            .expect("exact-budget snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation,
        generation,
        "the independently composed and public genesis snapshots share one generation"
    );
    index
        .publish_delta_table(vec![frozen])
        .expect("publish tombstone-heavy Delta epoch");
    let result = index
        .search_paginated(&cx, "content:alpha", 10, 0, true)
        .expect("the public query must fit its production work budget");
    assert_eq!(result.total_count, Some(2));
    assert_eq!(result.hits.len(), 2);
}

#[test]
fn query_fuel_checkpoint_diagnostics_and_cancellation_are_deterministic() {
    fn exhaust_once(cx: &Cx) -> (u64, u64, u64, u64, u64, u64) {
        let checkpoint = QueryCheckpoint::new(cx, "fuel_test", 4, 5);
        checkpoint
            .admit(QueryWorkKind::Segment, 1)
            .expect("admit segment");
        checkpoint
            .admit(QueryWorkKind::DictionaryBlock, 1)
            .expect("admit dictionary block");
        checkpoint
            .admit(QueryWorkKind::PostingBlock, 1)
            .expect("admit posting block");
        checkpoint
            .admit(QueryWorkKind::PositionDocument, 1)
            .expect("admit position document");
        let error = checkpoint
            .admit(QueryWorkKind::PostingBlock, 1)
            .expect_err("fifth coarse unit must exhaust a four-unit budget");
        fuel_diagnostics(&QuillIndexError::from(error))
    }

    let cx = Cx::for_testing();
    let first = exhaust_once(&cx);
    assert_eq!(exhaust_once(&cx), first);
    assert_eq!(first, (4, 4, 1, 1, 1, 1));

    for (schedule, kind) in [
        QueryWorkKind::Segment,
        QueryWorkKind::DictionaryBlock,
        QueryWorkKind::PostingBlock,
        QueryWorkKind::PositionDocument,
    ]
    .into_iter()
    .enumerate()
    {
        let seed = 0xf0e1_0000_u64
            .checked_add(u64::try_from(schedule).expect("checkpoint schedule fits u64"))
            .expect("checkpoint schedule seed");
        let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(10_000));
        let region = lab.state.create_root_region(Budget::INFINITE);
        let cancelled = Arc::new(Cx::for_testing());
        let observed = Arc::new(AtomicBool::new(false));

        let query_cx = Arc::clone(&cancelled);
        let query_observed = Arc::clone(&observed);
        let (query, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !query_cx.is_cancel_requested() {
                    yield_now().await;
                }
                let checkpoint = QueryCheckpoint::new(query_cx.as_ref(), "checkpoint_test", 4, 5);
                assert!(matches!(
                    checkpoint.admit(kind, 1),
                    Err(ArgusError::QueryCancelled {
                        phase: "checkpoint_test"
                    })
                ));
                query_observed.store(true, Ordering::SeqCst);
            })
            .expect("create checkpoint query task");
        let cancel_cx = Arc::clone(&cancelled);
        let (cancel, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                yield_now().await;
                cancel_cx.set_cancel_requested(true);
            })
            .expect("create checkpoint cancellation task");

        lab.scheduler.lock().schedule(query, 0);
        lab.scheduler.lock().schedule(cancel, 0);
        let report = lab.run_until_quiescent_with_report();
        assert!(
            observed.load(Ordering::SeqCst),
            "checkpoint kind {kind:?} did not observe cancellation"
        );
        assert!(report.quiescent, "checkpoint schedule did not quiesce");
        assert!(
            report.oracle_report.all_passed(),
            "checkpoint schedule oracle failed: {:?}",
            report.oracle_report
        );
        assert!(
            report.invariant_violations.is_empty(),
            "checkpoint schedule invariants failed: {:?}",
            report.invariant_violations
        );
    }
}

#[cfg(feature = "pruning-conformance")]
#[test]
fn query_interruption_location_is_absent_outside_a_competitive_union() {
    let cx = Cx::for_testing();
    let controller = Arc::new(ConformanceCancellationController::default());
    assert_eq!(controller.query_interruption_location(), None);

    let exhausted = QueryCheckpoint::new_with_controller(
        &cx,
        "location_edge_fuel",
        0,
        1,
        Arc::clone(&controller),
    );
    assert!(matches!(
        exhausted.admit(QueryWorkKind::Segment, 1),
        Err(ArgusError::QueryFuelExhausted {
            budget: 0,
            consumed: 0,
            segments_touched: 0,
            dictionary_blocks: 0,
            posting_blocks: 0,
            position_docs: 0,
        })
    ));
    assert_eq!(controller.query_interruption_location(), None);

    controller
        .arm(ConformanceCancellationStage::QueryCollection, 1)
        .expect("arm non-union cancellation edge");
    let cancelled = QueryCheckpoint::new_with_controller(
        &cx,
        "location_edge_cancel",
        u64::MAX,
        0,
        Arc::clone(&controller),
    );
    assert!(matches!(
        cancelled.admit(QueryWorkKind::Segment, 0),
        Err(ArgusError::QueryCancelled {
            phase: "location_edge_cancel"
        })
    ));
    assert!(controller.fired());
    assert_eq!(controller.query_interruption_location(), None);
    controller.disarm();
    cx.set_cancel_requested(false);
}

/// The recorded location must carry the query-work context in effect at the
/// *firing* admission, not one sampled at any earlier admission.
///
/// `admit` now reads that context only inside the branch that fires, so this
/// pins the read point rather than the value: a context captured once and
/// reused — the failure mode moving the read introduces — records refill 1's
/// unscored `Seek` at window 0 instead of refill 2's scored `Drain` at 4096,
/// and every assertion below separates the two.
#[cfg(feature = "pruning-conformance")]
#[test]
fn query_interruption_location_binds_the_firing_admissions_context() {
    let cx = Cx::for_testing();
    let controller = Arc::new(ConformanceCancellationController::default());
    controller
        .arm(ConformanceCancellationStage::QueryCollection, 2)
        .expect("arm the second collection checkpoint");
    // `upper_bound` below `budget` keeps metering off, so both admissions
    // reach the checkpoint on the zero-unit path and charge nothing.
    let checkpoint = QueryCheckpoint::new_with_controller(
        &cx,
        "location_context_binding",
        u64::MAX,
        0,
        Arc::clone(&controller),
    );

    checkpoint.set_conformance_query_work_context(Some(ConformanceQueryWorkContext {
        refill_ordinal: 1,
        window_start: 0,
        phase: ConformanceQueryWorkPhase::Seek,
        candidate_doc: Some(7),
        candidate_scored: false,
    }));
    checkpoint
        .admit(QueryWorkKind::PostingBlock, 0)
        .expect("the first collection checkpoint must not fire");
    assert_eq!(controller.query_interruption_location(), None);

    checkpoint.set_conformance_query_work_context(Some(ConformanceQueryWorkContext {
        refill_ordinal: 2,
        window_start: 4_096,
        phase: ConformanceQueryWorkPhase::Drain,
        candidate_doc: Some(9_000),
        candidate_scored: true,
    }));
    assert!(matches!(
        checkpoint.admit(QueryWorkKind::PostingBlock, 0),
        Err(ArgusError::QueryCancelled {
            phase: "location_context_binding"
        })
    ));
    assert!(controller.fired());

    let location = controller
        .query_interruption_location()
        .expect("the firing admission records its exact location");
    assert_eq!(location.refill_ordinal(), 2);
    assert_eq!(location.window_start(), 4_096);
    assert_eq!(location.phase(), ConformanceQueryWorkPhase::Drain);
    assert_eq!(location.candidate_doc(), Some(9_000));
    assert!(location.candidate_scored());

    checkpoint.set_conformance_query_work_context(None);
    controller.disarm();
    cx.set_cancel_requested(false);
}

#[test]
fn content_hash_has_unambiguous_fields_and_canonical_metadata() {
    let first = IndexableDocument::new("ab", "c")
        .with_title("title")
        .with_metadata("zeta", "last")
        .with_metadata("alpha", "first");
    let reordered = IndexableDocument::new("ab", "c")
        .with_title("title")
        .with_metadata("alpha", "first")
        .with_metadata("zeta", "last");
    let boundary_shifted = IndexableDocument::new("a", "bc")
        .with_title("title")
        .with_metadata("alpha", "first")
        .with_metadata("zeta", "last");

    let first_hash = indexable_document_content_hash(&first).expect("hash first document");
    assert_eq!(
        first_hash,
        indexable_document_content_hash(&reordered).expect("hash reordered metadata")
    );
    assert_ne!(
        first_hash,
        indexable_document_content_hash(&boundary_shifted)
            .expect("hash field-boundary-shifted document")
    );
}
