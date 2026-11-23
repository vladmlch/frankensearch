use super::*;

// ─── Tests ──────────────────────────────────────────────────────────

#[test]
fn search_empty_query_returns_no_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let mut phases = Vec::new();
        let metrics = searcher
            .search(&cx, "", 10, |_| None, |p| phases.push(format!("{p:?}")))
            .await
            .unwrap();

        assert!(phases.is_empty());
        assert!(metrics.phase1_total_ms.abs() < f64::EPSILON);
    });
}

#[test]
fn search_zero_k_returns_no_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let mut phases = Vec::new();
        searcher
            .search(&cx, "test", 0, |_| None, |p| phases.push(format!("{p:?}")))
            .await
            .unwrap();

        assert!(phases.is_empty());
    });
}

#[test]
fn search_with_hits_leaves_zero_signal_unset() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let metrics = searcher
            .search(&cx, "test", 5, |_| None, |_| {})
            .await
            .unwrap();

        assert!(metrics.semantic_candidates > 0);
        assert_eq!(metrics.zero_signal, None);
    });
}

#[test]
fn index_classified_zero_signal_propagates_to_metrics() {
    // A zero-vector query embedding makes the index's classified fast
    // lane return an empty result with a typed reason; the searcher must
    // carry that classification into `TwoTierMetrics.zero_signal` instead
    // of collapsing it into a bare empty answer. (Index-state reasons
    // such as AllTombstoned travel the identical `index_zero_signal`
    // path and are covered by the index-crate classified tests.)
    struct ZeroEmbedder {
        dimension: usize,
    }

    impl Embedder for ZeroEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dimension;
            Box::pin(async move { Ok(vec![0.0; dim]) })
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn id(&self) -> &str {
            "zero-embedder"
        }

        fn model_name(&self) -> &str {
            "zero-embedder"
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder = Arc::new(ZeroEmbedder { dimension: 4 });
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let metrics = searcher
            .search(&cx, "test", 5, |_| None, |_| {})
            .await
            .unwrap();

        assert_eq!(metrics.semantic_candidates, 0);
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::ZeroNormQuery),
            "an empty semantic lane must say why, not just return nothing"
        );
    });
}

#[test]
fn hybrid_search_keeps_lexical_results_when_semantic_lane_has_zero_signal() {
    // Explicit hybrid policy (bd-tqhc): a semantic lane with zero signal
    // must not fail or empty a hybrid search — lexical results still
    // flow, and the typed reason still lands in the metrics so callers
    // can tell "lexical-only because the semantic lane had nothing"
    // apart from a genuinely empty corpus.
    struct ZeroQueryEmbedder;

    impl Embedder for ZeroQueryEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move { Ok(vec![0.0; 4]) })
        }

        fn dimension(&self) -> usize {
            4
        }

        fn id(&self) -> &str {
            "zero-embedder"
        }

        fn model_name(&self) -> &str {
            "zero-embedder"
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder: Arc<dyn Embedder> = Arc::new(ZeroQueryEmbedder);
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher =
            TwoTierSearcher::new(index, embedder, TwoTierConfig::default()).with_lexical(lexical);

        let mut initial_results = Vec::new();
        let metrics = searcher
            .search(
                &cx,
                "test",
                5,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_results = results;
                    }
                },
            )
            .await
            .unwrap();

        assert!(
            !initial_results.is_empty(),
            "hybrid search must keep lexical results when the semantic lane is empty"
        );
        assert!(
            initial_results
                .iter()
                .all(|result| result.doc_id.starts_with("lex-doc-")),
            "every surviving hit is lexical-sourced"
        );
        assert_eq!(metrics.semantic_candidates, 0);
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::ZeroNormQuery),
            "the hybrid path must still carry the semantic lane's typed reason"
        );
    });
}

#[test]
fn negation_excluding_every_candidate_classifies_as_filter() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        // Every document's text contains the negated term, so the index
        // produces candidates and the negation filter then removes all
        // of them: the searcher-side post-scan classification must say
        // "filter", not inherit an index-state reason.
        let metrics = searcher
            .search(
                &cx,
                "test -banned",
                5,
                |_| Some("banned text".to_owned()),
                |_| {},
            )
            .await
            .unwrap();

        assert_eq!(metrics.semantic_candidates, 0);
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::FilterEliminatedAll)
        );
    });
}

#[test]
fn search_whitespace_query_returns_no_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let mut phases = Vec::new();
        let metrics = searcher
            .search(
                &cx,
                "   \t\n  ",
                10,
                |_| None,
                |p| phases.push(format!("{p:?}")),
            )
            .await
            .unwrap();

        assert!(phases.is_empty());
        assert!(metrics.phase1_total_ms.abs() < f64::EPSILON);
    });
}

#[test]
fn search_fast_only_yields_initial_phase() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let mut phase_count = 0;
        let mut got_initial = false;
        let metrics = searcher
            .search(
                &cx,
                "test query",
                5,
                |_| None,
                |phase| {
                    phase_count += 1;
                    if matches!(phase, SearchPhase::Initial { .. }) {
                        got_initial = true;
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(phase_count, 1);
        assert!(got_initial);
        assert!(metrics.phase1_total_ms > 0.0);
        assert!(
            metrics.skip_reason.is_some(),
            "should report skip reason for no quality embedder"
        );
    });
}

/// bd-8nqz.1: BOTH the hybrid path and the direct lexical fallback
/// hydrate deferred metadata through the batch's pinned scoring snapshot.
/// The fallback previously RE-SEARCHED (a second query that could read a
/// newer generation than the one that scored the candidates — metadata
/// arrived as "direct"); it now hydrates the retained candidates in
/// place, so every lane converges on "hydrated" and no lane ever pays a
/// second query or crosses generations.
#[test]
fn deferred_lexical_metadata_hydrates_in_every_lane_via_the_pinned_batch() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let hybrid = TwoTierSearcher::new(
            build_test_index(4),
            Arc::new(StubEmbedder::new("fast", 4)),
            TwoTierConfig::default(),
        )
        .with_lexical(Arc::new(DeferredMetadataLexical));
        let (hybrid_results, _) = hybrid
            .search_collect(&cx, "natural language retrieval request", 2)
            .await
            .expect("hybrid search");
        let hybrid_lexical: Vec<_> = hybrid_results
            .iter()
            .filter(|result| result.lexical_score.is_some())
            .collect();
        assert!(!hybrid_lexical.is_empty());
        assert!(hybrid_lexical.iter().all(|result| {
            result
                .metadata
                .as_deref()
                .and_then(|metadata| metadata.get("path"))
                .and_then(serde_json::Value::as_str)
                == Some("hydrated")
        }));

        let fallback = TwoTierSearcher::new(
            build_test_index(4),
            Arc::new(FailingEmbedder),
            TwoTierConfig::default(),
        )
        .with_lexical(Arc::new(DeferredMetadataLexical));
        let (fallback_results, metrics) = fallback
            .search_collect(&cx, "natural language retrieval request", 2)
            .await
            .expect("lexical fallback");
        assert_eq!(metrics.skip_reason.as_deref(), Some("fast_embed_failed"));
        assert!(!fallback_results.is_empty());
        assert!(fallback_results.iter().all(|result| {
            result
                .metadata
                .as_deref()
                .and_then(|metadata| metadata.get("path"))
                .and_then(serde_json::Value::as_str)
                == Some("hydrated")
        }));
    });
}

#[test]
fn search_with_quality_yields_two_phases() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);

        let mut phase_count = 0;
        let mut got_initial = false;
        let mut got_refined = false;
        let mut initial_rank_bytes = Vec::new();
        let mut refined_rank_bytes = Vec::new();
        let metrics = searcher
            .search(
                &cx,
                "test query",
                5,
                |_| None,
                |phase| {
                    phase_count += 1;
                    match phase {
                        SearchPhase::Initial { results, .. } => {
                            got_initial = true;
                            initial_rank_bytes = results
                                .iter()
                                .map(|result| (result.doc_id.to_string(), result.score.to_bits()))
                                .collect();
                        }
                        SearchPhase::Refined { results, .. } => {
                            got_refined = true;
                            refined_rank_bytes = results
                                .iter()
                                .map(|result| (result.doc_id.to_string(), result.score.to_bits()))
                                .collect();
                        }
                        SearchPhase::Reranked { .. } => got_refined = true,
                        SearchPhase::RefinementFailed { .. } => {}
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(phase_count, 2);
        assert!(got_initial);
        assert!(got_refined);
        // Initial preserves the fast index's deterministic top-k tie order;
        // Refined reorders the retained equal-score candidates by canonical
        // doc_id through the blend path.
        assert_eq!(
            initial_rank_bytes,
            vec![
                ("doc-8".to_owned(), 1.0_f32.to_bits()),
                ("doc-0".to_owned(), 1.0_f32.to_bits()),
                ("doc-4".to_owned(), 1.0_f32.to_bits()),
                ("doc-9".to_owned(), 0.0_f32.to_bits()),
                ("doc-1".to_owned(), 0.0_f32.to_bits()),
            ]
        );
        assert_eq!(
            refined_rank_bytes,
            vec![
                ("doc-0".to_owned(), 1.0_f32.to_bits()),
                ("doc-4".to_owned(), 1.0_f32.to_bits()),
                ("doc-8".to_owned(), 1.0_f32.to_bits()),
                ("doc-1".to_owned(), 0.0_f32.to_bits()),
                ("doc-2".to_owned(), 0.0_f32.to_bits()),
            ]
        );
        assert!(metrics.quality_embed_ms > 0.0);
    });
}

#[test]
fn public_search_cancellation_after_initial_stops_before_quality_phase() {
    let quality_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let quality = Arc::new(CountingEmbedder::new("quality", 4, quality_calls.clone()));

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("post_initial_cancellation"));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_host_adapter(adapter.clone());

        let mut phases = Vec::new();
        let error = searcher
            .search(
                &cx,
                "test query",
                5,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { .. } => {
                        phases.push("initial");
                        cx.cancel_with(
                            asupersync::CancelKind::User,
                            Some("cancel after initial callback"),
                        );
                    }
                    SearchPhase::Refined { .. } => phases.push("refined"),
                    SearchPhase::Reranked { .. } => phases.push("reranked"),
                    SearchPhase::RefinementFailed { .. } => phases.push("refinement_failed"),
                },
            )
            .await
            .expect_err("post-Initial Cx cancellation must stop refinement");

        assert!(matches!(
            error,
            SearchError::Cancelled { phase, .. } if phase == "initial_to_quality"
        ));
        assert_eq!(phases, vec!["initial"]);
        assert_eq!(
            quality_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the quality embedder must not run after Initial cancels the real Cx"
        );

        let lifecycle_hooks = adapter.lifecycle_events();
        assert!(matches!(
            lifecycle_hooks.as_slice(),
            [
                AdapterLifecycleEvent::SessionStart { .. },
                AdapterLifecycleEvent::HealthTick { .. },
                AdapterLifecycleEvent::SessionStop { .. },
            ]
        ));

        // Counted rather than collected: the assertion below only ever asked
        // how many matched, and the predicate, its order, and what it
        // discriminates are unchanged.
        let terminal_lifecycle_events = adapter
            .telemetry_events()
            .into_iter()
            .filter(|envelope| {
                matches!(
                    &envelope.event,
                    TelemetryEvent::Lifecycle {
                        state: LifecycleState::Degraded,
                        severity: LifecycleSeverity::Warn,
                        reason: Some(reason),
                        ..
                    } if reason.starts_with("cancelled:initial_to_quality:")
                )
            })
            .count();
        assert_eq!(
            terminal_lifecycle_events, 1,
            "post-Initial cancellation must emit one terminal lifecycle telemetry event"
        );
    });
}

#[test]
fn public_search_cancellation_after_quality_embed_stops_before_vector_search() {
    let quality_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // This vector has the embedder's declared dimension but the wrong
    // returned length. `quality_scores_for_hits` would reject it if the
    // post-embed checkpoint did not stop vector work first.
    let quality = Arc::new(CancellingEmbedder::new(
        "quality",
        4,
        vec![1.0, 0.0, 0.0],
        quality_calls.clone(),
    ));

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("quality_embed_cancellation"));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_host_adapter(adapter.clone());

        let mut phases = Vec::new();
        let error = searcher
            .search(
                &cx,
                "test query",
                5,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { .. } => phases.push("initial"),
                    SearchPhase::Refined { .. } => phases.push("refined"),
                    SearchPhase::Reranked { .. } => phases.push("reranked"),
                    SearchPhase::RefinementFailed { .. } => phases.push("refinement_failed"),
                },
            )
            .await
            .expect_err("a real Cx cancellation after quality embedding must stop vector work");

        assert!(matches!(
            error,
            SearchError::Cancelled { phase, reason }
                if phase == "quality_embed_to_prf"
                    && reason == "user: cancel after quality embed"
        ));
        assert_eq!(phases, vec!["initial"]);
        assert_eq!(
            quality_calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the controlled quality embedder must be the stage that cancels the real Cx"
        );

        let lifecycle_hooks = adapter.lifecycle_events();
        assert!(matches!(
            lifecycle_hooks.as_slice(),
            [
                AdapterLifecycleEvent::SessionStart { .. },
                AdapterLifecycleEvent::HealthTick { .. },
                AdapterLifecycleEvent::SessionStop { .. },
            ]
        ));
        // Counted rather than collected, as above.
        let terminal_lifecycle_events = adapter
            .telemetry_events()
            .into_iter()
            .filter(|envelope| {
                matches!(
                    &envelope.event,
                    TelemetryEvent::Lifecycle {
                        state: LifecycleState::Degraded,
                        severity: LifecycleSeverity::Warn,
                        reason: Some(reason),
                        ..
                    } if reason.starts_with("cancelled:quality_embed_to_prf:")
                )
            })
            .count();
        assert_eq!(
            terminal_lifecycle_events, 1,
            "quality-embed cancellation must emit one terminal lifecycle telemetry event"
        );
    });
}

#[test]
fn owner_backed_quality_identity_checkpoint_preserves_uncancelled_refinement_bits_and_order() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("quality-identity-control");
        let fast_binding = artifact_binding("quality-identity-fast", 4, 41);
        let quality_binding = artifact_binding("quality-identity-quality", 4, 41);
        let index = owner_backed_two_tier_index(&dir, &fast_binding, &quality_binding);
        let fast = Arc::new(IdentityCountingEmbedder::new(
            "fast",
            in_memory_identity("quality-identity-fast", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let quality = Arc::new(IdentityCountingEmbedder::new(
            "quality",
            in_memory_identity("quality-identity-quality", 4),
            vec![0.0, 1.0, 0.0, 0.0],
        ));
        let searcher = TwoTierSearcher::new(
            index,
            Arc::clone(&fast) as Arc<dyn Embedder>,
            TwoTierConfig::default(),
        )
        .with_quality_embedder(Arc::clone(&quality) as Arc<dyn Embedder>);

        let mut phases = Vec::new();
        let mut refined = Vec::new();
        searcher
            .search(
                &cx,
                "query",
                3,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { .. } => phases.push("initial"),
                    SearchPhase::Refined { results, .. } => {
                        phases.push("refined");
                        refined = results;
                    }
                    SearchPhase::Reranked { .. } => phases.push("reranked"),
                    SearchPhase::RefinementFailed { error, .. } => {
                        panic!("owner-backed refinement must not fail: {error}");
                    }
                },
            )
            .await
            .expect("uncancelled owner-backed refinement must succeed");

        assert_eq!(phases, vec!["initial", "refined"]);
        assert_eq!(
            refined
                .iter()
                .map(|result| (result.doc_id.as_str(), result.score.to_bits()))
                .collect::<Vec<_>>(),
            vec![
                ("doc-quality-only", 1.0_f32.to_bits()),
                ("doc-near", 0.3_f32.to_bits()),
                ("doc-far", 0.0_f32.to_bits()),
            ],
            "the checkpoint must be an observation only: uncancelled owner-backed scores and order stay bit-identical"
        );
        assert_eq!(fast.embed_count(), 1);
        assert_eq!(quality.embed_count(), 1);
        assert_eq!(quality.identity_count(), 4);
    });
}

#[test]
fn public_search_cancellation_during_quality_producer_recheck_stops_before_owner_activation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("quality-identity-cancellation");
        let fast_binding = artifact_binding("quality-identity-cancel-fast", 4, 43);
        let quality_binding = artifact_binding("quality-identity-cancel-quality", 4, 43);
        let index = owner_backed_two_tier_index(&dir, &fast_binding, &quality_binding);
        let fast = Arc::new(IdentityCountingEmbedder::new(
            "fast",
            in_memory_identity("quality-identity-cancel-fast", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        // Admission, the pre-inference check and the actual bound response
        // each read the same producer. The fourth call cancels during the
        // post-response recheck, before any quality owner is activated.
        let quality = Arc::new(
            IdentityCountingEmbedder::new(
                "quality",
                in_memory_identity("quality-identity-cancel-quality", 4),
                vec![0.0, 1.0, 0.0, 0.0],
            )
            .cancel_on_identity_call(cx.clone(), 4),
        );
        let adapter = Arc::new(RecordingHostAdapter::new("quality_identity_cancellation"));
        let searcher = TwoTierSearcher::new(
            index,
            Arc::clone(&fast) as Arc<dyn Embedder>,
            TwoTierConfig::default(),
        )
        .with_quality_embedder(Arc::clone(&quality) as Arc<dyn Embedder>)
        .with_host_adapter(adapter.clone());

        let mut phases = Vec::new();
        let error = searcher
            .search(
                &cx,
                "query",
                3,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { .. } => phases.push("initial"),
                    SearchPhase::Refined { .. } => phases.push("refined"),
                    SearchPhase::Reranked { .. } => phases.push("reranked"),
                    SearchPhase::RefinementFailed { .. } => phases.push("refinement_failed"),
                },
            )
            .await
            .expect_err(
                "cancellation from the owner-backed quality identity must stop before activation",
            );

        assert!(matches!(
            error,
            SearchError::Cancelled { phase, reason }
                if phase == "quality_embed_to_prf"
                    && reason == "user: cancel after quality identity"
        ));
        assert_eq!(phases, vec!["initial"]);
        assert_eq!(fast.embed_count(), 1);
        assert_eq!(quality.embed_count(), 1);
        assert_eq!(
            quality.identity_count(),
            4,
            "the cancellation must originate in the post-response producer check"
        );
        assert!(
            adapter
                .telemetry_events()
                .iter()
                .filter_map(|envelope| {
                    match &envelope.event {
                        TelemetryEvent::Search { query, .. } => Some(query.phase),
                        _ => None,
                    }
                })
                .eq([SearchEventPhase::Initial]),
            "cancellation before owner activation must not publish Refined or RefinementFailed"
        );
        assert_single_cancelled_session_stop(&adapter, "quality_embed_to_prf");
    });
}

#[cfg(feature = "rerank")]
#[test]
fn public_search_cancellation_after_refined_stops_before_rerank_hydration() {
    let rerank_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let text_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("post_refined_cancellation"));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_reranker(Arc::new(CountingReranker::new(rerank_calls.clone())))
            .with_host_adapter(adapter.clone());

        let text_calls_for_search = text_calls.clone();
        let mut phases = Vec::new();
        let error = searcher
            .search(
                &cx,
                "test query",
                5,
                move |_| {
                    text_calls_for_search.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Some("rerank text".to_owned())
                },
                |phase| match phase {
                    SearchPhase::Initial { .. } => phases.push("initial"),
                    SearchPhase::Refined { .. } => {
                        phases.push("refined");
                        cx.cancel_with(
                            asupersync::CancelKind::User,
                            Some("cancel after refined callback"),
                        );
                    }
                    SearchPhase::Reranked { .. } => phases.push("reranked"),
                    SearchPhase::RefinementFailed { .. } => phases.push("refinement_failed"),
                },
            )
            .await
            .expect_err("post-Refined Cx cancellation must stop rerank hydration");

        assert!(matches!(
            error,
            SearchError::Cancelled { phase, reason }
                if phase == "refined_to_rerank"
                    && reason == "user: cancel after refined callback"
        ));
        assert_eq!(phases, vec!["initial", "refined"]);
        assert_eq!(
            text_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "phase-3 text hydration must not start after Refined cancels the real Cx"
        );
        assert_eq!(
            rerank_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the reranker must not start after Refined cancels the real Cx"
        );
        assert_single_cancelled_session_stop(&adapter, "refined_to_rerank");
    });
}

#[cfg(feature = "rerank")]
#[test]
fn public_search_cancellation_after_reranker_stops_before_score_mutation() {
    let rerank_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("post_reranker_cancellation"));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_reranker(Arc::new(CountingReranker::cancelling(rerank_calls.clone())))
            .with_host_adapter(adapter.clone());

        let mut phases = Vec::new();
        let mut refined_results = Vec::new();
        let error = searcher
            .search(
                &cx,
                "test query",
                5,
                |_| Some("rerank text".to_owned()),
                |phase| match phase {
                    SearchPhase::Initial { .. } => phases.push("initial"),
                    SearchPhase::Refined { results, .. } => {
                        phases.push("refined");
                        refined_results = results;
                    }
                    SearchPhase::Reranked { .. } => phases.push("reranked"),
                    SearchPhase::RefinementFailed { .. } => phases.push("refinement_failed"),
                },
            )
            .await
            .expect_err("reranker-originated Cx cancellation must stop before mutation");

        assert!(matches!(
            error,
            SearchError::Cancelled { phase, reason }
                if phase == "reranker_to_score_validation"
                    && reason == "user: cancel after reranker scores"
        ));
        assert_eq!(phases, vec!["initial", "refined"]);
        assert_eq!(
            rerank_calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the controlled reranker must return scores after cancelling the real Cx"
        );
        assert!(
            refined_results.iter().all(|result| {
                result.rerank_score.is_none() && result.source != ScoreSource::Reranked
            }),
            "no reranked score mutation may reach a public phase"
        );
        assert_single_cancelled_session_stop(&adapter, "reranker_to_score_validation");
    });
}

#[test]
fn public_search_refined_callback_cancellation_stops_before_rerank() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let adapter = Arc::new(RecordingHostAdapter::new("completion_cancellation"));
        let searcher = TwoTierSearcher::new(
            build_test_index_with_quality(4),
            Arc::new(StubEmbedder::new("fast", 4)),
            TwoTierConfig::default(),
        )
        .with_quality_embedder(Arc::new(StubEmbedder::new("quality", 4)))
        .with_host_adapter(adapter.clone());

        let mut phases = Vec::new();
        let error = searcher
            .search(
                &cx,
                "test query",
                5,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { .. } => phases.push("initial"),
                    SearchPhase::Refined { .. } => {
                        phases.push("refined");
                        cx.cancel_with(
                            asupersync::CancelKind::User,
                            Some("cancel from refined callback"),
                        );
                    }
                    SearchPhase::Reranked { .. } => phases.push("reranked"),
                    SearchPhase::RefinementFailed { .. } => phases.push("refinement_failed"),
                },
            )
            .await
            .expect_err("Refined callback cancellation must stop before reranking");

        assert!(matches!(
            error,
            SearchError::Cancelled { phase, reason }
                if phase == "refined_to_rerank"
                    && reason == "user: cancel from refined callback"
        ));
        assert_eq!(phases, vec!["initial", "refined"]);
        assert!(
            adapter
                .telemetry_events()
                .iter()
                .filter_map(|envelope| match &envelope.event {
                    TelemetryEvent::Search { query, .. } => Some(query.phase),
                    _ => None,
                })
                .eq([SearchEventPhase::Initial, SearchEventPhase::Refined]),
            "a Refined callback cancellation must not publish Reranked"
        );
        assert_single_cancelled_session_stop(&adapter, "refined_to_rerank");
    });
}

#[test]
fn configured_quality_embedder_without_quality_index_stays_initial() {
    let quality_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let quality = Arc::new(CountingEmbedder::new("quality", 4, quality_calls.clone()));

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);

        let mut phase_count = 0;
        let mut got_initial = false;
        let mut got_refined = false;
        let metrics = searcher
            .search(
                &cx,
                "test query",
                5,
                |_| None,
                |phase| {
                    phase_count += 1;
                    match phase {
                        SearchPhase::Initial { .. } => got_initial = true,
                        SearchPhase::Refined { .. } | SearchPhase::Reranked { .. } => {
                            got_refined = true;
                        }
                        SearchPhase::RefinementFailed { .. } => {}
                    }
                },
            )
            .await
            .expect("fast-tier search should succeed");

        assert_eq!(phase_count, 1);
        assert!(got_initial);
        assert!(!got_refined);
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("quality_index_unavailable")
        );
        assert_eq!(metrics.phase2_vectors_searched, 0);
        assert_eq!(
            quality_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "quality embedding must not start without a quality index"
        );
    });
}

#[test]
fn refined_phase_metrics_report_actual_fused_count() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4); // 10 docs in fixture index
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);

        let mut refined_fused_count = None;
        let mut refined_result_len = None;
        searcher
            .search(
                &cx,
                "test query",
                20, // ask for more than available docs to exercise truncation
                |_| None,
                |phase| {
                    if let SearchPhase::Refined {
                        metrics, results, ..
                    } = phase
                    {
                        refined_fused_count = Some(metrics.fused_count);
                        refined_result_len = Some(results.len());
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert_eq!(
            refined_fused_count, refined_result_len,
            "refined phase should report fused_count equal to emitted result length"
        );
    });
}

#[test]
fn initial_phase_metrics_report_fast_index_scope() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4); // 10 docs in fixture index
        let expected_doc_count = index.doc_count();
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

        let mut phase1_vectors = None;
        let mut phase1_semantic_candidates = None;
        searcher
            .search(
                &cx,
                "test query",
                3, // keep k below index size to ensure hit count != scan scope
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { metrics, .. } = phase {
                        phase1_vectors = Some(metrics.vectors_searched);
                        phase1_semantic_candidates = Some(metrics.fused_count);
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert_eq!(
            phase1_vectors,
            Some(expected_doc_count),
            "phase-1 vectors_searched should describe fast-tier search scope"
        );
        assert!(
            phase1_semantic_candidates.is_some_and(|count| count <= expected_doc_count),
            "fused candidates should not exceed index size"
        );
    });
}

#[test]
fn fast_only_config_skips_quality() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));

        let config = TwoTierConfig {
            fast_only: true,
            ..TwoTierConfig::default()
        };

        let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);

        let mut phase_count = 0;
        let metrics = searcher
            .search(&cx, "test", 5, |_| None, |_| phase_count += 1)
            .await
            .unwrap();

        assert_eq!(phase_count, 1, "fast_only should skip quality phase");
        assert_eq!(metrics.skip_reason.as_deref(), Some("fast_only"));
    });
}

#[test]
fn phase_gate_can_preempt_quality_phase() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let mut phase_gate = PhaseGate::new(PhaseGateConfig {
            timeout_queries: 1,
            ..PhaseGateConfig::default()
        });
        let _ = phase_gate.update(&PhaseObservation {
            fast_score: 1.0,
            quality_score: 0.0,
            user_signal: Some(false),
        });
        assert!(phase_gate.should_skip_quality());

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_phase_gate(phase_gate);

        let mut phase_count = 0;
        let metrics = searcher
            .search(&cx, "test", 5, |_| None, |_| phase_count += 1)
            .await
            .expect("search should succeed");

        assert_eq!(phase_count, 1, "phase gate should skip quality phase");
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("phase_gate_skip_quality")
        );
    });
}

#[test]
fn phase_gate_updates_after_refinement_and_can_skip_later_queries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_phase_gate_config(PhaseGateConfig {
                timeout_queries: 1,
                ..PhaseGateConfig::default()
            });

        let mut first_phase_count = 0;
        let first_metrics = searcher
            .search(&cx, "test", 5, |_| None, |_| first_phase_count += 1)
            .await
            .expect("first search should succeed");
        assert_eq!(
            first_phase_count, 2,
            "first query should still run refinement before the gate learns"
        );
        assert!(
            first_metrics.skip_reason.is_none(),
            "first query should not report skip"
        );

        let mut second_phase_count = 0;
        let second_metrics = searcher
            .search(&cx, "test", 5, |_| None, |_| second_phase_count += 1)
            .await
            .expect("second search should succeed");
        assert_eq!(
            second_phase_count, 1,
            "second query should reflect the learned gate decision"
        );
        assert_eq!(
            second_metrics.skip_reason.as_deref(),
            Some("phase_gate_skip_quality")
        );
    });
}

#[test]
fn search_params_override_plumbs_through_phase1_vector_scan() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index.clone(), fast, TwoTierConfig::default())
            .with_search_params(SearchParams {
                parallel_enabled: false,
                parallel_threshold: usize::MAX,
                parallel_chunk_size: 1,
            });

        let (results, metrics) = searcher
            .search_collect(&cx, "test", 5)
            .await
            .expect("search should succeed");
        assert!(!results.is_empty());
        assert_eq!(
            metrics.phase1_vectors_searched,
            index.doc_count(),
            "phase-1 should still scan the same candidate set with override params"
        );
    });
}
