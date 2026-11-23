use super::*;

#[test]
fn search_collect_returns_best_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

        let (results, metrics) = searcher.search_collect(&cx, "test", 5).await.unwrap();

        assert!(!results.is_empty());
        assert!(metrics.phase1_total_ms > 0.0);
    });
}

#[test]
fn search_collect_rejects_negations_without_text_provider() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

        let err = searcher
            .search_collect(&cx, "ownership -unsafe", 5)
            .await
            .expect_err("search_collect should reject exclusion queries without text provider");

        assert!(matches!(err, SearchError::QueryParseError { .. }));
        if let SearchError::QueryParseError { detail, .. } = err {
            assert!(detail.contains("text provider"));
        }
    });
}

#[test]
fn search_collect_with_text_applies_negations() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

        let (results, _) = searcher
            .search_collect_with_text(&cx, "ownership -unsafe", 10, |doc_id| {
                let text = if doc_id == "doc-0" {
                    "unsafe ownership example"
                } else {
                    "safe ownership example"
                };
                Some(text.to_owned())
            })
            .await
            .expect("search should succeed");

        assert!(!results.is_empty());
        assert!(!results.iter().any(|r| r.doc_id == "doc-0"));
    });
}

#[test]
fn exclusion_filters_semantic_results_case_insensitive() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

        let mut initial_results = Vec::new();
        searcher
            .search(
                &cx,
                "ownership -RuSt",
                10,
                |doc_id| {
                    let text = match doc_id {
                        "doc-0" => "Rust ownership and borrowing",
                        "doc-1" => "RUST lifetimes and traits",
                        _ => "safe memory patterns",
                    };
                    Some(text.to_owned())
                },
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_results = results;
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert_eq!(initial_results.len(), 8);
        assert!(!initial_results.iter().any(|r| r.doc_id == "doc-0"));
        assert!(!initial_results.iter().any(|r| r.doc_id == "doc-1"));
    });
}

#[test]
fn exclusion_filters_lexical_and_semantic_candidates_before_fusion() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_lexical(lexical);

        let mut initial_results = Vec::new();
        searcher
            .search(
                &cx,
                r#"query -unsafe NOT "danger zone""#,
                10,
                |doc_id| {
                    let text = match doc_id {
                        "doc-0" => "unsafe pointer dance",
                        "doc-1" => "all checks passed",
                        "lex-doc-0" => "contains danger zone marker",
                        "lex-doc-1" => "safe lexical candidate",
                        "lex-doc-2" => "UNSAFE lexical candidate",
                        _ => "safe semantic content",
                    };
                    Some(text.to_owned())
                },
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_results = results;
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert!(!initial_results.iter().any(|r| r.doc_id == "doc-0"));
        assert!(!initial_results.iter().any(|r| r.doc_id == "lex-doc-0"));
        assert!(!initial_results.iter().any(|r| r.doc_id == "lex-doc-2"));
        assert!(initial_results.iter().any(|r| r.doc_id == "lex-doc-1"));
    });
}

#[test]
fn exclusion_can_eliminate_all_results_without_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

        let mut initial_results = Vec::new();
        let metrics = searcher
            .search(
                &cx,
                "-unsafe",
                10,
                |_doc_id| Some("unsafe across all docs".to_owned()),
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_results = results;
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert!(initial_results.is_empty());
        assert!(metrics.phase1_total_ms > 0.0);
    });
}

#[test]
fn exclusion_full_pipeline_rust_unsafe_returns_safe_docs() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_lexical(lexical);

        let mut initial_results = Vec::new();
        searcher
            .search(
                &cx,
                "rust -unsafe",
                12,
                |doc_id| {
                    let text = match doc_id {
                        "doc-0" => "unsafe rust pointer tricks",
                        "doc-1" => "rust ownership and borrowing",
                        "lex-doc-0" => "unsafe lexical result",
                        "lex-doc-1" => "safe rust lexical result",
                        "lex-doc-2" => "safe rust patterns",
                        _ => "safe rust systems programming",
                    };
                    Some(text.to_owned())
                },
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_results = results;
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert!(
            !initial_results
                .iter()
                .any(|result| result.doc_id == "doc-0"),
            "unsafe semantic result should be filtered"
        );
        assert!(
            !initial_results
                .iter()
                .any(|result| result.doc_id == "lex-doc-0"),
            "unsafe lexical result should be filtered"
        );
        assert!(
            initial_results
                .iter()
                .all(|result| result.doc_id != "doc-0" && result.doc_id != "lex-doc-0")
        );
        assert!(
            !initial_results.is_empty(),
            "expected at least one safe rust result to remain"
        );
    });
}

#[test]
fn exclusion_filter_hydrates_each_semantic_candidate_once() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
        let hydrated_candidates = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hydrated_candidate_ids = Arc::new(std::sync::atomic::AtomicU16::new(0));
        let mut initial_results = Vec::new();

        let metrics = searcher
            .search(
                &cx,
                "rust systems -unsafe",
                10,
                |doc_id| {
                    hydrated_candidates.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let candidate_bit = match doc_id {
                        "doc-0" => 1 << 0,
                        "doc-1" => 1 << 1,
                        "doc-2" => 1 << 2,
                        "doc-3" => 1 << 3,
                        "doc-4" => 1 << 4,
                        "doc-5" => 1 << 5,
                        "doc-6" => 1 << 6,
                        "doc-7" => 1 << 7,
                        "doc-8" => 1 << 8,
                        "doc-9" => 1 << 9,
                        _ => 1 << 15,
                    };
                    hydrated_candidate_ids
                        .fetch_or(candidate_bit, std::sync::atomic::Ordering::Relaxed);
                    let text = if doc_id == "doc-0" {
                        "unsafe rust systems"
                    } else {
                        "safe rust systems"
                    };
                    Some(text.to_owned())
                },
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_results = results;
                    }
                },
            )
            .await
            .expect("negated search should succeed");

        assert_eq!(
            metrics.phase1_vectors_searched, 10,
            "the fixture must expose all ten semantic vectors to Phase 1"
        );
        assert_eq!(
            hydrated_candidates.load(std::sync::atomic::Ordering::Relaxed),
            metrics.phase1_vectors_searched,
            "exclusion filtering must perform exactly one hydration per fixture candidate"
        );
        assert_eq!(
            hydrated_candidate_ids.load(std::sync::atomic::Ordering::Relaxed),
            (1_u16 << 10) - 1,
            "exclusion filtering must hydrate every distinct fixture candidate exactly once"
        );
        assert_eq!(metrics.semantic_candidates, 9);
        assert_eq!(initial_results.len(), 9);
        assert!(
            !initial_results
                .iter()
                .any(|result| matches!(result.doc_id.as_str(), "doc-0"))
        );
    });
}

#[test]
fn exclusion_matching_normalizes_unicode_forms() {
    let parsed = ParsedQuery::parse("rust -café");
    let exclusions = to_exclusions(&parsed);
    let decomposed_text = "safe docs with caf\u{0065}\u{0301} references";
    let matched = find_negative_match(decomposed_text, &exclusions);
    assert_eq!(matched, Some("café".to_owned()));
}

#[test]
fn exclusion_term_matching_requires_word_boundaries_for_word_terms() {
    let parsed = ParsedQuery::parse("query -he");
    let exclusions = to_exclusions(&parsed);
    let text = "the theorem should stay included";
    let matched = find_negative_match(text, &exclusions);
    assert_eq!(matched, None);
}

#[test]
fn exclusion_term_matching_excludes_whole_word_occurrences() {
    let parsed = ParsedQuery::parse("query -he");
    let exclusions = to_exclusions(&parsed);
    let text = "we saw he walk home";
    let matched = find_negative_match(text, &exclusions);
    assert_eq!(matched, Some("he".to_owned()));
}

#[test]
fn exclusion_term_matching_keeps_substring_behavior_for_path_like_terms() {
    let parsed = ParsedQuery::parse("query -src/main.rs");
    let exclusions = to_exclusions(&parsed);
    let text = "candidate path=/workspace/src/main.rs.bak";
    let matched = find_negative_match(text, &exclusions);
    assert_eq!(matched, Some("src/main.rs".to_owned()));
}

#[test]
fn refined_phase_uses_zero_fast_score_for_lexical_only_candidates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_lexical(lexical);

        let mut refined_results = Vec::new();
        searcher
            .search(
                &cx,
                "query",
                12,
                |_| None,
                |phase| {
                    if let SearchPhase::Refined { results, .. } = phase {
                        refined_results = results;
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert!(
            !refined_results.is_empty(),
            "refined phase should produce results"
        );
        let lexical_only = refined_results
            .iter()
            .find(|result| result.doc_id.starts_with("lex-doc-"))
            .expect("at least one lexical-only candidate should survive top-k");
        assert!(
            lexical_only.fast_score.is_none(),
            "lexical-only refined result carries no fast-tier score, matching the \
             Initial phase and the sync searcher (the fused conversion only ever \
             assigns fast_score from the semantic pool)"
        );
        assert_eq!(
            lexical_only.source,
            ScoreSource::Lexical,
            "lexical-only refined result should retain lexical provenance when quality score is absent"
        );
        assert!(
            lexical_only
                .lexical_score
                .is_some_and(|score| score > 0.0_f32),
            "lexical-only refined result should preserve lexical score for diagnostics"
        );
    });
}

#[test]
fn fused_hits_to_scored_results_preserves_lexical_metadata() {
    let fused = vec![
        frankensearch_core::types::FusedHit {
            doc_id: "lex-doc-1".into(),
            rrf_score: 1.5,
            lexical_rank: Some(0),
            semantic_rank: None,
            hash_rank: None,
            semantic_index: None,
            lexical_score: Some(3.0),
            semantic_score: None,
            hash_score: None,
            in_both_sources: false,
        },
        frankensearch_core::types::FusedHit {
            doc_id: "sem-doc-1".into(),
            rrf_score: 1.0,
            lexical_rank: None,
            semantic_rank: Some(0),
            hash_rank: None,
            semantic_index: Some(0),
            lexical_score: None,
            semantic_score: Some(0.8),
            hash_score: None,
            in_both_sources: false,
        },
    ];
    let lexical_results = vec![ScoredResult {
        doc_id: "lex-doc-1".into(),
        score: 3.0,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(3.0),
        rerank_score: None,
        explanation: None,
        metadata: Some(Arc::new(serde_json::json!({
            "title": "Lexical doc",
            "section": "api",
        }))),
    }];

    let scored = fused_hits_to_scored_results(&fused, &lexical_results, false, "fast-test", 60.0);
    let lexical = scored
        .iter()
        .find(|result| result.doc_id == "lex-doc-1")
        .expect("lexical fused result must exist");
    assert_eq!(
        lexical.metadata,
        Some(Arc::new(serde_json::json!({
            "title": "Lexical doc",
            "section": "api",
        }))),
        "lexical metadata should be preserved through fused conversion"
    );

    let semantic = scored
        .iter()
        .find(|result| result.doc_id == "sem-doc-1")
        .expect("semantic fused result must exist");
    assert!(
        semantic.metadata.is_none(),
        "semantic-only fused result should not synthesize metadata"
    );
}

#[test]
fn host_adapter_receives_initial_and_refined_search_events() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_host_adapter(adapter.clone());

        let _ = searcher
            .search(&cx, "test query", 5, |_| None, |_| {})
            .await
            .expect("search should succeed");

        let events = adapter.telemetry_events();
        let search_events: Vec<_> = events
            .iter()
            .filter_map(|event| match &event.event {
                TelemetryEvent::Search {
                    correlation, query, ..
                } => Some((correlation, query)),
                _ => None,
            })
            .collect();
        assert_eq!(
            search_events.len(),
            2,
            "expected initial + refined search telemetry"
        );

        let (initial_event_id, root_request_id) = {
            let (correlation, query) = search_events[0];
            assert_eq!(query.phase, SearchEventPhase::Initial);
            (
                correlation.event_id.clone(),
                correlation.root_request_id.clone(),
            )
        };
        assert!(
            is_valid_ulid_like(&initial_event_id),
            "initial event id should be ULID-like"
        );
        assert!(
            is_valid_ulid_like(&root_request_id),
            "root request id should be ULID-like"
        );

        let saw_refined_event = {
            let (correlation, query) = search_events[1];
            assert_eq!(query.phase, SearchEventPhase::Refined);
            assert!(is_valid_ulid_like(&correlation.event_id));
            assert_eq!(correlation.root_request_id, root_request_id);
            assert_eq!(
                correlation.parent_event_id.as_deref(),
                Some(initial_event_id.as_str())
            );
            assert_ne!(correlation.event_id, initial_event_id);
            true
        };
        assert!(saw_refined_event, "second event should be a search event");
    });
}

#[test]
fn host_adapter_receives_refinement_failed_search_event() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(PendingEmbedder::new("quality-pending", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));
        let config = TwoTierConfig {
            quality_timeout_ms: 0,
            ..TwoTierConfig::default()
        };

        let searcher = TwoTierSearcher::new(index, fast, config)
            .with_quality_embedder(quality)
            .with_host_adapter(adapter.clone());

        let _ = searcher
            .search(&cx, "timeout query", 5, |_| None, |_| {})
            .await
            .expect("search should degrade on timeout without failing");

        let events = adapter.telemetry_events();
        let search_events: Vec<_> = events
            .iter()
            .filter_map(|event| match &event.event {
                TelemetryEvent::Search {
                    correlation,
                    query,
                    results,
                    ..
                } => Some((correlation, query, results)),
                _ => None,
            })
            .collect();
        assert_eq!(
            search_events.len(),
            2,
            "expected initial + refinement_failed search events"
        );

        let (initial_event_id, root_request_id, initial_result_count) = {
            let (correlation, query, results) = search_events[0];
            assert_eq!(query.phase, SearchEventPhase::Initial);
            assert!(is_valid_ulid_like(&correlation.event_id));
            assert!(is_valid_ulid_like(&correlation.root_request_id));
            (
                correlation.event_id.clone(),
                correlation.root_request_id.clone(),
                results.result_count,
            )
        };

        let saw_refinement_failed = {
            let (correlation, query, results) = search_events[1];
            assert_eq!(query.phase, SearchEventPhase::RefinementFailed);
            assert_eq!(results.result_count, initial_result_count);
            assert!(is_valid_ulid_like(&correlation.event_id));
            assert_eq!(correlation.root_request_id, root_request_id);
            assert_eq!(
                correlation.parent_event_id.as_deref(),
                Some(initial_event_id.as_str())
            );
            assert_ne!(correlation.event_id, initial_event_id);
            true
        };
        assert!(
            saw_refinement_failed,
            "second event should be refinement_failed"
        );
    });
}

#[test]
fn host_adapter_receives_fast_and_quality_embedding_events() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_host_adapter(adapter.clone());

        let _ = searcher
            .search(&cx, "embed telemetry", 5, |_| None, |_| {})
            .await
            .expect("search should succeed");

        let events = adapter.telemetry_events();
        let mut root_request_id = String::new();
        for event in &events {
            if let TelemetryEvent::Search { correlation, .. } = &event.event {
                root_request_id = correlation.root_request_id.clone();
                break;
            }
        }
        assert!(
            !root_request_id.is_empty(),
            "search events should provide root request id"
        );

        let embedding_events: Vec<_> = events
            .iter()
            .filter_map(|event| match &event.event {
                TelemetryEvent::Embedding {
                    correlation,
                    job,
                    embedder,
                    status,
                    ..
                } => Some((correlation, job, embedder, status)),
                _ => None,
            })
            .collect();
        assert_eq!(
            embedding_events.len(),
            2,
            "expected fast + quality embedding events"
        );

        let fast_event = embedding_events
            .iter()
            .find(|(_, job, _, _)| job.stage == EmbeddingStage::Fast);
        assert!(fast_event.is_some(), "fast embedding event missing");
        let (fast_correlation, fast_job, fast_embedder, fast_status) =
            fast_event.expect("fast event should exist");
        assert_eq!(**fast_status, EmbeddingStatus::Completed);
        assert_eq!(fast_embedder.id, "fast");
        assert_eq!(fast_embedder.tier, EmbedderTier::Fast);
        assert_eq!(fast_job.doc_count, 1);
        assert_eq!(fast_job.queue_depth, 0);
        assert_eq!(fast_correlation.root_request_id, root_request_id);
        assert!(!fast_job.job_id.is_empty(), "fast job id should be present");

        let quality_event = embedding_events
            .iter()
            .find(|(_, job, _, _)| job.stage == EmbeddingStage::Quality);
        assert!(quality_event.is_some(), "quality embedding event missing");
        let (quality_correlation, quality_job, quality_embedder, quality_status) =
            quality_event.expect("quality event should exist");
        assert_eq!(**quality_status, EmbeddingStatus::Completed);
        assert_eq!(quality_embedder.id, "quality");
        assert_eq!(quality_embedder.tier, EmbedderTier::Quality);
        assert_eq!(quality_job.doc_count, 1);
        assert_eq!(quality_job.queue_depth, 0);
        assert_eq!(quality_correlation.root_request_id, root_request_id);
        assert!(
            !quality_job.job_id.is_empty(),
            "quality job id should be present"
        );

        let snapshot = searcher.runtime_metrics_collector.snapshot();
        assert_eq!(snapshot.embedding_events_emitted, 2);
    });
}

#[test]
fn host_adapter_receives_lifecycle_and_resource_events_with_runtime_hooks() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_host_adapter(adapter.clone());

        let _ = searcher
            .search(&cx, "lifecycle telemetry", 5, |_| None, |_| {})
            .await
            .expect("search should succeed");

        let emitted = adapter.telemetry_events();
        let lifecycle_events: Vec<_> = emitted
            .iter()
            .filter_map(|envelope| match &envelope.event {
                TelemetryEvent::Lifecycle {
                    correlation,
                    state,
                    severity,
                    uptime_ms,
                    ..
                } => Some((correlation, state, severity, uptime_ms)),
                _ => None,
            })
            .collect();
        assert!(
            lifecycle_events.len() >= 3,
            "expected start/health/stop lifecycle telemetry events"
        );
        assert!(lifecycle_events.iter().any(|(_, state, severity, _)| {
            **state == LifecycleState::Started && **severity == LifecycleSeverity::Info
        }));
        assert!(
            lifecycle_events
                .iter()
                .any(|(_, state, severity, uptime_ms)| {
                    **state == LifecycleState::Stopped
                        && **severity == LifecycleSeverity::Info
                        && uptime_ms.is_some()
                })
        );
        assert!(
            lifecycle_events
                .iter()
                .any(|(_, state, _, _)| **state == LifecycleState::Healthy)
        );

        let resource_events: Vec<_> = emitted
            .iter()
            .filter_map(|envelope| match &envelope.event {
                TelemetryEvent::Resource { sample, .. } => Some(sample),
                _ => None,
            })
            .collect();
        assert!(
            !resource_events.is_empty(),
            "expected resource telemetry from phase health emission"
        );
        assert!(
            resource_events.iter().all(|sample| sample.interval_ms > 0),
            "resource interval should always be non-zero"
        );
        assert!(resource_events.iter().all(|sample| {
            sample.cpu_pct.is_finite() && (0.0..=100.0).contains(&sample.cpu_pct)
        }));
        let collector_snapshot = searcher.runtime_metrics_collector.snapshot();
        assert!(
            collector_snapshot.resource_events_emitted >= 1,
            "phase health should emit at least one resource event"
        );

        let lifecycle_hooks = adapter.lifecycle_events();
        assert!(matches!(
            lifecycle_hooks.first(),
            Some(AdapterLifecycleEvent::SessionStart { .. })
        ));
        assert!(
            lifecycle_hooks
                .iter()
                .any(|event| matches!(event, AdapterLifecycleEvent::HealthTick { .. }))
        );
        assert!(matches!(
            lifecycle_hooks.last(),
            Some(AdapterLifecycleEvent::SessionStop { .. })
        ));
    });
}

#[test]
fn host_adapter_receives_failed_fast_embedding_event() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_lexical(lexical)
            .with_host_adapter(adapter.clone());

        let _ = searcher
            .search(&cx, "fallback query", 5, |_| None, |_| {})
            .await
            .expect("search should fall back to lexical-only results");

        let events = adapter.telemetry_events();
        let embedding_events: Vec<_> = events
            .iter()
            .filter_map(|event| match &event.event {
                TelemetryEvent::Embedding {
                    job,
                    embedder,
                    status,
                    ..
                } => Some((job, embedder, status)),
                _ => None,
            })
            .collect();
        assert_eq!(
            embedding_events.len(),
            1,
            "expected one failed fast embedding event"
        );

        let (job, embedder, status) = embedding_events[0];
        assert_eq!(job.stage, EmbeddingStage::Fast);
        assert_eq!(embedder.id, "failing-embedder");
        assert_eq!(embedder.tier, EmbedderTier::Hash);
        assert_eq!(*status, EmbeddingStatus::Failed);

        let snapshot = searcher.runtime_metrics_collector.snapshot();
        assert_eq!(snapshot.embedding_events_emitted, 1);
    });
}

#[test]
fn live_search_stream_health_reflects_emitted_search_events() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_host_adapter(adapter);

        let _ = searcher
            .search(&cx, "stream health", 5, |_| None, |_| {})
            .await
            .expect("search should succeed");

        let health = searcher.live_search_stream_health();
        assert_eq!(health.emitted_total, 1);
        assert_eq!(health.buffered, 1);

        let drained = searcher.drain_live_search_stream(10);
        assert_eq!(drained.len(), 1);
        let drained_health = searcher.live_search_stream_health();
        assert_eq!(drained_health.buffered, 0);
    });
}

#[test]
fn metrics_track_query_class() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

        let (_, metrics) = searcher
            .search_collect(&cx, "how does distributed consensus work", 5)
            .await
            .unwrap();

        assert!(metrics.query_class.is_some());
        assert!(metrics.fast_embedder_id.is_some());
    });
}

#[test]
fn debug_impl_works() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
    let debug_str = format!("{searcher:?}");
    assert!(debug_str.contains("TwoTierSearcher"));
    assert!(debug_str.contains("fast"));
}

#[test]
fn metrics_exporter_receives_search_and_embedding_callbacks() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let exporter = Arc::new(RecordingExporter::default());
        let config = TwoTierConfig::default().with_metrics_exporter(exporter.clone());

        let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);
        let _ = searcher
            .search(&cx, "test query", 5, |_| None, |_| {})
            .await
            .unwrap();

        {
            let search_events = exporter.search.lock().expect("search lock");
            assert_eq!(search_events.len(), 2);
            assert!(search_events.iter().any(|m| !m.refined));
            assert!(search_events.iter().any(|m| m.refined));
        }
        {
            let embedding_events = exporter.embedding.lock().expect("embedding lock");
            assert!(embedding_events.len() >= 2);
        }
        {
            let errors = exporter.errors.lock().expect("errors lock");
            assert!(errors.is_empty());
        }
    });
}

#[test]
fn metrics_exporter_receives_degradation_errors() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let exporter = Arc::new(RecordingExporter::default());
        let config = TwoTierConfig::default().with_metrics_exporter(exporter.clone());

        let searcher = TwoTierSearcher::new(index, embedder, config).with_lexical(lexical);
        let _ = searcher
            .search(&cx, "test", 5, |_| None, |_| {})
            .await
            .unwrap();

        {
            let errors = exporter.errors.lock().expect("errors lock");
            assert!(!errors.is_empty());
        }
        {
            let search_events = exporter.search.lock().expect("search lock");
            assert_eq!(search_events.len(), 1);
            assert!(!search_events[0].refined);
        }
    });
}

// ─── Cache-wiring tests ────────────────────────────────────────────

#[test]
fn embedding_cache_wraps_fast_tier() {
    let fast_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fast = Arc::new(CountingEmbedder::new("fast", 4, fast_calls.clone()));

    let index = build_test_index(4);
    let searcher =
        TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_embedding_cache(64);

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // First search: fast embedder called
        let _ = searcher
            .search(&cx, "hello world", 5, |_| None, |_| {})
            .await
            .unwrap();
        let after_first = fast_calls.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after_first >= 1,
            "fast embedder should be called at least once"
        );

        // Same query again: should hit cache, no additional inner calls
        let _ = searcher
            .search(&cx, "hello world", 5, |_| None, |_| {})
            .await
            .unwrap();
        let after_second = fast_calls.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            after_first, after_second,
            "repeated query should hit cache (fast tier)"
        );
    });
}

#[test]
fn embedding_cache_wraps_quality_tier_when_set_before() {
    let fast_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let quality_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fast = Arc::new(CountingEmbedder::new("fast", 4, fast_calls));
    let quality = Arc::new(CountingEmbedder::new("quality", 4, quality_calls.clone()));

    let index = build_test_index_with_quality(4);
    // quality set BEFORE cache — both should be wrapped
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
        .with_quality_embedder(quality)
        .with_embedding_cache(64);

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let _ = searcher
            .search(&cx, "cache test", 5, |_| None, |_| {})
            .await
            .unwrap();
        let q_after_first = quality_calls.load(std::sync::atomic::Ordering::Relaxed);

        let _ = searcher
            .search(&cx, "cache test", 5, |_| None, |_| {})
            .await
            .unwrap();
        let q_after_second = quality_calls.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            q_after_first, q_after_second,
            "repeated query should hit cache (quality tier, set before cache)"
        );
    });
}

#[test]
fn embedding_cache_wraps_quality_tier_when_set_after() {
    let fast_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let quality_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fast = Arc::new(CountingEmbedder::new("fast", 4, fast_calls));
    let quality = Arc::new(CountingEmbedder::new("quality", 4, quality_calls.clone()));

    let index = build_test_index_with_quality(4);
    // cache set BEFORE quality — quality should still be auto-wrapped
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
        .with_embedding_cache(64)
        .with_quality_embedder(quality);

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let _ = searcher
            .search(&cx, "order test", 5, |_| None, |_| {})
            .await
            .unwrap();
        let q_after_first = quality_calls.load(std::sync::atomic::Ordering::Relaxed);

        let _ = searcher
            .search(&cx, "order test", 5, |_| None, |_| {})
            .await
            .unwrap();
        let q_after_second = quality_calls.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            q_after_first, q_after_second,
            "repeated query should hit cache (quality tier, set after cache)"
        );
    });
}

#[test]
fn different_queries_are_cache_misses() {
    let fast_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fast = Arc::new(CountingEmbedder::new("fast", 4, fast_calls.clone()));

    let index = build_test_index(4);
    let searcher =
        TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_embedding_cache(64);

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let _ = searcher
            .search(&cx, "query alpha", 5, |_| None, |_| {})
            .await
            .unwrap();
        let after_alpha = fast_calls.load(std::sync::atomic::Ordering::Relaxed);

        let _ = searcher
            .search(&cx, "query beta", 5, |_| None, |_| {})
            .await
            .unwrap();
        let after_beta = fast_calls.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after_beta > after_alpha,
            "different query should be a cache miss"
        );
    });
}
