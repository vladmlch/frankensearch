use super::*;

#[test]
fn graph_ranking_enabled_stub_is_noop() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let config = TwoTierConfig {
            graph_ranking_enabled: true,
            graph_ranking_weight: 0.9,
            ..TwoTierConfig::default()
        };
        let searcher = TwoTierSearcher::new(index, fast, config).with_lexical(lexical);

        let mut saw_initial = false;
        let metrics = searcher
            .search(
                &cx,
                "graph ranking stub query",
                5,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        saw_initial = true;
                        assert!(
                            !results.is_empty(),
                            "phase-1 results should still be present"
                        );
                    }
                },
            )
            .await
            .expect("search should succeed with stubbed graph path");

        assert!(saw_initial, "phase-1 callback should fire");
        assert!(
            metrics.phase1_total_ms >= 0.0,
            "phase-1 latency should remain well-formed"
        );
    });
}

#[test]
fn lexical_short_circuit_skips_vector_scan_for_short_keywords() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_lexical(lexical);

        let mut initial_sources = Vec::new();
        let metrics = searcher
            .search(
                &cx,
                "search",
                3,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_sources = results.iter().map(|result| result.source).collect();
                    }
                },
            )
            .await
            .expect("short-keyword search should succeed");

        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("lexical_short_circuit")
        );
        assert_eq!(metrics.phase1_vectors_searched, 0);
        assert_eq!(metrics.semantic_candidates, 0);
        assert_eq!(metrics.lexical_candidates, 3);
        assert_eq!(initial_sources, vec![ScoreSource::Lexical; 3]);
    });
}

#[test]
fn lexical_short_circuit_keeps_natural_language_hybrid() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let expected_doc_count = index.doc_count();
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_lexical(lexical);

        let metrics = searcher
            .search(
                &cx,
                "how should the search pipeline rank documents",
                3,
                |_| None,
                |_| {},
            )
            .await
            .expect("natural-language search should stay hybrid");

        assert_ne!(
            metrics.skip_reason.as_deref(),
            Some("lexical_short_circuit")
        );
        assert_eq!(metrics.phase1_vectors_searched, expected_doc_count);
        assert!(metrics.semantic_candidates > 0);
        assert_eq!(metrics.lexical_candidates, 3);
    });
}

#[test]
fn lexical_short_circuit_allows_non_semantic_natural_language_when_saturated() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast: Arc<dyn Embedder> = Arc::new(NonSemanticEmbedder::new("custom-hash", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_lexical(lexical);

        let metrics = searcher
            .search(
                &cx,
                "how should the search pipeline rank documents",
                3,
                |_| None,
                |_| {},
            )
            .await
            .expect("non-semantic saturated lexical search should succeed");

        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("lexical_short_circuit")
        );
        assert_eq!(metrics.phase1_vectors_searched, 0);
        assert_eq!(metrics.semantic_candidates, 0);
        assert_eq!(metrics.lexical_candidates, 3);
    });
}

#[test]
fn lexical_short_circuit_zero_hit_gate_requires_non_semantic_without_quality() {
    assert!(should_lexical_short_circuit(
        QueryClass::ShortKeyword,
        0,
        10,
        false,
        false,
        false
    ));
    assert!(!should_lexical_short_circuit(
        QueryClass::ShortKeyword,
        0,
        10,
        false,
        true,
        false
    ));
    assert!(!should_lexical_short_circuit(
        QueryClass::ShortKeyword,
        0,
        10,
        false,
        false,
        true
    ));
}

#[cfg(feature = "graph")]
#[test]
fn graph_ranking_with_document_graph_can_add_graph_only_candidate() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let config = TwoTierConfig {
            graph_ranking_enabled: true,
            graph_ranking_weight: 1.0,
            ..TwoTierConfig::default()
        };
        let mut graph = DocumentGraph::new();
        graph.add_edge(
            "doc-0",
            "doc-extra",
            frankensearch_core::EdgeType::Reference,
            1.0,
        );
        let searcher =
            TwoTierSearcher::new(index, fast, config).with_document_graph(Arc::new(graph));

        let mut initial_docs = Vec::new();
        searcher
            .search(
                &cx,
                "graph only candidate query",
                10,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_docs = results.into_iter().map(|result| result.doc_id).collect();
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert!(
            initial_docs.iter().any(|doc_id| doc_id == "doc-extra"),
            "graph channel should be able to contribute graph-only candidates",
        );
    });
}

#[cfg(feature = "graph")]
#[test]
fn graph_ranking_candidates_respect_negation_filters() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let config = TwoTierConfig {
            graph_ranking_enabled: true,
            graph_ranking_weight: 1.0,
            ..TwoTierConfig::default()
        };
        let mut graph = DocumentGraph::new();
        graph.add_edge(
            "doc-0",
            "doc-extra",
            frankensearch_core::EdgeType::Reference,
            1.0,
        );
        let searcher =
            TwoTierSearcher::new(index, fast, config).with_document_graph(Arc::new(graph));

        let mut initial_docs_without_negation = Vec::new();
        searcher
            .search(
                &cx,
                "graph only candidate query",
                10,
                |doc_id| {
                    if doc_id == "doc-extra" {
                        Some("unsafe edge content".to_owned())
                    } else {
                        Some("safe content".to_owned())
                    }
                },
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_docs_without_negation =
                            results.into_iter().map(|result| result.doc_id).collect();
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert!(
            initial_docs_without_negation
                .iter()
                .any(|doc_id| doc_id == "doc-extra"),
            "graph channel should contribute graph-only candidate without negation",
        );

        let mut initial_docs_with_negation = Vec::new();
        searcher
            .search(
                &cx,
                "graph only candidate query -unsafe",
                10,
                |doc_id| {
                    if doc_id == "doc-extra" {
                        Some("unsafe edge content".to_owned())
                    } else {
                        Some("safe content".to_owned())
                    }
                },
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_docs_with_negation =
                            results.into_iter().map(|result| result.doc_id).collect();
                    }
                },
            )
            .await
            .expect("search should succeed");

        assert!(
            !initial_docs_with_negation
                .iter()
                .any(|doc_id| doc_id == "doc-extra"),
            "graph candidates matching negation should be filtered before fusion",
        );
    });
}

/// Phase-1 doc order for `build_test_index(4)`, whose doc `i` is the one-hot `e_{i mod 4}`.
/// The stub query embeds to `e_0`, so `doc-{0,4,8}` score 1.0 and the other seven score 0.0.
#[cfg(feature = "graph")]
fn initial_docs_for(config: TwoTierConfig, graph: Option<DocumentGraph>) -> Vec<String> {
    let mut docs: Vec<String> = Vec::new();
    let sink = &mut docs;
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, config);
        let searcher = match graph {
            Some(graph) => searcher.with_document_graph(Arc::new(graph)),
            None => searcher,
        };

        searcher
            .search(
                &cx,
                "neighbor smoothing query",
                10,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        *sink = results
                            .into_iter()
                            .map(|result| result.doc_id.to_string())
                            .collect();
                    }
                },
            )
            .await
            .expect("search should succeed");
    });
    docs
}

// ─── Phase-1 score corrections: hubness demotion + smoothing (bd-kdjr) ──────────────────

fn pool_searcher(config: TwoTierConfig) -> TwoTierSearcher {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    TwoTierSearcher::new(index, fast, config)
}

fn vhit(index: u32, score: f32, doc_id: &str) -> VectorHit {
    VectorHit {
        index,
        score,
        doc_id: doc_id.into(),
    }
}

/// Position of `id` in a Phase-1 result list — i.e. the rank fusion assigned it.
fn rank(docs: &[String], id: &str) -> usize {
    docs.iter()
        .position(|doc| doc.as_str() == id)
        .expect("rank helper requires the requested document")
}

/// PARITY GATE. Both corrections off (the default) ⇒ `correct_phase1_pool` returns the pool
/// untouched: same order, same score bits, and no sort — even with an `r_d` table attached.
#[test]
fn correct_phase1_pool_is_identity_when_both_corrections_are_off() {
    let hits = vec![
        vhit(0, 0.9, "doc-0"),
        vhit(4, 0.9, "doc-4"),
        vhit(1, 0.1, "doc-1"),
    ];
    let searcher = pool_searcher(TwoTierConfig::default())
        .with_hubness_table(Arc::from(vec![0.0, 0.0, 0.0, 0.0, 1.0].as_slice()));

    let out = searcher.correct_phase1_pool(hits.clone());

    assert_eq!(out.len(), hits.len());
    for (a, b) in out.iter().zip(&hits) {
        assert_eq!(a.doc_id, b.doc_id, "order must be preserved (no sort)");
        assert_eq!(a.score.to_bits(), b.score.to_bits(), "doc {}", a.doc_id);
    }
}

/// A hub (high `r_d`) must be demoted below an equally-scoring non-hub, and the pool re-sorted
/// so the demotion reaches fusion's position-assigned ranks.
#[test]
fn hubness_demotes_a_hub_below_an_equal_scoring_non_hub_and_resorts() {
    // doc-4 is the hub: r_d = 1.0 at vector-store index 4.
    let r_d = vec![0.0, 0.0, 0.0, 0.0, 1.0];
    let config = TwoTierConfig {
        hubness_beta: 0.5,
        ..TwoTierConfig::default()
    };
    let searcher = pool_searcher(config).with_hubness_table(Arc::from(r_d.as_slice()));

    // doc-4 leads on raw score (tie with doc-0, earlier position).
    let hits = vec![
        vhit(4, 1.0, "doc-4"),
        vhit(0, 1.0, "doc-0"),
        vhit(1, 0.2, "doc-1"),
    ];
    let out = searcher.correct_phase1_pool(hits);

    let ids: Vec<&str> = out.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(
        ids,
        ["doc-0", "doc-4", "doc-1"],
        "hub doc-4 (1.0 − 0.5·1.0 = 0.5) must fall behind doc-0 (1.0) and stay ahead of doc-1 (0.2)",
    );
    assert!(
        (out[1].score - 0.5).abs() < 1e-6,
        "hub score: {}",
        out[1].score
    );
    // A doc with no r_d entry / zero hubness is untouched.
    assert!((out[0].score - 1.0).abs() < 1e-6);
}

/// ORDERING GATE. Hubness runs BEFORE smoothing, so diffusion propagates de-hubbed scores.
/// `doc-1` borrows from its only `Similar` neighbour, the hub `doc-4`. Hubness-first gives
/// `doc-1 = 0.5·0.0 + 0.5·(1.0 − 0.5·1.0) = 0.25`; smoothing-first would give
/// `0.5·0.0 + 0.5·1.0 = 0.5`, letting the hub inject its inflated score into exactly the
/// specific answer the penalty exists to rescue. The exact score discriminates the two.
#[cfg(feature = "graph")]
#[test]
fn hubness_is_applied_before_smoothing_so_hubs_do_not_leak_into_neighbor_means() {
    let mut graph = DocumentGraph::new();
    graph.add_edge("doc-1", "doc-4", frankensearch_core::EdgeType::Similar, 1.0);

    let config = TwoTierConfig {
        hubness_beta: 0.5,
        neighbor_smoothing_alpha: 0.5,
        ..TwoTierConfig::default()
    };
    let searcher = pool_searcher(config)
        .with_hubness_table(Arc::from(vec![0.0, 0.0, 0.0, 0.0, 1.0].as_slice()))
        .with_document_graph(Arc::new(graph));

    let hits = vec![
        vhit(0, 1.0, "doc-0"),
        vhit(4, 1.0, "doc-4"),
        vhit(1, 0.0, "doc-1"),
    ];
    let out = searcher.correct_phase1_pool(hits);

    let score_of = |id: &str| out.iter().find(|h| h.doc_id == id).unwrap().score;
    assert!(
        (score_of("doc-1") - 0.25).abs() < 1e-6,
        "doc-1 must diffuse from the DE-HUBBED doc-4 (expected 0.25, got {}); \
         0.5 would mean smoothing ran first",
        score_of("doc-1"),
    );
    // doc-4 keeps its own demoted score (no outgoing Similar edges ⇒ α collapses).
    assert!((score_of("doc-4") - 0.5).abs() < 1e-6);
    assert!((score_of("doc-0") - 1.0).abs() < 1e-6);
    // And the pool is sorted exactly once, descending.
    let ids: Vec<&str> = out.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(ids, ["doc-0", "doc-4", "doc-1"]);
}

/// Phase-1 doc order with an `r_d` table attached — end-to-end through the real searcher, so
/// the correction is proven to be *reached* by the pipeline, not merely correct in isolation.
fn initial_docs_with_hubness(config: TwoTierConfig, hubness: Option<Arc<[f32]>>) -> Vec<String> {
    let mut docs: Vec<String> = Vec::new();
    let sink = &mut docs;
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, config);
        let searcher = match hubness {
            Some(table) => searcher.with_hubness_table(table),
            None => searcher,
        };

        searcher
            .search(
                &cx,
                "hubness correction query",
                10,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        *sink = results
                            .into_iter()
                            .map(|result| result.doc_id.to_string())
                            .collect();
                    }
                },
            )
            .await
            .expect("search should succeed");
    });
    docs
}

/// END-TO-END GATE. `doc-0`, `doc-4` and `doc-8` all score 1.0 (they are `e_0`, the query) and
/// the top-k heap hands them back as `doc-8, doc-0, doc-4`. Marking the *leader* `doc-8` a hub
/// must push it behind both equal-scoring non-hubs in the FUSED output, while keeping it ahead
/// of the zero-scorers (demoted, not dropped). With `hubness_beta = 0.0` (default) the same
/// table is inert.
#[test]
fn hubness_demotes_a_hub_through_fusion_and_is_inert_by_default() {
    // r_d = 1.0 only at vector-store index 8 (doc-8), the baseline rank-0 doc.
    let mut r_d = vec![0.0f32; 10];
    r_d[8] = 1.0;
    let table: Arc<[f32]> = Arc::from(r_d.as_slice());

    let baseline = initial_docs_with_hubness(TwoTierConfig::default(), Some(table.clone()));
    assert_eq!(
        baseline,
        initial_docs_with_hubness(TwoTierConfig::default(), None),
        "beta=0.0 must make the attached r_d table inert",
    );
    assert_eq!(
        rank(&baseline, "doc-8"),
        0,
        "uncorrected, the hub leads the pool: {baseline:?}",
    );

    let corrected = initial_docs_with_hubness(
        TwoTierConfig {
            hubness_beta: 0.5,
            ..TwoTierConfig::default()
        },
        Some(table),
    );
    // 1.0 − 0.5·1.0 = 0.5: behind both 1.0 non-hubs, ahead of the 0.0 pool.
    assert!(
        rank(&corrected, "doc-8") > rank(&corrected, "doc-0")
            && rank(&corrected, "doc-8") > rank(&corrected, "doc-4"),
        "hubness must demote the hub behind the equal-scoring non-hubs: {corrected:?}",
    );
    assert!(
        rank(&corrected, "doc-8") < rank(&corrected, "doc-1"),
        "the hub is demoted, not dropped — 0.5 still beats the zero-scorers: {corrected:?}",
    );
}

/// PARITY GATE for the wiring (bd-kdjr). `neighbor_smoothing_alpha` defaults to `0.0`, so
/// attaching a document graph must not perturb Phase-1 at all — the smoothing arm is never
/// entered and `fast_hits` moves through untouched. This is what makes the wiring a
/// zero-cost default rather than a behaviour change.
#[cfg(feature = "graph")]
#[test]
fn smoothing_is_inert_by_default_even_with_a_document_graph() {
    let mut graph = DocumentGraph::new();
    graph.add_edge("doc-7", "doc-0", frankensearch_core::EdgeType::Similar, 1.0);

    let without_graph = initial_docs_for(TwoTierConfig::default(), None);
    let with_graph = initial_docs_for(TwoTierConfig::default(), Some(graph));

    assert_eq!(
        without_graph, with_graph,
        "alpha=0.0 must leave the Phase-1 ranking byte-identical",
    );
}

/// END-TO-END GATE for the wiring (bd-kdjr). `doc-7` is a zero-scoring candidate that sits on
/// a `Similar` edge to `doc-0`, a perfect-scoring hit. Diffusion lifts it to `α·1.0 = 0.5`,
/// which must carry through `neighbor_smooth_ranked`'s re-sort *and* rank-based fusion to
/// move it ahead of the untouched zero-scorers it started behind. Proves the smoothed score
/// actually reaches the final ranking rather than being discarded by position-assigned ranks.
#[cfg(feature = "graph")]
#[test]
fn smoothing_promotes_a_graph_neighbor_of_a_top_hit_through_fusion() {
    let mut graph = DocumentGraph::new();
    graph.add_edge("doc-7", "doc-0", frankensearch_core::EdgeType::Similar, 1.0);

    let smoothing = TwoTierConfig {
        neighbor_smoothing_alpha: 0.5,
        ..TwoTierConfig::default()
    };

    let baseline = initial_docs_for(TwoTierConfig::default(), Some(graph.clone()));
    let smoothed = initial_docs_for(smoothing, Some(graph));

    // `doc-2` is an isolated zero-scorer: no Similar edges, so smoothing collapses to
    // identity on it. It is the fixed reference `doc-7` must overtake.
    assert!(
        rank(&baseline, "doc-7") > rank(&baseline, "doc-2"),
        "unsmoothed, doc-7 trails the isolated zero-scorer doc-2: {baseline:?}",
    );
    assert!(
        rank(&smoothed, "doc-7") < rank(&smoothed, "doc-2"),
        "smoothing must promote doc-7 (neighbor of top hit doc-0) past doc-2: {smoothed:?}",
    );
    // The perfect-scoring hit is isolated in the graph, so diffusion must not disturb it.
    assert_eq!(rank(&smoothed, "doc-0"), 0, "top hit must stay on top");
}

#[test]
fn quality_timeout_emits_refinement_failed_with_timeout_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(PendingEmbedder::new("quality-pending", 4));
        let config = TwoTierConfig {
            quality_timeout_ms: 0,
            ..TwoTierConfig::default()
        };
        let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);

        let mut saw_initial = false;
        let mut saw_timeout = false;
        let metrics = searcher
            .search(
                &cx,
                "timeout me",
                5,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { .. } => saw_initial = true,
                    SearchPhase::RefinementFailed { error, .. } => {
                        saw_timeout = matches!(error, SearchError::SearchTimeout { .. });
                    }
                    SearchPhase::Refined { .. } | SearchPhase::Reranked { .. } => {}
                },
            )
            .await
            .expect("search should return metrics even when refinement times out");

        assert!(saw_initial, "phase 1 should still run");
        assert!(
            saw_timeout,
            "phase 2 timeout should degrade to RefinementFailed with SearchTimeout"
        );
        assert!(
            metrics
                .skip_reason
                .as_ref()
                .is_some_and(|reason| reason.contains("Search timed out")),
            "timeout should be recorded in skip reason"
        );
    });
}

#[test]
fn fast_embed_failure_with_lexical_degrades_gracefully() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);

        let searcher =
            TwoTierSearcher::new(index, embedder, TwoTierConfig::default()).with_lexical(lexical);

        let mut got_initial = false;
        let mut initial_count = 0;
        searcher
            .search(
                &cx,
                "test",
                5,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        got_initial = true;
                        initial_count = results.len();
                    }
                },
            )
            .await
            .unwrap();

        assert!(got_initial, "should fall back to lexical-only results");
        assert!(initial_count > 0, "should have lexical results");
    });
}

#[test]
fn non_semantic_fast_embedder_short_circuits_to_lexical() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder: Arc<dyn Embedder> = Arc::new(NonSemanticEmbedder::new("fnv1a-test", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher =
            TwoTierSearcher::new(index, embedder, TwoTierConfig::default()).with_lexical(lexical);

        let mut initial_results = Vec::new();
        let mut initial_skip = None;
        let metrics = searcher
            .search(
                &cx,
                "rust ownership",
                5,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial {
                        results, metrics, ..
                    } = phase
                    {
                        initial_results = results;
                        initial_skip = metrics.skip_reason;
                    }
                },
            )
            .await
            .expect("lexical short-circuit should succeed");

        assert_eq!(initial_results.len(), 3);
        assert!(
            initial_results
                .iter()
                .all(|result| matches!(result.source, ScoreSource::Lexical))
        );
        assert_eq!(metrics.phase1_vectors_searched, 0);
        assert!(metrics.fast_embed_ms.abs() < f64::EPSILON);
        assert_eq!(metrics.semantic_candidates, 0);
        assert_eq!(metrics.lexical_candidates, 3);
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("non_semantic_fast_embedder_lexical_short_circuit")
        );
        assert_eq!(
            initial_skip.as_deref(),
            Some("non_semantic_fast_embedder_lexical_short_circuit")
        );
    });
}

#[test]
fn non_semantic_fast_embedder_without_lexical_marks_vector_control() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder: Arc<dyn Embedder> = Arc::new(NonSemanticEmbedder::new("fnv1a-test", 4));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let mut initial_skip = None;
        let mut initial_sources = Vec::new();
        let mut initial_hash_control = 0;
        let metrics = searcher
            .search(
                &cx,
                "rust ownership",
                5,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial {
                        results, metrics, ..
                    } = phase
                    {
                        initial_sources = results.iter().map(|hit| hit.source).collect();
                        initial_skip = metrics.skip_reason;
                        initial_hash_control = metrics.hash_control_candidates;
                    }
                },
            )
            .await
            .expect("hash vector control path should still return hits");

        assert!(
            !initial_sources.is_empty(),
            "explicit hash control still searches the hash generation"
        );
        assert!(
            initial_sources
                .iter()
                .all(|source| matches!(source, ScoreSource::HashControl)),
            "hash control hits must not use the semantic-fast source tag"
        );
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("non_semantic_fast_embedder_vector_control")
        );
        assert_eq!(
            initial_skip.as_deref(),
            Some("non_semantic_fast_embedder_vector_control")
        );
        assert!(
            initial_hash_control > 0,
            "Initial PhaseMetrics must count hash-control hits, not hide them"
        );
        assert_eq!(metrics.hash_control_candidates, initial_hash_control);
    });
}

#[test]
fn hash_control_does_not_schedule_quality_refinement() {
    let quality_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let quality = Arc::new(CountingEmbedder::new("quality", 4, quality_calls.clone()));

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast: Arc<dyn Embedder> = Arc::new(NonSemanticEmbedder::new("fnv1a-test", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_lexical(lexical);

        let mut got_refined = false;
        let metrics = searcher
            .search(
                &cx,
                "rust ownership",
                5,
                |_| None,
                |phase| {
                    if matches!(
                        phase,
                        SearchPhase::Refined { .. } | SearchPhase::Reranked { .. }
                    ) {
                        got_refined = true;
                    }
                },
            )
            .await
            .expect("hash control with lexical should stay lexical-only");

        assert!(!got_refined);
        assert!(!searcher.should_run_quality());
        assert_eq!(metrics.phase1_vectors_searched, 0);
        assert_eq!(metrics.semantic_candidates, 0);
        assert_eq!(metrics.phase2_vectors_searched, 0);
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("non_semantic_fast_embedder_lexical_short_circuit")
        );
        assert_eq!(
            quality_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "quality must not refine hash-control ranks"
        );
    });
}

#[test]
fn hash_control_with_quality_attached_is_still_vector_control() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast: Arc<dyn Embedder> = Arc::new(NonSemanticEmbedder::new("fnv1a-test", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);

        let metrics = searcher
            .search(&cx, "rust ownership", 5, |_| None, |_| {})
            .await
            .expect("hash control search should still return hits");

        assert_eq!(metrics.phase2_vectors_searched, 0);
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("non_semantic_fast_embedder_vector_control")
        );
    });
}

#[test]
fn fast_embed_failure_without_lexical_returns_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let mut phase_count = 0;
        let err = searcher
            .search(
                &cx,
                "test",
                5,
                |_| None,
                |_| {
                    phase_count += 1;
                },
            )
            .await
            .expect_err("semantic-only fast embed failure should return an error");

        assert!(matches!(err, SearchError::EmbeddingFailed { .. }));
        assert_eq!(
            phase_count, 0,
            "hard failure in phase 1 should not emit any search phases"
        );
    });
}

#[test]
fn fast_embed_failure_with_quality_configured_skips_refinement() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
        let quality: Arc<dyn Embedder> = Arc::new(StubEmbedder::new("quality", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_lexical(lexical);

        let mut phase_count = 0;
        let mut saw_initial = false;
        let mut saw_refined = false;
        let mut saw_refinement_failed = false;
        let metrics = searcher
            .search(
                &cx,
                "test",
                5,
                |_| None,
                |phase| {
                    phase_count += 1;
                    match phase {
                        SearchPhase::Initial { .. } => saw_initial = true,
                        SearchPhase::Refined { .. } => saw_refined = true,
                        SearchPhase::Reranked { .. } => saw_refined = true,
                        SearchPhase::RefinementFailed { .. } => saw_refinement_failed = true,
                    }
                },
            )
            .await
            .expect("search should degrade gracefully");

        assert_eq!(phase_count, 1);
        assert!(saw_initial, "initial lexical fallback should be emitted");
        assert!(
            !saw_refined,
            "quality phase must be skipped without fast candidates"
        );
        assert!(
            !saw_refinement_failed,
            "skipping refinement should not emit refinement failure"
        );
        assert_eq!(metrics.skip_reason.as_deref(), Some("fast_embed_failed"));
    });
}

#[test]
fn fast_embed_cancellation_propagates_even_with_lexical() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder: Arc<dyn Embedder> = Arc::new(CancelledEmbedder);
        let lexical: Arc<dyn LexicalRead> = Arc::new(StubLexical);
        let searcher =
            TwoTierSearcher::new(index, embedder, TwoTierConfig::default()).with_lexical(lexical);

        let err = searcher
            .search(&cx, "test", 5, |_| None, |_| {})
            .await
            .expect_err("cancelled embed should propagate");

        assert!(matches!(err, SearchError::Cancelled { .. }));
    });
}

#[test]
fn lexical_cancellation_propagates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        let embedder = Arc::new(StubEmbedder::new("fast", 4));
        let lexical: Arc<dyn LexicalRead> = Arc::new(CancelledLexical);
        let searcher =
            TwoTierSearcher::new(index, embedder, TwoTierConfig::default()).with_lexical(lexical);

        let err = searcher
            .search(&cx, "test", 5, |_| None, |_| {})
            .await
            .expect_err("cancelled lexical search should propagate");

        assert!(matches!(err, SearchError::Cancelled { .. }));
    });
}
