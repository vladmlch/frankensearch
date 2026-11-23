use super::*;

#[test]
fn refined_promotes_quality_candidates_beyond_displayed_top_k() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_promotion_index();
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let config = TwoTierConfig {
            candidate_multiplier: 2,
            ..TwoTierConfig::default()
        };
        let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);

        let mut initial_display = Vec::new();
        let mut refined_results = Vec::new();
        let metrics = searcher
            .search(
                &cx,
                "query",
                2,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { results, .. } => initial_display = results,
                    SearchPhase::Refined { results, .. } => refined_results = results,
                    _ => {}
                },
            )
            .await
            .expect("search should succeed");

        assert_eq!(initial_display.len(), 2);
        assert!(
            initial_display
                .iter()
                .all(|result| result.doc_id != "doc-c"),
            "doc-c starts outside the displayed top-k: {initial_display:?}"
        );
        assert_eq!(
            refined_results.first().map(|result| result.doc_id.as_str()),
            Some("doc-c"),
            "quality tier must promote doc-c past the Initial page boundary"
        );
        assert_eq!(
            refined_results
                .iter()
                .find(|result| result.doc_id == "doc-c")
                .and_then(|result| result.quality_score),
            Some(1.0),
            "promoted candidate carries its quality evidence"
        );
        // Quality scoring covered the full retained pool (k * multiplier),
        // not just the displayed page.
        assert_eq!(metrics.phase2_vectors_searched, 4);
    });
}

#[test]
fn refined_ordering_requires_lexical_re_fusion() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_promotion_index();
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let config = TwoTierConfig {
            candidate_multiplier: 2,
            ..TwoTierConfig::default()
        };
        let searcher = TwoTierSearcher::new(index, fast, config)
            .with_quality_embedder(quality)
            .with_lexical(Arc::new(StubLexical));

        let mut initial_display = Vec::new();
        let mut refined_results = Vec::new();
        searcher
            .search(
                &cx,
                "natural language retrieval query",
                2,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { results, .. } => initial_display = results,
                    SearchPhase::Refined { results, .. } => refined_results = results,
                    _ => {}
                },
            )
            .await
            .expect("search should succeed");

        // doc-c is absent from the Initial page but must surface in the
        // refined page via quality promotion...
        assert!(
            initial_display
                .iter()
                .all(|result| result.doc_id != "doc-c"),
            "doc-c starts outside the displayed top-k: {initial_display:?}"
        );
        assert!(
            refined_results
                .iter()
                .any(|result| result.doc_id == "doc-c"),
            "refined page must promote doc-c: {refined_results:?}"
        );
        // ...and the lexical-only candidate must survive refinement with
        // its provenance and score intact — impossible under the old
        // blend-only handoff, where a lexical-only doc's reconstructed
        // fast score collapsed to 0.0 and it fell out of the page.
        let lexical_only = refined_results
            .iter()
            .find(|result| result.doc_id.starts_with("lex-doc-"))
            .expect("lexical-only candidate must survive lexical re-fusion");
        assert_eq!(lexical_only.source, ScoreSource::Lexical);
        assert!(
            lexical_only
                .lexical_score
                .is_some_and(|score| score > 0.0_f32),
            "lexical score is preserved through re-fusion"
        );
        assert!(lexical_only.fast_score.is_none());
    });
}

// ─── Test Helpers ───────────────────────────────────────────────────

/// bd-3gslf probe: a facade searcher over a Quill index finds the code
/// that contains `HashMap::new` for that query. Quill's parser drops an
/// unknown `field:` fragment, so without the backend's `query_text` the
/// lexical lane searched for nothing.
#[cfg(feature = "quill")]
#[test]
fn quill_lexical_lane_keeps_code_identifiers_literal() {
    use frankensearch_core::traits::LexicalWrite;
    use frankensearch_core::types::IndexableDocument;

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("quill dir");
        let quill = frankensearch_quill::QuillIndex::create(
            &cx,
            dir.path(),
            frankensearch_quill::QuillConfig::default(),
        )
        .await
        .expect("create quill index");
        let docs = [
            IndexableDocument::new("lex-code", "fn build() { let seen = HashMap::new(); }"),
            IndexableDocument::new("lex-note", "a new idea for the garden"),
        ];
        LexicalWrite::index_documents(&quill, &cx, &docs)
            .await
            .expect("index documents");
        LexicalWrite::commit(&quill, &cx).await.expect("commit");
        let lexical: Arc<dyn LexicalRead> = Arc::new(quill);
        assert!(
            lexical
                .search(&cx, "HashMap::new", 5)
                .await
                .unwrap()
                .is_empty(),
            "the raw query names an unknown field"
        );
        let searcher = TwoTierSearcher::new(
            build_test_index(4),
            Arc::new(StubEmbedder::new("fast", 4)),
            TwoTierConfig::default(),
        )
        .with_lexical(lexical);
        let (results, metrics) = searcher
            .search_collect(&cx, "HashMap::new", 5)
            .await
            .expect("search");
        assert!(metrics.lexical_candidates > 0, "{metrics:?}");
        assert_eq!(
            results
                .iter()
                .find(|hit| hit.doc_id.starts_with("lex-"))
                .map(|hit| hit.doc_id.as_str()),
            Some("lex-code"),
            "{results:?}"
        );
    });
}

/// bd-ctzo C1: an owner-backed index refuses a query from a different
/// embedding space BEFORE the embedder runs, and admits and serves the
/// query from the space it was actually written in.
///
/// The positive control and the refusal share one fixture and differ in
/// exactly one thing: which model produced the query embedder. Without
/// this guard the refused case is the dangerous one — same dimension, so
/// every vector API accepts it, and the search returns confident
/// nonsense.
#[test]
fn owner_backed_index_refuses_a_foreign_space_before_embedding() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("foreign-space");
        let binding = artifact_binding("ctzo-searcher-model", 4, 17);
        let index = owner_backed_index(&dir, &binding);

        // Positive control: the embedder that produced this index.
        let matching = Arc::new(IdentityCountingEmbedder::new(
            "matching",
            in_memory_identity("ctzo-searcher-model", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let searcher = TwoTierSearcher::new(
            Arc::clone(&index),
            Arc::clone(&matching) as Arc<dyn Embedder>,
            TwoTierConfig::default(),
        );
        let mut results = Vec::new();
        searcher
            .search(
                &cx,
                "query",
                2,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results: hits, .. } = phase {
                        results = hits;
                    }
                },
            )
            .await
            .expect("the producing embedder is admitted");
        assert_eq!(results.first().map(|r| r.doc_id.as_str()), Some("doc-a"));
        assert_eq!(
            matching.embed_count(),
            1,
            "an admitted query must actually embed"
        );

        // Same dimension, different space: refused, and NOTHING ran.
        let foreign = Arc::new(IdentityCountingEmbedder::new(
            "foreign",
            in_memory_identity("a-different-model", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let searcher = TwoTierSearcher::new(
            index,
            Arc::clone(&foreign) as Arc<dyn Embedder>,
            TwoTierConfig::default(),
        );
        let error = searcher
            .search(&cx, "query", 2, |_| None, |_| {})
            .await
            .expect_err("a foreign embedding space must be refused");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "query_embedding.fast.space_identity"
            ),
            "got {error:?}"
        );
        assert_eq!(
            foreign.embed_count(),
            0,
            "the refusal must precede the embedder, not follow it"
        );
    });
}

/// bd-ctzo C1: a certified-compatible foreign producer is comparison-grade
/// telemetry and never an admission basis, so it is refused here too —
/// again before any inference.
#[test]
fn owner_backed_index_refuses_a_certified_compatible_producer_before_embedding() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("compatible-producer");
        let binding = artifact_binding("ctzo-producer-model", 4, 23);
        let index = owner_backed_index(&dir, &binding);

        let mut compatible = in_memory_identity("ctzo-producer-model", 4);
        "a-different-implementation".clone_into(&mut compatible.producer.backend);
        let reference = in_memory_identity("ctzo-producer-model", 4);
        assert_eq!(
            compatible.space.fingerprint(),
            reference.space.fingerprint(),
            "the fixture is only meaningful while the SPACE is unchanged"
        );
        assert_ne!(
            compatible.producer.fingerprint(),
            reference.producer.fingerprint(),
            "the fixture is only meaningful while the PRODUCER differs"
        );
        assert_eq!(
            compatible.producer.golden_vectors, reference.producer.golden_vectors,
            "the fixture is only meaningful while both producers are certified compatible"
        );

        let embedder = Arc::new(IdentityCountingEmbedder::new(
            "compatible",
            compatible,
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let searcher = TwoTierSearcher::new(
            index,
            Arc::clone(&embedder) as Arc<dyn Embedder>,
            TwoTierConfig::default(),
        );
        let error = searcher
            .search(&cx, "query", 2, |_| None, |_| {})
            .await
            .expect_err("a certified-compatible producer must be refused");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "search_activation.fast.producer_conformance"
            ),
            "got {error:?}"
        );
        assert_eq!(embedder.embed_count(), 0, "refused before any inference");
    });
}

/// bd-ctzo C1: the legacy path must survive. Every canonical production
/// index on disk today is a v1 artifact that retains no owner and carries
/// no space identity, and an embedder that declares no identity bundle is
/// still a legal `Embedder`. Neither may be turned into a refusal by this
/// guard — the guard binds only where there is an admitted identity to
/// bind to.
#[test]
fn legacy_unidentified_index_stays_searchable_without_any_identity() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index(4);
        assert!(
            index.fast_admitted_binding().is_none(),
            "the fixture is only meaningful while the index is legacy-unidentified"
        );
        // StubEmbedder deliberately does not implement `identity()`.
        let embedder = Arc::new(StubEmbedder::new("stub-fast", 4));
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let mut initial = Vec::new();
        searcher
            .search(
                &cx,
                "query",
                3,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial = results;
                    }
                },
            )
            .await
            .expect("a legacy index stays searchable");
        assert!(
            !initial.is_empty(),
            "the legacy lane must still return results"
        );
    });
}

#[test]
fn reopened_v1_checks_both_producers_before_embedding_or_cache() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("v1-producer-reopen");
        let identity = in_memory_identity("same-model-revision", 4);
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).unwrap();
        builder
            .set_fast_embedder_id("same-id")
            .set_quality_embedder_id("same-id");
        builder.set_fast_identity(&identity).unwrap();
        builder.set_quality_identity(&identity).unwrap();
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        builder
            .add_quality_record("doc-quality-only", &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        drop(builder.finish().unwrap());
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).unwrap());
        assert!(index.fast_admitted_binding().is_none());
        assert!(index.quality_admitted_binding().is_none());
        assert!(index.fast_declared_identity().is_none());
        let matching = Arc::new(IdentityCountingEmbedder::new(
            "same-id",
            identity.clone(),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let searcher = TwoTierSearcher::new(
            Arc::clone(&index),
            matching.clone(),
            TwoTierConfig::default(),
        )
        .with_quality_embedder(matching.clone())
        .with_embedding_cache(8);
        let mut phases = Vec::new();
        searcher
            .search(
                &cx,
                "query",
                2,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { results, .. } => {
                        phases.push("Initial");
                        assert_eq!(results[0].doc_id.as_str(), "doc-a");
                    }
                    SearchPhase::Refined { results, .. } => {
                        phases.push("Refined");
                        assert_eq!(results[0].doc_id.as_str(), "doc-a");
                        assert!(
                            results.iter().any(|hit| hit.doc_id == "doc-quality-only"),
                            "quality retrieval must reach beyond the fast index"
                        );
                    }
                    phase => panic!("unexpected phase {phase:?}"),
                },
            )
            .await
            .unwrap();
        assert_eq!(phases, ["Initial", "Refined"]);
        assert_eq!(matching.embed_count(), 2);

        let mut filtered_phases = 0;
        searcher
            .search(
                &cx,
                "query -excluded",
                2,
                |id| {
                    Some(
                        if id == "doc-quality-only" {
                            "excluded"
                        } else {
                            "safe"
                        }
                        .to_owned(),
                    )
                },
                |phase| match phase {
                    SearchPhase::Initial { results, .. } | SearchPhase::Refined { results, .. } => {
                        assert_eq!(
                            results.len(),
                            1,
                            "quality must not reintroduce an excluded document"
                        );
                        assert_eq!(results[0].doc_id.as_str(), "doc-a");
                        filtered_phases += 1;
                    }
                    phase => panic!("unexpected filtered phase {phase:?}"),
                },
            )
            .await
            .unwrap();
        assert_eq!(filtered_phases, 2);

        let mut changed = identity.clone();
        "different-execution-backend".clone_into(&mut changed.producer.backend);
        assert_eq!(changed.space, identity.space);
        for tier in ["fast", "quality"] {
            let foreign = Arc::new(IdentityCountingEmbedder::new(
                "same-id",
                changed.clone(),
                vec![1.0, 0.0, 0.0, 0.0],
            ));
            let fast: Arc<dyn Embedder> = if tier == "fast" {
                foreign.clone()
            } else {
                matching.clone()
            };
            let quality: Arc<dyn Embedder> = if tier == "quality" {
                foreign.clone()
            } else {
                matching.clone()
            };
            let searcher = TwoTierSearcher::new(Arc::clone(&index), fast, TwoTierConfig::default())
                .with_quality_embedder(quality)
                .with_embedding_cache(8);
            let calls_before = matching.embed_count();
            let mut phase_count = 0;
            let error = searcher
                .search(&cx, "query", 1, |_| None, |_| phase_count += 1)
                .await
                .expect_err("same model and width do not admit another producer");
            assert!(
                matches!(error, SearchError::InvalidConfig { ref field, .. }
                if field == &format!("search_activation.{tier}.producer_revision")),
                "{error}"
            );
            assert_eq!(foreign.embed_count(), 0);
            assert_eq!(matching.embed_count(), calls_before);
            assert_eq!(phase_count, 0);
        }
        for revision in ["", identity.space.immutable_revision.as_str()] {
            assert!(
                matches!(
                    admit_legacy_tier_embedder(
                        matching.as_ref(),
                        Some(&identity),
                        revision,
                        "fast",
                    ),
                    Err(SearchError::InvalidConfig { field, .. })
                        if field == "search_activation.fast.producer_revision"
                ),
                "an old header cannot prove this identified semantic producer"
            );
        }
    });
}

#[test]
fn identified_queries_use_actual_bound_values_across_reopened_layouts_and_raw_caches() {
    const QUERY: &str = "find the quality document beyond the fast pool";
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for layout in ["stamped-v1", "exact-v2", "native-v2"] {
            let index = query_admission_index(layout);
            for cached in [false, true] {
                let fast = Arc::new(
                    query_admission_embedder("fast")
                        .with_bound_response_values(vec![0.0, 1.0, 0.0, 0.0])
                        .with_async_bound_response(),
                );
                let mut quality = query_admission_embedder("quality").with_async_bound_response();
                // Raw inference would match doc-a; the bound response must
                // retrieve doc-quality-only and survive PRF in its 6D space.
                quality.vector = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
                quality.bound_response_values = Some(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
                let quality = Arc::new(quality);
                let (fast_provider, quality_provider): (Arc<dyn Embedder>, Arc<dyn Embedder>) =
                    if cached {
                        let fast_cache = Arc::new(CachedEmbedder::new(fast.clone(), 8));
                        let quality_cache = Arc::new(CachedEmbedder::new(quality.clone(), 8));
                        for _ in 0..2 {
                            assert_eq!(fast_cache.embed(&cx, QUERY).await.unwrap(), fast.vector);
                            assert_eq!(
                                quality_cache.embed(&cx, QUERY).await.unwrap(),
                                quality.vector
                            );
                        }
                        assert_eq!(fast.embed_count(), 1);
                        assert_eq!(quality.embed_count(), 1);
                        (fast_cache, quality_cache)
                    } else {
                        (fast.clone(), quality.clone())
                    };
                let searcher = TwoTierSearcher::new(
                    Arc::clone(&index),
                    fast_provider,
                    TwoTierConfig::default(),
                )
                .with_quality_embedder(quality_provider)
                .with_prf_config(PrfConfig {
                    enabled: true,
                    alpha: 0.5,
                    top_k_feedback: 1,
                    min_feedback_docs: 1,
                    score_weighted: false,
                });
                for _ in 0..2 {
                    let mut phases = Vec::new();
                    searcher
                        .search(
                            &cx,
                            QUERY,
                            3,
                            |_| None,
                            |phase| match phase {
                                SearchPhase::Initial { results, .. } => {
                                    assert_eq!(results[0].doc_id, "doc-b", "{layout}");
                                    assert_eq!(results[0].fast_score, Some(1.0));
                                    phases.push("initial");
                                }
                                SearchPhase::Refined { results, .. } => {
                                    let quality_hit = results
                                        .iter()
                                        .find(|hit| hit.doc_id == "doc-quality-only")
                                        .expect("independent bound quality retrieval");
                                    let score = quality_hit.quality_score.unwrap();
                                    assert!(
                                        score > 0.0 && score < 1.0,
                                        "PRF must preserve the 6D query space"
                                    );
                                    phases.push("refined");
                                }
                                phase => panic!("{layout}: unexpected phase {phase:?}"),
                            },
                        )
                        .await
                        .unwrap();
                    assert_eq!(phases, ["initial", "refined"]);
                }
                for provider in [&fast, &quality] {
                    let bound_calls = provider.bound_embeds.load(Ordering::Relaxed);
                    assert_eq!(bound_calls, if cached { 1 } else { 2 });
                    assert_eq!(provider.embed_count(), bound_calls + u64::from(cached));
                }
            }
        }
    });
}

#[test]
fn identified_query_response_refusals_cannot_score_or_hide_behind_lexical_results() {
    const QUERY: &str = "find the quality document beyond the fast pool";
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for layout in ["stamped-v1", "exact-v2", "native-v2"] {
            let index = query_admission_index(layout);
            for tier in ["fast", "quality"] {
                for cached in [false, true] {
                    for fault in [
                        "space",
                        "producer",
                        "input",
                        "storage",
                        "malformed",
                        "short",
                        "long",
                        "nan",
                        "infinity",
                    ] {
                        let mut refused = query_admission_embedder(tier);
                        let mut identity = refused.identity.clone();
                        let mut values = refused.vector.clone();
                        match fault {
                            "space" => {
                                identity = in_memory_identity(
                                    "foreign-response",
                                    identity.space.dimension,
                                );
                            }
                            "producer" => {
                                identity.producer.backend = "foreign-backend".to_owned();
                            }
                            "input" => {
                                identity.input.query_instruction = "foreign-instruction".to_owned();
                                identity.space.input_contract_fingerprint =
                                    identity.input.fingerprint();
                                identity.producer.space_fingerprint = identity.space.fingerprint();
                            }
                            "storage" => {
                                identity.storage.quantization = QuantizationFormat::F16;
                            }
                            "malformed" => identity.space.immutable_revision.clear(),
                            "short" => {
                                values.pop();
                            }
                            "long" => values.push(0.0),
                            "nan" => values[0] = f32::NAN,
                            "infinity" => values[0] = f32::INFINITY,
                            _ => unreachable!(),
                        }
                        refused.bound_response_identity = Some(identity);
                        refused.bound_response_values = Some(values);
                        let refused = Arc::new(refused);
                        let provider: Arc<dyn Embedder> = if cached {
                            let cache = Arc::new(CachedEmbedder::new(refused.clone(), 8));
                            cache.embed(&cx, QUERY).await.unwrap();
                            cache.embed(&cx, QUERY).await.unwrap();
                            cache
                        } else {
                            refused.clone()
                        };
                        let (fast, quality): (Arc<dyn Embedder>, Arc<dyn Embedder>) =
                            if tier == "fast" {
                                (provider, Arc::new(query_admission_embedder("quality")))
                            } else {
                                (Arc::new(query_admission_embedder("fast")), provider)
                            };
                        let searcher = TwoTierSearcher::new(
                            Arc::clone(&index),
                            fast,
                            TwoTierConfig::default(),
                        )
                        .with_quality_embedder(quality)
                        .with_lexical(Arc::new(StubLexical))
                        .with_prf_config(PrfConfig {
                            enabled: true,
                            alpha: 0.5,
                            top_k_feedback: 1,
                            min_feedback_docs: 1,
                            score_weighted: false,
                        });
                        let text_reads = AtomicU64::new(0);
                        let mut phases = Vec::new();
                        let mut initial = None;
                        let mut refusal = None;
                        let outcome = searcher.search(
                            &cx,
                            "find the quality document beyond the fast pool -excluded",
                            3,
                            |_| {
                                text_reads.fetch_add(1, Ordering::Relaxed);
                                Some("retained text".to_owned())
                            },
                            |phase| match phase {
                                SearchPhase::Initial { results, .. } => {
                                    initial = Some(serde_json::to_value(results).unwrap());
                                    phases.push("initial");
                                }
                                SearchPhase::RefinementFailed { initial_results, error, .. } => {
                                    assert_eq!(Some(serde_json::to_value(initial_results).unwrap()), initial);
                                    refusal = Some(error);
                                    phases.push("refinement_failed");
                                }
                                phase => panic!("{layout}/{tier}/{fault}: invalid response published {phase:?}"),
                            },
                        ).await;
                        let error = if tier == "fast" {
                            assert!(
                                phases.is_empty(),
                                "foreign fast query cannot publish Initial"
                            );
                            assert_eq!(text_reads.load(Ordering::Relaxed), 0);
                            outcome.expect_err("invalid fast response is terminal")
                        } else {
                            let metrics =
                                outcome.expect("retain Initial on refused quality response");
                            assert_eq!(phases, ["initial", "refinement_failed"]);
                            assert_eq!(metrics.phase2_vectors_searched, 0);
                            assert_eq!(metrics.quality_search_ms.to_bits(), 0.0_f64.to_bits());
                            refusal.expect("quality response refusal")
                        };
                        assert!(
                            matches!(
                                error,
                                SearchError::InvalidConfig { .. }
                                    | SearchError::DimensionMismatch { .. }
                                    | SearchError::UnverifiableRemoteSpace { .. }
                            ),
                            "{layout}/{tier}/{fault}: {error}"
                        );
                        assert_eq!(refused.bound_embeds.load(Ordering::Relaxed), 1);
                        assert_eq!(refused.embed_count(), 1 + u64::from(cached));
                    }
                }
            }
        }
    });
}

#[test]
fn query_producer_drift_during_suspension_refuses_late_success_and_failure() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for layout in ["stamped-v1", "exact-v2", "native-v2"] {
            let index = query_admission_index(layout);
            for tier in ["fast", "quality"] {
                for cached in [false, true] {
                    for provider_fails in [false, true] {
                        let mut drifting =
                            query_admission_embedder(tier).with_async_bound_response();
                        let mut replacement = drifting.identity.clone();
                        replacement.producer.backend = "replacement-backend".to_owned();
                        // Even an apparently matching response is stale when
                        // the producer has changed since request admission.
                        drifting.bound_response_identity = Some(drifting.identity.clone());
                        drifting.replacement_identity = Some(replacement);
                        if provider_fails {
                            drifting.bound_action = Some(BoundResponseAction::Fail);
                        }
                        let drifting = Arc::new(drifting);
                        let provider: Arc<dyn Embedder> = if cached {
                            Arc::new(CachedEmbedder::new(drifting.clone(), 8))
                        } else {
                            drifting.clone()
                        };
                        let (fast, quality): (Arc<dyn Embedder>, Arc<dyn Embedder>) =
                            if tier == "fast" {
                                (provider, Arc::new(query_admission_embedder("quality")))
                            } else {
                                (Arc::new(query_admission_embedder("fast")), provider)
                            };
                        let searcher = TwoTierSearcher::new(
                            Arc::clone(&index),
                            fast,
                            TwoTierConfig::default(),
                        )
                        .with_quality_embedder(quality)
                        .with_lexical(Arc::new(StubLexical));
                        let mut phases = Vec::new();
                        let mut initial = None;
                        let mut refusal = None;
                        let outcome = searcher
                            .search(
                                &cx,
                                "find the quality document beyond the fast pool",
                                3,
                                |_| None,
                                |phase| match phase {
                                    SearchPhase::Initial { results, .. } => {
                                        initial = Some(serde_json::to_value(results).unwrap());
                                        phases.push("initial");
                                    }
                                    SearchPhase::RefinementFailed {
                                        initial_results,
                                        error,
                                        ..
                                    } => {
                                        assert_eq!(
                                            Some(serde_json::to_value(initial_results).unwrap()),
                                            initial
                                        );
                                        refusal = Some(error);
                                        phases.push("refinement_failed");
                                    }
                                    phase => {
                                        panic!("{layout}/{tier}: drift published {phase:?}")
                                    }
                                },
                            )
                            .await;
                        let error = if tier == "fast" {
                            assert!(phases.is_empty());
                            outcome.expect_err("fast producer drift is terminal")
                        } else {
                            let metrics = outcome.unwrap();
                            assert_eq!(phases, ["initial", "refinement_failed"]);
                            assert_eq!(metrics.phase2_vectors_searched, 0);
                            refusal.unwrap()
                        };
                        assert!(
                            matches!(error, SearchError::UnverifiableRemoteSpace { .. }),
                            "{error}"
                        );
                        assert_eq!(drifting.bound_embeds.load(Ordering::Relaxed), 1);
                    }
                }
            }
        }
    });
}

#[test]
fn captured_quality_identity_cannot_change_in_the_initial_callback() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = query_admission_index("exact-v2");
        let fast = Arc::new(query_admission_embedder("fast"));
        let mut quality = query_admission_embedder("quality");
        let mut replacement = quality.identity.clone();
        replacement.producer.backend = "changed-after-initial".to_owned();
        quality.replacement_identity = Some(replacement);
        let quality = Arc::new(quality);
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality.clone());
        let mut phases = Vec::new();
        let mut initial = None;
        searcher
            .search(
                &cx,
                "query",
                3,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { results, .. } => {
                        initial = Some(serde_json::to_value(results).unwrap());
                        quality.identity_changed.store(true, Ordering::Relaxed);
                        phases.push("initial");
                    }
                    SearchPhase::RefinementFailed {
                        initial_results,
                        error,
                        ..
                    } => {
                        assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                        assert_eq!(
                            Some(serde_json::to_value(initial_results).unwrap()),
                            initial
                        );
                        phases.push("refinement_failed");
                    }
                    phase => panic!("changed quality provider published {phase:?}"),
                },
            )
            .await
            .unwrap();
        assert_eq!(phases, ["initial", "refinement_failed"]);
        assert_eq!(quality.bound_embeds.load(Ordering::Relaxed), 0);
        assert_eq!(quality.embed_count(), 0);
    });
}

#[test]
fn query_cancellation_after_bound_suspension_dominates_every_provider_outcome() {
    for layout in ["stamped-v1", "exact-v2", "native-v2"] {
        for tier in ["fast", "quality"] {
            for cached in [false, true] {
                for action in [
                    BoundResponseAction::Cancel,
                    BoundResponseAction::CancelAndFail,
                    BoundResponseAction::CancelAndReturnTypedError,
                ] {
                    asupersync::test_utils::run_test_with_cx(|cx| async move {
                        let index = query_admission_index(layout);
                        let mut cancelled =
                            query_admission_embedder(tier).with_async_bound_response();
                        cancelled.bound_action = Some(action);
                        let cancelled = Arc::new(cancelled);
                        let provider: Arc<dyn Embedder> = if cached {
                            Arc::new(CachedEmbedder::new(cancelled.clone(), 8))
                        } else {
                            cancelled.clone()
                        };
                        let (fast, quality): (Arc<dyn Embedder>, Arc<dyn Embedder>) =
                            if tier == "fast" {
                                (provider, Arc::new(query_admission_embedder("quality")))
                            } else {
                                (Arc::new(query_admission_embedder("fast")), provider)
                            };
                        let adapter =
                            Arc::new(RecordingHostAdapter::new("query_bound_cancellation"));
                        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                            .with_quality_embedder(quality)
                            .with_lexical(Arc::new(StubLexical))
                            .with_host_adapter(adapter.clone());
                        let text_reads = AtomicU64::new(0);
                        let mut phases = Vec::new();
                        let error = searcher
                            .search(
                                &cx,
                                "find the quality document beyond the fast pool -excluded",
                                3,
                                |_| {
                                    text_reads.fetch_add(1, Ordering::Relaxed);
                                    Some("retained text".to_owned())
                                },
                                |phase| match phase {
                                    SearchPhase::Initial { .. } => phases.push("initial"),
                                    phase => {
                                        panic!("cancelled {tier} response published {phase:?}")
                                    }
                                },
                            )
                            .await
                            .expect_err("provider cancellation must remain terminal");
                        let expected_phase = if tier == "fast" {
                            "fast_embed_to_activation"
                        } else {
                            "quality_embed_to_prf"
                        };
                        assert!(
                            matches!(error, SearchError::Cancelled { ref phase, .. } if phase == expected_phase),
                            "{error}"
                        );
                        if tier == "fast" {
                            assert!(phases.is_empty());
                            assert_eq!(text_reads.load(Ordering::Relaxed), 0);
                        } else {
                            assert_eq!(phases, ["initial"]);
                        }
                        assert_eq!(cancelled.bound_embeds.load(Ordering::Relaxed), 1);
                        assert_single_cancelled_session_stop(&adapter, expected_phase);
                    });
                }
            }
        }
    }
}

#[test]
fn query_output_contracts_are_admitted_before_any_tier_or_cache_runs() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in ["fast", "quality"] {
            for fault in [
                "dimension",
                "quantization",
                "format",
                "endianness",
                "malformed",
                "refused",
            ] {
                let index = query_admission_index("exact-v2");
                let mut refused = query_admission_embedder(tier);
                match fault {
                    "dimension" => refused.reported_dimension = Some(refused.vector.len() + 1),
                    "quantization" => {
                        refused.identity.storage.quantization = QuantizationFormat::F16;
                    }
                    "format" => refused.identity.storage.format = "fsvi-v2".to_owned(),
                    "endianness" => {
                        refused.identity.storage.endianness = "little-endian".to_owned();
                    }
                    "malformed" => refused.identity.input.canonicalization.clear(),
                    "refused" => refused.identity_refusal = Some("provider.identity"),
                    _ => unreachable!(),
                }
                let refused = Arc::new(refused);
                let other = Arc::new(query_admission_embedder(if tier == "fast" {
                    "quality"
                } else {
                    "fast"
                }));
                let (fast, quality): (Arc<dyn Embedder>, Arc<dyn Embedder>) = if tier == "fast" {
                    (refused.clone(), other.clone())
                } else {
                    (other.clone(), refused.clone())
                };
                let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                    .with_quality_embedder(quality)
                    .with_embedding_cache(8)
                    .with_lexical(Arc::new(StubLexical));
                let error = searcher
                    .search(
                        &cx,
                        "query",
                        3,
                        |_| None,
                        |phase| panic!("invalid contract published {phase:?}"),
                    )
                    .await
                    .expect_err("the request must validate both contracts before inference");
                assert!(
                    matches!(error, SearchError::InvalidConfig { .. }),
                    "{tier}/{fault}: {error}"
                );
                for provider in [&refused, &other] {
                    assert_eq!(provider.bound_embeds.load(Ordering::Relaxed), 0);
                    assert_eq!(provider.embed_count(), 0);
                }
            }
        }
    });
}

#[test]
fn only_the_explicit_default_missing_identity_can_enter_the_legacy_query_lane() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for field in ["embedder.identity", "provider.identity"] {
            let mut provider = query_admission_embedder("fast");
            provider.identity_refusal = Some(field);
            let provider = Arc::new(provider);
            let searcher = TwoTierSearcher::new(
                build_test_index(4),
                provider.clone(),
                TwoTierConfig::default(),
            );
            let outcome = searcher.search_collect(&cx, "query", 3).await;
            if field == "embedder.identity" {
                assert!(!outcome.unwrap().0.is_empty());
                assert_eq!(provider.embed_count(), 1);
            } else {
                assert!(
                    matches!(outcome, Err(SearchError::InvalidConfig { field, .. }) if field == "provider.identity")
                );
                assert_eq!(provider.embed_count(), 0);
            }
            assert_eq!(provider.bound_embeds.load(Ordering::Relaxed), 0);
        }
    });
}
