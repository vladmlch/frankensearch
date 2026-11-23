use super::*;

#[cfg(feature = "rerank")]
fn test_scored_result(doc_id: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.to_owned().into(),
        score,
        source: ScoreSource::SemanticFast,
        index: None,
        fast_score: Some(score),
        quality_score: None,
        lexical_score: None,
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

#[cfg(feature = "rerank")]
#[test]
fn mmr_partial_quality_pool_falls_back_entirely_to_fast_space() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("mmr-partial-quality");
        let fast_binding = artifact_binding("mmr-fast", 4, 41);
        let quality_binding = artifact_binding("mmr-quality", 4, 41);
        let index = owner_backed_partial_quality_index(&dir, &fast_binding, &quality_binding);
        let fast = Arc::new(IdentityCountingEmbedder::new(
            "fast",
            in_memory_identity("mmr-fast", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let searcher =
            TwoTierSearcher::new(index, fast as Arc<dyn Embedder>, TwoTierConfig::default());
        let results = [
            test_scored_result("doc-shared", 1.0),
            test_scored_result("doc-fast-only", 0.5),
        ];

        let embeddings = searcher
            .mmr_embeddings_in_one_space(&cx, &results)
            .expect("load one-space MMR pool")
            .expect("the complete fast tier covers this pool");

        assert_eq!(embeddings[0], vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(embeddings[1], vec![0.0, 1.0, 0.0, 0.0]);
        assert_ne!(
            embeddings[0],
            vec![0.0, 1.0, 0.0, 0.0],
            "the shared document's quality vector must not be mixed with a fast fallback"
        );
    });
}

#[cfg(feature = "rerank")]
#[test]
fn mmr_skips_pool_with_no_tier_covering_every_candidate() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("mmr-no-common-space");
        let fast_binding = artifact_binding("mmr-fast", 4, 43);
        let quality_binding = artifact_binding("mmr-quality", 4, 43);
        let index = owner_backed_partial_quality_index(&dir, &fast_binding, &quality_binding);
        let fast = Arc::new(IdentityCountingEmbedder::new(
            "fast",
            in_memory_identity("mmr-fast", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let searcher =
            TwoTierSearcher::new(index, fast as Arc<dyn Embedder>, TwoTierConfig::default());
        let results = [
            test_scored_result("doc-fast-only", 1.0),
            test_scored_result("doc-quality-query", 0.5),
        ];

        assert!(
            searcher
                .mmr_embeddings_in_one_space(&cx, &results)
                .expect("lookups remain valid")
                .is_none(),
            "MMR must skip when no one vector space covers the whole pool"
        );
    });
}

#[test]
fn prf_never_falls_back_to_fast_vectors_for_quality_query_expansion() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("prf-quality-only-feedback");
        let fast_binding = artifact_binding("prf-fast", 4, 47);
        let quality_binding = artifact_binding("prf-quality", 4, 47);
        let index = owner_backed_partial_quality_index(&dir, &fast_binding, &quality_binding);
        let fast = Arc::new(IdentityCountingEmbedder::new(
            "fast",
            in_memory_identity("prf-fast", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let quality = Arc::new(IdentityCountingEmbedder::new(
            "quality",
            in_memory_identity("prf-quality", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let searcher =
            TwoTierSearcher::new(index, fast as Arc<dyn Embedder>, TwoTierConfig::default())
                .with_quality_embedder(quality as Arc<dyn Embedder>)
                .with_prf_config(PrfConfig {
                    enabled: true,
                    alpha: 0.5,
                    top_k_feedback: 2,
                    min_feedback_docs: 2,
                    score_weighted: false,
                });

        let mut initial = Vec::new();
        let mut refined = Vec::new();
        let mut refinement_error = None;
        searcher
            .search(
                &cx,
                "natural language query with enough words",
                4,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { results, .. } => {
                        initial = results
                            .iter()
                            .map(|result| result.doc_id.to_string())
                            .collect();
                    }
                    SearchPhase::Refined { results, .. }
                    | SearchPhase::Reranked { results, .. } => {
                        refined = results
                            .iter()
                            .map(|result| result.doc_id.to_string())
                            .collect();
                    }
                    SearchPhase::RefinementFailed { error, .. } => {
                        refinement_error = Some(error.to_string());
                    }
                },
            )
            .await
            .expect("run partial-quality PRF search");

        assert_eq!(refinement_error, None, "refinement must not fail");
        assert_eq!(
            initial.len(),
            2,
            "both fast documents feed the PRF candidate loop"
        );
        assert!(initial.contains(&"doc-fast-only".to_owned()));
        let query_rank = refined
            .iter()
            .position(|doc_id| doc_id == "doc-quality-query")
            .expect("quality-query document remains in refined pool");
        let wrong_rank = refined
            .iter()
            .position(|doc_id| doc_id == "doc-quality-wrong")
            .expect("quality-wrong document remains in refined pool");
        assert!(
            query_rank < wrong_rank,
            "a fast-tier fallback would rotate the quality query toward doc-quality-wrong: {refined:?}"
        );
    });
}

/// bd-ctzo C2, on the REAL async orchestration: the refined phase must
/// RETRIEVE from the quality tier, not rescore the fast pool.
///
/// `doc-quality-only` exists in the quality tier and NOWHERE in the fast
/// tier, so no ranking, re-ranking, promotion or blend over a
/// fast-selected pool can ever produce it. Its presence in the refined
/// results is therefore a direct observation of independent retrieval, and
/// its absence is the exact failure mode
/// `quality_scores_for_hits(&quality_vec, &fast_hits)` had in production.
/// The fast-pool assertion below keeps the fixture honest: if a later
/// change put the document into the fast tier, this test would start
/// passing for the wrong reason, and that assertion fails first.
#[test]
fn refined_phase_retrieves_from_the_quality_tier_not_the_fast_pool() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("quality-retrieval");
        let fast_binding = artifact_binding("ctzo-refine-fast", 4, 31);
        let quality_binding = artifact_binding("ctzo-refine-quality", 4, 31);
        let index = owner_backed_two_tier_index(&dir, &fast_binding, &quality_binding);

        let fast = Arc::new(IdentityCountingEmbedder::new(
            "fast",
            in_memory_identity("ctzo-refine-fast", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        let quality = Arc::new(IdentityCountingEmbedder::new(
            "quality",
            in_memory_identity("ctzo-refine-quality", 4),
            vec![0.0, 1.0, 0.0, 0.0],
        ));
        let searcher =
            TwoTierSearcher::new(index, fast as Arc<dyn Embedder>, TwoTierConfig::default())
                .with_quality_embedder(quality as Arc<dyn Embedder>);

        let mut initial: Vec<String> = Vec::new();
        let mut refined: Vec<String> = Vec::new();
        let mut quality_only_index: Option<Option<u32>> = None;
        let metrics = searcher
            .search(
                &cx,
                "query",
                3,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { results, .. } => {
                        initial = results.iter().map(|r| r.doc_id.to_string()).collect();
                    }
                    SearchPhase::Refined { results, .. }
                    | SearchPhase::Reranked { results, .. } => {
                        refined = results.iter().map(|r| r.doc_id.to_string()).collect();
                        quality_only_index = results
                            .iter()
                            .find(|r| r.doc_id == "doc-quality-only")
                            .map(|r| r.index);
                    }
                    SearchPhase::RefinementFailed { error, .. } => {
                        panic!("refinement must not fail: {error}")
                    }
                },
            )
            .await
            .expect("both tiers are admitted and their embedders join");

        assert!(
            !initial.contains(&"doc-quality-only".to_owned()),
            "the fixture is only meaningful while the fast tier cannot reach \
             doc-quality-only, got {initial:?}"
        );
        assert!(
            refined.contains(&"doc-quality-only".to_owned()),
            "the refined phase must reach a document the fast tier does not contain; \
             got {refined:?}"
        );
        // bd-ctzo C4: the coverage receipt reaches the caller, and every
        // number in it comes from the owner witness or the returned
        // candidates.
        let coverage = metrics.coverage.as_ref().expect("owner-backed coverage");
        match &coverage.quality {
            frankensearch_core::TierQueryCoverageV1::Witnessed {
                generation_sequence,
                live_count,
                contributed_candidates,
            } => {
                assert_eq!(*generation_sequence, 31, "read from the owner's witness");
                assert_eq!(*live_count, 3, "the quality tier holds three live docs");
                assert!(
                    *contributed_candidates > 0,
                    "the quality tier served candidates, so it must be credited with them"
                );
            }
            other => panic!("quality tier must be witnessed, got {other:?}"),
        }
        assert_eq!(
            coverage.fast,
            frankensearch_core::TierQueryCoverageV1::NotRequested,
            "phase 2 activates the quality arm only; the fast pool came from phase 1"
        );
        assert!(
            !coverage.redacted_summary().contains("doc-"),
            "a coverage log line must never carry a document id: {}",
            coverage.redacted_summary()
        );
        // A document with no fast row must not carry a QUALITY row ordinal
        // in the field everything downstream reads as a fast-tier index.
        assert_eq!(
            quality_only_index,
            Some(None),
            "a quality-only document has no fast-tier row and must say so"
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}

#[test]
fn native_fast_and_both_tiers_preserve_progressive_hybrid_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for with_quality in [false, true] {
            let dir = owner_backed_dir("native-fast-progressive")
                .canonicalize()
                .expect("native persistence requires a canonical fixture root");
            let fast_binding = artifact_binding("native-progressive-fast", 4, 67);
            let quality_binding = artifact_binding("native-progressive-quality", 4, 67);
            let mut index = if with_quality {
                owner_backed_two_tier_index(&dir, &fast_binding, &quality_binding)
            } else {
                let path = dir.join("fast.fsvi");
                let mut writer =
                    frankensearch_index::VectorIndex::create_v2(&path, fast_binding.clone())
                        .unwrap();
                writer.write_record("doc-a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
                writer.write_record("doc-b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
                writer
                    .write_tombstone_record("doc-dead", &[1.0, 0.0, 0.0, 0.0])
                    .unwrap();
                writer.finish().unwrap();
                Arc::new(
                    TwoTierIndex::open_admitted_v2_with_paths(
                        &frankensearch_index::TwoTierIndexPaths::new(path),
                        TwoTierConfig::default(),
                        &fast_binding,
                        None,
                    )
                    .unwrap(),
                )
            };
            let fast_graph_path = dir.join("fast.fshnsw");
            let quality_graph_path = dir.join("quality.fshnsw");
            let params = frankensearch_index::native_hnsw::HnswParams::default();
            let mut exact = None;
            for mode in [
                "exact",
                "native-fast",
                "native-both",
                "native-async",
                "reopened",
                "refused-reload",
            ] {
                let native = mode != "exact";
                let native_quality = !matches!(mode, "exact" | "native-fast");
                let asynchronous = matches!(mode, "native-async" | "refused-reload");
                if native && !index.has_native_fast_hnsw() {
                    let index = Arc::get_mut(&mut index).expect("previous searcher dropped");
                    index.enable_native_fast_hnsw(params, 17).unwrap();
                }
                if native_quality && with_quality && !index.has_native_quality_hnsw() {
                    Arc::get_mut(&mut index)
                        .expect("previous searcher dropped")
                        .enable_native_quality_hnsw(params, 19)
                        .unwrap();
                }
                if mode == "reopened" {
                    let receipt = index
                        .save_native_fast_hnsw(&fast_graph_path)
                        .expect("persist the fast graph with its owner receipt");
                    assert_eq!(receipt.artifact_generation.sequence, 67);
                    if with_quality {
                        let receipt = index
                            .save_native_quality_hnsw(&quality_graph_path)
                            .expect("persist the quality graph with its owner receipt");
                        assert_eq!(receipt.artifact_generation.sequence, 67);
                    }
                    // Release the original owners and topology. The next
                    // invocation must use newly admitted FSVI bytes and
                    // graphs loaded from the persisted artifacts.
                    drop(index);
                    let paths = if with_quality {
                        frankensearch_index::TwoTierIndexPaths::new(dir.join("vector.fast.idx"))
                            .with_quality_index(dir.join("vector.quality.idx"))
                    } else {
                        frankensearch_index::TwoTierIndexPaths::new(dir.join("fast.fsvi"))
                    };
                    let mut reopened = TwoTierIndex::open_admitted_v2_with_paths(
                        &paths,
                        TwoTierConfig::default(),
                        &fast_binding,
                        with_quality.then_some(&quality_binding),
                    )
                    .expect("reopen the persisted owners");
                    assert!(!reopened.has_native_fast_hnsw());
                    assert!(!reopened.has_native_quality_hnsw());
                    reopened
                        .load_native_fast_hnsw(&fast_graph_path, params, 17)
                        .expect("load the persisted fast graph");
                    if with_quality {
                        reopened
                            .load_native_quality_hnsw(&quality_graph_path, params, 19)
                            .expect("load the persisted quality graph");
                    }
                    index = Arc::new(reopened);
                }
                if mode == "refused-reload" {
                    let current = Arc::get_mut(&mut index).expect("previous searcher dropped");
                    current
                        .load_native_fast_hnsw(&fast_graph_path, params, 18)
                        .expect_err("a foreign seed cannot replace the loaded fast policy");
                    if with_quality {
                        // Only the artifact generation differs in these
                        // stale fixtures: models, vectors, parameters, and
                        // construction seeds match the serving generation.
                        let stale_dir = owner_backed_dir("native-progressive-stale")
                            .canonicalize()
                            .expect("native persistence requires a canonical fixture root");
                        let mut stale = owner_backed_two_tier_index(
                            &stale_dir,
                            &artifact_binding("native-progressive-fast", 4, 66),
                            &artifact_binding("native-progressive-quality", 4, 66),
                        );
                        let stale = Arc::get_mut(&mut stale).unwrap();
                        stale.enable_native_fast_hnsw(params, 17).unwrap();
                        stale.enable_native_quality_hnsw(params, 19).unwrap();
                        let stale_fast_path = stale_dir.join("fast.fshnsw");
                        let stale_quality_path = stale_dir.join("quality.fshnsw");
                        stale.save_native_fast_hnsw(&stale_fast_path).unwrap();
                        stale.save_native_quality_hnsw(&stale_quality_path).unwrap();
                        for (path, seed) in [(&quality_graph_path, 19), (&stale_fast_path, 17)] {
                            current
                                .load_native_fast_hnsw(path, params, seed)
                                .expect_err("foreign tier or generation cannot replace fast");
                        }
                        for (path, seed) in [(&fast_graph_path, 17), (&stale_quality_path, 19)] {
                            current
                                .load_native_quality_hnsw(path, params, seed)
                                .expect_err("foreign tier or generation cannot replace quality");
                        }
                    }
                }
                assert_eq!(index.has_native_fast_hnsw(), native, "{mode}");
                assert_eq!(
                    index.has_native_quality_hnsw(),
                    native_quality && with_quality,
                    "{mode}"
                );
                let identity = in_memory_identity("native-progressive-fast", 4);
                let mut fast = IdentityCountingEmbedder::new(
                    "fast",
                    identity.clone(),
                    vec![1.0, 0.0, 0.0, 0.0],
                )
                .with_bound_response_identity(identity);
                if asynchronous {
                    fast = fast.with_async_bound_response();
                }
                let fast = Arc::new(fast);
                let mut searcher = TwoTierSearcher::new(
                    Arc::clone(&index),
                    fast.clone(),
                    TwoTierConfig::default(),
                )
                .with_lexical(Arc::new(StubLexical))
                .with_embedding_cache(8);
                let quality = Arc::new(IdentityCountingEmbedder::new(
                    "quality",
                    in_memory_identity("native-progressive-quality", 4),
                    vec![0.0, 1.0, 0.0, 0.0],
                ));
                if with_quality {
                    searcher = searcher.with_quality_embedder(quality.clone());
                }
                let mut phases = Vec::new();
                let mut snapshots = Vec::new();
                let metrics = searcher
                    .search(
                        &cx,
                        "find the best matching document -excluded",
                        8,
                        |doc| {
                            Some(if matches!(doc, "doc-b" | "doc-far" | "lex-doc-1") {
                                "excluded document".to_owned()
                            } else {
                                "retained document".to_owned()
                            })
                        },
                        |phase| {
                            let (name, results) = match phase {
                                SearchPhase::Initial { results, .. } => ("initial", results),
                                SearchPhase::Refined { results, .. } => ("refined", results),
                                phase => panic!("unexpected native phase: {phase:?}"),
                            };
                            assert!(results.iter().all(|hit| !matches!(
                                hit.doc_id.as_str(),
                                "doc-dead" | "doc-b" | "doc-far" | "lex-doc-1"
                            )));
                            assert!(results.iter().any(|hit| hit.doc_id == "lex-doc-0"));
                            if with_quality {
                                let quality_only =
                                    results.iter().find(|hit| hit.doc_id == "doc-quality-only");
                                if name == "initial" {
                                    assert!(quality_only.is_none());
                                } else {
                                    let hit = quality_only
                                        .expect("quality retrieval reaches beyond the fast pool");
                                    assert_eq!(hit.index, None);
                                    assert!(hit.quality_score.is_some());
                                }
                            }
                            phases.push(name);
                            snapshots.push(serde_json::to_value(results).unwrap());
                        },
                    )
                    .await
                    .expect("serve the admitted native tiers");
                assert_eq!(
                    phases,
                    if with_quality {
                        vec!["initial", "refined"]
                    } else {
                        vec!["initial"]
                    }
                );
                assert_eq!(fast.bound_embeds.load(Ordering::Relaxed), 1);
                assert_eq!(
                    quality.bound_embeds.load(Ordering::Relaxed),
                    u64::from(with_quality),
                    "{mode} must bind every identified quality response"
                );
                assert_eq!(fast.embed_count(), 1);
                assert!(fast.identity_count() >= 3);
                if let Some(expected) = &exact {
                    assert_eq!(
                        &snapshots, expected,
                        "{mode}: native and exact public results agree"
                    );
                } else {
                    exact = Some(snapshots);
                }
                if native {
                    let coverage = metrics.coverage.expect("native fast coverage");
                    assert_eq!(
                        coverage.fast,
                        frankensearch_core::TierQueryCoverageV1::Witnessed {
                            generation_sequence: 67,
                            live_count: 2,
                            contributed_candidates: 1,
                        }
                    );
                    assert_eq!(
                        coverage.quality,
                        if with_quality {
                            frankensearch_core::TierQueryCoverageV1::Witnessed {
                                generation_sequence: 67,
                                live_count: 3,
                                contributed_candidates: 2,
                            }
                        } else {
                            frankensearch_core::TierQueryCoverageV1::NotRequested
                        }
                    );
                    // HashControl for both shapes: `artifact_binding` and
                    // `in_memory_identity` build `explicit_test_model`
                    // control spaces, which never report a semantic
                    // topology (bd-ctzo C4). The quality coverage above is
                    // what distinguishes progressive from fast-only.
                    assert_eq!(
                        coverage.topology,
                        frankensearch_core::RetrievalTopology::HashControl
                    );
                }
            }
        }
    });
}

#[test]
fn native_fast_refuses_foreign_bound_responses_before_lexical_filtering() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("native-fast-foreign-response");
        let binding = artifact_binding("native-fast-response", 4, 68);
        let mut index = owner_backed_index(&dir, &binding);
        Arc::get_mut(&mut index)
            .unwrap()
            .enable_native_fast_hnsw(frankensearch_index::native_hnsw::HnswParams::default(), 23)
            .unwrap();
        let advertised = in_memory_identity("native-fast-response", 4);
        let mut foreign_producer = advertised.clone();
        "foreign-bound-backend".clone_into(&mut foreign_producer.producer.backend);
        for (response, field) in [
            (
                in_memory_identity("foreign-fast-space", 4),
                "query_embedding.fast.space_identity",
            ),
            (
                foreign_producer,
                "search_activation.fast.producer_conformance",
            ),
        ] {
            for cached in [false, true] {
                for exact_override in [false, true] {
                    let fast = Arc::new(
                        IdentityCountingEmbedder::new(
                            "fast",
                            advertised.clone(),
                            vec![1.0, 0.0, 0.0, 0.0],
                        )
                        .with_bound_response_identity(response.clone()),
                    );
                    let mut searcher = TwoTierSearcher::new(
                        Arc::clone(&index),
                        fast.clone(),
                        TwoTierConfig::default(),
                    )
                    .with_lexical(Arc::new(StubLexical));
                    if cached {
                        searcher = searcher.with_embedding_cache(8);
                    }
                    if exact_override {
                        searcher = searcher.with_search_params(SearchParams {
                            parallel_enabled: false,
                            parallel_threshold: usize::MAX,
                            parallel_chunk_size: 1,
                        });
                    }
                    let text_reads = AtomicU64::new(0);
                    let error = searcher
                        .search(
                            &cx,
                            "find a matching document -excluded",
                            3,
                            |_| {
                                text_reads.fetch_add(1, Ordering::Relaxed);
                                Some("retained document".to_owned())
                            },
                            |phase| panic!("foreign fast response published {phase:?}"),
                        )
                        .await
                        .expect_err("the actual response must be admitted");
                    assert!(
                        matches!(&error, SearchError::InvalidConfig { field: actual, .. } if actual == field)
                            || (cached
                                && matches!(error, SearchError::UnverifiableRemoteSpace { .. }))
                    );
                    assert_eq!(text_reads.load(Ordering::Relaxed), 0);
                    assert_eq!(fast.bound_embeds.load(Ordering::Relaxed), 1);
                    assert_eq!(fast.embed_count(), 1);
                    assert!(fast.identity_count() >= 3);
                }
            }
        }
    });
}

#[test]
fn native_fast_cancellation_dominates_refusal_and_lexical_fallback() {
    for asynchronous in [false, true] {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = owner_backed_dir("native-fast-cancel");
            let binding = artifact_binding("native-fast-cancel", 4, 69);
            let mut index = owner_backed_index(&dir, &binding);
            Arc::get_mut(&mut index)
                .unwrap()
                .enable_native_fast_hnsw(
                    frankensearch_index::native_hnsw::HnswParams::default(),
                    29,
                )
                .unwrap();
            let mut fast = IdentityCountingEmbedder::new(
                "fast",
                in_memory_identity("native-fast-cancel", 4),
                vec![1.0, 0.0, 0.0, 0.0],
            )
            .with_bound_response_values(vec![1.0, 0.0, 0.0]);
            fast.bound_action = Some(BoundResponseAction::Cancel);
            if asynchronous {
                fast = fast.with_async_bound_response();
            }
            let fast = Arc::new(fast);
            let adapter = Arc::new(RecordingHostAdapter::new("native_fast_cancellation"));
            let searcher = TwoTierSearcher::new(index, fast.clone(), TwoTierConfig::default())
                .with_lexical(Arc::new(StubLexical))
                .with_embedding_cache(8)
                .with_host_adapter(adapter.clone());
            let text_reads = AtomicU64::new(0);
            let error = searcher
                .search(
                    &cx,
                    "find a matching document -excluded",
                    3,
                    |_| {
                        text_reads.fetch_add(1, Ordering::Relaxed);
                        Some("retained document".to_owned())
                    },
                    |phase| panic!("cancelled fast response published {phase:?}"),
                )
                .await
                .expect_err("native cancellation remains terminal");
            assert!(
                matches!(error, SearchError::Cancelled { phase, .. } if phase == "fast_embed_to_activation")
            );
            assert_eq!(text_reads.load(Ordering::Relaxed), 0);
            assert_eq!(fast.bound_embeds.load(Ordering::Relaxed), 1);
            assert_eq!(fast.embed_count(), 1);
            assert!(fast.identity_count() >= 3);
            assert_single_cancelled_session_stop(&adapter, "fast_embed_to_activation");
        });
    }
}

#[test]
fn native_fast_exact_override_observes_filter_cancellation_before_initial() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("native-fast-filter-cancel");
        let binding = artifact_binding("native-fast-filter-cancel", 4, 71);
        let mut index = owner_backed_index(&dir, &binding);
        Arc::get_mut(&mut index)
            .unwrap()
            .enable_native_fast_hnsw(frankensearch_index::native_hnsw::HnswParams::default(), 31)
            .unwrap();
        let adapter = Arc::new(RecordingHostAdapter::new("native_fast_filter_cancellation"));
        let searcher = TwoTierSearcher::new(
            index,
            Arc::new(IdentityCountingEmbedder::new(
                "fast",
                in_memory_identity("native-fast-filter-cancel", 4),
                vec![1.0, 0.0, 0.0, 0.0],
            )),
            TwoTierConfig::default(),
        )
        .with_search_params(SearchParams {
            parallel_enabled: false,
            parallel_threshold: usize::MAX,
            parallel_chunk_size: 1,
        })
        .with_host_adapter(adapter.clone());
        let text_reads = AtomicU64::new(0);
        let error = searcher
            .search(
                &cx,
                "find a matching document -excluded",
                3,
                |_| {
                    text_reads.fetch_add(1, Ordering::Relaxed);
                    cx.cancel_with(
                        asupersync::CancelKind::User,
                        Some("cancel during exclusion filtering"),
                    );
                    Some("retained document".to_owned())
                },
                |phase| panic!("filter cancellation published {phase:?}"),
            )
            .await
            .expect_err("native exact override must not publish after cancellation");
        assert!(text_reads.load(Ordering::Relaxed) > 0);
        assert!(matches!(error, SearchError::Cancelled { phase, .. }
            if phase == "fast_result_to_publish"));
        assert_single_cancelled_session_stop(&adapter, "fast_result_to_publish");
    });
}

#[test]
fn native_fast_exact_override_and_zero_signal_keep_their_contracts() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("native-fast-exact-override");
        let path = dir.join("fast.fsvi");
        let binding = artifact_binding("native-fast-override", 4, 70);
        let mut ids = ["a", "b", "c", "d"];
        // FSVI record order is (doc_id FNV-1a hash, doc_id); core exports
        // the same FNV-1a the index uses for doc-id hashing.
        ids.sort_by_key(|id| frankensearch_core::filter::fnv1a_hash(id.as_bytes()));
        write_v2_tier(
            &path,
            &binding,
            &[
                (ids[0], &[1.0, 0.0, 0.0, 0.0]),
                (ids[1], &[0.0, 0.0, 1.0, 0.0]),
                (ids[2], &[0.0, -1.0, 0.0, 0.0]),
                (ids[3], &[0.0, 1.0, 0.0, 0.0]),
            ],
        );
        let config = TwoTierConfig {
            candidate_multiplier: 1,
            mrl_search_dims: 1,
            ..TwoTierConfig::default()
        };
        let mut index = TwoTierIndex::open_admitted_v2_with_paths(
            &frankensearch_index::TwoTierIndexPaths::new(path),
            config.clone(),
            &binding,
            None,
        )
        .unwrap();
        index
            .enable_native_fast_hnsw(
                frankensearch_index::native_hnsw::HnswParams {
                    m0: 2,
                    ef_search: 1,
                    ..frankensearch_index::native_hnsw::HnswParams::default()
                },
                3,
            )
            .unwrap();
        let index = Arc::new(index);
        for (exact_override, zero_query) in [(false, false), (true, false), (false, true)] {
            let fast = Arc::new(IdentityCountingEmbedder::new(
                "fast",
                in_memory_identity("native-fast-override", 4),
                if zero_query {
                    vec![0.0; 4]
                } else {
                    vec![0.0, 1.0, 0.0, 0.0]
                },
            ));
            let mut searcher =
                TwoTierSearcher::new(Arc::clone(&index), fast.clone(), config.clone());
            if exact_override {
                searcher = searcher.with_search_params(SearchParams {
                    parallel_enabled: false,
                    parallel_threshold: usize::MAX,
                    parallel_chunk_size: 1,
                });
            }
            let (hits, metrics) = searcher.search_collect(&cx, "query", 1).await.unwrap();
            if zero_query {
                assert!(hits.is_empty());
                assert_eq!(metrics.zero_signal, Some(ZeroSignalReason::ZeroNormQuery));
                assert!(metrics.coverage.is_none());
            } else {
                assert_eq!(hits[0].doc_id, ids[if exact_override { 3 } else { 0 }]);
                assert_eq!(metrics.coverage.is_some(), !exact_override);
            }
            assert_eq!(fast.bound_embeds.load(Ordering::Relaxed), 1);
        }
    });
}

#[test]
fn native_quality_retrieval_preserves_progressive_results_and_bound_identity_through_prf() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("native-quality-retrieval");
        let fast_binding = artifact_binding("native-quality-fast", 4, 53);
        let quality_binding = artifact_binding("native-quality-provider", 4, 53);
        let mut index = owner_backed_two_tier_index(&dir, &fast_binding, &quality_binding);
        let snapshot = |hits: &[ScoredResult]| {
            hits.iter()
                .map(|hit| {
                    (
                        hit.doc_id.to_string(),
                        hit.score.to_bits(),
                        hit.source,
                        hit.index,
                        hit.fast_score.map(f32::to_bits),
                        hit.quality_score.map(f32::to_bits),
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut exact_results = None;
        for native in [false, true] {
            if native {
                Arc::get_mut(&mut index)
                    .expect("previous searcher released its index")
                    .enable_native_quality_hnsw(
                        frankensearch_index::native_hnsw::HnswParams::default(),
                        7,
                    )
                    .expect("enable native quality retrieval");
            }
            assert_eq!(index.has_native_quality_hnsw(), native);
            let fast = Arc::new(IdentityCountingEmbedder::new(
                "fast",
                in_memory_identity("native-quality-fast", 4),
                vec![1.0, 0.0, 0.0, 0.0],
            ));
            let identity = in_memory_identity("native-quality-provider", 4);
            let quality = Arc::new(
                IdentityCountingEmbedder::new(
                    "quality",
                    identity.clone(),
                    vec![0.0, 1.0, 0.0, 0.0],
                )
                .with_bound_response_identity(identity),
            );
            let searcher = TwoTierSearcher::new(Arc::clone(&index), fast, TwoTierConfig::default())
                .with_quality_embedder(quality.clone())
                .with_prf_config(PrfConfig {
                    enabled: true,
                    alpha: 0.5,
                    top_k_feedback: 1,
                    min_feedback_docs: 1,
                    score_weighted: false,
                });

            let mut phases = Vec::new();
            let mut initial = Vec::new();
            let mut refined = Vec::new();
            let metrics = searcher
                .search(
                    &cx,
                    "find the quality document beyond the fast pool",
                    3,
                    |_| None,
                    |phase| match phase {
                        SearchPhase::Initial { results, .. } => {
                            phases.push("initial");
                            initial = results;
                        }
                        SearchPhase::Refined { results, .. } => {
                            phases.push("refined");
                            refined = results;
                        }
                        phase => panic!("unexpected progressive phase: {phase:?}"),
                    },
                )
                .await
                .expect("retrieve from both admitted tiers");
            assert_eq!(phases, ["initial", "refined"]);
            assert!(initial.iter().all(|hit| hit.doc_id != "doc-quality-only"));
            let quality_only = refined
                .iter()
                .find(|hit| hit.doc_id == "doc-quality-only")
                .expect("quality retrieval reaches outside the fast pool");
            assert_eq!(quality_only.index, None);
            let quality_score = quality_only.quality_score.expect("quality evidence");
            assert!(
                quality_score > 0.0 && quality_score < 1.0,
                "feedback must rotate the query away from its original exact match"
            );
            assert_eq!(quality.bound_embeds.load(Ordering::Relaxed), 1);
            assert_eq!(quality.embed_count(), 1);
            assert_eq!(quality.identity_count(), 3);
            let observed = (snapshot(&initial), snapshot(&refined), metrics.coverage);
            if let Some(expected) = &exact_results {
                assert_eq!(
                    &observed, expected,
                    "full-width native retrieval matches exact"
                );
            } else {
                exact_results = Some(observed);
            }
        }
    });
}

#[test]
fn native_quality_refuses_foreign_bound_responses_with_prf_and_embedding_cache() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("native-quality-foreign-response");
        let fast_binding = artifact_binding("native-response-fast", 4, 59);
        let quality_binding = artifact_binding("native-response-quality", 4, 59);
        let mut index = owner_backed_two_tier_index(&dir, &fast_binding, &quality_binding);
        Arc::get_mut(&mut index)
            .expect("the native fixture has one owner")
            .enable_native_quality_hnsw(frankensearch_index::native_hnsw::HnswParams::default(), 11)
            .expect("enable native quality retrieval");
        let advertised = in_memory_identity("native-response-quality", 4);
        let mut foreign_producer = advertised.clone();
        "foreign-bound-backend".clone_into(&mut foreign_producer.producer.backend);
        for (response, expected_field) in [
            (
                in_memory_identity("foreign-bound-space", 4),
                "query_embedding.quality.space_identity",
            ),
            (
                foreign_producer,
                "search_activation.quality.producer_conformance",
            ),
        ] {
            for cached in [false, true] {
                let fast = Arc::new(IdentityCountingEmbedder::new(
                    "fast",
                    in_memory_identity("native-response-fast", 4),
                    vec![1.0, 0.0, 0.0, 0.0],
                ));
                let quality = Arc::new(
                    IdentityCountingEmbedder::new(
                        "quality",
                        advertised.clone(),
                        vec![0.0, 1.0, 0.0, 0.0],
                    )
                    .with_bound_response_identity(response.clone()),
                );
                let mut searcher =
                    TwoTierSearcher::new(Arc::clone(&index), fast, TwoTierConfig::default())
                        .with_quality_embedder(quality.clone())
                        .with_prf_config(PrfConfig {
                            enabled: true,
                            alpha: 0.5,
                            top_k_feedback: 1,
                            min_feedback_docs: 1,
                            score_weighted: false,
                        });
                if cached {
                    searcher = searcher.with_embedding_cache(8);
                }
                let mut phases = Vec::new();
                let mut initial = Vec::new();
                let mut failure = None;
                let metrics = searcher
                    .search(
                        &cx,
                        "find the quality document beyond the fast pool",
                        3,
                        |_| None,
                        |phase| match phase {
                            SearchPhase::Initial { results, .. } => {
                                phases.push("initial");
                                initial = results;
                            }
                            SearchPhase::RefinementFailed {
                                initial_results,
                                error,
                                ..
                            } => {
                                phases.push("refinement_failed");
                                assert_eq!(
                                    serde_json::to_value(initial_results).unwrap(),
                                    serde_json::to_value(&initial).unwrap(),
                                    "a refused response preserves every displayed Initial field"
                                );
                                failure = Some(error);
                            }
                            phase => panic!("foreign quality response escaped: {phase:?}"),
                        },
                    )
                    .await
                    .expect("identity refusal preserves the delivered Initial phase");
                assert_eq!(phases, ["initial", "refinement_failed"]);
                assert!(!initial.is_empty());
                assert!(
                    matches!(&failure, Some(SearchError::InvalidConfig { field, .. })
                    if field == expected_field)
                        || (cached
                            && matches!(
                                failure,
                                Some(SearchError::UnverifiableRemoteSpace { .. })
                            ))
                );
                assert_eq!(quality.bound_embeds.load(Ordering::Relaxed), 1);
                assert_eq!(quality.embed_count(), 1);
                assert!(quality.identity_count() >= 3);
                assert_eq!(metrics.phase2_vectors_searched, 0);
                assert_eq!(metrics.quality_search_ms.to_bits(), 0.0_f64.to_bits());
                assert!(metrics.coverage.is_none());
            }
        }
    });
}

#[test]
fn native_quality_cancellation_dominates_a_refused_bound_response() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = owner_backed_dir("native-quality-bound-cancellation");
        let fast_binding = artifact_binding("native-cancel-fast", 4, 61);
        let quality_binding = artifact_binding("native-cancel-quality", 4, 61);
        let mut index = owner_backed_two_tier_index(&dir, &fast_binding, &quality_binding);
        Arc::get_mut(&mut index)
            .expect("the native fixture has one owner")
            .enable_native_quality_hnsw(frankensearch_index::native_hnsw::HnswParams::default(), 13)
            .expect("enable native quality retrieval");
        let fast = Arc::new(IdentityCountingEmbedder::new(
            "fast",
            in_memory_identity("native-cancel-fast", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        ));
        // Preflight accepts the advertised identity. Binding the actual
        // response cancels this invocation and then rejects its malformed
        // vector length; cancellation must win over that coincident refusal.
        let mut quality = IdentityCountingEmbedder::new(
            "quality",
            in_memory_identity("native-cancel-quality", 4),
            vec![0.0, 1.0, 0.0, 0.0],
        )
        .with_bound_response_values(vec![0.0, 1.0, 0.0]);
        quality.bound_action = Some(BoundResponseAction::Cancel);
        let quality = Arc::new(quality);
        let adapter = Arc::new(RecordingHostAdapter::new(
            "native_quality_bound_cancellation",
        ));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality.clone())
            .with_embedding_cache(8)
            .with_host_adapter(adapter.clone());
        let mut phases = Vec::new();
        let error = searcher
            .search(
                &cx,
                "find the quality document beyond the fast pool",
                3,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { .. } => phases.push("initial"),
                    phase => panic!("cancelled quality response published {phase:?}"),
                },
            )
            .await
            .expect_err("cancellation must not become RefinementFailed");
        assert!(matches!(
            error,
            SearchError::Cancelled { phase, reason }
                if phase == "quality_embed_to_prf"
                    && reason == "user: bound response cancelled"
        ));
        assert_eq!(phases, ["initial"]);
        assert_eq!(quality.bound_embeds.load(Ordering::Relaxed), 1);
        assert_eq!(quality.embed_count(), 1);
        assert!(quality.identity_count() >= 3);
        assert_single_cancelled_session_stop(&adapter, "quality_embed_to_prf");
    });
}
