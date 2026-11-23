use super::*;

/// Build a bound query embedding in the same SPACE as an artifact's
/// identity, so the admission law joins and only the tested component can
/// diverge.
///
/// The query's storage identity is deliberately NOT the artifact's: a
/// query embedding is an in-process `Vec<f32>` and must carry an f32
/// storage identity, while the artifact persists f16. Storage is a
/// separate component from the embedding space, so the space fingerprints
/// still join — which is exactly why activation verifies space and
/// producer, and leaves persisted storage identity to admission, where
/// the artifact's own binding is the authority.
fn bound_query(identity: &EmbeddingIdentityBundleV1, vector: &[f32]) -> BoundQueryEmbedding {
    let mut query_identity = identity.clone();
    query_identity.storage.quantization = QuantizationFormat::F32;
    "in-memory-f32".clone_into(&mut query_identity.storage.format);
    "native-f32-values".clone_into(&mut query_identity.storage.endianness);
    BoundQueryEmbedding::new(vector.to_vec(), query_identity)
        .expect("bind a query embedding in the fixture's space")
}

#[test]
fn native_graph_load_rejects_policy_changes_without_replacing_either_tier() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let fast_path = root.join("fast.fsvi");
    let quality_path = root.join("quality.fsvi");
    let fast_graph = root.join("fast.fshnsw");
    let quality_graph = root.join("quality.fshnsw");
    let (fast_binding, fast_identity) = fsvi_v2_binding("persist-fast", 4, 74);
    let (quality_binding, quality_identity) = fsvi_v2_binding("persist-quality", 4, 74);
    write_v2_tier(
        &fast_path,
        &fast_binding,
        &[("fast", &[1.0, 0.0, 0.0, 0.0])],
    );
    write_v2_tier(
        &quality_path,
        &quality_binding,
        &[("quality", &[0.0, 1.0, 0.0, 0.0])],
    );
    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
    let open = || {
        TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &fast_binding,
            Some(&quality_binding),
        )
        .unwrap()
    };
    let params = HnswParams::default();
    let mut index = open();
    assert_eq!(index.fast_index_path(), fast_path);
    assert_eq!(index.quality_index_path(), Some(quality_path.as_path()));
    index.enable_native_fast_hnsw(params, 42).unwrap();
    index.enable_native_quality_hnsw(params, 43).unwrap();
    let fast_receipt = index.save_native_fast_hnsw(&fast_graph).unwrap();
    let quality_receipt = index.save_native_quality_hnsw(&quality_graph).unwrap();
    assert_eq!(fast_receipt.seed, 42);
    assert_eq!(quality_receipt.seed, 43);
    drop(index);

    let mut index = open();
    assert!(!index.has_native_fast_hnsw());
    assert!(!index.has_native_quality_hnsw());
    index
        .load_native_fast_hnsw(&fast_graph, params, 42)
        .unwrap();
    index
        .load_native_quality_hnsw(&quality_graph, params, 43)
        .unwrap();
    let embeddings = TieredQueryEmbeddings::progressive(
        bound_query(&fast_identity, &[1.0, 0.0, 0.0, 0.0]),
        bound_query(&quality_identity, &[0.0, 1.0, 0.0, 0.0]),
    );
    let before = index
        .activate_owner_backed_search(&embeddings)
        .unwrap()
        .search_union(2)
        .unwrap();
    for changed in [
        HnswParams {
            m: params.m + 1,
            ..params
        },
        HnswParams {
            m0: params.m0 + 1,
            ..params
        },
        HnswParams {
            ef_construction: params.ef_construction + 1,
            ..params
        },
        HnswParams {
            ef_search: params.ef_search + 1,
            ..params
        },
    ] {
        for error in [
            index
                .load_native_fast_hnsw(&fast_graph, changed, 42)
                .unwrap_err(),
            index
                .load_native_quality_hnsw(&quality_graph, changed, 43)
                .unwrap_err(),
        ] {
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field.ends_with(".policy")));
        }
    }
    assert!(
        index
            .load_native_fast_hnsw(&fast_graph, params, 43)
            .is_err()
    );
    assert!(
        index
            .load_native_quality_hnsw(&quality_graph, params, 42)
            .is_err()
    );
    assert!(index.has_native_fast_hnsw());
    assert!(index.has_native_quality_hnsw());
    let fast_policy = index.native_fast_hnsw.as_ref().unwrap();
    let quality_policy = index.native_quality_hnsw.as_ref().unwrap();
    assert_eq!((fast_policy.params, fast_policy.seed), (params, 42));
    assert_eq!((quality_policy.params, quality_policy.seed), (params, 43));
    assert_eq!(
        index
            .activate_owner_backed_search(&embeddings)
            .unwrap()
            .search_union(2)
            .unwrap(),
        before,
    );

    // Corruption must not discard a loaded graph or start a hidden rebuild.
    fs::write(&fast_graph, b"corrupt graph").unwrap();
    assert!(
        index
            .load_native_fast_hnsw(&fast_graph, params, 42)
            .is_err()
    );
    fs::rename(&quality_graph, root.join("moved.fshnsw")).unwrap();
    assert!(
        index
            .load_native_quality_hnsw(&quality_graph, params, 43)
            .is_err()
    );
    assert!(!quality_graph.exists());
    assert_eq!(fs::read(&fast_graph).unwrap(), b"corrupt graph");
    assert_eq!(
        index
            .activate_owner_backed_search(&embeddings)
            .unwrap()
            .search_union(2)
            .unwrap(),
        before,
    );
}

#[test]
fn native_graph_persistence_requires_present_graph_and_quality_owner() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let fast_path = root.join("fast.fsvi");
    let graph_path = root.join("missing.fshnsw");
    let (binding, _) = fsvi_v2_binding("persist-fast-only", 4, 75);
    write_v2_tier(&fast_path, &binding, &[("fast", &[1.0, 0.0, 0.0, 0.0])]);
    let mut index = TwoTierIndex::open_admitted_v2_with_paths(
        &TwoTierIndexPaths::new(&fast_path),
        TwoTierConfig::default(),
        &binding,
        None,
    )
    .unwrap();
    assert!(index.save_native_fast_hnsw(&graph_path).is_err());
    index
        .enable_native_quality_hnsw(HnswParams::default(), 1)
        .unwrap();
    assert!(index.save_native_quality_hnsw(&graph_path).is_err());
    assert!(matches!(
        index.load_native_quality_hnsw(&graph_path, HnswParams::default(), 1),
        Err(SearchError::InvalidConfig { field, .. })
            if field == "two_tier.native_quality_hnsw.owner"
    ));
    assert!(!index.has_native_quality_hnsw());
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn native_graph_save_refuses_vector_and_receipt_aliases_before_writing() {
    for vector_name in ["graph.fshnsw", "graph.fshnsw.receipt"] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let fast_path = root.join(vector_name);
        let graph_path = root.join("graph.fshnsw");
        let (binding, _) = fsvi_v2_binding("persist-alias", 4, 76);
        write_v2_tier(&fast_path, &binding, &[("fast", &[1.0, 0.0, 0.0, 0.0])]);
        let before = fs::read(&fast_path).unwrap();
        let mut index = TwoTierIndex::open_admitted_v2_with_paths(
            &TwoTierIndexPaths::new(&fast_path),
            TwoTierConfig::default(),
            &binding,
            None,
        )
        .unwrap();
        assert_eq!(index.fast_index_path(), fast_path);
        index
            .enable_native_fast_hnsw(HnswParams::default(), 1)
            .unwrap();
        let saved = index.save_native_fast_hnsw(&graph_path);
        assert!(
            matches!(&saved,
                Err(SearchError::InvalidConfig { value, .. }) if value == "vector-alias"
            ),
            "{vector_name}: {saved:?}"
        );
        assert_eq!(fs::read(&fast_path).unwrap(), before);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);

        #[cfg(unix)]
        {
            let hard_link = root.join("hard-link.fshnsw");
            fs::hard_link(&fast_path, &hard_link).unwrap();
            assert!(index.save_native_fast_hnsw(&hard_link).is_err());
            assert_eq!(fs::read(&fast_path).unwrap(), before);
            assert_eq!(fs::read(&hard_link).unwrap(), before);
            assert!(
                !native_hnsw_generation_receipt_path(&hard_link)
                    .unwrap()
                    .exists()
            );
        }
    }
}

#[test]
fn native_fast_full_beam_matches_exact_with_tombstones_and_independent_quality() {
    for precision in [QuantizationFormat::F16, QuantizationFormat::F32] {
        let directory = tempfile::tempdir().unwrap();
        let fast_path = directory.path().join("fast.fsvi");
        let quality_path = directory.path().join("quality.fsvi");
        let (binding, mut identity) = fsvi_v2_binding("native-fast-model", 4, 70);
        identity.storage.quantization = precision;
        let fast_binding =
            FsviV2IdentityBinding::new(binding.generation(), identity.freeze().unwrap()).unwrap();
        let (quality_binding, quality_identity) =
            fsvi_v2_binding("native-fast-independent-quality", 4, 70);
        let mut writer = VectorIndex::create_v2(&fast_path, fast_binding.clone()).unwrap();
        for row in 0..96 {
            let id = format!("fast-{row:03}");
            if row % 10 == 0 {
                writer.write_record(&id, &[1.0, 0.0, 0.0, 0.0]).unwrap();
            } else {
                writer
                    .write_tombstone_record(&id, &[1.0, 0.0, 0.0, 0.0])
                    .unwrap();
            }
        }
        writer.finish().unwrap();
        write_v2_tier(
            &quality_path,
            &quality_binding,
            &[("quality-only", &[0.0, 1.0, 0.0, 0.0])],
        );
        let mut index = TwoTierIndex::open_admitted_v2_with_paths(
            &TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path),
            TwoTierConfig::default(),
            &fast_binding,
            Some(&quality_binding),
        )
        .unwrap();
        assert!(!index.has_native_fast_hnsw());
        let before = index.fast_admitted_owner().unwrap().witness().clone();
        index
            .enable_native_fast_hnsw(
                HnswParams {
                    ef_search: 96,
                    ..HnswParams::default()
                },
                3,
            )
            .unwrap();
        assert!(index.has_native_fast_hnsw());
        assert!(!index.has_native_quality_hnsw());
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
        let TierSource::AdmittedV2 { owner, .. } = &index.fast_source else {
            panic!("fast tier must retain its admitted owner");
        };
        assert_eq!(Arc::strong_count(owner), 2);
        fs::rename(&fast_path, directory.path().join("moved-fast.fsvi")).unwrap();

        let fast_query = bound_query(&identity, &[1.0, 0.0, 0.0, 0.0]);
        let fast_embeddings = TieredQueryEmbeddings::fast_only(fast_query);
        for k in [0, 1, 7, 96] {
            let activated = index
                .activate_owner_backed_search(&fast_embeddings)
                .unwrap();
            let hits = activated.search_fast(k).unwrap();
            assert_eq!(
                hits,
                activated.fast().unwrap().search_top_k(k).unwrap(),
                "precision={precision:?}, k={k}"
            );
            assert!(hits.iter().all(|hit| {
                index
                    .fast_admitted_owner()
                    .unwrap()
                    .row(hit.index as usize)
                    .unwrap()
                    .flags()
                    .is_live()
            }));
            let coverage = activated.coverage(&hits, k).unwrap();
            assert_eq!(coverage.quality, TierQueryCoverageV1::NotRequested);
            assert_eq!(
                coverage.fast,
                TierQueryCoverageV1::Witnessed {
                    generation_sequence: 70,
                    live_count: 10,
                    contributed_candidates: hits.len() as u64,
                }
            );
        }
        let foreign =
            TieredQueryEmbeddings::fast_only(bound_query(&quality_identity, &[1.0, 0.0, 0.0, 0.0]));
        assert!(index.activate_owner_backed_search(&foreign).is_err());
        let quality_embeddings = TieredQueryEmbeddings::quality_only(bound_query(
            &quality_identity,
            &[0.0, 1.0, 0.0, 0.0],
        ));
        let quality_exact = index
            .activate_owner_backed_search(&quality_embeddings)
            .unwrap()
            .search_quality(1)
            .unwrap();
        let progressive_embeddings = TieredQueryEmbeddings::progressive(
            bound_query(&identity, &[1.0, 0.0, 0.0, 0.0]),
            bound_query(&quality_identity, &[0.0, 1.0, 0.0, 0.0]),
        );
        let union_before = index
            .activate_owner_backed_search(&progressive_embeddings)
            .unwrap()
            .search_union(11)
            .unwrap();
        index
            .enable_native_quality_hnsw(HnswParams::default(), 5)
            .unwrap();
        assert!(index.has_native_fast_hnsw());
        assert!(index.has_native_quality_hnsw());
        assert_eq!(
            index
                .activate_owner_backed_search(&quality_embeddings)
                .unwrap()
                .search_quality(1)
                .unwrap(),
            quality_exact
        );
        let both = index
            .activate_owner_backed_search(&progressive_embeddings)
            .unwrap();
        let union = both.search_union(11).unwrap();
        assert_eq!(union, union_before);
        assert!(union.iter().any(|hit| hit.doc_id == "quality-only"));
        let coverage = both.coverage(&union, 11).unwrap();
        assert_eq!(
            coverage.fast,
            TierQueryCoverageV1::Witnessed {
                generation_sequence: 70,
                live_count: 10,
                contributed_candidates: 10,
            }
        );
        assert_eq!(
            coverage.quality,
            TierQueryCoverageV1::Witnessed {
                generation_sequence: 70,
                live_count: 1,
                contributed_candidates: 1,
            }
        );
        assert_eq!(index.fast_admitted_owner().unwrap().witness(), &before);
        let before_hits = index
            .activate_owner_backed_search(&fast_embeddings)
            .unwrap()
            .search_fast(3)
            .unwrap();
        assert!(
            index
                .enable_native_fast_hnsw(
                    HnswParams {
                        m0: 1,
                        ..HnswParams::default()
                    },
                    8
                )
                .is_err()
        );
        assert_eq!(
            index
                .activate_owner_backed_search(&fast_embeddings)
                .unwrap()
                .search_fast(3)
                .unwrap(),
            before_hits
        );
        assert!(index.has_native_quality_hnsw());
    }
}

#[test]
fn native_fast_coverage_counts_approximate_candidates_not_the_exact_oracle() {
    let directory = tempfile::tempdir().unwrap();
    let fast_path = directory.path().join("fast.fsvi");
    let (binding, identity) = fsvi_v2_binding("native-fast-coverage", 4, 73);
    let mut ids = ["a", "b", "c", "d"];
    ids.sort_by_key(|id| crate::fnv1a_hash(id.as_bytes()));
    write_v2_tier(
        &fast_path,
        &binding,
        &[
            (ids[0], &[1.0, 0.0, 0.0, 0.0]),
            (ids[1], &[0.0, 0.0, 1.0, 0.0]),
            (ids[2], &[0.0, -1.0, 0.0, 0.0]),
            (ids[3], &[0.0, 1.0, 0.0, 0.0]),
        ],
    );
    let mut index = TwoTierIndex::open_admitted_v2_with_paths(
        &TwoTierIndexPaths::new(&fast_path),
        TwoTierConfig::default(),
        &binding,
        None,
    )
    .unwrap();
    // Deliberately narrow beam: the exact winner is beyond an intervening
    // valley on the physical 0-1-2-3 path, distinguishing routed retrieval.
    index
        .enable_native_fast_hnsw(
            HnswParams {
                m0: 2,
                ef_search: 1,
                ..HnswParams::default()
            },
            3,
        )
        .unwrap();
    let embeddings =
        TieredQueryEmbeddings::fast_only(bound_query(&identity, &[0.0, 1.0, 0.0, 0.0]));
    let activated = index.activate_owner_backed_search(&embeddings).unwrap();
    let exact = activated.fast().unwrap().search_top_k(1).unwrap();
    let approximate = activated.search_fast(1).unwrap();
    assert_eq!(exact[0].doc_id, ids[3]);
    assert_eq!(approximate[0].doc_id, ids[0]);
    for (returned, expected_contribution) in [(&approximate, 1), (&exact, 0)] {
        let coverage = activated.coverage(returned, 1).unwrap();
        assert_eq!(
            coverage.fast,
            TierQueryCoverageV1::Witnessed {
                generation_sequence: 73,
                live_count: 4,
                contributed_candidates: expected_contribution,
            }
        );
        assert_eq!(coverage.quality, TierQueryCoverageV1::NotRequested);
    }
}

#[test]
fn native_quality_full_beam_matches_exact_with_tombstones_and_partial_coverage() {
    for precision in [QuantizationFormat::F16, QuantizationFormat::F32] {
        let directory = tempfile::tempdir().unwrap();
        let fast_path = directory.path().join("fast.fsvi");
        let quality_path = directory.path().join("quality.fsvi");
        let (fast_binding, _) = fsvi_v2_binding("native-quality-fast", 4, 71);
        let (quality_binding, mut quality_identity) =
            fsvi_v2_binding("native-quality-model", 4, 71);
        quality_identity.storage.quantization = precision;
        let quality_binding = FsviV2IdentityBinding::new(
            quality_binding.generation(),
            quality_identity.freeze().unwrap(),
        )
        .unwrap();
        write_v2_tier(
            &fast_path,
            &fast_binding,
            &[("fast-only", &[1.0, 0.0, 0.0, 0.0])],
        );
        let mut writer = VectorIndex::create_v2(&quality_path, quality_binding.clone()).unwrap();
        for row in 0..96 {
            let id = format!("quality-{row:03}");
            if row % 10 == 0 {
                writer.write_record(&id, &[0.0, 1.0, 0.0, 0.0]).unwrap();
            } else {
                writer
                    .write_tombstone_record(&id, &[0.0, 1.0, 0.0, 0.0])
                    .unwrap();
            }
        }
        writer.finish().unwrap();
        let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
        let mut index = TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &fast_binding,
            Some(&quality_binding),
        )
        .unwrap();
        assert!(!index.has_native_quality_hnsw());
        let before = index.quality_admitted_owner().unwrap().witness().clone();
        index
            .enable_native_quality_hnsw(
                HnswParams {
                    ef_search: 96,
                    ..HnswParams::default()
                },
                3,
            )
            .unwrap();
        assert!(index.has_native_quality_hnsw());
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
        let Some(TierSource::AdmittedV2 { owner, .. }) = index.quality_source.as_ref() else {
            panic!("quality must retain its admitted owner");
        };
        assert_eq!(
            Arc::strong_count(owner),
            2,
            "tier and graph share one owner"
        );
        fs::rename(&quality_path, directory.path().join("moved-quality.fsvi")).unwrap();
        let embeddings = TieredQueryEmbeddings::quality_only(bound_query(
            &quality_identity,
            &[0.0, 1.0, 0.0, 0.0],
        ));
        for k in [0, 1, 7, 96] {
            let activated = index.activate_owner_backed_search(&embeddings).unwrap();
            let exact = activated.quality().unwrap().search_top_k(k).unwrap();
            let hits = activated.search_quality(k).unwrap();
            assert_eq!(hits, exact, "precision={precision:?}, k={k}");
            assert!(hits.iter().all(|hit| {
                index
                    .quality_admitted_owner()
                    .unwrap()
                    .row(hit.index as usize)
                    .unwrap()
                    .flags()
                    .is_live()
            }));
            assert_eq!(
                activated.coverage(&hits, k).unwrap().quality,
                TierQueryCoverageV1::Witnessed {
                    generation_sequence: 71,
                    live_count: 10,
                    contributed_candidates: hits.len() as u64,
                }
            );
        }
        assert_eq!(index.quality_admitted_owner().unwrap().witness(), &before);
        let before_hits = index
            .activate_owner_backed_search(&embeddings)
            .unwrap()
            .search_quality(1)
            .unwrap();
        assert!(
            index
                .enable_native_quality_hnsw(
                    HnswParams {
                        m0: 1,
                        ..HnswParams::default()
                    },
                    9,
                )
                .is_err()
        );
        assert!(index.has_native_quality_hnsw());
        assert_eq!(
            index
                .activate_owner_backed_search(&embeddings)
                .unwrap()
                .search_quality(1)
                .unwrap(),
            before_hits
        );
    }
}

#[test]
fn native_quality_coverage_counts_approximate_candidates_not_the_exact_oracle() {
    let directory = tempfile::tempdir().unwrap();
    let fast_path = directory.path().join("fast.fsvi");
    let quality_path = directory.path().join("quality.fsvi");
    let (fast_binding, _) = fsvi_v2_binding("native-coverage-fast", 4, 72);
    let (quality_binding, quality_identity) = fsvi_v2_binding("native-coverage-quality", 4, 72);
    let mut ids = ["a", "b", "c", "d"];
    ids.sort_by_key(|id| crate::fnv1a_hash(id.as_bytes()));
    write_v2_tier(
        &fast_path,
        &fast_binding,
        &[("fast", &[1.0, 0.0, 0.0, 0.0])],
    );
    write_v2_tier(
        &quality_path,
        &quality_binding,
        &[
            (ids[0], &[1.0, 0.0, 0.0, 0.0]),
            (ids[1], &[0.0, 0.0, 1.0, 0.0]),
            (ids[2], &[0.0, -1.0, 0.0, 0.0]),
            (ids[3], &[0.0, 1.0, 0.0, 0.0]),
        ],
    );
    let mut index = TwoTierIndex::open_admitted_v2_with_paths(
        &TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path),
        TwoTierConfig::default(),
        &fast_binding,
        Some(&quality_binding),
    )
    .unwrap();
    // Seed 3 puts all four rows at layer zero. Degree two preserves the
    // 0-1-2-3 physical path; beam one stops at row 0's tied neighbour rather
    // than crossing row 2 to the exact winner. This distinguishes the modes.
    index
        .enable_native_quality_hnsw(
            HnswParams {
                m0: 2,
                ef_search: 1,
                ..HnswParams::default()
            },
            3,
        )
        .unwrap();
    let embeddings =
        TieredQueryEmbeddings::quality_only(bound_query(&quality_identity, &[0.0, 1.0, 0.0, 0.0]));
    let activated = index.activate_owner_backed_search(&embeddings).unwrap();
    let exact = activated.quality().unwrap().search_top_k(1).unwrap();
    let approximate = activated.search_quality(1).unwrap();
    assert_eq!(exact[0].doc_id, ids[3]);
    assert_eq!(approximate[0].doc_id, ids[0]);
    for (returned, expected_contribution) in [(&approximate, 1), (&exact, 0)] {
        assert_eq!(
            activated.coverage(returned, 1).unwrap().quality,
            TierQueryCoverageV1::Witnessed {
                generation_sequence: 72,
                live_count: 4,
                contributed_candidates: expected_contribution,
            }
        );
    }
}

#[test]
fn native_policies_survive_transactional_replacement_and_missing_quality() {
    for enable_fast in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let make_generation = |sequence, quality_doc: Option<&str>| {
            let generation_dir = directory.path().join(format!("generation-{sequence}"));
            fs::create_dir(&generation_dir).unwrap();
            let fast_path = generation_dir.join("fast.fsvi");
            let (fast_binding, _) = fsvi_v2_binding("native-refresh-fast", 4, sequence);
            write_v2_tier(
                &fast_path,
                &fast_binding,
                &[(&format!("fast-{sequence}"), &[1.0, 0.0, 0.0, 0.0])],
            );
            let mut paths = TwoTierIndexPaths::new(fast_path);
            let quality_binding = quality_doc.map(|doc| {
                let quality_path = generation_dir.join("quality.fsvi");
                let (binding, _) = fsvi_v2_binding("native-refresh-quality", 4, sequence);
                write_v2_tier(&quality_path, &binding, &[(doc, &[0.0, 1.0, 0.0, 0.0])]);
                paths = paths.clone().with_quality_index(quality_path);
                binding
            });
            (paths, fast_binding, quality_binding)
        };
        let (paths, fast, quality) = make_generation(1, Some("old-quality"));
        let open = || {
            TwoTierIndex::open_admitted_v2_with_paths(
                &paths,
                TwoTierConfig::default(),
                &fast,
                quality.as_ref(),
            )
            .unwrap()
        };
        let mut index = open();
        let mut retained_reader = open();
        assert_eq!(index.fast_index_path(), paths.fast_index());
        assert_eq!(index.quality_index_path(), paths.quality_index());
        if enable_fast {
            index
                .enable_native_fast_hnsw(HnswParams::default(), 3)
                .unwrap();
            retained_reader
                .enable_native_fast_hnsw(HnswParams::default(), 3)
                .unwrap();
        }
        index
            .enable_native_quality_hnsw(HnswParams::default(), 7)
            .unwrap();
        retained_reader
            .enable_native_quality_hnsw(HnswParams::default(), 7)
            .unwrap();
        let (_, identity) = fsvi_v2_binding("native-refresh-quality", 4, 1);
        let embeddings =
            TieredQueryEmbeddings::quality_only(bound_query(&identity, &[0.0, 1.0, 0.0, 0.0]));
        let before = index.quality_admitted_owner().unwrap().witness().clone();
        let before_fast = index.fast_admitted_owner().unwrap().witness().clone();
        let (bad_paths, bad_fast, bad_quality) = make_generation(4, Some("bad-quality"));
        fs::write(bad_paths.quality_index().unwrap(), b"invalid fsvi").unwrap();
        assert!(
            index
                .try_replace_admitted_v2(&bad_paths, &bad_fast, bad_quality.as_ref())
                .is_err()
        );
        assert!(index.has_native_quality_hnsw());
        assert_eq!(index.has_native_fast_hnsw(), enable_fast);
        assert_eq!(index.fast_admitted_owner().unwrap().witness(), &before_fast);
        assert_eq!(index.quality_admitted_owner().unwrap().witness(), &before);
        assert_eq!(index.fast_index_path(), paths.fast_index());
        assert_eq!(index.quality_index_path(), paths.quality_index());
        assert_eq!(
            index
                .activate_owner_backed_search(&embeddings)
                .unwrap()
                .search_quality(1)
                .unwrap()[0]
                .doc_id,
            "old-quality"
        );

        let (mixed_paths, mixed_fast, _) = make_generation(5, Some("mixed-quality"));
        let (foreign_quality, _) = fsvi_v2_binding("native-refresh-quality", 4, 6);
        write_v2_tier(
            mixed_paths.quality_index().unwrap(),
            &foreign_quality,
            &[("mixed-quality", &[0.0, 1.0, 0.0, 0.0])],
        );
        // Both files are independently admissible and their legacy nonces
        // agree. Native activation must still join their full v2 generations.
        let ordinary = TwoTierIndex::open_admitted_v2_with_paths(
            &mixed_paths,
            TwoTierConfig::default(),
            &mixed_fast,
            Some(&foreign_quality),
        )
        .unwrap();
        assert!(ordinary.quality_admitted_owner().is_some());
        assert!(matches!(
            index.try_replace_admitted_v2(&mixed_paths, &mixed_fast, Some(&foreign_quality)),
            Err(SearchError::InvalidConfig { field, .. })
                if field == if enable_fast {
                    "two_tier.native_fast_hnsw.generation"
                } else {
                    "two_tier.native_quality_hnsw.generation"
                }
        ));
        assert!(index.has_native_quality_hnsw());
        assert_eq!(index.has_native_fast_hnsw(), enable_fast);
        assert_eq!(index.fast_admitted_owner().unwrap().witness(), &before_fast);
        assert_eq!(index.quality_admitted_owner().unwrap().witness(), &before);
        assert_eq!(index.fast_index_path(), paths.fast_index());
        assert_eq!(index.quality_index_path(), paths.quality_index());

        let (fast_only_paths, fast_only_binding, _) = make_generation(2, None);
        index
            .try_replace_admitted_v2(&fast_only_paths, &fast_only_binding, None)
            .unwrap();
        assert!(!index.has_native_quality_hnsw());
        assert_eq!(index.has_native_fast_hnsw(), enable_fast);
        assert_eq!(index.fast_index_path(), fast_only_paths.fast_index());
        assert_eq!(index.quality_index_path(), None);
        let (_, fast_identity) = fsvi_v2_binding("native-refresh-fast", 4, 1);
        let fast_embeddings =
            TieredQueryEmbeddings::fast_only(bound_query(&fast_identity, &[1.0, 0.0, 0.0, 0.0]));
        assert_eq!(
            index
                .activate_owner_backed_search(&fast_embeddings)
                .unwrap()
                .search_fast(1)
                .unwrap()[0]
                .doc_id,
            "fast-2"
        );
        assert!(index.quality_admitted_owner().is_none());
        assert!(index.activate_owner_backed_search(&embeddings).is_err());

        let (successor_paths, successor_fast, successor_quality) =
            make_generation(3, Some("new-quality"));
        index
            .try_replace_admitted_v2(
                &successor_paths,
                &successor_fast,
                successor_quality.as_ref(),
            )
            .unwrap();
        assert!(index.has_native_quality_hnsw());
        assert_eq!(index.has_native_fast_hnsw(), enable_fast);
        assert_eq!(index.fast_index_path(), successor_paths.fast_index());
        assert_eq!(index.quality_index_path(), successor_paths.quality_index());
        assert_eq!(retained_reader.fast_index_path(), paths.fast_index());
        assert_eq!(retained_reader.quality_index_path(), paths.quality_index());
        assert_eq!(
            index
                .activate_owner_backed_search(&fast_embeddings)
                .unwrap()
                .search_fast(1)
                .unwrap()[0]
                .doc_id,
            "fast-3"
        );
        assert_eq!(
            retained_reader
                .activate_owner_backed_search(&fast_embeddings)
                .unwrap()
                .search_fast(1)
                .unwrap()[0]
                .doc_id,
            "fast-1"
        );
        assert_eq!(
            retained_reader.fast_admitted_owner().unwrap().witness(),
            &before_fast
        );
        assert_eq!(
            index
                .activate_owner_backed_search(&embeddings)
                .unwrap()
                .search_quality(1)
                .unwrap()[0]
                .doc_id,
            "new-quality"
        );
        assert_eq!(
            retained_reader
                .activate_owner_backed_search(&embeddings)
                .unwrap()
                .search_quality(1)
                .unwrap()[0]
                .doc_id,
            "old-quality"
        );
        assert_eq!(
            retained_reader.quality_admitted_owner().unwrap().witness(),
            &before
        );
        assert!(
            !open().has_native_quality_hnsw(),
            "a fresh open starts exact"
        );
        assert!(!open().has_native_fast_hnsw());
    }
}

/// bd-ctzo C2: the document that matters is ABSENT from the fast tier's
/// top-k, so `QualityOnly` must reach it by searching the quality owner
/// directly, and `FullProgressive` must reach it through an independent
/// per-tier union -- not by rescoring a fast-selected pool.
///
/// The fixture is deliberately adversarial: the fast tier's vectors put
/// doc-far last, so any implementation that takes the fast top-1 and
/// rescores it can never surface doc-far. The final assertion is the
/// planted negative for exactly that anti-pattern.
#[test]
fn quality_only_and_union_reach_a_document_outside_the_fast_pool() {
    let dir = temp_index_dir("ctzo-out-of-fast-pool");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    let (fast_binding, fast_identity) = fsvi_v2_binding("ctzo-fast-model", 4, 71);
    let (quality_binding, quality_identity) = fsvi_v2_binding("ctzo-quality-model", 4, 71);

    // doc-far is orthogonal to the fast query and adjacent to it in the
    // quality space: the tiers genuinely disagree about relevance.
    write_v2_tier(
        &fast_path,
        &fast_binding,
        &[
            ("doc-near", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-far", &[0.0, 0.0, 1.0, 0.0]),
        ],
    );
    // The quality tier additionally holds a document the FAST TIER DOES
    // NOT CONTAIN AT ALL. No amount of rescoring a fast-selected pool can
    // ever produce it, which is what makes the union assertion below able
    // to tell independent retrieval from fast-pool rescoring.
    write_v2_tier(
        &quality_path,
        &quality_binding,
        &[
            ("doc-near", &[0.0, 0.0, 1.0, 0.0]),
            ("doc-far", &[0.0, 0.5, 0.0, 0.0]),
            ("doc-quality-only", &[0.0, 1.0, 0.0, 0.0]),
        ],
    );

    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
    let index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &fast_binding,
        Some(&quality_binding),
    )
    .expect("admit both tiers");

    let fast_query = bound_query(&fast_identity, &[1.0, 0.0, 0.0, 0.0]);
    let quality_query = bound_query(&quality_identity, &[0.0, 1.0, 0.0, 0.0]);

    // The fast pool at k=1 contains ONLY doc-near.
    let fast_only_embeddings = TieredQueryEmbeddings::fast_only(fast_query.clone());
    let fast_only = index
        .activate_owner_backed_search(&fast_only_embeddings)
        .expect("fast-only activation");
    // HashControl, not FastOnly: `fsvi_v2_binding` builds its identities
    // from `explicit_test_model`, which is a deterministic control space.
    // The retrieval SHAPE under test is unchanged -- what changed
    // (bd-ctzo C4) is that a control-lane query no longer reports one of
    // the three semantic topology names. Before that guard these
    // assertions read `FastOnly` / `QualityOnly` / `FullProgressive`,
    // which was this fixture claiming semantic availability it has never
    // had.
    assert_eq!(fast_only.topology(), RetrievalTopology::HashControl);
    let fast_pool = fast_only.search_fast(1).expect("fast search");
    assert_eq!(fast_pool[0].doc_id, "doc-near");
    assert!(
        !fast_pool.iter().any(|hit| hit.doc_id == "doc-far"),
        "the fixture is only meaningful while doc-far is outside the fast pool"
    );

    // QualityOnly retrieves from the quality owner directly and reaches it.
    let quality_only_embeddings = TieredQueryEmbeddings::quality_only(quality_query.clone());
    let quality_only = index
        .activate_owner_backed_search(&quality_only_embeddings)
        .expect("quality-only activation");
    assert_eq!(quality_only.topology(), RetrievalTopology::HashControl);
    assert_eq!(
        quality_only.search_quality(1).expect("quality search")[0].doc_id,
        "doc-quality-only",
        "QualityOnly must reach a document the fast tier does not even contain"
    );
    assert!(
        quality_only.search_fast(1).is_err(),
        "a quality-only activation exposes no fast retrieval"
    );

    // FullProgressive unions independent per-tier retrieval.
    let progressive_embeddings = TieredQueryEmbeddings::progressive(fast_query, quality_query);
    let progressive = index
        .activate_owner_backed_search(&progressive_embeddings)
        .expect("progressive activation");
    assert_eq!(progressive.topology(), RetrievalTopology::HashControl);
    let union = progressive.search_union(2).expect("union");
    let union_ids: Vec<&str> = union.iter().map(|hit| hit.doc_id.as_str()).collect();
    // THE DISCRIMINATING ASSERTION: doc-quality-only exists in no fast
    // pool, so a union that quietly degraded to fast-pool retrieval --
    // or to rescoring one -- cannot satisfy this.
    assert!(
        union_ids.contains(&"doc-quality-only"),
        "the union must carry independent quality retrieval, got {union_ids:?}"
    );
    assert!(
        union_ids.contains(&"doc-near"),
        "the union must also carry the fast tier's own selection, got {union_ids:?}"
    );
    // Deterministic order: scores descending, doc_id breaking ties, so the
    // result never depends on which tier answered first.
    for pair in union.windows(2) {
        assert!(
            pair[0].score > pair[1].score
                || (pair[0].score.to_bits() == pair[1].score.to_bits()
                    && pair[0].doc_id <= pair[1].doc_id),
            "union ordering must be deterministic, got {union_ids:?}"
        );
    }

    // The same fixture states the anti-pattern directly: the fast pool
    // this document would have to be rescored from never contained it.
    assert!(
        !fast_pool.iter().any(|hit| hit.doc_id == "doc-quality-only"),
        "rescoring a fast-selected pool can never surface doc-quality-only; that is \
         precisely why direct quality retrieval and the union are first-class operations"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// bd-ctzo C1/C3: every identity join happens at activation, before any
/// vector read is reachable. Each case diverges in exactly one component
/// and must be refused by field name; the matching activation in the same
/// test is the positive control.
#[test]
fn activation_refuses_every_identity_divergence_before_any_read() {
    let dir = temp_index_dir("ctzo-activation-refusals");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (binding, identity) = fsvi_v2_binding("ctzo-guard-model", 4, 81);
    write_v2_tier(
        &fast_path,
        &binding,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ],
    );
    let paths = TwoTierIndexPaths::new(&fast_path);
    let index =
        TwoTierIndex::open_admitted_v2_with_paths(&paths, TwoTierConfig::default(), &binding, None)
            .expect("admit the fast tier");

    // The identity every join below is measured against is the artifact's
    // own: it is the binding admission proved this file's canonical bytes
    // equal, retained on the tier. There is no parameter through which a
    // caller could offer a different one, which is why "a binding foreign
    // to the artifact" is absent from the cases below -- it stopped being
    // a refusal and became unrepresentable.
    assert_eq!(
        index.fast_admitted_binding(),
        Some(&binding),
        "the tier must retain the exact binding it was admitted under"
    );

    // Positive control: the matching query activates and serves.
    let matching = TieredQueryEmbeddings::fast_only(bound_query(&identity, &[1.0, 0.0, 0.0, 0.0]));
    let activated = index
        .activate_owner_backed_search(&matching)
        .expect("the matching query activates");
    assert_eq!(activated.search_fast(1).expect("search")[0].doc_id, "doc-a");

    // (i) Same dimension, different embedding space: the case raw vector
    // APIs silently accept.
    let (_, foreign_identity) = fsvi_v2_binding("a-foreign-model", 4, 81);
    let foreign =
        TieredQueryEmbeddings::fast_only(bound_query(&foreign_identity, &[1.0, 0.0, 0.0, 0.0]));
    let error = index
        .activate_owner_backed_search(&foreign)
        .expect_err("a same-dimension foreign space must be refused");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. }
                if field == "query_embedding.fast.space_identity"
        ),
        "got {error:?}"
    );

    // (ii) THE SAME SPACE, A DIFFERENT PRODUCER, CERTIFIED COMPATIBLE.
    // This is the case a fingerprint-only join admits and the complete
    // bd-9xuj law does not: the space fingerprint is byte-identical (only
    // producer-side fields move, and none of them feed the space), and
    // both producers carry the same pinned golden-vector certificate, so
    // `verify_producer_conformance` returns
    // `ConformanceCompatibleProducer` rather than an error. That verdict
    // is comparison-grade telemetry, so activation must REFUSE it rather
    // than admit with a log line.
    let mut compatible_identity = identity.clone();
    "a-different-implementation".clone_into(&mut compatible_identity.producer.backend);
    assert_eq!(
        compatible_identity.space.fingerprint(),
        identity.space.fingerprint(),
        "the fixture is only meaningful while the SPACE is unchanged"
    );
    assert_ne!(
        compatible_identity.producer.fingerprint(),
        identity.producer.fingerprint(),
        "the fixture is only meaningful while the PRODUCER differs"
    );
    assert_eq!(
        compatible_identity.producer.golden_vectors, identity.producer.golden_vectors,
        "the fixture is only meaningful while both producers are certified compatible"
    );
    let compatible =
        TieredQueryEmbeddings::fast_only(bound_query(&compatible_identity, &[1.0, 0.0, 0.0, 0.0]));
    let error = index
        .activate_owner_backed_search(&compatible)
        .expect_err("a certified-compatible foreign producer must be refused");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. }
                if field == "search_activation.fast.producer_conformance"
        ),
        "got {error:?}"
    );

    // (iii) A quality-bound query against an index that retains no quality
    // owner: typed refusal, never a silent fast-tier substitution.
    let quality_bound =
        TieredQueryEmbeddings::quality_only(bound_query(&identity, &[1.0, 0.0, 0.0, 0.0]));
    let error = index
        .activate_owner_backed_search(&quality_bound)
        .expect_err("a missing quality owner must be refused");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. }
                if field == "search_activation.quality.owner"
        ),
        "got {error:?}"
    );

    // (iv) A legacy v1 index retains no owner at all, so typed activation
    // is unreachable rather than degraded.
    let legacy_dir = temp_index_dir("ctzo-legacy");
    fs::create_dir_all(&legacy_dir).expect("create temp dir");
    let legacy_path = legacy_dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&legacy_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write v1 fixture");
    let legacy = TwoTierIndex::open(&legacy_dir, TwoTierConfig::default()).expect("open legacy v1");
    assert!(
        legacy.fast_admitted_binding().is_none(),
        "a legacy v1 tier retains no admitted identity to join against"
    );
    let error = legacy
        .activate_owner_backed_search(&matching)
        .expect_err("a legacy tier is not searchable under typed activation");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. }
                if field == "search_activation.fast.owner"
        ),
        "got {error:?}"
    );

    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&legacy_dir);
}

/// bd-ctzo C4: coverage is reconstructed from the retained owner's own
/// witnesses, never from a caller-supplied scalar.
#[test]
fn activated_coverage_comes_from_the_owner_witness() {
    let dir = temp_index_dir("ctzo-coverage");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (binding, identity) = fsvi_v2_binding("ctzo-coverage-model", 4, 91);
    write_v2_tier(
        &fast_path,
        &binding,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
        ],
    );
    let index = TwoTierIndex::open_admitted_v2_with_paths(
        &TwoTierIndexPaths::new(&fast_path),
        TwoTierConfig::default(),
        &binding,
        None,
    )
    .expect("admit the fast tier");
    let embeddings =
        TieredQueryEmbeddings::fast_only(bound_query(&identity, &[1.0, 0.0, 0.0, 0.0]));
    let activated = index
        .activate_owner_backed_search(&embeddings)
        .expect("activate");
    let tier = activated.fast().expect("fast tier activated");
    assert_eq!(tier.generation_sequence(), 91);
    assert_eq!(tier.live_count(), 3);
    assert_eq!(
        tier.owner().witness().record_count,
        3,
        "coverage is read from the retained witness, not from the caller"
    );

    // The assembled receipt: every populated field is the owner's witness
    // or a count taken against the results actually returned.
    let returned = activated.search_fast(2).expect("fast search");
    assert_eq!(returned.len(), 2);
    let coverage = activated.coverage(&returned, 2).expect("coverage");
    assert_eq!(
        coverage.fast,
        TierQueryCoverageV1::Witnessed {
            generation_sequence: 91,
            live_count: 3,
            contributed_candidates: 2,
        }
    );
    assert_eq!(
        coverage.quality,
        TierQueryCoverageV1::NotRequested,
        "a tier the query never bound is NOT REQUESTED -- not a tier that covered nothing"
    );
    assert_eq!(coverage.quality.witnessed_live_count(), None);
    assert!(
        coverage.is_hash_control(),
        "the fixture space is a deterministic control, so the receipt must say so"
    );
    // CONTRIBUTION IS COUNTED, NOT PREDICTED: hand the same activation a
    // result set containing a document this tier did not serve, and the
    // count drops rather than tracking `returned.len()`.
    let foreign = vec![VectorHit {
        index: u32::MAX,
        score: 1.0,
        doc_id: "doc-from-somewhere-else".into(),
    }];
    let diluted = activated.coverage(&foreign, 2).expect("coverage");
    assert_eq!(
        diluted.fast,
        TierQueryCoverageV1::Witnessed {
            generation_sequence: 91,
            live_count: 3,
            contributed_candidates: 0,
        }
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn plain_v1_open_has_no_admitted_owners() {
    let dir = temp_index_dir("plain-v1-no-owners");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write v1 fixture");
    let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open v1");
    assert!(index.fast_admitted_owner().is_none());
    assert!(index.quality_admitted_owner().is_none());
    let _ = fs::remove_dir_all(&dir);
}
