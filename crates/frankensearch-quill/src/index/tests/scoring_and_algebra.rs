use super::*;

#[test]
fn tombstone_fold_keeps_multi_must_score_bits_across_delta_seal() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let generation = index
            .snapshot()
            .expect("tombstone-fold snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let mut delta =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("score-order Delta");

        for global_docid in 0..3 {
            apply_tokenized_delta_document(
                &mut delta,
                global_docid,
                &format!("dead-a-{global_docid}"),
                "a",
            );
        }
        apply_tokenized_delta_document(&mut delta, 3, "target", "a b b c c x x");
        apply_tokenized_delta_document(&mut delta, 4, "live-bc", "b c x x");
        apply_tokenized_delta_document(&mut delta, 5, "live-c", "c x x x");
        for document_id in ["dead-a-0", "dead-a-1", "dead-a-2"] {
            assert!(delta.delete_delta_id(document_id).is_some());
        }
        let sealed = Arc::new(delta.freeze(generation));
        let a = sealed
            .find_term(CONTENT_FIELD, b"a")
            .expect("physical a term");
        assert_eq!(a.physical_doc_freq(), 4);
        assert_eq!(a.live_doc_freq(), 1);
        assert_eq!(
            DeltaPostingCursor::new(&sealed, CONTENT_FIELD, b"a")
                .expect("live a cursor")
                .cost(),
            1,
            "Delta traversal cost must match the sealed live posting count"
        );
        index
            .publish_delta_table(vec![Arc::clone(&sealed)])
            .expect("publish score-order Delta");
        let query = "content:a AND content:b AND content:c";
        let before = index
            .search_paginated(&cx, query, 10, 0, true)
            .expect("query multi-MUST Delta");
        assert_eq!(before.hits.len(), 1);
        assert_eq!(before.hits[0].document_id, "target");

        index
            .seal_delta_snapshot(
                &cx,
                Arc::clone(&sealed),
                Vec::new(),
                DeltaFlushInput {
                    segment_id: 0xe5_4032,
                    created_unix_s: 1_700_000_032,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .await
            .expect("seal score-order Delta");
        let after = index
            .search_paginated(&cx, query, 10, 0, true)
            .expect("query multi-MUST Keeper");
        assert_eq!(after.hits.len(), 1);
        assert_eq!(after.hits[0].document_id, "target");
        assert_eq!(
            before.hits[0].score.to_bits(),
            after.hits[0].score.to_bits()
        );
        assert_eq!(before, after);
    });
}

#[test]
fn delta_and_equivalent_sealed_term_have_identical_score_bits() {
    run_with_cx(|cx| async move {
        let keeper =
            Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        apply_alpha_delta(&mut delta, 0, "doc", 1);
        let frozen = Arc::new(delta.freeze(generation));
        let composite = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
            .expect("pre-seal composite snapshot");
        let pre_stats = composite
            .bm25_field_stats(CONTENT_FIELD)
            .expect("pre-seal content stats");
        let pre_df = composite
            .bm25_doc_freq(CONTENT_FIELD, b"alpha")
            .expect("pre-seal alpha df");
        let mut pre_scorer = lower_delta_term(&frozen, &composite, CONTENT_FIELD, b"alpha", 1.0)
            .expect("pre-seal Delta term scorer");

        let sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
        sealed
            .index_documents(&cx, &[IndexableDocument::new("doc", "alpha")])
            .await
            .expect("accumulate equivalent sealed document");
        sealed.commit(&cx).await.expect("seal equivalent document");
        let keeper = sealed
            .snapshot()
            .expect("sealed parity snapshot is authoritative");
        let post_stats = snapshot_field(&keeper, CONTENT_FIELD).expect("post-seal stats");
        let post_df = snapshot_doc_freq(&keeper, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
            .expect("post-seal alpha df");
        assert_eq!(pre_stats, post_stats);
        assert_eq!(pre_df, post_df);

        let mut post_scorer = lower_term(
            &keeper.segments()[0],
            &keeper,
            DEFAULT_SCHEMA,
            CONTENT_FIELD,
            b"alpha",
            1.0,
        )
        .expect("post-seal term scorer");
        assert_eq!(
            pre_scorer.score().expect("pre-seal score").to_bits(),
            post_scorer.score().expect("post-seal score").to_bits(),
            "BM25 score must be bit-identical across the seal boundary"
        );
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn basic_record_option_scores_and_bounds_repeated_delta_occurrences_as_presence() {
    let keeper = Arc::new(
        KeeperSnapshot::in_memory(POSITIONLESS_QG_SCHEMA)
            .expect("genesis positionless Keeper snapshot"),
    );
    let generation = keeper.loaded_manifest().manifest.generation;
    let mut delta =
        DeltaSegment::new(POSITIONLESS_QG_SCHEMA, 0, usize::MAX).expect("Basic Delta shard");
    apply_tokenized_delta_document(&mut delta, 0, "repeat", "token token neutral");
    apply_tokenized_delta_document(&mut delta, 1, "single", "token single neutral");
    let frozen = Arc::new(delta.freeze(generation));
    let composite = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
        .expect("positionless Delta composite snapshot");
    let stats = composite
        .bm25_field_stats(CONTENT_FIELD)
        .expect("positionless Delta content statistics");
    assert_eq!((stats.total_tokens, stats.doc_count), (6, 2));
    assert_eq!(
        composite
            .bm25_doc_freq(CONTENT_FIELD, b"token")
            .expect("positionless Delta token document frequency"),
        2
    );

    let mut cursor = DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"token")
        .expect("positionless Delta token cursor");
    assert_eq!((cursor.doc(), cursor.freq()), (Some(0), Some(1)));
    assert!(cursor.positions_handle().is_none());
    assert_eq!(cursor.next().expect("advance Basic Delta cursor"), Some(1));
    assert_eq!(cursor.freq(), Some(1));
    assert!(cursor.positions_handle().is_none());

    let average = stats
        .average_field_length()
        .expect("non-empty positionless Delta average");
    let weight = crate::contract::idf(2, 2) * (1.0 + crate::contract::BM25_K1);
    let bound = cursor
        .term_score_upper_bound(average, weight, TermRecordOption::Basic)
        .expect("positionless Delta term bound");
    let mut scorer = lower_delta_term(&frozen, &composite, CONTENT_FIELD, b"token", 1.0)
        .expect("positionless Delta term scorer");
    let repeated_score = scorer.score().expect("score repeated Delta occurrence");
    assert_eq!(
        repeated_score.to_bits(),
        bound.to_bits(),
        "Delta Basic term bound must use the scorer's effective tf=1"
    );
    assert_eq!(scorer.next().expect("advance Basic Delta scorer"), Some(1));
    let single_score = scorer.score().expect("score single Delta occurrence");
    assert_eq!(
        repeated_score.to_bits(),
        single_score.to_bits(),
        "Delta Basic postings score document presence, not retained raw frequency"
    );
    assert_eq!(scorer.next().expect("exhaust Basic Delta scorer"), None);
}

/// The shipping sealed lowering must put a ranked term on the validated
/// arrival path, and the Delta lowering must not
/// (`bd-quill-pruned-topdocs-term-order-9wu3p`).
///
/// This is the link a fixture inside `argus` cannot supply for itself. The
/// seeking fill is only reachable from [`TermCursor::Validated`], and that
/// arm is only reachable by handing `TermScorer::new_validated` a concrete
/// `ValidatedArrivalCursor`. Whether *shipping* does that is a property of
/// `lower_leaf_term`, so this drives that function directly rather than a
/// scorer assembled by a test helper: without it the whole capability could
/// be green in unit tests and dead in production.
///
/// The Delta half is what keeps the claim honest. Delta has no block-skip
/// structure and cannot answer arrival reads, so it must stay opaque; an
/// assertion that only checked the sealed side would pass just as well if
/// every leaf had been wired to the validated arm.
#[test]
fn shipping_sealed_lowering_reaches_the_validated_arrival_path() {
    run_with_cx(|cx| async move {
        let sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
        sealed
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("one", "alpha"),
                    IndexableDocument::new("three", "alpha alpha alpha"),
                    IndexableDocument::new("two", "alpha alpha"),
                ],
            )
            .await
            .expect("accumulate sealed corpus");
        sealed.commit(&cx).await.expect("seal corpus");
        // `lower_leaf_term` reads composite statistics, so it takes the
        // composed search snapshot rather than the Keeper snapshot the
        // segments come from. The Keeper handle is cloned rather than moved
        // because the sealed leaf borrows a segment out of it.
        let keeper = sealed.snapshot().expect("sealed snapshot is authoritative");
        let composed = QuillSearchSnapshot::compose(0, Arc::clone(&keeper), Vec::new())
            .expect("composed sealed snapshot");
        let checkpoint: QueryCheckpointHandle<'_> =
            sealed
                .reader
                .query_checkpoint(&cx, "arrival_path_lowering", u64::MAX, u64::MAX);

        reset_term_field_ord_bridge_calls();
        let sealed_scorer = lower_leaf_term(
            QueryLeaf::Sealed(&keeper.segments()[0]),
            &composed,
            DEFAULT_SCHEMA,
            CONTENT_FIELD,
            b"alpha",
            None,
            1.0,
            true,
            &checkpoint,
        )
        .expect("shipping sealed lowering");
        assert!(
            sealed_scorer.arrival_path_is_validated(),
            "the shipping sealed lowering must reach the validated arrival path, \
             or the seeking fill is unreachable in production"
        );
        assert_eq!(
            term_field_ord_bridge_calls(),
            0,
            "shipping sealed lowering must validate DOCLEN directly from the schema iterator",
        );

        let keeper =
            Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        apply_alpha_delta(&mut delta, 0, "one", 1);
        let frozen = Arc::new(delta.freeze(generation));
        let composite = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
            .expect("composite snapshot");
        let delta_checkpoint: QueryCheckpointHandle<'_> =
            sealed
                .reader
                .query_checkpoint(&cx, "arrival_path_lowering_delta", u64::MAX, u64::MAX);
        let delta_scorer = lower_leaf_term(
            QueryLeaf::Delta(&frozen),
            &composite,
            DEFAULT_SCHEMA,
            CONTENT_FIELD,
            b"alpha",
            None,
            1.0,
            true,
            &delta_checkpoint,
        )
        .expect("shipping Delta lowering");
        assert!(
            !delta_scorer.arrival_path_is_validated(),
            "Delta cannot answer arrival reads and must stay on the opaque path"
        );
    });
}

#[test]
fn delta_and_sealed_term_cursors_have_rank_exact_corpus_parity() {
    run_with_cx(|cx| async move {
        let keeper =
            Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        for (docid, document_id, frequency) in [(0, "one", 1), (1, "three", 3), (2, "two", 2)] {
            apply_alpha_delta(&mut delta, docid, document_id, frequency);
        }
        let frozen = Arc::new(delta.freeze(generation));
        let composite = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
            .expect("pre-seal composite snapshot");
        let mut pre = lower_delta_term(&frozen, &composite, CONTENT_FIELD, b"alpha", 1.0)
            .expect("pre-seal Delta scorer");

        let sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
        sealed
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("one", "alpha"),
                    IndexableDocument::new("three", "alpha alpha alpha"),
                    IndexableDocument::new("two", "alpha alpha"),
                ],
            )
            .await
            .expect("accumulate equivalent corpus");
        sealed.commit(&cx).await.expect("seal equivalent corpus");
        let sealed_snapshot = sealed
            .snapshot()
            .expect("sealed cursor snapshot is authoritative");
        let mut post = lower_term(
            &sealed_snapshot.segments()[0],
            &sealed_snapshot,
            DEFAULT_SCHEMA,
            CONTENT_FIELD,
            b"alpha",
            1.0,
        )
        .expect("post-seal scorer");

        let collect_rows = |scorer: &mut ReferenceScorer<'_>| {
            let mut rows = Vec::new();
            while let Some(docid) = scorer.doc() {
                rows.push((docid, scorer.score().expect("score current row").to_bits()));
                scorer.next().expect("advance scorer");
            }
            rows
        };
        let pre_rows = collect_rows(&mut pre);
        let post_rows = collect_rows(&mut post);
        assert_eq!(
            pre_rows, post_rows,
            "per-doc score bits must survive sealing"
        );

        let mut pre_rank = pre_rows;
        let mut post_rank = post_rows;
        let best_first = |left: &(u32, u32), right: &(u32, u32)| {
            f32::from_bits(right.1)
                .total_cmp(&f32::from_bits(left.1))
                .then_with(|| left.0.cmp(&right.0))
        };
        pre_rank.sort_by(best_first);
        post_rank.sort_by(best_first);
        assert_eq!(pre_rank, post_rank, "RankExact order must survive sealing");
    });
}

#[test]
fn mixed_sealed_and_delta_leaves_share_composite_bm25_statistics() {
    run_with_cx(|cx| async move {
        let mixed = QuillIndex::in_memory(deterministic_config()).expect("mixed index");
        mixed
            .index_documents(&cx, &[IndexableDocument::new("sealed", "alpha")])
            .await
            .expect("accumulate sealed row");
        mixed.commit(&cx).await.expect("commit sealed row");
        let keeper = mixed
            .snapshot()
            .expect("mixed fixture snapshot is authoritative");
        let generation = keeper.loaded_manifest().manifest.generation;
        let delta_base = u64::from(DOC_ORDS_PER_LEASE);
        let delta_docid = u32::try_from(delta_base).expect("second lease fits u32");
        let mut delta =
            DeltaSegment::new(DEFAULT_SCHEMA, delta_base, usize::MAX).expect("Delta shard");
        apply_alpha_delta(&mut delta, delta_docid, "delta", 1);
        let frozen = Arc::new(delta.freeze(generation));
        mixed
            .publish_delta_table(vec![Arc::clone(&frozen)])
            .expect("publish mixed Delta leaf");
        let composite =
            QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::clone(&frozen)])
                .expect("mixed composite snapshot");
        assert_eq!(composite.bm25_doc_count(), 2);
        assert_eq!(
            composite
                .bm25_field_stats(CONTENT_FIELD)
                .expect("content stats")
                .total_tokens,
            2
        );
        assert_eq!(
            composite
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("composite alpha df"),
            2
        );

        let mut mixed_sealed = lower_composite_sealed_term(
            &keeper.segments()[0],
            &composite,
            DEFAULT_SCHEMA,
            CONTENT_FIELD,
            b"alpha",
            1.0,
        )
        .expect("mixed sealed scorer");
        let mixed_sealed_score = mixed_sealed.score().expect("score sealed leaf");
        let mut mixed_delta = lower_delta_term(&frozen, &composite, CONTENT_FIELD, b"alpha", 1.0)
            .expect("mixed Delta scorer");
        let mixed_delta_score = mixed_delta.score().expect("score Delta leaf");

        let all_sealed = QuillIndex::in_memory(deterministic_config()).expect("all-sealed index");
        all_sealed
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("sealed", "alpha"),
                    IndexableDocument::new("delta", "alpha"),
                ],
            )
            .await
            .expect("accumulate equivalent all-sealed rows");
        all_sealed
            .commit(&cx)
            .await
            .expect("commit equivalent all-sealed rows");
        let all_sealed_snapshot = all_sealed
            .snapshot()
            .expect("all-sealed fixture snapshot is authoritative");
        let mut oracle = lower_term(
            &all_sealed_snapshot.segments()[0],
            &all_sealed_snapshot,
            DEFAULT_SCHEMA,
            CONTENT_FIELD,
            b"alpha",
            1.0,
        )
        .expect("all-sealed oracle scorer");
        assert_eq!(
            mixed_sealed_score.to_bits(),
            oracle.score().expect("score first oracle row").to_bits()
        );
        assert_eq!(oracle.next().expect("advance oracle"), Some(1));
        assert_eq!(
            mixed_delta_score.to_bits(),
            oracle.score().expect("score second oracle row").to_bits()
        );
        assert_eq!(mixed_delta_score.to_bits(), mixed_sealed_score.to_bits());

        let mixed_results = mixed
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("execute public mixed query");
        let oracle_results = all_sealed
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("execute public all-sealed oracle");
        let mixed_rows = mixed_results
            .hits
            .iter()
            .map(|hit| (hit.document_id.as_str(), hit.score.to_bits()))
            .collect::<Vec<_>>();
        let oracle_rows = oracle_results
            .hits
            .iter()
            .map(|hit| (hit.document_id.as_str(), hit.score.to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(mixed_rows, oracle_rows);
        assert_eq!(mixed_results.total_count, oracle_results.total_count);
        assert_eq!(mixed_results.doc_count, oracle_results.doc_count);
        assert_eq!(
            mixed
                .search_paginated(&cx, "alpha", 1, 1, true)
                .expect("paginate mixed tie")
                .hits[0]
                .document_id,
            "delta",
            "global docid ascending must break equal-score cross-residency ties before offset"
        );
        assert_eq!(
            mixed.collect_docids(&cx, "alpha").expect("mixed docset"),
            vec![0, delta_docid]
        );
    });
}

#[test]
fn rank_pruning_gate_matches_runtime_union_capabilities() {
    let parser = DefaultQueryParser::new(DEFAULT_SCHEMA).expect("bind shipping parser");
    let direct_two = parser.parse("content:alpha OR content:beta");
    assert!(
        query_has_prunable_root_union(&direct_two.query, 1.0),
        "two direct term children are MaxScore-capable"
    );

    let default_two = parser.parse("alpha OR beta");
    // Structural mirror (bd-5o5z8): unfielded terms keep their per-term
    // expansion groups, so this root classifies TermGroups; grouped
    // MaxScore is gated on GROUPED_MAX_SCORE_ENABLED (bd-2zsx6) and the
    // gate must agree with the runtime either way.
    assert_eq!(
        query_has_prunable_root_union(&default_two.query, 1.0),
        GROUPED_MAX_SCORE_ENABLED,
        "grouped default-field roots follow the grouped-MaxScore lever"
    );

    let default_nine =
        parser.parse("alpha OR beta OR gamma OR delta OR epsilon OR zeta OR eta OR theta OR iota");
    assert!(
        !query_has_prunable_root_union(&default_nine.query, 1.0),
        "nine term groups have no grouped BMW: a group owns no physical \
         block-max list, so wide grouped roots stay postings-only"
    );

    assert!(
        matches!(
            prunable_scorer_shape(&default_two.query, 1.0),
            Some(PrunableScorerShape::Union {
                children: 2,
                kind: UnionChildKind::TermGroups,
            })
        ),
        "a default two-word query lowers to two per-term expansion groups"
    );
    assert!(
        matches!(
            prunable_scorer_shape(&direct_two.query, 1.0),
            Some(PrunableScorerShape::Union {
                children: 2,
                kind: UnionChildKind::DirectTerms,
            })
        ),
        "field-scoped children lower to direct terms, not groups"
    );
    let mixed = parser.parse("content:alpha OR beta");
    assert!(
        matches!(
            prunable_scorer_shape(&mixed.query, 1.0),
            Some(PrunableScorerShape::Union {
                children: 2,
                kind: UnionChildKind::Mixed,
            })
        ),
        "a field-scoped term beside a default-term group is a Mixed root"
    );
    assert!(
        !query_has_prunable_root_union(&mixed.query, 1.0),
        "mixed direct/grouped roots have no pruning lever and stay \
         postings-only"
    );

    let direct_nine = parser.parse(
        "content:alpha OR content:beta OR content:gamma OR content:delta OR \
         content:epsilon OR content:zeta OR content:eta OR content:theta OR content:iota",
    );
    assert!(
        query_has_prunable_root_union(&direct_nine.query, 1.0),
        "nine direct sealed term children are BMW-capable"
    );
}

#[test]
fn mixed_snapshot_disjunction_count_free_matches_exhaustive_at_pinned_k() {
    const DOCS_PER_RESIDENCY: u32 = 5_000;
    run_with_cx(|cx| async move {
        let mixed = QuillIndex::in_memory(QuillConfig {
            // This fixture requires one segment spanning more than the
            // 4,096-document union horizon. A wall-clock visibility
            // barrier would make its shape depend on concurrent test load.
            max_visibility_lag_ms: u64::MAX,
            ..deterministic_config()
        })
        .expect("mixed index");
        let mut sealed_documents = Vec::with_capacity(
            usize::try_from(DOCS_PER_RESIDENCY).expect("fixture count fits usize"),
        );
        for ordinal in 0..DOCS_PER_RESIDENCY {
            let content = if ordinal == 4_352 {
                std::iter::repeat_n("alpha", 64)
                    .chain(std::iter::once("beta"))
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                "alpha beta".to_owned()
            };
            let title = if ordinal.is_multiple_of(2) {
                "alpha"
            } else {
                "unrelated"
            };
            sealed_documents.push(
                IndexableDocument::new(format!("sealed-{ordinal:05}"), content).with_title(title),
            );
        }
        for batch in sealed_documents.chunks(PARALLEL_INGEST_MIN_DOCS_PER_SHARD - 1) {
            mixed
                .index_documents(&cx, batch)
                .await
                .expect("accumulate monolithic sealed fixture");
        }
        mixed.commit(&cx).await.expect("seal large fixture");
        let keeper = mixed
            .snapshot()
            .expect("mixed pruning snapshot is authoritative");
        assert_eq!(
            keeper.segments().len(),
            1,
            "mixed-snapshot pruning fixture must remain one monolithic sealed segment"
        );
        assert_eq!(
            keeper
                .segments()
                .first()
                .expect("mixed-snapshot fixture has its asserted segment")
                .at_seal_doc_count(),
            DOCS_PER_RESIDENCY,
            "sealed pruning leaf must span the complete 5,000-document residency"
        );
        let sealed_snapshot = QuillSearchSnapshot::compose(0, Arc::clone(&keeper), Vec::new())
            .expect("sealed-only statistics snapshot");
        let sealed_stats = sealed_snapshot
            .bm25_field_stats(CONTENT_FIELD)
            .expect("sealed content statistics");

        let generation = keeper.loaded_manifest().manifest.generation;
        let delta_base = u64::from(DOC_ORDS_PER_LEASE);
        let mut delta =
            DeltaSegment::new(DEFAULT_SCHEMA, delta_base, usize::MAX).expect("large Delta shard");
        for ordinal in 0..DOCS_PER_RESIDENCY {
            let global_docid = u32::try_from(delta_base + u64::from(ordinal))
                .expect("Delta fixture docid fits u32");
            let content = if ordinal == 4_352 {
                "alpha beta beta beta beta beta beta beta beta beta"
            } else {
                "alpha beta gamma delta epsilon zeta eta theta"
            };
            apply_tokenized_delta_document(
                &mut delta,
                global_docid,
                &format!("delta-{ordinal:05}"),
                content,
            );
        }
        let delta = Arc::new(delta.freeze(generation));
        mixed
            .publish_delta_table(vec![delta])
            .expect("publish large mixed snapshot");
        let live_snapshot = mixed
            .search_snapshot()
            .expect("mixed pruning search snapshot is authoritative");
        let live_stats = live_snapshot
            .bm25_field_stats(CONTENT_FIELD)
            .expect("mixed content statistics");
        assert_ne!(
            sealed_stats
                .average_field_length()
                .expect("sealed avgdl")
                .to_bits(),
            live_stats
                .average_field_length()
                .expect("mixed avgdl")
                .to_bits(),
            "query-time avgdl must differ from the value at block seal time"
        );

        let direct_query = "content:alpha OR content:beta";
        let parsed = DefaultQueryParser::new(DEFAULT_SCHEMA)
            .expect("bind shipping parser")
            .parse(direct_query);
        let segment = live_snapshot
            .keeper_snapshot()
            .segments()
            .first()
            .expect("mixed fixture has one sealed segment");
        assert_eq!(segment.cached_rank_pruning_term_count(), 0);
        let exact_probe = mixed
            .search_paginated(&cx, direct_query, 1, 0, true)
            .expect("execute exact-count cache probe");
        assert_eq!(exact_probe.total_count, Some(10_000));
        assert_eq!(
            segment.cached_rank_pruning_term_count(),
            0,
            "exact counting must not trigger BLOCKMAX validation"
        );
        let direct_checkpoint: QueryCheckpointHandle<'_> =
            QueryCheckpoint::new(&cx, "lower_query_test", u64::MAX, 0);
        let mut direct = lower_query(
            &cx,
            &direct_checkpoint,
            &parsed.query,
            1.0,
            QueryLeaf::Sealed(segment),
            &live_snapshot,
            None,
            DEFAULT_SCHEMA,
            1_024,
            true,
            true,
        )
        .expect("lower direct disjunction with pruning metadata");
        let mut direct_collector = TopDocsCollector::new(1, 0).expect("top-one collector");
        direct_collector
            .collect(&mut direct, segment)
            .expect("collect direct disjunction");
        let (maxscore_windows, bmw_windows) = direct
            .pruning_window_counts()
            .expect("direct disjunction remains a top-level union");
        assert!(
            maxscore_windows > 0,
            "direct disjunction silently fell back"
        );
        assert_eq!(bmw_windows, 0);
        assert_eq!(
            segment.cached_rank_pruning_term_count(),
            2,
            "only content alpha/beta should be cached"
        );

        for limit in [1, 10, 100, 1_000] {
            let count_free = mixed
                .search_paginated(&cx, direct_query, limit, 0, false)
                .expect("execute count-free disjunction");
            // Same-route exhaustive reference (bd-1i4j4): shipping
            // counted invocations now lower the ORDINARY union to match
            // Tantivy's MultiCollector, so the pruning-safety oracle here
            // forces the pre-repair specialized route explicitly.
            let exhaustive = mixed
                .reader
                .execute_ranked_query_inner(
                    &cx,
                    &parsed.query,
                    &mixed
                        .search_snapshot()
                        .expect("mixed pruning search snapshot is authoritative"),
                    limit,
                    0,
                    true,
                    true,
                    Vec::new(),
                    None,
                    None,
                    #[cfg(feature = "pruning-conformance")]
                    None,
                    #[cfg(feature = "profile-internals")]
                    None,
                )
                .expect("execute exhaustive counted disjunction");
            assert_eq!(count_free.total_count, None);
            assert_eq!(
                exhaustive.total_count,
                Some(u64::from(DOCS_PER_RESIDENCY) * 2)
            );
            assert_eq!(count_free.hits.len(), exhaustive.hits.len());
            for (candidate, oracle) in count_free.hits.iter().zip(exhaustive.hits.iter()) {
                assert_eq!(candidate.document_id, oracle.document_id);
                assert_eq!(candidate.global_docid, oracle.global_docid);
                assert_eq!(candidate.score.to_bits(), oracle.score.to_bits());
            }
            assert_eq!(segment.cached_rank_pruning_term_count(), 2);
        }

        let nested_count_free = mixed
            .search_paginated(&cx, "alpha OR beta", 10, 0, false)
            .expect("execute nested count-free fallback");
        let mut nested_parsed = mixed
            .reader
            .default_parser()
            .expect("mixed fixture has a default parser")
            .parse_lenient("alpha OR beta");
        let _ = canonicalize_query(&mut nested_parsed.query);
        let nested_exhaustive = mixed
            .reader
            .execute_ranked_query_inner(
                &cx,
                &nested_parsed.query,
                &mixed
                    .search_snapshot()
                    .expect("mixed pruning search snapshot is authoritative"),
                10,
                0,
                true,
                true,
                Vec::new(),
                None,
                None,
                #[cfg(feature = "pruning-conformance")]
                None,
                #[cfg(feature = "profile-internals")]
                None,
            )
            .expect("execute nested exhaustive oracle");
        assert_eq!(nested_count_free.total_count, None);
        assert_eq!(nested_exhaustive.total_count, Some(10_000));
        assert_eq!(nested_count_free.hits.len(), nested_exhaustive.hits.len());
        for (candidate, oracle) in nested_count_free
            .hits
            .iter()
            .zip(nested_exhaustive.hits.iter())
        {
            assert_eq!(candidate.document_id, oracle.document_id);
            assert_eq!(candidate.global_docid, oracle.global_docid);
            assert_eq!(candidate.score.to_bits(), oracle.score.to_bits());
        }
        assert_eq!(
            segment.cached_rank_pruning_term_count(),
            2,
            "the two per-term expansion groups validate their present \
             members once each (bd-5o5z8); absent title:beta still never \
             validates"
        );
    });
}

const E410_ALGEBRA_DOCS: usize = 48;

const E410_ALGEBRA_VOCAB: [&str; 8] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
];

fn e410_lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

/// Deterministic 48-document corpus over an eight-word vocabulary with
/// varying term frequencies and document lengths, shared by the E4.10
/// public-surface property suites (bd-quill-e4-argus-3ycz.10).
fn e410_algebra_documents() -> Vec<IndexableDocument> {
    let mut state = 0xe410_5eed_u64;
    let mut documents = Vec::with_capacity(E410_ALGEBRA_DOCS);
    for ordinal in 0..E410_ALGEBRA_DOCS {
        let word_count =
            3 + usize::try_from(e410_lcg_next(&mut state) % 8).expect("word count fits usize");
        let mut words = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            words.push(
                E410_ALGEBRA_VOCAB
                    [usize::try_from(e410_lcg_next(&mut state) % 8).expect("pick fits usize")],
            );
        }
        let title = E410_ALGEBRA_VOCAB
            [usize::try_from(e410_lcg_next(&mut state) % 8).expect("title fits usize")];
        documents.push(
            IndexableDocument::new(format!("algebra-{ordinal:02}"), words.join(" "))
                .with_title(title),
        );
    }
    documents
}

/// E4.10 query-algebra property suite at the preparsed public surface
/// (bd-quill-e4-argus-3ycz.10): Should-clause commutativity of the result
/// set and Must-clause associativity hold end-to-end through plan and
/// collect. Laws compare result sets with epsilon-bounded scores because
/// clause order feeds non-associative f32 accumulation. Runs behind
/// `bench-internals` because associativity needs programmatic nesting the
/// shipping string parser cannot construct.
#[cfg(feature = "bench-internals")]
#[test]
fn e410_public_api_algebra_laws_hold() {
    const DOCS: usize = E410_ALGEBRA_DOCS;
    const VOCAB: [&str; 8] = E410_ALGEBRA_VOCAB;
    const SCORE_EPSILON: f32 = 1e-4;
    fn scored_set(result: &QuillSearchResult) -> Vec<(String, f32)> {
        let mut rows = result
            .hits
            .iter()
            .map(|hit| (hit.document_id.clone(), hit.score))
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        rows
    }
    fn assert_epsilon_equal_sets(label: &str, left: &[(String, f32)], right: &[(String, f32)]) {
        assert_eq!(
            left.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            right.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            "{label}: result document sets diverged",
        );
        for ((document_id, left_score), (_, right_score)) in left.iter().zip(right) {
            let drift = (left_score - right_score).abs()
                / left_score.abs().max(right_score.abs()).max(1e-12);
            assert!(
                drift <= SCORE_EPSILON,
                "{label}: {document_id} score drift {left_score} vs {right_score} \
                 breaches ScoreEpsilon",
            );
        }
    }
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("algebra index");
        index
            .index_documents(&cx, &e410_algebra_documents())
            .await
            .expect("accumulate algebra corpus");
        index.commit(&cx).await.expect("seal algebra corpus");

        let content_term = |text: &str| {
            Query::term(
                Vec::from([QueryField::new(CONTENT_FIELD, 1.0)]),
                text.to_owned(),
            )
        };

        let mut nonempty_should = 0_usize;
        let mut nonempty_must = 0_usize;
        let mut seed_state = 0x5eed_e410_u64;
        for seed in 0_u64..16 {
            let mut chosen = Vec::with_capacity(3);
            while chosen.len() < 3 {
                let pick =
                    usize::try_from(e410_lcg_next(&mut seed_state) % 8).expect("pick fits usize");
                if !chosen.contains(&pick) {
                    chosen.push(pick);
                }
            }
            let terms = [VOCAB[chosen[0]], VOCAB[chosen[1]], VOCAB[chosen[2]]];

            let should = |order: [usize; 3]| {
                Query::boolean(
                    order
                        .into_iter()
                        .map(|slot| BooleanClause::new(Occur::Should, content_term(terms[slot])))
                        .collect(),
                    None,
                )
            };
            let baseline = index
                .search_preparsed_paginated(&cx, &should([0, 1, 2]), DOCS, 0, true)
                .expect("baseline Should union");
            let baseline_rows = scored_set(&baseline);
            if !baseline_rows.is_empty() {
                nonempty_should += 1;
            }
            for order in [[2, 0, 1], [1, 2, 0], [2, 1, 0]] {
                let permuted = index
                    .search_preparsed_paginated(&cx, &should(order), DOCS, 0, true)
                    .expect("permuted Should union");
                assert_eq!(
                    permuted.total_count, baseline.total_count,
                    "seed {seed}: Should permutation changed the exact count",
                );
                assert_epsilon_equal_sets(
                    &format!("seed {seed} Should order {order:?}"),
                    &baseline_rows,
                    &scored_set(&permuted),
                );
            }

            let must = |query: Query| BooleanClause::new(Occur::Must, query);
            let flat = Query::boolean(
                Vec::from([
                    must(content_term(terms[0])),
                    must(content_term(terms[1])),
                    must(content_term(terms[2])),
                ]),
                None,
            );
            let right_inner = Query::boolean(
                Vec::from([must(content_term(terms[1])), must(content_term(terms[2]))]),
                None,
            );
            let right_nested = Query::boolean(
                Vec::from([must(content_term(terms[0])), must(right_inner)]),
                None,
            );
            let left_inner = Query::boolean(
                Vec::from([must(content_term(terms[0])), must(content_term(terms[1]))]),
                None,
            );
            let left_nested = Query::boolean(
                Vec::from([must(left_inner), must(content_term(terms[2]))]),
                None,
            );
            let flat_result = index
                .search_preparsed_paginated(&cx, &flat, DOCS, 0, true)
                .expect("flat Must intersection");
            let flat_rows = scored_set(&flat_result);
            if !flat_rows.is_empty() {
                nonempty_must += 1;
            }
            for (shape, query) in [("right-nested", right_nested), ("left-nested", left_nested)] {
                let nested = index
                    .search_preparsed_paginated(&cx, &query, DOCS, 0, true)
                    .expect("nested Must intersection");
                assert_eq!(
                    nested.total_count, flat_result.total_count,
                    "seed {seed}: {shape} Must association changed the exact count",
                );
                assert_epsilon_equal_sets(
                    &format!("seed {seed} Must {shape}"),
                    &flat_rows,
                    &scored_set(&nested),
                );
            }
        }
        assert!(
            nonempty_should >= 8,
            "Should-commutativity draws were mostly vacuous: {nonempty_should}/16",
        );
        assert!(
            nonempty_must >= 1,
            "every Must-associativity draw was vacuous",
        );
    });
}

/// E4.10 pagination metamorphic at the shipping string surface
/// (bd-quill-e4-argus-3ycz.10): page concatenation reproduces the full
/// ranking bit-for-bit on both the nested default lowering and the
/// prunable direct-syntax union, with and without exact counting, and an
/// offset past the last match yields an empty page.
#[test]
fn e410_public_api_pagination_metamorphic_holds() {
    const DOCS: usize = E410_ALGEBRA_DOCS;
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("pagination index");
        index
            .index_documents(&cx, &e410_algebra_documents())
            .await
            .expect("accumulate pagination corpus");
        index.commit(&cx).await.expect("seal pagination corpus");

        for query in ["alpha epsilon", "content:alpha OR content:epsilon"] {
            for exact_count in [true, false] {
                let full = index
                    .search_paginated(&cx, query, DOCS, 0, exact_count)
                    .expect("full ranking");
                let full_rows = full
                    .hits
                    .iter()
                    .map(|hit| {
                        (
                            hit.document_id.clone(),
                            hit.global_docid,
                            hit.score.to_bits(),
                        )
                    })
                    .collect::<Vec<_>>();
                assert!(
                    !full_rows.is_empty(),
                    "pagination fixture query {query:?} must match documents",
                );
                for page_size in [1_usize, 7] {
                    let mut paged_rows = Vec::with_capacity(full_rows.len());
                    let mut offset = 0_usize;
                    loop {
                        let page = index
                            .search_paginated(&cx, query, page_size, offset, exact_count)
                            .expect("ranking page");
                        if exact_count {
                            assert_eq!(
                                page.total_count, full.total_count,
                                "{query:?}: page at offset {offset} changed the exact count",
                            );
                        } else {
                            assert_eq!(page.total_count, None);
                        }
                        if page.hits.is_empty() {
                            break;
                        }
                        paged_rows.extend(page.hits.iter().map(|hit| {
                            (
                                hit.document_id.clone(),
                                hit.global_docid,
                                hit.score.to_bits(),
                            )
                        }));
                        offset += page_size;
                    }
                    assert_eq!(
                        paged_rows, full_rows,
                        "{query:?} page-size {page_size} exact_count {exact_count}: \
                         page concatenation must reproduce the full ranking bit-for-bit",
                    );
                    let overshoot = index
                        .search_paginated(&cx, query, page_size, full_rows.len(), exact_count)
                        .expect("overshoot page");
                    assert!(
                        overshoot.hits.is_empty(),
                        "{query:?}: offset past the last match must yield an empty page",
                    );
                }
            }
        }
    });
}

/// bd-quill-rejected-ingest-publishes-partial-batch-aihri: a rejected
/// scalar ingest is a NO-OP, on every ingest route.
///
/// The defect this pins was route-dependent, which is why the assertions
/// run over both routes from one body: the serial route checked duplicates
/// per document INSIDE its accumulation loop, so the unique prefix of a
/// rejected batch stayed staged and the next `commit()` published it, while
/// the wider routes validated the whole batch up front and did not. A fix
/// applied to one route only leaves this test red.
///
/// Three things must hold after a rejected batch, and each fails
/// differently if partial staging comes back:
///   1. the next `commit()` publishes nothing from the rejected batch;
///   2. the rejected batch's unique prefix is not searchable;
///   3. the prefix id is not CLAIMED — re-ingesting it later succeeds,
///      which fails if `uncommitted_ids` kept it.
///
/// The fourth property — that rejection does not arm the commit-retry
/// guard — is pinned separately by
/// [`a_rejected_batch_does_not_arm_the_commit_retry_guard`], because a
/// single test that can fail for two unrelated reasons reports the wrong
/// one: the guard fires before any commit and would mask the publication
/// evidence entirely.
#[test]
fn a_rejected_batch_publishes_nothing_on_the_next_commit_on_every_route() {
    run_with_cx(|cx| async move {
        // (serial route, fan-out-eligible route): `deterministic_ingest`
        // resolves the router to one shard and forces the serial path;
        // the wider config lets the planner take a parallel route. The
        // batch size is what makes the second one eligible.
        for (label, config, filler_count) in [
            ("serial", deterministic_config(), 0_usize),
            (
                "wide",
                QuillConfig {
                    deterministic_ingest: false,
                    max_ingest_shards: 4,
                    ..QuillConfig::default()
                },
                600,
            ),
        ] {
            let index = QuillIndex::in_memory(config).expect("atomic-ingest index");
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("aihri-live", "alpha original")],
                )
                .await
                .unwrap_or_else(|error| panic!("{label}: seed ingest: {error}")); // ubs:ignore -- Failed seed ingest must fail test setup.
            index
                .commit(&cx)
                .await
                .unwrap_or_else(|error| panic!("{label}: seed commit: {error}")); // ubs:ignore -- Failed seed commit must fail test setup.
            assert_eq!(
                index.doc_count().expect("seed count is authoritative"),
                1,
                "{label}: seed"
            );

            // A batch whose LAST document duplicates a committed id. Every
            // document before it is unique and would be accepted one at a
            // time -- that prefix is exactly what must not survive.
            let mut rejected = Vec::with_capacity(filler_count + 2);
            rejected.push(IndexableDocument::new(
                "aihri-prefix",
                "prefix beta searchable",
            ));
            for ordinal in 0..filler_count {
                rejected.push(IndexableDocument::new(
                    format!("aihri-filler-{ordinal}"),
                    "filler beta searchable",
                ));
            }
            rejected.push(IndexableDocument::new("aihri-live", "gamma duplicate"));

            let error = index
                .index_documents(&cx, &rejected)
                .await
                .expect_err("duplicate id must reject the batch");
            assert!(
                error.to_string().contains("duplicate live document id"),
                "{label}: unexpected rejection: {error}",
            );

            index
                .commit(&cx)
                .await
                .unwrap_or_else(|error| panic!("{label}: commit after rejection: {error}")); // ubs:ignore -- Failed recovery commit must fail this test.

            // 1. nothing from the rejected batch was published.
            assert_eq!(
                index
                    .doc_count()
                    .expect("rejected-batch count is authoritative"),
                1,
                "{label}: the rejected batch published fragments",
            );
            // 2. its unique prefix is not searchable.
            assert_eq!(
                index
                    .search_paginated(&cx, "beta", 10, 0, true)
                    .unwrap_or_else(|error| panic!("{label}: search beta: {error}")) // ubs:ignore -- Failed query must not count as an empty result.
                    .hits
                    .len(),
                0,
                "{label}: a fragment of the rejected batch is searchable",
            );
            // 3. the prefix id was not claimed by the rejected batch.
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("aihri-prefix", "epsilon reingested")],
                )
                .await
                .unwrap_or_else(|error| {
                    panic!("{label}: rejected id must not stay claimed: {error}") // ubs:ignore -- A leaked identity claim must fail this test.
                });
            index
                .commit(&cx)
                .await
                .unwrap_or_else(|error| panic!("{label}: reingest commit: {error}")); // ubs:ignore -- Failed reingest commit must fail this test.
            assert_eq!(
                index
                    .doc_count()
                    .expect("reingested count is authoritative"),
                2,
                "{label}: reingest"
            );
        }
    });
}

/// bd-quill-rejected-ingest-publishes-partial-batch-aihri, second property:
/// a rejected batch does not arm the commit-retry guard.
///
/// This is the caller-visible half of "all-or-nothing". If a rejection
/// leaves nothing staged, there is nothing to reconcile, so forcing a
/// commit before the caller may retry a corrected batch is a spurious
/// obligation — and one that USED to be load-bearing, because the commit
/// it forced was the very commit that published the rejected prefix.
#[test]
fn a_rejected_batch_does_not_arm_the_commit_retry_guard() {
    run_with_cx(|cx| async move {
        for (label, config, filler_count) in [
            ("serial", deterministic_config(), 0_usize),
            (
                "wide",
                QuillConfig {
                    deterministic_ingest: false,
                    max_ingest_shards: 4,
                    ..QuillConfig::default()
                },
                600,
            ),
        ] {
            let index = QuillIndex::in_memory(config).expect("retry-guard index");
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("aihri-guard-live", "alpha original")],
                )
                .await
                .unwrap_or_else(|error| panic!("{label}: seed ingest: {error}")); // ubs:ignore -- Failed seed ingest must fail test setup.
            index
                .commit(&cx)
                .await
                .unwrap_or_else(|error| panic!("{label}: seed commit: {error}")); // ubs:ignore -- Failed seed commit must fail test setup.

            let mut rejected = Vec::with_capacity(filler_count + 2);
            rejected.push(IndexableDocument::new(
                "aihri-guard-prefix",
                "prefix beta searchable",
            ));
            for ordinal in 0..filler_count {
                rejected.push(IndexableDocument::new(
                    format!("aihri-guard-filler-{ordinal}"),
                    "filler beta searchable",
                ));
            }
            rejected.push(IndexableDocument::new(
                "aihri-guard-live",
                "gamma duplicate",
            ));
            index
                .index_documents(&cx, &rejected)
                .await
                .expect_err("duplicate id must reject the batch");

            // No intervening commit: the retry guard, if armed, refuses here.
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("aihri-guard-after", "delta ok")],
                )
                .await
                .unwrap_or_else(|error| {
                    panic!("{label}: a rejected batch must not block the next ingest: {error}") // ubs:ignore -- A blocked successor must fail this test.
                });
            index
                .commit(&cx)
                .await
                .unwrap_or_else(|error| panic!("{label}: commit: {error}")); // ubs:ignore -- Failed commit must fail this positive test.
            assert_eq!(
                index
                    .doc_count()
                    .expect("retry-guard count is authoritative"),
                2,
                "{label}: seed plus the accepted batch"
            );
        }
    });
}

/// Scalar ingestion refuses duplicate live document ids, both in-batch
/// and against the committed snapshot: the Delta lane owns logical
/// upserts, so the scalar lane pins the reject posture that the lexical
/// engine's upsert-replaces semantics intentionally diverge from
/// (bd-quill-e4-argus-3ycz.10).
#[test]
fn e410_scalar_duplicate_document_id_is_rejected() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("dup index");
        index
            .index_documents(&cx, &[IndexableDocument::new("dup-doc", "alpha original")])
            .await
            .expect("first ingest");
        let in_batch = index
            .index_documents(
                &cx,
                &[IndexableDocument::new("dup-doc", "beta replacement")],
            )
            .await
            .expect_err("uncommitted duplicate id must be rejected");
        assert!(
            in_batch.to_string().contains("duplicate live document id"),
            "unexpected in-batch duplicate error: {in_batch}",
        );
        index.commit(&cx).await.expect("seal original document");
        let committed = index
            .index_documents(
                &cx,
                &[IndexableDocument::new("dup-doc", "gamma replacement")],
            )
            .await
            .expect_err("committed duplicate id must be rejected");
        assert!(
            committed.to_string().contains("duplicate live document id"),
            "unexpected committed duplicate error: {committed}",
        );
        let original = index
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("search the surviving document");
        assert_eq!(original.hits.len(), 1);
        assert_eq!(original.hits[0].document_id, "dup-doc");
        assert_eq!(
            index
                .doc_count()
                .expect("duplicate-rejection count is authoritative"),
            1,
            "rejected ingests must not change doc_count",
        );
    });
}

/// A document indexed without metadata must materialize as absent, not as
/// an empty JSON object. Ingest always writes a canonical metadata object,
/// so the empty map round-trips as `{}` unless the materialization
/// boundary folds it away — while the Tantivy oracle omits the field
/// entirely. `IndexableDocument::new` starts with an empty map, so leaving
/// `Some({})` in place flips `metadata.is_some()` on the majority of
/// documents at the default-engine flip (fusion winner re-attachment,
/// fsfs JSON output) and allocates a per-hit `Arc` carrying no
/// information (bd-quill-e4-argus-3ycz.10).
#[test]
fn e410_empty_metadata_materializes_as_absent_not_empty_object() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("metadata index");
        index
            .index_documents(
                &cx,
                &[
                    IndexableDocument::new("doc-bare", "alpha bare document"),
                    IndexableDocument::new("doc-meta", "alpha annotated document")
                        .with_metadata("lang", "rust"),
                ],
            )
            .await
            .expect("ingest mixed-metadata batch");
        index.commit(&cx).await.expect("seal mixed-metadata batch");
        let annotated = serde_json::json!({ "lang": "rust" });

        let hydrated = index
            .search_results(&cx, "alpha", 10)
            .expect("search the hydrated envelope");
        assert_eq!(hydrated.len(), 2, "both documents match the shared term");
        for result in &hydrated {
            match result.doc_id.as_str() {
                "doc-bare" => assert!(
                    result.metadata.is_none(),
                    "empty stored metadata must materialize as absent, got {:?}",
                    result.metadata,
                ),
                "doc-meta" => assert_eq!(
                    result.metadata.as_deref(),
                    Some(&annotated),
                    "present metadata must survive the same boundary",
                ),
                other => panic!("unexpected document id {other}"), // ubs:ignore -- Wrong identity must fail this hydration test.
            }
        }

        // The deferred-fusion lane re-attaches winners' metadata through
        // the same boundary, so hydration must agree hit for hit.
        let batch = LexicalRead::search_candidates(&index, &cx, "alpha", 10)
            .await
            .expect("deferred fusion candidates");
        let (mut candidates, pin) = batch.into_parts();
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.metadata.is_none()),
            "deferred candidates carry no metadata before hydration",
        );
        LexicalRead::hydrate_candidates(&index, &cx, pin.as_ref(), &mut candidates)
            .await
            .expect("hydrate fusion winners");
        for candidate in &candidates {
            match candidate.doc_id.as_str() {
                "doc-bare" => assert!(
                    candidate.metadata.is_none(),
                    "hydration must not resurrect an empty metadata object",
                ),
                "doc-meta" => assert_eq!(
                    candidate.metadata.as_deref(),
                    Some(&annotated),
                    "hydration must restore the annotated metadata verbatim",
                ),
                other => panic!("unexpected document id {other}"), // ubs:ignore -- Wrong identity must fail this hydration test.
            }
        }
    });
}
