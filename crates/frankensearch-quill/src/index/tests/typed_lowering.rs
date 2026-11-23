#[cfg(feature = "bench-internals")]
use crate::keeper::KeeperWriter;

use super::*;

fn apply_typed_delta_document(
    delta: &mut DeltaSegment,
    global_docid: u32,
    document_id: &str,
    content: &str,
    rank: u64,
) {
    let mut term_positions = BTreeMap::<&str, Vec<u32>>::new();
    for (position, term) in content.split_ascii_whitespace().enumerate() {
        term_positions.entry(term).or_default().push(
            u32::try_from(position).expect("typed fixture position fits the Delta wire type"),
        );
    }
    let token_count = u32::try_from(term_positions.values().map(Vec::len).sum::<usize>())
        .expect("typed fixture token count fits u32");
    let fieldnorms = [
        DeltaFieldNorm {
            field_ord: 0,
            raw_length: 1,
            fieldnorm_id: fieldnorm_to_id(1),
        },
        DeltaFieldNorm {
            field_ord: 1,
            raw_length: token_count,
            fieldnorm_id: fieldnorm_to_id(token_count),
        },
        DeltaFieldNorm {
            field_ord: 4,
            raw_length: 0,
            fieldnorm_id: fieldnorm_to_id(0),
        },
    ];
    let mut postings = Vec::with_capacity(term_positions.len() + 1);
    postings.push(DeltaTermPosting {
        field_ord: 0,
        term: document_id.as_bytes(),
        frequency: 1,
        positions: None,
    });
    for (term, positions) in &term_positions {
        postings.push(DeltaTermPosting {
            field_ord: 1,
            term: term.as_bytes(),
            frequency: u32::try_from(positions.len()).expect("typed fixture frequency fits u32"),
            positions: Some(positions),
        });
    }
    let stored = [
        DeltaStoredValue::new(0, document_id.as_bytes()),
        DeltaStoredValue::new(1, content.as_bytes()),
        DeltaStoredValue::new(3, b""),
    ];
    delta
        .apply_document_with_values(
            global_docid,
            DocId::from(document_id),
            shipping_content_hash(document_id, content),
            &fieldnorms,
            &postings,
            &[
                DeltaNumericValue::u64(2, rank),
                DeltaNumericValue::u64(5, rank),
            ],
            &stored,
        )
        .expect("apply typed lowering fixture document");
}

fn typed_delta_snapshot(
    lease_base: u64,
    global_docid: u32,
    document_id: &str,
    content: &str,
    rank: u64,
) -> Arc<DeltaSnapshot> {
    let mut delta = DeltaSegment::new(DELTA_PARITY_SCHEMA, lease_base, usize::MAX)
        .expect("typed lowering Delta lease");
    apply_typed_delta_document(&mut delta, global_docid, document_id, content, rank);
    Arc::new(delta.freeze(0))
}

fn typed_residency_index(
    sealed: &[Arc<DeltaSnapshot>],
    deltas: &[Arc<DeltaSnapshot>],
    config: QuillConfig,
) -> QuillIndex {
    let genesis =
        KeeperSnapshot::in_memory(DELTA_PARITY_SCHEMA).expect("typed lowering genesis Keeper");
    let keeper = if sealed.is_empty() {
        genesis
    } else {
        let mut manifest = genesis
            .next_manifest()
            .expect("typed lowering successor MANIFEST");
        let mut encoded = Vec::new();
        for (index, delta) in sealed.iter().enumerate() {
            let segment = flush_delta_snapshot(
                delta,
                DeltaFlushInput {
                    segment_id: 0xe5_5000
                        + u64::try_from(index).expect("typed segment index fits u64"),
                    created_unix_s: 0,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .expect("flush typed lowering Delta")
            .expect("typed lowering Delta is non-empty");
            manifest.segments.push(manifest_segment(
                &segment,
                u64::try_from(index + 1).expect("typed seal sequence fits u64"),
            ));
            manifest.docid_high_watermark = manifest.docid_high_watermark.max(delta.lease_end());
            encoded.push(segment);
        }
        manifest.field_stats = DELTA_PARITY_SCHEMA
            .fields
            .iter()
            .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
            .map(|field| ManifestFieldStats {
                field_ord: field.id,
                total_tokens: sealed
                    .iter()
                    .map(|delta| {
                        delta
                            .live_total_tokens(field.id)
                            .expect("typed Delta field statistics")
                    })
                    .sum(),
                doc_count: sealed
                    .iter()
                    .map(|delta| {
                        u32::try_from(delta.live_document_count())
                            .expect("typed Delta count fits u32")
                    })
                    .sum(),
            })
            .collect();
        genesis
            .publish_owned_segments(&manifest, encoded)
            .expect("publish typed lowering Keeper")
    };
    let generation = keeper.loaded_manifest().manifest.generation;
    let rebound = deltas
        .iter()
        .map(|delta| Arc::new(delta.rebind_keeper_generation(generation)))
        .collect::<Vec<_>>();
    let index = QuillIndex::from_backend(IndexBackend::Memory(keeper), DELTA_PARITY_SCHEMA, config)
        .expect("bind typed lowering index");
    index
        .publish_delta_table(rebound)
        .expect("publish typed lowering Delta table");
    index
}

fn execute_typed_query(
    index: &QuillIndex,
    cx: &Cx,
    query: &Query,
) -> (QuillSearchResult, Vec<u32>) {
    let snapshot = index
        .search_snapshot()
        .expect("typed-query snapshot is authoritative");
    let ranked = index
        .execute_ranked_query(cx, query, &snapshot, 10, 0, true, Vec::new())
        .expect("execute typed ranked query");
    let docids = index
        .execute_docid_query(cx, query, &snapshot)
        .expect("execute typed doc-set query");
    (ranked, docids)
}

#[test]
fn two_arm_variants_glob_recognizes_only_the_exact_arm_shape() {
    use crate::query::QueryField;

    // The CASS grammar's two-arm shape: Term over a field set that
    // includes the glob's single field, glob pattern = the term plus one
    // trailing star, explicit Or, both clauses Should, Term first.
    let canonical_term = Query::term(
        Vec::from([
            QueryField::new(6, 1.0),
            QueryField::new(7, 1.0),
            QueryField::new(8, 1.0),
            QueryField::new(9, 1.0),
        ]),
        "error".to_owned(),
    );
    let canonical_glob = Query::glob(Vec::from([7]), "error*".to_owned());
    let arena = Query::boolean(
        Vec::from([
            BooleanClause::new(Occur::Should, canonical_term),
            BooleanClause::new(Occur::Should, canonical_glob),
        ]),
        Some(BooleanOperator::Or),
    );
    let QueryNode::Boolean { clauses, operator } = arena.node(arena.root_id()) else {
        panic!("the fixture must build a Boolean root");
    };
    assert_eq!(
        two_arm_variants_glob(clauses, &arena, *operator),
        Some(clauses[1].query),
        "the canonical two-arm shape must route its glob child through the literal drop"
    );

    // Every near-miss keeps its literal: the detector only fires when
    // the exact arm's coverage is guaranteed by the shape itself.
    let term_over = |field_ids: &[u16], text: &str| {
        Query::term(
            field_ids
                .iter()
                .copied()
                .map(|field_id| QueryField::new(field_id, 1.0))
                .collect(),
            text.to_owned(),
        )
    };
    let glob_over =
        |field_ids: &[u16], pattern: &str| Query::glob(field_ids.to_vec(), pattern.to_owned());
    let near_misses = [
        // Explicit And, and no explicit operator at all.
        (
            term_over(&[6, 7], "error"),
            glob_over(&[7], "error*"),
            Some(BooleanOperator::And),
            Occur::Should,
        ),
        (
            term_over(&[6, 7], "error"),
            glob_over(&[7], "error*"),
            None,
            Occur::Should,
        ),
        // Required or forbidden Term arm.
        (
            term_over(&[6, 7], "error"),
            glob_over(&[7], "error*"),
            Some(BooleanOperator::Or),
            Occur::Must,
        ),
        (
            term_over(&[6, 7], "error"),
            glob_over(&[7], "error*"),
            Some(BooleanOperator::Or),
            Occur::MustNot,
        ),
        // Term arm that skips the glob's field.
        (
            term_over(&[6], "error"),
            glob_over(&[7], "error*"),
            Some(BooleanOperator::Or),
            Occur::Should,
        ),
        // Multi-field glob.
        (
            term_over(&[6, 7], "error"),
            glob_over(&[7, 6], "error*"),
            Some(BooleanOperator::Or),
            Occur::Should,
        ),
        // Pattern is not the term plus one trailing star.
        (
            term_over(&[6, 7], "error"),
            glob_over(&[7], "errors*"),
            Some(BooleanOperator::Or),
            Occur::Should,
        ),
        (
            term_over(&[6, 7], "error"),
            glob_over(&[7], "err*r"),
            Some(BooleanOperator::Or),
            Occur::Should,
        ),
        (
            term_over(&[6, 7], "error"),
            glob_over(&[7], "error"),
            Some(BooleanOperator::Or),
            Occur::Should,
        ),
    ];
    for (term, glob, operator, occur) in near_misses {
        let arena = Query::boolean(
            Vec::from([
                BooleanClause::new(occur, term),
                BooleanClause::new(Occur::Should, glob),
            ]),
            operator,
        );
        let QueryNode::Boolean { clauses, operator } = arena.node(arena.root_id()) else {
            panic!("the fixture must build a Boolean root");
        };
        assert_eq!(
            two_arm_variants_glob(clauses, &arena, *operator),
            None,
            "a shape without the exact-arm guarantee must keep its glob literal"
        );
    }

    // Clause order swapped: glob first must not fire either — the
    // detector pins the grammar's construction order, and a miss is
    // always the conservative plain path.
    let arena = Query::boolean(
        Vec::from([
            BooleanClause::new(Occur::Should, glob_over(&[7], "error*")),
            BooleanClause::new(Occur::Should, term_over(&[6, 7], "error")),
        ]),
        Some(BooleanOperator::Or),
    );
    let QueryNode::Boolean { clauses, operator } = arena.node(arena.root_id()) else {
        panic!("the fixture must build a Boolean root");
    };
    assert_eq!(
        two_arm_variants_glob(clauses, &arena, *operator),
        None,
        "a swapped clause order must keep its glob literal"
    );
}

#[test]
fn range_set_and_glob_lowering_is_identical_across_residency() {
    run_with_cx(|cx| async move {
        let second_base = u64::from(DOC_ORDS_PER_LEASE);
        let second_docid = u32::try_from(second_base).expect("second fixture lease fits u32");
        let first = typed_delta_snapshot(0, 0, "first", "alpha alpine", 11);
        let second = typed_delta_snapshot(second_base, second_docid, "second", "alpine beta", 17);
        let empty = typed_residency_index(&[], &[], deterministic_config());
        let all_delta = typed_residency_index(
            &[],
            &[Arc::clone(&first), Arc::clone(&second)],
            deterministic_config(),
        );
        let mixed = typed_residency_index(
            &[Arc::clone(&first)],
            &[Arc::clone(&second)],
            deterministic_config(),
        );
        let all_sealed = typed_residency_index(
            &[Arc::clone(&first), Arc::clone(&second)],
            &[],
            deterministic_config(),
        );
        assert_eq!(
            all_delta
                .doc_count()
                .expect("all-Delta count is authoritative"),
            2
        );
        assert_eq!(empty.doc_count().expect("empty count is authoritative"), 0);
        assert_eq!(
            mixed.doc_count().expect("mixed count is authoritative"),
            2,
            "public count includes the Delta leaf"
        );
        assert_eq!(
            all_sealed
                .doc_count()
                .expect("all-sealed count is authoritative"),
            2
        );

        let queries = [
            (
                Query::boost(
                    Query::range(
                        2,
                        Bound::Included(QueryValue::U64(11)),
                        Bound::Included(QueryValue::U64(17)),
                    ),
                    2.5,
                ),
                2.5_f32,
            ),
            (
                Query::range(
                    1,
                    Bound::Included(QueryValue::Str("alpha".to_owned())),
                    Bound::Included(QueryValue::Str("alpine".to_owned())),
                ),
                1.0,
            ),
            (
                Query::range(
                    5,
                    Bound::Included(QueryValue::U64(11)),
                    Bound::Included(QueryValue::U64(17)),
                ),
                1.0,
            ),
            (
                Query::set(
                    1,
                    Vec::from([
                        QueryValue::Str("alpha".to_owned()),
                        QueryValue::Str("alpine".to_owned()),
                        QueryValue::Str("alpha".to_owned()),
                    ]),
                ),
                1.0,
            ),
            (
                Query::set(
                    2,
                    Vec::from([
                        QueryValue::U64(11),
                        QueryValue::U64(17),
                        QueryValue::U64(11),
                    ]),
                ),
                1.0,
            ),
            (
                Query::set(
                    5,
                    Vec::from([
                        QueryValue::U64(11),
                        QueryValue::U64(17),
                        QueryValue::U64(11),
                    ]),
                ),
                1.0,
            ),
            (Query::glob(Vec::from([1]), "alp*".to_owned()), 1.0),
            (Query::glob(Vec::from([1, 1]), "alp*".to_owned()), 2.0),
        ];
        for (query, expected_score) in &queries {
            let delta_evidence = execute_typed_query(&all_delta, &cx, query);
            let mixed_evidence = execute_typed_query(&mixed, &cx, query);
            let sealed_evidence = execute_typed_query(&all_sealed, &cx, query);
            assert_eq!(mixed_evidence, delta_evidence, "mixed query {query:?}");
            assert_eq!(sealed_evidence, delta_evidence, "sealed query {query:?}");
            assert_eq!(delta_evidence.1, vec![0, second_docid]);
            assert_eq!(delta_evidence.0.total_count, Some(2));
            assert!(
                delta_evidence
                    .0
                    .hits
                    .iter()
                    .all(|hit| { hit.score.to_bits() == expected_score.to_bits() })
            );
        }

        let phrase_prefix = Query::phrase(
            Vec::from([crate::query::QueryField::new(1, 1.0)]),
            Vec::from([
                crate::query::PositionedTerm::new(0, "alpha"),
                crate::query::PositionedTerm::new(1, "alp"),
            ]),
            0,
            true,
        );
        let delta_evidence = execute_typed_query(&all_delta, &cx, &phrase_prefix);
        let mixed_evidence = execute_typed_query(&mixed, &cx, &phrase_prefix);
        let sealed_evidence = execute_typed_query(&all_sealed, &cx, &phrase_prefix);
        assert_eq!(mixed_evidence, delta_evidence);
        assert_eq!(sealed_evidence, delta_evidence);
        assert_eq!(delta_evidence.1, vec![0]);
        assert_eq!(delta_evidence.0.total_count, Some(1));

        for empty_range in [
            Query::range(
                1,
                Bound::Included(QueryValue::Str("zeta".to_owned())),
                Bound::Included(QueryValue::Str("alpha".to_owned())),
            ),
            Query::range(
                1,
                Bound::Excluded(QueryValue::Str("alpha".to_owned())),
                Bound::Included(QueryValue::Str("alpha".to_owned())),
            ),
        ] {
            let (ranked, docids) = execute_typed_query(&mixed, &cx, &empty_range);
            assert!(ranked.hits.is_empty());
            assert_eq!(ranked.total_count, Some(0));
            assert!(docids.is_empty());
        }

        let invalid_queries = [
            Query::range(
                99,
                Bound::Included(QueryValue::U64(0)),
                Bound::Included(QueryValue::U64(1)),
            ),
            Query::range(
                2,
                Bound::Included(QueryValue::Str("wrong".to_owned())),
                Bound::Unbounded,
            ),
            Query::boost(Query::all(), f32::INFINITY),
            Query::phrase(
                Vec::from([crate::query::QueryField::new(1, 1.0)]),
                Vec::from([
                    crate::query::PositionedTerm::new(1, "alpha"),
                    crate::query::PositionedTerm::new(0, "beta"),
                ]),
                1,
                false,
            ),
            Query::range(
                1,
                Bound::Included(QueryValue::Str(
                    "x".repeat(crate::grimoire::MAX_TERM_BYTES + 1),
                )),
                Bound::Unbounded,
            ),
        ];
        for query in &invalid_queries {
            let errors = [&empty, &all_delta, &mixed, &all_sealed].map(|index| {
                let snapshot = index
                    .search_snapshot()
                    .expect("typed validation snapshot is authoritative");
                let ranked = index
                    .execute_ranked_query(&cx, query, &snapshot, 10, 0, true, Vec::new())
                    .expect_err("invalid ranked query must fail before leaf iteration")
                    .to_string();
                let docset = index
                    .execute_docid_query(&cx, query, &snapshot)
                    .expect_err("invalid docset query must fail before leaf iteration")
                    .to_string();
                assert_eq!(ranked, docset, "collector validation parity for {query:?}");
                ranked
            });
            assert!(
                errors.windows(2).all(|pair| pair[0] == pair[1]),
                "residency-independent validation for {query:?}: {errors:?}"
            );
        }

        let mut bounded = deterministic_config();
        bounded.glob_expansion_limit = 1;
        let bounded = typed_residency_index(&[], &[first, second], bounded);
        let snapshot = bounded
            .search_snapshot()
            .expect("bounded typed snapshot is authoritative");
        let glob = Query::glob(Vec::from([1]), "alp*".to_owned());
        assert!(matches!(
            bounded.execute_ranked_query(&cx, &glob, &snapshot, 10, 0, true, Vec::new()),
            Err(QuillIndexError::Dictionary(
                TermDictionaryError::GlobExpansionLimitExceeded {
                    pattern,
                    field_ord: 1,
                    limit: 1,
                    actual: 2,
                }
            )) if pattern == "alp*"
        ));
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn positionless_string_phrase_fails_with_typed_capability_before_execution() {
    let index = QuillIndex::in_memory_with_schema(POSITIONLESS_QG_SCHEMA, deterministic_config())
        .expect("construct the QG positionless fixture");
    let cx = Cx::for_testing();

    let term_result = index
        .search_paginated(&cx, "alpha", 10, 0, true)
        .expect("position-independent queries remain servable");
    assert!(term_result.hits.is_empty());
    assert_eq!(term_result.total_count, Some(0));

    let assert_positions_error = |error: &QuillIndexError| {
        assert!(matches!(
            error,
            QuillIndexError::QueryCapability(QueryCapabilityError::PositionsRequired {
                schema: "qg-positionless-typed-capability-test",
                field,
                operator: QueryExplanation::Phrase,
                capability: crate::query::IndexCapability::Positions,
            }) if field == "content"
        ));
    };
    for query in ["\"alpha beta\"", "\"alpha beta\"~2", "\"alpha beta\"*"] {
        let ranked = index
            .search_paginated(&cx, query, 10, 0, true)
            .expect_err("a position-dependent phrase must fail before query execution");
        assert_positions_error(&ranked);
        let docset = index
            .collect_docids(&cx, query)
            .expect_err("the scoreless path must preserve the capability error");
        assert_positions_error(&docset);
        let bench_ranked = index
            .bench_search_sealed_forced(&cx, query, 10, false, None)
            .expect_err("the QG ranked harness must reject before sealed collection");
        assert_positions_error(&bench_ranked);
        let bench_docset = index
            .bench_collect_docids_forced(&cx, query, false)
            .expect_err("the QG scoreless harness must reject before sealed collection");
        assert_positions_error(&bench_docset);
    }
}

#[cfg(feature = "bench-internals")]
#[test]
fn positionless_queries_match_positioned_ids_order_and_scores() {
    run_with_cx(|cx| async move {
        let positioned =
            QuillIndex::in_memory_with_schema(POSITIONED_QG_SCHEMA, deterministic_config())
                .expect("construct the positioned control");
        let positionless =
            QuillIndex::in_memory_with_schema(POSITIONLESS_QG_SCHEMA, deterministic_config())
                .expect("construct the positionless fixture");
        let documents = fixture_documents();
        positioned
            .index_documents(&cx, &documents)
            .await
            .expect("index positioned control documents");
        positionless
            .index_documents(&cx, &documents)
            .await
            .expect("index positionless fixture documents");
        positioned
            .commit(&cx)
            .await
            .expect("publish positioned control");
        positionless
            .commit(&cx)
            .await
            .expect("publish positionless fixture");

        for query in [
            "rust",
            "rust OR python",
            "rust AND systems",
            "title:guide",
            "ord:[0 TO 1]",
        ] {
            let control = positioned
                .search_paginated(&cx, query, 10, 0, true)
                .unwrap_or_else(|error| {
                    panic!("positioned control failed for {query:?}: {error}") // ubs:ignore -- Failed live control must fail this test.
                });
            let candidate = positionless
                .search_paginated(&cx, query, 10, 0, true)
                .unwrap_or_else(|error| {
                    panic!("positionless fixture failed for {query:?}: {error}") // ubs:ignore -- Failed candidate must fail this test.
                });
            assert_eq!(
                candidate, control,
                "positions must not affect ids, order, scores, or exact count for {query:?}"
            );
        }

        let phrase = positioned
            .search_paginated(&cx, "\"rust ownership\"", 10, 0, true)
            .expect("positioned phrase behavior remains available");
        assert!(
            phrase.hits.iter().any(|hit| hit.document_id == "rust-1"),
            "the positioned control must exercise a real phrase hit"
        );
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn positionless_basic_scoring_uses_presence_for_repeated_edge_ngrams() {
    run_with_cx(|cx| async move {
        let positioned =
            QuillIndex::in_memory_with_schema(POSITIONED_QG_SCHEMA, deterministic_config())
                .expect("construct the positioned control");
        let positionless =
            QuillIndex::in_memory_with_schema(POSITIONLESS_QG_SCHEMA, deterministic_config())
                .expect("construct the Basic fixture");
        assert!(
            positionless
                .search_paginated(&cx, "token", 10, 0, true)
                .expect("search empty Basic fixture")
                .hits
                .is_empty(),
            "an empty Basic field must have no term hits"
        );
        let repeated = crate::scribe::cass_generate_edge_ngrams("tokens tokens");
        let single = crate::scribe::cass_generate_edge_ngrams("tokens abcdef");
        assert_eq!(
            repeated.split_whitespace().count(),
            single.split_whitespace().count(),
            "control documents must retain equal field lengths"
        );
        let documents = vec![
            IndexableDocument::new("repeat", repeated).with_title("neutral"),
            IndexableDocument::new("single", single).with_title("neutral"),
        ];
        positioned
            .index_documents(&cx, &documents)
            .await
            .expect("index positioned control documents");
        positionless
            .index_documents(&cx, &documents)
            .await
            .expect("index Basic fixture documents");
        let assert_record_semantics =
            |positioned: &QuillSearchResult, positionless: &QuillSearchResult, phase: &str| {
                let score = |result: &QuillSearchResult, document_id: &str| {
                    result
                        .hits
                        .iter()
                        .find(|hit| hit.document_id == document_id)
                        .unwrap_or_else(|| panic!("{phase}: missing {document_id} hit")) // ubs:ignore -- Missing hit must fail this scoring test.
                        .score
                };
                let basic_repeat = score(positionless, "repeat");
                let basic_single = score(positionless, "single");
                let positioned_repeat = score(positioned, "repeat");
                let positioned_single = score(positioned, "single");

                assert_eq!(
                    basic_repeat.to_bits(),
                    basic_single.to_bits(),
                    "{phase}: Basic postings score document presence, not retained raw frequency"
                );
                assert_eq!(
                    basic_single.to_bits(),
                    positioned_single.to_bits(),
                    "{phase}: a unit frequency is unchanged across record options"
                );
                assert!(
                    positioned_repeat > positioned_single,
                    "{phase}: frequency-bearing postings retain the repeated edge n-gram boost"
                );
            };

        positioned
            .commit(&cx)
            .await
            .expect("publish positioned control");
        positionless
            .commit(&cx)
            .await
            .expect("publish Basic fixture");

        let positioned = positioned
            .search_paginated(&cx, "token", 10, 0, true)
            .expect("search positioned control");
        let positionless = positionless
            .search_paginated(&cx, "token", 10, 0, true)
            .expect("search Basic fixture");
        assert_record_semantics(&positioned, &positionless, "sealed");
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn positionless_basic_scoring_survives_multiple_segments_and_reopen() {
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary Basic scoring directory");
        let writer = QuillWriterState::create_with_schema(
            &cx,
            directory.path(),
            POSITIONLESS_QG_SCHEMA,
            deterministic_config(),
        )
        .await
        .expect("create durable Basic writer");
        let index = QuillIndex::from_writer(writer);
        let repeated = crate::scribe::cass_generate_edge_ngrams("tokens tokens");
        let single = crate::scribe::cass_generate_edge_ngrams("tokens abcdef");

        index
            .index_document(
                &cx,
                &IndexableDocument::new("repeat", repeated).with_title("neutral"),
            )
            .await
            .expect("index repeated edge n-gram segment");
        index
            .commit(&cx)
            .await
            .expect("publish first Basic segment");
        index
            .index_document(
                &cx,
                &IndexableDocument::new("single", single).with_title("neutral"),
            )
            .await
            .expect("index single edge n-gram segment");
        index
            .commit(&cx)
            .await
            .expect("publish second Basic segment");
        assert_eq!(
            index
                .snapshot()
                .expect("Basic fixture snapshot is authoritative")
                .segments()
                .len(),
            2,
            "fixture must exercise composite scoring across two sealed segments"
        );

        let before_reopen = index
            .search_paginated(&cx, "token", 10, 0, true)
            .expect("search two-segment Basic fixture");
        assert_eq!(before_reopen.hits.len(), 2);
        assert_eq!(
            before_reopen.hits[0].score.to_bits(),
            before_reopen.hits[1].score.to_bits(),
            "Basic tf=1 must remain exact across sealed segments"
        );

        drop(index);
        let writer = KeeperWriter::open(&cx, directory.path(), POSITIONLESS_QG_SCHEMA)
            .await
            .expect("reopen custom-schema Keeper");
        let reopened = QuillIndex::from_backend(
            IndexBackend::Durable(writer),
            POSITIONLESS_QG_SCHEMA,
            deterministic_config(),
        )
        .expect("bind reopened Basic index");
        let after_reopen = reopened
            .search_paginated(&cx, "token", 10, 0, true)
            .expect("search reopened Basic fixture");
        assert_eq!(
            after_reopen, before_reopen,
            "reopen must preserve Basic ids, order, score bits, and exact count"
        );
    });
}

#[test]
fn default_fast_only_ord_range_and_set_execute_before_leaf_iteration() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let empty_set = index
            .search_paginated(&cx, "ord: IN [0 1]", 10, 0, true)
            .expect("fast-only set is valid against an empty snapshot");
        assert!(empty_set.hits.is_empty());
        assert_eq!(empty_set.total_count, Some(0));
        let empty_range = index
            .search_paginated(&cx, "ord:[0 TO 1]", 10, 0, true)
            .expect("fast-only range is valid against an empty snapshot");
        assert!(empty_range.hits.is_empty());
        assert_eq!(empty_range.total_count, Some(0));

        index
            .index_documents(&cx, &fixture_documents())
            .await
            .expect("accumulate fixture");
        index.commit(&cx).await.expect("publish fixture");

        let cancelled = Cx::for_testing();
        cancelled.set_cancel_requested(true);
        let snapshot = index
            .search_snapshot()
            .expect("fast-only fixture snapshot is authoritative");
        let segment = snapshot
            .keeper_snapshot()
            .segments()
            .first()
            .expect("committed fast-only segment");
        assert!(matches!(
            lower_leaf_numeric_range(
                &cancelled,
                QueryLeaf::Sealed(segment),
                DEFAULT_SCHEMA,
                ORD_FIELD,
                &Bound::Included(QueryValue::U64(0)),
                &Bound::Included(QueryValue::U64(1)),
                1.0,
                QueryLoweringMode::Scored,
            ),
            Err(QuillIndexError::Cancelled {
                phase: "numeric range"
            })
        ));

        let ranked = index
            .search_paginated(&cx, "ord:[0 TO 1]", 10, 0, true)
            .expect("execute default fast-only range");
        assert_eq!(ranked.total_count, Some(2));
        assert_eq!(
            ranked
                .hits
                .iter()
                .map(|hit| (hit.document_id.as_str(), hit.score.to_bits()))
                .collect::<Vec<_>>(),
            vec![("rust-1", 1.0_f32.to_bits()), ("rust-2", 1.0_f32.to_bits())]
        );
        assert_eq!(
            index
                .collect_docids(&cx, "ord:[0 TO 1]")
                .expect("collect fast-only range docids"),
            vec![0, 1]
        );
        assert_eq!(
            index
                .search_paginated(&cx, "ord:{0 TO 2}", 10, 0, true)
                .expect("execute exclusive fast-only range")
                .hits
                .iter()
                .map(|hit| hit.document_id.as_str())
                .collect::<Vec<_>>(),
            vec!["rust-2"]
        );
        assert!(
            index
                .search_paginated(&cx, "ord:[0 TO 1]^2.5", 10, 0, true)
                .expect("execute boosted fast-only range")
                .hits
                .iter()
                .all(|hit| hit.score.to_bits() == 2.5_f32.to_bits())
        );
        let populated_set = index
            .search_paginated(&cx, "ord: IN [0 1]", 10, 0, true)
            .expect("execute default fast-only set");
        assert_eq!(populated_set.total_count, Some(2));
        assert_eq!(
            populated_set
                .hits
                .iter()
                .map(|hit| (hit.document_id.as_str(), hit.score.to_bits()))
                .collect::<Vec<_>>(),
            vec![("rust-1", 1.0_f32.to_bits()), ("rust-2", 1.0_f32.to_bits())]
        );
        assert_eq!(
            index
                .collect_docids(&cx, "ord: IN [0 1]")
                .expect("collect fast-only set docids"),
            vec![0, 1]
        );
        assert!(
            index
                .search_paginated(&cx, "ord: IN [0 1 1]^2.5", 10, 0, true)
                .expect("execute boosted deduplicated fast-only set")
                .hits
                .iter()
                .all(|hit| hit.score.to_bits() == 2.5_f32.to_bits())
        );
    });
}
