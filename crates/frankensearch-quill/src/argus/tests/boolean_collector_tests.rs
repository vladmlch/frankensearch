use super::*;

#[test]
fn bm25_score_uses_raw_average_cached_norm_and_boost_order()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(3)];
    let encoded = EncodedDocLenSection::encode(0, 1, &[7], &[DocLenFieldInput::new(7, &lengths)])?;
    let section = encoded.section(&[7])?;
    let field = section.field(7).expect("field exists");
    let snapshot = snapshot(7, 9, 3)?;
    let mut scorer = term(vec![Posting::new(0, 2)], field, &snapshot, 2, 1, 1.0)?;

    // Spreadsheet row: N=3, n=2, total_tokens=9, dl=3, f=2.
    // idf=0.47000363, norm=1.2, tf=0.625, score=0.646255.
    let score = scorer.score()?;
    assert_eq!(score.to_bits(), 1_059_418_360);
    let expected = expected_term_score(&snapshot, 2, fieldnorm_to_id(3), 2, 1.0);
    assert_eq!(score.to_bits(), expected.to_bits());

    let mut boosted = term(vec![Posting::new(0, 2)], field, &snapshot, 2, 1, 2.0)?;
    let boosted_score = boosted.score()?;
    let expected_boosted = expected_term_score(&snapshot, 2, fieldnorm_to_id(3), 2, 2.0);
    assert_eq!(boosted_score.to_bits(), expected_boosted.to_bits());
    Ok(())
}

#[test]
fn fieldnorm_boundaries_use_the_shared_snapshot_cache() -> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(0), Some(40), Some(42), Some(2_013_265_944)];
    let encoded = EncodedDocLenSection::encode(0, 4, &[3], &[DocLenFieldInput::new(3, &lengths)])?;
    let section = encoded.section(&[3])?;
    let field = section.field(3).expect("field exists");
    let snapshot = snapshot(3, 2_013_266_026, 4)?;
    let mut scorer = term(
        vec![
            Posting::new(0, 1),
            Posting::new(1, 2),
            Posting::new(2, 3),
            Posting::new(3, 4),
        ],
        field,
        &snapshot,
        4,
        4,
        1.0,
    )?;

    for (doc, fieldnorm_id, frequency) in [(0, 0, 1), (1, 40, 2), (2, 41, 3), (3, 255, 4)] {
        assert_eq!(scorer.doc(), Some(doc));
        let score = scorer.score()?;
        let expected = expected_term_score(&snapshot, 4, fieldnorm_id, frequency, 1.0);
        assert_eq!(score.to_bits(), expected.to_bits());
        scorer.next()?;
    }
    assert_eq!(scorer.doc(), None);
    Ok(())
}

#[test]
fn title_boost_wins_and_equal_scores_tie_by_global_docid() -> Result<(), Box<dyn std::error::Error>>
{
    let content_lengths = [Some(1), Some(1), Some(1)];
    let title_lengths = [Some(1), Some(1), Some(1)];
    let inputs = [
        DocLenFieldInput::new(1, &content_lengths),
        DocLenFieldInput::new(2, &title_lengths),
    ];
    let encoded = EncodedDocLenSection::encode(0, 3, &[1, 2], &inputs)?;
    let section = encoded.section(&[1, 2])?;
    let content = section.field(1).expect("content field");
    let title = section.field(2).expect("title field");
    let content_snapshot = snapshot(1, 3, 3)?;
    let title_snapshot = snapshot(2, 3, 3)?;
    let content_term = term(postings(&[0, 2]), content, &content_snapshot, 2, 2, 1.0)?;
    let title_term = term(postings(&[1]), title, &title_snapshot, 1, 1, 2.0)?;
    let mut query = ReferenceScorer::boolean(vec![
        ScorerClause::should(content_term),
        ScorerClause::should(title_term),
    ])?;
    let winners = query.top_k(3, &AllLiveDocs)?;

    assert_eq!(
        winners
            .iter()
            .map(|hit| hit.global_docid)
            .collect::<Vec<_>>(),
        vec![1, 0, 2]
    );
    assert_eq!(winners[1].score.to_bits(), winners[2].score.to_bits());
    assert!(winners[0].score > winners[1].score);
    Ok(())
}

#[test]
fn boolean_occurs_match_tantivy_boundaries() -> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1); 5];
    let encoded = EncodedDocLenSection::encode(0, 5, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 5, 5)?;

    let mut must_with_optional = ReferenceScorer::boolean(vec![
        ScorerClause::must(term(postings(&[1, 2]), field, &snapshot, 2, 2, 1.0)?),
        ScorerClause::should(term(postings(&[2, 3]), field, &snapshot, 2, 2, 1.0)?),
        ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
    ])?;
    let hits = must_with_optional.top_k(5, &AllLiveDocs)?;
    assert_eq!(
        hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
        vec![1]
    );
    let expected_must = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
    assert_eq!(hits[0].score.to_bits(), expected_must.to_bits());

    let mut should_with_exclusion = ReferenceScorer::boolean(vec![
        ScorerClause::should(term(postings(&[2, 3]), field, &snapshot, 2, 2, 1.0)?),
        ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
    ])?;
    let hits = should_with_exclusion.top_k(5, &AllLiveDocs)?;
    assert_eq!(
        hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
        vec![3]
    );

    let mut negative_only = ReferenceScorer::boolean(vec![ScorerClause::must_not(term(
        postings(&[1, 2]),
        field,
        &snapshot,
        2,
        2,
        1.0,
    )?)])?;
    assert!(negative_only.top_k(5, &AllLiveDocs)?.is_empty());

    let mut multiple_negative_only = ReferenceScorer::boolean(vec![
        ScorerClause::must_not(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
        ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
    ])?;
    assert_eq!(multiple_negative_only.segment_num_docs(), 5);
    assert_eq!(multiple_negative_only.size_hint(), 0);
    assert_eq!(multiple_negative_only.cost(), 0);
    assert!(multiple_negative_only.top_k(5, &AllLiveDocs)?.is_empty());
    Ok(())
}

/// Executable Boolean-shape proof for the oversized-query-token class
/// (`bd-quill-e0-contracts-j53p.8`, register entry DIV-004).
///
/// Quill's symmetric admission rule means an oversized (>65,530-byte)
/// query token can never name a dictionary entry, so its lowered leaf is
/// [`ReferenceScorer::empty`]. `boolean_with_mode` then shorts
/// `Must(empty)` to `MatchNone` and drops empty `Should`/`MustNot`
/// clauses. The oracle keeps the same unmatchable leaf as an empty
/// posting list. This proof enumerates every clause list of length 1..=3
/// over {matchable, oversized} x {`Must`, `Should`, `MustNot`} and asserts the
/// two shapes return identical hits with bit-identical scores.
#[test]
fn oversized_token_boolean_shapes_are_result_equivalent() -> Result<(), Box<dyn std::error::Error>>
{
    const DOCS: u32 = 5;
    let lengths = [Some(1); DOCS as usize];
    let encoded = EncodedDocLenSection::encode(
        0,
        u64::from(DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(DOCS), u64::from(DOCS))?;
    // The matchable leaf hits docs {1, 2, 3}; the oversized leaf is
    // unmatchable under every semantics.
    let matchable_docs: [u32; 3] = [1, 2, 3];
    let occurs = [Occur::Must, Occur::Should, Occur::MustNot];
    // (use_quill_empty, occur) -> scorer for one clause
    let build = |use_quill_empty: bool,
                 oversized: bool,
                 occur: Occur|
     -> Result<ScorerClause<'_>, ArgusError> {
        let scorer = if oversized {
            if use_quill_empty {
                ReferenceScorer::empty()
            } else {
                term(postings(&[]), field, &snapshot, 0, 0, 1.0)?
            }
        } else {
            term(postings(&matchable_docs), field, &snapshot, 3, 3, 1.0)?
        };
        Ok(ScorerClause::new(occur, scorer))
    };
    let mut case_count = 0_u32;
    for len in 1..=3_usize {
        let slots = usize::pow(6, u32::try_from(len)?);
        for combo in 0..slots {
            let mut remaining = combo;
            let mut quill_clauses = Vec::with_capacity(len);
            let mut oracle_clauses = Vec::with_capacity(len);
            for _slot in 0..len {
                let oversized = remaining % 2 == 1;
                let occur = occurs[(remaining / 2) % 3];
                remaining /= 6;
                quill_clauses.push(build(true, oversized, occur)?);
                oracle_clauses.push(build(false, oversized, occur)?);
            }
            let mut quill = ReferenceScorer::boolean(quill_clauses)?;
            let mut oracle = ReferenceScorer::boolean(oracle_clauses)?;
            let quill_hits = quill.top_k(usize::try_from(DOCS)?, &AllLiveDocs)?;
            let oracle_hits = oracle.top_k(usize::try_from(DOCS)?, &AllLiveDocs)?;
            assert_eq!(
                quill_hits
                    .iter()
                    .map(|hit| hit.global_docid)
                    .collect::<Vec<_>>(),
                oracle_hits
                    .iter()
                    .map(|hit| hit.global_docid)
                    .collect::<Vec<_>>(),
                "hit sets differ at combo {combo} (len {len})"
            );
            for (quill_hit, oracle_hit) in quill_hits.iter().zip(&oracle_hits) {
                assert_eq!(
                    quill_hit.score.to_bits(),
                    oracle_hit.score.to_bits(),
                    "scores differ for doc {} at combo {combo} (len {len})",
                    quill_hit.global_docid
                );
            }
            case_count += 1;
        }
    }
    assert_eq!(case_count, 6 + 36 + 216);
    Ok(())
}

/// The doc-set collector path (unscored lowering) must preserve the same
/// oversized semantics: empty `Should`/`MustNot` drop, `Must(empty)`
/// shorts to `MatchNone`, and the remaining shapes match identical sets.
#[test]
fn oversized_token_unscored_shapes_are_result_equivalent() -> Result<(), Box<dyn std::error::Error>>
{
    const DOCS: u32 = 5;
    let lengths = [Some(1); DOCS as usize];
    let encoded = EncodedDocLenSection::encode(
        0,
        u64::from(DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(DOCS), u64::from(DOCS))?;
    let matchable_docs: [u32; 3] = [1, 2, 3];
    let occurs = [Occur::Must, Occur::Should, Occur::MustNot];
    for occur in occurs {
        for oversized in [true, false] {
            let make = |use_quill_empty: bool| -> Result<ReferenceScorer<'_>, ArgusError> {
                let scorer = if oversized {
                    if use_quill_empty {
                        ReferenceScorer::empty()
                    } else {
                        term(postings(&[]), field, &snapshot, 0, 0, 1.0)?
                    }
                } else {
                    term(postings(&matchable_docs), field, &snapshot, 3, 3, 1.0)?
                };
                ReferenceScorer::boolean_unscored(vec![ScorerClause::new(occur, scorer)])
            };
            let mut quill = make(true)?;
            let mut oracle = make(false)?;
            assert_eq!(
                quill.collect_doc_set(&AllLiveDocs)?,
                oracle.collect_doc_set(&AllLiveDocs)?,
                "doc sets differ for occur {occur:?} oversized={oversized}"
            );
        }
    }
    Ok(())
}

#[test]
fn required_cost_sort_and_optional_composition_pin_f32_order()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1)];
    let encoded = EncodedDocLenSection::encode(0, 1, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 1, 1)?;
    let boosts = [1.0_f32, 1.0e8, -1.0e8, 2.0];
    let costs = [900_u64, 3, 200, 200];
    let mut clauses = Vec::new();
    for (&boost, &cost) in boosts.iter().zip(&costs) {
        clauses.push(ScorerClause::must(term(
            postings(&[0]),
            field,
            &snapshot,
            1,
            cost,
            boost,
        )?));
    }
    clauses.push(ScorerClause::should(term(
        postings(&[0]),
        field,
        &snapshot,
        1,
        1,
        3.0,
    )?));
    let mut query = ReferenceScorer::boolean(clauses)?;
    let actual = query.score()?;

    let score = |boost| expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, boost);
    // Required order after stable cost sort: boosts 1e8, -1e8, 2, 1.
    // Tantivy's intersection expression is `(left + right) + sum(others)`;
    // RequiredOptional then adds the optional aggregate.
    let required = score(1.0e8) + score(-1.0e8) + (score(2.0) + score(1.0));
    let mut expected = 0.0_f32;
    expected += required;
    expected += score(3.0);
    assert_eq!(actual.to_bits(), expected.to_bits());
    Ok(())
}

#[test]
fn intersection_danger_seek_preserves_nested_union_score_order()
-> Result<(), Box<dyn std::error::Error>> {
    const DOC_COUNT: usize = 30_001;
    let lengths = vec![Some(1); DOC_COUNT];
    let encoded = EncodedDocLenSection::encode(
        0,
        DOC_COUNT as u64,
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, DOC_COUNT as u64, DOC_COUNT as u64)?;

    let optional = ReferenceScorer::boolean(vec![
        ScorerClause::should(term(postings(&[0, 16_000]), field, &snapshot, 2, 2, 0.0)?),
        ScorerClause::should(term(
            postings(&[0, 20_000, 30_000]),
            field,
            &snapshot,
            3,
            2,
            -1.0e8,
        )?),
        ScorerClause::should(term(postings(&[0, 11_000]), field, &snapshot, 2, 2, 0.0)?),
        ScorerClause::should(term(
            postings(&[0, 20_000, 30_000]),
            field,
            &snapshot,
            3,
            2,
            1.0e8,
        )?),
        ScorerClause::should(term(
            postings(&[0, 20_000, 30_000]),
            field,
            &snapshot,
            3,
            2,
            1.0,
        )?),
    ])?;
    let required_optional = ReferenceScorer::boolean(vec![
        ScorerClause::must(term(
            postings(&[0, 11_000, 20_000]),
            field,
            &snapshot,
            3,
            2,
            0.0,
        )?),
        ScorerClause::should(optional),
    ])?;
    let nested_union = ReferenceScorer::boolean(vec![
        ScorerClause::should(required_optional),
        ScorerClause::should(term(postings(&[0, 20_000]), field, &snapshot, 2, 2, 0.0)?),
    ])?;
    let mut query = ReferenceScorer::boolean(vec![
        ScorerClause::must(term(
            postings(&[0, 10_000, 20_000]),
            field,
            &snapshot,
            3,
            1,
            0.0,
        )?),
        ScorerClause::must(nested_union),
    ])?;

    assert_eq!(query.doc(), Some(0));
    assert_eq!(query.next()?, Some(20_000));
    let small = expected_term_score(&snapshot, 3, fieldnorm_to_id(1), 1, 1.0);
    assert_ne!(small.to_bits(), 0.0_f32.to_bits());
    assert_eq!(query.score()?.to_bits(), 0.0_f32.to_bits());
    Ok(())
}

#[test]
fn buffered_union_reproduces_oracle_swap_remove_order_across_horizon()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = vec![Some(1); 5_001];
    let encoded =
        EncodedDocLenSection::encode(0, 5_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 5_001, 5_001)?;
    let clauses = vec![
        ScorerClause::should(term(postings(&[0]), field, &snapshot, 1, 1, 1.0)?),
        ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, 1.0e8)?),
        ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, -1.0e8)?),
        ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, 1.0)?),
    ];
    let mut query = ReferenceScorer::boolean(clauses)?;
    assert_eq!(query.doc(), Some(0));
    query.next()?;
    assert_eq!(query.doc(), Some(5_000));

    let small = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, 1.0);
    let positive = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, 1.0e8);
    let negative = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, -1.0e8);
    // The doc-0 scorer exhausts in the first window and `swap_remove`
    // moves the final small scorer ahead of the two large scorers.
    let expected = ((0.0_f32 + small) + positive) + negative;
    let parse_order = ((0.0_f32 + positive) + negative) + small;
    assert_ne!(expected.to_bits(), parse_order.to_bits());
    assert_eq!(query.score()?.to_bits(), expected.to_bits());
    Ok(())
}

#[test]
fn buffered_union_ranked_topdocs_tracks_term_union_order_across_horizon()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = vec![Some(1); 5_001];
    let encoded =
        EncodedDocLenSection::encode(0, 5_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 5_001, 5_001)?;
    let clauses = vec![
        ScorerClause::should(term(postings(&[1, 5_000]), field, &snapshot, 2, 1, 1.0e8)?),
        ScorerClause::should(term(postings(&[3, 5_000]), field, &snapshot, 2, 1, -1.0e8)?),
        ScorerClause::should(term(postings(&[0, 5_000]), field, &snapshot, 2, 1, 1.0)?),
        ScorerClause::should(term(postings(&[2, 5_000]), field, &snapshot, 2, 1, 1.0)?),
    ];
    let mut query = ReferenceScorer::boolean_topdocs(clauses)?;
    assert_eq!(query.doc(), Some(0));
    while query.doc().is_some_and(|doc| doc < 5_000) {
        query.next()?;
    }
    assert_eq!(query.doc(), Some(5_000));

    let small = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
    let positive = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0e8);
    let negative = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, -1.0e8);
    // The current-document traversal reaches the target in B, C, A, D
    // order. A scorer-major buffered union would retain A, B, C, D.
    let tantivy_term_union = (((0.0_f32 + negative) + small) + positive) + small;
    let scorer_major = (((0.0_f32 + positive) + negative) + small) + small;
    assert_ne!(tantivy_term_union.to_bits(), scorer_major.to_bits());
    assert_eq!(query.score()?.to_bits(), tantivy_term_union.to_bits());
    Ok(())
}

#[test]
fn composite_size_hints_and_costs_match_tantivy_runtime_order()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = vec![Some(1); 10_000];
    let encoded =
        EncodedDocLenSection::encode(0, 10_000, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 10_000, 10_000)?;
    let first_docs = (0_u32..500).collect::<Vec<_>>();
    let second_docs = (0_u32..1_000).collect::<Vec<_>>();
    let inner = ReferenceScorer::boolean(vec![
        ScorerClause::must(term(
            postings(&first_docs),
            field,
            &snapshot,
            500,
            500,
            1.0,
        )?),
        ScorerClause::must(term(
            postings(&second_docs),
            field,
            &snapshot,
            1_000,
            1_000,
            1.0,
        )?),
    ])?;
    assert_eq!(inner.size_hint(), 60);
    assert_eq!(inner.cost(), 500);

    let nested_exclude = ReferenceScorer::boolean(vec![
        ScorerClause::must(inner),
        ScorerClause::must_not(term(postings(&[9_999]), field, &snapshot, 1, 1, 1.0)?),
    ])?;
    assert_eq!(nested_exclude.doc(), Some(0));
    assert_eq!(nested_exclude.size_hint(), 60);
    assert_eq!(nested_exclude.cost(), 60);

    let third_docs = (0_u32..100).collect::<Vec<_>>();
    let outer = ReferenceScorer::boolean(vec![
        ScorerClause::must(nested_exclude),
        ScorerClause::must(term(
            postings(&third_docs),
            field,
            &snapshot,
            100,
            100,
            1.0,
        )?),
    ])?;
    assert_eq!(outer.cost(), 60);

    let mut union = ReferenceScorer::boolean(vec![
        ScorerClause::should(term(postings(&[0]), field, &snapshot, 1, 1, 1.0)?),
        ScorerClause::should(term(postings(&[0, 4_096]), field, &snapshot, 2, 2, 1.0)?),
    ])?;
    assert_eq!(union.doc(), Some(0));
    assert_eq!(union.size_hint(), 2);
    assert_eq!(union.cost(), 2);
    assert_eq!(union.next()?, Some(4_096));
    assert_eq!(union.size_hint(), 0);
    assert_eq!(union.cost(), 0);

    // BooleanWeight passes SegmentReader::num_docs(), not physical BM25 N.
    assert_eq!(estimate_intersection([25, 25].into_iter(), 50), 15);
    assert_eq!(estimate_union([25, 25].into_iter(), 50), 32);
    assert_ne!(estimate_intersection([25, 25].into_iter(), 100), 15);
    assert_ne!(estimate_union([25, 25].into_iter(), 100), 32);
    Ok(())
}

#[test]
fn max_global_docid_survives_union_and_intersection_exhaustion()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1)];
    let encoded = EncodedDocLenSection::encode(
        u64::from(u32::MAX),
        1_u64 << 32,
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 1, 1)?;

    let mut union = ReferenceScorer::boolean(vec![
        ScorerClause::should(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
        ScorerClause::should(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
    ])?;
    assert_eq!(union.doc(), Some(u32::MAX));
    assert!(union.score()?.is_finite());
    assert_eq!(union.next()?, None);
    assert_eq!(union.next()?, None);

    let mut intersection = ReferenceScorer::boolean(vec![
        ScorerClause::must(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
        ScorerClause::must(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
    ])?;
    assert_eq!(intersection.doc(), Some(u32::MAX));
    assert!(intersection.score()?.is_finite());
    assert_eq!(intersection.next()?, None);
    assert_eq!(intersection.next()?, None);
    Ok(())
}

#[test]
fn randomized_boolean_matches_naive_set_oracle() -> Result<(), Box<dyn std::error::Error>> {
    const DOCS: usize = 48;
    let lengths = [Some(1); DOCS];
    let encoded =
        EncodedDocLenSection::encode(0, DOCS as u64, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, DOCS as u64, DOCS as u64)?;

    for seed in 1_u64..=24 {
        let mut state = seed;
        let mut membership = [[false; DOCS]; 5];
        for doc in 0..DOCS {
            for term_membership in &mut membership {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                term_membership[doc] = state >> 61 != 0;
            }
        }
        let term_docs = |term_index: usize| {
            membership[term_index]
                .iter()
                .enumerate()
                .filter_map(|(doc, present)| {
                    present.then_some(u32::try_from(doc).expect("DOCS fits in u32"))
                })
                .collect::<Vec<_>>()
        };
        let docs_by_term = std::array::from_fn::<_, 5, _>(term_docs);
        let leaf = |term_index: usize| {
            term(
                postings(&docs_by_term[term_index]),
                field,
                &snapshot,
                docs_by_term[term_index].len() as u64,
                docs_by_term[term_index].len() as u64,
                1.0,
            )
        };
        let scored_query = || {
            let either = ReferenceScorer::boolean(vec![
                ScorerClause::should(leaf(0)?),
                ScorerClause::should(leaf(1)?),
            ])?;
            ReferenceScorer::boolean(vec![
                ScorerClause::must(either),
                ScorerClause::must(leaf(2)?),
                ScorerClause::must_not(leaf(3)?),
                ScorerClause::should(leaf(4)?),
            ])
        };
        let unscored_query = || {
            let either = ReferenceScorer::boolean_unscored(vec![
                ScorerClause::should(leaf(0)?),
                ScorerClause::should(leaf(1)?),
            ])?;
            ReferenceScorer::boolean_unscored(vec![
                ScorerClause::must(either),
                ScorerClause::must(leaf(2)?),
                ScorerClause::must_not(leaf(3)?),
                ScorerClause::should(leaf(4)?),
            ])
        };
        let mut query = scored_query()?;
        let ranked = query.top_k(DOCS, &AllLiveDocs)?;
        let mut actual = ranked
            .iter()
            .map(|hit| hit.global_docid)
            .collect::<Vec<_>>();
        actual.sort_unstable();
        let expected = (0..DOCS)
            .filter(|&doc| {
                (membership[0][doc] || membership[1][doc])
                    && membership[2][doc]
                    && !membership[3][doc]
            })
            .map(|doc| u32::try_from(doc).expect("DOCS fits in u32"))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "seed {seed}");

        let offset = usize::try_from(seed % 11).expect("bounded seed offset");
        let limit = 7;
        let mut counted_query = scored_query()?;
        let mut counted_collector = TopDocsCollector::with_exact_count(limit, offset)?;
        counted_collector.collect(&mut counted_query, &AllLiveDocs)?;
        let counted = counted_collector.finish()?;
        let page_start = offset.min(ranked.len());
        let page_end = page_start.saturating_add(limit).min(ranked.len());
        assert_eq!(counted.hits, ranked[page_start..page_end], "seed {seed}");
        assert_eq!(
            counted.total_count,
            Some(expected.len() as u64),
            "seed {seed}"
        );

        let mut unscored = unscored_query()?;
        assert_eq!(
            unscored.collect_doc_set(&AllLiveDocs)?,
            expected,
            "seed {seed}"
        );
    }
    Ok(())
}

#[test]
fn e410_seeded_query_algebra_and_pagination_metamorphics_hold()
-> Result<(), Box<dyn std::error::Error>> {
    const DOCS: usize = 48;
    const PAGE: usize = 7;
    let lengths = [Some(1); DOCS];
    let encoded =
        EncodedDocLenSection::encode(0, DOCS as u64, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, DOCS as u64, DOCS as u64)?;

    for seed in 1_u64..=32 {
        let mut state = seed;
        let mut membership = [[false; DOCS]; 3];
        for doc in 0..DOCS {
            for term_membership in &mut membership {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                term_membership[doc] = state >> 63 != 0;
            }
        }
        let docs_by_term = std::array::from_fn::<_, 3, _>(|term_index| {
            membership[term_index]
                .iter()
                .enumerate()
                .filter_map(|(doc, present)| {
                    present.then_some(u32::try_from(doc).expect("DOCS fits u32"))
                })
                .collect::<Vec<_>>()
        });
        let leaf = |term_index: usize| {
            term(
                postings(&docs_by_term[term_index]),
                field,
                &snapshot,
                docs_by_term[term_index].len() as u64,
                docs_by_term[term_index].len() as u64,
                1.0,
            )
        };
        let should_set = |order: [usize; 3]| {
            let mut scorer = ReferenceScorer::boolean_unscored(
                order
                    .into_iter()
                    .map(|index| leaf(index).map(ScorerClause::should))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            scorer.collect_doc_set(&AllLiveDocs)
        };
        let expected_union = (0..DOCS)
            .filter(|&doc| membership.iter().any(|term| term[doc]))
            .map(|doc| u32::try_from(doc).expect("DOCS fits u32"))
            .collect::<Vec<_>>();
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            assert_eq!(
                should_set(order)?,
                expected_union,
                "Should result-set commutativity failed for seed {seed} order={order:?}",
            );
        }

        let mut left_associated = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::must(ReferenceScorer::boolean_unscored(vec![
                ScorerClause::must(leaf(0)?),
                ScorerClause::must(leaf(1)?),
            ])?),
            ScorerClause::must(leaf(2)?),
        ])?;
        let mut right_associated = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::must(leaf(0)?),
            ScorerClause::must(ReferenceScorer::boolean_unscored(vec![
                ScorerClause::must(leaf(1)?),
                ScorerClause::must(leaf(2)?),
            ])?),
        ])?;
        let expected_intersection = (0..DOCS)
            .filter(|&doc| membership.iter().all(|term| term[doc]))
            .map(|doc| u32::try_from(doc).expect("DOCS fits u32"))
            .collect::<Vec<_>>();
        assert_eq!(
            left_associated.collect_doc_set(&AllLiveDocs)?,
            expected_intersection,
            "left-associated Must drifted from the set oracle for seed {seed}",
        );
        assert_eq!(
            right_associated.collect_doc_set(&AllLiveDocs)?,
            expected_intersection,
            "right-associated Must drifted from the set oracle for seed {seed}",
        );

        let ranked_should = || {
            ReferenceScorer::boolean(vec![
                ScorerClause::should(leaf(0)?),
                ScorerClause::should(leaf(1)?),
                ScorerClause::should(leaf(2)?),
            ])
        };
        let mut full_scorer = ranked_should()?;
        let full = full_scorer.top_k(DOCS, &AllLiveDocs)?;
        let mut scored_docids = full.iter().map(|hit| hit.global_docid).collect::<Vec<_>>();
        scored_docids.sort_unstable();
        assert_eq!(
            scored_docids, expected_union,
            "scored Should lowering drifted from the set oracle for seed {seed}",
        );
        let mut concatenated = Vec::new();
        for offset in (0..DOCS).step_by(PAGE) {
            let mut page_scorer = ranked_should()?;
            let mut collector = TopDocsCollector::new(PAGE, offset)?;
            collector.collect(&mut page_scorer, &AllLiveDocs)?;
            concatenated.extend(collector.finish()?.hits);
        }
        assert_eq!(
            concatenated, full,
            "page concatenation drifted from the full ranking for seed {seed}",
        );
    }
    Ok(())
}

#[test]
fn global_collectors_page_count_and_materialize_only_winners()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = vec![Some(1); 10];
    let encoded = EncodedDocLenSection::encode(0, 10, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 10, 10)?;
    let left_docs = [0, 2, 4, 6, 8];
    let right_docs = [1, 3, 5, 7, 9];

    let collect = |limit: usize,
                   offset: usize,
                   exact_count: bool,
                   live_docs: &dyn LiveDocs|
     -> Result<CollectedTopDocs, ArgusError> {
        let mut collector = if exact_count {
            TopDocsCollector::with_exact_count(limit, offset)?
        } else {
            TopDocsCollector::new(limit, offset)?
        };
        for docs in [&left_docs[..], &right_docs[..]] {
            let mut scorer = term(postings(docs), field, &snapshot, 10, 5, 1.0)?;
            collector.collect(&mut scorer, live_docs)?;
        }
        collector.finish()
    };

    let full = collect(10, 0, false, &AllLiveDocs)?;
    assert_eq!(
        full.hits
            .iter()
            .map(|hit| hit.global_docid)
            .collect::<Vec<_>>(),
        (0..10).collect::<Vec<_>>()
    );
    assert_eq!(full.total_count, None);

    let page = collect(3, 4, false, &AllLiveDocs)?;
    assert_eq!(page.hits, full.hits[4..7]);
    let past_end = collect(3, 20, false, &AllLiveDocs)?;
    assert!(past_end.hits.is_empty());

    let live = |doc| doc % 3 != 0;
    let live_full = collect(10, 0, false, &live)?;
    let counted = collect(2, 1, true, &live)?;
    assert_eq!(counted.total_count, Some(6));
    assert_eq!(counted.hits, live_full.hits[1..3]);
    let count_only = collect(0, usize::MAX, true, &live)?;
    assert!(count_only.hits.is_empty());
    assert_eq!(count_only.total_count, Some(6));

    let mut unconsumed = term(postings(&left_docs), field, &snapshot, 10, 5, 1.0)?;
    let mut empty_page = TopDocsCollector::new(0, usize::MAX)?;
    empty_page.collect(&mut unconsumed, &AllLiveDocs)?;
    assert_eq!(unconsumed.doc(), Some(0));
    assert!(empty_page.finish()?.hits.is_empty());
    assert!(matches!(
        TopDocsCollector::new(1, usize::MAX),
        Err(ArgusError::CollectorWindowOverflow {
            limit: 1,
            offset: usize::MAX
        })
    ));

    let mut materialized = Vec::new();
    let hits = materialize_doc_ids(&page.hits, |global_docid| {
        materialized.push(global_docid);
        Ok(format!("winner-{global_docid}").into())
    })?;
    assert_eq!(materialized, vec![4, 5, 6]);
    assert_eq!(hits.len(), page.hits.len());
    Ok(())
}

#[test]
fn topdocs_merge_reproduces_single_collector_under_any_partition_and_order()
-> Result<(), ArgusError> {
    // Deterministic scored stream with heavy ties: seven score buckets
    // over 257 docids so cutoff-boundary ties cross every partition.
    let mut state = 0x5eed_cafe_0000_0001_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let entries: Vec<(u32, f32)> = (0..257_u32)
        .map(|docid| {
            #[allow(clippy::cast_precision_loss)]
            let bucket = (next() % 7) as f32;
            (docid, 0.25_f32.mul_add(bucket, 1.0))
        })
        .collect();

    let build = |limit: usize, offset: usize, exact: bool| {
        if exact {
            TopDocsCollector::with_exact_count(limit, offset)
        } else {
            TopDocsCollector::new(limit, offset)
        }
    };
    let page_key = |collected: CollectedTopDocs| {
        (
            collected
                .hits
                .iter()
                .map(|hit| (hit.global_docid, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            collected.total_count,
        )
    };

    for (limit, offset, exact) in [
        (10_usize, 0_usize, false),
        (10, 0, true),
        (5, 7, true),
        (3, 1, false),
        (0, 0, true),
    ] {
        let mut baseline = build(limit, offset, exact)?;
        for &(docid, score) in &entries {
            baseline.record_live(docid, Some(score))?;
        }
        let expected = page_key(baseline.finish()?);

        for partitions in [1_usize, 4, 7] {
            for rotation in 0..partitions {
                let mut partials = Vec::with_capacity(partitions);
                for _ in 0..partitions {
                    partials.push(build(limit, offset, exact)?);
                }
                for (position, &(docid, score)) in entries.iter().enumerate() {
                    partials[position % partitions].record_live(docid, Some(score))?;
                }
                let mut fold = build(limit, offset, exact)?;
                // Merge in a rotated order: the retained set must be
                // identical no matter which partial folds first.
                for step in 0..partitions {
                    let partial = partials[(step + rotation) % partitions].empty_like()?;
                    let taken =
                        std::mem::replace(&mut partials[(step + rotation) % partitions], partial);
                    fold.merge(taken)?;
                }
                assert_eq!(
                    page_key(fold.finish()?),
                    expected,
                    "merge diverged: limit={limit} offset={offset} exact={exact} \
                     partitions={partitions} rotation={rotation}",
                );
            }
        }
    }

    // Shape mismatches fail closed instead of silently mixing windows.
    let mut ten = TopDocsCollector::new(10, 0)?;
    assert!(ten.merge(TopDocsCollector::new(5, 0)?).is_err());
    assert!(ten.merge(TopDocsCollector::new(10, 1)?).is_err());
    assert!(
        ten.merge(TopDocsCollector::with_exact_count(10, 0)?)
            .is_err()
    );
    Ok(())
}

#[test]
fn doc_set_uses_recursive_lazy_unscored_unions() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[derive(Clone)]
    struct CountingFieldNorm<'a> {
        field: DocLenField<'a>,
        reads: Arc<AtomicUsize>,
    }

    impl FieldNormReader for CountingFieldNorm<'_> {
        fn field_ord(&self) -> u16 {
            self.field.field_ord()
        }

        fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
            self.reads.fetch_add(1, AtomicOrdering::SeqCst);
            self.field.fieldnorm_id(u64::from(global_docid))
        }
    }

    let lengths = vec![Some(1); 5];
    let encoded = EncodedDocLenSection::encode(0, 5, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 5, 5)?;
    let reads = Arc::new(AtomicUsize::new(0));
    let leaf = |docs: &[u32]| {
        Ok::<_, ArgusError>(ReferenceScorer::term(TermScorer::new(
            VecCursor::new(postings(docs), docs.len() as u64, 5),
            CountingFieldNorm {
                field,
                reads: Arc::clone(&reads),
            },
            snapshot.clone(),
            docs.len() as u64,
            TermRecordOption::WithFreqs,
            1.0,
        )?))
    };
    let union = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::should(leaf(&[0, 2, 4])?),
        ScorerClause::should(leaf(&[1, 2, 3])?),
    ])?;
    let excluded = leaf(&[3])?;
    let mut query = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::must(union),
        ScorerClause::must_not(excluded),
    ])?;
    assert_eq!(query.collect_doc_set(&AllLiveDocs)?, vec![0, 1, 2, 4]);
    assert_eq!(
        reads.load(AtomicOrdering::SeqCst),
        7,
        "each physical posting is fieldnorm-validated once; none is scored"
    );

    let mut scored_union = ReferenceScorer::boolean(vec![
        ScorerClause::should(term(postings(&[0, 2]), field, &snapshot, 2, 2, 1.0)?),
        ScorerClause::should(term(postings(&[1, 2]), field, &snapshot, 2, 2, 1.0)?),
    ])?;
    assert!(matches!(
        scored_union.collect_doc_set(&AllLiveDocs),
        Err(ArgusError::ScoredTreeForUnscoredCollector)
    ));
    Ok(())
}
