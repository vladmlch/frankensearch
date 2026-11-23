use super::*;

#[test]
fn phrase_scorer_reads_sealed_postings_and_positions() -> Result<(), Box<dyn std::error::Error>> {
    let first_postings = [Posting::new(1, 2), Posting::new(2, 1), Posting::new(3, 1)];
    let encoded_first_postings = EncodedPostingList::encode(&first_postings)?;
    let first_posting_list = encoded_first_postings.posting_list()?;
    let encoded_first_positions = EncodedPositionList::encode(&first_postings, &[0, 4, 0, 1])?;
    let first_positions = encoded_first_positions.position_list(&first_posting_list)?;

    let second_postings = [Posting::new(1, 2), Posting::new(2, 1), Posting::new(3, 1)];
    let encoded_second_postings = EncodedPostingList::encode(&second_postings)?;
    let second_posting_list = encoded_second_postings.posting_list()?;
    let encoded_second_positions = EncodedPositionList::encode(&second_postings, &[1, 5, 2, 2])?;
    let second_positions = encoded_second_positions.position_list(&second_posting_list)?;

    let lengths = [Some(8); 4];
    let encoded_doclens =
        EncodedDocLenSection::encode(0, 4, &[6], &[DocLenFieldInput::new(6, &lengths)])?;
    let doclens = encoded_doclens.section(&[6])?;
    let field = doclens.field(6).expect("field exists");
    let snapshot = snapshot(6, 32, 4)?;
    let mut phrase = PhraseScorer::new(
        vec![
            PhraseTerm::new(
                6,
                0,
                SealedPostingCursor::with_positions(&first_positions, 4)?,
                3,
            ),
            PhraseTerm::new(
                6,
                1,
                SealedPostingCursor::with_positions(&second_positions, 4)?,
                3,
            ),
        ],
        field,
        snapshot.clone(),
        1.0,
    )?;

    assert_eq!(phrase.doc(), Some(1));
    let expected = expected_phrase_score(
        &snapshot,
        &[3, 3],
        field.fieldnorm_id(1).expect("fieldnorm"),
        2,
        1.0,
    );
    assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
    assert_eq!(phrase.next()?, Some(3));
    let expected = expected_phrase_score(
        &snapshot,
        &[3, 3],
        field.fieldnorm_id(3).expect("fieldnorm"),
        1,
        1.0,
    );
    assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
    assert_eq!(phrase.next()?, None);
    Ok(())
}

#[test]
fn phrase_scorer_filters_candidates_counts_frequency_and_seeks()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(8); 8];
    let encoded = EncodedDocLenSection::encode(0, 8, &[5], &[DocLenFieldInput::new(5, &lengths)])?;
    let section = encoded.section(&[5])?;
    let field = section.field(5).expect("field exists");
    let snapshot = snapshot(5, 64, 8)?;
    let build = || {
        let first = PositionedVecCursor::new(
            vec![(1, vec![0, 4]), (2, vec![0]), (3, vec![2]), (5, vec![1])],
            4,
            8,
        );
        let second = PositionedVecCursor::new(
            vec![(1, vec![1, 5]), (2, vec![2]), (3, vec![3]), (5, vec![0])],
            4,
            8,
        );
        PhraseScorer::new(
            vec![
                PhraseTerm::new(5, 0, first, 4),
                PhraseTerm::new(5, 1, second, 4),
            ],
            field,
            snapshot.clone(),
            1.0,
        )
    };

    let mut phrase = build()?;
    assert_eq!(phrase.doc(), Some(1));
    let candidate_estimate = estimate_intersection([4, 4].into_iter(), 8);
    assert_eq!(phrase.cost, u64::from(candidate_estimate) * 20);
    assert_eq!(phrase.size_hint, candidate_estimate / 20);
    let expected = expected_phrase_score(
        &snapshot,
        &[4, 4],
        field.fieldnorm_id(1).expect("fieldnorm"),
        2,
        1.0,
    );
    assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
    let decode_capacity = phrase.decode_scratch.capacity();
    let position_capacities = phrase
        .slots
        .iter()
        .map(|slot| slot.positions.capacity())
        .collect::<Vec<_>>();
    assert_eq!(phrase.next()?, Some(3));
    assert_eq!(phrase.decode_scratch.capacity(), decode_capacity);
    assert_eq!(
        phrase
            .slots
            .iter()
            .map(|slot| slot.positions.capacity())
            .collect::<Vec<_>>(),
        position_capacities
    );
    let expected = expected_phrase_score(
        &snapshot,
        &[4, 4],
        field.fieldnorm_id(3).expect("fieldnorm"),
        1,
        1.0,
    );
    assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
    assert_eq!(phrase.next()?, None);
    assert_eq!(phrase.next()?, None);

    let mut phrase = build()?;
    assert_eq!(phrase.seek(2)?, Some(3));
    assert_eq!(phrase.seek(3)?, Some(3));
    assert_eq!(phrase.seek(4)?, None);

    let mut scorer = ReferenceScorer::phrase(build()?);
    let winners = scorer.top_k(10, &AllLiveDocs)?;
    assert_eq!(
        winners
            .iter()
            .map(|winner| winner.global_docid)
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
    Ok(())
}

#[test]
fn phrase_slop_shares_one_budget_and_charges_transposition_twice()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(8); 4];
    let encoded = EncodedDocLenSection::encode(0, 4, &[5], &[DocLenFieldInput::new(5, &lengths)])?;
    let section = encoded.section(&[5])?;
    let field = section.field(5).expect("field exists");
    let snapshot = snapshot(5, 32, 4)?;

    let build = |slop| {
        let first = PositionedVecCursor::new(
            vec![(0, vec![0]), (1, vec![0]), (2, vec![1]), (3, vec![0])],
            4,
            4,
        );
        let second = PositionedVecCursor::new(
            vec![(0, vec![2]), (1, vec![3]), (2, vec![0]), (3, vec![2])],
            4,
            4,
        );
        let third = PositionedVecCursor::new(
            vec![(0, vec![3]), (1, vec![4]), (2, vec![2]), (3, vec![4])],
            4,
            4,
        );
        PhraseScorer::new_with_slop(
            vec![
                PhraseTerm::new(5, 0, first, 4),
                PhraseTerm::new(5, 1, second, 4),
                PhraseTerm::new(5, 2, third, 4),
            ],
            field,
            snapshot.clone(),
            1.0,
            slop,
        )
    };

    let mut one = build(1)?;
    assert_eq!(one.doc(), Some(0), "one inserted token spends one unit");
    assert_eq!(
        one.next()?,
        None,
        "transposition and two independent gaps exceed one shared unit"
    );

    let mut two = build(2)?;
    assert_eq!(two.doc(), Some(0));
    assert_eq!(
        two.next()?,
        Some(1),
        "two gaps spend the complete shared budget"
    );
    assert_eq!(two.next()?, Some(3), "two independent gaps spend two units");
    assert_eq!(two.next()?, None);

    let build_transposition = |slop| {
        PhraseScorer::new_with_slop(
            vec![
                PhraseTerm::new(5, 0, PositionedVecCursor::new(vec![(2, vec![1])], 1, 4), 1),
                PhraseTerm::new(5, 1, PositionedVecCursor::new(vec![(2, vec![0])], 1, 4), 1),
            ],
            field,
            snapshot.clone(),
            1.0,
            slop,
        )
    };
    assert_eq!(build_transposition(1)?.doc(), None);
    assert_eq!(build_transposition(2)?.doc(), Some(2));

    let mut three = build(3)?;
    assert_eq!(three.doc(), Some(0));
    assert_eq!(three.next()?, Some(1));
    assert_eq!(
        three.next()?,
        Some(2),
        "three-term transposition spends three"
    );
    assert_eq!(three.next()?, Some(3));
    assert_eq!(three.next()?, None);
    Ok(())
}

#[test]
fn phrase_slop_overflow_cannot_become_a_match() -> Result<(), ArgusError> {
    let mut positions = vec![u32::MAX];
    let mut slops = vec![u32::MAX];
    let mut positions_buffer = Vec::new();
    let mut slops_buffer = Vec::new();
    assert_eq!(
        intersect_phrase_positions_carrying_slop(
            &mut positions,
            &mut slops,
            &[0],
            u32::MAX,
            false,
            &mut positions_buffer,
            &mut slops_buffer,
        )?,
        0
    );
    Ok(())
}

#[test]
fn phrase_cost_estimate_follows_tantivy_child_cost_order() -> Result<(), Box<dyn std::error::Error>>
{
    let lengths = [Some(2); 12];
    let encoded = EncodedDocLenSection::encode(0, 12, &[4], &[DocLenFieldInput::new(4, &lengths)])?;
    let section = encoded.section(&[4])?;
    let field = section.field(4).expect("field exists");
    let snapshot = snapshot(4, 24, 12)?;
    let broad = PositionedVecCursor::new(
        vec![
            (0, vec![0]),
            (1, vec![0]),
            (2, vec![0]),
            (3, vec![0]),
            (4, vec![0]),
        ],
        5,
        12,
    );
    let narrow = PositionedVecCursor::new(vec![(0, vec![1])], 1, 12);
    let phrase = PhraseScorer::new(
        vec![
            PhraseTerm::new(4, 0, broad, 5),
            PhraseTerm::new(4, 1, narrow, 1),
        ],
        field,
        snapshot,
        1.0,
    )?;

    assert_eq!(estimate_intersection([5, 1].into_iter(), 12), 0);
    assert_eq!(estimate_intersection([1, 5].into_iter(), 12), 1);
    let scorer = ReferenceScorer::phrase(phrase);
    assert_eq!(scorer.doc(), Some(0));
    assert_eq!(scorer.cost(), 20);
    assert_eq!(scorer.size_hint(), 0);
    Ok(())
}

#[test]
fn phrase_scorer_handles_repeated_terms_same_position_or_and_query_gaps()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(6); 16];
    let encoded = EncodedDocLenSection::encode(0, 16, &[2], &[DocLenFieldInput::new(2, &lengths)])?;
    let section = encoded.section(&[2])?;
    let field = section.field(2).expect("field exists");
    let snapshot = snapshot(2, 96, 16)?;

    let repeated_a = || PositionedVecCursor::new(vec![(7, vec![0, 1, 2])], 1, 16);
    let repeated_b = PositionedVecCursor::new(vec![(7, vec![2, 3])], 1, 16);
    let repeated = PhraseScorer::new(
        vec![
            PhraseTerm::new(2, 0, repeated_a(), 1),
            PhraseTerm::new(2, 1, repeated_a(), 1),
            PhraseTerm::new(2, 2, repeated_b, 1),
        ],
        field,
        snapshot.clone(),
        1.0,
    )?;
    assert_eq!(repeated.doc(), Some(7));
    let expected = expected_phrase_score(
        &snapshot,
        &[1, 1, 1],
        field.fieldnorm_id(7).expect("fieldnorm"),
        2,
        1.0,
    );
    assert_eq!(repeated.score()?.to_bits(), expected.to_bits());

    let alternative_x = PositionedVecCursor::new(vec![(8, vec![5]), (10, vec![1])], 2, 16);
    let alternative_y = PositionedVecCursor::new(vec![(9, vec![10]), (10, vec![1])], 2, 16);
    let following =
        PositionedVecCursor::new(vec![(8, vec![6]), (9, vec![11]), (10, vec![2])], 3, 16);
    let mut alternatives = PhraseScorer::new(
        vec![
            PhraseTerm::new(2, 0, alternative_x, 2),
            PhraseTerm::new(2, 0, alternative_y, 2),
            PhraseTerm::new(2, 1, following, 3),
        ],
        field,
        snapshot.clone(),
        1.0,
    )?;
    for expected_doc in [8, 9, 10] {
        assert_eq!(alternatives.doc(), Some(expected_doc));
        let expected = expected_phrase_score(
            &snapshot,
            &[2, 2, 3],
            field
                .fieldnorm_id(u64::from(expected_doc))
                .expect("fieldnorm"),
            1,
            1.0,
        );
        assert_eq!(alternatives.score()?.to_bits(), expected.to_bits());
        alternatives.next()?;
    }
    assert_eq!(alternatives.doc(), None);

    let gap_start = PositionedVecCursor::new(vec![(6, vec![10]), (11, vec![20])], 2, 16);
    let gap_end = PositionedVecCursor::new(vec![(6, vec![12]), (11, vec![21])], 2, 16);
    let mut gapped = PhraseScorer::new(
        vec![
            PhraseTerm::new(2, 4, gap_start, 2),
            PhraseTerm::new(2, 6, gap_end, 2),
        ],
        field,
        snapshot,
        1.0,
    )?;
    assert_eq!(gapped.doc(), Some(6));
    assert_eq!(gapped.next()?, None);
    Ok(())
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn phrase_scorer_matches_tantivy_corpus_docs_ranks_and_score_bits()
-> Result<(), Box<dyn std::error::Error>> {
    use frankensearch_lexical::tantivy_crate::{
        Index, TantivyDocument, Term,
        collector::{DocSetCollector, TopDocs},
        doc,
        query::PhraseQuery,
        schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions},
    };

    let mut schema_builder = Schema::builder();
    let indexing = TextFieldIndexing::default()
        .set_tokenizer("default")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let body = schema_builder.add_text_field(
        "body",
        TextOptions::default().set_indexing_options(indexing),
    );
    let index = Index::create_in_ram(schema_builder.build());
    let mut writer = index.writer_with_num_threads::<TantivyDocument>(1, 15_000_000)?;
    let corpus = [
        "a b",
        "a b a b",
        "a x b",
        "a only",
        "b a",
        "a b filler filler filler filler filler filler",
    ];
    for text in corpus {
        writer.add_document(doc!(body => text))?;
    }
    writer.commit()?;
    drop(writer);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let query = PhraseQuery::new(vec![
        Term::from_field_text(body, "a"),
        Term::from_field_text(body, "b"),
    ]);
    let mut oracle_docset = searcher
        .search(&query, &DocSetCollector)?
        .into_iter()
        .map(|address| {
            assert_eq!(address.segment_ord, 0);
            address.doc_id
        })
        .collect::<Vec<_>>();
    oracle_docset.sort_unstable();
    assert_eq!(oracle_docset, vec![0, 1, 5]);
    let oracle_ranked = searcher
        .search(&query, &TopDocs::with_limit(corpus.len()).order_by_score())?
        .into_iter()
        .map(|(score, address)| {
            assert_eq!(address.segment_ord, 0);
            (address.doc_id, score.to_bits())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        oracle_ranked
            .iter()
            .map(|(doc, _)| *doc)
            .collect::<Vec<_>>(),
        vec![1, 0, 5]
    );

    let lengths = [Some(2), Some(4), Some(3), Some(2), Some(2), Some(8)];
    let encoded = EncodedDocLenSection::encode(0, 6, &[0], &[DocLenFieldInput::new(0, &lengths)])?;
    let section = encoded.section(&[0])?;
    let field = section.field(0).expect("field exists");
    let snapshot = snapshot(0, 21, 6)?;
    let a = PositionedVecCursor::new(
        vec![
            (0, vec![0]),
            (1, vec![0, 2]),
            (2, vec![0]),
            (3, vec![0]),
            (4, vec![1]),
            (5, vec![0]),
        ],
        6,
        6,
    );
    let b = PositionedVecCursor::new(
        vec![
            (0, vec![1]),
            (1, vec![1, 3]),
            (2, vec![2]),
            (4, vec![0]),
            (5, vec![1]),
        ],
        5,
        6,
    );
    let phrase = PhraseScorer::new(
        vec![PhraseTerm::new(0, 0, a, 6), PhraseTerm::new(0, 1, b, 5)],
        field,
        snapshot,
        1.0,
    )?;
    let mut scorer = ReferenceScorer::phrase(phrase);
    let quill_ranked = scorer
        .top_k(corpus.len(), &AllLiveDocs)?
        .into_iter()
        .map(|hit| (hit.global_docid, hit.score.to_bits()))
        .collect::<Vec<_>>();
    assert_eq!(quill_ranked, oracle_ranked);
    Ok(())
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn phrase_slop_matches_tantivy_docs_ranks_and_score_bits() -> Result<(), Box<dyn std::error::Error>>
{
    use frankensearch_lexical::tantivy_crate::{
        Index, TantivyDocument, Term,
        collector::TopDocs,
        doc,
        query::PhraseQuery,
        schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions},
    };

    let mut schema_builder = Schema::builder();
    let indexing = TextFieldIndexing::default()
        .set_tokenizer("default")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let body = schema_builder.add_text_field(
        "body",
        TextOptions::default().set_indexing_options(indexing),
    );
    let index = Index::create_in_ram(schema_builder.build());
    let mut writer = index.writer_with_num_threads::<TantivyDocument>(1, 15_000_000)?;
    let corpus = [
        "a b c",
        "a x b c",
        "a x b x c",
        "b a c",
        "a c b",
        "a b c a b c",
    ];
    for text in corpus {
        writer.add_document(doc!(body => text))?;
    }
    writer.commit()?;
    drop(writer);
    let reader = index.reader()?;
    let searcher = reader.searcher();

    let lengths = [Some(3), Some(4), Some(5), Some(3), Some(3), Some(6)];
    let encoded = EncodedDocLenSection::encode(0, 6, &[0], &[DocLenFieldInput::new(0, &lengths)])?;
    let section = encoded.section(&[0])?;
    let field = section.field(0).expect("field exists");
    let snapshot = snapshot(0, 24, 6)?;
    let build = |slop| {
        let a = PositionedVecCursor::new(
            vec![
                (0, vec![0]),
                (1, vec![0]),
                (2, vec![0]),
                (3, vec![1]),
                (4, vec![0]),
                (5, vec![0, 3]),
            ],
            6,
            6,
        );
        let b = PositionedVecCursor::new(
            vec![
                (0, vec![1]),
                (1, vec![2]),
                (2, vec![2]),
                (3, vec![0]),
                (4, vec![2]),
                (5, vec![1, 4]),
            ],
            6,
            6,
        );
        let c = PositionedVecCursor::new(
            vec![
                (0, vec![2]),
                (1, vec![3]),
                (2, vec![4]),
                (3, vec![2]),
                (4, vec![1]),
                (5, vec![2, 5]),
            ],
            6,
            6,
        );
        PhraseScorer::new_with_slop(
            vec![
                PhraseTerm::new(0, 0, a, 6),
                PhraseTerm::new(0, 1, b, 6),
                PhraseTerm::new(0, 2, c, 6),
            ],
            field,
            snapshot.clone(),
            1.0,
            slop,
        )
    };

    for slop in [1, 2] {
        let mut oracle = PhraseQuery::new(vec![
            Term::from_field_text(body, "a"),
            Term::from_field_text(body, "b"),
            Term::from_field_text(body, "c"),
        ]);
        oracle.set_slop(slop);
        let oracle_ranked = searcher
            .search(&oracle, &TopDocs::with_limit(corpus.len()).order_by_score())?
            .into_iter()
            .map(|(score, address)| {
                assert_eq!(address.segment_ord, 0);
                (address.doc_id, score.to_bits())
            })
            .collect::<Vec<_>>();

        let mut quill = ReferenceScorer::phrase(build(slop)?);
        let quill_ranked = quill
            .top_k(corpus.len(), &AllLiveDocs)?
            .into_iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(quill_ranked, oracle_ranked, "slop={slop}");
    }
    Ok(())
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn phrase_same_position_alternatives_are_reviewed_or_divergence()
-> Result<(), Box<dyn std::error::Error>> {
    use frankensearch_lexical::tantivy_crate::{
        Index, TantivyDocument, Term,
        collector::DocSetCollector,
        doc,
        query::PhraseQuery,
        schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions},
        tokenizer::{PreTokenizedString, Token},
    };

    let mut schema_builder = Schema::builder();
    let indexing = TextFieldIndexing::default()
        .set_tokenizer("default")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let body = schema_builder.add_text_field(
        "body",
        TextOptions::default().set_indexing_options(indexing),
    );
    let index = Index::create_in_ram(schema_builder.build());
    let token = |text: &str, position: usize, offset_from: usize, offset_to: usize| Token {
        offset_from,
        offset_to,
        position,
        text: text.to_owned(),
        position_length: 1,
    };
    let docs = [
        PreTokenizedString {
            text: "a c".to_owned(),
            tokens: vec![token("a", 0, 0, 1), token("c", 1, 2, 3)],
        },
        PreTokenizedString {
            text: "b c".to_owned(),
            tokens: vec![token("b", 0, 0, 1), token("c", 1, 2, 3)],
        },
        PreTokenizedString {
            text: "a/b c".to_owned(),
            tokens: vec![
                token("a", 0, 0, 3),
                token("b", 0, 0, 3),
                token("c", 1, 4, 5),
            ],
        },
    ];
    let mut writer = index.writer_with_num_threads::<TantivyDocument>(1, 15_000_000)?;
    for text in docs {
        writer.add_document(doc!(body => text))?;
    }
    writer.commit()?;
    drop(writer);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let oracle_query = PhraseQuery::new_with_offset(vec![
        (0, Term::from_field_text(body, "a")),
        (0, Term::from_field_text(body, "b")),
        (1, Term::from_field_text(body, "c")),
    ]);
    let mut oracle_docids = searcher
        .search(&oracle_query, &DocSetCollector)?
        .into_iter()
        .map(|address| {
            assert_eq!(address.segment_ord, 0);
            address.doc_id
        })
        .collect::<Vec<_>>();
    oracle_docids.sort_unstable();
    assert_eq!(oracle_docids, vec![2]);

    let lengths = [Some(2), Some(2), Some(3)];
    let encoded = EncodedDocLenSection::encode(0, 3, &[0], &[DocLenFieldInput::new(0, &lengths)])?;
    let section = encoded.section(&[0])?;
    let field = section.field(0).expect("field exists");
    let phrase = PhraseScorer::new(
        vec![
            PhraseTerm::new(
                0,
                0,
                PositionedVecCursor::new(vec![(0, vec![0]), (2, vec![0])], 2, 3),
                2,
            ),
            PhraseTerm::new(
                0,
                0,
                PositionedVecCursor::new(vec![(1, vec![0]), (2, vec![0])], 2, 3),
                2,
            ),
            PhraseTerm::new(
                0,
                1,
                PositionedVecCursor::new(vec![(0, vec![1]), (1, vec![1]), (2, vec![1])], 3, 3),
                3,
            ),
        ],
        field,
        snapshot(0, 7, 3)?,
        1.0,
    )?;
    let mut scorer = ReferenceScorer::phrase(phrase);
    let mut quill_docids = scorer
        .top_k(3, &AllLiveDocs)?
        .into_iter()
        .map(|hit| hit.global_docid)
        .collect::<Vec<_>>();
    quill_docids.sort_unstable();
    assert_eq!(quill_docids, vec![0, 1, 2]);
    assert_ne!(quill_docids, oracle_docids);
    Ok(())
}

#[test]
fn phrase_frequency_exceeds_u8_and_matches_u32_position_boundary()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(400); 16];
    let encoded = EncodedDocLenSection::encode(0, 16, &[9], &[DocLenFieldInput::new(9, &lengths)])?;
    let section = encoded.section(&[9])?;
    let field = section.field(9).expect("field exists");
    let snapshot = snapshot(9, 6_400, 16)?;
    let first = PositionedVecCursor::new(
        vec![(11, (0..=300).collect()), (12, vec![u32::MAX - 1])],
        2,
        16,
    );
    let second =
        PositionedVecCursor::new(vec![(11, (1..=301).collect()), (12, vec![u32::MAX])], 2, 16);
    let mut phrase = PhraseScorer::new(
        vec![
            PhraseTerm::new(9, 0, first, 2),
            PhraseTerm::new(9, 1, second, 2),
        ],
        field,
        snapshot.clone(),
        1.0,
    )?;

    assert_eq!(phrase.doc(), Some(11));
    let expected = expected_phrase_score(
        &snapshot,
        &[2, 2],
        field.fieldnorm_id(11).expect("fieldnorm"),
        301,
        1.0,
    );
    assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
    assert_eq!(phrase.next()?, Some(12));
    let expected = expected_phrase_score(
        &snapshot,
        &[2, 2],
        field.fieldnorm_id(12).expect("fieldnorm"),
        1,
        1.0,
    );
    assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
    assert_eq!(phrase.next()?, None);
    Ok(())
}

#[test]
fn phrase_constructor_rejects_missing_positions_and_invalid_slot_order()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(2); 4];
    let encoded = EncodedDocLenSection::encode(0, 4, &[3], &[DocLenFieldInput::new(3, &lengths)])?;
    let section = encoded.section(&[3])?;
    let field = section.field(3).expect("field exists");
    let snapshot = snapshot(3, 8, 4)?;

    let wrong_field = PhraseScorer::new(
        vec![
            PhraseTerm::new(2, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
            PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
        ],
        field,
        snapshot.clone(),
        1.0,
    );
    assert!(matches!(
        wrong_field,
        Err(ArgusError::PhraseTermFieldMismatch {
            term_field: 2,
            stats_field: 3
        })
    ));

    let missing_positions = PhraseScorer::new(
        vec![
            PhraseTerm::new(3, 0, VecCursor::new(postings(&[0]), 1, 4), 1),
            PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
        ],
        field,
        snapshot.clone(),
        1.0,
    );
    assert!(matches!(
        missing_positions,
        Err(ArgusError::MissingPositions {
            field_ord: 3,
            global_docid: 0
        })
    ));

    let descending = PhraseScorer::new(
        vec![
            PhraseTerm::new(3, 2, PositionedVecCursor::new(vec![(0, vec![2])], 1, 4), 1),
            PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
        ],
        field,
        snapshot.clone(),
        1.0,
    );
    assert!(matches!(descending, Err(ArgusError::InvalidPhrase { .. })));

    let one_slot = PhraseScorer::new(
        vec![
            PhraseTerm::new(3, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
            PhraseTerm::new(3, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
        ],
        field,
        snapshot,
        1.0,
    );
    assert!(matches!(one_slot, Err(ArgusError::InvalidPhrase { .. })));
    Ok(())
}
