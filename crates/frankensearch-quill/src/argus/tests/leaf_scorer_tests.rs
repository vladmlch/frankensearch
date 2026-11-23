use super::*;

#[test]
fn all_scorer_is_half_open_unit_scored_and_progresses_monotonically()
-> Result<(), Box<dyn std::error::Error>> {
    let mut all = ReferenceScorer::all(7, 10, 2)?;
    assert_eq!(all.doc(), Some(7));
    assert_eq!(all.cost(), 3);
    assert_eq!(all.size_hint(), 3);
    assert_eq!(all.segment_num_docs(), 2);
    assert_eq!(all.score()?.to_bits(), 1.0_f32.to_bits());

    assert_eq!(all.advance(6)?, Some(7));
    assert_eq!(all.advance(7)?, Some(7));
    assert_eq!(all.advance(8)?, Some(8));
    assert_eq!(all.next()?, Some(9));
    assert_eq!(all.advance(9)?, Some(9));
    assert_eq!(all.next()?, None);
    assert_eq!(all.next()?, None, "exhaustion is fused");
    assert_eq!(
        all.advance(0)?,
        None,
        "an exhausted scorer cannot resurrect"
    );
    assert!(matches!(
        all.score(),
        Err(ArgusError::CursorInvariant(
            "cannot score an exhausted All scorer"
        ))
    ));

    let visible = |docid| docid != 8;
    let mut top_docs = ReferenceScorer::all(7, 10, 2)?;
    assert_eq!(
        top_docs.top_k(3, &visible)?,
        vec![
            ScoredDoc {
                global_docid: 7,
                score: 1.0,
            },
            ScoredDoc {
                global_docid: 9,
                score: 1.0,
            },
        ]
    );
    let mut doc_set = ReferenceScorer::all(7, 10, 2)?;
    assert_eq!(doc_set.collect_doc_set(&visible)?, vec![7, 9]);

    let mut boosted = ReferenceScorer::all_with_boost(7, 10, 2, 2.5)?;
    assert_eq!(boosted.score()?.to_bits(), 2.5_f32.to_bits());
    assert!(matches!(
        ReferenceScorer::all_with_boost(7, 10, 2, f32::NAN),
        Err(ArgusError::InvalidAllBoost { boost_bits }) if boost_bits == f32::NAN.to_bits()
    ));
    assert!(matches!(
        ReferenceScorer::all_with_boost(7, 10, 2, f32::INFINITY),
        Err(ArgusError::InvalidAllBoost { boost_bits })
            if boost_bits == f32::INFINITY.to_bits()
    ));
    Ok(())
}

#[test]
fn constant_score_from_docids_walks_the_set_and_rejects_bad_input()
-> Result<(), Box<dyn std::error::Error>> {
    let mut scorer = ReferenceScorer::constant_score_from_docids(vec![7, 9, 12], 4, 1.0)?;
    assert_eq!(scorer.doc(), Some(7));
    assert_eq!(scorer.cost(), 3);
    assert_eq!(scorer.size_hint(), 3);
    assert_eq!(scorer.segment_num_docs(), 4);
    assert_eq!(scorer.score()?.to_bits(), 1.0_f32.to_bits());
    assert_eq!(scorer.advance(8)?, Some(9));
    assert_eq!(scorer.next()?, Some(12));
    assert_eq!(scorer.next()?, None);
    assert_eq!(scorer.next()?, None, "exhaustion is fused");
    assert_eq!(
        scorer.advance(0)?,
        None,
        "an exhausted scorer cannot resurrect"
    );
    assert!(matches!(
        scorer.score(),
        Err(ArgusError::CursorInvariant(
            "cannot score an exhausted constant-score predicate"
        ))
    ));

    let mut empty = ReferenceScorer::constant_score_from_docids(Vec::new(), 4, 1.0)?;
    assert_eq!(empty.doc(), None);
    assert_eq!(empty.next()?, None, "the empty set stays exhausted");

    assert!(matches!(
        ReferenceScorer::constant_score_from_docids(vec![9, 7], 4, 1.0),
        Err(ArgusError::CursorInvariant(
            "constant-score docids did not advance monotonically"
        ))
    ));
    assert!(matches!(
        ReferenceScorer::constant_score_from_docids(vec![7, 7], 4, 1.0),
        Err(ArgusError::CursorInvariant(
            "constant-score docids did not advance monotonically"
        ))
    ));
    assert!(matches!(
        ReferenceScorer::constant_score_from_docids(vec![7], 4, f32::NAN),
        Err(ArgusError::InvalidConstantBoost { boost_bits })
            if boost_bits == f32::NAN.to_bits()
    ));
    assert!(matches!(
        ReferenceScorer::constant_score_from_docids(vec![7], 4, f32::INFINITY),
        Err(ArgusError::InvalidConstantBoost { boost_bits })
            if boost_bits == f32::INFINITY.to_bits()
    ));
    Ok(())
}

#[test]
fn all_scorer_validates_empty_and_u32_boundary_domains() -> Result<(), Box<dyn std::error::Error>> {
    let empty = ReferenceScorer::all(17, 17, 0)?;
    assert_eq!(empty.doc(), None);
    assert_eq!(empty.cost(), 0);
    assert_eq!(empty.size_hint(), 0);
    assert_eq!(empty.segment_num_docs(), 0);

    assert!(matches!(
        ReferenceScorer::all(18, 17, 0),
        Err(ArgusError::InvalidAllRange { .. })
    ));
    assert!(matches!(
        ReferenceScorer::all(0, MAX_GLOBAL_DOCID_EXCLUSIVE + 1, 0),
        Err(ArgusError::InvalidAllRange { .. })
    ));
    assert!(matches!(
        ReferenceScorer::all(17, 18, 2),
        Err(ArgusError::InvalidAllRange { .. })
    ));

    let mut final_doc = ReferenceScorer::all(u64::from(u32::MAX), MAX_GLOBAL_DOCID_EXCLUSIVE, 1)?;
    assert_eq!(final_doc.doc(), Some(u32::MAX));
    assert_eq!(final_doc.score()?.to_bits(), 1.0_f32.to_bits());
    assert_eq!(final_doc.advance(u32::MAX)?, Some(u32::MAX));
    assert_eq!(final_doc.next()?, None);

    let mut full_domain = ReferenceScorer::all(0, MAX_GLOBAL_DOCID_EXCLUSIVE, u32::MAX)?;
    assert_eq!(full_domain.cost(), MAX_GLOBAL_DOCID_EXCLUSIVE);
    assert_eq!(full_domain.size_hint(), u32::MAX);
    assert_eq!(full_domain.segment_num_docs(), u32::MAX);
    assert_eq!(full_domain.advance(u32::MAX)?, Some(u32::MAX));
    assert_eq!(full_domain.next()?, None);
    Ok(())
}

#[test]
fn all_anchor_composes_with_exclusion_for_top_docs_and_doc_sets()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1); 4];
    let encoded =
        EncodedDocLenSection::encode(10, 14, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 4, 4)?;

    let mut scored = ReferenceScorer::boolean(vec![
        ScorerClause::must(ReferenceScorer::all(10, 14, 4)?),
        ScorerClause::must_not(term(postings(&[11, 13]), field, &snapshot, 2, 2, 1.0)?),
    ])?;
    assert_eq!(
        scored.top_k(4, &AllLiveDocs)?,
        vec![
            ScoredDoc {
                global_docid: 10,
                score: 1.0,
            },
            ScoredDoc {
                global_docid: 12,
                score: 1.0,
            },
        ],
        "the All anchor contributes Tantivy's unit score and MustNot only gates matches"
    );

    let mut unscored = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::must(ReferenceScorer::all_with_boost(10, 14, 4, 2.5)?),
        ScorerClause::must_not(term(postings(&[11, 13]), field, &snapshot, 2, 2, 1.0)?),
    ])?;
    assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![10, 12]);
    Ok(())
}

#[test]
fn boolean_raw_all_does_not_change_required_or_optional_term_score_bits()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1); 4];
    let encoded = EncodedDocLenSection::encode(0, 4, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 4, 4)?;
    let expected = expected_term_score(
        &snapshot,
        2,
        field.fieldnorm_id(1).expect("fieldnorm exists"),
        1,
        1.0,
    );

    let mut all_and_term = ReferenceScorer::boolean(vec![
        ScorerClause::must(ReferenceScorer::all(0, 4, 4)?),
        ScorerClause::must(term(postings(&[1, 3]), field, &snapshot, 2, 2, 1.0)?),
    ])?;
    let hits = all_and_term.top_k(4, &AllLiveDocs)?;
    assert_eq!(
        hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert!(
        hits.iter()
            .all(|hit| hit.score.to_bits() == expected.to_bits())
    );

    let mut optional_all = ReferenceScorer::boolean(vec![
        ScorerClause::must(term(postings(&[1, 3]), field, &snapshot, 2, 2, 1.0)?),
        ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
    ])?;
    let hits = optional_all.top_k(4, &AllLiveDocs)?;
    assert_eq!(
        hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert!(
        hits.iter()
            .all(|hit| hit.score.to_bits() == expected.to_bits())
    );

    let mut unscored = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::must(ReferenceScorer::all(0, 4, 4)?),
        ScorerClause::must(term(postings(&[1, 3]), field, &snapshot, 2, 2, 1.0)?),
        ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
    ])?;
    assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![1, 3]);
    Ok(())
}

#[test]
fn boolean_raw_should_all_is_restored_once_when_it_supplies_matching()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1); 4];
    let encoded = EncodedDocLenSection::encode(0, 4, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 4, 4)?;
    let term_score = expected_term_score(
        &snapshot,
        1,
        field.fieldnorm_id(1).expect("fieldnorm exists"),
        1,
        1.0,
    );
    let expected_overlap = term_score + 1.0;

    let mut query = ReferenceScorer::boolean(vec![
        ScorerClause::should(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
        ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
    ])?;
    assert_eq!(query.doc(), Some(0));
    assert_eq!(query.score()?.to_bits(), 1.0_f32.to_bits());
    assert_eq!(query.next()?, Some(1));
    assert_eq!(query.score()?.to_bits(), expected_overlap.to_bits());
    Ok(())
}

#[test]
fn boolean_duplicate_raw_all_collapses_to_one_unit_score() -> Result<(), Box<dyn std::error::Error>>
{
    let mut required = ReferenceScorer::boolean(vec![
        ScorerClause::must(ReferenceScorer::all(0, 4, 4)?),
        ScorerClause::must(ReferenceScorer::all(0, 4, 4)?),
    ])?;
    assert_eq!(required.score()?.to_bits(), 1.0_f32.to_bits());

    let mut optional = ReferenceScorer::boolean(vec![
        ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
        ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
    ])?;
    assert_eq!(optional.score()?.to_bits(), 1.0_f32.to_bits());

    let mut unscored = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
        ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
    ])?;
    assert_eq!(unscored.cost(), 4);
    assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![0, 1, 2, 3]);
    Ok(())
}

#[test]
fn boolean_raw_all_exclusion_lowers_to_empty_in_both_modes()
-> Result<(), Box<dyn std::error::Error>> {
    let mut scored = ReferenceScorer::boolean(vec![
        ScorerClause::must(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
        ScorerClause::must_not(ReferenceScorer::all(0, 4, 4)?),
    ])?;
    assert_eq!(scored.doc(), None);
    assert_eq!(scored.cost(), 0);
    assert_eq!(scored.size_hint(), 0);
    assert_eq!(scored.segment_num_docs(), 0);
    assert!(scored.top_k(4, &AllLiveDocs)?.is_empty());

    let mut unscored = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::must(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
        ScorerClause::must_not(ReferenceScorer::all(0, 4, 4)?),
    ])?;
    assert_eq!(unscored.doc(), None);
    assert_eq!(unscored.cost(), 0);
    assert_eq!(unscored.size_hint(), 0);
    assert_eq!(unscored.segment_num_docs(), 0);
    assert!(unscored.collect_doc_set(&AllLiveDocs)?.is_empty());
    Ok(())
}

#[test]
fn boolean_boosted_all_is_not_elided_from_score_bits() -> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1); 4];
    let encoded = EncodedDocLenSection::encode(0, 4, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 4, 4)?;
    let term_score = expected_term_score(
        &snapshot,
        1,
        field.fieldnorm_id(1).expect("fieldnorm exists"),
        1,
        1.0,
    );

    let mut required = ReferenceScorer::boolean(vec![
        ScorerClause::must(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
        ScorerClause::must(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
    ])?;
    let expected_required = term_score + 2.5;
    assert_eq!(required.doc(), Some(1));
    assert_eq!(required.score()?.to_bits(), expected_required.to_bits());

    let mut optional = ReferenceScorer::boolean(vec![
        ScorerClause::must(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
        ScorerClause::should(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
    ])?;
    let mut expected_optional = 0.0_f32;
    expected_optional += term_score;
    expected_optional += 2.5;
    assert_eq!(optional.doc(), Some(1));
    assert_eq!(optional.score()?.to_bits(), expected_optional.to_bits());

    let mut unscored = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::must(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
        ScorerClause::must(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
    ])?;
    assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![1]);
    Ok(())
}

#[test]
fn basic_record_option_clamps_repeated_edge_ngram_frequency()
-> Result<(), Box<dyn std::error::Error>> {
    // Minimized from the pinned 40-path CASS witness: the target prefix
    // field has 1,775 analyzed tokens, and seven `tokens` source values
    // each emit the `token` edge n-gram. Across the corpus that term has
    // df=16 and the prefix field has 84,416 total tokens.
    let mut lengths = vec![Some(0); 40];
    lengths[39] = Some(1_775);
    let encoded = EncodedDocLenSection::encode(0, 40, &[9], &[DocLenFieldInput::new(9, &lengths)])?;
    let section = encoded.section(&[9])?;
    let field = section.field(9).expect("content_prefix field exists");
    let snapshot = snapshot(9, 84_416, 40)?;
    let scorer = |record_option| {
        TermScorer::new(
            VecCursor::new(vec![Posting::new(39, 7)], 1, 40),
            field,
            snapshot.clone(),
            16,
            record_option,
            1.0,
        )
    };

    let basic = scorer(TermRecordOption::Basic)?;
    let with_freqs = scorer(TermRecordOption::WithFreqs)?;
    let with_positions = scorer(TermRecordOption::WithFreqsAndPositions)?;
    let fieldnorm_id = field.fieldnorm_id(39).expect("target fieldnorm exists");
    let expected_basic = expected_term_score(&snapshot, 16, fieldnorm_id, 1, 1.0);
    let expected_frequency = expected_term_score(&snapshot, 16, fieldnorm_id, 7, 1.0);

    assert_eq!(basic.score()?.to_bits(), expected_basic.to_bits());
    assert_eq!(with_freqs.score()?.to_bits(), expected_frequency.to_bits());
    assert_eq!(
        with_positions.score()?.to_bits(),
        expected_frequency.to_bits(),
        "frequency-bearing record options must retain the raw tf"
    );
    assert_eq!(
        (with_freqs.score()? - basic.score()?).to_bits(),
        0x3f41_a57a,
        "the minimized Basic-vs-frequency leaf delta is 0.756431222"
    );

    let maximum = TermScorer::new(
        VecCursor::new(vec![Posting::new(39, u32::MAX)], 1, 40),
        field,
        snapshot.clone(),
        16,
        TermRecordOption::Basic,
        1.0,
    )?;
    assert_eq!(
        maximum.score()?.to_bits(),
        expected_basic.to_bits(),
        "Basic scoring must remain presence-only at maximum stored frequency"
    );

    let bound_rows = vec![Posting::new(39, 7)];
    let (encoded_postings, encoded_block_max) =
        EncodedPostingList::encode_with_block_max(&bound_rows, |doc| {
            field.fieldnorm_id(u64::from(doc))
        })?;
    let posting_list = encoded_postings.posting_list()?;
    let bounds = encoded_block_max.block_max_list(&posting_list, field)?;
    let bounds = Arc::<[BlockMaxEntry]>::from(bounds.entries());
    let bounded = |record_option| {
        TermScorer::new(
            SealedPostingCursor::with_block_max(
                posting_list.cursor()?,
                Arc::clone(&bounds),
                posting_list.doc_freq(),
                40,
            ),
            field,
            snapshot.clone(),
            16,
            record_option,
            1.0,
        )
    };
    for (record_option, expected) in [
        (TermRecordOption::Basic, expected_basic),
        (TermRecordOption::WithFreqs, expected_frequency),
        (TermRecordOption::WithFreqsAndPositions, expected_frequency),
    ] {
        let scorer = bounded(record_option)?;
        assert_eq!(
            scorer
                .term_score_upper_bound
                .expect("sealed term bound")
                .to_bits(),
            expected.to_bits(),
            "{record_option:?} whole-term bound must use its effective frequency"
        );
        let competitive = scorer.competitive_cursor().expect("competitive cursor");
        assert_eq!(
            competitive
                .current_block_upper_bound()
                .expect("current block bound")
                .to_bits(),
            expected.to_bits(),
            "{record_option:?} current-block bound must use its effective frequency"
        );
    }
    Ok(())
}

#[test]
fn term_scorer_rejects_nonprogress_backward_seek_and_resurrection()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1); 8];
    let encoded = EncodedDocLenSection::encode(0, 8, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 8, 8)?;
    let scorer = |fault| {
        TermScorer::new(
            FaultCursor::new(fault),
            field,
            snapshot.clone(),
            2,
            TermRecordOption::WithFreqs,
            1.0,
        )
    };

    assert!(
        PostingCursor::validated_post_move_contract(&FaultCursor::new(CursorFault::StickyNext))
            .is_none()
    );

    let mut sticky = scorer(CursorFault::StickyNext)?;
    assert!(matches!(sticky.next(), Err(ArgusError::CursorInvariant(_))));
    let mut backward = scorer(CursorFault::BackwardNext)?;
    assert!(matches!(
        backward.next(),
        Err(ArgusError::CursorInvariant(_))
    ));
    let mut below_target = scorer(CursorFault::BelowTargetAdvance)?;
    assert!(matches!(
        below_target.seek(4),
        Err(ArgusError::CursorInvariant(_))
    ));
    let mut moved = scorer(CursorFault::MovesDespiteSatisfiedTarget)?;
    assert!(matches!(moved.seek(4), Err(ArgusError::CursorInvariant(_))));
    let mut resurrecting = scorer(CursorFault::ResurrectsAfterExhaustion)?;
    assert_eq!(resurrecting.next()?, None);
    assert!(matches!(
        resurrecting.next(),
        Err(ArgusError::CursorInvariant(_))
    ));

    assert!(matches!(
        TermScorer::new(
            VecCursor::new(Vec::new(), 1, 8),
            field,
            snapshot,
            2,
            TermRecordOption::WithFreqs,
            1.0,
        ),
        Err(ArgusError::CursorInvariant(_))
    ));
    Ok(())
}

#[test]
fn sealed_cursor_adapter_preserves_option_state_and_max_docid() -> Result<(), ArgusError> {
    let source = [
        Posting::new(3, 2),
        Posting::new(130, 1),
        Posting::new(u32::MAX, 7),
    ];
    let encoded = EncodedPostingList::encode(&source)?;
    let list = encoded.posting_list()?;
    let expected_positions = [1, 4, 9, 0, 1, 2, 3, 4, 5, 6];
    let encoded_positions = EncodedPositionList::encode(&source, &expected_positions)?;
    let positions = encoded_positions.position_list(&list)?;
    let mut cursor = SealedPostingCursor::with_positions(&positions, 3)?;

    assert!(PostingCursor::validated_post_move_contract(&cursor).is_some());
    assert_eq!(PostingCursor::doc(&cursor), Some(3));
    assert_eq!(PostingCursor::freq(&cursor), Some(2));
    let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
    assert_eq!(handle.posting_ordinal(), 0);
    let mut decoded_positions = vec![u32::MAX];
    handle.decode_into(&mut decoded_positions)?;
    assert_eq!(decoded_positions, vec![1, 4]);
    assert_eq!(PostingCursor::advance(&mut cursor, 3)?, Some(3));
    assert_eq!(PostingCursor::advance(&mut cursor, 4)?, Some(130));
    let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
    assert_eq!(handle.posting_ordinal(), 1);
    handle.decode_into(&mut decoded_positions)?;
    assert_eq!(decoded_positions, vec![9]);
    assert_eq!(
        PostingCursor::advance(&mut cursor, u32::MAX)?,
        Some(u32::MAX)
    );
    assert_eq!(PostingCursor::freq(&cursor), Some(7));
    let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
    assert_eq!(handle.posting_ordinal(), 2);
    handle.decode_into(&mut decoded_positions)?;
    assert_eq!(decoded_positions, vec![0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(PostingCursor::next(&mut cursor)?, None);
    assert_eq!(PostingCursor::next(&mut cursor)?, None);
    assert_eq!(PostingCursor::doc(&cursor), None);
    assert_eq!(PostingCursor::freq(&cursor), None);
    assert!(PostingCursor::positions_handle(&cursor).is_none());

    let cursor = SealedPostingCursor::new(&list, 3)?;
    assert!(PostingCursor::validated_post_move_contract(&cursor).is_some());
    assert_eq!(PostingCursor::doc(&cursor), Some(3));
    assert!(PostingCursor::positions_handle(&cursor).is_none());
    Ok(())
}

#[cfg(feature = "bench-internals")]
#[test]
fn forced_post_move_validation_control_preserves_sealed_cursor_scores()
-> Result<(), Box<dyn std::error::Error>> {
    let postings = [Posting::new(0, 2), Posting::new(1, 1), Posting::new(2, 3)];
    let encoded_postings = EncodedPostingList::encode(&postings)?;
    let posting_list = encoded_postings.posting_list()?;
    let lengths = [Some(8); 3];
    let encoded_doclens =
        EncodedDocLenSection::encode(0, 3, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let doclens = encoded_doclens.section(&[1])?;
    let fieldnorms = doclens.field(1).expect("fixture field exists");
    let snapshot = snapshot(1, 24, 3)?;

    let scores = |trusted_post_move_contract| -> Result<Vec<(u32, u32)>, ArgusError> {
        let cursor = SealedPostingCursor::new(&posting_list, 3)?;
        let cursor: Box<dyn PostingCursor> = if trusted_post_move_contract {
            Box::new(cursor)
        } else {
            Box::new(ForcedPostMoveValidationCursor { inner: cursor })
        };
        assert_eq!(
            cursor.validated_post_move_contract().is_some(),
            trusted_post_move_contract,
            "the control must select the intended post-move contract"
        );
        let mut scorer = TermScorer::new(
            cursor,
            fieldnorms,
            snapshot.clone(),
            3,
            TermRecordOption::WithFreqs,
            1.0,
        )?;
        let mut observed = Vec::new();
        while let Some(doc) = scorer.doc() {
            observed.push((doc, scorer.score()?.to_bits()));
            scorer.next()?;
        }
        Ok(observed)
    };

    assert_eq!(scores(true)?, scores(false)?);
    Ok(())
}
