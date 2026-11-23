use super::*;

#[test]
fn numeric_range_leaf_has_constant_score_and_fused_cursor_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let entries = [
        NumericEntry::i64(-5, 100),
        NumericEntry::i64(0, 101),
        NumericEntry::i64(5, 102),
        NumericEntry::i64(5, 103),
        NumericEntry::i64(10, 104),
        NumericEntry::i64(20, 106),
    ];
    let fields = [NumericFieldInput::new(0, &entries)];
    let encoded = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 100, 108, &fields)?;
    let field = encoded
        .section()?
        .field(0)
        .expect("created_at NUMERIC field");

    let mut range = ReferenceScorer::numeric_range(
        field,
        Bound::Included(NumericValue::I64(5)),
        Bound::Excluded(NumericValue::I64(10)),
        8,
        8,
    )?;
    assert_eq!(range.doc(), Some(102));
    assert_eq!(range.cost(), 6, "Tantivy RangeDocSet uses floor(0.8N)");
    assert_eq!(range.size_hint(), 0, "Tantivy range hint is floor(N/10)");
    assert_eq!(range.score()?.to_bits(), 1.0_f32.to_bits());
    assert_eq!(range.advance(103)?, Some(103));
    assert_eq!(range.score()?.to_bits(), 1.0_f32.to_bits());
    assert_eq!(range.next()?, None);
    assert_eq!(range.next()?, None, "exhaustion is fused");
    assert!(matches!(
        range.score(),
        Err(ArgusError::CursorInvariant(
            "cannot score an exhausted numeric range"
        ))
    ));

    let mut boosted = ReferenceScorer::numeric_range_with_boost(
        field,
        Bound::Included(NumericValue::I64(5)),
        Bound::Excluded(NumericValue::I64(10)),
        8,
        8,
        2.5,
    )?;
    assert_eq!(boosted.score()?.to_bits(), 2.5_f32.to_bits());

    let inner = ReferenceScorer::numeric_range(
        field,
        Bound::Included(NumericValue::I64(5)),
        Bound::Excluded(NumericValue::I64(10)),
        8,
        8,
    )?;
    let mut replaced = ReferenceScorer::constant_score(inner, 3.25)?;
    assert_eq!(replaced.score()?.to_bits(), 3.25_f32.to_bits());
    assert!(matches!(
        ReferenceScorer::constant_score(replaced, f32::NAN),
        Err(ArgusError::InvalidConstantBoost { boost_bits })
            if boost_bits == f32::NAN.to_bits()
    ));

    let overlap = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::should(ReferenceScorer::numeric_range(
            field,
            Bound::Unbounded,
            Bound::Included(NumericValue::I64(5)),
            8,
            8,
        )?),
        ScorerClause::should(ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::I64(5)),
            Bound::Included(NumericValue::I64(10)),
            8,
            8,
        )?),
    ])?;
    let mut deduplicated = ReferenceScorer::constant_score(overlap, 4.0)?;
    assert_eq!(deduplicated.cost(), 5, "overlapping matches count once");
    assert_eq!(deduplicated.size_hint(), 5);
    assert_eq!(deduplicated.score()?.to_bits(), 4.0_f32.to_bits());
    assert_eq!(
        deduplicated.collect_doc_set(&AllLiveDocs)?,
        vec![100, 101, 102, 103, 104]
    );

    let high = ReferenceScorer::constant_score(
        ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::I64(0)),
            Bound::Included(NumericValue::I64(0)),
            8,
            8,
        )?,
        1.0e8,
    )?;
    let low = ReferenceScorer::constant_score(
        ReferenceScorer::numeric_range(
            field,
            Bound::Unbounded,
            Bound::Included(NumericValue::I64(5)),
            8,
            8,
        )?,
        -1.0e8,
    )?;
    let unit = ReferenceScorer::constant_score(
        ReferenceScorer::numeric_range(
            field,
            Bound::Unbounded,
            Bound::Included(NumericValue::I64(10)),
            8,
            8,
        )?,
        1.0,
    )?;
    let mut cancellation = ReferenceScorer::boolean(vec![
        ScorerClause::must(unit),
        ScorerClause::must(low),
        ScorerClause::must(high),
    ])?;
    assert_eq!(cancellation.doc(), Some(101));
    assert_eq!(
        cancellation.score()?.to_bits(),
        1.0_f32.to_bits(),
        "unique-cardinality costs pin Tantivy's stable Must score order"
    );

    let mut unscored = ReferenceScorer::numeric_range(
        field,
        Bound::Unbounded,
        Bound::Included(NumericValue::I64(5)),
        8,
        8,
    )?;
    assert_eq!(
        unscored.collect_doc_set(&AllLiveDocs)?,
        vec![100, 101, 102, 103]
    );

    let empty = ReferenceScorer::numeric_range(
        field,
        Bound::Included(NumericValue::I64(30)),
        Bound::Included(NumericValue::I64(40)),
        8,
        8,
    )?;
    assert_eq!(empty.doc(), None);
    assert!(matches!(
        ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::U64(0)),
            Bound::Unbounded,
            8,
            8
        ),
        Err(ArgusError::Numeric(NumericCodecError::BoundTypeMismatch {
            field_ord: 0,
            ..
        }))
    ));
    Ok(())
}

#[test]
fn numeric_range_preserves_oracle_cost_order_and_u32_max() -> Result<(), Box<dyn std::error::Error>>
{
    let sparse_entries = [NumericEntry::i64(7, 0)];
    let sparse_fields = [NumericFieldInput::new(0, &sparse_entries)];
    let numeric = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 0, 10, &sparse_fields)?;
    let numeric_field = numeric
        .section()?
        .field(0)
        .expect("created_at NUMERIC field");
    let range = ReferenceScorer::numeric_range(
        numeric_field,
        Bound::Included(NumericValue::I64(7)),
        Bound::Included(NumericValue::I64(7)),
        10,
        10,
    )?;
    assert_eq!(
        range.cost(),
        8,
        "one exact match still has oracle range cost"
    );

    let lengths = [Some(1); 10];
    let encoded_lengths =
        EncodedDocLenSection::encode(0, 10, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let length_section = encoded_lengths.section(&[1])?;
    let fieldnorms = length_section.field(1).expect("fieldnorm column");
    let stats = snapshot(1, 10, 10)?;
    let positive = term(postings(&[0]), fieldnorms, &stats, 1, 1, 1.0e8)?;
    let negative = term(postings(&[0]), fieldnorms, &stats, 1, 2, -1.0e8)?;
    let mut intersection = ReferenceScorer::boolean(vec![
        ScorerClause::must(range),
        ScorerClause::must(positive),
        ScorerClause::must(negative),
    ])?;
    assert_eq!(intersection.doc(), Some(0));
    assert_eq!(intersection.cost(), 1);
    assert_eq!(
        intersection.score()?.to_bits(),
        1.0_f32.to_bits(),
        "terms cancel before the range's constant score is accumulated"
    );

    let high_entries = [
        NumericEntry::i64(0, u32::MAX - 1),
        NumericEntry::i64(1, u32::MAX),
    ];
    let high_fields = [NumericFieldInput::new(0, &high_entries)];
    let high_numeric = EncodedNumericSection::encode(
        RANGE_TEST_SCHEMA,
        u64::from(u32::MAX) - 1,
        u64::from(u32::MAX) + 1,
        &high_fields,
    )?;
    let high_field = high_numeric
        .section()?
        .field(0)
        .expect("high created_at NUMERIC field");
    let mut high =
        ReferenceScorer::numeric_range(high_field, Bound::Unbounded, Bound::Unbounded, 2, 2)?;
    assert_eq!(
        high.cost(),
        2,
        "full coverage lowers like Tantivy AllScorer"
    );
    assert_eq!(high.size_hint(), 2);
    assert_eq!(high.advance(u32::MAX)?, Some(u32::MAX));
    assert_eq!(high.score()?.to_bits(), 1.0_f32.to_bits());
    assert_eq!(high.next()?, None);
    Ok(())
}

#[test]
fn gh49_numeric_range_shares_the_live_domain_after_a_tombstone()
-> Result<(), Box<dyn std::error::Error>> {
    // A sealed segment that has acquired a tombstone has two counts: the
    // persisted NUMERIC column still holds one value per at-seal row,
    // while term and `All` leaves are built on the live count. Supplying
    // the at-seal count for both put the range leaf in a different segment
    // domain from its Boolean siblings, and `shared_segment_num_docs`
    // rejected the whole query with "Boolean children belong to different
    // segment domains" — surfacing to CASS as BACKEND_UNAVAILABLE for an
    // ordinary filtered search (#49).
    let entries = [
        NumericEntry::i64(0, 0),
        NumericEntry::i64(1, 1),
        NumericEntry::i64(2, 2),
    ];
    let fields = [NumericFieldInput::new(0, &entries)];
    let encoded = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 0, 3, &fields)?;
    let field = encoded
        .section()?
        .field(0)
        .expect("created_at NUMERIC field");

    // Three rows at seal, one since tombstoned.
    let at_seal_num_docs = 3;
    let live_num_docs = 2;

    let range = ReferenceScorer::numeric_range(
        field,
        Bound::Included(NumericValue::I64(0)),
        Bound::Included(NumericValue::I64(2)),
        at_seal_num_docs,
        live_num_docs,
    )?;
    let all = ReferenceScorer::all_with_boost(0, 3, live_num_docs, 1.0)?;

    // The combination is what failed: each leaf is fine alone.
    let mut combined =
        ReferenceScorer::boolean(vec![ScorerClause::must(all), ScorerClause::must(range)])?;
    let hits = combined.top_k(3, &AllLiveDocs)?;
    assert!(
        !hits.is_empty(),
        "a filtered query over a tombstoned segment must still return results"
    );

    // And the column is still validated against its own domain: the live
    // count must not be able to make a legitimate column look corrupt.
    assert!(
        ReferenceScorer::numeric_range(
            field,
            Bound::Unbounded,
            Bound::Unbounded,
            at_seal_num_docs,
            live_num_docs,
        )
        .is_ok(),
        "three values over three at-seal rows is valid however few are live"
    );
    Ok(())
}

#[test]
fn numeric_ranges_preserve_boolean_multiplicity_exclusion_and_live_filtering()
-> Result<(), Box<dyn std::error::Error>> {
    let entries = [
        NumericEntry::i64(0, 0),
        NumericEntry::i64(1, 1),
        NumericEntry::i64(2, 2),
    ];
    let fields = [NumericFieldInput::new(0, &entries)];
    let encoded = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 0, 3, &fields)?;
    let field = encoded
        .section()?
        .field(0)
        .expect("created_at NUMERIC field");
    let left = || {
        ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::I64(0)),
            Bound::Included(NumericValue::I64(1)),
            3,
            3,
        )
    };
    let right = || {
        ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::I64(1)),
            Bound::Included(NumericValue::I64(2)),
            3,
            3,
        )
    };
    let all = || ReferenceScorer::numeric_range(field, Bound::Unbounded, Bound::Unbounded, 3, 3);

    let mut required_full_range = ReferenceScorer::boolean(vec![
        ScorerClause::must(all()?),
        ScorerClause::must(left()?),
    ])?;
    let mut required_full_range_hits = required_full_range.top_k(3, &AllLiveDocs)?;
    required_full_range_hits.sort_unstable_by_key(|hit| hit.global_docid);
    assert_eq!(
        required_full_range_hits
            .iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>(),
        vec![(0, 1.0_f32.to_bits()), (1, 1.0_f32.to_bits())],
        "Tantivy removes a full-cardinality unit range after it lowers to raw AllScorer"
    );

    let mut required_boosted_full_range = ReferenceScorer::boolean(vec![
        ScorerClause::must(ReferenceScorer::numeric_range_with_boost(
            field,
            Bound::Unbounded,
            Bound::Unbounded,
            3,
            3,
            2.5,
        )?),
        ScorerClause::must(left()?),
    ])?;
    let mut required_boosted_hits = required_boosted_full_range.top_k(3, &AllLiveDocs)?;
    required_boosted_hits.sort_unstable_by_key(|hit| hit.global_docid);
    assert_eq!(
        required_boosted_hits
            .iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>(),
        vec![(0, 3.5_f32.to_bits()), (1, 3.5_f32.to_bits())],
        "a boosted full-cardinality range remains score-bearing"
    );

    let mut should = ReferenceScorer::boolean(vec![
        ScorerClause::should(left()?),
        ScorerClause::should(right()?),
    ])?;
    let mut hits = should.top_k(3, &AllLiveDocs)?;
    hits.sort_unstable_by_key(|hit| hit.global_docid);
    assert_eq!(
        hits.iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            (0, 1.0_f32.to_bits()),
            (1, 2.0_f32.to_bits()),
            (2, 1.0_f32.to_bits()),
        ],
        "overlapping constant-score ranges retain one contribution per child"
    );

    let mut excluded = ReferenceScorer::boolean(vec![
        ScorerClause::must(all()?),
        ScorerClause::must_not(ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::I64(1)),
            Bound::Included(NumericValue::I64(1)),
            3,
            3,
        )?),
    ])?;
    let excluded_hits = excluded.top_k(3, &AllLiveDocs)?;
    assert_eq!(
        excluded_hits
            .iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>(),
        vec![(0, 1.0_f32.to_bits()), (2, 1.0_f32.to_bits())],
        "MustNot gates matches without adding a score"
    );

    let mut unscored = ReferenceScorer::boolean_unscored(vec![
        ScorerClause::should(left()?),
        ScorerClause::should(right()?),
    ])?;
    assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![0, 1, 2]);
    let mut live_filtered = all()?;
    assert_eq!(
        live_filtered
            .top_k(3, &|docid| docid != 1)?
            .iter()
            .map(|hit| hit.global_docid)
            .collect::<Vec<_>>(),
        vec![0, 2],
        "tombstone filtering happens after the physical range cursor"
    );

    let repeated_doc = [NumericEntry::i64(0, 0), NumericEntry::i64(1, 0)];
    let repeated_fields = [NumericFieldInput::new(0, &repeated_doc)];
    let repeated = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 0, 2, &repeated_fields)?;
    let repeated_field = repeated
        .section()?
        .field(0)
        .expect("multi-valued NUMERIC fixture");
    let ordinary =
        ReferenceScorer::numeric_range(repeated_field, Bound::Unbounded, Bound::Unbounded, 2, 2)?;
    assert_eq!(ordinary.cost(), 1);
    assert_eq!(ordinary.size_hint(), 0);
    // Cardinality is validated against the column's own domain, so the
    // live count deliberately differs here: a column holding more values
    // than at-seal rows is corrupt whatever the snapshot currently shows.
    assert!(matches!(
        ReferenceScorer::numeric_range(repeated_field, Bound::Unbounded, Bound::Unbounded, 1, 2),
        Err(ArgusError::InvalidNumericCardinality {
            field_ord: 0,
            value_count: 2,
            segment_num_docs: 1,
        })
    ));
    Ok(())
}

#[test]
fn numeric_ranges_compose_with_live_domain_after_tombstones()
-> Result<(), Box<dyn std::error::Error>> {
    let entries = [
        NumericEntry::i64(0, 0),
        NumericEntry::i64(1, 1),
        NumericEntry::i64(2, 2),
    ];
    let fields = [NumericFieldInput::new(0, &entries)];
    let encoded = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 0, 3, &fields)?;
    let field = encoded
        .section()?
        .field(0)
        .expect("created_at NUMERIC field");
    let range = ReferenceScorer::numeric_range(
        field,
        Bound::Included(NumericValue::I64(2)),
        Bound::Included(NumericValue::I64(2)),
        3,
        2,
    )?;
    let boosted_range = ReferenceScorer::numeric_range_with_boost(
        field,
        Bound::Unbounded,
        Bound::Unbounded,
        3,
        2,
        2.5,
    )?;
    let mut query = ReferenceScorer::boolean(vec![
        ScorerClause::must(ReferenceScorer::all_with_boost(0, 3, 2, 3.0)?),
        ScorerClause::must(boosted_range),
        ScorerClause::should(range),
    ])?;
    let hits = query.top_k(3, &|docid| docid != 1)?;
    assert_eq!(
        hits.iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>(),
        vec![(2, 6.5_f32.to_bits()), (0, 5.5_f32.to_bits())],
        "physical numeric rows survive validation while Boolean domains use the live count"
    );
    Ok(())
}

#[test]
fn tombstones_keep_physical_stats_but_never_enter_heap_or_materializer()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(2), Some(2)];
    let encoded = EncodedDocLenSection::encode(0, 2, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let pre_compaction = snapshot(1, 4, 2)?;
    let mut query = term(postings(&[0, 1]), field, &pre_compaction, 2, 2, 1.0)?;
    let live = |doc| doc == 1;
    let winners = query.top_k(10, &live)?;
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0].global_docid, 1);
    let expected = expected_term_score(&pre_compaction, 2, fieldnorm_to_id(2), 1, 1.0);
    assert_eq!(winners[0].score.to_bits(), expected.to_bits());

    let mut calls = Vec::new();
    let hits = materialize_doc_ids(&winners, |doc| {
        calls.push(doc);
        Ok(format!("external-{doc}").into())
    })?;
    assert_eq!(calls, vec![1]);
    assert_eq!(hits[0].document_id, "external-1");

    let compacted = snapshot(1, 2, 1)?;
    let compacted_score = expected_term_score(&compacted, 1, fieldnorm_to_id(2), 1, 1.0);
    assert_ne!(expected.to_bits(), compacted_score.to_bits());
    Ok(())
}

#[test]
fn deleted_matches_are_scored_before_live_filtering_to_preserve_union_state()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = vec![Some(1); 20_001];
    let encoded =
        EncodedDocLenSection::encode(0, 20_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let snapshot = snapshot(1, 20_001, 20_001)?;
    let required = term(postings(&[5_000, 10_000]), field, &snapshot, 2, 2, 1.0)?;
    let optional = ReferenceScorer::boolean(vec![
        ScorerClause::should(term(postings(&[0, 9_500]), field, &snapshot, 2, 2, 0.0)?),
        ScorerClause::should(term(
            postings(&[0, 10_000, 20_000]),
            field,
            &snapshot,
            3,
            2,
            -1.0e8,
        )?),
        ScorerClause::should(term(postings(&[0, 5_000]), field, &snapshot, 2, 2, 0.0)?),
        ScorerClause::should(term(
            postings(&[0, 10_000, 20_000]),
            field,
            &snapshot,
            3,
            2,
            1.0e8,
        )?),
        ScorerClause::should(term(
            postings(&[0, 10_000, 20_000]),
            field,
            &snapshot,
            3,
            2,
            1.0,
        )?),
    ])?;
    let mut query = ReferenceScorer::boolean(vec![
        ScorerClause::must(required),
        ScorerClause::should(optional),
    ])?;
    let winners = query.top_k(1, &|doc| doc == 10_000)?;
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0].global_docid, 10_000);

    let required_score = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
    let optional_score = expected_term_score(&snapshot, 3, fieldnorm_to_id(1), 1, 1.0);
    let expected = required_score + optional_score;
    assert_eq!(winners[0].score.to_bits(), expected.to_bits());
    assert_ne!(winners[0].score.to_bits(), required_score.to_bits());
    Ok(())
}

#[test]
fn malformed_scoring_inputs_fail_without_reaching_idf_assertion()
-> Result<(), Box<dyn std::error::Error>> {
    let lengths = [Some(1)];
    let encoded = EncodedDocLenSection::encode(0, 1, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
    let section = encoded.section(&[1])?;
    let field = section.field(1).expect("field exists");
    let valid = snapshot(1, 1, 1)?;

    assert!(matches!(
        TermScorer::new(
            VecCursor::new(postings(&[0]), 1, 1),
            field,
            valid.clone(),
            2,
            TermRecordOption::WithFreqs,
            1.0
        ),
        Err(ArgusError::InvalidDocFrequency {
            doc_freq: 2,
            doc_count: 1
        })
    ));
    let empty_field = snapshot(1, 0, 1)?;
    assert!(matches!(
        TermScorer::new(
            VecCursor::new(postings(&[0]), 1, 1),
            field,
            empty_field,
            1,
            TermRecordOption::WithFreqs,
            1.0
        ),
        Err(ArgusError::InvalidSnapshot { .. })
    ));
    assert!(matches!(
        TermScorer::new(
            VecCursor::new(postings(&[0]), 1, 1),
            field,
            valid,
            1,
            TermRecordOption::WithFreqs,
            f32::NAN
        ),
        Err(ArgusError::InvalidBoost { .. })
    ));
    Ok(())
}

#[test]
fn e410_zero_document_snapshot_has_no_average_or_tf_division() -> Result<(), ArgusError> {
    let empty = snapshot(1, 0, 0)?;
    assert_eq!(empty.doc_count(), 0);
    assert_eq!(empty.total_tokens(), 0);
    assert_eq!(empty.average_field_length(), None);
    assert!(empty.tf_cache.iter().all(|value| value.to_bits() == 0));
    assert!(matches!(
        snapshot(1, 1, 0),
        Err(ArgusError::InvalidSnapshot {
            field_ord: 1,
            reason: "an empty snapshot cannot contain field tokens",
        })
    ));
    Ok(())
}
