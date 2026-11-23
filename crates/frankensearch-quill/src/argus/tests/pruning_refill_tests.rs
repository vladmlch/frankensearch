use super::*;

/// The drained-group pin (`bd-bt2t`). An ESSENTIAL group whose postings all
/// fall inside one `UNION_HORIZON` window drains to an empty `active` and
/// carries its remaining documents as buffered residue instead. Candidate
/// discovery must enumerate that residue; enumerating only `active` reports
/// the group as having no documents in the window and the root silently
/// drops them from the top-k.
///
/// This case CANNOT be pinned by `grouped_max_score_matches_exhaustive_and_prunes`:
/// that fixture runs at k=1 where the dropped documents tie the incumbent
/// at exactly `2e8` (the ~1.0 contribution is below the f32 ULP of ~16 at
/// that magnitude) and lose the ascending-docid tie-break anyway — so
/// deleting them changes nothing observable, and a broken fix passes. Here
/// the drained group's documents must WIN the top-k outright, so losing
/// them changes both the ranking and the hit count.
#[test]
fn grouped_max_score_drained_essential_group_keeps_its_documents()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 8_192;
    let lengths = vec![Some(1); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(NUM_DOCS), u64::from(NUM_DOCS))?;
    // group0 = terms {A, B, C}, group1 = terms {D, E}.
    //   A holds the early docs 0/1/2 and fills the k=3 heap in window 0,
    //     establishing a cutoff that only group0 can beat.
    //   B and C hold 5000/5001/5002. When the root advances group0 past the
    //     first horizon, group0 refills onto [5000, 9096) where B and C both
    //     exhaust inside the window: active EMPTY, current = 5000, residue
    //     = {5001, 5002}. That is the state under test.
    //   D and E (docs 6000/6001, unit boost) keep group1 non-essential, so
    //     the window genuinely prunes rather than degenerating to exhaustive.
    let rows_by_term = vec![
        vec![Posting::new(0, 1), Posting::new(1, 1), Posting::new(2, 1)],
        vec![
            Posting::new(5_000, 1),
            Posting::new(5_001, 1),
            Posting::new(5_002, 1),
        ],
        vec![
            Posting::new(5_000, 1),
            Posting::new(5_001, 1),
            Posting::new(5_002, 1),
        ],
        vec![Posting::new(6_000, 1)],
        vec![Posting::new(6_001, 1)],
    ];
    let boosts = [1.0e8, 1.0e8, 1.0e8, 1.0, 1.0];
    let group_sizes = [3_usize, 2_usize];
    let fieldnorm_id = fieldnorm_to_id(1);
    let encoded_terms = rows_by_term
        .iter()
        .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
        .collect::<Result<Vec<_>, _>>()?;
    let posting_lists = encoded_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

    let mut oracle = sealed_grouped_union(
        &posting_lists,
        None,
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
        &group_sizes,
    )?;
    let mut oracle_collector = TopDocsCollector::new(3, 0)?;
    oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
    let oracle_hits = oracle_collector.finish()?.hits;

    let mut candidate = sealed_grouped_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
        &group_sizes,
    )?;
    let mut candidate_collector = TopDocsCollector::new(3, 0)?;
    candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
    let stats = candidate
        .union_pruning_stats()
        .expect("top-level union retains pruning stats");
    let candidate_hits = candidate_collector.finish()?.hits;

    // The drained group's documents outscore the heap incumbents 2:1, so
    // losing the residue would change the ranking AND shrink the hit count.
    assert_eq!(
        oracle_hits
            .iter()
            .map(|hit| hit.global_docid)
            .collect::<Vec<_>>(),
        vec![5_000, 5_001, 5_002],
        "exhaustive top-3 must be the drained group's documents",
    );
    assert_hits_bit_exact(&candidate_hits, &oracle_hits);
    assert!(
        stats.max_score_windows >= 1,
        "grouped MaxScore should prune at least one window (got {stats:?})"
    );
    Ok(())
}

#[test]
fn grouped_max_score_matches_exhaustive_and_prunes() -> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 8_192;
    let lengths = vec![Some(32); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(NUM_DOCS) * 32, u64::from(NUM_DOCS))?;
    // Two multi-field groups of two terms. group0 (high ceiling) is active
    // at doc 0 (window 0) and doc 5_000 (window 1), so it stays *essential*;
    // group1 (low ceiling) holds doc 5_000 and — uniquely — doc 6_000. With
    // k=1 the heap fills on doc 0 (score ~2e8), so in window 1 group1 is
    // non-essential and doc 6_000 (present only in group1) is skipped while
    // the true top-1 (doc 0) is preserved bit-for-bit.
    let rows_by_term = vec![
        vec![Posting::new(0, 7), Posting::new(5_000, 7)],
        vec![Posting::new(0, 11), Posting::new(5_000, 11)],
        vec![Posting::new(5_000, 5)],
        vec![Posting::new(6_000, 17)],
    ];
    let boosts = [1.0e8, 1.0e8, 1.0, 1.0];
    let group_sizes = [2_usize, 2_usize];
    let fieldnorm_id = fieldnorm_to_id(32);
    let encoded_terms = rows_by_term
        .iter()
        .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
        .collect::<Result<Vec<_>, _>>()?;
    let posting_lists = encoded_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

    for record_option in [TermRecordOption::WithFreqs, TermRecordOption::Basic] {
        let mut oracle = sealed_grouped_union_with_record_option(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
            &group_sizes,
            record_option,
        )?;
        let mut oracle_collector = TopDocsCollector::new(1, 0)?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_hits = oracle_collector.finish()?.hits;

        let mut candidate = sealed_grouped_union_with_record_option(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
            &group_sizes,
            record_option,
        )?;
        let mut candidate_collector = TopDocsCollector::new(1, 0)?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let candidate_hits = candidate_collector.finish()?.hits;

        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert_eq!(candidate_hits[0].global_docid, 0);
        assert!(
            stats.max_score_windows >= 1,
            "{record_option:?} grouped MaxScore should prune (got {stats:?})"
        );
    }
    Ok(())
}

/// Deliberately broken historical control: after draining the initial
/// exhaustive buffer, attempt exactly one competitive refill and stop when
/// that window contains no candidates. Production must keep refilling
/// because a later horizon can still contain an essential-term winner.
fn collect_until_first_empty_competitive_refill_for_negative_control(
    union: &mut BufferedUnionScorer<'_>,
    collector: &mut TopDocsCollector,
) -> Result<(), ArgusError> {
    while let Some(doc) = union.doc() {
        collector.record_live(doc, Some(union.score()?))?;
        if union.advance_buffered().is_none() {
            break;
        }
    }
    let cutoff = collector.competitive_cutoff_score();
    if union.refill_with_cutoff(cutoff)? {
        assert!(
            union.advance_buffered().is_none(),
            "negative-control fixture requires an empty first competitive window",
        );
    }
    Ok(())
}

/// Conservative control that keeps applying the first legitimate
/// competitive cutoff while still allowing the real collector heap to
/// improve. The retained cutoff can admit extra work but cannot exclude a
/// document admitted by a later, higher cutoff.
fn collect_with_first_competitive_cutoff_for_control(
    union: &mut BufferedUnionScorer<'_>,
    collector: &mut TopDocsCollector,
) -> Result<(UnionPruningStats, Vec<u32>), ArgusError> {
    let mut first_cutoff = None;
    let mut refreshed_cutoff_bits = Vec::new();
    while let Some(doc) = union.doc() {
        collector.record_live(doc, Some(union.score()?))?;
        if union.advance_buffered().is_some() {
            continue;
        }
        loop {
            let refreshed_cutoff = collector.competitive_cutoff_score();
            if let Some(refreshed_cutoff) = refreshed_cutoff {
                first_cutoff.get_or_insert(refreshed_cutoff);
            }
            if !union.refill_with_cutoff(first_cutoff.or(refreshed_cutoff))? {
                return Ok((union.pruning_stats, refreshed_cutoff_bits));
            }
            if let Some(refreshed_cutoff) = refreshed_cutoff {
                refreshed_cutoff_bits.push(refreshed_cutoff.to_bits());
            }
            if union.advance_buffered().is_some() {
                break;
            }
        }
    }
    Ok((union.pruning_stats, refreshed_cutoff_bits))
}

#[test]
fn max_score_empty_competitive_window_refills_again_for_late_winner()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 9_001;
    const FIRST_HORIZON_END: u32 = 4_096;
    const TARGET_DOC: u32 = 9_000;
    let lengths = (0..NUM_DOCS)
        .map(|doc| {
            Some(if doc < FIRST_HORIZON_END {
                2
            } else if doc == TARGET_DOC {
                9
            } else {
                1
            })
        })
        .collect::<Vec<_>>();
    let total_tokens = lengths
        .iter()
        .flatten()
        .map(|length| u64::from(*length))
        .sum::<u64>();
    assert_eq!(total_tokens, 13_105);
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, total_tokens, u64::from(NUM_DOCS))?;
    let rows_by_term = vec![
        (0..NUM_DOCS)
            .map(|doc| Posting::new(doc, 1))
            .collect::<Vec<_>>(),
        (0..FIRST_HORIZON_END)
            .map(|doc| Posting::new(doc, 1))
            .collect::<Vec<_>>(),
        vec![Posting::new(TARGET_DOC, 8)],
    ];
    let boosts = [1.0, 1.0, 1.0];
    let encoded_terms = rows_by_term
        .iter()
        .map(|rows| {
            EncodedPostingList::encode_with_block_max(rows, |doc| {
                field.fieldnorm_id(u64::from(doc))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let posting_lists = encoded_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

    let mut broken = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut broken_collector = TopDocsCollector::new(1, 0)?;
    let ScorerNode::Union(broken_union) = &mut broken.node else {
        panic!("UNION_HORIZON fixture must lower to a buffered union");
    };
    collect_until_first_empty_competitive_refill_for_negative_control(
        broken_union,
        &mut broken_collector,
    )?;
    assert_eq!(broken_union.pruning_stats.max_score_windows, 1);
    assert_eq!(broken_union.pruning_stats.block_max_wand_windows, 0);
    assert_eq!(broken_union.pruning_stats.candidate_docs, 0);
    assert_eq!(broken_union.pruning_stats.blocks_skipped, 0);
    let broken_hits = broken_collector.finish()?.hits;
    assert_eq!(broken_hits[0].global_docid, 0);
    assert_ne!(broken_hits[0].global_docid, TARGET_DOC);

    let mut oracle = sealed_union(
        &posting_lists,
        None,
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut oracle_collector = TopDocsCollector::new(1, 0)?;
    oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
    let oracle_hits = oracle_collector.finish()?.hits;

    let mut candidate = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut candidate_collector = TopDocsCollector::new(1, 0)?;
    candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
    let stats = candidate
        .union_pruning_stats()
        .expect("top-level union retains pruning stats");
    let candidate_hits = candidate_collector.finish()?.hits;
    assert_hits_bit_exact(&candidate_hits, &oracle_hits);
    assert_eq!(candidate_hits[0].global_docid, TARGET_DOC);
    assert_eq!(stats.max_score_windows, 2);
    assert_eq!(stats.block_max_wand_windows, 0);
    assert_eq!(stats.candidate_docs, 1);
    assert_eq!(stats.blocks_skipped, 0);
    Ok(())
}

#[test]
fn block_max_wand_empty_competitive_window_refills_again_for_late_winner()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 9_001;
    const TARGET_DOC: u32 = 9_000;
    const FIELD_LENGTH: u32 = 64;
    assert_eq!(BMW_MIN_CLAUSES, 9, "fixture requires nine direct terms");

    let lengths = vec![Some(FIELD_LENGTH); usize::try_from(NUM_DOCS)?];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).ok_or(ArgusError::CursorInvariant(
        "BMW UNION_HORIZON fixture has no field 1",
    ))?;
    let snapshot = snapshot(
        1,
        u64::from(NUM_DOCS) * u64::from(FIELD_LENGTH),
        u64::from(NUM_DOCS),
    )?;

    // Every term is dense so all nine cursors remain active across the
    // first two 4,096-document horizons and their aggregate cost forces
    // BMW rather than MaxScore. Fixed field lengths isolate frequency as
    // the only score lever: doc 0 establishes the first cutoff, the low-TF
    // middle horizon is entirely skippable, and doc 9,000 is strictly
    // stronger than the incumbent in the terminal horizon.
    let dense_rows = (0..NUM_DOCS)
        .map(|doc| {
            Posting::new(
                doc,
                if doc == 0 {
                    4
                } else if doc == TARGET_DOC {
                    8
                } else {
                    1
                },
            )
        })
        .collect::<Vec<_>>();
    let rows_by_term = (0..BMW_MIN_CLAUSES)
        .map(|_| dense_rows.clone())
        .collect::<Vec<_>>();
    let boosts = vec![1.0; BMW_MIN_CLAUSES];
    let encoded_terms = rows_by_term
        .iter()
        .map(|rows| {
            EncodedPostingList::encode_with_block_max(rows, |doc| {
                field.fieldnorm_id(u64::from(doc))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let posting_lists = encoded_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

    let mut broken = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut broken_collector = TopDocsCollector::new(1, 0)?;
    let ScorerNode::Union(broken_union) = &mut broken.node else {
        return Err(Box::new(ArgusError::CursorInvariant(
            "UNION_HORIZON fixture must lower to a buffered union",
        )));
    };
    collect_until_first_empty_competitive_refill_for_negative_control(
        broken_union,
        &mut broken_collector,
    )?;
    assert_eq!(broken_union.pruning_stats.max_score_windows, 0);
    assert_eq!(broken_union.pruning_stats.block_max_wand_windows, 1);
    assert_eq!(broken_union.pruning_stats.candidate_docs, 0);
    assert!(broken_union.pruning_stats.blocks_skipped > 0);
    assert_eq!(broken_union.conformance_refills.len(), 2);
    assert_eq!(
        broken_union.conformance_refills[1].strategy,
        ConformanceUnionRefillStrategy::BlockMaxWand,
    );
    assert!(broken_union.conformance_refills[1].buffer_empty);
    assert!(broken_union.conformance_refills[1].live_work_remains);
    let broken_hits = broken_collector.finish()?.hits;
    assert_eq!(broken_hits[0].global_docid, 0);
    assert_ne!(broken_hits[0].global_docid, TARGET_DOC);

    let mut oracle = sealed_union(
        &posting_lists,
        None,
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut oracle_collector = TopDocsCollector::new(1, 0)?;
    oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
    let oracle_hits = oracle_collector.finish()?.hits;

    let mut candidate = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut candidate_collector = TopDocsCollector::new(1, 0)?;
    candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
    let ScorerNode::Union(candidate_union) = &candidate.node else {
        return Err(Box::new(ArgusError::CursorInvariant(
            "UNION_HORIZON fixture must lower to a buffered union",
        )));
    };
    let stats = candidate_union.pruning_stats;
    let trace = candidate_union.conformance_refills.clone();
    let candidate_hits = candidate_collector.finish()?.hits;

    assert_hits_bit_exact(&candidate_hits, &oracle_hits);
    assert_eq!(candidate_hits[0].global_docid, TARGET_DOC);
    assert_eq!(stats.max_score_windows, 0);
    assert_eq!(stats.block_max_wand_windows, 2);
    assert!(stats.blocks_skipped > 0);

    assert_eq!(
        trace.len(),
        3,
        "expected one exhaustive and two BMW refills"
    );
    assert_eq!(
        (
            trace[0].ordinal,
            trace[0].window_start,
            trace[0].horizon_end,
            trace[0].strategy,
        ),
        (1, 0, 4_096, ConformanceUnionRefillStrategy::Exhaustive),
    );
    assert!(trace[0].cutoff_bits.is_none());
    assert!(trace[0].candidate_docs > 0);
    assert!(!trace[0].buffer_empty);
    assert!(trace[0].live_work_remains);

    assert_eq!(
        (
            trace[1].ordinal,
            trace[1].window_start,
            trace[1].horizon_end,
            trace[1].strategy,
        ),
        (
            2,
            4_096,
            8_192,
            ConformanceUnionRefillStrategy::BlockMaxWand,
        ),
    );
    assert!(
        trace[1]
            .cutoff_bits
            .is_some_and(|bits| f32::from_bits(bits).is_finite())
    );
    assert_eq!(trace[1].candidate_docs, 0);
    assert!(trace[1].buffer_empty);
    assert!(trace[1].live_work_remains);

    assert_eq!(
        (
            trace[2].ordinal,
            trace[2].window_start,
            trace[2].horizon_end,
            trace[2].strategy,
        ),
        (
            3,
            8_192,
            12_288,
            ConformanceUnionRefillStrategy::BlockMaxWand,
        ),
    );
    assert!(
        trace[2]
            .cutoff_bits
            .is_some_and(|bits| f32::from_bits(bits).is_finite())
    );
    assert!(trace[2].candidate_docs > 0);
    assert!(!trace[2].buffer_empty);
    assert!(!trace[2].live_work_remains);
    Ok(())
}

#[test]
fn older_monotone_cutoff_is_exact_but_admits_more_candidates()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 10_001;
    const FIRST_DOC: u32 = 0;
    const WINNER_DOC: u32 = 5_000;
    const STALE_ONLY_DOC: u32 = 10_000;
    let lengths = vec![Some(1); usize::try_from(NUM_DOCS)?];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(NUM_DOCS), u64::from(NUM_DOCS))?;
    let rows_by_term = vec![
        vec![Posting::new(FIRST_DOC, 1)],
        vec![Posting::new(WINNER_DOC, 1)],
        vec![Posting::new(STALE_ONLY_DOC, 1)],
        vec![Posting::new(STALE_ONLY_DOC, 1)],
    ];
    // The low companion makes direct-term MaxScore admissible without
    // changing the ordering we need to prove:
    //
    // * first cutoff ~= 2.0;
    // * middle winner ~= 4.0;
    // * stale-only document ~= 2.75.
    //
    // With the old cutoff, the 2.5 term remains essential and admits the
    // stale-only document. With the refreshed cutoff, 2.5 + 0.25 is below
    // the heap threshold and both terms are safely non-essential.
    let boosts = [2.0, 4.0, 2.5, 0.25];
    let encoded_terms = rows_by_term
        .iter()
        .map(|rows| {
            EncodedPostingList::encode_with_block_max(rows, |doc| {
                field.fieldnorm_id(u64::from(doc))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let posting_lists = encoded_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

    let mut exhaustive = sealed_union(
        &posting_lists,
        None,
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut exhaustive_collector = TopDocsCollector::new(1, 0)?;
    exhaustive_collector.collect(&mut exhaustive, &AllLiveDocs)?;
    let exhaustive_hits = exhaustive_collector.finish()?.hits;

    let mut refreshed = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut refreshed_collector = TopDocsCollector::new(1, 0)?;
    refreshed_collector.collect(&mut refreshed, &AllLiveDocs)?;
    let refreshed_stats = refreshed
        .union_pruning_stats()
        .expect("top-level refreshed union retains pruning stats");
    let refreshed_hits = refreshed_collector.finish()?.hits;

    let mut fixed = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut fixed_collector = TopDocsCollector::new(1, 0)?;
    let ScorerNode::Union(fixed_union) = &mut fixed.node else {
        return Err(
            std::io::Error::other("cutoff-control fixture must lower to a buffered union").into(),
        );
    };
    let (fixed_stats, refreshed_cutoff_bits) =
        collect_with_first_competitive_cutoff_for_control(fixed_union, &mut fixed_collector)?;
    let fixed_hits = fixed_collector.finish()?.hits;

    assert_hits_bit_exact(&refreshed_hits, &exhaustive_hits);
    assert_hits_bit_exact(&fixed_hits, &exhaustive_hits);
    assert_eq!(exhaustive_hits[0].global_docid, WINNER_DOC);
    assert_eq!(refreshed_stats.max_score_windows, 2);
    assert_eq!(fixed_stats.max_score_windows, 2);
    assert_eq!(refreshed_stats.candidate_docs, 1);
    assert_eq!(fixed_stats.candidate_docs, 2);
    assert!(
        fixed_stats.candidate_docs > refreshed_stats.candidate_docs,
        "older cutoff must be no more selective than the refreshed cutoff",
    );
    assert_eq!(refreshed_cutoff_bits.len(), 2);
    let first_cutoff = f32::from_bits(refreshed_cutoff_bits[0]);
    let second_cutoff = f32::from_bits(refreshed_cutoff_bits[1]);
    assert_eq!(
        second_cutoff.total_cmp(&first_cutoff),
        std::cmp::Ordering::Greater,
        "the real collector cutoff must increase after the middle-horizon winner",
    );
    Ok(())
}

#[test]
fn maxscore_two_pass_preserves_swap_remove_score_bits() -> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 5_001;
    let lengths = vec![Some(1); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(NUM_DOCS), u64::from(NUM_DOCS))?;
    let mut rows_by_term = vec![vec![Posting::new(0, 1)], vec![Posting::new(5_000, 1)]];
    rows_by_term.extend((0..6).map(|_| vec![Posting::new(5_000, 1)]));
    let boosts = [5.0e7, 1.0e8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let fieldnorm_id = fieldnorm_to_id(1);
    let encoded_terms = rows_by_term
        .iter()
        .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
        .collect::<Result<Vec<_>, _>>()?;
    let posting_lists = encoded_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

    let mut oracle = sealed_union(
        &posting_lists,
        None,
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut oracle_collector = TopDocsCollector::new(1, 0)?;
    oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
    let oracle_hits = oracle_collector.finish()?.hits;

    let mut candidate = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut candidate_collector = TopDocsCollector::new(1, 0)?;
    candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
    let stats = candidate
        .union_pruning_stats()
        .expect("top-level union retains pruning stats");
    let candidate_hits = candidate_collector.finish()?.hits;
    assert_hits_bit_exact(&candidate_hits, &oracle_hits);
    assert_eq!(candidate_hits[0].global_docid, 5_000);
    assert_eq!(stats.max_score_windows, 1);
    assert_eq!(stats.block_max_wand_windows, 0);
    assert_eq!(stats.candidate_docs, 1);
    assert_eq!(stats.blocks_skipped, 0);

    let small = expected_term_score(&snapshot, 1, fieldnorm_id, 1, 2.0);
    let huge = expected_term_score(&snapshot, 1, fieldnorm_id, 1, 1.0e8);
    // Every singleton scorer that exhausts at doc 5_000 is replaced in
    // place with the active vector's last scorer. That cascade scores all
    // six small terms before the huge term, rather than parse order's
    // huge-then-small sequence.
    let mut union_order = 0.0_f32;
    for _ in 0..6 {
        union_order += small;
    }
    union_order += huge;
    let mut parse_order = huge;
    for _ in 0..6 {
        parse_order += small;
    }
    assert_ne!(union_order.to_bits(), parse_order.to_bits());
    assert_eq!(candidate_hits[0].score.to_bits(), union_order.to_bits());
    Ok(())
}
