use super::*;

#[cfg(feature = "bench-internals")]
fn timed_encoded_grouped_union(
    encoded_doclens: &EncodedDocLenSection,
    encoded_terms: &[(EncodedPostingList, EncodedBlockMax)],
    cached_metadata: Option<&[Arc<ValidatedTermPruningMetadata>]>,
    snapshot: &Bm25FieldSnapshot,
    rows_by_term: &[Vec<Posting>],
    boosts: &[f32],
    segment_num_docs: u32,
    limit: usize,
    group_size: usize,
    rank_pruning: bool,
) -> Result<(u128, Vec<ScoredDoc>, UnionPruningStats), Box<dyn std::error::Error>> {
    timed_encoded_grouped_union_with_contract(
        encoded_doclens,
        encoded_terms,
        cached_metadata,
        snapshot,
        rows_by_term,
        boosts,
        segment_num_docs,
        limit,
        group_size,
        rank_pruning,
        true,
    )
}

#[cfg(feature = "bench-internals")]
fn timed_encoded_grouped_union_with_contract(
    encoded_doclens: &EncodedDocLenSection,
    encoded_terms: &[(EncodedPostingList, EncodedBlockMax)],
    cached_metadata: Option<&[Arc<ValidatedTermPruningMetadata>]>,
    snapshot: &Bm25FieldSnapshot,
    rows_by_term: &[Vec<Posting>],
    boosts: &[f32],
    segment_num_docs: u32,
    limit: usize,
    group_size: usize,
    rank_pruning: bool,
    trusted_post_move_contract: bool,
) -> Result<(u128, Vec<ScoredDoc>, UnionPruningStats), Box<dyn std::error::Error>> {
    if group_size == 0
        || encoded_terms.len() != rows_by_term.len()
        || encoded_terms.len() != boosts.len()
        || !encoded_terms.len().is_multiple_of(group_size)
        || cached_metadata.is_some_and(|metadata| metadata.len() != encoded_terms.len())
    {
        return Err(Box::new(ArgusError::CursorInvariant(
            "encoded grouped-union fixture cardinalities disagree",
        )));
    }
    let started = std::time::Instant::now();
    let doclens = encoded_doclens.section(&[1])?;
    let fieldnorms = doclens.field(1).ok_or(ArgusError::CursorInvariant(
        "encoded grouped-union fixture has no field 1",
    ))?;
    let mut outer = Vec::new();
    outer
        .try_reserve_exact(encoded_terms.len() / group_size)
        .map_err(|_| ArgusError::Allocation {
            resource: "encoded grouped-union outer clauses",
            count: encoded_terms.len() / group_size,
        })?;
    let mut group = Vec::new();
    group
        .try_reserve_exact(group_size)
        .map_err(|_| ArgusError::Allocation {
            resource: "encoded grouped-union inner clauses",
            count: group_size,
        })?;
    for (index, (encoded_postings, encoded_block_max)) in encoded_terms.iter().enumerate() {
        let cached = cached_metadata.map(|metadata| Arc::clone(&metadata[index]));
        let cursor = if let Some(metadata) = cached {
            if rank_pruning {
                SealedPostingCursor::from_validated_pruning(
                    encoded_postings.as_bytes(),
                    metadata,
                    segment_num_docs,
                )?
            } else {
                let size_hint = metadata.doc_freq();
                SealedPostingCursor::from_owned(
                    metadata.cursor(encoded_postings.as_bytes())?,
                    size_hint,
                    segment_num_docs,
                )
            }
        } else if rank_pruning {
            let postings = encoded_postings.posting_list()?;
            let metadata =
                Arc::new(postings.into_pruning_metadata(encoded_block_max.as_bytes(), fieldnorms)?);
            SealedPostingCursor::from_validated_pruning(
                encoded_postings.as_bytes(),
                metadata,
                segment_num_docs,
            )?
        } else {
            let postings = encoded_postings.posting_list()?;
            let size_hint = postings.doc_freq();
            SealedPostingCursor::from_owned(postings.into_cursor()?, size_hint, segment_num_docs)
        };
        let cursor: Box<dyn PostingCursor> = if trusted_post_move_contract {
            Box::new(cursor)
        } else {
            Box::new(ForcedPostMoveValidationCursor { inner: cursor })
        };
        let term = ReferenceScorer::term(TermScorer::new(
            cursor,
            fieldnorms,
            snapshot.clone(),
            u64::try_from(rows_by_term[index].len()).map_err(|_| {
                ArgusError::CursorInvariant("fixture document frequency exceeds u64")
            })?,
            TermRecordOption::WithFreqs,
            boosts[index],
        )?);
        group.push(ScorerClause::should(term));
        if group.len() == group_size {
            let grouped = ReferenceScorer::boolean(std::mem::take(&mut group))?;
            outer.push(ScorerClause::should(grouped));
            group
                .try_reserve_exact(group_size)
                .map_err(|_| ArgusError::Allocation {
                    resource: "encoded grouped-union inner clauses",
                    count: group_size,
                })?;
        }
    }
    let mut scorer = ReferenceScorer::boolean(outer)?;
    let mut collector = TopDocsCollector::new(limit, 0)?;
    collector.collect(&mut scorer, &AllLiveDocs)?;
    let stats = scorer
        .union_pruning_stats()
        .ok_or(ArgusError::CursorInvariant(
            "profile scorer is not a top-level union",
        ))?;
    let hits = collector.finish()?.hits;
    let elapsed_us = started.elapsed().as_micros();
    Ok((elapsed_us, hits, stats))
}

#[cfg(feature = "bench-internals")]
fn validate_encoded_pruning_metadata(
    encoded_doclens: &EncodedDocLenSection,
    encoded_terms: &[(EncodedPostingList, EncodedBlockMax)],
) -> Result<Vec<Arc<ValidatedTermPruningMetadata>>, Box<dyn std::error::Error>> {
    let doclens = encoded_doclens.section(&[1])?;
    let fieldnorms = doclens.field(1).ok_or(ArgusError::CursorInvariant(
        "encoded grouped-union fixture has no field 1",
    ))?;
    let mut metadata = Vec::new();
    metadata
        .try_reserve_exact(encoded_terms.len())
        .map_err(|_| ArgusError::Allocation {
            resource: "encoded grouped-union cached metadata",
            count: encoded_terms.len(),
        })?;
    for (encoded_postings, encoded_block_max) in encoded_terms {
        let postings = encoded_postings.posting_list()?;
        metadata.push(Arc::new(
            postings.into_pruning_metadata(encoded_block_max.as_bytes(), fieldnorms)?,
        ));
    }
    Ok(metadata)
}

#[test]
fn block_max_wand_skips_low_blocks_and_exact_count_disables_it()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 8_192;
    let lengths = vec![Some(8); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(NUM_DOCS) * 8, u64::from(NUM_DOCS))?;
    let rows_by_term = (0..9)
        .map(|term_index| {
            (0..NUM_DOCS)
                .map(|doc| {
                    let frequency = if term_index == 0 && doc == 1 {
                        32
                    } else if term_index == 0 && doc == 4_352 {
                        64
                    } else {
                        1
                    };
                    Posting::new(doc, frequency)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let boosts = [4.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
    let fieldnorm_id = fieldnorm_to_id(8);
    let encoded_terms = rows_by_term
        .iter()
        .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
        .collect::<Result<Vec<_>, _>>()?;
    let posting_lists = encoded_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

    for limit in [1, 10, 100, 1_000] {
        let mut oracle = sealed_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::new(limit, 0)?;
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
        let mut candidate_collector = TopDocsCollector::new(limit, 0)?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let candidate_hits = candidate_collector.finish()?.hits;
        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert_eq!(candidate_hits[0].global_docid, 4_352);
        assert_eq!(stats.max_score_windows, 0);
        assert_eq!(stats.block_max_wand_windows, 1);
        if limit == 1 {
            assert!(stats.blocks_skipped > 0);
            assert!((1..UNION_HORIZON_U64).contains(&stats.candidate_docs));
        }
    }

    let live = |doc: u32| !doc.is_multiple_of(7);
    let mut live_oracle = sealed_union(
        &posting_lists,
        None,
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut live_oracle_collector = TopDocsCollector::new(3, 2)?;
    live_oracle_collector.collect(&mut live_oracle, &live)?;
    let live_oracle_hits = live_oracle_collector.finish()?.hits;
    let mut live_candidate = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut live_candidate_collector = TopDocsCollector::new(3, 2)?;
    live_candidate_collector.collect(&mut live_candidate, &live)?;
    let live_stats = live_candidate
        .union_pruning_stats()
        .expect("top-level union retains pruning stats");
    let live_candidate_hits = live_candidate_collector.finish()?.hits;
    assert_hits_bit_exact(&live_candidate_hits, &live_oracle_hits);
    assert!(live_stats.block_max_wand_windows > 0);

    let mut counted_oracle = sealed_union(
        &posting_lists,
        None,
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut oracle_collector = TopDocsCollector::with_exact_count(3, 2)?;
    oracle_collector.collect(&mut counted_oracle, &live)?;
    let oracle_counted = oracle_collector.finish()?;
    let mut counted_candidate = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &boosts,
        NUM_DOCS,
    )?;
    let mut candidate_collector = TopDocsCollector::with_exact_count(3, 2)?;
    candidate_collector.collect(&mut counted_candidate, &live)?;
    let counted_stats = counted_candidate
        .union_pruning_stats()
        .expect("top-level union retains pruning stats");
    let candidate_counted = candidate_collector.finish()?;
    assert_hits_bit_exact(&candidate_counted.hits, &oracle_counted.hits);
    assert_eq!(candidate_counted.total_count, Some(7_021));
    assert_eq!(candidate_counted.total_count, oracle_counted.total_count);
    assert_eq!(counted_stats, UnionPruningStats::default());
    Ok(())
}

#[test]
fn bmw_term_bound_only_cursor_falls_back_without_error() -> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 8_192;
    let lengths = vec![Some(8); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(NUM_DOCS) * 8, u64::from(NUM_DOCS))?;
    let rows = (0..NUM_DOCS)
        .map(|doc| Posting::new(doc, 1))
        .collect::<Vec<_>>();

    let mut oracle_clauses = Vec::new();
    let mut bounded_clauses = Vec::new();
    for _ in 0..BMW_MIN_CLAUSES {
        oracle_clauses.push(ScorerClause::should(ReferenceScorer::term(
            TermScorer::new(
                VecCursor::new(rows.clone(), u64::from(NUM_DOCS), NUM_DOCS),
                field,
                snapshot.clone(),
                u64::from(NUM_DOCS),
                TermRecordOption::WithFreqs,
                1.0,
            )?,
        )));
        bounded_clauses.push(ScorerClause::should(ReferenceScorer::term(
            TermScorer::new(
                TermBoundOnlyCursor(VecCursor::new(rows.clone(), u64::from(NUM_DOCS), NUM_DOCS)),
                field,
                snapshot.clone(),
                u64::from(NUM_DOCS),
                TermRecordOption::WithFreqs,
                1.0,
            )?,
        )));
    }
    let mut oracle = ReferenceScorer::boolean(oracle_clauses)?;
    let mut oracle_collector = TopDocsCollector::new(10, 0)?;
    oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
    let oracle_hits = oracle_collector.finish()?.hits;
    let mut bounded = ReferenceScorer::boolean(bounded_clauses)?;
    let mut bounded_collector = TopDocsCollector::new(10, 0)?;
    bounded_collector.collect(&mut bounded, &AllLiveDocs)?;
    let stats = bounded
        .union_pruning_stats()
        .expect("top-level union retains pruning stats");
    let bounded_hits = bounded_collector.finish()?.hits;
    assert_hits_bit_exact(&bounded_hits, &oracle_hits);
    assert_eq!(stats, UnionPruningStats::default());
    Ok(())
}

#[test]
fn maxscore_equal_cutoff_keeps_better_docid_and_negative_weight_falls_back()
-> Result<(), Box<dyn std::error::Error>> {
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
    let rows_by_term = vec![
        vec![Posting::new(0, 1), Posting::new(5_000, 16)],
        vec![Posting::new(0, 1), Posting::new(5_000, 16)],
    ];
    let boosts = [1.0, 1.0];
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
    let term_score = expected_term_score(&snapshot, 2, fieldnorm_id, 16, 1.0);
    let cutoff = term_score + term_score;

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
    oracle_collector.record_live(10_000, Some(cutoff))?;
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
    candidate_collector.record_live(10_000, Some(cutoff))?;
    candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
    let stats = candidate
        .union_pruning_stats()
        .expect("top-level union retains pruning stats");
    let candidate_hits = candidate_collector.finish()?.hits;
    assert_hits_bit_exact(&candidate_hits, &oracle_hits);
    assert_eq!(candidate_hits[0].global_docid, 5_000);
    assert_eq!(candidate_hits[0].score.to_bits(), cutoff.to_bits());
    assert_eq!(stats.max_score_windows, 1);
    assert!(stats.candidate_docs >= 1);

    let negative_boosts = [1.0, -0.25];
    let mut fallback_oracle = sealed_union(
        &posting_lists,
        None,
        field,
        &snapshot,
        &rows_by_term,
        &negative_boosts,
        NUM_DOCS,
    )?;
    let mut oracle_collector = TopDocsCollector::new(1, 0)?;
    oracle_collector.collect(&mut fallback_oracle, &AllLiveDocs)?;
    let oracle_hits = oracle_collector.finish()?.hits;
    let mut fallback = sealed_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        &rows_by_term,
        &negative_boosts,
        NUM_DOCS,
    )?;
    let mut fallback_collector = TopDocsCollector::new(1, 0)?;
    fallback_collector.collect(&mut fallback, &AllLiveDocs)?;
    let fallback_stats = fallback
        .union_pruning_stats()
        .expect("top-level union retains pruning stats");
    let fallback_hits = fallback_collector.finish()?.hits;
    assert_hits_bit_exact(&fallback_hits, &oracle_hits);
    assert_eq!(fallback_stats, UnionPruningStats::default());
    Ok(())
}

#[test]
fn randomized_maxscore_matches_exhaustive_for_pinned_k_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 12_288;
    let lengths = vec![Some(8); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, u64::from(NUM_DOCS) * 8, u64::from(NUM_DOCS))?;
    let fieldnorm_id = fieldnorm_to_id(8);
    let boosts = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

    for seed in 1_u64..=3 {
        let mut state = seed;
        let mut rows_by_term = (0..8).map(|_| Vec::new()).collect::<Vec<_>>();
        for doc in 0..NUM_DOCS {
            for rows in &mut rows_by_term {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if state >> 62 != 0 {
                    let frequency = 1 + u32::try_from((state >> 54) & 15)
                        .expect("randomized frequency nibble fits u32");
                    rows.push(Posting::new(doc, frequency));
                }
            }
        }
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
            for k in [1, 10, 100, 1_000] {
                let mut oracle = sealed_union_with_record_option(
                    &posting_lists,
                    None,
                    field,
                    &snapshot,
                    &rows_by_term,
                    &boosts,
                    NUM_DOCS,
                    record_option,
                )?;
                let mut oracle_collector = TopDocsCollector::new(k, 0)?;
                oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
                let oracle_hits = oracle_collector.finish()?.hits;
                let mut candidate = sealed_union_with_record_option(
                    &posting_lists,
                    Some(&block_max),
                    field,
                    &snapshot,
                    &rows_by_term,
                    &boosts,
                    NUM_DOCS,
                    record_option,
                )?;
                let mut candidate_collector = TopDocsCollector::new(k, 0)?;
                candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
                let pruning_stats = candidate
                    .union_pruning_stats()
                    .expect("top-level union retains pruning stats");
                let candidate_hits = candidate_collector.finish()?.hits;
                assert_hits_bit_exact(&candidate_hits, &oracle_hits);
                assert!(
                    pruning_stats.max_score_windows > 0,
                    "{record_option:?} seed {seed}, k {k} silently fell back"
                );
                assert_eq!(pruning_stats.block_max_wand_windows, 0);
            }
        }
    }
    Ok(())
}

#[test]
fn randomized_block_max_wand_matches_exhaustive_for_pinned_k_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 8_192;
    let lengths = (0..NUM_DOCS)
        .map(|doc| Some(if doc.is_multiple_of(2) { 4 } else { 16 }))
        .collect::<Vec<_>>();
    let total_tokens = lengths
        .iter()
        .map(|length| u64::from(length.expect("fixture length is present")))
        .sum();
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(1, total_tokens, u64::from(NUM_DOCS))?;
    let boosts = [4.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02];

    for seed in 11_u64..=12 {
        let mut state = seed;
        let mut rows_by_term = (0..9).map(|_| Vec::new()).collect::<Vec<_>>();
        for doc in 0..NUM_DOCS {
            for rows in &mut rows_by_term {
                state = state
                    .wrapping_mul(2_862_933_555_777_941_757)
                    .wrapping_add(3_037_000_493);
                if state >> 62 != 0 {
                    let frequency = 1 + u32::try_from((state >> 52) & 31)
                        .expect("randomized frequency bits fit u32");
                    rows.push(Posting::new(doc, frequency));
                }
            }
        }
        assert!(
            rows_by_term.iter().map(Vec::len).sum::<usize>()
                >= usize::try_from(BMW_MIN_TOTAL_COST).expect("BMW threshold fits usize")
        );
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

        for record_option in [TermRecordOption::WithFreqs, TermRecordOption::Basic] {
            for k in [1, 10, 100, 1_000] {
                let mut oracle = sealed_union_with_record_option(
                    &posting_lists,
                    None,
                    field,
                    &snapshot,
                    &rows_by_term,
                    &boosts,
                    NUM_DOCS,
                    record_option,
                )?;
                let mut oracle_collector = TopDocsCollector::new(k, 0)?;
                oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
                let oracle_hits = oracle_collector.finish()?.hits;
                let mut candidate = sealed_union_with_record_option(
                    &posting_lists,
                    Some(&block_max),
                    field,
                    &snapshot,
                    &rows_by_term,
                    &boosts,
                    NUM_DOCS,
                    record_option,
                )?;
                let mut candidate_collector = TopDocsCollector::new(k, 0)?;
                candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
                let pruning_stats = candidate
                    .union_pruning_stats()
                    .expect("top-level union retains pruning stats");
                let candidate_hits = candidate_collector.finish()?.hits;
                assert_hits_bit_exact(&candidate_hits, &oracle_hits);
                assert_eq!(pruning_stats.max_score_windows, 0);
                assert!(
                    pruning_stats.block_max_wand_windows > 0,
                    "{record_option:?} seed {seed}, k {k} silently fell back"
                );
            }
        }
    }
    Ok(())
}

// Uses `frankensearch_core::bench_support`, which is feature-gated. This
// compiled on default features only by accident: an unconditional
// `frankensearch-lexical` dev-dependency carrying `bench-internals`
// unified `frankensearch-core/bench-internals` on for every Quill test
// build. Once that dependency became optional the free ride ended, so the
// gate is now declared where the dependency actually is (bd-552b).
#[cfg(feature = "bench-internals")]
#[test]
#[ignore = "remote-only E4.4 disjunction performance profile"]
fn e44_disjunction_profile_100k_and_1m() -> Result<(), Box<dyn std::error::Error>> {
    const PAIRED_ROUNDS: usize = 21;
    const TRIALS: usize = 7;
    for num_docs in [100_000_u32, 1_000_000] {
        let fieldnorm_id = fieldnorm_to_id(8);
        let lengths = vec![Some(8); usize::try_from(num_docs).expect("profile count fits usize")];
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(num_docs),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let snapshot = snapshot(1, u64::from(num_docs) * 8, u64::from(num_docs))?;

        let mut rows_by_term = Vec::with_capacity(8);
        let mut rare = Vec::new();
        rare.extend((0..16).map(|doc| Posting::new(doc, 64)));
        rare.extend(
            (8_192..num_docs)
                .step_by(8_192)
                .map(|doc| Posting::new(doc, 32)),
        );
        rows_by_term.push(rare);
        for divisor in 2_u32..=9 {
            rows_by_term.push(
                (0..num_docs)
                    .filter(|doc| doc % divisor == divisor - 1)
                    .map(|doc| Posting::new(doc, 1))
                    .collect(),
            );
        }

        let encoded_terms = rows_by_term
            .iter()
            .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
            .collect::<Result<Vec<_>, _>>()?;
        let boosts = [4.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        for (strategy, clause_count, group_size) in [
            (UnionPruningStrategy::MaxScore, 8, 1),
            (UnionPruningStrategy::BlockMaxWand, 9, 1),
        ] {
            let encoded_terms = &encoded_terms[..clause_count];
            let rows_by_term = &rows_by_term[..clause_count];
            let boosts = &boosts[..clause_count];
            let cold_exhaustive = timed_encoded_grouped_union(
                &encoded_doclens,
                encoded_terms,
                None,
                &snapshot,
                rows_by_term,
                boosts,
                num_docs,
                10,
                group_size,
                false,
            )?;
            let cold_pruned = timed_encoded_grouped_union(
                &encoded_doclens,
                encoded_terms,
                None,
                &snapshot,
                rows_by_term,
                boosts,
                num_docs,
                10,
                group_size,
                true,
            )?;
            assert_hits_bit_exact(&cold_pruned.1, &cold_exhaustive.1);
            let forced_validation_exhaustive = timed_encoded_grouped_union_with_contract(
                &encoded_doclens,
                encoded_terms,
                None,
                &snapshot,
                rows_by_term,
                boosts,
                num_docs,
                10,
                group_size,
                false,
                false,
            )?;
            assert_hits_bit_exact(&forced_validation_exhaustive.1, &cold_exhaustive.1);
            let cached_metadata =
                validate_encoded_pruning_metadata(&encoded_doclens, encoded_terms)?;
            let cache_payload_bytes = cached_metadata.iter().fold(0_usize, |bytes, term| {
                bytes.saturating_add(term.heap_bytes())
            });
            let cached_trusted_validation = timed_encoded_grouped_union_with_contract(
                &encoded_doclens,
                encoded_terms,
                Some(&cached_metadata),
                &snapshot,
                rows_by_term,
                boosts,
                num_docs,
                10,
                group_size,
                false,
                true,
            )?;
            let cached_forced_validation = timed_encoded_grouped_union_with_contract(
                &encoded_doclens,
                encoded_terms,
                Some(&cached_metadata),
                &snapshot,
                rows_by_term,
                boosts,
                num_docs,
                10,
                group_size,
                false,
                false,
            )?;
            assert_hits_bit_exact(&cached_trusted_validation.1, &cached_forced_validation.1);

            // This is a maintenance diagnostic: both arms run the same
            // sealed cursor and cache payload in one invocation. It
            // controls only the redundant post-move validation branch;
            // it does not put an incumbent in the comparison.
            let h3_null = frankensearch_core::bench_support::paired_median_ratio(
                PAIRED_ROUNDS,
                1,
                || {
                    let _ = std::hint::black_box(
                        timed_encoded_grouped_union_with_contract(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            false,
                            true,
                        )
                        .expect("run first trusted H3 null arm"),
                    );
                },
                || {
                    let _ = std::hint::black_box(
                        timed_encoded_grouped_union_with_contract(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            false,
                            true,
                        )
                        .expect("run second trusted H3 null arm"),
                    );
                },
            );
            let h3_trusted_over_forced = frankensearch_core::bench_support::paired_median_ratio(
                PAIRED_ROUNDS,
                1,
                || {
                    let _ = std::hint::black_box(
                        timed_encoded_grouped_union_with_contract(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            false,
                            false,
                        )
                        .expect("run forced-validation H3 control arm"),
                    );
                },
                || {
                    let _ = std::hint::black_box(
                        timed_encoded_grouped_union_with_contract(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            false,
                            true,
                        )
                        .expect("run trusted-validation H3 control arm"),
                    );
                },
            );

            let null = frankensearch_core::bench_support::paired_median_ratio(
                PAIRED_ROUNDS,
                1,
                || {
                    let _ = std::hint::black_box(
                        timed_encoded_grouped_union(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            false,
                        )
                        .expect("run first exhaustive null arm"),
                    );
                },
                || {
                    let _ = std::hint::black_box(
                        timed_encoded_grouped_union(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            false,
                        )
                        .expect("run second exhaustive null arm"),
                    );
                },
            );
            let lever = frankensearch_core::bench_support::paired_median_ratio(
                PAIRED_ROUNDS,
                1,
                || {
                    let _ = std::hint::black_box(
                        timed_encoded_grouped_union(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            false,
                        )
                        .expect("run paired exhaustive arm"),
                    );
                },
                || {
                    let _ = std::hint::black_box(
                        timed_encoded_grouped_union(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            true,
                        )
                        .expect("run paired pruned arm"),
                    );
                },
            );

            // Warm both paths before collecting alternating-order medians.
            let _ = timed_encoded_grouped_union(
                &encoded_doclens,
                encoded_terms,
                Some(&cached_metadata),
                &snapshot,
                rows_by_term,
                boosts,
                num_docs,
                10,
                group_size,
                false,
            )?;
            let _ = timed_encoded_grouped_union(
                &encoded_doclens,
                encoded_terms,
                Some(&cached_metadata),
                &snapshot,
                rows_by_term,
                boosts,
                num_docs,
                10,
                group_size,
                true,
            )?;

            let mut exhaustive_us = Vec::with_capacity(TRIALS);
            let mut pruned_us = Vec::with_capacity(TRIALS);
            let mut evidence = None;
            for trial in 0..TRIALS {
                let run_exhaustive = || {
                    timed_encoded_grouped_union(
                        &encoded_doclens,
                        encoded_terms,
                        Some(&cached_metadata),
                        &snapshot,
                        rows_by_term,
                        boosts,
                        num_docs,
                        10,
                        group_size,
                        false,
                    )
                };
                let run_pruned = || {
                    timed_encoded_grouped_union(
                        &encoded_doclens,
                        encoded_terms,
                        Some(&cached_metadata),
                        &snapshot,
                        rows_by_term,
                        boosts,
                        num_docs,
                        10,
                        group_size,
                        true,
                    )
                };
                let (exhaustive, pruned) = if trial % 2 == 0 {
                    (run_exhaustive()?, run_pruned()?)
                } else {
                    let pruned = run_pruned()?;
                    let exhaustive = run_exhaustive()?;
                    (exhaustive, pruned)
                };
                assert_hits_bit_exact(&pruned.1, &exhaustive.1);
                match strategy {
                    UnionPruningStrategy::MaxScore => {
                        assert!(pruned.2.max_score_windows > 0);
                        assert_eq!(pruned.2.block_max_wand_windows, 0);
                    }
                    UnionPruningStrategy::BlockMaxWand => {
                        assert!(pruned.2.block_max_wand_windows > 0);
                    }
                }
                exhaustive_us.push(exhaustive.0);
                pruned_us.push(pruned.0);
                let checksum = pruned.1.iter().fold(0_u64, |state, hit| {
                    state
                        .wrapping_mul(16_777_619)
                        .wrapping_add(u64::from(hit.global_docid))
                        .wrapping_add(u64::from(hit.score.to_bits()))
                });
                let current_evidence = (pruned.1.len(), checksum);
                if let Some(expected) = evidence {
                    assert_eq!(current_evidence, expected);
                } else {
                    evidence = Some(current_evidence);
                }
            }
            exhaustive_us.sort_unstable();
            pruned_us.sort_unstable();
            let exhaustive_median_us = exhaustive_us[TRIALS / 2];
            let pruned_median_us = pruned_us[TRIALS / 2];
            let (hit_count, checksum) = evidence.expect("profile executes at least one trial");
            let strategy_label = match strategy {
                UnionPruningStrategy::MaxScore => "maxscore",
                UnionPruningStrategy::BlockMaxWand => "block_max_wand",
            };
            eprintln!(
                "e44_profile strategy={strategy_label} docs={num_docs} \
                 cold_exhaustive_us={} cold_pruned_us={} cache_payload_bytes={} \
                 exhaustive_median_us={exhaustive_median_us} \
                 pruned_median_us={pruned_median_us} trials={TRIALS} hits={} \
                 checksum={checksum} null_median={:.6} null_p5={:.6} null_p95={:.6} \
                 lever_median={:.6} lever_p5={:.6} lever_p95={:.6} \
                 h3_null_median={:.6} h3_null_p5={:.6} h3_null_p95={:.6} \
                 h3_trusted_over_forced_median={:.6} \
                 h3_trusted_over_forced_p5={:.6} h3_trusted_over_forced_p95={:.6} \
                 paired_rounds={PAIRED_ROUNDS}",
                cold_exhaustive.0,
                cold_pruned.0,
                cache_payload_bytes,
                hit_count,
                null.median,
                null.p5,
                null.p95,
                lever.median,
                lever.p5,
                lever.p95,
                h3_null.median,
                h3_null.p5,
                h3_null.p95,
                h3_trusted_over_forced.median,
                h3_trusted_over_forced.p5,
                h3_trusted_over_forced.p95,
            );
        }
    }
    Ok(())
}
