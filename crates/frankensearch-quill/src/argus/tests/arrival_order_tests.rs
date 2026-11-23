use super::*;

/// A pruned ranked-root window must accumulate in Tantivy's doc-major term
/// order rather than in scorer-vector order
/// (`bd-quill-pruned-topdocs-term-order-9wu3p`).
///
/// The fixture is the shipping shape: `boolean_topdocs` over seven direct
/// frequency terms carrying real validated BLOCKMAX, a `TopDocs` cutoff
/// filled by an entirely exhaustive first window, and survivors in later,
/// pruned windows. The BLOCKMAX-free arm is the oracle — without ceilings
/// every one of its windows falls back to the exhaustive fill — so the two
/// arms differ only in whether pruning ran.
///
/// Every clause shares one snapshot document frequency, so `idf` cancels
/// and each small clause contributes exactly `2.0 / 1.0e8` of the huge one.
/// That ratio sits below half an ulp of the huge contribution while six of
/// them together sit above it, so the huge term absorbs all six smalls when
/// it is summed first and cannot absorb them when it is summed last. The
/// postings are laid out so the pinned stable sort puts the huge clause
/// last at both survivors while the scorer vector holds it first.
///
/// Doc `10_000` is the drift probe: it is reached in the refill *after* the
/// first pruned one, with every clause arriving simultaneously, so its
/// equal-document order is exactly the permutation the pruned window left
/// behind. Doc `15_000` is reached with one active clause, which routes its
/// refill back through the exhaustive fill.
#[test]
fn pruned_topdocs_windows_keep_tantivy_doc_major_term_order()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 16_384;
    const DENSE_END: u32 = 4_096;
    const HUGE_ENTRY: u32 = 4_096;
    const PRUNED_SURVIVOR: u32 = 6_000;
    const INHERITED_SURVIVOR: u32 = 10_000;
    const EXHAUSTIVE_TAIL: u32 = 15_000;
    const SMALL_TERMS: u32 = 6;
    const SNAPSHOT_DOC_FREQ: u64 = 4_099;
    const MATCHED_DOCS: u64 = 4_106;

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

    let mut rows_by_term = vec![vec![
        Posting::new(HUGE_ENTRY, 1),
        Posting::new(PRUNED_SURVIVOR, 1),
        Posting::new(INHERITED_SURVIVOR, 1),
        Posting::new(EXHAUSTIVE_TAIL, 1),
    ]];
    for index in 0..SMALL_TERMS {
        let mut rows = (0..DENSE_END)
            .map(|doc| Posting::new(doc, 1))
            .collect::<Vec<_>>();
        // One private posting per small clause, all of them above the huge
        // clause's window entry. That makes the huge clause hold the lowest
        // previous posting beneath each survivor, so the stable sort parks
        // it last there while the scorer vector still holds it first.
        rows.push(Posting::new(DENSE_END + 1 + index, 1));
        rows.push(Posting::new(PRUNED_SURVIVOR, 1));
        rows.push(Posting::new(INHERITED_SURVIVOR, 1));
        rows_by_term.push(rows);
    }
    for rows in &rows_by_term {
        assert!(
            rows.windows(2).all(|pair| pair[0].doc_id < pair[1].doc_id),
            "fixture postings must be strictly ascending"
        );
    }
    assert_eq!(
        u64::try_from(rows_by_term[1].len())?,
        SNAPSHOT_DOC_FREQ,
        "the shared snapshot document frequency mirrors one small clause"
    );
    let boosts = [1.0e8_f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    assert_eq!(rows_by_term.len(), boosts.len());
    assert!(
        (2..=MAX_SCORE_MAX_CLAUSES).contains(&boosts.len()),
        "the fixture must land inside the MaxScore clause range"
    );

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

    let small = expected_term_score(&snapshot, SNAPSHOT_DOC_FREQ, fieldnorm_id, 1, 2.0);
    let huge = expected_term_score(&snapshot, SNAPSHOT_DOC_FREQ, fieldnorm_id, 1, 1.0e8);
    // The union accumulator starts at 0.0, and 0.0 + x is exactly x.
    let mut dense = 0.0_f32;
    for _ in 0..SMALL_TERMS {
        dense += small;
    }
    let doc_major = dense + huge;
    let mut scorer_major = huge;
    for _ in 0..SMALL_TERMS {
        scorer_major += small;
    }
    assert_ne!(
        doc_major.to_bits(),
        scorer_major.to_bits(),
        "the fixture must plant a non-associative f32 sum or it proves nothing"
    );
    assert_eq!(
        scorer_major.to_bits(),
        huge.to_bits(),
        "each small contribution alone must vanish into the huge one"
    );

    for limit in [1_usize, 2, 4, 8, 64] {
        let mut oracle = sealed_topdocs_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            SNAPSHOT_DOC_FREQ,
            &boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::new(limit, 0)?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_stats = oracle
            .union_pruning_stats()
            .expect("the ranked root is a buffered union");
        let oracle_hits = oracle_collector.finish()?.hits;
        assert_eq!(
            oracle_stats,
            UnionPruningStats::default(),
            "the BLOCKMAX-free oracle must stay exhaustive at limit {limit}"
        );

        let (mut candidate, _work) = validated_topdocs_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            SNAPSHOT_DOC_FREQ,
            &boosts,
            NUM_DOCS,
        )?;
        let mut candidate_collector = TopDocsCollector::new(limit, 0)?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("the ranked root is a buffered union");
        let candidate_hits = candidate_collector.finish()?.hits;

        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert_eq!(candidate_hits.len(), limit);
        assert!(
            stats.max_score_windows >= 1,
            "rank pruning must actually engage at limit {limit}"
        );
        assert_eq!(
            stats.block_max_wand_windows, 0,
            "seven direct-term clauses take MaxScore, not block-max WAND"
        );
        assert_eq!(stats.blocks_skipped, 0, "MaxScore never skips blocks");
        assert!(stats.candidate_docs >= 1);

        let score_of = |docid: u32| -> u32 {
            candidate_hits
                .iter()
                .find(|hit| hit.global_docid == docid)
                .unwrap_or_else(|| panic!("doc {docid} is missing from the limit-{limit} page"))
                .score
                .to_bits()
        };
        assert_eq!(candidate_hits[0].global_docid, PRUNED_SURVIVOR);
        assert_eq!(
            score_of(PRUNED_SURVIVOR),
            doc_major.to_bits(),
            "the pruned window's survivor must keep Tantivy's doc-major order at limit {limit}"
        );
        if limit >= 2 {
            assert_eq!(
                score_of(INHERITED_SURVIVOR),
                doc_major.to_bits(),
                "a refill after a pruned one must not inherit scorer-major drift"
            );
        }
        if limit >= 4 {
            assert_eq!(score_of(HUGE_ENTRY), huge.to_bits());
            assert_eq!(
                score_of(EXHAUSTIVE_TAIL),
                huge.to_bits(),
                "the single-clause tail refill routes back through the exhaustive fill"
            );
        }
        if limit >= 8 {
            assert_eq!(
                score_of(0),
                dense.to_bits(),
                "the exhaustive first window is unchanged"
            );
            assert_eq!(
                candidate_hits
                    .iter()
                    .map(|hit| hit.global_docid)
                    .take(8)
                    .collect::<Vec<_>>(),
                vec![
                    PRUNED_SURVIVOR,
                    INHERITED_SURVIVOR,
                    HUGE_ENTRY,
                    EXHAUSTIVE_TAIL,
                    0,
                    1,
                    2,
                    3
                ],
                "ranked order must stay score-descending then docid-ascending"
            );
        }
    }

    let mut counted_scorer = sealed_topdocs_union(
        &posting_lists,
        Some(&block_max),
        field,
        &snapshot,
        SNAPSHOT_DOC_FREQ,
        &boosts,
        NUM_DOCS,
    )?;
    let mut counted_collector = TopDocsCollector::with_exact_count(4, 0)?;
    counted_collector.collect(&mut counted_scorer, &AllLiveDocs)?;
    let counted_stats = counted_scorer
        .union_pruning_stats()
        .expect("the ranked root is a buffered union");
    let counted_page = counted_collector.finish()?;
    assert_eq!(
        counted_stats,
        UnionPruningStats::default(),
        "exact-count collection must keep pruning disabled"
    );
    assert_eq!(counted_page.total_count, Some(MATCHED_DOCS));
    assert_eq!(
        counted_page.hits[0].score.to_bits(),
        doc_major.to_bits(),
        "the counted exhaustive path already used the doc-major order"
    );
    Ok(())
}

/// A pruned doc-major window must rebuild Tantivy's equal-document order
/// from arrival predecessors instead of walking to it
/// (`bd-quill-pruned-topdocs-term-order-9wu3p`, forward correction of the
/// rejected `20b0a7b5`).
///
/// This is the reviewer's hostile shape generalized. With
/// `X = {5, 8, 10}`, `Y = {7, 10}`, `Z = {10}`, stepping reaches 10 as
/// `X + Y + Z`: `X`'s stop at 8 and `Y`'s at 7 order them by decreasing
/// previous posting, and `Z`, which skipped nothing, parks last. Here each
/// small clause owns one private posting below the meeting document — the
/// role of 8 and 7 — while the huge clause owns none, so the pinned order
/// puts the huge clause last and its `1.0e8` boost cannot absorb the six
/// smalls that precede it. Summed in scorer-vector order, where the huge
/// clause is first, every small vanishes; the two orders differ in raw f32
/// bits, and the fixture asserts that before relying on it.
///
/// The candidate arm and the BLOCKMAX-free oracle differ only in whether
/// pruning ran, so bit-exact agreement is the parity proof, and
/// `arrival_walk_windows` proves the pruned window reached it by seeking
/// rather than by falling back to the exact walk.
#[test]
fn pruned_topdocs_window_rebuilds_arrival_order_without_walking()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 24_097;
    const CUTOFF_DOC: u32 = 0;
    const MEETING_DOC: u32 = 5_400;
    const TAIL_DOC: u32 = 20_000;
    const PRIVATE_BASE: u32 = 5_001;
    const PRIVATE_STRIDE: u32 = 32;
    const PRIVATE_RUN: u32 = 20;
    const SMALL_TERMS: u32 = 6;
    const CUTOFF_DOCS: u32 = 4;
    const SNAPSHOT_DOC_FREQ: u64 = 8;
    /// One step per clause at each document of the two exhaustive windows,
    /// plus slack. Walking the pruned window instead would add all
    /// `SMALL_TERMS * PRIVATE_RUN` private postings.
    const STEP_BUDGET: u64 = 60;

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

    // The cutoff documents fill the `TopDocs` heap inside the first,
    // always-exhaustive window, so every later window sees a finite cutoff
    // and can prune. Clause 0 skips nothing between them and the meeting
    // document, so its arrival predecessor there is the oldest of all.
    let cutoff_docs = || (CUTOFF_DOC..CUTOFF_DOC + CUTOFF_DOCS).map(|doc| Posting::new(doc, 1));
    let mut rows_by_term = vec![
        cutoff_docs()
            .chain([Posting::new(MEETING_DOC, 1), Posting::new(TAIL_DOC, 1)])
            .collect::<Vec<_>>(),
    ];
    for index in 0..SMALL_TERMS {
        // A private run per small clause, none of them competitive and none
        // of them a candidate: they exist only so each clause reaches the
        // meeting document from a different, strictly later posting.
        let privates = (0..PRIVATE_RUN)
            .map(|step| Posting::new(PRIVATE_BASE + index * PRIVATE_STRIDE + step, 1));
        rows_by_term.push(
            cutoff_docs()
                .chain(privates)
                .chain([Posting::new(MEETING_DOC, 1), Posting::new(TAIL_DOC, 1)])
                .collect::<Vec<_>>(),
        );
    }
    let boosts = [1.0e8_f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    assert_eq!(rows_by_term.len(), boosts.len());
    for rows in &rows_by_term {
        assert!(
            rows.windows(2).all(|pair| pair[0].doc_id < pair[1].doc_id),
            "fixture postings must be strictly ascending"
        );
        assert!(
            rows.iter().all(|row| row.doc_id < NUM_DOCS),
            "fixture postings must stay inside the segment"
        );
    }
    // Every operand is a `const`, so this holds at compile time or not at
    // all: as a `const` item it is checked whether or not this test is
    // selected to run, and a fixture edit that broke it would fail the
    // build rather than wait for the test that happens to exercise it.
    const _: () = assert!(
        PRIVATE_BASE + (SMALL_TERMS - 1) * PRIVATE_STRIDE + PRIVATE_RUN < MEETING_DOC,
        "every private run must close before the meeting document"
    );

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

    let small = expected_term_score(&snapshot, SNAPSHOT_DOC_FREQ, fieldnorm_id, 1, 2.0);
    let huge = expected_term_score(&snapshot, SNAPSHOT_DOC_FREQ, fieldnorm_id, 1, 1.0e8);
    let mut dense = 0.0_f32;
    for _ in 0..SMALL_TERMS {
        dense += small;
    }
    let doc_major = dense + huge;
    let mut scorer_major = huge;
    for _ in 0..SMALL_TERMS {
        scorer_major += small;
    }
    assert_ne!(
        doc_major.to_bits(),
        scorer_major.to_bits(),
        "the fixture must plant a non-associative f32 sum or it proves nothing"
    );
    assert_eq!(
        scorer_major.to_bits(),
        huge.to_bits(),
        "each small contribution alone must vanish into the huge one"
    );

    for limit in [1_usize, 2, 4] {
        let mut oracle = sealed_topdocs_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            SNAPSHOT_DOC_FREQ,
            &boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::new(limit, 0)?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_hits = oracle_collector.finish()?.hits;

        let (mut candidate, work) = validated_topdocs_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            SNAPSHOT_DOC_FREQ,
            &boosts,
            NUM_DOCS,
        )?;
        let mut candidate_collector = TopDocsCollector::new(limit, 0)?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("the ranked root is a buffered union");
        let candidate_hits = candidate_collector.finish()?.hits;

        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert_eq!(candidate_hits[0].global_docid, MEETING_DOC);
        assert_eq!(
            candidate_hits[0].score.to_bits(),
            doc_major.to_bits(),
            "the pruned window must rebuild the doc-major order at limit {limit}"
        );
        assert!(
            stats.max_score_windows >= 1,
            "rank pruning must actually engage at limit {limit}"
        );
        assert!(
            stats.arrival_walk_windows <= 1,
            "only the terminal window, where every clause exhausts, may walk"
        );
        assert!(
            total_steps(&work) <= STEP_BUDGET,
            "the pruned window's scoring cursors must move past the private \
             runs by seeking, not step through them: {} steps at limit \
             {limit} (steps only; decodes are unmeasured)",
            total_steps(&work)
        );
    }
    Ok(())
}

/// A block-max WAND window whose competitive set is empty must cost the
/// real scoring cursors one seek per clause, not one step per posting
/// (`bd-quill-pruned-topdocs-term-order-9wu3p`, forward correction of the
/// rejected `20b0a7b5`).
///
/// Nine direct frequency terms with real validated BLOCKMAX and enough
/// aggregate cost to select BMW rather than `MaxScore`. Document 0 fills
/// the `TopDocs` cutoff, the middle horizon is dense and entirely
/// skippable, and document 9,000 is the only competitive survivor. The
/// dense horizon is the measurement: discovery skips its blocks in shadow,
/// and `blocks_skipped` says so, but only the counter on the scoring
/// cursors can show whether the real scan skipped them too. Walking that
/// horizon costs thousands of steps, which is what the rejected parent did
/// after paying for discovery.
///
/// The huge clause holds nothing between the cutoff and the survivor, so
/// its arrival predecessor there is older than the window and it sorts
/// last; the eight smalls each omit a different residue class, so they
/// separate after a bounded number of refinement rounds. Their mutual order
/// cannot change the sum — they contribute equally — while the huge clause
/// landing last is exactly what stops it absorbing them.
#[test]
fn empty_block_max_wand_window_costs_seeks_not_steps() -> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 12_301;
    const FIELD_LENGTH: u32 = 64;
    const CUTOFF_DOC: u32 = 0;
    /// Enough cutoff documents that the `TopDocs` heap is full before the
    /// first window ends: with no cutoff there is nothing to prune against,
    /// and the dense horizon would be scored exhaustively by both arms,
    /// which would make the step budget below prove nothing.
    const CUTOFF_DOCS: u32 = 4;
    const DENSE_START: u32 = 4_096;
    const DENSE_END: u32 = 12_300;
    const TARGET_DOC: u32 = 9_000;
    const SMALL_TERMS: u32 = 8;
    const SNAPSHOT_DOC_FREQ: u64 = 4_096;
    /// The exhaustive first and terminal windows step a clause at a time;
    /// walking the skipped horizon instead would add tens of thousands.
    ///
    /// This bounds *steps* — `next` calls on the scoring cursors — and
    /// nothing else. It is not a cost bound: predecessor probes below a
    /// parked block re-decode that block per probe, and neither this fixture
    /// nor any other in this file counts physical decodes or fuel.
    const STEP_BUDGET: u64 = 2_000;
    assert_eq!(BMW_MIN_CLAUSES, 9, "fixture requires nine direct terms");

    let lengths = vec![Some(FIELD_LENGTH); usize::try_from(NUM_DOCS)?];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(
        1,
        u64::from(NUM_DOCS) * u64::from(FIELD_LENGTH),
        u64::from(NUM_DOCS),
    )?;

    let cutoff_docs = || (CUTOFF_DOC..CUTOFF_DOC + CUTOFF_DOCS).map(|doc| Posting::new(doc, 4));
    let mut rows_by_term = vec![
        cutoff_docs()
            .chain([Posting::new(TARGET_DOC, 8), Posting::new(DENSE_END, 1)])
            .collect::<Vec<_>>(),
    ];
    for residue in 0..SMALL_TERMS {
        let mut rows = cutoff_docs().collect::<Vec<_>>();
        rows.extend((DENSE_START..=DENSE_END).filter_map(|doc| {
            if doc == TARGET_DOC {
                return Some(Posting::new(doc, 8));
            }
            (doc % SMALL_TERMS != residue).then_some(Posting::new(doc, 1))
        }));
        rows_by_term.push(rows);
    }
    let mut boosts = vec![2.0_f32; BMW_MIN_CLAUSES];
    boosts[0] = 1.0e8;
    assert_eq!(rows_by_term.len(), boosts.len());
    assert_eq!(rows_by_term.len(), BMW_MIN_CLAUSES);
    let total_cost: u64 = rows_by_term
        .iter()
        .map(|rows| u64::try_from(rows.len()).unwrap_or(u64::MAX))
        .sum();
    assert!(
        total_cost >= 16_384,
        "aggregate cost must select block-max WAND, not MaxScore: {total_cost}"
    );

    let fieldnorm_id = fieldnorm_to_id(FIELD_LENGTH);
    let encoded_terms = rows_by_term
        .iter()
        .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
        .collect::<Result<Vec<_>, _>>()?;
    let posting_lists = encoded_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

    let small = expected_term_score(&snapshot, SNAPSHOT_DOC_FREQ, fieldnorm_id, 8, 2.0);
    let huge = expected_term_score(&snapshot, SNAPSHOT_DOC_FREQ, fieldnorm_id, 8, 1.0e8);
    let mut dense = 0.0_f32;
    for _ in 0..SMALL_TERMS {
        dense += small;
    }
    let doc_major = dense + huge;
    let mut scorer_major = huge;
    for _ in 0..SMALL_TERMS {
        scorer_major += small;
    }
    assert_ne!(
        doc_major.to_bits(),
        scorer_major.to_bits(),
        "the fixture must plant a non-associative f32 sum or it proves nothing"
    );

    for limit in [1_usize, 4] {
        let mut oracle = sealed_topdocs_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            SNAPSHOT_DOC_FREQ,
            &boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::new(limit, 0)?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_stats = oracle
            .union_pruning_stats()
            .expect("the ranked root is a buffered union");
        let oracle_hits = oracle_collector.finish()?.hits;
        assert_eq!(
            oracle_stats,
            UnionPruningStats::default(),
            "the BLOCKMAX-free oracle must stay exhaustive at limit {limit}"
        );

        let (mut candidate, work) = validated_topdocs_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            SNAPSHOT_DOC_FREQ,
            &boosts,
            NUM_DOCS,
        )?;
        let mut candidate_collector = TopDocsCollector::new(limit, 0)?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("the ranked root is a buffered union");
        let candidate_hits = candidate_collector.finish()?.hits;

        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert_eq!(candidate_hits[0].global_docid, TARGET_DOC);
        assert_eq!(
            candidate_hits[0].score.to_bits(),
            doc_major.to_bits(),
            "the survivor must keep the doc-major order at limit {limit}"
        );
        assert!(
            stats.block_max_wand_windows >= 2,
            "block-max WAND must run in the skipped and the surviving window"
        );
        assert!(
            stats.blocks_skipped > 0,
            "discovery must actually skip blocks at limit {limit}"
        );
        assert!(
            stats.arrival_walk_windows <= 1,
            "only the terminal window, where every clause exhausts, may walk: {} walked",
            stats.arrival_walk_windows
        );
        assert!(
            total_steps(&work) <= STEP_BUDGET,
            "scoring cursors must move by seeking over the skipped horizon: \
             {} steps at limit {limit} (steps only; decodes are unmeasured)",
            total_steps(&work)
        );
    }
    Ok(())
}

/// Deterministic pseudo-random source: a step-versus-seek oracle needs many
/// posting shapes, not a curated one.
struct ArrivalRng(u64);

impl ArrivalRng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(self.0 >> 33).expect("a 31-bit shift fits u32")
    }

    fn below(&mut self, bound: u32) -> u32 {
        self.next_u32() % bound
    }
}

/// Randomized step-versus-seek oracle for the doc-major arrival order.
///
/// Every seed builds one ranked direct-term union twice over identical
/// postings: the BLOCKMAX-free arm scores every window by stepping, the
/// BLOCKMAX arm prunes and, wherever no clause exhausts inside a window,
/// reaches its candidates by seeking and rebuilds the equal-document order
/// from arrival predecessors. Bit-exact agreement across both arms is the
/// property; a curated fixture cannot cover the shapes that matter here —
/// clauses sharing a predecessor, then sharing the one before that, and
/// separating only deeper in — but random dense postings produce them
/// constantly.
#[test]
fn randomized_pruned_topdocs_matches_stepped_arrival_order()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 9_000;
    const FIELD_LENGTH: u32 = 8;
    const CLAUSES: usize = 9;
    const SNAPSHOT_DOC_FREQ: u64 = 3_000;

    let lengths = vec![Some(FIELD_LENGTH); usize::try_from(NUM_DOCS)?];
    let encoded_doclens = EncodedDocLenSection::encode(
        0,
        u64::from(NUM_DOCS),
        &[1],
        &[DocLenFieldInput::new(1, &lengths)],
    )?;
    let doclens = encoded_doclens.section(&[1])?;
    let field = doclens.field(1).expect("field exists");
    let snapshot = snapshot(
        1,
        u64::from(NUM_DOCS) * u64::from(FIELD_LENGTH),
        u64::from(NUM_DOCS),
    )?;
    let fieldnorm_id = fieldnorm_to_id(FIELD_LENGTH);

    let mut pruned_windows = 0_u64;
    let mut sought_windows = 0_u64;
    for seed in 0..6_u64 {
        const SMALL_BOOSTS: [f32; 4] = [1.0, 2.0, 3.0, 5.0];
        let mut rng = ArrivalRng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut rows_by_term = Vec::new();
        let mut boosts = Vec::new();
        for clause in 0..CLAUSES {
            // Dense enough that clauses routinely share runs of consecutive
            // documents, which is what forces the arrival refinement past
            // its first round.
            let keep = 40 + rng.below(50);
            let mut rows = Vec::new();
            for doc in 0..NUM_DOCS {
                if rng.below(100) < keep {
                    rows.push(Posting::new(doc, 1 + rng.below(8)));
                }
            }
            if rows.is_empty() {
                rows.push(Posting::new(rng.below(NUM_DOCS), 1));
            }
            rows_by_term.push(rows);
            // One dominating clause per seed keeps a real cutoff in play,
            // which is what makes the other clauses prunable at all.
            boosts.push(if clause == 0 {
                1.0e8
            } else {
                SMALL_BOOSTS[usize::try_from(rng.below(4))?]
            });
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

        for limit in [1_usize, 5] {
            let mut oracle = sealed_topdocs_union(
                &posting_lists,
                None,
                field,
                &snapshot,
                SNAPSHOT_DOC_FREQ,
                &boosts,
                NUM_DOCS,
            )?;
            let mut oracle_collector = TopDocsCollector::new(limit, 0)?;
            oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
            let oracle_hits = oracle_collector.finish()?.hits;

            let (mut candidate, _work) = validated_topdocs_union(
                &posting_lists,
                Some(&block_max),
                field,
                &snapshot,
                SNAPSHOT_DOC_FREQ,
                &boosts,
                NUM_DOCS,
            )?;
            let mut candidate_collector = TopDocsCollector::new(limit, 0)?;
            candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
            let stats = candidate
                .union_pruning_stats()
                .expect("the ranked root is a buffered union");
            let candidate_hits = candidate_collector.finish()?.hits;

            assert_eq!(
                candidate_hits.len(),
                oracle_hits.len(),
                "seed {seed} limit {limit} returned a different page size"
            );
            for (rank, (actual, expected)) in candidate_hits.iter().zip(&oracle_hits).enumerate() {
                assert_eq!(
                    actual.global_docid, expected.global_docid,
                    "seed {seed} limit {limit} rank {rank} docid differs"
                );
                assert_eq!(
                    actual.score.to_bits(),
                    expected.score.to_bits(),
                    "seed {seed} limit {limit} rank {rank} score bits differ for doc {}",
                    actual.global_docid
                );
            }
            let windows = stats
                .max_score_windows
                .saturating_add(stats.block_max_wand_windows);
            pruned_windows = pruned_windows.saturating_add(windows);
            sought_windows =
                sought_windows.saturating_add(windows.saturating_sub(stats.arrival_walk_windows));
        }
    }
    assert!(
        pruned_windows > 0,
        "the randomized corpus must actually prune, or it proves nothing"
    );
    assert!(
        sought_windows > 0,
        "the randomized corpus must actually reach a pruned window by seeking"
    );
    Ok(())
}
