use super::*;

/// A wrapper that launders a capability it does not implement must fail
/// closed (`bd-quill-pruned-topdocs-term-order-9wu3p`).
///
/// Two designs have failed this fixture, in the same way. First the gate was
/// a `final_doc()`-shaped one, spread across two independently defaulted
/// trait methods. Then it was a `validated_arrival_metadata()` token — which
/// a wrapper can simply *forward* from an embedded sealed cursor while
/// inheriting the defaulted read, so the engine saw genuine evidence and
/// then probed the wrapper. Both times the defaulted read answered
/// `Ok(None)`: a legal answer meaning "no earlier posting". Every key
/// settled on the first refinement round, every equal-document group
/// collapsed to its incoming order, and the union summed in scorer-vector
/// order with nothing raised. Neither of those trait methods exists now.
///
/// Binding proof to each result was not enough either: a wrapper holding two
/// sealed cursors could probe one while the engine drove the other, and
/// every receipt it returned was genuine, correctly minted and correctly
/// targeted, about the wrong posting list. All three designs asked an
/// overridable trait for evidence about itself. There is no arrival method
/// on `PostingCursor` any more, so this wrapper — which forwards every
/// accessor it honestly can — is simply an opaque cursor, and opaque cursors
/// walk.
///
/// The fixture is the arrival shape, where scorer-vector order lets the
/// huge clause absorb the smalls, so a silent gate failure shows up as a
/// wrong f32 score rather than a slow query. Every pruned window must take
/// the exact walk and the scores must stay bit-exact against the exhaustive
/// oracle.
#[test]
fn laundered_arrival_capability_cannot_enable_seeking() -> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 24_097;
    const CUTOFF_DOCS: u32 = 4;
    const MEETING_DOC: u32 = 5_400;
    const TAIL_DOC: u32 = 20_000;
    const PRIVATE_BASE: u32 = 5_001;
    const PRIVATE_STRIDE: u32 = 32;
    const PRIVATE_RUN: u32 = 20;
    const SMALL_TERMS: u32 = 6;
    const SNAPSHOT_DOC_FREQ: u64 = 8;

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

    let cutoff_docs = || (0..CUTOFF_DOCS).map(|doc| Posting::new(doc, 1));
    let mut rows_by_term = vec![
        cutoff_docs()
            .chain([Posting::new(MEETING_DOC, 1), Posting::new(TAIL_DOC, 1)])
            .collect::<Vec<_>>(),
    ];
    for index in 0..SMALL_TERMS {
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
    assert_ne!(
        doc_major.to_bits(),
        huge.to_bits(),
        "the fixture must be able to show a collapsed order or it proves nothing"
    );

    // The wrapper embeds a genuine sealed cursor and forwards everything a
    // wrapper can honestly forward. There is nothing left for it to launder:
    // the arrival read is not on the trait, so wrapping is itself the whole
    // of what it can do, and that lands it on the opaque arm.
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
        let oracle_hits = oracle_collector.finish()?.hits;

        let mut partial = wrapped_topdocs_union(
            &posting_lists,
            &block_max,
            field,
            &snapshot,
            SNAPSHOT_DOC_FREQ,
            &boosts,
            NUM_DOCS,
            |inner| PartialArrivalCursor { inner },
        )?;
        let mut partial_collector = TopDocsCollector::new(limit, 0)?;
        partial_collector.collect(&mut partial, &AllLiveDocs)?;
        let stats = partial
            .union_pruning_stats()
            .expect("the ranked root is a buffered union");
        let partial_hits = partial_collector.finish()?.hits;

        assert_hits_bit_exact(&partial_hits, &oracle_hits);
        assert_eq!(
            partial_hits[0].score.to_bits(),
            doc_major.to_bits(),
            "a half-capable clause must not collapse the arrival order at limit {limit}"
        );
        let pruned = stats
            .max_score_windows
            .saturating_add(stats.block_max_wand_windows);
        assert!(
            pruned >= 1,
            "the fixture must still prune at limit {limit}, or the gate is untested"
        );
        assert_eq!(
            stats.arrival_walk_windows, pruned,
            "every pruned window must take the exact walk when the capability is absent"
        );
    }

    Ok(())
}

/// A wrapper holding two sealed cursors, driving one and answering history
/// from the other.
///
/// This is the substitution attack that survived receipt sealing and target
/// binding: every answer `shadow` produced was genuine, correctly minted and
/// correctly targeted, and described a posting list the engine never moved.
/// Both lists carry the meeting document, so `doc()` agrees wherever the
/// union looks; only the history beneath it differs, which is exactly what
/// decides an equal-document order.
struct SubstitutingArrivalCursor<'a> {
    driven: SealedPostingCursor<'a>,
    shadow: SealedPostingCursor<'a>,
}

impl SubstitutingArrivalCursor<'_> {
    /// What this wrapper *would* answer if anything asked it for history.
    ///
    /// Nothing can: there is no arrival method on [`PostingCursor`]. The
    /// method exists so the fixture can prove the two lists really disagree,
    /// which is what makes the negative non-vacuous.
    fn shadow_doc_before(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.shadow.sealed_doc_before(target)
    }
}

impl PostingCursor for SubstitutingArrivalCursor<'_> {
    fn doc(&self) -> Option<u32> {
        self.driven.doc()
    }

    fn freq(&self) -> Option<u32> {
        self.driven.freq()
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.driven.positions_handle()
    }

    fn size_hint(&self) -> u32 {
        self.driven.size_hint()
    }

    fn cost(&self) -> u64 {
        self.driven.cost()
    }

    fn segment_num_docs(&self) -> u32 {
        self.driven.segment_num_docs()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.driven.next()
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.driven.advance(target)
    }

    fn validated_post_move_contract(&self) -> Option<ValidatedPostMoveContract> {
        self.driven.validated_post_move_contract()
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        self.driven.fork_for_pruning()
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        self.driven
            .term_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn supports_block_max(&self) -> bool {
        self.driven.supports_block_max()
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        self.driven
            .current_block_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        self.driven.current_block_last_doc()
    }

    fn current_work_block(&self) -> Option<u64> {
        self.driven.current_work_block()
    }

    fn block_permit_for_next(&self) -> u64 {
        self.driven.block_permit_for_next()
    }

    fn block_permit_for_advance(&self, target: u32) -> u64 {
        self.driven.block_permit_for_advance(target)
    }
}

/// A wrapper cannot substitute the cursor whose history the union reads
/// (`bd-quill-pruned-topdocs-term-order-9wu3p`).
///
/// Sealing the mint, then sealing the result, then binding the result to
/// its probed document each closed one route and left this one: a wrapper
/// holding two sealed cursors, driving one and probing the other. Every
/// receipt it returned passed every check and described the wrong list.
///
/// The route is gone rather than guarded. There is no arrival method on
/// `PostingCursor`, so a wrapper has nothing to override and nothing to
/// forward: it is an opaque cursor, it reports no reachable last document,
/// and every pruned window takes the exact walk. The union reads a history
/// only from a value it also advances and scores.
#[test]
fn two_sealed_cursor_wrapper_cannot_substitute_arrival_history()
-> Result<(), Box<dyn std::error::Error>> {
    const NUM_DOCS: u32 = 24_097;
    const CUTOFF_DOCS: u32 = 4;
    const MEETING_DOC: u32 = 5_400;
    const TAIL_DOC: u32 = 20_000;
    const DRIVEN_BASE: u32 = 5_001;
    const SHADOW_BASE: u32 = 5_200;
    const PRIVATE_RUN: u32 = 20;
    const SMALL_TERMS: u32 = 6;
    const SNAPSHOT_DOC_FREQ: u64 = 8;

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

    // The two lists agree on every document the union visits and disagree
    // only in the private run beneath the meeting document — the history an
    // arrival order is built from. The shadow run is placed above the driven
    // one so a substituted answer would reorder the clauses, not merely
    // perturb them.
    let cutoff_docs = || (0..CUTOFF_DOCS).map(|doc| Posting::new(doc, 1));
    let head = || [Posting::new(MEETING_DOC, 1), Posting::new(TAIL_DOC, 1)];
    let mut driven_rows = vec![cutoff_docs().chain(head()).collect::<Vec<_>>()];
    let mut shadow_rows = driven_rows.clone();
    for index in 0..SMALL_TERMS {
        driven_rows.push(
            cutoff_docs()
                .chain(
                    (0..PRIVATE_RUN)
                        .map(|step| Posting::new(DRIVEN_BASE + index * PRIVATE_RUN + step, 1)),
                )
                .chain(head())
                .collect::<Vec<_>>(),
        );
        shadow_rows.push(
            cutoff_docs()
                .chain((0..PRIVATE_RUN).map(|step| {
                    Posting::new(
                        SHADOW_BASE + (SMALL_TERMS - 1 - index) * PRIVATE_RUN + step,
                        1,
                    )
                }))
                .chain(head())
                .collect::<Vec<_>>(),
        );
    }
    let boosts = [1.0e8_f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let fieldnorm_id = fieldnorm_to_id(1);
    let driven_terms = driven_rows
        .iter()
        .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
        .collect::<Result<Vec<_>, _>>()?;
    let shadow_terms = shadow_rows
        .iter()
        .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
        .collect::<Result<Vec<_>, _>>()?;
    let driven_lists = driven_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let shadow_lists = shadow_terms
        .iter()
        .map(|(postings, _)| postings.posting_list())
        .collect::<Result<Vec<_>, _>>()?;
    let driven_block_max = validated_block_max_entries(&driven_terms, &driven_lists, field)?;

    // Non-vacuity: the substitution must actually change the answer, or a
    // green test would prove nothing about substitution.
    let mut probe = SubstitutingArrivalCursor {
        driven: SealedPostingCursor::new(&driven_lists[1], NUM_DOCS)?,
        shadow: SealedPostingCursor::new(&shadow_lists[1], NUM_DOCS)?,
    };
    probe.advance(MEETING_DOC)?;
    assert_eq!(
        probe.doc(),
        Some(MEETING_DOC),
        "both lists must carry the meeting document, or the wrapper is detectable by position"
    );
    let honest = probe.driven.sealed_doc_before(MEETING_DOC)?;
    let substituted = probe.shadow_doc_before(MEETING_DOC)?;
    assert_ne!(
        honest, substituted,
        "the shadow list must answer differently, or this negative is vacuous"
    );

    let mut oracle = sealed_topdocs_union(
        &driven_lists,
        None,
        field,
        &snapshot,
        SNAPSHOT_DOC_FREQ,
        &boosts,
        NUM_DOCS,
    )?;
    let mut oracle_collector = TopDocsCollector::new(4, 0)?;
    oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
    let oracle_hits = oracle_collector.finish()?.hits;

    let mut substituting = wrapped_topdocs_union(
        &driven_lists,
        &driven_block_max,
        field,
        &snapshot,
        SNAPSHOT_DOC_FREQ,
        &boosts,
        NUM_DOCS,
        |driven| SubstitutingArrivalCursor {
            driven,
            shadow: SealedPostingCursor::new(&shadow_lists[0], NUM_DOCS)
                .expect("shadow fixture opens"),
        },
    )?;
    let mut collector = TopDocsCollector::new(4, 0)?;
    collector.collect(&mut substituting, &AllLiveDocs)?;
    let stats = substituting
        .union_pruning_stats()
        .expect("the ranked root is a buffered union");
    let hits = collector.finish()?.hits;

    assert_hits_bit_exact(&hits, &oracle_hits);
    let pruned = stats
        .max_score_windows
        .saturating_add(stats.block_max_wand_windows);
    assert!(
        pruned >= 1,
        "the fixture must still prune, or the route is untested"
    );
    assert_eq!(
        stats.arrival_walk_windows, pruned,
        "a wrapper must never reach the seeking fill, whatever it embeds"
    );
    Ok(())
}

/// Admits freely until armed, then refuses every admission the way an
/// exhausted real checkpoint does, counting every call it saw.
#[derive(Default)]
struct ArmedRefusalCheckpoint {
    armed: std::sync::atomic::AtomicBool,
    admissions: std::sync::atomic::AtomicUsize,
}

impl ArmedRefusalCheckpoint {
    fn arm(&self) {
        self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn admissions(&self) -> usize {
        self.admissions.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl QueryWorkCheckpoint for ArmedRefusalCheckpoint {
    fn admit(&self, _kind: QueryWorkKind, _units: u64) -> Result<(), ArgusError> {
        self.admissions
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.armed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(ArgusError::QueryFuelExhausted {
                budget: 11,
                consumed: 13,
                segments_touched: 17,
                dictionary_blocks: 19,
                posting_blocks: 23,
                position_docs: 29,
            });
        }
        Ok(())
    }
}

/// A deep backward descent is charged for the block it decodes and for
/// nothing else (`bd-quill-pruned-topdocs-term-order-9wu3p`).
///
/// The retained block is only worth having if the *charge* follows the
/// decode: a permit predicate that still reported "decodes" for every probe
/// would bill a query for work the cache elided, and one that reported
/// "free" for the first probe would let a decode run unadmitted. Both are
/// impossible by construction now — the read and its permit are written
/// against the same `ArrivalPlan` — and this pins the observable behaviour
/// of that: a parked-block control that costs nothing at all, then many
/// probes descending through one block for one physical decode and one
/// admitted posting block.
///
/// This is a cost unit and proves nothing about ordering. The semantic
/// control lives in the full `TopDocs` fixtures, which compare a pruned arm
/// bit-for-bit against an exhaustive oracle:
/// `pruned_topdocs_window_rebuilds_arrival_order_without_walking`,
/// `empty_block_max_wand_window_costs_seeks_not_steps` and
/// `randomized_pruned_topdocs_matches_stepped_arrival_order`.
#[test]
fn deep_backward_descent_is_charged_once() -> Result<(), Box<dyn std::error::Error>> {
    let postings = (0..=300)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    let checkpoint = Arc::new(CountingCheckpoint::default());
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let segment_num_docs = u32::try_from(postings.len())?;
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    let mut cursor = ValidatedArrivalCursor::new(sealed, checkpoint_for_cursor);

    cursor.advance(300)?;
    let decoded_after_seek = cursor.decoded_blocks();
    let admitted_after_seek = checkpoint
        .admitted_units
        .load(std::sync::atomic::Ordering::SeqCst);

    // Negative control: probes answered inside the parked block cost
    // nothing at all. Without this the descent's "one decode, one unit"
    // would be equally satisfied by a cursor that charged for every probe
    // and happened to decode once.
    for probe in [299_u32, 298, 280, 257] {
        assert_eq!(
            cursor.arrival_doc_before(probe)?,
            Some(probe - 1),
            "parked probe={probe}"
        );
    }
    assert_eq!(
        cursor.decoded_blocks(),
        decoded_after_seek,
        "a probe inside the parked block must not decode"
    );
    assert_eq!(
        checkpoint
            .admitted_units
            .load(std::sync::atomic::Ordering::SeqCst),
        admitted_after_seek,
        "a probe inside the parked block must not be charged"
    );

    // Descend out of the parked block, one posting at a time.
    let mut probes = 0_usize;
    let mut probe = 256_u32;
    while probe > 128 {
        let answer = cursor
            .arrival_doc_before(probe)?
            .expect("a dense fixture always has an earlier posting");
        assert_eq!(answer, probe - 1, "probe={probe}");
        probes += 1;
        probe = answer;
    }

    assert!(
        probes >= 32,
        "the descent must probe deeply enough to expose re-decoding: {probes} probes"
    );
    assert_eq!(
        cursor.decoded_blocks() - decoded_after_seek,
        1,
        "a descent through one retained block must decode it exactly once"
    );
    assert_eq!(
        checkpoint
            .admitted_units
            .load(std::sync::atomic::Ordering::SeqCst)
            - admitted_after_seek,
        1,
        "exactly the decoded block may be charged"
    );
    Ok(())
}

/// A refused arrival probe is terminal for the cursor that saw it
/// (`bd-quill-pruned-topdocs-term-order-9wu3p`).
///
/// The probe once admitted through the checkpoint directly from `&self`, on
/// the reasoning that a probe never moves and so leaves the cursor nowhere a
/// refusal has to freeze. That defends the wrong invariant. The doctrine is
/// about admission: a refused cursor "decodes nothing, mutates nothing, and
/// admits nothing again". Without a latch the next move polled and charged a
/// query already told to stop.
///
/// The admission count is what proves the latch. A second refusal with
/// identical fields could be a fresh refusal from a deterministic
/// checkpoint; a second refusal with *no additional admission* can only be
/// the retained one.
#[test]
fn refused_arrival_probe_is_terminal_without_readmitting() -> Result<(), Box<dyn std::error::Error>>
{
    const PARKED_DOC: u32 = 200;
    const PROBE_TARGET: u32 = 50;

    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    assert!(
        list.block_count() > 1,
        "fixture must cross a posting-block boundary"
    );
    let checkpoint = Arc::new(ArmedRefusalCheckpoint::default());
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let segment_num_docs = u32::try_from(postings.len())?;
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    // The probe has to be one that actually decodes, or the refusal it is
    // supposed to trigger never happens and the test proves nothing. This
    // is read from the sealed cursor before it is moved into the validated
    // cursor that will own it.
    let mut cursor = ValidatedArrivalCursor::new(sealed, checkpoint_for_cursor);

    cursor.advance(PARKED_DOC)?;
    assert_eq!(cursor.doc(), Some(PARKED_DOC));

    let before_refusal = checkpoint.admissions();
    checkpoint.arm();

    /// The planted refusal's fields, or `None` for any other error, so a
    /// wrong variant fails as a mismatch rather than as a panic.
    const PLANTED_REFUSAL: (u64, u64, u64, u64, u64, u64) = (11, 13, 17, 19, 23, 29);
    let refusal_fields = |error: &ArgusError| match error {
        ArgusError::QueryFuelExhausted {
            budget,
            consumed,
            segments_touched,
            dictionary_blocks,
            posting_blocks,
            position_docs,
        } => Some((
            *budget,
            *consumed,
            *segments_touched,
            *dictionary_blocks,
            *posting_blocks,
            *position_docs,
        )),
        _ => None,
    };

    let first = cursor
        .arrival_doc_before(PROBE_TARGET)
        .expect_err("an armed checkpoint refuses the decoding probe");
    assert_eq!(
        refusal_fields(&first),
        Some(PLANTED_REFUSAL),
        "the refused probe must carry the planted refusal verbatim"
    );
    assert_eq!(
        checkpoint.admissions(),
        before_refusal + 1,
        "the probe admits exactly once before it decodes"
    );

    let repeated = cursor
        .arrival_doc_before(PROBE_TARGET)
        .expect_err("a refused cursor stays refused");
    assert_eq!(
        refusal_fields(&repeated),
        Some(PLANTED_REFUSAL),
        "the repeated probe must replay the retained refusal, not a fresh one"
    );
    assert_eq!(
        checkpoint.admissions(),
        before_refusal + 1,
        "a refused cursor must not admit again for a second probe"
    );

    let moved = cursor
        .next()
        .expect_err("a move after a refused probe repeats the refusal");
    assert_eq!(
        refusal_fields(&moved),
        Some(PLANTED_REFUSAL),
        "the move after a refused probe must replay the retained refusal"
    );
    assert_eq!(
        checkpoint.admissions(),
        before_refusal + 1,
        "a refused cursor must not admit again for a move"
    );
    assert_eq!(
        cursor.doc(),
        Some(PARKED_DOC),
        "a refused cursor stays exactly where the refusal found it"
    );
    Ok(())
}
