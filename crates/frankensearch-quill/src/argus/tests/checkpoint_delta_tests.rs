use super::*;

#[test]
fn checkpoint_cursor_charges_entered_blocks_and_admits_every_move()
-> Result<(), Box<dyn std::error::Error>> {
    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    let block_count = list.block_count();
    assert!(
        block_count > 1,
        "fixture must cross a posting-block boundary"
    );
    let checkpoint = Arc::new(CountingCheckpoint::default());
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let segment_num_docs = u32::try_from(postings.len())?;
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, checkpoint_for_cursor)?;

    assert!(PostingCursor::validated_post_move_contract(&cursor).is_some());

    let mut moves = 0_usize;
    while cursor.next()?.is_some() {
        moves += 1;
    }
    // The cursor starts on the first posting, so exhausting it takes one
    // move per remaining posting plus the final move that reports `None`.
    assert_eq!(moves + 1, postings.len());

    // The block already decoded at construction is charged by
    // `admit_initial_block` at the call site, which this test does not
    // call, so the cursor itself charges only the blocks its own moves
    // entered: one per boundary crossed.
    assert_eq!(
        checkpoint
            .admitted_units
            .load(std::sync::atomic::Ordering::SeqCst),
        u64::try_from(block_count - 1)?,
        "a move charges one unit per block it enters and nothing for an in-block step"
    );
    // Every move admits once to observe cancellation before it moves, and
    // a move that enters a block admits a second time for its permit.
    assert_eq!(
        checkpoint
            .admissions
            .load(std::sync::atomic::Ordering::SeqCst),
        postings.len() + block_count - 1,
        "one pre-move poll per move, plus one permit per entered block"
    );
    Ok(())
}

#[test]
fn checkpoint_cursor_surfaces_cancellation_on_an_in_block_move()
-> Result<(), Box<dyn std::error::Error>> {
    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    assert!(
        list.block_count() > 1,
        "fixture must cross a posting-block boundary"
    );
    // Admission 1 is the first move's pre-move poll. That move stays
    // inside the decoded block, so it never reaches a permit — it is
    // exactly the move a cursor that only admits entered blocks cannot
    // refuse, and it is refused here before it moves at all.
    let checkpoint = Arc::new(CancelOnNthAdmission::new(1));
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let segment_num_docs = u32::try_from(postings.len())?;
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, checkpoint_for_cursor)?;
    let entry_doc = PostingCursor::doc(&cursor);
    let entry_block = PostingCursor::current_work_block(&cursor);

    let error = cursor
        .next()
        .expect_err("an in-block step must observe a refusing checkpoint");
    assert!(
        matches!(
            error,
            ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test"
            }
        ),
        "expected the checkpoint's typed refusal, got {error:?}"
    );
    assert_eq!(
        checkpoint
            .units_at_fire
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the pre-move poll is admitted with zero units so it charges no fuel"
    );
    assert_eq!(
        checkpoint
            .admissions
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the refusal must stop the move before it reaches a permit"
    );
    assert_eq!(
        PostingCursor::doc(&cursor),
        entry_doc,
        "a refused move must not move the cursor"
    );
    assert_eq!(PostingCursor::current_work_block(&cursor), entry_block);
    Ok(())
}

#[test]
fn checkpoint_cursor_surfaces_cancellation_on_the_exhausting_move()
-> Result<(), Box<dyn std::error::Error>> {
    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    assert!(
        list.block_count() > 1,
        "fixture must cross a posting-block boundary"
    );
    // Each move polls once before moving, and the two moves that enter a
    // block add a permit, so the exhausting move's poll is admission
    // `moves + entered_blocks`. That move enters no block at all, which
    // makes it the other move a block-only cursor could never refuse.
    let exhausting_admission = postings.len() + list.block_count() - 1;
    let checkpoint = Arc::new(CancelOnNthAdmission::new(exhausting_admission));
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let segment_num_docs = u32::try_from(postings.len())?;
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, checkpoint_for_cursor)?;

    let mut moves = 0_usize;
    let error = loop {
        match cursor.next() {
            Ok(Some(_)) => moves += 1,
            Ok(None) => panic!("the exhausting move must observe the refusing checkpoint"),
            Err(error) => break error,
        }
    };
    assert_eq!(
        moves + 1,
        postings.len(),
        "the refusal must land on the exhausting move, not on an earlier one"
    );
    assert!(
        matches!(
            error,
            ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test"
            }
        ),
        "expected the checkpoint's typed refusal, got {error:?}"
    );
    assert_eq!(
        checkpoint
            .units_at_fire
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "an exhausted cursor enters no block, so its final move charges no fuel"
    );
    Ok(())
}

#[test]
fn checkpoint_cursor_charges_a_unit_only_where_it_enters_a_block()
-> Result<(), Box<dyn std::error::Error>> {
    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    let block_count = list.block_count();
    assert!(
        block_count > 1,
        "fixture must cross a posting-block boundary"
    );
    let checkpoint = Arc::new(UnitSequenceCheckpoint::default());
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let segment_num_docs = u32::try_from(postings.len())?;
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, checkpoint_for_cursor)?;

    while cursor.next()?.is_some() {}

    let units = checkpoint
        .units
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        units.len(),
        postings.len() + block_count - 1,
        "one pre-move poll per move, plus one permit per entered block"
    );
    assert_eq!(
        units.iter().sum::<u64>(),
        u64::try_from(block_count - 1)?,
        "the scan charges exactly one unit per block its moves entered"
    );
    assert_eq!(
        units.first(),
        Some(&0),
        "the first admission is a poll, before any movement"
    );
    assert_eq!(
        units.get(0..3),
        Some([0, 0, 0].as_slice()),
        "consecutive in-block steps poll for cancellation while charging nothing"
    );
    assert_eq!(
        units.last(),
        Some(&0),
        "the exhausting move enters no block and charges nothing"
    );
    // A permit is admitted immediately before the move that decodes, so
    // every unit in the sequence is preceded by that move's own poll.
    let permit_positions: Vec<usize> = units
        .iter()
        .enumerate()
        .filter_map(|(index, charged)| (*charged == 1).then_some(index))
        .collect();
    assert_eq!(permit_positions.len(), block_count - 1);
    for position in permit_positions {
        assert_eq!(
            units.get(position - 1),
            Some(&0),
            "a permit must follow its own move's poll"
        );
    }
    Ok(())
}

/// A skip that crosses more than one block still charges exactly one unit,
/// and that same admission is refusable.
///
/// Stepping with `next` cannot pin either fact: consecutive steps only ever
/// move one block, so there the per-move charge and the distance travelled
/// are indistinguishable. A sealed skip seek decodes only its destination
/// block, which is why its permit is one for any ordinal change, while a
/// Delta walk charges every logical block it crosses. This is therefore
/// the one move where a cursor that adopted the cumulative rule would
/// over-charge fuel, and the one move whose refusal a `next`-only test can
/// never observe.
#[test]
fn checkpoint_cursor_charges_one_unit_for_a_multi_block_skip_and_refuses_on_it()
-> Result<(), Box<dyn std::error::Error>> {
    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    assert!(
        list.block_count() >= 3,
        "fixture must offer a skip across at least two block boundaries"
    );
    let segment_num_docs = u32::try_from(postings.len())?;
    let skip_target = u32::try_from(postings.len() - 1)?;

    let recorder = Arc::new(UnitSequenceCheckpoint::default());
    let recorder_for_cursor: Arc<dyn QueryWorkCheckpoint> = recorder.clone();
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, recorder_for_cursor)?;
    let entry_block = PostingCursor::current_work_block(&cursor);
    assert_eq!(cursor.advance(skip_target)?, Some(skip_target));
    let landed_block = PostingCursor::current_work_block(&cursor);
    let blocks_travelled = landed_block
        .zip(entry_block)
        .map_or(0, |(landed, entry)| landed.saturating_sub(entry));
    assert!(
        blocks_travelled >= 2,
        "the fixture must skip a whole block, not step: {entry_block:?} -> {landed_block:?}"
    );
    let units = recorder
        .units
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        units.as_slice(),
        [0, 1].as_slice(),
        "the skip polls first, then takes one permit for the block it will decode"
    );

    // The same skip must be refusable *at its permit*, before it moves.
    // Admission 1 is the pre-move poll and admission 2 is that permit.
    let checkpoint = Arc::new(CancelOnNthAdmission::new(2));
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, checkpoint_for_cursor)?;
    let refused_doc = PostingCursor::doc(&cursor);
    let refused_block = PostingCursor::current_work_block(&cursor);
    let error = cursor
        .advance(skip_target)
        .expect_err("a skip must observe a refusing checkpoint");
    assert!(
        matches!(
            error,
            ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test"
            }
        ),
        "expected the checkpoint's typed refusal, got {error:?}"
    );
    assert_eq!(
        checkpoint
            .units_at_fire
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the refused permit carries the single block the skip would have decoded"
    );
    assert_eq!(
        checkpoint
            .admissions
            .load(std::sync::atomic::Ordering::SeqCst),
        2,
        "the refusal must land on the permit, not on a later move"
    );
    assert_eq!(
        PostingCursor::doc(&cursor),
        refused_doc,
        "a permit refused before the seek must leave the cursor where it was"
    );
    assert_eq!(
        PostingCursor::current_work_block(&cursor),
        refused_block,
        "a refused permit must decode nothing"
    );
    Ok(())
}

/// A refusal at the construction permit must happen before anything is
/// decoded, and must therefore stop the caller building a cursor at all.
///
/// This is the pre-work gate at its earliest point: the first block is
/// decoded while the inner cursor is being constructed, so the only
/// honest place to admit it is *before* that construction runs. The test
/// proves the ordering by never constructing the inner cursor on the
/// refusing path — a design that admitted afterwards could not fail here.
#[test]
fn construction_permit_is_refused_before_any_decode() -> Result<(), Box<dyn std::error::Error>> {
    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    let checkpoint = Arc::new(CancelOnNthAdmission::new(1));

    let refusal = CheckpointPostingCursor::admit_initial_block(checkpoint.as_ref(), true)
        .expect_err("the construction permit must be refusable");
    assert!(
        matches!(
            refusal,
            ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test"
            }
        ),
        "expected the checkpoint's typed refusal, got {refusal:?}"
    );
    assert_eq!(
        checkpoint
            .units_at_fire
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the construction permit carries the block construction will decode"
    );
    assert_eq!(
        checkpoint
            .admissions
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "a refused construction permit must not admit twice"
    );

    // A term with nothing to decode is charged nothing, so an empty
    // posting list cannot be refused for work it will never do.
    let idle = Arc::new(CountingCheckpoint::default());
    CheckpointPostingCursor::admit_initial_block(idle.as_ref(), false)?;
    assert_eq!(
        idle.admitted_units
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "an empty term must not be charged for a decode it never performs"
    );

    // On the admitted path the cursor builds and its first move takes no
    // permit, because construction already paid for the decoded block.
    let admitting = Arc::new(CountingCheckpoint::default());
    CheckpointPostingCursor::admit_initial_block(admitting.as_ref(), true)?;
    let admitting_for_cursor: Arc<dyn QueryWorkCheckpoint> = admitting.clone();
    let sealed = SealedPostingCursor::new(&list, u32::try_from(postings.len())?)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, admitting_for_cursor)?;
    cursor.next()?;
    assert_eq!(
        admitting
            .admitted_units
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "an in-block first step adds no unit on top of the construction permit"
    );
    Ok(())
}

/// A refusal must end the cursor: no decode, no mutation, no admission,
/// and no externally visible progress on any later move.
///
/// The refusal is injected on an in-block step, so the move that carried
/// it charged nothing — which means every later admission this test
/// forbids would be pure post-refusal work. Both move shapes are driven
/// after the refusal, because they are separate entry points and a guard
/// on only one of them is exactly the bug this pins.
#[test]
fn refused_checkpoint_cursor_makes_no_further_progress_or_admission()
-> Result<(), Box<dyn std::error::Error>> {
    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    let segment_num_docs = u32::try_from(postings.len())?;
    // Admission 1 is the first move's pre-move poll, so the refusal lands
    // before the cursor has moved at all.
    let checkpoint = Arc::new(CancelOnNthAdmission::new(1));
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let sealed = SealedPostingCursor::new(&list, segment_num_docs)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, checkpoint_for_cursor)?;

    let refusal = cursor
        .next()
        .expect_err("the injected refusal must surface on the in-block step");
    assert!(
        matches!(
            refusal,
            ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test"
            }
        ),
        "expected the checkpoint's typed refusal, got {refusal:?}"
    );
    let refused_doc = PostingCursor::doc(&cursor);
    let refused_block = PostingCursor::current_work_block(&cursor);
    let refused_admissions = checkpoint
        .admissions
        .load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(
        refused_admissions, 1,
        "the pre-move poll is refused before the move reaches a permit"
    );

    for attempt in 0..2 {
        let repeated = cursor
            .next()
            .expect_err("a refused cursor must keep refusing next()");
        assert!(
            matches!(
                repeated,
                ArgusError::QueryCancelled {
                    phase: "checkpoint_cursor_unit_test"
                }
            ),
            "attempt {attempt} lost the refusal: {repeated:?}"
        );
        let skipped = cursor
            .advance(segment_num_docs - 1)
            .expect_err("a refused cursor must keep refusing advance()");
        assert!(
            matches!(
                skipped,
                ArgusError::QueryCancelled {
                    phase: "checkpoint_cursor_unit_test"
                }
            ),
            "attempt {attempt} lost the refusal on advance: {skipped:?}"
        );
    }

    assert_eq!(
        checkpoint
            .admissions
            .load(std::sync::atomic::Ordering::SeqCst),
        refused_admissions,
        "a refused cursor must not admit again, so it can charge no further fuel"
    );
    assert_eq!(
        PostingCursor::doc(&cursor),
        refused_doc,
        "a refused cursor must not move: its reported document must not change"
    );
    assert_eq!(
        PostingCursor::current_work_block(&cursor),
        refused_block,
        "a refused cursor must not decode: its work block must not change"
    );
    Ok(())
}

/// A repeated refusal must be the *same* refusal, field for field.
///
/// Fuel exhaustion carries the progress diagnostics a caller reports, so
/// substituting a cancellation — or a fuel error with re-derived counters
/// — would silently change what the query says happened. This is what
/// separates retaining the refusal from merely remembering that one
/// occurred.
#[test]
fn refused_checkpoint_cursor_repeats_the_exact_fuel_error() -> Result<(), Box<dyn std::error::Error>>
{
    let postings = (0..257)
        .map(|docid| Posting::new(docid, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&postings)?;
    let list = encoded.posting_list()?;
    let checkpoint = Arc::new(FuelExhaustedOnFirstAdmission::default());
    let checkpoint_for_cursor: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let sealed = SealedPostingCursor::new(&list, u32::try_from(postings.len())?)?;
    let mut cursor = CheckpointPostingCursor::new(sealed, checkpoint_for_cursor)?;

    let first = cursor.next().expect_err("fuel exhaustion must surface");
    let repeated = cursor
        .next()
        .expect_err("a fuel-exhausted cursor must keep refusing");
    for (label, error) in [("first", &first), ("repeated", &repeated)] {
        assert!(
            matches!(
                error,
                ArgusError::QueryFuelExhausted {
                    budget: FuelExhaustedOnFirstAdmission::BUDGET,
                    consumed: FuelExhaustedOnFirstAdmission::CONSUMED,
                    segments_touched: 1,
                    dictionary_blocks: 2,
                    posting_blocks: FuelExhaustedOnFirstAdmission::POSTING_BLOCKS,
                    position_docs: 0,
                }
            ),
            "{label} refusal lost its exact fuel diagnostics: {error:?}"
        );
    }
    assert_eq!(
        checkpoint
            .admissions
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the repeated refusal must come from the cursor, not from a fresh admission"
    );
    Ok(())
}

const DELTA_TOMBSTONE_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "positioned",
    kind: FieldKind::Text {
        analyzer: crate::schema::Analyzer::FrankensearchDefault,
        positions: true,
    },
    stored: false,
}];
const DELTA_TOMBSTONE_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "argus-delta-tombstone-tests",
    fields: &DELTA_TOMBSTONE_FIELDS,
};
/// Physical postings after the first live one, all superseded but one.
const DELTA_TOMBSTONE_RUN: u32 = 300;

/// One live posting, then a long run of superseded ones, then one live.
///
/// Re-applying the same document id supersedes its previous posting, so
/// the term's physical list holds `DELTA_TOMBSTONE_RUN - 1` tombstones
/// between the two live postings — the shape whose walk used to run
/// unadmitted from end to end.
fn delta_tombstone_snapshot() -> Result<DeltaSnapshot, Box<dyn std::error::Error>> {
    let mut delta = crate::delta::DeltaSegment::new(DELTA_TOMBSTONE_SCHEMA, 0, usize::MAX)?;
    let norms = [crate::delta::DeltaFieldNorm {
        field_ord: 0,
        raw_length: 1,
        fieldnorm_id: fieldnorm_to_id(1),
    }];
    let mut apply =
        |global_docid: u32, document_id: &str| -> Result<(), Box<dyn std::error::Error>> {
            let position = [global_docid];
            delta.apply_document(
                global_docid,
                DocId::from(document_id),
                &norms,
                &[crate::delta::DeltaTermPosting {
                    field_ord: 0,
                    term: b"alpha",
                    frequency: 1,
                    positions: Some(&position),
                }],
            )?;
            Ok(())
        };
    apply(0, "first-live")?;
    for ordinal in 1..=DELTA_TOMBSTONE_RUN {
        apply(ordinal, "superseded")?;
    }
    Ok(delta.freeze(1))
}

/// One exact logical block of live Delta rows.
///
/// This shape distinguishes the final real row at ordinal 127 from a
/// nonexistent row at ordinal 128: both are where a block-only accounting
/// implementation is tempted to charge a phantom next block.
fn delta_exact_block_snapshot() -> Result<DeltaSnapshot, Box<dyn std::error::Error>> {
    let mut delta = crate::delta::DeltaSegment::new(DELTA_TOMBSTONE_SCHEMA, 0, usize::MAX)?;
    let norms = [crate::delta::DeltaFieldNorm {
        field_ord: 0,
        raw_length: 1,
        fieldnorm_id: fieldnorm_to_id(1),
    }];
    for global_docid in 0..DELTA_LOGICAL_BLOCK_POSTINGS {
        let document_id = format!("exact-block-{global_docid}");
        let position = [global_docid];
        delta.apply_document(
            global_docid,
            DocId::from(document_id.as_str()),
            &norms,
            &[crate::delta::DeltaTermPosting {
                field_ord: 0,
                term: b"alpha",
                frequency: 1,
                positions: Some(&position),
            }],
        )?;
    }
    Ok(delta.freeze(1))
}

#[test]
fn delta_missing_term_open_polls_without_charging_a_row() -> Result<(), Box<dyn std::error::Error>>
{
    let snapshot = delta_tombstone_snapshot()?;
    let checkpoint = Arc::new(CancelOnNthAdmission::new(1));

    let Err(error) =
        DeltaPostingCursor::new_admitted(&snapshot, 0, b"missing", checkpoint.as_ref())
    else {
        panic!("a missing-term open must observe its cancellation poll")
    };
    assert!(
        matches!(
            error,
            ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test"
            }
        ),
        "expected the checkpoint's typed refusal, got {error:?}"
    );
    assert_eq!(
        checkpoint
            .units_at_fire
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "a missing term has no physical row to charge"
    );
    assert_eq!(
        checkpoint
            .admissions
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the missing-term open must issue exactly its zero-unit poll"
    );
    Ok(())
}

#[test]
fn delta_exact_logical_block_has_no_phantom_admission() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = delta_exact_block_snapshot()?;
    let checkpoint = Arc::new(UnitSequenceCheckpoint::default());
    let mut cursor = DeltaPostingCursor::new_admitted(&snapshot, 0, b"alpha", checkpoint.as_ref())?;
    let mut rows = usize::from(cursor.doc().is_some());
    while cursor.next_admitted(checkpoint.as_ref())?.is_some() {
        rows += 1;
    }

    assert_eq!(rows, usize::try_from(DELTA_LOGICAL_BLOCK_POSTINGS)?);
    let units = checkpoint
        .units
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        units.len(),
        usize::try_from(DELTA_LOGICAL_BLOCK_POSTINGS)? + 2,
        "one open poll, the first real block, then one poll for every remaining row and the final empty step"
    );
    assert_eq!(units.get(0..2), Some([0, 1].as_slice()));
    assert!(
        units[2..].iter().all(|&units| units == 0),
        "ordinal 128 is past the physical end and must not buy a second block"
    );
    assert_eq!(
        units.iter().sum::<u64>(),
        1,
        "exactly one 128-row Delta block was opened"
    );
    Ok(())
}

/// A Delta walk over a tombstone run admits every physical step, so it
/// can be refused in the middle of the run and stops there.
///
/// This is the case no wrapper-level admission could reach: the whole run
/// is traversed inside a single `next`, so admitting per move would have
/// left it uncancellable and charged its blocks afterwards. The counts
/// below are the accounting this walk actually produces — one poll per
/// physical step and one unit per logical block the run crosses.
#[test]
fn delta_tombstone_walk_is_admitted_and_refusable_mid_run() -> Result<(), Box<dyn std::error::Error>>
{
    let snapshot = delta_tombstone_snapshot()?;

    let recorder = Arc::new(UnitSequenceCheckpoint::default());
    let recorder_handle: Arc<dyn QueryWorkCheckpoint> = recorder.clone();
    let cursor =
        DeltaPostingCursor::new_admitted(&snapshot, 0, b"alpha", recorder_handle.as_ref())?;
    let mut cursor = CheckpointPostingCursor::new(cursor, Arc::clone(&recorder_handle))?;
    // Construction pulls the first live posting: one physical step, whose
    // ordinal opens the first logical block.
    {
        let units = recorder
            .units
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Opening polls before the lookup, then construction's first
        // real step opens ordinal 0's block. Metadata uses the frozen
        // O(1) live count, so it neither scans nor charges three blocks.
        assert_eq!(units.as_slice(), [0, 1].as_slice());
    }

    assert!(cursor.next()?.is_some(), "the run ends at a live posting");
    let units = recorder
        .units
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    // One zero-unit open poll, one construction step, one wrapper poll,
    // then one admission per remaining physical step of the run.
    assert_eq!(
        units.len(),
        usize::try_from(DELTA_TOMBSTONE_RUN)? + 3,
        "the open poll, construction's step, the pre-move poll, and one \
         admission per remaining physical step of the run"
    );
    assert_eq!(
        units.iter().sum::<u64>(),
        3,
        "one unit for each real physical block at ordinals 0, 128 and 256"
    );
    Ok(())
}

#[test]
fn delta_tombstone_walk_refused_mid_run_makes_no_further_progress()
-> Result<(), Box<dyn std::error::Error>> {
    let snapshot = delta_tombstone_snapshot()?;
    // Admission 1 is the zero-unit open poll, 2 is construction's step
    // at ordinal 0, 3 is the wrapper's pre-move poll, and the walk's step
    // at physical ordinal N is admission N + 3. Refuse at ordinal 128 —
    // a block boundary in the middle of the tombstone run.
    let checkpoint = Arc::new(CancelOnNthAdmission::new(131));
    let handle: Arc<dyn QueryWorkCheckpoint> = checkpoint.clone();
    let inner = DeltaPostingCursor::new_admitted(&snapshot, 0, b"alpha", handle.as_ref())?;
    let mut cursor = CheckpointPostingCursor::new(inner, Arc::clone(&handle))?;
    let doc_before = PostingCursor::doc(&cursor);
    let block_before = PostingCursor::current_work_block(&cursor);

    let error = cursor
        .next()
        .expect_err("the walk must observe the refusal mid-run");
    assert!(
        matches!(
            error,
            ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test"
            }
        ),
        "expected the checkpoint's typed refusal, got {error:?}"
    );
    assert_eq!(
        checkpoint
            .units_at_fire
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the refused step is the one that opens a logical block"
    );
    let admissions_at_refusal = checkpoint
        .admissions
        .load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(admissions_at_refusal, 131);
    assert_eq!(
        PostingCursor::doc(&cursor),
        doc_before,
        "a refused walk must not publish a new current posting"
    );
    assert_eq!(
        PostingCursor::current_work_block(&cursor),
        block_before,
        "a refused walk must not advance the logical block"
    );

    // Sticky: the refusal repeats without walking or admitting again.
    assert!(cursor.next().is_err());
    assert!(cursor.advance(u32::MAX).is_err());
    assert_eq!(
        checkpoint
            .admissions
            .load(std::sync::atomic::Ordering::SeqCst),
        admissions_at_refusal,
        "a refused Delta cursor must not resume walking"
    );
    assert_eq!(PostingCursor::doc(&cursor), doc_before);
    Ok(())
}

/// A refusal inside the walk must leave the raw cursor's physical state
/// exactly as the refused step found it.
///
/// The wrapper's terminal refusal hides this: it stops calling the inner
/// cursor at all. Here the Delta cursor is driven directly, so the only
/// thing standing between a refusal and a corrupted walk is that the
/// admission precedes the physical step. The resumed walk is the proof
/// that nothing was consumed or skipped — it must land on the next live
/// posting having read every remaining row exactly once.
#[test]
fn raw_delta_cursor_refused_mid_walk_keeps_its_physical_position()
-> Result<(), Box<dyn std::error::Error>> {
    let snapshot = delta_tombstone_snapshot()?;
    // Construction spends admission 1 on the zero-unit open poll and
    // admission 2 on the step at ordinal 0, so the walk's step at physical
    // ordinal N is admission N + 2. Refuse at ordinal 128, a block
    // boundary deep inside the tombstone run.
    let checkpoint = Arc::new(CancelOnNthAdmission::new(130));
    let mut cursor = DeltaPostingCursor::new_admitted(&snapshot, 0, b"alpha", checkpoint.as_ref())?;
    assert_eq!(cursor.current_ordinal, Some(0));
    assert_eq!(cursor.next_ordinal, 1);
    assert_eq!(cursor.physical_len, DELTA_TOMBSTONE_RUN + 1);

    let error = cursor
        .next_admitted(checkpoint.as_ref())
        .expect_err("the walk must observe the refusal mid-run");
    assert!(
        matches!(
            error,
            ArgusError::QueryCancelled {
                phase: "checkpoint_cursor_unit_test"
            }
        ),
        "expected the checkpoint's typed refusal, got {error:?}"
    );

    // The refused step never ran: its ordinal was not consumed, the last
    // physical docid is still the row before it, and the published
    // posting is untouched.
    assert_eq!(
        cursor.next_ordinal, 128,
        "the refused step must not consume its ordinal"
    );
    assert_eq!(
        cursor.last_physical_doc,
        Some(127),
        "the last physical row read must still be the one before the refusal"
    );
    assert_eq!(cursor.current_ordinal, Some(0));
    assert_eq!(cursor.doc(), Some(0));
    assert!(
        cursor.remaining.is_some(),
        "a refused walk must keep its physical iterator"
    );

    // Resuming unadmitted proves the iterator sat exactly at ordinal 128:
    // the walk reaches the trailing live posting and ends having consumed
    // every physical row once.
    assert_eq!(cursor.pull_next_live(None)?, Some(DELTA_TOMBSTONE_RUN));
    assert_eq!(cursor.next_ordinal, DELTA_TOMBSTONE_RUN + 1);
    assert_eq!(cursor.current_ordinal, Some(DELTA_TOMBSTONE_RUN));
    Ok(())
}

/// Live postings scattered between tombstones, across two block seams.
///
/// Every ordinal carries the same churn identity except the chosen ones,
/// so each unique identity stays live while the churn rows are superseded
/// down to their last.
fn delta_interleaved_snapshot() -> Result<DeltaSnapshot, Box<dyn std::error::Error>> {
    let mut delta = crate::delta::DeltaSegment::new(DELTA_TOMBSTONE_SCHEMA, 0, usize::MAX)?;
    let norms = [crate::delta::DeltaFieldNorm {
        field_ord: 0,
        raw_length: 1,
        fieldnorm_id: fieldnorm_to_id(1),
    }];
    for ordinal in 0..DELTA_TOMBSTONE_RUN {
        let identity = if DELTA_INTERLEAVED_LIVE.contains(&ordinal) {
            format!("live-{ordinal}")
        } else {
            "churn".to_owned()
        };
        let position = [ordinal];
        delta.apply_document(
            ordinal,
            DocId::from(identity.as_str()),
            &norms,
            &[crate::delta::DeltaTermPosting {
                field_ord: 0,
                term: b"alpha",
                frequency: 1,
                positions: Some(&position),
            }],
        )?;
    }
    Ok(delta.freeze(1))
}

/// Ordinals kept live: one inside the first logical block, then one just
/// past each of the 128 and 256 boundaries.
///
/// 258 rather than 250 on purpose — 250 still sits in the second block,
/// so it would not have exercised the third at all.
const DELTA_INTERLEAVED_LIVE: [u32; 3] = [10, 130, 258];

#[test]
fn delta_walk_skips_tombstones_across_block_boundaries() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = delta_interleaved_snapshot()?;
    // The final churn row survives as its identity's newest posting.
    let last_churn = DELTA_TOMBSTONE_RUN - 1;

    let mut cursor = DeltaPostingCursor::new(&snapshot, 0, b"alpha")?;
    assert_eq!(
        cursor.doc(),
        Some(DELTA_INTERLEAVED_LIVE[0]),
        "construction must skip the tombstones before the first live row"
    );
    let mut walked = vec![cursor.doc().ok_or("first live posting")?];
    while let Some(doc) = cursor.next()? {
        walked.push(doc);
    }
    assert_eq!(
        walked,
        vec![
            DELTA_INTERLEAVED_LIVE[0],
            DELTA_INTERLEAVED_LIVE[1],
            DELTA_INTERLEAVED_LIVE[2],
            last_churn,
        ],
        "the walk must yield exactly the live postings, in order"
    );

    // A seek crossing both seams in one call lands on the first live
    // posting at or past the target, not on a tombstone.
    let mut seeking = DeltaPostingCursor::new(&snapshot, 0, b"alpha")?;
    assert_eq!(
        seeking.advance(DELTA_INTERLEAVED_LIVE[1] - 1)?,
        Some(DELTA_INTERLEAVED_LIVE[1])
    );
    assert_eq!(
        seeking.advance(DELTA_INTERLEAVED_LIVE[2])?,
        Some(DELTA_INTERLEAVED_LIVE[2]),
        "a seek landing exactly on a live posting must not move past it"
    );
    assert_eq!(seeking.advance(last_churn)?, Some(last_churn));
    assert_eq!(seeking.advance(u32::MAX)?, None);
    Ok(())
}
