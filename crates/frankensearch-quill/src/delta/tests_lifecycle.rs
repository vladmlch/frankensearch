use std::mem::size_of;

use frankensearch_core::DocId;

use crate::config::DEFAULT_DELTA_BUDGET_BYTES;
use crate::delta::arena::BlockMeta;
use crate::delta::types::PostingRecord;
use crate::delta::{
    DeltaError, DeltaSegment, DeltaTermPosting, MAX_GLOBAL_DOCID_EXCLUSIVE,
    TOMBSTONE_WORDS_PER_LEASE,
};
use crate::grimoire::MAX_TERM_BYTES;
use crate::scribe::{DOC_ORDS_PER_LEASE, TermInterner};

#[cfg(feature = "conformance-internals")]
use super::conformance_snapshot_fixture;
use super::{TEST_SCHEMA, apply_positioned, expected_single_positioned_bytes, norms};

#[test]
fn posting_handles_cannot_cross_delta_owners_or_generations() -> Result<(), DeltaError> {
    let mut left = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    let mut right = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    apply_positioned(&mut left, 0, "left", b"term", &[1, 2])?;
    apply_positioned(&mut right, 0, "right", b"term", &[7, 8])?;

    let left_term = left.find_term(1, b"term").expect("left term");
    let left_posting = left_term.postings().next().expect("left posting");
    let right_term = right.find_term(1, b"term").expect("right term");
    assert!(right_term.positions(left_posting).is_none());

    let mut stale_generation = right_term.postings().next().expect("right posting");
    stale_generation.generation = stale_generation.generation.wrapping_add(1);
    assert!(right_term.positions(stale_generation).is_none());
    Ok(())
}

#[test]
fn frozen_snapshot_is_immutable_and_rejects_cross_owner_postings() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    apply_positioned(&mut delta, 0, "first", b"term", &[1, 4])?;
    let frozen = delta.freeze(7);

    assert_eq!(frozen.keeper_generation(), 7);
    assert_ne!(delta.owner_id, frozen.segment.owner_id);
    {
        let mutable_term = delta.find_term(1, b"term").expect("mutable term");
        let mutable_posting = mutable_term.postings().next().expect("mutable posting");
        let frozen_term = frozen.find_term(1, b"term").expect("frozen term");
        let frozen_posting = frozen_term.postings().next().expect("frozen posting");

        assert_eq!(
            mutable_term
                .positions(mutable_posting)
                .expect("mutable owner positions")
                .collect::<Vec<_>>(),
            [1, 4]
        );
        assert_eq!(
            frozen_term
                .positions(frozen_posting)
                .expect("frozen owner positions")
                .collect::<Vec<_>>(),
            [1, 4]
        );
        assert!(frozen_term.positions(mutable_posting).is_none());
        assert!(mutable_term.positions(frozen_posting).is_none());
    }

    apply_positioned(&mut delta, 1, "second", b"term", &[9])?;
    assert_eq!(delta.physical_document_count(), 2);
    assert_eq!(delta.live_document_count(), 2);
    assert_eq!(frozen.segment().physical_document_count(), 1);
    assert_eq!(frozen.live_document_count(), 1);
    assert_eq!(
        frozen
            .find_term(1, b"term")
            .expect("held frozen term")
            .live_doc_freq(),
        1
    );
    assert_eq!(frozen.live_total_tokens(1), Some(2));
    Ok(())
}

#[test]
fn frozen_snapshot_ordered_terms_reuse_one_backing_and_isolate_successors() -> Result<(), DeltaError>
{
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    apply_positioned(&mut delta, 0, "zeta", b"zeta", &[0])?;
    apply_positioned(&mut delta, 1, "alpha", b"alpha", &[0])?;
    let frozen = delta.freeze(7);

    let first_view = frozen.ordered_terms();
    let first_backing = first_view.term_ids.as_ptr();
    let first_terms = first_view
        .map(|term| term.term().to_vec())
        .collect::<Vec<_>>();
    let second_view = frozen.ordered_terms();
    let second_backing = second_view.term_ids.as_ptr();
    let second_terms = second_view
        .map(|term| term.term().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(first_terms, [b"alpha".to_vec(), b"zeta".to_vec()]);
    assert_eq!(second_terms, first_terms);
    assert_eq!(
        first_backing, second_backing,
        "later readers must reuse the first frozen snapshot's retained IDs"
    );

    apply_positioned(&mut delta, 2, "middle", b"middle", &[0])?;
    let successor = delta.freeze(8);
    let successor_view = successor.ordered_terms();
    assert_eq!(
        successor_view
            .clone()
            .map(|term| term.term().to_vec())
            .collect::<Vec<_>>(),
        [b"alpha".to_vec(), b"middle".to_vec(), b"zeta".to_vec()]
    );
    assert_ne!(
        first_backing,
        successor_view.term_ids.as_ptr(),
        "a successor freeze must retain its own generation's IDs"
    );
    assert_eq!(
        frozen
            .ordered_terms()
            .map(|term| term.term().to_vec())
            .collect::<Vec<_>>(),
        first_terms,
        "a mutable successor must not change the frozen ordered view"
    );
    let rebound = frozen.rebind_keeper_generation(9);
    assert!(
        rebound.ordered_term_ids.get().is_none(),
        "rebinding must not copy a predecessor cache into a new owner"
    );
    Ok(())
}

#[test]
fn frozen_snapshot_field_ordered_terms_visit_only_requested_candidates() -> Result<(), DeltaError> {
    const OUTER_FIELD_TERMS: u32 = 32;

    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    for docid in 0..OUTER_FIELD_TERMS {
        let keyword_term = format!("keyword-{docid:02}");
        let plain_term = format!("plain-{docid:02}");
        delta.apply_document(
            docid,
            DocId::from(format!("outer-{docid}")),
            &norms(1, 0, 1),
            &[
                DeltaTermPosting {
                    field_ord: 0,
                    term: keyword_term.as_bytes(),
                    frequency: 1,
                    positions: None,
                },
                DeltaTermPosting {
                    field_ord: 2,
                    term: plain_term.as_bytes(),
                    frequency: 1,
                    positions: None,
                },
            ],
        )?;
    }
    let positions = [0];
    for (offset, term) in [&b"alpha"[..], &b"omega"[..]].into_iter().enumerate() {
        let docid = OUTER_FIELD_TERMS
            + u32::try_from(offset).expect("two target terms fit the Delta docid domain");
        delta.apply_document(
            docid,
            DocId::from(format!("target-{offset}")),
            &norms(0, 1, 0),
            &[DeltaTermPosting {
                field_ord: 1,
                term,
                frequency: 1,
                positions: Some(&positions),
            }],
        )?;
    }
    let frozen = delta.freeze(7);
    let global_view = frozen.ordered_terms();
    let field_view = frozen.ordered_terms_for_field(1);
    let field_start = usize::try_from(OUTER_FIELD_TERMS).expect("bounded fixture length");

    assert_eq!(global_view.len(), field_start * 2 + 2);
    assert_eq!(
        field_view.len(),
        2,
        "the view must retain only field-1 candidates"
    );
    assert_eq!(
        field_view.term_ids.as_ptr(),
        global_view.term_ids[field_start..].as_ptr(),
        "the field view must be a subslice of the one retained ID backing"
    );
    let mut visited = 0;
    let observed = field_view
        .inspect(|_| visited += 1)
        .map(|term| term.term().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        visited, 2,
        "the field view must not visit surrounding fields"
    );
    assert_eq!(observed, [b"alpha".to_vec(), b"omega".to_vec()]);
    Ok(())
}

#[cfg(feature = "conformance-internals")]
#[test]
fn conformance_digest_binds_content_beyond_lineage_and_live_count() -> Result<(), DeltaError> {
    let baseline = conformance_snapshot_fixture(b"plain-a", &[1, 4], b"stored-a", "doc-1")?;
    let rebound = baseline.rebind_keeper_generation(7);
    assert_eq!(
        baseline.publication_lineage(),
        rebound.publication_lineage()
    );
    assert_eq!(
        baseline
            .conformance_content_sha256()
            .expect("valid frozen schema"),
        rebound
            .conformance_content_sha256()
            .expect("valid rebound schema"),
        "owner-isolated copies of identical content must retain one witness"
    );

    let variants = [
        (
            "posting key",
            conformance_snapshot_fixture(b"plain-b", &[1, 4], b"stored-a", "doc-1")?,
        ),
        (
            "position payload",
            conformance_snapshot_fixture(b"plain-a", &[1, 5], b"stored-a", "doc-1")?,
        ),
        (
            "stored field",
            conformance_snapshot_fixture(b"plain-a", &[1, 4], b"stored-b", "doc-1")?,
        ),
        (
            "tombstone topology",
            conformance_snapshot_fixture(b"plain-a", &[1, 4], b"stored-a", "doc-0")?,
        ),
    ];
    let baseline_digest = baseline
        .conformance_content_sha256()
        .expect("valid baseline schema");
    for (changed_state, mut variant) in variants {
        variant.lineage_id = baseline.lineage_id;
        assert_eq!(
            baseline.publication_lineage(),
            variant.publication_lineage(),
            "{changed_state} fixture must preserve publication lineage"
        );
        assert_eq!(
            baseline.live_document_count(),
            variant.live_document_count(),
            "{changed_state} fixture must preserve live count"
        );
        assert_ne!(
            baseline_digest,
            variant
                .conformance_content_sha256()
                .expect("valid variant schema"),
            "{changed_state} must change the canonical Delta witness"
        );
    }
    Ok(())
}

#[test]
fn frozen_empty_epoch_drops_reset_retained_chain_storage() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    for docid in 0..260 {
        apply_positioned(
            &mut delta,
            docid,
            &format!("doc-{docid}"),
            b"term",
            &[docid],
        )?;
    }
    assert!(!delta.posting_arena.slots.is_empty());
    assert!(!delta.position_arena.slots.is_empty());

    let populated = delta.freeze(8);
    let populated_term = populated.find_term(1, b"term").expect("frozen term");
    let populated_postings = populated_term.postings().collect::<Vec<_>>();
    assert_eq!(populated_postings.len(), 260);
    for (expected_docid, posting) in (0_u32..260).zip(populated_postings) {
        assert_eq!(posting.global_docid, expected_docid);
        assert_eq!(
            populated_term
                .positions(posting)
                .expect("frozen multi-block position")
                .collect::<Vec<_>>(),
            [expected_docid]
        );
    }

    delta.reset_after_seal(0)?;
    assert_eq!(delta.posting_arena.active_blocks, 0);
    assert_eq!(delta.position_arena.active_blocks, 0);
    assert!(
        !delta.posting_arena.slots.is_empty(),
        "the mutable writer should retain reusable capacity"
    );

    let frozen = delta.freeze(9);
    assert!(frozen.segment.posting_arena.blocks.is_empty());
    assert!(frozen.segment.posting_arena.slots.is_empty());
    assert!(frozen.segment.position_arena.blocks.is_empty());
    assert!(frozen.segment.position_arena.slots.is_empty());
    assert_eq!(frozen.live_document_count(), 0);
    Ok(())
}

#[test]
fn budget_accuracy_has_an_independent_component_oracle() -> Result<(), DeltaError> {
    let term = b"oracle";
    let document_id = "oracle-id";
    let positions = [1, 2, 3];
    let expected = expected_single_positioned_bytes(term, document_id);

    for (budget, expected_seal) in [
        (expected - 1, true),
        (expected, true),
        (expected + 1, false),
    ] {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, budget)?;
        let applied = apply_positioned(&mut delta, 0, document_id, term, &positions)?;
        assert_eq!(applied.bytes_used, expected);
        assert_eq!(applied.seal_required, expected_seal);
        assert_eq!(delta.should_seal(), expected_seal);
    }

    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    apply_positioned(&mut delta, 0, document_id, term, &positions)?;
    apply_positioned(&mut delta, 1, document_id, term, &[4])?;
    let second_row_bytes = size_of::<BlockMeta>()
        + 2 * size_of::<PostingRecord>()
        + size_of::<BlockMeta>()
        + 4 * size_of::<u32>()
        + size_of::<u32>()
        + size_of::<DocId>()
        + document_id.len()
        + size_of::<u32>()
        + size_of::<u32>()
        + 3 * (size_of::<u32>() + size_of::<u8>())
        + size_of::<Option<u64>>()
        + size_of::<Option<Vec<u8>>>()
        + size_of::<u64>();
    assert_eq!(delta.bytes_used(), expected + second_row_bytes);
    Ok(())
}

#[test]
fn budget_accepts_whole_document_and_ignores_retained_reset_capacity() -> Result<(), DeltaError> {
    let positions = (0_u32..300).collect::<Vec<_>>();
    let mut probe = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    let used = apply_positioned(&mut probe, 0, "large", b"large", &positions)?.bytes_used;

    let mut exact = DeltaSegment::new(TEST_SCHEMA, 0, used)?;
    let exact_apply = apply_positioned(&mut exact, 0, "large", b"large", &positions)?;
    assert_eq!(exact_apply.bytes_used, used);
    assert!(exact_apply.seal_required);
    assert!(exact.should_seal());
    assert_eq!(exact.physical_document_count(), 1);
    let posting = exact
        .find_term(1, b"large")
        .expect("large term")
        .postings()
        .next()
        .expect("large posting");
    assert_eq!(posting.frequency, 300);

    let mut roomy = DeltaSegment::new(TEST_SCHEMA, 0, used + 1)?;
    let roomy_apply = apply_positioned(&mut roomy, 0, "large", b"large", &positions)?;
    assert!(!roomy_apply.seal_required);
    assert!(!roomy.should_seal());

    let reserved = exact.bytes_reserved();
    assert!(reserved >= exact.bytes_used());
    exact.reset_after_seal(u64::from(DOC_ORDS_PER_LEASE))?;
    assert_eq!(exact.bytes_used(), 0);
    assert!(exact.bytes_reserved() <= reserved);
    assert!(exact.bytes_reserved() > 0);
    assert!(!exact.should_seal());
    Ok(())
}

#[test]
fn upsert_delete_and_readd_keep_physical_history_but_filter_live_rows() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    apply_positioned(&mut delta, 0, "A", b"term", &[0])?;
    let replacement = apply_positioned(&mut delta, 1, "A", b"term", &[1])?;
    assert_eq!(replacement.replaced_delta_docid, Some(0));
    assert_eq!(delta.probe_id("A"), Some(1));
    assert!(delta.is_tombstoned(0));
    assert!(!delta.is_tombstoned(1));

    assert_eq!(delta.delete_delta_id("A"), Some(1));
    assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
    assert_eq!(delta.delete_delta_id("A"), None);
    assert_eq!(delta.probe_id("A"), None);
    assert!(delta.is_tombstoned(1));
    assert_eq!(delta.live_document_count(), 0);

    let readd = apply_positioned(&mut delta, 2, "A", b"term", &[2])?;
    assert_eq!(readd.replaced_delta_docid, None);
    assert_eq!(delta.probe_id("A"), Some(2));
    assert_eq!(delta.probe_id("sealed-only-miss"), None);
    assert_eq!(delta.physical_document_count(), 3);
    assert_eq!(delta.live_document_count(), 1);
    assert_eq!(delta.live_total_tokens(1), Some(1));

    let frozen = delta.freeze(7);
    assert_eq!(frozen.physical_document_count(), 3);
    assert_eq!(frozen.live_document_count(), 1);

    let term = delta.find_term(1, b"term").expect("term view");
    assert_eq!(term.physical_doc_freq(), 3);
    assert_eq!(term.live_doc_freq(), 1);
    assert_eq!(
        delta
            .live_documents()
            .map(|(docid, id)| (docid, id.as_str()))
            .collect::<Vec<_>>(),
        [(2, "A")]
    );
    Ok(())
}

#[test]
fn tombstones_cover_word_and_lease_boundaries() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    for global_docid in [0, 63, 64, DOC_ORDS_PER_LEASE - 1] {
        apply_positioned(
            &mut delta,
            global_docid,
            "boundary-id",
            b"boundary",
            &[global_docid],
        )?;
    }
    assert!(delta.is_tombstoned(0));
    assert!(delta.is_tombstoned(63));
    assert!(delta.is_tombstoned(64));
    assert!(!delta.is_tombstoned(DOC_ORDS_PER_LEASE - 1));
    assert_eq!(
        delta.delete_delta_id("boundary-id"),
        Some(DOC_ORDS_PER_LEASE - 1)
    );
    assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
    assert!(delta.is_tombstoned(DOC_ORDS_PER_LEASE - 1));
    assert_eq!(delta.tombstone_words.len(), TOMBSTONE_WORDS_PER_LEASE);
    assert_eq!(delta.live_document_count(), 0);
    Ok(())
}

#[test]
fn semantic_rejections_are_atomic() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    let initial = delta.memory_stats();
    let valid_norms = norms(0, 2, 0);
    let valid_positions = [0, 1];
    let valid_posting = DeltaTermPosting {
        field_ord: 1,
        term: b"valid",
        frequency: 2,
        positions: Some(&valid_positions),
    };

    assert!(matches!(
        delta.apply_document(0, DocId::from(""), &valid_norms, &[valid_posting]),
        Err(DeltaError::EmptyDocumentId)
    ));
    assert_eq!(delta.memory_stats(), initial);
    assert!(matches!(
        delta.apply_document(
            DOC_ORDS_PER_LEASE,
            DocId::from("outside"),
            &valid_norms,
            &[valid_posting]
        ),
        Err(DeltaError::DocumentOutsideLease { .. })
    ));
    assert_eq!(delta.memory_stats(), initial);
    assert!(matches!(
        delta.apply_document(
            0,
            DocId::from("few-norms"),
            &valid_norms[..2],
            &[valid_posting]
        ),
        Err(DeltaError::FieldNormCount { .. })
    ));
    assert_eq!(delta.memory_stats(), initial);

    let mut bad_norms = valid_norms;
    bad_norms[1].fieldnorm_id = bad_norms[1].fieldnorm_id.wrapping_add(1);
    assert!(matches!(
        delta.apply_document(0, DocId::from("bad-norm"), &bad_norms, &[valid_posting]),
        Err(DeltaError::FieldNormMismatch { .. })
    ));
    assert_eq!(delta.memory_stats(), initial);

    let mut wrong_order = valid_norms;
    wrong_order.swap(0, 1);
    assert!(matches!(
        delta.apply_document(
            0,
            DocId::from("wrong-order"),
            &wrong_order,
            &[valid_posting]
        ),
        Err(DeltaError::FieldNormOrder { .. })
    ));
    assert_eq!(delta.memory_stats(), initial);

    let descending = [2, 1];
    let invalid_cases = [
        DeltaTermPosting {
            field_ord: 1,
            term: b"missing-positions",
            frequency: 1,
            positions: None,
        },
        DeltaTermPosting {
            field_ord: 0,
            term: b"forbidden-positions",
            frequency: 2,
            positions: Some(&valid_positions),
        },
        DeltaTermPosting {
            field_ord: 1,
            term: b"count-mismatch",
            frequency: 1,
            positions: Some(&valid_positions),
        },
        DeltaTermPosting {
            field_ord: 1,
            term: b"descending",
            frequency: 2,
            positions: Some(&descending),
        },
        DeltaTermPosting {
            field_ord: 3,
            term: b"stored-only",
            frequency: 1,
            positions: None,
        },
        DeltaTermPosting {
            field_ord: 0,
            term: b"zero-frequency",
            frequency: 0,
            positions: None,
        },
    ];
    for posting in invalid_cases {
        assert!(
            delta
                .apply_document(0, DocId::from("invalid"), &valid_norms, &[posting])
                .is_err()
        );
        assert_eq!(delta.memory_stats(), initial);
    }

    let duplicate = [valid_posting, valid_posting];
    assert!(matches!(
        delta.apply_document(0, DocId::from("duplicate"), &valid_norms, &duplicate),
        Err(DeltaError::DuplicateDocumentTerm { .. })
    ));
    assert_eq!(delta.memory_stats(), initial);

    apply_positioned(&mut delta, 0, "accepted", b"valid", &valid_positions)?;
    let accepted = delta.memory_stats();
    assert!(matches!(
        apply_positioned(&mut delta, 0, "out-of-order", b"valid", &valid_positions),
        Err(DeltaError::DocumentOrder { .. })
    ));
    assert_eq!(delta.memory_stats(), accepted);
    assert!(matches!(
        delta.reset_after_seal(1),
        Err(DeltaError::MisalignedLeaseBase { .. })
    ));
    assert_eq!(delta.memory_stats(), accepted);
    assert_eq!(delta.lease_base(), 0);
    Ok(())
}

#[test]
fn term_length_boundary_matches_the_sealed_contract() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    let accepted_term = vec![b'a'; MAX_TERM_BYTES];
    delta.apply_document(
        0,
        DocId::from("max-term"),
        &norms(0, 0, 1),
        &[DeltaTermPosting {
            field_ord: 2,
            term: &accepted_term,
            frequency: 1,
            positions: None,
        }],
    )?;
    assert!(delta.find_term(2, &accepted_term).is_some());
    assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());

    let before = delta.memory_stats();
    let rejected_term = vec![b'b'; MAX_TERM_BYTES + 1];
    assert!(matches!(
        delta.apply_document(
            1,
            DocId::from("oversized-term"),
            &norms(0, 0, 1),
            &[DeltaTermPosting {
                field_ord: 2,
                term: &rejected_term,
                frequency: 1,
                positions: None,
            }]
        ),
        Err(DeltaError::TermTooLarge { .. })
    ));
    assert_eq!(delta.memory_stats(), before);
    Ok(())
}

#[test]
fn same_lease_reset_drops_all_logical_state_before_reuse() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    apply_positioned(&mut delta, 7, "old", b"old-term", &[1, 3])?;
    apply_positioned(&mut delta, 8, "old", b"old-term", &[5])?;
    let retained_before = delta.bytes_reserved();
    delta.reset_after_seal(0)?;

    assert_eq!(delta.bytes_used(), 0);
    assert_eq!(delta.next_docid_floor(), 9);
    assert!(delta.bytes_reserved() <= retained_before);
    assert_eq!(delta.probe_id("old"), None);
    assert!(delta.find_term(1, b"old-term").is_none());
    assert!(!delta.is_tombstoned(7));
    assert_eq!(delta.raw_field_length(1, 7), None);
    assert!(matches!(
        apply_positioned(&mut delta, 8, "overlap", b"new-term", &[9]),
        Err(DeltaError::DocumentBeforeFloor { .. })
    ));

    apply_positioned(&mut delta, 9, "new", b"new-term", &[11, 13])?;
    assert_eq!(delta.probe_id("new"), Some(9));
    assert_eq!(delta.live_document_count(), 1);
    let term = delta.find_term(1, b"new-term").expect("new term");
    let posting = term.postings().next().expect("new posting");
    assert_eq!(posting.global_docid, 9);
    assert_eq!(
        term.positions(posting)
            .expect("new positions")
            .collect::<Vec<_>>(),
        [11, 13]
    );
    assert_eq!(delta.raw_field_length(1, 9), Some(2));
    Ok(())
}

#[test]
fn reset_reuses_allocation_across_many_lease_cycles() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    let mut steady_reserved = None;
    for cycle in 0_u64..8 {
        let lease_base = cycle * u64::from(DOC_ORDS_PER_LEASE);
        assert_eq!(delta.lease_base(), lease_base);
        let lease_base_u32 = u32::try_from(lease_base).expect("test lease remains inside u32");
        for ordinal in 0_u32..128 {
            let global_docid = lease_base_u32 + ordinal;
            apply_positioned(&mut delta, global_docid, "same-id", b"soak", &[ordinal])?;
        }
        assert_eq!(delta.physical_document_count(), 128);
        assert_eq!(delta.live_document_count(), 1);
        assert_eq!(
            delta
                .find_term(1, b"soak")
                .expect("soak term")
                .physical_doc_freq(),
            128
        );
        let stats = delta.memory_stats();
        tracing::info!(
            cycle,
            bytes_used = stats.bytes_used,
            bytes_reserved = stats.bytes_reserved,
            term_count = stats.term_count,
            physical_document_count = stats.physical_document_count,
            live_document_count = stats.live_document_count,
            posting_blocks = stats.posting_blocks,
            position_blocks = stats.position_blocks,
            "delta arena reset/reuse soak cycle"
        );
        let reserved = delta.bytes_reserved();
        if let Some(expected) = steady_reserved {
            assert_eq!(
                reserved, expected,
                "retained bytes drifted in cycle {cycle}"
            );
        } else {
            steady_reserved = Some(reserved);
        }

        let next_lease = lease_base + u64::from(DOC_ORDS_PER_LEASE);
        delta.reset_after_seal(next_lease)?;
        assert_eq!(delta.bytes_used(), 0);
        assert!(delta.bytes_reserved() <= reserved);
        assert_eq!(delta.physical_document_count(), 0);
        assert_eq!(delta.live_document_count(), 0);
        assert!(delta.find_term(1, b"soak").is_none());
        assert!(!delta.should_seal());
    }
    Ok(())
}

#[test]
fn constructor_and_lease_boundaries_are_explicit() -> Result<(), DeltaError> {
    assert!(matches!(
        DeltaSegment::new(TEST_SCHEMA, 0, 0),
        Err(DeltaError::ZeroBudget)
    ));
    assert!(matches!(
        DeltaSegment::new(TEST_SCHEMA, 1, 1),
        Err(DeltaError::MisalignedLeaseBase { .. })
    ));
    assert!(matches!(
        DeltaSegment::new(TEST_SCHEMA, MAX_GLOBAL_DOCID_EXCLUSIVE, 1),
        Err(DeltaError::LeaseOutOfRange { .. })
    ));

    let final_lease = MAX_GLOBAL_DOCID_EXCLUSIVE - u64::from(DOC_ORDS_PER_LEASE);
    let mut delta = DeltaSegment::with_default_budget(TEST_SCHEMA, final_lease)?;
    assert_eq!(delta.budget_bytes(), DEFAULT_DELTA_BUDGET_BYTES);
    assert_eq!(delta.lease_end(), MAX_GLOBAL_DOCID_EXCLUSIVE);
    delta.apply_document(u32::MAX, DocId::from("last"), &norms(0, 0, 0), &[])?;
    assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
    assert_eq!(delta.probe_id("last"), Some(u32::MAX));
    assert!(matches!(
        delta.reset_after_seal(0),
        Err(DeltaError::LeaseRegression { .. })
    ));
    Ok(())
}

#[test]
fn shared_term_interner_reports_exact_delta_accounting() {
    let mut terms = TermInterner::new();
    let before = terms.bytes_used();
    let (alpha, alpha_bytes) = terms.intern_accounted(0, b"alpha");
    assert_eq!(terms.bytes_used() - before, alpha_bytes);

    let (alpha_again, duplicate_bytes) = terms.intern_accounted(0, b"alpha");
    assert_eq!(alpha_again, alpha);
    assert_eq!(duplicate_bytes, 0);

    let before_other_field = terms.bytes_used();
    let (other_field, other_field_bytes) = terms.intern_accounted(1, b"alpha");
    assert_ne!(other_field, alpha);
    assert_eq!(terms.bytes_used() - before_other_field, other_field_bytes);
    assert_eq!(terms.find(0, b"alpha"), Some(alpha));
    assert_eq!(terms.find(1, b"alpha"), Some(other_field));
}
