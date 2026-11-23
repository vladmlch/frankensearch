use std::mem::size_of;

use frankensearch_core::DocId;

use crate::contract::fieldnorm_to_id;
use crate::quiver::NumericValue;
use crate::schema::{Analyzer, FieldDescriptor, FieldKind, SchemaDescriptor};
use crate::scribe::{ArenaSpan, FIELD_PREFIX_BYTES, TERM_BUCKET_BYTES_ESTIMATE};

use super::arena::{BlockMeta, ChainState, TypedChainArena};
use super::types::{PostingRecord, TermChain};
use super::*;

#[path = "tests_lifecycle.rs"]
mod lifecycle;

const TEST_FIELDS: [FieldDescriptor; 4] = [
    FieldDescriptor {
        id: 0,
        name: "keyword",
        kind: FieldKind::Keyword,
        stored: false,
    },
    FieldDescriptor {
        id: 1,
        name: "positioned",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 2,
        name: "plain_text",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 3,
        name: "stored",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
];
pub(super) const TEST_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "delta-tests",
    fields: &TEST_FIELDS,
};

const VALUE_FIELDS: [FieldDescriptor; 5] = [
    FieldDescriptor {
        id: 0,
        name: "keyword",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "signed",
        kind: FieldKind::I64 {
            indexed: true,
            fast: false,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "unsigned",
        kind: FieldKind::U64 {
            indexed: true,
            fast: true,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 3,
        name: "fast_only",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "opaque",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
];
const VALUE_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "delta-value-tests",
    fields: &VALUE_FIELDS,
};

pub(super) fn norms(keyword: u32, positioned: u32, plain: u32) -> [DeltaFieldNorm; 3] {
    [
        DeltaFieldNorm {
            field_ord: 0,
            raw_length: keyword,
            fieldnorm_id: fieldnorm_to_id(keyword),
        },
        DeltaFieldNorm {
            field_ord: 1,
            raw_length: positioned,
            fieldnorm_id: fieldnorm_to_id(positioned),
        },
        DeltaFieldNorm {
            field_ord: 2,
            raw_length: plain,
            fieldnorm_id: fieldnorm_to_id(plain),
        },
    ]
}

fn value_norms(keyword: u32) -> [DeltaFieldNorm; 1] {
    [DeltaFieldNorm {
        field_ord: 0,
        raw_length: keyword,
        fieldnorm_id: fieldnorm_to_id(keyword),
    }]
}

pub(super) fn apply_positioned(
    delta: &mut DeltaSegment,
    global_docid: u32,
    document_id: &str,
    term: &[u8],
    positions: &[u32],
) -> Result<DeltaApply, DeltaError> {
    let frequency = u32::try_from(positions.len()).expect("test position count fits u32");
    let applied = delta.apply_document(
        global_docid,
        DocId::from(document_id),
        &norms(0, frequency, 0),
        &[DeltaTermPosting {
            field_ord: 1,
            term,
            frequency,
            positions: Some(positions),
        }],
    )?;
    assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
    Ok(applied)
}

#[cfg(feature = "conformance-internals")]
pub(super) fn conformance_snapshot_fixture(
    plain_term: &[u8],
    positions: &[u32],
    stored_bytes: &[u8],
    tombstoned_id: &str,
) -> Result<DeltaSnapshot, DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    let first_postings = [
        DeltaTermPosting {
            field_ord: 1,
            term: b"positioned",
            frequency: u32::try_from(positions.len()).expect("test position count fits u32"),
            positions: Some(positions),
        },
        DeltaTermPosting {
            field_ord: 2,
            term: plain_term,
            frequency: 1,
            positions: None,
        },
    ];
    delta.apply_document_with_values(
        0,
        DocId::from("doc-0"),
        0x1111,
        &norms(0, 2, 1),
        &first_postings,
        &[],
        &[DeltaStoredValue::new(3, stored_bytes)],
    )?;

    let second_positions = [8, 9];
    let second_postings = [
        DeltaTermPosting {
            field_ord: 1,
            term: b"positioned",
            frequency: 2,
            positions: Some(&second_positions),
        },
        DeltaTermPosting {
            field_ord: 2,
            term: b"plain-stable",
            frequency: 1,
            positions: None,
        },
    ];
    delta.apply_document_with_values(
        1,
        DocId::from("doc-1"),
        0x2222,
        &norms(0, 2, 1),
        &second_postings,
        &[],
        &[DeltaStoredValue::new(3, b"stored-stable")],
    )?;
    assert!(
        delta.delete_delta_id(tombstoned_id).is_some(),
        "fixture tombstone must name a live row"
    );
    Ok(delta.freeze(7))
}

pub(super) fn expected_single_positioned_bytes(term: &[u8], document_id: &str) -> usize {
    let term_table_bytes =
        FIELD_PREFIX_BYTES + term.len() + size_of::<ArenaSpan>() + TERM_BUCKET_BYTES_ESTIMATE;
    let posting_bytes = size_of::<BlockMeta>() + size_of::<PostingRecord>();
    let position_bytes = 2 * size_of::<BlockMeta>() + 3 * size_of::<u32>();
    let document_id_bytes = size_of::<DocId>() + document_id.len();
    let identity_bytes =
        size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE + document_id.len();
    let fieldnorm_bytes = 3 * (size_of::<u32>() + size_of::<u8>());
    let seal_sidecar_bytes = size_of::<Option<u64>>() + size_of::<Option<Vec<u8>>>();
    term_table_bytes
        + size_of::<TermChain>()
        + posting_bytes
        + position_bytes
        + size_of::<u32>()
        + document_id_bytes
        + size_of::<u32>()
        + size_of::<u32>()
        + fieldnorm_bytes
        + seal_sidecar_bytes
        + identity_bytes
}

#[test]
fn seal_sidecars_survive_tombstones_freeze_and_writer_reset() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(VALUE_SCHEMA, 0, usize::MAX)?;
    let fast_only = 31_u64;
    delta.apply_document_with_values(
        0,
        DocId::from("same-id"),
        0x1122_3344_5566_7788,
        &value_norms(1),
        &[],
        &[
            DeltaNumericValue::i64(1, -7),
            DeltaNumericValue::u64(2, 9),
            DeltaNumericValue::u64(3, fast_only),
        ],
        &[
            DeltaStoredValue::new(0, b"first-keyword"),
            DeltaStoredValue::new(4, b"first-opaque"),
        ],
    )?;

    assert_eq!(delta.content_hash(0), Some(0x1122_3344_5566_7788));
    assert_eq!(delta.numeric_value(1, 0), Some(NumericValue::I64(-7)));
    assert_eq!(delta.numeric_value(2, 0), Some(NumericValue::U64(9)));
    assert_eq!(
        delta.numeric_value(3, 0),
        Some(NumericValue::U64(fast_only))
    );
    assert_eq!(delta.stored_value(0, 0), Some(b"first-keyword".as_slice()));
    assert_eq!(
        delta.stored_value(1, 0),
        Some((-7_i64).to_le_bytes().as_slice())
    );
    assert_eq!(delta.stored_value(2, 0), None);
    assert_eq!(
        delta.stored_value(3, 0),
        Some(fast_only.to_le_bytes().as_slice())
    );
    assert_eq!(delta.stored_value(4, 0), Some(b"first-opaque".as_slice()));
    assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());

    let signed_from_stored = 42_i64.to_le_bytes();
    let next_fast_only = 63_u64;
    let replacement = delta.apply_document_with_values(
        1,
        DocId::from("same-id"),
        0x8877_6655_4433_2211,
        &value_norms(0),
        &[],
        &[
            DeltaNumericValue::u64(2, 17),
            DeltaNumericValue::u64(3, next_fast_only),
        ],
        &[
            DeltaStoredValue::new(1, &signed_from_stored),
            DeltaStoredValue::new(4, b"second-opaque"),
        ],
    )?;
    assert_eq!(replacement.replaced_delta_docid, Some(0));
    assert!(delta.is_tombstoned(0));
    assert_eq!(delta.content_hash(0), Some(0x1122_3344_5566_7788));
    assert_eq!(delta.stored_value(4, 0), Some(b"first-opaque".as_slice()));
    assert_eq!(delta.numeric_value(1, 1), Some(NumericValue::I64(42)));
    assert_eq!(
        delta.stored_value(1, 1),
        Some(signed_from_stored.as_slice())
    );
    assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());

    let frozen = delta.freeze(19);
    assert!(frozen.segment().is_tombstoned(0));
    assert_eq!(frozen.content_hash(0), Some(0x1122_3344_5566_7788));
    assert_eq!(frozen.numeric_value(1, 1), Some(NumericValue::I64(42)));
    assert_eq!(
        frozen.numeric_value(3, 1),
        Some(NumericValue::U64(next_fast_only))
    );
    assert_eq!(frozen.stored_value(4, 1), Some(b"second-opaque".as_slice()));

    delta.reset_after_seal(0)?;
    assert_eq!(delta.content_hash(0), None);
    assert_eq!(delta.numeric_value(1, 1), None);
    assert_eq!(delta.stored_value(4, 1), None);
    assert_eq!(delta.bytes_used(), 0);

    delta.apply_document(2, DocId::from("convenience"), &value_norms(0), &[])?;
    assert_eq!(delta.content_hash(2), None);
    assert_eq!(delta.numeric_value(1, 2), None);
    assert_eq!(delta.stored_value(4, 2), None);
    assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
    assert_eq!(frozen.content_hash(1), Some(0x8877_6655_4433_2211));
    Ok(())
}

#[test]
fn seal_sidecar_validation_is_typed_and_atomic() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(VALUE_SCHEMA, 0, usize::MAX)?;
    let initial = delta.memory_stats();

    assert!(matches!(
        delta.apply_document_with_values(
            0,
            DocId::from("unknown"),
            1,
            &value_norms(0),
            &[],
            &[DeltaNumericValue::u64(5, 1)],
            &[],
        ),
        Err(DeltaError::UnknownValueField { field_ord: 5 })
    ));
    assert_eq!(delta.memory_stats(), initial);

    assert!(matches!(
        delta.apply_document_with_values(
            0,
            DocId::from("duplicate"),
            1,
            &value_norms(0),
            &[],
            &[DeltaNumericValue::i64(1, 1), DeltaNumericValue::i64(1, 2)],
            &[],
        ),
        Err(DeltaError::DuplicateValueField { field_ord: 1, .. })
    ));
    assert_eq!(delta.memory_stats(), initial);

    assert!(matches!(
        delta.apply_document_with_values(
            0,
            DocId::from("non-numeric"),
            1,
            &value_norms(0),
            &[],
            &[DeltaNumericValue::u64(0, 1)],
            &[],
        ),
        Err(DeltaError::NonNumericField { field_ord: 0, .. })
    ));
    assert_eq!(delta.memory_stats(), initial);

    assert!(matches!(
        delta.apply_document_with_values(
            0,
            DocId::from("fast-wrong-type"),
            1,
            &value_norms(0),
            &[],
            &[DeltaNumericValue::i64(3, -1)],
            &[],
        ),
        Err(DeltaError::NumericTypeMismatch {
            field_ord: 3,
            expected: "u64",
            actual: "i64",
            ..
        })
    ));
    assert_eq!(delta.memory_stats(), initial);

    assert!(matches!(
        delta.apply_document_with_values(
            0,
            DocId::from("wrong-type"),
            1,
            &value_norms(0),
            &[],
            &[DeltaNumericValue::u64(1, 1)],
            &[],
        ),
        Err(DeltaError::NumericTypeMismatch {
            field_ord: 1,
            expected: "i64",
            actual: "u64",
            ..
        })
    ));
    assert_eq!(delta.memory_stats(), initial);

    assert!(matches!(
        delta.apply_document_with_values(
            0,
            DocId::from("not-stored"),
            1,
            &value_norms(0),
            &[],
            &[],
            &[DeltaStoredValue::new(2, &1_u64.to_le_bytes())],
        ),
        Err(DeltaError::NonStoredField { field_ord: 2, .. })
    ));
    assert_eq!(delta.memory_stats(), initial);

    assert!(matches!(
        delta.apply_document_with_values(
            0,
            DocId::from("bad-width"),
            1,
            &value_norms(0),
            &[],
            &[],
            &[DeltaStoredValue::new(3, b"short")],
        ),
        Err(DeltaError::InvalidNumericBytes {
            field_ord: 3,
            expected: 8,
            actual: 5,
            ..
        })
    ));
    assert_eq!(delta.memory_stats(), initial);

    let signed = 1_i64.to_le_bytes();
    assert!(matches!(
        delta.apply_document_with_values(
            0,
            DocId::from("cross-slice-duplicate"),
            1,
            &value_norms(0),
            &[],
            &[DeltaNumericValue::i64(1, 1)],
            &[DeltaStoredValue::new(1, &signed)],
        ),
        Err(DeltaError::DuplicateValueField { field_ord: 1, .. })
    ));
    assert_eq!(delta.memory_stats(), initial);
    Ok(())
}

#[test]
fn unrolled_chains_grow_exponentially_then_cap() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    for global_docid in 0_u32..300 {
        let positions = [global_docid * 2, global_docid * 2 + 1];
        apply_positioned(&mut delta, global_docid, "shared-id", b"growth", &positions)?;
    }

    let term_index = delta
        .terms
        .find(1, b"growth")
        .expect("growth term is present");
    let chain = &delta.chains[usize::try_from(term_index).expect("u32 fits usize")];
    assert_eq!(
        delta.posting_arena.block_limits(&chain.postings),
        vec![1, 2, 4, 8, 16, 32, 64, 128, 128]
    );
    assert_eq!(
        delta.position_arena.block_limits(&chain.positions),
        vec![1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128]
    );
    assert_eq!(chain.postings.len, 300);
    assert_eq!(chain.positions.len, 600);
    assert!(chain.postings.tail.is_some());
    assert!(chain.positions.tail.is_some());

    let term = delta.find_term(1, b"growth").expect("growth term view");
    let last = term.postings().last().expect("last posting");
    assert_eq!(last.global_docid, 299);
    assert_eq!(
        term.positions(last)
            .expect("position view")
            .collect::<Vec<_>>(),
        [598, 599]
    );
    Ok(())
}

#[test]
fn arena_preflight_matches_mixed_reuse_replacement_and_growth() -> Result<(), DeltaError> {
    let mut arena = TypedChainArena::<u32>::new("mixed reuse test arena");
    let mut short_chains = [ChainState::default(); 3];
    arena.reserve_appends(&[
        (short_chains[0], 1),
        (short_chains[1], 1),
        (short_chains[2], 1),
    ])?;
    for (value, chain) in short_chains.iter_mut().enumerate() {
        arena.append_reserved(chain, u32::try_from(value).expect("small test value"));
    }
    assert_eq!(arena.blocks.len(), 3);
    assert!(arena.blocks.iter().all(|block| block.reserved == 1));

    arena.reset();
    let mut grown = ChainState::default();
    arena.reserve_appends(&[(grown, 8)])?;
    for value in 0_u32..8 {
        arena.append_reserved(&mut grown, value);
    }

    assert_eq!(arena.block_limits(&grown), vec![1, 2, 4, 8]);
    assert_eq!(
        arena.iter(&grown).copied().collect::<Vec<_>>(),
        (0..8).collect::<Vec<_>>()
    );
    assert_eq!(
        arena.blocks.len(),
        4,
        "one descriptor grows past high water"
    );
    assert_eq!(
        arena.blocks[0].reserved, 1,
        "first descriptor reuses in place"
    );
    assert_eq!(arena.blocks[1].reserved, 2, "second descriptor is replaced");
    assert_eq!(arena.blocks[2].reserved, 4, "third descriptor is replaced");
    assert_eq!(arena.blocks[3].reserved, 8, "fourth descriptor is new");
    Ok(())
}

#[test]
fn append_read_interleave_preserves_field_namespace_and_positions() -> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    let first_positions = [0, 0, 4];
    delta.apply_document(
        0,
        DocId::from("first"),
        &norms(1, 3, 2),
        &[
            DeltaTermPosting {
                field_ord: 0,
                term: b"same",
                frequency: 1,
                positions: None,
            },
            DeltaTermPosting {
                field_ord: 1,
                term: b"same",
                frequency: 3,
                positions: Some(&first_positions),
            },
            DeltaTermPosting {
                field_ord: 2,
                term: b"plain",
                frequency: 2,
                positions: None,
            },
        ],
    )?;

    {
        let keyword = delta.find_term(0, b"same").expect("keyword term");
        let positioned = delta.find_term(1, b"same").expect("positioned term");
        let plain = delta.find_term(2, b"plain").expect("positionless term");
        let keyword_posting = keyword.postings().next().expect("keyword posting");
        let positioned_posting = positioned.postings().next().expect("positioned posting");
        let plain_posting = plain.postings().next().expect("positionless posting");
        assert!(!keyword_posting.has_positions());
        assert!(keyword.positions(keyword_posting).is_none());
        assert_eq!(
            plain_posting.frequency, 1,
            "positionless fields retain Basic document-presence postings"
        );
        assert_eq!(
            positioned
                .positions(positioned_posting)
                .expect("positions")
                .collect::<Vec<_>>(),
            first_positions
        );
    }

    let second_positions = [2, 7];
    apply_positioned(&mut delta, 1, "second", b"same", &second_positions)?;
    let positioned = delta.find_term(1, b"same").expect("positioned term");
    let postings = positioned.postings().collect::<Vec<_>>();
    assert_eq!(
        postings
            .iter()
            .map(|posting| posting.global_docid)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(
        positioned
            .positions(postings[1])
            .expect("second positions")
            .collect::<Vec<_>>(),
        second_positions
    );
    assert_eq!(delta.raw_field_length(1, 0), Some(3));
    assert_eq!(delta.fieldnorm_id(1, 0), Some(fieldnorm_to_id(3)));
    assert_eq!(delta.live_total_tokens(1), Some(5));
    assert_eq!(delta.raw_field_length(3, 0), None);

    let sorted = delta
        .sorted_terms()
        .into_iter()
        .map(|term| (term.field_ord(), term.term().to_vec()))
        .collect::<Vec<_>>();
    assert_eq!(
        sorted,
        [
            (0, b"same".to_vec()),
            (1, b"same".to_vec()),
            (2, b"plain".to_vec()),
        ]
    );
    Ok(())
}

/// The maintained live count must equal a walk of each chain, on every
/// axis that can change liveness.
///
/// The walks are the definitions; the counters are the optimisation. This
/// covers independent chains, a replacement whose term set changes, an
/// already-set tombstone bit, and a reset generation, rather than merely
/// comparing one counter to a hand-picked number.
#[test]
fn maintained_live_count_matches_a_walk_on_every_liveness_axis() -> Result<(), DeltaError> {
    fn walk_live(delta: &DeltaSegment, term_bytes: &[u8]) -> usize {
        let term = delta
            .find_term(1, term_bytes)
            .expect("fixture term is present");
        term.postings().filter(|row| term.is_live(*row)).count()
    }
    fn assert_agrees(delta: &DeltaSegment, expected: &[(&[u8], usize)], label: &str) {
        for &(term_bytes, expected) in expected {
            let term = delta
                .find_term(1, term_bytes)
                .expect("fixture term is present");
            assert_eq!(
                term.live_doc_freq(),
                walk_live(delta, term_bytes),
                "counter disagreed with a walk for {term_bytes:?} after {label}"
            );
            assert_eq!(
                term.live_doc_freq(),
                expected,
                "unexpected live count for {term_bytes:?} after {label}"
            );
        }
    }

    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    let apply = |delta: &mut DeltaSegment,
                 docid: u32,
                 id: &str,
                 terms: &[&[u8]]|
     -> Result<(), DeltaError> {
        let position = [docid];
        let postings = terms
            .iter()
            .map(|term| DeltaTermPosting {
                field_ord: 1,
                term,
                frequency: 1,
                positions: Some(&position),
            })
            .collect::<Vec<_>>();
        delta.apply_document(
            docid,
            DocId::from(id),
            &norms(
                0,
                u32::try_from(terms.len()).expect("test term count fits u32"),
                0,
            ),
            &postings,
        )?;
        Ok(())
    };

    apply(&mut delta, 0, "a", &[b"alpha", b"beta"])?;
    apply(&mut delta, 1, "b", &[b"alpha", b"beta"])?;
    apply(&mut delta, 2, "c", &[b"beta", b"gamma"])?;
    assert_agrees(
        &delta,
        &[(b"alpha", 2), (b"beta", 3), (b"gamma", 1)],
        "independent fresh chains",
    );

    // Replacing b changes its term set. The old alpha and beta postings
    // retire before the new beta and delta postings become live.
    apply(&mut delta, 3, "b", &[b"beta", b"delta"])?;
    assert_agrees(
        &delta,
        &[(b"alpha", 1), (b"beta", 3), (b"gamma", 1), (b"delta", 1)],
        "replacement with changed terms",
    );

    // Deletion retires the surviving row of a without touching the other
    // independent chains.
    assert_eq!(delta.delete_delta_id("a"), Some(0));
    assert_agrees(
        &delta,
        &[(b"alpha", 0), (b"beta", 2), (b"gamma", 1), (b"delta", 1)],
        "one deletion",
    );

    // Superseded then deleted: only b's later physical row may retire.
    assert_eq!(delta.delete_delta_id("b"), Some(3));
    assert_agrees(
        &delta,
        &[(b"alpha", 0), (b"beta", 1), (b"gamma", 1), (b"delta", 0)],
        "supersede-then-delete",
    );

    // Deleting c sets its bit once. Re-marking that exact bit must leave
    // every counter intact; this is the duplicate-tombstone branch the
    // public identity path reaches after an already retired row.
    assert_eq!(delta.delete_delta_id("c"), Some(2));
    let before_duplicate_mark = delta
        .sorted_terms()
        .into_iter()
        .map(|term| (term.term().to_vec(), term.live_doc_freq()))
        .collect::<Vec<_>>();
    assert_eq!(delta.mark_tombstone(2), 0);
    assert_eq!(
        delta
            .sorted_terms()
            .into_iter()
            .map(|term| (term.term().to_vec(), term.live_doc_freq()))
            .collect::<Vec<_>>(),
        before_duplicate_mark,
        "an already-set tombstone bit must not decrement again"
    );
    assert_agrees(
        &delta,
        &[(b"alpha", 0), (b"beta", 0), (b"gamma", 0), (b"delta", 0)],
        "same-bit duplicate tombstone",
    );

    let canonical_terms = delta
        .sorted_terms()
        .into_iter()
        .map(|term| term.term().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        canonical_terms,
        [
            b"alpha".to_vec(),
            b"beta".to_vec(),
            b"delta".to_vec(),
            b"gamma".to_vec(),
        ],
        "dictionary enumeration remains canonical even when all rows are tombstoned"
    );
    assert_eq!(delta.term_count(), 4);

    // A frozen epoch carries the counters with its chains, needing no
    // publication-time walk.
    let snapshot = delta.freeze(7);
    let frozen = snapshot.segment();
    assert_agrees(
        frozen,
        &[(b"alpha", 0), (b"beta", 0), (b"gamma", 0), (b"delta", 0)],
        "freeze",
    );

    let generation_before_reset = delta.generation;
    delta.reset_after_seal(0)?;
    assert_eq!(delta.generation, generation_before_reset.wrapping_add(1));
    assert_eq!(delta.term_count(), 0);
    apply(&mut delta, 4, "next-generation", &[b"zeta", b"alpha"])?;
    assert_agrees(&delta, &[(b"alpha", 1), (b"zeta", 1)], "reset generation");
    assert_eq!(
        delta
            .sorted_terms()
            .into_iter()
            .map(|term| term.term().to_vec())
            .collect::<Vec<_>>(),
        [b"alpha".to_vec(), b"zeta".to_vec()],
        "a reset generation rebuilds dictionary order independently"
    );
    Ok(())
}

#[test]
fn interleaved_terms_keep_independent_chain_links_across_growth_boundaries()
-> Result<(), DeltaError> {
    let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
    for global_docid in 0_u32..300 {
        let alpha_position = [global_docid * 2];
        let beta_position = [global_docid * 2 + 1];
        delta.apply_document(
            global_docid,
            DocId::from("same-id"),
            &norms(0, 2, 0),
            &[
                DeltaTermPosting {
                    field_ord: 1,
                    term: b"alpha",
                    frequency: 1,
                    positions: Some(&alpha_position),
                },
                DeltaTermPosting {
                    field_ord: 1,
                    term: b"beta",
                    frequency: 1,
                    positions: Some(&beta_position),
                },
            ],
        )?;
    }

    let expected_limits = [1, 2, 4, 8, 16, 32, 64, 128, 128];
    for (term_bytes, parity) in [(b"alpha".as_slice(), 0), (b"beta".as_slice(), 1)] {
        let term_index = delta.terms.find(1, term_bytes).expect("interleaved term");
        let chain = &delta.chains[usize::try_from(term_index).expect("u32 fits usize")];
        assert_eq!(
            delta.posting_arena.block_limits(&chain.postings),
            expected_limits
        );
        assert_eq!(
            delta.position_arena.block_limits(&chain.positions),
            expected_limits
        );
        let term = delta.find_term(1, term_bytes).expect("term view");
        for posting in term.postings() {
            assert_eq!(
                term.positions(posting)
                    .expect("one position")
                    .collect::<Vec<_>>(),
                [posting.global_docid * 2 + parity]
            );
        }
    }
    Ok(())
}
