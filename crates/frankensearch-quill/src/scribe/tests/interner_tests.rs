use super::*;

use std::hash::BuildHasher;

use crate::contract::fieldnorm_to_id;
use crate::error::QuillError;
use crate::grimoire::MAX_TERM_BYTES;
use crate::schema::DEFAULT_SCHEMA;

#[test]
fn intern_roundtrip_and_dense_ids() {
    let mut interner = TermInterner::new();
    let a = interner.intern(0, b"alpha");
    let b = interner.intern(0, b"beta");
    let a2 = interner.intern(0, b"alpha");
    assert_eq!(a, 0);
    assert_eq!(b, 1);
    assert_eq!(a2, a, "re-intern must return the same id");
    assert_eq!(interner.len(), 2);
    assert_eq!(interner.field_and_term(a), (0, b"alpha".as_slice()));
    assert_eq!(interner.field_and_term(b), (0, b"beta".as_slice()));
}

#[test]
fn same_term_different_field_is_distinct() {
    let mut interner = TermInterner::new();
    let content = interner.intern(0, b"rust");
    let title = interner.intern(1, b"rust");
    assert_ne!(content, title, "field namespacing must separate terms");
    assert_eq!(interner.field_and_term(content), (0, b"rust".as_slice()));
    assert_eq!(interner.field_and_term(title), (1, b"rust".as_slice()));
}

#[test]
fn sorted_ids_match_composite_byte_order_and_field_grouping() {
    let mut interner = TermInterner::new();
    // Insert deliberately out of (field, term) order.
    let ids = [
        interner.intern(1, b"zebra"),
        interner.intern(0, b"zebra"),
        interner.intern(1, b"alpha"),
        interner.intern(0, b"alpha"),
        interner.intern(258, b"alpha"), // field 0x0102: BE prefix ordering test
    ];
    assert_eq!(ids.len(), 5);
    let sorted = interner.sorted_ids();
    let keys: Vec<Vec<u8>> = sorted
        .iter()
        .map(|id| interner.composite_key(*id).to_vec())
        .collect();
    let mut expect = keys.clone();
    expect.sort();
    assert_eq!(keys, expect, "sorted_ids must equal raw byte order");
    // Field grouping: all field-0 keys precede field-1, which precede 258.
    let fields: Vec<u16> = sorted
        .iter()
        .map(|id| interner.field_and_term(*id).0)
        .collect();
    assert_eq!(fields, vec![0, 0, 1, 1, 258]);
}

#[test]
fn msd_sorted_ids_match_comparison_oracle_on_shared_prefixes() {
    let mut interner = TermInterner::new();
    let mut rng = DeterministicRng(0x6d73_642d_7465_726d);
    for ordinal in (0_u32..4_096).rev() {
        let field_ord = u16::try_from(rng.choose(513)).expect("bounded field ordinal");
        let mut term = vec![b'a'; rng.choose(96)];
        term.extend_from_slice(&rng.next().to_be_bytes());
        term.extend_from_slice(&ordinal.to_be_bytes());
        interner.intern(field_ord, &term);
    }
    interner.intern(0, b"");
    interner.intern(0, b"a");
    interner.intern(0, b"a\0");
    interner.intern(u16::MAX, b"\xff");

    let actual = interner.sorted_ids();
    let mut expected =
        (0..u32::try_from(interner.len()).expect("bounded term count")).collect::<Vec<_>>();
    expected.sort_unstable_by(|a, b| interner.composite_key(*a).cmp(interner.composite_key(*b)));
    assert_eq!(actual, expected);
}

#[test]
fn collision_path_verifies_bytes() {
    // Every key collides: correctness must come from byte verification.
    let mut interner: TermInterner<ConstBuild> = TermInterner::with_hasher(ConstBuild::default());
    let mut ids = Vec::new();
    for i in 0..200u32 {
        let term = format!("term-{i:03}");
        ids.push(interner.intern((i % 3) as u16, term.as_bytes()));
    }
    // Re-intern everything; ids must be stable.
    for i in 0..200u32 {
        let term = format!("term-{i:03}");
        assert_eq!(
            interner.intern((i % 3) as u16, term.as_bytes()),
            ids[i as usize]
        );
    }
    assert_eq!(interner.len(), 200);
    let sorted = interner.sorted_ids();
    assert_eq!(sorted.len(), 200);
    let keys: Vec<Vec<u8>> = sorted
        .iter()
        .map(|id| interner.composite_key(*id).to_vec())
        .collect();
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "strict byte order");
}

#[test]
fn collision_bucket_growth_reports_exact_live_byte_increments() {
    let mut interner: TermInterner<ConstBuild> = TermInterner::with_hasher(ConstBuild::default());
    let mut ids = Vec::new();
    for (field_ord, term) in [(0, b"alpha".as_slice()), (1, b"beta"), (2, b"gamma")] {
        let before = interner.bytes_used();
        let (id, added_bytes) = interner.intern_accounted(field_ord, term);
        assert_eq!(interner.bytes_used() - before, added_bytes);
        ids.push(id);
    }
    assert_eq!(interner.find(0, b"alpha"), Some(ids[0]));
    assert_eq!(interner.find(1, b"beta"), Some(ids[1]));
    assert_eq!(interner.find(2, b"gamma"), Some(ids[2]));

    let before_duplicate = interner.bytes_used();
    let (duplicate, duplicate_bytes) = interner.intern_accounted(1, b"beta");
    assert_eq!(duplicate, ids[1]);
    assert_eq!(duplicate_bytes, 0);
    assert_eq!(interner.bytes_used(), before_duplicate);
}

#[test]
fn collision_bucket_reserved_bytes_match_full_rescan() {
    let mut interner: TermInterner<ConstBuild> = TermInterner::with_hasher(ConstBuild::default());
    assert_eq!(
        interner.bytes_reserved(),
        rescanned_term_interner_bytes_reserved(&interner)
    );

    for i in 0..257_u32 {
        let term = format!("collision-reservation-{i:03}");
        interner.intern((i % 5) as u16, term.as_bytes());
        assert_eq!(
            interner.bytes_reserved(),
            rescanned_term_interner_bytes_reserved(&interner),
            "cached collision capacity diverged after insertion {i}"
        );
    }

    let before_duplicate = interner.bytes_reserved();
    assert_eq!(
        interner.intern(3, b"collision-reservation-003"),
        3,
        "duplicate lookup must retain its original id"
    );
    assert_eq!(interner.bytes_reserved(), before_duplicate);
    assert_eq!(
        interner.bytes_reserved(),
        rescanned_term_interner_bytes_reserved(&interner)
    );

    let cloned = interner.clone();
    assert_eq!(
        cloned.bytes_reserved(),
        rescanned_term_interner_bytes_reserved(&cloned),
        "clone must recompute capacities through its cloned containers"
    );

    interner.reset();
    assert_eq!(
        interner.bytes_reserved(),
        rescanned_term_interner_bytes_reserved(&interner),
        "reset drops collision vectors while retaining outer containers"
    );
}

/// Pins the span-mirror invariant documented on [`BucketEntry`]:
/// `entry.span == spans[entry.id]` for every entry reachable from
/// `buckets`, across One buckets, forced `Many` collisions, duplicates,
/// and reset/reuse cycles. If an insert path ever stores a stale or
/// wrong span inline, `matches` would verify against the wrong key
/// bytes and dedup would silently break — this test catches the
/// mirror drift directly.
#[test]
fn bucket_entries_mirror_spans_across_collisions_and_reset() {
    fn assert_span_mirror<S: BuildHasher>(interner: &TermInterner<S>) {
        let mut seen = 0_usize;
        for bucket in interner.buckets.values() {
            match bucket {
                Bucket::One(entry) => {
                    assert_eq!(
                        entry.span, interner.spans[entry.id as usize],
                        "One-bucket entry span diverged from spans[{}]",
                        entry.id
                    );
                    seen += 1;
                }
                Bucket::Many(entries) => {
                    for entry in entries {
                        assert_eq!(
                            entry.span, interner.spans[entry.id as usize],
                            "Many-bucket entry span diverged from spans[{}]",
                            entry.id
                        );
                        seen += 1;
                    }
                }
            }
        }
        assert_eq!(
            seen,
            interner.len(),
            "every interned id must appear exactly once across buckets"
        );
    }

    // Normal hasher: One-dominant buckets, duplicates on the hit path.
    let mut interner = TermInterner::new();
    for i in 0..500_u32 {
        let term = format!("mirror-{i:03}");
        let first = interner.intern((i % 4) as u16, term.as_bytes());
        let again = interner.intern((i % 4) as u16, term.as_bytes());
        assert_eq!(first, again, "duplicate intern must dedup via the mirror");
    }
    assert_span_mirror(&interner);

    // Reset drops both containers together; reuse re-establishes the
    // mirror against fresh arena spans.
    interner.reset();
    assert_span_mirror(&interner);
    for i in 0..100_u32 {
        let term = format!("post-reset-{i:03}");
        interner.intern(0, term.as_bytes());
    }
    assert_span_mirror(&interner);

    // Constant hasher: every key rides the Many verification path.
    let mut colliding: TermInterner<ConstBuild> = TermInterner::with_hasher(ConstBuild::default());
    for i in 0..200_u32 {
        let term = format!("mirror-collide-{i:03}");
        colliding.intern((i % 3) as u16, term.as_bytes());
    }
    assert_span_mirror(&colliding);
}

#[test]
fn budget_accounting_is_monotone_and_bounded() {
    let mut interner = TermInterner::new();
    let long = vec![b'x'; MAX_TERM_BYTES];
    interner.intern(0, &long);
    let after_long = interner.bytes_used();
    interner.intern(0, b"z");
    let mut prev = interner.bytes_used();
    assert!(
        prev >= after_long,
        "reusing scratch for a shorter key must not reduce live bytes"
    );
    let mut term_bytes_total = long.len() + b"z".len() + 2 * FIELD_PREFIX_BYTES;
    for i in 0..5000u32 {
        let term = format!("budget-term-{i}");
        term_bytes_total += term.len() + FIELD_PREFIX_BYTES;
        interner.intern(0, term.as_bytes());
        let now = interner.bytes_used();
        assert!(now >= prev, "bytes_used must be monotone under inserts");
        prev = now;
    }
    // Accounting must at least cover the raw key bytes and stay within an
    // order of magnitude (approximation contract documented on bytes_used;
    // arena reservation rounds up to chunk granularity).
    assert!(prev >= term_bytes_total, "must cover raw key bytes");
    assert!(
        prev <= term_bytes_total.max(DEFAULT_ARENA_CHUNK_BYTES) * 10,
        "accounting should not wildly overestimate: {prev} vs {term_bytes_total}"
    );
}

#[test]
fn empty_arena_reserves_nothing_until_first_write() {
    let mut arena = ByteArena::with_chunk_size(DEFAULT_ARENA_CHUNK_BYTES);
    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_reserved(), 0);
    assert_eq!(arena.chunk_count(), 0);

    arena.reset();
    assert_eq!(arena.bytes_reserved(), 0);
    assert_eq!(arena.chunk_count(), 0);

    let span = arena.push(b"first-term");
    assert_eq!(arena.resolve(span), b"first-term");
    assert_eq!(arena.bytes_reserved(), DEFAULT_ARENA_CHUNK_BYTES);
    assert_eq!(arena.chunk_count(), 1);
}

#[test]
fn oversized_terms_get_dedicated_chunks_and_reset_drops_them() {
    let mut arena = ByteArena::with_chunk_size(4096);
    let big = vec![0xAB; 100_000];
    let span = arena.push(&big);
    assert_eq!(arena.resolve(span), big.as_slice());
    // Standard chunk still usable after the oversized insert.
    let small = arena.push(b"small");
    assert_eq!(arena.resolve(small), b"small");
    let chunks_with_big = arena.chunk_count();
    arena.reset();
    assert!(
        arena.chunk_count() < chunks_with_big,
        "reset must drop dedicated oversized chunks (RSS ratchet guard)"
    );
    assert_eq!(arena.bytes_used(), 0);
}

#[test]
fn reset_reuse_is_allocation_stable_across_cycles() {
    // Soak: after the first cycle establishes capacity, later identical
    // cycles must not grow reserved bytes or chunk count (the no-leak /
    // RSS-stability contract). Logs per-cycle stats for diagnosis.
    let mut interner = TermInterner::new();
    let mut reserved_after_cycle = Vec::new();
    for cycle in 0..8 {
        for i in 0..20_000u32 {
            let term = format!("cycle-term-{i}");
            interner.intern((i % 4) as u16, term.as_bytes());
        }
        let (used, reserved, chunks) = interner.arena_stats();
        tracing::info!(
            cycle,
            used,
            reserved,
            chunks,
            terms = interner.len(),
            "soak cycle"
        );
        reserved_after_cycle.push((reserved, chunks));
        interner.reset();
        assert_eq!(interner.len(), 0);
        assert!(interner.is_empty());
    }
    let (first_reserved, first_chunks) = reserved_after_cycle[1];
    for (i, (reserved, chunks)) in reserved_after_cycle.iter().enumerate().skip(2) {
        assert_eq!(
            (*reserved, *chunks),
            (first_reserved, first_chunks),
            "cycle {i} changed arena footprint — allocation not stable"
        );
    }
}

#[test]
fn empty_term_and_empty_interner_edges() {
    let mut interner = TermInterner::new();
    assert!(interner.is_empty());
    assert!(interner.sorted_ids().is_empty());
    // Empty term bytes are legal at this layer (the tokenizer never emits
    // them, but the interner must not corrupt on them).
    let id = interner.intern(7, b"");
    assert_eq!(interner.field_and_term(id), (7, b"".as_slice()));
    assert_eq!(interner.intern(7, b""), id);
}

#[test]
fn fieldnorm_columns_retain_exact_lengths_and_quantized_boundaries() {
    assert_eq!(fieldnorm_to_id(40), 40);
    assert_eq!(fieldnorm_to_id(41), 40);
    assert_eq!(fieldnorm_to_id(42), 41);

    let counts = [0_u32, 1, 40, 41, 42, 100];
    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    for (doc_ord, count) in counts.into_iter().enumerate() {
        let text = "word ".repeat(count as usize);
        accumulator
            .add_document(
                u32::try_from(doc_ord).expect("small doc ordinal"),
                &[IndexedFieldValue::new(1, &text)],
            )
            .expect("document accumulates");
    }
    let content = accumulator.field(1).expect("content field");
    assert_eq!(content.document_lengths(), &counts);
    let expected_ids: Vec<u8> = counts.into_iter().map(fieldnorm_to_id).collect();
    assert_eq!(content.fieldnorm_ids(), expected_ids);
    assert_eq!(
        content.total_tokens(),
        counts.into_iter().map(u64::from).sum::<u64>()
    );
}

#[test]
fn document_validation_is_atomic_and_ordinals_are_strictly_ascending() {
    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    assert!(matches!(
        accumulator.add_document(
            4,
            &[
                IndexedFieldValue::new(1, "first"),
                IndexedFieldValue::new(1, "duplicate"),
            ],
        ),
        Err(AccumulatorError::DuplicateField {
            doc_ord: 4,
            field_ord: 1
        })
    ));
    assert_eq!(accumulator.document_count(), 0);
    assert_eq!(accumulator.token_count(), 0);
    assert!(accumulator.terms().is_empty());

    assert!(matches!(
        accumulator.add_document(4, &[IndexedFieldValue::new(99, "unknown")]),
        Err(AccumulatorError::UnknownField { field_ord: 99 })
    ));
    assert!(matches!(
        accumulator.add_document(4, &[IndexedFieldValue::new(3, "stored")]),
        Err(AccumulatorError::NonStringField {
            field_ord: 3,
            field_name: "metadata_json"
        })
    ));
    assert_eq!(accumulator.document_count(), 0);

    accumulator
        .add_document(4, &[IndexedFieldValue::new(1, "valid")])
        .expect("first valid document");
    let token_count = accumulator.token_count();
    assert!(matches!(
        accumulator.add_document(4, &[]),
        Err(AccumulatorError::OutOfOrderDocument {
            previous: 4,
            current: 4
        })
    ));
    assert!(matches!(
        accumulator.add_document(3, &[]),
        Err(AccumulatorError::OutOfOrderDocument {
            previous: 4,
            current: 3
        })
    ));
    assert_eq!(accumulator.document_count(), 1);
    assert_eq!(accumulator.token_count(), token_count);

    // The analyzer-availability guard is bound to the FAMILY, not to the
    // accumulator: `ColumnarAccumulator::new` now uses
    // `CompiledAnalyzerFamily`, which implements every compiled pipeline so
    // that CASS-schema indexes are ingestable. The refusal therefore has to
    // be exercised through a deliberately narrow family — the default-only
    // tokenizer — which is what a caller supplying its own analyzer gets.
    assert!(matches!(
        ColumnarAccumulator::with_analyzer(
            UNSUPPORTED_ANALYZER_SCHEMA,
            FrankensearchTokenizer::default()
        ),
        Err(QuillError::Resource {
            resource: "analyzer pipeline",
            detail,
        })
            if detail.contains("field 0 (cass_text)")
                && detail.contains("CassHyphenNormalize")
    ));
    // The complete family accepts the same schema, which is the point of
    // the change: no compiled descriptor is unindexable.
    assert!(ColumnarAccumulator::new(UNSUPPORTED_ANALYZER_SCHEMA).is_ok());
}

#[test]
fn lease_relative_document_ordinal_boundary_is_enforced_atomically() {
    let mut at_boundary = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    at_boundary
        .add_document(
            DOC_ORDS_PER_LEASE - 1,
            &[IndexedFieldValue::new(1, "last legal document")],
        )
        .expect("last lease-relative ordinal is valid");
    assert_eq!(at_boundary.document_ords(), &[DOC_ORDS_PER_LEASE - 1]);

    let mut outside = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    assert_eq!(
        outside.add_document(
            DOC_ORDS_PER_LEASE,
            &[IndexedFieldValue::new(1, "must not mutate")],
        ),
        Err(AccumulatorError::DocumentOutsideLease {
            doc_ord: DOC_ORDS_PER_LEASE,
        })
    );
    assert_eq!(outside.document_count(), 0);
    assert_eq!(outside.token_count(), 0);
    assert!(outside.terms().is_empty());
}

#[test]
fn oversized_document_token_is_not_counted_but_keeps_following_position() {
    let oversized = "x".repeat(MAX_TERM_BYTES + 1);
    let text = format!("head {oversized} tail");
    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    let report = accumulator
        .add_document(0, &[IndexedFieldValue::new(1, &text)])
        .expect("document accumulates");
    assert_eq!(report.admitted_tokens, 2);
    assert_eq!(report.oversized_tokens, 1);
    let content = accumulator.field(1).expect("content field");
    assert_eq!(content.positions(), Some([0, 2].as_slice()));
    assert_eq!(content.document_lengths(), &[2]);
    assert_eq!(content.fieldnorm_ids(), &[fieldnorm_to_id(2)]);
    assert_eq!(content.total_tokens(), 2);
}

#[test]
fn near_two_mib_document_preserves_u32_phrase_tail_positions() {
    const TARGET_BYTES: usize = 2 * 1024 * 1024;
    // Exactly 32 bytes: the 65,535 repeats put `needle` at position 65,535
    // and `tail` at 65,536, so a u16 position would wrap across the phrase.
    const FILLER: &str = "abcdefghijklmnopqrstuvwxyz01234 ";
    const TAIL: &str = "needle tail";
    let repeats = (TARGET_BYTES - TAIL.len()) / FILLER.len();
    let mut text = String::with_capacity(repeats * FILLER.len() + TAIL.len());
    for _ in 0..repeats {
        text.push_str(FILLER);
    }
    text.push_str(TAIL);
    assert!(text.len() <= TARGET_BYTES);
    assert!(TARGET_BYTES - text.len() < FILLER.len());
    assert_eq!(repeats, usize::from(u16::MAX));

    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    accumulator
        .add_document(17, &[IndexedFieldValue::new(1, &text)])
        .expect("large document accumulates");
    let content = accumulator.field(1).expect("content field");
    let positions = content.positions().expect("content stores positions");
    let expected_needle = u32::try_from(repeats).expect("fixture position fits u32");
    assert_eq!(
        &positions[positions.len() - 2..],
        &[expected_needle, expected_needle + 1]
    );
    assert_eq!(positions[positions.len() - 2], u32::from(u16::MAX));
    assert!(positions[positions.len() - 1] > u32::from(u16::MAX));
    assert_eq!(&content.doc_ords()[content.len() - 2..], &[17, 17]);
    let needle_id = content.term_ids()[content.len() - 2];
    let tail_id = content.term_ids()[content.len() - 1];
    assert_eq!(
        accumulator.terms().field_and_term(needle_id),
        (1, b"needle".as_slice())
    );
    assert_eq!(
        accumulator.terms().field_and_term(tail_id),
        (1, b"tail".as_slice())
    );
    assert_eq!(
        content.document_lengths(),
        &[u32::try_from(repeats + 2).expect("fixture count fits u32")]
    );
    assert!(
        phrase_matches(content, 17, &[needle_id, tail_id]),
        "tail phrase must match beyond the u16 position boundary"
    );
    assert!(
        !phrase_matches(content, 17, &[tail_id, needle_id]),
        "reversed tail phrase must not match"
    );
}

#[test]
fn low_cardinality_columns_count_toward_budget_and_reset_reuses_capacity() {
    let text = "x ".repeat(20_000);
    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    let baseline = accumulator.bytes_reserved();
    let first = accumulator
        .add_document(0, &[IndexedFieldValue::new(1, &text)])
        .expect("first cycle");
    assert!(first.bytes_reserved > baseline + 20_000 * 8);
    assert!(accumulator.should_flush(first.bytes_used));
    assert!(!accumulator.should_flush(first.bytes_used + 1));
    let retained = first.bytes_reserved;
    let live = first.bytes_used;

    accumulator.reset();
    assert_eq!(accumulator.document_count(), 0);
    assert_eq!(accumulator.token_count(), 0);
    assert!(accumulator.terms().is_empty());
    for field in accumulator.fields() {
        assert!(field.document_lengths().is_empty());
        assert!(field.fieldnorm_ids().is_empty());
    }
    for field in accumulator.stored_fields() {
        assert_eq!(field.document_count(), 0);
        assert_eq!(field.offsets(), &[0]);
        assert!(field.presence().is_empty());
        assert!(field.blob().is_empty());
    }
    assert_eq!(accumulator.bytes_reserved(), retained);
    assert!(
        !accumulator.should_flush(live),
        "retained spare capacity must not force one-document flush cycles"
    );

    let second = accumulator
        .add_document(0, &[IndexedFieldValue::new(1, &text)])
        .expect("second cycle");
    assert_eq!(second.bytes_reserved, retained);
    assert_eq!(second.bytes_used, live);
    assert!(accumulator.should_flush(live));
}

#[test]
fn shorter_final_token_cannot_reduce_accumulator_live_bytes() {
    let long = "x".repeat(MAX_TERM_BYTES);
    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    let after_long = accumulator
        .add_document(0, &[IndexedFieldValue::new(1, &long)])
        .expect("maximum admitted token accumulates")
        .bytes_used;
    let after_short = accumulator
        .add_document(1, &[IndexedFieldValue::new(1, "y")])
        .expect("shorter token accumulates")
        .bytes_used;
    assert!(
        after_short >= after_long,
        "reusable analyzer scratch must not make live accounting decrease"
    );
}

#[test]
fn budget_includes_outer_field_column_allocation() {
    let accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    let outer_fields = accumulator
        .fields
        .capacity()
        .saturating_mul(std::mem::size_of::<FieldTokenColumns>());
    assert!(outer_fields > 0);
    let outer_stored_fields = accumulator
        .stored_fields
        .capacity()
        .saturating_mul(std::mem::size_of::<StoredFieldColumns>());
    assert!(outer_stored_fields > 0);
    let outer_numeric_fields = accumulator
        .numeric_fields
        .capacity()
        .saturating_mul(std::mem::size_of::<NumericFieldColumns>());
    assert!(outer_numeric_fields > 0);
    let exact_components = accumulator
        .terms
        .bytes_reserved()
        .saturating_add(outer_fields)
        .saturating_add(
            accumulator
                .fields
                .iter()
                .map(FieldTokenColumns::bytes_reserved)
                .sum::<usize>(),
        )
        .saturating_add(outer_stored_fields)
        .saturating_add(
            accumulator
                .stored_fields
                .iter()
                .map(StoredFieldColumns::bytes_reserved)
                .sum::<usize>(),
        )
        .saturating_add(outer_numeric_fields)
        .saturating_add(
            accumulator
                .numeric_fields
                .iter()
                .map(NumericFieldColumns::bytes_reserved)
                .sum::<usize>(),
        )
        .saturating_add(
            accumulator
                .document_ords
                .capacity()
                .saturating_mul(std::mem::size_of::<u32>()),
        )
        .saturating_add(accumulator.seen_fields.capacity().div_ceil(8))
        .saturating_add(accumulator.analyzer.bytes_reserved());
    assert_eq!(accumulator.bytes_reserved(), exact_components);
}
