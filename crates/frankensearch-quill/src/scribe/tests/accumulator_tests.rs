use super::*;

use crate::contract::fieldnorm_to_id;
use crate::grimoire::MAX_TERM_BYTES;
use crate::quiver::NumericValue;
use crate::schema::{
    Analyzer as AnalyzerKind, CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA,
};
use crate::segment::{SectionKind, SegmentReader};

#[test]
#[should_panic(expected = "analyzed token position exceeds the u32 contract")]
fn tokenizer_position_increment_cannot_saturate() {
    let _ = next_token_position(u32::MAX);
}

#[test]
fn admission_drops_oversized_token_without_collapsing_position_gap() {
    let oversized = "X".repeat(MAX_TERM_BYTES + 1);
    let input = format!("head {oversized} tail");
    let mut analyzer = FrankensearchTokenizer::default();
    let mut admitted = Vec::new();
    let report = analyze_admitted(
        &mut analyzer,
        AnalyzerKind::FrankensearchDefault,
        &input,
        &mut |token| admitted.push((token.text.clone(), token.position)),
    )
    .expect("default family supports default analysis");
    assert_eq!(
        report,
        AnalysisReport {
            raw_tokens: 3,
            admitted_tokens: 2,
            oversized_tokens: 1,
        }
    );
    assert_eq!(
        admitted,
        vec![("head".to_owned(), 0), ("tail".to_owned(), 2)]
    );

    let maximum = "X".repeat(MAX_TERM_BYTES);
    let mut maximum_tokens = Vec::new();
    let maximum_report = analyze_admitted(
        &mut analyzer,
        AnalyzerKind::FrankensearchDefault,
        &maximum,
        &mut |token| maximum_tokens.push(token.text.len()),
    )
    .expect("default family supports default analysis");
    assert_eq!(maximum_report.admitted_tokens, 1);
    assert_eq!(maximum_report.oversized_tokens, 0);
    assert_eq!(maximum_tokens, vec![MAX_TERM_BYTES]);

    let lowercase_expands = "İ".repeat(MAX_TERM_BYTES / 3 + 1);
    assert!(lowercase_expands.len() < MAX_TERM_BYTES);
    let expansion_report = analyze_admitted(
        &mut analyzer,
        AnalyzerKind::FrankensearchDefault,
        &lowercase_expands,
        &mut |_| {},
    )
    .expect("default family supports default analysis");
    assert_eq!(expansion_report.raw_tokens, 1);
    assert_eq!(expansion_report.admitted_tokens, 0);
    assert_eq!(expansion_report.oversized_tokens, 1);
}

#[test]
fn injected_analyzer_preserves_same_position_duplicate_tokens() {
    let mut accumulator = ColumnarAccumulator::with_analyzer(DEFAULT_SCHEMA, SamePositionAnalyzer)
        .expect("valid schema and injected analyzer");
    accumulator
        .add_document(0, &[IndexedFieldValue::new(1, "compound-part")])
        .expect("injected analysis accumulates");
    let content = accumulator.field(1).expect("content column");
    assert_eq!(content.positions(), Some([0, 0].as_slice()));
    assert_eq!(content.document_lengths(), &[2]);
    assert_ne!(content.term_ids()[0], content.term_ids()[1]);
    assert_eq!(
        accumulator.terms().field_and_term(content.term_ids()[0]),
        (1, b"compound".as_slice())
    );
    assert_eq!(
        accumulator.terms().field_and_term(content.term_ids()[1]),
        (1, b"part".as_slice())
    );
}

#[test]
fn analyzer_family_dispatches_each_kind_in_a_mixed_schema() {
    let mut accumulator =
        ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, MixedCassAnalyzer)
            .expect("mixed CASS analyzer family covers the complete schema");
    accumulator
        .add_document(
            0,
            &[
                IndexedFieldValue::new(6, "hyphen source"),
                IndexedFieldValue::new(8, "prefix source"),
            ],
        )
        .expect("both analyzer kinds dispatch through one family");

    let hyphen = accumulator.field(6).expect("title column");
    let prefix = accumulator.field(8).expect("title_prefix column");
    assert_eq!(hyphen.positions(), Some([7].as_slice()));
    assert_eq!(prefix.positions(), None);
    assert_eq!(hyphen.document_lengths(), &[1]);
    assert_eq!(prefix.document_lengths(), &[1]);
    assert_eq!(
        accumulator.terms().field_and_term(hyphen.term_ids()[0]),
        (6, b"hyphen-dispatch".as_slice())
    );
    assert_eq!(
        accumulator.terms().field_and_term(prefix.term_ids()[0]),
        (8, b"prefix-dispatch".as_slice())
    );

    let mut default = FrankensearchTokenizer::default();
    assert_eq!(
        analyze_admitted(
            &mut default,
            AnalyzerKind::CassHyphenNormalize,
            "unsupported",
            &mut |_| {},
        ),
        Err(UnsupportedAnalysis {
            analyzer: AnalyzerKind::CassHyphenNormalize,
        })
    );
}

#[test]
fn native_cass_family_accumulates_alternatives_and_prefixes() {
    let mut accumulator =
        ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, CassAnalyzer::default())
            .expect("native CASS family covers the complete semantic schema");
    let report = accumulator
        .add_document(
            0,
            &[
                IndexedFieldValue::new(6, "BD-Q3FY search"),
                IndexedFieldValue::new(8, "BD-Q3FY"),
            ],
        )
        .expect("native CASS analysis accumulates");
    assert_eq!(report.admitted_tokens, 5);
    assert_eq!(report.oversized_tokens, 0);

    let title = accumulator.field(6).expect("title column");
    let title_prefix = accumulator.field(8).expect("title prefix column");
    assert_eq!(title.positions(), Some([0, 0, 0, 1].as_slice()));
    assert_eq!(title.document_lengths(), &[4]);
    assert_eq!(title_prefix.positions(), None);
    assert_eq!(title_prefix.document_lengths(), &[1]);

    let term_id = |field_ord: u16, term: &[u8]| {
        (0..u32::try_from(accumulator.terms().len()).expect("small term table"))
            .find(|id| accumulator.terms().field_and_term(*id) == (field_ord, term))
            .expect("expected CASS term is interned")
    };
    let compound = term_id(6, b"bd-q3fy");
    let left = term_id(6, b"bd");
    let right = term_id(6, b"q3fy");
    let search = term_id(6, b"search");
    // E1 owns the position-column contract. The production phrase-scorer
    // differential that consumes these alternatives belongs to E4.5.
    assert!(phrase_matches(title, 0, &[compound, search]));
    assert!(phrase_matches(title, 0, &[left, search]));
    assert!(phrase_matches(title, 0, &[right, search]));
    assert!(!phrase_matches(title, 0, &[left, right]));
    assert_eq!(
        accumulator
            .terms()
            .field_and_term(title_prefix.term_ids()[0]),
        (8, b"bd-q3fy".as_slice())
    );
}

#[test]
fn accumulator_columns_fieldnorms_and_namespacing_are_schema_driven() {
    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    let report = accumulator
        .add_document(
            3,
            &[
                IndexedFieldValue::new(2, "Rust"),
                IndexedFieldValue::new(0, ""),
                IndexedFieldValue::new(1, "Rust POL-358"),
            ],
        )
        .expect("document accumulates");
    assert_eq!(report.admitted_tokens, 5);
    assert_eq!(report.oversized_tokens, 0);
    assert_eq!(accumulator.document_ords(), &[3]);
    assert_eq!(accumulator.token_count(), 5);

    let id = accumulator.field(0).expect("id column");
    assert_eq!(id.doc_ords(), &[3]);
    assert_eq!(id.positions(), None);
    assert_eq!(id.document_lengths(), &[1]);
    assert_eq!(id.fieldnorm_ids(), &[fieldnorm_to_id(1)]);

    let content = accumulator.field(1).expect("content column");
    assert_eq!(content.doc_ords(), &[3, 3, 3]);
    assert_eq!(content.positions(), Some([0, 1, 2].as_slice()));
    assert_eq!(content.document_lengths(), &[3]);
    assert_eq!(content.fieldnorm_ids(), &[fieldnorm_to_id(3)]);
    assert_eq!(content.total_tokens(), 3);

    let title = accumulator.field(2).expect("title column");
    assert_eq!(title.doc_ords(), &[3]);
    assert_eq!(title.positions(), Some([0].as_slice()));
    assert_eq!(title.document_lengths(), &[1]);

    let content_rust = content.term_ids()[0];
    let title_rust = title.term_ids()[0];
    assert_ne!(content_rust, title_rust);
    assert_eq!(
        accumulator.terms().field_and_term(content_rust),
        (1, b"rust".as_slice())
    );
    assert_eq!(
        accumulator.terms().field_and_term(title_rust),
        (2, b"rust".as_slice())
    );
    assert_eq!(
        accumulator.terms().field_and_term(id.term_ids()[0]),
        (0, b"".as_slice()),
        "an empty keyword is one legal raw token"
    );

    accumulator
        .add_document(7, &[IndexedFieldValue::new(1, "---")])
        .expect("gapped ordinal accumulates");
    assert_eq!(accumulator.document_ords(), &[3, 7]);
    for field in accumulator.fields() {
        assert_eq!(field.document_lengths().len(), 2);
        assert_eq!(field.fieldnorm_ids().len(), 2);
    }
    assert_eq!(
        accumulator.field(0).expect("id").document_lengths(),
        &[1, 0]
    );
    assert_eq!(
        accumulator.field(1).expect("content").document_lengths(),
        &[3, 0]
    );
    assert_eq!(
        accumulator.field(2).expect("title").document_lengths(),
        &[1, 0]
    );
}

#[test]
fn positional_and_nonpositional_fields_have_exactly_their_schema_columns() {
    let mut accumulator = ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
    accumulator
        .add_document(
            11,
            &[
                IndexedFieldValue::new(2, "same term"),
                IndexedFieldValue::new(1, "same term"),
                IndexedFieldValue::new(0, "key"),
            ],
        )
        .expect("document accumulates");

    let positioned = accumulator.field(1).expect("positioned field");
    let unpositioned = accumulator.field(2).expect("unpositioned field");
    assert_eq!(positioned.len(), 2);
    assert_eq!(unpositioned.len(), 2);
    assert_eq!(positioned.doc_ords(), unpositioned.doc_ords());
    assert_eq!(positioned.positions(), Some([0, 1].as_slice()));
    assert_eq!(unpositioned.positions(), None);
    assert_eq!(positioned.document_lengths(), &[2]);
    assert_eq!(unpositioned.document_lengths(), &[2]);
    assert_eq!(positioned.fieldnorm_ids(), unpositioned.fieldnorm_ids());
    assert!(accumulator.field(3).is_none(), "stored-only has no columns");
}

#[test]
fn stored_columns_preserve_opaque_absent_and_present_empty_values() {
    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    let ordinal = 41_u64.to_le_bytes();
    accumulator
        .add_document_with_stored(
            5,
            &[
                IndexedFieldValue::new(0, "doc-a"),
                IndexedFieldValue::new(1, "body"),
                IndexedFieldValue::new(2, ""),
            ],
            &[
                StoredFieldValue::new(3, b""),
                StoredFieldValue::new(4, &ordinal),
            ],
        )
        .expect("first document accumulates");
    accumulator
        .add_document_with_stored(
            8,
            &[
                IndexedFieldValue::new(0, "doc-b"),
                IndexedFieldValue::new(1, ""),
            ],
            &[],
        )
        .expect("second document accumulates");

    assert_eq!(
        accumulator
            .stored_fields()
            .iter()
            .map(StoredFieldColumns::field_ord)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4]
    );
    let ids = accumulator.stored_field(0).expect("stored id");
    assert_eq!(ids.value(0), Some(b"doc-a".as_slice()));
    assert_eq!(ids.value(1), Some(b"doc-b".as_slice()));
    let titles = accumulator.stored_field(2).expect("stored title");
    assert_eq!(titles.value(0), Some(b"".as_slice()));
    assert_eq!(titles.value(1), None);
    let metadata = accumulator.stored_field(3).expect("stored metadata");
    assert_eq!(metadata.value(0), Some(b"".as_slice()));
    assert_eq!(metadata.value(1), None);
    assert_eq!(metadata.presence(), &[1, 0]);
    assert_eq!(metadata.offsets(), &[0, 0, 0]);
    assert_eq!(
        accumulator
            .stored_field(4)
            .expect("stored ordinal")
            .value(0),
        Some(ordinal.as_slice())
    );
}

#[test]
fn stored_columns_follow_each_shipped_descriptor_without_name_hardcoding() {
    fn assert_stored_fields<A: TokenAnalyzer>(
        accumulator: &ColumnarAccumulator<A>,
        expected: &[u16],
    ) {
        assert_eq!(
            accumulator
                .stored_fields()
                .iter()
                .map(StoredFieldColumns::field_ord)
                .collect::<Vec<_>>(),
            expected,
            "schema={} should derive stored ordinals only from descriptors",
            accumulator.schema().name
        );
    }

    let default = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid default schema");
    assert_stored_fields(&default, &[0, 1, 2, 3, 4]);

    let fsfs = ColumnarAccumulator::new(FSFS_CHUNK_SCHEMA).expect("valid FSFS schema");
    assert_stored_fields(&fsfs, &[0, 1, 2, 3, 4, 5, 7]);

    let cass = ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, CassAnalyzer::default())
        .expect("native analyzer covers the CASS descriptor");
    assert_stored_fields(&cass, &[0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]);
}

#[test]
fn stored_input_validation_is_atomic_and_rejects_divergent_duplicates() {
    let mut unstored = ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
    assert_eq!(
        unstored.add_document_with_stored(1, &[], &[StoredFieldValue::new(0, b"not stored")],),
        Err(AccumulatorError::NonStoredField {
            field_ord: 0,
            field_name: "key",
        })
    );
    assert_eq!(unstored.document_count(), 0);
    assert!(unstored.terms().is_empty());
    assert_eq!(
        unstored
            .stored_field(3)
            .expect("stored-only column")
            .document_count(),
        0
    );

    let mut duplicate = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
    assert_eq!(
        duplicate.add_document_with_stored(
            2,
            &[IndexedFieldValue::new(0, "indexed")],
            &[StoredFieldValue::new(0, b"different stored bytes")],
        ),
        Err(AccumulatorError::DuplicateField {
            doc_ord: 2,
            field_ord: 0,
        })
    );
    assert_eq!(duplicate.document_count(), 0);
    assert!(duplicate.terms().is_empty());
    for field in duplicate.stored_fields() {
        assert_eq!(field.document_count(), 0);
        assert_eq!(field.offsets(), &[0]);
    }
}

#[test]
fn numeric_input_is_typed_stored_and_validation_is_atomic() {
    let mut accumulator =
        ColumnarAccumulator::new(INDEXED_NUMERIC_SCHEMA).expect("valid numeric schema");
    let high_unsigned = (1_u64 << 63) + 17;
    accumulator
        .add_document_with_numeric(3, &[], &[IndexedNumericValue::u64(0, high_unsigned)])
        .expect("typed unsigned value accumulates");

    assert_eq!(accumulator.document_ords(), &[3]);
    assert_eq!(
        accumulator
            .numeric_field(0)
            .expect("indexed numeric column")
            .values(),
        &[Some(NumericValue::U64(high_unsigned))]
    );
    assert_eq!(
        accumulator
            .stored_field(0)
            .expect("stored numeric column")
            .value(0),
        Some(high_unsigned.to_le_bytes().as_slice())
    );

    assert_eq!(
        accumulator.add_document_with_numeric(4, &[], &[IndexedNumericValue::i64(0, -1)]),
        Err(AccumulatorError::NumericTypeMismatch {
            field_ord: 0,
            field_name: "sequence",
            expected: "u64",
            actual: "i64",
        })
    );
    assert_eq!(
        accumulator.add_document_with_stored(4, &[], &[StoredFieldValue::new(0, &[1, 2, 3])],),
        Err(AccumulatorError::InvalidNumericBytes {
            field_ord: 0,
            field_name: "sequence",
            expected: 8,
            actual: 3,
        })
    );
    assert_eq!(
        accumulator.add_document_with_values(
            4,
            &[],
            &[IndexedNumericValue::u64(0, 9)],
            &[StoredFieldValue::new(0, &9_u64.to_le_bytes())],
        ),
        Err(AccumulatorError::DuplicateField {
            doc_ord: 4,
            field_ord: 0,
        })
    );

    assert_eq!(accumulator.document_ords(), &[3]);
    assert_eq!(
        accumulator
            .numeric_field(0)
            .expect("indexed numeric column")
            .values(),
        &[Some(NumericValue::U64(high_unsigned))]
    );
    let stored = accumulator.stored_field(0).expect("stored numeric column");
    assert_eq!(stored.document_count(), 1);
    assert_eq!(
        stored.value(0),
        Some(high_unsigned.to_le_bytes().as_slice())
    );

    let next = 42_u64;
    accumulator
        .add_document_with_stored(4, &[], &[StoredFieldValue::new(0, &next.to_le_bytes())])
        .expect("rejected document did not advance accumulator state");
    assert_eq!(accumulator.document_ords(), &[3, 4]);
    assert_eq!(
        accumulator
            .numeric_field(0)
            .expect("indexed numeric column")
            .values(),
        &[
            Some(NumericValue::U64(high_unsigned)),
            Some(NumericValue::U64(next)),
        ]
    );
}

#[test]
fn sparse_numeric_seal_rebases_both_types_through_u32_max() {
    let mut accumulator = ColumnarAccumulator::new(SIGNED_UNSIGNED_NUMERIC_SCHEMA)
        .expect("valid signed/unsigned numeric schema");
    accumulator
        .add_document_with_numeric(
            0,
            &[],
            &[
                IndexedNumericValue::i64(0, -7),
                IndexedNumericValue::u64(1, 0),
            ],
        )
        .expect("first sparse numeric document accumulates");
    accumulator
        .add_document_with_numeric(17, &[], &[IndexedNumericValue::i64(0, i64::MAX)])
        .expect("middle sparse numeric document accumulates");
    accumulator
        .add_document_with_numeric(
            DOC_ORDS_PER_LEASE - 1,
            &[],
            &[
                IndexedNumericValue::i64(0, i64::MIN),
                IndexedNumericValue::u64(1, u64::MAX),
            ],
        )
        .expect("lease-boundary numeric document accumulates");

    let documents = [
        FlushDocumentInput::new(0, "first", 1),
        FlushDocumentInput::new(17, "middle", 2),
        FlushDocumentInput::new(DOC_ORDS_PER_LEASE - 1, "last", 3),
    ];
    let lease_docid_base = (1_u64 << 32) - u64::from(DOC_ORDS_PER_LEASE);
    let encoded = flush_accumulator(
        &accumulator,
        FlushSegmentInput {
            segment_id: 91,
            lease_docid_base,
            created_unix_s: 0,
            engine_version: 0,
            documents: &documents,
        },
    )
    .expect("sparse signed/unsigned segment seals");
    let reader = SegmentReader::from_owned(encoded.into_bytes(), SIGNED_UNSIGNED_NUMERIC_SCHEMA)
        .expect("sparse signed/unsigned segment reopens");
    assert_eq!(reader.header().docid_lo, lease_docid_base);
    assert_eq!(reader.header().docid_hi, 1_u64 << 32);
    let numeric_bytes = reader
        .section(SectionKind::NUMERIC)
        .expect("NUMERIC checksum verifies")
        .expect("numeric schema requires NUMERIC");
    let section = crate::quiver::NumericSection::parse(
        numeric_bytes,
        SIGNED_UNSIGNED_NUMERIC_SCHEMA,
        lease_docid_base,
        1_u64 << 32,
    )
    .expect("sparse NUMERIC payload validates");
    let base_docid = u32::try_from(lease_docid_base).expect("lease base fits u32");
    assert_eq!(
        section
            .field(0)
            .expect("signed field")
            .entries()
            .collect::<Vec<_>>(),
        vec![
            crate::quiver::NumericEntry::i64(i64::MIN, u32::MAX),
            crate::quiver::NumericEntry::i64(-7, base_docid),
            crate::quiver::NumericEntry::i64(i64::MAX, base_docid + 17),
        ]
    );
    assert_eq!(
        section
            .field(1)
            .expect("unsigned field")
            .entries()
            .collect::<Vec<_>>(),
        vec![
            crate::quiver::NumericEntry::u64(0, base_docid),
            crate::quiver::NumericEntry::u64(u64::MAX, u32::MAX),
        ]
    );
}

#[test]
fn randomized_accumulation_preserves_all_parallel_column_invariants() {
    const SEEDS: [u64; 12] = [
        1,
        0x9e37_79b9_7f4a_7c15,
        0xd1b5_4a32_d192_ed03,
        0x94d0_49bb_1331_11eb,
        0x2545_f491_4f6c_dd1d,
        0x1234_5678_9abc_def0,
        0x0fed_cba9_8765_4321,
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        u64::MAX,
    ];

    for seed in SEEDS {
        let mut rng = DeterministicRng(seed);
        let mut accumulator =
            ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
        let mut expected_doc_ords = Vec::new();
        let mut expected_lengths = [Vec::new(), Vec::new(), Vec::new()];
        let mut next_doc_ord = u32::try_from(rng.choose(4)).expect("small ordinal");

        for _ in 0..96 {
            let keyword = (rng.choose(4) != 0).then(|| randomized_text(&mut rng));
            let positioned = (rng.choose(5) != 0).then(|| randomized_text(&mut rng));
            let unpositioned = (rng.choose(3) != 0).then(|| randomized_text(&mut rng));

            let mut owned_values = Vec::new();
            if let Some(text) = keyword {
                expected_lengths[0].push(1);
                owned_values.push((0_u16, text));
            } else {
                expected_lengths[0].push(0);
            }
            if let Some(text) = positioned {
                expected_lengths[1]
                    .push(u32::try_from(analyzed_tokens(&text).len()).expect("small token count"));
                owned_values.push((1_u16, text));
            } else {
                expected_lengths[1].push(0);
            }
            if let Some(text) = unpositioned {
                expected_lengths[2]
                    .push(u32::try_from(analyzed_tokens(&text).len()).expect("small token count"));
                owned_values.push((2_u16, text));
            } else {
                expected_lengths[2].push(0);
            }
            if owned_values.len() > 1 {
                let swap_with = rng.choose(owned_values.len());
                owned_values.swap(0, swap_with);
            }
            let values: Vec<_> = owned_values
                .iter()
                .map(|(field_ord, text)| IndexedFieldValue::new(*field_ord, text))
                .collect();

            expected_doc_ords.push(next_doc_ord);
            accumulator
                .add_document(next_doc_ord, &values)
                .expect("generated document accumulates");
            next_doc_ord += u32::try_from(rng.choose(3) + 1).expect("small ordinal step");
        }

        assert_eq!(
            accumulator.document_ords(),
            expected_doc_ords,
            "seed={seed:#x} document ordinals"
        );
        for (field_index, expected) in expected_lengths.iter().enumerate() {
            let field_ord = u16::try_from(field_index).expect("small field index");
            let field = accumulator
                .field(field_ord)
                .expect("generated field column");
            assert_eq!(
                field.term_ids().len(),
                field.doc_ords().len(),
                "seed={seed:#x} field={field_ord} term/doc parallelism"
            );
            if let Some(positions) = field.positions() {
                assert_eq!(
                    positions.len(),
                    field.term_ids().len(),
                    "seed={seed:#x} field={field_ord} position parallelism"
                );
            }
            assert_eq!(
                field.document_lengths(),
                expected,
                "seed={seed:#x} field={field_ord} document lengths"
            );
            let expected_norms: Vec<_> = expected.iter().copied().map(fieldnorm_to_id).collect();
            assert_eq!(
                field.fieldnorm_ids(),
                expected_norms,
                "seed={seed:#x} field={field_ord} fieldnorms"
            );
            assert_eq!(
                field.total_tokens(),
                expected.iter().copied().map(u64::from).sum::<u64>(),
                "seed={seed:#x} field={field_ord} total tokens"
            );
            assert!(
                field.doc_ords().windows(2).all(|pair| pair[0] <= pair[1]),
                "seed={seed:#x} field={field_ord} row order"
            );

            for ((doc_ord, expected_length), doc_index) in
                expected_doc_ords.iter().zip(expected).zip(0_usize..)
            {
                let rows: Vec<_> = field
                    .doc_ords()
                    .iter()
                    .enumerate()
                    .filter_map(|(row, row_doc)| (row_doc == doc_ord).then_some(row))
                    .collect();
                assert_eq!(
                    rows.len(),
                    usize::try_from(*expected_length).expect("generated length fits usize"),
                    "seed={seed:#x} field={field_ord} doc_index={doc_index}"
                );
                if let Some(positions) = field.positions() {
                    for (expected_position, row) in rows.into_iter().enumerate() {
                        assert_eq!(
                            positions[row],
                            u32::try_from(expected_position).expect("small generated position"),
                            "seed={seed:#x} field={field_ord} doc_index={doc_index}"
                        );
                    }
                }
            }

            for term_id in field.term_ids() {
                assert_eq!(
                    accumulator.terms().field_and_term(*term_id).0,
                    field_ord,
                    "seed={seed:#x} field={field_ord} term namespace"
                );
            }
        }
        assert_eq!(
            accumulator.token_count(),
            expected_lengths
                .iter()
                .flatten()
                .map(|length| usize::try_from(*length).expect("generated length fits usize"))
                .sum::<usize>(),
            "seed={seed:#x} total token count"
        );
        tracing::info!(
            seed,
            documents = accumulator.document_count(),
            tokens = accumulator.token_count(),
            bytes_reserved = accumulator.bytes_reserved(),
            "Scribe randomized accumulation invariant case passed"
        );
    }
}
