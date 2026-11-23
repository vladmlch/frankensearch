use super::*;

use crate::grimoire::{TermDictionary, TermSectionLengths};
use crate::quiver::{
    EncodedStoredMetaSection, FieldStats, IdHashCodecError, IdHashSection, IdMapSection,
    PositionList, Posting, PostingList, StatsSection,
};
use crate::segment::{SectionKind, SegmentReader};

#[test]
fn delta_flush_reuses_a_primed_snapshot_term_order_without_byte_drift() {
    let input = DeltaFlushInput {
        segment_id: 0xfeed_beef,
        created_unix_s: 1_700_000_000,
        engine_version: 0x0001_0002,
    };
    let canonical = flush_delta_snapshot(&delta_streaming_snapshot(), input)
        .expect("unprimed Delta flush")
        .expect("nonempty Delta snapshot seals");

    let snapshot = delta_streaming_snapshot();
    reset_sorted_ids_calls();
    let primed_terms = snapshot
        .ordered_terms()
        .map(|term| term.term().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        primed_terms,
        [b"alpha".to_vec(), b"beta".to_vec(), b"zeta".to_vec()]
    );
    assert_eq!(
        sorted_ids_calls(),
        1,
        "priming must build the immutable snapshot's one canonical ID order"
    );

    let streamed = flush_delta_snapshot(&snapshot, input)
        .expect("primed Delta flush")
        .expect("nonempty primed Delta snapshot seals");
    assert_eq!(streamed.as_bytes(), canonical.as_bytes());
    assert_eq!(
        sorted_ids_calls(),
        1,
        "a primed Delta flush must reuse the cached IDs instead of sorting a second full view"
    );
}

#[test]
fn radix_flush_is_byte_identical_and_reopens_every_section() {
    crate::quiver::reset_stored_meta_materializations();
    let mut accumulator =
        ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid mixed schema");
    accumulator
        .add_document_with_stored(
            2,
            &[
                IndexedFieldValue::new(0, "doc-a"),
                IndexedFieldValue::new(1, "alpha beta alpha"),
                IndexedFieldValue::new(2, "alpha alpha"),
            ],
            &[StoredFieldValue::new(3, b"stored-a")],
        )
        .expect("first sparse document accumulates");
    accumulator
        .add_document_with_stored(
            5,
            &[
                IndexedFieldValue::new(0, "doc-b"),
                IndexedFieldValue::new(1, "beta alpha"),
                IndexedFieldValue::new(2, "gamma"),
            ],
            &[StoredFieldValue::new(3, b"stored-b")],
        )
        .expect("second sparse document accumulates");
    let identities = [
        FlushDocumentInput::from_canonical_content(2, "doc-a", b"canonical-a"),
        FlushDocumentInput::from_canonical_content(5, "doc-b", b"canonical-b"),
    ];
    let input = FlushSegmentInput {
        segment_id: 0xfeed_beef,
        lease_docid_base: 65_536,
        created_unix_s: 1_700_000_000,
        engine_version: 0x0001_0002,
        documents: &identities,
    };

    let first = flush_accumulator(&accumulator, input).expect("first flush");
    let second = flush_accumulator(&accumulator, input).expect("second flush");
    assert_eq!(
        crate::quiver::stored_meta_materializations(),
        0,
        "shipping accumulator flush must emit borrowed STOREDMETA directly"
    );
    let owned_control = EncodedStoredMetaSection::encode_accumulator(
        65_538,
        65_542,
        input.lease_docid_base,
        &accumulator,
    )
    .expect("the former owned STOREDMETA control still encodes");
    assert_eq!(crate::quiver::stored_meta_materializations(), 1);
    assert_eq!(first.as_bytes(), second.as_bytes());
    assert_eq!(accumulator.document_ords(), &[2, 5]);

    let reader = SegmentReader::from_bytes(first.as_bytes(), MIXED_POSITION_SCHEMA)
        .expect("segment reopens");
    reader.verify().expect("all section witnesses verify");
    assert_eq!(reader.header().docid_lo, 65_538);
    assert_eq!(reader.header().docid_hi, 65_542);
    assert_eq!(reader.header().doc_count, 2);
    let postings_bytes = reader
        .section(SectionKind::POSTINGS)
        .expect("POSTINGS checksum")
        .expect("POSTINGS required");
    let positions_bytes = reader
        .section(SectionKind::POSITIONS)
        .expect("POSITIONS checksum")
        .expect("POSITIONS required");
    let blockmax_bytes = reader
        .section(SectionKind::BLOCKMAX)
        .expect("BLOCKMAX checksum")
        .expect("BLOCKMAX required");
    let sections = TermSectionLengths {
        postings: u64::try_from(postings_bytes.len()).expect("POSTINGS length"),
        positions: Some(u64::try_from(positions_bytes.len()).expect("POSITIONS length")),
        blockmax: u64::try_from(blockmax_bytes.len()).expect("BLOCKMAX length"),
    };
    let dictionary = TermDictionary::parse(
        reader
            .section(SectionKind::TERMDICT)
            .expect("TERMDICT checksum")
            .expect("TERMDICT required"),
        MIXED_POSITION_SCHEMA,
        sections,
    )
    .expect("TERMDICT reopens");
    assert_eq!(dictionary.term_count() as usize, accumulator.terms().len());

    for term_id in 0..u32::try_from(accumulator.terms().len()).expect("term count fits u32") {
        let (field_ord, term) = accumulator.terms().field_and_term(term_id);
        let field = accumulator
            .fields()
            .iter()
            .find(|field| field.field_ord() == field_ord)
            .expect("term field exists");
        let mut expected_runs = Vec::<(u32, u32, Vec<u32>)>::new();
        for index in 0..field.term_ids().len() {
            if field.term_ids()[index] != term_id {
                continue;
            }
            let doc_id = u32::try_from(input.lease_docid_base + u64::from(field.doc_ords()[index]))
                .expect("fixture document id fits u32");
            if let Some((previous_doc, frequency, run_positions)) = expected_runs.last_mut()
                && *previous_doc == doc_id
            {
                *frequency += 1;
                if let Some(positions) = field.positions() {
                    run_positions.push(positions[index]);
                }
            } else {
                expected_runs.push((
                    doc_id,
                    1,
                    field
                        .positions()
                        .map_or_else(Vec::new, |positions| vec![positions[index]]),
                ));
            }
        }
        let entry = dictionary
            .lookup(field_ord, term)
            .expect("term lookup is valid")
            .expect("every interned term is emitted");
        assert_eq!(
            entry.metadata.doc_freq,
            u32::try_from(expected_runs.len()).expect("fixture doc frequency fits u32"),
            "doc_freq drift for field {field_ord} term {:?}",
            String::from_utf8_lossy(term)
        );
        let posting_start =
            usize::try_from(entry.metadata.postings.offset).expect("posting offset");
        let posting_end =
            posting_start + usize::try_from(entry.metadata.postings.len).expect("posting length");
        let posting_list = PostingList::parse(
            &postings_bytes[posting_start..posting_end],
            entry.metadata.doc_freq,
        )
        .expect("posting list reopens");
        let decoded = posting_list.decode_all().expect("posting list decodes");
        let expected_postings = expected_runs
            .iter()
            .map(|(doc_id, frequency, _)| Posting::new(*doc_id, *frequency))
            .collect::<Vec<_>>();
        assert_eq!(
            decoded,
            expected_postings,
            "posting drift for field {field_ord} term {:?}",
            String::from_utf8_lossy(term)
        );
        if field.positions().is_some() {
            let position_span = entry.metadata.positions.expect("position span required");
            let position_start = usize::try_from(position_span.offset).expect("position offset");
            let position_end =
                position_start + usize::try_from(position_span.len).expect("position length");
            let position_list = PositionList::parse(
                &positions_bytes[position_start..position_end],
                &posting_list,
            )
            .expect("position list reopens");
            for (ordinal, (_, _, expected_positions)) in expected_runs.iter().enumerate() {
                let actual_positions = position_list
                    .positions_for_ordinal(
                        u32::try_from(ordinal).expect("fixture posting ordinal fits u32"),
                    )
                    .expect("position run exists")
                    .collect::<Result<Vec<_>, _>>()
                    .expect("position run decodes");
                assert_eq!(actual_positions, *expected_positions);
            }
        } else {
            assert!(entry.metadata.positions.is_none());
        }
    }

    let alpha = dictionary
        .lookup(1, b"alpha")
        .expect("alpha lookup")
        .expect("alpha exists");
    assert_eq!(alpha.metadata.doc_freq, 2);
    let posting_start = usize::try_from(alpha.metadata.postings.offset).expect("posting offset");
    let posting_end =
        posting_start + usize::try_from(alpha.metadata.postings.len).expect("posting length");
    let postings = PostingList::parse(
        &postings_bytes[posting_start..posting_end],
        alpha.metadata.doc_freq,
    )
    .expect("alpha postings reopen");
    assert_eq!(
        postings.decode_all().expect("alpha postings decode"),
        [Posting::new(65_538, 2), Posting::new(65_541, 1)]
    );
    let position_span = alpha.metadata.positions.expect("alpha stores positions");
    let position_start = usize::try_from(position_span.offset).expect("position offset");
    let position_end =
        position_start + usize::try_from(position_span.len).expect("position length");
    let positions = PositionList::parse(&positions_bytes[position_start..position_end], &postings)
        .expect("alpha positions reopen");
    assert_eq!(
        positions
            .positions_for_ordinal(0)
            .expect("first alpha run")
            .collect::<Result<Vec<_>, _>>()
            .expect("first alpha positions"),
        [0, 2]
    );
    assert_eq!(
        positions
            .positions_for_ordinal(1)
            .expect("second alpha run")
            .collect::<Result<Vec<_>, _>>()
            .expect("second alpha positions"),
        [1]
    );

    let basic_alpha = dictionary
        .lookup(2, b"alpha")
        .expect("positionless alpha lookup")
        .expect("positionless alpha exists");
    let basic_start =
        usize::try_from(basic_alpha.metadata.postings.offset).expect("Basic posting offset");
    let basic_end = basic_start
        + usize::try_from(basic_alpha.metadata.postings.len).expect("Basic posting length");
    let basic_postings = PostingList::parse(
        &postings_bytes[basic_start..basic_end],
        basic_alpha.metadata.doc_freq,
    )
    .expect("Basic postings reopen");
    assert_eq!(
        basic_postings.decode_all().expect("Basic postings decode"),
        [Posting::new(65_538, 2)],
        "positionless text preserves term frequency without storing position lists"
    );
    let positionless = accumulator.field(2).expect("positionless field");
    assert_eq!(
        positionless.document_lengths(),
        &[2, 1],
        "Basic postings still retain complete token counts for fieldnorms"
    );

    let id_map = IdMapSection::parse(
        reader
            .section(SectionKind::IDMAP)
            .expect("IDMAP checksum")
            .expect("IDMAP required"),
        65_538,
        65_542,
    )
    .expect("IDMAP reopens");
    assert_eq!(id_map.get(65_538).expect("first id").document_id(), "doc-a");
    assert!(!id_map.contains(65_539));
    assert!(!id_map.contains(65_540));
    assert_eq!(
        id_map.get(65_541).expect("second id").document_id(),
        "doc-b"
    );
    let id_hash = IdHashSection::parse(
        reader
            .section(SectionKind::IDHASH)
            .expect("IDHASH checksum")
            .expect("IDHASH required"),
        id_map,
    )
    .expect("IDHASH reopens");
    assert_eq!(id_hash.lookup("doc-a"), Some(65_538));
    assert_eq!(id_hash.lookup("doc-b"), Some(65_541));

    let stats = StatsSection::parse(
        reader
            .section(SectionKind::STATS)
            .expect("STATS checksum")
            .expect("STATS required"),
        &[0, 1, 2],
        2,
    )
    .expect("STATS reopens");
    assert_eq!(stats.field(0), Some(FieldStats::new(0, 2, 2)));
    assert_eq!(stats.field(1), Some(FieldStats::new(1, 5, 2)));
    assert_eq!(stats.field(2), Some(FieldStats::new(2, 3, 2)));

    let stored = crate::quiver::StoredMetaSection::parse(
        reader
            .section(SectionKind::STOREDMETA)
            .expect("STOREDMETA checksum")
            .expect("STOREDMETA required"),
        65_538,
        65_542,
        &[3],
    )
    .expect("STOREDMETA reopens");
    let stored_field = stored.field(3).expect("stored field");
    assert_eq!(stored_field.get(65_538), Some(b"stored-a".as_slice()));
    assert_eq!(stored_field.get(65_539), None);
    assert_eq!(stored_field.get(65_541), Some(b"stored-b".as_slice()));
    assert_eq!(
        reader
            .section(SectionKind::STOREDMETA)
            .expect("STOREDMETA checksum")
            .expect("STOREDMETA required"),
        owned_control.as_bytes(),
        "borrowed shipping emission must remain byte-identical to the owned control"
    );
}

#[test]
fn e18_accumulator_invariants_hold_under_randomized_doc_streams() {
    // Randomized ascending-doc-ord streams must keep every parallel column
    // aligned, doc ords monotonic, and the reset-reuse contract intact
    // (logical state cleared, scratch capacity retained).
    let mut rng = DeterministicRng(0x1f9a_c3d7_5e21_0b46);
    for _round in 0..64 {
        let mut accumulator =
            ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid mixed schema");
        let doc_count = 1 + rng.choose(24);
        let mut previous: Option<u32> = None;
        let mut expected_docs = 0_u32;
        let mut expected_tokens = 0_usize;
        for _ in 0..doc_count {
            let doc_ord =
                previous.map_or(0, |prev| prev + 1 + u32::try_from(rng.choose(4)).unwrap());
            previous = Some(doc_ord);
            let keyword = randomized_text(&mut rng);
            let positional = randomized_text(&mut rng);
            let plain = randomized_text(&mut rng);
            let accumulation = accumulator
                .add_document_with_stored(
                    doc_ord,
                    &[
                        IndexedFieldValue::new(0, &keyword),
                        IndexedFieldValue::new(1, &positional),
                        IndexedFieldValue::new(2, &plain),
                    ],
                    &[StoredFieldValue::new(3, keyword.as_bytes())],
                )
                .expect("randomized document accumulates");
            expected_docs += 1;
            expected_tokens += usize::try_from(accumulation.admitted_tokens).unwrap();
        }

        assert_eq!(
            accumulator.document_count(),
            usize::try_from(expected_docs).unwrap()
        );
        assert_eq!(accumulator.token_count(), expected_tokens);
        let ords = accumulator.document_ords();
        assert!(
            ords.windows(2).all(|pair| pair[0] < pair[1]),
            "document ords must be strictly ascending: {ords:?}"
        );
        for field in accumulator.fields() {
            let rows = field.term_ids().len();
            assert_eq!(field.doc_ords().len(), rows, "doc_ords column width drift");
            assert!(
                field.fieldnorm_ids().len() <= accumulator.document_count(),
                "fieldnorm ids cannot exceed one per document"
            );
            if let Some(positions) = field.positions() {
                assert_eq!(positions.len(), rows, "positions column width drift");
            }
            assert!(
                field.doc_ords().windows(2).all(|pair| pair[0] <= pair[1]),
                "per-field doc ords must be non-decreasing"
            );
        }
        assert!(accumulator.bytes_reserved() >= accumulator.bytes_used());

        // Reset-reuse: logical state clears, scratch capacity is retained,
        // and the accumulator is immediately usable again.
        let reserved_before = accumulator.bytes_reserved();
        accumulator.reset();
        assert_eq!(accumulator.document_count(), 0);
        assert_eq!(accumulator.token_count(), 0);
        assert!(accumulator.document_ords().is_empty());
        assert!(
            accumulator.bytes_reserved() >= reserved_before / 2,
            "reset must retain most scratch capacity for reuse"
        );
        accumulator
            .add_document(0, &[IndexedFieldValue::new(1, "reuse after reset")])
            .expect("accumulator is reusable after reset");
        assert_eq!(accumulator.document_count(), 1);
    }
}

#[test]
fn e18_radix_flush_is_deterministic_across_100_seeds() {
    // Same accumulator input -> byte-identical segment, and the Automatic
    // (possibly Rayon) path matches the serial Scalar reference, over 100
    // randomized seeds spanning the parallel-radix row-count threshold.
    for seed in 0..100_u64 {
        let mut rng = DeterministicRng(0xd1b5_4f00_a37e_c119 ^ seed.wrapping_mul(0x9E37_79B9));
        let mut accumulator =
            ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid mixed schema");
        let doc_count = 4 + rng.choose(28);
        let mut identities = Vec::with_capacity(doc_count);
        let mut doc_ord = 0_u32;
        for index in 0..doc_count {
            doc_ord += u32::try_from(rng.choose(3)).unwrap();
            let keyword = format!("doc-{seed}-{index}");
            let positional = randomized_text(&mut rng);
            let plain = randomized_text(&mut rng);
            accumulator
                .add_document_with_stored(
                    doc_ord,
                    &[
                        IndexedFieldValue::new(0, &keyword),
                        IndexedFieldValue::new(1, &positional),
                        IndexedFieldValue::new(2, &plain),
                    ],
                    &[StoredFieldValue::new(3, positional.as_bytes())],
                )
                .expect("seeded document accumulates");
            identities.push((doc_ord, keyword));
            doc_ord += 1;
        }
        let flush_docs: Vec<FlushDocumentInput<'_>> = identities
            .iter()
            .map(|(ord, id)| FlushDocumentInput::from_canonical_content(*ord, id, id.as_bytes()))
            .collect();
        let input = FlushSegmentInput {
            segment_id: 0x5EED_0000 | seed,
            lease_docid_base: 65_536,
            created_unix_s: 1_700_000_000,
            engine_version: 0x0001_0002,
            documents: &flush_docs,
        };

        let scalar_a = flush_accumulator_with_mode(&accumulator, input, FlushMode::Scalar)
            .expect("scalar flush a");
        let scalar_b = flush_accumulator_with_mode(&accumulator, input, FlushMode::Scalar)
            .expect("scalar flush b");
        assert_eq!(
            scalar_a.as_bytes(),
            scalar_b.as_bytes(),
            "seed {seed}: repeated scalar flush must be byte-identical"
        );
        let automatic = flush_accumulator_with_mode(&accumulator, input, FlushMode::Automatic)
            .expect("automatic flush");
        assert_eq!(
            automatic.as_bytes(),
            scalar_a.as_bytes(),
            "seed {seed}: automatic radix flush must match the scalar reference"
        );
        SegmentReader::from_bytes(automatic.as_bytes(), MIXED_POSITION_SCHEMA)
            .expect("seeded segment reopens")
            .verify()
            .expect("seeded segment witnesses verify");
    }
}

#[test]
fn e18_e2e_ingest_roundtrip_reopens_idmap_doclens_and_postings() {
    // Full accumulate -> flush -> reopen round-trip over a randomized 40-doc
    // corpus: every document id round-trips through IDMAP, each indexed field
    // reopens its DOCLEN, and every interned term's POSTINGS decode with the
    // exact recorded doc_freq. (The 2-doc exhaustive exact-position readback
    // is `radix_flush_is_byte_identical_and_reopens_every_section`; this
    // covers scale + randomization.)
    let mut rng = DeterministicRng(0x00e2_e0e2_face_1234);
    let mut accumulator =
        ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid mixed schema");
    let doc_count = 40_usize;
    let lease_base = 65_536_u64;
    let mut doc_ord = 0_u32;
    let mut identities = Vec::with_capacity(doc_count);
    for index in 0..doc_count {
        doc_ord += u32::try_from(rng.choose(3)).unwrap();
        let id = format!("e2e-doc-{index}");
        accumulator
            .add_document_with_stored(
                doc_ord,
                &[
                    IndexedFieldValue::new(0, &id),
                    IndexedFieldValue::new(1, &randomized_text(&mut rng)),
                    IndexedFieldValue::new(2, &randomized_text(&mut rng)),
                ],
                &[StoredFieldValue::new(3, id.as_bytes())],
            )
            .expect("e2e document accumulates");
        identities.push((doc_ord, id));
        doc_ord += 1;
    }
    let flush_docs: Vec<FlushDocumentInput<'_>> = identities
        .iter()
        .map(|(ord, id)| FlushDocumentInput::from_canonical_content(*ord, id, id.as_bytes()))
        .collect();
    let input = FlushSegmentInput {
        segment_id: 0x00E2_E000,
        lease_docid_base: lease_base,
        created_unix_s: 1_700_000_000,
        engine_version: 0x0001_0002,
        documents: &flush_docs,
    };
    let segment = flush_accumulator(&accumulator, input).expect("e2e flush");
    let reader = SegmentReader::from_bytes(segment.as_bytes(), MIXED_POSITION_SCHEMA)
        .expect("segment reopens");
    reader.verify().expect("all section witnesses verify");
    assert_eq!(
        reader.header().doc_count,
        u32::try_from(doc_count).expect("doc count fits u32")
    );

    // IDMAP: every ingested document id round-trips at its global docid.
    let id_map = IdMapSection::parse(
        reader
            .section(SectionKind::IDMAP)
            .expect("IDMAP checksum")
            .expect("IDMAP required"),
        reader.header().docid_lo,
        reader.header().docid_hi,
    )
    .expect("IDMAP reopens");
    for (ord, id) in &identities {
        let global = lease_base + u64::from(*ord);
        assert_eq!(
            id_map.get(global).expect("id present").document_id(),
            id.as_str(),
            "IDMAP round-trip drift at global docid {global}"
        );
    }

    // DOCLEN: reopens (verify() already checked its witness) with the
    // always-populated keyword field present, and returns a fieldnorm for
    // every ingested keyword document. The fieldnorm array is indexed over
    // the dense global-docid range, so its length tracks the range span, not
    // the sparse document count.
    let doclen = crate::quiver::DocLenSection::parse(
        reader
            .section(SectionKind::DOCLEN)
            .expect("DOCLEN checksum")
            .expect("DOCLEN required"),
        reader.header().docid_lo,
        reader.header().docid_hi,
        &[0, 1, 2],
    )
    .expect("DOCLEN reopens");
    let keyword_doclen = doclen.field(0).expect("keyword field DOCLEN reopens");
    for (ord, _) in &identities {
        let global = lease_base + u64::from(*ord);
        assert!(
            keyword_doclen.fieldnorm_id(global).is_some(),
            "keyword fieldnorm missing for global docid {global}"
        );
    }

    // POSTINGS: every interned term decodes with the recorded doc_freq.
    let postings_bytes = reader
        .section(SectionKind::POSTINGS)
        .expect("POSTINGS checksum")
        .expect("POSTINGS required");
    let positions_bytes = reader
        .section(SectionKind::POSITIONS)
        .expect("POSITIONS checksum")
        .expect("POSITIONS required");
    let blockmax_bytes = reader
        .section(SectionKind::BLOCKMAX)
        .expect("BLOCKMAX checksum")
        .expect("BLOCKMAX required");
    let dictionary = TermDictionary::parse(
        reader
            .section(SectionKind::TERMDICT)
            .expect("TERMDICT checksum")
            .expect("TERMDICT required"),
        MIXED_POSITION_SCHEMA,
        TermSectionLengths {
            postings: u64::try_from(postings_bytes.len()).expect("POSTINGS length"),
            positions: Some(u64::try_from(positions_bytes.len()).expect("POSITIONS length")),
            blockmax: u64::try_from(blockmax_bytes.len()).expect("BLOCKMAX length"),
        },
    )
    .expect("TERMDICT reopens");
    assert_eq!(dictionary.term_count() as usize, accumulator.terms().len());
    for term_id in 0..u32::try_from(accumulator.terms().len()).expect("term count fits u32") {
        let (field_ord, term) = accumulator.terms().field_and_term(term_id);
        let metadata = dictionary
            .lookup(field_ord, term)
            .expect("term lookup is valid")
            .expect("every interned term is emitted")
            .metadata;
        let start = usize::try_from(metadata.postings.offset).expect("posting offset");
        let end = start + usize::try_from(metadata.postings.len).expect("posting length");
        let posting_list = PostingList::parse(&postings_bytes[start..end], metadata.doc_freq)
            .expect("posting list reopens");
        assert_eq!(
            posting_list.decode_all().expect("postings decode").len(),
            usize::try_from(metadata.doc_freq).expect("doc freq fits usize"),
            "doc_freq drift for field {field_ord} term {:?}",
            String::from_utf8_lossy(term)
        );
    }
}

#[test]
fn e18_cancelled_commit_seals_no_segment_and_leaves_no_temp_files() {
    // Leak oracle: a batch is ingested (accumulated in memory), then the
    // seal/commit is cancelled mid-flight. commit() must return Cancelled and
    // publish nothing — no `seg-*.fslx` and no staged temp file in the index
    // directory; the accumulator arenas drop when the index drops. (The
    // manifest-slot variant is
    // keeper::tests::invalid_or_cancelled_proposal_creates_no_manifest_slot.)
    use crate::index::{QuillIndex, QuillIndexError};
    use frankensearch_core::IndexableDocument;

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temp index directory");
        let index =
            QuillIndex::create(&cx, directory.path(), crate::config::QuillConfig::default())
                .await
                .expect("create shipping-schema index");
        let documents = [
            IndexableDocument::new("cancel-a", "alpha beta gamma alpha"),
            IndexableDocument::new("cancel-b", "delta epsilon delta"),
        ];
        index
            .index_documents(&cx, &documents)
            .await
            .expect("uncancelled ingest accumulates the batch");

        cx.set_cancel_requested(true);
        let cancelled = matches!(
            index.commit(&cx).await,
            Err(QuillIndexError::Cancelled { .. })
        );
        assert!(cancelled, "cancelled commit must return Cancelled");

        let mut artifacts = Vec::new();
        for entry in std::fs::read_dir(directory.path()).expect("read index directory") {
            let entry = entry.expect("directory entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("seg-") || name.contains(".tmp") {
                artifacts.push(name);
            }
        }
        assert!(
            artifacts.is_empty(),
            "cancelled commit sealed or staged segment artifacts: {artifacts:?}"
        );
    });
}

#[test]
fn scalar_and_automatic_flush_are_byte_identical_at_parallel_row_boundary() {
    assert_eq!(FlushMode::default(), FlushMode::Automatic);
    let automatic_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("four-thread automatic flush pool");

    for row_count in [
        PARALLEL_RADIX_ROWS_PER_CHUNK - 1,
        PARALLEL_RADIX_ROWS_PER_CHUNK,
        PARALLEL_RADIX_ROWS_PER_CHUNK + 1,
    ] {
        let text = "boundary ".repeat(row_count);
        let mut accumulator =
            ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
        accumulator
            .add_document(0, &[IndexedFieldValue::new(1, &text)])
            .expect("boundary document accumulates");
        assert_eq!(accumulator.token_count(), row_count);

        let identities = [FlushDocumentInput::from_canonical_content(
            0,
            "boundary-doc",
            text.as_bytes(),
        )];
        let input = FlushSegmentInput {
            segment_id: u64::try_from(row_count).expect("row count fits u64"),
            lease_docid_base: 0,
            created_unix_s: 0,
            engine_version: 0,
            documents: &identities,
        };
        let automatic = automatic_pool
            .install(|| flush_accumulator(&accumulator, input))
            .expect("automatic flush");
        let scalar = flush_accumulator_with_mode(&accumulator, input, FlushMode::Scalar)
            .expect("scalar flush");

        assert_eq!(
            scalar.as_bytes(),
            automatic.as_bytes(),
            "canonical bytes drift at {row_count} token rows"
        );
    }
}

#[test]
fn radix_partition_preserves_equal_term_source_order_in_both_widths() {
    let one_pass = stable_radix_partition(
        vec![
            FlushTokenRow {
                term_id: 2,
                doc_ord: 1,
                position: 7,
            },
            FlushTokenRow {
                term_id: 0,
                doc_ord: 2,
                position: 8,
            },
            FlushTokenRow {
                term_id: 2,
                doc_ord: 3,
                position: 9,
            },
        ],
        3,
    )
    .expect("one-pass radix partition");
    assert_eq!(
        one_pass.rows,
        [
            FlushTokenRow {
                term_id: 0,
                doc_ord: 2,
                position: 8,
            },
            FlushTokenRow {
                term_id: 2,
                doc_ord: 1,
                position: 7,
            },
            FlushTokenRow {
                term_id: 2,
                doc_ord: 3,
                position: 9,
            },
        ]
    );

    let two_pass = stable_radix_partition(
        vec![
            FlushTokenRow {
                term_id: 65_536,
                doc_ord: 4,
                position: 1,
            },
            FlushTokenRow {
                term_id: 1,
                doc_ord: 5,
                position: 2,
            },
            FlushTokenRow {
                term_id: 65_536,
                doc_ord: 6,
                position: 3,
            },
            FlushTokenRow {
                term_id: 65_535,
                doc_ord: 7,
                position: 4,
            },
        ],
        65_537,
    )
    .expect("two-pass radix partition");
    assert_eq!(
        two_pass
            .rows
            .iter()
            .map(|row| (row.term_id, row.doc_ord))
            .collect::<Vec<_>>(),
        [(1, 5), (65_535, 7), (65_536, 4), (65_536, 6)]
    );

    let source = (0_u32..10_000)
        .map(|ordinal| FlushTokenRow {
            term_id: ordinal % 17,
            doc_ord: ordinal,
            position: ordinal,
        })
        .collect::<Vec<_>>();
    let serial = stable_radix_partition_with_chunks(source.clone(), 17, 1)
        .expect("serial reference partition");
    let parallel =
        stable_radix_partition_with_chunks(source, 17, 4).expect("four-chunk parallel partition");
    assert_eq!(parallel.rows, serial.rows);
    assert_eq!(parallel.ranges, serial.ranges);

    let mut seed = 0x7f4a_7c15_d2b7_4407_u64;
    let mut high_cardinality_source = Vec::new();
    for ordinal in 0_u32..12_000 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let term_id = if ordinal == 0 {
            65_536
        } else {
            u32::try_from(seed % 65_537).expect("bounded seeded term id")
        };
        high_cardinality_source.push(FlushTokenRow {
            term_id,
            doc_ord: ordinal,
            position: ordinal,
        });
    }
    let high_cardinality_serial =
        stable_radix_partition_with_chunks(high_cardinality_source.clone(), 65_537, 1)
            .expect("two-pass serial reference partition");
    let high_cardinality_parallel =
        stable_radix_partition_with_chunks(high_cardinality_source, 65_537, 4)
            .expect("two-pass four-chunk parallel partition");
    assert_eq!(high_cardinality_parallel.rows, high_cardinality_serial.rows);
    assert_eq!(
        high_cardinality_parallel.ranges,
        high_cardinality_serial.ranges
    );
}

#[test]
fn flush_rejects_sidecar_drift_duplicates_and_docid_overflow() {
    let empty = ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
    assert!(matches!(
        flush_accumulator(
            &empty,
            FlushSegmentInput {
                segment_id: 1,
                lease_docid_base: 0,
                created_unix_s: 0,
                engine_version: 0,
                documents: &[],
            }
        ),
        Err(FlushError::EmptyAccumulator)
    ));

    let mut accumulator = ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
    accumulator
        .add_document(7, &[IndexedFieldValue::new(1, "same")])
        .expect("first document");
    accumulator
        .add_document(9, &[IndexedFieldValue::new(1, "same")])
        .expect("second document");
    let one_identity = [FlushDocumentInput::new(7, "only-one", 1)];
    assert!(matches!(
        flush_accumulator(
            &accumulator,
            FlushSegmentInput {
                segment_id: 2,
                lease_docid_base: 0,
                created_unix_s: 0,
                engine_version: 0,
                documents: &one_identity,
            }
        ),
        Err(FlushError::DocumentCountMismatch {
            expected: 2,
            actual: 1
        })
    ));
    let drifted = [
        FlushDocumentInput::new(7, "first", 1),
        FlushDocumentInput::new(8, "second", 2),
    ];
    assert!(matches!(
        flush_accumulator(
            &accumulator,
            FlushSegmentInput {
                segment_id: 3,
                lease_docid_base: 0,
                created_unix_s: 0,
                engine_version: 0,
                documents: &drifted,
            }
        ),
        Err(FlushError::DocumentOrdinalMismatch {
            index: 1,
            expected: 9,
            actual: 8
        })
    ));
    let duplicates = [
        FlushDocumentInput::new(7, "duplicate", 1),
        FlushDocumentInput::new(9, "duplicate", 2),
    ];
    assert!(matches!(
        flush_accumulator(
            &accumulator,
            FlushSegmentInput {
                segment_id: 4,
                lease_docid_base: 0,
                created_unix_s: 0,
                engine_version: 0,
                documents: &duplicates,
            }
        ),
        Err(FlushError::IdHash(
            IdHashCodecError::DuplicateDocumentId { .. }
        ))
    ));
    let valid = [
        FlushDocumentInput::new(7, "first", 1),
        FlushDocumentInput::new(9, "second", 2),
    ];
    assert!(matches!(
        flush_accumulator(
            &accumulator,
            FlushSegmentInput {
                segment_id: 5,
                lease_docid_base: 1,
                created_unix_s: 0,
                engine_version: 0,
                documents: &valid,
            }
        ),
        Err(FlushError::MisalignedLeaseBase {
            lease_docid_base: 1
        })
    ));
    assert!(matches!(
        flush_accumulator(
            &accumulator,
            FlushSegmentInput {
                segment_id: 6,
                lease_docid_base: 1_u64 << 32,
                created_unix_s: 0,
                engine_version: 0,
                documents: &valid,
            }
        ),
        Err(FlushError::DocumentIdOverflow { doc_ord: 7, .. })
    ));

    let mut boundary =
        ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid boundary schema");
    boundary
        .add_document(
            DOC_ORDS_PER_LEASE - 1,
            &[IndexedFieldValue::new(1, "last global document")],
        )
        .expect("last lease ordinal accumulates");
    let boundary_identity = [FlushDocumentInput::new(
        DOC_ORDS_PER_LEASE - 1,
        "last-global-document",
        4,
    )];
    let boundary_segment = flush_accumulator(
        &boundary,
        FlushSegmentInput {
            segment_id: 7,
            lease_docid_base: (1_u64 << 32) - u64::from(DOC_ORDS_PER_LEASE),
            created_unix_s: 0,
            engine_version: 0,
            documents: &boundary_identity,
        },
    )
    .expect("u32::MAX remains a valid posting docid");
    assert_eq!(boundary_segment.header().docid_lo, u64::from(u32::MAX));
    assert_eq!(boundary_segment.header().docid_hi, 1_u64 << 32);

    let mut numeric =
        ColumnarAccumulator::new(INDEXED_NUMERIC_SCHEMA).expect("valid numeric schema");
    numeric
        .add_document_with_stored(0, &[], &[StoredFieldValue::new(0, &42_u64.to_le_bytes())])
        .expect("numeric stored bytes accumulate");
    let numeric_identity = [FlushDocumentInput::new(0, "numeric", 3)];
    let numeric_segment = flush_accumulator(
        &numeric,
        FlushSegmentInput {
            segment_id: 8,
            lease_docid_base: 0,
            created_unix_s: 0,
            engine_version: 0,
            documents: &numeric_identity,
        },
    )
    .expect("indexed numeric field seals");
    let numeric_reader = crate::segment::SegmentReader::from_owned(
        numeric_segment.into_bytes(),
        INDEXED_NUMERIC_SCHEMA,
    )
    .expect("numeric segment reopens");
    let numeric_bytes = numeric_reader
        .section(SectionKind::NUMERIC)
        .expect("NUMERIC checksum verifies")
        .expect("indexed numeric schema requires NUMERIC");
    let numeric_section =
        crate::quiver::NumericSection::parse(numeric_bytes, INDEXED_NUMERIC_SCHEMA, 0, 1)
            .expect("NUMERIC payload validates");
    assert_eq!(
        numeric_section
            .field(0)
            .expect("sequence field")
            .entries()
            .collect::<Vec<_>>(),
        vec![crate::quiver::NumericEntry::u64(42, 0)]
    );
}

#[test]
fn fast_only_unstored_numeric_seals_sparse_column() {
    let mut accumulator =
        ColumnarAccumulator::new(FAST_ONLY_NUMERIC_SCHEMA).expect("valid fast-only numeric schema");
    accumulator
        .add_document_with_numeric(0, &[], &[IndexedNumericValue::u64(0, u64::MAX)])
        .expect("fast-only value accumulates");
    accumulator
        .add_document_with_numeric(17, &[], &[])
        .expect("missing fast-only value accumulates");
    assert!(accumulator.stored_fields().is_empty());

    let documents = [
        FlushDocumentInput::new(0, "present", 1),
        FlushDocumentInput::new(17, "missing", 2),
    ];
    let encoded = flush_accumulator(
        &accumulator,
        FlushSegmentInput {
            segment_id: 9,
            lease_docid_base: 0,
            created_unix_s: 0,
            engine_version: 0,
            documents: &documents,
        },
    )
    .expect("fast-only unstored field seals");
    let reader = SegmentReader::from_owned(encoded.into_bytes(), FAST_ONLY_NUMERIC_SCHEMA)
        .expect("fast-only segment reopens");
    assert!(
        reader
            .section(SectionKind::STOREDMETA)
            .expect("section lookup")
            .is_none()
    );
    let numeric = reader
        .section(SectionKind::NUMERIC)
        .expect("NUMERIC checksum")
        .expect("fast-only schema requires NUMERIC");
    let section = crate::quiver::NumericSection::parse(numeric, FAST_ONLY_NUMERIC_SCHEMA, 0, 18)
        .expect("fast-only NUMERIC validates");
    assert_eq!(
        section
            .field(0)
            .expect("fast-only column")
            .entries()
            .collect::<Vec<_>>(),
        vec![crate::quiver::NumericEntry::u64(u64::MAX, 0)]
    );
}
