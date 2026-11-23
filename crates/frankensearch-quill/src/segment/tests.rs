use std::error::Error;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;

use frankensearch_core::SearchError;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::QuillError;
use crate::schema::{Analyzer, DEFAULT_SCHEMA, FieldDescriptor, FieldKind, SchemaDescriptor};

use super::codec::{
    align_up, crc32, published_segment_id, read_u32_at, read_u64_at, schema_has_numeric_columns,
    schema_has_positions, schema_has_stored_fields,
};
use super::{
    EncodedSegment, FILE_PREFIX_LEN, FIXED_HEADER_LEN, FSLX_SECTION_ALIGNMENT, HEADER_CRC_LEN,
    MAX_DOCID_EXCLUSIVE, OPTIONAL_SKIPPABLE, PlannedSection, SECTION_ENTRY_LEN, SectionEntry,
    SectionInput, SectionKind, SegmentAssembler, SegmentHeader, SegmentHeaderInput, SegmentLimits,
    SegmentReader, TRAILER_LEN,
};

type TestResult = Result<(), Box<dyn Error>>;

fn init_replay_tracing() {
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
}

const MINIMAL_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "term",
    kind: FieldKind::Keyword,
    stored: false,
}];
const MINIMAL_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "segment-test-minimal",
    fields: &MINIMAL_FIELDS,
};
const CONDITIONAL_FIELDS: [FieldDescriptor; 2] = [
    FieldDescriptor {
        id: 0,
        name: "body",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "rank",
        kind: FieldKind::I64 {
            indexed: true,
            fast: false,
        },
        stored: false,
    },
];
const CONDITIONAL_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "segment-test-all-conditionals",
    fields: &CONDITIONAL_FIELDS,
};
const U64_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "unsigned",
    kind: FieldKind::U64 {
        indexed: true,
        fast: false,
    },
    stored: false,
}];
const U64_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "segment-test-indexed-u64",
    fields: &U64_FIELDS,
};
const PINNED_DEFAULT_ENTRIES: [SectionEntry; 10] = [
    SectionEntry {
        kind: SectionKind::TERMDICT,
        flags: 0,
        offset: 384,
        len: 4,
        xxh3: 0xb311_f463_fe75_7daa,
    },
    SectionEntry {
        kind: SectionKind::POSTINGS,
        flags: 0,
        offset: 448,
        len: 5,
        xxh3: 0x226e_c10f_2d51_8a15,
    },
    SectionEntry {
        kind: SectionKind::POSITIONS,
        flags: 0,
        offset: 512,
        len: 6,
        xxh3: 0x8464_7628_6cec_0ef4,
    },
    SectionEntry {
        kind: SectionKind::BLOCKMAX,
        flags: 0,
        offset: 576,
        len: 7,
        xxh3: 0x5ae0_ae5f_0af5_abf3,
    },
    SectionEntry {
        kind: SectionKind::DOCLEN,
        flags: 0,
        offset: 640,
        len: 8,
        xxh3: 0x4bd4_2ac3_c7ea_e3b1,
    },
    SectionEntry {
        kind: SectionKind::IDMAP,
        flags: 0,
        offset: 704,
        len: 9,
        xxh3: 0x38cc_9c70_1a48_1084,
    },
    SectionEntry {
        kind: SectionKind::IDHASH,
        flags: 0,
        offset: 768,
        len: 10,
        xxh3: 0x2518_0e76_4f5a_2769,
    },
    SectionEntry {
        kind: SectionKind::NUMERIC,
        flags: 0,
        offset: 832,
        len: 11,
        xxh3: 0xf32b_8ca9_4e2a_52bc,
    },
    SectionEntry {
        kind: SectionKind::STOREDMETA,
        flags: 0,
        offset: 896,
        len: 12,
        xxh3: 0xffff_1c57_4139_b461,
    },
    SectionEntry {
        kind: SectionKind::STATS,
        flags: 0,
        offset: 960,
        len: 13,
        xxh3: 0xd325_c744_07be_efd7,
    },
];
const PINNED_DEFAULT_PAYLOADS: [&[u8]; 10] = [
    &[0x01, 0x12, 0x23, 0x34],
    &[0x02, 0x13, 0x24, 0x35, 0x46],
    &[0x03, 0x14, 0x25, 0x36, 0x47, 0x58],
    &[0x04, 0x15, 0x26, 0x37, 0x48, 0x59, 0x6a],
    &[0x05, 0x16, 0x27, 0x38, 0x49, 0x5a, 0x6b, 0x7c],
    &[0x06, 0x17, 0x28, 0x39, 0x4a, 0x5b, 0x6c, 0x7d, 0x8e],
    &[0x07, 0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f, 0xa0],
    &[
        0x08, 0x19, 0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x90, 0xa1, 0xb2,
    ],
    &[
        0x09, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x80, 0x91, 0xa2, 0xb3, 0xc4,
    ],
    &[
        0x0a, 0x1b, 0x2c, 0x3d, 0x4e, 0x5f, 0x70, 0x81, 0x92, 0xa3, 0xb4, 0xc5, 0xd6,
    ],
];

#[derive(Clone, Debug)]
struct OwnedSection {
    kind: SectionKind,
    flags: u16,
    bytes: Vec<u8>,
}

/// The shipping schema as it was before its text fields moved to CJK
/// bigrams (fba273fb). The committed v1 golden was encoded with it, so it
/// keeps pinning the wire format byte for byte.
const LEGACY_DEFAULT_FIELDS: [FieldDescriptor; 5] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: crate::schema::FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "content",
        kind: crate::schema::FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "title",
        kind: crate::schema::FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "metadata_json",
        kind: crate::schema::FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "ord",
        kind: crate::schema::FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
];
const LEGACY_DEFAULT_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "frankensearch-default-v1",
    fields: &LEGACY_DEFAULT_FIELDS,
};

fn fixture_header(schema: SchemaDescriptor) -> SegmentHeaderInput {
    SegmentHeaderInput {
        segment_id: 0x0123_4567_89ab_cdef,
        schema,
        docid_lo: 100,
        docid_hi: 107,
        doc_count: 5,
        created_unix_s: 1_726_000_123,
        engine_version: 0x0002_0001,
    }
}

fn fixture_sections(schema: SchemaDescriptor, large: bool) -> Vec<OwnedSection> {
    let mut result = Vec::new();
    for raw in SectionKind::TERMDICT.raw()..=SectionKind::STATS.raw() {
        let kind = SectionKind::from_raw(raw);
        let present = match kind {
            SectionKind::POSITIONS => schema_has_positions(schema),
            SectionKind::NUMERIC => schema_has_numeric_columns(schema),
            SectionKind::STOREDMETA => schema_has_stored_fields(schema),
            _ => true,
        };
        if present {
            let len = if large && kind == SectionKind::POSTINGS {
                5_137
            } else {
                usize::from(raw) + 3
            };
            let raw_byte = u8::try_from(raw).unwrap_or(0);
            let bytes = (0..len)
                .map(|index| {
                    let low = u8::try_from(index & 0xff).unwrap_or(0);
                    low.wrapping_mul(17).wrapping_add(raw_byte)
                })
                .collect();
            result.push(OwnedSection {
                kind,
                flags: 0,
                bytes,
            });
        }
    }
    result
}

fn encode_owned(
    header: SegmentHeaderInput,
    sections: &[OwnedSection],
) -> Result<EncodedSegment, QuillError> {
    let borrowed: Vec<_> = sections
        .iter()
        .map(|section| SectionInput {
            kind: section.kind,
            flags: section.flags,
            bytes: &section.bytes,
        })
        .collect();
    EncodedSegment::encode(header, &borrowed)
}

fn assemble_owned(
    header: SegmentHeaderInput,
    sections: &[OwnedSection],
) -> Result<EncodedSegment, QuillError> {
    let planned = sections
        .iter()
        .map(|section| PlannedSection::new(section.kind, section.bytes.len()))
        .collect::<Vec<_>>();
    let mut assembler = SegmentAssembler::new(header, &planned)?;
    for section in sections {
        assembler.copy_section(section.kind, &section.bytes)?;
    }
    assembler.finish()
}

fn encode_owned_with_unregistered_extensions(
    header: SegmentHeaderInput,
    sections: &[OwnedSection],
) -> Result<EncodedSegment, QuillError> {
    let borrowed: Vec<_> = sections
        .iter()
        .map(|section| SectionInput {
            kind: section.kind,
            flags: section.flags,
            bytes: &section.bytes,
        })
        .collect();
    EncodedSegment::encode_with_limits_impl(header, &borrowed, SegmentLimits::default(), true)
}

#[test]
fn planned_assembler_is_byte_identical_to_canonical_writer() -> TestResult {
    for &(schema, large) in &[
        (MINIMAL_SCHEMA, false),
        (MINIMAL_SCHEMA, true),
        (DEFAULT_SCHEMA, false),
        (DEFAULT_SCHEMA, true),
    ] {
        let sections = fixture_sections(schema, large);
        let encoded = encode_owned(fixture_header(schema), &sections)?;
        let assembled = assemble_owned(fixture_header(schema), &sections)?;
        assert_eq!(assembled, encoded);
        SegmentReader::from_bytes(assembled.as_bytes(), schema)?.verify()?;
    }
    Ok(())
}

#[test]
fn planned_assembler_rejects_out_of_order_and_inexact_writes() -> TestResult {
    let sections = fixture_sections(MINIMAL_SCHEMA, false);
    let planned = sections
        .iter()
        .map(|section| PlannedSection::new(section.kind, section.bytes.len()))
        .collect::<Vec<_>>();
    let first = &sections[0];
    assert!(!first.bytes.is_empty());

    let mut wrong_kind = SegmentAssembler::new(fixture_header(MINIMAL_SCHEMA), &planned)?;
    let error = wrong_kind
        .copy_section(sections[1].kind, &sections[1].bytes)
        .expect_err("out-of-order section must fail");
    assert!(error.to_string().contains("expected"));

    let mut short = SegmentAssembler::new(fixture_header(MINIMAL_SCHEMA), &planned)?;
    let error = short
        .copy_section(first.kind, &first.bytes[..first.bytes.len() - 1])
        .expect_err("short section must fail");
    assert!(error.to_string().contains("writer ended"));

    let mut long = SegmentAssembler::new(fixture_header(MINIMAL_SCHEMA), &planned)?;
    let error = long
        .write_section(first.kind, |bytes| {
            bytes.extend_from_slice(&first.bytes);
            bytes.push(0);
        })
        .expect_err("long section must fail");
    assert!(error.to_string().contains("writer ended"));

    let incomplete = SegmentAssembler::new(fixture_header(MINIMAL_SCHEMA), &planned)?;
    let error = incomplete
        .finish()
        .expect_err("missing planned sections must fail");
    assert!(error.to_string().contains("was not written"));
    Ok(())
}

/// Load the committed v1 fixture without invoking the production writer or
/// any checksum helper. The textual hex form keeps the durable bytes easy
/// to review while still freezing every byte across future format bumps.
fn pinned_v1_fixture() -> Result<Vec<u8>, Box<dyn Error>> {
    let hex: String = include_str!("../../tests/fixtures/fslx-v1-golden.hex")
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect();
    if !hex.len().is_multiple_of(2) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "pinned FSLX v1 fixture has an odd number of hex digits",
        )
        .into());
    }

    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for pair in hex.as_bytes().as_chunks::<2>().0 {
        bytes.push(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?);
    }
    Ok(bytes)
}

#[test]
fn reader_accepts_committed_pinned_v1_fixture() -> TestResult {
    let bytes = pinned_v1_fixture()?;
    let reader = SegmentReader::from_bytes(&bytes, LEGACY_DEFAULT_SCHEMA)?;

    assert_eq!(
        reader.header(),
        SegmentHeader {
            segment_id: 0x0123_4567_89ab_cdef,
            schema_id: 0xa312_ebf6_d136_07a5,
            docid_lo: 100,
            docid_hi: 107,
            doc_count: 5,
            created_unix_s: 1_726_000_123,
            engine_version: 0x0002_0001,
            section_count: 10,
        }
    );
    assert_eq!(reader.section_entries(), &PINNED_DEFAULT_ENTRIES);
    assert_eq!(reader.file_len(), 985);
    assert_eq!(reader.file_xxh3(), 0x3d1c_8570_73b5_390f);
    for (entry, payload) in PINNED_DEFAULT_ENTRIES.iter().zip(PINNED_DEFAULT_PAYLOADS) {
        assert_eq!(
            reader.section(entry.kind)?.expect("pinned fixture section"),
            payload
        );
    }
    reader.verify()?;
    Ok(())
}

#[test]
fn pinned_wire_oracle_and_roundtrip_preserve_header_table_and_payloads() -> TestResult {
    let owned = fixture_sections(LEGACY_DEFAULT_SCHEMA, false);
    let first = encode_owned(fixture_header(LEGACY_DEFAULT_SCHEMA), &owned)?;
    let second = encode_owned(fixture_header(LEGACY_DEFAULT_SCHEMA), &owned)?;
    let header_len = usize::try_from(read_u32_at(first.as_bytes(), 12).expect("header len"))?;
    let header_crc =
        read_u32_at(first.as_bytes(), FILE_PREFIX_LEN + header_len).expect("header crc");
    let trailer_start = first.as_bytes().len() - TRAILER_LEN;
    let trailer_crc = read_u32_at(first.as_bytes(), trailer_start + 8).expect("trailer crc");
    assert_eq!(
        &first.as_bytes()[..FILE_PREFIX_LEN],
        b"FSLXSEG\0\x01\0\0\0\x50\x01\0\0"
    );
    assert_eq!(first.file_len(), 985);
    assert_eq!(header_len, 336);
    assert_eq!(first.header().schema_id, 0xa312_ebf6_d136_07a5);
    assert_eq!(header_crc, 0x3e9c_358f);
    assert_eq!(first.file_xxh3(), 0x3d1c_8570_73b5_390f);
    assert_eq!(trailer_crc, 0x4752_5d26);
    assert_eq!(first.section_entries(), &PINNED_DEFAULT_ENTRIES);
    assert_eq!(first.as_bytes(), pinned_v1_fixture()?);
    assert_eq!(first.as_bytes(), second.as_bytes());
    assert_eq!(first.file_xxh3(), second.file_xxh3());
    assert_eq!(first.source_xxh3(), xxh3_64(first.as_bytes()));
    assert_eq!(first.source_xxh3(), second.source_xxh3());

    let reader = SegmentReader::from_owned(first.as_bytes().to_vec(), LEGACY_DEFAULT_SCHEMA)?;
    assert_eq!(reader.header(), first.header());
    assert_eq!(reader.section_entries(), first.section_entries());
    assert_eq!(reader.file_len(), first.file_len());
    assert_eq!(reader.file_xxh3(), first.file_xxh3());
    for expected in &owned {
        assert_eq!(
            reader.section(expected.kind)?.expect("fixture section"),
            expected.bytes
        );
    }
    reader.verify()?;

    let borrowed = SegmentReader::from_bytes(first.as_bytes(), LEGACY_DEFAULT_SCHEMA)?;
    assert_eq!(borrowed.header(), reader.header());
    borrowed.verify()?;

    // The shipping schema's analyzer change moves only the header's schema
    // identity: same length, section table and payloads.
    let shipping_owned = fixture_sections(DEFAULT_SCHEMA, false);
    let shipping = encode_owned(fixture_header(DEFAULT_SCHEMA), &shipping_owned)?;
    assert_eq!(shipping.header().schema_id, DEFAULT_SCHEMA.schema_id()?);
    assert_ne!(shipping.header().schema_id, first.header().schema_id);
    assert_eq!(shipping.file_len(), first.file_len());
    assert_eq!(shipping.section_entries(), first.section_entries());
    for (legacy, current) in owned.iter().zip(&shipping_owned) {
        assert_eq!(legacy.bytes, current.bytes);
    }
    SegmentReader::from_bytes(shipping.as_bytes(), DEFAULT_SCHEMA)?.verify()?;
    Ok(())
}

#[test]
fn encoded_segment_clone_shares_backing_and_unique_extraction_is_zero_copy() -> TestResult {
    let owned = fixture_sections(MINIMAL_SCHEMA, false);
    let unique = encode_owned(fixture_header(MINIMAL_SCHEMA), &owned)?;
    assert_eq!(Arc::strong_count(&unique.bytes), 1);
    let unique_ptr = unique.as_bytes().as_ptr();
    let extracted = unique.into_bytes();
    assert_eq!(
        extracted.as_ptr(),
        unique_ptr,
        "a uniquely owned segment must extract its bytes without copying"
    );

    let original = encode_owned(fixture_header(MINIMAL_SCHEMA), &owned)?;
    let shared = original.clone();
    assert_eq!(Arc::strong_count(&original.bytes), 2);
    assert!(
        std::ptr::eq(original.as_bytes(), shared.as_bytes()),
        "clone must share the immutable byte backing"
    );
    assert_eq!(original, shared);

    let copied = shared.into_bytes();
    assert_eq!(
        Arc::strong_count(&original.bytes),
        1,
        "shared extraction must release its backing reference"
    );
    assert_ne!(
        copied.as_ptr(),
        original.as_bytes().as_ptr(),
        "extraction from a shared segment must copy, never steal"
    );
    assert_eq!(copied, extracted);
    assert_eq!(original.as_bytes(), extracted.as_slice());
    Ok(())
}

#[test]
fn from_encoded_reader_retains_backing_and_matches_from_owned() -> TestResult {
    let owned = fixture_sections(DEFAULT_SCHEMA, false);
    let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
    let baseline = SegmentReader::from_owned(encoded.as_bytes().to_vec(), DEFAULT_SCHEMA)?;

    let sibling = encoded.clone();
    let reader = SegmentReader::from_encoded(encoded, DEFAULT_SCHEMA)?;
    assert_eq!(reader.header(), baseline.header());
    assert_eq!(reader.section_entries(), baseline.section_entries());
    assert_eq!(reader.file_xxh3(), baseline.file_xxh3());
    assert_eq!(reader.file_len(), baseline.file_len());
    reader.verify()?;
    assert!(
        std::ptr::eq(reader.source_bytes(), sibling.as_bytes()),
        "the reader must retain the encoded segment's shared byte backing"
    );
    Ok(())
}

#[test]
fn every_segment_byte_class_flip_is_public_typed_and_never_panics() -> TestResult {
    const SEED: u64 = 0x6d5a_56da_1b2c_3e4f;

    init_replay_tracing();
    let owned = fixture_sections(CONDITIONAL_SCHEMA, false);
    let encoded = encode_owned(fixture_header(CONDITIONAL_SCHEMA), &owned)?;
    let entries = encoded.section_entries().to_vec();
    assert_eq!(entries.len(), 10, "fixture must cover every v1 section");
    let golden = encoded.into_bytes();
    let header_len =
        usize::try_from(read_u32_at(&golden, 12).ok_or("corruption-sweep header length")?)?;
    let header_crc_offset = FILE_PREFIX_LEN + header_len;
    let trailer_start = golden.len() - TRAILER_LEN;
    let mut cases = vec![
        ("magic", 0),
        ("format version", 8),
        ("header length", 12),
        ("fixed header", FILE_PREFIX_LEN + 8),
        ("section table", table_entry_offset(0)),
        ("header CRC", header_crc_offset),
        ("alignment padding", header_crc_offset + HEADER_CRC_LEN),
        ("file checksum", trailer_start),
        ("trailer CRC", trailer_start + 8),
    ];
    for entry in entries {
        cases.push(("section payload", usize::try_from(entry.offset)?));
    }

    for (region, offset) in cases.iter().copied() {
        tracing::info!(
            codec = "FSLX_CONTAINER",
            seed = SEED,
            case = offset,
            region,
            replay = "every_segment_byte_class_flip_is_public_typed_and_never_panics",
            "starting byte-class corruption case"
        );
        let offset_u64 = u64::try_from(offset)?;
        let bit = u32::try_from((SEED ^ offset_u64) & 7)?;
        let mut mutated = golden.clone();
        mutated[offset] ^= 1_u8 << bit;

        let result = catch_unwind(AssertUnwindSafe(|| {
            SegmentReader::from_owned(mutated, CONDITIONAL_SCHEMA)
                .and_then(|reader| reader.verify())
        }));
        assert!(
            result.is_ok(),
            "seed={SEED:#x} offset={offset} region={region}: parser panicked"
        );
        let Err(codec_error) = result.expect("byte-class parser result") else {
            panic!(
                "seed={SEED:#x} offset={offset} region={region}: byte-class mutation was accepted"
            );
        };
        let public_error: SearchError = codec_error.into();
        assert!(
            matches!(public_error, SearchError::IndexCorrupted { .. }),
            "seed={SEED:#x} offset={offset} region={region}: mutation was not public typed corruption"
        );
    }

    tracing::info!(
        seed = SEED,
        mutations = cases.len(),
        "FSLX v1 byte-class corruption sweep passed"
    );
    Ok(())
}

#[test]
fn schema_conditionals_require_exact_known_section_sets() -> TestResult {
    for schema in [
        MINIMAL_SCHEMA,
        DEFAULT_SCHEMA,
        CONDITIONAL_SCHEMA,
        U64_SCHEMA,
    ] {
        let owned = fixture_sections(schema, false);
        let encoded = encode_owned(fixture_header(schema), &owned)?;
        let reader = SegmentReader::from_owned(encoded.into_bytes(), schema)?;
        assert_eq!(
            reader.section(SectionKind::POSITIONS)?.is_some(),
            schema_has_positions(schema)
        );
        assert_eq!(
            reader.section(SectionKind::NUMERIC)?.is_some(),
            schema_has_numeric_columns(schema)
        );
        assert_eq!(
            reader.section(SectionKind::STOREDMETA)?.is_some(),
            schema_has_stored_fields(schema)
        );
        reader.verify()?;
    }

    assert!(schema_has_numeric_columns(U64_SCHEMA));
    let mut missing_u64_numeric = fixture_sections(U64_SCHEMA, false);
    assert!(
        missing_u64_numeric
            .iter()
            .any(|section| section.kind == SectionKind::NUMERIC)
    );
    missing_u64_numeric.retain(|section| section.kind != SectionKind::NUMERIC);
    assert!(matches!(
        encode_owned(fixture_header(U64_SCHEMA), &missing_u64_numeric),
        Err(QuillError::Invariant { .. })
    ));

    for conditional in [
        SectionKind::POSITIONS,
        SectionKind::NUMERIC,
        SectionKind::STOREDMETA,
    ] {
        let mut missing = fixture_sections(CONDITIONAL_SCHEMA, false);
        missing.retain(|section| section.kind != conditional);
        assert!(matches!(
            encode_owned(fixture_header(CONDITIONAL_SCHEMA), &missing),
            Err(QuillError::Invariant { .. })
        ));

        let mut unexpected = fixture_sections(MINIMAL_SCHEMA, false);
        unexpected.push(OwnedSection {
            kind: conditional,
            flags: 0,
            bytes: vec![1],
        });
        unexpected.sort_unstable_by_key(|section| section.kind);
        assert!(matches!(
            encode_owned(fixture_header(MINIMAL_SCHEMA), &unexpected),
            Err(QuillError::Invariant { .. })
        ));
    }

    let encoded = encode_owned(
        fixture_header(MINIMAL_SCHEMA),
        &fixture_sections(MINIMAL_SCHEMA, false),
    )?;
    let stats_index = encoded
        .section_entries()
        .iter()
        .position(|entry| entry.kind == SectionKind::STATS)
        .expect("STATS entry");
    let mut unexpected_numeric = encoded.into_bytes();
    let kind_offset = table_entry_offset(stats_index);
    unexpected_numeric[kind_offset..kind_offset + 2]
        .copy_from_slice(&SectionKind::NUMERIC.raw().to_le_bytes());
    rewrite_header_crc(&mut unexpected_numeric);
    assert!(matches!(
        SegmentReader::from_owned(unexpected_numeric, MINIMAL_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));
    Ok(())
}

#[test]
fn section_layout_is_canonical_aligned_and_zero_padded() -> TestResult {
    let owned = fixture_sections(CONDITIONAL_SCHEMA, false);
    let encoded = encode_owned(fixture_header(CONDITIONAL_SCHEMA), &owned)?;
    let bytes = encoded.as_bytes();
    let declared_header_len = usize::try_from(read_u32_at(bytes, 12).expect("header len"))?;
    let mut cursor = FILE_PREFIX_LEN + declared_header_len + HEADER_CRC_LEN;
    for entry in encoded.section_entries() {
        let expected = align_up(cursor, FSLX_SECTION_ALIGNMENT).expect("alignment");
        let offset = usize::try_from(entry.offset)?;
        assert_eq!(offset, expected);
        assert_eq!(offset % FSLX_SECTION_ALIGNMENT, 0);
        assert!(bytes[cursor..offset].iter().all(|byte| *byte == 0));
        cursor = offset + usize::try_from(entry.len)?;
    }
    assert_eq!(cursor, bytes.len() - TRAILER_LEN);
    Ok(())
}

fn rewrite_header_crc(bytes: &mut [u8]) {
    let declared = usize::try_from(read_u32_at(bytes, 12).expect("header length"))
        .expect("usize header length");
    let header_end = FILE_PREFIX_LEN + declared;
    let checksum = crc32(&bytes[FILE_PREFIX_LEN..header_end]);
    bytes[header_end..header_end + HEADER_CRC_LEN].copy_from_slice(&checksum.to_le_bytes());
}

fn table_entry_offset(index: usize) -> usize {
    FILE_PREFIX_LEN + FIXED_HEADER_LEN + index * SECTION_ENTRY_LEN
}

#[test]
fn reader_skips_unknown_optional_sections_but_writer_requires_registry_entry() -> TestResult {
    let mut owned = fixture_sections(MINIMAL_SCHEMA, false);
    owned.push(OwnedSection {
        kind: SectionKind::from_raw(11),
        flags: OPTIONAL_SKIPPABLE,
        bytes: b"future optional payload".to_vec(),
    });
    assert!(matches!(
        encode_owned(fixture_header(MINIMAL_SCHEMA), &owned),
        Err(QuillError::Invariant { .. })
    ));

    let encoded =
        encode_owned_with_unregistered_extensions(fixture_header(MINIMAL_SCHEMA), &owned)?;
    let reader = SegmentReader::from_owned(encoded.as_bytes().to_vec(), MINIMAL_SCHEMA)?;
    assert!(reader.section(SectionKind::from_raw(11))?.is_none());
    assert!(
        reader
            .section_entries()
            .iter()
            .any(|entry| entry.kind == SectionKind::from_raw(11))
    );
    reader.verify()?;

    let unknown = encoded
        .section_entries()
        .last()
        .copied()
        .expect("unknown section entry");
    let mut corrupt_unknown = encoded.as_bytes().to_vec();
    corrupt_unknown[usize::try_from(unknown.offset)?] ^= 0x20;
    let reader = SegmentReader::from_owned(corrupt_unknown, MINIMAL_SCHEMA)?;
    assert!(reader.section(SectionKind::from_raw(11))?.is_none());
    assert!(matches!(
        reader.verify(),
        Err(QuillError::IndexCorrupted { .. })
    ));

    let mut caller_required = owned.clone();
    caller_required.last_mut().expect("unknown section").flags = 0;
    assert!(matches!(
        encode_owned(fixture_header(MINIMAL_SCHEMA), &caller_required),
        Err(QuillError::Invariant { .. })
    ));

    let mut durable_required = encoded.into_bytes();
    let last_index = owned.len() - 1;
    let flags_offset = table_entry_offset(last_index) + 2;
    durable_required[flags_offset..flags_offset + 2].copy_from_slice(&0_u16.to_le_bytes());
    rewrite_header_crc(&mut durable_required);
    assert!(matches!(
        SegmentReader::from_owned(durable_required, MINIMAL_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));
    Ok(())
}

#[test]
fn writer_rejects_bad_flags_order_ranges_and_budgets() -> TestResult {
    let mut bad_flags = fixture_sections(MINIMAL_SCHEMA, false);
    bad_flags[0].flags = OPTIONAL_SKIPPABLE;
    assert!(matches!(
        encode_owned(fixture_header(MINIMAL_SCHEMA), &bad_flags),
        Err(QuillError::Invariant { .. })
    ));

    let mut descending = fixture_sections(MINIMAL_SCHEMA, false);
    descending.swap(0, 1);
    assert!(matches!(
        encode_owned(fixture_header(MINIMAL_SCHEMA), &descending),
        Err(QuillError::Invariant { .. })
    ));

    let mut reversed = fixture_header(MINIMAL_SCHEMA);
    reversed.docid_lo = 108;
    assert!(matches!(
        encode_owned(reversed, &fixture_sections(MINIMAL_SCHEMA, false)),
        Err(QuillError::Invariant { .. })
    ));
    let mut empty = fixture_header(MINIMAL_SCHEMA);
    empty.docid_hi = empty.docid_lo;
    assert!(matches!(
        encode_owned(empty, &fixture_sections(MINIMAL_SCHEMA, false)),
        Err(QuillError::Invariant { .. })
    ));
    let mut wider_than_u32 = fixture_header(MINIMAL_SCHEMA);
    wider_than_u32.docid_lo = MAX_DOCID_EXCLUSIVE - 1;
    wider_than_u32.docid_hi = MAX_DOCID_EXCLUSIVE + 1;
    wider_than_u32.doc_count = 1;
    assert!(matches!(
        encode_owned(wider_than_u32, &fixture_sections(MINIMAL_SCHEMA, false)),
        Err(QuillError::Invariant { .. })
    ));
    let mut too_many_docs = fixture_header(MINIMAL_SCHEMA);
    too_many_docs.doc_count = 8;
    assert!(matches!(
        encode_owned(too_many_docs, &fixture_sections(MINIMAL_SCHEMA, false)),
        Err(QuillError::Invariant { .. })
    ));
    let mut u32_boundary = fixture_header(MINIMAL_SCHEMA);
    u32_boundary.docid_lo = MAX_DOCID_EXCLUSIVE - 1;
    u32_boundary.docid_hi = MAX_DOCID_EXCLUSIVE;
    u32_boundary.doc_count = 1;
    encode_owned(u32_boundary, &fixture_sections(MINIMAL_SCHEMA, false))?;

    let owned = fixture_sections(MINIMAL_SCHEMA, false);
    let borrowed: Vec<_> = owned
        .iter()
        .map(|section| SectionInput::new(section.kind, &section.bytes))
        .collect();
    assert!(matches!(
        EncodedSegment::encode_with_limits(
            fixture_header(MINIMAL_SCHEMA),
            &borrowed,
            SegmentLimits {
                max_sections: owned.len() - 1,
                ..SegmentLimits::default()
            }
        ),
        Err(QuillError::Resource { .. })
    ));
    assert!(matches!(
        EncodedSegment::encode_with_limits(
            fixture_header(MINIMAL_SCHEMA),
            &borrowed,
            SegmentLimits {
                max_file_bytes: 1,
                ..SegmentLimits::default()
            }
        ),
        Err(QuillError::Resource { .. })
    ));

    let encoded = EncodedSegment::encode(fixture_header(MINIMAL_SCHEMA), &borrowed)?;
    assert!(matches!(
        SegmentReader::from_owned_with_limits(
            encoded.into_bytes(),
            MINIMAL_SCHEMA,
            SegmentLimits {
                max_sections: owned.len() - 1,
                ..SegmentLimits::default()
            }
        ),
        Err(QuillError::Resource { .. })
    ));
    Ok(())
}

#[test]
fn section_checksums_are_lazy_and_cached_per_section() -> TestResult {
    let owned = fixture_sections(MINIMAL_SCHEMA, false);
    let encoded = encode_owned(fixture_header(MINIMAL_SCHEMA), &owned)?;
    let postings = encoded
        .section_entries()
        .iter()
        .find(|entry| entry.kind == SectionKind::POSTINGS)
        .copied()
        .expect("POSTINGS entry");
    let mut corrupt_bytes = encoded.into_bytes();
    let postings_offset = usize::try_from(postings.offset)?;
    corrupt_bytes[postings_offset] ^= 0x80;

    let reader = SegmentReader::from_owned(corrupt_bytes, MINIMAL_SCHEMA)?;
    assert!(reader.section(SectionKind::TERMDICT)?.is_some());
    for _ in 0..2 {
        assert!(matches!(
            reader.section(SectionKind::POSTINGS),
            Err(QuillError::IndexCorrupted { .. })
        ));
    }
    assert!(matches!(
        reader.verify(),
        Err(QuillError::IndexCorrupted { .. })
    ));
    Ok(())
}

#[test]
fn position_free_paths_do_not_touch_a_corrupt_positions_section() -> TestResult {
    assert!(matches!(DEFAULT_SCHEMA.fields[0].kind, FieldKind::Keyword));
    assert!(schema_has_positions(DEFAULT_SCHEMA));
    let owned = fixture_sections(DEFAULT_SCHEMA, false);
    let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
    let positions = encoded
        .section_entries()
        .iter()
        .find(|entry| entry.kind == SectionKind::POSITIONS)
        .copied()
        .expect("default-schema POSITIONS entry");
    let mut corrupt_bytes = encoded.into_bytes();
    let positions_offset = usize::try_from(positions.offset)?;
    corrupt_bytes[positions_offset] ^= 0x80;

    // Opening and the sections needed by an ordinary keyword term path
    // remain independent of POSITIONS first-touch validation.
    let reader = SegmentReader::from_owned(corrupt_bytes, DEFAULT_SCHEMA)?;
    for kind in [
        SectionKind::TERMDICT,
        SectionKind::POSTINGS,
        SectionKind::BLOCKMAX,
        SectionKind::DOCLEN,
    ] {
        assert!(reader.section(kind)?.is_some(), "kind={}", kind.raw());
    }
    assert!(matches!(
        reader.section(SectionKind::POSITIONS),
        Err(QuillError::IndexCorrupted { .. })
    ));
    Ok(())
}

#[test]
fn metadata_free_paths_do_not_touch_a_corrupt_stored_meta_section() -> TestResult {
    assert!(schema_has_stored_fields(DEFAULT_SCHEMA));
    let owned = fixture_sections(DEFAULT_SCHEMA, false);
    let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
    let stored_meta = encoded
        .section_entries()
        .iter()
        .find(|entry| entry.kind == SectionKind::STOREDMETA)
        .copied()
        .expect("default-schema STOREDMETA entry");
    let mut corrupt_bytes = encoded.into_bytes();
    let stored_meta_offset = usize::try_from(stored_meta.offset)?;
    corrupt_bytes[stored_meta_offset] ^= 0x20;

    // Structural open and score-only sections stay independent of lazy
    // stored-field checksum validation.
    let reader = SegmentReader::from_owned(corrupt_bytes, DEFAULT_SCHEMA)?;
    for kind in [
        SectionKind::TERMDICT,
        SectionKind::POSTINGS,
        SectionKind::BLOCKMAX,
        SectionKind::DOCLEN,
        SectionKind::STATS,
    ] {
        assert!(reader.section(kind)?.is_some(), "kind={}", kind.raw());
    }
    assert!(matches!(
        reader.section(SectionKind::STOREDMETA),
        Err(QuillError::IndexCorrupted { .. })
    ));
    Ok(())
}

#[test]
fn verify_freshly_checks_whole_file_after_section_hashes_pass() -> TestResult {
    let owned = fixture_sections(MINIMAL_SCHEMA, false);
    let encoded = encode_owned(fixture_header(MINIMAL_SCHEMA), &owned)?;
    let postings_index = encoded
        .section_entries()
        .iter()
        .position(|entry| entry.kind == SectionKind::POSTINGS)
        .expect("POSTINGS index");
    let postings = encoded.section_entries()[postings_index];
    let mut bytes = encoded.into_bytes();
    let postings_offset = usize::try_from(postings.offset)?;
    bytes[postings_offset] ^= 0x40;
    let postings_len = usize::try_from(postings.len)?;
    let replacement_hash = xxh3_64(&bytes[postings_offset..postings_offset + postings_len]);
    let hash_offset = table_entry_offset(postings_index) + 20;
    bytes[hash_offset..hash_offset + 8].copy_from_slice(&replacement_hash.to_le_bytes());
    rewrite_header_crc(&mut bytes);

    let actual_prefix_xxh3 = xxh3_64(&bytes[..bytes.len() - TRAILER_LEN]);
    let reader = SegmentReader::from_owned(bytes, MINIMAL_SCHEMA)?;
    assert_ne!(actual_prefix_xxh3, reader.file_xxh3());
    assert!(reader.section(SectionKind::POSTINGS)?.is_some());
    assert!(matches!(
        reader.verify_file_witness(),
        Err(QuillError::IndexCorrupted { .. })
    ));
    assert!(matches!(
        reader.verify(),
        Err(QuillError::IndexCorrupted { .. })
    ));
    Ok(())
}

#[test]
fn retryable_temp_adopts_exact_existing_bytes() -> TestResult {
    let directory = tempfile::tempdir()?.keep();
    let owned = fixture_sections(DEFAULT_SCHEMA, true);
    let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
    let first = encoded.write_temp(&directory)?;
    let temp_path = first.path().to_path_buf();
    let expected = fs::read(&temp_path)?;

    let retry = encoded.write_temp_retryable(&directory)?;

    assert_eq!(retry.path(), temp_path);
    assert_eq!(retry.segment_id(), first.segment_id());
    assert_eq!(retry.file_len(), first.file_len());
    assert_eq!(retry.file_xxh3(), first.file_xxh3());
    assert_eq!(retry.source_xxh3(), first.source_xxh3());
    assert_eq!(fs::read(&temp_path)?, expected);
    Ok(())
}

#[test]
fn retryable_temp_rejects_and_preserves_differing_bytes() -> TestResult {
    let directory = tempfile::tempdir()?.keep();
    let owned = fixture_sections(DEFAULT_SCHEMA, true);
    let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
    let pending = encoded.write_temp(&directory)?;
    let temp_path = pending.path().to_path_buf();
    let mut differing = fs::read(&temp_path)?;
    let changed = differing
        .last_mut()
        .ok_or("encoded segment fixture must not be empty")?;
    *changed ^= 0x80;
    fs::write(&temp_path, &differing)?;

    assert!(matches!(
        encoded.write_temp_retryable(&directory),
        Err(QuillError::Io(error))
            if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
    assert_eq!(fs::read(&temp_path)?, differing);
    Ok(())
}

#[test]
fn streamed_witness_agrees_with_mapped_verifier_and_rejects_corruption() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("seg-0123456789abcdef.fslx");
    let mut owned = fixture_sections(DEFAULT_SCHEMA, true);
    owned[0].bytes.resize(137_123, 0x5a);
    let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
    let prefix_len = encoded.as_bytes().len() - TRAILER_LEN;
    assert!(prefix_len > 2 * 64 * 1024);
    assert_ne!(prefix_len % (16 * 1024), 0);
    fs::write(&path, encoded.as_bytes())?;
    let open = || {
        SegmentReader::open_published_checked(
            &path,
            DEFAULT_SCHEMA,
            SegmentLimits::default(),
            |reader, file| reader.verify_streamed_file_witness(file),
        )
    };
    {
        let (reader, witness) = open()?;
        assert_eq!(witness, reader.verify_file_witness()?);
        assert_eq!(witness, encoded.file_xxh3());
        for section in &owned {
            assert_eq!(
                reader.section(section.kind)?,
                Some(section.bytes.as_slice())
            );
        }
    }

    // Mutate only after dropping the published mapping. The old trailer
    // remains intact, so structural open succeeds but both hashes refuse.
    let mut corrupt = encoded.as_bytes().to_vec();
    let final_section = encoded.section_entries().last().ok_or("missing section")?;
    let offset = usize::try_from(final_section.offset + final_section.len - 1)?;
    assert!(offset >= prefix_len / (16 * 1024) * (16 * 1024));
    corrupt[offset] ^= 0x40;
    fs::write(&path, &corrupt)?;
    assert!(matches!(
        SegmentReader::from_owned(corrupt, DEFAULT_SCHEMA)?.verify_file_witness(),
        Err(QuillError::IndexCorrupted { .. })
    ));
    assert!(matches!(open(), Err(QuillError::IndexCorrupted { .. })));
    for length in [0, FILE_PREFIX_LEN, encoded.as_bytes().len() - 1] {
        fs::write(&path, &encoded.as_bytes()[..length])?;
        assert!(
            matches!(open(), Err(QuillError::IndexCorrupted { .. })),
            "truncation at {length} must fail"
        );
    }
    Ok(())
}

#[test]
fn mmap_matches_owned_reader_and_temp_creation_never_overwrites() -> TestResult {
    let owned = fixture_sections(DEFAULT_SCHEMA, true);
    let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
    let directory = tempfile::tempdir()?;
    let pending = encoded.write_temp(directory.path())?;
    assert_eq!(
        pending.path().file_name().and_then(|name| name.to_str()),
        Some(".tmp-segment-0123456789abcdef")
    );
    assert_eq!(pending.segment_id(), encoded.header().segment_id);
    assert_eq!(pending.file_len(), encoded.file_len());
    assert_eq!(pending.file_xxh3(), encoded.file_xxh3());
    assert_eq!(pending.source_xxh3(), encoded.source_xxh3());
    let pending_bytes = fs::read(pending.path())?;
    assert_eq!(pending_bytes, encoded.as_bytes());
    assert_eq!(pending.source_xxh3(), xxh3_64(&pending_bytes));

    assert!(matches!(
        SegmentReader::open_published(pending.path(), DEFAULT_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));
    let second_create = encoded.write_temp(directory.path());
    assert!(matches!(
        second_create,
        Err(QuillError::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
    assert_eq!(fs::read(pending.path())?, encoded.as_bytes());

    let published = directory.path().join("seg-0123456789abcdef.fslx");
    fs::rename(pending.path(), &published)?;
    let mapped = SegmentReader::open_published(&published, DEFAULT_SCHEMA)?;
    let memory = SegmentReader::from_owned(encoded.as_bytes().to_vec(), DEFAULT_SCHEMA)?;
    assert_eq!(mapped.header(), memory.header());
    assert_eq!(mapped.section_entries(), memory.section_entries());
    for expected in &owned {
        assert_eq!(
            mapped.section(expected.kind)?,
            memory.section(expected.kind)?
        );
    }
    mapped.verify()?;

    let wrong_name = directory.path().join("seg-fedcba9876543210.fslx");
    fs::write(&wrong_name, encoded.as_bytes())?;
    assert!(matches!(
        SegmentReader::open_published(&wrong_name, DEFAULT_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));
    Ok(())
}

#[test]
fn published_segment_names_are_canonical_lowercase_hex() {
    assert_eq!(
        published_segment_id(Path::new("seg-0123456789abcdef.fslx")),
        Ok(0x0123_4567_89ab_cdef)
    );
    for invalid in [
        ".tmp-segment-0123456789abcdef",
        "seg-0123456789ABCDEF.fslx",
        "seg-1234.fslx",
        "seg-0123456789abcdef.fslx.quarantine",
    ] {
        assert!(
            published_segment_id(Path::new(invalid)).is_err(),
            "accepted {invalid}"
        );
    }
}

#[test]
fn framing_layout_and_schema_corruption_are_typed() -> TestResult {
    let owned = fixture_sections(MINIMAL_SCHEMA, false);
    let encoded = encode_owned(fixture_header(MINIMAL_SCHEMA), &owned)?;
    let original = encoded.as_bytes();

    let mut bad_magic = original.to_vec();
    bad_magic[0] ^= 1;
    assert!(matches!(
        SegmentReader::from_owned(bad_magic, MINIMAL_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));

    let mut bad_version = original.to_vec();
    bad_version[8..12].copy_from_slice(&2_u32.to_le_bytes());
    assert!(matches!(
        SegmentReader::from_owned(bad_version, MINIMAL_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));

    let mut bad_reserved = original.to_vec();
    let reserved_offset = FILE_PREFIX_LEN + 36;
    bad_reserved[reserved_offset..reserved_offset + 4].copy_from_slice(&1_u32.to_le_bytes());
    rewrite_header_crc(&mut bad_reserved);
    assert!(matches!(
        SegmentReader::from_owned(bad_reserved, MINIMAL_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));

    let mut bad_offset = original.to_vec();
    let first_offset_field = table_entry_offset(0) + 4;
    let first_offset = read_u64_at(&bad_offset, first_offset_field).expect("first offset");
    bad_offset[first_offset_field..first_offset_field + 8]
        .copy_from_slice(&(first_offset + 1).to_le_bytes());
    rewrite_header_crc(&mut bad_offset);
    assert!(matches!(
        SegmentReader::from_owned(bad_offset, MINIMAL_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));

    let declared = usize::try_from(read_u32_at(original, 12).expect("header length"))?;
    let header_crc_end = FILE_PREFIX_LEN + declared + HEADER_CRC_LEN;
    let first_section = usize::try_from(encoded.section_entries()[0].offset)?;
    assert!(header_crc_end < first_section);
    let mut bad_padding = original.to_vec();
    bad_padding[header_crc_end] = 1;
    assert!(matches!(
        SegmentReader::from_owned(bad_padding, MINIMAL_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));

    let mut bad_trailer = original.to_vec();
    let last = bad_trailer.len() - 1;
    bad_trailer[last] ^= 1;
    assert!(matches!(
        SegmentReader::from_owned(bad_trailer, MINIMAL_SCHEMA),
        Err(QuillError::IndexCorrupted { .. })
    ));

    assert!(matches!(
        SegmentReader::from_owned(original.to_vec(), DEFAULT_SCHEMA),
        Err(QuillError::UnknownSchema { .. })
    ));
    Ok(())
}

#[test]
fn every_kib_torn_boundary_and_hostile_prefix_fails_without_panicking() -> TestResult {
    const HOSTILE_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

    init_replay_tracing();
    let owned = fixture_sections(DEFAULT_SCHEMA, true);
    let encoded = encode_owned(fixture_header(DEFAULT_SCHEMA), &owned)?;
    let bytes = encoded.as_bytes();
    assert!(bytes.len() > 4 * 1024);
    for cut in (0..bytes.len()).step_by(1024) {
        let result = catch_unwind(AssertUnwindSafe(|| {
            SegmentReader::from_owned(bytes[..cut].to_vec(), DEFAULT_SCHEMA)
        }));
        assert!(result.is_ok(), "parser panicked at torn boundary {cut}");
        assert!(matches!(
            result.expect("caught result"),
            Err(QuillError::IndexCorrupted { .. })
        ));
    }
    let final_cut = bytes.len() - 1;
    let result = catch_unwind(AssertUnwindSafe(|| {
        SegmentReader::from_owned(bytes[..final_cut].to_vec(), DEFAULT_SCHEMA)
    }));
    assert!(result.is_ok());
    assert!(matches!(
        result.expect("caught final-cut result"),
        Err(QuillError::IndexCorrupted { .. })
    ));

    let mut hostile_header = bytes.to_vec();
    hostile_header[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    let result = catch_unwind(AssertUnwindSafe(|| {
        SegmentReader::from_owned(hostile_header, DEFAULT_SCHEMA)
    }));
    assert!(result.is_ok());
    assert!(matches!(
        result.expect("caught hostile-header result"),
        Err(QuillError::Resource { .. })
    ));

    let mut state = HOSTILE_SEED;
    for case in 0..256 {
        tracing::info!(
            codec = "FSLX_CONTAINER",
            seed = HOSTILE_SEED,
            case,
            replay = "every_kib_torn_boundary_and_hostile_prefix_fails_without_panicking",
            "starting hostile-prefix case"
        );
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let len = usize::try_from(state % 513).unwrap_or(0);
        let mut input = Vec::new();
        input.try_reserve_exact(len)?;
        for index in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let byte =
                u8::try_from((state ^ u64::try_from(index).unwrap_or(0)) & 0xff).unwrap_or(0);
            input.push(byte);
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            SegmentReader::from_owned(input, DEFAULT_SCHEMA)
        }));
        assert!(
            result.is_ok(),
            "seed={HOSTILE_SEED:#x} hostile case {case} panicked"
        );
        assert!(matches!(
            result.expect("caught hostile result"),
            Err(QuillError::IndexCorrupted { .. })
        ));
    }
    Ok(())
}
