use crate::scribe::ColumnarAccumulator;

use super::super::*;
use super::*;

#[test]
fn stored_meta_roundtrip_distinguishes_absent_empty_and_opaque_bytes() -> TestResult {
    let expected_fields = [2_u16, 7];
    let large = vec![b'x'; 70_000];
    let non_utf8 = [0xff, 0x00, 0x80];
    let metadata = [
        None,
        Some(b"".as_slice()),
        Some(non_utf8.as_slice()),
        Some(large.as_slice()),
        None,
    ];
    let content = [
        Some(b"alpha".as_slice()),
        None,
        Some(b"".as_slice()),
        None,
        Some(b"omega".as_slice()),
    ];
    let inputs = [
        StoredMetaFieldInput::new(2, &metadata),
        StoredMetaFieldInput::new(7, &content),
    ];
    let encoded = EncodedStoredMetaSection::encode(100, 105, &expected_fields, &inputs)?;
    assert_eq!(encoded.field_count(), 2);
    assert_eq!(encoded.blob_bytes(), 70_013);
    assert_eq!(&encoded.as_bytes()[..2], &2_u16.to_le_bytes());
    assert_eq!(&encoded.as_bytes()[6..8], &7_u16.to_le_bytes());
    assert_eq!(
        u32::from_le_bytes(encoded.as_bytes()[2..6].try_into()?),
        u32::try_from(2 * STORED_META_DIRECTORY_ENTRY_LEN)?
    );

    let section = encoded.section(&expected_fields)?;
    assert_eq!(section.docid_lo(), 100);
    assert_eq!(section.docid_hi(), 105);
    assert_eq!(section.span(), 5);
    assert_eq!(
        section
            .fields()
            .map(StoredMetaField::field_ord)
            .collect::<Vec<_>>(),
        expected_fields
    );
    let metadata = section.field(2).ok_or("metadata field")?;
    assert!(!metadata.is_present(99));
    assert!(!metadata.is_present(100));
    assert!(metadata.is_present(101));
    assert_eq!(metadata.get(100), None);
    assert_eq!(metadata.get(101), Some(b"".as_slice()));
    assert_eq!(metadata.get(102), Some(non_utf8.as_slice()));
    assert_eq!(metadata.get(103), Some(large.as_slice()));
    assert_eq!(metadata.get(104), None);
    assert_eq!(metadata.get(105), None);
    let content = section.field(7).ok_or("content field")?;
    assert_eq!(content.get(100), Some(b"alpha".as_slice()));
    assert_eq!(content.get(102), Some(b"".as_slice()));
    assert_eq!(content.get(104), Some(b"omega".as_slice()));

    let empty = EncodedStoredMetaSection::encode(7, 7, &[], &[])?;
    assert!(empty.as_bytes().is_empty());
    assert_eq!(empty.section(&[])?.field_count(), 0);
    Ok(())
}

#[test]
fn stored_meta_wire_is_exactly_packed_little_endian() -> TestResult {
    let expected_fields = [2_u16, 7];
    let field_two = [None, Some(b"".as_slice()), Some([0xff, 0x00].as_slice())];
    let field_seven = [Some(b"A".as_slice()), None, Some(b"BC".as_slice())];
    let encoded = EncodedStoredMetaSection::encode(
        40,
        43,
        &expected_fields,
        &[
            StoredMetaFieldInput::new(2, &field_two),
            StoredMetaFieldInput::new(7, &field_seven),
        ],
    )?;
    let golden = [
        0x02, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x07, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff,
        0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03,
        0x00, 0x00, 0x00, b'A', b'B', b'C',
    ];
    assert_eq!(encoded.as_bytes(), golden);

    let section = StoredMetaSection::parse(&golden, 40, 43, &expected_fields)?;
    let two = section.field(2).ok_or("field two")?;
    assert_eq!(two.get(40), None);
    assert_eq!(two.get(41), Some(b"".as_slice()));
    assert_eq!(two.get(42), Some([0xff, 0x00].as_slice()));
    let seven = section.field(7).ok_or("field seven")?;
    assert_eq!(seven.get(40), Some(b"A".as_slice()));
    assert_eq!(seven.get(41), None);
    assert_eq!(seven.get(42), Some(b"BC".as_slice()));

    let mut later_directory_offset = golden;
    later_directory_offset[8..12].copy_from_slice(&32_u32.to_le_bytes());
    assert!(matches!(
        StoredMetaSection::parse(&later_directory_offset, 40, 43, &expected_fields),
        Err(StoredMetaCodecError::TerminalOffsetMismatch { field_ord: 2, .. })
    ));
    assert!(matches!(
        StoredMetaSection::parse(&golden, 40, 43, &[2]),
        Err(StoredMetaCodecError::NonCanonicalFieldOffset { field_ord: 2, .. })
    ));
    Ok(())
}

#[test]
fn stored_meta_seals_sparse_scribe_columns_without_value_staging() -> TestResult {
    use crate::schema::DEFAULT_SCHEMA;
    use crate::scribe::{IndexedFieldValue, StoredFieldValue};

    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA)?;
    accumulator.add_document_with_stored(
        1,
        &[
            IndexedFieldValue::new(0, "doc-a"),
            IndexedFieldValue::new(1, ""),
        ],
        &[StoredFieldValue::new(3, b"")],
    )?;
    let opaque = [0xff, 0x00, 0x80];
    accumulator.add_document_with_stored(
        3,
        &[IndexedFieldValue::new(0, "doc-b")],
        &[StoredFieldValue::new(3, &opaque)],
    )?;

    let encoded = EncodedStoredMetaSection::encode_accumulator(100, 105, 100, &accumulator)?;
    let expected_fields = [0_u16, 1, 2, 3, 4];
    let section = encoded.section(&expected_fields)?;
    assert_eq!(section.span(), 5);
    let ids = section.field(0).ok_or("stored id")?;
    assert_eq!(ids.get(100), None);
    assert_eq!(ids.get(101), Some(b"doc-a".as_slice()));
    assert_eq!(ids.get(102), None);
    assert_eq!(ids.get(103), Some(b"doc-b".as_slice()));
    assert_eq!(ids.get(104), None);
    let content = section.field(1).ok_or("stored content")?;
    assert_eq!(content.get(101), Some(b"".as_slice()));
    assert_eq!(content.get(103), None);
    let metadata = section.field(3).ok_or("stored metadata")?;
    assert_eq!(metadata.get(101), Some(b"".as_slice()));
    assert_eq!(metadata.get(103), Some(opaque.as_slice()));
    assert_eq!(metadata.blob(), opaque);

    assert!(matches!(
        EncodedStoredMetaSection::encode_accumulator(102, 105, 100, &accumulator),
        Err(StoredMetaCodecError::SourceDocumentOutOfRange {
            index: 0,
            docid: 101,
            ..
        })
    ));
    Ok(())
}

#[test]
fn stored_meta_rejects_noncanonical_bytes_and_resource_abuse() -> TestResult {
    let expected_fields = [3_u16];
    let values = [None, Some(b"x".as_slice()), Some(b"".as_slice())];
    let input = [StoredMetaFieldInput::new(3, &values)];
    assert!(matches!(
        EncodedStoredMetaSection::encode(4, 3, &expected_fields, &input),
        Err(StoredMetaCodecError::InvalidDocIdRange { .. })
    ));
    assert!(matches!(
        EncodedStoredMetaSection::encode(0, 3, &[3, 3], &input),
        Err(StoredMetaCodecError::NonAscendingFields { .. })
    ));
    assert!(matches!(
        EncodedStoredMetaSection::encode(0, 3, &[4], &input),
        Err(StoredMetaCodecError::UnexpectedField { .. })
    ));
    assert!(matches!(
        EncodedStoredMetaSection::encode(0, 4, &expected_fields, &input),
        Err(StoredMetaCodecError::ColumnLengthMismatch { .. })
    ));
    assert!(matches!(
        EncodedStoredMetaSection::encode_with_limits(
            0,
            3,
            &expected_fields,
            &input,
            StoredMetaLimits {
                max_fields: 1,
                max_docid_span: 3,
                max_section_bytes: 1_000,
                max_field_blob_bytes: 1_000,
                max_value_bytes: 0,
            }
        ),
        Err(StoredMetaCodecError::ResourceLimit {
            resource: "value bytes",
            actual: 1,
            limit: 0,
        })
    ));

    let encoded = EncodedStoredMetaSection::encode(0, 3, &expected_fields, &input)?;
    assert!(matches!(
        EncodedStoredMetaSection::encode_with_limits(
            0,
            3,
            &expected_fields,
            &input,
            StoredMetaLimits {
                max_docid_span: 2,
                ..StoredMetaLimits::default()
            },
        ),
        Err(StoredMetaCodecError::ResourceLimit {
            resource: "docid span",
            actual: 3,
            limit: 2,
        })
    ));
    assert!(matches!(
        EncodedStoredMetaSection::encode_with_limits(
            0,
            3,
            &expected_fields,
            &input,
            StoredMetaLimits {
                max_field_blob_bytes: 0,
                ..StoredMetaLimits::default()
            },
        ),
        Err(StoredMetaCodecError::ResourceLimit {
            resource: "field blob bytes",
            actual: 1,
            limit: 0,
        })
    ));
    assert!(matches!(
        EncodedStoredMetaSection::encode_with_limits(
            0,
            3,
            &expected_fields,
            &input,
            StoredMetaLimits {
                max_section_bytes: 23,
                ..StoredMetaLimits::default()
            },
        ),
        Err(StoredMetaCodecError::ResourceLimit {
            resource: "section bytes",
            actual: 24,
            limit: 23,
        })
    ));

    for limits in [
        StoredMetaLimits {
            max_fields: 0,
            ..StoredMetaLimits::default()
        },
        StoredMetaLimits {
            max_docid_span: 2,
            ..StoredMetaLimits::default()
        },
        StoredMetaLimits {
            max_section_bytes: 23,
            ..StoredMetaLimits::default()
        },
        StoredMetaLimits {
            max_field_blob_bytes: 0,
            ..StoredMetaLimits::default()
        },
        StoredMetaLimits {
            max_value_bytes: 0,
            ..StoredMetaLimits::default()
        },
    ] {
        assert!(
            StoredMetaSection::parse_with_limits(
                encoded.as_bytes(),
                0,
                3,
                &expected_fields,
                limits,
            )
            .is_err(),
            "parse limit {limits:?} must be enforced"
        );
    }

    let present_empty = [Some(b"".as_slice())];
    let empty_encoded = EncodedStoredMetaSection::encode_with_limits(
        9,
        10,
        &expected_fields,
        &[StoredMetaFieldInput::new(3, &present_empty)],
        StoredMetaLimits {
            max_value_bytes: 0,
            ..StoredMetaLimits::default()
        },
    )?;
    assert_eq!(
        empty_encoded
            .section(&expected_fields)?
            .field(3)
            .ok_or("empty field")?
            .get(9),
        Some(b"".as_slice())
    );

    for cut in 0..encoded.as_bytes().len() {
        assert!(
            StoredMetaSection::parse(&encoded.as_bytes()[..cut], 0, 3, &expected_fields).is_err(),
            "cut={cut}"
        );
    }
    let mut bad_field_offset = encoded.as_bytes().to_vec();
    bad_field_offset[2..6].copy_from_slice(&7_u32.to_le_bytes());
    assert!(matches!(
        StoredMetaSection::parse(&bad_field_offset, 0, 3, &expected_fields),
        Err(StoredMetaCodecError::NonCanonicalFieldOffset { .. })
    ));
    let presence_start = STORED_META_DIRECTORY_ENTRY_LEN;
    let offsets_start = presence_start + 1;
    let mut bad_presence_tail = encoded.as_bytes().to_vec();
    bad_presence_tail[presence_start] |= 0x80;
    assert!(matches!(
        StoredMetaSection::parse(&bad_presence_tail, 0, 3, &expected_fields),
        Err(StoredMetaCodecError::NonCanonicalPresencePadding { .. })
    ));
    let mut bad_first_offset = encoded.as_bytes().to_vec();
    bad_first_offset[offsets_start..offsets_start + 4].copy_from_slice(&1_u32.to_le_bytes());
    assert!(matches!(
        StoredMetaSection::parse(&bad_first_offset, 0, 3, &expected_fields),
        Err(StoredMetaCodecError::NonZeroFirstOffset { .. })
    ));
    let mut descending = encoded.as_bytes().to_vec();
    descending[presence_start] |= 1;
    descending[offsets_start + 4..offsets_start + 8].copy_from_slice(&1_u32.to_le_bytes());
    descending[offsets_start + 8..offsets_start + 12].copy_from_slice(&0_u32.to_le_bytes());
    assert!(matches!(
        StoredMetaSection::parse(&descending, 0, 3, &expected_fields),
        Err(StoredMetaCodecError::DescendingOffsets { .. })
    ));
    let mut absent_with_bytes = encoded.as_bytes().to_vec();
    absent_with_bytes[presence_start] &= !(1 << 1);
    assert!(matches!(
        StoredMetaSection::parse(&absent_with_bytes, 0, 3, &expected_fields),
        Err(StoredMetaCodecError::AbsentValueHasBytes { index: 1, .. })
    ));
    let mut trailing = encoded.as_bytes().to_vec();
    trailing.push(0);
    assert!(matches!(
        StoredMetaSection::parse(&trailing, 0, 3, &expected_fields),
        Err(StoredMetaCodecError::TrailingBytes { .. })
    ));
    Ok(())
}

#[test]
fn stored_meta_concat_preserves_holes_blobs_and_schedule_equivalence() -> TestResult {
    let expected_fields = [3_u16];
    let first_values = [Some(b"a".as_slice()), None, Some(b"".as_slice())];
    let second_values = [Some(b"bc".as_slice()), Some(b"d".as_slice())];
    let third_values = [None, Some(b"ef".as_slice())];
    let first = EncodedStoredMetaSection::encode(
        10,
        13,
        &expected_fields,
        &[StoredMetaFieldInput::new(3, &first_values)],
    )?;
    let second = EncodedStoredMetaSection::encode(
        15,
        17,
        &expected_fields,
        &[StoredMetaFieldInput::new(3, &second_values)],
    )?;
    let third = EncodedStoredMetaSection::encode(
        17,
        19,
        &expected_fields,
        &[StoredMetaFieldInput::new(3, &third_values)],
    )?;
    let first_section = first.section(&expected_fields)?;
    let second_section = second.section(&expected_fields)?;
    let third_section = third.section(&expected_fields)?;
    let direct = EncodedStoredMetaSection::concatenate(
        &[
            first_section.clone(),
            second_section.clone(),
            third_section.clone(),
        ],
        &expected_fields,
    )?;

    let monolithic_values = [
        Some(b"a".as_slice()),
        None,
        Some(b"".as_slice()),
        None,
        None,
        Some(b"bc".as_slice()),
        Some(b"d".as_slice()),
        None,
        Some(b"ef".as_slice()),
    ];
    let monolithic = EncodedStoredMetaSection::encode(
        10,
        19,
        &expected_fields,
        &[StoredMetaFieldInput::new(3, &monolithic_values)],
    )?;
    assert_eq!(direct.as_bytes(), monolithic.as_bytes());
    let direct_section = direct.section(&expected_fields)?;
    let merged_field = direct_section.field(3).ok_or("merged stored field")?;
    assert_eq!(merged_field.blob(), b"abcdef");
    assert_eq!(merged_field.get(12), Some(b"".as_slice()));
    assert_eq!(merged_field.get(13), None);
    assert_eq!(merged_field.get(14), None);

    let staged_prefix = EncodedStoredMetaSection::concatenate(
        &[first_section.clone(), second_section.clone()],
        &expected_fields,
    )?;
    let staged_prefix_section = staged_prefix.section(&expected_fields)?;
    let staged = EncodedStoredMetaSection::concatenate(
        &[staged_prefix_section, third_section.clone()],
        &expected_fields,
    )?;
    assert_eq!(staged.as_bytes(), direct.as_bytes());

    let staged_suffix = EncodedStoredMetaSection::concatenate(
        &[second_section.clone(), third_section.clone()],
        &expected_fields,
    )?;
    let staged_suffix_section = staged_suffix.section(&expected_fields)?;
    let right_associated = EncodedStoredMetaSection::concatenate(
        &[first_section.clone(), staged_suffix_section],
        &expected_fields,
    )?;
    assert_eq!(right_associated.as_bytes(), direct.as_bytes());

    assert!(matches!(
        EncodedStoredMetaSection::concatenate(
            &[second_section.clone(), first_section.clone()],
            &expected_fields,
        ),
        Err(StoredMetaCodecError::ConcatRangeOrder { .. })
    ));
    let overlapping = EncodedStoredMetaSection::encode(
        12,
        14,
        &expected_fields,
        &[StoredMetaFieldInput::new(
            3,
            &[Some(b"x".as_slice()), Some(b"y".as_slice())],
        )],
    )?;
    assert!(matches!(
        EncodedStoredMetaSection::concatenate(
            &[first_section, overlapping.section(&expected_fields)?],
            &expected_fields,
        ),
        Err(StoredMetaCodecError::ConcatRangeOrder { .. })
    ));
    assert!(matches!(
        EncodedStoredMetaSection::concatenate(&[], &expected_fields),
        Err(StoredMetaCodecError::EmptyConcat)
    ));
    Ok(())
}

#[test]
fn stored_meta_concat_rebases_multiple_fields_without_decoding_blobs() -> TestResult {
    let expected_fields = [3_u16, 8];
    let left_three = [Some(b"a".as_slice()), None];
    let left_eight = [Some(b"".as_slice()), Some(b"x".as_slice())];
    let right_three = [Some(b"b".as_slice()), Some(b"".as_slice())];
    let right_eight = [None, Some([0xff, 0x00].as_slice())];
    let left = EncodedStoredMetaSection::encode(
        0,
        2,
        &expected_fields,
        &[
            StoredMetaFieldInput::new(3, &left_three),
            StoredMetaFieldInput::new(8, &left_eight),
        ],
    )?;
    let right = EncodedStoredMetaSection::encode(
        3,
        5,
        &expected_fields,
        &[
            StoredMetaFieldInput::new(3, &right_three),
            StoredMetaFieldInput::new(8, &right_eight),
        ],
    )?;
    let left_section = left.section(&expected_fields)?;
    let right_section = right.section(&expected_fields)?;
    let merged = EncodedStoredMetaSection::concatenate(
        &[left_section.clone(), right_section.clone()],
        &expected_fields,
    )?;

    let monolithic_three = [
        Some(b"a".as_slice()),
        None,
        None,
        Some(b"b".as_slice()),
        Some(b"".as_slice()),
    ];
    let monolithic_eight = [
        Some(b"".as_slice()),
        Some(b"x".as_slice()),
        None,
        None,
        Some([0xff, 0x00].as_slice()),
    ];
    let monolithic = EncodedStoredMetaSection::encode(
        0,
        5,
        &expected_fields,
        &[
            StoredMetaFieldInput::new(3, &monolithic_three),
            StoredMetaFieldInput::new(8, &monolithic_eight),
        ],
    )?;
    assert_eq!(merged.as_bytes(), monolithic.as_bytes());
    let merged_section = merged.section(&expected_fields)?;
    assert_eq!(merged_section.field(3).ok_or("field three")?.blob(), b"ab");
    assert_eq!(
        merged_section.field(8).ok_or("field eight")?.blob(),
        [b'x', 0xff, 0x00]
    );
    assert!(matches!(
        EncodedStoredMetaSection::concatenate_with_limits(
            &[left_section, right_section],
            &expected_fields,
            StoredMetaLimits {
                max_value_bytes: 0,
                ..StoredMetaLimits::default()
            },
        ),
        Err(StoredMetaCodecError::ResourceLimit {
            resource: "value bytes",
            actual: 1,
            limit: 0,
        })
    ));
    Ok(())
}
