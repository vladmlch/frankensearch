use super::super::*;
use super::*;

#[test]
fn doclen_roundtrip_quantizes_aligns_and_keeps_presence_external() -> TestResult {
    assert!(!std::mem::needs_drop::<DocLenSection<'static>>());

    let expected_fields = [1_u16, 2];
    let content = [Some(0), Some(41), None, Some(2_013_265_944), Some(u32::MAX)];
    let title = [Some(1), Some(42), Some(0), Some(65), Some(100)];
    let inputs = [
        DocLenFieldInput::new(1, &content),
        DocLenFieldInput::new(2, &title),
    ];
    let encoded = EncodedDocLenSection::encode(10, 15, &expected_fields, &inputs)?;
    assert_eq!(encoded.field_count(), 2);
    assert_eq!(encoded.as_bytes().len(), 133);
    assert_eq!(&encoded.as_bytes()[..6], &[1, 0, 64, 0, 0, 0]);
    assert_eq!(&encoded.as_bytes()[6..12], &[2, 0, 128, 0, 0, 0]);
    assert!(encoded.as_bytes()[12..64].iter().all(|&byte| byte == 0));
    assert!(encoded.as_bytes()[69..128].iter().all(|&byte| byte == 0));

    let section = encoded.section(&expected_fields)?;
    assert_eq!(section.docid_lo(), 10);
    assert_eq!(section.docid_hi(), 15);
    assert_eq!(section.field_count(), 2);
    assert_eq!(section.fields().len(), 2);
    assert_eq!(
        section
            .fields()
            .map(DocLenField::field_ord)
            .collect::<Vec<_>>(),
        expected_fields
    );
    assert!(section.field(0).is_none());
    assert!(section.field(3).is_none());
    let content = section.field(1).ok_or("content DOCLEN field")?;
    assert_eq!(content.fieldnorm_ids(), &[0, 40, 0, 255, 255]);
    assert_eq!(content.fieldnorm_id(9), None);
    assert_eq!(content.fieldnorm_id(10), Some(0));
    assert_eq!(content.fieldnorm_id(12), Some(DOCLEN_HOLE_FIELDNORM_ID));
    assert_eq!(content.fieldnorm_id(13), Some(255));
    assert_eq!(content.fieldnorm_id(15), None);
    assert_eq!(content.decoded_fieldnorm(11), Some(40));
    assert_eq!(content.decoded_fieldnorm(13), Some(2_013_265_944));
    assert_eq!(content.fieldnorm_id(12), content.fieldnorm_id(10));
    assert_ne!(content.fieldnorm_id(12), content.fieldnorm_id(13));

    let empty = EncodedDocLenSection::encode(7, 7, &[], &[])?;
    assert!(empty.as_bytes().is_empty());
    assert_eq!(empty.section(&[])?.field_count(), 0);
    Ok(())
}

#[test]
fn doclen_concat_inserts_gaps_preserves_raw_ids_and_is_associative() -> TestResult {
    let expected_fields = [1_u16, 4];
    let a_one = [Some(1), Some(41)];
    let a_four = [Some(65), Some(100)];
    let b_one = [Some(u32::MAX)];
    let b_four = [Some(0)];
    let c_one = [Some(42), Some(2_013_265_944)];
    let c_four = [None, Some(1)];
    let a = EncodedDocLenSection::encode(
        10,
        12,
        &expected_fields,
        &[
            DocLenFieldInput::new(1, &a_one),
            DocLenFieldInput::new(4, &a_four),
        ],
    )?;
    let b = EncodedDocLenSection::encode(
        14,
        15,
        &expected_fields,
        &[
            DocLenFieldInput::new(1, &b_one),
            DocLenFieldInput::new(4, &b_four),
        ],
    )?;
    let c = EncodedDocLenSection::encode(
        16,
        18,
        &expected_fields,
        &[
            DocLenFieldInput::new(1, &c_one),
            DocLenFieldInput::new(4, &c_four),
        ],
    )?;
    let first_section = a.section(&expected_fields)?;
    let second_section = b.section(&expected_fields)?;
    let third_section = c.section(&expected_fields)?;
    let sections = [
        first_section.clone(),
        second_section.clone(),
        third_section.clone(),
    ];
    let direct = EncodedDocLenSection::concatenate(&sections, &expected_fields)?;

    let monolithic_one = [
        Some(1),
        Some(41),
        None,
        None,
        Some(u32::MAX),
        None,
        Some(42),
        Some(2_013_265_944),
    ];
    let monolithic_four = [
        Some(65),
        Some(100),
        None,
        None,
        Some(0),
        None,
        None,
        Some(1),
    ];
    let monolithic = EncodedDocLenSection::encode(
        10,
        18,
        &expected_fields,
        &[
            DocLenFieldInput::new(1, &monolithic_one),
            DocLenFieldInput::new(4, &monolithic_four),
        ],
    )?;
    assert_eq!(direct.as_bytes(), monolithic.as_bytes());
    let direct_section = direct.section(&expected_fields)?;
    let direct_one = direct_section.field(1).ok_or("merged DOCLEN field 1")?;
    assert_eq!(
        &direct_one.fieldnorm_ids()[..2],
        first_section
            .field(1)
            .ok_or("source DOCLEN field 1")?
            .fieldnorm_ids()
    );
    assert_eq!(
        &direct_one.fieldnorm_ids()[2..4],
        &[DOCLEN_HOLE_FIELDNORM_ID; 2]
    );
    assert_eq!(direct_one.fieldnorm_ids()[5], DOCLEN_HOLE_FIELDNORM_ID);

    let left_pair = EncodedDocLenSection::concatenate(&sections[..2], &expected_fields)?;
    let combined_section = left_pair.section(&expected_fields)?;
    let left_associated = EncodedDocLenSection::concatenate(
        &[combined_section, third_section.clone()],
        &expected_fields,
    )?;
    assert_eq!(left_associated, direct);
    let right_pair = EncodedDocLenSection::concatenate(&sections[1..], &expected_fields)?;
    let trailing_section = right_pair.section(&expected_fields)?;
    let right_associated = EncodedDocLenSection::concatenate(
        &[first_section.clone(), trailing_section],
        &expected_fields,
    )?;
    assert_eq!(right_associated, direct);
    Ok(())
}

#[test]
fn doclen_concat_rejects_empty_reversed_overlap_field_drift_and_limits() -> TestResult {
    let expected_fields = [1_u16];
    let left_lengths = [Some(1), Some(2)];
    let right_lengths = [Some(3)];
    let left = EncodedDocLenSection::encode(
        10,
        12,
        &expected_fields,
        &[DocLenFieldInput::new(1, &left_lengths)],
    )?;
    let right = EncodedDocLenSection::encode(
        14,
        15,
        &expected_fields,
        &[DocLenFieldInput::new(1, &right_lengths)],
    )?;
    let overlap = EncodedDocLenSection::encode(
        11,
        12,
        &expected_fields,
        &[DocLenFieldInput::new(1, &right_lengths)],
    )?;
    let left_section = left.section(&expected_fields)?;
    let right_section = right.section(&expected_fields)?;
    let overlap_section = overlap.section(&expected_fields)?;
    assert!(matches!(
        EncodedDocLenSection::concatenate(&[], &expected_fields),
        Err(DocLenCodecError::EmptyConcat)
    ));
    assert!(matches!(
        EncodedDocLenSection::concatenate(
            &[right_section.clone(), left_section.clone()],
            &expected_fields
        ),
        Err(DocLenCodecError::ConcatRangeOrder { index: 1, .. })
    ));
    assert!(matches!(
        EncodedDocLenSection::concatenate(
            &[left_section.clone(), overlap_section],
            &expected_fields
        ),
        Err(DocLenCodecError::ConcatRangeOrder { index: 1, .. })
    ));
    assert!(matches!(
        EncodedDocLenSection::concatenate(&[left_section.clone(), right_section.clone()], &[2]),
        Err(DocLenCodecError::UnexpectedField { index: 0, .. })
    ));
    assert!(matches!(
        EncodedDocLenSection::concatenate_with_limits(
            &[left_section, right_section],
            &expected_fields,
            DocLenLimits {
                max_docid_span: 4,
                ..DocLenLimits::default()
            }
        ),
        Err(DocLenCodecError::ResourceLimit {
            resource: "docid span",
            ..
        })
    ));
    Ok(())
}

#[test]
fn randomized_doclen_roundtrip_matches_quantization_oracle() -> TestResult {
    const BASE_SEED: u64 = 0x8b8b_1e17_d0c1_ea55;
    const FIELDS: [u16; 3] = [1, 4, 9];
    const BOUNDARY_SPANS: [usize; 9] = [0, 1, 2, 127, 128, 129, 511, 512, 513];

    init_replay_tracing();
    for case in 0_u64..48 {
        let seed = BASE_SEED ^ case.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        tracing::info!(
            codec = "DOCLEN",
            seed,
            case,
            replay = "randomized_doclen_roundtrip_matches_quantization_oracle",
            "starting randomized roundtrip"
        );
        let mut state = seed;
        let span = if case < u64::try_from(BOUNDARY_SPANS.len())? {
            BOUNDARY_SPANS[usize::try_from(case)?]
        } else {
            usize::try_from(random_u32(&mut state) % 600)?
        };
        let docid_lo = u64::from(random_u32(&mut state) % 1_000_000);
        let docid_hi = docid_lo
            .checked_add(u64::try_from(span)?)
            .ok_or("randomized DOCLEN docid range overflow")?;

        let mut columns = Vec::with_capacity(FIELDS.len());
        for field_index in 0..FIELDS.len() {
            let mut lengths = Vec::with_capacity(span);
            for ordinal in 0..span {
                let value = match (ordinal + field_index) % 29 {
                    0 => None,
                    1 => Some(0),
                    2 => Some(u32::MAX),
                    _ => Some(random_u32(&mut state)),
                };
                lengths.push(value);
            }
            columns.push(lengths);
        }
        let inputs: Vec<_> = FIELDS
            .iter()
            .copied()
            .zip(&columns)
            .map(|(field_ord, lengths)| DocLenFieldInput::new(field_ord, lengths))
            .collect();

        let encoded = EncodedDocLenSection::encode(docid_lo, docid_hi, &FIELDS, &inputs)?;
        assert_eq!(
            encoded,
            EncodedDocLenSection::encode(docid_lo, docid_hi, &FIELDS, &inputs)?,
            "seed={seed:#x} case={case}: non-deterministic bytes"
        );
        let section = encoded.section(&FIELDS)?;
        for (&field_ord, lengths) in FIELDS.iter().zip(&columns) {
            let field = section.field(field_ord).ok_or("randomized DOCLEN field")?;
            for (ordinal, length) in lengths.iter().copied().enumerate() {
                let docid = docid_lo + u64::try_from(ordinal)?;
                let expected = length
                    .map(crate::contract::fieldnorm_to_id)
                    .unwrap_or(DOCLEN_HOLE_FIELDNORM_ID);
                assert_eq!(
                    field.fieldnorm_id(docid),
                    Some(expected),
                    "seed={seed:#x} case={case} field={field_ord} ordinal={ordinal}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn doclen_rejects_bad_ranges_fields_lengths_and_layout() -> TestResult {
    let lengths = [Some(1), None, Some(3)];
    let input = [DocLenFieldInput::new(1, &lengths)];
    assert!(matches!(
        EncodedDocLenSection::encode(4, 3, &[1], &input),
        Err(DocLenCodecError::InvalidDocIdRange { .. })
    ));
    assert!(matches!(
        EncodedDocLenSection::encode(0, 3, &[1, 1], &input),
        Err(DocLenCodecError::NonAscendingFields { .. })
    ));
    assert!(matches!(
        EncodedDocLenSection::encode(0, 3, &[2], &input),
        Err(DocLenCodecError::UnexpectedField { .. })
    ));
    assert!(matches!(
        EncodedDocLenSection::encode(0, 4, &[1], &input),
        Err(DocLenCodecError::ColumnLengthMismatch { .. })
    ));
    assert!(matches!(
        EncodedDocLenSection::encode_with_limits(
            0,
            3,
            &[1],
            &input,
            DocLenLimits {
                max_fields: 1,
                max_docid_span: 2,
                max_section_bytes: 1_000,
            }
        ),
        Err(DocLenCodecError::ResourceLimit {
            resource: "docid span",
            ..
        })
    ));
    assert!(matches!(
        EncodedDocLenSection::encode_with_limits(
            0,
            3,
            &[1],
            &input,
            DocLenLimits {
                max_fields: 0,
                max_docid_span: 3,
                max_section_bytes: 67,
            }
        ),
        Err(DocLenCodecError::ResourceLimit {
            resource: "field count",
            ..
        })
    ));
    assert!(matches!(
        EncodedDocLenSection::encode_with_limits(
            0,
            3,
            &[1],
            &input,
            DocLenLimits {
                max_fields: 1,
                max_docid_span: 3,
                max_section_bytes: 66,
            }
        ),
        Err(DocLenCodecError::ResourceLimit {
            resource: "section bytes",
            actual: 67,
            limit: 66,
        })
    ));

    let encoded = EncodedDocLenSection::encode(0, 3, &[1], &input)?;
    for cut in 0..encoded.as_bytes().len() {
        assert!(
            DocLenSection::parse(&encoded.as_bytes()[..cut], 0, 3, &[1]).is_err(),
            "cut={cut}"
        );
    }
    let mut offset = encoded.as_bytes().to_vec();
    offset[2..6].copy_from_slice(&65_u32.to_le_bytes());
    assert!(matches!(
        DocLenSection::parse(&offset, 0, 3, &[1]),
        Err(DocLenCodecError::NonCanonicalOffset { .. })
    ));
    let mut padding = encoded.as_bytes().to_vec();
    padding[6] = 1;
    assert!(matches!(
        DocLenSection::parse(&padding, 0, 3, &[1]),
        Err(DocLenCodecError::NonZeroPadding { offset: 6 })
    ));
    let mut trailing = encoded.as_bytes().to_vec();
    trailing.push(0);
    assert!(matches!(
        DocLenSection::parse(&trailing, 0, 3, &[1]),
        Err(DocLenCodecError::TrailingBytes { .. })
    ));
    assert!(matches!(
        DocLenSection::parse(encoded.as_bytes(), 0, 3, &[2]),
        Err(DocLenCodecError::UnexpectedField { .. })
    ));
    Ok(())
}
