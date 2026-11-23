use super::super::*;
use super::*;

#[test]
fn stats_wire_is_packed_little_endian_and_uses_raw_average() -> TestResult {
    let expected_fields = [1_u16, 2];
    let rows = [FieldStats::new(1, 21, 3), FieldStats::new(2, 3, 3)];
    let encoded = EncodedStatsSection::encode(&expected_fields, &rows, 3)?;
    assert_eq!(encoded.field_count(), 2);
    assert_eq!(encoded.as_bytes().len(), 2 * STATS_ENTRY_LEN);
    assert_eq!(
        encoded.as_bytes(),
        &[
            1, 0, 21, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
        ]
    );
    let section = encoded.section(&expected_fields)?;
    assert_eq!(section.rows(), rows);
    assert_eq!(section.segment_doc_count(), 3);
    let content = section.field(1).ok_or("content STATS row")?;
    assert_eq!(
        content.average_field_length().map(f32::to_bits),
        Some(7.0_f32.to_bits())
    );

    let raw = FieldStats::new(1, 148, 3)
        .average_field_length()
        .ok_or("non-empty raw average")?;
    let decoded = [41_u32, 42, 65]
        .into_iter()
        .map(crate::contract::fieldnorm_to_id)
        .map(crate::contract::id_to_fieldnorm)
        .sum::<u32>() as f32
        / 3.0;
    assert_eq!(raw.to_bits(), (148.0_f32 / 3.0).to_bits());
    assert_ne!(raw.to_bits(), decoded.to_bits());
    Ok(())
}

#[test]
fn stats_rejects_field_denominator_and_byte_corruption() -> TestResult {
    let rows = [FieldStats::new(1, 5, 2), FieldStats::new(2, 0, 2)];
    assert!(matches!(
        EncodedStatsSection::encode(&[1, 1], &rows, 2),
        Err(StatsCodecError::NonAscendingFields { .. })
    ));
    assert!(matches!(
        EncodedStatsSection::encode(&[1, 3], &rows, 2),
        Err(StatsCodecError::UnexpectedField { .. })
    ));
    let wrong_count = [FieldStats::new(1, 5, 1)];
    assert!(matches!(
        EncodedStatsSection::encode(&[1], &wrong_count, 2),
        Err(StatsCodecError::DocCountMismatch { .. })
    ));
    let mixed_denominators = [FieldStats::new(1, 5, 2), FieldStats::new(2, 0, 1)];
    assert!(matches!(
        EncodedStatsSection::encode(&[1, 2], &mixed_denominators, 2),
        Err(StatsCodecError::DocCountMismatch { .. })
    ));
    let tokens_without_docs = [FieldStats::new(1, 1, 0)];
    assert!(matches!(
        EncodedStatsSection::encode(&[1], &tokens_without_docs, 0),
        Err(StatsCodecError::TokensWithoutDocuments { .. })
    ));
    let one_row = [FieldStats::new(1, 5, 2)];
    assert!(matches!(
        EncodedStatsSection::encode_with_limits(
            &[1],
            &one_row,
            2,
            StatsLimits {
                max_fields: 0,
                max_section_bytes: 14,
            }
        ),
        Err(StatsCodecError::ResourceLimit {
            resource: "field count",
            ..
        })
    ));
    assert!(matches!(
        EncodedStatsSection::encode_with_limits(
            &[1],
            &one_row,
            2,
            StatsLimits {
                max_fields: 1,
                max_section_bytes: 13,
            }
        ),
        Err(StatsCodecError::ResourceLimit {
            resource: "section bytes",
            actual: 14,
            limit: 13,
        })
    ));

    let encoded = EncodedStatsSection::encode(&[1, 2], &rows, 2)?;
    for cut in 0..encoded.as_bytes().len() {
        assert!(matches!(
            StatsSection::parse(&encoded.as_bytes()[..cut], &[1, 2], 2),
            Err(StatsCodecError::LengthMismatch { .. })
        ));
    }
    let mut trailing = encoded.as_bytes().to_vec();
    trailing.push(0);
    assert!(matches!(
        StatsSection::parse(&trailing, &[1, 2], 2),
        Err(StatsCodecError::LengthMismatch { .. })
    ));
    let mut wrong_field = encoded.as_bytes().to_vec();
    wrong_field[0..2].copy_from_slice(&9_u16.to_le_bytes());
    assert!(matches!(
        StatsSection::parse(&wrong_field, &[1, 2], 2),
        Err(StatsCodecError::UnexpectedField { .. })
    ));
    let mut wrong_doc_count = encoded.as_bytes().to_vec();
    wrong_doc_count[10..14].copy_from_slice(&1_u32.to_le_bytes());
    assert!(matches!(
        StatsSection::parse(&wrong_doc_count, &[1, 2], 2),
        Err(StatsCodecError::DocCountMismatch { .. })
    ));
    Ok(())
}

#[test]
fn stats_aggregation_checked_sums_identical_field_sets() -> TestResult {
    let first_rows = [FieldStats::new(1, 21, 3), FieldStats::new(2, 3, 3)];
    let second_rows = [FieldStats::new(1, 9, 2), FieldStats::new(2, 0, 2)];
    let first = EncodedStatsSection::encode(&[1, 2], &first_rows, 3)?.section(&[1, 2])?;
    let second = EncodedStatsSection::encode(&[1, 2], &second_rows, 2)?.section(&[1, 2])?;
    let aggregate = aggregate_field_stats([&first, &second])?;
    assert_eq!(
        aggregate,
        [
            SnapshotFieldStats {
                field_ord: 1,
                total_tokens: 30,
                doc_count: 5,
            },
            SnapshotFieldStats {
                field_ord: 2,
                total_tokens: 3,
                doc_count: 5,
            },
        ]
    );
    assert_eq!(
        aggregate[0].average_field_length().map(f32::to_bits),
        Some(6.0_f32.to_bits())
    );
    assert!(aggregate_field_stats(std::iter::empty::<&StatsSection>())?.is_empty());
    assert_eq!(
        SnapshotFieldStats {
            field_ord: 1,
            total_tokens: 0,
            doc_count: 0,
        }
        .average_field_length(),
        None
    );

    let short_rows = [FieldStats::new(1, 9, 2)];
    let short = EncodedStatsSection::encode(&[1], &short_rows, 2)?.section(&[1])?;
    assert!(matches!(
        aggregate_field_stats([&first, &short]),
        Err(StatsCodecError::AggregateFieldCountMismatch { .. })
    ));
    let wrong_field_rows = [FieldStats::new(1, 9, 2), FieldStats::new(3, 0, 2)];
    let wrong_field =
        EncodedStatsSection::encode(&[1, 3], &wrong_field_rows, 2)?.section(&[1, 3])?;
    assert!(matches!(
        aggregate_field_stats([&first, &wrong_field]),
        Err(StatsCodecError::AggregateFieldMismatch { .. })
    ));
    let maximum_rows = [FieldStats::new(1, u64::MAX, 1)];
    let one_rows = [FieldStats::new(1, 1, 1)];
    let maximum = EncodedStatsSection::encode(&[1], &maximum_rows, 1)?.section(&[1])?;
    let one = EncodedStatsSection::encode(&[1], &one_rows, 1)?.section(&[1])?;
    assert!(matches!(
        aggregate_field_stats([&maximum, &one]),
        Err(StatsCodecError::AggregateOverflow {
            counter: "total_tokens",
            ..
        })
    ));
    Ok(())
}

#[test]
fn compaction_replaces_stats_and_canonically_fills_removed_doc() -> TestResult {
    let original_lengths = [Some(41), Some(42), Some(65)];
    let replacement_lengths = [Some(41), None, Some(65)];
    let original_input = [DocLenFieldInput::new(1, &original_lengths)];
    let replacement_input = [DocLenFieldInput::new(1, &replacement_lengths)];
    let original_doclen = EncodedDocLenSection::encode(0, 3, &[1], &original_input)?;
    let replacement_doclen = EncodedDocLenSection::encode(0, 3, &[1], &replacement_input)?;
    let original_ids = original_doclen
        .section(&[1])?
        .field(1)
        .ok_or("original DOCLEN field")?
        .fieldnorm_ids()
        .to_vec();
    let replacement_ids = replacement_doclen
        .section(&[1])?
        .field(1)
        .ok_or("replacement DOCLEN field")?
        .fieldnorm_ids()
        .to_vec();
    assert_eq!(replacement_ids[0], original_ids[0]);
    assert_eq!(replacement_ids[1], DOCLEN_HOLE_FIELDNORM_ID);
    assert_eq!(replacement_ids[2], original_ids[2]);

    let retained_decoded_tokens = [original_ids[0], original_ids[2]]
        .into_iter()
        .map(crate::contract::id_to_fieldnorm)
        .map(u64::from)
        .sum::<u64>();
    let replacement_rows = [FieldStats::new(1, retained_decoded_tokens, 2)];
    let replacement_stats =
        EncodedStatsSection::encode(&[1], &replacement_rows, 2)?.section(&[1])?;
    let aggregate = aggregate_field_stats([&replacement_stats])?;
    assert_eq!(aggregate[0].total_tokens, retained_decoded_tokens);
    assert_eq!(aggregate[0].doc_count, 2);
    assert_ne!(aggregate[0].total_tokens, 148);
    assert_eq!(
        aggregate[0].average_field_length().map(f32::to_bits),
        Some((retained_decoded_tokens as f32 / 2.0).to_bits())
    );
    Ok(())
}
