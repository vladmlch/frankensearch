use std::ops::Bound;

use super::super::*;
use super::*;

#[test]
fn numeric_wire_golden_is_exact_and_independently_decodable() -> TestResult {
    const EXPECTED: [u8; 48] = [
        0, 0, 2, 0, 0, 0, // signed field header
        255, 255, 255, 255, 255, 255, 255, 255, 2, 0, 0, 0, // -1, doc 2
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 1, doc 0
        1, 0, 1, 0, 0, 0, // unsigned field header
        0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, // 2^63, doc 1
    ];
    let signed = [NumericEntry::i64(1, 0), NumericEntry::i64(-1, 2)];
    let unsigned = [NumericEntry::u64(1_u64 << 63, 1)];
    let fields = [
        NumericFieldInput::new(0, &signed),
        NumericFieldInput::new(1, &unsigned),
    ];
    let encoded = EncodedNumericSection::encode(NUMERIC_TEST_SCHEMA, 0, 3, &fields)?;
    assert_eq!(encoded.as_bytes(), EXPECTED);

    let section = NumericSection::parse(&EXPECTED, NUMERIC_TEST_SCHEMA, 0, 3)?;
    assert_eq!(
        section
            .field(0)
            .ok_or("signed wire-golden field")?
            .entries()
            .collect::<Vec<_>>(),
        vec![NumericEntry::i64(-1, 2), NumericEntry::i64(1, 0)]
    );
    assert_eq!(
        section
            .field(1)
            .ok_or("unsigned wire-golden field")?
            .entries()
            .collect::<Vec<_>>(),
        vec![NumericEntry::u64(1_u64 << 63, 1)]
    );
    Ok(())
}

#[test]
fn numeric_roundtrip_preserves_typed_extremes_and_range_bounds() -> TestResult {
    let signed = [
        NumericEntry::i64(i64::MAX, 107),
        NumericEntry::i64(-5, 104),
        NumericEntry::i64(i64::MIN, 109),
        NumericEntry::i64(-5, 101),
        NumericEntry::i64(0, 102),
    ];
    let unsigned = [
        NumericEntry::u64(u64::MAX, 108),
        NumericEntry::u64(0, 103),
        NumericEntry::u64(1_u64 << 63, 105),
    ];
    let fields = [
        NumericFieldInput::new(0, &signed),
        NumericFieldInput::new(1, &unsigned),
    ];
    let encoded = EncodedNumericSection::encode(NUMERIC_TEST_SCHEMA, 100, 110, &fields)?;
    assert_eq!(
        encoded.as_bytes().len(),
        2 * NUMERIC_FIELD_HEADER_LEN + 8 * NUMERIC_PAIR_LEN
    );
    let section = encoded.section()?;
    assert_eq!(section.field_count(), 2);
    assert_eq!(section.entry_count(), 8);

    let signed_field = section.field(0).ok_or("signed NUMERIC field")?;
    assert!(signed_field.is_signed());
    assert_eq!(
        signed_field.entries().collect::<Vec<_>>(),
        vec![
            NumericEntry::i64(i64::MIN, 109),
            NumericEntry::i64(-5, 101),
            NumericEntry::i64(-5, 104),
            NumericEntry::i64(0, 102),
            NumericEntry::i64(i64::MAX, 107),
        ]
    );
    let signed_range = signed_field.range_docids(
        Bound::Included(NumericValue::I64(-5)),
        Bound::Excluded(NumericValue::I64(i64::MAX)),
    )?;
    assert_eq!(signed_range.as_slice(), &[101, 102, 104]);
    assert!(signed_range.contains(102));
    assert!(!signed_range.contains(99));
    assert!(!signed_range.contains(107));
    let reversed = signed_field.range_docids(
        Bound::Included(NumericValue::I64(10)),
        Bound::Included(NumericValue::I64(-10)),
    )?;
    assert!(reversed.is_empty());
    assert!(matches!(
        signed_field.range_docids(Bound::Included(NumericValue::U64(0)), Bound::Unbounded),
        Err(NumericCodecError::BoundTypeMismatch {
            field_ord: 0,
            expected: "i64",
            actual: "u64"
        })
    ));

    let unsigned_field = section.field(1).ok_or("unsigned NUMERIC field")?;
    assert!(!unsigned_field.is_signed());
    assert_eq!(
        unsigned_field.entries().collect::<Vec<_>>(),
        vec![
            NumericEntry::u64(0, 103),
            NumericEntry::u64(1_u64 << 63, 105),
            NumericEntry::u64(u64::MAX, 108),
        ]
    );
    let high_unsigned = unsigned_field.range_docids(
        Bound::Included(NumericValue::U64(1_u64 << 63)),
        Bound::Unbounded,
    )?;
    assert_eq!(high_unsigned.iter().collect::<Vec<_>>(), vec![105, 108]);
    Ok(())
}

#[test]
fn numeric_k_way_merge_handles_overlapping_values_and_is_associative() -> TestResult {
    let a_signed = [
        NumericEntry::i64(7, 101),
        NumericEntry::i64(i64::MIN, 102),
        NumericEntry::i64(-5, 100),
    ];
    let a_unsigned = [NumericEntry::u64(9, 103), NumericEntry::u64(0, 100)];
    let b_signed = [
        NumericEntry::i64(i64::MAX, 111),
        NumericEntry::i64(-5, 110),
        NumericEntry::i64(0, 112),
    ];
    let b_unsigned = [NumericEntry::u64(u64::MAX, 110), NumericEntry::u64(0, 112)];
    let c_signed = [NumericEntry::i64(7, 121), NumericEntry::i64(-5, 120)];
    let c_unsigned = [
        NumericEntry::u64(1_u64 << 63, 121),
        NumericEntry::u64(9, 120),
    ];
    let a = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        100,
        104,
        &[
            NumericFieldInput::new(0, &a_signed),
            NumericFieldInput::new(1, &a_unsigned),
        ],
    )?;
    let b = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        110,
        113,
        &[
            NumericFieldInput::new(0, &b_signed),
            NumericFieldInput::new(1, &b_unsigned),
        ],
    )?;
    let c = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        120,
        122,
        &[
            NumericFieldInput::new(0, &c_signed),
            NumericFieldInput::new(1, &c_unsigned),
        ],
    )?;
    let first_section = a.section()?;
    let second_section = b.section()?;
    let third_section = c.section()?;
    let sections = [
        first_section.clone(),
        second_section.clone(),
        third_section.clone(),
    ];
    let direct = EncodedNumericSection::merge_sorted(NUMERIC_TEST_SCHEMA, &sections)?;
    assert_eq!(direct.entry_count(), 14);
    let direct_section = direct.section()?;
    assert_eq!(
        direct_section
            .field(0)
            .ok_or("merged signed NUMERIC field")?
            .entries()
            .collect::<Vec<_>>(),
        vec![
            NumericEntry::i64(i64::MIN, 102),
            NumericEntry::i64(-5, 100),
            NumericEntry::i64(-5, 110),
            NumericEntry::i64(-5, 120),
            NumericEntry::i64(0, 112),
            NumericEntry::i64(7, 101),
            NumericEntry::i64(7, 121),
            NumericEntry::i64(i64::MAX, 111),
        ]
    );
    assert_eq!(
        direct_section
            .field(1)
            .ok_or("merged unsigned NUMERIC field")?
            .entries()
            .collect::<Vec<_>>(),
        vec![
            NumericEntry::u64(0, 100),
            NumericEntry::u64(0, 112),
            NumericEntry::u64(9, 103),
            NumericEntry::u64(9, 120),
            NumericEntry::u64(1_u64 << 63, 121),
            NumericEntry::u64(u64::MAX, 110),
        ]
    );

    let all_signed = [
        NumericEntry::i64(7, 101),
        NumericEntry::i64(i64::MIN, 102),
        NumericEntry::i64(-5, 100),
        NumericEntry::i64(i64::MAX, 111),
        NumericEntry::i64(-5, 110),
        NumericEntry::i64(0, 112),
        NumericEntry::i64(7, 121),
        NumericEntry::i64(-5, 120),
    ];
    let all_unsigned = [
        NumericEntry::u64(9, 103),
        NumericEntry::u64(0, 100),
        NumericEntry::u64(u64::MAX, 110),
        NumericEntry::u64(0, 112),
        NumericEntry::u64(1_u64 << 63, 121),
        NumericEntry::u64(9, 120),
    ];
    let monolithic = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        100,
        122,
        &[
            NumericFieldInput::new(0, &all_signed),
            NumericFieldInput::new(1, &all_unsigned),
        ],
    )?;
    assert_eq!(direct.as_bytes(), monolithic.as_bytes());

    let left_pair = EncodedNumericSection::merge_sorted(
        NUMERIC_TEST_SCHEMA,
        &[first_section.clone(), second_section.clone()],
    )?;
    let combined_section = left_pair.section()?;
    let left_associated = EncodedNumericSection::merge_sorted(
        NUMERIC_TEST_SCHEMA,
        &[combined_section, third_section.clone()],
    )?;
    assert_eq!(left_associated, direct);
    let right_pair =
        EncodedNumericSection::merge_sorted(NUMERIC_TEST_SCHEMA, &[second_section, third_section])?;
    let trailing_section = right_pair.section()?;
    let right_associated = EncodedNumericSection::merge_sorted(
        NUMERIC_TEST_SCHEMA,
        &[first_section, trailing_section],
    )?;
    assert_eq!(right_associated, direct);
    Ok(())
}

#[test]
fn numeric_merge_rejects_empty_reversed_overlap_and_resource_exhaustion() -> TestResult {
    let left = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        10,
        12,
        &[
            NumericFieldInput::new(0, &[NumericEntry::i64(1, 10)]),
            NumericFieldInput::new(1, &[NumericEntry::u64(1, 11)]),
        ],
    )?;
    let right = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        14,
        16,
        &[
            NumericFieldInput::new(0, &[NumericEntry::i64(1, 14)]),
            NumericFieldInput::new(1, &[NumericEntry::u64(1, 15)]),
        ],
    )?;
    let overlap = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        11,
        13,
        &[
            NumericFieldInput::new(0, &[NumericEntry::i64(2, 11)]),
            NumericFieldInput::new(1, &[]),
        ],
    )?;
    let left_section = left.section()?;
    let right_section = right.section()?;
    let overlap_section = overlap.section()?;
    assert!(matches!(
        EncodedNumericSection::merge_sorted(NUMERIC_TEST_SCHEMA, &[]),
        Err(NumericCodecError::EmptyMerge)
    ));
    assert!(matches!(
        EncodedNumericSection::merge_sorted(
            NUMERIC_TEST_SCHEMA,
            &[right_section.clone(), left_section.clone()]
        ),
        Err(NumericCodecError::MergeRangeOrder { index: 1, .. })
    ));
    assert!(matches!(
        EncodedNumericSection::merge_sorted(
            NUMERIC_TEST_SCHEMA,
            &[left_section.clone(), overlap_section]
        ),
        Err(NumericCodecError::MergeRangeOrder { index: 1, .. })
    ));
    assert!(matches!(
        EncodedNumericSection::merge_sorted_with_limits(
            NUMERIC_TEST_SCHEMA,
            &[left_section, right_section],
            NumericLimits {
                max_total_entries: 3,
                ..NumericLimits::default()
            }
        ),
        Err(NumericCodecError::ResourceLimit {
            resource: "total entries",
            ..
        })
    ));
    Ok(())
}

fn assert_unsigned_range_matches(
    field: NumericField<'_>,
    entries: &[NumericEntry],
    lower: Bound<NumericValue>,
    upper: Bound<NumericValue>,
) -> TestResult {
    let actual = field.range_docids(lower, upper)?.as_slice().to_vec();
    let mut expected = entries
        .iter()
        .filter(|entry| {
            let NumericValue::U64(value) = entry.value() else {
                return false;
            };
            let lower_matches = match lower {
                Bound::Unbounded => true,
                Bound::Included(NumericValue::U64(bound)) => value >= bound,
                Bound::Excluded(NumericValue::U64(bound)) => value > bound,
                Bound::Included(_) | Bound::Excluded(_) => false,
            };
            let upper_matches = match upper {
                Bound::Unbounded => true,
                Bound::Included(NumericValue::U64(bound)) => value <= bound,
                Bound::Excluded(NumericValue::U64(bound)) => value < bound,
                Bound::Included(_) | Bound::Excluded(_) => false,
            };
            lower_matches && upper_matches
        })
        .map(|entry| entry.docid())
        .collect::<Vec<_>>();
    expected.sort_unstable();
    assert_eq!(actual, expected, "lower={lower:?} upper={upper:?}");
    Ok(())
}

#[test]
fn numeric_randomized_unsigned_ranges_match_brute_force() -> TestResult {
    const SEED: u64 = 0x7f4a_7c15_d1b5_4a32;
    const BOUNDARIES: [u64; 6] = [
        0,
        1,
        (1_u64 << 63) - 1,
        1_u64 << 63,
        (1_u64 << 63) + 1,
        u64::MAX,
    ];
    init_replay_tracing();
    tracing::info!(
        codec = "NUMERIC_U64",
        seed = SEED,
        replay = "numeric_randomized_unsigned_ranges_match_brute_force",
        "starting randomized unsigned range suite"
    );
    let mut state = SEED;
    let mut values = BOUNDARIES.to_vec();
    for _ in 0..257 {
        values.push((u64::from(random_u32(&mut state)) << 32) | u64::from(random_u32(&mut state)));
    }
    let entries = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            NumericEntry::u64(
                value,
                20_000_u32 + u32::try_from(index).expect("small test index"),
            )
        })
        .collect::<Vec<_>>();
    let fields = [
        NumericFieldInput::new(0, &[]),
        NumericFieldInput::new(1, &entries),
    ];
    let encoded = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        20_000,
        20_000 + u64::try_from(entries.len())?,
        &fields,
    )?;
    let section = encoded.section()?;
    let field = section.field(1).ok_or("unsigned NUMERIC field")?;

    for &left in &BOUNDARIES {
        for &right in &BOUNDARIES {
            let lower_bounds = [
                Bound::Unbounded,
                Bound::Included(NumericValue::U64(left)),
                Bound::Excluded(NumericValue::U64(left)),
            ];
            let upper_bounds = [
                Bound::Unbounded,
                Bound::Included(NumericValue::U64(right)),
                Bound::Excluded(NumericValue::U64(right)),
            ];
            for lower in lower_bounds {
                for upper in upper_bounds {
                    assert_unsigned_range_matches(field, &entries, lower, upper)?;
                }
            }
        }
    }
    for query in 0..128 {
        let left = (u64::from(random_u32(&mut state)) << 32) | u64::from(random_u32(&mut state));
        let right = (u64::from(random_u32(&mut state)) << 32) | u64::from(random_u32(&mut state));
        let lower = match query % 3 {
            0 => Bound::Unbounded,
            1 => Bound::Included(NumericValue::U64(left)),
            _ => Bound::Excluded(NumericValue::U64(left)),
        };
        let upper = match (query / 3) % 3 {
            0 => Bound::Unbounded,
            1 => Bound::Included(NumericValue::U64(right)),
            _ => Bound::Excluded(NumericValue::U64(right)),
        };
        assert_unsigned_range_matches(field, &entries, lower, upper)?;
    }
    Ok(())
}

#[test]
fn numeric_randomized_ranges_match_brute_force() -> TestResult {
    const DOCS: usize = 257;
    const SEED: u64 = 0xc3a5_c85c_97cb_3127;

    init_replay_tracing();
    tracing::info!(
        codec = "NUMERIC_I64",
        seed = SEED,
        replay = "numeric_randomized_ranges_match_brute_force",
        "starting randomized signed range suite"
    );
    let mut state = SEED;
    let mut signed = Vec::with_capacity(DOCS);
    for ordinal in 0..DOCS {
        let high = u64::from(random_u32(&mut state));
        let low = u64::from(random_u32(&mut state));
        let value = i64::from_le_bytes(((high << 32) | low).to_le_bytes());
        let docid = 10_000_u32
            .checked_add(u32::try_from(ordinal)?)
            .ok_or("test docid overflow")?;
        signed.push(NumericEntry::i64(value, docid));
    }
    let fields = [
        NumericFieldInput::new(0, &signed),
        NumericFieldInput::new(1, &[]),
    ];
    let encoded = EncodedNumericSection::encode(
        NUMERIC_TEST_SCHEMA,
        10_000,
        10_000 + u64::try_from(DOCS)?,
        &fields,
    )?;
    let field = encoded.section()?.field(0).ok_or("signed NUMERIC field")?;
    for query in 0..256 {
        let left = i64::from_le_bytes(
            ((u64::from(random_u32(&mut state)) << 32) | u64::from(random_u32(&mut state)))
                .to_le_bytes(),
        );
        let right = i64::from_le_bytes(
            ((u64::from(random_u32(&mut state)) << 32) | u64::from(random_u32(&mut state)))
                .to_le_bytes(),
        );
        let lower = match query % 3 {
            0 => Bound::Unbounded,
            1 => Bound::Included(NumericValue::I64(left)),
            _ => Bound::Excluded(NumericValue::I64(left)),
        };
        let upper = match (query / 3) % 3 {
            0 => Bound::Unbounded,
            1 => Bound::Included(NumericValue::I64(right)),
            _ => Bound::Excluded(NumericValue::I64(right)),
        };
        let actual = field.range_docids(lower, upper)?.as_slice().to_vec();
        let mut expected = signed
            .iter()
            .filter(|entry| {
                let NumericValue::I64(value) = entry.value() else {
                    return false;
                };
                let lower_matches = match lower {
                    Bound::Unbounded => true,
                    Bound::Included(NumericValue::I64(bound)) => value >= bound,
                    Bound::Excluded(NumericValue::I64(bound)) => value > bound,
                    Bound::Included(_) | Bound::Excluded(_) => false,
                };
                let upper_matches = match upper {
                    Bound::Unbounded => true,
                    Bound::Included(NumericValue::I64(bound)) => value <= bound,
                    Bound::Excluded(NumericValue::I64(bound)) => value < bound,
                    Bound::Included(_) | Bound::Excluded(_) => false,
                };
                lower_matches && upper_matches
            })
            .map(|entry| entry.docid())
            .collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(actual, expected, "seed={SEED:#x} range query {query}");
    }
    Ok(())
}

#[test]
fn numeric_rejects_type_range_order_and_resource_corruption() -> TestResult {
    let signed = [NumericEntry::i64(-1, 100), NumericEntry::i64(1, 101)];
    let unsigned = [NumericEntry::u64(7, 102)];
    let fields = [
        NumericFieldInput::new(0, &signed),
        NumericFieldInput::new(1, &unsigned),
    ];
    let encoded = EncodedNumericSection::encode(NUMERIC_TEST_SCHEMA, 100, 103, &fields)?;
    assert!(matches!(
        EncodedNumericSection::encode(
            NUMERIC_TEST_SCHEMA,
            100,
            103,
            &[
                NumericFieldInput::new(0, &[NumericEntry::u64(1, 100)]),
                NumericFieldInput::new(1, &[]),
            ]
        ),
        Err(NumericCodecError::ValueTypeMismatch { field_ord: 0, .. })
    ));
    assert!(matches!(
        EncodedNumericSection::encode(
            NUMERIC_TEST_SCHEMA,
            100,
            103,
            &[
                NumericFieldInput::new(0, &[NumericEntry::i64(0, 99)]),
                NumericFieldInput::new(1, &[]),
            ]
        ),
        Err(NumericCodecError::DocIdOutOfRange { docid: 99, .. })
    ));
    assert!(matches!(
        EncodedNumericSection::encode_with_limits(
            NUMERIC_TEST_SCHEMA,
            100,
            103,
            &fields,
            NumericLimits {
                max_total_entries: 2,
                ..NumericLimits::default()
            }
        ),
        Err(NumericCodecError::ResourceLimit {
            resource: "total entries",
            ..
        })
    ));
    assert!(matches!(
        NumericSection::parse_with_limits(
            encoded.as_bytes(),
            NUMERIC_TEST_SCHEMA,
            100,
            103,
            NumericLimits {
                max_fields: 1,
                ..NumericLimits::default()
            }
        ),
        Err(NumericCodecError::ResourceLimit {
            resource: "indexed-or-fast numeric fields",
            ..
        })
    ));
    assert!(matches!(
        NumericSection::parse_with_limits(
            encoded.as_bytes(),
            NUMERIC_TEST_SCHEMA,
            100,
            103,
            NumericLimits {
                max_entries_per_field: 1,
                ..NumericLimits::default()
            }
        ),
        Err(NumericCodecError::ResourceLimit {
            resource: "entries per field",
            ..
        })
    ));
    assert!(matches!(
        NumericSection::parse_with_limits(
            encoded.as_bytes(),
            NUMERIC_TEST_SCHEMA,
            100,
            103,
            NumericLimits {
                max_total_entries: 2,
                ..NumericLimits::default()
            }
        ),
        Err(NumericCodecError::ResourceLimit {
            resource: "total entries",
            ..
        })
    ));
    assert!(matches!(
        NumericSection::parse_with_limits(
            encoded.as_bytes(),
            NUMERIC_TEST_SCHEMA,
            100,
            103,
            NumericLimits {
                max_docid_span: 2,
                ..NumericLimits::default()
            }
        ),
        Err(NumericCodecError::ResourceLimit {
            resource: "docid span",
            ..
        })
    ));
    assert!(matches!(
        NumericSection::parse_with_limits(
            encoded.as_bytes(),
            NUMERIC_TEST_SCHEMA,
            100,
            103,
            NumericLimits {
                max_section_bytes: u64::try_from(encoded.as_bytes().len().saturating_sub(1))?,
                ..NumericLimits::default()
            }
        ),
        Err(NumericCodecError::ResourceLimit {
            resource: "section bytes",
            ..
        })
    ));

    let mut wrong_field = encoded.as_bytes().to_vec();
    wrong_field[..2].copy_from_slice(&9_u16.to_le_bytes());
    assert!(matches!(
        NumericSection::parse(&wrong_field, NUMERIC_TEST_SCHEMA, 100, 103),
        Err(NumericCodecError::UnexpectedField { index: 0, .. })
    ));
    let mut bad_docid = encoded.as_bytes().to_vec();
    let first_docid = NUMERIC_FIELD_HEADER_LEN + std::mem::size_of::<u64>();
    bad_docid[first_docid..first_docid + 4].copy_from_slice(&99_u32.to_le_bytes());
    assert!(matches!(
        NumericSection::parse(&bad_docid, NUMERIC_TEST_SCHEMA, 100, 103),
        Err(NumericCodecError::DocIdOutOfRange { docid: 99, .. })
    ));
    let mut duplicate_pair = encoded.as_bytes().to_vec();
    let first = NUMERIC_FIELD_HEADER_LEN..NUMERIC_FIELD_HEADER_LEN + NUMERIC_PAIR_LEN;
    let second_start = NUMERIC_FIELD_HEADER_LEN + NUMERIC_PAIR_LEN;
    duplicate_pair.copy_within(first, second_start);
    assert!(matches!(
        NumericSection::parse(&duplicate_pair, NUMERIC_TEST_SCHEMA, 100, 103),
        Err(NumericCodecError::NonCanonicalPairOrder {
            field_ord: 0,
            index: 1
        })
    ));
    let mut trailing = encoded.as_bytes().to_vec();
    trailing.push(0);
    assert!(matches!(
        NumericSection::parse(&trailing, NUMERIC_TEST_SCHEMA, 100, 103),
        Err(NumericCodecError::TrailingBytes { .. })
    ));
    for length in 0..encoded.as_bytes().len() {
        assert!(matches!(
            NumericSection::parse(&encoded.as_bytes()[..length], NUMERIC_TEST_SCHEMA, 100, 103),
            Err(NumericCodecError::Truncated { .. })
        ));
    }
    Ok(())
}

#[test]
fn arbitrary_numeric_bytes_never_panic() {
    const SEED: u64 = 0x9f4c_72a1_15bd_e603;

    init_replay_tracing();
    let mut state = SEED;
    for case in 0..1_000 {
        tracing::info!(
            codec = "NUMERIC",
            seed = SEED,
            case,
            replay = "arbitrary_numeric_bytes_never_panic",
            "starting hostile-byte case"
        );
        let length = usize::try_from(random_u32(&mut state) % 257).unwrap_or(0);
        let bytes = (0..length)
            .map(|_| random_u32(&mut state).to_le_bytes()[0])
            .collect::<Vec<_>>();
        let parsed = std::panic::catch_unwind(|| {
            NumericSection::parse(&bytes, NUMERIC_TEST_SCHEMA, 100, 132)
        });
        assert!(
            parsed.is_ok(),
            "seed={SEED:#x} NUMERIC parser panic case={case} len={length}"
        );
        if let Ok(Ok(section)) = parsed {
            for field in section.fields() {
                let ranged = std::panic::catch_unwind(|| {
                    if field.is_signed() {
                        field.range_docids(
                            Bound::Included(NumericValue::I64(-7)),
                            Bound::Excluded(NumericValue::I64(9)),
                        )
                    } else {
                        field.range_docids(
                            Bound::Included(NumericValue::U64(7)),
                            Bound::Excluded(NumericValue::U64(9)),
                        )
                    }
                });
                assert!(
                    ranged.is_ok(),
                    "seed={SEED:#x} NUMERIC range panic case={case}"
                );
            }
        }
    }
}

#[test]
fn arbitrary_doclen_and_stats_bytes_never_panic() {
    const SEED: u64 = 0x6d8f_3a51_c0de_f00d;

    init_replay_tracing();
    let mut state = SEED;
    for case in 0..1_000 {
        tracing::info!(
            codec = "DOCLEN_STATS",
            seed = SEED,
            case,
            replay = "arbitrary_doclen_and_stats_bytes_never_panic",
            "starting hostile-byte case"
        );
        let length = usize::try_from(random_u32(&mut state) % 257).unwrap_or(0);
        let bytes: Vec<u8> = (0..length)
            .map(|_| random_u32(&mut state).to_le_bytes()[0])
            .collect();
        let span = u64::from(random_u32(&mut state) % 32);
        let doclen =
            std::panic::catch_unwind(|| DocLenSection::parse(&bytes, 100, 100 + span, &[1, 2]));
        assert!(
            doclen.is_ok(),
            "seed={SEED:#x} DOCLEN parser panic case={case} len={length}"
        );
        let stats_parse = std::panic::catch_unwind(|| StatsSection::parse(&bytes, &[1, 2], 3));
        assert!(
            stats_parse.is_ok(),
            "seed={SEED:#x} STATS parser panic case={case} len={length}"
        );
    }
}

#[test]
fn arbitrary_bytes_never_panic_parser_or_cursor() {
    const SEED: u64 = 0x91e1_0da5_c79e_7b1d;

    init_replay_tracing();
    let mut state = SEED;
    for case in 0..1_000 {
        tracing::info!(
            codec = "POSTINGS",
            seed = SEED,
            case,
            replay = "arbitrary_bytes_never_panic_parser_or_cursor",
            "starting hostile-byte case"
        );
        let length = usize::try_from(random_u32(&mut state) % 257).unwrap_or(0);
        let bytes: Vec<u8> = (0..length)
            .map(|_| random_u32(&mut state).to_le_bytes()[0])
            .collect();
        let expected = random_u32(&mut state) % 300;
        let parsed = std::panic::catch_unwind(|| PostingList::parse(&bytes, expected));
        assert!(
            parsed.is_ok(),
            "seed={SEED:#x} parser panic case={case} len={length}"
        );
        if let Ok(Ok(list)) = parsed {
            let decoded = std::panic::catch_unwind(|| list.decode_all());
            assert!(
                decoded.is_ok(),
                "seed={SEED:#x} cursor panic case={case} len={length}"
            );
        }
    }
}
