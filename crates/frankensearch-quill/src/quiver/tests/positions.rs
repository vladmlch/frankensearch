use crate::quiver::position_cursor::PositionByteReader;

use super::*;

#[test]
fn positions_wire_golden_and_full_u32_vint_domain() -> TestResult {
    let postings = [Posting::new(10, 3), Posting::new(20, 2)];
    let posting_bytes = EncodedPostingList::encode(&postings)?;
    let posting_list = posting_bytes.posting_list()?;
    let encoded = EncodedPositionList::encode(&postings, &[0, 0, 7, 130, 130])?;
    assert_eq!(
        encoded.as_bytes(),
        [1, 0, 0, 0, 0, 0, 0, 0, 7, 0x82, 0x01, 0]
    );
    let positions = encoded.position_list(&posting_list)?;
    assert_eq!(collect_positions(&positions, 0)?, [0, 0, 7]);
    assert_eq!(collect_positions(&positions, 1)?, [130, 130]);

    let boundary_values = [0, 127, 128, 16_383, 16_384, 65_535, 65_536, u32::MAX];
    let boundary_postings: Vec<Posting> = boundary_values
        .iter()
        .enumerate()
        .map(|(index, _)| Ok(Posting::new(100 + u32::try_from(index)?, 1)))
        .collect::<Result<_, std::num::TryFromIntError>>()?;
    let boundary_posting_bytes = EncodedPostingList::encode(&boundary_postings)?;
    let boundary_posting_list = boundary_posting_bytes.posting_list()?;
    let boundary_encoded = EncodedPositionList::encode(&boundary_postings, &boundary_values)?;
    let boundary_list = boundary_encoded.position_list(&boundary_posting_list)?;
    assert_eq!(
        boundary_list.block_bytes(0),
        Some(
            &[
                0x00, 0x7f, 0x80, 0x01, 0xff, 0x7f, 0x80, 0x80, 0x01, 0xff, 0xff, 0x03, 0x80, 0x80,
                0x04, 0xff, 0xff, 0xff, 0xff, 0x0f,
            ][..]
        )
    );
    for (ordinal, expected) in boundary_values.iter().copied().enumerate() {
        assert_eq!(
            collect_positions(&boundary_list, u32::try_from(ordinal)?)?,
            [expected]
        );
    }
    Ok(())
}

#[test]
fn positions_empty_count_alignment_and_resource_limits() -> TestResult {
    let empty_postings = EncodedPostingList::encode(&[])?;
    let empty_posting_list = empty_postings.posting_list()?;
    let empty = EncodedPositionList::encode(&[], &[])?;
    assert_eq!(empty.as_bytes(), [0, 0, 0, 0]);
    let empty_list = empty.position_list(&empty_posting_list)?;
    assert_eq!(empty_list.doc_freq(), 0);
    assert_eq!(empty_list.total_positions(), 0);
    assert_eq!(empty_list.block_count(), 0);
    assert!(empty_list.cursor()?.positions()?.is_none());

    let postings = [Posting::new(1, 2)];
    assert!(matches!(
        EncodedPositionList::encode(&postings, &[7]),
        Err(PositionCodecError::PositionCountMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        EncodedPositionList::encode(&postings, &[7, 8, 9]),
        Err(PositionCodecError::PositionCountMismatch {
            expected: 2,
            actual: 3
        })
    ));
    assert!(matches!(
        EncodedPositionList::encode(&postings, &[8, 7]),
        Err(PositionCodecError::NonAscendingPosition { .. })
    ));
    assert!(matches!(
        EncodedPositionList::encode_with_limits(
            &postings,
            &[7, 8],
            PositionListLimits {
                max_bytes: usize::MAX,
                max_blocks: usize::MAX,
                max_positions: 1,
            }
        ),
        Err(PositionCodecError::PositionLimitExceeded {
            limit: 1,
            actual: 2
        })
    ));
    assert!(matches!(
        EncodedPositionList::encode_with_limits(
            &postings,
            &[7, 8],
            PositionListLimits {
                max_bytes: 5,
                max_blocks: 1,
                max_positions: 2,
            }
        ),
        Err(PositionCodecError::ByteLimitExceeded { limit: 5, .. })
    ));
    assert!(matches!(
        EncodedPositionList::encode_with_limits(
            &postings,
            &[7, 8],
            PositionListLimits {
                max_bytes: usize::MAX,
                max_blocks: 0,
                max_positions: 2,
            }
        ),
        Err(PositionCodecError::BlockLimitExceeded {
            limit: 0,
            actual: 1
        })
    ));

    let posting_bytes = EncodedPostingList::encode(&postings)?;
    let posting_list = posting_bytes.posting_list()?;
    let encoded = EncodedPositionList::encode(&postings, &[7, 8])?;
    assert!(matches!(
        PositionList::parse_with_limits(
            encoded.as_bytes(),
            &posting_list,
            PositionListLimits {
                max_bytes: encoded.as_bytes().len() - 1,
                max_blocks: 1,
                max_positions: 2,
            }
        ),
        Err(PositionCodecError::ByteLimitExceeded { .. })
    ));
    assert!(matches!(
        PositionList::parse_with_limits(
            encoded.as_bytes(),
            &posting_list,
            PositionListLimits {
                max_bytes: encoded.as_bytes().len(),
                max_blocks: 0,
                max_positions: 2,
            }
        ),
        Err(PositionCodecError::BlockLimitExceeded {
            limit: 0,
            actual: 1
        })
    ));
    assert!(matches!(
        PositionList::parse_with_limits(
            encoded.as_bytes(),
            &posting_list,
            PositionListLimits {
                max_bytes: encoded.as_bytes().len(),
                max_blocks: 1,
                max_positions: 1,
            }
        ),
        Err(PositionCodecError::PositionLimitExceeded {
            limit: 1,
            actual: 2
        })
    ));
    Ok(())
}

#[test]
fn positions_fresh_blocks_cover_exact_fit_and_oversized_singleton() -> TestResult {
    let exact_postings = [Posting::new(1, 4_096), Posting::new(2, 1)];
    let exact_posting_bytes = EncodedPostingList::encode(&exact_postings)?;
    let exact_posting_list = exact_posting_bytes.posting_list()?;
    let exact_values = zero_positions(&exact_postings)?;
    let exact = EncodedPositionList::encode(&exact_postings, &exact_values)?;
    let exact_list = exact.position_list(&exact_posting_list)?;
    assert_eq!(exact_list.block_count(), 2);
    assert_eq!(exact_list.blocks()[0].byte_len(), 4_096);
    assert_eq!(exact_list.blocks()[0].posting_count(), 1);
    assert_eq!(exact_list.blocks()[1].byte_len(), 1);
    assert_eq!(exact_list.blocks()[1].base_posting_ordinal(), 1);

    let oversized_postings = [Posting::new(10, 4_097), Posting::new(20, 1)];
    let oversized_posting_bytes = EncodedPostingList::encode(&oversized_postings)?;
    let oversized_posting_list = oversized_posting_bytes.posting_list()?;
    let oversized_values = zero_positions(&oversized_postings)?;
    let oversized = EncodedPositionList::encode(&oversized_postings, &oversized_values)?;
    let oversized_list = oversized.position_list(&oversized_posting_list)?;
    assert_eq!(oversized_list.block_count(), 2);
    assert_eq!(oversized_list.blocks()[0].byte_len(), 4_097);
    assert_eq!(oversized_list.blocks()[0].posting_count(), 1);
    assert_eq!(collect_positions(&oversized_list, 0)?.len(), 4_097);

    let invalid_postings = [Posting::new(100, 3_000), Posting::new(200, 3_000)];
    let invalid_posting_bytes = EncodedPostingList::encode(&invalid_postings)?;
    let invalid_posting_list = invalid_posting_bytes.posting_list()?;
    let oversized_multi = raw_positions(&[(0, 0)], &vec![0; 6_000])?;
    assert!(matches!(
        PositionList::parse(&oversized_multi, &invalid_posting_list),
        Err(PositionCodecError::OversizedMultiDocumentBlock {
            block_index: 0,
            posting_count: 2,
            byte_len: 6_000
        })
    ));
    Ok(())
}

#[test]
fn randomized_positions_roundtrip_duplicates_and_large_u32_values() -> TestResult {
    const BASE_SEED: u64 = 0x243f_6a88_85a3_08d3;

    init_replay_tracing();
    for case in 0_u64..32 {
        let seed = BASE_SEED ^ case;
        tracing::info!(
            codec = "POSITIONS",
            seed,
            case,
            replay = "randomized_positions_roundtrip_duplicates_and_large_u32_values",
            "starting randomized positions case"
        );
        let mut state = seed;
        let count = if case == 0 {
            0
        } else {
            1 + usize::try_from(random_u32(&mut state) % 96)?
        };
        let mut postings = Vec::with_capacity(count);
        let mut expected_runs = Vec::with_capacity(count);
        let mut flat = Vec::new();
        let mut doc_id = random_u32(&mut state) % 100;
        for ordinal in 0..count {
            if ordinal != 0 {
                doc_id = doc_id
                    .checked_add(1 + random_u32(&mut state) % 9)
                    .ok_or("random position fixture docid overflow")?;
            }
            let freq = 1 + random_u32(&mut state) % 8;
            postings.push(Posting::new(doc_id, freq));
            let mut run = Vec::with_capacity(usize::try_from(freq)?);
            let mut position = if ordinal % 5 == 0 {
                65_536 + random_u32(&mut state) % 100_000
            } else {
                random_u32(&mut state) % 10_000
            };
            for index in 0..freq {
                if index != 0 {
                    position = position
                        .checked_add(random_u32(&mut state) % 7)
                        .ok_or("random position fixture overflow")?;
                }
                run.push(position);
                flat.push(position);
            }
            expected_runs.push(run);
        }

        let posting_bytes = EncodedPostingList::encode(&postings)?;
        let posting_list = posting_bytes.posting_list()?;
        let encoded = EncodedPositionList::encode(&postings, &flat)?;
        assert_eq!(encoded, EncodedPositionList::encode(&postings, &flat)?);
        let list = encoded.position_list(&posting_list)?;
        assert_eq!(list.doc_freq(), u32::try_from(count)?);
        assert_eq!(list.total_positions(), u64::try_from(flat.len())?);
        for (ordinal, expected) in expected_runs.iter().enumerate() {
            assert_eq!(
                collect_positions(&list, u32::try_from(ordinal)?)?,
                *expected,
                "seed={seed:#x} case={case} ordinal={ordinal}"
            );
        }
    }
    Ok(())
}

#[test]
fn position_cursor_reuses_state_across_posting_and_position_blocks() -> TestResult {
    let mut postings = Vec::with_capacity(270);
    let mut expected_runs = Vec::with_capacity(270);
    let mut flat = Vec::new();
    for ordinal in 0_u32..270 {
        postings.push(Posting::new(10 + ordinal * 3, 24));
        let base = 70_000 + ordinal * 100;
        let run: Vec<u32> = (0_u32..24).map(|index| base + index / 2).collect();
        flat.extend_from_slice(&run);
        expected_runs.push(run);
    }
    let posting_bytes = EncodedPostingList::encode(&postings)?;
    let posting_list = posting_bytes.posting_list()?;
    assert_eq!(posting_list.blocks()[1].base_posting_ordinal, 128);
    let encoded = EncodedPositionList::encode(&postings, &flat)?;
    let positions = encoded.position_list(&posting_list)?;
    assert!(positions.block_count() >= 2);
    assert_ne!(positions.blocks()[1].base_posting_ordinal(), 128);

    let mut cursor = positions.cursor()?;
    assert_eq!(cursor.posting_ordinal(), Some(0));
    assert_eq!(
        cursor
            .positions()?
            .ok_or("position cursor start")?
            .collect::<Result<Vec<_>, _>>()?,
        expected_runs[0]
    );
    assert_eq!(cursor.advance(postings[127].doc_id)?, Some(postings[127]));
    assert_eq!(cursor.posting_ordinal(), Some(127));
    assert_eq!(
        cursor
            .positions()?
            .ok_or("position cursor 127")?
            .collect::<Result<Vec<_>, _>>()?,
        expected_runs[127]
    );
    let mut owned_decode = Vec::new();
    assert!(cursor.decode_current_positions_into(&mut owned_decode)?);
    assert_eq!(owned_decode, expected_runs[127]);
    assert_eq!(cursor.next()?, Some(postings[128]));
    assert_eq!(cursor.posting_ordinal(), Some(128));
    assert_eq!(
        cursor
            .positions()?
            .ok_or("position cursor 128")?
            .collect::<Result<Vec<_>, _>>()?,
        expected_runs[128]
    );

    let position_seam = usize::try_from(positions.blocks()[1].base_posting_ordinal())?;
    let before_seam = position_seam
        .checked_sub(1)
        .ok_or("position seam at zero")?;
    assert_eq!(
        cursor.advance(postings[before_seam].doc_id)?,
        Some(postings[before_seam])
    );
    assert!(cursor.decode_current_positions_into(&mut owned_decode)?);
    assert_eq!(owned_decode, expected_runs[before_seam]);
    assert_eq!(cursor.next()?, Some(postings[position_seam]));
    assert_eq!(
        cursor
            .positions()?
            .ok_or("sequential position seam")?
            .collect::<Result<Vec<_>, _>>()?,
        expected_runs[position_seam]
    );
    let after_seam = position_seam + 7;
    assert_eq!(
        cursor.advance(postings[after_seam].doc_id)?,
        Some(postings[after_seam])
    );
    assert_eq!(
        cursor
            .positions()?
            .ok_or("position seam in-block advance")?
            .collect::<Result<Vec<_>, _>>()?,
        expected_runs[after_seam]
    );
    assert_eq!(cursor.advance(u32::MAX)?, None);
    assert_eq!(cursor.next()?, None);
    assert!(cursor.positions()?.is_none());

    let maximum_postings = [Posting::new(0, 2), Posting::new(u32::MAX, 1)];
    let maximum_posting_bytes = EncodedPostingList::encode(&maximum_postings)?;
    let maximum_posting_list = maximum_posting_bytes.posting_list()?;
    let maximum_encoded = EncodedPositionList::encode(&maximum_postings, &[1, 1, u32::MAX])?;
    let maximum_positions = maximum_encoded.position_list(&maximum_posting_list)?;
    let mut maximum_cursor = maximum_positions.cursor()?;
    assert_eq!(maximum_cursor.advance(u32::MAX)?, Some(maximum_postings[1]));
    assert_eq!(
        maximum_cursor
            .positions()?
            .ok_or("u32 max position cursor")?
            .collect::<Result<Vec<_>, _>>()?,
        [u32::MAX]
    );
    assert!(maximum_cursor.decode_current_positions_into(&mut owned_decode)?);
    assert_eq!(owned_decode, [u32::MAX]);
    assert_eq!(maximum_cursor.next()?, None);
    Ok(())
}

#[test]
fn position_cursor_owned_decode_consumes_each_run_once_and_preserves_replay() -> TestResult {
    let postings = [
        Posting::new(10, 2),
        Posting::new(20, 3),
        Posting::new(30, 1),
    ];
    let runs = [vec![1, 4], vec![7, 7, 130], vec![u32::MAX]];
    let flat = runs.iter().flatten().copied().collect::<Vec<_>>();
    let posting_bytes = EncodedPostingList::encode(&postings)?;
    let posting_list = posting_bytes.posting_list()?;
    let encoded = EncodedPositionList::encode(&postings, &flat)?;
    let positions = encoded.position_list(&posting_list)?;
    assert_eq!(positions.block_count(), 1);

    let mut cursor = positions.cursor()?;
    let initial_offset = cursor.position_reader.position();
    let mut decoded = vec![999];
    assert!(cursor.decode_current_positions_into(&mut decoded)?);
    assert_eq!(decoded, runs[0]);
    assert!(cursor.current_positions_consumed);
    let second_run_offset = cursor.position_reader.position();
    assert!(second_run_offset > initial_offset);

    assert_eq!(
        cursor
            .positions()?
            .ok_or("replay first consumed run")?
            .collect::<Result<Vec<_>, _>>()?,
        runs[0]
    );
    assert_eq!(cursor.position_reader.position(), second_run_offset);
    assert!(cursor.decode_current_positions_into(&mut decoded)?);
    assert_eq!(decoded, runs[0]);
    assert_eq!(cursor.position_reader.position(), second_run_offset);

    assert_eq!(cursor.next()?, Some(postings[1]));
    assert!(!cursor.current_positions_consumed);
    assert_eq!(cursor.current_position_offset, second_run_offset);
    assert_eq!(cursor.position_reader.position(), second_run_offset);
    assert!(cursor.decode_current_positions_into(&mut decoded)?);
    assert_eq!(decoded, runs[1]);
    let third_run_offset = cursor.position_reader.position();
    assert!(third_run_offset > second_run_offset);

    assert_eq!(cursor.advance(postings[2].doc_id)?, Some(postings[2]));
    assert!(!cursor.current_positions_consumed);
    assert_eq!(cursor.current_position_offset, third_run_offset);
    assert_eq!(cursor.position_reader.position(), third_run_offset);
    assert!(cursor.decode_current_positions_into(&mut decoded)?);
    assert_eq!(decoded, runs[2]);
    let exhausted_offset = cursor.position_reader.position();
    assert_eq!(cursor.next()?, None);
    assert_eq!(cursor.position_reader.position(), exhausted_offset);
    assert!(!cursor.current_positions_consumed);
    assert!(!cursor.decode_current_positions_into(&mut decoded)?);
    assert!(decoded.is_empty());

    let mut truncated = positions.cursor()?;
    let original_offset = truncated.position_reader.position();
    truncated.position_reader.end = original_offset + 1;
    decoded.push(999);
    assert!(matches!(
        truncated.decode_current_positions_into(&mut decoded),
        Err(PositionCodecError::Truncated { .. })
    ));
    assert!(decoded.is_empty());
    assert!(!truncated.current_positions_consumed);
    assert_eq!(truncated.position_reader.position(), original_offset);

    let mut retry = positions.cursor()?;
    let valid_reader = retry.position_reader.clone();
    let valid_offset = retry.current_position_offset;
    let noncanonical = [0x81, 0x00];
    retry.position_reader = PositionByteReader::new(&noncanonical, 0, noncanonical.len())?;
    decoded.push(999);
    assert!(matches!(
        retry.decode_current_positions_into(&mut decoded),
        Err(PositionCodecError::NonCanonicalVint { domain: "u32", .. })
    ));
    assert!(decoded.is_empty());
    assert!(!retry.current_positions_consumed);
    assert_eq!(retry.position_reader.position(), 0);

    let overflowing_delta = [0xff, 0xff, 0xff, 0xff, 0x0f, 0x01];
    retry.position_reader =
        PositionByteReader::new(&overflowing_delta, 0, overflowing_delta.len())?;
    assert!(matches!(
        retry.decode_current_positions_into(&mut decoded),
        Err(PositionCodecError::PositionOverflow {
            posting_ordinal: 0,
            previous: u32::MAX,
            delta: 1,
        })
    ));
    assert!(decoded.is_empty());
    assert!(!retry.current_positions_consumed);
    assert_eq!(retry.position_reader.position(), 0);

    retry.position_reader = valid_reader;
    retry.current_position_offset = valid_offset;
    assert!(retry.decode_current_positions_into(&mut decoded)?);
    assert_eq!(decoded, runs[0], "a failed decode must remain retryable");
    Ok(())
}

#[test]
fn position_cursor_randomized_owned_decode_replay_and_seek_match_rows() -> TestResult {
    const BASE_SEED: u64 = 0x517c_c1b7_2722_0a95;
    const CASES: u64 = 8;
    const POSTING_COUNT: usize = 300;

    for case in 0..CASES {
        let seed = BASE_SEED ^ case;
        let mut state = seed;
        let mut postings = Vec::with_capacity(POSTING_COUNT);
        let mut expected_runs = Vec::with_capacity(POSTING_COUNT);
        let mut flat = Vec::new();
        let mut doc_id = random_u32(&mut state) % 100;
        for ordinal in 0..POSTING_COUNT {
            if ordinal != 0 {
                doc_id = doc_id
                    .checked_add(1 + random_u32(&mut state) % 7)
                    .ok_or("random cursor fixture docid overflow")?;
            }
            let frequency = 8 + random_u32(&mut state) % 25;
            postings.push(Posting::new(doc_id, frequency));
            let mut position = random_u32(&mut state) % 1_000;
            let mut run = Vec::with_capacity(usize::try_from(frequency)?);
            for index in 0..frequency {
                if index != 0 {
                    position = position
                        .checked_add(random_u32(&mut state) % 6)
                        .ok_or("random cursor fixture position overflow")?;
                }
                run.push(position);
                flat.push(position);
            }
            expected_runs.push(run);
        }

        let posting_bytes = EncodedPostingList::encode(&postings)?;
        let posting_list = posting_bytes.posting_list()?;
        let encoded = EncodedPositionList::encode(&postings, &flat)?;
        let positions = encoded.position_list(&posting_list)?;
        assert!(posting_list.block_count() >= 3, "seed={seed:#x}");
        assert!(positions.block_count() >= 2, "seed={seed:#x}");

        let mut cursor = positions.cursor()?;
        let mut decoded = Vec::new();
        while let Some(posting_ordinal) = cursor.posting_ordinal() {
            let ordinal = usize::try_from(posting_ordinal)?;
            assert_eq!(cursor.current(), Some(postings[ordinal]), "seed={seed:#x}");
            assert_eq!(
                cursor
                    .positions()?
                    .ok_or("random cursor current positions")?
                    .collect::<Result<Vec<_>, _>>()?,
                expected_runs[ordinal],
                "immutable replay seed={seed:#x} ordinal={ordinal}",
            );

            match random_u32(&mut state) % 3 {
                0 => {}
                1 => {
                    assert!(cursor.decode_current_positions_into(&mut decoded)?);
                    assert_eq!(decoded, expected_runs[ordinal]);
                }
                _ => {
                    assert!(cursor.decode_current_positions_into(&mut decoded)?);
                    assert_eq!(decoded, expected_runs[ordinal]);
                    let reader_offset = cursor.position_reader.position();
                    assert!(cursor.decode_current_positions_into(&mut decoded)?);
                    assert_eq!(decoded, expected_runs[ordinal]);
                    assert_eq!(cursor.position_reader.position(), reader_offset);
                }
            }

            let remaining = POSTING_COUNT - ordinal;
            if remaining == 1 {
                assert_eq!(cursor.next()?, None);
            } else if random_u32(&mut state) & 1 == 0 {
                assert_eq!(cursor.next()?, Some(postings[ordinal + 1]));
            } else {
                let maximum_skip = (remaining - 1).min(17);
                let skip = 1 + usize::try_from(random_u32(&mut state))? % maximum_skip;
                let target = ordinal + skip;
                assert_eq!(
                    cursor.advance(postings[target].doc_id)?,
                    Some(postings[target]),
                    "seek seed={seed:#x} from={ordinal} to={target}",
                );
            }
        }
        assert!(cursor.positions()?.is_none(), "seed={seed:#x}");
        assert!(!cursor.decode_current_positions_into(&mut decoded)?);
        assert!(decoded.is_empty());
    }
    Ok(())
}

#[test]
fn positions_q1_concat_preserves_payloads_and_is_associative() -> TestResult {
    let left_postings = sparse_postings(100, 100);
    let right_postings = sparse_postings(300, 100_000);
    let left_posting_bytes = EncodedPostingList::encode(&left_postings)?;
    let right_posting_bytes = EncodedPostingList::encode(&right_postings)?;
    let left_posting_list = left_posting_bytes.posting_list()?;
    let right_posting_list = right_posting_bytes.posting_list()?;
    let left_values = zero_positions(&left_postings)?;
    let right_values = zero_positions(&right_postings)?;
    let left_encoded = EncodedPositionList::encode(&left_postings, &left_values)?;
    let right_encoded = EncodedPositionList::encode(&right_postings, &right_values)?;
    let left = left_encoded.position_list(&left_posting_list)?;
    let right = right_encoded.position_list(&right_posting_list)?;

    let merged_encoded = EncodedPositionList::concatenate(&[&left, &right])?;
    let expected_blocks = left.block_count() + right.block_count();
    assert!(matches!(
        EncodedPositionList::concatenate_with_limits(
            &[&left, &right],
            PositionListLimits {
                max_bytes: usize::MAX,
                max_blocks: left.block_count(),
                max_positions: u64::MAX,
            }
        ),
        Err(PositionCodecError::BlockLimitExceeded { actual, .. })
            if actual == left.block_count() + 1
    ));
    assert!(matches!(
        EncodedPositionList::concatenate_with_limits(
            &[&left, &right],
            PositionListLimits {
                max_bytes: usize::MAX,
                max_blocks: expected_blocks,
                max_positions: merged_encoded.total_positions() - 1,
            }
        ),
        Err(PositionCodecError::PositionLimitExceeded { .. })
    ));
    assert!(matches!(
        EncodedPositionList::concatenate_with_limits(
            &[&left, &right],
            PositionListLimits {
                max_bytes: merged_encoded.as_bytes().len() - 1,
                max_blocks: expected_blocks,
                max_positions: merged_encoded.total_positions(),
            }
        ),
        Err(PositionCodecError::ByteLimitExceeded { .. })
    ));
    let mut merged_posting_bytes = left_posting_bytes.as_bytes().to_vec();
    merged_posting_bytes.extend_from_slice(right_posting_bytes.as_bytes());
    let merged_postings = PostingList::parse(&merged_posting_bytes, 400)?;
    let merged = merged_encoded.position_list(&merged_postings)?;
    assert_eq!(merged.doc_freq(), 400);
    assert_eq!(
        merged.block_count(),
        left.block_count() + right.block_count()
    );
    let mut expected_payloads = Vec::new();
    for source in [&left, &right] {
        for block_index in 0..source.block_count() {
            expected_payloads.push(
                source
                    .block_bytes(block_index)
                    .ok_or("source position block")?
                    .to_vec(),
            );
        }
    }
    let mut actual_payloads = Vec::new();
    for block_index in 0..merged.block_count() {
        actual_payloads.push(
            merged
                .block_bytes(block_index)
                .ok_or("merged position block")?
                .to_vec(),
        );
    }
    assert_eq!(actual_payloads, expected_payloads);
    assert_eq!(
        merged.blocks()[left.block_count()].base_posting_ordinal(),
        100
    );
    assert_eq!(
        collect_positions(&merged, 100)?.len(),
        usize::try_from(right_postings[0].freq)?
    );

    let middle_postings = sparse_postings(120, 100_000);
    let tail_postings = sparse_postings(180, 200_000);
    let middle_posting_bytes = EncodedPostingList::encode(&middle_postings)?;
    let tail_posting_bytes = EncodedPostingList::encode(&tail_postings)?;
    let middle_posting_list = middle_posting_bytes.posting_list()?;
    let tail_posting_list = tail_posting_bytes.posting_list()?;
    let middle_encoded =
        EncodedPositionList::encode(&middle_postings, &zero_positions(&middle_postings)?)?;
    let tail_encoded =
        EncodedPositionList::encode(&tail_postings, &zero_positions(&tail_postings)?)?;
    let middle = middle_encoded.position_list(&middle_posting_list)?;
    let tail = tail_encoded.position_list(&tail_posting_list)?;

    let direct = EncodedPositionList::concatenate(&[&left, &middle, &tail])?;
    let first_pair = EncodedPositionList::concatenate(&[&left, &middle])?;
    let mut first_pair_posting_bytes = left_posting_bytes.as_bytes().to_vec();
    first_pair_posting_bytes.extend_from_slice(middle_posting_bytes.as_bytes());
    let first_pair_postings = PostingList::parse(&first_pair_posting_bytes, 220)?;
    let first_pair_list = first_pair.position_list(&first_pair_postings)?;
    let staged = EncodedPositionList::concatenate(&[&first_pair_list, &tail])?;
    assert_eq!(direct, staged);
    assert!(matches!(
        EncodedPositionList::concatenate(&[&tail, &left]),
        Err(PositionCodecError::NonAscendingConcat { .. })
    ));
    Ok(())
}

#[test]
fn positions_parser_rejects_structured_corruption_and_never_panics() -> TestResult {
    let one_posting_bytes = EncodedPostingList::encode(&[Posting::new(10, 1)])?;
    let one_posting = one_posting_bytes.posting_list()?;
    let valid = EncodedPositionList::encode(&[Posting::new(10, 1)], &[0])?;
    for cut in 0..valid.as_bytes().len() {
        assert!(
            PositionList::parse(&valid.as_bytes()[..cut], &one_posting).is_err(),
            "cut={cut}"
        );
    }
    assert!(matches!(
        PositionList::parse(&[0, 0, 0, 0], &one_posting),
        Err(PositionCodecError::InvalidBlockCount { .. })
    ));
    assert!(matches!(
        PositionList::parse(&raw_positions(&[(1, 0)], &[0])?, &one_posting),
        Err(PositionCodecError::InvalidDirectoryStart { .. })
    ));
    assert!(matches!(
        PositionList::parse(&[1, 0, 0, 0, 0x80, 0, 0, 0], &one_posting),
        Err(PositionCodecError::NonCanonicalVint { domain: "u32", .. })
    ));
    assert!(matches!(
        PositionList::parse(&[1, 0, 0, 0, 0, 0x80, 0, 0], &one_posting),
        Err(PositionCodecError::NonCanonicalVint { domain: "u64", .. })
    ));
    assert!(matches!(
        PositionList::parse(&[1, 0, 0, 0, 0, 0, 0x80, 0], &one_posting),
        Err(PositionCodecError::NonCanonicalVint { domain: "u32", .. })
    ));
    assert!(matches!(
        PositionList::parse(
            &[1, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0x10],
            &one_posting
        ),
        Err(PositionCodecError::VintOverflow { domain: "u32", .. })
    ));

    let mut offset_overflow = vec![1, 0, 0, 0, 0];
    offset_overflow.extend_from_slice(&[0xff; 9]);
    offset_overflow.extend_from_slice(&[0x02, 0]);
    assert!(matches!(
        PositionList::parse(&offset_overflow, &one_posting),
        Err(PositionCodecError::VintOverflow { domain: "u64", .. })
    ));

    let two_postings = [Posting::new(10, 1), Posting::new(20, 1)];
    let two_posting_bytes = EncodedPostingList::encode(&two_postings)?;
    let two_posting_list = two_posting_bytes.posting_list()?;
    assert!(matches!(
        PositionList::parse(
            &raw_positions(&[(0, 0), (0, 1)], &[0, 0])?,
            &two_posting_list
        ),
        Err(PositionCodecError::NonAscendingBlockOrdinal { .. })
    ));
    assert!(matches!(
        PositionList::parse(
            &raw_positions(&[(0, 0), (1, 0)], &[0, 0])?,
            &two_posting_list
        ),
        Err(PositionCodecError::NonAscendingBlockOffset { .. })
    ));
    assert!(matches!(
        PositionList::parse(
            &raw_positions(&[(0, 0), (2, 1)], &[0, 0])?,
            &two_posting_list
        ),
        Err(PositionCodecError::BlockOrdinalOutOfRange {
            block_index: 1,
            first_posting_ordinal: 2,
            doc_freq: 2
        })
    ));
    assert!(matches!(
        PositionList::parse(
            &raw_positions(&[(0, 0), (1, 3)], &[0, 0])?,
            &two_posting_list
        ),
        Err(PositionCodecError::BlockOffsetOutOfRange { .. })
    ));
    assert!(matches!(
        PositionList::parse(
            &raw_positions(&[(0, 0), (1, 1)], &[0x80, 0x01, 0])?,
            &two_posting_list
        ),
        Err(PositionCodecError::Truncated { .. })
    ));

    let overflow_postings = [Posting::new(10, 2)];
    let overflow_posting_bytes = EncodedPostingList::encode(&overflow_postings)?;
    let overflow_posting_list = overflow_posting_bytes.posting_list()?;
    assert!(matches!(
        PositionList::parse(
            &raw_positions(&[(0, 0)], &[0xff, 0xff, 0xff, 0xff, 0x0f, 1])?,
            &overflow_posting_list
        ),
        Err(PositionCodecError::PositionOverflow { .. })
    ));
    assert!(matches!(
        PositionList::parse(&raw_positions(&[(0, 0)], &[0, 0])?, &one_posting),
        Err(PositionCodecError::TrailingBlockBytes {
            block_index: 0,
            remaining: 1
        })
    ));

    const SEED: u64 = 0x1319_8a2e_0370_7344;
    init_replay_tracing();
    let mut state = SEED;
    for len in 0..64 {
        tracing::info!(
            codec = "POSITIONS",
            seed = SEED,
            case = len,
            replay = "positions_parser_rejects_structured_corruption_and_never_panics",
            "starting hostile-byte case"
        );
        let bytes: Vec<u8> = (0..len)
            .map(|_| random_u32(&mut state).to_le_bytes()[0])
            .collect();
        let parsed = std::panic::catch_unwind(|| PositionList::parse(&bytes, &two_posting_list));
        assert!(
            parsed.is_ok(),
            "seed={SEED:#x} POSITIONS parser panic len={len}"
        );
    }
    Ok(())
}
