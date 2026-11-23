use crate::quiver::bitpack::{unpack_scalar_into, unpack_wide_into};
use crate::quiver::block_max::{
    BlockMaxByteReader, canonical_block_max_frequency_code, decode_block_for_block_max,
    vint64_length, write_vint64,
};
use crate::quiver::postings::BITMAP_KIND;

use super::super::*;
use super::*;

#[test]
fn blockmax_fresh_seal_roundtrips_all_posting_block_shapes() -> TestResult {
    for count in [0, 1, 127, 128, 129, 255, 256, 400] {
        let mut postings = sparse_postings(count, 100);
        for (ordinal, posting) in postings.iter_mut().enumerate() {
            posting.freq = match ordinal % 7 {
                0 => 1,
                1 => 254,
                2 => 255,
                3 => 256,
                4 => 65_535,
                5 => u32::MAX,
                _ => 17,
            };
        }
        let ordinary_postings = EncodedPostingList::encode(&postings)?;
        let (encoded_postings, encoded) =
            EncodedPostingList::encode_with_block_max(&postings, fixture_fieldnorm)?;
        assert_eq!(
            encoded_postings, ordinary_postings,
            "posting parity count={count}"
        );
        let posting_list = encoded_postings.posting_list()?;
        let recomputed = EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?;
        assert_eq!(encoded, recomputed, "integrated parity count={count}");
        let (repeated_postings, repeated_bounds) =
            EncodedPostingList::encode_with_block_max(&postings, fixture_fieldnorm)?;
        assert_eq!(
            encoded_postings, repeated_postings,
            "posting determinism count={count}"
        );
        assert_eq!(encoded, repeated_bounds, "bound determinism count={count}");
        let parsed = BlockMaxList::parse_with_fieldnorms(
            encoded.as_bytes(),
            &posting_list,
            fixture_fieldnorm,
        )?;
        assert_eq!(parsed.entry_count(), posting_list.block_count());
        assert_eq!(encoded.entry_count(), posting_list.block_count());
        assert_eq!(parsed.as_bytes(), encoded.as_bytes());
        for (block_index, (entry, block)) in parsed
            .entries()
            .iter()
            .zip(posting_list.blocks())
            .enumerate()
        {
            assert_eq!(entry.first_doc(), block.first_doc, "count={count}");
            assert_eq!(
                entry.block_offset(),
                u64::try_from(block.byte_offset)?,
                "count={count}"
            );
            let decoded = decode_block_for_block_max(&posting_list, block_index)?;
            assert_eq!(
                entry.max_frequency_code(),
                canonical_block_max_frequency_code(&decoded),
                "count={count}"
            );
            assert_eq!(
                entry.min_fieldnorm(),
                crate::contract::id_to_fieldnorm(entry.min_fieldnorm_id())
            );
        }
        assert_eq!(parsed.cursor().current(), parsed.entries().first().copied());
    }

    let (_, singleton) =
        EncodedPostingList::encode_with_block_max(&[Posting::new(128, 256)], |_| Some(0))?;
    assert_eq!(singleton.as_bytes(), [0x80, 0, 0, 0, 0, 0xff, 0]);
    Ok(())
}

#[test]
fn blockmax_u64_vints_are_canonical_through_the_full_domain() -> TestResult {
    for value in [
        0,
        1,
        127,
        128,
        16_383,
        16_384,
        u64::from(u32::MAX),
        u64::MAX,
    ] {
        let mut bytes = Vec::new();
        write_vint64(value, &mut bytes);
        assert_eq!(bytes.len(), vint64_length(value));
        let mut reader = BlockMaxByteReader::new(&bytes);
        assert_eq!(reader.read_vint()?, value);
        reader.finish()?;
    }
    assert_eq!(
        {
            let mut bytes = Vec::new();
            write_vint64(u64::MAX, &mut bytes);
            bytes
        },
        [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]
    );
    Ok(())
}

#[test]
fn blockmax_parser_rejects_wire_and_cross_section_corruption() -> TestResult {
    let encoded_postings = EncodedPostingList::encode(&[Posting::new(128, 7)])?;
    let posting_list = encoded_postings.posting_list()?;
    let encoded = EncodedBlockMax::encode(&posting_list, |_| Some(4))?;
    assert_eq!(encoded.as_bytes(), [0x80, 0, 0, 0, 0, 7, 4]);

    for cut in 0..encoded.as_bytes().len() {
        assert!(
            BlockMaxList::parse_with_fieldnorms(&encoded.as_bytes()[..cut], &posting_list, |_| {
                Some(4)
            },)
            .is_err(),
            "cut={cut}"
        );
    }

    let mut first_doc = encoded.as_bytes().to_vec();
    first_doc[0] ^= 1;
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(&first_doc, &posting_list, |_| Some(4)),
        Err(BlockMaxError::FirstDocMismatch { .. })
    ));

    let mut block_offset = encoded.as_bytes().to_vec();
    block_offset[4] = 1;
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(&block_offset, &posting_list, |_| Some(4)),
        Err(BlockMaxError::BlockOffsetMismatch { .. })
    ));

    let mut zero_frequency = encoded.as_bytes().to_vec();
    zero_frequency[5] = 0;
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(&zero_frequency, &posting_list, |_| Some(4)),
        Err(BlockMaxError::ZeroMaximumFrequency { .. })
    ));

    let mut wrong_frequency = encoded.as_bytes().to_vec();
    wrong_frequency[5] = 6;
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(&wrong_frequency, &posting_list, |_| Some(4)),
        Err(BlockMaxError::MaximumFrequencyMismatch { .. })
    ));

    let mut wrong_fieldnorm = encoded.as_bytes().to_vec();
    wrong_fieldnorm[6] = u8::MAX;
    let merge_only = BlockMaxConcatList::parse(&wrong_fieldnorm, &posting_list)?;
    assert_eq!(merge_only.entry_count(), 1);
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(&wrong_fieldnorm, &posting_list, |_| Some(4)),
        Err(BlockMaxError::MinimumFieldnormMismatch { .. })
    ));
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(encoded.as_bytes(), &posting_list, |_| None),
        Err(BlockMaxError::MissingFieldnorm { doc_id: 128, .. })
    ));

    let mut trailing = encoded.as_bytes().to_vec();
    trailing.push(0);
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(&trailing, &posting_list, |_| Some(4)),
        Err(BlockMaxError::TrailingBytes { remaining: 1, .. })
    ));

    let mut noncanonical = 128_u32.to_le_bytes().to_vec();
    noncanonical.extend_from_slice(&[0x80, 0x00, 7, 4]);
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(&noncanonical, &posting_list, |_| Some(4)),
        Err(BlockMaxError::NonCanonicalVint { .. })
    ));

    let mut overflow = 128_u32.to_le_bytes().to_vec();
    overflow.extend_from_slice(&[0x80; 9]);
    overflow.extend_from_slice(&[0x02, 7, 4]);
    assert!(matches!(
        BlockMaxList::parse_with_fieldnorms(&overflow, &posting_list, |_| Some(4)),
        Err(BlockMaxError::VintOverflow { .. })
    ));
    Ok(())
}

#[test]
fn randomized_blockmax_mutations_are_typed_and_never_panic() -> TestResult {
    const BASE_SEED: u64 = 0xb10c_4a58_f00d_5eed;

    init_replay_tracing();
    let encoded_postings = EncodedPostingList::encode(&sparse_postings(400, 100))?;
    let posting_list = encoded_postings.posting_list()?;
    let canonical = EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?;
    let canonical_len = canonical.as_bytes().len();
    let canonical_len_u64 = u64::try_from(canonical_len)?;

    for case in 0_u64..512 {
        let seed = BASE_SEED ^ case.wrapping_mul(0xa076_1d64_78bd_642f);
        tracing::info!(
            codec = "BLOCKMAX",
            seed,
            case,
            replay = "randomized_blockmax_mutations_are_typed_and_never_panic",
            "starting mutation case"
        );
        let mut state = seed;
        let mut bytes = if case < canonical_len_u64 {
            let mut bytes = canonical.as_bytes().to_vec();
            let offset = usize::try_from(case)?;
            let bit = random_u32(&mut state) & 7;
            bytes[offset] ^= 1_u8 << bit;
            bytes
        } else {
            let len = usize::try_from(random_u32(&mut state) % 257)?;
            (0..len)
                .map(|_| random_u32(&mut state).to_le_bytes()[0])
                .collect()
        };
        if case >= canonical_len_u64 && case.is_multiple_of(17) {
            bytes.push(random_u32(&mut state).to_le_bytes()[0]);
        }

        let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            BlockMaxList::parse_with_fieldnorms(&bytes, &posting_list, fixture_fieldnorm)
        }));
        assert!(
            parsed.is_ok(),
            "seed={seed:#x} case={case} len={}: BLOCKMAX parser panicked",
            bytes.len()
        );
        let codec_result: Result<BlockMaxList<'_>, BlockMaxError> =
            parsed.expect("caught BLOCKMAX parser result");
        if case < canonical_len_u64 {
            assert!(
                codec_result.is_err(),
                "seed={seed:#x} case={case}: mutated canonical BLOCKMAX byte was accepted"
            );
        }
    }
    Ok(())
}

#[test]
fn blockmax_public_score_view_requires_validated_doclen() -> TestResult {
    let postings = [Posting::new(128, 7), Posting::new(130, 3)];
    let document_lengths = [Some(3), None, Some(41)];
    let field_inputs = [DocLenFieldInput::new(7, &document_lengths)];
    let doclen = EncodedDocLenSection::encode(128, 131, &[7], &field_inputs)?;
    let section = doclen.section(&[7])?;
    let fieldnorms = section.field(7).ok_or("validated field")?;

    let (encoded_postings, encoded_bounds) =
        EncodedPostingList::encode_with_block_max(&postings, |doc_id| {
            fieldnorms.fieldnorm_id(u64::from(doc_id))
        })?;
    let posting_list = encoded_postings.posting_list()?;
    let validated = encoded_bounds.block_max_list(&posting_list, fieldnorms)?;
    assert_eq!(validated.entry_count(), 1);

    let mut understated = encoded_bounds.as_bytes().to_vec();
    let minimum = understated.last_mut().ok_or("fieldnorm byte")?;
    *minimum = u8::MAX;
    assert!(matches!(
        BlockMaxList::parse(&understated, &posting_list, fieldnorms),
        Err(BlockMaxError::MinimumFieldnormMismatch { .. })
    ));

    let maximum_length = [Some(u32::MAX)];
    let maximum_input = [DocLenFieldInput::new(7, &maximum_length)];
    let maximum_doclen = EncodedDocLenSection::encode(200, 201, &[7], &maximum_input)?;
    let maximum_section = maximum_doclen.section(&[7])?;
    let maximum_field = maximum_section.field(7).ok_or("maximum fieldnorm")?;
    let (maximum_postings, maximum_bounds) =
        EncodedPostingList::encode_with_block_max(&[Posting::new(200, 1)], |doc_id| {
            maximum_field.fieldnorm_id(u64::from(doc_id))
        })?;
    let maximum_list = maximum_postings.posting_list()?;
    let maximum_validated = maximum_bounds.block_max_list(&maximum_list, maximum_field)?;
    assert_eq!(maximum_validated.entries()[0].min_fieldnorm_id(), u8::MAX);
    Ok(())
}

#[test]
fn blockmax_cursor_advance_matches_validated_posting_blocks() -> TestResult {
    let postings = sparse_postings(400, 100);
    let encoded_postings = EncodedPostingList::encode(&postings)?;
    let posting_list = encoded_postings.posting_list()?;
    let encoded = EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?;
    let list =
        BlockMaxList::parse_with_fieldnorms(encoded.as_bytes(), &posting_list, fixture_fieldnorm)?;
    let last_doc = postings.last().ok_or("non-empty postings")?.doc_id;
    let targets = [
        0,
        postings[0].doc_id,
        postings[127].doc_id,
        postings[127].doc_id + 1,
        postings[128].doc_id,
        last_doc,
        last_doc + 1,
        u32::MAX,
    ];
    for target in targets {
        let expected_index = posting_list
            .blocks()
            .iter()
            .position(|block| block.last_doc >= target);
        let mut cursor = list.cursor();
        let actual = cursor.advance(target);
        assert_eq!(cursor.block_index(), expected_index, "target={target}");
        assert_eq!(
            cursor.last_doc(),
            expected_index.map(|index| posting_list.blocks()[index].last_doc),
            "target={target}"
        );
        assert_eq!(
            actual,
            expected_index.and_then(|index| list.entries().get(index).copied()),
            "target={target}"
        );
    }

    let mut cursor = list.cursor();
    for block_index in 0..list.entry_count() {
        assert_eq!(cursor.block_index(), Some(block_index));
        if block_index + 1 == list.entry_count() {
            assert_eq!(cursor.next(), None);
            assert_eq!(cursor.next(), None);
        } else {
            assert_eq!(cursor.next(), list.entries().get(block_index + 1).copied());
        }
    }

    let mut no_rewind = list.cursor();
    let late_target = postings[200].doc_id;
    let late_entry = no_rewind.advance(late_target);
    let late_index = no_rewind.block_index();
    assert!(late_index.is_some());
    assert_eq!(no_rewind.advance(0), late_entry);
    assert_eq!(no_rewind.block_index(), late_index);

    let max_postings = [Posting::new(u32::MAX - 1, 3), Posting::new(u32::MAX, 7)];
    let max_encoded = EncodedPostingList::encode(&max_postings)?;
    let max_posting_list = max_encoded.posting_list()?;
    let max_bounds = EncodedBlockMax::encode(&max_posting_list, fixture_fieldnorm)?;
    let max_list = BlockMaxList::parse_with_fieldnorms(
        max_bounds.as_bytes(),
        &max_posting_list,
        fixture_fieldnorm,
    )?;
    let mut max_cursor = max_list.cursor();
    assert_eq!(
        max_cursor.advance(u32::MAX),
        max_list.entries().first().copied()
    );
    assert_eq!(max_cursor.last_doc(), Some(u32::MAX));
    assert_eq!(max_cursor.next(), None);
    assert_eq!(max_cursor.advance(u32::MAX), None);
    Ok(())
}

#[test]
fn blockmax_covers_bitmap_posting_blocks() -> TestResult {
    let postings = postings_with_span(511);
    let ordinary_postings = EncodedPostingList::encode(&postings)?;
    let (encoded_postings, encoded) =
        EncodedPostingList::encode_with_block_max(&postings, fixture_fieldnorm)?;
    assert_eq!(encoded_postings, ordinary_postings);
    let posting_list = encoded_postings.posting_list()?;
    assert_eq!(posting_list.blocks()[0].kind, PostingBlockKind::Bitmap);

    assert_eq!(
        encoded,
        EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?
    );
    let parsed =
        BlockMaxList::parse_with_fieldnorms(encoded.as_bytes(), &posting_list, fixture_fieldnorm)?;
    let entry = parsed
        .entries()
        .first()
        .copied()
        .ok_or("one bitmap bound")?;
    assert_eq!(parsed.entry_count(), 1);
    assert_eq!(entry.first_doc(), postings[0].doc_id);
    assert_eq!(entry.block_offset(), 0);
    assert_eq!(entry.max_frequency(), 9);
    assert_eq!(
        entry.min_fieldnorm_id(),
        postings
            .iter()
            .filter_map(|posting| fixture_fieldnorm(posting.doc_id))
            .min()
            .ok_or("bitmap fieldnorm minimum")?
    );
    Ok(())
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn blockmax_bounds_dominate_every_score_across_live_avgdl_regimes() -> TestResult {
    let mut postings = dense_postings(400, 100);
    for (ordinal, posting) in postings.iter_mut().enumerate() {
        posting.freq = match ordinal / POSTINGS_PER_BLOCK {
            0 => u32::try_from(ordinal % 37 + 1)?,
            1 => u32::try_from(ordinal % 254 + 1)?,
            2 => match ordinal % 11 {
                0 => u32::MAX,
                1 => 256,
                2 => 255,
                _ => u32::try_from(ordinal % 31 + 1)?,
            },
            _ => u32::try_from(ordinal % 17 + 1)?,
        };
    }
    let encoded_postings = EncodedPostingList::encode(&postings)?;
    let posting_list = encoded_postings.posting_list()?;
    let encoded = EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?;
    let blockmax =
        BlockMaxList::parse_with_fieldnorms(encoded.as_bytes(), &posting_list, fixture_fieldnorm)?;
    let snapshot_pairs = blockmax.entries().to_vec();

    for average_fieldnorm in [0.25_f32, 1.0, 17.5, 1_000.0, 1_000_000.0] {
        for doc_count in [400_u64, 1_000, 100_000] {
            let idf = crate::contract::idf(400, doc_count);
            for boost in [0.25_f32, 1.0, 4.0] {
                let weight = idf * (1.0 + crate::contract::BM25_K1) * boost;
                for (block_index, entry) in blockmax.entries().iter().copied().enumerate() {
                    let bound = entry
                        .score_upper_bound(average_fieldnorm, weight)
                        .ok_or("valid positive block bound")?;
                    let decoded = decode_block_for_block_max(&posting_list, block_index)?;
                    for within in 0..usize::from(decoded.posting_count) {
                        let frequency = decoded.freqs[within] as f32;
                        let fieldnorm_id =
                            fixture_fieldnorm(decoded.docs[within]).ok_or("fixture fieldnorm")?;
                        let norm = crate::contract::cached_tf_component(
                            crate::contract::id_to_fieldnorm(fieldnorm_id),
                            average_fieldnorm,
                        );
                        let score = weight * (frequency / (frequency + norm));
                        assert!(
                            bound >= score,
                            "block={block_index} within={within} avgdl={average_fieldnorm} doc_count={doc_count} boost={boost} bound={bound} score={score}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(blockmax.entries(), snapshot_pairs);
    Ok(())
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn randomized_blockmax_bounds_dominate_seeded_oracle() -> TestResult {
    let boundary_counts = [1, 127, 128, 129, 255, 256, 400];

    init_replay_tracing();
    for case in 0_u64..32 {
        let seed = 0xd1b5_4a32_d192_ed03 ^ case;
        tracing::info!(
            codec = "BLOCKMAX",
            seed,
            case,
            replay = "randomized_blockmax_bounds_dominate_seeded_oracle",
            "starting seeded score-bound case"
        );
        let mut state = seed;
        let count = boundary_counts[usize::try_from(case)? % boundary_counts.len()];
        let mut doc_id = random_u32(&mut state) % 100;
        let mut postings = Vec::with_capacity(count);
        for ordinal in 0..count {
            if ordinal != 0 {
                let random = random_u32(&mut state);
                let step = if case.is_multiple_of(2) {
                    1 + random % 3
                } else {
                    5 + random % 12
                };
                doc_id = doc_id.checked_add(step).ok_or("seeded docid overflow")?;
            }
            let random = random_u32(&mut state);
            let frequency = if case.is_multiple_of(4) {
                1 + random % 254
            } else {
                match ordinal % 43 {
                    0 => 255,
                    1 => 256,
                    2 => u32::MAX,
                    _ => 1 + random % 254,
                }
            };
            postings.push(Posting::new(doc_id, frequency));
        }
        let fieldnorm_for_doc = |doc_id: u32| {
            u8::try_from((u64::from(doc_id).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ seed) & 0xff).ok()
        };
        let (encoded_postings, encoded_bounds) =
            EncodedPostingList::encode_with_block_max(&postings, fieldnorm_for_doc)?;
        let posting_list = encoded_postings.posting_list()?;
        let bounds = BlockMaxList::parse_with_fieldnorms(
            encoded_bounds.as_bytes(),
            &posting_list,
            fieldnorm_for_doc,
        )?;

        for live_avgdl in [0.25_f32, 1.0, 17.5, 1_000.0, 1_000_000.0] {
            for weight in [0.0_f32, 0.25, 3.25] {
                for (block_index, entry) in bounds.entries().iter().copied().enumerate() {
                    let upper = entry
                        .score_upper_bound(live_avgdl, weight)
                        .ok_or("valid seeded bound")?;
                    let decoded = decode_block_for_block_max(&posting_list, block_index)?;
                    for within in 0..usize::from(decoded.posting_count) {
                        let frequency = decoded.freqs[within] as f32;
                        let fieldnorm_id =
                            fieldnorm_for_doc(decoded.docs[within]).ok_or("seeded fieldnorm")?;
                        let norm = crate::contract::cached_tf_component(
                            crate::contract::id_to_fieldnorm(fieldnorm_id),
                            live_avgdl,
                        );
                        let score = weight * (frequency / (frequency + norm));
                        assert!(
                            upper >= score,
                            "seed={seed:#x} case={case} block={block_index} within={within} avgdl={live_avgdl} weight={weight} upper={upper} score={score}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn blockmax_q1_concat_preserves_bounds_and_rebases_only_offsets() -> TestResult {
    let left_postings = sparse_postings(100, 100);
    let right_postings = sparse_postings(300, 10_000);
    let left_encoded = EncodedPostingList::encode(&left_postings)?;
    let right_encoded = EncodedPostingList::encode(&right_postings)?;
    let left_list = left_encoded.posting_list()?;
    let right_list = right_encoded.posting_list()?;
    let left_bounds = EncodedBlockMax::encode(&left_list, fixture_fieldnorm)?;
    let right_bounds = EncodedBlockMax::encode(&right_list, fixture_fieldnorm)?;
    let left_blockmax =
        BlockMaxList::parse_with_fieldnorms(left_bounds.as_bytes(), &left_list, fixture_fieldnorm)?;
    let right_blockmax = BlockMaxList::parse_with_fieldnorms(
        right_bounds.as_bytes(),
        &right_list,
        fixture_fieldnorm,
    )?;
    let left_concat = left_bounds.concat_list(&left_list)?;
    let right_concat = right_bounds.concat_list(&right_list)?;

    let merged_bounds = EncodedBlockMax::concatenate(&[&left_concat, &right_concat])?;
    let mut merged_posting_bytes = left_encoded.as_bytes().to_vec();
    merged_posting_bytes.extend_from_slice(right_encoded.as_bytes());
    let merged_postings = PostingList::parse(&merged_posting_bytes, 400)?;
    let merged = BlockMaxList::parse_with_fieldnorms(
        merged_bounds.as_bytes(),
        &merged_postings,
        fixture_fieldnorm,
    )?;
    assert_eq!(merged.entry_count(), 4);
    assert_eq!(
        merged_postings
            .blocks()
            .iter()
            .map(|block| block.posting_count)
            .collect::<Vec<_>>(),
        [100, 128, 128, 44]
    );

    let left_prefix = u64::try_from(left_encoded.as_bytes().len())?;
    let mut expected_entries = left_blockmax.entries().to_vec();
    expected_entries.extend(right_blockmax.entries().iter().map(|entry| BlockMaxEntry {
        first_doc: entry.first_doc,
        block_offset: left_prefix + entry.block_offset,
        max_frequency_code: entry.max_frequency_code,
        min_fieldnorm_id: entry.min_fieldnorm_id,
    }));
    assert_eq!(merged.entries(), expected_entries);

    for left_count in 1..POSTINGS_PER_BLOCK {
        let left = sparse_postings(left_count, 10);
        let right = sparse_postings(3, 100_000);
        let left_bytes = EncodedPostingList::encode(&left)?;
        let right_bytes = EncodedPostingList::encode(&right)?;
        let left_posting_list = left_bytes.posting_list()?;
        let right_posting_list = right_bytes.posting_list()?;
        let left_bound_bytes = EncodedBlockMax::encode(&left_posting_list, fixture_fieldnorm)?;
        let right_bound_bytes = EncodedBlockMax::encode(&right_posting_list, fixture_fieldnorm)?;
        let left_bound_list = left_bound_bytes.concat_list(&left_posting_list)?;
        let right_bound_list = right_bound_bytes.concat_list(&right_posting_list)?;
        let bounds = EncodedBlockMax::concatenate(&[&left_bound_list, &right_bound_list])?;
        let mut postings_bytes = left_bytes.as_bytes().to_vec();
        postings_bytes.extend_from_slice(right_bytes.as_bytes());
        let postings_list =
            PostingList::parse(&postings_bytes, u32::try_from(left_count + right.len())?)?;
        let parsed = BlockMaxList::parse_with_fieldnorms(
            bounds.as_bytes(),
            &postings_list,
            fixture_fieldnorm,
        )?;
        assert_eq!(parsed.entry_count(), 2, "left_count={left_count}");
        assert_eq!(parsed.entries()[0].block_offset, 0);
        assert_eq!(
            parsed.entries()[1].block_offset,
            u64::try_from(left_bytes.as_bytes().len())?,
            "left_count={left_count}"
        );
    }
    Ok(())
}

#[test]
fn blockmax_concat_rebasing_is_associative_and_checks_q1_order() -> TestResult {
    let a_postings = EncodedPostingList::encode(&sparse_postings(100, 100))?;
    let b_postings = EncodedPostingList::encode(&sparse_postings(300, 10_000))?;
    let c_postings = EncodedPostingList::encode(&sparse_postings(17, 100_000))?;
    let a_list = a_postings.posting_list()?;
    let b_list = b_postings.posting_list()?;
    let c_list = c_postings.posting_list()?;
    let a_bounds = EncodedBlockMax::encode(&a_list, fixture_fieldnorm)?;
    let b_bounds = EncodedBlockMax::encode(&b_list, fixture_fieldnorm)?;
    let c_bounds = EncodedBlockMax::encode(&c_list, fixture_fieldnorm)?;
    let a_blockmax = a_bounds.concat_list(&a_list)?;
    let b_blockmax = b_bounds.concat_list(&b_list)?;
    let c_blockmax = c_bounds.concat_list(&c_list)?;

    let direct = EncodedBlockMax::concatenate(&[&a_blockmax, &b_blockmax, &c_blockmax])?;
    let ab = EncodedBlockMax::concatenate(&[&a_blockmax, &b_blockmax])?;
    let mut ab_posting_bytes = a_postings.as_bytes().to_vec();
    ab_posting_bytes.extend_from_slice(b_postings.as_bytes());
    let ab_posting_list = PostingList::parse(&ab_posting_bytes, 400)?;
    let first_pair_blockmax = ab.concat_list(&ab_posting_list)?;
    let staged = EncodedBlockMax::concatenate(&[&first_pair_blockmax, &c_blockmax])?;
    assert_eq!(staged, direct);

    assert!(matches!(
        EncodedBlockMax::concatenate(&[&c_blockmax, &a_blockmax]),
        Err(BlockMaxError::NonAscendingConcat { .. })
    ));
    Ok(())
}

#[test]
fn for_and_bitmap_bitpacked_payloads_match_scalar_reference() -> TestResult {
    let frame_input = sparse_postings(POSTINGS_PER_BLOCK, 100);
    let frame = EncodedPostingList::encode(&frame_input)?;
    let frame_bytes = frame.as_bytes();
    let doc_width = frame_bytes[8];
    let freq_width = frame_bytes[10];
    let doc_len = bitpack::packed_byte_len(POSTINGS_PER_BLOCK - 1, doc_width)?;
    let doc_payload = &frame_bytes[11..11 + doc_len];
    let mut scalar_docs = [0_u32; POSTINGS_PER_BLOCK - 1];
    let mut wide_docs = [0_u32; POSTINGS_PER_BLOCK - 1];
    unpack_scalar_into(doc_payload, doc_width, &mut scalar_docs)?;
    unpack_wide_into(doc_payload, doc_width, &mut wide_docs)?;
    assert_eq!(scalar_docs, wide_docs);
    let freq_payload = &frame_bytes[11 + doc_len..];
    let mut scalar_freqs = [0_u32; POSTINGS_PER_BLOCK];
    let mut wide_freqs = [0_u32; POSTINGS_PER_BLOCK];
    unpack_scalar_into(freq_payload, freq_width, &mut scalar_freqs)?;
    unpack_wide_into(freq_payload, freq_width, &mut wide_freqs)?;
    assert_eq!(scalar_freqs, wide_freqs);

    let bitmap_input = dense_postings(POSTINGS_PER_BLOCK, 1_000);
    let bitmap = EncodedPostingList::encode(&bitmap_input)?;
    let bitmap_bytes = bitmap.as_bytes();
    assert_eq!(bitmap_bytes[0], BITMAP_KIND);
    let bitmap_freq_width = bitmap_bytes[4 + 4 + 2 + 64 + 1];
    let bitmap_freq_payload = &bitmap_bytes[4 + 4 + 2 + 64 + 2..];
    unpack_scalar_into(bitmap_freq_payload, bitmap_freq_width, &mut scalar_freqs)?;
    unpack_wide_into(bitmap_freq_payload, bitmap_freq_width, &mut wide_freqs)?;
    assert_eq!(scalar_freqs, wide_freqs);
    Ok(())
}
