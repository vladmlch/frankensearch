use crate::quiver::bitpack::{pack_lsb, unpack_scalar_into, unpack_wide_into};
use crate::quiver::postings::{
    BITMAP_KIND, FOR_KIND, FREQ_ALL_ONE, FREQ_BITPACKED, POSTINGS_PER_BLOCK_U8, VINT_KIND,
};

use super::*;

#[test]
fn independent_bitpack_known_answers() -> TestResult {
    assert_eq!(pack_lsb(&[1, 2, 5], 3)?, [0x51, 0x01]);
    assert_eq!(pack_lsb(&[1, 2, 3], 2)?, [0x39]);
    assert_eq!(
        pack_lsb(&[0x7856_3412, 0xdead_beef], 32)?,
        [0x12, 0x34, 0x56, 0x78, 0xef, 0xbe, 0xad, 0xde]
    );

    for (packed, width, expected) in [
        (&[0x51, 0x01][..], 3, &[1, 2, 5][..]),
        (&[0x39][..], 2, &[1, 2, 3][..]),
    ] {
        let mut scalar = vec![0; expected.len()];
        let mut wide = vec![0; expected.len()];
        unpack_scalar_into(packed, width, &mut scalar)?;
        unpack_wide_into(packed, width, &mut wide)?;
        assert_eq!(scalar, expected);
        assert_eq!(wide, expected);
    }
    Ok(())
}

#[test]
fn scalar_and_wide_match_every_width_and_shape() -> TestResult {
    const BASE_SEED: u64 = 0x0123_4567_89ab_cdef;

    init_replay_tracing();
    let mut counts: Vec<usize> = (0..=16).collect();
    counts.extend_from_slice(&[120, 127, 128]);
    for width in 0..=32 {
        for &count in &counts {
            let seed = BASE_SEED ^ u64::try_from(count)?;
            tracing::info!(
                codec = "BITPACK_SIMD",
                seed,
                width,
                count,
                replay = "scalar_and_wide_match_every_width_and_shape",
                "starting scalar/SIMD parity case"
            );
            let expected = values(width, count, seed);
            let packed = pack_lsb(&expected, width)?;
            let mut scalar = vec![u32::MAX; count];
            let mut wide = vec![u32::MAX; count];
            unpack_scalar_into(&packed, width, &mut scalar)?;
            unpack_wide_into(&packed, width, &mut wide)?;
            assert_eq!(
                scalar, expected,
                "seed={seed:#x} scalar width={width} count={count}"
            );
            assert_eq!(
                wide, expected,
                "seed={seed:#x} wide width={width} count={count}"
            );
        }
    }
    Ok(())
}

#[test]
fn wide_decode_is_independent_of_slice_alignment() -> TestResult {
    const BASE_SEED: u64 = 0xa5a5_5a5a_dead_beef;

    init_replay_tracing();
    for offset in 0..=31 {
        for width in 1..=32 {
            let seed = BASE_SEED ^ u64::try_from(offset)?;
            tracing::info!(
                codec = "BITPACK_SIMD",
                seed,
                offset,
                width,
                replay = "wide_decode_is_independent_of_slice_alignment",
                "starting unaligned SIMD case"
            );
            let expected = values(width, 127, seed);
            let packed = pack_lsb(&expected, width)?;
            let mut storage = vec![0_u8; offset];
            storage.extend_from_slice(&packed);
            storage.extend_from_slice(&[0_u8; 7]);
            let input = &storage[offset..offset + packed.len()];
            let mut actual = vec![0_u32; expected.len()];
            unpack_wide_into(input, width, &mut actual)?;
            assert_eq!(
                actual, expected,
                "seed={seed:#x} offset={offset} width={width}"
            );
        }
    }
    Ok(())
}

#[test]
fn bitpack_rejects_truncation_overlong_payloads_bad_widths_and_padding() -> TestResult {
    const TRUNCATION_SEED: u64 = 0xfeed_face_cafe_babe;
    const PADDING_SEED: u64 = 0x55aa_aa55;

    init_replay_tracing();
    for width in 1..=32 {
        tracing::info!(
            codec = "BITPACK",
            seed = TRUNCATION_SEED,
            width,
            replay = "bitpack_rejects_truncation_overlong_payloads_bad_widths_and_padding",
            "starting truncated-payload cases"
        );
        let expected = values(width, 127, TRUNCATION_SEED);
        let packed = pack_lsb(&expected, width)?;
        for cut in 0..packed.len() {
            let mut scalar = vec![0_u32; expected.len()];
            let mut wide = vec![0_u32; expected.len()];
            assert!(
                matches!(
                    unpack_scalar_into(&packed[..cut], width, &mut scalar),
                    Err(BitpackError::LengthMismatch { .. })
                ),
                "seed={TRUNCATION_SEED:#x} scalar width={width} cut={cut}"
            );
            assert!(
                matches!(
                    unpack_wide_into(&packed[..cut], width, &mut wide),
                    Err(BitpackError::LengthMismatch { .. })
                ),
                "seed={TRUNCATION_SEED:#x} wide width={width} cut={cut}"
            );
        }
    }

    let mut output = [0_u32; 1];
    assert!(matches!(
        unpack_scalar_into(&[], 33, &mut output),
        Err(BitpackError::InvalidWidth { width: 33 })
    ));
    assert!(matches!(
        unpack_wide_into(&[], 33, &mut output),
        Err(BitpackError::InvalidWidth { width: 33 })
    ));
    assert!(matches!(
        unpack_scalar_into(&[0], 0, &mut output),
        Err(BitpackError::LengthMismatch { .. })
    ));
    assert!(matches!(
        pack_lsb(&[1], 0),
        Err(BitpackError::ValueOutOfRange { .. })
    ));
    assert!(matches!(
        pack_lsb(&[8], 3),
        Err(BitpackError::ValueOutOfRange { .. })
    ));

    for (width, count) in [(1, 1), (3, 3), (5, 7), (7, 9)] {
        tracing::info!(
            codec = "BITPACK",
            seed = PADDING_SEED,
            width,
            count,
            replay = "bitpack_rejects_truncation_overlong_payloads_bad_widths_and_padding",
            "starting noncanonical-padding case"
        );
        let expected = values(width, count, PADDING_SEED);
        let mut noncanonical = pack_lsb(&expected, width)?;
        let used_bits = usize::from(width) * count % 8;
        assert_ne!(used_bits, 0);
        if let Some(last) = noncanonical.last_mut() {
            *last |= 1_u8 << used_bits;
        }
        let mut scalar = vec![0_u32; count];
        let mut wide = vec![0_u32; count];
        assert!(
            matches!(
                unpack_scalar_into(&noncanonical, width, &mut scalar),
                Err(BitpackError::NonCanonicalPadding { .. })
            ),
            "seed={PADDING_SEED:#x} scalar width={width} count={count}"
        );
        assert!(
            matches!(
                unpack_wide_into(&noncanonical, width, &mut wide),
                Err(BitpackError::NonCanonicalPadding { .. })
            ),
            "seed={PADDING_SEED:#x} wide width={width} count={count}"
        );
    }

    let mut overlong = pack_lsb(&[5], 3)?;
    overlong.push(0);
    assert!(matches!(
        unpack_wide_into(&overlong, 3, &mut output),
        Err(BitpackError::LengthMismatch { .. })
    ));
    Ok(())
}

#[test]
fn posting_roundtrip_covers_empty_partial_full_and_multiblock_shapes() -> TestResult {
    for count in [0, 1, 2, 127, 128, 129, 255, 256, 257, 400] {
        let expected = sparse_postings(count, 100);
        let encoded = EncodedPostingList::encode(&expected)?;
        let repeated = EncodedPostingList::encode(&expected)?;
        assert_eq!(encoded, repeated, "determinism count={count}");
        let parsed = encoded.posting_list()?;
        assert_eq!(parsed.doc_freq(), u32::try_from(count)?);
        assert_eq!(parsed.decode_all()?, expected, "roundtrip count={count}");
        assert_eq!(
            parsed.block_count(),
            count.div_ceil(POSTINGS_PER_BLOCK),
            "block count={count}"
        );
    }
    Ok(())
}

#[test]
fn owned_lists_reopen_with_shape_derived_limits() -> TestResult {
    let posting_count = DEFAULT_MAX_POSTINGS_PER_TERM + 1;
    let block_count = posting_count.div_ceil(POSTINGS_PER_BLOCK);
    let limits = EncodedPostingList::validation_limits(posting_count, block_count);
    assert_eq!(limits.max_postings, posting_count);
    assert_eq!(limits.max_blocks, block_count);

    // Avoid allocating a multi-million-posting fixture: an empty synthetic
    // owned wrapper must get past the hostile-input cap and fail on its
    // actual byte/count mismatch instead.
    let owned = EncodedPostingList {
        bytes: Vec::new(),
        doc_freq: u32::try_from(posting_count)?,
        block_count,
    };
    let expected_doc_freq = u32::try_from(posting_count)?;
    assert!(matches!(
        owned.posting_list(),
        Err(PostingCodecError::DocFrequencyMismatch {
            expected,
            actual: 0
        }) if expected == expected_doc_freq
    ));
    Ok(())
}

#[test]
fn randomized_posting_roundtrip_and_cursor_match_linear_oracle() -> TestResult {
    const BASE_SEED: u64 = 0x6a09_e667_f3bc_c909;
    let boundary_counts = [0, 1, 2, 126, 127, 128, 129, 255, 256, 257, 511, 512];

    init_replay_tracing();
    for case in 0_u64..48 {
        let seed = BASE_SEED ^ case;
        tracing::info!(
            codec = "POSTINGS",
            seed,
            case,
            replay = "randomized_posting_roundtrip_and_cursor_match_linear_oracle",
            "starting randomized posting case"
        );
        let mut state = seed;
        let count = if case < u64::try_from(boundary_counts.len())? {
            boundary_counts[usize::try_from(case)?]
        } else {
            usize::try_from(random_u32(&mut state) % 513)?
        };
        let mut doc_id = random_u32(&mut state) % 100;
        let mut expected = Vec::with_capacity(count);
        for ordinal in 0..count {
            if ordinal != 0 {
                let random = random_u32(&mut state);
                let step = if ordinal % 37 == 0 {
                    512 + random % 4_096
                } else {
                    1 + random % 7
                };
                doc_id = doc_id.checked_add(step).ok_or("fixture docid overflow")?;
            }
            let random = random_u32(&mut state);
            let freq = match ordinal % 41 {
                0 => u32::MAX,
                1 => 256,
                _ => 1 + random % 31,
            };
            expected.push(Posting::new(doc_id, freq));
        }

        let encoded = EncodedPostingList::encode(&expected)?;
        assert_eq!(encoded, EncodedPostingList::encode(&expected)?);
        let list = encoded.posting_list()?;
        assert_eq!(
            list.decode_all_bounded(count)?,
            expected,
            "seed={seed:#x} case={case}"
        );

        let mut cursor = list.cursor()?;
        for (ordinal, posting) in expected.iter().copied().enumerate() {
            assert_eq!(
                cursor.current(),
                Some(posting),
                "seed={seed:#x} case={case}"
            );
            assert_eq!(cursor.posting_ordinal(), Some(u32::try_from(ordinal)?));
            cursor.next()?;
        }
        assert_eq!(cursor.current(), None);

        for _ in 0..8 {
            let target = random_u32(&mut state) % doc_id.saturating_add(2);
            let oracle = expected
                .iter()
                .copied()
                .find(|posting| posting.doc_id >= target);
            assert_eq!(
                list.cursor()?.advance(target)?,
                oracle,
                "seed={seed:#x} case={case}"
            );
        }
    }
    Ok(())
}

#[test]
fn writer_selects_bitmap_at_511_and_for_at_512() -> TestResult {
    let bitmap_input = postings_with_span(511);
    let bitmap = EncodedPostingList::encode(&bitmap_input)?;
    let bitmap_list = bitmap.posting_list()?;
    assert_eq!(bitmap_list.blocks()[0].kind, PostingBlockKind::Bitmap);
    assert_eq!(bitmap_list.decode_all()?, bitmap_input);

    let for_input = postings_with_span(512);
    let frame = EncodedPostingList::encode(&for_input)?;
    let frame_list = frame.posting_list()?;
    assert_eq!(
        frame_list.blocks()[0].kind,
        PostingBlockKind::FrameOfReference
    );
    assert_eq!(frame_list.decode_all()?, for_input);

    let mut invalid_bitmap = bitmap.as_bytes().to_vec();
    invalid_bitmap[4 + 4 + 2 + 63] |= 0x80;
    assert!(matches!(
        PostingList::parse(&invalid_bitmap, u32::from(POSTINGS_PER_BLOCK_U8)),
        Err(PostingCodecError::InvalidBitmap { .. })
    ));
    Ok(())
}

#[test]
fn reader_accepts_width_zero_for_and_width_32_extremes() -> TestResult {
    let consecutive: Vec<Posting> = dense_postings(POSTINGS_PER_BLOCK, 7)
        .into_iter()
        .map(|posting| Posting::new(posting.doc_id, 1))
        .collect();
    let width_zero = forced_for(&consecutive)?;
    assert_eq!(
        width_zero,
        [FOR_KIND, POSTINGS_PER_BLOCK_U8, 6, 0, 7, 0, 0, 0, 0, 0]
    );
    let parsed = PostingList::parse(&width_zero, u32::from(POSTINGS_PER_BLOCK_U8))?;
    assert_eq!(parsed.decode_all()?, consecutive);

    let mut extremes = Vec::with_capacity(POSTINGS_PER_BLOCK);
    extremes.push(Posting::new(0, 1));
    extremes.push(Posting::new(0x8000_0001, u32::MAX));
    for doc_id in 0x8000_0002..=0x8000_007f {
        extremes.push(Posting::new(doc_id, 1));
    }
    assert_eq!(extremes.len(), POSTINGS_PER_BLOCK);
    let encoded = EncodedPostingList::encode(&extremes)?;
    assert_eq!(encoded.as_bytes()[8], 32, "doc width");
    assert_eq!(encoded.as_bytes()[9], FREQ_BITPACKED);
    assert_eq!(encoded.as_bytes()[10], 32, "frequency width");
    assert_eq!(encoded.posting_list()?.decode_all()?, extremes);
    Ok(())
}

#[test]
fn vint_wire_goldens_and_canonicality() -> TestResult {
    assert_eq!(
        EncodedPostingList::encode(&[Posting::new(0, 127)])?.as_bytes(),
        [VINT_KIND, 1, 2, 0, 0, 127]
    );
    assert_eq!(
        EncodedPostingList::encode(&[Posting::new(128, 1)])?.as_bytes(),
        [VINT_KIND, 1, 3, 0, 0x80, 0x01, 0x01]
    );
    assert_eq!(
        EncodedPostingList::encode(&[Posting::new(u32::MAX, 1)])?.as_bytes(),
        [VINT_KIND, 1, 6, 0, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x01]
    );

    let overlong = raw_block(VINT_KIND, 1, &[0x80, 0x00, 0x01])?;
    assert!(matches!(
        PostingList::parse(&overlong, 1),
        Err(PostingCodecError::NonCanonicalVint { .. })
    ));
    let overflow = raw_block(VINT_KIND, 1, &[0xff, 0xff, 0xff, 0xff, 0x10, 0x01])?;
    assert!(matches!(
        PostingList::parse(&overflow, 1),
        Err(PostingCodecError::VintOverflow { .. })
    ));
    let zero_frequency = raw_block(VINT_KIND, 1, &[0x01, 0x00])?;
    assert!(matches!(
        PostingList::parse(&zero_frequency, 1),
        Err(PostingCodecError::ZeroFrequency { .. })
    ));

    let first_block = forced_for(&dense_postings(POSTINGS_PER_BLOCK, 7))?;
    let mut interior_zero_frequency = first_block;
    interior_zero_frequency.extend_from_slice(&raw_block(VINT_KIND, 1, &[0xc8, 0x01, 0x00])?);
    assert!(matches!(
        PostingList::parse(&interior_zero_frequency, 129),
        Err(PostingCodecError::ZeroFrequency { index: 128, .. })
    ));
    Ok(())
}

#[test]
fn encoder_rejects_zero_frequency_duplicates_and_descending_docids() {
    assert!(matches!(
        EncodedPostingList::encode(&[Posting::new(1, 0)]),
        Err(PostingCodecError::ZeroFrequency { .. })
    ));
    assert!(matches!(
        EncodedPostingList::encode(&[Posting::new(1, 1), Posting::new(1, 2)]),
        Err(PostingCodecError::NonAscendingInput { .. })
    ));
    assert!(matches!(
        EncodedPostingList::encode(&[Posting::new(2, 1), Posting::new(1, 1)]),
        Err(PostingCodecError::NonAscendingInput { .. })
    ));
}

#[test]
fn parser_rejects_header_payload_width_and_docfreq_corruption() -> TestResult {
    let expected = sparse_postings(400, 100);
    let encoded = EncodedPostingList::encode(&expected)?;
    for cut in 0..encoded.as_bytes().len() {
        assert!(
            PostingList::parse(&encoded.as_bytes()[..cut], encoded.doc_freq()).is_err(),
            "cut={cut}"
        );
    }

    let mut invalid_kind = encoded.as_bytes().to_vec();
    invalid_kind[0] = 9;
    assert!(matches!(
        PostingList::parse(&invalid_kind, encoded.doc_freq()),
        Err(PostingCodecError::InvalidBlockKind { .. })
    ));
    let mut invalid_count = encoded.as_bytes().to_vec();
    invalid_count[1] = 0;
    assert!(matches!(
        PostingList::parse(&invalid_count, encoded.doc_freq()),
        Err(PostingCodecError::InvalidBlockCount { .. })
    ));
    let mut declared_too_long = encoded.as_bytes().to_vec();
    let declared = u16::from_le_bytes([declared_too_long[2], declared_too_long[3]]);
    let longer = declared.saturating_add(1).to_le_bytes();
    declared_too_long[2..4].copy_from_slice(&longer);
    assert!(PostingList::parse(&declared_too_long, encoded.doc_freq()).is_err());
    assert!(matches!(
        PostingList::parse(encoded.as_bytes(), encoded.doc_freq() - 1),
        Err(PostingCodecError::DocFrequencyExceeded { .. })
    ));
    assert!(matches!(
        PostingList::parse(encoded.as_bytes(), encoded.doc_freq() + 1),
        Err(PostingCodecError::DocFrequencyMismatch { .. })
    ));

    let mut nonminimal_doc_payload = vec![0_u8; 4 + 1 + 1 + 16];
    nonminimal_doc_payload[..4].copy_from_slice(&7_u32.to_le_bytes());
    nonminimal_doc_payload[4] = 1;
    nonminimal_doc_payload[5] = FREQ_ALL_ONE;
    let nonminimal_doc = raw_block(FOR_KIND, POSTINGS_PER_BLOCK_U8, &nonminimal_doc_payload)?;
    assert!(matches!(
        PostingList::parse(&nonminimal_doc, u32::from(POSTINGS_PER_BLOCK_U8)),
        Err(PostingCodecError::NonCanonicalWidth { .. })
    ));

    let mut noncanonical_freq_payload = Vec::new();
    noncanonical_freq_payload.extend_from_slice(&7_u32.to_le_bytes());
    noncanonical_freq_payload.extend_from_slice(&[0, FREQ_BITPACKED, 0]);
    let noncanonical_freq = raw_block(FOR_KIND, POSTINGS_PER_BLOCK_U8, &noncanonical_freq_payload)?;
    assert!(matches!(
        PostingList::parse(&noncanonical_freq, u32::from(POSTINGS_PER_BLOCK_U8)),
        Err(PostingCodecError::NonCanonicalFrequencyEncoding { .. })
    ));
    Ok(())
}

#[test]
fn parser_and_materializer_enforce_resource_budgets_early() -> TestResult {
    let first = forced_for(&dense_postings(POSTINGS_PER_BLOCK, 7))?;
    let mut malformed_tail = first.clone();
    malformed_tail.extend_from_slice(&[9, 0, 0, 0]);
    assert!(matches!(
        PostingList::parse(&malformed_tail, 0),
        Err(PostingCodecError::DocFrequencyExceeded {
            expected: 0,
            actual: 128
        })
    ));

    let second = forced_for(&dense_postings(POSTINGS_PER_BLOCK, 1_000))?;
    let first_len = first.len();
    let mut two_blocks = first;
    two_blocks.extend_from_slice(&second);
    let one_block_limit = PostingListLimits {
        max_blocks: 1,
        max_postings: 256,
    };
    assert!(matches!(
        PostingList::parse_with_limits(&two_blocks, 256, one_block_limit),
        Err(PostingCodecError::BlockBudgetExhausted {
            limit: 1,
            validated: 1
        })
    ));
    assert!(
        PostingList::parse_with_limits(
            &two_blocks[..first_len],
            128,
            PostingListLimits {
                max_blocks: 1,
                max_postings: 128,
            }
        )
        .is_ok()
    );
    let mut cap_plus_tail = two_blocks[..first_len].to_vec();
    cap_plus_tail.push(0xff);
    assert!(matches!(
        PostingList::parse_with_limits(&cap_plus_tail, 128, one_block_limit),
        Err(PostingCodecError::BlockBudgetExhausted {
            limit: 1,
            validated: 1
        })
    ));
    assert!(matches!(
        PostingList::parse(&[], u32::try_from(DEFAULT_MAX_POSTINGS_PER_TERM + 1)?),
        Err(PostingCodecError::PostingLimitExceeded { .. })
    ));
    let list = PostingList::parse(&two_blocks, 256)?;
    assert!(list.block_bytes(usize::MAX).is_none());
    assert!(matches!(
        list.decode_all_bounded(255),
        Err(PostingCodecError::MaterializationLimitExceeded {
            limit: 255,
            actual: 256
        })
    ));
    Ok(())
}

#[test]
fn structured_arithmetic_and_cross_block_corruption_is_rejected() -> TestResult {
    let mut for_doc_overflow_payload = u32::MAX.to_le_bytes().to_vec();
    for_doc_overflow_payload.extend_from_slice(&[0, FREQ_ALL_ONE]);
    let for_doc_overflow = raw_block(FOR_KIND, POSTINGS_PER_BLOCK_U8, &for_doc_overflow_payload)?;
    assert!(matches!(
        PostingList::parse(&for_doc_overflow, 128),
        Err(PostingCodecError::ArithmeticOverflow { field: "docid", .. })
    ));

    let mut bitmap_overflow_payload = (u32::MAX - 100).to_le_bytes().to_vec();
    bitmap_overflow_payload.extend_from_slice(&128_u16.to_le_bytes());
    bitmap_overflow_payload.extend_from_slice(&[u8::MAX; 16]);
    bitmap_overflow_payload.extend_from_slice(&[0_u8; 48]);
    bitmap_overflow_payload.push(FREQ_ALL_ONE);
    let bitmap_overflow = raw_block(BITMAP_KIND, POSTINGS_PER_BLOCK_U8, &bitmap_overflow_payload)?;
    assert!(matches!(
        PostingList::parse(&bitmap_overflow, 128),
        Err(PostingCodecError::ArithmeticOverflow {
            field: "bitmap docid",
            ..
        })
    ));

    let vint_doc_overflow = raw_block(VINT_KIND, 2, &[0xff, 0xff, 0xff, 0xff, 0x0f, 1, 0, 1])?;
    assert!(matches!(
        PostingList::parse(&vint_doc_overflow, 2),
        Err(PostingCodecError::ArithmeticOverflow {
            field: "vint docid",
            ..
        })
    ));

    let mut stored_freqs = [0_u32; POSTINGS_PER_BLOCK];
    stored_freqs[0] = u32::MAX;
    let mut freq_overflow_payload = 0_u32.to_le_bytes().to_vec();
    freq_overflow_payload.extend_from_slice(&[0, FREQ_BITPACKED, 32]);
    freq_overflow_payload.extend_from_slice(&pack_lsb(&stored_freqs, 32)?);
    let freq_overflow = raw_block(FOR_KIND, POSTINGS_PER_BLOCK_U8, &freq_overflow_payload)?;
    assert!(matches!(
        PostingList::parse(&freq_overflow, 128),
        Err(PostingCodecError::ArithmeticOverflow {
            field: "frequency",
            ..
        })
    ));

    let one = raw_block(VINT_KIND, 1, &[1, 1])?;
    let mut equal_seam = one.clone();
    equal_seam.extend_from_slice(&one);
    assert!(matches!(
        PostingList::parse(&equal_seam, 2),
        Err(PostingCodecError::NonAscendingDecoded {
            posting_ordinal: 1,
            ..
        })
    ));
    Ok(())
}

#[test]
fn cursor_starts_on_first_advances_over_holes_and_fuses_at_u32_max() -> TestResult {
    let expected = vec![
        Posting::new(0, 1),
        Posting::new(2, 2),
        Posting::new(100, 3),
        Posting::new(u32::MAX, 4),
    ];
    let encoded = EncodedPostingList::encode(&expected)?;
    let list = encoded.posting_list()?;
    let mut cursor = list.cursor()?;
    assert_eq!(cursor.current(), Some(expected[0]));
    assert_eq!(cursor.posting_ordinal(), Some(0));
    assert_eq!(cursor.advance(0)?, Some(expected[0]));
    assert_eq!(cursor.advance(1)?, Some(expected[1]));
    assert_eq!(cursor.advance(2)?, Some(expected[1]));
    assert_eq!(cursor.advance(3)?, Some(expected[2]));
    assert_eq!(cursor.advance(u32::MAX)?, Some(expected[3]));
    assert_eq!(cursor.doc(), Some(u32::MAX));
    assert_eq!(cursor.freq(), Some(4));
    assert_eq!(cursor.next()?, None);
    assert_eq!(cursor.next()?, None);
    assert_eq!(cursor.advance(0)?, None);
    Ok(())
}

#[test]
fn consumed_cursor_matches_borrowed_cursor_across_blocks() -> TestResult {
    let expected = sparse_postings(400, 100);
    let encoded = EncodedPostingList::encode(&expected)?;
    let borrowed_list = encoded.posting_list()?;
    let owned_list = encoded.posting_list()?;
    let mut borrowed = borrowed_list.cursor()?;
    let mut owned = owned_list.into_cursor()?;
    let mut owned_clone = owned.clone();

    loop {
        assert_eq!(owned.current(), borrowed.current());
        assert_eq!(owned_clone.current(), borrowed.current());
        assert_eq!(owned.posting_ordinal(), borrowed.posting_ordinal());
        assert_eq!(owned_clone.posting_ordinal(), borrowed.posting_ordinal());
        assert_eq!(owned.block_index(), borrowed.block_index());
        assert_eq!(owned_clone.block_index(), borrowed.block_index());
        assert_eq!(owned.block_last_doc(), borrowed.block_last_doc());
        assert_eq!(owned_clone.block_last_doc(), borrowed.block_last_doc());

        let next = borrowed.next()?;
        assert_eq!(owned.next()?, next);
        assert_eq!(owned_clone.next()?, next);
        if next.is_none() {
            break;
        }
    }

    let targets = [
        0,
        expected[127].doc_id,
        expected[127].doc_id + 1,
        expected[255].doc_id + 1,
        expected.last().ok_or("non-empty fixture")?.doc_id,
        u32::MAX,
    ];
    for target in targets {
        let borrowed_list = encoded.posting_list()?;
        let mut borrowed = borrowed_list.cursor()?;
        let mut owned = encoded.posting_list()?.into_cursor()?;
        assert_eq!(owned.advance(target)?, borrowed.advance(target)?);
        assert_eq!(owned.block_index(), borrowed.block_index());
        assert_eq!(owned.block_last_doc(), borrowed.block_last_doc());
    }
    Ok(())
}

/// The decode predicates must agree with the moves they predict on every
/// step of a full scan, in both directions of the answer.
///
/// A caller admits a block's cost *before* the decode, so drift is not a
/// cosmetic defect: predicting `false` where a decode happens lets work
/// run unadmitted — the exact hole an admission gate exists to close —
/// and predicting `true` where none happens charges fuel for work never
/// performed. The observation is `block_index()`, which changes only when
/// `load_block` ran. The fixture asserts both answers actually occur, so
/// a predicate hard-coded either way fails here rather than passing
/// vacuously.
#[test]
fn next_decode_predicate_agrees_with_every_observed_decode() -> TestResult {
    let expected = dense_postings(POSTINGS_PER_BLOCK * 2 + 3, 0);
    let encoded = EncodedPostingList::encode(&expected)?;
    let mut cursor = encoded.posting_list()?.into_cursor()?;

    let mut predicted_decodes = 0_usize;
    let mut observed_decodes = 0_usize;
    let mut in_block_steps = 0_usize;
    loop {
        let predicted = cursor.next_decodes_block();
        let before = cursor.block_index();
        let moved = cursor.next()?;
        let after = cursor.block_index();
        let decoded = matches!((before, after), (Some(before), Some(after)) if before != after);
        assert_eq!(
            predicted, decoded,
            "prediction disagreed at {before:?} -> {after:?}"
        );
        predicted_decodes += usize::from(predicted);
        observed_decodes += usize::from(decoded);
        in_block_steps += usize::from(!decoded && moved.is_some());
        if moved.is_none() {
            break;
        }
    }

    assert_eq!(predicted_decodes, observed_decodes);
    assert_eq!(
        observed_decodes, 2,
        "a three-block fixture decodes twice after construction"
    );
    assert!(
        in_block_steps > 0,
        "the fixture must also exercise the false answer"
    );
    Ok(())
}

/// The same obligation for skips, across every interesting target class.
///
/// Each target gets a fresh cursor so the prediction is made from the
/// construction state, which is where a skip permit is decided.
#[test]
fn advance_decode_predicate_agrees_with_every_observed_skip() -> TestResult {
    let expected = dense_postings(POSTINGS_PER_BLOCK * 3, 0);
    let encoded = EncodedPostingList::encode(&expected)?;
    let last = expected.last().ok_or("non-empty fixture")?.doc_id;
    let targets = [
        0,
        expected[0].doc_id,
        expected[1].doc_id,
        expected[POSTINGS_PER_BLOCK - 1].doc_id,
        expected[POSTINGS_PER_BLOCK].doc_id,
        expected[POSTINGS_PER_BLOCK * 2].doc_id,
        last,
        last.saturating_add(1),
        u32::MAX,
    ];

    let mut predicted_true = 0_usize;
    let mut predicted_false = 0_usize;
    for target in targets {
        let mut cursor = encoded.posting_list()?.into_cursor()?;
        let predicted = cursor.advance_decodes_block(target);
        let before = cursor.block_index();
        cursor.advance(target)?;
        let after = cursor.block_index();
        let decoded = matches!((before, after), (Some(before), Some(after)) if before != after);
        assert_eq!(
            predicted, decoded,
            "prediction disagreed for target={target}: {before:?} -> {after:?}"
        );
        if predicted {
            predicted_true += 1;
        } else {
            predicted_false += 1;
        }
    }
    assert!(
        predicted_true > 0 && predicted_false > 0,
        "the target matrix must exercise both answers"
    );
    Ok(())
}

/// Build a term whose POSITIONS blocks are cut off the POSTINGS boundary.
///
/// Many positions per posting force several POSITIONS blocks across a
/// span that also crosses several POSTINGS blocks, and the two boundary
/// sets do not coincide — which is the only shape in which the two-stage
/// seek's first stage actually fires.
/// Positions per posting, chosen so a run is 31 bytes and the POSITIONS
/// boundaries land at ordinals 0/132/264 while POSTINGS land at 0/128/256.
/// The two sets must not coincide, or the seek's first stage never fires.
const DISPLACED_POSITIONS_PER_POSTING: u32 = 29;

fn positional_multi_block_fixture() -> (Vec<Posting>, Vec<u32>) {
    let mut postings = Vec::with_capacity(270);
    let mut flat = Vec::new();
    for ordinal in 0_u32..270 {
        postings.push(Posting::new(
            10 + ordinal * 3,
            DISPLACED_POSITIONS_PER_POSTING,
        ));
        let base = 70_000 + ordinal * 100;
        flat.extend((0..DISPLACED_POSITIONS_PER_POSTING).map(|index| base + index / 2));
    }
    (postings, flat)
}

/// Assert the displaced boundary layout both tests depend on.
///
/// These are stated exactly rather than probed, because the whole point
/// of the fixture is a specific misalignment: if the encoder's run size
/// ever shifts the POSITIONS cuts, the seek being exercised is no longer
/// the one these tests claim to exercise, and that must fail loudly here
/// instead of quietly weakening the evidence downstream.
fn assert_displaced_layout(posting_list: &PostingList<'_>, positions: &PositionList<'_>) {
    let posting_starts: Vec<u32> = posting_list
        .blocks()
        .iter()
        .map(|block| block.base_posting_ordinal)
        .collect();
    let position_starts: Vec<u32> = positions
        .blocks()
        .iter()
        .map(PositionBlockMeta::base_posting_ordinal)
        .collect();
    assert_eq!(
        posting_starts,
        vec![0, 128, 256],
        "POSTINGS layout moved: {posting_starts:?}"
    );
    assert_eq!(
        position_starts,
        vec![0, 132, 264],
        "POSITIONS layout moved: {position_starts:?}"
    );
}

/// The positional decode mirror must equal the decodes that actually
/// happened, counted at `load_block` itself — including on a far jump
/// that reads strictly fewer blocks than it crosses.
///
/// A positional seek is two-stage: a jump to the destination POSITIONS
/// block, then a posting-by-posting walk to the target. The jump skips
/// POSTINGS blocks without reading them while the walk reads every block
/// it crosses, so the decode count is neither a flag nor the change in
/// block index. This test earns that claim instead of asserting it: the
/// fixture is asserted to span multiple blocks of *both* kinds, and at
/// least one seek must be observed reading fewer blocks than its block
/// index moved — which is exactly the case a block-index-delta predictor
/// gets wrong. Verification is against `decoded_blocks()`, the counter at
/// the single decode site, never against a second derivation.
#[test]
fn positional_far_jump_reads_fewer_blocks_than_it_crosses() -> TestResult {
    let (postings, flat) = positional_multi_block_fixture();
    let posting_bytes = EncodedPostingList::encode(&postings)?;
    let posting_list = posting_bytes.posting_list()?;
    let encoded = EncodedPositionList::encode(&postings, &flat)?;
    let positions = encoded.position_list(&posting_list)?;
    assert_displaced_layout(&posting_list, &positions);

    // Ordinal 269 sits in POSTINGS block 2, and its POSITIONS block
    // starts at ordinal 264 — also in POSTINGS block 2. The seek
    // therefore jumps straight from block 0 into block 2, reading one
    // block while crossing two, and the walk that follows stays inside
    // that block. This is the exact case a block-index-delta predictor
    // gets wrong.
    let target = postings[269].doc_id;
    let mut cursor = positions.cursor()?;
    let entry_block = cursor.posting_block_index().ok_or("positioned cursor")?;
    let predicted = cursor.advance_decode_count(target);
    let baseline = cursor.decoded_blocks();
    cursor.advance(target)?;
    let observed = u64::try_from(cursor.decoded_blocks() - baseline)?;
    let landed_block = cursor.posting_block_index().ok_or("landed cursor")?;
    let crossed = u64::try_from(landed_block.saturating_sub(entry_block))?;

    assert_eq!(entry_block, 0);
    assert_eq!(landed_block, 2);
    assert_eq!(crossed, 2, "the seek must cross two POSTINGS blocks");
    assert_eq!(
        observed, 1,
        "a direct jump reads only the block it lands in"
    );
    assert_eq!(predicted, observed, "the mirror must match the observation");

    // A seek that stays inside the decoded block reads nothing, so the
    // mirror is exercised in both directions.
    let mut idle = positions.cursor()?;
    let idle_baseline = idle.decoded_blocks();
    let idle_predicted = idle.advance_decode_count(postings[0].doc_id);
    idle.advance(postings[0].doc_id)?;
    assert_eq!(idle_predicted, 0);
    assert_eq!(idle.decoded_blocks(), idle_baseline);
    Ok(())
}

/// The mirror must hold from a *later* cursor state, not only from the
/// construction state, and must predict nothing once exhausted.
///
/// A predicate that happens to be right at block zero can still be wrong
/// mid-term, where the positions block index and the posting block index
/// have drifted apart, so this walks a seam after a far jump and then
/// checks the fused state.
#[test]
fn positional_mirror_holds_at_a_later_seam_and_when_exhausted() -> TestResult {
    let (postings, flat) = positional_multi_block_fixture();
    let posting_bytes = EncodedPostingList::encode(&postings)?;
    let posting_list = posting_bytes.posting_list()?;
    let encoded = EncodedPositionList::encode(&postings, &flat)?;
    let positions = encoded.position_list(&posting_list)?;
    assert_displaced_layout(&posting_list, &positions);
    let mut cursor = positions.cursor()?;

    // Land just short of the POSTINGS seam at ordinal 128, which sits
    // inside the first POSITIONS block (0..132) — a later state whose two
    // block indices have already drifted apart.
    cursor.advance(postings[127].doc_id)?;
    assert_eq!(cursor.posting_block_index(), Some(0));

    // Cross the POSTINGS seam precisely: one step, one block.
    let predicted_seam = cursor.next_decode_count();
    let before_seam = cursor.decoded_blocks();
    assert_eq!(
        cursor.next()?.map(|posting| posting.doc_id),
        Some(postings[128].doc_id)
    );
    let observed_seam = u64::try_from(cursor.decoded_blocks() - before_seam)?;
    assert_eq!(observed_seam, 1, "stepping onto ordinal 128 enters a block");
    assert_eq!(predicted_seam, observed_seam);
    assert_eq!(cursor.posting_block_index(), Some(1));

    // Cross the displaced POSITIONS seam at ordinal 132, which enters no
    // POSTINGS block: the mirror must predict zero there.
    cursor.advance(postings[131].doc_id)?;
    let predicted_positions_seam = cursor.next_decode_count();
    let before_positions_seam = cursor.decoded_blocks();
    assert_eq!(
        cursor.next()?.map(|posting| posting.doc_id),
        Some(postings[132].doc_id)
    );
    assert_eq!(
        u64::try_from(cursor.decoded_blocks() - before_positions_seam)?,
        0,
        "a POSITIONS seam alone decodes no POSTINGS block"
    );
    assert_eq!(predicted_positions_seam, 0);
    assert_eq!(cursor.posting_block_index(), Some(1));

    while cursor.next()?.is_some() {}
    assert_eq!(cursor.posting_block_index(), None, "cursor must be fused");
    assert_eq!(
        cursor.next_decode_count(),
        0,
        "an exhausted cursor predicts no decode"
    );
    assert_eq!(
        cursor.advance_decode_count(u32::MAX),
        0,
        "an exhausted cursor predicts no decode for any seek"
    );
    let before_fused = cursor.decoded_blocks();
    assert_eq!(cursor.next()?, None);
    assert_eq!(cursor.advance(u32::MAX)?, None);
    assert_eq!(
        cursor.decoded_blocks(),
        before_fused,
        "a fused cursor must not decode"
    );
    Ok(())
}

/// The same obligation for positional stepping, at every step of a scan.
#[test]
fn positional_next_decode_mirror_equals_observed_decodes() -> TestResult {
    let postings: Vec<Posting> = (0..POSTINGS_PER_BLOCK * 2 + 5)
        .map(|index| Ok(Posting::new(u32::try_from(index)?, 1)))
        .collect::<Result<_, std::num::TryFromIntError>>()?;
    let values: Vec<u32> = (0..postings.len())
        .map(|index| u32::try_from(index % 11).unwrap_or(0))
        .collect();
    let posting_bytes = EncodedPostingList::encode(&postings)?;
    let encoded = EncodedPositionList::encode(&postings, &values)?;
    let posting_list = posting_bytes.posting_list()?;
    let positions = encoded.position_list(&posting_list)?;
    let mut cursor = positions.cursor()?;

    let mut total_predicted = 0_u64;
    let mut steps_that_decoded = 0_usize;
    loop {
        let predicted = cursor.next_decode_count();
        let baseline = cursor.decoded_blocks();
        let moved = cursor.next()?;
        let observed = u64::try_from(cursor.decoded_blocks() - baseline)?;
        assert_eq!(predicted, observed, "positional step mirror disagreed");
        total_predicted += predicted;
        steps_that_decoded += usize::from(observed > 0);
        if moved.is_none() {
            break;
        }
    }
    assert_eq!(
        steps_that_decoded, 2,
        "a three-block fixture decodes twice after construction"
    );
    assert_eq!(total_predicted, 2);
    Ok(())
}

#[test]
fn consumed_cursor_clone_retains_u32_max_block_state() -> TestResult {
    let mut expected = dense_postings(POSTINGS_PER_BLOCK, 0);
    expected.push(Posting::new(u32::MAX, 9));
    let encoded = EncodedPostingList::encode(&expected)?;
    let mut cursor = encoded.posting_list()?.into_cursor()?;

    assert_eq!(cursor.block_index(), Some(0));
    assert_eq!(cursor.block_last_doc(), Some(127));
    assert_eq!(cursor.advance(u32::MAX)?, expected.last().copied());
    assert_eq!(cursor.doc(), Some(u32::MAX));
    assert_eq!(cursor.posting_ordinal(), Some(128));
    assert_eq!(cursor.block_index(), Some(1));
    assert_eq!(cursor.block_last_doc(), Some(u32::MAX));

    let mut cloned = cursor.clone();
    assert_eq!(cursor.next()?, None);
    assert_eq!(cursor.block_index(), None);
    assert_eq!(cursor.block_last_doc(), None);
    assert_eq!(cloned.doc(), Some(u32::MAX));
    assert_eq!(cloned.block_index(), Some(1));
    assert_eq!(cloned.block_last_doc(), Some(u32::MAX));
    assert_eq!(cloned.advance(u32::MAX)?, expected.last().copied());
    assert_eq!(cloned.next()?, None);
    assert_eq!(cloned.next()?, None);
    assert_eq!(cloned.advance(u32::MAX)?, None);
    Ok(())
}

#[test]
fn advance_matches_linear_scan_across_block_boundaries() -> TestResult {
    let expected = sparse_postings(400, 100);
    let encoded = EncodedPostingList::encode(&expected)?;
    let list = encoded.posting_list()?;
    let last_doc = expected.last().map_or(0, |posting| posting.doc_id);
    for target in (0..=last_doc.saturating_add(1)).step_by(13) {
        let linear = expected
            .iter()
            .copied()
            .find(|posting| posting.doc_id >= target);
        let mut cursor = list.cursor()?;
        assert_eq!(cursor.advance(target)?, linear, "target={target}");
        if let Some(posting) = linear {
            let expected_ordinal = expected
                .iter()
                .position(|candidate| *candidate == posting)
                .and_then(|ordinal| u32::try_from(ordinal).ok());
            assert_eq!(cursor.posting_ordinal(), expected_ordinal);
        }
    }
    Ok(())
}

/// A backward descent through one block must decode it once, not once per
/// posting (`bd-quill-pruned-topdocs-term-order-9wu3p`).
///
/// This is the amplification the retained block exists to remove. A
/// doc-major union resolving an equal-document tie walks a clause's history
/// backwards, and the read serving it used to be stateless: every probe into
/// a block the cursor was not parked in decoded that whole block again, so a
/// descent through 128 postings cost 128 decodes where a forward walk costs
/// one.
///
/// The test descends from the last posting of the final block to the first
/// posting of the block before it, which is a probe per posting, and pins
/// the physical decode count at one. It also checks each answer against a
/// linear scan, because a cache that returned stale documents would satisfy
/// a decode bound perfectly.
#[test]
fn backward_descent_decodes_a_block_once() -> TestResult {
    let expected = (0..=300)
        .map(|doc| Posting::new(doc, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&expected)?;
    let list = encoded.posting_list()?;
    let mut cursor = list.cursor()?;
    assert!(
        cursor.block_count() >= 3,
        "fixture must have a block behind the parked one"
    );

    // Park in the last block, then walk backwards out of it.
    cursor.advance(300)?;
    let parked_block = cursor.block_index().expect("cursor is positioned");
    assert!(parked_block >= 2, "the descent must leave the parked block");
    let decoded_after_seek = cursor.decoded_blocks();

    let doc_before_linear = |target: u32| {
        expected
            .iter()
            .rev()
            .find(|posting| posting.doc_id < target)
            .map(|posting| posting.doc_id)
    };

    // Descend one posting at a time from the start of the parked block into
    // the block behind it. Every probe below `first_doc` of the parked block
    // reads the retained block.
    let parked_first = 256_u32;
    let mut probes = 0_usize;
    let mut probe = parked_first;
    while probe > 128 {
        let answer = cursor.doc_before(probe)?;
        assert_eq!(answer, doc_before_linear(probe), "probe={probe}");
        probes += 1;
        probe = answer.expect("a dense fixture always has an earlier posting");
    }
    assert!(
        probes >= 32,
        "the descent must probe deeply enough to expose re-decoding: {probes} probes"
    );
    assert_eq!(
        cursor.decoded_blocks() - decoded_after_seek,
        1,
        "a descent through one retained block must decode it exactly once"
    );
    Ok(())
}

/// Cloning a warm cursor is cold, and answers the same documents.
///
/// A pruning fork clones the cursor. If the retained block came with it,
/// every fork would perform an infallible `Vec` clone — the allocation the
/// cache's fallible reservation policy exists to avoid, and one a fork has
/// no way to refuse. Dropping it costs the clone one decode if it ever
/// descends, and costs correctness nothing, which is what this pins: the
/// clone re-decodes where the original would not, and both answer
/// identically.
#[test]
fn cloning_a_warm_cursor_is_cold_and_answers_identically() -> TestResult {
    let expected = (0..=300)
        .map(|doc| Posting::new(doc, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&expected)?;
    let list = encoded.posting_list()?;
    let mut warm = list.cursor()?;
    warm.advance(300)?;
    // Warm the retained block by descending out of the parked one.
    assert_eq!(warm.doc_before(255)?, Some(254));
    let warm_decodes = warm.decoded_blocks();
    // A second probe in the same block is served from the retained copy.
    assert_eq!(warm.doc_before(254)?, Some(253));
    assert_eq!(
        warm.decoded_blocks(),
        warm_decodes,
        "the original is warm and must not decode again"
    );

    let mut cold = warm.clone();
    assert_eq!(
        cold.doc(),
        warm.doc(),
        "a clone keeps the position it was taken at"
    );
    assert_eq!(
        cold.doc_before(254)?,
        Some(253),
        "a cold clone must answer exactly what the warm cursor answered"
    );
    assert_eq!(
        cold.decoded_blocks() - warm_decodes,
        1,
        "a cold clone re-decodes the block it did not inherit"
    );
    // And having paid once, it is warm in its own right.
    let rewarmed = cold.decoded_blocks();
    assert_eq!(cold.doc_before(253)?, Some(252));
    assert_eq!(cold.decoded_blocks(), rewarmed);
    Ok(())
}

/// A probe answered by the parked block or by validated metadata decodes
/// nothing and is charged nothing.
#[test]
fn parked_and_metadata_probes_are_free() -> TestResult {
    let expected = (0..=300)
        .map(|doc| Posting::new(doc, 1))
        .collect::<Vec<_>>();
    let encoded = EncodedPostingList::encode(&expected)?;
    let list = encoded.posting_list()?;
    let mut cursor = list.cursor()?;
    cursor.advance(300)?;
    let baseline = cursor.decoded_blocks();

    // Inside the parked block.
    assert!(!cursor.doc_before_decodes_block(299));
    assert_eq!(cursor.doc_before(299)?, Some(298));
    // Below every posting: answered from the block table alone.
    assert!(!cursor.doc_before_decodes_block(0));
    assert_eq!(cursor.doc_before(0)?, None);
    assert_eq!(
        cursor.decoded_blocks(),
        baseline,
        "parked and metadata probes must not decode"
    );
    Ok(())
}

#[test]
fn q1_df_100_plus_300_preserves_blocks_and_decoded_behavior() -> TestResult {
    let left_postings = sparse_postings(100, 100);
    let right_postings = sparse_postings(300, 10_000);
    let left = EncodedPostingList::encode(&left_postings)?;
    let right = EncodedPostingList::encode(&right_postings)?;
    let left_list = left.posting_list()?;
    let right_list = right.posting_list()?;

    let concatenated = EncodedPostingList::concatenate(&[&left_list, &right_list])?;
    assert_eq!(concatenated.doc_freq(), 400);
    assert_eq!(
        concatenated.block_count(),
        left_list.block_count() + right_list.block_count()
    );
    let merged = concatenated.posting_list()?;
    assert_eq!(
        merged
            .blocks()
            .iter()
            .map(|block| block.posting_count)
            .collect::<Vec<_>>(),
        [100, 128, 128, 44]
    );

    let mut expected_raw = Vec::with_capacity(left_list.block_count() + right_list.block_count());
    for index in 0..left_list.block_count() {
        expected_raw.push(
            left_list
                .block_bytes(index)
                .ok_or("validated left block bytes")?
                .to_vec(),
        );
    }
    for index in 0..right_list.block_count() {
        expected_raw.push(
            right_list
                .block_bytes(index)
                .ok_or("validated right block bytes")?
                .to_vec(),
        );
    }
    let mut merged_raw = Vec::with_capacity(merged.block_count());
    for index in 0..merged.block_count() {
        merged_raw.push(
            merged
                .block_bytes(index)
                .ok_or("validated merged block bytes")?
                .to_vec(),
        );
    }
    assert_eq!(merged_raw, expected_raw);

    let mut expected = left_postings.clone();
    expected.extend_from_slice(&right_postings);
    assert_eq!(merged.decode_all()?, expected);
    let monolithic = EncodedPostingList::encode(&expected)?;
    assert_eq!(monolithic.posting_list()?.decode_all()?, expected);
    assert_ne!(concatenated.as_bytes(), monolithic.as_bytes());

    let seam_doc = right_postings[0].doc_id;
    let mut cursor = merged.cursor()?;
    assert_eq!(cursor.advance(seam_doc - 1)?, Some(right_postings[0]));
    assert_eq!(cursor.advance(seam_doc)?, Some(right_postings[0]));
    assert_eq!(cursor.next()?, Some(right_postings[1]));

    assert!(matches!(
        EncodedPostingList::concatenate(&[&right_list, &left_list]),
        Err(PostingCodecError::ConcatRangeOrder { .. })
    ));
    assert!(matches!(
        EncodedPostingList::concatenate(&[]),
        Err(PostingCodecError::EmptyConcat)
    ));
    assert!(matches!(
        EncodedPostingList::concatenate_with_limits(
            &[&left_list, &right_list],
            PostingListLimits {
                max_blocks: DEFAULT_MAX_POSTING_BLOCKS,
                max_postings: 399,
            },
        ),
        Err(PostingCodecError::PostingLimitExceeded {
            limit: 399,
            actual: 400,
        })
    ));
    assert!(matches!(
        EncodedPostingList::concatenate_with_limits(
            &[&left_list, &right_list],
            PostingListLimits {
                max_blocks: 3,
                max_postings: DEFAULT_MAX_POSTINGS_PER_TERM,
            },
        ),
        Err(PostingCodecError::BlockBudgetExhausted {
            limit: 3,
            validated: 4,
        })
    ));
    Ok(())
}

#[test]
fn posting_concat_is_associative_and_copies_exact_source_bytes() -> TestResult {
    let a = EncodedPostingList::encode(&sparse_postings(100, 100))?;
    let b = EncodedPostingList::encode(&sparse_postings(120, 10_000))?;
    let c = EncodedPostingList::encode(&sparse_postings(80, 100_000))?;
    let first_list = a.posting_list()?;
    let second_list = b.posting_list()?;
    let third_list = c.posting_list()?;

    let direct = EncodedPostingList::concatenate(&[&first_list, &second_list, &third_list])?;
    let first_pair = EncodedPostingList::concatenate(&[&first_list, &second_list])?;
    let combined_list = first_pair.posting_list()?;
    let staged = EncodedPostingList::concatenate(&[&combined_list, &third_list])?;
    assert_eq!(staged, direct);

    let mut exact = Vec::new();
    exact.extend_from_slice(a.as_bytes());
    exact.extend_from_slice(b.as_bytes());
    exact.extend_from_slice(c.as_bytes());
    assert_eq!(direct.as_bytes(), exact);
    assert_eq!(direct.doc_freq(), 300);
    assert_eq!(
        direct.block_count(),
        a.block_count() + b.block_count() + c.block_count()
    );
    Ok(())
}

#[test]
fn every_partial_size_is_legal_at_an_interior_seam() -> TestResult {
    let right = sparse_postings(3, 100_000);
    let right_encoded = EncodedPostingList::encode(&right)?;
    for left_count in 1..POSTINGS_PER_BLOCK {
        let left = sparse_postings(left_count, 10);
        let left_encoded = EncodedPostingList::encode(&left)?;
        let mut bytes = left_encoded.as_bytes().to_vec();
        bytes.extend_from_slice(right_encoded.as_bytes());
        let list = PostingList::parse(&bytes, u32::try_from(left_count + right.len())?)?;
        let mut expected = left;
        expected.extend_from_slice(&right);
        assert_eq!(list.decode_all()?, expected, "left_count={left_count}");
        assert_eq!(list.blocks()[0].kind, PostingBlockKind::Vint);
        assert_eq!(list.blocks()[1].kind, PostingBlockKind::Vint);
    }
    Ok(())
}
