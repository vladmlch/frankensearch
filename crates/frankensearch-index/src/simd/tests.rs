use super::*;

/// The runtime AVX2 i8·i8 dot must be bit-identical to the portable `wide`
/// kernel across every dim shape (full vectors, sub-32 tails, odd lengths) —
/// integer addition is associative, so the 256-bit reduction equals the
/// scalar/`wide` sum exactly. Skips when the host lacks AVX2 (dispatch falls
/// back to the generic kernel there).
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_dot_matches_generic() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next_i8 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_possible_truncation)]
        ((state >> 24) as i8)
    };
    for &dim in &[
        0_usize, 1, 7, 8, 31, 32, 33, 63, 64, 65, 100, 256, 384, 511, 512,
    ] {
        let s: Vec<i8> = (0..dim).map(|_| next_i8()).collect();
        let q: Vec<i8> = (0..dim).map(|_| next_i8()).collect();
        let generic = dot_i8_i8_generic(&s, &q);
        // SAFETY: avx2 verified present above.
        #[allow(unsafe_code)]
        let avx2 = unsafe { dot_i8_i8_avx2(&s, &q) };
        assert_eq!(generic, avx2, "dim={dim}");
    }
}

#[test]
fn four_row_int8_dot_matches_four_independent_dots() {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    let mut next_i8 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_possible_truncation)]
        ((state >> 24) as i8)
    };

    for &dim in &[
        0_usize, 1, 7, 31, 32, 33, 63, 64, 65, 255, 256, 383, 384, 385, 1_040, 1_041,
    ] {
        let mut stored: Vec<i8> = (0..(4 * dim)).map(|_| next_i8()).collect();
        let mut query: Vec<i8> = (0..dim).map(|_| next_i8()).collect();
        if dim >= 2 {
            query[0] = i8::MIN;
            query[1] = i8::MAX;
            for row in 0..4 {
                stored[row * dim] = i8::MIN;
                stored[(row * dim) + 1] = i8::MAX;
            }
        }

        let expected =
            core::array::from_fn(|row| dot_i8_i8(&stored[row * dim..(row + 1) * dim], &query));
        assert_eq!(dot_i8x4_i8(&stored, &query), expected, "dim={dim}");
        assert_eq!(
            dot_i8x4_i8_generic(&stored, &query),
            expected,
            "generic dim={dim}"
        );

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present above and `stored` has four rows.
            #[allow(unsafe_code)]
            let avx2 = unsafe { dot_i8x4_i8_avx2(&stored, &query) };
            assert_eq!(avx2, expected, "avx2 dim={dim}");
        }
    }
}

/// The runtime AVX2 4-bit prepared dot must be bit-identical to the portable
/// `wide` kernel across packed-length shapes (full 16-byte chunks, sub-chunk
/// tails) — integer/in-range, so the 256-bit accumulation equals the `wide`
/// sum exactly. Skips when the host lacks AVX2.
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_dot4bit_matches_generic() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut state = 0xdead_beef_cafe_1234_u64;
    let mut next_u8 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_possible_truncation)]
        ((state >> 24) as u8)
    };
    for &len in &[0_usize, 1, 7, 15, 16, 17, 31, 32, 33, 100, 192, 256] {
        let q: Vec<u8> = (0..len).map(|_| next_u8()).collect();
        let s: Vec<u8> = (0..len).map(|_| next_u8()).collect();
        let prepared = prepare_4bit_query(&q);
        let generic = dot_4bit_prepared_generic(&s, &prepared);
        // SAFETY: avx2 verified present above.
        #[allow(unsafe_code)]
        let avx2 = unsafe { dot_4bit_prepared_avx2(&s, &prepared) };
        assert_eq!(generic, avx2, "len={len}");
    }
}

/// The runtime AVX2+F16C f16·f32 dot must be **bit-identical** to the portable
/// `wide` kernel for finite inputs (f16→f32 is exact; the accumulation and
/// `reduce_add` reductions are byte-for-byte the same). Embeddings are always
/// finite, so the test uses finite f16 (NaN payloads are the only legitimate
/// divergence and never occur in real vectors). Skips without AVX2+F16C.
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_f16dot_matches_generic() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
        return;
    }
    let mut state = 0x1357_9bdf_2468_ace0_u64;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
    };
    for &dim in &[1_usize, 7, 8, 9, 16, 17, 31, 64, 100, 256, 384, 512] {
        let q: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let mut bytes = Vec::with_capacity(dim * 2);
        for _ in 0..dim {
            bytes.extend_from_slice(&f16::from_f32(next_f32()).to_le_bytes());
        }
        let generic = dot_product_f16_bytes_f32_generic(&bytes, &q);
        // SAFETY: avx2+f16c verified present above.
        #[allow(unsafe_code)]
        let avx2 = unsafe { dot_product_f16_bytes_f32_avx2(&bytes, &q) };
        assert_eq!(generic.to_bits(), avx2.to_bits(), "dim={dim}");
    }
}

/// The runtime AVX2+F16C f16-slice dot must be bit-identical to the portable
/// `wide` kernel for finite inputs (same exact decode + reduce-through-`wide`).
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_f16slicedot_matches_generic() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
        return;
    }
    let mut state = 0x2468_ace0_1357_9bdf_u64;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
    };
    for &dim in &[1_usize, 7, 8, 9, 16, 17, 31, 64, 100, 256, 384, 512] {
        let q: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let s: Vec<f16> = (0..dim).map(|_| f16::from_f32(next_f32())).collect();
        let generic = dot_product_f16_f32_generic(&s, &q);
        // SAFETY: avx2+f16c verified present above.
        #[allow(unsafe_code)]
        let avx2 = unsafe { dot_product_f16_f32_avx2(&s, &q) };
        assert_eq!(generic.to_bits(), avx2.to_bits(), "dim={dim}");
    }
}

/// The runtime AVX2 f32-bytes dot must be bit-identical to the portable `wide`
/// kernel (same 4-accumulator grouping + reduce-through-`wide` + `mul_add`
/// tail; f32 LE bytes are native on x86). Skips without AVX2.
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_f32dot_matches_generic() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
    };
    for &dim in &[1_usize, 7, 8, 9, 16, 31, 32, 33, 64, 100, 256, 384, 512] {
        let q: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let mut bytes = Vec::with_capacity(dim * 4);
        for _ in 0..dim {
            bytes.extend_from_slice(&next_f32().to_le_bytes());
        }
        let generic = dot_product_f32_bytes_f32_generic(&bytes, &q);
        // SAFETY: avx2 verified present above.
        #[allow(unsafe_code)]
        let avx2 = unsafe { dot_product_f32_bytes_f32_avx2(&bytes, &q) };
        assert_eq!(generic.to_bits(), avx2.to_bits(), "dim={dim}");
    }
}

/// The runtime AVX2 f32-slice dot must be bit-identical to the portable `wide`
/// kernel (same 4-accumulator grouping + reduce-through-`wide` + separate
/// mul+add tail). Skips without AVX2.
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_f32slicedot_matches_generic() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut state = 0x7c6d_5e4f_3a2b_1908_u64;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
    };
    for &dim in &[1_usize, 7, 8, 9, 16, 31, 32, 33, 64, 100, 256, 384, 512] {
        let a: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let b: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let generic = dot_product_f32_f32_generic(&a, &b);
        // SAFETY: avx2 verified present above.
        #[allow(unsafe_code)]
        let avx2 = unsafe { dot_product_f32_f32_avx2(&a, &b) };
        assert_eq!(generic.to_bits(), avx2.to_bits(), "dim={dim}");
    }
}

/// The runtime AVX2+F16C int8 slab quantizer must be byte-for-byte identical to
/// the scalar kernel for finite inputs (`max` is exact, round-half-away is
/// emulated exactly, clamp/cast unchanged). Skips without AVX2+F16C.
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_quantize_i8_matches_generic() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
        return;
    }
    let mut state = 0x51ed_270b_9c4d_a3f8_u64;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
    };
    // Various lengths incl. sub-8 tails + a zero-vector edge (max_abs == 0).
    for &n in &[0_usize, 1, 7, 8, 9, 16, 31, 100, 384, 769] {
        let v: Vec<f16> = (0..n).map(|_| f16::from_f32(next_f32() * 3.0)).collect();
        let generic = quantize_f16_slab_to_i8_generic(&v);
        // SAFETY: avx2+f16c verified present above.
        #[allow(unsafe_code)]
        let avx2 = unsafe { quantize_f16_slab_to_i8_avx2(&v) };
        assert_eq!(generic, avx2, "n={n}");
    }
    let zeros = vec![f16::from_f32(0.0); 40];
    // SAFETY: avx2+f16c verified present above.
    #[allow(unsafe_code)]
    let avx2_zero = unsafe { quantize_f16_slab_to_i8_avx2(&zeros) };
    assert_eq!(quantize_f16_slab_to_i8_generic(&zeros), avx2_zero);
}

/// The mmap-backed little-endian f16 byte quantizer must match its scalar
/// byte fallback exactly, including sub-8 tails and the all-zero slab.
#[test]
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
fn avx2_quantize_i8_bytes_matches_generic() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
        return;
    }
    let mut state = 0x8734_2190_abcd_ef55_u64;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
    };
    for &n in &[0_usize, 1, 7, 8, 9, 16, 31, 100, 384, 769] {
        let mut bytes = Vec::with_capacity(n * 2);
        for _ in 0..n {
            bytes.extend_from_slice(&f16::from_f32(next_f32() * 3.0).to_le_bytes());
        }
        let generic = quantize_f16_le_bytes_to_i8_generic(&bytes);
        // SAFETY: avx2+f16c verified present above.
        #[allow(unsafe_code)]
        let avx2 = unsafe { quantize_f16_le_bytes_to_i8_avx2(&bytes) };
        assert_eq!(generic, avx2, "n={n}");
    }
    let zeros = vec![0_u8; 80];
    // SAFETY: avx2+f16c verified present above.
    #[allow(unsafe_code)]
    let avx2_zero = unsafe { quantize_f16_le_bytes_to_i8_avx2(&zeros) };
    assert_eq!(quantize_f16_le_bytes_to_i8_generic(&zeros), avx2_zero);
}

/// The runtime AVX2+F16C 4-bit slab packer must be byte-for-byte identical to
/// the scalar kernel across dim shapes (full 8-chunks, sub-8 tails incl. odd
/// dim) and multiple vectors. Skips without AVX2+F16C.
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_pack_4bit_matches_generic() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
        return;
    }
    let mut state = 0xa1b2_c3d4_e5f6_0789_u64;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
    };
    for &dim in &[1_usize, 2, 7, 8, 9, 15, 16, 17, 33, 64, 100, 384] {
        for &count in &[1_usize, 3] {
            let v: Vec<f16> = (0..dim * count)
                .map(|_| f16::from_f32(next_f32() * 2.5))
                .collect();
            let generic = pack_f16_slab_to_4bit_generic(&v, dim);
            // SAFETY: avx2+f16c verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { pack_f16_slab_to_4bit_avx2(&v, dim) };
            assert_eq!(generic, avx2, "dim={dim} count={count}");
        }
    }
}

/// The mmap-backed little-endian f16 byte packer must match the shipped
/// portable byte path exactly across full chunks, tails, and odd dimensions.
#[test]
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
fn avx2_pack_4bit_bytes_matches_generic() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
        return;
    }
    let mut state = 0x1976_0321_dcab_8e55_u64;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
    };
    for &dim in &[1_usize, 2, 7, 8, 9, 15, 16, 17, 33, 64, 100, 384] {
        for &count in &[1_usize, 3] {
            let mut bytes = Vec::with_capacity(dim * count * 2);
            for _ in 0..dim * count {
                bytes.extend_from_slice(&f16::from_f32(next_f32() * 2.5).to_le_bytes());
            }
            let generic = pack_f16_le_bytes_to_4bit_generic(&bytes, dim);
            // SAFETY: avx2+f16c verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { pack_f16_le_bytes_to_4bit_avx2(&bytes, dim) };
            assert_eq!(generic, avx2, "dim={dim} count={count}");
        }
    }
}

/// The runtime F16C f32→f16 encoder must be bit-identical to
/// `half::f16::from_f32` for finite inputs (both IEEE round-to-nearest-even),
/// across normal, sub-8 tails, tiny/subnormal and near-f16-max magnitudes.
#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_f16encode_matches_generic() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
        return;
    }
    let mut state = 0x6e3a_91c5_f072_8d4b_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        let u = (state >> 40) as f32 / (1_u64 << 23) as f32;
        // Mix magnitudes: most in [-1,1] (normalized embeddings) + some large
        // (near f16 max) + some tiny (f16 subnormal range).
        match state & 3 {
            0 => (u - 0.5) * 2.0,
            1 => (u - 0.5) * 130_000.0,
            2 => (u - 0.5) * 1e-6,
            _ => (u - 0.5) * 64.0,
        }
    };
    for &n in &[0_usize, 1, 7, 8, 9, 17, 64, 100, 385] {
        let src: Vec<f32> = (0..n).map(|_| next()).collect();
        let mut generic = Vec::new();
        encode_f32_to_f16_extend_generic(&src, &mut generic);
        let mut avx2 = Vec::new();
        // SAFETY: avx2+f16c verified present above.
        #[allow(unsafe_code)]
        unsafe {
            encode_f32_to_f16_extend_avx2(&src, &mut avx2);
        }
        let gb: Vec<u16> = generic.iter().map(|h| h.to_bits()).collect();
        let ab: Vec<u16> = avx2.iter().map(|h| h.to_bits()).collect();
        assert_eq!(gb, ab, "n={n}");
    }
}

/// Exhaustive proof that the SIMD f16->f32 widen matches the scalar reference
/// (`f16::to_f32`) for **every** one of the 65 536 f16 bit patterns. Finite,
/// zero and subnormal inputs must be bit-identical; inf/nan must map to
/// inf/nan with matching sign (the nan payload's quiet bit may legitimately
/// differ between the two decode methods).
#[test]
fn simd_f16_widen_is_bit_exact() {
    for base in (0u32..=0xFFFF).step_by(8) {
        let lanes = [
            base,
            base + 1,
            base + 2,
            base + 3,
            base + 4,
            base + 5,
            base + 6,
            base + 7,
        ];
        let widened = widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes));
        let out = bytemuck::cast::<f32x8, [f32; 8]>(widened);
        for (lane, &simd) in lanes.iter().zip(out.iter()) {
            let bits = u16::try_from(*lane).expect("<= 0xFFFF");
            let scalar = f16::from_bits(bits).to_f32();
            if scalar.is_nan() {
                assert!(simd.is_nan(), "bits={bits:#06x}: scalar nan, simd={simd}");
                assert_eq!(
                    scalar.is_sign_negative(),
                    simd.is_sign_negative(),
                    "bits={bits:#06x}: nan sign mismatch"
                );
            } else {
                assert_eq!(
                    scalar.to_bits(),
                    simd.to_bits(),
                    "bits={bits:#06x}: scalar={scalar} simd={simd}"
                );
            }
        }
    }
}

/// The 16-byte SIMD load path must decode to exactly the same lanes as the
/// scalar reference for representative patterns (sign, subnormal, zero, large,
/// inf, nan), proving the little-endian byte reinterpretation is correct.
#[test]
fn simd_f16_bytes_load_matches_scalar() {
    let samples: [f16; 8] = [
        f16::from_f32(0.0),
        f16::from_f32(-0.0),
        f16::from_f32(1.5),
        f16::from_f32(-2.25),
        f16::from_bits(0x0001), // smallest positive subnormal
        f16::from_f32(65504.0), // f16::MAX
        f16::INFINITY,
        f16::NAN,
    ];
    let mut bytes = [0u8; 16];
    for (i, v) in samples.iter().enumerate() {
        bytes[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
    }
    let out = bytemuck::cast::<f32x8, [f32; 8]>(widen8_f16_bytes(&bytes));
    for (v, &simd) in samples.iter().zip(out.iter()) {
        let scalar = v.to_f32();
        if scalar.is_nan() {
            assert!(simd.is_nan(), "expected nan for {v:?}");
        } else {
            assert_eq!(scalar.to_bits(), simd.to_bits(), "mismatch for {v:?}");
        }
    }
}

#[test]
fn dot_i8_i8_matches_scalar() {
    fn scalar(a: &[i8], b: &[i8]) -> i32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| i32::from(x) * i32::from(y))
            .sum()
    }
    // Lengths that exercise the 8-wide loop, a partial tail, and extremes.
    let to_i8 = |i: usize, m: usize| -> i8 {
        i8::try_from(i32::try_from(i * m % 255).expect("< 255") - 127).expect("in range")
    };
    for len in [0_usize, 1, 7, 8, 9, 16, 17, 256, 384, 385] {
        let a: Vec<i8> = (0..len).map(|i| to_i8(i, 37)).collect();
        let b: Vec<i8> = (0..len).map(|i| to_i8(i, 53)).collect();
        assert_eq!(dot_i8_i8(&a, &b), scalar(&a, &b), "len={len}");
    }
    // Worst-case magnitude: all -128 * -128 over 512 dims must not overflow i32.
    let a = vec![i8::MIN; 512];
    assert_eq!(dot_i8_i8(&a, &a), 512 * 128 * 128);
}

#[test]
fn dot_packed_4bit_matches_scalar() {
    fn lo(b: u8) -> i32 {
        i32::from(((b & 0x0F) ^ 0x08).cast_signed() - 8)
    }
    fn hi(b: u8) -> i32 {
        i32::from(((b >> 4) ^ 0x08).cast_signed() - 8)
    }
    fn scalar(s: &[u8], q: &[u8]) -> i32 {
        s.iter()
            .zip(q)
            .map(|(&a, &b)| lo(a) * lo(b) + hi(a) * hi(b))
            .sum()
    }
    // Lengths exercising the 16-wide loop, a partial tail, and extremes.
    for len in [0_usize, 1, 5, 15, 16, 17, 32, 33, 192, 193] {
        let s: Vec<u8> = (0..len)
            .map(|i| u8::try_from((i * 37 + 11) % 256).expect("bounded by % 256"))
            .collect();
        let q: Vec<u8> = (0..len)
            .map(|i| u8::try_from((i * 53 + 7) % 256).expect("bounded by % 256"))
            .collect();
        assert_eq!(dot_packed_4bit(&s, &q), scalar(&s, &q), "len={len}");
        let prepared = prepare_4bit_query(&q);
        assert_eq!(
            dot_4bit_prepared(&s, &prepared),
            scalar(&s, &q),
            "prepared len={len}"
        );
    }
    // All nibbles = -7 (0x99 byte): each dim contributes 49.
    let a = vec![0x99_u8; 16];
    assert_eq!(dot_packed_4bit(&a, &a), 32 * 49);
    let prepared = prepare_4bit_query(&a);
    assert_eq!(dot_4bit_prepared(&a, &prepared), 32 * 49);
}

/// End-to-end quality gate for the int8 ADC two-pass (`bd-b5wl`): does an int8
/// pass-1 (top `k*mult` candidates) + exact f16 rescore recover the true f16
/// top-k? Self-contained over random L2-normalized vectors. Prints the measured
/// recall@10 (run with `-- --nocapture`) and asserts a conservative floor.
#[test]
#[allow(clippy::cast_possible_truncation)]
fn int8_two_pass_recall_at_10() {
    fn xorshift(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 40) as f32 / (1_u64 << 24) as f32).mul_add(2.0, -1.0)
    }
    fn normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-9 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }
    fn quant_i8(x: f32) -> i8 {
        let s = (x * 127.0).round();
        if s >= 127.0 {
            127
        } else if s <= -127.0 {
            -127
        } else {
            s as i8
        }
    }

    let (dim, n, k, queries) = (128_usize, 3000_usize, 10_usize, 25_usize);
    // recall@k for a given mult == fraction of the exact top-k that land in the
    // int8 top-(k*mult) candidate set (pass-2 rescores exactly, so any candidate
    // that is truly top-k is recovered). Monotonic in mult; sweep to find the
    // smallest candidate budget that still holds recall (smaller => less select).
    let mults = [2_usize, 3, 5, 10, 20];
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut vecs_f16: Vec<Vec<f16>> = Vec::with_capacity(n);
    let mut vecs_i8: Vec<Vec<i8>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| xorshift(&mut state)).collect();
        normalize(&mut v);
        vecs_f16.push(v.iter().map(|&x| f16::from_f32(x)).collect());
        vecs_i8.push(v.iter().map(|&x| quant_i8(x)).collect());
    }

    let mut recall_sums = vec![0.0_f64; mults.len()];
    for _ in 0..queries {
        let mut q: Vec<f32> = (0..dim).map(|_| xorshift(&mut state)).collect();
        normalize(&mut q);
        let qi8: Vec<i8> = q.iter().map(|&x| quant_i8(x)).collect();

        let mut exact: Vec<(f32, usize)> = vecs_f16
            .iter()
            .enumerate()
            .map(|(i, fv)| (dot_product_f16_f32(fv, &q).expect("dot"), i))
            .collect();
        exact.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let exact_set: std::collections::HashSet<usize> =
            exact[..k].iter().map(|&(_, i)| i).collect();

        let mut p1: Vec<(i32, usize)> = vecs_i8
            .iter()
            .enumerate()
            .map(|(i, iv)| (dot_i8_i8(iv, &qi8), i))
            .collect();
        p1.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        for (mi, &mult) in mults.iter().enumerate() {
            let cand = (k * mult).min(p1.len());
            let cand_set: std::collections::HashSet<usize> =
                p1[..cand].iter().map(|&(_, i)| i).collect();
            let hit = exact_set.iter().filter(|i| cand_set.contains(i)).count();
            recall_sums[mi] += hit as f64 / k as f64;
        }
    }
    for (mi, &mult) in mults.iter().enumerate() {
        let avg = recall_sums[mi] / queries as f64;
        println!("int8 two-pass recall@{k} mult={mult} (n={n}, dim={dim}): {avg:.4}");
    }
    // Gate on the largest mult; smaller mults are reported for tuning.
    let avg_max = recall_sums[mults.len() - 1] / queries as f64;
    assert!(
        avg_max >= 0.80,
        "int8 two-pass recall@{k} too low: {avg_max:.4}"
    );
}

/// Viability probe for a **binary-quantization** first pass (Meilisearch-style):
/// pack `sign(x_i)` to bits, rank by Hamming agreement (`popcnt` — fast even on
/// SSE2, 1/16 the bytes of f16), then exact f16 rescore. Reports recall@10 vs the
/// candidate budget so we know whether binary ADC is worth building (it is much
/// coarser than int8, so the question is how big a `mult` it needs).
#[test]
fn binary_quant_recall_at_10() {
    fn xorshift(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 40) as f32 / (1_u64 << 24) as f32).mul_add(2.0, -1.0)
    }
    fn normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-9 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }
    fn pack_bits(v: &[f32]) -> Vec<u64> {
        let mut bits = vec![0_u64; v.len().div_ceil(64)];
        for (i, &x) in v.iter().enumerate() {
            if x >= 0.0 {
                bits[i / 64] |= 1_u64 << (i % 64);
            }
        }
        bits
    }
    // Lower hamming distance == more sign agreement == more similar.
    fn hamming(a: &[u64], b: &[u64]) -> u32 {
        a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
    }

    let (dim, n, k, queries) = (128_usize, 3000_usize, 10_usize, 25_usize);
    let mults = [5_usize, 10, 20, 50, 100];
    let mut state = 0x2468_ace0_1357_9bdf_u64;
    let mut vecs_f16: Vec<Vec<f16>> = Vec::with_capacity(n);
    let mut vecs_bits: Vec<Vec<u64>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| xorshift(&mut state)).collect();
        normalize(&mut v);
        vecs_bits.push(pack_bits(&v));
        vecs_f16.push(v.iter().map(|&x| f16::from_f32(x)).collect());
    }

    let mut recall_sums = vec![0.0_f64; mults.len()];
    for _ in 0..queries {
        let mut q: Vec<f32> = (0..dim).map(|_| xorshift(&mut state)).collect();
        normalize(&mut q);
        let qbits = pack_bits(&q);

        let mut exact: Vec<(f32, usize)> = vecs_f16
            .iter()
            .enumerate()
            .map(|(i, fv)| (dot_product_f16_f32(fv, &q).expect("dot"), i))
            .collect();
        exact.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let exact_set: std::collections::HashSet<usize> =
            exact[..k].iter().map(|&(_, i)| i).collect();

        // Rank by ascending hamming (descending agreement), index tie-break.
        let mut ranked: Vec<(u32, usize)> = vecs_bits
            .iter()
            .enumerate()
            .map(|(i, bv)| (hamming(bv, &qbits), i))
            .collect();
        ranked.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        for (mi, &mult) in mults.iter().enumerate() {
            let cand = (k * mult).min(ranked.len());
            let cand_set: std::collections::HashSet<usize> =
                ranked[..cand].iter().map(|&(_, i)| i).collect();
            let hit = exact_set.iter().filter(|i| cand_set.contains(i)).count();
            recall_sums[mi] += hit as f64 / k as f64;
        }
    }
    for (mi, &mult) in mults.iter().enumerate() {
        let avg = recall_sums[mi] / queries as f64;
        println!("binary-quant recall@{k} mult={mult} (n={n}, dim={dim}): {avg:.4}");
    }
    // No hard gate — this is a viability probe; just keep it from silently
    // returning 0 (which would signal a logic error).
    assert!(recall_sums[mults.len() - 1] / queries as f64 > 0.0);
}

fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn scalar_dot_f16(stored: &[f16], query: &[f32]) -> f32 {
    stored.iter().zip(query).map(|(x, y)| x.to_f32() * y).sum()
}

fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm = vec.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return vec.to_vec();
    }
    vec.iter().map(|value| value / norm).collect()
}

#[test]
fn simd_matches_scalar_f32() {
    let a = vec![
        0.4, -0.1, 0.6, 0.2, -0.3, 0.8, 0.7, -0.5, 0.9, -0.6, 0.11, 0.25, 0.41, -0.72, 0.55, 0.31,
    ];
    let b = vec![
        -0.8, 0.7, 0.6, -0.2, 0.3, 0.9, -0.4, 0.1, 0.12, 0.21, -0.14, 0.75, -0.22, 0.35, 0.66,
        -0.19,
    ];
    let simd = dot_product_f32_f32(&a, &b).expect("dot product");
    let scalar = scalar_dot_f32(&a, &b);
    assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
}

#[test]
fn simd_matches_scalar_f16() {
    let query = vec![
        0.4, -0.1, 0.6, 0.2, -0.3, 0.8, 0.7, -0.5, 0.9, -0.6, 0.11, 0.25, 0.41, -0.72, 0.55, 0.31,
    ];
    let stored = vec![
        f16::from_f32(-0.8),
        f16::from_f32(0.7),
        f16::from_f32(0.6),
        f16::from_f32(-0.2),
        f16::from_f32(0.3),
        f16::from_f32(0.9),
        f16::from_f32(-0.4),
        f16::from_f32(0.1),
        f16::from_f32(0.12),
        f16::from_f32(0.21),
        f16::from_f32(-0.14),
        f16::from_f32(0.75),
        f16::from_f32(-0.22),
        f16::from_f32(0.35),
        f16::from_f32(0.66),
        f16::from_f32(-0.19),
    ];
    let simd = dot_product_f16_f32(&stored, &query).expect("dot product");
    let scalar = scalar_dot_f16(&stored, &query);
    assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
}

#[test]
fn remainder_elements_are_handled() {
    let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let b = vec![0.9, 0.8, 0.7, 0.6, 0.5];
    let simd = dot_product_f32_f32(&a, &b).expect("dot product");
    let scalar = scalar_dot_f32(&a, &b);
    assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
}

#[test]
fn zero_vector_dot_product_is_zero() {
    let stored = vec![f16::from_f32(0.0); 16];
    let query = vec![1.0; 16];
    let result = dot_product_f16_f32(&stored, &query).expect("dot product");
    assert!(result.abs() < f32::EPSILON);
}

#[test]
fn nan_input_propagates_nan() {
    let mut a = vec![1.0; 16];
    a[3] = f32::NAN;
    let b = vec![1.0; 16];
    let result = dot_product_f32_f32(&a, &b).expect("dot product");
    assert!(result.is_nan());
}

#[test]
fn dimension_mismatch_returns_error() {
    let a = vec![1.0; 8];
    let b = vec![1.0; 7];
    let err = dot_product_f32_f32(&a, &b).expect_err("must fail");
    assert!(matches!(
        err,
        SearchError::DimensionMismatch {
            expected: 8,
            found: 7
        }
    ));
}

#[test]
fn f16_precision_error_is_bounded_for_unit_vectors() {
    let pattern = [
        0.11_f32, -0.07, 0.19, 0.02, -0.13, 0.23, 0.31, -0.17, 0.05, -0.29, 0.37, 0.41,
    ];
    let mut stored_full = Vec::with_capacity(384);
    let mut query = Vec::with_capacity(384);
    for index in 0..384 {
        let value = pattern[index % pattern.len()];
        let other = pattern[(index + 3) % pattern.len()];
        stored_full.push(value);
        query.push(other);
    }
    let stored_full = normalize(&stored_full);
    let query = normalize(&query);
    let stored_f16: Vec<f16> = stored_full.iter().copied().map(f16::from_f32).collect();

    let f32_dot = scalar_dot_f32(&stored_full, &query);
    let f16_dot = dot_product_f16_f32(&stored_f16, &query).expect("dot product");
    assert!(
        (f32_dot - f16_dot).abs() < 0.01,
        "f32_dot={f32_dot}, f16_dot={f16_dot}"
    );
}

// ─── bd-1l4g tests begin ───

#[test]
fn cosine_similarity_f16_matches_dot_product() {
    let stored: Vec<f16> = (0_u16..16)
        .map(|i| f16::from_f32(f32::from(i) * 0.1))
        .collect();
    let query: Vec<f32> = (0_u16..16).map(|i| f32::from(i) * 0.2).collect();

    let cosine = cosine_similarity_f16(&stored, &query).expect("cosine");
    let dot = dot_product_f16_f32(&stored, &query).expect("dot");
    assert!(
        (cosine - dot).abs() < f32::EPSILON,
        "cosine_similarity_f16 should delegate to dot_product_f16_f32"
    );
}

#[test]
fn cosine_similarity_f16_dimension_mismatch() {
    let stored = vec![f16::from_f32(1.0); 8];
    let query = vec![1.0_f32; 9];
    let err = cosine_similarity_f16(&stored, &query).expect_err("must fail");
    assert!(matches!(
        err,
        SearchError::DimensionMismatch {
            expected: 8,
            found: 9
        }
    ));
}

#[test]
fn dot_product_f16_f32_dimension_mismatch() {
    let stored = vec![f16::from_f32(1.0); 4];
    let query = vec![1.0_f32; 5];
    let err = dot_product_f16_f32(&stored, &query).expect_err("must fail");
    assert!(matches!(
        err,
        SearchError::DimensionMismatch {
            expected: 4,
            found: 5
        }
    ));
}

#[test]
fn empty_vectors_dot_product_f32() {
    let result = dot_product_f32_f32(&[], &[]).expect("dot product");
    assert!(result.abs() < f32::EPSILON);
}

#[test]
fn empty_vectors_dot_product_f16() {
    let result = dot_product_f16_f32(&[], &[]).expect("dot product");
    assert!(result.abs() < f32::EPSILON);
}

#[test]
fn exactly_eight_elements_f32() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let simd = dot_product_f32_f32(&a, &b).expect("dot product");
    let scalar = scalar_dot_f32(&a, &b);
    assert!(
        (simd - scalar).abs() < 1e-6,
        "exactly 8 elements (one full SIMD chunk, no remainder)"
    );
}

#[test]
fn single_element_dot_product() {
    let a = vec![3.0_f32];
    let b = vec![4.0_f32];
    let result = dot_product_f32_f32(&a, &b).expect("dot product");
    assert!((result - 12.0).abs() < f32::EPSILON);
}

#[test]
fn self_dot_product_is_norm_squared() {
    let v = vec![3.0_f32, 4.0];
    let result = dot_product_f32_f32(&v, &v).expect("dot product");
    assert!((result - 25.0).abs() < f32::EPSILON); // 3^2 + 4^2 = 25
}

#[test]
fn f16_nan_propagates() {
    let stored = vec![
        f16::from_f32(1.0),
        f16::NAN,
        f16::from_f32(1.0),
        f16::from_f32(1.0),
    ];
    let query = vec![1.0_f32; 4];
    let result = dot_product_f16_f32(&stored, &query).expect("dot product");
    assert!(result.is_nan());
}

#[test]
fn large_256d_matches_scalar_f32() {
    let a: Vec<f32> = (0_u16..256).map(|i| (f32::from(i) * 0.01).sin()).collect();
    let b: Vec<f32> = (0_u16..256).map(|i| (f32::from(i) * 0.02).cos()).collect();
    let simd = dot_product_f32_f32(&a, &b).expect("dot product");
    let scalar = scalar_dot_f32(&a, &b);
    assert!(
        (simd - scalar).abs() < 1e-4,
        "256d: simd={simd}, scalar={scalar}"
    );
}

// ── vpmaddubs approximate int8 dot (bd-b5wl) ────────────────────────────

/// Deterministic int8 vectors with a bounded magnitude (no RNG in tests).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // bounded xorshift fixture
fn i8_vec(n: usize, seed: u64, bound: i32) -> Vec<i8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // map into [-bound, bound]
            let m = (2 * bound + 1) as u64;
            (((s % m) as i32) - bound) as i8
        })
        .collect()
}

/// Whether the production dispatcher can select the approximate,
/// pair-saturating AVX2 kernel on this test host.
fn maddubs_saturating_path_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// CORRECTNESS GATE. On magnitudes where no adjacent-pair `u8·i8` sum can exceed `i16`, the
/// `vpmaddubs` kernel is **bit-exact** to the scalar dot — proving the domain shift and the
/// `128·Σq` bias are correct. Bound 90: `u∈[0,218]`, `q∈[-90,90]`, pair-sum ≤ 2·218·90 ≈ 39k
/// can still saturate, so use bound 63 for the strict-exact case: 2·191·63 ≈ 24k < 32767.
#[test]
fn maddubs_is_bit_exact_when_pairs_do_not_saturate() {
    for dim in [32usize, 64, 128, 384, 100 /* has a scalar tail */] {
        let stored = i8_vec(dim, 0x1234, 63);
        let query = i8_vec(dim, 0x9abc, 63);
        let exact = dot_i8_i8_generic(&stored, &query);
        let bias = maddubs_query_bias(&query, dim);
        let approx = dot_i8_i8_maddubs(&stored, &query, bias);
        assert_eq!(
            exact, approx,
            "dim {dim}: maddubs must be exact on non-saturating magnitudes"
        );
    }
}

/// RECALL-SAFETY GATE. The real pass-1 input is int8 quantized from L2-normalized vectors, so a
/// 384-dim component is typically `~127/√384 ≈ ±6`, rarely past `±40`. At those magnitudes an
/// adjacent-pair `u8·i8` sum is `≤ 2·(40+128)·40 ≈ 13 440 < 32 767`, so `vpmaddubs` **never
/// saturates** and the kernel is bit-exact — zero recall risk on realistic quantized data.
/// Asserted bit-exact top-10 ordering over a realistic-magnitude corpus (the recall proxy).
#[test]
fn maddubs_is_exact_on_realistically_quantized_magnitudes() {
    let dim = 384;
    let query = i8_vec(dim, 0x5555, 40);
    let bias = maddubs_query_bias(&query, dim);
    for v in 0..64usize {
        let stored = i8_vec(dim, 0x1000 + v as u64, 40);
        assert_eq!(
            dot_i8_i8_generic(&stored, &query),
            dot_i8_i8_maddubs(&stored, &query, bias),
            "realistic magnitude (|x|≤40): maddubs must be bit-exact, so top-k order is exact"
        );
    }
}

/// BOUNDARY / honesty test. At the adversarial *uniform* full int8 range (`±127` on every
/// component — not a real quantized distribution), adjacent pairs DO saturate and the kernel is
/// NOT bit-exact. This documents the limit: `dot_i8_i8_maddubs` is only recall-safe when the
/// quantizer keeps per-pair magnitudes under the `i16` ceiling. Callers must not feed it
/// arbitrary int8 without that guarantee.
#[test]
fn maddubs_saturates_at_adversarial_uniform_full_range() {
    let dim = 384;
    let query = i8_vec(dim, 0x7777, 127);
    let bias = maddubs_query_bias(&query, dim);
    let mut any_diff = false;
    for v in 0..64usize {
        let stored = i8_vec(dim, 0x2000 + v as u64, 127);
        if dot_i8_i8_generic(&stored, &query) != dot_i8_i8_maddubs(&stored, &query, bias) {
            any_diff = true;
            break;
        }
    }
    if maddubs_saturating_path_available() {
        assert!(
            any_diff,
            "uniform ±127 is expected to saturate the AVX2 maddubs path; if it no longer does, revisit the recall bound"
        );
    } else {
        assert!(
            !any_diff,
            "non-AVX2 dispatch must retain the exact generic fallback at the full int8 range"
        );
    }
}

/// CORRECTNESS + ORDERING GATE for the FMA f16 dot (scan lever). FMA is not bit-identical to
/// mul+add (it rounds the product-sum once vs twice), but the difference is sub-ULP relative to
/// the score scale, and — the load-bearing property for a *ranking* kernel — it must not reorder
/// a corpus vs the shipped kernel. Asserts a tiny relative delta AND identical top-k order on a
/// realistic f16 corpus.
#[test]
fn fma_f16_dot_is_ulp_close_and_order_preserving() {
    let dim = 384_u64;
    let query: Vec<f32> = {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 40) as f32 / 1e6 - 0.5)
            .collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        v
    };
    let mk_bytes = |seed: u64| -> Vec<u8> {
        let mut s = seed | 1;
        let mut f: Vec<f32> = (0..dim)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / 1e6) - 0.5
            })
            .collect();
        let n = f.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut f {
            *x /= n;
        }
        f.iter()
            .flat_map(|&x| f16::from_f32(x).to_le_bytes())
            .collect()
    };

    let mut base: Vec<(f32, usize)> = Vec::new();
    let mut fma: Vec<(f32, usize)> = Vec::new();
    let mut max_rel = 0.0_f64;
    for v in 0..128usize {
        let bytes = mk_bytes(0x50 + v as u64);
        let b = dot_product_f16_bytes_f32(&bytes, &query).expect("base");
        let f = dot_product_f16_bytes_f32_fma(&bytes, &query);
        let scale = f64::from(b.abs().max(1e-6));
        max_rel = max_rel.max(f64::from((b - f).abs()) / scale);
        base.push((b, v));
        fma.push((f, v));
    }
    assert!(
        max_rel < 1e-4,
        "FMA vs mul+add relative delta {max_rel} exceeds the sub-ULP ranking-safe band"
    );
    base.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    fma.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let btop: Vec<usize> = base[..10].iter().map(|&(_, v)| v).collect();
    let ftop: Vec<usize> = fma[..10].iter().map(|&(_, v)| v).collect();
    assert_eq!(
        btop, ftop,
        "FMA must not reorder the top-10 vs the shipped f16 dot"
    );
}

/// The batched kernel (the one the row-blocked scan uses) must equal four single-row
/// [`dot_i8_i8_maddubs`] calls exactly — same domain shift, same shared bias.
#[test]
fn maddubs_batched_matches_four_single_calls() {
    for dim in [32usize, 128, 384, 100] {
        let query = i8_vec(dim, 0xCAFE, 40);
        let bias = maddubs_query_bias(&query, dim);
        let mut rows = Vec::with_capacity(dim * 4);
        let singles: Vec<i32> = (0..4_u64)
            .map(|r| {
                let row = i8_vec(dim, 0x3000 + r, 40);
                let s = dot_i8_i8_maddubs(&row, &query, bias);
                rows.extend_from_slice(&row);
                s
            })
            .collect();
        let batched = dot_i8x4_i8_maddubs(&rows, &query, bias);
        assert_eq!(batched.to_vec(), singles, "dim {dim}: batched != 4× single");
    }
}

/// A deterministic pseudo-normalized f32 vector (models an embedding). No RNG.
#[allow(clippy::cast_possible_truncation)] // intended f64→f32 rounding of a unit sample
fn norm_f32(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 11) as f64 / (1u64 << 53) as f64) as f32) - 0.5
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Quantize an f32 query to int8 with the shipped per-query max-abs scale (see
/// `search::quantize_i8_query`, replicated here to keep the test in-crate).
fn quantize_query_i8(q: &[f32]) -> Vec<i8> {
    let max_abs = q.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0; q.len()];
    }
    let scale = 127.0 / max_abs;
    q.iter()
        .map(|&x| {
            #[allow(clippy::cast_possible_truncation)]
            let v = (x * scale).round().clamp(-127.0, 127.0) as i8;
            v
        })
        .collect()
}

fn top_k_indices<F: Fn(usize) -> f32>(n: usize, k: usize, score: F) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = (0..n).map(|i| (score(i), i)).collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored[..k.min(n)].iter().map(|&(_, i)| i).collect()
}

fn recall(truth: &[usize], cand: &[usize]) -> f64 {
    let hit = truth.iter().filter(|t| cand.contains(t)).count();
    hit as f64 / truth.len() as f64
}

/// END-TO-END RECALL PROOF (bd-b5wl). On a realistic corpus quantized by the **shipped**
/// `quantize_f16_slab_to_i8` (global `127/max_abs` scale — so a few components reach ±127 and
/// `vpmaddubs` genuinely SATURATES, unlike the bit-exact `|x|≤40` cases above), the maddubs
/// pass-1 must preserve the exact-f32 top-k recall at least as well as the exact int8 pass-1.
/// This is the deterministic proof that swapping in the approximate kernel does not cost recall
/// vs the f32 reference — the guarantee the two-pass scan relies on before its f16 rescore.
#[test]
fn maddubs_pass1_preserves_f32_recall_under_real_saturation() {
    let dim = 384;
    let n = 512;
    let k = 10;
    let mult = 3; // pass-1 keeps top k·mult candidates

    // Realistic corpus + query, exact f32 ground truth.
    let docs: Vec<Vec<f32>> = (0..n).map(|d| norm_f32(dim, 0x100 + d as u64)).collect();
    let query = norm_f32(dim, 0xF00D);
    let f32_top_k = top_k_indices(n, k, |d| scalar_dot_f32(&docs[d], &query));

    // Shipped quantization: whole doc slab (global scale) -> int8; per-query int8.
    let flat_f16: Vec<f16> = docs
        .iter()
        .flat_map(|v| v.iter().map(|&x| f16::from_f32(x)))
        .collect();
    let slab_i8 = quantize_f16_slab_to_i8(&flat_f16);
    let doc_i8: Vec<&[i8]> = (0..n).map(|d| &slab_i8[d * dim..(d + 1) * dim]).collect();
    let query_i8 = quantize_query_i8(&query);
    let bias = maddubs_query_bias(&query_i8, dim);

    // Confirm the corpus actually saturates maddubs when the approximate AVX2 path is active.
    // On other architectures, the same fixture instead proves that runtime dispatch retains
    // the exact generic fallback before the common candidate-recall checks below.
    let saturates = (0..n).any(|d| {
        dot_i8_i8_generic(doc_i8[d], &query_i8) != dot_i8_i8_maddubs(doc_i8[d], &query_i8, bias)
    });
    if maddubs_saturating_path_available() {
        assert!(
            saturates,
            "real quantized corpus must exercise AVX2 maddubs saturation for this proof to bite"
        );
    } else {
        assert!(
            !saturates,
            "non-AVX2 dispatch must retain the exact generic fallback on the real corpus"
        );
    }

    // Pass-1 candidate sets (top k·mult) by exact int8 vs approximate maddubs.
    let cand = k * mult;
    #[allow(clippy::cast_precision_loss)]
    let int8_cands = top_k_indices(n, cand, |d| dot_i8_i8_generic(doc_i8[d], &query_i8) as f32);
    #[allow(clippy::cast_precision_loss)]
    let maddubs_cands = top_k_indices(n, cand, |d| {
        dot_i8_i8_maddubs(doc_i8[d], &query_i8, bias) as f32
    });

    let int8_recall = recall(&f32_top_k, &int8_cands);
    let maddubs_recall = recall(&f32_top_k, &maddubs_cands);
    assert!(
        maddubs_recall >= int8_recall,
        "maddubs pass-1 must not lose recall vs exact int8: maddubs={maddubs_recall} int8={int8_recall}"
    );
    assert!(
        (maddubs_recall - 1.0).abs() < f64::EPSILON,
        "maddubs pass-1 recall@{k} of the exact-f32 top-{k} into top-{cand} must be 1.0, got {maddubs_recall}"
    );
}

// ─── bd-1l4g tests end ───
