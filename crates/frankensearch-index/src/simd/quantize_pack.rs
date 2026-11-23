use half::f16;
use wide::f32x8;

use super::widen8_f16_bytes;

/// Quantize an f16 slab to int8 with one corpus-wide max-abs scale (the lazy int8
/// ADC slab build).
///
/// Runtime-dispatches to an AVX2+F16C kernel when available: the
/// build is **decode-bound** (`f16::to_f32` is software, run twice — once for
/// max-abs, once to quantize) and `round` is a per-element scalar op, both of which
/// `vcvtph2ps` + vector round crush. Falls back to the portable scalar kernel.
#[must_use]
pub fn quantize_f16_slab_to_i8(vectors_f16: &[f16]) -> Vec<i8> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { quantize_f16_slab_to_i8_avx2(vectors_f16) };
        }
    }
    quantize_f16_slab_to_i8_generic(vectors_f16)
}

/// Quantize a contiguous little-endian f16 byte slab to int8 with one corpus-wide
/// max-abs scale.
///
/// This is the mmap-backed twin of [`quantize_f16_slab_to_i8`]: FSVI stores f16
/// vectors as little-endian bytes, so the file-backed int8 two-pass path can use
/// the same AVX2+F16C quantization pipeline without first materializing a `Vec<f16>`.
#[must_use]
pub fn quantize_f16_le_bytes_to_i8(bytes: &[u8]) -> Vec<i8> {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above; on
            // little-endian x86, FSVI little-endian bytes are valid F16C inputs.
            #[allow(unsafe_code)]
            return unsafe { quantize_f16_le_bytes_to_i8_avx2(bytes) };
        }
    }
    quantize_f16_le_bytes_to_i8_generic(bytes)
}

/// Hand-written AVX2+F16C int8 slab quantizer. Bit-identical to the scalar kernel:
/// `max` is exact/associative so the vector max-abs equals the scalar fold; the
/// round is `f32::round` (half-away-from-zero) emulated as `trunc(v + copysign(0.5,
/// v))` (exact for `|v| ≤ 127`); the clamp + `as i8` cast are unchanged. Proven by
/// `avx2_quantize_i8_matches_generic`.
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available (the dispatch in
/// [`quantize_f16_slab_to_i8`] guarantees this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
pub(super) unsafe fn quantize_f16_slab_to_i8_avx2(vectors_f16: &[f16]) -> Vec<i8> {
    use core::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_ZERO, _mm_cvtsi128_si32, _mm_loadu_si128, _mm256_add_ps,
        _mm256_and_ps, _mm256_castsi256_si128, _mm256_cvtph_ps, _mm256_cvttps_epi32,
        _mm256_extracti128_si256, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps,
        _mm256_round_ps, _mm256_set_epi64x, _mm256_set1_ps, _mm256_setzero_ps, _mm256_shuffle_epi8,
        _mm256_storeu_ps,
    };
    let n = vectors_f16.len();
    let chunks = n / 8;
    // `half::f16` is `repr(transparent)` over `u16`, so the slab is `n*2` LE bytes.
    // SAFETY: `vectors_f16` has `n` f16 = `n*2` bytes, contiguous.
    #[allow(unsafe_code)]
    let bytes = unsafe { std::slice::from_raw_parts(vectors_f16.as_ptr().cast::<u8>(), n * 2) };

    // Pass 1: corpus-wide max-abs (F16C decode + clear sign bit + vector max).
    let mut max_abs = 0.0_f32;
    // SAFETY: avx2+f16c by contract; loads are `c < chunks`-bounded.
    unsafe {
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
        let mut vmax = _mm256_setzero_ps();
        for c in 0..chunks {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(c * 16).cast()));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(x, abs_mask));
        }
        let mut arr = [0.0_f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), vmax);
        for &v in &arr {
            max_abs = max_abs.max(v);
        }
    }
    for &x in &vectors_f16[chunks * 8..] {
        max_abs = max_abs.max(x.to_f32().abs());
    }
    if max_abs <= 0.0 {
        return vec![0_i8; n];
    }
    let scale = 127.0 / max_abs;

    // Pass 2: quantize (decode → ×scale → round-half-away → clamp → i8).
    let mut out: Vec<i8> = Vec::with_capacity(n);
    // SAFETY: avx2+f16c by contract; loads are `c < chunks`-bounded.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let vhalf = _mm256_set1_ps(0.5);
        let vmaxc = _mm256_set1_ps(127.0);
        let vminc = _mm256_set1_ps(-127.0);
        let vsign = _mm256_set1_ps(-0.0);
        // Select the low byte from each i32 lane. `vpshufb` operates within
        // 128-bit lanes, so the two four-byte halves are joined as one u64.
        let zero_bytes = i64::from_le_bytes([0x80; 8]);
        let lane_select = i64::from_le_bytes([0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80]);
        let lane_byte_mask = _mm256_set_epi64x(zero_bytes, lane_select, zero_bytes, lane_select);
        for c in 0..chunks {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(c * 16).cast()));
            let v = _mm256_mul_ps(x, vscale);
            // copysign(0.5, v): OR the 0.5 magnitude with v's sign bit.
            let half_signed = _mm256_or_ps(vhalf, _mm256_and_ps(v, vsign));
            let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
                _mm256_add_ps(v, half_signed),
            );
            let clamped = _mm256_min_ps(_mm256_max_ps(rounded, vminc), vmaxc);
            let i = _mm256_cvttps_epi32(clamped);
            let lane_bytes = _mm256_shuffle_epi8(i, lane_byte_mask);
            #[allow(clippy::cast_sign_loss)]
            let low = _mm_cvtsi128_si32(_mm256_castsi256_si128(lane_bytes)) as u32;
            #[allow(clippy::cast_sign_loss)]
            let high = _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(lane_bytes)) as u32;
            let packed = u64::from(low) | (u64::from(high) << 32);
            // SAFETY: capacity is `n`, and this loop initializes exactly
            // `chunks * 8 <= n` consecutive bytes before `set_len` below.
            out.as_mut_ptr()
                .add(c * 8)
                .cast::<u64>()
                .write_unaligned(packed);
        }
        out.set_len(chunks * 8);
    }
    for &x in &vectors_f16[chunks * 8..] {
        #[allow(clippy::cast_possible_truncation)]
        out.push((x.to_f32() * scale).round().clamp(-127.0, 127.0) as i8);
    }
    out
}

/// Hand-written AVX2+F16C int8 quantizer over little-endian f16 bytes. Same
/// arithmetic as [`quantize_f16_slab_to_i8_avx2`], but reads the mapped FSVI byte
/// slab directly.
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available and that the target is
/// little-endian. The dispatch in [`quantize_f16_le_bytes_to_i8`] guarantees this.
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,f16c")]
#[allow(
    clippy::cast_ptr_alignment,
    clippy::many_single_char_names,
    unsafe_code
)]
#[must_use]
pub(super) unsafe fn quantize_f16_le_bytes_to_i8_avx2(bytes: &[u8]) -> Vec<i8> {
    use core::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_ZERO, _mm_cvtsi128_si32, _mm_loadu_si128, _mm256_add_ps,
        _mm256_and_ps, _mm256_castsi256_si128, _mm256_cvtph_ps, _mm256_cvttps_epi32,
        _mm256_extracti128_si256, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps,
        _mm256_round_ps, _mm256_set_epi64x, _mm256_set1_ps, _mm256_setzero_ps, _mm256_shuffle_epi8,
        _mm256_storeu_ps,
    };
    let n = bytes.len() / 2;
    let chunks = n / 8;

    let mut max_abs = 0.0_f32;
    // SAFETY: avx2+f16c by contract; loads are `c < chunks`-bounded.
    unsafe {
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
        let mut vmax = _mm256_setzero_ps();
        for c in 0..chunks {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(c * 16).cast()));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(x, abs_mask));
        }
        let mut arr = [0.0_f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), vmax);
        for &v in &arr {
            max_abs = max_abs.max(v);
        }
    }
    let mut tail = chunks * 16;
    while tail + 2 <= bytes.len() {
        let value = f16::from_le_bytes([bytes[tail], bytes[tail + 1]])
            .to_f32()
            .abs();
        max_abs = max_abs.max(value);
        tail += 2;
    }
    if max_abs <= 0.0 {
        return vec![0_i8; n];
    }
    let scale = 127.0 / max_abs;

    let mut out: Vec<i8> = Vec::with_capacity(n);
    // SAFETY: avx2+f16c by contract; loads are `c < chunks`-bounded.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let vhalf = _mm256_set1_ps(0.5);
        let vmaxc = _mm256_set1_ps(127.0);
        let vminc = _mm256_set1_ps(-127.0);
        let vsign = _mm256_set1_ps(-0.0);
        // Select the low byte from each i32 lane. `vpshufb` operates within
        // 128-bit lanes, so the two four-byte halves are joined as one u64.
        let zero_bytes = i64::from_le_bytes([0x80; 8]);
        let lane_select = i64::from_le_bytes([0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80]);
        let lane_byte_mask = _mm256_set_epi64x(zero_bytes, lane_select, zero_bytes, lane_select);
        for c in 0..chunks {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(c * 16).cast()));
            let v = _mm256_mul_ps(x, vscale);
            let half_signed = _mm256_or_ps(vhalf, _mm256_and_ps(v, vsign));
            let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
                _mm256_add_ps(v, half_signed),
            );
            let clamped = _mm256_min_ps(_mm256_max_ps(rounded, vminc), vmaxc);
            let i = _mm256_cvttps_epi32(clamped);
            let lane_bytes = _mm256_shuffle_epi8(i, lane_byte_mask);
            #[allow(clippy::cast_sign_loss)]
            let low = _mm_cvtsi128_si32(_mm256_castsi256_si128(lane_bytes)) as u32;
            #[allow(clippy::cast_sign_loss)]
            let high = _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(lane_bytes)) as u32;
            let packed = u64::from(low) | (u64::from(high) << 32);
            // SAFETY: capacity is `n`, and this loop initializes exactly
            // `chunks * 8 <= n` consecutive bytes before `set_len` below.
            out.as_mut_ptr()
                .add(c * 8)
                .cast::<u64>()
                .write_unaligned(packed);
        }
        out.set_len(chunks * 8);
    }
    let mut tail = chunks * 16;
    while tail + 2 <= bytes.len() {
        #[allow(clippy::cast_possible_truncation)]
        out.push(
            (f16::from_le_bytes([bytes[tail], bytes[tail + 1]]).to_f32() * scale)
                .round()
                .clamp(-127.0, 127.0) as i8,
        );
        tail += 2;
    }
    out
}

/// Portable scalar int8 slab quantizer — the AVX2+F16C-dispatch fallback. Exposed
/// (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn quantize_f16_slab_to_i8_generic(vectors_f16: &[f16]) -> Vec<i8> {
    let max_abs = vectors_f16
        .iter()
        .map(|x| x.to_f32().abs())
        .fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0_i8; vectors_f16.len()];
    }
    let scale = 127.0 / max_abs;
    vectors_f16
        .iter()
        .map(|x| {
            #[allow(clippy::cast_possible_truncation)]
            let q = (x.to_f32() * scale).round().clamp(-127.0, 127.0) as i8;
            q
        })
        .collect()
}

/// Portable scalar little-endian f16-byte int8 quantizer — the byte-slab fallback
/// and the A/B baseline for the mmap-backed FSVI path.
#[doc(hidden)]
#[must_use]
pub fn quantize_f16_le_bytes_to_i8_generic(bytes: &[u8]) -> Vec<i8> {
    let mut max_abs = 0.0_f32;
    for chunk in bytes.as_chunks::<2>().0 {
        let value = f16::from_le_bytes(*chunk).to_f32().abs();
        max_abs = max_abs.max(value);
    }
    if max_abs <= 0.0 {
        return vec![0_i8; bytes.len() / 2];
    }
    let scale = 127.0 / max_abs;
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| {
            #[allow(clippy::cast_possible_truncation)]
            let q = (f16::from_le_bytes(*chunk).to_f32() * scale)
                .round()
                .clamp(-127.0, 127.0) as i8;
            q
        })
        .collect()
}

/// Quantize one f32 to a signed 4-bit nibble (`[-7, 7]`, 4-bit two's complement in
/// the low 4 bits) given a scale — the per-element op inside the 4-bit slab build.
#[inline(always)]
fn nibble_of_4bit(value: f32, scale: f32) -> u8 {
    #[allow(clippy::cast_possible_truncation)]
    let q = (value * scale).round().clamp(-7.0, 7.0) as i8;
    q.cast_unsigned() & 0x0F
}

/// Pack a contiguous little-endian f16 byte slab into signed 4-bit nibbles with
/// one corpus-wide max-abs scale.
///
/// This is the mmap-backed twin of [`pack_f16_slab_to_4bit`]: FSVI stores f16
/// vectors as little-endian bytes, so the file-backed 4-bit two-pass path can use
/// the AVX2+F16C quantize-and-pack pipeline without materializing a `Vec<f16>`.
#[must_use]
pub fn pack_f16_le_bytes_to_4bit(bytes: &[u8], dim: usize) -> Vec<u8> {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above; on
            // little-endian x86, FSVI little-endian bytes are valid F16C inputs.
            #[allow(unsafe_code)]
            return unsafe { pack_f16_le_bytes_to_4bit_avx2(bytes, dim) };
        }
    }
    pack_f16_le_bytes_to_4bit_generic(bytes, dim)
}

/// Exact pre-SIMD-compaction AVX2 packer retained for same-binary performance
/// comparisons. Production callers should use [`pack_f16_le_bytes_to_4bit`].
#[doc(hidden)]
#[must_use]
pub fn pack_f16_le_bytes_to_4bit_scalar_pack(bytes: &[u8], dim: usize) -> Vec<u8> {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { pack_f16_le_bytes_to_4bit_avx2_impl::<false, true>(bytes, dim) };
        }
    }
    pack_f16_le_bytes_to_4bit_generic(bytes, dim)
}

/// Exact explicit-`vroundps` AVX2 path retained for same-binary performance
/// comparisons. Production callers should use [`pack_f16_le_bytes_to_4bit`].
#[doc(hidden)]
#[must_use]
pub fn pack_f16_le_bytes_to_4bit_explicit_round(bytes: &[u8], dim: usize) -> Vec<u8> {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { pack_f16_le_bytes_to_4bit_avx2_impl::<true, true>(bytes, dim) };
        }
    }
    pack_f16_le_bytes_to_4bit_generic(bytes, dim)
}

/// Pack a contiguous f16 slab (`count·dim`) into signed 4-bit nibbles (2 dims/byte,
/// `dim.div_ceil(2)` bytes/vector) with one corpus-wide max-abs scale.
///
/// This is the lazy
/// 4-bit ADC slab build (the **wired-default** two-pass pass-1 storage). Like the
/// int8 slab, it is decode-bound; runtime-dispatches to AVX2+F16C when available.
#[must_use]
pub fn pack_f16_slab_to_4bit(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { pack_f16_slab_to_4bit_avx2(vectors_f16, dim) };
        }
    }
    pack_f16_slab_to_4bit_generic(vectors_f16, dim)
}

/// Hand-written AVX2+F16C 4-bit packer over little-endian f16 bytes.
/// `vcvtph2ps` decodes 8 f16/instruction for both the max-abs pass and the
/// quantize pass; the nibble values come from the SAME `×scale` →
/// round-half-away → clamp(-7,7) → `cvttps_epi32` pipeline as the int8 kernel.
/// Eight quantized dimensions are compacted in-register with `vpshufb` and
/// become one 32-bit write. Bit-identical to the portable byte kernel
/// (`avx2_pack_4bit_bytes_matches_generic`).
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available and the target is
/// little-endian. The dispatch in [`pack_f16_le_bytes_to_4bit`] guarantees this.
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,f16c")]
#[allow(
    clippy::cast_ptr_alignment,
    clippy::many_single_char_names,
    unsafe_code
)]
#[must_use]
unsafe fn pack_f16_le_bytes_to_4bit_avx2_impl<
    const SIMD_NIBBLE_PACK: bool,
    const EXPLICIT_ROUND: bool,
>(
    bytes: &[u8],
    dim: usize,
) -> Vec<u8> {
    use core::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_ZERO, _mm_cvtsi128_si32, _mm_loadu_si128, _mm256_add_ps,
        _mm256_and_ps, _mm256_and_si256, _mm256_castsi256_si128, _mm256_cvtph_ps,
        _mm256_cvttps_epi32, _mm256_extracti128_si256, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps,
        _mm256_or_ps, _mm256_or_si256, _mm256_round_ps, _mm256_set_epi64x, _mm256_set1_epi8,
        _mm256_set1_epi16, _mm256_set1_ps, _mm256_setzero_ps, _mm256_shuffle_epi8,
        _mm256_srli_epi16, _mm256_storeu_ps, _mm256_storeu_si256,
    };
    if dim == 0 {
        return Vec::new();
    }
    let n = bytes.len() / 2;
    let total_chunks = n / 8;

    // Pass 1: corpus-wide max-abs (F16C decode + clear sign + vector max).
    let mut max_abs = 0.0_f32;
    // SAFETY: avx2+f16c by contract; loads are `c < total_chunks`-bounded.
    unsafe {
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
        let mut vmax = _mm256_setzero_ps();
        for c in 0..total_chunks {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(c * 16).cast()));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(x, abs_mask));
        }
        let mut arr = [0.0_f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), vmax);
        for &v in &arr {
            max_abs = max_abs.max(v);
        }
    }
    let mut tail = total_chunks * 16;
    while tail + 2 <= bytes.len() {
        let value = f16::from_le_bytes([bytes[tail], bytes[tail + 1]])
            .to_f32()
            .abs();
        max_abs = max_abs.max(value);
        tail += 2;
    }
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };

    // Pass 2: per-vector quantize → pack 2 dims/byte.
    let count = n / dim;
    let bytes_per_vector = dim.div_ceil(2);
    let mut slab = vec![0_u8; count * bytes_per_vector];
    // SAFETY: avx2+f16c by contract; within-vector loads are `d+8 ≤ dim`-bounded.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let vhalf = _mm256_set1_ps(0.5);
        let vmaxc = _mm256_set1_ps(7.0);
        let vminc = _mm256_set1_ps(-7.0);
        let vsign = _mm256_set1_ps(-0.0);
        // Each 128-bit lane selects the low byte from four i32 quantized lanes,
        // then compacts the two nibble-pair bytes from offsets 0 and 2.
        let zero_bytes = i64::from_le_bytes([0x80; 8]);
        let lane_select = i64::from_le_bytes([0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80]);
        let pair_select = i64::from_le_bytes([0, 2, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]);
        let lane_byte_mask = _mm256_set_epi64x(zero_bytes, lane_select, zero_bytes, lane_select);
        let pair_byte_mask = _mm256_set_epi64x(zero_bytes, pair_select, zero_bytes, pair_select);
        let nibble_mask = _mm256_set1_epi8(0x0f);
        let even_nibble_mask = _mm256_set1_epi16(0x000f);
        let mut tmp = [0_i32; 8];
        for v in 0..count {
            let base = v * dim;
            let out = v * bytes_per_vector;
            let mut d = 0;
            while d + 8 <= dim {
                let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add((base + d) * 2).cast()));
                let vv = _mm256_mul_ps(x, vscale);
                let half_signed = _mm256_or_ps(vhalf, _mm256_and_ps(vv, vsign));
                let biased = _mm256_add_ps(vv, half_signed);
                let clamp_input = if EXPLICIT_ROUND {
                    _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(biased)
                } else {
                    biased
                };
                let clamped = _mm256_min_ps(_mm256_max_ps(clamp_input, vminc), vmaxc);
                let quantized = _mm256_cvttps_epi32(clamped);
                if SIMD_NIBBLE_PACK {
                    let lane_bytes = _mm256_shuffle_epi8(quantized, lane_byte_mask);
                    let nibbles = _mm256_and_si256(lane_bytes, nibble_mask);
                    let pairs = _mm256_or_si256(
                        _mm256_and_si256(nibbles, even_nibble_mask),
                        _mm256_srli_epi16::<4>(nibbles),
                    );
                    let compact = _mm256_shuffle_epi8(pairs, pair_byte_mask);
                    #[allow(clippy::cast_sign_loss)]
                    let low = (_mm_cvtsi128_si32(_mm256_castsi256_si128(compact)) as u32) & 0xffff;
                    #[allow(clippy::cast_sign_loss)]
                    let high =
                        (_mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(compact)) as u32) & 0xffff;
                    let packed = low | (high << 16);
                    // SAFETY: `d + 8 <= dim` guarantees four output bytes remain;
                    // unaligned writes are valid for the byte-backed slab.
                    slab.as_mut_ptr()
                        .add(out + d / 2)
                        .cast::<u32>()
                        .write_unaligned(packed);
                } else {
                    _mm256_storeu_si256(tmp.as_mut_ptr().cast(), quantized);
                    // Exact pre-change path: stack round-trip + four scalar packs.
                    for m in 0..4 {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let lo = (tmp[2 * m] as u8) & 0x0F;
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let hi = (tmp[2 * m + 1] as u8) & 0x0F;
                        slab[out + d / 2 + m] = lo | (hi << 4);
                    }
                }
                d += 8;
            }
            // Scalar tail for the final < 8 dims (handles odd dim / last partial byte).
            while d < dim {
                let byte = (base + d) * 2;
                let value = f16::from_le_bytes([bytes[byte], bytes[byte + 1]]).to_f32();
                let nib = nibble_of_4bit(value, scale);
                if d % 2 == 0 {
                    slab[out + d / 2] |= nib;
                } else {
                    slab[out + d / 2] |= nib << 4;
                }
                d += 1;
            }
        }
    }
    slab
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
pub(super) unsafe fn pack_f16_le_bytes_to_4bit_avx2(bytes: &[u8], dim: usize) -> Vec<u8> {
    // SAFETY: inherited from this function's AVX2+F16C contract.
    unsafe { pack_f16_le_bytes_to_4bit_avx2_impl::<true, false>(bytes, dim) }
}

/// AVX2+F16C f16-slice adapter for [`pack_f16_slab_to_4bit`].
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
pub(super) unsafe fn pack_f16_slab_to_4bit_avx2(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    let n = vectors_f16.len();
    // SAFETY: `half::f16` is `repr(transparent)` over `u16`; `n` f16 = `n*2`
    // contiguous bytes, and x86_64 is little-endian.
    let bytes = unsafe { std::slice::from_raw_parts(vectors_f16.as_ptr().cast::<u8>(), n * 2) };
    // SAFETY: inherited from this function's AVX2+F16C contract.
    unsafe { pack_f16_le_bytes_to_4bit_avx2(bytes, dim) }
}

/// Portable little-endian f16-byte 4-bit packer — the mmap-backed fallback and
/// exact A/B baseline. It retains the shipped wide-SIMD decode while leaving the
/// round, clamp, and nibble packing scalar.
#[doc(hidden)]
#[must_use]
pub fn pack_f16_le_bytes_to_4bit_generic(bytes: &[u8], dim: usize) -> Vec<u8> {
    if dim == 0 {
        return Vec::new();
    }
    let n = bytes.len();
    let mut maxv = f32x8::splat(0.0);
    let mut i = 0;
    while i + 16 <= n {
        let values = widen8_f16_bytes(bytes[i..i + 16].try_into().expect("16 bytes"));
        maxv = maxv.max(values.abs());
        i += 16;
    }
    let mut max_abs = maxv.to_array().into_iter().fold(0.0_f32, f32::max);
    while i + 2 <= n {
        let value = f16::from_le_bytes([bytes[i], bytes[i + 1]]).to_f32().abs();
        max_abs = max_abs.max(value);
        i += 2;
    }
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let count = bytes.len() / (dim * 2);
    let bytes_per_vector = dim.div_ceil(2);
    let mut slab = vec![0_u8; count * bytes_per_vector];
    for v in 0..count {
        let base = v * dim * 2;
        let out = v * bytes_per_vector;
        let mut d = 0;
        while d + 8 <= dim {
            let offset = base + d * 2;
            let values = widen8_f16_bytes(bytes[offset..offset + 16].try_into().expect("16 bytes"));
            for (j, value) in values.to_array().into_iter().enumerate() {
                let dd = d + j;
                let nib = nibble_of_4bit(value, scale);
                if dd % 2 == 0 {
                    slab[out + dd / 2] |= nib;
                } else {
                    slab[out + dd / 2] |= nib << 4;
                }
            }
            d += 8;
        }
        while d < dim {
            let offset = base + d * 2;
            let value = f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32();
            let nib = nibble_of_4bit(value, scale);
            if d % 2 == 0 {
                slab[out + d / 2] |= nib;
            } else {
                slab[out + d / 2] |= nib << 4;
            }
            d += 1;
        }
    }
    slab
}

/// Portable scalar 4-bit slab packer — the AVX2+F16C-dispatch fallback. Exposed
/// (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn pack_f16_slab_to_4bit_generic(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    if dim == 0 {
        return Vec::new();
    }
    let max_abs = vectors_f16
        .iter()
        .map(|x| x.to_f32().abs())
        .fold(0.0_f32, f32::max);
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let count = vectors_f16.len() / dim;
    let bytes_per_vector = dim.div_ceil(2);
    let mut slab = vec![0_u8; count * bytes_per_vector];
    for v in 0..count {
        let base = v * dim;
        let out = v * bytes_per_vector;
        for d in 0..dim {
            let nib = nibble_of_4bit(vectors_f16[base + d].to_f32(), scale);
            if d % 2 == 0 {
                slab[out + d / 2] |= nib;
            } else {
                slab[out + d / 2] |= nib << 4;
            }
        }
    }
    slab
}

/// Encode an f32 slice to f16, appending to `dst` — the per-element `f32→f16`
/// conversion at the heart of every index build (`InMemoryVectorIndex::from_vectors`,
/// FSVI writes).
///
/// Runtime-dispatches to F16C `vcvtps2ph` (8 f16/instruction) when
/// available; the software `half::f16::from_f32` is the fallback.
pub fn encode_f32_to_f16_extend(src: &[f32], dst: &mut Vec<f16>) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above.
            #[allow(unsafe_code)]
            unsafe {
                encode_f32_to_f16_extend_avx2(src, dst);
            }
            return;
        }
    }
    encode_f32_to_f16_extend_generic(src, dst);
}

/// Hand-written AVX2+F16C f32→f16 encoder (`vcvtps2ph`, round-to-nearest-even).
/// Bit-identical to `half::f16::from_f32` for finite inputs (both IEEE
/// round-to-nearest-even); proven by `avx2_f16encode_matches_generic`.
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available (the dispatch in
/// [`encode_f32_to_f16_extend`] guarantees this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
pub(super) unsafe fn encode_f32_to_f16_extend_avx2(src: &[f32], dst: &mut Vec<f16>) {
    use core::arch::x86_64::{
        __m128i, _MM_FROUND_TO_NEAREST_INT, _mm_storeu_si128, _mm256_cvtps_ph, _mm256_loadu_ps,
    };
    let n = src.len();
    let chunks = n / 8;
    dst.reserve(n);
    let base = dst.len();
    // `half::f16` is `repr(transparent)` over `u16`, so we write the converted f16
    // bits straight into the Vec's spare capacity (no per-element push — that was
    // the bottleneck) and bump the length once.
    // SAFETY: `reserve(n)` guarantees `n` slots past `base`; `f16` and `u16` share
    // layout, so the `__m128i` (8 f16) store at `out + c*8` (`c*8+8 ≤ n`) is in
    // bounds; the scalar tail fills `[chunks*8, n)`; `set_len(base+n)` is then valid.
    unsafe {
        let out = dst.as_mut_ptr().add(base).cast::<u16>();
        for c in 0..chunks {
            let x = _mm256_loadu_ps(src.as_ptr().add(c * 8));
            let h = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(x);
            _mm_storeu_si128(out.add(c * 8).cast::<__m128i>(), h);
        }
        for (i, &v) in src.iter().enumerate().skip(chunks * 8) {
            *out.add(i) = f16::from_f32(v).to_bits();
        }
        dst.set_len(base + n);
    }
}

/// Portable scalar f32→f16 encoder — the F16C-dispatch fallback. Exposed
/// (doc-hidden) for the bench A/B.
#[doc(hidden)]
pub fn encode_f32_to_f16_extend_generic(src: &[f32], dst: &mut Vec<f16>) {
    dst.extend(src.iter().map(|&v| f16::from_f32(v)));
}
