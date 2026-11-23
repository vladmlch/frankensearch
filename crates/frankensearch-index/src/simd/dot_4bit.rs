use wide::{i8x16, i16x16};

use super::PreparedQuery4bit;

/// Sign-extend the low nibble of a packed byte (4-bit two's complement → i8).
#[inline(always)]
fn nibble_lo(b: u8) -> i32 {
    i32::from(((b & 0x0F) ^ 0x08).cast_signed() - 8)
}

/// Sign-extend the high nibble of a packed byte (4-bit two's complement → i8).
#[inline(always)]
fn nibble_hi(b: u8) -> i32 {
    i32::from(((b >> 4) ^ 0x08).cast_signed() - 8)
}

#[cfg(target_arch = "x86_64")]
const FOUR_BIT_DIM384_BYTES: usize = 384 / 2;
#[cfg(target_arch = "x86_64")]
const FOUR_BIT_DIM384_CHUNKS: usize = FOUR_BIT_DIM384_BYTES / 16;

/// Pre-unpack a packed 4-bit query (the loop-invariant operand of a scan).
///
/// For each
/// 16-byte chunk, store the sign-extended low/high nibble lanes (`i16x16`); the
/// remainder bytes go to a scalar `(low, high)` tail.
#[must_use]
pub fn prepare_4bit_query(query: &[u8]) -> PreparedQuery4bit {
    let (chunks, remainder) = query.as_chunks::<16>();
    let mut low = Vec::with_capacity(query.len() / 16);
    let mut high = Vec::with_capacity(query.len() / 16);
    for qa in chunks {
        let q = i16x16::from_i8x16(i8x16::from(qa.map(u8::cast_signed)));
        low.push((q << 12_i32) >> 12_i32);
        high.push((q << 8_i32) >> 12_i32);
    }
    let tail = remainder
        .iter()
        .map(|&b| (nibble_lo(b), nibble_hi(b)))
        .collect();
    PreparedQuery4bit { low, high, tail }
}

/// Dot product of a packed 4-bit `stored` vector against a [`PreparedQuery4bit`].
///
/// The stored nibbles are decoded per call; the query was decoded once. Result is
/// exact (per-dim products ≤ 49). Identical to `dot_packed_4bit(stored, query)`.
///
/// SIMD: load 16 packed bytes → `i16x16`, extract nibbles via arithmetic
/// `(x<<12)>>12` / `(x<<8)>>12`, multiply by the prepared query lanes, and
/// accumulate vertically (flushing before any `i16` lane can overflow).
#[must_use]
pub fn dot_4bit_prepared(stored: &[u8], query: &PreparedQuery4bit) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe {
                if stored.len() == FOUR_BIT_DIM384_BYTES
                    && query.low.len() == FOUR_BIT_DIM384_CHUNKS
                    && query.high.len() == FOUR_BIT_DIM384_CHUNKS
                    && query.tail.is_empty()
                {
                    dot_4bit_prepared_avx2_384(stored, query)
                } else {
                    dot_4bit_prepared_avx2(stored, query)
                }
            };
        }
    }
    dot_4bit_prepared_generic(stored, query)
}

/// Exact pre-specialization dispatch retained for same-binary performance
/// comparisons. Production callers should use [`dot_4bit_prepared`].
#[doc(hidden)]
#[must_use]
pub fn dot_4bit_prepared_dynamic(stored: &[u8], query: &PreparedQuery4bit) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { dot_4bit_prepared_avx2(stored, query) };
        }
    }
    dot_4bit_prepared_generic(stored, query)
}

/// Hand-written AVX2 4-bit prepared dot: load 16 packed bytes, sign-extend to
/// i16, arithmetic-shift out the low/high nibbles, 256-bit `vpmullw` against the
/// prepared query lanes, accumulate in i16 (flushing every 16 chunks before any
/// lane can overflow), and reduce via `vpmaddwd`. Bit-identical to the portable
/// `wide` kernel (integer, in-range — proven by `avx2_dot4bit_matches_generic`).
///
/// # Safety
/// Caller must ensure the `avx2` target feature is available (the runtime
/// dispatch in [`dot_4bit_prepared`] guarantees this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
pub(super) unsafe fn dot_4bit_prepared_avx2(stored: &[u8], query: &PreparedQuery4bit) -> i32 {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_shuffle_epi32,
        _mm_unpackhi_epi64, _mm256_add_epi16, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mullo_epi16,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_srai_epi16,
    };
    let chunks = stored.len() / 16;
    let n = chunks.min(query.low.len());
    let mut sum = 0_i32;
    // SAFETY: avx2 guaranteed by the caller; every load is bounded by `i < n`
    // (`n ≤ stored.len()/16`); the scalar tail covers the remainder.
    unsafe {
        // Reduce 16 i16 lanes → i32 (`vpmaddwd` by ones, then hsum the 8 i32).
        let hsum16 = |acc: __m256i| -> i32 {
            let m = _mm256_madd_epi16(acc, _mm256_set1_epi16(1));
            let p128 = _mm_add_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256::<1>(m));
            let p64 = _mm_add_epi32(p128, _mm_unpackhi_epi64(p128, p128));
            let p32 = _mm_add_epi32(p64, _mm_shuffle_epi32::<0b01>(p64));
            _mm_cvtsi128_si32(p32)
        };
        let mut acc = _mm256_setzero_si256();
        let mut pending = 0_usize;
        for i in 0..n {
            let sbytes = _mm_loadu_si128(stored.as_ptr().add(i * 16).cast::<__m128i>());
            let s = _mm256_cvtepi8_epi16(sbytes);
            let s_low = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<12>(s));
            let s_high = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<8>(s));
            // The prepared query lanes are `i16x16` (32 bytes, lane order) — load
            // their bytes directly as `__m256i`.
            let q_low = _mm256_loadu_si256(core::ptr::from_ref(&query.low[i]).cast::<__m256i>());
            let q_high = _mm256_loadu_si256(core::ptr::from_ref(&query.high[i]).cast::<__m256i>());
            let prod = _mm256_add_epi16(
                _mm256_mullo_epi16(s_low, q_low),
                _mm256_mullo_epi16(s_high, q_high),
            );
            acc = _mm256_add_epi16(acc, prod);
            pending += 1;
            if pending == 16 {
                sum += hsum16(acc);
                acc = _mm256_setzero_si256();
                pending = 0;
            }
        }
        sum += hsum16(acc);
    }
    for (sb, &(qlo, qhi)) in stored[chunks * 16..].iter().zip(&query.tail) {
        sum += nibble_lo(*sb) * qlo + nibble_hi(*sb) * qhi;
    }
    sum
}

/// Fixed-shape AVX2 prepared 4-bit dot for the production `MiniLM` dimension.
/// Twelve expanded packed chunks fit in one vertical `i16` accumulator, so the
/// dynamic loop/flush/tail machinery disappears while the reduction tree stays
/// identical to [`dot_4bit_prepared_avx2`].
///
/// # Safety
/// The caller must ensure AVX2 is available, `stored` contains exactly 192
/// bytes, and the prepared query contains exactly twelve lanes with no tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
unsafe fn dot_4bit_prepared_avx2_384(stored: &[u8], query: &PreparedQuery4bit) -> i32 {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_shuffle_epi32,
        _mm_unpackhi_epi64, _mm256_add_epi16, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mullo_epi16,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_srai_epi16,
    };

    macro_rules! accumulate_chunk {
        ($chunk:expr, $acc:ident) => {{
            let sbytes = _mm_loadu_si128(stored.as_ptr().add($chunk * 16).cast::<__m128i>());
            let s = _mm256_cvtepi8_epi16(sbytes);
            let s_low = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<12>(s));
            let s_high = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<8>(s));
            let q_low = _mm256_loadu_si256(query.low.as_ptr().add($chunk).cast::<__m256i>());
            let q_high = _mm256_loadu_si256(query.high.as_ptr().add($chunk).cast::<__m256i>());
            let prod = _mm256_add_epi16(
                _mm256_mullo_epi16(s_low, q_low),
                _mm256_mullo_epi16(s_high, q_high),
            );
            $acc = _mm256_add_epi16($acc, prod);
        }};
    }

    // SAFETY: the dispatch wrapper verifies the exact 192-byte/twelve-chunk
    // shape and AVX2. Twelve chunks accumulate at most 12*98=1176 per i16 lane.
    unsafe {
        let mut acc = _mm256_setzero_si256();
        accumulate_chunk!(0, acc);
        accumulate_chunk!(1, acc);
        accumulate_chunk!(2, acc);
        accumulate_chunk!(3, acc);
        accumulate_chunk!(4, acc);
        accumulate_chunk!(5, acc);
        accumulate_chunk!(6, acc);
        accumulate_chunk!(7, acc);
        accumulate_chunk!(8, acc);
        accumulate_chunk!(9, acc);
        accumulate_chunk!(10, acc);
        accumulate_chunk!(11, acc);
        let pairs = _mm256_madd_epi16(acc, _mm256_set1_epi16(1));
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(pairs),
            _mm256_extracti128_si256::<1>(pairs),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        _mm_cvtsi128_si32(sum32)
    }
}

/// Portable (`wide`-SIMD) 4-bit prepared dot — the AVX2-dispatch fallback and the
/// path on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_4bit_prepared_generic(stored: &[u8], query: &PreparedQuery4bit) -> i32 {
    let mut sum = 0_i32;
    let mut acc = i16x16::splat(0);
    let mut pending = 0_usize;
    let (s16, s_rem) = stored.as_chunks::<16>();
    for (sa, (q_low, q_high)) in s16.iter().zip(query.low.iter().zip(&query.high)) {
        let s = i16x16::from_i8x16(i8x16::from(sa.map(u8::cast_signed)));
        let s_low = (s << 12_i32) >> 12_i32;
        let s_high = (s << 8_i32) >> 12_i32;
        acc += s_low * *q_low + s_high * *q_high;
        pending += 1;
        if pending == 16 {
            sum += i32::from(acc.reduce_add());
            acc = i16x16::splat(0);
            pending = 0;
        }
    }
    sum += i32::from(acc.reduce_add());
    for (sb, &(qlo, qhi)) in s_rem.iter().zip(&query.tail) {
        sum += nibble_lo(*sb) * qlo + nibble_hi(*sb) * qhi;
    }
    sum
}

/// Dot product of two vectors stored as packed signed 4-bit nibbles.
///
/// Layout: 2 dims per
/// byte, low nibble = even dim, high nibble = odd dim, each a 4-bit two's
/// complement in `[-7, 7]`. `stored` and `query` must have equal packed length.
/// Result is exact (per-dim products ≤ 49). For a scan over many stored vectors,
/// prefer [`prepare_4bit_query`] + [`dot_4bit_prepared`] to decode the query once.
#[must_use]
pub fn dot_packed_4bit(stored: &[u8], query: &[u8]) -> i32 {
    dot_4bit_prepared(stored, &prepare_4bit_query(query))
}
