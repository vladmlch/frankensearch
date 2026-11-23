use wide::{i16x8, i32x8};

/// Symmetric int8 dot product (both operands int8-quantized) returning the raw
/// `i32` inner product `Σ stored[i] * query[i]`.
///
/// This is the candidate **pass-1 kernel** for an int8 ADC two-pass scan (`bd-b5wl`):
/// quantized vectors are 1 byte/elem (half the bandwidth of f16) and the multiply
/// accumulates in integer lanes. `i16x8::widening_mul` keeps every product in full i32
/// precision, so the only overflow bound is the i32 accumulator (a 512-dim dot of
/// ±127 values peaks at ~8.3M, far below `i32::MAX`) — exact for any realistic dim.
///
/// Lengths are assumed equal (caller-guaranteed in the scan); a short tail is
/// handled scalar. Returns the raw integer dot; the caller applies the dequant
/// scale.
///
/// Dispatches to a hand-written AVX2 kernel at runtime when the CPU supports it:
/// the portable `wide` path only reaches AVX2 with a global `+avx2` build, which
/// the published binary can't assume (it would `SIGILL` on older hosts). Because
/// the dot is integer and integer addition is associative, the 256-bit `madd`
/// reduction is **bit-identical** to the generic kernel (asserted by
/// `avx2_dot_matches_generic`).
#[must_use]
pub fn dot_i8_i8(stored: &[i8], query: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe {
                if stored.len() == 384 && query.len() == 384 {
                    dot_i8_i8_avx2_384(stored, query)
                } else {
                    dot_i8_i8_avx2(stored, query)
                }
            };
        }
    }
    dot_i8_i8_generic(stored, query)
}

/// Four adjacent int8-vector dots with one shared query decode.
///
/// `stored_rows` contains four row-major vectors of `query.len()` elements each.
/// The AVX2 path loads and sign-extends every 32-byte query block once, then reuses
/// those lanes across all four stored rows. Each returned score is bit-identical
/// to a separate [`dot_i8_i8`] call.
#[inline(never)]
pub(super) fn dot_i8x4_i8(stored_rows: &[i8], query: &[i8]) -> [i32; 4] {
    let required = query
        .len()
        .checked_mul(4)
        .expect("four-row int8 dot length overflow");
    assert!(
        stored_rows.len() >= required,
        "four-row int8 dot requires {required} stored elements"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present and the caller supplies four full rows.
            #[allow(unsafe_code)]
            return unsafe { dot_i8x4_i8_avx2(stored_rows, query) };
        }
    }
    dot_i8x4_i8_generic(stored_rows, query)
}

pub(super) fn dot_i8x4_i8_generic(stored_rows: &[i8], query: &[i8]) -> [i32; 4] {
    let dim = query.len();
    if dim == 0 {
        return [0; 4];
    }
    let (row0, rest) = stored_rows.split_at(dim);
    let (row1, rest) = rest.split_at(dim);
    let (row2, row3) = rest.split_at(dim);
    [
        dot_i8_i8_generic(row0, query),
        dot_i8_i8_generic(row1, query),
        dot_i8_i8_generic(row2, query),
        dot_i8_i8_generic(&row3[..dim], query),
    ]
}

/// AVX2 four-row int8 dot. Query loads and i8-to-i16 sign extensions are shared
/// across the four independent row accumulators.
///
/// # Safety
/// The caller must ensure AVX2 is available and `stored_rows` contains at least
/// four complete `query.len()`-element rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
pub(super) unsafe fn dot_i8x4_i8_avx2(stored_rows: &[i8], query: &[i8]) -> [i32; 4] {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };

    macro_rules! reduce_i32x8 {
        ($acc:expr) => {{
            let sum128 = _mm_add_epi32(
                _mm256_castsi256_si128($acc),
                _mm256_extracti128_si256::<1>($acc),
            );
            let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
            let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
            _mm_cvtsi128_si32(sum32)
        }};
    }

    let n = query.len();
    debug_assert!(stored_rows.len() >= n.saturating_mul(4));
    // SAFETY: AVX2 is guaranteed by the caller. Each 32-byte query/stored load is
    // bounded by `i + 32 <= n`; row bases are within the four-row input slice.
    unsafe {
        let query_ptr = query.as_ptr();
        let stored_ptr = stored_rows.as_ptr();
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let mut i = 0_usize;

        while i + 32 <= n {
            let q = _mm256_loadu_si256(query_ptr.add(i).cast::<__m256i>());
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));

            macro_rules! accumulate_row {
                ($acc:ident, $row:expr) => {{
                    let s = _mm256_loadu_si256(stored_ptr.add(($row * n) + i).cast::<__m256i>());
                    let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
                    let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
                    let products = _mm256_add_epi32(
                        _mm256_madd_epi16(s_lo, q_lo),
                        _mm256_madd_epi16(s_hi, q_hi),
                    );
                    $acc = _mm256_add_epi32($acc, products);
                }};
            }

            accumulate_row!(acc0, 0);
            accumulate_row!(acc1, 1);
            accumulate_row!(acc2, 2);
            accumulate_row!(acc3, 3);
            i += 32;
        }

        let mut result = [
            reduce_i32x8!(acc0),
            reduce_i32x8!(acc1),
            reduce_i32x8!(acc2),
            reduce_i32x8!(acc3),
        ];
        while i < n {
            let q = i32::from(*query_ptr.add(i));
            result[0] += i32::from(*stored_ptr.add(i)) * q;
            result[1] += i32::from(*stored_ptr.add(n + i)) * q;
            result[2] += i32::from(*stored_ptr.add((2 * n) + i)) * q;
            result[3] += i32::from(*stored_ptr.add((3 * n) + i)) * q;
            i += 1;
        }
        result
    }
}

/// Four-row `vpmaddubs` int8 dot (bd-b5wl) — the batched twin of [`dot_i8x4_i8_avx2`] and the kernel
/// the row-blocked int8 scan uses. **APPROXIMATE** (saturating); see [`dot_i8_i8_avx2_maddubs`].
///
/// The batched form gains extra over the single-row kernel: `vpmaddubs` takes the i8 query directly,
/// so the shared query decode (2× `vpvmovsxbw` per 32 in [`dot_i8x4_i8_avx2`]) vanishes entirely, and
/// each of the 4 rows drops its 2× stored widening for one `vpxor` + `vpmaddubs`. `q_bias128 = 128·Σ
/// q_i` is subtracted once from every row (same query). Bit-exact to four [`dot_i8_i8_maddubs`] calls.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
unsafe fn dot_i8x4_i8_avx2_maddubs(stored_rows: &[i8], query: &[i8], q_bias128: i32) -> [i32; 4] {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi8, _mm256_set1_epi16,
        _mm256_setzero_si256, _mm256_xor_si256,
    };

    macro_rules! reduce_i32x8 {
        ($acc:expr) => {{
            let sum128 = _mm_add_epi32(
                _mm256_castsi256_si128($acc),
                _mm256_extracti128_si256::<1>($acc),
            );
            let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
            let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
            _mm_cvtsi128_si32(sum32)
        }};
    }

    let n = query.len();
    debug_assert!(stored_rows.len() >= n.saturating_mul(4));
    // SAFETY: AVX2 guaranteed by the caller. Each 32-byte load is `i + 32 <= n`-bounded; row bases
    // are within the four-row slice. Scalar tail accumulates in the u8 domain (see single-row).
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let flip = _mm256_set1_epi8(0x80_u8.cast_signed());
        let query_ptr = query.as_ptr();
        let stored_ptr = stored_rows.as_ptr();
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let mut i = 0_usize;

        while i + 32 <= n {
            let q = _mm256_loadu_si256(query_ptr.add(i).cast::<__m256i>());
            macro_rules! accumulate_row {
                ($acc:ident, $row:expr) => {{
                    let s = _mm256_loadu_si256(stored_ptr.add(($row * n) + i).cast::<__m256i>());
                    let u = _mm256_xor_si256(s, flip);
                    let prod = _mm256_maddubs_epi16(u, q);
                    $acc = _mm256_add_epi32($acc, _mm256_madd_epi16(prod, ones));
                }};
            }
            accumulate_row!(acc0, 0);
            accumulate_row!(acc1, 1);
            accumulate_row!(acc2, 2);
            accumulate_row!(acc3, 3);
            i += 32;
        }

        let mut acc_u = [
            reduce_i32x8!(acc0),
            reduce_i32x8!(acc1),
            reduce_i32x8!(acc2),
            reduce_i32x8!(acc3),
        ];
        while i < n {
            let q = i32::from(*query_ptr.add(i));
            acc_u[0] += (i32::from(*stored_ptr.add(i)) + 128) * q;
            acc_u[1] += (i32::from(*stored_ptr.add(n + i)) + 128) * q;
            acc_u[2] += (i32::from(*stored_ptr.add((2 * n) + i)) + 128) * q;
            acc_u[3] += (i32::from(*stored_ptr.add((3 * n) + i)) + 128) * q;
            i += 1;
        }
        [
            acc_u[0] - q_bias128,
            acc_u[1] - q_bias128,
            acc_u[2] - q_bias128,
            acc_u[3] - q_bias128,
        ]
    }
}

/// Batched approximate `vpmaddubs` int8 dot with AVX2 dispatch (bd-b5wl). Falls back to four exact
/// [`dot_i8_i8_generic`] calls off-AVX2. See [`dot_i8_i8_avx2_maddubs`] for the saturation caveat;
/// gate callers on recall, not bit-exactness. `q_bias128 = 128·Σ q_i`.
#[doc(hidden)]
#[must_use]
pub fn dot_i8x4_i8_maddubs(stored_rows: &[i8], query: &[i8], q_bias128: i32) -> [i32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { dot_i8x4_i8_avx2_maddubs(stored_rows, query, q_bias128) };
        }
    }
    let _ = q_bias128;
    let n = query.len();
    [
        dot_i8_i8_generic(&stored_rows[..n], query),
        dot_i8_i8_generic(&stored_rows[n..2 * n], query),
        dot_i8_i8_generic(&stored_rows[2 * n..3 * n], query),
        dot_i8_i8_generic(&stored_rows[3 * n..4 * n], query),
    ]
}

/// Hand-written AVX2 i8·i8 dot: 256-bit `vpmaddwd` over sign-extended i16 lanes,
/// two accumulators, horizontal-summed; scalar tail.
///
/// # Safety
/// The caller must ensure the `avx2` target feature is available; the runtime
/// dispatch in [`dot_i8_i8`] guarantees this.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
pub(super) unsafe fn dot_i8_i8_avx2(stored: &[i8], query: &[i8]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };
    let n = stored.len().min(query.len());
    // SAFETY: avx2 is guaranteed by the caller / `dot_i8_i8` dispatch; every load
    // is `i + 32 <= n`-bounded; the scalar tail covers `n % 32`.
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0_usize;
        while i + 32 <= n {
            let s = _mm256_loadu_si256(stored.as_ptr().add(i).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add(i).cast::<__m256i>());
            // Sign-extend low/high 16 i8 lanes to i16, then pairwise multiply-add.
            let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(s_lo, q_lo));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(s_hi, q_hi));
            i += 32;
        }
        let acc = _mm256_add_epi32(acc0, acc1);
        // Horizontal sum of the 8 i32 lanes.
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        let mut result = _mm_cvtsi128_si32(sum32);
        while i < n {
            result += i32::from(stored[i]) * i32::from(query[i]);
            i += 1;
        }
        result
    }
}

/// Fixed-shape AVX2 i8 dot for the production `MiniLM` dimension. Expanding the
/// twelve 32-byte blocks removes the dynamic length, loop control, and scalar
/// tail from every corpus-vector score while retaining the shipped two-lane
/// accumulation and exact integer reduction tree.
///
/// # Safety
/// The caller must ensure AVX2 is available and both slices contain exactly 384
/// elements. [`dot_i8_i8`] enforces both preconditions before dispatching here.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
unsafe fn dot_i8_i8_avx2_384(stored: &[i8], query: &[i8]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };

    macro_rules! accumulate32 {
        ($offset:expr, $acc0:ident, $acc1:ident) => {{
            let s = _mm256_loadu_si256(stored.as_ptr().add($offset).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add($offset).cast::<__m256i>());
            let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));
            $acc0 = _mm256_add_epi32($acc0, _mm256_madd_epi16(s_lo, q_lo));
            $acc1 = _mm256_add_epi32($acc1, _mm256_madd_epi16(s_hi, q_hi));
        }};
    }

    // SAFETY: the wrapper verifies AVX2 and exact 384-element inputs. Every
    // expanded load therefore stays within the two slices.
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        accumulate32!(0, acc0, acc1);
        accumulate32!(32, acc0, acc1);
        accumulate32!(64, acc0, acc1);
        accumulate32!(96, acc0, acc1);
        accumulate32!(128, acc0, acc1);
        accumulate32!(160, acc0, acc1);
        accumulate32!(192, acc0, acc1);
        accumulate32!(224, acc0, acc1);
        accumulate32!(256, acc0, acc1);
        accumulate32!(288, acc0, acc1);
        accumulate32!(320, acc0, acc1);
        accumulate32!(352, acc0, acc1);
        let acc = _mm256_add_epi32(acc0, acc1);
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        _mm_cvtsi128_si32(sum32)
    }
}

/// `vpmaddubs`-based i8·i8 dot (bd-b5wl route-next). **APPROXIMATE** — the pass-1 ADC scan.
///
/// The shipped [`dot_i8_i8_avx2`] sign-extends **both** operands (4× `vpmovsxbw` per 32 int8) before
/// `vpmaddwd`. `vpmaddubs` (`_mm256_maddubs_epi16`) multiply-adds a u8 operand against an i8 operand
/// directly, eliminating the *stored*-operand widening — the dominant memory traffic, since `stored`
/// is streamed once per row. Domain shift `u_i = s_i + 128` (a single `vpxor` with `0x80`, since
/// `i8 + 128 ≡ i8 ⊕ 0x80` in two's complement) makes `stored` unsigned; the identity
///
/// ```text
/// Σ s_i·q_i = Σ (u_i − 128)·q_i = maddubs_reduce(u, q) − 128·Σ q_i
/// ```
///
/// folds the bias into a per-query scalar `128·Σ q_i` (computed once by the caller and passed in).
///
/// **Why approximate:** `vpmaddubs` saturates each adjacent-pair sum to `i16` (`u8·i8` products reach
/// ±32 640, a pair-sum can exceed ±32 767). The shipped quantizer's global `127/max_abs` scale keeps
/// *most* int8-quantized-cosine components small (~±32) but leaves a tail at ±127 that **does**
/// saturate — so this kernel is approximate on real quantized data, not bit-exact. It is gated on
/// **recall**, never bit-exactness: `maddubs_pass1_preserves_f32_recall_under_real_saturation` proves
/// the pass-1 recall@k of the exact-f32 top-k is preserved despite the saturation, and the two-pass
/// scan exact-rescores the top `k·mult` candidates in f16. `q_bias128 = 128·Σ q_i`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
#[allow(clippy::many_single_char_names)] // s/q/u lane names are idiomatic kernel style
unsafe fn dot_i8_i8_avx2_maddubs(stored: &[i8], query: &[i8], q_bias128: i32) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi8, _mm256_set1_epi16,
        _mm256_setzero_si256, _mm256_xor_si256,
    };
    let n = stored.len().min(query.len());
    // SAFETY: avx2 guaranteed by the caller/dispatch; every load is `i + 32 <= n`-bounded; the
    // scalar tail covers `n % 32`. `q_bias128 = 128·Σ q_i` is supplied over the same `n`.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let flip = _mm256_set1_epi8(0x80_u8.cast_signed());
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0_usize;
        while i + 32 <= n {
            let s = _mm256_loadu_si256(stored.as_ptr().add(i).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add(i).cast::<__m256i>());
            // u = s + 128 (i8 -> u8 domain) via sign-bit flip; maddubs(u8, i8) -> i16 pair-sums
            // (SATURATING); widen those to i32 by multiply-add against ones.
            let u = _mm256_xor_si256(s, flip);
            let prod0 = _mm256_maddubs_epi16(u, q);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod0, ones));
            i += 32;
            if i + 32 <= n {
                let s = _mm256_loadu_si256(stored.as_ptr().add(i).cast::<__m256i>());
                let q = _mm256_loadu_si256(query.as_ptr().add(i).cast::<__m256i>());
                let u = _mm256_xor_si256(s, flip);
                let prod1 = _mm256_maddubs_epi16(u, q);
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(prod1, ones));
                i += 32;
            }
        }
        let acc = _mm256_add_epi32(acc0, acc1);
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        // Σ_aligned u_i·q_i in the u8 domain. The scalar tail must accumulate in the SAME u8 domain
        // — `(s_i + 128)·q_i` — so the single `128·Σq` bias (computed over all `n`) subtracts
        // exactly. Adding the tail in the signed domain would over-subtract by `128·Σ_tail q_i`.
        let mut acc_u = _mm_cvtsi128_si32(sum32);
        while i < n {
            acc_u += (i32::from(stored[i]) + 128) * i32::from(query[i]);
            i += 1;
        }
        acc_u - q_bias128
    }
}

/// Approximate `vpmaddubs` i8·i8 dot with runtime AVX2 dispatch (bd-b5wl). Falls back to the exact
/// [`dot_i8_i8_generic`] off-AVX2 or off-x86. See [`dot_i8_i8_avx2_maddubs`] for the saturation
/// caveat; callers must gate on recall, not bit-exactness. `q_bias128` is ignored by the fallback.
#[doc(hidden)]
#[must_use]
pub fn dot_i8_i8_maddubs(stored: &[i8], query: &[i8], q_bias128: i32) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { dot_i8_i8_avx2_maddubs(stored, query, q_bias128) };
        }
    }
    let _ = q_bias128;
    dot_i8_i8_generic(stored, query)
}

/// `128·Σ q_i` over the common prefix — the per-query bias for [`dot_i8_i8_maddubs`].
#[doc(hidden)]
#[must_use]
pub fn maddubs_query_bias(query: &[i8], dim: usize) -> i32 {
    let n = query.len().min(dim);
    128 * query[..n].iter().map(|&x| i32::from(x)).sum::<i32>()
}

/// Portable (`wide`-SIMD) i8·i8 dot — the AVX2-dispatch fallback and the path on
/// non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) so the `dot_product` bench
/// can A/B it against the AVX2 kernel.
#[doc(hidden)]
#[must_use]
pub fn dot_i8_i8_generic(stored: &[i8], query: &[i8]) -> i32 {
    // Four independent `i32x8` accumulators break the single-accumulator integer-add
    // dependency chain. The i8→i16 decode is a cheap sign-extend (unlike the
    // decode-bound f16 dot, where extra accumulators regress — see
    // docs/NEGATIVE_EVIDENCE.md), so this kernel is sum-chain-bound and the extra ILP
    // measured ~6% (dim256) to ~16% (dim384) faster. Integer sum is associative, so
    // the result is bit-identical to the prior single-accumulator kernel.
    #[inline(always)]
    fn w8<const O: usize>(a: &[i8; 32]) -> i16x8 {
        i16x8::from([
            i16::from(a[O]),
            i16::from(a[O + 1]),
            i16::from(a[O + 2]),
            i16::from(a[O + 3]),
            i16::from(a[O + 4]),
            i16::from(a[O + 5]),
            i16::from(a[O + 6]),
            i16::from(a[O + 7]),
        ])
    }
    let mut acc0 = i32x8::splat(0);
    let mut acc1 = i32x8::splat(0);
    let mut acc2 = i32x8::splat(0);
    let mut acc3 = i32x8::splat(0);

    let (s32, s_rem32) = stored.as_chunks::<32>();
    let (q32, q_rem32) = query.as_chunks::<32>();
    for (s, q) in s32.iter().zip(q32) {
        acc0 += w8::<0>(s).widening_mul(w8::<0>(q));
        acc1 += w8::<8>(s).widening_mul(w8::<8>(q));
        acc2 += w8::<16>(s).widening_mul(w8::<16>(q));
        acc3 += w8::<24>(s).widening_mul(w8::<24>(q));
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3);

    // Tail: remaining full 8-chunks, then a scalar remainder.
    let (s8, s_rem) = s_rem32.as_chunks::<8>();
    let (q8, q_rem) = q_rem32.as_chunks::<8>();
    for (s, q) in s8.iter().zip(q8) {
        sum += i16x8::from(s.map(i16::from)).widening_mul(i16x8::from(q.map(i16::from)));
    }
    let mut result = sum.reduce_add();
    for (s, q) in s_rem.iter().zip(q_rem) {
        result += i32::from(*s) * i32::from(*q);
    }
    result
}
