use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use wide::f32x8;

use super::{decode8_f32, ensure_same_len, load8_f32, widen8_f16_bytes, widen8_f16_slice};

/// Dot product between two f32 vectors.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn dot_product_f32_f32(a: &[f32], b: &[f32]) -> SearchResult<f32> {
    ensure_same_len(a.len(), b.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present; lengths checked equal above.
            #[allow(unsafe_code)]
            return Ok(unsafe { dot_product_f32_f32_avx2(a, b) });
        }
    }
    Ok(dot_product_f32_f32_generic(a, b))
}

/// Hand-written AVX2 f32·f32 dot — 256-bit `vmulps`/`vaddps` mirroring the `wide`
/// kernel's 4 accumulators, `(acc0+acc1)+(acc2+acc3)` reduction (routed through
/// `f32x8::reduce_add`), 8-chunk tail, and separate-mul+add scalar tail →
/// **bit-identical** (`avx2_f32slicedot_matches_generic`). f32 has no decode win,
/// so the gain is purely the 256-bit width over the `wide` SSE2-default path.
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in [`dot_product_f32_f32`]
/// guarantees it) and `a.len() == b.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::many_single_char_names)] // a/b lane/pointer names are idiomatic kernel style
pub(super) unsafe fn dot_product_f32_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let n = a.len().min(b.len());
    let groups = n / 32;
    let chunks = n / 8;
    let mut acc_arr = [0.0_f32; 8];
    // SAFETY: avx2 by contract; every load is group/chunk-bounded by `o+8 ≤ n`.
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for g in 0..groups {
            let o = g * 32;
            acc0 = _mm256_add_ps(
                acc0,
                _mm256_mul_ps(_mm256_loadu_ps(ap.add(o)), _mm256_loadu_ps(bp.add(o))),
            );
            acc1 = _mm256_add_ps(
                acc1,
                _mm256_mul_ps(
                    _mm256_loadu_ps(ap.add(o + 8)),
                    _mm256_loadu_ps(bp.add(o + 8)),
                ),
            );
            acc2 = _mm256_add_ps(
                acc2,
                _mm256_mul_ps(
                    _mm256_loadu_ps(ap.add(o + 16)),
                    _mm256_loadu_ps(bp.add(o + 16)),
                ),
            );
            acc3 = _mm256_add_ps(
                acc3,
                _mm256_mul_ps(
                    _mm256_loadu_ps(ap.add(o + 24)),
                    _mm256_loadu_ps(bp.add(o + 24)),
                ),
            );
        }
        let mut sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut c = groups * 4;
        while c < chunks {
            let o = c * 8;
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(_mm256_loadu_ps(ap.add(o)), _mm256_loadu_ps(bp.add(o))),
            );
            c += 1;
        }
        _mm256_storeu_ps(acc_arr.as_mut_ptr(), sum);
    }
    let mut result = f32x8::from(acc_arr).reduce_add();
    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }
    result
}

/// Dot product between an f16 stored vector and an f32 query vector.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn dot_product_f16_f32(stored: &[f16], query: &[f32]) -> SearchResult<f32> {
    ensure_same_len(stored.len(), query.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present; lengths checked equal above.
            #[allow(unsafe_code)]
            return Ok(unsafe { dot_product_f16_f32_avx2(stored, query) });
        }
    }
    Ok(dot_product_f16_f32_generic(stored, query))
}

/// Hand-written AVX2+F16C f16-slice·f32 dot — same `vcvtph2ps` hardware decode +
/// `reduce`-through-`wide` trick as [`dot_product_f16_bytes_f32`], so it is
/// **bit-identical** to the portable kernel (the slice tail is separate mul+add,
/// matched here; proven by `avx2_f16slicedot_matches_generic`).
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available (the dispatch in
/// [`dot_product_f16_f32`] guarantees this) and `stored.len() == query.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
pub(super) unsafe fn dot_product_f16_f32_avx2(stored: &[f16], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let n = stored.len().min(query.len());
    let chunks = n / 8;
    let mut arr = [0.0_f32; 8];
    // SAFETY: avx2+f16c by contract; loads are `c < chunks`-bounded; `&[f16]` is
    // contiguous little-endian f16 (2 bytes/elem), so a 16-byte load is 8 f16.
    // Four independent accumulators break the per-iteration `vaddps` dependency
    // chain (the kernel is ~4-cycle-add-latency-bound, not decode-throughput-bound
    // at one accumulator); the `(s0+s1)+(s2+s3)` tree + grouped chunk→lane mapping
    // is matched bit-for-bit in `dot_product_f16_f32_generic`.
    macro_rules! mul_chunk {
        ($c:expr) => {{
            let f16bits = _mm_loadu_si128(stored.as_ptr().add($c * 8).cast::<__m128i>());
            let s = _mm256_cvtph_ps(f16bits);
            let q = _mm256_loadu_ps(query.as_ptr().add($c * 8));
            _mm256_mul_ps(s, q)
        }};
    }
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut c = 0;
        while c + 4 <= chunks {
            s0 = _mm256_add_ps(s0, mul_chunk!(c));
            s1 = _mm256_add_ps(s1, mul_chunk!(c + 1));
            s2 = _mm256_add_ps(s2, mul_chunk!(c + 2));
            s3 = _mm256_add_ps(s3, mul_chunk!(c + 3));
            c += 4;
        }
        while c < chunks {
            s0 = _mm256_add_ps(s0, mul_chunk!(c));
            c += 1;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    }
    let mut result = f32x8::from(arr).reduce_add();
    for index in (chunks * 8)..n {
        result += stored[index].to_f32() * query[index];
    }
    result
}

/// Portable (`wide`-SIMD) f16-slice·f32 dot — the AVX2+F16C-dispatch fallback and
/// the path on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f16_f32_generic(stored: &[f16], query: &[f32]) -> f32 {
    let n = stored.len().min(query.len());
    let chunks = n / 8;
    // Four independent accumulators with the `(s0+s1)+(s2+s3)` tree — bit-for-bit
    // the grouped chunk→lane mapping of `dot_product_f16_f32_avx2`.
    let prod = |c: usize| -> f32x8 {
        let s: &[f16; 8] = stored[c * 8..c * 8 + 8].try_into().expect("8 f16");
        let q: &[f32; 8] = query[c * 8..c * 8 + 8].try_into().expect("8 f32");
        widen8_f16_slice(s) * f32x8::from(*q)
    };
    let mut s0 = f32x8::splat(0.0);
    let mut s1 = f32x8::splat(0.0);
    let mut s2 = f32x8::splat(0.0);
    let mut s3 = f32x8::splat(0.0);
    let mut c = 0;
    while c + 4 <= chunks {
        s0 += prod(c);
        s1 += prod(c + 1);
        s2 += prod(c + 2);
        s3 += prod(c + 3);
        c += 4;
    }
    while c < chunks {
        s0 += prod(c);
        c += 1;
    }
    let mut result = ((s0 + s1) + (s2 + s3)).reduce_add();
    for index in (chunks * 8)..n {
        result += stored[index].to_f32() * query[index];
    }
    result
}

/// Cosine similarity helper for f16 stored vectors.
///
/// Assumes both vectors are already L2-normalized and therefore returns the
/// raw dot product value.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn cosine_similarity_f16(stored: &[f16], query: &[f32]) -> SearchResult<f32> {
    dot_product_f16_f32(stored, query)
}

/// Dot product between f16 bytes and an f32 query vector.
///
/// Avoids intermediate allocation by decoding f16s on the fly.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when `stored_bytes.len()` is not
/// exactly `query.len() * 2`.
pub fn dot_product_f16_bytes_f32(stored_bytes: &[u8], query: &[f32]) -> SearchResult<f32> {
    let dim = query.len();
    if stored_bytes.len() != dim * 2 {
        return Err(SearchError::DimensionMismatch {
            expected: dim,
            found: stored_bytes.len() / 2,
        });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above;
            // `stored_bytes.len() == dim*2` checked above.
            #[allow(unsafe_code)]
            return Ok(unsafe { dot_product_f16_bytes_f32_avx2(stored_bytes, query) });
        }
    }
    Ok(dot_product_f16_bytes_f32_generic(stored_bytes, query))
}

/// Hand-written AVX2+F16C f16·f32 dot. `vcvtph2ps` (`_mm256_cvtph_ps`) decodes 8
/// f16 in one instruction — the portable `wide` decode (`widen8_f16_bytes`) is
/// software, and this kernel is decode-bound — then the **same** separate-mul+add
/// f32 accumulation, the **same** `wide::f32x8::reduce_add` final reduction (the
/// 256-bit accumulator is round-tripped through `f32x8`), and the **same** scalar
/// `mul_add` tail as the generic kernel. f16→f32 is exact (f32 has more mantissa
/// bits), so the products / accumulation / reduction are byte-for-byte the generic
/// path → **bit-identical** (proven by `avx2_f16dot_matches_generic`).
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available (the dispatch in
/// [`dot_product_f16_bytes_f32`] guarantees this) and `stored_bytes.len() == query.len()*2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
pub(super) unsafe fn dot_product_f16_bytes_f32_avx2(stored_bytes: &[u8], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let dim = query.len();
    let chunks = dim / 8;
    let mut arr = [0.0_f32; 8];
    // Four independent accumulators break the single-`vaddps` dependency chain
    // (latency-bound at one accumulator); grouped chunk→lane mapping + `(s0+s1)+
    // (s2+s3)` tree matched bit-for-bit in `dot_product_f16_bytes_f32_generic`.
    macro_rules! mul_chunk {
        ($c:expr) => {{
            let f16bits = _mm_loadu_si128(stored_bytes.as_ptr().add($c * 16).cast::<__m128i>());
            let stored = _mm256_cvtph_ps(f16bits);
            let q = _mm256_loadu_ps(query.as_ptr().add($c * 8));
            _mm256_mul_ps(stored, q)
        }};
    }
    // SAFETY: avx2+f16c by contract; every load is `chunk_index < chunks`-bounded.
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut c = 0;
        while c + 4 <= chunks {
            s0 = _mm256_add_ps(s0, mul_chunk!(c));
            s1 = _mm256_add_ps(s1, mul_chunk!(c + 1));
            s2 = _mm256_add_ps(s2, mul_chunk!(c + 2));
            s3 = _mm256_add_ps(s3, mul_chunk!(c + 3));
            c += 4;
        }
        while c < chunks {
            s0 = _mm256_add_ps(s0, mul_chunk!(c));
            c += 1;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    }
    // Final reduce + scalar tail are byte-for-byte the generic path.
    let mut result = f32x8::from(arr).reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        let val = f16::from_le_bytes([b[0], b[1]]).to_f32();
        result = val.mul_add(query[index], result);
    }
    result
}

/// `vfmadd`-based f16-bytes·f32 dot — **REJECTED bench-only candidate**, retained so the negative
/// result stays reproducible (`docs/NEGATIVE_EVIDENCE.md`, 2026-07-10). Same 4-accumulator structure
/// and F16C decode as [`dot_product_f16_bytes_f32_avx2`], but the SIMD main loop fuses `stored·q +
/// acc` into one `_mm256_fmadd_ps` instead of a separate `_mm256_mul_ps` + `_mm256_add_ps`. It is
/// sub-ULP-different but order-preserving (`fma_f16_dot_is_ulp_close_and_order_preserving`) — so it
/// *could* have shipped. It does not: the kernel is **`cvtph2ps`-decode-bound**, not FP-port-bound,
/// so fusing the arithmetic buys nothing. Null-controlled microbench: fma/ORIG median **1.0351**,
/// inside the null floor [0.9140, 1.0488] (if anything marginally slower). Do not wire into the
/// scan.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
unsafe fn dot_product_f16_bytes_f32_fma_avx2(stored_bytes: &[u8], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm256_add_ps, _mm256_cvtph_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let dim = query.len();
    let chunks = dim / 8;
    let mut arr = [0.0_f32; 8];
    macro_rules! fma_chunk {
        ($c:expr, $acc:expr) => {{
            let f16bits = _mm_loadu_si128(stored_bytes.as_ptr().add($c * 16).cast::<__m128i>());
            let stored = _mm256_cvtph_ps(f16bits);
            let q = _mm256_loadu_ps(query.as_ptr().add($c * 8));
            $acc = _mm256_fmadd_ps(stored, q, $acc);
        }};
    }
    // SAFETY: avx2+f16c+fma by contract; every load is `chunk_index < chunks`-bounded.
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut c = 0;
        while c + 4 <= chunks {
            fma_chunk!(c, s0);
            fma_chunk!(c + 1, s1);
            fma_chunk!(c + 2, s2);
            fma_chunk!(c + 3, s3);
            c += 4;
        }
        while c < chunks {
            fma_chunk!(c, s0);
            c += 1;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    }
    let mut result = f32x8::from(arr).reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        let val = f16::from_le_bytes([b[0], b[1]]).to_f32();
        result = val.mul_add(query[index], result);
    }
    result
}

/// FMA f16-bytes·f32 dot with AVX2+F16C+FMA dispatch (scan lever). Falls back to the exact
/// [`dot_product_f16_bytes_f32_generic`] off-dispatch. See [`dot_product_f16_bytes_f32_fma_avx2`] for
/// the sub-ULP (ranking-safe) caveat.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f16_bytes_f32_fma(stored_bytes: &[u8], query: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("f16c")
            && std::is_x86_feature_detected!("fma")
        {
            // SAFETY: avx2+f16c+fma verified present by the runtime checks above.
            #[allow(unsafe_code)]
            return unsafe { dot_product_f16_bytes_f32_fma_avx2(stored_bytes, query) };
        }
    }
    dot_product_f16_bytes_f32_generic(stored_bytes, query)
}

/// Portable (`wide`-SIMD) f16-bytes·f32 dot — the AVX2+F16C-dispatch fallback and
/// the path on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f16_bytes_f32_generic(stored_bytes: &[u8], query: &[f32]) -> f32 {
    let dim = query.len();
    let chunks = dim / 8;
    // Four independent accumulators with the `(s0+s1)+(s2+s3)` tree — bit-for-bit
    // the grouped chunk→lane mapping of `dot_product_f16_bytes_f32_avx2`.
    let prod = |c: usize| -> f32x8 {
        let block: &[u8; 16] = stored_bytes[c * 16..c * 16 + 16]
            .try_into()
            .expect("16-byte f16 block");
        let q: &[f32; 8] = query[c * 8..c * 8 + 8]
            .try_into()
            .expect("8-element query block");
        widen8_f16_bytes(block) * f32x8::from(*q)
    };
    let mut s0 = f32x8::splat(0.0);
    let mut s1 = f32x8::splat(0.0);
    let mut s2 = f32x8::splat(0.0);
    let mut s3 = f32x8::splat(0.0);
    let mut c = 0;
    while c + 4 <= chunks {
        s0 += prod(c);
        s1 += prod(c + 1);
        s2 += prod(c + 2);
        s3 += prod(c + 3);
        c += 4;
    }
    while c < chunks {
        s0 += prod(c);
        c += 1;
    }

    let mut result = ((s0 + s1) + (s2 + s3)).reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        let val = f16::from_le_bytes([b[0], b[1]]).to_f32();
        result = val.mul_add(query[index], result);
    }

    result
}

/// Dot product between f32 bytes and an f32 query vector.
///
/// Avoids intermediate allocation by decoding f32s on the fly.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when `stored_bytes.len()` is not
/// exactly `query.len() * 4`.
pub fn dot_product_f32_bytes_f32(stored_bytes: &[u8], query: &[f32]) -> SearchResult<f32> {
    let dim = query.len();
    if stored_bytes.len() != dim * 4 {
        return Err(SearchError::DimensionMismatch {
            expected: dim,
            found: stored_bytes.len() / 4,
        });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present; `stored_bytes.len() == dim*4` checked.
            #[allow(unsafe_code)]
            return Ok(unsafe { dot_product_f32_bytes_f32_avx2(stored_bytes, query) });
        }
    }
    Ok(dot_product_f32_bytes_f32_generic(stored_bytes, query))
}

/// Hand-written AVX2 f32-bytes·f32 dot — 256-bit `vmulps`/`vaddps` with the SAME
/// 4-accumulator grouping, `(acc0+acc1)+(acc2+acc3)` reduction, `wide` 8-chunk
/// tail (routed through `f32x8::reduce_add`), and `mul_add` scalar tail as the
/// portable kernel. f32 LE bytes are native on x86, so an unaligned `loadu_ps` of
/// the stored bytes yields the same lanes as `decode8_f32` → **bit-identical**
/// (proven by `avx2_f32dot_matches_generic`). f32 has no decode win, so the gain
/// is purely the 256-bit width over the `wide` SSE2-default path.
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in
/// [`dot_product_f32_bytes_f32`] guarantees it) and `stored_bytes.len() == query.len()*4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
#[allow(clippy::cast_ptr_alignment)] // unaligned loadu/storeu intrinsics
pub(super) unsafe fn dot_product_f32_bytes_f32_avx2(stored_bytes: &[u8], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let dim = query.len();
    let groups = dim / 32;
    let chunks = dim / 8;
    let mut acc_arr = [0.0_f32; 8];
    // SAFETY: avx2 by contract; `stored_bytes.len() == dim*4`, so a `*const f32`
    // load of 8 floats at element offset o (o+8 ≤ dim) is in-bounds; `loadu_ps`
    // is unaligned. Every offset below is group/chunk-bounded by `o+8 ≤ dim`.
    unsafe {
        let sptr = stored_bytes.as_ptr().cast::<f32>();
        let qptr = query.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for g in 0..groups {
            let o = g * 32;
            acc0 = _mm256_add_ps(
                acc0,
                _mm256_mul_ps(_mm256_loadu_ps(sptr.add(o)), _mm256_loadu_ps(qptr.add(o))),
            );
            acc1 = _mm256_add_ps(
                acc1,
                _mm256_mul_ps(
                    _mm256_loadu_ps(sptr.add(o + 8)),
                    _mm256_loadu_ps(qptr.add(o + 8)),
                ),
            );
            acc2 = _mm256_add_ps(
                acc2,
                _mm256_mul_ps(
                    _mm256_loadu_ps(sptr.add(o + 16)),
                    _mm256_loadu_ps(qptr.add(o + 16)),
                ),
            );
            acc3 = _mm256_add_ps(
                acc3,
                _mm256_mul_ps(
                    _mm256_loadu_ps(sptr.add(o + 24)),
                    _mm256_loadu_ps(qptr.add(o + 24)),
                ),
            );
        }
        let mut sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut c = groups * 4;
        while c < chunks {
            let o = c * 8;
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(_mm256_loadu_ps(sptr.add(o)), _mm256_loadu_ps(qptr.add(o))),
            );
            c += 1;
        }
        _mm256_storeu_ps(acc_arr.as_mut_ptr(), sum);
    }
    let mut result = f32x8::from(acc_arr).reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 4..];
        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        result = val.mul_add(query[index], result);
    }
    result
}

/// Portable (`wide`-SIMD) f32-bytes·f32 dot — the AVX2-dispatch fallback and the
/// path on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f32_bytes_f32_generic(stored_bytes: &[u8], query: &[f32]) -> f32 {
    let dim = query.len();
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);

    // Main loop: 32 elements (128 stored bytes) per iteration, 4 accumulators.
    let groups = dim / 32;
    for g in 0..groups {
        let bo = g * 128;
        let qo = g * 32;
        let block: &[u8; 128] = stored_bytes[bo..bo + 128]
            .try_into()
            .expect("128-byte f32 block");
        let q: &[f32; 32] = query[qo..qo + 32]
            .try_into()
            .expect("32-element query block");
        acc0 +=
            decode8_f32(block[0..32].try_into().expect("32-byte sub-block")) * load8_f32::<0>(q);
        acc1 +=
            decode8_f32(block[32..64].try_into().expect("32-byte sub-block")) * load8_f32::<8>(q);
        acc2 +=
            decode8_f32(block[64..96].try_into().expect("32-byte sub-block")) * load8_f32::<16>(q);
        acc3 +=
            decode8_f32(block[96..128].try_into().expect("32-byte sub-block")) * load8_f32::<24>(q);
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);

    // Tail of full 8-element chunks not covered by the 32-wide main loop.
    let chunks = dim / 8;
    for chunk_index in (groups * 4)..chunks {
        let bo = chunk_index * 32;
        let qo = chunk_index * 8;
        let block: &[u8; 32] = stored_bytes[bo..bo + 32].try_into().expect("32-byte block");
        let q: &[f32; 8] = query[qo..qo + 8].try_into().expect("8-element block");
        sum += decode8_f32(block) * f32x8::from(*q);
    }

    let mut result = sum.reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 4..];
        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        result = val.mul_add(query[index], result);
    }

    result
}

/// Portable (`wide`-SIMD) f32·f32 dot — the AVX2-dispatch fallback and the path
/// on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f32_f32_generic(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);

    let (a32, a_rem32) = a.as_chunks::<32>();
    let (b32, b_rem32) = b.as_chunks::<32>();
    for (a_block, b_block) in a32.iter().zip(b32) {
        acc0 += load8_f32::<0>(a_block) * load8_f32::<0>(b_block);
        acc1 += load8_f32::<8>(a_block) * load8_f32::<8>(b_block);
        acc2 += load8_f32::<16>(a_block) * load8_f32::<16>(b_block);
        acc3 += load8_f32::<24>(a_block) * load8_f32::<24>(b_block);
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    let (a8, a_rem) = a_rem32.as_chunks::<8>();
    let (b8, b_rem) = b_rem32.as_chunks::<8>();

    for (a_block, b_block) in a8.iter().zip(b8) {
        sum += f32x8::from(*a_block) * f32x8::from(*b_block);
    }

    let mut result = sum.reduce_add();
    for (x, y) in a_rem.iter().zip(b_rem) {
        result += x * y;
    }
    result
}
