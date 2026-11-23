use ft_core::{DType, Device, TensorMeta};
use wide::f32x8;

#[cfg(test)]
use super::certificate_trace;
use super::{CLS_Q_CACHE_MIN_SEQ, H, HD, NH};

// Adapted from wide 1.7.0's f32x8::exp, polynomial_5 and vm_pow2n.
// Alteration: explicitly separate multiplication and addition on every target;
// construct exponent bits through safe scalar lane access.
//
// Copyright (c) 2020 Daniel "Lokathor" Gee.
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
// claim that you wrote the original software. If you use this software
// in a product, an acknowledgment in the product documentation would be
// appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
// misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

/// Preserve the non-FMA x86 exponential's rounding on every vector target.
/// `wide::exp` uses fused operations on NEON, which changes model output bits.
/// Keep this expression tree: reassociation or `mul_add` changes the producer.
#[inline]
pub(super) fn exp_vec8_non_fused(input: f32x8) -> f32x8 {
    fn pow2(exponents: f32x8) -> f32x8 {
        f32x8::new(exponents.to_array().map(|n| {
            debug_assert!((-150.0..=127.0).contains(&n));
            if n < -149.0 {
                0.0
            } else if n < -126.0 {
                // Adding 2^23 encodes an integral shift (0..=22) in the
                // mantissa, without a float-to-integer conversion.
                let shift = (n + 149.0 + 8_388_608.0).to_bits() & 0x007f_ffff;
                f32::from_bits(1_u32 << shift)
            } else {
                f32::from_bits((n + (127.0 + 8_388_608.0)).to_bits() << 23)
            }
        }))
    }

    let lanes = input.to_array();
    // Exceptional lanes have fixed outputs. Sanitize them before reduction so
    // the exponent construction only receives finite, bounded integers.
    let bounded = f32x8::new(lanes.map(|x| {
        if (-103.63..=88.723).contains(&x) {
            x
        } else {
            0.0
        }
    }));
    let exponent = (bounded * f32x8::LOG2_E).round_ties_even();
    let max_exponent = f32x8::splat(127.0);
    let scale = pow2((exponent - max_exponent).max(f32x8::ZERO));
    let power = pow2(exponent.min(max_exponent));
    let reduced = bounded - exponent * f32x8::splat(0.693_359_4);
    let reduced = reduced - exponent * f32x8::splat(-2.121_944_4e-4);
    let squared = reduced * reduced;
    let fourth = squared * squared;
    let pair_23 = f32x8::splat(1.0 / 120.0) * reduced + f32x8::splat(1.0 / 24.0);
    let pair_45 = f32x8::splat(1.0 / 5040.0) * reduced + f32x8::splat(1.0 / 720.0);
    let pair_01 = f32x8::splat(1.0 / 6.0) * reduced + f32x8::splat(1.0 / 2.0);
    let polynomial = pair_23 * squared + (pair_45 * fourth + pair_01);
    let increment = polynomial * squared + reduced;
    let result = ((increment + f32x8::ONE) * scale * power).to_array();
    f32x8::new(std::array::from_fn(|i| {
        if lanes[i].is_nan() {
            f32::from_bits(0x7fc0_0101)
        } else if lanes[i] > 88.723 {
            f32::INFINITY
        } else if lanes[i] < -103.63 {
            0.0
        } else {
            result[i]
        }
    }))
}

// Adapted from Arm optimized-routines expf.c and exp2f_data.c at
// f2e4faf58c6c671154f472a76eaaa977bb36c870. Modified to spell out the five
// fused operations used by the validated GNU x86 producer, including range
// reduction, rather than relying on compiler contraction or host libm.
// The final f32 rounding is part of this scalar producer's arithmetic.
//
// MIT License
// Copyright (c) 1999-2022, Arm Limited.
// Copyright (c) 2017-2025, Arm Limited.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#[inline]
pub(super) fn scalar_exp_fma(input: f32) -> f32 {
    const TABLE: [u64; 32] = [
        0x3ff0_0000_0000_0000,
        0x3fef_d9b0_d315_8574,
        0x3fef_b558_6cf9_890f,
        0x3fef_9301_d012_5b51,
        0x3fef_72b8_3c7d_517b,
        0x3fef_5487_3168_b9aa,
        0x3fef_387a_6e75_6238,
        0x3fef_1e9d_f51f_dee1,
        0x3fef_06fe_0a31_b715,
        0x3fee_f1a7_373a_a9cb,
        0x3fee_dea6_4c12_3422,
        0x3fee_ce08_6061_892d,
        0x3fee_bfda_d536_2a27,
        0x3fee_b42b_569d_4f82,
        0x3fee_ab07_dd48_5429,
        0x3fee_a47e_b03a_5585,
        0x3fee_a09e_667f_3bcd,
        0x3fee_9f75_e8ec_5f74,
        0x3fee_a114_73eb_0187,
        0x3fee_a589_994c_ce13,
        0x3fee_ace5_422a_a0db,
        0x3fee_b737_b0cd_c5e5,
        0x3fee_c491_82a3_f090,
        0x3fee_d503_b23e_255d,
        0x3fee_e89f_995a_d3ad,
        0x3fee_ff76_f2fb_5e47,
        0x3fef_199b_dd85_529c,
        0x3fef_3720_dcef_9069,
        0x3fef_5818_dcfb_a487,
        0x3fef_7c97_337b_9b5f,
        0x3fef_a4af_a2a4_90da,
        0x3fef_d076_5b6e_4540,
    ];
    const INV_LN2_SCALED: f64 = f64::from_bits(0x4047_1547_652b_82fe);
    const SHIFT: f64 = f64::from_bits(0x4338_0000_0000_0000);
    const CUBIC: f64 = f64::from_bits(0x3ebc_6af8_4b91_2394);
    const QUADRATIC: f64 = f64::from_bits(0x3f2e_bfce_50fa_c4f3);
    const LINEAR: f64 = f64::from_bits(0x3f96_2e42_ff0c_52d6);

    if input.is_nan() {
        return input + input;
    }
    if input > f32::from_bits(0x42b1_7217) {
        return f32::INFINITY;
    }
    if input < f32::from_bits(0xc2cf_f1b4) {
        return 0.0;
    }
    let value = f64::from(input);
    let shifted = INV_LN2_SCALED.mul_add(value, SHIFT);
    let exponent_bits = shifted.to_bits();
    let exponent = shifted - SHIFT;
    let reduced = INV_LN2_SCALED.mul_add(value, -exponent);
    let table_index = usize::from(exponent_bits.to_le_bytes()[0] & 31);
    let scale = f64::from_bits(TABLE[table_index].wrapping_add(exponent_bits.wrapping_shl(47)));
    let high = CUBIC.mul_add(reduced, QUADRATIC);
    let low = LINEAR.mul_add(reduced, 1.0);
    let polynomial = high.mul_add(reduced * reduced, low);
    // The algorithm intentionally rounds its double-precision result once to f32.
    #[allow(clippy::cast_possible_truncation)]
    let rounded = (polynomial * scale) as f32;
    rounded
}

/// In-place fused scale + numerically-stable softmax for one attention-score row.
/// Computes `softmax(scale · row)` over `row` (one head's query position), using
/// the 8-wide non-fused polynomial `exp` (~1-2 ULP) and a scalar tail with
/// explicit fused operations matching the qualified GNU producer.
/// `scale > 0`, so the argmax (hence the stabilising max-subtraction) is
/// unchanged by the scale: `exp(scale·(x − max)) == exp(scale·x)/exp(scale·max)`.
fn softmax_row_fused(row: &mut [f32], scale: f32) {
    let n = row.len();
    let mut max_raw = f32::NEG_INFINITY;
    for &x in row.iter() {
        max_raw = max_raw.max(x);
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut sum_v = f32x8::splat(0.0);
    let mut i = 0;
    // Issue 4 independent exps per iteration so the core overlaps their latency (the
    // exp is a long-latency polynomial that starves a one-group-in-flight loop). A
    // single accumulator added in the base loop's exact order keeps sum_v
    // BIT-IDENTICAL — only the exp scheduling changes. The attention softmax is the
    // dominant growing frame (~24% of the forward at seq 512); this pays a clean
    // ~3.9% there (bench softmax_exp_ilp), where it matters, and is within noise at
    // shorter rows (where softmax is a small fraction). A 4-accumulator variant that
    // breaks the sum chain wins LESS, confirming the bottleneck is exp latency (not
    // the reduction), so the bit-identical single-accumulator form is also the fastest.
    while i + 32 <= n {
        let e0 = exp_vec8_non_fused((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v);
        let e1 = exp_vec8_non_fused((f32x8_from_slice(&row[i + 8..i + 16]) - max_v) * scale_v);
        let e2 = exp_vec8_non_fused((f32x8_from_slice(&row[i + 16..i + 24]) - max_v) * scale_v);
        let e3 = exp_vec8_non_fused((f32x8_from_slice(&row[i + 24..i + 32]) - max_v) * scale_v);
        row[i..i + 8].copy_from_slice(&e0.to_array());
        row[i + 8..i + 16].copy_from_slice(&e1.to_array());
        row[i + 16..i + 24].copy_from_slice(&e2.to_array());
        row[i + 24..i + 32].copy_from_slice(&e3.to_array());
        sum_v += e0;
        sum_v += e1;
        sum_v += e2;
        sum_v += e3;
        i += 32;
    }
    while i + 8 <= n {
        let e = exp_vec8_non_fused((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v);
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum: f32 = sum_v.to_array().iter().sum();
    while i < n {
        let e = scalar_exp_fma((row[i] - max_raw) * scale);
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// Fused scale + softmax over the last dim of the attention scores `[rows, n]`
/// (row-major), returning the same buffer/layout. Replaces the separate
/// `mul_scalar` + `tensor_softmax` tape ops with one vectorized pass.
///
/// Profiling showed the attention softmax is the dominant *growing* f32 frame —
/// ~24% of the per-doc forward wall-clock at seq 512 (12·S² scalar `expf` calls,
/// which ONNX/MLAS fuses + vectorizes). Softmax normalisation makes the ~1e-6
/// relative exp error immaterial to the ranking (validated against the
/// numpy/ONNX reference: ranking + logit tolerance unchanged). Rows are
/// independent, so they parallelize across the forward's ambient rayon pool —
/// matching the multicore the stock softmax kernel had. This stays intra-forward
/// (the doc loop is sequential), so the nested-rayon + `Mutex` deadlock class
/// remains closed by construction.
pub(super) fn fast_softmax_inplace(data: &mut [f32], rows: usize, n: usize, scale: f32) {
    debug_assert_eq!(data.len(), rows * n);
    if n == 0 {
        return;
    }
    // Parallelize only when there is enough work to amortise the fan-out and a
    // pool is actually available; otherwise stay serial (small seq / 1 thread).
    if rows >= 8 && rows * n >= 8192 && rayon::current_num_threads() > 1 {
        use rayon::prelude::{ParallelIterator, ParallelSliceMut};
        data.par_chunks_exact_mut(n)
            .for_each(|row| softmax_row_fused(row, scale));
    } else {
        data.chunks_exact_mut(n)
            .for_each(|row| softmax_row_fused(row, scale));
    }
}

/// Exact-form GELU `0.5·x·(1 + erf(x/√2))` for one f32x8 lane group, with `erf`
/// from the Abramowitz–Stegun 7.1.26 rational×exp approximation (max abs error
/// 1.5e-7 — at the f32 precision floor, so indistinguishable from libm `erff` for
/// ranking). `erf` is odd, so it is evaluated on `|z|` and the sign is reapplied.
#[inline]
fn gelu_vec8(x: f32x8) -> f32x8 {
    const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let one = f32x8::splat(1.0);
    let z = x * f32x8::splat(C);
    let az = z.abs();
    let t = one / (one + f32x8::splat(0.327_591_1) * az);
    // erf poly = t·(a1 + t·(a2 + t·(a3 + t·(a4 + t·a5))))  (A–S 7.1.26)
    let a1 = f32x8::splat(0.254_829_6);
    let a2 = f32x8::splat(-0.284_496_73);
    let a3 = f32x8::splat(1.421_413_7);
    let a4 = f32x8::splat(-1.453_152);
    let a5 = f32x8::splat(1.061_405_4);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let erf_abs = one - poly * exp_vec8_non_fused(-(z * z));
    let erf = erf_abs.copysign(z);
    f32x8::splat(0.5) * x * (one + erf)
}

/// Scalar GELU matching [`gelu_vec8`] (same A–S erf) for the < 8-element tail.
#[inline]
fn gelu_scalar(x: f32) -> f32 {
    const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let z = x * C;
    let az = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * az);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_73 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let erf = (1.0 - poly * (-(z * z)).exp()).copysign(z);
    0.5 * x * (1.0 + erf)
}

/// Vectorized exact-GELU over a flat activation buffer, in place. Replaces the
/// `tensor_gelu` tape op (scalar libm `erff`) — profiling put GELU at ~10-14% of
/// the per-doc forward (a wide `[m, 1536]` elementwise transcendental). GELU is
/// elementwise so chunks are independent and parallelize across the forward's
/// rayon pool. A–S erf keeps the result within ~1e-7 of exact, so the ranking is
/// unchanged (validated against the numpy/ONNX reference).
pub(super) fn fast_gelu_inplace(data: &mut [f32]) {
    let process = |chunk: &mut [f32]| {
        let n = chunk.len();
        let mut i = 0;
        // Process 4 independent f32x8 lane groups per iteration. GELU is a pure
        // elementwise map (no cross-lane reduction), so the four gelu_vec8 chains
        // are independent — issuing them back-to-back lets the core overlap their
        // latency (the erf Horner poly + exp, high-latency but pipelined) instead of
        // stalling one group at a time. Byte-identical to a one-group loop (same op
        // at the same position); measured ~4-5% faster on the in-cache FFN GELU
        // widths, fading toward memory-bound at multi-MB buffers (bench `gelu_ilp`).
        while i + 32 <= n {
            let g0 = gelu_vec8(f32x8_from_slice(&chunk[i..i + 8]));
            let g1 = gelu_vec8(f32x8_from_slice(&chunk[i + 8..i + 16]));
            let g2 = gelu_vec8(f32x8_from_slice(&chunk[i + 16..i + 24]));
            let g3 = gelu_vec8(f32x8_from_slice(&chunk[i + 24..i + 32]));
            chunk[i..i + 8].copy_from_slice(&g0.to_array());
            chunk[i + 8..i + 16].copy_from_slice(&g1.to_array());
            chunk[i + 16..i + 24].copy_from_slice(&g2.to_array());
            chunk[i + 24..i + 32].copy_from_slice(&g3.to_array());
            i += 32;
        }
        while i + 8 <= n {
            let g = gelu_vec8(f32x8_from_slice(&chunk[i..i + 8]));
            chunk[i..i + 8].copy_from_slice(&g.to_array());
            i += 8;
        }
        while i < n {
            chunk[i] = gelu_scalar(chunk[i]);
            i += 1;
        }
    };
    if data.len() >= 8192 && rayon::current_num_threads() > 1 {
        use rayon::prelude::{ParallelIterator, ParallelSliceMut};
        // Elementwise, so any chunking is correct; keep chunks a multiple of 8 so
        // only the final chunk has a scalar tail.
        data.par_chunks_mut(2048).for_each(process);
    } else {
        process(data);
    }
}

#[inline]
fn f32x8_from_slice(slice: &[f32]) -> f32x8 {
    let mut buf = [0.0f32; 8];
    buf.copy_from_slice(&slice[..8]);
    f32x8::new(buf)
}

#[inline]
fn dot_hd(q: &[f32], k: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), HD);
    debug_assert_eq!(k.len(), HD);
    (f32x8_from_slice(&q[0..]) * f32x8_from_slice(&k[0..])
        + f32x8_from_slice(&q[8..]) * f32x8_from_slice(&k[8..])
        + f32x8_from_slice(&q[16..]) * f32x8_from_slice(&k[16..])
        + f32x8_from_slice(&q[24..]) * f32x8_from_slice(&k[24..]))
    .reduce_add()
}

#[inline]
fn q_lanes(q: &[f32]) -> [f32x8; 4] {
    debug_assert_eq!(q.len(), HD);
    [
        f32x8_from_slice(&q[0..]),
        f32x8_from_slice(&q[8..]),
        f32x8_from_slice(&q[16..]),
        f32x8_from_slice(&q[24..]),
    ]
}

#[inline]
fn dot_hd_q_lanes(q: [f32x8; 4], k: &[f32]) -> f32 {
    debug_assert_eq!(k.len(), HD);
    (q[0] * f32x8_from_slice(&k[0..])
        + q[1] * f32x8_from_slice(&k[8..])
        + q[2] * f32x8_from_slice(&k[16..])
        + q[3] * f32x8_from_slice(&k[24..]))
    .reduce_add()
}

fn weighted_value_sum_hd(qkv: &[f32], probs: &[f32], s_len: usize, head: usize, out: &mut [f32]) {
    debug_assert_eq!(probs.len(), s_len);
    debug_assert_eq!(out.len(), HD);
    const STRIDE: usize = 3 * H;
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);
    for (j, &prob) in probs.iter().enumerate() {
        let p = f32x8::splat(prob);
        let base = j * STRIDE + 2 * H + head * HD;
        acc0 += p * f32x8_from_slice(&qkv[base..base + 8]);
        acc1 += p * f32x8_from_slice(&qkv[base + 8..base + 16]);
        acc2 += p * f32x8_from_slice(&qkv[base + 16..base + 24]);
        acc3 += p * f32x8_from_slice(&qkv[base + 24..base + 32]);
    }
    out[0..8].copy_from_slice(&acc0.to_array());
    out[8..16].copy_from_slice(&acc1.to_array());
    out[16..24].copy_from_slice(&acc2.to_array());
    out[24..32].copy_from_slice(&acc3.to_array());
}

/// Reusable per-document attention scratch. Holds the head-major repack operands and
/// the two `bmm` output buffers so a whole forward's worth of attention calls (every
/// document × every layer) reuse one set of allocations instead of allocating —
/// and zero-filling — fresh `Vec`s each call. The score buffer dominates (`NH·S²`),
/// so eliding its per-call realloc is the bulk of the win. Buffers only ever grow
/// (`ensure`), and every element each `bmm`/repack touches is overwritten, so reuse
/// is bit-identical to fresh allocation.
#[derive(Default)]
pub(super) struct AttnScratch {
    q_hm: Vec<f32>,
    kt: Vec<f32>,
    v_hm: Vec<f32>,
    scores: Vec<f32>,
    ctx_hm: Vec<f32>,
}

impl AttnScratch {
    /// Grow (never shrink) every buffer to hold one document's full self-attention at
    /// sequence length `s_len`. The CLS-only path uses strict prefixes of these.
    fn ensure(&mut self, s_len: usize) {
        let hm = s_len * H; // NH * s_len * HD == s_len * H
        let sc = NH * s_len * s_len;
        if self.q_hm.len() < hm {
            self.q_hm.resize(hm, 0.0);
        }
        if self.kt.len() < hm {
            self.kt.resize(hm, 0.0);
        }
        if self.v_hm.len() < hm {
            self.v_hm.resize(hm, 0.0);
        }
        if self.scores.len() < sc {
            self.scores.resize(sc, 0.0);
        }
        if self.ctx_hm.len() < hm {
            self.ctx_hm.resize(hm, 0.0);
        }
    }
}

/// Multi-head self-attention for one document, writing the `[s_len, H]` context into
/// `out`. `qkv` is the fused QKV linear output `[s_len, 3H]` (Q at column 0, K at
/// column H, V at column 2H).
///
/// The per-query `f32x8` kernel was cheap to launch (no tape, no per-head gemm) but
/// runs the S² inner loop one dot at a time, which dominates at seq ≥ 256 (it was
/// ~9× over the FLOP floor — a register-blocked GEMM does the same MACs far faster).
/// So repack each head's Q/K/V into the head-major layout the batched `bmm` kernel
/// (`gemm::sgemm`, cache-blocking) wants, do `QKᵀ` and ·V as two batched GEMMs with a
/// vectorized softmax between, and transpose the context back. The repacks and both
/// GEMM outputs land in the caller's reused `scratch`, so the hot loop does not
/// allocate. The `bmm` kernel parallelizes across the NH head-batches internally and
/// stays intra-forward, so the nested-rayon + `Mutex` deadlock class remains closed.
/// Bit-exact to the bmm tape path (within f32 reassociation) — validated by parity +
/// `forward_batch_matches_per_doc`.
pub(super) fn fused_attention(
    scratch: &mut AttnScratch,
    qkv: &[f32],
    s_len: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(qkv.len(), s_len * 3 * H);
    debug_assert_eq!(out.len(), s_len * H);
    if s_len == 0 {
        return;
    }
    const STRIDE: usize = 3 * H;
    scratch.ensure(s_len);
    let hm = s_len * H;
    let sc = NH * s_len * s_len;
    let AttnScratch {
        q_hm,
        kt,
        v_hm,
        scores,
        ctx_hm,
    } = scratch;
    let (q_hm, kt, v_hm) = (&mut q_hm[..hm], &mut kt[..hm], &mut v_hm[..hm]);
    let scores = &mut scores[..sc];
    let ctx_hm = &mut ctx_hm[..hm];
    // Repack into head-major operands: Q/V as [NH, S, HD], K transposed to [NH, HD, S]
    // so QKᵀ is a plain `bmm(Q, Kᵀ)`. Q/V are contiguous head-major copies.
    for j in 0..s_len {
        let base = j * STRIDE;
        for h in 0..NH {
            let hmj = (h * s_len + j) * HD;
            q_hm[hmj..hmj + HD].copy_from_slice(&qkv[base + h * HD..base + h * HD + HD]);
            v_hm[hmj..hmj + HD]
                .copy_from_slice(&qkv[base + 2 * H + h * HD..base + 2 * H + h * HD + HD]);
        }
    }
    // Kᵀ: each output row `kt[h, d, :]` is written sequentially (contiguous stores)
    // from the strided source column, instead of strided stores into kt.
    for h in 0..NH {
        for d in 0..HD {
            let col = H + h * HD + d;
            let row = &mut kt[h * HD * s_len + d * s_len..h * HD * s_len + d * s_len + s_len];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = qkv[j * STRIDE + col];
            }
        }
    }
    // scores[NH, S, S] = Q @ Kᵀ, then in-place fused-scale softmax over the last dim.
    let qm = TensorMeta::from_shape(vec![NH, s_len, HD], DType::F32, Device::Cpu);
    let km = TensorMeta::from_shape(vec![NH, HD, s_len], DType::F32, Device::Cpu);
    ft_api::bmm_tensor_contiguous_f32_into(q_hm, kt, &qm, &km, scores)
        .expect("attn QKᵀ bmm: shapes are internally consistent");
    #[cfg(test)]
    certificate_trace::detailed("attention.scores", &[NH, s_len, s_len], scores);
    fast_softmax_inplace(scores, NH * s_len, s_len, scale);
    #[cfg(test)]
    certificate_trace::detailed("attention.softmax", &[NH, s_len, s_len], scores);
    // ctx_hm[NH, S, HD] = scores @ V.
    let sm = TensorMeta::from_shape(vec![NH, s_len, s_len], DType::F32, Device::Cpu);
    let vm = TensorMeta::from_shape(vec![NH, s_len, HD], DType::F32, Device::Cpu);
    ft_api::bmm_tensor_contiguous_f32_into(scores, v_hm, &sm, &vm, ctx_hm)
        .expect("attn ·V bmm: shapes are internally consistent");
    // Transpose the head-major context back to token-major [S, H] into `out`.
    for h in 0..NH {
        for i in 0..s_len {
            let src = (h * s_len + i) * HD;
            out[i * H + h * HD..i * H + h * HD + HD].copy_from_slice(&ctx_hm[src..src + HD]);
        }
    }
}

/// Self-attention for the `[CLS]` query ONLY (token 0), writing its `[H]` context into
/// `out`.
///
/// The reranker's logit is read solely from `[CLS]` of the final layer, so the last
/// encoder layer never needs any other token's attention output. `[CLS]` still attends
/// to every key, but the operation is rank-1 per head: one `[HD]` query row scores all
/// keys, then one probability row weights the values. Running that directly over the
/// fused QKV buffer avoids the old head-major K/V repacks plus two tiny `bmm` launches
/// whose `m = 1` shape is below the GEMM kernel's useful blocking regime. The softmax
/// kernel and output layout stay the same; only the CLS-only final layer uses this path.
pub(super) fn fused_attention_cls(
    scratch: &mut AttnScratch,
    qkv: &[f32],
    s_len: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(qkv.len(), s_len * 3 * H);
    debug_assert_eq!(out.len(), H);
    if s_len == 0 {
        out.fill(0.0);
        return;
    }
    const STRIDE: usize = 3 * H;
    scratch.ensure(s_len);
    let sc = NH * s_len;
    for h in 0..NH {
        let row = &mut scratch.scores[h * s_len..(h + 1) * s_len];
        if s_len >= CLS_Q_CACHE_MIN_SEQ {
            let q = q_lanes(&qkv[h * HD..h * HD + HD]);
            for (j, slot) in row.iter_mut().enumerate() {
                let k_base = j * STRIDE + H + h * HD;
                *slot = dot_hd_q_lanes(q, &qkv[k_base..k_base + HD]);
            }
        } else {
            let q = &qkv[h * HD..h * HD + HD];
            for (j, slot) in row.iter_mut().enumerate() {
                let k_base = j * STRIDE + H + h * HD;
                *slot = dot_hd(q, &qkv[k_base..k_base + HD]);
            }
        }
    }
    let scores = &mut scratch.scores[..sc];
    fast_softmax_inplace(scores, NH, s_len, scale);
    for h in 0..NH {
        let row = &scores[h * s_len..(h + 1) * s_len];
        weighted_value_sum_hd(qkv, row, s_len, h, &mut out[h * HD..h * HD + HD]);
    }
}
