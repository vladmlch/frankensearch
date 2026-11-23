//! Portable SIMD dot-product helpers for vector search.
#![allow(clippy::inline_always)]

use frankensearch_core::{SearchError, SearchResult};
use half::f16;
#[cfg(target_endian = "little")]
use wide::u16x8;
use wide::{f32x8, i16x16, u32x8};

mod dot_4bit;
mod dot_float;
mod dot_int8;
mod quantize_pack;
#[cfg(test)]
mod tests;

#[inline(always)]
pub(crate) fn dot_i8x4_i8(stored_rows: &[i8], query: &[i8]) -> [i32; 4] {
    dot_int8::dot_i8x4_i8(stored_rows, query)
}

pub use self::dot_4bit::{
    dot_4bit_prepared, dot_4bit_prepared_dynamic, dot_4bit_prepared_generic, dot_packed_4bit,
    prepare_4bit_query,
};
pub use self::dot_float::{
    cosine_similarity_f16, dot_product_f16_bytes_f32, dot_product_f16_bytes_f32_fma,
    dot_product_f16_bytes_f32_generic, dot_product_f16_f32, dot_product_f16_f32_generic,
    dot_product_f32_bytes_f32, dot_product_f32_bytes_f32_generic, dot_product_f32_f32,
    dot_product_f32_f32_generic,
};
pub use self::dot_int8::{
    dot_i8_i8, dot_i8_i8_generic, dot_i8_i8_maddubs, dot_i8x4_i8_maddubs, maddubs_query_bias,
};
pub use self::quantize_pack::{
    encode_f32_to_f16_extend, encode_f32_to_f16_extend_generic, pack_f16_le_bytes_to_4bit,
    pack_f16_le_bytes_to_4bit_explicit_round, pack_f16_le_bytes_to_4bit_generic,
    pack_f16_le_bytes_to_4bit_scalar_pack, pack_f16_slab_to_4bit, pack_f16_slab_to_4bit_generic,
    quantize_f16_le_bytes_to_i8, quantize_f16_le_bytes_to_i8_generic, quantize_f16_slab_to_i8,
    quantize_f16_slab_to_i8_generic,
};

#[cfg(all(test, target_arch = "x86_64"))]
use self::dot_4bit::dot_4bit_prepared_avx2;
#[cfg(all(test, target_arch = "x86_64"))]
use self::dot_float::{
    dot_product_f16_bytes_f32_avx2, dot_product_f16_f32_avx2, dot_product_f32_bytes_f32_avx2,
    dot_product_f32_f32_avx2,
};
#[cfg(test)]
use self::dot_int8::dot_i8x4_i8_generic;
#[cfg(all(test, target_arch = "x86_64"))]
use self::dot_int8::{dot_i8_i8_avx2, dot_i8x4_i8_avx2};
#[cfg(all(test, target_arch = "x86_64"))]
use self::quantize_pack::{
    encode_f32_to_f16_extend_avx2, pack_f16_slab_to_4bit_avx2, quantize_f16_slab_to_i8_avx2,
};
#[cfg(all(test, target_arch = "x86_64", target_endian = "little"))]
use self::quantize_pack::{pack_f16_le_bytes_to_4bit_avx2, quantize_f16_le_bytes_to_i8_avx2};

// ── SIMD load/decode helpers ────────────────────────────────────────────────
//
// The f32 byte-decoding dot-product kernel below uses four independent f32x8
// accumulators per 32 elements. A single accumulator serializes on SIMD add
// latency; four accumulators expose enough instruction-level parallelism to run
// the loop at throughput instead.

/// Load 8 consecutive `f32` lanes from a fixed 32-element array at a const offset.
#[inline(always)]
fn load8_f32<const OFF: usize>(a: &[f32; 32]) -> f32x8 {
    f32x8::from([
        a[OFF],
        a[OFF + 1],
        a[OFF + 2],
        a[OFF + 3],
        a[OFF + 4],
        a[OFF + 5],
        a[OFF + 6],
        a[OFF + 7],
    ])
}

/// Decode 8 little-endian `f32` values from a 32-byte block to `f32x8`.
#[inline(always)]
fn decode8_f32(b: &[u8; 32]) -> f32x8 {
    f32x8::from([
        f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        f32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        f32::from_le_bytes([b[16], b[17], b[18], b[19]]),
        f32::from_le_bytes([b[20], b[21], b[22], b[23]]),
        f32::from_le_bytes([b[24], b[25], b[26], b[27]]),
        f32::from_le_bytes([b[28], b[29], b[30], b[31]]),
    ])
}

// ── Branchless SIMD f16 → f32 widen ─────────────────────────────────────────
//
// The f16 dot-product paths are decode-bound: per-element scalar `f16::to_f32()`
// dominates (see docs/NEGATIVE_EVIDENCE.md). This widens 8 f16 lanes at once via
// the Giesen "magic multiply" trick (a denormal-as-float multiply renormalizes
// both normals and subnormals in one shot), with a pure-integer inf/nan fixup.
//
// It is **bit-exact** to `f16::to_f32()` for every finite (incl. subnormal) and
// zero input, and maps inf→inf / nan→nan with the sign preserved (the nan
// payload's quiet bit may differ, which only affects the bits of a NaN score —
// never produced by real, finite, L2-normalized embeddings). Exhaustively
// verified over all 65 536 bit patterns by `simd_f16_widen_is_bit_exact`.
//
// Because the decoded f32 values are bit-identical for finite data and the
// accumulation order is unchanged, every dot-product score is bit-identical to
// the prior scalar-decode kernel on real corpora — no determinism/golden risk.

/// 2^112 — the Giesen magic factor that rebiases the f16 exponent.
const F16_WIDEN_MAGIC: f32 = f32::from_bits(0x7780_0000);

/// Widen 8 f16 bit-patterns (held in the low 16 bits of each `u32x8` lane) to f32.
#[inline(always)]
fn widen8_f16_lanes(h: u32x8) -> f32x8 {
    let sign = (h & u32x8::splat(0x0000_8000)) << 16_u32;
    let exp_mant = (h & u32x8::splat(0x0000_7fff)) << 13_u32;
    let scaled = bytemuck::cast::<u32x8, f32x8>(exp_mant) * f32x8::splat(F16_WIDEN_MAGIC);
    let scaled_bits = bytemuck::cast::<f32x8, u32x8>(scaled);

    // inf/nan: f16 exponent field == 0x7c00. Detect via carry out of bit 15
    // ((he + 0x0400) sets bit 15 iff he == 0x7c00), spread to a full lane mask,
    // and OR the f32 exponent up to all-ones (0xff << 23).
    let he = h & u32x8::splat(0x0000_7c00);
    let carry = (he + u32x8::splat(0x0000_0400)) & u32x8::splat(0x0000_8000);
    let infnan_mask = (carry >> 15_u32) * u32x8::splat(0xff << 23);

    bytemuck::cast::<u32x8, f32x8>((scaled_bits | infnan_mask) | sign)
}

/// Decode 8 little-endian f16 values from a 16-byte block to `f32x8` (SIMD widen).
///
/// On little-endian targets (x86/ARM — the only ones the FSVI LE format targets),
/// the 16 bytes already *are* 8 little-endian `u16` lanes, so they load straight
/// into a `u16x8` and zero-extend to `u32x8` (one SIMD load + widen) — avoiding the
/// 8 scalar `from_le_bytes` reads and the stack round-trip of building a `[u32; 8]`.
#[cfg(target_endian = "little")]
#[inline(always)]
pub(crate) fn widen8_f16_bytes(b: &[u8; 16]) -> f32x8 {
    let lanes = bytemuck::cast::<[u8; 16], u16x8>(*b);
    widen8_f16_lanes(u32x8::from(lanes))
}

/// Big-endian fallback: decode each `u16` explicitly as little-endian.
#[cfg(target_endian = "big")]
#[inline(always)]
pub(crate) fn widen8_f16_bytes(b: &[u8; 16]) -> f32x8 {
    let lanes: [u32; 8] = [
        u32::from(u16::from_le_bytes([b[0], b[1]])),
        u32::from(u16::from_le_bytes([b[2], b[3]])),
        u32::from(u16::from_le_bytes([b[4], b[5]])),
        u32::from(u16::from_le_bytes([b[6], b[7]])),
        u32::from(u16::from_le_bytes([b[8], b[9]])),
        u32::from(u16::from_le_bytes([b[10], b[11]])),
        u32::from(u16::from_le_bytes([b[12], b[13]])),
        u32::from(u16::from_le_bytes([b[14], b[15]])),
    ];
    widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes))
}

/// Widen 8 consecutive `f16` values (from a fixed array) to `f32x8` (SIMD widen).
#[inline(always)]
pub(crate) fn widen8_f16_slice(s: &[f16; 8]) -> f32x8 {
    let lanes: [u32; 8] = [
        u32::from(s[0].to_bits()),
        u32::from(s[1].to_bits()),
        u32::from(s[2].to_bits()),
        u32::from(s[3].to_bits()),
        u32::from(s[4].to_bits()),
        u32::from(s[5].to_bits()),
        u32::from(s[6].to_bits()),
        u32::from(s[7].to_bits()),
    ];
    widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes))
}

/// A query pre-unpacked into per-16-byte-chunk sign-extended low/high nibble lanes
/// (+ a scalar tail).
///
/// The query nibbles are decoded **once** rather than for
/// every stored vector in a scan. See [`prepare_4bit_query`] / [`dot_4bit_prepared`].
pub struct PreparedQuery4bit {
    low: Vec<i16x16>,
    high: Vec<i16x16>,
    tail: Vec<(i32, i32)>,
}

const fn ensure_same_len(expected: usize, found: usize) -> SearchResult<()> {
    if expected != found {
        return Err(SearchError::DimensionMismatch { expected, found });
    }
    Ok(())
}
