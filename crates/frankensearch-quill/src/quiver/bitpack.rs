use thiserror::Error;
use wide::u32x8;

// Production decoding routes per width through `unpack_dispatch_into`: the
// portable eight-lane `wide::u32x8` kernel only inside the measured winning
// band [WIDE_UNPACK_MIN_WIDTH, WIDE_UNPACK_MAX_WIDTH], and the scalar
// reservoir everywhere else. A same-binary A/B rejected a single
// wide-for-all policy on 2026-07-17 (narrow widths regress); the 2026-07-23
// per-width gate (bd-bz7q) confirmed a TWO-SIDED crossover with valid A/A
// null controls and sanctioned this banded dispatcher.

/// Smallest bit width for which the eight-lane `wide::u32x8` unpacker beats
/// the scalar reservoir. Widths below this decode with the scalar path.
///
/// The boundary is measured, not assumed: the `postings_decode_ab` per-width
/// gate (`QUILL_POSTINGS_AB_INNER=512`, 256 nonzero cells, zero invalid A/A
/// nulls) showed widths 1-2 as decidable regressions (mean wide/scalar
/// 1.29-1.98) and width 3 inside the directional null floor at 1.19, while
/// widths 4-28 win with valid null controls (mean ratios 0.47-0.89).
pub const WIDE_UNPACK_MIN_WIDTH: u8 = 4;

/// Largest bit width routed to the `wide::u32x8` unpacker. The same gate
/// found the SIMD advantage inverts at very wide encodings: widths 29-31
/// have zero wins and mean wide/scalar ratios 1.09-1.12 (a decidable
/// regression at w30/o31), and width 32 is a directional-floor wash. The
/// scalar reservoir is at least as fast once each value spans nearly a full
/// `u32`, so wide routing stops at 28 to avoid shipping the tail regression.
pub const WIDE_UNPACK_MAX_WIDTH: u8 = 28;

/// Typed failures for the canonical FSLX bitpacked-value stream.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BitpackError {
    /// FSLX values use at most all 32 bits of a `u32`.
    #[error("bit width {width} is outside the supported range 0..=32")]
    InvalidWidth {
        /// Rejected width.
        width: u8,
    },
    /// Computing the canonical byte length overflowed `usize`.
    #[error("packed length overflow for {count} values at width {width}")]
    LengthOverflow {
        /// Number of encoded values.
        count: usize,
        /// Bits per value.
        width: u8,
    },
    /// A payload was not exactly the canonical length for its shape.
    #[error("packed payload length mismatch: expected {expected} bytes, got {actual}")]
    LengthMismatch {
        /// Canonical payload length.
        expected: usize,
        /// Supplied payload length.
        actual: usize,
    },
    /// An encoder input cannot be represented at the requested width.
    #[error("value {value} at index {index} does not fit in {width} bits")]
    ValueOutOfRange {
        /// Value ordinal.
        index: usize,
        /// Rejected value.
        value: u32,
        /// Requested width.
        width: u8,
    },
    /// Canonical LSB-first packing requires unused high bits to be zero.
    #[error(
        "packed payload has non-zero padding in final byte {byte:#04x}; only {used_bits} bits are used"
    )]
    NonCanonicalPadding {
        /// Final payload byte.
        byte: u8,
        /// Low bits belonging to encoded values.
        used_bits: usize,
    },
}

fn validate_width(width: u8) -> Result<(), BitpackError> {
    if width <= 32 {
        Ok(())
    } else {
        Err(BitpackError::InvalidWidth { width })
    }
}

pub(super) fn packed_byte_len(count: usize, width: u8) -> Result<usize, BitpackError> {
    validate_width(width)?;
    count
        .checked_mul(usize::from(width))
        .map(|bits| bits.div_ceil(u8::BITS as usize))
        .ok_or(BitpackError::LengthOverflow { count, width })
}

fn value_mask(width: u8) -> u32 {
    debug_assert!((1..=32).contains(&width));
    u32::MAX >> (u32::BITS - u32::from(width))
}

fn validate_payload(input: &[u8], width: u8, count: usize) -> Result<(), BitpackError> {
    let expected = packed_byte_len(count, width)?;
    if input.len() != expected {
        return Err(BitpackError::LengthMismatch {
            expected,
            actual: input.len(),
        });
    }

    let used_bits = count
        .checked_mul(usize::from(width))
        .ok_or(BitpackError::LengthOverflow { count, width })?
        % (u8::BITS as usize);
    if used_bits != 0 {
        let final_byte = input.last().copied().ok_or(BitpackError::LengthMismatch {
            expected,
            actual: input.len(),
        })?;
        let allowed = u8::MAX >> ((u8::BITS as usize) - used_bits);
        if final_byte & !allowed != 0 {
            return Err(BitpackError::NonCanonicalPadding {
                byte: final_byte,
                used_bits,
            });
        }
    }
    Ok(())
}

/// Canonically pack `u32` values LSB-first into a byte stream.
#[allow(clippy::cast_possible_truncation)]
pub fn pack_lsb(values: &[u32], width: u8) -> Result<Vec<u8>, BitpackError> {
    validate_width(width)?;
    let output_len = packed_byte_len(values.len(), width)?;
    if width == 0 {
        if let Some((index, &value)) = values.iter().enumerate().find(|(_, value)| **value != 0) {
            return Err(BitpackError::ValueOutOfRange {
                index,
                value,
                width,
            });
        }
        return Ok(Vec::new());
    }

    let mask = value_mask(width);
    let width = usize::from(width);
    let mut output = Vec::with_capacity(output_len);
    let mut reservoir = 0_u64;
    let mut available = 0_usize;
    for (index, &value) in values.iter().enumerate() {
        if value > mask {
            return Err(BitpackError::ValueOutOfRange {
                index,
                value,
                width: width as u8,
            });
        }
        reservoir |= u64::from(value) << available;
        available += width;
        while available >= u8::BITS as usize {
            output.push((reservoir & u64::from(u8::MAX)) as u8);
            reservoir >>= u8::BITS;
            available -= u8::BITS as usize;
        }
    }
    if available != 0 {
        output.push((reservoir & u64::from(u8::MAX)) as u8);
    }
    debug_assert_eq!(output.len(), output_len);
    Ok(output)
}

/// Decode the canonical stream with the scalar reference reservoir.
#[allow(clippy::cast_possible_truncation)]
pub fn unpack_scalar_into(input: &[u8], width: u8, output: &mut [u32]) -> Result<(), BitpackError> {
    validate_payload(input, width, output.len())?;
    if width == 0 {
        output.fill(0);
        return Ok(());
    }

    let width = usize::from(width);
    let mask = u64::from(value_mask(width as u8));
    let mut bytes = input.iter().copied();
    let mut reservoir = 0_u64;
    let mut available = 0_usize;
    for value in output {
        while available < width {
            // `validate_payload` proved the exact source length.
            let byte = bytes.next().ok_or(BitpackError::LengthMismatch {
                expected: input.len(),
                actual: input.len(),
            })?;
            reservoir |= u64::from(byte) << available;
            available += u8::BITS as usize;
        }
        *value = (reservoir & mask) as u32;
        reservoir >>= width;
        available -= width;
    }
    Ok(())
}

/// Unpack one byte-aligned group of eight fixed-width values.
///
/// Eight `WIDTH`-bit values occupy exactly `WIDTH` bytes. We gather the
/// possibly crossing little-endian words per lane, then perform the actual
/// variable shifts, merge, and mask in one `wide::u32x8` kernel.
#[allow(clippy::cast_possible_truncation)]
fn unpack_eight<const WIDTH: usize>(input: &[u8]) -> [u32; 8] {
    debug_assert!((1..=32).contains(&WIDTH));
    debug_assert_eq!(input.len(), WIDTH);

    let mut words = [0_u32; 9];
    for (word, chunk) in words.iter_mut().zip(input.chunks(4)) {
        let mut little_endian = [0_u8; 4];
        little_endian[..chunk.len()].copy_from_slice(chunk);
        *word = u32::from_le_bytes(little_endian);
    }

    let mut low_words = [0_u32; 8];
    let mut high_words = [0_u32; 8];
    let mut right_shifts = [0_u32; 8];
    let mut left_shifts = [0_u32; 8];
    for lane in 0..8 {
        let start = lane * WIDTH;
        let word_index = start / (u32::BITS as usize);
        let right_shift = start % (u32::BITS as usize);
        low_words[lane] = words[word_index];
        right_shifts[lane] = right_shift as u32;
        if right_shift + WIDTH > u32::BITS as usize {
            high_words[lane] = words[word_index + 1];
            left_shifts[lane] = (u32::BITS as usize - right_shift) as u32;
        }
    }

    let mask = u32::MAX >> (u32::BITS as usize - WIDTH);
    (((u32x8::from(low_words) >> u32x8::from(right_shifts))
        | (u32x8::from(high_words) << u32x8::from(left_shifts)))
        & u32x8::splat(mask))
    .to_array()
}

macro_rules! dispatch_eight {
    ($width:expr, $input:expr) => {
        match $width {
            1 => Ok(unpack_eight::<1>($input)),
            2 => Ok(unpack_eight::<2>($input)),
            3 => Ok(unpack_eight::<3>($input)),
            4 => Ok(unpack_eight::<4>($input)),
            5 => Ok(unpack_eight::<5>($input)),
            6 => Ok(unpack_eight::<6>($input)),
            7 => Ok(unpack_eight::<7>($input)),
            8 => Ok(unpack_eight::<8>($input)),
            9 => Ok(unpack_eight::<9>($input)),
            10 => Ok(unpack_eight::<10>($input)),
            11 => Ok(unpack_eight::<11>($input)),
            12 => Ok(unpack_eight::<12>($input)),
            13 => Ok(unpack_eight::<13>($input)),
            14 => Ok(unpack_eight::<14>($input)),
            15 => Ok(unpack_eight::<15>($input)),
            16 => Ok(unpack_eight::<16>($input)),
            17 => Ok(unpack_eight::<17>($input)),
            18 => Ok(unpack_eight::<18>($input)),
            19 => Ok(unpack_eight::<19>($input)),
            20 => Ok(unpack_eight::<20>($input)),
            21 => Ok(unpack_eight::<21>($input)),
            22 => Ok(unpack_eight::<22>($input)),
            23 => Ok(unpack_eight::<23>($input)),
            24 => Ok(unpack_eight::<24>($input)),
            25 => Ok(unpack_eight::<25>($input)),
            26 => Ok(unpack_eight::<26>($input)),
            27 => Ok(unpack_eight::<27>($input)),
            28 => Ok(unpack_eight::<28>($input)),
            29 => Ok(unpack_eight::<29>($input)),
            30 => Ok(unpack_eight::<30>($input)),
            31 => Ok(unpack_eight::<31>($input)),
            32 => Ok(unpack_eight::<32>($input)),
            width => Err(BitpackError::InvalidWidth { width }),
        }
    };
}

/// Decode the canonical stream eight lanes at a time with `wide::u32x8`.
pub fn unpack_wide_into(input: &[u8], width: u8, output: &mut [u32]) -> Result<(), BitpackError> {
    validate_payload(input, width, output.len())?;
    if width == 0 {
        output.fill(0);
        return Ok(());
    }

    let width_bytes = usize::from(width);
    let output_len = output.len();
    let groups = output_len / 8;
    for group in 0..groups {
        let source_start = group * width_bytes;
        let source = input.get(source_start..source_start + width_bytes).ok_or(
            BitpackError::LengthMismatch {
                expected: input.len(),
                actual: input.len(),
            },
        )?;
        let decoded = dispatch_eight!(width, source)?;
        let destination_start = group * 8;
        let destination = output
            .get_mut(destination_start..destination_start + 8)
            .ok_or(BitpackError::LengthMismatch {
                expected: output_len,
                actual: output_len,
            })?;
        destination.copy_from_slice(&decoded);
    }

    let decoded_values = groups * 8;
    if decoded_values != output_len {
        let source_start = groups * width_bytes;
        let source = input
            .get(source_start..)
            .ok_or(BitpackError::LengthMismatch {
                expected: input.len(),
                actual: input.len(),
            })?;
        let destination = output
            .get_mut(decoded_values..)
            .ok_or(BitpackError::LengthMismatch {
                expected: output_len,
                actual: output_len,
            })?;
        unpack_scalar_into(source, width, destination)?;
    }
    Ok(())
}

/// Decode a canonical bitpacked stream, routing per width to whichever
/// kernel the measured gate proved faster (bd-bz7q).
///
/// The eight-lane `wide::u32x8` kernel runs only inside the measured
/// winning band `[WIDE_UNPACK_MIN_WIDTH, WIDE_UNPACK_MAX_WIDTH]`; narrower
/// and wider widths use the scalar reservoir, which the per-width gate
/// proved at least as fast at both extremes. Both kernels decode the
/// identical canonical stream to identical values — the choice is purely
/// throughput — so the dispatcher is behaviorally transparent to callers
/// and to the differential fixtures that pin scalar/wide equality across
/// widths 0..=32.
///
/// # Errors
///
/// Returns the same typed width/padding/length failures as the underlying
/// scalar and wide unpackers.
pub fn unpack_dispatch_into(
    input: &[u8],
    width: u8,
    output: &mut [u32],
) -> Result<(), BitpackError> {
    if (WIDE_UNPACK_MIN_WIDTH..=WIDE_UNPACK_MAX_WIDTH).contains(&width) {
        unpack_wide_into(input, width, output)
    } else {
        unpack_scalar_into(input, width, output)
    }
}
