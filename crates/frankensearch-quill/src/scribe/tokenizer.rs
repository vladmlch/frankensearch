use wide::u8x32;

use crate::schema::Analyzer as AnalyzerKind;

use super::analyzer::{AnalyzedToken, AnalyzedTokenRef, TokenAnalyzer, sealed};
use super::cass::is_cass_cjk;

/// Default implementation of the shipping frankensearch analyzer.
///
/// This fuses Tantivy's `SimpleTokenizer` and `LowerCaser`: split on
/// non-alphanumeric Unicode scalar values, ASCII-lowercase in place, and use
/// the full Unicode lowercase expansion otherwise. It deliberately does not
/// enforce [`MAX_TERM_BYTES`]; admission belongs to document/query consumers
/// so a dropped document token retains its position gap.
///
/// Token boundaries are found by a one-pass 32-byte portable SIMD classifier,
/// falling back to the scalar char-walk only at non-ASCII scalars and the short
/// tail. `wide` lowers the same safe code to the available SSE2/AVX2, NEON, or
/// wasm SIMD implementation without exposing architecture intrinsics here. The
/// emitted stream is byte-parity-identical to
/// [`analyze_default_scalar_reference`] — the retained scalar oracle — which
/// the tokenizer parity tests pin. The quill crate root remains
/// `#![forbid(unsafe_code)]`.
#[derive(Debug, Clone, Default)]
pub struct FrankensearchTokenizer {
    token: AnalyzedToken,
    /// Test-only discriminator for callers that accidentally cross the owned
    /// compatibility bridge when the borrowed token stream is sufficient.
    #[cfg(test)]
    owned_bridge_calls: usize,
}

impl sealed::Sealed for FrankensearchTokenizer {}

/// Legacy one-pass 8-byte SWAR comparison retained for exact-parity tests and
/// the same-binary tokenizer microbenchmark.
#[cfg(any(test, feature = "bench-internals"))]
#[derive(Debug, Clone, Default)]
pub struct BoundaryMaskTokenizer {
    token: AnalyzedToken,
}

#[cfg(any(test, feature = "bench-internals"))]
impl sealed::Sealed for BoundaryMaskTokenizer {}

#[inline]
pub(super) fn tokenizer_next_char(text: &str, offset: usize) -> Option<(char, usize)> {
    let remaining = text.get(offset..)?;
    let first = *remaining.as_bytes().first()?;
    if first.is_ascii() {
        Some((char::from(first), offset + 1))
    } else {
        let ch = remaining.chars().next()?;
        Some((ch, offset + ch.len_utf8()))
    }
}

#[inline]
fn tokenizer_is_alphanumeric(ch: char) -> bool {
    if ch.is_ascii() {
        ch.is_ascii_alphanumeric()
    } else {
        ch.is_alphanumeric()
    }
}

#[inline]
pub(super) fn next_token_position(position: u32) -> u32 {
    position
        .checked_add(1)
        .expect("analyzed token position exceeds the u32 contract")
}

/// Bytes classified per 64-bit SWAR word.
#[cfg(any(test, feature = "bench-internals"))]
pub(super) const SWAR_LANES: usize = 8;
/// SWAR broadcast of the byte `0x01` into every lane.
#[cfg(any(test, feature = "bench-internals"))]
const SWAR_ONES: u64 = 0x0101_0101_0101_0101;
/// SWAR broadcast of the byte `0x80` (per-lane high bit) into every lane.
#[cfg(any(test, feature = "bench-internals"))]
const SWAR_HIGH: u64 = 0x8080_8080_8080_8080;

/// Bytes classified by the portable SIMD tokenizer in one pass.
const SIMD_TOKENIZER_LANES: usize = 32;

/// Return bit `lane` for ASCII alphanumeric and non-ASCII bytes in one chunk.
///
/// The caller guarantees that the complete 32-byte chunk is in bounds. A local
/// array keeps the load safe and lets `wide` select the best portable vector
/// representation for the compilation target.
#[inline]
fn simd_tokenizer_masks(bytes: &[u8], at: usize) -> (u32, u32) {
    let mut chunk = [0_u8; SIMD_TOKENIZER_LANES];
    chunk.copy_from_slice(&bytes[at..at + SIMD_TOKENIZER_LANES]);
    let lanes = u8x32::new(chunk);
    let digits = lanes.simd_ge(b'0') & lanes.simd_le(b'9');
    let uppercase = lanes.simd_ge(b'A') & lanes.simd_le(b'Z');
    let lowercase = lanes.simd_ge(b'a') & lanes.simd_le(b'z');
    let alphanumeric = (digits | uppercase | lowercase).to_bitmask();
    let non_ascii = lanes.simd_ge(0x80).to_bitmask();
    (alphanumeric, non_ascii)
}

/// Per-lane marker (`0x80` in the lane) where `lo <= byte <= hi`.
///
/// Correct for lanes whose byte is `< 0x80`, with `lo <= 128` and `hi < 128`.
/// The compare is borrow-safe *per lane*: each lane's guard (top) bit is forced
/// before the broadcast threshold is subtracted, so every lane stays in
/// `[1, 255]` (`0x80 + byte - threshold`, `byte < 0x80`, `threshold <= 128`) and
/// no borrow ever crosses a lane boundary — unlike the bare Bit-Twiddling-Hacks
/// `hasless`, whose subtraction lets a low lane's underflow corrupt the next
/// lane's marker. Lanes holding a byte `>= 0x80` produce an unspecified marker;
/// every caller pairs this with an explicit high-bit test so a non-ASCII lane
/// terminates the span before its marker is consulted.
#[inline]
#[cfg(any(test, feature = "bench-internals"))]
const fn swar_range_mark(word: u64, lo: u64, hi: u64) -> u64 {
    let guarded = word | SWAR_HIGH;
    // 0x80 in each lane whose byte is >= lo.
    let ge_lo = guarded.wrapping_sub(SWAR_ONES.wrapping_mul(lo)) & SWAR_HIGH;
    // 0x80 in each lane whose byte is > hi (i.e. >= hi + 1).
    let gt_hi = guarded.wrapping_sub(SWAR_ONES.wrapping_mul(hi + 1)) & SWAR_HIGH;
    // In range iff at-or-above lo and not above hi.
    ge_lo & !gt_hi
}

/// Per-lane marker (`0x80` in the lane) where the ASCII byte is `[0-9A-Za-z]`.
///
/// This is the SWAR equivalent of [`u8::is_ascii_alphanumeric`] and therefore of
/// [`tokenizer_is_alphanumeric`] for ASCII scalar values. See
/// [`swar_range_mark`] for the non-ASCII lane caveat.
#[inline]
#[cfg(any(test, feature = "bench-internals"))]
pub(super) const fn swar_ascii_alnum_mark(word: u64) -> u64 {
    swar_range_mark(word, b'0' as u64, b'9' as u64)
        | swar_range_mark(word, b'A' as u64, b'Z' as u64)
        | swar_range_mark(word, b'a' as u64, b'z' as u64)
}

/// Load the eight bytes at `at` as a little-endian word so lane 0 is the byte at
/// the lowest offset. The caller guarantees `at + SWAR_LANES <= bytes.len()`.
#[inline]
#[cfg(any(test, feature = "bench-internals"))]
fn swar_load(bytes: &[u8], at: usize) -> u64 {
    let mut lanes = [0_u8; SWAR_LANES];
    lanes.copy_from_slice(&bytes[at..at + SWAR_LANES]);
    u64::from_le_bytes(lanes)
}

#[cfg(any(test, feature = "bench-internals"))]
impl TokenAnalyzer for BoundaryMaskTokenizer {
    fn supports(&self, analyzer: AnalyzerKind) -> bool {
        analyzer == AnalyzerKind::FrankensearchDefault
    }

    fn analyze(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        debug_assert_eq!(analyzer, AnalyzerKind::FrankensearchDefault);
        const NO_TOKEN: usize = usize::MAX;
        let bytes = text.as_bytes();
        let len = text.len();
        let mut cursor = 0;
        let mut position = 0_u32;
        let mut offset_from = NO_TOKEN;
        let mut all_ascii = true;

        while cursor < len {
            if cursor + SWAR_LANES <= len {
                let word = swar_load(bytes, cursor);
                if word & SWAR_HIGH == 0 {
                    let alnum = swar_ascii_alnum_mark(word);
                    let prior_lane = if offset_from == NO_TOKEN { 0 } else { 0x80 };
                    let mut transitions = alnum ^ ((alnum << 8) | prior_lane);
                    while transitions != 0 {
                        let lane = (transitions.trailing_zeros() as usize) / SWAR_LANES;
                        let lane_mark = 0x80_u64 << (lane * SWAR_LANES);
                        let at = cursor + lane;
                        transitions &= !lane_mark;

                        if alnum & lane_mark != 0 {
                            debug_assert_eq!(offset_from, NO_TOKEN);
                            offset_from = at;
                            all_ascii = true;
                        } else {
                            debug_assert_ne!(offset_from, NO_TOKEN);
                            self.token.text.clear();
                            let source = &text[offset_from..at];
                            if all_ascii {
                                self.token.text.push_str(source);
                                self.token.text.make_ascii_lowercase();
                            } else {
                                for source_char in source.chars() {
                                    self.token.text.extend(source_char.to_lowercase());
                                }
                            }
                            self.token.position = position;
                            self.token.offset_from = offset_from;
                            self.token.offset_to = at;
                            self.token.position_length = 1;
                            sink(&self.token);

                            position = next_token_position(position);
                            offset_from = NO_TOKEN;
                        }
                    }
                    cursor += SWAR_LANES;
                    continue;
                }
            }

            let Some((ch, next)) = tokenizer_next_char(text, cursor) else {
                break;
            };
            if tokenizer_is_alphanumeric(ch) {
                if offset_from == NO_TOKEN {
                    offset_from = cursor;
                    all_ascii = ch.is_ascii();
                } else {
                    all_ascii &= ch.is_ascii();
                }
            } else if offset_from != NO_TOKEN {
                self.token.text.clear();
                let source = &text[offset_from..cursor];
                if all_ascii {
                    self.token.text.push_str(source);
                    self.token.text.make_ascii_lowercase();
                } else {
                    for source_char in source.chars() {
                        self.token.text.extend(source_char.to_lowercase());
                    }
                }
                self.token.position = position;
                self.token.offset_from = offset_from;
                self.token.offset_to = cursor;
                self.token.position_length = 1;
                sink(&self.token);

                position = next_token_position(position);
                offset_from = NO_TOKEN;
            }
            cursor = next;
        }

        if offset_from != NO_TOKEN {
            self.token.text.clear();
            let source = &text[offset_from..len];
            if all_ascii {
                self.token.text.push_str(source);
                self.token.text.make_ascii_lowercase();
            } else {
                for source_char in source.chars() {
                    self.token.text.extend(source_char.to_lowercase());
                }
            }
            self.token.position = position;
            self.token.offset_from = offset_from;
            self.token.offset_to = len;
            self.token.position_length = 1;
            sink(&self.token);
        }
    }

    fn bytes_reserved(&self) -> usize {
        self.token.text.capacity()
    }

    fn reset(&mut self) {
        self.token.text.clear();
        self.token.position = 0;
        self.token.offset_from = 0;
        self.token.offset_to = 0;
        self.token.position_length = 0;
    }
}

#[inline]
fn emit_normalized_token(
    scratch: &mut String,
    source: &str,
    all_ascii: bool,
    position: u32,
    offset_from: usize,
    offset_to: usize,
    sink: &mut dyn for<'token> FnMut(AnalyzedTokenRef<'token>),
) {
    if all_ascii && !source.as_bytes().iter().any(u8::is_ascii_uppercase) {
        sink(AnalyzedTokenRef {
            text: source,
            position,
            offset_from,
            offset_to,
            position_length: 1,
        });
        return;
    }

    scratch.clear();
    if all_ascii {
        scratch.push_str(source);
        scratch.make_ascii_lowercase();
    } else {
        for source_char in source.chars() {
            scratch.extend(source_char.to_lowercase());
        }
    }
    sink(AnalyzedTokenRef {
        text: scratch.as_str(),
        position,
        offset_from,
        offset_to,
        position_length: 1,
    });
}

impl FrankensearchTokenizer {
    fn analyze_default_borrowed(
        &mut self,
        text: &str,
        sink: &mut dyn for<'token> FnMut(AnalyzedTokenRef<'token>),
    ) {
        const NO_TOKEN: usize = usize::MAX;
        let bytes = text.as_bytes();
        let len = text.len();
        let mut cursor = 0;
        let mut position = 0_u32;
        let mut offset_from = NO_TOKEN;
        let mut all_ascii = true;

        while cursor < len {
            if cursor + SIMD_TOKENIZER_LANES <= len {
                let (alphanumeric, non_ascii) = simd_tokenizer_masks(bytes, cursor);
                let ascii_lanes = if non_ascii == 0 {
                    SIMD_TOKENIZER_LANES
                } else {
                    non_ascii.trailing_zeros() as usize
                };
                if ascii_lanes != 0 {
                    let valid = if ascii_lanes == SIMD_TOKENIZER_LANES {
                        u32::MAX
                    } else {
                        (1_u32 << ascii_lanes) - 1
                    };
                    let alphanumeric = alphanumeric & valid;
                    let prior_lane = u32::from(offset_from != NO_TOKEN);
                    let mut transitions =
                        (alphanumeric ^ ((alphanumeric << 1) | prior_lane)) & valid;
                    while transitions != 0 {
                        let lane = transitions.trailing_zeros() as usize;
                        let lane_mark = 1_u32 << lane;
                        let at = cursor + lane;
                        transitions &= !lane_mark;

                        if alphanumeric & lane_mark != 0 {
                            debug_assert_eq!(offset_from, NO_TOKEN);
                            offset_from = at;
                            all_ascii = true;
                        } else {
                            debug_assert_ne!(offset_from, NO_TOKEN);
                            let source = &text[offset_from..at];
                            emit_normalized_token(
                                &mut self.token.text,
                                source,
                                all_ascii,
                                position,
                                offset_from,
                                at,
                                sink,
                            );

                            position = next_token_position(position);
                            offset_from = NO_TOKEN;
                        }
                    }
                    cursor += ascii_lanes;
                    if ascii_lanes == SIMD_TOKENIZER_LANES {
                        continue;
                    }
                }
            }

            let Some((ch, next)) = tokenizer_next_char(text, cursor) else {
                break;
            };
            if tokenizer_is_alphanumeric(ch) {
                if offset_from == NO_TOKEN {
                    offset_from = cursor;
                    all_ascii = ch.is_ascii();
                } else {
                    all_ascii &= ch.is_ascii();
                }
            } else if offset_from != NO_TOKEN {
                let source = &text[offset_from..cursor];
                emit_normalized_token(
                    &mut self.token.text,
                    source,
                    all_ascii,
                    position,
                    offset_from,
                    cursor,
                    sink,
                );

                position = next_token_position(position);
                offset_from = NO_TOKEN;
            }
            cursor = next;
        }

        if offset_from != NO_TOKEN {
            let source = &text[offset_from..len];
            emit_normalized_token(
                &mut self.token.text,
                source,
                all_ascii,
                position,
                offset_from,
                len,
                sink,
            );
        }
    }
}

impl TokenAnalyzer for FrankensearchTokenizer {
    fn supports(&self, analyzer: AnalyzerKind) -> bool {
        matches!(
            analyzer,
            AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams
        )
    }

    fn analyze(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        #[cfg(test)]
        {
            self.owned_bridge_calls += 1;
        }
        let mut owned = AnalyzedToken::default();
        self.analyze_borrowed(analyzer, text, &mut |token| {
            owned.text.clear();
            owned.text.push_str(token.text);
            owned.position = token.position;
            owned.offset_from = token.offset_from;
            owned.offset_to = token.offset_to;
            owned.position_length = token.position_length;
            sink(&owned);
        });
    }

    fn analyze_borrowed(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn for<'token> FnMut(AnalyzedTokenRef<'token>),
    ) {
        debug_assert!(self.supports(analyzer));
        if analyzer == AnalyzerKind::FrankensearchDefault {
            self.analyze_default_borrowed(text, sink);
            return;
        }
        let mut delta = 0_u32;
        self.analyze_default_borrowed(text, &mut |token| {
            if token.text.is_empty() || !token.text.chars().all(is_cass_cjk) {
                let mut out = token;
                out.position = out
                    .position
                    .checked_add(delta)
                    .expect("analyzed token position exceeds the u32 contract");
                sink(out);
                return;
            }
            let bounds = token
                .text
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(token.text.len()))
                .collect::<Vec<_>>();
            if bounds.len() <= 2 {
                let mut out = token;
                out.position = out
                    .position
                    .checked_add(delta)
                    .expect("analyzed token position exceeds the u32 contract");
                sink(out);
                return;
            }
            let pairs = bounds.len() - 2;
            for pair in 0..pairs {
                let pos = token
                    .position
                    .checked_add(delta)
                    .and_then(|v| v.checked_add(u32::try_from(pair).ok()?))
                    .expect("analyzed token position exceeds the u32 contract");
                sink(AnalyzedTokenRef {
                    text: &token.text[bounds[pair]..bounds[pair + 2]],
                    position: pos,
                    offset_from: token.offset_from + bounds[pair],
                    offset_to: token.offset_from + bounds[pair + 2],
                    position_length: 1,
                });
            }
            delta = delta
                .checked_add(u32::try_from(pairs - 1).expect("CJK run exceeds u32"))
                .expect("analyzed token position exceeds the u32 contract");
        });
    }

    fn bytes_reserved(&self) -> usize {
        self.token.text.capacity()
    }
    fn reset(&mut self) {
        self.token.text.clear();
        self.token.position = 0;
        self.token.offset_from = 0;
        self.token.offset_to = 0;
        self.token.position_length = 0;
    }
}

#[cfg(test)]
impl FrankensearchTokenizer {
    pub(crate) const fn owned_bridge_calls(&self) -> usize {
        self.owned_bridge_calls
    }
}

/// Pure scalar char-walk reference for the default frankensearch analyzer.
///
/// This is the byte-parity oracle for the SWAR fast path in
/// [`FrankensearchTokenizer::analyze`] (bd-quill-e1-scribe-bejd.1): it walks one
/// Unicode scalar value at a time with no register-width classification, so the
/// two implementations share no boundary-finding code and a divergence in
/// either is caught by diffing their streams. It is also the A/B baseline arm in
/// the `tokenizer_simd_ab` bench. Emission (slice, lowercase-path selection,
/// position/offset assignment) is identical to the production analyzer.
#[cfg(any(test, feature = "bench-internals"))]
pub fn analyze_default_scalar_reference(text: &str, sink: &mut dyn FnMut(&AnalyzedToken)) {
    let mut token = AnalyzedToken::default();
    let mut cursor = 0;
    let mut position = 0_u32;

    while let Some((ch, next_cursor)) = tokenizer_next_char(text, cursor) {
        if !tokenizer_is_alphanumeric(ch) {
            cursor = next_cursor;
            continue;
        }

        let offset_from = cursor;
        let mut offset_to = next_cursor;
        let mut resume_at = next_cursor;
        let mut all_ascii = ch.is_ascii();
        while let Some((next_ch, after_next)) = tokenizer_next_char(text, resume_at) {
            if !tokenizer_is_alphanumeric(next_ch) {
                resume_at = after_next;
                break;
            }
            all_ascii &= next_ch.is_ascii();
            offset_to = after_next;
            resume_at = after_next;
        }

        token.text.clear();
        let source = &text[offset_from..offset_to];
        if all_ascii {
            token.text.push_str(source);
            token.text.make_ascii_lowercase();
        } else {
            for source_char in source.chars() {
                token.text.extend(source_char.to_lowercase());
            }
        }
        token.position = position;
        token.offset_from = offset_from;
        token.offset_to = offset_to;
        token.position_length = 1;
        sink(&token);

        position = next_token_position(position);
        cursor = resume_at;
    }
}
