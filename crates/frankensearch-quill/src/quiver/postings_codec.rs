use super::bitpack::{self, BitpackError};
use super::postings::{
    BLOCK_HEADER_LEN, FREQ_ALL_ONE, FREQ_BITPACKED, POSTINGS_PER_BLOCK, POSTINGS_PER_BLOCK_U8,
    Posting, PostingBlockKind, PostingCodecError, VINT_KIND,
};

#[derive(Clone, Copy)]
enum FrequencyEncoding {
    AllOne,
    Bitpacked(u8),
}

pub(super) struct DecodedBlock {
    pub(super) kind: PostingBlockKind,
    pub(super) posting_count: u8,
    pub(super) byte_len: usize,
    pub(super) first_doc: u32,
    pub(super) last_doc: u32,
    pub(super) docs: [u32; POSTINGS_PER_BLOCK],
    pub(super) freqs: [u32; POSTINGS_PER_BLOCK],
}

struct PayloadReader<'a> {
    bytes: &'a [u8],
    position: usize,
    base_offset: usize,
    block_offset: usize,
}

impl<'a> PayloadReader<'a> {
    const fn new(bytes: &'a [u8], base_offset: usize, block_offset: usize) -> Self {
        Self {
            bytes,
            position: 0,
            base_offset,
            block_offset,
        }
    }

    fn absolute_position(&self) -> usize {
        self.base_offset + self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], PostingCodecError> {
        let offset = self.absolute_position();
        let remaining = self.bytes.len().saturating_sub(self.position);
        let end =
            self.position
                .checked_add(length)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset: self.block_offset,
                    field: "payload cursor",
                })?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or(PostingCodecError::Truncated {
                offset,
                needed: length,
                remaining,
            })?;
        self.position = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, PostingCodecError> {
        let offset = self.absolute_position();
        self.take(1)?
            .first()
            .copied()
            .ok_or(PostingCodecError::Truncated {
                offset,
                needed: 1,
                remaining: 0,
            })
    }

    fn read_u16(&mut self) -> Result<u16, PostingCodecError> {
        let bytes = self.take(2)?;
        let mut little_endian = [0_u8; 2];
        little_endian.copy_from_slice(bytes);
        Ok(u16::from_le_bytes(little_endian))
    }

    fn read_u32(&mut self) -> Result<u32, PostingCodecError> {
        let bytes = self.take(4)?;
        let mut little_endian = [0_u8; 4];
        little_endian.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(little_endian))
    }

    // Profile-directed inline+cold split (bd-b6tc): the 1-byte case is inlined
    // into callers to skip the non-inlined-call prologue; the rare multi-byte
    // loop is out-of-line and `#[cold]`. Byte-identical.
    #[inline]
    fn read_vint(&mut self) -> Result<u32, PostingCodecError> {
        if let Some(&byte) = self.bytes.get(self.position)
            && byte & 0x80 == 0
        {
            self.position += 1;
            return Ok(u32::from(byte));
        }
        self.read_vint_multibyte()
    }

    #[cold]
    #[inline(never)]
    fn read_vint_multibyte(&mut self) -> Result<u32, PostingCodecError> {
        let start = self.absolute_position();
        let mut value = 0_u32;
        for byte_index in 0..5 {
            let byte = self.read_u8()?;
            if byte_index == 4 && byte & 0xf0 != 0 {
                return Err(PostingCodecError::VintOverflow { offset: start });
            }
            value |= u32::from(byte & 0x7f) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint_length(value) != byte_index + 1 {
                    return Err(PostingCodecError::NonCanonicalVint { offset: start });
                }
                return Ok(value);
            }
        }
        Err(PostingCodecError::VintOverflow { offset: start })
    }

    fn finish(self) -> Result<(), PostingCodecError> {
        let remaining = self.bytes.len().saturating_sub(self.position);
        if remaining == 0 {
            Ok(())
        } else {
            Err(PostingCodecError::TrailingPayload {
                block_offset: self.block_offset,
                remaining,
            })
        }
    }
}

pub(super) fn validate_encoder_input(postings: &[Posting]) -> Result<(), PostingCodecError> {
    let mut previous = None;
    for (index, posting) in postings.iter().enumerate() {
        if posting.freq == 0 {
            return Err(PostingCodecError::ZeroFrequency {
                index,
                doc_id: posting.doc_id,
            });
        }
        if let Some(previous_doc) = previous {
            if posting.doc_id <= previous_doc {
                return Err(PostingCodecError::NonAscendingInput {
                    index,
                    previous: previous_doc,
                    doc_id: posting.doc_id,
                });
            }
        }
        previous = Some(posting.doc_id);
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
fn bit_width(value: u32) -> u8 {
    (u32::BITS - value.leading_zeros()) as u8
}

fn map_bitpack<T>(
    block_offset: usize,
    result: Result<T, BitpackError>,
) -> Result<T, PostingCodecError> {
    result.map_err(|source| PostingCodecError::Bitpack {
        block_offset,
        source,
    })
}

fn encode_frequency_parts(
    postings: &[Posting],
    block_offset: usize,
) -> Result<(Vec<u8>, Vec<u8>), PostingCodecError> {
    let stored: Vec<u32> = postings.iter().map(|posting| posting.freq - 1).collect();
    let width = stored.iter().copied().map(bit_width).max().unwrap_or(0);
    if width == 0 {
        return Ok((vec![FREQ_ALL_ONE], Vec::new()));
    }
    let packed = map_bitpack(block_offset, bitpack::pack_lsb(&stored, width))?;
    Ok((vec![FREQ_BITPACKED, width], packed))
}

fn append_block(
    output: &mut Vec<u8>,
    kind: PostingBlockKind,
    posting_count: u8,
    payload: &[u8],
) -> Result<(), PostingCodecError> {
    let block_offset = output.len();
    let payload_len =
        u16::try_from(payload.len()).map_err(|_| PostingCodecError::PayloadTooLarge {
            block_offset,
            length: payload.len(),
        })?;
    output.push(kind.code());
    output.push(posting_count);
    output.extend_from_slice(&payload_len.to_le_bytes());
    output.extend_from_slice(payload);
    Ok(())
}

pub(super) fn encode_for_block(
    postings: &[Posting],
    output: &mut Vec<u8>,
) -> Result<(), PostingCodecError> {
    let block_offset = output.len();
    let first_doc = postings.first().map_or(0, |posting| posting.doc_id);
    let deltas: Vec<u32> = postings
        .windows(2)
        .map(|pair| pair[1].doc_id - pair[0].doc_id - 1)
        .collect();
    let doc_width = deltas.iter().copied().map(bit_width).max().unwrap_or(0);
    let packed_docs = map_bitpack(block_offset, bitpack::pack_lsb(&deltas, doc_width))?;
    let (freq_meta, packed_freqs) = encode_frequency_parts(postings, block_offset)?;

    let mut payload =
        Vec::with_capacity(4 + 1 + freq_meta.len() + packed_docs.len() + packed_freqs.len());
    payload.extend_from_slice(&first_doc.to_le_bytes());
    payload.push(doc_width);
    payload.extend_from_slice(&freq_meta);
    payload.extend_from_slice(&packed_docs);
    payload.extend_from_slice(&packed_freqs);
    append_block(
        output,
        PostingBlockKind::FrameOfReference,
        POSTINGS_PER_BLOCK_U8,
        &payload,
    )
}

pub(super) fn encode_bitmap_block(
    postings: &[Posting],
    output: &mut Vec<u8>,
) -> Result<(), PostingCodecError> {
    let block_offset = output.len();
    let first_doc = postings.first().map_or(0, |posting| posting.doc_id);
    let last_doc = postings.last().map_or(first_doc, |posting| posting.doc_id);
    let span_u64 = u64::from(last_doc) - u64::from(first_doc) + 1;
    let span = u16::try_from(span_u64).map_err(|_| PostingCodecError::InvalidBitmapSpan {
        block_offset,
        span: u16::MAX,
    })?;
    if !(128..512).contains(&span) {
        return Err(PostingCodecError::InvalidBitmapSpan { block_offset, span });
    }

    let mut bitmap = [0_u8; 64];
    for posting in postings {
        let relative = usize::try_from(posting.doc_id - first_doc).map_err(|_| {
            PostingCodecError::ArithmeticOverflow {
                block_offset,
                field: "bitmap relative docid",
            }
        })?;
        let byte = bitmap
            .get_mut(relative / 8)
            .ok_or(PostingCodecError::InvalidBitmap {
                block_offset,
                invariant: "docid inside 512-bit container",
            })?;
        *byte |= 1_u8 << (relative % 8);
    }
    let (freq_meta, packed_freqs) = encode_frequency_parts(postings, block_offset)?;
    let mut payload =
        Vec::with_capacity(4 + 2 + bitmap.len() + freq_meta.len() + packed_freqs.len());
    payload.extend_from_slice(&first_doc.to_le_bytes());
    payload.extend_from_slice(&span.to_le_bytes());
    payload.extend_from_slice(&bitmap);
    payload.extend_from_slice(&freq_meta);
    payload.extend_from_slice(&packed_freqs);
    append_block(
        output,
        PostingBlockKind::Bitmap,
        POSTINGS_PER_BLOCK_U8,
        &payload,
    )
}

pub(super) fn encode_vint_block(
    postings: &[Posting],
    output: &mut Vec<u8>,
) -> Result<(), PostingCodecError> {
    if postings.is_empty() {
        return Ok(());
    }
    // The payload length is exactly computable from vint widths, so the block
    // header and bytes go straight into `output` in one reserved extension —
    // a fresh per-block Vec here grew push-by-push into six-figure realloc
    // counts per seal. Error checks stay ahead of every write so a failed
    // block still leaves `output` untouched, and the byte layout matches
    // `append_block` exactly: [kind][count][payload_len: u16 LE][payload].
    let mut payload_len = 0_usize;
    let mut previous_doc = 0_u32;
    for (index, posting) in postings.iter().enumerate() {
        let stored_doc = if index == 0 {
            posting.doc_id
        } else {
            posting.doc_id - previous_doc - 1
        };
        payload_len += vint_length(stored_doc) + vint_length(posting.freq);
        previous_doc = posting.doc_id;
    }
    let posting_count =
        u8::try_from(postings.len()).map_err(|_| PostingCodecError::InvalidBlockCount {
            offset: output.len(),
            kind: VINT_KIND,
            count: u8::MAX,
        })?;
    let block_offset = output.len();
    let payload_len_u16 =
        u16::try_from(payload_len).map_err(|_| PostingCodecError::PayloadTooLarge {
            block_offset,
            length: payload_len,
        })?;
    output.reserve(4 + payload_len);
    output.push(PostingBlockKind::Vint.code());
    output.push(posting_count);
    output.extend_from_slice(&payload_len_u16.to_le_bytes());
    let payload_start = output.len();
    let mut previous_doc = 0_u32;
    for (index, posting) in postings.iter().enumerate() {
        let stored_doc = if index == 0 {
            posting.doc_id
        } else {
            posting.doc_id - previous_doc - 1
        };
        write_vint(stored_doc, output);
        write_vint(posting.freq, output);
        previous_doc = posting.doc_id;
    }
    debug_assert_eq!(output.len() - payload_start, payload_len);
    Ok(())
}

pub(super) fn vint_length(mut value: u32) -> usize {
    let mut length = 1;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn write_vint(mut value: u32, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn read_frequency_meta(
    reader: &mut PayloadReader<'_>,
) -> Result<FrequencyEncoding, PostingCodecError> {
    match reader.read_u8()? {
        FREQ_ALL_ONE => Ok(FrequencyEncoding::AllOne),
        FREQ_BITPACKED => Ok(FrequencyEncoding::Bitpacked(reader.read_u8()?)),
        kind => Err(PostingCodecError::InvalidFrequencyKind {
            block_offset: reader.block_offset,
            kind,
        }),
    }
}

fn decode_frequencies(
    reader: &mut PayloadReader<'_>,
    encoding: FrequencyEncoding,
    count: usize,
    output: &mut [u32; POSTINGS_PER_BLOCK],
) -> Result<(), PostingCodecError> {
    match encoding {
        FrequencyEncoding::AllOne => output[..count].fill(1),
        FrequencyEncoding::Bitpacked(width) => {
            let packed_len =
                map_bitpack(reader.block_offset, bitpack::packed_byte_len(count, width))?;
            let bytes = reader.take(packed_len)?;
            let mut stored = [0_u32; POSTINGS_PER_BLOCK];
            map_bitpack(
                reader.block_offset,
                bitpack::unpack_dispatch_into(bytes, width, &mut stored[..count]),
            )?;
            let required = stored[..count]
                .iter()
                .copied()
                .map(bit_width)
                .max()
                .unwrap_or(0);
            if required == 0 {
                return Err(PostingCodecError::NonCanonicalFrequencyEncoding {
                    block_offset: reader.block_offset,
                });
            }
            if width != required {
                return Err(PostingCodecError::NonCanonicalWidth {
                    block_offset: reader.block_offset,
                    field: "frequency",
                    encoded: width,
                    required,
                });
            }
            for (destination, stored_freq) in output[..count]
                .iter_mut()
                .zip(stored[..count].iter().copied())
            {
                *destination =
                    stored_freq
                        .checked_add(1)
                        .ok_or(PostingCodecError::ArithmeticOverflow {
                            block_offset: reader.block_offset,
                            field: "frequency",
                        })?;
            }
        }
    }
    Ok(())
}

pub(super) fn decode_block_at(
    bytes: &[u8],
    offset: usize,
    base_posting_ordinal: usize,
) -> Result<DecodedBlock, PostingCodecError> {
    let term_tail = bytes.get(offset..).ok_or(PostingCodecError::Truncated {
        offset,
        needed: BLOCK_HEADER_LEN,
        remaining: 0,
    })?;
    let mut header = PayloadReader::new(term_tail, offset, offset);
    let kind_code = header.read_u8()?;
    let posting_count = header.read_u8()?;
    let payload_len = usize::from(header.read_u16()?);
    let kind = PostingBlockKind::from_code(kind_code, offset)?;
    let valid_count = match kind {
        PostingBlockKind::FrameOfReference | PostingBlockKind::Bitmap => {
            usize::from(posting_count) == POSTINGS_PER_BLOCK
        }
        PostingBlockKind::Vint => (1..POSTINGS_PER_BLOCK).contains(&usize::from(posting_count)),
    };
    if !valid_count {
        return Err(PostingCodecError::InvalidBlockCount {
            offset,
            kind: kind_code,
            count: posting_count,
        });
    }
    let payload_start =
        offset
            .checked_add(BLOCK_HEADER_LEN)
            .ok_or(PostingCodecError::ArithmeticOverflow {
                block_offset: offset,
                field: "payload start",
            })?;
    let payload = header.take(payload_len)?;
    let byte_len =
        BLOCK_HEADER_LEN
            .checked_add(payload_len)
            .ok_or(PostingCodecError::ArithmeticOverflow {
                block_offset: offset,
                field: "block byte length",
            })?;

    let mut decoded = match kind {
        PostingBlockKind::FrameOfReference => decode_for_payload(payload, payload_start, offset)?,
        PostingBlockKind::Bitmap => decode_bitmap_payload(payload, payload_start, offset)?,
        PostingBlockKind::Vint => decode_vint_payload(
            payload,
            payload_start,
            offset,
            usize::from(posting_count),
            base_posting_ordinal,
        )?,
    };
    decoded.kind = kind;
    decoded.posting_count = posting_count;
    decoded.byte_len = byte_len;
    Ok(decoded)
}

fn decode_for_payload(
    payload: &[u8],
    payload_start: usize,
    block_offset: usize,
) -> Result<DecodedBlock, PostingCodecError> {
    let mut reader = PayloadReader::new(payload, payload_start, block_offset);
    let first_doc = reader.read_u32()?;
    let doc_width = reader.read_u8()?;
    let freq_encoding = read_frequency_meta(&mut reader)?;
    let packed_doc_len = map_bitpack(
        block_offset,
        bitpack::packed_byte_len(POSTINGS_PER_BLOCK - 1, doc_width),
    )?;
    let packed_docs = reader.take(packed_doc_len)?;
    let mut stored_deltas = [0_u32; POSTINGS_PER_BLOCK - 1];
    map_bitpack(
        block_offset,
        bitpack::unpack_dispatch_into(packed_docs, doc_width, &mut stored_deltas),
    )?;
    let required_width = stored_deltas
        .iter()
        .copied()
        .map(bit_width)
        .max()
        .unwrap_or(0);
    if doc_width != required_width {
        return Err(PostingCodecError::NonCanonicalWidth {
            block_offset,
            field: "doc delta",
            encoded: doc_width,
            required: required_width,
        });
    }

    let mut docs = [0_u32; POSTINGS_PER_BLOCK];
    docs[0] = first_doc;
    for index in 1..POSTINGS_PER_BLOCK {
        let step = stored_deltas[index - 1].checked_add(1).ok_or(
            PostingCodecError::ArithmeticOverflow {
                block_offset,
                field: "doc delta",
            },
        )?;
        docs[index] =
            docs[index - 1]
                .checked_add(step)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "docid",
                })?;
    }
    let mut freqs = [0_u32; POSTINGS_PER_BLOCK];
    decode_frequencies(&mut reader, freq_encoding, POSTINGS_PER_BLOCK, &mut freqs)?;
    reader.finish()?;
    Ok(DecodedBlock {
        kind: PostingBlockKind::FrameOfReference,
        posting_count: POSTINGS_PER_BLOCK_U8,
        byte_len: 0,
        first_doc,
        last_doc: docs[POSTINGS_PER_BLOCK - 1],
        docs,
        freqs,
    })
}

fn decode_bitmap_payload(
    payload: &[u8],
    payload_start: usize,
    block_offset: usize,
) -> Result<DecodedBlock, PostingCodecError> {
    let mut reader = PayloadReader::new(payload, payload_start, block_offset);
    let first_doc = reader.read_u32()?;
    let span = reader.read_u16()?;
    if !(128..512).contains(&span) {
        return Err(PostingCodecError::InvalidBitmapSpan { block_offset, span });
    }
    let bitmap = reader.take(64)?;
    let last_relative = usize::from(span) - 1;
    let first_is_set = bitmap.first().is_some_and(|byte| byte & 1 != 0);
    let last_is_set = bitmap
        .get(last_relative / 8)
        .is_some_and(|byte| byte & (1_u8 << (last_relative % 8)) != 0);
    if !first_is_set || !last_is_set {
        return Err(PostingCodecError::InvalidBitmap {
            block_offset,
            invariant: "first and last span bits set",
        });
    }

    let mut docs = [0_u32; POSTINGS_PER_BLOCK];
    let mut cardinality = 0_usize;
    for (byte_index, &byte) in bitmap.iter().enumerate() {
        // Iterate only the set bits (ascending) via `trailing_zeros` instead of
        // testing all 8 positions: the bitmap holds 128 set bits across a
        // 128..512 span, so larger (sparser) spans skip up to 384 zero bits
        // (profile-directed, bd-udz8). Byte-identical — same relative values in
        // the same order, same span/cardinality checks.
        let mut bits = byte;
        while bits != 0 {
            let bit_index = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let relative = byte_index * 8 + bit_index;
            if relative >= usize::from(span) {
                return Err(PostingCodecError::InvalidBitmap {
                    block_offset,
                    invariant: "zero bits beyond inclusive span",
                });
            }
            let destination =
                docs.get_mut(cardinality)
                    .ok_or(PostingCodecError::InvalidBitmap {
                        block_offset,
                        invariant: "exactly 128 set bits",
                    })?;
            *destination = first_doc
                .checked_add(u32::try_from(relative).map_err(|_| {
                    PostingCodecError::ArithmeticOverflow {
                        block_offset,
                        field: "bitmap docid",
                    }
                })?)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "bitmap docid",
                })?;
            cardinality += 1;
        }
    }
    if cardinality != POSTINGS_PER_BLOCK {
        return Err(PostingCodecError::InvalidBitmap {
            block_offset,
            invariant: "exactly 128 set bits",
        });
    }

    let freq_encoding = read_frequency_meta(&mut reader)?;
    let mut freqs = [0_u32; POSTINGS_PER_BLOCK];
    decode_frequencies(&mut reader, freq_encoding, POSTINGS_PER_BLOCK, &mut freqs)?;
    reader.finish()?;
    Ok(DecodedBlock {
        kind: PostingBlockKind::Bitmap,
        posting_count: POSTINGS_PER_BLOCK_U8,
        byte_len: 0,
        first_doc,
        last_doc: docs[POSTINGS_PER_BLOCK - 1],
        docs,
        freqs,
    })
}

fn decode_vint_payload(
    payload: &[u8],
    payload_start: usize,
    block_offset: usize,
    count: usize,
    base_posting_ordinal: usize,
) -> Result<DecodedBlock, PostingCodecError> {
    let mut reader = PayloadReader::new(payload, payload_start, block_offset);
    let mut docs = [0_u32; POSTINGS_PER_BLOCK];
    let mut freqs = [0_u32; POSTINGS_PER_BLOCK];
    for index in 0..count {
        let stored_doc = reader.read_vint()?;
        let doc_id = if index == 0 {
            stored_doc
        } else {
            let step = stored_doc
                .checked_add(1)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "vint doc delta",
                })?;
            docs[index - 1]
                .checked_add(step)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "vint docid",
                })?
        };
        let freq = reader.read_vint()?;
        if freq == 0 {
            let posting_ordinal = base_posting_ordinal.checked_add(index).ok_or(
                PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "posting ordinal",
                },
            )?;
            return Err(PostingCodecError::ZeroFrequency {
                index: posting_ordinal,
                doc_id,
            });
        }
        docs[index] = doc_id;
        freqs[index] = freq;
    }
    reader.finish()?;
    Ok(DecodedBlock {
        kind: PostingBlockKind::Vint,
        posting_count: u8::try_from(count).unwrap_or(u8::MAX),
        byte_len: 0,
        first_doc: docs[0],
        last_doc: docs[count - 1],
        docs,
        freqs,
    })
}
