#![allow(clippy::wildcard_imports)]

use std::error::Error;

use crate::schema::{FieldDescriptor, FieldKind, SchemaDescriptor};

use super::block_max::write_vint64;
use super::postings_codec::{encode_for_block, write_vint};
pub(super) use super::*;

mod bitpack_postings;
mod block_max;
mod doclen;
mod id_map_hash;
mod numeric_and_fuzz;
mod positions;
mod stats;
mod stored_meta;

pub(super) type TestResult = Result<(), Box<dyn Error>>;

pub(super) fn init_replay_tracing() {
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
}

pub(super) const NUMERIC_TEST_FIELDS: [FieldDescriptor; 2] = [
    FieldDescriptor {
        id: 0,
        name: "signed",
        kind: FieldKind::I64 {
            indexed: true,
            fast: false,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 1,
        name: "unsigned",
        kind: FieldKind::U64 {
            indexed: true,
            fast: true,
        },
        stored: false,
    },
];
pub(super) const NUMERIC_TEST_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "quiver-numeric-test-v1",
    fields: &NUMERIC_TEST_FIELDS,
};

#[allow(clippy::cast_possible_truncation)]
pub(super) fn random_u32(state: &mut u64) -> u32 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    (value.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
}

pub(super) fn values(width: u8, count: usize, seed: u64) -> Vec<u32> {
    if width == 0 {
        return vec![0; count];
    }
    let mask = u32::MAX >> (u32::BITS - u32::from(width));
    let mut state = seed;
    let mut output: Vec<u32> = (0..count).map(|_| random_u32(&mut state) & mask).collect();
    if let Some(last) = output.last_mut() {
        *last = mask;
    }
    output
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn dense_postings(count: usize, start: u32) -> Vec<Posting> {
    (0..count)
        .map(|index| Posting::new(start + index as u32, (index % 7 + 1) as u32))
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn sparse_postings(count: usize, start: u32) -> Vec<Posting> {
    (0..count)
        .map(|index| Posting::new(start + index as u32 * 11, (index % 7 + 1) as u32))
        .collect()
}

pub(super) fn postings_with_span(span: u32) -> Vec<Posting> {
    let mut postings: Vec<Posting> = (0_u32..127)
        .map(|index| Posting::new(1_000 + index, index % 5 + 1))
        .collect();
    postings.push(Posting::new(1_000 + span - 1, 9));
    postings
}

#[allow(clippy::cast_possible_truncation, clippy::unnecessary_wraps)]
pub(super) fn fixture_fieldnorm(doc_id: u32) -> Option<u8> {
    Some((doc_id % 251) as u8)
}

pub(super) fn raw_block(kind: u8, count: u8, payload: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let payload_len = u16::try_from(payload.len())?;
    let mut bytes = vec![kind, count];
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

pub(super) fn forced_for(postings: &[Posting]) -> Result<Vec<u8>, PostingCodecError> {
    let mut bytes = Vec::new();
    encode_for_block(postings, &mut bytes)?;
    Ok(bytes)
}

pub(super) fn raw_positions(
    directory: &[(u32, u64)],
    payload: &[u8],
) -> Result<Vec<u8>, Box<dyn Error>> {
    let block_count = u32::try_from(directory.len())?;
    let mut bytes = block_count.to_le_bytes().to_vec();
    for &(first_posting_ordinal, block_offset) in directory {
        write_vint(first_posting_ordinal, &mut bytes);
        write_vint64(block_offset, &mut bytes);
    }
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

pub(super) fn collect_positions(
    positions: &PositionList<'_>,
    posting_ordinal: u32,
) -> Result<Vec<u32>, PositionCodecError> {
    positions.positions_for_ordinal(posting_ordinal)?.collect()
}

pub(super) fn zero_positions(postings: &[Posting]) -> Result<Vec<u32>, PositionCodecError> {
    let count = postings.iter().try_fold(0_usize, |total, posting| {
        let freq =
            usize::try_from(posting.freq).map_err(|_| PositionCodecError::ArithmeticOverflow {
                field: "test position count",
            })?;
        total
            .checked_add(freq)
            .ok_or(PositionCodecError::ArithmeticOverflow {
                field: "test position count",
            })
    })?;
    Ok(vec![0; count])
}
