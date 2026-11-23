use super::block_max::vint64_length;
use super::positions::{
    POSITION_DIRECTORY_HEADER_LEN, PositionBlockMeta, PositionCodecError, RawPositionBlock,
};
use super::postings::{Posting, PostingBlockMeta, PostingCursor, PostingList};
use super::postings_codec::{vint_length, write_vint};

/// Streaming decoder for one frequency-bounded document position run.
pub struct PositionIter<'a> {
    pub(super) reader: PositionByteReader<'a>,
    pub(super) posting_ordinal: u32,
    pub(super) remaining: u32,
    pub(super) previous: Option<u32>,
}

impl Iterator for PositionIter<'_> {
    type Item = Result<u32, PositionCodecError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let encoded = match self.reader.read_u32_vint() {
            Ok(value) => value,
            Err(error) => {
                self.remaining = 0;
                return Some(Err(error));
            }
        };
        let position = if let Some(previous) = self.previous {
            match previous.checked_add(encoded) {
                Some(position) => position,
                None => {
                    self.remaining = 0;
                    return Some(Err(PositionCodecError::PositionOverflow {
                        posting_ordinal: self.posting_ordinal,
                        previous,
                        delta: encoded,
                    }));
                }
            }
        } else {
            encoded
        };
        self.previous = Some(position);
        self.remaining -= 1;
        Some(Ok(position))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::try_from(self.remaining).unwrap_or(usize::MAX);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PositionIter<'_> {}

/// Posting cursor paired with allocation-free access to the current positions.
pub struct PositionCursor<'a> {
    pub(super) position_bytes: &'a [u8],
    pub(super) position_blocks: &'a [PositionBlockMeta],
    pub(super) position_block_index: Option<usize>,
    pub(super) current_position_offset: usize,
    pub(super) current_positions_consumed: bool,
    pub(super) position_reader: PositionByteReader<'a>,
    pub(super) postings: PostingCursor<'a>,
}

impl PositionCursor<'_> {
    /// Current posting, including a valid `u32::MAX` docid when present.
    #[must_use]
    pub fn current(&self) -> Option<Posting> {
        self.postings.current()
    }

    /// Current absolute docid.
    #[must_use]
    pub fn doc(&self) -> Option<u32> {
        self.postings.doc()
    }

    /// Current term frequency.
    #[must_use]
    pub fn freq(&self) -> Option<u32> {
        self.postings.freq()
    }

    /// Current zero-based posting ordinal.
    #[must_use]
    pub fn posting_ordinal(&self) -> Option<u32> {
        self.postings.posting_ordinal()
    }

    /// Current zero-based POSTINGS block index.
    #[must_use]
    pub const fn posting_block_index(&self) -> Option<usize> {
        self.postings.block_index()
    }

    /// Stream the current posting's absolute positions without allocation.
    ///
    /// # Errors
    ///
    /// Returns an internal invariant failure if the paired validated streams
    /// disagree.
    pub fn positions(&self) -> Result<Option<PositionIter<'_>>, PositionCodecError> {
        let Some(posting_ordinal) = self.postings.posting_ordinal() else {
            return Ok(None);
        };
        let freq = self
            .postings
            .freq()
            .ok_or(PositionCodecError::CursorInvariant {
                field: "current position frequency",
            })?;
        if self.position_block_index.is_none() {
            return Err(PositionCodecError::CursorInvariant {
                field: "current position block",
            });
        }
        let reader = if self.current_positions_consumed {
            PositionByteReader::new(
                self.position_bytes,
                self.current_position_offset,
                self.position_reader.end,
            )?
        } else {
            self.position_reader.clone()
        };
        Ok(Some(PositionIter {
            reader,
            posting_ordinal,
            remaining: freq,
            previous: None,
        }))
    }

    /// Decode the current run while advancing the cursor-owned POSITIONS
    /// reader to the next run.
    ///
    /// A repeated call for the same posting remains correct by replaying from
    /// the retained run boundary, but only the first successful call mutates
    /// the owned reader. On every failure both the owned reader and the
    /// consumed marker remain unchanged and `output` is empty.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, corruption, or paired-stream invariant
    /// failure.
    pub(crate) fn decode_current_positions_into(
        &mut self,
        output: &mut Vec<u32>,
    ) -> Result<bool, PositionCodecError> {
        output.clear();
        let Some(posting_ordinal) = self.postings.posting_ordinal() else {
            return Ok(false);
        };
        let freq = self
            .postings
            .freq()
            .ok_or(PositionCodecError::CursorInvariant {
                field: "current position frequency",
            })?;
        if self.position_block_index.is_none() {
            return Err(PositionCodecError::CursorInvariant {
                field: "current position block",
            });
        }
        let expected =
            usize::try_from(freq).map_err(|_| PositionCodecError::ArithmeticOverflow {
                field: "current position count",
            })?;
        output
            .try_reserve_exact(expected)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "decoded current positions",
                count: expected,
            })?;

        let mut next_reader = if self.current_positions_consumed {
            PositionByteReader::new(
                self.position_bytes,
                self.current_position_offset,
                self.position_reader.end,
            )?
        } else {
            self.position_reader.clone()
        };
        if let Err(error) = decode_position_run(&mut next_reader, posting_ordinal, freq, output) {
            output.clear();
            return Err(error);
        }
        if !self.current_positions_consumed {
            self.position_reader = next_reader;
            self.current_positions_consumed = true;
        }
        Ok(true)
    }

    /// Move strictly forward by one posting. Exhaustion is fused.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the paired POSTINGS cursor cannot decode its
    /// next validated block.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Posting>, PositionCodecError> {
        self.advance_one()
    }

    /// Land on the first posting whose docid is at least `target`.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the paired POSTINGS cursor cannot decode the
    /// selected validated block.
    pub fn advance(&mut self, target: u32) -> Result<Option<Posting>, PositionCodecError> {
        let Some(current) = self.postings.current() else {
            return Ok(None);
        };
        if current.doc_id >= target {
            return Ok(Some(current));
        }

        let destination = self
            .position_blocks
            .partition_point(|block| block.last_doc < target);
        let Some(destination_block) = self.position_blocks.get(destination) else {
            let landed = self.postings.advance(target)?;
            if landed.is_some() {
                return Err(PositionCodecError::CursorInvariant {
                    field: "position advance exhaustion",
                });
            }
            self.position_block_index = None;
            self.position_reader = PositionByteReader::new(
                self.position_bytes,
                self.position_bytes.len(),
                self.position_bytes.len(),
            )?;
            self.current_position_offset = self.position_reader.position();
            self.current_positions_consumed = false;
            return Ok(None);
        };

        let current_block =
            self.position_block_index
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position advance current block",
                })?;
        if destination > current_block {
            let landed = self.postings.advance(destination_block.first_doc)?;
            if landed.map(|posting| posting.doc_id) != Some(destination_block.first_doc)
                || self.postings.posting_ordinal() != Some(destination_block.base_posting_ordinal)
            {
                return Err(PositionCodecError::CursorInvariant {
                    field: "position advance block seek",
                });
            }
            self.position_block_index = Some(destination);
            self.position_reader =
                PositionByteReader::from_block(self.position_bytes, destination_block)?;
            self.current_position_offset = self.position_reader.position();
            self.current_positions_consumed = false;
        } else if destination < current_block {
            return Err(PositionCodecError::CursorInvariant {
                field: "position advance moved backward",
            });
        }

        while self.postings.doc().is_some_and(|doc| doc < target) {
            self.advance_one()?;
        }
        Ok(self.postings.current())
    }

    /// Number of POSTINGS block decodes [`Self::next`] would perform.
    ///
    /// `Self::advance_one` touches the paired posting cursor exactly once,
    /// so a positional step decodes whatever that single step decodes.
    #[must_use]
    pub fn next_decode_count(&self) -> u64 {
        u64::from(self.postings.next_decodes_block())
    }

    /// POSTINGS blocks the paired posting cursor has actually decoded.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn decoded_blocks(&self) -> usize {
        self.postings.decoded_blocks()
    }

    /// Number of POSTINGS block decodes [`Self::advance`] would perform.
    ///
    /// A positional seek is two-stage and can decode more than one block, so
    /// a caller admitting before the work needs the count, not a flag. This
    /// mirrors [`Self::advance`] stage by stage from block metadata alone:
    ///
    /// 1. A satisfied target, an unresolvable positions block index, or a
    ///    backward destination decode nothing — `advance` returns or fails
    ///    before touching the posting cursor.
    /// 2. With no positions block for the target, `advance` seeks the posting
    ///    cursor straight at the target, decoding at most its landing block.
    /// 3. A forward destination seeks to that block's `first_doc`, decoding
    ///    only the block it lands in, because a direct seek skips the blocks
    ///    between without reading them.
    /// 4. The remaining walk to `target` steps posting by posting, and
    ///    stepping decodes *every* block it crosses — which is why this
    ///    counts the span rather than assuming one.
    ///
    /// A conversion that cannot fit `u64` saturates upward: over-admitting is
    /// a visible over-charge, while under-admitting would let a decode run
    /// unadmitted.
    #[must_use]
    pub fn advance_decode_count(&self, target: u32) -> u64 {
        let Some(current) = self.postings.current() else {
            return 0;
        };
        if current.doc_id >= target {
            return 0;
        }
        let Some(current_posting_block) = self.postings.block_index() else {
            return 0;
        };

        let destination = self
            .position_blocks
            .partition_point(|block| block.last_doc < target);
        let Some(destination_block) = self.position_blocks.get(destination) else {
            return u64::from(self.postings.advance_decodes_block(target));
        };
        let Some(current_block) = self.position_block_index else {
            return 0;
        };
        if destination < current_block {
            return 0;
        }

        let mut decodes = 0_u64;
        let mut block = current_posting_block;
        if destination > current_block {
            let Some(landing) = self.postings.block_containing(destination_block.first_doc) else {
                return decodes;
            };
            if landing != block {
                decodes += 1;
                block = landing;
            }
        }

        let final_block = self
            .postings
            .block_containing(target)
            .unwrap_or_else(|| self.postings.block_count().saturating_sub(1));
        decodes.saturating_add(u64::try_from(final_block.saturating_sub(block)).unwrap_or(u64::MAX))
    }

    fn advance_one(&mut self) -> Result<Option<Posting>, PositionCodecError> {
        let Some(posting_ordinal) = self.postings.posting_ordinal() else {
            return Ok(None);
        };
        let freq = self
            .postings
            .freq()
            .ok_or(PositionCodecError::CursorInvariant {
                field: "position cursor frequency",
            })?;
        if !self.current_positions_consumed {
            consume_position_run(&mut self.position_reader, posting_ordinal, freq)?;
        }
        let next = self.postings.next()?;
        let Some(next_posting) = next else {
            if !self.position_reader.is_empty() {
                return Err(PositionCodecError::CursorInvariant {
                    field: "final position block exhaustion",
                });
            }
            self.position_block_index = None;
            self.current_position_offset = self.position_reader.position();
            self.current_positions_consumed = false;
            return Ok(None);
        };

        let block_index = self
            .position_block_index
            .ok_or(PositionCodecError::CursorInvariant {
                field: "position cursor block index",
            })?;
        let block =
            self.position_blocks
                .get(block_index)
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position cursor block metadata",
                })?;
        let block_end_ordinal = block
            .base_posting_ordinal
            .checked_add(block.posting_count)
            .ok_or(PositionCodecError::ArithmeticOverflow {
                field: "position cursor block end ordinal",
            })?;
        let next_ordinal =
            self.postings
                .posting_ordinal()
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position cursor next ordinal",
                })?;
        if next_ordinal == block_end_ordinal {
            if !self.position_reader.is_empty() {
                return Err(PositionCodecError::CursorInvariant {
                    field: "position block exhaustion",
                });
            }
            let next_block_index =
                block_index
                    .checked_add(1)
                    .ok_or(PositionCodecError::ArithmeticOverflow {
                        field: "next position block index",
                    })?;
            let next_block = self.position_blocks.get(next_block_index).ok_or(
                PositionCodecError::CursorInvariant {
                    field: "next position block metadata",
                },
            )?;
            if next_block.base_posting_ordinal != next_ordinal
                || next_block.first_doc != next_posting.doc_id
            {
                return Err(PositionCodecError::CursorInvariant {
                    field: "next position block alignment",
                });
            }
            self.position_block_index = Some(next_block_index);
            self.position_reader = PositionByteReader::from_block(self.position_bytes, next_block)?;
        } else if next_ordinal > block_end_ordinal {
            return Err(PositionCodecError::CursorInvariant {
                field: "position cursor skipped block seam",
            });
        }
        self.current_position_offset = self.position_reader.position();
        self.current_positions_consumed = false;
        Ok(Some(next_posting))
    }
}

#[derive(Clone, Debug)]
pub(super) struct PositionByteReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
    pub(super) end: usize,
}

impl<'a> PositionByteReader<'a> {
    pub(super) fn new(
        bytes: &'a [u8],
        start: usize,
        end: usize,
    ) -> Result<Self, PositionCodecError> {
        if start > end || end > bytes.len() {
            return Err(PositionCodecError::Truncated {
                offset: start.min(bytes.len()),
                needed: end.saturating_sub(start),
                remaining: bytes.len().saturating_sub(start.min(bytes.len())),
            });
        }
        Ok(Self {
            bytes,
            cursor: start,
            end,
        })
    }

    pub(super) fn from_block(
        bytes: &'a [u8],
        block: &PositionBlockMeta,
    ) -> Result<Self, PositionCodecError> {
        let end = block.byte_offset.checked_add(block.byte_len).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "position block reader end",
            },
        )?;
        Self::new(bytes, block.byte_offset, end)
    }

    pub(super) const fn position(&self) -> usize {
        self.cursor
    }

    pub(super) const fn remaining(&self) -> usize {
        self.end - self.cursor
    }

    pub(super) const fn is_empty(&self) -> bool {
        self.cursor == self.end
    }

    fn read_byte(&mut self) -> Result<u8, PositionCodecError> {
        let byte =
            self.bytes
                .get(self.cursor)
                .copied()
                .ok_or_else(|| PositionCodecError::Truncated {
                    offset: self.cursor,
                    needed: 1,
                    remaining: self.remaining(),
                })?;
        if self.cursor >= self.end {
            return Err(PositionCodecError::Truncated {
                offset: self.cursor,
                needed: 1,
                remaining: 0,
            });
        }
        self.cursor += 1;
        Ok(byte)
    }

    // Profile-directed (bd-b6tc): `read_u32_vint` was 11% of query self-time,
    // paying a non-inlined call's prologue on every gap-encoded position. The
    // 1-byte case (value < 128) dominates positions and is always canonical, so
    // it is inlined into callers while the rare multi-byte loop is out-of-line
    // and `#[cold]`. Byte-identical to the former single function.
    #[inline]
    pub(super) fn read_u32_vint(&mut self) -> Result<u32, PositionCodecError> {
        if self.cursor < self.end
            && let Some(&byte) = self.bytes.get(self.cursor)
            && byte & 0x80 == 0
        {
            self.cursor += 1;
            return Ok(u32::from(byte));
        }
        self.read_u32_vint_multibyte()
    }

    #[cold]
    #[inline(never)]
    fn read_u32_vint_multibyte(&mut self) -> Result<u32, PositionCodecError> {
        let start = self.cursor;
        let mut value = 0_u32;
        for byte_index in 0..5 {
            let byte = self.read_byte()?;
            let payload = byte & 0x7f;
            if byte_index == 4 && payload > 0x0f {
                return Err(PositionCodecError::VintOverflow {
                    offset: start,
                    domain: "u32",
                });
            }
            value |= u32::from(payload) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint_length(value) != byte_index + 1 {
                    return Err(PositionCodecError::NonCanonicalVint {
                        offset: start,
                        domain: "u32",
                    });
                }
                return Ok(value);
            }
        }
        Err(PositionCodecError::VintOverflow {
            offset: start,
            domain: "u32",
        })
    }

    // Profile-directed inline+cold split (bd-b6tc): 1-byte fast path inlined,
    // multi-byte loop out-of-line `#[cold]`. Byte-identical.
    #[inline]
    pub(super) fn read_u64_vint(&mut self) -> Result<u64, PositionCodecError> {
        if self.cursor < self.end
            && let Some(&byte) = self.bytes.get(self.cursor)
            && byte & 0x80 == 0
        {
            self.cursor += 1;
            return Ok(u64::from(byte));
        }
        self.read_u64_vint_multibyte()
    }

    #[cold]
    #[inline(never)]
    fn read_u64_vint_multibyte(&mut self) -> Result<u64, PositionCodecError> {
        let start = self.cursor;
        let mut value = 0_u64;
        for byte_index in 0..10 {
            let byte = self.read_byte()?;
            let payload = byte & 0x7f;
            if byte_index == 9 && payload > 1 {
                return Err(PositionCodecError::VintOverflow {
                    offset: start,
                    domain: "u64",
                });
            }
            value |= u64::from(payload) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint64_length(value) != byte_index + 1 {
                    return Err(PositionCodecError::NonCanonicalVint {
                        offset: start,
                        domain: "u64",
                    });
                }
                return Ok(value);
            }
        }
        Err(PositionCodecError::VintOverflow {
            offset: start,
            domain: "u64",
        })
    }
}

pub(super) fn expected_position_count(
    postings: &PostingList<'_>,
    limit: u64,
) -> Result<u64, PositionCodecError> {
    let mut total = 0_u64;
    let mut cursor = postings.cursor()?;
    while let Some(posting) = cursor.current() {
        total = total.checked_add(u64::from(posting.freq)).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "frequency-derived position count",
            },
        )?;
        if total > limit {
            return Err(PositionCodecError::PositionLimitExceeded {
                limit,
                actual: total,
            });
        }
        cursor.next()?;
    }
    Ok(total)
}

pub(super) fn expected_input_position_count(
    postings: &[Posting],
    limit: u64,
) -> Result<u64, PositionCodecError> {
    let mut total = 0_u64;
    for posting in postings {
        total = total.checked_add(u64::from(posting.freq)).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "frequency-derived position count",
            },
        )?;
        if total > limit {
            return Err(PositionCodecError::PositionLimitExceeded {
                limit,
                actual: total,
            });
        }
    }
    Ok(total)
}

pub(super) fn validated_position_run_len(
    posting_ordinal: u32,
    positions: &[u32],
) -> Result<usize, PositionCodecError> {
    let Some((&first, rest)) = positions.split_first() else {
        return Ok(0);
    };
    let mut length = vint_length(first);
    let mut previous = first;
    for (index, &position) in rest.iter().enumerate() {
        if position < previous {
            return Err(PositionCodecError::NonAscendingPosition {
                posting_ordinal,
                position_index: index + 1,
                previous,
                position,
            });
        }
        length = length.checked_add(vint_length(position - previous)).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "encoded position run length",
            },
        )?;
        previous = position;
    }
    Ok(length)
}

pub(super) fn write_position_run(positions: &[u32], output: &mut Vec<u8>) {
    let Some((&first, rest)) = positions.split_first() else {
        return;
    };
    write_vint(first, output);
    let mut previous = first;
    for &position in rest {
        write_vint(position - previous, output);
        previous = position;
    }
}

pub(super) fn position_directory_len(
    directory: &[RawPositionBlock],
) -> Result<usize, PositionCodecError> {
    let mut length = POSITION_DIRECTORY_HEADER_LEN;
    for block in directory {
        length = length
            .checked_add(vint_length(block.first_posting_ordinal))
            .and_then(|value| value.checked_add(vint64_length(block.block_offset)))
            .ok_or(PositionCodecError::ArithmeticOverflow {
                field: "position directory length",
            })?;
    }
    Ok(length)
}

#[cfg(test)]
std::thread_local! {
    static POSITION_RUN_WORK_FOR_TEST: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
pub fn reset_position_run_work_for_test() {
    POSITION_RUN_WORK_FOR_TEST.set((0, 0));
}

#[cfg(test)]
pub fn position_run_work_for_test() -> (usize, usize) {
    POSITION_RUN_WORK_FOR_TEST.get()
}

#[cfg(test)]
fn record_position_run_work_for_test(decoded: bool) {
    let (owned_decodes, scan_consumes) = POSITION_RUN_WORK_FOR_TEST.get();
    POSITION_RUN_WORK_FOR_TEST.set(if decoded {
        (owned_decodes.saturating_add(1), scan_consumes)
    } else {
        (owned_decodes, scan_consumes.saturating_add(1))
    });
}

pub(super) fn consume_position_run(
    reader: &mut PositionByteReader<'_>,
    posting_ordinal: u32,
    freq: u32,
) -> Result<(), PositionCodecError> {
    // Peel the first position out of the hot loop so the remaining iterations
    // drop the per-position `Option` match and re-wrap that distinguished it
    // (profile-directed, bd-2fha: this run was ~14% of query self-time). The
    // first position is the raw delta with no predecessor; every later position
    // is a checked prefix sum — byte-identical to the former `Option` form.
    if freq == 0 {
        return Ok(());
    }
    let mut previous = reader.read_u32_vint()?;
    for _ in 1..freq {
        let encoded = reader.read_u32_vint()?;
        previous = previous
            .checked_add(encoded)
            .ok_or(PositionCodecError::PositionOverflow {
                posting_ordinal,
                previous,
                delta: encoded,
            })?;
    }
    #[cfg(test)]
    record_position_run_work_for_test(false);
    Ok(())
}

fn decode_position_run(
    reader: &mut PositionByteReader<'_>,
    posting_ordinal: u32,
    freq: u32,
    output: &mut Vec<u32>,
) -> Result<(), PositionCodecError> {
    if freq == 0 {
        return Ok(());
    }
    let mut previous = reader.read_u32_vint()?;
    output.push(previous);
    for _ in 1..freq {
        let encoded = reader.read_u32_vint()?;
        previous = previous
            .checked_add(encoded)
            .ok_or(PositionCodecError::PositionOverflow {
                posting_ordinal,
                previous,
                delta: encoded,
            })?;
        output.push(previous);
    }
    #[cfg(test)]
    record_position_run_work_for_test(true);
    Ok(())
}

pub(super) fn positions_for_ordinal<'a>(
    position_bytes: &'a [u8],
    position_blocks: &[PositionBlockMeta],
    posting_bytes: &[u8],
    posting_blocks: &[PostingBlockMeta],
    doc_freq: u32,
    posting_ordinal: u32,
) -> Result<PositionIter<'a>, PositionCodecError> {
    if posting_ordinal >= doc_freq {
        return Err(PositionCodecError::PostingOrdinalOutOfRange {
            posting_ordinal,
            doc_freq,
        });
    }
    let after =
        position_blocks.partition_point(|block| block.base_posting_ordinal <= posting_ordinal);
    let block_index = after
        .checked_sub(1)
        .ok_or(PositionCodecError::CursorInvariant {
            field: "position block lookup",
        })?;
    let block = position_blocks
        .get(block_index)
        .ok_or(PositionCodecError::CursorInvariant {
            field: "position block index",
        })?;
    let block_end_ordinal = block
        .base_posting_ordinal
        .checked_add(block.posting_count)
        .ok_or(PositionCodecError::ArithmeticOverflow {
            field: "position block end ordinal",
        })?;
    if posting_ordinal >= block_end_ordinal {
        return Err(PositionCodecError::CursorInvariant {
            field: "position block ordinal range",
        });
    }
    let block_end = block.byte_offset.checked_add(block.byte_len).ok_or(
        PositionCodecError::ArithmeticOverflow {
            field: "position block end",
        },
    )?;
    let mut reader = PositionByteReader::new(position_bytes, block.byte_offset, block_end)?;
    let mut postings = PostingCursor::new(posting_bytes, posting_blocks)?;
    let first = postings
        .advance(block.first_doc)?
        .ok_or(PositionCodecError::CursorInvariant {
            field: "position block first posting seek",
        })?;
    if first.doc_id != block.first_doc
        || postings.posting_ordinal() != Some(block.base_posting_ordinal)
    {
        return Err(PositionCodecError::CursorInvariant {
            field: "position block first posting metadata",
        });
    }

    loop {
        let current_ordinal =
            postings
                .posting_ordinal()
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position ordinal scan",
                })?;
        let freq = postings.freq().ok_or(PositionCodecError::CursorInvariant {
            field: "position frequency scan",
        })?;
        if current_ordinal == posting_ordinal {
            return Ok(PositionIter {
                reader,
                posting_ordinal,
                remaining: freq,
                previous: None,
            });
        }
        consume_position_run(&mut reader, current_ordinal, freq)?;
        if current_ordinal > posting_ordinal || postings.next()?.is_none() {
            return Err(PositionCodecError::CursorInvariant {
                field: "position ordinal reachability",
            });
        }
    }
}
