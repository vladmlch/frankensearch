use crate::schema::SchemaDescriptor;

use super::codec::{
    ReferenceValidator, append_bytes, checked_add_size, clone_bytes, composite_key, copy_bytes,
    validate_composite_key, validate_metadata_basic, validate_positions_section, validate_schema,
    vint_len, write_vint,
};
use super::types::{
    EncodedBlock, TermDictionaryError, TermDictionaryLimits, TermInput, TermMetadata,
    TermSectionLengths,
};
use super::{BLOCK_COUNT_BYTES, TERM_BLOCK_TARGET_BYTES, TERM_RESTART_INTERVAL, TermDictionary};

/// Owned bytes produced by the canonical TERMDICT writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedTermDictionary {
    bytes: Vec<u8>,
    term_count: u32,
    block_count: u32,
    restart_count: usize,
}

impl EncodedTermDictionary {
    /// Encode strictly composite-key-sorted inputs with default resource caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid schema/reference metadata, duplicate or
    /// descending keys, overlong terms, resource exhaustion, or arithmetic
    /// overflow.
    pub fn encode_sorted(
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
        terms: &[TermInput<'_>],
    ) -> Result<Self, TermDictionaryError> {
        Self::encode_sorted_with_limits(schema, sections, terms, TermDictionaryLimits::default())
    }

    /// Encode strictly sorted inputs with explicit resource caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error when input, metadata, or the resulting canonical
    /// bytes violate the FSLX contract or supplied budgets.
    pub fn encode_sorted_with_limits(
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
        terms: &[TermInput<'_>],
        limits: TermDictionaryLimits,
    ) -> Result<Self, TermDictionaryError> {
        validate_schema(schema)?;
        validate_positions_section(schema, sections)?;
        if BLOCK_COUNT_BYTES > limits.max_bytes {
            return Err(TermDictionaryError::ByteBudgetExceeded {
                limit: limits.max_bytes,
                actual: BLOCK_COUNT_BYTES,
            });
        }
        if terms.len() > limits.max_terms {
            return Err(TermDictionaryError::TermBudgetExceeded {
                limit: limits.max_terms,
                actual: terms.len(),
            });
        }
        if !terms.is_empty() && limits.max_blocks == 0 {
            return Err(TermDictionaryError::BlockBudgetExceeded {
                limit: 0,
                actual: 1,
            });
        }
        if !terms.is_empty() && limits.max_restarts == 0 {
            return Err(TermDictionaryError::RestartBudgetExceeded {
                limit: 0,
                actual: 1,
            });
        }
        let term_count =
            u32::try_from(terms.len()).map_err(|_| TermDictionaryError::ValueOutOfRange {
                field: "term_count",
                value: u64::try_from(terms.len()).unwrap_or(u64::MAX),
                offset: 0,
            })?;

        let mut blocks = Vec::new();
        let mut references = ReferenceValidator::new(schema, sections)?;
        let mut previous_global = Vec::new();
        let mut previous_block = Vec::new();
        let mut current = new_block_buffer(limits.max_bytes)?;
        let mut current_first = None;
        let mut current_count = 0_usize;
        let mut completed_block_bytes = 0_usize;
        let mut restart_count = 0_usize;

        for (index, input) in terms.iter().enumerate() {
            let key = composite_key(input.field_ord, input.term, index)?;
            let has_positions = validate_composite_key(schema, &key, index)?;
            if !previous_global.is_empty() && key.as_slice() <= previous_global.as_slice() {
                return Err(TermDictionaryError::NonAscendingInput { index });
            }
            validate_metadata_basic(
                input.field_ord,
                has_positions,
                input.metadata,
                sections,
                index,
            )?;
            references.push(input.metadata)?;

            let mut encoded_entry = encode_entry(
                &key,
                &previous_block,
                current_count,
                input.metadata,
                has_positions,
            )?;
            if current_count != 0
                && current.len().checked_add(encoded_entry.len()).ok_or(
                    TermDictionaryError::SizeOverflow {
                        field: "block length",
                    },
                )? > TERM_BLOCK_TARGET_BYTES
            {
                finish_encoded_block(
                    &mut blocks,
                    &mut current,
                    &mut current_first,
                    current_count,
                    limits,
                    &mut completed_block_bytes,
                    &mut restart_count,
                )?;
                current = new_block_buffer(limits.max_bytes)?;
                current_count = 0;
                previous_block.clear();
                encoded_entry = encode_entry(
                    &key,
                    &previous_block,
                    current_count,
                    input.metadata,
                    has_positions,
                )?;
            }

            let minimum_output_len = checked_add_size(
                BLOCK_COUNT_BYTES,
                checked_add_size(
                    completed_block_bytes,
                    checked_add_size(current.len(), encoded_entry.len(), "pending block bytes")?,
                    "encoded blocks bytes",
                )?,
                "minimum TERMDICT length",
            )?;
            if minimum_output_len > limits.max_bytes {
                return Err(TermDictionaryError::ByteBudgetExceeded {
                    limit: limits.max_bytes,
                    actual: minimum_output_len,
                });
            }

            if current_first.is_none() {
                current_first = Some(clone_bytes(&key, "block first key")?);
            }
            append_bytes(&mut current, &encoded_entry, "encoded block bytes")?;
            current_count =
                current_count
                    .checked_add(1)
                    .ok_or(TermDictionaryError::SizeOverflow {
                        field: "block entry count",
                    })?;
            copy_bytes(&mut previous_block, &key, "previous block key")?;
            copy_bytes(&mut previous_global, &key, "previous global key")?;
        }
        if current_count != 0 {
            finish_encoded_block(
                &mut blocks,
                &mut current,
                &mut current_first,
                current_count,
                limits,
                &mut completed_block_bytes,
                &mut restart_count,
            )?;
        }
        references.finish()?;

        let block_count =
            u32::try_from(blocks.len()).map_err(|_| TermDictionaryError::ValueOutOfRange {
                field: "block_count",
                value: u64::try_from(blocks.len()).unwrap_or(u64::MAX),
                offset: 0,
            })?;
        if blocks.len() > limits.max_blocks {
            return Err(TermDictionaryError::BlockBudgetExceeded {
                limit: limits.max_blocks,
                actual: blocks.len(),
            });
        }
        debug_assert_eq!(
            restart_count,
            blocks
                .iter()
                .map(|block| block.entry_count.div_ceil(TERM_RESTART_INTERVAL))
                .sum::<usize>()
        );
        debug_assert_eq!(
            completed_block_bytes,
            blocks.iter().map(|block| block.bytes.len()).sum::<usize>()
        );

        let mut relative_offset = 0_u64;
        let mut output_len = BLOCK_COUNT_BYTES;
        for block in &blocks {
            output_len = checked_add_size(
                output_len,
                vint_len(u64::try_from(block.first_key.len()).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "index first-key length",
                    }
                })?),
                "block index key length vint",
            )?;
            output_len =
                checked_add_size(output_len, block.first_key.len(), "block index key bytes")?;
            output_len = checked_add_size(
                output_len,
                vint_len(relative_offset),
                "block index offset vint",
            )?;
            relative_offset = relative_offset
                .checked_add(u64::try_from(block.bytes.len()).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "relative block offset",
                    }
                })?)
                .ok_or(TermDictionaryError::SizeOverflow {
                    field: "relative block offset",
                })?;
        }
        output_len = checked_add_size(
            output_len,
            usize::try_from(relative_offset).map_err(|_| TermDictionaryError::SizeOverflow {
                field: "blocks area length",
            })?,
            "TERMDICT length",
        )?;
        if output_len > limits.max_bytes {
            return Err(TermDictionaryError::ByteBudgetExceeded {
                limit: limits.max_bytes,
                actual: output_len,
            });
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(output_len)
            .map_err(|_| TermDictionaryError::Allocation {
                context: "encoded TERMDICT",
                count: output_len,
            })?;
        bytes.extend_from_slice(&block_count.to_le_bytes());
        relative_offset = 0;
        for block in &blocks {
            write_vint(
                u64::try_from(block.first_key.len()).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "index key length",
                    }
                })?,
                &mut bytes,
            );
            bytes.extend_from_slice(&block.first_key);
            write_vint(relative_offset, &mut bytes);
            relative_offset = relative_offset
                .checked_add(u64::try_from(block.bytes.len()).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "relative block offset",
                    }
                })?)
                .ok_or(TermDictionaryError::SizeOverflow {
                    field: "relative block offset",
                })?;
        }
        for block in &blocks {
            bytes.extend_from_slice(&block.bytes);
        }
        debug_assert_eq!(bytes.len(), output_len);

        let owned_limits = TermDictionaryLimits {
            max_bytes: bytes.len(),
            max_blocks: blocks.len(),
            max_terms: terms.len(),
            max_restarts: restart_count,
        };
        let parsed = TermDictionary::parse_with_limits(&bytes, schema, sections, owned_limits)?;
        debug_assert_eq!(parsed.restart_count(), restart_count);
        debug_assert_eq!(parsed.term_count(), term_count);
        Ok(Self {
            bytes,
            term_count,
            block_count,
            restart_count,
        })
    }

    /// Exact canonical durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its canonical bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of encoded terms.
    #[must_use]
    pub const fn term_count(&self) -> u32 {
        self.term_count
    }

    /// Number of encoded blocks.
    #[must_use]
    pub const fn block_count(&self) -> u32 {
        self.block_count
    }

    /// Re-open the owned bytes through the same validating reader.
    ///
    /// # Errors
    ///
    /// Returns a typed error if an internal encoder invariant was violated.
    pub fn dictionary(
        &self,
        schema: SchemaDescriptor,
        sections: TermSectionLengths,
    ) -> Result<TermDictionary<'_>, TermDictionaryError> {
        TermDictionary::parse_with_limits(
            &self.bytes,
            schema,
            sections,
            TermDictionaryLimits {
                max_bytes: self.bytes.len(),
                max_blocks: usize::try_from(self.block_count).unwrap_or(usize::MAX),
                max_terms: usize::try_from(self.term_count).unwrap_or(usize::MAX),
                max_restarts: self.restart_count,
            },
        )
    }
}

fn new_block_buffer(max_bytes: usize) -> Result<Vec<u8>, TermDictionaryError> {
    let mut bytes = Vec::new();
    let reserve = TERM_BLOCK_TARGET_BYTES.min(max_bytes.saturating_sub(BLOCK_COUNT_BYTES));
    bytes
        .try_reserve(reserve)
        .map_err(|_| TermDictionaryError::Allocation {
            context: "TERMDICT block",
            count: reserve,
        })?;
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    Ok(bytes)
}

fn finish_encoded_block(
    blocks: &mut Vec<EncodedBlock>,
    current: &mut Vec<u8>,
    current_first: &mut Option<Vec<u8>>,
    entry_count: usize,
    limits: TermDictionaryLimits,
    completed_block_bytes: &mut usize,
    restart_count: &mut usize,
) -> Result<(), TermDictionaryError> {
    if blocks.len() >= limits.max_blocks {
        return Err(TermDictionaryError::BlockBudgetExceeded {
            limit: limits.max_blocks,
            actual: blocks.len().saturating_add(1),
        });
    }
    let next_restart_count = restart_count
        .checked_add(entry_count.div_ceil(TERM_RESTART_INTERVAL))
        .ok_or(TermDictionaryError::SizeOverflow {
            field: "restart count",
        })?;
    if next_restart_count > limits.max_restarts {
        return Err(TermDictionaryError::RestartBudgetExceeded {
            limit: limits.max_restarts,
            actual: next_restart_count,
        });
    }
    let next_completed_bytes = completed_block_bytes.checked_add(current.len()).ok_or(
        TermDictionaryError::SizeOverflow {
            field: "encoded blocks bytes",
        },
    )?;
    let minimum_output_len = checked_add_size(
        BLOCK_COUNT_BYTES,
        next_completed_bytes,
        "minimum TERMDICT length",
    )?;
    if minimum_output_len > limits.max_bytes {
        return Err(TermDictionaryError::ByteBudgetExceeded {
            limit: limits.max_bytes,
            actual: minimum_output_len,
        });
    }
    let count = u16::try_from(entry_count).map_err(|_| TermDictionaryError::ValueOutOfRange {
        field: "block entry_count",
        value: u64::try_from(entry_count).unwrap_or(u64::MAX),
        offset: 0,
    })?;
    current[..2].copy_from_slice(&count.to_le_bytes());
    blocks
        .try_reserve(1)
        .map_err(|_| TermDictionaryError::Allocation {
            context: "encoded block list",
            count: blocks.len().saturating_add(1),
        })?;
    blocks.push(EncodedBlock {
        first_key: current_first
            .take()
            .ok_or(TermDictionaryError::EmptyBlock {
                block_index: blocks.len(),
            })?,
        bytes: std::mem::take(current),
        entry_count,
    });
    *completed_block_bytes = next_completed_bytes;
    *restart_count = next_restart_count;
    Ok(())
}

fn encode_entry(
    key: &[u8],
    previous: &[u8],
    entry_ordinal: usize,
    metadata: TermMetadata,
    has_positions: bool,
) -> Result<Vec<u8>, TermDictionaryError> {
    let length = encoded_entry_len(key, previous, entry_ordinal, metadata, has_positions)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| TermDictionaryError::Allocation {
            context: "encoded term entry",
            count: length,
        })?;
    if entry_ordinal.is_multiple_of(TERM_RESTART_INTERVAL) {
        write_vint(
            u64::try_from(key.len()).map_err(|_| TermDictionaryError::SizeOverflow {
                field: "restart key length",
            })?,
            &mut output,
        );
        output.extend_from_slice(key);
    } else {
        let shared = longest_shared_prefix(previous, key);
        let suffix = &key[shared..];
        write_vint(
            u64::try_from(shared).map_err(|_| TermDictionaryError::SizeOverflow {
                field: "shared prefix length",
            })?,
            &mut output,
        );
        write_vint(
            u64::try_from(suffix.len()).map_err(|_| TermDictionaryError::SizeOverflow {
                field: "suffix length",
            })?,
            &mut output,
        );
        output.extend_from_slice(suffix);
    }
    write_vint(u64::from(metadata.doc_freq), &mut output);
    write_vint(metadata.postings.offset, &mut output);
    write_vint(metadata.postings.len, &mut output);
    if has_positions {
        let positions =
            metadata
                .positions
                .ok_or_else(|| TermDictionaryError::PositionsPresenceMismatch {
                    field_ord: u16::from_be_bytes([key[0], key[1]]),
                    expected: true,
                    actual: false,
                })?;
        write_vint(positions.offset, &mut output);
        write_vint(positions.len, &mut output);
    }
    write_vint(metadata.blockmax.offset, &mut output);
    write_vint(metadata.blockmax.len, &mut output);
    debug_assert_eq!(output.len(), length);
    Ok(output)
}

pub(in crate::grimoire) fn encoded_entry_len(
    key: &[u8],
    previous: &[u8],
    entry_ordinal: usize,
    metadata: TermMetadata,
    has_positions: bool,
) -> Result<usize, TermDictionaryError> {
    let key_len = if entry_ordinal.is_multiple_of(TERM_RESTART_INTERVAL) {
        checked_add_size(
            vint_len(
                u64::try_from(key.len()).map_err(|_| TermDictionaryError::SizeOverflow {
                    field: "restart key length",
                })?,
            ),
            key.len(),
            "full key encoding",
        )?
    } else {
        let shared = longest_shared_prefix(previous, key);
        let suffix_len = key.len().saturating_sub(shared);
        let prefix_len =
            vint_len(
                u64::try_from(shared).map_err(|_| TermDictionaryError::SizeOverflow {
                    field: "shared prefix length",
                })?,
            );
        checked_add_size(
            checked_add_size(
                prefix_len,
                vint_len(u64::try_from(suffix_len).map_err(|_| {
                    TermDictionaryError::SizeOverflow {
                        field: "suffix length",
                    }
                })?),
                "compressed key lengths",
            )?,
            suffix_len,
            "compressed key bytes",
        )?
    };
    let mut length = key_len;
    for value in [
        u64::from(metadata.doc_freq),
        metadata.postings.offset,
        metadata.postings.len,
    ] {
        length = checked_add_size(length, vint_len(value), "term payload")?;
    }
    if has_positions {
        let positions =
            metadata
                .positions
                .ok_or_else(|| TermDictionaryError::PositionsPresenceMismatch {
                    field_ord: u16::from_be_bytes([key[0], key[1]]),
                    expected: true,
                    actual: false,
                })?;
        length = checked_add_size(length, vint_len(positions.offset), "positions offset")?;
        length = checked_add_size(length, vint_len(positions.len), "positions length")?;
    }
    length = checked_add_size(
        length,
        vint_len(metadata.blockmax.offset),
        "blockmax offset",
    )?;
    checked_add_size(length, vint_len(metadata.blockmax.len), "blockmax length")
}

fn longest_shared_prefix(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| left.len().min(right.len()))
}
