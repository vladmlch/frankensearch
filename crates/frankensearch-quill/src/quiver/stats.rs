use thiserror::Error;

use super::doclen::STATS_ENTRY_LEN;

/// One at-seal FSLX STATS row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldStats {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// BM25 token numerator captured at seal time.
    ///
    /// Fresh seals store the exact raw token count. A compaction replacement
    /// may instead sum decoded fieldnorms for the retained documents.
    pub total_tokens: u64,
    /// Segment-wide at-seal document count (Tantivy `max_doc` denominator).
    pub doc_count: u32,
}

impl FieldStats {
    /// Construct one schema-ordered at-seal statistics row.
    #[must_use]
    pub const fn new(field_ord: u16, total_tokens: u64, doc_count: u32) -> Self {
        Self {
            field_ord,
            total_tokens,
            doc_count,
        }
    }

    /// Raw average field length used by Tantivy BM25.
    ///
    /// This is deliberately not an average of decoded fieldnorm buckets.
    #[must_use]
    pub fn average_field_length(self) -> Option<f32> {
        (self.doc_count != 0).then(|| self.total_tokens as f32 / self.doc_count as f32)
    }
}

/// Explicit resource ceilings for an FSLX STATS section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StatsLimits {
    /// Maximum schema-derived Text/Keyword field count.
    pub max_fields: usize,
    /// Maximum complete packed section size.
    pub max_section_bytes: u64,
}

impl Default for StatsLimits {
    fn default() -> Self {
        Self {
            max_fields: 65_536,
            max_section_bytes: u64::from(u32::MAX),
        }
    }
}

/// Typed failures from STATS encoding, validation, or snapshot aggregation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StatsCodecError {
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("STATS {resource} {actual} exceeds limit {limit}")]
    ResourceLimit {
        /// Bounded resource name.
        resource: &'static str,
        /// Rejected amount.
        actual: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// Schema field ordinals must be strictly ascending.
    #[error(
        "STATS field ordinals are not strictly ascending at index {index}: previous {previous}, current {current}"
    )]
    NonAscendingFields {
        /// Rejected field index.
        index: usize,
        /// Previous ordinal.
        previous: u16,
        /// Current ordinal.
        current: u16,
    },
    /// A row disagreed with the schema-derived field set.
    #[error("STATS field {index} mismatch: expected {expected:?}, got {actual:?}")]
    UnexpectedField {
        /// Field position in schema order.
        index: usize,
        /// Expected ordinal, if any.
        expected: Option<u16>,
        /// Supplied or decoded ordinal, if any.
        actual: Option<u16>,
    },
    /// Every indexed field uses the segment-wide at-seal denominator.
    #[error(
        "STATS field {field_ord} has doc_count {actual}, expected segment doc_count {expected}"
    )]
    DocCountMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Required segment at-seal document count.
        expected: u32,
        /// Supplied count.
        actual: u32,
    },
    /// A zero-document segment cannot contain indexed tokens.
    #[error("STATS field {field_ord} has {total_tokens} tokens with zero documents")]
    TokensWithoutDocuments {
        /// Schema field ordinal.
        field_ord: u16,
        /// Impossible raw token count.
        total_tokens: u64,
    },
    /// Packed STATS rows have one exact schema-derived length.
    #[error("STATS byte length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Required packed length.
        expected: usize,
        /// Supplied bytes.
        actual: usize,
    },
    /// Checked layout arithmetic overflowed.
    #[error("STATS arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout or aggregate component.
        field: &'static str,
    },
    /// Snapshot segments did not carry the same schema-derived field count.
    #[error("STATS aggregate segment {segment_index} has {actual} fields, expected {expected}")]
    AggregateFieldCountMismatch {
        /// Zero-based segment ordinal in the aggregation input.
        segment_index: usize,
        /// Required row count.
        expected: usize,
        /// Supplied row count.
        actual: usize,
    },
    /// Snapshot segments did not carry the same field at one schema position.
    #[error(
        "STATS aggregate segment {segment_index} field {field_index} is {actual}, expected {expected}"
    )]
    AggregateFieldMismatch {
        /// Zero-based segment ordinal.
        segment_index: usize,
        /// Schema field position.
        field_index: usize,
        /// Required ordinal.
        expected: u16,
        /// Supplied ordinal.
        actual: u16,
    },
    /// A checked raw-token or document-count sum exceeded u64.
    #[error("STATS aggregate overflow for field {field_ord} {counter}")]
    AggregateOverflow {
        /// Schema field ordinal.
        field_ord: u16,
        /// Counter being summed.
        counter: &'static str,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for STATS {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

/// Owned canonical bytes produced by the fresh-seal STATS writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedStatsSection {
    bytes: Vec<u8>,
    segment_doc_count: u32,
    field_count: usize,
}

impl EncodedStatsSection {
    /// Encode the schema-derived set of at-seal BM25 counters.
    ///
    /// # Errors
    ///
    /// Rejects missing/extra fields, a per-field denominator that differs from
    /// `segment_doc_count`, impossible zero-document token counts, resource
    /// abuse, arithmetic overflow, and allocation failure.
    pub fn encode(
        expected_field_ords: &[u16],
        rows: &[FieldStats],
        segment_doc_count: u32,
    ) -> Result<Self, StatsCodecError> {
        Self::encode_with_limits(
            expected_field_ords,
            rows,
            segment_doc_count,
            StatsLimits::default(),
        )
    }

    /// Encode with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`].
    pub fn encode_with_limits(
        expected_field_ords: &[u16],
        rows: &[FieldStats],
        segment_doc_count: u32,
        limits: StatsLimits,
    ) -> Result<Self, StatsCodecError> {
        validate_stats_rows(expected_field_ords, rows, segment_doc_count, limits)?;
        let section_len = stats_section_len(expected_field_ords.len(), limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(section_len)
            .map_err(|_| StatsCodecError::Allocation {
                resource: "section bytes",
                bytes: section_len,
            })?;
        for row in rows {
            bytes.extend_from_slice(&row.field_ord.to_le_bytes());
            bytes.extend_from_slice(&row.total_tokens.to_le_bytes());
            bytes.extend_from_slice(&row.doc_count.to_le_bytes());
        }
        Ok(Self {
            bytes,
            segment_doc_count,
            field_count: rows.len(),
        })
    }

    /// Borrow the exact canonical durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of schema-indexed text fields in this section.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }

    /// Re-open the owned bytes through the validating reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated.
    pub fn section(&self, expected_field_ords: &[u16]) -> Result<StatsSection, StatsCodecError> {
        StatsSection::parse(&self.bytes, expected_field_ords, self.segment_doc_count)
    }
}

/// Owned, fully validated FSLX STATS rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StatsSection {
    rows: Vec<FieldStats>,
    segment_doc_count: u32,
}

impl StatsSection {
    /// Validate a packed STATS payload against its schema and segment header.
    ///
    /// # Errors
    ///
    /// Rejects wrong byte length, missing/extra/reordered fields, denominator
    /// drift, impossible zero-document tokens, resource abuse, and allocation
    /// failure.
    pub fn parse(
        bytes: &[u8],
        expected_field_ords: &[u16],
        segment_doc_count: u32,
    ) -> Result<Self, StatsCodecError> {
        Self::parse_with_limits(
            bytes,
            expected_field_ords,
            segment_doc_count,
            StatsLimits::default(),
        )
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`].
    pub fn parse_with_limits(
        bytes: &[u8],
        expected_field_ords: &[u16],
        segment_doc_count: u32,
        limits: StatsLimits,
    ) -> Result<Self, StatsCodecError> {
        validate_stats_expected_fields(expected_field_ords, limits.max_fields)?;
        let expected_len = stats_section_len(expected_field_ords.len(), limits)?;
        if bytes.len() != expected_len {
            return Err(StatsCodecError::LengthMismatch {
                expected: expected_len,
                actual: bytes.len(),
            });
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(expected_field_ords.len())
            .map_err(|_| StatsCodecError::Allocation {
                resource: "decoded rows",
                bytes: expected_field_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<FieldStats>()),
            })?;
        for (index, &expected_field) in expected_field_ords.iter().enumerate() {
            let offset = index * STATS_ENTRY_LEN;
            let field_ord = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if field_ord != expected_field {
                return Err(StatsCodecError::UnexpectedField {
                    index,
                    expected: Some(expected_field),
                    actual: Some(field_ord),
                });
            }
            let total_tokens = u64::from_le_bytes([
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
                bytes[offset + 8],
                bytes[offset + 9],
            ]);
            let doc_count = u32::from_le_bytes([
                bytes[offset + 10],
                bytes[offset + 11],
                bytes[offset + 12],
                bytes[offset + 13],
            ]);
            rows.push(FieldStats::new(field_ord, total_tokens, doc_count));
        }
        validate_stats_rows(expected_field_ords, &rows, segment_doc_count, limits)?;
        Ok(Self {
            rows,
            segment_doc_count,
        })
    }

    /// Schema-ordered at-seal counters.
    #[must_use]
    pub fn rows(&self) -> &[FieldStats] {
        &self.rows
    }

    /// Segment-wide at-seal document count used by every row.
    #[must_use]
    pub const fn segment_doc_count(&self) -> u32 {
        self.segment_doc_count
    }

    /// Resolve one field's at-seal counters.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<FieldStats> {
        self.rows
            .binary_search_by_key(&field_ord, |row| row.field_ord)
            .ok()
            .and_then(|index| self.rows.get(index))
            .copied()
    }
}

/// Snapshot-level checked sum for one schema field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotFieldStats {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Checked snapshot token numerator under the BM25 lifecycle contract.
    ///
    /// Keeper rows remain at-seal until compaction; Delta-local tombstones are
    /// excluded immediately.
    pub total_tokens: u64,
    /// Keeper at-seal rows plus live Delta rows used as BM25 `N`.
    pub doc_count: u64,
}

impl SnapshotFieldStats {
    /// Raw Tantivy-compatible average field length.
    ///
    /// Returns `None` for an empty snapshot rather than producing NaN.
    #[must_use]
    pub fn average_field_length(self) -> Option<f32> {
        (self.doc_count != 0).then(|| self.total_tokens as f32 / self.doc_count as f32)
    }
}

/// Checked-sum identical validated STATS sections across live segments.
///
/// Tombstones do not alter these at-seal counters. Compaction removes the old
/// segment row and supplies a replacement whose token numerator may be
/// re-derived from decoded fieldnorms for retained documents.
///
/// # Errors
///
/// Returns a typed error for field-set drift, counter overflow, or allocation
/// failure.
pub fn aggregate_field_stats<'a>(
    segments: impl IntoIterator<Item = &'a StatsSection>,
) -> Result<Vec<SnapshotFieldStats>, StatsCodecError> {
    let mut segments = segments.into_iter();
    let Some(first) = segments.next() else {
        return Ok(Vec::new());
    };
    let first_rows = first.rows();
    let mut aggregate = Vec::new();
    aggregate
        .try_reserve_exact(first_rows.len())
        .map_err(|_| StatsCodecError::Allocation {
            resource: "aggregate rows",
            bytes: first_rows
                .len()
                .saturating_mul(std::mem::size_of::<SnapshotFieldStats>()),
        })?;
    for row in first_rows {
        aggregate.push(SnapshotFieldStats {
            field_ord: row.field_ord,
            total_tokens: row.total_tokens,
            doc_count: u64::from(row.doc_count),
        });
    }
    for (relative_segment_index, segment) in segments.enumerate() {
        let segment_index = relative_segment_index + 1;
        let rows = segment.rows();
        if rows.len() != aggregate.len() {
            return Err(StatsCodecError::AggregateFieldCountMismatch {
                segment_index,
                expected: aggregate.len(),
                actual: rows.len(),
            });
        }
        for (field_index, (total, row)) in aggregate.iter_mut().zip(rows).enumerate() {
            if row.field_ord != total.field_ord {
                return Err(StatsCodecError::AggregateFieldMismatch {
                    segment_index,
                    field_index,
                    expected: total.field_ord,
                    actual: row.field_ord,
                });
            }
            total.total_tokens = total.total_tokens.checked_add(row.total_tokens).ok_or(
                StatsCodecError::AggregateOverflow {
                    field_ord: row.field_ord,
                    counter: "total_tokens",
                },
            )?;
            total.doc_count = total
                .doc_count
                .checked_add(u64::from(row.doc_count))
                .ok_or(StatsCodecError::AggregateOverflow {
                    field_ord: row.field_ord,
                    counter: "doc_count",
                })?;
        }
    }
    Ok(aggregate)
}

fn validate_stats_expected_fields(
    expected_field_ords: &[u16],
    max_fields: usize,
) -> Result<(), StatsCodecError> {
    if expected_field_ords.len() > max_fields {
        return Err(StatsCodecError::ResourceLimit {
            resource: "field count",
            actual: u64::try_from(expected_field_ords.len()).unwrap_or(u64::MAX),
            limit: u64::try_from(max_fields).unwrap_or(u64::MAX),
        });
    }
    for (index, pair) in expected_field_ords.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(StatsCodecError::NonAscendingFields {
                index: index + 1,
                previous: pair[0],
                current: pair[1],
            });
        }
    }
    Ok(())
}

fn validate_stats_rows(
    expected_field_ords: &[u16],
    rows: &[FieldStats],
    segment_doc_count: u32,
    limits: StatsLimits,
) -> Result<(), StatsCodecError> {
    validate_stats_expected_fields(expected_field_ords, limits.max_fields)?;
    let compared = expected_field_ords.len().max(rows.len());
    for index in 0..compared {
        let expected = expected_field_ords.get(index).copied();
        let actual = rows.get(index).map(|row| row.field_ord);
        if expected != actual {
            return Err(StatsCodecError::UnexpectedField {
                index,
                expected,
                actual,
            });
        }
    }
    for row in rows {
        if row.doc_count != segment_doc_count {
            return Err(StatsCodecError::DocCountMismatch {
                field_ord: row.field_ord,
                expected: segment_doc_count,
                actual: row.doc_count,
            });
        }
        if segment_doc_count == 0 && row.total_tokens != 0 {
            return Err(StatsCodecError::TokensWithoutDocuments {
                field_ord: row.field_ord,
                total_tokens: row.total_tokens,
            });
        }
    }
    Ok(())
}

fn stats_section_len(field_count: usize, limits: StatsLimits) -> Result<usize, StatsCodecError> {
    if field_count > limits.max_fields {
        return Err(StatsCodecError::ResourceLimit {
            resource: "field count",
            actual: u64::try_from(field_count).unwrap_or(u64::MAX),
            limit: u64::try_from(limits.max_fields).unwrap_or(u64::MAX),
        });
    }
    let section_len =
        field_count
            .checked_mul(STATS_ENTRY_LEN)
            .ok_or(StatsCodecError::ArithmeticOverflow {
                field: "section length",
            })?;
    let section_len_u64 = u64::try_from(section_len).unwrap_or(u64::MAX);
    if section_len_u64 > limits.max_section_bytes {
        return Err(StatsCodecError::ResourceLimit {
            resource: "section bytes",
            actual: section_len_u64,
            limit: limits.max_section_bytes,
        });
    }
    Ok(section_len)
}
