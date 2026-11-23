use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::{Bound, Range};

use thiserror::Error;

use crate::schema::{FieldKind, SchemaDescriptor};
use crate::scribe::{ColumnarAccumulator, TokenAnalyzer};

use super::{read_u16, read_u32, read_u64};

/// Packed size of one `{ field_ord: u16, count: u32 }` NUMERIC field header.
pub const NUMERIC_FIELD_HEADER_LEN: usize = 6;
/// Packed size of one `{ value_bits: u64, docid: u32 }` NUMERIC pair.
pub const NUMERIC_PAIR_LEN: usize = 12;

/// One schema-typed numeric scalar.
///
/// The enum tag exists only in memory. FSLX stores the exact eight value bits;
/// the field descriptor supplies the durable signedness tag.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NumericValue {
    /// Signed 64-bit scalar.
    I64(i64),
    /// Unsigned 64-bit scalar.
    U64(u64),
}

impl NumericValue {
    /// Exact little-endian bits persisted in one NUMERIC pair.
    #[must_use]
    pub const fn value_bits(self) -> u64 {
        match self {
            Self::I64(value) => u64::from_le_bytes(value.to_le_bytes()),
            Self::U64(value) => value,
        }
    }

    /// Exact little-endian bytes persisted in one NUMERIC pair or stored field.
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.value_bits().to_le_bytes()
    }
}

/// One typed NUMERIC pair supplied to the deterministic writer or returned by
/// the validated reader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericEntry {
    value: NumericValue,
    docid: u32,
}

impl NumericEntry {
    /// Construct one signed pair.
    #[must_use]
    pub const fn i64(value: i64, docid: u32) -> Self {
        Self {
            value: NumericValue::I64(value),
            docid,
        }
    }

    /// Construct one unsigned pair.
    #[must_use]
    pub const fn u64(value: u64, docid: u32) -> Self {
        Self {
            value: NumericValue::U64(value),
            docid,
        }
    }

    /// Schema-typed scalar.
    #[must_use]
    pub const fn value(self) -> NumericValue {
        self.value
    }

    /// Absolute global document ID.
    #[must_use]
    pub const fn docid(self) -> u32 {
        self.docid
    }
}

/// One schema-ordered numeric field supplied to the writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericFieldInput<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Present values. Input order is irrelevant; the writer emits canonical
    /// typed `(value, docid)` order.
    pub entries: &'a [NumericEntry],
}

impl<'a> NumericFieldInput<'a> {
    /// Construct one schema-ordered input field.
    #[must_use]
    pub const fn new(field_ord: u16, entries: &'a [NumericEntry]) -> Self {
        Self { field_ord, entries }
    }
}

/// Explicit resource ceilings for an FSLX NUMERIC section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericLimits {
    /// Maximum schema-derived indexed-or-fast numeric field count.
    pub max_fields: usize,
    /// Maximum entries in one numeric field.
    pub max_entries_per_field: u64,
    /// Maximum entries summed across all numeric fields.
    pub max_total_entries: u64,
    /// Maximum half-open global docid span.
    pub max_docid_span: u64,
    /// Maximum complete section size.
    pub max_section_bytes: u64,
}

impl Default for NumericLimits {
    fn default() -> Self {
        Self {
            max_fields: usize::from(u16::MAX) + 1,
            max_entries_per_field: u64::from(u32::MAX),
            max_total_entries: u64::from(u32::MAX),
            max_docid_span: 1_u64 << 32,
            max_section_bytes: u64::from(u32::MAX),
        }
    }
}

/// Typed failures from NUMERIC encoding, validation, or range materialization.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NumericCodecError {
    /// The supplied compile-time descriptor is not internally valid.
    #[error("invalid NUMERIC schema: {detail}")]
    InvalidSchema {
        /// Descriptor validation detail.
        detail: String,
    },
    /// A segment's half-open docid range was reversed or exceeded u32 payloads.
    #[error("invalid NUMERIC docid range [{docid_lo}, {docid_hi})")]
    InvalidDocIdRange {
        /// Inclusive lower bound.
        docid_lo: u64,
        /// Exclusive upper bound.
        docid_hi: u64,
    },
    /// Merge requires at least one validated source section.
    #[error("cannot merge an empty NUMERIC section list")]
    EmptyMerge,
    /// Merge inputs must be ordered and non-overlapping by segment range.
    #[error("NUMERIC merge section {index} begins at {current_lo}, before prior end {previous_hi}")]
    MergeRangeOrder {
        /// Rejected source index.
        index: usize,
        /// Prior source's exclusive high bound.
        previous_hi: u64,
        /// Current source's inclusive low bound.
        current_lo: u64,
    },
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("NUMERIC {resource} {actual} exceeds limit {limit}")]
    ResourceLimit {
        /// Bounded resource name.
        resource: &'static str,
        /// Rejected amount.
        actual: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// Input fields must follow the schema-derived indexed-numeric order.
    #[error("NUMERIC field {index} mismatch: expected {expected:?}, got {actual:?}")]
    UnexpectedField {
        /// Field position in schema order.
        index: usize,
        /// Expected ordinal, if any.
        expected: Option<u16>,
        /// Supplied or decoded ordinal, if present.
        actual: Option<u16>,
    },
    /// One in-memory value disagreed with its field descriptor's signedness.
    #[error("NUMERIC field {field_ord} expects {expected}, got {actual}")]
    ValueTypeMismatch {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Schema type name.
        expected: &'static str,
        /// Supplied type name.
        actual: &'static str,
    },
    /// A pair names a document outside its owning segment range.
    #[error("NUMERIC field {field_ord} docid {docid} is outside [{docid_lo}, {docid_hi})")]
    DocIdOutOfRange {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Rejected absolute document ID.
        docid: u32,
        /// Inclusive segment lower bound.
        docid_lo: u64,
        /// Exclusive segment upper bound.
        docid_hi: u64,
    },
    /// Durable pairs must be strictly ordered by typed value and then docid.
    #[error("NUMERIC field {field_ord} pairs are not in canonical order at index {index}")]
    NonCanonicalPairOrder {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Rejected pair index.
        index: usize,
    },
    /// A range bound disagreed with its field descriptor's signedness.
    #[error("NUMERIC range for field {field_ord} expects {expected}, got {actual}")]
    BoundTypeMismatch {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Schema type name.
        expected: &'static str,
        /// Supplied type name.
        actual: &'static str,
    },
    /// Sparse accumulator documents must map into the exact segment range.
    #[error(
        "NUMERIC source document {index} maps to docid {docid}, outside [{docid_lo}, {docid_hi})"
    )]
    SourceDocumentOutOfRange {
        /// Completed-document index in the accumulator.
        index: usize,
        /// Rebased global document ID.
        docid: u64,
        /// Inclusive segment lower bound.
        docid_lo: u64,
        /// Exclusive segment upper bound.
        docid_hi: u64,
    },
    /// Rebased accumulator documents must remain strictly ascending.
    #[error(
        "NUMERIC source documents are not strictly ascending at index {index}: previous {previous}, current {current}"
    )]
    NonAscendingSourceDocuments {
        /// Rejected completed-document index.
        index: usize,
        /// Previous rebased global document ID.
        previous: u64,
        /// Current rebased global document ID.
        current: u64,
    },
    /// A private Scribe column invariant was violated before sealing.
    #[error("invalid NUMERIC source column {field_ord}: expected {expected} values, got {actual}")]
    InvalidSourceColumn {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Required completed-document count.
        expected: usize,
        /// Actual numeric-column length.
        actual: usize,
    },
    /// Checked layout arithmetic overflowed.
    #[error("NUMERIC arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout component being computed.
        field: &'static str,
    },
    /// Durable bytes ended before the canonical layout did.
    #[error("truncated NUMERIC section: expected at least {expected} bytes, got {actual}")]
    Truncated {
        /// Minimum required length.
        expected: usize,
        /// Available length.
        actual: usize,
    },
    /// Canonical sections end exactly after the final pair.
    #[error("NUMERIC section has trailing bytes: expected {expected}, got {actual}")]
    TrailingBytes {
        /// Canonical complete size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for NUMERIC {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NumericKind {
    I64,
    U64,
}

impl NumericKind {
    const fn name(self) -> &'static str {
        match self {
            Self::I64 => "i64",
            Self::U64 => "u64",
        }
    }

    const fn value_from_bits(self, bits: u64) -> NumericValue {
        match self {
            Self::I64 => NumericValue::I64(i64::from_le_bytes(bits.to_le_bytes())),
            Self::U64 => NumericValue::U64(bits),
        }
    }

    const fn accepts(self, value: NumericValue) -> bool {
        matches!(
            (self, value),
            (Self::I64, NumericValue::I64(_)) | (Self::U64, NumericValue::U64(_))
        )
    }
}

/// Owned canonical bytes produced by the fresh-seal NUMERIC writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedNumericSection {
    bytes: Vec<u8>,
    schema: SchemaDescriptor,
    docid_lo: u64,
    docid_hi: u64,
    entry_count: usize,
}

impl EncodedNumericSection {
    /// Encode every indexed-or-fast numeric field in canonical schema and typed-value
    /// order.
    ///
    /// # Errors
    ///
    /// Rejects schema/field/type drift, out-of-range docids, duplicate pairs,
    /// resource abuse, arithmetic overflow, and allocation failure.
    pub fn encode(
        schema: SchemaDescriptor,
        docid_lo: u64,
        docid_hi: u64,
        fields: &[NumericFieldInput<'_>],
    ) -> Result<Self, NumericCodecError> {
        Self::encode_with_limits(schema, docid_lo, docid_hi, fields, NumericLimits::default())
    }

    /// Encode with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`].
    pub fn encode_with_limits(
        schema: SchemaDescriptor,
        docid_lo: u64,
        docid_hi: u64,
        fields: &[NumericFieldInput<'_>],
        limits: NumericLimits,
    ) -> Result<Self, NumericCodecError> {
        let expected = numeric_schema_fields(schema, limits.max_fields)?;
        validate_numeric_input_fields(&expected, fields)?;
        checked_numeric_span(docid_lo, docid_hi, limits)?;

        let mut total_entries = 0_u64;
        let mut total_len = 0_usize;
        for field in fields {
            let count = u64::try_from(field.entries.len()).unwrap_or(u64::MAX);
            if count > limits.max_entries_per_field {
                return Err(NumericCodecError::ResourceLimit {
                    resource: "entries per field",
                    actual: count,
                    limit: limits.max_entries_per_field,
                });
            }
            if count > u64::from(u32::MAX) {
                return Err(NumericCodecError::ResourceLimit {
                    resource: "durable field entry count",
                    actual: count,
                    limit: u64::from(u32::MAX),
                });
            }
            total_entries =
                total_entries
                    .checked_add(count)
                    .ok_or(NumericCodecError::ArithmeticOverflow {
                        field: "total entry count",
                    })?;
            if total_entries > limits.max_total_entries {
                return Err(NumericCodecError::ResourceLimit {
                    resource: "total entries",
                    actual: total_entries,
                    limit: limits.max_total_entries,
                });
            }
            let field_len = field
                .entries
                .len()
                .checked_mul(NUMERIC_PAIR_LEN)
                .and_then(|pairs| pairs.checked_add(NUMERIC_FIELD_HEADER_LEN))
                .ok_or(NumericCodecError::ArithmeticOverflow {
                    field: "field byte length",
                })?;
            total_len =
                total_len
                    .checked_add(field_len)
                    .ok_or(NumericCodecError::ArithmeticOverflow {
                        field: "section byte length",
                    })?;
        }
        let total_len_u64 = u64::try_from(total_len).unwrap_or(u64::MAX);
        if total_len_u64 > limits.max_section_bytes {
            return Err(NumericCodecError::ResourceLimit {
                resource: "section bytes",
                actual: total_len_u64,
                limit: limits.max_section_bytes,
            });
        }

        for ((_, kind), field) in expected.iter().zip(fields) {
            for &entry in field.entries {
                validate_numeric_entry_type(field.field_ord, *kind, entry.value)?;
                validate_numeric_docid(field.field_ord, entry.docid, docid_lo, docid_hi)?;
            }
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| NumericCodecError::Allocation {
                resource: "section bytes",
                bytes: total_len,
            })?;
        for ((_, kind), field) in expected.iter().zip(fields) {
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(field.entries.len())
                .map_err(|_| NumericCodecError::Allocation {
                    resource: "field sort scratch",
                    bytes: field
                        .entries
                        .len()
                        .saturating_mul(std::mem::size_of::<NumericEntry>()),
                })?;
            entries.extend_from_slice(field.entries);
            entries.sort_unstable_by(|left, right| numeric_entry_cmp(*kind, *left, *right));
            if let Some(index) = entries
                .windows(2)
                .position(|pair| numeric_entry_cmp(*kind, pair[0], pair[1]).is_ge())
            {
                return Err(NumericCodecError::NonCanonicalPairOrder {
                    field_ord: field.field_ord,
                    index: index + 1,
                });
            }

            bytes.extend_from_slice(&field.field_ord.to_le_bytes());
            let count =
                u32::try_from(entries.len()).map_err(|_| NumericCodecError::ResourceLimit {
                    resource: "durable field entry count",
                    actual: u64::try_from(entries.len()).unwrap_or(u64::MAX),
                    limit: u64::from(u32::MAX),
                })?;
            bytes.extend_from_slice(&count.to_le_bytes());
            for entry in entries {
                bytes.extend_from_slice(&entry.value.value_bits().to_le_bytes());
                bytes.extend_from_slice(&entry.docid.to_le_bytes());
            }
        }
        debug_assert_eq!(bytes.len(), total_len);
        Ok(Self {
            bytes,
            schema,
            docid_lo,
            docid_hi,
            entry_count: usize::try_from(total_entries).unwrap_or(usize::MAX),
        })
    }

    /// K-way merge ordered, non-overlapping canonical NUMERIC sections.
    ///
    /// Each indexed-or-fast numeric field is merged independently by its schema-typed
    /// `(value, docid)` order. The implementation never concatenates pair
    /// streams and never materializes all entries for a global re-sort.
    ///
    /// # Errors
    ///
    /// Rejects empty input, schema/type drift, range overlap/reversal,
    /// resource abuse, arithmetic overflow, or allocation failure.
    pub fn merge_sorted(
        schema: SchemaDescriptor,
        sections: &[NumericSection<'_>],
    ) -> Result<Self, NumericCodecError> {
        Self::merge_sorted_with_limits(schema, sections, NumericLimits::default())
    }

    /// K-way merge with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::merge_sorted`].
    pub fn merge_sorted_with_limits(
        schema: SchemaDescriptor,
        sections: &[NumericSection<'_>],
        limits: NumericLimits,
    ) -> Result<Self, NumericCodecError> {
        let Some(first) = sections.first() else {
            return Err(NumericCodecError::EmptyMerge);
        };
        let expected = numeric_schema_fields(schema, limits.max_fields)?;
        for (section_index, section) in sections.iter().enumerate() {
            let compared = expected.len().max(section.fields.len());
            for field_index in 0..compared {
                let expected_field = expected.get(field_index).copied();
                let actual_field = section.fields.get(field_index);
                if expected_field.map(|(field_ord, _)| field_ord)
                    != actual_field.map(|field| field.field_ord)
                {
                    return Err(NumericCodecError::UnexpectedField {
                        index: field_index,
                        expected: expected_field.map(|(field_ord, _)| field_ord),
                        actual: actual_field.map(|field| field.field_ord),
                    });
                }
                if let (Some((field_ord, expected_kind)), Some(actual)) =
                    (expected_field, actual_field)
                    && expected_kind != actual.kind
                {
                    return Err(NumericCodecError::ValueTypeMismatch {
                        field_ord,
                        expected: expected_kind.name(),
                        actual: actual.kind.name(),
                    });
                }
            }
            if section_index != 0 {
                let previous_hi = sections[section_index - 1].docid_hi;
                if section.docid_lo < previous_hi {
                    return Err(NumericCodecError::MergeRangeOrder {
                        index: section_index,
                        previous_hi,
                        current_lo: section.docid_lo,
                    });
                }
            }
        }

        let docid_lo = first.docid_lo;
        let docid_hi = sections
            .last()
            .map(|section| section.docid_hi)
            .ok_or(NumericCodecError::EmptyMerge)?;
        checked_numeric_span(docid_lo, docid_hi, limits)?;

        let mut field_counts = Vec::new();
        field_counts
            .try_reserve_exact(expected.len())
            .map_err(|_| NumericCodecError::Allocation {
                resource: "merged field counts",
                bytes: expected.len().saturating_mul(std::mem::size_of::<usize>()),
            })?;
        let mut total_entries = 0_u64;
        let mut total_len = 0_usize;
        for field_index in 0..expected.len() {
            let mut field_count = 0_u64;
            for section in sections {
                let count = u64::try_from(section.fields[field_index].count).unwrap_or(u64::MAX);
                field_count = field_count.checked_add(count).ok_or(
                    NumericCodecError::ArithmeticOverflow {
                        field: "merged field entry count",
                    },
                )?;
            }
            if field_count > limits.max_entries_per_field {
                return Err(NumericCodecError::ResourceLimit {
                    resource: "entries per field",
                    actual: field_count,
                    limit: limits.max_entries_per_field,
                });
            }
            if field_count > u64::from(u32::MAX) {
                return Err(NumericCodecError::ResourceLimit {
                    resource: "durable field entry count",
                    actual: field_count,
                    limit: u64::from(u32::MAX),
                });
            }
            total_entries = total_entries.checked_add(field_count).ok_or(
                NumericCodecError::ArithmeticOverflow {
                    field: "merged total entry count",
                },
            )?;
            if total_entries > limits.max_total_entries {
                return Err(NumericCodecError::ResourceLimit {
                    resource: "total entries",
                    actual: total_entries,
                    limit: limits.max_total_entries,
                });
            }
            let field_count =
                usize::try_from(field_count).map_err(|_| NumericCodecError::ResourceLimit {
                    resource: "host field entry count",
                    actual: field_count,
                    limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                })?;
            let field_len = field_count
                .checked_mul(NUMERIC_PAIR_LEN)
                .and_then(|pairs| pairs.checked_add(NUMERIC_FIELD_HEADER_LEN))
                .ok_or(NumericCodecError::ArithmeticOverflow {
                    field: "merged field byte length",
                })?;
            total_len =
                total_len
                    .checked_add(field_len)
                    .ok_or(NumericCodecError::ArithmeticOverflow {
                        field: "merged section byte length",
                    })?;
            field_counts.push(field_count);
        }
        let total_len_u64 = u64::try_from(total_len).unwrap_or(u64::MAX);
        if total_len_u64 > limits.max_section_bytes {
            return Err(NumericCodecError::ResourceLimit {
                resource: "section bytes",
                actual: total_len_u64,
                limit: limits.max_section_bytes,
            });
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| NumericCodecError::Allocation {
                resource: "merged section bytes",
                bytes: total_len,
            })?;
        for (field_index, &(field_ord, kind)) in expected.iter().enumerate() {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let count = u32::try_from(field_counts[field_index]).map_err(|_| {
                NumericCodecError::ResourceLimit {
                    resource: "durable field entry count",
                    actual: u64::try_from(field_counts[field_index]).unwrap_or(u64::MAX),
                    limit: u64::from(u32::MAX),
                }
            })?;
            bytes.extend_from_slice(&count.to_le_bytes());

            let mut positions = Vec::new();
            positions.try_reserve_exact(sections.len()).map_err(|_| {
                NumericCodecError::Allocation {
                    resource: "field merge cursors",
                    bytes: sections.len().saturating_mul(std::mem::size_of::<usize>()),
                }
            })?;
            positions.resize(sections.len(), 0_usize);
            let mut heap = BinaryHeap::new();
            heap.try_reserve_exact(sections.len())
                .map_err(|_| NumericCodecError::Allocation {
                    resource: "field merge heap",
                    bytes: sections
                        .len()
                        .saturating_mul(std::mem::size_of::<(u64, u32, usize, u64)>()),
                })?;
            for (source_index, section) in sections.iter().enumerate() {
                if let Some(entry) = section.field_at(field_index).entry(0) {
                    heap.push(Reverse(numeric_merge_heap_entry(kind, entry, source_index)));
                }
            }

            let mut emitted = 0_usize;
            let mut previous = None;
            while let Some(Reverse((_, docid, source_index, value_bits))) = heap.pop() {
                let entry = NumericEntry {
                    value: kind.value_from_bits(value_bits),
                    docid,
                };
                if let Some(previous) = previous
                    && !numeric_entry_cmp(kind, previous, entry).is_lt()
                {
                    return Err(NumericCodecError::NonCanonicalPairOrder {
                        field_ord,
                        index: emitted,
                    });
                }
                bytes.extend_from_slice(&value_bits.to_le_bytes());
                bytes.extend_from_slice(&docid.to_le_bytes());
                previous = Some(entry);
                emitted = emitted
                    .checked_add(1)
                    .ok_or(NumericCodecError::ArithmeticOverflow {
                        field: "merged emitted entry count",
                    })?;

                let next = positions[source_index].checked_add(1).ok_or(
                    NumericCodecError::ArithmeticOverflow {
                        field: "field merge cursor",
                    },
                )?;
                positions[source_index] = next;
                if let Some(entry) = sections[source_index].field_at(field_index).entry(next) {
                    heap.push(Reverse(numeric_merge_heap_entry(kind, entry, source_index)));
                }
            }
            debug_assert_eq!(emitted, field_counts[field_index]);
        }
        debug_assert_eq!(bytes.len(), total_len);
        Ok(Self {
            bytes,
            schema,
            docid_lo,
            docid_hi,
            entry_count: usize::try_from(total_entries).unwrap_or(usize::MAX),
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

    /// Number of present pairs across all fields.
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Re-open the owned bytes through the validating zero-copy reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated.
    pub fn section(&self) -> Result<NumericSection<'_>, NumericCodecError> {
        NumericSection::parse(&self.bytes, self.schema, self.docid_lo, self.docid_hi)
    }

    /// Encode the schema's indexed-or-fast numeric columns directly from one Scribe
    /// accumulator, rebasing sparse lease-relative document ordinals exactly
    /// once at the seal boundary.
    ///
    /// # Errors
    ///
    /// Rejects source-column drift, invalid rebases, schema/type mismatch, and
    /// every ordinary NUMERIC encoding failure.
    pub fn encode_accumulator<A: TokenAnalyzer>(
        schema: SchemaDescriptor,
        docid_lo: u64,
        docid_hi: u64,
        lease_docid_base: u64,
        accumulator: &ColumnarAccumulator<A>,
    ) -> Result<Self, NumericCodecError> {
        let expected = numeric_schema_fields(schema, NumericLimits::default().max_fields)?;
        if accumulator.numeric_fields().len() != expected.len() {
            return Err(NumericCodecError::UnexpectedField {
                index: accumulator.numeric_fields().len().min(expected.len()),
                expected: expected
                    .get(accumulator.numeric_fields().len())
                    .map(|(field_ord, _)| *field_ord),
                actual: accumulator
                    .numeric_fields()
                    .get(expected.len())
                    .map(crate::scribe::NumericFieldColumns::field_ord),
            });
        }
        let document_count = accumulator.document_count();
        let mut docids = Vec::new();
        docids
            .try_reserve_exact(document_count)
            .map_err(|_| NumericCodecError::Allocation {
                resource: "rebased source docids",
                bytes: document_count.saturating_mul(std::mem::size_of::<u32>()),
            })?;
        let mut previous = None;
        for (index, &doc_ord) in accumulator.document_ords().iter().enumerate() {
            let docid = lease_docid_base.checked_add(u64::from(doc_ord)).ok_or(
                NumericCodecError::ArithmeticOverflow {
                    field: "source document rebase",
                },
            )?;
            if docid < docid_lo || docid >= docid_hi || docid > u64::from(u32::MAX) {
                return Err(NumericCodecError::SourceDocumentOutOfRange {
                    index,
                    docid,
                    docid_lo,
                    docid_hi,
                });
            }
            if let Some(previous) = previous
                && docid <= previous
            {
                return Err(NumericCodecError::NonAscendingSourceDocuments {
                    index,
                    previous,
                    current: docid,
                });
            }
            previous = Some(docid);
            docids.push(u32::try_from(docid).map_err(|_| {
                NumericCodecError::SourceDocumentOutOfRange {
                    index,
                    docid,
                    docid_lo,
                    docid_hi,
                }
            })?);
        }

        let mut owned_fields = Vec::new();
        owned_fields
            .try_reserve_exact(expected.len())
            .map_err(|_| NumericCodecError::Allocation {
                resource: "source numeric fields",
                bytes: expected
                    .len()
                    .saturating_mul(std::mem::size_of::<Vec<NumericEntry>>()),
            })?;
        for ((&doc_count_field_ord, kind), column) in expected
            .iter()
            .map(|(field_ord, kind)| (field_ord, kind))
            .zip(accumulator.numeric_fields())
        {
            if column.field_ord() != doc_count_field_ord {
                return Err(NumericCodecError::UnexpectedField {
                    index: owned_fields.len(),
                    expected: Some(doc_count_field_ord),
                    actual: Some(column.field_ord()),
                });
            }
            if column.values().len() != document_count {
                return Err(NumericCodecError::InvalidSourceColumn {
                    field_ord: column.field_ord(),
                    expected: document_count,
                    actual: column.values().len(),
                });
            }
            let present_count = column
                .values()
                .iter()
                .filter(|value| value.is_some())
                .count();
            let mut entries = Vec::new();
            entries.try_reserve_exact(present_count).map_err(|_| {
                NumericCodecError::Allocation {
                    resource: "source numeric entries",
                    bytes: present_count.saturating_mul(std::mem::size_of::<NumericEntry>()),
                }
            })?;
            for (&docid, value) in docids.iter().zip(column.values()) {
                if let Some(value) = value {
                    validate_numeric_entry_type(column.field_ord(), *kind, *value)?;
                    entries.push(NumericEntry {
                        value: *value,
                        docid,
                    });
                }
            }
            owned_fields.push(entries);
        }
        let fields = expected
            .iter()
            .zip(&owned_fields)
            .map(|((field_ord, _), entries)| NumericFieldInput::new(*field_ord, entries))
            .collect::<Vec<_>>();
        Self::encode(schema, docid_lo, docid_hi, &fields)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NumericFieldMeta {
    field_ord: u16,
    kind: NumericKind,
    pairs: Range<usize>,
    count: usize,
}

/// Borrowed, fully validated FSLX NUMERIC section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumericSection<'a> {
    bytes: &'a [u8],
    docid_lo: u64,
    docid_hi: u64,
    fields: Vec<NumericFieldMeta>,
    entry_count: usize,
}

impl<'a> NumericSection<'a> {
    /// Validate canonical field order, schema-typed pair order, and Q1 docid
    /// containment without copying pair payloads.
    ///
    /// # Errors
    ///
    /// Rejects schema/field drift, truncation, trailing bytes, noncanonical
    /// pair order, out-of-range docids, resource abuse, and arithmetic
    /// overflow.
    pub fn parse(
        bytes: &'a [u8],
        schema: SchemaDescriptor,
        docid_lo: u64,
        docid_hi: u64,
    ) -> Result<Self, NumericCodecError> {
        Self::parse_with_limits(bytes, schema, docid_lo, docid_hi, NumericLimits::default())
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`].
    pub fn parse_with_limits(
        bytes: &'a [u8],
        schema: SchemaDescriptor,
        docid_lo: u64,
        docid_hi: u64,
        limits: NumericLimits,
    ) -> Result<Self, NumericCodecError> {
        let expected = numeric_schema_fields(schema, limits.max_fields)?;
        checked_numeric_span(docid_lo, docid_hi, limits)?;
        let section_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if section_len > limits.max_section_bytes {
            return Err(NumericCodecError::ResourceLimit {
                resource: "section bytes",
                actual: section_len,
                limit: limits.max_section_bytes,
            });
        }
        let mut fields = Vec::new();
        fields
            .try_reserve_exact(expected.len())
            .map_err(|_| NumericCodecError::Allocation {
                resource: "field metadata",
                bytes: expected
                    .len()
                    .saturating_mul(std::mem::size_of::<NumericFieldMeta>()),
            })?;
        let mut cursor = 0_usize;
        let mut total_entries = 0_u64;
        for (index, &(field_ord, kind)) in expected.iter().enumerate() {
            let header_end = cursor.checked_add(NUMERIC_FIELD_HEADER_LEN).ok_or(
                NumericCodecError::ArithmeticOverflow {
                    field: "field header end",
                },
            )?;
            if header_end > bytes.len() {
                return Err(NumericCodecError::Truncated {
                    expected: header_end,
                    actual: bytes.len(),
                });
            }
            let actual_field = read_u16(bytes, cursor).ok_or(NumericCodecError::Truncated {
                expected: cursor + std::mem::size_of::<u16>(),
                actual: bytes.len(),
            })?;
            if actual_field != field_ord {
                return Err(NumericCodecError::UnexpectedField {
                    index,
                    expected: Some(field_ord),
                    actual: Some(actual_field),
                });
            }
            let count_offset = cursor + std::mem::size_of::<u16>();
            let count_u32 = read_u32(bytes, count_offset).ok_or(NumericCodecError::Truncated {
                expected: header_end,
                actual: bytes.len(),
            })?;
            let count =
                usize::try_from(count_u32).map_err(|_| NumericCodecError::ResourceLimit {
                    resource: "host field entry count",
                    actual: u64::from(count_u32),
                    limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                })?;
            let count_u64 = u64::from(count_u32);
            if count_u64 > limits.max_entries_per_field {
                return Err(NumericCodecError::ResourceLimit {
                    resource: "entries per field",
                    actual: count_u64,
                    limit: limits.max_entries_per_field,
                });
            }
            total_entries = total_entries.checked_add(count_u64).ok_or(
                NumericCodecError::ArithmeticOverflow {
                    field: "total entry count",
                },
            )?;
            if total_entries > limits.max_total_entries {
                return Err(NumericCodecError::ResourceLimit {
                    resource: "total entries",
                    actual: total_entries,
                    limit: limits.max_total_entries,
                });
            }
            let pairs_len = count.checked_mul(NUMERIC_PAIR_LEN).ok_or(
                NumericCodecError::ArithmeticOverflow {
                    field: "field pair bytes",
                },
            )?;
            let pairs_start = header_end;
            let pairs_end = pairs_start.checked_add(pairs_len).ok_or(
                NumericCodecError::ArithmeticOverflow {
                    field: "field pair end",
                },
            )?;
            if pairs_end > bytes.len() {
                return Err(NumericCodecError::Truncated {
                    expected: pairs_end,
                    actual: bytes.len(),
                });
            }

            let mut previous = None;
            for pair_index in 0..count {
                let offset = pairs_start + pair_index * NUMERIC_PAIR_LEN;
                let value_bits = read_u64(bytes, offset).ok_or(NumericCodecError::Truncated {
                    expected: offset + std::mem::size_of::<u64>(),
                    actual: bytes.len(),
                })?;
                let docid_offset = offset + std::mem::size_of::<u64>();
                let docid = read_u32(bytes, docid_offset).ok_or(NumericCodecError::Truncated {
                    expected: offset + NUMERIC_PAIR_LEN,
                    actual: bytes.len(),
                })?;
                validate_numeric_docid(field_ord, docid, docid_lo, docid_hi)?;
                let entry = NumericEntry {
                    value: kind.value_from_bits(value_bits),
                    docid,
                };
                if let Some(previous) = previous
                    && !numeric_entry_cmp(kind, previous, entry).is_lt()
                {
                    return Err(NumericCodecError::NonCanonicalPairOrder {
                        field_ord,
                        index: pair_index,
                    });
                }
                previous = Some(entry);
            }
            fields.push(NumericFieldMeta {
                field_ord,
                kind,
                pairs: pairs_start..pairs_end,
                count,
            });
            cursor = pairs_end;
        }
        if cursor != bytes.len() {
            return Err(NumericCodecError::TrailingBytes {
                expected: cursor,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            fields,
            entry_count: usize::try_from(total_entries).unwrap_or(usize::MAX),
        })
    }

    /// Number of schema-derived indexed-or-fast numeric fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Number of present pairs across all fields.
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Look up one schema field without decoding or copying its pair stream.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<NumericField<'a>> {
        let index = self
            .fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()?;
        Some(self.field_at(index))
    }

    /// Iterate fields in canonical schema order.
    #[must_use]
    pub fn fields(&self) -> impl ExactSizeIterator<Item = NumericField<'a>> + '_ {
        (0..self.fields.len()).map(|index| self.field_at(index))
    }

    fn field_at(&self, index: usize) -> NumericField<'a> {
        let meta = &self.fields[index];
        NumericField {
            bytes: &self.bytes[meta.pairs.clone()],
            field_ord: meta.field_ord,
            kind: meta.kind,
            docid_lo: self.docid_lo,
            docid_hi: self.docid_hi,
            count: meta.count,
        }
    }
}

/// Borrowed, schema-typed view of one validated NUMERIC field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericField<'a> {
    bytes: &'a [u8],
    field_ord: u16,
    kind: NumericKind,
    docid_lo: u64,
    docid_hi: u64,
    count: usize,
}

impl<'a> NumericField<'a> {
    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(self) -> u16 {
        self.field_ord
    }

    /// Whether this field uses signed typed ordering.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self.kind, NumericKind::I64)
    }

    /// Number of present `(value, docid)` pairs.
    #[must_use]
    pub const fn len(self) -> usize {
        self.count
    }

    /// Whether this field has no present values.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.count == 0
    }

    /// Borrow one decoded pair by typed-value order.
    #[must_use]
    pub fn entry(self, index: usize) -> Option<NumericEntry> {
        let offset = index.checked_mul(NUMERIC_PAIR_LEN)?;
        let value_bits = read_u64(self.bytes, offset)?;
        let docid = read_u32(self.bytes, offset.checked_add(std::mem::size_of::<u64>())?)?;
        Some(NumericEntry {
            value: self.kind.value_from_bits(value_bits),
            docid,
        })
    }

    /// Iterate all validated pairs in canonical typed-value order.
    #[must_use]
    pub const fn entries(self) -> NumericEntries<'a> {
        NumericEntries {
            field: self,
            next: 0,
            end: self.count,
        }
    }

    /// Borrow the canonical pair run covered by typed range bounds.
    ///
    /// The iterator remains in `(value, docid)` order and performs no
    /// allocation. Callers that need cursor-friendly order can select live
    /// entries, check cancellation, and sort only the surviving document IDs.
    ///
    /// # Errors
    ///
    /// Returns [`NumericCodecError::BoundTypeMismatch`] when either bound does
    /// not match this field's schema type.
    pub fn range_entries(
        self,
        lower: Bound<NumericValue>,
        upper: Bound<NumericValue>,
    ) -> Result<NumericEntries<'a>, NumericCodecError> {
        validate_numeric_bound(self.field_ord, self.kind, &lower)?;
        validate_numeric_bound(self.field_ord, self.kind, &upper)?;
        let start = match lower {
            Bound::Unbounded => 0,
            Bound::Included(value) => self
                .partition_point(|entry| numeric_value_cmp(self.kind, entry.value, value).is_lt()),
            Bound::Excluded(value) => self
                .partition_point(|entry| !numeric_value_cmp(self.kind, entry.value, value).is_gt()),
        };
        let end = match upper {
            Bound::Unbounded => self.count,
            Bound::Included(value) => self
                .partition_point(|entry| !numeric_value_cmp(self.kind, entry.value, value).is_gt()),
            Bound::Excluded(value) => self
                .partition_point(|entry| numeric_value_cmp(self.kind, entry.value, value).is_lt()),
        };
        Ok(NumericEntries {
            field: self,
            next: start.min(end),
            end,
        })
    }

    /// Binary-search typed bounds and materialize a docid-sorted predicate set.
    ///
    /// The returned set is sorted by global docid rather than by value, so E4
    /// can intersect it with posting cursors without probing every numeric row.
    ///
    /// # Errors
    ///
    /// Returns [`NumericCodecError::BoundTypeMismatch`] when a bound does not
    /// match this field's schema type, or [`NumericCodecError::Allocation`] if
    /// the bounded result set cannot be reserved.
    pub fn range_docids(
        self,
        lower: Bound<NumericValue>,
        upper: Bound<NumericValue>,
    ) -> Result<NumericDocIdSet, NumericCodecError> {
        let entries = self.range_entries(lower, upper)?;
        let count = entries.len();
        let mut docids = Vec::new();
        docids
            .try_reserve_exact(count)
            .map_err(|_| NumericCodecError::Allocation {
                resource: "range docid set",
                bytes: count.saturating_mul(std::mem::size_of::<u32>()),
            })?;
        for entry in entries {
            docids.push(entry.docid);
        }
        docids.sort_unstable();
        docids.dedup();
        Ok(NumericDocIdSet {
            docids,
            docid_lo: self.docid_lo,
            docid_hi: self.docid_hi,
        })
    }

    fn partition_point(self, mut predicate: impl FnMut(NumericEntry) -> bool) -> usize {
        let mut left = 0_usize;
        let mut right = self.count;
        while left < right {
            let middle = left + (right - left) / 2;
            let Some(entry) = self.entry(middle) else {
                break;
            };
            if predicate(entry) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        left
    }
}

/// Iterator over one validated NUMERIC field.
#[derive(Clone, Debug)]
pub struct NumericEntries<'a> {
    field: NumericField<'a>,
    next: usize,
    end: usize,
}

impl Iterator for NumericEntries<'_> {
    type Item = NumericEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.end {
            return None;
        }
        let entry = self.field.entry(self.next)?;
        self.next += 1;
        Some(entry)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NumericEntries<'_> {}

/// Docid-sorted result of one schema-typed NUMERIC range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumericDocIdSet {
    docids: Vec<u32>,
    docid_lo: u64,
    docid_hi: u64,
}

impl NumericDocIdSet {
    /// Number of matching global document IDs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docids.len()
    }

    /// Whether no document satisfies the bounds.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docids.is_empty()
    }

    /// Sorted unique global document IDs.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.docids
    }

    /// Check one absolute document ID by binary search.
    #[must_use]
    pub fn contains(&self, docid: u32) -> bool {
        u64::from(docid) >= self.docid_lo
            && u64::from(docid) < self.docid_hi
            && self.docids.binary_search(&docid).is_ok()
    }

    /// Iterate matching global document IDs in cursor-friendly order.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.docids.iter().copied()
    }
}

fn numeric_schema_fields(
    schema: SchemaDescriptor,
    max_fields: usize,
) -> Result<Vec<(u16, NumericKind)>, NumericCodecError> {
    schema
        .validate()
        .map_err(|error| NumericCodecError::InvalidSchema {
            detail: error.to_string(),
        })?;
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(schema.fields.len().min(max_fields))
        .map_err(|_| NumericCodecError::Allocation {
            resource: "schema field metadata",
            bytes: schema
                .fields
                .len()
                .min(max_fields)
                .saturating_mul(std::mem::size_of::<(u16, NumericKind)>()),
        })?;
    for field in schema.fields {
        let kind = match field.kind {
            FieldKind::I64 { indexed, fast } if indexed || fast => NumericKind::I64,
            FieldKind::U64 { indexed, fast } if indexed || fast => NumericKind::U64,
            _ => continue,
        };
        if fields.len() == max_fields {
            return Err(NumericCodecError::ResourceLimit {
                resource: "indexed-or-fast numeric fields",
                actual: u64::try_from(fields.len()).unwrap_or(u64::MAX) + 1,
                limit: u64::try_from(max_fields).unwrap_or(u64::MAX),
            });
        }
        fields.push((field.id, kind));
    }
    Ok(fields)
}

fn validate_numeric_input_fields(
    expected: &[(u16, NumericKind)],
    fields: &[NumericFieldInput<'_>],
) -> Result<(), NumericCodecError> {
    let common = expected.len().min(fields.len());
    for index in 0..common {
        if expected[index].0 != fields[index].field_ord {
            return Err(NumericCodecError::UnexpectedField {
                index,
                expected: Some(expected[index].0),
                actual: Some(fields[index].field_ord),
            });
        }
    }
    if expected.len() != fields.len() {
        let index = common;
        return Err(NumericCodecError::UnexpectedField {
            index,
            expected: expected.get(index).map(|(field_ord, _)| *field_ord),
            actual: fields.get(index).map(|field| field.field_ord),
        });
    }
    Ok(())
}

fn checked_numeric_span(
    docid_lo: u64,
    docid_hi: u64,
    limits: NumericLimits,
) -> Result<u64, NumericCodecError> {
    if docid_lo > docid_hi || docid_hi > 1_u64 << 32 {
        return Err(NumericCodecError::InvalidDocIdRange { docid_lo, docid_hi });
    }
    let span = docid_hi - docid_lo;
    if span > limits.max_docid_span {
        return Err(NumericCodecError::ResourceLimit {
            resource: "docid span",
            actual: span,
            limit: limits.max_docid_span,
        });
    }
    Ok(span)
}

fn numeric_value_type(value: NumericValue) -> &'static str {
    match value {
        NumericValue::I64(_) => "i64",
        NumericValue::U64(_) => "u64",
    }
}

fn validate_numeric_entry_type(
    field_ord: u16,
    kind: NumericKind,
    value: NumericValue,
) -> Result<(), NumericCodecError> {
    if kind.accepts(value) {
        Ok(())
    } else {
        Err(NumericCodecError::ValueTypeMismatch {
            field_ord,
            expected: kind.name(),
            actual: numeric_value_type(value),
        })
    }
}

fn validate_numeric_bound(
    field_ord: u16,
    kind: NumericKind,
    bound: &Bound<NumericValue>,
) -> Result<(), NumericCodecError> {
    let value = match bound {
        Bound::Included(value) | Bound::Excluded(value) => *value,
        Bound::Unbounded => return Ok(()),
    };
    if kind.accepts(value) {
        Ok(())
    } else {
        Err(NumericCodecError::BoundTypeMismatch {
            field_ord,
            expected: kind.name(),
            actual: numeric_value_type(value),
        })
    }
}

fn validate_numeric_docid(
    field_ord: u16,
    docid: u32,
    docid_lo: u64,
    docid_hi: u64,
) -> Result<(), NumericCodecError> {
    let docid_u64 = u64::from(docid);
    if docid_u64 < docid_lo || docid_u64 >= docid_hi {
        Err(NumericCodecError::DocIdOutOfRange {
            field_ord,
            docid,
            docid_lo,
            docid_hi,
        })
    } else {
        Ok(())
    }
}

fn numeric_value_cmp(
    kind: NumericKind,
    left: NumericValue,
    right: NumericValue,
) -> std::cmp::Ordering {
    match (kind, left, right) {
        (NumericKind::I64, NumericValue::I64(left), NumericValue::I64(right)) => left.cmp(&right),
        (NumericKind::U64, NumericValue::U64(left), NumericValue::U64(right)) => left.cmp(&right),
        _ => unreachable!("NUMERIC values are validated against their schema type before sorting"),
    }
}

fn numeric_entry_cmp(
    kind: NumericKind,
    left: NumericEntry,
    right: NumericEntry,
) -> std::cmp::Ordering {
    numeric_value_cmp(kind, left.value, right.value).then_with(|| left.docid.cmp(&right.docid))
}

fn numeric_merge_heap_entry(
    kind: NumericKind,
    entry: NumericEntry,
    source_index: usize,
) -> (u64, u32, usize, u64) {
    let value_bits = entry.value.value_bits();
    let ordered_value = match kind {
        NumericKind::I64 => value_bits ^ (1_u64 << 63),
        NumericKind::U64 => value_bits,
    };
    (ordered_value, entry.docid, source_index, value_bits)
}
