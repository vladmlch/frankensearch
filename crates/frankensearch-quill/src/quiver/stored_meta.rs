use std::ops::Range;

use thiserror::Error;

use crate::scribe::{ColumnarAccumulator, TokenAnalyzer};

use super::id_map::append_repeated_u32;
use super::stored_meta_section::{
    StoredMetaField, StoredMetaSection, checked_stored_meta_span, stored_meta_layout,
    stored_meta_offset, stored_meta_presence_bit, stored_meta_presence_len,
    validate_stored_meta_expected_fields, validate_stored_meta_field_set,
};

/// Packed size of one `{ field_ord: u16, field_offset: u32 }` STOREDMETA
/// directory entry.
pub const STORED_META_DIRECTORY_ENTRY_LEN: usize = 6;

/// Explicit resource ceilings for an FSLX STOREDMETA section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoredMetaLimits {
    /// Maximum schema-derived stored field count.
    pub max_fields: usize,
    /// Maximum half-open global docid span represented by each field.
    pub max_docid_span: u64,
    /// Maximum complete section size.
    pub max_section_bytes: u64,
    /// Maximum concatenated byte blob for one stored field.
    pub max_field_blob_bytes: u64,
    /// Maximum single stored value size.
    pub max_value_bytes: u64,
}

impl Default for StoredMetaLimits {
    fn default() -> Self {
        Self {
            max_fields: 65_536,
            max_docid_span: u64::from(u32::MAX),
            max_section_bytes: u64::from(u32::MAX),
            max_field_blob_bytes: u64::from(u32::MAX),
            max_value_bytes: u64::from(u32::MAX),
        }
    }
}

/// One schema-ordered opaque stored-field column supplied to the writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoredMetaFieldInput<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Values indexed by `global_docid - docid_lo`; `None` is an absent field
    /// or a segment-range hole, while `Some(&[])` is a present empty value.
    pub values: &'a [Option<&'a [u8]>],
}

impl<'a> StoredMetaFieldInput<'a> {
    /// Construct one schema-ordered input column.
    #[must_use]
    pub const fn new(field_ord: u16, values: &'a [Option<&'a [u8]>]) -> Self {
        Self { field_ord, values }
    }
}

/// Typed failures from STOREDMETA encoding, validation, or concatenation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StoredMetaCodecError {
    /// A segment's half-open docid range was reversed.
    #[error("invalid STOREDMETA docid range [{docid_lo}, {docid_hi})")]
    InvalidDocIdRange {
        /// Inclusive lower bound.
        docid_lo: u64,
        /// Exclusive upper bound.
        docid_hi: u64,
    },
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("STOREDMETA {resource} {actual} exceeds limit {limit}")]
    ResourceLimit {
        /// Bounded resource name.
        resource: &'static str,
        /// Rejected amount.
        actual: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// Expected schema field ordinals must be strictly ascending.
    #[error(
        "STOREDMETA field ordinals are not strictly ascending at index {index}: previous {previous}, current {current}"
    )]
    NonAscendingFields {
        /// Rejected field index.
        index: usize,
        /// Previous ordinal.
        previous: u16,
        /// Current ordinal.
        current: u16,
    },
    /// Input or durable field identity disagreed with the schema-derived set.
    #[error("STOREDMETA field {index} mismatch: expected {expected:?}, got {actual:?}")]
    UnexpectedField {
        /// Field position in schema order.
        index: usize,
        /// Expected ordinal, if any.
        expected: Option<u16>,
        /// Supplied or decoded ordinal, if present.
        actual: Option<u16>,
    },
    /// Every stored field column must cover the complete segment span.
    #[error("STOREDMETA field {field_ord} has {actual} entries, expected {expected}")]
    ColumnLengthMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Required span.
        expected: usize,
        /// Supplied entries.
        actual: usize,
    },
    /// Sparse accumulator documents must map into the exact segment range.
    #[error(
        "STOREDMETA source document {index} maps to docid {docid}, outside [{docid_lo}, {docid_hi})"
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
        "STOREDMETA source documents are not strictly ascending at index {index}: previous {previous}, current {current}"
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
    #[error("invalid STOREDMETA source column {field_ord} at value {index}: {detail}")]
    InvalidSourceColumn {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Rejected completed-document index.
        index: usize,
        /// Violated invariant.
        detail: &'static str,
    },
    /// Checked layout arithmetic overflowed.
    #[error("STOREDMETA arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout component being computed.
        field: &'static str,
    },
    /// A section-relative field offset did not fit its durable u32 slot.
    #[error("STOREDMETA offset {offset} for field {field_ord} does not fit u32")]
    OffsetUnrepresentable {
        /// Schema field ordinal.
        field_ord: u16,
        /// Computed offset.
        offset: usize,
    },
    /// A durable field offset did not equal the canonical packed layout.
    #[error(
        "STOREDMETA field {field_ord} uses non-canonical offset {encoded}; expected {expected}"
    )]
    NonCanonicalFieldOffset {
        /// Schema field ordinal.
        field_ord: u16,
        /// Durable section-relative offset.
        encoded: u32,
        /// Required section-relative offset.
        expected: u32,
    },
    /// Unused high bits in the final presence byte must be zero.
    #[error("STOREDMETA field {field_ord} has non-zero unused presence bits in byte {byte:#04x}")]
    NonCanonicalPresencePadding {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected final presence byte.
        byte: u8,
    },
    /// Every offset table begins at zero.
    #[error("STOREDMETA field {field_ord} first offset is {actual}, expected 0")]
    NonZeroFirstOffset {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected first offset.
        actual: u32,
    },
    /// Offsets must be monotone.
    #[error("STOREDMETA field {field_ord} offsets descend at value {index}: {start} to {end}")]
    DescendingOffsets {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected value ordinal.
        index: usize,
        /// Start offset.
        start: u32,
        /// End offset.
        end: u32,
    },
    /// An offset pointed beyond the field blob declared by the next directory
    /// entry or section end.
    #[error(
        "STOREDMETA field {field_ord} offset {offset} at value {index} exceeds blob length {blob_len}"
    )]
    OffsetOutOfBounds {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected value ordinal.
        index: usize,
        /// Rejected offset.
        offset: u32,
        /// Exact available blob length.
        blob_len: usize,
    },
    /// An absent value or hole cannot own blob bytes.
    #[error("STOREDMETA field {field_ord} absent value {index} spans bytes {start}..{end}")]
    AbsentValueHasBytes {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected value ordinal.
        index: usize,
        /// Start offset.
        start: u32,
        /// End offset.
        end: u32,
    },
    /// The terminal offset must name the exact end of the field payload.
    #[error(
        "STOREDMETA field {field_ord} terminal offset is {terminal}, but blob length is {blob_len}"
    )]
    TerminalOffsetMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Encoded terminal offset.
        terminal: u32,
        /// Exact bytes available before the next field or section end.
        blob_len: usize,
    },
    /// Durable bytes ended before a canonical component did.
    #[error("truncated STOREDMETA section: expected at least {expected} bytes, got {actual}")]
    Truncated {
        /// Minimum required length.
        expected: usize,
        /// Available length.
        actual: usize,
    },
    /// Canonical sections end exactly after their final field blob.
    #[error("STOREDMETA section has trailing bytes: expected {expected}, got {actual}")]
    TrailingBytes {
        /// Canonical complete size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// A cached lookup plan was paired with bytes other than the exact source
    /// it completely validated.
    #[error("validated STOREDMETA metadata is bound to a different byte source")]
    DifferentByteSource,
    /// Concatenation requires at least one validated source section.
    #[error("cannot concatenate an empty STOREDMETA section list")]
    EmptyConcat,
    /// Concat inputs must be ordered and non-overlapping.
    #[error(
        "STOREDMETA concat section {index} begins at {current_lo}, before prior end {previous_hi}"
    )]
    ConcatRangeOrder {
        /// Rejected source index.
        index: usize,
        /// Prior source's exclusive high bound.
        previous_hi: u64,
        /// Current source's inclusive low bound.
        current_lo: u64,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for STOREDMETA {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StoredMetaFieldMeta {
    pub(super) field_ord: u16,
    pub(super) presence: Range<usize>,
    pub(super) offsets: Range<usize>,
    pub(super) blob: Range<usize>,
}

/// Owned, allocation-bounded metadata produced by one complete STOREDMETA
/// validation.
///
/// This remains crate-private so callers cannot pair the validated ranges with
/// unrelated bytes. [`StoredMetaSection::from_validated_metadata`] proves the
/// exact live slice address and length before exposing a borrowed lookup.
#[derive(Debug)]
pub struct ValidatedStoredMetaLookupPlan {
    pub(super) source_start: usize,
    pub(super) source_len: usize,
    pub(super) docid_lo: u64,
    pub(super) fields: Vec<StoredMetaFieldMeta>,
}

/// Cheap borrowed lookup over one exact STOREDMETA source validated by a
/// [`ValidatedStoredMetaLookupPlan`].
#[derive(Clone, Copy, Debug)]
pub struct StoredMetaLookup<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) docid_lo: u64,
    pub(super) fields: &'a [StoredMetaFieldMeta],
}

impl<'a> StoredMetaLookup<'a> {
    /// Bind one stored field for repeated zero-copy lookups.
    #[must_use]
    pub fn field(self, field_ord: u16) -> Option<StoredMetaField<'a>> {
        let index = self
            .fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()?;
        self.field_at(index)
    }

    fn field_at(self, index: usize) -> Option<StoredMetaField<'a>> {
        let field = self.fields.get(index)?;
        Some(StoredMetaField {
            field_ord: field.field_ord,
            docid_lo: self.docid_lo,
            presence: self.bytes.get(field.presence.clone())?,
            offsets: self.bytes.get(field.offsets.clone())?,
            blob: self.bytes.get(field.blob.clone())?,
        })
    }
}

/// Owned canonical bytes produced by the fresh-seal STOREDMETA writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedStoredMetaSection {
    pub(super) bytes: Vec<u8>,
    pub(super) docid_lo: u64,
    pub(super) docid_hi: u64,
    field_count: usize,
    blob_bytes: u64,
}

/// Canonical STOREDMETA bytes that keep the accumulator's field blobs borrowed.
///
/// Byte-for-byte the same section as
/// [`EncodedStoredMetaSection::encode_accumulator`], emitted without
/// materializing it first.
///
/// # Why this exists (`bd-4xr99`)
///
/// A stored byte used to be copied twice per seal: once from the caller's
/// borrowed input into [`StoredFieldColumns`]'s owned blob, and again from that
/// blob into the section's owned `Vec<u8>` — which the segment assembler then
/// copied a third time into the durable buffer. Only the first copy is forced:
/// it is where borrowed input becomes bytes that outlive the caller. The
/// second buys nothing, because each field's blob is already contiguous and
/// already in final order; the section merely concatenates prefixes and blobs.
///
/// The sibling Delta seal path never had this shape — `build_delta_stored_meta`
/// assembles `Vec<Option<&[u8]>>` of borrowed slices — so this closes a
/// sibling-path asymmetry rather than inventing a new representation.
///
/// A section is an alternating run of owned prefixes and borrowed blobs: the
/// field directory, then per field a presence bitmap, an offset table, and that
/// field's blob last. [`Self::write_into`] emits them in order, so the durable
/// buffer receives each stored byte exactly once.
#[derive(Debug)]
pub struct BorrowedStoredMetaSection<'a> {
    /// Owned prefix runs: `prefixes[0]` is the field directory, and
    /// `prefixes[i + 1]` is field `i`'s presence bitmap plus offset table.
    prefixes: Vec<Vec<u8>>,
    /// Field blobs, borrowed from the accumulator in field order.
    blobs: Vec<&'a [u8]>,
    pub(super) docid_lo: u64,
    pub(super) docid_hi: u64,
    field_count: usize,
    blob_bytes: u64,
    total_len: usize,
}

#[cfg(test)]
std::thread_local! {
    static STORED_META_MATERIALIZATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub fn reset_stored_meta_materializations() {
    STORED_META_MATERIALIZATIONS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub fn stored_meta_materializations() -> usize {
    STORED_META_MATERIALIZATIONS.with(std::cell::Cell::get)
}

impl BorrowedStoredMetaSection<'_> {
    /// Exact durable length of the section this will emit.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.total_len
    }

    /// Whether the section carries no bytes at all.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Inclusive-exclusive global docid span this section covers.
    #[must_use]
    pub const fn docid_range(&self) -> (u64, u64) {
        (self.docid_lo, self.docid_hi)
    }

    /// Number of stored fields represented.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }

    /// Total opaque value bytes across every field.
    #[must_use]
    pub const fn blob_bytes(&self) -> u64 {
        self.blob_bytes
    }

    /// Append the exact durable section bytes to `output`.
    ///
    /// Emits directory, then per field the prefix followed by that field's
    /// borrowed blob — the same order the owned writer materializes.
    pub fn write_into(&self, output: &mut Vec<u8>) {
        let start = output.len();
        if let Some(directory) = self.prefixes.first() {
            output.extend_from_slice(directory);
        }
        for (prefix, blob) in self.prefixes.iter().skip(1).zip(&self.blobs) {
            output.extend_from_slice(prefix);
            output.extend_from_slice(blob);
        }
        debug_assert_eq!(output.len() - start, self.total_len);
    }

    /// Materialize the section into owned bytes.
    ///
    /// This reintroduces the copy the borrowed form exists to avoid, so it is
    /// for tests and for callers that genuinely need an owned section.
    ///
    /// # Errors
    ///
    /// Returns [`StoredMetaCodecError::Allocation`] when the exact owned
    /// section buffer cannot be reserved.
    pub fn to_encoded(&self) -> Result<EncodedStoredMetaSection, StoredMetaCodecError> {
        #[cfg(test)]
        STORED_META_MATERIALIZATIONS.with(|calls| calls.set(calls.get() + 1));
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.total_len)
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "materialized stored metadata section",
                bytes: self.total_len,
            })?;
        self.write_into(&mut bytes);
        Ok(EncodedStoredMetaSection {
            bytes,
            docid_lo: self.docid_lo,
            docid_hi: self.docid_hi,
            field_count: self.field_count,
            blob_bytes: self.blob_bytes,
        })
    }
}

/// Validated exact layout for directly concatenating STOREDMETA source views.
pub struct StoredMetaConcatPlan {
    pub(super) docid_lo: u64,
    pub(super) docid_hi: u64,
    pub(super) span: usize,
    presence_len: usize,
    blob_lengths: Vec<usize>,
    field_offsets: Vec<usize>,
    total_len: usize,
    total_blob_bytes: u64,
}

impl EncodedStoredMetaSection {
    /// Encode schema-derived opaque stored columns.
    ///
    /// # Errors
    ///
    /// Rejects invalid ranges, field-set drift, column-length mismatch,
    /// individual or aggregate blob overflow, resource abuse, and allocation
    /// failure.
    pub fn encode(
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        fields: &[StoredMetaFieldInput<'_>],
    ) -> Result<Self, StoredMetaCodecError> {
        Self::encode_with_limits(
            docid_lo,
            docid_hi,
            expected_field_ords,
            fields,
            StoredMetaLimits::default(),
        )
    }

    /// Encode with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`], including explicit
    /// resource-limit failures selected by `limits`.
    pub fn encode_with_limits(
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        fields: &[StoredMetaFieldInput<'_>],
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        validate_stored_meta_field_set(expected_field_ords, fields, limits.max_fields)?;
        let span = checked_stored_meta_span(docid_lo, docid_hi, limits)?;
        let mut blob_lengths = Vec::new();
        blob_lengths.try_reserve_exact(fields.len()).map_err(|_| {
            StoredMetaCodecError::Allocation {
                resource: "field blob lengths",
                bytes: fields.len().saturating_mul(std::mem::size_of::<usize>()),
            }
        })?;
        let mut total_blob_bytes = 0_u64;
        for field in fields {
            if field.values.len() != span {
                return Err(StoredMetaCodecError::ColumnLengthMismatch {
                    field_ord: field.field_ord,
                    expected: span,
                    actual: field.values.len(),
                });
            }
            let mut blob_len = 0_usize;
            for value in field.values.iter().flatten() {
                let value_len = u64::try_from(value.len()).unwrap_or(u64::MAX);
                if value_len > limits.max_value_bytes {
                    return Err(StoredMetaCodecError::ResourceLimit {
                        resource: "value bytes",
                        actual: value_len,
                        limit: limits.max_value_bytes,
                    });
                }
                blob_len = blob_len.checked_add(value.len()).ok_or(
                    StoredMetaCodecError::ArithmeticOverflow {
                        field: "field blob length",
                    },
                )?;
            }
            let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
            if blob_len_u64 > limits.max_field_blob_bytes {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "field blob bytes",
                    actual: blob_len_u64,
                    limit: limits.max_field_blob_bytes,
                });
            }
            if u32::try_from(blob_len).is_err() {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "durable field blob bytes",
                    actual: blob_len_u64,
                    limit: u64::from(u32::MAX),
                });
            }
            total_blob_bytes = total_blob_bytes.checked_add(blob_len_u64).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "total blob bytes",
                },
            )?;
            blob_lengths.push(blob_len);
        }

        let (field_offsets, total_len) =
            stored_meta_layout(expected_field_ords, span, &blob_lengths, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "section bytes",
                bytes: total_len,
            })?;
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&field_offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset)
                .map_err(|_| StoredMetaCodecError::OffsetUnrepresentable { field_ord, offset })?;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        let presence_len = stored_meta_presence_len(span)?;
        for ((field, &field_offset), &blob_len) in
            fields.iter().zip(&field_offsets).zip(&blob_lengths)
        {
            debug_assert_eq!(bytes.len(), field_offset);
            let presence_start = bytes.len();
            bytes.resize(
                presence_start.checked_add(presence_len).ok_or(
                    StoredMetaCodecError::ArithmeticOverflow {
                        field: "presence end",
                    },
                )?,
                0,
            );
            for (index, value) in field.values.iter().enumerate() {
                if value.is_some() {
                    bytes[presence_start + index / 8] |= 1 << (index % 8);
                }
            }
            let mut current_offset = 0_u32;
            bytes.extend_from_slice(&current_offset.to_le_bytes());
            for value in field.values {
                if let Some(value) = value {
                    current_offset = current_offset
                        .checked_add(u32::try_from(value.len()).map_err(|_| {
                            StoredMetaCodecError::ResourceLimit {
                                resource: "value bytes",
                                actual: u64::try_from(value.len()).unwrap_or(u64::MAX),
                                limit: u64::from(u32::MAX),
                            }
                        })?)
                        .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                            field: "value offset",
                        })?;
                }
                bytes.extend_from_slice(&current_offset.to_le_bytes());
            }
            debug_assert_eq!(usize::try_from(current_offset).ok(), Some(blob_len));
            for value in field.values.iter().flatten() {
                bytes.extend_from_slice(value);
            }
            tracing::debug!(
                field_ord = field.field_ord,
                blob_bytes = blob_len,
                "encoded Quill STOREDMETA field blob"
            );
        }
        debug_assert_eq!(bytes.len(), total_len);
        tracing::debug!(
            field_count = fields.len(),
            blob_bytes = total_blob_bytes,
            section_bytes = total_len,
            "encoded Quill STOREDMETA section"
        );
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            field_count: fields.len(),
            blob_bytes: total_blob_bytes,
        })
    }

    /// Seal the stored columns of one Scribe accumulator into owned bytes.
    ///
    /// Materializes what [`BorrowedStoredMetaSection::encode_accumulator`]
    /// produces. Prefer the borrowed form on the seal path: it emits the same
    /// bytes without copying every field blob first (`bd-4xr99`).
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as
    /// [`BorrowedStoredMetaSection::encode_accumulator`].
    pub fn encode_accumulator<A: TokenAnalyzer>(
        docid_lo: u64,
        docid_hi: u64,
        lease_docid_base: u64,
        accumulator: &ColumnarAccumulator<A>,
    ) -> Result<Self, StoredMetaCodecError> {
        BorrowedStoredMetaSection::encode_accumulator(
            docid_lo,
            docid_hi,
            lease_docid_base,
            accumulator,
        )?
        .to_encoded()
    }

    /// Owned counterpart of
    /// [`BorrowedStoredMetaSection::encode_accumulator_with_limits`].
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as
    /// [`BorrowedStoredMetaSection::encode_accumulator`].
    pub fn encode_accumulator_with_limits<A: TokenAnalyzer>(
        docid_lo: u64,
        docid_hi: u64,
        lease_docid_base: u64,
        accumulator: &ColumnarAccumulator<A>,
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        BorrowedStoredMetaSection::encode_accumulator_with_limits(
            docid_lo,
            docid_hi,
            lease_docid_base,
            accumulator,
            limits,
        )?
        .to_encoded()
    }
}

impl<'a> BorrowedStoredMetaSection<'a> {
    /// Seal the stored columns of one Scribe accumulator directly into the
    /// segment's positional global-docid span.
    ///
    /// `lease_docid_base` is the global docid corresponding to Scribe's local
    /// ordinal zero. Sparse local ordinals become zero-presence holes without
    /// materializing a span-sized value matrix. Field ordinals come directly
    /// from the accumulator's validated schema-derived stored columns.
    ///
    /// # Errors
    ///
    /// Rejects a range/rebase mismatch, malformed source-column invariant,
    /// durable offset overflow, resource abuse, or allocation failure.
    pub fn encode_accumulator<A: TokenAnalyzer>(
        docid_lo: u64,
        docid_hi: u64,
        lease_docid_base: u64,
        accumulator: &'a ColumnarAccumulator<A>,
    ) -> Result<Self, StoredMetaCodecError> {
        Self::encode_accumulator_with_limits(
            docid_lo,
            docid_hi,
            lease_docid_base,
            accumulator,
            StoredMetaLimits::default(),
        )
    }

    /// Seal Scribe columns with caller-selected validation and allocation
    /// ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode_accumulator`].
    pub fn encode_accumulator_with_limits<A: TokenAnalyzer>(
        docid_lo: u64,
        docid_hi: u64,
        lease_docid_base: u64,
        accumulator: &'a ColumnarAccumulator<A>,
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        let span = checked_stored_meta_span(docid_lo, docid_hi, limits)?;
        let document_ords = accumulator.document_ords();
        let stored_fields = accumulator.stored_fields();

        let mut previous_docid = None;
        for (index, &document_ord) in document_ords.iter().enumerate() {
            let docid = lease_docid_base
                .checked_add(u64::from(document_ord))
                .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                    field: "source document rebase",
                })?;
            if docid < docid_lo || docid >= docid_hi {
                return Err(StoredMetaCodecError::SourceDocumentOutOfRange {
                    index,
                    docid,
                    docid_lo,
                    docid_hi,
                });
            }
            if let Some(previous) = previous_docid
                && docid <= previous
            {
                return Err(StoredMetaCodecError::NonAscendingSourceDocuments {
                    index,
                    previous,
                    current: docid,
                });
            }
            previous_docid = Some(docid);
        }

        let mut expected_field_ords = Vec::new();
        expected_field_ords
            .try_reserve_exact(stored_fields.len())
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "accumulator field ordinals",
                bytes: stored_fields
                    .len()
                    .saturating_mul(std::mem::size_of::<u16>()),
            })?;
        expected_field_ords.extend(stored_fields.iter().map(|field| field.field_ord()));
        validate_stored_meta_expected_fields(&expected_field_ords, limits.max_fields)?;

        let expected_offset_count =
            document_ords
                .len()
                .checked_add(1)
                .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                    field: "source offset count",
                })?;
        let mut blob_lengths = Vec::new();
        blob_lengths
            .try_reserve_exact(stored_fields.len())
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "accumulator field blob lengths",
                bytes: stored_fields
                    .len()
                    .saturating_mul(std::mem::size_of::<usize>()),
            })?;
        let mut total_blob_bytes = 0_u64;
        for field in stored_fields {
            if field.document_count() != document_ords.len() {
                return Err(StoredMetaCodecError::ColumnLengthMismatch {
                    field_ord: field.field_ord(),
                    expected: document_ords.len(),
                    actual: field.document_count(),
                });
            }
            if field.presence().len() != document_ords.len() {
                return Err(StoredMetaCodecError::InvalidSourceColumn {
                    field_ord: field.field_ord(),
                    index: field.presence().len(),
                    detail: "presence length does not match completed documents",
                });
            }
            if field.offsets().len() != expected_offset_count {
                return Err(StoredMetaCodecError::InvalidSourceColumn {
                    field_ord: field.field_ord(),
                    index: field.offsets().len(),
                    detail: "offset count is not completed documents plus one",
                });
            }
            let first = field.offsets().first().copied().ok_or_else(|| {
                StoredMetaCodecError::InvalidSourceColumn {
                    field_ord: field.field_ord(),
                    index: 0,
                    detail: "offset table is empty",
                }
            })?;
            if first != 0 {
                return Err(StoredMetaCodecError::NonZeroFirstOffset {
                    field_ord: field.field_ord(),
                    actual: first,
                });
            }
            for index in 0..document_ords.len() {
                let presence = field.presence()[index];
                if presence > 1 {
                    return Err(StoredMetaCodecError::InvalidSourceColumn {
                        field_ord: field.field_ord(),
                        index,
                        detail: "presence byte is not canonical zero or one",
                    });
                }
                let start = field.offsets()[index];
                let end = field.offsets()[index + 1];
                if end < start {
                    return Err(StoredMetaCodecError::DescendingOffsets {
                        field_ord: field.field_ord(),
                        index,
                        start,
                        end,
                    });
                }
                if usize::try_from(end).map_or(true, |offset| offset > field.blob().len()) {
                    return Err(StoredMetaCodecError::OffsetOutOfBounds {
                        field_ord: field.field_ord(),
                        index,
                        offset: end,
                        blob_len: field.blob().len(),
                    });
                }
                if presence == 0 && start != end {
                    return Err(StoredMetaCodecError::AbsentValueHasBytes {
                        field_ord: field.field_ord(),
                        index,
                        start,
                        end,
                    });
                }
                let value_len = u64::from(end - start);
                if value_len > limits.max_value_bytes {
                    return Err(StoredMetaCodecError::ResourceLimit {
                        resource: "value bytes",
                        actual: value_len,
                        limit: limits.max_value_bytes,
                    });
                }
            }
            let terminal = field.offsets().last().copied().ok_or_else(|| {
                StoredMetaCodecError::InvalidSourceColumn {
                    field_ord: field.field_ord(),
                    index: 0,
                    detail: "offset table is empty",
                }
            })?;
            if usize::try_from(terminal).ok() != Some(field.blob().len()) {
                return Err(StoredMetaCodecError::TerminalOffsetMismatch {
                    field_ord: field.field_ord(),
                    terminal,
                    blob_len: field.blob().len(),
                });
            }
            let blob_len = field.blob().len();
            let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
            if blob_len_u64 > limits.max_field_blob_bytes {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "field blob bytes",
                    actual: blob_len_u64,
                    limit: limits.max_field_blob_bytes,
                });
            }
            total_blob_bytes = total_blob_bytes.checked_add(blob_len_u64).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "total blob bytes",
                },
            )?;
            blob_lengths.push(blob_len);
        }

        let (field_offsets, total_len) =
            stored_meta_layout(&expected_field_ords, span, &blob_lengths, limits)?;
        // One owned prefix run per field, plus the directory at index 0. Each
        // field's blob stays borrowed and is emitted straight into the durable
        // buffer by `write_into`, so a stored byte is copied once per seal
        // rather than twice (`bd-4xr99`).
        let mut prefixes: Vec<Vec<u8>> = Vec::new();
        prefixes
            .try_reserve_exact(stored_fields.len() + 1)
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "section prefix runs",
                bytes: (stored_fields.len() + 1) * std::mem::size_of::<Vec<u8>>(),
            })?;
        let mut blobs: Vec<&'a [u8]> = Vec::new();
        blobs.try_reserve_exact(stored_fields.len()).map_err(|_| {
            StoredMetaCodecError::Allocation {
                resource: "section blob views",
                bytes: stored_fields.len() * std::mem::size_of::<&[u8]>(),
            }
        })?;

        let directory_len = expected_field_ords
            .len()
            .checked_mul(STORED_META_DIRECTORY_ENTRY_LEN)
            .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                field: "directory length",
            })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(directory_len)
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "section directory bytes",
                bytes: directory_len,
            })?;
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&field_offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset)
                .map_err(|_| StoredMetaCodecError::OffsetUnrepresentable { field_ord, offset })?;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        // Running length of the section emitted so far, which the per-field
        // `field_offset` assertions below are stated against. The owned writer
        // read this off `bytes.len()`; the prefixes are separate buffers now,
        // so it is tracked explicitly and means exactly the same thing.
        let mut emitted = bytes.len();
        prefixes.push(bytes);

        let presence_len = stored_meta_presence_len(span)?;
        let field_prefix_len = span
            .checked_add(1)
            .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
            .and_then(|offsets_len| presence_len.checked_add(offsets_len))
            .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                field: "field prefix length",
            })?;
        for ((field, &field_offset), &blob_len) in
            stored_fields.iter().zip(&field_offsets).zip(&blob_lengths)
        {
            debug_assert_eq!(emitted, field_offset);
            // Prefix-local: this field's presence bitmap starts at 0 of its own
            // run, where the owned writer indexed from the section start.
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(field_prefix_len).map_err(|_| {
                StoredMetaCodecError::Allocation {
                    resource: "field prefix bytes",
                    bytes: field_prefix_len,
                }
            })?;
            let presence_start = 0_usize;
            let presence_end = presence_start.checked_add(presence_len).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "presence end",
                },
            )?;
            bytes.resize(presence_end, 0);
            let mut current_offset = 0_u32;
            bytes.extend_from_slice(&current_offset.to_le_bytes());
            let mut source_index = 0_usize;
            for ordinal in 0..span {
                let global_docid = docid_lo
                    .checked_add(u64::try_from(ordinal).map_err(|_| {
                        StoredMetaCodecError::ArithmeticOverflow {
                            field: "global document ordinal",
                        }
                    })?)
                    .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                        field: "global document id",
                    })?;
                let next_source_docid = document_ords
                    .get(source_index)
                    .map(|&document_ord| {
                        lease_docid_base.checked_add(u64::from(document_ord)).ok_or(
                            StoredMetaCodecError::ArithmeticOverflow {
                                field: "source document rebase",
                            },
                        )
                    })
                    .transpose()?;
                if next_source_docid == Some(global_docid) {
                    if field.presence()[source_index] == 1 {
                        bytes[presence_start + ordinal / 8] |= 1 << (ordinal % 8);
                    }
                    current_offset = field.offsets()[source_index + 1];
                    source_index += 1;
                }
                bytes.extend_from_slice(&current_offset.to_le_bytes());
            }
            debug_assert_eq!(source_index, document_ords.len());
            debug_assert_eq!(usize::try_from(current_offset).ok(), Some(blob_len));
            emitted = emitted
                .checked_add(bytes.len())
                .and_then(|used| used.checked_add(blob_len))
                .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                    field: "emitted section length",
                })?;
            prefixes.push(bytes);
            blobs.push(field.blob());
            tracing::debug!(
                field_ord = field.field_ord(),
                blob_bytes = blob_len,
                "encoded Quill STOREDMETA accumulator field blob"
            );
        }
        debug_assert_eq!(emitted, total_len);
        tracing::debug!(
            field_count = stored_fields.len(),
            blob_bytes = total_blob_bytes,
            section_bytes = total_len,
            "encoded Quill STOREDMETA section from Scribe accumulator"
        );
        Ok(BorrowedStoredMetaSection {
            prefixes,
            blobs,
            docid_lo,
            docid_hi,
            field_count: stored_fields.len(),
            blob_bytes: total_blob_bytes,
            total_len,
        })
    }
}

impl EncodedStoredMetaSection {
    /// Concatenate ordered non-overlapping STOREDMETA sections.
    ///
    /// Inter-segment docid gaps become holes. Source values remain opaque; the
    /// output only rebases offsets and concatenates their exact bytes.
    ///
    /// # Errors
    ///
    /// Rejects empty input, field drift, range overlap/reversal, resource
    /// abuse, arithmetic overflow, or allocation failure.
    pub fn concatenate(
        sections: &[StoredMetaSection<'_>],
        expected_field_ords: &[u16],
    ) -> Result<Self, StoredMetaCodecError> {
        Self::concatenate_with_limits(sections, expected_field_ords, StoredMetaLimits::default())
    }

    /// Preflight an ordinary concat for direct emission into a larger buffer.
    pub(crate) fn plan_concatenate(
        sections: &[StoredMetaSection<'_>],
        expected_field_ords: &[u16],
    ) -> Result<StoredMetaConcatPlan, StoredMetaCodecError> {
        StoredMetaConcatPlan::new(sections, expected_field_ords, StoredMetaLimits::default())
    }

    /// Concatenate with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::concatenate`].
    pub fn concatenate_with_limits(
        sections: &[StoredMetaSection<'_>],
        expected_field_ords: &[u16],
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        let plan = StoredMetaConcatPlan::new(sections, expected_field_ords, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(plan.total_len)
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "section bytes",
                bytes: plan.total_len,
            })?;
        plan.append_to(sections, expected_field_ords, &mut bytes);
        Ok(Self {
            bytes,
            docid_lo: plan.docid_lo,
            docid_hi: plan.docid_hi,
            field_count: expected_field_ords.len(),
            blob_bytes: plan.total_blob_bytes,
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

    /// Number of schema-stored fields.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }

    /// Sum of opaque field blob bytes, excluding directories and offsets.
    #[must_use]
    pub const fn blob_bytes(&self) -> u64 {
        self.blob_bytes
    }

    /// Re-open the owned bytes through the validating zero-copy reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated.
    pub fn section(
        &self,
        expected_field_ords: &[u16],
    ) -> Result<StoredMetaSection<'_>, StoredMetaCodecError> {
        StoredMetaSection::parse(
            &self.bytes,
            self.docid_lo,
            self.docid_hi,
            expected_field_ords,
        )
    }
}

impl StoredMetaConcatPlan {
    fn new(
        sections: &[StoredMetaSection<'_>],
        expected_field_ords: &[u16],
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        let Some(first) = sections.first() else {
            return Err(StoredMetaCodecError::EmptyConcat);
        };
        validate_stored_meta_expected_fields(expected_field_ords, limits.max_fields)?;
        for (section_index, section) in sections.iter().enumerate() {
            let compared = expected_field_ords.len().max(section.fields.len());
            for field_index in 0..compared {
                let expected = expected_field_ords.get(field_index).copied();
                let actual = section.fields.get(field_index).map(|field| field.field_ord);
                if expected != actual {
                    return Err(StoredMetaCodecError::UnexpectedField {
                        index: field_index,
                        expected,
                        actual,
                    });
                }
            }
            if section_index != 0 {
                let previous_hi = sections[section_index - 1].docid_hi;
                if section.docid_lo < previous_hi {
                    return Err(StoredMetaCodecError::ConcatRangeOrder {
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
            .ok_or(StoredMetaCodecError::EmptyConcat)?;
        let span = checked_stored_meta_span(docid_lo, docid_hi, limits)?;

        let mut blob_lengths = Vec::new();
        blob_lengths
            .try_reserve_exact(expected_field_ords.len())
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "concat field blob lengths",
                bytes: expected_field_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<usize>()),
            })?;
        let mut total_blob_bytes = 0_u64;
        for (field_index, &field_ord) in expected_field_ords.iter().enumerate() {
            let mut blob_len = 0_usize;
            for section in sections {
                let field =
                    section
                        .field_at(field_index)
                        .ok_or(StoredMetaCodecError::UnexpectedField {
                            index: field_index,
                            expected: Some(field_ord),
                            actual: None,
                        })?;
                let source_span = usize::try_from(section.span()).map_err(|_| {
                    StoredMetaCodecError::ResourceLimit {
                        resource: "source docid span",
                        actual: section.span(),
                        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                    }
                })?;
                for ordinal in 0..source_span {
                    let start = stored_meta_offset(field.offsets, ordinal).ok_or_else(|| {
                        StoredMetaCodecError::Truncated {
                            expected: (ordinal + 1).saturating_mul(std::mem::size_of::<u32>()),
                            actual: field.offsets.len(),
                        }
                    })?;
                    let end = stored_meta_offset(field.offsets, ordinal + 1).ok_or_else(|| {
                        StoredMetaCodecError::Truncated {
                            expected: (ordinal + 2).saturating_mul(std::mem::size_of::<u32>()),
                            actual: field.offsets.len(),
                        }
                    })?;
                    let value_len = u64::from(end - start);
                    if value_len > limits.max_value_bytes {
                        return Err(StoredMetaCodecError::ResourceLimit {
                            resource: "value bytes",
                            actual: value_len,
                            limit: limits.max_value_bytes,
                        });
                    }
                }
                blob_len = blob_len.checked_add(field.blob.len()).ok_or(
                    StoredMetaCodecError::ArithmeticOverflow {
                        field: "concatenated field blob length",
                    },
                )?;
            }
            let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
            if blob_len_u64 > limits.max_field_blob_bytes {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "field blob bytes",
                    actual: blob_len_u64,
                    limit: limits.max_field_blob_bytes,
                });
            }
            if u32::try_from(blob_len).is_err() {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "durable field blob bytes",
                    actual: blob_len_u64,
                    limit: u64::from(u32::MAX),
                });
            }
            total_blob_bytes = total_blob_bytes.checked_add(blob_len_u64).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "total blob bytes",
                },
            )?;
            blob_lengths.push(blob_len);
        }

        let (field_offsets, total_len) =
            stored_meta_layout(expected_field_ords, span, &blob_lengths, limits)?;
        Ok(Self {
            docid_lo,
            docid_hi,
            span,
            presence_len: stored_meta_presence_len(span)?,
            blob_lengths,
            field_offsets,
            total_len,
            total_blob_bytes,
        })
    }

    /// Exact canonical byte length of the merged STOREDMETA payload.
    pub const fn encoded_len(&self) -> usize {
        self.total_len
    }

    /// Append the canonical payload after [`Self::new`] validated the sources.
    pub fn append_to(
        &self,
        sections: &[StoredMetaSection<'_>],
        expected_field_ords: &[u16],
        bytes: &mut Vec<u8>,
    ) {
        let section_start = bytes.len();
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&self.field_offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset).expect("validated STOREDMETA field offset fits u32");
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        for (field_index, ((&field_ord, &field_offset), &blob_len)) in expected_field_ords
            .iter()
            .zip(&self.field_offsets)
            .zip(&self.blob_lengths)
            .enumerate()
        {
            debug_assert_eq!(bytes.len() - section_start, field_offset);
            let presence_start = bytes.len();
            bytes.resize(presence_start + self.presence_len, 0);
            let mut current_offset = 0_u32;
            bytes.extend_from_slice(&current_offset.to_le_bytes());
            let mut output_ordinal = 0_usize;
            let mut cursor = self.docid_lo;
            for section in sections {
                let gap = usize::try_from(section.docid_lo - cursor)
                    .expect("validated STOREDMETA gap fits usize");
                output_ordinal += gap;
                append_repeated_u32(bytes, current_offset, gap);

                let field = section
                    .field_at(field_index)
                    .expect("validated STOREDMETA source field");
                let source_span =
                    usize::try_from(section.span()).expect("validated STOREDMETA span fits usize");
                for ordinal in 0..source_span {
                    if stored_meta_presence_bit(field.presence, ordinal).unwrap_or(false) {
                        bytes[presence_start + output_ordinal / 8] |= 1 << (output_ordinal % 8);
                    }
                    let start = stored_meta_offset(field.offsets, ordinal)
                        .expect("validated STOREDMETA source offset");
                    let end = stored_meta_offset(field.offsets, ordinal + 1)
                        .expect("validated STOREDMETA source offset end");
                    current_offset = current_offset
                        .checked_add(end - start)
                        .expect("validated STOREDMETA blob length fits u32");
                    bytes.extend_from_slice(&current_offset.to_le_bytes());
                    output_ordinal += 1;
                }
                cursor = section.docid_hi;
            }
            debug_assert_eq!(output_ordinal, self.span);
            debug_assert_eq!(usize::try_from(current_offset).ok(), Some(blob_len));
            for section in sections {
                let field = section
                    .field_at(field_index)
                    .expect("validated STOREDMETA source field");
                bytes.extend_from_slice(field.blob);
            }
            tracing::debug!(
                field_ord,
                blob_bytes = blob_len,
                source_segments = sections.len(),
                "concatenated Quill STOREDMETA field blob"
            );
        }
        debug_assert_eq!(bytes.len() - section_start, self.total_len);
        tracing::debug!(
            field_count = expected_field_ords.len(),
            blob_bytes = self.total_blob_bytes,
            section_bytes = self.total_len,
            source_segments = sections.len(),
            "concatenated Quill STOREDMETA section"
        );
    }
}
