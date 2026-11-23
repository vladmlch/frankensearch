use thiserror::Error;

use crate::schema::{FieldKind, SchemaDescriptor};

/// Byte alignment of every per-field FSLX DOCLEN column.
pub const DOCLEN_ALIGNMENT: usize = 64;
/// Canonical byte written for a global-docid hole in a DOCLEN column.
///
/// This is only a canonical fill value. It is also the valid fieldnorm ID for
/// an empty present field, so presence must come from IDMAP or a posting cursor.
pub const DOCLEN_HOLE_FIELDNORM_ID: u8 = 0;
/// Packed size of one `{ field_ord: u16, offset: u32 }` directory entry.
pub const DOCLEN_DIRECTORY_ENTRY_LEN: usize = 6;
/// Packed size of one `{ field_ord: u16, total_tokens: u64, doc_count: u32 }` row.
pub const STATS_ENTRY_LEN: usize = 14;

/// Explicit resource ceilings for an FSLX DOCLEN section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DocLenLimits {
    /// Maximum schema-derived Text/Keyword field count.
    pub max_fields: usize,
    /// Maximum half-open global docid span represented by each column.
    pub max_docid_span: u64,
    /// Maximum complete section size, including directory and padding.
    pub max_section_bytes: u64,
}

impl Default for DocLenLimits {
    fn default() -> Self {
        Self {
            max_fields: 65_536,
            max_docid_span: u64::from(u32::MAX),
            max_section_bytes: u64::from(u32::MAX),
        }
    }
}

/// One raw document-length column supplied to the deterministic DOCLEN writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DocLenFieldInput<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Raw token counts by `global_docid - docid_lo`; `None` denotes a hole.
    pub document_lengths: &'a [Option<u32>],
}

impl<'a> DocLenFieldInput<'a> {
    /// Construct one schema-ordered input column.
    #[must_use]
    pub const fn new(field_ord: u16, document_lengths: &'a [Option<u32>]) -> Self {
        Self {
            field_ord,
            document_lengths,
        }
    }
}

/// Typed failures from encoding or validating an FSLX DOCLEN section.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DocLenCodecError {
    /// A segment's half-open docid range was reversed.
    #[error("invalid DOCLEN docid range [{docid_lo}, {docid_hi})")]
    InvalidDocIdRange {
        /// Inclusive lower bound.
        docid_lo: u64,
        /// Exclusive upper bound.
        docid_hi: u64,
    },
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("DOCLEN {resource} {actual} exceeds limit {limit}")]
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
        "DOCLEN field ordinals are not strictly ascending at index {index}: previous {previous}, current {current}"
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
    #[error("DOCLEN field {index} mismatch: expected {expected:?}, got {actual:?}")]
    UnexpectedField {
        /// Field position in schema order.
        index: usize,
        /// Expected ordinal, if the schema has one at this position.
        expected: Option<u16>,
        /// Supplied or decoded ordinal, if present.
        actual: Option<u16>,
    },
    /// Every field column must cover the complete global-docid span.
    #[error("DOCLEN field {field_ord} has {actual} entries, expected {expected}")]
    ColumnLengthMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Required span.
        expected: usize,
        /// Supplied entries.
        actual: usize,
    },
    /// Checked layout arithmetic overflowed.
    #[error("DOCLEN layout arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout component being computed.
        field: &'static str,
    },
    /// A canonical payload-relative offset did not fit the durable u32 field.
    #[error("DOCLEN offset {offset} for field {field_ord} does not fit u32")]
    OffsetUnrepresentable {
        /// Schema field ordinal.
        field_ord: u16,
        /// Computed offset.
        offset: usize,
    },
    /// A durable directory offset did not equal the one canonical layout.
    #[error("DOCLEN field {field_ord} uses non-canonical offset {encoded}; expected {expected}")]
    NonCanonicalOffset {
        /// Schema field ordinal.
        field_ord: u16,
        /// Durable payload-relative offset.
        encoded: u32,
        /// Required payload-relative offset.
        expected: u32,
    },
    /// Alignment padding is required to be all zeroes.
    #[error("DOCLEN has non-zero alignment padding at byte {offset}")]
    NonZeroPadding {
        /// First rejected padding byte.
        offset: usize,
    },
    /// Durable bytes ended before the canonical layout did.
    #[error("truncated DOCLEN section: expected at least {expected} bytes, got {actual}")]
    Truncated {
        /// Minimum length required by the validated prefix.
        expected: usize,
        /// Available length.
        actual: usize,
    },
    /// Canonical sections end exactly after their final field column.
    #[error("DOCLEN section has trailing bytes: expected {expected}, got {actual}")]
    TrailingBytes {
        /// Canonical complete size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Concatenation requires at least one validated source section.
    #[error("cannot concatenate an empty DOCLEN section list")]
    EmptyConcat,
    /// Concat inputs must be ordered and non-overlapping.
    #[error("DOCLEN concat section {index} begins at {current_lo}, before prior end {previous_hi}")]
    ConcatRangeOrder {
        /// Rejected source index.
        index: usize,
        /// Prior source's exclusive high bound.
        previous_hi: u64,
        /// Current source's inclusive low bound.
        current_lo: u64,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for DOCLEN {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

/// Owned canonical bytes produced by the fresh-seal DOCLEN writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedDocLenSection {
    bytes: Vec<u8>,
    docid_lo: u64,
    docid_hi: u64,
    field_count: usize,
}

impl EncodedDocLenSection {
    /// Encode raw token counts through the pinned Tantivy fieldnorm table.
    ///
    /// `expected_field_ords` is the schema-derived, strictly ascending set of
    /// Text/Keyword fields. Every input column must match it exactly.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid ranges, field-set drift, column-length
    /// mismatch, resource-limit exhaustion, layout overflow, or allocation
    /// failure.
    pub fn encode(
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        fields: &[DocLenFieldInput<'_>],
    ) -> Result<Self, DocLenCodecError> {
        Self::encode_with_limits(
            docid_lo,
            docid_hi,
            expected_field_ords,
            fields,
            DocLenLimits::default(),
        )
    }

    /// Encode with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`], plus explicit
    /// resource-limit failures selected by `limits`.
    pub fn encode_with_limits(
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        fields: &[DocLenFieldInput<'_>],
        limits: DocLenLimits,
    ) -> Result<Self, DocLenCodecError> {
        validate_doclen_field_set(expected_field_ords, fields, limits.max_fields)?;
        let span = checked_doclen_span(docid_lo, docid_hi, limits)?;
        for field in fields {
            if field.document_lengths.len() != span {
                return Err(DocLenCodecError::ColumnLengthMismatch {
                    field_ord: field.field_ord,
                    expected: span,
                    actual: field.document_lengths.len(),
                });
            }
        }

        let (offsets, total_len) = doclen_layout(expected_field_ords, span, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| DocLenCodecError::Allocation {
                resource: "section bytes",
                bytes: total_len,
            })?;
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset)
                .map_err(|_| DocLenCodecError::OffsetUnrepresentable { field_ord, offset })?;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for (field, &offset) in fields.iter().zip(&offsets) {
            bytes.resize(offset, 0);
            bytes.extend(field.document_lengths.iter().map(|length| {
                length.map_or(DOCLEN_HOLE_FIELDNORM_ID, crate::contract::fieldnorm_to_id)
            }));
        }
        debug_assert_eq!(bytes.len(), total_len);
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            field_count: fields.len(),
        })
    }

    /// Concatenate ordered non-overlapping DOCLEN sections.
    ///
    /// Inter-segment docid gaps receive the canonical hole byte. Source
    /// fieldnorm IDs are copied exactly; concat never decodes or re-quantizes
    /// them.
    ///
    /// # Errors
    ///
    /// Rejects empty input, field drift, range overlap/reversal, resource
    /// abuse, arithmetic overflow, or allocation failure.
    pub fn concatenate(
        sections: &[DocLenSection<'_>],
        expected_field_ords: &[u16],
    ) -> Result<Self, DocLenCodecError> {
        Self::concatenate_with_limits(sections, expected_field_ords, DocLenLimits::default())
    }

    /// Concatenate with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::concatenate`].
    pub fn concatenate_with_limits(
        sections: &[DocLenSection<'_>],
        expected_field_ords: &[u16],
        limits: DocLenLimits,
    ) -> Result<Self, DocLenCodecError> {
        let Some(first) = sections.first() else {
            return Err(DocLenCodecError::EmptyConcat);
        };
        validate_doclen_expected_fields(expected_field_ords, limits.max_fields)?;
        for (section_index, section) in sections.iter().enumerate() {
            let compared = expected_field_ords.len().max(section.field_count());
            for field_index in 0..compared {
                let expected = expected_field_ords.get(field_index).copied();
                let actual = section.field_at(field_index).map(DocLenField::field_ord);
                if expected != actual {
                    return Err(DocLenCodecError::UnexpectedField {
                        index: field_index,
                        expected,
                        actual,
                    });
                }
            }
            if section_index != 0 {
                let previous_hi = sections[section_index - 1].docid_hi;
                if section.docid_lo < previous_hi {
                    return Err(DocLenCodecError::ConcatRangeOrder {
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
            .ok_or(DocLenCodecError::EmptyConcat)?;
        let span = checked_doclen_span(docid_lo, docid_hi, limits)?;
        let (offsets, total_len) = doclen_layout(expected_field_ords, span, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| DocLenCodecError::Allocation {
                resource: "concatenated section bytes",
                bytes: total_len,
            })?;
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset)
                .map_err(|_| DocLenCodecError::OffsetUnrepresentable { field_ord, offset })?;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        for (field_index, &offset) in offsets.iter().enumerate() {
            bytes.resize(offset, 0);
            let mut cursor = docid_lo;
            for (section_index, section) in sections.iter().enumerate() {
                let gap = section.docid_lo.checked_sub(cursor).ok_or(
                    DocLenCodecError::ConcatRangeOrder {
                        index: section_index,
                        previous_hi: cursor,
                        current_lo: section.docid_lo,
                    },
                )?;
                let gap = usize::try_from(gap).map_err(|_| DocLenCodecError::ResourceLimit {
                    resource: "concat gap",
                    actual: gap,
                    limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                })?;
                let gap_end =
                    bytes
                        .len()
                        .checked_add(gap)
                        .ok_or(DocLenCodecError::ArithmeticOverflow {
                            field: "concat gap end",
                        })?;
                bytes.resize(gap_end, DOCLEN_HOLE_FIELDNORM_ID);
                let field = section.field_at(field_index).ok_or_else(|| {
                    DocLenCodecError::UnexpectedField {
                        index: field_index,
                        expected: expected_field_ords.get(field_index).copied(),
                        actual: None,
                    }
                })?;
                bytes.extend_from_slice(field.fieldnorm_ids());
                cursor = section.docid_hi;
            }
        }
        debug_assert_eq!(bytes.len(), total_len);
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            field_count: expected_field_ords.len(),
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

    /// Number of schema-indexed text columns in this section.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }

    /// Re-open the owned bytes through the validating reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated.
    pub fn section(
        &self,
        expected_field_ords: &[u16],
    ) -> Result<DocLenSection<'_>, DocLenCodecError> {
        DocLenSection::parse(
            &self.bytes,
            self.docid_lo,
            self.docid_hi,
            expected_field_ords,
        )
    }
}

/// Borrowed view of one validated DOCLEN field column.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DocLenField<'a> {
    field_ord: u16,
    docid_lo: u64,
    fieldnorm_ids: &'a [u8],
}

impl<'a> DocLenField<'a> {
    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(self) -> u16 {
        self.field_ord
    }

    /// Borrow all fieldnorm IDs in global-docid order.
    #[must_use]
    pub const fn fieldnorm_ids(self) -> &'a [u8] {
        self.fieldnorm_ids
    }

    /// Fetch one fieldnorm ID with one checked subtraction and slice lookup.
    ///
    /// `None` means the docid is outside this segment. The byte cannot identify
    /// holes: callers score only docids proven present by IDMAP/posting state.
    #[must_use]
    pub fn fieldnorm_id(self, global_docid: u64) -> Option<u8> {
        let ordinal = global_docid.checked_sub(self.docid_lo)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        self.fieldnorm_ids.get(ordinal).copied()
    }

    /// Decode one in-range fieldnorm ID through the pinned Tantivy table.
    #[must_use]
    pub fn decoded_fieldnorm(self, global_docid: u64) -> Option<u32> {
        self.fieldnorm_id(global_docid)
            .map(crate::contract::id_to_fieldnorm)
    }
}

/// Borrowed, fully validated FSLX DOCLEN section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocLenSection<'a> {
    bytes: &'a [u8],
    docid_lo: u64,
    docid_hi: u64,
    field_count: usize,
}

impl<'a> DocLenSection<'a> {
    /// Validate a DOCLEN payload against its segment range and schema field set.
    ///
    /// # Errors
    ///
    /// Rejects field drift, noncanonical offsets or padding, truncation,
    /// trailing bytes, arithmetic overflow, and resource abuse. Validation is
    /// allocation-free.
    pub fn parse(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
    ) -> Result<Self, DocLenCodecError> {
        Self::parse_with_limits(
            bytes,
            docid_lo,
            docid_hi,
            expected_field_ords,
            DocLenLimits::default(),
        )
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`], including a
    /// resource-limit error when `limits` is exceeded.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        limits: DocLenLimits,
    ) -> Result<Self, DocLenCodecError> {
        Self::parse_with_expected_fields(
            bytes,
            docid_lo,
            docid_hi,
            expected_field_ords.iter().copied(),
            expected_field_ords.len(),
            limits,
        )
    }

    pub(crate) fn parse_schema(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        schema: SchemaDescriptor,
    ) -> Result<Self, DocLenCodecError> {
        let expected_fields = doclen_schema_field_ords(schema);
        let field_count = expected_fields.clone().count();
        Self::parse_with_expected_fields(
            bytes,
            docid_lo,
            docid_hi,
            expected_fields,
            field_count,
            DocLenLimits::default(),
        )
    }

    fn parse_with_expected_fields<I>(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: I,
        field_count: usize,
        limits: DocLenLimits,
    ) -> Result<Self, DocLenCodecError>
    where
        I: Clone + Iterator<Item = u16>,
    {
        validate_doclen_expected_field_iter(
            expected_field_ords.clone(),
            field_count,
            limits.max_fields,
        )?;
        let span = checked_doclen_span(docid_lo, docid_hi, limits)?;
        let section_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if section_len > limits.max_section_bytes {
            return Err(DocLenCodecError::ResourceLimit {
                resource: "section bytes",
                actual: section_len,
                limit: limits.max_section_bytes,
            });
        }
        let directory_len = field_count.checked_mul(DOCLEN_DIRECTORY_ENTRY_LEN).ok_or(
            DocLenCodecError::ArithmeticOverflow {
                field: "directory length",
            },
        )?;
        if bytes.len() < directory_len {
            return Err(DocLenCodecError::Truncated {
                expected: directory_len,
                actual: bytes.len(),
            });
        }

        let canonical_len =
            doclen_layout_len_iter(expected_field_ords.clone(), field_count, span, limits)?;
        let mut previous_end = directory_len;
        let mut canonical_cursor = directory_len;
        for (index, expected_field) in expected_field_ords.enumerate() {
            let canonical_offset =
                align_doclen(canonical_cursor).ok_or(DocLenCodecError::ArithmeticOverflow {
                    field: "aligned column offset",
                })?;
            let entry = index * DOCLEN_DIRECTORY_ENTRY_LEN;
            let actual_field = u16::from_le_bytes([bytes[entry], bytes[entry + 1]]);
            if actual_field != expected_field {
                return Err(DocLenCodecError::UnexpectedField {
                    index,
                    expected: Some(expected_field),
                    actual: Some(actual_field),
                });
            }
            let encoded_offset = u32::from_le_bytes([
                bytes[entry + 2],
                bytes[entry + 3],
                bytes[entry + 4],
                bytes[entry + 5],
            ]);
            let expected_offset = u32::try_from(canonical_offset).map_err(|_| {
                DocLenCodecError::OffsetUnrepresentable {
                    field_ord: expected_field,
                    offset: canonical_offset,
                }
            })?;
            if encoded_offset != expected_offset {
                return Err(DocLenCodecError::NonCanonicalOffset {
                    field_ord: expected_field,
                    encoded: encoded_offset,
                    expected: expected_offset,
                });
            }
            if bytes.len() < canonical_offset {
                return Err(DocLenCodecError::Truncated {
                    expected: canonical_offset,
                    actual: bytes.len(),
                });
            }
            if let Some(relative) = bytes[previous_end..canonical_offset]
                .iter()
                .position(|&byte| byte != 0)
            {
                return Err(DocLenCodecError::NonZeroPadding {
                    offset: previous_end + relative,
                });
            }
            let end =
                canonical_offset
                    .checked_add(span)
                    .ok_or(DocLenCodecError::ArithmeticOverflow {
                        field: "column end",
                    })?;
            if bytes.len() < end {
                return Err(DocLenCodecError::Truncated {
                    expected: end,
                    actual: bytes.len(),
                });
            }
            previous_end = end;
            canonical_cursor = end;
        }
        if bytes.len() > canonical_len {
            return Err(DocLenCodecError::TrailingBytes {
                expected: canonical_len,
                actual: bytes.len(),
            });
        }
        if bytes.len() < canonical_len {
            return Err(DocLenCodecError::Truncated {
                expected: canonical_len,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            field_count,
        })
    }

    /// Exact durable bytes after canonical validation.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Inclusive lower global docid bound.
    #[must_use]
    pub const fn docid_lo(&self) -> u64 {
        self.docid_lo
    }

    /// Exclusive upper global docid bound.
    #[must_use]
    pub const fn docid_hi(&self) -> u64 {
        self.docid_hi
    }

    /// Number of validated schema fields.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }

    /// Bind one field once before the scoring loop.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<DocLenField<'a>> {
        let mut lo = 0;
        let mut hi = self.field_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let candidate = self.field_at(mid)?;
            if candidate.field_ord() < field_ord {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        self.field_at(lo)
            .filter(|candidate| candidate.field_ord() == field_ord)
    }

    /// Iterate borrowed field views in schema order.
    #[must_use]
    pub fn fields(&self) -> impl ExactSizeIterator<Item = DocLenField<'a>> + '_ {
        DocLenFields {
            section: self,
            next_index: 0,
        }
    }

    fn field_at(&self, index: usize) -> Option<DocLenField<'a>> {
        if index >= self.field_count {
            return None;
        }
        let entry = index.checked_mul(DOCLEN_DIRECTORY_ENTRY_LEN)?;
        let field_ord = u16::from_le_bytes([
            *self.bytes.get(entry)?,
            *self.bytes.get(entry.checked_add(1)?)?,
        ]);
        let offset_start = entry.checked_add(2)?;
        let field_offset = usize::try_from(u32::from_le_bytes([
            *self.bytes.get(offset_start)?,
            *self.bytes.get(offset_start.checked_add(1)?)?,
            *self.bytes.get(offset_start.checked_add(2)?)?,
            *self.bytes.get(offset_start.checked_add(3)?)?,
        ]))
        .ok()?;
        let span = usize::try_from(self.docid_hi.checked_sub(self.docid_lo)?).ok()?;
        let field_end = field_offset.checked_add(span)?;
        Some(DocLenField {
            field_ord,
            docid_lo: self.docid_lo,
            fieldnorm_ids: self.bytes.get(field_offset..field_end)?,
        })
    }
}

#[derive(Clone, Debug)]
struct DocLenFields<'section, 'bytes> {
    section: &'section DocLenSection<'bytes>,
    next_index: usize,
}

impl<'bytes> Iterator for DocLenFields<'_, 'bytes> {
    type Item = DocLenField<'bytes>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.next_index;
        if index >= self.section.field_count() {
            return None;
        }
        self.next_index += 1;
        self.section.field_at(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.section.field_count().saturating_sub(self.next_index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DocLenFields<'_, '_> {}

fn validate_doclen_expected_fields(
    expected_field_ords: &[u16],
    max_fields: usize,
) -> Result<(), DocLenCodecError> {
    validate_doclen_expected_field_iter(
        expected_field_ords.iter().copied(),
        expected_field_ords.len(),
        max_fields,
    )
}

fn validate_doclen_expected_field_iter<I>(
    expected_field_ords: I,
    field_count: usize,
    max_fields: usize,
) -> Result<(), DocLenCodecError>
where
    I: Iterator<Item = u16>,
{
    if field_count > max_fields {
        return Err(DocLenCodecError::ResourceLimit {
            resource: "field count",
            actual: u64::try_from(field_count).unwrap_or(u64::MAX),
            limit: u64::try_from(max_fields).unwrap_or(u64::MAX),
        });
    }
    let mut previous = None;
    for (index, current) in expected_field_ords.enumerate() {
        if let Some(previous_field) = previous
            && previous_field >= current
        {
            return Err(DocLenCodecError::NonAscendingFields {
                index,
                previous: previous_field,
                current,
            });
        }
        previous = Some(current);
    }
    Ok(())
}

fn doclen_schema_field_ords(schema: SchemaDescriptor) -> impl Clone + Iterator<Item = u16> {
    schema.fields.iter().filter_map(|field| match field.kind {
        FieldKind::Keyword | FieldKind::Text { .. } => Some(field.id),
        FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => None,
    })
}

fn validate_doclen_field_set(
    expected_field_ords: &[u16],
    fields: &[DocLenFieldInput<'_>],
    max_fields: usize,
) -> Result<(), DocLenCodecError> {
    validate_doclen_expected_fields(expected_field_ords, max_fields)?;
    let compared = expected_field_ords.len().max(fields.len());
    for index in 0..compared {
        let expected = expected_field_ords.get(index).copied();
        let actual = fields.get(index).map(|field| field.field_ord);
        if expected != actual {
            return Err(DocLenCodecError::UnexpectedField {
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn checked_doclen_span(
    docid_lo: u64,
    docid_hi: u64,
    limits: DocLenLimits,
) -> Result<usize, DocLenCodecError> {
    let span = docid_hi
        .checked_sub(docid_lo)
        .ok_or(DocLenCodecError::InvalidDocIdRange { docid_lo, docid_hi })?;
    if span > limits.max_docid_span {
        return Err(DocLenCodecError::ResourceLimit {
            resource: "docid span",
            actual: span,
            limit: limits.max_docid_span,
        });
    }
    usize::try_from(span).map_err(|_| DocLenCodecError::ResourceLimit {
        resource: "host docid span",
        actual: span,
        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
    })
}

fn doclen_layout(
    field_ords: &[u16],
    span: usize,
    limits: DocLenLimits,
) -> Result<(Vec<usize>, usize), DocLenCodecError> {
    let directory_len = field_ords
        .len()
        .checked_mul(DOCLEN_DIRECTORY_ENTRY_LEN)
        .ok_or(DocLenCodecError::ArithmeticOverflow {
            field: "directory length",
        })?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(field_ords.len())
        .map_err(|_| DocLenCodecError::Allocation {
            resource: "directory offsets",
            bytes: field_ords
                .len()
                .saturating_mul(std::mem::size_of::<usize>()),
        })?;
    let mut cursor = directory_len;
    for &field_ord in field_ords {
        let offset = align_doclen(cursor).ok_or(DocLenCodecError::ArithmeticOverflow {
            field: "aligned column offset",
        })?;
        if u32::try_from(offset).is_err() {
            return Err(DocLenCodecError::OffsetUnrepresentable { field_ord, offset });
        }
        offsets.push(offset);
        cursor = offset
            .checked_add(span)
            .ok_or(DocLenCodecError::ArithmeticOverflow {
                field: "column end",
            })?;
    }
    let section_len = u64::try_from(cursor).unwrap_or(u64::MAX);
    if section_len > limits.max_section_bytes {
        return Err(DocLenCodecError::ResourceLimit {
            resource: "section bytes",
            actual: section_len,
            limit: limits.max_section_bytes,
        });
    }
    Ok((offsets, cursor))
}

fn doclen_layout_len_iter<I>(
    field_ords: I,
    field_count: usize,
    span: usize,
    limits: DocLenLimits,
) -> Result<usize, DocLenCodecError>
where
    I: Iterator<Item = u16>,
{
    let directory_len = field_count.checked_mul(DOCLEN_DIRECTORY_ENTRY_LEN).ok_or(
        DocLenCodecError::ArithmeticOverflow {
            field: "directory length",
        },
    )?;
    let mut cursor = directory_len;
    for field_ord in field_ords {
        let offset = align_doclen(cursor).ok_or(DocLenCodecError::ArithmeticOverflow {
            field: "aligned column offset",
        })?;
        if u32::try_from(offset).is_err() {
            return Err(DocLenCodecError::OffsetUnrepresentable { field_ord, offset });
        }
        cursor = offset
            .checked_add(span)
            .ok_or(DocLenCodecError::ArithmeticOverflow {
                field: "column end",
            })?;
    }
    let section_len = u64::try_from(cursor).unwrap_or(u64::MAX);
    if section_len > limits.max_section_bytes {
        return Err(DocLenCodecError::ResourceLimit {
            resource: "section bytes",
            actual: section_len,
            limit: limits.max_section_bytes,
        });
    }
    Ok(cursor)
}

fn align_doclen(value: usize) -> Option<usize> {
    value
        .checked_add(DOCLEN_ALIGNMENT - 1)
        .map(|sum| sum & !(DOCLEN_ALIGNMENT - 1))
}
