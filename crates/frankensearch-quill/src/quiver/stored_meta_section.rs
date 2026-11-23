use super::stored_meta::{
    STORED_META_DIRECTORY_ENTRY_LEN, StoredMetaCodecError, StoredMetaFieldInput,
    StoredMetaFieldMeta, StoredMetaLimits, StoredMetaLookup, ValidatedStoredMetaLookupPlan,
};

/// Borrowed zero-copy view of one validated STOREDMETA field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoredMetaField<'a> {
    pub(super) field_ord: u16,
    pub(super) docid_lo: u64,
    pub(super) presence: &'a [u8],
    pub(super) offsets: &'a [u8],
    pub(super) blob: &'a [u8],
}

impl<'a> StoredMetaField<'a> {
    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(self) -> u16 {
        self.field_ord
    }

    /// Exact concatenated opaque blob for diagnostics and concat proofs.
    #[must_use]
    pub const fn blob(self) -> &'a [u8] {
        self.blob
    }

    /// Whether one in-range document carries this field.
    #[must_use]
    pub fn is_present(self, global_docid: u64) -> bool {
        let Some(ordinal) = global_docid.checked_sub(self.docid_lo) else {
            return false;
        };
        let Ok(ordinal) = usize::try_from(ordinal) else {
            return false;
        };
        stored_meta_presence_bit(self.presence, ordinal).unwrap_or(false)
    }

    /// Borrow one opaque value with two little-endian offset reads.
    ///
    /// `None` means the docid is out of range or the value is absent.
    /// `Some(&[])` is a present empty value.
    #[must_use]
    pub fn get(self, global_docid: u64) -> Option<&'a [u8]> {
        let ordinal = global_docid.checked_sub(self.docid_lo)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        if !stored_meta_presence_bit(self.presence, ordinal)? {
            return None;
        }
        let start = usize::try_from(stored_meta_offset(self.offsets, ordinal)?).ok()?;
        let end = usize::try_from(stored_meta_offset(self.offsets, ordinal + 1)?).ok()?;
        self.blob.get(start..end)
    }
}

/// Borrowed, fully validated FSLX STOREDMETA section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredMetaSection<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) docid_lo: u64,
    pub(super) docid_hi: u64,
    pub(super) fields: Vec<StoredMetaFieldMeta>,
}

impl<'a> StoredMetaSection<'a> {
    /// Validate a STOREDMETA payload against its segment range and stored-field
    /// schema set.
    ///
    /// # Errors
    ///
    /// Rejects field drift, noncanonical directory offsets or presence bits,
    /// malformed value offsets, absent values owning bytes, truncation,
    /// trailing bytes, resource abuse, and allocation failure.
    pub fn parse(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
    ) -> Result<Self, StoredMetaCodecError> {
        Self::parse_with_limits(
            bytes,
            docid_lo,
            docid_hi,
            expected_field_ords,
            StoredMetaLimits::default(),
        )
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`], including explicit
    /// limit failures selected by `limits`.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        validate_stored_meta_expected_fields(expected_field_ords, limits.max_fields)?;
        let span = checked_stored_meta_span(docid_lo, docid_hi, limits)?;
        let section_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if section_len > limits.max_section_bytes {
            return Err(StoredMetaCodecError::ResourceLimit {
                resource: "section bytes",
                actual: section_len,
                limit: limits.max_section_bytes,
            });
        }
        let directory_len = expected_field_ords
            .len()
            .checked_mul(STORED_META_DIRECTORY_ENTRY_LEN)
            .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                field: "directory length",
            })?;
        if bytes.len() < directory_len {
            return Err(StoredMetaCodecError::Truncated {
                expected: directory_len,
                actual: bytes.len(),
            });
        }
        if expected_field_ords.is_empty() {
            if bytes.is_empty() {
                return Ok(Self {
                    bytes,
                    docid_lo,
                    docid_hi,
                    fields: Vec::new(),
                });
            }
            return Err(StoredMetaCodecError::TrailingBytes {
                expected: 0,
                actual: bytes.len(),
            });
        }

        let presence_len = stored_meta_presence_len(span)?;
        let offset_count = span
            .checked_add(1)
            .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                field: "offset count",
            })?;
        let offsets_len = offset_count.checked_mul(std::mem::size_of::<u32>()).ok_or(
            StoredMetaCodecError::ArithmeticOverflow {
                field: "offset table length",
            },
        )?;
        let mut fields = Vec::new();
        fields
            .try_reserve_exact(expected_field_ords.len())
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "field metadata",
                bytes: expected_field_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<StoredMetaFieldMeta>()),
            })?;
        let mut canonical_field_offset = directory_len;
        for (index, &expected_field) in expected_field_ords.iter().enumerate() {
            let entry = index * STORED_META_DIRECTORY_ENTRY_LEN;
            let actual_field = u16::from_le_bytes([bytes[entry], bytes[entry + 1]]);
            if actual_field != expected_field {
                return Err(StoredMetaCodecError::UnexpectedField {
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
            let expected_offset = u32::try_from(canonical_field_offset).map_err(|_| {
                StoredMetaCodecError::OffsetUnrepresentable {
                    field_ord: expected_field,
                    offset: canonical_field_offset,
                }
            })?;
            if encoded_offset != expected_offset {
                return Err(StoredMetaCodecError::NonCanonicalFieldOffset {
                    field_ord: expected_field,
                    encoded: encoded_offset,
                    expected: expected_offset,
                });
            }
            let next_field_offset = if index + 1 == expected_field_ords.len() {
                bytes.len()
            } else {
                let next_entry = entry + STORED_META_DIRECTORY_ENTRY_LEN;
                usize::try_from(u32::from_le_bytes([
                    bytes[next_entry + 2],
                    bytes[next_entry + 3],
                    bytes[next_entry + 4],
                    bytes[next_entry + 5],
                ]))
                .map_err(|_| StoredMetaCodecError::ResourceLimit {
                    resource: "host field offset",
                    actual: u64::from(u32::MAX),
                    limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                })?
            };
            if next_field_offset > bytes.len() {
                return Err(StoredMetaCodecError::Truncated {
                    expected: next_field_offset,
                    actual: bytes.len(),
                });
            }
            let presence_end = canonical_field_offset.checked_add(presence_len).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "presence end",
                },
            )?;
            let offsets_end = presence_end.checked_add(offsets_len).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "offset table end",
                },
            )?;
            if next_field_offset < offsets_end || bytes.len() < offsets_end {
                return Err(StoredMetaCodecError::Truncated {
                    expected: offsets_end,
                    actual: next_field_offset.min(bytes.len()),
                });
            }
            let presence = canonical_field_offset..presence_end;
            let offsets = presence_end..offsets_end;
            let blob = offsets_end..next_field_offset;
            if span % 8 != 0 {
                let used = span % 8;
                let allowed = u8::MAX >> (8 - used);
                let final_byte = bytes[presence.end - 1];
                if final_byte & !allowed != 0 {
                    return Err(StoredMetaCodecError::NonCanonicalPresencePadding {
                        field_ord: expected_field,
                        byte: final_byte,
                    });
                }
            }
            let blob_len = blob.len();
            let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
            if blob_len_u64 > limits.max_field_blob_bytes {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "field blob bytes",
                    actual: blob_len_u64,
                    limit: limits.max_field_blob_bytes,
                });
            }
            let offset_bytes = &bytes[offsets.clone()];
            let first_offset =
                stored_meta_offset(offset_bytes, 0).ok_or(StoredMetaCodecError::Truncated {
                    expected: offsets.start + std::mem::size_of::<u32>(),
                    actual: bytes.len(),
                })?;
            if first_offset != 0 {
                return Err(StoredMetaCodecError::NonZeroFirstOffset {
                    field_ord: expected_field,
                    actual: first_offset,
                });
            }
            for value_index in 0..span {
                let start = stored_meta_offset(offset_bytes, value_index).ok_or(
                    StoredMetaCodecError::Truncated {
                        expected: offsets.start + (value_index + 1) * 4,
                        actual: bytes.len(),
                    },
                )?;
                let end = stored_meta_offset(offset_bytes, value_index + 1).ok_or(
                    StoredMetaCodecError::Truncated {
                        expected: offsets.start + (value_index + 2) * 4,
                        actual: bytes.len(),
                    },
                )?;
                if end < start {
                    return Err(StoredMetaCodecError::DescendingOffsets {
                        field_ord: expected_field,
                        index: value_index,
                        start,
                        end,
                    });
                }
                if usize::try_from(end).map_or(true, |offset| offset > blob_len) {
                    return Err(StoredMetaCodecError::OffsetOutOfBounds {
                        field_ord: expected_field,
                        index: value_index,
                        offset: end,
                        blob_len,
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
                let is_present = stored_meta_presence_bit(&bytes[presence.clone()], value_index)
                    .unwrap_or(false);
                if !is_present && start != end {
                    return Err(StoredMetaCodecError::AbsentValueHasBytes {
                        field_ord: expected_field,
                        index: value_index,
                        start,
                        end,
                    });
                }
            }
            let terminal =
                stored_meta_offset(offset_bytes, span).ok_or(StoredMetaCodecError::Truncated {
                    expected: offsets.end,
                    actual: bytes.len(),
                })?;
            if usize::try_from(terminal).ok() != Some(blob_len) {
                if index + 1 == expected_field_ords.len()
                    && usize::try_from(terminal).is_ok_and(|terminal| terminal < blob_len)
                {
                    return Err(StoredMetaCodecError::TrailingBytes {
                        expected: offsets_end + usize::try_from(terminal).unwrap_or(0),
                        actual: bytes.len(),
                    });
                }
                return Err(StoredMetaCodecError::TerminalOffsetMismatch {
                    field_ord: expected_field,
                    terminal,
                    blob_len,
                });
            }
            fields.push(StoredMetaFieldMeta {
                field_ord: expected_field,
                presence,
                offsets,
                blob,
            });
            canonical_field_offset = next_field_offset;
        }
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            fields,
        })
    }

    /// Perform one complete validation and retain only owned lookup metadata.
    ///
    /// The returned plan is bound to this exact byte slice by address and
    /// length. It owns ranges only, never references the source bytes.
    pub(crate) fn validate_metadata(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
    ) -> Result<ValidatedStoredMetaLookupPlan, StoredMetaCodecError> {
        let section = Self::parse(bytes, docid_lo, docid_hi, expected_field_ords)?;
        Ok(ValidatedStoredMetaLookupPlan {
            source_start: bytes.as_ptr() as usize,
            source_len: bytes.len(),
            docid_lo: section.docid_lo,
            fields: section.fields,
        })
    }

    /// Construct a cheap borrowed lookup over the exact bytes validated by
    /// [`Self::validate_metadata`].
    pub(crate) fn from_validated_metadata<'b>(
        bytes: &'b [u8],
        metadata: &'b ValidatedStoredMetaLookupPlan,
    ) -> Result<StoredMetaLookup<'b>, StoredMetaCodecError> {
        if bytes.len() != metadata.source_len || bytes.as_ptr() as usize != metadata.source_start {
            return Err(StoredMetaCodecError::DifferentByteSource);
        }
        Ok(StoredMetaLookup {
            bytes,
            docid_lo: metadata.docid_lo,
            fields: &metadata.fields,
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

    /// Half-open global docid span.
    #[must_use]
    pub const fn span(&self) -> u64 {
        self.docid_hi - self.docid_lo
    }

    /// Number of validated stored fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Bind one stored field for repeated zero-copy lookups.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<StoredMetaField<'a>> {
        let index = self
            .fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()?;
        self.field_at(index)
    }

    /// Iterate borrowed field views in schema order.
    #[must_use]
    pub fn fields(&self) -> impl ExactSizeIterator<Item = StoredMetaField<'a>> + '_ {
        self.fields.iter().map(|field| StoredMetaField {
            field_ord: field.field_ord,
            docid_lo: self.docid_lo,
            presence: &self.bytes[field.presence.clone()],
            offsets: &self.bytes[field.offsets.clone()],
            blob: &self.bytes[field.blob.clone()],
        })
    }

    pub(super) fn field_at(&self, index: usize) -> Option<StoredMetaField<'a>> {
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

pub(super) fn validate_stored_meta_expected_fields(
    expected_field_ords: &[u16],
    max_fields: usize,
) -> Result<(), StoredMetaCodecError> {
    if expected_field_ords.len() > max_fields {
        return Err(StoredMetaCodecError::ResourceLimit {
            resource: "field count",
            actual: u64::try_from(expected_field_ords.len()).unwrap_or(u64::MAX),
            limit: u64::try_from(max_fields).unwrap_or(u64::MAX),
        });
    }
    for (index, pair) in expected_field_ords.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(StoredMetaCodecError::NonAscendingFields {
                index: index + 1,
                previous: pair[0],
                current: pair[1],
            });
        }
    }
    Ok(())
}

pub(super) fn validate_stored_meta_field_set(
    expected_field_ords: &[u16],
    fields: &[StoredMetaFieldInput<'_>],
    max_fields: usize,
) -> Result<(), StoredMetaCodecError> {
    validate_stored_meta_expected_fields(expected_field_ords, max_fields)?;
    let compared = expected_field_ords.len().max(fields.len());
    for index in 0..compared {
        let expected = expected_field_ords.get(index).copied();
        let actual = fields.get(index).map(|field| field.field_ord);
        if expected != actual {
            return Err(StoredMetaCodecError::UnexpectedField {
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

pub(super) fn checked_stored_meta_span(
    docid_lo: u64,
    docid_hi: u64,
    limits: StoredMetaLimits,
) -> Result<usize, StoredMetaCodecError> {
    let span = docid_hi
        .checked_sub(docid_lo)
        .ok_or(StoredMetaCodecError::InvalidDocIdRange { docid_lo, docid_hi })?;
    if span > limits.max_docid_span {
        return Err(StoredMetaCodecError::ResourceLimit {
            resource: "docid span",
            actual: span,
            limit: limits.max_docid_span,
        });
    }
    usize::try_from(span).map_err(|_| StoredMetaCodecError::ResourceLimit {
        resource: "host docid span",
        actual: span,
        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
    })
}

pub(super) fn stored_meta_presence_len(span: usize) -> Result<usize, StoredMetaCodecError> {
    span.checked_add(7)
        .map(|bits| bits / 8)
        .ok_or(StoredMetaCodecError::ArithmeticOverflow {
            field: "presence bitmap length",
        })
}

pub(super) fn stored_meta_layout(
    field_ords: &[u16],
    span: usize,
    blob_lengths: &[usize],
    limits: StoredMetaLimits,
) -> Result<(Vec<usize>, usize), StoredMetaCodecError> {
    let directory_len = field_ords
        .len()
        .checked_mul(STORED_META_DIRECTORY_ENTRY_LEN)
        .ok_or(StoredMetaCodecError::ArithmeticOverflow {
            field: "directory length",
        })?;
    let presence_len = stored_meta_presence_len(span)?;
    let offsets_len = span
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or(StoredMetaCodecError::ArithmeticOverflow {
            field: "offset table length",
        })?;
    let field_prefix_len =
        presence_len
            .checked_add(offsets_len)
            .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                field: "field prefix length",
            })?;
    let mut field_offsets = Vec::new();
    field_offsets
        .try_reserve_exact(field_ords.len())
        .map_err(|_| StoredMetaCodecError::Allocation {
            resource: "directory offsets",
            bytes: field_ords
                .len()
                .saturating_mul(std::mem::size_of::<usize>()),
        })?;
    let mut cursor = directory_len;
    for (index, &field_ord) in field_ords.iter().enumerate() {
        if u32::try_from(cursor).is_err() {
            return Err(StoredMetaCodecError::OffsetUnrepresentable {
                field_ord,
                offset: cursor,
            });
        }
        field_offsets.push(cursor);
        let blob_len =
            *blob_lengths
                .get(index)
                .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                    field: "field blob layout",
                })?;
        cursor = cursor
            .checked_add(field_prefix_len)
            .and_then(|end| end.checked_add(blob_len))
            .ok_or(StoredMetaCodecError::ArithmeticOverflow { field: "field end" })?;
    }
    let section_len = u64::try_from(cursor).unwrap_or(u64::MAX);
    if section_len > limits.max_section_bytes {
        return Err(StoredMetaCodecError::ResourceLimit {
            resource: "section bytes",
            actual: section_len,
            limit: limits.max_section_bytes,
        });
    }
    Ok((field_offsets, cursor))
}

pub(super) fn stored_meta_presence_bit(presence: &[u8], ordinal: usize) -> Option<bool> {
    let byte = *presence.get(ordinal / 8)?;
    Some(byte & (1 << (ordinal % 8)) != 0)
}

pub(super) fn stored_meta_offset(offsets: &[u8], index: usize) -> Option<u32> {
    let start = index.checked_mul(std::mem::size_of::<u32>())?;
    let end = start.checked_add(std::mem::size_of::<u32>())?;
    let bytes = offsets.get(start..end)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}
