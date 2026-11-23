use std::mem::size_of;

use ahash::AHashSet;
use frankensearch_core::DocId;

use crate::contract::fieldnorm_to_id;
use crate::grimoire::MAX_TERM_BYTES;
use crate::quiver::NumericValue;
use crate::schema::FieldKind;
use crate::scribe::DOC_ORDS_PER_LEASE;

use super::types::{
    DeltaApply, DeltaError, DeltaFieldNorm, DeltaNumericValue, DeltaStoredValue, DeltaTermPosting,
    PostingRecord, ResolvedDocumentValues, TermChain,
};
use super::{DeltaSegment, HASH_SLOT_ESTIMATE, MAX_GLOBAL_DOCID_EXCLUSIVE, NEW_TERM_SENTINEL};

impl DeltaSegment {
    /// Apply one complete document without seal sidecar values.
    ///
    /// The physical row records an absent content hash and absent
    /// numeric/stored values. It remains process-locally searchable, but a
    /// later FSLX seal rejects it rather than fabricating an identity witness.
    /// Use [`Self::apply_document_with_values`] for sealable rows.
    ///
    /// # Errors
    ///
    /// Delegates all validation and allocation failures to
    /// [`Self::apply_document_with_values`].
    pub fn apply_document(
        &mut self,
        global_docid: u32,
        document_id: DocId,
        fieldnorms: &[DeltaFieldNorm],
        postings: &[DeltaTermPosting<'_>],
    ) -> Result<DeltaApply, DeltaError> {
        self.apply_document_inner(
            global_docid,
            document_id,
            None,
            fieldnorms,
            postings,
            &[],
            &[],
        )
    }

    /// Apply one complete document with every sidecar required for sealing.
    ///
    /// Numeric and stored values may be in any order, but a field may appear in
    /// only one input slice. A stored indexed-or-fast numeric supplied through
    /// `numeric_values` receives canonical little-endian stored bytes. Supplying
    /// its exact eight stored bytes instead also populates the numeric column.
    /// A budget crossing accepts the entire row and reports `seal_required`; a
    /// document is never split.
    ///
    /// # Errors
    ///
    /// Semantic validation and all fallible capacity planning complete before
    /// logical state changes.
    pub fn apply_document_with_values(
        &mut self,
        global_docid: u32,
        document_id: DocId,
        content_hash: u64,
        fieldnorms: &[DeltaFieldNorm],
        postings: &[DeltaTermPosting<'_>],
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<DeltaApply, DeltaError> {
        self.apply_document_inner(
            global_docid,
            document_id,
            Some(content_hash),
            fieldnorms,
            postings,
            numeric_values,
            stored_values,
        )
    }

    fn apply_document_inner(
        &mut self,
        global_docid: u32,
        document_id: DocId,
        content_hash: Option<u64>,
        fieldnorms: &[DeltaFieldNorm],
        postings: &[DeltaTermPosting<'_>],
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<DeltaApply, DeltaError> {
        self.validate_document(
            global_docid,
            &document_id,
            fieldnorms,
            postings,
            numeric_values,
            stored_values,
        )?;
        let (resolved_numeric_values, resolved_stored_values) =
            self.resolve_document_values(numeric_values, stored_values)?;
        let mut document_terms = Vec::new();
        reserve_vec(
            &mut document_terms,
            postings.len(),
            "delta document term scratch",
        )?;
        let mut posting_plans = Vec::new();
        let mut position_plans = Vec::new();
        reserve_vec(
            &mut posting_plans,
            postings.len(),
            "delta posting reservation plan",
        )?;
        reserve_vec(
            &mut position_plans,
            postings.len(),
            "delta position reservation plan",
        )?;
        let mut new_term_count = 0_usize;
        for posting in postings {
            let planned_term = self.terms.find(posting.field_ord, posting.term);
            let chain = if let Some(term_index) = planned_term {
                let chain =
                    self.chains[usize::try_from(term_index).expect("u32 term id fits usize")];
                if let Some(previous) = chain.last_docid
                    && global_docid <= previous
                {
                    return Err(DeltaError::PostingOrder {
                        field_ord: posting.field_ord,
                        term: posting.term.to_vec(),
                        previous,
                        global_docid,
                    });
                }
                chain
            } else {
                new_term_count =
                    new_term_count
                        .checked_add(1)
                        .ok_or(DeltaError::CountOverflow {
                            resource: "term",
                            count: usize::MAX,
                        })?;
                TermChain::default()
            };
            let position_count = posting.positions.map_or(0, <[u32]>::len);
            let final_position_count = chain.positions.len.checked_add(position_count).ok_or(
                DeltaError::CountOverflow {
                    resource: "position",
                    count: usize::MAX,
                },
            )?;
            u32::try_from(final_position_count).map_err(|_| DeltaError::CountOverflow {
                resource: "position",
                count: final_position_count,
            })?;
            posting_plans.push((chain.postings, 1));
            position_plans.push((chain.positions, position_count));
            document_terms.push(planned_term.unwrap_or(NEW_TERM_SENTINEL));
        }
        let final_term_count =
            self.chains
                .len()
                .checked_add(new_term_count)
                .ok_or(DeltaError::CountOverflow {
                    resource: "term",
                    count: usize::MAX,
                })?;
        u32::try_from(final_term_count).map_err(|_| DeltaError::CountOverflow {
            resource: "term",
            count: final_term_count,
        })?;
        let final_document_term_count = self
            .document_term_ids
            .len()
            .checked_add(postings.len())
            .ok_or(DeltaError::CountOverflow {
                resource: "document term",
                count: usize::MAX,
            })?;
        u32::try_from(final_document_term_count).map_err(|_| DeltaError::CountOverflow {
            resource: "document term",
            count: final_document_term_count,
        })?;
        self.reserve_document(postings.len())?;
        self.posting_arena.reserve_appends(&posting_plans)?;
        self.position_arena.reserve_appends(&position_plans)?;
        let document_id_len = document_id.len();
        let overlay_id = document_id.clone();
        let identity_was_live = self.live_ids.contains_key(&overlay_id);
        let stored_value_bytes = resolved_stored_values
            .iter()
            .flatten()
            .map(Vec::len)
            .sum::<usize>();

        let mut added_bytes = 0_usize;
        for (posting, planned_term) in postings.iter().zip(&mut document_terms) {
            let field_index = self
                .field_index(posting.field_ord)
                .expect("validated indexed-string field has a fieldnorm column");
            let frequency = if self.fields[field_index].positions {
                posting.frequency
            } else {
                1
            };
            let fieldnorm_id = fieldnorms[field_index].fieldnorm_id;
            let (term_index, term_bytes) = if *planned_term == NEW_TERM_SENTINEL {
                self.terms.intern_accounted(posting.field_ord, posting.term)
            } else {
                (*planned_term, 0)
            };
            *planned_term = term_index;
            added_bytes = added_bytes.saturating_add(term_bytes);
            let term_offset = usize::try_from(term_index).expect("u32 term id fits usize");
            if term_offset == self.chains.len() {
                self.chains.push(TermChain::default());
                added_bytes = added_bytes.saturating_add(size_of::<TermChain>());
            }
            let chain = &mut self.chains[term_offset];
            let position_start = if let Some(positions) = posting.positions {
                let start = u32::try_from(chain.positions.len)
                    .expect("preflighted position start fits u32");
                for &position in positions {
                    added_bytes = added_bytes.saturating_add(
                        self.position_arena
                            .append_reserved(&mut chain.positions, position),
                    );
                }
                start
            } else {
                0
            };
            added_bytes = added_bytes.saturating_add(self.posting_arena.append_reserved(
                &mut chain.postings,
                PostingRecord {
                    global_docid,
                    frequency,
                    position_start,
                    has_positions: posting.positions.is_some(),
                },
            ));
            chain.last_docid = Some(global_docid);
            // The row just appended is live until something tombstones it.
            // Saturating here would silently under-count a term rather than
            // fail, and the arena's own preflight already bounds the chain
            // length, so an overflow at this point is a broken invariant.
            chain.live_len = chain
                .live_len
                .checked_add(1)
                .expect("preflighted chain length fits u32");
            chain.max_frequency_code = chain
                .max_frequency_code
                .max(crate::contract::block_max_frequency_to_code(frequency));
            chain.min_fieldnorm_id = chain.min_fieldnorm_id.min(fieldnorm_id);
        }

        self.document_docids.push(global_docid);
        self.document_ids.push(document_id);
        self.document_content_hashes.push(content_hash);
        self.document_term_ids.extend_from_slice(&document_terms);
        let offset = u32::try_from(self.document_term_ids.len())
            .expect("preflighted document-term offset fits u32");
        self.document_term_offsets.push(offset);
        for (column, input) in self.fieldnorms.iter_mut().zip(fieldnorms) {
            column.raw_lengths.push(input.raw_length);
            column.fieldnorm_ids.push(input.fieldnorm_id);
        }
        for (column, value) in self.numeric_fields.iter_mut().zip(resolved_numeric_values) {
            column.values.push(value);
        }
        for (column, value) in self.stored_fields.iter_mut().zip(resolved_stored_values) {
            column.payload_bytes = column
                .payload_bytes
                .checked_add(value.as_ref().map_or(0, Vec::len))
                .expect("validated stored payload length fits usize");
            column.values.push(value);
        }
        added_bytes = added_bytes
            .saturating_add(size_of::<u32>())
            .saturating_add(size_of::<DocId>() + document_id_len)
            .saturating_add(size_of::<Option<u64>>())
            .saturating_add(size_of::<u32>())
            .saturating_add(postings.len().saturating_mul(size_of::<u32>()))
            .saturating_add(
                self.fieldnorms
                    .len()
                    .saturating_mul(size_of::<u32>() + size_of::<u8>()),
            )
            .saturating_add(
                self.numeric_fields
                    .len()
                    .saturating_mul(size_of::<Option<NumericValue>>()),
            )
            .saturating_add(
                self.stored_fields
                    .len()
                    .saturating_mul(size_of::<Option<Vec<u8>>>()),
            )
            .saturating_add(stored_value_bytes);
        let replaced_delta_docid = self.live_ids.insert(overlay_id, global_docid);
        if !identity_was_live {
            added_bytes = added_bytes.saturating_add(
                size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE + document_id_len,
            );
        }
        if let Some(replaced) = replaced_delta_docid {
            added_bytes = added_bytes.saturating_add(self.mark_tombstone(replaced));
        }
        self.logical_bytes_used = self.logical_bytes_used.saturating_add(added_bytes);
        let bytes_used = self.logical_bytes_used;
        Ok(DeltaApply {
            replaced_delta_docid,
            bytes_used,
            seal_required: bytes_used >= self.budget_bytes,
        })
    }

    fn validate_document(
        &self,
        global_docid: u32,
        document_id: &str,
        fieldnorms: &[DeltaFieldNorm],
        postings: &[DeltaTermPosting<'_>],
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<(), DeltaError> {
        if self.lease_ordinal(global_docid).is_none() {
            return Err(DeltaError::DocumentOutsideLease {
                global_docid,
                lease_base: self.lease_base,
                lease_end: self.lease_end,
            });
        }
        if u64::from(global_docid) < self.next_docid_floor {
            return Err(DeltaError::DocumentBeforeFloor {
                global_docid,
                floor: self.next_docid_floor,
            });
        }
        if let Some(&previous) = self.document_docids.last()
            && global_docid <= previous
        {
            return Err(DeltaError::DocumentOrder {
                previous,
                global_docid,
            });
        }
        if document_id.is_empty() {
            return Err(DeltaError::EmptyDocumentId);
        }
        if document_id.len() > u32::MAX as usize {
            return Err(DeltaError::DocumentIdTooLarge {
                bytes: document_id.len(),
            });
        }
        if fieldnorms.len() != self.fields.len() {
            return Err(DeltaError::FieldNormCount {
                expected: self.fields.len(),
                actual: fieldnorms.len(),
            });
        }
        for (index, (field, input)) in self.fields.iter().zip(fieldnorms).enumerate() {
            if input.field_ord != field.field_ord {
                return Err(DeltaError::FieldNormOrder {
                    index,
                    expected: field.field_ord,
                    actual: input.field_ord,
                });
            }
            let expected = fieldnorm_to_id(input.raw_length);
            if input.fieldnorm_id != expected {
                return Err(DeltaError::FieldNormMismatch {
                    field_ord: input.field_ord,
                    raw_length: input.raw_length,
                    expected,
                    actual: input.fieldnorm_id,
                });
            }
        }

        let mut unique: AHashSet<(u16, &[u8])> = AHashSet::new();
        unique
            .try_reserve(postings.len())
            .map_err(|_| DeltaError::Allocation {
                resource: "delta document term validation",
                additional: postings.len(),
            })?;
        for posting in postings {
            let Some(field_index) = self.field_index(posting.field_ord) else {
                return Err(DeltaError::UnknownPostingField {
                    field_ord: posting.field_ord,
                });
            };
            if posting.term.len() > MAX_TERM_BYTES {
                return Err(DeltaError::TermTooLarge {
                    field_ord: posting.field_ord,
                    bytes: posting.term.len(),
                });
            }
            if !unique.insert((posting.field_ord, posting.term)) {
                return Err(DeltaError::DuplicateDocumentTerm {
                    field_ord: posting.field_ord,
                    term: posting.term.to_vec(),
                });
            }
            if posting.frequency == 0 {
                return Err(DeltaError::ZeroFrequency {
                    field_ord: posting.field_ord,
                    term: posting.term.to_vec(),
                });
            }
            match (self.fields[field_index].positions, posting.positions) {
                (true, None) => {
                    return Err(DeltaError::MissingPositions {
                        field_ord: posting.field_ord,
                        term: posting.term.to_vec(),
                    });
                }
                (false, Some(_)) => {
                    return Err(DeltaError::ForbiddenPositions {
                        field_ord: posting.field_ord,
                        term: posting.term.to_vec(),
                    });
                }
                (true, Some(positions)) => {
                    if usize::try_from(posting.frequency).ok() != Some(positions.len()) {
                        return Err(DeltaError::PositionCountMismatch {
                            field_ord: posting.field_ord,
                            term: posting.term.to_vec(),
                            frequency: posting.frequency,
                            positions: positions.len(),
                        });
                    }
                    if let Some(pair) = positions.windows(2).find(|pair| pair[0] > pair[1]) {
                        return Err(DeltaError::DescendingPosition {
                            field_ord: posting.field_ord,
                            term: posting.term.to_vec(),
                            previous: pair[0],
                            current: pair[1],
                        });
                    }
                }
                (false, None) => {}
            }
        }
        self.validate_document_values(global_docid, numeric_values, stored_values)
    }

    fn validate_document_values(
        &self,
        global_docid: u32,
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<(), DeltaError> {
        let mut seen_fields = Vec::new();
        reserve_vec(
            &mut seen_fields,
            self.schema.fields.len(),
            "delta document value validation",
        )?;
        seen_fields.resize(self.schema.fields.len(), false);

        for value in numeric_values {
            let field_index = usize::from(value.field_ord);
            let Some(field) = self.schema.fields.get(field_index) else {
                return Err(DeltaError::UnknownValueField {
                    field_ord: value.field_ord,
                });
            };
            if field.id != value.field_ord {
                return Err(DeltaError::UnknownValueField {
                    field_ord: value.field_ord,
                });
            }
            if std::mem::replace(&mut seen_fields[field_index], true) {
                return Err(DeltaError::DuplicateValueField {
                    global_docid,
                    field_ord: value.field_ord,
                });
            }
            match (field.kind, value.value) {
                (FieldKind::I64 { indexed, fast }, NumericValue::I64(_)) if indexed || fast => {}
                (FieldKind::U64 { indexed, fast }, NumericValue::U64(_)) if indexed || fast => {}
                (
                    FieldKind::I64 {
                        indexed: false,
                        fast: false,
                    }
                    | FieldKind::U64 {
                        indexed: false,
                        fast: false,
                    },
                    _,
                ) => {
                    return Err(DeltaError::NonIndexedNumericField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
                (FieldKind::I64 { .. }, actual) => {
                    return Err(DeltaError::NumericTypeMismatch {
                        field_ord: field.id,
                        field_name: field.name,
                        expected: "i64",
                        actual: numeric_value_type_name(actual),
                    });
                }
                (FieldKind::U64 { .. }, actual) => {
                    return Err(DeltaError::NumericTypeMismatch {
                        field_ord: field.id,
                        field_name: field.name,
                        expected: "u64",
                        actual: numeric_value_type_name(actual),
                    });
                }
                _ => {
                    return Err(DeltaError::NonNumericField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
            }
        }

        for value in stored_values {
            let field_index = usize::from(value.field_ord);
            let Some(field) = self.schema.fields.get(field_index) else {
                return Err(DeltaError::UnknownValueField {
                    field_ord: value.field_ord,
                });
            };
            if field.id != value.field_ord {
                return Err(DeltaError::UnknownValueField {
                    field_ord: value.field_ord,
                });
            }
            if std::mem::replace(&mut seen_fields[field_index], true) {
                return Err(DeltaError::DuplicateValueField {
                    global_docid,
                    field_ord: value.field_ord,
                });
            }
            if !field.stored {
                return Err(DeltaError::NonStoredField {
                    field_ord: field.id,
                    field_name: field.name,
                });
            }
            if matches!(field.kind, FieldKind::I64 { .. } | FieldKind::U64 { .. })
                && value.bytes.len() != size_of::<u64>()
            {
                return Err(DeltaError::InvalidNumericBytes {
                    field_ord: field.id,
                    field_name: field.name,
                    expected: size_of::<u64>(),
                    actual: value.bytes.len(),
                });
            }
            if u32::try_from(value.bytes.len()).is_err() {
                return Err(DeltaError::StoredValueTooLarge {
                    field_ord: field.id,
                    bytes: value.bytes.len(),
                });
            }
        }
        for field in &self.stored_fields {
            let appended = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map_or_else(
                    || {
                        stored_values
                            .iter()
                            .find(|value| value.field_ord == field.field_ord)
                            .map_or(0, |value| value.bytes.len())
                    },
                    |_| size_of::<u64>(),
                );
            let final_bytes = field.payload_bytes.checked_add(appended);
            if final_bytes.is_none_or(|bytes| u32::try_from(bytes).is_err()) {
                return Err(DeltaError::StoredBlobTooLarge {
                    field_ord: field.field_ord,
                    current: field.payload_bytes,
                    appended,
                });
            }
        }
        Ok(())
    }

    fn resolve_document_values(
        &self,
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<ResolvedDocumentValues, DeltaError> {
        let mut resolved_numeric_values = Vec::new();
        reserve_vec(
            &mut resolved_numeric_values,
            self.numeric_fields.len(),
            "delta numeric value scratch",
        )?;
        for field in &self.numeric_fields {
            let value = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.value)
                .or_else(|| {
                    stored_values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .and_then(|value| {
                            numeric_value_from_le_bytes(
                                self.schema.fields[usize::from(field.field_ord)].kind,
                                value.bytes,
                            )
                        })
                });
            resolved_numeric_values.push(value);
        }

        let mut resolved_stored_values = Vec::new();
        reserve_vec(
            &mut resolved_stored_values,
            self.stored_fields.len(),
            "delta stored value scratch",
        )?;
        for field in &self.stored_fields {
            let numeric_bytes = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.value.to_le_bytes());
            let bytes = numeric_bytes
                .as_ref()
                .map(|bytes| bytes.as_slice())
                .or_else(|| {
                    stored_values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .map(|value| value.bytes)
                });
            let value = if let Some(bytes) = bytes {
                let mut owned = Vec::new();
                reserve_vec(&mut owned, bytes.len(), "delta stored value bytes")?;
                owned.extend_from_slice(bytes);
                Some(owned)
            } else {
                None
            };
            resolved_stored_values.push(value);
        }
        Ok((resolved_numeric_values, resolved_stored_values))
    }

    fn reserve_document(&mut self, term_count: usize) -> Result<(), DeltaError> {
        reserve_vec(&mut self.document_docids, 1, "delta document docids")?;
        reserve_vec(&mut self.document_ids, 1, "delta document ids")?;
        reserve_vec(
            &mut self.document_content_hashes,
            1,
            "delta document content hashes",
        )?;
        reserve_vec(
            &mut self.document_term_offsets,
            1,
            "delta document term offsets",
        )?;
        reserve_vec(
            &mut self.document_term_ids,
            term_count,
            "delta document term ids",
        )?;
        reserve_vec(&mut self.chains, term_count, "delta term chains")?;
        self.live_ids
            .try_reserve(1)
            .map_err(|_| DeltaError::Allocation {
                resource: "delta live identity overlay",
                additional: 1,
            })?;
        for column in &mut self.fieldnorms {
            reserve_vec(&mut column.raw_lengths, 1, "delta raw field lengths")?;
            reserve_vec(&mut column.fieldnorm_ids, 1, "delta fieldnorm bytes")?;
        }
        for column in &mut self.numeric_fields {
            reserve_vec(&mut column.values, 1, "delta numeric values")?;
        }
        for column in &mut self.stored_fields {
            reserve_vec(&mut column.values, 1, "delta stored values")?;
        }
        Ok(())
    }
}

fn numeric_value_type_name(value: NumericValue) -> &'static str {
    match value {
        NumericValue::I64(_) => "i64",
        NumericValue::U64(_) => "u64",
    }
}

fn numeric_value_from_le_bytes(kind: FieldKind, bytes: &[u8]) -> Option<NumericValue> {
    let bytes: [u8; 8] = bytes.try_into().ok()?;
    match kind {
        FieldKind::I64 { .. } => Some(NumericValue::I64(i64::from_le_bytes(bytes))),
        FieldKind::U64 { .. } => Some(NumericValue::U64(u64::from_le_bytes(bytes))),
        FieldKind::Keyword | FieldKind::Text { .. } | FieldKind::StoredOnly => None,
    }
}

pub(in crate::delta) fn validate_lease_base(lease_base: u64) -> Result<u64, DeltaError> {
    if !lease_base.is_multiple_of(u64::from(DOC_ORDS_PER_LEASE)) {
        return Err(DeltaError::MisalignedLeaseBase { lease_base });
    }
    let lease_end = lease_base.saturating_add(u64::from(DOC_ORDS_PER_LEASE));
    if lease_end > MAX_GLOBAL_DOCID_EXCLUSIVE {
        return Err(DeltaError::LeaseOutOfRange {
            lease_base,
            lease_end,
        });
    }
    Ok(lease_end)
}

pub(in crate::delta) fn reserve_vec<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), DeltaError> {
    values
        .try_reserve(additional)
        .map_err(|_| DeltaError::Allocation {
            resource,
            additional,
        })
}
