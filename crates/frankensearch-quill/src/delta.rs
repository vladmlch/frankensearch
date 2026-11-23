//! Lease-bounded mutable storage for Quill's searchable delta segment.
//!
//! This module owns mutable storage invariants and the immutable freeze
//! boundary. Composite publication lives in `index`, allocation-free cursor
//! adapters live in `argus`, and canonical FSLX sealing lives in `scribe`.

mod arena;
#[cfg(feature = "conformance-internals")]
mod conformance;
mod ingest;
mod snapshot;
mod types;
mod views;

#[cfg(test)]
mod tests;

use std::mem::size_of;
use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::AHashMap;
use frankensearch_core::DocId;

use crate::config::DEFAULT_DELTA_BUDGET_BYTES;
use crate::quiver::{NumericValue, POSTINGS_PER_BLOCK};
use crate::schema::{FieldKind, SchemaDescriptor};
use crate::scribe::{DOC_ORDS_PER_LEASE, TermInterner};

use arena::TypedChainArena;
use ingest::validate_lease_base;
pub use snapshot::DeltaSnapshot;
pub(crate) use snapshot::{DeltaOrderedTerms, DeltaStoredMaterialization};
pub use types::{
    DeltaApply, DeltaBlockMax, DeltaError, DeltaFieldNorm, DeltaMemoryStats, DeltaNumericValue,
    DeltaPosting, DeltaStoredValue, DeltaTermPosting,
};
use types::{FieldLayout, FieldNormColumn, NumericColumn, PostingRecord, StoredColumn, TermChain};
pub use views::{DeltaLiveDocuments, DeltaPositions, DeltaPostings, DeltaTerm};

/// First unrolled-chain block size.
pub const DELTA_CHAIN_INITIAL_CAPACITY: usize = 1;
/// Maximum unrolled-chain block size, matching a sealed posting block.
pub const DELTA_CHAIN_MAX_CAPACITY: usize = POSTINGS_PER_BLOCK;

const MAX_GLOBAL_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;
const TOMBSTONE_WORD_BITS: usize = u64::BITS as usize;
const TOMBSTONE_WORDS_PER_LEASE: usize = DOC_ORDS_PER_LEASE as usize / TOMBSTONE_WORD_BITS;
const HASH_SLOT_ESTIMATE: usize = 2 * size_of::<usize>();
const NEW_TERM_SENTINEL: u32 = u32::MAX;
static NEXT_DELTA_OWNER_ID: AtomicU64 = AtomicU64::new(1);

/// One mutable, lease-bounded delta generation.
#[derive(Debug)]
pub struct DeltaSegment {
    schema: SchemaDescriptor,
    lease_base: u64,
    lease_end: u64,
    next_docid_floor: u64,
    budget_bytes: usize,
    owner_id: u64,
    generation: u64,
    fields: Vec<FieldLayout>,
    terms: TermInterner,
    chains: Vec<TermChain>,
    posting_arena: TypedChainArena<PostingRecord>,
    position_arena: TypedChainArena<u32>,
    document_docids: Vec<u32>,
    document_ids: Vec<DocId>,
    document_content_hashes: Vec<Option<u64>>,
    document_term_offsets: Vec<u32>,
    document_term_ids: Vec<u32>,
    fieldnorms: Vec<FieldNormColumn>,
    numeric_fields: Vec<NumericColumn>,
    stored_fields: Vec<StoredColumn>,
    live_ids: AHashMap<DocId, u32>,
    tombstone_words: Vec<u64>,
    tombstone_count: usize,
    logical_bytes_used: usize,
}

impl DeltaSegment {
    /// Construct a delta with the pinned default 8 MiB budget.
    ///
    /// # Errors
    ///
    /// Rejects an invalid schema or lease range.
    pub fn with_default_budget(
        schema: SchemaDescriptor,
        lease_base: u64,
    ) -> Result<Self, DeltaError> {
        Self::new(schema, lease_base, DEFAULT_DELTA_BUDGET_BYTES)
    }

    /// Construct an empty delta for one aligned Q1 lease.
    ///
    /// # Errors
    ///
    /// Rejects zero budget, invalid schema, misalignment, or a lease extending
    /// past the global u32 posting domain.
    pub fn new(
        schema: SchemaDescriptor,
        lease_base: u64,
        budget_bytes: usize,
    ) -> Result<Self, DeltaError> {
        if budget_bytes == 0 {
            return Err(DeltaError::ZeroBudget);
        }
        schema
            .validate()
            .map_err(|error| DeltaError::InvalidSchema {
                detail: error.to_string(),
            })?;
        let lease_end = validate_lease_base(lease_base)?;
        let fields: Vec<FieldLayout> = schema
            .fields
            .iter()
            .filter_map(|field| match field.kind {
                FieldKind::Keyword => Some(FieldLayout {
                    field_ord: field.id,
                    positions: false,
                }),
                FieldKind::Text { positions, .. } => Some(FieldLayout {
                    field_ord: field.id,
                    positions,
                }),
                FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => None,
            })
            .collect();
        let fieldnorms = fields
            .iter()
            .map(|_| FieldNormColumn {
                raw_lengths: Vec::new(),
                fieldnorm_ids: Vec::new(),
            })
            .collect();
        let numeric_fields = schema
            .fields
            .iter()
            .filter_map(|field| match field.kind {
                kind if kind.has_numeric_column() => Some(NumericColumn {
                    field_ord: field.id,
                    values: Vec::new(),
                }),
                _ => None,
            })
            .collect();
        let stored_fields = schema
            .fields
            .iter()
            .filter(|field| field.stored)
            .map(|field| StoredColumn {
                field_ord: field.id,
                values: Vec::new(),
                payload_bytes: 0,
            })
            .collect();
        let mut tombstone_words = Vec::new();
        tombstone_words
            .try_reserve_exact(TOMBSTONE_WORDS_PER_LEASE)
            .map_err(|_| DeltaError::Allocation {
                resource: "delta tombstone bitmap",
                additional: TOMBSTONE_WORDS_PER_LEASE,
            })?;
        Ok(Self {
            schema,
            lease_base,
            lease_end,
            next_docid_floor: lease_base,
            budget_bytes,
            owner_id: NEXT_DELTA_OWNER_ID.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            fields,
            terms: TermInterner::new(),
            chains: Vec::new(),
            posting_arena: TypedChainArena::new("delta posting arena"),
            position_arena: TypedChainArena::new("delta position arena"),
            document_docids: Vec::new(),
            document_ids: Vec::new(),
            document_content_hashes: Vec::new(),
            document_term_offsets: vec![0],
            document_term_ids: Vec::new(),
            fieldnorms,
            numeric_fields,
            stored_fields,
            live_ids: AHashMap::new(),
            tombstone_words,
            tombstone_count: 0,
            logical_bytes_used: 0,
        })
    }

    /// Compile-time schema carried by this generation.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.schema
    }

    /// Inclusive global lease base.
    #[must_use]
    pub const fn lease_base(&self) -> u64 {
        self.lease_base
    }

    /// Exclusive global lease end.
    #[must_use]
    pub const fn lease_end(&self) -> u64 {
        self.lease_end
    }

    /// First global docid admissible after any same-lease reset.
    #[must_use]
    pub const fn next_docid_floor(&self) -> u64 {
        self.next_docid_floor
    }

    /// Configured logical-byte seal threshold.
    #[must_use]
    pub const fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Freeze the current generation for lock-free epoch publication.
    ///
    /// The returned snapshot owns independent arenas and sidecars. Subsequent
    /// writes or resets of this mutable segment cannot change the snapshot,
    /// and owner-bound posting handles cannot cross the freeze boundary.
    /// `keeper_generation` binds the epoch to the MANIFEST view current at the
    /// batch boundary; composite publication rejects a stale witness.
    #[must_use]
    pub fn freeze(&self, keeper_generation: u64) -> DeltaSnapshot {
        let snapshot_owner_id = NEXT_DELTA_OWNER_ID.fetch_add(1, Ordering::Relaxed);
        let segment = Self {
            schema: self.schema,
            lease_base: self.lease_base,
            lease_end: self.lease_end,
            next_docid_floor: self.next_docid_floor,
            budget_bytes: self.budget_bytes,
            owner_id: snapshot_owner_id,
            generation: self.generation,
            fields: self.fields.clone(),
            terms: self.terms.clone(),
            chains: self.chains.clone(),
            posting_arena: self.posting_arena.snapshot_copy(),
            position_arena: self.position_arena.snapshot_copy(),
            document_docids: self.document_docids.clone(),
            document_ids: self.document_ids.clone(),
            document_content_hashes: self.document_content_hashes.clone(),
            document_term_offsets: self.document_term_offsets.clone(),
            document_term_ids: self.document_term_ids.clone(),
            fieldnorms: self.fieldnorms.clone(),
            numeric_fields: self.numeric_fields.clone(),
            stored_fields: self.stored_fields.clone(),
            live_ids: self.live_ids.clone(),
            tombstone_words: self.tombstone_words.clone(),
            tombstone_count: self.tombstone_count,
            logical_bytes_used: self.logical_bytes_used,
        };
        // No publication pass: the live counts ride along inside the cloned
        // chains, already exact.
        DeltaSnapshot {
            segment,
            ordered_term_ids: OnceLock::new(),
            keeper_generation,
            // Each frozen epoch is a distinct publication candidate, even
            // when two freezes observe the same mutable Delta generation.
            // Rebinding deliberately preserves this witness; a later freeze
            // must never be replaceable by an earlier, shorter snapshot.
            lineage_id: snapshot_owner_id,
            #[cfg(test)]
            materialization_attempts: AtomicUsize::new(0),
        }
    }
}

impl DeltaSegment {
    /// Probe the live delta overlay. A miss directs the caller to sealed IDHASH.
    #[must_use]
    pub fn probe_id(&self, document_id: &str) -> Option<u32> {
        self.live_ids.get(document_id).copied()
    }

    /// Delete a live delta identity. A repeated or sealed-only delete is a miss.
    pub fn delete_delta_id(&mut self, document_id: &str) -> Option<u32> {
        let (removed_id, global_docid) = self.live_ids.remove_entry(document_id)?;
        let identity_bytes =
            size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE + removed_id.len();
        self.logical_bytes_used = self
            .logical_bytes_used
            .checked_sub(identity_bytes)
            .expect("live identity bytes are accounted");
        let tombstone_bytes = self.mark_tombstone(global_docid);
        self.logical_bytes_used = self.logical_bytes_used.saturating_add(tombstone_bytes);
        Some(global_docid)
    }

    /// Find a composite term in O(1) expected time with exact collision checks.
    #[must_use]
    pub fn find_term(&self, field_ord: u16, term: &[u8]) -> Option<DeltaTerm<'_>> {
        Some(DeltaTerm {
            delta: self,
            term_index: self.terms.find(field_ord, term)?,
        })
    }

    /// Term views in canonical `(field_ord BE, term bytes)` order.
    #[must_use]
    pub fn sorted_terms(&self) -> Vec<DeltaTerm<'_>> {
        self.terms
            .sorted_ids()
            .into_iter()
            .map(|term_index| DeltaTerm {
                delta: self,
                term_index,
            })
            .collect()
    }

    /// Number of distinct `(field, term)` keys retained by this generation.
    ///
    /// Unlike [`Self::sorted_terms`], this is O(1) and does not allocate a
    /// temporary ordered view. Callers that need only a dictionary-work
    /// ceiling must use this count rather than materializing every term.
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Physical fieldnorm, including a tombstoned row.
    #[must_use]
    pub fn fieldnorm_id(&self, field_ord: u16, global_docid: u32) -> Option<u8> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        let field = self.field_index(field_ord)?;
        self.fieldnorms[field].fieldnorm_ids.get(row).copied()
    }

    /// Physical content hash, including a tombstoned row.
    #[must_use]
    pub fn content_hash(&self, global_docid: u32) -> Option<u64> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        self.document_content_hashes.get(row).copied().flatten()
    }

    /// Physical indexed-or-fast numeric value, including a tombstoned row.
    #[must_use]
    pub fn numeric_value(&self, field_ord: u16, global_docid: u32) -> Option<NumericValue> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        let field = self.numeric_field_index(field_ord)?;
        self.numeric_fields[field]
            .values
            .get(row)
            .copied()
            .flatten()
    }

    /// Physical stored bytes, including a tombstoned row.
    #[must_use]
    pub fn stored_value(&self, field_ord: u16, global_docid: u32) -> Option<&[u8]> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        let field = self.stored_field_index(field_ord)?;
        self.stored_fields[field]
            .values
            .get(row)
            .and_then(Option::as_deref)
    }

    /// Exact raw length for snapshot-level BM25 aggregation.
    #[must_use]
    pub fn raw_field_length(&self, field_ord: u16, global_docid: u32) -> Option<u32> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        let field = self.field_index(field_ord)?;
        self.fieldnorms[field].raw_lengths.get(row).copied()
    }

    /// Exact live token numerator for an indexed string field.
    #[must_use]
    pub fn live_total_tokens(&self, field_ord: u16) -> Option<u64> {
        let field = self.field_index(field_ord)?;
        Some(
            self.document_docids
                .iter()
                .zip(&self.fieldnorms[field].raw_lengths)
                .filter(|(docid, _)| !self.is_tombstoned(**docid))
                .map(|(_, length)| u64::from(*length))
                .sum(),
        )
    }

    /// Whether a physical delta row has been superseded or deleted.
    #[must_use]
    pub fn is_tombstoned(&self, global_docid: u32) -> bool {
        let Some(ordinal) = self.lease_ordinal(global_docid) else {
            return false;
        };
        let word = ordinal / TOMBSTONE_WORD_BITS;
        let bit = ordinal % TOMBSTONE_WORD_BITS;
        self.tombstone_words
            .get(word)
            .is_some_and(|value| value & (1_u64 << bit) != 0)
    }

    #[must_use]
    pub fn physical_document_count(&self) -> usize {
        self.document_docids.len()
    }

    #[must_use]
    pub fn live_document_count(&self) -> usize {
        self.document_docids
            .len()
            .saturating_sub(self.tombstone_count)
    }

    /// Ordered live rows for snapshot composition and sealing.
    #[must_use]
    pub fn live_documents(&self) -> DeltaLiveDocuments<'_> {
        DeltaLiveDocuments {
            delta: self,
            index: 0,
        }
    }

    /// Logical current-generation bytes used by the seal trigger.
    ///
    /// Active unrolled slots (including last-block slack) are counted;
    /// reset-retained capacity is not.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.logical_bytes_used
    }

    #[cfg(test)]
    fn recompute_bytes_used(&self) -> usize {
        let identity_bytes = self
            .live_ids
            .keys()
            .map(|id| size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE + id.len())
            .sum::<usize>();
        let document_id_bytes = self
            .document_ids
            .iter()
            .map(|id| size_of::<DocId>() + id.len())
            .sum::<usize>();
        let fieldnorm_bytes = self
            .fieldnorms
            .iter()
            .map(|column| {
                column
                    .raw_lengths
                    .len()
                    .saturating_mul(size_of::<u32>())
                    .saturating_add(column.fieldnorm_ids.len())
            })
            .sum::<usize>();
        let numeric_value_bytes = self
            .numeric_fields
            .iter()
            .map(|column| {
                column
                    .values
                    .len()
                    .saturating_mul(size_of::<Option<NumericValue>>())
            })
            .sum::<usize>();
        let stored_value_bytes = self
            .stored_fields
            .iter()
            .map(|column| {
                column
                    .values
                    .len()
                    .saturating_mul(size_of::<Option<Vec<u8>>>())
                    .saturating_add(column.values.iter().flatten().map(Vec::len).sum::<usize>())
            })
            .sum::<usize>();
        self.terms
            .bytes_used()
            .saturating_add(self.chains.len().saturating_mul(size_of::<TermChain>()))
            .saturating_add(self.posting_arena.bytes_used())
            .saturating_add(self.position_arena.bytes_used())
            .saturating_add(self.document_docids.len().saturating_mul(size_of::<u32>()))
            .saturating_add(document_id_bytes)
            .saturating_add(
                self.document_content_hashes
                    .len()
                    .saturating_mul(size_of::<Option<u64>>()),
            )
            .saturating_add(
                self.document_term_offsets
                    .len()
                    .saturating_sub(1)
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.document_term_ids
                    .len()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(fieldnorm_bytes)
            .saturating_add(numeric_value_bytes)
            .saturating_add(stored_value_bytes)
            .saturating_add(identity_bytes)
            .saturating_add(self.tombstone_words.len().saturating_mul(size_of::<u64>()))
    }

    /// Complete retained allocation for reset/RSS diagnostics.
    #[must_use]
    pub fn bytes_reserved(&self) -> usize {
        let identity_bytes = self
            .live_ids
            .keys()
            .map(|id| id.len())
            .sum::<usize>()
            .saturating_add(
                self.live_ids
                    .capacity()
                    .saturating_mul(size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE),
            );
        let document_id_bytes = self
            .document_ids
            .iter()
            .map(|id| id.len())
            .sum::<usize>()
            .saturating_add(
                self.document_ids
                    .capacity()
                    .saturating_mul(size_of::<DocId>()),
            );
        let fieldnorm_bytes = self
            .fieldnorms
            .iter()
            .map(|column| {
                column
                    .raw_lengths
                    .capacity()
                    .saturating_mul(size_of::<u32>())
                    .saturating_add(column.fieldnorm_ids.capacity())
            })
            .sum::<usize>();
        let numeric_value_bytes = self
            .numeric_fields
            .capacity()
            .saturating_mul(size_of::<NumericColumn>())
            .saturating_add(
                self.numeric_fields
                    .iter()
                    .map(|column| {
                        column
                            .values
                            .capacity()
                            .saturating_mul(size_of::<Option<NumericValue>>())
                    })
                    .sum::<usize>(),
            );
        let stored_value_bytes = self
            .stored_fields
            .capacity()
            .saturating_mul(size_of::<StoredColumn>())
            .saturating_add(
                self.stored_fields
                    .iter()
                    .map(|column| {
                        column
                            .values
                            .capacity()
                            .saturating_mul(size_of::<Option<Vec<u8>>>())
                            .saturating_add(
                                column
                                    .values
                                    .iter()
                                    .flatten()
                                    .map(Vec::capacity)
                                    .sum::<usize>(),
                            )
                    })
                    .sum::<usize>(),
            );
        self.fields
            .capacity()
            .saturating_mul(size_of::<FieldLayout>())
            .saturating_add(self.terms.bytes_reserved())
            .saturating_add(
                self.chains
                    .capacity()
                    .saturating_mul(size_of::<TermChain>()),
            )
            .saturating_add(self.posting_arena.bytes_reserved())
            .saturating_add(self.position_arena.bytes_reserved())
            .saturating_add(
                self.document_docids
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(document_id_bytes)
            .saturating_add(
                self.document_content_hashes
                    .capacity()
                    .saturating_mul(size_of::<Option<u64>>()),
            )
            .saturating_add(
                self.document_term_offsets
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.document_term_ids
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.fieldnorms
                    .capacity()
                    .saturating_mul(size_of::<FieldNormColumn>()),
            )
            .saturating_add(fieldnorm_bytes)
            .saturating_add(numeric_value_bytes)
            .saturating_add(stored_value_bytes)
            .saturating_add(identity_bytes)
            .saturating_add(
                self.tombstone_words
                    .capacity()
                    .saturating_mul(size_of::<u64>()),
            )
    }

    /// Whether a nonempty generation reached its budget.
    #[must_use]
    pub fn should_seal(&self) -> bool {
        !self.document_docids.is_empty() && self.bytes_used() >= self.budget_bytes
    }

    /// Snapshot stable memory/cardinality diagnostics.
    #[must_use]
    pub fn memory_stats(&self) -> DeltaMemoryStats {
        DeltaMemoryStats {
            bytes_used: self.bytes_used(),
            bytes_reserved: self.bytes_reserved(),
            term_count: self.terms.len(),
            physical_document_count: self.physical_document_count(),
            live_document_count: self.live_document_count(),
            posting_blocks: self.posting_arena.active_blocks,
            position_blocks: self.position_arena.active_blocks,
        }
    }

    /// Clear a sealed generation while retaining writer allocations for the
    /// next aligned lease. Rust's borrow rules prevent reset while a term view
    /// lives; published [`DeltaSnapshot`] epochs own independent compact copies.
    ///
    /// # Errors
    ///
    /// Rejects an invalid next lease before changing state.
    pub fn reset_after_seal(&mut self, next_lease_base: u64) -> Result<(), DeltaError> {
        let next_lease_end = validate_lease_base(next_lease_base)?;
        if next_lease_base < self.lease_base {
            return Err(DeltaError::LeaseRegression {
                current_lease_base: self.lease_base,
                next_lease_base,
            });
        }
        let next_docid_floor = if next_lease_base == self.lease_base {
            self.document_docids
                .last()
                .map_or(self.next_docid_floor, |docid| u64::from(*docid) + 1)
        } else {
            next_lease_base
        };
        self.lease_base = next_lease_base;
        self.lease_end = next_lease_end;
        self.next_docid_floor = next_docid_floor;
        self.generation = self.generation.wrapping_add(1);
        self.terms.reset();
        self.chains.clear();
        self.posting_arena.reset();
        self.position_arena.reset();
        self.document_docids.clear();
        self.document_ids.clear();
        self.document_content_hashes.clear();
        self.document_term_offsets.clear();
        self.document_term_offsets.push(0);
        self.document_term_ids.clear();
        for column in &mut self.fieldnorms {
            column.raw_lengths.clear();
            column.fieldnorm_ids.clear();
        }
        for column in &mut self.numeric_fields {
            column.values.clear();
        }
        for column in &mut self.stored_fields {
            column.values.clear();
            column.payload_bytes = 0;
        }
        self.live_ids.clear();
        self.tombstone_words.fill(0);
        self.tombstone_words.clear();
        self.tombstone_count = 0;
        self.logical_bytes_used = 0;
        Ok(())
    }

    fn field_index(&self, field_ord: u16) -> Option<usize> {
        self.fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()
    }

    fn numeric_field_index(&self, field_ord: u16) -> Option<usize> {
        self.numeric_fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()
    }

    fn stored_field_index(&self, field_ord: u16) -> Option<usize> {
        self.stored_fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()
    }

    fn lease_ordinal(&self, global_docid: u32) -> Option<usize> {
        let ordinal = u64::from(global_docid).checked_sub(self.lease_base)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        (ordinal < DOC_ORDS_PER_LEASE as usize).then_some(ordinal)
    }

    /// Drop one newly tombstoned document out of its terms' live counts.
    ///
    /// The document's term list is recorded at apply time, so this touches
    /// exactly the chains that hold a row for it. A document that never
    /// landed — or whose generation was already cleared — contributes no
    /// terms and therefore no decrements.
    fn retire_document_postings(&mut self, global_docid: u32) {
        let Ok(document_index) = self.document_docids.binary_search(&global_docid) else {
            return;
        };
        let Some(&start) = self.document_term_offsets.get(document_index) else {
            return;
        };
        let Some(&end) = self.document_term_offsets.get(document_index + 1) else {
            return;
        };
        let (Ok(start), Ok(end)) = (usize::try_from(start), usize::try_from(end)) else {
            return;
        };
        let Some(term_count) = end.checked_sub(start) else {
            return;
        };
        for offset in 0..term_count {
            // Copy one ID out before taking the disjoint mutable chain
            // borrow. Cloning the whole slice here used transient storage
            // proportional to document term count on every replacement and
            // deletion.
            let Some(term_slot) = start.checked_add(offset) else {
                return;
            };
            let Some(term_index) = self.document_term_ids.get(term_slot).copied() else {
                return;
            };
            let Ok(term_offset) = usize::try_from(term_index) else {
                continue;
            };
            if let Some(chain) = self.chains.get_mut(term_offset) {
                // A count that would go negative means a posting was retired
                // twice, which the exactly-once caller forbids.
                chain.live_len = chain
                    .live_len
                    .checked_sub(1)
                    .expect("a live posting is retired at most once");
            }
        }
    }

    fn mark_tombstone(&mut self, global_docid: u32) -> usize {
        let ordinal = self
            .lease_ordinal(global_docid)
            .expect("live delta identity belongs to this lease");
        let word = ordinal / TOMBSTONE_WORD_BITS;
        let bit = ordinal % TOMBSTONE_WORD_BITS;
        let prior_words = self.tombstone_words.len();
        if prior_words <= word {
            self.tombstone_words.resize(word + 1, 0);
        }
        let mask = 1_u64 << bit;
        if self.tombstone_words[word] & mask == 0 {
            self.tombstone_words[word] |= mask;
            self.tombstone_count += 1;
            // Exactly-once by construction: this branch is the transition
            // from live to tombstoned, so a document superseded and later
            // deleted cannot be counted off twice. Only the terms this
            // document actually contributed are touched, which is a walk of
            // its own term list rather than of any chain.
            self.retire_document_postings(global_docid);
        }
        self.tombstone_words
            .len()
            .saturating_sub(prior_words)
            .saturating_mul(size_of::<u64>())
    }
}
