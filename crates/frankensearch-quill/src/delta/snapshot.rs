use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use frankensearch_core::DocId;
#[cfg(feature = "conformance-internals")]
use sha2::{Digest, Sha256};

#[cfg(feature = "conformance-internals")]
use crate::error::QuillError;
use crate::quiver::NumericValue;
use crate::schema::SchemaDescriptor;

use super::DeltaSegment;
#[cfg(feature = "conformance-internals")]
use super::conformance::{
    conformance_hash_bytes, conformance_hash_delta_documents, conformance_hash_delta_fields,
    conformance_hash_delta_terms, conformance_hash_delta_values, conformance_hash_delta_visibility,
    conformance_hash_usize,
};
use super::views::{DeltaLiveDocuments, DeltaTerm};

/// Immutable, owner-isolated copy of one published delta generation.
///
/// Publication deliberately deep-copies the mutable shard state. Readers can
/// then retain this value behind an [`std::sync::Arc`] without observing later
/// appends, tombstones, or arena reuse by the writer. The copy receives a fresh
/// owner identity so a posting handle from either side cannot resolve positions
/// through the other side's arenas.
#[derive(Debug)]
pub struct DeltaSnapshot {
    pub(in crate::delta) segment: DeltaSegment,
    pub(in crate::delta) ordered_term_ids: OnceLock<Box<[u32]>>,
    pub(in crate::delta) keeper_generation: u64,
    pub(in crate::delta) lineage_id: u64,
    #[cfg(test)]
    pub(in crate::delta) materialization_attempts: AtomicUsize,
}

/// Stored-value materialization outcome for one selected Delta row.
pub enum DeltaStoredMaterialization<'a> {
    /// The row is absent or tombstoned in this Delta.
    NotLive,
    /// The row is live; its requested stored field may still be absent.
    Live(Option<&'a [u8]>),
}

impl DeltaSnapshot {
    /// Read-only view of the frozen delta generation.
    #[must_use]
    pub const fn segment(&self) -> &DeltaSegment {
        &self.segment
    }

    /// Canonically ordered term views for this immutable generation.
    ///
    /// Query lowering must pre-admit the first view before calling this; the
    /// cache materializes the complete retained ID slice in one operation.
    #[must_use]
    pub(crate) fn ordered_terms(&self) -> DeltaOrderedTerms<'_> {
        let term_ids = self.cached_ordered_term_ids();
        DeltaOrderedTerms {
            delta: &self.segment,
            term_ids,
            next_term: 0,
        }
    }

    /// Canonically ordered terms for one field in this immutable generation.
    ///
    /// The cached composite order begins with big-endian field bytes, so the
    /// selected field is one contiguous subslice of the single retained ID
    /// buffer. Query lowering must pre-admit the first view before calling
    /// this method.
    #[must_use]
    pub(crate) fn ordered_terms_for_field(&self, field_ord: u16) -> DeltaOrderedTerms<'_> {
        let term_ids = self.cached_ordered_term_ids();
        let first = term_ids
            .partition_point(|term_id| self.segment.terms.field_and_term(*term_id).0 < field_ord);
        let past_last = term_ids
            .partition_point(|term_id| self.segment.terms.field_and_term(*term_id).0 <= field_ord);
        DeltaOrderedTerms {
            delta: &self.segment,
            term_ids: &term_ids[first..past_last],
            next_term: 0,
        }
    }

    fn cached_ordered_term_ids(&self) -> &[u32] {
        self.ordered_term_ids
            .get_or_init(|| self.segment.terms.sorted_ids().into_boxed_slice())
    }

    /// Number of physical rows retained by this immutable delta generation.
    ///
    /// Tombstoned rows remain physical work for posting scans and therefore
    /// remain part of admission and fuel accounting even though they are not
    /// included in [`Self::live_document_count`].
    #[must_use]
    pub fn physical_document_count(&self) -> usize {
        self.segment.physical_document_count()
    }

    /// Keeper MANIFEST generation this delta epoch was derived from.
    #[must_use]
    pub const fn keeper_generation(&self) -> u64 {
        self.keeper_generation
    }

    /// Clone this exact frozen epoch against a successor Keeper generation.
    ///
    /// Sealing one shard advances the MANIFEST generation for every shard.
    /// Surviving immutable epochs use this operation to retain their private
    /// lineage witness while receiving owner-isolated arena storage for the
    /// successor composite publication.
    #[must_use]
    pub fn rebind_keeper_generation(&self, keeper_generation: u64) -> Self {
        let mut rebound = self.segment.freeze(keeper_generation);
        rebound.lineage_id = self.lineage_id;
        rebound
    }

    /// Private shard/epoch/lease identity retained across Keeper rebinding.
    pub(crate) const fn publication_lineage(&self) -> (u64, u64, u64, u64) {
        (
            self.lineage_id,
            self.segment.generation,
            self.segment.lease_base,
            self.segment.lease_end,
        )
    }

    /// Canonical conformance witness for this immutable publication candidate.
    ///
    /// The digest binds every search-visible value, every physical row and
    /// tombstone witness, and the publication lineage. It deliberately omits
    /// only `owner_id`: freezing or Keeper rebinding must give arenas a fresh
    /// owner while retaining equal content. This work is absent from shipping
    /// builds.
    ///
    /// # Errors
    ///
    /// Returns a schema error if the frozen descriptor cannot be encoded.
    #[cfg(feature = "conformance-internals")]
    pub(crate) fn conformance_content_sha256(&self) -> Result<[u8; 32], QuillError> {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch/quill/delta-snapshot-content/v1\0");
        conformance_hash_bytes(&mut hasher, &self.segment.schema.canonical_encoding()?);
        hasher.update(self.keeper_generation.to_be_bytes());
        hasher.update(self.lineage_id.to_be_bytes());
        hasher.update(self.segment.generation.to_be_bytes());
        hasher.update(self.segment.lease_base.to_be_bytes());
        hasher.update(self.segment.lease_end.to_be_bytes());
        hasher.update(self.segment.next_docid_floor.to_be_bytes());
        conformance_hash_usize(&mut hasher, self.segment.budget_bytes);
        conformance_hash_delta_fields(&mut hasher, &self.segment);
        conformance_hash_delta_documents(&mut hasher, &self.segment);
        conformance_hash_delta_terms(&mut hasher, &self.segment);
        conformance_hash_delta_values(&mut hasher, &self.segment);
        conformance_hash_delta_visibility(&mut hasher, &self.segment);
        Ok(hasher.finalize().into())
    }

    /// Compile-time schema carried by this generation.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.segment.schema()
    }

    /// Inclusive global lease base.
    #[must_use]
    pub const fn lease_base(&self) -> u64 {
        self.segment.lease_base()
    }

    /// Exclusive global lease end.
    #[must_use]
    pub const fn lease_end(&self) -> u64 {
        self.segment.lease_end()
    }

    /// Number of visible rows after applying delta-local tombstones.
    #[must_use]
    pub fn live_document_count(&self) -> usize {
        self.segment.live_document_count()
    }

    /// Whether one global document id names a visible row in this generation.
    #[must_use]
    pub fn is_live_document(&self, global_docid: u32) -> bool {
        self.segment
            .document_docids
            .binary_search(&global_docid)
            .is_ok()
            && !self.segment.is_tombstoned(global_docid)
    }

    /// Materialize one visible row's stable external identifier.
    #[must_use]
    pub fn materialize_document_id(&self, global_docid: u32) -> Option<DocId> {
        #[cfg(test)]
        self.record_materialization_attempt();
        let row = self
            .segment
            .document_docids
            .binary_search(&global_docid)
            .ok()?;
        if self.segment.is_tombstoned(global_docid) {
            return None;
        }
        self.segment.document_ids.get(row).cloned()
    }

    /// Exact live token numerator for one indexed string field.
    #[must_use]
    pub fn live_total_tokens(&self, field_ord: u16) -> Option<u64> {
        self.segment.live_total_tokens(field_ord)
    }

    /// Physical fieldnorm byte for one row in this frozen generation.
    #[must_use]
    pub fn fieldnorm_id(&self, field_ord: u16, global_docid: u32) -> Option<u8> {
        self.segment.fieldnorm_id(field_ord, global_docid)
    }

    /// Physical content hash, including a tombstoned row.
    #[must_use]
    pub fn content_hash(&self, global_docid: u32) -> Option<u64> {
        self.segment.content_hash(global_docid)
    }

    /// Physical indexed-or-fast numeric value, including a tombstoned row.
    #[must_use]
    pub fn numeric_value(&self, field_ord: u16, global_docid: u32) -> Option<NumericValue> {
        self.segment.numeric_value(field_ord, global_docid)
    }

    /// Physical stored bytes, including a tombstoned row.
    #[must_use]
    pub fn stored_value(&self, field_ord: u16, global_docid: u32) -> Option<&[u8]> {
        self.segment.stored_value(field_ord, global_docid)
    }

    /// Materialize stored bytes only when this snapshot owns a live row.
    ///
    /// The outcome distinguishes a missing or tombstoned row from a live row
    /// whose requested stored field is absent.
    #[must_use]
    pub(crate) fn materialize_stored_value(
        &self,
        field_ord: u16,
        global_docid: u32,
    ) -> DeltaStoredMaterialization<'_> {
        #[cfg(test)]
        self.record_materialization_attempt();
        if !self.is_live_document(global_docid) {
            return DeltaStoredMaterialization::NotLive;
        }
        DeltaStoredMaterialization::Live(self.segment.stored_value(field_ord, global_docid))
    }

    #[cfg(test)]
    pub(crate) fn reset_materialization_attempts(&self) {
        self.materialization_attempts.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn materialization_attempts(&self) -> usize {
        self.materialization_attempts.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn record_materialization_attempt(&self) {
        self.materialization_attempts
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Live-only fieldnorm byte for one visible row in this generation.
    #[must_use]
    pub fn live_fieldnorm_id(&self, field_ord: u16, global_docid: u32) -> Option<u8> {
        if self.segment.is_tombstoned(global_docid) {
            return None;
        }
        self.segment.fieldnorm_id(field_ord, global_docid)
    }

    /// Look up one immutable composite `(field, term)` key.
    #[must_use]
    pub fn find_term(&self, field_ord: u16, term: &[u8]) -> Option<DeltaTerm<'_>> {
        self.segment.find_term(field_ord, term)
    }

    /// Ordered live rows for snapshot composition and later cursor lowering.
    #[must_use]
    pub fn live_documents(&self) -> DeltaLiveDocuments<'_> {
        self.segment.live_documents()
    }

    /// Covering Q1 interval of every physically allocated row in this epoch.
    ///
    /// Tombstoned rows remain part of the interval because their docids were
    /// allocated and therefore cannot collide with a sealed Keeper range.
    pub(crate) fn occupied_docid_range(&self) -> Option<(u64, u64)> {
        let first = u64::from(*self.segment.document_docids.first()?);
        let last = u64::from(*self.segment.document_docids.last()?);
        Some((first, last + 1))
    }
}

/// Allocation-free traversal of one frozen Delta generation's canonical order.
#[derive(Debug, Clone)]
pub struct DeltaOrderedTerms<'a> {
    delta: &'a DeltaSegment,
    pub(in crate::delta) term_ids: &'a [u32],
    next_term: usize,
}

impl<'a> Iterator for DeltaOrderedTerms<'a> {
    type Item = DeltaTerm<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let term_index = *self.term_ids.get(self.next_term)?;
        self.next_term += 1;
        Some(DeltaTerm {
            delta: self.delta,
            term_index,
        })
    }
}

impl ExactSizeIterator for DeltaOrderedTerms<'_> {
    fn len(&self) -> usize {
        self.term_ids.len() - self.next_term
    }
}

impl crate::argus::LiveDocs for DeltaSnapshot {
    fn is_live(&self, global_docid: u32) -> bool {
        self.is_live_document(global_docid)
    }
}
