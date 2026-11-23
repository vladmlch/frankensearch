use crate::delta::{DeltaSnapshot, DeltaStoredMaterialization};
use crate::keeper::{KeeperSnapshot, LiveDocumentFloor, Manifest, validate_manifest_successor};
use crate::quiver::SnapshotFieldStats;
use crate::schema::{FieldKind, SchemaDescriptor};
use crate::segment::SectionKind;
use arc_swap::ArcSwap;
use frankensearch_core::DocId;
#[cfg(test)]
use std::cell::Cell;
use std::sync::Arc;
use thiserror::Error;

use super::error_results::{METADATA_FIELD, QuillDocumentWitness, QuillIndexError, invalid_state};
use super::query_cache::RankedQueryCache;
use super::query_cursors::{required_section, snapshot_doc_freq};

/// Typed failure from process-local Keeper plus Delta snapshot composition.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SnapshotError {
    /// A Delta epoch was frozen against a different schema.
    #[error("delta schema {actual} does not match Keeper schema {expected}")]
    SchemaMismatch {
        /// Keeper schema name.
        expected: &'static str,
        /// Delta schema name.
        actual: &'static str,
    },
    /// A Delta epoch was frozen against a different Keeper generation.
    #[error(
        "delta lease starting at {lease_base} belongs to Keeper generation {actual}, expected {expected}"
    )]
    KeeperGenerationMismatch {
        /// Generation carried by the composite Keeper snapshot.
        expected: u64,
        /// Generation witnessed when the Delta epoch was frozen.
        actual: u64,
        /// Delta lease identifying the stale epoch.
        lease_base: u64,
    },
    /// A Delta epoch's physical covering interval intersects a sealed range.
    #[error(
        "delta occupied range [{delta_lo}, {delta_hi}) overlaps Keeper segment {segment_id:#018x} range [{keeper_lo}, {keeper_hi})"
    )]
    KeeperDeltaDocidOverlap {
        /// Inclusive first physically allocated Delta docid.
        delta_lo: u64,
        /// Exclusive Delta covering upper bound.
        delta_hi: u64,
        /// Conflicting sealed segment identity.
        segment_id: u64,
        /// Inclusive sealed covering lower bound.
        keeper_lo: u64,
        /// Exclusive sealed covering upper bound.
        keeper_hi: u64,
    },
    /// Two independently published shards claimed overlapping Q1 leases.
    #[error("delta leases [{first_base}, {first_end}) and [{second_base}, {second_end}) overlap")]
    OverlappingDeltaLeases {
        /// First inclusive lease base.
        first_base: u64,
        /// First exclusive lease end.
        first_end: u64,
        /// Second inclusive lease base.
        second_base: u64,
        /// Second exclusive lease end.
        second_end: u64,
    },
    /// A seal replacement table omitted one independently published shard.
    #[error("delta seal replacement table omitted the epoch in lease [{lease_base}, {lease_end})")]
    MissingDeltaRebind {
        /// Inclusive lease base of the disappearing process-local epoch.
        lease_base: u64,
        /// Exclusive lease end of the disappearing process-local epoch.
        lease_end: u64,
    },
    /// A complete publication attempted to replace a newer Keeper generation.
    #[error("Keeper generation regressed from current {current} to proposed {proposed}")]
    KeeperGenerationRegression {
        /// Currently published durable generation.
        current: u64,
        /// Stale proposed generation.
        proposed: u64,
    },
    /// One generation number named two different durable MANIFEST images.
    #[error("Keeper generation {generation} identifies two different MANIFEST images")]
    KeeperGenerationCollision {
        /// Reused generation number.
        generation: u64,
    },
    /// A forward complete publication was not a valid Keeper successor.
    #[error("invalid Keeper snapshot transition: {detail}")]
    KeeperTransition {
        /// Underlying Keeper transition rejection.
        detail: String,
    },
    /// A non-genesis Keeper snapshot omitted an indexed field's statistics.
    #[error("Keeper snapshot has no field statistics for indexed field {field_ord}")]
    MissingKeeperFieldStats {
        /// Missing schema field ordinal.
        field_ord: u16,
    },
    /// A schema-compatible Delta failed to expose one indexed field's tokens.
    #[error("delta snapshot has no live token statistics for indexed field {field_ord}")]
    MissingDeltaFieldStats {
        /// Missing schema field ordinal.
        field_ord: u16,
    },
    /// One checked composite counter exceeded its u64 representation.
    #[error("composite snapshot {counter} overflowed")]
    CounterOverflow {
        /// Stable name of the counter being aggregated.
        counter: &'static str,
    },
    /// A per-shard publication named no current Delta slot.
    #[error("delta shard {shard} is outside the published shard count {shard_count}")]
    ShardOutOfBounds {
        /// Requested zero-based shard slot.
        shard: usize,
        /// Current number of published shard slots.
        shard_count: usize,
    },
    /// The process-local publication epoch cannot advance further.
    #[error("process-local snapshot epoch exhausted")]
    EpochExhausted,
    /// A bounded publication sidecar could not be allocated.
    #[error("unable to allocate {additional} entries for {resource}")]
    Allocation {
        /// Stable allocation purpose.
        resource: &'static str,
        /// Requested additional element count.
        additional: usize,
    },
}

/// One immutable, internally consistent process-local search view.
///
/// The Keeper component retains sealed segments and MANIFEST tombstones. Each
/// Delta component is an owner-isolated frozen epoch. BM25 statistics use the
/// hybrid lifecycle contract: sealed tombstones remain in Keeper's at-seal
/// counts until compaction, while Delta-local tombstones are excluded
/// immediately because those rows will never be sealed.
#[derive(Clone, Copy, Debug)]
pub(super) struct DeltaLeaseLookup {
    pub(super) lease_base: u64,
    pub(super) lease_end: u64,
    pub(super) delta_index: usize,
}

#[cfg(test)]
thread_local! {
    static DELTA_LEASE_OVERLAP_COMPARISONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_delta_lease_overlap_comparisons() {
    DELTA_LEASE_OVERLAP_COMPARISONS.with(|comparisons| comparisons.set(0));
}

#[cfg(test)]
pub(super) fn delta_lease_overlap_comparisons() -> usize {
    DELTA_LEASE_OVERLAP_COMPARISONS.with(Cell::get)
}

pub struct QuillSearchSnapshot {
    pub(super) snapshot_epoch: u64,
    pub(super) keeper: Arc<KeeperSnapshot>,
    pub(super) deltas: Box<[Arc<DeltaSnapshot>]>,
    delta_lease_lookup: Vec<DeltaLeaseLookup>,
    pub(super) field_stats: Box<[SnapshotFieldStats]>,
    pub(super) bm25_doc_count: u64,
    pub(super) live_doc_count: u64,
}

impl QuillSearchSnapshot {
    pub(super) fn compose(
        snapshot_epoch: u64,
        keeper: Arc<KeeperSnapshot>,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Self, SnapshotError> {
        let schema = keeper.schema();
        let manifest = &keeper.loaded_manifest().manifest;
        validate_delta_table(schema, manifest, &deltas)?;
        let delta_lease_lookup = build_delta_lease_lookup(&deltas)?;

        let delta_live_doc_count = delta_live_document_count(&deltas)?;
        let bm25_doc_count = keeper
            .at_seal_doc_count()
            .checked_add(delta_live_doc_count)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "BM25 document count",
            })?;
        let live_doc_count = keeper.doc_count().checked_add(delta_live_doc_count).ok_or(
            SnapshotError::CounterOverflow {
                counter: "live document count",
            },
        )?;

        let field_stats = composite_field_stats(
            schema,
            manifest,
            keeper.segments().is_empty(),
            &deltas,
            bm25_doc_count,
        )?;

        Ok(Self {
            snapshot_epoch,
            keeper,
            deltas: deltas.into_boxed_slice(),
            delta_lease_lookup,
            field_stats,
            bm25_doc_count,
            live_doc_count,
        })
    }

    /// Monotone process-local publication epoch.
    #[must_use]
    pub const fn snapshot_epoch(&self) -> u64 {
        self.snapshot_epoch
    }

    /// Durable MANIFEST generation paired with every Delta epoch in this view.
    #[must_use]
    pub fn keeper_generation(&self) -> u64 {
        self.keeper.loaded_manifest().manifest.generation
    }

    /// Pinned immutable Keeper component.
    #[must_use]
    pub fn keeper_snapshot(&self) -> &KeeperSnapshot {
        &self.keeper
    }

    /// Number of frozen shard epochs in this view.
    #[must_use]
    pub fn delta_count(&self) -> usize {
        self.deltas.len()
    }

    /// Immutable shard epochs paired with this Keeper generation.
    #[must_use]
    pub fn delta_snapshots(&self) -> &[Arc<DeltaSnapshot>] {
        &self.deltas
    }

    /// Snapshot-global BM25 `N`: Keeper at-seal rows plus live Delta rows.
    #[must_use]
    pub const fn bm25_doc_count(&self) -> u64 {
        self.bm25_doc_count
    }

    /// Public live rows across Keeper tombstones and Delta-local tombstones.
    #[must_use]
    pub const fn live_doc_count(&self) -> u64 {
        self.live_doc_count
    }

    /// Checked composite BM25 statistics for one indexed string field.
    #[must_use]
    pub fn bm25_field_stats(&self, field_ord: u16) -> Option<SnapshotFieldStats> {
        self.field_stats
            .binary_search_by_key(&field_ord, |row| row.field_ord)
            .ok()
            .and_then(|index| self.field_stats.get(index))
            .copied()
    }

    /// Keeper physical document frequency plus live Delta document frequency.
    ///
    /// # Errors
    ///
    /// Returns typed dictionary, allocation, or checked-counter failures.
    pub fn bm25_doc_freq(&self, field_ord: u16, term: &[u8]) -> Result<u64, QuillIndexError> {
        if self.bm25_field_stats(field_ord).is_none() {
            return Err(invalid_state(format!(
                "snapshot has no BM25 statistics for field {field_ord}"
            )));
        }
        let mut total = snapshot_doc_freq(&self.keeper, self.keeper.schema(), field_ord, term)?;
        for delta in &self.deltas {
            let delta_doc_freq = delta
                .find_term(field_ord, term)
                .map_or(0, |found| found.live_doc_freq());
            total = total
                .checked_add(u64::try_from(delta_doc_freq).map_err(|_| {
                    SnapshotError::CounterOverflow {
                        counter: "live Delta document frequency",
                    }
                })?)
                .ok_or(SnapshotError::CounterOverflow {
                    counter: "snapshot document frequency",
                })?;
        }
        Ok(total)
    }

    /// Materialize one live winner from either immutable residency.
    #[must_use]
    pub fn materialize_document_id(&self, global_docid: u32) -> Option<DocId> {
        self.keeper
            .materialize_document_id(global_docid)
            .or_else(|| {
                self.delta_for_docid(global_docid)?
                    .materialize_document_id(global_docid)
            })
    }

    pub(super) fn resolve_document_witness(
        &self,
        document_id: &str,
    ) -> Result<Option<QuillDocumentWitness>, QuillIndexError> {
        for delta in self.deltas.iter().rev() {
            if let Some(global_docid) = delta.segment().probe_id(document_id) {
                let content_hash = delta.content_hash(global_docid).ok_or_else(|| {
                    invalid_state(format!(
                        "live Delta document {document_id:?} has no resumable content witness"
                    ))
                })?;
                return Ok(Some(QuillDocumentWitness {
                    global_docid,
                    content_hash,
                }));
            }
        }
        Ok(self
            .keeper
            .resolve_document_id(document_id)?
            .map(|resolved| QuillDocumentWitness {
                global_docid: resolved.global_docid,
                content_hash: resolved.content_hash,
            }))
    }

    pub(super) fn resolve_document_id(
        &self,
        document_id: &str,
    ) -> Result<Option<u32>, QuillIndexError> {
        self.resolve_document_witness(document_id)
            .map(|witness| witness.map(|witness| witness.global_docid))
    }

    /// Read one stored column for a live document.
    ///
    /// The five-field shipping schema keeps everything a document stores in a
    /// single JSON `METADATA` column, so its consumers never name an ordinal. A
    /// wider schema stores each column separately — the CASS profile stores its
    /// title and preview as their own columns — and a consumer reading those
    /// back has to say which one it wants.
    ///
    /// Returns `None` when the document is not live in this snapshot or the
    /// column holds nothing for it. Stored text columns hold their source UTF-8
    /// bytes; stored numeric columns hold the canonical eight little-endian
    /// bytes.
    ///
    /// # Errors
    ///
    /// Returns a typed section-validation failure if the segment's STOREDMETA
    /// section is absent or malformed.
    pub fn stored_field_value(
        &self,
        field_ord: u16,
        global_docid: u32,
    ) -> Result<Option<Vec<u8>>, QuillIndexError> {
        self.materialize_stored_value(field_ord, global_docid)
    }

    pub(super) fn materialize_metadata(
        &self,
        global_docid: u32,
    ) -> Result<Option<Arc<serde_json::Value>>, QuillIndexError> {
        let Some(stored) = self.materialize_stored_value(METADATA_FIELD, global_docid)? else {
            return Ok(None);
        };
        let metadata: serde_json::Value = serde_json::from_slice(&stored)?;
        // Ingest always writes a canonical metadata object, so a document indexed
        // without metadata stores `{}`. Tantivy omits the field entirely in that
        // case, and an empty object carries no information anyway — materialize
        // both as `None` so the public envelope never hands out an
        // information-free `Some({})` (with its per-hit `Arc`) and so
        // `metadata.is_some()` means the same thing on both engines.
        if metadata.as_object().is_some_and(serde_json::Map::is_empty) {
            return Ok(None);
        }
        Ok(Some(Arc::new(metadata)))
    }

    pub(super) fn materialize_stored_value(
        &self,
        field_ord: u16,
        global_docid: u32,
    ) -> Result<Option<Vec<u8>>, QuillIndexError> {
        if let Some(delta) = self.delta_for_docid(global_docid) {
            match delta.materialize_stored_value(field_ord, global_docid) {
                DeltaStoredMaterialization::NotLive => {}
                DeltaStoredMaterialization::Live(stored) => {
                    return Ok(stored.map(<[u8]>::to_vec));
                }
            }
        }

        let Some(segment) = self.keeper.live_segment_for_docid(global_docid) else {
            return Ok(None);
        };
        let stored =
            segment.stored_meta_lookup(required_section(segment, SectionKind::STOREDMETA)?)?;
        Ok(stored
            .field(field_ord)
            .and_then(|field| field.get(u64::from(global_docid)))
            .map(<[u8]>::to_vec))
    }

    /// Clone the pinned immutable Keeper component.
    #[must_use]
    pub fn keeper_snapshot_arc(&self) -> Arc<KeeperSnapshot> {
        Arc::clone(&self.keeper)
    }

    fn delta_for_docid(&self, global_docid: u32) -> Option<&DeltaSnapshot> {
        let global_docid = u64::from(global_docid);
        let predecessor = self
            .delta_lease_lookup
            .partition_point(|entry| entry.lease_base <= global_docid)
            .checked_sub(1)?;
        let entry = self.delta_lease_lookup.get(predecessor)?;
        (global_docid < entry.lease_end)
            .then(|| self.deltas.get(entry.delta_index))?
            .map(Arc::as_ref)
    }
}

fn build_delta_lease_lookup(
    deltas: &[Arc<DeltaSnapshot>],
) -> Result<Vec<DeltaLeaseLookup>, SnapshotError> {
    let mut lookup = Vec::new();
    lookup
        .try_reserve_exact(deltas.len())
        .map_err(|_| SnapshotError::Allocation {
            resource: "Delta lease lookup index",
            additional: deltas.len(),
        })?;
    for (delta_index, delta) in deltas.iter().enumerate() {
        lookup.push(DeltaLeaseLookup {
            lease_base: delta.lease_base(),
            lease_end: delta.lease_end(),
            delta_index,
        });
    }
    lookup.sort_unstable_by_key(|entry| entry.lease_base);
    for pair in lookup.windows(2) {
        #[cfg(test)]
        DELTA_LEASE_OVERLAP_COMPARISONS.with(|comparisons| {
            comparisons.set(comparisons.get() + 1);
        });
        let first = pair[0];
        let second = pair[1];
        if first.lease_end > second.lease_base {
            return Err(SnapshotError::OverlappingDeltaLeases {
                first_base: first.lease_base,
                first_end: first.lease_end,
                second_base: second.lease_base,
                second_end: second.lease_end,
            });
        }
    }
    Ok(lookup)
}

fn validate_delta_table(
    schema: SchemaDescriptor,
    manifest: &Manifest,
    deltas: &[Arc<DeltaSnapshot>],
) -> Result<(), SnapshotError> {
    let keeper_generation = manifest.generation;
    for delta in deltas {
        if delta.schema() != schema {
            return Err(SnapshotError::SchemaMismatch {
                expected: schema.name,
                actual: delta.schema().name,
            });
        }
        if delta.keeper_generation() != keeper_generation {
            return Err(SnapshotError::KeeperGenerationMismatch {
                expected: keeper_generation,
                actual: delta.keeper_generation(),
                lease_base: delta.lease_base(),
            });
        }
        if let Some((delta_lo, delta_hi)) = delta.occupied_docid_range() {
            for segment in &manifest.segments {
                if delta_lo < segment.docid_hi && segment.docid_lo < delta_hi {
                    return Err(SnapshotError::KeeperDeltaDocidOverlap {
                        delta_lo,
                        delta_hi,
                        segment_id: segment.segment_id,
                        keeper_lo: segment.docid_lo,
                        keeper_hi: segment.docid_hi,
                    });
                }
            }
        }
    }
    Ok(())
}

pub(super) fn validate_delta_seal_replacements(
    current: &QuillSearchSnapshot,
    sealed: &Arc<DeltaSnapshot>,
    replacements: &[Arc<DeltaSnapshot>],
) -> Result<(), SnapshotError> {
    for survivor in current
        .delta_snapshots()
        .iter()
        .filter(|delta| !Arc::ptr_eq(delta, sealed))
    {
        let lineage = survivor.publication_lineage();
        if !replacements
            .iter()
            .any(|replacement| replacement.publication_lineage() == lineage)
        {
            return Err(SnapshotError::MissingDeltaRebind {
                lease_base: survivor.lease_base(),
                lease_end: survivor.lease_end(),
            });
        }
    }
    Ok(())
}

fn delta_live_document_count(deltas: &[Arc<DeltaSnapshot>]) -> Result<u64, SnapshotError> {
    deltas.iter().try_fold(0_u64, |total, delta| {
        let count = u64::try_from(delta.live_document_count()).map_err(|_| {
            SnapshotError::CounterOverflow {
                counter: "live Delta document count",
            }
        })?;
        total
            .checked_add(count)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "live Delta document count",
            })
    })
}

fn composite_field_stats(
    schema: SchemaDescriptor,
    manifest: &Manifest,
    genesis_without_segments: bool,
    deltas: &[Arc<DeltaSnapshot>],
    bm25_doc_count: u64,
) -> Result<Box<[SnapshotFieldStats]>, SnapshotError> {
    let indexed_field_count = schema
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        .count();
    let mut field_stats = Vec::new();
    field_stats
        .try_reserve_exact(indexed_field_count)
        .map_err(|_| SnapshotError::Allocation {
            resource: "composite field statistics",
            additional: indexed_field_count,
        })?;
    for field in schema
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
    {
        let sealed_total_tokens = manifest
            .field_stats
            .iter()
            .find(|row| row.field_ord == field.id)
            .map(|row| row.total_tokens)
            .or_else(|| genesis_without_segments.then_some(0))
            .ok_or(SnapshotError::MissingKeeperFieldStats {
                field_ord: field.id,
            })?;
        let total_tokens = deltas
            .iter()
            .try_fold(sealed_total_tokens, |total, delta| {
                let delta_tokens = delta.live_total_tokens(field.id).ok_or(
                    SnapshotError::MissingDeltaFieldStats {
                        field_ord: field.id,
                    },
                )?;
                total
                    .checked_add(delta_tokens)
                    .ok_or(SnapshotError::CounterOverflow {
                        counter: "field token count",
                    })
            })?;
        field_stats.push(SnapshotFieldStats {
            field_ord: field.id,
            total_tokens,
            doc_count: bm25_doc_count,
        });
    }
    Ok(field_stats.into_boxed_slice())
}

fn manifest_document_counts(manifest: &Manifest) -> Result<(u64, u64), SnapshotError> {
    manifest
        .segments
        .iter()
        .try_fold((0_u64, 0_u64), |(at_seal, live), segment| {
            let at_seal = at_seal.checked_add(u64::from(segment.doc_count)).ok_or(
                SnapshotError::CounterOverflow {
                    counter: "Keeper at-seal document count",
                },
            )?;
            let live = live
                .checked_add(u64::from(segment.live_doc_count()))
                .ok_or(SnapshotError::CounterOverflow {
                    counter: "Keeper live document count",
                })?;
            Ok((at_seal, live))
        })
}

fn validate_complete_keeper_transition(
    current: &Manifest,
    proposed: &Manifest,
) -> Result<(), SnapshotError> {
    if proposed.generation < current.generation {
        return Err(SnapshotError::KeeperGenerationRegression {
            current: current.generation,
            proposed: proposed.generation,
        });
    }
    if proposed.generation == current.generation {
        return if proposed == current {
            Ok(())
        } else {
            Err(SnapshotError::KeeperGenerationCollision {
                generation: proposed.generation,
            })
        };
    }
    // The Keeper has already admitted `proposed` with the caller's
    // `PublishIntent`; this re-derives the process-local view of that
    // decision, so the live-document floor is not re-applied here.
    validate_manifest_successor(current, proposed, LiveDocumentFloor::Waived).map_err(|error| {
        SnapshotError::KeeperTransition {
            detail: error.to_string(),
        }
    })
}

fn validate_durable_keeper_catch_up(
    current: &Manifest,
    proposed: &Manifest,
) -> Result<bool, SnapshotError> {
    if proposed.generation < current.generation {
        return Err(SnapshotError::KeeperGenerationRegression {
            current: current.generation,
            proposed: proposed.generation,
        });
    }
    if proposed.generation == current.generation {
        return if proposed == current {
            Ok(false)
        } else {
            Err(SnapshotError::KeeperGenerationCollision {
                generation: proposed.generation,
            })
        };
    }
    if proposed.schema_id != current.schema_id {
        return Err(SnapshotError::KeeperTransition {
            detail: format!(
                "durable catch-up changed schema_id from {:#018x} to {:#018x}",
                current.schema_id, proposed.schema_id
            ),
        });
    }
    if proposed.docid_high_watermark < current.docid_high_watermark {
        return Err(SnapshotError::KeeperTransition {
            detail: format!(
                "durable catch-up rolled docid_high_watermark back from {} to {}",
                current.docid_high_watermark, proposed.docid_high_watermark
            ),
        });
    }
    Ok(true)
}

pub(super) struct PreparedSealedPublication {
    pub(super) snapshot_epoch: u64,
    pub(super) expected_keeper_generation: u64,
    pub(super) expected_schema_id: u64,
    pub(super) expected_docid_high_watermark: u64,
    pub(super) schema: SchemaDescriptor,
    pub(super) field_stats: Box<[SnapshotFieldStats]>,
    pub(super) bm25_doc_count: u64,
    pub(super) live_doc_count: u64,
    pub(super) delta_live_doc_count: u64,
    pub(super) deltas: Box<[Arc<DeltaSnapshot>]>,
    pub(super) delta_lease_lookup: Vec<DeltaLeaseLookup>,
}

/// Lock-free publisher for one authoritative Keeper plus all-Delta view.
///
/// Readers call [`Self::load`] once and retain that `Arc` for the whole query.
/// Per-shard updates use compare-and-swap so concurrent shard writers cannot
/// lose one another's epochs. A seal transition must use
/// [`Self::publish_complete`] to replace Keeper and every Delta in one swap.
pub struct SnapshotPublisher {
    pub(super) current: ArcSwap<QuillSearchSnapshot>,
    pub(super) ranked_query_cache: RankedQueryCache,
}

impl SnapshotPublisher {
    /// Construct the first process-local view at epoch zero.
    ///
    /// # Errors
    ///
    /// Rejects stale generations, schema drift, overlapping leases, missing
    /// field statistics, allocation failure, and checked-counter overflow.
    pub fn new(
        keeper: Arc<KeeperSnapshot>,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Self, SnapshotError> {
        let initial = Arc::new(QuillSearchSnapshot::compose(0, keeper, deltas)?);
        Ok(Self {
            current: ArcSwap::new(initial),
            ranked_query_cache: RankedQueryCache::default(),
        })
    }

    /// Clone the current immutable view without taking a reader lock.
    #[must_use]
    pub fn load(&self) -> Arc<QuillSearchSnapshot> {
        self.current.load_full()
    }

    pub(super) fn prepare_sealed_manifest(
        &self,
        schema: SchemaDescriptor,
        proposed: &Manifest,
    ) -> Result<PreparedSealedPublication, SnapshotError> {
        self.prepare_sealed_manifest_with_deltas(schema, proposed, Vec::new())
    }

    pub(super) fn prepare_sealed_manifest_with_deltas(
        &self,
        schema: SchemaDescriptor,
        proposed: &Manifest,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<PreparedSealedPublication, SnapshotError> {
        let current = self.current.load_full();
        validate_complete_keeper_transition(&current.keeper.loaded_manifest().manifest, proposed)?;
        validate_delta_table(schema, proposed, &deltas)?;
        let delta_lease_lookup = build_delta_lease_lookup(&deltas)?;
        let snapshot_epoch = current
            .snapshot_epoch()
            .checked_add(1)
            .ok_or(SnapshotError::EpochExhausted)?;
        let (sealed_doc_count, sealed_live_doc_count) = manifest_document_counts(proposed)?;
        let delta_live_doc_count = delta_live_document_count(&deltas)?;
        let bm25_doc_count = sealed_doc_count.checked_add(delta_live_doc_count).ok_or(
            SnapshotError::CounterOverflow {
                counter: "BM25 document count",
            },
        )?;
        let live_doc_count = sealed_live_doc_count
            .checked_add(delta_live_doc_count)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "live document count",
            })?;
        let field_stats = composite_field_stats(
            schema,
            proposed,
            proposed.segments.is_empty(),
            &deltas,
            bm25_doc_count,
        )?;
        Ok(PreparedSealedPublication {
            snapshot_epoch,
            expected_keeper_generation: proposed.generation,
            expected_schema_id: proposed.schema_id,
            expected_docid_high_watermark: proposed.docid_high_watermark,
            schema,
            field_stats,
            bm25_doc_count,
            live_doc_count,
            delta_live_doc_count,
            deltas: deltas.into_boxed_slice(),
            delta_lease_lookup,
        })
    }

    pub(super) fn prepare_equivalent_sealed_successor(
        &self,
    ) -> Result<PreparedSealedPublication, SnapshotError> {
        let current = self.current.load_full();
        if !current.deltas.is_empty() {
            return Err(SnapshotError::KeeperTransition {
                detail: "concat merge cannot discard active Delta epochs".to_owned(),
            });
        }
        let snapshot_epoch = current
            .snapshot_epoch()
            .checked_add(1)
            .ok_or(SnapshotError::EpochExhausted)?;
        let manifest = &current.keeper.loaded_manifest().manifest;
        let expected_keeper_generation =
            manifest
                .generation
                .checked_add(1)
                .ok_or_else(|| SnapshotError::KeeperTransition {
                    detail: "Keeper MANIFEST generation exhausted".to_owned(),
                })?;
        let mut field_stats = Vec::new();
        field_stats
            .try_reserve_exact(current.field_stats.len())
            .map_err(|_| SnapshotError::Allocation {
                resource: "concat-merge field statistics",
                additional: current.field_stats.len(),
            })?;
        field_stats.extend_from_slice(&current.field_stats);
        Ok(PreparedSealedPublication {
            snapshot_epoch,
            expected_keeper_generation,
            expected_schema_id: manifest.schema_id,
            expected_docid_high_watermark: manifest.docid_high_watermark,
            schema: current.keeper.schema(),
            field_stats: field_stats.into_boxed_slice(),
            bm25_doc_count: current.bm25_doc_count,
            live_doc_count: current.live_doc_count,
            delta_live_doc_count: 0,
            deltas: Box::default(),
            delta_lease_lookup: Vec::new(),
        })
    }

    /// Check that Keeper's reopened successor is exactly compatible with the
    /// retained process-local publication plan. This is deliberately fallible:
    /// callers retain their retry state until it passes.
    pub(super) fn validate_prepared_sealed(
        keeper: &KeeperSnapshot,
        prepared: &PreparedSealedPublication,
    ) -> Result<(), SnapshotError> {
        let manifest = &keeper.loaded_manifest().manifest;
        if manifest.generation != prepared.expected_keeper_generation {
            return Err(SnapshotError::KeeperTransition {
                detail: format!(
                    "Keeper installed generation {} but the retained publication requires {}",
                    manifest.generation, prepared.expected_keeper_generation
                ),
            });
        }
        if manifest.schema_id != prepared.expected_schema_id || keeper.schema() != prepared.schema {
            return Err(SnapshotError::KeeperTransition {
                detail: "Keeper installed a schema that differs from the retained publication"
                    .to_owned(),
            });
        }
        if manifest.docid_high_watermark != prepared.expected_docid_high_watermark {
            return Err(SnapshotError::KeeperTransition {
                detail: format!(
                    "Keeper installed watermark {} but the retained publication requires {}",
                    manifest.docid_high_watermark, prepared.expected_docid_high_watermark
                ),
            });
        }
        if keeper
            .at_seal_doc_count()
            .checked_add(prepared.delta_live_doc_count)
            != Some(prepared.bm25_doc_count)
        {
            return Err(SnapshotError::KeeperTransition {
                detail: "Keeper plus retained Delta BM25 count differs from the publication"
                    .to_owned(),
            });
        }
        if keeper
            .doc_count()
            .checked_add(prepared.delta_live_doc_count)
            != Some(prepared.live_doc_count)
        {
            return Err(SnapshotError::KeeperTransition {
                detail: "Keeper plus retained Delta live count differs from the publication"
                    .to_owned(),
            });
        }
        for expected in &prepared.field_stats {
            let keeper_stats = manifest
                .field_stats
                .iter()
                .find(|row| row.field_ord == expected.field_ord);
            let keeper_total_tokens = keeper_stats
                .map(|row| row.total_tokens)
                .or_else(|| manifest.segments.is_empty().then_some(0));
            let keeper_doc_count = keeper_stats
                .map(|row| u64::from(row.doc_count))
                .or_else(|| manifest.segments.is_empty().then_some(0));
            let mut delta_total_tokens = 0_u64;
            for delta in &prepared.deltas {
                let delta_tokens = delta.live_total_tokens(expected.field_ord).ok_or(
                    SnapshotError::MissingDeltaFieldStats {
                        field_ord: expected.field_ord,
                    },
                )?;
                delta_total_tokens = delta_total_tokens.checked_add(delta_tokens).ok_or(
                    SnapshotError::CounterOverflow {
                        counter: "retained Delta field token count",
                    },
                )?;
            }
            if keeper_total_tokens.and_then(|total| total.checked_add(delta_total_tokens))
                != Some(expected.total_tokens)
            {
                return Err(SnapshotError::KeeperTransition {
                    detail: format!(
                        "Keeper plus retained Delta token count differs for field {}",
                        expected.field_ord
                    ),
                });
            }
            if keeper_doc_count.and_then(|count| count.checked_add(prepared.delta_live_doc_count))
                != Some(expected.doc_count)
            {
                return Err(SnapshotError::KeeperTransition {
                    detail: format!(
                        "Keeper plus retained Delta document count differs for field {}",
                        expected.field_ord
                    ),
                });
            }
        }

        Ok(())
    }

    /// Install a successor only after [`Self::validate_prepared_sealed`] has
    /// accepted it while the caller still owns every retained retry artifact.
    pub(super) fn install_validated_prepared_sealed(
        &self,
        keeper: Arc<KeeperSnapshot>,
        prepared: PreparedSealedPublication,
    ) -> Arc<QuillSearchSnapshot> {
        let next = Arc::new(QuillSearchSnapshot {
            snapshot_epoch: prepared.snapshot_epoch,
            keeper,
            deltas: prepared.deltas,
            delta_lease_lookup: prepared.delta_lease_lookup,
            field_stats: prepared.field_stats,
            bm25_doc_count: prepared.bm25_doc_count,
            live_doc_count: prepared.live_doc_count,
        });
        self.current.store(Arc::clone(&next));
        next
    }

    /// Atomically replace Keeper and every shard Delta in one publication.
    ///
    /// This is the seal boundary: callers supply a complete authoritative set,
    /// already quiesced against batch writers. The method retries only to keep
    /// the local epoch monotone if another publication wins the CAS first.
    ///
    /// # Errors
    ///
    /// Returns the same composition failures as [`Self::new`], or epoch
    /// exhaustion.
    pub fn publish_complete(
        &self,
        mut keeper: Arc<KeeperSnapshot>,
        mut deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Arc<QuillSearchSnapshot>, SnapshotError> {
        loop {
            let current = self.current.load_full();
            validate_complete_keeper_transition(
                &current.keeper.loaded_manifest().manifest,
                &keeper.loaded_manifest().manifest,
            )?;
            let epoch = current
                .snapshot_epoch()
                .checked_add(1)
                .ok_or(SnapshotError::EpochExhausted)?;
            let next = Arc::new(QuillSearchSnapshot::compose(epoch, keeper, deltas)?);
            let previous = self.current.compare_and_swap(&current, Arc::clone(&next));
            if Arc::ptr_eq(&current, &previous) {
                return Ok(next);
            }
            keeper = Arc::clone(&next.keeper);
            deltas = clone_delta_arcs(&next.deltas)?;
        }
    }

    /// Advance a read-only view to a fully authenticated newer durable
    /// snapshot, including when the reader did not observe every intermediate
    /// generation.
    ///
    /// Writer publication continues to use [`Self::publish_complete`] and its
    /// exact-successor validation. This path accepts only forward generations
    /// with the same schema and a non-regressing document-id watermark, and it
    /// never discards process-local Delta epochs.
    pub(super) fn publish_durable_catch_up(
        &self,
        keeper: &Arc<KeeperSnapshot>,
    ) -> Result<Option<Arc<QuillSearchSnapshot>>, SnapshotError> {
        loop {
            let current = self.current.load_full();
            if !current.deltas.is_empty() {
                return Err(SnapshotError::KeeperTransition {
                    detail: "durable read-only catch-up cannot discard active Delta epochs"
                        .to_owned(),
                });
            }
            if !validate_durable_keeper_catch_up(
                &current.keeper.loaded_manifest().manifest,
                &keeper.loaded_manifest().manifest,
            )? {
                return Ok(None);
            }
            let epoch = current
                .snapshot_epoch()
                .checked_add(1)
                .ok_or(SnapshotError::EpochExhausted)?;
            let next = Arc::new(QuillSearchSnapshot::compose(
                epoch,
                Arc::clone(keeper),
                Vec::new(),
            )?);
            let previous = self.current.compare_and_swap(&current, Arc::clone(&next));
            if Arc::ptr_eq(&current, &previous) {
                return Ok(Some(next));
            }
        }
    }

    /// Atomically replace one shard's frozen Delta epoch.
    ///
    /// Concurrent shard publications are merged through a compare-and-swap
    /// retry; each retry starts from the latest complete composite view.
    ///
    /// # Errors
    ///
    /// Rejects an unknown shard slot and all ordinary composition failures.
    pub fn publish_delta(
        &self,
        shard: usize,
        mut delta: Arc<DeltaSnapshot>,
    ) -> Result<Arc<QuillSearchSnapshot>, SnapshotError> {
        loop {
            let current = self.current.load_full();
            if shard >= current.deltas.len() {
                return Err(SnapshotError::ShardOutOfBounds {
                    shard,
                    shard_count: current.deltas.len(),
                });
            }
            let epoch = current
                .snapshot_epoch()
                .checked_add(1)
                .ok_or(SnapshotError::EpochExhausted)?;
            let mut deltas = clone_delta_arcs(&current.deltas)?;
            deltas[shard] = delta;
            let next = Arc::new(QuillSearchSnapshot::compose(
                epoch,
                Arc::clone(&current.keeper),
                deltas,
            )?);
            let previous = self.current.compare_and_swap(&current, Arc::clone(&next));
            if Arc::ptr_eq(&current, &previous) {
                return Ok(next);
            }
            delta = Arc::clone(&next.deltas[shard]);
        }
    }
}

fn clone_delta_arcs(
    deltas: &[Arc<DeltaSnapshot>],
) -> Result<Vec<Arc<DeltaSnapshot>>, SnapshotError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(deltas.len())
        .map_err(|_| SnapshotError::Allocation {
            resource: "Delta snapshot table",
            additional: deltas.len(),
        })?;
    cloned.extend(deltas.iter().cloned());
    Ok(cloned)
}
