use crate::config::QuillConfig;
use crate::delta::DeltaSnapshot;
#[cfg(feature = "bench-internals")]
use crate::keeper::BenchmarkQuillDirectorySyncState;
use crate::keeper::{
    KeeperError, KeeperSnapshot, KeeperWriter, Manifest, ManifestSegment,
    PublicationAuthorityPhase, PublicationReadState, PublishIntent,
};
use crate::query::{DefaultQueryParser, QueryParserConfigError};
use crate::schema::{DEFAULT_SCHEMA, FieldKind, SchemaDescriptor};
use crate::scribe::{
    ColumnarAccumulator, DeltaFlushInput, DocIdAllocator, ShardRouter, flush_delta_snapshot,
};
use crate::segment::EncodedSegment;
use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
#[cfg(feature = "conformance-internals")]
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::time::Instant;
use tracing::Instrument;

use super::checkpoint::PendingDeltaSeal;
#[cfg(feature = "conformance-internals")]
use super::conformance::{
    ConformancePendingWriterState, conformance_hash_bytes, conformance_hash_field_stats,
    conformance_hash_manifest_segment, conformance_hash_optional_delta_seal,
    conformance_hash_optional_lease, conformance_hash_optional_manifest,
    conformance_hash_optional_u64, conformance_manifest_bytes, conformance_usize_to_u64,
};
use super::error_results::{
    QuillIndexError, check_cancel, invalid_state, open_sidecarless_writer, record_snapshot_fields,
    validate_config, validate_docid_watermark, validate_non_durable_quarantine,
};
use super::ingest_doc::{manifest_segment, merge_field_stats};
use super::parallel_ingest::{ScribeShardState, StagedFlush, ingest_parallelism_at_construction};
use super::reader::QuillReader;
use super::snapshot::{
    PreparedSealedPublication, QuillSearchSnapshot, SnapshotPublisher,
    validate_delta_seal_replacements,
};

/// Scalar Quill writer state guarded by [`super::QuillIndex::writer`].
pub(super) struct QuillWriterState {
    pub(super) reader: QuillReader,
    pub(super) backend: IndexBackend,
    pub(super) shards: Vec<ScribeShardState>,
    pub(super) shard_router: ShardRouter,
    pub(super) docid_allocator: DocIdAllocator,
    pub(super) uncommitted_ids: BTreeSet<String>,
    pub(super) next_lease_base: u64,
    pub(super) next_seal_seq: u64,
    pub(super) staged_flush: Option<StagedFlush>,
    pub(super) pending_segments: Vec<ManifestSegment>,
    pub(super) pending_owned_segments: Vec<EncodedSegment>,
    pub(super) pending_field_stats: BTreeMap<u16, (u64, u32)>,
    pub(super) pending_manifest: Option<Manifest>,
    pub(super) pending_replacement_manifest: Option<Manifest>,
    pub(super) pending_delta_seal: Option<PendingDeltaSeal>,
    pub(super) unpublished_since: Option<Instant>,
    /// A prior scalar ingest returned an error after mutation may have begun.
    ///
    /// Successfully installed segments may span multiple bounded batches, but
    /// an ambiguous partial batch must be reconciled by `commit` before any
    /// further scalar indexing is accepted.
    pub(super) ingest_retry_required: bool,
    /// Test-only, per index, default OFF. When set, the shared-nothing routes
    /// arm the shard rendezvous from their REAL plan so a test can force
    /// overlap. It must stay opt-in: arming it for every `cfg(test)` fan-out
    /// changes scheduling across the whole suite and makes a serial pool wait
    /// out the rendezvous deadline on every batch.
    #[cfg(test)]
    pub(super) force_ingest_overlap_rendezvous: bool,
    /// Test-only, default OFF. When set to `Some(n)`, the serial ingest route
    /// requests cancellation on the request `Cx` once exactly `n` documents of
    /// the current batch have been accumulated, so a test can reproduce a
    /// batch that fails midway with real partial shard state behind it
    /// (gh#45). The cancellation is discovered by the ordinary
    /// `check_cancel` poll; nothing else is short-circuited.
    #[cfg(test)]
    pub(super) cancel_ingest_after_documents: Option<usize>,
    /// Documents actually analyzed by the adaptive budget proof. This is
    /// per-index and test-only so a regression can distinguish bounded-wave
    /// refusal from eagerly analyzing the whole rejected batch.
    #[cfg(test)]
    pub(super) parallel_budget_documents_analyzed: AtomicUsize,
}

impl Deref for QuillWriterState {
    type Target = QuillReader;

    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

pub(super) enum IndexBackend {
    Durable(KeeperWriter),
    Memory(KeeperSnapshot),
}

impl IndexBackend {
    /// Retained state for local bookkeeping only.
    ///
    /// This must never be used to answer an authority-sensitive facade call:
    /// an abandoned durable publication can leave it behind disk.
    pub(super) const fn retained_snapshot_for_bookkeeping(&self) -> &KeeperSnapshot {
        match self {
            Self::Durable(writer) => writer.retained_snapshot_for_bookkeeping(),
            Self::Memory(snapshot) => snapshot,
        }
    }

    /// Return the proven Keeper authority view, failing closed after a dropped
    /// durable publication future until its writer has reconciled.
    ///
    /// This deliberately says nothing about the process-local published
    /// snapshot. The narrow post-publication installation path needs the
    /// durable successor while the local `ArcSwap` still names its predecessor.
    pub(super) fn proven_authority_snapshot(&self) -> Result<&KeeperSnapshot, KeeperError> {
        match self {
            Self::Durable(writer) => writer.snapshot(),
            Self::Memory(snapshot) => Ok(snapshot),
        }
    }

    const fn publication_awaits_reconciliation(&self) -> bool {
        match self {
            Self::Durable(writer) => writer.publication_awaits_reconciliation(),
            Self::Memory(_) => false,
        }
    }

    fn bind_publication_read_state(&mut self, state: PublicationReadState) {
        if let Self::Durable(writer) = self {
            writer.bind_publication_read_state(state);
        }
    }

    #[cfg(feature = "bench-internals")]
    pub(super) fn bind_benchmark_qg4_directory_sync_state(
        &mut self,
        state: Arc<BenchmarkQuillDirectorySyncState>,
    ) {
        if let Self::Durable(writer) = self {
            writer.bind_benchmark_qg4_directory_sync_state(state);
        }
    }
}

impl QuillWriterState {
    /// Create a shipping-schema index or open an existing compatible index.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub(super) async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Self::create_with_schema(cx, directory, DEFAULT_SCHEMA, config).await
    }

    /// Open an existing shipping-schema index for mutation and search.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub(super) async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer =
                open_sidecarless_writer(cx, directory.into(), DEFAULT_SCHEMA, &config, false)
                    .await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), DEFAULT_SCHEMA, config)?;
            record_snapshot_fields(&open_span, index.authority_snapshot()?);
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Open an existing on-disk index bound to an explicit compiled schema.
    ///
    /// The schema is not inferred from the persisted MANIFEST: Keeper verifies
    /// the supplied descriptor against the stored `schema_id`, so passing the
    /// wrong descriptor is a typed schema failure rather than a silent
    /// misread.
    ///
    /// # Errors
    ///
    /// Returns the same typed configuration, admission, recovery, or schema
    /// failures as [`Self::open`].
    pub(super) async fn open_with_schema(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer =
                open_sidecarless_writer(cx, directory.into(), schema, &config, false).await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), schema, config)?;
            record_snapshot_fields(&open_span, index.authority_snapshot()?);
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Create an on-disk index bound to an explicit compiled schema.
    ///
    /// [`Self::create`] pins [`DEFAULT_SCHEMA`]; this is the seam for the other
    /// compiled descriptors, notably `CASS_SEMANTIC_SCHEMA`, whose durable
    /// indexes were previously constructible only in memory.
    ///
    /// # Errors
    ///
    /// Returns the same typed configuration, admission, and durability
    /// failures as [`Self::create`].
    pub(super) async fn create_with_schema(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer =
                open_sidecarless_writer(cx, directory.into(), schema, &config, true).await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), schema, config)?;
            record_snapshot_fields(&open_span, index.authority_snapshot()?);
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Construct an owned-buffer genesis index without filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or parser failures.
    pub(super) fn in_memory(config: QuillConfig) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        validate_non_durable_quarantine(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let _open_entered = open_span.enter();
        let snapshot = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
        let index = Self::from_backend(IndexBackend::Memory(snapshot), DEFAULT_SCHEMA, config)?;
        record_snapshot_fields(&open_span, index.authority_snapshot()?);
        Ok(index)
    }

    /// Build an owned-buffer index for one explicit compile-time schema.
    ///
    /// This constructor is intentionally limited to conformance and benchmark
    /// builds. It lets the mixed-residency gauntlet exercise the same private
    /// schema binding used by durable opens without expanding the shipping API.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or genesis-snapshot failures.
    #[cfg(feature = "bench-internals")]
    pub(super) fn in_memory_with_schema(
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let snapshot = KeeperSnapshot::in_memory(schema)?;
        Self::from_backend(IndexBackend::Memory(snapshot), schema, config)
    }

    /// Bind an existing owned Keeper snapshot to the private writer facade.
    ///
    /// The conformance gauntlet uses this to replay identical tombstone and
    /// field-stat history before varying only current document residency.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, parser/schema, or snapshot failures.
    #[cfg(any(feature = "bench-internals", feature = "conformance-internals"))]
    pub(super) fn from_in_memory_snapshot(
        snapshot: KeeperSnapshot,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let schema = snapshot.schema();
        Self::from_backend(IndexBackend::Memory(snapshot), schema, config)
    }

    pub(super) fn from_backend(
        mut backend: IndexBackend,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        // The shipping string-query parser is bound only for schemas that
        // actually carry its mandatory default fields. A compiled descriptor
        // may legitimately analyze `content`/`title` with a different
        // pipeline — `CASS_SEMANTIC_SCHEMA` uses the CASS analyzers — and such
        // an index is still a perfectly valid index for ingest, preparsed
        // queries, and its own parser. Absence is already a first-class state:
        // `default_parser()` returns a typed "string query APIs are
        // unavailable" refusal rather than panicking, and another construction
        // path already binds `None`.
        //
        // Only the two field-shape refusals fall back. `InvalidSchema` means
        // the descriptor itself violates the dense-ID contract, which is a real
        // defect for every schema, so it still propagates.
        let parser = match DefaultQueryParser::new(schema) {
            Ok(parser) => Some(parser),
            Err(
                QueryParserConfigError::MissingDefaultField { .. }
                | QueryParserConfigError::InvalidDefaultField { .. },
            ) => None,
            Err(error) => return Err(error.into()),
        };
        let initial_snapshot = backend.retained_snapshot_for_bookkeeping().clone();
        let manifest = &initial_snapshot.loaded_manifest().manifest;
        let initial_generation = manifest.generation;
        let next_lease_base = validate_docid_watermark(manifest.docid_high_watermark)?;
        let detected_parallelism = ingest_parallelism_at_construction();
        let shard_router = ShardRouter::from_config(&config, detected_parallelism);
        let docid_allocator = DocIdAllocator::open(next_lease_base, shard_router.shard_count())
            .map_err(|error| invalid_state(error.to_string()))?;
        let mut shards = Vec::new();
        shards
            .try_reserve_exact(shard_router.shard_count())
            .map_err(|_| invalid_state("could not allocate Scribe shard table"))?;
        for _ in 0..shard_router.shard_count() {
            shards.push(ScribeShardState {
                accumulator: ColumnarAccumulator::new(schema)?,
                identities: Vec::new(),
                current_lease_base: None,
                scratch_metadata: Vec::new(),
            });
        }
        let next_seal_seq = manifest
            .segments
            .iter()
            .map(|segment| segment.seal_seq)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        let publication_read_state = PublicationReadState::new(initial_generation)?;
        backend.bind_publication_read_state(publication_read_state.clone());
        let published_snapshot = Arc::new(SnapshotPublisher::new(
            Arc::new(initial_snapshot),
            Vec::new(),
        )?);
        Ok(Self {
            reader: QuillReader {
                config,
                schema,
                parser,
                published_snapshot,
                publication_read_state,
                #[cfg(test)]
                publication_read_pause: Arc::default(),
                #[cfg(feature = "conformance-internals")]
                conformance_controller: Arc::default(),
            },
            backend,
            shards,
            shard_router,
            docid_allocator,
            uncommitted_ids: BTreeSet::new(),
            next_lease_base,
            next_seal_seq,
            staged_flush: None,
            pending_segments: Vec::new(),
            pending_owned_segments: Vec::new(),
            pending_field_stats: BTreeMap::new(),
            pending_manifest: None,
            pending_replacement_manifest: None,
            pending_delta_seal: None,
            unpublished_since: None,
            ingest_retry_required: false,
            #[cfg(test)]
            force_ingest_overlap_rendezvous: false,
            #[cfg(test)]
            cancel_ingest_after_documents: None,
            #[cfg(test)]
            parallel_budget_documents_analyzed: AtomicUsize::new(0),
        })
    }

    /// Return Keeper's proven durable authority without admitting the retained
    /// process-local view as public authority.
    ///
    /// This is reserved for the synchronous post-publication successor
    /// installation window. Keeper still rejects a pending or unreconciled
    /// publication; only the intentional local predecessor/successor mismatch
    /// is permitted here.
    pub(super) fn proven_authority_snapshot(&self) -> Result<&KeeperSnapshot, QuillIndexError> {
        Ok(self.backend.proven_authority_snapshot()?)
    }

    /// Reject a successor that cannot be represented by the shared
    /// reader-visible publication word before mutating an owned Keeper.
    fn preflight_successor_generation(generation: u64) -> Result<(), QuillIndexError> {
        PublicationReadState::validate_generation(generation)?;
        Ok(())
    }

    /// Apply the packed-generation preflight at the common prepared
    /// publication boundary shared by seal, scalar, bulk, concat, and delete
    /// mutation paths.
    pub(super) fn preflight_prepared_publication(
        prepared: &PreparedSealedPublication,
    ) -> Result<(), QuillIndexError> {
        Self::preflight_successor_generation(prepared.expected_keeper_generation)
    }

    /// Return authority only when Keeper and the published process-local view
    /// name the same generation.
    ///
    /// Planning and all public facade reads use this strict path. It fails
    /// closed when either the durable writer or the process-local published
    /// view needs reconciliation after a publication future drop.
    pub(super) fn authority_snapshot(&self) -> Result<&KeeperSnapshot, QuillIndexError> {
        let published = self.reader.checked_published_snapshot()?;
        let snapshot = self.proven_authority_snapshot()?;
        let published_generation = published.keeper_generation();
        let durable_generation = snapshot.loaded_manifest().manifest.generation;
        if published_generation != durable_generation {
            return Err(QuillIndexError::PublicationReconciliationRequired {
                published_generation,
                durable_generation,
            });
        }
        Ok(snapshot)
    }

    /// Whether an abandoned or incompletely surfaced publication must be
    /// reconciled before another mutation may plan against this writer.
    fn publication_awaits_reconciliation(&self) -> bool {
        self.backend.publication_awaits_reconciliation()
            || matches!(
                self.reader.publication_read_state.load().phase(),
                PublicationAuthorityPhase::DurableAhead | PublicationAuthorityPhase::Indeterminate
            )
            || self.published_snapshot.load().keeper_generation()
                != self
                    .backend
                    .retained_snapshot_for_bookkeeping()
                    .loaded_manifest()
                    .manifest
                    .generation
    }

    /// Clear only the scalar transaction state whose MANIFEST has already
    /// been proven and installed into the local publication. Keeping this
    /// separate from durable I/O lets recovery retain every artifact until the
    /// exact successor has passed `validate_prepared_sealed`.
    pub(super) fn finish_published_pending_segments(&mut self) {
        self.pending_segments.clear();
        self.pending_owned_segments.clear();
        self.pending_field_stats.clear();
        self.pending_manifest = None;
        self.pending_replacement_manifest = None;
        self.uncommitted_ids.clear();
        for shard in &self.shards {
            self.uncommitted_ids.extend(
                shard
                    .identities
                    .iter()
                    .map(|identity| identity.document_id.clone()),
            );
        }
        if self
            .shards
            .iter()
            .all(|shard| shard.accumulator.document_count() == 0)
        {
            self.unpublished_since = None;
        }
    }

    /// Drop every uncommitted scalar mutation and return the writer to the
    /// published generation.
    ///
    /// This is the recovery for a replacement that cannot complete: the
    /// tombstones for the superseded identities are already staged in the
    /// retained proposal, but the rows meant to stand behind them did not all
    /// land. Publishing the proposal would delete documents nobody asked to
    /// delete; publishing the rows without the proposal would leave two live
    /// rows per replaced identity. Neither half is usable alone, so both are
    /// discarded and the previous committed generation stays authoritative
    /// (gh#45).
    ///
    /// Only bookkeeping is touched. Allocated document ids, `seal_seq` values
    /// and any segment file a mid-batch flush already installed stay consumed:
    /// the watermark and seal sequence are monotone by contract, and an
    /// installed segment no MANIFEST references is an ordinary grace-window GC
    /// candidate. Delta transaction state is not scalar state and is left
    /// alone.
    pub(super) fn discard_scalar_transaction(&mut self) {
        for shard in &mut self.shards {
            shard.accumulator.reset();
            shard.identities.clear();
            shard.current_lease_base = None;
        }
        self.staged_flush = None;
        self.pending_segments.clear();
        self.pending_owned_segments.clear();
        self.pending_field_stats.clear();
        self.pending_manifest = None;
        self.pending_replacement_manifest = None;
        self.uncommitted_ids.clear();
        self.unpublished_since = None;
        self.ingest_retry_required = false;
    }

    /// Reconcile an abandoned durable publication and install the exact
    /// matching process-local successor before allowing strict reads again.
    ///
    /// The writer lock held by the public wrapper excludes new mutation while
    /// Keeper opens the durable MANIFEST. Retained Delta and scalar retry
    /// state is not taken or cleared until the reopened successor validates.
    pub(super) async fn reconcile_publication(
        &mut self,
        cx: &Cx,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let authority = match &mut self.backend {
            IndexBackend::Durable(writer) => {
                Arc::new(writer.reconcile_publication(cx).await?.clone())
            }
            IndexBackend::Memory(snapshot) => Arc::new(snapshot.clone()),
        };
        let local = self.published_snapshot.load();
        let local_generation = local.keeper_generation();
        let durable_generation = authority.loaded_manifest().manifest.generation;
        if durable_generation < local_generation {
            return Err(invalid_state(format!(
                "reconciliation observed durable generation {durable_generation} behind local generation {local_generation}"
            )));
        }
        if durable_generation == local_generation {
            self.reader
                .publication_read_state
                .stabilize_reconciled_same_generation(local_generation)?;
            return Ok(local);
        }

        if self.pending_delta_seal.is_some() {
            let pending = self
                .pending_delta_seal
                .as_ref()
                .expect("checked pending Delta seal remains present");
            SnapshotPublisher::validate_prepared_sealed(authority.as_ref(), &pending.prepared)?;
            let next_lease_base = self.next_lease_base.max(pending.successor_watermark);
            let docid_allocator =
                DocIdAllocator::open(next_lease_base, self.shard_router.shard_count())
                    .map_err(|error| invalid_state(error.to_string()))?;
            let pending = self
                .pending_delta_seal
                .take()
                .expect("validated pending Delta seal remains present");
            let installed = self
                .reader
                .install_validated_prepared_sealed(authority, pending.prepared)?;
            self.next_seal_seq = pending.next_seal_seq;
            self.next_lease_base = next_lease_base;
            self.docid_allocator = docid_allocator;
            return Ok(installed);
        }

        if let Some(manifest) = self.pending_manifest.as_ref() {
            let prepared = self
                .published_snapshot
                .prepare_sealed_manifest(self.schema, manifest)?;
            SnapshotPublisher::validate_prepared_sealed(authority.as_ref(), &prepared)?;
            let installed = self
                .reader
                .install_validated_prepared_sealed(authority, prepared)?;
            self.finish_published_pending_segments();
            return Ok(installed);
        }

        if self.staged_flush.is_some()
            || !self.pending_segments.is_empty()
            || !self.pending_owned_segments.is_empty()
            || !self.pending_field_stats.is_empty()
            || self.pending_replacement_manifest.is_some()
        {
            return Err(invalid_state(
                "durable authority advanced while a scalar transaction lacks its retained MANIFEST proposal",
            ));
        }
        if !local.delta_snapshots().is_empty() {
            return Err(invalid_state(
                "durable authority advanced without a retained Delta-seal proposal",
            ));
        }

        // Delete, compact, concat, and bulk-completion transitions do not
        // retain a custom process-local plan. Their durable successor is
        // complete by construction, so rebuild a no-Delta composite directly
        // from the reopened Keeper snapshot.
        let recovered_next_seal = authority
            .segments()
            .iter()
            .map(|segment| segment.manifest().seal_seq)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted during reconciliation"))?;
        let next_lease_base = self
            .next_lease_base
            .max(authority.loaded_manifest().manifest.docid_high_watermark);
        let docid_allocator =
            DocIdAllocator::open(next_lease_base, self.shard_router.shard_count())
                .map_err(|error| invalid_state(error.to_string()))?;
        let installed = self.reader.publish_complete(authority, Vec::new())?;
        self.next_seal_seq = self.next_seal_seq.max(recovered_next_seal);
        self.next_lease_base = next_lease_base;
        self.docid_allocator = docid_allocator;
        Ok(installed)
    }

    /// Atomically publish the complete process-local Delta table for the
    /// current durable Keeper generation.
    ///
    /// Batch writers call this after freezing every shard at one quiescent
    /// boundary. A same-generation publication changes only process-local
    /// visibility; cross-process readers continue to observe the MANIFEST.
    ///
    /// # Errors
    ///
    /// Rejects schema/generation drift, overlapping leases, Keeper-range
    /// overlap, allocation failure, and process-local epoch exhaustion.
    pub(super) fn publish_delta_table(
        &self,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        if self.pending_delta_seal.is_some() {
            return Err(invalid_state(
                "resume the retained Delta seal before publishing another Delta table",
            ));
        }
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "Delta publication requires the scalar writer to be fully committed",
            ));
        }
        self.reader
            .publish_complete(Arc::new(self.authority_snapshot()?.clone()), deltas)
    }

    /// Seal one already-published Delta epoch into Keeper and atomically
    /// replace the complete process-local Delta table.
    ///
    /// The frozen source remains retained through FSLX construction, segment
    /// fsync/install, and MANIFEST publication. The final `ArcSwap` installs the
    /// new Keeper plus all replacement Delta epochs in one non-cancellable
    /// step; only then may this method release its source Arc. Consequently a
    /// reader can observe the document in the old Delta or the new Keeper, but
    /// never a visibility gap between them.
    ///
    /// `replacement_deltas` must already be frozen against the successor
    /// MANIFEST generation. Multi-shard callers must rebind every surviving
    /// shard epoch through [`DeltaSnapshot::rebind_keeper_generation`], not
    /// only the shard being sealed; omission is rejected before FSLX work.
    ///
    /// # Errors
    ///
    /// Rejects scalar pending state, an unpublished source epoch, invalid
    /// replacement epochs, cancellation before durable authority changes, or
    /// any FSLX/Keeper publication failure. An installed-but-unreferenced
    /// segment remains invisible and is safe for exact retry.
    pub(super) async fn seal_delta_snapshot(
        &mut self,
        cx: &Cx,
        sealed: Arc<DeltaSnapshot>,
        replacement_deltas: Vec<Arc<DeltaSnapshot>>,
        input: DeltaFlushInput,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        check_cancel(cx, "Delta seal")?;
        if self.pending_delta_seal.is_some() {
            return Err(invalid_state(
                "a prior Delta seal requires resume_pending_delta_seal before new mutation",
            ));
        }
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "Delta seal requires the scalar writer to be fully committed",
            ));
        }
        let current = self.published_snapshot.load();
        if !current
            .delta_snapshots()
            .iter()
            .any(|delta| Arc::ptr_eq(delta, &sealed))
        {
            return Err(invalid_state(
                "Delta seal source is not present in the published process-local epoch",
            ));
        }
        if sealed.keeper_generation() != current.keeper_generation() {
            return Err(invalid_state(
                "Delta seal source is stale relative to the published Keeper",
            ));
        }
        validate_delta_seal_replacements(&current, &sealed, &replacement_deltas)?;

        let build_source = Arc::clone(&sealed);
        let encoded = spawn_blocking(move || flush_delta_snapshot(&build_source, input))
            .await?
            .map(Arc::new);
        check_cancel(cx, "Delta segment install")?;

        let mut manifest = self.authority_snapshot()?.next_manifest()?;
        // Keeper owns the one authoritative publish timestamp. Retaining a
        // zero here also lets an ambiguous retry compare the exact logical
        // proposal after Keeper normalizes its installed timestamp.
        manifest.last_publish_unix_s = 0;
        let mut next_seal_seq = self.next_seal_seq;
        if let Some(encoded) = &encoded {
            manifest
                .segments
                .push(manifest_segment(encoded, next_seal_seq));
            manifest
                .segments
                .sort_unstable_by_key(|segment| segment.docid_lo);
            next_seal_seq = next_seal_seq
                .checked_add(1)
                .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        }
        let successor_watermark = replacement_deltas.iter().fold(
            manifest.docid_high_watermark.max(sealed.lease_end()),
            |watermark, delta| watermark.max(delta.lease_end()),
        );
        manifest.docid_high_watermark = successor_watermark;
        let live_document_count = u32::try_from(sealed.live_document_count())
            .map_err(|_| invalid_state("Delta live document count does not fit u32"))?;
        let mut pending_field_stats = BTreeMap::new();
        if live_document_count != 0 {
            for field in
                self.schema.fields.iter().filter(|field| {
                    matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. })
                })
            {
                let total_tokens = sealed.live_total_tokens(field.id).ok_or_else(|| {
                    invalid_state(format!(
                        "Delta seal omitted indexed field {} statistics",
                        field.id
                    ))
                })?;
                pending_field_stats.insert(field.id, (total_tokens, live_document_count));
            }
        }
        manifest.field_stats = merge_field_stats(&manifest.field_stats, &pending_field_stats)?;
        let prepared = self
            .published_snapshot
            .prepare_sealed_manifest_with_deltas(self.schema, &manifest, replacement_deltas)?;
        self.pending_delta_seal = Some(PendingDeltaSeal {
            encoded,
            segment_installed: false,
            manifest,
            prepared,
            next_seal_seq,
            successor_watermark,
        });
        drop(sealed);
        self.finish_pending_delta_seal(cx).await
    }

    /// Resume an exact Delta seal proposal retained across cancellation or a
    /// dropped future.
    ///
    /// Until this succeeds, the old process-local Delta epoch remains visible
    /// and every writer mutation is rejected. The retained zero-timestamp
    /// MANIFEST proposal and canonical FSLX bytes make reconciliation exact
    /// even if the prior durable publication won ambiguously.
    ///
    /// # Errors
    ///
    /// Returns a typed error when no seal awaits reconciliation or when segment
    /// installation, MANIFEST publication, or Keeper reopening fails again.
    pub(super) async fn resume_pending_delta_seal(
        &mut self,
        cx: &Cx,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        if self.pending_delta_seal.is_none() {
            return Err(invalid_state("no Delta seal awaits reconciliation"));
        }
        self.finish_pending_delta_seal(cx).await
    }

    async fn finish_pending_delta_seal(
        &mut self,
        cx: &Cx,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let (encoded, segment_installed, manifest, memory_owned) = {
            let pending = self
                .pending_delta_seal
                .as_ref()
                .expect("Delta seal completion requires retained state");
            let encoded = pending.encoded.clone();
            let memory_owned = if matches!(&self.backend, IndexBackend::Memory(_)) {
                pending
                    .encoded
                    .iter()
                    .map(|encoded| encoded.as_ref().clone())
                    .collect()
            } else {
                Vec::new()
            };
            (
                encoded,
                pending.segment_installed,
                pending.manifest.clone(),
                memory_owned,
            )
        };
        Self::preflight_prepared_publication(
            &self
                .pending_delta_seal
                .as_ref()
                .expect("Delta seal completion retains its prepared publication")
                .prepared,
        )?;

        if let (Some(encoded), IndexBackend::Durable(writer)) = (encoded, &mut self.backend)
            && !segment_installed
        {
            writer
                .publish_encoded_segment_retryable(cx, encoded)
                .await?;
            self.pending_delta_seal
                .as_mut()
                .expect("installed Delta segment retains transaction state")
                .segment_installed = true;
        }
        check_cancel(cx, "Delta MANIFEST publish")?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer
                    .publish_with_intent(cx, &manifest, PublishIntent::PreserveLiveDocuments)
                    .await?;
            }
            IndexBackend::Memory(snapshot) => {
                *snapshot = snapshot.publish_owned_segments_with_intent(
                    &manifest,
                    memory_owned,
                    PublishIntent::PreserveLiveDocuments,
                )?;
            }
        }

        // No await or cancellation checkpoint is permitted after MANIFEST
        // authority changes. Proving the installed state and swapping the
        // local epoch therefore complete in the same poll that observes
        // publication.
        // Keep the retained retry proposal intact until Keeper's proven
        // successor has been obtained. If that proof fails, a caller can
        // resume the exact proposal rather than silently losing it after the
        // durable side may have advanced.
        let authority = Arc::new(self.proven_authority_snapshot()?.clone());
        let pending = self
            .pending_delta_seal
            .as_ref()
            .expect("published Delta seal retains its prepared local swap");
        SnapshotPublisher::validate_prepared_sealed(authority.as_ref(), &pending.prepared)?;
        let next_lease_base = self.next_lease_base.max(pending.successor_watermark);
        let docid_allocator =
            DocIdAllocator::open(next_lease_base, self.shard_router.shard_count())
                .map_err(|error| invalid_state(error.to_string()))?;
        let pending = self
            .pending_delta_seal
            .take()
            .expect("published Delta seal retains its prepared local swap");
        let installed = self
            .reader
            .install_validated_prepared_sealed(authority, pending.prepared)?;
        self.next_seal_seq = pending.next_seal_seq;
        self.next_lease_base = next_lease_base;
        self.docid_allocator = docid_allocator;
        Ok(installed)
    }

    /// Durable index directory, or `None` for an owned-buffer index.
    #[must_use]
    pub(super) fn directory(&self) -> Option<&Path> {
        self.backend.retained_snapshot_for_bookkeeping().directory()
    }

    /// Whether the writer holds documents or installed segments not yet visible.
    #[must_use]
    pub(super) fn has_uncommitted_changes(&self) -> bool {
        self.shards
            .iter()
            .any(|shard| shard.accumulator.document_count() != 0)
            || self.staged_flush.is_some()
            || !self.pending_segments.is_empty()
            || self.pending_manifest.is_some()
            || self.pending_replacement_manifest.is_some()
            || self.pending_delta_seal.is_some()
            || self.publication_awaits_reconciliation()
    }

    #[cfg(feature = "conformance-internals")]
    pub(super) fn conformance_pending_writer_state(
        &self,
    ) -> Result<ConformancePendingWriterState, QuillIndexError> {
        let authority = self.authority_snapshot()?;
        let shard_count = conformance_usize_to_u64(self.shards.len(), "writer shard count")?;
        let dirty_shard_count = conformance_usize_to_u64(
            self.shards
                .iter()
                .filter(|shard| {
                    shard.accumulator.document_count() != 0
                        || !shard.identities.is_empty()
                        || shard.current_lease_base.is_some()
                })
                .count(),
            "dirty writer shard count",
        )?;
        let pending_identity_count = self.shards.iter().try_fold(0_u64, |total, shard| {
            total
                .checked_add(conformance_usize_to_u64(
                    shard.identities.len(),
                    "pending identity count",
                )?)
                .ok_or_else(|| invalid_state("pending identity count overflow"))
        })?;
        let uncommitted_id_count =
            conformance_usize_to_u64(self.uncommitted_ids.len(), "uncommitted id count")?;
        let pending_segment_count =
            conformance_usize_to_u64(self.pending_segments.len(), "pending segment count")?;
        let pending_owned_segment_count = conformance_usize_to_u64(
            self.pending_owned_segments.len(),
            "pending owned segment count",
        )?;
        let staged_flush_present = self.staged_flush.is_some();
        let pending_manifest_present = self.pending_manifest.is_some();
        let pending_replacement_manifest_present = self.pending_replacement_manifest.is_some();
        let pending_delta_seal_present = self.pending_delta_seal.is_some();
        let unpublished_since_present = self.unpublished_since.is_some();
        let mut flags = 0_u8;
        if staged_flush_present {
            flags |= ConformancePendingWriterState::STAGED_FLUSH_PRESENT;
        }
        if pending_manifest_present {
            flags |= ConformancePendingWriterState::PENDING_MANIFEST_PRESENT;
        }
        if pending_replacement_manifest_present {
            flags |= ConformancePendingWriterState::PENDING_REPLACEMENT_MANIFEST_PRESENT;
        }
        if pending_delta_seal_present {
            flags |= ConformancePendingWriterState::PENDING_DELTA_SEAL_PRESENT;
        }
        if unpublished_since_present {
            flags |= ConformancePendingWriterState::UNPUBLISHED_SINCE_PRESENT;
        }
        if self.ingest_retry_required {
            flags |= ConformancePendingWriterState::INGEST_RETRY_REQUIRED;
        }

        let published = self.published_snapshot.load();
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch/quill/pending-writer-state/v2\0");
        conformance_hash_bytes(&mut hasher, &self.schema.canonical_encoding()?);
        conformance_hash_bytes(
            &mut hasher,
            &conformance_manifest_bytes(
                &published.keeper_snapshot().loaded_manifest().manifest,
                "published composite MANIFEST",
            )?,
        );
        hasher.update(published.snapshot_epoch().to_be_bytes());
        hasher.update(published.keeper_generation().to_be_bytes());
        hasher.update(published.bm25_doc_count().to_be_bytes());
        hasher.update(published.live_doc_count().to_be_bytes());
        hasher.update(
            conformance_usize_to_u64(published.delta_count(), "published Delta count")?
                .to_be_bytes(),
        );
        for delta in published.delta_snapshots() {
            conformance_hash_bytes(&mut hasher, &delta.conformance_content_sha256()?);
        }
        conformance_hash_bytes(
            &mut hasher,
            &conformance_manifest_bytes(
                &authority.loaded_manifest().manifest,
                "writer backend MANIFEST",
            )?,
        );
        hasher.update([u8::from(matches!(&self.backend, IndexBackend::Durable(_)))]);
        hasher.update(self.next_lease_base.to_be_bytes());
        hasher.update(self.next_seal_seq.to_be_bytes());
        hasher.update(self.docid_allocator.watermark().to_be_bytes());
        hasher.update(shard_count.to_be_bytes());
        hasher.update(dirty_shard_count.to_be_bytes());
        hasher.update(pending_identity_count.to_be_bytes());
        hasher.update(uncommitted_id_count.to_be_bytes());
        hasher.update(pending_segment_count.to_be_bytes());
        hasher.update(pending_owned_segment_count.to_be_bytes());
        hasher.update([flags]);

        for (shard_index, shard) in self.shards.iter().enumerate() {
            hasher
                .update(conformance_usize_to_u64(shard_index, "writer shard index")?.to_be_bytes());
            hasher.update(
                conformance_usize_to_u64(
                    shard.accumulator.document_count(),
                    "shard document count",
                )?
                .to_be_bytes(),
            );
            hasher.update(
                conformance_usize_to_u64(shard.accumulator.token_count(), "shard token count")?
                    .to_be_bytes(),
            );
            hasher.update(
                conformance_usize_to_u64(shard.accumulator.bytes_used(), "shard bytes used")?
                    .to_be_bytes(),
            );
            conformance_hash_optional_u64(&mut hasher, shard.current_lease_base);
            conformance_hash_optional_lease(
                &mut hasher,
                self.docid_allocator.live_lease(shard_index),
            );
            hasher.update(
                conformance_usize_to_u64(
                    shard.accumulator.document_ords().len(),
                    "shard document ordinal count",
                )?
                .to_be_bytes(),
            );
            for document_ord in shard.accumulator.document_ords() {
                hasher.update(document_ord.to_be_bytes());
            }
            hasher.update(
                conformance_usize_to_u64(shard.identities.len(), "shard identity count")?
                    .to_be_bytes(),
            );
            for identity in &shard.identities {
                hasher.update(identity.doc_ord.to_be_bytes());
                conformance_hash_bytes(&mut hasher, identity.document_id.as_bytes());
                // The preimage bytes are no longer retained; the witness that
                // is actually persisted stands in for them here.
                hasher.update(identity.content_hash.to_be_bytes());
            }
        }

        hasher.update(uncommitted_id_count.to_be_bytes());
        for document_id in &self.uncommitted_ids {
            conformance_hash_bytes(&mut hasher, document_id.as_bytes());
        }

        match &self.staged_flush {
            None => hasher.update([0]),
            Some(staged) => {
                hasher.update([1]);
                hasher.update(
                    conformance_usize_to_u64(staged.shard, "staged flush shard")?.to_be_bytes(),
                );
                conformance_hash_bytes(&mut hasher, staged.encoded.as_bytes());
                conformance_hash_manifest_segment(&mut hasher, &staged.manifest_segment);
                conformance_hash_field_stats(&mut hasher, &staged.pending_field_stats)?;
                hasher.update(staged.next_seal_seq.to_be_bytes());
            }
        }

        hasher.update(pending_segment_count.to_be_bytes());
        for segment in &self.pending_segments {
            conformance_hash_manifest_segment(&mut hasher, segment);
        }
        hasher.update(pending_owned_segment_count.to_be_bytes());
        for segment in &self.pending_owned_segments {
            conformance_hash_bytes(&mut hasher, segment.as_bytes());
        }
        conformance_hash_field_stats(&mut hasher, &self.pending_field_stats)?;
        conformance_hash_optional_manifest(
            &mut hasher,
            self.pending_manifest.as_ref(),
            "pending MANIFEST",
        )?;
        conformance_hash_optional_manifest(
            &mut hasher,
            self.pending_replacement_manifest.as_ref(),
            "pending replacement MANIFEST",
        )?;
        conformance_hash_optional_delta_seal(&mut hasher, self.pending_delta_seal.as_ref())?;

        Ok(ConformancePendingWriterState {
            digest_sha256: hasher.finalize().into(),
            shard_count,
            dirty_shard_count,
            pending_identity_count,
            uncommitted_id_count,
            pending_segment_count,
            pending_owned_segment_count,
            flags,
        })
    }

    pub(super) fn has_active_deltas(&self) -> bool {
        !self.published_snapshot.load().delta_snapshots().is_empty()
    }
}
