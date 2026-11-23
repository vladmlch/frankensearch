#[cfg(test)]
use crate::argus::TopDocsCollector;
use crate::config::QuillConfig;
use crate::delta::DeltaSnapshot;
#[cfg(feature = "bench-internals")]
use crate::keeper::BenchmarkQuillDirectorySyncState;
use crate::keeper::{
    CompactionPolicy, CompactionReport, GarbageCollectionOptions, GarbageCollectionReport,
    KeeperSnapshot,
};
#[cfg(any(test, feature = "bench-internals"))]
use crate::query::Query;
#[cfg(test)]
use crate::query::QueryDiagnostic;
use crate::schema::SchemaDescriptor;
use crate::scribe::DeltaFlushInput;
use crate::snippet::SnippetConfig;
use asupersync::Cx;
use asupersync::sync::{Mutex, OwnedMutexGuard};
#[cfg(feature = "bench-internals")]
use frankensearch_core::DocId;
#[cfg(feature = "bench-internals")]
use frankensearch_core::SearchError;
use frankensearch_core::{IndexableDocument, ScoredResult};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(test)]
use super::checkpoint::{
    QueryCheckpointHandle, prepare_snapshot_doc_freqs, query_work_upper_bound,
};
#[cfg(feature = "pruning-conformance")]
use super::conformance::ConformancePruningTraceReceipt;
#[cfg(feature = "conformance-internals")]
use super::conformance::{ConformanceCancellationController, ConformancePendingWriterState};
use super::error_results::{
    QuillDocumentWitness, QuillHit, QuillIndexError, QuillScoreExplanation, QuillSearchResult,
    QuillSnippetHit, check_cancel, invalid_state, map_lock_error_for_index, map_try_lock_error,
};
use super::ingest_doc::{IngestBatch, OwnedIngestBatch, SchemaDocument};
#[cfg(feature = "conformance-internals")]
use super::reader::PinnedSearchBundle;
use super::reader::QuillReader;
use super::snapshot::QuillSearchSnapshot;
#[cfg(test)]
use super::writer::IndexBackend;
use super::writer::QuillWriterState;

/// Shared Quill index handle with lock-free readers and one cancel-aware writer.
///
/// The same handle implements [`frankensearch_core::LexicalRead`] for progressive fusion and
/// exposes synchronous reader methods for direct and synchronous consumers.
///
/// ```no_run
/// # use asupersync::Cx;
/// # use frankensearch_core::IndexableDocument;
/// # use frankensearch_quill::{
/// #     CompactionPolicy, QuillConfig, QuillIndex, SegmentStatsProvider, SnippetConfig,
/// # };
/// # async fn use_quill(cx: &Cx) -> Result<(), Box<dyn std::error::Error>> {
/// let index = QuillIndex::in_memory(QuillConfig::default())?;
/// index
///     .index_document(cx, &IndexableDocument::new("guide", "native lexical search"))
///     .await?;
/// index.commit(cx).await?;
///
/// assert!(index.path().is_none());
/// assert_eq!(index.doc_count()?, 1);
/// let page = index.search_paginated(cx, "lexical", 10, 0, true)?;
/// let full = index.search_results(cx, "lexical", 10)?;
/// let ids = index.search_doc_ids(cx, "lexical", 10)?;
/// let enriched = index.search_with_snippets(cx, "lexical", 10, &SnippetConfig::default())?;
/// let stats = index.segment_stats()?;
/// assert_eq!(page.total_count, Some(1));
/// assert_eq!(full.len(), ids.len());
/// assert_eq!(enriched.len(), ids.len());
/// assert_eq!(stats.live_docs, 1);
///
/// let _ = index.compact(cx, CompactionPolicy::default()).await?;
/// assert!(index.delete_document(cx, "guide").await?);
/// index.delete_all(cx).await?;
///
/// let path = "target/quill-doc-example";
/// let _created = QuillIndex::create(cx, path, QuillConfig::default()).await?;
/// let _opened = QuillIndex::open(cx, path, QuillConfig::default()).await?;
/// # Ok(())
/// # }
/// ```
pub struct QuillIndex {
    pub(super) reader: QuillReader,
    pub(super) writer: Arc<Mutex<QuillWriterState>>,
    pub(super) directory: Option<PathBuf>,
    #[cfg(feature = "bench-internals")]
    pub(super) benchmark_qg4_directory_sync_state: Option<Arc<BenchmarkQuillDirectorySyncState>>,
}

/// Live arm for one controlled QG-4 Quill MANIFEST directory-sync observation.
///
/// Construct this immediately before the timed benchmark commit and call
/// [`Self::finish`] after timing stops. It is neither serializable nor
/// cloneable, so a caller cannot replay it as a declarative receipt.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[derive(Debug)]
pub struct BenchmarkQuillDirectorySyncArm {
    pub(super) state: Arc<BenchmarkQuillDirectorySyncState>,
    nonce: u64,
    pub(super) finished: bool,
    #[cfg(test)]
    pub(super) fail_after_commit: bool,
}

/// Engine-observed result of one controlled QG-4 Quill commit.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkQuillDirectorySyncReceipt {
    /// On-disk root whose MANIFEST directory sync emitted the observation.
    pub root: PathBuf,
    /// Opaque session identity minted by the benchmark observer.
    pub session_nonce: u64,
    /// MANIFEST generation installed by the observed directory sync.
    pub manifest_generation: u64,
    /// Successful MANIFEST directory-sync calls observed in the armed session.
    pub observed_sync_count: u8,
}

#[cfg(feature = "bench-internals")]
impl BenchmarkQuillDirectorySyncArm {
    /// Finalize the receipt outside the timed interval.
    ///
    /// # Errors
    ///
    /// Returns [`QuillIndexError::Config`] when the armed session cannot show
    /// exactly one proven MANIFEST directory sync: the timed commit returned
    /// without an observed sync, more than one sync occurred while the session
    /// was armed, the session was replaced or cleared before finalization, the
    /// commit was never completed through the benchmark commit API, the armed
    /// commit failed after its observed sync, or the successful sync did not
    /// stamp its MANIFEST generation.
    pub fn finish(&mut self) -> Result<BenchmarkQuillDirectorySyncReceipt, QuillIndexError> {
        self.finished = true;
        let (root, manifest_generation) = self
            .state
            .finish(self.nonce)
            .map_err(QuillIndexError::Config)?;
        Ok(BenchmarkQuillDirectorySyncReceipt {
            root,
            session_nonce: self.nonce,
            manifest_generation,
            observed_sync_count: 1,
        })
    }
}

#[cfg(feature = "bench-internals")]
impl Drop for BenchmarkQuillDirectorySyncArm {
    fn drop(&mut self) {
        if !self.finished {
            self.state.disarm(self.nonce);
        }
    }
}

impl QuillIndex {
    pub(super) fn from_writer(writer: QuillWriterState) -> Self {
        #[cfg(feature = "bench-internals")]
        let mut writer = writer;
        let directory = writer.directory().map(Path::to_path_buf);
        #[cfg(feature = "bench-internals")]
        let benchmark_qg4_directory_sync_state = directory.as_ref().map(|root| {
            let state = Arc::new(BenchmarkQuillDirectorySyncState::new(root.clone()));
            writer
                .backend
                .bind_benchmark_qg4_directory_sync_state(Arc::clone(&state));
            state
        });
        let reader = writer.reader.clone();
        Self {
            reader,
            writer: Arc::new(Mutex::with_name("quill.index.writer", writer)),
            directory,
            #[cfg(feature = "bench-internals")]
            benchmark_qg4_directory_sync_state,
        }
    }

    #[cfg(test)]
    pub(super) fn from_backend(
        backend: IndexBackend,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(QuillWriterState::from_backend(
            backend, schema, config,
        )?))
    }

    #[cfg(test)]
    pub(super) fn writer_mut(&mut self) -> &mut QuillWriterState {
        Arc::get_mut(&mut self.writer)
            .expect("exclusive test index owns its writer mutex")
            .get_mut()
            .expect("exclusive test index cannot have a poisoned writer")
    }

    pub(super) async fn lock_writer(
        &self,
        cx: &Cx,
        phase: &'static str,
    ) -> Result<OwnedMutexGuard<QuillWriterState>, QuillIndexError> {
        OwnedMutexGuard::lock(Arc::clone(&self.writer), cx)
            .await
            .map_err(|error| map_lock_error_for_index(error, phase))
    }

    #[cfg(test)]
    pub(super) fn execute_ranked_query(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        limit: usize,
        offset: usize,
        exact_count: bool,
        diagnostics: Vec<QueryDiagnostic>,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.reader.execute_ranked_query(
            cx,
            query,
            snapshot,
            limit,
            offset,
            exact_count,
            diagnostics,
        )
    }

    #[cfg(test)]
    pub(super) fn execute_docid_query(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<u32>, QuillIndexError> {
        self.reader.execute_docid_query(cx, query, snapshot)
    }

    #[cfg(test)]
    pub(super) fn collect_sealed_segments(
        &self,
        cx: &Cx,
        collector: &mut TopDocsCollector,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        rank_pruning: bool,
        topdocs_root: bool,
        fan_out: bool,
    ) -> Result<(), QuillIndexError> {
        let work_upper_bound = query_work_upper_bound(
            query,
            snapshot,
            self.reader.schema,
            self.reader.config.glob_expansion_limit,
        )?;
        let concrete_checkpoint = self.reader.query_checkpoint(
            cx,
            "search",
            self.reader.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        let prepared_doc_freqs = prepare_snapshot_doc_freqs(snapshot);
        self.reader.collect_sealed_segments(
            cx,
            &checkpoint,
            collector,
            query,
            snapshot,
            prepared_doc_freqs.as_ref(),
            rank_pruning,
            topdocs_root,
            #[cfg(feature = "pruning-conformance")]
            None,
            #[cfg(feature = "profile-internals")]
            None,
            fan_out && !metering,
        )
    }

    /// Create a shipping-schema index or open an existing compatible index.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::create(cx, directory, config).await?,
        ))
    }

    /// Open an existing shipping-schema index for mutation and search.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::open(cx, directory, config).await?,
        ))
    }

    /// Construct an owned-buffer genesis index without filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or parser failures.
    pub fn in_memory(config: QuillConfig) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(QuillWriterState::in_memory(config)?))
    }

    /// Build an owned-buffer index for one explicit compile-time schema.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or parser failures.
    #[cfg(feature = "bench-internals")]
    pub fn in_memory_with_schema(
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(QuillWriterState::in_memory_with_schema(
            schema, config,
        )?))
    }

    /// Create an on-disk index bound to an explicit compiled schema.
    ///
    /// [`Self::create`] pins [`crate::schema::DEFAULT_SCHEMA`]. Consumers that index a
    /// different compiled descriptor — `CASS_SEMANTIC_SCHEMA` being the one
    /// with a live consumer — need a durable index, not the bench-gated
    /// in-memory seam, so this is ordinary shipping API.
    ///
    /// # Errors
    ///
    /// Returns the same typed configuration, admission, and durability
    /// failures as [`Self::create`].
    pub async fn create_with_schema(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::create_with_schema(cx, directory, schema, config).await?,
        ))
    }

    /// Open an existing on-disk index bound to an explicit compiled schema.
    ///
    /// The descriptor is checked against the persisted `schema_id` rather than
    /// inferred, so opening a CASS index with the shipping descriptor is a
    /// typed schema failure instead of a silent misread.
    ///
    /// # Errors
    ///
    /// Returns the same typed configuration, admission, recovery, or schema
    /// failures as [`Self::open`].
    pub async fn open_with_schema(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::open_with_schema(cx, directory, schema, config).await?,
        ))
    }

    /// Bind an existing owned Keeper snapshot to the private writer facade.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or snapshot-validation failures.
    #[cfg(any(feature = "bench-internals", feature = "conformance-internals"))]
    pub fn from_in_memory_snapshot(
        snapshot: KeeperSnapshot,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::from_in_memory_snapshot(snapshot, config)?,
        ))
    }

    /// Return one exact process-local snapshot for a facade read.
    ///
    /// The reader validates the one pinned `Arc` between two `Acquire` loads
    /// of the shared packed publication state. It never treats the writer lock
    /// as an authority proxy: `Preparing` remains readable, while
    /// `DurableAhead` and `Indeterminate` refuse synchronously.
    pub(super) fn checked_published_snapshot(
        &self,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        self.reader.checked_published_snapshot()
    }

    /// Current authoritative immutable Keeper snapshot.
    ///
    /// # Errors
    ///
    /// Returns a typed reconciliation-required failure when publication may
    /// have advanced durable authority beyond this process-local view.
    pub fn snapshot(&self) -> Result<Arc<KeeperSnapshot>, QuillIndexError> {
        Ok(self.checked_published_snapshot()?.keeper_snapshot_arc())
    }

    /// Ranked page, expanded prefix, and exact count taken on ONE pinned
    /// publication, with that publication's physical digest.
    ///
    /// Conformance witnesses need the three observations to describe the
    /// same snapshot: three separate `search_paginated` calls may each load
    /// a different publication if a refresh lands between them. This runs all
    /// three against one loaded `QuillSearchSnapshot`. The count is a separate
    /// invocation on that snapshot and cannot affect hit selection, tie
    /// expansion, or ordering (bd-pjvl1).
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parsing, lowering, scoring, collection, or
    /// reconciliation-required failures.
    #[cfg(feature = "conformance-internals")]
    pub fn search_paginated_pinned(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        expanded_limit: usize,
    ) -> Result<PinnedSearchBundle, QuillIndexError> {
        self.reader
            .search_paginated_pinned(cx, query, limit, offset, expanded_limit)
    }

    /// Current authoritative process-local Keeper plus Delta snapshot.
    ///
    /// # Errors
    ///
    /// Returns a typed reconciliation-required failure when publication may
    /// have advanced durable authority beyond this process-local view.
    pub fn search_snapshot(&self) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        self.checked_published_snapshot()
    }

    /// Clone the deterministic real-`Cx` cancellation requester used by the
    /// method-bound conformance harness.
    ///
    /// This API and its per-index state do not exist in normal builds.
    #[cfg(feature = "conformance-internals")]
    #[must_use]
    pub fn conformance_cancellation_controller(&self) -> Arc<ConformanceCancellationController> {
        Arc::clone(&self.reader.conformance_controller)
    }

    /// Execute one ranked search with an invocation-bound pruning trace.
    ///
    /// # Errors
    ///
    /// Returns the ordinary typed query failure, or a typed conformance error
    /// if the exact invocation does not produce one dense receipt per sealed
    /// segment. Partial receipts are failed and discarded before return.
    #[cfg(feature = "pruning-conformance")]
    #[doc(hidden)]
    pub fn search_paginated_with_conformance_pruning_trace(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<(QuillSearchResult, ConformancePruningTraceReceipt), QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader
            .search_paginated_with_conformance_pruning_trace_on(
                cx,
                query,
                limit,
                offset,
                exact_count,
                snapshot.as_ref(),
            )
    }

    /// Capture a fixed-size digest receipt for the complete retained scalar
    /// writer transaction.
    ///
    /// This conformance-only surface hashes actual canonical pending FSLX
    /// bytes and identity rows rather than reporting only a dirty boolean.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the writer is busy or any retained MANIFEST
    /// proposal cannot be canonically encoded.
    #[cfg(feature = "conformance-internals")]
    pub fn conformance_pending_writer_state(
        &self,
    ) -> Result<ConformancePendingWriterState, QuillIndexError> {
        let writer = self.writer.try_lock().map_err(map_try_lock_error)?;
        writer.conformance_pending_writer_state()
    }

    /// Resolve one published document through IDHASH and return its IDMAP hash.
    ///
    /// The probe is allocation-free for sealed segments. It returns `None`
    /// when the identifier is absent or tombstoned.
    ///
    /// # Errors
    ///
    /// Returns typed identity corruption or an invalid live-Delta witness.
    pub fn document_witness(
        &self,
        document_id: &str,
    ) -> Result<Option<QuillDocumentWitness>, QuillIndexError> {
        self.search_snapshot()?
            .resolve_document_witness(document_id)
    }

    /// Probe the published ID map for one exact identifier in a benchmark-only
    /// terminal visibility check.
    ///
    /// This deliberately has the same exact-ID membership semantics as the
    /// retained Tantivy terminal-reader probe: it neither parses a user query
    /// nor scores a broader result set.  The caller owns the expected ID, so a
    /// live witness is returned as that one ID and an absent/tombstoned ID is
    /// returned as an empty result.
    ///
    /// # Errors
    ///
    /// Returns typed IDHASH/IDMAP corruption or an invalid live-Delta
    /// witness.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_search_exact_id(
        &self,
        document_id: &str,
    ) -> Result<Vec<DocId>, QuillIndexError> {
        Ok(self
            .document_witness(document_id)?
            .map(|_| vec![document_id.to_owned().into()])
            .unwrap_or_default())
    }

    /// Durable index directory, or `None` for an owned-buffer index.
    #[must_use]
    pub fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
    }

    /// Alias for [`Self::directory`] used by engine-neutral callers.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.directory()
    }

    /// Number of live documents in the published Keeper-plus-Delta view.
    ///
    /// # Errors
    ///
    /// Returns a typed reconciliation-required failure when publication
    /// authority cannot prove the process-local snapshot readable.
    pub fn doc_count(&self) -> Result<u64, QuillIndexError> {
        Ok(self.checked_published_snapshot()?.live_doc_count())
    }

    /// Whether the writer holds documents or installed segments not yet visible.
    #[must_use]
    pub fn has_uncommitted_changes(&self) -> bool {
        self.writer
            .try_lock()
            .map_or(true, |writer| writer.has_uncommitted_changes())
    }

    /// Atomically publish the complete process-local Delta table.
    ///
    /// # Errors
    ///
    /// Rejects schema or generation drift, overlapping leases, Keeper-range
    /// overlap, allocation failure, process-local epoch exhaustion, or a busy
    /// writer.
    pub fn publish_delta_table(
        &self,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let writer = self.writer.try_lock().map_err(map_try_lock_error)?;
        writer.publish_delta_table(deltas)
    }

    /// Seal one published Delta epoch into Keeper.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, writer-lock, validation, encoding, or
    /// publication failures.
    pub async fn seal_delta_snapshot(
        &self,
        cx: &Cx,
        sealed: Arc<DeltaSnapshot>,
        replacement_deltas: Vec<Arc<DeltaSnapshot>>,
        input: DeltaFlushInput,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "Delta seal writer lock").await?;
        writer
            .seal_delta_snapshot(cx, sealed, replacement_deltas, input)
            .await
    }

    /// Resume a retained Delta seal proposal.
    ///
    /// # Errors
    ///
    /// Returns a typed error when no seal awaits reconciliation or when the
    /// writer lock, segment installation, or publication fails.
    pub async fn resume_pending_delta_seal(
        &self,
        cx: &Cx,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let mut writer = self
            .lock_writer(cx, "Delta seal resume writer lock")
            .await?;
        writer.resume_pending_delta_seal(cx).await
    }

    /// Reconcile a possibly dropped durable publication under the writer lock.
    ///
    /// This reopens Keeper authority, installs the matching process-local
    /// composite, and only then releases strict public reads. A failed proof
    /// deliberately retains its operation-specific retry state.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, writer-lock, durable reopen, snapshot
    /// installation, or publication-authority validation failures.
    pub async fn reconcile_publication(
        &self,
        cx: &Cx,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let mut writer = self
            .lock_writer(cx, "publication reconciliation writer lock")
            .await?;
        writer.reconcile_publication(cx).await
    }

    /// Accumulate one document into the scalar writer.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, writer-lock, duplicate-id, accumulation,
    /// flush, or publication failures.
    pub async fn index_document(
        &self,
        cx: &Cx,
        document: &IndexableDocument,
    ) -> Result<(), QuillIndexError> {
        self.index_documents(cx, std::slice::from_ref(document))
            .await
    }

    /// Accumulate one bounded batch into the scalar writer.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, writer-lock, duplicate-id, accumulation,
    /// flush, or publication failures.
    pub async fn index_documents(
        &self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        let mut writer = self.lock_writer(cx, "index writer lock").await?;
        writer.index_documents(cx, documents).await
    }

    /// Accumulate one bounded batch already projected onto this index's
    /// compiled schema.
    ///
    /// This is the entry point for an index opened with
    /// [`Self::create_with_schema`], whose schema is wider than the five-field
    /// shipping shape [`IndexableDocument`] describes.
    ///
    /// # Errors
    ///
    /// Identical to [`Self::index_documents`].
    pub async fn index_schema_documents(
        &self,
        cx: &Cx,
        documents: &[SchemaDocument],
    ) -> Result<(), QuillIndexError> {
        let mut writer = self.lock_writer(cx, "index writer lock").await?;
        writer.index_schema_documents(cx, documents).await
    }

    /// Accumulate one bounded batch through the scalar writer without
    /// parallel-planner fanout for a conformance fixture.
    ///
    /// This proof-only seam does not alter shipping ingest: ordinary
    /// [`Self::index_documents`] keeps its adaptive shared-nothing and
    /// fixed-width internal fanout. The resulting leaf uses the same scalar
    /// accumulator, segment encoding, commit path, and query execution as a
    /// production scalar write; only ingest parallelization is suppressed.
    /// Scalar lease, arena-budget, visibility-publication, and tier-merge
    /// rules remain live, so callers must validate the realized leaf geometry
    /// before treating it as proof evidence.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, writer-lock, duplicate-ID, accumulation,
    /// flush, or publication failures.
    #[cfg(feature = "pruning-conformance")]
    #[doc(hidden)]
    pub async fn index_documents_with_scalar_topology_conformance(
        &self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        let mut writer = self
            .lock_writer(cx, "scalar topology conformance writer lock")
            .await?;
        writer
            .index_documents_with_scalar_topology_conformance(cx, documents)
            .await
    }

    /// Seal pending writes and atomically publish the next MANIFEST.
    ///
    /// # Errors
    ///
    /// Returns typed writer-lock, flush, segment-install, manifest-transition,
    /// durability, or cancellation failures.
    pub async fn commit(&self, cx: &Cx) -> Result<Arc<KeeperSnapshot>, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "commit writer lock").await?;
        writer.commit(cx).await?;
        drop(writer);
        self.snapshot()
    }

    /// Arm a MANIFEST directory-sync observation immediately before one QG-4 commit.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_arm_qg4_directory_sync(
        &self,
    ) -> Result<BenchmarkQuillDirectorySyncArm, QuillIndexError> {
        let state = self
            .benchmark_qg4_directory_sync_state
            .as_ref()
            .ok_or_else(|| {
                QuillIndexError::Config(SearchError::InvalidConfig {
                    field: "quill.qg4.directory_sync".to_owned(),
                    value: "not_on_disk_benchmark_index".to_owned(),
                    reason: "QG-4 directory-sync observation requires an on-disk Quill index"
                        .to_owned(),
                })
            })?
            .clone();
        let nonce = state.arm().map_err(QuillIndexError::Config)?;
        Ok(BenchmarkQuillDirectorySyncArm {
            state,
            nonce,
            finished: false,
            #[cfg(test)]
            fail_after_commit: false,
        })
    }

    /// Execute the public Quill commit path while an already-armed QG-4 observer watches its
    /// real MANIFEST directory sync.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub async fn benchmark_commit_qg4_directory_sync(
        &self,
        cx: &Cx,
        arm: &mut BenchmarkQuillDirectorySyncArm,
    ) -> Result<(), QuillIndexError> {
        if arm.finished {
            return Err(QuillIndexError::Config(SearchError::InvalidConfig {
                field: "quill.qg4.directory_sync".to_owned(),
                value: "foreign_or_disarmed_arm".to_owned(),
                reason: "a finalized QG-4 directory-sync arm cannot be reused".to_owned(),
            }));
        }
        let state = self
            .benchmark_qg4_directory_sync_state
            .as_ref()
            .ok_or_else(|| {
                QuillIndexError::Config(SearchError::InvalidConfig {
                    field: "quill.qg4.directory_sync".to_owned(),
                    value: "not_on_disk_benchmark_index".to_owned(),
                    reason: "QG-4 directory-sync observation requires an on-disk Quill index"
                        .to_owned(),
                })
            })?;
        if !Arc::ptr_eq(state, &arm.state) {
            return Err(QuillIndexError::Config(SearchError::InvalidConfig {
                field: "quill.qg4.directory_sync".to_owned(),
                value: "foreign_arm".to_owned(),
                reason: "the supplied QG-4 directory-sync arm belongs to another index".to_owned(),
            }));
        }

        match Self::commit(self, cx).await {
            Ok(_) => {
                #[cfg(test)]
                if arm.fail_after_commit {
                    arm.state.record_commit_failure(arm.nonce);
                    return Err(QuillIndexError::Config(SearchError::InvalidConfig {
                        field: "quill.qg4.directory_sync".to_owned(),
                        value: "test_failure_after_commit".to_owned(),
                        reason: "test-only failure after the real QG-4 commit completed".to_owned(),
                    }));
                }
                arm.state
                    .record_commit_success(arm.nonce)
                    .map_err(QuillIndexError::Config)
            }
            Err(error) => {
                arm.state.record_commit_failure(arm.nonce);
                Err(error)
            }
        }
    }

    /// Finish an explicitly configured bulk build with one final
    /// bound-consecutive concat pass and clear the durable bulk marker.
    ///
    /// Intermediate MANIFEST generations remain crash-resumable according to
    /// [`QuillConfig::bulk_publish_segment_cadence`]. After this succeeds the
    /// shard set has at most one sealed segment and ordinary tier merging is
    /// re-enabled for subsequent commits.
    ///
    /// # Errors
    ///
    /// Rejects a non-bulk writer and propagates writer-lock, seal, concat,
    /// publication, or cancellation failures.
    pub async fn finish_bulk_load(&self, cx: &Cx) -> Result<Arc<KeeperSnapshot>, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "bulk finish writer lock").await?;
        writer.finish_bulk_load(cx).await?;
        drop(writer);
        self.snapshot()
    }

    /// Replace one exact committed manifest run with a Q1 concat merge.
    ///
    /// # Errors
    ///
    /// Rejects writer-lock failure, uncommitted state, cancellation, a
    /// nonconsecutive source run, codec failure, or publication failure.
    pub async fn concat_merge(
        &self,
        cx: &Cx,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<Arc<KeeperSnapshot>, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "concat merge writer lock").await?;
        writer
            .concat_merge(cx, source_segment_ids, output_segment_id, created_unix_s)
            .await?;
        drop(writer);
        self.snapshot()
    }

    /// Fold tombstones into immutable positional holes for eligible segments.
    ///
    /// # Errors
    ///
    /// Rejects writer-lock failure, uncommitted scalar or Delta state, invalid
    /// policy values, cancellation, source/codec failures, or publication
    /// failures.
    pub async fn compact(
        &self,
        cx: &Cx,
        policy: CompactionPolicy,
    ) -> Result<CompactionReport, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "compaction writer lock").await?;
        writer.compact(cx, policy).await
    }

    /// Reclaim segment files no durable MANIFEST slot references.
    ///
    /// Runs the Keeper's writer-owned garbage sweep under the held admission.
    /// A file is unlinked only when neither `MANIFEST` nor `MANIFEST.prev`
    /// references it and a full grace period has elapsed since the
    /// publication that retired it (its durable retirement receipt), so a
    /// reader still holding the previous generation keeps working: it had the
    /// whole grace period to open its segments, and a segment it has already
    /// mapped survives the unlink. The same sweep runs once at open; this
    /// entry point lets a long-lived or frequently publishing index reclaim
    /// without reopening, since a merge's folded inputs become collectable
    /// only after later publications age them out. An in-memory index has
    /// nothing on disk to reclaim and reports an empty sweep.
    ///
    /// # Errors
    ///
    /// Rejects writer-lock failure, uncommitted state, cancellation, or the
    /// Keeper's typed sweep failure.
    pub async fn collect_garbage(
        &self,
        cx: &Cx,
    ) -> Result<GarbageCollectionReport, QuillIndexError> {
        let mut writer = self
            .lock_writer(cx, "garbage collection writer lock")
            .await?;
        writer.collect_garbage(cx, None).await
    }

    /// Reclaim unreferenced segment files with an explicit grace window.
    ///
    /// Identical to [`Self::collect_garbage`] except for the window. Shorten
    /// it only for a directory no reader can have resolved, such as a private
    /// build directory that has never been published: the window is what
    /// protects a reader that read an older `MANIFEST` and has not opened its
    /// segments yet. Files either `MANIFEST` slot references are kept at any
    /// window.
    ///
    /// # Errors
    ///
    /// Rejects writer-lock failure, uncommitted state, cancellation, or the
    /// Keeper's typed sweep failure.
    pub async fn collect_garbage_with(
        &self,
        cx: &Cx,
        options: GarbageCollectionOptions,
    ) -> Result<GarbageCollectionReport, QuillIndexError> {
        let mut writer = self
            .lock_writer(cx, "garbage collection writer lock")
            .await?;
        writer.collect_garbage(cx, Some(options)).await
    }

    /// Delete one live document id and publish the successor snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed writer-lock, cancellation, identity-resolution, or
    /// successor-publication failures.
    pub async fn delete_document(
        &self,
        cx: &Cx,
        document_id: &str,
    ) -> Result<bool, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "delete document writer lock").await?;
        writer.delete_document(cx, document_id).await
    }

    /// Delete a bounded batch of live document IDs and publish one successor
    /// snapshot.
    ///
    /// IDs that are already absent are ignored. Publishing once for the batch
    /// keeps bulk tombstone setup from paying one durable MANIFEST transition
    /// per ID while preserving the same atomic visibility boundary.
    ///
    /// # Errors
    ///
    /// Returns typed writer-lock, cancellation, identity-resolution, or
    /// successor-publication failures.
    pub async fn delete_documents(
        &self,
        cx: &Cx,
        document_ids: &[&str],
    ) -> Result<usize, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "delete documents writer lock").await?;
        writer.delete_documents(cx, document_ids).await
    }

    /// Delete every live document and publish an empty successor snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed writer-lock, cancellation, or successor-publication
    /// failures.
    pub async fn delete_all(&self, cx: &Cx) -> Result<(), QuillIndexError> {
        let mut writer = self.lock_writer(cx, "delete all writer lock").await?;
        writer.delete_all(cx).await
    }

    pub(super) async fn upsert_documents(
        &self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        self.upsert_documents_batch(cx, IngestBatch::Default(documents))
            .await
    }

    /// Upsert one bounded batch already projected onto this index's schema.
    ///
    /// # Errors
    ///
    /// Identical to `Self::upsert_documents`.
    pub async fn upsert_schema_documents(
        &self,
        cx: &Cx,
        documents: &[SchemaDocument],
    ) -> Result<(), QuillIndexError> {
        self.upsert_documents_batch(cx, IngestBatch::Schema(documents))
            .await
    }

    pub(super) async fn upsert_documents_batch(
        &self,
        cx: &Cx,
        documents: IngestBatch<'_>,
    ) -> Result<(), QuillIndexError> {
        check_cancel(cx, "upsert documents")?;
        let mut final_positions = HashMap::new();
        final_positions
            .try_reserve(documents.len())
            .map_err(|_| invalid_state("could not reserve upsert identity map"))?;
        for (position, document) in documents.iter().enumerate() {
            check_cancel(cx, "upsert documents")?;
            final_positions.insert(document.id(), position);
        }

        let canonical = if final_positions.len() == documents.len() {
            None
        } else {
            let mut canonical =
                OwnedIngestBatch::with_capacity_like(documents, final_positions.len())?;
            for (position, document) in documents.iter().enumerate() {
                check_cancel(cx, "upsert documents")?;
                if final_positions.get(document.id()) == Some(&position) {
                    canonical.push(document)?;
                }
            }
            Some(canonical)
        };
        drop(final_positions);

        let documents = canonical
            .as_ref()
            .map_or(documents, OwnedIngestBatch::as_batch);
        let mut writer = self.lock_writer(cx, "upsert writer lock").await?;
        writer.upsert_documents_batch(cx, documents).await
    }

    /// Parse and exhaustively execute one query over the published snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, section-validation,
    /// scorer, collector, or allocation failures.
    pub fn search_paginated(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader
            .search_paginated_on(cx, query, limit, offset, exact_count, snapshot.as_ref())
    }

    /// Search for full lexical results with canonical stored metadata.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, stored-metadata, or
    /// allocation failures.
    pub fn search_results(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredResult>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader
            .scored_results_on(cx, query, limit, true, snapshot.as_ref())
    }

    /// Search the identifier-only lane used by hot lexical consumers.
    ///
    /// The returned immutable page can be cloned without copying document IDs
    /// and remains valid across later snapshot publication.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, section-validation,
    /// scorer, collector, or allocation failures.
    pub fn search_doc_ids(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Arc<[QuillHit]>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        Ok(self
            .reader
            .search_paginated_on(cx, query, limit, 0, false, snapshot.as_ref())?
            .hits)
    }

    /// Search with the incumbent enriched result shape.
    ///
    /// Snippets use the same pinned snapshot, compiled query terms, global
    /// document frequencies, and stored source content as the ranked search.
    /// Schemas without stored content return `None` for that hit's snippet.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, stored-content,
    /// snippet-term compilation, UTF-8, or allocation failures.
    pub fn search_with_snippets(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
    ) -> Result<Vec<QuillSnippetHit>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader
            .search_with_snippets_on(cx, query, limit, snippet_config, snapshot.as_ref())
    }

    /// Generate snippets for chosen documents of the published snapshot; see
    /// [`super::QuillSearchIndex::snippets_for_documents`].
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query parsing, snippet-term compilation,
    /// stored-content, UTF-8, or allocation failures.
    pub fn snippets_for_documents(
        &self,
        cx: &Cx,
        query: &str,
        document_ids: &[&str],
        snippet_config: &SnippetConfig,
        max_source_bytes: usize,
    ) -> Result<Vec<Option<String>>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader.snippets_for_documents_on(
            cx,
            query,
            document_ids,
            snippet_config,
            max_source_bytes,
            snapshot.as_ref(),
        )
    }

    /// Split one document's BM25 score by clause and field on the published
    /// snapshot; see [`super::QuillSearchIndex::explain_document`].
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parsing, lowering, scoring, or dictionary
    /// failures.
    pub fn explain_document(
        &self,
        cx: &Cx,
        query: &str,
        document_id: &str,
    ) -> Result<Option<QuillScoreExplanation>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader
            .explain_document_on(cx, query, document_id, snapshot.as_ref())
    }

    /// Collect the complete deterministic set of matching global document IDs.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, cursor, or allocation
    /// failures.
    pub fn collect_docids(&self, cx: &Cx, query: &str) -> Result<Vec<u32>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader.collect_docids_on(cx, query, snapshot.as_ref())
    }

    /// Execute one already-built query tree through the ranked mixed-state path.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, cursor, or allocation failures.
    #[cfg(feature = "bench-internals")]
    pub fn search_preparsed_paginated(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader.search_preparsed_paginated_on(
            cx,
            query,
            limit,
            offset,
            exact_count,
            snapshot.as_ref(),
        )
    }

    /// Execute one already-built query tree through the scoreless id-set path.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, or allocation failures.
    #[cfg(feature = "bench-internals")]
    pub fn collect_preparsed_docids(
        &self,
        cx: &Cx,
        query: &Query,
    ) -> Result<Vec<u32>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader
            .collect_preparsed_docids_on(cx, query, snapshot.as_ref())
    }

    /// Bench-only forced sealed fan-out path for scoreless id-set collection.
    ///
    /// # Errors
    ///
    /// Returns the same typed parsing, lowering, collection, and cancellation
    /// failures as [`Self::collect_docids`].
    #[cfg(feature = "bench-internals")]
    pub fn bench_collect_docids_forced(
        &self,
        cx: &Cx,
        query: &str,
        fan_out: bool,
    ) -> Result<Vec<u32>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader
            .bench_collect_docids_forced_on(cx, query, fan_out, snapshot.as_ref())
    }

    /// Bench-only forced sealed fan-out path.
    ///
    /// # Errors
    ///
    /// Returns the same typed parsing, lowering, search, and cancellation
    /// failures as [`Self::search`].
    #[cfg(feature = "bench-internals")]
    pub fn bench_search_sealed_forced(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        fan_out: bool,
        rank_pruning: Option<bool>,
    ) -> Result<Vec<(u32, u32)>, QuillIndexError> {
        let snapshot = self.checked_published_snapshot()?;
        self.reader.bench_search_sealed_forced_on(
            cx,
            query,
            limit,
            fan_out,
            rank_pruning,
            snapshot.as_ref(),
        )
    }
}
