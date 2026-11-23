use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use asupersync::Cx;
use asupersync::sync::Mutex;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::SearchFuture;
use frankensearch_core::types::ScoredResult;
#[cfg(test)]
use tantivy::directory::Directory as _;
use tantivy::{Index, IndexWriter};
use tracing::debug;

use crate::benchmark::{
    BenchmarkDirectorySyncState, BenchmarkMaterializedWidth, BenchmarkRetainedTantivyReader,
    BenchmarkTantivyDirectorySyncArm, BenchmarkWriterAttestation, BenchmarkWriterJoinReceipt,
    BenchmarkWriterMode, BenchmarkWriterReceipt, ObservedMmapDirectory,
    reject_positions_disagreement, validate_benchmark_writer_threads,
};
#[cfg(test)]
use crate::index::WriterCall;
use crate::index::{TantivyIndex, WriterPlan, call_fixed_writer};
use crate::schema::{build_schema, build_schema_with_positions};

impl TantivyIndex {
    /// Return the writer-pool width successfully materialized by the
    /// benchmark-only Tantivy constructor.
    ///
    /// `None` means no width is authenticated, which is the honest answer for
    /// the shipping-auto path; it is never a stand-in for an inferred width.
    #[doc(hidden)]
    #[must_use]
    pub const fn benchmark_materialized_writer_threads(&self) -> Option<usize> {
        self.benchmark_writer_threads
    }

    /// Typed receipt for the benchmark writer this index was constructed with.
    #[doc(hidden)]
    #[must_use]
    pub const fn benchmark_writer_receipt(&self) -> Option<&BenchmarkWriterReceipt> {
        self.benchmark_writer_receipt.as_ref()
    }

    /// Take the one-shot live attestation for this benchmark writer.
    ///
    /// The attestation is minted only after the writer constructor succeeds.
    /// Taking it does not remove the descriptive receipt, so a later rearm can
    /// mint a fresh attestation for the replacement writer.
    #[doc(hidden)]
    pub fn take_benchmark_writer_attestation(&mut self) -> Option<BenchmarkWriterAttestation> {
        self.benchmark_writer_attestation.take()
    }

    /// Construct an in-memory oracle through Tantivy's **shipping** writer
    /// selection: `Index::writer(heap)` is called literally, and the resulting
    /// pool width is reported as unobservable rather than guessed.
    ///
    /// # Errors
    ///
    /// Returns the ordinary Tantivy writer-construction error, including the
    /// heap-too-small case, which stays fail-closed here.
    #[doc(hidden)]
    pub fn in_memory_with_shipping_auto_writer(
        writer_heap_bytes: usize,
        positions: bool,
    ) -> SearchResult<Self> {
        let (schema, fields) = build_schema_with_positions(positions);
        let index = Index::create_in_ram(schema.clone());
        reject_positions_disagreement(&index, positions)?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            None,
            writer_heap_bytes,
            WriterPlan::BenchmarkShippingAuto,
            None,
        )
    }

    /// Create an in-memory index with an explicit writer heap budget.
    ///
    /// This exists only for same-binary benchmark comparisons of Tantivy's
    /// writer parallelism. Shipping callers always use [`Self::in_memory`].
    #[doc(hidden)]
    pub fn in_memory_with_writer_heap_bytes(writer_heap_bytes: usize) -> SearchResult<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema.clone());
        Self::from_index(index, schema, fields, None, writer_heap_bytes)
    }

    /// Create an in-memory oracle with an explicitly pinned benchmark schema,
    /// writer count, and heap budget.
    ///
    /// This is the same shipping analyzer, document conversion, writer, and
    /// commit path used by [`LexicalSearch`]. Only the knobs that the Quill QG
    /// matrix must hold equal across engines are exposed. Shipping callers use
    /// [`Self::in_memory`] instead.
    ///
    /// # Errors
    ///
    /// Returns an invalid-config error for zero writer threads, or the ordinary
    /// Tantivy writer-construction error for an unsupported heap budget.
    #[doc(hidden)]
    pub fn in_memory_with_benchmark_config(
        writer_heap_bytes: usize,
        writer_threads: usize,
        positions: bool,
    ) -> SearchResult<Self> {
        validate_benchmark_writer_threads(writer_threads)?;
        let (schema, fields) = build_schema_with_positions(positions);
        let index = Index::create_in_ram(schema.clone());
        reject_positions_disagreement(&index, positions)?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            None,
            writer_heap_bytes,
            WriterPlan::BenchmarkFixed(writer_threads),
            None,
        )
    }

    /// Create an on-disk oracle with the same explicit benchmark pins as
    /// [`Self::in_memory_with_benchmark_config`].
    ///
    /// # Errors
    ///
    /// Returns a typed filesystem, invalid-config, or Tantivy construction
    /// error.
    #[doc(hidden)]
    pub fn create_with_benchmark_config(
        path: &Path,
        writer_heap_bytes: usize,
        writer_threads: usize,
        positions: bool,
    ) -> SearchResult<Self> {
        validate_benchmark_writer_threads(writer_threads)?;
        let (schema, fields) = build_schema_with_positions(positions);
        std::fs::create_dir_all(path).map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        })?;
        let benchmark_directory_sync_state = Arc::new(BenchmarkDirectorySyncState::new());
        let directory = ObservedMmapDirectory::open(path, benchmark_directory_sync_state.clone())
            .map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        })?;
        let index = Index::create(directory, schema.clone(), tantivy::IndexSettings::default())
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;
        reject_positions_disagreement(&index, positions)?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            writer_heap_bytes,
            WriterPlan::BenchmarkFixed(writer_threads),
            Some(benchmark_directory_sync_state),
        )
    }

    /// Reopen an on-disk oracle with the benchmark's pinned writer budget.
    ///
    /// # Errors
    ///
    /// Returns a typed missing-index, invalid-config, or Tantivy construction
    /// error.
    #[doc(hidden)]
    pub fn open_with_benchmark_config(
        path: &Path,
        writer_heap_bytes: usize,
        writer_threads: usize,
        positions: bool,
    ) -> SearchResult<Self> {
        validate_benchmark_writer_threads(writer_threads)?;
        if !path.exists() {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }
        let (schema, fields) = build_schema_with_positions(positions);
        let benchmark_directory_sync_state = Arc::new(BenchmarkDirectorySyncState::new());
        let directory = ObservedMmapDirectory::open(path, benchmark_directory_sync_state.clone())
            .map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        })?;
        let index = Index::open(directory).map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        })?;
        reject_positions_disagreement(&index, positions)?;
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            writer_heap_bytes,
            WriterPlan::BenchmarkFixed(writer_threads),
            Some(benchmark_directory_sync_state),
        )
    }

    /// Join every indexing and merge worker, then rearm the same benchmark index.
    ///
    /// The returned duration measures only Tantivy's
    /// [`IndexWriter::wait_merging_threads`] call. Constructing the replacement
    /// writer is excluded because it is harness bookkeeping, not incumbent
    /// maintenance. Searchable segment counts make the lifecycle boundary
    /// auditable without relying on timing variance.
    ///
    /// # Errors
    ///
    /// Returns an invalid writer-thread configuration, poisoned writer mutex,
    /// Tantivy worker/merge failure, segment metadata failure, or writer
    /// reconstruction failure.
    #[doc(hidden)]
    pub fn benchmark_join_workers_and_rearm(
        self,
        writer_heap_bytes: usize,
        writer_threads: usize,
    ) -> SearchResult<(Self, BenchmarkWriterJoinReceipt)> {
        validate_benchmark_writer_threads(writer_threads)?;
        let Self {
            index,
            fields,
            reader,
            writer,
            read_only: _,
            doc_count,
            ord_table,
            path,
            benchmark_writer_threads: _,
            benchmark_writer_receipt,
            benchmark_writer_attestation: _,
            benchmark_directory_sync_state,
            benchmark_qg4_no_merge_prepared: _,
            #[cfg(test)]
                observed_writer_call: _,
        } = self;
        let mut receipt = Self::benchmark_join_writer(&index, writer)?;
        // The rearm reconstructs through the same helper the fixed plan uses,
        // so its observation is the call it actually made rather than the one
        // the old writer had made.
        #[cfg(test)]
        let mut observed_writer_call = WriterCall::Auto;
        let writer = call_fixed_writer(
            &index,
            writer_threads,
            writer_heap_bytes,
            #[cfg(test)]
            &mut observed_writer_call,
        )
        .map_err(|error| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(error),
        })?;
        receipt.writer_rearmed = true;
        let benchmark_writer_receipt =
            benchmark_writer_receipt.map(|previous| BenchmarkWriterReceipt {
                mode: BenchmarkWriterMode::Fixed {
                    threads: writer_threads,
                },
                writer_heap_bytes,
                materialized_width: BenchmarkMaterializedWidth::Authenticated(writer_threads),
                writer_rearmed: true,
                ..previous
            });
        let benchmark_writer_attestation = benchmark_writer_receipt
            .as_ref()
            .map(|receipt| BenchmarkWriterAttestation::mint(receipt.clone()));
        Ok((
            Self {
                index,
                fields,
                reader,
                // The rearm always constructs a real writer, so this handle is
                // never read-only.
                writer: Mutex::new(Some(writer)),
                read_only: false,
                doc_count,
                ord_table,
                path,
                benchmark_writer_threads: Some(writer_threads),
                benchmark_writer_receipt,
                benchmark_writer_attestation,
                benchmark_directory_sync_state,
                benchmark_qg4_no_merge_prepared: AtomicBool::new(false),
                #[cfg(test)]
                observed_writer_call,
            },
            receipt,
        ))
    }

    /// Join every indexing and merge worker while retaining a read-only search
    /// handle and without constructing another writer.
    ///
    /// The returned handle is the only terminal benchmark surface capable of
    /// proving that a query returned after the actual worker join.  Segment
    /// metadata is retained in the receipt for diagnostics but is not a
    /// searchability proof.
    ///
    /// # Errors
    ///
    /// Returns a poisoned writer-mutex, Tantivy worker/merge, or segment
    /// metadata error.
    #[doc(hidden)]
    pub fn benchmark_join_workers_retaining_reader(
        self,
    ) -> SearchResult<(BenchmarkRetainedTantivyReader, BenchmarkWriterJoinReceipt)> {
        let Self {
            index,
            fields,
            reader,
            writer,
            read_only: _,
            doc_count,
            ord_table,
            path,
            benchmark_writer_threads,
            benchmark_writer_receipt,
            benchmark_writer_attestation: _,
            benchmark_directory_sync_state: _,
            benchmark_qg4_no_merge_prepared: _,
            #[cfg(test)]
                observed_writer_call: _,
        } = self;
        let receipt = Self::benchmark_join_writer(&index, writer)?;
        drop((doc_count, ord_table, path, benchmark_writer_threads));
        drop(benchmark_writer_receipt);
        Ok((BenchmarkRetainedTantivyReader { fields, reader }, receipt))
    }

    /// Join every indexing and merge worker without constructing another writer.
    ///
    /// This is the terminal lifecycle fence for one-shot bulk measurements. It
    /// consumes the benchmark index, waits for every incumbent background
    /// worker, and returns only the counted receipt. Callers that will perform
    /// subsequent writes must use [`Self::benchmark_join_workers_and_rearm`].
    ///
    /// # Errors
    ///
    /// Returns a poisoned writer-mutex, Tantivy worker/merge, or segment
    /// metadata error.
    #[doc(hidden)]
    pub fn benchmark_join_workers(self) -> SearchResult<BenchmarkWriterJoinReceipt> {
        let (_reader, receipt) = self.benchmark_join_workers_retaining_reader()?;
        Ok(receipt)
    }

    fn benchmark_join_writer(
        index: &Index,
        writer: Mutex<Option<IndexWriter>>,
    ) -> SearchResult<BenchmarkWriterJoinReceipt> {
        let searchable_segments_before = index
            .searchable_segment_ids()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?
            .len();
        let Some(writer) = writer
            .into_inner()
            .map_err(|error| Self::map_writer_lock_error("tantivy.benchmark_join", error))?
        else {
            // A read-only handle has no indexing workers to join, so it can
            // never be the subject of a writer-join benchmark.
            return Err(Self::read_only_writer_error("benchmark_join"));
        };
        let timer = std::time::Instant::now();
        writer
            .wait_merging_threads()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;
        let join_elapsed_ns = u64::try_from(timer.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let searchable_segments_after = index
            .searchable_segment_ids()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?
            .len();
        Ok(BenchmarkWriterJoinReceipt {
            searchable_segments_before,
            searchable_segments_after,
            join_elapsed_ns,
            writer_rearmed: false,
        })
    }

    /// Disable automatic segment merging for a force-merge benchmark setup.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or writer-lock error.
    #[doc(hidden)]
    pub async fn benchmark_disable_auto_merge(&self, cx: &Cx) -> SearchResult<()> {
        let writer_guard =
            self.writer.lock(cx).await.map_err(|error| {
                Self::map_writer_lock_error("tantivy.benchmark_no_merge", error)
            })?;
        let Some(writer) = writer_guard.as_ref() else {
            return Err(Self::read_only_writer_error("benchmark_no_merge"));
        };
        writer.set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));
        Ok(())
    }

    /// Prepare a fresh benchmark index for a causally bound QG-4 directory-sync observation.
    ///
    /// This must run before any document is indexed. It installs Tantivy's real
    /// `NoMergePolicy`, so the one armed commit cannot be confused with an
    /// asynchronous merge-completion metadata save.
    #[doc(hidden)]
    pub async fn benchmark_prepare_qg4_no_merge_directory_sync(&self, cx: &Cx) -> SearchResult<()> {
        if self.benchmark_directory_sync_state.is_none() {
            return Err(SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync".to_owned(),
                value: "not_on_disk_benchmark_index".to_owned(),
                reason: "QG-4 directory-sync observation requires an on-disk benchmark index"
                    .to_owned(),
            });
        }
        let writer_guard = self.writer.lock(cx).await.map_err(|error| {
            Self::map_writer_lock_error("tantivy.qg4.directory_sync_no_merge", error)
        })?;
        let Some(writer) = writer_guard.as_ref() else {
            return Err(Self::read_only_writer_error("qg4.directory_sync_no_merge"));
        };
        if self.doc_count.load(Ordering::Acquire) != 0 {
            return Err(SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync_no_merge".to_owned(),
                value: "documents_already_indexed".to_owned(),
                reason: "QG-4 no-merge setup must precede the warm fixture".to_owned(),
            });
        }
        writer.set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));
        self.benchmark_qg4_no_merge_prepared
            .store(true, Ordering::Release);
        debug!(
            has_shared_observer = true,
            "prepared QG-4 Tantivy no-merge fixture"
        );
        Ok(())
    }

    /// Arm a shared directory observer immediately before the timed QG-4 commit.
    ///
    /// Receipt construction is deliberately deferred to
    /// [`BenchmarkTantivyDirectorySyncArm::finish`], after the timed call
    /// returns its actual Tantivy opstamp.
    #[doc(hidden)]
    pub fn benchmark_arm_qg4_directory_sync(
        &self,
    ) -> SearchResult<BenchmarkTantivyDirectorySyncArm> {
        if !self.benchmark_qg4_no_merge_prepared.load(Ordering::Acquire) {
            return Err(SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync_no_merge".to_owned(),
                value: "not_prepared".to_owned(),
                reason:
                    "call benchmark_prepare_qg4_no_merge_directory_sync before warming the fixture"
                        .to_owned(),
            });
        }
        let state = self
            .benchmark_directory_sync_state
            .as_ref()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync".to_owned(),
                value: "not_on_disk_benchmark_index".to_owned(),
                reason: "QG-4 directory-sync observation requires an on-disk benchmark index"
                    .to_owned(),
            })?
            .clone();
        let root = self
            .path
            .clone()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync".to_owned(),
                value: "missing_index_root".to_owned(),
                reason: "the benchmark index has no on-disk root to bind into the receipt"
                    .to_owned(),
            })?;
        let nonce = state.arm()?;
        Ok(BenchmarkTantivyDirectorySyncArm {
            state,
            root,
            nonce,
            commit_opstamp: None,
            finished: false,
        })
    }

    /// Execute the same commit path as [`frankensearch_core::traits::LexicalWrite::commit`] while an already-armed
    /// QG-4 observer watches Tantivy's real directory sync.
    ///
    /// The only observation work in the timed commit is the fixed atomic state
    /// transition performed by the wrapped successful `sync_directory` call.
    /// The returned opstamp is stamped into the supplied arm and finalized into
    /// a receipt after timing.
    #[doc(hidden)]
    pub async fn benchmark_commit_qg4_directory_sync(
        &self,
        cx: &Cx,
        arm: &mut BenchmarkTantivyDirectorySyncArm,
    ) -> SearchResult<()> {
        let state = self
            .benchmark_directory_sync_state
            .as_ref()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync".to_owned(),
                value: "not_on_disk_benchmark_index".to_owned(),
                reason: "QG-4 directory-sync observation requires an on-disk benchmark index"
                    .to_owned(),
            })?;
        if !Arc::ptr_eq(state, &arm.state) {
            return Err(SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync".to_owned(),
                value: "foreign_arm".to_owned(),
                reason: "the supplied QG-4 directory-sync arm belongs to another index".to_owned(),
            });
        }

        let opstamp = {
            let mut writer_guard = self
                .writer
                .lock(cx)
                .await
                .map_err(|error| Self::map_writer_lock_error("tantivy.qg4.commit", error))?;
            let Some(writer) = writer_guard.as_mut() else {
                return Err(Self::read_only_writer_error("qg4.commit"));
            };
            writer
                .commit()
                .map_err(|error| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(error),
                })?
        };

        self.reader
            .reload()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;
        let searcher = self.reader.searcher();
        let actual = usize::try_from(searcher.num_docs()).unwrap_or(usize::MAX);
        self.doc_count.store(actual, Ordering::Relaxed);
        self.persist_ord_table();
        arm.commit_opstamp = Some(opstamp);
        debug!(doc_count = actual, opstamp, "QG-4 Tantivy commit completed");
        Ok(())
    }

    /// Drive a second real wrapped sync in a focused ambiguity test.
    #[cfg(test)]
    pub(super) fn benchmark_test_second_armed_directory_sync(&self) -> SearchResult<()> {
        let state = self
            .benchmark_directory_sync_state
            .as_ref()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync".to_owned(),
                value: "not_on_disk_benchmark_index".to_owned(),
                reason: "QG-4 directory-sync observation requires an on-disk benchmark index"
                    .to_owned(),
            })?
            .clone();
        let root = self
            .path
            .as_deref()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync".to_owned(),
                value: "missing_index_root".to_owned(),
                reason: "the benchmark index has no on-disk root to bind into the receipt"
                    .to_owned(),
            })?;
        ObservedMmapDirectory::open(root, state)
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?
            .sync_directory()
            .map_err(SearchError::from)
    }

    /// Force-merge every currently searchable segment and reload the reader.
    ///
    /// This exists only for QG-5's same-binary oracle arm. It invokes
    /// Tantivy's real merge machinery and waits for the merge future; no
    /// benchmark-only codec or document path is involved.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation, writer-lock, merge, or reader-reload
    /// error.
    #[doc(hidden)]
    pub async fn benchmark_force_merge(&self, cx: &Cx) -> SearchResult<()> {
        let segment_ids =
            self.index
                .searchable_segment_ids()
                .map_err(|error| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(error),
                })?;
        if segment_ids.len() < 2 {
            return Ok(());
        }
        {
            let mut writer_guard = self
                .writer
                .lock(cx)
                .await
                .map_err(|error| Self::map_writer_lock_error("tantivy.force_merge", error))?;
            let Some(writer) = writer_guard.as_mut() else {
                return Err(Self::read_only_writer_error("force_merge"));
            };
            writer
                .merge(&segment_ids)
                .wait()
                .map_err(|error| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(error),
                })?;
        }
        self.reader
            .reload()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;
        Ok(())
    }

    /// Return `(searchable_segments, managed_index_bytes)` for index-build benchmarks.
    #[doc(hidden)]
    pub fn benchmark_index_layout(&self) -> SearchResult<(usize, u64)> {
        let usage =
            self.reader
                .searcher()
                .space_usage()
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;
        let segment_count = usage.segments().len();
        let index_bytes = usage.total().get_bytes();
        Ok((segment_count, index_bytes))
    }

    /// Scored-collector baseline retained for the exact hydration A/B benchmark.
    #[doc(hidden)]
    pub fn hydrate_fusion_metadata_scored_for_bench<'a>(
        &'a self,
        _cx: &'a Cx,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        // The scored-collector baseline deliberately reads the CURRENT
        // generation, matching what the retired combined-trait path did, so the
        // A/B keeps comparing collectors rather than pinning strategies.
        Box::pin(async move { self.hydrate_with_searcher(&self.reader.searcher(), results, false) })
    }

    /// Unscored-collector arm of the same A/B.
    ///
    /// The shipping path reaches this through `hydrate_candidates` with a
    /// pinned searcher; the benchmark needs it without a batch in hand, and on
    /// the same current-generation footing as the scored baseline above, so
    /// the measured difference stays the collector and nothing else.
    #[doc(hidden)]
    pub fn hydrate_fusion_metadata_unscored_for_bench<'a>(
        &'a self,
        _cx: &'a Cx,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move { self.hydrate_with_searcher(&self.reader.searcher(), results, true) })
    }
}
