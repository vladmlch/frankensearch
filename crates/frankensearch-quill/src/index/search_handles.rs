use crate::config::QuillConfig;
use crate::keeper::{
    BlueGreenEngine, CurrentPointer, KeeperSnapshot, LexicalLayout, PublicationReadState,
    inspect_lexical_layout, load_manifest_pair,
};
use crate::query::{DefaultQueryParser, Query, QueryParserConfigError};
use crate::schema::{DEFAULT_SCHEMA, SchemaDescriptor};
use crate::snippet::SnippetConfig;
use crate::stats::{SegmentStats, SegmentStatsProvider};
use arc_swap::ArcSwap;
use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
#[cfg(test)]
use asupersync::runtime::yield_now;
use frankensearch_core::{
    IndexableDocument, LexicalCandidateBatch, LexicalHydrationContext, LexicalRead, LexicalWrite,
    ScoredResult, SearchError, SearchFuture, SearchResult,
};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[cfg(feature = "conformance-internals")]
use super::conformance::{ConformanceCancellationController, ConformanceCancellationStage};
use super::error_results::{
    QuillHit, QuillIndexError, QuillScoreExplanation, QuillSearchResult, QuillSnippetHit,
    check_cancel, invalid_state, validate_config, validate_non_durable_quarantine,
};
#[cfg(feature = "profile-internals")]
use super::profile::QuillProfiledSearchOutcome;
use super::quill_index::QuillIndex;
#[cfg(feature = "conformance-internals")]
use super::reader::PinnedSearchBundle;
use super::reader::QuillReader;
#[cfg(all(feature = "bench-internals", feature = "conformance-internals"))]
use super::reader::pinned_snapshot_sha256;
use super::snapshot::{QuillSearchSnapshot, SnapshotPublisher};

/// Read-only Quill handle for consumers that must coexist with another
/// process holding the durable writer lease.
///
/// Opening this handle pins the currently published MANIFEST without creating
/// or acquiring `LOCK`. A fresh handle therefore observes the latest
/// cross-process publication while a live watcher remains the sole writer.
#[derive(Clone)]
pub struct QuillSearchIndex {
    pub(super) reader: QuillReader,
    pub(super) directory: PathBuf,
}

/// Read-only Quill handle bound to a lexical root rather than one engine
/// directory.
///
/// A blue-green publication changes `<root>/CURRENT`, not the MANIFEST inside
/// a fixed child. This facade resolves and validates that pointer afresh on
/// every refresh, then atomically replaces the entire reader state. Searches
/// clone the active [`Arc`] before beginning, so a search already in flight
/// remains pinned to its prior engine generation.
#[derive(Clone)]
pub struct RootBoundQuillSearchIndex {
    lexical_root: PathBuf,
    pub(super) config: QuillConfig,
    pub(super) state: Arc<ArcSwap<RootBoundQuillReaderState>>,
    #[cfg(test)]
    pub(super) refresh_pause: Arc<RootRefreshPause>,
}

pub(super) struct RootBoundQuillReaderState {
    pub(super) reader: Arc<QuillSearchIndex>,
    pub(super) active_path: PathBuf,
    pub(super) pointer: Option<CurrentPointer>,
    pub(super) generation: u64,
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct RootRefreshPause {
    /// Which refresh stage to park at; `STAGE_NONE` disables the seam.
    pub(super) stage: AtomicU8,
    pub(super) arrived: AtomicBool,
    pub(super) release: AtomicBool,
}

#[cfg(test)]
impl RootRefreshPause {
    /// The refresh has not reached a paused stage.
    const STAGE_NONE: u8 = 0;
    /// Between the two layout-inspection cancellation checks, i.e. while the
    /// blocking `inspect_lexical_layout` result is in hand but nothing has been
    /// opened yet. This is the OPEN window.
    pub(super) const STAGE_INSPECTION: u8 = 1;
    /// After the replacement engine directory has been opened and before its
    /// generation is read, i.e. the VALIDATE window. A reader exists here but
    /// is not installed, so cancelling must drop it.
    pub(super) const STAGE_VALIDATION: u8 = 2;
    /// Immediately before the `ArcSwap` store. The SWAP window.
    pub(super) const STAGE_SWAP: u8 = 3;

    /// Park the refresh at `stage` until the paired canceller releases it.
    ///
    /// Cancellation windows are otherwise a race: a test that merely requests
    /// cancellation and hopes the refresh has reached the interesting
    /// checkpoint proves nothing on the runs where it has not. Parking makes
    /// the window deterministic, so the `LabRuntime` schedule that follows is
    /// reproducible rather than lucky.
    async fn wait_at(&self, stage: u8) {
        let armed = self.stage.load(Ordering::Acquire);
        if armed == Self::STAGE_NONE || armed != stage {
            return;
        }
        self.arrived.store(true, Ordering::Release);
        while !self.release.load(Ordering::Acquire) {
            yield_now().await;
        }
    }
}

/// Read-only in-memory handle for conformance schemas with their own parser.
///
/// This type intentionally exposes only typed-query execution. It neither
/// constructs Scribe writer shards nor offers string-query or mutation APIs,
/// so a schema whose analyzer and query-language contracts differ from the
/// shipping defaults cannot accidentally enter those paths.
#[cfg(feature = "bench-internals")]
#[derive(Clone)]
pub struct PreparsedQuillIndex {
    pub(super) reader: QuillReader,
}

#[cfg(feature = "bench-internals")]
impl PreparsedQuillIndex {
    /// Bind an owned Keeper snapshot for typed query execution.
    ///
    /// # Errors
    ///
    /// Returns typed configuration or snapshot-validation failures.
    pub fn from_in_memory_snapshot(
        snapshot: KeeperSnapshot,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let schema = snapshot.schema();
        let generation = snapshot.loaded_manifest().manifest.generation;
        let publication_read_state = PublicationReadState::new(generation)?;
        let published_snapshot = Arc::new(SnapshotPublisher::new(Arc::new(snapshot), Vec::new())?);
        Ok(Self {
            reader: QuillReader {
                config,
                schema,
                parser: None,
                published_snapshot,
                publication_read_state,
                #[cfg(test)]
                publication_read_pause: Arc::default(),
                #[cfg(feature = "conformance-internals")]
                conformance_controller: Arc::default(),
            },
        })
    }

    /// Execute one already-built query tree against the pinned snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed lowering, collection, cancellation, or allocation
    /// failures.
    pub fn search_preparsed_paginated(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.reader
            .search_preparsed_paginated(cx, query, limit, offset, exact_count)
    }

    /// Ranked page, expanded prefix, and exact count for one already-built
    /// query tree, all on ONE loaded publication with its physical digest.
    ///
    /// See [`QuillSearchIndex::search_paginated_pinned`]; this is the same
    /// contract for the compiled-schema (CASS) read path.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, lowering, scoring, or collection failures.
    #[cfg(feature = "conformance-internals")]
    pub fn search_preparsed_paginated_pinned(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        expanded_limit: usize,
    ) -> Result<PinnedSearchBundle, QuillIndexError> {
        let published = self.reader.published_snapshot.load();
        let observed = self.reader.search_preparsed_paginated_on(
            cx,
            query,
            limit,
            offset,
            false,
            published.as_ref(),
        )?;
        let expanded = self.reader.search_preparsed_paginated_on(
            cx,
            query,
            expanded_limit,
            0,
            false,
            published.as_ref(),
        )?;
        let count =
            self.reader
                .search_preparsed_paginated_on(cx, query, 0, 0, true, published.as_ref())?;
        Ok(PinnedSearchBundle {
            observed,
            expanded,
            count,
            snapshot_sha256: pinned_snapshot_sha256(published.as_ref()),
            snapshot_epoch: published.snapshot_epoch(),
            keeper_generation: published.keeper_generation(),
        })
    }

    /// Number of live documents in the pinned snapshot.
    #[must_use]
    pub fn doc_count(&self) -> u64 {
        self.reader.published_snapshot.load().live_doc_count()
    }
}

impl QuillSearchIndex {
    /// Open the latest published shipping-schema snapshot without acquiring the
    /// durable writer lease.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, configuration, recovery, schema, or parser
    /// failures.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        validate_non_durable_quarantine(&config)?;
        check_cancel(cx, "read-only index open")?;
        let directory = directory.into();
        let open_directory = directory.clone();
        let snapshot =
            spawn_blocking(move || KeeperSnapshot::open(open_directory, DEFAULT_SCHEMA)).await?;
        check_cancel(cx, "read-only index open")?;
        let generation = snapshot.loaded_manifest().manifest.generation;
        let publication_read_state = PublicationReadState::new(generation)?;
        let published_snapshot = Arc::new(SnapshotPublisher::new(Arc::new(snapshot), Vec::new())?);
        Ok(Self {
            reader: QuillReader {
                parser: Some(DefaultQueryParser::new(DEFAULT_SCHEMA)?),
                config,
                schema: DEFAULT_SCHEMA,
                published_snapshot,
                publication_read_state,
                #[cfg(test)]
                publication_read_pause: Arc::default(),
                #[cfg(feature = "conformance-internals")]
                conformance_controller: Arc::default(),
            },
            directory,
        })
    }

    /// Open the latest published snapshot of an index built on `schema`.
    ///
    /// The reader binds the shipping string parser only when `schema` can
    /// supply it a default field. A schema that cannot — the CASS profile,
    /// which parses through [`crate::query::CassQueryParser`] — opens without
    /// one and is read through
    /// [`Self::search_preparsed_paginated`]; the string entry points then
    /// return a typed parser failure rather than silently searching the wrong
    /// fields.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, configuration, recovery, schema, or parser
    /// failures.
    pub async fn open_with_schema(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        validate_non_durable_quarantine(&config)?;
        check_cancel(cx, "read-only index open")?;
        let directory = directory.into();
        let open_directory = directory.clone();
        let snapshot = spawn_blocking(move || KeeperSnapshot::open(open_directory, schema)).await?;
        check_cancel(cx, "read-only index open")?;
        let generation = snapshot.loaded_manifest().manifest.generation;
        let publication_read_state = PublicationReadState::new(generation)?;
        let published_snapshot = Arc::new(SnapshotPublisher::new(Arc::new(snapshot), Vec::new())?);
        let parser = match DefaultQueryParser::new(schema) {
            Ok(parser) => Some(parser),
            Err(
                QueryParserConfigError::MissingDefaultField { .. }
                | QueryParserConfigError::InvalidDefaultField { .. },
            ) => None,
            Err(error) => return Err(error.into()),
        };
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
            directory,
        })
    }

    /// Read one stored column for a live document in the pinned publication.
    ///
    /// See [`QuillSearchSnapshot::stored_field_value`]. This is the entry point
    /// a wider schema's consumer uses to read back the columns it stored, the
    /// way the shipping schema's consumers read `metadata`.
    ///
    /// # Errors
    ///
    /// Returns a typed reconciliation-required failure when publication
    /// authority cannot prove the process-local snapshot readable, or a
    /// section-validation failure from the segment.
    pub fn stored_field_value(
        &self,
        field_ord: u16,
        global_docid: u32,
    ) -> Result<Option<Vec<u8>>, QuillIndexError> {
        self.reader
            .checked_published_snapshot()?
            .stored_field_value(field_ord, global_docid)
    }

    /// Durable index directory bound to this published snapshot.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.directory
    }

    /// Live segment count in the proven-readable published snapshot.
    ///
    /// A consumer's compaction policy reads this to decide whether merging is
    /// worthwhile.
    ///
    /// # Errors
    ///
    /// Returns a typed reconciliation-required failure when publication
    /// authority cannot prove the process-local snapshot readable.
    pub fn segment_count(&self) -> Result<usize, QuillIndexError> {
        Ok(self
            .reader
            .checked_published_snapshot()?
            .keeper_snapshot()
            .segments()
            .len())
    }

    /// Number of live documents in the proven-readable published snapshot.
    ///
    /// # Errors
    ///
    /// Returns a typed reconciliation-required failure when publication
    /// authority cannot prove the process-local snapshot readable.
    pub fn doc_count(&self) -> Result<u64, QuillIndexError> {
        Ok(self.reader.checked_published_snapshot()?.live_doc_count())
    }

    /// Clone the deterministic real-`Cx` cancellation requester used by the
    /// method-bound conformance harness.
    ///
    /// This read-only conformance seam does not exist in normal builds.
    #[cfg(feature = "conformance-internals")]
    #[must_use]
    #[doc(hidden)]
    pub fn conformance_cancellation_controller(&self) -> Arc<ConformanceCancellationController> {
        Arc::clone(&self.reader.conformance_controller)
    }

    /// Durable MANIFEST generation pinned by the currently published snapshot.
    #[must_use]
    pub fn keeper_generation(&self) -> u64 {
        self.reader.published_snapshot.load().keeper_generation()
    }

    /// Refresh this read-only handle to the latest durable MANIFEST.
    ///
    /// Queries that already loaded the prior snapshot remain pinned to it;
    /// later queries observe the successor after the atomic publication swap.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, recovery, schema, or snapshot-transition
    /// failures. A regressed durable generation is rejected.
    pub async fn refresh(&self, cx: &Cx) -> Result<bool, QuillIndexError> {
        check_cancel(cx, "read-only index refresh")?;
        // Catch-up decides on the MANIFEST alone: an identical one installs
        // nothing. Reading the two small MANIFEST slots first spares the full
        // reopen, which authenticates every segment by hashing the whole file,
        // when nothing was published. Anything else takes the full path below,
        // so regressions and collisions keep their typed errors.
        let probe_directory = self.directory.clone();
        let on_disk = spawn_blocking(move || load_manifest_pair(probe_directory)).await;
        let published = self.reader.published_snapshot.load();
        if published.deltas.is_empty()
            && on_disk
                .is_ok_and(|loaded| loaded.manifest == published.keeper.loaded_manifest().manifest)
        {
            return Ok(false);
        }
        drop(published);
        let directory = self.directory.clone();
        // Refresh with the schema this reader was OPENED with, not the shipping
        // default: a reader bound to a wider schema (the CASS profile) would
        // otherwise fail every refresh with a manifest schema mismatch.
        let schema = self.reader.schema;
        let snapshot = spawn_blocking(move || KeeperSnapshot::open(directory, schema)).await?;
        check_cancel(cx, "read-only index refresh")?;

        Ok(self
            .reader
            .publish_durable_catch_up(&Arc::new(snapshot))?
            .is_some())
    }

    /// Execute a paginated ranked query against the pinned publication.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parsing, lowering, scoring, or collection
    /// failures.
    pub fn search_paginated(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.reader
            .search_paginated(cx, query, limit, offset, exact_count)
    }

    /// Ranked page, expanded prefix, and exact count taken on ONE pinned
    /// publication, with that publication's physical digest.
    ///
    /// See [`QuillSearchIndex::search_paginated_pinned`]; this is the same
    /// operation on the writer-side handle's reader.
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

    /// Execute one already-built query tree against the pinned publication.
    ///
    /// This is the read entry point for a compiled schema that carries its own
    /// query language — the CASS profile parses through
    /// [`crate::query::CassQueryParser`] — because such a schema has no
    /// default field for the shipping string parser to bind.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, lowering, scoring, or collection failures.
    pub fn search_preparsed_paginated(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let published = self.reader.published_snapshot.load();
        self.reader.search_preparsed_uncached_on(
            cx,
            query,
            limit,
            offset,
            exact_count,
            published.as_ref(),
        )
    }

    /// Execute one ordinary ranked search and retain a diagnostic sidecar receipt.
    ///
    /// The profile feature is intentionally separate from timing and
    /// conformance features. The returned error variant preserves the ordinary
    /// typed failure and its same-invocation receipt instead of fabricating a
    /// successful result.
    ///
    /// # Errors
    ///
    /// Returns a typed error only when profile admission or finalization itself
    /// fails; ordinary search failures are represented in the returned outcome.
    #[cfg(feature = "profile-internals")]
    #[doc(hidden)]
    pub fn search_paginated_with_profile(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillProfiledSearchOutcome, QuillIndexError> {
        self.reader
            .search_paginated_with_profile(cx, query, limit, offset, exact_count)
    }

    /// Search for full lexical results with canonical stored metadata.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, metadata, or allocation
    /// failures.
    pub fn search_results(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredResult>, QuillIndexError> {
        self.reader.scored_results(cx, query, limit, true)
    }

    /// Search the identifier-only lane and return an immutable shared hit page.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parsing, scoring, or collection failures.
    pub fn search_doc_ids(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Arc<[QuillHit]>, QuillIndexError> {
        self.reader.search_doc_ids(cx, query, limit)
    }

    /// Search the pinned publication and generate snippets from stored content.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, stored-content, snippet,
    /// UTF-8, or allocation failures.
    pub fn search_with_snippets(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
    ) -> Result<Vec<QuillSnippetHit>, QuillIndexError> {
        self.reader
            .search_with_snippets(cx, query, limit, snippet_config)
    }

    /// Generate snippets for chosen documents of the pinned publication.
    ///
    /// Uses the query terms, global document frequencies and analyzer that
    /// [`Self::search_with_snippets`] uses, so a document it returns gets the
    /// same snippet here — but only the listed documents are materialized, so
    /// a caller that ranks with [`Self::search_doc_ids`] pays for snippets
    /// only on what it shows. At most `max_source_bytes` of each stored text
    /// are scanned (cut on a character boundary). A document holding no query
    /// term gets its opening fragment; one that is not live gets `None`.
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
        self.reader.snippets_for_documents(
            cx,
            query,
            document_ids,
            snippet_config,
            max_source_bytes,
        )
    }

    /// Split one document's BM25 score for `query` by clause and field.
    ///
    /// Each term, phrase and glob clause of the parsed query is scored on its
    /// own against `document_id` by the evaluator a search runs, so the parts
    /// add up to the score that search reports for the document. Returns
    /// `None` when the document is not live or `query` does not match it.
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
        let published = self.reader.published_snapshot.load();
        self.reader
            .explain_document_on(cx, query, document_id, published.as_ref())
    }

    /// Collect every matching global document id from the pinned publication.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parsing, lowering, cursor, or allocation
    /// failures.
    pub fn collect_docids(&self, cx: &Cx, query: &str) -> Result<Vec<u32>, QuillIndexError> {
        self.reader.collect_docids(cx, query)
    }
}

impl RootBoundQuillSearchIndex {
    /// Open a read-only Quill reader from a lexical root.
    ///
    /// Direct Quill roots remain readable for migration compatibility. A
    /// blue-green root must carry a published, Quill-backed `CURRENT`; an
    /// unadopted child is deliberately rejected because this read-only API
    /// never writes `CURRENT` on a caller's behalf.
    ///
    /// # Errors
    ///
    /// Returns typed layout, cancellation, recovery, schema, or parser
    /// failures. No writer lease is acquired and the lexical root is never
    /// mutated.
    pub async fn open(
        cx: &Cx,
        lexical_root: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        let lexical_root = lexical_root.into();
        let state = Self::open_state(cx, &lexical_root, &config).await?;
        Ok(Self {
            lexical_root,
            config,
            state: Arc::new(ArcSwap::from_pointee(state)),
            #[cfg(test)]
            refresh_pause: Arc::default(),
        })
    }

    /// Lexical root whose `CURRENT` pointer is followed on refresh.
    #[must_use]
    pub fn lexical_root(&self) -> &Path {
        &self.lexical_root
    }

    /// Active Quill engine directory currently pinned by this reader.
    #[must_use]
    pub fn active_path(&self) -> PathBuf {
        self.state.load().active_path.clone()
    }

    /// Durable MANIFEST generation currently pinned by this reader.
    #[must_use]
    pub fn keeper_generation(&self) -> u64 {
        self.state.load().generation
    }

    /// Refresh this reader from the lexical root's current layout.
    ///
    /// The replacement is fully inspected and opened before the final
    /// cancellation check and one atomic state swap. Therefore cancellation,
    /// a malformed `CURRENT`, a missing engine directory, or an invalid Quill
    /// snapshot always leave the previous reader live for in-flight and later
    /// searches.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, layout, recovery, schema, or
    /// snapshot-transition failures. A generation regression within the same
    /// active engine directory is rejected.
    pub async fn refresh(&self, cx: &Cx) -> Result<bool, QuillIndexError> {
        check_cancel(cx, "root-bound reader refresh")?;
        let replacement = Self::open_state_inner(
            cx,
            &self.lexical_root,
            &self.config,
            #[cfg(test)]
            Some(self.refresh_pause.as_ref()),
        )
        .await?;
        self.install_validated_replacement(cx, replacement).await
    }

    /// Install one replacement that has already passed root-layout and Quill
    /// snapshot validation. Keeping this visibility boundary separate makes
    /// cancellation linearizable: no mutable reader state is touched before
    /// the final check immediately preceding the `ArcSwap` store.
    #[allow(clippy::unused_async_trait_impl)]
    pub(super) async fn install_validated_replacement(
        &self,
        cx: &Cx,
        replacement: RootBoundQuillReaderState,
    ) -> Result<bool, QuillIndexError> {
        #[cfg(test)]
        self.refresh_pause
            .wait_at(RootRefreshPause::STAGE_SWAP)
            .await;

        check_cancel(cx, "root-bound reader refresh swap")?;
        let current = self.state.load_full();
        if replacement.active_path == current.active_path {
            if replacement.generation < current.generation {
                return Err(invalid_state(format!(
                    "root-bound refresh regressed Quill generation from {} to {} at {}",
                    current.generation,
                    replacement.generation,
                    current.active_path.display()
                )));
            }
            if replacement.generation == current.generation
                && replacement.pointer == current.pointer
            {
                return Ok(false);
            }
        }

        self.state.store(Arc::new(replacement));
        Ok(true)
    }

    pub(super) async fn open_state(
        cx: &Cx,
        lexical_root: &Path,
        config: &QuillConfig,
    ) -> Result<RootBoundQuillReaderState, QuillIndexError> {
        Self::open_state_inner(
            cx,
            lexical_root,
            config,
            #[cfg(test)]
            None,
        )
        .await
    }

    /// `open_state` with the test-only cancellation-window seam threaded in.
    ///
    /// The seam is a parameter rather than shared state because `open_state`
    /// is called both by `open` — where there is no reader to retain and no
    /// window worth parking in — and by `refresh`, which is the path whose
    /// cancellation behaviour the acceptance names.
    async fn open_state_inner(
        cx: &Cx,
        lexical_root: &Path,
        config: &QuillConfig,
        #[cfg(test)] pause: Option<&RootRefreshPause>,
    ) -> Result<RootBoundQuillReaderState, QuillIndexError> {
        check_cancel(cx, "root-bound reader layout inspection")?;
        let root_for_inspection = lexical_root.to_path_buf();
        let layout = spawn_blocking(move || inspect_lexical_layout(&root_for_inspection)).await?;
        #[cfg(test)]
        if let Some(pause) = pause {
            pause.wait_at(RootRefreshPause::STAGE_INSPECTION).await;
        }
        check_cancel(cx, "root-bound reader layout inspection")?;

        let (active_path, pointer) = match layout {
            LexicalLayout::DirectQuill => (lexical_root.to_path_buf(), None),
            LexicalLayout::BlueGreen {
                pointer,
                pointer_on_disk: true,
            } if pointer.engine() == BlueGreenEngine::Quill => {
                let active_path = pointer.engine_dir(lexical_root);
                (active_path, Some(pointer))
            }
            LexicalLayout::BlueGreen {
                pointer,
                pointer_on_disk: false,
            } => {
                return Err(invalid_state(format!(
                    "lexical root {} has an unadopted {} child {:?}; read-only open refuses to publish CURRENT",
                    lexical_root.display(),
                    pointer.engine().label(),
                    pointer.dir_name()
                )));
            }
            LexicalLayout::BlueGreen { pointer, .. } => {
                return Err(invalid_state(format!(
                    "lexical root {} CURRENT selects {}, but this reader only supports Quill",
                    lexical_root.display(),
                    pointer.engine().label()
                )));
            }
            layout => {
                return Err(invalid_state(format!(
                    "lexical root {} has {} layout; refusing to select an engine",
                    lexical_root.display(),
                    layout.label()
                )));
            }
        };

        let reader =
            Arc::new(QuillSearchIndex::open(cx, active_path.clone(), config.clone()).await?);
        #[cfg(test)]
        if let Some(pause) = pause {
            pause.wait_at(RootRefreshPause::STAGE_VALIDATION).await;
        }
        check_cancel(cx, "root-bound reader validation")?;
        let generation = reader.keeper_generation();
        Ok(RootBoundQuillReaderState {
            reader,
            active_path,
            pointer,
            generation,
        })
    }
}

impl SegmentStatsProvider for QuillIndex {
    fn segment_stats(&self) -> SearchResult<SegmentStats> {
        self.reader.segment_stats()
    }
}

impl SegmentStatsProvider for QuillSearchIndex {
    fn segment_stats(&self) -> SearchResult<SegmentStats> {
        self.reader.segment_stats()
    }
}

impl SegmentStatsProvider for RootBoundQuillSearchIndex {
    fn segment_stats(&self) -> SearchResult<SegmentStats> {
        let state = self.state.load_full();
        SegmentStatsProvider::segment_stats(state.reader.as_ref())
    }
}

/// Backend tag carried by Quill hydration contexts (bd-8nqz.1).
pub const QUILL_LEXICAL_BACKEND: &str = "quill";

/// Feature-only snapshot pin that carries the deterministic hydration
/// checkpoint alongside the immutable scoring generation.
#[cfg(feature = "conformance-internals")]
struct ConformanceHydrationPin {
    snapshot: Arc<QuillSearchSnapshot>,
    controller: Arc<ConformanceCancellationController>,
}

/// Shared generation-pinned candidate search for Quill readers (bd-8nqz.1).
fn quill_search_candidates<'a>(
    reader: &'a QuillReader,
    cx: &'a Cx,
    query: &'a str,
    limit: usize,
) -> SearchFuture<'a, LexicalCandidateBatch> {
    Box::pin(async move {
        let (results, snapshot) = reader
            .scored_results_pinned(cx, query, limit, false)
            .map_err(SearchError::from)?;
        #[cfg(not(feature = "conformance-internals"))]
        let context = LexicalHydrationContext::new(QUILL_LEXICAL_BACKEND, Box::new(snapshot));
        #[cfg(feature = "conformance-internals")]
        let context = LexicalHydrationContext::new(
            QUILL_LEXICAL_BACKEND,
            Box::new(ConformanceHydrationPin {
                snapshot,
                controller: Arc::clone(&reader.conformance_controller),
            }),
        );
        Ok(LexicalCandidateBatch::deferred(results, context))
    })
}

/// Shared pinned hydration for Quill readers (bd-8nqz.1): metadata is
/// materialized from the exact snapshot that scored the batch, never from a
/// newer published generation.
fn quill_hydrate_candidates<'a>(
    cx: &'a Cx,
    context: Option<&'a LexicalHydrationContext>,
    results: &'a mut [ScoredResult],
) -> SearchFuture<'a, ()> {
    Box::pin(async move {
        check_cancel(cx, "fusion metadata hydration").map_err(SearchError::from)?;
        let Some(context) = context else {
            return Err(SearchError::SubsystemError {
                subsystem: "quill.hydration",
                source: "deferred Quill candidates require their batch context; \
                         none was provided"
                    .into(),
            });
        };
        #[cfg(not(feature = "conformance-internals"))]
        let Some(snapshot) = context.downcast_ref::<Arc<QuillSearchSnapshot>>() else {
            return Err(SearchError::SubsystemError {
                subsystem: "quill.hydration",
                source: format!(
                    "hydration context from backend {:?} is not a Quill snapshot pin; \
                     refusing cross-engine hydration",
                    context.backend()
                )
                .into(),
            });
        };
        #[cfg(feature = "conformance-internals")]
        let Some(conformance_pin) = context.downcast_ref::<ConformanceHydrationPin>() else {
            return Err(SearchError::SubsystemError {
                subsystem: "quill.hydration",
                source: format!(
                    "hydration context from backend {:?} is not a Quill snapshot pin; \
                     refusing cross-engine hydration",
                    context.backend()
                )
                .into(),
            });
        };
        #[cfg(feature = "conformance-internals")]
        let snapshot = &conformance_pin.snapshot;
        for result in results
            .iter_mut()
            .filter(|result| result.lexical_score.is_some())
        {
            #[cfg(feature = "conformance-internals")]
            conformance_pin
                .controller
                .checkpoint(ConformanceCancellationStage::FusionHydration, cx);
            check_cancel(cx, "fusion metadata hydration").map_err(SearchError::from)?;
            let Some(global_docid) = snapshot
                .resolve_document_id(result.doc_id.as_str())
                .map_err(SearchError::from)?
            else {
                result.metadata = None;
                continue;
            };
            result.metadata = snapshot
                .materialize_metadata(global_docid)
                .map_err(SearchError::from)?;
        }
        Ok(())
    })
}

impl LexicalRead for QuillIndex {
    fn query_text<'q>(&self, query: &'q str) -> Cow<'q, str> {
        crate::query::escape_unknown_field_colons(query, self.reader.schema)
    }

    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let snapshot = self
                .checked_published_snapshot()
                .map_err(SearchError::from)?;
            self.reader
                .scored_results_on(cx, query, limit, true, snapshot.as_ref())
                .map_err(SearchError::from)
        })
    }

    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        Box::pin(async move {
            let snapshot = self
                .checked_published_snapshot()
                .map_err(SearchError::from)?;
            let results = self
                .reader
                .scored_results_on(cx, query, limit, false, snapshot.as_ref())
                .map_err(SearchError::from)?;
            #[cfg(not(feature = "conformance-internals"))]
            let context = LexicalHydrationContext::new(QUILL_LEXICAL_BACKEND, Box::new(snapshot));
            #[cfg(feature = "conformance-internals")]
            let context = LexicalHydrationContext::new(
                QUILL_LEXICAL_BACKEND,
                Box::new(ConformanceHydrationPin {
                    snapshot,
                    controller: Arc::clone(&self.reader.conformance_controller),
                }),
            );
            Ok(LexicalCandidateBatch::deferred(results, context))
        })
    }

    fn hydrate_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        quill_hydrate_candidates(cx, context, results)
    }

    fn doc_count(&self) -> SearchResult<usize> {
        let count = Self::doc_count(self).map_err(SearchError::from)?;
        usize::try_from(count).map_err(|_| SearchError::SubsystemError {
            subsystem: "quill",
            source: "published Quill document count does not fit usize".into(),
        })
    }
}

impl LexicalWrite for QuillIndex {
    fn index_document<'a>(
        &'a self,
        cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.upsert_documents(cx, std::slice::from_ref(doc))
                .await
                .map_err(SearchError::from)
        })
    }

    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.upsert_documents(cx, docs)
                .await
                .map_err(SearchError::from)
        })
    }

    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            Self::commit(self, cx)
                .await
                .map(drop)
                .map_err(SearchError::from)
        })
    }
}

/// The read-only reader implements ONLY [`LexicalRead`] — no writer lease is
/// ever acquired on a read-only open (bd-8nqz.1).
impl LexicalRead for QuillSearchIndex {
    fn query_text<'q>(&self, query: &'q str) -> Cow<'q, str> {
        crate::query::escape_unknown_field_colons(query, self.reader.schema)
    }

    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            self.reader
                .scored_results(cx, query, limit, true)
                .map_err(SearchError::from)
        })
    }

    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        quill_search_candidates(&self.reader, cx, query, limit)
    }

    fn hydrate_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        quill_hydrate_candidates(cx, context, results)
    }

    fn doc_count(&self) -> SearchResult<usize> {
        let count = Self::doc_count(self).map_err(SearchError::from)?;
        usize::try_from(count).map_err(|_| SearchError::SubsystemError {
            subsystem: "quill",
            source: "published Quill document count does not fit usize".into(),
        })
    }
}

/// The root-bound reader pins the complete active reader in each returned
/// future. A blue-green refresh may replace `CURRENT` concurrently, but that
/// cannot redirect an already-admitted search to a different engine.
impl LexicalRead for RootBoundQuillSearchIndex {
    fn query_text<'q>(&self, query: &'q str) -> Cow<'q, str> {
        LexicalRead::query_text(self.state.load_full().reader.as_ref(), query)
    }

    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        let reader = Arc::clone(&self.state.load_full().reader);
        Box::pin(async move { LexicalRead::search(reader.as_ref(), cx, query, limit).await })
    }

    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        let reader = Arc::clone(&self.state.load_full().reader);
        Box::pin(
            async move { LexicalRead::search_candidates(reader.as_ref(), cx, query, limit).await },
        )
    }

    fn hydrate_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        let reader = Arc::clone(&self.state.load_full().reader);
        Box::pin(async move {
            LexicalRead::hydrate_candidates(reader.as_ref(), cx, context, results).await
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        let state = self.state.load_full();
        LexicalRead::doc_count(state.reader.as_ref())
    }
}
