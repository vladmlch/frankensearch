//! Index refresh worker (asupersync background task).
//!
//! [`RefreshWorker`] periodically leases jobs from the [`EmbeddingQueue`],
//! embeds documents in batches, and rebuilds the vector index. It runs as an
//! asupersync task within a structured concurrency region.
//!
//! # Single-writer guarantee
//!
//! The worker is the **only** component that writes to vector indices. All
//! reads go through the [`IndexCache`] which provides
//! atomic snapshot replacement.
//!
//! # Identity-bound refresh (bd-9xuj T2 C4-write)
//!
//! Every embedding harvested by this worker is identity-bound at the embedder
//! boundary ([`frankensearch_core::traits::Embedder::embed_batch_bound`]) and carried as a
//! [`frankensearch_core::BoundQueryEmbedding`]; no raw vector enters a `RefreshRecord`. Two
//! lanes exist:
//!
//! - **Canonical bootstrap lane** ([`RefreshWorker::run_cycle`]): admissible
//!   only while the canonical generation retains nothing (missing artifacts
//!   or an empty legacy v1 seed). It publishes through the legacy two-tier
//!   writer with the producing identity declared process-locally
//!   (`TwoTierIndexBuilder::set_*_identity`) — DECLARED, never attested.
//! - **Staged identity-bound replacement**
//!   ([`RefreshWorker::stage_identity_bound_generation`]): the typed merge
//!   that replaces the former blanket refusal. It admits the existing
//!   generation's ATTESTED FSVI v2 identity through
//!   [`frankensearch_index::VectorIndex::open_admitted_v2`] (plain [`frankensearch_index::VectorIndex::open`] is
//!   strictly v1 and can never see a v2 identity), joins every bound
//!   embedding against it, and republishes a complete FSVI v2 replacement via
//!   [`frankensearch_index::VectorIndex::create_v2`] into a non-canonical staging directory.
//!
//! # Canonical publication is gated (composite generation authority)
//!
//! Installing a staged fast/quality pair over the canonical filenames is a
//! SPLIT publication — two renames with no atomic pair authority. Until the
//! composite generation-authority primitive lands (bd-xomn.1/.3), canonical
//! publication of a fully admitted v2 replacement is refused with the typed
//! `composite-generation-authority-unavailable` reason, *before any queue
//! drain*, so the permanent condition can never consume retry budget or
//! drop queued documents. [`RefreshWorker::publish_staged_canonical`] pins
//! the same refusal at the staged seam.
//!
//! # Pre-drain classification is read-only and performs real admission (r2)
//!
//! The r2 successor of the NO-GO'd C4-write slice (868c0801) repairs the
//! pre-drain seam in both directions:
//! - what it CLAIMS it now PERFORMS: attested v2 tiers are fully admitted at
//!   classification ([`frankensearch_index::VectorIndex::open_admitted_v2`]: exact binding,
//!   recomputed content/docset digests) with the sealed owners retained, so
//!   a header-valid/content-corrupt artifact fails with its own typed
//!   corruption error instead of the composite-authority refusal;
//! - what it MUST NOT do it no longer does: v1 tiers are classified through
//!   the read-only [`frankensearch_index::two_tier::observe_tier`] — the
//!   mutable [`frankensearch_index::VectorIndex::open`] (stale-WAL deletion, corrupt-trailer
//!   truncation) never runs during classification, so a mixed v2+v1
//!   generation can never mutate while being classified for refusal.
//!
//! # Lifecycle
//!
//! Cancellation during inference restores unacknowledged jobs without spending
//! retry budget. Once synchronous publication starts, it completes before jobs
//! are acknowledged. Dropping a suspended cycle provides the same recovery.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asupersync::Cx;
use tracing::{debug, error, info, warn};

use frankensearch_core::TieredQueryEmbeddings;
use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::error::SearchResult;
use frankensearch_core::traits::Embedder;
use frankensearch_index::two_tier::ActivatedTierSearch;
use frankensearch_index::{FsviV2IdentityBinding, TwoTierIndex, ValidatedFsviBytes};

use crate::cache::IndexCache;
use crate::queue::EmbeddingQueue;

mod admission;
mod pipeline;

use admission::composite_authority_refusal;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the index refresh worker.
#[derive(Debug, Clone)]
pub struct RefreshWorkerConfig {
    /// How often to poll the queue for new jobs. Default: 1000ms.
    pub poll_interval: Duration,
    /// Maximum documents to embed per refresh cycle. Default: 1000.
    pub max_docs_per_cycle: usize,
    /// Directory where vector indices are written. Required.
    pub index_dir: PathBuf,
    /// `TwoTierConfig` for newly built indices.
    pub index_config: TwoTierConfig,
}

impl RefreshWorkerConfig {
    /// Create a config with the given index directory and defaults.
    #[must_use]
    pub fn new(index_dir: impl Into<PathBuf>) -> Self {
        Self {
            poll_interval: Duration::from_secs(1),
            max_docs_per_cycle: 1000,
            index_dir: index_dir.into(),
            index_config: TwoTierConfig::default(),
        }
    }

    /// Override the poll interval.
    #[must_use]
    pub const fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Override the max docs per cycle.
    #[must_use]
    pub const fn with_max_docs_per_cycle(mut self, max: usize) -> Self {
        self.max_docs_per_cycle = max;
        self
    }

    /// Override the index config.
    #[must_use]
    pub fn with_index_config(mut self, config: TwoTierConfig) -> Self {
        self.index_config = config;
        self
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Lock-free counters for refresh worker telemetry.
#[derive(Debug, Default)]
pub struct RefreshMetrics {
    /// Total refresh cycles executed.
    pub cycles: AtomicU64,
    /// Total documents embedded.
    pub docs_embedded: AtomicU64,
    /// Total documents that failed embedding.
    pub docs_failed: AtomicU64,
    /// Total index rebuilds (successful).
    pub index_rebuilds: AtomicU64,
    /// Total index rebuild failures.
    pub rebuild_failures: AtomicU64,
    /// Total embedding time in microseconds.
    pub embed_time_us: AtomicU64,
    /// Total rebuild time in microseconds.
    pub rebuild_time_us: AtomicU64,
}

impl RefreshMetrics {
    /// Snapshot of the current metrics.
    #[must_use]
    pub fn snapshot(&self) -> RefreshMetricsSnapshot {
        RefreshMetricsSnapshot {
            cycles: self.cycles.load(Ordering::Relaxed),
            docs_embedded: self.docs_embedded.load(Ordering::Relaxed),
            docs_failed: self.docs_failed.load(Ordering::Relaxed),
            index_rebuilds: self.index_rebuilds.load(Ordering::Relaxed),
            rebuild_failures: self.rebuild_failures.load(Ordering::Relaxed),
            embed_time_us: self.embed_time_us.load(Ordering::Relaxed),
            rebuild_time_us: self.rebuild_time_us.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of refresh metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefreshMetricsSnapshot {
    /// Total refresh cycles executed.
    pub cycles: u64,
    /// Total documents embedded.
    pub docs_embedded: u64,
    /// Total documents that failed embedding.
    pub docs_failed: u64,
    /// Total index rebuilds (successful).
    pub index_rebuilds: u64,
    /// Total index rebuild failures.
    pub rebuild_failures: u64,
    /// Total embedding time in microseconds.
    pub embed_time_us: u64,
    /// Total rebuild time in microseconds.
    pub rebuild_time_us: u64,
}

// ---------------------------------------------------------------------------
// Staged identity-bound generation
// ---------------------------------------------------------------------------

/// A proven, non-canonical identity-bound v2 replacement generation
/// (bd-9xuj T2 C4-write, owner retention per r2).
///
/// Produced by [`RefreshWorker::stage_identity_bound_generation`]: the tiers
/// live under the `v2-staged/` subdirectory of the index dir — never the
/// canonical filenames — and `index` is the staged pair re-admitted through
/// [`TwoTierIndex::open_admitted_v2_with_paths`], so its identity is
/// header-ATTESTED (`index.fast_identity_is_attested()`) and the sealed
/// [`ValidatedFsviBytes`] admission owners are RETAINED inside it
/// ([`Self::fast_admitted_owner`] / [`Self::quality_admitted_owner`]): the
/// `Arc`'d bytes, complete witness, and publication state stay the authority
/// for every read — replacing or renaming the staged files afterwards cannot
/// alter what this generation serves. Canonical installation is refused by
/// [`RefreshWorker::publish_staged_canonical`] until the composite
/// generation-authority primitive lands.
pub struct StagedIdentityBoundGeneration {
    /// The staged generation, opened through exact v2 admission with its
    /// sealed owners retained.
    pub index: TwoTierIndex,
    /// Staged fast-tier artifact path.
    pub fast_path: PathBuf,
    /// Staged quality-tier artifact path, when a quality tier was staged.
    pub quality_path: Option<PathBuf>,
    /// Exact binding the staged fast tier was written and admitted under.
    pub fast_binding: FsviV2IdentityBinding,
    /// Exact binding the staged quality tier was written and admitted under.
    pub quality_binding: Option<FsviV2IdentityBinding>,
}

impl StagedIdentityBoundGeneration {
    /// The retained sealed admission owner of the staged fast tier.
    ///
    /// Always `Some` for a value produced by
    /// [`RefreshWorker::stage_identity_bound_generation`] (the staged pair
    /// is opened exclusively through exact v2 admission); typed as `Option`
    /// only because `index` is a public field.
    #[must_use]
    pub fn fast_admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        self.index.fast_admitted_owner()
    }

    /// The retained sealed admission owner of the staged quality tier, when
    /// a quality tier was staged.
    #[must_use]
    pub fn quality_admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        self.index.quality_admitted_owner()
    }

    /// Activate typed owner-backed search over this staged generation
    /// (bd-core-vector-space-search-guard-ctzo).
    ///
    /// This is the production consumption of
    /// [`TwoTierIndex::activate_owner_backed_search`] on a staged generation.
    /// The bindings it joins against are the ones `index` retained when each
    /// staged tier was admitted — the same values this struct exposes as
    /// [`Self::fast_binding`] / [`Self::quality_binding`] — so no caller can
    /// substitute an identity of its own choosing.
    ///
    /// Every identity join runs before any vector byte is reachable, and the
    /// returned capability is the only route to a read.
    ///
    /// # Errors
    ///
    /// Propagates the typed activation refusals, which name the exact tier
    /// and contract field.
    pub fn activate_search<'staged, 'query>(
        &'staged self,
        embeddings: &'query TieredQueryEmbeddings,
    ) -> SearchResult<ActivatedTierSearch<'staged, 'query>> {
        self.index.activate_owner_backed_search(embeddings)
    }
}

impl std::fmt::Debug for StagedIdentityBoundGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedIdentityBoundGeneration")
            .field("fast_path", &self.fast_path)
            .field("quality_path", &self.quality_path)
            .field("fast_binding", &self.fast_binding)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Refresh worker
// ---------------------------------------------------------------------------

/// Background worker that drains the embedding queue and rebuilds the index.
///
/// # Architecture
///
/// ```text
/// EmbeddingQueue ──drain──> RefreshWorker ──embed_batch_bound──> RefreshRecord
///                                                         │
///                                                    ┌────┘
///                                                    ▼
///                                              IndexCache.replace()
/// ```
///
/// The worker is the single writer for vector indices. It:
/// 1. Drains pending jobs from the [`EmbeddingQueue`]
/// 2. Batch-embeds via the fast-tier [`Embedder`] (and optionally
///    quality-tier), binding every output to its producing identity
/// 3. Rebuilds the `TwoTierIndex` (bootstrap lane) or refuses typed
/// 4. Atomically replaces the cached index via [`IndexCache::replace`]
///
/// # Identity admission boundary (r2)
///
/// A canonical generation that retains content is admissible for replacement
/// only when its identity is ATTESTED (FSVI v2 header), joins the executing
/// embedders' identity as the same space and the same producer, AND fully
/// admits through [`frankensearch_index::VectorIndex::open_admitted_v2`] — content and docset
/// digests recomputed, sealed owners retained. Pre-drain classification is
/// strictly READ-ONLY on the canonical artifacts: v1 tiers are observed
/// header-only (never the mutable [`frankensearch_index::VectorIndex::open`], which deletes stale
/// WAL sidecars and truncates corrupt trailers), and content-retaining v1
/// tiers keep the landed containment refusal (`identityless-fsvi-v1`),
/// applied conservatively (an all-tombstoned v1 slab fails closed). A
/// refused non-empty cycle aborts before draining, leaving queued jobs and
/// retry counts untouched. Fully admitted v2 replacements can be staged and
/// proven ([`Self::stage_identity_bound_generation`]) but canonical
/// installation is refused pre-drain until composite generation authority
/// lands (see the module docs).
///
/// # Cancellation
///
/// The worker checks cancellation before leasing, after embedding awaits and
/// before publication. Provider cancellation is not a retryable embedding
/// failure or permission to publish a fast-only replacement. Dropping a
/// suspended cycle restores its unacknowledged leased jobs. The synchronous
/// rebuild/cache publication section finishes before jobs are acknowledged.
pub struct RefreshWorker {
    config: RefreshWorkerConfig,
    queue: Arc<EmbeddingQueue>,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    cache: Arc<IndexCache>,
    metrics: Arc<RefreshMetrics>,
}

impl std::fmt::Debug for RefreshWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshWorker")
            .field("config", &self.config)
            .field("fast_embedder", &self.fast_embedder.id())
            .finish_non_exhaustive()
    }
}

impl RefreshWorker {
    /// Create a new refresh worker.
    #[must_use]
    pub fn new(
        config: RefreshWorkerConfig,
        queue: Arc<EmbeddingQueue>,
        fast_embedder: Arc<dyn Embedder>,
        cache: Arc<IndexCache>,
    ) -> Self {
        Self {
            config,
            queue,
            fast_embedder,
            quality_embedder: None,
            cache,
            metrics: Arc::new(RefreshMetrics::default()),
        }
    }

    /// Set the quality-tier embedder for two-tier index building.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.quality_embedder = Some(embedder);
        self
    }

    /// Shared reference to the metrics counters.
    #[must_use]
    pub const fn metrics(&self) -> &Arc<RefreshMetrics> {
        &self.metrics
    }

    /// Run the refresh loop.
    ///
    /// Polls the queue at `poll_interval`, embeds batches, and rebuilds the
    /// index. Returns `Ok(())` when the `Cx` is cancelled.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` on unrecoverable failures (e.g., index directory
    /// inaccessible). Transient embedding failures are logged and retried
    /// via the queue's retry mechanism.
    pub async fn run(&self, cx: &Cx) -> SearchResult<()> {
        info!(
            target: "frankensearch.refresh",
            poll_interval_ms = u64::try_from(self.config.poll_interval.as_millis()).unwrap_or(u64::MAX),
            max_docs = self.config.max_docs_per_cycle,
            index_dir = %self.config.index_dir.display(),
            "refresh worker started"
        );

        loop {
            if cx.checkpoint().is_err() || cx.is_cancel_requested() {
                info!(
                    target: "frankensearch.refresh",
                    "refresh worker shutting down (cancel requested)"
                );
                return Ok(());
            }

            // Cancel-aware sleep: Instant must come from this Cx, not wall_now().
            asupersync::time::sleep(cx.now(), self.config.poll_interval).await;

            if cx.checkpoint().is_err() || cx.is_cancel_requested() {
                info!(
                    target: "frankensearch.refresh",
                    "refresh worker shutting down (cancel requested)"
                );
                return Ok(());
            }

            // Run one refresh cycle. Transient errors are logged, not propagated.
            match self.run_cycle(cx).await {
                Ok(0) => {
                    // No work to do — continue polling.
                }
                Ok(n) => {
                    debug!(
                        target: "frankensearch.refresh",
                        docs = n,
                        "refresh cycle complete"
                    );
                }
                Err(e) => {
                    error!(
                        target: "frankensearch.refresh",
                        error = %e,
                        "refresh cycle failed"
                    );
                    // Continue polling — next cycle may succeed.
                }
            }

            self.metrics.cycles.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Run a single refresh cycle.
    ///
    /// Returns the number of documents successfully embedded and installed in
    /// the cache, or an error if admission, rebuilding or installation failed.
    /// Unacknowledged jobs remain reserved until installation succeeds; dropping
    /// this future restores them without consuming retry budget.
    ///
    /// # Errors
    ///
    /// Returns identity-admission, index creation/writing and cache publication
    /// errors, or cancellation without consuming retry budget. Individual
    /// embedding failures use the queue's normal retry policy.
    pub async fn run_cycle(&self, cx: &Cx) -> SearchResult<usize> {
        Self::check_refresh_cancellation(cx, "refresh.run_cycle")?;
        // Preserve the pre-drain refusal: a permanent admission/publication
        // barrier must not consume work or retry budget, or start inference.
        if self.queue.is_empty() {
            return Ok(0);
        }
        // Reject an invalid inference budget before reserving jobs or doing I/O.
        self.inference_batch_size()?;
        self.ensure_canonical_cycle_admissible()?;

        // One reservation covers the complete cycle, including both embedding
        // awaits and publication. No queue mutex is held across any of them.
        let batch = self.queue.lease_batch_up_to(self.config.max_docs_per_cycle);
        if batch.is_empty() {
            return Ok(0);
        }

        // A competing cycle may have finished between the first admission check
        // and our lease acquisition. Never use that earlier classification to
        // overwrite a generation that has since become non-bootstrap.
        self.ensure_canonical_cycle_admissible()?;
        let expected = self.cache.current();
        let total_jobs = batch.len();
        debug!(
            target: "frankensearch.refresh",
            jobs = total_jobs,
            "starting refresh cycle"
        );

        // Embed all documents, identity-bound at the embedder boundary.
        let embedded = self.embed_batch(cx, batch.jobs()).await?;
        Self::check_refresh_cancellation(cx, "refresh.before_publication")?;

        if embedded.is_empty() {
            // All embeddings failed — nothing to index. Their explicit retry
            // decisions already released their reservations.
            warn!(
                target: "frankensearch.refresh",
                jobs = total_jobs,
                "all embeddings failed in cycle"
            );
            return Ok(0);
        }

        let embedded_count = embedded.len();

        // Refuse an already-obsolete cycle before touching canonical files.
        // A second, atomic check at installation handles a race during rebuild.
        if !Arc::ptr_eq(&expected, &self.cache.current()) {
            return Err(Self::superseded_cache_publication());
        }

        // Rebuilding and cache installation are synchronous. Once this section
        // starts, finish it before acknowledging jobs; never suspend between
        // installing the accepted snapshot and recording its content hashes.
        let rebuild_start = Instant::now();
        match self.rebuild_index(&embedded) {
            Ok(new_index) => {
                let rebuild_us =
                    u64::try_from(rebuild_start.elapsed().as_micros()).unwrap_or(u64::MAX);
                self.metrics
                    .rebuild_time_us
                    .fetch_add(rebuild_us, Ordering::Relaxed);

                // Another cache reload/replacement can run during inference.
                // Preserve its snapshot rather than unconditionally replacing
                // it with this older cycle's candidate. This is an in-process
                // cache fence, not durable composite-generation authority.
                let installed = self
                    .cache
                    .replace_if_current(&expected, new_index)
                    .and_then(|replaced| {
                        replaced
                            .then_some(())
                            .ok_or_else(Self::superseded_cache_publication)
                    });
                if let Err(error) = installed {
                    self.metrics
                        .rebuild_failures
                        .fetch_add(1, Ordering::Relaxed);
                    error!(
                        target: "frankensearch.refresh",
                        error = %error,
                        "cache publication refused; retaining unacknowledged jobs"
                    );
                    // Contract refusal and a lost publication race are not
                    // embedding failures. Lease drop restores these jobs with
                    // their original retry counts and no successful hash entry.
                    return Err(error);
                }

                self.metrics.index_rebuilds.fetch_add(1, Ordering::Relaxed);
                for record in &embedded {
                    self.queue
                        .record_embedded(&record.doc_id, &record.content_hash);
                }

                info!(
                    target: "frankensearch.refresh",
                    docs = embedded_count,
                    rebuild_ms = rebuild_us / 1000,
                    "index rebuilt and swapped"
                );

                Ok(embedded_count)
            }
            Err(e) => {
                self.metrics
                    .rebuild_failures
                    .fetch_add(1, Ordering::Relaxed);

                // Missing/invalid embeddings may already have received a retry
                // outcome. Retry only still-owned jobs, exactly once, using the
                // reserved capacity even when new producers filled the queue.
                let dropped_requeues = batch.retry_unacknowledged();
                if dropped_requeues > 0 {
                    self.metrics.docs_failed.fetch_add(
                        u64::try_from(dropped_requeues).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                }

                error!(
                    target: "frankensearch.refresh",
                    error = %e,
                    dropped_requeues,
                    "index rebuild failed, attempted to requeue jobs"
                );

                Err(e)
            }
        }
    }

    /// Canonical installation of a staged generation — REFUSED in this slice.
    ///
    /// Installing the staged fast/quality pair over the canonical filenames
    /// would be a split two-rename publication with no atomic pair
    /// authority; a crash between the renames leaves mixed generations
    /// serving. This slice deliberately does not invent a pair-atomicity
    /// primitive. The refusal is typed (`refresh.canonical_publication` /
    /// `composite-generation-authority-unavailable`) and reopens when the
    /// composite generation-authority primitive lands (bd-xomn.1/.3).
    ///
    /// # Errors
    ///
    /// Always returns the typed composite-authority refusal in this slice.
    pub fn publish_staged_canonical(
        &self,
        staged: &StagedIdentityBoundGeneration,
    ) -> SearchResult<()> {
        warn!(
            target: "frankensearch.refresh",
            staged_fast = %staged.fast_path.display(),
            index_dir = %self.config.index_dir.display(),
            "canonical publication of a staged split generation refused: composite \
             generation authority unavailable (bd-xomn.1/.3)"
        );
        Err(composite_authority_refusal(&self.config.index_dir))
    }

    /// Reference to the index directory.
    #[must_use]
    pub fn index_dir(&self) -> &Path {
        &self.config.index_dir
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests;
