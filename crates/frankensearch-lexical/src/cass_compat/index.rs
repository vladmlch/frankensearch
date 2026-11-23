use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::error::{SearchError, SearchResult};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use tantivy::SegmentMeta;
use tantivy::indexer::{LogMergePolicy, MergeCandidate, MergePolicy, NoMergePolicy, UserOperation};
#[cfg(feature = "tantivy-oracle")]
use tantivy::query::Query;
use tantivy::{Index, IndexReader, IndexWriter, doc};
use tracing::{debug, info, warn};

use super::preview::{cass_build_content_prefix_and_preview, cass_generate_edge_ngrams};
#[cfg(feature = "tantivy-oracle")]
use super::query::{
    CassOracleProfileObservation, CassOracleProfileOutcome, CassQueryFilters,
    MAX_CASS_ORACLE_DOCUMENT_ID_AGGREGATE_BYTES, MAX_CASS_ORACLE_DOCUMENT_ID_BYTES,
    MAX_CASS_ORACLE_FETCH_HITS, MAX_CASS_ORACLE_TOKENS, cass_oracle_bound,
    cass_oracle_invalid_bound, cass_oracle_preflight, cass_parse_boolean_query_bounded,
    cass_regex_query_cached, cass_sanitize_query, cass_tokens_have_boolean_operators,
};
#[cfg(feature = "tantivy-oracle")]
use super::query_builder::{
    CassRegexQueryFactory, cass_fail_closed_tantivy_query, cass_try_build_tantivy_query_from_tokens,
};
#[cfg(feature = "tantivy-oracle")]
use super::schema::cass_document_identity;
use super::schema::{
    CASS_SCHEMA_HASH, CassDocument, CassDocumentRef, CassFields, cass_build_schema,
    cass_fields_from_schema,
};
use super::tantivy_err;
use super::tokenizer::cass_ensure_tokenizer;

/// Adapter that re-wraps an `Arc<dyn MergePolicy>` as a `Box<dyn MergePolicy>`,
/// so a policy snapshot taken via [`IndexWriter::get_merge_policy`] (which
/// returns an `Arc`) can be reinstalled via
/// [`IndexWriter::set_merge_policy`] (which takes a `Box`).
///
/// This is the missing glue that lets [`CassTantivyIndex::force_merge_bounded`]
/// preserve and restore the *exact* previously-configured merge policy
/// (including any caller customization) instead of resetting to
/// `LogMergePolicy::default()`.
#[derive(Debug)]
struct ArcMergePolicy(Arc<dyn MergePolicy>);

impl MergePolicy for ArcMergePolicy {
    fn compute_merge_candidates(&self, segments: &[SegmentMeta]) -> Vec<MergeCandidate> {
        self.0.compute_merge_candidates(segments)
    }
}

/// Minimum time (ms) between merge operations.
const MERGE_COOLDOWN_MS: i64 = 300_000;
/// Segment count threshold above which merge is triggered.
const MERGE_SEGMENT_THRESHOLD: usize = 4;
/// Cap cass rebuilds to a more aggressive Tantivy worker pool size. The
/// rebuild path is bulk-load oriented and can profit from many more indexing
/// workers on 32+ core servers than the original workstation-oriented cap.
///
/// Operators can still lower this via `CASS_TANTIVY_MAX_WRITER_THREADS`.
const CASS_MAX_WRITER_THREADS: usize = 32;
/// Reserve a healthy minimum heap budget so large repairs do not churn tiny
/// segments and constant flush/merge cycles on multi-million-message corpora.
const CASS_MIN_WRITER_HEAP_BYTES: usize = 256 * 1024 * 1024;
/// Scale the heap budget with the worker count so each indexing thread has
/// enough in-memory runway before Tantivy has to flush segments.
const CASS_WRITER_HEAP_PER_THREAD_BYTES: usize = 128 * 1024 * 1024;
/// During cass bulk rebuilds, delay background merges until there is a
/// meaningful backlog of segments instead of burning CPU on near-continuous
/// compaction while the canonical database is still streaming documents in.
const CASS_BULK_LOAD_MIN_SEGMENTS_PER_MERGE: usize = 256;
/// Large cass rebuild batches should be split into multiple independent writer
/// submissions so Tantivy's worker pool can consume them concurrently instead
/// of idling behind a single giant `IndexWriter::run(...)` call.
const CASS_PARALLEL_ADD_MIN_DOCS: usize = 2_048;
const CASS_PARALLEL_ADD_TARGET_BATCH_DOCS: usize = 512;
const CASS_PARALLEL_ADD_MAX_BATCHES: usize = 64;

/// Global last merge timestamp (ms since epoch).
static LAST_MERGE_TS: AtomicI64 = AtomicI64::new(0);

/// Merge status for cass-compatible Tantivy segment optimization.
#[derive(Debug, Clone)]
pub struct CassMergeStatus {
    pub segment_count: usize,
    pub last_merge_ts: i64,
    pub ms_since_last_merge: i64,
    pub merge_threshold: usize,
    pub cooldown_ms: i64,
}

impl CassMergeStatus {
    #[must_use]
    pub const fn should_merge(&self) -> bool {
        self.segment_count >= self.merge_threshold
            && (self.ms_since_last_merge < 0 || self.ms_since_last_merge >= self.cooldown_ms)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CassWriterConfig {
    pub(super) num_threads: usize,
    pub(super) heap_size_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CassAddPlan {
    pub(super) parallel_batches: bool,
    pub(super) batch_docs: usize,
    pub(super) batch_count: usize,
}

fn cass_writer_config() -> CassWriterConfig {
    let available_parallelism = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    cass_writer_config_for_parallelism(available_parallelism)
}

pub(super) fn cass_writer_config_for_parallelism(available_parallelism: usize) -> CassWriterConfig {
    let max_threads = std::env::var("CASS_TANTIVY_MAX_WRITER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(CASS_MAX_WRITER_THREADS);
    let num_threads = available_parallelism.clamp(1, max_threads);
    let heap_size_bytes = num_threads
        .saturating_mul(CASS_WRITER_HEAP_PER_THREAD_BYTES)
        .max(CASS_MIN_WRITER_HEAP_BYTES);
    CassWriterConfig {
        num_threads,
        heap_size_bytes,
    }
}

fn cass_parallel_add_min_docs() -> usize {
    std::env::var("CASS_TANTIVY_PARALLEL_ADD_MIN_DOCS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(CASS_PARALLEL_ADD_MIN_DOCS)
}

fn cass_parallel_add_target_batch_docs() -> usize {
    std::env::var("CASS_TANTIVY_PARALLEL_ADD_BATCH_DOCS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(CASS_PARALLEL_ADD_TARGET_BATCH_DOCS)
}

fn cass_parallel_add_max_batches() -> usize {
    std::env::var("CASS_TANTIVY_PARALLEL_ADD_MAX_BATCHES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(CASS_PARALLEL_ADD_MAX_BATCHES)
}

pub(super) fn cass_parallel_add_plan(
    doc_count: usize,
    available_parallelism: usize,
    target_batch_docs: usize,
    min_parallel_docs: usize,
    max_batches: usize,
) -> CassAddPlan {
    if doc_count == 0 {
        return CassAddPlan {
            parallel_batches: false,
            batch_docs: 1,
            batch_count: 0,
        };
    }

    let target_batch_docs = target_batch_docs.max(1);
    let max_batches = max_batches.max(1);
    let parallelism_batches = available_parallelism.max(1).saturating_mul(2).max(1);
    let batch_count = doc_count
        .div_ceil(target_batch_docs)
        .min(max_batches)
        .min(parallelism_batches)
        .max(1);
    let batch_docs = doc_count.div_ceil(batch_count).max(1);

    CassAddPlan {
        parallel_batches: doc_count >= min_parallel_docs && batch_count > 1,
        batch_docs,
        batch_count,
    }
}

/// Tantivy index compatible with cass lexical schema and lifecycle.
pub struct CassTantivyIndex {
    pub(super) index: Index,
    pub(super) writer: IndexWriter,
    fields: CassFields,
}

impl CassTantivyIndex {
    /// Create a fresh in-memory CASS oracle with one deterministic writer.
    ///
    /// This constructor is restricted to differential campaigns whose native
    /// `DocAddress` evidence must be reproducible across same-seed replays.
    /// Shipping callers retain the host-scaled writer configuration exposed by
    /// [`Self::open_or_create`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if schema extraction or writer construction
    /// fails.
    #[cfg(feature = "tantivy-oracle")]
    #[doc(hidden)]
    pub fn in_memory_single_threaded_oracle() -> SearchResult<Self> {
        let mut index = Index::create_in_ram(cass_build_schema());
        cass_ensure_tokenizer(&mut index);
        let fields = cass_fields_from_schema(&index.schema())?;
        let writer_config = cass_writer_config_for_parallelism(1);
        let writer = index
            .writer_with_num_threads(writer_config.num_threads, writer_config.heap_size_bytes)
            .map_err(tantivy_err)?;
        Ok(Self {
            index,
            writer,
            fields,
        })
    }

    /// Open existing index or create/rebuild as needed.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if filesystem I/O, schema extraction, or Tantivy
    /// index creation/opening fails.
    pub fn open_or_create(path: &Path) -> SearchResult<Self> {
        let available_parallelism = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        Self::open_or_create_with_writer_parallelism(path, available_parallelism)
    }

    /// Open existing index or create/rebuild as needed, using an explicit
    /// Tantivy writer parallelism budget instead of the host-wide default.
    ///
    /// This is used by cass staged shard builds so multiple isolated shard
    /// writers can run concurrently without every shard trying to claim the
    /// machine-wide maximum writer thread count.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if filesystem I/O, schema extraction, or Tantivy
    /// index creation/opening fails.
    pub fn open_or_create_with_writer_parallelism(
        path: &Path,
        available_parallelism: usize,
    ) -> SearchResult<Self> {
        std::fs::create_dir_all(path).map_err(tantivy_err)?;

        let meta_path = path.join("schema_hash.json");
        let needs_rebuild = if meta_path.exists()
            && let Ok(meta) = std::fs::read_to_string(&meta_path)
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&meta)
            && json.get("schema_hash").and_then(|v| v.as_str()) == Some(CASS_SCHEMA_HASH)
        {
            false
        } else {
            true
        };

        if needs_rebuild {
            if let Err(e) = std::fs::remove_dir_all(path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    return Err(SearchError::from(e));
                }
            }
            std::fs::create_dir_all(path).map_err(tantivy_err)?;
        }

        let mut index = if path.join("meta.json").exists() && !needs_rebuild {
            match Index::open_in_dir(path) {
                Ok(idx) => idx,
                Err(e) => {
                    warn!(
                        error = %e,
                        "failed to open existing cass-compatible index; rebuilding"
                    );
                    if let Err(e) = std::fs::remove_dir_all(path) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            return Err(SearchError::from(e));
                        }
                    }
                    std::fs::create_dir_all(path).map_err(tantivy_err)?;
                    Index::create_in_dir(path, cass_build_schema()).map_err(tantivy_err)?
                }
            }
        } else {
            Index::create_in_dir(path, cass_build_schema()).map_err(tantivy_err)?
        };

        cass_ensure_tokenizer(&mut index);
        std::fs::write(
            &meta_path,
            format!("{{\"schema_hash\":\"{CASS_SCHEMA_HASH}\"}}"),
        )
        .map_err(tantivy_err)?;

        let actual_schema = index.schema();
        let writer_config = cass_writer_config_for_parallelism(available_parallelism.max(1));
        debug!(
            tantivy_writer_threads = writer_config.num_threads,
            tantivy_writer_heap_mb = writer_config.heap_size_bytes / (1024 * 1024),
            "opening cass-compatible tantivy writer"
        );
        let writer = index
            .writer_with_num_threads(writer_config.num_threads, writer_config.heap_size_bytes)
            .map_err(tantivy_err)?;
        let fields = cass_fields_from_schema(&actual_schema)?;
        Ok(Self {
            index,
            writer,
            fields,
        })
    }

    #[must_use]
    pub const fn fields(&self) -> CassFields {
        self.fields
    }

    /// Open an [`IndexReader`] for this index.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] when Tantivy reader construction fails.
    pub fn reader(&self) -> SearchResult<IndexReader> {
        self.index.reader().map_err(tantivy_err)
    }

    /// Observe the CASS-compatible parser/search path for Quill conformance.
    ///
    /// The CASS-profile analogue of [`crate::TantivyIndex::oracle_observe_query`].
    /// Like that producer it retains the full Tantivy `DocAddress`, the exact
    /// total count, raw score **bits**, and an expanded cutoff tie group, because
    /// the gauntlet comparator classifies `TieOrder` from native ordering and
    /// cannot do so once that evidence is canonicalized away.
    ///
    /// Two deliberate differences from the default-profile producer, both
    /// required by the CASS activation contract:
    ///
    /// * **No empty-query short circuit.** The default profile returns an empty
    ///   observation for a blank query. Under CASS an empty query carrying
    ///   structured filters is a legitimate filter-only browse that matches
    ///   documents, so short-circuiting would report zero hits where the engine
    ///   returns a full filtered page.
    /// * **Snippets are always `None`.** The activation profile is snippet-free
    ///   (`snippets=disabled` in the schema contract preimage); preview and
    ///   highlight parity belong to a separate lane, and `SnippetWindow` is not
    ///   an active allowance for this profile.
    ///
    /// Document identity is the profile contract's `"{source_id}#{msg_idx}"`
    /// rendering (see [`super::cass_document_identity`]). It is pinned by the contract
    /// preimage rather than chosen here, because changing the rendering silently
    /// changes every mismatch signature already recorded in the divergence
    /// register.
    ///
    /// Cancellation is the caller's responsibility: this is a synchronous
    /// Tantivy path, so the gauntlet adapter checks its `Cx` before calling
    /// rather than threading a context through the oracle.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if reader/searcher construction, query execution,
    /// or stored-document loading fails.
    #[cfg(feature = "tantivy-oracle")]
    pub fn cass_oracle_observe_query(
        &self,
        raw_query: &str,
        filters: &CassQueryFilters,
        limit: usize,
        tie_expansion_limit: usize,
    ) -> SearchResult<crate::OracleQueryObservation> {
        cass_oracle_preflight(raw_query, filters, limit, tie_expansion_limit)?;
        let fields = self.fields();
        let tokens = cass_parse_boolean_query_bounded(raw_query, MAX_CASS_ORACLE_TOKENS).map_err(
            |actual| cass_oracle_invalid_bound("token_count", actual, MAX_CASS_ORACLE_TOKENS),
        )?;
        let has_boolean_operators = cass_tokens_have_boolean_operators(&tokens);
        let query = cass_fail_closed_tantivy_query(
            raw_query,
            cass_try_build_tantivy_query_from_tokens(
                tokens,
                has_boolean_operators,
                filters,
                &fields,
                cass_regex_query_cached,
            ),
        );
        self.cass_oracle_observe_built_query(&*query, limit, tie_expansion_limit)
    }

    /// Observe the real CASS query build and retrieval boundaries without the
    /// compatibility API's silent match-none fallback.
    ///
    /// The returned in-memory witness carries plaintext parser values only so
    /// the gauntlet can immediately redact them. It is not an artifact DTO.
    /// Query construction is total here: a regex, lowering, reader, search, or
    /// document-loading failure is retained as [`super::CassOracleProfileOutcome::Error`]
    /// instead of becoming an indistinguishable successful empty result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] before token retention, filter
    /// cloning, or search when any request bound is exceeded.
    #[cfg(feature = "tantivy-oracle")]
    pub fn cass_oracle_observe_query_profile(
        &self,
        raw_query: &str,
        filters: &CassQueryFilters,
        limit: usize,
        tie_expansion_limit: usize,
    ) -> SearchResult<CassOracleProfileObservation> {
        self.cass_oracle_observe_query_profile_with_regex_factory(
            raw_query,
            filters,
            limit,
            tie_expansion_limit,
            cass_regex_query_cached,
        )
    }

    #[cfg(feature = "tantivy-oracle")]
    pub(super) fn cass_oracle_observe_query_profile_with_regex_factory(
        &self,
        raw_query: &str,
        filters: &CassQueryFilters,
        limit: usize,
        tie_expansion_limit: usize,
        regex_query_factory: CassRegexQueryFactory,
    ) -> SearchResult<CassOracleProfileObservation> {
        cass_oracle_preflight(raw_query, filters, limit, tie_expansion_limit)?;
        let tokens = cass_parse_boolean_query_bounded(raw_query, MAX_CASS_ORACLE_TOKENS).map_err(
            |actual| cass_oracle_invalid_bound("token_count", actual, MAX_CASS_ORACLE_TOKENS),
        )?;
        let has_boolean_operators = cass_tokens_have_boolean_operators(&tokens);
        let fields = self.fields();
        let outcome = cass_try_build_tantivy_query_from_tokens(
            tokens.clone(),
            has_boolean_operators,
            filters,
            &fields,
            regex_query_factory,
        )
        .and_then(|query| self.cass_oracle_observe_built_query(&*query, limit, tie_expansion_limit))
        .map_or_else(
            CassOracleProfileOutcome::Error,
            CassOracleProfileOutcome::Success,
        );
        Ok(CassOracleProfileObservation {
            sanitized_query: cass_sanitize_query(raw_query),
            tokens,
            has_boolean_operators,
            filters: filters.clone(),
            outcome,
        })
    }

    #[cfg(feature = "tantivy-oracle")]
    fn cass_oracle_observe_built_query(
        &self,
        query: &dyn Query,
        limit: usize,
        tie_expansion_limit: usize,
    ) -> SearchResult<crate::OracleQueryObservation> {
        let searcher = self.reader()?.searcher();
        let fields = self.fields();

        // Fetch past `limit` so the cutoff tie group can be observed whole; a
        // truncated group is reported through `cutoff_tie_complete` instead of
        // being silently accepted as complete.
        let fetch_limit = if limit == 0 {
            0
        } else {
            limit.checked_add(tie_expansion_limit).ok_or_else(|| {
                cass_oracle_invalid_bound(
                    "expanded_fetch_hits",
                    usize::MAX,
                    MAX_CASS_ORACLE_FETCH_HITS,
                )
            })?
        };
        cass_oracle_bound(
            "expanded_fetch_hits",
            fetch_limit,
            MAX_CASS_ORACLE_FETCH_HITS,
        )?;
        let search_result = crate::execute_query_with_offset(&searcher, query, fetch_limit, 0)?;

        let mut materialized = Vec::with_capacity(search_result.hits.len());
        let mut retained_identity_bytes = 0_usize;
        for hit in search_result.hits {
            let document = crate::load_doc(&searcher, hit.doc_address)?;
            let doc_id = cass_document_identity(&document, &fields);
            cass_oracle_bound(
                "document_identity_bytes",
                doc_id.len(),
                MAX_CASS_ORACLE_DOCUMENT_ID_BYTES,
            )?;
            retained_identity_bytes = retained_identity_bytes.saturating_add(doc_id.len());
            cass_oracle_bound(
                "aggregate_document_identity_bytes",
                retained_identity_bytes,
                MAX_CASS_ORACLE_DOCUMENT_ID_AGGREGATE_BYTES,
            )?;
            materialized.push(crate::OracleRankedHit {
                doc_id,
                score_bits: hit.bm25_score.to_bits(),
                rank: hit.rank,
                segment_ord: hit.doc_address.segment_ord,
                segment_doc_id: hit.doc_address.doc_id,
                snippet: None,
            });
        }

        let top_len = limit.min(materialized.len());
        let cutoff_bits = top_len
            .checked_sub(1)
            .and_then(|index| materialized.get(index))
            .map(|hit| hit.score_bits);
        let mut cutoff_tie_group = Vec::new();
        if let Some(cutoff) = cutoff_bits {
            for hit in &materialized {
                if f32::from_bits(hit.score_bits)
                    .total_cmp(&f32::from_bits(cutoff))
                    .is_eq()
                {
                    retained_identity_bytes =
                        retained_identity_bytes.saturating_add(hit.doc_id.len());
                    cass_oracle_bound(
                        "aggregate_document_identity_bytes",
                        retained_identity_bytes,
                        MAX_CASS_ORACLE_DOCUMENT_ID_AGGREGATE_BYTES,
                    )?;
                    cutoff_tie_group.push(hit.clone());
                }
            }
        }
        // The group is complete when either every match was fetched or the last
        // fetched hit scored strictly below the cutoff — otherwise the expansion
        // budget ended inside the tie and the comparator must not assume more.
        let cutoff_tie_complete = cutoff_bits.is_none_or(|cutoff| {
            search_result.total_count <= fetch_limit
                || materialized.last().is_none_or(|last| {
                    !f32::from_bits(last.score_bits)
                        .total_cmp(&f32::from_bits(cutoff))
                        .is_eq()
                })
        });
        let expanded_hits = materialized.clone();
        materialized.truncate(top_len);

        let doc_count =
            usize::try_from(searcher.num_docs()).map_err(|_| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: "current Tantivy reader document count does not fit usize".into(),
            })?;

        Ok(crate::OracleQueryObservation {
            hits: materialized,
            cutoff_tie_group,
            cutoff_tie_complete,
            total_count: search_result.total_count,
            doc_count,
            expanded_hits,
            searcher_sha256: crate::oracle_searcher_sha256(&searcher),
        })
    }

    /// Delete all indexed documents.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if Tantivy delete fails.
    pub fn delete_all(&mut self) -> SearchResult<()> {
        self.writer.delete_all_documents().map_err(tantivy_err)?;
        Ok(())
    }

    /// Commit all pending writer operations.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if Tantivy commit fails.
    pub fn commit(&mut self) -> SearchResult<()> {
        self.writer.commit().map_err(tantivy_err)?;
        Ok(())
    }

    pub fn configure_bulk_load_merge_policy(&mut self) {
        let mut merge_policy = LogMergePolicy::default();
        merge_policy.set_min_num_segments(CASS_BULK_LOAD_MIN_SEGMENTS_PER_MERGE);
        self.writer.set_merge_policy(Box::new(merge_policy));
        debug!(
            min_num_segments = CASS_BULK_LOAD_MIN_SEGMENTS_PER_MERGE,
            "configured cass bulk-load merge policy"
        );
    }

    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.index
            .searchable_segment_ids()
            .map_or(0, |ids| ids.len())
    }

    #[must_use]
    pub fn merge_status(&self) -> CassMergeStatus {
        let last_merge_ts = LAST_MERGE_TS.load(Ordering::Relaxed);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
        let ms_since_last = if last_merge_ts > 0 {
            now_ms - last_merge_ts
        } else {
            -1
        };
        CassMergeStatus {
            segment_count: self.segment_count(),
            last_merge_ts,
            ms_since_last_merge: ms_since_last,
            merge_threshold: MERGE_SEGMENT_THRESHOLD,
            cooldown_ms: MERGE_COOLDOWN_MS,
        }
    }

    /// Trigger async merge when threshold/cooldown permit.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if segment enumeration fails.
    pub fn optimize_if_idle(&mut self) -> SearchResult<bool> {
        let segment_ids = self.index.searchable_segment_ids().map_err(tantivy_err)?;
        let segment_count = segment_ids.len();
        if segment_count < MERGE_SEGMENT_THRESHOLD {
            debug!(
                segments = segment_count,
                threshold = MERGE_SEGMENT_THRESHOLD,
                "skipping merge: below threshold"
            );
            return Ok(false);
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
        let last_merge = LAST_MERGE_TS.load(Ordering::Relaxed);
        if last_merge > 0 && (now_ms - last_merge) < MERGE_COOLDOWN_MS {
            debug!(
                ms_since_last = now_ms - last_merge,
                cooldown = MERGE_COOLDOWN_MS,
                "skipping merge: cooldown active"
            );
            return Ok(false);
        }

        info!(
            segments = segment_count,
            "starting cass-compatible segment merge"
        );
        let _merge_future = self.writer.merge(&segment_ids);
        LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
        Ok(true)
    }

    /// Force immediate merge and block until completion.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if segment enumeration or merge fails.
    pub fn force_merge(&mut self) -> SearchResult<()> {
        let segment_ids = self.index.searchable_segment_ids().map_err(tantivy_err)?;
        if segment_ids.is_empty() {
            return Ok(());
        }
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));

        let merge_future = self.writer.merge(&segment_ids);
        match merge_future.wait() {
            Ok(_) => {
                LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
                Ok(())
            }
            Err(err) => Err(tantivy_err(err)),
        }
    }

    /// Force a full merge while never opening more than `batch_size` segments
    /// at once, returning the number of merge operations performed.
    ///
    /// [`force_merge`](Self::force_merge) merges *all* searchable segments in a
    /// single `merge()` call, which memory-maps every input segment
    /// simultaneously. On an index that has accumulated many segments (e.g.
    /// after a long ingest run with transient auto-merge failures), that single
    /// call can map more regions than the OS allows (`vm.max_map_count`,
    /// 65530 by default on Linux) and fail with an out-of-memory / mmap error,
    /// leaving the caller with no in-wrapper recovery path.
    ///
    /// This consolidates the index in bounded passes instead: each iteration
    /// merges at most `batch_size` segments, so peak mmap pressure is capped at
    /// `batch_size * files-per-segment` regardless of how many segments exist.
    /// Repeated passes drive the segment count down to a single top-level
    /// segment.
    ///
    /// For the duration of the operation a [`NoMergePolicy`] is installed: a
    /// completed `merge()` calls `consider_merge_options()` internally, and the
    /// previously installed [`LogMergePolicy`] (or any caller-configured
    /// policy) could otherwise schedule its own unbounded merge of the
    /// remaining same-tier segments — reintroducing exactly the mmap pressure
    /// we are avoiding. The *previously configured* merge policy is preserved
    /// (via [`IndexWriter::get_merge_policy`]) and restored on return — on both
    /// the success and the error paths — so any caller-side customization
    /// (e.g. [`Self::configure_bulk_load_merge_policy`]) is not silently
    /// dropped. Restoration round-trips the `Arc`-shared policy back through
    /// tantivy's `Box`-typed setter via a private `ArcMergePolicy` adapter.
    ///
    /// `batch_size` is clamped to a minimum of 2 (a batch of 1 can never make
    /// progress).
    ///
    /// # Caller contract
    ///
    /// The writer lock is held exclusively for the entire call, which can take
    /// many minutes on a large index. Any other process attempting to acquire
    /// the writer lock on the same directory fails fast with `LockBusy`.
    /// Callers must coordinate externally — e.g. stop the ingester before
    /// invoking this method.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if segment enumeration or any
    /// individual batch merge fails. The previous merge policy is restored
    /// even on the error path.
    pub fn force_merge_bounded(&mut self, batch_size: usize) -> SearchResult<usize> {
        let batch_size = batch_size.max(2);

        // Snapshot the prior merge policy via the cheap Arc clone exposed by
        // tantivy 0.26, then install NoMergePolicy for the duration of the
        // bounded loop. The Arc snapshot is the key bit: restoring
        // `LogMergePolicy::default()` (as a naive implementation would) would
        // silently discard any caller-side customization such as
        // [`Self::configure_bulk_load_merge_policy`]'s tuned
        // `min_num_segments`.
        let prior = self.writer.get_merge_policy();
        self.writer.set_merge_policy(Box::new(NoMergePolicy));

        let mut merges = 0usize;
        let outcome: SearchResult<usize> = (|| {
            loop {
                let segment_ids = self.index.searchable_segment_ids().map_err(tantivy_err)?;
                if segment_ids.len() <= 1 {
                    return Ok(merges);
                }
                let take = segment_ids.len().min(batch_size);
                self.writer
                    .merge(&segment_ids[..take])
                    .wait()
                    .map_err(tantivy_err)?;
                merges += 1;
            }
        })();

        // Restore the *original* merge policy (not a fresh default) on every
        // exit path that is not a panic. Wrapping `prior` in
        // [`ArcMergePolicy`] lets us hand the previously-shared `Arc` back to
        // Tantivy through its `Box<dyn MergePolicy>` setter; the inner Arc is
        // simply re-shared, no rebuilding of policy state.
        self.writer
            .set_merge_policy(Box::new(ArcMergePolicy(prior)));

        if matches!(outcome, Ok(n) if n > 0) {
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
            LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
        }

        outcome
    }

    /// Add a batch of cass-compatible documents.
    ///
    /// The per-doc prep work (edge-ngram generation and preview extraction) is
    /// CPU-heavy; for non-trivial batches we run it in parallel on rayon and
    /// then feed the pre-built `TantivyDocument`s serially into the writer
    /// (which needs `&mut self`). On a 4-core indexing run this keeps the
    /// tantivy writer threads fed instead of starving on a single-threaded
    /// producer.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if adding a document to Tantivy fails.
    pub fn add_cass_documents(&mut self, docs: &[CassDocument]) -> SearchResult<()> {
        // Below this size the rayon split/join overhead isn't worth it.
        const PARALLEL_PREP_THRESHOLD: usize = 8;

        let fields = self.fields;
        let writer_parallelism = cass_writer_config().num_threads.max(1);
        let add_plan = cass_parallel_add_plan(
            docs.len(),
            writer_parallelism,
            cass_parallel_add_target_batch_docs(),
            cass_parallel_add_min_docs(),
            cass_parallel_add_max_batches(),
        );

        debug!(
            docs = docs.len(),
            batch_docs = add_plan.batch_docs,
            batch_count = add_plan.batch_count,
            parallel_batches = add_plan.parallel_batches,
            writer_parallelism,
            "submitting cass-compatible documents to tantivy writer"
        );

        if add_plan.parallel_batches {
            let writer = &self.writer;
            docs.par_chunks(add_plan.batch_docs)
                .try_for_each(|chunk| -> SearchResult<()> {
                    writer
                        .run(
                            chunk
                                .iter()
                                .map(|cass_doc| build_cass_tantivy_document(fields, cass_doc))
                                .map(UserOperation::Add),
                        )
                        .map_err(tantivy_err)?;
                    Ok(())
                })?;
        } else {
            let prepared: Vec<tantivy::TantivyDocument> = if docs.len() < PARALLEL_PREP_THRESHOLD {
                docs.iter()
                    .map(|cass_doc| build_cass_tantivy_document(fields, cass_doc))
                    .collect()
            } else {
                docs.par_iter()
                    .map(|cass_doc| build_cass_tantivy_document(fields, cass_doc))
                    .collect()
            };
            self.writer
                .run(prepared.into_iter().map(UserOperation::Add))
                .map_err(tantivy_err)?;
        }
        Ok(())
    }

    /// Add a batch of borrowed cass-compatible documents.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::SubsystemError`] if adding a document to Tantivy fails.
    pub fn add_cass_document_refs(&mut self, docs: &[CassDocumentRef<'_>]) -> SearchResult<()> {
        const PARALLEL_PREP_THRESHOLD: usize = 8;

        let fields = self.fields;
        let writer_parallelism = cass_writer_config().num_threads.max(1);
        let add_plan = cass_parallel_add_plan(
            docs.len(),
            writer_parallelism,
            cass_parallel_add_target_batch_docs(),
            cass_parallel_add_min_docs(),
            cass_parallel_add_max_batches(),
        );

        debug!(
            docs = docs.len(),
            batch_docs = add_plan.batch_docs,
            batch_count = add_plan.batch_count,
            parallel_batches = add_plan.parallel_batches,
            writer_parallelism,
            "submitting borrowed cass-compatible documents to tantivy writer"
        );

        if add_plan.parallel_batches {
            let writer = &self.writer;
            docs.par_chunks(add_plan.batch_docs)
                .try_for_each(|chunk| -> SearchResult<()> {
                    writer
                        .run(
                            chunk
                                .iter()
                                .copied()
                                .map(|cass_doc| build_cass_tantivy_document_ref(fields, cass_doc))
                                .map(UserOperation::Add),
                        )
                        .map_err(tantivy_err)?;
                    Ok(())
                })?;
        } else {
            let prepared: Vec<tantivy::TantivyDocument> = if docs.len() < PARALLEL_PREP_THRESHOLD {
                docs.iter()
                    .copied()
                    .map(|cass_doc| build_cass_tantivy_document_ref(fields, cass_doc))
                    .collect()
            } else {
                docs.par_iter()
                    .copied()
                    .map(|cass_doc| build_cass_tantivy_document_ref(fields, cass_doc))
                    .collect()
            };
            self.writer
                .run(prepared.into_iter().map(UserOperation::Add))
                .map_err(tantivy_err)?;
        }
        Ok(())
    }
}

/// Build a single `TantivyDocument` for a cass document. Pure function so it
/// can run on any rayon worker.
fn build_cass_tantivy_document(
    fields: CassFields,
    cass_doc: &CassDocument,
) -> tantivy::TantivyDocument {
    let mut d = doc! {
        fields.agent => cass_doc.agent.clone(),
        fields.source_path => cass_doc.source_path.clone(),
        fields.msg_idx => cass_doc.msg_idx,
        fields.content => cass_doc.content.clone(),
        fields.source_id => cass_doc.source_id.clone(),
        fields.origin_kind => cass_doc.origin_kind.clone(),
    };

    if let Some(host) = &cass_doc.origin_host
        && !host.is_empty()
    {
        d.add_text(fields.origin_host, host);
    }
    if let Some(field) = fields.conversation_id
        && let Some(conversation_id) = cass_doc.conversation_id
    {
        d.add_i64(field, conversation_id);
    }
    if let Some(workspace) = &cass_doc.workspace {
        d.add_text(fields.workspace, workspace);
    }
    if let Some(workspace_original) = &cass_doc.workspace_original {
        d.add_text(fields.workspace_original, workspace_original);
    }
    if let Some(ts) = cass_doc.created_at {
        d.add_i64(fields.created_at, ts);
    }
    if let Some(title) = &cass_doc.title {
        d.add_text(fields.title, title);
        d.add_text(fields.title_prefix, cass_generate_edge_ngrams(title));
    }
    let (content_prefix, preview) = cass_build_content_prefix_and_preview(&cass_doc.content);
    d.add_text(fields.content_prefix, content_prefix);
    d.add_text(fields.preview, preview);
    d
}

fn build_cass_tantivy_document_ref(
    fields: CassFields,
    cass_doc: CassDocumentRef<'_>,
) -> tantivy::TantivyDocument {
    let mut d = doc! {
        fields.agent => cass_doc.agent,
        fields.source_path => cass_doc.source_path,
        fields.msg_idx => cass_doc.msg_idx,
        fields.content => cass_doc.content,
        fields.source_id => cass_doc.source_id,
        fields.origin_kind => cass_doc.origin_kind,
    };

    if let Some(host) = cass_doc.origin_host
        && !host.is_empty()
    {
        d.add_text(fields.origin_host, host);
    }
    if let Some(field) = fields.conversation_id
        && let Some(conversation_id) = cass_doc.conversation_id
    {
        d.add_i64(field, conversation_id);
    }
    if let Some(workspace) = cass_doc.workspace {
        d.add_text(fields.workspace, workspace);
    }
    if let Some(workspace_original) = cass_doc.workspace_original {
        d.add_text(fields.workspace_original, workspace_original);
    }
    if let Some(ts) = cass_doc.created_at {
        d.add_i64(fields.created_at, ts);
    }
    if let Some(title) = cass_doc.title {
        d.add_text(fields.title, title);
        d.add_text(fields.title_prefix, cass_generate_edge_ngrams(title));
    }
    let (content_prefix, preview) = cass_build_content_prefix_and_preview(cass_doc.content);
    d.add_text(fields.content_prefix, content_prefix);
    d.add_text(fields.preview, preview);
    d
}
