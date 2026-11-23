//! Refresh embedding, bootstrap index rebuild, and staged v2 generation pipeline.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

use asupersync::Cx;
use tracing::{info, warn};

use frankensearch_core::BoundQueryEmbedding;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::ArtifactGenerationIdentityV1;
use frankensearch_core::traits::IdentityBoundEmbedding;
use frankensearch_index::{
    FsviV2IdentityBinding, TwoTierIndex, TwoTierIndexPaths, VECTOR_INDEX_FAST_FILENAME,
    VECTOR_INDEX_QUALITY_FILENAME, VectorIndex, next_publication_nonce,
};

use crate::queue::{EmbeddingJob, JobOutcome};

use super::admission::{
    AdmittedCanonicalTier, ExistingGenerationClass, STAGED_V2_DIR_NAME, artifact_identity_for,
    fingerprint_hex, generation_nonce, next_generation_sequence, require_same_producer,
};
use super::{RefreshWorker, StagedIdentityBoundGeneration};

/// The fast and quality vectors joined for one document during a staged merge.
///
/// Either tier may be absent: a row carrying only a fast vector has no quality
/// sibling yet, and a quality-only row belongs to a reachable
/// `AttestedV2 { fast: None, quality: Some(_) }` generation that must land in
/// the staged quality tier without fabricating a fast row.
type MergedTierVectors = (Option<Vec<f32>>, Option<Vec<f32>>);

// ---------------------------------------------------------------------------
// Refresh record (identity-bound carrier)
// ---------------------------------------------------------------------------

/// A document with its identity-bound embeddings, ready for index insertion
/// (bd-9xuj T2 C4-write).
///
/// Both tiers carry [`BoundQueryEmbedding`] — a vector bound at the embedder
/// boundary to the complete identity bundle that produced it, validated at
/// bind time. The embeddings are produced ONLY via
/// [`frankensearch_core::traits::Embedder::embed_batch_bound`]; no raw `Vec<f32>` enters this record, so
/// every downstream seam can verify space and producer identity instead of
/// trusting provenance-free floats.
#[derive(Debug)]
pub(super) struct RefreshRecord {
    pub(super) doc_id: String,
    fast_embedding: BoundQueryEmbedding,
    quality_embedding: Option<BoundQueryEmbedding>,
    pub(super) content_hash: String,
}

/// Convert an embedder-bound output into the C1r2 verifier carrier.
///
/// [`BoundQueryEmbedding::new`] re-validates the bundle and precomputes the
/// identity and space fingerprints, so seam checks are string compares.
fn into_bound_query(bound: IdentityBoundEmbedding) -> SearchResult<BoundQueryEmbedding> {
    BoundQueryEmbedding::new(bound.values, bound.identity)
}

impl RefreshWorker {
    /// Check cancellation only outside the synchronous publication section.
    pub(super) fn check_refresh_cancellation(cx: &Cx, phase: &str) -> SearchResult<()> {
        if cx.checkpoint().is_err() || cx.is_cancel_requested() {
            return Err(SearchError::Cancelled {
                phase: phase.to_owned(),
                reason: cx.cancel_reason().map_or_else(
                    || "refresh checkpoint interrupted".to_owned(),
                    |reason| reason.to_string(),
                ),
            });
        }
        Ok(())
    }

    pub(super) fn superseded_cache_publication() -> SearchError {
        SearchError::InvalidConfig {
            field: "refresh.cache_publication".to_owned(),
            value: "superseded".to_owned(),
            reason: "the cached snapshot changed during this refresh; \
                     pending work was not acknowledged"
                .to_owned(),
        }
    }

    /// Bound each model call independently of the cycle's publication budget.
    pub(super) fn inference_batch_size(&self) -> SearchResult<usize> {
        let size = self.queue.config().batch_size;
        if size == 0 {
            return Err(SearchError::InvalidConfig {
                field: "embedding_queue.batch_size".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }
        Ok(size)
    }

    /// Embed one cycle in bounded model calls, preserving submission order.
    ///
    /// The cycle keeps one lease across all chunks and publishes once. A failed
    /// chunk decides only its own retries; healthy chunks remain unacknowledged
    /// until publication, including when a later chunk is cancelled. The queue's
    /// batch size bounds inference independently of `max_docs_per_cycle`.
    pub(super) async fn embed_batch(
        &self,
        cx: &Cx,
        jobs: &[EmbeddingJob],
    ) -> SearchResult<Vec<RefreshRecord>> {
        let batch_size = self.inference_batch_size()?;
        let mut records = Vec::with_capacity(jobs.len());
        for chunk in jobs.chunks(batch_size) {
            Self::check_refresh_cancellation(cx, "refresh.before_batch")?;
            records.extend(self.embed_inference_batch(cx, chunk).await?);
        }
        Ok(records)
    }

    /// Embed one bounded batch, binding every output to its producing identity.
    ///
    /// Ordinary failures are requeued for retry. Cancellation propagates
    /// without retrying or degrading a cancelled quality call to fast-only;
    /// the owning cycle's lease restores unfinished work.
    async fn embed_inference_batch(
        &self,
        cx: &Cx,
        jobs: &[EmbeddingJob],
    ) -> SearchResult<Vec<RefreshRecord>> {
        let embed_start = Instant::now();

        // Collect texts for batch embedding.
        let texts: Vec<&str> = jobs.iter().map(|j| j.canonical_text.as_str()).collect();

        // Fast-tier identity-bound embedding (required). An identity-less
        // embedder fails here typed (`embedder.identity`), which also means
        // it can no longer publish provenance-free vectors through this
        // worker — that is the C4-write bound-carrier contract.
        let fast_result = self.fast_embedder.embed_batch_bound(cx, &texts).await;
        Self::check_refresh_cancellation(cx, "refresh.fast_embedding")?;
        let fast_bound = match fast_result {
            Ok(embeddings) => embeddings,
            Err(error @ SearchError::Cancelled { .. }) => return Err(error),
            Err(e) => {
                warn!(
                    target: "frankensearch.refresh",
                    error = %e,
                    batch_size = jobs.len(),
                    "fast-tier bound batch embedding failed, requeueing all"
                );
                let mut dropped_requeues = 0usize;
                for job in jobs {
                    if !self.requeue_job(job.clone(), "fast_batch_embedding_failed") {
                        dropped_requeues = dropped_requeues.saturating_add(1);
                    }
                }
                if dropped_requeues > 0 {
                    warn!(
                        target: "frankensearch.refresh",
                        dropped_requeues,
                        "failed to requeue some jobs after fast-tier batch failure"
                    );
                }
                self.metrics.docs_failed.fetch_add(
                    u64::try_from(jobs.len()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                return Ok(Vec::new());
            }
        };
        if self.reject_malformed_batch(jobs, fast_bound.len(), "fast") {
            return Ok(Vec::new());
        }

        // Quality-tier identity-bound embedding (optional).
        let mut quality_bound: Option<Vec<Option<IdentityBoundEmbedding>>> =
            if let Some(ref quality) = self.quality_embedder {
                let result = quality.embed_batch_bound(cx, &texts).await;
                Self::check_refresh_cancellation(cx, "refresh.quality_embedding")?;
                match result {
                    Ok(embeddings) => {
                        if self.reject_malformed_batch(jobs, embeddings.len(), "quality") {
                            return Ok(Vec::new());
                        }
                        Some(embeddings.into_iter().map(Some).collect())
                    }
                    Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                    Err(e) => {
                        warn!(
                            target: "frankensearch.refresh",
                            error = %e,
                            "quality-tier bound batch embedding failed, proceeding with fast only"
                        );
                        None
                    }
                }
            } else {
                None
            };

        let embed_us = u64::try_from(embed_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.metrics
            .embed_time_us
            .fetch_add(embed_us, Ordering::Relaxed);

        // Positional association is admissible only after exact cardinality
        // checks. A short response may have omitted a MIDDLE input, not a suffix.
        let mut records = Vec::with_capacity(jobs.len());
        for (i, (job, fast_ib)) in jobs.iter().zip(fast_bound).enumerate() {
            let fast_embedding = match into_bound_query(fast_ib) {
                Ok(bound) => bound,
                Err(e) => {
                    warn!(
                        target: "frankensearch.refresh",
                        doc_id = %job.doc_id,
                        error = %e,
                        "fast-tier bound embedding failed bind-time validation, requeueing"
                    );
                    if !self.requeue_job(job.clone(), "fast_bound_validation_failed") {
                        warn!(
                            target: "frankensearch.refresh",
                            doc_id = %job.doc_id,
                            "failed to requeue job after bind-time validation failure"
                        );
                    }
                    self.metrics.docs_failed.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };
            let quality_embedding = quality_bound
                .as_mut()
                .and_then(|bound| bound.get_mut(i))
                .and_then(Option::take)
                .and_then(|ib| match into_bound_query(ib) {
                    Ok(bound) => Some(bound),
                    Err(e) => {
                        warn!(
                            target: "frankensearch.refresh",
                            doc_id = %job.doc_id,
                            error = %e,
                            "quality-tier bound embedding failed bind-time validation, \
                             proceeding fast-only for this document"
                        );
                        None
                    }
                });

            records.push(RefreshRecord {
                doc_id: job.doc_id.clone(),
                fast_embedding,
                quality_embedding,
                content_hash: job.content_hash.clone(),
            });
        }

        self.metrics.docs_embedded.fetch_add(
            u64::try_from(records.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.queue
            .metrics()
            .total_embed_time_us
            .fetch_add(embed_us, Ordering::Relaxed);

        Ok(records)
    }

    /// Reject the whole inference chunk if positional correspondence is unknown.
    ///
    /// Missing or extra outputs cannot be assigned to input documents safely.
    /// Retry this chunk only, including malformed quality responses: silently
    /// publishing fast-only here would acknowledge work whose quality output
    /// never satisfied the batch contract. Cancellation is checked before this
    /// method, so it never spends retry budget for a cancelled model call.
    fn reject_malformed_batch(
        &self,
        jobs: &[EmbeddingJob],
        returned: usize,
        tier: &'static str,
    ) -> bool {
        if returned == jobs.len() {
            return false;
        }

        let mut dropped_requeues = 0usize;
        for job in jobs {
            if !self.requeue_job(job.clone(), "batch_cardinality_mismatch") {
                dropped_requeues = dropped_requeues.saturating_add(1);
            }
        }
        self.metrics.docs_failed.fetch_add(
            u64::try_from(jobs.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        warn!(
            target: "frankensearch.refresh",
            tier,
            expected = jobs.len(),
            returned,
            dropped_requeues,
            "discarding malformed embedding batch before associating vectors with documents"
        );
        true
    }

    fn requeue_job(&self, job: EmbeddingJob, reason: &'static str) -> bool {
        let doc_id = job.doc_id.clone();
        match self.queue.requeue(job) {
            JobOutcome::Retryable => true,
            JobOutcome::Failed => {
                warn!(
                    target: "frankensearch.refresh",
                    doc_id = %doc_id,
                    reason,
                    "failed to requeue job"
                );
                false
            }
            outcome => {
                warn!(
                    target: "frankensearch.refresh",
                    doc_id = %doc_id,
                    reason,
                    ?outcome,
                    "unexpected requeue outcome"
                );
                false
            }
        }
    }

    /// Rebuild the `TwoTierIndex` from identity-bound records — the
    /// bootstrap lane (content-free canonical generation only).
    ///
    /// The producing identities are declared on the builder
    /// (`set_*_identity`, process-local, DECLARED — the persisted artifacts
    /// stay v1 and never read as attested), and every bound record is
    /// re-verified against the executing embedder's identity under the
    /// SameProducer-only law before its vector is written.
    pub(super) fn rebuild_index(&self, records: &[RefreshRecord]) -> SearchResult<TwoTierIndex> {
        // Repeat the admission check after embedding to close the race where
        // another publisher installs a generation between the pre-drain check
        // and this rebuild. Only this race fallback consumes one retry.
        self.ensure_canonical_cycle_admissible()?;

        let mut builder =
            TwoTierIndex::create(&self.config.index_dir, self.config.index_config.clone())?;

        builder.set_fast_embedder_id(self.fast_embedder.id());
        if let Some(ref quality) = self.quality_embedder {
            builder.set_quality_embedder_id(quality.id());
        }

        // Declare the producing identities. Records exist only when bound
        // harvesting succeeded, so the fast identity is available; a quality
        // identity is required exactly when quality-bound records exist.
        let fast_expected = self.fast_embedder.identity()?.clone();
        builder.set_fast_identity(&fast_expected)?;
        let need_quality_identity = records
            .iter()
            .any(|record| record.quality_embedding.is_some());
        let quality_expected = if need_quality_identity {
            let Some(quality_embedder) = &self.quality_embedder else {
                return Err(SearchError::InvalidConfig {
                    field: "refresh.quality_identity".to_owned(),
                    value: "missing-quality-embedder".to_owned(),
                    reason: "quality-bound records exist without a quality embedder".to_owned(),
                });
            };
            let expected = quality_embedder.identity()?.clone();
            builder.set_quality_identity(&expected)?;
            Some(expected)
        } else {
            None
        };

        // Keep only the latest update per doc_id from this cycle.
        let mut latest_by_doc_id = HashMap::new();
        let mut consumed = vec![false; records.len()];
        for (idx, record) in records.iter().enumerate() {
            if let Some(previous) = latest_by_doc_id.insert(record.doc_id.as_str(), idx) {
                consumed[previous] = true;
            }
        }

        for (idx, record) in records.iter().enumerate() {
            if consumed[idx] {
                continue;
            }
            // Guard 7 at the write seam: every vector written must be bound
            // to the identity being declared for this generation. This is
            // the defense against an embedder whose bound outputs disagree
            // with its `identity()` (SameProducer only; certified-compatible
            // foreign producers are refused, see [`require_same_producer`]).
            require_same_producer(&record.fast_embedding, &fast_expected, "fast")?;
            if let Some(quality_embedding) = &record.quality_embedding {
                let Some(expected) = &quality_expected else {
                    return Err(SearchError::InvalidConfig {
                        field: "refresh.quality_identity".to_owned(),
                        value: "missing-quality-identity".to_owned(),
                        reason: "quality-bound record exists without a declared quality identity"
                            .to_owned(),
                    });
                };
                require_same_producer(quality_embedding, expected, "quality")?;
            }
            builder.add_record(
                &record.doc_id,
                record.fast_embedding.vector(),
                record
                    .quality_embedding
                    .as_ref()
                    .map(BoundQueryEmbedding::vector),
            )?;
        }

        builder.finish()
    }

    // -----------------------------------------------------------------------
    // Staged identity-bound replacement (the C4-write merge)
    // -----------------------------------------------------------------------

    /// Embed `jobs` identity-bound with NO queue interaction: staging must
    /// not consume retry budget or record content hashes, because the
    /// canonical generation is not being published. Any failure propagates.
    async fn embed_jobs_bound_strict(
        &self,
        cx: &Cx,
        jobs: &[EmbeddingJob],
    ) -> SearchResult<Vec<RefreshRecord>> {
        let batch_size = self.inference_batch_size()?;
        let mut records = Vec::with_capacity(jobs.len());
        for chunk in jobs.chunks(batch_size) {
            Self::check_refresh_cancellation(cx, "refresh.staged_before_batch")?;
            records.extend(self.embed_staged_batch_strict(cx, chunk).await?);
        }
        Ok(records)
    }

    /// Strict inference for one bounded chunk. A failure invalidates the
    /// whole staging attempt; no prefix is written or acknowledged.
    async fn embed_staged_batch_strict(
        &self,
        cx: &Cx,
        jobs: &[EmbeddingJob],
    ) -> SearchResult<Vec<RefreshRecord>> {
        let embed_start = Instant::now();
        let texts: Vec<&str> = jobs.iter().map(|j| j.canonical_text.as_str()).collect();
        let fast_result = self.fast_embedder.embed_batch_bound(cx, &texts).await;
        Self::check_refresh_cancellation(cx, "refresh.staged_fast_embedding")?;
        let fast_bound = fast_result?;
        if fast_bound.len() != jobs.len() {
            return Err(SearchError::InvalidConfig {
                field: "refresh.staged_embedding".to_owned(),
                value: fast_bound.len().to_string(),
                reason: format!(
                    "fast embedder returned {} bound vectors for {} inputs",
                    fast_bound.len(),
                    jobs.len()
                ),
            });
        }
        let quality_bound = match &self.quality_embedder {
            Some(quality) => {
                let quality_result = quality.embed_batch_bound(cx, &texts).await;
                Self::check_refresh_cancellation(cx, "refresh.staged_quality_embedding")?;
                let bound = quality_result?;
                if bound.len() != jobs.len() {
                    return Err(SearchError::InvalidConfig {
                        field: "refresh.staged_embedding".to_owned(),
                        value: bound.len().to_string(),
                        reason: format!(
                            "quality embedder returned {} bound vectors for {} inputs",
                            bound.len(),
                            jobs.len()
                        ),
                    });
                }
                Some(bound)
            }
            None => None,
        };
        let embed_us = u64::try_from(embed_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.metrics
            .embed_time_us
            .fetch_add(embed_us, Ordering::Relaxed);

        let mut quality_iter = quality_bound.map(Vec::into_iter);
        let mut records = Vec::with_capacity(jobs.len());
        for (job, fast_ib) in jobs.iter().zip(fast_bound) {
            let quality_embedding = match quality_iter.as_mut().and_then(Iterator::next) {
                Some(ib) => Some(into_bound_query(ib)?),
                None => None,
            };
            records.push(RefreshRecord {
                doc_id: job.doc_id.clone(),
                fast_embedding: into_bound_query(fast_ib)?,
                quality_embedding,
                content_hash: job.content_hash.clone(),
            });
        }
        self.metrics.docs_embedded.fetch_add(
            u64::try_from(records.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(records)
    }

    /// Allocate a fresh non-canonical attempt directory without reusing or
    /// removing any prior artifact. Exclusive directory creation handles
    /// collisions; the nonce is uniqueness material, not an authority token.
    /// Failed attempts are retained for explicit cleanup, never deleted here.
    fn allocate_staging_directory(&self) -> SearchResult<PathBuf> {
        let root = self.config.index_dir.join(STAGED_V2_DIR_NAME);
        std::fs::create_dir_all(&root).map_err(SearchError::Io)?;
        for attempt in 0_u64..16 {
            let nonce = generation_nonce(&self.config.index_dir, "staged-directory", attempt);
            let path = root.join(format!("attempt-{}", fingerprint_hex(&nonce)));
            let result = {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::DirBuilderExt as _;
                    std::fs::DirBuilder::new().mode(0o700).create(&path)
                }
                #[cfg(not(unix))]
                {
                    std::fs::create_dir(&path)
                }
            };
            match result {
                Ok(()) => return Ok(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(SearchError::Io(error)),
            }
        }
        Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate an unused staging directory after 16 attempts",
        )))
    }

    /// Stage the typed identity-bound replacement generation — the merge
    /// that replaces the former blanket refusal (bd-9xuj T2 C4-write).
    ///
    /// Reads the existing canonical generation through exact FSVI v2
    /// admission (attested identity only; live v1 keeps the typed
    /// `identityless-fsvi-v1` refusal), joins every bound embedding's space
    /// fingerprint against the tier's attested identity
    /// ([`BoundQueryEmbedding::verify_space_identity`]) and applies the
    /// SameProducer-only producer law
    /// ([`BoundQueryEmbedding::verify_producer_conformance`] via
    /// [`require_same_producer`]), merges the carried live rows with the new
    /// records (new wins per `doc_id`; tombstoned rows are never
    /// resurrected), and writes the replacement via
    /// [`VectorIndex::create_v2`] into a fresh attempt directory under
    /// non-canonical `v2-staged/`. Prior attempts and legacy flat staging
    /// files are never overwritten or removed. The staged pair is re-admitted
    /// through
    /// [`TwoTierIndex::open_admitted_v2_with_paths`], so the returned
    /// generation's identity is proven from its own header bytes.
    ///
    /// The queue is untouched: `jobs` are caller-supplied, nothing is
    /// drained, no content hash is recorded, and the canonical generation's
    /// bytes are not modified. Canonical installation is a separate,
    /// currently-refused step ([`Self::publish_staged_canonical`]).
    /// Failed attempts may leave isolated files; cleanup is explicit and is
    /// not part of this operation. Per-attempt isolation is not canonical
    /// publication authority or a garbage-collection policy.
    ///
    /// # Errors
    ///
    /// Typed identity refusals (space, producer, legacy, republication),
    /// admission failures, and I/O errors from staging.
    #[allow(clippy::too_many_lines)]
    pub async fn stage_identity_bound_generation(
        &self,
        cx: &Cx,
        jobs: &[EmbeddingJob],
    ) -> SearchResult<StagedIdentityBoundGeneration> {
        Self::check_refresh_cancellation(cx, "refresh.staged_before_admission")?;
        self.inference_batch_size()?;
        // 1+3 (merged in r2). Gates over the existing canonical generation
        //    (typed refusals, fast-first, identical to the canonical lane)
        //    AND exact admission of the existing attested tiers in the same
        //    step: the reconstructed binding re-proves space/producer/input
        //    equality bit-for-bit against the artifact's own header, the
        //    content/docset digests are recomputed, and the sealed owners
        //    are RETAINED — the same owners the pre-drain check proves, not
        //    a second open.
        let (fast_state, quality_state) = self.canonical_tier_states()?;
        let (fast_admitted, quality_admitted) =
            match self.admit_existing_generation(&fast_state, &quality_state)? {
                ExistingGenerationClass::BootstrapReplaceable => (None, None),
                ExistingGenerationClass::AttestedV2 { fast, quality } => (fast, quality),
            };

        // 2. Harvest identity-bound records (strict; no queue interaction).
        let records = self.embed_jobs_bound_strict(cx, jobs).await?;
        Self::check_refresh_cancellation(cx, "refresh.staged_before_merge")?;

        // 4. Per-embedding seam verification (C1r2 verifiers): each bound
        //    embedding must be the same producer as the identity being
        //    republished, and must join the attested space of any tier it is
        //    merged into. (Producer equality against the ATTESTED producer
        //    follows transitively: record == executing identity ==
        //    reconstructed binding == header, all checked above.)
        let fast_expected = self.fast_embedder.identity()?.clone();
        let quality_expected = match &self.quality_embedder {
            Some(quality)
                if records
                    .iter()
                    .any(|record| record.quality_embedding.is_some()) =>
            {
                Some(quality.identity()?.clone())
            }
            _ => None,
        };
        for record in &records {
            require_same_producer(&record.fast_embedding, &fast_expected, "fast")?;
            if let Some(AdmittedCanonicalTier {
                attested_space_hex, ..
            }) = &fast_admitted
            {
                record
                    .fast_embedding
                    .verify_space_identity(attested_space_hex, "fast")?;
            }
            if let Some(quality_embedding) = &record.quality_embedding {
                let Some(expected) = &quality_expected else {
                    return Err(SearchError::InvalidConfig {
                        field: "refresh.quality_identity".to_owned(),
                        value: "missing-quality-identity".to_owned(),
                        reason: "quality-bound record exists without a quality embedder identity"
                            .to_owned(),
                    });
                };
                require_same_producer(quality_embedding, expected, "quality")?;
                if let Some(AdmittedCanonicalTier {
                    attested_space_hex, ..
                }) = &quality_admitted
                {
                    quality_embedding.verify_space_identity(attested_space_hex, "quality")?;
                }
            }
        }

        // 5. Merge: carried live rows first (tombstones stay dead), then the
        //    new records override per doc_id (last write in the batch wins).
        //    A row's fast slot is `None` only for a carried quality row with
        //    no live fast sibling — including the reachable quality-only
        //    generation (`AttestedV2 { fast: None, quality: Some(_) }`):
        //    such rows land in the staged quality tier and never fabricate a
        //    fast row.
        let mut merged: HashMap<String, MergedTierVectors> = HashMap::new();
        let mut quality_lookup: HashMap<String, Vec<f32>> = HashMap::new();
        if let Some(AdmittedCanonicalTier {
            owner: quality_owner,
            ..
        }) = &quality_admitted
        {
            for i in 0..quality_owner.record_count() {
                let row = quality_owner.row(i)?;
                if !row.flags().is_live() {
                    continue;
                }
                // Last-wins on duplicate doc_ids, matching the fast-tier
                // loop below (the former first-wins `Entry::Vacant` here was
                // an unintended asymmetry between the two carry loops).
                quality_lookup.insert(row.doc_id().to_owned(), quality_owner.vector_at_f32(i)?);
            }
        }
        if let Some(AdmittedCanonicalTier {
            owner: fast_owner, ..
        }) = &fast_admitted
        {
            for i in 0..fast_owner.record_count() {
                let row = fast_owner.row(i)?;
                if !row.flags().is_live() {
                    continue;
                }
                let doc_id = row.doc_id().to_owned();
                let vector = fast_owner.vector_at_f32(i)?;
                let quality = quality_lookup.get(&doc_id).cloned();
                merged.insert(doc_id, (Some(vector), quality));
            }
        }
        for (doc_id, vector) in quality_lookup {
            // Carried quality rows whose doc has no live fast row keep an
            // empty fast slot: they must reach the staged quality tier even
            // when no fast tier was admitted at all.
            merged.entry(doc_id).or_insert((None, Some(vector)));
        }
        for record in &records {
            merged.insert(
                record.doc_id.clone(),
                (
                    Some(record.fast_embedding.vector().to_vec()),
                    record
                        .quality_embedding
                        .as_ref()
                        .map(|bound| bound.vector().to_vec()),
                ),
            );
        }
        if merged.is_empty() {
            return Err(SearchError::InvalidConfig {
                field: "refresh.staged_generation".to_owned(),
                value: "empty".to_owned(),
                reason: "no rows to stage: the job batch produced no records and the canonical \
                         generation carries no live rows"
                    .to_owned(),
            });
        }

        // 6. Write the staged replacement via the production v2 writer.
        Self::check_refresh_cancellation(cx, "refresh.staged_before_write")?;
        let staged_dir = self.allocate_staging_directory()?;
        let staged_fast = staged_dir.join(VECTOR_INDEX_FAST_FILENAME);
        let staged_quality = staged_dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        // Both staged tiers carry ONE publication nonce, the successor of
        // the admitted destination pair's (mirroring the v1 publish_tier
        // route, bd-miio8): the staged pair re-admits as a same-publication
        // pair instead of a legacy 0 == 0 exemption, and a crash between the
        // two eventual per-tier installs stays detectable at open time.
        let publication_nonce = next_publication_nonce(
            fast_admitted
                .as_ref()
                .or(quality_admitted.as_ref())
                .map_or(0, |tier| tier.owner.metadata().publication_nonce),
        );

        let fast_sequence =
            next_generation_sequence(fast_admitted.as_ref().map(|tier| &tier.owner))?;
        let fast_artifact_bundle = artifact_identity_for(self.fast_embedder.as_ref())?;
        let fast_generation = ArtifactGenerationIdentityV1::new(
            fast_sequence,
            generation_nonce(&self.config.index_dir, "fast", fast_sequence),
        )?;
        let fast_binding =
            FsviV2IdentityBinding::new(fast_generation, fast_artifact_bundle.freeze()?)?;
        let mut fast_writer = VectorIndex::create_v2(&staged_fast, fast_binding.clone())?
            .with_publication_nonce(publication_nonce);
        for (doc_id, (fast_vector, _)) in &merged {
            // Quality-only carried rows have no fast slot; never fabricate a
            // fast row for them.
            if let Some(vector) = fast_vector {
                fast_writer.write_record(doc_id, vector)?;
            }
        }
        fast_writer.finish()?;

        let quality_rows: Vec<(&String, &Vec<f32>)> = merged
            .iter()
            .filter_map(|(doc_id, (_, quality))| quality.as_ref().map(|q| (doc_id, q)))
            .collect();
        let quality_binding = if quality_rows.is_empty() {
            None
        } else {
            let Some(quality_embedder) = &self.quality_embedder else {
                return Err(SearchError::InvalidConfig {
                    field: "refresh.quality_identity".to_owned(),
                    value: "missing-quality-embedder".to_owned(),
                    reason: "quality rows were merged without a quality embedder".to_owned(),
                });
            };
            let quality_sequence =
                next_generation_sequence(quality_admitted.as_ref().map(|tier| &tier.owner))?;
            let quality_bundle = artifact_identity_for(quality_embedder.as_ref())?;
            let quality_generation = ArtifactGenerationIdentityV1::new(
                quality_sequence,
                generation_nonce(&self.config.index_dir, "quality", quality_sequence),
            )?;
            let binding = FsviV2IdentityBinding::new(quality_generation, quality_bundle.freeze()?)?;
            let mut quality_writer = VectorIndex::create_v2(&staged_quality, binding.clone())?
                .with_publication_nonce(publication_nonce);
            for (doc_id, vector) in quality_rows {
                quality_writer.write_record(doc_id, vector)?;
            }
            quality_writer.finish()?;
            Some(binding)
        };

        // 7. Prove the staged generation by re-admitting it through the only
        //    legitimate v2 open path; the returned index's identity is
        //    header-attested by construction.
        let mut staged_paths = TwoTierIndexPaths::new(&staged_fast);
        if quality_binding.is_some() {
            staged_paths = staged_paths.with_quality_index(&staged_quality);
        }
        let index = TwoTierIndex::open_admitted_v2_with_paths(
            &staged_paths,
            self.config.index_config.clone(),
            &fast_binding,
            quality_binding.as_ref(),
        )?;
        debug_assert!(index.fast_identity_is_attested());
        // The staged pair shares one nonce by construction, so admission must
        // never have silently degraded the quality tier (round-2 hardening).
        debug_assert_eq!(quality_binding.is_some(), index.has_quality_index());

        info!(
            target: "frankensearch.refresh",
            staged_fast = %staged_fast.display(),
            staged_quality = quality_binding.is_some(),
            rows = merged.len(),
            generation_sequence = fast_sequence,
            "identity-bound replacement generation staged and admitted (non-canonical)"
        );

        Ok(StagedIdentityBoundGeneration {
            index,
            fast_path: staged_fast,
            quality_path: quality_binding.as_ref().map(|_| staged_quality),
            fast_binding,
            quality_binding,
        })
    }
}
