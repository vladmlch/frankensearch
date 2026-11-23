use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use tracing::{Instrument, info, info_span};

use super::runtime::{
    ShadowSearchPath, ShadowState, build_shadow_state, poll_shadow_on_pool, seed_shadow_corpus,
    shadow_status,
};
use super::types::{
    AlwaysAdmitShadowLoad, ShadowDegradationKind, ShadowDivergenceRecord, ShadowDocument,
    ShadowLexicalConfig, ShadowLoadProbe, ShadowObservationRecord, ShadowQuery, ShadowShedReason,
    ShadowStatus, classify_ranked_hits, micros, ranked_hits,
};
use super::{
    SHADOW_DIVERGENCE_SCHEMA_VERSION, SHADOW_OBSERVATION_SCHEMA_VERSION, SHADOW_SAMPLE_DENOMINATOR,
};
use crate::{Cx, IndexableDocument, ScoredResult, SearchFuture, SearchResult};

/// Query-only bridge for serving adapters whose richer direct result API
/// cannot itself implement [`crate::traits::LexicalRead`].
///
/// The bridge accepts an already-produced serving result slice and schedules
/// the same bounded comparison used by [`ShadowLexical`]. It exists for fsfs's
/// snippet-preserving direct Quill path; the serving results are never changed.
#[derive(Clone)]
pub struct ShadowLexicalObserver {
    shadow: Arc<dyn crate::traits::LexicalRead>,
    state: Arc<ShadowState>,
}

impl ShadowLexicalObserver {
    /// Build a query-only observer.
    ///
    /// # Errors
    ///
    /// Returns [`crate::SearchError::InvalidConfig`] for an invalid policy.
    pub fn new(
        shadow: Arc<dyn crate::traits::LexicalRead>,
        config: ShadowLexicalConfig,
    ) -> SearchResult<Self> {
        Self::with_load_probe(shadow, config, Arc::new(AlwaysAdmitShadowLoad))
    }

    /// Build an observer with an injected pressure probe.
    ///
    /// # Errors
    ///
    /// Returns [`crate::SearchError::InvalidConfig`] for an invalid policy.
    pub fn with_load_probe(
        shadow: Arc<dyn crate::traits::LexicalRead>,
        config: ShadowLexicalConfig,
        load_probe: Arc<dyn ShadowLoadProbe>,
    ) -> SearchResult<Self> {
        Ok(Self {
            shadow,
            state: build_shadow_state(config, load_probe)?,
        })
    }

    /// Seed the corpus required by exact-generation replay artifacts.
    pub fn seed_corpus(&self, documents: &[IndexableDocument], manifest_generation: u64) {
        seed_shadow_corpus(&self.state, documents, manifest_generation);
    }

    /// Mark the query-only oracle unhealthy at a successor serving generation.
    ///
    /// Existing tasks observe generation drift, and later queries are shed
    /// until the adapter constructs a matching oracle snapshot.
    pub fn mark_generation_degraded(&self, manifest_generation: u64, detail: impl Into<String>) {
        self.state
            .manifest_generation
            .store(manifest_generation, Ordering::Release);
        self.state.shadow_ready.store(false, Ordering::Release);
        self.state.record_degradation(
            manifest_generation,
            ShadowDegradationKind::GenerationDrift,
            detail,
            true,
        );
    }

    /// Persist a typed adapter-level failure without affecting serving.
    pub fn record_degradation(&self, kind: ShadowDegradationKind, detail: impl Into<String>) {
        self.state.record_degradation(
            self.state.manifest_generation.load(Ordering::Acquire),
            kind,
            detail,
            true,
        );
    }

    /// Submit already-produced serving results without awaiting shadow work.
    pub fn observe_serving_results(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        serving_results: &[ScoredResult],
        fusion_candidates: bool,
        serve_latency_micros: u64,
    ) {
        ShadowLexical::submit_shadow_parts(
            &self.shadow,
            &self.state,
            cx,
            query,
            limit,
            serving_results,
            if fusion_candidates {
                ShadowSearchPath::FusionCandidates
            } else {
                ShadowSearchPath::Search
            },
            serve_latency_micros,
        );
    }

    /// Snapshot in-process evidence counters.
    #[must_use]
    pub fn status(&self) -> ShadowStatus {
        shadow_status(&self.state)
    }
}

/// A lexical backend addressed through both split capabilities.
///
/// `dyn LexicalRead + LexicalWrite` is not expressible, so a backend that is
/// used for both is held as two coerced handles to the same object rather than
/// behind a re-unified trait — which would be the compatibility shim the split
/// exists to remove.
struct SplitLexical {
    read: Arc<dyn crate::traits::LexicalRead>,
    write: Arc<dyn crate::traits::LexicalWrite>,
}

impl SplitLexical {
    fn new<B>(backend: Arc<B>) -> Self
    where
        B: crate::traits::LexicalRead + crate::traits::LexicalWrite + 'static,
    {
        Self {
            read: backend.clone(),
            write: backend,
        }
    }
}

/// A serving-first lexical adapter with bounded, region-owned shadow queries.
pub struct ShadowLexical {
    serving: SplitLexical,
    shadow: SplitLexical,
    state: Arc<ShadowState>,
}

impl ShadowLexical {
    /// Build a default-off or explicitly enabled wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`crate::SearchError::InvalidConfig`] for an invalid sampling,
    /// task, score, or artifact policy.
    pub fn new<S, D>(
        serving: Arc<S>,
        shadow: Arc<D>,
        config: ShadowLexicalConfig,
    ) -> SearchResult<Self>
    where
        S: crate::traits::LexicalRead + crate::traits::LexicalWrite + 'static,
        D: crate::traits::LexicalRead + crate::traits::LexicalWrite + 'static,
    {
        Self::with_load_probe(serving, shadow, config, Arc::new(AlwaysAdmitShadowLoad))
    }

    /// Build a wrapper with an injected host-pressure probe.
    ///
    /// # Errors
    ///
    /// Returns [`crate::SearchError::InvalidConfig`] for an invalid policy.
    pub fn with_load_probe<S, D>(
        serving: Arc<S>,
        shadow: Arc<D>,
        config: ShadowLexicalConfig,
        load_probe: Arc<dyn ShadowLoadProbe>,
    ) -> SearchResult<Self>
    where
        S: crate::traits::LexicalRead + crate::traits::LexicalWrite + 'static,
        D: crate::traits::LexicalRead + crate::traits::LexicalWrite + 'static,
    {
        Ok(Self {
            serving: SplitLexical::new(serving),
            shadow: SplitLexical::new(shadow),
            state: build_shadow_state(config, load_probe)?,
        })
    }

    /// Seed exact-replay corpus state for indexes opened after a prior commit.
    pub fn seed_corpus(&self, documents: &[IndexableDocument], manifest_generation: u64) {
        seed_shadow_corpus(&self.state, documents, manifest_generation);
    }

    /// Enable or disable new shadow work. In-flight tasks remain region-owned
    /// and drain or cancel with their asupersync region.
    pub fn set_enabled(&self, enabled: bool) {
        self.state.enabled.store(enabled, Ordering::Release);
        info!(
            event = "shadow_lifecycle",
            enabled,
            manifest_generation = self.state.manifest_generation.load(Ordering::Acquire),
            "shadow lexical lifecycle state changed"
        );
    }

    /// Mark a rebuilt shadow index healthy or unhealthy.
    pub fn set_shadow_ready(&self, ready: bool) {
        self.state.shadow_ready.store(ready, Ordering::Release);
    }

    /// Snapshot deterministic in-process counters.
    #[must_use]
    pub fn status(&self) -> ShadowStatus {
        shadow_status(&self.state)
    }

    pub(super) fn submit_shadow(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        serving_results: &[ScoredResult],
        path: ShadowSearchPath,
        serve_latency_micros: u64,
    ) {
        Self::submit_shadow_parts(
            &self.shadow.read,
            &self.state,
            cx,
            query,
            limit,
            serving_results,
            path,
            serve_latency_micros,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit_shadow_parts(
        shadow: &Arc<dyn crate::traits::LexicalRead>,
        state: &Arc<ShadowState>,
        cx: &Cx,
        query: &str,
        limit: usize,
        serving_results: &[ScoredResult],
        path: ShadowSearchPath,
        serve_latency_micros: u64,
    ) {
        if !state.enabled.load(Ordering::Acquire) {
            return;
        }

        let generation = state.manifest_generation.load(Ordering::Acquire);
        let ordinal = state.sample_ordinal.fetch_add(1, Ordering::Relaxed);
        if ordinal % SHADOW_SAMPLE_DENOMINATOR >= u64::from(state.sample_rate_basis_points) {
            state.sampled_out.fetch_add(1, Ordering::Relaxed);
            info!(
                event = "shadow_query_shed",
                query,
                manifest_generation = generation,
                shed_reason = "sample_rate",
                sample_ordinal = ordinal,
                "shadow lexical query was sampled out without affecting serving"
            );
            return;
        }
        state.sampled.fetch_add(1, Ordering::Relaxed);

        if !state.shadow_ready.load(Ordering::Acquire) {
            state.record_shed(ShadowShedReason::ShadowDegraded, query, generation);
            return;
        }
        if let Some(reason) = state.load_probe.shed_reason() {
            state.record_shed(reason, query, generation);
            return;
        }
        if !state.allow_inline_shadow && cx.blocking_pool_handle().is_none() {
            state.record_shed(ShadowShedReason::BlockingPoolUnavailable, query, generation);
            state.record_degradation(
                generation,
                ShadowDegradationKind::BlockingPoolUnavailable,
                "shadow search requires a caller-owned blocking pool; attach one to the request context",
                false,
            );
            return;
        }
        let Some(in_flight) = state.try_admit() else {
            state.record_shed(ShadowShedReason::Capacity, query, generation);
            return;
        };

        let query = query.to_owned();
        let query_for_log = query.clone();
        let serving_top_k = ranked_hits(serving_results);
        let shadow = Arc::clone(shadow);
        let task_state = Arc::clone(state);
        let corpus_revision = task_state.corpus_revision.load(Ordering::Acquire);
        let span = info_span!(
            "shadow_lexical_query",
            query = %query,
            limit,
            manifest_generation = generation,
            serve_latency_micros,
            shadow_latency_micros = tracing::field::Empty,
            divergence_class = tracing::field::Empty,
            shed_reason = tracing::field::Empty,
        );
        let spawn_result = cx.spawn(move |task_cx| {
            async move {
                let in_flight = Arc::new(in_flight);
                let query_in_flight = Arc::clone(&in_flight);
                let query_cx = task_cx.clone();
                let shadow_query = query.clone();
                let shadow_started = Instant::now();
                let query_future = async move {
                    // Retain admission while a pool poll is still running,
                    // even if cancellation drops the waiting shadow task.
                    let _in_flight = query_in_flight;
                    match path {
                        ShadowSearchPath::Search => {
                            shadow.search(&query_cx, &shadow_query, limit).await
                        }
                        ShadowSearchPath::FusionCandidates => shadow
                            .search_candidates(&query_cx, &shadow_query, limit)
                            .await
                            // Comparison needs ranks, not hydrated metadata.
                            .map(|batch| batch.into_parts().0),
                    }
                }
                .instrument(tracing::Span::current());
                let outcome = poll_shadow_on_pool(
                    &task_cx,
                    Box::pin(query_future),
                    task_state.allow_inline_shadow,
                )
                .await;
                let shadow_latency_micros = micros(shadow_started.elapsed());
                tracing::Span::current().record("shadow_latency_micros", shadow_latency_micros);

                if generation != task_state.manifest_generation.load(Ordering::Acquire)
                    || corpus_revision != task_state.corpus_revision.load(Ordering::Acquire)
                {
                    tracing::Span::current()
                        .record("shed_reason", ShadowShedReason::GenerationDrift.label());
                    task_state.record_shed(ShadowShedReason::GenerationDrift, &query, generation);
                    task_state.record_degradation(
                        generation,
                        ShadowDegradationKind::GenerationDrift,
                        "manifest generation or corpus revision changed during shadow query",
                        true,
                    );
                    return;
                }

                let shadow_results = match outcome {
                    Ok(results) => results,
                    Err(error) => {
                        task_state.record_degradation(
                            generation,
                            ShadowDegradationKind::Search,
                            error.to_string(),
                            true,
                        );
                        return;
                    }
                };
                let shadow_top_k = ranked_hits(&shadow_results);
                let classification =
                    classify_ranked_hits(&serving_top_k, &shadow_top_k, task_state.score_epsilon);
                tracing::Span::current().record("divergence_class", classification.label());
                task_state.completed.fetch_add(1, Ordering::Relaxed);
                let observation = ShadowObservationRecord {
                    schema_version: SHADOW_OBSERVATION_SCHEMA_VERSION,
                    manifest_generation: generation,
                    query: query.clone(),
                    classification,
                    serve_latency_micros,
                    shadow_latency_micros,
                };
                if let Err(error) = task_state.sink.append_observation(&observation) {
                    task_state.record_degradation(
                        generation,
                        ShadowDegradationKind::ArtifactWrite,
                        error.to_string(),
                        false,
                    );
                }
                if !classification.is_artifact() {
                    return;
                }

                task_state.divergences.fetch_add(1, Ordering::Relaxed);
                let (corpus_hash, documents) = task_state.corpus_snapshot();
                let record = ShadowDivergenceRecord {
                    schema_version: SHADOW_DIVERGENCE_SCHEMA_VERSION,
                    manifest_generation: generation,
                    corpus_hash,
                    documents,
                    query: ShadowQuery {
                        text: query,
                        limit,
                        fusion_candidates: path.fusion_candidates(),
                    },
                    serving_top_k,
                    shadow_top_k,
                    classification,
                    serve_latency_micros,
                    shadow_latency_micros,
                };
                if let Err(error) = task_state.sink.append_divergence(&record) {
                    task_state.record_degradation(
                        generation,
                        ShadowDegradationKind::ArtifactWrite,
                        error.to_string(),
                        false,
                    );
                }
            }
            .instrument(span)
        });
        if let Err(error) = spawn_result {
            state.record_shed(
                ShadowShedReason::RuntimeUnavailable,
                &query_for_log,
                generation,
            );
            state.record_degradation(
                generation,
                ShadowDegradationKind::RuntimeUnavailable,
                error.to_string(),
                false,
            );
        }
    }

    fn record_corpus_documents(&self, documents: &[IndexableDocument]) {
        let mut corpus = self
            .state
            .corpus
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        corpus.extend(
            documents
                .iter()
                .map(|document| (document.id.clone(), ShadowDocument::from(document))),
        );
        drop(corpus);
        self.state.corpus_revision.fetch_add(1, Ordering::AcqRel);
    }

    async fn mirror_documents(&self, cx: &Cx, documents: &[IndexableDocument]) {
        if !self.state.enabled.load(Ordering::Acquire)
            || !self.state.shadow_ready.load(Ordering::Acquire)
        {
            return;
        }
        if let Err(error) = self.shadow.write.index_documents(cx, documents).await {
            self.state.shadow_ready.store(false, Ordering::Release);
            self.state.record_degradation(
                self.state.manifest_generation.load(Ordering::Acquire),
                ShadowDegradationKind::Index,
                error.to_string(),
                true,
            );
        }
    }
}

impl crate::traits::LexicalRead for ShadowLexical {
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let serve_started = Instant::now();
            let results = self.serving.read.search(cx, query, limit).await?;
            self.submit_shadow(
                cx,
                query,
                limit,
                &results,
                ShadowSearchPath::Search,
                micros(serve_started.elapsed()),
            );
            Ok(results)
        })
    }

    /// Forward the serving backend's candidate batch, pin and all.
    ///
    /// The batch's hydration context belongs to the serving backend, and
    /// [`LexicalRead::hydrate_candidates`](crate::traits::LexicalRead::hydrate_candidates) forwards there too, so the pin stays
    /// coherent end to end and the decorator never mints or inspects one.
    ///
    /// A previous note here claimed the eager default was deliberate because
    /// threading the serving backend's private context through the shadow
    /// comparison was unsound. It is not needed: the comparison consumes
    /// `batch.results()`, exactly the metadata-less rows the combined trait's
    /// `search_fusion_candidates` already submitted. Taking the eager default
    /// instead silently dropped the serving backend's deferred path — the same
    /// regression `bd-8nqz.1` fixed for Tantivy.
    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, crate::traits::LexicalCandidateBatch> {
        Box::pin(async move {
            let serve_started = Instant::now();
            let batch = self
                .serving
                .read
                .search_candidates(cx, query, limit)
                .await?;
            self.submit_shadow(
                cx,
                query,
                limit,
                batch.results(),
                ShadowSearchPath::FusionCandidates,
                micros(serve_started.elapsed()),
            );
            Ok(batch)
        })
    }

    fn hydrate_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        context: Option<&'a crate::traits::LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        self.serving.read.hydrate_candidates(cx, context, results)
    }

    fn doc_count(&self) -> SearchResult<usize> {
        self.serving.read.doc_count()
    }
}

impl crate::traits::LexicalWrite for ShadowLexical {
    fn index_document<'a>(
        &'a self,
        cx: &'a Cx,
        document: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.serving.write.index_document(cx, document).await?;
            self.record_corpus_documents(std::slice::from_ref(document));
            self.mirror_documents(cx, std::slice::from_ref(document))
                .await;
            Ok(())
        })
    }

    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        documents: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.serving.write.index_documents(cx, documents).await?;
            self.record_corpus_documents(documents);
            self.mirror_documents(cx, documents).await;
            Ok(())
        })
    }

    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.serving.write.commit(cx).await?;
            let generation = self
                .state
                .manifest_generation
                .fetch_add(1, Ordering::AcqRel)
                .saturating_add(1);
            if self.state.enabled.load(Ordering::Acquire)
                && self.state.shadow_ready.load(Ordering::Acquire)
                && let Err(error) = self.shadow.write.commit(cx).await
            {
                self.state.shadow_ready.store(false, Ordering::Release);
                self.state.record_degradation(
                    generation,
                    ShadowDegradationKind::Commit,
                    error.to_string(),
                    true,
                );
            }
            Ok(())
        })
    }
}
