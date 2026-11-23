use std::sync::{Arc, Mutex};

#[cfg(feature = "graph")]
use frankensearch_core::DocumentGraph;
use frankensearch_core::canonicalize::Canonicalizer;
use frankensearch_core::host_adapter::HostAdapter;
use frankensearch_core::traits::{Embedder, LexicalRead, Reranker};
use frankensearch_core::{
    LiveSearchFrame, LiveSearchStreamEmitter, RuntimeMetricsCollector, SearchStreamHealth,
};
use frankensearch_embed::CachedEmbedder;
use frankensearch_index::SearchParams;

use crate::adaptive::AdaptiveFusion;
use crate::calibration::CalibratorConfig;
use crate::circuit_breaker::CircuitBreaker;
use crate::conformal::{AdaptiveConformalState, ConformalSearchCalibration};
use crate::mmr::MmrConfig;
use crate::normalize::{AdaptiveNqcDenseWeight, NqcDenseWeight};
use crate::phase_gate::{PhaseGate, PhaseGateConfig};
use crate::prf::PrfConfig;
use crate::rrf::RrfTiebreak;

use super::TwoTierSearcher;

impl TwoTierSearcher {
    /// Set the quality-tier embedder for progressive refinement.
    ///
    /// If `with_embedding_cache` was called first, the quality embedder is
    /// automatically wrapped with a `CachedEmbedder` at the same capacity.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        if let Some(cap) = self.embedding_cache_capacity {
            self.quality_embedder = Some(Arc::new(CachedEmbedder::new(embedder, cap)));
        } else {
            self.quality_embedder = Some(embedder);
        }
        self
    }

    /// Set the lexical search backend for hybrid RRF fusion.
    ///
    /// Read-only by contract (bd-8nqz.1): the searcher never indexes or
    /// commits through this handle, so any [`LexicalRead`] — including a
    /// leaseless read-only reader — is a valid backend.
    #[must_use]
    pub fn with_lexical(mut self, lexical: Arc<dyn LexicalRead>) -> Self {
        self.lexical = Some(lexical);
        self
    }

    /// Set per-tier RRF fusion weights (default `1.0` / `1.0` = neutral).
    ///
    /// Up-weighting the *stronger* tier for the workload (~1.3×) makes the hybrid strictly
    /// dominate the best single tier on both recall and nDCG (see `docs/NEGATIVE_EVIDENCE.md`).
    /// Non-finite or `≤ 0` values fall back to `1.0`.
    #[must_use]
    pub fn with_rrf_weights(mut self, lexical_weight: f64, semantic_weight: f64) -> Self {
        self.rrf_lexical_weight = lexical_weight;
        self.rrf_semantic_weight = semantic_weight;
        self
    }

    /// Set the RRF tie-break strategy (default [`RrfTiebreak::LexicalThenId`]).
    ///
    /// [`RrfTiebreak::Hash`] breaks score ties by an unbiased hash of `doc_id` instead of
    /// favoring the lexical tier — it stops the legacy tiebreak from systematically
    /// demoting vector-only best-answers (see `docs/NEGATIVE_EVIDENCE.md`).
    #[must_use]
    pub fn with_rrf_tiebreak(mut self, tiebreak: RrfTiebreak) -> Self {
        self.rrf_tiebreak = tiebreak;
        self
    }

    /// Enable the opt-in **NQC dense down-weight** (default OFF), applied at the phase-1
    /// hybrid fusion.
    ///
    /// Per query, the dense tier's phase-1 fusion weight is scaled by
    /// `clip(1 − beta·CDF(nqc_cv(lexical scores)), w_min, 1)`, where `CDF` is the empirical
    /// percentile from `weight` — a [`NqcDenseWeight`] built offline from a sample of observed
    /// NQC values. High lexical commitment (high NQC), where the dense tier tends to add little
    /// or hurt, gets a lower dense weight. See `docs/SEARCH_QUALITY_FINDINGS.md`.
    ///
    /// `beta <= 0` (the default) or an empty `weight` leaves fusion **byte-identical**. Use
    /// `w_min > 0` (the measured `beta ≈ 0.5` floors the multiplier at `0.5`): a scaled weight
    /// reaching `<= 0` is treated as neutral `1.0` by the tier-weight sanitizer. NOTE: this
    /// applies at PHASE-1 only; phase-2 refines the vector tiers from that seeded ranking (see
    /// the async-wiring trace in `docs/NEGATIVE_EVIDENCE.md`) — validate the refined effect via A/B.
    #[must_use]
    pub fn with_nqc_dense_downweight(
        mut self,
        beta: f32,
        w_min: f32,
        weight: NqcDenseWeight,
    ) -> Self {
        self.nqc_downweight_beta = beta;
        self.nqc_downweight_w_min = w_min;
        self.nqc_dense_weight = weight;
        // An explicit static sketch overrides the default-on adaptive down-weight (which would
        // otherwise take precedence in `effective_semantic_weight`).
        self.nqc_adaptive = None;
        self
    }

    /// Enable the **self-driving rolling** NQC dense down-weight (**on by default**; this rebuilds
    /// it with explicit parameters).
    ///
    /// Unlike [`Self::with_nqc_dense_downweight`] (a caller-supplied static sketch), this builds
    /// and refreshes the cv→percentile sketch *online* from the observed query stream via an
    /// [`AdaptiveNqcDenseWeight`] — no external sample management. It takes precedence over the
    /// static sketch when both are set. Cold start is **neutral** (empty sketch → weight `1.0`,
    /// byte-identical fusion) until `min_samples` queries warm it, so it is safe to enable by
    /// default. `beta <= 0` disables it. See [`AdaptiveNqcDenseWeight::new`] for the parameters.
    #[must_use]
    pub fn with_nqc_dense_downweight_adaptive(
        mut self,
        beta: f32,
        w_min: f32,
        capacity: usize,
        min_samples: usize,
        rebuild_every: usize,
    ) -> Self {
        self.nqc_adaptive = Some(Mutex::new(AdaptiveNqcDenseWeight::new(
            beta,
            w_min,
            capacity,
            min_samples,
            rebuild_every,
        )));
        self
    }

    /// Enable the rolling NQC dense down-weight with the blessed production defaults
    /// ([`AdaptiveNqcDenseWeight::production_default`]) — the recommended one-call setup.
    #[must_use]
    pub fn with_nqc_dense_downweight_adaptive_defaults(mut self) -> Self {
        self.nqc_adaptive = Some(Mutex::new(AdaptiveNqcDenseWeight::production_default()));
        self
    }

    /// Disable any NQC dense down-weight (static or adaptive) → **byte-identical** fusion.
    /// The opt-out / A-B escape hatch (also clears a default-on adaptive down-weight).
    #[must_use]
    pub fn with_nqc_dense_downweight_disabled(mut self) -> Self {
        self.nqc_adaptive = None;
        self.nqc_downweight_beta = 0.0;
        self
    }

    /// Set the cross-encoder reranker for Phase 2.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Choose how the cross-encoder's rerank scores combine with the retrieval order.
    ///
    /// Defaults to [`frankensearch_rerank::RerankCombine::PureReorder`] (legacy behavior).
    /// [`frankensearch_rerank::RerankCombine::RrfCombine`] is the measured-safe integration
    /// that rank-fuses the pre-rerank order with the rerank order so retrieval vetoes deep
    /// false positives the reranker would promote (see `docs/NEGATIVE_EVIDENCE.md`).
    #[cfg(feature = "rerank")]
    #[must_use]
    pub fn with_rerank_combine(mut self, combine: frankensearch_rerank::RerankCombine) -> Self {
        self.rerank_combine = combine;
        self
    }

    /// Configure pseudo-relevance feedback for quality query expansion.
    #[must_use]
    pub const fn with_prf_config(mut self, config: PrfConfig) -> Self {
        self.prf_config = config;
        self
    }

    /// Configure MMR diversity reranking for final results.
    #[must_use]
    pub const fn with_mmr_config(mut self, config: MmrConfig) -> Self {
        self.mmr_config = config;
        self
    }

    /// Attach adaptive fusion posteriors used for dynamic RRF-K and blend weights.
    ///
    /// The adaptive state is shared so callers can feed explicit feedback events.
    #[must_use]
    pub fn with_adaptive_fusion(mut self, adaptive_fusion: Arc<AdaptiveFusion>) -> Self {
        self.adaptive_fusion = Some(adaptive_fusion);
        self
    }

    /// Calibrate semantic scores before blending/refinement.
    #[must_use]
    pub fn with_score_calibrator(mut self, calibrator: CalibratorConfig) -> Self {
        self.score_calibrator = Some(calibrator);
        self
    }

    /// Enable a quality-tier circuit breaker for automatic Phase-2 skipping.
    #[must_use]
    pub fn with_circuit_breaker(mut self, circuit_breaker: Arc<CircuitBreaker>) -> Self {
        self.circuit_breaker = Some(circuit_breaker);
        self
    }

    /// Enable sequential-testing phase gating for automatic Phase-2 skipping.
    #[must_use]
    pub fn with_phase_gate(mut self, phase_gate: PhaseGate) -> Self {
        self.phase_gate = Some(Mutex::new(phase_gate));
        self
    }

    /// Configure phase gating directly from a gate config.
    #[must_use]
    pub fn with_phase_gate_config(self, config: PhaseGateConfig) -> Self {
        self.with_phase_gate(PhaseGate::new(config))
    }

    /// Attach conformal calibration used to scale candidate budgets.
    #[must_use]
    pub fn with_conformal_calibration(mut self, calibration: ConformalSearchCalibration) -> Self {
        self.conformal_calibration = Some(calibration);
        self
    }

    /// Attach adaptive conformal state used with conformal candidate budgeting.
    #[allow(clippy::missing_const_for_fn)]
    #[must_use]
    pub fn with_adaptive_conformal_state(mut self, state: AdaptiveConformalState) -> Self {
        self.adaptive_conformal_state = Some(Mutex::new(state));
        self
    }

    /// Override brute-force vector-search parallelism parameters for Phase 1.
    ///
    /// An explicit override selects exact owner-backed scanning instead of an
    /// opted-in native fast graph; the actual bound query is still admitted.
    /// Legacy ANN dispatch retains its existing parameter behavior.
    #[must_use]
    pub const fn with_search_params(mut self, params: SearchParams) -> Self {
        self.search_params = Some(params);
        self
    }

    /// Convert a single searcher into a federated searcher seed.
    ///
    /// This provides a direct wiring path from `TwoTierSearcher` into
    /// `FederatedSearcher` without requiring callers to manually bridge types.
    #[must_use]
    pub fn into_federated(
        self: Arc<Self>,
        name: impl Into<String>,
        weight: f32,
    ) -> crate::federated::FederatedSearcher {
        crate::federated::FederatedSearcher::new().add_index(name, self, weight)
    }

    /// Attach an optional document graph used for phase-1 graph ranking.
    #[cfg(feature = "graph")]
    #[must_use]
    pub fn with_document_graph(mut self, graph: Arc<DocumentGraph>) -> Self {
        self.document_graph = Some(graph);
        self
    }

    /// Attach the per-doc query-hubness table `r_d`, indexed by [`frankensearch_core::types::VectorHit::index`] (i.e. aligned
    /// with the vector store), as produced by [`crate::hubness::compute_query_hubness`] from a
    /// background query sample.
    ///
    /// Inert unless [`frankensearch_core::config::TwoTierConfig::hubness_beta`] is `> 0.0`. `r_d` must be a *query-distribution*
    /// statistic: query-free proxies were measured and rejected (`64ac8b7`).
    #[must_use]
    pub fn with_hubness_table(mut self, hubness: Arc<[f32]>) -> Self {
        self.hubness_table = Some(hubness);
        self
    }

    /// Set the host adapter used to receive canonical telemetry envelopes.
    #[must_use]
    pub fn with_host_adapter(mut self, host_adapter: Arc<dyn HostAdapter>) -> Self {
        self.host_adapter = Some(host_adapter);
        self
    }

    /// Override the runtime telemetry collector used for canonical envelope assembly.
    #[must_use]
    pub fn with_runtime_metrics_collector(
        mut self,
        collector: Arc<RuntimeMetricsCollector>,
    ) -> Self {
        self.runtime_metrics_collector = collector;
        self
    }

    /// Override the live-search stream emitter used for timeline/live-feed frames.
    #[must_use]
    pub fn with_live_search_stream_emitter(
        mut self,
        emitter: Arc<LiveSearchStreamEmitter>,
    ) -> Self {
        self.live_search_stream_emitter = emitter;
        self
    }

    /// Override the query canonicalizer.
    #[must_use]
    pub fn with_canonicalizer(mut self, canonicalizer: Box<dyn Canonicalizer>) -> Self {
        self.canonicalizer = canonicalizer;
        self
    }

    /// Wrap the fast (and quality, if set) embedders with a query embedding cache.
    ///
    /// Raw requests may reuse raw vectors. All identified tiers instead use
    /// the bound operation, preserving the response's complete identity; raw
    /// cache entries cannot satisfy those requests. `capacity` controls the
    /// maximum cache entries per embedder.
    ///
    /// Safe to call in any builder order: if `with_quality_embedder` is called
    /// later, the quality embedder is automatically wrapped at the same capacity.
    #[must_use]
    pub fn with_embedding_cache(mut self, capacity: usize) -> Self {
        self.embedding_cache_capacity = Some(capacity);
        self.fast_embedder = Arc::new(CachedEmbedder::new(self.fast_embedder, capacity));
        if let Some(qe) = self.quality_embedder.take() {
            self.quality_embedder = Some(Arc::new(CachedEmbedder::new(qe, capacity)));
        }
        self
    }

    /// Snapshot live-search stream health counters.
    #[must_use]
    pub fn live_search_stream_health(&self) -> SearchStreamHealth {
        self.live_search_stream_emitter.health()
    }

    /// Drain buffered live-search frames from oldest to newest.
    #[must_use]
    pub fn drain_live_search_stream(&self, max_items: usize) -> Vec<LiveSearchFrame> {
        self.live_search_stream_emitter.drain(max_items)
    }

    /// Reset the phase gate, clearing accumulated evidence and any decision.
    ///
    /// Call this after index rebuilds or significant corpus changes so the gate
    /// re-evaluates whether quality refinement is beneficial for the new data.
    /// No-op if the searcher was not configured with a phase gate.
    pub fn reset_phase_gate(&self) {
        if let Some(phase_gate) = self.phase_gate.as_ref() {
            let _ = phase_gate
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .reset();
        }
    }
}
