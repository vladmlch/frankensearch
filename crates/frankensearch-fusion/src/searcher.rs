//! Progressive two-tier search orchestrator.
//!
//! [`TwoTierSearcher`] coordinates fast-tier and quality-tier search phases,
//! delivering results incrementally via an async callback. Consumers see
//! fast results in ~15ms, then optionally receive quality-refined results.
//!
//! # Callback Protocol
//!
//! The `on_phase` callback fires at most twice:
//! 1. [`frankensearch_core::types::SearchPhase::Initial`] — fast-tier results (always fired if search starts).
//! 2. [`frankensearch_core::types::SearchPhase::Refined`] or [`frankensearch_core::types::SearchPhase::RefinementFailed`] — quality tier
//!    result or graceful degradation (only fired when quality embedder is available
//!    and `fast_only` is false).

mod admission;
mod builders;
mod orchestration;
mod results;
mod scoring;
mod telemetry;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ahash::AHashMap;
use asupersync::Cx;

#[cfg(feature = "graph")]
use frankensearch_core::DocumentGraph;
use frankensearch_core::ParsedQuery;
use frankensearch_core::canonicalize::{Canonicalizer, DefaultCanonicalizer};
use frankensearch_core::config::{TwoTierConfig, TwoTierMetrics, ZeroSignalReason};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::explanation::{
    ExplainedSource, ExplanationPhase, HitExplanation, RankMovement, ScoreComponent,
};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::host_adapter::HostAdapter;
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::traits::{Embedder, LexicalHydrationContext, LexicalRead, Reranker};
use frankensearch_core::types::{
    BoundQueryEmbedding, ScoreSource, ScoredResult, TieredQueryEmbeddings, VectorHit,
};
use frankensearch_core::{
    EmbeddingStage, EmbeddingStatus, LiveSearchStreamEmitter, RuntimeMetricsCollector,
};
use frankensearch_index::{SearchParams, TwoTierIndex};

use crate::adaptive::{AdaptiveFusion, SignalSource};
use crate::blend::{
    blend_two_tier, blend_two_tier_aligned_vector_index, build_borrowed_rank_map,
    compute_rank_changes_with_maps, kendall_tau_with_refined_rank,
};
use crate::calibration::CalibratorConfig;
use crate::circuit_breaker::CircuitBreaker;
use crate::conformal::{AdaptiveConformalState, ConformalSearchCalibration};
#[cfg(feature = "graph")]
use crate::graph_rank::GraphRanker;
use crate::mmr::MmrConfig;
use crate::normalize::{AdaptiveNqcDenseWeight, NqcDenseWeight};
use crate::phase_gate::PhaseGate;
use crate::prf::{PrfConfig, prf_expand};
use crate::rrf::{
    RrfConfig, RrfTiebreak, candidate_count, fuse_by_strategy, fuse_by_strategy_for_vector_lane,
};

use admission::{
    QueryEmbeddingIdentities, SemanticAdmission, admit_legacy_tier_embedder, admit_tier_embedder,
    cancellation_checkpoint, capture_query_identity, embed_admitted_query,
    terminal_query_embedding_error,
};
use results::{
    NormalizedExclusions, filter_scored_results_by_negations, filter_vector_hits_by_negations,
    fused_hits_to_scored_results, is_shipped_hash_embedder, poll_immediate,
    promoted_quality_source, scaled_budget, should_lexical_short_circuit, vector_exclusion_source,
    vector_hits_to_scored_results,
};
use telemetry::CpuJiffiesSnapshot;

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub use results::{
    bench_normalize_for_negation_match_fast, bench_normalize_for_negation_match_reference,
};

/// Progressive two-tier search orchestrator.
///
/// Coordinates fast-tier embedding, optional lexical search, RRF fusion,
/// optional quality-tier refinement, and optional cross-encoder reranking.
///
/// # Usage
///
/// ```rust,ignore
/// let searcher = TwoTierSearcher::new(index, fast_embedder, config)
///     .with_quality_embedder(quality)
///     .with_lexical(tantivy)
///     .with_reranker(reranker);
///
/// let metrics = searcher.search(&cx, "distributed consensus", 10, text_fn, |phase| {
///     match phase {
///         SearchPhase::Initial { results, .. } => display(&results),
///         SearchPhase::Refined { results, .. } => update_display(&results),
///         SearchPhase::RefinementFailed { .. } => { /* keep initial */ }
///     }
/// }).await?;
/// ```
pub struct TwoTierSearcher {
    index: Arc<TwoTierIndex>,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    lexical: Option<Arc<dyn LexicalRead>>,
    reranker: Option<Arc<dyn Reranker>>,
    #[cfg(feature = "rerank")]
    rerank_combine: frankensearch_rerank::RerankCombine,
    host_adapter: Option<Arc<dyn HostAdapter>>,
    runtime_metrics_collector: Arc<RuntimeMetricsCollector>,
    live_search_stream_emitter: Arc<LiveSearchStreamEmitter>,
    canonicalizer: Box<dyn Canonicalizer>,
    config: TwoTierConfig,
    prf_config: PrfConfig,
    mmr_config: MmrConfig,
    /// Per-tier RRF weights + tiebreak applied to hybrid fusion (default: neutral 1.0 /
    /// legacy tiebreak). See [`RrfConfig::lexical_weight`] / [`RrfTiebreak`].
    rrf_lexical_weight: f64,
    rrf_semantic_weight: f64,
    rrf_tiebreak: RrfTiebreak,
    /// Opt-in NQC dense down-weight (default off, `beta = 0.0`). Applied at the phase-1
    /// hybrid fusion. See [`Self::with_nqc_dense_downweight`].
    nqc_downweight_beta: f32,
    nqc_downweight_w_min: f32,
    nqc_dense_weight: NqcDenseWeight,
    /// Self-driving rolling NQC dense down-weight — **on by default**
    /// ([`AdaptiveNqcDenseWeight::production_default`]). Takes precedence over the static sketch
    /// above and learns the query distribution online. See
    /// [`Self::with_nqc_dense_downweight_adaptive`] / [`Self::with_nqc_dense_downweight_disabled`].
    /// Behind a `Mutex` so per-query observation mutates the rolling state while `search` stays
    /// `&self`.
    nqc_adaptive: Option<Mutex<AdaptiveNqcDenseWeight>>,
    adaptive_fusion: Option<Arc<AdaptiveFusion>>,
    score_calibrator: Option<CalibratorConfig>,
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    phase_gate: Option<Mutex<PhaseGate>>,
    conformal_calibration: Option<ConformalSearchCalibration>,
    adaptive_conformal_state: Option<Mutex<AdaptiveConformalState>>,
    search_params: Option<SearchParams>,
    #[cfg(feature = "graph")]
    document_graph: Option<Arc<DocumentGraph>>,
    /// Per-doc query-hubness `r_d`, indexed by [`VectorHit::index`]. See `with_hubness_table`.
    hubness_table: Option<Arc<[f32]>>,
    /// When set, `with_quality_embedder` auto-wraps with `CachedEmbedder`.
    embedding_cache_capacity: Option<usize>,
    resource_cpu_state: Mutex<Option<CpuJiffiesSnapshot>>,
}

/// Candidate pools retained from Phase 1 for Phase-2 refinement.
///
/// The Initial display stays bounded to `k`, but refinement must see the
/// full `k * candidate_multiplier` pool: quality scores can promote
/// candidates that never reached the displayed page, and the lexical pool is
/// re-fused against blended vector scores without a second lexical query
/// (sync-searcher parity, bd-tm83).
struct RefinementPools {
    /// Semantic pool after calibration, exclusion filtering, and pool
    /// corrections — exactly what Phase-1 fusion consumed.
    fast_hits: Vec<VectorHit>,
    /// Lexical pool after exclusion filtering. `None` when no lexical
    /// subsystem is configured; `Some` (possibly empty) when one answered.
    lexical: Option<Vec<ScoredResult>>,
}

/// What the quality tier contributed to a refinement (bd-ctzo C2).
///
/// The two variants are not two encodings of one thing — they answer
/// different questions. `Retrieved` is the quality tier's OWN top-k, which can
/// name documents the fast tier never selected. `RescoredFastPool` can only
/// ever re-rank documents the fast tier already chose, so a relevant document
/// outside the fast pool is invisible to it. The second remains the path for
/// legacy artifacts, where there is no admitted quality owner to retrieve
/// from.
enum QualityPool {
    /// The quality owner's own top-k for this query.
    Retrieved(Vec<VectorHit>),
    /// Quality scores aligned positionally to the fast pool.
    RescoredFastPool(Vec<Option<f32>>),
}

/// Phase-1 fused results plus the refinement pools when refinement is
/// meaningful. Short-circuit and fallback exits carry no pools because the
/// Phase-2 gate cannot open on those paths.
struct Phase1Outcome {
    /// Fused candidates bounded to `k * candidate_multiplier`.
    results: Vec<ScoredResult>,
    /// Retained pools for Phase-2 refinement, when applicable.
    pools: Option<RefinementPools>,
}

impl Phase1Outcome {
    /// Wrap an exit that carries no refinement pools.
    fn without_pools(results: Vec<ScoredResult>) -> Self {
        Self {
            results,
            pools: None,
        }
    }
}

impl TwoTierSearcher {
    /// Create a new searcher with a fast-tier embedder.
    #[must_use]
    pub fn new(
        index: Arc<TwoTierIndex>,
        fast_embedder: Arc<dyn Embedder>,
        config: TwoTierConfig,
    ) -> Self {
        if let Err(e) = config.validate() {
            tracing::warn!(
                error = %e,
                "Invalid TwoTierConfig supplied; search behavior may be degenerate"
            );
        }

        // Say once, at construction, that this searcher has no semantic fast
        // tier. Per-query the condition is already recorded as
        // `skip_reason = "non_semantic_fast_embedder_lexical_short_circuit"`,
        // but a metrics field nobody reads is why hash-only deployments look
        // healthy: queries succeed and return plausible lexical hits. Logging
        // it per query would spam a hot path, so it belongs here (`bd-a6zt`).
        if is_shipped_hash_embedder(fast_embedder.as_ref()) {
            tracing::warn!(
                fast_embedder = %fast_embedder.id(),
                "TwoTierSearcher constructed with a NON-SEMANTIC hash fast embedder; semantic \
                 retrieval is unavailable and results will be lexical-only where a lexical \
                 backend is attached. If a semantic model is expected, the index generation \
                 must be rebuilt with it — installing a model alone does not repair an index \
                 whose vectors carry hash identity",
            );
        }

        Self {
            index,
            fast_embedder,
            quality_embedder: None,
            lexical: None,
            reranker: None,
            #[cfg(feature = "rerank")]
            rerank_combine: frankensearch_rerank::RerankCombine::PureReorder,
            host_adapter: None,
            runtime_metrics_collector: Arc::new(RuntimeMetricsCollector::default()),
            live_search_stream_emitter: Arc::new(LiveSearchStreamEmitter::default()),
            canonicalizer: Box::new(DefaultCanonicalizer::default()),
            config,
            prf_config: PrfConfig::default(),
            mmr_config: MmrConfig::default(),
            rrf_lexical_weight: 1.0,
            rrf_semantic_weight: 1.0,
            rrf_tiebreak: RrfTiebreak::LexicalThenId,
            nqc_downweight_beta: 0.0,
            nqc_downweight_w_min: 0.0,
            nqc_dense_weight: NqcDenseWeight::new(),
            // Self-driving NQC dense down-weight ON BY DEFAULT — neutral during its 128-query
            // warm-up, then realizing the measured +0.0022 nDCG@10 gain. Opt out via
            // `with_nqc_dense_downweight_disabled`; an explicit static sketch also overrides it.
            nqc_adaptive: Some(Mutex::new(AdaptiveNqcDenseWeight::production_default())),
            adaptive_fusion: None,
            score_calibrator: None,
            circuit_breaker: None,
            phase_gate: None,
            conformal_calibration: None,
            adaptive_conformal_state: None,
            search_params: None,
            #[cfg(feature = "graph")]
            document_graph: None,
            hubness_table: None,
            embedding_cache_capacity: None,
            resource_cpu_state: Mutex::new(None),
        }
    }

    /// Decide whether this query may touch this index's vectors, before it is
    /// embedded (bd-ctzo C1).
    ///
    /// The join is between two things that exist before any inference runs:
    /// the identity bundle the configured embedder declares, and the binding
    /// each tier was ADMITTED under and now retains
    /// ([`TwoTierIndex::fast_admitted_binding`]). It applies the same
    /// bd-9xuj admission law
    /// ([`EmbeddingIdentityBundleV1::verify_exact_producer_with`]) that
    /// [`TwoTierIndex::activate_owner_backed_search`] applies to the bound
    /// query afterwards. A refused configuration never reaches an embedder.
    /// The captured query contracts also admit every actual bound response
    /// after inference, before feedback-vector reads, exact scoring or graph
    /// traversal. A provider cannot relabel raw values with a later identity.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] naming the tier and contract
    /// field when a stored v1 producer fingerprint differs, or when an
    /// owner-backed tier's embedder is not identity-aware, produces a different
    /// embedding space, or is a different
    /// producer than the one that wrote the index — including the
    /// certified-compatible case, which is comparison-grade telemetry and
    /// never an admission basis.
    fn admit_semantic_query(&self) -> SearchResult<(SemanticAdmission, QueryEmbeddingIdentities)> {
        let identities = QueryEmbeddingIdentities {
            fast: capture_query_identity(self.fast_embedder.as_ref(), "fast")?,
            quality: self
                .quality_embedder
                .as_ref()
                .filter(|_| self.index.has_quality_index())
                .map(|embedder| capture_query_identity(embedder.as_ref(), "quality"))
                .transpose()?
                .flatten(),
        };
        let Some(fast_binding) = self.index.fast_admitted_binding() else {
            if self.index.quality_admitted_binding().is_some() {
                return Err(SearchError::InvalidConfig {
                    field: "search_activation.mixed_admission".to_owned(),
                    value: "quality".to_owned(),
                    reason: "this index retains an admitted quality owner but a legacy fast \
                             tier; a half-attested pair has no coherent identity to search \
                             under and requires a reindex"
                        .to_owned(),
                });
            }
            admit_legacy_tier_embedder(
                self.fast_embedder.as_ref(),
                identities.fast.as_ref(),
                self.index.fast_embedder_revision(),
                "fast",
            )?;
            if let (Some(embedder), Some(revision)) = (
                self.quality_embedder.as_ref(),
                self.index.quality_embedder_revision(),
            ) {
                admit_legacy_tier_embedder(
                    embedder.as_ref(),
                    identities.quality.as_ref(),
                    revision,
                    "quality",
                )?;
            }
            return Ok((SemanticAdmission::LegacyUnidentified, identities));
        };
        admit_tier_embedder(
            self.fast_embedder.as_ref(),
            identities.fast.as_ref(),
            fast_binding,
            "fast",
        )?;
        // The quality tier is admitted only when BOTH halves are present. A
        // configured quality embedder with no retained quality owner is not
        // an error — `should_run_quality` already keeps such a search on the
        // Initial phase — and a retained quality owner with no configured
        // embedder is simply never consulted.
        let quality = match (
            self.quality_embedder.as_ref(),
            self.index.quality_admitted_binding(),
        ) {
            (Some(embedder), Some(binding)) => {
                admit_tier_embedder(
                    embedder.as_ref(),
                    identities.quality.as_ref(),
                    binding,
                    "quality",
                )?;
                true
            }
            _ => false,
        };
        Ok((SemanticAdmission::OwnerBacked { quality }, identities))
    }

    /// Run Phase 1: fast embedding + optional lexical + RRF fusion.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    async fn run_phase1(
        &self,
        cx: &Cx,
        semantic_query: &str,
        _parsed_query: &ParsedQuery,
        normalized_exclusions: Option<&NormalizedExclusions>,
        k: usize,
        query_class: QueryClass,
        identity: Option<&EmbeddingIdentityBundleV1>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        metrics: &mut TwoTierMetrics,
        root_request_id: Option<&str>,
    ) -> SearchResult<Phase1Outcome> {
        let candidate_target = self.conformal_candidate_target(k);
        let base_candidates =
            candidate_count(candidate_target, 0, self.config.candidate_multiplier.max(1));

        let lexical_short_circuit =
            self.lexical.is_some() && is_shipped_hash_embedder(self.fast_embedder.as_ref());

        // Adaptive budgets: identifiers lean lexical, NL leans semantic.
        let semantic_budget =
            scaled_budget(base_candidates, query_class.semantic_budget_multiplier());
        let lexical_budget = if lexical_short_circuit {
            candidate_target
        } else {
            scaled_budget(base_candidates, query_class.lexical_budget_multiplier())
        };

        let rrf_config = RrfConfig {
            k: self.effective_rrf_k(query_class),
            lexical_weight: self.rrf_lexical_weight,
            semantic_weight: self.rrf_semantic_weight,
            tiebreak: self.rrf_tiebreak,
        };

        // Colons in identifiers and URLs stay literal for the lexical lane.
        let lexical_query = self
            .lexical
            .as_ref()
            .map_or(std::borrow::Cow::Borrowed(semantic_query), |lex| {
                lex.query_text(semantic_query)
            });
        if lexical_short_circuit && let Some(lex) = self.lexical.as_ref() {
            let start_lex = Instant::now();
            let lex_res = lex.search(cx, &lexical_query, lexical_budget).await;
            match lex_res {
                Ok(results) => {
                    metrics.lexical_search_ms = start_lex.elapsed().as_secs_f64() * 1000.0;
                    metrics.lexical_candidates = results.len();
                    let results = if let Some(exclusions) = normalized_exclusions {
                        let filtered = filter_scored_results_by_negations(
                            results, exclusions, text_fn, "lexical",
                        );
                        metrics.lexical_candidates = filtered.len();
                        filtered
                    } else {
                        results
                    };
                    metrics.skip_reason =
                        Some("non_semantic_fast_embedder_lexical_short_circuit".to_owned());
                    metrics.semantic_candidates = 0;
                    metrics.vector_search_ms = 0.0;
                    metrics.rrf_fusion_ms = 0.0;
                    return Ok(Phase1Outcome::without_pools(
                        results.into_iter().take(k).collect(),
                    ));
                }
                Err(e) => {
                    metrics.lexical_search_ms = start_lex.elapsed().as_secs_f64() * 1000.0;
                    if matches!(e, SearchError::Cancelled { .. }) {
                        return Err(e);
                    }
                    self.export_error(&e);
                    tracing::warn!(
                        error = %e,
                        "lexical short-circuit failed, falling back to vector path"
                    );
                }
            }
        }

        // If the embedder is an API embedder (genuinely async), we sequentially await
        // the futures to avoid blocking the executor inside `rayon::join` and to prevent
        // `poll_immediate` from aborting the Pending futures.
        let is_async_embedder =
            self.fast_embedder.category() == frankensearch_core::traits::ModelCategory::ApiEmbedder;
        let embed_fast = || {
            embed_admitted_query(
                cx,
                self.fast_embedder.as_ref(),
                identity,
                "fast",
                semantic_query,
            )
        };

        let (embed_timed, lexical_timed) = if is_async_embedder {
            let start_embed = Instant::now();
            let embed_res = embed_fast().await;
            let embed_elapsed = start_embed.elapsed();

            let lex_res = if embed_res
                .as_ref()
                .is_err_and(terminal_query_embedding_error)
            {
                (None, Duration::ZERO)
            } else if let Some(lex) = self.lexical.as_ref() {
                let start_lex = Instant::now();
                let res = lex
                    .search_candidates(cx, &lexical_query, lexical_budget)
                    .await;
                (Some(res), start_lex.elapsed())
            } else {
                (None, Duration::ZERO)
            };

            ((embed_res, embed_elapsed), lex_res)
        } else {
            // Run embedding and lexical search in parallel via rayon::join.
            // Both trait methods are sync-in-async (complete on first poll), so
            // rayon gives true thread-level parallelism without an async runtime.
            rayon::join(
                || {
                    let start = Instant::now();
                    let result = poll_immediate(Box::pin(embed_fast()))
                        .unwrap_or_else(|| {
                            Err(SearchError::EmbeddingFailed {
                                model: self.fast_embedder.id().to_owned(),
                                source: "embedder returned Pending; only sync-in-async embedders are supported in the rayon path".into(),
                            })
                        });
                    (result, start.elapsed())
                },
                || {
                    self.lexical.as_ref().map_or_else(
                        || (None, Duration::ZERO),
                        |lex| {
                            let start = Instant::now();
                            let result = poll_immediate(lex.search_candidates(
                                cx,
                                &lexical_query,
                                lexical_budget,
                            ))
                            .unwrap_or_else(|| {
                                    Err(SearchError::SubsystemError {
                                        subsystem: "lexical",
                                        source: "lexical search returned Pending; only sync-in-async implementations are supported in the rayon path".into(),
                                    })
                                });
                            (Some(result), start.elapsed())
                        },
                    )
                },
            )
        };

        let (fast_embed_result, fast_embed_elapsed) = embed_timed;
        metrics.fast_embed_ms = fast_embed_elapsed.as_secs_f64() * 1000.0;
        // Refuse invalid responses before lexical hydration/filter callbacks or
        // a lexical-only fallback can publish an Initial response.
        cancellation_checkpoint(cx, "fast_embed_to_activation")?;
        let fast_embed_result = match fast_embed_result {
            Err(error) if terminal_query_embedding_error(&error) => return Err(error),
            result => result,
        };
        let native_embeddings = match &fast_embed_result {
            Ok((values, Some(identity))) if self.index.fast_admitted_binding().is_some() => {
                let bound = BoundQueryEmbedding::new(values.clone(), identity.clone())?;
                let embeddings = TieredQueryEmbeddings::fast_only(bound);
                // Exact and native retrieval both admit the actual response
                // before touching retained vector bytes or invoking filters.
                self.index.activate_owner_backed_search(&embeddings)?;
                cancellation_checkpoint(cx, "fast_activation_to_search")?;
                self.index.has_native_fast_hnsw().then_some(embeddings)
            }
            _ => None,
        };

        // bd-8nqz.1: the candidate batch's hydration context pins the exact
        // snapshot that scored these candidates; it travels alongside the
        // results so hydration can never read a newer generation.
        let mut lexical_context: Option<LexicalHydrationContext> = None;
        let mut lexical_results: Option<Vec<ScoredResult>> = match lexical_timed {
            (Some(Ok(batch)), elapsed) => {
                metrics.lexical_search_ms = elapsed.as_secs_f64() * 1000.0;
                let (results, context) = batch.into_parts();
                metrics.lexical_candidates = results.len();
                lexical_context = context;
                Some(results)
            }
            (Some(Err(e)), elapsed) => {
                metrics.lexical_search_ms = elapsed.as_secs_f64() * 1000.0;
                // Propagate cancellation errors — the caller must see them.
                if matches!(e, SearchError::Cancelled { .. }) {
                    return Err(e);
                }
                self.export_error(&e);
                tracing::warn!(error = %e, "lexical search failed, continuing without lexical");
                None
            }
            (None, _) => None,
        };
        if let Some(exclusions) = normalized_exclusions {
            lexical_results = lexical_results.map(|results| {
                let filtered =
                    filter_scored_results_by_negations(results, exclusions, text_fn, "lexical");
                metrics.lexical_candidates = filtered.len();
                filtered
            });
        }

        if lexical_short_circuit
            && let Some(lexical) = lexical_results.as_ref()
            && fast_embed_result.is_ok()
        {
            let lexical = self
                .lexical_results_for_direct_return(cx, lexical, lexical_context.as_ref(), metrics)
                .await?;
            metrics.skip_reason =
                Some("non_semantic_fast_embedder_lexical_short_circuit".to_owned());
            metrics.semantic_candidates = 0;
            metrics.vector_search_ms = 0.0;
            metrics.rrf_fusion_ms = 0.0;
            return Ok(Phase1Outcome::without_pools(
                lexical.into_iter().take(k).collect(),
            ));
        }

        match fast_embed_result {
            Ok((query_vec, _)) => {
                self.export_embedding_metrics(
                    self.fast_embedder.as_ref(),
                    1,
                    metrics.fast_embed_ms,
                );
                if let Some(root_request_id) = root_request_id {
                    let _ = self.emit_embedding_telemetry(
                        self.fast_embedder.as_ref(),
                        EmbeddingStage::Fast,
                        EmbeddingStatus::Completed,
                        fast_embed_elapsed,
                        root_request_id,
                        None,
                    );
                }
                if is_shipped_hash_embedder(self.fast_embedder.as_ref())
                    && metrics.skip_reason.is_none()
                {
                    // Hash vectors still run as an explicit control path, but
                    // they are never unlabeled semantic hits (bd-a6zt). A
                    // quality embedder does not make this a semantic lane.
                    metrics.skip_reason =
                        Some("non_semantic_fast_embedder_vector_control".to_owned());
                }
                if let Some(lexical) = lexical_results.as_ref()
                    && should_lexical_short_circuit(
                        query_class,
                        lexical.len(),
                        k,
                        self.config.graph_ranking_enabled,
                        self.fast_embedder.is_semantic(),
                        self.quality_embedder.is_some(),
                    )
                {
                    let lexical = self
                        .lexical_results_for_direct_return(
                            cx,
                            lexical,
                            lexical_context.as_ref(),
                            metrics,
                        )
                        .await?;
                    metrics.skip_reason = Some("lexical_short_circuit".to_owned());
                    metrics.semantic_candidates = 0;
                    metrics.phase1_vectors_searched = 0;
                    metrics.vector_search_ms = 0.0;
                    return Ok(Phase1Outcome::without_pools(
                        lexical.into_iter().take(base_candidates).collect(),
                    ));
                }

                // Vector search, through the classified fast lane so an empty
                // semantic result carries its typed `ZeroSignalReason`
                // (bd-tqhc). The default path classifies inline (and drives
                // the index's once-per-transition availability logging); an
                // explicit `search_params` override keeps its exact scan
                // configuration and classifies lazily, only when the result
                // comes back empty.
                let search_start = Instant::now();
                let (mut fast_hits, index_zero_signal) =
                    match (self.search_params, native_embeddings.as_ref()) {
                        (None, Some(_)) if query_vec.iter().all(|&value| value == 0.0) => {
                            (Vec::new(), Some(ZeroSignalReason::ZeroNormQuery))
                        }
                        (None, Some(embeddings)) => {
                            let hits = self
                                .index
                                .activate_owner_backed_search(embeddings)?
                                .search_fast(semantic_budget)?;
                            cancellation_checkpoint(cx, "native_fast_search_to_filter")?;
                            let zero_signal = if hits.is_empty() {
                                self.classify_fast_empty(&query_vec, semantic_budget)
                            } else {
                                None
                            };
                            (hits, zero_signal)
                        }
                        (None, None) => {
                            let classified = self
                                .index
                                .search_fast_classified(&query_vec, semantic_budget)?;
                            (classified.hits, classified.zero_signal)
                        }
                        (Some(params), embeddings) => {
                            let hits = if let Some(embeddings) = embeddings {
                                let activated =
                                    self.index.activate_owner_backed_search(embeddings)?;
                                let fast =
                                    activated.fast().ok_or_else(|| SearchError::InvalidConfig {
                                        field: "search_activation.fast".to_owned(),
                                        value: "missing".to_owned(),
                                        reason:
                                            "exact native override requires the admitted fast tier"
                                                .to_owned(),
                                    })?;
                                let hits =
                                    fast.search_top_k_with_params(semantic_budget, params)?;
                                cancellation_checkpoint(cx, "native_fast_search_to_filter")?;
                                hits
                            } else {
                                self.index.search_fast_with_params(
                                    &query_vec,
                                    semantic_budget,
                                    Some(params),
                                )?
                            };
                            let zero_signal = if hits.is_empty() {
                                self.classify_fast_empty(&query_vec, semantic_budget)
                            } else {
                                None
                            };
                            (hits, zero_signal)
                        }
                    };
                let prefilter_empty = fast_hits.is_empty();
                self.apply_score_calibration_to_hits(&mut fast_hits);
                let fast_hits = if let Some(exclusions) = normalized_exclusions {
                    filter_vector_hits_by_negations(
                        fast_hits,
                        exclusions,
                        text_fn,
                        vector_exclusion_source(self.fast_embedder.as_ref()),
                    )
                } else {
                    fast_hits
                };
                // Dense-score corrections (hubness demotion, k-NN diffusion) over the semantic
                // pool. Both off by default, in which case the pool is returned by move and this
                // costs nothing. Timed inside `vector_search_ms` — they are part of producing the
                // semantic candidate list, and leaving them unaccounted would understate Phase-1.
                let fast_hits = self.correct_phase1_pool(fast_hits);
                if self.search_params.is_none()
                    && index_zero_signal != Some(ZeroSignalReason::ZeroNormQuery)
                    && let Some(embeddings) = native_embeddings.as_ref()
                {
                    cancellation_checkpoint(cx, "native_fast_filter_to_coverage")?;
                    let activated = self.index.activate_owner_backed_search(embeddings)?;
                    metrics.coverage = Some(activated.coverage(&fast_hits, semantic_budget)?);
                    cancellation_checkpoint(cx, "native_fast_coverage_to_fusion")?;
                }

                metrics.vector_search_ms = search_start.elapsed().as_secs_f64() * 1000.0;
                metrics.publish_vector_candidates(
                    fast_hits.len(),
                    is_shipped_hash_embedder(self.fast_embedder.as_ref()),
                );
                metrics.phase1_vectors_searched = self.index.doc_count();
                metrics.zero_signal = if fast_hits.is_empty() {
                    if prefilter_empty {
                        // The index itself produced nothing; carry its typed
                        // classification (None only if diagnostic
                        // classification failed, which degrades to an
                        // unclassified lane rather than a fabricated reason).
                        index_zero_signal
                    } else {
                        // The index produced candidates; caller-supplied
                        // negation exclusions or pool corrections removed
                        // every one of them downstream of the scan.
                        Some(ZeroSignalReason::FilterEliminatedAll)
                    }
                } else {
                    None
                };

                let graph_candidates: Option<Vec<ScoredResult>> = if self
                    .config
                    .graph_ranking_enabled
                    && self.config.graph_ranking_weight > 0.0
                {
                    #[cfg(feature = "graph")]
                    {
                        self.document_graph
                            .as_ref()
                            .map(|graph| {
                                GraphRanker::new().rank_phase1(
                                    cx,
                                    semantic_query,
                                    graph.as_ref(),
                                    &fast_hits,
                                    semantic_budget,
                                )
                            })
                            .transpose()?
                            .flatten()
                    }
                    #[cfg(not(feature = "graph"))]
                    {
                        None
                    }
                } else {
                    None
                };
                let graph_candidates = graph_candidates.map(|results| {
                    if let Some(exclusions) = normalized_exclusions {
                        filter_scored_results_by_negations(results, exclusions, text_fn, "graph")
                    } else {
                        results
                    }
                });

                // RRF fusion if lexical results are available.
                let fuse_start = Instant::now();
                let mut results = lexical_results.as_ref().map_or_else(
                    || {
                        graph_candidates.as_ref().map_or_else(
                            || {
                                vector_hits_to_scored_results(
                                    &fast_hits,
                                    base_candidates,
                                    &self.config,
                                    self.fast_embedder.id(),
                                )
                            },
                            |graph| {
                                let fused = fuse_by_strategy_for_vector_lane(
                                    self.config.fusion_strategy,
                                    &[],
                                    &fast_hits,
                                    graph,
                                    self.config.graph_ranking_weight,
                                    base_candidates,
                                    0,
                                    &rrf_config,
                                    frankensearch_core::is_hash_generation_id(
                                        self.fast_embedder.id(),
                                    ),
                                );
                                fused_hits_to_scored_results(
                                    &fused,
                                    &[],
                                    self.config.explain,
                                    self.fast_embedder.id(),
                                    rrf_config.k,
                                )
                            },
                        )
                    },
                    |lexical| {
                        // Opt-in NQC dense down-weight (default off = byte-identical): scale the
                        // dense tier's phase-1 fusion weight per query. Only where lexical is
                        // present; the no-lexical arm above keeps the base weight.
                        let rrf_config = RrfConfig {
                            semantic_weight: self.effective_semantic_weight(lexical),
                            ..rrf_config.clone()
                        };
                        let fused = graph_candidates.as_ref().map_or_else(
                            || {
                                fuse_by_strategy_for_vector_lane(
                                    self.config.fusion_strategy,
                                    lexical,
                                    &fast_hits,
                                    &[],
                                    0.0,
                                    base_candidates,
                                    0,
                                    &rrf_config,
                                    frankensearch_core::is_hash_generation_id(
                                        self.fast_embedder.id(),
                                    ),
                                )
                            },
                            |graph| {
                                fuse_by_strategy_for_vector_lane(
                                    self.config.fusion_strategy,
                                    lexical,
                                    &fast_hits,
                                    graph,
                                    self.config.graph_ranking_weight,
                                    base_candidates,
                                    0,
                                    &rrf_config,
                                    frankensearch_core::is_hash_generation_id(
                                        self.fast_embedder.id(),
                                    ),
                                )
                            },
                        );
                        fused_hits_to_scored_results(
                            &fused,
                            lexical,
                            self.config.explain,
                            self.fast_embedder.id(),
                            rrf_config.k,
                        )
                    },
                );
                if lexical_results.is_some()
                    && let Some(lexical) = self.lexical.as_ref()
                    && let Some(context) = lexical_context.as_ref()
                {
                    lexical
                        .hydrate_candidates(cx, Some(context), &mut results)
                        .await?;
                }
                metrics.rrf_fusion_ms = fuse_start.elapsed().as_secs_f64() * 1000.0;

                Ok(Phase1Outcome {
                    results,
                    pools: Some(RefinementPools {
                        fast_hits,
                        lexical: lexical_results,
                    }),
                })
            }
            Err(embed_err) => {
                if let Some(root_request_id) = root_request_id {
                    let status = if matches!(&embed_err, SearchError::Cancelled { .. }) {
                        EmbeddingStatus::Cancelled
                    } else {
                        EmbeddingStatus::Failed
                    };
                    let _ = self.emit_embedding_telemetry(
                        self.fast_embedder.as_ref(),
                        EmbeddingStage::Fast,
                        status,
                        fast_embed_elapsed,
                        root_request_id,
                        None,
                    );
                }
                if matches!(embed_err, SearchError::Cancelled { .. }) {
                    return Err(embed_err);
                }
                // Graceful degradation: use lexical-only results if available.
                if let Some(ref lexical) = lexical_results {
                    self.export_error(&embed_err);
                    tracing::warn!(
                        error = %embed_err,
                        "fast embedding failed, falling back to lexical-only results"
                    );
                    let lexical = self
                        .lexical_results_for_direct_return(
                            cx,
                            lexical,
                            lexical_context.as_ref(),
                            metrics,
                        )
                        .await?;
                    metrics.skip_reason = Some("fast_embed_failed".to_owned());
                    Ok(Phase1Outcome::without_pools(
                        lexical.into_iter().take(k).collect(),
                    ))
                } else {
                    Err(embed_err)
                }
            }
        }
    }

    /// Run Phase 2: quality embedding + blend + optional rerank.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    async fn run_phase2(
        &self,
        cx: &Cx,
        query: &str,
        query_class: QueryClass,
        k: usize,
        initial_results: &[ScoredResult],
        pools: Option<&RefinementPools>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        exclusions: Option<&NormalizedExclusions>,
        admission: SemanticAdmission,
        identity: Option<&EmbeddingIdentityBundleV1>,
        metrics: &mut TwoTierMetrics,
        root_request_id: Option<&str>,
        parent_event_id: Option<String>,
    ) -> SearchResult<Vec<ScoredResult>> {
        let quality_embedder =
            self.quality_embedder
                .as_ref()
                .ok_or_else(|| SearchError::EmbedderUnavailable {
                    model: "quality".into(),
                    reason: "no quality embedder configured".into(),
                })?;

        // Quality embedding.
        let embed_start = Instant::now();
        let quality_output =
            embed_admitted_query(cx, quality_embedder.as_ref(), identity, "quality", query).await;
        let (mut quality_vec, quality_identity) = match quality_output {
            Ok(output) => {
                let quality_embed_elapsed = embed_start.elapsed();
                metrics.quality_embed_ms = quality_embed_elapsed.as_secs_f64() * 1000.0;
                self.export_embedding_metrics(
                    quality_embedder.as_ref(),
                    1,
                    metrics.quality_embed_ms,
                );
                if let Some(root_request_id) = root_request_id {
                    let _ = self.emit_embedding_telemetry(
                        quality_embedder.as_ref(),
                        EmbeddingStage::Quality,
                        EmbeddingStatus::Completed,
                        quality_embed_elapsed,
                        root_request_id,
                        parent_event_id.clone(),
                    );
                }
                output
            }
            Err(err) => {
                let quality_embed_elapsed = embed_start.elapsed();
                metrics.quality_embed_ms = quality_embed_elapsed.as_secs_f64() * 1000.0;
                if let Some(root_request_id) = root_request_id {
                    let status = if matches!(&err, SearchError::Cancelled { .. }) {
                        EmbeddingStatus::Cancelled
                    } else {
                        EmbeddingStatus::Failed
                    };
                    let _ = self.emit_embedding_telemetry(
                        quality_embedder.as_ref(),
                        EmbeddingStage::Quality,
                        status,
                        quality_embed_elapsed,
                        root_request_id,
                        parent_event_id,
                    );
                }
                return Err(err);
            }
        };

        cancellation_checkpoint(cx, "quality_embed_to_prf")?;

        if let Some(identity) = &quality_identity
            && self.index.quality_admitted_binding().is_some()
        {
            // Admit the identity that actually accompanied this response before
            // reading feedback vectors, scoring or traversing the graph. The
            // provider's advertised identity cannot relabel a foreign response.
            let bound = BoundQueryEmbedding::new(quality_vec.clone(), identity.clone())?;
            let embeddings = TieredQueryEmbeddings::quality_only(bound);
            self.index.activate_owner_backed_search(&embeddings)?;
            cancellation_checkpoint(cx, "quality_activation_to_prf")?;
        }

        if self.prf_config.should_expand(&query_class) {
            let mut feedback_embeddings = Vec::new();
            for result in initial_results.iter().take(self.prf_config.top_k_feedback) {
                // PRF interpolates these document vectors with `quality_vec`, so
                // every feedback value must come from that exact embedding space.
                // `semantic_vector_for_doc_id` is intentionally unsuitable here:
                // it silently falls back to the fast tier for a document missing
                // quality coverage, which would make a same-dimension foreign
                // vector look valid and poison the expanded query.
                let feedback = self.index.quality_vector_for_doc_id(&result.doc_id);
                cancellation_checkpoint(cx, "prf_vector_lookup")?;
                match feedback {
                    Ok(Some(embedding)) => {
                        let weight = if self.prf_config.score_weighted {
                            f64::from(result.score.max(0.0))
                        } else {
                            1.0
                        };
                        feedback_embeddings.push((embedding, weight));
                    }
                    Ok(None) => {}
                    Err(err) => {
                        self.export_error(&err);
                        tracing::warn!(
                            error = %err,
                            doc_id = %result.doc_id,
                            "prf feedback vector lookup failed; continuing with remaining candidates"
                        );
                    }
                }
            }

            if feedback_embeddings.len() >= self.prf_config.min_feedback_docs {
                cancellation_checkpoint(cx, "prf_lookup_to_expansion")?;
                let refs = feedback_embeddings
                    .iter()
                    .map(|(embedding, weight)| (embedding.as_slice(), *weight))
                    .collect::<Vec<_>>();
                if let Some(expanded) =
                    prf_expand(&quality_vec, &refs, self.prf_config.clamped_alpha())
                {
                    quality_vec = expanded;
                }
            }
        }

        cancellation_checkpoint(cx, "prf_to_fast_pool")?;

        // Get quality scores for the full retained semantic pool (not just
        // the displayed page): quality can promote candidates that never
        // reached the Initial top-k (bd-tm83). When the caller supplied no
        // pools, fall back to reconstructing the pool from `initial_results`.
        let search_start = Instant::now();
        let fast_hits: Vec<VectorHit> = pools.map_or_else(
            || {
                initial_results
                    .iter()
                    .map(|r| VectorHit {
                        index: r.index.unwrap_or(u32::MAX),
                        // Keep missing semantic-fast source at 0.0 so blending semantics
                        // remain consistent with blend_two_tier contract.
                        score: r.fast_score.unwrap_or(0.0_f32),
                        doc_id: r.doc_id.clone(),
                    })
                    .collect()
            },
            |pools| pools.fast_hits.clone(),
        );
        // NOTE: Do NOT calibrate fast_hits here — these scores were already calibrated
        // in Phase 1 (line ~894) and stored in ScoredResult.fast_score. Applying
        // calibration again would double-transform the scores, destroying discriminative
        // power for non-idempotent calibrators (TemperatureScaling, PlattScaling,
        // IsotonicRegression).

        // bd-ctzo C2: RETRIEVE from the quality tier, do not rescore the fast
        // pool.
        //
        // `quality_scores_for_hits(&quality_vec, &fast_hits)` scores exactly
        // the documents the FAST tier already selected. Its own comment above
        // claims quality "can promote candidates that never reached the
        // Initial top-k", and within the retained pool that is true — but a
        // document the fast tier ranked below `k * candidate_multiplier`, or
        // does not contain at all, is unreachable no matter what the quality
        // tier thinks of it. That is the anti-pattern C2 forbids, and it was
        // the production path.
        //
        // When the quality tier is an admitted owner, the query is instead
        // embedded, bound, and used to search the quality owner directly; the
        // two pools are then unioned by canonical doc_id in the blend. The
        // fast pool is NOT re-retrieved through the activation: phase 1's pool
        // already went through ANN candidate retrieval, score calibration,
        // negation filtering and the dense-score corrections, and
        // `ActivatedTierSearch::search_fast` reproduces none of that. Union
        // here therefore means "phase 1's real fast pool ∪ the quality owner's
        // own top-k", which is strictly more than either arm alone.
        let quality_budget = fast_hits.len().max(k);
        cancellation_checkpoint(cx, "fast_pool_to_quality_search")?;
        let mut quality_coverage = None;
        let quality_pool = match admission {
            SemanticAdmission::OwnerBacked { quality: true } => {
                // The PRF-expanded vector stays inside this space by
                // construction: it is a convex mix of this query and vectors
                // read out of this very index. Preserve the identity that
                // accompanied the admitted response through that expansion.
                let quality_identity = quality_identity.ok_or_else(|| SearchError::InvalidConfig {
                    field: "query_embedding.quality.identity".to_owned(),
                    value: "missing_bound_response".to_owned(),
                    reason: "owner-backed quality search requires the actual bound response identity".to_owned(),
                })?;
                cancellation_checkpoint(cx, "quality_identity_to_activation")?;
                let bound = BoundQueryEmbedding::new(quality_vec.clone(), quality_identity)?;
                let embeddings = TieredQueryEmbeddings::quality_only(bound);
                let activated = self.index.activate_owner_backed_search(&embeddings)?;
                cancellation_checkpoint(cx, "quality_activation_to_search")?;
                let hits = activated.search_quality(quality_budget)?;
                cancellation_checkpoint(cx, "quality_search_to_coverage")?;
                // bd-ctzo C4: the coverage receipt is built from the retained
                // owner's witness and the candidates this tier ACTUALLY
                // returned, then carried on the metrics so a caller can read
                // it. It is assembled after retrieval on purpose — a
                // contribution count taken before the results exist would be
                // a prediction, not a measurement.
                let coverage = activated.coverage(&hits, quality_budget)?;
                tracing::debug!(
                    coverage = %coverage.redacted_summary(),
                    "quality tier served under owner-backed activation"
                );
                quality_coverage = Some(coverage);
                QualityPool::Retrieved(hits)
            }
            SemanticAdmission::LegacyUnidentified
                if self
                    .index
                    .quality_embedder_revision()
                    .is_some_and(|revision| !revision.is_empty()) =>
            {
                // Reopened v1 indexes now retain the complete producer fingerprint.
                // Their quality tier can retrieve its own candidates after that join;
                // rescoring only the fast pool would limit quality-tier recall.
                // This is still a v1 read, with no fabricated v2 coverage witness.
                let quality_identity = quality_identity.ok_or_else(|| SearchError::InvalidConfig {
                    field: "query_embedding.quality.identity".to_owned(),
                    value: "missing_bound_response".to_owned(),
                    reason: "producer-stamped quality search requires the actual bound response identity".to_owned(),
                })?;
                let bound = BoundQueryEmbedding::new(quality_vec.clone(), quality_identity)?;
                cancellation_checkpoint(cx, "quality_identity_to_search")?;
                let hits = self
                    .index
                    .search_quality_with_producer(&bound, quality_budget)?;
                cancellation_checkpoint(cx, "quality_search_to_calibration")?;
                QualityPool::Retrieved(hits)
            }
            SemanticAdmission::OwnerBacked { quality: false }
            | SemanticAdmission::LegacyUnidentified => QualityPool::RescoredFastPool(
                self.index
                    .quality_scores_for_hits(&quality_vec, &fast_hits)?,
            ),
        };
        metrics.quality_search_ms = search_start.elapsed().as_secs_f64() * 1000.0;

        cancellation_checkpoint(cx, "quality_search_to_calibration")?;
        let quality_pool = match (quality_pool, exclusions) {
            (QualityPool::Retrieved(hits), Some(exclusions)) => QualityPool::Retrieved(
                filter_vector_hits_by_negations(hits, exclusions, text_fn, "semantic_quality"),
            ),
            (pool, _) => pool,
        };
        let completed_coverage = quality_coverage.map(|mut coverage| {
            if let QualityPool::Retrieved(hits) = &quality_pool
                && let frankensearch_core::TierQueryCoverageV1::Witnessed {
                    contributed_candidates,
                    ..
                } = &mut coverage.quality
            {
                // Excluded candidates did not participate in refinement. The
                // retained pool is the measured contribution; it must not be
                // reconstructed by another search with different filters.
                *contributed_candidates = hits.len() as u64;
            }
            if let Some(previous) = metrics.coverage.as_ref()
                && matches!(
                    previous.fast,
                    frankensearch_core::TierQueryCoverageV1::Witnessed { .. }
                )
            {
                coverage.topology = if previous.is_hash_control() || coverage.is_hash_control() {
                    frankensearch_core::RetrievalTopology::HashControl
                } else {
                    frankensearch_core::RetrievalTopology::FullProgressive
                };
                coverage.fast = previous.fast.clone();
            }
            coverage
        });

        // Calibration is a pure per-element score transform, so it is applied
        // in whichever shape the pool arrived in — never by converting between
        // them.
        //
        // On the rescored path that shape is the aligned `Vec<Option<f32>>`
        // the index returned: the quality tier there IS a re-scored subset of
        // `fast_hits` (same doc_ids, same indices), so materializing a
        // `Vec<VectorHit>` would clone one `String` doc_id per hit for values
        // that are only ever read as `&str`. Docs left `None` blend at their
        // fast-only score without penalty.
        //
        // On the retrieved path the pool is the quality owner's own hits and
        // carries doc_ids the fast pool may not have at all, so there is
        // nothing to align to and each hit's score is transformed in place.
        //
        // The default no-calibrator path skips both entirely.
        let mut quality_pool = quality_pool;
        if self.score_calibrator.is_some() {
            match &mut quality_pool {
                QualityPool::RescoredFastPool(scores) => {
                    for score in scores.iter_mut().flatten() {
                        *score = self.calibrate_score(*score);
                    }
                }
                QualityPool::Retrieved(hits) => {
                    for hit in hits.iter_mut() {
                        hit.score = self.calibrate_score(hit.score);
                    }
                }
            }
        }
        let quality_pool = quality_pool;

        cancellation_checkpoint(cx, "quality_calibration_to_evidence")?;

        // Quality evidence, keyed by canonical document identity so the two
        // pools can disagree about which documents exist at all.
        let quality_scores_by_doc: AHashMap<&str, f32> = match &quality_pool {
            QualityPool::RescoredFastPool(scores) => fast_hits
                .iter()
                .zip(scores.iter())
                .filter_map(|(hit, score)| score.map(|s| (hit.doc_id.as_str(), s)))
                .collect(),
            QualityPool::Retrieved(hits) => hits
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score))
                .collect(),
        };
        // "Incomplete" means a fast candidate the quality tier had nothing to
        // say about — under retrieval that is a fast document the quality
        // top-k did not reach, which is a real and different fact from "the
        // quality tier has no vector for it".
        metrics.incomplete_embeddings = fast_hits
            .iter()
            .filter(|hit| !quality_scores_by_doc.contains_key(hit.doc_id.as_str()))
            .count();
        metrics.phase2_vectors_searched = quality_scores_by_doc.len();

        cancellation_checkpoint(cx, "quality_evidence_to_blend")?;

        // Blend fast + quality scores.
        let blend_start = Instant::now();
        let blend_factor = self.effective_blend_factor(query_class);
        let blended = match &quality_pool {
            QualityPool::RescoredFastPool(scores) => {
                blend_two_tier_aligned_vector_index(&fast_hits, scores, blend_factor)
            }
            // The union blend: a document in both tiers blends, a fast-only
            // document keeps its fast score, and a QUALITY-ONLY document keeps
            // its quality score instead of being dropped. That last case is
            // the whole behaviour change — it is the document the rescoring
            // path could never return.
            QualityPool::Retrieved(hits) => {
                let mut blended = blend_two_tier(&fast_hits, hits, blend_factor);
                let fast_indices: AHashMap<&str, u32> = fast_hits
                    .iter()
                    .map(|hit| (hit.doc_id.as_str(), hit.index))
                    .collect();
                // `VectorHit::index` is a FAST-tier row ordinal everywhere
                // downstream (the hubness table is indexed by it). A
                // quality-only document has no fast row, so it carries the
                // established "no index" sentinel rather than a quality row
                // ordinal that would silently read as a fast one.
                for hit in &mut blended {
                    hit.index = fast_indices
                        .get(hit.doc_id.as_str())
                        .copied()
                        .unwrap_or(u32::MAX);
                }
                blended
            }
        };
        metrics.blend_ms = blend_start.elapsed().as_secs_f64() * 1000.0;

        cancellation_checkpoint(cx, "quality_blend_to_rank")?;

        // Compute rank changes (initial vs refined).
        // Precompute rank maps once, then pass to both functions.
        let initial_rank = build_borrowed_rank_map(&fast_hits);
        let refined_rank = build_borrowed_rank_map(&blended);
        let rank_changes = compute_rank_changes_with_maps(&initial_rank, &refined_rank);
        let tau = kendall_tau_with_refined_rank(&fast_hits, &refined_rank);
        metrics.kendall_tau = tau;
        metrics.rank_changes = rank_changes;
        if let Some(adaptive_fusion) = self.adaptive_fusion.as_ref() {
            let success = tau.is_some_and(|value| value < 0.98);
            let signal = if success {
                SignalSource::Click
            } else {
                SignalSource::Skip
            };
            let _ = adaptive_fusion.update_blend(query_class, success, signal);
        }
        self.maybe_update_adaptive_conformal(tau);

        cancellation_checkpoint(cx, "quality_rank_to_result_construction")?;

        let initial_by_doc: AHashMap<&str, &ScoredResult> = initial_results
            .iter()
            .map(|result| (result.doc_id.as_str(), result))
            .collect();
        let fast_scores_by_doc: AHashMap<&str, f32> = fast_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect();

        // Convert blended to scored results.
        let mut fast_min = f32::INFINITY;
        let mut fast_max = f32::NEG_INFINITY;
        let mut qual_min = f32::INFINITY;
        let mut qual_max = f32::NEG_INFINITY;
        if self.config.explain {
            for &s in fast_scores_by_doc.values() {
                if s.is_finite() {
                    fast_min = fast_min.min(s);
                    fast_max = fast_max.max(s);
                }
            }
            for &s in quality_scores_by_doc.values() {
                if s.is_finite() {
                    qual_min = qual_min.min(s);
                    qual_max = qual_max.max(s);
                }
            }
        }

        // Shared refined-phase explanation builder. `rank_map` supplies the
        // phase-1 ordering the movement delta is measured against: the fast
        // pool order for the blend-only path, the fused order for the
        // lexical re-fusion path.
        let build_refined_explanation = |final_score: f32,
                                         rank: usize,
                                         doc_id: &str,
                                         fast_score: Option<f32>,
                                         quality_score: Option<f32>,
                                         rank_map: &std::collections::HashMap<&str, usize>|
         -> Option<Box<HitExplanation>> {
            if !self.config.explain {
                return None;
            }
            let mut components = Vec::new();

            let fast_norm = |s: f32| -> f64 {
                if !s.is_finite() {
                    return 0.0;
                }
                let range = fast_max - fast_min;
                if range > 0.01 {
                    f64::from(((s - fast_min) / range).clamp(0.0, 1.0))
                } else {
                    f64::from(s.clamp(0.0, 1.0))
                }
            };

            let qual_norm = |s: f32| -> f64 {
                if !s.is_finite() {
                    return 0.0;
                }
                let range = qual_max - qual_min;
                if range > 0.01 {
                    f64::from(((s - qual_min) / range).clamp(0.0, 1.0))
                } else {
                    f64::from(s.clamp(0.0, 1.0))
                }
            };

            // Fast component
            if let Some(s) = fast_score {
                components.push(ScoreComponent {
                    source: ExplainedSource::vector_fast(self.fast_embedder.id(), f64::from(s)),
                    raw_score: f64::from(s),
                    normalized_score: fast_norm(s),
                    rrf_contribution: 0.0,
                    weight: 1.0 - f64::from(blend_factor),
                });
            }

            // Quality component
            if let Some(s) = quality_score {
                components.push(ScoreComponent {
                    source: ExplainedSource::SemanticQuality {
                        embedder: self
                            .quality_embedder
                            .as_ref()
                            .map_or_else(|| "quality-embedder".to_owned(), |e| e.id().to_owned()),
                        cosine_sim: f64::from(s),
                    },
                    raw_score: f64::from(s),
                    normalized_score: qual_norm(s),
                    rrf_contribution: 0.0,
                    weight: f64::from(blend_factor),
                });
            }

            // Rank movement
            let rank_movement = rank_map.get(doc_id).map(|&i_rank| {
                let refined_rank = i64::try_from(rank).unwrap_or(i64::MAX);
                let initial_rank = i64::try_from(i_rank).unwrap_or(i64::MAX);
                let delta_i64 = refined_rank - initial_rank;
                let delta = i32::try_from(delta_i64).unwrap_or_else(|_| {
                    if delta_i64.is_negative() {
                        i32::MIN
                    } else {
                        i32::MAX
                    }
                });
                let reason = match delta.cmp(&0) {
                    std::cmp::Ordering::Less => "promoted",
                    std::cmp::Ordering::Greater => "demoted",
                    std::cmp::Ordering::Equal => "stable",
                };
                RankMovement {
                    initial_rank: i_rank,
                    refined_rank: rank,
                    delta,
                    reason: reason.to_owned(),
                }
            });

            Some(Box::new(HitExplanation {
                final_score: f64::from(final_score),
                components,
                phase: ExplanationPhase::Refined,
                rank_movement,
            }))
        };

        // Lexical re-fusion (bd-tm83): when Phase 1 retained a lexical pool,
        // rerun the configured fusion strategy against the blended semantic
        // pool so the refined ordering reflects both sources — the same
        // shape as the sync searcher — without a second lexical query.
        if let Some(lexical_pool) = pools.and_then(|pools| pools.lexical.as_ref()) {
            let rrf_config = RrfConfig {
                k: self.effective_rrf_k(query_class),
                lexical_weight: self.rrf_lexical_weight,
                semantic_weight: self.effective_semantic_weight(lexical_pool),
                tiebreak: self.rrf_tiebreak,
            };
            cancellation_checkpoint(cx, "quality_rank_to_refusion")?;
            let fused = fuse_by_strategy(
                self.config.fusion_strategy,
                lexical_pool,
                &blended,
                &[],
                0.0,
                k,
                0,
                &rrf_config,
            );
            cancellation_checkpoint(cx, "quality_refusion_to_result_construction")?;
            let mut results = fused_hits_to_scored_results(
                &fused,
                lexical_pool,
                self.config.explain,
                self.fast_embedder.id(),
                rrf_config.k,
            );
            let initial_fused_rank: std::collections::HashMap<&str, usize> = initial_results
                .iter()
                .enumerate()
                .map(|(rank, result)| (result.doc_id.as_str(), rank))
                .collect();
            for (rank, result) in results.iter_mut().enumerate() {
                let initial = initial_by_doc.get(result.doc_id.as_str()).copied();
                let fast_score = fast_scores_by_doc.get(result.doc_id.as_str()).copied();
                let quality_score = quality_scores_by_doc.get(result.doc_id.as_str()).copied();
                // Restore the raw fast-tier score (the fused conversion
                // surfaces the blended tier score) and attach quality
                // evidence; provenance and lexical metadata carry over from
                // the conversion, with hydrated phase-1 metadata as a
                // fallback for winners the raw pool did not cover.
                result.fast_score = fast_score;
                result.quality_score = quality_score;
                result.source = promoted_quality_source(result.source, quality_score.is_some());
                if result.metadata.is_none() {
                    result.metadata = initial.and_then(|initial| initial.metadata.clone());
                }
                result.explanation = build_refined_explanation(
                    result.score,
                    rank,
                    result.doc_id.as_str(),
                    fast_score,
                    quality_score,
                    &initial_fused_rank,
                );
            }
            cancellation_checkpoint(cx, "quality_result_construction_to_publish")?;
            if let Some(coverage) = completed_coverage {
                metrics.coverage = Some(coverage);
            }
            return Ok(results);
        }

        cancellation_checkpoint(cx, "quality_result_preparation_to_construction")?;

        #[allow(unused_mut)] // mut needed when `rerank` feature is enabled
        let mut results: Vec<ScoredResult> = blended
            .iter()
            .enumerate()
            .take(k)
            .map(|(rank, hit)| {
                let initial = initial_by_doc.get(hit.doc_id.as_str()).copied();
                let fast_score = fast_scores_by_doc.get(hit.doc_id.as_str()).copied();
                let quality_score = quality_scores_by_doc.get(hit.doc_id.as_str()).copied();
                let original_source =
                    initial.map_or(ScoreSource::SemanticFast, |result| result.source);
                let source = promoted_quality_source(original_source, quality_score.is_some());

                let explanation = build_refined_explanation(
                    hit.score,
                    rank,
                    hit.doc_id.as_str(),
                    fast_score,
                    quality_score,
                    &initial_rank,
                );

                ScoredResult {
                    doc_id: hit.doc_id.clone(),
                    score: hit.score,
                    source,
                    index: if hit.index == u32::MAX {
                        None
                    } else {
                        Some(hit.index)
                    },
                    fast_score,
                    quality_score,
                    lexical_score: initial.and_then(|result| result.lexical_score),
                    rerank_score: None,
                    explanation,
                    metadata: initial.and_then(|result| result.metadata.clone()),
                }
            })
            .collect();

        cancellation_checkpoint(cx, "quality_result_construction_to_publish")?;
        if let Some(coverage) = completed_coverage {
            metrics.coverage = Some(coverage);
        }
        Ok(results)
    }
}

// Implement Debug manually since trait objects don't derive Debug.
impl std::fmt::Debug for TwoTierSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoTierSearcher")
            .field("fast_embedder", &self.fast_embedder.id())
            .field(
                "quality_embedder",
                &self.quality_embedder.as_ref().map(|e| e.id()),
            )
            .field("has_lexical", &self.lexical.is_some())
            .field("has_reranker", &self.reranker.is_some())
            .field("has_host_adapter", &self.host_adapter.is_some())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
