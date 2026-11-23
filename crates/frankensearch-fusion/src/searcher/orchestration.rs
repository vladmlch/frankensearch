use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::time::timeout;
use tracing::instrument;

use frankensearch_core::ParsedQuery;
use frankensearch_core::config::TwoTierMetrics;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::host_adapter::AdapterLifecycleEvent;
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::traits::LexicalHydrationContext;
use frankensearch_core::types::{PhaseMetrics, ScoredResult, SearchPhase};
use frankensearch_core::{LifecycleSeverity, LifecycleState, SearchEventPhase};

use crate::circuit_breaker::QualityOutcome;

use super::TwoTierSearcher;
use super::admission::{SemanticAdmission, cancellation_checkpoint};
use super::results::{
    NormalizedExclusions, is_shipped_hash_embedder, normalize_for_negation_match,
};
use super::telemetry::{next_telemetry_identifier, telemetry_timestamp_now};

impl TwoTierSearcher {
    /// Execute progressive search, calling `on_phase` as results become available.
    ///
    /// Fires [`SearchPhase::Initial`] first, then optionally
    /// [`SearchPhase::Refined`] or [`SearchPhase::RefinementFailed`].
    ///
    /// Returns collected metrics from all phases.
    ///
    /// # Parameters
    ///
    /// * `cx` — Capability context for cancellation.
    /// * `query` — The search query string.
    /// * `k` — Maximum number of results per phase.
    /// * `text_fn` — Retrieves document text by `doc_id` for reranking and
    ///   exclusion-query filtering.
    ///   Pass `|_| None` only when reranking is not needed and the query does
    ///   not contain exclusions (`-term`, `NOT "phrase"`).
    /// * `on_phase` — Callback invoked once per search phase.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Cancelled` if the operation is cancelled via `cx`.
    /// Returns `SearchError::EmbeddingFailed` if fast embedding fails and no
    /// lexical backend is available as fallback.
    #[instrument(skip_all, fields(query_len = query.len(), k))]
    #[allow(clippy::too_many_lines)]
    pub async fn search(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
        mut on_phase: impl FnMut(SearchPhase) + Send,
    ) -> SearchResult<TwoTierMetrics> {
        let mut metrics = TwoTierMetrics::default();

        if query.is_empty() || k == 0 {
            return Ok(metrics);
        }

        // Canonicalize query.
        let canon_query = self.canonicalizer.canonicalize_query(query);
        if canon_query.trim().is_empty() {
            return Ok(metrics);
        }
        let parsed_query = ParsedQuery::parse(&canon_query);
        let normalized_exclusions = if parsed_query.has_negations() {
            Some(NormalizedExclusions {
                terms: parsed_query
                    .negative_terms
                    .iter()
                    .map(|t| normalize_for_negation_match(t))
                    .collect(),
                phrases: parsed_query
                    .negative_phrases
                    .iter()
                    .map(|p| normalize_for_negation_match(p))
                    .collect(),
            })
        } else {
            None
        };
        let semantic_query = if parsed_query.is_positive_empty() {
            canon_query.as_str()
        } else {
            parsed_query.positive.as_str()
        };

        tracing::debug!(
            included_terms = parsed_query.positive.split_whitespace().count(),
            excluded_terms = parsed_query.negation_count(),
            has_negations = parsed_query.has_negations(),
            "query_parsed"
        );

        let query_class = QueryClass::classify(semantic_query);
        metrics.query_class = Some(query_class);
        metrics.fast_embedder_id = Some(self.fast_embedder.id().to_owned());

        // bd-ctzo C1: the identity join happens HERE -- before the fast
        // embedder is invoked, before any embedding cache is consulted, and
        // before a single vector or ANN node is read. A refusal therefore
        // costs a fingerprint comparison, and the work it would have
        // authorized never starts.
        let admission = self.admit_semantic_query();
        cancellation_checkpoint(cx, "query_identity_admission")?;
        let (admission, identities) = admission?;
        match admission {
            SemanticAdmission::OwnerBacked { quality } => tracing::debug!(
                quality_owner_backed = quality,
                "query admitted against retained per-tier index identity"
            ),
            SemanticAdmission::LegacyUnidentified => tracing::debug!(
                "index retains no admitted FSVI v2 owner; searching a legacy-unidentified \
                 artifact with no space identity to join against"
            ),
        }
        let telemetry_root_request_id = self
            .host_adapter
            .as_ref()
            .map(|_| next_telemetry_identifier("root"));
        let mut telemetry_initial_event_id: Option<String> = None;
        let mut telemetry_last_event_id: Option<String> = None;
        let search_started_at = Instant::now();

        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
            self.emit_host_lifecycle_hook(&AdapterLifecycleEvent::SessionStart {
                ts: telemetry_timestamp_now(),
            });
            telemetry_last_event_id = self.emit_lifecycle_telemetry(
                root_request_id,
                None,
                LifecycleState::Started,
                LifecycleSeverity::Info,
                None,
                Some(0),
            );
        }

        // Phase 1: Initial (fast tier).
        let phase1_start = Instant::now();
        let initial = self
            .run_phase1(
                cx,
                semantic_query,
                &parsed_query,
                normalized_exclusions.as_ref(),
                k,
                query_class,
                identities.fast.as_ref(),
                &text_fn,
                &mut metrics,
                telemetry_root_request_id.as_deref(),
            )
            .await
            .and_then(|outcome| {
                cancellation_checkpoint(cx, "fast_result_to_publish")?;
                Ok(outcome)
            });
        metrics.phase1_total_ms = phase1_start.elapsed().as_secs_f64() * 1000.0;

        let (initial_hits, refinement_pools) = match initial {
            Ok(outcome) => (outcome.results, outcome.pools),
            Err(err) => {
                self.export_error(&err);
                if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                    // A cancelled Initial stops like every later cancellation
                    // boundary, as a typed warning rather than a failure.
                    let (severity, reason) = match &err {
                        SearchError::Cancelled { phase, reason } => (
                            LifecycleSeverity::Warn,
                            format!("cancelled:{phase}:{reason}"),
                        ),
                        error => (LifecycleSeverity::Error, error.to_string()),
                    };
                    self.emit_session_stop_telemetry(
                        root_request_id,
                        telemetry_last_event_id.clone(),
                        LifecycleState::Degraded,
                        severity,
                        Some(reason),
                        search_started_at.elapsed(),
                    );
                }
                return Err(err);
            }
        };

        let initial_latency = phase1_start.elapsed();
        let display_hits: Vec<_> = initial_hits.iter().take(k).cloned().collect();

        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
            telemetry_initial_event_id = self.emit_search_telemetry(
                semantic_query,
                query_class,
                SearchEventPhase::Initial,
                display_hits.len(),
                metrics.lexical_candidates,
                metrics.semantic_candidates,
                metrics.hash_control_candidates,
                initial_latency,
                root_request_id,
                None,
            );
            if telemetry_initial_event_id.is_some() {
                telemetry_last_event_id = telemetry_initial_event_id.clone();
            }
            self.emit_phase_health_telemetry(
                root_request_id,
                telemetry_initial_event_id
                    .clone()
                    .or_else(|| telemetry_last_event_id.clone()),
                search_started_at.elapsed(),
            );
        }

        on_phase(SearchPhase::Initial {
            results: display_hits.clone(),
            latency: initial_latency,
            metrics: PhaseMetrics {
                embedder_id: self.fast_embedder.id().to_owned(),
                vectors_searched: metrics.phase1_vectors_searched,
                lexical_candidates: metrics.lexical_candidates,
                fused_count: display_hits.len(),
                skip_reason: metrics.skip_reason.clone(),
                hash_control_candidates: metrics.hash_control_candidates,
            },
        });

        // The Initial callback is a public phase boundary: a consumer may
        // cancel the invocation after accepting the fast result. Observe that
        // authority before starting quality embedding or its synchronous
        // vector work.
        if let Err(error) = cx.checkpoint() {
            let phase = "initial_to_quality".to_owned();
            let reason = cx
                .cancel_reason()
                .map_or_else(|| error.to_string(), |reason| reason.to_string());
            let cancellation = SearchError::Cancelled {
                phase: phase.clone(),
                reason: reason.clone(),
            };
            if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                self.emit_session_stop_telemetry(
                    root_request_id,
                    telemetry_last_event_id.clone(),
                    LifecycleState::Degraded,
                    LifecycleSeverity::Warn,
                    Some(format!("cancelled:{phase}:{reason}")),
                    search_started_at.elapsed(),
                );
            }
            return Err(cancellation);
        }
        self.export_search_metrics(query_class, &metrics, display_hits.len(), false);

        // Phase 2: Quality refinement (optional).
        // Runs even if Phase 1 was lexical-only (fast_score is None), effectively
        // performing a "lexical -> quality rerank" flow.  Skipped when fast
        // embedding failed entirely (phase1_vectors_searched == 0) because the
        // embedding pipeline is degraded and quality refinement would be unreliable.
        let quality_circuit_open = if self.should_run_quality()
            && let Some(circuit_breaker) = self.circuit_breaker.as_ref()
        {
            let (skip, _evidence) = circuit_breaker.should_skip_quality();
            skip
        } else {
            false
        };
        let quality_phase_gate_skip =
            self.should_run_quality() && self.phase_gate_should_skip_quality();

        if self.should_run_quality()
            && !quality_circuit_open
            && !quality_phase_gate_skip
            && !initial_hits.is_empty()
            && metrics.phase1_vectors_searched > 0
        {
            let phase2_start = Instant::now();
            metrics.quality_embedder_id = self.quality_embedder.as_ref().map(|e| e.id().to_owned());

            let phase2_future = Box::pin(self.run_phase2(
                cx,
                semantic_query,
                query_class,
                k,
                &initial_hits,
                refinement_pools.as_ref(),
                &text_fn,
                normalized_exclusions.as_ref(),
                admission,
                identities.quality.as_ref(),
                &mut metrics,
                telemetry_root_request_id.as_deref(),
                telemetry_initial_event_id.clone(),
            ));
            let timeout_budget = Duration::from_millis(self.config.quality_timeout_ms);
            let timeout_start = cx.now();
            let phase2_result = timeout(timeout_start, timeout_budget, phase2_future).await;

            match phase2_result {
                Err(_elapsed) => {
                    let phase2_latency = phase2_start.elapsed();
                    metrics.phase2_total_ms = phase2_latency.as_secs_f64() * 1000.0;
                    let phase2_latency_ms =
                        u64::try_from(phase2_latency.as_millis()).unwrap_or(u64::MAX);
                    let timeout_error = SearchError::SearchTimeout {
                        elapsed_ms: phase2_latency_ms,
                        budget_ms: self.config.quality_timeout_ms,
                    };
                    metrics.skip_reason = Some(timeout_error.to_string());
                    self.export_error(&timeout_error);
                    if let Some(circuit_breaker) = self.circuit_breaker.as_ref() {
                        let outcome = QualityOutcome::Slow {
                            latency_ms: phase2_latency_ms,
                        };
                        let _ = circuit_breaker.record_outcome(&outcome);
                    }
                    if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                        let refinement_failed_event_id = self.emit_search_telemetry(
                            semantic_query,
                            query_class,
                            SearchEventPhase::RefinementFailed,
                            display_hits.len(),
                            metrics.lexical_candidates,
                            metrics.semantic_candidates,
                            metrics.hash_control_candidates,
                            phase2_latency,
                            root_request_id,
                            telemetry_initial_event_id.clone(),
                        );
                        if let Some(event_id) = &refinement_failed_event_id {
                            telemetry_last_event_id = Some(event_id.clone());
                        }
                        self.emit_phase_health_telemetry(
                            root_request_id,
                            refinement_failed_event_id.or_else(|| telemetry_last_event_id.clone()),
                            search_started_at.elapsed(),
                        );
                    }
                    on_phase(SearchPhase::RefinementFailed {
                        initial_results: display_hits.clone(),
                        error: timeout_error,
                        latency: phase2_latency,
                    });
                }
                Ok(phase2_outcome) => match phase2_outcome {
                    Ok(refined_results) => {
                        if let Err(cancellation) = cancellation_checkpoint(cx, "phase2_to_refined")
                        {
                            let (phase, reason) = match cancellation {
                                SearchError::Cancelled { phase, reason } => (phase, reason),
                                error => return Err(error),
                            };
                            if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                                self.emit_session_stop_telemetry(
                                    root_request_id,
                                    telemetry_last_event_id.clone(),
                                    LifecycleState::Degraded,
                                    LifecycleSeverity::Warn,
                                    Some(format!("cancelled:{phase}:{reason}")),
                                    search_started_at.elapsed(),
                                );
                            }
                            return Err(SearchError::Cancelled { phase, reason });
                        }

                        let phase2_latency = phase2_start.elapsed();
                        let refined_count = refined_results.len();
                        metrics.phase2_total_ms = phase2_latency.as_secs_f64() * 1000.0;
                        self.maybe_record_phase_gate_observation(
                            &initial_hits,
                            &refined_results,
                            query_class,
                        );
                        if let Some(circuit_breaker) = self.circuit_breaker.as_ref() {
                            let outcome = QualityOutcome::Success {
                                latency_ms: u64::try_from(phase2_latency.as_millis())
                                    .unwrap_or(u64::MAX),
                                tau_improvement: metrics.kendall_tau.unwrap_or(0.0),
                            };
                            let _ = circuit_breaker.record_outcome(&outcome);
                        }
                        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                            let refined_event_id = self.emit_search_telemetry(
                                semantic_query,
                                query_class,
                                SearchEventPhase::Refined,
                                refined_count,
                                metrics.lexical_candidates,
                                metrics.semantic_candidates,
                                metrics.hash_control_candidates,
                                phase2_latency,
                                root_request_id,
                                telemetry_initial_event_id.clone(),
                            );
                            if let Some(event_id) = &refined_event_id {
                                telemetry_last_event_id = Some(event_id.clone());
                            }
                            self.emit_phase_health_telemetry(
                                root_request_id,
                                refined_event_id.or_else(|| telemetry_last_event_id.clone()),
                                search_started_at.elapsed(),
                            );
                        }
                        self.export_search_metrics(query_class, &metrics, refined_count, true);
                        on_phase(SearchPhase::Refined {
                            results: refined_results.clone(),
                            latency: phase2_latency,
                            metrics: PhaseMetrics {
                                embedder_id: self
                                    .quality_embedder
                                    .as_ref()
                                    .map_or("none", |e| e.id())
                                    .to_owned(),
                                vectors_searched: metrics.phase2_vectors_searched,
                                lexical_candidates: metrics.lexical_candidates,
                                fused_count: refined_count,
                                skip_reason: None,
                                hash_control_candidates: metrics.hash_control_candidates,
                            },
                            rank_changes: metrics.rank_changes.clone(),
                        });

                        // Refined is a public phase boundary. A consumer may
                        // cancel after accepting it, before phase-3 hydration
                        // calls text_fn or invokes the reranker.
                        if let Err(SearchError::Cancelled { phase, reason }) =
                            cancellation_checkpoint(cx, "refined_to_rerank")
                        {
                            if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                                self.emit_session_stop_telemetry(
                                    root_request_id,
                                    telemetry_last_event_id.clone(),
                                    LifecycleState::Degraded,
                                    LifecycleSeverity::Warn,
                                    Some(format!("cancelled:{phase}:{reason}")),
                                    search_started_at.elapsed(),
                                );
                            }
                            return Err(SearchError::Cancelled { phase, reason });
                        }

                        // Phase 3: Reranking (optional, only if feature enabled and reranker present)
                        #[cfg(feature = "rerank")]
                        if self.reranker.is_some() {
                            let phase3_start = Instant::now();
                            match self
                                .run_phase3(
                                    cx,
                                    semantic_query,
                                    k,
                                    refined_results,
                                    &text_fn,
                                    &mut metrics,
                                )
                                .await
                            {
                                Ok(reranked_results) => {
                                    let phase3_latency = phase3_start.elapsed();
                                    let reranked_count = reranked_results.len();

                                    if let Some(root_request_id) =
                                        telemetry_root_request_id.as_deref()
                                    {
                                        let reranked_event_id = self.emit_search_telemetry(
                                            semantic_query,
                                            query_class,
                                            SearchEventPhase::Reranked,
                                            reranked_count,
                                            metrics.lexical_candidates,
                                            metrics.semantic_candidates,
                                            metrics.hash_control_candidates,
                                            phase3_latency,
                                            root_request_id,
                                            telemetry_initial_event_id.clone(),
                                        );
                                        if let Some(event_id) = &reranked_event_id {
                                            telemetry_last_event_id = Some(event_id.clone());
                                        }
                                    }

                                    on_phase(SearchPhase::Reranked {
                                        results: reranked_results,
                                        latency: phase3_latency + phase2_latency,
                                        metrics: PhaseMetrics {
                                            embedder_id: self
                                                .reranker
                                                .as_ref()
                                                .map_or("none", |r| r.id())
                                                .to_owned(),
                                            vectors_searched: metrics.phase2_vectors_searched,
                                            lexical_candidates: metrics.lexical_candidates,
                                            fused_count: reranked_count,
                                            skip_reason: None,
                                            hash_control_candidates: metrics
                                                .hash_control_candidates,
                                        },
                                    });
                                }
                                Err(SearchError::Cancelled { phase, reason }) => {
                                    if let Some(root_request_id) =
                                        telemetry_root_request_id.as_deref()
                                    {
                                        self.emit_session_stop_telemetry(
                                            root_request_id,
                                            telemetry_last_event_id.clone(),
                                            LifecycleState::Degraded,
                                            LifecycleSeverity::Warn,
                                            Some(format!("cancelled:{phase}:{reason}")),
                                            search_started_at.elapsed(),
                                        );
                                    }
                                    return Err(SearchError::Cancelled { phase, reason });
                                }
                                Err(err) => {
                                    self.export_error(&err);
                                    tracing::warn!(error = %err, "reranking phase failed");
                                    // Don't emit RefinementFailed here because Refined already succeeded.
                                }
                            }
                        }
                    }
                    Err(SearchError::Cancelled { phase, reason }) => {
                        if let Some(circuit_breaker) = self.circuit_breaker.as_ref() {
                            let _ = circuit_breaker.record_outcome(&QualityOutcome::Error);
                        }
                        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                            self.emit_session_stop_telemetry(
                                root_request_id,
                                telemetry_last_event_id.clone(),
                                LifecycleState::Degraded,
                                LifecycleSeverity::Warn,
                                Some(format!("cancelled:{phase}:{reason}")),
                                search_started_at.elapsed(),
                            );
                        }
                        return Err(SearchError::Cancelled { phase, reason });
                    }
                    Err(err) => {
                        let phase2_latency = phase2_start.elapsed();
                        metrics.phase2_total_ms = phase2_latency.as_secs_f64() * 1000.0;
                        metrics.skip_reason = Some(format!("{err}"));
                        self.export_error(&err);
                        if let Some(circuit_breaker) = self.circuit_breaker.as_ref() {
                            let _ = circuit_breaker.record_outcome(&QualityOutcome::Error);
                        }
                        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                            let refinement_failed_event_id = self.emit_search_telemetry(
                                semantic_query,
                                query_class,
                                SearchEventPhase::RefinementFailed,
                                display_hits.len(),
                                metrics.lexical_candidates,
                                metrics.semantic_candidates,
                                metrics.hash_control_candidates,
                                phase2_latency,
                                root_request_id,
                                telemetry_initial_event_id.clone(),
                            );
                            if let Some(event_id) = &refinement_failed_event_id {
                                telemetry_last_event_id = Some(event_id.clone());
                            }
                            self.emit_phase_health_telemetry(
                                root_request_id,
                                refinement_failed_event_id
                                    .or_else(|| telemetry_last_event_id.clone()),
                                search_started_at.elapsed(),
                            );
                        }
                        on_phase(SearchPhase::RefinementFailed {
                            initial_results: display_hits.clone(),
                            error: err,
                            latency: phase2_latency,
                        });
                    }
                },
            }
        } else if metrics.skip_reason.is_none() {
            if is_shipped_hash_embedder(self.fast_embedder.as_ref()) {
                metrics.skip_reason = Some("non_semantic_fast_embedder_vector_control".to_owned());
            } else if self.should_run_quality() {
                if quality_circuit_open {
                    metrics.skip_reason = Some("circuit_breaker_open".to_owned());
                } else if quality_phase_gate_skip {
                    metrics.skip_reason = Some("phase_gate_skip_quality".to_owned());
                } else if initial_hits.is_empty() {
                    metrics.skip_reason = Some("no_fast_phase_candidates".to_owned());
                } else {
                    metrics.skip_reason = Some("vector_index_unavailable".to_owned());
                }
            } else if self.config.fast_only {
                metrics.skip_reason = Some("fast_only".to_owned());
            } else if self.quality_embedder.is_none() {
                metrics.skip_reason = Some("no_quality_embedder".to_owned());
            } else {
                metrics.skip_reason = Some("quality_index_unavailable".to_owned());
            }
        }

        if let Err(cancellation) = cancellation_checkpoint(cx, "search_completion_to_session_stop")
        {
            let (phase, reason) = match cancellation {
                SearchError::Cancelled { phase, reason } => (phase, reason),
                error => return Err(error),
            };
            if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                self.emit_session_stop_telemetry(
                    root_request_id,
                    telemetry_last_event_id.clone(),
                    LifecycleState::Degraded,
                    LifecycleSeverity::Warn,
                    Some(format!("cancelled:{phase}:{reason}")),
                    search_started_at.elapsed(),
                );
            }
            return Err(SearchError::Cancelled { phase, reason });
        }

        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
            self.emit_session_stop_telemetry(
                root_request_id,
                telemetry_last_event_id,
                LifecycleState::Stopped,
                LifecycleSeverity::Info,
                metrics.skip_reason.clone(),
                search_started_at.elapsed(),
            );
        }

        Ok(metrics)
    }

    /// Convenience method that collects all phases and returns the best results.
    ///
    /// Returns the refined results if Phase 2 succeeds, otherwise the initial results.
    ///
    /// This method cannot evaluate exclusion clauses because it does not accept
    /// a document text provider. Use [`search_collect_with_text`](Self::search_collect_with_text)
    /// or [`search`](Self::search) when querying with exclusions.
    ///
    /// # Errors
    ///
    /// Same as [`search`](Self::search).
    pub async fn search_collect(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        let canon_query = self.canonicalizer.canonicalize_query(query);
        if ParsedQuery::parse(&canon_query).has_negations() {
            return Err(SearchError::QueryParseError {
                query: query.to_owned(),
                detail: "search_collect requires a text provider for exclusion syntax; use search_collect_with_text() or search()".to_owned(),
            });
        }

        self.search_collect_with_text(cx, query, k, |_| None).await
    }

    /// Convenience method that collects all phases while using `text_fn` for
    /// reranking and exclusion filtering.
    ///
    /// Returns the refined results if Phase 2 succeeds, otherwise the initial results.
    ///
    /// # Errors
    ///
    /// Same as [`search`](Self::search).
    pub async fn search_collect_with_text(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        let mut best_results = Vec::new();
        let metrics = self
            .search(cx, query, k, text_fn, |phase| match phase {
                SearchPhase::Initial { results, .. }
                | SearchPhase::Refined { results, .. }
                | SearchPhase::Reranked { results, .. } => {
                    best_results = results;
                }
                SearchPhase::RefinementFailed { .. } => {
                    // Keep the initial results already stored in best_results.
                }
            })
            .await?;
        Ok((best_results, metrics))
    }

    /// Prepare fusion candidates for direct return to the caller.
    ///
    /// Eager batches (no hydration context) already carry full metadata. A
    /// deferred batch is hydrated in place from its PINNED scoring snapshot —
    /// never by re-querying, which would both pay a second search and read a
    /// possibly newer generation than the one that scored these candidates
    /// (bd-8nqz.1). The candidates arrive already negation-filtered.
    pub(super) async fn lexical_results_for_direct_return(
        &self,
        cx: &Cx,
        fusion_candidates: &[ScoredResult],
        context: Option<&LexicalHydrationContext>,
        metrics: &mut TwoTierMetrics,
    ) -> SearchResult<Vec<ScoredResult>> {
        let mut results = fusion_candidates.to_vec();
        if let (Some(lexical), Some(context)) = (self.lexical.as_ref(), context) {
            let start = Instant::now();
            lexical
                .hydrate_candidates(cx, Some(context), &mut results)
                .await?;
            metrics.lexical_search_ms += start.elapsed().as_secs_f64() * 1000.0;
        }
        metrics.lexical_candidates = results.len();
        Ok(results)
    }
}
