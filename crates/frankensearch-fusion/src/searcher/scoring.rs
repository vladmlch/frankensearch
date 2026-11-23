#[cfg(feature = "rerank")]
use std::time::Instant;

#[cfg(feature = "rerank")]
use asupersync::Cx;

#[cfg(feature = "rerank")]
use frankensearch_core::config::TwoTierMetrics;
use frankensearch_core::config::ZeroSignalReason;
#[cfg(feature = "rerank")]
use frankensearch_core::error::SearchResult;
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::types::{ScoredResult, VectorHit};

#[cfg(feature = "rerank")]
use crate::mmr::mmr_rerank;
use crate::normalize::nqc_cv_iter;
use crate::phase_gate::PhaseObservation;

use super::TwoTierSearcher;
#[cfg(feature = "rerank")]
use super::admission::cancellation_checkpoint;
use super::results::is_shipped_hash_embedder;

impl TwoTierSearcher {
    /// The dense-tier phase-1 fusion weight for this query: the static `rrf_semantic_weight`,
    /// scaled by the per-query NQC dense down-weight when enabled (`beta > 0`). Off (default)
    /// returns `rrf_semantic_weight` unchanged with zero extra work.
    pub(super) fn effective_semantic_weight(&self, lexical: &[ScoredResult]) -> f64 {
        // The self-driving rolling down-weight takes precedence: it observes this query's NQC
        // (warming the online sketch) and scores it against the sketch from prior queries.
        if let Some(adaptive) = &self.nqc_adaptive {
            let cv = nqc_cv_iter(lexical.iter().map(|hit| hit.score));
            let factor = adaptive
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .weight_for_cv(cv);
            return self.rrf_semantic_weight * f64::from(factor);
        }
        if self.nqc_downweight_beta <= 0.0 {
            return self.rrf_semantic_weight;
        }
        let cv = if self.nqc_dense_weight.is_empty() {
            0.0
        } else {
            nqc_cv_iter(lexical.iter().map(|hit| hit.score))
        };
        let factor = self.nqc_dense_weight.dense_weight(
            cv,
            self.nqc_downweight_beta,
            self.nqc_downweight_w_min,
        );
        self.rrf_semantic_weight * f64::from(factor)
    }

    /// The Phase-1 neighbour-smoothing config, or `None` when the pass is disabled.
    ///
    /// Returning `None` (rather than an identity [`crate::smooth::SmoothConfig`]) is deliberate: it lets the
    /// caller skip the pass entirely instead of paying `neighbor_smooth`'s identity clone of the
    /// whole pool. `alpha == 0.0` is the default, so the default path never allocates.
    #[cfg(feature = "graph")]
    fn smoothing_config(&self) -> Option<crate::smooth::SmoothConfig> {
        let cfg = crate::smooth::SmoothConfig {
            alpha: self.config.neighbor_smoothing_alpha,
            m: self.config.neighbor_smoothing_m,
            mutual: self.config.neighbor_smoothing_mutual,
        };
        (!cfg.is_identity()).then_some(cfg)
    }

    /// The attached `r_d` table paired with a live hubness config, or `None` when the pass is
    /// disabled (default `beta == 0.0`, or no table attached). Same skip-don't-clone rationale as
    /// [`Self::smoothing_config`].
    fn hubness_correction(&self) -> Option<(&[f32], crate::hubness::HubnessConfig)> {
        let cfg = crate::hubness::HubnessConfig {
            beta: self.config.hubness_beta,
            ..crate::hubness::HubnessConfig::default()
        };
        if cfg.is_identity() {
            return None;
        }
        self.hubness_table.as_deref().map(|rd| (rd, cfg))
    }

    /// Phase-1 dense-score corrections over the retrieved semantic pool: query-hubness demotion,
    /// then k-NN graph diffusion, re-sorted **once** at the end.
    ///
    /// Both passes are off by default, and when both are off the pool is returned by move — no
    /// call, no clone, no sort — so the default path is byte-identical and pays nothing.
    ///
    /// **Hubness runs before smoothing** because diffusion must propagate *de-hubbed* scores. A hub
    /// is by construction adjacent to many docs, so smoothing an uncorrected pool would let each hub
    /// inject its inflated score into the neighbour means of exactly the specific answers the
    /// hubness penalty exists to rescue.
    ///
    /// **The re-sort is a correctness requirement, not cosmetics.** The rank-based fusion operators
    /// take each hit's rank from its *position* in this slice and never re-sort it (a vector index
    /// already returns score-sorted hits). Either correction rewrites scores, so without a re-sort a
    /// demoted hub keeps its rank, a promoted doc keeps its worse one, and the merge's near-sorted
    /// precondition is violated. Smoothing's output values do not depend on its input order, so
    /// sorting once after both passes is equivalent to sorting after each — and cheaper.
    pub(super) fn correct_phase1_pool(&self, hits: Vec<VectorHit>) -> Vec<VectorHit> {
        let hubness = self.hubness_correction();

        #[cfg(feature = "graph")]
        if let (Some(graph), Some(smooth_cfg)) =
            (self.document_graph.as_ref(), self.smoothing_config())
        {
            let demoted = match hubness {
                Some((rd, cfg)) => crate::hubness::apply_hubness_penalty(&hits, rd, &cfg),
                None => hits,
            };
            // Smoothing is the last stage here, and it re-sorts internally.
            return crate::smooth::neighbor_smooth_ranked(&demoted, graph, &smooth_cfg);
        }

        match hubness {
            Some((rd, cfg)) => {
                let mut demoted = crate::hubness::apply_hubness_penalty(&hits, rd, &cfg);
                demoted.sort_unstable_by(VectorHit::cmp_rank);
                demoted
            }
            None => hits,
        }
    }

    /// Classify an empty fast-tier result produced by the explicit
    /// `search_params` lane (bd-tqhc).
    ///
    /// Runs the classified fast lane purely for its state census — hit
    /// production stays with the params lane that already answered, and the
    /// two lanes scan the same records, so an empty params result implies an
    /// empty (classified) default result. Classification is diagnostic and
    /// must never fail the search: errors degrade to an unclassified lane at
    /// debug level rather than a fabricated reason.
    ///
    /// Known narrow gap: if the classified lane were to find hits where the
    /// params lane found none, this returns `None` and the empty result
    /// carries no reason. That needs the two lanes to disagree about
    /// *emptiness* specifically — the params lane is the exact scan and the
    /// classified lane's int8 two-pass is a candidate superset of it, so an
    /// empty exact result implies an empty int8 result and the disagreement
    /// cannot occur in that direction. Closing it properly wants a public
    /// census on `TwoTierIndex` (as `InMemoryVectorIndex` already exposes,
    /// which is how the sync searcher avoids both this gap and the second
    /// scan); that lives in `two_tier.rs` and is deferred rather than
    /// entangled with the in-flight ANN tranche.
    pub(super) fn classify_fast_empty(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> Option<ZeroSignalReason> {
        match self.index.search_fast_classified(query_vec, k) {
            Ok(classified) => classified.zero_signal,
            Err(error) => {
                tracing::debug!(
                    error = %error,
                    "fast-tier zero-signal classification failed; lane left unclassified"
                );
                None
            }
        }
    }

    /// Load one complete MMR candidate pool from a single embedding space.
    ///
    /// Prefer quality when every candidate has quality coverage. Otherwise
    /// retry the entire pool in the fast space. A pool spanning quality-only
    /// and fast-only documents has no valid comparison space, so MMR is
    /// skipped instead of comparing unrelated coordinates.
    #[cfg(feature = "rerank")]
    pub(super) fn mmr_embeddings_in_one_space(
        &self,
        cx: &Cx,
        results: &[ScoredResult],
    ) -> SearchResult<Option<Vec<Vec<f32>>>> {
        let mut quality = Vec::with_capacity(results.len());
        for result in results {
            let embedding = self.index.quality_vector_for_doc_id(&result.doc_id)?;
            cancellation_checkpoint(cx, "mmr_quality_vector_lookup")?;
            let Some(embedding) = embedding else {
                quality.clear();
                break;
            };
            quality.push(embedding);
        }
        if quality.len() == results.len() {
            return Ok(Some(quality));
        }

        let mut fast = Vec::with_capacity(results.len());
        for result in results {
            let embedding = self.index.fast_vector_for_doc_id(&result.doc_id)?;
            cancellation_checkpoint(cx, "mmr_fast_vector_lookup")?;
            let Some(embedding) = embedding else {
                return Ok(None);
            };
            fast.push(embedding);
        }
        Ok(Some(fast))
    }

    /// Run Phase 3: cross-encoder reranking and MMR diversity.
    #[cfg(feature = "rerank")]
    pub(super) async fn run_phase3(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        mut results: Vec<ScoredResult>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        metrics: &mut TwoTierMetrics,
    ) -> SearchResult<Vec<ScoredResult>> {
        if let Some(ref reranker) = self.reranker {
            let rerank_start = Instant::now();
            let rerank_budget = k.min(results.len());
            if rerank_budget > 0 {
                frankensearch_rerank::pipeline::rerank_step_with_combine(
                    cx,
                    reranker.as_ref(),
                    query,
                    &mut results,
                    text_fn,
                    rerank_budget,
                    5,
                    self.rerank_combine,
                )
                .await?;
            }
            metrics.rerank_ms = rerank_start.elapsed().as_secs_f64() * 1000.0;
        }

        cancellation_checkpoint(cx, "rerank_to_mmr")?;

        if self.mmr_config.enabled && results.len() > 1 {
            let pool = results.len().min(self.mmr_config.candidate_pool.max(1));
            if pool > 1 {
                if let Some(embeddings) = self.mmr_embeddings_in_one_space(cx, &results[..pool])? {
                    cancellation_checkpoint(cx, "mmr_lookup_to_rerank")?;
                    let scores = results
                        .iter()
                        .take(pool)
                        .map(|result| f64::from(result.score))
                        .collect::<Vec<_>>();
                    let refs = embeddings
                        .iter()
                        .map(std::vec::Vec::as_slice)
                        .collect::<Vec<_>>();
                    let order = mmr_rerank(&scores, &refs, pool, &self.mmr_config);
                    if order.len() == pool {
                        // `order` is a distinct permutation of `0..pool` (mmr_rerank
                        // never repeats a candidate), so reorder the head by MOVING
                        // each element into its MMR position instead of cloning the
                        // pool twice (once into `head`, once per `order` index). Take
                        // each slot at most once; the tail is moved through untouched.
                        let split = pool.min(results.len());
                        let tail = results.split_off(split);
                        let mut slots: Vec<Option<ScoredResult>> =
                            results.into_iter().map(Some).collect();
                        let mut reranked = Vec::with_capacity(slots.len() + tail.len());
                        for idx in order {
                            if let Some(item) = slots.get_mut(idx).and_then(Option::take) {
                                reranked.push(item);
                            }
                        }
                        reranked.extend(tail);
                        results = reranked;
                    }
                }
            }
        }

        cancellation_checkpoint(cx, "mmr_to_explanation")?;

        if self.config.explain {
            for (final_rank, result) in results.iter_mut().enumerate() {
                if let Some(ref mut explanation) = result.explanation {
                    if let Some(ref mut movement) = explanation.rank_movement {
                        movement.refined_rank = final_rank;
                        let refined_i64 = i64::try_from(final_rank).unwrap_or(i64::MAX);
                        let initial_i64 = i64::try_from(movement.initial_rank).unwrap_or(i64::MAX);
                        let delta_i64 = refined_i64.saturating_sub(initial_i64);
                        movement.delta = i32::try_from(delta_i64).unwrap_or_else(|_| {
                            if delta_i64.is_negative() {
                                i32::MIN
                            } else {
                                i32::MAX
                            }
                        });
                        movement.reason = match movement.delta.cmp(&0) {
                            std::cmp::Ordering::Less => "promoted".to_owned(),
                            std::cmp::Ordering::Greater => "demoted".to_owned(),
                            std::cmp::Ordering::Equal => "stable".to_owned(),
                        };
                    }
                }
            }
        }

        cancellation_checkpoint(cx, "explanation_to_publish")?;
        Ok(results)
    }

    /// Whether quality refinement should run.
    pub(super) fn should_run_quality(&self) -> bool {
        !self.config.fast_only
            && self.quality_embedder.is_some()
            && self.index.has_quality_index()
            && !is_shipped_hash_embedder(self.fast_embedder.as_ref())
    }

    pub(super) fn phase_gate_should_skip_quality(&self) -> bool {
        let Some(phase_gate) = self.phase_gate.as_ref() else {
            return false;
        };
        phase_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .should_skip_quality()
    }

    pub(super) fn maybe_record_phase_gate_observation(
        &self,
        initial_results: &[ScoredResult],
        refined_results: &[ScoredResult],
        query_class: QueryClass,
    ) {
        let Some(phase_gate) = self.phase_gate.as_ref() else {
            return;
        };
        let (Some(initial_top), Some(refined_top)) =
            (initial_results.first(), refined_results.first())
        else {
            return;
        };

        let evidence = {
            let mut phase_gate = phase_gate
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            phase_gate.update(&PhaseObservation {
                fast_score: f64::from(initial_top.score),
                quality_score: f64::from(refined_top.score),
                user_signal: None,
            })
        };
        for record in evidence {
            tracing::debug!(
                reason_code = %record.reason_code,
                event_type = %record.event_type,
                pipeline_state = %record.pipeline_state,
                query_class = ?query_class,
                source_component = %record.source_component,
                "{}",
                record.reason_human
            );
        }
    }

    pub(super) fn effective_rrf_k(&self, query_class: QueryClass) -> f64 {
        self.adaptive_fusion
            .as_ref()
            .map_or(self.config.rrf_k, |adaptive| adaptive.rrf_k(query_class))
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn effective_blend_factor(&self, query_class: QueryClass) -> f32 {
        self.adaptive_fusion.as_ref().map_or_else(
            || self.config.quality_weight as f32,
            |adaptive| adaptive.blend_factor(query_class) as f32,
        )
    }

    pub(super) fn conformal_candidate_target(&self, requested_k: usize) -> usize {
        let Some(calibration) = self.conformal_calibration.as_ref() else {
            return requested_k;
        };
        let alpha = self.adaptive_conformal_state.as_ref().map_or(0.1, |state| {
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .alpha
        });
        match calibration.required_k_checked(alpha) {
            Ok(required) => requested_k.max(required),
            Err(error) => {
                self.export_error(&error);
                requested_k
            }
        }
    }

    /// Apply the configured score calibrator to a single score (identity if no
    /// calibrator is set). Non-finite calibrated values collapse to `0.0`, matching
    /// [`Self::apply_score_calibration_to_hits`].
    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn calibrate_score(&self, score: f32) -> f32 {
        let Some(calibrator) = self.score_calibrator.as_ref() else {
            return score;
        };
        let calibrated = calibrator.calibrate(f64::from(score)) as f32;
        if calibrated.is_finite() {
            calibrated
        } else {
            0.0
        }
    }

    pub(super) fn apply_score_calibration_to_hits(&self, hits: &mut [VectorHit]) {
        if self.score_calibrator.is_none() {
            return;
        }
        for hit in hits {
            hit.score = self.calibrate_score(hit.score);
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn maybe_update_adaptive_conformal(&self, tau: Option<f64>) {
        let Some(calibration) = self.conformal_calibration.as_ref() else {
            return;
        };
        let Some(state_lock) = self.adaptive_conformal_state.as_ref() else {
            return;
        };

        let observed_error_rate = tau.map_or(0.5_f32, |value| {
            let normalized = value.midpoint(1.0).clamp(0.0, 1.0);
            (1.0 - normalized) as f32
        });

        let mut state = state_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(error) = state.update(observed_error_rate, calibration) {
            self.export_error(&error);
            tracing::warn!(
                error = %error,
                observed_error_rate,
                "adaptive conformal update failed"
            );
        }
    }
}
