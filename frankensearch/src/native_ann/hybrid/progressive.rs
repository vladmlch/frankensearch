//! Lazy native refinement over the same retained fast and lexical candidates.

use std::collections::BTreeSet;

use frankensearch_core::LexicalCandidateBatch;
use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_core::traits::Reranker;
use frankensearch_fusion::{RrfConfig, rrf_fuse_for_vector_lane};

use super::refinement::{RefinementFusion, admit_pair, checked_lexical, refined_winners};
use super::{NativeAnnIndex, checkpoint, hydrate_winners, invalid, join_sources};
use crate::{Cx, Embedder, LexicalRead, ScoredResult, SearchError, SearchResult, VectorHit};

pub(super) mod rerank;
use rerank::RerankRequest;

impl NativeAnnIndex {
    /// Prepare native progressive search without starting inference or retrieval.
    ///
    /// The first `next_phase` retrieves this fast owner plus lexical candidates.
    /// Only a second call starts independent retrieval from the optional quality
    /// owner. Stopping after initial results starts no quality inference. With
    /// no quality arm, the stream yields one initial phase unless an optional
    /// [`NativeProgressiveSearch::with_reranker`] stage is attached. Direct
    /// quality-only retrieval remains [`Self::search_hybrid_quality_text`].
    ///
    /// Generation, producer, space-kind and document-ID joins use the same
    /// admission boundary as [`Self::search_hybrid_refined_text`]. Refinement
    /// uses its exact blend, RRF, raw-score and owner-specific row-index policy,
    /// but reuses the original fast candidate pool and pinned lexical snapshot.
    /// This does not prove a composite lexical/vector generation or change the
    /// default searcher. Graph work remains synchronous on the polling executor.
    ///
    /// # Errors
    ///
    /// Refuses invalid admission or overflow of the three-times candidate
    /// budget. Every configured identity is checked even for zero-k requests.
    pub fn search_hybrid_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        fast_embedder: &'a dyn Embedder,
        quality: Option<(&'a Self, &'a dyn Embedder)>,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        let search = NativeProgressiveSearch {
            fast: self,
            cx,
            fast_embedder,
            quality,
            lexical,
            text,
            k,
            reranker: None,
            allowed_documents: None,
            candidate_multiplier: 3,
            rrf: RrfConfig::default(),
            quality_weight: 0.7,
            fast_ef: None,
            quality_ef: None,
            budget: k.checked_mul(3).ok_or_else(|| {
                invalid(
                    "progressive.candidate_budget",
                    "overflow",
                    "three times the requested result count must fit usize",
                )
            })?,
            state: State::Initial,
        };
        search.admit()?;
        Ok(search)
    }
}

/// Actual returned candidate counts, not graph visits or corpus-coverage claims.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativePhaseCandidates {
    /// Candidates retrieved from the retained fast owner.
    pub fast: usize,
    /// Candidates independently retrieved from the retained quality owner.
    pub quality: usize,
    /// Candidates in the original pinned lexical scoring batch.
    pub lexical: usize,
}

/// A requested phase of caller-owned native progressive retrieval.
#[derive(Debug)]
pub enum NativeSearchPhase {
    /// Fast-plus-lexical results ready without any quality inference.
    Initial {
        /// Hydrated winners in fused rank order.
        results: Vec<ScoredResult>,
        /// Observed candidate counts; quality is zero in this phase.
        candidates: NativePhaseCandidates,
    },
    /// Winners after independent quality retrieval and candidate union.
    Refined {
        /// Hydrated winners; these may include documents outside the fast pool.
        results: Vec<ScoredResult>,
        /// Counts from retained fast/lexical pools and the new quality pool.
        candidates: NativePhaseCandidates,
    },
    /// A non-cancellation refinement failure; initial results remain valid.
    RefinementFailed {
        /// Original initial results, unchanged by the attempted refinement.
        initial_results: Vec<ScoredResult>,
        /// Failure in admission, quality retrieval, fusion, or hydration.
        error: SearchError,
    },
    /// Final cross-encoder ordering of the retained retrieval candidate window.
    Reranked {
        /// At most the requested `k` winners, with original retrieval evidence.
        results: Vec<ScoredResult>,
        /// Counts from the retrieval phase, not cross-encoder model calls.
        candidates: NativePhaseCandidates,
        /// Number of candidate documents that had text and were scored.
        evaluated: usize,
    },
    /// Non-cancellation rerank failure; the last displayed results remain valid.
    RerankFailed {
        /// The previous displayed page, unchanged by the failed rerank.
        previous_results: Vec<ScoredResult>,
        /// Model availability, response admission, or inference failure.
        error: SearchError,
    },
}

struct PendingRefinement {
    fast: Vec<VectorHit>,
    batch: LexicalCandidateBatch,
    initial_results: Vec<ScoredResult>,
}

struct PendingRerank {
    results: Vec<ScoredResult>,
    candidates: NativePhaseCandidates,
    // Retain the scoring snapshot through text resolution and model inference.
    // The caller's text resolver must itself select the same document versions.
    _batch: LexicalCandidateBatch,
}

enum State {
    Initial,
    Refine(PendingRefinement),
    Rerank(PendingRerank),
    Done,
}

/// Lazy search retaining the exact fast pool and lexical scoring snapshot.
///
/// Dropping this stream after initial results starts no quality work. Dropping
/// a *polled, still-pending* `next_phase` future permanently finishes the stream
/// and releases its in-flight work and snapshot pin. Subsequent calls return
/// `None` rather than retrying inference, traversal, or lexical retrieval.
/// Dropping an unpolled future starts no phase and leaves the stream unchanged.
///
/// There are no detached tasks or hidden runtimes. Cancellation checkpoints
/// surround synchronous graph work; no preemptive interruption or implicit
/// quality timeout is claimed. A provider cancellation propagates as `Err`,
/// never as a successful refinement or a `RefinementFailed` degradation.
///
/// Construct with [`NativeAnnIndex::search_hybrid_progressive`].
pub struct NativeProgressiveSearch<'a> {
    fast: &'a NativeAnnIndex,
    cx: &'a Cx,
    fast_embedder: &'a dyn Embedder,
    quality: Option<(&'a NativeAnnIndex, &'a dyn Embedder)>,
    lexical: &'a dyn LexicalRead,
    text: &'a str,
    k: usize,
    budget: usize,
    candidate_multiplier: usize,
    rrf: RrfConfig,
    quality_weight: f32,
    fast_ef: Option<usize>,
    quality_ef: Option<usize>,
    reranker: Option<RerankRequest<'a>>,
    // Installed only by the built-cohort scope adapter, together with its
    // filtered lexical reader. None preserves the ordinary unscoped path.
    allowed_documents: Option<&'a BTreeSet<String>>,
    state: State,
}

impl<'a> NativeProgressiveSearch<'a> {
    /// Set the fusion policy for BOTH Initial and Refined before either starts.
    ///
    /// Defaults are the existing RRF policy and 0.7 quality blend. `rrf` is
    /// owned by this query, so later caller configuration cannot change a phase.
    /// Its semantic weight also weights a hash-control vector lane without
    /// relabelling that lane. Every source weight must be strictly positive;
    /// zero is not a way to disable a required provider or its admission checks.
    ///
    /// The quality weight applies to documents present in BOTH vector pools.
    /// One-sided candidates keep their normalized score, including at 0 and 1,
    /// as in the existing union-preserving blend. Raw scores, physical rows and
    /// required-quality failure behavior do not change with weighting.
    ///
    /// # Errors
    /// Refuses cancellation, started queries, non-finite/out-of-range policy,
    /// or a maximum two-source RRF score outside public f32 result precision.
    pub fn with_fusion(mut self, rrf: RrfConfig, quality_weight: f32) -> SearchResult<Self> {
        self.configure()?;
        if !quality_weight.is_finite() || !(0.0..=1.0).contains(&quality_weight) {
            return Err(invalid(
                "progressive.quality_weight",
                "out-of-range",
                "quality blend weight must be finite and between zero and one",
            ));
        }
        if !rrf.k.is_finite()
            || rrf.k < 0.0
            || !rrf.lexical_weight.is_finite()
            || rrf.lexical_weight <= 0.0
            || !rrf.semantic_weight.is_finite()
            || rrf.semantic_weight <= 0.0
        {
            return Err(invalid(
                "progressive.rrf",
                "invalid",
                "RRF K must be finite and nonnegative, and source weights finite and positive",
            ));
        }
        let first_rank = 1.0 / (rrf.k + 1.0);
        let maximum_score = first_rank * rrf.lexical_weight + first_rank * rrf.semantic_weight;
        if !maximum_score.is_finite() || maximum_score > f64::from(f32::MAX) {
            return Err(invalid(
                "progressive.rrf",
                "score-overflow",
                "the maximum fused score must fit the public f32 result",
            ));
        }
        self.rrf = rrf;
        self.quality_weight = quality_weight;
        Ok(self)
    }

    /// Request `multiplier * max(k, rerank_window)` candidates from EACH source.
    ///
    /// The default is three. This is independent of the displayed page size and
    /// may change both pool-local normalization and final ranking. Configuring
    /// the reranker before or after this method produces the same budget.
    /// Scoped retrieval retains its own eligible-candidate and widening rules.
    /// This bounds requested hits, not graph visits or provider peak memory.
    ///
    /// # Errors
    /// Refuses zero, checked-arithmetic overflow, cancellation or a started query.
    pub fn with_candidate_multiplier(mut self, multiplier: usize) -> SearchResult<Self> {
        self.configure()?;
        if multiplier == 0 {
            return Err(invalid(
                "progressive.candidate_multiplier",
                "zero",
                "candidate multiplier must be positive",
            ));
        }
        self.budget = self
            .result_window()
            .checked_mul(multiplier)
            .ok_or_else(|| {
                invalid(
                    "progressive.candidate_budget",
                    "overflow",
                    "the configured candidate budget must fit usize",
                )
            })?;
        self.candidate_multiplier = multiplier;
        Ok(self)
    }

    /// Override the fast and quality native graph beams independently.
    ///
    /// None uses that owner's admitted default. The existing search policy
    /// raises a beam to its requested-hit floor, caps it at physical row count,
    /// and widens it on filtered underfill. Exact owners still ignore the beam.
    /// These settings do not rebuild graphs, alter source counts or certify ANN
    /// recall. Every configured provider is still independently admitted.
    ///
    /// # Errors
    /// Refuses a zero beam, cancellation or configuration after a phase starts.
    pub fn with_beam_widths(
        mut self,
        fast: Option<usize>,
        quality: Option<usize>,
    ) -> SearchResult<Self> {
        self.configure()?;
        if fast == Some(0) || quality == Some(0) {
            return Err(invalid(
                "progressive.beam",
                "zero",
                "an explicit graph beam must be positive",
            ));
        }
        self.fast_ef = fast;
        self.quality_ef = quality;
        Ok(self)
    }

    fn configure(&self) -> SearchResult<()> {
        checkpoint(self.cx, "native_ann.progressive_configuration")?;
        if !matches!(self.state, State::Initial) {
            return Err(invalid(
                "progressive.configuration",
                "started",
                "query policy must be fixed before requesting its first phase",
            ));
        }
        Ok(())
    }

    #[cfg(feature = "quill")]
    pub(in crate::native_ann) fn with_allowed_documents(
        mut self,
        allowed: &'a BTreeSet<String>,
    ) -> SearchResult<Self> {
        checkpoint(self.cx, "native_ann.scope.configure")?;
        if !matches!(self.state, State::Initial) {
            return Err(invalid(
                "scope.configuration",
                "started",
                "scope membership must be frozen before the first phase",
            ));
        }
        self.allowed_documents = Some(allowed);
        Ok(self)
    }

    async fn search_tier(
        &self,
        index: &NativeAnnIndex,
        embedder: &dyn Embedder,
        ef: Option<usize>,
    ) -> SearchResult<Vec<VectorHit>> {
        match self.allowed_documents {
            Some(allowed) => {
                index
                    .search_text_filtered(
                        self.cx,
                        embedder,
                        self.text,
                        self.budget.min(allowed.len()),
                        ef,
                        |id| allowed.contains(id),
                    )
                    .await
            }
            None => {
                index
                    .search_text(self.cx, embedder, self.text, self.budget, ef)
                    .await
            }
        }
    }

    /// Add a lazy final cross-encoder stage before requesting the first phase.
    ///
    /// `top_k` bounds the rerank window, independently of the displayed page.
    /// Retrieval budgets expand to the configured multiplier times
    /// `max(k, top_k)` (three by default) so reranking can
    /// promote candidates outside the initial page. Enabling this option may
    /// therefore change initial fusion relative to a smaller, unreranked query.
    /// Only requesting the final phase resolves text or calls the reranker;
    /// abandoning earlier phases incurs no cross-encoder inference.
    ///
    /// `text_fn` must resolve text from the same document versions as retrieval,
    /// for example by capturing the consumer's immutable source snapshot. This
    /// API retains the lexical scoring pin, but cannot prove the provenance of
    /// an arbitrary closure. Missing text skips that candidate. When no text is
    /// available, the final `next_phase` returns `None`, not a fabricated rerank.
    ///
    /// A failed quality refinement still permits reranking the retained initial
    /// pool. Inference is caller-owned and cancellation/drop is terminal. Raw
    /// retrieval scores and their explanations are preserved; `rerank_score`
    /// records the new ordering signal. No model is downloaded or substituted.
    ///
    /// # Errors
    ///
    /// Refuses zero/overflowing budgets, cancellation, or configuration after
    /// a phase has started. Model availability is checked only at the final phase.
    pub fn with_reranker(
        mut self,
        reranker: &'a dyn Reranker,
        text_fn: &'a (dyn Fn(&str) -> Option<String> + Send + Sync),
        top_k: usize,
    ) -> SearchResult<Self> {
        checkpoint(self.cx, "native_ann.rerank_configuration")?;
        if !matches!(self.state, State::Initial) || top_k == 0 {
            return Err(invalid(
                "rerank.configuration",
                "started-or-empty-window",
                "attach a positive rerank window before requesting any phase",
            ));
        }
        self.budget = self
            .k
            .max(top_k)
            .checked_mul(self.candidate_multiplier)
            .ok_or_else(|| {
                invalid(
                    "rerank.candidate_budget",
                    "overflow",
                    "rerank retrieval budget must fit usize",
                )
            })?;
        self.reranker = Some(RerankRequest {
            reranker,
            text: text_fn,
            limit: top_k,
        });
        Ok(self)
    }

    /// Whether every phase finished or a polled phase was abandoned.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.state, State::Done)
    }

    /// Execute every configured phase and return only the final successful page.
    ///
    /// This consumes an unstarted query with all of its configured fusion,
    /// candidate, beam and scope settings. Missing rerank text keeps the last
    /// valid retrieval page. Use `next_phase` instead for early delivery and
    /// explicit degradation pages.
    ///
    /// # Errors
    /// Propagates any required phase failure or cancellation, including errors
    /// carried by RefinementFailed/RerankFailed; it never silently returns an
    /// earlier page as successful refinement. Refuses an already-started query.
    pub async fn collect(mut self) -> SearchResult<Vec<ScoredResult>> {
        self.configure()?;
        let mut page = Vec::new();
        while let Some(phase) = self.next_phase().await? {
            page = match phase {
                NativeSearchPhase::Initial { results, .. }
                | NativeSearchPhase::Refined { results, .. }
                | NativeSearchPhase::Reranked { results, .. } => results,
                NativeSearchPhase::RefinementFailed { error, .. }
                | NativeSearchPhase::RerankFailed { error, .. } => return Err(error),
            };
        }
        Ok(page)
    }

    /// Compute one requested phase, then finish permanently after the last.
    ///
    /// # Errors
    ///
    /// Initial failures and any cancellation propagate directly. After initial
    /// success, other stage failures carry the preceding results. Optional
    /// reranking follows quality success or non-cancellation failure. Every
    /// returned `Err` leaves this stream finished.
    pub async fn next_phase(&mut self) -> SearchResult<Option<NativeSearchPhase>> {
        // Own the phase before the first suspension. Cancellation by dropping
        // this future cannot leave a runnable copy of the work in the stream.
        match std::mem::replace(&mut self.state, State::Done) {
            State::Done => Ok(None),
            State::Initial => self.initial().await.map(Some),
            State::Refine(pending) => self.refine(pending).await.map(Some),
            State::Rerank(pending) => self.rerank(pending).await,
        }
    }

    fn admit(&self) -> SearchResult<bool> {
        if let Some(quality) = self.quality {
            return admit_pair(self.cx, self.fast, self.fast_embedder, quality);
        }
        checkpoint(self.cx, "native_ann.progressive_admission")?;
        let identity = self.fast_embedder.identity()?;
        self.fast.admit_identity(identity)?;
        // Classify the same admitted reference, not a second provider call.
        Ok(matches!(
            identity.space.kind,
            EmbeddingSpaceKindV1::HashControl
        ))
    }

    async fn initial(&mut self) -> SearchResult<NativeSearchPhase> {
        let is_hash = self.admit()?;
        if self.k == 0 {
            return Ok(NativeSearchPhase::Initial {
                results: Vec::new(),
                candidates: NativePhaseCandidates::default(),
            });
        }
        let (fast, batch) = join_sources(
            self.cx,
            self.search_tier(self.fast, self.fast_embedder, self.fast_ef),
            checked_lexical(self.cx, self.lexical, self.text, self.budget),
        )
        .await?;
        let candidates = NativePhaseCandidates {
            fast: fast.len(),
            quality: 0,
            lexical: batch.results().len(),
        };
        let fused = rrf_fuse_for_vector_lane(
            batch.results(),
            &fast,
            self.result_window(),
            0,
            &self.rrf,
            is_hash,
        );
        let mut results = hydrate_winners(self.cx, self.lexical, fused, &batch).await?;
        checkpoint(self.cx, "native_ann.progressive_initial_complete")?;
        if self.quality.is_some() {
            self.state = State::Refine(PendingRefinement {
                fast,
                batch,
                initial_results: results.clone(),
            });
            results.truncate(self.k);
        } else {
            results = self.finish_retrieval(results, candidates, batch);
        }
        Ok(NativeSearchPhase::Initial {
            results,
            candidates,
        })
    }

    async fn refine(&mut self, pending: PendingRefinement) -> SearchResult<NativeSearchPhase> {
        match self.refined_results(&pending).await {
            Ok((results, candidates)) => {
                let results = self.finish_retrieval(results, candidates, pending.batch);
                Ok(NativeSearchPhase::Refined {
                    results,
                    candidates,
                })
            }
            Err(error) => {
                checkpoint(self.cx, "native_ann.progressive_failure")?;
                if matches!(error, SearchError::Cancelled { .. }) {
                    return Err(error);
                }
                let candidates = NativePhaseCandidates {
                    fast: pending.fast.len(),
                    quality: 0,
                    lexical: pending.batch.results().len(),
                };
                let initial_results =
                    self.finish_retrieval(pending.initial_results, candidates, pending.batch);
                Ok(NativeSearchPhase::RefinementFailed {
                    initial_results,
                    error,
                })
            }
        }
    }

    async fn refined_results(
        &self,
        pending: &PendingRefinement,
    ) -> SearchResult<(Vec<ScoredResult>, NativePhaseCandidates)> {
        let is_hash = self.admit()?;
        let (quality, embedder) = self.quality.ok_or_else(|| {
            invalid(
                "progressive.quality",
                "missing",
                "refinement requires its admitted arm",
            )
        })?;
        let quality_hits = self.search_tier(quality, embedder, self.quality_ef).await?;
        let candidates = NativePhaseCandidates {
            fast: pending.fast.len(),
            quality: quality_hits.len(),
            lexical: pending.batch.results().len(),
        };
        let results = refined_winners(
            self.cx,
            self.lexical,
            &pending.batch,
            &pending.fast,
            &quality_hits,
            self.result_window(),
            RefinementFusion {
                quality_weight: self.quality_weight,
                rrf: &self.rrf,
                is_hash,
            },
        )
        .await?;
        checkpoint(self.cx, "native_ann.progressive_refined_complete")?;
        Ok((results, candidates))
    }

    fn result_window(&self) -> usize {
        self.reranker
            .map_or(self.k, |request| self.k.max(request.limit))
    }

    fn finish_retrieval(
        &mut self,
        mut results: Vec<ScoredResult>,
        candidates: NativePhaseCandidates,
        batch: LexicalCandidateBatch,
    ) -> Vec<ScoredResult> {
        if self.reranker.is_some() && !results.is_empty() {
            let displayed = results.iter().take(self.k).cloned().collect();
            self.state = State::Rerank(PendingRerank {
                results,
                candidates,
                _batch: batch,
            });
            displayed
        } else {
            results.truncate(self.k);
            results
        }
    }

    async fn rerank(&self, pending: PendingRerank) -> SearchResult<Option<NativeSearchPhase>> {
        let request = self.reranker.as_ref().ok_or_else(|| {
            invalid(
                "rerank.configuration",
                "missing",
                "a queued rerank must retain its provider",
            )
        })?;
        match request.plan(self.cx, self.text, &pending.results).await {
            Ok(Some(plan)) => {
                let evaluated = plan.evaluated;
                let mut results = plan.apply(pending.results, |result| result);
                results.truncate(self.k);
                checkpoint(self.cx, "native_ann.rerank_complete")?;
                Ok(Some(NativeSearchPhase::Reranked {
                    results,
                    candidates: pending.candidates,
                    evaluated,
                }))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                checkpoint(self.cx, "native_ann.rerank_failure")?;
                if matches!(error, SearchError::Cancelled { .. }) {
                    return Err(error);
                }
                let mut previous_results = pending.results;
                previous_results.truncate(self.k);
                Ok(Some(NativeSearchPhase::RerankFailed {
                    previous_results,
                    error,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests;
