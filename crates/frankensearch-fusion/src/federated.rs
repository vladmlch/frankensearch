//! Federated search across multiple independent [`TwoTierSearcher`] instances.
//!
//! A single query fans out to multiple indices, then gathered results are fused
//! into one ranked list. Fusion accumulates contributions in index-name order,
//! not response order, so scheduling cannot change scores for the same replies.

use std::future::Future;
use std::future::poll_fn;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use asupersync::Cx;
use asupersync::time::timeout;
use frankensearch_core::{ScoreSource, ScoredResult, SearchError, SearchResult};
use tracing::{debug, warn};

use crate::normalize::{NormalizationMethod, normalize_scores_with_method};
use crate::searcher::TwoTierSearcher;

const DEFAULT_FEDERATED_RRF_K: f64 = 60.0;

/// Fusion methods supported by [`FederatedSearcher`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FederatedFusion {
    /// Reciprocal Rank Fusion across indices.
    ///
    /// Score contribution:
    /// `weight / (k + rank + 1)`.
    Rrf { k: f64 },

    /// Weighted sum of normalized per-index scores.
    WeightedScore {
        /// Normalization applied independently per index before weighted sum.
        normalization: NormalizationMethod,
    },

    /// `CombMNZ`: weighted normalized sum multiplied by appearance count.
    ///
    /// `comb_mnz = weighted_sum * count(indices_containing_doc)`.
    CombMnz {
        /// Normalization applied independently per index before weighted sum.
        normalization: NormalizationMethod,
    },
}

impl Default for FederatedFusion {
    fn default() -> Self {
        Self::Rrf {
            k: DEFAULT_FEDERATED_RRF_K,
        }
    }
}

/// Configuration for federated search behavior.
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// How shard results are fused.
    pub fusion_method: FederatedFusion,
    /// Per-index timeout budget in milliseconds.
    pub per_index_timeout_ms: u64,
    /// Minimum number of indices that must respond successfully — a FLOOR,
    /// enforced after the gather completes. Fewer successful responses fail the
    /// query with [`SearchError::FederatedInsufficientResponses`]. This knob
    /// never stops the gather early; that is `wait_for_indices`.
    pub min_indices: usize,
    /// Candidate multiplier per index before global fusion.
    pub candidate_pool_factor: usize,
    /// Maximum number of indices to query.
    pub max_indices: usize,
    /// Latency/recall trade-off: stop gathering (cancelling still-in-flight
    /// shards) once this many shards have responded successfully. `usize::MAX`
    /// (the default) and `0` both mean "wait for every dispatched shard"
    /// (bounded per shard by `per_index_timeout_ms`). Cancelled shards are
    /// reported in [`FederatedCoverage::cancelled_in_flight`] (bd-3zh67).
    pub wait_for_indices: usize,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            fusion_method: FederatedFusion::default(),
            per_index_timeout_ms: 500,
            min_indices: 1,
            candidate_pool_factor: 3,
            max_indices: usize::MAX,
            wait_for_indices: usize::MAX,
        }
    }
}

#[derive(Debug, Clone)]
struct FederatedIndex {
    name: String,
    searcher: Arc<TwoTierSearcher>,
    weight: f32,
}

#[derive(Debug)]
struct ShardResult {
    name: String,
    weight: f32,
    hits: Vec<ScoredResult>,
}

#[derive(Debug)]
enum ShardCompletion {
    Completed(ShardResult),
    Failed { index: String, error: SearchError },
    TimedOut { index: String },
    Cancelled { phase: String, reason: String },
}

type ShardFuture<'a> = Pin<Box<dyn Future<Output = ShardCompletion> + Send + 'a>>;

#[derive(Debug, Clone)]
struct AggregateDoc {
    template: ScoredResult,
    primary_index: String,
    primary_rank: usize,
    primary_contribution: f32,
    fused_score: f32,
    /// Interned shard ids (rank in the sorted distinct-shard-name list) this doc
    /// appeared in. Was a `BTreeSet<String>` cloning a shard-name `String` on every
    /// `accumulate_doc` call (thousands per fuse); interning to integer ids and
    /// deduping once at output (`into_ranked_hits`) drops that per-call churn.
    /// Ids are the sorted-name rank, so sorting them ascending == sorting the names
    /// lexicographically (the `BTreeSet<String>` iteration order the output kept).
    appeared_in: Vec<u32>,
}

/// A fused hit returned by federated search.
#[derive(Debug, Clone)]
pub struct FederatedHit {
    /// The fused result payload.
    pub result: ScoredResult,
    /// Index where the strongest contribution came from.
    pub source_index: String,
    /// Rank in `source_index` (0-based).
    pub source_rank: usize,
    /// All indices where this document appeared.
    pub appeared_in: Vec<String>,
}

/// A shard that returned an error during a federated query.
#[derive(Debug, Clone)]
pub struct FederatedShardError {
    /// Index name as registered via `add_index`.
    pub index: String,
    /// Rendered shard error (the underlying `SearchError` display).
    pub error: String,
}

/// Per-query record of which shards were dispatched and how each one ended.
///
/// Partial federation was previously invisible: shard failures and timeouts
/// were swallowed into `tracing::warn!` and a healthy-looking `Ok(hits)` came
/// back from whatever subset answered (bd-3zh67 HALF B). Callers that care
/// about recall must check [`FederatedCoverage::is_complete`].
#[derive(Debug, Clone, Default)]
pub struct FederatedCoverage {
    /// Shards dispatched for this query (bounded by `max_indices`), in
    /// registration order.
    pub queried: Vec<String>,
    /// Shards whose results are included in the fused output.
    pub answered: Vec<String>,
    /// Shards that returned an error.
    pub failed: Vec<FederatedShardError>,
    /// Shards that exceeded `per_index_timeout_ms`.
    pub timed_out: Vec<String>,
    /// Shards still in flight when the gather stopped early because
    /// `wait_for_indices` was satisfied.
    pub cancelled_in_flight: Vec<String>,
}

impl FederatedCoverage {
    /// True when every dispatched shard contributed results.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.answered.len() == self.queried.len()
    }
}

/// Fused hits plus the shard coverage that produced them.
#[derive(Debug, Clone)]
pub struct FederatedResponse {
    /// Globally fused, truncated-to-limit hits.
    pub hits: Vec<FederatedHit>,
    /// Which shards were queried / answered / failed / timed out / were
    /// cancelled for this query.
    pub coverage: FederatedCoverage,
}

/// Multi-index search orchestrator with scatter-gather fusion.
#[derive(Debug, Default)]
pub struct FederatedSearcher {
    indices: Vec<FederatedIndex>,
    config: FederatedConfig,
}

impl FederatedSearcher {
    /// Create a federated searcher with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the federated config.
    #[must_use]
    pub const fn with_config(mut self, config: FederatedConfig) -> Self {
        self.config = config;
        self
    }

    /// Add an index by name with a fusion weight.
    ///
    /// `weight <= 0.0` effectively disables score contribution from this index.
    /// Names must be unique among dispatched indices: they identify coverage,
    /// fusion membership and the primary source. Search rejects duplicates
    /// before starting any shard work.
    #[must_use]
    pub fn add_index(
        mut self,
        name: impl Into<String>,
        searcher: Arc<TwoTierSearcher>,
        weight: f32,
    ) -> Self {
        self.indices.push(FederatedIndex {
            name: name.into(),
            searcher,
            weight,
        });
        self
    }

    /// Number of configured indices.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true when no indices are configured.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn validate_shard_names(&self) -> SearchResult<()> {
        let mut names = AHashSet::with_capacity(self.indices.len().min(self.config.max_indices));
        for index in self.indices.iter().take(self.config.max_indices) {
            if !names.insert(index.name.as_str()) {
                return Err(SearchError::InvalidConfig {
                    field: "federated.indices".to_owned(),
                    value: "duplicate_name".to_owned(),
                    reason:
                        "dispatched index names must be unique for unambiguous fusion and coverage"
                            .to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Execute federated search using a shared document-text provider.
    ///
    /// `text_fn` resolves global document IDs for exclusion filtering and
    /// reranking. For separate per-index stores, use
    /// [`Self::search_with_index_text`] to receive the originating index name.
    ///
    /// # Errors
    ///
    /// Same as [`Self::search_with_index_text`].
    pub async fn search<F>(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        text_fn: F,
    ) -> SearchResult<FederatedResponse>
    where
        F: Fn(&str) -> Option<String> + Send + Sync,
    {
        self.search_with_index_text(cx, query, limit, |_index: &str, doc_id: &str| {
            text_fn(doc_id)
        })
        .await
    }

    /// Execute federated search with text resolved from each originating index.
    ///
    /// `text_fn(index_name, doc_id)` receives the exact name registered with
    /// [`Self::add_index`]. Each shard's exclusion filtering and optional
    /// reranking use its own bound provider, even when multiple indices return
    /// the same document ID. The callback may borrow request-scoped stores; it
    /// need not be `'static`. It is not called for undispatched indices.
    ///
    /// Document IDs still denote global fusion identity: equal IDs merge into
    /// one hit. Applications with unrelated index-local IDs must namespace IDs
    /// when indexing; selecting a text store does not change deduplication.
    ///
    /// The gather waits for every dispatched shard (each bounded by
    /// `per_index_timeout_ms`) unless `wait_for_indices` requests an early
    /// stop. `min_indices` is a floor validated after the gather.
    /// Contributions are accumulated in index-name order, making scores and
    /// cutoff membership independent of completion order for the same shard
    /// replies. Explicit early stopping can still change which shards reply.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` before dispatch for duplicate names.
    /// Returns `SearchError::Cancelled` when cancellation is requested via `cx`.
    /// Returns the first shard error when no shard completes successfully.
    /// Returns `SearchError::FederatedInsufficientResponses` when fewer than
    /// `min_indices` shards complete successfully.
    pub async fn search_with_index_text<F>(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        text_fn: F,
    ) -> SearchResult<FederatedResponse>
    where
        F: Fn(&str, &str) -> Option<String> + Send + Sync,
    {
        if query.is_empty() || limit == 0 || self.indices.is_empty() {
            return Ok(FederatedResponse {
                hits: Vec::new(),
                coverage: FederatedCoverage::default(),
            });
        }
        self.validate_shard_names()?;

        let candidate_pool_factor = self.config.candidate_pool_factor.max(1);
        let per_index_limit = limit.saturating_mul(candidate_pool_factor);
        let timeout_budget = Duration::from_millis(self.config.per_index_timeout_ms);
        let (shard_results, coverage) = self
            .collect_shard_results(cx, query, per_index_limit, timeout_budget, &text_fn)
            .await?;

        if shard_results.len() < self.config.min_indices {
            return Err(SearchError::FederatedInsufficientResponses {
                required: self.config.min_indices,
                received: shard_results.len(),
            });
        }

        let mut hits = match self.config.fusion_method {
            FederatedFusion::Rrf { k } => fuse_rrf(&shard_results, k),
            FederatedFusion::WeightedScore { normalization } => {
                fuse_weighted(&shard_results, normalization, false)
            }
            FederatedFusion::CombMnz { normalization } => {
                fuse_weighted(&shard_results, normalization, true)
            }
        };

        hits.truncate(limit);
        Ok(FederatedResponse { hits, coverage })
    }

    #[allow(clippy::too_many_lines)]
    async fn collect_shard_results<F>(
        &self,
        cx: &Cx,
        query: &str,
        per_index_limit: usize,
        timeout_budget: Duration,
        text_fn: &F,
    ) -> SearchResult<(Vec<ShardResult>, FederatedCoverage)>
    where
        F: Fn(&str, &str) -> Option<String> + Send + Sync,
    {
        let mut coverage = FederatedCoverage {
            queried: self
                .indices
                .iter()
                .take(self.config.max_indices)
                .map(|index| index.name.clone())
                .collect(),
            ..FederatedCoverage::default()
        };
        let mut pending: Vec<ShardFuture<'_>> = self
            .indices
            .iter()
            .take(self.config.max_indices)
            .map(|index| {
                let index_name = index.name.clone();
                let index_weight = index.weight;
                let searcher = Arc::clone(&index.searcher);
                Box::pin(async move {
                    let timeout_start = cx.now();
                    let response = {
                        // Bind before the shard starts: exclusion and rerank
                        // lookups must never hydrate from another index's store.
                        let shard_text = |doc_id: &str| text_fn(index_name.as_str(), doc_id);
                        let future = Box::pin(searcher.search_collect_with_text(
                            cx,
                            query,
                            per_index_limit,
                            shard_text,
                        ));
                        timeout(timeout_start, timeout_budget, future).await
                    };
                    match response {
                        Ok(Ok((hits, _metrics))) => ShardCompletion::Completed(ShardResult {
                            name: index_name,
                            weight: index_weight,
                            hits,
                        }),
                        Ok(Err(SearchError::Cancelled { phase, reason })) => {
                            ShardCompletion::Cancelled { phase, reason }
                        }
                        Ok(Err(error)) => ShardCompletion::Failed {
                            index: index_name,
                            error,
                        },
                        Err(_elapsed) => ShardCompletion::TimedOut { index: index_name },
                    }
                }) as ShardFuture<'_>
            })
            .collect();

        let mut shard_results = Vec::new();
        let mut first_shard_error: Option<SearchError> = None;
        while !pending.is_empty() {
            let ready_batch = poll_fn(|task_cx| {
                let mut ready = Vec::new();
                let mut idx = 0;
                while idx < pending.len() {
                    let poll = pending[idx].as_mut().poll(task_cx);
                    if let Poll::Ready(completion) = poll {
                        ready.push(completion);
                        drop(pending.swap_remove(idx));
                    } else {
                        idx += 1;
                    }
                }
                if ready.is_empty() {
                    Poll::Pending
                } else {
                    Poll::Ready(ready)
                }
            })
            .await;

            for completion in ready_batch {
                match completion {
                    ShardCompletion::Completed(shard) => {
                        debug!(
                            index = %shard.name,
                            hit_count = shard.hits.len(),
                            "federated shard search completed"
                        );
                        coverage.answered.push(shard.name.clone());
                        shard_results.push(shard);
                    }
                    ShardCompletion::Cancelled { phase, reason } => {
                        return Err(SearchError::Cancelled { phase, reason });
                    }
                    ShardCompletion::Failed { index, error } => {
                        warn!(
                            index = %index,
                            error = %error,
                            "federated shard search failed; continuing with remaining indices"
                        );
                        coverage.failed.push(FederatedShardError {
                            index,
                            error: error.to_string(),
                        });
                        if first_shard_error.is_none() {
                            first_shard_error = Some(error);
                        }
                    }
                    ShardCompletion::TimedOut { index } => {
                        warn!(
                            index = %index,
                            timeout_ms = self.config.per_index_timeout_ms,
                            "federated shard timed out; continuing with remaining indices"
                        );
                        coverage.timed_out.push(index);
                    }
                }
            }

            // bd-3zh67: the early stop is gated on `wait_for_indices` alone.
            // `min_indices` is a floor (validated by the caller after the
            // gather) and must never cancel in-flight shards.
            if self.config.wait_for_indices > 0
                && shard_results.len() >= self.config.wait_for_indices
            {
                break;
            }
        }

        // Anything dropped from `pending` on the early break was cancelled
        // in flight: dispatched, but neither answered nor failed nor timed out.
        if !pending.is_empty() {
            drop(pending);
            coverage.cancelled_in_flight = coverage
                .queried
                .iter()
                .filter(|name| {
                    !coverage.answered.iter().any(|n| n == *name)
                        && !coverage.failed.iter().any(|f| &f.index == *name)
                        && !coverage.timed_out.iter().any(|n| n == *name)
                })
                .cloned()
                .collect();
        }

        if shard_results.is_empty()
            && let Some(error) = first_shard_error
        {
            return Err(error);
        }

        Ok((shard_results, coverage))
    }
}

/// Floating-point addition is not associative. Sorting only the final hits
/// cannot repair a sum that changed with shard completion order, including a
/// changed top-k boundary. Keep every fusion method on one canonical traversal.
fn ordered_shards(shards: &[ShardResult]) -> Vec<&ShardResult> {
    let mut ordered: Vec<_> = shards.iter().collect();
    ordered.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    ordered
}

/// Intern the distinct shard names to their **sorted rank**. Returns the sorted
/// distinct names (indexable by id) and a `name → id` lookup. Because ids are the
/// sorted-name rank, sorting a doc's `appeared_in` ids ascending reproduces the
/// lexicographic name order the old `BTreeSet<String>` iteration gave — so the
/// public `FederatedHit::appeared_in` order is unchanged.
#[allow(clippy::cast_possible_truncation)]
fn intern_shard_names(shards: &[ShardResult]) -> (Vec<&str>, AHashMap<&str, u32>) {
    let mut names: Vec<&str> = shards.iter().map(|s| s.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    let name_to_id: AHashMap<&str, u32> = names
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i as u32))
        .collect();
    (names, name_to_id)
}

fn fuse_rrf(shards: &[ShardResult], k: f64) -> Vec<FederatedHit> {
    let k = sanitize_rrf_k(k);
    // aHash (not SipHash): the cross-shard merge keys on owned `doc_id` strings;
    // the DoS-resistant default hasher is the bottleneck here, and the sibling
    // single-node `rrf_fuse_with_graph` already uses `AHashMap`. Output is
    // unchanged — `into_ranked_hits` sorts by a strict total order (the final
    // `doc_id` tiebreak), so map drain order never reaches the result.
    let (names, name_to_id) = intern_shard_names(shards);
    let mut docs: AHashMap<String, AggregateDoc> = AHashMap::new();
    for shard in ordered_shards(shards) {
        // NaN.max(0.0) propagates NaN — guard explicitly.
        if !shard.weight.is_finite() || shard.weight <= 0.0 {
            continue;
        }
        let weight = shard.weight;
        let shard_id = name_to_id[shard.name.as_str()];

        for (rank, hit) in shard.hits.iter().enumerate() {
            let contribution = weight * rank_contribution(k, rank);
            accumulate_doc(
                &mut docs,
                hit,
                &shard.name,
                shard_id,
                rank,
                contribution,
                contribution,
            );
        }
    }

    into_ranked_hits(docs, &names, false)
}

fn fuse_weighted(
    shards: &[ShardResult],
    normalization: NormalizationMethod,
    apply_comb_mnz: bool,
) -> Vec<FederatedHit> {
    let (names, name_to_id) = intern_shard_names(shards);
    let mut docs: AHashMap<String, AggregateDoc> = AHashMap::new();

    for shard in ordered_shards(shards) {
        // NaN.max(0.0) propagates NaN — guard explicitly.
        if !shard.weight.is_finite() || shard.weight <= 0.0 || shard.hits.is_empty() {
            continue;
        }
        let weight = shard.weight;
        let shard_id = name_to_id[shard.name.as_str()];

        let raw_scores: Vec<f32> = shard.hits.iter().map(|hit| hit.score).collect();
        let normalized = normalize_scores_with_method(&raw_scores, normalization);
        for (rank, (hit, normalized_score)) in shard.hits.iter().zip(normalized).enumerate() {
            let contribution = weight * normalized_score.max(0.0);
            accumulate_doc(
                &mut docs,
                hit,
                &shard.name,
                shard_id,
                rank,
                contribution,
                contribution,
            );
        }
    }

    into_ranked_hits(docs, &names, apply_comb_mnz)
}

#[inline]
fn sanitize_rrf_k(k: f64) -> f64 {
    if k.is_finite() && k >= 0.0 {
        k
    } else {
        DEFAULT_FEDERATED_RRF_K
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn rank_contribution(k: f64, rank: usize) -> f32 {
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    (1.0 / (k + f64::from(rank_u32) + 1.0)) as f32
}

#[allow(clippy::too_many_arguments)]
fn accumulate_doc(
    docs: &mut AHashMap<String, AggregateDoc>,
    hit: &ScoredResult,
    shard_name: &str,
    shard_id: u32,
    rank: usize,
    contribution: f32,
    primary_signal: f32,
) {
    let entry = docs.entry(hit.doc_id.to_string()).or_insert_with(|| {
        let mut template = hit.clone();
        template.score = 0.0;
        AggregateDoc {
            template,
            primary_index: shard_name.to_owned(),
            primary_rank: rank,
            primary_contribution: primary_signal,
            fused_score: 0.0,
            appeared_in: Vec::new(),
        }
    });

    entry.fused_score += contribution;
    // Push the interned id (deduped once at output). A shard contributes a given
    // doc at most once, so duplicates only arise from a shard with a repeated
    // doc_id — `into_ranked_hits` sort+dedups, matching the old `BTreeSet` semantics.
    entry.appeared_in.push(shard_id);

    let update_primary = match primary_signal.total_cmp(&entry.primary_contribution) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => {
            rank < entry.primary_rank
                || (rank == entry.primary_rank && shard_name < entry.primary_index.as_str())
        }
    };
    if update_primary {
        shard_name.clone_into(&mut entry.primary_index);
        entry.primary_rank = rank;
        entry.primary_contribution = primary_signal;
        entry.template = hit.clone();
    }
}

#[allow(clippy::cast_precision_loss)]
fn into_ranked_hits(
    mut docs: AHashMap<String, AggregateDoc>,
    names: &[&str],
    apply_comb_mnz: bool,
) -> Vec<FederatedHit> {
    let mut output: Vec<FederatedHit> = docs
        .drain()
        .map(|(_, mut aggregate)| {
            // Dedup the interned ids once here (the old `BTreeSet` deduped on
            // insert). Ascending id order == lexicographic name order, so the
            // materialized `appeared_in` names are sorted, as before.
            aggregate.appeared_in.sort_unstable();
            aggregate.appeared_in.dedup();
            let appearance_count = aggregate.appeared_in.len();
            if apply_comb_mnz {
                aggregate.fused_score *= appearance_count as f32;
            }
            aggregate.template.score = aggregate.fused_score;
            if appearance_count > 1 {
                aggregate.template.source = ScoreSource::Hybrid;
            }
            let appeared_in: Vec<String> = aggregate
                .appeared_in
                .iter()
                .map(|&id| names[id as usize].to_owned())
                .collect();
            FederatedHit {
                result: aggregate.template,
                source_index: aggregate.primary_index,
                source_rank: aggregate.primary_rank,
                appeared_in,
            }
        })
        .collect();

    // Unstable sort (no scratch alloc): the comparator is a strict total order
    // ending in a `doc_id` tiebreak, and `doc_id` is the unique map key, so no two
    // entries compare Equal — identical output to a stable sort. Same bit-identical
    // sibling fix as `blend_two_tier` / `rrf_fuse` (this cross-shard path is niche,
    // not separately benched).
    output.sort_unstable_by(|left, right| {
        right
            .result
            .score
            .total_cmp(&left.result.score)
            .then(right.appeared_in.len().cmp(&left.appeared_in.len()))
            .then(left.source_rank.cmp(&right.source_rank))
            .then_with(|| left.result.doc_id.cmp(&right.result.doc_id))
    });

    output
}

#[cfg(test)]
mod tests;
