use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use compact_str::CompactString;
use serde::{Deserialize, Serialize};

mod coverage_codec;
mod coverage_pair;
mod coverage_scope;
mod coverage_tier;
mod query_embeddings;

pub use coverage_codec::retrieval_topology_fits_request;
pub use coverage_pair::{
    CrossTierUniverseEvidenceV1, DerivedRetrievalTopologyV1, RealizedTierCoverageV1,
    RetrievalIntentV1, TierCoveragePairV1, UntrustedDerivedRetrievalTopologyV1,
    UntrustedTierCoveragePairV1,
};
pub use coverage_scope::{
    CoverageBasisV1, CoverageCountsV1, CoverageRelationV1, CoverageScopeV1, CoverageTierV1,
    CoverageUnknownReasonV1, CoverageUnverifiedReasonV1, CoverageWitnessV1,
    CrossScopeCoverageEvidenceV1, TIER_COVERAGE_SCHEMA_VERSION_V1,
};
pub use coverage_tier::{
    TierCoverageRealizationV1, TierCoverageRequestV1, TierCoverageV1, TrustedTierCoverageContextV1,
    UntrustedTierCoverageV1,
};
pub use query_embeddings::{
    BoundQueryEmbedding, SearchCoverageV1, SpaceIdentityAdmission, TierQueryCoverageV1,
    TieredQueryEmbeddings,
};

/// Document identifier type.
///
/// `CompactString` (SSO) stores ids ≤24 bytes inline, so the per-query
/// `limit_all` `doc_id` clones (RRF/blend/resolve materialization) are a stack
/// memcpy instead of a heap alloc — 29.8× cheaper for short ids
/// (`doc_id_clone_sso` bench). Drop-in for `String` at read sites (`Deref<str>`,
/// `as_str`, `From`, `PartialEq`, `Hash`, `Ord`, serde).
pub type DocId = CompactString;

use crate::SearchError;
use crate::explanation::HitExplanation;
use crate::query_class::QueryClass;

// ---------------------------------------------------------------------------
// Document types
// ---------------------------------------------------------------------------

/// A document to be indexed for search.
///
/// This is the input type consumed by both vector indexing and lexical indexing.
/// It intentionally does NOT carry computed data (embeddings, BM25 scores) --
/// those are produced during indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexableDocument {
    /// Unique document identifier (caller-defined).
    pub id: String,
    /// Main searchable text content.
    pub content: String,
    /// Optional title (receives BM25 boost in lexical search).
    pub title: Option<String>,
    /// Extensible key-value metadata. Stored alongside results and available
    /// in `ScoredResult.metadata` after search.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl IndexableDocument {
    /// Creates a new document with the required fields.
    #[must_use]
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            title: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the optional title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Adds a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Search result types
// ---------------------------------------------------------------------------

/// A raw hit from vector similarity search.
///
/// Produced by the vector index before fusion. Scores are raw cosine similarity
/// values (not normalized), typically in the range \[-1.0, 1.0\].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    /// Positional index into the vector store (used for fast lookup).
    pub index: u32,
    /// Raw cosine similarity score.
    pub score: f32,
    /// Document identifier resolved from the index.
    pub doc_id: DocId,
}

impl VectorHit {
    /// Ordering by score descending with NaN-safe semantics.
    /// NaN sorts below all real values (treated as worst possible score).
    #[must_use]
    pub fn cmp_by_score(&self, other: &Self) -> std::cmp::Ordering {
        // Map NaN to NEG_INFINITY so it sorts last in descending order.
        let a = if self.score.is_nan() {
            f32::NEG_INFINITY
        } else {
            self.score
        };
        let b = if other.score.is_nan() {
            f32::NEG_INFINITY
        } else {
            other.score
        };
        // Descending: higher scores first.
        b.total_cmp(&a)
    }

    /// [`Self::cmp_by_score`] refined into a **total order** by breaking ties on `doc_id`.
    ///
    /// Phase-1 score-correction passes (hubness demotion, k-NN smoothing) rewrite scores and must
    /// re-sort, because the rank-based fusion operators take each hit's rank from its *position*
    /// in the slice. Sorting on score alone leaves tied hits in an order that depends on the
    /// upstream pool, so an identical query could fuse differently across runs; the `doc_id`
    /// tiebreak makes the corrected pool replayable.
    ///
    /// Inherits `cmp_by_score`'s NaN-last semantics. That is *not* what a bare
    /// `b.score.total_cmp(&a.score)` gives: IEEE 754 `totalOrder` ranks `+NaN` above `+inf`, so a
    /// descending `total_cmp` would sort a NaN score to rank 0 — exactly the doc a corrupted
    /// correction statistic would produce.
    #[must_use]
    pub fn cmp_rank(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp_by_score(other)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// The retrieval shape a search actually ran with.
///
/// Requested versus realized topology is first-class telemetry (bd-9xuj):
/// a caller that asked for full progressive search and silently received a
/// hash-candidate pool rescored by the quality model was the defining
/// failure this vocabulary exists to make impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case", tag = "topology")]
pub enum RetrievalTopology {
    /// No semantic arm ran; lexical results only.
    LexicalOnly,
    /// Non-semantic hash vectors, as an explicit test/control lane or a
    /// declared lexical degradation — never semantic availability.
    HashControl,
    /// Fast-tier retrieval only.
    FastOnly,
    /// Quality index as the primary retrieval arm.
    QualityOnly,
    /// Fast retrieval for latency plus direct quality retrieval unioned
    /// before fusion; never a mere rescoring of the fast pool.
    FullProgressive,
    /// Progressive retrieval where only part of the doc set has
    /// quality-tier coverage.
    PartialQuality {
        /// Fraction of live documents with quality-tier embeddings, in
        /// parts per million for exact serialization.
        coverage_ppm: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RetrievalTopologyTagWireV1 {
    LexicalOnly,
    HashControl,
    FastOnly,
    QualityOnly,
    FullProgressive,
    PartialQuality,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetrievalTopologyWireV1 {
    topology: RetrievalTopologyTagWireV1,
    #[serde(default)]
    coverage_ppm: RetrievalTopologyFieldV1<u32>,
}

#[derive(Default)]
enum RetrievalTopologyFieldV1<T> {
    #[default]
    Missing,
    Present(T),
}

impl<'de, T> Deserialize<'de> for RetrievalTopologyFieldV1<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Self::Present)
    }
}

impl<'de> Deserialize<'de> for RetrievalTopology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RetrievalTopologyWireV1::deserialize(deserializer)?;
        match (wire.topology, wire.coverage_ppm) {
            (RetrievalTopologyTagWireV1::LexicalOnly, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::LexicalOnly)
            }
            (RetrievalTopologyTagWireV1::HashControl, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::HashControl)
            }
            (RetrievalTopologyTagWireV1::FastOnly, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::FastOnly)
            }
            (RetrievalTopologyTagWireV1::QualityOnly, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::QualityOnly)
            }
            (RetrievalTopologyTagWireV1::FullProgressive, RetrievalTopologyFieldV1::Missing) => {
                Ok(Self::FullProgressive)
            }
            (
                RetrievalTopologyTagWireV1::PartialQuality,
                RetrievalTopologyFieldV1::Present(coverage_ppm),
            ) => Ok(Self::PartialQuality { coverage_ppm }),
            (RetrievalTopologyTagWireV1::PartialQuality, RetrievalTopologyFieldV1::Missing) => {
                Err(serde::de::Error::missing_field("coverage_ppm"))
            }
            (_, RetrievalTopologyFieldV1::Present(_)) => Err(serde::de::Error::custom(
                "coverage_ppm is valid only for partial_quality",
            )),
        }
    }
}

impl RetrievalTopology {
    /// Stable `snake_case` code for logs and machine payloads.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::LexicalOnly => "lexical_only",
            Self::HashControl => "hash_control",
            Self::FastOnly => "fast_only",
            Self::QualityOnly => "quality_only",
            Self::FullProgressive => "full_progressive",
            Self::PartialQuality { .. } => "partial_quality",
        }
    }

    /// True when semantic vectors contribute to results. `HashControl` is
    /// deliberately non-semantic.
    #[must_use]
    pub const fn is_semantic(&self) -> bool {
        matches!(
            self,
            Self::FastOnly
                | Self::QualityOnly
                | Self::FullProgressive
                | Self::PartialQuality { .. }
        )
    }
}

/// A hit from hybrid fusion (lexical + semantic combined via RRF).
///
/// RRF scores are computed in f64 for precision during accumulation of many
/// small `1/(K+rank+1)` values, then carried as f64 throughout fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedHit {
    /// Document identifier.
    pub doc_id: DocId,
    /// RRF-fused score (f64 for precision during fusion).
    pub rrf_score: f64,
    /// Rank in the lexical (BM25) source, if present.
    pub lexical_rank: Option<usize>,
    /// Rank in the semantic (vector) source, if present.
    ///
    /// Omitted when absent so a remapped hash-control hit does not publish a
    /// `semantic_rank` key. Hash ranks live on [`Self::hash_rank`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_rank: Option<usize>,
    /// Rank in the hash-control vector list. Never set together with
    /// `semantic_rank`: a hash generation is not a semantic source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_rank: Option<usize>,
    /// Internal vector index, if present.
    pub semantic_index: Option<u32>,
    /// Raw BM25 score from lexical search, if applicable.
    pub lexical_score: Option<f32>,
    /// Raw cosine similarity from semantic search, if applicable.
    ///
    /// Omitted when absent so a remapped hash-control hit does not publish a
    /// `semantic_score` key. Hash scores live on [`Self::hash_score`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_score: Option<f32>,
    /// Vector score from a hash-control generation. Never set together with
    /// `semantic_score`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_score: Option<f32>,
    /// True if this document appeared in both lexical and semantic results.
    pub in_both_sources: bool,
}

impl FusedHit {
    /// Move vector ranks and scores off semantic fields when this hit is
    /// hash control.
    pub fn remap_hash_control_ranks(&mut self) {
        if self.hash_rank.is_none() {
            self.hash_rank = self.semantic_rank.take();
        } else {
            self.semantic_rank = None;
        }
        if self.hash_score.is_none() {
            self.hash_score = self.semantic_score.take();
        } else {
            self.semantic_score = None;
        }
    }

    /// Four-level deterministic tie-breaking for RRF results:
    /// 1. Higher RRF score first
    /// 2. Documents in both sources preferred
    /// 3. Higher lexical score preferred
    /// 4. Lexicographic doc\_id (deterministic fallback)
    #[must_use]
    pub fn cmp_for_ranking(&self, other: &Self) -> std::cmp::Ordering {
        // 1. RRF score descending
        other
            .rrf_score
            .total_cmp(&self.rrf_score)
            // 2. in_both_sources preferred (true > false)
            .then(other.in_both_sources.cmp(&self.in_both_sources))
            // 3. Lexical score descending (treat None as -inf)
            .then_with(|| {
                let a = self.lexical_score.unwrap_or(f32::NEG_INFINITY);
                let b = other.lexical_score.unwrap_or(f32::NEG_INFINITY);
                b.total_cmp(&a)
            })
            // 4. doc_id ascending (deterministic)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// Which search backend produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreSource {
    /// Lexical (BM25) search only.
    Lexical,
    /// Fast-tier semantic search only.
    SemanticFast,
    /// Quality-tier semantic search only.
    SemanticQuality,
    /// Hybrid fusion (lexical + vector via RRF).
    ///
    /// Vector here may be semantic or a hash-control generation. Hash-only
    /// vector hits use [`Self::HashControl`], not this variant.
    Hybrid,
    /// Result was reranked by cross-encoder.
    Reranked,
    /// Hash / FNV / JL control vector search. Not semantic.
    HashControl,
}

/// True when `embedder_id` is a hash/fnv/jl control identity, not a semantic model.
#[must_use]
pub fn is_hash_generation_id(embedder_id: &str) -> bool {
    let id = embedder_id.to_ascii_lowercase();
    id == "hash"
        || id.starts_with("hash-")
        || id.starts_with("hash/")
        || id.starts_with("fnv1a-")
        || id.starts_with("jl-")
}

/// The final scored search result delivered to consumers.
///
/// Intentionally does NOT carry document text. Text is expensive and most
/// consumers only need `doc_id` + scores. When text is needed (e.g., for
/// reranking or display), look it up from your document store via `doc_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    /// Unique document identifier.
    pub doc_id: DocId,
    /// Primary relevance score (RRF or blended, truncated to f32).
    pub score: f32,
    /// Which search backend produced this result.
    pub source: ScoreSource,
    /// Internal vector index, if applicable.
    pub index: Option<u32>,
    /// Score from fast-tier semantic search, if applicable.
    pub fast_score: Option<f32>,
    /// Score from quality-tier semantic search, if applicable.
    pub quality_score: Option<f32>,
    /// BM25 score from lexical search, if applicable.
    pub lexical_score: Option<f32>,
    /// Cross-encoder score from reranking, if applicable.
    pub rerank_score: Option<f32>,
    /// Detailed explanation of scoring (if enabled). `Box`ed so the (usually
    /// `None`, `explain=false`) common case doesn't carry ~88 B of inline
    /// `HitExplanation` in every `ScoredResult` — halves the struct (168→88 B),
    /// speeding `limit_all` materialization + phase clones and halving result-set
    /// memory (`scoredresult_box_ab` bench). `Box<HitExplanation>` derefs to
    /// `HitExplanation`; the heap alloc happens only when `explain=true`.
    pub explanation: Option<Box<HitExplanation>>,
    /// Arbitrary document metadata (from index stored fields). `Arc`-wrapped so
    /// the per-winner metadata materialization at `limit_all` is a refcount bump,
    /// not a deep `Value` clone (map + strings + arrays re-allocated) — measured
    /// 200–278× cheaper for realistic metadata (`metadata_clone_ab` bench).
    /// `Arc<Value>` derefs to `Value`, so reads (`.get`, `.as_object`, filters via
    /// `.as_deref()`) are unchanged.
    pub metadata: Option<Arc<serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// Search mode and phases
// ---------------------------------------------------------------------------

/// Search mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchMode {
    /// BM25 keyword matching only.
    Lexical,
    /// Embedding similarity only.
    Semantic,
    /// RRF fusion of lexical + semantic.
    Hybrid,
    /// Progressive: fast semantic -> quality refinement + lexical fusion.
    TwoTier,
}

/// Diagnostic metrics for a search phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    /// Which embedder was used for this phase.
    pub embedder_id: String,
    /// Number of vectors searched.
    pub vectors_searched: usize,
    /// Number of lexical candidates retrieved.
    pub lexical_candidates: usize,
    /// Number of results after fusion.
    pub fused_count: usize,
    /// Why this phase skipped semantic work, when it did.
    ///
    /// Absent on a healthy semantic Initial. Present when results are
    /// lexical-only or otherwise degraded (hash embedder, missing quality
    /// index, circuit breaker, …). Callers that treat Initial as "semantic
    /// hits" must inspect this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
    /// Number of hash/fnv/JL control-vector candidates retrieved.
    ///
    /// Hash hits are not semantic. Absent from older payloads; deserializes
    /// as `0`. Present on Initial when the fast lane is a hash-control
    /// generation.
    #[serde(default, skip_serializing_if = "usize_is_zero")]
    pub hash_control_candidates: usize,
}

/// Structured telemetry for a completed search request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Which search mode ran.
    pub mode: SearchMode,
    /// Query class selected by the classifier, when available.
    pub query_class: Option<QueryClass>,
    /// End-to-end latency for the search request.
    pub total_latency_ms: f64,
    /// Latency for phase 1 (`Initial`) when available.
    pub phase1_latency_ms: Option<f64>,
    /// Latency for phase 2 (`Refined`) when available.
    pub phase2_latency_ms: Option<f64>,
    /// Number of results returned to the caller.
    pub result_count: usize,
    /// Number of lexical candidates retrieved.
    pub lexical_candidates: usize,
    /// Number of semantic candidates retrieved.
    ///
    /// Hash/fnv/JL control hits are counted in [`Self::hash_control_candidates`],
    /// not here.
    pub semantic_candidates: usize,
    /// Number of hash/fnv/JL control-vector candidates retrieved.
    ///
    /// Absent from older payloads; deserializes as `0`.
    #[serde(default, skip_serializing_if = "usize_is_zero")]
    pub hash_control_candidates: usize,
    /// Whether quality refinement was applied.
    pub refined: bool,
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if requires fn(&Field)"
)]
const fn usize_is_zero(value: &usize) -> bool {
    *value == 0
}

/// Structured telemetry for an embedding operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetrics {
    /// Stable embedder identifier (for example, `potion-multilingual-128M`).
    pub embedder_id: String,
    /// Number of texts embedded in this operation.
    pub batch_size: usize,
    /// Embedding operation latency.
    pub duration_ms: f64,
    /// Embedding vector dimension.
    pub dimension: usize,
    /// Whether this embedder is semantically meaningful.
    pub is_semantic: bool,
}

/// Structured telemetry for index update operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetrics {
    /// Document count after the update.
    pub doc_count: usize,
    /// Total on-disk index size after the update.
    pub index_size_bytes: u64,
    /// Number of documents added or modified by this update.
    pub updated_docs: usize,
    /// Whether the index was marked stale during this update.
    pub staleness_detected: bool,
}

/// Tracks how rankings changed between initial and refined phases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankChanges {
    /// Documents that moved up in ranking after refinement.
    pub promoted: usize,
    /// Documents that moved down in ranking after refinement.
    pub demoted: usize,
    /// Documents whose rank did not change.
    pub stable: usize,
}

impl RankChanges {
    /// Total number of documents tracked.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.promoted
            .saturating_add(self.demoted)
            .saturating_add(self.stable)
    }
}

/// Progressive search phases for three-tier display.
///
/// The iterator contract:
/// 1. Always yields `Initial` first (~15ms).
/// 2. Then yields either `Refined` or `RefinementFailed` (never both).
/// 3. Optionally yields `Reranked` if a cross-encoder is configured.
/// 4. Iterator is fused after yielding final phase (`next()` returns `None`).
///
/// Consumers can stop after `Initial` if latency-sensitive.
///
/// # Example
///
/// ```rust,ignore
/// for phase in searcher.search("distributed consensus", 10) {
///     match phase {
///         SearchPhase::Initial { results, .. } => display_immediately(&results),
///         SearchPhase::Refined { results, .. } => update_display(&results),
///         SearchPhase::Reranked { results, .. } => update_display_with_final_scores(&results),
///         SearchPhase::RefinementFailed { initial_results, error, .. } => {
///             // Keep showing initial results, log the error
///             log_warning(&error);
///         }
///     }
/// }
/// ```
#[derive(Debug)]
pub enum SearchPhase {
    /// Fast-tier results ready for immediate display.
    ///
    /// Contains RRF-fused results from fast embedding + BM25.
    /// Scores are RRF values (~0.01-0.03 range), sorted descending
    /// with deterministic tie-breaking.
    Initial {
        /// Fast-tier search results.
        results: Vec<ScoredResult>,
        /// Time elapsed for this phase.
        latency: Duration,
        /// Diagnostic metrics for this phase.
        metrics: PhaseMetrics,
    },

    /// Quality-refined results ready to replace initial display.
    ///
    /// Contains blended scores (0.7 quality + 0.3 fast by default).
    /// Results may have different ordering than `Initial`.
    Refined {
        /// Quality-refined search results.
        results: Vec<ScoredResult>,
        /// Time elapsed for this phase.
        latency: Duration,
        /// Diagnostic metrics for this phase.
        metrics: PhaseMetrics,
        /// How rankings changed compared to Initial.
        rank_changes: RankChanges,
    },

    /// Final reranked results from cross-encoder inference.
    ///
    /// Contains scores produced by a cross-encoder model processing
    /// (query, doc) pairs directly. Most accurate but slowest phase.
    Reranked {
        /// Reranked search results.
        results: Vec<ScoredResult>,
        /// Time elapsed for this phase (including earlier refinement).
        latency: Duration,
        /// Diagnostic metrics for this phase.
        metrics: PhaseMetrics,
    },

    /// Quality refinement failed; initial results remain valid.
    ///
    /// This is NOT an error state -- it is graceful degradation.
    /// The consumer should display `initial_results` and log the error.
    RefinementFailed {
        /// The original `Initial` results, carried forward unchanged.
        initial_results: Vec<ScoredResult>,
        /// Why refinement failed (timeout, model error, etc.).
        error: SearchError,
        /// How long we waited before failing.
        latency: Duration,
    },
}

#[cfg(test)]
mod tests_coverage_algebra;
#[cfg(test)]
mod tests_embeddings_and_topology;
#[cfg(test)]
mod tests_results_and_metrics;
#[cfg(test)]
mod tests_support;
