use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    SHADOW_ARTIFACT_DIRECTORY, SHADOW_DIVERGENCE_SCHEMA_VERSION, SHADOW_SAMPLE_DENOMINATOR,
};
use crate::{IndexableDocument, ScoredResult, SearchError, SearchResult};

const CORPUS_HASH_DOMAIN: &[u8] = b"frankensearch/shadow-corpus/v1\0";

/// Runtime policy for one shadow wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowLexicalConfig {
    /// Explicit opt-in. The default is off.
    pub enabled: bool,
    /// Allow shadow polls on the serving scheduler when no blocking pool is
    /// attached to the caller's context. This can block serving and is intended
    /// for explicit deterministic execution, such as `LabRuntime` comparisons.
    /// The default requires caller-owned blocking capacity and sheds otherwise.
    pub allow_inline_shadow: bool,
    /// Deterministic sample rate in basis points.
    pub sample_rate_basis_points: u16,
    /// Maximum number of admitted background comparisons.
    pub max_in_flight: usize,
    /// Maximum accepted absolute score delta before score divergence.
    pub score_epsilon: f32,
    /// Evidence directory, normally `<index>/.quill-shadow`.
    pub artifact_directory: PathBuf,
    /// Initial committed manifest generation.
    pub initial_generation: u64,
}

impl ShadowLexicalConfig {
    /// Construct a default-off policy rooted beneath `index_root`.
    #[must_use]
    pub fn for_index_root(index_root: impl AsRef<Path>) -> Self {
        Self {
            artifact_directory: index_root.as_ref().join(SHADOW_ARTIFACT_DIRECTORY),
            ..Self::default()
        }
    }

    pub(super) fn validate(&self) -> SearchResult<()> {
        if u64::from(self.sample_rate_basis_points) > SHADOW_SAMPLE_DENOMINATOR {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_sample_rate_basis_points".to_owned(),
                value: self.sample_rate_basis_points.to_string(),
                reason: format!("expected 0..={SHADOW_SAMPLE_DENOMINATOR}"),
            });
        }
        if self.max_in_flight == 0 {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_max_in_flight".to_owned(),
                value: self.max_in_flight.to_string(),
                reason: "expected at least one background task".to_owned(),
            });
        }
        if !self.score_epsilon.is_finite() || self.score_epsilon < 0.0 {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_score_epsilon".to_owned(),
                value: self.score_epsilon.to_string(),
                reason: "expected a finite non-negative value".to_owned(),
            });
        }
        if self.artifact_directory.as_os_str().is_empty() {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_artifact_directory".to_owned(),
                value: String::new(),
                reason: "expected a non-empty path".to_owned(),
            });
        }
        Ok(())
    }
}

impl Default for ShadowLexicalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allow_inline_shadow: false,
            sample_rate_basis_points: 10_000,
            max_in_flight: 2,
            score_epsilon: 1.0e-5,
            artifact_directory: PathBuf::from(SHADOW_ARTIFACT_DIRECTORY),
            initial_generation: 0,
        }
    }
}

/// Reason a sampled shadow comparison was shed before admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowShedReason {
    /// Host CPU pressure crossed the configured ceiling.
    CpuPressure,
    /// Host I/O pressure crossed the configured ceiling.
    IoPressure,
    /// An operator or lifecycle controller paused comparisons.
    ManualPause,
    /// The bounded background-task gate was full.
    Capacity,
    /// The shadow index is known to be incomplete or failed.
    ShadowDegraded,
    /// The caller did not provide a spawn-capable runtime context.
    RuntimeUnavailable,
    /// The caller did not provide blocking capacity or opt into inline polls.
    BlockingPoolUnavailable,
    /// Index generation changed while the comparison ran.
    GenerationDrift,
}

impl ShadowShedReason {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::CpuPressure => "cpu_pressure",
            Self::IoPressure => "io_pressure",
            Self::ManualPause => "manual_pause",
            Self::Capacity => "capacity",
            Self::ShadowDegraded => "shadow_degraded",
            Self::RuntimeUnavailable => "runtime_unavailable",
            Self::BlockingPoolUnavailable => "blocking_pool_unavailable",
            Self::GenerationDrift => "generation_drift",
        }
    }
}

/// Synchronous, allocation-free load-shed probe used before task admission.
pub trait ShadowLoadProbe: Send + Sync {
    /// Return a shed reason, or `None` when shadow work may be admitted.
    fn shed_reason(&self) -> Option<ShadowShedReason>;
}

/// Probe that admits every sampled comparison.
#[derive(Debug, Default)]
pub struct AlwaysAdmitShadowLoad;

impl ShadowLoadProbe for AlwaysAdmitShadowLoad {
    fn shed_reason(&self) -> Option<ShadowShedReason> {
        None
    }
}

/// Mutable probe for host pressure controllers and deterministic tests.
#[derive(Debug, Default)]
pub struct AtomicShadowLoadProbe {
    cpu_pressure: AtomicBool,
    io_pressure: AtomicBool,
    manually_paused: AtomicBool,
}

impl AtomicShadowLoadProbe {
    /// Update the CPU pressure gate.
    pub fn set_cpu_pressure(&self, pressured: bool) {
        self.cpu_pressure.store(pressured, Ordering::Release);
    }

    /// Update the I/O pressure gate.
    pub fn set_io_pressure(&self, pressured: bool) {
        self.io_pressure.store(pressured, Ordering::Release);
    }

    /// Pause or resume shadow admission without affecting serving.
    pub fn set_manually_paused(&self, paused: bool) {
        self.manually_paused.store(paused, Ordering::Release);
    }
}

impl ShadowLoadProbe for AtomicShadowLoadProbe {
    fn shed_reason(&self) -> Option<ShadowShedReason> {
        if self.manually_paused.load(Ordering::Acquire) {
            Some(ShadowShedReason::ManualPause)
        } else if self.cpu_pressure.load(Ordering::Acquire) {
            Some(ShadowShedReason::CpuPressure)
        } else if self.io_pressure.load(Ordering::Acquire) {
            Some(ShadowShedReason::IoPressure)
        } else {
            None
        }
    }
}

/// Engine-neutral document carried by an exact-replay shadow artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowDocument {
    /// Stable document identifier.
    pub id: String,
    /// Searchable content.
    pub content: String,
    /// Optional boosted title.
    pub title: Option<String>,
    /// Canonically ordered metadata.
    pub metadata: BTreeMap<String, String>,
}

impl From<&IndexableDocument> for ShadowDocument {
    fn from(document: &IndexableDocument) -> Self {
        Self {
            id: document.id.clone(),
            content: document.content.clone(),
            title: document.title.clone(),
            metadata: document
                .metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }
    }
}

impl From<ShadowDocument> for IndexableDocument {
    fn from(document: ShadowDocument) -> Self {
        Self {
            id: document.id,
            content: document.content,
            title: document.title,
            metadata: document.metadata.into_iter().collect(),
        }
    }
}

/// Query identity recorded for shrink and exact-generation replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowQuery {
    /// Raw query text supplied to both engines.
    pub text: String,
    /// Requested top-k limit.
    pub limit: usize,
    /// Whether the fusion-candidate path was exercised.
    pub fusion_candidates: bool,
}

/// One top-k result with exact floating-point witnesses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowRankedHit {
    /// One-based rank.
    pub rank: usize,
    /// Stable document identifier.
    pub document_id: String,
    /// Exact result score bits.
    pub score_bits: u32,
    /// Exact lexical score bits, when supplied separately.
    pub lexical_score_bits: Option<u32>,
}

impl ShadowRankedHit {
    fn from_result(rank: usize, result: &ScoredResult) -> Self {
        Self {
            rank,
            document_id: result.doc_id.to_string(),
            score_bits: result.score.to_bits(),
            lexical_score_bits: result.lexical_score.map(f32::to_bits),
        }
    }

    fn score(&self) -> f32 {
        f32::from_bits(self.score_bits)
    }
}

/// Stable classification emitted by the production shadow comparator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowDivergenceClass {
    /// Identical document order and exact score bits.
    Exact,
    /// Identical order with score deltas inside the configured epsilon.
    ScoreEpsilon,
    /// The same document set was returned in a different order.
    TieOrder,
    /// Document identities or result cardinality differed.
    RankMismatch,
    /// Document order matched but a score exceeded the epsilon.
    ScoreMismatch,
}

impl ShadowDivergenceClass {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ScoreEpsilon => "score_epsilon",
            Self::TieOrder => "tie_order",
            Self::RankMismatch => "rank_mismatch",
            Self::ScoreMismatch => "score_mismatch",
        }
    }

    pub(super) const fn is_artifact(self) -> bool {
        !matches!(self, Self::Exact)
    }
}

/// Versioned production record written to
/// `.quill-shadow/divergences.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowDivergenceRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Committed manifest generation observed by both queries.
    pub manifest_generation: u64,
    /// SHA-256 identity of the replay corpus.
    pub corpus_hash: String,
    /// Full corpus required by the shrinker and exact replay.
    pub documents: Vec<ShadowDocument>,
    /// Query supplied to both engines.
    pub query: ShadowQuery,
    /// Untouched serving top-k list.
    pub serving_top_k: Vec<ShadowRankedHit>,
    /// Background shadow top-k list.
    pub shadow_top_k: Vec<ShadowRankedHit>,
    /// Classified comparison result.
    pub classification: ShadowDivergenceClass,
    /// Serving latency measured before shadow admission.
    pub serve_latency_micros: u64,
    /// Background shadow latency.
    pub shadow_latency_micros: u64,
}

impl ShadowDivergenceRecord {
    /// Validate the versioned artifact schema and replay invariants.
    ///
    /// # Errors
    ///
    /// Returns a stable reason for an unsupported version, malformed corpus
    /// hash, duplicate document identity, invalid rank sequence, non-finite
    /// score, or an `exact` record in the divergence-only stream.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SHADOW_DIVERGENCE_SCHEMA_VERSION {
            return Err(format!("unsupported shadow schema {}", self.schema_version));
        }
        if self.corpus_hash.len() != 64
            || !self
                .corpus_hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("corpus_hash must be 64 hexadecimal characters".to_owned());
        }
        if self.documents.is_empty() {
            return Err("documents must not be empty".to_owned());
        }
        let mut document_ids = BTreeSet::new();
        for document in &self.documents {
            if document.id.is_empty() {
                return Err("documents contain an empty id".to_owned());
            }
            if !document_ids.insert(document.id.as_str()) {
                return Err(format!("duplicate document id {:?}", document.id));
            }
        }
        if self
            .documents
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err("documents must be ordered by ascending id".to_owned());
        }
        if self.corpus_hash != compute_shadow_corpus_hash(&self.documents) {
            return Err("corpus_hash does not match documents".to_owned());
        }
        validate_ranked_hits("serving_top_k", &self.serving_top_k)?;
        validate_ranked_hits("shadow_top_k", &self.shadow_top_k)?;
        if !self.classification.is_artifact() {
            return Err("divergence stream cannot contain exact comparisons".to_owned());
        }
        Ok(())
    }
}

fn validate_ranked_hits(label: &str, hits: &[ShadowRankedHit]) -> Result<(), String> {
    let mut identities = BTreeSet::new();
    for (index, hit) in hits.iter().enumerate() {
        if hit.rank != index.saturating_add(1) {
            return Err(format!("{label} ranks must be contiguous and one-based"));
        }
        if hit.document_id.is_empty() || !identities.insert(hit.document_id.as_str()) {
            return Err(format!(
                "{label} contains an empty or duplicate document id"
            ));
        }
        if !hit.score().is_finite()
            || hit
                .lexical_score_bits
                .is_some_and(|bits| !f32::from_bits(bits).is_finite())
        {
            return Err(format!("{label} contains a non-finite score"));
        }
    }
    Ok(())
}

/// Typed shadow failure written separately from classified divergences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowDegradationKind {
    /// The configured shadow oracle could not be found or opened.
    OracleUnavailable,
    /// The oracle corpus could not be exported for exact replay.
    CorpusExport,
    /// Shadow document indexing failed.
    Index,
    /// Shadow commit failed after the serving commit succeeded.
    Commit,
    /// Background shadow search returned an error.
    Search,
    /// No spawn-capable runtime was attached to the request context.
    RuntimeUnavailable,
    /// Shadow search requires caller-owned blocking capacity or inline opt-in.
    BlockingPoolUnavailable,
    /// A divergence artifact could not be persisted.
    ArtifactWrite,
    /// Generation or corpus revision changed during comparison.
    GenerationDrift,
    /// Host pressure sampling failed; cached admission state remains in use.
    PressureSample,
}

/// Append-only, doctor-readable shadow degradation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowDegradationRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Committed generation at the failure boundary.
    pub manifest_generation: u64,
    /// Typed failure class.
    pub kind: ShadowDegradationKind,
    /// Stable human-readable detail.
    pub detail: String,
}

/// One completed comparison, including clean observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowObservationRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Committed generation shared by both engines.
    pub manifest_generation: u64,
    /// Raw query text.
    pub query: String,
    /// Classified result.
    pub classification: ShadowDivergenceClass,
    /// Serving latency measured before background admission.
    pub serve_latency_micros: u64,
    /// Background shadow latency.
    pub shadow_latency_micros: u64,
}

/// In-process counters for sampling, load shedding, and lifecycle proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowStatus {
    /// Current opt-in state.
    pub enabled: bool,
    /// Current committed generation.
    pub manifest_generation: u64,
    /// Current deterministic corpus hash.
    pub corpus_hash: String,
    /// Currently admitted background comparisons.
    pub in_flight: usize,
    /// Queries selected by the deterministic sampler.
    pub sampled: u64,
    /// Queries rejected by the deterministic sampler.
    pub sampled_out: u64,
    /// Sampled queries shed by pressure, capacity, or lifecycle state.
    pub shed: u64,
    /// Completed background comparisons.
    pub completed: u64,
    /// Classified divergence artifacts attempted.
    pub divergences: u64,
    /// Shadow errors or lifecycle degradations.
    pub degradations: u64,
}

pub(super) fn ranked_hits(results: &[ScoredResult]) -> Vec<ShadowRankedHit> {
    results
        .iter()
        .enumerate()
        .map(|(index, result)| ShadowRankedHit::from_result(index.saturating_add(1), result))
        .collect()
}

pub(super) fn classify_ranked_hits(
    serving: &[ShadowRankedHit],
    shadow: &[ShadowRankedHit],
    score_epsilon: f32,
) -> ShadowDivergenceClass {
    if serving.len() != shadow.len() {
        return ShadowDivergenceClass::RankMismatch;
    }
    let same_order = serving
        .iter()
        .zip(shadow)
        .all(|(left, right)| left.document_id == right.document_id);
    if same_order {
        let exact_scores = serving.iter().zip(shadow).all(|(left, right)| {
            left.score_bits == right.score_bits
                && left.lexical_score_bits == right.lexical_score_bits
        });
        if exact_scores {
            return ShadowDivergenceClass::Exact;
        }
        let within_epsilon = serving.iter().zip(shadow).all(|(left, right)| {
            score_delta_within(left.score(), right.score(), score_epsilon)
                && optional_score_delta_within(
                    left.lexical_score_bits,
                    right.lexical_score_bits,
                    score_epsilon,
                )
        });
        return if within_epsilon {
            ShadowDivergenceClass::ScoreEpsilon
        } else {
            ShadowDivergenceClass::ScoreMismatch
        };
    }

    let serving_ids = serving
        .iter()
        .map(|hit| hit.document_id.as_str())
        .collect::<BTreeSet<_>>();
    let shadow_ids = shadow
        .iter()
        .map(|hit| hit.document_id.as_str())
        .collect::<BTreeSet<_>>();
    if serving_ids == shadow_ids {
        ShadowDivergenceClass::TieOrder
    } else {
        ShadowDivergenceClass::RankMismatch
    }
}

fn score_delta_within(left: f32, right: f32, epsilon: f32) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= epsilon
}

fn optional_score_delta_within(left: Option<u32>, right: Option<u32>, epsilon: f32) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            score_delta_within(f32::from_bits(left), f32::from_bits(right), epsilon)
        }
        _ => false,
    }
}

/// Compute the canonical SHA-256 replay identity for an id-ordered corpus.
#[must_use]
pub fn compute_shadow_corpus_hash(documents: &[ShadowDocument]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CORPUS_HASH_DOMAIN);
    for document in documents {
        hash_bytes(&mut hasher, document.id.as_bytes());
        hash_bytes(&mut hasher, document.content.as_bytes());
        match document.title.as_deref() {
            Some(title) => {
                hasher.update([1]);
                hash_bytes(&mut hasher, title.as_bytes());
            }
            None => hasher.update([0]),
        }
        hasher.update(
            u64::try_from(document.metadata.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for (key, value) in &document.metadata {
            hash_bytes(&mut hasher, key.as_bytes());
            hash_bytes(&mut hasher, value.as_bytes());
        }
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len().saturating_mul(2));
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing into String cannot fail");
    }
    encoded
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(bytes);
}

pub(super) fn micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
