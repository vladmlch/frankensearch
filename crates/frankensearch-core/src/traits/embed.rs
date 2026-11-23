use std::fmt;

use asupersync::Cx;
use serde::{Deserialize, Serialize};

use super::{SearchFuture, embedding_checkpoint};
use crate::error::{SearchError, SearchResult};
use crate::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};

fn bounded_embedder_diagnostic_id(id: &str) -> String {
    if !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        id.to_owned()
    } else {
        "<redacted-embedder-id>".to_owned()
    }
}

/// Vector output bound to the complete space, producer, input, and storage contracts.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityBoundEmbedding {
    /// Raw f32 vector values.
    pub values: Vec<f32>,
    /// Complete validated identity bundle used to produce the vector.
    pub identity: EmbeddingIdentityBundleV1,
}

impl fmt::Debug for IdentityBoundEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IdentityBoundEmbedding")
            .field("dimension", &self.values.len())
            .field("identity", &self.identity.fingerprint())
            .finish_non_exhaustive()
    }
}

impl IdentityBoundEmbedding {
    /// Validate the identity, exact vector dimension, and finite coordinates.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the identity is malformed, the vector
    /// length disagrees with its space, or any coordinate is NaN or infinite.
    /// Valid coordinates are never normalized, repaired, or otherwise changed.
    pub fn validate(&self) -> SearchResult<()> {
        let dimension = validate_embedding_identity(&self.identity)?;
        validate_embedding_values(&self.values, dimension)
    }
}

// Keep identity checks separate from per-row checks: a batch shares one
// immutable identity, which must be admitted even when the batch is empty.
fn validate_embedding_identity(identity: &EmbeddingIdentityBundleV1) -> SearchResult<usize> {
    identity.validate()?;
    let dimension =
        usize::try_from(identity.space.dimension).map_err(|_| SearchError::InvalidConfig {
            field: "identity_bound_embedding.dimension".to_owned(),
            value: identity.space.dimension.to_string(),
            reason: "dimension does not fit usize".to_owned(),
        })?;
    if identity.storage.quantization != QuantizationFormat::F32 {
        return Err(SearchError::InvalidConfig {
            field: "identity_bound_embedding.storage.quantization".to_owned(),
            value: format!("{:?}", identity.storage.quantization),
            reason: "an in-process Vec<f32> output must carry an f32 storage identity".to_owned(),
        });
    }
    if !identity.storage.format.starts_with("in-memory-") {
        return Err(SearchError::InvalidConfig {
            field: "identity_bound_embedding.storage.format".to_owned(),
            value: identity.storage.format.clone(),
            reason: "an in-process Vec<f32> output must carry an in-memory storage format"
                .to_owned(),
        });
    }
    if !matches!(
        identity.storage.endianness.as_str(),
        "native-f32-values" | "native-test-only"
    ) {
        return Err(SearchError::InvalidConfig {
            field: "identity_bound_embedding.storage.endianness".to_owned(),
            value: identity.storage.endianness.clone(),
            reason: "an in-process Vec<f32> output must carry a native-value contract".to_owned(),
        });
    }
    Ok(dimension)
}

fn validate_embedder_identity(
    identity: &EmbeddingIdentityBundleV1,
    reported_dimension: usize,
) -> SearchResult<usize> {
    let dimension = validate_embedding_identity(identity)?;
    if reported_dimension != dimension {
        return Err(SearchError::InvalidConfig {
            field: "embedder.dimension".to_owned(),
            value: reported_dimension.to_string(),
            reason: format!("the immutable identity declares {dimension} vector elements"),
        });
    }
    Ok(dimension)
}

fn embedding_identity_changed() -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: "embedding.bound".to_owned(),
        reason:
            "embedding producer or response differs from the identity admitted before inference"
                .to_owned(),
    }
}

// Compare full contracts, never operational names or dimensions alone. A
// provider that loses identity after inference must not become a value-only
// failure that a partial-coverage builder can silently skip.
fn validate_embedding_producer(
    expected: &EmbeddingIdentityBundleV1,
    actual: SearchResult<&EmbeddingIdentityBundleV1>,
    dimension: usize,
) -> SearchResult<()> {
    if actual.is_ok_and(|identity| identity == expected)
        && usize::try_from(expected.space.dimension).ok() == Some(dimension)
    {
        Ok(())
    } else {
        Err(embedding_identity_changed())
    }
}

fn validate_embedding_response(
    response: &IdentityBoundEmbedding,
    expected: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    // Reject foreign fields before invoking diagnostics that might echo them.
    if &response.identity != expected {
        return Err(embedding_identity_changed());
    }
    response.validate()
}

fn validate_embedding_values(values: &[f32], dimension: usize) -> SearchResult<()> {
    if values.len() != dimension {
        return Err(SearchError::InvalidConfig {
            field: "identity_bound_embedding.values".to_owned(),
            value: values.len().to_string(),
            reason: format!("expected {dimension} vector elements"),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(SearchError::InvalidConfig {
            field: "identity_bound_embedding.values".to_owned(),
            value: format!("non-finite coordinate at index {index}"),
            reason: "embedding coordinates must be finite".to_owned(),
        });
    }
    Ok(())
}

fn validate_embedding_batch_length(actual: usize, expected: usize) -> SearchResult<()> {
    if actual != expected {
        return Err(SearchError::InvalidConfig {
            field: "embedder.batch_length".to_owned(),
            value: actual.to_string(),
            reason: format!("expected exactly {expected} output vectors, one per input"),
        });
    }
    Ok(())
}

// ─── Model Category ─────────────────────────────────────────────────────────

/// Classification of an embedding model by its speed/quality tradeoff.
///
/// Used by `EmbedderStack` to pair a fast-tier and quality-tier embedder
/// for the two-tier progressive search pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCategory {
    /// Hash-based (FNV-1a): ultra-fast, deterministic, not semantically meaningful.
    HashEmbedder,
    /// Static token embeddings (Model2Vec/potion): fast with good semantic quality.
    StaticEmbedder,
    /// Transformer inference (MiniLM/BGE): highest quality but slower.
    TransformerEmbedder,
    /// Cloud API embeddings (`OpenAI`, Gemini): high quality, network-dependent latency.
    ApiEmbedder,
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HashEmbedder => write!(f, "hash_embedder"),
            Self::StaticEmbedder => write!(f, "static_embedder"),
            Self::TransformerEmbedder => write!(f, "transformer_embedder"),
            Self::ApiEmbedder => write!(f, "api_embedder"),
        }
    }
}

impl ModelCategory {
    /// Returns the default progressive tier for this model category.
    #[must_use]
    pub const fn default_tier(self) -> ModelTier {
        match self {
            Self::HashEmbedder | Self::StaticEmbedder => ModelTier::Fast,
            Self::TransformerEmbedder | Self::ApiEmbedder => ModelTier::Quality,
        }
    }

    /// Whether this category is semantically meaningful by default.
    #[must_use]
    pub const fn default_semantic_flag(self) -> bool {
        !matches!(self, Self::HashEmbedder)
    }
}

/// Tier assignment in the progressive two-tier pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTier {
    /// Ultra-fast path for immediate results.
    Fast,
    /// Higher-quality path for deferred refinement.
    Quality,
}

impl fmt::Display for ModelTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fast => write!(f, "fast"),
            Self::Quality => write!(f, "quality"),
        }
    }
}

/// Static metadata describing an embedder implementation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Stable model identifier used in index metadata.
    pub id: String,
    /// Human-friendly model name.
    pub name: String,
    /// Embedding dimensionality.
    pub dimension: usize,
    /// Embedder category by architecture/performance profile.
    pub category: ModelCategory,
    /// Default tier assignment in progressive search.
    pub tier: ModelTier,
    /// Whether embeddings encode semantic similarity.
    pub is_semantic: bool,
    /// Whether Matryoshka truncation is supported.
    pub supports_mrl: bool,
    /// Optional upstream model id (e.g., `HuggingFace`).
    pub huggingface_id: Option<String>,
    /// Optional model footprint on disk.
    pub size_bytes: Option<u64>,
    /// Optional model license string.
    pub license: Option<String>,
}

// ─── Embedder Trait ─────────────────────────────────────────────────────────

/// Core trait for text embedding models.
///
/// Implementations run under structured concurrency, so each async operation
/// receives a capability context (`&Cx`) as its first parameter.
///
/// # Contract
///
/// - `embed()` and `embed_batch()` are raw inference primitives; any caller
///   persisting, comparing, caching, or transporting vectors must use
///   `embed_bound()` or `embed_batch_bound()` so space and producer identity
///   travel with the values.
/// - Batches return exactly one vector per input, in input order. Bound
///   operations reject wrong counts, wrong dimensions, and non-finite values.
/// - `dimension()` must be constant for the lifetime of the embedder and agree
///   with its immutable identity.
/// - `id()` must be stable across process restarts for diagnostics and registry
///   selection, but never establishes vector-space compatibility.
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector of f32 floats.
    ///
    /// The returned vector has exactly `self.dimension()` elements.
    /// This raw primitive carries no compatibility proof; use
    /// [`Self::embed_bound`] outside an implementation-local inference path.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if embedding inference fails.
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>>;

    /// Embed a batch of text strings.
    ///
    /// Default implementation calls `embed` in a loop and observes cancellation
    /// between inputs. Neural models should override this to exploit batch
    /// inference and provide checkpoints within their own long-running work.
    /// This raw primitive carries no compatibility proof; use
    /// [`Self::embed_batch_bound`] when values leave the embedder boundary.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if any embedding inference fails or is cancelled.
    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            embedding_checkpoint(cx, "embedder.embed_batch")?;
            let mut out = Vec::with_capacity(texts.len());
            for text in texts {
                embedding_checkpoint(cx, "embedder.embed_batch")?;
                let outcome = self.embed(cx, text).await;
                embedding_checkpoint(cx, "embedder.embed_batch")?;
                out.push(outcome?);
            }
            embedding_checkpoint(cx, "embedder.embed_batch")?;
            Ok(out)
        })
    }

    /// Embed one input and bind the output to the complete verified identity.
    ///
    /// Admit the identity before inference; reject malformed or cancelled
    /// output before it can be persisted, compared, or cached. Producer drift
    /// is refused, and cancellation takes precedence over an inference error.
    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            embedding_checkpoint(cx, "embedder.embed_bound")?;
            let identity = self.identity()?.clone();
            let dimension = validate_embedder_identity(&identity, self.dimension())?;
            let outcome = self.embed(cx, text).await;
            embedding_checkpoint(cx, "embedder.embed_bound")?;
            let values = outcome?;
            validate_embedding_producer(&identity, self.identity(), self.dimension())?;
            validate_embedding_values(&values, dimension)?;
            embedding_checkpoint(cx, "embedder.embed_bound")?;
            Ok(IdentityBoundEmbedding { values, identity })
        })
    }

    /// Embed a batch and bind every output to the same verified identity.
    ///
    /// Validation is all-or-error: a backend returning too few or too many
    /// vectors cannot silently drop inputs through a downstream `zip`. Even
    /// an empty batch must carry a valid identity. Producer identity is checked
    /// again after inference; cancellation outranks a backend error.
    /// Inference implementations remain responsible for cancellation checkpoints
    /// within blocking work.
    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            embedding_checkpoint(cx, "embedder.embed_batch_bound")?;
            let identity = self.identity()?.clone();
            let dimension = validate_embedder_identity(&identity, self.dimension())?;
            let outcome = self.embed_batch(cx, texts).await;
            embedding_checkpoint(cx, "embedder.embed_batch_bound")?;
            let vectors = outcome?;
            validate_embedding_producer(&identity, self.identity(), self.dimension())?;
            validate_embedding_batch_length(vectors.len(), texts.len())?;
            let bound = vectors
                .into_iter()
                .map(|values| {
                    embedding_checkpoint(cx, "embedder.embed_batch_bound")?;
                    validate_embedding_values(&values, dimension)?;
                    Ok(IdentityBoundEmbedding {
                        values,
                        identity: identity.clone(),
                    })
                })
                .collect::<SearchResult<Vec<_>>>()?;
            embedding_checkpoint(cx, "embedder.embed_batch_bound")?;
            Ok(bound)
        })
    }

    /// Whether [`Self::embed_batch_bound`] binds, for each input, exactly what
    /// [`Self::embed_bound`] would: the same producer, identity and checks,
    /// served as one batch.
    ///
    /// Batching callers (the facade `IndexBuilder`) send batches only when
    /// this is `true` and otherwise keep one `embed_bound` per input. The core
    /// defaults satisfy it for a producer that overrides neither bound
    /// operation, so such producers may return `true`. A producer that
    /// overrides only `embed_bound` must keep the default `false`: the default
    /// `embed_batch_bound` would bind its raw batch output instead.
    fn bound_batch_is_native(&self) -> bool {
        false
    }

    /// Complete immutable identity of this embedder and its output/storage contract.
    ///
    /// Legacy/custom implementations that have not supplied a complete identity
    /// fail closed here; raw model names and dimensions never synthesize
    /// compatibility.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the implementation is not identity-aware.
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Err(SearchError::InvalidConfig {
            field: "embedder.identity".to_owned(),
            value: bounded_embedder_diagnostic_id(self.id()),
            reason: "embedder did not supply a complete immutable identity bundle".to_owned(),
        })
    }

    /// The dimensionality of embedding vectors produced by this model.
    fn dimension(&self) -> usize;

    /// A unique, stable identifier for this embedder.
    ///
    /// Examples: `"fnv-hash-384"`, `"potion-multilingual-128M"`, `"all-MiniLM-L6-v2"`.
    /// This is operational metadata only. Persistence and compatibility checks
    /// must use the complete immutable identity bundle.
    fn id(&self) -> &str;

    /// Human-readable model name.
    fn model_name(&self) -> &str;

    /// Whether this embedder is loaded and operational.
    fn is_ready(&self) -> bool {
        true
    }

    /// Whether this embedder produces semantically meaningful vectors.
    ///
    /// Hash embedders return `false`; neural models return `true`.
    fn is_semantic(&self) -> bool;

    /// The speed/quality category of this embedder.
    fn category(&self) -> ModelCategory;

    /// Default progressive tier assignment.
    fn tier(&self) -> ModelTier {
        self.category().default_tier()
    }

    /// Whether this model supports Matryoshka Representation Learning
    /// (dimension truncation for faster search with controlled quality loss).
    fn supports_mrl(&self) -> bool {
        false
    }

    /// Truncate and re-normalize embedding to `target_dim`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when `target_dim` is zero.
    fn truncate_embedding(&self, embedding: &[f32], target_dim: usize) -> SearchResult<Vec<f32>> {
        if target_dim == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: "0".to_owned(),
                reason: "target dimension must be at least 1".to_owned(),
            });
        }

        if target_dim >= embedding.len() {
            return Ok(embedding.to_vec());
        }

        Ok(l2_normalize(&embedding[..target_dim]))
    }
}

// ─── Synchronous Embedder Bridge ─────────────────────────────────────────

/// Synchronous embedding interface for host projects that call embedders from
/// non-async contexts.
///
/// Implement this trait for embedders whose `embed` operations are inherently
/// synchronous (e.g., hash embedders, CPU-only ONNX inference). The companion
/// [`SyncEmbedderAdapter`] wraps any `SyncEmbed` implementor into a full
/// async [`Embedder`], suitable for use anywhere frankensearch expects one.
///
/// # Example
///
/// ```ignore
/// use frankensearch_core::traits::{SyncEmbed, SyncEmbedderAdapter, Embedder};
///
/// struct MyHashEmbedder { dim: usize }
///
/// impl SyncEmbed for MyHashEmbedder {
///     fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> { /* ... */ }
///     fn dimension(&self) -> usize { self.dim }
///     fn id(&self) -> &str { "my-hash" }
///     fn model_name(&self) -> &str { "My Hash Embedder" }
///     fn is_semantic(&self) -> bool { false }
///     fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
/// }
///
/// // Use it as a full async Embedder:
/// let adapted: Box<dyn Embedder> = Box::new(SyncEmbedderAdapter(MyHashEmbedder { dim: 256 }));
/// ```
pub trait SyncEmbed: Send + Sync {
    /// Synchronously embed a single text into a vector.
    ///
    /// This raw primitive carries no compatibility proof; callers that persist,
    /// compare, cache, or transport the vector must use
    /// [`Self::embed_bound_sync`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] when embedding fails (for example model load,
    /// inference, or input validation failures).
    fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>>;

    /// Synchronously embed a batch of texts.
    ///
    /// Default implementation calls [`embed_sync`](Self::embed_sync) for each text.
    /// Use [`Self::embed_batch_bound_sync`] when vectors leave an
    /// implementation-local inference path.
    ///
    /// # Errors
    ///
    /// Returns the first [`SearchError`] encountered while embedding any item
    /// in the batch.
    fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_sync(t)).collect()
    }

    /// Synchronously embed one input and bind it to the complete identity.
    ///
    /// # Errors
    ///
    /// Returns the embedding error or fails closed when identity, dimension,
    /// or finite-coordinate validation fails. Identity is admitted before work
    /// and checked again before returning values.
    fn embed_bound_sync(&self, text: &str) -> SearchResult<IdentityBoundEmbedding> {
        let identity = self.identity()?.clone();
        let dimension = validate_embedder_identity(&identity, self.dimension())?;
        let values = self.embed_sync(text)?;
        validate_embedding_producer(&identity, self.identity(), self.dimension())?;
        validate_embedding_values(&values, dimension)?;
        Ok(IdentityBoundEmbedding { values, identity })
    }

    /// Synchronously embed a batch and bind every output to one identity.
    ///
    /// # Errors
    ///
    /// Returns the first embedding or validation error. A batch must return
    /// exactly one finite, correctly dimensioned vector per input; no partial
    /// batch is returned. Identity is validated even for an empty batch and
    /// producer drift during inference is refused.
    fn embed_batch_bound_sync(&self, texts: &[&str]) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        let identity = self.identity()?.clone();
        let dimension = validate_embedder_identity(&identity, self.dimension())?;
        let vectors = self.embed_batch_sync(texts)?;
        validate_embedding_producer(&identity, self.identity(), self.dimension())?;
        validate_embedding_batch_length(vectors.len(), texts.len())?;
        vectors
            .into_iter()
            .map(|values| {
                validate_embedding_values(&values, dimension)?;
                Ok(IdentityBoundEmbedding {
                    values,
                    identity: identity.clone(),
                })
            })
            .collect()
    }

    /// Complete immutable identity of this embedder and its output/storage contract.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the implementation is not identity-aware.
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Err(SearchError::InvalidConfig {
            field: "sync_embedder.identity".to_owned(),
            value: bounded_embedder_diagnostic_id(self.id()),
            reason: "embedder did not supply a complete immutable identity bundle".to_owned(),
        })
    }

    /// The output dimensionality of embedding vectors.
    fn dimension(&self) -> usize;

    /// Unique, stable operational identifier for this embedder.
    ///
    /// It never substitutes for the immutable identity bundle in persistence or
    /// compatibility checks.
    fn id(&self) -> &str;

    /// Human-readable model name.
    fn model_name(&self) -> &str {
        self.id()
    }

    /// Whether the embedder is loaded and operational.
    fn is_ready(&self) -> bool {
        true
    }

    /// Whether this embedder produces semantically meaningful vectors.
    fn is_semantic(&self) -> bool;

    /// The speed/quality category of this embedder.
    fn category(&self) -> ModelCategory;

    /// Default progressive tier assignment.
    fn tier(&self) -> ModelTier {
        self.category().default_tier()
    }

    /// Whether this model supports Matryoshka Representation Learning.
    fn supports_mrl(&self) -> bool {
        false
    }
}

/// Adapts a [`SyncEmbed`] implementor into a full async [`Embedder`].
///
/// Each operation invokes its corresponding synchronous method when polled.
/// Bound operations preserve custom `embed_bound_sync`/`embed_batch_bound_sync`
/// dispatch and validate the actual responses against the pre-inference identity;
/// they never replace a provider's bound result with relabeled raw inference.
/// Cancellation is checked before work and after either success or failure.
///
/// This adapter does not create a blocking pool or preempt running inference.
/// Expensive synchronous providers must still be polled on the caller's blocking
/// lane. Raw methods remain raw, including for legacy identity-unaware providers.
pub struct SyncEmbedderAdapter<T: SyncEmbed>(pub T);

impl<T: SyncEmbed + 'static> Embedder for SyncEmbedderAdapter<T> {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            embedding_checkpoint(cx, "sync_embed.embed")?;
            let outcome = self.0.embed_sync(text);
            embedding_checkpoint(cx, "sync_embed.embed")?;
            outcome
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            embedding_checkpoint(cx, "sync_embed.embed_batch")?;
            let outcome = self.0.embed_batch_sync(texts);
            embedding_checkpoint(cx, "sync_embed.embed_batch")?;
            outcome
        })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            embedding_checkpoint(cx, "sync_embed.embed_bound")?;
            let identity = self.0.identity()?.clone();
            validate_embedder_identity(&identity, self.0.dimension())?;
            let outcome = self.0.embed_bound_sync(text);
            embedding_checkpoint(cx, "sync_embed.embed_bound")?;
            let response = outcome?;
            validate_embedding_producer(&identity, self.0.identity(), self.0.dimension())?;
            validate_embedding_response(&response, &identity)?;
            embedding_checkpoint(cx, "sync_embed.embed_bound")?;
            Ok(response)
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            embedding_checkpoint(cx, "sync_embed.embed_batch_bound")?;
            let identity = self.0.identity()?.clone();
            validate_embedder_identity(&identity, self.0.dimension())?;
            let outcome = self.0.embed_batch_bound_sync(texts);
            embedding_checkpoint(cx, "sync_embed.embed_batch_bound")?;
            let responses = outcome?;
            validate_embedding_producer(&identity, self.0.identity(), self.0.dimension())?;
            validate_embedding_batch_length(responses.len(), texts.len())?;
            for response in &responses {
                embedding_checkpoint(cx, "sync_embed.embed_batch_bound")?;
                validate_embedding_response(response, &identity)?;
            }
            embedding_checkpoint(cx, "sync_embed.embed_batch_bound")?;
            Ok(responses)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.0.identity()
    }

    fn dimension(&self) -> usize {
        self.0.dimension()
    }

    fn id(&self) -> &str {
        self.0.id()
    }

    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    fn is_ready(&self) -> bool {
        self.0.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.0.is_semantic()
    }

    fn category(&self) -> ModelCategory {
        self.0.category()
    }

    fn tier(&self) -> ModelTier {
        self.0.tier()
    }

    fn supports_mrl(&self) -> bool {
        self.0.supports_mrl()
    }
}

// ─── Embedding Utilities ──────────────────────────────────────────────────

/// L2-normalizes a vector to unit length.
///
/// Returns a zero vector if the input has zero norm (avoids division by zero).
#[must_use]
pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq < f32::EPSILON {
        return vec![0.0; vec.len()];
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    vec.iter().map(|x| x * inv_norm).collect()
}

/// L2-normalizes a vector to unit length **in place**.
///
/// Bit-identical to [`l2_normalize`] (same `norm_sq` accumulation, same
/// `is_finite`/`EPSILON` guard, same `x * inv_norm` scaling, zero vector on zero
/// norm) but reuses the caller's owned buffer instead of allocating a fresh `Vec` —
/// for callers that already own the vector (e.g. the hash embedder builds its
/// accumulator then normalizes it), this drops one dimension-sized allocation.
pub fn l2_normalize_in_place(vec: &mut [f32]) {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq < f32::EPSILON {
        for x in vec.iter_mut() {
            *x = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    // Element-wise scale — runtime-AVX2 (~1.7× over the SSE2 auto-vec on this
    // no-global-avx2 build); bit-identical (per-element IEEE multiply).
    crate::simd::scale_f32_in_place(vec, inv_norm);
}

/// Computes cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
///
/// # Panics
///
/// Panics in debug mode if the vectors have different lengths.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Runtime length check — debug_assert is stripped in release builds,
    // and zip would silently truncate mismatched vectors.
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let denom = norm_a * norm_b;
    if !denom.is_finite() || denom < f32::EPSILON {
        return 0.0;
    }
    dot / denom
}

/// Truncates an embedding to a target dimension and re-normalizes.
///
/// Only meaningful for models that support Matryoshka Representation Learning (MRL),
/// where the first N dimensions capture most of the variance.
///
/// Returns the original vector unchanged if `target_dim >= embedding.len()`.
#[must_use]
pub fn truncate_embedding(embedding: &[f32], target_dim: usize) -> Vec<f32> {
    if target_dim >= embedding.len() {
        return embedding.to_vec();
    }
    l2_normalize(&embedding[..target_dim])
}
