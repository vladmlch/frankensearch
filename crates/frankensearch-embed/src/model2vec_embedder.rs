//! `Model2Vec` static token embedding for the fast tier.
//!
//! Wraps the exact manifest-registered potion-multilingual-128M artifact set.
//! It looks up pre-computed per-token embeddings and mean-pools them. No
//! transformer inference, no GPU needed.
//!
//! Performance: ~0.57ms per embedding (223x faster than `MiniLM-L6-v2`).
//!
//! Memory: ~32MB resident for a 32K-vocab × 256-dim model.
//!
//! Only two files are required:
//! - `tokenizer.json` (`HuggingFace` BPE tokenizer)
//! - `model.safetensors` (static embedding matrix)

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::Instant;

use asupersync::Cx;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use tokenizers::Tokenizer;
use tracing::instrument;

use crate::model_manifest::{
    MODEL2VEC_OUTPUT_NORMALIZATION_V1, MODEL2VEC_POOLING_V1, MODEL2VEC_PREPROCESSING_V1,
    MODEL2VEC_PREPROCESSING_V2, MODEL2VEC_SEQUENCE_POLICY_V1, ModelArtifactManifestV1,
};
use crate::model_registry::{ensure_model_storage_layout, model_directory_variants};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};

/// Required files for a `Model2Vec` model.
const REQUIRED_FILES: [&str; 2] = ["tokenizer.json", "model.safetensors"];

/// Batch size at/above which `embed_batch` dispatches per-document embedding across
/// Rayon threads. Each `embed_sync` is ~0.57 ms of independent CPU work (tokenize →
/// static-row gather → mean-pool → normalize), so parallel dispatch amortizes Rayon
/// scheduling at a much smaller batch than the FNV hash embedder needs (256); smaller
/// batches stay serial to preserve single/few-doc latency.
const PARALLEL_BATCH_MIN: usize = 8;

/// Default model name for the primary fast-tier model.
const DEFAULT_MODEL_NAME: &str = "potion-multilingual-128M";

/// Default `HuggingFace` model ID for the primary fast-tier model.
const DEFAULT_HF_ID: &str = "minishlab/potion-multilingual-128M";

mod compact_unigram;
mod registered;
mod safetensors_stream;
use compact_unigram::{CompactUnigramTokenizer, is_unigram_tokenizer};
pub use registered::RegisteredModel2Vec;
use safetensors_stream::SafetensorsF32Matrix;

/// A loaded `Model2Vec` tokenizer. A Unigram vocabulary uses the compact
/// model in [`compact_unigram`], which segments exactly as the library does at
/// a fraction of the build time and memory; every other model loads through
/// the library unchanged.
enum Model2VecTokenizer {
    Library(Tokenizer),
    CompactUnigram(CompactUnigramTokenizer),
}

impl Model2VecTokenizer {
    fn from_file(path: &Path) -> tokenizers::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        if is_unigram_tokenizer(&content) {
            Ok(Self::CompactUnigram(serde_json::from_str(&content)?))
        } else {
            Ok(Self::Library(serde_json::from_str(&content)?))
        }
    }

    fn encode_fast(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> tokenizers::Result<tokenizers::Encoding> {
        match self {
            Self::Library(tokenizer) => tokenizer.encode_fast(text, add_special_tokens),
            Self::CompactUnigram(tokenizer) => tokenizer.encode_fast(text, add_special_tokens),
        }
    }

    #[cfg(any(test, feature = "bench-internals"))]
    fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> tokenizers::Result<tokenizers::Encoding> {
        match self {
            Self::Library(tokenizer) => tokenizer.encode(text, add_special_tokens),
            Self::CompactUnigram(tokenizer) => tokenizer.encode(text, add_special_tokens),
        }
    }

    fn get_padding(&self) -> Option<&tokenizers::PaddingParams> {
        match self {
            Self::Library(tokenizer) => tokenizer.get_padding(),
            Self::CompactUnigram(tokenizer) => tokenizer.get_padding(),
        }
    }

    #[cfg(test)]
    fn get_added_vocabulary(&self) -> &tokenizers::AddedVocabulary {
        match self {
            Self::Library(tokenizer) => tokenizer.get_added_vocabulary(),
            Self::CompactUnigram(tokenizer) => tokenizer.get_added_vocabulary(),
        }
    }

    #[cfg(test)]
    fn get_truncation(&self) -> Option<&tokenizers::TruncationParams> {
        match self {
            Self::Library(tokenizer) => tokenizer.get_truncation(),
            Self::CompactUnigram(tokenizer) => tokenizer.get_truncation(),
        }
    }

    #[cfg(test)]
    fn with_padding(&mut self, padding: Option<tokenizers::PaddingParams>) {
        match self {
            Self::Library(tokenizer) => {
                tokenizer.with_padding(padding);
            }
            Self::CompactUnigram(tokenizer) => {
                tokenizer.with_padding(padding);
            }
        }
    }
}

/// Static token embedding model (`Model2Vec` / potion).
///
/// After construction, all fields are immutable — no `Mutex` needed.
/// The struct is `Send + Sync` by construction.
///
/// # Loading
///
/// ```rust,ignore
/// let embedder = Model2VecEmbedder::load("/path/to/model")?;
/// let embedding = embedder.embed_sync("hello world");
/// assert_eq!(embedding.len(), 256);
/// ```
pub struct Model2VecEmbedder {
    /// `HuggingFace` tokenizer (compact model for a Unigram vocabulary).
    tokenizer: Model2VecTokenizer,
    /// Flat embedding matrix: `embeddings[token_id * dim .. (token_id + 1) * dim]`.
    embeddings: Vec<f32>,
    /// Output dimensionality.
    dimensions: usize,
    /// Vocabulary size (number of rows in the embedding matrix).
    vocab_size: usize,
    /// Human-readable model name.
    name: String,
    /// Directory the model was loaded from.
    model_dir: PathBuf,
    /// Complete identity derived from the verified frozen manifest.
    identity: EmbeddingIdentityBundleV1,
    /// Token id excluded from pooling: the `WordPiece` unknown token under the
    /// V2 preprocessing contract, matching reference `model2vec`. `None`
    /// under V1, which pools every in-matrix id.
    dropped_token_id: Option<u32>,
    /// Test-only witness that the shipping tokenizer returned the offset-free encoding shape.
    #[cfg(test)]
    last_tokenizer_route_was_offset_free: AtomicBool,
}

impl fmt::Debug for Model2VecEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model2VecEmbedder")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("vocab_size", &self.vocab_size)
            .field("model_dir", &"<redacted>")
            .field("identity", &self.identity.fingerprint())
            .finish_non_exhaustive()
    }
}

impl Model2VecEmbedder {
    /// Load a `Model2Vec` model from a directory containing `tokenizer.json`
    /// and `model.safetensors`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` if required files are missing.
    /// Returns `SearchError::ModelLoadFailed` if files exist but cannot be parsed.
    #[instrument(skip_all, fields(model = DEFAULT_MODEL_NAME))]
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        Self::load_with_name(model_dir, DEFAULT_MODEL_NAME)
    }

    /// Load the registered potion model with a custom display identifier.
    ///
    /// The supplied name does not select or attest compatibility; production
    /// model bytes and runtime identity still come exclusively from the frozen
    /// potion manifest.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` if required files are missing.
    /// Returns `SearchError::ModelLoadFailed` if files exist but cannot be parsed.
    pub fn load_with_name(model_dir: impl AsRef<Path>, name: &str) -> SearchResult<Self> {
        let model_dir = model_dir.as_ref();
        #[cfg(test)]
        {
            Self::load_explicit_test_model(model_dir, name)
        }
        #[cfg(not(test))]
        {
            Self::load_preverified(model_dir, name, admit_registered_potion(model_dir)?)
        }
    }

    /// Load the registered potion model, reusing the process-wide instance
    /// when one is already resident.
    ///
    /// Prefer this over [`Self::load`] in anything that may load the model
    /// more than once per process — a daemon, a server, a runtime rebuilt per
    /// request. Building the 500 353-piece Unigram tokenizer and reading the
    /// 512 MB matrix costs seconds and half a gigabyte, and neither depends on
    /// the caller, so the second and later loads in a process should be free
    /// (GH #46).
    ///
    /// **What keeps a changed model from being served from cache is
    /// `admit_registered_potion`, which runs on EVERY call before the cache is
    /// consulted** — not the cache key. The receipt it checks carries each
    /// file's size, mtime, ctime, device and inode, so a rewritten or replaced
    /// model invalidates it and forces a full hash pass; an artifact that
    /// fails admission returns before the lookup and can never be handed back.
    /// Do not weaken that call on the theory that the key protects you: in
    /// production the identity fingerprint is derived from a frozen manifest
    /// and is the same constant for every successful load. It is in the key so
    /// that a future caller loading a DIFFERENT registered model cannot
    /// collide, not to detect a changed one.
    ///
    /// The cache holds only a [`Weak`] reference, so the matrix is released as
    /// soon as the last caller drops its `Arc`.
    ///
    /// The key also carries the display name, so two callers that load the
    /// same directory under different names get separate instances rather than
    /// one whose `name` disagrees with what the caller asked for. Callers that
    /// want to share should agree on the name.
    ///
    /// This does NOT amortise across processes; a one-shot CLI invocation
    /// still pays a full load. Cross-process reuse would need a compiled
    /// tokenizer representation that `tokenizers` 0.23 does not provide.
    ///
    /// # Errors
    ///
    /// Same as [`Self::load`].
    #[instrument(skip_all, fields(model = DEFAULT_MODEL_NAME))]
    pub fn load_shared(model_dir: impl AsRef<Path>) -> SearchResult<Arc<Self>> {
        Self::load_shared_with_name(model_dir, DEFAULT_MODEL_NAME)
    }

    /// [`Self::load_shared`] with a custom display identifier, mirroring
    /// [`Self::load_with_name`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::load_with_name`].
    pub fn load_shared_with_name(
        model_dir: impl AsRef<Path>,
        name: &str,
    ) -> SearchResult<Arc<Self>> {
        let model_dir = model_dir.as_ref();
        #[cfg(test)]
        let identity = EmbeddingIdentityBundleV1::explicit_test_model(name, 1);
        #[cfg(not(test))]
        let identity = admit_registered_potion(model_dir)?;
        Self::load_shared_preverified(model_dir, name, identity)
    }

    /// Cache lookup and, on a miss, the full load.
    ///
    /// The cache lock is held ACROSS the load. That serialises concurrent
    /// first loads, which is the point: releasing it would let two threads
    /// each allocate a 512 MB matrix for the same model and then discard one.
    /// Loading is a startup event, and this crate carries a single fast-tier
    /// model, so the contention window is not on any steady-state path.
    // `significant_drop_tightening` wants the guard released before the load.
    // Releasing it is precisely the behaviour this design rejects: it would
    // let two threads each build a 512 MB matrix for the same model and throw
    // one away.
    #[allow(clippy::significant_drop_tightening)]
    fn load_shared_preverified(
        model_dir: &Path,
        name: &str,
        identity: EmbeddingIdentityBundleV1,
    ) -> SearchResult<Arc<Self>> {
        let key = SharedModelKey {
            dir: canonical_model_dir(model_dir),
            name: name.to_owned(),
            identity: identity.fingerprint(),
        };

        let mut cache = shared_model_cache()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        if let Some(existing) = cache.get(&key).and_then(Weak::upgrade) {
            return Ok(existing);
        }

        let embedder = Arc::new(Self::load_preverified(model_dir, name, identity)?);
        // Drop entries whose last strong reference is gone, so a long-lived
        // process that cycles through model directories does not accumulate
        // dead keys.
        cache.retain(|_, weak| weak.strong_count() > 0);
        cache.insert(key, Arc::downgrade(&embedder));
        Ok(embedder)
    }

    fn load_preverified(
        model_dir: &Path,
        name: &str,
        mut identity: EmbeddingIdentityBundleV1,
    ) -> SearchResult<Self> {
        // Validate required files exist
        for filename in &REQUIRED_FILES {
            let path = model_dir.join(filename);
            if !path.exists() {
                return Err(SearchError::ModelNotFound {
                    name: format!("{name} (missing {filename} in {})", model_dir.display()),
                });
            }
        }

        // Per-phase load timings (GH #46).
        //
        // A cold load of the 128M model is seconds long and the split between
        // building the 500k-piece tokenizer and streaming the 512 MB matrix is
        // what decides which one is worth attacking. Reporters were
        // reconstructing that split through the Python binding because the Rust
        // binary emitted one event at the end; these make the real binary say
        // it. Each phase logs as it completes, so a load that never finishes
        // still attributes the time it spent.
        let load_started = Instant::now();
        let elapsed_ms = |started: Instant| started.elapsed().as_secs_f64() * 1_000.0;

        // Load tokenizer
        let tokenizer_started = Instant::now();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Model2VecTokenizer::from_file(&tokenizer_path).map_err(|e| {
            SearchError::ModelLoadFailed {
                path: tokenizer_path,
                source: format!("failed to load tokenizer: {e}").into(),
            }
        })?;
        let tokenizer_build_ms = elapsed_ms(tokenizer_started);
        tracing::debug!(
            model = name,
            tokenizer_build_ms,
            "Model2Vec tokenizer built"
        );

        // Locate the embedding tensor WITHOUT reading the matrix.
        //
        // This used to be `std::fs::read` of the whole file plus
        // `SafeTensors::deserialize` over that buffer, which meant the raw 512
        // MB of file bytes and the decoded 512 MB `Vec<f32>` were live at the
        // same time — a ~1 GB peak and roughly 2.6 GB of pages touched for a
        // 512 MB artifact (GH #46). Only the header is read here; the matrix
        // is streamed straight into its final `Vec<f32>` below, and only after
        // every admission check has passed.
        let header_started = Instant::now();
        let safetensors_path = model_dir.join("model.safetensors");
        let located = SafetensorsF32Matrix::locate(&safetensors_path).map_err(|e| {
            SearchError::ModelLoadFailed {
                path: safetensors_path.clone(),
                source: e.into(),
            }
        })?;
        let header_locate_ms = elapsed_ms(header_started);
        tracing::debug!(
            model = name,
            header_locate_ms,
            "Model2Vec tensor header located"
        );

        let admission_started = Instant::now();
        let vocab_size = located.vocab_size;
        let dimensions = located.dimensions;
        let parsed_dimension =
            u32::try_from(dimensions).map_err(|_| SearchError::InvalidConfig {
                field: "model2vec.dimension".to_owned(),
                value: dimensions.to_string(),
                reason: "parsed tensor dimension exceeds the identity schema".to_owned(),
            })?;

        if identity.producer.backend == "explicit-test-backend" {
            identity.space.dimension = parsed_dimension;
            identity.storage.dimension = parsed_dimension;
            identity.producer.golden_vectors.dimension = parsed_dimension;
            identity.producer.space_fingerprint = identity.space.fingerprint();
        }
        if identity.space.dimension != parsed_dimension {
            return Err(SearchError::ModelLoadFailed {
                path: safetensors_path.clone(),
                source: format!(
                    "parsed embedding dimension {parsed_dimension} disagrees with attested dimension {}",
                    identity.space.dimension
                )
                .into(),
            });
        }
        identity.validate()?;
        let dropped_token_id = if identity.space.model_preprocessing == MODEL2VEC_PREPROCESSING_V2 {
            Some(wordpiece_unknown_token_id(&tokenizer).map_err(|reason| {
                SearchError::ModelLoadFailed {
                    path: model_dir.join("tokenizer.json"),
                    source: reason.into(),
                }
            })?)
        } else {
            None
        };
        let admission_ms = elapsed_ms(admission_started);
        tracing::debug!(model = name, admission_ms, "Model2Vec identity admitted");

        // Stream the matrix into its final buffer. Every check that could
        // reject this artifact — receipt, dtype, shape, attested dimension,
        // identity — has already run, so no borrowed or decoded bytes can
        // outlive a failed admission.
        let stream_started = Instant::now();
        let embeddings = located
            .read_values()
            .map_err(|e| SearchError::ModelLoadFailed {
                path: safetensors_path,
                source: e.into(),
            })?;
        let matrix_stream_ms = elapsed_ms(stream_started);
        tracing::debug!(
            model = name,
            matrix_stream_ms,
            values = embeddings.len(),
            "Model2Vec matrix streamed"
        );

        #[cfg(test)]
        record_model2vec_full_load(model_dir);

        // One line carrying the whole split, so a field report does not have to
        // stitch four events together (GH #46).
        tracing::debug!(
            model = name,
            tokenizer_build_ms,
            header_locate_ms,
            admission_ms,
            matrix_stream_ms,
            total_ms = elapsed_ms(load_started),
            "Model2Vec load phase timings"
        );

        tracing::info!(
            model = %identity.space.logical_model_id,
            vocab_size,
            dimensions,
            manifest = %identity.producer.provenance_manifest_fingerprint,
            identity = %identity.fingerprint(),
            "Model2Vec model loaded"
        );

        Ok(Self {
            tokenizer,
            embeddings,
            dimensions,
            vocab_size,
            name: name.to_owned(),
            model_dir: model_dir.to_owned(),
            identity,
            dropped_token_id,
            #[cfg(test)]
            last_tokenizer_route_was_offset_free: AtomicBool::new(false),
        })
    }

    #[cfg(test)]
    fn load_explicit_test_model(model_dir: &Path, name: &str) -> SearchResult<Self> {
        Self::load_preverified(
            model_dir,
            name,
            EmbeddingIdentityBundleV1::explicit_test_model(name, 1),
        )
    }

    /// Synchronous embedding (no async overhead for ~0.57ms operation).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbeddingFailed` if tokenization fails or
    /// all tokens are out-of-vocabulary.
    pub fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> {
        if text.is_empty() {
            // Empty text → return zero vector (consistent with hash embedder)
            return Ok(vec![0.0; self.dimensions]);
        }

        // `Model2Vec` consumes token IDs only. Avoid constructing offsets, token strings, and
        // word IDs that this private boundary immediately discards.
        let encoding =
            self.tokenizer
                .encode_fast(text, false)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("tokenization failed: {e}").into(),
                })?;

        #[cfg(test)]
        self.last_tokenizer_route_was_offset_free.store(
            !encoding.get_ids().is_empty()
                && encoding.get_tokens().iter().all(String::is_empty)
                && encoding
                    .get_offsets()
                    .iter()
                    .all(|&offset| offset == (0, 0))
                && encoding.get_word_ids().iter().all(Option::is_none),
            Ordering::Relaxed,
        );

        Ok(self.embed_token_ids(encoding.get_ids()))
    }

    #[inline]
    fn embed_token_ids(&self, token_ids: &[u32]) -> Vec<f32> {
        // V2 contract: drop the unknown token before pooling, as reference
        // `model2vec` does. Only texts that actually contain it pay for the
        // filtered copy.
        let filtered;
        let token_ids = match self.dropped_token_id {
            Some(dropped) if token_ids.contains(&dropped) => {
                filtered = token_ids
                    .iter()
                    .copied()
                    .filter(|id| *id != dropped)
                    .collect::<Vec<_>>();
                filtered.as_slice()
            }
            _ => token_ids,
        };
        if token_ids.is_empty() {
            return vec![0.0; self.dimensions];
        }

        // Mean pool: accumulate embeddings for in-vocabulary tokens
        let mut sum = vec![0.0_f32; self.dimensions];
        let count = crate::simd::accumulate_model2vec_rows(
            &mut sum,
            &self.embeddings,
            token_ids,
            self.vocab_size,
        );

        if count == 0 {
            // All tokens were OOV — return zero vector
            return vec![0.0; self.dimensions];
        }

        // Store each rounded mean value before adding its square in the same strict
        // left-to-right `Iterator::sum()` order, starting from 0.0, as the former pass.
        finish_mean_pool_and_normalize(&mut sum, count);
        sum
    }

    #[cfg(test)]
    fn last_tokenizer_route_was_offset_free(&self) -> bool {
        self.last_tokenizer_route_was_offset_free
            .load(Ordering::Relaxed)
    }

    /// Former pre-lever `encode` route retained only for the existing internal benchmark.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_embed_sync_former_encode(&self, text: &str) -> SearchResult<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimensions]);
        }

        let encoding =
            self.tokenizer
                .encode(text, false)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("tokenization failed: {e}").into(),
                })?;
        Ok(self.embed_token_ids(encoding.get_ids()))
    }

    /// Former pre-lever mean-pool finish retained only for the internal paired benchmark.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_embed_sync_former_finish(&self, text: &str) -> SearchResult<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimensions]);
        }

        let encoding =
            self.tokenizer
                .encode_fast(text, false)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("tokenization failed: {e}").into(),
                })?;
        Ok(self.embed_token_ids_with_former_finish(encoding.get_ids()))
    }

    #[cfg(feature = "bench-internals")]
    fn embed_token_ids_with_former_finish(&self, token_ids: &[u32]) -> Vec<f32> {
        if token_ids.is_empty() {
            return vec![0.0; self.dimensions];
        }

        let mut sum = vec![0.0_f32; self.dimensions];
        let count = crate::simd::accumulate_model2vec_rows(
            &mut sum,
            &self.embeddings,
            token_ids,
            self.vocab_size,
        );
        if count == 0 {
            return vec![0.0; self.dimensions];
        }

        finish_mean_pool_and_normalize_former(&mut sum, count);
        sum
    }

    /// Embed a batch of texts, dispatching per-document `embed_sync` across Rayon
    /// threads once the batch reaches [`PARALLEL_BATCH_MIN`]. Each document is
    /// independent CPU-bound work, so the result is **identical** to the serial loop
    /// (Rayon's indexed `collect` preserves input order); only the wall-clock differs.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbeddingFailed` if tokenization fails or
    /// all tokens are out-of-vocabulary for any text in the batch.
    pub fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        if texts.len() >= PARALLEL_BATCH_MIN {
            texts.par_iter().map(|text| self.embed_sync(text)).collect()
        } else {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed_sync(text)?);
            }
            Ok(results)
        }
    }

    /// The directory this model was loaded from.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Vocabulary size (number of token embeddings in the matrix).
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[inline]
fn finish_mean_pool_and_normalize(sum: &mut [f32], count: usize) {
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / count as f32;
    let mut norm_sq = 0.0_f32;
    for value in sum.iter_mut() {
        *value *= inv;
        norm_sq += *value * *value;
    }
    if norm_sq.is_finite() && norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for value in sum {
            *value *= inv_norm;
        }
    } else {
        sum.fill(0.0);
    }
}

/// Exact former finish sequence retained only as an independently callable oracle.
#[cfg(any(test, feature = "bench-internals"))]
fn finish_mean_pool_and_normalize_former(sum: &mut [f32], count: usize) {
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / count as f32;
    for value in sum.iter_mut() {
        *value *= inv;
    }
    let norm_sq: f32 = sum.iter().map(|value| value * value).sum();
    if norm_sq.is_finite() && norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for value in sum {
            *value *= inv_norm;
        }
    } else {
        sum.fill(0.0);
    }
}

/// The unknown-token id of a `WordPiece` tokenizer, which the V2 preprocessing
/// contract drops before pooling. Any other tokenizer model cannot honor V2.
fn wordpiece_unknown_token_id(tokenizer: &Model2VecTokenizer) -> Result<u32, String> {
    let Model2VecTokenizer::Library(tokenizer) = tokenizer else {
        return Err(
            "the V2 Model2Vec preprocessing contract requires a WordPiece tokenizer".to_owned(),
        );
    };
    let tokenizers::models::ModelWrapper::WordPiece(wordpiece) = tokenizer.get_model() else {
        return Err(
            "the V2 Model2Vec preprocessing contract requires a WordPiece tokenizer".to_owned(),
        );
    };
    tokenizer.token_to_id(&wordpiece.unk_token).ok_or_else(|| {
        format!(
            "WordPiece unknown token {:?} is absent from the tokenizer vocabulary",
            wordpiece.unk_token
        )
    })
}

fn validate_registered_execution_contract(
    identity: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    let preprocessing = identity.space.model_preprocessing.as_str();
    if preprocessing != MODEL2VEC_PREPROCESSING_V1 && preprocessing != MODEL2VEC_PREPROCESSING_V2 {
        return Err(SearchError::InvalidConfig {
            field: "model2vec.execution_contract".to_owned(),
            value: identity.space.logical_model_id.clone(),
            reason: "registered model preprocessing disagrees with the native Model2Vec backend"
                .to_owned(),
        });
    }
    for (field, actual, expected) in [
        (
            "sequence policy",
            identity.space.sequence_policy.as_str(),
            MODEL2VEC_SEQUENCE_POLICY_V1,
        ),
        (
            "pooling",
            identity.space.pooling.as_str(),
            MODEL2VEC_POOLING_V1,
        ),
        (
            "output normalization",
            identity.space.output_normalization.as_str(),
            MODEL2VEC_OUTPUT_NORMALIZATION_V1,
        ),
    ] {
        if actual != expected {
            return Err(SearchError::InvalidConfig {
                field: "model2vec.execution_contract".to_owned(),
                value: identity.space.logical_model_id.clone(),
                reason: format!("registered {field} disagrees with the native Model2Vec backend"),
            });
        }
    }
    Ok(())
}

fn embed_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

impl Embedder for Model2VecEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            embed_checkpoint(cx, "model2vec.embed")?;
            self.embed_sync(text)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            embed_checkpoint(cx, "model2vec.embed_batch")?;
            self.embed_batch_sync(texts)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    /// Both bound operations are the core defaults over this producer's own
    /// `embed`/`embed_batch`, so a bound batch binds what singles would.
    fn bound_batch_is_native(&self) -> bool {
        true
    }

    fn dimension(&self) -> usize {
        self.dimensions
    }

    fn id(&self) -> &str {
        &self.name
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

/// Admit a model directory as the registered potion artifact and derive its
/// runtime identity.
///
/// The ONE place this policy lives. Both public constructors call it, so
/// `load` and `load_shared` cannot drift apart — and the shared constructor's
/// central safety claim is that this runs on EVERY call, before any cached
/// instance is handed back.
///
/// The native manifest is derived from the potion download manifest, whose
/// `.verified` receipt is minted after a full hash pass. Reusing that receipt
/// is what keeps the 512 MB safetensors file from being re-hashed on every
/// process start; any mismatch — size, mtime, ctime, device or inode — falls
/// back to the full pass, so a model replaced or rewritten on disk cannot be
/// admitted on the strength of the old receipt.
#[cfg(not(test))]
fn admit_registered_potion(model_dir: &Path) -> SearchResult<EmbeddingIdentityBundleV1> {
    let verified = ModelArtifactManifestV1::potion_128m_native()?.verify_dir_cached(
        &crate::model_manifest::ModelManifest::potion_128m(),
        model_dir,
    )?;
    let identity = verified.identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
    validate_registered_execution_contract(&identity)?;
    Ok(identity)
}

/// Test hook: how many FULL loads (tokenizer construction plus matrix read)
/// each model directory has cost this process.
///
/// Counted PER DIRECTORY rather than as one global tally so an assertion is
/// unaffected by the other tests loading their own fixtures in parallel.
/// `#[cfg(test)]` — a shipping build carries neither the map nor the lock.
#[cfg(test)]
fn model2vec_full_load_counts() -> &'static Mutex<HashMap<PathBuf, u64>> {
    static COUNTS: OnceLock<Mutex<HashMap<PathBuf, u64>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Full loads charged to `model_dir` so far in this process.
#[cfg(test)]
fn model2vec_full_loads_for(model_dir: &Path) -> u64 {
    model2vec_full_load_counts()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(model_dir)
        .copied()
        .unwrap_or(0)
}

#[cfg(test)]
fn record_model2vec_full_load(model_dir: &Path) {
    *model2vec_full_load_counts()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(canonical_model_dir(model_dir))
        .or_insert(0) += 1;
}

/// Canonical form of a model directory, used as the cache key and as the
/// per-directory counter key so both agree.
///
/// Canonicalising makes `./model` and its absolute path one entry — and on
/// macOS, where `/var` is a symlink to `/private/var`, keeps a temp directory
/// from being two. A path that cannot be canonicalised (it may not exist; the
/// load reports that properly) is used as given.
fn canonical_model_dir(model_dir: &Path) -> PathBuf {
    std::fs::canonicalize(model_dir).unwrap_or_else(|_| model_dir.to_owned())
}

/// Key for the process-wide loaded-model cache.
///
/// The identity fingerprint is here so that two DIFFERENT registered models
/// cannot collide, not to detect a changed one — in production it is derived
/// from a frozen manifest and is constant across successful loads. Detection
/// of a changed model is `admit_registered_potion`'s job, and it runs before
/// this key is ever looked up.
#[derive(Clone, PartialEq, Eq, Hash)]
struct SharedModelKey {
    dir: PathBuf,
    name: String,
    identity: String,
}

type SharedModelCache = Mutex<HashMap<SharedModelKey, Weak<Model2VecEmbedder>>>;

fn shared_model_cache() -> &'static SharedModelCache {
    static CACHE: OnceLock<SharedModelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Search for a `Model2Vec` model directory in standard locations.
///
/// Checks these paths in order:
/// 1. `$FRANKENSEARCH_MODEL_DIR/<model_name>/`
/// 2. `$XDG_DATA_HOME/frankensearch/models/<model_name>/`
/// 3. `~/.local/share/frankensearch/models/<model_name>/` (or macOS
///    `~/Library/Application Support/frankensearch/models/<model_name>/`)
/// 4. `~/.cache/huggingface/hub/models--<hf_id>/snapshots/*/`
///
/// Returns `None` if no directory with the required files is found.
#[must_use]
pub fn find_model_dir(model_name: &str) -> Option<PathBuf> {
    find_model_dir_with_hf_id(model_name, DEFAULT_HF_ID)
}

/// Search for a `Model2Vec` model directory with a specific `HuggingFace` ID.
#[must_use]
pub fn find_model_dir_with_hf_id(model_name: &str, hf_id: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();

    // 1. Explicit env var override
    if let Ok(dir) = std::env::var("FRANKENSEARCH_MODEL_DIR") {
        let base = PathBuf::from(dir);
        for variant in model_directory_variants(model_name) {
            candidates.push(base.join(variant));
        }
        candidates.push(base);
    }

    // 2-3. Standard frankensearch model layout (created on first access)
    let model_root = ensure_model_storage_layout();
    for variant in model_directory_variants(model_name) {
        candidates.push(model_root.join(variant));
    }

    // 4. HuggingFace cache
    if let Some(cache_dir) = frankensearch_core::platform_dirs::cache_dir() {
        let hf_dir = cache_dir
            .join("huggingface/hub")
            .join(format!("models--{}", hf_id.replace('/', "--")));
        if let Ok(snapshots) = std::fs::read_dir(hf_dir.join("snapshots")) {
            for entry in snapshots.flatten() {
                candidates.push(entry.path());
            }
        }
    }

    // Check each candidate for required files
    for candidate in &candidates {
        if has_required_files(candidate) {
            return Some(candidate.clone());
        }
    }

    None
}

/// Check if a directory contains all required `Model2Vec` files.
fn has_required_files(dir: &Path) -> bool {
    REQUIRED_FILES.iter().all(|f| dir.join(f).exists())
}

#[cfg(test)]
mod tests;
