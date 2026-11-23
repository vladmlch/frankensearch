//! Caching wrapper for any [`Embedder`] implementation.
//!
//! `CachedEmbedder` sits between the search pipeline and an inner embedder,
//! caching repeated raw and identity-bound query embeddings separately. Bound
//! entries retain the provider's actual complete response and are keyed by its
//! independently admitted identity and the query. Raw entries can never supply
//! bound requests. Bound batches bypass the cache and are never deduplicated.
//!
//! The cache uses FIFO eviction with a bounded capacity (default 128 entries).
//! Cache hits return a cloned `Vec<f32>`, which is cheap (~1.5 KiB for 384-dim).
//!
//! # Thread Safety
//!
//! The cache is protected by a `std::sync::Mutex`, keeping the wrapper `Send + Sync`.
//! The lock is held only for the brief `HashMap` lookup/insert — never across an
//! async `.await` boundary. Clearing replaces the admission epoch, so inference
//! started before a clear cannot refill or evict entries in the new cache.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use asupersync::Cx;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{
    Embedder, IdentityBoundEmbedding, ModelCategory, ModelTier, SearchFuture,
};
use frankensearch_core::{QuantizationFormat, SearchError, SearchResult};

mod state;

pub use state::CacheStats;
use state::{BoundCacheKey, CacheState};

/// Default maximum number of cached query embeddings.
const DEFAULT_CAPACITY: usize = 128;

fn cache_checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|_| SearchError::Cancelled {
        phase: "embedding.cache".to_owned(),
        reason: "embedding request cancelled".to_owned(),
    })
}

/// Caching wrapper around any [`Embedder`].
///
/// Caches raw `embed()` and `embed_batch()` results for previously seen queries,
/// and `embed_bound()` responses under their complete producer/input identity.
/// Raw and bound entries are separate, sharing one total capacity and statistics.
/// Every bound lookup admits the current identity and native output contract;
/// hits and fills require the producer to retain that identity. A response's
/// identity must exactly match, and its actual values and identity are retained.
/// Bound batches retain the original input order and duplicate slots and invoke
/// the provider's bound batch operation once, even for empty input. Responses
/// are validated all-or-error without relabeling; admission against a stored
/// generation remains the caller's responsibility. Bound batches do not affect
/// cache statistics or contents, whether they succeed or fail.
///
/// Raw responses are validated before insertion: batches must contain exactly
/// one finite, correctly sized vector per distinct miss. An invalid batch never
/// inserts a valid-looking prefix into the cache. Cancellation after inference
/// takes precedence over either a successful response or a provider error.
///
/// # Construction
///
/// ```ignore
/// use frankensearch_embed::CachedEmbedder;
///
/// let inner: Arc<dyn Embedder> = /* ... */;
/// let cached = CachedEmbedder::new(inner, 128);
/// ```
pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    state: Mutex<CacheState>,
}

impl std::fmt::Debug for CachedEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.cache_stats();
        f.debug_struct("CachedEmbedder")
            .field("inner_id", &self.inner.id())
            .field("hits", &stats.hits)
            .field("misses", &stats.misses)
            .field("entries", &stats.entries)
            .field("capacity", &stats.capacity)
            .finish_non_exhaustive()
    }
}

impl CachedEmbedder {
    /// Wrap an embedder with a bounded query cache.
    ///
    /// `capacity` controls the maximum number of cached embeddings before
    /// FIFO eviction begins.
    #[must_use]
    pub fn new(inner: Arc<dyn Embedder>, capacity: usize) -> Self {
        Self {
            inner,
            state: Mutex::new(CacheState::new(capacity)),
        }
    }

    /// Wrap an embedder with the default capacity (128 entries).
    #[must_use]
    pub fn with_default_capacity(inner: Arc<dyn Embedder>) -> Self {
        Self::new(inner, DEFAULT_CAPACITY)
    }

    fn state_lock(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn invalid_response(&self, detail: impl Into<String>) -> SearchError {
        SearchError::EmbeddingFailed {
            model: self.inner.id().to_owned(),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                detail.into(),
            )),
        }
    }

    fn validate_bound_response(response: &IdentityBoundEmbedding) -> SearchResult<()> {
        response.validate().map_err(|_| SearchError::InvalidConfig {
            field: "embedding.cache.bound_response".to_owned(),
            value: String::new(),
            reason: "provider returned an invalid identity, representation, dimension or non-finite vector"
                .to_owned(),
        })
    }

    fn bound_identity_error() -> SearchError {
        SearchError::UnverifiableRemoteSpace {
            producer: "embedding.cache.bound".to_owned(),
            reason:
                "producer or response does not retain the admitted complete native-f32 identity"
                    .to_owned(),
        }
    }

    fn capture_bound_identity(&self) -> SearchResult<EmbeddingIdentityBundleV1> {
        let identity = self
            .inner
            .identity()
            .map_err(|_| Self::bound_identity_error())?
            .clone();
        if identity.validate().is_err()
            || usize::try_from(identity.space.dimension).ok() != Some(self.inner.dimension())
            || identity.storage.quantization != QuantizationFormat::F32
            || !identity.storage.format.starts_with("in-memory-")
            || !matches!(
                identity.storage.endianness.as_str(),
                "native-f32-values" | "native-test-only"
            )
        {
            return Err(Self::bound_identity_error());
        }
        Ok(identity)
    }

    fn validate_bound_producer(&self, expected: &EmbeddingIdentityBundleV1) -> SearchResult<()> {
        if self
            .inner
            .identity()
            .is_ok_and(|identity| identity == expected)
            && usize::try_from(expected.space.dimension).ok() == Some(self.inner.dimension())
        {
            Ok(())
        } else {
            Err(Self::bound_identity_error())
        }
    }

    fn validate_vector(&self, values: &[f32]) -> SearchResult<()> {
        let expected = self.inner.dimension();
        if values.len() != expected {
            return Err(SearchError::DimensionMismatch {
                expected,
                found: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            // Do not echo query text, vector values or a provider's raw payload.
            return Err(self.invalid_response("embedding response contains non-finite values"));
        }
        Ok(())
    }

    fn validate_batch(&self, embedded: &[Vec<f32>], expected: usize) -> SearchResult<()> {
        // Validate the entire response before indexing it or mutating the cache.
        // Short and oversized batches used to panic after caching a partial prefix.
        if embedded.len() != expected {
            return Err(self.invalid_response(format!(
                "embedding batch returned {} vectors for {expected} distinct inputs",
                embedded.len(),
            )));
        }
        for values in embedded {
            self.validate_vector(values)?;
        }
        Ok(())
    }

    fn fill_batch_misses(
        out: &mut [Option<Vec<f32>>],
        slot_miss: &[Option<usize>],
        embedded: Vec<Vec<f32>>,
    ) {
        // Cardinality was checked before admission. Move each result into its
        // last output slot, cloning only for same-batch duplicate queries.
        let mut miss_use_counts = vec![0_usize; embedded.len()];
        for maybe_idx in slot_miss {
            if let Some(idx) = *maybe_idx {
                miss_use_counts[idx] += 1;
            }
        }
        let mut embedded_slots: Vec<Option<Vec<f32>>> = embedded.into_iter().map(Some).collect();
        for (slot, maybe_idx) in out.iter_mut().zip(slot_miss) {
            if let Some(idx) = *maybe_idx {
                let use_count = &mut miss_use_counts[idx];
                *use_count = use_count.saturating_sub(1);
                let vec = if *use_count == 0 {
                    embedded_slots[idx].take().expect("embedding slot filled")
                } else {
                    embedded_slots[idx]
                        .as_ref()
                        .expect("embedding slot filled")
                        .clone()
                };
                *slot = Some(vec);
            }
        }
    }

    /// Return a snapshot of cache statistics.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.state_lock().stats()
    }

    /// Clear all cached embeddings and reset statistics.
    ///
    /// This also fences admission by already-running requests. They may return
    /// their validated results to their original callers, but cannot refill the
    /// cleared cache, change its reset statistics or evict newer entries.
    pub fn clear_cache(&self) {
        self.state_lock().clear();
    }

    /// Reference to the inner embedder.
    #[must_use]
    pub fn inner(&self) -> &dyn Embedder {
        &*self.inner
    }
}

impl Embedder for CachedEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            // Resolve lazily: constructing an unpolled future must not capture
            // a stale cache hit, update statistics or begin provider work.
            cache_checkpoint(cx)?;
            let epoch = {
                let mut cache = self.state_lock();
                cache_checkpoint(cx)?;
                if let Some(vec) = cache.get(text) {
                    return Ok(vec);
                }
                Arc::clone(&cache.epoch)
            };
            let outcome = self.inner.embed(cx, text).await;
            cache_checkpoint(cx)?;
            let vec = outcome?;
            self.validate_vector(&vec)?;
            {
                let mut cache = self.state_lock();
                cache_checkpoint(cx)?;
                if Arc::ptr_eq(&epoch, &cache.epoch) {
                    cache.insert(text.to_owned(), vec.clone());
                }
            }
            Ok(vec)
        })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            cache_checkpoint(cx)?;
            let identity = self.capture_bound_identity();
            cache_checkpoint(cx)?;
            let key = BoundCacheKey {
                identity: Arc::new(identity?),
                text: text.to_owned(),
            };
            let (cached, epoch) = {
                let mut cache = self.state_lock();
                cache_checkpoint(cx)?;
                (cache.lookup_bound(&key), Arc::clone(&cache.epoch))
            };
            if let Some(response) = cached {
                // Provider callbacks run outside the mutex: even identity() may
                // reenter clear_cache(). Its epoch fences any deferred hit count.
                let producer = self.validate_bound_producer(&key.identity);
                cache_checkpoint(cx)?;
                producer?;
                {
                    let mut cache = self.state_lock();
                    cache_checkpoint(cx)?;
                    if Arc::ptr_eq(&epoch, &cache.epoch) {
                        cache.record_bound_hit(&key);
                    }
                }
                cache_checkpoint(cx)?;
                return Ok(response);
            }
            let outcome = self.inner.embed_bound(cx, text).await;
            cache_checkpoint(cx)?;
            let producer = self.validate_bound_producer(&key.identity);
            cache_checkpoint(cx)?;
            producer?;
            let response = outcome?;
            // Compare before diagnostics inspect provider-controlled fields.
            // Equal names, dimensions, or mathematical spaces are insufficient.
            if response.identity != *key.identity {
                return Err(Self::bound_identity_error());
            }
            Self::validate_bound_response(&response)?;
            {
                let mut cache = self.state_lock();
                cache_checkpoint(cx)?;
                if Arc::ptr_eq(&epoch, &cache.epoch) {
                    cache.insert_bound(key, response.clone());
                }
            }
            cache_checkpoint(cx)?;
            Ok(response)
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            cache_checkpoint(cx)?;
            // Never route this through the raw cache or deduplicate its inputs.
            // The batch provider owns response identity and per-input ordering.
            let outcome = self.inner.embed_batch_bound(cx, texts).await;
            cache_checkpoint(cx)?;
            let responses = outcome?;
            if responses.len() != texts.len() {
                return Err(SearchError::InvalidConfig {
                    field: "embedder.batch_length".to_owned(),
                    value: responses.len().to_string(),
                    reason: format!(
                        "expected exactly {} identity-bound output vectors, one per input",
                        texts.len(),
                    ),
                });
            }
            if let Some(first) = responses.first() {
                for response in &responses {
                    cache_checkpoint(cx)?;
                    Self::validate_bound_response(response)?;
                    if response.identity != first.identity {
                        return Err(SearchError::UnverifiableRemoteSpace {
                            producer: "embedding.cache.bound_batch".to_owned(),
                            reason: "a bound embedding batch must carry one producing identity"
                                .to_owned(),
                        });
                    }
                }
            }
            cache_checkpoint(cx)?;
            Ok(responses)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            cache_checkpoint(cx)?;
            // Pass 1 (single lock scope, released BEFORE the await so the batched
            // inner call holds no lock): resolve cache hits and collect the *distinct*
            // misses. A miss text repeated within the batch folds onto one inner
            // embedding and counts as a hit — the in-batch dedup the old per-text loop
            // got for free — so a batch of repeated queries embeds each text at most once.
            let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
            let mut miss_texts: Vec<&str> = Vec::new();
            // Output slot -> index into `miss_texts` (None once the slot is resolved).
            let mut slot_miss: Vec<Option<usize>> = Vec::with_capacity(texts.len());
            // Dedup map: distinct miss text -> its index in `miss_texts`.
            let mut miss_index: HashMap<&str, usize> = HashMap::new();
            let epoch = {
                let mut cache = self.state_lock();
                for &text in texts {
                    cache_checkpoint(cx)?;
                    if let Some(&idx) = miss_index.get(text) {
                        // Repeat of a text already queued this batch: fold onto the
                        // same inner result, no second inner call, record a hit.
                        cache.record_hit();
                        out.push(None);
                        slot_miss.push(Some(idx));
                        continue;
                    }
                    match cache.get(text) {
                        Some(vec) => {
                            out.push(Some(vec));
                            slot_miss.push(None);
                        }
                        None => {
                            let idx = miss_texts.len();
                            miss_texts.push(text);
                            miss_index.insert(text, idx);
                            out.push(None);
                            slot_miss.push(Some(idx));
                        }
                    }
                }
                Arc::clone(&cache.epoch)
            };
            // ONE batched inner call for all distinct misses — vs the old per-text loop
            // that called `inner.embed` N times, defeating a batching inner (e.g.
            // fastembed embeds the whole batch in a single model invocation). N → 1.
            if !miss_texts.is_empty() {
                cache_checkpoint(cx)?;
                let all_slots_are_distinct_misses = miss_texts.len() == texts.len();
                let outcome = self.inner.embed_batch(cx, &miss_texts).await;
                cache_checkpoint(cx)?;
                let embedded = outcome?;
                self.validate_batch(&embedded, miss_texts.len())?;
                {
                    let mut cache = self.state_lock();
                    cache_checkpoint(cx)?;
                    if Arc::ptr_eq(&epoch, &cache.epoch) {
                        for (idx, vec) in embedded.iter().enumerate() {
                            cache.insert(miss_texts[idx].to_owned(), vec.clone());
                        }
                    }
                }
                if all_slots_are_distinct_misses {
                    return Ok(embedded);
                }
                Self::fill_batch_misses(&mut out, &slot_miss, embedded);
            }
            cache_checkpoint(cx)?;
            Ok(out
                .into_iter()
                .map(|v| v.expect("every slot filled"))
                .collect())
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.inner.identity()
    }

    /// The cache serves bound batches from the inner producer's bound batch,
    /// so it is as faithful to single inputs as that producer is.
    fn bound_batch_is_native(&self) -> bool {
        self.inner.bound_batch_is_native()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> ModelCategory {
        self.inner.category()
    }

    fn tier(&self) -> ModelTier {
        self.inner.tier()
    }

    fn supports_mrl(&self) -> bool {
        self.inner.supports_mrl()
    }
}

#[cfg(test)]
mod tests;
