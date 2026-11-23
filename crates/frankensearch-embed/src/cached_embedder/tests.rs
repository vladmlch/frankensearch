use std::sync::atomic::{AtomicUsize, Ordering};

use frankensearch_core::traits::l2_normalize;

use super::*;

mod bound_cache;
mod response_and_fencing;

/// Test double: counts how many times `embed()` is called.
struct CountingEmbedder {
    dim: usize,
    calls: AtomicUsize,
    identity: EmbeddingIdentityBundleV1,
    bound_calls: AtomicUsize,
    bound_response_identity: Option<EmbeddingIdentityBundleV1>,
}

impl CountingEmbedder {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            calls: AtomicUsize::new(0),
            identity: EmbeddingIdentityBundleV1::explicit_test_model(
                "counting-test",
                u32::try_from(dim).unwrap_or(u32::MAX),
            ),
            bound_calls: AtomicUsize::new(0),
            bound_response_identity: None,
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Embedder for CountingEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut vec = vec![0.0_f32; self.dim];
        // Deterministic: use text length to seed a simple pattern
        for (i, b) in text.bytes().enumerate() {
            vec[i % self.dim] += f32::from(b);
        }
        let normalized = l2_normalize(&vec);
        Box::pin(async move { Ok(normalized) })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        self.bound_calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            let bound = IdentityBoundEmbedding {
                values: self.embed(cx, text).await?,
                identity: self
                    .bound_response_identity
                    .as_ref()
                    .unwrap_or(&self.identity)
                    .clone(),
            };
            bound.validate()?;
            Ok(bound)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &'static str {
        "counting-test"
    }

    fn model_name(&self) -> &'static str {
        "Counting Test Embedder"
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

/// Counts `embed` vs `embed_batch` inner calls separately, so a test can prove
/// `CachedEmbedder::embed_batch` funnels misses through ONE `inner.embed_batch`
/// (not N `inner.embed`), which is the whole point when the inner batches.
struct BatchCountingEmbedder {
    dim: usize,
    embed_calls: AtomicUsize,
    batch_calls: AtomicUsize,
    identity: EmbeddingIdentityBundleV1,
}

impl BatchCountingEmbedder {
    fn deterministic(dim: usize, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0_f32; dim];
        for (i, b) in text.bytes().enumerate() {
            vec[i % dim] += f32::from(b);
        }
        l2_normalize(&vec)
    }
}

impl Embedder for BatchCountingEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.embed_calls.fetch_add(1, Ordering::Relaxed);
        let v = Self::deterministic(self.dim, text);
        Box::pin(async move { Ok(v) })
    }

    fn embed_batch<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        let out: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| Self::deterministic(self.dim, t))
            .collect();
        Box::pin(async move { Ok(out) })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
    fn id(&self) -> &'static str {
        "batch-counting-test"
    }
    fn model_name(&self) -> &'static str {
        "Batch Counting Test Embedder"
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

#[test]
fn embed_batch_funnels_misses_through_one_inner_embed_batch() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(BatchCountingEmbedder {
            dim: 64,
            embed_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
            identity: EmbeddingIdentityBundleV1::explicit_test_model("batch-counting-test", 64),
        });
        let cached = CachedEmbedder::new(inner.clone(), 128);
        let texts = ["a", "bb", "ccc", "dddd", "eeeee"];
        let refs: Vec<&str> = texts.to_vec();

        let out = cached.embed_batch(&cx, &refs).await.expect("embed_batch");

        // The win: all misses go through ONE inner.embed_batch, zero inner.embed.
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 1);
        assert_eq!(inner.embed_calls.load(Ordering::Relaxed), 0);
        // Correctness: same vectors the direct (uncached) embed_batch produces.
        for (i, t) in texts.iter().enumerate() {
            assert_eq!(out[i], BatchCountingEmbedder::deterministic(64, t));
        }

        // A second call is fully cache-served: no further inner work.
        let out2 = cached.embed_batch(&cx, &refs).await.expect("embed_batch 2");
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 1);
        assert_eq!(inner.embed_calls.load(Ordering::Relaxed), 0);
        assert_eq!(out, out2);
    });
}

fn make_cached(capacity: usize) -> (CachedEmbedder, Arc<CountingEmbedder>) {
    let inner = Arc::new(CountingEmbedder::new(64));
    let cached = CachedEmbedder::new(inner.clone(), capacity);
    (cached, inner)
}

#[test]
fn cache_wrapper_forwards_complete_identity_exactly() {
    let (cached, inner) = make_cached(16);
    assert_eq!(cached.identity().unwrap(), inner.identity().unwrap());
}

#[test]
fn bound_embedding_bypasses_warm_raw_cache_and_rejects_foreign_response_identity() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let response_identity =
            EmbeddingIdentityBundleV1::explicit_test_model("foreign-response", 64);
        let mut inner = CountingEmbedder::new(64);
        inner.bound_response_identity = Some(response_identity.clone());
        let inner = Arc::new(inner);
        let cached = CachedEmbedder::new(inner.clone(), 16);
        assert_ne!(cached.identity().unwrap(), &response_identity);

        let raw = cached.embed(&cx, "query").await.unwrap();
        assert_eq!(cached.embed(&cx, "query").await.unwrap(), raw);
        assert_eq!(inner.call_count(), 1);
        let warmed_stats = cached.cache_stats();

        for _ in 0..2 {
            assert!(matches!(
                cached.embed_bound(&cx, "query").await,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
        }
        assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
        assert_eq!(inner.call_count(), 3);
        assert_eq!(cached.cache_stats().entries, warmed_stats.entries);
        assert_eq!(cached.cache_stats().hits, warmed_stats.hits);
        assert_eq!(cached.cache_stats().misses, warmed_stats.misses + 2);
        assert!(cached.state_lock().bound_entries.is_empty());

        assert_eq!(cached.embed(&cx, "query").await.unwrap(), raw);
        assert_eq!(inner.call_count(), 3);
        assert_eq!(cached.cache_stats().hits, warmed_stats.hits + 1);
    });
}

#[test]
fn cache_hit_avoids_inner_call() {
    let (cached, inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let v1 = cached.embed(&cx, "hello world").await.unwrap();
        let v2 = cached.embed(&cx, "hello world").await.unwrap();
        assert_eq!(v1, v2);
        assert_eq!(inner.call_count(), 1);
    });
}

#[test]
fn cache_miss_calls_inner() {
    let (cached, inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "query a").await.unwrap();
        cached.embed(&cx, "query b").await.unwrap();
        assert_eq!(inner.call_count(), 2);
    });
}

#[test]
fn stats_track_hits_and_misses() {
    let (cached, _inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "alpha").await.unwrap();
        cached.embed(&cx, "alpha").await.unwrap();
        cached.embed(&cx, "beta").await.unwrap();
        let stats = cached.cache_stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.entries, 2);
    });
}

#[test]
fn eviction_at_capacity() {
    let (cached, inner) = make_cached(2);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "first").await.unwrap();
        cached.embed(&cx, "second").await.unwrap();
        // Cache full (2 entries), all freq 0 → S3-FIFO evicts the oldest Small
        // entry ("first") on the third insert.
        cached.embed(&cx, "third").await.unwrap();
        assert_eq!(inner.call_count(), 3);

        // "first" was evicted → miss; re-inserting it evicts "second".
        cached.embed(&cx, "first").await.unwrap();
        assert_eq!(inner.call_count(), 4);

        // "third" is still cached → hit (no inner call).
        cached.embed(&cx, "third").await.unwrap();
        assert_eq!(inner.call_count(), 4);
    });
}

#[test]
fn s3fifo_keeps_reused_key_through_scan() {
    // The S3-FIFO win over plain FIFO: a key re-requested while resident is
    // promoted to Main and survives a scan of cold one-hit-wonders that overflows
    // the cache — a FIFO would have evicted it by insertion order.
    let (cached, inner) = make_cached(4);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "hot").await.unwrap(); // miss → Small
        cached.embed(&cx, "hot").await.unwrap(); // hit → freq++ (promotable)
        assert_eq!(inner.call_count(), 1);
        let after_hot = inner.call_count();

        // Scan: 6 unique cold keys through a capacity-4 cache.
        for i in 0..6 {
            cached.embed(&cx, &format!("cold-{i}")).await.unwrap();
        }
        assert_eq!(inner.call_count(), after_hot + 6); // all cold = misses

        // "hot" survived the scan (promoted to Main) → still a hit.
        cached.embed(&cx, "hot").await.unwrap();
        assert_eq!(
            inner.call_count(),
            after_hot + 6,
            "S3-FIFO must keep the reused 'hot' key through the cold scan"
        );
    });
}

#[test]
fn clear_resets_stats_and_entries() {
    let (cached, _inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "test").await.unwrap();
        assert_eq!(cached.cache_stats().entries, 1);

        cached.clear_cache();
        let stats = cached.cache_stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    });
}

#[test]
fn delegates_trait_methods_to_inner() {
    let inner = Arc::new(CountingEmbedder::new(64));
    let cached = CachedEmbedder::new(inner, 16);

    assert_eq!(cached.dimension(), 64);
    assert_eq!(cached.id(), "counting-test");
    assert_eq!(cached.model_name(), "Counting Test Embedder");
    assert!(!cached.is_semantic());
    assert_eq!(cached.category(), ModelCategory::HashEmbedder);
}

#[test]
fn with_default_capacity_uses_128() {
    let inner = Arc::new(CountingEmbedder::new(64));
    let cached = CachedEmbedder::with_default_capacity(inner);
    assert_eq!(cached.cache_stats().capacity, 128);
}

#[test]
fn debug_format_includes_stats() {
    let inner = Arc::new(CountingEmbedder::new(64));
    let cached = CachedEmbedder::new(inner, 16);
    let dbg = format!("{cached:?}");
    assert!(dbg.contains("CachedEmbedder"));
    assert!(dbg.contains("counting-test"));
}

#[test]
fn embed_batch_uses_per_item_cache() {
    let (cached, inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // Pre-warm "alpha" into cache
        cached.embed(&cx, "alpha").await.unwrap();
        assert_eq!(inner.call_count(), 1);

        // Batch with "alpha" (cached) and "beta" (miss)
        let batch = cached.embed_batch(&cx, &["alpha", "beta"]).await.unwrap();
        assert_eq!(batch.len(), 2);
        // Only "beta" should have triggered an inner call
        assert_eq!(inner.call_count(), 2);
    });
}

#[test]
fn duplicate_insert_is_idempotent() {
    let (cached, inner) = make_cached(4);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "same").await.unwrap();
        assert_eq!(inner.call_count(), 1);
        assert_eq!(cached.cache_stats().entries, 1);
        // Re-embed same query — should be a cache hit, not a duplicate insert
        cached.embed(&cx, "same").await.unwrap();
        assert_eq!(inner.call_count(), 1);
        assert_eq!(cached.cache_stats().entries, 1);
    });
}

// ─── bd-1ocg tests begin ───

#[test]
fn cache_stats_debug_clone_copy_eq() {
    let stats = CacheStats {
        hits: 5,
        misses: 3,
        entries: 8,
        capacity: 128,
    };
    let copied = stats; // Copy
    let cloned = { stats }; // Clone trait is available (Copy implies Clone)
    assert_eq!(stats, copied);
    assert_eq!(stats, cloned);

    let different = CacheStats {
        hits: 0,
        misses: 0,
        entries: 0,
        capacity: 128,
    };
    assert_ne!(stats, different);

    let dbg = format!("{stats:?}");
    assert!(dbg.contains("CacheStats"));
    assert!(dbg.contains("hits: 5"));
}

#[test]
fn capacity_one_evicts_immediately() {
    let (cached, inner) = make_cached(1);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "first").await.unwrap();
        assert_eq!(cached.cache_stats().entries, 1);

        // Second insert evicts "first"
        cached.embed(&cx, "second").await.unwrap();
        assert_eq!(inner.call_count(), 2);
        assert_eq!(cached.cache_stats().entries, 1);

        // "first" is evicted, so it's a miss
        cached.embed(&cx, "first").await.unwrap();
        assert_eq!(inner.call_count(), 3);

        // "second" was evicted by "first" re-insert
        cached.embed(&cx, "second").await.unwrap();
        assert_eq!(inner.call_count(), 4);
    });
}

#[test]
fn inner_accessor_returns_same_embedder() {
    let inner = Arc::new(CountingEmbedder::new(64));
    let cached = CachedEmbedder::new(inner, 16);
    assert_eq!(cached.inner().id(), "counting-test");
    assert_eq!(cached.inner().dimension(), 64);
    assert_eq!(cached.inner().model_name(), "Counting Test Embedder");
}

#[test]
fn is_ready_delegates() {
    let inner = Arc::new(CountingEmbedder::new(64));
    let cached = CachedEmbedder::new(inner, 16);
    // CountingEmbedder uses default is_ready() which returns true
    assert!(cached.is_ready());
}

#[test]
fn tier_delegates() {
    let inner = Arc::new(CountingEmbedder::new(64));
    let cached = CachedEmbedder::new(inner, 16);
    // CountingEmbedder uses default tier() which returns ModelTier::Fast
    assert_eq!(cached.tier(), ModelTier::Fast);
}

#[test]
fn supports_mrl_delegates() {
    let inner = Arc::new(CountingEmbedder::new(64));
    let cached = CachedEmbedder::new(inner, 16);
    // CountingEmbedder uses default supports_mrl() which returns false
    assert!(!cached.supports_mrl());
}

#[test]
fn clear_then_reuse_resets_everything() {
    let (cached, inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "alpha").await.unwrap();
        cached.embed(&cx, "alpha").await.unwrap(); // hit
        assert_eq!(cached.cache_stats().hits, 1);
        assert_eq!(cached.cache_stats().misses, 1);

        cached.clear_cache();
        assert_eq!(cached.cache_stats().hits, 0);
        assert_eq!(cached.cache_stats().misses, 0);
        assert_eq!(cached.cache_stats().entries, 0);

        // After clear, "alpha" is a miss again
        cached.embed(&cx, "alpha").await.unwrap();
        assert_eq!(inner.call_count(), 2); // called again
        assert_eq!(cached.cache_stats().misses, 1);
        assert_eq!(cached.cache_stats().entries, 1);
    });
}

#[test]
fn sequential_evictions_maintain_fifo_order() {
    let (cached, inner) = make_cached(3);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // Fill cache: a, b, c
        cached.embed(&cx, "a").await.unwrap();
        cached.embed(&cx, "b").await.unwrap();
        cached.embed(&cx, "c").await.unwrap();
        assert_eq!(inner.call_count(), 3);
        assert_eq!(cached.cache_stats().entries, 3);

        // Insert d -> evicts a (FIFO)
        cached.embed(&cx, "d").await.unwrap();
        assert_eq!(inner.call_count(), 4);

        // a is evicted (miss), b is still cached (hit)
        cached.embed(&cx, "a").await.unwrap();
        assert_eq!(inner.call_count(), 5); // miss
        cached.embed(&cx, "b").await.unwrap();
        // b was evicted when d was added (b was 2nd oldest after a was evicted,
        // then a was re-added evicting b)
        // Actually let's check: after d inserted, cache = [b, c, d]
        // Then a inserted -> evicts b, cache = [c, d, a]
        // So b should be a miss
        assert_eq!(inner.call_count(), 6); // b is a miss
    });
}

#[test]
fn empty_string_embedding() {
    let (cached, inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let v1 = cached.embed(&cx, "").await.unwrap();
        let v2 = cached.embed(&cx, "").await.unwrap();
        assert_eq!(v1, v2);
        assert_eq!(inner.call_count(), 1); // second is cache hit
    });
}

#[test]
fn stats_entries_accurate_after_evictions() {
    let (cached, _inner) = make_cached(2);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "x").await.unwrap();
        cached.embed(&cx, "y").await.unwrap();
        assert_eq!(cached.cache_stats().entries, 2);

        // Evict x, add z
        cached.embed(&cx, "z").await.unwrap();
        assert_eq!(cached.cache_stats().entries, 2); // stays at capacity

        // Evict y, add w
        cached.embed(&cx, "w").await.unwrap();
        assert_eq!(cached.cache_stats().entries, 2);
    });
}

#[test]
fn debug_format_after_operations() {
    let (cached, _inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        cached.embed(&cx, "test").await.unwrap();
        cached.embed(&cx, "test").await.unwrap(); // hit
        let dbg = format!("{cached:?}");
        assert!(dbg.contains("hits"));
        assert!(dbg.contains("misses"));
        assert!(dbg.contains("entries"));
        assert!(dbg.contains("capacity"));
    });
}

#[test]
fn embed_batch_empty_input() {
    let (cached, _inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let empty: &[&str] = &[];
        let result = cached.embed_batch(&cx, empty).await.unwrap();
        assert!(result.is_empty());
        assert_eq!(cached.cache_stats().entries, 0);
    });
}

#[test]
fn embed_batch_deduplicates_within_batch() {
    let (cached, inner) = make_cached(16);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // Batch with duplicate items: "hello" appears twice, "world" once
        let batch = cached
            .embed_batch(&cx, &["hello", "hello", "world"])
            .await
            .unwrap();
        assert_eq!(batch.len(), 3);
        // Only 2 unique texts → 2 inner calls (second "hello" hits cache)
        assert_eq!(inner.call_count(), 2);
        // Both "hello" embeddings should be identical
        assert_eq!(batch[0], batch[1]);
        // "world" should differ
        assert_ne!(batch[0], batch[2]);
        // Stats: 1 hit (second "hello"), 2 misses (first "hello" + "world")
        assert_eq!(cached.cache_stats().hits, 1);
        assert_eq!(cached.cache_stats().misses, 2);
    });
}

// ─── bd-1ocg tests end ───
