use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use asupersync::Cx;
use frankensearch_core::generation::{
    EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
};
use frankensearch_core::traits::{Embedder, IdentityBoundEmbedding, ModelCategory, SearchFuture};
use frankensearch_core::{QuantizationFormat, SearchError, SearchResult};

use super::super::{CacheStats, CachedEmbedder};

type IdentityHook = (usize, Box<dyn FnOnce() + Send>);

/// A real suspending bound operation with values independent of the raw API.
/// Immutable bundles are selected atomically to model a reload at any boundary.
struct BoundCacheProvider {
    identities: [EmbeddingIdentityBundleV1; 2],
    selected: AtomicUsize,
    dimension: AtomicUsize,
    unavailable: AtomicBool,
    raw_calls: AtomicUsize,
    bound_calls: AtomicUsize,
    identity_calls: AtomicUsize,
    identity_hook: Mutex<Option<IdentityHook>>,
    values: Mutex<Vec<f32>>,
    response_identity: Mutex<Option<EmbeddingIdentityBundleV1>>,
    // 0: success, 1: error, 2: cancelled success, 3: cancelled error.
    action: AtomicUsize,
}

impl BoundCacheProvider {
    fn new() -> Self {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("bound-query-test", 2);
        // Synthetic semantic authority permits an explicit weights-revision
        // mutation; this fixture makes no claim of real model verification.
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifacts.push(EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "c".repeat(64),
            size: 1,
        });
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        let mut replacement = identity.clone();
        replacement
            .producer
            .implementation_revision
            .push_str("-new");
        replacement.validate().unwrap();
        Self {
            identities: [identity, replacement],
            selected: AtomicUsize::new(0),
            dimension: AtomicUsize::new(2),
            unavailable: AtomicBool::new(false),
            raw_calls: AtomicUsize::new(0),
            bound_calls: AtomicUsize::new(0),
            identity_calls: AtomicUsize::new(0),
            identity_hook: Mutex::new(None),
            values: Mutex::new(vec![-0.0, f32::from_bits(1)]),
            response_identity: Mutex::new(None),
            action: AtomicUsize::new(0),
        }
    }

    fn at_identity_read(&self, read: usize, action: impl FnOnce() + Send + 'static) {
        *self.identity_hook.lock().unwrap() = Some((read, Box::new(action)));
    }
}

impl Embedder for BoundCacheProvider {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![99.0, 0.0])
        })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        _text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.bound_calls.fetch_add(1, Ordering::Relaxed);
            let response = IdentityBoundEmbedding {
                values: self.values.lock().unwrap().clone(),
                identity: self
                    .response_identity
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| {
                        self.identities[self.selected.load(Ordering::Relaxed)].clone()
                    }),
            };
            let action = self.action.load(Ordering::Relaxed);
            let mut yielded = false;
            std::future::poll_fn(|task| {
                if yielded {
                    std::task::Poll::Ready(())
                } else {
                    yielded = true;
                    task.waker().wake_by_ref();
                    std::task::Poll::Pending
                }
            })
            .await;
            if action >= 2 {
                cx.set_cancel_requested(true);
            }
            if action % 2 == 1 {
                return Err(SearchError::EmbeddingFailed {
                    model: "bound-query-test".to_owned(),
                    source: "bound provider failed".into(),
                });
            }
            Ok(response)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        let read = self.identity_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let hook = {
            let mut hook = self.identity_hook.lock().unwrap();
            if hook.as_ref().is_some_and(|(at, _)| *at == read) {
                hook.take()
            } else {
                None
            }
        };
        if let Some((_, hook)) = hook {
            hook();
        }
        if self.unavailable.load(Ordering::Relaxed) {
            return Err(SearchError::InvalidConfig {
                field: "private-canary".to_owned(),
                value: "private-canary".to_owned(),
                reason: "private-canary".to_owned(),
            });
        }
        Ok(&self.identities[self.selected.load(Ordering::Relaxed)])
    }

    fn dimension(&self) -> usize {
        self.dimension.load(Ordering::Relaxed)
    }

    fn id(&self) -> &'static str {
        "bound-query-test"
    }

    fn model_name(&self) -> &'static str {
        "Bound Query Test Provider"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

#[test]
fn bound_query_cache_retains_actual_bits_and_cannot_alias_raw_queries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(BoundCacheProvider::new());
        let cached = CachedEmbedder::new(inner.clone(), 4);
        // NULs and identity-looking text are still ordinary raw queries.
        let query = format!("\0bound:{}:query", inner.identities[0].fingerprint());
        assert_eq!(cached.embed(&cx, &query).await.unwrap(), vec![99.0, 0.0]);
        let response = cached.embed_bound(&cx, &query).await.unwrap();
        *inner.values.lock().unwrap() = vec![1.0, 0.0];
        let hit = cached.embed_bound(&cx, &query).await.unwrap();
        assert_eq!(hit.identity, response.identity);
        assert_eq!(hit.identity, inner.identities[0]);
        assert_eq!(hit.values[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(hit.values[1].to_bits(), 1);
        assert_eq!(cached.embed(&cx, &query).await.unwrap(), vec![99.0, 0.0]);
        assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
        assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
        assert_eq!(cached.cache_stats().entries, 2);
        assert_eq!(cached.cache_stats().hits, 2);
        assert_eq!(cached.cache_stats().misses, 2);
    });
}

#[test]
fn bound_query_keys_include_every_contract_for_same_name_and_dimension() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for component in 0..6 {
            let mut provider = BoundCacheProvider::new();
            provider.identities[1] = provider.identities[0].clone();
            let changed = &mut provider.identities[1];
            match component {
                0 => changed.space.immutable_revision.push_str("-new"),
                1 => changed.space.artifacts[0].sha256 = "a".repeat(64),
                2 => changed.space.tokenizer_fingerprint = "b".repeat(64),
                3 => changed.input.canonicalization.push_str("-new"),
                4 => changed.producer.implementation_revision.push_str("-new"),
                _ => changed.storage.format.push_str("-new"),
            }
            changed.space.input_contract_fingerprint = changed.input.fingerprint();
            changed.producer.space_fingerprint = changed.space.fingerprint();
            changed.validate().unwrap();
            assert_eq!(
                provider.identities[1].space.logical_model_id,
                provider.identities[0].space.logical_model_id
            );
            let inner = Arc::new(provider);
            let cached = CachedEmbedder::new(inner.clone(), 4);
            let original = cached.embed_bound(&cx, "query").await.unwrap();
            inner.selected.store(1, Ordering::Relaxed);
            *inner.values.lock().unwrap() = vec![0.0, 1.0];
            let replacement = cached.embed_bound(&cx, "query").await.unwrap();
            assert_eq!(replacement.values, vec![0.0, 1.0]);
            assert_eq!(replacement.identity, inner.identities[1]);
            assert_ne!(replacement.identity, original.identity);
            assert_eq!(cached.embed_bound(&cx, "query").await.unwrap(), replacement);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
            assert_eq!(cached.cache_stats().entries, 2);
        }
    });
}

#[test]
fn bound_invalid_producer_contract_never_reads_a_warm_cache_or_dispatches() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for invalid in 0..6 {
            let mut provider = BoundCacheProvider::new();
            provider.identities[1] = provider.identities[0].clone();
            match invalid {
                0 => provider.identities[1].space.dimension = 0,
                1 => {
                    "fsvi-v2".clone_into(&mut provider.identities[1].storage.format);
                    "little-endian".clone_into(&mut provider.identities[1].storage.endianness);
                }
                2 => provider.identities[1].storage.quantization = QuantizationFormat::F16,
                3 => {
                    "private-canary".clone_into(&mut provider.identities[1].storage.endianness);
                }
                _ => {}
            }
            let inner = Arc::new(provider);
            let cached = CachedEmbedder::new(inner.clone(), 4);
            let original = cached.embed_bound(&cx, "query").await.unwrap();
            let before = cached.cache_stats();
            inner.selected.store(1, Ordering::Relaxed);
            if invalid == 4 {
                inner.dimension.store(3, Ordering::Relaxed);
            } else if invalid == 5 {
                inner.unavailable.store(true, Ordering::Relaxed);
            }
            let error = cached.embed_bound(&cx, "query").await.unwrap_err();
            assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
            assert!(!error.to_string().contains("private-canary"));
            assert_eq!(cached.cache_stats(), before);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
            inner.selected.store(0, Ordering::Relaxed);
            inner.dimension.store(2, Ordering::Relaxed);
            inner.unavailable.store(false, Ordering::Relaxed);
            assert_eq!(cached.embed_bound(&cx, "query").await.unwrap(), original);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
        }
    });
}

#[test]
fn bound_foreign_and_malformed_responses_never_pollute_cache_and_retry_recovers() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for invalid in 0..5 {
            let inner = Arc::new(BoundCacheProvider::new());
            match invalid {
                0 => {
                    *inner.response_identity.lock().unwrap() = Some(inner.identities[1].clone());
                }
                1 => {
                    let mut malformed = inner.identities[0].clone();
                    "private-canary".clone_into(&mut malformed.storage.format);
                    *inner.response_identity.lock().unwrap() = Some(malformed);
                }
                2 => *inner.values.lock().unwrap() = vec![1.0],
                3 => *inner.values.lock().unwrap() = vec![f32::NAN, 0.0],
                _ => *inner.values.lock().unwrap() = vec![0.0, f32::INFINITY],
            }
            let cached = CachedEmbedder::new(inner.clone(), 4);
            cached.embed(&cx, "warm").await.unwrap();
            for _ in 0..2 {
                let error = cached.embed_bound(&cx, "warm").await.unwrap_err();
                if invalid < 2 {
                    assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                } else {
                    assert!(matches!(error, SearchError::InvalidConfig { .. }));
                }
                assert!(!error.to_string().contains("private-canary"));
                assert_eq!(cached.cache_stats().entries, 1);
            }
            *inner.response_identity.lock().unwrap() = None;
            *inner.values.lock().unwrap() = vec![0.0, 1.0];
            let response = cached.embed_bound(&cx, "warm").await.unwrap();
            assert_eq!(response.values, vec![0.0, 1.0]);
            assert_eq!(cached.embed_bound(&cx, "warm").await.unwrap(), response);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 3);
            assert_eq!(cached.cache_stats().entries, 2);
        }
    });
}

#[test]
fn bound_inflight_producer_drift_precedes_late_success_and_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for action in [0, 1] {
            let inner = Arc::new(BoundCacheProvider::new());
            inner.action.store(action, Ordering::Relaxed);
            let cached = CachedEmbedder::new(inner.clone(), 4);
            let mut flight = cached.embed_bound(&cx, "query");
            {
                let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
            }
            inner.selected.store(1, Ordering::Relaxed);
            assert!(matches!(
                flight.await,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
            assert_eq!(cached.cache_stats().entries, 0);
            inner.action.store(0, Ordering::Relaxed);
            let response = cached.embed_bound(&cx, "query").await.unwrap();
            assert_eq!(response.identity, inner.identities[1]);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
        }
    });
}

#[test]
fn bound_inflight_cancellation_precedes_success_and_provider_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for action in [2, 3] {
            let inner = Arc::new(BoundCacheProvider::new());
            inner.action.store(action, Ordering::Relaxed);
            let cached = CachedEmbedder::new(inner.clone(), 4);
            cached.embed(&cx, "warm").await.unwrap();
            assert!(matches!(
                cached.embed_bound(&cx, "warm").await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(cached.cache_stats().entries, 1);
            assert!(cached.state_lock().bound_entries.is_empty());
            cx.set_cancel_requested(false);
            inner.action.store(0, Ordering::Relaxed);
            let response = cached.embed_bound(&cx, "warm").await.unwrap();
            assert_eq!(cached.embed_bound(&cx, "warm").await.unwrap(), response);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
        }
    });
}

#[test]
fn bound_clear_fences_suspended_fill_and_preserves_new_raw_or_bound_entry() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for bound_refill in [false, true] {
            let inner = Arc::new(BoundCacheProvider::new());
            let cached = CachedEmbedder::new(inner.clone(), 1);
            let mut flight = cached.embed_bound(&cx, "old");
            {
                let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
            }
            cached.clear_cache();
            cached.clear_cache();
            *inner.values.lock().unwrap() = vec![0.0, 1.0];
            if bound_refill {
                cached.embed_bound(&cx, "new").await.unwrap();
            } else {
                cached.embed(&cx, "new").await.unwrap();
            }
            let after_clear = cached.cache_stats();
            let old = flight.await.unwrap();
            assert_eq!(old.values[0].to_bits(), (-0.0_f32).to_bits());
            assert_eq!(old.values[1].to_bits(), 1);
            assert_eq!(cached.cache_stats(), after_clear);
            if bound_refill {
                assert_eq!(
                    cached.embed_bound(&cx, "new").await.unwrap().values,
                    vec![0.0, 1.0]
                );
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
            } else {
                assert_eq!(cached.embed(&cx, "new").await.unwrap(), vec![99.0, 0.0]);
                assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
            }
            cached.embed_bound(&cx, "old").await.unwrap();
            assert_eq!(
                inner.bound_calls.load(Ordering::Relaxed),
                if bound_refill { 3 } else { 2 }
            );
            assert_eq!(cached.cache_stats().entries, 1);
        }
    });
}

#[test]
fn bound_hit_rechecks_producer_before_serving_and_counts_only_valid_hits() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(BoundCacheProvider::new());
        let cached = CachedEmbedder::new(inner.clone(), 4);
        cached.embed_bound(&cx, "warm").await.unwrap();
        let before = cached.cache_stats();
        let provider = Arc::downgrade(&inner);
        inner.at_identity_read(4, move || {
            provider
                .upgrade()
                .unwrap()
                .selected
                .store(1, Ordering::Relaxed);
        });
        assert!(matches!(
            cached.embed_bound(&cx, "warm").await,
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
        assert_eq!(cached.cache_stats(), before);
        assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            cached.embed_bound(&cx, "warm").await.unwrap().identity,
            inner.identities[1]
        );
        assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
    });
}

#[test]
fn bound_identity_callback_can_clear_cache_without_deadlock_or_reset_stat_mutation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for during_hit in [false, true] {
            let inner = Arc::new(BoundCacheProvider::new());
            let cached = Arc::new(CachedEmbedder::new(inner.clone(), 4));
            if during_hit {
                cached.embed_bound(&cx, "warm").await.unwrap();
            }
            let cache = Arc::downgrade(&cached);
            inner.at_identity_read(if during_hit { 4 } else { 2 }, move || {
                cache.upgrade().unwrap().clear_cache();
            });
            cached.embed_bound(&cx, "warm").await.unwrap();
            assert_eq!(
                cached.cache_stats(),
                CacheStats {
                    hits: 0,
                    misses: 0,
                    entries: 0,
                    capacity: 4
                }
            );
            cached.embed_bound(&cx, "warm").await.unwrap();
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
            assert_eq!(cached.cache_stats().entries, 1);
        }
    });
}

#[test]
fn bound_cancellation_in_identity_callback_dominates_drift_or_identity_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for boundary in [1, 2, 4] {
            for unavailable in [false, true] {
                let inner = Arc::new(BoundCacheProvider::new());
                let cached = CachedEmbedder::new(inner.clone(), 4);
                if boundary == 4 {
                    cached.embed_bound(&cx, "warm").await.unwrap();
                }
                let before = cached.cache_stats();
                let provider = Arc::downgrade(&inner);
                let cancel_cx = cx.clone();
                inner.at_identity_read(boundary, move || {
                    let provider = provider.upgrade().unwrap();
                    provider.selected.store(1, Ordering::Relaxed);
                    provider.unavailable.store(unavailable, Ordering::Relaxed);
                    cancel_cx.set_cancel_requested(true);
                });
                assert!(matches!(
                    cached.embed_bound(&cx, "warm").await,
                    Err(SearchError::Cancelled { .. })
                ));
                assert_eq!(cached.cache_stats().entries, before.entries);
                assert_eq!(cached.cache_stats().hits, before.hits);
                assert_eq!(
                    inner.bound_calls.load(Ordering::Relaxed),
                    usize::from(boundary != 1)
                );
                cx.set_cancel_requested(false);
            }
        }
    });
}

#[test]
fn bound_unpolled_future_is_lazy_and_resolves_again_after_clear() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(BoundCacheProvider::new());
        let cached = CachedEmbedder::new(inner.clone(), 4);
        cached.embed_bound(&cx, "warm").await.unwrap();
        let before = cached.cache_stats();
        let reads = inner.identity_calls.load(Ordering::Relaxed);
        let future = cached.embed_bound(&cx, "warm");
        assert_eq!(cached.cache_stats(), before);
        assert_eq!(inner.identity_calls.load(Ordering::Relaxed), reads);
        assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
        cached.clear_cache();
        *inner.values.lock().unwrap() = vec![1.0, 0.0];
        assert_eq!(future.await.unwrap().values, vec![1.0, 0.0]);
        assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
        assert_eq!(cached.cache_stats().hits, 0);
        assert_eq!(cached.cache_stats().misses, 1);
    });
}

#[test]
fn bound_and_raw_entries_share_capacity_and_zero_capacity_still_validates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for capacity in [0, 1] {
            let inner = Arc::new(BoundCacheProvider::new());
            let cached = CachedEmbedder::new(inner.clone(), capacity);
            for index in 0..3 {
                let query = format!("query-{index}");
                cached.embed(&cx, &query).await.unwrap();
                assert_eq!(cached.cache_stats().entries, capacity);
                cached.embed_bound(&cx, &query).await.unwrap();
                assert_eq!(cached.cache_stats().entries, capacity);
            }
            assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 3);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 3);
            let (ghosts, ghost_set) = {
                let state = cached.state_lock();
                (state.ghost.len(), state.ghost_set.len())
            };
            assert!(ghosts <= capacity);
            assert!(ghost_set <= capacity);
            cached.clear_cache();
            *inner.values.lock().unwrap() = vec![f32::NAN, 0.0];
            assert!(matches!(
                cached.embed_bound(&cx, "query").await,
                Err(SearchError::InvalidConfig { .. })
            ));
            assert_eq!(cached.cache_stats().entries, 0);
        }
    });
}

#[test]
fn bound_hot_entries_survive_cold_raw_scans_under_shared_s3_fifo() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(BoundCacheProvider::new());
        let cached = CachedEmbedder::new(inner.clone(), 4);
        let hot = cached.embed_bound(&cx, "hot").await.unwrap();
        cached.embed_bound(&cx, "hot").await.unwrap();
        for index in 0..12 {
            cached.embed(&cx, &format!("cold-{index}")).await.unwrap();
            assert!(cached.cache_stats().entries <= 4);
        }
        assert_eq!(cached.embed_bound(&cx, "hot").await.unwrap(), hot);
        assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
        assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 12);
    });
}
