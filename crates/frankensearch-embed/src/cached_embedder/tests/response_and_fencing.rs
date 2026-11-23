use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use asupersync::Cx;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};
use frankensearch_core::{SearchError, SearchResult};

use super::super::CachedEmbedder;
use super::make_cached;

struct ResponseEmbedder {
    response: Mutex<Vec<Vec<f32>>>,
    calls: AtomicUsize,
    identity: EmbeddingIdentityBundleV1,
    yield_response: bool,
    cancel_response: AtomicBool,
}

impl ResponseEmbedder {
    fn new() -> Self {
        Self {
            response: Mutex::new(vec![vec![1.0, 0.0]]),
            calls: AtomicUsize::new(0),
            identity: EmbeddingIdentityBundleV1::explicit_test_model("cache-response-test", 2),
            yield_response: false,
            cancel_response: AtomicBool::new(false),
        }
    }

    async fn before_response(&self, cx: &Cx) {
        if self.yield_response {
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
        }
        if self.cancel_response.load(Ordering::Relaxed) {
            cx.set_cancel_requested(true);
        }
    }
}

impl Embedder for ResponseEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let response = self.response.lock().unwrap()[0].clone();
        Box::pin(async move {
            self.before_response(cx).await;
            Ok(response)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        _texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let response = self.response.lock().unwrap().clone();
        Box::pin(async move {
            self.before_response(cx).await;
            Ok(response)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        2
    }

    fn id(&self) -> &'static str {
        "cache-response-test"
    }

    fn model_name(&self) -> &'static str {
        "Cache Response Test Embedder"
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

#[test]
fn malformed_batch_cardinality_is_an_error_without_partial_cache_admission() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for returned in [0, 1, 3] {
            let inner = Arc::new(ResponseEmbedder::new());
            let cached = CachedEmbedder::new(inner.clone(), 2);
            let warm = cached.embed(&cx, "warm").await.unwrap();
            *inner.response.lock().unwrap() = vec![vec![0.0, 1.0]; returned];
            let error = cached
                .embed_batch(&cx, &["warm", "first", "second", "first"])
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
            assert_eq!(cached.cache_stats().entries, 1);
            assert!(!cached.state_lock().entries.contains_key("first"));
            assert_eq!(cached.embed(&cx, "warm").await.unwrap(), warm);
            assert_eq!(inner.calls.load(Ordering::Relaxed), 2);
        }
    });
}

#[test]
fn invalid_batch_vector_does_not_cache_the_valid_prefix() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for invalid in [
            vec![],
            vec![1.0],
            vec![1.0, 0.0, 0.0],
            vec![f32::NAN, 0.0],
            vec![0.0, f32::INFINITY],
            vec![f32::NEG_INFINITY, 0.0],
        ] {
            let inner = Arc::new(ResponseEmbedder::new());
            let cached = CachedEmbedder::new(inner.clone(), 2);
            let warm = cached.embed(&cx, "warm").await.unwrap();
            *inner.response.lock().unwrap() = vec![vec![0.0, 1.0], invalid];
            assert!(matches!(
                cached.embed_batch(&cx, &["first", "second"]).await,
                Err(SearchError::DimensionMismatch { .. } | SearchError::EmbeddingFailed { .. })
            ));
            assert_eq!(cached.cache_stats().entries, 1);
            assert!(!cached.state_lock().entries.contains_key("first"));
            assert_eq!(cached.embed(&cx, "warm").await.unwrap(), warm);
        }
    });
}

#[test]
fn invalid_single_vectors_are_not_returned_or_cached_and_retry_can_recover() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for invalid in [vec![1.0], vec![f32::NAN, 0.0], vec![0.0, f32::INFINITY]] {
            let inner = Arc::new(ResponseEmbedder::new());
            let cached = CachedEmbedder::new(inner.clone(), 4);
            *inner.response.lock().unwrap() = vec![invalid];
            for _ in 0..2 {
                assert!(cached.embed(&cx, "query").await.is_err());
                assert_eq!(cached.cache_stats().entries, 0);
            }
            *inner.response.lock().unwrap() = vec![vec![1.0, 0.0]];
            assert_eq!(cached.embed(&cx, "query").await.unwrap(), vec![1.0, 0.0]);
            assert_eq!(cached.embed(&cx, "query").await.unwrap(), vec![1.0, 0.0]);
            assert_eq!(inner.calls.load(Ordering::Relaxed), 3);
        }
    });
}

#[test]
fn disabling_storage_does_not_disable_provider_response_validation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(ResponseEmbedder::new());
        let cached = CachedEmbedder::new(inner.clone(), 0);
        assert!(matches!(
            cached.embed_batch(&cx, &["first", "second"]).await,
            Err(SearchError::EmbeddingFailed { .. })
        ));
        *inner.response.lock().unwrap() = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        assert_eq!(
            cached.embed_batch(&cx, &["first", "second"]).await.unwrap(),
            vec![vec![0.0, 0.0], vec![1.0, 0.0]]
        );
        assert_eq!(cached.cache_stats().entries, 0);
    });
}

#[test]
fn unpolled_futures_do_not_touch_cache_or_capture_pre_clear_hits() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (cached, inner) = make_cached(4);
        cached.embed(&cx, "warm").await.unwrap();
        let before = cached.cache_stats();
        let hit = cached.embed(&cx, "warm");
        let miss = cached.embed(&cx, "new");
        assert_eq!(cached.cache_stats(), before);
        assert_eq!(inner.call_count(), 1);
        drop(miss);
        cached.clear_cache();
        hit.await.unwrap();
        assert_eq!(inner.call_count(), 2);
        assert_eq!(cached.cache_stats().misses, 1);
        assert_eq!(cached.cache_stats().hits, 0);
    });
}

#[test]
fn clear_fences_in_flight_single_response_and_preserves_new_entries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for refill in [false, true] {
            let mut provider = ResponseEmbedder::new();
            provider.yield_response = true;
            let inner = Arc::new(provider);
            let cached = CachedEmbedder::new(inner.clone(), 1);
            let mut flight = cached.embed(&cx, "old");
            {
                let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
            }
            assert_eq!(inner.calls.load(Ordering::Relaxed), 1);
            cached.clear_cache();
            cached.clear_cache();
            if refill {
                *inner.response.lock().unwrap() = vec![vec![0.0, 1.0]];
                cached.embed(&cx, "new").await.unwrap();
            }
            let after_clear = cached.cache_stats();
            assert_eq!(flight.await.unwrap(), vec![1.0, 0.0]);
            assert_eq!(cached.cache_stats(), after_clear);
            assert!(!cached.state_lock().entries.contains_key("old"));
            if refill {
                assert_eq!(cached.embed(&cx, "new").await.unwrap(), vec![0.0, 1.0]);
                assert_eq!(inner.calls.load(Ordering::Relaxed), 2);
            }
        }
    });
}

#[test]
fn clear_fences_in_flight_batch_without_losing_order_or_duplicates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut provider = ResponseEmbedder::new();
        provider.yield_response = true;
        let inner = Arc::new(provider);
        *inner.response.lock().unwrap() = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cached = CachedEmbedder::new(inner.clone(), 1);
        let texts = ["first", "second", "first"];
        let mut flight = cached.embed_batch(&cx, &texts);
        {
            let mut task = std::task::Context::from_waker(std::task::Waker::noop());
            assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
        }
        cached.clear_cache();
        *inner.response.lock().unwrap() = vec![vec![0.0, 0.0]];
        cached.embed(&cx, "new").await.unwrap();
        let after_clear = cached.cache_stats();
        assert_eq!(
            flight.await.unwrap(),
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]]
        );
        assert_eq!(cached.cache_stats(), after_clear);
        assert!(cached.state_lock().entries.contains_key("new"));
        assert_eq!(cached.embed(&cx, "new").await.unwrap(), vec![0.0, 0.0]);
        assert_eq!(inner.calls.load(Ordering::Relaxed), 2);
    });
}

#[test]
fn cancellation_precedes_hits_misses_empty_batches_and_bound_calls() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (cached, inner) = make_cached(4);
        cached.embed(&cx, "warm").await.unwrap();
        let before = cached.cache_stats();
        let constructed_hit = cached.embed(&cx, "warm");
        cx.set_cancel_requested(true);
        assert!(matches!(
            constructed_hit.await,
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            cached.embed(&cx, "new").await,
            Err(SearchError::Cancelled { .. })
        ));
        for texts in [vec![], vec!["warm"], vec!["warm", "new"]] {
            assert!(matches!(
                cached.embed_batch(&cx, &texts).await,
                Err(SearchError::Cancelled { .. })
            ));
        }
        assert!(matches!(
            cached.embed_bound(&cx, "warm").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(cached.cache_stats(), before);
        assert_eq!(inner.call_count(), 1);
        assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn cancellation_after_provider_success_does_not_admit_single_or_batch_misses() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(ResponseEmbedder::new());
        let cached = CachedEmbedder::new(inner.clone(), 4);
        cached.embed(&cx, "warm").await.unwrap();
        inner.cancel_response.store(true, Ordering::Relaxed);
        assert!(matches!(
            cached.embed(&cx, "new").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(cached.cache_stats().entries, 1);
        cx.set_cancel_requested(false);
        assert!(matches!(
            cached.embed_batch(&cx, &["warm", "new"]).await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(cached.cache_stats().entries, 1);
        assert!(!cached.state_lock().entries.contains_key("new"));
        cx.set_cancel_requested(false);
        inner.cancel_response.store(false, Ordering::Relaxed);
        assert_eq!(cached.embed(&cx, "new").await.unwrap(), vec![1.0, 0.0]);
        assert_eq!(cached.cache_stats().entries, 2);
    });
}
