use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use asupersync::Cx;

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};
#[cfg(any(feature = "native", feature = "rerank"))]
use frankensearch_embed::DetectOptions;
use frankensearch_embed::auto_detect::{EmbedderStack, TwoTierAvailability};
use frankensearch_index::{
    TwoTierIndex, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
};
#[cfg(feature = "quill")]
use frankensearch_quill::{QuillConfig, QuillIndex};

use super::super::IndexBuilder;
#[cfg(any(feature = "native", feature = "rerank"))]
use super::super::detect_embedder_stack_with_pool;
use super::{
    BoundBuildFault, IdentityStubEmbedder, RecordingExporter, bound_build_stack, fast_only_stack,
    stub_stack,
};

#[cfg(any(feature = "native", feature = "rerank"))]
#[test]
fn native_detection_requires_a_live_worker_and_preserves_cancellation() {
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(0, 1)
        .build()
        .unwrap();
    let pool = runtime.blocking_handle().unwrap();
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    cx.cancel_fast(asupersync::CancelKind::User);
    let error = runtime
        .block_on(detect_embedder_stack_with_pool(
            &cx,
            None,
            &DetectOptions {
                offline: Some(true),
            },
            pool.clone(),
        ))
        .expect_err("pre-cancelled detection must not load a model");
    assert!(matches!(error, SearchError::Cancelled { .. }));
    let uncancelled = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    assert!(runtime.shutdown_timeout(std::time::Duration::from_secs(5)));
    let driver = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .unwrap();
    let error = driver
        .block_on(detect_embedder_stack_with_pool(
            &uncancelled,
            None,
            &DetectOptions {
                offline: Some(true),
            },
            pool,
        ))
        .expect_err("a closed pool must not run detection inline");
    assert!(
        matches!(error, SearchError::EmbeddingFailed { .. }),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("cannot admit model loading worker")
    );
}

#[cfg(any(feature = "native", feature = "rerank"))]
#[test]
fn native_detection_stopped_pool_on_live_runtime_never_loads_inline() {
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .unwrap();
    let pool = asupersync::runtime::blocking_pool::BlockingPool::new(0, 1);
    let handle = pool.handle();
    assert!(pool.shutdown_and_wait(std::time::Duration::from_secs(2)));
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let model_root = tempfile::tempdir().unwrap();
    let result = runtime.block_on(detect_embedder_stack_with_pool(
        &cx,
        Some(model_root.path()),
        &DetectOptions {
            offline: Some(true),
        },
        handle,
    ));
    eprintln!("stopped pool detection: {result:?}");
    let error = result.expect_err("stopped pool must reject model loading");
    assert!(
        matches!(error, SearchError::EmbeddingFailed { .. }),
        "{error}"
    );
    assert!(
        error.to_string().contains("blocking pool is shut down"),
        "{error}"
    );
    assert!(runtime.shutdown_timeout(std::time::Duration::from_secs(2)));
}

#[cfg(any(feature = "native", feature = "rerank"))]
#[test]
fn native_detection_shutdown_after_spawn_never_loads_inline() {
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .unwrap();
    let pool = asupersync::runtime::blocking_pool::BlockingPool::new(0, 1);
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let model_root = tempfile::tempdir().unwrap();
    runtime.block_on(async {
        let options = DetectOptions {
            offline: Some(true),
        };
        let mut detection = Box::pin(detect_embedder_stack_with_pool(
            &cx,
            Some(model_root.path()),
            &options,
            pool.handle(),
        ));
        std::future::poll_fn(|task_cx| {
            assert!(detection.as_mut().poll(task_cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        // The first poll admitted the wrapper; the current-thread runtime
        // has not polled it yet, so shutdown races the actual pool dispatch.
        assert!(pool.shutdown_and_wait(std::time::Duration::from_secs(2)));
        let result = detection.await;
        eprintln!("pool shutdown after spawn: {result:?}");
        let error = result.expect_err("rejected dispatch must not start a model loader");
        assert!(
            matches!(error, SearchError::EmbeddingFailed { .. }),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("dispatch fell back to an async executor"),
            "{error}"
        );
    });
    assert!(runtime.shutdown_timeout(std::time::Duration::from_secs(2)));
}

#[cfg(any(feature = "native", feature = "rerank"))]
#[test]
fn native_detection_request_cancellation_wins_over_model_error() {
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(0, 1)
        .build()
        .unwrap();
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let model_root = tempfile::tempdir().unwrap();
    runtime.block_on(async {
        let options = DetectOptions {
            offline: Some(true),
        };
        let mut detection = Box::pin(detect_embedder_stack_with_pool(
            &cx,
            Some(model_root.path()),
            &options,
            runtime.blocking_handle().unwrap(),
        ));
        std::future::poll_fn(|task_cx| {
            assert!(detection.as_mut().poll(task_cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        cx.cancel_fast(asupersync::CancelKind::User);
        let result = detection.await;
        eprintln!("cancelled request detection: {result:?}");
        assert!(
            matches!(result, Err(SearchError::Cancelled { .. })),
            "request cancellation after spawn must take precedence over model failure"
        );
    });
    assert!(runtime.shutdown_timeout(std::time::Duration::from_secs(2)));
}

#[cfg(any(feature = "native", feature = "rerank"))]
#[test]
fn native_detection_cancellation_wins_over_completed_model_error() {
    use std::time::{Duration, Instant};

    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .unwrap();
    let external_pool = asupersync::runtime::blocking_pool::BlockingPool::new(0, 1);
    let pool = external_pool.handle();
    let model_root = tempfile::tempdir().unwrap();
    let options = DetectOptions {
        offline: Some(true),
    };
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    // The uncancelled control proves that this root produces a real loader
    // error. No fake embedder or substituted worker result is involved.
    let control = runtime.block_on(detect_embedder_stack_with_pool(
        &cx,
        Some(model_root.path()),
        &options,
        pool.clone(),
    ));
    assert!(
        matches!(control, Err(SearchError::EmbedderUnavailable { .. })),
        "{control:?}"
    );

    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let _blocker = pool.spawn(move || {
        started_tx.send(()).unwrap();
        // Disconnect or the fuse releases the worker even if an assertion
        // unwinds before the explicit release below.
        let _ = release_rx.recv_timeout(Duration::from_secs(5));
    });
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    runtime.block_on(async {
        let mut detection = Box::pin(detect_embedder_stack_with_pool(
            &cx,
            Some(model_root.path()),
            &options,
            pool.clone(),
        ));
        std::future::poll_fn(|task_cx| {
            assert!(detection.as_mut().poll(task_cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        let started = Instant::now();
        while pool.pending_count() != 1 {
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "loader not queued"
            );
            asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
        }
        release_tx.send(()).unwrap();
        // Do not poll detection: let its actual worker finish while the
        // original request is still live, then cancel before consuming it.
        // Joining the external pool proves completion; separate relaxed
        // pending/busy counters can transiently both read zero at dequeue.
        assert!(external_pool.shutdown_and_wait(Duration::from_secs(2)));
        cx.cancel_fast(asupersync::CancelKind::User);
        let result = detection.await;
        eprintln!("cancelled request with completed loader error: {result:?}");
        assert!(
            matches!(result, Err(SearchError::Cancelled { .. })),
            "caller cancellation must precede unwrapping a completed model error"
        );
    });
    assert!(runtime.shutdown_timeout(Duration::from_secs(2)));
}

/// A hash-only build must report itself as a DEGRADED generation
/// (`bd-a6zt` Cause A). The vectors written carry the hash embedder's
/// identity, so this index can never answer semantically and installing a
/// model later cannot repair it — yet the build succeeds and returns
/// plausible results at query time. Without a machine-readable signal on
/// the build result there is nothing for a caller to check, which is
/// exactly how this shipped unnoticed.
#[cfg(feature = "hash")]
#[test]
fn build_reports_hash_only_generation_as_degraded() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let hash_stack = EmbedderStack::from_parts(
            Arc::new(frankensearch_embed::HashEmbedder::default_256()) as Arc<dyn Embedder>,
            None,
        );
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(hash_stack)
            .add_document("doc-1", "Hello world")
            .add_document("doc-2", "Distributed consensus")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 2, "the degraded build still succeeds");
        assert_eq!(
            stats.embedder_availability,
            TwoTierAvailability::HashOnly,
            "a hash-only stack must be reported, not inferred",
        );
        assert!(
            stats.is_degraded_generation(),
            "hash-only generations are permanently non-semantic and must say so",
        );
    });
}

#[cfg(feature = "hash")]
#[test]
fn build_does_not_write_a_hash_quality_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let hash_stack = EmbedderStack::from_parts(
            Arc::new(frankensearch_embed::HashEmbedder::default_256()) as Arc<dyn Embedder>,
            Some(Arc::new(frankensearch_embed::HashEmbedder::default_384()) as Arc<dyn Embedder>),
        );
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(hash_stack)
            .add_document("doc-1", "Hello world")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.embedder_availability, TwoTierAvailability::HashOnly);
        assert!(
            !stats.has_quality_index,
            "hash control must not mint a quality generation: quality_indexed={}",
            stats.quality_indexed
        );
        let index = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
        assert!(!index.has_quality_index());
    });
}

#[test]
fn build_refuses_auto_detected_hash_only_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let error = IndexBuilder::new(dir.path())
            .add_document("doc-1", "Hello world")
            .build(&cx)
            .await
            .expect_err("auto-detect hash must not write a generation");
        assert!(
            matches!(error, SearchError::EmbedderUnavailable { .. }),
            "expected EmbedderUnavailable, got: {error:?}"
        );
    });
}

/// The converse, so the signal cannot be a constant `true`: a healthy
/// two-tier build must report `Full` and must NOT be flagged degraded.
#[test]
fn build_reports_healthy_generation_as_not_degraded() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "Hello world")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.embedder_availability, TwoTierAvailability::Full);
        assert!(!stats.is_degraded_generation());
    });
}

/// A semantic fast tier with no quality tier is degraded too, but it is a
/// *different* degradation: the generation is still semantic, so it is
/// repairable by adding a quality model without a rebuild. Pinning both
/// arms keeps the two cases from collapsing into one boolean.
#[test]
fn build_reports_fast_only_generation_as_degraded_but_semantic() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(fast_only_stack())
            .add_document("doc-1", "Hello world")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.embedder_availability, TwoTierAvailability::FastOnly);
        assert!(stats.is_degraded_generation());
        assert_ne!(
            stats.embedder_availability,
            TwoTierAvailability::HashOnly,
            "fast-only must not be conflated with the non-semantic hash case",
        );
    });
}

#[test]
fn build_happy_path() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "Hello world")
            .add_document("doc-2", "Distributed consensus")
            .add_document("doc-3", "Vector search algorithms")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 3);
        assert_eq!(stats.error_count, 0);
        assert!(stats.has_quality_index);
        assert!(stats.total_ms > 0.0);
        assert!(stats.embed_ms > 0.0);

        // Verify the index can be opened.
        let index = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
        assert_eq!(index.doc_count(), 3);
        assert!(index.has_quality_index());
    });
}

/// bd-9xuj T2-C2: `build` must thread the embedders' REAL identities into
/// the built index, not just their id strings. Observable through the
/// persisted v1 headers: the id string AND the whole producer fingerprint
/// survive a reopen. The typed bundle and admitted v2 owner do not, and
/// their absence must never be filled in from the surviving strings.
#[test]
fn build_threads_typed_identity_into_persisted_headers() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = IdentityStubEmbedder::new("identity-fast", 4);
        let quality = IdentityStubEmbedder::new("identity-quality", 4);
        let fast_revision = fast.identity.fingerprint();
        let quality_revision = quality.identity.fingerprint();
        let stack = EmbedderStack::from_parts(Arc::new(fast), Some(Arc::new(quality)));

        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stack)
            .add_document("doc-1", "Hello world")
            .add_document("doc-2", "Distributed consensus")
            .build(&cx)
            .await
            .unwrap();
        assert_eq!(stats.doc_count, 2);

        let reopened = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
        assert_eq!(reopened.fast_embedder_id(), "identity-fast");
        assert_eq!(
            reopened.fast_embedder_revision(),
            fast_revision.as_str(),
            "the fast header must bind the complete producing identity, \
             including execution precision and conformance certificate"
        );
        assert_eq!(reopened.quality_embedder_id(), Some("identity-quality"));
        assert_eq!(
            reopened.quality_embedder_revision(),
            Some(quality_revision.as_str())
        );
        assert_eq!(
            reopened.fast_space_fingerprint_hex(),
            None,
            "v1 artifacts persist no space identity; reopen is typed \
             legacy-unidentified, never fabricated from header strings"
        );
        assert_eq!(reopened.quality_space_fingerprint_hex(), None);
    });
}

/// The legacy arm stays legacy: a stack of identity-less embedders keeps
/// building exactly as before — empty header revisions, no identity —
/// so the typed threading cannot have made identity a requirement.
#[test]
fn build_without_identity_bundles_stays_legacy_unidentified() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "Hello world")
            .build(&cx)
            .await
            .unwrap();
        assert_eq!(stats.doc_count, 1);

        let reopened = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
        assert_eq!(reopened.fast_embedder_id(), "stub-fast");
        assert_eq!(reopened.fast_embedder_revision(), "");
        assert_eq!(reopened.quality_embedder_revision(), Some(""));
        assert_eq!(reopened.fast_space_fingerprint_hex(), None);
        assert_eq!(reopened.quality_space_fingerprint_hex(), None);
    });
}

/// bd-9xuj T2-C2 follow-up (review #8151): threading identities through
/// `build` introduced a NEW production failure mode — a build that
/// previously succeeded now fails when an embedder's declared identity
/// does not describe the vectors it actually emits. Pin the failure:
/// declared space dimension 8, emitted vectors dimension 4 →
/// Producer admission rejects with typed `InvalidConfig` at
/// `fast_identity.space.dimension` — never a panic, and never an index
/// carrying an identity that lies about its vectors.
#[test]
fn build_fails_typed_when_declared_identity_dimension_mismatches_vectors() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        // Emits 4-dim vectors but declares an 8-dim space: exactly the
        // self-inconsistent embedder rejected before inference or writing.
        let lying = IdentityStubEmbedder {
            id: "identity-lying-fast",
            dim: 4,
            identity:
                frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                    "identity-lying-fast",
                    8,
                ),
        };
        let stack = EmbedderStack::from_parts(Arc::new(lying), None);
        let error = IndexBuilder::new(dir.path())
            .with_embedder_stack(stack)
            .add_document("doc-1", "Hello world")
            .build(&cx)
            .await
            .expect_err(
                "a declared identity that does not describe the emitted vectors \
                 must fail the build with a typed error",
            );
        assert!(
            matches!(
                &error,
                SearchError::InvalidConfig {
                    field,
                    value,
                    reason,
                } if field == "fast_identity.space.dimension"
                    && value == "8"
                    && reason.contains("does not describe")
            ),
            "expected typed InvalidConfig at fast_identity.space.dimension \
             (value 8, refusing an identity that does not describe the \
             written vectors), got {error:?}"
        );
    });
}

#[test]
fn build_bound_response_identity_and_cancellation_failures_preserve_persisted_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in ["fast", "quality"] {
            for fault in [
                BoundBuildFault::ForeignIdentity,
                BoundBuildFault::MalformedForeignIdentity,
                BoundBuildFault::ChangedProducer,
                BoundBuildFault::CancelWithError,
            ] {
                let dir = tempfile::tempdir().unwrap();
                let (seed, _, _) = bound_build_stack("", BoundBuildFault::None);
                IndexBuilder::new(dir.path())
                    .with_embedder_stack(seed)
                    .add_document("retained", "previous complete document")
                    .build(&cx)
                    .await
                    .unwrap();
                let before = [VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME]
                    .map(|name| (name, std::fs::read(dir.path().join(name)).unwrap()));
                let (stack, fast, quality) = bound_build_stack(tier, fault);
                let error = IndexBuilder::new(dir.path())
                    .with_embedder_stack(stack)
                    .add_document("accepted", "valid first document")
                    .add_document("rejected", "fault-trigger")
                    .build(&cx)
                    .await
                    .unwrap_err();
                cx.set_cancel_requested(false);
                if fault == BoundBuildFault::CancelWithError {
                    assert!(matches!(error, SearchError::Cancelled { .. }));
                } else {
                    assert!(
                        matches!(error, SearchError::UnverifiableRemoteSpace { producer, .. }
                        if producer == format!("index_builder.{tier}"))
                    );
                }
                assert_eq!(fast.raw_calls.load(Ordering::SeqCst), 0);
                assert_eq!(quality.raw_calls.load(Ordering::SeqCst), 0);
                let faulty = if tier == "fast" { &fast } else { &quality };
                assert_eq!(faulty.bound_calls.load(Ordering::SeqCst), 2);
                for (name, bytes) in &before {
                    assert_eq!(std::fs::read(dir.path().join(name)).unwrap(), *bytes);
                }
                let reopened = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
                assert_eq!(reopened.doc_count(), 1);
            }
        }
    });
}

#[test]
fn build_bound_response_validation_keeps_only_usable_vectors_and_truthful_coverage() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in ["fast", "quality"] {
            for fault in [BoundBuildFault::NonFinite, BoundBuildFault::WrongDimension] {
                let dir = tempfile::tempdir().unwrap();
                let (stack, fast, quality) = bound_build_stack(tier, fault);
                let stats = IndexBuilder::new(dir.path())
                    .with_embedder_stack(stack)
                    .add_document("accepted", "valid first document")
                    .add_document("rejected", "fault-trigger")
                    .build(&cx)
                    .await
                    .unwrap();
                assert_eq!(stats.quality_indexed, 1);
                assert_eq!(stats.error_count, usize::from(tier == "fast"));
                assert_eq!(stats.quality_errors.len(), usize::from(tier == "quality"));
                assert_eq!(stats.doc_count, if tier == "fast" { 1 } else { 2 });
                assert_eq!(fast.raw_calls.load(Ordering::SeqCst), 0);
                assert_eq!(quality.raw_calls.load(Ordering::SeqCst), 0);
                let path = dir.path().join(if tier == "fast" {
                    VECTOR_INDEX_FAST_FILENAME
                } else {
                    VECTOR_INDEX_QUALITY_FILENAME
                });
                let index = frankensearch_index::VectorIndex::open_read_only(&path).unwrap();
                assert_eq!(index.record_count(), 1);
                assert_eq!(index.doc_id_at(0).unwrap(), "accepted");
                assert!(
                    index
                        .vector_at_f32(0)
                        .unwrap()
                        .iter()
                        .all(|value| value.is_finite())
                );
            }
        }
    });
}

#[test]
fn build_rechecks_both_producers_after_final_progress_callback() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in ["fast", "quality"] {
            let dir = tempfile::tempdir().unwrap();
            let (stack, fast, quality) = bound_build_stack("", BoundBuildFault::None);
            let changed = if tier == "fast" {
                Arc::clone(&fast)
            } else {
                Arc::clone(&quality)
            };
            let error = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .with_progress(move |progress| {
                    if progress.completed == progress.total {
                        changed.changed.store(true, Ordering::SeqCst);
                    }
                })
                .add_document("document", "all inference completed successfully")
                .build(&cx)
                .await
                .unwrap_err();
            assert!(
                matches!(error, SearchError::UnverifiableRemoteSpace { producer, .. }
                if producer == format!("index_builder.{tier}"))
            );
            assert_eq!(fast.bound_calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.bound_calls.load(Ordering::SeqCst), 1);
            assert!(!dir.path().join(VECTOR_INDEX_FAST_FILENAME).exists());
            assert!(!dir.path().join(VECTOR_INDEX_QUALITY_FILENAME).exists());
        }
    });
}

#[test]
fn build_identity_refusal_cannot_fall_back_to_raw_inference() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in ["fast", "quality"] {
            let dir = tempfile::tempdir().unwrap();
            let (stack, fast, quality) = bound_build_stack(tier, BoundBuildFault::RejectIdentity);
            let error = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .add_document("document", "must not be embedded")
                .build(&cx)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
            for embedder in [&fast, &quality] {
                assert_eq!(embedder.raw_calls.load(Ordering::SeqCst), 0);
                assert_eq!(embedder.bound_calls.load(Ordering::SeqCst), 0);
            }
            assert!(!dir.path().join(VECTOR_INDEX_FAST_FILENAME).exists());
            assert!(!dir.path().join(VECTOR_INDEX_QUALITY_FILENAME).exists());
        }
    });
}

#[test]
fn build_rejects_persisted_output_contract_before_inference_or_generation_replacement() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in ["fast", "quality"] {
            let dir = tempfile::tempdir().unwrap();
            let (seed, _, _) = bound_build_stack("", BoundBuildFault::None);
            IndexBuilder::new(dir.path())
                .with_embedder_stack(seed)
                .add_document("retained", "previous complete document")
                .build(&cx)
                .await
                .unwrap();
            let before = [VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME]
                .map(|name| (name, std::fs::read(dir.path().join(name)).unwrap()));
            let (stack, fast, quality) =
                bound_build_stack(tier, BoundBuildFault::PersistedOutputContract);
            let error = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .add_document("replacement", "must not be embedded")
                .build(&cx)
                .await
                .unwrap_err();
            assert!(
                matches!(error, SearchError::InvalidConfig { field, value, .. }
                if field == format!("{tier}_identity.storage") && value == "output_contract")
            );
            for embedder in [&fast, &quality] {
                assert_eq!(embedder.raw_calls.load(Ordering::SeqCst), 0);
                assert_eq!(embedder.bound_calls.load(Ordering::SeqCst), 0);
            }
            for (name, bytes) in &before {
                assert_eq!(std::fs::read(dir.path().join(name)).unwrap(), *bytes);
            }
            let reopened = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
            assert_eq!(reopened.doc_count(), 1);
            assert!(reopened.has_quality_index());
        }
    });
}

/// A producer whose bound batch is its native operation (GH #57). When
/// `native`, its single bound path refuses, so a build that embedded one
/// document at a time would fail every document. When `faulty`, texts
/// steer faults: `reject` fails any batch holding it; `nan`, `short` and
/// `surplus` corrupt that row or the batch's cardinality; `foreign`
/// answers under another identity; `drift` changes the producer's own
/// identity while failing the batch; `cancel` cancels the caller and
/// fails, `racing` cancels it and succeeds.
struct BatchNativeEmbedder {
    inner: IdentityStubEmbedder,
    native: bool,
    faulty: bool,
    foreign_identity: EmbeddingIdentityBundleV1,
    drifted: AtomicBool,
    single_calls: AtomicUsize,
    batches: Mutex<Vec<Vec<String>>>,
}

impl BatchNativeEmbedder {
    fn new(id: &'static str, dim: usize, native: bool, faulty: bool) -> Arc<Self> {
        let inner = IdentityStubEmbedder::new(id, dim);
        let mut foreign_identity = inner.identity.clone();
        foreign_identity
            .producer
            .implementation_revision
            .push_str("-foreign");
        Arc::new(Self {
            inner,
            native,
            faulty,
            foreign_identity,
            drifted: AtomicBool::new(false),
            single_calls: AtomicUsize::new(0),
            batches: Mutex::default(),
        })
    }

    fn batch_widths(&self) -> Vec<usize> {
        self.batches.lock().unwrap().iter().map(Vec::len).collect()
    }
}

impl Embedder for BatchNativeEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.inner.embed(cx, text)
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, frankensearch_core::IdentityBoundEmbedding> {
        self.single_calls.fetch_add(1, Ordering::SeqCst);
        if self.native {
            return Box::pin(async move {
                Err(SearchError::EmbeddingFailed {
                    model: self.id().to_owned(),
                    source: "a batch-native build must not embed one input at a time".into(),
                })
            });
        }
        self.inner.embed_bound(cx, text)
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<frankensearch_core::IdentityBoundEmbedding>> {
        Box::pin(async move {
            self.batches
                .lock()
                .unwrap()
                .push(texts.iter().map(|text| (*text).to_owned()).collect());
            let faulted = |fault: &str| self.faulty && texts.contains(&fault);
            if faulted("reject") {
                return Err(SearchError::EmbeddingFailed {
                    model: self.id().to_owned(),
                    source: "injected batch rejection".into(),
                });
            }
            if faulted("drift") {
                self.drifted.store(true, Ordering::SeqCst);
                return Err(SearchError::EmbeddingFailed {
                    model: self.id().to_owned(),
                    source: "failed while the producer changed".into(),
                });
            }
            if faulted("cancel") || faulted("racing") {
                cx.set_cancel_requested(true);
            }
            if faulted("cancel") {
                return Err(SearchError::EmbeddingFailed {
                    model: self.id().to_owned(),
                    source: "cancelled provider failure".into(),
                });
            }
            let mut outputs = Vec::with_capacity(texts.len());
            for text in texts {
                let mut bound = self.inner.embed_bound(cx, text).await?;
                if self.faulty && *text == "nan" {
                    bound.values[0] = f32::NAN;
                }
                if self.faulty && *text == "foreign" {
                    bound.identity = self.foreign_identity.clone();
                }
                outputs.push(bound);
            }
            if faulted("short") {
                outputs.pop();
            }
            if faulted("surplus") {
                outputs.push(outputs[0].clone());
            }
            Ok(outputs)
        })
    }

    fn bound_batch_is_native(&self) -> bool {
        self.native
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        if self.drifted.load(Ordering::SeqCst) {
            Ok(&self.foreign_identity)
        } else {
            self.inner.identity()
        }
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
    fn is_semantic(&self) -> bool {
        true
    }
    fn category(&self) -> ModelCategory {
        self.inner.category()
    }
}

fn batch_native_stack(
    native: [bool; 2],
    faulty_tier: &str,
) -> (
    EmbedderStack,
    Arc<BatchNativeEmbedder>,
    Arc<BatchNativeEmbedder>,
) {
    let fast = BatchNativeEmbedder::new("batch-fast", 4, native[0], faulty_tier == "fast");
    let quality = BatchNativeEmbedder::new("batch-quality", 6, native[1], faulty_tier == "quality");
    let stack = EmbedderStack::from_parts(
        Arc::clone(&fast) as Arc<dyn Embedder>,
        Some(Arc::clone(&quality) as Arc<dyn Embedder>),
    );
    (stack, fast, quality)
}

fn indexed_vectors(path: &Path) -> std::collections::BTreeMap<String, Vec<f32>> {
    let index = frankensearch_index::VectorIndex::open_read_only(path).unwrap();
    (0..index.record_count())
        .map(|slot| {
            (
                index.doc_id_at(slot).unwrap().to_owned(),
                index.vector_at_f32(slot).unwrap(),
            )
        })
        .collect()
}

/// GH #57: a batch-native producer embeds each `batch_size` group in one
/// bound call per tier — tails and `batch_size` 1 included — and writes
/// exactly the vectors its single-input path would.
#[test]
fn batch_native_build_embeds_each_group_in_one_bound_call() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let docs = [
            ("d1", "alpha"),
            ("d2", "bravo two"),
            ("d3", "alpha"),
            ("d4", "charlie three"),
            ("d5", "delta"),
        ];
        for (batch_size, widths) in [(2, vec![2, 2, 1]), (1, vec![1; 5]), (8, vec![5])] {
            let dir = tempfile::tempdir().unwrap();
            let exporter = Arc::new(RecordingExporter::default());
            let (stack, fast, quality) = batch_native_stack([true, true], "");
            let mut builder = IndexBuilder::new(dir.path())
                .with_config(TwoTierConfig::default().with_metrics_exporter(exporter.clone()))
                .with_embedder_stack(stack)
                .with_batch_size(batch_size);
            for (id, text) in docs {
                builder = builder.add_document(id, text);
            }
            let stats = builder.build(&cx).await.unwrap();

            assert_eq!(stats.doc_count, 5);
            assert_eq!(stats.quality_indexed, 5);
            assert!(stats.errors.is_empty() && stats.quality_errors.is_empty());
            let reported_widths = exporter
                .embedding
                .lock()
                .unwrap()
                .iter()
                .map(|metrics| metrics.batch_size)
                .collect::<Vec<_>>();
            assert_eq!(
                reported_widths,
                widths.iter().flat_map(|&w| [w, w]).collect::<Vec<_>>(),
                "metrics report the real request width, fast then quality per group"
            );
            for (name, embedder) in [
                (VECTOR_INDEX_FAST_FILENAME, &fast),
                (VECTOR_INDEX_QUALITY_FILENAME, &quality),
            ] {
                assert_eq!(embedder.single_calls.load(Ordering::SeqCst), 0);
                assert_eq!(embedder.batch_widths(), widths);
                let vectors = indexed_vectors(&dir.path().join(name));
                assert_eq!(vectors.len(), docs.len());
                for (id, text) in docs {
                    let single = embedder.inner.embed(&cx, text).await.unwrap();
                    assert_eq!(vectors[id], single, "{name} vector for {id}");
                }
            }
        }
    });
}

/// GH #57: an input that fails its batch is found by bisection with the
/// same producer and fails alone — in the fast tier it enters no vector
/// tier, in the quality tier it stays fast-only — while its neighbours
/// keep their vectors.
#[test]
fn batch_native_build_isolates_a_bad_document_by_bisection() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in ["fast", "quality"] {
            for fault in ["reject", "nan", "short", "surplus"] {
                let dir = tempfile::tempdir().unwrap();
                let (stack, fast, quality) = batch_native_stack([true, true], tier);
                let stats = IndexBuilder::new(dir.path())
                    .with_embedder_stack(stack)
                    .with_batch_size(4)
                    .add_document("a", "first")
                    .add_document("b", "second")
                    .add_document("bad", fault)
                    .add_document("d", "fourth")
                    .build(&cx)
                    .await
                    .unwrap();

                let case = format!("{tier}/{fault}");
                let (faulty, clean) = if tier == "fast" {
                    (&fast, &quality)
                } else {
                    (&quality, &fast)
                };
                assert_eq!(faulty.batch_widths(), [4, 2, 2, 1, 1], "{case}");
                assert_eq!(faulty.batches.lock().unwrap()[3], [fault], "{case}");
                let clean_widths = if tier == "fast" { [3] } else { [4] };
                assert_eq!(clean.batch_widths(), clean_widths, "{case}");
                let failed = if tier == "fast" {
                    assert_eq!(stats.doc_count, 3, "{case}");
                    assert!(stats.quality_errors.is_empty(), "{case}");
                    &stats.errors
                } else {
                    assert_eq!(stats.doc_count, 4, "{case}");
                    assert!(stats.errors.is_empty(), "{case}");
                    &stats.quality_errors
                };
                assert_eq!(
                    failed.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
                    ["bad"],
                    "{case}"
                );
                assert_eq!(stats.quality_indexed, 3, "{case}");
                let name = if tier == "fast" {
                    VECTOR_INDEX_FAST_FILENAME
                } else {
                    VECTOR_INDEX_QUALITY_FILENAME
                };
                let vectors = indexed_vectors(&dir.path().join(name));
                assert_eq!(
                    vectors.keys().map(String::as_str).collect::<Vec<_>>(),
                    ["a", "b", "d"],
                    "{case}"
                );
                assert!(vectors.values().flatten().all(|value| value.is_finite()));
                // bd-8nqz.3: the failed document is still lexically searchable.
                #[cfg(feature = "quill")]
                {
                    let receipt = stats.lexical.expect("default Quill receipt");
                    assert_eq!(receipt.indexed, 4, "{case}");
                    let lexical = QuillIndex::open(&cx, &receipt.path, QuillConfig::default())
                        .await
                        .unwrap();
                    let hits = lexical.search_results(&cx, fault, 10).unwrap();
                    assert_eq!(hits.len(), 1, "{case}");
                    assert_eq!(hits[0].doc_id, "bad", "{case}");
                }
            }
        }
    });
}

/// GH #57: a foreign response identity, a producer that changed while its
/// batch failed, or a cancellation (failed or racing a success) ends the
/// build without splitting the batch, is exported, and writes nothing.
#[test]
fn batch_native_build_stops_on_foreign_identity_or_cancellation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in ["fast", "quality"] {
            for fault in ["foreign", "drift", "cancel", "racing"] {
                let dir = tempfile::tempdir().unwrap();
                let exporter = Arc::new(RecordingExporter::default());
                let (stack, fast, quality) = batch_native_stack([true, true], tier);
                let error = IndexBuilder::new(dir.path())
                    .with_config(TwoTierConfig::default().with_metrics_exporter(exporter.clone()))
                    .with_embedder_stack(stack)
                    .with_batch_size(4)
                    .add_document("a", "first")
                    .add_document("bad", fault)
                    .add_document("c", "third")
                    .build(&cx)
                    .await
                    .unwrap_err();
                cx.set_cancel_requested(false);

                let case = format!("{tier}/{fault}");
                if matches!(fault, "cancel" | "racing") {
                    assert!(matches!(error, SearchError::Cancelled { .. }), "{case}");
                } else {
                    assert!(
                        matches!(&error, SearchError::UnverifiableRemoteSpace { producer, .. }
                        if *producer == format!("index_builder.{tier}")),
                        "{case}: {error:?}"
                    );
                }
                let faulty = if tier == "fast" { &fast } else { &quality };
                assert_eq!(faulty.batch_widths(), [3], "{case}: no split");
                assert_eq!(exporter.errors.lock().unwrap().len(), 1, "{case}");
                assert!(!dir.path().join(VECTOR_INDEX_FAST_FILENAME).exists());
                assert!(!dir.path().join(VECTOR_INDEX_QUALITY_FILENAME).exists());
            }
        }
    });
}

/// GH #57: a producer that does not declare a native bound batch — the
/// core default, which a provider overriding only `embed_bound` keeps —
/// is embedded one document at a time, beside a tier that does batch.
#[test]
fn bound_producers_without_native_batches_keep_one_call_per_document() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for native in [[false, false], [true, false], [false, true]] {
            let dir = tempfile::tempdir().unwrap();
            let (stack, fast, quality) = batch_native_stack(native, "");
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .add_document("a", "first")
                .add_document("b", "second")
                .add_document("c", "third")
                .build(&cx)
                .await
                .unwrap();
            assert_eq!((stats.doc_count, stats.quality_indexed), (3, 3));
            for (embedder, batches) in [(&fast, native[0]), (&quality, native[1])] {
                let (widths, singles) = if batches { (vec![3], 0) } else { (vec![], 3) };
                assert_eq!(embedder.batch_widths(), widths, "{native:?}");
                assert_eq!(embedder.single_calls.load(Ordering::SeqCst), singles);
            }
        }
    });
}
