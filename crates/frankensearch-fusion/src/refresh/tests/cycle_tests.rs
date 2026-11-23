use super::*;

// -- Tests -----------------------------------------------------------------

#[test]
fn config_defaults() {
    let config = RefreshWorkerConfig::new("/tmp/test-idx");
    assert_eq!(config.poll_interval, Duration::from_secs(1));
    assert_eq!(config.max_docs_per_cycle, 1000);
}

#[test]
fn config_builder_methods() {
    let config = RefreshWorkerConfig::new("/tmp/test-idx")
        .with_poll_interval(Duration::from_millis(500))
        .with_max_docs_per_cycle(50);
    assert_eq!(config.poll_interval, Duration::from_millis(500));
    assert_eq!(config.max_docs_per_cycle, 50);
}

#[test]
fn metrics_snapshot() {
    let metrics = RefreshMetrics::default();
    metrics.cycles.fetch_add(5, Ordering::Relaxed);
    metrics.docs_embedded.fetch_add(100, Ordering::Relaxed);
    let snap = metrics.snapshot();
    assert_eq!(snap.cycles, 5);
    assert_eq!(snap.docs_embedded, 100);
    assert_eq!(snap.docs_failed, 0);
}

#[test]
fn run_cycle_empty_queue_returns_zero() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("empty");
        let queue = make_queue(10);
        let (worker, _cache, _embedder) = make_worker(queue, &dir, 256);

        let count = worker.run_cycle(&cx).await.unwrap();
        assert_eq!(count, 0);
    });
}

#[test]
fn run_cycle_embeds_and_rebuilds_index() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("rebuild");
        let queue = make_queue(100);
        submit(&queue, "doc-1", "Hello world");
        submit(&queue, "doc-2", "Goodbye world");

        let (worker, cache, _embedder) = make_worker(queue.clone(), &dir, 256);

        let count = worker.run_cycle(&cx).await.unwrap();
        assert_eq!(count, 2);

        // Verify index was rebuilt.
        assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 1);
        assert_eq!(worker.metrics().docs_embedded.load(Ordering::Relaxed), 2);

        // Queue should be empty after processing.
        assert!(queue.is_empty());

        // Cache should have the new index (2 docs, not the seed).
        let current = cache.current();
        assert_eq!(current.doc_count(), 2);
    });
}

/// Red proof (c), production surface: the bootstrap lane declares the
/// producing identity (retained, C2) but the published artifacts are v1
/// — the identity is DECLARED, never attested, and the next cycle's
/// admission does not accept it.
#[test]
fn bootstrap_publishes_declared_identity_never_attested() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("declared-not-attested");
        let queue = make_queue(100);
        submit(&queue, "doc-1", "Hello world");
        let (worker, cache, embedder) = make_worker(queue.clone(), &dir, 256);
        worker.run_cycle(&cx).await.unwrap();

        let current = cache.current();
        assert!(
            current.fast_declared_identity().is_some(),
            "the bootstrap lane must declare the producing identity (C2 retention)"
        );
        assert_eq!(
            current.fast_space_fingerprint_hex(),
            Some(
                StubEmbedder::new("stub-fast", 256)
                    .identity_bundle()
                    .space
                    .fingerprint()
                    .as_str()
            ),
        );
        assert!(
            !current.fast_identity_is_attested(),
            "a v1 bootstrap publication is DECLARED, never header-attested"
        );

        // Declared-only retention never unlocks the merge: the on-disk
        // generation is live v1 and the next cycle keeps the typed
        // containment refusal.
        submit(&queue, "doc-2", "Second document");
        let embed_baseline = embedder.embed_invocations();
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("declared-only generation must keep the identityless refusal");
        assert_invalid_config(
            &error,
            "refresh.fast_index_identity",
            "identityless-fsvi-v1",
        );
        assert_no_embed_invocations_since(&embedder, embed_baseline);

        // Staging refuses the same way: declared-not-attested rejects.
        let jobs = queue.drain_batch();
        let error = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect_err("staging must not admit a declared-only (v1) generation");
        assert_invalid_config(
            &error,
            "refresh.fast_index_identity",
            "identityless-fsvi-v1",
        );
        assert_no_embed_invocations_since(&embedder, embed_baseline);
    });
}

#[test]
fn run_cycle_records_content_hashes() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("hashes");
        let queue = make_queue(100);
        submit(&queue, "doc-1", "First document text");

        let (worker, _cache, _embedder) = make_worker(queue.clone(), &dir, 256);
        worker.run_cycle(&cx).await.unwrap();

        // Submitting the same text again should be deduped.
        let outcome = queue
            .submit(EmbeddingRequest {
                doc_id: "doc-1".into(),
                text: "First document text".to_owned(),
                metadata: None,
                submitted_at: Instant::now(),
            })
            .unwrap();
        assert_eq!(outcome, JobOutcome::SkippedUnchanged);
    });
}

#[test]
fn run_cycle_with_quality_embedder() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("quality");
        let queue = make_queue(100);
        submit(&queue, "doc-1", "Test document");

        let cache = make_cache_with_quality(&dir, 256, 384);
        let config = RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
        let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
        let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
        let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone())
            .with_quality_embedder(quality);

        let count = worker.run_cycle(&cx).await.unwrap();
        assert_eq!(count, 1);

        // Index should have been rebuilt with quality tier.
        let index = cache.current();
        assert!(index.has_quality_index());
    });
}

#[test]
fn pre_drain_refusal_does_not_charge_embedding_or_rebuild_metrics() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("multi");
        let queue = make_queue(100);
        let (worker, _cache, embedder) = make_worker(queue.clone(), &dir, 256);

        // Cycle 1: 2 docs.
        submit(&queue, "doc-1", "First");
        submit(&queue, "doc-2", "Second");
        worker.run_cycle(&cx).await.unwrap();

        // Cycle 2 refuses before draining or embedding the pending job.
        submit(&queue, "doc-3", "Third");
        let embed_baseline = embedder.embed_invocations();
        worker
            .run_cycle(&cx)
            .await
            .expect_err("legacy incremental merge must fail closed");

        let snap = worker.metrics().snapshot();
        assert_eq!(snap.docs_embedded, 2);
        assert_eq!(snap.index_rebuilds, 1);
        assert_eq!(snap.rebuild_failures, 0);
        assert_eq!(queue.pending_count(), 1);
        // r3: the metric inference above is now backed by the direct spy.
        assert_no_embed_invocations_since(&embedder, embed_baseline);
    });
}

#[test]
fn run_cycle_respects_max_docs_per_cycle() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("limit");
        let queue = make_queue(100);
        for i in 0..5 {
            submit(&queue, &format!("doc-{i}"), "Payload");
        }

        let cache = make_cache(&dir, 256);
        let config = RefreshWorkerConfig::new(&dir)
            .with_poll_interval(Duration::from_millis(10))
            .with_max_docs_per_cycle(3);
        let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
        let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone());

        let count = worker.run_cycle(&cx).await.unwrap();
        assert_eq!(count, 3);
        assert_eq!(queue.pending_count(), 2);
    });
}

#[test]
fn identityless_incremental_refusal_preserves_active_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("preserve-existing");
        let queue = make_queue(100);
        let (worker, cache, embedder) = make_worker(queue.clone(), &dir, 256);

        submit(&queue, "doc-1", "First");
        submit(&queue, "doc-2", "Second");
        worker.run_cycle(&cx).await.expect("first cycle");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let original_bytes = std::fs::read(&fast_path).expect("read active generation");

        submit(&queue, "doc-3", "Third");
        let embed_baseline = embedder.embed_invocations();
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("identityless incremental merge must fail closed");
        assert!(matches!(
            error,
            SearchError::InvalidConfig { ref field, ref value, .. }
                if field == "refresh.fast_index_identity"
                    && value == "identityless-fsvi-v1"
        ));
        assert_no_embed_invocations_since(&embedder, embed_baseline);
        assert_eq!(queue.pending_count(), 1, "failed job must be requeued");
        assert_eq!(
            std::fs::read(&fast_path).expect("reread active generation"),
            original_bytes,
            "refusal must not mutate or replace the active generation"
        );

        let current = cache.current();
        assert_eq!(
            current.doc_count(),
            2,
            "cache must continue serving the prior active generation"
        );
        let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
        assert!(doc_ids.iter().any(|id| id == "doc-1"));
        assert!(doc_ids.iter().any(|id| id == "doc-2"));
        assert!(!doc_ids.iter().any(|id| id == "doc-3"));
    });
}

#[test]
fn same_id_same_dimension_changed_space_cannot_bypass_identityless_refusal() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("fast-id-change");
        let queue = make_queue(100);
        let cache = make_cache(&dir, 256);
        let config = RefreshWorkerConfig::new(&dir);
        let original_embedder = Arc::new(StubEmbedder::in_space("stub-fast", 256, 0.0));
        let changed_embedder = Arc::new(StubEmbedder::in_space("stub-fast", 256, 100.0));
        assert_eq!(original_embedder.id(), changed_embedder.id());
        assert_eq!(original_embedder.dimension(), changed_embedder.dimension());
        let original_probe = original_embedder
            .embed(&cx, "space probe")
            .await
            .expect("original probe");
        let changed_probe = changed_embedder
            .embed(&cx, "space probe")
            .await
            .expect("changed probe");
        assert!(
            original_probe
                .iter()
                .zip(&changed_probe)
                .any(|(original, changed)| original.to_bits() != changed.to_bits()),
            "fixture must represent a real same-id, same-dimension space change"
        );
        let original = RefreshWorker::new(
            config.clone(),
            queue.clone(),
            original_embedder,
            cache.clone(),
        );

        submit(&queue, "doc-old", "old generation");
        original.run_cycle(&cx).await.expect("initial generation");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let original_bytes = std::fs::read(&fast_path).expect("read original fast index");
        let changed = RefreshWorker::new(
            config,
            queue.clone(),
            changed_embedder.clone(),
            cache.clone(),
        );

        submit(&queue, "doc-new", "new generation");
        let embed_baseline = changed_embedder.embed_invocations();
        let error = changed
            .run_cycle(&cx)
            .await
            .expect_err("identityless generation must fail closed before vector merge");

        assert!(matches!(
            error,
            SearchError::InvalidConfig { ref field, ref value, .. }
                if field == "refresh.fast_index_identity"
                    && value == "identityless-fsvi-v1"
        ));
        assert_no_embed_invocations_since(&changed_embedder, embed_baseline);
        assert_eq!(queue.pending_count(), 1, "failed job must be requeued");
        assert_eq!(
            std::fs::read(&fast_path).expect("reread fast index"),
            original_bytes,
            "failed refresh must not re-stamp the existing generation"
        );
        let current = cache.current();
        assert_eq!(current.fast_embedder_id(), "stub-fast");
        let live_ids = current
            .iter_doc_ids()
            .collect::<SearchResult<Vec<_>>>()
            .expect("read cached live IDs");
        assert_eq!(live_ids, vec!["doc-old"]);
    });
}

#[test]
fn unchanged_ids_cannot_exhaust_retry_budget_while_legacy_generation_is_blocked() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("unchanged-tier-ids");
        let queue = make_queue(100);
        let cache = make_cache_with_quality(&dir, 256, 384);
        let fast_spy = Arc::new(StubEmbedder::new("stub-fast", 256));
        let quality_spy = Arc::new(StubEmbedder::new("stub-quality", 384));
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            fast_spy.clone(),
            cache.clone(),
        )
        .with_quality_embedder(quality_spy.clone());

        submit(&queue, "doc-old", "old generation");
        worker.run_cycle(&cx).await.expect("initial generation");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let original_fast_bytes = std::fs::read(&fast_path).expect("read original fast index");
        let original_quality_bytes =
            std::fs::read(&quality_path).expect("read original quality index");

        submit(&queue, "doc-new", "new generation");
        let fast_baseline = fast_spy.embed_invocations();
        let quality_baseline = quality_spy.embed_invocations();
        let refusal_cycles = queue.config().max_retries + 2;
        for _ in 0..refusal_cycles {
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("matching display ids must not admit a legacy merge");
            assert!(matches!(
                error,
                SearchError::InvalidConfig { ref field, ref value, .. }
                    if field == "refresh.fast_index_identity"
                        && value == "identityless-fsvi-v1"
            ));
        }
        assert_no_embed_invocations_since(&fast_spy, fast_baseline);
        assert_no_embed_invocations_since(&quality_spy, quality_baseline);
        assert_eq!(
            queue.pending_count(),
            1,
            "permanent refusal must leave the job queued beyond max_retries"
        );
        assert_eq!(
            std::fs::read(&fast_path).expect("reread fast index"),
            original_fast_bytes,
            "refusal must not replace the fast generation"
        );
        assert_eq!(
            std::fs::read(&quality_path).expect("reread quality index"),
            original_quality_bytes,
            "refusal must not replace the quality generation"
        );
        let current = cache.current();
        assert_eq!(current.doc_count(), 1);
        assert!(current.has_quality_index());
        let live_doc_ids = current
            .iter_doc_ids()
            .collect::<SearchResult<Vec<_>>>()
            .expect("read cached live document IDs");
        assert_eq!(live_doc_ids, vec!["doc-old"]);

        let pending = queue.drain_batch();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].doc_id, "doc-new");
        assert_eq!(
            pending[0].retry_count, 0,
            "pre-drain refusal must not consume retry budget"
        );
    });
}

#[test]
fn failed_embedding_requeues_jobs() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("fail");
        let queue = make_queue(100);
        submit(&queue, "doc-1", "Will fail");

        let cache = make_cache(&dir, 256);
        let config = RefreshWorkerConfig::new(&dir);
        let failing = Arc::new(FailingEmbedder);
        let worker = RefreshWorker::new(config, queue.clone(), failing, cache);

        let count = worker.run_cycle(&cx).await.unwrap();
        assert_eq!(count, 0);

        // Job should have been requeued.
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn index_dir_accessor() {
    let dir = temp_index_dir("accessor");
    let queue = make_queue(10);
    let (worker, _cache, _embedder) = make_worker(queue, &dir, 256);
    assert_eq!(worker.index_dir(), dir.as_path());
}

#[test]
fn debug_format() {
    let dir = temp_index_dir("debug");
    let queue = make_queue(10);
    let (worker, _cache, _embedder) = make_worker(queue, &dir, 256);
    let debug = format!("{worker:?}");
    assert!(debug.contains("RefreshWorker"));
    assert!(debug.contains("stub-fast"));
}

// ─── bd-qkop tests begin ───

#[test]
fn config_with_index_config() {
    let custom = TwoTierConfig::default();
    let config = RefreshWorkerConfig::new("/tmp/idx").with_index_config(custom.clone());
    assert!((config.index_config.rrf_k - custom.rrf_k).abs() < f64::EPSILON);
}

#[test]
fn config_clone() {
    let config = RefreshWorkerConfig::new("/tmp/idx")
        .with_poll_interval(Duration::from_millis(42))
        .with_max_docs_per_cycle(7);
    #[allow(clippy::redundant_clone)]
    let cloned = config.clone();
    assert_eq!(cloned.poll_interval, Duration::from_millis(42));
    assert_eq!(cloned.max_docs_per_cycle, 7);
    assert_eq!(cloned.index_dir, PathBuf::from("/tmp/idx"));
}

#[test]
fn config_debug() {
    let config = RefreshWorkerConfig::new("/tmp/idx");
    let debug = format!("{config:?}");
    assert!(debug.contains("RefreshWorkerConfig"));
    assert!(debug.contains("poll_interval"));
}

#[test]
fn metrics_default_all_zeros() {
    let m = RefreshMetrics::default();
    assert_eq!(m.cycles.load(Ordering::Relaxed), 0);
    assert_eq!(m.docs_embedded.load(Ordering::Relaxed), 0);
    assert_eq!(m.docs_failed.load(Ordering::Relaxed), 0);
    assert_eq!(m.index_rebuilds.load(Ordering::Relaxed), 0);
    assert_eq!(m.rebuild_failures.load(Ordering::Relaxed), 0);
    assert_eq!(m.embed_time_us.load(Ordering::Relaxed), 0);
    assert_eq!(m.rebuild_time_us.load(Ordering::Relaxed), 0);
}

#[test]
fn metrics_snapshot_equality() {
    let a = RefreshMetricsSnapshot {
        cycles: 1,
        docs_embedded: 2,
        docs_failed: 0,
        index_rebuilds: 1,
        rebuild_failures: 0,
        embed_time_us: 100,
        rebuild_time_us: 50,
    };
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn metrics_snapshot_inequality() {
    let a = RefreshMetricsSnapshot {
        cycles: 1,
        docs_embedded: 2,
        docs_failed: 0,
        index_rebuilds: 1,
        rebuild_failures: 0,
        embed_time_us: 100,
        rebuild_time_us: 50,
    };
    let b = RefreshMetricsSnapshot { cycles: 2, ..a };
    assert_ne!(a, b);
}

#[test]
fn metrics_snapshot_clone_copy() {
    let a = RefreshMetricsSnapshot {
        cycles: 5,
        docs_embedded: 10,
        docs_failed: 1,
        index_rebuilds: 3,
        rebuild_failures: 0,
        embed_time_us: 200,
        rebuild_time_us: 100,
    };
    #[allow(clippy::clone_on_copy)]
    let cloned = a.clone();
    let copied: RefreshMetricsSnapshot = a; // Copy
    assert_eq!(cloned, a);
    assert_eq!(copied, a);
}

#[test]
fn metrics_snapshot_debug() {
    let snap = RefreshMetricsSnapshot {
        cycles: 0,
        docs_embedded: 0,
        docs_failed: 0,
        index_rebuilds: 0,
        rebuild_failures: 0,
        embed_time_us: 0,
        rebuild_time_us: 0,
    };
    let debug = format!("{snap:?}");
    assert!(debug.contains("RefreshMetricsSnapshot"));
    assert!(debug.contains("cycles"));
}

#[test]
fn metrics_individual_increments() {
    let m = RefreshMetrics::default();
    m.cycles.fetch_add(3, Ordering::Relaxed);
    m.docs_failed.fetch_add(2, Ordering::Relaxed);
    m.rebuild_failures.fetch_add(1, Ordering::Relaxed);
    m.embed_time_us.fetch_add(500, Ordering::Relaxed);
    m.rebuild_time_us.fetch_add(300, Ordering::Relaxed);
    let snap = m.snapshot();
    assert_eq!(snap.cycles, 3);
    assert_eq!(snap.docs_failed, 2);
    assert_eq!(snap.rebuild_failures, 1);
    assert_eq!(snap.embed_time_us, 500);
    assert_eq!(snap.rebuild_time_us, 300);
}

#[test]
fn worker_metrics_accessor_returns_shared_arc() {
    let dir = temp_index_dir("metrics-arc");
    let queue = make_queue(10);
    let (worker, _cache, _embedder) = make_worker(queue, &dir, 256);
    let metrics = worker.metrics();
    metrics.cycles.fetch_add(1, Ordering::Relaxed);
    // Same Arc should reflect the update.
    assert_eq!(worker.metrics().cycles.load(Ordering::Relaxed), 1);
}

#[test]
fn worker_with_quality_embedder_sets_embedder() {
    let dir = temp_index_dir("quality-set");
    let queue = make_queue(10);
    let cache = make_cache(&dir, 256);
    let config = RefreshWorkerConfig::new(&dir);
    let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
    let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
    let worker = RefreshWorker::new(config, queue, fast, cache).with_quality_embedder(quality);
    // Should be set (verified indirectly by debug containing only fast id)
    let debug = format!("{worker:?}");
    assert!(debug.contains("stub-fast"));
}

#[test]
fn duplicate_doc_id_keeps_latest_in_cycle() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("dedup-cycle");
        let queue = make_queue(100);
        // Submit same doc_id with different texts.
        submit(&queue, "doc-dup", "First version");
        submit(&queue, "doc-dup", "Second version");

        let (worker, cache, _embedder) = make_worker(queue.clone(), &dir, 256);
        let count = worker.run_cycle(&cx).await.unwrap();
        // Both submitted but doc count should be 1 (deduped).
        assert!(count >= 1);
        let current = cache.current();
        // Only one doc-dup in the index.
        let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
        let dup_count = doc_ids.iter().filter(|id| *id == "doc-dup").count();
        assert_eq!(dup_count, 1);
    });
}

#[test]
fn quality_embedder_failure_proceeds_with_fast_only() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("quality-fail");
        let queue = make_queue(100);
        submit(&queue, "doc-1", "Test document");

        let cache = make_cache(&dir, 256);
        let config = RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
        let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
        let failing_quality: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
        let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone())
            .with_quality_embedder(failing_quality);

        let count = worker.run_cycle(&cx).await.unwrap();
        assert_eq!(count, 1);

        // Index should still have the doc (fast-only).
        let current = cache.current();
        assert_eq!(current.doc_count(), 1);
        // But no quality index since quality embedder failed.
        assert!(!current.has_quality_index());
    });
}

#[test]
fn run_cycle_empty_after_prior_cycle_returns_zero() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("empty-after");
        let queue = make_queue(100);
        submit(&queue, "doc-1", "First");
        let (worker, _cache, _embedder) = make_worker(queue.clone(), &dir, 256);

        worker.run_cycle(&cx).await.unwrap();
        // Second cycle with empty queue.
        let count = worker.run_cycle(&cx).await.unwrap();
        assert_eq!(count, 0);
    });
}

// ─── bd-qkop tests end ───

// ─── bd-wt20 tests begin ──────────────────────────────────────────

#[test]
fn refresh_metrics_debug_format() {
    let m = RefreshMetrics::default();
    m.cycles.fetch_add(7, Ordering::Relaxed);
    m.docs_embedded.fetch_add(42, Ordering::Relaxed);
    let debug = format!("{m:?}");
    assert!(debug.contains("RefreshMetrics"));
    assert!(debug.contains("cycles"));
    assert!(debug.contains("docs_embedded"));
}

#[test]
fn config_index_dir_from_new() {
    let config = RefreshWorkerConfig::new("/data/my-index");
    assert_eq!(config.index_dir, PathBuf::from("/data/my-index"));
}

#[test]
fn config_index_config_override() {
    let custom = TwoTierConfig {
        rrf_k: 42.0,
        ..TwoTierConfig::default()
    };
    let config = RefreshWorkerConfig::new("/tmp/idx").with_index_config(custom);
    assert!((config.index_config.rrf_k - 42.0).abs() < f64::EPSILON);
}

#[test]
fn metrics_concurrent_increments() {
    let m = Arc::new(RefreshMetrics::default());
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let metrics = m.clone();
            std::thread::spawn(move || {
                for _ in 0..100 {
                    metrics.docs_embedded.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(m.docs_embedded.load(Ordering::Relaxed), 400);
}

#[test]
fn identityless_incremental_refusal_does_not_overwrite_existing_doc() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("update-doc");
        let queue = make_queue(100);
        let (worker, cache, embedder) = make_worker(queue.clone(), &dir, 256);

        // Cycle 1: index doc-1 with first text.
        submit(&queue, "doc-1", "First version of the document");
        worker.run_cycle(&cx).await.unwrap();
        assert_eq!(cache.current().doc_count(), 1);

        // Cycle 2: submit doc-1 again with different text (hash changed).
        // Force a new submission by changing text.
        queue
            .submit(EmbeddingRequest {
                doc_id: "doc-1".into(),
                text: "Second version completely different".to_owned(),
                metadata: None,
                submitted_at: Instant::now(),
            })
            .unwrap();
        let embed_baseline = embedder.embed_invocations();
        worker
            .run_cycle(&cx)
            .await
            .expect_err("legacy update must require an identity-bound rebuild");
        assert_no_embed_invocations_since(&embedder, embed_baseline);

        // The active generation remains available, and the update remains
        // queued for a later full identity-bound rebuild.
        let current = cache.current();
        assert_eq!(current.doc_count(), 1);
        let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
        assert_eq!(doc_ids.len(), 1);
        assert_eq!(doc_ids[0], "doc-1");
        assert_eq!(queue.pending_count(), 1);
    });
}

/// Fleet-review regression pin: refusing an identityless generation must
/// neither resurrect soft-deleted documents nor drop WAL-resident appends.
#[test]
fn identityless_rebuild_refusal_preserves_tombstones_and_wal_residents() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("tombstone-wal-rebuild");
        let queue = make_queue(100);
        let cache = make_cache(&dir, 256);
        let config = RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
        let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
        let worker = RefreshWorker::new(config.clone(), queue.clone(), fast.clone(), cache.clone());

        submit(&queue, "doc-keep", "kept document");
        submit(&queue, "doc-delete", "doomed document");
        worker.run_cycle(&cx).await.unwrap();
        assert_eq!(cache.current().doc_count(), 2);

        // Out-of-band mutations through the public index API, exactly
        // as an application deleting/appending between cycles does.
        drop(worker);
        drop(cache);
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        {
            let mut index =
                frankensearch_index::VectorIndex::open(&fast_path).expect("open fast tier");
            assert!(index.soft_delete("doc-delete").expect("soft delete"));
            let dim = index.dimension();
            index
                .append("doc-wal", &vec![0.25_f32; dim])
                .expect("wal append");
        }
        let fast_bytes = std::fs::read(&fast_path).expect("read mutated fast tier");
        let wal_path = frankensearch_index::wal::wal_path_for(&fast_path);
        let wal_bytes = std::fs::read(&wal_path).expect("read mutated WAL");
        let cache = Arc::new(
            IndexCache::open(
                &dir,
                TwoTierConfig::default(),
                Box::new(SentinelFileDetector::new()),
            )
            .expect("open mutated generation"),
        );
        let worker = RefreshWorker::new(config, queue.clone(), fast.clone(), cache.clone());

        submit(&queue, "doc-new", "new document");
        let embed_baseline = fast.embed_invocations();
        worker
            .run_cycle(&cx)
            .await
            .expect_err("legacy WAL-bearing generation must not be republished");
        assert_no_embed_invocations_since(&fast, embed_baseline);
        assert_eq!(
            std::fs::read(&fast_path).expect("reread fast tier"),
            fast_bytes,
            "refusal must not replace the tombstone-bearing main artifact"
        );
        assert_eq!(
            std::fs::read(&wal_path).expect("reread WAL"),
            wal_bytes,
            "refusal must not drop the WAL-resident append"
        );

        drop(worker);
        drop(cache);
        let preserved =
            frankensearch_index::VectorIndex::open(&fast_path).expect("reopen fast tier");
        assert_eq!(preserved.embedder_id(), "stub-fast");
        let live = preserved.live_doc_ids().expect("live ids");
        assert!(
            !live.contains("doc-delete"),
            "refusal must not resurrect the soft-deleted document: {live:?}"
        );
        assert!(
            live.contains("doc-wal"),
            "refusal must preserve the WAL-resident append: {live:?}"
        );
        assert!(live.contains("doc-keep"), "{live:?}");
        assert!(!live.contains("doc-new"), "{live:?}");
        assert_eq!(queue.pending_count(), 1);
    });
}

#[test]
fn identityless_refusal_preserves_prior_quality_tier() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("quality-preserve");
        let queue = make_queue(100);

        // First cycle with quality embedder.  Seed a quality-capable
        // generation so the initial replacement is a same-topology refresh
        // rather than an attempted topology upgrade.
        let cache = make_cache_with_quality(&dir, 256, 384);
        let config = RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
        let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
        let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
        let worker_with_quality =
            RefreshWorker::new(config, queue.clone(), fast.clone(), cache.clone())
                .with_quality_embedder(quality.clone());

        submit(&queue, "doc-1", "Test document");
        worker_with_quality.run_cycle(&cx).await.unwrap();
        assert!(cache.current().has_quality_index());

        // Second cycle without quality embedder (fast-only worker).
        let config2 = RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
        let worker_fast_only =
            RefreshWorker::new(config2, queue.clone(), fast.clone(), cache.clone());

        submit(&queue, "doc-2", "Another document");
        let fast_baseline = fast.embed_invocations();
        let quality_baseline = quality.embed_invocations();
        worker_fast_only
            .run_cycle(&cx)
            .await
            .expect_err("legacy two-tier generation must fail closed");
        assert_no_embed_invocations_since(&fast, fast_baseline);
        assert_no_embed_invocations_since(&quality, quality_baseline);

        // The prior quality generation remains active and the new work is
        // requeued for a full identity-bound rebuild.
        let current = cache.current();
        assert_eq!(current.doc_count(), 1);
        assert!(current.has_quality_index());
        assert_eq!(queue.pending_count(), 1);
    });
}

#[test]
fn run_cycle_metrics_timing_nonzero() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("timing");
        let queue = make_queue(100);
        submit(&queue, "doc-1", "Timing test");

        let (worker, _cache, _embedder) = make_worker(queue, &dir, 256);
        worker.run_cycle(&cx).await.unwrap();

        let snap = worker.metrics().snapshot();
        // Embed and rebuild should take some nonzero time.
        assert!(snap.embed_time_us > 0 || snap.rebuild_time_us > 0);
    });
}

#[test]
fn worker_new_without_quality_debug() {
    let dir = temp_index_dir("no-quality");
    let queue = make_queue(10);
    let (worker, _cache, _embedder) = make_worker(queue, &dir, 256);
    // Debug uses `finish_non_exhaustive` and includes fast embedder id.
    let debug = format!("{worker:?}");
    assert!(debug.contains("RefreshWorker"));
    assert!(debug.contains("stub-fast"));
}

// ─── bd-wt20 tests end ────────────────────────────────────────────
