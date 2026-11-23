use std::collections::{BTreeMap, HashMap};
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::Poll;
use std::time::Instant;

// The split capabilities, so `wrapper.search(..)` / `.index_document(..)`
// resolve now that the combined trait no longer covers both.
use crate::traits::{LexicalRead as _, LexicalWrite as _};

use asupersync::Budget;
use asupersync::lab::{LabConfig, LabRuntime};
use asupersync::runtime::RuntimeBuilder;
use compact_str::CompactString;

use super::artifacts::ShadowArtifactSink;
use super::runtime::{ShadowSearchPath, poll_shadow_on_pool};
use super::types::{classify_ranked_hits, ranked_hits};
use super::*;
use crate::types::ScoreSource;
use crate::{Cx, IndexableDocument, ScoredResult, SearchError, SearchFuture, SearchResult};

struct StaticLexical {
    results: Vec<ScoredResult>,
    fail_index: AtomicBool,
    fail_doc_count: AtomicBool,
    search_count: AtomicUsize,
}

impl StaticLexical {
    fn new(results: Vec<ScoredResult>) -> Self {
        Self {
            results,
            fail_index: AtomicBool::new(false),
            fail_doc_count: AtomicBool::new(false),
            search_count: AtomicUsize::new(0),
        }
    }
}

impl crate::traits::LexicalRead for StaticLexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            self.search_count.fetch_add(1, Ordering::Relaxed);
            Ok(self.results.iter().take(limit).cloned().collect())
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        if self.fail_doc_count.load(Ordering::Acquire) {
            return Err(SearchError::InvalidConfig {
                field: "shadow.test.doc_count".to_owned(),
                value: "unavailable".to_owned(),
                reason: "injected serving document-count failure".to_owned(),
            });
        }
        Ok(self.results.len())
    }
}

impl crate::traits::LexicalWrite for StaticLexical {
    fn index_document<'a>(
        &'a self,
        _cx: &'a Cx,
        _document: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            if self.fail_index.load(Ordering::Acquire) {
                Err(SearchError::InvalidConfig {
                    field: "shadow.test".to_owned(),
                    value: "fail".to_owned(),
                    reason: "injected shadow index failure".to_owned(),
                })
            } else {
                Ok(())
            }
        })
    }

    fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

struct SlowShadow {
    delay: std::time::Duration,
}

impl crate::traits::LexicalRead for SlowShadow {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        _limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            std::thread::sleep(self.delay);
            Ok(Vec::new())
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Ok(0)
    }
}

impl crate::traits::LexicalWrite for SlowShadow {
    fn index_document<'a>(
        &'a self,
        _cx: &'a Cx,
        _document: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

fn result(id: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: CompactString::from(id),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

fn enabled_config(root: &Path) -> ShadowLexicalConfig {
    ShadowLexicalConfig {
        enabled: true,
        artifact_directory: root.join(SHADOW_ARTIFACT_DIRECTORY),
        ..ShadowLexicalConfig::default()
    }
}

#[test]
fn classifier_covers_exact_epsilon_tie_rank_and_score_mismatch() {
    let exact = ranked_hits(&[result("a", 1.0), result("b", 0.5)]);
    assert_eq!(
        classify_ranked_hits(&exact, &exact, 0.01),
        ShadowDivergenceClass::Exact
    );
    let epsilon = ranked_hits(&[result("a", 1.005), result("b", 0.5)]);
    assert_eq!(
        classify_ranked_hits(&exact, &epsilon, 0.01),
        ShadowDivergenceClass::ScoreEpsilon
    );
    let tie = ranked_hits(&[result("b", 0.5), result("a", 1.0)]);
    assert_eq!(
        classify_ranked_hits(&exact, &tie, 0.01),
        ShadowDivergenceClass::TieOrder
    );
    let rank = ranked_hits(&[result("a", 1.0), result("c", 0.5)]);
    assert_eq!(
        classify_ranked_hits(&exact, &rank, 0.01),
        ShadowDivergenceClass::RankMismatch
    );
    let score = ranked_hits(&[result("a", 0.8), result("b", 0.5)]);
    assert_eq!(
        classify_ranked_hits(&exact, &score, 0.01),
        ShadowDivergenceClass::ScoreMismatch
    );
}

#[test]
fn schema_roundtrip_and_doctor_summary_reject_malformed_lines() {
    let temp = tempfile::tempdir().expect("tempdir");
    let directory = temp.path().join(SHADOW_ARTIFACT_DIRECTORY);
    let sink = ShadowArtifactSink::new(directory.clone());
    let documents = vec![ShadowDocument {
        id: "doc".to_owned(),
        content: "alpha".to_owned(),
        title: None,
        metadata: BTreeMap::new(),
    }];
    let record = ShadowDivergenceRecord {
        schema_version: SHADOW_DIVERGENCE_SCHEMA_VERSION,
        manifest_generation: 7,
        corpus_hash: compute_shadow_corpus_hash(&documents),
        documents,
        query: ShadowQuery {
            text: "alpha".to_owned(),
            limit: 10,
            fusion_candidates: true,
        },
        serving_top_k: ranked_hits(&[result("doc", 1.0)]),
        shadow_top_k: ranked_hits(&[]),
        classification: ShadowDivergenceClass::RankMismatch,
        serve_latency_micros: 4,
        shadow_latency_micros: 9,
    };
    sink.append_divergence(&record).expect("append");
    let line = fs::read_to_string(directory.join(SHADOW_DIVERGENCES_FILE)).expect("read artifact");
    let decoded: ShadowDivergenceRecord = serde_json::from_str(line.trim()).expect("roundtrip");
    assert_eq!(decoded, record);
    decoded.validate().expect("schema validates");

    let mut malformed = OpenOptions::new()
        .append(true)
        .open(directory.join(SHADOW_DIVERGENCES_FILE))
        .expect("open append");
    writeln!(malformed, "{{not-json").expect("append malformed");
    let summary =
        ShadowArtifactSummary::read_artifact_directory(directory).expect("doctor summary");
    assert_eq!(summary.divergence_count, 1);
    assert_eq!(summary.malformed_line_count, 1);
    assert_eq!(summary.latest_generation, Some(7));
}

#[test]
fn deterministic_sampling_and_pressure_shedding_never_query_shadow() {
    let temp = tempfile::tempdir().expect("tempdir");
    let serving = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
    let shadow = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
    let pressure = Arc::new(AtomicShadowLoadProbe::default());
    let wrapper = ShadowLexical::with_load_probe(
        serving,
        shadow.clone(),
        ShadowLexicalConfig {
            sample_rate_basis_points: 2_500,
            ..enabled_config(temp.path())
        },
        pressure.clone(),
    )
    .expect("wrapper");
    let cx = Cx::for_testing();
    let serving_results = vec![result("a", 1.0)];
    for _ in 0..10_000 {
        wrapper.submit_shadow(
            &cx,
            "alpha",
            1,
            &serving_results,
            ShadowSearchPath::Search,
            1,
        );
    }
    let status = wrapper.status();
    assert_eq!(status.sampled, 2_500);
    assert_eq!(status.sampled_out, 7_500);
    assert_eq!(status.shed, 2_500);
    assert_eq!(shadow.search_count.load(Ordering::Acquire), 0);

    pressure.set_cpu_pressure(true);
    wrapper.submit_shadow(&cx, "cpu", 1, &serving_results, ShadowSearchPath::Search, 1);
    assert_eq!(shadow.search_count.load(Ordering::Acquire), 0);
    pressure.set_cpu_pressure(false);
    pressure.set_io_pressure(true);
    wrapper.submit_shadow(&cx, "io", 1, &serving_results, ShadowSearchPath::Search, 1);
    assert_eq!(shadow.search_count.load(Ordering::Acquire), 0);
    assert_eq!(wrapper.status().shed, 2_502);
}

#[test]
fn document_count_propagates_serving_authority_failure() {
    let temp = tempfile::tempdir().expect("tempdir");
    let serving = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
    let shadow = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
    let wrapper =
        ShadowLexical::new(serving.clone(), shadow, enabled_config(temp.path())).expect("wrapper");

    assert_eq!(wrapper.doc_count().expect("serving count"), 1);
    serving.fail_doc_count.store(true, Ordering::Release);
    let error = wrapper
        .doc_count()
        .expect_err("shadow wrapper must not fabricate a serving count");
    assert!(matches!(
        error,
        SearchError::InvalidConfig { field, .. }
            if field == "shadow.test.doc_count"
    ));
}

#[test]
fn shadow_index_failure_is_typed_persisted_and_never_fails_serving() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().expect("tempdir");
        let serving = Arc::new(StaticLexical::new(Vec::new()));
        let shadow = Arc::new(StaticLexical::new(Vec::new()));
        shadow.fail_index.store(true, Ordering::Release);
        let wrapper =
            ShadowLexical::new(serving, shadow, enabled_config(temp.path())).expect("wrapper");
        wrapper
            .index_document(&cx, &IndexableDocument::new("doc", "alpha"))
            .await
            .expect("serving indexing must remain healthy");
        assert_eq!(wrapper.status().degradations, 1);
        let summary = ShadowArtifactSummary::read_index_root(temp.path()).expect("doctor summary");
        assert_eq!(summary.degradation_count, 1);
    });
}

#[test]
fn labruntime_shadow_result_is_non_interfering_and_region_reaches_quiescence() {
    let temp = tempfile::tempdir().expect("tempdir");
    let serving_result = vec![result("serving", 3.0), result("second", 2.0)];
    let serving = Arc::new(StaticLexical::new(serving_result.clone()));
    let shadow = Arc::new(StaticLexical::new(vec![result("shadow", 9.0)]));
    let mut config = enabled_config(temp.path());
    config.allow_inline_shadow = true;
    let wrapper = Arc::new(ShadowLexical::new(serving, shadow, config).expect("wrapper"));
    wrapper.seed_corpus(&[IndexableDocument::new("serving", "alpha")], 0);

    let mut lab = LabRuntime::new(LabConfig::new(0x5ad0_0001).max_steps(100_000));
    let region = lab.state.create_root_region(Budget::INFINITE);
    let task_wrapper = wrapper.clone();
    let (task, handle) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            let cx = Cx::current().expect("LabRuntime task context");
            let actual = task_wrapper
                .search(&cx, "alpha", 10)
                .await
                .expect("serving search");
            assert_eq!(
                ranked_hits(&actual),
                ranked_hits(&serving_result),
                "serving top-k must remain bit-identical"
            );
        })
        .expect("create task");
    lab.scheduler.lock().schedule(task, 0);
    let report = lab.run_until_quiescent_with_report();
    assert!(report.quiescent, "seed=0x5ad00001");
    assert!(report.oracle_report.all_passed(), "seed=0x5ad00001");
    assert!(report.invariant_violations.is_empty(), "seed=0x5ad00001");
    drop(handle);
    assert_eq!(wrapper.status().in_flight, 0);
    let summary = ShadowArtifactSummary::read_index_root(temp.path()).expect("summary");
    assert_eq!(summary.divergence_count, 1);
}

#[test]
fn wall_clock_guard_never_awaits_slow_shadow_backend() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wrapper = Arc::new(
        ShadowLexical::new(
            Arc::new(StaticLexical::new(vec![result("serving", 1.0)])),
            Arc::new(SlowShadow {
                delay: std::time::Duration::from_millis(250),
            }),
            enabled_config(temp.path()),
        )
        .expect("wrapper"),
    );
    let runtime = RuntimeBuilder::current_thread()
        .build()
        .expect("build runtime");
    let task_wrapper = wrapper.clone();
    let started = Instant::now();
    let serving_task = runtime.handle().spawn(async move {
        let cx = Cx::current().expect("runtime installs a spawn-capable context");
        task_wrapper
            .search(&cx, "latency guard", 10)
            .await
            .expect("serving result")
    });
    let actual = runtime.block_on(serving_task);
    let elapsed = started.elapsed();
    assert_eq!(ranked_hits(&actual), ranked_hits(&[result("serving", 1.0)]));
    assert_eq!(
        wrapper.status().sampled,
        1,
        "wall-clock guard must exercise a real admitted shadow task"
    );
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "serving took {elapsed:?} and appears to have awaited the 250ms shadow"
    );
}

#[test]
fn missing_pool_sheds_shadow_without_polling_its_backend() {
    let temp = tempfile::tempdir().expect("tempdir");
    let shadow = Arc::new(StaticLexical::new(vec![result("shadow", 2.0)]));
    let config = enabled_config(temp.path());
    assert!(!config.allow_inline_shadow);
    let wrapper = Arc::new(
        ShadowLexical::new(
            Arc::new(StaticLexical::new(vec![result("serving", 1.0)])),
            shadow.clone(),
            config,
        )
        .expect("wrapper"),
    );
    let runtime = RuntimeBuilder::current_thread().build().expect("runtime");
    let task_wrapper = wrapper.clone();
    let task = runtime.handle().spawn(async move {
        let cx = Cx::current().expect("request context");
        assert!(cx.blocking_pool_handle().is_none());
        task_wrapper.search(&cx, "no pool", 10).await
    });
    let actual = runtime.block_on(task).expect("serving result");
    assert_eq!(ranked_hits(&actual), ranked_hits(&[result("serving", 1.0)]));
    assert_eq!(shadow.search_count.load(Ordering::Acquire), 0);
    let status = wrapper.status();
    assert_eq!(status.sampled, 1);
    assert_eq!(status.shed, 1);
    assert_eq!(status.degradations, 1);
    assert_eq!(status.completed, 0);
    assert_eq!(status.in_flight, 0);
}

#[test]
fn caller_pool_completes_slow_shadow_without_delaying_serving() {
    let temp = tempfile::tempdir().expect("tempdir");
    let wrapper = Arc::new(
        ShadowLexical::new(
            Arc::new(StaticLexical::new(vec![result("serving", 1.0)])),
            Arc::new(SlowShadow {
                delay: std::time::Duration::from_millis(250),
            }),
            enabled_config(temp.path()),
        )
        .expect("wrapper"),
    );
    wrapper.seed_corpus(&[IndexableDocument::new("serving", "pool latency")], 0);
    let runtime = RuntimeBuilder::current_thread()
        .blocking_threads(0, 1)
        .build()
        .expect("runtime");
    let task_wrapper = wrapper.clone();
    let started = Instant::now();
    let task = runtime.handle().spawn(async move {
        let cx = Cx::current().expect("request context");
        task_wrapper.search(&cx, "pool latency", 10).await
    });
    let actual = runtime.block_on(task).expect("serving result");
    let elapsed = started.elapsed();
    assert_eq!(ranked_hits(&actual), ranked_hits(&[result("serving", 1.0)]));
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "serving took {elapsed:?} despite caller-owned blocking capacity"
    );
    // The runtime keeps driving background work after block_on returns.
    // Wait separately for comparison and durable observation completion;
    // this wait is deliberately outside the serving latency measurement.
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while wrapper.status().in_flight != 0 && Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let status = wrapper.status();
    assert_eq!(status.sampled, 1);
    assert_eq!(status.shed, 0);
    assert_eq!(
        status.degradations,
        0,
        "comparison degradations: {:?}",
        std::fs::read_to_string(
            temp.path()
                .join(SHADOW_ARTIFACT_DIRECTORY)
                .join(SHADOW_DEGRADATIONS_FILE)
        )
    );
    assert_eq!(status.completed, 1);
    assert_eq!(status.divergences, 1);
    assert_eq!(status.in_flight, 0);
    let summary = ShadowArtifactSummary::read_index_root(temp.path()).expect("summary");
    assert_eq!(summary.observation_count, 1);
    assert_eq!(summary.divergence_count, 1);
    assert_eq!(summary.degradation_count, 0);
}

#[test]
fn shadow_pool_preserves_wakes_and_polls_off_the_serving_thread() {
    let runtime = RuntimeBuilder::current_thread()
        .blocking_threads(0, 1)
        .build()
        .expect("runtime");
    let serving_thread = std::thread::current().id();
    let polls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&polls);
    let task = runtime.handle().spawn(async move {
        let cx = Cx::current().expect("request context");
        let future = std::future::poll_fn(move |context| {
            assert_ne!(std::thread::current().id(), serving_thread);
            let count = observed.fetch_add(1, Ordering::AcqRel);
            if count < 2 {
                // Wakes issued before Pending, including duplicates, must
                // neither disappear nor trigger unbounded eager polling.
                context.waker().wake_by_ref();
                context.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(Ok(vec![result("shadow", 3.0)]))
            }
        });
        poll_shadow_on_pool(&cx, Box::pin(future), false).await
    });
    let actual = runtime.block_on(task).expect("shadow result");
    assert_eq!(polls.load(Ordering::Acquire), 3);
    assert_eq!(ranked_hits(&actual), ranked_hits(&[result("shadow", 3.0)]));
}

#[test]
fn shadow_pool_does_not_publish_a_result_cancelled_during_its_poll() {
    let runtime = RuntimeBuilder::current_thread()
        .blocking_threads(0, 1)
        .build()
        .expect("runtime");
    let cx = runtime.request_cx_with_budget(Budget::INFINITE);
    let cancel_cx = cx.clone();
    let future = async move {
        cancel_cx.set_cancel_requested(true);
        Ok(vec![result("must-not-publish", 1.0)])
    };
    let error = runtime
        .block_on(poll_shadow_on_pool(&cx, Box::pin(future), false))
        .expect_err("cancelled result");
    assert!(matches!(error, SearchError::Cancelled { .. }));
}

#[test]
fn inline_opt_in_does_not_publish_a_cancelled_shadow_result() {
    let runtime = RuntimeBuilder::current_thread().build().expect("runtime");
    let cx = runtime.request_cx_with_budget(Budget::INFINITE);
    assert!(cx.blocking_pool_handle().is_none());
    let cancel_cx = cx.clone();
    let future = async move {
        cancel_cx.set_cancel_requested(true);
        Ok(vec![result("must-not-publish", 1.0)])
    };
    let error = runtime
        .block_on(poll_shadow_on_pool(&cx, Box::pin(future), true))
        .expect_err("cancelled inline result");
    assert!(matches!(error, SearchError::Cancelled { .. }));
}

#[test]
fn shadow_pool_shutdown_never_polls_the_backend_inline() {
    let runtime = RuntimeBuilder::current_thread().build().expect("runtime");
    let pool = asupersync::runtime::blocking_pool::BlockingPool::new(0, 1);
    let handle = pool.handle();
    assert!(pool.shutdown_and_wait(std::time::Duration::from_secs(2)));
    let polls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&polls);
    let task = runtime.handle().spawn(async move {
        let cx = Cx::current()
            .expect("request context")
            .with_blocking_pool_handle(Some(handle));
        let future = async move {
            observed.fetch_add(1, Ordering::AcqRel);
            Ok(vec![result("must-not-run", 1.0)])
        };
        poll_shadow_on_pool(&cx, Box::pin(future), false).await
    });
    let error = runtime.block_on(task).expect_err("stopped pool");
    assert!(error.to_string().contains("blocking pool is shut down"));
    assert_eq!(polls.load(Ordering::Acquire), 0);
}

#[test]
fn enable_disable_lifecycle_admits_only_enabled_queries() {
    let temp = tempfile::tempdir().expect("tempdir");
    let serving = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
    let shadow = Arc::new(StaticLexical::new(vec![result("a", 1.0)]));
    let wrapper =
        ShadowLexical::new(serving, shadow, enabled_config(temp.path())).expect("wrapper");
    wrapper.set_enabled(false);
    assert!(!wrapper.status().enabled);
    wrapper.set_enabled(true);
    assert!(wrapper.status().enabled);
    wrapper.set_shadow_ready(false);
    let cx = Cx::for_testing();
    wrapper.submit_shadow(
        &cx,
        "disabled-shadow",
        1,
        &[result("a", 1.0)],
        ShadowSearchPath::Search,
        1,
    );
    assert_eq!(wrapper.status().shed, 1);
    assert_eq!(wrapper.status().in_flight, 0);
}

#[test]
fn corpus_hash_is_order_and_hashmap_iteration_independent() {
    let mut left_metadata = HashMap::new();
    left_metadata.insert("z".to_owned(), "last".to_owned());
    left_metadata.insert("a".to_owned(), "first".to_owned());
    let left = IndexableDocument {
        id: "b".to_owned(),
        content: "beta".to_owned(),
        title: None,
        metadata: left_metadata,
    };
    let right = IndexableDocument::new("a", "alpha");
    let ordered = BTreeMap::from([
        (left.id.clone(), ShadowDocument::from(&left)),
        (right.id.clone(), ShadowDocument::from(&right)),
    ]);
    let forward = ordered.values().cloned().collect::<Vec<_>>();
    let reverse_input = [right, left];
    let reverse_map = reverse_input
        .iter()
        .map(|document| (document.id.clone(), ShadowDocument::from(document)))
        .collect::<BTreeMap<_, _>>();
    let reverse = reverse_map.values().cloned().collect::<Vec<_>>();
    assert_eq!(
        compute_shadow_corpus_hash(&forward),
        compute_shadow_corpus_hash(&reverse)
    );
}
