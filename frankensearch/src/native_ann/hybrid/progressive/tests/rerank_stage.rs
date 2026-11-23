use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use crate::{ScoreSource, SearchError};

use super::super::{NativeSearchPhase, State};
use super::{
    CrossEncoder, Lexical, Provider, Reply, RerankReply, fast_index, quality_index, source_text,
};

#[test]
fn rerank_is_lazy_and_can_promote_a_candidate_outside_the_displayed_page() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let index = fast_index(&cx, &fast);
        let lexical = Lexical::new(&["z-fast", "middle", "near"]);
        let model = CrossEncoder::new(RerankReply::Winner);
        let text_calls = AtomicUsize::new(0);
        let text = |id: &str| {
            text_calls.fetch_add(1, Ordering::SeqCst);
            source_text(id)
        };
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
            .unwrap()
            .with_reranker(&model, &text, 3)
            .unwrap();
        let NativeSearchPhase::Initial { results, .. } =
            stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("initial")
        };
        assert_eq!(results[0].doc_id, "z-fast");
        assert_eq!(model.calls.load(Ordering::SeqCst), 0);
        assert_eq!(text_calls.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
        let State::Rerank(pending) = &stream.state else {
            panic!("retained rerank pool")
        };
        let original = pending
            .results
            .iter()
            .find(|hit| hit.doc_id == "near")
            .unwrap()
            .clone();
        let NativeSearchPhase::Reranked {
            results, evaluated, ..
        } = stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("reranked")
        };
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "near");
        assert_eq!(results[0].rerank_score, Some(0.9));
        assert_eq!(results[0].source, ScoreSource::Reranked);
        assert_eq!(results[0].score.to_bits(), original.score.to_bits());
        assert_eq!(results[0].fast_score, original.fast_score);
        assert_eq!(results[0].index, original.index);
        assert!(Arc::ptr_eq(
            results[0].metadata.as_ref().unwrap(),
            original.metadata.as_ref().unwrap()
        ));
        assert_eq!(evaluated, 3);
        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
        assert_eq!(text_calls.load(Ordering::SeqCst), 3);
        assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        assert!(stream.next_phase().await.unwrap().is_none());
    });
}

#[test]
fn rerank_follows_refinement_without_requerying_or_changing_the_snapshot() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let index = fast_index(&cx, &fast);
        let qindex = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["z-fast", "middle", "near", "a-quality"]);
        let model = CrossEncoder::new(RerankReply::Winner);
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, Some((&qindex, &quality)), &lexical, "query", 1)
            .unwrap()
            .with_reranker(&model, &source_text, 4)
            .unwrap();
        assert!(matches!(
            stream.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Initial { .. })
        ));
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        assert!(matches!(
            stream.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Refined { .. })
        ));
        assert_eq!(model.calls.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
        lexical.current_generation.store(42, Ordering::SeqCst);
        let NativeSearchPhase::Reranked {
            results, evaluated, ..
        } = stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("rerank after refinement")
        };
        assert_eq!(results[0].doc_id, "near");
        assert!(results[0].fast_score.is_some() && results[0].quality_score.is_some());
        assert_eq!(results[0].metadata.as_deref().unwrap()["generation"], 1);
        assert_eq!(evaluated, 4);
        assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
    });
}

#[test]
fn failed_quality_still_allows_explicit_reranking_of_the_valid_initial_pool() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let mut quality = Provider::new("quality", 3);
        quality.reply = Reply::Failed;
        let index = fast_index(&cx, &fast);
        let qindex = quality_index(&cx, &quality);
        let lexical = Lexical::new(&[]);
        let model = CrossEncoder::new(RerankReply::Winner);
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, Some((&qindex, &quality)), &lexical, "query", 1)
            .unwrap()
            .with_reranker(&model, &source_text, 3)
            .unwrap();
        let NativeSearchPhase::Initial { results, .. } =
            stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("initial")
        };
        let expected = serde_json::to_value(results).unwrap();
        let NativeSearchPhase::RefinementFailed {
            initial_results, ..
        } = stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("failure must be exposed")
        };
        assert_eq!(serde_json::to_value(initial_results).unwrap(), expected);
        let NativeSearchPhase::Reranked { results, .. } =
            stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("independent final stage")
        };
        assert_eq!(results[0].doc_id, "near");
        assert!(results[0].quality_score.is_none());
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn malformed_rerank_responses_never_publish_partial_or_cross_document_scores() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let index = fast_index(&cx, &fast);
        for reply in [
            RerankReply::DuplicateRank,
            RerankReply::ForeignId,
            RerankReply::OutOfRange,
            RerankReply::MissingScore,
            RerankReply::NonFinite,
            RerankReply::NonFiniteLogit,
            RerankReply::Failed,
        ] {
            let lexical = Lexical::new(&["z-fast", "middle", "near"]);
            let model = CrossEncoder::new(reply);
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 2)
                .unwrap()
                .with_reranker(&model, &source_text, 3)
                .unwrap();
            let NativeSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            let metadata = Arc::clone(results[0].metadata.as_ref().unwrap());
            let expected = serde_json::to_value(results).unwrap();
            let NativeSearchPhase::RerankFailed {
                previous_results, ..
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("must fail atomically")
            };
            assert!(Arc::ptr_eq(
                previous_results[0].metadata.as_ref().unwrap(),
                &metadata
            ));
            assert_eq!(serde_json::to_value(previous_results).unwrap(), expected);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
        }
    });
}

#[test]
fn pending_rerank_drop_releases_provider_and_snapshot_without_retrying() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let index = fast_index(&cx, &fast);
        let lexical = Lexical::new(&["z-fast"]);
        let model = CrossEncoder::new(RerankReply::Pending);
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
            .unwrap()
            .with_reranker(&model, &source_text, 3)
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        drop(stream.next_phase()); // An unpolled final future is still inert.
        assert_eq!(model.calls.load(Ordering::SeqCst), 0);
        assert!(!stream.is_finished());
        let mut future = Box::pin(stream.next_phase());
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
        drop(future);
        assert_eq!(model.drops.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        assert!(stream.is_finished());
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn reranker_cancellation_is_not_a_successful_fallback_phase() {
    for reply in [RerankReply::Cancelled, RerankReply::CancelWithScores] {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            let lexical = Lexical::new(&["z-fast"]);
            let model = CrossEncoder::new(reply);
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap()
                .with_reranker(&model, &source_text, 3)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(matches!(
                stream.next_phase().await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(stream.is_finished());
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }
}

#[test]
fn missing_text_is_skipped_without_false_scores_or_a_false_reranked_phase() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let index = fast_index(&cx, &fast);
        let lexical = Lexical::new(&[]);
        let model = CrossEncoder::new(RerankReply::Winner);
        let only_near = |id: &str| (id == "near").then(|| format!("version-one:{id}"));
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 4)
            .unwrap()
            .with_reranker(&model, &only_near, 3)
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        let NativeSearchPhase::Reranked {
            results, evaluated, ..
        } = stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("one scored document")
        };
        assert_eq!(evaluated, 1);
        assert_eq!(
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["near", "z-fast", "middle", "a-quality"]
        );
        assert!(
            results[1..]
                .iter()
                .all(|hit| hit.rerank_score.is_none() && hit.source != ScoreSource::Reranked)
        );
        let no_text = |_: &str| None;
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
            .unwrap()
            .with_reranker(&model, &no_text, 3)
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn rerank_configuration_and_zero_result_requests_start_no_model_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let index = fast_index(&cx, &fast);
        let lexical = Lexical::new(&[]);
        let model = CrossEncoder::new(RerankReply::Unavailable);
        for budget in [0, usize::MAX] {
            assert!(
                index
                    .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                    .unwrap()
                    .with_reranker(&model, &source_text, budget)
                    .is_err()
            );
        }
        let mut empty = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 0)
            .unwrap()
            .with_reranker(&model, &source_text, 3)
            .unwrap();
        assert!(matches!(
            empty.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Initial { .. })
        ));
        assert!(empty.next_phase().await.unwrap().is_none());
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        assert!(stream.with_reranker(&model, &source_text, 3).is_err());
        let mut unavailable = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
            .unwrap()
            .with_reranker(&model, &source_text, 3)
            .unwrap();
        assert!(unavailable.next_phase().await.unwrap().is_some());
        assert!(matches!(
            unavailable.next_phase().await.unwrap(),
            Some(NativeSearchPhase::RerankFailed {
                error: SearchError::RerankerUnavailable { .. },
                ..
            })
        ));
        assert_eq!(model.calls.load(Ordering::SeqCst), 0);
    });
}
