use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll, Waker};

use frankensearch_core::generation::ArtifactGenerationIdentityV1;
use frankensearch_fusion::RrfConfig;

use crate::{ScoreSource, SearchError};

use super::super::{NativePhaseCandidates, NativeSearchPhase, State};
use super::{
    CrossEncoder, Lexical, Provider, Reply, RerankReply, fast_index, generation, native_index,
    quality_index, source_text,
};

#[test]
fn configured_defaults_collect_the_existing_refined_page() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let index = fast_index(&cx, &fast);
        let qindex = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["a-quality", "z-fast"]);
        let expected = index
            .search_hybrid_refined_text(&cx, &fast, (&qindex, &quality), &lexical, "query", 3)
            .await
            .unwrap();
        let actual = index
            .search_hybrid_progressive(&cx, &fast, Some((&qindex, &quality)), &lexical, "query", 3)
            .unwrap()
            .with_fusion(RrfConfig::default(), 0.7)
            .unwrap()
            .with_candidate_multiplier(3)
            .unwrap()
            .with_beam_widths(None, None)
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    });
}

#[test]
fn configured_quality_weight_changes_ranking_not_raw_evidence_or_row_space() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let index = fast_index(&cx, &fast);
        let qindex = quality_index(&cx, &quality);
        let lexical = Lexical::new(&[]);
        for (weight, winner) in [(0.0, "z-fast"), (1.0, "a-quality")] {
            let results = index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&qindex, &quality)),
                    &lexical,
                    "query",
                    4,
                )
                .unwrap()
                .with_fusion(RrfConfig::default(), weight)
                .unwrap()
                .with_candidate_multiplier(1)
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(results[0].doc_id, winner);
            assert_eq!(results.len(), 4);
            let a = results
                .iter()
                .find(|hit| hit.doc_id == "a-quality")
                .unwrap();
            assert_eq!(a.fast_score, Some(0.0));
            assert_eq!(a.quality_score, Some(1.0));
            assert_eq!(a.source, ScoreSource::SemanticQuality);
            // Zero is a real fast score, so even at quality weight 1 the
            // index belongs to the contributing fast owner, not to a rank.
            assert_eq!(a.index, Some(2));
            assert_eq!(index.owner.doc_id_at(2).unwrap(), a.doc_id);
            let z = results.iter().find(|hit| hit.doc_id == "z-fast").unwrap();
            assert_eq!(z.fast_score, Some(1.0));
            assert_eq!(z.quality_score, Some(-1.0));
        }
        assert_eq!(fast.calls.load(Ordering::SeqCst), 2);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn configured_rrf_weights_and_k_are_applied_to_both_phases() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let index = fast_index(&cx, &fast);
        let qindex = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["lexical-only"]);
        for (k, expected_score) in [(0.0, 2.0_f32), (60.0, 2.0_f32 / 61.0)] {
            for prefer_lexical in [false, true] {
                let config = RrfConfig {
                    k,
                    lexical_weight: if prefer_lexical { 2.0 } else { 1.0 },
                    semantic_weight: if prefer_lexical { 1.0 } else { 2.0 },
                    ..RrfConfig::default()
                };
                let mut stream = index
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&qindex, &quality)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap()
                    .with_fusion(config, 0.7)
                    .unwrap();
                let NativeSearchPhase::Initial { results, .. } =
                    stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial")
                };
                assert_eq!(
                    results[0].doc_id,
                    if prefer_lexical {
                        "lexical-only"
                    } else {
                        "z-fast"
                    }
                );
                assert_eq!(results[0].score.to_bits(), expected_score.to_bits());
                let NativeSearchPhase::Refined { results, .. } =
                    stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("refined")
                };
                assert_eq!(
                    results[0].doc_id,
                    if prefer_lexical {
                        "lexical-only"
                    } else {
                        "a-quality"
                    }
                );
                assert_eq!(results[0].score.to_bits(), expected_score.to_bits());
                if prefer_lexical {
                    assert_eq!(results[0].source, ScoreSource::Lexical);
                    assert!(results[0].index.is_none());
                    assert_eq!(results[0].metadata.as_deref().unwrap()["generation"], 1);
                } else {
                    assert_eq!(results[0].quality_score, Some(1.0));
                    assert!(results[0].fast_score.is_none());
                    assert_eq!(
                        qindex
                            .owner
                            .doc_id_at(results[0].index.unwrap() as usize)
                            .unwrap(),
                        results[0].doc_id
                    );
                }
            }
        }
    });
}

#[test]
fn candidate_multiplier_is_effective_and_independent_of_reranker_configuration_order() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let index = fast_index(&cx, &fast);
        let qindex = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["z-fast", "middle", "near", "a-quality"]);
        for multiplier in [1, 2, 4] {
            let mut stream = index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&qindex, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap()
                .with_candidate_multiplier(multiplier)
                .unwrap();
            let NativeSearchPhase::Initial {
                candidates,
                results,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            assert_eq!(results.len(), 1);
            assert_eq!(
                candidates,
                NativePhaseCandidates {
                    fast: multiplier,
                    quality: 0,
                    lexical: multiplier
                }
            );
            let NativeSearchPhase::Refined { candidates, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("refined")
            };
            assert_eq!(
                candidates,
                NativePhaseCandidates {
                    fast: multiplier,
                    quality: multiplier,
                    lexical: multiplier
                }
            );
        }
        let model = CrossEncoder::new(RerankReply::Winner);
        let mut pages = Vec::new();
        for reranker_first in [false, true] {
            let stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap();
            let mut stream = if reranker_first {
                stream
                    .with_reranker(&model, &source_text, 3)
                    .unwrap()
                    .with_candidate_multiplier(1)
                    .unwrap()
            } else {
                stream
                    .with_candidate_multiplier(1)
                    .unwrap()
                    .with_reranker(&model, &source_text, 3)
                    .unwrap()
            };
            let NativeSearchPhase::Initial { candidates, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            assert_eq!(candidates.fast, 3);
            assert_eq!(candidates.lexical, 3);
            let NativeSearchPhase::Reranked {
                results, evaluated, ..
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("reranked")
            };
            assert_eq!(evaluated, 3);
            assert_eq!(results[0].doc_id, "near");
            pages.push(serde_json::to_value(results).unwrap());
        }
        assert_eq!(pages[0], pages[1]);
    });
}

#[test]
fn invalid_or_overflowing_query_policy_is_not_sanitized_into_default_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let index = fast_index(&cx, &fast);
        let lexical = Lexical::new(&[]);
        let fresh = |k| {
            index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", k)
                .unwrap()
        };
        for k in [0, 2] {
            for weight in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
                assert!(fresh(k).with_fusion(RrfConfig::default(), weight).is_err());
            }
            for value in [f64::NAN, f64::INFINITY, -1.0] {
                assert!(
                    fresh(k)
                        .with_fusion(
                            RrfConfig {
                                k: value,
                                ..RrfConfig::default()
                            },
                            0.7
                        )
                        .is_err()
                );
            }
            for value in [f64::NAN, f64::INFINITY, -1.0, 0.0, f64::MAX] {
                assert!(
                    fresh(k)
                        .with_fusion(
                            RrfConfig {
                                lexical_weight: value,
                                ..RrfConfig::default()
                            },
                            0.7
                        )
                        .is_err()
                );
                assert!(
                    fresh(k)
                        .with_fusion(
                            RrfConfig {
                                semantic_weight: value,
                                ..RrfConfig::default()
                            },
                            0.7
                        )
                        .is_err()
                );
            }
            assert!(fresh(k).with_candidate_multiplier(0).is_err());
            assert!(fresh(k).with_beam_widths(Some(0), None).is_err());
            assert!(fresh(k).with_beam_widths(None, Some(0)).is_err());
        }
        assert!(fresh(2).with_candidate_multiplier(usize::MAX).is_err());
        let model = CrossEncoder::new(RerankReply::Unavailable);
        assert!(
            fresh(1)
                .with_candidate_multiplier(2)
                .unwrap()
                .with_reranker(&model, &source_text, usize::MAX)
                .is_err()
        );
        assert!(
            fresh(1)
                .with_reranker(&model, &source_text, 2)
                .unwrap()
                .with_candidate_multiplier(usize::MAX)
                .is_err()
        );
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        assert_eq!(model.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn tuning_cannot_change_a_started_query_or_disable_required_quality() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let mut quality = Provider::new("quality", 3);
        quality.reply = Reply::Failed;
        let index = fast_index(&cx, &fast);
        let qindex = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["z-fast"]);
        for setting in 0..4 {
            let mut stream = index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&qindex, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            match setting {
                0 => assert!(stream.with_fusion(RrfConfig::default(), 0.0).is_err()),
                1 => assert!(stream.with_candidate_multiplier(1).is_err()),
                2 => assert!(stream.with_beam_widths(Some(1), None).is_err()),
                _ => assert!(stream.collect().await.is_err()),
            }
        }
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        let error = index
            .search_hybrid_progressive(&cx, &fast, Some((&qindex, &quality)), &lexical, "query", 1)
            .unwrap()
            .with_fusion(RrfConfig::default(), 0.0)
            .unwrap()
            .collect()
            .await
            .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "native_ann.test.provider")
        );
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
    });
}

#[test]
fn beam_overrides_reach_each_tier_before_retained_owner_rescoring() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Provider::new("fast", 2);
        let lexical = Lexical::new(&[]);
        let mut changed_fast = false;
        let mut changed_quality = false;
        for reverse in [false, true] {
            // Both graph distances round to 1.0, but their retained f32
            // dot products differ. A full beam must see the better row.
            let low = [1.0e-8, 1.0];
            let high = [2.0e-8, 1.0];
            let rows: [(&str, &[f32]); 2] = if reverse {
                [("a", &high), ("b", &low)]
            } else {
                [("a", &low), ("b", &high)]
            };
            let winner = if reverse { "a" } else { "b" };
            let mut index = native_index(&cx, &provider, generation(), &rows);
            index.default_ef_search = 1;
            let narrow = index
                .search_hybrid_progressive(&cx, &provider, None, &lexical, "query", 1)
                .unwrap()
                .with_candidate_multiplier(1)
                .unwrap()
                .collect()
                .await
                .unwrap();
            let wide = index
                .search_hybrid_progressive(&cx, &provider, None, &lexical, "query", 1)
                .unwrap()
                .with_candidate_multiplier(1)
                .unwrap()
                .with_beam_widths(Some(2), None)
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(wide[0].doc_id, winner);
            assert_eq!(wide[0].fast_score, Some(high[0]));
            changed_fast |= narrow[0].doc_id != wide[0].doc_id;
            let empty = native_index(&cx, &provider, generation(), &[]);
            let narrow = empty
                .search_hybrid_progressive(
                    &cx,
                    &provider,
                    Some((&index, &provider)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap()
                .with_candidate_multiplier(1)
                .unwrap()
                .collect()
                .await
                .unwrap();
            let wide = empty
                .search_hybrid_progressive(
                    &cx,
                    &provider,
                    Some((&index, &provider)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap()
                .with_candidate_multiplier(1)
                .unwrap()
                .with_beam_widths(None, Some(2))
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(wide[0].doc_id, winner);
            assert_eq!(wide[0].quality_score, Some(high[0]));
            assert!(wide[0].fast_score.is_none());
            assert_eq!(
                index
                    .owner
                    .doc_id_at(wide[0].index.unwrap() as usize)
                    .unwrap(),
                winner
            );
            changed_quality |= narrow[0].doc_id != wide[0].doc_id;
        }
        assert!(
            changed_fast && changed_quality,
            "the overrides must change actual retrieval, not only stored configuration"
        );
    });
}

#[test]
fn lazy_refinement_retrieves_a_winner_outside_the_fast_candidate_pool() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (padding, quality_row) in [("middle", 2_u32), ("quality-padding", 1)] {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            // FSVI persists hash-sorted rows, not insertion positions. Use a
            // different quality cohort so the winning document really has
            // distinct physical positions in the two retained owners.
            let quality_index = native_index(
                &cx,
                &quality,
                generation(),
                &[
                    ("a-quality", &[1.0, 0.0, 0.0]),
                    ("near", &[0.0, 1.0, 0.0]),
                    (padding, &[0.0, 0.0, 1.0]),
                    ("z-fast", &[-1.0, 0.0, 0.0]),
                ],
            );
            assert_eq!(fast_index.owner.doc_id_at(2).unwrap(), "a-quality");
            assert_eq!(
                quality_index.owner.doc_id_at(quality_row as usize).unwrap(),
                "a-quality"
            );
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            let NativeSearchPhase::Initial {
                results,
                candidates,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("expected initial phase");
            };
            assert_eq!(results[0].doc_id, "z-fast");
            assert_eq!(
                candidates,
                NativePhaseCandidates {
                    fast: 3,
                    quality: 0,
                    lexical: 0
                }
            );
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            let State::Refine(pending) = &stream.state else {
                panic!("missing retained pool")
            };
            assert!(pending.fast.iter().all(|hit| hit.doc_id != "a-quality"));
            let NativeSearchPhase::Refined {
                results,
                candidates,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("expected independent refinement");
            };
            assert_eq!(results[0].doc_id, "a-quality");
            assert_eq!(results[0].quality_score, Some(1.0));
            assert!(results[0].fast_score.is_none());
            assert_eq!(results[0].source, ScoreSource::SemanticQuality);
            assert_eq!(results[0].index, Some(quality_row));
            let row = results[0].index.unwrap() as usize;
            assert_eq!(
                quality_index.owner.doc_id_at(row).unwrap(),
                results[0].doc_id
            );
            assert_eq!(
                quality_index.owner.vector_at_f32(row).unwrap(),
                vec![1.0, 0.0, 0.0]
            );
            assert_eq!(
                candidates,
                NativePhaseCandidates {
                    fast: 3,
                    quality: 3,
                    lexical: 0
                }
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
        }
    });
}

#[test]
fn stopping_after_initial_starts_no_quality_work_and_releases_the_pin() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let mut quality = Provider::new("quality", 3);
        quality.reply = Reply::Pending;
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["z-fast"]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                1,
            )
            .unwrap();
        assert!(matches!(
            stream.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Initial { .. })
        ));
        assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
        drop(stream);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        assert_eq!(quality.drops.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn refinement_failure_carries_the_unchanged_initial_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let mut quality = Provider::new("quality", 3);
        quality.reply = Reply::Failed;
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["z-fast"]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                1,
            )
            .unwrap();
        let NativeSearchPhase::Initial { results, .. } =
            stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("expected initial");
        };
        let expected = serde_json::to_value(results).unwrap();
        let NativeSearchPhase::RefinementFailed {
            initial_results,
            error,
        } = stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("provider failure must be explicit");
        };
        assert_eq!(serde_json::to_value(initial_results).unwrap(), expected);
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "native_ann.test.provider")
        );
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn provider_cancellation_is_terminal_not_successful_degradation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let mut quality = Provider::new("quality", 3);
        quality.reply = Reply::Cancelled;
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&[]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                1,
            )
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        assert!(
            matches!(stream.next_phase().await, Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.provider")
        );
        assert!(stream.is_finished());
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
    });
}

#[test]
fn dropping_a_polled_refinement_future_fuses_the_stream_and_drops_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let mut quality = Provider::new("quality", 3);
        quality.reply = Reply::Pending;
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["z-fast"]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                1,
            )
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        let mut future = Box::pin(stream.next_phase());
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
        drop(future);
        assert!(stream.is_finished());
        assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn fast_only_and_zero_k_have_a_single_initial_phase() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let index = fast_index(&cx, &fast);
        let lexical = Lexical::new(&[]);
        let mut empty = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 0)
            .unwrap();
        let NativeSearchPhase::Initial {
            results,
            candidates,
        } = empty.next_phase().await.unwrap().unwrap()
        else {
            panic!("zero k yields an empty initial phase");
        };
        assert!(results.is_empty());
        assert_eq!(candidates, NativePhaseCandidates::default());
        assert!(empty.next_phase().await.unwrap().is_none());
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
            .unwrap();
        let NativeSearchPhase::Initial { results, .. } =
            stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("fast-only phase");
        };
        assert_eq!(results[0].doc_id, "z-fast");
        assert!(stream.is_finished());
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
    });
}

#[test]
fn refinement_reuses_the_scoring_snapshot_and_matches_the_existing_refined_api() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["a-quality", "z-fast"]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                2,
            )
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        lexical.current_generation.store(42, Ordering::SeqCst);
        let NativeSearchPhase::Refined { results, .. } =
            stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("expected refined");
        };
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
        assert_eq!(lexical.current_generation.load(Ordering::SeqCst), 42);
        assert!(
            results
                .iter()
                .all(|hit| hit.metadata.as_deref().unwrap()["generation"] == 1)
        );
        let eager = fast_index
            .search_hybrid_refined_text(
                &cx,
                &fast,
                (&quality_index, &quality),
                &lexical,
                "query",
                2,
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(results).unwrap(),
            serde_json::to_value(eager).unwrap()
        );
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
    });
}

#[test]
fn changed_quality_identity_is_rejected_before_refinement_inference() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&[]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                1,
            )
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        quality.advertise_foreign.store(true, Ordering::SeqCst);
        let NativeSearchPhase::RefinementFailed {
            initial_results,
            error,
        } = stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("changed producer must not be admitted");
        };
        assert_eq!(initial_results[0].doc_id, "z-fast");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "query_embedding.native_ann.producer_conformance")
        );
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        assert!(stream.is_finished());
    });
}

#[test]
fn dropping_a_polled_initial_future_does_not_restart_either_source() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut fast = Provider::new("fast", 2);
        fast.reply = Reply::Pending;
        let index = fast_index(&cx, &fast);
        let mut lexical = Lexical::new(&[]);
        lexical.pending_search = true;
        let mut stream = index
            .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
            .unwrap();
        let mut future = Box::pin(stream.next_phase());
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
        drop(future);
        assert_eq!(fast.drops.load(Ordering::SeqCst), 1);
        assert!(stream.is_finished());
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn pair_admission_and_budget_overflow_refuse_without_any_provider_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let fast_index = fast_index(&cx, &fast);
        let foreign_index = native_index(
            &cx,
            &quality,
            ArtifactGenerationIdentityV1::new(1, [0x62; 16]).unwrap(),
            &[],
        );
        let lexical = Lexical::new(&[]);
        for k in [0, 1] {
            assert!(matches!(
                fast_index.search_hybrid_progressive(&cx, &fast, Some((&foreign_index, &quality)), &lexical, "query", k),
                Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.tier_generation"
            ));
        }
        assert!(matches!(
            fast_index.search_hybrid_progressive(&cx, &fast, None, &lexical, "query", usize::MAX),
            Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.progressive.candidate_budget"
        ));
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn dropping_pending_refinement_hydration_releases_the_original_pin() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let mut lexical = Lexical::new(&["a-quality", "z-fast"]);
        lexical.refinement_hydration = Reply::Pending;
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                2,
            )
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        let mut future = Box::pin(stream.next_phase());
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        assert_eq!(lexical.hydration_calls.load(Ordering::SeqCst), 2);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
        drop(future);
        assert_eq!(lexical.hydration_drops.load(Ordering::SeqCst), 2);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        assert!(stream.is_finished());
        assert!(stream.next_phase().await.unwrap().is_none());
    });
}

#[test]
fn a_substituted_actual_quality_response_cannot_produce_a_refined_phase() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let mut quality = Provider::new("quality", 3);
        quality.reply = Reply::ForeignProducer;
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&[]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                1,
            )
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        let NativeSearchPhase::RefinementFailed {
            initial_results,
            error,
        } = stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("substituted producer must fail");
        };
        assert_eq!(initial_results[0].doc_id, "z-fast");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "query_embedding.native_ann.producer_conformance")
        );
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        assert!(stream.next_phase().await.unwrap().is_none());
    });
}

#[test]
fn cancellation_after_partial_refinement_hydration_is_not_partial_success() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let mut lexical = Lexical::new(&["a-quality", "z-fast"]);
        lexical.refinement_hydration = Reply::Cancelled;
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                2,
            )
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        assert!(
            matches!(stream.next_phase().await, Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.hydration")
        );
        assert_eq!(lexical.hydration_calls.load(Ordering::SeqCst), 2);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        assert!(stream.next_phase().await.unwrap().is_none());
    });
}

#[test]
fn empty_fast_owner_does_not_prevent_direct_quality_retrieval() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut fast = Provider::new("fast", 2);
        fast.reply = Reply::Pending;
        let quality = Provider::new("quality", 3);
        let fast_index = native_index(&cx, &fast, generation(), &[]);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&[]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                1,
            )
            .unwrap();
        let NativeSearchPhase::Initial {
            results,
            candidates,
        } = stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("initial phase");
        };
        assert!(results.is_empty());
        assert_eq!(candidates.fast, 0);
        let NativeSearchPhase::Refined { results, .. } =
            stream.next_phase().await.unwrap().unwrap()
        else {
            panic!("quality must retrieve independently");
        };
        assert_eq!(results[0].doc_id, "a-quality");
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn dropping_an_unpolled_phase_does_not_consume_its_work_or_snapshot() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2);
        let quality = Provider::new("quality", 3);
        let fast_index = fast_index(&cx, &fast);
        let quality_index = quality_index(&cx, &quality);
        let lexical = Lexical::new(&["z-fast"]);
        let mut stream = fast_index
            .search_hybrid_progressive(
                &cx,
                &fast,
                Some((&quality_index, &quality)),
                &lexical,
                "query",
                1,
            )
            .unwrap();
        drop(stream.next_phase());
        assert!(!stream.is_finished());
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        assert!(matches!(
            stream.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Initial { .. })
        ));
        drop(stream.next_phase());
        assert!(!stream.is_finished());
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
        assert!(matches!(
            stream.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Refined { .. })
        ));
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
    });
}
