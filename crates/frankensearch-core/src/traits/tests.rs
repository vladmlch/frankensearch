use asupersync::test_utils::run_test_with_cx;

use super::*;
use crate::error::{SearchError, SearchResult};
use crate::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use crate::types::{EmbeddingMetrics, IndexMetrics, ScoredResult, SearchMetrics};

struct BoundSyncEmbedder {
    identity: EmbeddingIdentityBundleV1,
    output_dimension: usize,
}

impl SyncEmbed for BoundSyncEmbedder {
    fn embed_sync(&self, _text: &str) -> SearchResult<Vec<f32>> {
        Ok(vec![1.0; self.output_dimension])
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.output_dimension
    }

    fn id(&self) -> &'static str {
        "bound-sync-fixture"
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

struct UnsortedSyncReranker;

impl SyncRerank for UnsortedSyncReranker {
    fn rerank_sync(
        &self,
        _query: &str,
        _documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        Ok(vec![
            RerankScore {
                doc_id: "doc-a".into(),
                score: 0.8,
                original_rank: 2,
                raw_logit: None,
            },
            RerankScore {
                doc_id: "doc-b".into(),
                score: 0.8,
                original_rank: 1,
                raw_logit: None,
            },
            RerankScore {
                doc_id: "doc-c".into(),
                score: 0.3,
                original_rank: 0,
                raw_logit: None,
            },
        ])
    }

    fn id(&self) -> &'static str {
        "unsorted-sync-reranker"
    }

    fn model_name(&self) -> &'static str {
        "Unsorted Sync Reranker"
    }
}

struct UnboundSyncEmbedder;

impl SyncEmbed for UnboundSyncEmbedder {
    fn embed_sync(&self, _text: &str) -> SearchResult<Vec<f32>> {
        Ok(vec![0.0])
    }

    fn dimension(&self) -> usize {
        1
    }

    fn id(&self) -> &'static str {
        "legacy\nforged-log-line"
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

#[test]
fn model_category_display() {
    assert_eq!(ModelCategory::HashEmbedder.to_string(), "hash_embedder");
    assert_eq!(ModelCategory::StaticEmbedder.to_string(), "static_embedder");
    assert_eq!(
        ModelCategory::TransformerEmbedder.to_string(),
        "transformer_embedder"
    );
}

#[test]
fn model_category_serialization() {
    let json = serde_json::to_string(&ModelCategory::StaticEmbedder).unwrap();
    let decoded: ModelCategory = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, ModelCategory::StaticEmbedder);
}

#[test]
fn model_category_equality() {
    assert_eq!(ModelCategory::HashEmbedder, ModelCategory::HashEmbedder);
    assert_ne!(ModelCategory::HashEmbedder, ModelCategory::StaticEmbedder);
    assert_ne!(
        ModelCategory::StaticEmbedder,
        ModelCategory::TransformerEmbedder
    );
}

#[test]
fn model_category_default_tier() {
    assert_eq!(ModelCategory::HashEmbedder.default_tier(), ModelTier::Fast);
    assert_eq!(
        ModelCategory::StaticEmbedder.default_tier(),
        ModelTier::Fast
    );
    assert_eq!(
        ModelCategory::TransformerEmbedder.default_tier(),
        ModelTier::Quality
    );
}

#[test]
fn model_tier_display() {
    assert_eq!(ModelTier::Fast.to_string(), "fast");
    assert_eq!(ModelTier::Quality.to_string(), "quality");
}

#[test]
fn model_info_roundtrip() {
    let info = ModelInfo {
        id: "potion-multilingual-128M".to_owned(),
        name: "Potion 128M".to_owned(),
        dimension: 256,
        category: ModelCategory::StaticEmbedder,
        tier: ModelTier::Fast,
        is_semantic: true,
        supports_mrl: false,
        huggingface_id: Some("minishlab/potion-multilingual-128M".to_owned()),
        size_bytes: Some(128_000_000),
        license: Some("apache-2.0".to_owned()),
    };

    let json = serde_json::to_string(&info).unwrap();
    let decoded: ModelInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, info);
}

#[test]
fn rerank_document_construction() {
    let doc = RerankDocument {
        doc_id: "doc-1".into(),
        text: "Some content".into(),
    };
    assert_eq!(doc.doc_id, "doc-1");
    assert_eq!(doc.text, "Some content");
}

#[test]
fn rerank_score_serialization() {
    let score = RerankScore {
        doc_id: "doc-1".into(),
        score: 0.92,
        original_rank: 3,
        raw_logit: None,
    };

    let json = serde_json::to_string(&score).unwrap();
    let decoded: RerankScore = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.doc_id, "doc-1");
    assert!((decoded.score - 0.92).abs() < 1e-6);
    assert_eq!(decoded.original_rank, 3);
}

// Compile-time checks for trait object safety
#[test]
fn embedder_trait_is_object_safe() {
    fn _takes_dyn_embedder(_: &dyn Embedder) {}
}

#[test]
fn sync_bound_outputs_carry_identity_and_fail_on_shape_drift() {
    let embedder = BoundSyncEmbedder {
        identity: EmbeddingIdentityBundleV1::explicit_test_model("bound-sync-fixture", 3),
        output_dimension: 3,
    };
    let bound = embedder.embed_bound_sync("text").unwrap();
    assert_eq!(bound.values, vec![1.0; 3]);
    assert_eq!(bound.identity, embedder.identity);
    assert_eq!(
        embedder.embed_batch_bound_sync(&["a", "b"]).unwrap().len(),
        2
    );

    let drifted = BoundSyncEmbedder {
        identity: EmbeddingIdentityBundleV1::explicit_test_model("bound-sync-fixture", 2),
        output_dimension: 3,
    };
    assert!(drifted.embed_bound_sync("text").is_err());
}

#[test]
fn missing_identity_diagnostic_redacts_untrusted_embedder_id() {
    let error = UnboundSyncEmbedder.identity().unwrap_err();
    assert!(error.to_string().contains("<redacted-embedder-id>"));
    assert!(!error.to_string().contains("forged-log-line"));
}

#[test]
fn identity_bound_debug_redacts_vector_values() {
    let bound = IdentityBoundEmbedding {
        values: vec![12_345.5, -9_876.25],
        identity: EmbeddingIdentityBundleV1::explicit_test_model("debug-redaction", 2),
    };
    let debug = format!("{bound:?}");
    assert!(debug.contains("dimension"));
    assert!(debug.contains(&bound.identity.fingerprint()));
    assert!(!debug.contains("12345"));
    assert!(!debug.contains("9876"));
}

#[test]
fn identity_bound_output_rejects_non_memory_f32_storage_claims() {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("bound-storage-fixture", 2);
    identity.storage.quantization = QuantizationFormat::F16;
    let bound = IdentityBoundEmbedding {
        values: vec![1.0, 2.0],
        identity,
    };
    assert!(bound.validate().is_err());

    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("bound-storage-fixture", 2);
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.endianness = "little-endian".to_owned();
    let bound = IdentityBoundEmbedding {
        values: vec![1.0, 2.0],
        identity,
    };
    assert!(bound.validate().is_err());

    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("bound-storage-fixture", 2);
    identity.storage.endianness = "little-endian".to_owned();
    let bound = IdentityBoundEmbedding {
        values: vec![1.0, 2.0],
        identity,
    };
    assert!(bound.validate().is_err());
}

#[test]
fn sync_embed_adapter_observes_cancel_before_blocking_work() {
    run_test_with_cx(|cx| async move {
        cx.cancel_fast(asupersync::CancelKind::User);
        let adapter = SyncEmbedderAdapter(BoundSyncEmbedder {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("cancel-sync-fixture", 3),
            output_dimension: 3,
        });
        let error = adapter
            .embed(&cx, "text")
            .await
            .expect_err("cancelled sync adapter must fail closed");
        match error {
            SearchError::Cancelled { phase, .. } => {
                assert_eq!(phase, "sync_embed.embed");
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    });
}

#[test]
fn async_bound_outputs_carry_forwarded_identity() {
    run_test_with_cx(|cx| async move {
        let identity = EmbeddingIdentityBundleV1::explicit_test_model("bound-async-fixture", 3);
        let adapter = SyncEmbedderAdapter(BoundSyncEmbedder {
            identity: identity.clone(),
            output_dimension: 3,
        });
        let bound = adapter.embed_bound(&cx, "text").await.unwrap();
        assert_eq!(bound.values, vec![1.0; 3]);
        assert_eq!(bound.identity, identity);
        assert_eq!(
            adapter
                .embed_batch_bound(&cx, &["a", "b"])
                .await
                .unwrap()
                .len(),
            2
        );
    });
}

#[test]
fn reranker_trait_is_object_safe() {
    fn _takes_dyn_reranker(_: &dyn Reranker) {}
}

#[test]
fn split_lexical_traits_are_object_safe() {
    // Both halves must stay object-safe: `ShadowLexical` holds them as
    // `Arc<dyn _>`, and a read-only consumer must be able to take
    // `&dyn LexicalRead` without dragging in a mutation surface.
    fn _takes_dyn_read(_: &dyn LexicalRead) {}
    fn _takes_dyn_write(_: &dyn LexicalWrite) {}
}

#[test]
fn metrics_exporter_trait_is_object_safe() {
    fn _takes_dyn_metrics_exporter(_: &dyn MetricsExporter) {}
}

#[test]
fn sync_reranker_adapter_sorts_descending_for_trait_contract() {
    run_test_with_cx(|cx| async move {
        let adapter = SyncRerankerAdapter(UnsortedSyncReranker);
        let docs = vec![
            RerankDocument {
                doc_id: "doc-a".into(),
                text: "alpha".to_owned(),
            },
            RerankDocument {
                doc_id: "doc-b".into(),
                text: "beta".to_owned(),
            },
            RerankDocument {
                doc_id: "doc-c".into(),
                text: "gamma".to_owned(),
            },
        ];
        let scores = adapter
            .rerank(&cx, "query", &docs)
            .await
            .expect("adapter rerank should succeed");
        let ids = scores
            .iter()
            .map(|score| score.doc_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["doc-b", "doc-a", "doc-c"]);
    });
}

#[test]
fn noop_metrics_exporter_callbacks_are_noops() {
    let exporter = NoOpMetricsExporter;

    let search_metrics = SearchMetrics {
        mode: crate::types::SearchMode::Hybrid,
        query_class: None,
        total_latency_ms: 10.0,
        phase1_latency_ms: Some(4.0),
        phase2_latency_ms: Some(6.0),
        result_count: 8,
        lexical_candidates: 30,
        semantic_candidates: 25,
        hash_control_candidates: 0,
        refined: true,
    };
    let embedding_metrics = EmbeddingMetrics {
        embedder_id: "fnv-hash-384".into(),
        batch_size: 1,
        duration_ms: 0.07,
        dimension: 384,
        is_semantic: false,
    };
    let index_metrics = IndexMetrics {
        doc_count: 100,
        index_size_bytes: 4096,
        updated_docs: 1,
        staleness_detected: false,
    };

    exporter.on_search_completed(&search_metrics);
    exporter.on_embedding_completed(&embedding_metrics);
    exporter.on_index_updated(&index_metrics);
    exporter.on_error(&SearchError::SearchTimeout {
        elapsed_ms: 11,
        budget_ms: 10,
    });
}

// ─── Utility function tests ─────────────────────────────────────────

#[test]
fn l2_normalize_produces_unit_vector() {
    let v = vec![3.0, 4.0];
    let normalized = l2_normalize(&v);
    let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn l2_normalize_zero_vector() {
    let v = vec![0.0, 0.0, 0.0];
    let normalized = l2_normalize(&v);
    assert!(normalized.iter().all(|&x| x == 0.0));
}

#[test]
fn l2_normalize_in_place_matches_allocating() {
    // The in-place variant must be bit-identical to the allocating one for
    // every input, including zero / near-zero / non-finite norm cases.
    let cases: &[Vec<f32>] = &[
        vec![],
        vec![0.0, 0.0, 0.0],
        vec![3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![-1.5, 0.25, 1e-3, 7.0, -7.0],
        vec![1e-30, 1e-30],       // near-zero norm → zero vector
        vec![f32::MAX, f32::MAX], // non-finite norm_sq → zero vector
    ];
    for v in cases {
        let allocating = l2_normalize(v);
        let mut in_place = v.clone();
        l2_normalize_in_place(&mut in_place);
        assert_eq!(in_place, allocating, "input={v:?}");
    }
}

#[test]
fn cosine_similarity_identical() {
    let v = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&v, &v);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert!(cosine_similarity(&a, &b).abs() < 1e-6);
}

#[test]
fn cosine_similarity_zero_vector() {
    let a = vec![1.0, 2.0];
    let b = vec![0.0, 0.0];
    assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
}

#[test]
fn truncate_embedding_reduces_dim() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let t = truncate_embedding(&v, 2);
    assert_eq!(t.len(), 2);
    let norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn truncate_embedding_noop_when_larger() {
    let v = vec![1.0, 2.0];
    assert_eq!(truncate_embedding(&v, 10), v);
}

#[test]
fn model_category_default_semantic_flag() {
    assert!(!ModelCategory::HashEmbedder.default_semantic_flag());
    assert!(ModelCategory::StaticEmbedder.default_semantic_flag());
    assert!(ModelCategory::TransformerEmbedder.default_semantic_flag());
}

/// Stand-in for the eager-batch backends — Tantivy, FTS5, the shadow
/// adapter — which implement `LexicalRead` without overriding
/// `search_candidates` or `hydrate_candidates`.
struct EagerOnlyLexical;

fn lexical_hit(id: &str, metadata: Option<serde_json::Value>) -> ScoredResult {
    ScoredResult {
        doc_id: compact_str::CompactString::from(id),
        score: 1.0,
        source: crate::types::ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(1.0),
        rerank_score: None,
        explanation: None,
        metadata: metadata.map(std::sync::Arc::new),
    }
}

impl LexicalRead for EagerOnlyLexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        _limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async {
            // An eager backend attaches metadata during the scoring search
            // itself, so there is nothing left to hydrate afterwards.
            Ok(vec![lexical_hit(
                "doc-a",
                Some(serde_json::json!({"rev": "v1"})),
            )])
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Ok(1)
    }
}

/// An eager batch carries its metadata and needs no context.
///
/// This is the positive half: the default hydration path must stay a
/// no-op for the shape it actually serves, or making it fail closed would
/// break every eager backend.
#[test]
fn eager_candidates_hydrate_without_a_context() {
    run_test_with_cx(|cx| async move {
        let backend = EagerOnlyLexical;
        let batch = backend
            .search_candidates(&cx, "alpha", 10)
            .await
            .expect("eager candidates");
        assert!(
            !batch.is_deferred(),
            "a backend that does not override search_candidates issues eager batches"
        );
        let (mut winners, context) = batch.into_parts();
        assert!(context.is_none(), "an eager batch carries no snapshot pin");
        assert!(
            winners[0].metadata.is_some(),
            "eager metadata is attached by the scoring search, so it is already \
             from the scoring generation"
        );
        backend
            .hydrate_candidates(&cx, context.as_ref(), &mut winners)
            .await
            .expect("hydrating an eager batch is a no-op, not an error");
    });
}

/// A foreign context must fail closed rather than silently succeed.
///
/// The negative half, and the one that was broken: the default used to be
/// `let _ = context; Ok(())`, so mixing another engine's snapshot pin into
/// an eager-batch backend returned success and left deferred winners with
/// no metadata restored. `bd-8nqz.1` requires that callers cannot mix a
/// hydration capability across engines or generations.
#[test]
fn a_foreign_hydration_context_is_rejected_with_a_typed_error() {
    run_test_with_cx(|cx| async move {
        let backend = EagerOnlyLexical;
        // A pin minted by some other engine, exactly as a mixing caller
        // would present it.
        let foreign = LexicalHydrationContext::new("quill", Box::new(7_u64));
        let mut winners = vec![lexical_hit("doc-a", None)];

        let error = backend
            .hydrate_candidates(&cx, Some(&foreign), &mut winners)
            .await
            .expect_err("a backend that issues no context must refuse to receive one");
        match error {
            SearchError::SubsystemError { subsystem, source } => {
                assert_eq!(subsystem, "lexical.hydration");
                let message = source.to_string();
                assert!(
                    message.contains("quill"),
                    "the rejection must name the foreign backend: {message}"
                );
            }
            other => panic!("expected a typed subsystem error, got {other:?}"),
        }
    });
}

struct BatchContractSyncEmbedder {
    identity: EmbeddingIdentityBundleV1,
    output_dimension: usize,
    outputs: Vec<Vec<f32>>,
    calls: std::sync::atomic::AtomicUsize,
}

impl BatchContractSyncEmbedder {
    fn new(outputs: Vec<Vec<f32>>) -> Self {
        Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("batch-contract", 3),
            output_dimension: 3,
            outputs,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl SyncEmbed for BatchContractSyncEmbedder {
    fn embed_sync(&self, _text: &str) -> SearchResult<Vec<f32>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(self.outputs.first().cloned().unwrap_or_default())
    }

    fn embed_batch_sync(&self, _texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(self.outputs.clone())
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.output_dimension
    }

    fn id(&self) -> &'static str {
        "batch-contract"
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

fn assert_contract_error(error: SearchError, expected_field: &str) {
    match error {
        SearchError::InvalidConfig { field, .. } => assert_eq!(field, expected_field),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn identity_bound_rejects_non_finite_values_without_repair_or_disclosure() {
    for bits in [
        0x7fc0_0000,
        0xffc0_0042,
        0x7f80_0001,
        0x7f80_0000,
        0xff80_0000,
    ] {
        let bound = IdentityBoundEmbedding {
            values: vec![12_345.5, f32::from_bits(bits), -9_876.25],
            identity: EmbeddingIdentityBundleV1::explicit_test_model("finite-contract", 3),
        };
        let error = bound
            .validate()
            .expect_err("non-finite output must fail closed");
        let message = error.to_string();
        assert!(message.contains("index 1"));
        assert!(!message.contains("12345"));
        assert!(!message.contains("9876"));
        assert_contract_error(error, "identity_bound_embedding.values");
        assert_eq!(bound.values[1].to_bits(), bits);
    }
}

#[test]
fn sync_bound_batch_cardinality_is_exact_including_empty_inputs() {
    for input_count in 0..=4 {
        let inputs = vec!["input"; input_count];
        for output_count in 0..=5 {
            let embedder = BatchContractSyncEmbedder::new(vec![vec![1.0; 3]; output_count]);
            let result = embedder.embed_batch_bound_sync(&inputs);
            if input_count == output_count {
                assert_eq!(result.unwrap().len(), input_count);
            } else {
                assert_contract_error(result.unwrap_err(), "embedder.batch_length");
            }
            assert_eq!(embedder.call_count(), 1);
        }
    }
}

#[test]
fn async_bound_batch_cardinality_is_exact_including_empty_inputs() {
    run_test_with_cx(|cx| async move {
        for input_count in 0..=4 {
            let inputs = vec!["input"; input_count];
            for output_count in 0..=5 {
                let adapter = SyncEmbedderAdapter(BatchContractSyncEmbedder::new(vec![
                    vec![1.0; 3];
                    output_count
                ]));
                let result = adapter.embed_batch_bound(&cx, &inputs).await;
                if input_count == output_count {
                    assert_eq!(result.unwrap().len(), input_count);
                } else {
                    assert_contract_error(result.unwrap_err(), "embedder.batch_length");
                }
                assert_eq!(adapter.0.call_count(), 1);
            }
        }
    });
}

#[test]
fn sync_bound_single_and_batch_reject_bad_shapes_and_non_finite_rows() {
    for output in [
        vec![],
        vec![1.0; 2],
        vec![1.0; 4],
        vec![1.0, f32::NAN, 0.0],
        vec![1.0, f32::INFINITY, 0.0],
        vec![1.0, f32::NEG_INFINITY, 0.0],
    ] {
        let single = BatchContractSyncEmbedder::new(vec![output.clone()]);
        assert_contract_error(
            single.embed_bound_sync("private-input").unwrap_err(),
            "identity_bound_embedding.values",
        );
        // A valid first row must not let an invalid later row escape as
        // a partially accepted batch.
        let batch = BatchContractSyncEmbedder::new(vec![vec![1.0; 3], output]);
        assert_contract_error(
            batch
                .embed_batch_bound_sync(&["first", "second"])
                .unwrap_err(),
            "identity_bound_embedding.values",
        );
    }
}

#[test]
fn async_bound_single_and_batch_reject_bad_shapes_and_non_finite_rows() {
    run_test_with_cx(|cx| async move {
        for output in [
            vec![],
            vec![1.0; 2],
            vec![1.0; 4],
            vec![1.0, f32::NAN, 0.0],
            vec![1.0, f32::INFINITY, 0.0],
            vec![1.0, f32::NEG_INFINITY, 0.0],
        ] {
            let single = SyncEmbedderAdapter(BatchContractSyncEmbedder::new(vec![output.clone()]));
            assert_contract_error(
                single.embed_bound(&cx, "private-input").await.unwrap_err(),
                "identity_bound_embedding.values",
            );
            let batch =
                SyncEmbedderAdapter(BatchContractSyncEmbedder::new(vec![vec![1.0; 3], output]));
            assert_contract_error(
                batch
                    .embed_batch_bound(&cx, &["first", "second"])
                    .await
                    .unwrap_err(),
                "identity_bound_embedding.values",
            );
        }
    });
}

#[test]
fn bound_identity_and_declared_dimension_are_admitted_before_inference() {
    run_test_with_cx(|cx| async move {
        for invalid_dimension in [false, true] {
            let mut embedder = BatchContractSyncEmbedder::new(vec![vec![1.0; 3]]);
            if invalid_dimension {
                embedder.output_dimension = 4;
            } else {
                embedder.identity.storage.format = "fsvi-v2".to_owned();
            }
            assert!(embedder.embed_bound_sync("input").is_err());
            assert!(embedder.embed_batch_bound_sync(&["input"]).is_err());
            assert!(embedder.embed_batch_bound_sync(&[]).is_err());
            assert_eq!(embedder.call_count(), 0);

            let adapter = SyncEmbedderAdapter(embedder);
            assert!(adapter.embed_bound(&cx, "input").await.is_err());
            assert!(adapter.embed_batch_bound(&cx, &["input"]).await.is_err());
            assert!(adapter.embed_batch_bound(&cx, &[]).await.is_err());
            assert_eq!(adapter.0.call_count(), 0);
        }
    });
}

fn assert_bound_rows_unchanged(
    rows: &[IdentityBoundEmbedding],
    expected: &BatchContractSyncEmbedder,
) {
    assert_eq!(rows.len(), expected.outputs.len());
    for (row, values) in rows.iter().zip(&expected.outputs) {
        row.validate().unwrap();
        assert_eq!(row.identity, expected.identity);
        let actual_bits: Vec<_> = row.values.iter().map(|value| value.to_bits()).collect();
        let expected_bits: Vec<_> = values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(actual_bits, expected_bits);
    }
}

#[test]
fn valid_bound_rows_preserve_order_identity_and_exact_float_bits() {
    run_test_with_cx(|cx| async move {
        let adapter = SyncEmbedderAdapter(BatchContractSyncEmbedder::new(vec![
            vec![0.0, -0.0, f32::from_bits(1)],
            vec![1.0, -7.25, f32::MAX],
        ]));
        let sync = adapter
            .0
            .embed_batch_bound_sync(&["first", "second"])
            .unwrap();
        assert_bound_rows_unchanged(&sync, &adapter.0);
        let asynchronous = adapter
            .embed_batch_bound(&cx, &["first", "second"])
            .await
            .unwrap();
        assert_bound_rows_unchanged(&asynchronous, &adapter.0);
        assert_eq!(adapter.0.call_count(), 2);
    });
}

#[test]
fn cancelled_bound_calls_never_dispatch_inference() {
    run_test_with_cx(|cx| async move {
        let adapter = SyncEmbedderAdapter(BatchContractSyncEmbedder::new(vec![vec![1.0; 3]]));
        cx.cancel_fast(asupersync::CancelKind::User);
        assert!(matches!(
            adapter.embed_bound(&cx, "input").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            adapter.embed_batch_bound(&cx, &["input"]).await,
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            adapter.embed_batch_bound(&cx, &[]).await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(adapter.0.call_count(), 0);
    });
}

struct CancelDuringInference {
    identity: EmbeddingIdentityBundleV1,
    calls: std::sync::atomic::AtomicUsize,
}

impl CancelDuringInference {
    fn new() -> Self {
        Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("cancel-during", 3),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Embedder for CancelDuringInference {
    fn embed<'a>(&'a self, cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            cx.cancel_fast(asupersync::CancelKind::User);
            // Deliberately return success after cancellation, modeling a
            // backend that checks only before its blocking inference.
            Ok(vec![1.0; 3])
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        3
    }

    fn id(&self) -> &'static str {
        "cancel-during"
    }

    fn model_name(&self) -> &'static str {
        "Cancellation fixture"
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

#[test]
fn default_async_batch_stops_after_cancellation_in_first_input() {
    run_test_with_cx(|cx| async move {
        let embedder = CancelDuringInference::new();
        assert!(matches!(
            embedder
                .embed_batch(&cx, &["first", "second", "third"])
                .await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(embedder.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    });
}

#[test]
fn bound_single_discards_success_returned_after_cancellation() {
    run_test_with_cx(|cx| async move {
        let embedder = CancelDuringInference::new();
        assert!(matches!(
            embedder.embed_bound(&cx, "input").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(embedder.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    });
}
