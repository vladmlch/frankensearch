use std::path::PathBuf;
use std::sync::Arc;
#[cfg(not(any(feature = "lexical", feature = "quill")))]
use std::sync::atomic::{AtomicUsize, Ordering};

use frankensearch_core::config::TwoTierConfig;
#[cfg(any(
    feature = "quill",
    feature = "hash",
    not(any(feature = "lexical", feature = "quill"))
))]
use frankensearch_core::error::SearchError;
#[cfg(feature = "hash")]
use frankensearch_core::traits::Embedder;
#[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
use frankensearch_core::traits::LexicalWrite;
#[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
use frankensearch_core::types::IndexableDocument;
use frankensearch_embed::auto_detect::{EmbedderStack, TwoTierAvailability};
#[cfg(target_os = "linux")]
use frankensearch_index::{FsviV2IdentityBinding, TwoTierIndexPaths};
#[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
use frankensearch_lexical::TantivyIndex;
#[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
use frankensearch_quill::{BlueGreenEngine, CurrentPointer, publish_current};
#[cfg(feature = "quill")]
use frankensearch_quill::{QuillConfig, QuillIndex};

#[cfg(feature = "quill")]
use super::super::LexicalReaderBackend;
#[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
use super::super::lexical::open_lexical_reader;
#[cfg(target_os = "linux")]
use super::super::open_admitted_v2_sync_with_residual_sidecar_cache;
use super::super::{
    IndexBuildStats, IndexBuilder, IndexProgress, IndexSizeBreakdown, LexicalArmReceipt,
    open_hybrid,
};
#[cfg(not(any(feature = "lexical", feature = "quill")))]
use super::cancel_on_first_stack;
#[cfg(target_os = "linux")]
use super::owned_admitted_v2_sync_dir;
use super::{RecordingExporter, SelectiveFailEmbedder, StubEmbedder, stub_stack};

#[test]
fn debug_impl() {
    let builder = IndexBuilder::new("/tmp/test").add_document("doc-1", "content");
    let debug = format!("{builder:?}");
    assert!(debug.contains("IndexBuilder"));
    assert!(debug.contains("doc_count"));
}

#[test]
fn batch_size_zero_clamped_to_one() {
    let builder = IndexBuilder::new("/tmp/test").with_batch_size(0);
    assert_eq!(builder.batch_size, 1);
}

#[test]
fn batch_size_one_still_works() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .with_batch_size(1)
            .add_document("doc-1", "First document")
            .add_document("doc-2", "Second document")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 2);
        assert_eq!(stats.error_count, 0);
    });
}

#[test]
fn index_build_stats_debug_clone() {
    let stats = IndexBuildStats {
        source_count: 6,
        doc_count: 5,
        error_count: 1,
        errors: vec![("bad-doc".into(), "embed failed".into())],
        quality_indexed: 4,
        quality_errors: vec![("slow-doc".into(), "quality timeout".into())],
        lexical: Some(LexicalArmReceipt {
            backend: "quill",
            path: PathBuf::from("/tmp/idx/lexical"),
            attempted: 6,
            indexed: 6,
            errors: Vec::new(),
            generation: None,
            published: true,
        }),
        size_bytes: IndexSizeBreakdown {
            total: 300,
            vector_fast: 100,
            vector_quality: 50,
            lexical: 150,
        },
        total_ms: 42.0,
        embed_ms: 30.0,
        lexical_ms: 5.0,
        has_quality_index: true,
        embedder_availability: TwoTierAvailability::Full,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.source_count, 6);
    assert_eq!(cloned.doc_count, 5);
    assert_eq!(cloned.error_count, 1);
    assert_eq!(cloned.errors.len(), 1);
    assert_eq!(cloned.quality_indexed, 4);
    assert_eq!(cloned.quality_errors.len(), 1);
    assert_eq!(cloned.lexical.as_ref().map(|arm| arm.indexed), Some(6));
    assert_eq!(cloned.size_bytes.total, 300);
    assert!(cloned.has_quality_index);
    let dbg = format!("{stats:?}");
    assert!(dbg.contains("IndexBuildStats"));
    assert!(dbg.contains("LexicalArmReceipt"));
}

#[test]
fn index_progress_debug_clone() {
    let progress = IndexProgress {
        completed: 50,
        total: 100,
        phase: "embedding",
    };
    let cloned = progress.clone();
    assert_eq!(cloned.completed, 50);
    assert_eq!(cloned.total, 100);
    assert_eq!(cloned.phase, "embedding");
    let dbg = format!("{progress:?}");
    assert!(dbg.contains("IndexProgress"));
}

#[test]
fn build_emits_embedding_and_index_metrics() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let exporter = Arc::new(RecordingExporter::default());
        let config = TwoTierConfig::default().with_metrics_exporter(exporter.clone());

        let stats = IndexBuilder::new(dir.path())
            .with_config(config)
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "Hello world")
            .add_document("doc-2", "Distributed consensus")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 2);
        assert_eq!(stats.error_count, 0);

        let embedding_count = {
            let embedding_events = exporter.embedding.lock().expect("embedding lock");
            embedding_events.len()
        };
        let (index_count, indexed_docs, indexed_bytes) = {
            let index_events = exporter.index.lock().expect("index lock");
            (
                index_events.len(),
                index_events.first().map_or(0, |event| event.doc_count),
                index_events
                    .first()
                    .map_or(0, |event| event.index_size_bytes),
            )
        };
        let error_count = {
            let errors = exporter.errors.lock().expect("errors lock");
            errors.len()
        };

        assert!(embedding_count >= 4);
        assert_eq!(index_count, 1);
        assert_eq!(indexed_docs, 2);
        assert!(indexed_bytes > 0);
        assert_eq!(error_count, 0);
    });
}

/// Per-arm quality receipts (bd-8nqz.3): a document whose quality
/// embedding fails stays in the fast tier and is reported per-document,
/// not silently absorbed into an aggregate.
#[test]
fn quality_receipts_track_partial_quality_failures() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stack = EmbedderStack::from_parts(
            Arc::new(StubEmbedder {
                id: "stub-fast",
                dim: 4,
            }),
            Some(Arc::new(SelectiveFailEmbedder)),
        );
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stack)
            .add_document("doc-clean", "quality succeeds here")
            .add_document("doc-marked", "fail-fast-embedding marker text")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.source_count, 2);
        assert_eq!(stats.doc_count, 2, "fast tier holds both documents");
        assert_eq!(stats.error_count, 0, "no fast embedding failed");
        assert_eq!(stats.quality_indexed, 1);
        assert_eq!(stats.quality_errors.len(), 1);
        assert_eq!(stats.quality_errors[0].0, "doc-marked");
    });
}

/// The preserved gate: when every fast embedding fails, the build still
/// errors — an index generation without a single vector record is not
/// representable by `TwoTierIndexBuilder::finish` (empty-generation
/// support is bd-tqhc index-crate scope, not facade scope).
#[test]
fn all_embeddings_failing_still_errors() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stack = EmbedderStack::from_parts(Arc::new(SelectiveFailEmbedder), None);
        let result = IndexBuilder::new(dir.path())
            .with_embedder_stack(stack)
            .add_document("doc-a", "fail-fast-embedding alpha")
            .add_document("doc-b", "fail-fast-embedding beta")
            .build(&cx)
            .await;

        assert!(result.is_err());
    });
}

/// Empty-content documents are valid source documents: they embed (the
/// embedder decides what an empty text means) and they are admitted to
/// the lexical arm (which may index zero tokens for them).
#[test]
fn empty_content_documents_are_admitted() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-empty", "")
            .add_document("doc-full", "real content")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.source_count, 2);
        assert_eq!(stats.doc_count, 2);
        assert_eq!(stats.error_count, 0);
        #[cfg(any(feature = "lexical", feature = "quill"))]
        {
            let receipt = stats.lexical.as_ref().expect("lexical arm receipt");
            assert_eq!(receipt.attempted, 2);
            assert_eq!(
                receipt.indexed + receipt.errors.len(),
                receipt.attempted,
                "every attempted document is accounted for exactly once",
            );
        }
    });
}

/// Duplicate IDs: pin that the lexical arm reports per-document errors in
/// the receipt instead of voiding the whole arm, and that accounting
/// stays exact.
#[cfg(feature = "quill")]
#[test]
fn duplicate_ids_land_in_lexical_receipt_not_aggregate_failure() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-dup", "alpha duplicate content")
            .add_document("doc-dup", "beta duplicate content")
            .add_document("doc-ok", "gamma unique content")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.source_count, 3);
        let receipt = stats.lexical.as_ref().expect("lexical arm receipt");
        assert_eq!(receipt.attempted, 3);
        assert_eq!(receipt.indexed, 2, "first doc-dup and doc-ok both index");
        assert_eq!(receipt.errors.len(), 1);
        assert_eq!(receipt.errors[0].0, "doc-dup");
        assert!(
            receipt.errors[0].1.contains("duplicate live document id"),
            "the receipt carries the typed duplicate rejection: {:?}",
            receipt.errors[0].1,
        );
        assert!(receipt.published);

        // The clean document AFTER the rejected duplicate survived — the
        // reconcile-commit recovery keeps one bad document from voiding
        // the rest of the arm.
        let lexical = QuillIndex::open(&cx, dir.path().join("lexical"), QuillConfig::default())
            .await
            .unwrap();
        let gamma_ids = lexical
            .search_doc_ids(&cx, "gamma", 10)
            .unwrap()
            .iter()
            .map(|hit| hit.document_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(gamma_ids, vec!["doc-ok".to_owned()]);
    });
}

/// bd-8nqz.2: a tantivy meta.json squatting in the lexical dir must be a
/// typed refusal, not a silent Quill initialization beside it (MANIFEST
/// absence is not emptiness).
#[cfg(feature = "quill")]
#[test]
fn build_refuses_quill_init_over_foreign_lexical_layout() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let lexical_dir = dir.path().join("lexical");
        std::fs::create_dir_all(&lexical_dir).unwrap();
        std::fs::write(lexical_dir.join("meta.json"), b"{}").unwrap();

        let error = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "content")
            .build(&cx)
            .await
            .expect_err("foreign lexical layout must refuse Quill init");
        let message = error.to_string();
        assert!(
            message.contains("direct-tantivy"),
            "error must carry the typed layout label: {message}"
        );
        assert!(
            !lexical_dir.join("MANIFEST").exists(),
            "no Quill artifacts may appear beside the foreign index"
        );
    });
}

/// bd-8nqz.2: `open_hybrid` reports a mixed lexical layout as a typed
/// error instead of silently picking an engine.
#[cfg(feature = "quill")]
#[test]
fn open_hybrid_reports_mixed_layout_as_typed_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "Alpha content")
            .build(&cx)
            .await
            .unwrap();
        std::fs::write(dir.path().join("lexical").join("meta.json"), b"{}").unwrap();

        let error = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
            .await
            .expect_err("mixed layout must be a typed error");
        assert!(
            error.to_string().contains("mixed"),
            "error must carry the typed layout label: {error}"
        );
    });
}

/// bd-8nqz.2: when both backends are compiled, a published Tantivy
/// generation is a supported rollback target. The root inspector decides
/// the engine; no Quill open is attempted against Tantivy's `meta.json`.
#[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
#[test]
fn open_lexical_reader_opens_current_tantivy_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let lexical_root = root.path().join("lexical");
        let tantivy_dir = lexical_root.join("tantivy-v1");
        std::fs::create_dir_all(&tantivy_dir).unwrap();

        let tantivy = TantivyIndex::create(&tantivy_dir).unwrap();
        LexicalWrite::index_document(
            &tantivy,
            &cx,
            &IndexableDocument::new("tantivy-doc", "rollback sentinel"),
        )
        .await
        .unwrap();
        LexicalWrite::commit(&tantivy, &cx).await.unwrap();
        // Tantivy's IndexWriter holds an exclusive directory lock, so the
        // rollback generation has to be CLOSED before a reader may open
        // it. Without this the open below fails LockBusy — which is what
        // it did, undetected, from the day this test was written: it is
        // gated on all(quill, lexical-tantivy) and no lane ever enabled
        // both, so `-p frankensearch --lib open_lexical_reader` reported a
        // vacuous "ok. 0 passed" in every single-feature selection
        // (bd-8nqz.2, bd-jt7b2's defect class one layer up).
        drop(tantivy);
        publish_current(
            &lexical_root,
            &CurrentPointer::new(BlueGreenEngine::Tantivy, "tantivy-v1", 0).unwrap(),
        )
        .unwrap();

        let (lexical, backend) = open_lexical_reader(&cx, &lexical_root).await.unwrap();
        assert_eq!(backend, Some(LexicalReaderBackend::TantivyOracle));
        let lexical = lexical.expect("published Tantivy generation must open");
        let hits = lexical.search(&cx, "rollback", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "tantivy-doc");
    });
}

/// A fast embedder that cancels the caller context but completes its own
/// work must still stop the vector-only build at the post-embed checkpoint,
/// before the second document is touched.
#[cfg(not(any(feature = "lexical", feature = "quill")))]
#[test]
fn vector_only_fast_cancellation_stops_later_documents() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let error = IndexBuilder::new(dir.path())
            .with_embedder_stack(cancel_on_first_stack("fast-canceller", Arc::clone(&calls)))
            .add_document("doc-1", "first")
            .add_document("doc-2", "must not be embedded")
            .build(&cx)
            .await
            .expect_err("fast cancellation must abort the vector-only build");

        assert!(
            matches!(
                &error,
                SearchError::Cancelled { phase, reason }
                    if phase == "fast document embedding completion"
                        && reason.contains("cancel-on-first embedder")
            ),
            "caller-context cancellation must stay typed, got {error:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    });
}

/// Quality cancellation must not be downgraded into a fast-only success
/// receipt. It aborts the vector-only build before the next document.
#[cfg(not(any(feature = "lexical", feature = "quill")))]
#[test]
fn vector_only_quality_cancellation_is_not_degraded() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let error = IndexBuilder::new(dir.path())
            .with_embedder_stack(cancel_on_first_stack(
                "quality-canceller",
                Arc::clone(&calls),
            ))
            .add_document("doc-1", "first")
            .add_document("doc-2", "must not be embedded")
            .build(&cx)
            .await
            .expect_err("quality cancellation must abort the vector-only build");

        assert!(
            matches!(
                &error,
                SearchError::Cancelled { phase, reason }
                    if phase == "quality-canceller embedding"
                        && reason == "cancel-on-first embedder"
            ),
            "typed quality cancellation must survive unchanged, got {error:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    });
}

/// bd-8nqz.3 cancellation matrix: a cancelled `Cx` aborts the build with
/// the typed cancellation error — never a per-document receipt entry,
/// never recovery-machinery churn — and a cleared `Cx` builds clean.
#[cfg(feature = "quill")]
#[test]
fn build_rejects_cancelled_cx_with_typed_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        cx.set_cancel_requested(true);
        let error = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "content")
            .build(&cx)
            .await
            .expect_err("cancelled cx must reject the build");
        assert!(
            matches!(error, SearchError::Cancelled { .. }),
            "typed cancellation must survive to the caller, got {error:?}"
        );

        // Retry-clean: a cleared cx builds successfully from scratch.
        cx.set_cancel_requested(false);
        let fresh = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(fresh.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "content")
            .build(&cx)
            .await
            .expect("cleared cx must build clean");
        assert_eq!(stats.doc_count, 1);
    });
}

/// `open_hybrid` (bd-8nqz.3): the ergonomic opener must attach the
/// lexical arm the advertised examples previously dropped, and the
/// attached reader must actually answer through the trait object.
#[cfg(any(feature = "lexical", feature = "quill"))]
#[test]
fn open_hybrid_attaches_lexical_arm() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "Alpha retrieval content")
            .add_document("doc-2", "Beta ranking content")
            .build(&cx)
            .await
            .unwrap();

        let parts = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
            .await
            .unwrap();
        assert_eq!(parts.vectors.doc_count(), 2);
        #[cfg(feature = "quill")]
        assert_eq!(parts.lexical_backend, Some(LexicalReaderBackend::Quill));
        let lexical = parts.lexical.expect("lexical arm must be attached");
        let hits = lexical.search(&cx, "Alpha", 5).await.unwrap();
        assert!(!hits.is_empty(), "trait-object search must answer");
    });
}

/// Without a lexical backend compiled in, `open_hybrid` still opens the
/// vector arms and reports the lexical arm as absent rather than erroring.
#[cfg(not(any(feature = "lexical", feature = "quill")))]
#[test]
fn open_hybrid_without_lexical_backend_reports_absent_arm() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "Alpha retrieval content")
            .build(&cx)
            .await
            .unwrap();

        let parts = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
            .await
            .unwrap();
        assert_eq!(parts.vectors.doc_count(), 1);
        assert!(parts.lexical.is_none());
        assert!(parts.lexical_backend.is_none());
    });
}

#[cfg(feature = "hash")]
#[test]
fn open_hybrid_refuses_a_hash_identity_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let hash_stack = EmbedderStack::from_parts(
            Arc::new(frankensearch_embed::HashEmbedder::default_256()) as Arc<dyn Embedder>,
            None,
        );
        IndexBuilder::new(dir.path())
            .with_embedder_stack(hash_stack)
            .add_document("doc-1", "Hello world")
            .build(&cx)
            .await
            .unwrap();

        let error = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
            .await
            .expect_err("hash generations must not open as hybrid search");
        assert!(
            matches!(error, SearchError::EmbedderUnavailable { .. }),
            "expected EmbedderUnavailable, got: {error:?}"
        );
    });
}

#[cfg(target_os = "linux")]
#[test]
fn admitted_v2_sync_opener_constructs_the_shipping_residual_cache_product() {
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use frankensearch_core::{BoundQueryEmbedding, TieredQueryEmbeddings};

    let dir = owned_admitted_v2_sync_dir();
    let source_path = dir.join("fast.fsvi");
    let cache_dir = dir.join("residual-cache");
    std::fs::create_dir(&cache_dir).expect("create a fresh owned cache directory");

    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("facade-route", 2);
    "fsvi-v2".clone_into(&mut identity.storage.format);
    identity.storage.quantization = QuantizationFormat::F16;
    "little-endian".clone_into(&mut identity.storage.endianness);
    let binding = FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(91, [0x6d; 16]).expect("create a test generation"),
        identity.freeze().expect("freeze test identity"),
    )
    .expect("create a valid v2 binding");
    let mut writer = frankensearch_index::VectorIndex::create_v2(&source_path, binding.clone())
        .expect("create an admitted-v2 fixture");
    writer
        .write_record("exact-winner", &[0.0, 1.0])
        .expect("write winner");
    writer
        .write_record("other", &[1.0, 0.0])
        .expect("write other row");
    writer.finish().expect("seal fixture");

    let paths = TwoTierIndexPaths::new(&source_path);
    #[cfg(feature = "ann")]
    let discarded_ann_path = dir.join("discarded-by-sync-product.hnsw");
    #[cfg(feature = "ann")]
    let paths = paths.with_fast_ann(&discarded_ann_path);
    let searcher = open_admitted_v2_sync_with_residual_sidecar_cache(
        &paths,
        &binding,
        None,
        &cache_dir,
        TwoTierConfig {
            fast_only: true,
            // Force the ordinary two-tier opener's ANN threshold. The
            // synchronous residual facade must still not build or persist
            // this configured sidecar because it discards that container.
            hnsw_threshold: 0,
            ..TwoTierConfig::default()
        },
    )
    .expect("the facade opener constructs the shipping sync product");
    let query = TieredQueryEmbeddings::fast_only(
        BoundQueryEmbedding::new(
            vec![0.0, 1.0],
            EmbeddingIdentityBundleV1::explicit_test_model("facade-route", 2),
        )
        .expect("bind v2 query identity"),
    );
    let (results, _) = searcher
        .search_collect(&query, 1)
        .expect("search through the facade product opener");
    assert_eq!(results[0].doc_id, "exact-winner");
    assert_eq!(
        std::fs::read_dir(&cache_dir)
            .expect("inspect owned cache directory")
            .flatten()
            .count(),
        1,
        "the default facade opener must route through residual-cache publication"
    );
    #[cfg(feature = "ann")]
    assert!(
        !discarded_ann_path.exists(),
        "the owner-only synchronous facade must strip ANN paths instead of persisting an \
         artifact its in-memory product discards"
    );
}
