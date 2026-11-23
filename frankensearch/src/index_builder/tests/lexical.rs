#[cfg(feature = "lexical-tantivy")]
use std::path::PathBuf;
#[cfg(any(feature = "lexical", feature = "quill", feature = "lexical-tantivy"))]
use std::sync::Arc;

use frankensearch_core::config::TwoTierConfig;
#[cfg(any(feature = "quill", all(feature = "hash", feature = "lexical-tantivy")))]
use frankensearch_core::error::SearchError;
#[cfg(any(
    feature = "lexical-tantivy",
    all(feature = "lexical", not(feature = "quill"))
))]
use frankensearch_core::traits::LexicalRead;
use frankensearch_core::types::IndexableDocument;
#[cfg(any(feature = "lexical", feature = "quill", feature = "lexical-tantivy"))]
use frankensearch_embed::auto_detect::EmbedderStack;
use frankensearch_index::TwoTierIndex;
#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
use frankensearch_index::{VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME};
#[cfg(any(
    feature = "lexical-tantivy",
    all(feature = "lexical", not(feature = "quill"))
))]
use frankensearch_lexical::TantivyIndex;
#[cfg(feature = "quill")]
use frankensearch_quill::{
    DEFAULT_SCHEMA, EncodedSegment, QuillConfig, QuillIndex, SectionInput, SectionKind,
    SegmentHeaderInput, SegmentReader, load_manifest_pair,
};

use super::super::IndexBuilder;
#[cfg(feature = "quill")]
use super::super::open_hybrid;
#[cfg(any(feature = "lexical", feature = "quill", feature = "lexical-tantivy"))]
use super::SelectiveFailEmbedder;
use super::{fast_only_stack, stub_stack};

#[test]
fn build_fast_only() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(fast_only_stack())
            .add_document("doc-1", "Test content")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 1);
        assert!(!stats.has_quality_index);

        let index = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
        assert!(!index.has_quality_index());
    });
}

#[test]
fn build_empty_documents_returns_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let result = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .build(&cx)
            .await;

        assert!(result.is_err());
    });
}

#[test]
fn build_with_progress_callback() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let progress_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = progress_count.clone();

        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .with_batch_size(2)
            .add_document("doc-1", "First")
            .add_document("doc-2", "Second")
            .add_document("doc-3", "Third")
            .with_progress(move |_p| {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            })
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 3);
        assert!(progress_count.load(std::sync::atomic::Ordering::Relaxed) > 0);
    });
}

#[test]
fn build_with_title() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document_with_title("doc-1", "Content here", "My Title")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 1);
    });
}

#[test]
fn build_with_multiple_documents() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let docs = vec![
            IndexableDocument::new("a", "Alpha content"),
            IndexableDocument::new("b", "Beta content"),
            IndexableDocument::new("c", "Gamma content"),
        ];

        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_documents(docs)
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 3);
    });
}

#[cfg(any(feature = "lexical", feature = "quill"))]
#[test]
fn build_wires_lexical_index_when_feature_enabled() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-1", "Alpha retrieval content")
            .add_document("doc-2", "Beta ranking content")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.doc_count, 2);

        #[cfg(feature = "quill")]
        let hits = {
            let lexical = QuillIndex::open(&cx, dir.path().join("lexical"), QuillConfig::default())
                .await
                .unwrap();
            lexical.search_results(&cx, "Alpha", 5).unwrap()
        };
        #[cfg(all(feature = "lexical", not(feature = "quill")))]
        let hits = {
            let lexical = TantivyIndex::open(&dir.path().join("lexical")).unwrap();
            lexical.search(&cx, "Alpha", 5).await.unwrap()
        };
        assert!(!hits.is_empty());
    });
}

#[cfg(feature = "lexical-tantivy")]
fn retained_tantivy_builder_dir() -> PathBuf {
    let dir = tempfile::Builder::new()
        .prefix("frankensearch-tantivy-builder-")
        .tempdir()
        .expect("create Tantivy builder fixture")
        .keep();
    eprintln!("retained Tantivy builder fixture: {}", dir.display());
    dir
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
fn tantivy_hash_stack() -> EmbedderStack {
    EmbedderStack::from_parts(
        Arc::new(frankensearch_embed::HashEmbedder::default_256()),
        None,
    )
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_explicit_selection_publishes_queryable_index() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let stats = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .add_document_with_title("doc-alpha", "ordinary body", "sentinelalpha")
            .add_document("doc-beta", "sentinelbeta retrieval")
            .build(&cx)
            .await
            .expect("publish explicit Tantivy index");

        assert_eq!(stats.source_count, 2);
        assert_eq!(stats.doc_count, 2);
        assert_eq!(stats.error_count, 0);
        let receipt = stats.lexical.expect("Tantivy receipt");
        assert_eq!(receipt.backend, "tantivy");
        assert_eq!(receipt.path, dir.join("lexical"));
        assert_eq!(receipt.attempted, 2);
        assert_eq!(receipt.indexed, 2);
        assert!(receipt.errors.is_empty());
        assert!(receipt.published);
        assert!(stats.size_bytes.lexical > 0);
        assert_eq!(
            stats.size_bytes.total,
            stats.size_bytes.vector_fast
                + stats.size_bytes.vector_quality
                + stats.size_bytes.lexical,
        );
        let vectors = TwoTierIndex::open(&dir, TwoTierConfig::default())
            .expect("reopen vector generation from the same build");
        assert_eq!(vectors.doc_count(), 2);
        let lexical = TantivyIndex::open_read_only(&receipt.path)
            .expect("reopen the committed Tantivy generation");
        for (query, expected) in [("sentinelalpha", "doc-alpha"), ("sentinelbeta", "doc-beta")] {
            let hits = lexical.search(&cx, query, 10).await.expect("query Tantivy");
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, expected);
        }
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_admits_empty_lexical_directory() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let lexical_path = dir.join("lexical");
        std::fs::create_dir(&lexical_path).expect("create empty lexical directory");
        let stats = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .add_document("doc-empty-dir", "emptydirsentinel")
            .build(&cx)
            .await
            .expect("build into an empty lexical directory");
        let receipt = stats.lexical.expect("published lexical receipt");
        assert_eq!(receipt.backend, "tantivy");
        assert_eq!(receipt.indexed, 1);
        assert!(receipt.published);
        let lexical = TantivyIndex::open_read_only(&lexical_path).expect("reopen Tantivy");
        let hits = lexical
            .search(&cx, "emptydirsentinel", 10)
            .await
            .expect("query the published generation");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-empty-dir");
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_reopens_existing_tantivy_and_preserves_upserts() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let lexical_path = dir.join("lexical");
        let existing = TantivyIndex::create(&lexical_path).expect("create existing Tantivy");
        frankensearch_core::traits::LexicalWrite::index_documents(
            &existing,
            &cx,
            &[
                IndexableDocument::new("doc-upsert", "oldsentinel"),
                IndexableDocument::new("doc-retained", "retainedsentinel"),
            ],
        )
        .await
        .expect("index existing documents");
        frankensearch_core::traits::LexicalWrite::commit(&existing, &cx)
            .await
            .expect("commit existing Tantivy");
        drop(existing);

        let stats = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .add_document("doc-upsert", "newsentinel")
            .build(&cx)
            .await
            .expect("reopen and update the existing Tantivy generation");
        let receipt = stats.lexical.expect("updated lexical receipt");
        assert_eq!(receipt.attempted, 1);
        assert_eq!(receipt.indexed, 1);
        assert!(receipt.errors.is_empty());
        assert!(receipt.published);
        let lexical = TantivyIndex::open_read_only(&lexical_path).expect("reopen updated index");
        assert_eq!(
            LexicalRead::doc_count(&lexical).expect("count documents"),
            2
        );
        assert!(
            lexical
                .search(&cx, "oldsentinel", 10)
                .await
                .expect("query replaced content")
                .is_empty()
        );
        for (query, expected) in [
            ("newsentinel", "doc-upsert"),
            ("retainedsentinel", "doc-retained"),
        ] {
            let hits = lexical
                .search(&cx, query, 10)
                .await
                .expect("query existing index");
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, expected);
        }
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_writer_conflict_preserves_existing_vectors_and_documents() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .add_document("doc-retained", "retainedwritersentinel")
            .build(&cx)
            .await
            .expect("publish existing generation");
        let lexical_path = dir.join("lexical");
        let existing = TantivyIndex::open(&lexical_path).expect("hold existing writer");
        let vector_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let before_vectors = std::fs::read(&vector_path).expect("read existing vectors");
        let before_meta =
            std::fs::read(lexical_path.join("meta.json")).expect("read existing Tantivy metadata");
        let error = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .add_document("doc-refused", "refusedwritersentinel")
            .build(&cx)
            .await
            .expect_err("writer contention must refuse before replacing vectors");
        assert!(matches!(
            error,
            SearchError::SubsystemError {
                subsystem: "tantivy",
                ..
            }
        ));
        assert_eq!(std::fs::read(&vector_path).unwrap(), before_vectors);
        assert_eq!(
            std::fs::read(lexical_path.join("meta.json")).unwrap(),
            before_meta,
        );
        let hits = existing
            .search(&cx, "retainedwritersentinel", 10)
            .await
            .expect("query retained document while original writer remains live");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-retained");
        assert!(
            existing
                .search(&cx, "refusedwritersentinel", 10)
                .await
                .expect("query refused document")
                .is_empty()
        );
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_late_writer_creation_failure_preserves_existing_vectors() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let mut vectors = TwoTierIndex::create(&dir, TwoTierConfig::default()).unwrap();
        vectors
            .add_fast_record("doc-retained", &[1.0, 0.0])
            .unwrap();
        drop(vectors.finish().unwrap());
        let vector_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let before_vectors = std::fs::read(&vector_path).unwrap();
        let lexical_path = dir.join("lexical");
        let obstruction = lexical_path.clone();
        let error = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .with_progress(move |_| {
                std::fs::write(&obstruction, b"retained late obstruction").unwrap();
            })
            .add_document("doc-refused", "latewriterfailure")
            .build(&cx)
            .await
            .expect_err("late writer admission failure precedes vector publication");
        assert!(matches!(
            error,
            SearchError::SubsystemError {
                subsystem: "tantivy",
                ..
            }
        ));
        assert_eq!(std::fs::read(&vector_path).unwrap(), before_vectors);
        assert_eq!(
            std::fs::read(&lexical_path).unwrap(),
            b"retained late obstruction",
        );
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_refuses_mixed_markers_before_touching_vectors() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for marker in ["MANIFEST", "CURRENT"] {
            let dir = retained_tantivy_builder_dir();
            let lexical_path = dir.join("lexical");
            let lexical = TantivyIndex::create(&lexical_path).expect("create Tantivy");
            frankensearch_core::traits::LexicalWrite::index_documents(
                &lexical,
                &cx,
                &[IndexableDocument::new(
                    "doc-retained",
                    "mixedmarkersentinel",
                )],
            )
            .await
            .expect("index retained document");
            frankensearch_core::traits::LexicalWrite::commit(&lexical, &cx)
                .await
                .expect("publish retained document");
            drop(lexical);
            let marker_path = lexical_path.join(marker);
            std::fs::write(&marker_path, b"retained conflicting marker").unwrap();
            let before_meta = std::fs::read(lexical_path.join("meta.json")).unwrap();
            let error = IndexBuilder::new(&dir)
                .with_embedder_stack(tantivy_hash_stack())
                .with_tantivy_lexical()
                .with_progress(|_| panic!("mixed layout must refuse before embedding"))
                .add_document("doc-refused", "refusedmixedsentinel")
                .build(&cx)
                .await
                .expect_err("readable Tantivy does not authorize a mixed layout");
            assert!(matches!(
                error,
                SearchError::InvalidConfig { field, .. } if field == "data_dir/lexical"
            ));
            assert!(!dir.join(VECTOR_INDEX_FAST_FILENAME).exists());
            assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());
            assert_eq!(
                std::fs::read(&marker_path).unwrap(),
                b"retained conflicting marker",
            );
            assert_eq!(
                std::fs::read(lexical_path.join("meta.json")).unwrap(),
                before_meta,
            );
            let lexical = TantivyIndex::open_read_only(&lexical_path).unwrap();
            let hits = lexical
                .search(&cx, "mixedmarkersentinel", 10)
                .await
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "doc-retained");
        }
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_refuses_foreign_schema_without_replacing_vectors() {
    use frankensearch_lexical::{
        Count, Index, IndexRecordOption, IndexWriter, Schema, TantivyDocument, Term, TermQuery,
    };

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let lexical_path = dir.join("lexical");
        std::fs::create_dir(&lexical_path).unwrap();
        let template = TantivyIndex::in_memory().unwrap();
        let template_schema = template.index_handle().schema();
        let tokenizers = template.index_handle().tokenizers().clone();
        let mut fields: Vec<_> = template_schema
            .fields()
            .map(|(_, entry)| entry.clone())
            .collect();
        drop(template);
        // Every field has the shipping name and options, but the adapter's
        // hard-coded ID/content handles point at the wrong fields.
        fields.swap(0, 1);
        let mut schema = Schema::builder();
        for field in fields {
            schema.add_field(field);
        }
        let schema = schema.build();
        let id = schema.get_field("id").unwrap();
        let mut foreign = Index::create_in_dir(&lexical_path, schema).unwrap();
        foreign.set_tokenizers(tokenizers);
        let mut writer: IndexWriter = foreign.writer(50_000_000).unwrap();
        let mut document = TantivyDocument::default();
        document.add_text(id, "doc-retained");
        writer.add_document(document).unwrap();
        writer.commit().unwrap();
        drop(writer);

        let mut vectors = TwoTierIndex::create(&dir, TwoTierConfig::default()).unwrap();
        vectors
            .add_fast_record("doc-retained", &[1.0, 0.0])
            .unwrap();
        drop(vectors.finish().unwrap());
        let vector_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let before_vectors = std::fs::read(&vector_path).unwrap();
        let before_meta = std::fs::read(lexical_path.join("meta.json")).unwrap();
        let error = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .add_document("doc-refused", "foreign schema must not accept writes")
            .build(&cx)
            .await
            .expect_err("openable Tantivy is not sufficient adapter-schema admission");
        assert!(matches!(
            error,
            SearchError::InvalidConfig { field, .. } if field == "data_dir/lexical.schema"
        ));
        assert_eq!(std::fs::read(&vector_path).unwrap(), before_vectors);
        assert_eq!(
            std::fs::read(lexical_path.join("meta.json")).unwrap(),
            before_meta,
        );
        let reader: frankensearch_lexical::IndexReader = foreign.reader().unwrap();
        let retained = TermQuery::new(
            Term::from_field_text(id, "doc-retained"),
            IndexRecordOption::Basic,
        );
        assert_eq!(reader.searcher().search(&retained, &Count).unwrap(), 1);
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_preserves_legacy_schema_without_ordinal_field() {
    use frankensearch_lexical::{Index, Schema};

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let lexical_path = dir.join("lexical");
        std::fs::create_dir(&lexical_path).unwrap();
        let template = TantivyIndex::in_memory().unwrap();
        let template_schema = template.index_handle().schema();
        let mut schema = Schema::builder();
        for (_, entry) in template_schema
            .fields()
            .filter(|(_, entry)| entry.name() != "ord")
        {
            schema.add_field(entry.clone());
        }
        drop(template);
        drop(Index::create_in_dir(&lexical_path, schema.build()).unwrap());
        let stats = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .add_document("doc-legacy", "legacyordinalsentry")
            .build(&cx)
            .await
            .expect("the supported pre-ordinal schema remains writable");
        assert!(stats.lexical.unwrap().published);
        let lexical = TantivyIndex::open_read_only(&lexical_path).unwrap();
        assert!(lexical.index_handle().schema().get_field("ord").is_err());
        let hits = lexical
            .search(&cx, "legacyordinalsentry", 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-legacy");
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_refuses_foreign_directory_before_touching_vectors() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let lexical_path = dir.join("lexical");
        std::fs::create_dir(&lexical_path).expect("create foreign lexical fixture");
        let foreign_path = lexical_path.join("foreign.format");
        std::fs::write(&foreign_path, b"foreign lexical sentinel")
            .expect("retain foreign-format bytes");
        let vector_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        std::fs::write(&vector_path, b"existing vector sentinel").expect("retain vector bytes");
        let error = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .with_progress(|_| panic!("foreign layout must refuse before embedding"))
            .add_document("doc-denied", "not indexed")
            .build(&cx)
            .await
            .expect_err("foreign lexical layout must refuse before vector creation");
        assert!(matches!(
            error,
            SearchError::SubsystemError {
                subsystem: "tantivy",
                ..
            }
        ));
        assert_eq!(
            std::fs::read(&vector_path).expect("read retained vector bytes"),
            b"existing vector sentinel",
        );
        assert_eq!(
            std::fs::read(&foreign_path).expect("read retained foreign bytes"),
            b"foreign lexical sentinel",
        );
        assert!(!lexical_path.join("meta.json").exists());
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_feature_alone_preserves_default_backend() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let stats = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .add_document("doc-default", "defaultsentinel")
            .build(&cx)
            .await
            .expect("build without an explicit Tantivy selection");
        assert_eq!(stats.doc_count, 1);
        #[cfg(feature = "quill")]
        {
            let receipt = stats.lexical.expect("default Quill receipt");
            assert_eq!(receipt.backend, "quill");
            assert!(receipt.published);
            assert!(!receipt.path.join("meta.json").exists());
            let lexical = QuillIndex::open(&cx, &receipt.path, QuillConfig::default())
                .await
                .expect("reopen the default Quill generation");
            let hits = lexical
                .search_results(&cx, "defaultsentinel", 10)
                .expect("query default Quill generation");
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "doc-default");
        }
        #[cfg(not(feature = "quill"))]
        {
            assert!(stats.lexical.is_none());
            assert_eq!(stats.lexical_ms.to_bits(), 0.0_f64.to_bits());
            assert_eq!(stats.size_bytes.lexical, 0);
            assert!(!dir.join("lexical").exists());
        }
    });
}

#[cfg(feature = "lexical-tantivy")]
#[test]
fn tantivy_builder_keeps_failed_embedding_documents_searchable() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        // Inject only the embedding failure; indexing and query execution
        // still use the real on-disk Tantivy backend.
        let stack = EmbedderStack::from_parts(Arc::new(SelectiveFailEmbedder), None);
        let stats = IndexBuilder::new(&dir)
            .with_embedder_stack(stack)
            .with_tantivy_lexical()
            .add_document("doc-ok", "ordinary text")
            .add_document("doc-failed", "fail-fast-embedding rescuedsentinel")
            .build(&cx)
            .await
            .expect("publish partial vectors and complete lexical arm");

        assert_eq!(stats.source_count, 2);
        assert_eq!(stats.doc_count, 1);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.errors.len(), 1);
        assert_eq!(stats.errors[0].0, "doc-failed");
        let receipt = stats.lexical.expect("Tantivy receipt");
        assert_eq!(receipt.attempted, 2);
        assert_eq!(receipt.indexed, 2);
        assert!(receipt.errors.is_empty());
        assert!(receipt.published);
        let lexical = TantivyIndex::open_read_only(&receipt.path).expect("reopen Tantivy");
        let hits = lexical
            .search(&cx, "rescuedsentinel", 10)
            .await
            .expect("query document absent from vector tier");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-failed");
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_cancellation_prevents_lexical_publication() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let cancel_cx = cx.clone();
        let error = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .with_progress(move |progress| {
                assert_eq!(progress.phase, "embedding");
                assert_eq!(progress.completed, progress.total);
                cancel_cx.cancel_fast(asupersync::CancelKind::User);
            })
            .add_document("doc-cancel", "cancelledsentinel")
            .build(&cx)
            .await
            .expect_err("cancellation after embedding remains fatal");
        assert!(matches!(
            error,
            SearchError::Cancelled { phase, .. } if phase == "vector index finalize"
        ));
        assert!(!dir.join("lexical").exists());
    });
}

#[cfg(all(feature = "hash", feature = "lexical-tantivy"))]
#[test]
fn tantivy_builder_lexical_publication_error_is_fatal() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = retained_tantivy_builder_dir();
        let lexical_path = dir.join("lexical");
        std::fs::write(&lexical_path, b"retained path obstruction")
            .expect("create a file where the lexical directory is required");
        let error = IndexBuilder::new(&dir)
            .with_embedder_stack(tantivy_hash_stack())
            .with_tantivy_lexical()
            .add_document("doc-blocked", "blocked publication")
            .build(&cx)
            .await
            .expect_err("lexical initialization failure cannot return successful stats");
        assert!(matches!(
            error,
            SearchError::SubsystemError {
                subsystem: "tantivy",
                ..
            }
        ));
        assert_eq!(
            std::fs::read(&lexical_path).expect("read retained obstruction"),
            b"retained path obstruction",
        );
    });
}

/// Eager TERMDICT admission deliberately upgrades malformed dictionary
/// bytes from a query-time subsystem failure to an open-time corruption
/// diagnosis. Pin that public facade contract with a structurally valid,
/// checksum-consistent FSLX image whose dictionary payload is malformed.
#[cfg(feature = "quill")]
#[test]
fn open_hybrid_reports_malformed_termdict_as_index_corrupted() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        IndexBuilder::new(dir.path())
            .with_embedder_stack(stub_stack())
            .add_document("doc-alpha", "alpha retrieval sentinel")
            .build(&cx)
            .await
            .expect("build intact hybrid fixture");

        let intact = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
            .await
            .expect("open intact hybrid fixture");
        let lexical = intact.lexical.as_ref().expect("Quill lexical arm");
        let hits = lexical
            .search(&cx, "alpha", 10)
            .await
            .expect("search intact Quill arm");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id.as_str(), "doc-alpha");
        drop(intact);

        let lexical_dir = dir.path().join("lexical");
        let segment_path = std::fs::read_dir(&lexical_dir)
            .expect("read lexical fixture directory")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "fslx")
            })
            .expect("published FSLX fixture");
        let reader = SegmentReader::from_owned(
            std::fs::read(&segment_path).expect("read intact FSLX fixture"),
            DEFAULT_SCHEMA,
        )
        .expect("parse intact FSLX fixture");
        let header = reader.header();
        let malformed_termdict = 1_u32.to_le_bytes();
        let payloads = reader
            .section_entries()
            .iter()
            .map(|entry| {
                let bytes = if entry.kind == SectionKind::TERMDICT {
                    malformed_termdict.to_vec()
                } else {
                    reader
                        .section(entry.kind)
                        .expect("verify intact section")
                        .expect("known section")
                        .to_vec()
                };
                (entry.kind, entry.flags, bytes)
            })
            .collect::<Vec<_>>();
        let sections = payloads
            .iter()
            .map(|(kind, flags, bytes)| SectionInput {
                kind: *kind,
                flags: *flags,
                bytes,
            })
            .collect::<Vec<_>>();
        let corrupted = EncodedSegment::encode(
            SegmentHeaderInput {
                segment_id: header.segment_id,
                schema: DEFAULT_SCHEMA,
                docid_lo: header.docid_lo,
                docid_hi: header.docid_hi,
                doc_count: header.doc_count,
                created_unix_s: header.created_unix_s,
                engine_version: header.engine_version,
            },
            &sections,
        )
        .expect("encode checksum-consistent malformed TERMDICT fixture");
        std::fs::write(&segment_path, corrupted.as_bytes())
            .expect("publish malformed FSLX fixture");

        let mut manifest = load_manifest_pair(&lexical_dir)
            .expect("load lexical fixture manifest")
            .manifest;
        let witness = manifest
            .segments
            .iter_mut()
            .find(|segment| segment.segment_id == header.segment_id)
            .expect("manifest witness for fixture segment");
        witness.file_len = corrupted.file_len();
        witness.file_xxh3 = corrupted.file_xxh3();
        let manifest_bytes = manifest
            .to_bytes()
            .expect("encode updated fixture manifest");
        std::fs::write(lexical_dir.join("MANIFEST"), &manifest_bytes)
            .expect("publish updated fixture manifest");
        let previous_manifest = lexical_dir.join("MANIFEST.prev");
        if previous_manifest.exists() {
            std::fs::write(previous_manifest, &manifest_bytes)
                .expect("keep equal-generation fixture manifests byte-identical");
        }

        let error = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
            .await
            .expect_err("malformed TERMDICT must fail during facade open");
        assert!(
            matches!(&error, SearchError::IndexCorrupted { .. }),
            "expected IndexCorrupted, got {error:?}"
        );
        if let SearchError::IndexCorrupted { path, detail } = error {
            assert_eq!(path, segment_path);
            assert!(
                detail.contains("TERMDICT declares 1 blocks in only 4 bytes"),
                "corruption detail must identify the malformed dictionary: {detail}"
            );
        }
    });
}

/// bd-8nqz.3: lexical admission is independent of embedding outcome. A
/// document whose fast embedding fails is exactly the document that needs
/// lexical fallback, so it MUST be lexically searchable — the previous
/// contract (embed-gated staging) silently dropped it from both arms.
#[cfg(any(feature = "lexical", feature = "quill"))]
#[test]
fn lexical_admission_survives_fast_embedding_failures() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let stack = EmbedderStack::from_parts(Arc::new(SelectiveFailEmbedder), None);
        let stats = IndexBuilder::new(dir.path())
            .with_embedder_stack(stack)
            .with_batch_size(2)
            .add_document("doc-first", "first-success sentinel")
            .add_document("doc-failed", "fail-fast-embedding admitted sentinel")
            .add_document("doc-last", "last-success sentinel")
            .build(&cx)
            .await
            .unwrap();

        assert_eq!(stats.source_count, 3);
        assert_eq!(stats.doc_count, 2, "vector arm holds only embedded docs");
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.errors[0].0, "doc-failed");

        let receipt = stats.lexical.as_ref().expect("lexical arm receipt");
        assert_eq!(receipt.attempted, 3, "every valid source doc attempted");
        assert_eq!(receipt.indexed, 3);
        assert!(receipt.errors.is_empty());
        assert!(receipt.published);
        assert!(
            stats.size_bytes.lexical > 0,
            "lexical bytes must be visible"
        );
        assert_eq!(
            stats.size_bytes.total,
            stats.size_bytes.vector_fast
                + stats.size_bytes.vector_quality
                + stats.size_bytes.lexical,
        );

        // The vector arm must NOT contain the failed document...
        let index = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
        assert_eq!(index.doc_count(), 2);

        // ...but the lexical arm MUST.
        #[cfg(feature = "quill")]
        let admitted_ids = {
            let lexical = QuillIndex::open(&cx, dir.path().join("lexical"), QuillConfig::default())
                .await
                .unwrap();
            lexical
                .search_doc_ids(&cx, "admitted", 10)
                .unwrap()
                .iter()
                .map(|hit| hit.document_id.clone())
                .collect::<Vec<_>>()
        };
        #[cfg(all(feature = "lexical", not(feature = "quill")))]
        let admitted_ids = {
            let lexical = TantivyIndex::open(&dir.path().join("lexical")).unwrap();
            lexical
                .search_doc_ids(&cx, "admitted", 10)
                .unwrap()
                .into_iter()
                .map(|hit| hit.doc_id.to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(admitted_ids, vec!["doc-failed".to_owned()]);
    });
}
