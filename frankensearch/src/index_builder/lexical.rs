use std::path::Path;
use std::sync::Arc;

use asupersync::Cx;

#[cfg(any(feature = "quill", feature = "lexical-tantivy"))]
use frankensearch_core::error::SearchError;
use frankensearch_core::error::SearchResult;
use frankensearch_core::traits::LexicalRead;
#[cfg(any(feature = "quill", feature = "lexical-tantivy"))]
use frankensearch_core::types::IndexableDocument;
#[cfg(any(
    feature = "lexical-tantivy",
    all(feature = "lexical", not(feature = "quill"))
))]
use frankensearch_lexical::TantivyIndex;
#[cfg(feature = "quill")]
use frankensearch_quill::{
    BlueGreenEngine, LexicalLayout, QuillConfig, QuillIndex, RootBoundQuillSearchIndex,
    inspect_lexical_layout,
};

#[cfg(any(feature = "quill", feature = "lexical-tantivy"))]
use super::embed::build_checkpoint;
#[cfg(any(feature = "quill", feature = "lexical-tantivy"))]
use super::types::LexicalArmReceipt;
use super::types::LexicalReaderBackend;

#[cfg(feature = "quill")]
pub(super) async fn open_lexical_reader(
    cx: &Cx,
    dir: &Path,
) -> SearchResult<(Option<Arc<dyn LexicalRead>>, Option<LexicalReaderBackend>)> {
    // bd-8nqz.2: the Quill path stays bound to the lexical *root*, not the
    // engine child selected during this call. `RootBoundQuillSearchIndex`
    // validates CURRENT before its atomic refresh swap, so a future refresh
    // can follow a later Quill publication without ever mutating this root.
    let layout = inspect_lexical_layout(dir).map_err(|source| SearchError::SubsystemError {
        subsystem: "facade.lexical.layout",
        source: Box::new(source),
    })?;
    match layout {
        LexicalLayout::Empty => Ok((None, None)),
        // NOTE: deliberately two arms, not an or-pattern — `pointer` is only
        // bound in the BlueGreen variant, so `DirectQuill | BlueGreen {..} if
        // pointer.engine() == ...` is E0408 (pointer not bound in all
        // patterns) whenever this fn is compiled (feature "quill" enabled by
        // downstream workspaces such as mcp_agent_mail_rust).
        LexicalLayout::DirectQuill => {
            let index = RootBoundQuillSearchIndex::open(cx, dir, QuillConfig::default()).await?;
            Ok((Some(Arc::new(index)), Some(LexicalReaderBackend::Quill)))
        }
        LexicalLayout::BlueGreen { ref pointer, .. }
            if pointer.engine() == BlueGreenEngine::Quill =>
        {
            let index = RootBoundQuillSearchIndex::open(cx, dir, QuillConfig::default()).await?;
            Ok((Some(Arc::new(index)), Some(LexicalReaderBackend::Quill)))
        }
        #[cfg(feature = "lexical-tantivy")]
        LexicalLayout::DirectTantivy => {
            let index = TantivyIndex::open(dir)?;
            Ok((
                Some(Arc::new(index)),
                Some(LexicalReaderBackend::TantivyOracle),
            ))
        }
        #[cfg(feature = "lexical-tantivy")]
        LexicalLayout::BlueGreen { ref pointer, .. }
            if pointer.engine() == BlueGreenEngine::Tantivy =>
        {
            let index = TantivyIndex::open(&pointer.engine_dir(dir))?;
            Ok((
                Some(Arc::new(index)),
                Some(LexicalReaderBackend::TantivyOracle),
            ))
        }
        ref layout => Err(SearchError::InvalidConfig {
            field: "data_dir/lexical".to_owned(),
            value: dir.display().to_string(),
            reason: format!(
                "lexical layout is {}, which this build cannot open",
                layout.label()
            ),
        }),
    }
}

#[cfg(all(feature = "lexical", not(feature = "quill")))]
pub(super) async fn open_lexical_reader(
    _cx: &Cx,
    dir: &Path,
) -> SearchResult<(Option<Arc<dyn LexicalRead>>, Option<LexicalReaderBackend>)> {
    let index = TantivyIndex::open(dir)?;
    Ok((
        Some(Arc::new(index)),
        Some(LexicalReaderBackend::TantivyOracle),
    ))
}

#[cfg(not(any(feature = "lexical", feature = "quill")))]
pub(super) async fn open_lexical_reader(
    _cx: &Cx,
    _dir: &Path,
) -> SearchResult<(Option<Arc<dyn LexicalRead>>, Option<LexicalReaderBackend>)> {
    Ok((None, None))
}

#[cfg(feature = "quill")]
pub(super) async fn build_lexical_index(
    cx: &Cx,
    data_dir: &Path,
    documents: &[IndexableDocument],
) -> SearchResult<LexicalArmReceipt> {
    build_checkpoint(cx, "Quill lexical index initialization")?;
    // bd-8nqz.2: never initialize Quill on top of a foreign, damaged,
    // blue-green, or ambiguous layout — MANIFEST absence is NOT emptiness.
    // Empty proceeds; DirectQuill preserves the existing create-over-own
    // behavior; everything else is a typed refusal.
    match inspect_lexical_layout(data_dir).map_err(|source| SearchError::SubsystemError {
        subsystem: "facade.lexical.layout",
        source: Box::new(source),
    })? {
        LexicalLayout::Empty | LexicalLayout::DirectQuill => {}
        layout => {
            return Err(SearchError::InvalidConfig {
                field: "data_dir/lexical".to_owned(),
                value: data_dir.display().to_string(),
                reason: format!(
                    "refusing to initialize Quill over a {} lexical layout; \
                     inspect or repair the directory first",
                    layout.label()
                ),
            });
        }
    }

    let config = QuillConfig {
        bulk_load_mode: true,
        ..QuillConfig::default()
    };

    let lexical = QuillIndex::create(cx, data_dir, config).await?;
    build_checkpoint(cx, "Quill lexical document indexing")?;

    // Per-document indexing so one rejected document (duplicate id, oversized
    // field) cannot silently void the whole arm; failures land in the
    // receipt, not in an aggregate error.
    let mut indexed = 0usize;
    let mut errors: Vec<(String, String)> = Vec::new();
    let mut documents_iter = documents.iter();
    while let Some(document) = documents_iter.next() {
        build_checkpoint(cx, "Quill lexical document indexing")?;
        match lexical.index_document(cx, document).await {
            Ok(()) => indexed += 1,
            // Cancellation is a caller contract, not a document defect: abort
            // the build with the typed error instead of laundering it into
            // the receipt and spinning the recovery machinery.
            Err(error @ frankensearch_quill::QuillIndexError::Cancelled { .. }) => {
                return Err(error.into());
            }
            Err(error) => {
                tracing::warn!(
                    doc_id = %document.id,
                    error = %error,
                    "lexical indexing failed for document"
                );
                errors.push((document.id.clone(), error.to_string()));
                // A failed batch arms Quill's fail-closed retry guard: the
                // batch is ambiguous until a commit reconciles its accepted
                // prefix (Quill contract test: successful_sealed_batches_
                // compose_but_failed_batches_require_commit_retry). Reconcile
                // so one rejected document cannot void the rest of the arm.
                match lexical.commit(cx).await {
                    Ok(_) => {}
                    Err(error @ frankensearch_quill::QuillIndexError::Cancelled { .. }) => {
                        return Err(error.into());
                    }
                    Err(recovery_error) => {
                        tracing::warn!(
                            error = %recovery_error,
                            "lexical writer recovery failed; remaining documents skipped"
                        );
                        // Exact accounting: every unattempted document is recorded,
                        // never silently dropped.
                        for skipped in documents_iter {
                            errors.push((
                                skipped.id.clone(),
                                format!(
                                    "skipped: lexical writer recovery failed after prior \
                                     error: {recovery_error}"
                                ),
                            ));
                        }
                        break;
                    }
                }
            }
        }
    }
    build_checkpoint(cx, "Quill lexical publication")?;
    let _ = lexical.finish_bulk_load(cx).await?;
    build_checkpoint(cx, "Quill lexical publication complete")?;
    Ok(LexicalArmReceipt {
        backend: "quill",
        path: data_dir.to_path_buf(),
        attempted: documents.len(),
        indexed,
        errors,
        generation: None,
        published: true,
    })
}

#[cfg(feature = "lexical-tantivy")]
pub(super) fn prepare_existing_tantivy_lexical_index(
    data_dir: &Path,
) -> SearchResult<Option<TantivyIndex>> {
    let map_io_error = |source| SearchError::SubsystemError {
        subsystem: "tantivy",
        source: Box::new(source),
    };
    let entries = match std::fs::read_dir(data_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(map_io_error(error)),
    };
    let mut nonempty = false;
    for entry in entries {
        let entry = entry.map_err(map_io_error)?;
        let name = entry.file_name();
        if name == "MANIFEST" || name == "CURRENT" {
            return Err(SearchError::InvalidConfig {
                field: "data_dir/lexical".to_owned(),
                value: data_dir.display().to_string(),
                reason: "refusing Tantivy writes over Quill or blue-green lexical markers"
                    .to_owned(),
            });
        }
        nonempty = true;
    }
    if !nonempty {
        return Ok(None);
    }
    let reader = TantivyIndex::open_read_only(data_dir)?;
    validate_tantivy_adapter_schema(&reader, data_dir)?;
    drop(reader);
    // `open`, unlike `create`, cannot initialize a new format over foreign or
    // damaged content. Retain its writer through embedding: a read-only probe
    // would admit an existing writer conflict until after vector publication.
    let writer = TantivyIndex::open(data_dir)?;
    validate_tantivy_adapter_schema(&writer, data_dir)?;
    Ok(Some(writer))
}

#[cfg(feature = "lexical-tantivy")]
pub(super) fn validate_tantivy_adapter_schema(
    index: &TantivyIndex,
    data_dir: &Path,
) -> SearchResult<()> {
    use frankensearch_lexical::tantivy_crate::schema::{
        FAST, FieldType, IndexRecordOption, STORED, STRING, TextFieldIndexing, TextOptions,
    };

    // The adapter addresses these four fields by their original numeric slots.
    // Merely opening Tantivy metadata does not validate those slots or options.
    let searchable_text = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("frankensearch_default")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let expected = [
        ("id", STRING | STORED),
        ("content", searchable_text.clone()),
        ("title", searchable_text),
        ("metadata_json", TextOptions::default().set_stored()),
    ];
    let schema = index.index_handle().schema();
    let mut fields = schema.fields();
    let mismatch = || SearchError::InvalidConfig {
        field: "data_dir/lexical.schema".to_owned(),
        value: data_dir.display().to_string(),
        reason: "existing Tantivy schema is incompatible with the frankensearch adapter".to_owned(),
    };
    for (name, options) in expected {
        let Some((_, entry)) = fields.next() else {
            return Err(mismatch());
        };
        if entry.name() != name || entry.field_type() != &FieldType::Str(options) {
            return Err(mismatch());
        }
    }
    // The adapter resolves `ord` by name, and deliberately supports the older
    // schema without it. Unrelated extra fields do not change the four slots.
    if let Ok(ord) = schema.get_field("ord")
        && schema.get_field_entry(ord).field_type() != &FieldType::U64((FAST | STORED).into())
    {
        return Err(mismatch());
    }
    Ok(())
}

#[cfg(feature = "lexical-tantivy")]
pub(super) async fn build_tantivy_lexical_index(
    cx: &Cx,
    data_dir: &Path,
    documents: &[IndexableDocument],
    lexical: TantivyIndex,
) -> SearchResult<LexicalArmReceipt> {
    use frankensearch_core::traits::LexicalWrite;

    build_checkpoint(cx, "Tantivy lexical index initialization")?;
    let mut indexed = 0usize;
    let mut errors: Vec<(String, String)> = Vec::new();
    for document in documents {
        build_checkpoint(cx, "Tantivy lexical document indexing")?;
        match lexical
            .index_documents(cx, std::slice::from_ref(document))
            .await
        {
            Ok(()) => indexed += 1,
            Err(error @ SearchError::Cancelled { .. }) => return Err(error),
            Err(error) => {
                tracing::warn!(
                    doc_id = %document.id,
                    error = %error,
                    "lexical indexing failed for document"
                );
                errors.push((document.id.clone(), error.to_string()));
            }
        }
    }
    build_checkpoint(cx, "Tantivy lexical publication")?;
    lexical.commit(cx).await?;
    build_checkpoint(cx, "Tantivy lexical publication complete")?;
    Ok(LexicalArmReceipt {
        backend: "tantivy",
        path: data_dir.to_path_buf(),
        attempted: documents.len(),
        indexed,
        errors,
        generation: None,
        published: true,
    })
}
