use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use asupersync::Cx;
#[cfg(any(feature = "native", feature = "rerank"))]
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use tracing::{instrument, warn};

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{Embedder, MetricsExporter};
use frankensearch_core::types::IndexableDocument;
#[cfg(any(feature = "native", feature = "rerank"))]
use frankensearch_embed::DetectOptions;
use frankensearch_embed::auto_detect::EmbedderStack;
use frankensearch_index::{TwoTierIndex, TwoTierIndexBuilder};
#[cfg(feature = "lexical-tantivy")]
use frankensearch_lexical::TantivyIndex;

#[cfg(any(feature = "native", feature = "rerank"))]
use super::embed::detect_embedder_stack_with_pool;
use super::embed::{
    BatchDocOutcome, BuildEmbeddingIdentities, build_checkpoint, capture_build_identity,
    compute_size_breakdown, embed_build_document, embed_build_texts, export_embedding_completed,
    export_error, export_index_updated, is_terminal_build_error, validate_build_producer,
};
#[cfg(feature = "quill")]
use super::lexical::build_lexical_index;
#[cfg(feature = "lexical-tantivy")]
use super::lexical::{
    build_tantivy_lexical_index, prepare_existing_tantivy_lexical_index,
    validate_tantivy_adapter_schema,
};
#[cfg(not(any(feature = "quill", feature = "lexical-tantivy")))]
use super::types::LexicalArmReceipt;
use super::types::{IndexBuildStats, IndexProgress};

#[cfg(any(feature = "quill", feature = "lexical-tantivy"))]
#[derive(Debug, Clone, Copy)]
enum IndexLexicalBackend {
    #[cfg(feature = "quill")]
    Quill,
    #[cfg(feature = "lexical-tantivy")]
    Tantivy,
}

/// Fluent builder for creating frankensearch indexes.
///
/// Handles embedder auto-detection, vector index creation, batch embedding,
/// and error aggregation behind a simple API.
pub struct IndexBuilder {
    data_dir: PathBuf,
    config: TwoTierConfig,
    documents: Vec<IndexableDocument>,
    embedder_stack: Option<EmbedderStack>,
    #[cfg(any(feature = "quill", feature = "lexical-tantivy"))]
    lexical_backend: Option<IndexLexicalBackend>,
    #[cfg(any(feature = "native", feature = "rerank"))]
    native_quality: Option<(PathBuf, BlockingPoolHandle)>,
    pub(super) batch_size: usize,
    on_progress: Option<Box<dyn FnMut(IndexProgress) + Send>>,
    #[cfg(all(
        any(feature = "lexical", feature = "quill"),
        feature = "bench-internals"
    ))]
    clone_lexical_staging_for_benchmark: bool,
}

impl IndexBuilder {
    /// Create a new builder targeting the given directory.
    #[must_use]
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            config: TwoTierConfig::default(),
            documents: Vec::new(),
            embedder_stack: None,
            #[cfg(feature = "quill")]
            lexical_backend: Some(IndexLexicalBackend::Quill),
            #[cfg(all(feature = "lexical-tantivy", not(feature = "quill")))]
            lexical_backend: None,
            #[cfg(any(feature = "native", feature = "rerank"))]
            native_quality: None,
            batch_size: 32,
            on_progress: None,
            #[cfg(all(
                any(feature = "lexical", feature = "quill"),
                feature = "bench-internals"
            ))]
            clone_lexical_staging_for_benchmark: false,
        }
    }

    /// Override the search/index configuration.
    #[must_use]
    pub fn with_config(mut self, config: TwoTierConfig) -> Self {
        self.config = config;
        self
    }

    /// Use a specific embedder stack instead of auto-detecting.
    ///
    /// Identity-aware producers are checked on every bound embedding response
    /// and again before publication. Existing custom producers whose default
    /// `identity()` reports `embedder.identity` remain on the explicit legacy
    /// v1 build path: their raw vectors receive no identity or attestation.
    /// Any other identity failure aborts the build; a declared identity can
    /// never fall back to raw inference or be silently replaced mid-build.
    #[must_use]
    pub fn with_embedder_stack(mut self, stack: EmbedderStack) -> Self {
        self.embedder_stack = Some(stack);
        self
    }

    /// Write the lexical arm in Tantivy format for an explicitly selected
    /// Tantivy consumer or comparator.
    ///
    /// This requires `lexical-tantivy`. Enabling that feature alone does not
    /// select a writer: builds still default to Quill when `quill` is enabled,
    /// and otherwise omit the lexical arm. This method overrides that choice
    /// for this build only; it is never an automatic fallback from Quill.
    ///
    /// Open the resulting `lexical` directory with [`TantivyIndex::open_read_only`]
    /// or [`TantivyIndex::open`]. This does not change [`super::open_hybrid`]'s feature
    /// requirements or migrate an existing lexical directory to a new format.
    /// A nonempty lexical directory must already be readable as Tantivy; the
    /// build checks this and acquires its writer before creating vector files
    /// or embedding documents. Conflicting Quill or blue-green markers refuse.
    /// The adapter's field positions and options must match; older indexes
    /// without the optional insertion-ordinal field remain supported.
    #[cfg(feature = "lexical-tantivy")]
    #[must_use]
    pub fn with_tantivy_lexical(mut self) -> Self {
        self.lexical_backend = Some(IndexLexicalBackend::Tantivy);
        self
    }

    /// Prefer verified native F32 quality when auto-detecting this build's stack.
    ///
    /// Model loading runs on `pool`; an absent or invalid native installation
    /// uses the existing quality detector. An explicit [`Self::with_embedder_stack`]
    /// takes precedence. Reuse the selected stack from
    /// [`detect_embedder_stack_with_pool`] when the same host also searches the
    /// index. The caller owns and drains the pool.
    #[cfg(any(feature = "native", feature = "rerank"))]
    #[must_use]
    pub fn with_native_quality(
        mut self,
        model_root: impl Into<PathBuf>,
        pool: BlockingPoolHandle,
    ) -> Self {
        self.native_quality = Some((model_root.into(), pool));
        self
    }

    /// Set the batch size for embedding operations. Default: 32.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Set a progress callback.
    #[must_use]
    pub fn with_progress(mut self, callback: impl FnMut(IndexProgress) + Send + 'static) -> Self {
        self.on_progress = Some(Box::new(callback));
        self
    }

    /// Retain the former deep-clone staging path for same-binary performance comparisons.
    #[cfg(all(
        any(feature = "lexical", feature = "quill"),
        feature = "bench-internals"
    ))]
    #[doc(hidden)]
    #[must_use]
    pub fn with_clone_lexical_staging_for_benchmark(mut self) -> Self {
        self.clone_lexical_staging_for_benchmark = true;
        self
    }

    /// Add a single document to be indexed.
    #[must_use]
    pub fn add_document(mut self, id: impl Into<String>, content: impl Into<String>) -> Self {
        self.documents
            .push(IndexableDocument::new(id.into(), content.into()));
        self
    }

    /// Add a document with title.
    #[must_use]
    pub fn add_document_with_title(
        mut self,
        id: impl Into<String>,
        content: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        self.documents
            .push(IndexableDocument::new(id.into(), content.into()).with_title(title.into()));
        self
    }

    /// Add multiple documents.
    #[must_use]
    pub fn add_documents(mut self, docs: impl IntoIterator<Item = IndexableDocument>) -> Self {
        self.documents.extend(docs);
        self
    }

    /// Build the index, embedding all documents and writing FSVI files.
    ///
    /// Returns build statistics including per-document errors.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if no documents were added.
    /// Returns `SearchError::Io` if the data directory cannot be created.
    /// Individual document embedding failures are collected in `IndexBuildStats.errors`
    /// rather than aborting the build. Structured cancellation is never collected as a
    /// document failure: it is returned immediately as [`SearchError::Cancelled`].
    #[allow(clippy::too_many_lines)]
    #[instrument(skip_all, fields(doc_count = self.documents.len(), data_dir = %self.data_dir.display()))]
    pub async fn build(mut self, cx: &Cx) -> SearchResult<IndexBuildStats> {
        let start = Instant::now();
        let metrics_exporter = self.config.metrics_exporter.clone();

        build_checkpoint(cx, "index build start")?;

        if self.documents.is_empty() {
            let error = SearchError::InvalidConfig {
                field: "documents".to_owned(),
                value: "0".to_owned(),
                reason: "at least one document is required".to_owned(),
            };
            export_error(metrics_exporter.as_ref(), &error);
            return Err(error);
        }

        #[cfg(feature = "lexical-tantivy")]
        let tantivy_lexical = if matches!(self.lexical_backend, Some(IndexLexicalBackend::Tantivy))
        {
            match prepare_existing_tantivy_lexical_index(&self.data_dir.join("lexical")) {
                Ok(index) => index,
                Err(error) => {
                    export_error(metrics_exporter.as_ref(), &error);
                    return Err(error);
                }
            }
        } else {
            None
        };

        // Resolve embedder stack.
        let stack = match self.embedder_stack.take() {
            Some(stack) => stack,
            #[cfg(any(feature = "native", feature = "rerank"))]
            None if self.native_quality.is_some() => {
                let (root, pool) = self
                    .native_quality
                    .take()
                    .expect("native quality configured");
                match detect_embedder_stack_with_pool(
                    cx,
                    Some(&root),
                    &DetectOptions::default(),
                    pool,
                )
                .await
                {
                    Ok(stack) => stack,
                    Err(error) => {
                        export_error(metrics_exporter.as_ref(), &error);
                        return Err(error);
                    }
                }
            }
            None => match EmbedderStack::auto_detect_semantic_with(Some(&self.data_dir)) {
                Ok(stack) => stack,
                Err(error) => {
                    export_error(metrics_exporter.as_ref(), &error);
                    return Err(error);
                }
            },
        };
        build_checkpoint(cx, "index builder initialization")?;

        // A degraded stack here is not a transient runtime condition: the
        // vectors written below carry this embedder's identity, so the
        // generation is permanently degraded and a later model install cannot
        // repair it — only a compatible rebuild can. Auto-detect HashOnly is
        // refused above via `auto_detect_semantic_with`. Hash remains
        // reachable only through an explicit stack (tests / control policy).
        let embedder_availability = stack.availability();
        if embedder_availability.is_degraded() {
            tracing::warn!(
                availability = %embedder_availability,
                data_dir = %self.data_dir.display(),
                fast_embedder = %stack.fast().id(),
                detail = stack.degradation_message().unwrap_or_default(),
                "building index with a DEGRADED embedder stack; this generation is written with \
                 that embedder's vector identity and installing a model later will NOT repair it \
                 — a compatible rebuild is required",
            );
        }

        let fast_embedder = stack.fast_arc();
        let quality_embedder = stack.quality_arc().and_then(|quality| {
            if fast_embedder.is_semantic() && quality.is_semantic() {
                Some(quality)
            } else {
                warn!(
                    fast_embedder = %fast_embedder.id(),
                    quality_embedder = %quality.id(),
                    "dropping non-semantic quality embedder; a hash control stack does not write a quality generation"
                );
                None
            }
        });

        // Capture each producer independently before allocating a writer.
        // These exact values bind both the persisted header and every accepted
        // response; consulting only identity() before inference would let a
        // provider return foreign vectors under an unchanged advertised label.
        let identities = BuildEmbeddingIdentities {
            fast: capture_build_identity(fast_embedder.as_ref(), "fast")
                .inspect_err(|error| export_error(metrics_exporter.as_ref(), error))?,
            quality: quality_embedder
                .as_ref()
                .map(|embedder| capture_build_identity(embedder.as_ref(), "quality"))
                .transpose()
                .inspect_err(|error| export_error(metrics_exporter.as_ref(), error))?
                .flatten(),
        };

        // Create index builder.
        let mut index_builder = match TwoTierIndex::create(&self.data_dir, self.config) {
            Ok(builder) => builder,
            Err(error) => {
                export_error(metrics_exporter.as_ref(), &error);
                return Err(error);
            }
        };
        index_builder.set_fast_embedder_id(fast_embedder.id());
        if let Some(ref qe) = quality_embedder {
            index_builder.set_quality_embedder_id(qe.id());
        }
        if let Some(identity) = &identities.fast {
            index_builder
                .set_fast_identity(identity)
                .inspect_err(|error| export_error(metrics_exporter.as_ref(), error))?;
        }
        if let Some(identity) = &identities.quality {
            index_builder
                .set_quality_identity(identity)
                .inspect_err(|error| export_error(metrics_exporter.as_ref(), error))?;
        }

        let total = self.documents.len();
        let mut errors = Vec::new();
        let mut doc_count = 0usize;
        let mut quality_indexed = 0usize;
        let mut quality_errors: Vec<(String, String)> = Vec::new();
        let mut embed_ms = 0.0f64;
        #[cfg(any(feature = "lexical", feature = "quill"))]
        let mut lexical_docs = Vec::with_capacity(total);

        // Keep the old borrowed loop available only for the same-binary benchmark arm. This is the
        // exact former residency behavior: all originals stay in `self.documents` while successful
        // documents are deep-cloned into lexical staging.
        #[cfg(all(
            any(feature = "lexical", feature = "quill"),
            feature = "bench-internals"
        ))]
        if self.clone_lexical_staging_for_benchmark {
            for (batch_idx, batch) in self.documents.chunks(self.batch_size).enumerate() {
                let batch_start = Instant::now();
                for doc in batch {
                    match Self::embed_and_add(
                        cx,
                        &fast_embedder,
                        quality_embedder.as_deref(),
                        &identities,
                        &mut index_builder,
                        doc,
                        metrics_exporter.as_ref(),
                    )
                    .await
                    {
                        // Benchmark arm: pins the exact former residency AND
                        // former admission behavior (embed-gated lexical
                        // staging); quality receipts are not collected here.
                        Ok(_) => {
                            doc_count += 1;
                            lexical_docs.push(doc.clone());
                        }
                        Err(
                            error @ (SearchError::Cancelled { .. }
                            | SearchError::UnverifiableRemoteSpace { .. }),
                        ) => return Err(error),
                        Err(err) => {
                            tracing::warn!(doc_id = %doc.id, error = %err, "failed to embed document");
                            errors.push((doc.id.clone(), err.to_string()));
                        }
                    }
                }
                embed_ms += batch_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(ref mut callback) = self.on_progress {
                    let completed = (batch_idx + 1).saturating_mul(self.batch_size);
                    callback(IndexProgress {
                        completed: completed.min(total),
                        total,
                        phase: "embedding",
                    });
                }
            }
        } else {
            // `build` owns the input documents, so move successful values into lexical staging.
            let mut documents = std::mem::take(&mut self.documents).into_iter();
            let batch_count = total.div_ceil(self.batch_size);
            for batch_idx in 0..batch_count {
                let batch_start = Instant::now();
                let batch = documents.by_ref().take(self.batch_size).collect::<Vec<_>>();
                let outcomes = Self::embed_and_add_batch(
                    cx,
                    &fast_embedder,
                    quality_embedder.as_deref(),
                    &identities,
                    &mut index_builder,
                    &batch,
                    metrics_exporter.as_ref(),
                )
                .await?;
                for (doc, outcome) in batch.into_iter().zip(outcomes) {
                    match outcome {
                        BatchDocOutcome::Added(quality_error) => {
                            doc_count += 1;
                            if let Some(message) = quality_error {
                                quality_errors.push((doc.id.clone(), message));
                            } else if quality_embedder.is_some() {
                                quality_indexed += 1;
                            }
                        }
                        BatchDocOutcome::Failed(message) => {
                            tracing::warn!(doc_id = %doc.id, error = %message, "failed to embed document");
                            errors.push((doc.id.clone(), message));
                        }
                    }
                    // bd-8nqz.3: lexical admission is independent of embedding
                    // outcome — the documents most in need of lexical
                    // fallback must stay lexically searchable.
                    lexical_docs.push(doc);
                }
                embed_ms += batch_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(ref mut callback) = self.on_progress {
                    let completed = (batch_idx + 1).saturating_mul(self.batch_size);
                    callback(IndexProgress {
                        completed: completed.min(total),
                        total,
                        phase: "embedding",
                    });
                }
            }
        }

        #[cfg(all(
            any(feature = "lexical", feature = "quill"),
            not(feature = "bench-internals")
        ))]
        {
            // `build` owns the input documents, so move successful values into lexical staging.
            let mut documents = std::mem::take(&mut self.documents).into_iter();
            let batch_count = total.div_ceil(self.batch_size);
            for batch_idx in 0..batch_count {
                let batch_start = Instant::now();
                let batch = documents.by_ref().take(self.batch_size).collect::<Vec<_>>();
                let outcomes = Self::embed_and_add_batch(
                    cx,
                    &fast_embedder,
                    quality_embedder.as_deref(),
                    &identities,
                    &mut index_builder,
                    &batch,
                    metrics_exporter.as_ref(),
                )
                .await?;
                for (doc, outcome) in batch.into_iter().zip(outcomes) {
                    match outcome {
                        BatchDocOutcome::Added(quality_error) => {
                            doc_count += 1;
                            if let Some(message) = quality_error {
                                quality_errors.push((doc.id.clone(), message));
                            } else if quality_embedder.is_some() {
                                quality_indexed += 1;
                            }
                        }
                        BatchDocOutcome::Failed(message) => {
                            tracing::warn!(doc_id = %doc.id, error = %message, "failed to embed document");
                            errors.push((doc.id.clone(), message));
                        }
                    }
                    // bd-8nqz.3: lexical admission is independent of embedding
                    // outcome — the documents most in need of lexical
                    // fallback must stay lexically searchable.
                    lexical_docs.push(doc);
                }
                embed_ms += batch_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(ref mut callback) = self.on_progress {
                    let completed = (batch_idx + 1).saturating_mul(self.batch_size);
                    callback(IndexProgress {
                        completed: completed.min(total),
                        total,
                        phase: "embedding",
                    });
                }
            }
        }

        // Without lexical indexing there is no staging clone to remove, so retain the former path
        // and its metrics/drop timing exactly.
        #[cfg(not(any(feature = "lexical", feature = "quill")))]
        for (batch_idx, batch) in self.documents.chunks(self.batch_size).enumerate() {
            let batch_start = Instant::now();
            let outcomes = Self::embed_and_add_batch(
                cx,
                &fast_embedder,
                quality_embedder.as_deref(),
                &identities,
                &mut index_builder,
                batch,
                metrics_exporter.as_ref(),
            )
            .await?;
            for (doc, outcome) in batch.iter().zip(outcomes) {
                match outcome {
                    BatchDocOutcome::Added(quality_error) => {
                        doc_count += 1;
                        if let Some(message) = quality_error {
                            quality_errors.push((doc.id.clone(), message));
                        } else if quality_embedder.is_some() {
                            quality_indexed += 1;
                        }
                    }
                    BatchDocOutcome::Failed(message) => {
                        tracing::warn!(doc_id = %doc.id, error = %message, "failed to embed document");
                        errors.push((doc.id.clone(), message));
                    }
                }
            }
            embed_ms += batch_start.elapsed().as_secs_f64() * 1_000.0;
            if let Some(ref mut callback) = self.on_progress {
                let completed = (batch_idx + 1).saturating_mul(self.batch_size);
                callback(IndexProgress {
                    completed: completed.min(total),
                    total,
                    phase: "embedding",
                });
            }
        }

        // Finalize index files.
        build_checkpoint(cx, "vector index finalize")?;
        if doc_count == 0 {
            let error = SearchError::InvalidConfig {
                field: "documents".to_owned(),
                value: format!("{total}"),
                reason: format!("all {total} documents failed to embed"),
            };
            export_error(metrics_exporter.as_ref(), &error);
            return Err(error);
        }

        // Admit a new writer only after embedding/cancellation succeeds, but
        // still before replacing vector generations. Recheck an initially
        // empty path in case another builder populated it while we embedded.
        #[cfg(feature = "lexical-tantivy")]
        let tantivy_lexical = if matches!(self.lexical_backend, Some(IndexLexicalBackend::Tantivy))
            && tantivy_lexical.is_none()
        {
            let lexical_path = self.data_dir.join("lexical");
            let prepared =
                prepare_existing_tantivy_lexical_index(&lexical_path).and_then(|existing| {
                    match existing {
                        Some(index) => Ok(index),
                        None => {
                            let index = TantivyIndex::create(&lexical_path)?;
                            validate_tantivy_adapter_schema(&index, &lexical_path)?;
                            Ok(index)
                        }
                    }
                });
            match prepared {
                Ok(index) => Some(index),
                Err(error) => {
                    export_error(metrics_exporter.as_ref(), &error);
                    return Err(error);
                }
            }
        } else {
            tantivy_lexical
        };
        #[cfg(feature = "lexical-tantivy")]
        if tantivy_lexical.is_some() {
            build_checkpoint(cx, "vector index finalize")?;
        }

        validate_build_producer(fast_embedder.as_ref(), identities.fast.as_ref(), "fast")
            .inspect_err(|error| export_error(metrics_exporter.as_ref(), error))?;
        if let Some(embedder) = quality_embedder.as_ref() {
            validate_build_producer(embedder.as_ref(), identities.quality.as_ref(), "quality")
                .inspect_err(|error| export_error(metrics_exporter.as_ref(), error))?;
        }
        build_checkpoint(cx, "vector index producer admission")?;
        let _index = match index_builder.finish() {
            Ok(index) => index,
            Err(error) => {
                export_error(metrics_exporter.as_ref(), &error);
                return Err(error);
            }
        };
        build_checkpoint(cx, "vector index finalized")?;

        #[cfg(not(any(feature = "quill", feature = "lexical-tantivy")))]
        let (lexical_receipt, lexical_ms): (Option<LexicalArmReceipt>, f64) = (None, 0.0);
        // The Tantivy-only feature lane retains the existing borrowed embedding
        // loop and its input ownership. An explicit writer can use those same
        // documents without a staging clone, including failed embeddings.
        #[cfg(all(feature = "lexical-tantivy", not(feature = "quill")))]
        let lexical_docs = self.documents.as_slice();
        #[cfg(feature = "quill")]
        let lexical_docs = lexical_docs.as_slice();
        #[cfg(any(feature = "quill", feature = "lexical-tantivy"))]
        let (lexical_receipt, lexical_ms) = if let Some(backend) =
            self.lexical_backend.filter(|_| !lexical_docs.is_empty())
        {
            build_checkpoint(cx, "lexical index build")?;
            let lexical_path = self.data_dir.join("lexical");
            let lexical_start = Instant::now();
            let result = match backend {
                #[cfg(feature = "quill")]
                IndexLexicalBackend::Quill => {
                    Box::pin(build_lexical_index(cx, &lexical_path, lexical_docs)).await
                }
                #[cfg(feature = "lexical-tantivy")]
                IndexLexicalBackend::Tantivy => {
                    Box::pin(build_tantivy_lexical_index(
                        cx,
                        &lexical_path,
                        lexical_docs,
                        tantivy_lexical.expect("Tantivy writer admitted before vector publication"),
                    ))
                    .await
                }
            };
            match result {
                Ok(receipt) => (
                    Some(receipt),
                    lexical_start.elapsed().as_secs_f64() * 1000.0,
                ),
                // Publication failure (create/seal/commit) stays fatal: a
                // half-written lexical index is worse than an absent arm.
                // Per-document indexing errors are NOT fatal; they are
                // reported in the receipt.
                Err(error) => {
                    export_error(metrics_exporter.as_ref(), &error);
                    return Err(error);
                }
            }
        } else {
            (None, 0.0)
        };

        build_checkpoint(cx, "index build completion")?;
        let has_quality = quality_embedder.is_some();
        let size_bytes = compute_size_breakdown(&self.data_dir);
        export_index_updated(
            metrics_exporter.as_ref(),
            doc_count,
            size_bytes.total,
            doc_count,
        );

        tracing::info!(
            doc_count,
            error_count = errors.len(),
            has_quality,
            total_ms = start.elapsed().as_secs_f64() * 1000.0,
            "index build complete"
        );

        let stats = IndexBuildStats {
            source_count: total,
            doc_count,
            error_count: errors.len(),
            errors,
            quality_indexed,
            quality_errors,
            lexical: lexical_receipt,
            size_bytes,
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            embed_ms,
            lexical_ms,
            has_quality_index: has_quality,
            embedder_availability,
        };

        Ok(stats)
    }

    /// Embed a single document and add it to the index builder.
    ///
    /// Returns `Ok(None)` when every attempted arm succeeded, and
    /// `Ok(Some(message))` when the fast tier succeeded but the quality
    /// embedding failed (fast-only degradation for this document). A fast
    /// embedding failure is a hard `Err`: the document enters no vector tier.
    async fn embed_and_add(
        cx: &Cx,
        fast_embedder: &Arc<dyn Embedder>,
        quality_embedder: Option<&dyn Embedder>,
        identities: &BuildEmbeddingIdentities,
        builder: &mut TwoTierIndexBuilder,
        doc: &IndexableDocument,
        metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    ) -> SearchResult<Option<String>> {
        let text = doc.content.as_str();

        // Fast embedding (required).
        build_checkpoint(cx, "fast document embedding")?;
        let fast_start = Instant::now();
        let fast_vec = match embed_build_document(
            cx,
            fast_embedder.as_ref(),
            identities.fast.as_ref(),
            "fast",
            text,
        )
        .await
        {
            Ok(fast_vec) => {
                let duration_ms = fast_start.elapsed().as_secs_f64() * 1000.0;
                export_embedding_completed(
                    metrics_exporter,
                    fast_embedder.as_ref(),
                    1,
                    duration_ms,
                );
                fast_vec
            }
            Err(error) => {
                export_error(metrics_exporter, &error);
                return Err(error);
            }
        };
        build_checkpoint(cx, "fast document embedding completion")?;
        builder.add_fast_record(&doc.id, &fast_vec)?;

        // Quality embedding (optional).
        if let Some(qe) = quality_embedder {
            build_checkpoint(cx, "quality document embedding")?;
            let quality_start = Instant::now();
            match embed_build_document(cx, qe, identities.quality.as_ref(), "quality", text).await {
                Ok(quality_vec) => {
                    build_checkpoint(cx, "quality document embedding completion")?;
                    let duration_ms = quality_start.elapsed().as_secs_f64() * 1000.0;
                    export_embedding_completed(metrics_exporter, qe, 1, duration_ms);
                    builder.add_quality_record(&doc.id, &quality_vec)?;
                }
                Err(
                    error @ (SearchError::Cancelled { .. }
                    | SearchError::UnverifiableRemoteSpace { .. }),
                ) => {
                    export_error(metrics_exporter, &error);
                    return Err(error);
                }
                Err(error) => {
                    export_error(metrics_exporter, &error);
                    tracing::debug!(
                        doc_id = %doc.id,
                        error = %error,
                        "quality embedding failed, fast-only for this document"
                    );
                    return Ok(Some(error.to_string()));
                }
            }
        }

        Ok(None)
    }

    /// Embed one `batch_size` group and add it to the index builder, with the
    /// per-document outcomes of [`Self::embed_and_add`] (GH #57).
    ///
    /// A tier whose producer batches natively ([`Embedder::bound_batch_is_native`])
    /// is embedded as one bound batch, the quality tier only for documents
    /// whose fast embedding succeeded; see [`embed_build_texts`]. When neither
    /// tier batches, every document takes the per-document path unchanged.
    /// A terminal failure (cancellation, identity refusal) is returned as `Err`.
    async fn embed_and_add_batch(
        cx: &Cx,
        fast_embedder: &Arc<dyn Embedder>,
        quality_embedder: Option<&dyn Embedder>,
        identities: &BuildEmbeddingIdentities,
        builder: &mut TwoTierIndexBuilder,
        docs: &[IndexableDocument],
        metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    ) -> SearchResult<Vec<BatchDocOutcome>> {
        let batches_fast = identities.fast.is_some() && fast_embedder.bound_batch_is_native();
        let batches_quality = quality_embedder.is_some_and(|embedder| {
            identities.quality.is_some() && embedder.bound_batch_is_native()
        });
        if !batches_fast && !batches_quality {
            let mut outcomes = Vec::with_capacity(docs.len());
            for doc in docs {
                match Self::embed_and_add(
                    cx,
                    fast_embedder,
                    quality_embedder,
                    identities,
                    builder,
                    doc,
                    metrics_exporter,
                )
                .await
                {
                    Ok(quality_error) => outcomes.push(BatchDocOutcome::Added(quality_error)),
                    Err(error) if is_terminal_build_error(&error) => return Err(error),
                    Err(error) => outcomes.push(BatchDocOutcome::Failed(error.to_string())),
                }
            }
            return Ok(outcomes);
        }

        let texts = docs
            .iter()
            .map(|doc| doc.content.as_str())
            .collect::<Vec<_>>();
        build_checkpoint(cx, "fast document embedding")?;
        let fast = embed_build_texts(
            cx,
            fast_embedder.as_ref(),
            identities.fast.as_ref(),
            "fast",
            &texts,
            metrics_exporter,
        )
        .await
        .inspect_err(|error| export_error(metrics_exporter, error))?;
        build_checkpoint(cx, "fast document embedding completion")?;
        let mut outcomes = Vec::with_capacity(docs.len());
        let mut embedded = Vec::with_capacity(docs.len());
        for (index, (doc, result)) in docs.iter().zip(fast).enumerate() {
            let added = match result {
                Ok(values) => builder.add_fast_record(&doc.id, &values),
                Err(error) => {
                    export_error(metrics_exporter, &error);
                    Err(error)
                }
            };
            match added {
                Ok(()) => {
                    outcomes.push(BatchDocOutcome::Added(None));
                    embedded.push(index);
                }
                Err(error) if is_terminal_build_error(&error) => return Err(error),
                Err(error) => outcomes.push(BatchDocOutcome::Failed(error.to_string())),
            }
        }

        let Some(quality) = quality_embedder else {
            return Ok(outcomes);
        };
        build_checkpoint(cx, "quality document embedding")?;
        let quality_texts = embedded
            .iter()
            .map(|&index| texts[index])
            .collect::<Vec<_>>();
        let results = embed_build_texts(
            cx,
            quality,
            identities.quality.as_ref(),
            "quality",
            &quality_texts,
            metrics_exporter,
        )
        .await
        .inspect_err(|error| export_error(metrics_exporter, error))?;
        build_checkpoint(cx, "quality document embedding completion")?;
        for (&index, result) in embedded.iter().zip(results) {
            match result {
                Ok(values) => {
                    // As on the per-document path, a quality record the
                    // builder refuses fails the whole document.
                    if let Err(error) = builder.add_quality_record(&docs[index].id, &values) {
                        if is_terminal_build_error(&error) {
                            return Err(error);
                        }
                        outcomes[index] = BatchDocOutcome::Failed(error.to_string());
                    }
                }
                Err(error) => {
                    export_error(metrics_exporter, &error);
                    tracing::debug!(
                        doc_id = %docs[index].id,
                        error = %error,
                        "quality embedding failed, fast-only for this document"
                    );
                    outcomes[index] = BatchDocOutcome::Added(Some(error.to_string()));
                }
            }
        }
        Ok(outcomes)
    }
}

impl std::fmt::Debug for IndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexBuilder")
            .field("data_dir", &self.data_dir)
            .field("doc_count", &self.documents.len())
            .field("batch_size", &self.batch_size)
            .field("has_embedder_stack", &self.embedder_stack.is_some())
            .finish_non_exhaustive()
    }
}
