use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use asupersync::Cx;
#[cfg(any(feature = "native", feature = "rerank"))]
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
#[cfg(any(feature = "native", feature = "rerank"))]
use tracing::warn;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{Embedder, MetricsExporter};
use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics};
#[cfg(any(feature = "native", feature = "rerank"))]
use frankensearch_embed::DetectOptions;
#[cfg(any(feature = "native", feature = "rerank"))]
use frankensearch_embed::auto_detect::EmbedderStack;
use frankensearch_index::{
    VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
};

use super::types::IndexSizeBreakdown;

/// Detect a semantic stack, preferring verified native F32 `MiniLM` quality.
///
/// Detection and eager model loading run on the caller's blocking pool.
/// Native artifacts live in `all-MiniLM-L6-v2-native` below the supplied model
/// root, or below the normal model cache when no root is supplied. Potion and
/// ONNX keep their standard sibling directories: their frozen artifact sets
/// must not be mixed into the native directory. If native loading fails, the
/// existing quality detector selects an available ONNX/remote model under
/// `options`. No model is downloaded by the native detector.
///
/// The returned stack keeps the selected backend's complete identity. Use that
/// same stack to build and search an index; selecting another backend does not
/// make an existing index compatible. This is load-time selection, not an
/// inference-time backend switch. The caller must drain the pool at shutdown.
///
/// # Errors
///
/// Returns policy, semantic-fast-model, cancellation, or worker errors. Missing
/// quality models retain the existing fast-only behavior.
#[cfg(any(feature = "native", feature = "rerank"))]
pub async fn detect_embedder_stack_with_pool(
    cx: &Cx,
    model_root: Option<&Path>,
    options: &DetectOptions,
    pool: BlockingPoolHandle,
) -> SearchResult<EmbedderStack> {
    build_checkpoint(cx, "embedder detection start")?;
    if pool.is_shutdown() {
        return Err(SearchError::EmbeddingFailed {
            model: "stack-detection".to_owned(),
            source: "cannot admit model loading worker: caller-owned blocking pool is shut down"
                .into(),
        });
    }
    let model_root = model_root.map(Path::to_path_buf);
    let options = *options;
    let request_cx = cx.clone();
    let worker_cx = cx.clone().with_blocking_pool_handle(Some(pool.clone()));
    let mut worker = worker_cx
        .spawn_blocking(move |child: Cx| -> SearchResult<EmbedderStack> {
            let checkpoint = |phase| {
                build_checkpoint(&request_cx, phase)?;
                build_checkpoint(&child, phase)
            };
            checkpoint("fast model detection")?;
            // Pool shutdown can race the check above. The runtime recovers a
            // rejected submission by executing it on its async wrapper task.
            // Refuse that fallback before any model discovery or file access.
            if Cx::is_active() {
                return Err(SearchError::EmbeddingFailed {
                    model: "stack-detection".to_owned(),
                    source: "blocking pool dispatch fell back to an async executor".into(),
                });
            }
            // Resolve caller policy before considering the local native model.
            // Detecting only fast avoids loading an ONNX session we would discard.
            let fast = EmbedderStack::auto_detect_fast_semantic_with_options(
                model_root.as_deref(),
                &options,
            );
            checkpoint("native model detection")?;
            let fast = fast?;
            let path = model_root
                .clone()
                .unwrap_or_else(frankensearch_embed::model_cache::resolve_cache_root)
                .join("all-MiniLM-L6-v2-native");
            if path.join("model.safetensors").is_file() {
                let native = frankensearch_rerank::NativeEmbedder::load_model(
                    &path,
                    frankensearch_rerank::NativeEmbeddingModel::AllMiniLmL6V2F32,
                );
                checkpoint("native model loaded")?;
                match native {
                    Ok(native) => {
                        let quality: Arc<dyn Embedder> =
                            Arc::new(native.with_blocking_pool(pool));
                        tracing::info!(
                            quality_embedder = quality.id(),
                            "selected verified native F32 quality model"
                        );
                        return Ok(EmbedderStack::from_parts(fast.fast_arc(), Some(quality)));
                    }
                    Err(error) => {
                        warn!(%error, "native quality model did not load; checking the existing quality backend");
                    }
                }
            }
            checkpoint("fallback quality detection")?;
            let quality = EmbedderStack::auto_detect_quality_with_options(
                model_root.as_deref(),
                &options,
            );
            checkpoint("fallback quality loaded")?;
            let quality = quality?;
            tracing::info!(
                quality_embedder = quality.as_ref().map(|embedder| embedder.id()),
                "native quality unavailable; selected existing quality backend"
            );
            Ok(EmbedderStack::from_parts(fast.fast_arc(), quality))
        })
        .map_err(|error| SearchError::EmbeddingFailed {
            model: "stack-detection".to_owned(),
            source: format!("cannot admit model loading worker: {error}").into(),
        })?;
    let result = worker.join(cx).await;
    // Joining does not inspect its Cx. Cancellation must take precedence over
    // both a successful stack and a backend/worker error already produced.
    build_checkpoint(cx, "embedder detection complete")?;
    result.map_err(|error| match error {
        asupersync::runtime::JoinError::Cancelled(_) => SearchError::Cancelled {
            phase: "embedder detection".to_owned(),
            reason: "model loading worker cancelled".to_owned(),
        },
        error => SearchError::EmbeddingFailed {
            model: "stack-detection".to_owned(),
            source: format!("model loading worker failed: {error}").into(),
        },
    })?
}

/// What one document's embedding produced in a batch build.
pub(super) enum BatchDocOutcome {
    /// Added to every attempted tier; `Some(message)` when only quality failed
    /// and the document is fast-only.
    Added(Option<String>),
    /// The fast embedding failed: the document enters no vector tier.
    Failed(String),
}

pub(super) struct BuildEmbeddingIdentities {
    pub(super) fast: Option<EmbeddingIdentityBundleV1>,
    pub(super) quality: Option<EmbeddingIdentityBundleV1>,
}

fn missing_build_identity(error: &SearchError) -> bool {
    matches!(error, SearchError::InvalidConfig { field, .. } if field == "embedder.identity")
}

pub(super) fn capture_build_identity(
    embedder: &dyn Embedder,
    tier: &'static str,
) -> SearchResult<Option<EmbeddingIdentityBundleV1>> {
    match embedder.identity() {
        Ok(identity) => {
            identity.validate()?;
            if usize::try_from(identity.space.dimension).ok() != Some(embedder.dimension()) {
                return Err(SearchError::InvalidConfig {
                    field: format!("{tier}_identity.space.dimension"),
                    value: identity.space.dimension.to_string(),
                    reason: "declared identity does not describe the producer's vector dimension"
                        .to_owned(),
                });
            }
            if identity.storage.quantization != QuantizationFormat::F32
                || !identity.storage.format.starts_with("in-memory-")
                || !matches!(
                    identity.storage.endianness.as_str(),
                    "native-f32-values" | "native-test-only"
                )
            {
                return Err(SearchError::InvalidConfig {
                    field: format!("{tier}_identity.storage"),
                    value: "output_contract".to_owned(),
                    reason: "embedding producers must declare an in-memory f32 output contract"
                        .to_owned(),
                });
            }
            Ok(Some(identity.clone()))
        }
        // Preserve only the pre-existing explicit legacy custom-embedder path.
        // This does not recover from a malformed or rejected declared identity,
        // and none of its raw vectors acquire typed identity or attestation.
        Err(error) if missing_build_identity(&error) => {
            tracing::debug!(
                tier,
                "building explicit legacy-unidentified custom embeddings"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

pub(super) fn validate_build_producer(
    embedder: &dyn Embedder,
    expected: Option<&EmbeddingIdentityBundleV1>,
    tier: &'static str,
) -> SearchResult<()> {
    let unchanged = match (expected, embedder.identity()) {
        (Some(expected), Ok(actual)) => {
            actual == expected
                && usize::try_from(expected.space.dimension).ok() == Some(embedder.dimension())
        }
        (None, Err(error)) => missing_build_identity(&error),
        _ => false,
    };
    if !unchanged {
        return Err(SearchError::UnverifiableRemoteSpace {
            producer: format!("index_builder.{tier}"),
            reason: "producer identity changed after build admission".to_owned(),
        });
    }
    Ok(())
}

pub(super) async fn embed_build_document(
    cx: &Cx,
    embedder: &dyn Embedder,
    identity: Option<&EmbeddingIdentityBundleV1>,
    tier: &'static str,
    text: &str,
) -> SearchResult<Vec<f32>> {
    validate_build_producer(embedder, identity, tier)?;
    let completion_phase = match tier {
        "fast" => "fast document embedding completion",
        _ => "quality document embedding completion",
    };
    let values = if let Some(expected) = identity {
        let response = embedder.embed_bound(cx, text).await;
        // Preserve an already-typed cancellation; otherwise caller cancellation
        // wins before validating or accepting a late response or provider error.
        if !matches!(&response, Err(SearchError::Cancelled { .. })) {
            build_checkpoint(cx, completion_phase)?;
        }
        let bound = response?;
        if &bound.identity != expected {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: format!("index_builder.{tier}"),
                reason: "embedding response does not carry the admitted producer identity"
                    .to_owned(),
            });
        }
        bound.validate()?;
        bound.values
    } else {
        let response = embedder.embed(cx, text).await;
        if !matches!(&response, Err(SearchError::Cancelled { .. })) {
            build_checkpoint(cx, completion_phase)?;
        }
        response?
    };
    validate_build_producer(embedder, identity, tier)?;
    Ok(values)
}

/// Build failures that end the whole build rather than one document:
/// cancellation and any producer or response identity refusal.
pub(super) const fn is_terminal_build_error(error: &SearchError) -> bool {
    matches!(
        error,
        SearchError::Cancelled { .. } | SearchError::UnverifiableRemoteSpace { .. }
    )
}

/// Embed `texts` for one tier: one vector or one ordinary failure per input,
/// in input order. A terminal failure is returned as `Err` (GH #57).
///
/// A producer whose bound batch binds what its singles would
/// ([`Embedder::bound_batch_is_native`]) gets the texts as one
/// `embed_batch_bound` call; a failed batch is split in halves through that
/// same operation down to single inputs, so one bad input fails alone and at
/// most `2 * texts.len() - 1` requests are made. Every other producer, and the
/// explicit legacy unidentified path, keeps one call per input.
pub(super) async fn embed_build_texts(
    cx: &Cx,
    embedder: &dyn Embedder,
    identity: Option<&EmbeddingIdentityBundleV1>,
    tier: &'static str,
    texts: &[&str],
    metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
) -> SearchResult<Vec<SearchResult<Vec<f32>>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let Some(expected) = identity.filter(|_| embedder.bound_batch_is_native()) else {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let started = Instant::now();
            match embed_build_document(cx, embedder, identity, tier, text).await {
                Ok(values) => {
                    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
                    export_embedding_completed(metrics_exporter, embedder, 1, duration_ms);
                    results.push(Ok(values));
                }
                Err(error) if is_terminal_build_error(&error) => return Err(error),
                Err(error) => results.push(Err(error)),
            }
        }
        return Ok(results);
    };
    let mut results: Vec<Option<SearchResult<Vec<f32>>>> =
        std::iter::repeat_with(|| None).take(texts.len()).collect();
    // A work stack of index ranges; the right half is pushed first so the
    // left half is embedded first and requests stay in input order.
    let mut pending = Vec::new();
    pending.push(0..texts.len());
    while let Some(range) = pending.pop() {
        match embed_build_bound_batch(
            cx,
            embedder,
            expected,
            tier,
            &texts[range.clone()],
            metrics_exporter,
        )
        .await
        {
            Ok(vectors) => {
                for (slot, values) in range.zip(vectors) {
                    results[slot] = Some(Ok(values));
                }
            }
            Err(error) if is_terminal_build_error(&error) => return Err(error),
            Err(error) if range.len() == 1 => results[range.start] = Some(Err(error)),
            Err(error) => {
                tracing::debug!(
                    tier,
                    batch_size = range.len(),
                    error = %error,
                    "bound embedding batch failed; splitting it with the same producer"
                );
                let middle = range.start + range.len() / 2;
                pending.push(middle..range.end);
                pending.push(range.start..middle);
            }
        }
    }
    Ok(results
        .into_iter()
        .map(|result| result.expect("every input slot resolves exactly once"))
        .collect())
}

/// One `embed_batch_bound` call under the admitted identity: the producer is
/// checked before and after inference (before any retry decision), every
/// response must carry the admitted identity (else terminal), and the batch
/// is admitted only when its cardinality and every value validate.
async fn embed_build_bound_batch(
    cx: &Cx,
    embedder: &dyn Embedder,
    expected: &EmbeddingIdentityBundleV1,
    tier: &'static str,
    texts: &[&str],
    metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
) -> SearchResult<Vec<Vec<f32>>> {
    validate_build_producer(embedder, Some(expected), tier)?;
    let completion_phase = match tier {
        "fast" => "fast document embedding completion",
        _ => "quality document embedding completion",
    };
    let started = Instant::now();
    let response = embedder.embed_batch_bound(cx, texts).await;
    if !matches!(&response, Err(SearchError::Cancelled { .. })) {
        build_checkpoint(cx, completion_phase)?;
    }
    validate_build_producer(embedder, Some(expected), tier)?;
    let bound = response?;
    if bound.len() != texts.len() {
        return Err(SearchError::EmbeddingFailed {
            model: embedder.id().to_owned(),
            source: format!(
                "bound batch returned {} outputs for {} inputs",
                bound.len(),
                texts.len()
            )
            .into(),
        });
    }
    if bound.iter().any(|response| &response.identity != expected) {
        return Err(SearchError::UnverifiableRemoteSpace {
            producer: format!("index_builder.{tier}"),
            reason: "embedding response does not carry the admitted producer identity".to_owned(),
        });
    }
    for response in &bound {
        response.validate()?;
    }
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    export_embedding_completed(metrics_exporter, embedder, texts.len(), duration_ms);
    Ok(bound.into_iter().map(|response| response.values).collect())
}

pub(super) fn build_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

pub(super) fn export_error(
    metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    error: &SearchError,
) {
    if let Some(exporter) = metrics_exporter {
        exporter.on_error(error);
    }
}

pub(super) fn export_embedding_completed(
    metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    embedder: &dyn Embedder,
    batch_size: usize,
    duration_ms: f64,
) {
    let Some(exporter) = metrics_exporter else {
        return;
    };
    let payload = EmbeddingMetrics {
        embedder_id: embedder.id().to_owned(),
        batch_size,
        duration_ms,
        dimension: embedder.dimension(),
        is_semantic: embedder.is_semantic(),
    };
    exporter.on_embedding_completed(&payload);
}

pub(super) fn export_index_updated(
    metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    doc_count: usize,
    index_size_bytes: u64,
    updated_docs: usize,
) {
    let Some(exporter) = metrics_exporter else {
        return;
    };
    let payload = IndexMetrics {
        doc_count,
        index_size_bytes,
        updated_docs,
        staleness_detected: false,
    };
    exporter.on_index_updated(&payload);
}

pub(super) fn compute_size_breakdown(data_dir: &Path) -> IndexSizeBreakdown {
    let fast_path = data_dir.join(VECTOR_INDEX_FAST_FILENAME);
    let fallback_path = data_dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
    let quality_path = data_dir.join(VECTOR_INDEX_QUALITY_FILENAME);

    let vector_fast = if fast_path.exists() {
        file_size_bytes(&fast_path)
    } else {
        file_size_bytes(&fallback_path)
    };
    let vector_quality = file_size_bytes(&quality_path);
    let lexical = dir_size_bytes(&data_dir.join("lexical"));

    IndexSizeBreakdown {
        total: vector_fast
            .saturating_add(vector_quality)
            .saturating_add(lexical),
        vector_fast,
        vector_quality,
        lexical,
    }
}

/// Recursive byte size of a directory tree; 0 when the directory is absent.
fn dir_size_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries.filter_map(Result::ok).fold(0u64, |acc, entry| {
        let path = entry.path();
        let size = if path.is_dir() {
            dir_size_bytes(&path)
        } else {
            file_size_bytes(&path)
        };
        acc.saturating_add(size)
    })
}

fn file_size_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |metadata| metadata.len())
}
