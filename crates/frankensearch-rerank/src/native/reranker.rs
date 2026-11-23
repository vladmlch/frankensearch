use std::path::Path;
use std::sync::{Arc, Mutex};

use asupersync::Cx;
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use asupersync::sync::{LockError, Mutex as AsyncMutex, OwnedMutexGuard};
use tokenizers::Tokenizer;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{RerankDocument, RerankScore, SyncRerank};

use super::model::Model;
use super::weights::{LinearPrecision, build_model, parse_weights};
use super::{
    DEFAULT_MAX_LENGTH, MAX_BATCH_TOKENS, MODEL_NAME, RERANKER_LAYERS, SAFETENSORS_FALLBACK,
    SAFETENSORS_PRIMARY, TOKENIZER_JSON, rerank_err,
};

/// Pure-Rust frankentorch cross-encoder reranker.
///
/// Clones share the loaded model and serialize inference. For async
/// [`Reranker`](frankensearch_core::traits::Reranker) use, attach a caller-owned
/// pool with [`Self::with_blocking_pool`]. Synchronous use needs no runtime.
#[derive(Clone)]
pub struct NativeReranker {
    /// A single frankentorch session behind a `Mutex`. Documents are reranked in
    /// a SEQUENTIAL loop, and each forward parallelizes internally across cores
    /// (the int8 Linear kernel + the f32 attention `bmm` / `softmax` /
    /// `layer_norm` ops use ambient rayon). Because there is no doc-level
    /// `par_iter`, nothing nests rayon while holding the lock, so the
    /// nested-rayon + `Mutex` deadlock is impossible by construction. Per-forward
    /// parallelism makes the common few-doc rerank fast (each forward uses all
    /// cores); a batched forward is the deferred next step for large-N throughput.
    pub(super) inner: Arc<Mutex<Model>>,
    pub(super) tokenizer: Arc<Tokenizer>,
    pub(super) admission: Arc<AsyncMutex<()>>,
    pub(super) blocking_pool: Option<BlockingPoolHandle>,
    pub(super) max_length: usize,
    pub(super) name: String,
    pub(super) id: String,
}

impl std::fmt::Debug for NativeReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeReranker")
            .field("name", &self.name)
            .field("max_length", &self.max_length)
            .finish_non_exhaustive()
    }
}

impl NativeReranker {
    /// Load the reranker from a model directory containing a safetensors weight file
    /// (`model_f32.safetensors` preferred, else `model.safetensors`) and `tokenizer.json`.
    ///
    /// # Errors
    /// `SearchError::ModelNotFound` when required files are missing;
    /// `SearchError::ModelLoadFailed` when the tokenizer or weights fail to load.
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        let dir = model_dir.as_ref();

        let tok_path = dir.join(TOKENIZER_JSON);
        if !tok_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!(
                    "{MODEL_NAME} (missing {TOKENIZER_JSON} in {})",
                    dir.display()
                ),
            });
        }
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| SearchError::ModelLoadFailed {
                path: tok_path.clone(),
                source: format!("tokenizer load failed: {e}").into(),
            })?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: DEFAULT_MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| SearchError::ModelLoadFailed {
                path: tok_path.clone(),
                source: format!("failed to enable truncation: {e}").into(),
            })?;

        let weights_path = {
            let primary = dir.join(SAFETENSORS_PRIMARY);
            if primary.is_file() {
                primary
            } else {
                dir.join(SAFETENSORS_FALLBACK)
            }
        };
        if !weights_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!(
                    "{MODEL_NAME} (missing {SAFETENSORS_PRIMARY} or {SAFETENSORS_FALLBACK} in {})",
                    dir.display()
                ),
            });
        }

        // Parse + quantize the weights once and build a single session. Documents
        // are reranked sequentially; each forward parallelizes internally across
        // cores. No pool is needed (and one session keeps the f32 embedding table
        // resident only once, ~47 MB instead of per-slot copies).
        let shared = parse_weights(&weights_path, LinearPrecision::Int8)?;
        if shared.encoder_layers != RERANKER_LAYERS {
            return Err(SearchError::ModelLoadFailed {
                path: weights_path,
                source: format!(
                    "reranker weights contain {} encoder layers, expected {RERANKER_LAYERS}",
                    shared.encoder_layers
                )
                .into(),
            });
        }
        let linear_int8 = shared.qw.len();
        let f32_params = shared.f32_params.len();
        let model = build_model(shared)?;

        tracing::info!(
            model = MODEL_NAME,
            linear_int8,
            f32_params,
            max_length = DEFAULT_MAX_LENGTH,
            model_dir = %dir.display(),
            "native frankentorch reranker loaded (int8 linear, parallel forward)"
        );

        Ok(Self {
            inner: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            admission: Arc::new(AsyncMutex::new(())),
            blocking_pool: None,
            max_length: DEFAULT_MAX_LENGTH,
            name: MODEL_NAME.to_owned(),
            id: MODEL_NAME.to_owned(),
        })
    }

    /// Attach the caller's bounded blocking pool for async reranking.
    ///
    /// The caller retains and drains the pool at shutdown. Cancellation cannot
    /// preempt a tensor kernel: the worker keeps model admission until it exits,
    /// checking cancellation between tokenizations and inference chunks. No
    /// runtime or pool is created internally; without a pool async calls fail.
    #[must_use]
    pub fn with_blocking_pool(mut self, pool: BlockingPoolHandle) -> Self {
        self.blocking_pool = Some(pool);
        self
    }

    async fn infer(
        &self,
        cx: &Cx,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        rerank_checkpoint(cx)?;
        let pool = self.blocking_pool.clone().ok_or_else(|| {
            rerank_err(
                "async inference",
                "requires a caller-owned blocking pool; attach it with NativeReranker::with_blocking_pool",
            )
        })?;
        if pool.is_shutdown() {
            return Err(rerank_err(
                "async inference",
                "caller-owned blocking pool is shut down",
            ));
        }
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        // Cancelled waiters must not occupy pool workers waiting for the model.
        let admission = OwnedMutexGuard::lock(Arc::clone(&self.admission), cx)
            .await
            .map_err(|error| match error {
                LockError::Cancelled => SearchError::Cancelled {
                    phase: "native.rerank".to_owned(),
                    reason: "native reranker admission cancelled".to_owned(),
                },
                error => rerank_err("model admission", error),
            })?;
        let owner = self.clone();
        let query = query.to_owned();
        let documents = documents.to_vec();
        let request_cx = cx.clone();
        let worker_cx = cx.clone().with_blocking_pool_handle(Some(pool));
        let mut worker = worker_cx
            .spawn_blocking(move |child| {
                let _admission = admission;
                // asupersync can recover a rejected pool submission by calling
                // it inline on its wrapper task. Refuse that fallback before
                // touching the model, including shutdown racing admission.
                if Cx::is_active() {
                    return Err(rerank_err(
                        "inference worker",
                        "blocking pool dispatch fell back to an async executor",
                    ));
                }
                let mut scores = owner.rerank_checked(&query, &documents, || {
                    rerank_checkpoint(&request_cx)?;
                    rerank_checkpoint(&child)
                })?;
                scores.sort_by(|lhs, rhs| {
                    rhs.score
                        .total_cmp(&lhs.score)
                        .then_with(|| lhs.original_rank.cmp(&rhs.original_rank))
                        .then_with(|| lhs.doc_id.cmp(&rhs.doc_id))
                });
                rerank_checkpoint(&child)?;
                rerank_checkpoint(&request_cx)?;
                Ok(scores)
            })
            .map_err(|error| rerank_err("cannot admit inference worker", error))?;
        let result = worker.join(cx).await.map_err(|error| match error {
            asupersync::runtime::JoinError::Cancelled(_) => SearchError::Cancelled {
                phase: "native.rerank".to_owned(),
                reason: "native reranker worker cancelled".to_owned(),
            },
            error => rerank_err("inference worker", error),
        });
        // TaskHandle::join is uninterruptible; it does not inspect its Cx.
        // A request cancelled during the last chunk must never publish scores.
        rerank_checkpoint(cx)?;
        result?
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rerank_checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "native.rerank".to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

impl NativeReranker {
    fn rerank_checked(
        &self,
        query: &str,
        documents: &[RerankDocument],
        checkpoint: impl Fn() -> SearchResult<()>,
    ) -> SearchResult<Vec<RerankScore>> {
        checkpoint()?;
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        // Tokenize every (query, doc) pair, then rerank in BATCHED chunks: each
        // chunk is a single `forward_batch` over the concatenated tokens, so the
        // int8 Linear weights are reused across all the chunk's documents' tokens
        // (the throughput win). Chunks are bounded by `MAX_BATCH_TOKENS` to keep
        // the tape memory-bounded. There is NO doc-level `par_iter` — the session
        // `Mutex` is locked once and the forward parallelizes internally — so the
        // nested-rayon + `Mutex` deadlock still cannot occur. Output order (and
        // `original_rank`) follows the input and the logits are deterministic.
        let mut encoded: Vec<(Vec<i64>, Vec<i64>)> = Vec::with_capacity(documents.len());
        for doc in documents {
            checkpoint()?;
            let encoding = self
                .tokenizer
                .encode((query, doc.text.as_str()), true)
                .map_err(|e| rerank_err("tokenize", e))?;
            // Truncate to `max_length` before the i64 conversion + collect (see
            // `crate::ids_to_truncated_i64`): for documents that tokenize past the cap
            // this materializes only `max_length` ids per side instead of the whole
            // sequence. `ids` and `typ` share the encoding length, so per-side
            // `take(max_length)` matches the old `if ids.len() > max { truncate both }`.
            let ids = crate::ids_to_truncated_i64(encoding.get_ids(), self.max_length);
            let typ = crate::ids_to_truncated_i64(encoding.get_type_ids(), self.max_length);
            encoded.push((ids, typ));
        }

        let mut model = self
            .inner
            .lock()
            .map_err(|e| rerank_err("lock", format!("reranker mutex poisoned: {e}")))?;
        let mut logits: Vec<f32> = Vec::with_capacity(documents.len());
        let mut chunk_start = 0usize;
        while chunk_start < encoded.len() {
            checkpoint()?;
            // Grow the chunk until adding the next doc would exceed the token
            // budget; always take at least one doc (a single over-budget doc runs
            // alone).
            let mut chunk_end = chunk_start + 1;
            let mut chunk_tokens = encoded[chunk_start].0.len();
            while chunk_end < encoded.len()
                && chunk_tokens + encoded[chunk_end].0.len() <= MAX_BATCH_TOKENS
            {
                chunk_tokens += encoded[chunk_end].0.len();
                chunk_end += 1;
            }
            logits.extend(model.forward_batch(&encoded[chunk_start..chunk_end])?);
            chunk_start = chunk_end;
        }
        drop(model);

        checkpoint()?;

        let out = documents
            .iter()
            .zip(logits)
            .enumerate()
            .map(|(rank, (doc, logit))| {
                let (score, raw_logit) = if logit.is_finite() {
                    (sigmoid(logit), Some(logit))
                } else {
                    (0.0, None)
                };
                RerankScore {
                    doc_id: doc.doc_id.clone(),
                    score,
                    original_rank: rank,
                    raw_logit,
                }
            })
            .collect();
        Ok(out)
    }
}

impl SyncRerank for NativeReranker {
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        self.rerank_checked(query, documents, || Ok(()))
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn is_available(&self) -> bool {
        true
    }
}

impl frankensearch_core::traits::Reranker for NativeReranker {
    fn rerank<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        documents: &'a [RerankDocument],
    ) -> frankensearch_core::traits::SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(self.infer(cx, query, documents))
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn is_available(&self) -> bool {
        self.blocking_pool
            .as_ref()
            .is_some_and(|pool| !pool.is_shutdown())
    }
}
