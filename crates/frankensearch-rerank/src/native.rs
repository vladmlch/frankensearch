//! Pure-Rust cross-encoder reranker backed by frankentorch (no ONNX / no `ort`).
//!
//! Reimplements the `cross-encoder/ms-marco-MiniLM-L6-v2` `BertForSequenceClassification`
//! forward pass (6 layers, hidden 384, 12 heads, exact GELU, `LayerNorm` eps 1e-12,
//! `[CLS]` pooler + classifier, `sigmoid(logit)`) on frankentorch tensors, matching the
//! ONNX dynamic-quant scheme: an **f32 substrate** (embeddings, `LayerNorm`, softmax,
//! GELU, tanh) with **int8 Linear matmuls** (bd-1nl13.10/.15). Every Linear (attention
//! QKV/output, FFN, pooler, classifier) is statically int8-quantized per output channel
//! at load; its activation is dynamically int8-quantized per row at forward
//! (`tensor_linear_int8_dynamic`). Validated against the numpy/ONNX reference
//! (bd-1nl13.2/.3): the reference ranking is preserved.
//!
//! Embedding lookups go through `tensor_index_select` rather than `tensor_embedding`:
//! `index_select` preserves the weight dtype (f32 in/f32 out, frankentorch-40i), whereas
//! `tensor_embedding`'s custom gather still materialises f64. The two are semantically
//! identical here (no `padding_idx`). `LayerNorm` hits frankentorch's f32 fused no-grad
//! fast path.
//!
//! The only reranker backend (ort/ONNX was removed in bd-1nl13.6); feature-gated
//! behind `native`.

use frankensearch_core::error::{SearchError, SearchResult};

/// Observation of the existing public certificate execution, never another engine.
#[cfg(test)]
pub(crate) mod certificate_trace;
mod kernels;
mod model;
mod reranker;
#[cfg(test)]
mod tests;
mod weights;

pub use reranker::NativeReranker;

pub(crate) use model::Model;
pub(crate) use weights::{LinearPrecision, build_model, parse_weights};

const H: usize = 384;
const RERANKER_LAYERS: usize = 6;
const NH: usize = 12;
const HD: usize = H / NH; // 32
const INTER: usize = 4 * H; // 1536 — FFN intermediate width
const EPS: f64 = 1.0e-12;
const EPS_F32: f32 = 1.0e-12;
const ATTN_SCALE_F32: f32 = 0.176_776_69;
const CLS_Q_CACHE_MIN_SEQ: usize = 256;
pub(crate) const DEFAULT_MAX_LENGTH: usize = 512;
/// Token budget per batched forward. Documents are reranked in chunks whose total
/// token count stays under this cap so the chunk's attention intermediates (which
/// co-exist on the tape until the per-chunk truncate) stay memory-bounded, while
/// still giving the int8 GEMM a large enough row count to amortize weight loads.
/// A single document longer than this is still processed alone (one-doc chunk).
const MAX_BATCH_TOKENS: usize = 2048;
/// Max per-document sequence length that the tape-free raw forward path handles.
/// The raw path's attention is itself a cache-blocking `gemm` bmm now (see
/// [`kernels::fused_attention`]), so it has no per-query disadvantage at long sequences —
/// it's strictly the tape path minus the per-op tape overhead, plus the CLS-only
/// final layer. We therefore route everything up to the default max length through
/// it; only configurations with a larger `max_length` fall back to the tape bmm
/// path. (Was 384, the crossover of the OLD per-query fused attention vs the tape
/// bmm — obsolete since the raw attention became a gemm.)
const FUSED_ATTN_MAX_SEQ: usize = DEFAULT_MAX_LENGTH;
const MODEL_NAME: &str = "ms-marco-minilm-l-6-v2";
pub(crate) const SAFETENSORS_PRIMARY: &str = "model_f32.safetensors";
pub(crate) const SAFETENSORS_FALLBACK: &str = "model.safetensors";
pub(crate) const TOKENIZER_JSON: &str = "tokenizer.json";

fn rerank_err(ctx: &str, e: impl std::fmt::Display) -> SearchError {
    SearchError::RerankFailed {
        model: MODEL_NAME.to_owned(),
        source: format!("{ctx}: {e}").into(),
    }
}

fn index_to_i64(index: usize, ctx: &str) -> SearchResult<i64> {
    i64::try_from(index).map_err(|_| rerank_err(ctx, format!("index {index} exceeds i64::MAX")))
}
