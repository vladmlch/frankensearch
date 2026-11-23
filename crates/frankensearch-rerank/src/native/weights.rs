use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use ft_api::{FrankenTorchSession, quantize_per_output_channel_i8};
use ft_core::{DType, Device, ExecutionMode, TensorMeta};

use frankensearch_core::error::{SearchError, SearchResult};

use super::model::Model;
use super::{H, rerank_err};

/// A Linear layer's weights, statically quantized to int8 with per-output-channel
/// f32 scales, plus its f32 bias. The three buffers are `Arc`-shared so the parsed
/// weights are stored once and cloned cheaply into every pooled session.
#[derive(Clone)]
pub(super) struct QLinear {
    /// Int8 weights. Row-major `[out, in]` when `packed` is false; NR=4
    /// panel-interleaved (from `pack_int8_weights_nr4`) when `packed` is true.
    pub(super) w_i8: Arc<Vec<i8>>,
    /// Per-output-channel f32 dequantization scales (len `out`).
    pub(super) w_scales: Arc<Vec<f32>>,
    /// f32 bias (len `out`).
    pub(super) bias: Arc<Vec<f32>>,
    pub(super) out: usize,
    pub(super) in_: usize,
    /// Whether `w_i8` is NR=4-packed (pre-packed SDOT kernel) vs row-major.
    pub(super) packed: bool,
}

/// Precision is selected at load time and belongs to the embedding producer's
/// identity. Reranking and the original native embedder keep their int8 path.
#[derive(Debug, Clone, Copy)]
pub enum LinearPrecision {
    Int8,
    F32,
}

#[derive(Clone)]
pub(super) enum LinearWeights {
    Int8(QLinear),
    F32 {
        /// Transposed once at load: contiguous `[in, out]` for the raw BMM kernel.
        transposed: Arc<Vec<f32>>,
        bias: Arc<Vec<f32>>,
        out: usize,
        in_: usize,
    },
}

impl LinearWeights {
    pub(super) fn from_f32(
        values: &[f32],
        bias: Vec<f32>,
        out: usize,
        in_: usize,
        precision: LinearPrecision,
    ) -> SearchResult<Self> {
        if out == 0 || in_ == 0 || in_.checked_mul(out) != Some(values.len()) || bias.len() != out {
            return Err(rerank_err(
                "linear.load",
                "invalid weight or bias dimensions",
            ));
        }
        match precision {
            LinearPrecision::Int8 => {
                let (w_i8, w_scales) = quantize_per_output_channel_i8(values, out, in_);
                let packed = cfg!(target_arch = "aarch64") && out % 4 == 0 && in_ % 16 == 0;
                let w_i8 = if packed {
                    ft_api::pack_int8_weights_nr4(&w_i8, out, in_)
                } else {
                    w_i8
                };
                Ok(Self::Int8(QLinear {
                    w_i8: Arc::new(w_i8),
                    w_scales: Arc::new(w_scales),
                    bias: Arc::new(bias),
                    out,
                    in_,
                    packed,
                }))
            }
            LinearPrecision::F32 => {
                let mut transposed = vec![0.0; values.len()];
                for row in 0..out {
                    for col in 0..in_ {
                        transposed[col * out + row] = values[row * in_ + col];
                    }
                }
                Ok(Self::F32 {
                    transposed: Arc::new(transposed),
                    bias: Arc::new(bias),
                    out,
                    in_,
                })
            }
        }
    }

    pub(super) fn forward(&self, x: &[f32], m: usize) -> SearchResult<Vec<f32>> {
        match self {
            Self::Int8(q) => {
                if m.checked_mul(q.in_) != Some(x.len()) {
                    return Err(rerank_err("linear.int8", "input dimensions disagree"));
                }
                Ok(if q.packed {
                    ft_api::linear_int8_dynamic_prepacked_f32(
                        x,
                        m,
                        q.in_,
                        &q.w_i8,
                        &q.w_scales,
                        q.out,
                        Some(&q.bias),
                    )
                } else {
                    ft_api::linear_int8_dynamic_f32(
                        x,
                        m,
                        q.in_,
                        &q.w_i8,
                        &q.w_scales,
                        q.out,
                        Some(&q.bias),
                    )
                })
            }
            Self::F32 {
                transposed,
                bias,
                out,
                in_,
            } => {
                if m.checked_mul(*in_) != Some(x.len()) {
                    return Err(rerank_err("linear.f32", "input dimensions disagree"));
                }
                let input = TensorMeta::from_shape(vec![1, m, *in_], DType::F32, Device::Cpu);
                let weights = TensorMeta::from_shape(vec![1, *in_, *out], DType::F32, Device::Cpu);
                let mut y = ft_api::bmm_tensor_contiguous_f32(x, transposed, &input, &weights)
                    .map_err(|e| rerank_err("linear.f32", e))?;
                for row in y.chunks_exact_mut(*out) {
                    for (value, offset) in row.iter_mut().zip(bias.iter()) {
                        *value += offset;
                    }
                }
                Ok(y)
            }
        }
    }
}

/// A weight tensor is a Linear weight iff it is a `.weight`
/// that is neither a `LayerNorm` gain nor an embedding table.
fn is_linear_weight(name: &str) -> bool {
    name.ends_with(".weight") && !name.contains("LayerNorm") && !name.contains("embeddings")
}

/// Decode complete little-endian F32 values without per-element work when the
/// host byte order and input alignment already match. The checked cast never
/// assumes that a byte buffer or a safetensors data offset is F32-aligned.
pub(super) fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    if cfg!(target_endian = "little")
        && let Ok(values) = bytemuck::try_cast_slice::<[u8; 4], f32>(chunks)
    {
        return values.to_vec();
    }
    chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

/// Parsed weight data: precision-selected Linear weights keyed by layer prefix, plus the f32
/// `embedding/LayerNorm` parameter values. [`build_model`] consumes this staging
/// representation so large embedding tables move into the session without a
/// second resident copy.
pub struct SharedWeights {
    pub(super) qw: HashMap<String, LinearWeights>,
    pub(super) f32_params: HashMap<String, (Arc<Vec<f32>>, Vec<usize>)>,
    pub(super) encoder_layers: usize,
}

/// Parse a safetensors file: retain or quantize the Linear weights according to
/// the explicit precision, and keep embeddings + `LayerNorm` as f32. Non-F32 tensors (e.g. the
/// I64 `position_ids` buffer) are skipped — those indices are regenerated at forward.
pub fn parse_weights(path: &Path, precision: LinearPrecision) -> SearchResult<SharedWeights> {
    let bytes = fs::read(path).map_err(|e| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: format!("read safetensors: {e}").into(),
    })?;
    if bytes.len() < 8 {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors file too small".into(),
        });
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")))
        .map_err(|_| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors header length exceeds usize::MAX".into(),
        })?;
    let header_end = 8usize
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors header length out of range".into(),
        })?;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|e| {
        SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: format!("safetensors header parse: {e}").into(),
        }
    })?;
    let data = &bytes[header_end..];
    let obj = header
        .as_object()
        .ok_or_else(|| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors header is not an object".into(),
        })?;

    // First pass: read every F32 tensor into raw (name -> (values, shape)).
    let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if dtype != "F32" {
            continue; // skip I64 position_ids and any non-float buffers
        }
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|u| {
                        usize::try_from(u).map_err(|_| SearchError::ModelLoadFailed {
                            path: path.to_path_buf(),
                            source: format!(
                                "safetensors tensor {name} shape dimension exceeds usize::MAX"
                            )
                            .into(),
                        })
                    })
                    .collect::<SearchResult<_>>()
            })
            .transpose()?
            .unwrap_or_default();
        let offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("safetensors tensor {name} missing data_offsets").into(),
            })?;
        let start = usize::try_from(
            offsets
                .first()
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .map_err(|_| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: format!("safetensors tensor {name} start offset exceeds usize::MAX").into(),
        })?;
        let end = usize::try_from(
            offsets
                .get(1)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .map_err(|_| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: format!("safetensors tensor {name} end offset exceeds usize::MAX").into(),
        })?;
        if start > end || end > data.len() {
            return Err(SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("safetensors tensor {name} has out-of-range offsets").into(),
            });
        }
        let vals = decode_f32_le(&data[start..end]);
        // Normalize HuggingFace BERT key conventions to the `bert.`-prefixed scheme
        // the shared encoder/`build_model` use. sentence-transformers all-MiniLM-L6-v2
        // ships bare `embeddings.*` / `encoder.*` keys; cross-encoder/ms-marco ships
        // `bert.`-prefixed ones — so this is a strict no-op there (those keys start with
        // `bert.embeddings`/`bert.encoder`, not bare `embeddings.`/`encoder.`). Backbone
        // keys only; `pooler`/`classifier` are left untouched.
        let key = if name.starts_with("embeddings.") || name.starts_with("encoder.") {
            format!("bert.{name}")
        } else {
            name.clone()
        };
        raw.insert(key, (vals, shape));
    }
    if raw.is_empty() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "no F32 tensors found in safetensors".into(),
        });
    }

    // Every retained tensor is now owned by `raw`; release the original 449 MiB
    // safetensors byte buffer before quantization creates another set of weights.
    drop(bytes);

    let encoder_layer_indices = raw
        .keys()
        .filter_map(|name| {
            name.strip_prefix("bert.encoder.layer.")?
                .split_once('.')?
                .0
                .parse::<usize>()
                .ok()
        })
        .collect::<BTreeSet<_>>();
    let encoder_layers = encoder_layer_indices
        .last()
        .copied()
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors has no numbered BERT encoder layers".into(),
        })?;
    if encoder_layer_indices.len() != encoder_layers
        || !(0..encoder_layers).all(|index| encoder_layer_indices.contains(&index))
    {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors BERT encoder layers are not contiguous from zero".into(),
        });
    }

    // Second pass, part one: fuse each layer's Q/K/V projections while the raw
    // tensors are still present. Missing or malformed projections are a hard
    // topology failure: every attested BERT encoder layer must be executable.
    let mut qw: HashMap<String, LinearWeights> = HashMap::new();
    // Fuse each layer's Q/K/V projections into one `[3H, H]` linear
    // (key `…attention.self.qkv`). In int8 mode the forward quantizes `emb` once and runs a
    // single int8 GEMM instead of three — cutting the per-call quant / rayon-launch
    // / dequant / tape-node overhead that is a real fraction of the short-sequence
    // forward (the SDOT math itself is at its M4 throughput ceiling). The stacked
    // weight re-quantizes per output channel, so each of the 3H rows keeps its own
    // scale and the fused output is byte-identical to the three separate linears.
    for i in 0..encoder_layers {
        let p = format!("bert.encoder.layer.{i}");
        let parts = ["query", "key", "value"];
        let mut stacked: Vec<f32> = Vec::with_capacity(3 * H * H);
        let mut bias: Vec<f32> = Vec::with_capacity(3 * H);
        for part in parts {
            let wn = format!("{p}.attention.self.{part}.weight");
            match raw.get(&wn) {
                Some((vals, shape)) if shape.len() == 2 && shape[0] == H && shape[1] == H => {
                    stacked.extend_from_slice(vals);
                    let b = raw
                        .get(&format!("{p}.attention.self.{part}.bias"))
                        .map(|(b, _)| b.clone())
                        .unwrap_or_else(|| vec![0.0f32; H]);
                    bias.extend_from_slice(&b);
                }
                _ => {
                    return Err(SearchError::ModelLoadFailed {
                        path: path.to_path_buf(),
                        source: format!(
                            "encoder layer {i} has missing or malformed {part} projection"
                        )
                        .into(),
                    });
                }
            }
        }
        let (out, in_) = (3 * H, H);
        qw.insert(
            format!("{p}.attention.self.qkv"),
            LinearWeights::from_f32(&stacked, bias, out, in_, precision)?,
        );
    }

    // Second pass, part two: remove and prepare each Linear weight together
    // with its bias. Removing entries lets the large f32 vectors move into their
    // final owners instead of being cloned from the staging map.
    let linear_weight_names = raw
        .keys()
        .filter(|name| is_linear_weight(name))
        .cloned()
        .collect::<Vec<_>>();
    for name in linear_weight_names {
        let (vals, shape) = raw
            .remove(&name)
            .expect("linear weight name was collected from raw");
        let prefix = name.strip_suffix(".weight").expect("ends_with .weight");
        let out = *shape.first().unwrap_or(&0);
        let in_ = *shape.get(1).unwrap_or(&0);
        if out == 0 || in_ == 0 || vals.len() != out * in_ {
            return Err(SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!(
                    "linear weight {name} bad shape {shape:?} for {} values",
                    vals.len()
                )
                .into(),
            });
        }
        let bias = raw
            .remove(&format!("{prefix}.bias"))
            .map_or_else(|| vec![0.0f32; out], |(bias, _)| bias);
        // The int8 mode pre-packs eligible aarch64 weights once; the F32 mode
        // transposes once. Neither layout is rebuilt during inference.
        qw.insert(
            prefix.to_string(),
            LinearWeights::from_f32(&vals, bias, out, in_, precision)?,
        );
    }
    if qw.is_empty() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "no Linear weights found".into(),
        });
    }

    let mut f32_params: HashMap<String, (Arc<Vec<f32>>, Vec<usize>)> = HashMap::new();
    for (name, (vals, shape)) in raw {
        if name.strip_suffix(".bias").is_some() && !name.contains("LayerNorm") {
            // Bias for a non-Linear auxiliary tensor; it is not a model leaf.
            continue;
        }
        f32_params.insert(name, (Arc::new(vals), shape));
    }

    Ok(SharedWeights {
        qw,
        f32_params,
        encoder_layers,
    })
}

/// Build a session by consuming parsed weights. Large embedding tables transfer
/// directly into the session; only the small `LayerNorm` values are duplicated
/// because the tape-free path also needs their raw buffers.
pub fn build_model(shared: SharedWeights) -> SearchResult<Model> {
    let SharedWeights {
        qw,
        f32_params,
        encoder_layers,
    } = shared;
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    session.no_grad_enter();
    let mut w = HashMap::with_capacity(f32_params.len());
    let mut raw_params = HashMap::with_capacity(f32_params.len());
    for (name, (vals, shape)) in f32_params {
        let raw_layer_norm = name.contains("LayerNorm").then(|| Arc::clone(&vals));
        let vals = Arc::try_unwrap(vals).unwrap_or_else(|vals| vals.as_ref().clone());
        let node = session
            .tensor_variable_f32(vals, shape, false)
            .map_err(|e| rerank_err("build_model", format!("create f32 tensor {name}: {e}")))?;
        if let Some(raw) = raw_layer_norm {
            raw_params.insert(name.clone(), raw);
        }
        w.insert(name, node);
    }
    // Tape boundary AFTER the persistent f32 leaves are created; each forward
    // truncates back to here to free intermediates while keeping parameters.
    let weights_boundary = session.autograd_graph_node_count();
    Ok(Model {
        s: session,
        w,
        qw,
        raw_params,
        encoder_layers,
        weights_boundary,
    })
}
