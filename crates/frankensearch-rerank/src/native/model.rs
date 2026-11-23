use std::collections::HashMap;
use std::sync::Arc;

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;

use frankensearch_core::error::SearchResult;

#[cfg(test)]
use super::certificate_trace;
use super::kernels::{
    AttnScratch, fast_gelu_inplace, fast_softmax_inplace, fused_attention, fused_attention_cls,
};
use super::weights::LinearWeights;
use super::{
    ATTN_SCALE_F32, EPS, EPS_F32, FUSED_ATTN_MAX_SEQ, H, HD, INTER, NH, index_to_i64, rerank_err,
};

/// Owns the frankentorch session and the loaded weight tensors. Mutated during the
/// forward pass, so it lives behind a `Mutex` in `NativeReranker`.
pub struct Model {
    pub(super) s: FrankenTorchSession,
    /// f32 leaf nodes for the non-Linear parameters (`word/position/token_type`
    /// embeddings and every `LayerNorm` weight/bias) — these stay in f32.
    pub(super) w: HashMap<String, TensorNodeId>,
    /// Precision-selected Linear weights (attention QKV/output, FFN, pooler,
    /// classifier), keyed by the layer prefix (the weight name minus `.weight`).
    pub(super) qw: HashMap<String, LinearWeights>,
    /// Raw f32 values for the `LayerNorm` weight/bias parameters (same data as the
    /// `w` leaves, shared via `Arc`), so the tape-free fused-layer path can call the
    /// `add_layer_norm` kernel directly without round-tripping through the session.
    pub(super) raw_params: HashMap<String, Arc<Vec<f32>>>,
    /// Contiguous encoder-layer count attested by the loaded weight set.
    pub(super) encoder_layers: usize,
    /// Autograd tape node count captured right after the persistent weights are
    /// loaded. Each forward pass truncates the tape back to this boundary to free
    /// that pass's intermediate activations, so the session does not grow
    /// unbounded across many rerank calls (a single long-doc forward can allocate
    /// ~25 MB attention tensors per layer; without truncation they would
    /// accumulate for the life of the process).
    pub(super) weights_boundary: usize,
}

impl Model {
    pub const fn encoder_layers(&self) -> usize {
        self.encoder_layers
    }

    fn g(&self, name: &str) -> SearchResult<TensorNodeId> {
        self.w
            .get(name)
            .copied()
            .ok_or_else(|| rerank_err("weights", format!("missing weight tensor {name}")))
    }

    /// `y = layer(x)` directly on raw f32 buffers (no tape node), using the
    /// precision selected by the producer at load time.
    fn linear_raw(&self, x: &[f32], m: usize, prefix: &str) -> SearchResult<Vec<f32>> {
        let q = self
            .qw
            .get(prefix)
            .ok_or_else(|| rerank_err("linear_raw", format!("missing linear weights {prefix}")))?;
        q.forward(x, m)
    }

    /// `layer_norm(a + b)` on raw f32 buffers (no tape node) — the tape-free
    /// counterpart of [`Self::add_ln`].
    fn add_ln_raw(&self, a: &[f32], b: &[f32], m: usize, prefix: &str) -> SearchResult<Vec<f32>> {
        let w = self
            .raw_params
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| rerank_err("add_ln_raw", format!("missing {prefix}.weight")))?;
        let bias = self
            .raw_params
            .get(&format!("{prefix}.bias"))
            .ok_or_else(|| rerank_err("add_ln_raw", format!("missing {prefix}.bias")))?;
        Ok(ft_api::add_layer_norm_forward_f32(
            a,
            b,
            Some(w),
            Some(bias),
            m,
            H,
            EPS_F32,
        ))
    }

    /// One encoder layer entirely on raw f32 buffers — no tape nodes or session
    /// round-trips. The loaded precision selects int8 or F32 linears; attention,
    /// add+LN and vectorized GELU use the same kernels as the tape path.
    /// Self-attention is per document (each `[lenₙ, H]` slice) via
    /// [`fused_attention`], so this path is for chunks where every doc is short. The
    /// `scratch` is reused across documents and layers to avoid per-call allocation.
    fn encoder_layer_raw(
        &self,
        emb: &[f32],
        total: usize,
        offsets: &[usize],
        lens: &[usize],
        p: &str,
        scale: f32,
        scratch: &mut AttnScratch,
    ) -> SearchResult<Vec<f32>> {
        // Fused QKV projection (batched over all the chunk's tokens).
        // Fused QKV projection output shape: [total, 3H].
        let qkv = self.linear_raw(emb, total, &format!("{p}.attention.self.qkv"))?;
        #[cfg(test)]
        certificate_trace::detailed("layer.qkv", &[total, 3 * H], &qkv);
        // Per-document self-attention written straight into one re-concatenated
        // [total, H] context (no per-doc temporary).
        let mut ctx = vec![0.0f32; total * H];
        for (&off, &len) in offsets.iter().zip(lens) {
            let qkv_doc = &qkv[off * 3 * H..(off + len) * 3 * H];
            fused_attention(
                scratch,
                qkv_doc,
                len,
                scale,
                &mut ctx[off * H..(off + len) * H],
            );
        }
        let attn = self.linear_raw(&ctx, total, &format!("{p}.attention.output.dense"))?;
        #[cfg(test)]
        {
            certificate_trace::detailed("layer.context", &[total, H], &ctx);
            certificate_trace::detailed("layer.attention_projection", &[total, H], &attn);
        }
        let emb = self.add_ln_raw(
            emb,
            &attn,
            total,
            &format!("{p}.attention.output.LayerNorm"),
        )?;
        // FFN: [total, H] -> [total, INTER] -> GELU -> [total, H].
        let mut inter = self.linear_raw(&emb, total, &format!("{p}.intermediate.dense"))?;
        #[cfg(test)]
        {
            certificate_trace::detailed("layer.attention_norm", &[total, H], &emb);
            certificate_trace::detailed("layer.ffn_projection", &[total, INTER], &inter);
        }
        debug_assert_eq!(inter.len(), total * INTER);
        fast_gelu_inplace(&mut inter);
        #[cfg(test)]
        certificate_trace::detailed("layer.gelu", &[total, INTER], &inter);
        let ffn = self.linear_raw(&inter, total, &format!("{p}.output.dense"))?;
        #[cfg(test)]
        certificate_trace::detailed("layer.ffn_output", &[total, H], &ffn);
        self.add_ln_raw(&emb, &ffn, total, &format!("{p}.output.LayerNorm"))
    }

    /// The FINAL encoder layer, computing ONLY each document's `[CLS]` row.
    ///
    /// The pooler reads the logit from `[CLS]` of the final layer alone, so the other
    /// tokens' layer outputs are dead. The QKV projection still runs over all tokens
    /// (CLS attends to every key, so K/V are needed in full), but attention is the
    /// CLS-query-only [`fused_attention_cls`], and the attn-out projection, residual
    /// add+LN, and the whole FFN then run on just `n_docs` rows (one `[CLS]` per doc)
    /// instead of `total`. Returns `[n_docs, H]` — exactly the pooler's input, so no
    /// post-hoc CLS gather is needed. Bit-exact to the `[CLS]` rows of
    /// [`encoder_layer_raw`] (same kernels, same reductions).
    fn encoder_layer_cls(
        &self,
        emb: &[f32],
        offsets: &[usize],
        lens: &[usize],
        total: usize,
        p: &str,
        scale: f32,
        scratch: &mut AttnScratch,
    ) -> SearchResult<Vec<f32>> {
        let n_docs = lens.len();
        // Fused QKV over all tokens (K/V needed in full for the CLS query's attention).
        // Fused QKV projection output shape: [total, 3H].
        let qkv = self.linear_raw(emb, total, &format!("{p}.attention.self.qkv"))?;
        // CLS-only self-attention per document written straight into a compact
        // [n_docs, H] context.
        let mut ctx = vec![0.0f32; n_docs * H];
        for (n, (&off, &len)) in offsets.iter().zip(lens).enumerate() {
            let qkv_doc = &qkv[off * 3 * H..(off + len) * 3 * H];
            fused_attention_cls(scratch, qkv_doc, len, scale, &mut ctx[n * H..(n + 1) * H]);
        }
        let attn = self.linear_raw(&ctx, n_docs, &format!("{p}.attention.output.dense"))?;
        // Residual is the input layer's CLS rows (the first token of each doc).
        let mut emb_cls = vec![0.0f32; n_docs * H];
        for (n, &off) in offsets.iter().enumerate() {
            emb_cls[n * H..(n + 1) * H].copy_from_slice(&emb[off * H..off * H + H]);
        }
        let emb_cls = self.add_ln_raw(
            &emb_cls,
            &attn,
            n_docs,
            &format!("{p}.attention.output.LayerNorm"),
        )?;
        // FFN on just the CLS rows: [n_docs, H] -> [n_docs, INTER] -> GELU -> [n_docs, H].
        let mut inter = self.linear_raw(&emb_cls, n_docs, &format!("{p}.intermediate.dense"))?;
        debug_assert_eq!(inter.len(), n_docs * INTER);
        fast_gelu_inplace(&mut inter);
        let ffn = self.linear_raw(&inter, n_docs, &format!("{p}.output.dense"))?;
        self.add_ln_raw(&emb_cls, &ffn, n_docs, &format!("{p}.output.LayerNorm"))
    }

    /// `y = x @ Wᵀ + b` for a matrix node, using the loaded linear precision.
    /// Reranking retains dynamic int8 activation quantization and static int8
    /// weight quantization. Either path returns a detached f32 node.
    fn linear(&mut self, x: TensorNodeId, prefix: &str) -> SearchResult<TensorNodeId> {
        let q = self
            .qw
            .get(prefix)
            .ok_or_else(|| rerank_err("linear", format!("missing linear weights {prefix}")))?;
        let q = match q {
            LinearWeights::Int8(q) => q,
            LinearWeights::F32 { out, in_, .. } => {
                let values = self
                    .s
                    .tensor_values_f32_borrowed(x)
                    .map_err(|e| rerank_err("linear.f32.input", e))?;
                let rows = values.len() / in_;
                let shape = vec![rows, *out];
                let result = q.forward(values, rows)?;
                return self
                    .s
                    .tensor_variable_f32(result, shape, false)
                    .map_err(|e| rerank_err("linear.f32.output", e));
            }
        };
        // Clone the Arcs + copy the dims so the `&self.qw` borrow ends before the
        // `&mut self.s` borrow below.
        let w_i8 = Arc::clone(&q.w_i8);
        let w_scales = Arc::clone(&q.w_scales);
        let bias = Arc::clone(&q.bias);
        let (out, in_, packed) = (q.out, q.in_, q.packed);
        if packed {
            self.s
                .tensor_linear_int8_dynamic_prepacked(x, &w_i8, &w_scales, out, in_, Some(&bias))
                .map_err(|e| rerank_err("linear.int8.packed", e))
        } else {
            self.s
                .tensor_linear_int8_dynamic(x, &w_i8, &w_scales, out, in_, Some(&bias))
                .map_err(|e| rerank_err("linear.int8", e))
        }
    }

    fn idx(&mut self, vals: &[i64]) -> SearchResult<TensorNodeId> {
        let f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        self.s
            .tensor_variable(f, vec![vals.len()], false)
            .map_err(|e| rerank_err("index_tensor", e))
    }

    /// Fused residual-add + `LayerNorm` `layer_norm(a + b)` (the "add & norm") in one
    /// op, so the residual sum is never materialized as its own tensor (2 per layer).
    fn add_ln(
        &mut self,
        a: TensorNodeId,
        b: TensorNodeId,
        prefix: &str,
    ) -> SearchResult<TensorNodeId> {
        let w = self.g(&format!("{prefix}.weight"))?;
        let bias = self.g(&format!("{prefix}.bias"))?;
        self.s
            .tensor_add_layer_norm(a, b, H, Some(w), Some(bias), EPS)
            .map_err(|e| rerank_err("add_layer_norm", e))
    }

    /// Exact GELU via the vectorized [`fast_gelu`] (A–S erf), replacing the
    /// `tensor_gelu` tape op's scalar libm `erff`. Round-trips through f32 values +
    /// a fresh leaf, consistent with the int8 linear (which already returns a
    /// detached f32 leaf), so there is no live autograd graph to preserve.
    fn gelu(&mut self, inter: TensorNodeId) -> SearchResult<TensorNodeId> {
        // In-place: `inter` is a single-use intermediate (only the next FFN linear
        // reads it), so rewrite its storage rather than round-tripping through a
        // fresh leaf — kills the per-layer extract+reinsert of the wide `[m, 1536]`.
        let slice = self
            .s
            .tensor_values_f32_mut(inter)
            .map_err(|e| rerank_err("ffn.gelu_mut", e))?;
        fast_gelu_inplace(slice);
        Ok(inter)
    }

    /// [S, H] -> [NH, S, HD]
    fn heads(&mut self, x: TensorNodeId, s_len: usize) -> SearchResult<TensorNodeId> {
        let r = self
            .s
            .tensor_reshape(x, vec![s_len, NH, HD])
            .map_err(|e| rerank_err("heads.reshape", e))?;
        self.s
            .tensor_transpose(r, 0, 1)
            .map_err(|e| rerank_err("heads.transpose", e))
    }

    /// Multi-head self-attention for one sequence: given the per-token `q`/`k`/`v`
    /// projections (each `[s_len, H]`) returns the context `[s_len, H]`. Shared by
    /// the single-pair [`Self::forward`] and the batched [`Self::forward_batch`]
    /// (which calls it on each document's contiguous token slice), so the two paths
    /// compute identical attention — the batched path is parity-exact.
    ///
    /// Hybrid: for short/medium sequences the [`fused_attention`] kernel wins (it
    /// avoids the heads-transpose materialization and the per-head `gemm` launch
    /// overhead that dominate there); for long sequences the `bmm` path wins (the
    /// `gemm` crate's cache-blocking amortizes the K/V re-reads that the naive fused
    /// loop repeats per query). `FUSED_ATTN_MAX_SEQ` is the measured crossover.
    /// Short/medium-sequence attention: `qkv` is the fused QKV linear output
    /// `[s_len, 3H]`; runs the [`fused_attention`] kernel over it (reading borrowed,
    /// writing one fresh ctx leaf). Tape-path reference for the single-pair
    /// [`Self::forward`]; production goes through the tape-free `encoder_layer_raw`.
    #[cfg(test)]
    fn attn_fused(
        &mut self,
        qkv: TensorNodeId,
        s_len: usize,
        scale: f32,
    ) -> SearchResult<TensorNodeId> {
        let ctx_vals = {
            let qkv_v = self
                .s
                .tensor_values_f32_borrowed(qkv)
                .map_err(|e| rerank_err("attn.qkv_vals", e))?;
            let mut scratch = AttnScratch::default();
            let mut ctx = vec![0.0f32; s_len * H];
            fused_attention(&mut scratch, qkv_v, s_len, scale, &mut ctx);
            ctx
        };
        self.s
            .tensor_variable_f32(ctx_vals, vec![s_len, H], false)
            .map_err(|e| rerank_err("attn.ctx", e))
    }

    /// Long-sequence attention: separate `q`/`k`/`v` `[s_len, H]` through the batched
    /// f32 `bmm` with in-place fused-scale softmax (the `gemm` crate's cache-blocking
    /// wins past `FUSED_ATTN_MAX_SEQ`).
    fn attn_bmm(
        &mut self,
        q: TensorNodeId,
        k: TensorNodeId,
        v: TensorNodeId,
        s_len: usize,
        scale: f32,
    ) -> SearchResult<TensorNodeId> {
        let q = self.heads(q, s_len)?;
        let k = self.heads(k, s_len)?;
        let v = self.heads(v, s_len)?;
        let kt = self
            .s
            .tensor_transpose(k, 1, 2)
            .map_err(|e| rerank_err("attn.kt", e))?; // [NH, HD, S]
        let scores = self
            .s
            .tensor_bmm(q, kt)
            .map_err(|e| rerank_err("attn.qk", e))?;
        {
            let slice = self
                .s
                .tensor_values_f32_mut(scores)
                .map_err(|e| rerank_err("attn.softmax_mut", e))?;
            fast_softmax_inplace(slice, NH * s_len, s_len, scale);
        }
        let ctx = self
            .s
            .tensor_bmm(scores, v)
            .map_err(|e| rerank_err("attn.ctx", e))?;
        let ctx = self
            .s
            .tensor_transpose(ctx, 0, 1)
            .map_err(|e| rerank_err("attn.ctx_t", e))?;
        self.s
            .tensor_reshape(ctx, vec![s_len, H])
            .map_err(|e| rerank_err("attn.ctx_reshape", e))
    }

    /// Self-attention for one document's `[s_len, H]` activation `emb`: routes to the
    /// fused-QKV + fused-attention kernel for short/medium sequences (one int8 GEMM
    /// for Q/K/V, no transpose/per-head-launch overhead), or the separate-QKV + bmm
    /// path for long sequences. Returns the context `[s_len, H]`. Used by the
    /// single-pair [`Self::forward`] reference; production uses `encoder_layer_raw`.
    #[cfg(test)]
    fn attention(
        &mut self,
        emb: TensorNodeId,
        p: &str,
        s_len: usize,
        scale: f32,
    ) -> SearchResult<TensorNodeId> {
        if s_len <= FUSED_ATTN_MAX_SEQ {
            let qkv = self.linear(emb, &format!("{p}.attention.self.qkv"))?;
            self.attn_fused(qkv, s_len, scale)
        } else {
            let q = self.linear(emb, &format!("{p}.attention.self.query"))?;
            let k = self.linear(emb, &format!("{p}.attention.self.key"))?;
            let v = self.linear(emb, &format!("{p}.attention.self.value"))?;
            self.attn_bmm(q, k, v, s_len, scale)
        }
    }

    /// Single-pair forward pass (batch = 1). Returns the raw logit. Retained as the
    /// per-document reference that [`Self::forward_batch`] is checked against
    /// (`forward_batch_matches_per_doc`); production reranking always goes through
    /// the batched path (which handles a one-document batch with negligible
    /// overhead), so this is `#[cfg(test)]`-only.
    ///
    /// Runs entirely in f32: weights are f32 leaves and every op preserves f32, so
    /// `embedding/matmul/softmax/layer_norm` all stay in the f32 kernels.
    #[cfg(test)]
    pub(super) fn forward(&mut self, ids: &[i64], typ: &[i64]) -> SearchResult<f32> {
        let s_len = ids.len();
        // embeddings: word + position + token_type, then LayerNorm.
        // `index_select(weight, dim=0, indices)` is the embedding lookup; unlike
        // `tensor_embedding` it preserves the f32 weight dtype (frankentorch-40i).
        let id_t = self.idx(ids)?;
        let pos: Vec<i64> = (0..s_len)
            .map(|i| index_to_i64(i, "forward.position"))
            .collect::<SearchResult<_>>()?;
        let pos_t = self.idx(&pos)?;
        let typ_t = self.idx(typ)?;
        let we = self.g("bert.embeddings.word_embeddings.weight")?;
        let pe = self.g("bert.embeddings.position_embeddings.weight")?;
        let te = self.g("bert.embeddings.token_type_embeddings.weight")?;
        let e_word = self
            .s
            .tensor_index_select(we, 0, id_t)
            .map_err(|e| rerank_err("embed.word", e))?;
        let e_pos = self
            .s
            .tensor_index_select(pe, 0, pos_t)
            .map_err(|e| rerank_err("embed.pos", e))?;
        let e_typ = self
            .s
            .tensor_index_select(te, 0, typ_t)
            .map_err(|e| rerank_err("embed.type", e))?;
        let emb_wp = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        // Fuse the third-embedding add with the embedding LayerNorm.
        let mut emb = self.add_ln(emb_wp, e_typ, "bert.embeddings.LayerNorm")?;

        let scale = ATTN_SCALE_F32;
        for i in 0..self.encoder_layers {
            let p = format!("bert.encoder.layer.{i}");
            // self-attention (fused-QKV + fused kernel, or separate-QKV + bmm by len)
            let ctx = self.attention(emb, &p, s_len, scale)?;
            let attn = self.linear(ctx, &format!("{p}.attention.output.dense"))?;
            emb = self.add_ln(emb, attn, &format!("{p}.attention.output.LayerNorm"))?;
            // feed-forward
            let inter = self.linear(emb, &format!("{p}.intermediate.dense"))?;
            let inter = self.gelu(inter)?;
            let ffn = self.linear(inter, &format!("{p}.output.dense"))?;
            emb = self.add_ln(emb, ffn, &format!("{p}.output.LayerNorm"))?;
        }
        // pooler on [CLS] (row 0) + classifier
        let cls = self
            .s
            .tensor_narrow(emb, 0, 0, 1)
            .map_err(|e| rerank_err("pooler.narrow", e))?; // [1, H]
        let pooled = self.linear(cls, "bert.pooler.dense")?;
        let pooled = self
            .s
            .tensor_tanh(pooled)
            .map_err(|e| rerank_err("pooler.tanh", e))?;
        let logit_t = self.linear(pooled, "classifier")?; // [1, 1]
        let vals = self
            .s
            .tensor_values_f32(logit_t)
            .map_err(|e| rerank_err("classifier.values", e))?;
        let logit = vals
            .first()
            .copied()
            .ok_or_else(|| rerank_err("classifier", "empty logit output"))?;
        // Free this forward pass's intermediate tape nodes (everything created
        // after the weights), keeping the loaded parameters, so the session's
        // arena does not grow unbounded across rerank calls.
        self.s.truncate_autograd_graph(self.weights_boundary);
        Ok(logit)
    }

    /// Batched (multi-document) forward — the throughput lever. Returns one raw
    /// logit per document, in input order.
    ///
    /// Layout is **varlen**: the documents' tokens are concatenated end-to-end
    /// (NO padding, NO mask) into one `[Σlenₙ, H]` activation. Every per-token op
    /// (the int8 Linears, `LayerNorm`, GELU, residuals) runs once over the whole
    /// `Σlenₙ` rows, so each statically-quantized weight is loaded once and reused
    /// across all the documents' tokens instead of being re-streamed per document
    /// — that weight-reuse is what lifts the int8 GEMM toward peak and beats a
    /// per-document runtime on multi-doc rerank throughput. Self-attention is the
    /// only op that must stay within a document, so it runs per-doc on each
    /// document's contiguous token slice via the shared [`Self::attn_block`] — so
    /// every document gets byte-identical computation to [`Self::forward`] and the
    /// batched logits are parity-exact, not approximate.
    ///
    /// The caller chunks the batch so `Σlenₙ` stays bounded (attention
    /// intermediates for the whole chunk co-exist on the tape until the final
    /// truncate); see [`super::MAX_BATCH_TOKENS`].
    pub(super) fn forward_batch(
        &mut self,
        batch: &[(Vec<i64>, Vec<i64>)],
    ) -> SearchResult<Vec<f32>> {
        let n_docs = batch.len();
        let lens: Vec<usize> = batch.iter().map(|(ids, _)| ids.len()).collect();
        let total: usize = lens.iter().sum();
        if total == 0 {
            return Ok(vec![0.0; n_docs]);
        }
        let mut offsets = Vec::with_capacity(n_docs);
        {
            let mut o = 0usize;
            for &l in &lens {
                offsets.push(o);
                o += l;
            }
        }
        // Flat token / position / type ids over the concatenated documents. Each
        // document's positions restart at 0 (positions are intra-document).
        let mut ids_flat = Vec::with_capacity(total);
        let mut pos_flat = Vec::with_capacity(total);
        let mut typ_flat = Vec::with_capacity(total);
        for (ids, typ) in batch {
            for (i, (&id, &t)) in ids.iter().zip(typ.iter()).enumerate() {
                ids_flat.push(id);
                pos_flat.push(index_to_i64(i, "forward_batch.position")?);
                typ_flat.push(t);
            }
        }
        // Embeddings → [total, H].
        let id_t = self.idx(&ids_flat)?;
        let pos_t = self.idx(&pos_flat)?;
        let typ_t = self.idx(&typ_flat)?;
        let we = self.g("bert.embeddings.word_embeddings.weight")?;
        let pe = self.g("bert.embeddings.position_embeddings.weight")?;
        let te = self.g("bert.embeddings.token_type_embeddings.weight")?;
        let e_word = self
            .s
            .tensor_index_select(we, 0, id_t)
            .map_err(|e| rerank_err("embed.word", e))?;
        let e_pos = self
            .s
            .tensor_index_select(pe, 0, pos_t)
            .map_err(|e| rerank_err("embed.pos", e))?;
        let e_typ = self
            .s
            .tensor_index_select(te, 0, typ_t)
            .map_err(|e| rerank_err("embed.type", e))?;
        let emb_wp = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        // Fuse the third-embedding add with the embedding LayerNorm.
        let mut emb = self.add_ln(emb_wp, e_typ, "bert.embeddings.LayerNorm")?;

        let scale = ATTN_SCALE_F32;
        // True once the final layer has already collapsed the activation to one
        // `[CLS]` row per document (`[n_docs, H]`), so the pooler gathers identity
        // rows instead of the per-doc offsets.
        let cls_prepacked = if lens.iter().all(|&l| l <= FUSED_ATTN_MAX_SEQ) {
            // Tape-free fused-layer path (every doc short): extract the activation
            // once, run the encoder layers entirely on raw f32 buffers through the
            // SAME optimized kernels (fused QKV + fused attention + add&norm +
            // vectorized GELU) with no per-op tape-node creation / leaf allocation /
            // truncation, then reinsert once for the pooler. Bit-identical to the
            // tape path. The FINAL layer computes only each doc's `[CLS]` row (the
            // sole token the pooler reads), skipping ~S× of its attn-out + FFN work.
            // One attention scratch reused across every layer (and document) of this
            // chunk's forward, so the head-major repacks + the two bmm outputs are
            // allocated once instead of per attention call.
            let mut scratch = AttnScratch::default();
            let mut emb_vals = self
                .s
                .tensor_values_f32(emb)
                .map_err(|e| rerank_err("batch.emb_extract", e))?;
            for i in 0..self.encoder_layers - 1 {
                let p = format!("bert.encoder.layer.{i}");
                emb_vals = self.encoder_layer_raw(
                    &emb_vals,
                    total,
                    &offsets,
                    &lens,
                    &p,
                    scale,
                    &mut scratch,
                )?;
            }
            let p_last = format!("bert.encoder.layer.{}", self.encoder_layers - 1);
            let cls_vals = self.encoder_layer_cls(
                &emb_vals,
                &offsets,
                &lens,
                total,
                &p_last,
                scale,
                &mut scratch,
            )?;
            emb = self
                .s
                .tensor_variable_f32(cls_vals, vec![n_docs, H], false)
                .map_err(|e| rerank_err("batch.emb_reinsert", e))?;
            true
        } else {
            // Long-document path: separate-QKV + cache-blocking bmm attention through
            // the tape (a long doc in the chunk; rare).
            for i in 0..self.encoder_layers {
                let p = format!("bert.encoder.layer.{i}");
                let q = self.linear(emb, &format!("{p}.attention.self.query"))?;
                let k = self.linear(emb, &format!("{p}.attention.self.key"))?;
                let v = self.linear(emb, &format!("{p}.attention.self.value"))?;
                let mut ctx_parts = Vec::with_capacity(n_docs);
                for n in 0..n_docs {
                    let (off, len) = (offsets[n], lens[n]);
                    let qn = self
                        .s
                        .tensor_narrow(q, 0, off, len)
                        .map_err(|e| rerank_err("batch.q_narrow", e))?;
                    let kn = self
                        .s
                        .tensor_narrow(k, 0, off, len)
                        .map_err(|e| rerank_err("batch.k_narrow", e))?;
                    let vn = self
                        .s
                        .tensor_narrow(v, 0, off, len)
                        .map_err(|e| rerank_err("batch.v_narrow", e))?;
                    ctx_parts.push(self.attn_bmm(qn, kn, vn, len, scale)?);
                }
                let ctx = self
                    .s
                    .tensor_cat(&ctx_parts, 0)
                    .map_err(|e| rerank_err("batch.ctx_cat", e))?; // [total, H]
                let attn = self.linear(ctx, &format!("{p}.attention.output.dense"))?;
                emb = self.add_ln(emb, attn, &format!("{p}.attention.output.LayerNorm"))?;
                let inter = self.linear(emb, &format!("{p}.intermediate.dense"))?;
                let inter = self.gelu(inter)?;
                let ffn = self.linear(inter, &format!("{p}.output.dense"))?;
                emb = self.add_ln(emb, ffn, &format!("{p}.output.LayerNorm"))?;
            }
            false
        };
        // Pooler on each document's [CLS] row → [N, H]. When the final layer already
        // emitted one CLS row per doc (`cls_prepacked`), those are rows 0..n_docs;
        // otherwise CLS is the first token of each doc, row `offsets[n]`.
        let cls_idx: Vec<i64> = if cls_prepacked {
            (0..n_docs)
                .map(|i| index_to_i64(i, "forward_batch.cls_idx"))
                .collect::<SearchResult<_>>()?
        } else {
            offsets
                .iter()
                .map(|&o| index_to_i64(o, "forward_batch.cls_offset"))
                .collect::<SearchResult<_>>()?
        };
        let cls_t = self.idx(&cls_idx)?;
        let cls = self
            .s
            .tensor_index_select(emb, 0, cls_t)
            .map_err(|e| rerank_err("batch.cls_gather", e))?; // [N, H]
        let pooled = self.linear(cls, "bert.pooler.dense")?;
        let pooled = self
            .s
            .tensor_tanh(pooled)
            .map_err(|e| rerank_err("pooler.tanh", e))?;
        let logit_t = self.linear(pooled, "classifier")?; // [N, 1]
        let vals = self
            .s
            .tensor_values_f32(logit_t)
            .map_err(|e| rerank_err("classifier.values", e))?;
        self.s.truncate_autograd_graph(self.weights_boundary);
        if vals.len() != n_docs {
            return Err(rerank_err(
                "classifier",
                format!("expected {n_docs} logits, got {}", vals.len()),
            ));
        }
        Ok(vals)
    }

    /// Sentence-embedding forward (the embedder head). Runs the SAME shared BERT
    /// encoder as the reranker over each input's tokens — identical embeddings build
    /// and `encoder_layer_raw` (same int8/SIMD kernels) — then replaces the reranker's
    /// `[CLS]` pooler + classifier with **mean-pooling over every token + L2-normalize**
    /// (the `sentence-transformers/all-MiniLM-L6-v2` head). Token-type ids are all 0
    /// (a single text, no query/doc split). Returns one `[H]` unit vector per input.
    ///
    /// Every input is ≤ `DEFAULT_MAX_LENGTH` (callers truncate at tokenization), so the
    /// whole batch goes through the tape-free raw path; there is no CLS-only shortcut
    /// because mean-pooling needs every token's final hidden state.
    pub fn embed_forward(&mut self, batch: &[Vec<i64>]) -> SearchResult<Vec<Vec<f32>>> {
        let n_docs = batch.len();
        let lens: Vec<usize> = batch.iter().map(Vec::len).collect();
        let total: usize = lens.iter().sum();
        if total == 0 {
            return Ok(vec![vec![0.0; H]; n_docs]);
        }
        let mut offsets = Vec::with_capacity(n_docs);
        {
            let mut o = 0usize;
            for &l in &lens {
                offsets.push(o);
                o += l;
            }
        }
        // Flat token / position / type ids over the concatenated inputs. Positions
        // restart at 0 per input; token-type is always 0 (single text).
        let mut ids_flat = Vec::with_capacity(total);
        let mut pos_flat = Vec::with_capacity(total);
        let mut typ_flat = Vec::with_capacity(total);
        for ids in batch {
            for (i, &id) in ids.iter().enumerate() {
                ids_flat.push(id);
                pos_flat.push(index_to_i64(i, "embed_forward.position")?);
                typ_flat.push(0i64);
            }
        }
        // Embeddings → [total, H]: word + position + token_type, then LayerNorm.
        #[cfg(test)]
        {
            certificate_trace::ids("embedding.token_ids", &ids_flat);
            certificate_trace::ids("embedding.position_ids", &pos_flat);
            certificate_trace::ids("embedding.type_ids", &typ_flat);
        }
        let id_t = self.idx(&ids_flat)?;
        let pos_t = self.idx(&pos_flat)?;
        let typ_t = self.idx(&typ_flat)?;
        let we = self.g("bert.embeddings.word_embeddings.weight")?;
        let pe = self.g("bert.embeddings.position_embeddings.weight")?;
        let te = self.g("bert.embeddings.token_type_embeddings.weight")?;
        let e_word = self
            .s
            .tensor_index_select(we, 0, id_t)
            .map_err(|e| rerank_err("embed.word", e))?;
        let e_pos = self
            .s
            .tensor_index_select(pe, 0, pos_t)
            .map_err(|e| rerank_err("embed.pos", e))?;
        let e_typ = self
            .s
            .tensor_index_select(te, 0, typ_t)
            .map_err(|e| rerank_err("embed.type", e))?;
        let emb_wp = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        let emb = self.add_ln(emb_wp, e_typ, "bert.embeddings.LayerNorm")?;

        // Encoder: every attested layer on raw f32 buffers (mean-pooling needs every token's
        // final hidden state, so no CLS-only last layer). Same kernels as the reranker.
        let scale = ATTN_SCALE_F32;
        let mut scratch = AttnScratch::default();
        let mut emb_vals = self
            .s
            .tensor_values_f32(emb)
            .map_err(|e| rerank_err("embed.extract", e))?;
        #[cfg(test)]
        {
            if certificate_trace::active() {
                for (label, node) in [
                    ("embedding.word", e_word),
                    ("embedding.position", e_pos),
                    ("embedding.type", e_typ),
                    ("embedding.word_position_sum", emb_wp),
                ] {
                    let values = self
                        .s
                        .tensor_values_f32(node)
                        .map_err(|e| rerank_err("embed.trace", e))?;
                    certificate_trace::floats(label, &[total, H], &values);
                }
                certificate_trace::floats("embedding.normalized", &[total, H], &emb_vals);
            }
        }
        for i in 0..self.encoder_layers {
            let p = format!("bert.encoder.layer.{i}");
            #[cfg(test)]
            certificate_trace::detail(true);
            emb_vals =
                self.encoder_layer_raw(&emb_vals, total, &offsets, &lens, &p, scale, &mut scratch)?;
            #[cfg(test)]
            certificate_trace::floats(&format!("{p}.output"), &[total, H], &emb_vals);
        }
        #[cfg(test)]
        certificate_trace::detail(false);
        self.s.truncate_autograd_graph(self.weights_boundary);

        // Mean-pool each input's token rows → [H], then L2-normalize to a unit vector.
        let mut out = Vec::with_capacity(n_docs);
        for (&off, &len) in offsets.iter().zip(&lens) {
            let mut acc = vec![0.0f32; H];
            if len > 0 {
                let doc = &emb_vals[off * H..(off + len) * H];
                for t in 0..len {
                    let row = &doc[t * H..t * H + H];
                    for (a, &r) in acc.iter_mut().zip(row) {
                        *a += r;
                    }
                }
                let inv = 1.0 / len as f32;
                for a in &mut acc {
                    *a *= inv;
                }
            }
            let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
            #[cfg(test)]
            certificate_trace::floats("embedding.mean_pool", &[H], &acc);
            if norm > 0.0 {
                let inv = 1.0 / norm;
                for a in &mut acc {
                    *a *= inv;
                }
            }
            #[cfg(test)]
            certificate_trace::floats("embedding.output", &[H], &acc);
            out.push(acc);
        }
        Ok(out)
    }
}
