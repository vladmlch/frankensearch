use std::io::Read;
use std::path::Path;

use frankensearch_core::{SearchError, SearchResult, VectorHit};

use crate::VectorIndex;

use super::types::{
    AnnSearchStats, DIST_DOT_MAX_DIMENSION, DistDotBudget, HNSW_DEFAULT_MAX_LAYER, HnswConfig,
    HnswMeta, HnswSidecarDigest,
};

pub(super) fn restore_dist_dot_score(distance: f32, budget: DistDotBudget) -> SearchResult<f32> {
    let restored_score = (1.0 - distance) / budget.radius_squared;
    let score_envelope = -1.0 - budget.score_tolerance..=1.0 + budget.score_tolerance;
    if !restored_score.is_finite() || !score_envelope.contains(&restored_score) {
        return Err(SearchError::SubsystemError {
            subsystem: "hnsw",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "DistDot returned distance {distance} (restored score {restored_score}); \
                     expected a score within [{}, {}]",
                    score_envelope.start(),
                    score_envelope.end()
                ),
            )),
        });
    }
    Ok(restored_score.clamp(-1.0, 1.0))
}

/// FNV-1a 64-bit. Chosen because it is deterministic across processes (unlike
/// `ahash`) and stdlib (`DefaultHasher` is randomized + deprecated for
/// persistence), keeps the fingerprint dependency-free, and is plenty for an
/// integrity-style check — we only need collision resistance against a small
/// number of accidental byte-level edits, not adversarial inputs.
const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    h
}

pub(super) fn fingerprint_doc_ids(doc_ids: &[String]) -> u64 {
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &doc_ids.len().to_le_bytes());
    for (index, doc_id) in doc_ids.iter().enumerate() {
        h = fnv1a_update(h, &index.to_le_bytes());
        h = fnv1a_update(h, &doc_id.len().to_le_bytes());
        h = fnv1a_update(h, doc_id.as_bytes());
    }
    h
}

pub(super) fn fingerprint_hnsw_sidecar(path: &Path) -> SearchResult<HnswSidecarDigest> {
    let mut file = std::fs::File::open(path).map_err(|error| {
        SearchError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to open HNSW sidecar '{}' for READY receipt: {error}",
                path.display()
            ),
        ))
    })?;
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    let mut byte_len = 0_u64;
    let mut fingerprint = FNV_OFFSET_BASIS_64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to read HNSW sidecar '{}' for READY receipt: {error}",
                    path.display()
                ),
            ))
        })?;
        if read == 0 {
            break;
        }
        let read_u64 = u64::try_from(read).map_err(|_| {
            SearchError::Io(std::io::Error::other(
                "HNSW sidecar read length exceeds u64",
            ))
        })?;
        byte_len = byte_len.checked_add(read_u64).ok_or_else(|| {
            SearchError::Io(std::io::Error::other("HNSW sidecar length exceeds u64"))
        })?;
        fingerprint = fnv1a_update(fingerprint, &buffer[..read]);
    }
    Ok(HnswSidecarDigest {
        byte_len,
        fnv1a64: fingerprint,
    })
}

/// Feed an `&[f32]` into the FNV-1a state in little-endian byte order without
/// any unsafe reinterpretation (this module holds no `unsafe`; the crate's
/// only `unsafe` sites are the SIMD kernels and the `memmap2` read path, each
/// under a per-site `#[allow(unsafe_code)]` beneath the workspace `deny`).
/// One `to_le_bytes()` per element compiles to a tight loop.
#[inline]
fn fnv1a_update_f32(mut h: u64, vec: &[f32]) -> u64 {
    for &f in vec {
        h = fnv1a_update(h, &f.to_bits().to_le_bytes());
    }
    h
}

/// Compute the persistence fingerprint from the (raw, un-normalized) vectors
/// and doc IDs used to build the graph.
///
/// Every vector contributes in live-row order. Doc IDs are mixed in alongside
/// their vectors, so either a vector edit or a doc-id permutation changes the
/// digest. The output is stored in the native metadata sidecar and re-derived
/// at load time by [`fingerprint_vector_index_positions`] against the live
/// `VectorIndex`; a mismatch means "doc IDs match but the underlying vector
/// bytes were silently swapped" → reject the persisted graph.
pub(super) fn fingerprint_vectors(doc_ids: &[String], vectors: &[Vec<f32>]) -> u64 {
    // Mix length so a truncated-but-prefix-matching index doesn't collide.
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &(doc_ids.len() as u64).to_le_bytes());
    for (i, (doc_id, vector)) in doc_ids.iter().zip(vectors).enumerate() {
        h = fnv1a_update(h, &(i as u64).to_le_bytes());
        h = fnv1a_update(h, doc_id.as_bytes());
        h = fnv1a_update_f32(h, vector);
    }
    h
}

/// Compute the fingerprint from a live `VectorIndex`, matching the layout
/// [`fingerprint_vectors`] produced at build time.
///
/// `expected_len` and `expected_dim` come from the persisted metadata. If the
/// live index has fewer live records than the persisted graph, the digest
/// will not match and the caller falls back to a rebuild — which is the right
/// behavior.
#[cfg(test)]
pub(super) fn fingerprint_live_vector_index(
    index: &VectorIndex,
    expected_len: usize,
    expected_dim: usize,
) -> SearchResult<u64> {
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &(expected_len as u64).to_le_bytes());

    // Walk live records (tombstones excluded), in row order — same iteration
    // order as `build_from_vector_index`.
    let mut live_idx = 0_usize;
    for raw in 0..index.record_count() {
        if index.is_deleted(raw) {
            continue;
        }
        if live_idx >= expected_len {
            // Live index has more records than the persisted graph.
            break;
        }
        let doc_id = index.doc_id_at(raw)?;
        h = fnv1a_update(h, &(live_idx as u64).to_le_bytes());
        h = fnv1a_update(h, doc_id.as_bytes());

        let vec = index.vector_at_f32(raw)?;
        if vec.len() != expected_dim {
            // Dimension drift — perturb the digest so the caller rejects and
            // rebuilds rather than asserting.
            return Ok(h.wrapping_add(1));
        }
        h = fnv1a_update_f32(h, &vec);
        live_idx += 1;
    }
    Ok(h)
}

/// Recompute a graph fingerprint from its original physical source rows.
///
/// Unlike a live-row scan, this deliberately includes rows
/// that were tombstoned after the graph was built. Soft deletion is a valid
/// in-process state transition: native candidates for those rows are filtered,
/// while an exact underfill repair scans the source's current live set. A row
/// move, document-identity change, or vector mutation still changes the digest
/// and fails closed.
pub(super) fn fingerprint_vector_index_positions(
    index: &VectorIndex,
    positions: &[u32],
    expected_doc_ids: &[String],
    expected_dim: usize,
) -> SearchResult<u64> {
    if positions.len() != expected_doc_ids.len() {
        return Err(ann_corrupted(
            &index.path,
            format!(
                "HNSW source map has {} rows for {} document identities",
                positions.len(),
                expected_doc_ids.len()
            ),
        ));
    }
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &(positions.len() as u64).to_le_bytes());
    for (logical_index, (&physical_position, expected_doc_id)) in
        positions.iter().zip(expected_doc_ids).enumerate()
    {
        let physical_position = usize::try_from(physical_position).map_err(|_| {
            ann_corrupted(
                &index.path,
                format!("HNSW source row {physical_position} does not fit usize"),
            )
        })?;
        if physical_position >= index.record_count() {
            return Err(ann_corrupted(
                &index.path,
                format!(
                    "HNSW source row {physical_position} exceeds {} physical records",
                    index.record_count()
                ),
            ));
        }
        if index.doc_id_at(physical_position)? != expected_doc_id {
            return Err(ann_corrupted(
                &index.path,
                format!("HNSW source row {physical_position} has a different document identity"),
            ));
        }
        let vector = index.vector_at_f32(physical_position)?;
        if vector.len() != expected_dim {
            return Err(SearchError::DimensionMismatch {
                expected: expected_dim,
                found: vector.len(),
            });
        }
        h = fnv1a_update(h, &(logical_index as u64).to_le_bytes());
        h = fnv1a_update(h, expected_doc_id.as_bytes());
        h = fnv1a_update_f32(h, &vector);
    }
    Ok(h)
}

/// Verify the graph's persisted row map covers every live source row exactly
/// once and binds the expected document identity. Same semantics as the public
/// `matches_vector_index` but doesn't need a constructed `HnswIndex`, so we
/// can check it before paying for the native graph load.
pub(super) fn meta_matches_live_doc_ids(
    meta: &HnswMeta,
    index: &VectorIndex,
) -> SearchResult<bool> {
    if meta.dimension != index.dimension() {
        return Ok(false);
    }
    source_map_matches_live_rows(
        index,
        &meta.doc_ids,
        &meta.source_positions,
        meta.source_record_count,
    )
}

pub(super) fn source_map_matches_live_rows(
    index: &VectorIndex,
    doc_ids: &[String],
    positions: &[u32],
    source_record_count: usize,
) -> SearchResult<bool> {
    if source_record_count != index.record_count() || positions.len() != doc_ids.len() {
        return Ok(false);
    }
    let mut covered = vec![false; source_record_count];
    for (id, &position) in doc_ids.iter().zip(positions) {
        let row = position as usize;
        let Some(seen) = covered.get_mut(row) else {
            return Ok(false);
        };
        if *seen || index.is_deleted(row) || id != index.doc_id_at(row)? {
            return Ok(false);
        }
        *seen = true;
    }
    Ok(covered
        .iter()
        .enumerate()
        .all(|(row, &seen)| seen != index.is_deleted(row)))
}

pub(super) fn fingerprint_source_map(positions: &[u32], source_record_count: usize) -> u64 {
    let mut fingerprint = fnv1a_update(
        FNV_OFFSET_BASIS_64,
        &(source_record_count as u64).to_le_bytes(),
    );
    for position in positions {
        fingerprint = fnv1a_update(fingerprint, &position.to_le_bytes());
    }
    fingerprint
}

pub(super) fn validate_config(config: HnswConfig) -> SearchResult<()> {
    if config.m == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_m must be greater than zero".to_owned(),
        });
    }
    if config.m > usize::from(u8::MAX) {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: config.m.to_string(),
            reason: format!("hnsw_m must be <= {}", u8::MAX),
        });
    }
    if config.ef_construction == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_ef_construction".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_ef_construction must be greater than zero".to_owned(),
        });
    }
    if config.ef_search == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_ef_search".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_ef_search must be greater than zero".to_owned(),
        });
    }
    if config.max_layer == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_max_layer".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_max_layer must be greater than zero".to_owned(),
        });
    }
    if config.max_layer > HNSW_DEFAULT_MAX_LAYER {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_max_layer".to_owned(),
            value: config.max_layer.to_string(),
            reason: format!("hnsw_max_layer must be <= {HNSW_DEFAULT_MAX_LAYER}"),
        });
    }
    Ok(())
}

pub(super) fn ann_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

/// Derive the normalization and score-restoration budget for `DistDot`.
///
/// A length-`n` f32 dot product has a forward-error bound conventionally
/// expressed as `gamma_k = k*u/(1-k*u)`, where `u = eps/2`. We budget `8n+32`
/// rounded operations: component rescaling, multiplication, lane-local
/// accumulation, horizontal reduction, score restoration, and a 2x
/// architecture margin. Scaling exact unit vectors to
/// `1/sqrt(1+gamma_k)` makes the worst bounded computed self-dot no greater
/// than one. Dimensions for which `k*u > 1/4` are rejected rather than hiding
/// an ill-conditioned error bound behind an arbitrary shrink factor.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn dist_dot_budget(dimension: usize) -> SearchResult<DistDotBudget> {
    if dimension == 0 || dimension > DIST_DOT_MAX_DIMENSION {
        return Err(SearchError::InvalidConfig {
            field: "dimension".to_owned(),
            value: dimension.to_string(),
            reason: format!(
                "DistDot requires a dimension in 1..={DIST_DOT_MAX_DIMENSION} \
                 so its f32 roundoff bound remains finite and conservative"
            ),
        });
    }

    let dimension = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
        field: "dimension".to_owned(),
        value: dimension.to_string(),
        reason: "dimension exceeds the DistDot f32 error model".to_owned(),
    })?;
    let rounded_operations = f64::from(dimension).mul_add(8.0, 32.0);
    let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
    let accumulated_roundoff = rounded_operations * unit_roundoff;
    debug_assert!(accumulated_roundoff <= 0.25);
    let gamma = accumulated_roundoff / (1.0 - accumulated_roundoff);
    let radius_squared = 1.0 / (1.0 + gamma);
    // Include a small fixed allowance for the final f32 subtraction and
    // division when converting DistDot's distance back to cosine score.
    let score_tolerance = gamma + 8.0 * f64::from(f32::EPSILON);
    Ok(DistDotBudget {
        radius_squared: radius_squared as f32,
        score_tolerance: score_tolerance as f32,
    })
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn normalize_for_dist_dot(mut vector: Vec<f32>, budget: DistDotBudget) -> Vec<f32> {
    // f64 accumulation prevents the normalization pass itself from consuming
    // the f32 error budget intended for hnsw_rs/anndists' distance reduction.
    let norm_squared = vector
        .iter()
        .map(|&value| {
            let value = f64::from(value);
            value * value
        })
        .sum::<f64>();
    if norm_squared > 0.0 && norm_squared.is_finite() {
        let radius = f64::from(budget.radius_squared).sqrt();
        let scale = radius / norm_squared.sqrt();
        for value in &mut vector {
            *value = (f64::from(*value) * scale) as f32;
        }
    }
    vector
}

#[cfg(test)]
#[allow(dead_code)] // retained as utility; direct callers use vector_component_close
pub(super) fn vectors_close(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&l, &r)| vector_component_close(l, r))
}

#[cfg(test)]
pub(super) fn vector_component_close(left: f32, right: f32) -> bool {
    if left.to_bits() == right.to_bits() {
        return true;
    }
    // Non-finite values (NaN, Inf) with different bit patterns are never close.
    if !left.is_finite() || !right.is_finite() {
        return false;
    }
    let diff = (left - right).abs();
    let scale = left.abs().max(right.abs()).max(1.0);
    diff <= (f32::EPSILON * 8.0 * scale)
}

pub(super) fn estimate_recall(ef_search: usize, k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let numerator = f64::from(u32::try_from(ef_search.max(1)).unwrap_or(u32::MAX));
    let denominator = f64::from(u32::try_from(k).unwrap_or(u32::MAX));
    let ratio = numerator / denominator;
    0.1_f64.mul_add(ratio.log2(), 0.9_f64).clamp(0.0, 1.0)
}

/// Recall@k of `approx` against exact `exact`: the fraction of exact `doc_id`s that
/// also appear in `approx`. Used by [`super::HnswIndex::certify_ef_search`] to build the
/// measured calibration sample fed to the conformal certificate. `k` is tiny (top-k),
/// so the nested membership scan is trivial.
pub(super) fn recall_at_k_of(approx: &[VectorHit], exact: &[VectorHit]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let overlap = exact
        .iter()
        .filter(|e| approx.iter().any(|a| a.doc_id == e.doc_id))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let ratio = overlap as f64 / exact.len() as f64;
    ratio
}

pub(super) fn certified_ann_recall_sample(
    result: SearchResult<(Vec<VectorHit>, AnnSearchStats)>,
    exact: &[VectorHit],
) -> f64 {
    match result {
        Ok((approx, stats)) if stats.fallback_reason.is_none() => recall_at_k_of(&approx, exact),
        Ok(_) | Err(_) => 0.0,
    }
}
