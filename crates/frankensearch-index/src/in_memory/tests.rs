#![allow(
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::redundant_clone,
    clippy::suboptimal_flops,
    clippy::unnecessary_literal_bound
)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use half::f16;

use crate::simd::dot_product_f16_f32;

use super::residual_sidecar::{
    ExactResidualSidecar, ResidualQueryTransform, ResidualSourceBinding,
    outward_f64_norm_from_square_sum, residual_lane_upper_bound,
};
use super::*;

mod identity_cache;
mod residual_sidecar;
mod search;

pub(super) fn test_run_nonce() -> u128 {
    static RUN_NONCE: OnceLock<u128> = OnceLock::new();
    *RUN_NONCE.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    })
}

pub(super) fn temp_index_path(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nonce = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let dir = std::env::temp_dir().join("frankensearch_in_memory_tests");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir.join(format!(
        "{name}-{}-{}-{nonce}.fsvi",
        std::process::id(),
        test_run_nonce()
    ))
}

pub(super) fn owned_temp_dir(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nonce = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let parent = std::env::temp_dir().join("frankensearch_in_memory_tests");
    std::fs::create_dir_all(&parent).expect("create temp parent");
    let dir = parent.join(format!(
        "{name}-{}-{}-{nonce}",
        std::process::id(),
        test_run_nonce()
    ));
    std::fs::create_dir(&dir).expect("create unique owned temp directory");
    dir
}

#[cfg(target_os = "linux")]
pub(super) fn write_new_owned_file(path: &Path, bytes: &[u8]) {
    use std::io::Write as _;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("create a unique owned test file without replacement");
    file.write_all(bytes)
        .expect("write a unique owned test file");
    file.sync_all().expect("sync a unique owned test file");
}

pub(super) fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(path.with_extension("fsvi.wal"));
}

pub(super) fn make_normalized_vec(dim: usize, seed: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|i| (seed + i as f32 * 0.1).sin()).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

pub(super) fn bind_test_residual_source(index: &mut InMemoryVectorIndex) {
    index.residual_source_binding = Some(ResidualSourceBinding {
        generation_fingerprint: [0x19; 32],
        vector_content_digest: [0x2a; 32],
        ordered_live_docset_digest: [0x3b; 32],
        space_fingerprint: [0x4c; 32],
    });
}

pub(super) fn finite_f16_bits(bits: u16) -> f32 {
    let value = f16::from_bits(bits);
    if value.is_finite() {
        value.to_f32()
    } else {
        0.0
    }
}

pub(super) fn bounded_finite_f32_bits(bits: u32) -> f32 {
    // Preserve sign, every subnormal, and a broad finite exponent range
    // while clearing the top exponent bit. That keeps the test inside the
    // sidecar's analytically admitted finite-envelope route rather than
    // asking a pruning proof to cover the intentional flat fallback.
    f32::from_bits(bits & 0xbfff_ffff)
}

pub(super) fn assert_residual_bounds_cover_exact_scores(
    index: &InMemoryVectorIndex,
    sidecar: &ExactResidualSidecar,
    query: &[f32],
) {
    let transformed =
        ResidualQueryTransform::from_query(query, sidecar).expect("finite test transform");
    assert!(
        transformed.f32_flat_envelope_is_finite,
        "the test vector range must use the interval route"
    );
    for group in 0..sidecar.group_count() {
        let group_start = group * sidecar.lanes;
        let active_lanes = (sidecar.count - group_start).min(sidecar.lanes);
        let centroid =
            &sidecar.centroids[group * sidecar.dimension..(group + 1) * sidecar.dimension];
        let mut centroid_dot = 0.0_f64;
        let mut centroid_norm_sq = 0.0_f64;
        for (&query_value, &centroid_value) in transformed.transformed.iter().zip(centroid) {
            centroid_dot += f64::from(query_value) * f64::from(centroid_value);
            centroid_norm_sq += f64::from(centroid_value) * f64::from(centroid_value);
        }
        let centroid_norm = outward_f64_norm_from_square_sum(centroid_norm_sq, sidecar.dimension)
            .expect("finite centroid norm has an outward f64 root");
        let mut partial = [0.0_f64; EXACT_RESIDUAL_LANES];
        for block_index in 0..sidecar.block_count() {
            for (lane, &partial_lane) in partial[..active_lanes].iter().enumerate() {
                let suffix_offset =
                    (group * (sidecar.block_count() + 1) + block_index) * sidecar.lanes + lane;
                let upper_bound = residual_lane_upper_bound(
                    sidecar,
                    &transformed,
                    block_index,
                    centroid_dot,
                    centroid_norm,
                    f64::from(sidecar.suffix_norms[suffix_offset]),
                    f64::from(sidecar.correction_norms[group * sidecar.lanes + lane]),
                    partial_lane,
                );
                let exact_score =
                    dot_product_f16_f32(index.vector_slice(group_start + lane), query)
                        .expect("finite f16 score");
                assert!(
                    upper_bound.is_finite() && upper_bound >= f64::from(exact_score),
                    "upper bound underestimates group={group} lane={lane} block={block_index}: \
                         upper={upper_bound:?}, exact={exact_score:?}"
                );
            }
            let start = block_index * sidecar.block;
            let end = (start + sidecar.block).min(sidecar.dimension);
            for transformed_dimension in start..end {
                let query_value = f64::from(transformed.transformed[transformed_dimension]);
                let base = group * sidecar.dimension * sidecar.lanes
                    + transformed_dimension * sidecar.lanes;
                for (lane, partial_lane) in partial[..active_lanes].iter_mut().enumerate() {
                    *partial_lane += query_value * f64::from(sidecar.residuals[base + lane]);
                }
            }
        }
    }
}
