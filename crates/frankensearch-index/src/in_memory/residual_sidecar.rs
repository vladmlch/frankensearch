use std::collections::BinaryHeap;
#[cfg(target_os = "linux")]
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::os::fd::OwnedFd;

use frankensearch_core::{SearchError, SearchResult};
use half::f16;
#[cfg(test)]
use sha2::{Digest, Sha256};

use crate::FsviV2Witness;

#[cfg(any(test, target_os = "linux"))]
use super::residual_io::SidecarCursor;
#[cfg(test)]
use super::residual_io::f32_bits_equal;
use super::residual_io::{residual_sidecar_error, try_filled_sidecar_vec};
use super::{
    EXACT_RESIDUAL_BLOCK, EXACT_RESIDUAL_LANES, EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES,
    EXACT_RESIDUAL_SIDECAR_HEADER_BYTES, EXACT_RESIDUAL_SIDECAR_MAX_BYTES,
    EXACT_RESIDUAL_SIDECAR_MAX_DIMENSION, HeapEntry,
};
#[cfg(any(test, target_os = "linux"))]
use super::{EXACT_RESIDUAL_SIDECAR_MAGIC, EXACT_RESIDUAL_SIDECAR_VERSION};
#[cfg(test)]
use super::{
    take_residual_query_suffix_allocation_failure, take_residual_query_transform_allocation_failure,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct ExactResidualLayout {
    pub(super) groups: usize,
    pub(super) blocks: usize,
    #[cfg(any(test, target_os = "linux"))]
    pub(super) permutation_len: usize,
    pub(super) centroid_len: usize,
    pub(super) residual_len: usize,
    pub(super) suffix_len: usize,
    pub(super) lane_len: usize,
    #[cfg(test)]
    pub(super) payload_bytes: usize,
    #[cfg(any(test, target_os = "linux"))]
    pub(super) encoded_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ResidualPruningCensus {
    pub(super) groups_scanned: usize,
    pub(super) eligible_lanes: usize,
    pub(super) lanes_pruned: usize,
    pub(super) exact_sidecar_scores: usize,
    pub(super) flat_fallback_rows: usize,
    pub(super) adaptive_fallbacks: usize,
}

impl ResidualPruningCensus {
    pub(super) fn merge(&mut self, other: Self) {
        self.groups_scanned = self.groups_scanned.saturating_add(other.groups_scanned);
        self.eligible_lanes = self.eligible_lanes.saturating_add(other.eligible_lanes);
        self.lanes_pruned = self.lanes_pruned.saturating_add(other.lanes_pruned);
        self.exact_sidecar_scores = self
            .exact_sidecar_scores
            .saturating_add(other.exact_sidecar_scores);
        self.flat_fallback_rows = self
            .flat_fallback_rows
            .saturating_add(other.flat_fallback_rows);
        self.adaptive_fallbacks = self
            .adaptive_fallbacks
            .saturating_add(other.adaptive_fallbacks);
    }
}

pub(super) struct ResidualScanOutcome {
    pub(super) heap: BinaryHeap<HeapEntry>,
    pub(super) census: ResidualPruningCensus,
}

pub(super) struct ResidualQueryTransform {
    pub(super) transformed: Vec<f32>,
    pub(super) norm: f64,
    pub(super) suffix_norms: Vec<f64>,
    pub(super) flat_f32_rounding_error: f64,
    pub(super) f32_flat_envelope_is_finite: bool,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExactResidualPublication {
    Published,
    DestinationExists,
}

/// An already-open unnamed inode plus the held no-follow parent descriptor.
/// Acquiring it proves that the optional cache can create an anonymous durable
/// candidate before source-derived sidecar allocation begins.
#[cfg(target_os = "linux")]
pub(super) struct ExactResidualPublicationCapability {
    pub(super) directory: OwnedFd,
    pub(super) destination: OsString,
    pub(super) file: File,
}

impl ExactResidualLayout {
    pub(super) fn for_shape(count: usize, dimension: usize) -> SearchResult<Self> {
        if dimension == 0 || dimension > EXACT_RESIDUAL_SIDECAR_MAX_DIMENSION {
            return Err(residual_sidecar_error(
                "dimension",
                "is outside the exact residual sidecar resource bound",
            ));
        }
        let groups = count
            .checked_add(EXACT_RESIDUAL_LANES - 1)
            .ok_or_else(|| residual_sidecar_error("record_count", "group count overflow"))?
            / EXACT_RESIDUAL_LANES;
        let blocks = dimension
            .checked_add(EXACT_RESIDUAL_BLOCK - 1)
            .ok_or_else(|| residual_sidecar_error("dimension", "block count overflow"))?
            / EXACT_RESIDUAL_BLOCK;
        let permutation_len = dimension;
        let centroid_len = groups
            .checked_mul(dimension)
            .ok_or_else(|| residual_sidecar_error("centroids", "length overflow"))?;
        let lane_len = groups
            .checked_mul(EXACT_RESIDUAL_LANES)
            .ok_or_else(|| residual_sidecar_error("lanes", "length overflow"))?;
        let residual_len = lane_len
            .checked_mul(dimension)
            .ok_or_else(|| residual_sidecar_error("residuals", "length overflow"))?;
        let suffix_len = lane_len
            .checked_mul(
                blocks
                    .checked_add(1)
                    .ok_or_else(|| residual_sidecar_error("suffix_norms", "length overflow"))?,
            )
            .ok_or_else(|| residual_sidecar_error("suffix_norms", "length overflow"))?;
        let payload_bytes = permutation_len
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|bytes| {
                centroid_len
                    .checked_add(residual_len)?
                    .checked_add(suffix_len)?
                    .checked_add(lane_len)?
                    .checked_mul(std::mem::size_of::<f32>())?
                    .checked_add(bytes)
            })
            .ok_or_else(|| residual_sidecar_error("payload", "length overflow"))?;
        let encoded_bytes = EXACT_RESIDUAL_SIDECAR_HEADER_BYTES
            .checked_add(payload_bytes)
            .and_then(|bytes| bytes.checked_add(EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES))
            .ok_or_else(|| residual_sidecar_error("payload", "length overflow"))?;
        if encoded_bytes > EXACT_RESIDUAL_SIDECAR_MAX_BYTES {
            return Err(residual_sidecar_error(
                "payload",
                "exceeds the exact residual sidecar resource bound",
            ));
        }
        Ok(Self {
            groups,
            blocks,
            #[cfg(any(test, target_os = "linux"))]
            permutation_len,
            centroid_len,
            residual_len,
            suffix_len,
            lane_len,
            #[cfg(test)]
            payload_bytes,
            #[cfg(any(test, target_os = "linux"))]
            encoded_bytes,
        })
    }
}

/// Immutable identity fields that bind an exact-residual sidecar to one admitted
/// FSVI v2 source generation.  Every member is a cryptographic witness from the
/// source image; display names and dimensions alone are never enough to adopt a
/// transformed search layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResidualSourceBinding {
    pub(super) generation_fingerprint: [u8; 32],
    pub(super) vector_content_digest: [u8; 32],
    pub(super) ordered_live_docset_digest: [u8; 32],
    pub(super) space_fingerprint: [u8; 32],
}

impl ResidualSourceBinding {
    pub(super) const fn from_witness(witness: &FsviV2Witness) -> Self {
        Self {
            generation_fingerprint: witness.generation_fingerprint,
            vector_content_digest: witness.vector_content_digest,
            ordered_live_docset_digest: witness.ordered_live_docset_digest,
            space_fingerprint: witness.space_fingerprint,
        }
    }
}

/// Versioned exact early-abandon layout.  Candidate groups are stored in a
/// dimension-major, lane-minor residual slab, with one centroid per group and a
/// suffix norm for every lane/block boundary.  Final scores are always computed
/// through the incumbent f16 flat dot product; this layout can only remove a
/// candidate after its conservative Cauchy--Schwarz upper bound proves it cannot
/// beat the current exact cutoff.
#[derive(Clone, Debug)]
pub(super) struct ExactResidualSidecar {
    #[cfg(any(test, target_os = "linux"))]
    pub(super) source: ResidualSourceBinding,
    pub(super) count: usize,
    pub(super) dimension: usize,
    pub(super) block: usize,
    pub(super) lanes: usize,
    pub(super) permutation: Vec<u32>,
    /// Group-major centroids in transformed dimension order.
    pub(super) centroids: Vec<f32>,
    /// Group-major, dimension-major, lane-minor residuals.
    pub(super) residuals: Vec<f32>,
    /// Group-major, block-boundary-major, lane-minor residual suffix norms.
    pub(super) suffix_norms: Vec<f32>,
    /// Whole-vector reconstruction-error norm per group/lane.  It accounts for
    /// f32 centroid/residual rounding before a lane can be safely abandoned.
    pub(super) correction_norms: Vec<f32>,
}

impl ExactResidualSidecar {
    pub(super) fn group_count(&self) -> usize {
        self.count.div_ceil(self.lanes)
    }

    pub(super) fn block_count(&self) -> usize {
        self.dimension.div_ceil(self.block)
    }

    #[cfg(test)]
    pub(super) fn is_bound_to(
        &self,
        source: &ResidualSourceBinding,
        count: usize,
        dimension: usize,
    ) -> bool {
        self.source == *source
            && self.count == count
            && self.dimension == dimension
            && self.block == EXACT_RESIDUAL_BLOCK
            && self.lanes == EXACT_RESIDUAL_LANES
    }

    #[cfg(any(test, target_os = "linux"))]
    pub(super) fn validated_layout(&self) -> SearchResult<ExactResidualLayout> {
        let layout = ExactResidualLayout::for_shape(self.count, self.dimension)?;
        if self.permutation.len() != layout.permutation_len
            || self.centroids.len() != layout.centroid_len
            || self.residuals.len() != layout.residual_len
            || self.suffix_norms.len() != layout.suffix_len
            || self.correction_norms.len() != layout.lane_len
        {
            return Err(residual_sidecar_error(
                "payload",
                "in-memory sidecar fields do not match its declared layout",
            ));
        }
        Ok(layout)
    }

    #[cfg(test)]
    pub(super) fn encode(&self) -> SearchResult<Vec<u8>> {
        let layout = self.validated_layout()?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(layout.encoded_bytes)
            .map_err(|_| residual_sidecar_error("payload", "allocation failed"))?;
        bytes.extend_from_slice(&EXACT_RESIDUAL_SIDECAR_MAGIC);
        bytes.extend_from_slice(&EXACT_RESIDUAL_SIDECAR_VERSION.to_le_bytes());
        for digest in [
            self.source.generation_fingerprint,
            self.source.vector_content_digest,
            self.source.ordered_live_docset_digest,
            self.source.space_fingerprint,
        ] {
            bytes.extend_from_slice(&digest);
        }
        bytes.extend_from_slice(
            &u64::try_from(self.count)
                .map_err(|_| residual_sidecar_error("record_count", "does not fit u64"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u64::try_from(self.dimension)
                .map_err(|_| residual_sidecar_error("dimension", "does not fit u64"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u32::try_from(self.block)
                .map_err(|_| residual_sidecar_error("block", "does not fit u32"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u32::try_from(self.lanes)
                .map_err(|_| residual_sidecar_error("lanes", "does not fit u32"))?
                .to_le_bytes(),
        );
        for value in &self.permutation {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for values in [
            &self.centroids,
            &self.residuals,
            &self.suffix_norms,
            &self.correction_norms,
        ] {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        debug_assert_eq!(
            bytes.len(),
            EXACT_RESIDUAL_SIDECAR_HEADER_BYTES + layout.payload_bytes
        );
        let digest = Sha256::digest(&bytes);
        bytes.extend_from_slice(&digest);
        Ok(bytes)
    }

    #[cfg(test)]
    pub(super) fn decode(bytes: &[u8]) -> SearchResult<Self> {
        if bytes.len() > EXACT_RESIDUAL_SIDECAR_MAX_BYTES {
            return Err(residual_sidecar_error(
                "payload",
                "exceeds the exact residual sidecar resource bound",
            ));
        }
        let mut cursor = SidecarCursor::new(bytes);
        if cursor.take_array::<8>("magic")? != EXACT_RESIDUAL_SIDECAR_MAGIC {
            return Err(residual_sidecar_error("magic", "invalid sidecar magic"));
        }
        if cursor.u32("version")? != EXACT_RESIDUAL_SIDECAR_VERSION {
            return Err(residual_sidecar_error(
                "version",
                "unsupported residual-sidecar schema version",
            ));
        }
        let source = ResidualSourceBinding {
            generation_fingerprint: cursor.take_array("generation_fingerprint")?,
            vector_content_digest: cursor.take_array("vector_content_digest")?,
            ordered_live_docset_digest: cursor.take_array("ordered_live_docset_digest")?,
            space_fingerprint: cursor.take_array("space_fingerprint")?,
        };
        let count = usize::try_from(cursor.u64("record_count")?).map_err(|_| {
            residual_sidecar_error("record_count", "does not fit the current platform")
        })?;
        let dimension = usize::try_from(cursor.u64("dimension")?).map_err(|_| {
            residual_sidecar_error("dimension", "does not fit the current platform")
        })?;
        let block = usize::try_from(cursor.u32("block")?)
            .map_err(|_| residual_sidecar_error("block", "does not fit the current platform"))?;
        let lanes = usize::try_from(cursor.u32("lanes")?)
            .map_err(|_| residual_sidecar_error("lanes", "does not fit the current platform"))?;
        if block != EXACT_RESIDUAL_BLOCK || lanes != EXACT_RESIDUAL_LANES {
            return Err(residual_sidecar_error(
                "layout",
                "sidecar layout is not the supported exact residual format",
            ));
        }
        let layout = ExactResidualLayout::for_shape(count, dimension)?;
        if bytes.len() != layout.encoded_bytes
            || cursor.remaining()
                != layout
                    .payload_bytes
                    .checked_add(EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES)
                    .ok_or_else(|| residual_sidecar_error("payload", "length overflow"))?
        {
            return Err(residual_sidecar_error(
                "payload",
                "length does not match the versioned sidecar layout",
            ));
        }
        let integrity_start = bytes
            .len()
            .checked_sub(EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES)
            .ok_or_else(|| residual_sidecar_error("integrity", "digest is truncated"))?;
        let supplied_digest: [u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES] = bytes[integrity_start..]
            .try_into()
            .map_err(|_| residual_sidecar_error("integrity", "digest is truncated"))?;
        let expected_digest: [u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES] =
            Sha256::digest(&bytes[..integrity_start]).into();
        if supplied_digest != expected_digest {
            return Err(residual_sidecar_error(
                "integrity",
                "whole-sidecar digest mismatch",
            ));
        }

        let mut permutation = Vec::new();
        permutation
            .try_reserve_exact(layout.permutation_len)
            .map_err(|_| residual_sidecar_error("permutation", "allocation failed"))?;
        let mut seen = Vec::new();
        seen.try_reserve_exact(dimension)
            .map_err(|_| residual_sidecar_error("permutation", "allocation failed"))?;
        seen.resize(dimension, false);
        for _ in 0..layout.permutation_len {
            let value = cursor.u32("permutation")?;
            let index = usize::try_from(value).map_err(|_| {
                residual_sidecar_error("permutation", "does not fit the current platform")
            })?;
            if index >= dimension || std::mem::replace(&mut seen[index], true) {
                return Err(residual_sidecar_error(
                    "permutation",
                    "must be a complete, duplicate-free dimension permutation",
                ));
            }
            permutation.push(value);
        }
        let centroids = cursor.f32_vec("centroids", layout.centroid_len, false)?;
        let residuals = cursor.f32_vec("residuals", layout.residual_len, false)?;
        let suffix_norms = cursor.f32_vec("suffix_norms", layout.suffix_len, true)?;
        let correction_norms = cursor.f32_vec("correction_norms", layout.lane_len, true)?;
        let digest = cursor.take_array::<EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES>("integrity")?;
        debug_assert_eq!(digest, supplied_digest);
        if !cursor.is_exhausted() {
            return Err(residual_sidecar_error(
                "payload",
                "trailing bytes are forbidden",
            ));
        }
        Ok(Self {
            source,
            count,
            dimension,
            block,
            lanes,
            permutation,
            centroids,
            residuals,
            suffix_norms,
            correction_norms,
        })
    }

    /// Validate the fixed metadata that binds a sidecar header to its source.
    /// The caller separately validates the exact shape and whole-file digest.
    #[cfg(target_os = "linux")]
    pub(super) fn header_matches_source(
        header: &[u8],
        source: &ResidualSourceBinding,
        count: usize,
        dimension: usize,
    ) -> SearchResult<bool> {
        if header.len() != EXACT_RESIDUAL_SIDECAR_HEADER_BYTES {
            return Ok(false);
        }
        let mut cursor = SidecarCursor::new(header);
        if cursor.take_array::<8>("magic")? != EXACT_RESIDUAL_SIDECAR_MAGIC
            || cursor.u32("version")? != EXACT_RESIDUAL_SIDECAR_VERSION
        {
            return Ok(false);
        }
        let encoded_source = ResidualSourceBinding {
            generation_fingerprint: cursor.take_array("generation_fingerprint")?,
            vector_content_digest: cursor.take_array("vector_content_digest")?,
            ordered_live_docset_digest: cursor.take_array("ordered_live_docset_digest")?,
            space_fingerprint: cursor.take_array("space_fingerprint")?,
        };
        let encoded_count = usize::try_from(cursor.u64("record_count")?)
            .map_err(|_| residual_sidecar_error("record_count", "does not fit this platform"))?;
        let encoded_dimension = usize::try_from(cursor.u64("dimension")?)
            .map_err(|_| residual_sidecar_error("dimension", "does not fit this platform"))?;
        let block = usize::try_from(cursor.u32("block")?)
            .map_err(|_| residual_sidecar_error("block", "does not fit this platform"))?;
        let lanes = usize::try_from(cursor.u32("lanes")?)
            .map_err(|_| residual_sidecar_error("lanes", "does not fit this platform"))?;
        if encoded_source != *source
            || encoded_count != count
            || encoded_dimension != dimension
            || block != EXACT_RESIDUAL_BLOCK
            || lanes != EXACT_RESIDUAL_LANES
        {
            return Ok(false);
        }
        Ok(cursor.is_exhausted())
    }

    #[cfg(test)]
    pub(super) fn exactly_matches_derived(&self, expected: &Self) -> bool {
        self.source == expected.source
            && self.count == expected.count
            && self.dimension == expected.dimension
            && self.block == expected.block
            && self.lanes == expected.lanes
            && self.permutation == expected.permutation
            && f32_bits_equal(&self.centroids, &expected.centroids)
            && f32_bits_equal(&self.residuals, &expected.residuals)
            && f32_bits_equal(&self.suffix_norms, &expected.suffix_norms)
            && f32_bits_equal(&self.correction_norms, &expected.correction_norms)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn outward_f32_norm(value: f64, field: &str) -> SearchResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(residual_sidecar_error(
            field,
            "norm must be finite and non-negative",
        ));
    }
    let narrowed = value.sqrt() as f32;
    if !narrowed.is_finite() {
        return Err(residual_sidecar_error(field, "norm does not fit f32"));
    }
    // Persist an outward-rounded envelope, not a rounded-down estimate.  The
    // additional epsilon term covers the f64->f32 conversion and later f32
    // multiplication in a conservative Cauchy--Schwarz upper bound.
    let outward = narrowed.mul_add(1.0 + 4.0 * f32::EPSILON, 4.0 * f32::EPSILON);
    if !outward.is_finite() {
        return Err(residual_sidecar_error(
            field,
            "outward norm does not fit f32",
        ));
    }
    Ok(outward)
}

impl ResidualQueryTransform {
    pub(super) fn from_query(query: &[f32], sidecar: &ExactResidualSidecar) -> SearchResult<Self> {
        #[cfg(test)]
        if take_residual_query_transform_allocation_failure() {
            return Err(residual_sidecar_error(
                "query",
                "injected transform allocation failure",
            ));
        }
        let mut transformed = Vec::new();
        transformed
            .try_reserve_exact(sidecar.dimension)
            .map_err(|_| residual_sidecar_error("query", "transform allocation failed"))?;
        for &source_dimension in &sidecar.permutation {
            let source_dimension = usize::try_from(source_dimension).map_err(|_| {
                residual_sidecar_error("query", "permutation does not fit this platform")
            })?;
            let value = *query.get(source_dimension).ok_or_else(|| {
                residual_sidecar_error("query", "permutation exceeds query dimension")
            })?;
            transformed.push(value);
        }
        let norm_sq = transformed
            .iter()
            .try_fold(0.0_f64, |sum, &value| {
                finite_f64_mul(f64::from(value), f64::from(value))
                    .and_then(|square| finite_f64_add(sum, square))
            })
            .unwrap_or(f64::INFINITY);
        // This norm multiplies the residual/correction envelopes in the
        // pruning inequality, so retain an outward f64 value rather than the
        // nearest-root value used only for presentation.
        let norm =
            outward_f64_norm_from_square_sum(norm_sq, sidecar.dimension).unwrap_or(f64::INFINITY);
        // The interval arithmetic itself is f64, but the authoritative scorer
        // is f32. `f32_flat_envelope` bounds the real absolute dot product of
        // this query against any finite f16 row. The standard Higham model for
        // at most `2d + 32` rounded f32 products/additions gives
        // `|fl(dot) - dot| <= gamma_(2d + 32) * envelope`, where
        // `gamma_n = n*u/(1-n*u)` and `u = f32::EPSILON / 2`. The scalar f16
        // fallback and the SIMD reduction both use no more than that operation
        // budget; the bound is intentionally looser than either implementation.
        // The relative model alone does not cover a subnormal f32 intermediate,
        // so the companion absolute term below budgets half a smallest
        // subnormal for every rounded operation as well.
        // If this envelope could approach f32 overflow, retain the incumbent
        // flat route rather than compare a finite interval to an `inf` tie.
        let f32_flat_sum = transformed.iter().try_fold(0.0_f64, |sum, value| {
            finite_f64_add(sum, f64::from(value.abs()))
        });
        let f32_flat_envelope = f32_flat_sum
            .and_then(|sum| finite_f64_mul(sum, f64::from(f16::MAX.to_f32())))
            .unwrap_or(f64::INFINITY);
        let flat_f32_rounding_error = f32_flat_rounding_error(sidecar.dimension, f32_flat_envelope)
            .filter(|error| error.is_finite())
            .unwrap_or(f64::INFINITY);
        #[cfg(test)]
        if take_residual_query_suffix_allocation_failure() {
            return Err(residual_sidecar_error(
                "query_suffix",
                "injected suffix allocation failure",
            ));
        }
        let mut suffix_norms =
            try_filled_sidecar_vec(sidecar.block_count() + 1, 0.0_f64, "query_suffix")?;
        let mut suffix_sum = 0.0_f64;
        for block_index in (0..sidecar.block_count()).rev() {
            let start = block_index * sidecar.block;
            let end = (start + sidecar.block).min(sidecar.dimension);
            for &value in &transformed[start..end] {
                suffix_sum += f64::from(value) * f64::from(value);
            }
            suffix_norms[block_index] =
                outward_f64_norm_from_square_sum(suffix_sum, sidecar.dimension)
                    .unwrap_or(f64::INFINITY);
        }
        Ok(Self {
            transformed,
            norm,
            suffix_norms,
            flat_f32_rounding_error,
            f32_flat_envelope_is_finite: finite_f64_add(f32_flat_envelope, flat_f32_rounding_error)
                .is_some_and(|total| total < f64::from(f32::MAX) * 0.5),
        })
    }
}

#[allow(clippy::cast_precision_loss)]
fn f32_rounding_gamma(dimension: usize) -> Option<f64> {
    let operation_count = f32_authoritative_dot_operation_count(dimension)?;
    let unit_roundoff = f64::from(f32::EPSILON) * 0.5;
    let rounded_operations = operation_count as f64;
    let numerator = finite_f64_mul(rounded_operations, unit_roundoff)?;
    let denominator = 1.0 - numerator;
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let gamma = numerator / denominator;
    gamma.is_finite().then_some(gamma)
}

#[allow(clippy::cast_precision_loss)]
fn f32_flat_rounding_error(dimension: usize, envelope: f64) -> Option<f64> {
    let operation_count = f32_authoritative_dot_operation_count(dimension)?;
    let relative = finite_f64_mul(envelope, f32_rounding_gamma(dimension)?)?;
    // IEEE-754 gradual underflow gives each subnormal-result rounding an
    // absolute error no greater than half the least f32 subnormal. Adding one
    // such quantum per multiply/add covers the part the classic relative
    // gamma model deliberately excludes.
    let subnormal_quantum = f64::from(f32::from_bits(1)) * 0.5;
    let subnormal_error = finite_f64_mul(operation_count as f64, subnormal_quantum)?;
    finite_f64_add(relative, subnormal_error)
}

/// Perform a floating-point product only when the resulting interval endpoint
/// stays finite. `f64` intentionally has no `checked_mul`; callers use this
/// helper to fail closed to the flat scan rather than admitting an overflowed
/// pruning bound.
pub(super) fn finite_f64_mul(left: f64, right: f64) -> Option<f64> {
    let product = left * right;
    product.is_finite().then_some(product)
}

/// As [`finite_f64_mul`], but for a conservative interval sum.
pub(super) fn finite_f64_add(left: f64, right: f64) -> Option<f64> {
    let sum = left + right;
    sum.is_finite().then_some(sum)
}

/// Outward round a finite non-negative norm after its f64 square root. The
/// residual bound consumes this result as a multiplier, so a nearest-even root
/// may not be used directly.
fn outward_f64_sqrt(square: f64) -> Option<f64> {
    if !square.is_finite() || square < 0.0 {
        return None;
    }
    let root = square.sqrt();
    root.is_finite().then(|| root.next_up())
}

/// Conservative scalar-operation budget for the authoritative
/// [`crate::simd::dot_product_f16_f32`] scorer. Each dimension has a multiply and a SIMD
/// accumulator add; the four accumulator tree adds 3 × 8 lanes, `reduce_add`
/// adds 7 lanes, and one spare operation covers dispatch-equivalent tails and
/// reduction details. Thus this is at least `2d + 32` for every actual route.
fn f32_authoritative_dot_operation_count(dimension: usize) -> Option<usize> {
    dimension.checked_mul(2)?.checked_add(32)
}

/// Copy a source-owned document id without relying on infallible `Clone` or
/// `to_owned` allocation. The in-memory product must retain ownership after
/// its file-backed source is dropped, but an allocation failure remains a typed
/// opening error rather than an abort.
pub(super) fn fallible_doc_id_copy(value: &str) -> SearchResult<String> {
    let mut copy = String::new();
    copy.try_reserve_exact(value.len())
        .map_err(|_| SearchError::InvalidConfig {
            field: "vectors.doc_id".to_owned(),
            value: value.len().to_string(),
            reason: "document-id allocation failed".to_owned(),
        })?;
    copy.push_str(value);
    Ok(copy)
}

#[allow(clippy::cast_precision_loss)]
fn f64_rounding_gamma(dimension: usize) -> Option<f64> {
    // Centroid and residual partials each have one multiply plus one add per
    // dimension, with room for the final combination terms. This controls the
    // f64 evaluation of the mathematical decomposition itself.
    let operation_count = dimension.checked_mul(4)?.checked_add(16)?;
    let unit_roundoff = f64::EPSILON * 0.5;
    let rounded_operations = operation_count as f64;
    let numerator = finite_f64_mul(rounded_operations, unit_roundoff)?;
    let denominator = 1.0 - numerator;
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let gamma = numerator / denominator;
    gamma.is_finite().then_some(gamma)
}

/// Convert a rounded f64 sum-of-squares into an outward norm. The f64 gamma
/// first inflates the finite reduction error, then `outward_f64_sqrt` expands
/// the final root. This value is consumed by the decomposition magnitude, so
/// neither nearest-even step is allowed to point inward.
pub(super) fn outward_f64_norm_from_square_sum(square_sum: f64, dimension: usize) -> Option<f64> {
    if !square_sum.is_finite() || square_sum < 0.0 {
        return None;
    }
    let inflation = finite_f64_mul(square_sum, f64_rounding_gamma(dimension)?)?;
    let outward_square = finite_f64_add(square_sum, inflation)?;
    outward_f64_sqrt(outward_square)
}

/// Conservative per-lane upper bound for the exact f16 score before the
/// current residual block has been accumulated.  It deliberately works in
/// f64 and inflates the result for the f32 products and reduction used by the
/// authoritative scorer.  A caller may prune only on a strict finite bound
/// below its current cutoff, preserving f32 score ties and the index tie-break.
#[allow(clippy::cast_precision_loss)]
pub(super) fn residual_lane_upper_bound(
    sidecar: &ExactResidualSidecar,
    transformed: &ResidualQueryTransform,
    block_index: usize,
    centroid_dot: f64,
    centroid_norm: f64,
    residual_norm: f64,
    correction_norm: f64,
    partial: f64,
) -> f64 {
    let magnitude = transformed.norm * (centroid_norm + residual_norm + correction_norm)
        + centroid_dot.abs()
        + partial.abs();
    // The `gamma` term covers the f64 evaluation above; the transform's f32
    // term covers the authoritative f16×f32 score. Both are analytic forward
    // error bounds, not an empirical tolerance. `block_index` participates
    // through `suffix_norms[block_index]`, which is the residual tail being
    // bounded at this exact scan point.
    let decomposition_rounding_error = f64_rounding_gamma(sidecar.dimension)
        .map_or(f64::INFINITY, |gamma| gamma * magnitude.max(1.0));
    centroid_dot
        + partial
        + transformed.suffix_norms[block_index] * residual_norm
        + transformed.norm * correction_norm
        + decomposition_rounding_error
        + transformed.flat_f32_rounding_error
}
