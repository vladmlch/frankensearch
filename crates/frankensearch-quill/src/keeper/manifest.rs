use thiserror::Error;

use super::constants::{
    CURRENT_ENGINE_VERSION, EMPTY_TOMBSTONES, FIELD_STATS_BYTES,
    MANIFEST_FLAG_BULK_MODE_IN_PROGRESS, MANIFEST_FORMAT_VERSION, MANIFEST_FORMAT_VERSION_V1,
    MANIFEST_MAGIC, MANIFEST_MIN_BYTES, MAX_MANIFEST_FIELDS, MAX_MANIFEST_SEGMENTS,
    SEGMENT_FIXED_BYTES, TOMBSTONE_ARRAY_MAX_CARDINALITY, TOMBSTONE_BITMAP_BYTES,
};
use super::errors::ManifestCodecError;
use super::manifest_validation::{
    ErrorClass, TombstoneContainers, array_binary_search, bitmap_contains,
    check_manifest_byte_limit, consume_tombstone_set, copy_bytes, count_to_usize, invalid,
    manifest_encoded_len, non_canonical, put_u16, put_u32, put_u64, set_bitmap_value,
    split_tombstone_docid, tombstone_chunk_count, tombstone_chunk_exists, tombstones_are_subset,
    usize_to_u64, validate_manifest, validate_manifest_shape, validate_tombstone_bytes,
    write_inserted_tombstone_container, write_removed_tombstone_container,
    write_singleton_tombstone_chunk, write_tombstone_container_header,
};
use super::stats_and_cursor::ByteCursor;

/// Owned canonical roaring-lite tombstone set.
///
/// The exact wire bytes remain authoritative so decoding preserves ARRAY or
/// BITMAP representation inside the hysteresis overlap. Cached cardinality
/// makes live-count rollups constant-time, while mutation rewrites only this
/// small delete-path value and copies untouched containers byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TombstoneSet {
    pub(in crate::keeper) encoded: Vec<u8>,
    pub(in crate::keeper) cardinality: u64,
}

impl TombstoneSet {
    /// Construct the canonical empty set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoded: EMPTY_TOMBSTONES.to_vec(),
            cardinality: 0,
        }
    }

    /// Build the one history-independent representation for sorted membership.
    ///
    /// Replacement segments use this path so equivalent merge schedules cannot
    /// retain ARRAY/BITMAP hysteresis from their source MANIFEST generations.
    pub(in crate::keeper) fn from_sorted_docids(
        docids: &[u32],
    ) -> Result<Self, ManifestCodecError> {
        if docids.is_empty() {
            return Ok(Self::new());
        }
        if let Some([previous, current]) = docids.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(invalid(format!(
                "concat tombstones are not strictly ascending at {}/{}",
                previous, current
            )));
        }

        let mut chunk_count = 0_usize;
        let mut encoded_len = std::mem::size_of::<u32>();
        let mut start = 0_usize;
        while start < docids.len() {
            let chunk_id = split_tombstone_docid(docids[start]).0;
            let end = start
                + docids[start..]
                    .partition_point(|docid| split_tombstone_docid(*docid).0 == chunk_id);
            let cardinality = end - start;
            let payload_len = if cardinality <= usize::from(TOMBSTONE_ARRAY_MAX_CARDINALITY) {
                cardinality.checked_mul(std::mem::size_of::<u16>())
            } else {
                Some(TOMBSTONE_BITMAP_BYTES)
            }
            .ok_or_else(|| invalid("concat tombstone payload length overflow"))?;
            encoded_len = encoded_len
                .checked_add(5)
                .and_then(|length| length.checked_add(payload_len))
                .ok_or_else(|| invalid("concat tombstone wire length overflow"))?;
            chunk_count = chunk_count
                .checked_add(1)
                .ok_or_else(|| invalid("concat tombstone chunk count overflow"))?;
            start = end;
        }
        let chunk_count_u32 = u32::try_from(chunk_count)
            .map_err(|_| invalid("concat tombstone chunk count does not fit u32"))?;
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(encoded_len)
            .map_err(|error| invalid(format!("concat tombstone allocation failed: {error}")))?;
        put_u32(&mut encoded, chunk_count_u32);
        let mut start = 0_usize;
        while start < docids.len() {
            let chunk_id = split_tombstone_docid(docids[start]).0;
            let end = start
                + docids[start..]
                    .partition_point(|docid| split_tombstone_docid(*docid).0 == chunk_id);
            let cardinality = end - start;
            if cardinality <= usize::from(TOMBSTONE_ARRAY_MAX_CARDINALITY) {
                write_tombstone_container_header(
                    &mut encoded,
                    chunk_id,
                    0,
                    u64::try_from(cardinality).unwrap_or(u64::MAX),
                )?;
                for &docid in &docids[start..end] {
                    put_u16(&mut encoded, split_tombstone_docid(docid).1);
                }
            } else {
                write_tombstone_container_header(
                    &mut encoded,
                    chunk_id,
                    1,
                    u64::try_from(cardinality).unwrap_or(u64::MAX),
                )?;
                let payload_start = encoded.len();
                encoded.resize(payload_start + TOMBSTONE_BITMAP_BYTES, 0);
                for &docid in &docids[start..end] {
                    set_bitmap_value(
                        &mut encoded[payload_start..],
                        split_tombstone_docid(docid).1,
                    );
                }
            }
            start = end;
        }
        let cardinality = u64::try_from(docids.len())
            .map_err(|_| invalid("concat tombstone cardinality does not fit u64"))?;
        debug_assert_eq!(
            validate_tombstone_bytes(&encoded, None).ok(),
            Some(cardinality)
        );
        Ok(Self::from_validated_bytes(encoded, cardinality))
    }

    /// Parse canonical roaring-lite bytes without normalizing container kinds.
    ///
    /// # Errors
    ///
    /// Rejects truncation, unknown kinds, noncanonical ordering/cardinality,
    /// resource-limit violations, trailing bytes, or allocation failure.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ManifestCodecError> {
        let cardinality = validate_tombstone_bytes(bytes, None)?;
        let encoded = copy_bytes(bytes, "tombstone set")?;
        Ok(Self {
            encoded,
            cardinality,
        })
    }

    fn from_validated_bytes(encoded: Vec<u8>, cardinality: u64) -> Self {
        Self {
            encoded,
            cardinality,
        }
    }

    /// Borrow the exact canonical wire representation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.encoded
    }

    /// Number of tombstoned global document ids.
    #[must_use]
    pub const fn cardinality(&self) -> u64 {
        self.cardinality
    }

    /// Whether the set contains no document ids.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Test one global document id without allocating.
    #[must_use]
    pub fn contains(&self, global_docid: u32) -> bool {
        let (chunk_id, low) = split_tombstone_docid(global_docid);
        let Ok(mut containers) = TombstoneContainers::new(&self.encoded) else {
            return false;
        };
        while let Ok(Some(container)) = containers.next_container() {
            match container.chunk_id.cmp(&chunk_id) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal => {
                    return match container.kind {
                        0 => array_binary_search(container.payload, low),
                        1 => bitmap_contains(container.payload, low),
                        _ => false,
                    };
                }
            }
        }
        false
    }

    /// Insert one global document id, promoting ARRAY to BITMAP at 4,097.
    ///
    /// # Errors
    ///
    /// Returns an allocation or invariant error without changing the set.
    pub fn insert(&mut self, global_docid: u32) -> Result<bool, ManifestCodecError> {
        if self.contains(global_docid) {
            return Ok(false);
        }
        let (chunk_id, low) = split_tombstone_docid(global_docid);
        let chunk_count = tombstone_chunk_count(&self.encoded)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(
                self.encoded
                    .len()
                    .checked_add(7)
                    .ok_or_else(|| invalid("tombstone insertion length overflow"))?,
            )
            .map_err(|error| invalid(format!("tombstone insertion allocation failed: {error}")))?;

        let mut containers = TombstoneContainers::new(&self.encoded)?;
        let mut inserted = false;
        let adds_chunk = !tombstone_chunk_exists(&self.encoded, chunk_id)?;
        let next_chunk_count = chunk_count
            .checked_add(u32::from(adds_chunk))
            .ok_or_else(|| invalid("tombstone chunk count overflow"))?;
        put_u32(&mut output, next_chunk_count);
        while let Some(container) = containers.next_container()? {
            if !inserted && chunk_id < container.chunk_id {
                write_singleton_tombstone_chunk(&mut output, chunk_id, low);
                inserted = true;
            }
            if container.chunk_id == chunk_id {
                write_inserted_tombstone_container(&mut output, container, low)?;
                inserted = true;
            } else {
                output.extend_from_slice(container.encoded);
            }
        }
        if !inserted {
            write_singleton_tombstone_chunk(&mut output, chunk_id, low);
        }
        let next_cardinality = self
            .cardinality
            .checked_add(1)
            .ok_or_else(|| invalid("tombstone cardinality overflow"))?;
        debug_assert_eq!(
            validate_tombstone_bytes(&output, None).ok(),
            Some(next_cardinality)
        );
        self.encoded = output;
        self.cardinality = next_cardinality;
        Ok(true)
    }

    /// Remove one id, retaining BITMAP through 3,584 and demoting at 3,583.
    ///
    /// Retained MANIFEST segments prohibit shrinking tombstones; this mutation
    /// is intended for isolated codec work and later replacement-segment
    /// compaction, not an in-place publication transition.
    ///
    /// # Errors
    ///
    /// Returns an allocation or invariant error without changing the set.
    pub fn remove(&mut self, global_docid: u32) -> Result<bool, ManifestCodecError> {
        if !self.contains(global_docid) {
            return Ok(false);
        }
        let (chunk_id, low) = split_tombstone_docid(global_docid);
        let chunk_count = tombstone_chunk_count(&self.encoded)?;
        let mut containers = TombstoneContainers::new(&self.encoded)?;
        let mut target_cardinality = None;
        while let Some(container) = containers.next_container()? {
            if container.chunk_id == chunk_id {
                target_cardinality = Some(container.cardinality);
                break;
            }
        }
        let target_cardinality =
            target_cardinality.ok_or_else(|| invalid("tombstone membership/container mismatch"))?;
        let removes_chunk = target_cardinality == 1;
        let next_chunk_count = chunk_count
            .checked_sub(u32::from(removes_chunk))
            .ok_or_else(|| invalid("tombstone chunk count underflow"))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.encoded.len())
            .map_err(|error| invalid(format!("tombstone removal allocation failed: {error}")))?;
        put_u32(&mut output, next_chunk_count);
        let mut containers = TombstoneContainers::new(&self.encoded)?;
        while let Some(container) = containers.next_container()? {
            if container.chunk_id == chunk_id {
                write_removed_tombstone_container(&mut output, container, low)?;
            } else {
                output.extend_from_slice(container.encoded);
            }
        }
        let next_cardinality = self
            .cardinality
            .checked_sub(1)
            .ok_or_else(|| invalid("tombstone cardinality underflow"))?;
        debug_assert_eq!(
            validate_tombstone_bytes(&output, None).ok(),
            Some(next_cardinality)
        );
        self.encoded = output;
        self.cardinality = next_cardinality;
        Ok(true)
    }

    pub(in crate::keeper) fn validate_range(
        &self,
        range: (u64, u64),
    ) -> Result<(), ManifestCodecError> {
        let validated = validate_tombstone_bytes(&self.encoded, Some(range))?;
        if validated != self.cardinality {
            return Err(invalid("cached tombstone cardinality mismatch"));
        }
        Ok(())
    }

    pub(in crate::keeper) fn is_monotone_superset_of(
        &self,
        previous: &Self,
    ) -> Result<bool, ManifestCodecError> {
        if previous == self {
            return Ok(true);
        }
        if previous.cardinality > self.cardinality
            || !tombstones_are_subset(&previous.encoded, &self.encoded)?
        {
            return Ok(false);
        }

        // Every chunk follows the mutation state machine across generations:
        // ARRAY promotes only above 4,096 entries, while a retained BITMAP
        // never demotes in this growth-only transition. This also rejects a
        // newly introduced, prematurely promoted BITMAP in the overlap.
        let mut old = TombstoneContainers::new(&previous.encoded)?;
        let mut new = TombstoneContainers::new(&self.encoded)?;
        let mut prior = old.next_container()?;
        while let Some(current) = new.next_container()? {
            if prior.is_some_and(|container| container.chunk_id < current.chunk_id) {
                return Ok(false);
            }
            let previous_kind = prior
                .filter(|container| container.chunk_id == current.chunk_id)
                .map(|container| container.kind);
            let representation_is_valid = match previous_kind {
                Some(1) => current.kind == 1,
                Some(0) | None => {
                    current.kind == 0
                        || (current.kind == 1
                            && current.cardinality > u64::from(TOMBSTONE_ARRAY_MAX_CARDINALITY))
                }
                _ => false,
            };
            if !representation_is_valid {
                return Ok(false);
            }
            if previous_kind.is_some() {
                prior = old.next_container()?;
            }
        }
        Ok(prior.is_none())
    }

    pub(in crate::keeper) fn encoded_len(&self) -> usize {
        self.encoded.len()
    }
}

impl Default for TombstoneSet {
    fn default() -> Self {
        Self::new()
    }
}

impl TryFrom<&[u8]> for TombstoneSet {
    type Error = ManifestCodecError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Self::from_bytes(bytes)
    }
}

/// One immutable segment referenced by a MANIFEST generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestSegment {
    /// Random, collision-checked segment identifier.
    pub segment_id: u64,
    /// Monotone seal order used for newest-first IDHASH probing.
    pub seal_seq: u64,
    /// Exact FSLX file length.
    pub file_len: u64,
    /// xxh3-64 over the FSLX prefix before the trailer.
    pub file_xxh3: u64,
    /// Inclusive lower bound of the segment's Q1 range.
    pub docid_lo: u64,
    /// Exclusive upper bound of the segment's Q1 range.
    pub docid_hi: u64,
    /// Live-at-seal document count.
    pub doc_count: u32,
    /// Canonical roaring-lite tombstone bytes.
    pub tombstones: TombstoneSet,
}

impl ManifestSegment {
    /// Live rows after applying this generation's tombstones.
    #[must_use]
    pub fn live_doc_count(&self) -> u32 {
        self.doc_count
            .saturating_sub(u32::try_from(self.tombstones.cardinality()).unwrap_or(u32::MAX))
    }

    /// Whether a ranged physical candidate survives this segment's tombstones.
    #[must_use]
    pub fn is_live(&self, global_docid: u32) -> bool {
        (self.docid_lo..self.docid_hi).contains(&u64::from(global_docid))
            && !self.tombstones.contains(global_docid)
    }

    pub(in crate::keeper) fn insert_tombstone(
        &mut self,
        global_docid: u32,
    ) -> Result<bool, ManifestCodecError> {
        if !(self.docid_lo..self.docid_hi).contains(&u64::from(global_docid)) {
            return Err(invalid(format!(
                "tombstoned docid {global_docid} is outside segment range [{}, {})",
                self.docid_lo, self.docid_hi
            )));
        }
        if self.tombstones.contains(global_docid) {
            return Ok(false);
        }
        if self.tombstones.cardinality() >= u64::from(self.doc_count) {
            return Err(invalid(format!(
                "segment {:#018x} cannot tombstone more than {} physical rows",
                self.segment_id, self.doc_count
            )));
        }
        self.tombstones.insert(global_docid)
    }
}

/// Size class assigned from one segment's Q1 covering-interval width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentSizeTier {
    /// Width is at most [`TierMergePolicy::small_max_docid_width`].
    Small,
    /// Width is above small and at most [`TierMergePolicy::medium_max_docid_width`].
    Medium,
    /// Width exceeds the medium boundary.
    Large,
}

/// Explicit configuration consumed by the bound-consecutive tier planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierMergePolicy {
    /// Same-tier run length that triggers one concat merge.
    pub fanout: usize,
    /// Inclusive upper width of the small tier.
    pub small_max_docid_width: u64,
    /// Inclusive upper width of the medium tier.
    pub medium_max_docid_width: u64,
    /// Maximum fraction of the output hull occupied by inter-segment holes.
    pub max_hole_ratio: f64,
}

impl TierMergePolicy {
    /// Build the Keeper policy from Quill's validated engine configuration.
    #[must_use]
    pub const fn from_config(config: &crate::config::QuillConfig) -> Self {
        Self {
            fanout: config.tier_fanout,
            small_max_docid_width: config.tier_small_max_docid_width,
            medium_max_docid_width: config.tier_medium_max_docid_width,
            max_hole_ratio: config.merge_max_hole_ratio,
        }
    }

    /// Classify one nonempty covering interval by width.
    #[must_use]
    pub const fn classify_width(self, width: u64) -> SegmentSizeTier {
        if width <= self.small_max_docid_width {
            SegmentSizeTier::Small
        } else if width <= self.medium_max_docid_width {
            SegmentSizeTier::Medium
        } else {
            SegmentSizeTier::Large
        }
    }
}

/// One exact manifest slice selected for a Q1-preserving concat merge.
#[derive(Debug, Clone, PartialEq)]
pub struct TierMergePlan {
    /// Common size tier of every source segment.
    pub tier: SegmentSizeTier,
    /// Source identities in current manifest order.
    pub source_segment_ids: Vec<u64>,
    /// Inclusive lower bound of the output hull.
    pub docid_lo: u64,
    /// Exclusive upper bound of the output hull.
    pub docid_hi: u64,
    /// Fraction of the output hull represented by gaps between sources.
    pub hole_ratio: f64,
}

/// Typed failure from allocation-free policy scanning or plan materialization.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TierPolicyError {
    /// A policy must merge at least two segments at a time.
    #[error("tier merge fanout must be at least two, got {fanout}")]
    InvalidFanout {
        /// Rejected fanout.
        fanout: usize,
    },
    /// The tier boundaries must be nonzero and strictly ascending.
    #[error("tier boundaries must satisfy 0 < small < medium, got small={small}, medium={medium}")]
    InvalidBoundaries {
        /// Rejected small boundary.
        small: u64,
        /// Rejected medium boundary.
        medium: u64,
    },
    /// The optional hole-ratio gate must be a finite fraction.
    #[error("tier merge hole ratio must be finite and in [0, 1], got {ratio}")]
    InvalidHoleRatio {
        /// Rejected ratio rendered without float equality in the error type.
        ratio: String,
    },
    /// A standalone segment supplied to the planner has an empty or reversed interval.
    #[error("segment {segment_id} has invalid docid interval [{docid_lo}, {docid_hi})")]
    InvalidSegmentRange {
        /// Rejected segment identity.
        segment_id: u64,
        /// Inclusive lower bound.
        docid_lo: u64,
        /// Exclusive upper bound.
        docid_hi: u64,
    },
    /// Standalone planner inputs must already be sorted and pairwise disjoint.
    #[error("segments {left_segment_id} and {right_segment_id} overlap or are out of docid order")]
    InvalidSegmentOrder {
        /// Earlier segment identity.
        left_segment_id: u64,
        /// Later segment identity.
        right_segment_id: u64,
    },
    /// A selected source-id vector could not be reserved.
    #[error("could not allocate {count} tier merge source ids")]
    Allocation {
        /// Requested source count.
        count: usize,
    },
}

/// Select the first same-tier, bound-consecutive manifest run admitted by the
/// optional hole-ratio gate.
///
/// The planner validates interval ordering and disjointness before selecting an
/// uninterrupted slice. That slice is therefore exactly Q1 R2: the output hull
/// cannot contain a skipped live interval, including when it crosses shards.
///
/// # Errors
///
/// Returns [`TierPolicyError`] for an invalid standalone policy, malformed
/// segment layout, or allocation failure. Engine callers normally construct
/// the policy from a validated [`crate::config::QuillConfig`].
pub fn plan_tier_merge(
    segments: &[ManifestSegment],
    policy: TierMergePolicy,
) -> Result<Option<TierMergePlan>, TierPolicyError> {
    validate_tier_policy(policy)?;
    validate_tier_segments(segments)?;
    if segments.len() < policy.fanout {
        return Ok(None);
    }
    for sources in segments.windows(policy.fanout) {
        let first = &sources[0];
        let tier = policy.classify_width(first.docid_hi - first.docid_lo);
        if sources
            .iter()
            .any(|segment| policy.classify_width(segment.docid_hi - segment.docid_lo) != tier)
        {
            continue;
        }
        let docid_lo = first.docid_lo;
        let docid_hi = sources[sources.len() - 1].docid_hi;
        let hull_width = docid_hi - docid_lo;
        let occupied_width = sources
            .iter()
            .map(|segment| segment.docid_hi - segment.docid_lo)
            .sum::<u64>();
        let hole_width = hull_width.saturating_sub(occupied_width);
        let hole_ratio = if hull_width == 0 {
            0.0
        } else {
            hole_width as f64 / hull_width as f64
        };
        if hole_ratio > policy.max_hole_ratio {
            continue;
        }
        let mut source_segment_ids = Vec::new();
        source_segment_ids
            .try_reserve_exact(sources.len())
            .map_err(|_| TierPolicyError::Allocation {
                count: sources.len(),
            })?;
        source_segment_ids.extend(sources.iter().map(|segment| segment.segment_id));
        return Ok(Some(TierMergePlan {
            tier,
            source_segment_ids,
            docid_lo,
            docid_hi,
            hole_ratio,
        }));
    }
    Ok(None)
}

fn validate_tier_policy(policy: TierMergePolicy) -> Result<(), TierPolicyError> {
    if policy.fanout < 2 {
        return Err(TierPolicyError::InvalidFanout {
            fanout: policy.fanout,
        });
    }
    if policy.small_max_docid_width == 0
        || policy.medium_max_docid_width <= policy.small_max_docid_width
    {
        return Err(TierPolicyError::InvalidBoundaries {
            small: policy.small_max_docid_width,
            medium: policy.medium_max_docid_width,
        });
    }
    if !policy.max_hole_ratio.is_finite() || !(0.0..=1.0).contains(&policy.max_hole_ratio) {
        return Err(TierPolicyError::InvalidHoleRatio {
            ratio: policy.max_hole_ratio.to_string(),
        });
    }
    Ok(())
}

fn validate_tier_segments(segments: &[ManifestSegment]) -> Result<(), TierPolicyError> {
    for segment in segments {
        if segment.docid_lo >= segment.docid_hi {
            return Err(TierPolicyError::InvalidSegmentRange {
                segment_id: segment.segment_id,
                docid_lo: segment.docid_lo,
                docid_hi: segment.docid_hi,
            });
        }
    }
    for pair in segments.windows(2) {
        if pair[0].docid_hi > pair[1].docid_lo {
            return Err(TierPolicyError::InvalidSegmentOrder {
                left_segment_id: pair[0].segment_id,
                right_segment_id: pair[1].segment_id,
            });
        }
    }
    Ok(())
}

/// Snapshot-level field statistics rolled up over referenced segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestFieldStats {
    /// Schema field ordinal.
    pub field_ord: u16,
    /// Sum of at-seal token counts.
    pub total_tokens: u64,
    /// Sum of at-seal document counts for this field.
    pub doc_count: u32,
}

/// Fully owned MANIFEST v2 contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Monotone publication generation. Genesis is generation 1.
    pub generation: u64,
    /// First docid that has never been leased; never rolls back.
    pub docid_high_watermark: u64,
    /// Durable schema descriptor hash.
    pub schema_id: u64,
    /// Informational packed Quill crate version.
    pub engine_version: u32,
    /// Versioned manifest flags.
    pub flags: u32,
    /// Wall-clock seconds when this generation was published; `0` means
    /// unknown (v1 images and hand-built manifests). The publisher stamps a
    /// zero field with the current time, so every keeper-written generation
    /// carries the cross-process freshness witness required by the
    /// visibility contract (bead bd-quill-duel-visibility-contract-9rk3).
    /// Informational only: never validated for monotonicity, because NTP
    /// steps make wall-clock ordering unreliable.
    pub last_publish_unix_s: i64,
    /// Segments in strictly ascending `docid_lo` order.
    pub segments: Vec<ManifestSegment>,
    /// Field statistics in strictly ascending `field_ord` order.
    pub field_stats: Vec<ManifestFieldStats>,
}

impl Manifest {
    /// Construct an empty generation using the current engine version.
    #[must_use]
    pub const fn empty(generation: u64, schema_id: u64, docid_high_watermark: u64) -> Self {
        Self {
            generation,
            docid_high_watermark,
            schema_id,
            engine_version: CURRENT_ENGINE_VERSION,
            flags: 0,
            last_publish_unix_s: 0,
            segments: Vec::new(),
            field_stats: Vec::new(),
        }
    }

    /// Whether the durable bulk-mode flag is set.
    #[must_use]
    pub const fn bulk_mode_in_progress(&self) -> bool {
        self.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS != 0
    }

    /// Validate writer-side canonical form and Q1 range invariants.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant or resource error before any bytes are
    /// written.
    pub fn validate(&self) -> Result<(), ManifestCodecError> {
        validate_manifest(self, ErrorClass::Invalid)
    }

    /// Encode canonical MANIFEST v1 bytes including the trailing CRC32.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid in-memory state, count conversion, or an
    /// output exceeding the eager-reader budget.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ManifestCodecError> {
        validate_manifest_shape(self, ErrorClass::Invalid)?;
        let estimated = manifest_encoded_len(self)?;
        self.validate()?;

        let segment_count =
            u32::try_from(self.segments.len()).map_err(|_| ManifestCodecError::ResourceLimit {
                resource: "segment count",
                actual: usize_to_u64(self.segments.len()),
                limit: u64::from(u32::MAX),
            })?;
        let field_count = u32::try_from(self.field_stats.len()).map_err(|_| {
            ManifestCodecError::ResourceLimit {
                resource: "field count",
                actual: usize_to_u64(self.field_stats.len()),
                limit: u64::from(u32::MAX),
            }
        })?;

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(estimated)
            .map_err(|error| invalid(format!("manifest allocation failed: {error}")))?;
        bytes.extend_from_slice(&MANIFEST_MAGIC);
        put_u32(&mut bytes, MANIFEST_FORMAT_VERSION);
        put_u64(&mut bytes, self.generation);
        put_u64(&mut bytes, self.docid_high_watermark);
        put_u64(&mut bytes, self.schema_id);
        put_u32(&mut bytes, self.engine_version);
        put_u32(&mut bytes, self.flags);
        bytes.extend_from_slice(&self.last_publish_unix_s.to_le_bytes());
        put_u32(&mut bytes, segment_count);
        for segment in &self.segments {
            put_u64(&mut bytes, segment.segment_id);
            put_u64(&mut bytes, segment.seal_seq);
            put_u64(&mut bytes, segment.file_len);
            put_u64(&mut bytes, segment.file_xxh3);
            put_u64(&mut bytes, segment.docid_lo);
            put_u64(&mut bytes, segment.docid_hi);
            put_u32(&mut bytes, segment.doc_count);
            bytes.extend_from_slice(segment.tombstones.as_bytes());
        }
        put_u32(&mut bytes, field_count);
        for stats in &self.field_stats {
            put_u16(&mut bytes, stats.field_ord);
            put_u64(&mut bytes, stats.total_tokens);
            put_u32(&mut bytes, stats.doc_count);
        }
        let checksum = crc32fast::hash(&bytes);
        put_u32(&mut bytes, checksum);
        debug_assert_eq!(bytes.len(), estimated);
        Ok(bytes)
    }

    /// Parse and eagerly validate canonical MANIFEST v1 bytes.
    ///
    /// # Errors
    ///
    /// Rejects unsupported versions, checksum failures, hostile counts,
    /// non-canonical tombstones, Q1 violations, truncation, and trailing bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ManifestCodecError> {
        check_manifest_byte_limit(bytes.len())?;
        if bytes.len() < MANIFEST_MIN_BYTES {
            return Err(ManifestCodecError::Truncated {
                offset: 0,
                needed: MANIFEST_MIN_BYTES,
                remaining: bytes.len(),
            });
        }

        let body_len = bytes.len() - 4;
        let stored_checksum = u32::from_le_bytes([
            bytes[body_len],
            bytes[body_len + 1],
            bytes[body_len + 2],
            bytes[body_len + 3],
        ]);
        let computed_checksum = crc32fast::hash(&bytes[..body_len]);
        if stored_checksum != computed_checksum {
            return Err(ManifestCodecError::ChecksumMismatch {
                stored: stored_checksum,
                computed: computed_checksum,
            });
        }

        let mut cursor = ByteCursor::new(&bytes[..body_len]);
        if cursor.take(8)? != MANIFEST_MAGIC {
            return Err(ManifestCodecError::BadMagic);
        }
        let version = cursor.u32()?;
        if version != MANIFEST_FORMAT_VERSION && version != MANIFEST_FORMAT_VERSION_V1 {
            return Err(ManifestCodecError::UnsupportedVersion { found: version });
        }
        let generation = cursor.u64()?;
        let docid_high_watermark = cursor.u64()?;
        let schema_id = cursor.u64()?;
        let engine_version = cursor.u32()?;
        let flags = cursor.u32()?;
        let last_publish_unix_s = if version == MANIFEST_FORMAT_VERSION {
            i64::from_le_bytes(cursor.take(8)?.try_into().map_err(|_| {
                ManifestCodecError::Truncated {
                    offset: cursor.position(),
                    needed: 8,
                    remaining: cursor.remaining(),
                }
            })?)
        } else {
            0
        };
        let segment_count = count_to_usize(cursor.u32()?, "segment count", MAX_MANIFEST_SEGMENTS)?;
        let minimum_segment_bytes = segment_count
            .checked_mul(SEGMENT_FIXED_BYTES + EMPTY_TOMBSTONES.len())
            .ok_or_else(|| non_canonical("segment byte count overflow"))?;
        if minimum_segment_bytes > cursor.remaining() {
            return Err(ManifestCodecError::Truncated {
                offset: cursor.position(),
                needed: minimum_segment_bytes,
                remaining: cursor.remaining(),
            });
        }

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(segment_count)
            .map_err(|error| non_canonical(format!("segment allocation failed: {error}")))?;
        for _ in 0..segment_count {
            let segment_id = cursor.u64()?;
            let seal_seq = cursor.u64()?;
            let file_len = cursor.u64()?;
            let file_xxh3 = cursor.u64()?;
            let docid_lo = cursor.u64()?;
            let docid_hi = cursor.u64()?;
            let doc_count = cursor.u32()?;
            let tombstone_start = cursor.position();
            let tombstone_count = consume_tombstone_set(&mut cursor, None)?;
            let tombstone_bytes = copy_bytes(
                &bytes[tombstone_start..cursor.position()],
                "tombstone bytes",
            )?;
            let tombstones = TombstoneSet::from_validated_bytes(tombstone_bytes, tombstone_count);
            segments.push(ManifestSegment {
                segment_id,
                seal_seq,
                file_len,
                file_xxh3,
                docid_lo,
                docid_hi,
                doc_count,
                tombstones,
            });
        }

        let field_count = count_to_usize(cursor.u32()?, "field count", MAX_MANIFEST_FIELDS)?;
        let minimum_stats_bytes = field_count
            .checked_mul(FIELD_STATS_BYTES)
            .ok_or_else(|| non_canonical("field-stats byte count overflow"))?;
        if minimum_stats_bytes > cursor.remaining() {
            return Err(ManifestCodecError::Truncated {
                offset: cursor.position(),
                needed: minimum_stats_bytes,
                remaining: cursor.remaining(),
            });
        }
        let mut field_stats = Vec::new();
        field_stats
            .try_reserve_exact(field_count)
            .map_err(|error| non_canonical(format!("field-stats allocation failed: {error}")))?;
        for _ in 0..field_count {
            field_stats.push(ManifestFieldStats {
                field_ord: cursor.u16()?,
                total_tokens: cursor.u64()?,
                doc_count: cursor.u32()?,
            });
        }
        if cursor.remaining() != 0 {
            return Err(ManifestCodecError::TrailingBytes {
                remaining: cursor.remaining(),
            });
        }

        let manifest = Self {
            generation,
            docid_high_watermark,
            schema_id,
            engine_version,
            flags,
            last_publish_unix_s,
            segments,
            field_stats,
        };
        validate_manifest(&manifest, ErrorClass::NonCanonical)?;
        Ok(manifest)
    }
}

/// Slot selected by read-only two-slot recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestSource {
    /// The primary `MANIFEST` was valid.
    Current,
    /// `MANIFEST` was absent, as in the crash window between the two renames.
    PreviousAfterMissingCurrent,
    /// `MANIFEST` existed but was corrupt, so only `MANIFEST.prev` was usable.
    PreviousAfterCorruptCurrent,
    /// No filesystem is attached; this is an owned-buffer genesis snapshot.
    InMemory,
}

/// A validated MANIFEST together with its recovery provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedManifest {
    /// Validated contents.
    pub manifest: Manifest,
    /// Durable slot that supplied the contents.
    pub source: ManifestSource,
}
