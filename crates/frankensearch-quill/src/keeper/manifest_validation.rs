use std::cmp::Ordering;

use super::constants::{
    FIELD_STATS_BYTES, MANIFEST_KNOWN_FLAGS, MANIFEST_V2_MIN_BYTES, MAX_DOCID_EXCLUSIVE,
    MAX_MANIFEST_BYTES, MAX_MANIFEST_FIELDS, MAX_MANIFEST_SEGMENTS, MAX_TOMBSTONE_CHUNKS,
    SEGMENT_FIXED_BYTES, TOMBSTONE_ARRAY_MAX_CARDINALITY, TOMBSTONE_BITMAP_BYTES,
    TOMBSTONE_BITMAP_MIN_CARDINALITY,
};
use super::errors::{KeeperError, LiveDocumentFloor, ManifestCodecError, manifest_live_doc_count};
use super::manifest::{Manifest, ManifestSegment};
use super::stats_and_cursor::ByteCursor;

pub fn validate_manifest_successor(
    previous: &Manifest,
    proposed: &Manifest,
    floor: LiveDocumentFloor,
) -> Result<(), KeeperError> {
    let expected = previous
        .generation
        .checked_add(1)
        .ok_or(KeeperError::GenerationExhausted {
            current: previous.generation,
        })?;
    if proposed.generation != expected {
        return Err(KeeperError::GenerationConflict {
            expected,
            proposed: proposed.generation,
        });
    }
    if proposed.schema_id != previous.schema_id {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "schema_id changed from {:#018x} to {:#018x}",
                previous.schema_id, proposed.schema_id
            ),
        });
    }
    if proposed.docid_high_watermark < previous.docid_high_watermark {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "docid_high_watermark rolled back from {} to {}",
                previous.docid_high_watermark, proposed.docid_high_watermark
            ),
        });
    }
    validate_segment_transitions(previous, proposed, floor)
}

/// Validate the per-segment rules of one successor: retained segments keep
/// their immutable metadata and only grow tombstones, new segments carry a
/// later `seal_seq`, and the field-stat rollup only moves when the segment set
/// does. With [`LiveDocumentFloor::Enforced`], the successor must also serve
/// at least as many live documents as `previous` (gh#45).
pub(in crate::keeper) fn validate_segment_transitions(
    previous: &Manifest,
    proposed: &Manifest,
    floor: LiveDocumentFloor,
) -> Result<(), KeeperError> {
    let previous_max_seal_seq = previous
        .segments
        .iter()
        .map(|segment| segment.seal_seq)
        .max();
    let mut previous_by_id = Vec::new();
    previous_by_id
        .try_reserve_exact(previous.segments.len())
        .map_err(|error| KeeperError::InvalidTransition {
            detail: format!("previous segment-index allocation failed: {error}"),
        })?;
    previous_by_id.extend(&previous.segments);
    previous_by_id.sort_unstable_by_key(|segment| segment.segment_id);

    let mut proposed_by_id = Vec::new();
    proposed_by_id
        .try_reserve_exact(proposed.segments.len())
        .map_err(|error| KeeperError::InvalidTransition {
            detail: format!("proposed segment-index allocation failed: {error}"),
        })?;
    proposed_by_id.extend(&proposed.segments);
    proposed_by_id.sort_unstable_by_key(|segment| segment.segment_id);

    let mut previous_index = 0;
    let mut proposed_index = 0;
    let mut segment_set_changed = false;
    while previous_index < previous_by_id.len() && proposed_index < proposed_by_id.len() {
        let old = previous_by_id[previous_index];
        let new = proposed_by_id[proposed_index];
        match old.segment_id.cmp(&new.segment_id) {
            std::cmp::Ordering::Less => {
                segment_set_changed = true;
                previous_index += 1;
            }
            std::cmp::Ordering::Greater => {
                validate_new_segment_seal_seq(previous_max_seal_seq, new)?;
                segment_set_changed = true;
                proposed_index += 1;
            }
            std::cmp::Ordering::Equal => {
                if old.seal_seq != new.seal_seq
                    || old.file_len != new.file_len
                    || old.file_xxh3 != new.file_xxh3
                    || old.docid_lo != new.docid_lo
                    || old.docid_hi != new.docid_hi
                    || old.doc_count != new.doc_count
                {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "immutable metadata changed for segment {:#018x}",
                            old.segment_id
                        ),
                    });
                }
                let monotone = new
                    .tombstones
                    .is_monotone_superset_of(&old.tombstones)
                    .map_err(|error| KeeperError::InvalidTransition {
                        detail: format!(
                            "cannot compare tombstones for segment {:#018x}: {error}",
                            old.segment_id
                        ),
                    })?;
                if !monotone {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "tombstones shrank or changed for retained segment {:#018x}",
                            old.segment_id
                        ),
                    });
                }
                previous_index += 1;
                proposed_index += 1;
            }
        }
    }
    if previous_index != previous_by_id.len() {
        segment_set_changed = true;
    }
    for new in &proposed_by_id[proposed_index..] {
        validate_new_segment_seal_seq(previous_max_seal_seq, new)?;
        segment_set_changed = true;
    }
    if !segment_set_changed && previous.field_stats != proposed.field_stats {
        return Err(KeeperError::InvalidTransition {
            detail: "field-stat rollup changed while the immutable segment set was unchanged"
                .to_owned(),
        });
    }
    if floor == LiveDocumentFloor::Enforced {
        // Every rule above is per segment. Tombstone growth on a retained
        // segment and outright segment removal are each legal on their own,
        // which is exactly how a replacement whose rows never landed publishes
        // as a valid, quietly hollow generation. The floor is the one rule
        // that looks at the corpus as a whole.
        let previous_live_docs = manifest_live_doc_count(previous);
        let proposed_live_docs = manifest_live_doc_count(proposed);
        if proposed_live_docs < previous_live_docs {
            return Err(KeeperError::LiveDocumentFloor {
                previous_generation: previous.generation,
                previous_live_docs,
                proposed_generation: proposed.generation,
                proposed_live_docs,
            });
        }
    }
    Ok(())
}

fn validate_new_segment_seal_seq(
    previous_max_seal_seq: Option<u64>,
    segment: &ManifestSegment,
) -> Result<(), KeeperError> {
    if let Some(previous_max) = previous_max_seal_seq
        && segment.seal_seq <= previous_max
    {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "new segment {:#018x} seal_seq {} does not advance prior maximum {}",
                segment.segment_id, segment.seal_seq, previous_max
            ),
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(in crate::keeper) struct TombstoneContainer<'a> {
    pub(in crate::keeper) chunk_id: u16,
    pub(in crate::keeper) kind: u8,
    pub(in crate::keeper) cardinality: u64,
    pub(in crate::keeper) payload: &'a [u8],
    pub(in crate::keeper) payload_offset: usize,
    pub(in crate::keeper) encoded: &'a [u8],
}

pub(in crate::keeper) struct TombstoneContainers<'a> {
    cursor: ByteCursor<'a>,
    remaining: usize,
}

impl<'a> TombstoneContainers<'a> {
    pub(in crate::keeper) fn new(bytes: &'a [u8]) -> Result<Self, ManifestCodecError> {
        let mut cursor = ByteCursor::new(bytes);
        let remaining =
            count_to_usize(cursor.u32()?, "tombstone chunk count", MAX_TOMBSTONE_CHUNKS)?;
        Ok(Self { cursor, remaining })
    }

    pub(in crate::keeper) fn next_container(
        &mut self,
    ) -> Result<Option<TombstoneContainer<'a>>, ManifestCodecError> {
        if self.remaining == 0 {
            if self.cursor.remaining() != 0 {
                return Err(non_canonical(format!(
                    "tombstone set has {} trailing bytes",
                    self.cursor.remaining()
                )));
            }
            return Ok(None);
        }

        let container_start = self.cursor.position();
        let chunk_id = self.cursor.u16()?;
        let kind = self.cursor.u8()?;
        let encoded_count = self.cursor.u16()?;
        let (cardinality, payload_length) = match kind {
            0 => (u64::from(encoded_count), usize::from(encoded_count) * 2),
            1 => (
                if encoded_count == 0 {
                    65_536
                } else {
                    u64::from(encoded_count)
                },
                TOMBSTONE_BITMAP_BYTES,
            ),
            other => {
                return Err(non_canonical(format!(
                    "tombstone container has unknown kind {other}"
                )));
            }
        };
        let payload_offset = self.cursor.position();
        let payload = self.cursor.take(payload_length)?;
        let encoded = &self.cursor.bytes[container_start..self.cursor.position()];
        self.remaining -= 1;
        Ok(Some(TombstoneContainer {
            chunk_id,
            kind,
            cardinality,
            payload,
            payload_offset,
            encoded,
        }))
    }
}

pub(in crate::keeper) fn tombstone_chunk_count(bytes: &[u8]) -> Result<u32, ManifestCodecError> {
    let mut cursor = ByteCursor::new(bytes);
    cursor.u32()
}

pub(in crate::keeper) fn tombstone_chunk_exists(
    bytes: &[u8],
    chunk_id: u16,
) -> Result<bool, ManifestCodecError> {
    let mut containers = TombstoneContainers::new(bytes)?;
    while let Some(container) = containers.next_container()? {
        match container.chunk_id.cmp(&chunk_id) {
            std::cmp::Ordering::Less => {}
            std::cmp::Ordering::Equal => return Ok(true),
            std::cmp::Ordering::Greater => return Ok(false),
        }
    }
    Ok(false)
}

pub(in crate::keeper) fn write_tombstone_container_header(
    output: &mut Vec<u8>,
    chunk_id: u16,
    kind: u8,
    cardinality: u64,
) -> Result<(), ManifestCodecError> {
    put_u16(output, chunk_id);
    output.push(kind);
    let encoded_count = if kind == 1 && cardinality == 65_536 {
        0
    } else {
        u16::try_from(cardinality)
            .map_err(|_| invalid("tombstone container cardinality does not fit wire count"))?
    };
    put_u16(output, encoded_count);
    Ok(())
}

pub(in crate::keeper) fn write_singleton_tombstone_chunk(
    output: &mut Vec<u8>,
    chunk_id: u16,
    low: u16,
) {
    put_u16(output, chunk_id);
    output.push(0);
    put_u16(output, 1);
    put_u16(output, low);
}

pub(in crate::keeper) fn write_inserted_tombstone_container(
    output: &mut Vec<u8>,
    container: TombstoneContainer<'_>,
    low: u16,
) -> Result<(), ManifestCodecError> {
    let next_cardinality = container
        .cardinality
        .checked_add(1)
        .ok_or_else(|| invalid("tombstone container cardinality overflow"))?;
    match container.kind {
        0 if container.cardinality < u64::from(TOMBSTONE_ARRAY_MAX_CARDINALITY) => {
            write_tombstone_container_header(output, container.chunk_id, 0, next_cardinality)?;
            let mut emitted = false;
            for index in 0..container.payload.len() / 2 {
                let current = array_value(container.payload, index);
                if !emitted && low < current {
                    put_u16(output, low);
                    emitted = true;
                }
                put_u16(output, current);
            }
            if !emitted {
                put_u16(output, low);
            }
        }
        0 => {
            write_tombstone_container_header(output, container.chunk_id, 1, next_cardinality)?;
            let payload_start = output.len();
            output.resize(payload_start + TOMBSTONE_BITMAP_BYTES, 0);
            for index in 0..container.payload.len() / 2 {
                set_bitmap_value(
                    &mut output[payload_start..],
                    array_value(container.payload, index),
                );
            }
            set_bitmap_value(&mut output[payload_start..], low);
        }
        1 => {
            let encoded_start = output.len();
            output.extend_from_slice(container.encoded);
            let count_offset = encoded_start + 3;
            let encoded_count = if next_cardinality == 65_536 {
                0
            } else {
                u16::try_from(next_cardinality)
                    .map_err(|_| invalid("bitmap tombstone cardinality does not fit wire count"))?
            };
            output[count_offset..count_offset + 2].copy_from_slice(&encoded_count.to_le_bytes());
            let payload_start = encoded_start + 5;
            set_bitmap_value(
                &mut output[payload_start..payload_start + TOMBSTONE_BITMAP_BYTES],
                low,
            );
        }
        _ => return Err(invalid("unknown validated tombstone container kind")),
    }
    Ok(())
}

pub(in crate::keeper) fn write_removed_tombstone_container(
    output: &mut Vec<u8>,
    container: TombstoneContainer<'_>,
    low: u16,
) -> Result<(), ManifestCodecError> {
    let next_cardinality = container
        .cardinality
        .checked_sub(1)
        .ok_or_else(|| invalid("tombstone container cardinality underflow"))?;
    match container.kind {
        0 if next_cardinality == 0 => {}
        0 => {
            write_tombstone_container_header(output, container.chunk_id, 0, next_cardinality)?;
            for index in 0..container.payload.len() / 2 {
                let current = array_value(container.payload, index);
                if current != low {
                    put_u16(output, current);
                }
            }
        }
        1 if next_cardinality < TOMBSTONE_BITMAP_MIN_CARDINALITY => {
            write_tombstone_container_header(output, container.chunk_id, 0, next_cardinality)?;
            for (byte_index, byte) in container.payload.iter().copied().enumerate() {
                let mut bits = byte;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let value = u16::try_from(byte_index * 8 + bit)
                        .map_err(|_| invalid("bitmap tombstone low-bit overflow"))?;
                    if value != low {
                        put_u16(output, value);
                    }
                    bits &= bits - 1;
                }
            }
        }
        1 => {
            let encoded_start = output.len();
            output.extend_from_slice(container.encoded);
            let count_offset = encoded_start + 3;
            let encoded_count = if next_cardinality == 65_536 {
                0
            } else {
                u16::try_from(next_cardinality)
                    .map_err(|_| invalid("bitmap tombstone cardinality does not fit wire count"))?
            };
            output[count_offset..count_offset + 2].copy_from_slice(&encoded_count.to_le_bytes());
            let payload_start = encoded_start + 5;
            clear_bitmap_value(
                &mut output[payload_start..payload_start + TOMBSTONE_BITMAP_BYTES],
                low,
            );
        }
        _ => return Err(invalid("unknown validated tombstone container kind")),
    }
    Ok(())
}

pub(in crate::keeper) fn array_binary_search(bytes: &[u8], value: u16) -> bool {
    let mut lo = 0_usize;
    let mut hi = bytes.len() / 2;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match array_value(bytes, mid).cmp(&value) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    false
}

pub(in crate::keeper) fn set_bitmap_value(bitmap: &mut [u8], value: u16) {
    let value = usize::from(value);
    bitmap[value / 8] |= 1 << (value % 8);
}

pub(in crate::keeper) fn clear_bitmap_value(bitmap: &mut [u8], value: u16) {
    let value = usize::from(value);
    bitmap[value / 8] &= !(1 << (value % 8));
}

pub(in crate::keeper) fn split_tombstone_docid(global_docid: u32) -> (u16, u16) {
    let bytes = global_docid.to_le_bytes();
    (
        u16::from_le_bytes([bytes[2], bytes[3]]),
        u16::from_le_bytes([bytes[0], bytes[1]]),
    )
}

pub(in crate::keeper) fn tombstones_are_subset(
    previous: &[u8],
    proposed: &[u8],
) -> Result<bool, ManifestCodecError> {
    if previous == proposed {
        return Ok(true);
    }
    let mut previous = TombstoneContainers::new(previous)?;
    let mut proposed = TombstoneContainers::new(proposed)?;
    let mut old = previous.next_container()?;
    let mut new = proposed.next_container()?;

    while let Some(old_container) = old {
        while new.is_some_and(|container| container.chunk_id < old_container.chunk_id) {
            new = proposed.next_container()?;
        }
        let Some(new_container) = new else {
            return Ok(false);
        };
        if new_container.chunk_id != old_container.chunk_id
            || !tombstone_container_is_subset(old_container, new_container)
        {
            return Ok(false);
        }
        old = previous.next_container()?;
        new = proposed.next_container()?;
    }
    Ok(true)
}

fn tombstone_container_is_subset(
    previous: TombstoneContainer<'_>,
    proposed: TombstoneContainer<'_>,
) -> bool {
    if previous.cardinality > proposed.cardinality {
        return false;
    }
    match (previous.kind, proposed.kind) {
        (0, 0) => array_is_subset_of_array(previous.payload, proposed.payload),
        (0, 1) => array_is_subset_of_bitmap(previous.payload, proposed.payload),
        (1, 0) => bitmap_is_subset_of_array(previous.payload, proposed.payload),
        (1, 1) => previous
            .payload
            .iter()
            .zip(proposed.payload)
            .all(|(old, new)| old & !new == 0),
        _ => false,
    }
}

fn array_is_subset_of_array(previous: &[u8], proposed: &[u8]) -> bool {
    let mut proposed_index = 0;
    for previous_index in 0..previous.len() / 2 {
        let old = array_value(previous, previous_index);
        while proposed_index < proposed.len() / 2 && array_value(proposed, proposed_index) < old {
            proposed_index += 1;
        }
        if proposed_index == proposed.len() / 2 || array_value(proposed, proposed_index) != old {
            return false;
        }
        proposed_index += 1;
    }
    true
}

fn array_is_subset_of_bitmap(previous: &[u8], proposed: &[u8]) -> bool {
    (0..previous.len() / 2).all(|index| bitmap_contains(proposed, array_value(previous, index)))
}

fn bitmap_is_subset_of_array(previous: &[u8], proposed: &[u8]) -> bool {
    let mut proposed_index = 0;
    for (byte_index, byte) in previous.iter().copied().enumerate() {
        let mut bits = byte;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let Ok(old) = u16::try_from(byte_index * 8 + bit) else {
                return false;
            };
            while proposed_index < proposed.len() / 2 && array_value(proposed, proposed_index) < old
            {
                proposed_index += 1;
            }
            if proposed_index == proposed.len() / 2 || array_value(proposed, proposed_index) != old
            {
                return false;
            }
            proposed_index += 1;
            bits &= bits - 1;
        }
    }
    true
}

pub(in crate::keeper) fn array_value(bytes: &[u8], index: usize) -> u16 {
    let offset = index * 2;
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

pub(in crate::keeper) fn bitmap_contains(bitmap: &[u8], value: u16) -> bool {
    let value = usize::from(value);
    bitmap[value / 8] & (1 << (value % 8)) != 0
}

#[derive(Debug, Clone, Copy)]
pub(in crate::keeper) enum ErrorClass {
    Invalid,
    NonCanonical,
}

pub(in crate::keeper) fn validate_manifest_shape(
    manifest: &Manifest,
    error_class: ErrorClass,
) -> Result<(), ManifestCodecError> {
    let reject = |detail: String| match error_class {
        ErrorClass::Invalid => ManifestCodecError::Invalid { detail },
        ErrorClass::NonCanonical => ManifestCodecError::NonCanonical { detail },
    };

    if manifest.generation == 0 {
        return Err(reject("generation zero is reserved".to_owned()));
    }
    if manifest.flags & !MANIFEST_KNOWN_FLAGS != 0 {
        return Err(reject(format!(
            "unknown flag bits {:#010x}",
            manifest.flags & !MANIFEST_KNOWN_FLAGS
        )));
    }
    if manifest.segments.len() > MAX_MANIFEST_SEGMENTS {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "segment count",
            actual: usize_to_u64(manifest.segments.len()),
            limit: usize_to_u64(MAX_MANIFEST_SEGMENTS),
        });
    }
    if manifest.field_stats.len() > MAX_MANIFEST_FIELDS {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "field count",
            actual: usize_to_u64(manifest.field_stats.len()),
            limit: usize_to_u64(MAX_MANIFEST_FIELDS),
        });
    }
    Ok(())
}

pub(in crate::keeper) fn validate_manifest(
    manifest: &Manifest,
    error_class: ErrorClass,
) -> Result<(), ManifestCodecError> {
    validate_manifest_shape(manifest, error_class)?;
    let reject = |detail: String| match error_class {
        ErrorClass::Invalid => ManifestCodecError::Invalid { detail },
        ErrorClass::NonCanonical => ManifestCodecError::NonCanonical { detail },
    };

    let mut segment_ids = Vec::new();
    segment_ids
        .try_reserve_exact(manifest.segments.len())
        .map_err(|error| reject(format!("segment-id validation allocation failed: {error}")))?;
    segment_ids.extend(manifest.segments.iter().map(|segment| segment.segment_id));
    segment_ids.sort_unstable();
    if let Some([duplicate, ..]) = segment_ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(reject(format!("duplicate segment_id {duplicate:#018x}")));
    }

    let mut seal_sequences = Vec::new();
    seal_sequences
        .try_reserve_exact(manifest.segments.len())
        .map_err(|error| {
            reject(format!(
                "seal-sequence validation allocation failed: {error}"
            ))
        })?;
    seal_sequences.extend(manifest.segments.iter().map(|segment| segment.seal_seq));
    seal_sequences.sort_unstable();
    if let Some([duplicate, ..]) = seal_sequences.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(reject(format!("duplicate seal_seq {duplicate}")));
    }

    let mut previous_hi = None;
    let mut maximum_hi = 0_u64;
    let mut total_documents = 0_u64;
    for (index, segment) in manifest.segments.iter().enumerate() {
        if segment.file_len == 0 {
            return Err(reject(format!("segment {index} has zero file_len")));
        }
        if segment.docid_lo >= segment.docid_hi {
            return Err(reject(format!(
                "segment {index} has empty or reversed range [{}, {})",
                segment.docid_lo, segment.docid_hi
            )));
        }
        if segment.docid_hi > MAX_DOCID_EXCLUSIVE {
            return Err(reject(format!(
                "segment {index} range exceeds the u32 payload domain"
            )));
        }
        if previous_hi.is_some_and(|hi| segment.docid_lo < hi) {
            return Err(reject(format!(
                "segment {index} overlaps or is out of docid_lo order"
            )));
        }
        let span = segment.docid_hi - segment.docid_lo;
        if u64::from(segment.doc_count) > span {
            return Err(reject(format!(
                "segment {index} doc_count {} exceeds range span {span}",
                segment.doc_count
            )));
        }
        segment
            .tombstones
            .validate_range((segment.docid_lo, segment.docid_hi))?;
        let tombstone_count = segment.tombstones.cardinality();
        if tombstone_count > u64::from(segment.doc_count) {
            return Err(reject(format!(
                "segment {index} tombstone count {tombstone_count} exceeds doc_count {}",
                segment.doc_count
            )));
        }
        total_documents = total_documents
            .checked_add(u64::from(segment.doc_count))
            .ok_or_else(|| reject("aggregate segment doc_count overflow".to_owned()))?;
        maximum_hi = maximum_hi.max(segment.docid_hi);
        previous_hi = Some(segment.docid_hi);
    }
    if manifest.docid_high_watermark < maximum_hi {
        return Err(reject(format!(
            "docid_high_watermark {} is below maximum docid_hi {maximum_hi}",
            manifest.docid_high_watermark
        )));
    }

    let mut previous_field = None;
    for stats in &manifest.field_stats {
        if previous_field.is_some_and(|field| stats.field_ord <= field) {
            return Err(reject(format!(
                "field stats are not strictly ordered at field_ord {}",
                stats.field_ord
            )));
        }
        if u64::from(stats.doc_count) != total_documents {
            return Err(reject(format!(
                "field {} doc_count {} differs from canonical at-seal denominator {total_documents} (bd-quill-e3-keeper-ndtk.10)",
                stats.field_ord, stats.doc_count
            )));
        }
        previous_field = Some(stats.field_ord);
    }
    Ok(())
}

pub(in crate::keeper) fn validate_tombstone_bytes(
    bytes: &[u8],
    range: Option<(u64, u64)>,
) -> Result<u64, ManifestCodecError> {
    let mut cursor = ByteCursor::new(bytes);
    let count = consume_tombstone_set(&mut cursor, range)?;
    if cursor.remaining() != 0 {
        return Err(non_canonical(format!(
            "tombstone set has {} trailing bytes",
            cursor.remaining()
        )));
    }
    Ok(count)
}

pub(in crate::keeper) fn consume_tombstone_set(
    cursor: &mut ByteCursor<'_>,
    range: Option<(u64, u64)>,
) -> Result<u64, ManifestCodecError> {
    let chunk_count = count_to_usize(cursor.u32()?, "tombstone chunk count", MAX_TOMBSTONE_CHUNKS)?;
    let mut previous_chunk = None;
    let mut total = 0_u64;
    for chunk_index in 0..chunk_count {
        let chunk_id = cursor.u16()?;
        if previous_chunk.is_some_and(|previous| chunk_id <= previous) {
            return Err(non_canonical(format!(
                "tombstone chunk {chunk_index} is not strictly ordered"
            )));
        }
        previous_chunk = Some(chunk_id);
        let kind = cursor.u8()?;
        let encoded_count = cursor.u16()?;
        let cardinality = match kind {
            0 => {
                if encoded_count == 0 {
                    return Err(non_canonical(format!(
                        "array tombstone chunk {chunk_index} is empty"
                    )));
                }
                if encoded_count > TOMBSTONE_ARRAY_MAX_CARDINALITY {
                    return Err(non_canonical(format!(
                        "array tombstone chunk {chunk_index} cardinality {encoded_count} exceeds promotion threshold {TOMBSTONE_ARRAY_MAX_CARDINALITY}"
                    )));
                }
                let count = usize::from(encoded_count);
                let mut previous_low = None;
                for _ in 0..count {
                    let low = cursor.u16()?;
                    if previous_low.is_some_and(|previous| low <= previous) {
                        return Err(non_canonical(format!(
                            "array tombstone chunk {chunk_index} is not strictly ordered"
                        )));
                    }
                    validate_tombstone_docid(chunk_id, low, range)?;
                    previous_low = Some(low);
                }
                u64::from(encoded_count)
            }
            1 => {
                let bitmap = cursor.take(8_192)?;
                let actual = bitmap
                    .iter()
                    .map(|byte| u64::from(byte.count_ones()))
                    .sum::<u64>();
                let expected = if encoded_count == 0 {
                    65_536
                } else {
                    u64::from(encoded_count)
                };
                if actual != expected {
                    return Err(non_canonical(format!(
                        "bitmap tombstone chunk {chunk_index} cardinality {actual} != {expected}"
                    )));
                }
                if actual < TOMBSTONE_BITMAP_MIN_CARDINALITY {
                    return Err(non_canonical(format!(
                        "bitmap tombstone chunk {chunk_index} cardinality {actual} is below demotion threshold {TOMBSTONE_BITMAP_MIN_CARDINALITY}"
                    )));
                }
                if let Some(bounds) = range {
                    validate_bitmap_range(bitmap, chunk_id, bounds)?;
                }
                actual
            }
            other => {
                return Err(non_canonical(format!(
                    "tombstone chunk {chunk_index} has unknown kind {other}"
                )));
            }
        };
        total = total
            .checked_add(cardinality)
            .ok_or_else(|| non_canonical("tombstone cardinality overflow"))?;
    }
    Ok(total)
}

fn validate_bitmap_range(
    bitmap: &[u8],
    chunk_id: u16,
    bounds: (u64, u64),
) -> Result<(), ManifestCodecError> {
    let (lo, hi) = bounds;
    let chunk_lo = u64::from(chunk_id) << 16;
    let chunk_hi = chunk_lo + 65_536;
    if lo <= chunk_lo && chunk_hi <= hi {
        return Ok(());
    }
    if hi <= chunk_lo || lo >= chunk_hi {
        return Err(non_canonical(format!(
            "bitmap tombstone chunk {chunk_id} is outside segment range [{lo}, {hi})"
        )));
    }

    for (byte_index, byte) in bitmap.iter().copied().enumerate() {
        let mut bits = byte;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            let low = u16::try_from(byte_index * 8 + bit as usize)
                .map_err(|_| non_canonical("bitmap low-bit overflow"))?;
            validate_tombstone_docid(chunk_id, low, Some(bounds))?;
            bits &= bits - 1;
        }
    }
    Ok(())
}

fn validate_tombstone_docid(
    chunk_id: u16,
    low: u16,
    range: Option<(u64, u64)>,
) -> Result<(), ManifestCodecError> {
    if let Some((lo, hi)) = range {
        let docid = (u64::from(chunk_id) << 16) | u64::from(low);
        if !(lo..hi).contains(&docid) {
            return Err(non_canonical(format!(
                "tombstoned docid {docid} is outside segment range [{lo}, {hi})"
            )));
        }
    }
    Ok(())
}

pub(in crate::keeper) fn check_manifest_byte_limit(
    length: usize,
) -> Result<(), ManifestCodecError> {
    if length > MAX_MANIFEST_BYTES {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: usize_to_u64(length),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        });
    }
    Ok(())
}

pub(in crate::keeper) fn manifest_encoded_len(
    manifest: &Manifest,
) -> Result<usize, ManifestCodecError> {
    let length = MANIFEST_V2_MIN_BYTES
        .checked_add(
            manifest
                .segments
                .len()
                .checked_mul(SEGMENT_FIXED_BYTES)
                .ok_or_else(|| invalid("manifest encoded length overflow"))?,
        )
        .and_then(|length| {
            manifest.segments.iter().try_fold(length, |total, segment| {
                total.checked_add(segment.tombstones.encoded_len())
            })
        })
        .and_then(|length| {
            manifest
                .field_stats
                .len()
                .checked_mul(FIELD_STATS_BYTES)
                .and_then(|stats| length.checked_add(stats))
        })
        .ok_or_else(|| invalid("manifest encoded length overflow"))?;
    check_manifest_byte_limit(length)?;
    Ok(length)
}

pub(in crate::keeper) fn count_to_usize(
    count: u32,
    resource: &'static str,
    limit: usize,
) -> Result<usize, ManifestCodecError> {
    let count = usize::try_from(count).map_err(|_| ManifestCodecError::ResourceLimit {
        resource,
        actual: u64::from(count),
        limit: usize_to_u64(limit),
    })?;
    if count > limit {
        return Err(ManifestCodecError::ResourceLimit {
            resource,
            actual: usize_to_u64(count),
            limit: usize_to_u64(limit),
        });
    }
    Ok(count)
}

pub(in crate::keeper) fn copy_bytes(
    bytes: &[u8],
    label: &'static str,
) -> Result<Vec<u8>, ManifestCodecError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(bytes.len())
        .map_err(|error| non_canonical(format!("{label} allocation failed: {error}")))?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}

pub(in crate::keeper) fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub(in crate::keeper) fn invalid(detail: impl Into<String>) -> ManifestCodecError {
    ManifestCodecError::Invalid {
        detail: detail.into(),
    }
}

pub(in crate::keeper) fn non_canonical(detail: impl Into<String>) -> ManifestCodecError {
    ManifestCodecError::NonCanonical {
        detail: detail.into(),
    }
}

pub(in crate::keeper) fn put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(in crate::keeper) fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(in crate::keeper) fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
