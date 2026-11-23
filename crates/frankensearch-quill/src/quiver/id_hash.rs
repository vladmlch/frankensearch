use frankensearch_core::DocId;
use thiserror::Error;
use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::id_map::{
    ID_HASH_ENTRY_LEN, ID_HASH_HEADER_LEN, ID_HASH_SEED, ID_MAP_HASH_LEN, ID_MAP_HEADER_LEN,
    IdMapConcatPlan, IdMapSection, id_map_hash, id_map_offset,
};
use super::{read_u32, read_u64};

/// Explicit resource ceilings for an FSLX IDHASH section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdHashLimits {
    /// Maximum present IDMAP rows examined while building or validating.
    pub max_entries: u64,
    /// Maximum power-of-two slot capacity.
    pub max_capacity: u64,
    /// Maximum complete section size.
    pub max_section_bytes: u64,
    /// Maximum cumulative linear-probe steps in one build or validation pass.
    pub max_probe_steps: u64,
}

impl Default for IdHashLimits {
    fn default() -> Self {
        Self {
            max_entries: u64::from(u32::MAX),
            max_capacity: 1_u64 << 28,
            max_section_bytes: u64::from(u32::MAX),
            max_probe_steps: 16_777_216,
        }
    }
}

/// Typed failures from IDHASH encoding or cross-section validation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum IdHashCodecError {
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("IDHASH {resource} {actual} exceeds limit {limit}")]
    ResourceLimit {
        /// Bounded resource name.
        resource: &'static str,
        /// Rejected amount.
        actual: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// Checked layout arithmetic overflowed.
    #[error("IDHASH arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout component being computed.
        field: &'static str,
    },
    /// Durable capacity must be non-zero.
    #[error("IDHASH capacity is zero")]
    ZeroCapacity,
    /// Durable capacity must be a power of two.
    #[error("IDHASH capacity {capacity} is not a power of two")]
    NonPowerOfTwoCapacity {
        /// Rejected slot count.
        capacity: u32,
    },
    /// The writer has exactly one canonical minimum capacity.
    #[error("IDHASH capacity {encoded} is non-canonical; expected {expected}")]
    NonCanonicalCapacity {
        /// Durable slot count.
        encoded: u32,
        /// Required slot count.
        expected: u32,
    },
    /// More than 70 percent of the durable slots were occupied.
    #[error("IDHASH load {entries}/{capacity} exceeds 0.7")]
    LoadFactorExceeded {
        /// Occupied slots.
        entries: usize,
        /// Total slots.
        capacity: usize,
    },
    /// Durable bytes ended before the canonical layout did.
    #[error("truncated IDHASH section: expected {expected} bytes, got {actual}")]
    Truncated {
        /// Required length.
        expected: usize,
        /// Available length.
        actual: usize,
    },
    /// Canonical sections end exactly after their final slot.
    #[error("IDHASH section has trailing bytes: expected {expected}, got {actual}")]
    TrailingBytes {
        /// Canonical length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Empty slots are canonical all-zero rows.
    #[error("IDHASH empty slot {slot} has non-zero key hash {key_hash:#018x}")]
    NonZeroEmptyHash {
        /// Rejected slot.
        slot: usize,
        /// Rejected durable hash.
        key_hash: u64,
    },
    /// A slot's segment-relative ordinal exceeded the IDMAP span.
    #[error("IDHASH slot {slot} references out-of-range ordinal {ordinal}")]
    OrdinalOutOfRange {
        /// Rejected slot.
        slot: usize,
        /// Decoded segment-relative ordinal.
        ordinal: u32,
    },
    /// A slot referenced an IDMAP hole.
    #[error("IDHASH slot {slot} references IDMAP hole ordinal {ordinal}")]
    OrdinalHole {
        /// Rejected slot.
        slot: usize,
        /// Segment-relative ordinal.
        ordinal: u32,
    },
    /// A durable seeded hash disagreed with the exact referenced ID bytes.
    #[error(
        "IDHASH slot {slot} key hash {encoded:#018x} disagrees with IDMAP hash {expected:#018x}"
    )]
    KeyHashMismatch {
        /// Rejected slot.
        slot: usize,
        /// Durable hash.
        encoded: u64,
        /// Required seeded xxh3 hash.
        expected: u64,
    },
    /// A candidate representative set selected the same exact identifier more than once.
    #[error(
        "IDHASH duplicate document id at ordinals {first_ordinal} and {duplicate_ordinal}; representative selection requires external tombstone/upsert context"
    )]
    DuplicateDocumentId {
        /// First physical occurrence.
        first_ordinal: usize,
        /// Later physical occurrence that made the mapping ambiguous.
        duplicate_ordinal: usize,
    },
    /// Merge-resolved representative ordinals must be strictly ascending for canonical insertion.
    #[error(
        "IDHASH representative ordinals are not strictly ascending at index {index}: previous {previous}, current {current}"
    )]
    NonAscendingRepresentativeOrdinals {
        /// Rejected representative-set index.
        index: usize,
        /// Previous segment-relative ordinal.
        previous: u32,
        /// Current segment-relative ordinal.
        current: u32,
    },
    /// A caller-selected representative ordinal exceeded the bound IDMAP span.
    #[error("IDHASH representative {index} references out-of-range ordinal {ordinal}")]
    RepresentativeOrdinalOutOfRange {
        /// Rejected representative-set index.
        index: usize,
        /// Segment-relative ordinal.
        ordinal: u32,
    },
    /// A caller-selected representative ordinal referenced an IDMAP hole.
    #[error("IDHASH representative {index} references IDMAP hole ordinal {ordinal}")]
    RepresentativeOrdinalHole {
        /// Rejected representative-set index.
        index: usize,
        /// Segment-relative ordinal.
        ordinal: u32,
    },
    /// A present IDMAP identifier was absent or unreachable through probing.
    #[error("IDHASH has no reachable mapping for IDMAP ordinal {ordinal}")]
    MissingMapping {
        /// Present IDMAP ordinal.
        ordinal: usize,
    },
    /// A slot was duplicated, unreachable, or referenced from a noncanonical cluster.
    #[error("IDHASH slot {slot} is not the canonical reachable mapping for ordinal {ordinal}")]
    NonCanonicalSlot {
        /// Rejected slot.
        slot: usize,
        /// Referenced IDMAP ordinal.
        ordinal: usize,
    },
    /// Every occupied slot must be the selected representative for an exact identifier.
    #[error(
        "IDHASH occupied slot count {occupied} disagrees with {selected} selected representatives"
    )]
    SelectedCountMismatch {
        /// Durable occupied slots.
        occupied: usize,
        /// Selected exact-identifier representatives reachable from IDMAP rows.
        selected: usize,
    },
    /// Linear probing exhausted every slot while encoding.
    #[error("IDHASH table with capacity {capacity} is full")]
    TableFull {
        /// Exhausted slot count.
        capacity: usize,
    },
    /// Reconstructing an absolute global docid overflowed the segment range.
    #[error("IDHASH ordinal {ordinal} cannot be rebased from docid_lo {docid_lo}")]
    DocIdOverflow {
        /// Segment-relative ordinal.
        ordinal: usize,
        /// Segment lower bound.
        docid_lo: u64,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for IDHASH {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

/// Owned canonical bytes produced by the IDHASH rebuild writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedIdHashSection {
    pub(super) bytes: Vec<u8>,
    capacity: u32,
    occupied: usize,
}

impl EncodedIdHashSection {
    /// Rebuild IDHASH deterministically from a validated IDMAP.
    ///
    /// A standalone section has no tombstone/upsert context, so duplicate
    /// physical identifiers fail closed instead of guessing a representative.
    /// The seal or merge layer must supply its externally resolved
    /// representative set; tombstones remain external to the immutable section.
    ///
    /// # Errors
    ///
    /// Rejects duplicate-ID ambiguity, resource abuse, arithmetic overflow, or
    /// allocation failure.
    pub fn encode(id_map: IdMapSection<'_>) -> Result<Self, IdHashCodecError> {
        Self::encode_with_limits(id_map, IdHashLimits::default())
    }

    /// Rebuild with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`].
    pub fn encode_with_limits(
        id_map: IdMapSection<'_>,
        limits: IdHashLimits,
    ) -> Result<Self, IdHashCodecError> {
        encode_id_hash_with_hasher(id_map, limits, seeded_id_hash)
    }

    /// Rebuild IDHASH from representatives already resolved by the seal or merge layer.
    ///
    /// `representative_ordinals` must contain exactly one physical
    /// representative for every exact identifier in `id_map`, in strictly
    /// ascending segment-relative ordinal order. With external tombstone state,
    /// the caller selects the unique live physical row for an identifier and
    /// fails if the upsert invariant admits multiple live rows. If every row is
    /// tombstoned, the caller retains the lowest global docid as a deterministic
    /// completeness representative. A merge container's `seal_seq` is only
    /// publication/probe priority and must not override that live-first choice.
    /// Sorting the resolved ordinals only after selection pins deterministic
    /// linear-probe insertion.
    ///
    /// # Errors
    ///
    /// Rejects unsorted, out-of-range, hole, duplicate, or incomplete representative
    /// sets, plus the resource and allocation failures from [`Self::encode`].
    pub fn encode_resolved_representatives(
        id_map: IdMapSection<'_>,
        representative_ordinals: &[u32],
    ) -> Result<Self, IdHashCodecError> {
        Self::encode_resolved_representatives_with_limits(
            id_map,
            representative_ordinals,
            IdHashLimits::default(),
        )
    }

    /// Rebuild IDHASH directly from the validated sources of a conceptual
    /// merged IDMAP, without first materializing that IDMAP's durable bytes.
    pub(crate) fn encode_resolved_concat(
        id_maps: &[IdMapSection<'_>],
        id_map_plan: &IdMapConcatPlan,
        representative_ordinals: &[u32],
    ) -> Result<Self, IdHashCodecError> {
        let docid_lo = id_map_plan.docid_lo;
        encode_resolved_id_hash_with_domain(
            docid_lo,
            id_map_plan.span,
            id_map_plan.present_count,
            |ordinal| concat_id_map_document_id_at_ordinal(id_maps, docid_lo, ordinal),
            id_maps
                .iter()
                .copied()
                .flat_map(|id_map| id_map.document_ids()),
            representative_ordinals,
            IdHashLimits::default(),
            seeded_id_hash,
        )
    }

    /// Rebuild from externally resolved representatives with caller-selected ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode_resolved_representatives`].
    pub fn encode_resolved_representatives_with_limits(
        id_map: IdMapSection<'_>,
        representative_ordinals: &[u32],
        limits: IdHashLimits,
    ) -> Result<Self, IdHashCodecError> {
        encode_resolved_id_hash_with_hasher(id_map, representative_ordinals, limits, seeded_id_hash)
    }

    /// Borrow the exact canonical durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Canonical power-of-two slot capacity.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of unique occupied identifier slots.
    #[must_use]
    pub const fn occupied(&self) -> usize {
        self.occupied
    }

    /// Re-open the owned bytes through the cross-validating reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated or `id_map` is
    /// not the exact matching mapping domain.
    pub fn section<'a>(
        &'a self,
        id_map: IdMapSection<'a>,
    ) -> Result<IdHashSection<'a>, IdHashCodecError> {
        IdHashSection::parse(&self.bytes, id_map)
    }
}

/// Borrowed IDHASH view that is query-capable only after exact IDMAP binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdHashSection<'a> {
    pub(super) bytes: &'a [u8],
    entries: &'a [u8],
    capacity: usize,
    occupied: usize,
    id_map: IdMapSection<'a>,
}

/// Non-borrowing probe metadata derived from a fully validated IDMAP/IDHASH
/// pair.
///
/// Keeper retains the immutable segment bytes separately. This compact plan
/// lets its delete path perform ordinary hash-table probes without reparsing
/// every physical identifier on each lookup or constructing a self-referential
/// cached section view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdHashLookupPlan {
    docid_lo: u64,
    span: usize,
    dense: bool,
    id_map_blob_offset: usize,
    id_map_len: usize,
    id_hash_len: usize,
    capacity: usize,
}

/// Admission-derived physical-row presence decision for one IDMAP domain.
///
/// A dense plan has already proved that every in-range ordinal has an IDMAP
/// entry. Hole-bearing plans deliberately defer to the byte-backed lookup so
/// the existing offset and identifier validation path remains authoritative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdMapPresence {
    /// The global document id is outside the admitted IDMAP domain.
    Absent,
    /// Admission proved that this in-range document id has a physical row.
    Present,
    /// The domain contains holes, so the IDMAP bytes must decide this row.
    RequiresIdMap,
}

impl<'a> IdHashSection<'a> {
    /// Validate durable slots against the exact IDMAP bytes used for lookups.
    ///
    /// # Errors
    ///
    /// Rejects invalid capacity/layout/load, non-zero empty rows, invalid or
    /// hole ordinals, hash mismatches, missing/duplicate/unreachable mappings,
    /// noncanonical placement, resource abuse, and allocation-independent
    /// overflow.
    pub fn parse(bytes: &'a [u8], id_map: IdMapSection<'a>) -> Result<Self, IdHashCodecError> {
        Self::parse_with_limits(bytes, id_map, IdHashLimits::default())
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`].
    pub fn parse_with_limits(
        bytes: &'a [u8],
        id_map: IdMapSection<'a>,
        limits: IdHashLimits,
    ) -> Result<Self, IdHashCodecError> {
        parse_id_hash_with_hasher(bytes, id_map, limits, seeded_id_hash)
    }

    /// Exact durable bytes after cross-section validation.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Power-of-two slot capacity.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.capacity
    }

    /// Number of unique occupied slots.
    #[must_use]
    pub const fn occupied(self) -> usize {
        self.occupied
    }

    /// Capture allocation-free lookup metadata after full cross-section
    /// validation.
    #[must_use]
    pub(crate) fn lookup_plan(self) -> IdHashLookupPlan {
        let span = self.id_map.offset_count();
        IdHashLookupPlan {
            docid_lo: self.id_map.docid_lo,
            span,
            dense: self.id_map.present_count() == span,
            id_map_blob_offset: self.id_map.bytes.len() - self.id_map.blob.len(),
            id_map_len: self.id_map.bytes.len(),
            id_hash_len: self.bytes.len(),
            capacity: self.capacity,
        }
    }

    /// Probe a document identifier, verifying exact IDMAP bytes on hash hits.
    #[must_use]
    pub fn lookup(self, document_id: &str) -> Option<u64> {
        self.lookup_with_hash(document_id, seeded_id_hash(document_id.as_bytes()))
    }

    /// Probe and materialize the exact stored identifier for a winning hit.
    #[must_use]
    pub fn lookup_materialized(self, document_id: &str) -> Option<(u64, DocId)> {
        let global_docid = self.lookup(document_id)?;
        Some((global_docid, self.id_map.materialize(global_docid)?))
    }

    pub(super) fn lookup_with_hash(self, document_id: &str, key_hash: u64) -> Option<u64> {
        let mut probe_steps = 0_u64;
        self.lookup_with_hash_and_budget(document_id, key_hash, &mut probe_steps, u64::MAX)
            .ok()
            .flatten()
    }

    fn lookup_with_hash_and_budget(
        self,
        document_id: &str,
        key_hash: u64,
        probe_steps: &mut u64,
        max_probe_steps: u64,
    ) -> Result<Option<u64>, IdHashCodecError> {
        let Some(mask) = self.capacity.checked_sub(1) else {
            return Ok(None);
        };
        let Some(mask_u64) = u64::try_from(mask).ok() else {
            return Ok(None);
        };
        let Some(mut slot) = usize::try_from(key_hash & mask_u64).ok() else {
            return Ok(None);
        };
        for _ in 0..self.capacity {
            consume_id_hash_probe_step(probe_steps, max_probe_steps)?;
            let Some((stored_hash, doc_ord_plus1)) = id_hash_slot(self.entries, slot) else {
                return Ok(None);
            };
            if doc_ord_plus1 == 0 {
                return Ok(None);
            }
            if stored_hash == key_hash {
                let Some(ordinal) = usize::try_from(doc_ord_plus1 - 1).ok() else {
                    return Ok(None);
                };
                let Some(stored_document_id) = self.id_map.document_id_at_ordinal(ordinal) else {
                    return Ok(None);
                };
                if stored_document_id.as_bytes() == document_id.as_bytes() {
                    let Some(ordinal) = u64::try_from(ordinal).ok() else {
                        return Ok(None);
                    };
                    return Ok(self.id_map.docid_lo.checked_add(ordinal));
                }
            }
            slot = (slot + 1) & mask;
        }
        Ok(None)
    }
}

impl IdHashLookupPlan {
    /// Decide physical-row presence before accessing the IDMAP section.
    ///
    /// `Present` is possible only when admission proved a dense map; a plan
    /// with holes always returns [`IdMapPresence::RequiresIdMap`] in range so
    /// callers retain the pre-existing byte-backed lookup.
    #[must_use]
    pub fn identity_row_presence(self, global_docid: u32) -> IdMapPresence {
        let Some(ordinal) = u64::from(global_docid).checked_sub(self.docid_lo) else {
            return IdMapPresence::Absent;
        };
        let Ok(ordinal) = usize::try_from(ordinal) else {
            return IdMapPresence::Absent;
        };
        if ordinal >= self.span {
            return IdMapPresence::Absent;
        }
        if self.dense {
            IdMapPresence::Present
        } else {
            IdMapPresence::RequiresIdMap
        }
    }

    /// Whether the bound IDMAP contains a physical row for this global docid.
    #[must_use]
    pub fn contains_global_docid(self, id_map_bytes: &[u8], global_docid: u32) -> bool {
        if id_map_bytes.len() != self.id_map_len {
            return false;
        }
        let Some(ordinal) = u64::from(global_docid).checked_sub(self.docid_lo) else {
            return false;
        };
        let Ok(ordinal) = usize::try_from(ordinal) else {
            return false;
        };
        self.document_id_bytes(id_map_bytes, ordinal).is_some()
    }

    /// Materialize exactly one bound IDMAP row by global document id.
    ///
    /// This performs two checked offset reads and one UTF-8 slice lookup. The
    /// complete IDMAP/IDHASH pair was validated before this plan was cached, so
    /// winner materialization never reparses or rescans the section span.
    #[must_use]
    pub fn materialize_global_docid(self, id_map_bytes: &[u8], global_docid: u32) -> Option<DocId> {
        if id_map_bytes.len() != self.id_map_len {
            return None;
        }
        let ordinal = u64::from(global_docid).checked_sub(self.docid_lo)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        let document_id = self.document_id_bytes(id_map_bytes, ordinal)?;
        Some(DocId::new(std::str::from_utf8(document_id).ok()?))
    }

    /// Probe one exact identifier and return its positional content witness.
    ///
    /// This shares the allocation-free IDHASH walk used by ordinary identity
    /// resolution, then reads only the winning row's validated IDMAP hash.
    #[must_use]
    pub fn lookup_with_content_hash(
        self,
        id_map_bytes: &[u8],
        id_hash_bytes: &[u8],
        document_id: &str,
    ) -> Option<(u64, u64)> {
        if id_map_bytes.len() != self.id_map_len || id_hash_bytes.len() != self.id_hash_len {
            return None;
        }
        let entries = id_hash_bytes.get(ID_HASH_HEADER_LEN..)?;
        let mask = self.capacity.checked_sub(1)?;
        let mask_u64 = u64::try_from(mask).ok()?;
        let key_hash = seeded_id_hash(document_id.as_bytes());
        let mut slot = usize::try_from(key_hash & mask_u64).ok()?;
        for _ in 0..self.capacity {
            let (stored_hash, doc_ord_plus1) = id_hash_slot(entries, slot)?;
            if doc_ord_plus1 == 0 {
                return None;
            }
            if stored_hash == key_hash {
                let ordinal = usize::try_from(doc_ord_plus1 - 1).ok()?;
                let stored_document_id = self.document_id_bytes(id_map_bytes, ordinal)?;
                if stored_document_id == document_id.as_bytes() {
                    let global_docid = self.docid_lo.checked_add(u64::try_from(ordinal).ok()?)?;
                    let content_hash = self.content_hash(id_map_bytes, ordinal)?;
                    return Some((global_docid, content_hash));
                }
            }
            slot = (slot + 1) & mask;
        }
        None
    }

    fn content_hash(self, id_map_bytes: &[u8], ordinal: usize) -> Option<u64> {
        if ordinal >= self.span {
            return None;
        }
        let hashes_len = self.span.checked_mul(ID_MAP_HASH_LEN)?;
        let hashes_start = self.id_map_blob_offset.checked_sub(hashes_len)?;
        let hashes = id_map_bytes.get(hashes_start..self.id_map_blob_offset)?;
        id_map_hash(hashes, ordinal)
    }

    fn document_id_bytes(self, id_map_bytes: &[u8], ordinal: usize) -> Option<&[u8]> {
        if ordinal >= self.span {
            return None;
        }
        let offsets = id_map_bytes.get(ID_MAP_HEADER_LEN..self.id_map_blob_offset)?;
        let start = usize::try_from(id_map_offset(offsets, ordinal)?).ok()?;
        let end = usize::try_from(id_map_offset(offsets, ordinal.checked_add(1)?)?).ok()?;
        if start == end {
            return None;
        }
        let blob = id_map_bytes.get(self.id_map_blob_offset..)?;
        blob.get(start..end)
    }
}

fn consume_id_hash_probe_step(
    probe_steps: &mut u64,
    max_probe_steps: u64,
) -> Result<(), IdHashCodecError> {
    *probe_steps = probe_steps.saturating_add(1);
    if *probe_steps > max_probe_steps {
        return Err(IdHashCodecError::ResourceLimit {
            resource: "probe steps",
            actual: *probe_steps,
            limit: max_probe_steps,
        });
    }
    Ok(())
}

pub(super) fn seeded_id_hash(document_id: &[u8]) -> u64 {
    xxh3_64_with_seed(document_id, ID_HASH_SEED)
}

pub(super) fn encode_id_hash_with_hasher<F>(
    id_map: IdMapSection<'_>,
    limits: IdHashLimits,
    hash: F,
) -> Result<EncodedIdHashSection, IdHashCodecError>
where
    F: Fn(&[u8]) -> u64 + Copy,
{
    validate_id_hash_entry_limit(id_map.present_count, limits)?;
    let provisional_capacity = canonical_id_hash_capacity(id_map.present_count, limits)?;
    let (mut entries, occupied) =
        build_id_hash_entries(id_map, provisional_capacity, limits, hash)?;
    let canonical_capacity = canonical_id_hash_capacity(occupied, limits)?;
    if canonical_capacity != provisional_capacity {
        entries = build_id_hash_entries(id_map, canonical_capacity, limits, hash)?.0;
    }
    let total_len = id_hash_section_len(canonical_capacity, limits)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total_len)
        .map_err(|_| IdHashCodecError::Allocation {
            resource: "section bytes",
            bytes: total_len,
        })?;
    let capacity_u32 =
        u32::try_from(canonical_capacity).map_err(|_| IdHashCodecError::ResourceLimit {
            resource: "durable capacity",
            actual: u64::try_from(canonical_capacity).unwrap_or(u64::MAX),
            limit: u64::from(u32::MAX),
        })?;
    bytes.extend_from_slice(&capacity_u32.to_le_bytes());
    bytes.extend_from_slice(&entries);
    debug_assert_eq!(bytes.len(), total_len);
    let encoded = EncodedIdHashSection {
        bytes,
        capacity: capacity_u32,
        occupied,
    };
    // A successful writer must always reopen under the same limits, including
    // the cumulative probe-work ceiling.
    parse_id_hash_with_hasher(encoded.as_bytes(), id_map, limits, hash)?;
    Ok(encoded)
}

pub(super) fn encode_resolved_id_hash_with_hasher<F>(
    id_map: IdMapSection<'_>,
    representative_ordinals: &[u32],
    limits: IdHashLimits,
    hash: F,
) -> Result<EncodedIdHashSection, IdHashCodecError>
where
    F: Fn(&[u8]) -> u64 + Copy,
{
    encode_resolved_id_hash_with_domain(
        id_map.docid_lo,
        id_map.offset_count(),
        id_map.present_count,
        |ordinal| id_map.document_id_at_ordinal(ordinal),
        id_map.document_ids(),
        representative_ordinals,
        limits,
        hash,
    )
}

fn concat_id_map_document_id_at_ordinal<'a>(
    id_maps: &[IdMapSection<'a>],
    docid_lo: u64,
    ordinal: usize,
) -> Option<&'a str> {
    let global_docid = docid_lo.checked_add(u64::try_from(ordinal).ok()?)?;
    let section_index = id_maps.partition_point(|id_map| id_map.docid_hi() <= global_docid);
    id_maps
        .get(section_index)
        .copied()?
        .get(global_docid)
        .map(|entry| entry.document_id())
}

fn encode_resolved_id_hash_with_domain<'a, F, I, G>(
    docid_lo: u64,
    span: usize,
    present_count: usize,
    document_id_at_ordinal: G,
    document_ids: I,
    representative_ordinals: &[u32],
    limits: IdHashLimits,
    hash: F,
) -> Result<EncodedIdHashSection, IdHashCodecError>
where
    F: Fn(&[u8]) -> u64 + Copy,
    G: Fn(usize) -> Option<&'a str> + Copy,
    I: IntoIterator<Item = (u64, &'a str)>,
{
    // Validation examines the complete physical IDMAP even though the durable
    // table contains only one externally selected row per exact identifier.
    validate_id_hash_entry_limit(present_count, limits)?;
    validate_id_hash_entry_limit(representative_ordinals.len(), limits)?;
    let capacity = canonical_id_hash_capacity(representative_ordinals.len(), limits)?;
    let entries = build_resolved_id_hash_entries(
        span,
        document_id_at_ordinal,
        representative_ordinals,
        capacity,
        limits,
        hash,
    )?;
    let total_len = id_hash_section_len(capacity, limits)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total_len)
        .map_err(|_| IdHashCodecError::Allocation {
            resource: "section bytes",
            bytes: total_len,
        })?;
    let capacity_u32 = u32::try_from(capacity).map_err(|_| IdHashCodecError::ResourceLimit {
        resource: "durable capacity",
        actual: u64::try_from(capacity).unwrap_or(u64::MAX),
        limit: u64::from(u32::MAX),
    })?;
    bytes.extend_from_slice(&capacity_u32.to_le_bytes());
    bytes.extend_from_slice(&entries);
    debug_assert_eq!(bytes.len(), total_len);
    let encoded = EncodedIdHashSection {
        bytes,
        capacity: capacity_u32,
        occupied: representative_ordinals.len(),
    };
    // This proves that the externally supplied set covers every exact IDMAP
    // identifier and that its durable bytes satisfy the same reader contract.
    validate_id_hash_with_domain(
        encoded.as_bytes(),
        docid_lo,
        span,
        present_count,
        document_id_at_ordinal,
        document_ids,
        limits,
        hash,
    )?;
    Ok(encoded)
}

fn build_resolved_id_hash_entries<'a, F, G>(
    span: usize,
    document_id_at_ordinal: G,
    representative_ordinals: &[u32],
    capacity: usize,
    limits: IdHashLimits,
    hash: F,
) -> Result<Vec<u8>, IdHashCodecError>
where
    F: Fn(&[u8]) -> u64 + Copy,
    G: Fn(usize) -> Option<&'a str> + Copy,
{
    let entries_len =
        capacity
            .checked_mul(ID_HASH_ENTRY_LEN)
            .ok_or(IdHashCodecError::ArithmeticOverflow {
                field: "entry bytes",
            })?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entries_len)
        .map_err(|_| IdHashCodecError::Allocation {
            resource: "entry bytes",
            bytes: entries_len,
        })?;
    entries.resize(entries_len, 0);
    let mask = capacity
        .checked_sub(1)
        .ok_or(IdHashCodecError::ZeroCapacity)?;
    let mut previous = None;
    let mut probe_steps = 0_u64;
    for (index, &ordinal_u32) in representative_ordinals.iter().enumerate() {
        if let Some(previous) = previous
            && ordinal_u32 <= previous
        {
            return Err(IdHashCodecError::NonAscendingRepresentativeOrdinals {
                index,
                previous,
                current: ordinal_u32,
            });
        }
        previous = Some(ordinal_u32);
        let ordinal = usize::try_from(ordinal_u32).map_err(|_| {
            IdHashCodecError::RepresentativeOrdinalOutOfRange {
                index,
                ordinal: ordinal_u32,
            }
        })?;
        if ordinal >= span {
            return Err(IdHashCodecError::RepresentativeOrdinalOutOfRange {
                index,
                ordinal: ordinal_u32,
            });
        }
        let document_id =
            document_id_at_ordinal(ordinal).ok_or(IdHashCodecError::RepresentativeOrdinalHole {
                index,
                ordinal: ordinal_u32,
            })?;
        let doc_ord_plus1 = ordinal_u32.checked_add(1).ok_or(
            IdHashCodecError::RepresentativeOrdinalOutOfRange {
                index,
                ordinal: ordinal_u32,
            },
        )?;
        let key_hash = hash(document_id.as_bytes());
        let mut slot = usize::try_from(key_hash & u64::try_from(mask).unwrap_or(u64::MAX))
            .map_err(|_| IdHashCodecError::ArithmeticOverflow {
                field: "initial resolved-representative probe slot",
            })?;
        let mut inserted = false;
        for _ in 0..capacity {
            consume_id_hash_probe_step(&mut probe_steps, limits.max_probe_steps)?;
            let (stored_hash, stored_plus1) =
                id_hash_slot(&entries, slot).ok_or(IdHashCodecError::ArithmeticOverflow {
                    field: "resolved-representative probe slot read",
                })?;
            if stored_plus1 == 0 {
                write_id_hash_slot(&mut entries, slot, key_hash, doc_ord_plus1)?;
                inserted = true;
                break;
            }
            if stored_hash == key_hash {
                let stored_ordinal = usize::try_from(stored_plus1 - 1).map_err(|_| {
                    IdHashCodecError::ArithmeticOverflow {
                        field: "stored resolved-representative ordinal",
                    }
                })?;
                let stored_document_id = document_id_at_ordinal(stored_ordinal).ok_or(
                    IdHashCodecError::OrdinalHole {
                        slot,
                        ordinal: stored_plus1 - 1,
                    },
                )?;
                if stored_document_id.as_bytes() == document_id.as_bytes() {
                    return Err(IdHashCodecError::DuplicateDocumentId {
                        first_ordinal: stored_ordinal,
                        duplicate_ordinal: ordinal,
                    });
                }
            }
            slot = (slot + 1) & mask;
        }
        if !inserted {
            return Err(IdHashCodecError::TableFull { capacity });
        }
    }
    Ok(entries)
}

pub(super) fn build_id_hash_entries<F>(
    id_map: IdMapSection<'_>,
    capacity: usize,
    limits: IdHashLimits,
    hash: F,
) -> Result<(Vec<u8>, usize), IdHashCodecError>
where
    F: Fn(&[u8]) -> u64 + Copy,
{
    let entries_len =
        capacity
            .checked_mul(ID_HASH_ENTRY_LEN)
            .ok_or(IdHashCodecError::ArithmeticOverflow {
                field: "entry bytes",
            })?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entries_len)
        .map_err(|_| IdHashCodecError::Allocation {
            resource: "entry bytes",
            bytes: entries_len,
        })?;
    entries.resize(entries_len, 0);
    let mask = capacity
        .checked_sub(1)
        .ok_or(IdHashCodecError::ZeroCapacity)?;
    let mut occupied = 0_usize;
    let mut probe_steps = 0_u64;
    for (global_docid, document_id) in id_map.document_ids() {
        let ordinal_u64 =
            global_docid
                .checked_sub(id_map.docid_lo)
                .ok_or(IdHashCodecError::DocIdOverflow {
                    ordinal: 0,
                    docid_lo: id_map.docid_lo,
                })?;
        let ordinal =
            usize::try_from(ordinal_u64).map_err(|_| IdHashCodecError::ResourceLimit {
                resource: "host ordinal",
                actual: ordinal_u64,
                limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
            })?;
        let doc_ord_plus1 = u32::try_from(ordinal)
            .ok()
            .and_then(|ordinal| ordinal.checked_add(1))
            .ok_or(IdHashCodecError::DocIdOverflow {
                ordinal,
                docid_lo: id_map.docid_lo,
            })?;
        let key_hash = hash(document_id.as_bytes());
        let mut slot = usize::try_from(key_hash & u64::try_from(mask).unwrap_or(u64::MAX))
            .map_err(|_| IdHashCodecError::ArithmeticOverflow {
                field: "initial probe slot",
            })?;
        let mut inserted = false;
        for _ in 0..capacity {
            consume_id_hash_probe_step(&mut probe_steps, limits.max_probe_steps)?;
            let (stored_hash, stored_plus1) =
                id_hash_slot(&entries, slot).ok_or(IdHashCodecError::ArithmeticOverflow {
                    field: "probe slot read",
                })?;
            if stored_plus1 == 0 {
                write_id_hash_slot(&mut entries, slot, key_hash, doc_ord_plus1)?;
                occupied += 1;
                inserted = true;
                break;
            }
            if stored_hash == key_hash {
                let stored_ordinal = usize::try_from(stored_plus1 - 1).map_err(|_| {
                    IdHashCodecError::ArithmeticOverflow {
                        field: "stored ordinal",
                    }
                })?;
                let stored_document_id = id_map.document_id_at_ordinal(stored_ordinal).ok_or(
                    IdHashCodecError::OrdinalHole {
                        slot,
                        ordinal: stored_plus1 - 1,
                    },
                )?;
                if stored_document_id.as_bytes() == document_id.as_bytes() {
                    return Err(IdHashCodecError::DuplicateDocumentId {
                        first_ordinal: stored_ordinal,
                        duplicate_ordinal: ordinal,
                    });
                }
            }
            slot = (slot + 1) & mask;
        }
        if !inserted {
            return Err(IdHashCodecError::TableFull { capacity });
        }
    }
    Ok((entries, occupied))
}

struct ValidatedIdHash<'a> {
    entries: &'a [u8],
    capacity: usize,
    occupied: usize,
}

pub(super) fn parse_id_hash_with_hasher<'a, F>(
    bytes: &'a [u8],
    id_map: IdMapSection<'a>,
    limits: IdHashLimits,
    hash: F,
) -> Result<IdHashSection<'a>, IdHashCodecError>
where
    F: Fn(&[u8]) -> u64 + Copy,
{
    let validated = validate_id_hash_with_domain(
        bytes,
        id_map.docid_lo,
        id_map.offset_count(),
        id_map.present_count,
        |ordinal| id_map.document_id_at_ordinal(ordinal),
        id_map.document_ids(),
        limits,
        hash,
    )?;
    Ok(IdHashSection {
        bytes,
        entries: validated.entries,
        capacity: validated.capacity,
        occupied: validated.occupied,
        id_map,
    })
}

fn validate_id_hash_with_domain<'bytes, 'ids, F, I, G>(
    bytes: &'bytes [u8],
    docid_lo: u64,
    span: usize,
    present_count: usize,
    document_id_at_ordinal: G,
    document_ids: I,
    limits: IdHashLimits,
    hash: F,
) -> Result<ValidatedIdHash<'bytes>, IdHashCodecError>
where
    F: Fn(&[u8]) -> u64 + Copy,
    G: Fn(usize) -> Option<&'ids str> + Copy,
    I: IntoIterator<Item = (u64, &'ids str)>,
{
    validate_id_hash_entry_limit(present_count, limits)?;
    if bytes.len() < ID_HASH_HEADER_LEN {
        return Err(IdHashCodecError::Truncated {
            expected: ID_HASH_HEADER_LEN,
            actual: bytes.len(),
        });
    }
    let capacity_u32 = read_u32(bytes, 0).ok_or(IdHashCodecError::Truncated {
        expected: ID_HASH_HEADER_LEN,
        actual: bytes.len(),
    })?;
    if capacity_u32 == 0 {
        return Err(IdHashCodecError::ZeroCapacity);
    }
    if !capacity_u32.is_power_of_two() {
        return Err(IdHashCodecError::NonPowerOfTwoCapacity {
            capacity: capacity_u32,
        });
    }
    let capacity = usize::try_from(capacity_u32).map_err(|_| IdHashCodecError::ResourceLimit {
        resource: "host capacity",
        actual: u64::from(capacity_u32),
        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
    })?;
    let capacity_u64 = u64::from(capacity_u32);
    if capacity_u64 > limits.max_capacity {
        return Err(IdHashCodecError::ResourceLimit {
            resource: "capacity",
            actual: capacity_u64,
            limit: limits.max_capacity,
        });
    }
    let expected_len = id_hash_section_len(capacity, limits)?;
    if bytes.len() < expected_len {
        return Err(IdHashCodecError::Truncated {
            expected: expected_len,
            actual: bytes.len(),
        });
    }
    if bytes.len() > expected_len {
        return Err(IdHashCodecError::TrailingBytes {
            expected: expected_len,
            actual: bytes.len(),
        });
    }
    let entries = bytes
        .get(ID_HASH_HEADER_LEN..)
        .ok_or(IdHashCodecError::Truncated {
            expected: expected_len,
            actual: bytes.len(),
        })?;
    let mut occupied = 0_usize;
    for slot in 0..capacity {
        let (key_hash, doc_ord_plus1) =
            id_hash_slot(entries, slot).ok_or(IdHashCodecError::Truncated {
                expected: expected_len,
                actual: bytes.len(),
            })?;
        if doc_ord_plus1 == 0 {
            if key_hash != 0 {
                return Err(IdHashCodecError::NonZeroEmptyHash { slot, key_hash });
            }
            continue;
        }
        occupied += 1;
        let ordinal_u32 = doc_ord_plus1 - 1;
        let ordinal =
            usize::try_from(ordinal_u32).map_err(|_| IdHashCodecError::OrdinalOutOfRange {
                slot,
                ordinal: ordinal_u32,
            })?;
        if ordinal >= span {
            return Err(IdHashCodecError::OrdinalOutOfRange {
                slot,
                ordinal: ordinal_u32,
            });
        }
        let document_id = document_id_at_ordinal(ordinal).ok_or(IdHashCodecError::OrdinalHole {
            slot,
            ordinal: ordinal_u32,
        })?;
        let expected_hash = hash(document_id.as_bytes());
        if key_hash != expected_hash {
            return Err(IdHashCodecError::KeyHashMismatch {
                slot,
                encoded: key_hash,
                expected: expected_hash,
            });
        }
    }
    if occupied
        .checked_mul(10)
        .is_none_or(|scaled| scaled > capacity.saturating_mul(7))
    {
        return Err(IdHashCodecError::LoadFactorExceeded {
            entries: occupied,
            capacity,
        });
    }
    let canonical_capacity = canonical_id_hash_capacity(occupied, limits)?;
    if canonical_capacity != capacity {
        return Err(IdHashCodecError::NonCanonicalCapacity {
            encoded: capacity_u32,
            expected: u32::try_from(canonical_capacity).unwrap_or(u32::MAX),
        });
    }
    // Validate exact ascending-selected-ordinal insertion without allocating a
    // mirror table. One bounded probe per physical IDMAP row both finds its
    // exact representative and proves that every preceding cluster ordinal was
    // inserted earlier. Duplicate physical rows resolve to the same selected
    // representative; only that representative's own row increments `selected`.
    let mask = capacity - 1;
    let mut selected = 0_usize;
    let mut probe_steps = 0_u64;
    for (global_docid, document_id) in document_ids {
        let ordinal = global_docid
            .checked_sub(docid_lo)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(IdHashCodecError::DocIdOverflow {
                ordinal: 0,
                docid_lo,
            })?;
        let key_hash = hash(document_id.as_bytes());
        let mut slot = usize::try_from(key_hash & u64::try_from(mask).unwrap_or(u64::MAX))
            .map_err(|_| IdHashCodecError::ArithmeticOverflow {
                field: "canonical probe slot",
            })?;
        let mut found = false;
        let mut max_preceding_ordinal = None;
        for _ in 0..capacity {
            consume_id_hash_probe_step(&mut probe_steps, limits.max_probe_steps)?;
            let (durable_hash, doc_ord_plus1) =
                id_hash_slot(entries, slot).ok_or(IdHashCodecError::Truncated {
                    expected: expected_len,
                    actual: bytes.len(),
                })?;
            if doc_ord_plus1 == 0 {
                return Err(IdHashCodecError::MissingMapping { ordinal });
            }
            let durable_ordinal = usize::try_from(doc_ord_plus1 - 1).map_err(|_| {
                IdHashCodecError::OrdinalOutOfRange {
                    slot,
                    ordinal: doc_ord_plus1 - 1,
                }
            })?;
            if durable_hash == key_hash {
                let durable_document_id = document_id_at_ordinal(durable_ordinal).ok_or(
                    IdHashCodecError::OrdinalHole {
                        slot,
                        ordinal: doc_ord_plus1 - 1,
                    },
                )?;
                if durable_document_id.as_bytes() == document_id.as_bytes() {
                    if max_preceding_ordinal.is_some_and(|preceding| preceding >= durable_ordinal) {
                        return Err(IdHashCodecError::NonCanonicalSlot {
                            slot,
                            ordinal: durable_ordinal,
                        });
                    }
                    if durable_ordinal == ordinal {
                        selected += 1;
                    }
                    found = true;
                    break;
                }
            }
            max_preceding_ordinal = Some(
                max_preceding_ordinal.map_or(durable_ordinal, |preceding: usize| {
                    preceding.max(durable_ordinal)
                }),
            );
            slot = (slot + 1) & mask;
        }
        if !found {
            return Err(IdHashCodecError::MissingMapping { ordinal });
        }
    }
    if occupied != selected {
        return Err(IdHashCodecError::SelectedCountMismatch { occupied, selected });
    }
    Ok(ValidatedIdHash {
        entries,
        capacity,
        occupied,
    })
}

fn validate_id_hash_entry_limit(
    entries: usize,
    limits: IdHashLimits,
) -> Result<(), IdHashCodecError> {
    let entries_u64 = u64::try_from(entries).unwrap_or(u64::MAX);
    if entries_u64 > limits.max_entries {
        return Err(IdHashCodecError::ResourceLimit {
            resource: "entry count",
            actual: entries_u64,
            limit: limits.max_entries,
        });
    }
    Ok(())
}

fn canonical_id_hash_capacity(
    entries: usize,
    limits: IdHashLimits,
) -> Result<usize, IdHashCodecError> {
    let mut capacity = 1_usize;
    while entries
        .checked_mul(10)
        .is_none_or(|scaled| scaled > capacity.saturating_mul(7))
    {
        capacity = capacity
            .checked_mul(2)
            .ok_or(IdHashCodecError::ArithmeticOverflow {
                field: "capacity growth",
            })?;
    }
    let capacity_u64 = u64::try_from(capacity).unwrap_or(u64::MAX);
    let durable_limit = limits.max_capacity.min(u64::from(u32::MAX));
    if capacity_u64 > durable_limit {
        return Err(IdHashCodecError::ResourceLimit {
            resource: "capacity",
            actual: capacity_u64,
            limit: durable_limit,
        });
    }
    Ok(capacity)
}

fn id_hash_section_len(capacity: usize, limits: IdHashLimits) -> Result<usize, IdHashCodecError> {
    let total_len = capacity
        .checked_mul(ID_HASH_ENTRY_LEN)
        .and_then(|entries| entries.checked_add(ID_HASH_HEADER_LEN))
        .ok_or(IdHashCodecError::ArithmeticOverflow {
            field: "section length",
        })?;
    let total_len_u64 = u64::try_from(total_len).unwrap_or(u64::MAX);
    if total_len_u64 > limits.max_section_bytes {
        return Err(IdHashCodecError::ResourceLimit {
            resource: "section bytes",
            actual: total_len_u64,
            limit: limits.max_section_bytes,
        });
    }
    Ok(total_len)
}

pub(super) fn id_hash_slot(entries: &[u8], slot: usize) -> Option<(u64, u32)> {
    let offset = slot.checked_mul(ID_HASH_ENTRY_LEN)?;
    let ordinal_offset = offset.checked_add(std::mem::size_of::<u64>())?;
    Some((
        read_u64(entries, offset)?,
        read_u32(entries, ordinal_offset)?,
    ))
}

pub(super) fn write_id_hash_slot(
    entries: &mut [u8],
    slot: usize,
    key_hash: u64,
    doc_ord_plus1: u32,
) -> Result<(), IdHashCodecError> {
    let offset =
        slot.checked_mul(ID_HASH_ENTRY_LEN)
            .ok_or(IdHashCodecError::ArithmeticOverflow {
                field: "slot offset",
            })?;
    let end = offset
        .checked_add(ID_HASH_ENTRY_LEN)
        .ok_or(IdHashCodecError::ArithmeticOverflow { field: "slot end" })?;
    let slot_bytes = entries
        .get_mut(offset..end)
        .ok_or(IdHashCodecError::ArithmeticOverflow {
            field: "slot write",
        })?;
    slot_bytes[..8].copy_from_slice(&key_hash.to_le_bytes());
    slot_bytes[8..].copy_from_slice(&doc_ord_plus1.to_le_bytes());
    Ok(())
}
