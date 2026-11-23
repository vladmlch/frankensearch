use std::ops::Range;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};

use arc_swap::ArcSwap;
use frankensearch_core::DocId;
use frankensearch_index::mapped_file::ReadOnlyMappedFile;

use crate::error::QuillError;
use crate::grimoire::{
    TermDictionary, TermDictionaryError, TermMetadata, TermSectionLengths,
    ValidatedTermDictionaryMetadata,
};
use crate::quiver::{
    IdHashLookupPlan, IdHashSection, IdMapPresence, IdMapSection, StoredMetaCodecError,
    StoredMetaLookup, StoredMetaSection, ValidatedStoredMetaLookupPlan,
    ValidatedTermPruningMetadata,
};
use crate::schema::SchemaDescriptor;
use crate::segment::{EncodedSegment, SectionEntry, SectionKind, SegmentHeader, SegmentReader};

use super::errors::{KeeperError, ManifestCodecError};
use super::manifest::{ManifestSegment, TombstoneSet};
use super::manifest_load::validate_segment_witnesses;
use super::manifest_validation::{
    TombstoneContainers, array_binary_search, array_value, bitmap_contains, invalid,
    split_tombstone_docid, tombstone_chunk_count, usize_to_u64,
};

#[derive(Clone)]
pub(in crate::keeper) struct TombstoneIndex {
    chunks: Vec<TombstoneChunkIndex>,
}

#[derive(Clone, Copy)]
pub(in crate::keeper) struct TombstoneChunkIndex {
    pub(in crate::keeper) chunk_id: u16,
    pub(in crate::keeper) kind: u8,
    pub(in crate::keeper) payload_offset: usize,
    pub(in crate::keeper) payload_length: usize,
}

impl TombstoneIndex {
    pub(in crate::keeper) fn build(tombstones: &TombstoneSet) -> Result<Self, ManifestCodecError> {
        let chunk_count =
            usize::try_from(tombstone_chunk_count(tombstones.as_bytes())?).map_err(|_| {
                ManifestCodecError::ResourceLimit {
                    resource: "host tombstone chunk count",
                    actual: u64::from(u32::MAX),
                    limit: usize_to_u64(usize::MAX),
                }
            })?;
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(chunk_count)
            .map_err(|error| invalid(format!("tombstone index allocation failed: {error}")))?;
        let mut containers = TombstoneContainers::new(tombstones.as_bytes())?;
        while let Some(container) = containers.next_container()? {
            chunks.push(TombstoneChunkIndex {
                chunk_id: container.chunk_id,
                kind: container.kind,
                payload_offset: container.payload_offset,
                payload_length: container.payload.len(),
            });
        }
        Ok(Self { chunks })
    }

    pub(in crate::keeper) fn contains(&self, encoded: &[u8], global_docid: u32) -> bool {
        let (chunk_id, low) = split_tombstone_docid(global_docid);
        let Ok(index) = self
            .chunks
            .binary_search_by_key(&chunk_id, |chunk| chunk.chunk_id)
        else {
            return false;
        };
        let chunk = self.chunks[index];
        let Some(payload_end) = chunk.payload_offset.checked_add(chunk.payload_length) else {
            return false;
        };
        let Some(payload) = encoded.get(chunk.payload_offset..payload_end) else {
            return false;
        };
        match chunk.kind {
            0 => array_binary_search(payload, low),
            1 => bitmap_contains(payload, low),
            _ => false,
        }
    }
}

pub(in crate::keeper) enum RecoveredSegmentBacking {
    Mapped(SegmentReader<ReadOnlyMappedFile>),
    Owned(SegmentReader<EncodedSegment>),
}

impl RecoveredSegmentBacking {
    pub(in crate::keeper) fn header(&self) -> SegmentHeader {
        match self {
            Self::Mapped(reader) => reader.header(),
            Self::Owned(reader) => reader.header(),
        }
    }

    pub(in crate::keeper) fn section(
        &self,
        kind: SectionKind,
    ) -> Result<Option<&[u8]>, QuillError> {
        match self {
            Self::Mapped(reader) => reader.section(kind),
            Self::Owned(reader) => reader.section(kind),
        }
    }

    pub(in crate::keeper) fn section_entries(&self) -> &[SectionEntry] {
        match self {
            Self::Mapped(reader) => reader.section_entries(),
            Self::Owned(reader) => reader.section_entries(),
        }
    }

    pub(in crate::keeper) fn verify(&self) -> Result<(), QuillError> {
        match self {
            Self::Mapped(reader) => reader.verify(),
            Self::Owned(reader) => reader.verify(),
        }
    }

    pub(in crate::keeper) fn source_bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(reader) => reader.source_bytes(),
            Self::Owned(reader) => reader.source_bytes(),
        }
    }

    /// Trailer-declared xxh3-64 over the file prefix.
    ///
    /// Callers must authenticate this value against the backing bytes before
    /// using it as a content-identity witness for a manifest binding.
    pub(in crate::keeper) fn file_xxh3(&self) -> u64 {
        match self {
            Self::Mapped(reader) => reader.file_xxh3(),
            Self::Owned(reader) => reader.file_xxh3(),
        }
    }

    /// Whether this backing is a memory-mapped durable file.
    ///
    /// Mapped bytes are mutable out from under the process by external file
    /// writes, and a warm reader's section-checksum gates do not re-verify on
    /// later access, so no in-place rebind path may treat a mapped backing's
    /// re-validation as fresh. Mapped successor generations must go through a
    /// durable reopen.
    const fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped(_))
    }
}

fn required_identity_section<'a>(
    path: &Path,
    reader: &'a RecoveredSegmentBacking,
    kind: SectionKind,
) -> Result<&'a [u8], KeeperError> {
    reader
        .section(kind)
        .map_err(|source| KeeperError::SegmentOpen {
            path: path.to_path_buf(),
            source,
        })?
        .ok_or_else(|| KeeperError::MissingIdentitySection {
            path: path.to_path_buf(),
            kind,
        })
}

fn validate_identity_section_range(
    path: &Path,
    reader: &RecoveredSegmentBacking,
    kind: SectionKind,
    validated_bytes: &[u8],
) -> Result<Range<usize>, KeeperError> {
    let entry = reader
        .section_entries()
        .iter()
        .find(|entry| entry.kind == kind)
        .ok_or_else(|| KeeperError::MissingIdentitySection {
            path: path.to_path_buf(),
            kind,
        })?;
    let start =
        usize::try_from(entry.offset).map_err(|_| KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: format!("{kind:?} offset does not fit host usize"),
        })?;
    let len = usize::try_from(entry.len).map_err(|_| KeeperError::SegmentMetadataMismatch {
        path: path.to_path_buf(),
        detail: format!("{kind:?} length does not fit host usize"),
    })?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: format!("{kind:?} byte range overflowed"),
        })?;
    let range = start..end;
    let ranged = reader.source_bytes().get(range.clone()).ok_or_else(|| {
        KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: format!("{kind:?} byte range exceeds the immutable backing"),
        }
    })?;
    if !std::ptr::eq(ranged, validated_bytes) {
        return Err(KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: format!("{kind:?} validated bytes disagree with its section-table range"),
        });
    }
    Ok(range)
}

fn required_section_length(
    path: &Path,
    reader: &RecoveredSegmentBacking,
    kind: SectionKind,
) -> Result<u64, KeeperError> {
    reader
        .section_entries()
        .iter()
        .find(|entry| entry.kind == kind)
        .map(|entry| entry.len)
        .ok_or_else(|| KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: format!("segment is missing required {kind:?} section"),
        })
}

fn validate_term_dictionary_metadata(
    path: &Path,
    reader: &RecoveredSegmentBacking,
    schema: SchemaDescriptor,
) -> Result<ValidatedTermDictionaryMetadata, KeeperError> {
    let dictionary = reader
        .section(SectionKind::TERMDICT)
        .map_err(|source| KeeperError::SegmentOpen {
            path: path.to_path_buf(),
            source,
        })?
        .ok_or_else(|| KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: "segment is missing required TERMDICT section".to_owned(),
        })?;
    let sections = TermSectionLengths {
        postings: required_section_length(path, reader, SectionKind::POSTINGS)?,
        positions: reader
            .section_entries()
            .iter()
            .find(|entry| entry.kind == SectionKind::POSITIONS)
            .map(|entry| entry.len),
        blockmax: required_section_length(path, reader, SectionKind::BLOCKMAX)?,
    };
    TermDictionary::validate_metadata(dictionary, schema, sections)
        .map_err(|source| term_dictionary_admission_error(path, source))
}

pub(in crate::keeper) fn term_dictionary_admission_error(
    path: &Path,
    source: TermDictionaryError,
) -> KeeperError {
    match source {
        source @ (TermDictionaryError::InvalidSchema { .. }
        | TermDictionaryError::ByteBudgetExceeded { .. }
        | TermDictionaryError::BlockBudgetExceeded { .. }
        | TermDictionaryError::TermBudgetExceeded { .. }
        | TermDictionaryError::RestartBudgetExceeded { .. }
        | TermDictionaryError::NonAscendingInput { .. }
        | TermDictionaryError::InvalidRange
        | TermDictionaryError::MaterializationLimitExceeded { .. }
        | TermDictionaryError::GlobExpansionLimitExceeded { .. }
        | TermDictionaryError::Allocation { .. }
        | TermDictionaryError::SizeOverflow { .. }) => {
            KeeperError::SegmentTermDictionaryUnavailable {
                path: path.to_path_buf(),
                source,
            }
        }
        source @ (TermDictionaryError::ImplausibleBlockCount { .. }
        | TermDictionaryError::Truncated { .. }
        | TermDictionaryError::NonCanonicalVint { .. }
        | TermDictionaryError::VintOverflow { .. }
        | TermDictionaryError::ValueOutOfRange { .. }
        | TermDictionaryError::TermTooLong { .. }
        | TermDictionaryError::UnknownField { .. }
        | TermDictionaryError::NonTermField { .. }
        | TermDictionaryError::NonAscendingKey { .. }
        | TermDictionaryError::ZeroDocFrequency { .. }
        | TermDictionaryError::EmptyReference { .. }
        | TermDictionaryError::ReferenceOverflow { .. }
        | TermDictionaryError::ReferenceOutOfBounds { .. }
        | TermDictionaryError::NonContiguousReference { .. }
        | TermDictionaryError::SectionLengthMismatch { .. }
        | TermDictionaryError::PositionsPresenceMismatch { .. }
        | TermDictionaryError::PositionsSectionMismatch { .. }
        | TermDictionaryError::NonAscendingIndexKey { .. }
        | TermDictionaryError::InvalidBlockOffset { .. }
        | TermDictionaryError::EmptyBlock { .. }
        | TermDictionaryError::ImplausibleEntryCount { .. }
        | TermDictionaryError::OversizedBlock { .. }
        | TermDictionaryError::PrematureBlockSplit { .. }
        | TermDictionaryError::InvalidPrefix { .. }
        | TermDictionaryError::NonCanonicalPrefix { .. }
        | TermDictionaryError::IndexKeyMismatch { .. }
        | TermDictionaryError::TrailingBlockBytes { .. }
        | TermDictionaryError::TrailingEmptyBytes { .. }) => KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: source.to_string(),
        },
    }
}

fn first_tombstone_hole(
    tombstones: &TombstoneSet,
    id_map: IdMapSection<'_>,
) -> Result<Option<u32>, ManifestCodecError> {
    let mut containers = TombstoneContainers::new(tombstones.as_bytes())?;
    while let Some(container) = containers.next_container()? {
        match container.kind {
            0 => {
                for index in 0..container.payload.len() / 2 {
                    let global_docid = (u32::from(container.chunk_id) << 16)
                        | u32::from(array_value(container.payload, index));
                    if !id_map.contains(u64::from(global_docid)) {
                        return Ok(Some(global_docid));
                    }
                }
            }
            1 => {
                for (byte_index, byte) in container.payload.iter().copied().enumerate() {
                    let mut bits = byte;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        let low = u16::try_from(byte_index * 8 + bit)
                            .map_err(|_| invalid("bitmap tombstone low-bit overflow"))?;
                        let global_docid = (u32::from(container.chunk_id) << 16) | u32::from(low);
                        if !id_map.contains(u64::from(global_docid)) {
                            return Ok(Some(global_docid));
                        }
                        bits &= bits - 1;
                    }
                }
            }
            _ => return Err(invalid("unknown validated tombstone container kind")),
        }
    }
    Ok(None)
}

/// One live external-identifier resolution routed to its immutable segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedDocumentId {
    /// Segment containing the unique live physical row.
    pub segment_id: u64,
    /// Current segment publication order witness.
    pub seal_seq: u64,
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// Unseeded xxh3-64 witness over the canonical indexed document.
    pub content_hash: u64,
}

#[derive(Clone)]
pub(in crate::keeper) struct CachedRankPruningTerm {
    term_ord: u32,
    term_metadata: TermMetadata,
    pruning: Arc<ValidatedTermPruningMetadata>,
}

pub(in crate::keeper) const MAX_RANK_PRUNING_CACHE_TERMS: usize = 128;
const MAX_RANK_PRUNING_CACHE_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

/// Sparse lock-free cache for immutable per-term rank-pruning metadata.
///
/// Searches overwhelmingly revisit a small vocabulary relative to the full
/// dictionary, so a copy-on-write sorted sidecar avoids allocating one cache
/// cell per durable term. Concurrent first touches may duplicate validation,
/// but compare-and-swap publishes exactly one paired result and readers never
/// block one another.
pub(in crate::keeper) struct RankPruningCache {
    pub(in crate::keeper) terms: ArcSwap<Vec<CachedRankPruningTerm>>,
}

impl RankPruningCache {
    pub(in crate::keeper) fn new() -> Self {
        Self {
            terms: ArcSwap::from_pointee(Vec::new()),
        }
    }

    pub(in crate::keeper) fn get(
        &self,
        term_ord: u32,
        term_metadata: TermMetadata,
    ) -> Result<Option<Arc<ValidatedTermPruningMetadata>>, &'static str> {
        let terms = self.terms.load();
        let Ok(index) = terms.binary_search_by_key(&term_ord, |term| term.term_ord) else {
            return Ok(None);
        };
        let cached = &terms[index];
        if cached.term_metadata != term_metadata {
            return Err("cached rank-pruning term metadata disagrees with TERMDICT");
        }
        Ok(Some(Arc::clone(&cached.pruning)))
    }

    pub(in crate::keeper) fn insert(
        &self,
        term_ord: u32,
        term_metadata: TermMetadata,
        pruning: Arc<ValidatedTermPruningMetadata>,
    ) -> Result<Arc<ValidatedTermPruningMetadata>, &'static str> {
        loop {
            let current = self.terms.load_full();
            match current.binary_search_by_key(&term_ord, |term| term.term_ord) {
                Ok(index) => {
                    let cached = &current[index];
                    if cached.term_metadata != term_metadata {
                        return Err("concurrent rank-pruning cache entry disagrees with TERMDICT");
                    }
                    return Ok(Arc::clone(&cached.pruning));
                }
                Err(insertion) => {
                    let retained_bytes = current.iter().fold(0_usize, |bytes, term| {
                        bytes.saturating_add(term.pruning.heap_bytes())
                    });
                    if current.len() >= MAX_RANK_PRUNING_CACHE_TERMS
                        || pruning.heap_bytes()
                            > MAX_RANK_PRUNING_CACHE_PAYLOAD_BYTES.saturating_sub(retained_bytes)
                    {
                        return Ok(pruning);
                    }
                    let mut next = Vec::new();
                    next.try_reserve_exact(current.len().saturating_add(1))
                        .map_err(|_| "could not allocate rank-pruning cache sidecar")?;
                    next.extend(current.iter().cloned());
                    next.insert(
                        insertion,
                        CachedRankPruningTerm {
                            term_ord,
                            term_metadata,
                            pruning: Arc::clone(&pruning),
                        },
                    );
                    let next = Arc::new(next);
                    let previous = self.terms.compare_and_swap(&current, next);
                    if Arc::ptr_eq(&current, &previous) {
                        return Ok(pruning);
                    }
                }
            }
        }
    }

    #[cfg(test)]
    pub(in crate::keeper) fn len(&self) -> usize {
        self.terms.load().len()
    }
}

/// One immutable mapped or owned segment admitted by a recovered snapshot.
///
/// Structural framing, MANIFEST witnesses, and the IDMAP-bound IDHASH identity
/// pair are checked during open or owned publication. All unrelated payload
/// hashes remain lazy and are checked on first access.
#[cfg(test)]
pub(in crate::keeper) struct TermDictionaryCacheCounters {
    full_validations: AtomicU64,
    borrowed_views: AtomicU64,
    metadata_reuses: AtomicU64,
}

#[cfg(test)]
impl TermDictionaryCacheCounters {
    pub(in crate::keeper) const fn new() -> Self {
        Self {
            full_validations: AtomicU64::new(0),
            borrowed_views: AtomicU64::new(0),
            metadata_reuses: AtomicU64::new(0),
        }
    }
}

/// Successful-only lazy cache for one immutable segment's validated
/// STOREDMETA layout. The owned ranges remain bound to their exact source
/// slice; the cache never retains a borrowed view into the segment bytes.
pub(in crate::keeper) struct StoredMetaLookupCache {
    expected_field_ords: Box<[u16]>,
    plan: OnceLock<ValidatedStoredMetaLookupPlan>,
    #[cfg(test)]
    lookup_calls: AtomicU64,
    #[cfg(test)]
    successful_plan_builds: AtomicU64,
}

impl StoredMetaLookupCache {
    pub(in crate::keeper) fn new(schema: SchemaDescriptor) -> Self {
        let expected_field_ords = schema
            .fields
            .iter()
            .filter(|field| field.stored)
            .map(|field| field.id)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            expected_field_ords,
            plan: OnceLock::new(),
            #[cfg(test)]
            lookup_calls: AtomicU64::new(0),
            #[cfg(test)]
            successful_plan_builds: AtomicU64::new(0),
        }
    }

    pub(in crate::keeper) fn lookup<'a>(
        &'a self,
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
    ) -> Result<StoredMetaLookup<'a>, StoredMetaCodecError> {
        #[cfg(test)]
        self.lookup_calls.fetch_add(1, AtomicOrdering::Relaxed);
        if let Some(plan) = self.plan.get() {
            return StoredMetaSection::from_validated_metadata(bytes, plan);
        }

        // Validate speculatively before publishing. Concurrent first readers
        // may duplicate this bounded CPU work, but none can block an
        // asupersync worker on a cancellation-blind mutex. Failed validation
        // is never retained; exactly one successful plan wins publication.
        let plan = StoredMetaSection::validate_metadata(
            bytes,
            docid_lo,
            docid_hi,
            &self.expected_field_ords,
        )?;
        if self.plan.set(plan).is_ok() {
            #[cfg(test)]
            self.successful_plan_builds
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        let plan = self
            .plan
            .get()
            .expect("STOREDMETA plan must be initialized after successful validation");
        StoredMetaSection::from_validated_metadata(bytes, plan)
    }

    #[cfg(test)]
    pub(in crate::keeper) fn counts(&self) -> (u64, u64) {
        (
            self.lookup_calls.load(AtomicOrdering::Relaxed),
            self.successful_plan_builds.load(AtomicOrdering::Relaxed),
        )
    }
}

/// A full-prefix FSLX witness authenticated against one MANIFEST binding.
///
/// Mapped and arbitrary byte-backed readers mint this receipt only after a
/// fresh trailer-witness recomputation over the actual prefix. A freshly
/// encoded owned segment may instead carry the witness computed while its
/// canonical immutable bytes were assembled. Both paths bind the witness to
/// the MANIFEST through [`super::manifest_load::validate_segment_witnesses`]. It is safe to carry
/// the receipt across an owned tombstone-only rebind only with the exact same
/// immutable backing allocation.
#[derive(Clone)]
pub(in crate::keeper) struct AuthenticatedFileWitness {
    pub(in crate::keeper) file_xxh3: u64,
    #[cfg(test)]
    pub(in crate::keeper) full_prefix_hash_count: Arc<AtomicU64>,
}

impl AuthenticatedFileWitness {
    #[cfg(test)]
    pub(in crate::keeper) fn mint_after_full_prefix_validation(
        file_xxh3: u64,
        full_prefix_hash_count: Arc<AtomicU64>,
    ) -> Self {
        Self {
            file_xxh3,
            full_prefix_hash_count,
        }
    }

    #[cfg(not(test))]
    pub(in crate::keeper) fn mint_after_full_prefix_validation(file_xxh3: u64) -> Self {
        Self { file_xxh3 }
    }

    #[cfg(test)]
    pub(in crate::keeper) fn mint_from_fresh_owned_encoding(file_xxh3: u64) -> Self {
        Self {
            file_xxh3,
            full_prefix_hash_count: Arc::new(AtomicU64::new(0)),
        }
    }

    #[cfg(not(test))]
    pub(in crate::keeper) fn mint_from_fresh_owned_encoding(file_xxh3: u64) -> Self {
        Self { file_xxh3 }
    }

    pub(in crate::keeper) const fn file_xxh3(&self) -> u64 {
        self.file_xxh3
    }

    #[cfg(test)]
    pub(in crate::keeper) fn full_prefix_hash_count(&self) -> u64 {
        self.full_prefix_hash_count.load(AtomicOrdering::Relaxed)
    }
}

#[derive(Clone)]
pub struct RecoveredSegment {
    pub(in crate::keeper) path: PathBuf,
    pub(in crate::keeper) manifest: ManifestSegment,
    pub(in crate::keeper) reader: Arc<RecoveredSegmentBacking>,
    pub(in crate::keeper) tombstone_index: TombstoneIndex,
    pub(in crate::keeper) id_lookup: IdHashLookupPlan,
    pub(in crate::keeper) identity_sections: IdentitySectionRanges,
    pub(in crate::keeper) live_doc_count: u32,
    pub(in crate::keeper) rank_pruning_cache: Arc<RankPruningCache>,
    pub(in crate::keeper) term_dictionary_metadata: Arc<ValidatedTermDictionaryMetadata>,
    pub(in crate::keeper) stored_meta_lookup_cache: Arc<StoredMetaLookupCache>,
    /// Whole-file xxh3 witness authenticated against the exact immutable
    /// backing and manifest binding. Owned tombstone-only rebinds carry this
    /// receipt unchanged after their O(1) successor admission.
    pub(in crate::keeper) authenticated_file_witness: AuthenticatedFileWitness,
    #[cfg(test)]
    pub(in crate::keeper) term_dictionary_cache_counters: Arc<TermDictionaryCacheCounters>,
    /// Test-only call/attempt counter; range and tombstone misses are included.
    #[cfg(test)]
    pub(in crate::keeper) document_id_materialization_calls: Arc<AtomicU64>,
}

/// Exact IDMAP/IDHASH byte ranges validated against one immutable backing.
///
/// Identity lookup is a high-frequency admission path. Retaining these ranges
/// avoids re-entering the generic checked-section dispatcher for every probe;
/// the pair was checksum-validated and cross-validated before this value was
/// constructed, and the backing remains owned or read-only mapped for the
/// lifetime of the binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::keeper) struct IdentitySectionRanges {
    pub(in crate::keeper) id_map: Range<usize>,
    pub(in crate::keeper) id_hash: Range<usize>,
}

impl RecoveredSegment {
    pub(in crate::keeper) fn bind(
        path: PathBuf,
        manifest: ManifestSegment,
        reader: SegmentReader<ReadOnlyMappedFile>,
        schema: SchemaDescriptor,
        authenticated_file_witness: AuthenticatedFileWitness,
    ) -> Result<Self, KeeperError> {
        Self::bind_backing(
            path,
            manifest,
            RecoveredSegmentBacking::Mapped(reader),
            schema,
            authenticated_file_witness,
        )
    }

    pub(in crate::keeper) fn bind_owned(
        path: PathBuf,
        manifest: ManifestSegment,
        encoded: EncodedSegment,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        // `EncodedSegment` is the private canonical assembly product: its
        // prefix witness was computed while the exact immutable byte image
        // was built and is retained alongside the shared backing. Reuse that
        // receipt here instead of hashing the same newly assembled prefix a
        // second time. Mapped, borrowed, and arbitrary owned byte readers do
        // not enter this seam and continue through full-prefix verification.
        let encoded_file_xxh3 = encoded.file_xxh3();
        let reader = SegmentReader::from_encoded(encoded, schema).map_err(|source| {
            KeeperError::SegmentOpen {
                path: path.clone(),
                source,
            }
        })?;
        if reader.file_xxh3() != encoded_file_xxh3 {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "fresh owned encoding witness {encoded_file_xxh3:#018x} disagrees with parsed trailer {:#018x}",
                    reader.file_xxh3()
                ),
            });
        }
        validate_segment_witnesses(&path, &manifest, &reader, || Ok(encoded_file_xxh3))?;
        let authenticated_file_witness =
            AuthenticatedFileWitness::mint_from_fresh_owned_encoding(encoded_file_xxh3);
        Self::bind_backing(
            path,
            manifest,
            RecoveredSegmentBacking::Owned(reader),
            schema,
            authenticated_file_witness,
        )
    }

    fn bind_backing(
        path: PathBuf,
        manifest: ManifestSegment,
        reader: RecoveredSegmentBacking,
        schema: SchemaDescriptor,
        authenticated_file_witness: AuthenticatedFileWitness,
    ) -> Result<Self, KeeperError> {
        Self::bind_shared(
            path,
            manifest,
            Arc::new(reader),
            Arc::new(RankPruningCache::new()),
            schema,
            authenticated_file_witness,
            None,
        )
    }

    /// Bind one manifest generation over a shared immutable backing.
    ///
    /// `reuse_from` names the predecessor binding of the exact same backing
    /// when the caller is performing a tombstone-only manifest rebind. This
    /// path is OWNED-ONLY: a mapped predecessor is rejected up front with a
    /// typed reopen-required transition error, because mapped bytes can
    /// change on disk behind warm checksum gates and neither reuse nor an
    /// in-place "fresh" validation would be honest there. For owned
    /// backings, the already-validated TERMDICT metadata is reused only
    /// after every content-identity witness holds: the backing must be the
    /// same `Arc` allocation, the trailer-verified whole-file xxh3 recorded
    /// at validation time must equal this reader's, the schema must match,
    /// and the live TERMDICT slice must still be the exact address/length
    /// the metadata was minted for. Any witness mismatch rejects this rebind;
    /// a replacement backing must enter through a fresh authenticated binding.
    fn bind_shared(
        path: PathBuf,
        manifest: ManifestSegment,
        reader: Arc<RecoveredSegmentBacking>,
        rank_pruning_cache: Arc<RankPruningCache>,
        schema: SchemaDescriptor,
        authenticated_file_witness: AuthenticatedFileWitness,
        reuse_from: Option<&Self>,
    ) -> Result<Self, KeeperError> {
        // TRUST BOUNDARY: in-place rebind (any binding that names a
        // predecessor) is owned-only. A mapped backing's bytes can change on
        // disk after open while the reader's stored trailer hash and warm
        // section-checksum gates keep vouching for the ORIGINAL bytes, so
        // neither the reuse path nor its fresh-validation fallback would be
        // honest for a mapped rebind. Mapped successor generations require a
        // durable reopen, which re-verifies everything from actual bytes.
        if reuse_from.is_some() && reader.is_mapped() {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "mapped segment {} cannot rebind in place; a durable reopen \
                     is required (validated TERMDICT metadata reuse is owned-only)",
                    path.display()
                ),
            });
        }
        if authenticated_file_witness.file_xxh3() != reader.file_xxh3() {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "authenticated file_xxh3 {:#018x} disagrees with retained backing {:#018x}",
                    authenticated_file_witness.file_xxh3(),
                    reader.file_xxh3()
                ),
            });
        }
        let id_map_bytes = required_identity_section(&path, &reader, SectionKind::IDMAP)?;
        let id_map_range =
            validate_identity_section_range(&path, &reader, SectionKind::IDMAP, id_map_bytes)?;
        let id_map = IdMapSection::parse(id_map_bytes, manifest.docid_lo, manifest.docid_hi)
            .map_err(|source| KeeperError::IdMapCorrupted {
                path: path.clone(),
                source,
            })?;
        if u64::try_from(id_map.present_count()).unwrap_or(u64::MAX)
            != u64::from(manifest.doc_count)
        {
            return Err(KeeperError::AtSealDocCountMismatch {
                path,
                manifest: manifest.doc_count,
                id_map: id_map.present_count(),
            });
        }
        let tombstone_hole =
            first_tombstone_hole(&manifest.tombstones, id_map).map_err(|source| {
                KeeperError::ManifestCorrupted {
                    path: path.clone(),
                    source,
                }
            })?;
        if let Some(global_docid) = tombstone_hole {
            return Err(KeeperError::TombstoneReferencesHole {
                path,
                segment_id: manifest.segment_id,
                global_docid,
            });
        }
        let id_hash_bytes = required_identity_section(&path, &reader, SectionKind::IDHASH)?;
        let id_hash_range =
            validate_identity_section_range(&path, &reader, SectionKind::IDHASH, id_hash_bytes)?;
        let id_hash = IdHashSection::parse(id_hash_bytes, id_map).map_err(|source| {
            KeeperError::IdHashCorrupted {
                path: path.clone(),
                source,
            }
        })?;
        let id_lookup = id_hash.lookup_plan();
        let tombstone_index = TombstoneIndex::build(&manifest.tombstones).map_err(|source| {
            KeeperError::ManifestCorrupted {
                path: path.clone(),
                source,
            }
        })?;
        let live_doc_count = manifest.live_doc_count();

        // A tombstone-only owned rebind may retain a STOREDMETA plan only
        // when its exact immutable backing is structurally the same. This
        // check intentionally does not touch STOREDMETA: corrupted payloads
        // must remain lazy until a stored-value request reaches them.
        let stored_meta_lookup_cache = match reuse_from {
            Some(previous)
                if Arc::ptr_eq(&previous.reader, &reader)
                    && previous.authenticated_file_witness.file_xxh3()
                        == authenticated_file_witness.file_xxh3()
                    && schema == previous.term_dictionary_metadata.schema() =>
            {
                Arc::clone(&previous.stored_meta_lookup_cache)
            }
            _ => Arc::new(StoredMetaLookupCache::new(schema)),
        };

        // Reuse the predecessor's validated TERMDICT metadata only when every
        // content-identity witness proves the backing is byte-identical to
        // what that metadata was completely validated against.
        let reused_metadata = match reuse_from {
            Some(previous) => {
                let identical_backing = Arc::ptr_eq(&previous.reader, &reader)
                    && previous.authenticated_file_witness.file_xxh3()
                        == authenticated_file_witness.file_xxh3()
                    && schema == previous.term_dictionary_metadata.schema()
                    && validated_term_dictionary_is_bound(
                        &reader,
                        &previous.term_dictionary_metadata,
                    );
                if !identical_backing {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "owned segment {} lost its authenticated immutable backing during rebind",
                            path.display()
                        ),
                    });
                }
                Some(Arc::clone(&previous.term_dictionary_metadata))
            }
            None => None,
        };
        #[cfg(test)]
        let term_dictionary_cache_counters = match (&reused_metadata, reuse_from) {
            // A reused binding shares its predecessor's counters so the
            // observable contract stays "exactly one complete validation per
            // unique immutable backing", not merely per binding.
            (Some(_), Some(previous)) => {
                let counters = Arc::clone(&previous.term_dictionary_cache_counters);
                counters
                    .metadata_reuses
                    .fetch_add(1, AtomicOrdering::Relaxed);
                counters
            }
            _ => Arc::new(TermDictionaryCacheCounters::new()),
        };
        #[cfg(test)]
        let document_id_materialization_calls = reuse_from.map_or_else(
            || Arc::new(AtomicU64::new(0)),
            |previous| Arc::clone(&previous.document_id_materialization_calls),
        );
        let term_dictionary_metadata = match reused_metadata {
            Some(metadata) => metadata,
            None => {
                let metadata = Arc::new(validate_term_dictionary_metadata(&path, &reader, schema)?);
                #[cfg(test)]
                term_dictionary_cache_counters
                    .full_validations
                    .fetch_add(1, AtomicOrdering::Relaxed);
                metadata
            }
        };
        Ok(Self {
            path,
            manifest,
            reader,
            tombstone_index,
            id_lookup,
            identity_sections: IdentitySectionRanges {
                id_map: id_map_range,
                id_hash: id_hash_range,
            },
            live_doc_count,
            rank_pruning_cache,
            term_dictionary_metadata,
            stored_meta_lookup_cache,
            authenticated_file_witness,
            #[cfg(test)]
            term_dictionary_cache_counters,
            #[cfg(test)]
            document_id_materialization_calls,
        })
    }

    /// Rebind this immutable backing under a successor manifest generation.
    ///
    /// Tombstones live in the MANIFEST, not in the FSLX image, so an owned
    /// tombstone-only rebind retains its authenticated whole-file receipt
    /// rather than rehashing the immutable prefix. Rebind is owned-only: a
    /// mapped backing is rejected there with a typed reopen-required
    /// transition error and must go through a durable reopen instead.
    pub(in crate::keeper) fn rebind(&self, manifest: ManifestSegment) -> Result<Self, KeeperError> {
        self.rebind_with_backing(manifest, Arc::clone(&self.reader))
    }

    pub(in crate::keeper) fn rebind_with_backing(
        &self,
        manifest: ManifestSegment,
        reader: Arc<RecoveredSegmentBacking>,
    ) -> Result<Self, KeeperError> {
        self.validate_owned_rebind_admission(&manifest, &reader)?;
        let schema = self.term_dictionary_metadata.schema();
        Self::bind_shared(
            self.path.clone(),
            manifest,
            reader,
            Arc::clone(&self.rank_pruning_cache),
            schema,
            self.authenticated_file_witness.clone(),
            Some(self),
        )
    }

    /// Constant-time successor admission for an immutable owned backing.
    ///
    /// The in-memory publication path has already run
    /// [`super::manifest_validation::validate_manifest_successor`], which establishes the successor's
    /// complete generation-level transition. This local check makes the
    /// receipt dependency explicit before reusing it: the exact owned Arc,
    /// its authenticated trailer value, and all retained per-segment immutable
    /// metadata must still match the predecessor.
    fn validate_owned_rebind_admission(
        &self,
        successor: &ManifestSegment,
        reader: &Arc<RecoveredSegmentBacking>,
    ) -> Result<(), KeeperError> {
        if self.reader.is_mapped() || reader.is_mapped() {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "mapped segment {} cannot rebind in place; a durable reopen \
                     is required (validated TERMDICT metadata reuse is owned-only)",
                    self.path.display()
                ),
            });
        }
        if !Arc::ptr_eq(&self.reader, reader) {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "owned segment {} rebind requires the predecessor's exact authenticated \
                     immutable backing",
                    self.path.display()
                ),
            });
        }
        if self.manifest.file_xxh3 != self.authenticated_file_witness.file_xxh3()
            || self.authenticated_file_witness.file_xxh3() != self.reader.file_xxh3()
        {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "owned segment {} retained backing no longer agrees with its authenticated \
                     file_xxh3",
                    self.path.display()
                ),
            });
        }
        if self.manifest.segment_id != successor.segment_id
            || self.manifest.seal_seq != successor.seal_seq
            || self.manifest.file_len != successor.file_len
            || self.manifest.file_xxh3 != successor.file_xxh3
            || self.manifest.docid_lo != successor.docid_lo
            || self.manifest.docid_hi != successor.docid_hi
            || self.manifest.doc_count != successor.doc_count
        {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "immutable metadata changed for retained segment {:#018x}",
                    self.manifest.segment_id
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn term_dictionary(
        &self,
        schema: SchemaDescriptor,
    ) -> Result<TermDictionary<'_>, KeeperError> {
        if schema != self.term_dictionary_metadata.schema() {
            return Err(term_dictionary_admission_error(
                &self.path,
                TermDictionaryError::InvalidSchema {
                    detail: "query schema disagrees with validated TERMDICT metadata".to_owned(),
                },
            ));
        }
        let bytes = self
            .reader
            .section(SectionKind::TERMDICT)
            .map_err(|source| KeeperError::SegmentOpen {
                path: self.path.clone(),
                source,
            })?
            .ok_or_else(|| KeeperError::SegmentMetadataMismatch {
                path: self.path.clone(),
                detail: "validated TERMDICT section disappeared".to_owned(),
            })?;
        let dictionary =
            bind_validated_term_dictionary(&self.path, bytes, &self.term_dictionary_metadata)?;
        #[cfg(test)]
        self.term_dictionary_cache_counters
            .borrowed_views
            .fetch_add(1, AtomicOrdering::Relaxed);
        Ok(dictionary)
    }

    #[cfg(test)]
    pub(crate) fn term_dictionary_cache_counts(&self) -> (u64, u64) {
        (
            self.term_dictionary_cache_counters
                .full_validations
                .load(AtomicOrdering::Relaxed),
            self.term_dictionary_cache_counters
                .borrowed_views
                .load(AtomicOrdering::Relaxed),
        )
    }

    /// Test-only count of identity-verified TERMDICT metadata reuses across
    /// tombstone-only rebinds of this segment's immutable backing.
    #[cfg(test)]
    pub(crate) fn term_dictionary_metadata_reuse_count(&self) -> u64 {
        self.term_dictionary_cache_counters
            .metadata_reuses
            .load(AtomicOrdering::Relaxed)
    }

    /// Test-only count of full-prefix hashes used to authenticate this immutable binding.
    #[cfg(test)]
    pub(crate) fn authenticated_file_witness_hash_count(&self) -> u64 {
        self.authenticated_file_witness.full_prefix_hash_count()
    }

    /// Estimated payload bytes retained by this binding's validated TERMDICT
    /// metadata.
    ///
    /// Payload estimate only: excludes `Arc` control-block overhead,
    /// allocation alignment, and allocator slack — not an exact RSS claim.
    /// Rebound generations of the same immutable backing share one metadata
    /// allocation, and each binding reports the FULL payload of that shared
    /// object: summing this value across concurrently held snapshot
    /// generations double-counts the shared allocation.
    #[must_use]
    pub fn term_dictionary_metadata_payload_bytes(&self) -> usize {
        self.term_dictionary_metadata.payload_bytes()
    }

    pub(crate) fn cached_rank_pruning_metadata(
        &self,
        term_ord: u32,
        term_metadata: TermMetadata,
    ) -> Result<Option<Arc<ValidatedTermPruningMetadata>>, &'static str> {
        self.rank_pruning_cache.get(term_ord, term_metadata)
    }

    pub(crate) fn cache_rank_pruning_metadata(
        &self,
        term_ord: u32,
        term_metadata: TermMetadata,
        pruning: Arc<ValidatedTermPruningMetadata>,
    ) -> Result<Arc<ValidatedTermPruningMetadata>, &'static str> {
        self.rank_pruning_cache
            .insert(term_ord, term_metadata, pruning)
    }

    #[cfg(test)]
    pub(crate) fn cached_rank_pruning_term_count(&self) -> usize {
        self.rank_pruning_cache.len()
    }

    /// Canonical published path, or a stable synthetic label for owned bytes.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Immutable MANIFEST metadata for this segment.
    #[must_use]
    pub const fn manifest(&self) -> &ManifestSegment {
        &self.manifest
    }

    /// At-seal physical row count used by BM25 statistics.
    #[must_use]
    pub const fn at_seal_doc_count(&self) -> u32 {
        self.manifest.doc_count
    }

    /// Current MANIFEST tombstone cardinality.
    #[must_use]
    pub fn tombstone_count(&self) -> u64 {
        self.manifest.tombstones.cardinality()
    }

    /// Current live row count after tombstones.
    #[must_use]
    pub const fn doc_count(&self) -> u32 {
        self.live_doc_count
    }

    /// Whether this segment's current MANIFEST state hides one document id.
    #[must_use]
    pub fn is_tombstoned(&self, global_docid: u32) -> bool {
        self.tombstone_index
            .contains(self.manifest.tombstones.as_bytes(), global_docid)
    }

    /// Materialize one live external identifier from the cached IDMAP layout.
    ///
    /// The lookup touches only this row's two offsets and identifier bytes. It
    /// returns `None` for an out-of-range id, an IDMAP hole, or a tombstoned row.
    #[must_use]
    pub fn materialize_document_id(&self, global_docid: u32) -> Option<DocId> {
        #[cfg(test)]
        self.document_id_materialization_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        if !(self.manifest.docid_lo..self.manifest.docid_hi).contains(&u64::from(global_docid))
            || self.is_tombstoned(global_docid)
        {
            return None;
        }
        let id_map = self.reader.section(SectionKind::IDMAP).ok().flatten()?;
        self.id_lookup
            .materialize_global_docid(id_map, global_docid)
    }

    #[cfg(test)]
    pub(crate) fn document_id_materialization_call_count(&self) -> u64 {
        self.document_id_materialization_calls
            .load(AtomicOrdering::Relaxed)
    }

    /// Borrow the validated STOREDMETA lookup for this segment's exact live
    /// section bytes. The plan initializes only after the caller's lazy
    /// section checksum access succeeds.
    pub(crate) fn stored_meta_lookup<'a>(
        &'a self,
        bytes: &'a [u8],
    ) -> Result<StoredMetaLookup<'a>, StoredMetaCodecError> {
        self.stored_meta_lookup_cache
            .lookup(bytes, self.manifest.docid_lo, self.manifest.docid_hi)
    }

    #[cfg(test)]
    pub(crate) fn stored_meta_lookup_cache_counts(&self) -> (u64, u64) {
        self.stored_meta_lookup_cache.counts()
    }

    fn contains_identity_row(&self, global_docid: u32) -> bool {
        match self.id_lookup.identity_row_presence(global_docid) {
            IdMapPresence::Absent => false,
            IdMapPresence::Present => true,
            IdMapPresence::RequiresIdMap => self
                .reader
                .section(SectionKind::IDMAP)
                .ok()
                .flatten()
                .is_some_and(|id_map| self.id_lookup.contains_global_docid(id_map, global_docid)),
        }
    }

    pub(in crate::keeper) fn lookup_document_id(
        &self,
        document_id: &str,
    ) -> Result<Option<(u32, u64)>, KeeperError> {
        let source = self.reader.source_bytes();
        let id_map = source
            .get(self.identity_sections.id_map.clone())
            .ok_or_else(|| KeeperError::SegmentMetadataMismatch {
                path: self.path.clone(),
                detail: "validated IDMAP range exceeds the immutable backing".to_owned(),
            })?;
        let id_hash = source
            .get(self.identity_sections.id_hash.clone())
            .ok_or_else(|| KeeperError::SegmentMetadataMismatch {
                path: self.path.clone(),
                detail: "validated IDHASH range exceeds the immutable backing".to_owned(),
            })?;
        self.id_lookup
            .lookup_with_content_hash(id_map, id_hash, document_id)
            .map(|(global_docid, content_hash)| {
                u32::try_from(global_docid)
                    .map(|global_docid| (global_docid, content_hash))
                    .map_err(|_| KeeperError::SegmentMetadataMismatch {
                        path: self.path.clone(),
                        detail: format!("IDHASH returned non-u32 global docid {global_docid}"),
                    })
            })
            .transpose()
    }

    /// Structurally validated FSLX header.
    #[must_use]
    pub fn header(&self) -> SegmentHeader {
        self.reader.header()
    }

    /// Verify and borrow one section payload on first access.
    ///
    /// # Errors
    ///
    /// Returns a typed segment-corruption error when the payload checksum does
    /// not match its section-table witness.
    pub fn section(&self, kind: SectionKind) -> Result<Option<&[u8]>, QuillError> {
        self.reader.section(kind)
    }

    /// Validated section table, including unknown optional extensions.
    #[must_use]
    pub fn section_entries(&self) -> &[SectionEntry] {
        self.reader.section_entries()
    }

    /// Borrow the complete immutable FSLX image for byte-level verification.
    #[must_use]
    pub fn source_bytes(&self) -> &[u8] {
        self.reader.source_bytes()
    }

    /// Eagerly recompute every segment checksum for doctor-style verification.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error for any checksum mismatch.
    pub fn verify(&self) -> Result<(), QuillError> {
        self.reader.verify()
    }
}

pub(in crate::keeper) fn bind_validated_term_dictionary<'a>(
    path: &Path,
    bytes: &'a [u8],
    metadata: &'a ValidatedTermDictionaryMetadata,
) -> Result<TermDictionary<'a>, KeeperError> {
    TermDictionary::from_validated_metadata(bytes, metadata)
        .map_err(|source| term_dictionary_admission_error(path, source))
}

/// Whether `metadata` is still bound to `reader`'s live TERMDICT slice.
///
/// Resolves the TERMDICT section (its payload checksum gate is a warm
/// `OnceLock` on a shared backing, so no bytes are re-hashed) and checks the
/// exact address/length binding recorded at validation time. This is the
/// pointer fast path of the once-per-unique-backing reuse contract; it never
/// re-validates content.
fn validated_term_dictionary_is_bound(
    reader: &RecoveredSegmentBacking,
    metadata: &ValidatedTermDictionaryMetadata,
) -> bool {
    reader
        .section(SectionKind::TERMDICT)
        .ok()
        .flatten()
        .is_some_and(|bytes| TermDictionary::from_validated_metadata(bytes, metadata).is_ok())
}

impl crate::argus::LiveDocs for RecoveredSegment {
    fn is_live(&self, global_docid: u32) -> bool {
        (self.manifest.docid_lo..self.manifest.docid_hi).contains(&u64::from(global_docid))
            && self.contains_identity_row(global_docid)
            && !self.is_tombstoned(global_docid)
    }
}

/// One retained FSLX artifact omitted from the active MANIFEST after
/// durability recovery proved it unrepairable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantinedSegment {
    /// Stable identifier encoded in the canonical segment file name.
    pub segment_id: u64,
    /// Retained path ending in `.quarantine`.
    pub path: PathBuf,
    /// At-seal row count from the most recent retained MANIFEST witness.
    ///
    /// Older quarantines can outlive the two-slot MANIFEST history, in which
    /// case the estimate is intentionally unknown rather than guessed.
    pub estimated_missing_docs: Option<u64>,
}
