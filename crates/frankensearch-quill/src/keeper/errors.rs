use std::io;
use std::path::PathBuf;

use asupersync::sync::LockError;
use frankensearch_core::SearchError;
use thiserror::Error;

use crate::error::QuillError;
use crate::grimoire::TermDictionaryError;
use crate::quiver::{IdHashCodecError, IdMapCodecError};
use crate::segment::SectionKind;

use super::manifest::Manifest;

/// Typed failures from MANIFEST byte validation.
#[derive(Debug, Error)]
pub enum ManifestCodecError {
    /// An in-memory manifest violates a writer invariant.
    #[error("invalid manifest: {detail}")]
    Invalid {
        /// Violated invariant.
        detail: String,
    },
    /// The byte stream ended before a field was complete.
    #[error("truncated manifest at byte {offset}: need {needed} bytes, only {remaining} remain")]
    Truncated {
        /// Byte offset where the read began.
        offset: usize,
        /// Bytes required by the field.
        needed: usize,
        /// Bytes still available.
        remaining: usize,
    },
    /// The magic did not identify a Quill MANIFEST.
    #[error("invalid manifest magic")]
    BadMagic,
    /// The reader does not implement the encoded version.
    #[error("unsupported manifest format version {found}")]
    UnsupportedVersion {
        /// Encoded version.
        found: u32,
    },
    /// The trailing CRC did not authenticate the preceding bytes.
    #[error("manifest CRC mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    ChecksumMismatch {
        /// CRC stored in the trailer.
        stored: u32,
        /// CRC recomputed by the reader.
        computed: u32,
    },
    /// A count or file length exceeds an eager-reader resource budget.
    #[error("manifest {resource} limit exceeded: {actual} > {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Observed value.
        actual: u64,
        /// Admitted maximum.
        limit: u64,
    },
    /// Structurally parseable bytes violate a canonical or Q1 invariant.
    #[error("non-canonical manifest: {detail}")]
    NonCanonical {
        /// Violated invariant.
        detail: String,
    },
    /// Valid fields were followed by bytes not owned by the format.
    #[error("manifest has {remaining} trailing bytes before its CRC")]
    TrailingBytes {
        /// Unconsumed byte count.
        remaining: usize,
    },
}

/// Policy for folding immutable-segment tombstones into positional holes.
///
/// A segment is eligible only when its tombstone density is strictly greater
/// than `tombstone_density`. Equality is deliberately a no-op, which prevents
/// repeated work at the configured boundary. Compaction always preserves the
/// surviving global document ids and never performs the reserved renumbering
/// deep-compaction operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionPolicy {
    /// Per-segment tombstone density threshold in `(0, 1]`.
    pub tombstone_density: f64,
}

impl CompactionPolicy {
    /// Construct a caller-selected density policy.
    #[must_use]
    pub const fn new(tombstone_density: f64) -> Self {
        Self { tombstone_density }
    }
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self::new(crate::config::DEFAULT_COMPACTION_TOMBSTONE_DENSITY)
    }
}

/// Observable work and byte accounting from one atomic compaction pass.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompactionReport {
    /// MANIFEST generation inspected by the pass.
    pub generation_before: u64,
    /// Published generation, equal to `generation_before` for a no-op.
    pub generation_after: u64,
    /// Immutable segments examined by the density policy.
    pub examined_segments: usize,
    /// Eligible segments rewritten or removed.
    pub compacted_segments: usize,
    /// Fully deleted segments removed without emitting an empty FSLX file.
    pub removed_segments: usize,
    /// Physical rows converted into positional holes.
    pub dropped_documents: u64,
    /// Sum of immutable source file lengths for eligible segments.
    pub input_bytes: u64,
    /// Sum of replacement FSLX file lengths.
    pub output_bytes: u64,
}

impl CompactionReport {
    /// Whether this pass published a successor generation.
    #[must_use]
    pub const fn changed(self) -> bool {
        self.compacted_segments != 0
    }
}

/// Typed failures from tombstone-folding segment compaction.
#[derive(Debug, Error)]
pub enum CompactionError {
    /// The threshold must match the configuration contract.
    #[error("compaction tombstone density {density:?} is outside (0, 1]")]
    InvalidDensity {
        /// Rejected threshold.
        density: f64,
    },
    /// A source carries an extension whose rewrite semantics are unknown.
    #[error("compaction source {segment_id:#018x} carries unsupported section kind {section_kind}")]
    UnknownSection {
        /// Source segment id.
        segment_id: u64,
        /// Raw durable section kind.
        section_kind: u16,
    },
    /// Checked rewrite arithmetic overflowed.
    #[error("compaction arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Failed counter or offset.
        field: &'static str,
    },
    /// A fallible allocation failed without panicking.
    #[error("unable to reserve {count} entries for compaction {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested element count.
        count: usize,
    },
    /// One section codec rejected source or rebuilt bytes.
    #[error("compaction {section:?} codec failed: {detail}")]
    SectionCodec {
        /// Affected FSLX section.
        section: SectionKind,
        /// Typed codec's stable diagnosis.
        detail: String,
    },
    /// Existing concat-source validation failed before the rewrite.
    #[error("compaction source validation failed: {detail}")]
    SourceValidation {
        /// Stable source diagnosis.
        detail: String,
    },
    /// A collision-free immutable output identity could not be derived.
    #[error("compaction could not derive a collision-free segment id for {source_id:#018x}")]
    OutputIdExhausted {
        /// Source segment being replaced.
        source_id: u64,
    },
    /// The successor MANIFEST failed canonical validation.
    #[error("compaction successor manifest is invalid: {detail}")]
    InvalidManifest {
        /// Manifest diagnosis.
        detail: String,
    },
    /// The caller cancelled before the visibility boundary.
    #[error("compaction cancelled before MANIFEST publication")]
    Cancelled,
    /// Final FSLX framing or temp-file staging failed.
    #[error("compaction segment operation failed: {source}")]
    Segment {
        /// Segment-layer diagnosis.
        #[source]
        source: QuillError,
    },
}

/// Typed failures from Q1-preserving whole-segment concat merge.
#[derive(Debug, Error)]
pub enum ConcatMergeError {
    /// A merge must replace at least two immutable segments.
    #[error("concat merge requires at least two source segments, got {count}")]
    TooFewSources {
        /// Supplied source count.
        count: usize,
    },
    /// One caller-supplied source id is absent from the current snapshot.
    #[error("concat merge source {segment_id:#018x} at position {position} is not live")]
    SourceNotFound {
        /// Position in caller order.
        position: usize,
        /// Missing immutable segment id.
        segment_id: u64,
    },
    /// Caller order did not name one uninterrupted manifest slice.
    #[error(
        "concat merge source {actual:#018x} at position {position} is not manifest-consecutive; expected {expected:#018x}"
    )]
    NonConsecutiveSources {
        /// Position in caller order.
        position: usize,
        /// Required segment at this manifest position.
        expected: u64,
        /// Caller-supplied segment id.
        actual: u64,
    },
    /// A live id appears after the chosen manifest suffix has ended.
    #[error(
        "concat merge source {actual:#018x} at position {position} wraps past the manifest suffix"
    )]
    SourceRunPastManifest {
        /// Position in caller order.
        position: usize,
        /// Live source id that wrapped to an earlier manifest position.
        actual: u64,
    },
    /// The proposed immutable output id already belongs to a live segment.
    #[error("concat merge output segment id {segment_id:#018x} collides with a live segment")]
    OutputSegmentCollision {
        /// Colliding segment id.
        segment_id: u64,
    },
    /// A source carries an extension whose merge semantics are unknown.
    #[error(
        "concat merge source {segment_id:#018x} carries unsupported section kind {section_kind}"
    )]
    UnknownSection {
        /// Source segment id.
        segment_id: u64,
        /// Raw durable section kind.
        section_kind: u16,
    },
    /// Checked merge arithmetic overflowed.
    #[error("concat merge arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Failed counter or offset.
        field: &'static str,
    },
    /// A fallible allocation failed without panicking.
    #[error("unable to reserve {count} entries for concat merge {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested element count.
        count: usize,
    },
    /// One section codec rejected source or rebuilt bytes.
    #[error("concat merge {section:?} codec failed: {detail}")]
    SectionCodec {
        /// Affected FSLX section.
        section: SectionKind,
        /// Typed codec's stable diagnosis.
        detail: String,
    },
    /// External-id state admitted more than one live physical row.
    #[error(
        "concat merge document id {document_id:?} has multiple live rows {first_global_docid}/{duplicate_global_docid}"
    )]
    MultipleLiveDocumentIds {
        /// Exact external identifier.
        document_id: String,
        /// First live global document id.
        first_global_docid: u32,
        /// Duplicate live global document id.
        duplicate_global_docid: u32,
    },
    /// The successor MANIFEST failed canonical validation.
    #[error("concat merge successor manifest is invalid: {detail}")]
    InvalidManifest {
        /// Manifest diagnosis.
        detail: String,
    },
    /// The caller cancelled before the visibility boundary.
    #[error("concat merge cancelled before MANIFEST publication")]
    Cancelled,
    /// Final FSLX framing or temp-file staging failed.
    #[error("concat merge segment operation failed: {source}")]
    Segment {
        /// Segment-layer diagnosis.
        #[source]
        source: QuillError,
    },
}

/// What a publication is allowed to do to the number of live documents.
///
/// A Keeper proposal is derived state: the writer stages a tombstone for every
/// identity a replacement batch supersedes and then appends the replacement
/// rows. When the rows never land — the ingest failed after the tombstones
/// were staged — the derived proposal is a deletion nobody asked for, and at
/// the MANIFEST level it is indistinguishable from a deliberate one: it
/// validates, every segment authenticates, and readers simply answer from what
/// is left (gh#45). The intent says which of the two the caller means, so the
/// Keeper can refuse the one it did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PublishIntent {
    /// An additive or one-for-one replacing publication.
    ///
    /// The successor must serve at least as many live documents as the
    /// generation it replaces. A proposal that would shrink the live set is
    /// refused with [`KeeperError::LiveDocumentFloor`] before any durable or
    /// in-memory state changes. `commit`, upsert, Delta seals, bulk-load
    /// completion, tier merges and compaction publish with this intent.
    PreserveLiveDocuments,
    /// The caller deliberately removed live documents.
    ///
    /// `delete_documents` and `delete_all` publish with this intent, and so do
    /// the raw [`super::writer::KeeperWriter::publish`] and
    /// [`super::snapshot::KeeperSnapshot::publish_owned_segments`] entry points, whose proposal
    /// the caller constructed tombstone by tombstone. The live-document floor
    /// is waived for that one publication; every other transition rule still
    /// applies.
    ExplicitDeletion,
}

impl PublishIntent {
    pub(in crate::keeper) const fn live_document_floor(self) -> LiveDocumentFloor {
        match self {
            Self::PreserveLiveDocuments => LiveDocumentFloor::Enforced,
            Self::ExplicitDeletion => LiveDocumentFloor::Waived,
        }
    }
}

/// Whether one successor validation enforces the live-document floor.
///
/// Shape-only validations — reopening an already published MANIFEST pair,
/// re-checking a proposal against the on-disk predecessor inside the durable
/// choreography, or staging tombstones into a proposal that has not been
/// offered for publication yet — waive it: the floor is decided exactly once,
/// at the publication entry point that knows the caller's [`PublishIntent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveDocumentFloor {
    Enforced,
    Waived,
}

/// Saturating count of live rows a MANIFEST serves across all its segments.
pub(in crate::keeper) fn manifest_live_doc_count(manifest: &Manifest) -> u64 {
    manifest.segments.iter().fold(0_u64, |total, segment| {
        total.saturating_add(u64::from(segment.live_doc_count()))
    })
}

/// Keeper I/O, recovery, locking, and publication failures.
#[derive(Debug, Error)]
pub enum KeeperError {
    /// An immutable lexical snapshot could not produce or re-open its exact receipt.
    #[error("Quill exact lexical snapshot receipt rejected: {detail}")]
    ExactSnapshotReceipt {
        /// Stable receipt or descriptor diagnosis.
        detail: String,
    },
    /// Neither MANIFEST slot exists.
    #[error("Quill index not found at {directory}")]
    IndexNotFound {
        /// Index directory that was inspected.
        directory: PathBuf,
    },
    /// One durable slot contains invalid bytes.
    #[error("corrupt manifest at {path}: {source}")]
    ManifestCorrupted {
        /// Invalid slot.
        path: PathBuf,
        /// Codec diagnosis.
        #[source]
        source: ManifestCodecError,
    },
    /// Proposed in-memory state failed writer-side validation.
    #[error("invalid proposed manifest: {source}")]
    InvalidManifest {
        /// Writer invariant diagnosis.
        #[source]
        source: ManifestCodecError,
    },
    /// A whole-segment concat merge failed before publication completed.
    #[error("keeper concat merge failed: {source}")]
    ConcatMerge {
        /// Merge planning, codec, or cancellation diagnosis.
        #[source]
        source: ConcatMergeError,
    },
    /// A tombstone-folding compaction failed before publication completed.
    #[error("keeper compaction failed: {source}")]
    Compaction {
        /// Policy, rewrite, codec, or cancellation diagnosis.
        #[source]
        source: CompactionError,
    },
    /// Neither existing slot is valid.
    #[error("no valid manifest in {directory}: current={current}; previous={previous}")]
    NoValidManifest {
        /// Index directory that was inspected.
        directory: PathBuf,
        /// Primary-slot diagnosis.
        current: String,
        /// Previous-slot diagnosis.
        previous: String,
    },
    /// Two valid slots do not form an admitted generation pair.
    #[error(
        "invalid manifest generation pair in {directory}: current={current}, previous={previous}"
    )]
    InvalidGenerationPair {
        /// Index directory containing the pair.
        directory: PathBuf,
        /// Current generation.
        current: u64,
        /// Previous generation.
        previous: u64,
    },
    /// Two individually valid slots disagree on durable snapshot identity.
    #[error(
        "invalid manifest pair in {directory}: current generation {current}, previous generation {previous}: {detail}"
    )]
    InvalidManifestPair {
        /// Index directory containing the pair.
        directory: PathBuf,
        /// Current generation.
        current: u64,
        /// Previous generation.
        previous: u64,
        /// Cross-slot invariant that failed.
        detail: String,
    },
    /// The proposed publication is not the next monotone generation.
    #[error("manifest generation conflict: expected {expected}, proposed {proposed}")]
    GenerationConflict {
        /// Required next generation.
        expected: u64,
        /// Proposed generation.
        proposed: u64,
    },
    /// Another process still owns the cross-process writer admission.
    #[error("Quill writer is busy at {path}; owner pid={owner_pid:?}")]
    WriterBusy {
        /// Canonical writer-lock path.
        path: PathBuf,
        /// Best-effort owner diagnosis from a complete lock record.
        owner_pid: Option<u32>,
    },
    /// The persistent writer-lock record is malformed and cannot be reclaimed.
    #[error("corrupt Quill writer lock at {path}: {detail}")]
    WriterLockCorrupted {
        /// Canonical writer-lock path.
        path: PathBuf,
        /// Failed wire, type, or liveness invariant.
        detail: String,
    },
    /// Cancellation was observed while acquiring or initializing a writer.
    #[error("Quill writer admission cancelled")]
    WriterAdmissionCancelled,
    /// An extant `O_EXCL` artifact already owns the proposed generation.
    #[error("Quill generation {generation} is already claimed at {path}")]
    GenerationClaimConflict {
        /// Existing canonical claim path.
        path: PathBuf,
        /// Claimed MANIFEST generation.
        generation: u64,
    },
    /// The current generation cannot be incremented.
    #[error("manifest generation space exhausted at {current}")]
    GenerationExhausted {
        /// Last representable generation.
        current: u64,
    },
    /// The retained writer view may lag a durable publication whose future was abandoned.
    #[error(
        "Keeper snapshot requires reconciliation: retained generation {retained_generation}, \
         abandoned proposed generation {proposed_generation}"
    )]
    PublicationReconciliationRequired {
        /// Generation in the retained, non-authoritative view.
        retained_generation: u64,
        /// Generation that the abandoned future may have made durable.
        proposed_generation: u64,
    },
    /// A proposed generation would roll back durable index identity or state.
    #[error("invalid manifest transition: {detail}")]
    InvalidTransition {
        /// Violated monotonicity invariant.
        detail: String,
    },
    /// A publication declared [`PublishIntent::PreserveLiveDocuments`] but its
    /// successor would serve fewer live documents than the generation it
    /// replaces.
    ///
    /// This is the shape of a replacement whose tombstones were staged but
    /// whose replacement rows never landed (gh#45). Nothing was written: the
    /// outgoing generation remains authoritative. A caller that really means
    /// to remove documents publishes with [`PublishIntent::ExplicitDeletion`].
    #[error(
        "live-document floor violated: generation {proposed_generation} would serve \
         {proposed_live_docs} live documents where generation {previous_generation} serves \
         {previous_live_docs}; a replacement whose rows did not land is refused, and a \
         deliberate removal must publish as one (delete_documents / delete_all, or \
         PublishIntent::ExplicitDeletion on the Keeper)"
    )]
    LiveDocumentFloor {
        /// Generation the proposal would replace.
        previous_generation: u64,
        /// Live documents the outgoing generation serves.
        previous_live_docs: u64,
        /// Generation the proposal would publish.
        proposed_generation: u64,
        /// Live documents the proposal would serve.
        proposed_live_docs: u64,
    },
    /// The caller's expected schema does not match the durable MANIFEST.
    #[error("Quill schema mismatch at {path}: expected {expected:#018x}, found {found:#018x}")]
    SchemaMismatch {
        /// MANIFEST path carrying the durable schema identity.
        path: PathBuf,
        /// Hash of the schema requested by the caller.
        expected: u64,
        /// Hash recorded in the selected MANIFEST.
        found: u64,
    },
    /// The supplied schema descriptor itself is invalid.
    #[error("invalid Quill schema descriptor: {source}")]
    InvalidSchema {
        /// Descriptor validation failure.
        #[source]
        source: QuillError,
    },
    /// A MANIFEST-referenced segment could not be opened structurally.
    #[error("cannot open Quill segment at {path}: {source}")]
    SegmentOpen {
        /// Canonical published segment path.
        path: PathBuf,
        /// Segment reader diagnosis.
        #[source]
        source: QuillError,
    },
    /// Canonical segment bytes could not be staged for an idempotent install.
    #[error("cannot stage Quill segment for publication: {source}")]
    SegmentInstall {
        /// Segment-layer write or validation diagnosis.
        #[source]
        source: QuillError,
    },
    /// A segment disagrees with its immutable MANIFEST witnesses.
    #[error("Quill segment metadata mismatch at {path}: {detail}")]
    SegmentMetadataMismatch {
        /// Canonical published segment path.
        path: PathBuf,
        /// Failed identity, range, length, or checksum-witness comparison.
        detail: String,
    },
    /// Eager TERMDICT admission stopped for a reason that does not prove the
    /// durable segment bytes are corrupt.
    #[error("cannot admit Quill TERMDICT metadata at {path}: {source}")]
    SegmentTermDictionaryUnavailable {
        /// Canonical published segment path.
        path: PathBuf,
        /// Schema, resource-budget, allocation, or host-size diagnosis.
        #[source]
        source: TermDictionaryError,
    },
    /// A queryable segment omitted one required identity section.
    #[error("Quill segment at {path} is missing required identity section {kind:?}")]
    MissingIdentitySection {
        /// Canonical published segment path.
        path: PathBuf,
        /// Missing IDMAP or IDHASH section.
        kind: SectionKind,
    },
    /// A segment's IDMAP payload failed canonical cross-section validation.
    #[error("corrupt Quill IDMAP at {path}: {source}")]
    IdMapCorrupted {
        /// Canonical published segment path.
        path: PathBuf,
        /// Typed IDMAP diagnosis.
        #[source]
        source: IdMapCodecError,
    },
    /// A segment's IDHASH payload failed canonical IDMAP-bound validation.
    #[error("corrupt Quill IDHASH at {path}: {source}")]
    IdHashCorrupted {
        /// Canonical published segment path.
        path: PathBuf,
        /// Typed IDHASH diagnosis.
        #[source]
        source: IdHashCodecError,
    },
    /// IDMAP presence disagreed with the segment and MANIFEST at-seal count.
    #[error("Quill segment document count mismatch at {path}: manifest={manifest}, IDMAP={id_map}")]
    AtSealDocCountMismatch {
        /// Canonical published segment path.
        path: PathBuf,
        /// Count authenticated by the MANIFEST and segment header.
        manifest: u32,
        /// Present rows validated from IDMAP.
        id_map: usize,
    },
    /// A durable tombstone named a positional IDMAP hole.
    #[error(
        "Quill segment {segment_id:#018x} at {path} tombstones missing IDMAP row {global_docid}"
    )]
    TombstoneReferencesHole {
        /// Canonical published segment path.
        path: PathBuf,
        /// Segment identity carrying the invalid tombstone.
        segment_id: u64,
        /// Tombstoned global document id with no physical row.
        global_docid: u32,
    },
    /// More than one current segment exposed the same live external identifier.
    #[error(
        "Quill upsert invariant violated in {directory}: document identifier resolves to live docids {first_global_docid} and {duplicate_global_docid}"
    )]
    MultipleLiveDocumentIds {
        /// Snapshot directory, or the in-memory sentinel.
        directory: PathBuf,
        /// First live global document id in descending seal order.
        first_global_docid: u32,
        /// Additional live global document id.
        duplicate_global_docid: u32,
    },
    /// A fallback MANIFEST and an extant generation claim disagree.
    #[error(
        "invalid interrupted-publish claim at {path}: recovered generation {recovered}, claimed generation {claimed}"
    )]
    InvalidRecoveryClaim {
        /// Claim path carrying the incompatible generation.
        path: PathBuf,
        /// Generation selected from `MANIFEST.prev`.
        recovered: u64,
        /// Generation named by the claim.
        claimed: u64,
    },
    /// A canonical generation claim is not the required zero-length file.
    #[error("invalid Quill generation claim at {path}: {detail}")]
    InvalidClaimArtifact {
        /// Canonical claim path.
        path: PathBuf,
        /// Failed no-follow type or length invariant.
        detail: String,
    },
    /// A claimed future generation must be resolved before garbage collection.
    #[error(
        "writer recovery required at {path}: current generation {current}, claimed generation {claimed}"
    )]
    ClaimedGenerationPending {
        /// Future claim blocking a garbage sweep.
        path: PathBuf,
        /// Selected durable generation.
        current: u64,
        /// Future claimed generation.
        claimed: u64,
    },
    /// A garbage-collection target was not one safe direct child name.
    #[error("unsafe Quill garbage path rejected: {path}")]
    UnsafeGarbagePath {
        /// Rejected relative or absolute target.
        path: PathBuf,
    },
    /// The supplied index-directory path stopped naming the opened directory.
    #[error("Quill index directory changed during garbage collection: {directory}")]
    GarbageDirectoryChanged {
        /// Directory whose identity was not stable for the sweep.
        directory: PathBuf,
    },
    /// Publishing over a corrupt primary requires the later writer-recovery lane.
    #[error("manifest recovery required before publish at {path}")]
    RecoveryRequired {
        /// Corrupt primary slot.
        path: PathBuf,
    },
    /// A stale temp file exists but is not byte-identical to this proposal.
    #[error("manifest temp conflicts with proposed bytes at {path}")]
    TempConflict {
        /// Temp file that cannot be safely reused or overwritten.
        path: PathBuf,
    },
    /// The cancel-aware in-process publish mutex was not acquired.
    #[error("manifest publish lock failed: {source}")]
    PublishLock {
        /// Asupersync lock failure.
        #[source]
        source: LockError,
    },
    /// A filesystem operation failed with path and operation context.
    #[error("keeper {operation} failed at {path}: {source}")]
    Io {
        /// Operation label.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Operating-system error.
        #[source]
        source: io::Error,
    },
}

impl From<ConcatMergeError> for KeeperError {
    fn from(source: ConcatMergeError) -> Self {
        Self::ConcatMerge { source }
    }
}

impl From<CompactionError> for KeeperError {
    fn from(source: CompactionError) -> Self {
        Self::Compaction { source }
    }
}

impl From<KeeperError> for SearchError {
    fn from(error: KeeperError) -> Self {
        match error {
            KeeperError::IndexNotFound { directory } => Self::IndexNotFound { path: directory },
            KeeperError::ManifestCorrupted { path, source } => Self::IndexCorrupted {
                path,
                detail: source.to_string(),
            },
            KeeperError::NoValidManifest {
                directory,
                current,
                previous,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!("no valid manifest: current={current}; previous={previous}"),
            },
            KeeperError::InvalidGenerationPair {
                directory,
                current,
                previous,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!(
                    "invalid manifest generation pair: current={current}, previous={previous}"
                ),
            },
            KeeperError::InvalidManifestPair {
                directory,
                current,
                previous,
                detail,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!(
                    "invalid manifest pair at generations {current}/{previous}: {detail}"
                ),
            },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::IndexCorrupted { detail, .. },
            } => Self::IndexCorrupted { path, detail },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::Io(source),
            } if source.kind() == io::ErrorKind::NotFound => Self::IndexCorrupted {
                path,
                detail: "MANIFEST-referenced segment is missing".to_owned(),
            },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::UnknownSchema { schema_id },
            } => Self::IndexCorrupted {
                path,
                detail: format!("MANIFEST-referenced segment has unknown schema {schema_id:#018x}"),
            },
            KeeperError::SegmentOpen { path, source } => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(KeeperError::SegmentOpen { path, source }),
            },
            KeeperError::SegmentMetadataMismatch { path, detail } => {
                Self::IndexCorrupted { path, detail }
            }
            KeeperError::MissingIdentitySection { path, kind } => Self::IndexCorrupted {
                path,
                detail: format!("MANIFEST-referenced segment is missing {kind:?}"),
            },
            KeeperError::IdMapCorrupted { path, source } => Self::IndexCorrupted {
                path,
                detail: source.to_string(),
            },
            KeeperError::IdHashCorrupted { path, source } => Self::IndexCorrupted {
                path,
                detail: source.to_string(),
            },
            KeeperError::AtSealDocCountMismatch {
                path,
                manifest,
                id_map,
            } => Self::IndexCorrupted {
                path,
                detail: format!(
                    "at-seal document count mismatch: manifest={manifest}, IDMAP={id_map}"
                ),
            },
            KeeperError::TombstoneReferencesHole {
                path,
                segment_id,
                global_docid,
            } => Self::IndexCorrupted {
                path,
                detail: format!(
                    "segment {segment_id:#018x} tombstones missing IDMAP row {global_docid}"
                ),
            },
            KeeperError::MultipleLiveDocumentIds {
                directory,
                first_global_docid,
                duplicate_global_docid,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!(
                    "document identifier has multiple live rows {first_global_docid}/{duplicate_global_docid}"
                ),
            },
            KeeperError::InvalidRecoveryClaim {
                path,
                recovered,
                claimed,
            } => Self::IndexCorrupted {
                path,
                detail: format!(
                    "recovered manifest generation {recovered} conflicts with claim {claimed}"
                ),
            },
            KeeperError::InvalidClaimArtifact { path, detail } => {
                Self::IndexCorrupted { path, detail }
            }
            KeeperError::WriterAdmissionCancelled => Self::Cancelled {
                phase: "keeper.writer_admission".to_owned(),
                reason: "writer admission cancelled".to_owned(),
            },
            KeeperError::ConcatMerge {
                source: ConcatMergeError::Cancelled,
            } => Self::Cancelled {
                phase: "keeper.concat_merge".to_owned(),
                reason: "concat merge cancelled before MANIFEST publication".to_owned(),
            },
            KeeperError::Compaction {
                source: CompactionError::Cancelled,
            } => Self::Cancelled {
                phase: "keeper.compact".to_owned(),
                reason: "compaction cancelled before MANIFEST publication".to_owned(),
            },
            KeeperError::PublishLock { source } => match source {
                LockError::Cancelled => Self::Cancelled {
                    phase: "keeper.publish".to_owned(),
                    reason: "manifest publish lock cancelled".to_owned(),
                },
                LockError::TimedOut(deadline) => Self::Cancelled {
                    phase: "keeper.publish".to_owned(),
                    reason: format!("manifest publish lock timed out at {deadline:?}"),
                },
                other => Self::SubsystemError {
                    subsystem: "quill",
                    source: Box::new(KeeperError::PublishLock { source: other }),
                },
            },
            other => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(other),
            },
        }
    }
}
