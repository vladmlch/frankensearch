use std::marker::PhantomData;

use thiserror::Error;

use crate::grimoire::MAX_TERM_BYTES;
use crate::quiver::NumericValue;
use crate::scribe::DOC_ORDS_PER_LEASE;

use super::DeltaSegment;
use super::arena::ChainState;

/// Typed delta construction and mutation failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DeltaError {
    /// A zero budget would request a seal after every document.
    #[error("delta budget must be greater than zero")]
    ZeroBudget,
    /// Q1 leases begin at fixed 65,536-document boundaries.
    #[error("delta lease base {lease_base} is not aligned to {DOC_ORDS_PER_LEASE} documents")]
    MisalignedLeaseBase { lease_base: u64 },
    /// The lease would exceed the durable u32 posting domain.
    #[error("delta lease [{lease_base}, {lease_end}) exceeds the global u32 docid domain")]
    LeaseOutOfRange { lease_base: u64, lease_end: u64 },
    /// A reset cannot return to an earlier allocator lease.
    #[error("delta lease reset regressed from {current_lease_base} to {next_lease_base}")]
    LeaseRegression {
        current_lease_base: u64,
        next_lease_base: u64,
    },
    /// A document is outside this delta generation's lease.
    #[error("global docid {global_docid} is outside delta lease [{lease_base}, {lease_end})")]
    DocumentOutsideLease {
        global_docid: u32,
        lease_base: u64,
        lease_end: u64,
    },
    /// Same-lease seals retain the first still-admissible global docid.
    #[error("global docid {global_docid} precedes delta continuation floor {floor}")]
    DocumentBeforeFloor { global_docid: u32, floor: u64 },
    /// Physical document rows must be strictly ascending.
    #[error("delta global docid {global_docid} does not follow prior docid {previous}")]
    DocumentOrder { previous: u32, global_docid: u32 },
    /// Durable IDMAP rows cannot carry an empty identifier.
    #[error("delta document id must not be empty")]
    EmptyDocumentId,
    /// Durable IDMAP offsets are u32.
    #[error("delta document id is {bytes} bytes and exceeds the u32 IDMAP domain")]
    DocumentIdTooLarge { bytes: usize },
    /// Every indexed string field needs one schema-ordered norm row.
    #[error("delta fieldnorm row has {actual} fields, expected {expected}")]
    FieldNormCount { expected: usize, actual: usize },
    /// Fieldnorm input order drifted from the schema.
    #[error("delta fieldnorm row {index} names field {actual}, expected {expected}")]
    FieldNormOrder {
        index: usize,
        expected: u16,
        actual: u16,
    },
    /// Quantized and raw field lengths disagree.
    #[error("delta field {field_ord} raw length {raw_length} maps to {expected}, got {actual}")]
    FieldNormMismatch {
        field_ord: u16,
        raw_length: u32,
        expected: u8,
        actual: u8,
    },
    /// Postings may target only Keyword or Text fields.
    #[error("delta posting names non-indexed-string field {field_ord}")]
    UnknownPostingField { field_ord: u16 },
    /// One durable term exceeded the FSLX ceiling.
    #[error("delta term for field {field_ord} is {bytes} bytes, limit {MAX_TERM_BYTES}")]
    TermTooLarge { field_ord: u16, bytes: usize },
    /// Inputs are grouped into one posting per term and document.
    #[error("delta document repeats term {term:?} in field {field_ord}")]
    DuplicateDocumentTerm { field_ord: u16, term: Vec<u8> },
    /// A completed posting always represents an occurrence.
    #[error("delta posting for field {field_ord} term {term:?} has zero frequency")]
    ZeroFrequency { field_ord: u16, term: Vec<u8> },
    /// Positioned Text fields require exact occurrence positions.
    #[error("delta positioned field {field_ord} term {term:?} omitted positions")]
    MissingPositions { field_ord: u16, term: Vec<u8> },
    /// Position-free fields cannot retain position payloads.
    #[error("delta position-free field {field_ord} term {term:?} supplied positions")]
    ForbiddenPositions { field_ord: u16, term: Vec<u8> },
    /// Frequency and position count must match.
    #[error(
        "delta field {field_ord} term {term:?} frequency {frequency} has {positions} positions"
    )]
    PositionCountMismatch {
        field_ord: u16,
        term: Vec<u8>,
        frequency: u32,
        positions: usize,
    },
    /// Positions are nondecreasing; equality is legal.
    #[error("delta field {field_ord} term {term:?} positions descend from {previous} to {current}")]
    DescendingPosition {
        field_ord: u16,
        term: Vec<u8>,
        previous: u32,
        current: u32,
    },
    /// A value named no field in the validated dense descriptor.
    #[error("delta value names unknown schema field {field_ord}")]
    UnknownValueField { field_ord: u16 },
    /// A document may supply a field through only one value input.
    #[error("delta document {global_docid} supplies field {field_ord} more than once")]
    DuplicateValueField { global_docid: u32, field_ord: u16 },
    /// A typed numeric value named a string or stored-only field.
    #[error("delta field {field_ord} ({field_name}) is not numeric")]
    NonNumericField {
        field_ord: u16,
        field_name: &'static str,
    },
    /// A typed numeric value named a field with no NUMERIC column.
    #[error("delta numeric field {field_ord} ({field_name}) is neither indexed nor fast")]
    NonIndexedNumericField {
        field_ord: u16,
        field_name: &'static str,
    },
    /// An in-memory numeric tag disagreed with the schema descriptor.
    #[error("delta numeric field {field_ord} ({field_name}) expects {expected}, got {actual}")]
    NumericTypeMismatch {
        field_ord: u16,
        field_name: &'static str,
        expected: &'static str,
        actual: &'static str,
    },
    /// A value named a field whose descriptor does not store original bytes.
    #[error("delta field {field_ord} ({field_name}) is not stored")]
    NonStoredField {
        field_ord: u16,
        field_name: &'static str,
    },
    /// Opaque bytes used as a numeric value must be exactly one scalar wide.
    #[error(
        "delta numeric field {field_ord} ({field_name}) has {actual} stored bytes, expected {expected}"
    )]
    InvalidNumericBytes {
        field_ord: u16,
        field_name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// One stored value cannot be represented by the durable u32 offsets.
    #[error("delta stored field {field_ord} value is {bytes} bytes and exceeds the u32 domain")]
    StoredValueTooLarge { field_ord: u16, bytes: usize },
    /// Appending a value would make one field blob exceed the durable u32 domain.
    #[error(
        "delta stored field {field_ord} blob cannot append {appended} bytes to its current {current} bytes"
    )]
    StoredBlobTooLarge {
        field_ord: u16,
        current: usize,
        appended: usize,
    },
    /// Per-term postings must remain strictly ascending.
    #[error(
        "delta field {field_ord} term {term:?} docid {global_docid} does not follow {previous}"
    )]
    PostingOrder {
        field_ord: u16,
        term: Vec<u8>,
        previous: u32,
        global_docid: u32,
    },
    /// A sidecar or typed arena could not reserve storage.
    #[error("unable to reserve {additional} additional entries for {resource}")]
    Allocation {
        resource: &'static str,
        additional: usize,
    },
    /// A logical locator exceeded its u32 domain.
    #[error("delta {resource} count {count} exceeds the u32 locator domain")]
    CountOverflow {
        resource: &'static str,
        count: usize,
    },
    /// The compile-time schema descriptor is invalid.
    #[error("invalid delta schema: {detail}")]
    InvalidSchema { detail: String },
}

/// Exact per-field document statistics retained by the delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaFieldNorm {
    /// Indexed Keyword/Text field ordinal.
    pub field_ord: u16,
    /// Exact admitted token count for BM25 aggregation.
    pub raw_length: u32,
    /// Tantivy-compatible quantized fieldnorm byte.
    pub fieldnorm_id: u8,
}

/// One schema-typed indexed-or-fast numeric value retained for sealing.
///
/// When the descriptor also sets `stored=true`, the delta derives the exact
/// canonical eight little-endian stored bytes automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaNumericValue {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Signed or unsigned value matching the field descriptor.
    pub value: NumericValue,
}

impl DeltaNumericValue {
    /// Construct one signed indexed value.
    #[must_use]
    pub const fn i64(field_ord: u16, value: i64) -> Self {
        Self {
            field_ord,
            value: NumericValue::I64(value),
        }
    }

    /// Construct one unsigned indexed value.
    #[must_use]
    pub const fn u64(field_ord: u16, value: u64) -> Self {
        Self {
            field_ord,
            value: NumericValue::U64(value),
        }
    }
}

/// One opaque schema-stored value retained for sealing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaStoredValue<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Borrowed opaque bytes copied into the delta on apply.
    pub bytes: &'a [u8],
}

impl<'a> DeltaStoredValue<'a> {
    /// Construct one borrowed stored value.
    #[must_use]
    pub const fn new(field_ord: u16, bytes: &'a [u8]) -> Self {
        Self { field_ord, bytes }
    }
}

/// One grouped term posting for a completed document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaTermPosting<'a> {
    /// Indexed Keyword/Text field ordinal.
    pub field_ord: u16,
    /// Exact term bytes without the field prefix.
    pub term: &'a [u8],
    /// Occurrence count in the document.
    ///
    /// Positionless fields retain this count only in their fieldnorm
    /// statistics; their Basic posting frequency is canonicalized to one.
    pub frequency: u32,
    /// Positions for a position-indexed Text field.
    pub positions: Option<&'a [u32]>,
}

/// Result of applying one complete document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaApply {
    /// Prior live delta docid for this external ID, if any.
    pub replaced_delta_docid: Option<u32>,
    /// Logical current-generation bytes after the apply.
    pub bytes_used: usize,
    /// Whether the accepted document crossed the seal threshold.
    pub seal_required: bool,
}

/// Stable memory and cardinality diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaMemoryStats {
    /// Logical bytes owned by the current generation.
    pub bytes_used: usize,
    /// Retained allocation, including reset-reusable capacity.
    pub bytes_reserved: usize,
    /// Number of distinct `(field, term)` keys.
    pub term_count: usize,
    /// Number of admitted rows, including tombstoned rows.
    pub physical_document_count: usize,
    /// Number of rows visible after applying delta tombstones.
    pub live_document_count: usize,
    /// Active unrolled posting blocks.
    pub posting_blocks: usize,
    /// Active unrolled position blocks.
    pub position_blocks: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::delta) struct PostingRecord {
    pub(in crate::delta) global_docid: u32,
    pub(in crate::delta) frequency: u32,
    pub(in crate::delta) position_start: u32,
    pub(in crate::delta) has_positions: bool,
}

/// Immutable posting yielded by a [`DeltaTerm`] view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaPosting<'a> {
    /// Absolute global document identifier.
    pub global_docid: u32,
    /// Occurrence count for this term in the document.
    pub frequency: u32,
    pub(in crate::delta) term_index: u32,
    pub(in crate::delta) position_start: u32,
    pub(in crate::delta) has_positions: bool,
    pub(in crate::delta) owner_id: u64,
    pub(in crate::delta) generation: u64,
    pub(in crate::delta) owner: PhantomData<&'a DeltaSegment>,
}

impl DeltaPosting<'_> {
    /// Whether this posting has an exact position slice.
    #[must_use]
    pub const fn has_positions(self) -> bool {
        self.has_positions
    }
}

/// Conservative whole-term impact envelope for a live Delta cursor.
///
/// Delta postings are append-only within one generation. The envelope keeps
/// the greatest frequency and smallest fieldnorm ever appended for the term.
/// A later upsert or delete may therefore make it looser, but never unsafe for
/// `MaxScore` or block-max pruning over the remaining live rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaBlockMax {
    pub(in crate::delta) max_frequency_code: u8,
    pub(in crate::delta) min_fieldnorm_id: u8,
}

impl DeltaBlockMax {
    /// Conservatively decoded greatest term frequency.
    #[must_use]
    pub const fn max_frequency(self) -> u32 {
        crate::contract::block_max_frequency_from_code(self.max_frequency_code)
    }

    /// Smallest quantized fieldnorm observed for this term.
    #[must_use]
    pub const fn min_fieldnorm_id(self) -> u8 {
        self.min_fieldnorm_id
    }

    /// Conservative BM25 tf-factor for one live-snapshot average length.
    #[must_use]
    pub fn tf_factor_upper_bound(self, live_avgdl: f32) -> Option<f32> {
        crate::contract::block_max_tf_factor(
            self.max_frequency_code,
            self.min_fieldnorm_id,
            live_avgdl,
        )
    }

    /// Apply a non-negative `idf * (1 + k1) * boost` term weight.
    #[must_use]
    pub fn score_upper_bound(self, live_avgdl: f32, nonnegative_weight: f32) -> Option<f32> {
        crate::contract::block_max_score(
            self.max_frequency_code,
            self.min_fieldnorm_id,
            live_avgdl,
            nonnegative_weight,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(in crate::delta) struct FieldLayout {
    pub(in crate::delta) field_ord: u16,
    pub(in crate::delta) positions: bool,
}

#[derive(Debug, Clone)]
pub(in crate::delta) struct FieldNormColumn {
    pub(in crate::delta) raw_lengths: Vec<u32>,
    pub(in crate::delta) fieldnorm_ids: Vec<u8>,
}

#[derive(Debug, Clone)]
pub(in crate::delta) struct NumericColumn {
    pub(in crate::delta) field_ord: u16,
    pub(in crate::delta) values: Vec<Option<NumericValue>>,
}

#[derive(Debug, Clone)]
pub(in crate::delta) struct StoredColumn {
    pub(in crate::delta) field_ord: u16,
    pub(in crate::delta) values: Vec<Option<Vec<u8>>>,
    pub(in crate::delta) payload_bytes: usize,
}

pub(in crate::delta) type ResolvedDocumentValues =
    (Vec<Option<NumericValue>>, Vec<Option<Vec<u8>>>);

#[derive(Debug, Clone, Copy)]
pub(in crate::delta) struct TermChain {
    pub(in crate::delta) postings: ChainState,
    pub(in crate::delta) positions: ChainState,
    pub(in crate::delta) last_docid: Option<u32>,
    pub(in crate::delta) max_frequency_code: u8,
    pub(in crate::delta) min_fieldnorm_id: u8,
    /// Postings in this chain that are not tombstoned.
    ///
    /// Counting survivors by walking the chain made every metadata read — a
    /// document frequency, a range or glob expansion, a snippet term choice —
    /// a full scan of the term, unbounded by anything the query admitted.
    /// The count is maintained where liveness actually changes instead: one
    /// increment per appended posting, one decrement per newly tombstoned
    /// document, and nothing at publication, because freezing clones the
    /// chain and a generation bump clears it.
    pub(in crate::delta) live_len: u32,
}

impl Default for TermChain {
    fn default() -> Self {
        Self {
            postings: ChainState::default(),
            positions: ChainState::default(),
            last_docid: None,
            max_frequency_code: 0,
            min_fieldnorm_id: u8::MAX,
            live_len: 0,
        }
    }
}
