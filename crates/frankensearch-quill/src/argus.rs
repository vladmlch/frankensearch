//! Argus query execution.
//!
//! This module contains the deliberately exhaustive scorer used as the
//! correctness anchor for every later pruning path. It mirrors the pinned
//! Tantivy scorer-tree arithmetic, including buffered `Should` unions, while
//! retaining Quill's explicit `Option<u32>` cursor state so `u32::MAX` remains
//! a searchable global document id.

use std::fmt;

use thiserror::Error;

use crate::quiver::{NumericCodecError, PositionCodecError, PostingCodecError};

pub use crate::query::Occur;

mod boolean_scorers;
mod collector;
mod cursor;
mod phrase_scorer;
mod reference_scorer;
mod segment_cursors;
mod term_scorer;
mod union_pruning;
mod union_scorer;

#[cfg(test)]
mod tests;

pub use collector::{
    AllLiveDocs, CollectedTopDocs, DocSetCollector, LiveDocs, MaterializedHit, ScoredDoc,
    TopDocsCollector, materialize_doc_ids,
};
#[cfg(test)]
pub(crate) use cursor::CursorWork;
pub(crate) use cursor::ValidatedArrivalCursor;
pub use cursor::{CheckpointPostingCursor, PostingCursor, ValidatedPostMoveContract};
pub use phrase_scorer::{PhraseScorer, PhraseTerm};
pub use reference_scorer::{ReferenceScorer, ScorerClause};
pub use segment_cursors::{DeltaFieldNorms, DeltaPostingCursor, SealedPostingCursor};
pub use term_scorer::{Bm25FieldSnapshot, FieldNormReader, TermRecordOption, TermScorer};
pub(crate) use union_scorer::PruningTelemetry;
#[cfg(any(test, feature = "pruning-conformance"))]
pub(crate) use union_scorer::{ConformanceUnionRefill, ConformanceUnionRefillStrategy};

/// Postings per Delta logical block, the granularity a Delta walk charges.
///
/// Delta postings are not encoded in blocks, so this is the accounting unit
/// that keeps a Delta scan's fuel comparable with a sealed scan's.
pub(in crate::argus) const DELTA_LOGICAL_BLOCK_POSTINGS: u32 = 128;
pub(in crate::argus) const UNION_HORIZON: usize = 4_096;
pub(in crate::argus) const UNION_HORIZON_U64: u64 = 4_096;
pub(crate) const MAX_SCORE_MAX_CLAUSES: usize = 8;
pub(crate) const BMW_MIN_CLAUSES: usize = MAX_SCORE_MAX_CLAUSES + 1;
pub(in crate::argus) const BMW_MIN_TOTAL_COST: u64 = 16_384;
pub(in crate::argus) const MAX_GLOBAL_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;

/// Owner-bound decoder for one cursor's compressed position runs.
///
/// Sealed and delta cursors implement this against their own immutable
/// storage. Callers cannot resolve an ordinal against an unrelated segment.
pub trait PositionsReader {
    /// Decode one posting's absolute positions into an already-cleared buffer.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, corruption, or cursor-identity failure.
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError>;
}

/// Borrowed capability for the current posting's later POSITIONS lookup.
///
/// The immutable borrow prevents its cursor from moving before the handle is
/// resolved. This keeps sealed and future delta position identities local to
/// their owning storage view without a GAT or per-hit allocation.
#[derive(Clone, Copy)]
pub struct PositionsHandle<'a> {
    reader: &'a dyn PositionsReader,
    posting_ordinal: u32,
}

impl fmt::Debug for PositionsHandle<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PositionsHandle")
            .field("posting_ordinal", &self.posting_ordinal)
            .finish_non_exhaustive()
    }
}

impl<'a> PositionsHandle<'a> {
    /// Bind a zero-based posting ordinal to its exact owning reader.
    #[must_use]
    pub fn new(reader: &'a dyn PositionsReader, posting_ordinal: u32) -> Self {
        Self {
            reader,
            posting_ordinal,
        }
    }

    /// Return the zero-based posting ordinal.
    #[must_use]
    pub const fn posting_ordinal(self) -> u32 {
        self.posting_ordinal
    }

    /// Decode this posting into a reusable caller-owned buffer.
    ///
    /// The buffer is cleared before decoding and remains empty on failure.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, corruption, or cursor-identity failure.
    pub fn decode_into(self, output: &mut Vec<u32>) -> Result<(), ArgusError> {
        output.clear();
        if let Err(error) = self.reader.decode_positions(self.posting_ordinal, output) {
            output.clear();
            return Err(error);
        }
        Ok(())
    }
}

/// Typed failures from exhaustive query evaluation.
#[derive(Debug, Error)]
pub enum ArgusError {
    /// A sealed posting cursor violated its already-validated storage contract.
    #[error(transparent)]
    Posting(#[from] PostingCodecError),
    /// A sealed position cursor violated its validated paired-stream contract.
    #[error(transparent)]
    Position(#[from] PositionCodecError),
    /// A numeric range could not be compiled from its validated field column.
    #[error(transparent)]
    Numeric(#[from] NumericCodecError),
    /// Snapshot counters cannot describe a valid scored field.
    #[error("invalid BM25 snapshot for field {field_ord}: {reason}")]
    InvalidSnapshot {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Failed invariant.
        reason: &'static str,
    },
    /// Snapshot term frequency exceeded the snapshot's BM25 document count.
    #[error("term doc_freq {doc_freq} exceeds snapshot N {doc_count}")]
    InvalidDocFrequency {
        /// Snapshot-level term document frequency.
        doc_freq: u64,
        /// Snapshot-level BM25 document count.
        doc_count: u64,
    },
    /// A scorer bound statistics and field lengths from different fields.
    #[error("BM25 field mismatch: fieldnorm field {fieldnorm_field}, stats field {stats_field}")]
    FieldMismatch {
        /// Field ordinal supplied by the DOCLEN column.
        fieldnorm_field: u16,
        /// Field ordinal supplied by snapshot STATS.
        stats_field: u16,
    },
    /// A phrase term cursor was opened for a field other than its snapshot.
    #[error("phrase term field {term_field} does not match snapshot field {stats_field}")]
    PhraseTermFieldMismatch {
        /// Field ordinal supplied when the phrase cursor was opened.
        term_field: u16,
        /// Field ordinal supplied by snapshot STATS.
        stats_field: u16,
    },
    /// A programmatic query supplied a non-finite or overflowing field boost.
    #[error("field {field_ord} has invalid boost bits 0x{boost_bits:08x}")]
    InvalidBoost {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Exact rejected f32 representation.
        boost_bits: u32,
    },
    /// A posting had no corresponding DOCLEN byte in its segment.
    #[error("field {field_ord} has no fieldnorm for global docid {global_docid}")]
    MissingFieldnorm {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Global Quill document id.
        global_docid: u32,
    },
    /// A phrase leaf was opened against a frequency-only posting cursor.
    #[error("field {field_ord} has no positions for global docid {global_docid}")]
    MissingPositions {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Global Quill document id whose phrase positions were required.
        global_docid: u32,
    },
    /// A coarse query-execution checkpoint observed structured cancellation.
    #[error("query execution cancelled during {phase}")]
    QueryCancelled {
        /// Stable query phase.
        phase: &'static str,
    },
    /// Deterministic coarse work exceeded the configured query fuel budget.
    #[error(
        "query fuel exhausted after {consumed}/{budget} units \
         (segments={segments_touched}, dictionary_blocks={dictionary_blocks}, \
         posting_blocks={posting_blocks}, position_docs={position_docs})"
    )]
    QueryFuelExhausted {
        /// Configured work-unit ceiling.
        budget: u64,
        /// Successfully admitted work units before exhaustion.
        consumed: u64,
        /// Segment transitions admitted before exhaustion.
        segments_touched: u64,
        /// Prefix-compressed dictionary blocks admitted before exhaustion.
        dictionary_blocks: u64,
        /// Posting blocks admitted before exhaustion.
        posting_blocks: u64,
        /// Per-document phrase-position verification or snippet enrichment admitted before exhaustion.
        position_docs: u64,
    },
    /// A phrase plan cannot describe an exact positional query.
    #[error("invalid exact phrase: {reason}")]
    InvalidPhrase {
        /// Failed phrase-plan invariant.
        reason: &'static str,
    },
    /// A cursor exposed internally inconsistent accessors.
    #[error("posting cursor invariant failed: {0}")]
    CursorInvariant(&'static str),
    /// Score arithmetic escaped the finite BM25 domain.
    #[error("non-finite score for global docid {global_docid}")]
    NonFiniteScore {
        /// Global Quill document id being scored.
        global_docid: u32,
    },
    /// A bounded internal allocation could not be reserved.
    #[error("could not allocate {count} entries for {resource}")]
    Allocation {
        /// Bounded allocation purpose.
        resource: &'static str,
        /// Requested element count.
        count: usize,
    },
    /// A winning global document id could not be resolved externally.
    #[error("no external document id for global docid {global_docid}")]
    MissingExternalDocId {
        /// Global Quill document id.
        global_docid: u32,
    },
    /// Offset pagination overflowed the host collection domain.
    #[error("top-k collector window overflows usize: limit {limit}, offset {offset}")]
    CollectorWindowOverflow {
        /// Requested page size.
        limit: usize,
        /// Requested number of leading hits to skip.
        offset: usize,
    },
    /// An exact match count exceeded the durable collection domain.
    #[error("exact match count overflowed u64")]
    MatchCountOverflow,
    /// A movement-only operation received a tree that scores while advancing.
    #[error("movement-only scoring requires a recursively unscored scorer tree")]
    ScoredTreeForUnscoredCollector,
    /// A caller attempted to score a deliberately unscored query tree.
    #[error("an unscored Boolean query has no BM25 score")]
    ScoreUnavailable,
    /// A NUMERIC column cannot contain more values than physical documents.
    #[error(
        "NUMERIC field {field_ord} has {value_count} values for {segment_num_docs} physical documents"
    )]
    InvalidNumericCardinality {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Validated values present in the field column.
        value_count: usize,
        /// Physical at-seal documents supplied by the segment plan.
        segment_num_docs: u32,
    },
    /// A programmatic constant-score predicate supplied a non-finite boost.
    #[error("constant-score predicate has invalid boost bits 0x{boost_bits:08x}")]
    InvalidConstantBoost {
        /// Exact rejected f32 representation.
        boost_bits: u32,
    },
    /// A match-all leaf cannot describe the supplied global document domain.
    #[error(
        "invalid All scorer range [{docid_lo}, {docid_hi}) for {segment_num_docs} live documents: {reason}"
    )]
    InvalidAllRange {
        /// Inclusive global document id lower bound.
        docid_lo: u64,
        /// Exclusive global document id upper bound.
        docid_hi: u64,
        /// Live documents reported by the segment.
        segment_num_docs: u32,
        /// Failed range invariant.
        reason: &'static str,
    },
    /// A programmatic match-all query supplied a non-finite boost.
    #[error("All scorer has invalid boost bits 0x{boost_bits:08x}")]
    InvalidAllBoost {
        /// Exact rejected f32 representation.
        boost_bits: u32,
    },
}

/// One coarse, deterministic query-work boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryWorkKind {
    /// Entering one sealed or Delta segment.
    Segment,
    /// Entering one prefix-compressed term-dictionary block.
    DictionaryBlock,
    /// Decoding or entering one posting block.
    PostingBlock,
    /// Per-document phrase-position verification or snippet enrichment.
    PositionDocument,
}

/// Exact competitive-union phase surrounding one conformance-only query-work
/// admission.
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConformanceQueryWorkPhase {
    /// Shadow cursors are discovering competitive documents for this refill.
    Discovery,
    /// A real scorer cursor is seeking to a selected candidate or horizon.
    Seek,
    /// A scored candidate contribution is complete and its real cursor is
    /// being drained or advanced.
    Drain,
}

/// Per-query location installed by the competitive-union scorer around an
/// exact conformance-only work admission.
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConformanceQueryWorkContext {
    pub(crate) refill_ordinal: u64,
    pub(crate) window_start: u32,
    pub(crate) phase: ConformanceQueryWorkPhase,
    pub(crate) candidate_doc: Option<u32>,
    pub(crate) candidate_scored: bool,
}

#[cfg(feature = "pruning-conformance")]
impl ConformanceQueryWorkContext {
    pub(in crate::argus) const fn new(
        refill_ordinal: u64,
        window_start: u32,
        phase: ConformanceQueryWorkPhase,
        candidate_doc: Option<u32>,
        candidate_scored: bool,
    ) -> Self {
        Self {
            refill_ordinal,
            window_start,
            phase,
            candidate_doc,
            candidate_scored,
        }
    }
}

/// Cancel-aware deterministic query-work admission.
///
/// Implementations must be thread-safe because ordinary, comfortably
/// under-budget queries may still fan sealed segments across Rayon. Fuel
/// exhaustion itself is executed serially by the shipping facade so its
/// progress diagnostics are schedule-independent.
pub trait QueryWorkCheckpoint: Send + Sync {
    /// Admit `units` of one coarse work kind.
    ///
    /// # Errors
    ///
    /// Returns [`ArgusError::QueryCancelled`] or
    /// [`ArgusError::QueryFuelExhausted`] before the guarded work begins.
    fn admit(&self, kind: QueryWorkKind, units: u64) -> Result<(), ArgusError>;

    /// Install or clear the exact competitive-refill location surrounding
    /// subsequent admissions. Normal builds do not contain this hook.
    #[cfg(feature = "pruning-conformance")]
    #[doc(hidden)]
    fn set_conformance_query_work_context(&self, _context: Option<ConformanceQueryWorkContext>) {}
}

/// One f32 accumulation step inside a buffered union window (bd-rtnwu).
///
/// Records the exact addition the union performed: which child (by stable
/// construction ordinal and by current runtime index), on which document,
/// with which accumulator, contribution, and result bit patterns.
#[cfg(feature = "pruning-conformance")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnionAddTrace {
    /// 1-based refill ordinal within this union's lifetime.
    pub refill_ordinal: u64,
    /// First document of the window being filled.
    pub window_start: u32,
    /// Which fill path performed the addition.
    pub fill_path: UnionFillPath,
    /// Stable child identity assigned at union construction.
    pub child_construction_ordinal: u32,
    /// Child position in the evolving scorer vector at the time of the add.
    pub runtime_index: usize,
    /// Document receiving the contribution.
    pub doc: u32,
    /// Accumulator bits BEFORE the addition (0.0 when first contribution).
    pub accumulator_bits: u32,
    /// Contribution bits returned by the child scorer.
    pub contribution_bits: u32,
    /// Accumulator bits AFTER the addition.
    pub result_bits: u32,
}

/// Which buffered-union fill implementation performed a traced step.
#[cfg(feature = "pruning-conformance")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnionFillPath {
    /// Ordinary scorer-vector exhaustive fill.
    Exhaustive,
    /// Ordinary scorer-vector candidate-restricted fill.
    Candidate,
    /// Specialized doc-major pivot-prefix exhaustive fill.
    TantivyTopDocsTerm,
    /// Specialized doc-major candidate-restricted fill.
    TantivyTopDocsTermCandidate,
}

/// One traced buffered-union lifecycle event (bd-rtnwu).
#[cfg(feature = "pruning-conformance")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnionTraceEvent {
    /// Union constructed: route decision inputs and stable child identities.
    Constructed {
        /// Whether the specialized doc-major route was taken.
        route_specialized: bool,
        /// The `topdocs_root` flag the lowering passed in.
        topdocs_root: bool,
        /// Children in construction order, described by cursor kind.
        child_kinds: Vec<&'static str>,
        /// Whole-group ceiling bits captured at construction, when bounded.
        group_ceiling_bits: Option<u32>,
    },
    /// A window refill began.
    Refill {
        /// 1-based refill ordinal.
        ordinal: u64,
        /// First document of the window.
        window_start: u32,
        /// Rank-pruning cutoff bits when one applied to this window.
        cutoff_bits: Option<u32>,
        /// Which fill path serviced the refill.
        fill_path: UnionFillPath,
    },
    /// One f32 accumulation step.
    Add(UnionAddTrace),
    /// An exhausted child left the scorer vector.
    SwapRemove {
        /// Runtime index removed.
        runtime_index: usize,
        /// Construction ordinal of the removed child.
        child_construction_ordinal: u32,
    },
    /// The evolving scorer vector was stably re-sorted (specialized path).
    SortSnapshot {
        /// Construction ordinals in post-sort runtime order.
        order: Vec<u32>,
    },
    /// A specialized-path pivot prefix was summed.
    PivotPrefix {
        /// Pivot document.
        doc: u32,
        /// Number of equal-document children summed.
        len: usize,
    },
}

/// Invocation-bound consumer of buffered-union trace events (bd-rtnwu).
///
/// Installed BEFORE `BufferedUnionScorer` construction because construction
/// immediately refills. Shipping builds contain no producer of these events:
/// every emission site is `pruning-conformance`-gated, and an absent sink
/// costs one `Option` check.
#[cfg(feature = "pruning-conformance")]
pub trait ScorerTraceSink: Send + Sync {
    /// Record one union lifecycle event.
    fn record(&self, event: UnionTraceEvent);
}

#[cfg(feature = "pruning-conformance")]
std::thread_local! {
    /// Invocation-scoped trace sink (bd-rtnwu). Thread-local by design:
    /// installing a sink threads NO lowering signatures, and the caveat is
    /// stated rather than hidden — segment fan-out workers do not inherit
    /// it, so trace fixtures must execute serially (single sealed segment).
    static SCORER_TRACE_SINK: std::cell::RefCell<Option<std::sync::Arc<dyn ScorerTraceSink>>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII installer for the invocation-scoped scorer trace sink (bd-rtnwu).
///
/// The sink is active on THIS thread from construction until drop. Install
/// BEFORE running the query: `BufferedUnionScorer` construction refills
/// immediately, and the construction event itself is part of the evidence.
#[cfg(feature = "pruning-conformance")]
pub struct ScorerTraceSinkGuard(());

#[cfg(feature = "pruning-conformance")]
impl ScorerTraceSinkGuard {
    /// Install `sink` for the current thread and return the guard.
    #[must_use]
    pub fn install(sink: std::sync::Arc<dyn ScorerTraceSink>) -> Self {
        SCORER_TRACE_SINK.with(|cell| *cell.borrow_mut() = Some(sink));
        Self(())
    }
}

#[cfg(feature = "pruning-conformance")]
impl Drop for ScorerTraceSinkGuard {
    fn drop(&mut self) {
        SCORER_TRACE_SINK.with(|cell| *cell.borrow_mut() = None);
    }
}

#[cfg(feature = "pruning-conformance")]
pub(in crate::argus) fn active_scorer_trace_sink() -> Option<std::sync::Arc<dyn ScorerTraceSink>> {
    SCORER_TRACE_SINK.with(|cell| cell.borrow().clone())
}
