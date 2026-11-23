//! Shipping Quill index orchestration and process-local snapshot publication.
//!
//! This module deliberately joins the already-final Scribe, FSLX, Keeper,
//! parser, cursor, scorer, and collector boundaries. Immutable Delta epochs,
//! their composite statistics, scorer adapters, and seal transaction are
//! assembled here; later mixed-state beads wire them into the public writer loop.

mod checkpoint;
#[cfg(feature = "conformance-internals")]
#[allow(clippy::module_name_repetitions)]
mod conformance;
#[allow(clippy::module_name_repetitions)]
mod error_results;
mod ingest_doc;
mod parallel_ingest;
#[cfg(feature = "profile-internals")]
#[allow(clippy::module_name_repetitions)]
mod profile;
mod query_cache;
mod query_cursors;
mod query_lowering;
mod query_predicates;
#[allow(clippy::module_name_repetitions)]
mod quill_index;
mod reader;
mod reader_exec;
mod search_handles;
#[allow(clippy::module_name_repetitions)]
mod snapshot;
mod writer;
mod writer_flush;
mod writer_ingest;
mod writer_lifecycle;

pub use error_results::{
    QuillClauseKind, QuillDocumentWitness, QuillHit, QuillIndexError, QuillScoreExplanation,
    QuillSearchResult, QuillSnippetHit, QuillTermScore,
};
pub use ingest_doc::{SchemaDocument, indexable_document_content_hash};
pub use quill_index::QuillIndex;
pub use search_handles::{QUILL_LEXICAL_BACKEND, QuillSearchIndex, RootBoundQuillSearchIndex};
pub use snapshot::{QuillSearchSnapshot, SnapshotError, SnapshotPublisher};

#[cfg(feature = "profile-internals")]
#[doc(hidden)]
pub use profile::{
    QuillProfileCacheDisposition, QuillProfileExecutionMode, QuillProfileOutcome,
    QuillProfileReceipt, QuillProfileSession, QuillProfileWorkUnits, QuillProfiledSearchOutcome,
};

#[cfg(feature = "pruning-conformance")]
pub use conformance::ConformanceQueryInterruptionLocation;
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
pub use conformance::{
    ConformancePruningExecutionMode, ConformancePruningRefillReceipt, ConformancePruningStrategy,
    ConformancePruningTraceReceipt, ConformanceSegmentPruningReceipt,
};

#[cfg(feature = "conformance-internals")]
pub use conformance::{
    ConformanceCancellationController, ConformanceCancellationStage, ConformancePendingWriterState,
};
#[cfg(feature = "conformance-internals")]
pub use reader::PinnedSearchBundle;

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub use quill_index::{BenchmarkQuillDirectorySyncArm, BenchmarkQuillDirectorySyncReceipt};
#[cfg(feature = "bench-internals")]
pub use search_handles::PreparsedQuillIndex;

#[cfg(test)]
pub(in crate::index) use checkpoint::{
    LifecycleTrigger, PendingDeltaSeal, PreparedSnapshotDocFreqs, QueryCheckpoint,
    QueryCheckpointHandle, SEGMENT_COUNT_FANOUT_THRESHOLD, SEGMENT_FANOUT_THRESHOLD,
    SnapshotDocFreqReservationOverrides, delta_ordered_view_blocks, prepare_snapshot_doc_freqs,
    query_work_upper_bound, sealed_segment_fanout, snippet_tail_work_upper_bound,
};
#[cfg(all(test, feature = "pruning-conformance"))]
pub(in crate::index) use conformance::{
    ConformancePruningTraceGuard, ConformancePruningTraceSession,
};
#[cfg(test)]
pub(in crate::index) use error_results::{
    CONTENT_FIELD, ID_FIELD, MAX_GLOBAL_DOCID_EXCLUSIVE, METADATA_FIELD, ORD_FIELD, TITLE_FIELD,
    invalid_state, validate_docid_watermark,
};
#[cfg(test)]
pub(in crate::index) use ingest_doc::{
    IngestBatch, canonical_document_content_hash, canonical_metadata, manifest_segment,
    merge_field_stats,
};
#[cfg(test)]
pub(in crate::index) use parallel_ingest::{
    FANOUT_WAVE_SHARD_DOCUMENTS, INTERNAL_PARALLEL_INGEST_SHARDS, MIN_ARENA_CHUNK_BYTES,
    PARALLEL_INGEST_MIN_DOCS_PER_SHARD, PARALLEL_INGEST_PLANNER_VERSION, ParallelBatchObservation,
    ParallelDocumentBudgetBound, ParallelIngestRoute, ParallelIngestTerminal,
    ValidatedBatchAdmission, catch_parallel_ingest_worker, parallel_arena_chunk_bytes,
    parallel_document_logical_upper_bound, plan_parallel_ingest,
    retain_prepared_metadata_within_budget, validate_parallel_source_len,
};
#[cfg(test)]
pub(in crate::index) use query_cache::{RankedQueryCache, raw_query_cache_fingerprint};
#[cfg(test)]
pub(in crate::index) use query_cursors::{
    compiled_snippet_terms, lower_composite_sealed_term, lower_delta_term, lower_leaf_term,
    lower_term, open_dictionary, open_owned_cursor, owned_fieldnorm_bridge_calls, required_section,
    reset_owned_fieldnorm_bridge_calls, reset_term_field_ord_bridge_calls, snapshot_doc_freq,
    snapshot_field, span, term_field_ord_bridge_calls, term_field_ords,
};
#[cfg(test)]
pub(in crate::index) use query_lowering::{
    GROUPED_MAX_SCORE_ENABLED, PrunableScorerShape, QueryLeaf, QueryLoweringMode, UnionChildKind,
    lower_query, prunable_scorer_shape, query_has_prunable_root_union, two_arm_variants_glob,
};
#[cfg(test)]
pub(in crate::index) use query_predicates::{
    lower_leaf_numeric_range, snapshot_string_range_terms,
};
#[cfg(test)]
pub(in crate::index) use search_handles::RootRefreshPause;
#[cfg(test)]
pub(in crate::index) use snapshot::{
    delta_lease_overlap_comparisons, reset_delta_lease_overlap_comparisons,
};
#[cfg(test)]
pub(in crate::index) use writer::{IndexBackend, QuillWriterState};

#[cfg(test)]
mod tests;
