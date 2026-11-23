//! Scribe ingest pipeline.
//!
//! Tokenization, per-shard arenas, columnar accumulation, and segment flush
//! land behind this module in the Quill E1 milestones.
//!
//! Milestones `bd-quill-e1-scribe-bejd.3` and `.4` provide the per-shard
//! foundations the rest of Scribe builds on:
//!
//! - [`ByteArena`]: chunked bump storage with reset-and-reuse (chunk capacity
//!   is retained across flush cycles so steady-state ingest does not touch the
//!   global allocator).
//! - [`TermInterner`]: composite-key interner mapping
//!   `(field_ord, term bytes)` to dense local `u32` ids. Keys are stored once,
//!   as the exact on-disk TERMDICT composite key (big-endian `field_ord`
//!   prefix + term bytes, FSLX §5.1), so [`TermInterner::sorted_ids`] yields
//!   ids in precisely the order the dictionary serializer needs.
//! - Budget accounting ([`TermInterner::bytes_used`]) that feeds the shard
//!   flush trigger (`QuillConfig::scribe_shard_budget_bytes`).
//! - [`FrankensearchTokenizer`]: the allocation-reusing portable-SIMD path for
//!   the shipping `SimpleTokenizer + LowerCaser` semantics.
//! - [`CassAnalyzer`]: the native CASS hyphen/CJK analyzer family plus the
//!   matching edge-prefix and bounded-preview helpers.
//! - [`ColumnarAccumulator`]: schema-driven, per-field `SoA` token columns plus
//!   raw document lengths and their Tantivy-compatible fieldnorm bytes.
//!
//! Invariants:
//! - No per-token heap allocation on the intern hot path: an existing term
//!   costs one hash + one arena byte-compare; a new term costs one bump copy.
//! - Term ids are assigned in first-insertion order and are deterministic for
//!   a given ingest order (hasher choice never influences ids or output —
//!   collision buckets only affect probe cost).
//! - `CompactString`/owned allocation happens only at dictionary-build time,
//!   never per token.
//! - Document ordinals in the accumulator are shard-lease-relative and
//!   strictly ascending. Segment build is responsible for the one-time global
//!   docid rebase.
//! - Every indexed string field has parallel term-id and doc-ordinal columns;
//!   only fields whose descriptor enables positions allocate a position
//!   column.

mod accumulator;
mod analyzer;
mod cass;
mod flush;
mod interner;
mod lease;
mod radix;
mod tokenizer;

#[cfg(test)]
mod tests;

/// Default arena chunk size. 1 MiB amortizes chunk-vector growth while
/// keeping per-shard reset cheap; term corpora that exceed it simply add
/// chunks (each retained across [`ByteArena::reset`]).
pub const DEFAULT_ARENA_CHUNK_BYTES: usize = 1 << 20;

/// Composite dictionary key prefix width: big-endian `field_ord: u16`
/// (FSLX §5.1). Big-endian so raw byte comparison equals `(field, term)`
/// lexicographic order.
pub const FIELD_PREFIX_BYTES: usize = 2;

/// Number of document ordinals in one Keeper lease (Q1 §§3, 5).
///
/// Accumulator ordinals are offsets within one lease, so the valid domain is
/// `0..DOC_ORDS_PER_LEASE`. Crossing the boundary forces a segment cut before
/// a later stage rebases the offsets to global document IDs.
pub const DOC_ORDS_PER_LEASE: u32 = 1 << 16;

pub use accumulator::{
    AccumulatorError, ColumnarAccumulator, DocumentAccumulation, FieldTokenColumns,
    IndexedFieldValue, IndexedNumericValue, NumericFieldColumns, StoredFieldColumns,
    StoredFieldValue,
};
#[cfg(test)]
use analyzer::sealed;
pub use analyzer::{
    AnalysisReport, AnalyzedToken, AnalyzedTokenRef, TokenAnalyzer, UnsupportedAnalysis,
    analyze_admitted,
};
pub(crate) use cass::is_cass_cjk;
pub use cass::{
    CASS_MAX_EDGE_NGRAM_CHARS, CASS_MAX_TOKEN_BYTES, CassAnalyzer, CompiledAnalyzerFamily,
    cass_build_preview, cass_generate_edge_ngrams,
};
#[cfg(feature = "bench-internals")]
pub use flush::flush_accumulator_with_staged_stored_meta;
pub use flush::{
    DeltaFlushInput, FlushDocumentInput, FlushError, FlushMode, FlushSegmentInput,
    flush_accumulator, flush_accumulator_with_mode, flush_delta_snapshot,
};
pub(crate) use interner::TERM_BUCKET_BYTES_ESTIMATE;
pub use interner::{ArenaSpan, ByteArena, TermInterner};
#[cfg(test)]
use interner::{Bucket, BucketEntry, reset_sorted_ids_calls, sorted_ids_calls};
pub use lease::{
    BatchDocIds, DOCID_LEASE_BLOCK, DocIdAllocator, DocIdAllocatorError, DocIdSpan,
    LeaseBurnRecord, LeaseBurnReport, LeaseGrant, ShardRouter, ShardRouterError,
};
#[cfg(test)]
use radix::{
    FlushTokenRow, PARALLEL_RADIX_ROWS_PER_CHUNK, stable_radix_partition,
    stable_radix_partition_with_chunks,
};
pub use tokenizer::FrankensearchTokenizer;
#[cfg(all(test, feature = "tantivy-oracle"))]
use tokenizer::tokenizer_next_char;
#[cfg(any(test, feature = "bench-internals"))]
pub use tokenizer::{BoundaryMaskTokenizer, analyze_default_scalar_reference};
#[cfg(test)]
use tokenizer::{SWAR_LANES, next_token_position, swar_ascii_alnum_mark};
