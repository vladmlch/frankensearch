use frankensearch_core::DocId;
use thiserror::Error;

use crate::delta::DeltaSnapshot;
use crate::error::QuillError;
use crate::grimoire::{EncodedTermDictionary, TermDictionaryError, TermSectionLengths};
use crate::quiver::{
    BlockMaxError, BorrowedStoredMetaSection, DocLenCodecError, DocLenFieldInput,
    EncodedDocLenSection, EncodedIdHashSection, EncodedIdMapSection, EncodedNumericSection,
    EncodedStoredMetaSection, FieldStats, IdHashCodecError, IdMapCodecError, IdMapEntryInput,
    NumericCodecError, NumericEntry, NumericFieldInput, NumericValue, PositionCodecError,
    StatsCodecError, StoredMetaCodecError, StoredMetaFieldInput,
};
use crate::schema::{FieldKind, SchemaDescriptor};
#[cfg(feature = "bench-internals")]
use crate::segment::SectionInput;
use crate::segment::{
    EncodedSegment, PlannedSection, SectionKind, SegmentAssembler, SegmentHeaderInput,
};

use super::DOC_ORDS_PER_LEASE;
use super::accumulator::ColumnarAccumulator;
use super::analyzer::TokenAnalyzer;
use super::radix::{
    build_doclen_columns, build_id_map_inputs, collect_flush_rows, durable_len,
    encode_delta_term_streams, encode_ordered_term_streams, filled_vec, rebase_doc_id,
    schema_has_positions, stable_radix_partition_for_mode, validate_flush_documents,
};

/// One external-identity row supplied when an accumulator is sealed.
///
/// Scribe deliberately keeps document identifiers out of its token columns.
/// The ordinal makes the sidecar self-checking: positional alignment alone
/// cannot silently attach an identifier to the wrong accumulated document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushDocumentInput<'a> {
    /// Shard-lease-relative document ordinal.
    pub doc_ord: u32,
    /// Non-empty external document identifier.
    pub document_id: &'a str,
    /// Unseeded xxh3-64 witness of canonical document content.
    pub content_hash: u64,
}

impl<'a> FlushDocumentInput<'a> {
    /// Construct a row from an externally computed content witness.
    #[must_use]
    pub const fn new(doc_ord: u32, document_id: &'a str, content_hash: u64) -> Self {
        Self {
            doc_ord,
            document_id,
            content_hash,
        }
    }

    /// Construct a row while hashing canonical document content.
    #[must_use]
    pub fn from_canonical_content(
        doc_ord: u32,
        document_id: &'a str,
        canonical_content: &[u8],
    ) -> Self {
        let entry = IdMapEntryInput::from_canonical_content(document_id, canonical_content);
        Self::new(doc_ord, entry.document_id, entry.content_hash)
    }
}

/// Fixed metadata and identity sidecar for one deterministic accumulator seal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushSegmentInput<'a> {
    /// Collision-checked immutable segment identifier.
    pub segment_id: u64,
    /// Global document ID corresponding to shard-local ordinal zero.
    pub lease_docid_base: u64,
    /// Informational creation timestamp persisted in the segment header.
    pub created_unix_s: i64,
    /// Packed engine version persisted in the segment header.
    pub engine_version: u32,
    /// One identity row per completed accumulator document, in the same order.
    pub documents: &'a [FlushDocumentInput<'a>],
}

/// Fixed metadata for sealing one immutable Delta epoch.
///
/// Delta owns its lease, live identity rows, and canonical per-document seal
/// sidecars, so callers supply only the values persisted in the FSLX header.
/// Supplying these values explicitly also makes Delta and accumulator seals
/// byte-comparable under deterministic tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaFlushInput {
    /// Collision-checked immutable segment identifier.
    pub segment_id: u64,
    /// Informational creation timestamp persisted in the segment header.
    pub created_unix_s: i64,
    /// Packed engine version persisted in the segment header.
    pub engine_version: u32,
}

/// Execution strategy for the stable radix partition used during a Scribe flush.
///
/// [`FlushMode::Automatic`] preserves the shipping behavior and may use Rayon
/// once the token-row count crosses the parallel chunk boundary.
/// [`FlushMode::Scalar`] enters the serial radix implementation directly: it
/// does not query Rayon or construct a parallel iterator. The explicit scalar
/// mode is the deterministic G1a reference path; later optimized strategies can
/// be added here without changing the accumulator or segment APIs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FlushMode {
    /// Select the existing row-count- and worker-aware implementation.
    #[default]
    Automatic,
    /// Force the stable single-threaded radix implementation.
    Scalar,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum StoredMetaBuildMode {
    #[default]
    Borrowed,
    #[cfg(feature = "bench-internals")]
    Staged,
}

/// Typed failures from [`flush_accumulator`].
#[derive(Debug, Error)]
pub enum FlushError {
    /// Empty immutable segments are not produced by the shard flush path.
    #[error("cannot flush an empty Scribe accumulator")]
    EmptyAccumulator,
    /// Q1 R1 requires every shard lease to begin on its 65,536-doc boundary.
    #[error(
        "lease document base {lease_docid_base} is not aligned to {DOC_ORDS_PER_LEASE} documents"
    )]
    MisalignedLeaseBase {
        /// Rejected global lease base.
        lease_docid_base: u64,
    },
    /// The external identity sidecar must be one-for-one with completed docs.
    #[error("flush identity sidecar has {actual} rows, expected {expected}")]
    DocumentCountMismatch {
        /// Completed accumulator document count.
        expected: usize,
        /// Supplied sidecar row count.
        actual: usize,
    },
    /// An identity row drifted from its accumulator document.
    #[error(
        "flush identity row {index} names local ordinal {actual}, expected accumulator ordinal {expected}"
    )]
    DocumentOrdinalMismatch {
        /// Completed-document index.
        index: usize,
        /// Accumulator ordinal at this index.
        expected: u32,
        /// Sidecar ordinal at this index.
        actual: u32,
    },
    /// Every live Delta identity must retain its canonical IDMAP witness.
    #[error("live Delta document {global_docid} has no canonical content hash")]
    MissingDeltaContentHash {
        /// Global document identifier whose seal sidecar is incomplete.
        global_docid: u32,
    },
    /// Every live Delta row must retain one raw length per indexed string field.
    #[error("live Delta document {global_docid} has no raw length for field {field_ord}")]
    MissingDeltaFieldLength {
        /// Indexed string field ordinal.
        field_ord: u16,
        /// Global document identifier whose seal sidecar is incomplete.
        global_docid: u32,
    },
    /// Rebase would leave the FSLX u32 global-document domain.
    #[error(
        "local document ordinal {doc_ord} rebased from lease base {lease_docid_base} exceeds u32"
    )]
    DocumentIdOverflow {
        /// Supplied lease base.
        lease_docid_base: u64,
        /// Rejected local ordinal.
        doc_ord: u32,
    },
    /// A private accumulator column invariant failed before durable bytes existed.
    #[error("invalid token column for field {field_ord}: {detail}")]
    InvalidTokenColumn {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Stable invariant name.
        detail: &'static str,
    },
    /// A token row referenced no term in the accumulator interner.
    #[error("field {field_ord} token row references out-of-range local term id {term_id}")]
    TermIdOutOfRange {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Rejected local term id.
        term_id: u32,
    },
    /// Every interned term must own at least one admitted token row.
    #[error("interned local term id {term_id} has no token rows")]
    TermHasNoRows {
        /// Orphaned local term id.
        term_id: u32,
    },
    /// Stable scatter relies on per-field source rows ascending by document.
    #[error("term {term_id} rows descend at partition index {index}: {previous} then {current}")]
    NonAscendingTermDocuments {
        /// Dense local term id.
        term_id: u32,
        /// Index within the term partition.
        index: usize,
        /// Prior local document ordinal.
        previous: u32,
        /// Current local document ordinal.
        current: u32,
    },
    /// One term frequency exceeded the durable u32 field.
    #[error("term {term_id} frequency for local document {doc_ord} exceeds u32")]
    TermFrequencyOverflow {
        /// Dense local term id.
        term_id: u32,
        /// Local document ordinal.
        doc_ord: u32,
    },
    /// Position deltas must not go backwards within one posting.
    #[error(
        "term {term_id} positions descend in local document {doc_ord}: {previous} then {current}"
    )]
    NonAscendingPositions {
        /// Dense local term id.
        term_id: u32,
        /// Local document ordinal.
        doc_ord: u32,
        /// Prior absolute token position.
        previous: u32,
        /// Current absolute token position.
        current: u32,
    },
    /// Checked host or durable-size arithmetic overflowed.
    #[error("Scribe flush arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Stable computation name.
        field: &'static str,
    },
    /// A fallible flush allocation could not be satisfied.
    #[error("unable to reserve {count} entries for Scribe flush {resource}")]
    Allocation {
        /// Stable allocation purpose.
        resource: &'static str,
        /// Requested entries or bytes.
        count: usize,
    },
    /// POSTINGS/BLOCKMAX encoding failed.
    #[error("Scribe POSTINGS/BLOCKMAX encoding failed: {0}")]
    BlockMax(#[from] BlockMaxError),
    /// POSITIONS encoding failed.
    #[error("Scribe POSITIONS encoding failed: {0}")]
    Positions(#[from] PositionCodecError),
    /// TERMDICT encoding failed.
    #[error("Scribe TERMDICT encoding failed: {0}")]
    TermDictionary(#[from] TermDictionaryError),
    /// DOCLEN encoding failed.
    #[error("Scribe DOCLEN encoding failed: {0}")]
    DocLen(#[from] DocLenCodecError),
    /// IDMAP encoding failed.
    #[error("Scribe IDMAP encoding failed: {0}")]
    IdMap(#[from] IdMapCodecError),
    /// IDHASH encoding failed.
    #[error("Scribe IDHASH encoding failed: {0}")]
    IdHash(#[from] IdHashCodecError),
    /// NUMERIC encoding failed.
    #[error("Scribe NUMERIC encoding failed: {0}")]
    Numeric(#[from] NumericCodecError),
    /// STOREDMETA encoding failed.
    #[error("Scribe STOREDMETA encoding failed: {0}")]
    StoredMeta(#[from] StoredMetaCodecError),
    /// STATS encoding failed.
    #[error("Scribe STATS encoding failed: {0}")]
    Stats(#[from] StatsCodecError),
    /// Final FSLX framing failed.
    #[error("Scribe segment framing failed: {0}")]
    Segment(#[from] QuillError),
}

/// Seal one shard accumulator into a complete deterministic FSLX mini-segment.
///
/// The accumulator is borrowed and is never reset by this function. A caller
/// may therefore publish the returned bytes first and clear the shard only
/// after Keeper makes the generation durable. Token rows are stably radix
/// partitioned by dense local term id; each resulting slice is consumed
/// directly by the posting builder, including stopword-heavy slices, without
/// another per-term triple copy.
///
/// # Errors
///
/// Returns [`FlushError`] for sidecar drift, rebase overflow, accumulator
/// invariant failure, codec failure, or final segment-framing failure.
pub fn flush_accumulator<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
    input: FlushSegmentInput<'_>,
) -> Result<EncodedSegment, FlushError> {
    flush_accumulator_with_mode(accumulator, input, FlushMode::Automatic)
}

/// Seal one shard accumulator using an explicit radix execution strategy.
///
/// This is the strategy-stable entry point for callers that require the scalar
/// reference implementation. [`FlushMode::Scalar`] bypasses every Rayon-aware
/// helper and enters the serial stable radix partition directly. All modes
/// produce the same canonical FSLX bytes for the same accumulator and metadata.
///
/// # Errors
///
/// Returns [`FlushError`] under the same conditions as [`flush_accumulator`].
pub fn flush_accumulator_with_mode<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
    input: FlushSegmentInput<'_>,
    mode: FlushMode,
) -> Result<EncodedSegment, FlushError> {
    flush_accumulator_with_stored_meta_mode(accumulator, input, mode, StoredMetaBuildMode::Borrowed)
}

/// Seal one shard with the former materialized STOREDMETA section.
///
/// This benchmark-only control retains the shipping radix and segment
/// assembler while restoring the full intermediate STOREDMETA byte image.
///
/// # Errors
///
/// Returns [`FlushError`] under the same conditions as [`flush_accumulator`].
#[cfg(feature = "bench-internals")]
pub fn flush_accumulator_with_staged_stored_meta<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
    input: FlushSegmentInput<'_>,
    mode: FlushMode,
) -> Result<EncodedSegment, FlushError> {
    flush_accumulator_with_stored_meta_mode(accumulator, input, mode, StoredMetaBuildMode::Staged)
}

fn flush_accumulator_with_stored_meta_mode<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
    input: FlushSegmentInput<'_>,
    mode: FlushMode,
    stored_meta_mode: StoredMetaBuildMode,
) -> Result<EncodedSegment, FlushError> {
    let flush_span = tracing::info_span!(
        target: crate::tracing_conventions::TARGET,
        crate::tracing_conventions::SCRIBE_FLUSH,
        phase = "flush",
        doc_count = accumulator.document_count(),
        token_count = accumulator.token_count(),
        arena_bytes_used = accumulator.bytes_used(),
        arena_bytes_reserved = accumulator.bytes_reserved(),
        arena_bytes_used_high_water = accumulator.bytes_used(),
        arena_bytes_reserved_high_water = accumulator.bytes_reserved(),
        term_count = accumulator.terms().len(),
        result_count = tracing::field::Empty,
        output_bytes = tracing::field::Empty,
        duration_us = tracing::field::Empty,
        mode = ?mode,
        stored_meta_mode = ?stored_meta_mode,
    );
    let _flush_timer = crate::tracing_conventions::StageTimer::new(&flush_span);
    let _flush_entered = flush_span.enter();
    let first_doc_ord = accumulator
        .first_doc_ord()
        .ok_or(FlushError::EmptyAccumulator)?;
    let last_doc_ord = accumulator
        .last_doc_ord()
        .ok_or(FlushError::EmptyAccumulator)?;
    let schema = accumulator.schema();
    if !input
        .lease_docid_base
        .is_multiple_of(u64::from(DOC_ORDS_PER_LEASE))
    {
        return Err(FlushError::MisalignedLeaseBase {
            lease_docid_base: input.lease_docid_base,
        });
    }

    validate_flush_documents(accumulator.document_ords(), input.documents)?;
    let docid_lo = rebase_doc_id(input.lease_docid_base, first_doc_ord)?;
    let last_docid = rebase_doc_id(input.lease_docid_base, last_doc_ord)?;
    let docid_hi = last_docid
        .checked_add(1)
        .ok_or(FlushError::ArithmeticOverflow {
            field: "exclusive document high bound",
        })?;
    let span =
        usize::try_from(docid_hi - docid_lo).map_err(|_| FlushError::ArithmeticOverflow {
            field: "document span host size",
        })?;
    let doc_count = u32::try_from(accumulator.document_count()).map_err(|_| {
        FlushError::ArithmeticOverflow {
            field: "segment document count",
        }
    })?;

    let (expected_field_ords, doclen_columns) =
        build_doclen_columns(accumulator, first_doc_ord, span)?;
    let rows = collect_flush_rows(accumulator)?;
    let partition = stable_radix_partition_for_mode(rows, accumulator.terms().len(), mode)?;
    let (postings_bytes, positions_bytes, blockmax_bytes, term_inputs) =
        encode_ordered_term_streams(
            accumulator,
            &partition,
            input.lease_docid_base,
            docid_lo,
            &doclen_columns,
        )?;

    let doclen_inputs = expected_field_ords
        .iter()
        .copied()
        .zip(&doclen_columns)
        .map(|(field_ord, values)| DocLenFieldInput::new(field_ord, values))
        .collect::<Vec<_>>();
    let doclen =
        EncodedDocLenSection::encode(docid_lo, docid_hi, &expected_field_ords, &doclen_inputs)?;

    let id_map_inputs = build_id_map_inputs(
        input.documents,
        first_doc_ord,
        span,
        accumulator.document_ords(),
    )?;
    let id_map = EncodedIdMapSection::encode(docid_lo, docid_hi, &id_map_inputs)?;
    let id_hash = EncodedIdHashSection::encode(id_map.section()?)?;

    let numeric = if accumulator.numeric_fields().is_empty() {
        None
    } else {
        Some(EncodedNumericSection::encode_accumulator(
            schema,
            docid_lo,
            docid_hi,
            input.lease_docid_base,
            accumulator,
        )?)
    };

    let stored_meta = if accumulator.stored_fields().is_empty() {
        None
    } else {
        Some(match stored_meta_mode {
            StoredMetaBuildMode::Borrowed => {
                AccumulatorStoredMeta::Borrowed(BorrowedStoredMetaSection::encode_accumulator(
                    docid_lo,
                    docid_hi,
                    input.lease_docid_base,
                    accumulator,
                )?)
            }
            #[cfg(feature = "bench-internals")]
            StoredMetaBuildMode::Staged => {
                AccumulatorStoredMeta::Staged(EncodedStoredMetaSection::encode_accumulator(
                    docid_lo,
                    docid_hi,
                    input.lease_docid_base,
                    accumulator,
                )?)
            }
        })
    };
    let stats_rows = accumulator
        .fields()
        .iter()
        .map(|field| FieldStats::new(field.field_ord(), field.total_tokens(), doc_count))
        .collect::<Vec<_>>();
    let stats = EncodedStatsSection::encode(&expected_field_ords, &stats_rows, doc_count)?;
    let segment_input = DeltaFlushInput {
        segment_id: input.segment_id,
        created_unix_s: input.created_unix_s,
        engine_version: input.engine_version,
    };
    let term_streams = (postings_bytes, positions_bytes, blockmax_bytes, term_inputs);
    let encoded = match stored_meta.as_ref() {
        #[cfg(feature = "bench-internals")]
        Some(AccumulatorStoredMeta::Staged(section)) => encode_canonical_segment_staged(
            segment_input,
            schema,
            docid_lo,
            docid_hi,
            doc_count,
            term_streams,
            &doclen,
            &id_map,
            &id_hash,
            numeric.as_ref(),
            Some(section),
            &stats,
        )?,
        _ => encode_canonical_segment(
            segment_input,
            schema,
            docid_lo,
            docid_hi,
            doc_count,
            term_streams,
            &doclen,
            &id_map,
            &id_hash,
            numeric.as_ref(),
            stored_meta.as_ref().map(AccumulatorStoredMeta::source),
            &stats,
        )?,
    };
    flush_span.record("result_count", u64::from(doc_count));
    flush_span.record("output_bytes", encoded.file_len());
    Ok(encoded)
}

/// Seal one immutable Delta epoch into the canonical FSLX mini-segment format.
///
/// Delta chains are already grouped by term, so this path skips only Scribe's
/// token-row radix partition. It sorts terms by the shared interner's canonical
/// composite bytes, folds Delta-local tombstones into holes, and then enters
/// the same posting/block-max append and final section assembly used by
/// [`flush_accumulator`]. The snapshot is borrowed for the complete build and
/// remains independently publishable until Keeper makes the returned bytes
/// durable.
///
/// # Errors
///
/// Returns [`FlushError`] when a live Delta row lacks a required seal sidecar,
/// when an immutable Delta invariant is inconsistent, or when any canonical
/// section codec rejects the lowered rows.
pub fn flush_delta_snapshot(
    snapshot: &DeltaSnapshot,
    input: DeltaFlushInput,
) -> Result<Option<EncodedSegment>, FlushError> {
    let mut live_documents = Vec::new();
    live_documents
        .try_reserve_exact(snapshot.live_document_count())
        .map_err(|_| FlushError::Allocation {
            resource: "live Delta document table",
            count: snapshot.live_document_count(),
        })?;
    live_documents.extend(snapshot.live_documents());
    let Some(&(first_docid, _)) = live_documents.first() else {
        return Ok(None);
    };
    let &(last_docid, _) = live_documents
        .last()
        .expect("a nonempty live Delta document table has a final row");
    let docid_lo = u64::from(first_docid);
    let docid_hi = u64::from(last_docid)
        .checked_add(1)
        .ok_or(FlushError::ArithmeticOverflow {
            field: "exclusive Delta document high bound",
        })?;
    let span =
        usize::try_from(docid_hi - docid_lo).map_err(|_| FlushError::ArithmeticOverflow {
            field: "Delta document span host size",
        })?;
    let doc_count =
        u32::try_from(live_documents.len()).map_err(|_| FlushError::ArithmeticOverflow {
            field: "Delta segment document count",
        })?;
    let schema = snapshot.schema();

    let (expected_field_ords, doclen_columns, stats_rows) =
        build_delta_doclen_and_stats(snapshot, &live_documents, docid_lo, span, doc_count)?;
    let term_streams =
        encode_delta_term_streams(snapshot, docid_lo, &expected_field_ords, &doclen_columns)?;
    let doclen_inputs = expected_field_ords
        .iter()
        .copied()
        .zip(&doclen_columns)
        .map(|(field_ord, values)| DocLenFieldInput::new(field_ord, values))
        .collect::<Vec<_>>();
    let doclen =
        EncodedDocLenSection::encode(docid_lo, docid_hi, &expected_field_ords, &doclen_inputs)?;

    let id_map_inputs = build_delta_id_map_inputs(snapshot, &live_documents, docid_lo, span)?;
    let id_map = EncodedIdMapSection::encode(docid_lo, docid_hi, &id_map_inputs)?;
    let id_hash = EncodedIdHashSection::encode(id_map.section()?)?;
    let numeric = encode_delta_numeric(snapshot, &live_documents, docid_lo, docid_hi)?;
    let stored_meta =
        encode_delta_stored_meta(snapshot, &live_documents, docid_lo, docid_hi, span)?;
    let stats = EncodedStatsSection::encode(&expected_field_ords, &stats_rows, doc_count)?;

    encode_canonical_segment(
        input,
        schema,
        docid_lo,
        docid_hi,
        doc_count,
        term_streams,
        &doclen,
        &id_map,
        &id_hash,
        numeric.as_ref(),
        stored_meta
            .as_ref()
            .map(|stored_meta| StoredMetaSource::Encoded(stored_meta.as_bytes())),
        &stats,
    )
    .map(Some)
}

type LiveDeltaDocument<'a> = (u32, &'a DocId);
type DeltaDoclenAndStats = (Vec<u16>, Vec<Vec<Option<u32>>>, Vec<FieldStats>);

fn build_delta_doclen_and_stats(
    snapshot: &DeltaSnapshot,
    live_documents: &[LiveDeltaDocument<'_>],
    docid_lo: u64,
    span: usize,
    doc_count: u32,
) -> Result<DeltaDoclenAndStats, FlushError> {
    let mut field_ords = Vec::new();
    let indexed_field_count = snapshot
        .schema()
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        .count();
    field_ords
        .try_reserve_exact(indexed_field_count)
        .map_err(|_| FlushError::Allocation {
            resource: "Delta DOCLEN field ordinals",
            count: indexed_field_count,
        })?;
    let mut columns = Vec::new();
    columns
        .try_reserve_exact(indexed_field_count)
        .map_err(|_| FlushError::Allocation {
            resource: "Delta DOCLEN columns",
            count: indexed_field_count,
        })?;
    let mut stats_rows = Vec::new();
    stats_rows
        .try_reserve_exact(indexed_field_count)
        .map_err(|_| FlushError::Allocation {
            resource: "Delta STATS rows",
            count: indexed_field_count,
        })?;

    for field in snapshot
        .schema()
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
    {
        let mut column = filled_vec(span, None, "Delta DOCLEN span")?;
        let mut total_tokens = 0_u64;
        for &(global_docid, _) in live_documents {
            let relative = u64::from(global_docid).checked_sub(docid_lo).ok_or(
                FlushError::ArithmeticOverflow {
                    field: "Delta DOCLEN relative document index",
                },
            )?;
            let relative =
                usize::try_from(relative).map_err(|_| FlushError::ArithmeticOverflow {
                    field: "Delta DOCLEN relative document host index",
                })?;
            let raw_length = snapshot
                .segment()
                .raw_field_length(field.id, global_docid)
                .ok_or(FlushError::MissingDeltaFieldLength {
                    field_ord: field.id,
                    global_docid,
                })?;
            let slot = column
                .get_mut(relative)
                .ok_or(FlushError::ArithmeticOverflow {
                    field: "Delta DOCLEN document span",
                })?;
            *slot = Some(raw_length);
            total_tokens = total_tokens.checked_add(u64::from(raw_length)).ok_or(
                FlushError::ArithmeticOverflow {
                    field: "Delta STATS token numerator",
                },
            )?;
        }
        field_ords.push(field.id);
        columns.push(column);
        stats_rows.push(FieldStats::new(field.id, total_tokens, doc_count));
    }
    Ok((field_ords, columns, stats_rows))
}

fn build_delta_id_map_inputs<'a>(
    snapshot: &'a DeltaSnapshot,
    live_documents: &[LiveDeltaDocument<'a>],
    docid_lo: u64,
    span: usize,
) -> Result<Vec<Option<IdMapEntryInput<'a>>>, FlushError> {
    let mut entries = filled_vec(span, None, "Delta IDMAP span")?;
    for &(global_docid, document_id) in live_documents {
        let relative = u64::from(global_docid).checked_sub(docid_lo).ok_or(
            FlushError::ArithmeticOverflow {
                field: "Delta IDMAP relative document index",
            },
        )?;
        let relative = usize::try_from(relative).map_err(|_| FlushError::ArithmeticOverflow {
            field: "Delta IDMAP relative document host index",
        })?;
        let content_hash = snapshot
            .content_hash(global_docid)
            .ok_or(FlushError::MissingDeltaContentHash { global_docid })?;
        let slot = entries
            .get_mut(relative)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "Delta IDMAP document span",
            })?;
        *slot = Some(IdMapEntryInput::new(document_id.as_str(), content_hash));
    }
    Ok(entries)
}

fn encode_delta_numeric(
    snapshot: &DeltaSnapshot,
    live_documents: &[LiveDeltaDocument<'_>],
    docid_lo: u64,
    docid_hi: u64,
) -> Result<Option<EncodedNumericSection>, FlushError> {
    let numeric_fields = snapshot
        .schema()
        .fields
        .iter()
        .filter(|field| field.kind.has_numeric_column())
        .collect::<Vec<_>>();
    if numeric_fields.is_empty() {
        return Ok(None);
    }

    let mut columns = Vec::new();
    columns
        .try_reserve_exact(numeric_fields.len())
        .map_err(|_| FlushError::Allocation {
            resource: "Delta NUMERIC columns",
            count: numeric_fields.len(),
        })?;
    for field in &numeric_fields {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(live_documents.len())
            .map_err(|_| FlushError::Allocation {
                resource: "Delta NUMERIC entries",
                count: live_documents.len(),
            })?;
        for &(global_docid, _) in live_documents {
            let Some(value) = snapshot.numeric_value(field.id, global_docid) else {
                continue;
            };
            entries.push(match value {
                NumericValue::I64(value) => NumericEntry::i64(value, global_docid),
                NumericValue::U64(value) => NumericEntry::u64(value, global_docid),
            });
        }
        columns.push(entries);
    }
    let inputs = numeric_fields
        .iter()
        .zip(&columns)
        .map(|(field, entries)| NumericFieldInput::new(field.id, entries))
        .collect::<Vec<_>>();
    Ok(Some(EncodedNumericSection::encode(
        snapshot.schema(),
        docid_lo,
        docid_hi,
        &inputs,
    )?))
}

fn encode_delta_stored_meta<'a>(
    snapshot: &'a DeltaSnapshot,
    live_documents: &[LiveDeltaDocument<'a>],
    docid_lo: u64,
    docid_hi: u64,
    span: usize,
) -> Result<Option<EncodedStoredMetaSection>, FlushError> {
    let stored_fields = snapshot
        .schema()
        .fields
        .iter()
        .filter(|field| field.stored)
        .collect::<Vec<_>>();
    if stored_fields.is_empty() {
        return Ok(None);
    }

    let mut columns = Vec::new();
    columns
        .try_reserve_exact(stored_fields.len())
        .map_err(|_| FlushError::Allocation {
            resource: "Delta STOREDMETA columns",
            count: stored_fields.len(),
        })?;
    for field in &stored_fields {
        let mut values = filled_vec(span, None, "Delta STOREDMETA span")?;
        for &(global_docid, _) in live_documents {
            let relative = u64::from(global_docid).checked_sub(docid_lo).ok_or(
                FlushError::ArithmeticOverflow {
                    field: "Delta STOREDMETA relative document index",
                },
            )?;
            let relative =
                usize::try_from(relative).map_err(|_| FlushError::ArithmeticOverflow {
                    field: "Delta STOREDMETA relative document host index",
                })?;
            let slot = values
                .get_mut(relative)
                .ok_or(FlushError::ArithmeticOverflow {
                    field: "Delta STOREDMETA document span",
                })?;
            *slot = snapshot.stored_value(field.id, global_docid);
        }
        columns.push(values);
    }
    let expected_field_ords = stored_fields
        .iter()
        .map(|field| field.id)
        .collect::<Vec<_>>();
    let inputs = stored_fields
        .iter()
        .zip(&columns)
        .map(|(field, values)| StoredMetaFieldInput::new(field.id, values))
        .collect::<Vec<_>>();
    Ok(Some(EncodedStoredMetaSection::encode(
        docid_lo,
        docid_hi,
        &expected_field_ords,
        &inputs,
    )?))
}

enum AccumulatorStoredMeta<'a> {
    Borrowed(BorrowedStoredMetaSection<'a>),
    #[cfg(feature = "bench-internals")]
    Staged(EncodedStoredMetaSection),
}

impl<'a> AccumulatorStoredMeta<'a> {
    fn source<'view>(&'view self) -> StoredMetaSource<'view, 'a> {
        match self {
            Self::Borrowed(section) => StoredMetaSource::Borrowed(section),
            #[cfg(feature = "bench-internals")]
            Self::Staged(section) => StoredMetaSource::Encoded(section.as_bytes()),
        }
    }
}

#[derive(Clone, Copy)]
enum StoredMetaSource<'view, 'data> {
    Encoded(&'view [u8]),
    Borrowed(&'view BorrowedStoredMetaSection<'data>),
}

impl StoredMetaSource<'_, '_> {
    fn len(self) -> usize {
        match self {
            Self::Encoded(bytes) => bytes.len(),
            Self::Borrowed(section) => section.len(),
        }
    }

    fn write_into(self, output: &mut Vec<u8>) {
        match self {
            Self::Encoded(bytes) => output.extend_from_slice(bytes),
            Self::Borrowed(section) => section.write_into(output),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_canonical_segment(
    input: DeltaFlushInput,
    schema: SchemaDescriptor,
    docid_lo: u64,
    docid_hi: u64,
    doc_count: u32,
    term_streams: OrderedTermStreams<'_>,
    doclen: &EncodedDocLenSection,
    id_map: &EncodedIdMapSection,
    id_hash: &EncodedIdHashSection,
    numeric: Option<&EncodedNumericSection>,
    stored_meta: Option<StoredMetaSource<'_, '_>>,
    stats: &EncodedStatsSection,
) -> Result<EncodedSegment, FlushError> {
    let (postings_bytes, positions_bytes, blockmax_bytes, term_inputs) = term_streams;
    let term_sections = TermSectionLengths {
        postings: durable_len(&postings_bytes, "POSTINGS length")?,
        positions: schema_has_positions(schema)
            .then(|| durable_len(&positions_bytes, "POSITIONS length"))
            .transpose()?,
        blockmax: durable_len(&blockmax_bytes, "BLOCKMAX length")?,
    };
    let termdict = EncodedTermDictionary::encode_sorted(schema, term_sections, &term_inputs)?;

    let mut planned = Vec::new();
    planned
        .try_reserve_exact(10)
        .map_err(|_| FlushError::Allocation {
            resource: "section plan",
            count: 10,
        })?;
    planned.push(PlannedSection::new(
        SectionKind::TERMDICT,
        termdict.as_bytes().len(),
    ));
    planned.push(PlannedSection::new(
        SectionKind::POSTINGS,
        postings_bytes.len(),
    ));
    if schema_has_positions(schema) {
        planned.push(PlannedSection::new(
            SectionKind::POSITIONS,
            positions_bytes.len(),
        ));
    }
    planned.push(PlannedSection::new(
        SectionKind::BLOCKMAX,
        blockmax_bytes.len(),
    ));
    planned.push(PlannedSection::new(
        SectionKind::DOCLEN,
        doclen.as_bytes().len(),
    ));
    planned.push(PlannedSection::new(
        SectionKind::IDMAP,
        id_map.as_bytes().len(),
    ));
    planned.push(PlannedSection::new(
        SectionKind::IDHASH,
        id_hash.as_bytes().len(),
    ));
    if let Some(numeric) = numeric {
        planned.push(PlannedSection::new(
            SectionKind::NUMERIC,
            numeric.as_bytes().len(),
        ));
    }
    if let Some(stored_meta) = stored_meta {
        planned.push(PlannedSection::new(
            SectionKind::STOREDMETA,
            stored_meta.len(),
        ));
    }
    planned.push(PlannedSection::new(
        SectionKind::STATS,
        stats.as_bytes().len(),
    ));

    let mut assembler = SegmentAssembler::new(
        SegmentHeaderInput {
            segment_id: input.segment_id,
            schema,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s: input.created_unix_s,
            engine_version: input.engine_version,
        },
        &planned,
    )?;
    assembler.copy_section(SectionKind::TERMDICT, termdict.as_bytes())?;
    assembler.copy_section(SectionKind::POSTINGS, &postings_bytes)?;
    if schema_has_positions(schema) {
        assembler.copy_section(SectionKind::POSITIONS, &positions_bytes)?;
    }
    assembler.copy_section(SectionKind::BLOCKMAX, &blockmax_bytes)?;
    assembler.copy_section(SectionKind::DOCLEN, doclen.as_bytes())?;
    assembler.copy_section(SectionKind::IDMAP, id_map.as_bytes())?;
    assembler.copy_section(SectionKind::IDHASH, id_hash.as_bytes())?;
    if let Some(numeric) = numeric {
        assembler.copy_section(SectionKind::NUMERIC, numeric.as_bytes())?;
    }
    if let Some(stored_meta) = stored_meta {
        assembler.write_section(SectionKind::STOREDMETA, |output| {
            stored_meta.write_into(output);
        })?;
    }
    assembler.copy_section(SectionKind::STATS, stats.as_bytes())?;
    assembler.finish().map_err(FlushError::from)
}

/// Exact pre-borrowed-emission writer retained only as a same-invocation
/// benchmark incumbent. It materializes STOREDMETA and then copies every
/// section through `EncodedSegment::encode`, matching the former shipping
/// composition rather than sharing the candidate's final assembler.
#[cfg(feature = "bench-internals")]
#[allow(clippy::too_many_arguments)]
fn encode_canonical_segment_staged(
    input: DeltaFlushInput,
    schema: SchemaDescriptor,
    docid_lo: u64,
    docid_hi: u64,
    doc_count: u32,
    term_streams: OrderedTermStreams<'_>,
    doclen: &EncodedDocLenSection,
    id_map: &EncodedIdMapSection,
    id_hash: &EncodedIdHashSection,
    numeric: Option<&EncodedNumericSection>,
    stored_meta: Option<&EncodedStoredMetaSection>,
    stats: &EncodedStatsSection,
) -> Result<EncodedSegment, FlushError> {
    let (postings_bytes, positions_bytes, blockmax_bytes, term_inputs) = term_streams;
    let term_sections = TermSectionLengths {
        postings: durable_len(&postings_bytes, "POSTINGS length")?,
        positions: schema_has_positions(schema)
            .then(|| durable_len(&positions_bytes, "POSITIONS length"))
            .transpose()?,
        blockmax: durable_len(&blockmax_bytes, "BLOCKMAX length")?,
    };
    let termdict = EncodedTermDictionary::encode_sorted(schema, term_sections, &term_inputs)?;

    let mut sections = Vec::new();
    sections
        .try_reserve_exact(10)
        .map_err(|_| FlushError::Allocation {
            resource: "staged section inputs",
            count: 10,
        })?;
    sections.push(SectionInput::new(
        SectionKind::TERMDICT,
        termdict.as_bytes(),
    ));
    sections.push(SectionInput::new(SectionKind::POSTINGS, &postings_bytes));
    if schema_has_positions(schema) {
        sections.push(SectionInput::new(SectionKind::POSITIONS, &positions_bytes));
    }
    sections.push(SectionInput::new(SectionKind::BLOCKMAX, &blockmax_bytes));
    sections.push(SectionInput::new(SectionKind::DOCLEN, doclen.as_bytes()));
    sections.push(SectionInput::new(SectionKind::IDMAP, id_map.as_bytes()));
    sections.push(SectionInput::new(SectionKind::IDHASH, id_hash.as_bytes()));
    if let Some(numeric) = numeric {
        sections.push(SectionInput::new(SectionKind::NUMERIC, numeric.as_bytes()));
    }
    if let Some(stored_meta) = stored_meta {
        sections.push(SectionInput::new(
            SectionKind::STOREDMETA,
            stored_meta.as_bytes(),
        ));
    }
    sections.push(SectionInput::new(SectionKind::STATS, stats.as_bytes()));

    EncodedSegment::encode(
        SegmentHeaderInput {
            segment_id: input.segment_id,
            schema,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s: input.created_unix_s,
            engine_version: input.engine_version,
        },
        &sections,
    )
    .map_err(FlushError::from)
}
