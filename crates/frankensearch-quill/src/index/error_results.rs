use crate::argus::{ArgusError, PruningTelemetry};
use crate::config::QuillConfig;
use crate::error::QuillError;
use crate::grimoire::TermDictionaryError;
use crate::keeper::{
    CurrentPointer, CurrentPointerError, KeeperError, KeeperSnapshot, KeeperWriter,
    PublicationAuthorityPhase, PublicationAuthorityState, TierPolicyError,
};
use crate::query::{
    QueryCapabilityError, QueryDiagnostic, QueryExplanation, QueryParserConfigError,
};
use crate::quiver::{
    BlockMaxError, DocLenCodecError, PositionCodecError, PostingCodecError, StoredMetaCodecError,
};
use crate::schema::SchemaDescriptor;
use crate::scribe::{AccumulatorError, FlushError};
use asupersync::Cx;
use asupersync::sync::{LockError, TryLockError};
use frankensearch_core::SearchError;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

use super::snapshot::SnapshotError;

pub(super) const ID_FIELD: u16 = 0;
pub(super) const CONTENT_FIELD: u16 = 1;
pub(super) const TITLE_FIELD: u16 = 2;
pub(super) const METADATA_FIELD: u16 = 3;
pub(super) const ORD_FIELD: u16 = 4;
pub(super) const MAX_GLOBAL_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;
pub(super) const CONTENT_HASH_DOMAIN: &[u8] = b"frankensearch.quill.idmap-content.v2\0";

/// Typed failure from the scalar shipping facade.
#[derive(Debug, Error)]
pub enum QuillIndexError {
    /// Engine configuration was rejected before opening resources.
    #[error("invalid Quill index configuration: {0}")]
    Config(#[source] frankensearch_core::SearchError),
    /// Keeper admission, recovery, or publication failed.
    #[error(transparent)]
    Keeper(#[from] KeeperError),
    /// The process-local published view no longer agrees with durable Keeper
    /// authority and must be reconciled before it can be returned to callers.
    #[error(
        "published Quill view requires reconciliation: published generation \
         {published_generation}, durable generation {durable_generation}"
    )]
    PublicationReconciliationRequired {
        /// Generation currently installed in the process-local reader view.
        published_generation: u64,
        /// Generation proven by the retained Keeper authority view.
        durable_generation: u64,
    },
    /// A lexical-root CURRENT pointer or layout could not be validated.
    #[error(transparent)]
    CurrentPointer(#[from] CurrentPointerError),
    /// FSLX framing or section access failed.
    #[error(transparent)]
    Quill(#[from] QuillError),
    /// Parser/schema binding failed.
    #[error(transparent)]
    Parser(#[from] QueryParserConfigError),
    /// A parsed or prebuilt operator requires an index capability the schema
    /// does not provide.
    #[error(transparent)]
    QueryCapability(#[from] QueryCapabilityError),
    /// One document could not be accumulated atomically.
    #[error(transparent)]
    Accumulator(#[from] AccumulatorError),
    /// A scalar accumulator could not be sealed.
    #[error(transparent)]
    Flush(#[from] FlushError),
    /// Bound-consecutive tier planning failed.
    #[error(transparent)]
    TierPolicy(#[from] TierPolicyError),
    /// A term dictionary was malformed or incompatible with its sections.
    #[error(transparent)]
    Dictionary(#[from] TermDictionaryError),
    /// A posting list was malformed.
    #[error(transparent)]
    Postings(#[from] PostingCodecError),
    /// A block-max list was malformed or inconsistent with POSTINGS/DOCLEN.
    #[error(transparent)]
    BlockMax(#[from] BlockMaxError),
    /// A positions list was malformed.
    #[error(transparent)]
    Positions(#[from] PositionCodecError),
    /// A field-length section was malformed.
    #[error(transparent)]
    DocLen(#[from] DocLenCodecError),
    /// A stored metadata column was malformed.
    #[error(transparent)]
    StoredMeta(#[from] StoredMetaCodecError),
    /// Exhaustive scorer construction or collection failed.
    #[error(transparent)]
    Argus(ArgusError),
    /// Composite process-local snapshot construction or publication failed.
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
    /// Canonical metadata serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The requested query shape is outside the scalar G1a surface.
    #[error("unsupported scalar Quill query: {detail}")]
    UnsupportedQuery { detail: String },
    /// An index lifecycle or arithmetic invariant failed.
    #[error("invalid scalar Quill index state: {detail}")]
    InvalidState { detail: String },
    /// Structured cancellation was observed before a state transition.
    #[error("scalar Quill operation cancelled during {phase}")]
    Cancelled { phase: &'static str },
    /// Deterministic coarse query work exceeded the configured budget.
    #[error(
        "scalar Quill query fuel exhausted after {consumed}/{budget} units \
         (segments={segments_touched}, dictionary_blocks={dictionary_blocks}, \
         posting_blocks={posting_blocks}, position_docs={position_docs})"
    )]
    QueryFuelExhausted {
        /// Configured work-unit ceiling.
        budget: u64,
        /// Successfully admitted work before exhaustion.
        consumed: u64,
        /// Segment transitions admitted before exhaustion.
        segments_touched: u64,
        /// Dictionary blocks admitted before exhaustion.
        dictionary_blocks: u64,
        /// Posting blocks admitted before exhaustion.
        posting_blocks: u64,
        /// Per-document phrase-position verification or snippet enrichment admitted before exhaustion.
        position_docs: u64,
    },
}

impl From<ArgusError> for QuillIndexError {
    pub(super) fn from(error: ArgusError) -> Self {
        match error {
            ArgusError::QueryCancelled { phase } => Self::Cancelled { phase },
            ArgusError::QueryFuelExhausted {
                budget,
                consumed,
                segments_touched,
                dictionary_blocks,
                posting_blocks,
                position_docs,
            } => Self::QueryFuelExhausted {
                budget,
                consumed,
                segments_touched,
                dictionary_blocks,
                posting_blocks,
                position_docs,
            },
            other => Self::Argus(other),
        }
    }
}

/// One final lexical winner.
#[derive(Clone, Debug, PartialEq)]
pub struct QuillHit {
    /// Stable external document identifier.
    pub document_id: String,
    /// Snapshot-global Q1 document identifier used as the native tie key.
    pub global_docid: u32,
    /// Exhaustive BM25 score.
    pub score: f32,
}

/// One exact IDHASH/IDMAP identity witness from the published snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuillDocumentWitness {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// Unseeded xxh3-64 witness over the canonical indexed document.
    pub content_hash: u64,
}

/// One enriched Quill hit returned by [`super::QuillIndex::search_with_snippets`].
#[derive(Clone, Debug)]
pub struct QuillSnippetHit {
    /// Stable external document identifier.
    pub document_id: String,
    /// Exhaustive BM25 score.
    pub score: f32,
    /// Zero-based result rank.
    pub rank: usize,
    /// Highlighted content fragment when the active schema stores source text.
    pub snippet: Option<String>,
    /// Stable explanation of the raw query shape.
    pub query_type: QueryExplanation,
    /// Canonical stored metadata.
    pub metadata: Option<Arc<serde_json::Value>>,
}

/// The kind of query clause a [`QuillTermScore`] itemizes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuillClauseKind {
    /// One analyzed term.
    Term,
    /// An exact-position phrase.
    Phrase,
    /// A glob pattern.
    Glob,
}

/// One query clause's share of a document's BM25 score in one field.
#[derive(Clone, Debug, PartialEq)]
pub struct QuillTermScore {
    /// The analyzed term, a phrase's terms joined by spaces, or a glob pattern.
    pub text: String,
    /// What kind of clause `text` came from.
    pub kind: QuillClauseKind,
    /// Schema name of the field the clause was scored in.
    pub field: &'static str,
    /// Contribution to the document's score, field and query boosts
    /// included; `0.0` when the clause does not match the document there.
    pub score: f32,
    /// Documents holding the term in this field, as the scorer counts them
    /// (terms only).
    pub doc_freq: Option<u64>,
    /// BM25 inverse document frequency the scorer applied (terms only).
    pub idf: Option<f32>,
}

/// A document's BM25 score split by query clause and field; see
/// [`super::QuillSearchIndex::explain_document`].
#[derive(Clone, Debug, PartialEq)]
pub struct QuillScoreExplanation {
    /// The document's score for the whole query, as a search reports it.
    pub score: f32,
    /// One entry per term, phrase or glob clause and field, in query order.
    /// Their scores add up to `score` up to f32 rounding; set, range and
    /// match-all clauses score a constant and are not itemized.
    pub terms: Vec<QuillTermScore>,
}

/// One globally paginated exhaustive result.
#[derive(Clone, Debug, PartialEq)]
pub struct QuillSearchResult {
    /// Immutable page-local hits in `(score desc, global_docid asc)` order.
    ///
    /// Cloning this page is constant-time and keeps the exact snapshot result
    /// alive across cache eviction or a later publication.
    pub hits: Arc<[QuillHit]>,
    /// Exact match count when requested.
    pub total_count: Option<u64>,
    /// Public live-document count for the committed snapshot.
    pub doc_count: u64,
    /// Lenient parser recovery diagnostics.
    pub diagnostics: Vec<QueryDiagnostic>,
}

impl From<QuillIndexError> for SearchError {
    pub(super) fn from(error: QuillIndexError) -> Self {
        match error {
            QuillIndexError::Config(source) => source,
            QuillIndexError::Keeper(source) => source.into(),
            QuillIndexError::Quill(source) => source.into(),
            QuillIndexError::Cancelled { phase } => Self::Cancelled {
                phase: phase.to_owned(),
                reason: "Quill observed request cancellation".to_owned(),
            },
            source => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(source),
            },
        }
    }
}

pub(super) fn record_pruning_telemetry(span: &tracing::Span, telemetry: PruningTelemetry) {
    span.record("plan", telemetry.plan());
    span.record("pruning_windows", telemetry.pruning_windows());
    span.record("blocks_skipped", telemetry.blocks_skipped());
    span.record("candidate_docs", telemetry.candidate_docs());
}

pub(super) fn record_snapshot_fields(span: &tracing::Span, snapshot: &KeeperSnapshot) {
    let manifest = &snapshot.loaded_manifest().manifest;
    span.record("generation", manifest.generation);
    span.record(
        "segment_count",
        u64::try_from(manifest.segments.len()).unwrap_or(u64::MAX),
    );
    span.record("doc_count", snapshot.doc_count());
}

pub(super) fn validate_config(config: &QuillConfig) -> Result<(), QuillIndexError> {
    config.validate().map_err(QuillIndexError::Config)
}

pub(super) fn validate_non_durable_quarantine(config: &QuillConfig) -> Result<(), QuillIndexError> {
    if config.quarantine_on_unrepairable {
        return Err(QuillIndexError::Config(SearchError::InvalidConfig {
            field: "quarantine_on_unrepairable".to_owned(),
            value: "true".to_owned(),
            reason: "requires an on-disk writer; in-memory and read-only indexes cannot \
                     quarantine segments"
                .to_owned(),
        }));
    }
    Ok(())
}

/// Open (or with `create`, create) the Keeper writer of an index without
/// repair sidecars. `quarantine_on_unrepairable` selects the writer that
/// verifies every segment and quarantines an unrepairable one.
pub(super) async fn open_sidecarless_writer(
    cx: &Cx,
    directory: PathBuf,
    schema: SchemaDescriptor,
    config: &QuillConfig,
    create: bool,
) -> Result<KeeperWriter, QuillIndexError> {
    if config.quarantine_on_unrepairable {
        return Err(QuillIndexError::Config(SearchError::InvalidConfig {
            field: "quarantine_on_unrepairable".to_owned(),
            value: "true".to_owned(),
            reason: "segment quarantine requires the durability feature".to_owned(),
        }));
    }
    Ok(if create {
        KeeperWriter::create(cx, directory, schema).await?
    } else {
        KeeperWriter::open(cx, directory, schema).await?
    })
}
pub(super) fn check_cancel(cx: &Cx, phase: &'static str) -> Result<(), QuillIndexError> {
    // bd-tb4c4: hot-path cancellation poll uses the Cx cancellation flag.
    if cx.is_cancelled() {
        Err(QuillIndexError::Cancelled { phase })
    } else {
        Ok(())
    }
}

pub(super) fn map_lock_error_for_index(error: LockError, phase: &'static str) -> QuillIndexError {
    match error {
        LockError::Cancelled | LockError::TimedOut(_) => QuillIndexError::Cancelled { phase },
        LockError::Poisoned | LockError::PolledAfterCompletion => {
            invalid_state(format!("Quill writer lock failed during {phase}: {error}"))
        }
    }
}

pub(super) fn map_try_lock_error(error: TryLockError) -> QuillIndexError {
    invalid_state(format!("Quill writer is unavailable: {error}"))
}

pub(super) fn publication_state_reconciliation_required(
    state: PublicationAuthorityState,
    published_generation: u64,
) -> QuillIndexError {
    // `DurableAhead(g)` is the one phase that proves a particular successor:
    // the blocking publisher writes it immediately after `MANIFEST` names
    // g+1. `Indeterminate(g)` deliberately reports only g, because claiming
    // that g+1 won before reopening would overstate durability.
    let durable_generation = if state.phase() == PublicationAuthorityPhase::DurableAhead {
        state.generation().saturating_add(1)
    } else {
        state.generation()
    };
    QuillIndexError::PublicationReconciliationRequired {
        published_generation,
        durable_generation,
    }
}

pub(super) fn invalid_state(detail: impl Into<String>) -> QuillIndexError {
    QuillIndexError::InvalidState {
        detail: detail.into(),
    }
}

pub(super) fn wall_clock_unix_s() -> Result<i64, QuillIndexError> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid_state("system clock precedes Unix epoch"))?
        .as_secs();
    i64::try_from(seconds).map_err(|_| invalid_state("Unix timestamp does not fit i64"))
}

/// The writer session's allocator opens exactly at the manifest watermark (a
/// lease may start inside its block); only a watermark beyond the u32 docid
/// space is refused.
pub(super) fn validate_docid_watermark(watermark: u64) -> Result<u64, QuillIndexError> {
    if watermark > MAX_GLOBAL_DOCID_EXCLUSIVE {
        return Err(invalid_state(
            "manifest document-id watermark exceeds the Q1 address space",
        ));
    }
    Ok(watermark)
}
