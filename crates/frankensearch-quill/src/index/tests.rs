#![allow(clippy::wildcard_imports)]

use super::*;

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
// Unconditional: the shipping-driver ingest-span capture needs both on every
// test build, not only under bench+conformance.
use std::io::{self, Write};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, SystemTime};

#[cfg(feature = "pruning-conformance")]
use arc_swap::ArcSwapOption;
use asupersync::Cx;
use asupersync::runtime::yield_now;
use asupersync::sync::OwnedMutexGuard;
use asupersync::types::Budget;
use asupersync::{LabConfig, LabRuntime};
use frankensearch_core::{
    DocId, IndexableDocument, LexicalHydrationContext, LexicalRead, LexicalWrite, ScoredResult,
    SearchError,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "bench-internals")]
use crate::argus::TermRecordOption;
use crate::argus::{
    ArgusError, CollectedTopDocs, DeltaFieldNorms, DeltaPostingCursor, DocSetCollector,
    FieldNormReader, PostingCursor, QueryWorkCheckpoint, QueryWorkKind, ReferenceScorer,
    TopDocsCollector,
};
#[cfg(feature = "pruning-conformance")]
use crate::argus::{ConformanceQueryWorkContext, ConformanceQueryWorkPhase};
use crate::config::QuillConfig;
use crate::contract::fieldnorm_to_id;
use crate::delta::{
    DeltaFieldNorm, DeltaNumericValue, DeltaSegment, DeltaSnapshot, DeltaStoredValue,
    DeltaTermPosting,
};
use crate::grimoire::{MAX_TERM_BYTES, TermDictionaryError};
#[cfg(unix)]
use crate::keeper::GarbageCollectionOptions;
use crate::keeper::{
    BlueGreenEngine, CURRENT_ENGINE_VERSION, CompactionError, CompactionPolicy, ConcatMergeError,
    CurrentPointer, KeeperError, KeeperSnapshot, KeeperWriter, MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
    Manifest, ManifestFieldStats, ManifestSegment, PublicationReadState, RecoveredSegment,
    publish_current,
};
use crate::query::{
    BooleanClause, BooleanOperator, DefaultQueryParser, Occur, Query, QueryField, QueryNode,
    QueryValue, canonicalize_query,
};
#[cfg(feature = "bench-internals")]
use crate::query::{QueryCapabilityError, QueryExplanation};
use crate::quiver::{
    EncodedPositionList, EncodedPostingList, PositionCodecError, PositionList, Posting,
    PostingList, SnapshotFieldStats, StatsSection, aggregate_field_stats,
};
use crate::schema::{
    Analyzer, DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA, FieldDescriptor, FieldKind, SchemaDescriptor,
};
use crate::scribe::{
    AccumulatorError, ColumnarAccumulator, DEFAULT_ARENA_CHUNK_BYTES, DOC_ORDS_PER_LEASE,
    DeltaFlushInput, FlushDocumentInput, FlushError, FlushMode, FlushSegmentInput,
    FrankensearchTokenizer, IndexedFieldValue, IndexedNumericValue, StoredFieldValue,
    flush_accumulator_with_mode, flush_delta_snapshot,
};
use crate::segment::{EncodedSegment, SectionKind};
use crate::snippet::SnippetConfig;
use crate::stats::SegmentStatsProvider;

mod support;
use support::*;

mod batch_lifecycle_and_concat_merge;
mod compaction_and_fuel;
mod composite_snapshot;
mod delta_seal;
mod fuel_and_tail_budget;
mod gc_and_fanout;
mod labruntime_concurrency;
mod lexical_contract_and_hydration;
mod manifest_and_sync_receipts;
mod parallel_adaptive_and_shards;
mod parallel_budget_and_driver;
mod profile_and_term_stats;
mod profile_pruning_unit;
mod public_reconciliation;
mod publication_durability;
mod root_bound_and_cache;
mod scoring_and_algebra;
mod typed_lowering;

#[cfg(feature = "profile-internals")]
#[test]
fn dynamic_string_terms_cannot_consume_the_exact_term_cache() {
    run_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("dynamic isolation profile directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create dynamic isolation profile writer");
        for ordinal in 0..2 {
            LexicalWrite::index_document(
                &writer,
                &cx,
                &IndexableDocument::new(format!("dynamic-{ordinal}"), "alpha"),
            )
            .await
            .expect("stage dynamic isolation profile document");
            LexicalWrite::commit(&writer, &cx)
                .await
                .expect("publish dynamic isolation profile segment");
        }
        let snapshot = writer
            .search_snapshot()
            .expect("dynamic isolation snapshot is authoritative");
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open dynamic isolation profile reader");
        let query = Query::boolean(
            Vec::from([
                crate::query::BooleanClause::new(
                    Occur::Should,
                    Query::term(
                        Vec::from([crate::query::QueryField::new(CONTENT_FIELD, 1.0)]),
                        "alpha".to_owned(),
                    ),
                ),
                crate::query::BooleanClause::new(
                    Occur::Should,
                    Query::set(
                        CONTENT_FIELD,
                        Vec::from([QueryValue::Str("alpha".to_owned())]),
                    ),
                ),
            ]),
            None,
        );
        let profile = QuillProfileSession::new(
            snapshot.snapshot_epoch(),
            snapshot
                .keeper_snapshot()
                .loaded_manifest()
                .manifest
                .generation,
            2,
            0,
        );
        let result = reader
            .reader
            .execute_ranked_query_inner(
                &cx,
                &query,
                &snapshot,
                10,
                0,
                true,
                false,
                Vec::new(),
                None,
                None,
                #[cfg(feature = "pruning-conformance")]
                None,
                Some(&profile),
            )
            .expect("execute exact-plus-dynamic isolation query");
        assert_eq!(result.hits.len(), 2);
        assert_eq!(profile.snapshot_doc_freq_calls.load(Ordering::Acquire), 3);
        assert_eq!(profile.global_doc_freq_probes.load(Ordering::Acquire), 6);
        assert_eq!(profile.term_dictionary_views.load(Ordering::Acquire), 10);
        assert_eq!(profile.segments_lowered.load(Ordering::Acquire), 2);
    });
}
