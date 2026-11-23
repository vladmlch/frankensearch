//! Keeper lifecycle and durability.
//!
//! This module owns the hand-rolled MANIFEST v1 wire format, Q1 range
//! validation, two-slot recovery, cross-process writer ownership, staged
//! durability repair, serialized publication, and writer-admitted garbage
//! collection. It also owns the structural Q1 concat-merge primitive, the
//! bound-consecutive tier planner, and tombstone-folding compaction.

mod admission;
mod artifact_install;
mod blue_green;
#[allow(clippy::module_name_repetitions)]
mod compaction;
mod concat_merge;
mod constants;
mod errors;
mod gc;
#[allow(clippy::module_name_repetitions)]
mod manifest;
mod manifest_load;
#[allow(clippy::module_name_repetitions)]
mod manifest_publish_choreography;
#[allow(clippy::module_name_repetitions)]
mod manifest_validation;
mod publish;
#[allow(clippy::module_name_repetitions)]
mod recovered_segment;
mod recovery;
mod slot_io;
#[allow(clippy::module_name_repetitions)]
mod snapshot;
mod stats_and_cursor;
#[allow(clippy::module_name_repetitions)]
mod writer;

// ── Public API re-exports (consumed by lib.rs, index.rs, tests, benches) ────
pub use crate::stats::{SegmentStats, SegmentStatsProvider};

pub use blue_green::{
    BlueGreenEngine, CURRENT_FILE_NAME, CURRENT_FORMAT_VERSION, CURRENT_MAGIC, CurrentPointer,
    CurrentPointerError, LexicalLayout, ResolvedCurrent, TANTIVY_INDEX_FORMAT_VERSION,
    inspect_lexical_layout, publish_current, resolve_current,
};
pub use constants::{
    CURRENT_ENGINE_VERSION, DEFAULT_GARBAGE_GRACE, EMPTY_TOMBSTONES,
    MANIFEST_FLAG_BULK_MODE_IN_PROGRESS, MANIFEST_FORMAT_VERSION, MANIFEST_FORMAT_VERSION_V1,
    MANIFEST_KNOWN_FLAGS, MANIFEST_MAGIC, MAX_MANIFEST_BYTES, MAX_MANIFEST_FIELDS,
    MAX_MANIFEST_SEGMENTS, WRITER_LOCK_FORMAT_VERSION, WRITER_LOCK_MAGIC, WRITER_LOCK_RECORD_BYTES,
    pack_engine_version, unpack_engine_version,
};
pub use errors::{
    CompactionError, CompactionPolicy, CompactionReport, ConcatMergeError, KeeperError,
    ManifestCodecError, PublishIntent,
};
pub use gc::{GarbageCollectionOptions, GarbageCollectionReport};
pub use manifest::{
    LoadedManifest, Manifest, ManifestFieldStats, ManifestSegment, ManifestSource, SegmentSizeTier,
    TierMergePlan, TierMergePolicy, TierPolicyError, TombstoneSet, plan_tier_merge,
};
pub use manifest_load::load_manifest_pair;
pub use recovered_segment::{QuarantinedSegment, RecoveredSegment, ResolvedDocumentId};
pub use snapshot::{ExactLexicalSnapshotDescriptor, KeeperSnapshot};
pub use writer::KeeperWriter;

// ── Crate-internal re-exports (consumed by src/index.rs) ────────────────────
#[cfg(feature = "bench-internals")]
pub(crate) use constants::BenchmarkQuillDirectorySyncState;
pub(crate) use errors::LiveDocumentFloor;
pub(crate) use manifest_validation::validate_manifest_successor;
pub(crate) use writer::{
    PublicationAuthorityPhase, PublicationAuthorityState, PublicationReadState,
};

#[cfg(test)]
pub(crate) use gc::collect_writer_garbage_at;
#[cfg(test)]
pub(crate) use manifest_publish_choreography::{
    drive_manifest_publish_to_checkpoint_for_test, fail_manifest_publish_at_checkpoint_for_test,
    manifest_publish_pause_test_serial_guard, pause_manifest_publish_at_checkpoint_for_test,
};
#[cfg(test)]
pub(crate) use publish::PublishCheckpoint;

// ── Test-only internal sibling imports for keeper/tests (`use super::*;`) ─────
#[cfg(test)]
use admission::WriterLockRecord;
#[cfg(all(test, any(unix, windows)))]
use admission::acquire_writer_admission;
#[cfg(all(test, target_os = "linux"))]
use admission::{
    linux_process_start_time, writer_lock_record_names_live_owner,
    writer_pid_start_nonce_from_linux_start_time,
};
#[cfg(all(test, unix))]
use artifact_install::retire_regular_artifact_portable;
#[cfg(test)]
use blue_green::CURRENT_FIXED_BYTES;
#[cfg(test)]
use concat_merge::{recovery_retryable, resolve_concat_representatives};
#[cfg(test)]
use constants::{
    MAX_TOMBSTONE_CHUNKS, TOMBSTONE_ARRAY_MAX_CARDINALITY, TOMBSTONE_BITMAP_MIN_CARDINALITY,
};
#[cfg(test)]
use gc::{
    GarbageCandidate, canonical_segment_name, classify_garbage_candidate,
    collect_abandoned_genesis_garbage_at, collect_writer_garbage_under_lock, safe_direct_child,
};
#[cfg(all(test, unix))]
use gc::{
    RETIREMENT_RECEIPT_BYTES, SegmentRetirementReceipt, UnreachabilityWitness,
    collect_writer_garbage_with_snapshot, open_gc_directory, required_gc_witness_time,
    retirement_receipt_name, segment_unreachability_floor_at, witness_orphaned_segments_at,
    witness_orphaned_segments_with_snapshot, write_unreachability_receipt,
};
#[cfg(all(test, unix))]
use manifest_load::validate_proposed_manifest_segments;
#[cfg(test)]
use manifest_publish_choreography::publish_manifest_choreography;
#[cfg(all(test, unix))]
use manifest_validation::usize_to_u64;
#[cfg(test)]
use manifest_validation::{
    invalid, put_u16, put_u32, tombstone_chunk_count, validate_tombstone_bytes,
};
#[cfg(all(test, unix))]
use publish::GenerationClaimGuard;
#[cfg(test)]
use publish::{ManifestPublisher, publish_pending_segment};
#[cfg(test)]
use recovered_segment::{
    MAX_RANK_PRUNING_CACHE_TERMS, RankPruningCache, bind_validated_term_dictionary,
    term_dictionary_admission_error,
};
#[cfg(all(test, not(windows)))]
use slot_io::sync_directory;
#[cfg(all(test, unix))]
use slot_io::{ManifestSlot, read_manifest_slot};
#[cfg(all(test, unix))]
use snapshot::manifest_matches_proposal;
#[cfg(all(test, unix))]
use stats_and_cursor::detect_live_writer;
#[cfg(test)]
use stats_and_cursor::managed_disk_bytes;
#[cfg(test)]
use writer::PublicationAuthorityTransition;
#[cfg(all(test, unix))]
use writer::WriterProtection;

#[cfg(test)]
mod tests;
