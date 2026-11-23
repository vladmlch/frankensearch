use std::collections::BTreeMap;
use std::path::PathBuf;
#[cfg(feature = "bench-internals")]
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{OnceLock, Weak};
use std::time::Duration;

use asupersync::sync::Mutex;
#[cfg(feature = "bench-internals")]
use frankensearch_core::{SearchError, SearchResult};

/// Eight-byte MANIFEST magic, including its trailing NUL.
pub const MANIFEST_MAGIC: [u8; 8] = *b"FSLXMAN\0";
/// Current durable MANIFEST format version written by this crate.
///
/// v2 adds the generation-level `last_publish_unix_s` freshness witness
/// (registry §6.2). Readers accept v1 images and decode the field as `0`
/// (unknown); v1 readers reject v2 images as an unsupported version.
pub const MANIFEST_FORMAT_VERSION: u32 = 2;
/// First durable MANIFEST format version, still accepted on read.
pub const MANIFEST_FORMAT_VERSION_V1: u32 = 1;
/// Maximum MANIFEST accepted by an eager reader.
pub const MAX_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
/// Maximum segment records admitted by one MANIFEST.
///
/// This is deliberately below the byte-format's theoretical limit so hostile
/// inputs cannot amplify a bounded file into millions of heap allocations.
pub const MAX_MANIFEST_SEGMENTS: usize = 65_536;
/// Maximum stats rows. `field_ord` is a `u16`, so no larger set is useful.
pub const MAX_MANIFEST_FIELDS: usize = 65_536;
/// Known MANIFEST flag: a bulk build is in progress.
pub const MANIFEST_FLAG_BULK_MODE_IN_PROGRESS: u32 = 1;
/// All flag bits understood by this reader.
pub const MANIFEST_KNOWN_FLAGS: u32 = MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
/// Canonical empty roaring-lite tombstone set (`chunk_count = 0`).
pub const EMPTY_TOMBSTONES: [u8; 4] = [0, 0, 0, 0];
/// Minimum age of an unreachable Quill artifact before writer GC may remove it.
pub const DEFAULT_GARBAGE_GRACE: Duration = Duration::from_secs(300);
/// Eight-byte writer-admission magic, including its trailing NUL.
pub const WRITER_LOCK_MAGIC: [u8; 8] = *b"FSLXLCK\0";
/// Current durable writer-lock record version.
pub const WRITER_LOCK_FORMAT_VERSION: u32 = 1;
/// Exact byte length of a v1 writer-lock record, including its CRC32.
pub const WRITER_LOCK_RECORD_BYTES: usize = 36;

/// Current Quill crate version in the MANIFEST packed-semver representation.
///
/// The build-time assertion in this module's tests intentionally forces this
/// value to change when `Cargo.toml` changes.
pub const CURRENT_ENGINE_VERSION: u32 = pack_engine_version(0, 4, 0);

pub(in crate::keeper) const MANIFEST_MIN_BYTES: usize = 8 + 4 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4;
/// v2 images carry the additional `last_publish_unix_s` word after `flags`.
pub(in crate::keeper) const MANIFEST_V2_MIN_BYTES: usize = MANIFEST_MIN_BYTES + 8;
pub(in crate::keeper) const SEGMENT_FIXED_BYTES: usize = 8 + 8 + 8 + 8 + 8 + 8 + 4;
pub(in crate::keeper) const FIELD_STATS_BYTES: usize = 2 + 8 + 4;
pub(in crate::keeper) const MAX_DOCID_EXCLUSIVE: u64 = 4_294_967_296;
pub(in crate::keeper) const TOMBSTONE_ARRAY_MAX_CARDINALITY: u16 = 4_096;
pub(in crate::keeper) const TOMBSTONE_BITMAP_MIN_CARDINALITY: u64 = 3_584;
pub(in crate::keeper) const TOMBSTONE_BITMAP_BYTES: usize = 8_192;
pub(in crate::keeper) const MAX_TOMBSTONE_CHUNKS: usize = 65_536;

// Each canonical index directory has one in-process serializer. Public
// mutations also require retained cross-process `KeeperWriter` admission, so
// distinct directories must not convoy behind one process-global mutex. The
// owned guard moves into blocking work; a cancelled async caller therefore
// cannot release same-directory serialization before that work exits.
pub(in crate::keeper) static PUBLISH_LOCKS: OnceLock<
    std::sync::Mutex<BTreeMap<PathBuf, Weak<Mutex<()>>>>,
> = OnceLock::new();

/// High bit marks a benchmark directory-sync nonce as armed but unobserved.
#[cfg(feature = "bench-internals")]
const BENCHMARK_QG4_DIRECTORY_SYNC_ARMED: u64 = 1 << 63;

/// Terminal armed-session state after more than one MANIFEST directory sync.
#[cfg(feature = "bench-internals")]
const BENCHMARK_QG4_DIRECTORY_SYNC_AMBIGUOUS: u64 = u64::MAX;

#[cfg(feature = "bench-internals")]
const BENCHMARK_QG4_COMMIT_PENDING: u8 = 0;
#[cfg(feature = "bench-internals")]
const BENCHMARK_QG4_COMMIT_SUCCEEDED: u8 = 1;
#[cfg(feature = "bench-internals")]
const BENCHMARK_QG4_COMMIT_FAILED: u8 = 2;

/// Shared benchmark-only state for one live Quill MANIFEST directory-sync arm.
///
/// Only the post-`sync_directory` MANIFEST checkpoint calls
/// [`Self::observe_successful_manifest_sync`]. The public benchmark commit
/// wrapper records whether the encompassing commit returned successfully.
#[cfg(feature = "bench-internals")]
#[derive(Debug)]
pub struct BenchmarkQuillDirectorySyncState {
    next_nonce: AtomicU64,
    session: AtomicU64,
    root: PathBuf,
    observed_generation: AtomicU64,
    commit_outcome: AtomicU8,
}

#[cfg(feature = "bench-internals")]
impl BenchmarkQuillDirectorySyncState {
    pub fn new(root: PathBuf) -> Self {
        Self {
            next_nonce: AtomicU64::new(1),
            session: AtomicU64::new(0),
            root,
            observed_generation: AtomicU64::new(0),
            commit_outcome: AtomicU8::new(BENCHMARK_QG4_COMMIT_PENDING),
        }
    }

    pub fn arm(&self) -> SearchResult<u64> {
        let nonce = self
            .next_nonce
            .try_update(
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
                |current| (current < BENCHMARK_QG4_DIRECTORY_SYNC_ARMED - 1).then_some(current + 1),
            )
            .map_err(|_| SearchError::InvalidConfig {
                field: "quill.qg4.directory_sync_nonce".to_owned(),
                value: "exhausted".to_owned(),
                reason: "benchmark directory-sync session nonce space exhausted".to_owned(),
            })?;
        self.session
            .compare_exchange(
                0,
                nonce | BENCHMARK_QG4_DIRECTORY_SYNC_ARMED,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            )
            .map_err(|_| SearchError::InvalidConfig {
                field: "quill.qg4.directory_sync_session".to_owned(),
                value: "already_armed".to_owned(),
                reason: "a previous benchmark directory-sync observation is still active"
                    .to_owned(),
            })?;
        self.observed_generation.store(0, AtomicOrdering::Release);
        self.commit_outcome
            .store(BENCHMARK_QG4_COMMIT_PENDING, AtomicOrdering::Release);
        Ok(nonce)
    }

    /// Record only the successful directory sync that follows MANIFEST install.
    pub fn observe_successful_manifest_sync(&self, generation: u64) {
        loop {
            let current = self.session.load(AtomicOrdering::Acquire);
            if current == 0 || current == BENCHMARK_QG4_DIRECTORY_SYNC_AMBIGUOUS {
                return;
            }
            if current & BENCHMARK_QG4_DIRECTORY_SYNC_ARMED != 0 {
                self.observed_generation
                    .store(generation, AtomicOrdering::Release);
                if self
                    .session
                    .compare_exchange(
                        current,
                        current & !BENCHMARK_QG4_DIRECTORY_SYNC_ARMED,
                        AtomicOrdering::AcqRel,
                        AtomicOrdering::Acquire,
                    )
                    .is_ok()
                {
                    return;
                }
                continue;
            }
            if self
                .session
                .compare_exchange(
                    current,
                    BENCHMARK_QG4_DIRECTORY_SYNC_AMBIGUOUS,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                )
                .is_ok()
            {
                return;
            }
        }
    }

    pub fn record_commit_success(&self, nonce: u64) -> SearchResult<()> {
        let session = self.session.load(AtomicOrdering::Acquire);
        if session != nonce
            && session != (nonce | BENCHMARK_QG4_DIRECTORY_SYNC_ARMED)
            && session != BENCHMARK_QG4_DIRECTORY_SYNC_AMBIGUOUS
        {
            return Err(SearchError::InvalidConfig {
                field: "quill.qg4.directory_sync_receipt".to_owned(),
                value: "foreign_or_disarmed_session".to_owned(),
                reason:
                    "the QG-4 directory-sync arm was replaced or cleared before commit completion"
                        .to_owned(),
            });
        }
        self.commit_outcome
            .compare_exchange(
                BENCHMARK_QG4_COMMIT_PENDING,
                BENCHMARK_QG4_COMMIT_SUCCEEDED,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            )
            .map_err(|_| SearchError::InvalidConfig {
                field: "quill.qg4.directory_sync_receipt".to_owned(),
                value: "commit_reused".to_owned(),
                reason: "a QG-4 directory-sync arm may complete only one benchmark commit"
                    .to_owned(),
            })
            .map(|_| ())
    }

    pub fn record_commit_failure(&self, nonce: u64) {
        let session = self.session.load(AtomicOrdering::Acquire);
        if session == nonce
            || session == (nonce | BENCHMARK_QG4_DIRECTORY_SYNC_ARMED)
            || session == BENCHMARK_QG4_DIRECTORY_SYNC_AMBIGUOUS
        {
            self.commit_outcome
                .store(BENCHMARK_QG4_COMMIT_FAILED, AtomicOrdering::Release);
        }
    }

    pub fn finish(&self, nonce: u64) -> SearchResult<(PathBuf, u64)> {
        let observed = self.session.swap(0, AtomicOrdering::AcqRel);
        if observed == nonce {
            match self.commit_outcome.load(AtomicOrdering::Acquire) {
                BENCHMARK_QG4_COMMIT_SUCCEEDED => {
                    let generation = self.observed_generation.load(AtomicOrdering::Acquire);
                    if generation != 0 {
                        return Ok((self.root.clone(), generation));
                    }
                    Err(SearchError::InvalidConfig {
                        field: "quill.qg4.directory_sync_receipt".to_owned(),
                        value: "missing_generation".to_owned(),
                        reason: "the observed QG-4 directory sync did not stamp its MANIFEST generation"
                            .to_owned(),
                    })
                }
                BENCHMARK_QG4_COMMIT_FAILED => Err(SearchError::InvalidConfig {
                    field: "quill.qg4.directory_sync_receipt".to_owned(),
                    value: "commit_failed_after_sync".to_owned(),
                    reason: "the armed QG-4 commit failed after the observed MANIFEST directory sync"
                        .to_owned(),
                }),
                _ => Err(SearchError::InvalidConfig {
                    field: "quill.qg4.directory_sync_receipt".to_owned(),
                    value: "commit_not_completed".to_owned(),
                    reason: "the armed QG-4 directory sync was not completed through the benchmark commit API"
                        .to_owned(),
                }),
            }
        } else {
            let (value, reason) = if observed == (nonce | BENCHMARK_QG4_DIRECTORY_SYNC_ARMED) {
                (
                    "zero_syncs",
                    "the timed commit returned without an observed MANIFEST directory sync",
                )
            } else if observed == BENCHMARK_QG4_DIRECTORY_SYNC_AMBIGUOUS {
                (
                    "multiple_syncs",
                    "more than one MANIFEST directory sync occurred while the session was armed",
                )
            } else {
                (
                    "foreign_or_disarmed_session",
                    "the QG-4 directory-sync session was replaced or cleared before finalization",
                )
            };
            Err(SearchError::InvalidConfig {
                field: "quill.qg4.directory_sync_receipt".to_owned(),
                value: value.to_owned(),
                reason: reason.to_owned(),
            })
        }
    }

    pub fn disarm(&self, nonce: u64) {
        let armed = nonce | BENCHMARK_QG4_DIRECTORY_SYNC_ARMED;
        let _ = self.session.compare_exchange(
            armed,
            0,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        );
        let _ = self.session.compare_exchange(
            nonce,
            0,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        );
        let _ = self.session.compare_exchange(
            BENCHMARK_QG4_DIRECTORY_SYNC_AMBIGUOUS,
            0,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        );
    }
}

/// Pack semantic-version components into the durable `engine_version` word.
///
/// Layout: `major:u8 << 24 | minor:u8 << 16 | patch:u16`. Prerelease and
/// build metadata are intentionally not durable.
#[must_use]
pub const fn pack_engine_version(major: u8, minor: u8, patch: u16) -> u32 {
    (major as u32) << 24 | (minor as u32) << 16 | patch as u32
}

/// Decode a packed `engine_version` into `(major, minor, patch)`.
#[must_use]
pub const fn unpack_engine_version(version: u32) -> (u8, u8, u16) {
    (
        version.to_be_bytes()[0],
        version.to_be_bytes()[1],
        u16::from_be_bytes([version.to_be_bytes()[2], version.to_be_bytes()[3]]),
    )
}
