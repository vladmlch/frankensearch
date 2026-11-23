use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::types::DocId;
use serde::Serialize;
use tantivy::collector::TopDocs;
use tantivy::directory::error::{
    DeleteError, LockError, OpenDirectoryError, OpenReadError, OpenWriteError,
};
use tantivy::directory::{
    Directory, DirectoryLock, FileHandle, Lock, MmapDirectory, WatchCallback, WatchHandle, WritePtr,
};
use tantivy::query::TermQuery;
use tantivy::schema::{IndexRecordOption, Schema};
use tantivy::{Index, IndexReader, Term};

use crate::execution::search_guarded;
use crate::schema::{SchemaFields, TOKENIZER_NAME};

/// Process-local issuer for opaque benchmark writer attestations.
///
/// A construction ID proves only that a distinct writer was constructed during
/// this process. It is deliberately not a persisted or cross-process identity.
static NEXT_BENCHMARK_WRITER_CONSTRUCTION_ID: AtomicU64 = AtomicU64::new(1);

pub fn validate_benchmark_writer_threads(writer_threads: usize) -> SearchResult<()> {
    if writer_threads == 0 {
        return Err(SearchError::InvalidConfig {
            field: "tantivy.writer_threads".to_owned(),
            value: writer_threads.to_string(),
            reason: "benchmark writer thread count must be positive".to_owned(),
        });
    }
    Ok(())
}

/// Which Tantivy writer constructor a benchmark index literally called.
///
/// This is a constructor identity, not a request record: `ShippingAuto` means
/// [`Index::writer`] was called and Tantivy chose the pool width itself;
/// `Fixed` means [`Index::writer_with_num_threads`] was called with exactly
/// the recorded width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkWriterMode {
    /// The shipping default path: `Index::writer(heap)`.
    ShippingAuto,
    /// The pinned path: `Index::writer_with_num_threads(threads, heap)`.
    Fixed {
        /// Exact width handed to Tantivy.
        threads: usize,
    },
}

/// Writer-pool width, and whether it is authenticated or merely unknown.
///
/// Tantivy 0.26.1 computes the `ShippingAuto` width inside `Index::writer`
/// from `available_parallelism()` clamped by the heap budget, and exposes no
/// accessor for it — `pub fn num_threads` does not exist anywhere in that
/// crate. Recomputing it here would mean copying `MAX_NUM_THREAD` and
/// `MEMORY_BUDGET_NUM_BYTES_MIN` into this crate and presenting an inference
/// as an observation, which would drift silently on any version bump. The
/// unknown is therefore typed, not filled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMaterializedWidth {
    /// The exact width Tantivy was given and used.
    Authenticated(usize),
    /// No trustworthy width is observable; evidence needing one must fail closed.
    Unobservable {
        /// Why the width cannot be authenticated.
        reason: BenchmarkWidthUnobservableReason,
    },
}

impl BenchmarkMaterializedWidth {
    /// The authenticated width, or `None` when no width is observable.
    #[must_use]
    pub const fn authenticated(self) -> Option<usize> {
        match self {
            Self::Authenticated(threads) => Some(threads),
            Self::Unobservable { .. } => None,
        }
    }
}

/// Why a width is not authenticated.
///
/// Typed and version-neutral on purpose: the receipt states *that* the
/// incumbent selects the width internally and publishes no accessor, which
/// stays true across upgrades, instead of pinning a version string that would
/// silently become a lie on the next bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkWidthUnobservableReason {
    /// The engine chose the pool width inside its own constructor and exposes
    /// no accessor for the value it chose.
    EngineSelectedWidthNotExposed,
}

/// Read the positions setting back out of a live index schema.
///
/// The benchmark reopen path receives a caller's `positions` claim but attaches
/// to a schema that already exists, so the claim is an assertion about someone
/// else's bytes. This reads the indexing option actually recorded on the
/// `content` field, which is the only authority.
pub fn positions_from_live_schema(schema: &Schema) -> Option<bool> {
    let field = schema.get_field("content").ok()?;
    match schema.get_field_entry(field).field_type() {
        tantivy::schema::FieldType::Str(options) => {
            options.get_indexing_options().map(|indexing| {
                indexing.index_option() == tantivy::schema::IndexRecordOption::WithFreqsAndPositions
            })
        }
        _ => None,
    }
}

/// Typed receipt for one benchmark writer construction.
///
/// Oracle identity is exactly what this crate can authenticate in-process: the
/// tokenizer name actually registered, the field names actually present in the
/// live index schema, and the positions flag that built that schema. Tantivy's
/// crate version and source identity are deliberately absent — they are not
/// observable here, and asserting them would be a restatement that can drift.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct BenchmarkWriterReceipt {
    /// Which constructor was called.
    pub mode: BenchmarkWriterMode,
    /// Heap budget handed to that constructor.
    pub writer_heap_bytes: usize,
    /// Authenticated width, or a typed unknown.
    pub materialized_width: BenchmarkMaterializedWidth,
    /// Positions option read back from the live index schema.
    pub positions: bool,
    /// Field names present in the live index schema, in schema order.
    pub schema_fields: Vec<String>,
    /// Tokenizer registered on this index. Owned so the whole receipt can be
    /// bound into an artifact without borrowing this crate's statics.
    pub tokenizer_name: String,
    /// Whether this writer replaced an earlier joined writer.
    pub writer_rearmed: bool,
}

/// One-shot live capability for a benchmark writer that was actually created.
///
/// This is deliberately neither `Clone` nor serializable/deserializable: it is
/// an in-process hand-off from the successful constructor branch to a live
/// benchmark consumer, not a persisted authentication claim. The descriptive
/// [`BenchmarkWriterReceipt`] remains available separately for diagnostics.
#[doc(hidden)]
#[derive(Debug)]
pub struct BenchmarkWriterAttestation {
    receipt: BenchmarkWriterReceipt,
    construction_id: u64,
}

impl BenchmarkWriterAttestation {
    /// Mint an attestation only after its writer constructor has succeeded.
    pub(super) fn mint(receipt: BenchmarkWriterReceipt) -> Self {
        let construction_id = NEXT_BENCHMARK_WRITER_CONSTRUCTION_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("benchmark writer construction IDs exhausted");
        Self {
            receipt,
            construction_id,
        }
    }

    /// Return the authenticated receipt carried by this live capability.
    #[must_use]
    pub const fn receipt(&self) -> &BenchmarkWriterReceipt {
        &self.receipt
    }

    /// Return this process-local writer-construction identity.
    #[must_use]
    pub const fn construction_id(&self) -> u64 {
        self.construction_id
    }
}

impl BenchmarkWriterReceipt {
    /// Seed a receipt from the constructor branch that actually ran.
    ///
    /// `mode` is produced by the branch itself rather than passed in beside it,
    /// so the recorded mode and heap cannot drift from the `Index::writer` or
    /// `writer_with_num_threads` call that really happened. `positions` is read
    /// back from the live schema, never from a caller's claim.
    pub(super) fn seed(
        mode: BenchmarkWriterMode,
        writer_heap_bytes: usize,
        index: &Index,
    ) -> SearchResult<Self> {
        let materialized_width = match mode {
            BenchmarkWriterMode::ShippingAuto => BenchmarkMaterializedWidth::Unobservable {
                reason: BenchmarkWidthUnobservableReason::EngineSelectedWidthNotExposed,
            },
            BenchmarkWriterMode::Fixed { threads } => {
                BenchmarkMaterializedWidth::Authenticated(threads)
            }
        };
        let schema = index.schema();
        // Absent positions evidence is a missing fact, not a false one. A
        // defaulted `false` would put an unearned claim into every artifact
        // this receipt is later bound into.
        let positions =
            positions_from_live_schema(&schema).ok_or_else(|| SearchError::InvalidConfig {
                field: "tantivy.positions".to_owned(),
                value: "unavailable".to_owned(),
                reason: "live index schema exposes no indexed content field, so the positions \
                         option cannot be authenticated"
                    .to_owned(),
            })?;
        Ok(Self {
            mode,
            writer_heap_bytes,
            materialized_width,
            positions,
            schema_fields: schema
                .fields()
                .map(|(_, entry)| entry.name().to_owned())
                .collect(),
            tokenizer_name: TOKENIZER_NAME.to_owned(),
            writer_rearmed: false,
        })
    }
}

/// Reject a benchmark caller whose `positions` claim disagrees with the schema
/// it is actually attaching to.
///
/// Creating an index builds the schema from this flag, but reopening one only
/// asserts about bytes that already exist. A silent disagreement would put a
/// false positions value into every downstream receipt.
pub fn reject_positions_disagreement(index: &Index, claimed: bool) -> SearchResult<()> {
    let observed =
        positions_from_live_schema(&index.schema()).ok_or_else(|| SearchError::InvalidConfig {
            field: "tantivy.positions".to_owned(),
            value: claimed.to_string(),
            reason: "live index schema exposes no indexed content field to authenticate positions"
                .to_owned(),
        })?;
    if observed == claimed {
        return Ok(());
    }
    Err(SearchError::InvalidConfig {
        field: "tantivy.positions".to_owned(),
        value: claimed.to_string(),
        reason: format!("live index schema records positions = {observed}"),
    })
}

/// Counted receipt for a benchmark-only Tantivy writer lifecycle fence.
///
/// `wait_merging_threads` consumes Tantivy's writer after joining its indexing
/// and segment-updater workers. Warm/update fixtures may rearm the same index
/// with the pinned writer configuration, while terminal bulk measurements
/// deliberately stop after the join so they do not construct and immediately
/// discard an unused replacement writer.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BenchmarkWriterJoinReceipt {
    /// Searchable segment count immediately before joining the writer.
    pub searchable_segments_before: usize,
    /// Searchable segment count after every background worker has joined.
    pub searchable_segments_after: usize,
    /// Time spent inside Tantivy's worker/merge join, excluding writer rearm.
    pub join_elapsed_ns: u64,
    /// Whether a replacement writer was successfully constructed after joining.
    pub writer_rearmed: bool,
}

/// Read-only Tantivy search handle retained across a terminal writer join.
///
/// This deliberately owns no writer.  A terminal benchmark obtains it only by
/// consuming [`crate::TantivyIndex`] through
/// [`crate::TantivyIndex::benchmark_join_workers_retaining_reader`], which joins the
/// real Tantivy workers first and leaves this handle alive for the required
/// post-join searchable-tail query.
#[doc(hidden)]
pub struct BenchmarkRetainedTantivyReader {
    pub(super) fields: SchemaFields,
    pub(super) reader: IndexReader,
}

impl BenchmarkRetainedTantivyReader {
    /// Run an exact stored-ID query through the reader retained after a writer
    /// join.
    ///
    /// This is intentionally a narrow benchmark seam rather than a general
    /// second lexical API: it proves that the terminal prepared-tail document
    /// is searchable *after* workers have quiesced, without creating or
    /// rearming a writer.
    ///
    /// # Errors
    ///
    /// Returns a typed Tantivy search failure.
    pub fn benchmark_search_exact_id(&self, document_id: &str) -> SearchResult<Vec<DocId>> {
        let query = TermQuery::new(
            Term::from_field_text(self.fields.id, document_id),
            IndexRecordOption::Basic,
        );
        let searcher = self.reader.searcher();
        let top_docs = search_guarded(&searcher, &query, &TopDocs::with_limit(2).order_by_score())?;

        // The exact `id` term itself authenticates membership.  Do not add a
        // stored-document hydration that Quill's matching IDHASH probe does
        // not pay; terminal proof cost is part of the timed lifecycle.
        Ok(top_docs
            .into_iter()
            .map(|_| document_id.to_owned().into())
            .collect())
    }
}

/// High bit marks a nonce as armed but not yet observed by the directory.
const BENCHMARK_DIRECTORY_SYNC_ARMED: u64 = 1 << 63;

/// Terminal state for an armed nonce that saw more than one directory sync.
const BENCHMARK_DIRECTORY_SYNC_AMBIGUOUS: u64 = u64::MAX;

/// Shared state between the benchmark-only directory wrapper and its caller.
///
/// The arm is installed before the timed interval. A successful real
/// `Directory::sync_directory` changes `ARMED(nonce)` into `OBSERVED(nonce)`;
/// a second successful call changes it into the ambiguous terminal state. The
/// state intentionally records neither a caller-supplied treatment label nor a
/// timestamp: Tantivy's real directory call is the observation.
#[derive(Debug)]
pub struct BenchmarkDirectorySyncState {
    next_nonce: AtomicU64,
    session: AtomicU64,
}

impl BenchmarkDirectorySyncState {
    pub fn new() -> Self {
        Self {
            next_nonce: AtomicU64::new(1),
            session: AtomicU64::new(0),
        }
    }

    pub fn arm(&self) -> SearchResult<u64> {
        let nonce = self
            .next_nonce
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current < BENCHMARK_DIRECTORY_SYNC_ARMED - 1).then_some(current + 1)
            })
            .map_err(|_| SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync_nonce".to_owned(),
                value: "exhausted".to_owned(),
                reason: "benchmark directory-sync session nonce space exhausted".to_owned(),
            })?;
        self.session
            .compare_exchange(
                0,
                nonce | BENCHMARK_DIRECTORY_SYNC_ARMED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| SearchError::InvalidConfig {
                field: "tantivy.qg4.directory_sync_session".to_owned(),
                value: "already_armed".to_owned(),
                reason: "a previous benchmark directory-sync observation is still active"
                    .to_owned(),
            })?;
        Ok(nonce)
    }

    fn observe_successful_sync(&self) {
        loop {
            let current = self.session.load(Ordering::Acquire);
            if current == 0 || current == BENCHMARK_DIRECTORY_SYNC_AMBIGUOUS {
                return;
            }
            let next = if current & BENCHMARK_DIRECTORY_SYNC_ARMED != 0 {
                current & !BENCHMARK_DIRECTORY_SYNC_ARMED
            } else {
                BENCHMARK_DIRECTORY_SYNC_AMBIGUOUS
            };
            if self
                .session
                .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }
}

/// Transparent `MmapDirectory` observer used only by the QG-4 benchmark path.
///
/// It delegates all Tantivy directory behavior. Only a successful delegated
/// directory sync updates the shared observation state.
#[derive(Clone, Debug)]
pub struct ObservedMmapDirectory {
    inner: MmapDirectory,
    state: Arc<BenchmarkDirectorySyncState>,
}

impl ObservedMmapDirectory {
    pub fn open(
        path: &Path,
        state: Arc<BenchmarkDirectorySyncState>,
    ) -> Result<Self, OpenDirectoryError> {
        Ok(Self {
            inner: MmapDirectory::open(path)?,
            state,
        })
    }
}

impl Directory for ObservedMmapDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.inner.get_file_handle(path)
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.inner.delete(path)
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.inner.exists(path)
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        self.inner.open_write(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.inner.atomic_read(path)
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        self.inner.atomic_write(path, data)
    }

    fn sync_directory(&self) -> std::io::Result<()> {
        let result = self.inner.sync_directory();
        if result.is_ok() {
            self.state.observe_successful_sync();
        }
        result
    }

    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        self.inner.acquire_lock(lock)
    }

    fn watch(&self, watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.inner.watch(watch_callback)
    }
}

/// Live arm for one controlled QG-4 Tantivy commit observation.
///
/// Construct this immediately before the timed benchmark commit and call
/// [`Self::finish`] after the timer stops. It is neither serializable nor
/// cloneable, so a caller cannot replay it as a declarative receipt.
#[doc(hidden)]
#[derive(Debug)]
pub struct BenchmarkTantivyDirectorySyncArm {
    pub(super) state: Arc<BenchmarkDirectorySyncState>,
    pub(super) root: PathBuf,
    pub(super) nonce: u64,
    pub(super) commit_opstamp: Option<u64>,
    pub(super) finished: bool,
}

/// Engine-observed result of one controlled QG-4 Tantivy commit.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkTantivyDirectorySyncReceipt {
    /// On-disk root whose wrapped Tantivy directory emitted the observation.
    pub root: PathBuf,
    /// Opaque session identity minted before the timed commit.
    pub session_nonce: u64,
    /// Opstamp returned by the same blocking Tantivy commit.
    pub commit_opstamp: u64,
    /// Successful directory-sync calls observed in the armed session.
    pub observed_sync_count: u8,
}

impl BenchmarkTantivyDirectorySyncArm {
    /// Finalize the receipt outside the timed interval.
    ///
    /// # Errors
    ///
    /// Returns an error when the timed commit did not retain its opstamp or
    /// the armed directory-sync session observed anything other than exactly
    /// one successful sync for this arm.
    pub fn finish(&mut self) -> SearchResult<BenchmarkTantivyDirectorySyncReceipt> {
        let observed = self.state.session.swap(0, Ordering::AcqRel);
        self.finished = true;
        if observed == self.nonce {
            let commit_opstamp =
                self.commit_opstamp
                    .take()
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "tantivy.qg4.directory_sync_receipt".to_owned(),
                        value: "missing_commit_opstamp".to_owned(),
                        reason: "the armed QG-4 commit did not stamp Tantivy's returned opstamp"
                            .to_owned(),
                    })?;
            return Ok(BenchmarkTantivyDirectorySyncReceipt {
                root: self.root.clone(),
                session_nonce: self.nonce,
                commit_opstamp,
                observed_sync_count: 1,
            });
        }
        let (value, reason) = if observed == (self.nonce | BENCHMARK_DIRECTORY_SYNC_ARMED) {
            (
                "zero_syncs",
                "the timed commit returned without an observed successful directory sync",
            )
        } else if observed == BENCHMARK_DIRECTORY_SYNC_AMBIGUOUS {
            (
                "multiple_syncs",
                "more than one successful directory sync occurred while the session was armed",
            )
        } else {
            (
                "foreign_or_disarmed_session",
                "the armed directory-sync session was replaced or cleared before finalization",
            )
        };
        Err(SearchError::InvalidConfig {
            field: "tantivy.qg4.directory_sync_receipt".to_owned(),
            value: value.to_owned(),
            reason: reason.to_owned(),
        })
    }
}

impl Drop for BenchmarkTantivyDirectorySyncArm {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let armed = self.nonce | BENCHMARK_DIRECTORY_SYNC_ARMED;
        let _ = self
            .state
            .session
            .compare_exchange(armed, 0, Ordering::AcqRel, Ordering::Acquire);
        let _ =
            self.state
                .session
                .compare_exchange(self.nonce, 0, Ordering::AcqRel, Ordering::Acquire);
        let _ = self.state.session.compare_exchange(
            BENCHMARK_DIRECTORY_SYNC_AMBIGUOUS,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}
