use std::cell::Cell;
use std::error::Error;
use std::fmt;
#[cfg(feature = "fts5")]
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use crate::metrics::{StorageMetrics, StorageMetricsSnapshot};
use crate::schema;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageConfig {
    pub db_path: PathBuf,
    pub wal_mode: bool,
    pub busy_timeout_ms: u64,
    pub cache_size_pages: i32,
    pub concurrent_transactions: bool,
}

impl StorageConfig {
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            db_path: PathBuf::from(":memory:"),
            ..Self::default()
        }
    }

    fn begin_sql(&self) -> &'static str {
        if self.concurrent_transactions {
            "BEGIN IMMEDIATE;"
        } else {
            "BEGIN;"
        }
    }

    fn transaction_behavior(&self) -> rusqlite::TransactionBehavior {
        if self.concurrent_transactions {
            rusqlite::TransactionBehavior::Immediate
        } else {
            rusqlite::TransactionBehavior::Deferred
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("storage.sqlite3"),
            wal_mode: true,
            busy_timeout_ms: 5_000,
            cache_size_pages: 2_000,
            concurrent_transactions: true,
        }
    }
}

pub struct Storage {
    conn: Mutex<rusqlite::Connection>,
    // Thread tag of the thread currently holding `conn`'s MutexGuard, or 0
    // when free. Lets `acquire_connection` fail a same-thread re-entry (a
    // nested `transaction`/`connection` call inside a running closure)
    // immediately instead of deadlocking on the mutex.
    lock_owner: AtomicU64,
    config: StorageConfig,
    metrics: StorageMetrics,
}

/// Exclusive access to the underlying `rusqlite` connection.
///
/// Holds the `Storage` mutex and dereferences to `&rusqlite::Connection` /
/// `&mut rusqlite::Connection`. Nested write sections inside a
/// [`rusqlite::Transaction`] must use `Transaction::savepoint()`; never call
/// `unchecked_transaction` on a connection that is already inside a
/// transaction.
pub struct StorageConnection<'a> {
    guard: MutexGuard<'a, rusqlite::Connection>,
    lock_owner: &'a AtomicU64,
}

impl Deref for StorageConnection<'_> {
    type Target = rusqlite::Connection;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl DerefMut for StorageConnection<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl fmt::Debug for StorageConnection<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StorageConnection").finish_non_exhaustive()
    }
}

impl Drop for StorageConnection<'_> {
    fn drop(&mut self) {
        // Clear the owner tag while the mutex is still held so another
        // thread only observes tag 0 after it can actually acquire.
        self.lock_owner.store(0, Ordering::Release);
    }
}

static FILE_BOOTSTRAP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Opaque per-thread tag for re-entrant lock detection. `ThreadId` cannot be
// converted to an integer on stable Rust, so each thread mints a unique tag
// from a process-wide counter on first use.
static NEXT_STORAGE_THREAD_TAG: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static STORAGE_THREAD_TAG: Cell<u64> = const { Cell::new(0) };
}

fn storage_thread_tag() -> u64 {
    STORAGE_THREAD_TAG.with(|tag| {
        let current = tag.get();
        if current != 0 {
            return current;
        }
        let minted = NEXT_STORAGE_THREAD_TAG.fetch_add(1, Ordering::Relaxed);
        tag.set(minted);
        minted
    })
}

// SQLite already serializes writers on `SQLITE_BUSY`, but a just-used peer
// connection can still hand back Busy while it releases its locks. Keep this
// outer handoff short and bounded: seven waits total 113 ms, enough to cross a
// normal scheduler handoff without turning persistent lock contention into a
// hang.
const CONNECTION_OPEN_MAX_ATTEMPTS: u32 = 8;
const CONNECTION_OPEN_RETRY_BASE_DELAY_MS: u64 = 1;
const CONNECTION_OPEN_RETRY_MAX_DELAY_MS: u64 = 50;

fn is_retryable_sqlite_error(error: &rusqlite::Error) -> bool {
    // Busy and Locked are the transient contention family. SchemaChanged
    // needs the next execute to re-prepare; retryable here because every
    // attempt starts a new statement.
    match error {
        rusqlite::Error::SqliteFailure(inner, _) => matches!(
            inner.code,
            rusqlite::ErrorCode::DatabaseBusy
                | rusqlite::ErrorCode::DatabaseLocked
                | rusqlite::ErrorCode::SchemaChanged
        ),
        _ => false,
    }
}

pub(crate) fn is_retryable_storage_error(error: &SearchError) -> bool {
    let mut current: Option<&(dyn Error + 'static)> = Some(error);
    while let Some(err) = current {
        if let Some(sqlite_error) = err.downcast_ref::<rusqlite::Error>() {
            return is_retryable_sqlite_error(sqlite_error);
        }
        current = err.source();
    }
    false
}

pub(crate) fn retry_transient_storage<T>(
    mut op: impl FnMut() -> SearchResult<T>,
    context: &'static str,
) -> SearchResult<T> {
    let mut attempt = 1;
    loop {
        match op() {
            Ok(value) => return Ok(value),
            Err(error)
                if is_retryable_storage_error(&error) && attempt < CONNECTION_OPEN_MAX_ATTEMPTS =>
            {
                let delay = connection_open_retry_delay(attempt);
                tracing::debug!(
                    target: "frankensearch.storage",
                    context,
                    next_attempt = attempt.saturating_add(1),
                    max_attempts = CONNECTION_OPEN_MAX_ATTEMPTS,
                    ?delay,
                    error = %error,
                    "retrying transient storage contention"
                );
                std::thread::sleep(delay);
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(feature = "fts5")]
pub(crate) async fn retry_transient_storage_async<T, F, Fut>(
    cx: &Cx,
    mut op: F,
    context: &'static str,
) -> SearchResult<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = SearchResult<T>>,
{
    let mut attempt = 1;
    loop {
        checkpoint_cx(cx, context)?;
        match op().await {
            Ok(value) => return Ok(value),
            Err(error)
                if is_retryable_storage_error(&error) && attempt < CONNECTION_OPEN_MAX_ATTEMPTS =>
            {
                let delay = connection_open_retry_delay(attempt);
                tracing::debug!(
                    target: "frankensearch.storage",
                    context,
                    next_attempt = attempt.saturating_add(1),
                    max_attempts = CONNECTION_OPEN_MAX_ATTEMPTS,
                    ?delay,
                    error = %error,
                    "retrying transient storage contention"
                );
                asupersync::time::sleep(cx.now(), delay).await;
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

/// Observe an asupersync cancellation checkpoint and map it to
/// [`SearchError::Cancelled`]. Async entry points receive `cx` from their
/// caller; synchronous entry points observe the ambient context via
/// [`checkpoint_ambient`].
pub(crate) fn checkpoint_cx(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    if let Err(error) = cx.checkpoint() {
        return Err(SearchError::Cancelled {
            phase: phase.to_owned(),
            reason: cx
                .cancel_reason()
                .map_or_else(|| error.to_string(), |reason| reason.to_string()),
        });
    }
    Ok(())
}

/// Checkpoint the ambient `asupersync` context when one is installed on this
/// thread (`Cx::current()`).
///
/// The storage entry points are synchronous and keep their pre-migration
/// signatures — callers such as `PersistentJobQueue::enqueue` have no `&Cx`
/// parameter, matching `frankensearch_durability`'s `codec_cx` convention for
/// synchronous APIs. Inside a runtime task `Cx::current()` returns the
/// caller's context, so `cx.checkpoint()` still observes cancellation before
/// the connection mutex is acquired or work runs; outside a task the check is
/// a no-op. Callers holding an explicit `&Cx` in a scope where it is not
/// ambient-installed should checkpoint it themselves before calling.
fn checkpoint_ambient(phase: &'static str) -> SearchResult<()> {
    Cx::current().map_or_else(|| Ok(()), |cx| checkpoint_cx(&cx, phase))
}

pub(crate) fn unretryable_rollback_error(
    original: &SearchError,
    rollback_err: &rusqlite::Error,
) -> SearchError {
    // The connection may still be inside a transaction. Do not let the
    // original Busy error (or a transient rollback error) trip another BEGIN
    // on a dirty connection.
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(std::io::Error::other(format!(
            "rollback failed ({rollback_err}); original error was: {original}"
        ))),
    }
}

fn connection_open_retry_delay(failed_attempt: u32) -> Duration {
    let growth = failed_attempt.saturating_sub(1).min(6);
    let delay_ms =
        (CONNECTION_OPEN_RETRY_BASE_DELAY_MS << growth).min(CONNECTION_OPEN_RETRY_MAX_DELAY_MS);
    Duration::from_millis(delay_ms)
}

fn retry_connection_open<T>(
    mut open: impl FnMut() -> Result<T, rusqlite::Error>,
    mut on_retry: impl FnMut(u32, Duration, &rusqlite::Error),
) -> Result<T, rusqlite::Error> {
    let mut attempt = 1;
    loop {
        match open() {
            Ok(connection) => return Ok(connection),
            Err(error)
                if is_retryable_sqlite_error(&error) && attempt < CONNECTION_OPEN_MAX_ATTEMPTS =>
            {
                let delay = connection_open_retry_delay(attempt);
                on_retry(attempt.saturating_add(1), delay, &error);
                std::thread::sleep(delay);
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

fn open_connection_with_retry(
    path: &std::path::Path,
) -> Result<rusqlite::Connection, rusqlite::Error> {
    let is_memory = path.as_os_str() == ":memory:";
    retry_connection_open(
        || {
            if is_memory {
                rusqlite::Connection::open_in_memory()
            } else {
                rusqlite::Connection::open(path)
            }
        },
        |next_attempt, delay, error| {
            tracing::debug!(
                target: "frankensearch.storage",
                path = %path.display(),
                next_attempt,
                max_attempts = CONNECTION_OPEN_MAX_ATTEMPTS,
                ?delay,
                error = %error,
                "retrying transient storage connection-open contention"
            );
        },
    )
}

#[derive(Debug)]
struct StorageStageError<E> {
    stage: &'static str,
    source: E,
}

impl<E> fmt::Display for StorageStageError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.stage, self.source)
    }
}

impl<E> Error for StorageStageError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("path", &self.config.db_path)
            .field("wal_mode", &self.config.wal_mode)
            .field("busy_timeout_ms", &self.config.busy_timeout_ms)
            .finish_non_exhaustive()
    }
}

impl Storage {
    pub fn open(config: StorageConfig) -> SearchResult<Self> {
        tracing::debug!(
            target: "frankensearch.storage",
            path = %config.db_path.display(),
            wal_mode = config.wal_mode,
            busy_timeout_ms = config.busy_timeout_ms,
            cache_size_pages = config.cache_size_pages,
            concurrent_transactions = config.concurrent_transactions,
            "opening storage connection"
        );

        let file_bootstrap_guard = if config.db_path.as_os_str() == ":memory:" {
            None
        } else {
            Some(FILE_BOOTSTRAP_LOCK.lock().map_err(|_| {
                map_storage_error_at(
                    "file bootstrap lock",
                    std::io::Error::other("lock poisoned"),
                )
            })?)
        };

        let conn = open_connection_with_retry(&config.db_path)
            .map_err(|error| map_storage_error_at("connection open", error))?;

        let storage = Self {
            conn: Mutex::new(conn),
            lock_owner: AtomicU64::new(0),
            config,
            metrics: StorageMetrics::default(),
        };

        storage.metrics.record_open();
        let version = {
            let mut conn = storage.acquire_connection()?;
            storage.apply_connection_pragmas(&conn)?;
            // `bootstrap` stamps `PRAGMA user_version = 1` inside `BEGIN
            // IMMEDIATE`; `journal_mode = WAL` is only enabled afterwards.
            schema::bootstrap(&mut conn)?;
            storage.apply_journal_mode(&conn)?;
            schema::current_version_at(&conn, "post-open schema verification")?
        };
        storage.metrics.record_schema_bootstrap();
        drop(file_bootstrap_guard);
        tracing::debug!(
            target: "frankensearch.storage",
            schema_version = version,
            "storage bootstrap complete"
        );

        Ok(storage)
    }

    pub fn open_in_memory() -> SearchResult<Self> {
        Self::open(StorageConfig::in_memory())
    }

    /// Open a real in-memory connection without running schema bootstrap.
    ///
    /// Tests which exercise the fail-closed bootstrap gate itself need to
    /// install only the exact schema surface under test. Keeping this seam
    /// test-only prevents production callers from bypassing bootstrap.
    #[cfg(all(test, feature = "fts5"))]
    pub(crate) fn open_unbootstrapped_in_memory_for_test() -> SearchResult<Self> {
        let config = StorageConfig::in_memory();
        let conn = open_connection_with_retry(&config.db_path)
            .map_err(|error| map_storage_error_at("test connection open", error))?;
        let storage = Self {
            conn: Mutex::new(conn),
            lock_owner: AtomicU64::new(0),
            config,
            metrics: StorageMetrics::default(),
        };
        {
            let conn = storage.acquire_connection()?;
            storage.apply_connection_pragmas(&conn)?;
            storage.apply_journal_mode(&conn)?;
        }
        Ok(storage)
    }

    /// Lock the shared connection and return exclusive access to it.
    ///
    /// The ambient `asupersync` context is checkpointed before the lock is
    /// acquired so a cancelled caller does not queue behind an in-flight
    /// transaction.
    ///
    /// Fails fast when called re-entrantly from the thread already holding the
    /// connection (e.g. inside a `transaction` closure): use the
    /// `&mut rusqlite::Transaction` handed to the closure instead.
    pub fn connection(&self) -> SearchResult<StorageConnection<'_>> {
        self.acquire_connection()
    }

    /// Run `f` with shared access to the raw `rusqlite` connection while the
    /// storage mutex is held.
    pub fn with_connection<F, T>(&self, f: F) -> SearchResult<T>
    where
        F: FnOnce(&rusqlite::Connection) -> SearchResult<T>,
    {
        let conn = self.connection()?;
        f(&conn)
    }

    /// Run `f` with exclusive `&mut rusqlite::Connection` access while the
    /// storage mutex is held.
    ///
    /// This is the escape hatch for operations that need
    /// `Connection::transaction_with_behavior`, `Transaction::savepoint`, or
    /// other `&mut self` entry points outside [`Storage::transaction`].
    pub fn with_connection_mut<F, T>(&self, f: F) -> SearchResult<T>
    where
        F: FnOnce(&mut rusqlite::Connection) -> SearchResult<T>,
    {
        let mut conn = self.connection()?;
        f(&mut conn)
    }

    #[must_use]
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    #[must_use]
    pub fn metrics_snapshot(&self) -> StorageMetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Run a closure inside a write transaction.
    ///
    /// `StorageConfig::concurrent_transactions` maps to `BEGIN IMMEDIATE`
    /// (`TransactionBehavior::Immediate`); `false` maps to a deferred `BEGIN`.
    /// The closure receives `&mut rusqlite::Transaction`, which derefs to
    /// `&rusqlite::Connection`; use `Transaction::savepoint()` for nested
    /// scopes.
    pub fn transaction<F, T>(&self, f: F) -> SearchResult<T>
    where
        F: FnMut(&mut rusqlite::Transaction<'_>) -> SearchResult<T>,
    {
        self.run_transaction(
            self.config.transaction_behavior(),
            self.config.begin_sql(),
            f,
        )
    }

    /// Run a closure inside a `BEGIN IMMEDIATE` transaction.
    ///
    /// Unlike [`Storage::transaction`] with `concurrent_transactions = false`,
    /// this always acquires a write lock immediately, preventing concurrent
    /// writers from interleaving reads and writes. Use this when correctness
    /// depends on serialized read-then-write (e.g. `claim_batch`).
    pub fn immediate_transaction<F, T>(&self, f: F) -> SearchResult<T>
    where
        F: FnMut(&mut rusqlite::Transaction<'_>) -> SearchResult<T>,
    {
        self.run_transaction(
            rusqlite::TransactionBehavior::Immediate,
            "BEGIN IMMEDIATE;",
            f,
        )
    }

    fn run_transaction<F, T>(
        &self,
        behavior: rusqlite::TransactionBehavior,
        begin_sql: &'static str,
        mut f: F,
    ) -> SearchResult<T>
    where
        F: FnMut(&mut rusqlite::Transaction<'_>) -> SearchResult<T>,
    {
        retry_transient_storage(
            || {
                checkpoint_ambient("storage transaction")?;
                self.transaction_once(behavior, begin_sql, &mut f)
            },
            "storage transaction",
        )
    }

    // The connection guard must outlive the `Transaction` borrowing it, so
    // it cannot be dropped before the commit/rollback match below.
    #[allow(clippy::significant_drop_tightening)]
    fn transaction_once<F, T>(
        &self,
        behavior: rusqlite::TransactionBehavior,
        begin_sql: &'static str,
        f: &mut F,
    ) -> SearchResult<T>
    where
        F: FnMut(&mut rusqlite::Transaction<'_>) -> SearchResult<T>,
    {
        tracing::trace!(
            target: "frankensearch.storage",
            begin_sql,
            "starting storage transaction"
        );

        let mut conn = self.acquire_connection()?;
        let mut tx = conn
            .transaction_with_behavior(behavior)
            .map_err(|error| map_storage_error_at("storage transaction begin", error))?;

        let outcome = catch_unwind(AssertUnwindSafe(|| f(&mut tx)));
        match outcome {
            Ok(Ok(value)) => {
                // If COMMIT fails, rusqlite drops the transaction with
                // DropBehavior::Rollback and issues a best-effort ROLLBACK.
                tx.commit()
                    .map_err(|error| map_storage_error_at("storage transaction commit", error))?;
                self.metrics.record_commit();
                tracing::trace!(target: "frankensearch.storage", "storage transaction committed");
                Ok(value)
            }
            Ok(Err(error)) => {
                let error = match tx.rollback() {
                    Ok(()) => error,
                    Err(rollback_err) => {
                        tracing::warn!(
                            target: "frankensearch.storage",
                            error = %rollback_err,
                            "rollback failed after closure error"
                        );
                        unretryable_rollback_error(&error, &rollback_err)
                    }
                };
                self.metrics.record_rollback();
                tracing::debug!(
                    target: "frankensearch.storage",
                    ?error,
                    "storage transaction rolled back due to closure error"
                );
                Err(error)
            }
            Err(payload) => {
                if let Err(rollback_err) = tx.rollback() {
                    tracing::error!(
                        target: "frankensearch.storage",
                        error = %rollback_err,
                        "critical: rollback failed during panic recovery"
                    );
                }
                self.metrics.record_rollback();
                tracing::error!(
                    target: "frankensearch.storage",
                    "storage transaction rolled back after panic"
                );
                resume_unwind(payload);
            }
        }
    }

    /// Lock the underlying connection mutex.
    ///
    /// Poisoned locks are recovered: a panic inside a transaction closure
    /// already rolled back, so the connection is in a consistent state.
    ///
    /// The ambient `asupersync` context is checkpointed first so a cancelled
    /// caller never queues on the mutex.
    fn acquire_connection(&self) -> SearchResult<StorageConnection<'_>> {
        checkpoint_ambient("storage connection acquire")?;
        let tag = storage_thread_tag();
        if self.lock_owner.load(Ordering::Acquire) == tag {
            return Err(map_storage_error_at(
                "connection acquire",
                std::io::Error::other(
                    "re-entrant storage access on the thread already holding the connection; \
                     use the connection or transaction handle passed to the closure",
                ),
            ));
        }
        let guard = self
            .conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.lock_owner.store(tag, Ordering::Release);
        Ok(StorageConnection {
            guard,
            lock_owner: &self.lock_owner,
        })
    }

    /// Per-connection pragmas applied before schema bootstrap.
    ///
    /// `busy_timeout` and `cache_size` are set up front so the bootstrap
    /// transaction itself benefits; `journal_mode` is deferred until after
    /// bootstrap so `PRAGMA user_version = 1` lands inside `BEGIN IMMEDIATE`
    /// before WAL is enabled.
    fn apply_connection_pragmas(&self, conn: &rusqlite::Connection) -> SearchResult<()> {
        tracing::trace!(
            target: "frankensearch.storage",
            busy_timeout_ms = self.config.busy_timeout_ms,
            cache_size_pages = self.config.cache_size_pages,
            "applying storage connection pragmas"
        );

        // PRAGMA statements still prepare+step, so a peer connection closing
        // or checkpointing can hand back a transient BUSY/LOCKED even though
        // these pragmas are connection-local. Route them through the same
        // bounded transient retry as every other storage operation.
        retry_transient_storage(
            || {
                conn.execute_batch("PRAGMA foreign_keys=ON;")
                    .map_err(|error| map_storage_error_at("apply foreign_keys pragma", error))
            },
            "apply foreign_keys pragma",
        )?;

        conn.busy_timeout(Duration::from_millis(self.config.busy_timeout_ms))
            .map_err(|error| map_storage_error_at("apply busy_timeout pragma", error))?;

        retry_transient_storage(
            || {
                conn.execute_batch(&format!(
                    "PRAGMA cache_size={};",
                    self.config.cache_size_pages
                ))
                .map_err(|error| map_storage_error_at("apply cache_size pragma", error))
            },
            "apply cache_size pragma",
        )?;

        Ok(())
    }

    /// Journal-mode pragmas, applied after bootstrap for file-backed
    /// databases. `journal_mode` is persistent on disk, so it is only changed
    /// once the `schema/user_version` marker transaction has committed.
    fn apply_journal_mode(&self, conn: &rusqlite::Connection) -> SearchResult<()> {
        tracing::trace!(
            target: "frankensearch.storage",
            wal_mode = self.config.wal_mode,
            "applying storage journal_mode pragma"
        );

        if self.config.wal_mode {
            retry_transient_storage(
                || {
                    conn.execute_batch("PRAGMA journal_mode=WAL;")
                        .map_err(|error| map_storage_error_at("apply journal_mode pragma", error))
                },
                "apply journal_mode pragma",
            )?;
        } else {
            let delete_result = retry_transient_storage(
                || {
                    conn.execute_batch("PRAGMA journal_mode=DELETE;")
                        .map_err(|error| {
                            map_storage_error_at("apply journal_mode delete pragma", error)
                        })
                },
                "apply journal_mode delete pragma",
            );
            if let Err(error) = delete_result {
                tracing::warn!(
                    target: "frankensearch.storage",
                    ?error,
                    "journal_mode=DELETE was not accepted by backend; falling back to WAL"
                );
                retry_transient_storage(
                    || {
                        conn.execute_batch("PRAGMA journal_mode=WAL;")
                            .map_err(|error| {
                                map_storage_error_at("apply journal_mode fallback pragma", error)
                            })
                    },
                    "apply journal_mode fallback pragma",
                )?;
            }
        }

        Ok(())
    }
}

pub(crate) fn map_storage_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    map_storage_error_at("storage operation", source)
}

pub(crate) fn map_storage_error_at<E>(stage: &'static str, source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(StorageStageError { stage, source }),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{self, AssertUnwindSafe};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use asupersync::Cx;
    use frankensearch_core::{SearchError, SearchResult};

    use crate::schema::{self, SCHEMA_VERSION};

    use super::{Storage, StorageConfig};

    fn busy_error() -> rusqlite::Error {
        rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY), None)
    }

    fn busy_snapshot_error() -> rusqlite::Error {
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY_SNAPSHOT),
            None,
        )
    }

    fn schema_changed_error() -> rusqlite::Error {
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_SCHEMA),
            None,
        )
    }

    fn no_such_table_error(name: &str) -> rusqlite::Error {
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
            Some(format!("no such table: {name}")),
        )
    }

    struct TempDbPath {
        path: PathBuf,
    }

    impl TempDbPath {
        fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "frankensearch-storage-{tag}-{}-{nanos}.sqlite3",
                process::id()
            ));
            Self { path }
        }

        fn config(&self) -> StorageConfig {
            StorageConfig {
                db_path: self.path.clone(),
                ..StorageConfig::default()
            }
        }
    }

    impl Drop for TempDbPath {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let candidate = if suffix.is_empty() {
                    self.path.clone()
                } else {
                    PathBuf::from(format!("{}{}", self.path.display(), suffix))
                };
                let _ = std::fs::remove_file(candidate);
            }
        }
    }

    const CONCURRENT_OPEN_THREADS: usize = 4;
    const CONCURRENT_OPEN_STRESS_ROUNDS: usize = 100;
    const REPEATED_CURRENT_SCHEMA_OPENS: usize = 32;

    fn run_concurrent_open_round(config: &StorageConfig) -> Vec<SearchResult<i64>> {
        let barrier = Arc::new(Barrier::new(CONCURRENT_OPEN_THREADS));
        let (tx, rx) = mpsc::channel::<SearchResult<i64>>();
        let mut handles = Vec::with_capacity(CONCURRENT_OPEN_THREADS);

        for _ in 0..CONCURRENT_OPEN_THREADS {
            let cfg = config.clone();
            let gate = Arc::clone(&barrier);
            let sender = tx.clone();
            handles.push(thread::spawn(move || {
                gate.wait();
                let open_result = Storage::open(cfg).and_then(|storage| {
                    let conn = storage.connection()?;
                    schema::current_version(&conn)
                });
                sender
                    .send(open_result)
                    .expect("result send should succeed");
            }));
        }
        drop(tx);

        let results: Vec<SearchResult<i64>> = rx.into_iter().collect();
        for handle in handles {
            handle.join().expect("concurrent opener thread should join");
        }
        results
    }

    fn insert_document_minimal(conn: &rusqlite::Connection, id: &str) -> SearchResult<()> {
        conn.execute(
            "INSERT INTO documents (\
            doc_id, content_preview, content_hash, content_length, created_at, updated_at\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6);",
            rusqlite::params![
                id,
                "fixture-content",
                vec![0x33_u8; 32],
                64_i64,
                1_739_499_200_i64,
                1_739_499_200_i64
            ],
        )
        .map_err(super::map_storage_error)?;
        Ok(())
    }

    fn count_documents(conn: &rusqlite::Connection) -> SearchResult<i64> {
        conn.query_row("SELECT COUNT(*) FROM documents;", [], |row| row.get(0))
            .map_err(super::map_storage_error)
    }

    fn pragma_i64(conn: &rusqlite::Connection, name: &str) -> SearchResult<i64> {
        conn.query_row(&format!("PRAGMA {name};"), [], |row| row.get(0))
            .map_err(super::map_storage_error)
    }

    fn pragma_text(conn: &rusqlite::Connection, name: &str) -> SearchResult<String> {
        conn.query_row(&format!("PRAGMA {name};"), [], |row| {
            row.get::<_, String>(0)
        })
        .map_err(super::map_storage_error)
    }

    /// (busy, `log_frames`, `checkpointed_frames`) from `PRAGMA
    /// wal_checkpoint(PASSIVE)` — a passive checkpoint never blocks and never
    /// truncates, so the counts also serve as a WAL growth probe.
    fn wal_checkpoint_stats(conn: &rusqlite::Connection) -> SearchResult<(i64, i64, i64)> {
        conn.query_row("PRAGMA wal_checkpoint(PASSIVE);", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .map_err(super::map_storage_error)
    }

    fn schema_version_markers(conn: &rusqlite::Connection) -> SearchResult<Vec<i64>> {
        let mut stmt = conn
            .prepare("SELECT version FROM schema_version ORDER BY version;")
            .map_err(super::map_storage_error)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(super::map_storage_error)?
            .collect::<Result<Vec<i64>, _>>()
            .map_err(super::map_storage_error)
    }

    fn artifact_len(path: &Path) -> u64 {
        match std::fs::metadata(path) {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => panic!(
                "metadata for {} should be readable: {error}",
                path.display()
            ),
        }
    }

    fn wal_artifact_lengths(db_path: &Path) -> [u64; 2] {
        let dot_wal = PathBuf::from(format!("{}.wal", db_path.display()));
        let dash_wal = PathBuf::from(format!("{}-wal", db_path.display()));
        [artifact_len(&dot_wal), artifact_len(&dash_wal)]
    }

    #[test]
    fn connection_open_retry_exhaustion_preserves_typed_busy_error() {
        let mut attempts = 0_u32;
        let mut observed_retries = Vec::new();

        let error = super::retry_connection_open(
            || {
                attempts = attempts.saturating_add(1);
                Err::<(), _>(busy_error())
            },
            |next_attempt, delay, error| {
                assert!(matches!(
                    error,
                    rusqlite::Error::SqliteFailure(inner, _)
                        if inner.code == rusqlite::ErrorCode::DatabaseBusy
                ));
                observed_retries.push((next_attempt, delay));
            },
        )
        .expect_err("persistent Busy must exhaust the bounded retry budget");

        assert!(matches!(
            error,
            rusqlite::Error::SqliteFailure(inner, _)
                if inner.code == rusqlite::ErrorCode::DatabaseBusy
        ));
        assert_eq!(attempts, super::CONNECTION_OPEN_MAX_ATTEMPTS);
        let expected_retries: Vec<_> = (1..super::CONNECTION_OPEN_MAX_ATTEMPTS)
            .map(|failed_attempt| {
                (
                    failed_attempt.saturating_add(1),
                    super::connection_open_retry_delay(failed_attempt),
                )
            })
            .collect();
        assert_eq!(observed_retries, expected_retries);
    }

    #[test]
    fn connection_open_retry_retries_busy_snapshot() {
        let mut attempts = 0_u32;

        let error = super::retry_connection_open(
            || {
                attempts = attempts.saturating_add(1);
                Err::<(), _>(busy_snapshot_error())
            },
            |_, _, _| {},
        )
        .expect_err("persistent BusySnapshot must exhaust the bounded retry budget");

        assert!(matches!(
            error,
            rusqlite::Error::SqliteFailure(inner, _)
                if inner.extended_code == rusqlite::ffi::SQLITE_BUSY_SNAPSHOT
        ));
        assert_eq!(attempts, super::CONNECTION_OPEN_MAX_ATTEMPTS);
    }

    #[test]
    fn transient_storage_error_detects_wrapped_busy_snapshot() {
        let error = super::map_storage_error(busy_snapshot_error());
        assert!(super::is_retryable_storage_error(&error));
    }

    #[test]
    fn retryable_storage_error_includes_schema_changed() {
        assert!(super::is_retryable_storage_error(
            &super::map_storage_error(schema_changed_error())
        ));
    }

    #[test]
    fn failed_rollback_error_is_not_retryable() {
        let error = super::unretryable_rollback_error(
            &super::map_storage_error(busy_error()),
            &busy_error(),
        );
        assert!(!super::is_retryable_storage_error(&error));
    }

    #[test]
    fn transient_storage_error_rejects_invalid_config() {
        let error = SearchError::InvalidConfig {
            field: "test".to_owned(),
            value: "busy snapshot locked".to_owned(),
            reason: "string match must not classify this as transient".to_owned(),
        };
        assert!(!super::is_retryable_storage_error(&error));
    }

    #[test]
    fn retry_transient_storage_retries_busy_snapshot_then_succeeds() {
        let mut attempts = 0_u32;
        let value = super::retry_transient_storage(
            || {
                attempts = attempts.saturating_add(1);
                if attempts < 3 {
                    Err(super::map_storage_error(busy_snapshot_error()))
                } else {
                    Ok(7_i32)
                }
            },
            "test",
        )
        .expect("BusySnapshot should succeed after bounded retries");
        assert_eq!(value, 7);
        assert_eq!(attempts, 3);
    }

    #[cfg(feature = "fts5")]
    #[test]
    fn retry_transient_storage_async_retries_busy_snapshot_then_succeeds() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let attempts = std::cell::Cell::new(0_u32);
            let value = super::retry_transient_storage_async(
                &cx,
                || {
                    let next = attempts.get().saturating_add(1);
                    attempts.set(next);
                    async move {
                        if next < 3 {
                            Err(super::map_storage_error(busy_snapshot_error()))
                        } else {
                            Ok(7_i32)
                        }
                    }
                },
                "test",
            )
            .await
            .expect("BusySnapshot should succeed after bounded retries");
            assert_eq!(value, 7);
            assert_eq!(attempts.get(), 3);
        });
    }

    #[test]
    fn connection_open_retry_propagates_non_busy_error_without_retry() {
        let mut attempts = 0_u32;
        let mut retries = 0_u32;

        let error = super::retry_connection_open(
            || {
                attempts = attempts.saturating_add(1);
                Err::<(), _>(no_such_table_error("unrelated"))
            },
            |_, _, _| retries = retries.saturating_add(1),
        )
        .expect_err("non-Busy connection errors must fail immediately");

        assert!(matches!(error, rusqlite::Error::SqliteFailure(_, Some(_))));
        assert_eq!(attempts, 1);
        assert_eq!(retries, 0);
    }

    #[test]
    fn connection_observes_ambient_cancellation() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let cx = Cx::for_testing();
        let _ambient = Cx::set_current(Some(cx.clone()));
        cx.set_cancel_requested(true);

        let error = storage
            .connection()
            .expect_err("ambient cancellation must fail the lock acquire");
        assert!(matches!(error, SearchError::Cancelled { .. }));
    }

    #[test]
    fn transaction_observes_ambient_cancellation() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let cx = Cx::for_testing();
        let _ambient = Cx::set_current(Some(cx.clone()));
        cx.set_cancel_requested(true);

        let result: SearchResult<()> = storage.transaction(|_conn| Ok(()));
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
    }

    #[test]
    fn storage_is_send_and_sync() {
        // `frankensearch-fsfs` shares one `Arc<Storage>` across workers.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Storage>();
    }

    #[test]
    fn open_in_memory_bootstraps_schema() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let conn = storage.connection().expect("connection guard");
        let version = schema::current_version(&conn).expect("schema version row");
        assert_eq!(version, SCHEMA_VERSION);
        drop(conn);

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.opens, 1);
        assert_eq!(metrics.schema_bootstraps, 1);
    }

    #[test]
    fn open_applies_configured_pragmas_deterministically() {
        let tmp = TempDbPath::new("pragmas");
        let storage = Storage::open(StorageConfig {
            db_path: tmp.path.clone(),
            wal_mode: true,
            busy_timeout_ms: 1_234,
            cache_size_pages: 321,
            concurrent_transactions: false,
        })
        .expect("storage should open with configured pragmas");

        let conn = storage.connection().expect("connection guard");
        assert_eq!(
            pragma_text(&conn, "journal_mode")
                .expect("journal_mode pragma should be queryable")
                .to_ascii_lowercase(),
            "wal"
        );
        assert_eq!(
            pragma_i64(&conn, "busy_timeout").expect("busy_timeout pragma should be queryable"),
            1_234
        );
        assert_eq!(
            pragma_i64(&conn, "cache_size").expect("cache_size pragma should be queryable"),
            321
        );
        assert_eq!(
            pragma_i64(&conn, "foreign_keys").expect("foreign_keys pragma should be queryable"),
            1,
            "foreign key enforcement must be enabled on every storage connection"
        );
    }

    #[test]
    fn schema_bootstrap_is_idempotent() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let version = storage
            .with_connection_mut(|conn| {
                schema::bootstrap(conn).expect("second bootstrap should succeed");
                schema::bootstrap(conn).expect("third bootstrap should succeed");
                schema::current_version(conn)
            })
            .expect("bootstrap should succeed");
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn concurrent_open_initializes_schema_consistently() {
        let tmp = TempDbPath::new("concurrent-bootstrap");
        let config = StorageConfig {
            busy_timeout_ms: 0,
            ..tmp.config()
        };
        let results = run_concurrent_open_round(&config);
        assert_eq!(results.len(), CONCURRENT_OPEN_THREADS);
        for (thread_idx, result) in results.into_iter().enumerate() {
            match result {
                Ok(version) => assert_eq!(
                    version, SCHEMA_VERSION,
                    "thread {thread_idx} should observe latest schema version"
                ),
                Err(error) => panic!("thread {thread_idx} failed concurrent open: {error}"),
            }
        }
    }

    #[test]
    fn concurrent_open_stays_stable_across_repeated_rounds() {
        let tmp = TempDbPath::new("concurrent-bootstrap-rounds");
        let config = StorageConfig {
            busy_timeout_ms: 0,
            ..tmp.config()
        };

        let seed = Storage::open(config.clone()).expect("schema seed should open");
        {
            let conn = seed.connection().expect("seed connection");
            assert_eq!(
                schema::current_version(&conn).expect("seed version should read"),
                SCHEMA_VERSION
            );
        }
        drop(seed);

        for round in 0..CONCURRENT_OPEN_STRESS_ROUNDS {
            let results = run_concurrent_open_round(&config);
            assert_eq!(
                results.len(),
                CONCURRENT_OPEN_THREADS,
                "round {round} should return one result per thread"
            );
            for (thread_idx, result) in results.into_iter().enumerate() {
                match result {
                    Ok(version) => assert_eq!(
                        version, SCHEMA_VERSION,
                        "round {round}, thread {thread_idx} should observe latest schema version"
                    ),
                    Err(error) => {
                        panic!("round {round}, thread {thread_idx} failed concurrent open: {error}")
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_current_schema_opens_do_not_mutate_marker_or_grow_wal() {
        let tmp = TempDbPath::new("current-schema-read-only");
        let config = StorageConfig {
            busy_timeout_ms: 0,
            ..tmp.config()
        };
        let seed = Storage::open(config.clone()).expect("schema seed should open");
        seed.with_connection(|conn| {
            conn.execute_batch("PRAGMA wal_autocheckpoint=0;")
                .expect("test should disable automatic checkpoints");
            // Drain the bootstrap frames so later WAL growth is attributable
            // purely to the repeated opens below.
            wal_checkpoint_stats(conn).expect("baseline checkpoint should run");
            Ok(())
        })
        .expect("seed connection should open");

        let before_lengths = wal_artifact_lengths(&tmp.path);
        let before_markers = {
            let conn = seed.connection().expect("seed connection");
            schema_version_markers(&conn).expect("baseline schema markers should be readable")
        };

        let mut opened = Vec::with_capacity(REPEATED_CURRENT_SCHEMA_OPENS);
        for iteration in 0..REPEATED_CURRENT_SCHEMA_OPENS {
            let storage = Storage::open(config.clone()).unwrap_or_else(|error| {
                panic!("current-schema open {iteration} should succeed: {error}")
            });
            {
                let conn = storage.connection().expect("opened connection");
                assert_eq!(
                    schema::current_version(&conn)
                        .expect("current-schema version should remain readable"),
                    SCHEMA_VERSION,
                    "open {iteration} should observe v1"
                );
            }
            opened.push(storage);
        }

        let (after_busy, _after_log, after_checkpointed) = {
            let conn = seed.connection().expect("seed connection");
            wal_checkpoint_stats(&conn).expect("final WAL stats should be observable")
        };
        let after_lengths = wal_artifact_lengths(&tmp.path);
        let after_markers = {
            let conn = seed.connection().expect("seed connection");
            schema_version_markers(&conn).expect("final schema markers should be readable")
        };

        assert_eq!(after_busy, 0, "passive checkpoint should never block");
        assert_eq!(
            after_checkpointed, 0,
            "read-only current-schema opens must not append WAL frames"
        );
        assert_eq!(
            after_lengths, before_lengths,
            "read-only current-schema opens must not grow either supported WAL artifact"
        );
        assert_eq!(
            after_markers, before_markers,
            "read-only current-schema opens must not rewrite schema markers"
        );

        drop(opened);
    }

    #[test]
    fn transaction_rolls_back_on_error() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        let result: SearchResult<()> = storage.transaction(|conn| {
            insert_document_minimal(conn, "doc-error")?;
            Err(SearchError::InvalidConfig {
                field: "test".to_owned(),
                value: "forced".to_owned(),
                reason: "force rollback".to_owned(),
            })
        });

        assert!(result.is_err(), "transaction should return original error");
        let conn = storage.connection().expect("connection guard");
        assert_eq!(
            count_documents(&conn).expect("count should work"),
            0,
            "document insert should have been rolled back"
        );
        drop(conn);

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_commits, 0);
        assert_eq!(metrics.tx_rollbacks, 1);
    }

    #[test]
    fn transaction_rolls_back_on_panic_and_connection_stays_usable() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        let panic_result = panic::catch_unwind(AssertUnwindSafe(|| {
            let _: SearchResult<()> = storage.transaction(|conn| {
                insert_document_minimal(conn, "doc-panic")
                    .expect("insert should succeed before panic");
                panic!("forced panic");
            });
        }));

        assert!(panic_result.is_err(), "panic should propagate to caller");
        let conn = storage.connection().expect("connection guard");
        assert_eq!(
            count_documents(&conn).expect("count should work"),
            0,
            "panic path should rollback transaction"
        );
        assert_eq!(
            schema::current_version(&conn).expect("connection should remain usable"),
            SCHEMA_VERSION
        );
        drop(conn);

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_rollbacks, 1);
        assert_eq!(metrics.tx_commits, 0);
    }

    #[test]
    fn commit_persists_after_reopen() {
        let tmp = TempDbPath::new("visibility");
        let config = tmp.config();

        let storage_a = Storage::open(config.clone()).expect("writer storage should open");

        storage_a
            .transaction(|conn| {
                insert_document_minimal(conn, "doc-visible")?;
                assert_eq!(
                    count_documents(conn)?,
                    1,
                    "write should be visible inside writer transaction"
                );
                Ok(())
            })
            .expect("transaction should commit");

        drop(storage_a);

        let storage_c = Storage::open(config).expect("post-commit storage should open");
        let conn = storage_c.connection().expect("reader connection");
        assert_eq!(
            count_documents(&conn).expect("count after commit"),
            1,
            "committed write should be visible to newly opened connection"
        );
    }

    #[test]
    fn rollback_is_not_persisted_after_reopen() {
        let tmp = TempDbPath::new("rollback");
        let config = tmp.config();

        let storage_a = Storage::open(config).expect("writer storage should open");

        let tx_result: SearchResult<()> = storage_a.transaction(|conn| {
            insert_document_minimal(conn, "doc-rollback")?;
            Err(SearchError::InvalidConfig {
                field: "test".to_owned(),
                value: "rollback".to_owned(),
                reason: "forced rollback".to_owned(),
            })
        });
        assert!(tx_result.is_err(), "transaction should rollback");
        let conn = storage_a.connection().expect("reader connection");
        assert_eq!(
            count_documents(&conn).expect("count after rollback"),
            0,
            "rolled back write should not persist in active connection state"
        );
    }

    #[test]
    fn dropped_connection_allows_reopen_and_write() {
        let tmp = TempDbPath::new("drop");
        let config = tmp.config();

        let storage_a = Storage::open(config.clone()).expect("first storage should open");
        drop(storage_a);

        let storage_b = Storage::open(config).expect("second storage should open");

        storage_b
            .transaction(|conn| {
                insert_document_minimal(conn, "doc-after-drop")?;
                Ok(())
            })
            .expect("second connection should remain fully usable after first drop");

        let conn = storage_b.connection().expect("reader connection");
        assert_eq!(count_documents(&conn).expect("count after write"), 1);
    }

    #[test]
    fn nested_transaction_error_rolls_back_outer_transaction() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        let result: SearchResult<()> = storage.transaction(|conn| {
            insert_document_minimal(conn, "doc-nested")?;
            storage.transaction(|_nested_conn| Ok(()))
        });

        assert!(
            result.is_err(),
            "nested transaction should fail and propagate error"
        );
        let conn = storage.connection().expect("connection guard");
        assert_eq!(
            count_documents(&conn).expect("count after nested failure"),
            0,
            "outer transaction should rollback when nested begin fails"
        );
        drop(conn);

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_rollbacks, 1);
        assert_eq!(metrics.tx_commits, 0);
    }

    #[test]
    fn nested_savepoint_inside_transaction_isolates_partial_failure() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        storage
            .transaction(|conn| {
                insert_document_minimal(conn, "doc-outer")?;
                let mut sp = conn.savepoint().map_err(super::map_storage_error)?;
                insert_document_minimal(&sp, "doc-inner")?;
                sp.rollback().map_err(super::map_storage_error)?;
                Ok(())
            })
            .expect("outer transaction should commit despite savepoint rollback");

        let conn = storage.connection().expect("connection guard");
        assert_eq!(
            count_documents(&conn).expect("count after savepoint rollback"),
            1,
            "only the outer write should survive the rolled-back savepoint"
        );
    }

    #[test]
    fn uncommitted_writes_are_invisible_to_concurrent_reader() {
        let tmp = TempDbPath::new("snapshot");
        let config = tmp.config();
        let reader = Storage::open(config.clone()).expect("reader storage should open");
        let writer_config = config;

        let (inserted_tx, inserted_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();

        let writer_thread = thread::spawn(move || {
            let writer = Storage::open(writer_config).expect("writer storage should open");
            writer
                .transaction(|conn| {
                    insert_document_minimal(conn, "doc-uncommitted")?;
                    inserted_tx
                        .send(())
                        .expect("writer should signal insert before commit");
                    release_rx
                        .recv()
                        .expect("writer should wait for reader inspection");
                    Ok(())
                })
                .expect("writer transaction should commit");
        });

        inserted_rx
            .recv()
            .expect("reader should observe insert signal");

        {
            let conn = reader.connection().expect("reader connection");
            assert_eq!(
                count_documents(&conn).expect("reader count during writer transaction"),
                0,
                "reader should not see uncommitted write from concurrent transaction"
            );
        }

        release_tx
            .send(())
            .expect("reader should release writer to commit");
        writer_thread.join().expect("writer thread should join");

        // In WAL mode without an explicit read transaction, each statement
        // sees the latest committed state.  After the writer commits, the
        // reader observes the new row immediately.
        let conn = reader.connection().expect("reader connection");
        assert_eq!(
            count_documents(&conn).expect("reader count after writer commit"),
            1,
            "reader without explicit transaction should see committed write"
        );
    }

    #[test]
    fn storage_config_serde_roundtrip() {
        let config = StorageConfig {
            db_path: PathBuf::from("/tmp/test.db"),
            wal_mode: false,
            busy_timeout_ms: 10_000,
            cache_size_pages: 500,
            concurrent_transactions: false,
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: StorageConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn storage_config_in_memory_path() {
        let config = StorageConfig::in_memory();
        assert_eq!(config.db_path, PathBuf::from(":memory:"));
        assert!(config.wal_mode);
        assert!(config.concurrent_transactions);
    }

    #[test]
    fn storage_config_begin_sql_concurrent_vs_standard() {
        let concurrent = StorageConfig {
            concurrent_transactions: true,
            ..StorageConfig::default()
        };
        assert_eq!(concurrent.begin_sql(), "BEGIN IMMEDIATE;");

        let standard = StorageConfig {
            concurrent_transactions: false,
            ..StorageConfig::default()
        };
        assert_eq!(standard.begin_sql(), "BEGIN;");
    }

    #[test]
    fn immediate_transaction_commits_successfully() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        storage
            .immediate_transaction(|conn| {
                insert_document_minimal(conn, "doc-immediate")?;
                Ok(())
            })
            .expect("immediate transaction should commit");

        let conn = storage.connection().expect("connection guard");
        assert_eq!(count_documents(&conn).expect("count after immediate tx"), 1);
        drop(conn);

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_commits, 1);
    }

    #[test]
    fn immediate_transaction_rolls_back_on_error() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        let result: SearchResult<()> = storage.immediate_transaction(|conn| {
            insert_document_minimal(conn, "doc-immediate-rollback")?;
            Err(SearchError::InvalidConfig {
                field: "test".to_owned(),
                value: "forced".to_owned(),
                reason: "force rollback".to_owned(),
            })
        });

        assert!(result.is_err());
        let conn = storage.connection().expect("connection guard");
        assert_eq!(count_documents(&conn).expect("count after rollback"), 0);
        drop(conn);

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_rollbacks, 1);
    }

    #[test]
    fn storage_debug_format() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let debug = format!("{storage:?}");
        assert!(debug.contains("Storage"));
        assert!(debug.contains(":memory:"));
    }

    #[test]
    fn multi_threaded_serial_writers_commit_disjoint_documents() {
        let tmp = TempDbPath::new("concurrent-writers");
        let config = tmp.config();
        let (first_done_tx, first_done_rx) = mpsc::channel::<()>();

        let cfg_a = config.clone();
        let first_writer = thread::spawn(move || {
            let storage = Storage::open(cfg_a).expect("first writer storage should open");
            storage
                .transaction(|conn| {
                    insert_document_minimal(conn, "doc-a")?;
                    Ok(())
                })
                .expect("first writer transaction should commit");
            first_done_tx
                .send(())
                .expect("first writer should signal completion");
        });

        let cfg_b = config.clone();
        let second_writer = thread::spawn(move || {
            first_done_rx
                .recv()
                .expect("second writer should wait for first writer");
            let storage = Storage::open(cfg_b).expect("second writer storage should open");
            storage
                .transaction(|conn| {
                    insert_document_minimal(conn, "doc-b")?;
                    Ok(())
                })
                .expect("second writer transaction should commit");
        });

        first_writer
            .join()
            .expect("first writer thread should not panic");
        second_writer
            .join()
            .expect("second writer thread should not panic");

        let verifier = Storage::open(config).expect("verifier storage should open");
        let conn = verifier.connection().expect("verifier connection");
        assert_eq!(
            count_documents(&conn).expect("count after concurrent writes"),
            2,
            "both writer transactions should persist exactly one row"
        );
    }
}
