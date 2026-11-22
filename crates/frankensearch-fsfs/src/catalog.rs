//! fsfs catalog/changelog schema and replay semantics.
//!
//! This module defines the persistent SQLite model for fsfs incremental
//! indexing (native `rusqlite`):
//! - `fsfs_catalog_files`: current file identity + indexing state
//! - `fsfs_catalog_changelog`: append-only mutation stream for replay
//! - `fsfs_catalog_replay_checkpoint`: deterministic resume cursor per consumer

use std::io;
use std::path::Path;
use std::time::Duration;

use frankensearch_core::{SearchError, SearchResult};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};

pub const CATALOG_SCHEMA_VERSION: i64 = 1;
const SUBSYSTEM: &str = "fsfs_catalog";

/// Busy timeout for catalog connections, matching `frankensearch-storage`'s
/// default so a contended shared `fsfs.db` waits instead of failing the
/// cleanup lane outright.
const CATALOG_BUSY_TIMEOUT: Duration = Duration::from_millis(5_000);

const LATEST_SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS fsfs_catalog_files (\
        file_key TEXT PRIMARY KEY,\
        mount_id TEXT NOT NULL,\
        canonical_path TEXT NOT NULL,\
        device INTEGER,\
        inode INTEGER,\
        content_hash BLOB,\
        revision INTEGER NOT NULL CHECK (revision >= 0),\
        ingestion_class TEXT NOT NULL CHECK (ingestion_class IN ('full_semantic_lexical', 'lexical_only', 'metadata_only', 'skip')),\
        pipeline_status TEXT NOT NULL CHECK (pipeline_status IN ('discovered', 'queued', 'embedding', 'indexed', 'failed', 'skipped', 'tombstoned')),\
        eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),\
        first_seen_ts INTEGER NOT NULL,\
        last_seen_ts INTEGER NOT NULL,\
        updated_ts INTEGER NOT NULL,\
        deleted_ts INTEGER,\
        last_error TEXT,\
        metadata_json TEXT,\
        UNIQUE(mount_id, canonical_path)\
    );",
    "CREATE TABLE IF NOT EXISTS fsfs_catalog_changelog (\
        change_id INTEGER PRIMARY KEY AUTOINCREMENT,\
        stream_seq INTEGER NOT NULL UNIQUE,\
        file_key TEXT NOT NULL REFERENCES fsfs_catalog_files(file_key) ON DELETE CASCADE,\
        revision INTEGER NOT NULL CHECK (revision >= 0),\
        change_kind TEXT NOT NULL CHECK (change_kind IN ('upsert', 'reclassified', 'status', 'tombstone')),\
        ingestion_class TEXT NOT NULL CHECK (ingestion_class IN ('full_semantic_lexical', 'lexical_only', 'metadata_only', 'skip')),\
        pipeline_status TEXT NOT NULL CHECK (pipeline_status IN ('discovered', 'queued', 'embedding', 'indexed', 'failed', 'skipped', 'tombstoned')),\
        content_hash BLOB,\
        event_ts INTEGER NOT NULL,\
        correlation_id TEXT NOT NULL,\
        replay_token TEXT NOT NULL UNIQUE,\
        applied_ts INTEGER,\
        UNIQUE(file_key, revision, change_kind)\
    );",
    "CREATE TABLE IF NOT EXISTS fsfs_catalog_replay_checkpoint (\
        consumer_id TEXT PRIMARY KEY,\
        last_applied_seq INTEGER NOT NULL,\
        updated_ts INTEGER NOT NULL\
    );",
    "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_catalog_dirty_lookup ON fsfs_catalog_files(pipeline_status, ingestion_class, last_seen_ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_catalog_revisions ON fsfs_catalog_files(file_key, revision DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_catalog_cleanup ON fsfs_catalog_files(deleted_ts, pipeline_status);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_catalog_content_hash ON fsfs_catalog_files(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_changelog_replay ON fsfs_catalog_changelog(stream_seq ASC);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_changelog_file_revision ON fsfs_catalog_changelog(file_key, revision DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_changelog_pending_apply ON fsfs_catalog_changelog(applied_ts, stream_seq ASC);",
];

pub const INDEX_CATALOG_DIRTY_LOOKUP: &str = "idx_fsfs_catalog_dirty_lookup";
pub const INDEX_CATALOG_REVISIONS: &str = "idx_fsfs_catalog_revisions";
pub const INDEX_CATALOG_CLEANUP: &str = "idx_fsfs_catalog_cleanup";
pub const INDEX_CATALOG_CONTENT_HASH: &str = "idx_fsfs_catalog_content_hash";
pub const INDEX_CHANGELOG_REPLAY: &str = "idx_fsfs_changelog_replay";
pub const INDEX_CHANGELOG_FILE_REVISION: &str = "idx_fsfs_changelog_file_revision";
pub const INDEX_CHANGELOG_PENDING_APPLY: &str = "idx_fsfs_changelog_pending_apply";

/// Incremental workload query: pick files that still require indexing work.
pub const DIRTY_CATALOG_LOOKUP_SQL: &str = "SELECT file_key, revision, ingestion_class, pipeline_status, last_seen_ts \
    FROM fsfs_catalog_files \
    WHERE pipeline_status IN ('discovered', 'queued', 'failed') \
      AND ingestion_class != 'skip' \
    ORDER BY last_seen_ts DESC \
    LIMIT ?1;";

/// Incremental workload query: stream changelog rows after a checkpoint.
pub const CHANGELOG_REPLAY_BATCH_SQL: &str = "SELECT stream_seq, file_key, revision, change_kind, ingestion_class, pipeline_status, event_ts \
    FROM fsfs_catalog_changelog \
    WHERE stream_seq > ?1 \
    ORDER BY stream_seq ASC \
    LIMIT ?2;";

/// Incremental workload query: purge old tombstones once retention allows it.
pub const CLEANUP_TOMBSTONES_SQL: &str = "DELETE FROM fsfs_catalog_files \
    WHERE deleted_ts IS NOT NULL \
      AND deleted_ts <= ?1 \
      AND pipeline_status = 'tombstoned';";

/// Delete tombstoned catalog rows at or before the provided cutoff timestamp.
///
/// # Errors
///
/// Returns an error if `SQLite` execution fails.
pub fn cleanup_tombstones(conn: &Connection, cutoff_ts_ms: i64) -> SearchResult<usize> {
    ensure_catalog_foreign_keys(conn)?;
    retry_catalog_op(
        || {
            conn.execute(CLEANUP_TOMBSTONES_SQL, rusqlite::params![cutoff_ts_ms])
                .map_err(catalog_error)
        },
        "catalog tombstone cleanup",
    )
}

/// Open a catalog database file and prune tombstoned rows past retention.
///
/// Missing database files are treated as empty catalogs and return `0`.
///
/// # Errors
///
/// Returns an error if the catalog cannot be opened/bootstrapped or SQL execution
/// fails.
pub fn cleanup_tombstones_for_path(db_path: &Path, cutoff_ts_ms: i64) -> SearchResult<usize> {
    if !db_path.exists() {
        return Ok(0);
    }
    let mut conn = open_catalog_connection(&db_path.display().to_string())?;
    bootstrap_catalog_schema(&mut conn)?;
    cleanup_tombstones(&conn, cutoff_ts_ms)
}

const CATALOG_OPEN_MAX_ATTEMPTS: u32 = 8;
const CATALOG_OPEN_RETRY_BASE_DELAY_MS: u64 = 1;
const CATALOG_OPEN_RETRY_MAX_DELAY_MS: u64 = 50;

fn open_catalog_connection(path: &str) -> SearchResult<Connection> {
    let path = path.to_owned();
    let mut attempt = 1;
    loop {
        match Connection::open(&path) {
            Ok(conn) => {
                return retry_catalog_op(
                    || configure_catalog_connection(&conn),
                    "catalog connection pragmas",
                )
                .map(|()| conn);
            }
            Err(error)
                if is_retryable_sqlite_error(&error) && attempt < CATALOG_OPEN_MAX_ATTEMPTS =>
            {
                let growth = attempt.saturating_sub(1).min(6);
                let delay_ms = (CATALOG_OPEN_RETRY_BASE_DELAY_MS << growth)
                    .min(CATALOG_OPEN_RETRY_MAX_DELAY_MS);
                tracing::debug!(
                    target: "frankensearch.fsfs.catalog",
                    path,
                    next_attempt = attempt.saturating_add(1),
                    max_attempts = CATALOG_OPEN_MAX_ATTEMPTS,
                    delay_ms,
                    error = %error,
                    "retrying transient catalog connection-open contention"
                );
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(catalog_error(error)),
        }
    }
}

/// Per-connection settings every catalog handle needs: foreign-key
/// enforcement so `ON DELETE CASCADE` fires on `fsfs_catalog_changelog`, and
/// a busy timeout so a contended `fsfs.db` waits instead of failing.
fn configure_catalog_connection(conn: &Connection) -> SearchResult<()> {
    conn.execute_batch("PRAGMA foreign_keys=ON;")
        .map_err(catalog_error)?;
    conn.busy_timeout(CATALOG_BUSY_TIMEOUT)
        .map_err(catalog_error)
}

/// Enable `PRAGMA foreign_keys` on an autocommit catalog connection so `ON
/// DELETE CASCADE` clauses actually fire. The pragma is a silent no-op
/// inside a transaction, so the autocommit guard keeps a caller-managed
/// transaction's setting untouched.
fn ensure_catalog_foreign_keys(conn: &Connection) -> SearchResult<()> {
    if !conn.is_autocommit() {
        return Ok(());
    }
    retry_catalog_op(
        || {
            conn.execute_batch("PRAGMA foreign_keys=ON;")
                .map_err(catalog_error)
        },
        "catalog foreign_keys pragma",
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogIngestionClass {
    FullSemanticLexical,
    LexicalOnly,
    MetadataOnly,
    Skip,
}

impl CatalogIngestionClass {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullSemanticLexical => "full_semantic_lexical",
            Self::LexicalOnly => "lexical_only",
            Self::MetadataOnly => "metadata_only",
            Self::Skip => "skip",
        }
    }
}

impl From<crate::config::IngestionClass> for CatalogIngestionClass {
    fn from(value: crate::config::IngestionClass) -> Self {
        match value {
            crate::config::IngestionClass::FullSemanticLexical => Self::FullSemanticLexical,
            crate::config::IngestionClass::LexicalOnly => Self::LexicalOnly,
            crate::config::IngestionClass::MetadataOnly => Self::MetadataOnly,
            crate::config::IngestionClass::Skip => Self::Skip,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogPipelineStatus {
    Discovered,
    Queued,
    Embedding,
    Indexed,
    Failed,
    Skipped,
    Tombstoned,
}

impl CatalogPipelineStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Discovered => "discovered",
            Self::Queued => "queued",
            Self::Embedding => "embedding",
            Self::Indexed => "indexed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::Tombstoned => "tombstoned",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogChangeKind {
    Upsert,
    Reclassified,
    Status,
    Tombstone,
}

impl CatalogChangeKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Upsert => "upsert",
            Self::Reclassified => "reclassified",
            Self::Status => "status",
            Self::Tombstone => "tombstone",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDecision {
    /// Incoming row is exactly the next expected sequence and should be applied.
    ApplyNext { next_checkpoint: i64 },
    /// Incoming row was already applied or superseded.
    Duplicate { checkpoint: i64 },
    /// Replay stream has a gap and should pause for deterministic recovery.
    Gap {
        checkpoint: i64,
        expected_next: i64,
        observed: i64,
    },
}

/// Deterministic replay classifier used by consumers resuming after crash or
/// restart.
#[must_use]
pub const fn classify_replay_sequence(last_applied_seq: i64, incoming_seq: i64) -> ReplayDecision {
    if incoming_seq <= last_applied_seq {
        return ReplayDecision::Duplicate {
            checkpoint: last_applied_seq,
        };
    }

    let expected_next = last_applied_seq.saturating_add(1);
    if incoming_seq == expected_next {
        return ReplayDecision::ApplyNext {
            next_checkpoint: incoming_seq,
        };
    }

    ReplayDecision::Gap {
        checkpoint: last_applied_seq,
        expected_next,
        observed: incoming_seq,
    }
}

/// Bootstrap the fsfs catalog/changelog schema to the supported latest version.
///
/// # Errors
///
/// Returns an error if schema DDL fails, the version marker is invalid, or the
/// transaction cannot be committed.
pub fn bootstrap_catalog_schema(conn: &mut Connection) -> SearchResult<()> {
    ensure_catalog_foreign_keys(conn)?;
    retry_catalog_op(
        || bootstrap_catalog_schema_once(&mut *conn),
        "catalog schema bootstrap",
    )
}

fn bootstrap_catalog_schema_once(conn: &mut Connection) -> SearchResult<()> {
    // Read-idempotent fast path: a marker already stamped at the current
    // version needs no `BEGIN IMMEDIATE`, so repeat callers (every
    // `cleanup_tombstones_for_path` pass) do not dirty a B-tree page or
    // contend with `frankensearch-storage` writers on the shared database.
    if catalog_schema_version_state(conn)?.is_some_and(|version| version == CATALOG_SCHEMA_VERSION)
    {
        return Ok(());
    }
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(catalog_error)?;
    match bootstrap_catalog_schema_inner(&tx) {
        // `Transaction::commit` consumes the handle; a failed commit still
        // rolls back on drop, so there is no separate rollback path here.
        Ok(()) => tx.commit().map_err(catalog_error),
        Err(error) => match tx.rollback() {
            Ok(()) => Err(error),
            Err(rollback_err) => {
                tracing::warn!(
                    target: "frankensearch.fsfs.catalog",
                    error = %rollback_err,
                    "rollback failed after catalog schema bootstrap error"
                );
                Err(unretryable_catalog_rollback_error(&error, &rollback_err))
            }
        },
    }
}

fn bootstrap_catalog_schema_inner(conn: &Connection) -> SearchResult<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
    )
    .map_err(catalog_error)?;

    let mut version = current_catalog_schema_version_optional(conn)?.unwrap_or(0);
    if version == 0 {
        for statement in LATEST_SCHEMA {
            conn.execute_batch(statement).map_err(catalog_error)?;
        }

        conn.execute(
            "INSERT OR REPLACE INTO fsfs_catalog_schema_version(version) VALUES (?1);",
            rusqlite::params![CATALOG_SCHEMA_VERSION],
        )
        .map_err(catalog_error)?;
        version = current_catalog_schema_version_optional(conn)?.ok_or_else(|| {
            SearchError::SubsystemError {
                subsystem: SUBSYSTEM,
                source: Box::new(io::Error::other(
                    "fsfs_catalog_schema_version table has no rows",
                )),
            }
        })?;
    }

    if version > CATALOG_SCHEMA_VERSION {
        return Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "catalog schema version {version} is newer than supported {CATALOG_SCHEMA_VERSION}"
            ))),
        });
    }

    if version < CATALOG_SCHEMA_VERSION {
        return Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "legacy catalog schema version {version} is unsupported; rebuild catalog to schema version {CATALOG_SCHEMA_VERSION}"
            ))),
        });
    }

    // `version == CATALOG_SCHEMA_VERSION`: either the preflight fast path
    // raced a concurrent bootstrap, or the fresh-schema branch above already
    // stamped the marker inside this transaction. Either way the marker row
    // is correct and must not be rewritten.
    Ok(())
}

/// Marker-table read for the bootstrap fast path: `None` when the table is
/// absent or holds no row, `Some(version)` otherwise. Unlike
/// `current_catalog_schema_version` this never errors on a missing table —
/// bootstrap itself creates it.
fn catalog_schema_version_state(conn: &Connection) -> SearchResult<Option<i64>> {
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master \
             WHERE type = 'table' AND name = 'fsfs_catalog_schema_version';",
            [],
            |row| row.get(0),
        )
        .map_err(catalog_error)?;
    if !table_exists {
        return Ok(None);
    }
    current_catalog_schema_version_optional(conn)
}

/// Return the current catalog schema version marker.
///
/// # Errors
///
/// Returns an error if the version table is missing/corrupt or cannot be
/// queried.
pub fn current_catalog_schema_version(conn: &Connection) -> SearchResult<i64> {
    retry_catalog_op(
        || {
            current_catalog_schema_version_optional(conn)?.ok_or_else(|| {
                SearchError::SubsystemError {
                    subsystem: SUBSYSTEM,
                    source: Box::new(io::Error::other(
                        "fsfs_catalog_schema_version table has no rows",
                    )),
                }
            })
        },
        "catalog schema version",
    )
}

fn current_catalog_schema_version_optional(conn: &Connection) -> SearchResult<Option<i64>> {
    conn.query_row(
        "SELECT version FROM fsfs_catalog_schema_version ORDER BY version DESC LIMIT 1;",
        [],
        |row| row.get::<_, i64>(0),
    )
    .optional()
    .map_err(catalog_error)
}

#[cfg(test)]
fn row_i64(row: &[rusqlite::types::Value], index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(rusqlite::types::Value::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}",
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn catalog_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(source),
    }
}

fn is_retryable_sqlite_error(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(inner, _)
            if matches!(
                inner.code,
                rusqlite::ffi::ErrorCode::DatabaseBusy
                    | rusqlite::ffi::ErrorCode::DatabaseLocked
                    | rusqlite::ffi::ErrorCode::SchemaChanged
            )
    )
}

fn is_retryable_catalog_error(error: &SearchError) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(err) = current {
        if let Some(sqlite) = err.downcast_ref::<rusqlite::Error>() {
            return is_retryable_sqlite_error(sqlite);
        }
        current = err.source();
    }
    false
}

fn unretryable_catalog_rollback_error(
    original: &SearchError,
    rollback_err: &rusqlite::Error,
) -> SearchError {
    SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "rollback failed ({rollback_err}); original error was: {original}"
        ))),
    }
}

fn retry_catalog_op<T>(
    mut op: impl FnMut() -> SearchResult<T>,
    context: &'static str,
) -> SearchResult<T> {
    let mut attempt = 1;
    loop {
        match op() {
            Ok(value) => return Ok(value),
            Err(error)
                if is_retryable_catalog_error(&error) && attempt < CATALOG_OPEN_MAX_ATTEMPTS =>
            {
                let growth = attempt.saturating_sub(1).min(6);
                let delay_ms = (CATALOG_OPEN_RETRY_BASE_DELAY_MS << growth)
                    .min(CATALOG_OPEN_RETRY_MAX_DELAY_MS);
                tracing::debug!(
                    target: "frankensearch.fsfs.catalog",
                    context,
                    next_attempt = attempt.saturating_add(1),
                    max_attempts = CATALOG_OPEN_MAX_ATTEMPTS,
                    delay_ms,
                    error = %error,
                    "retrying transient catalog contention"
                );
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CATALOG_SCHEMA_VERSION, CHANGELOG_REPLAY_BATCH_SQL, CLEANUP_TOMBSTONES_SQL,
        DIRTY_CATALOG_LOOKUP_SQL, INDEX_CATALOG_CLEANUP, INDEX_CATALOG_CONTENT_HASH,
        INDEX_CATALOG_DIRTY_LOOKUP, INDEX_CATALOG_REVISIONS, INDEX_CHANGELOG_FILE_REVISION,
        INDEX_CHANGELOG_PENDING_APPLY, INDEX_CHANGELOG_REPLAY, ReplayDecision,
        bootstrap_catalog_schema, catalog_error, classify_replay_sequence, cleanup_tombstones,
        cleanup_tombstones_for_path, current_catalog_schema_version,
    };
    use rusqlite::types::Value as SqliteValue;
    use rusqlite::{Connection, ffi};

    fn sqlite_failure(code: std::ffi::c_int) -> rusqlite::Error {
        rusqlite::Error::SqliteFailure(ffi::Error::new(code), None)
    }

    fn query_rows(
        conn: &Connection,
        sql: &str,
        params: &[SqliteValue],
    ) -> super::SearchResult<Vec<Vec<SqliteValue>>> {
        let mut statement = conn.prepare(sql).map_err(catalog_error)?;
        let column_count = statement.column_count();
        let rows = statement
            .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                (0..column_count)
                    .map(|index| row.get::<_, SqliteValue>(index))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(catalog_error)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(catalog_error)
    }

    fn table_exists(conn: &Connection, table_name: &str) -> bool {
        // Probe table existence with a zero-row SELECT: it proves the table
        // is actually readable, not merely listed in sqlite_master.
        query_rows(
            conn,
            &format!("SELECT 1 FROM \"{table_name}\" LIMIT 0"),
            &[],
        )
        .is_ok()
    }

    fn index_exists(conn: &Connection, table_name: &str, index_name: &str) -> bool {
        // Ask the schema catalog directly. The former `INDEXED BY` probe
        // was not stock-conformant: SQLite rejects `INDEXED BY` on a partial
        // index when the query does not carry the index's WHERE predicate
        // ("no query solution"), so every partial index read as missing.
        let params = [
            SqliteValue::Text(table_name.to_owned()),
            SqliteValue::Text(index_name.to_owned()),
        ];
        query_rows(
            conn,
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND tbl_name = ?1 AND name = ?2",
            &params,
        )
        .map(|rows| !rows.is_empty())
        .unwrap_or(false)
    }

    #[test]
    fn bootstrap_creates_catalog_tables_and_indexes() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");

        bootstrap_catalog_schema(&mut conn).expect("catalog bootstrap should succeed");

        assert_eq!(
            current_catalog_schema_version(&conn).expect("version should exist"),
            CATALOG_SCHEMA_VERSION
        );
        assert!(table_exists(&conn, "fsfs_catalog_files"));
        assert!(table_exists(&conn, "fsfs_catalog_changelog"));
        assert!(table_exists(&conn, "fsfs_catalog_replay_checkpoint"));

        for index in [
            INDEX_CATALOG_DIRTY_LOOKUP,
            INDEX_CATALOG_REVISIONS,
            INDEX_CATALOG_CLEANUP,
            INDEX_CATALOG_CONTENT_HASH,
        ] {
            assert!(
                index_exists(&conn, "fsfs_catalog_files", index),
                "missing required index {index}"
            );
        }
        for index in [
            INDEX_CHANGELOG_REPLAY,
            INDEX_CHANGELOG_FILE_REVISION,
            INDEX_CHANGELOG_PENDING_APPLY,
        ] {
            assert!(
                index_exists(&conn, "fsfs_catalog_changelog", index),
                "missing required index {index}"
            );
        }
    }

    #[test]
    fn bootstrap_is_idempotent() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");

        bootstrap_catalog_schema(&mut conn).expect("first bootstrap should succeed");
        bootstrap_catalog_schema(&mut conn).expect("second bootstrap should succeed");
        bootstrap_catalog_schema(&mut conn).expect("third bootstrap should succeed");

        assert_eq!(
            current_catalog_schema_version(&conn).expect("version should exist"),
            CATALOG_SCHEMA_VERSION
        );
    }

    #[test]
    fn bootstrap_rejects_legacy_schema_versions() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("schema version table should create");
        conn.execute_batch("INSERT INTO fsfs_catalog_schema_version(version) VALUES (-1);")
            .expect("legacy row should insert");

        let error =
            bootstrap_catalog_schema(&mut conn).expect_err("legacy schema should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("legacy catalog schema version -1 is unsupported"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn bootstrap_rejects_future_schema_versions() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("schema version table should create");
        let params = [SqliteValue::Integer(CATALOG_SCHEMA_VERSION + 10)];
        conn.execute(
            "INSERT INTO fsfs_catalog_schema_version(version) VALUES (?1);",
            params,
        )
        .expect("future version row should insert");

        let error =
            bootstrap_catalog_schema(&mut conn).expect_err("future schema should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("newer than supported"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn replay_classifier_is_deterministic_for_next_duplicate_and_gap() {
        assert_eq!(
            classify_replay_sequence(41, 42),
            ReplayDecision::ApplyNext {
                next_checkpoint: 42
            }
        );
        assert_eq!(
            classify_replay_sequence(41, 41),
            ReplayDecision::Duplicate { checkpoint: 41 }
        );
        assert_eq!(
            classify_replay_sequence(41, 45),
            ReplayDecision::Gap {
                checkpoint: 41,
                expected_next: 42,
                observed: 45
            }
        );
    }

    // ── Replay classifier edge cases ──────────────────────────────────

    #[test]
    fn replay_classifier_earlier_sequence_is_duplicate() {
        assert_eq!(
            classify_replay_sequence(100, 50),
            ReplayDecision::Duplicate { checkpoint: 100 }
        );
    }

    #[test]
    fn replay_classifier_zero_checkpoint_and_seq_one() {
        assert_eq!(
            classify_replay_sequence(0, 1),
            ReplayDecision::ApplyNext { next_checkpoint: 1 }
        );
    }

    #[test]
    fn replay_classifier_i64_max_checkpoint_saturates() {
        // saturating_add(1) on i64::MAX stays at i64::MAX
        // incoming = i64::MAX is the expected_next, so it should be ApplyNext
        assert_eq!(
            classify_replay_sequence(i64::MAX - 1, i64::MAX),
            ReplayDecision::ApplyNext {
                next_checkpoint: i64::MAX
            }
        );
    }

    // ── Enum as_str round-trips ─────────────────────────────────────

    #[test]
    fn ingestion_class_as_str_covers_all_variants() {
        use super::{CatalogChangeKind, CatalogIngestionClass, CatalogPipelineStatus};

        let classes = [
            (
                CatalogIngestionClass::FullSemanticLexical,
                "full_semantic_lexical",
            ),
            (CatalogIngestionClass::LexicalOnly, "lexical_only"),
            (CatalogIngestionClass::MetadataOnly, "metadata_only"),
            (CatalogIngestionClass::Skip, "skip"),
        ];
        for (variant, expected) in classes {
            assert_eq!(variant.as_str(), expected);
        }

        let statuses = [
            (CatalogPipelineStatus::Discovered, "discovered"),
            (CatalogPipelineStatus::Queued, "queued"),
            (CatalogPipelineStatus::Embedding, "embedding"),
            (CatalogPipelineStatus::Indexed, "indexed"),
            (CatalogPipelineStatus::Failed, "failed"),
            (CatalogPipelineStatus::Skipped, "skipped"),
            (CatalogPipelineStatus::Tombstoned, "tombstoned"),
        ];
        for (variant, expected) in statuses {
            assert_eq!(variant.as_str(), expected);
        }

        let kinds = [
            (CatalogChangeKind::Upsert, "upsert"),
            (CatalogChangeKind::Reclassified, "reclassified"),
            (CatalogChangeKind::Status, "status"),
            (CatalogChangeKind::Tombstone, "tombstone"),
        ];
        for (variant, expected) in kinds {
            assert_eq!(variant.as_str(), expected);
        }
    }

    // ── Schema version edge cases ───────────────────────────────────

    #[test]
    fn current_version_errors_when_table_has_no_rows() {
        let conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("create table");
        let error = current_catalog_schema_version(&conn).expect_err("should error with no rows");
        assert!(error.to_string().contains("no rows"));
    }

    // ── Original tests continue ─────────────────────────────────────

    #[test]
    fn incremental_workload_queries_execute_and_have_index_support() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap_catalog_schema(&mut conn).expect("catalog bootstrap should succeed");

        for index in [
            INDEX_CATALOG_DIRTY_LOOKUP,
            INDEX_CATALOG_REVISIONS,
            INDEX_CATALOG_CLEANUP,
            INDEX_CATALOG_CONTENT_HASH,
        ] {
            assert!(
                index_exists(&conn, "fsfs_catalog_files", index),
                "catalog index {index} should exist after bootstrap"
            );
        }

        for index in [
            INDEX_CHANGELOG_REPLAY,
            INDEX_CHANGELOG_FILE_REVISION,
            INDEX_CHANGELOG_PENDING_APPLY,
        ] {
            assert!(
                index_exists(&conn, "fsfs_catalog_changelog", index),
                "changelog index {index} should exist after bootstrap"
            );
        }

        let now = 1_710_000_000_000_i64;
        let file_params = [
            SqliteValue::Text("home:/tmp/a.txt".to_owned()),
            SqliteValue::Text("home".to_owned()),
            SqliteValue::Text("/tmp/a.txt".to_owned()),
            SqliteValue::Blob(vec![7_u8; 32]),
            SqliteValue::Integer(3),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("queued".to_owned()),
            SqliteValue::Integer(1),
            SqliteValue::Integer(now - 1_000),
            SqliteValue::Integer(now),
            SqliteValue::Integer(now),
        ];
        conn.execute("INSERT INTO fsfs_catalog_files \
             (file_key, mount_id, canonical_path, content_hash, revision, ingestion_class, pipeline_status, eligible, first_seen_ts, last_seen_ts, updated_ts) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11);", file_params)
        .expect("catalog seed row should insert");

        let changelog_params = [
            SqliteValue::Integer(1),
            SqliteValue::Text("home:/tmp/a.txt".to_owned()),
            SqliteValue::Integer(3),
            SqliteValue::Text("upsert".to_owned()),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("queued".to_owned()),
            SqliteValue::Blob(vec![7_u8; 32]),
            SqliteValue::Integer(now),
            SqliteValue::Text("corr-1".to_owned()),
            SqliteValue::Text("token-1".to_owned()),
        ];
        conn.execute("INSERT INTO fsfs_catalog_changelog \
             (stream_seq, file_key, revision, change_kind, ingestion_class, pipeline_status, content_hash, event_ts, correlation_id, replay_token) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);", changelog_params)
        .expect("changelog seed row should insert");

        let dirty_rows = query_rows(&conn, DIRTY_CATALOG_LOOKUP_SQL, &[SqliteValue::Integer(50)])
            .expect("dirty catalog lookup should execute");
        assert_eq!(dirty_rows.len(), 1);

        let replay_rows = query_rows(
            &conn,
            CHANGELOG_REPLAY_BATCH_SQL,
            &[SqliteValue::Integer(0), SqliteValue::Integer(100)],
        )
        .expect("replay batch query should execute");
        assert_eq!(replay_rows.len(), 1);

        conn.execute("UPDATE fsfs_catalog_files SET pipeline_status = 'tombstoned', deleted_ts = ?1 WHERE file_key = ?2;", [SqliteValue::Integer(now), SqliteValue::Text("home:/tmp/a.txt".to_owned())])
        .expect("tombstone update should succeed");
        conn.execute(CLEANUP_TOMBSTONES_SQL, [SqliteValue::Integer(now)])
            .expect("cleanup query should execute");

        let remaining = query_rows(&conn, "SELECT file_key FROM fsfs_catalog_files;", &[])
            .expect("remaining rows query should execute");
        assert!(remaining.is_empty(), "tombstone cleanup should remove row");

        let changelog_remaining =
            query_rows(&conn, "SELECT stream_seq FROM fsfs_catalog_changelog;", &[])
                .expect("changelog remaining rows query should execute");
        assert!(
            changelog_remaining.is_empty(),
            "deleting the tombstoned file row must cascade to fsfs_catalog_changelog"
        );
    }

    // ─── bd-1vl1 tests begin ───

    #[test]
    fn ingestion_class_debug_clone_copy_eq() {
        use super::CatalogIngestionClass;
        let a = CatalogIngestionClass::FullSemanticLexical;
        let b = a; // Copy
        assert_eq!(a, b);
        #[allow(clippy::clone_on_copy)]
        let c = a.clone();
        assert_eq!(a, c);
        let debug = format!("{a:?}");
        assert!(debug.contains("FullSemanticLexical"));
    }

    #[test]
    fn pipeline_status_debug_clone_copy_eq() {
        use super::CatalogPipelineStatus;
        let a = CatalogPipelineStatus::Embedding;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, CatalogPipelineStatus::Indexed);
        let debug = format!("{a:?}");
        assert!(debug.contains("Embedding"));
    }

    #[test]
    fn change_kind_debug_clone_copy_eq() {
        use super::CatalogChangeKind;
        let a = CatalogChangeKind::Tombstone;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, CatalogChangeKind::Upsert);
        let debug = format!("{a:?}");
        assert!(debug.contains("Tombstone"));
    }

    #[test]
    fn replay_decision_debug_clone_copy_eq() {
        let a = ReplayDecision::ApplyNext { next_checkpoint: 5 };
        let b = a;
        assert_eq!(a, b);
        let debug = format!("{a:?}");
        assert!(debug.contains("ApplyNext"));

        let dup = ReplayDecision::Duplicate { checkpoint: 3 };
        assert_ne!(a, dup);
        let debug_dup = format!("{dup:?}");
        assert!(debug_dup.contains("Duplicate"));

        let gap = ReplayDecision::Gap {
            checkpoint: 1,
            expected_next: 2,
            observed: 5,
        };
        let debug_gap = format!("{gap:?}");
        assert!(debug_gap.contains("Gap"));
    }

    #[test]
    fn ingestion_class_from_config_all_variants() {
        use super::CatalogIngestionClass;
        use crate::config::IngestionClass;

        assert_eq!(
            CatalogIngestionClass::from(IngestionClass::FullSemanticLexical),
            CatalogIngestionClass::FullSemanticLexical
        );
        assert_eq!(
            CatalogIngestionClass::from(IngestionClass::LexicalOnly),
            CatalogIngestionClass::LexicalOnly
        );
        assert_eq!(
            CatalogIngestionClass::from(IngestionClass::MetadataOnly),
            CatalogIngestionClass::MetadataOnly
        );
        assert_eq!(
            CatalogIngestionClass::from(IngestionClass::Skip),
            CatalogIngestionClass::Skip
        );
    }

    #[test]
    fn row_i64_unexpected_type_is_error() {
        use super::row_i64;

        let conn = Connection::open_in_memory().expect("in-memory connection");
        let rows =
            query_rows(&conn, "SELECT 'not_an_integer';", &[]).expect("text query should succeed");
        let row = &rows[0];
        let err = row_i64(row, 0, "test_field").expect_err("text should fail");
        let msg = err.to_string();
        assert!(msg.contains("unexpected type") || msg.contains("test_field"));
    }

    #[test]
    fn row_i64_missing_column_is_error() {
        use super::row_i64;

        let conn = Connection::open_in_memory().expect("in-memory connection");
        let rows = query_rows(&conn, "SELECT 1;", &[]).expect("query should succeed");
        let row = &rows[0];
        // Column index 5 doesn't exist (only column 0 present)
        let err = row_i64(row, 5, "missing_col").expect_err("missing column should fail");
        let msg = err.to_string();
        assert!(msg.contains("missing") || msg.contains("missing_col"));
    }

    #[test]
    fn row_i64_valid_integer() {
        use super::row_i64;

        let conn = Connection::open_in_memory().expect("in-memory connection");
        let rows = query_rows(&conn, "SELECT 42;", &[]).expect("query should succeed");
        let row = &rows[0];
        let value = row_i64(row, 0, "version").expect("integer should succeed");
        assert_eq!(value, 42);
    }

    #[test]
    fn catalog_error_helper_produces_subsystem_error() {
        let source = std::io::Error::other("test error");
        let err = catalog_error(source);
        match err {
            frankensearch_core::SearchError::SubsystemError {
                subsystem, source, ..
            } => {
                assert_eq!(subsystem, "fsfs_catalog");
                assert!(source.to_string().contains("test error"));
            }
            other => panic!("expected SubsystemError, got {other:?}"),
        }
    }

    #[test]
    fn catalog_retry_accepts_transient_sqlite_errors() {
        for code in [
            ffi::SQLITE_BUSY,
            ffi::SQLITE_BUSY_SNAPSHOT,
            ffi::SQLITE_LOCKED,
        ] {
            assert!(
                super::is_retryable_catalog_error(&catalog_error(sqlite_failure(code))),
                "code {code}"
            );
        }
        assert!(super::is_retryable_catalog_error(&catalog_error(
            sqlite_failure(ffi::SQLITE_SCHEMA)
        )));
        assert!(!super::is_retryable_catalog_error(&catalog_error(
            sqlite_failure(ffi::SQLITE_CONSTRAINT)
        )));
        assert!(!super::is_retryable_catalog_error(
            &super::unretryable_catalog_rollback_error(
                &catalog_error(sqlite_failure(ffi::SQLITE_BUSY)),
                &sqlite_failure(ffi::SQLITE_BUSY),
            )
        ));
    }

    #[test]
    fn schema_version_constant_is_positive() {
        const { assert!(CATALOG_SCHEMA_VERSION >= 1) };
    }

    #[test]
    fn index_name_constants_are_non_empty() {
        let names = [
            INDEX_CATALOG_DIRTY_LOOKUP,
            INDEX_CATALOG_REVISIONS,
            INDEX_CATALOG_CLEANUP,
            INDEX_CATALOG_CONTENT_HASH,
            INDEX_CHANGELOG_REPLAY,
            INDEX_CHANGELOG_FILE_REVISION,
            INDEX_CHANGELOG_PENDING_APPLY,
        ];
        for name in names {
            assert!(!name.is_empty());
            assert!(name.starts_with("idx_"));
        }
    }

    #[test]
    fn sql_constants_contain_expected_keywords() {
        assert!(DIRTY_CATALOG_LOOKUP_SQL.contains("fsfs_catalog_files"));
        assert!(DIRTY_CATALOG_LOOKUP_SQL.contains("pipeline_status"));
        assert!(DIRTY_CATALOG_LOOKUP_SQL.contains("LIMIT"));

        assert!(CHANGELOG_REPLAY_BATCH_SQL.contains("fsfs_catalog_changelog"));
        assert!(CHANGELOG_REPLAY_BATCH_SQL.contains("stream_seq"));

        assert!(CLEANUP_TOMBSTONES_SQL.contains("DELETE"));
        assert!(CLEANUP_TOMBSTONES_SQL.contains("tombstoned"));
    }

    #[test]
    fn cleanup_tombstones_executes_sql_against_existing_connection() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap_catalog_schema(&mut conn).expect("catalog bootstrap should succeed");

        let now = 1_710_000_000_000_i64;
        let params = [
            SqliteValue::Text("home:/tmp/a.txt".to_owned()),
            SqliteValue::Text("home".to_owned()),
            SqliteValue::Text("/tmp/a.txt".to_owned()),
            SqliteValue::Blob(vec![7_u8; 32]),
            SqliteValue::Integer(3),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("tombstoned".to_owned()),
            SqliteValue::Integer(1),
            SqliteValue::Integer(now - 1_000),
            SqliteValue::Integer(now),
            SqliteValue::Integer(now),
            SqliteValue::Integer(now),
        ];
        conn.execute("INSERT INTO fsfs_catalog_files \
             (file_key, mount_id, canonical_path, content_hash, revision, ingestion_class, pipeline_status, eligible, first_seen_ts, last_seen_ts, updated_ts, deleted_ts) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12);", params)
        .expect("seed tombstone row");

        let removed = cleanup_tombstones(&conn, now).expect("cleanup SQL should execute");
        assert_eq!(removed, 1);
    }

    #[test]
    fn cleanup_tombstones_for_path_prunes_old_tombstones() {
        let temp = tempfile::tempdir().expect("tempdir");
        let db_path = temp.path().join("catalog.db");
        let mut conn = Connection::open(&db_path).expect("open sqlite file");
        bootstrap_catalog_schema(&mut conn).expect("catalog bootstrap should succeed");

        let now = 1_710_000_000_000_i64;
        let old_cutoff = now - 10_000;

        let old_tombstone = [
            SqliteValue::Text("home:/tmp/old.txt".to_owned()),
            SqliteValue::Text("home".to_owned()),
            SqliteValue::Text("/tmp/old.txt".to_owned()),
            SqliteValue::Blob(vec![1_u8; 32]),
            SqliteValue::Integer(1),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("tombstoned".to_owned()),
            SqliteValue::Integer(1),
            SqliteValue::Integer(now - 20_000),
            SqliteValue::Integer(now - 15_000),
            SqliteValue::Integer(now - 15_000),
            SqliteValue::Integer(now - 15_000),
        ];
        conn.execute("INSERT INTO fsfs_catalog_files \
             (file_key, mount_id, canonical_path, content_hash, revision, ingestion_class, pipeline_status, eligible, first_seen_ts, last_seen_ts, updated_ts, deleted_ts) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12);", old_tombstone)
        .expect("old tombstone seed should insert");

        let fresh_tombstone = [
            SqliteValue::Text("home:/tmp/fresh.txt".to_owned()),
            SqliteValue::Text("home".to_owned()),
            SqliteValue::Text("/tmp/fresh.txt".to_owned()),
            SqliteValue::Blob(vec![2_u8; 32]),
            SqliteValue::Integer(1),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("tombstoned".to_owned()),
            SqliteValue::Integer(1),
            SqliteValue::Integer(now - 9_000),
            SqliteValue::Integer(now - 5_000),
            SqliteValue::Integer(now - 5_000),
            SqliteValue::Integer(now - 5_000),
        ];
        conn.execute("INSERT INTO fsfs_catalog_files \
             (file_key, mount_id, canonical_path, content_hash, revision, ingestion_class, pipeline_status, eligible, first_seen_ts, last_seen_ts, updated_ts, deleted_ts) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12);", fresh_tombstone)
        .expect("fresh tombstone seed should insert");

        for (stream_seq, file_key) in [(1_i64, "home:/tmp/old.txt"), (2_i64, "home:/tmp/fresh.txt")]
        {
            conn.execute("INSERT INTO fsfs_catalog_changelog \
                 (stream_seq, file_key, revision, change_kind, ingestion_class, pipeline_status, content_hash, event_ts, correlation_id, replay_token) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);",
                rusqlite::params![
                    stream_seq,
                    file_key,
                    1_i64,
                    "tombstone",
                    "full_semantic_lexical",
                    "tombstoned",
                    vec![9_u8; 32],
                    now,
                    format!("corr-{stream_seq}"),
                    format!("token-{stream_seq}"),
                ],
            )
            .expect("changelog seed row should insert");
        }

        let removed =
            cleanup_tombstones_for_path(&db_path, old_cutoff).expect("cleanup helper should work");
        assert_eq!(removed, 1, "only old tombstones should be removed");

        // Reopen connection for verification — `cleanup_tombstones_for_path`
        // ran its DELETE through a separate connection.
        drop(conn);
        let conn2 = Connection::open(&db_path).expect("reopen for verification");
        let remaining = query_rows(
            &conn2,
            "SELECT file_key FROM fsfs_catalog_files ORDER BY file_key;",
            &[],
        )
        .expect("remaining rows query should execute");
        assert_eq!(remaining.len(), 1);
        assert_eq!(
            remaining[0].first(),
            Some(&SqliteValue::Text("home:/tmp/fresh.txt".to_owned()))
        );

        // The old tombstone's changelog row must have cascaded away with its
        // parent; the surviving file keeps its own changelog history.
        let changelog_rows = query_rows(
            &conn2,
            "SELECT file_key FROM fsfs_catalog_changelog ORDER BY stream_seq;",
            &[],
        )
        .expect("changelog rows query should execute");
        assert_eq!(changelog_rows.len(), 1);
        assert_eq!(
            changelog_rows[0].first(),
            Some(&SqliteValue::Text("home:/tmp/fresh.txt".to_owned()))
        );
    }

    #[test]
    fn replay_classifier_gap_with_seq_two_above() {
        let result = classify_replay_sequence(10, 12);
        assert_eq!(
            result,
            ReplayDecision::Gap {
                checkpoint: 10,
                expected_next: 11,
                observed: 12,
            }
        );
    }

    #[test]
    fn replay_classifier_negative_sequences() {
        assert_eq!(
            classify_replay_sequence(-5, -4),
            ReplayDecision::ApplyNext {
                next_checkpoint: -4
            }
        );
        assert_eq!(
            classify_replay_sequence(-5, -5),
            ReplayDecision::Duplicate { checkpoint: -5 }
        );
        assert_eq!(
            classify_replay_sequence(-5, -3),
            ReplayDecision::Gap {
                checkpoint: -5,
                expected_next: -4,
                observed: -3,
            }
        );
    }

    #[test]
    fn replay_classifier_zero_zero_is_duplicate() {
        assert_eq!(
            classify_replay_sequence(0, 0),
            ReplayDecision::Duplicate { checkpoint: 0 }
        );
    }

    // ─── bd-1vl1 tests end ───
}
