use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use frankensearch_core::{SearchError, SearchResult};
use rusqlite::{Connection, Row, TransactionBehavior};

use crate::connection::{map_storage_error_at, retry_transient_storage};

pub const SCHEMA_VERSION: i64 = 1;

/// Governed marker table for the FTS5 Porter tokenizer rebuild.
///
/// This belongs to the storage schema rather than to the rebuild operation so
/// that a failed or cancelled rebuild cannot leave behind ad-hoc schema.
#[cfg(feature = "fts5")]
pub(crate) const PORTER_FTS5_REBUILD_TABLE: &str = "frankensearch_fts5_rebuild_version";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchemaVersionState {
    MissingTable,
    Empty,
    Present(i64),
}

/// Canonical v1 baseline schema applied to every new database.
///
/// There is exactly one schema version: the collapsed final form of the
/// retired prototype migrations. No historical migration path exists; any
/// `schema_version` marker other than `1` fails closed.
const SCHEMA_V1: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS documents (\
        doc_id TEXT PRIMARY KEY,\
        source_path TEXT,\
        content_preview TEXT NOT NULL,\
        content_hash BLOB NOT NULL,\
        content_length INTEGER NOT NULL,\
        created_at INTEGER NOT NULL,\
        updated_at INTEGER NOT NULL,\
        metadata_json TEXT\
    );",
    "CREATE TABLE IF NOT EXISTS embedding_jobs (\
        job_id INTEGER PRIMARY KEY AUTOINCREMENT,\
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
        embedder_id TEXT NOT NULL,\
        priority INTEGER NOT NULL DEFAULT 0,\
        submitted_at INTEGER NOT NULL,\
        started_at INTEGER,\
        completed_at INTEGER,\
        status TEXT NOT NULL DEFAULT 'pending',\
        retry_count INTEGER NOT NULL DEFAULT 0,\
        max_retries INTEGER NOT NULL DEFAULT 3,\
        error_message TEXT,\
        content_hash BLOB,\
        worker_id TEXT,\
        claim_epoch INTEGER NOT NULL DEFAULT 0,\
        UNIQUE(doc_id, embedder_id, status)\
    );",
    "CREATE TABLE IF NOT EXISTS embedding_status (\
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
        embedder_id TEXT NOT NULL,\
        embedder_revision TEXT,\
        status TEXT NOT NULL DEFAULT 'pending',\
        embedded_at INTEGER,\
        error_message TEXT,\
        retry_count INTEGER NOT NULL DEFAULT 0,\
        PRIMARY KEY(doc_id, embedder_id)\
    );",
    "CREATE TABLE IF NOT EXISTS content_hashes (\
        content_hash TEXT PRIMARY KEY,\
        first_doc_id TEXT NOT NULL,\
        seen_count INTEGER NOT NULL DEFAULT 1,\
        first_seen_at INTEGER NOT NULL,\
        last_seen_at INTEGER NOT NULL\
    );",
    "CREATE TABLE IF NOT EXISTS index_metadata (\
        index_name TEXT PRIMARY KEY,\
        index_type TEXT NOT NULL,\
        embedder_id TEXT NOT NULL,\
        embedder_revision TEXT,\
        dimension INTEGER NOT NULL,\
        record_count INTEGER NOT NULL DEFAULT 0,\
        file_path TEXT,\
        file_size_bytes INTEGER,\
        file_hash TEXT,\
        schema_version INTEGER,\
        built_at INTEGER,\
        build_duration_ms INTEGER,\
        source_doc_count INTEGER NOT NULL DEFAULT 0,\
        config_json TEXT,\
        fec_path TEXT,\
        fec_size_bytes INTEGER,\
        last_verified_at INTEGER,\
        last_repair_at INTEGER,\
        repair_count INTEGER NOT NULL DEFAULT 0,\
        mean_norm REAL,\
        variance REAL\
    );",
    "CREATE TABLE IF NOT EXISTS index_build_history (\
        build_id INTEGER PRIMARY KEY AUTOINCREMENT,\
        index_name TEXT NOT NULL,\
        built_at INTEGER NOT NULL,\
        build_duration_ms INTEGER NOT NULL,\
        record_count INTEGER NOT NULL,\
        source_doc_count INTEGER NOT NULL,\
        \"trigger\" TEXT NOT NULL,\
        config_json TEXT,\
        notes TEXT,\
        mean_norm REAL,\
        variance REAL\
    );",
    "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_embedding_status_pending ON embedding_status(status, doc_id) WHERE status = 'pending';",
    "CREATE INDEX IF NOT EXISTS idx_jobs_pending ON embedding_jobs(status, priority DESC, submitted_at ASC) WHERE status = 'pending';",
    "CREATE INDEX IF NOT EXISTS idx_jobs_processing ON embedding_jobs(status, started_at) WHERE status = 'processing';",
    "CREATE INDEX IF NOT EXISTS idx_build_history_index ON index_build_history(index_name, built_at DESC);",
    "CREATE TABLE IF NOT EXISTS search_history (\
        id INTEGER PRIMARY KEY AUTOINCREMENT,\
        query TEXT NOT NULL,\
        query_class TEXT,\
        result_count INTEGER,\
        phase1_latency_ms INTEGER,\
        phase2_latency_ms INTEGER,\
        top_results_json TEXT,\
        searched_at INTEGER NOT NULL\
    );",
    "CREATE INDEX IF NOT EXISTS idx_history_query ON search_history(query);",
    "CREATE INDEX IF NOT EXISTS idx_history_ts ON search_history(searched_at DESC);",
    "CREATE TABLE IF NOT EXISTS bookmarks (\
        id INTEGER PRIMARY KEY AUTOINCREMENT,\
        doc_id TEXT NOT NULL,\
        query TEXT,\
        note TEXT,\
        created_at INTEGER NOT NULL\
    );",
    "CREATE INDEX IF NOT EXISTS idx_bookmarks_doc_query ON bookmarks(doc_id, query);",
    "CREATE TABLE IF NOT EXISTS frankensearch_fts5_rebuild_version (\
        table_name TEXT PRIMARY KEY,\
        rebuild_version INTEGER NOT NULL\
    );",
];

pub fn bootstrap(conn: &mut Connection) -> SearchResult<()> {
    match schema_version_state(conn, "schema preflight")? {
        SchemaVersionState::Present(version) => {
            reject_future_version(version, "schema preflight")?;
            if version == SCHEMA_VERSION {
                tracing::trace!(
                    target: "frankensearch.storage",
                    schema_version = version,
                    "storage schema is current; skipping write transaction"
                );
                return Ok(());
            }
        }
        SchemaVersionState::MissingTable | SchemaVersionState::Empty => {}
    }

    retry_transient_storage(
        || bootstrap_write_transaction(&mut *conn),
        "schema bootstrap",
    )
}

fn bootstrap_write_transaction(conn: &mut Connection) -> SearchResult<()> {
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| map_storage_error_at("schema transaction begin", error))?;
    let result = catch_unwind(AssertUnwindSafe(|| bootstrap_inner(&tx)));
    match result {
        Ok(Ok(())) => tx
            .commit()
            .map_err(|commit_err| map_storage_error_at("schema transaction commit", commit_err)),
        Ok(Err(error)) => match tx.rollback() {
            Ok(()) => Err(error),
            Err(rollback_err) => {
                tracing::warn!(
                    target: "frankensearch.storage",
                    stage = "schema transaction rollback after bootstrap",
                    error = %rollback_err,
                    "rollback failed after schema bootstrap error"
                );
                Err(crate::connection::unretryable_rollback_error(
                    &error,
                    &rollback_err,
                ))
            }
        },
        Err(payload) => {
            if let Err(rollback_err) = tx.rollback() {
                tracing::error!(
                    target: "frankensearch.storage",
                    stage = "schema transaction rollback after panic",
                    error = %rollback_err,
                    "critical: rollback failed during schema bootstrap panic recovery"
                );
            }
            resume_unwind(payload);
        }
    }
}

fn bootstrap_inner(conn: &Connection) -> SearchResult<()> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);")
        .map_err(|error| map_storage_error_at("schema transaction marker table", error))?;

    let version = current_version_optional_at(conn, "schema transaction recheck")?.unwrap_or(0);
    reject_future_version(version, "schema transaction recheck")?;

    if version == SCHEMA_VERSION {
        // A concurrent bootstrap committed between our preflight read and
        // this BEGIN IMMEDIATE. The schema is already at target; treat the
        // write transaction as an idempotent no-op.
        return Ok(());
    }

    if version == 0 {
        tracing::debug!(
            target: "frankensearch.storage",
            to_version = SCHEMA_VERSION,
            "bootstrapping fresh storage database to v1 baseline schema"
        );

        for statement in SCHEMA_V1 {
            conn.execute_batch(statement)
                .map_err(|error| map_storage_error_at("fresh schema application", error))?;
        }

        conn.execute(
            "INSERT OR REPLACE INTO schema_version(version) VALUES (?1);",
            rusqlite::params![SCHEMA_VERSION],
        )
        .map_err(|error| map_storage_error_at("fresh schema marker write", error))?;

        // `PRAGMA user_version` is stamped inside the same `BEGIN IMMEDIATE`
        // transaction, before `PRAGMA journal_mode = WAL` is enabled.
        conn.execute_batch(&format!("PRAGMA user_version={SCHEMA_VERSION};"))
            .map_err(|error| map_storage_error_at("fresh schema user_version write", error))?;
    } else {
        // No migration path exists for the clean v1 baseline: any recorded
        // version below SCHEMA_VERSION is unsupported and fails closed.
        return Err(schema_contract_error(
            "schema migration selection",
            format!("no migration path from schema version {version} to {SCHEMA_VERSION}"),
        ));
    }

    tracing::debug!(
        target: "frankensearch.storage",
        schema_version = SCHEMA_VERSION,
        "storage schema bootstrap complete"
    );

    Ok(())
}

pub fn current_version(conn: &Connection) -> SearchResult<i64> {
    current_version_at(conn, "schema version read")
}

pub(crate) fn current_version_at(conn: &Connection, stage: &'static str) -> SearchResult<i64> {
    retry_transient_storage(
        || {
            current_version_optional_at(conn, stage)?
                .ok_or_else(|| schema_contract_error(stage, "schema_version table has no rows"))
        },
        stage,
    )
}

#[cfg(test)]
fn current_version_optional(conn: &Connection) -> SearchResult<Option<i64>> {
    current_version_optional_at(conn, "schema version read")
}

fn current_version_optional_at(
    conn: &Connection,
    stage: &'static str,
) -> SearchResult<Option<i64>> {
    match schema_version_state(conn, stage)? {
        SchemaVersionState::MissingTable => Err(map_storage_error_at(
            stage,
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some("no such table: schema_version".to_owned()),
            ),
        )),
        SchemaVersionState::Empty => Ok(None),
        SchemaVersionState::Present(version) => Ok(Some(version)),
    }
}

fn schema_version_state(
    conn: &Connection,
    stage: &'static str,
) -> SearchResult<SchemaVersionState> {
    let table_count = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'schema_version';",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|error| map_storage_error_at(stage, error))?;
    if table_count == 0 {
        return Ok(SchemaVersionState::MissingTable);
    }

    match conn.query_row(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1;",
        [],
        |row| row.get::<_, i64>(0),
    ) {
        Ok(version) => Ok(SchemaVersionState::Present(version)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(SchemaVersionState::Empty),
        Err(error) => Err(map_storage_error_at(stage, error)),
    }
}

fn reject_future_version(version: i64, stage: &'static str) -> SearchResult<()> {
    if version > SCHEMA_VERSION {
        return Err(schema_contract_error(
            stage,
            format!("schema version {version} is newer than supported {SCHEMA_VERSION}"),
        ));
    }
    Ok(())
}

pub(crate) fn row_i64(row: &Row<'_>, index: usize, field: &str) -> SearchResult<i64> {
    row_i64_at(row, index, field, "schema row decode")
}

fn row_i64_at(row: &Row<'_>, index: usize, field: &str, stage: &'static str) -> SearchResult<i64> {
    match row.get::<_, i64>(index) {
        Ok(value) => Ok(value),
        Err(rusqlite::Error::InvalidColumnIndex(_)) => Err(schema_contract_error(
            stage,
            format!("missing column for {field}"),
        )),
        Err(error) => Err(schema_contract_error(
            stage,
            format!("unexpected type for {field}: {error}"),
        )),
    }
}

fn schema_contract_error(stage: &'static str, message: impl Into<String>) -> SearchError {
    map_storage_error_at(stage, io::Error::other(message.into()))
}

#[cfg(test)]
fn storage_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    map_storage_error_at("storage operation", source)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process;
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use rusqlite::{Connection, params};

    use super::{
        SCHEMA_VERSION, bootstrap, current_version, current_version_optional, storage_error,
    };

    fn table_exists(conn: &Connection, table_name: &str) -> bool {
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1;",
            params![table_name],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)
        .unwrap_or(false)
    }

    fn index_exists(conn: &Connection, table_name: &str, index_name: &str) -> bool {
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type = 'index' AND tbl_name = ?1 AND name = ?2;",
            params![table_name, index_name],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)
        .unwrap_or(false)
    }

    fn seed_marker_row(conn: &Connection, version: i64) {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("schema_version table should be creatable");
        conn.execute(
            "INSERT OR REPLACE INTO schema_version(version) VALUES (?1);",
            params![version],
        )
        .expect("marker row should insert");
    }

    #[test]
    fn bootstrap_sets_latest_version_for_fresh_database() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");

        bootstrap(&mut conn).expect("bootstrap should succeed");
        assert_eq!(
            current_version(&conn).expect("schema version should exist"),
            SCHEMA_VERSION
        );
        assert!(
            index_exists(&conn, "embedding_status", "idx_embedding_status_pending"),
            "baseline schema should include pending-status index"
        );
        assert!(
            index_exists(&conn, "embedding_jobs", "idx_jobs_pending"),
            "baseline schema should include queue pending index"
        );
        assert!(
            index_exists(&conn, "embedding_jobs", "idx_jobs_processing"),
            "baseline schema should include queue processing index"
        );
        conn.execute_batch(
            "INSERT INTO documents(doc_id, content_preview, content_hash, content_length, created_at, updated_at) \
             VALUES ('fresh-doc', 'preview', X'0102', 7, 100, 100);",
        )
        .expect("fresh document");
        conn.execute_batch(
            "INSERT INTO embedding_jobs(doc_id, embedder_id, submitted_at) \
             VALUES ('fresh-doc', 'model', 100);",
        )
        .expect("jobs may omit the initial claim epoch");
        let claim_epoch = conn
            .query_row("SELECT claim_epoch FROM embedding_jobs;", [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("baseline schema includes claim epoch");
        assert_eq!(claim_epoch, 0);
        assert!(
            conn.execute_batch("UPDATE embedding_jobs SET claim_epoch = NULL;")
                .is_err(),
            "a claim epoch cannot become NULL"
        );
    }

    #[test]
    fn bootstrap_stamps_user_version_pragma() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");

        bootstrap(&mut conn).expect("bootstrap should succeed");
        let user_version = conn
            .query_row("PRAGMA user_version;", [], |row| row.get::<_, i64>(0))
            .expect("user_version pragma should be queryable");
        assert_eq!(user_version, SCHEMA_VERSION);
    }

    #[test]
    fn bootstrap_accepts_read_only_confirmation_that_marker_table_is_absent() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        assert_eq!(
            super::schema_version_state(&conn, "test absence probe")
                .expect("absence probe should be readable"),
            super::SchemaVersionState::MissingTable
        );

        bootstrap(&mut conn).expect("confirmed absent marker table should bootstrap");
        assert_eq!(
            current_version(&conn).expect("bootstrapped marker should be readable"),
            SCHEMA_VERSION
        );
    }

    #[test]
    fn bootstrap_does_not_swallow_malformed_marker_state_as_absence() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch("CREATE TABLE schema_version (version TEXT PRIMARY KEY);")
            .expect("malformed marker table should be creatable");
        conn.execute_batch("INSERT INTO schema_version(version) VALUES ('not-an-integer');")
            .expect("malformed marker should insert");

        let error = bootstrap(&mut conn).expect_err("malformed marker state must fail closed");
        let message = error.to_string();
        assert!(
            message.contains("schema preflight"),
            "error should retain the preflight stage: {message}"
        );
        assert!(
            message.contains("column type") || message.contains("Invalid column"),
            "malformed marker must not be reclassified as absence: {message}"
        );
        assert!(
            conn.is_autocommit(),
            "read-only preflight rejection must not leave a transaction active"
        );
    }

    #[test]
    fn bootstrap_rejects_future_schema_versions() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        seed_marker_row(&conn, SCHEMA_VERSION + 100);

        let error = bootstrap(&mut conn).expect_err("future schemas should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("newer than supported"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn bootstrap_seeds_baseline_when_marker_is_zero() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        seed_marker_row(&conn, 0);
        bootstrap(&mut conn)
            .expect("a zero marker is equivalent to an absent marker and bootstraps fresh");
        assert_eq!(current_version(&conn).unwrap(), SCHEMA_VERSION);
    }

    #[test]
    fn bootstrap_is_idempotent_at_latest_version() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");

        bootstrap(&mut conn).expect("first bootstrap should succeed");
        bootstrap(&mut conn).expect("second bootstrap should succeed");
        bootstrap(&mut conn).expect("third bootstrap should succeed");

        assert_eq!(
            current_version(&conn).expect("schema version should exist"),
            SCHEMA_VERSION
        );
    }

    #[test]
    fn concurrent_bootstrap_on_disk_is_race_safe() {
        const THREADS: usize = 6;

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let db_path = std::env::temp_dir().join(format!(
            "frankensearch-schema-bootstrap-{}-{nanos}.sqlite3",
            process::id()
        ));

        let barrier = Arc::new(Barrier::new(THREADS));
        let (tx, rx) = mpsc::channel::<i64>();
        let mut handles = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let gate = Arc::clone(&barrier);
            let sender = tx.clone();
            let path = db_path.clone();
            handles.push(thread::spawn(move || {
                gate.wait();
                let mut conn =
                    Connection::open(&path).expect("connection should open");
                // Retry bootstrap to handle SQLITE_BUSY under contention.
                let mut last_err = None;
                for _ in 0..10 {
                    match bootstrap(&mut conn) {
                        Ok(()) => {
                            last_err = None;
                            break;
                        }
                        Err(error) => {
                            assert!(
                                conn.is_autocommit(),
                                "failed bootstrap must not leave an active transaction: {error}"
                            );
                            let message = error.to_string().to_ascii_lowercase();
                            if message.contains("busy") || message.contains("locked") {
                                last_err = Some(error);
                                thread::sleep(std::time::Duration::from_millis(5));
                                continue;
                            }
                            panic!("bootstrap should not fail with non-retryable error: {error}");
                        }
                    }
                }
                if let Some(error) = last_err {
                    panic!("bootstrap should succeed after retries: {error}");
                }
                let mut version = None;
                for _ in 0..10 {
                    match current_version(&conn) {
                        Ok(value) => {
                            version = Some(value);
                            break;
                        }
                        Err(error) => {
                            let message = error.to_string().to_ascii_lowercase();
                            if message.contains("busy")
                                || message.contains("locked")
                                || message.contains("no rows")
                            {
                                thread::sleep(std::time::Duration::from_millis(5));
                                continue;
                            }
                            panic!(
                                "schema version lookup should not fail with non-retryable error: {error}"
                            );
                        }
                    }
                }
                let version = version.expect("schema version should exist after retries");
                sender.send(version).expect("sender should send version");
            }));
        }
        drop(tx);

        let versions: Vec<i64> = rx.iter().collect();
        for handle in handles {
            handle.join().expect("bootstrap thread should join");
        }

        assert_eq!(versions.len(), THREADS);
        assert!(
            versions.iter().all(|version| *version == SCHEMA_VERSION),
            "all concurrent bootstraps should resolve to the v1 schema"
        );

        let db_path_display = db_path.display().to_string();
        let cleanup_targets = [
            db_path,
            PathBuf::from(format!("{db_path_display}-wal")),
            PathBuf::from(format!("{db_path_display}-shm")),
        ];
        for target in cleanup_targets {
            let _ = std::fs::remove_file(target);
        }
    }

    // ── Schema version constant ─────────────────────────────────────────

    #[test]
    fn schema_version_is_one() {
        assert_eq!(SCHEMA_VERSION, 1);
    }

    // ── SCHEMA_V1 invariants ────────────────────────────────────────────

    #[test]
    fn baseline_schema_is_nonempty() {
        assert!(
            !super::SCHEMA_V1.is_empty(),
            "SCHEMA_V1 must have at least one statement"
        );
    }

    // ── All tables exist after bootstrap ─────────────────────────────────

    #[test]
    fn bootstrap_creates_all_expected_tables() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap(&mut conn).expect("bootstrap should succeed");

        let expected_tables = [
            "documents",
            "embedding_jobs",
            "embedding_status",
            "content_hashes",
            "index_metadata",
            "index_build_history",
            "search_history",
            "bookmarks",
            "schema_version",
            "frankensearch_fts5_rebuild_version",
        ];

        for table in expected_tables {
            assert!(
                table_exists(&conn, table),
                "table '{table}' should exist after bootstrap"
            );
        }
    }

    // ── All indices exist after bootstrap ────────────────────────────────

    #[test]
    fn bootstrap_creates_all_expected_indices() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap(&mut conn).expect("bootstrap should succeed");

        let expected_indices: &[(&str, &str)] = &[
            ("documents", "idx_documents_content_hash"),
            ("documents", "idx_documents_updated_at"),
            ("embedding_status", "idx_embedding_status_pending"),
            ("embedding_jobs", "idx_jobs_pending"),
            ("embedding_jobs", "idx_jobs_processing"),
            ("index_build_history", "idx_build_history_index"),
            ("search_history", "idx_history_query"),
            ("search_history", "idx_history_ts"),
            ("bookmarks", "idx_bookmarks_doc_query"),
        ];

        for &(table, idx) in expected_indices {
            assert!(
                index_exists(&conn, table, idx),
                "index '{idx}' should exist on '{table}' after bootstrap"
            );
        }
    }

    // ── Foreign key cascades ─────────────────────────────────────────────

    #[test]
    fn embedding_jobs_cascade_on_document_delete() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap(&mut conn).expect("bootstrap should succeed");
        conn.execute_batch("PRAGMA foreign_keys=ON;")
            .expect("enable foreign keys");
        conn.execute_batch(
            "INSERT INTO documents(doc_id, content_preview, content_hash, content_length, created_at, updated_at) \
             VALUES ('cascade-doc', 'preview', X'0102', 7, 100, 100);",
        )
        .expect("parent document");
        conn.execute_batch(
            "INSERT INTO embedding_jobs(doc_id, embedder_id, submitted_at) \
             VALUES ('cascade-doc', 'model', 100);",
        )
        .expect("child job");
        conn.execute_batch(
            "INSERT INTO embedding_status(doc_id, embedder_id) \
             VALUES ('cascade-doc', 'model');",
        )
        .expect("child status");

        conn.execute_batch("DELETE FROM documents WHERE doc_id = 'cascade-doc';")
            .expect("delete parent");

        for table in ["embedding_jobs", "embedding_status"] {
            let remaining: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM {table};"), [], |row| {
                    row.get(0)
                })
                .expect("count children");
            assert_eq!(
                remaining, 0,
                "ON DELETE CASCADE should remove {table} rows with the parent"
            );
        }
    }

    // ── row_i64 edge cases ──────────────────────────────────────────────

    #[test]
    fn row_i64_wrong_type_returns_error() {
        let conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch("CREATE TABLE test_row (val TEXT);")
            .expect("create");
        conn.execute("INSERT INTO test_row(val) VALUES (?1);", params!["hello"])
            .expect("insert");

        let decoded = conn
            .query_row("SELECT val FROM test_row LIMIT 1;", [], |row| {
                Ok(super::row_i64(row, 0, "test_field"))
            })
            .expect("query should return a row");
        let err = decoded.expect_err("text should not decode as i64");
        let msg = err.to_string();
        assert!(
            msg.contains("unexpected type") || msg.contains("test_field"),
            "error should mention type mismatch: {msg}"
        );
    }

    #[test]
    fn row_i64_missing_column_returns_error() {
        let conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch("CREATE TABLE test_row2 (val INTEGER);")
            .expect("create");
        conn.execute("INSERT INTO test_row2(val) VALUES (?1);", params![42_i64])
            .expect("insert");

        let decoded = conn
            .query_row("SELECT val FROM test_row2 LIMIT 1;", [], |row| {
                Ok(super::row_i64(row, 99, "missing_col"))
            })
            .expect("query should return a row");
        let err = decoded.expect_err("out-of-bounds index should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("missing column") || msg.contains("missing_col"),
            "error should mention missing column: {msg}"
        );
    }

    #[test]
    fn row_i64_valid_integer_returns_value() {
        let conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch("CREATE TABLE test_row3 (val INTEGER);")
            .expect("create");
        conn.execute("INSERT INTO test_row3(val) VALUES (?1);", params![42_i64])
            .expect("insert");

        let decoded = conn
            .query_row("SELECT val FROM test_row3 LIMIT 1;", [], |row| {
                Ok(super::row_i64(row, 0, "val"))
            })
            .expect("query should return a row");
        let value = decoded.expect("should extract integer");
        assert_eq!(value, 42);
    }

    // ── current_version on empty table ──────────────────────────────────

    #[test]
    fn current_version_optional_empty_table_returns_none() {
        let conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("create table");
        let result = current_version_optional(&conn).expect("should not error");
        assert_eq!(result, None);
    }

    #[test]
    fn current_version_empty_table_returns_error() {
        let conn = Connection::open_in_memory().expect("in-memory connection");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("create table");
        let err = current_version(&conn).expect_err("should fail on empty table");
        let msg = err.to_string();
        assert!(
            msg.contains("no rows"),
            "error should mention no rows: {msg}"
        );
    }

    #[test]
    fn current_version_missing_table_preserves_typed_cause_and_stage() {
        let conn = Connection::open_in_memory().expect("in-memory connection");
        let error = current_version(&conn).expect_err("missing marker table should fail");
        assert!(
            error.to_string().contains("schema version read"),
            "error should identify the failed stage: {error}"
        );

        let frankensearch_core::SearchError::SubsystemError { source, .. } = &error else {
            panic!("expected storage subsystem error, got {error:?}");
        };
        let typed_cause = std::error::Error::source(source.as_ref())
            .and_then(|cause| cause.downcast_ref::<rusqlite::Error>());
        assert!(
            matches!(
                typed_cause,
                Some(rusqlite::Error::SqliteFailure(_, Some(message)))
                    if message.contains("schema_version")
            ),
            "error chain should retain the missing-table cause, got {typed_cause:?}"
        );
    }

    // ── storage_error ───────────────────────────────────────────────────

    #[test]
    fn storage_error_wraps_io_error() {
        let io_err = std::io::Error::other("test error");
        let err = storage_error(io_err);
        match &err {
            frankensearch_core::SearchError::SubsystemError { subsystem, .. } => {
                assert_eq!(*subsystem, "storage");
            }
            other => panic!("expected SubsystemError, got: {other:?}"),
        }
    }

    // ── Tables are queryable after bootstrap ─────────────────────────────

    #[test]
    fn tables_accept_basic_queries_after_bootstrap() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap(&mut conn).expect("bootstrap should succeed");

        for query in [
            "SELECT COUNT(*) FROM documents;",
            "SELECT COUNT(*) FROM embedding_jobs;",
            "SELECT COUNT(*) FROM embedding_status;",
            "SELECT COUNT(*) FROM content_hashes;",
            "SELECT COUNT(*) FROM index_metadata;",
            "SELECT COUNT(*) FROM index_build_history;",
            "SELECT COUNT(*) FROM search_history;",
            "SELECT COUNT(*) FROM bookmarks;",
        ] {
            conn.query_row(query, [], |row| row.get::<_, i64>(0))
                .unwrap_or_else(|error| panic!("query should succeed: {query} ({error})"));
        }
    }

    // ── Schema version is always latest after bootstrap ──────────────────

    #[test]
    fn schema_version_row_contains_latest_after_fresh_bootstrap() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap(&mut conn).expect("bootstrap should succeed");

        let version = conn
            .query_row(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1;",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("query schema_version");
        assert_eq!(version, SCHEMA_VERSION);
    }

    // ── index_exists helper returns false for nonexistent ─────────────────

    #[test]
    fn index_exists_returns_false_for_nonexistent_index() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap(&mut conn).expect("bootstrap should succeed");
        assert!(!index_exists(
            &conn,
            "table_that_does_not_exist",
            "idx_nonexistent_never_created"
        ));
    }

    // ── Insert and query documents table ─────────────────────────────────

    #[test]
    fn documents_table_accepts_insert_and_select() {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        bootstrap(&mut conn).expect("bootstrap should succeed");

        conn.execute(
            "INSERT INTO documents(doc_id, source_path, content_preview, content_hash, content_length, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);",
            params![
                "doc-001",
                "/path/to/file.rs",
                "fn main() {}",
                vec![0xAB_u8, 0xCD],
                12_i64,
                1000_i64,
                1000_i64
            ],
        )
        .expect("insert should succeed");

        let doc_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM documents WHERE doc_id = 'doc-001';",
                [],
                |row| row.get(0),
            )
            .expect("select should succeed");
        assert_eq!(doc_count, 1);
    }
}
