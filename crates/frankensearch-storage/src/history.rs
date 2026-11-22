//! Search history and bookmark persistence.
//!
//! Stores recent search queries with metadata (result count, latency, query
//! class) and user bookmarks of individual search results.

use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use frankensearch_core::{SearchError, SearchResult};
use rusqlite::types::Value;
use rusqlite::{Connection, Row, params};

use crate::connection::{
    map_storage_error, map_storage_error_at, retry_transient_storage, unretryable_rollback_error,
};

fn with_history_transaction<T>(
    conn: &mut Connection,
    mut op: impl FnMut(&Connection) -> SearchResult<T>,
) -> SearchResult<T> {
    retry_transient_storage(
        || {
            // `&mut Connection` makes a nested `BEGIN` a compile-time
            // impossibility; `Transaction`'s `DropBehavior::Rollback` plus the
            // explicit `catch_unwind` arm below restore autocommit on error
            // and panic alike.
            let tx = conn
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .map_err(|error| map_storage_error_at("history transaction begin", error))?;
            let result = catch_unwind(AssertUnwindSafe(|| op(&tx)));
            match result {
                Ok(Ok(value)) => tx
                    .commit()
                    .map(|()| value)
                    .map_err(|error| map_storage_error_at("history transaction commit", error)),
                Ok(Err(error)) => match tx.rollback() {
                    Ok(()) => Err(error),
                    Err(rollback_err) => Err(unretryable_rollback_error(&error, &rollback_err)),
                },
                Err(payload) => {
                    if let Err(rollback_err) = tx.rollback() {
                        tracing::error!(
                            target: "frankensearch.storage",
                            error = %rollback_err,
                            "critical: rollback failed during history transaction panic recovery"
                        );
                    }
                    resume_unwind(payload);
                }
            }
        },
        "search history",
    )
}

// ─── Search History ────────────────────────────────────────────────────────

/// A single entry in the search history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHistoryEntry {
    /// Auto-incremented row ID.
    pub id: i64,
    /// The query text as entered by the user.
    pub query: String,
    /// Classification of the query (e.g. "identifier", "`natural_language`").
    pub query_class: Option<String>,
    /// Number of results returned.
    pub result_count: Option<i64>,
    /// Phase-1 (fast) latency in milliseconds.
    pub phase1_latency_ms: Option<i64>,
    /// Phase-2 (quality) latency in milliseconds.
    pub phase2_latency_ms: Option<i64>,
    /// JSON array of top-3 result doc IDs.
    pub top_results_json: Option<String>,
    /// Unix epoch timestamp (seconds) of when the search was performed.
    pub searched_at: i64,
}

/// Insert a search history entry, deduplicating queries within `dedup_window_secs`.
///
/// If the same query text was searched within the last `dedup_window_secs`
/// seconds, the existing entry is updated instead of creating a duplicate.
#[allow(clippy::too_many_arguments)]
pub fn record_search(
    conn: &mut Connection,
    query: &str,
    query_class: Option<&str>,
    result_count: Option<i64>,
    phase1_latency_ms: Option<i64>,
    phase2_latency_ms: Option<i64>,
    top_results_json: Option<&str>,
    searched_at: i64,
    dedup_window_secs: i64,
) -> SearchResult<()> {
    with_history_transaction(conn, |conn| {
        record_search_once(
            conn,
            query,
            query_class,
            result_count,
            phase1_latency_ms,
            phase2_latency_ms,
            top_results_json,
            searched_at,
            dedup_window_secs,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn record_search_once(
    conn: &Connection,
    query: &str,
    query_class: Option<&str>,
    result_count: Option<i64>,
    phase1_latency_ms: Option<i64>,
    phase2_latency_ms: Option<i64>,
    top_results_json: Option<&str>,
    searched_at: i64,
    dedup_window_secs: i64,
) -> SearchResult<()> {
    // Check for a recent duplicate within the dedup window.
    let window_start = searched_at.saturating_sub(dedup_window_secs);
    let existing_id = optional_first_i64(
        conn,
        "SELECT id FROM search_history \
     WHERE query = ?1 \
     AND searched_at >= ?2 \
     ORDER BY searched_at DESC LIMIT 1;",
        params![query, window_start],
    )?;

    if let Some(existing_id) = existing_id {
        // Update the existing entry with new metadata.
        conn.execute(
            "UPDATE search_history SET \
         result_count = ?1, phase1_latency_ms = ?2, phase2_latency_ms = ?3, \
         top_results_json = ?4, searched_at = ?5 \
         WHERE id = ?6;",
            params![
                result_count,
                phase1_latency_ms,
                phase2_latency_ms,
                top_results_json,
                searched_at,
                existing_id,
            ],
        )
        .map_err(map_storage_error)?;
    } else {
        // Insert a new entry.
        conn.execute(
            "INSERT INTO search_history \
         (query, query_class, result_count, phase1_latency_ms, \
          phase2_latency_ms, top_results_json, searched_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);",
            params![
                query,
                query_class,
                result_count,
                phase1_latency_ms,
                phase2_latency_ms,
                top_results_json,
                searched_at,
            ],
        )
        .map_err(map_storage_error)?;
    }

    Ok(())
}

fn escape_like_pattern(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '%' | '_' | '\\' => {
                escaped.push('\\');
                escaped.push(ch);
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn limit_to_i64(limit: usize, field: &'static str) -> SearchResult<i64> {
    i64::try_from(limit).map_err(|_| SearchError::InvalidConfig {
        field: field.to_owned(),
        value: limit.to_string(),
        reason: "limit does not fit in sqlite integer".to_owned(),
    })
}

/// Retrieve search history in most-recent-first order.
pub fn list_search_history(
    conn: &Connection,
    limit: usize,
) -> SearchResult<Vec<SearchHistoryEntry>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let limit_i64 = limit_to_i64(limit, "history.limit")?;
    let entries = retry_transient_storage(
        || {
            let mut stmt = conn
                .prepare(
                    "SELECT id, query, query_class, result_count, phase1_latency_ms, \
     phase2_latency_ms, top_results_json, searched_at \
     FROM search_history ORDER BY searched_at DESC LIMIT ?1;",
                )
                .map_err(map_storage_error)?;
            let mut rows = stmt.query(params![limit_i64]).map_err(map_storage_error)?;
            let mut entries = Vec::new();
            while let Some(row) = rows.next().map_err(map_storage_error)? {
                entries.push(parse_history_row(row)?);
            }
            Ok(entries)
        },
        "search history list",
    )?;

    Ok(entries)
}

/// Find history entries matching a prefix (for auto-suggest).
pub fn search_history_prefix(
    conn: &Connection,
    prefix: &str,
    limit: usize,
) -> SearchResult<Vec<SearchHistoryEntry>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let pattern = format!("{}%", escape_like_pattern(prefix));
    let limit_i64 = limit_to_i64(limit, "history.prefix_limit")?;
    let entries = retry_transient_storage(
        || {
            let mut stmt = conn
                .prepare(
                    "SELECT id, query, query_class, result_count, phase1_latency_ms, \
     phase2_latency_ms, top_results_json, searched_at \
     FROM search_history WHERE query LIKE ?1 ESCAPE '\\' \
     ORDER BY searched_at DESC LIMIT ?2;",
                )
                .map_err(map_storage_error)?;
            let mut rows = stmt
                .query(params![pattern, limit_i64])
                .map_err(map_storage_error)?;
            let mut entries = Vec::new();
            while let Some(row) = rows.next().map_err(map_storage_error)? {
                entries.push(parse_history_row(row)?);
            }
            Ok(entries)
        },
        "search history prefix",
    )?;

    Ok(entries)
}

/// Count total search history entries.
pub fn count_search_history(conn: &Connection) -> SearchResult<i64> {
    retry_transient_storage(|| count_search_history_once(conn), "search history count")
}

fn count_search_history_once(conn: &Connection) -> SearchResult<i64> {
    conn.query_row("SELECT COUNT(*) FROM search_history;", [], |row| {
        row.get::<_, i64>(0)
    })
    .map_err(map_storage_error)
}

/// Truncate search history to keep only the most recent `max_entries`.
pub fn truncate_search_history(conn: &mut Connection, max_entries: usize) -> SearchResult<i64> {
    let max_entries_i64 = limit_to_i64(max_entries, "history.max_entries")?;
    with_history_transaction(conn, |conn| {
        let count = count_search_history_once(conn)?;
        if count <= max_entries_i64 {
            return Ok(0);
        }

        let to_delete = count - max_entries_i64;
        conn.execute(
            "DELETE FROM search_history WHERE id IN (\
     SELECT id FROM search_history ORDER BY searched_at ASC LIMIT ?1);",
            params![to_delete],
        )
        .map_err(map_storage_error)?;

        Ok(to_delete)
    })
}

fn parse_history_row(row: &Row<'_>) -> SearchResult<SearchHistoryEntry> {
    let Some(Value::Integer(id)) = row_value(row, 0)? else {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other("missing search_history.id")),
        });
    };
    let Some(Value::Text(query)) = row_value(row, 1)? else {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other("missing search_history.query")),
        });
    };
    let Some(Value::Integer(searched_at)) = row_value(row, 7)? else {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other("missing search_history.searched_at")),
        });
    };

    Ok(SearchHistoryEntry {
        id,
        query,
        query_class: extract_opt_text(row, 2),
        result_count: extract_opt_i64(row, 3),
        phase1_latency_ms: extract_opt_i64(row, 4),
        phase2_latency_ms: extract_opt_i64(row, 5),
        top_results_json: extract_opt_text(row, 6),
        searched_at,
    })
}

// ─── Bookmarks ─────────────────────────────────────────────────────────────

/// A bookmarked search result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    /// Auto-incremented row ID.
    pub id: i64,
    /// Document ID of the bookmarked result.
    pub doc_id: String,
    /// The query that led to this bookmark (may be `None`).
    pub query: Option<String>,
    /// User annotation.
    pub note: Option<String>,
    /// Unix epoch timestamp (seconds) of creation.
    pub created_at: i64,
}

/// Add a bookmark. If the same (`doc_id`, query) pair exists, updates the note.
pub fn add_bookmark(
    conn: &mut Connection,
    doc_id: &str,
    query: Option<&str>,
    note: Option<&str>,
    created_at: i64,
) -> SearchResult<()> {
    with_history_transaction(conn, |conn| {
        add_bookmark_once(conn, doc_id, query, note, created_at)
    })
}

fn add_bookmark_once(
    conn: &Connection,
    doc_id: &str,
    query: Option<&str>,
    note: Option<&str>,
    created_at: i64,
) -> SearchResult<()> {
    // Check for existing bookmark with same (doc_id, query) pair.
    // Handle NULL query explicitly since SQL NULL != NULL.
    let existing_id = if let Some(q) = query {
        optional_first_i64(
            conn,
            "SELECT id FROM bookmarks \
         WHERE doc_id = ?1 AND query = ?2 LIMIT 1;",
            params![doc_id, q],
        )?
    } else {
        optional_first_i64(
            conn,
            "SELECT id FROM bookmarks \
         WHERE doc_id = ?1 AND query IS NULL LIMIT 1;",
            params![doc_id],
        )?
    };

    if let Some(existing_id) = existing_id {
        conn.execute(
            "UPDATE bookmarks SET note = ?1, created_at = ?2 WHERE id = ?3;",
            params![note, created_at, existing_id],
        )
        .map_err(map_storage_error)?;
    } else {
        conn.execute(
            "INSERT INTO bookmarks (doc_id, query, note, created_at) \
         VALUES (?1, ?2, ?3, ?4);",
            params![doc_id, query, note, created_at],
        )
        .map_err(map_storage_error)?;
    }
    Ok(())
}

/// Remove a bookmark by its row ID.
pub fn remove_bookmark(conn: &mut Connection, bookmark_id: i64) -> SearchResult<bool> {
    with_history_transaction(conn, |conn| remove_bookmark_once(conn, bookmark_id))
}

fn remove_bookmark_once(conn: &Connection, bookmark_id: i64) -> SearchResult<bool> {
    let existed = optional_first_i64(
        conn,
        "SELECT 1 FROM bookmarks WHERE id = ?1 LIMIT 1;",
        params![bookmark_id],
    )?
    .is_some();

    if !existed {
        return Ok(false);
    }

    conn.execute("DELETE FROM bookmarks WHERE id = ?1;", params![bookmark_id])
        .map_err(map_storage_error)?;

    // Verify deletion completed.
    let remaining = optional_first_i64(
        conn,
        "SELECT id FROM bookmarks WHERE id = ?1;",
        params![bookmark_id],
    )?;
    Ok(remaining.is_none())
}

/// Remove a bookmark by document ID (removes all bookmarks for that doc).
pub fn remove_bookmark_by_doc(conn: &mut Connection, doc_id: &str) -> SearchResult<()> {
    with_history_transaction(conn, |conn| {
        conn.execute("DELETE FROM bookmarks WHERE doc_id = ?1;", params![doc_id])
            .map_err(map_storage_error)?;
        Ok(())
    })
}

/// List all bookmarks in most-recent-first order.
pub fn list_bookmarks(conn: &Connection, limit: usize) -> SearchResult<Vec<Bookmark>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let limit_i64 = limit_to_i64(limit, "bookmarks.limit")?;
    let bookmarks = retry_transient_storage(
        || {
            let mut stmt = conn
                .prepare(
                    "SELECT id, doc_id, query, note, created_at \
     FROM bookmarks ORDER BY created_at DESC LIMIT ?1;",
                )
                .map_err(map_storage_error)?;
            let mut rows = stmt.query(params![limit_i64]).map_err(map_storage_error)?;
            let mut bookmarks = Vec::new();
            while let Some(row) = rows.next().map_err(map_storage_error)? {
                bookmarks.push(parse_bookmark_row(row)?);
            }
            Ok(bookmarks)
        },
        "bookmark list",
    )?;

    Ok(bookmarks)
}

/// Count total bookmarks.
pub fn count_bookmarks(conn: &Connection) -> SearchResult<i64> {
    retry_transient_storage(
        || {
            conn.query_row("SELECT COUNT(*) FROM bookmarks;", [], |row| {
                row.get::<_, i64>(0)
            })
            .map_err(map_storage_error)
        },
        "bookmark count",
    )
}

/// Check if a document is bookmarked.
pub fn is_bookmarked(conn: &Connection, doc_id: &str) -> SearchResult<bool> {
    retry_transient_storage(
        || {
            optional_first_i64(
                conn,
                "SELECT 1 FROM bookmarks WHERE doc_id = ?1 LIMIT 1;",
                params![doc_id],
            )
            .map(|bookmark| bookmark.is_some())
        },
        "bookmark exists",
    )
}

fn parse_bookmark_row(row: &Row<'_>) -> SearchResult<Bookmark> {
    let Some(Value::Integer(id)) = row_value(row, 0)? else {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other("missing bookmarks.id")),
        });
    };
    let Some(Value::Text(doc_id)) = row_value(row, 1)? else {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other("missing bookmarks.doc_id")),
        });
    };
    let Some(Value::Integer(created_at)) = row_value(row, 4)? else {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other("missing bookmarks.created_at")),
        });
    };

    Ok(Bookmark {
        id,
        doc_id,
        query: extract_opt_text(row, 2),
        note: extract_opt_text(row, 3),
        created_at,
    })
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn row_value(row: &Row<'_>, index: usize) -> SearchResult<Option<Value>> {
    match row.get::<_, Value>(index) {
        Ok(value) => Ok(Some(value)),
        Err(rusqlite::Error::InvalidColumnIndex(_)) => Ok(None),
        Err(error) => Err(map_storage_error(error)),
    }
}

fn optional_first_i64(
    conn: &Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> SearchResult<Option<i64>> {
    let mut stmt = conn.prepare(sql).map_err(map_storage_error)?;
    let mut rows = stmt.query(params).map_err(map_storage_error)?;
    rows.next().map_err(map_storage_error)?.map_or_else(
        || Ok(None),
        |row| row.get::<_, i64>(0).map(Some).map_err(map_storage_error),
    )
}

fn extract_opt_text(row: &Row<'_>, index: usize) -> Option<String> {
    match row.get::<_, Value>(index) {
        Ok(Value::Text(v)) => Some(v),
        _ => None,
    }
}

fn extract_opt_i64(row: &Row<'_>, index: usize) -> Option<i64> {
    match row.get::<_, Value>(index) {
        Ok(Value::Integer(v)) => Some(v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema;

    fn test_conn() -> Connection {
        let mut conn = Connection::open_in_memory().expect("in-memory connection");
        // Apply pragmas for foreign keys.
        conn.execute_batch("PRAGMA foreign_keys=ON;")
            .expect("pragma should succeed");
        schema::bootstrap(&mut conn).expect("schema bootstrap should succeed");
        conn
    }

    // ─── Search History Tests ──────────────────────────────────────────

    #[test]
    fn record_and_list_search_history() {
        let mut conn = test_conn();
        record_search(
            &mut conn,
            "rust async",
            Some("natural_language"),
            Some(42),
            Some(5),
            Some(120),
            Some("[\"doc1\",\"doc2\",\"doc3\"]"),
            1_700_000_000,
            60,
        )
        .expect("record should succeed");

        let entries = list_search_history(&conn, 10).expect("list should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].query, "rust async");
        assert_eq!(entries[0].query_class.as_deref(), Some("natural_language"));
        assert_eq!(entries[0].result_count, Some(42));
        assert_eq!(entries[0].phase1_latency_ms, Some(5));
        assert_eq!(entries[0].phase2_latency_ms, Some(120));
    }

    #[test]
    fn history_dedup_within_window() {
        let mut conn = test_conn();
        record_search(
            &mut conn,
            "tokio spawn",
            None,
            Some(10),
            None,
            None,
            None,
            1_700_000_000,
            60,
        )
        .expect("first record");

        // Same query within 60 seconds — should update, not insert.
        record_search(
            &mut conn,
            "tokio spawn",
            None,
            Some(15),
            None,
            None,
            None,
            1_700_000_030,
            60,
        )
        .expect("dedup record");

        let count = count_search_history(&conn).expect("count");
        assert_eq!(count, 1, "dedup should prevent duplicate entry");

        let entries = list_search_history(&conn, 10).expect("list");
        assert_eq!(
            entries[0].result_count,
            Some(15),
            "dedup should update result_count"
        );
    }

    #[test]
    fn history_no_dedup_outside_window() {
        let mut conn = test_conn();
        record_search(
            &mut conn,
            "async fn",
            None,
            Some(5),
            None,
            None,
            None,
            1_700_000_000,
            60,
        )
        .expect("first record");

        // Same query but 2 minutes later — should insert new entry.
        record_search(
            &mut conn,
            "async fn",
            None,
            Some(8),
            None,
            None,
            None,
            1_700_000_120,
            60,
        )
        .expect("second record");

        let count = count_search_history(&conn).expect("count");
        assert_eq!(count, 2, "entries outside dedup window should both persist");
    }

    #[test]
    fn history_mru_order() {
        let mut conn = test_conn();
        for (i, ts) in [1_700_000_000i64, 1_700_000_060, 1_700_000_120]
            .iter()
            .enumerate()
        {
            record_search(
                &mut conn,
                &format!("query-{i}"),
                None,
                None,
                None,
                None,
                None,
                *ts,
                0,
            )
            .expect("record");
        }

        let entries = list_search_history(&conn, 10).expect("list");
        assert_eq!(entries[0].query, "query-2", "most recent first");
        assert_eq!(entries[1].query, "query-1");
        assert_eq!(entries[2].query, "query-0", "oldest last");
    }

    #[test]
    fn history_truncation() {
        let mut conn = test_conn();
        for i in 0..10 {
            record_search(
                &mut conn,
                &format!("query-{i}"),
                None,
                None,
                None,
                None,
                None,
                1_700_000_000 + i64::from(i) * 60,
                0,
            )
            .expect("record");
        }

        assert_eq!(count_search_history(&conn).expect("count"), 10);

        let deleted = truncate_search_history(&mut conn, 5).expect("truncate");
        assert_eq!(deleted, 5);
        assert_eq!(
            count_search_history(&conn).expect("count after truncate"),
            5
        );

        // Remaining entries should be the 5 most recent.
        let entries = list_search_history(&conn, 10).expect("list");
        assert_eq!(entries[0].query, "query-9");
        assert_eq!(entries[4].query, "query-5");
    }

    #[test]
    fn history_truncation_noop_when_under_limit() {
        let mut conn = test_conn();
        record_search(
            &mut conn,
            "only-one",
            None,
            None,
            None,
            None,
            None,
            1_700_000_000,
            0,
        )
        .expect("record");

        let deleted = truncate_search_history(&mut conn, 100).expect("truncate");
        assert_eq!(deleted, 0);
        assert_eq!(count_search_history(&conn).expect("count"), 1);
    }

    #[test]
    fn history_prefix_search() {
        let mut conn = test_conn();
        for q in ["rust async", "rust ownership", "python async"] {
            record_search(&mut conn, q, None, None, None, None, None, 1_700_000_000, 0)
                .expect("record");
        }

        let rust_results = search_history_prefix(&conn, "rust", 10).expect("prefix search");
        assert_eq!(rust_results.len(), 2);

        let py_results = search_history_prefix(&conn, "python", 10).expect("prefix search");
        assert_eq!(py_results.len(), 1);

        let no_results = search_history_prefix(&conn, "golang", 10).expect("prefix search");
        assert!(no_results.is_empty());
    }

    #[test]
    fn history_prefix_search_escapes_wildcards() {
        let mut conn = test_conn();
        for q in ["100% coverage", "100 percent", "abc_def", "abcXdef"] {
            record_search(&mut conn, q, None, None, None, None, None, 1_700_000_000, 0)
                .expect("record");
        }

        let percent_results = search_history_prefix(&conn, "100%", 10).expect("prefix search");
        assert_eq!(percent_results.len(), 1);
        assert_eq!(percent_results[0].query, "100% coverage");

        let underscore_results = search_history_prefix(&conn, "abc_", 10).expect("prefix search");
        assert_eq!(underscore_results.len(), 1);
        assert_eq!(underscore_results[0].query, "abc_def");
    }

    // ─── Bookmark Tests ────────────────────────────────────────────────

    #[test]
    fn add_and_list_bookmarks() {
        let mut conn = test_conn();
        add_bookmark(
            &mut conn,
            "doc-123",
            Some("rust async"),
            Some("great explanation"),
            1_700_000_000,
        )
        .expect("add bookmark");

        let bookmarks = list_bookmarks(&conn, 10).expect("list");
        assert_eq!(bookmarks.len(), 1);
        assert_eq!(bookmarks[0].doc_id, "doc-123");
        assert_eq!(bookmarks[0].query.as_deref(), Some("rust async"));
        assert_eq!(bookmarks[0].note.as_deref(), Some("great explanation"));
    }

    #[test]
    fn bookmark_same_doc_different_queries() {
        let mut conn = test_conn();
        add_bookmark(
            &mut conn,
            "doc-1",
            Some("rust async"),
            Some("note-a"),
            1_700_000_000,
        )
        .expect("first bookmark");

        add_bookmark(
            &mut conn,
            "doc-1",
            Some("tokio spawn"),
            Some("note-b"),
            1_700_000_060,
        )
        .expect("second bookmark");

        let count = count_bookmarks(&conn).expect("count");
        assert_eq!(
            count, 2,
            "same doc from different queries should create separate bookmarks"
        );
    }

    #[test]
    fn bookmark_upsert_same_doc_and_query() {
        let mut conn = test_conn();
        add_bookmark(
            &mut conn,
            "doc-1",
            Some("rust async"),
            Some("old note"),
            1_700_000_000,
        )
        .expect("first");

        add_bookmark(
            &mut conn,
            "doc-1",
            Some("rust async"),
            Some("updated note"),
            1_700_000_300,
        )
        .expect("upsert");

        let count = count_bookmarks(&conn).expect("count");
        assert_eq!(count, 1, "same (doc_id, query) should upsert");

        let bookmarks = list_bookmarks(&conn, 10).expect("list");
        assert_eq!(
            bookmarks[0].note.as_deref(),
            Some("updated note"),
            "note should be updated"
        );
    }

    #[test]
    fn remove_bookmark_by_id() {
        let mut conn = test_conn();
        add_bookmark(&mut conn, "doc-rm", None, None, 1_700_000_000).expect("add");

        let bookmarks = list_bookmarks(&conn, 10).expect("list");
        assert_eq!(bookmarks.len(), 1);

        let removed = remove_bookmark(&mut conn, bookmarks[0].id).expect("remove");
        assert!(removed);
        assert_eq!(count_bookmarks(&conn).expect("count"), 0);
    }

    #[test]
    fn remove_bookmark_by_id_returns_false_when_missing() {
        let mut conn = test_conn();
        add_bookmark(&mut conn, "doc-rm-missing", None, None, 1_700_000_000).expect("add");
        let removed = remove_bookmark(&mut conn, 999_999).expect("remove missing");
        assert!(!removed, "missing bookmark id should report false");
        assert_eq!(
            count_bookmarks(&conn).expect("count"),
            1,
            "existing bookmark should remain untouched"
        );
    }

    #[test]
    fn remove_bookmark_by_doc_id() {
        let mut conn = test_conn();
        add_bookmark(&mut conn, "doc-rm2", Some("q1"), None, 1_700_000_000).expect("add 1");
        add_bookmark(&mut conn, "doc-rm2", Some("q2"), None, 1_700_000_060).expect("add 2");

        assert_eq!(count_bookmarks(&conn).expect("count"), 2);
        remove_bookmark_by_doc(&mut conn, "doc-rm2").expect("remove by doc");
        assert_eq!(count_bookmarks(&conn).expect("count"), 0);
    }

    #[test]
    fn is_bookmarked_check() {
        let mut conn = test_conn();
        assert!(!is_bookmarked(&conn, "doc-x").expect("check"));

        add_bookmark(&mut conn, "doc-x", None, None, 1_700_000_000).expect("add");
        assert!(is_bookmarked(&conn, "doc-x").expect("check after add"));
    }

    #[test]
    fn bookmark_mru_order() {
        let mut conn = test_conn();
        for (i, ts) in [1_700_000_000i64, 1_700_000_060, 1_700_000_120]
            .iter()
            .enumerate()
        {
            add_bookmark(&mut conn, &format!("doc-{i}"), None, None, *ts).expect("add");
        }

        let bookmarks = list_bookmarks(&conn, 10).expect("list");
        assert_eq!(bookmarks[0].doc_id, "doc-2", "most recent first");
        assert_eq!(bookmarks[2].doc_id, "doc-0", "oldest last");
    }

    // ─── Transaction rollback tests ────────────────────────────────────

    #[test]
    fn history_transaction_rolls_back_and_restores_autocommit_on_error() {
        let mut conn = test_conn();
        assert!(conn.is_autocommit());

        let result: SearchResult<()> = with_history_transaction(&mut conn, |conn| {
            conn.execute(
                "INSERT INTO bookmarks (doc_id, query, note, created_at) \
                 VALUES ('doc-tx', NULL, NULL, 1);",
                [],
            )
            .map_err(map_storage_error)?;
            Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other("forced history failure")),
            })
        });

        assert!(result.is_err(), "closure error must propagate");
        assert!(
            conn.is_autocommit(),
            "failed history transaction must restore autocommit"
        );
        assert_eq!(
            count_bookmarks(&conn).expect("count"),
            0,
            "rolled-back bookmark insert must not persist"
        );
    }

    #[test]
    fn history_transaction_rolls_back_and_restores_autocommit_on_panic() {
        let mut conn = test_conn();
        assert!(conn.is_autocommit());

        let result = catch_unwind(AssertUnwindSafe(|| {
            with_history_transaction(&mut conn, |_conn| -> SearchResult<()> {
                panic!("forced history panic")
            })
        }));

        assert!(result.is_err(), "panic must propagate to the caller");
        assert!(
            conn.is_autocommit(),
            "panicked history transaction must restore autocommit"
        );
    }
}
