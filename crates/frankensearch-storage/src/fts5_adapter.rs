//! FTS5 alternative lexical search adapter.
//!
//! Uses SQLite's built-in FTS5 implementation through `rusqlite` as an
//! alternative to Tantivy for BM25 full-text search. Both implement the split
//! [`frankensearch_core::LexicalRead`] / [`frankensearch_core::LexicalWrite`]
//! trait from `frankensearch-core`.
//!
//! # Advantages over Tantivy
//!
//! - Zero additional binary size (FTS5 is part of SQLite)
//! - MVCC concurrent reads and writes
//! - Single deployment artifact (one `.db` file)
//!
//! # Content mode
//!
//! The in-memory adapter below supports `Stored` and `Contentless` modes.
//! `External` shares the `Stored` layout because an in-memory index has no
//! separately governed content table. Persisted tables are inspected from
//! their SQLite metadata so that `Stored`, external-content, and contentless
//! layouts are never inferred from caller configuration.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use asupersync::Cx;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::SearchFuture;
use frankensearch_core::types::{IndexableDocument, ScoreSource, ScoredResult};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, Row, params};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::connection::{Storage, map_storage_error_at, retry_transient_storage_async};
use crate::schema::PORTER_FTS5_REBUILD_TABLE;

// ─── Constants ──────────────────────────────────────────────────────────────

/// BM25 boost applied to title field matches (mirrors Tantivy adapter).
const TITLE_BOOST: f64 = 2.0;

/// Maximum query length in characters before truncation.
const MAX_QUERY_LENGTH: usize = 10_000;

/// Default snippet window size in tokens.
const DEFAULT_SNIPPET_TOKENS: usize = 20;

/// The on-disk marker written after applying the Porter rebuild.
///
/// A Porter index built before the tokenizer contract was governed must be
/// rebuilt from its complete content source; accepting an unmarked table
/// would make terms silently unfindable.
pub const PORTER_FTS5_REBUILD_VERSION: i64 = 1;

/// Column index: content (primary search field).
const COL_CONTENT: usize = 2;

/// Name of the FTS5 virtual table inside the in-memory adapter connection.
const MEM_FTS5_TABLE: &str = "fts5_lexical";

// ─── Configuration ──────────────────────────────────────────────────────────

/// Content storage mode for the FTS5 index.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fts5ContentMode {
    /// FTS5 stores its own copy of the content (supports snippets).
    #[default]
    Stored,
    /// FTS5 indexes a separately governed content table.
    External,
    /// Index-only mode — no content retrieval or snippet support.
    Contentless,
}

/// Tokenizer selection for the FTS5 index.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fts5TokenizerChoice {
    /// Unicode-aware tokenizer with optional diacritic removal.
    #[default]
    Unicode61,
    /// English Porter stemming (wraps unicode61).
    Porter,
    /// Trigram tokenizer for substring matching (slower but more flexible).
    Trigram,
}

impl Fts5TokenizerChoice {
    /// The `tokenize=` option text accepted by `CREATE VIRTUAL TABLE ... fts5`.
    const fn fts5_tokenize(self) -> &'static str {
        match self {
            Self::Unicode61 => "unicode61",
            Self::Porter => "porter unicode61",
            Self::Trigram => "trigram",
        }
    }
}

/// Configuration for the FTS5 lexical search adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fts5AdapterConfig {
    /// Content storage mode.
    #[serde(default)]
    pub content_mode: Fts5ContentMode,
    /// Tokenizer to use.
    #[serde(default)]
    pub tokenizer: Fts5TokenizerChoice,
    /// BM25 boost for title field matches.
    #[serde(default = "default_title_boost")]
    pub title_boost: f64,
}

fn default_title_boost() -> f64 {
    TITLE_BOOST
}

impl Default for Fts5AdapterConfig {
    fn default() -> Self {
        Self {
            content_mode: Fts5ContentMode::default(),
            tokenizer: Fts5TokenizerChoice::default(),
            title_boost: TITLE_BOOST,
        }
    }
}

// ─── Row ID mapping ─────────────────────────────────────────────────────────

/// Maps between string `doc_ids` and the i64 rowids used by the FTS5 table.
///
/// The map is required for `Contentless` tables (whose columns cannot be
/// read back) and keeps `doc_count` and deletions uniform across modes.
///
/// `Clone` lets writers stage map mutations inside a `rusqlite` transaction
/// and publish them only after `COMMIT` succeeds, so a rolled-back batch can
/// never desynchronize the map from the FTS5 table.
#[derive(Debug, Default, Clone)]
struct RowIdMap {
    doc_to_row: HashMap<String, i64>,
    row_to_doc: HashMap<i64, String>,
    next_rowid: i64,
}

impl RowIdMap {
    fn new() -> Self {
        Self {
            doc_to_row: HashMap::new(),
            row_to_doc: HashMap::new(),
            next_rowid: 1,
        }
    }

    fn get_or_assign(&mut self, doc_id: &str) -> i64 {
        if let Some(&rowid) = self.doc_to_row.get(doc_id) {
            return rowid;
        }
        let rowid = self.next_rowid;
        self.next_rowid += 1;
        self.doc_to_row.insert(doc_id.to_owned(), rowid);
        self.row_to_doc.insert(rowid, doc_id.to_owned());
        rowid
    }

    fn get_rowid(&self, doc_id: &str) -> Option<i64> {
        self.doc_to_row.get(doc_id).copied()
    }

    fn get_doc_id(&self, rowid: i64) -> Option<&str> {
        self.row_to_doc.get(&rowid).map(String::as_str)
    }

    fn remove(&mut self, doc_id: &str) -> Option<i64> {
        if let Some(rowid) = self.doc_to_row.remove(doc_id) {
            self.row_to_doc.remove(&rowid);
            Some(rowid)
        } else {
            None
        }
    }
}

// ─── FTS5 Lexical Search ────────────────────────────────────────────────────

/// FTS5-backed implementation of the split lexical capabilities.
///
/// Indexes documents into a real SQLite FTS5 virtual table on an in-memory
/// `rusqlite::Connection` and runs BM25-ranked `MATCH` queries against it.
/// Thread-safe via internal `Mutex`.
pub struct Fts5LexicalSearch {
    conn: Mutex<Connection>,
    rowid_map: Mutex<RowIdMap>,
    config: Fts5AdapterConfig,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Fts5LexicalSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fts5LexicalSearch")
            .field("config", &self.config)
            .finish()
    }
}

impl Fts5LexicalSearch {
    /// Create a new FTS5 lexical search instance.
    ///
    /// # Panics
    ///
    /// Panics if the embedded SQLite cannot create an FTS5 virtual table,
    /// meaning the `rusqlite` build lacks the FTS5 module.
    #[must_use]
    pub fn new(config: Fts5AdapterConfig) -> Self {
        let conn = Connection::open_in_memory().expect("in-memory rusqlite connection should open");

        // `External` degrades to the contentful layout: an in-memory index
        // has no separately governed content table. `Contentless` enables
        // `contentless_delete` so rowid-scoped deletes stay available.
        let content_option = match config.content_mode {
            Fts5ContentMode::Stored | Fts5ContentMode::External => "",
            Fts5ContentMode::Contentless => ", content='', contentless_delete=1",
        };
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE {MEM_FTS5_TABLE} USING fts5(\
             doc_id UNINDEXED, title, content, metadata_json UNINDEXED, \
             tokenize='{}'{content_option});",
            config.tokenizer.fts5_tokenize()
        ))
        .expect("in-memory FTS5 table creation should succeed");

        Self {
            conn: Mutex::new(conn),
            rowid_map: Mutex::new(RowIdMap::new()),
            config,
        }
    }

    /// Create a new FTS5 lexical search instance with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(Fts5AdapterConfig::default())
    }

    /// Get the adapter configuration.
    #[must_use]
    pub fn config(&self) -> &Fts5AdapterConfig {
        &self.config
    }

    /// Truncate overly long queries to prevent pathological parsing.
    fn truncate_query(query: &str) -> &str {
        if query.len() <= MAX_QUERY_LENGTH {
            return query;
        }

        let Some((end, _)) = query.char_indices().nth(MAX_QUERY_LENGTH) else {
            return query;
        };
        warn!(
            original_len_bytes = query.len(),
            max_chars = MAX_QUERY_LENGTH,
            "fts5: query truncated"
        );
        &query[..end]
    }

    /// Build column values from an `IndexableDocument`.
    fn doc_to_columns(doc: &IndexableDocument) -> Vec<String> {
        let metadata_json = if doc.metadata.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&doc.metadata).unwrap_or_default()
        };

        vec![
            doc.id.clone(),
            doc.title.clone().unwrap_or_default(),
            doc.content.clone(),
            metadata_json,
        ]
    }

    /// The `metadata_json` select expression for the configured content mode.
    ///
    /// Column values cannot be read back from a contentless table, so a NULL
    /// literal keeps the result shape identical without touching the table.
    fn metadata_select(&self) -> &'static str {
        match self.config.content_mode {
            Fts5ContentMode::Contentless => "NULL",
            Fts5ContentMode::Stored | Fts5ContentMode::External => "metadata_json",
        }
    }

    /// Upsert one document into the FTS5 table under its mapped rowid.
    fn index_document_on(
        conn: &Connection,
        rowid_map: &mut RowIdMap,
        doc: &IndexableDocument,
    ) -> SearchResult<()> {
        if let Some(old_rowid) = rowid_map.get_rowid(&doc.id) {
            conn.execute(
                &format!("DELETE FROM {MEM_FTS5_TABLE} WHERE rowid = ?1;"),
                params![old_rowid],
            )
            .map_err(|error| map_storage_error_at("fts5 remove document for upsert", error))?;
        }

        let rowid = rowid_map.get_or_assign(&doc.id);
        let columns = Self::doc_to_columns(doc);
        conn.execute(
            &format!(
                "INSERT INTO {MEM_FTS5_TABLE}(rowid, doc_id, title, content, metadata_json) \
                 VALUES (?1, ?2, ?3, ?4, ?5);"
            ),
            params![rowid, columns[0], columns[1], columns[2], columns[3]],
        )
        .map_err(|error| map_storage_error_at("fts5 index document", error))?;
        Ok(())
    }

    /// Search with snippet generation (richer result type).
    #[allow(clippy::significant_drop_tightening)]
    pub fn search_with_snippets(&self, query: &str, limit: usize) -> SearchResult<Vec<Fts5Hit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| SearchError::InvalidConfig {
            field: "fts5.limit".to_owned(),
            value: limit.to_string(),
            reason: "does not fit SQLite's signed integer limit".to_owned(),
        })?;

        let conn = self.conn.lock().map_err(lock_error)?;
        let rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        // `snippet()` cannot be served by a contentless table; keep a NULL
        // placeholder there so the result shape stays the same.
        let snippet_select = match self.config.content_mode {
            Fts5ContentMode::Contentless => "NULL".to_owned(),
            Fts5ContentMode::Stored | Fts5ContentMode::External => format!(
                "snippet({MEM_FTS5_TABLE}, {COL_CONTENT}, '<b>', '</b>', '...', {DEFAULT_SNIPPET_TOKENS})"
            ),
        };
        let sql = format!(
            "SELECT rowid, bm25({0}, 0.0, ?2, 1.0, 0.0), {1}, {snippet_select} \
             FROM {0} WHERE {0} MATCH ?1 \
             ORDER BY bm25({0}, 0.0, ?2, 1.0, 0.0), rowid LIMIT ?3;",
            MEM_FTS5_TABLE,
            self.metadata_select(),
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|error| SearchError::QueryParseError {
                query: query.to_owned(),
                detail: error.to_string(),
            })?;
        let rows = stmt
            .query_map(
                params![query, self.config.title_boost, limit],
                |row| -> rusqlite::Result<(i64, f64, Option<String>, Option<String>)> {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                },
            )
            .map_err(|error| SearchError::QueryParseError {
                query: query.to_owned(),
                detail: error.to_string(),
            })?;

        let mut hits = Vec::new();
        for (rank, row) in rows.enumerate() {
            let (rowid, raw_score, metadata_json, snippet) =
                row.map_err(|error| SearchError::QueryParseError {
                    query: query.to_owned(),
                    detail: error.to_string(),
                })?;
            let doc_id = rowid_map.get_doc_id(rowid).unwrap_or("").to_owned();

            // FTS5 scores are negative (lower = better). Negate for positive.
            #[allow(clippy::cast_possible_truncation)]
            let bm25_score = (-raw_score) as f32;

            let metadata = metadata_json
                .filter(|s| !s.is_empty())
                .and_then(|s| serde_json::from_str(&s).ok());

            hits.push(Fts5Hit {
                doc_id,
                bm25_score,
                rank,
                snippet,
                metadata,
            });
        }

        debug!(hits = hits.len(), query, "fts5 search completed");
        Ok(hits)
    }

    /// Delete a single document by ID.
    ///
    /// Returns `true` if the document existed and was removed.
    pub fn delete_document(&self, doc_id: &str) -> SearchResult<bool> {
        let conn = self.conn.lock().map_err(lock_error)?;
        let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        let Some(rowid) = rowid_map.get_rowid(doc_id) else {
            return Ok(false);
        };
        conn.execute(
            &format!("DELETE FROM {MEM_FTS5_TABLE} WHERE rowid = ?1;"),
            params![rowid],
        )
        .map_err(|error| map_storage_error_at("fts5 delete document", error))?;
        // The map entry is dropped only after the row is gone, so a failed
        // DELETE cannot orphan the document id in `rowid_map`.
        rowid_map.remove(doc_id);
        // Keep the connection and ID map locked until both mutations complete.
        drop(rowid_map);
        drop(conn);
        Ok(true)
    }

    /// Delete all indexed documents.
    pub fn clear(&self) -> SearchResult<()> {
        let conn = self.conn.lock().map_err(lock_error)?;
        let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        // A contentless table only accepts the `delete-all` command; a
        // contentful table takes a plain unqualified DELETE.
        let clear_sql = match self.config.content_mode {
            Fts5ContentMode::Contentless => {
                format!("INSERT INTO {MEM_FTS5_TABLE}({MEM_FTS5_TABLE}) VALUES('delete-all');")
            }
            Fts5ContentMode::Stored | Fts5ContentMode::External => {
                format!("DELETE FROM {MEM_FTS5_TABLE};")
            }
        };
        conn.execute(&clear_sql, [])
            .map_err(|error| map_storage_error_at("fts5 clear", error))?;
        rowid_map.doc_to_row.clear();
        rowid_map.row_to_doc.clear();
        drop(rowid_map);
        drop(conn);

        debug!("fts5: cleared all documents");
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PersistedFts5Metadata {
    content_mode: Fts5ContentMode,
    tokenizer: String,
}

/// A read-only, persisted FTS5 search path.
///
/// Unlike [`Fts5LexicalSearch`], this reader executes `MATCH` against the
/// virtual table in the supplied [`Storage`] database. It revalidates the
/// table DDL and the governed rebuild marker on every open and search, so an
/// unrebuilt Porter index cannot be queried through an in-memory side path.
/// The table must expose `doc_id` and `metadata_json` columns; those are the
/// persisted storage contract required to produce `ScoredResult` values.
pub struct PersistedFts5LexicalSearch {
    storage: Arc<Storage>,
    table_name: String,
    config: Fts5AdapterConfig,
}

impl std::fmt::Debug for PersistedFts5LexicalSearch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PersistedFts5LexicalSearch")
            .field("table_name", &self.table_name)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PersistedFts5LexicalSearch {
    /// Open a governed persisted Porter FTS5 table.
    ///
    /// This checks the table's actual `sqlite_master` DDL rather than trusting
    /// an application-supplied content mode or tokenizer. It also requires the
    /// committed rebuild marker before returning a searchable reader.
    // The signature stays `async` so callers pin a stable `Future` even though
    // the rusqlite work inside is synchronous.
    #[allow(clippy::unused_async_trait_impl)]
    pub async fn open(cx: &Cx, storage: Arc<Storage>, table_name: &str) -> SearchResult<Self> {
        let table_name = validated_fts5_identifier(table_name)?;
        fts5_checkpoint(cx, "fts5.persisted.open")?;
        let metadata =
            storage.with_connection(|conn| ensure_porter_fts5_ready(conn, &table_name))?;
        Ok(Self {
            storage,
            table_name,
            config: Fts5AdapterConfig {
                content_mode: metadata.content_mode,
                tokenizer: Fts5TokenizerChoice::Porter,
                title_boost: TITLE_BOOST,
            },
        })
    }

    /// Return the verified table configuration read from persisted metadata.
    #[must_use]
    pub fn config(&self) -> &Fts5AdapterConfig {
        &self.config
    }

    /// Query the current persisted table count within a caller-owned runtime
    /// context.
    ///
    /// This is the authoritative alternative to the synchronous
    /// [`frankensearch_core::LexicalRead::doc_count`] capability, which cannot
    /// pin or await this adapter's live storage connection.
    ///
    /// # Errors
    ///
    /// Returns a storage or result-shape error when the current table count
    /// cannot be established.
    pub async fn current_doc_count(&self, cx: &Cx) -> SearchResult<usize> {
        retry_transient_storage_async(
            cx,
            || async {
                self.storage.with_connection(|conn| {
                    persisted_fts5_verified_doc_count(conn, &self.table_name)
                })
            },
            "persisted fts5 count",
        )
        .await
    }
}

impl frankensearch_core::LexicalRead for PersistedFts5LexicalSearch {
    #[instrument(skip_all, fields(table = %self.table_name, query = %query, limit = limit))]
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            fts5_checkpoint(cx, "fts5.persisted.search")?;
            let query = Fts5LexicalSearch::truncate_query(query);
            if query.trim().is_empty() {
                return Ok(Vec::new());
            }

            let limit = i64::try_from(limit).map_err(|_| SearchError::InvalidConfig {
                field: "fts5.limit".to_owned(),
                value: limit.to_string(),
                reason: "does not fit SQLite's signed integer limit".to_owned(),
            })?;
            let sql = format!(
                "SELECT doc_id, metadata_json, bm25({0}) FROM {0} \
                         WHERE {0} MATCH ?1 ORDER BY bm25({0}), rowid LIMIT ?2;",
                self.table_name
            );
            retry_transient_storage_async(
                cx,
                || async {
                    // This is intentionally in the live data path, not only in
                    // the rebuild helper: external DDL or marker changes fail
                    // the search closed before MATCH can return a stale Porter
                    // result.
                    self.storage.with_connection(|conn| {
                        ensure_porter_fts5_ready(conn, &self.table_name)?;
                        let mut stmt = conn.prepare(&sql).map_err(|error| {
                            map_storage_error_at("persisted Porter FTS5 search", error)
                        })?;
                        let mut rows = stmt.query(params![query, limit]).map_err(|error| {
                            map_storage_error_at("persisted Porter FTS5 search", error)
                        })?;
                        let mut results = Vec::new();
                        while let Some(row) = rows.next().map_err(|error| {
                            map_storage_error_at("persisted Porter FTS5 search", error)
                        })? {
                            results.push(decode_persisted_fts5_row(row)?);
                        }
                        Ok(results)
                    })
                },
                "persisted fts5 search",
            )
            .await
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Err(SearchError::SubsystemError {
            subsystem: "fts5",
            source: "the persisted FTS5 adapter has no synchronous pinned-generation count; \
                     call PersistedFts5LexicalSearch::current_doc_count with the caller's Cx"
                .into(),
        })
    }
}

/// Rebuild a persisted Porter FTS5 table.
///
/// The table's own DDL decides its content mode. Ordinary stored and external
/// tables use FTS5's `rebuild` command; contentless tables are rejected because
/// authoritative text and original rowids must be re-ingested instead.
///
/// Rebuild and marker promotion share one `BEGIN IMMEDIATE` worker transaction
/// through [`Storage::immediate_transaction`]. Transient SQLite contention
/// retries the whole transaction; a panic rolls back once before unwinding
/// continues.
pub async fn rebuild_porter_fts5_table(
    cx: &Cx,
    storage: &Storage,
    table_name: &str,
) -> SearchResult<()> {
    let table_name = validated_fts5_identifier(table_name)?;
    let metadata =
        storage.with_connection(|conn| read_persisted_fts5_metadata(conn, &table_name))?;
    ensure_rebuildable_porter_fts5(&table_name, &metadata)?;

    match storage.with_connection(|conn| read_porter_fts5_rebuild_marker(conn, &table_name))? {
        Some(PORTER_FTS5_REBUILD_VERSION) => return Ok(()),
        Some(version) if version > PORTER_FTS5_REBUILD_VERSION => {
            return Err(SearchError::InvalidConfig {
                field: "fts5.rebuild_version".to_owned(),
                value: version.to_string(),
                reason:
                    "database was rebuilt by a newer Porter FTS5 migration; refusing a downgrade"
                        .to_owned(),
            });
        }
        Some(_) | None => {}
    }

    fts5_checkpoint(cx, "porter fts5 rebuild")?;
    storage.immediate_transaction(|tx| {
        tx.execute(
            &format!("INSERT INTO {table_name}({table_name}) VALUES ('rebuild');"),
            [],
        )
        .map_err(|error| map_storage_error_at("rebuild Porter FTS5 table", error))?;

        tx.execute(
            &format!(
                "INSERT INTO {PORTER_FTS5_REBUILD_TABLE} (table_name, rebuild_version) \
                 VALUES (?1, ?2) \
                 ON CONFLICT(table_name) DO UPDATE SET rebuild_version = excluded.rebuild_version;"
            ),
            params![table_name, PORTER_FTS5_REBUILD_VERSION],
        )
        .map_err(|error| map_storage_error_at("write Porter FTS5 rebuild marker", error))?;
        Ok(())
    })
}

fn validated_fts5_identifier(table_name: &str) -> SearchResult<String> {
    let mut chars = table_name.chars();
    let valid_start = chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic());
    let valid_rest = chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric());
    if valid_start && valid_rest {
        Ok(table_name.to_owned())
    } else {
        Err(SearchError::InvalidConfig {
            field: "fts5.table_name".to_owned(),
            value: table_name.to_owned(),
            reason: "must be a SQLite ASCII identifier before it is interpolated into a rebuild statement".to_owned(),
        })
    }
}

fn ensure_porter_fts5_ready(
    conn: &Connection,
    table_name: &str,
) -> SearchResult<PersistedFts5Metadata> {
    let metadata = read_persisted_fts5_metadata(conn, table_name)?;
    ensure_rebuildable_porter_fts5(table_name, &metadata)?;
    ensure_porter_fts5_rebuild_version(
        table_name,
        read_porter_fts5_rebuild_marker(conn, table_name)?,
    )?;
    Ok(metadata)
}

fn ensure_porter_fts5_rebuild_version(
    table_name: &str,
    rebuild_version: Option<i64>,
) -> SearchResult<()> {
    match rebuild_version {
        Some(PORTER_FTS5_REBUILD_VERSION) => Ok(()),
        Some(version) if version > PORTER_FTS5_REBUILD_VERSION => Err(SearchError::InvalidConfig {
            field: "fts5.rebuild_version".to_owned(),
            value: version.to_string(),
            reason: "database was rebuilt by a newer Porter FTS5 migration; refusing a downgrade".to_owned(),
        }),
        Some(version) => Err(SearchError::InvalidConfig {
            field: "fts5.rebuild_version".to_owned(),
            value: version.to_string(),
            reason: "Porter FTS5 table has an obsolete rebuild marker; rebuild must complete and commit before search".to_owned(),
        }),
        None => Err(SearchError::InvalidConfig {
            field: "fts5.rebuild_version".to_owned(),
            value: table_name.to_owned(),
            reason: "Porter FTS5 table has no committed rebuild marker; refusing potentially stale search results".to_owned(),
        }),
    }
}

fn read_persisted_fts5_metadata(
    conn: &Connection,
    table_name: &str,
) -> SearchResult<PersistedFts5Metadata> {
    let mut stmt = conn
        .prepare("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?1;")
        .map_err(|error| map_storage_error_at("read persisted FTS5 table metadata", error))?;
    let mut rows = stmt
        .query(params![table_name])
        .map_err(|error| map_storage_error_at("read persisted FTS5 table metadata", error))?;

    let mut create_sql = None;
    while let Some(row) = rows
        .next()
        .map_err(|error| map_storage_error_at("read persisted FTS5 table metadata", error))?
    {
        if create_sql.is_some() {
            return Err(persisted_fts5_metadata_error(
                table_name,
                "sqlite_master returned more than one table definition",
            ));
        }
        create_sql = Some(match row.get_ref(0) {
            Ok(ValueRef::Text(sql)) => std::str::from_utf8(sql)
                .map_err(|_| {
                    persisted_fts5_metadata_error(table_name, "sqlite_master.sql is not text")
                })?
                .to_owned(),
            _ => {
                return Err(persisted_fts5_metadata_error(
                    table_name,
                    "sqlite_master.sql is not text",
                ));
            }
        });
    }
    let Some(create_sql) = create_sql else {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "table is absent from sqlite_master",
        ));
    };
    parse_persisted_fts5_metadata(table_name, &create_sql)
}

fn read_porter_fts5_rebuild_marker(
    conn: &Connection,
    table_name: &str,
) -> SearchResult<Option<i64>> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT rebuild_version FROM {PORTER_FTS5_REBUILD_TABLE} WHERE table_name = ?1;"
        ))
        .map_err(|error| map_storage_error_at("read Porter FTS5 rebuild marker", error))?;
    let mut rows = stmt
        .query(params![table_name])
        .map_err(|error| map_storage_error_at("read Porter FTS5 rebuild marker", error))?;

    let mut rebuild_version = None;
    while let Some(row) = rows
        .next()
        .map_err(|error| map_storage_error_at("read Porter FTS5 rebuild marker", error))?
    {
        if rebuild_version.is_some() {
            return Err(SearchError::InvalidConfig {
                field: "fts5.rebuild_version".to_owned(),
                value: table_name.to_owned(),
                reason: "governed Porter FTS5 marker table contains duplicate rows for one table"
                    .to_owned(),
            });
        }
        rebuild_version = Some(match row.get_ref(0) {
            Ok(ValueRef::Integer(version)) => version,
            Ok(value) => {
                return Err(SearchError::InvalidConfig {
                    field: "fts5.rebuild_version".to_owned(),
                    value: format!("{table_name}: {value:?}"),
                    reason: "refusing to query a Porter FTS5 table whose governed rebuild marker is not an integer".to_owned(),
                });
            }
            Err(_) => {
                return Err(SearchError::InvalidConfig {
                    field: "fts5.rebuild_version".to_owned(),
                    value: format!("{table_name}: missing column"),
                    reason: "refusing to query a Porter FTS5 table whose governed rebuild marker row is malformed".to_owned(),
                });
            }
        });
    }
    Ok(rebuild_version)
}

fn ensure_rebuildable_porter_fts5(
    table_name: &str,
    metadata: &PersistedFts5Metadata,
) -> SearchResult<()> {
    if !metadata
        .tokenizer
        .split_ascii_whitespace()
        .next()
        .is_some_and(|tokenizer| tokenizer.eq_ignore_ascii_case("porter"))
    {
        return Err(SearchError::InvalidConfig {
            field: "fts5.tokenize".to_owned(),
            value: metadata.tokenizer.clone(),
            reason: format!(
                "{table_name} is not a Porter FTS5 table according to its persisted sqlite_master definition"
            ),
        });
    }
    if metadata.content_mode == Fts5ContentMode::Contentless {
        return Err(SearchError::InvalidConfig {
            field: "fts5.content_mode".to_owned(),
            value: "contentless".to_owned(),
            reason: "Porter FTS5 rebuild requires authoritative text and original rowids; recreate the contentless table and re-ingest source documents rather than rebuilding from an index or preview".to_owned(),
        });
    }
    Ok(())
}

fn persisted_fts5_verified_doc_count(conn: &Connection, table_name: &str) -> SearchResult<usize> {
    let sql = format!(
        "SELECT schema_entry.sql, \
                (SELECT COUNT(*) FROM {PORTER_FTS5_REBUILD_TABLE} \
                 WHERE table_name = ?1), \
                (SELECT MIN(rebuild_version) FROM {PORTER_FTS5_REBUILD_TABLE} \
                 WHERE table_name = ?1), \
                (SELECT COUNT(*) FROM {table_name}) \
         FROM sqlite_master AS schema_entry \
         WHERE schema_entry.type = 'table' AND schema_entry.name = ?1;"
    );
    let mut stmt = conn.prepare(&sql).map_err(|error| {
        map_storage_error_at("verify and count persisted Porter FTS5 documents", error)
    })?;
    let mut rows = stmt.query(params![table_name]).map_err(|error| {
        map_storage_error_at("verify and count persisted Porter FTS5 documents", error)
    })?;

    let (create_sql, marker_count, rebuild_version, count) = match rows.next().map_err(|error| {
        map_storage_error_at("verify and count persisted Porter FTS5 documents", error)
    })? {
        Some(row) => {
            let create_sql = match row.get_ref(0) {
                Ok(ValueRef::Text(sql)) => std::str::from_utf8(sql)
                    .map_err(|_| {
                        persisted_fts5_metadata_error(table_name, "sqlite_master.sql is not text")
                    })?
                    .to_owned(),
                _ => {
                    return Err(persisted_fts5_metadata_error(
                        table_name,
                        "sqlite_master.sql is not text",
                    ));
                }
            };
            let Ok(ValueRef::Integer(marker_count)) = row.get_ref(1) else {
                return Err(SearchError::InvalidConfig {
                    field: "fts5.rebuild_version".to_owned(),
                    value: table_name.to_owned(),
                    reason: "governed Porter FTS5 marker cardinality is not an integer".to_owned(),
                });
            };
            let rebuild_version = match row.get_ref(2) {
                Ok(ValueRef::Integer(version)) => Some(version),
                Ok(ValueRef::Null) => None,
                Ok(value) => {
                    return Err(SearchError::InvalidConfig {
                        field: "fts5.rebuild_version".to_owned(),
                        value: format!("{table_name}: {value:?}"),
                        reason: "refusing to count a Porter FTS5 table whose governed rebuild marker is not an integer".to_owned(),
                    });
                }
                Err(_) => {
                    return Err(persisted_fts5_metadata_error(
                        table_name,
                        "verification result is missing the rebuild marker column",
                    ));
                }
            };
            let Ok(ValueRef::Integer(count)) = row.get_ref(3) else {
                return Err(persisted_fts5_metadata_error(
                    table_name,
                    "COUNT(*) did not return an integer",
                ));
            };
            (create_sql, marker_count, rebuild_version, count)
        }
        None => {
            return Err(persisted_fts5_metadata_error(
                table_name,
                "table is absent from sqlite_master",
            ));
        }
    };
    if rows
        .next()
        .map_err(|error| {
            map_storage_error_at("verify and count persisted Porter FTS5 documents", error)
        })?
        .is_some()
    {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "verification and COUNT(*) returned more than one row",
        ));
    }

    let metadata = parse_persisted_fts5_metadata(table_name, &create_sql)?;
    ensure_rebuildable_porter_fts5(table_name, &metadata)?;

    match marker_count {
        0 => ensure_porter_fts5_rebuild_version(table_name, None)?,
        1 => {}
        duplicates => {
            return Err(SearchError::InvalidConfig {
                field: "fts5.rebuild_version".to_owned(),
                value: format!("{table_name}: {duplicates} rows"),
                reason: "governed Porter FTS5 marker table contains duplicate rows for one table"
                    .to_owned(),
            });
        }
    }
    ensure_porter_fts5_rebuild_version(table_name, rebuild_version)?;

    usize::try_from(count).map_err(|_| {
        persisted_fts5_metadata_error(table_name, "COUNT(*) is negative or does not fit usize")
    })
}

fn decode_persisted_fts5_row(row: &Row<'_>) -> SearchResult<ScoredResult> {
    let doc_id = match row.get_ref(0) {
        Ok(ValueRef::Text(bytes)) => std::str::from_utf8(bytes)
            .map_err(|_| persisted_fts5_result_error("doc_id must be non-NULL TEXT"))?,
        _ => return Err(persisted_fts5_result_error("doc_id must be non-NULL TEXT")),
    };
    if doc_id.is_empty() {
        return Err(persisted_fts5_result_error("doc_id must not be empty"));
    }

    let metadata = match row.get_ref(1) {
        Err(_) => {
            return Err(persisted_fts5_result_error(
                "metadata_json column is missing",
            ));
        }
        Ok(ValueRef::Null) => None,
        Ok(ValueRef::Text([])) => None,
        Ok(ValueRef::Text(text)) => {
            let text = std::str::from_utf8(text)
                .map_err(|_| persisted_fts5_result_error("metadata_json must be TEXT or NULL"))?;
            Some(
                serde_json::from_str(text).map_err(|error| SearchError::InvalidConfig {
                    field: "fts5.metadata_json".to_owned(),
                    value: error.to_string(),
                    reason: "persisted FTS5 metadata_json is not valid JSON".to_owned(),
                })?,
            )
        }
        Ok(value) => {
            return Err(persisted_fts5_result_error(&format!(
                "metadata_json must be TEXT or NULL, got {value:?}"
            )));
        }
    };

    let raw_score = match row.get_ref(2) {
        Ok(ValueRef::Real(score)) => score,
        Ok(ValueRef::Integer(score)) => score as f64,
        Ok(value) => {
            return Err(persisted_fts5_result_error(&format!(
                "bm25 score must be REAL or INTEGER, got {value:?}"
            )));
        }
        Err(_) => return Err(persisted_fts5_result_error("bm25 score column is missing")),
    };
    let score = -raw_score;
    if !score.is_finite() || score > f64::from(f32::MAX) {
        return Err(persisted_fts5_result_error(
            "bm25 score is non-finite or does not fit f32",
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let score = score as f32;

    Ok(ScoredResult {
        doc_id: doc_id.to_owned().into(),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata,
    })
}

fn parse_persisted_fts5_metadata(
    table_name: &str,
    create_sql: &str,
) -> SearchResult<PersistedFts5Metadata> {
    let mut cursor = Fts5DdlCursor::new(create_sql);
    cursor.expect_keyword("CREATE", table_name)?;
    cursor.expect_keyword("VIRTUAL", table_name)?;
    cursor.expect_keyword("TABLE", table_name)?;
    if cursor.consume_keyword("IF") {
        cursor.expect_keyword("NOT", table_name)?;
        cursor.expect_keyword("EXISTS", table_name)?;
    }
    let declared_table = cursor
        .identifier()
        .ok_or_else(|| persisted_fts5_metadata_error(table_name, "missing virtual table name"))?;
    if !declared_table.eq_ignore_ascii_case(table_name) {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "sqlite_master definition declares a different table name",
        ));
    }
    cursor.expect_keyword("USING", table_name)?;
    cursor.expect_keyword("FTS5", table_name)?;
    let arguments = cursor.parenthesized(table_name)?;
    cursor.finish(table_name)?;

    let mut content_mode = Fts5ContentMode::Stored;
    let mut saw_content = false;
    let mut tokenizer = None;
    for argument in split_fts5_arguments(arguments, table_name)? {
        let Some((key, value)) = split_fts5_option(argument, table_name)? else {
            continue;
        };
        if key.eq_ignore_ascii_case("content") {
            if saw_content {
                return Err(persisted_fts5_metadata_error(
                    table_name,
                    "duplicate content option",
                ));
            }
            saw_content = true;
            content_mode = if parse_fts5_option_value(value, table_name)?.is_empty() {
                Fts5ContentMode::Contentless
            } else {
                Fts5ContentMode::External
            };
        } else if key.eq_ignore_ascii_case("tokenize") {
            if tokenizer.is_some() {
                return Err(persisted_fts5_metadata_error(
                    table_name,
                    "duplicate tokenize option",
                ));
            }
            tokenizer = Some(parse_fts5_option_value(value, table_name)?);
        }
    }

    let tokenizer = tokenizer.ok_or_else(|| {
        persisted_fts5_metadata_error(table_name, "missing explicit tokenize option")
    })?;
    Ok(PersistedFts5Metadata {
        content_mode,
        tokenizer,
    })
}

fn split_fts5_arguments<'a>(arguments: &'a str, table_name: &str) -> SearchResult<Vec<&'a str>> {
    let mut items = Vec::new();
    let mut start = 0;
    let mut depth = 0_u32;
    let mut quote = None;
    let bytes = arguments.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(delimiter) = quote {
            if byte == delimiter {
                if index + 1 < bytes.len() && bytes[index + 1] == delimiter {
                    index += 2;
                    continue;
                }
                quote = None;
            }
        } else {
            match byte {
                b'\'' | b'\"' | b'`' => quote = Some(byte),
                b'[' => quote = Some(b']'),
                b'(' => depth = depth.saturating_add(1),
                b')' => {
                    if depth == 0 {
                        return Err(persisted_fts5_metadata_error(
                            table_name,
                            "unbalanced parenthesis in FTS5 arguments",
                        ));
                    }
                    depth -= 1;
                }
                b',' if depth == 0 => {
                    items.push(arguments[start..index].trim());
                    start = index + 1;
                }
                _ => {}
            }
        }
        index += 1;
    }
    if quote.is_some() || depth != 0 {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "unterminated quote or parenthesis in FTS5 arguments",
        ));
    }
    let final_item = arguments[start..].trim();
    if final_item.is_empty() {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "empty FTS5 argument",
        ));
    }
    items.push(final_item);
    Ok(items)
}

fn split_fts5_option<'a>(
    argument: &'a str,
    table_name: &str,
) -> SearchResult<Option<(&'a str, &'a str)>> {
    let mut quote = None;
    let mut depth = 0_u32;
    let bytes = argument.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(delimiter) = quote {
            if byte == delimiter {
                if index + 1 < bytes.len() && bytes[index + 1] == delimiter {
                    index += 2;
                    continue;
                }
                quote = None;
            }
        } else {
            match byte {
                b'\'' | b'\"' | b'`' => quote = Some(byte),
                b'[' => quote = Some(b']'),
                b'(' => depth = depth.saturating_add(1),
                b')' => {
                    if depth == 0 {
                        return Err(persisted_fts5_metadata_error(
                            table_name,
                            "unbalanced parenthesis in an FTS5 argument",
                        ));
                    }
                    depth -= 1;
                }
                b'=' if depth == 0 => {
                    let key = argument[..index].trim();
                    let value = argument[index + 1..].trim();
                    if key.is_empty() || value.is_empty() {
                        return Err(persisted_fts5_metadata_error(
                            table_name,
                            "FTS5 option has an empty key or value",
                        ));
                    }
                    return Ok(Some((key, value)));
                }
                _ => {}
            }
        }
        index += 1;
    }
    if quote.is_some() || depth != 0 {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "unterminated quote or parenthesis in an FTS5 argument",
        ));
    }
    Ok(None)
}

fn parse_fts5_option_value(value: &str, table_name: &str) -> SearchResult<String> {
    let value = value.trim();
    let Some(quote) = value
        .as_bytes()
        .first()
        .copied()
        .filter(|quote| matches!(quote, b'\'' | b'\"' | b'`'))
    else {
        return Ok(value.to_ascii_lowercase());
    };
    if value.len() < 2 {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "unterminated quoted FTS5 option value",
        ));
    }

    let mut decoded = String::new();
    let bytes = value.as_bytes();
    let mut index = 1;
    while index < bytes.len() {
        if bytes[index] == quote {
            if index + 1 < bytes.len() && bytes[index + 1] == quote {
                decoded.push(quote as char);
                index += 2;
                continue;
            }
            if !value[index + 1..].trim().is_empty() {
                return Err(persisted_fts5_metadata_error(
                    table_name,
                    "trailing text after quoted FTS5 option value",
                ));
            }
            return Ok(decoded.to_ascii_lowercase());
        }
        let Some(character) = value[index..].chars().next() else {
            break;
        };
        decoded.push(character);
        index += character.len_utf8();
    }
    Err(persisted_fts5_metadata_error(
        table_name,
        "unterminated quoted FTS5 option value",
    ))
}

struct Fts5DdlCursor<'a> {
    source: &'a str,
    index: usize,
}

impl<'a> Fts5DdlCursor<'a> {
    const fn new(source: &'a str) -> Self {
        Self { source, index: 0 }
    }

    fn skip_whitespace(&mut self) {
        while self
            .source
            .as_bytes()
            .get(self.index)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.index += 1;
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        self.skip_whitespace();
        let remaining = &self.source[self.index..];
        let Some(candidate) = remaining.get(..keyword.len()) else {
            return false;
        };
        if !candidate.eq_ignore_ascii_case(keyword) {
            return false;
        }
        if remaining
            .as_bytes()
            .get(keyword.len())
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            return false;
        }
        self.index += keyword.len();
        true
    }

    fn expect_keyword(&mut self, keyword: &str, table_name: &str) -> SearchResult<()> {
        if self.consume_keyword(keyword) {
            Ok(())
        } else {
            Err(persisted_fts5_metadata_error(
                table_name,
                &format!("expected {keyword} in CREATE VIRTUAL TABLE definition"),
            ))
        }
    }

    fn identifier(&mut self) -> Option<String> {
        self.skip_whitespace();
        let byte = *self.source.as_bytes().get(self.index)?;
        let closing = match byte {
            b'\"' => Some(b'\"'),
            b'`' => Some(b'`'),
            b'[' => Some(b']'),
            _ => None,
        };
        if let Some(closing) = closing {
            self.index += 1;
            let start = self.index;
            while let Some(current) = self.source.as_bytes().get(self.index).copied() {
                if current == closing {
                    let value = self.source[start..self.index].to_owned();
                    self.index += 1;
                    return Some(value);
                }
                self.index += 1;
            }
            return None;
        }

        if !(byte == b'_' || byte.is_ascii_alphabetic()) {
            return None;
        }
        let start = self.index;
        self.index += 1;
        while self
            .source
            .as_bytes()
            .get(self.index)
            .is_some_and(|current| *current == b'_' || current.is_ascii_alphanumeric())
        {
            self.index += 1;
        }
        Some(self.source[start..self.index].to_owned())
    }

    fn parenthesized(&mut self, table_name: &str) -> SearchResult<&'a str> {
        self.skip_whitespace();
        if self.source.as_bytes().get(self.index) != Some(&b'(') {
            return Err(persisted_fts5_metadata_error(
                table_name,
                "expected FTS5 argument list",
            ));
        }
        self.index += 1;
        let start = self.index;
        let mut depth = 0_u32;
        let mut quote = None;
        while let Some(byte) = self.source.as_bytes().get(self.index).copied() {
            if let Some(delimiter) = quote {
                if byte == delimiter {
                    if self.source.as_bytes().get(self.index + 1) == Some(&delimiter) {
                        self.index += 2;
                        continue;
                    }
                    quote = None;
                }
            } else {
                match byte {
                    b'\'' | b'\"' | b'`' => quote = Some(byte),
                    b'[' => quote = Some(b']'),
                    b'(' => depth = depth.saturating_add(1),
                    b')' if depth == 0 => {
                        let end = self.index;
                        self.index += 1;
                        return Ok(&self.source[start..end]);
                    }
                    b')' => depth -= 1,
                    _ => {}
                }
            }
            self.index += 1;
        }
        Err(persisted_fts5_metadata_error(
            table_name,
            "unterminated FTS5 argument list",
        ))
    }

    fn finish(&mut self, table_name: &str) -> SearchResult<()> {
        self.skip_whitespace();
        if self.source.as_bytes().get(self.index) == Some(&b';') {
            self.index += 1;
            self.skip_whitespace();
        }
        if self.index == self.source.len() {
            Ok(())
        } else {
            Err(persisted_fts5_metadata_error(
                table_name,
                "unexpected trailing text in CREATE VIRTUAL TABLE definition",
            ))
        }
    }
}

fn fts5_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

fn persisted_fts5_metadata_error(table_name: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "fts5.persisted_metadata".to_owned(),
        value: table_name.to_owned(),
        reason: reason.to_owned(),
    }
}

fn persisted_fts5_result_error(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "fts5.persisted_result".to_owned(),
        value: reason.to_owned(),
        reason: "persisted FTS5 table does not meet the frankensearch result contract".to_owned(),
    }
}

// ─── Split lexical trait implementations ────────────────────────────────────

#[allow(clippy::significant_drop_tightening)]
impl frankensearch_core::LexicalRead for Fts5LexicalSearch {
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            fts5_checkpoint(cx, "fts5.search")?;
            let query = Self::truncate_query(query);

            if query.trim().is_empty() {
                return Ok(Vec::new());
            }
            let limit = i64::try_from(limit).map_err(|_| SearchError::InvalidConfig {
                field: "fts5.limit".to_owned(),
                value: limit.to_string(),
                reason: "does not fit SQLite's signed integer limit".to_owned(),
            })?;

            let conn = self.conn.lock().map_err(lock_error)?;
            let rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            let sql = format!(
                "SELECT rowid, bm25({0}, 0.0, ?2, 1.0, 0.0), {1} \
                 FROM {0} WHERE {0} MATCH ?1 \
                 ORDER BY bm25({0}, 0.0, ?2, 1.0, 0.0), rowid LIMIT ?3;",
                MEM_FTS5_TABLE,
                self.metadata_select(),
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|error| SearchError::QueryParseError {
                    query: query.to_owned(),
                    detail: error.to_string(),
                })?;
            let rows = stmt
                .query_map(
                    params![query, self.config.title_boost, limit],
                    |row| -> rusqlite::Result<(i64, f64, Option<String>)> {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                    },
                )
                .map_err(|error| SearchError::QueryParseError {
                    query: query.to_owned(),
                    detail: error.to_string(),
                })?;

            let mut results = Vec::new();
            for row in rows {
                let (rowid, raw_score, metadata_json) =
                    row.map_err(|error| SearchError::QueryParseError {
                        query: query.to_owned(),
                        detail: error.to_string(),
                    })?;
                let doc_id = rowid_map.get_doc_id(rowid).unwrap_or("").to_owned();

                // FTS5 BM25 scores are negative (lower = better).
                // Negate to produce positive scores (higher = better).
                #[allow(clippy::cast_possible_truncation)]
                let bm25_score = (-raw_score) as f32;

                let metadata = metadata_json
                    .filter(|s| !s.is_empty())
                    .and_then(|s| serde_json::from_str(&s).ok());

                results.push(ScoredResult {
                    doc_id: doc_id.into(),
                    score: bm25_score,
                    source: ScoreSource::Lexical,
                    index: None,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some(bm25_score),
                    rerank_score: None,
                    explanation: None,
                    metadata,
                });
            }
            debug!(hits = results.len(), "fts5 BM25 search completed");
            Ok(results)
        })
    }

    /// FTS5 attaches full metadata during `search`, so the inherited eager
    /// `search_candidates` and its no-op hydration are exact for this backend:
    /// there is no deferred path to lose and no snapshot to pin.
    fn doc_count(&self) -> SearchResult<usize> {
        Ok(self.rowid_map.lock().map_err(lock_error)?.doc_to_row.len())
    }
}

impl frankensearch_core::LexicalWrite for Fts5LexicalSearch {
    fn index_document<'a>(
        &'a self,
        cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            fts5_checkpoint(cx, "fts5.index_document")?;
            let mut conn = self.conn.lock().map_err(lock_error)?;
            let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            // The upsert's DELETE+INSERT pair runs in one transaction and the
            // rowid map is staged on a clone, so a failed write rolls back
            // both the FTS5 row and the map mutation together.
            let tx = conn
                .transaction()
                .map_err(|error| map_storage_error_at("fts5 index begin", error))?;
            let mut staged_map = rowid_map.clone();
            Self::index_document_on(&tx, &mut staged_map, doc)?;
            tx.commit()
                .map_err(|error| map_storage_error_at("fts5 index commit", error))?;
            *rowid_map = staged_map;
            drop(rowid_map);
            drop(conn);

            Ok(())
        })
    }

    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            fts5_checkpoint(cx, "fts5.index_documents")?;
            let mut conn = self.conn.lock().map_err(lock_error)?;
            let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            let tx = conn
                .transaction()
                .map_err(|error| map_storage_error_at("fts5 index batch begin", error))?;
            // Stage rowid assignments on a clone: a mid-batch checkpoint or a
            // failed commit drops `tx` (rolling back every FTS5 write) and
            // discards `staged_map`, leaving `rowid_map` in sync with the
            // table. The clone is published only after `tx.commit()` succeeds.
            let mut staged_map = rowid_map.clone();
            for doc in docs {
                fts5_checkpoint(cx, "fts5.index_documents")?;
                Self::index_document_on(&tx, &mut staged_map, doc)?;
            }
            tx.commit()
                .map_err(|error| map_storage_error_at("fts5 index batch commit", error))?;
            *rowid_map = staged_map;
            drop(rowid_map);
            drop(conn);

            debug!(count = docs.len(), "fts5: batch indexed documents");
            Ok(())
        })
    }

    fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
        // FTS5 on a single connection has no separate commit phase.
        Box::pin(async { Ok(()) })
    }
}

// ─── Hit type for snippet-aware search ──────────────────────────────────────

/// A hit from FTS5 search with optional snippet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fts5Hit {
    /// Document identifier.
    pub doc_id: String,
    /// BM25 relevance score (higher = better).
    pub bm25_score: f32,
    /// Position in results (0-indexed).
    pub rank: usize,
    /// Highlighted content snippet around matching terms.
    pub snippet: Option<String>,
    /// Document metadata.
    pub metadata: Option<serde_json::Value>,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn lock_error<T>(_: T) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fts5",
        source: Box::new(std::io::Error::other("fts5 mutex poisoned")),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::*;
    // The split capabilities, so `adapter.search(..)` / `.index_document(..)`
    // / `.doc_count()` resolve now that the combined trait is gone.
    use frankensearch_core::{LexicalRead as _, LexicalWrite as _};

    /// Helper: run async test code with a `Cx` (asupersync, NO tokio).
    fn run_with_cx<F, Fut>(f: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        asupersync::test_utils::run_test_with_cx(f);
    }

    fn make_doc(id: &str, content: &str) -> IndexableDocument {
        IndexableDocument::new(id, content)
    }

    fn make_doc_with_title(id: &str, title: &str, content: &str) -> IndexableDocument {
        IndexableDocument::new(id, content).with_title(title)
    }

    fn make_doc_with_metadata(
        id: &str,
        content: &str,
        key: &str,
        value: &str,
    ) -> IndexableDocument {
        IndexableDocument::new(id, content).with_metadata(key, value)
    }

    // -- Construction --

    #[test]
    fn new_instance_is_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        assert_eq!(search.doc_count().expect("document count"), 0);
    }

    #[test]
    fn document_count_fails_closed_when_row_authority_is_poisoned() {
        let search = Arc::new(Fts5LexicalSearch::with_defaults());
        let poisoner = Arc::clone(&search);
        std::thread::spawn(move || {
            let _guard = poisoner.rowid_map.lock().expect("lock row authority");
            panic!("poison row authority for document-count regression");
        })
        .join()
        .expect_err("poisoning worker must panic");

        assert!(matches!(
            search.doc_count(),
            Err(SearchError::SubsystemError {
                subsystem: "fts5",
                ..
            })
        ));
    }

    #[test]
    fn config_defaults_are_sane() {
        let config = Fts5AdapterConfig::default();
        assert_eq!(config.content_mode, Fts5ContentMode::Stored);
        assert_eq!(config.tokenizer, Fts5TokenizerChoice::Unicode61);
        assert!((config.title_boost - TITLE_BOOST).abs() < f64::EPSILON);
    }

    #[test]
    fn persisted_metadata_uses_ddl_for_content_mode_and_porter() {
        let stored = parse_persisted_fts5_metadata(
            "docs",
            "CREATE VIRTUAL TABLE docs USING fts5(doc_id, metadata_json, tokenize='porter unicode61');",
        )
        .expect("stored Porter definition should parse");
        assert_eq!(stored.content_mode, Fts5ContentMode::Stored);
        ensure_rebuildable_porter_fts5("docs", &stored)
            .expect("stored Porter definition should be rebuildable");

        let external = parse_persisted_fts5_metadata(
            "docs",
            "CREATE VIRTUAL TABLE docs USING fts5(doc_id, metadata_json, content='documents', tokenize='porter');",
        )
        .expect("external Porter definition should parse");
        assert_eq!(external.content_mode, Fts5ContentMode::External);
        ensure_rebuildable_porter_fts5("docs", &external)
            .expect("external Porter definition should be rebuildable");
    }

    #[test]
    fn persisted_metadata_rejects_contentless_or_non_porter_tables() {
        let contentless = parse_persisted_fts5_metadata(
            "docs",
            "CREATE VIRTUAL TABLE docs USING fts5(doc_id, metadata_json, content='', tokenize='porter');",
        )
        .expect("contentless Porter definition should parse before its policy check");
        assert_eq!(contentless.content_mode, Fts5ContentMode::Contentless);
        assert!(ensure_rebuildable_porter_fts5("docs", &contentless).is_err());

        let unicode = parse_persisted_fts5_metadata(
            "docs",
            "CREATE VIRTUAL TABLE docs USING fts5(doc_id, metadata_json, tokenize='unicode61');",
        )
        .expect("non-Porter definition should still parse");
        assert!(ensure_rebuildable_porter_fts5("docs", &unicode).is_err());
    }

    #[test]
    fn persisted_count_requires_async_authority_and_revalidates_the_marker() {
        run_with_cx(|cx| async move {
            let storage = Arc::new(
                Storage::open_unbootstrapped_in_memory_for_test()
                    .expect("in-memory storage should open"),
            );
            {
                let conn = storage.connection().expect("connection should open");
                conn.execute_batch(&format!(
                    "CREATE TABLE {PORTER_FTS5_REBUILD_TABLE}(\
                     table_name TEXT, rebuild_version INTEGER NOT NULL);"
                ))
                .expect("governed rebuild marker table should be created");
                conn.execute_batch(
                    "CREATE VIRTUAL TABLE persisted_docs USING fts5(\
                     doc_id UNINDEXED, metadata_json UNINDEXED, content, tokenize='porter');",
                )
                .expect("persisted Porter table should be created");
                conn.execute_batch(
                    "INSERT INTO persisted_docs(doc_id, metadata_json, content) VALUES \
                     ('doc-1', '{}', 'first document'), \
                     ('doc-2', '{}', 'second document');",
                )
                .expect("persisted documents should be inserted");
                conn.execute_batch(&format!(
                    "INSERT INTO {PORTER_FTS5_REBUILD_TABLE}(table_name, rebuild_version) \
                     VALUES ('persisted_docs', {PORTER_FTS5_REBUILD_VERSION});"
                ))
                .expect("governed rebuild marker should be installed");
                drop(conn);
            }

            let reader =
                PersistedFts5LexicalSearch::open(&cx, Arc::clone(&storage), "persisted_docs")
                    .await
                    .expect("governed persisted reader should open");
            assert!(matches!(
                reader.doc_count(),
                Err(SearchError::SubsystemError {
                    subsystem: "fts5",
                    ..
                })
            ));
            assert_eq!(
                reader
                    .current_doc_count(&cx)
                    .await
                    .expect("async count should pin live persisted authority"),
                2
            );

            {
                let conn = storage.connection().expect("connection should open");
                conn.execute_batch(&format!(
                    "UPDATE {PORTER_FTS5_REBUILD_TABLE} SET rebuild_version = 0 \
                     WHERE table_name = 'persisted_docs';"
                ))
                .expect("marker invalidation should succeed");
                drop(conn);
            }
            assert!(matches!(
                reader.current_doc_count(&cx).await,
                Err(SearchError::InvalidConfig { field, .. })
                    if field == "fts5.rebuild_version"
            ));

            {
                let conn = storage.connection().expect("connection should open");
                conn.execute_batch(&format!(
                    "UPDATE {PORTER_FTS5_REBUILD_TABLE} \
                     SET rebuild_version = {PORTER_FTS5_REBUILD_VERSION} \
                     WHERE table_name = 'persisted_docs';"
                ))
                .expect("governed marker should be restored");
                conn.execute_batch(&format!(
                    "INSERT INTO {PORTER_FTS5_REBUILD_TABLE}(table_name, rebuild_version) \
                     VALUES ('persisted_docs', {PORTER_FTS5_REBUILD_VERSION});"
                ))
                .expect("duplicate hostile markers should be inserted");
                drop(conn);
            }
            assert!(matches!(
                reader.current_doc_count(&cx).await,
                Err(SearchError::InvalidConfig { field, reason, .. })
                    if field == "fts5.rebuild_version" && reason.contains("duplicate")
            ));
        });
    }

    #[test]
    fn rebuild_porter_table_then_persisted_search_finds_stemmed_matches() {
        run_with_cx(|cx| async move {
            let storage = Arc::new(Storage::open_in_memory().expect("storage should open"));
            {
                let conn = storage.connection().expect("connection should open");
                conn.execute_batch(
                    "CREATE VIRTUAL TABLE porter_docs USING fts5(\
                     doc_id UNINDEXED, metadata_json UNINDEXED, content, tokenize='porter');",
                )
                .expect("Porter table should be created");
                conn.execute_batch(
                    "INSERT INTO porter_docs(doc_id, metadata_json, content) VALUES \
                     ('doc-1', '{}', 'running fast'), \
                     ('doc-2', '{}', 'walking slow');",
                )
                .expect("documents should be inserted");
            }

            // Without a committed rebuild marker the governed reader refuses
            // to serve potentially stale Porter results.
            assert!(
                PersistedFts5LexicalSearch::open(&cx, Arc::clone(&storage), "porter_docs")
                    .await
                    .is_err()
            );

            rebuild_porter_fts5_table(&cx, &storage, "porter_docs")
                .await
                .expect("rebuild should succeed");
            // The committed marker makes a second rebuild an idempotent no-op.
            rebuild_porter_fts5_table(&cx, &storage, "porter_docs")
                .await
                .expect("rebuild should be idempotent");

            let reader = PersistedFts5LexicalSearch::open(&cx, Arc::clone(&storage), "porter_docs")
                .await
                .expect("governed persisted reader should open");
            let hits = reader
                .search(&cx, "run", 10)
                .await
                .expect("persisted search should succeed");
            assert_eq!(hits.len(), 1, "Porter should stem 'running' to 'run'");
            assert_eq!(hits[0].doc_id, "doc-1");
            assert!(hits[0].score > 0.0);
            assert_eq!(reader.current_doc_count(&cx).await.expect("count"), 2);
        });
    }

    // -- Indexing --

    #[test]
    fn index_single_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc("doc1", "hello world of search");
            search.index_document(&cx, &doc).await.unwrap();
            assert_eq!(search.doc_count().expect("document count"), 1);
        });
    }

    #[test]
    fn index_batch_documents() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let docs = vec![
                make_doc("a", "first document"),
                make_doc("b", "second document"),
                make_doc("c", "third document"),
            ];
            search.index_documents(&cx, &docs).await.unwrap();
            assert_eq!(search.doc_count().expect("document count"), 3);
        });
    }

    #[test]
    fn upsert_replaces_existing_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc_v1 = make_doc("doc1", "original content");
            search.index_document(&cx, &doc_v1).await.unwrap();
            assert_eq!(search.doc_count().expect("document count"), 1);

            let doc_v2 = make_doc("doc1", "updated content completely different");
            search.index_document(&cx, &doc_v2).await.unwrap();
            assert_eq!(search.doc_count().expect("document count"), 1);

            // Search should find updated content.
            let results = search.search(&cx, "updated", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");

            // Old content should not match.
            let results = search.search(&cx, "original", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    // -- Search --

    #[test]
    fn search_finds_matching_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "rust programming language"))
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "python programming language"))
                .await
                .unwrap();

            let results = search.search(&cx, "rust", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");
            assert_eq!(results[0].source, ScoreSource::Lexical);
            assert!(results[0].lexical_score.is_some());
            assert!(results[0].score > 0.0);
        });
    }

    #[test]
    fn search_returns_results_sorted_by_relevance() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            // doc1 mentions "search" more times -> higher BM25.
            search
                .index_document(
                    &cx,
                    &make_doc("doc1", "search search search algorithms for search"),
                )
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "search algorithms"))
                .await
                .unwrap();

            let results = search.search(&cx, "search", 10).await.unwrap();
            assert_eq!(results.len(), 2);
            // Higher TF should produce higher BM25 score.
            assert!(results[0].score >= results[1].score);
        });
    }

    #[test]
    fn search_empty_query_returns_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello world"))
                .await
                .unwrap();

            let results = search.search(&cx, "", 10).await.unwrap();
            assert!(results.is_empty());

            let results = search.search(&cx, "   ", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn search_no_match_returns_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello world"))
                .await
                .unwrap();

            let results = search.search(&cx, "zzzznonexistent", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn search_respects_limit() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            for i in 0..10 {
                search
                    .index_document(
                        &cx,
                        &make_doc(&format!("doc{i}"), "common term in all docs"),
                    )
                    .await
                    .unwrap();
            }

            let results = search.search(&cx, "common", 3).await.unwrap();
            assert_eq!(results.len(), 3);
        });
    }

    #[test]
    fn search_preserves_metadata() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc_with_metadata("doc1", "searchable content", "category", "test");
            search.index_document(&cx, &doc).await.unwrap();

            let results = search.search(&cx, "searchable", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            let meta = results[0].metadata.as_ref().unwrap();
            assert_eq!(meta["category"], "test");
        });
    }

    #[test]
    fn search_with_title_and_content() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc_with_title("doc1", "Important Title", "body text here");
            search.index_document(&cx, &doc).await.unwrap();

            // Should match on title.
            let results = search.search(&cx, "important", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");

            // Should match on content.
            let results = search.search(&cx, "body", 10).await.unwrap();
            assert_eq!(results.len(), 1);
        });
    }

    // -- Snippets --

    #[test]
    #[allow(clippy::significant_drop_tightening)]
    fn search_with_snippets_returns_highlighted_text() {
        let search = Fts5LexicalSearch::with_defaults();

        {
            let conn = search.conn.lock().unwrap();
            let mut rowid_map = search.rowid_map.lock().unwrap();

            let doc = make_doc("doc1", "The quick brown fox jumps over the lazy dog");
            let rowid = rowid_map.get_or_assign(&doc.id);
            let columns = Fts5LexicalSearch::doc_to_columns(&doc);
            conn.execute(
                &format!(
                    "INSERT INTO {MEM_FTS5_TABLE}(rowid, doc_id, title, content, metadata_json) \
                     VALUES (?1, ?2, ?3, ?4, ?5);"
                ),
                params![rowid, columns[0], columns[1], columns[2], columns[3]],
            )
            .expect("document should be indexed");
        }

        let hits = search.search_with_snippets("fox", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc1");
        assert!(hits[0].snippet.is_some());
        let snippet = hits[0].snippet.as_ref().unwrap();
        assert!(
            snippet.contains("<b>fox</b>"),
            "snippet should highlight match: {snippet}"
        );
    }

    // -- Delete --

    #[test]
    fn delete_document_removes_it() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "findable content"))
                .await
                .unwrap();
            assert_eq!(search.doc_count().expect("document count"), 1);

            let removed = search.delete_document("doc1").unwrap();
            assert!(removed);
            assert_eq!(search.doc_count().expect("document count"), 0);

            let results = search.search(&cx, "findable", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let search = Fts5LexicalSearch::with_defaults();
        let removed = search.delete_document("nonexistent").unwrap();
        assert!(!removed);
    }

    // -- Clear --

    #[test]
    fn clear_removes_all_documents() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello"))
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "world"))
                .await
                .unwrap();
            assert_eq!(search.doc_count().expect("document count"), 2);

            search.clear().unwrap();
            assert_eq!(search.doc_count().expect("document count"), 0);

            let results = search.search(&cx, "hello", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    // -- Commit is no-op --

    #[test]
    fn commit_succeeds_without_error() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search.commit(&cx).await.unwrap();
        });
    }

    // -- Edge cases --

    #[test]
    fn document_with_empty_content() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc("doc1", "");
            search.index_document(&cx, &doc).await.unwrap();
            assert_eq!(search.doc_count().expect("document count"), 1);

            let results = search.search(&cx, "anything", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn document_with_special_characters() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc(
                "doc1",
                "error: fn<T>(x: &mut Vec<u8>) -> Result<(), Box<dyn Error>>",
            );
            search.index_document(&cx, &doc).await.unwrap();

            let results = search.search(&cx, "error", 10).await.unwrap();
            assert_eq!(results.len(), 1);
        });
    }

    #[test]
    fn batch_upsert_mixed_new_and_existing() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "original"))
                .await
                .unwrap();
            assert_eq!(search.doc_count().expect("document count"), 1);

            let batch = vec![
                make_doc("doc1", "updated"),   // existing
                make_doc("doc2", "brand new"), // new
            ];
            search.index_documents(&cx, &batch).await.unwrap();
            assert_eq!(search.doc_count().expect("document count"), 2);
        });
    }

    // -- Transaction safety --

    #[test]
    fn cancelled_index_documents_leaves_index_unchanged() {
        run_with_cx(|cx| async move {
            let search = Fts5LexicalSearch::with_defaults();
            search
                .index_document(&cx, &make_doc("keep", "stable seed content"))
                .await
                .unwrap();
            assert_eq!(search.doc_count().expect("document count"), 1);

            cx.cancel_with(
                asupersync::CancelKind::User,
                Some("fts5 batch cancel probe"),
            );
            let docs = vec![
                make_doc("batch-a", "cancelled alpha"),
                make_doc("batch-b", "cancelled bravo"),
            ];
            let error = search
                .index_documents(&cx, &docs)
                .await
                .expect_err("cancelled batch must not commit");
            assert!(matches!(error, SearchError::Cancelled { .. }));

            // The cancelled cx fails every later checkpoint, so verify with a
            // fresh context.
            let verify_cx = Cx::for_testing();
            assert_eq!(
                search.doc_count().expect("document count"),
                1,
                "cancelled batch must not change doc_count"
            );
            let hits = search.search(&verify_cx, "stable", 10).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "keep");
            let hits = search.search(&verify_cx, "cancelled", 10).await.unwrap();
            assert!(
                hits.is_empty(),
                "cancelled batch documents must not become searchable"
            );
        });
    }

    #[test]
    #[allow(clippy::significant_drop_tightening)]
    fn failed_index_documents_rolls_back_rowid_map() {
        run_with_cx(|cx| async move {
            let search = Fts5LexicalSearch::with_defaults();
            search
                .index_document(&cx, &make_doc("seed", "stable seed content"))
                .await
                .unwrap();
            assert_eq!(search.doc_count().expect("document count"), 1);

            // Occupy the rowid the batch's second document will be assigned,
            // without registering it in `rowid_map`. The batch then fails
            // mid-transaction on a rowid constraint after the first document
            // was already inserted.
            {
                let conn = search.conn.lock().expect("lock connection");
                conn.execute(
                    &format!(
                        "INSERT INTO {MEM_FTS5_TABLE}(rowid, doc_id, title, content, metadata_json) \
                         VALUES (?1, ?2, ?3, ?4, ?5);"
                    ),
                    params![3_i64, "ghost", "", "", ""],
                )
                .expect("ghost row should insert");
            }

            let docs = vec![
                make_doc("batch-a", "alpha content"),
                make_doc("batch-b", "bravo content"),
            ];
            let error = search
                .index_documents(&cx, &docs)
                .await
                .expect_err("mid-batch rowid conflict must fail the batch");
            assert!(!matches!(error, SearchError::Cancelled { .. }));

            assert_eq!(
                search.doc_count().expect("document count"),
                1,
                "rolled-back batch must not change doc_count"
            );
            let hits = search.search(&cx, "alpha", 10).await.unwrap();
            assert!(
                hits.is_empty(),
                "rolled-back document must not be searchable"
            );
            let hits = search.search(&cx, "stable", 10).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "seed");
        });
    }

    // -- Content modes and tokenizers --

    #[test]
    fn contentless_mode_indexes_upserts_searches_and_deletes() {
        let config = Fts5AdapterConfig {
            content_mode: Fts5ContentMode::Contentless,
            ..Fts5AdapterConfig::default()
        };
        let search = Fts5LexicalSearch::new(config);
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "contentless alpha"))
                .await
                .unwrap();
            // The upsert exercises the contentless_delete rowid path.
            search
                .index_document(&cx, &make_doc("doc1", "contentless beta"))
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "contentless gamma"))
                .await
                .unwrap();
            assert_eq!(search.doc_count().expect("document count"), 2);

            let hits = search.search(&cx, "beta", 10).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "doc1");
            assert!(hits[0].metadata.is_none());

            // Snippets are NULL placeholders on a contentless table.
            let hits = search.search_with_snippets("beta", 10).unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "doc1");
            assert!(hits[0].snippet.is_none());

            assert!(search.delete_document("doc1").unwrap());
            assert_eq!(search.doc_count().expect("document count"), 1);
            search.clear().unwrap();
            assert_eq!(search.doc_count().expect("document count"), 0);
        });
    }

    #[test]
    fn porter_tokenizer_stems_terms() {
        let search = Fts5LexicalSearch::new(Fts5AdapterConfig {
            tokenizer: Fts5TokenizerChoice::Porter,
            ..Fts5AdapterConfig::default()
        });
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "running quickly"))
                .await
                .unwrap();

            // Porter stems 'running' to 'run', so the stemmed query matches.
            let hits = search.search(&cx, "run", 10).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "doc1");
        });
    }

    #[test]
    fn trigram_tokenizer_matches_substrings() {
        let search = Fts5LexicalSearch::new(Fts5AdapterConfig {
            tokenizer: Fts5TokenizerChoice::Trigram,
            ..Fts5AdapterConfig::default()
        });
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello world"))
                .await
                .unwrap();

            // Trigram indexes byte substrings, so partial words match.
            let hits = search.search(&cx, "ello wor", 10).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "doc1");
        });
    }

    // -- Trait object safety --

    #[test]
    fn fts5_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fts5LexicalSearch>();
    }

    // -- Config serialization --

    #[test]
    fn config_serde_roundtrip() {
        let config = Fts5AdapterConfig {
            content_mode: Fts5ContentMode::Contentless,
            tokenizer: Fts5TokenizerChoice::Porter,
            title_boost: 3.0,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Fts5AdapterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content_mode, Fts5ContentMode::Contentless);
        assert_eq!(deserialized.tokenizer, Fts5TokenizerChoice::Porter);
        assert!((deserialized.title_boost - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn content_mode_default_is_stored() {
        assert_eq!(Fts5ContentMode::default(), Fts5ContentMode::Stored);
    }

    #[test]
    fn tokenizer_default_is_unicode61() {
        assert_eq!(
            Fts5TokenizerChoice::default(),
            Fts5TokenizerChoice::Unicode61
        );
    }

    // -- Query truncation --

    #[test]
    fn long_query_is_truncated() {
        let long_query = "a".repeat(MAX_QUERY_LENGTH + 100);
        let truncated = Fts5LexicalSearch::truncate_query(&long_query);
        assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
    }

    #[test]
    fn multibyte_query_uses_character_limit() {
        let long_query = "\u{00E9}".repeat(MAX_QUERY_LENGTH + 3);
        let truncated = Fts5LexicalSearch::truncate_query(&long_query);
        assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
        assert_eq!(truncated.len(), MAX_QUERY_LENGTH * '\u{00E9}'.len_utf8());
    }

    #[test]
    fn multibyte_query_within_character_limit_is_unchanged() {
        let query = "\u{00E9}".repeat(MAX_QUERY_LENGTH / 2 + 1);
        assert!(query.len() > MAX_QUERY_LENGTH);
        assert_eq!(Fts5LexicalSearch::truncate_query(&query), query);
    }

    #[test]
    fn normal_query_is_not_truncated() {
        let query = "normal search query";
        let result = Fts5LexicalSearch::truncate_query(query);
        assert_eq!(result, query);
    }

    // -- Debug impl --

    #[test]
    fn debug_format_never_waits_for_row_authority() {
        let search = Fts5LexicalSearch::with_defaults();
        let debug = format!("{search:?}");
        assert!(debug.contains("Fts5LexicalSearch"));
        assert!(debug.contains("config"));
        assert!(!debug.contains("doc_count"));
    }

    // -- Fts5Hit serde --

    #[test]
    fn fts5_hit_serde_roundtrip() {
        let hit = Fts5Hit {
            doc_id: "doc1".into(),
            bm25_score: 1.5,
            rank: 0,
            snippet: Some("hello <b>world</b>".to_owned()),
            metadata: None,
        };
        let json = serde_json::to_string(&hit).unwrap();
        let deserialized: Fts5Hit = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.doc_id, "doc1");
        assert!((deserialized.bm25_score - 1.5).abs() < f32::EPSILON);
    }

    // -- cass#301 merge/finalize scaling probe --

    /// Generate a deterministic, lexically varied document body of roughly
    /// `target_bytes`. Mixing a finite vocabulary with the doc index keeps the
    /// posting lists realistic (many shared terms, a few doc-unique terms)
    /// without any RNG, so the probe is reproducible across runs and hosts.
    fn synthetic_body(doc_index: usize, target_bytes: usize) -> String {
        const VOCAB: &[&str] = &[
            "retry",
            "backoff",
            "structured",
            "concurrency",
            "channel",
            "reserve",
            "commit",
            "cancel",
            "region",
            "scope",
            "embedder",
            "lexical",
            "semantic",
            "fusion",
            "rank",
            "vector",
            "index",
            "segment",
            "merge",
            "finalize",
            "tokenizer",
            "posting",
            "document",
            "search",
            "query",
            "score",
            "bm25",
        ];
        use std::fmt::Write as _;
        let mut body = String::with_capacity(target_bytes + 32);
        let mut counter = doc_index;
        while body.len() < target_bytes {
            let word = VOCAB[counter % VOCAB.len()];
            body.push_str(word);
            // Sprinkle a doc-unique token so each document has distinct terms.
            if counter % 11 == 0 {
                let _ = write!(body, " d{doc_index}t{counter}");
            }
            body.push(' ');
            counter = counter.wrapping_add(1).wrapping_add(doc_index);
        }
        body
    }

    /// cass#301: scaling probe for the frankensearch FTS5 lexical-index build
    /// path (`Fts5LexicalSearch::index_documents`, the exact path cass drives
    /// during `cass index --full`).
    ///
    /// Feeds an increasing number of documents simulating ~10MB -> ~40MB of
    /// indexed content and prints the wall-time of the index-build (finalize)
    /// phase plus a representative search at each size. The reported
    /// `build_ms_per_mb` (build time normalised by content size) is the
    /// diagnostic: if it is roughly flat across sizes the build is linear; if
    /// it climbs ~linearly with content size the build is O(N^2).
    ///
    /// Run with:
    /// `cargo test -p frankensearch-storage --features fts5 --release \
    ///    fts5_index_build_scaling_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "cass#301 scaling probe: run explicitly with --ignored --nocapture"]
    fn fts5_index_build_scaling_probe() {
        use std::time::Instant;

        // ~50 KB per document. Content megabytes => doc_count = mb * 20.
        const DOC_BYTES: usize = 50 * 1024;
        let content_mbs: Vec<usize> = std::env::var("FTS5_PROBE_MBS")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            })
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| vec![10, 20, 30, 40]);

        run_with_cx(|cx| async move {
            eprintln!(
                "FTS5_PROBE doc_bytes={DOC_BYTES} sizes_mb={content_mbs:?} (cass#301 build/finalize scaling)"
            );
            for &mb in &content_mbs {
                let doc_count = mb * (1024 * 1024) / DOC_BYTES;
                let docs: Vec<IndexableDocument> = (0..doc_count)
                    .map(|i| {
                        IndexableDocument::new(format!("doc-{i}"), synthetic_body(i, DOC_BYTES))
                            .with_title(format!("Document {i}"))
                    })
                    .collect();

                let search = Fts5LexicalSearch::with_defaults();

                // Index-build / finalize phase — the wedge phase in cass#301.
                let build_started = Instant::now();
                search
                    .index_documents(&cx, &docs)
                    .await
                    .expect("index_documents");
                search.commit(&cx).await.expect("commit");
                let build_elapsed = build_started.elapsed();

                // Representative query against the freshly built index.
                let search_started = Instant::now();
                let hits = search
                    .search(&cx, "structured concurrency retry", 10)
                    .await
                    .expect("search");
                let search_elapsed = search_started.elapsed();

                let build_ms = build_elapsed.as_secs_f64() * 1_000.0;
                let search_ms = search_elapsed.as_secs_f64() * 1_000.0;
                eprintln!(
                    "FTS5_PROBE content_mb={mb} docs={doc_count} build_ms={build_ms:.1} \
                     build_ms_per_mb={:.2} search_ms={search_ms:.3} hits={} doc_count={}",
                    build_ms / mb as f64,
                    hits.len(),
                    search.doc_count().expect("document count"),
                );
                assert_eq!(search.doc_count().expect("document count"), doc_count);
            }
        });
    }
}
