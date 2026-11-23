use std::path::{Path, PathBuf};

use frankensearch_core::error::{SearchError, SearchResult};
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions,
};
use tantivy::{Index, IndexReader, ReloadPolicy};
use tracing::warn;

use super::tantivy_err;
use super::tokenizer::cass_ensure_tokenizer;

/// Schema version namespace used for cass-compatible Tantivy indexes.
pub const CASS_SCHEMA_VERSION: &str = "v8";
/// Content hash used to detect schema/tokenizer changes that require rebuild.
pub const CASS_SCHEMA_HASH: &str =
    "tantivy-schema-v8-hyphen-cjk-bigrams-bounded-content-prefix-preview-stored-content-external";

/// Returns true if the given stored hash matches the current schema hash.
#[must_use]
pub fn cass_schema_hash_matches(stored: &str) -> bool {
    stored == CASS_SCHEMA_HASH
}

/// Named fields used by cass-compatible query and indexing code.
#[derive(Clone, Copy, Debug)]
pub struct CassFields {
    pub agent: Field,
    pub workspace: Field,
    pub workspace_original: Field,
    pub source_path: Field,
    pub msg_idx: Field,
    pub created_at: Field,
    pub title: Field,
    pub content: Field,
    pub title_prefix: Field,
    pub content_prefix: Field,
    pub preview: Field,
    pub source_id: Field,
    pub origin_kind: Field,
    pub origin_host: Field,
    pub conversation_id: Option<Field>,
}

/// Canonical CASS document identity: `"{source_id}#{msg_idx}"`.
///
/// The CASS schema deliberately has no dedicated id field — a document is a
/// message inside a source, so identity is the pair. The synthetic conformance
/// corpus repeats `source_id` on purpose (several messages share one session),
/// which is exactly why `source_id` alone cannot serve as the key and `msg_idx`
/// supplies per-message uniqueness.
///
/// This exact rendering is part of the CASS profile contract
/// (`document_identity=source_id#msg_idx` in the schema contract preimage) and
/// must match on both the subject and the oracle side. It is not an adapter
/// convenience: the gauntlet comparator keys every divergence signature on this
/// string, so changing the format invalidates the register.
///
/// Missing stored fields degrade to an empty `source_id` and `msg_idx` 0 rather
/// than erroring, so a malformed document surfaces as a *comparable* identity
/// mismatch in the differential report instead of aborting the campaign.
#[cfg(feature = "tantivy-oracle")]
#[must_use]
pub fn cass_document_identity(document: &tantivy::TantivyDocument, fields: &CassFields) -> String {
    use tantivy::schema::Value as _;

    let source_id = document
        .get_first(fields.source_id)
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let msg_idx = document
        .get_first(fields.msg_idx)
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    cass_document_identity_parts(source_id, msg_idx)
}

/// Render the canonical CASS document identity from its two components.
///
/// This is the single source of truth for the `"{source_id}#{msg_idx}"`
/// rendering named by the CASS profile contract. It lives in this crate rather
/// than in the gauntlet because the dependency runs gauntlet → lexical: the
/// corpus generator and both engine adapters must agree byte-for-byte, and the
/// only way to guarantee that is for all of them to call one function.
///
/// Prefer [`cass_document_identity`] when you already hold a stored document;
/// use this form when you are constructing the corpus and hold the parts.
#[cfg(feature = "tantivy-oracle")]
#[must_use]
pub fn cass_document_identity_parts(source_id: &str, msg_idx: u64) -> String {
    format!("{source_id}#{msg_idx}")
}

/// Cass-specific lexical document shape for index ingestion.
#[derive(Debug, Clone)]
pub struct CassDocument {
    pub agent: String,
    pub workspace: Option<String>,
    pub workspace_original: Option<String>,
    pub source_path: String,
    pub msg_idx: u64,
    pub created_at: Option<i64>,
    pub title: Option<String>,
    pub content: String,
    pub source_id: String,
    pub origin_kind: String,
    pub origin_host: Option<String>,
    pub conversation_id: Option<i64>,
}

#[derive(Debug, Clone, Copy)]
pub struct CassDocumentRef<'a> {
    pub agent: &'a str,
    pub workspace: Option<&'a str>,
    pub workspace_original: Option<&'a str>,
    pub source_path: &'a str,
    pub msg_idx: u64,
    pub created_at: Option<i64>,
    pub title: Option<&'a str>,
    pub content: &'a str,
    pub source_id: &'a str,
    pub origin_kind: &'a str,
    pub origin_host: Option<&'a str>,
    pub conversation_id: Option<i64>,
}

/// Build cass-compatible Tantivy schema.
#[must_use]
pub fn cass_build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    let indexed_text = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("hyphen_normalize")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let stored_indexed_text = indexed_text.clone().set_stored();
    let prefix_text = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("prefix_normalize")
            .set_index_option(IndexRecordOption::Basic),
    );

    schema_builder.add_text_field("agent", STRING | STORED);
    schema_builder.add_text_field("workspace", STRING | STORED);
    schema_builder.add_text_field("workspace_original", STORED);
    schema_builder.add_text_field("source_path", STORED);
    schema_builder.add_u64_field("msg_idx", INDEXED | STORED);
    schema_builder.add_i64_field("created_at", INDEXED | STORED | FAST);
    schema_builder.add_text_field("title", stored_indexed_text);
    schema_builder.add_text_field("content", indexed_text);
    schema_builder.add_text_field("title_prefix", prefix_text.clone());
    schema_builder.add_text_field("content_prefix", prefix_text);
    schema_builder.add_text_field("preview", STORED);
    schema_builder.add_text_field("source_id", STRING | STORED);
    schema_builder.add_text_field("origin_kind", STRING | STORED);
    schema_builder.add_text_field("origin_host", STRING | STORED);
    schema_builder.add_i64_field("conversation_id", STORED);
    schema_builder.build()
}

/// Extract cass-compatible schema fields from a Tantivy schema handle.
///
/// # Errors
///
/// Returns [`SearchError::InvalidConfig`] when required fields are missing.
pub fn cass_fields_from_schema(schema: &Schema) -> SearchResult<CassFields> {
    let get = |name: &str| {
        schema
            .get_field(name)
            .map_err(|_| SearchError::InvalidConfig {
                field: "schema".to_string(),
                value: name.to_string(),
                reason: format!("schema missing required field `{name}`"),
            })
    };

    Ok(CassFields {
        agent: get("agent")?,
        workspace: get("workspace")?,
        workspace_original: get("workspace_original")?,
        source_path: get("source_path")?,
        msg_idx: get("msg_idx")?,
        created_at: get("created_at")?,
        title: get("title")?,
        content: get("content")?,
        title_prefix: get("title_prefix")?,
        content_prefix: get("content_prefix")?,
        preview: get("preview")?,
        source_id: get("source_id")?,
        origin_kind: get("origin_kind")?,
        origin_host: get("origin_host")?,
        conversation_id: schema.get_field("conversation_id").ok(),
    })
}

/// Open a cass-compatible search reader (no writer) with caller-specified reload policy.
///
/// This centralizes tokenizer registration + field extraction, mirroring the expectations of
/// cass query execution code while keeping Tantivy lifecycle ownership inside frankensearch.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] if Tantivy open/build operations fail, or
/// [`SearchError::InvalidConfig`] if schema fields are incomplete.
pub fn cass_open_search_reader(
    index_path: &Path,
    reload_policy: ReloadPolicy,
) -> SearchResult<(IndexReader, CassFields)> {
    let mut index = Index::open_in_dir(index_path).map_err(tantivy_err)?;
    cass_ensure_tokenizer(&mut index);
    let schema = index.schema();
    let fields = cass_fields_from_schema(&schema)?;
    let reader = index
        .reader_builder()
        .reload_policy(reload_policy)
        .try_into()
        .map_err(tantivy_err)?;
    if let Err(e) = reader.reload() {
        warn!(error = %e, "index reader reload failed — searches may serve stale results");
    }
    Ok((reader, fields))
}

/// Resolve cass-compatible index directory under a data root.
///
/// # Errors
///
/// Returns [`SearchError::SubsystemError`] if the directory cannot be created.
pub fn cass_index_dir(base: &Path) -> SearchResult<PathBuf> {
    let dir = base.join("index").join(CASS_SCHEMA_VERSION);
    std::fs::create_dir_all(&dir).map_err(tantivy_err)?;
    Ok(dir)
}
