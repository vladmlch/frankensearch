//! Native CASS profile surface for Quill-backed indexes.
//!
//! The CASS lexical profile historically lived only in the Tantivy backend
//! (`frankensearch-lexical::cass_compat`). Quill already owns the parts that
//! actually needed an engine — [`crate::schema::CASS_SEMANTIC_SCHEMA`], the
//! [`crate::schema::Analyzer::CassHyphenNormalize`] and
//! [`crate::schema::Analyzer::CassPrefixNormalize`] pipelines, and
//! [`crate::query::CassQueryParser`] — so this module carries only the
//! remaining engine-independent pieces: the ingest document shape, the
//! derived-column rules, and the on-disk namespacing that lets an index
//! declare which schema generation produced it.
//!
//! # Why the schema sentinel changes
//!
//! A Quill index is FSLX; it cannot read Tantivy segment files at all. Any
//! consumer moving from the Tantivy CASS backend to this one must rebuild,
//! and [`CASS_SCHEMA_HASH`] is exactly the mechanism that makes that rebuild
//! automatic rather than a corrupt read: a consumer stores the sentinel beside
//! its index and calls [`cass_schema_hash_matches`] on open. A changed
//! sentinel means "discard and reindex", which is the correct and only safe
//! answer here.

use std::path::{Path, PathBuf};

use crate::index::SchemaDocument;
use crate::scribe::cass_generate_edge_ngrams;

#[cfg(test)]
mod tests;

/// Schema generation namespace for Quill-backed CASS indexes.
///
/// This is deliberately not `v8`: `v8` names the Tantivy-era on-disk
/// generation, and reusing it would invite a consumer to open FSLX segments
/// with a Tantivy reader or the reverse.
pub const CASS_SCHEMA_VERSION: &str = "v9-quill";

/// Rebuild sentinel recorded beside a CASS index.
///
/// It is a descriptive marker rather than a computed digest, matching the
/// incumbent contract: consumers compare it verbatim and rebuild on any
/// difference. It names the engine because the engine determines the on-disk
/// format, and the analyzer family because a tokenizer change silently
/// invalidates every posting.
pub const CASS_SCHEMA_HASH: &str = "quill-fslx-schema-v9-hyphen-cjk-bigrams-bounded-content-prefix-preview-stored-content-external";

/// Longest preview retained in the stored `preview` column, in characters.
pub const PREVIEW_MAX_CHARS: usize = 400;

/// Largest content prefix fed to the edge-ngram column, in bytes.
pub const CONTENT_PREFIX_MAX_BYTES: usize = 4 * 1024;

/// Whether a stored sentinel names this exact schema generation.
///
/// Any difference — older generation, Tantivy-era generation, or corruption —
/// answers `false`, which callers treat as "rebuild".
#[must_use]
pub fn cass_schema_hash_matches(stored: &str) -> bool {
    stored == CASS_SCHEMA_HASH
}

/// Resolve (and create) the schema-qualified index directory beneath `base`.
///
/// Namespacing by schema generation is what lets a rebuild land beside the old
/// index instead of on top of it, so a half-finished migration never leaves a
/// consumer pointing at mixed-generation segments.
///
/// # Errors
///
/// Returns the underlying I/O error when the directory cannot be created.
pub fn cass_index_dir(base: &Path) -> std::io::Result<PathBuf> {
    let dir = base.join("index").join(CASS_SCHEMA_VERSION);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Segment-count threshold at which a CASS index wants compaction.
pub const CASS_MERGE_SEGMENT_THRESHOLD: usize = 4;

/// Minimum interval between CASS compactions, in milliseconds.
pub const CASS_MERGE_COOLDOWN_MS: i64 = 300_000;

/// Whether a CASS index currently wants compaction, and why.
///
/// Mirrors the incumbent's shape and thresholds so a consumer's merge policy
/// does not silently change meaning across the engine swap. `last_merge_ts` and
/// `ms_since_last_merge` are supplied by the caller because Quill does not own
/// a global merge clock: a negative `ms_since_last_merge` means "never merged",
/// which the cooldown treats as elapsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassMergeStatus {
    /// Live segment count in the published snapshot.
    pub segment_count: usize,
    /// Epoch milliseconds of the last compaction, or 0 if never.
    pub last_merge_ts: i64,
    /// Milliseconds since the last compaction, negative if never.
    pub ms_since_last_merge: i64,
    /// Segment count at or above which compaction is wanted.
    pub merge_threshold: usize,
    /// Minimum interval between compactions.
    pub cooldown_ms: i64,
}

impl CassMergeStatus {
    /// Whether the segment count has reached the threshold and the cooldown has
    /// elapsed.
    #[must_use]
    pub const fn should_merge(&self) -> bool {
        self.segment_count >= self.merge_threshold
            && (self.ms_since_last_merge < 0 || self.ms_since_last_merge >= self.cooldown_ms)
    }

    /// Build a status from a live segment count and merge clock.
    #[must_use]
    pub const fn new(segment_count: usize, last_merge_ts: i64, now_ms: i64) -> Self {
        Self {
            segment_count,
            last_merge_ts,
            ms_since_last_merge: if last_merge_ts > 0 {
                now_ms - last_merge_ts
            } else {
                -1
            },
            merge_threshold: CASS_MERGE_SEGMENT_THRESHOLD,
            cooldown_ms: CASS_MERGE_COOLDOWN_MS,
        }
    }
}

/// One CASS message, owned.
///
/// The field set mirrors the incumbent ingest DTO exactly. The three derived
/// columns (`title_prefix`, `content_prefix`, `preview`) are deliberately
/// absent: they are computed at index time by [`CassDerivedColumns::derive`],
/// so a caller cannot desynchronize them from the source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassDocument {
    /// Agent that produced the session.
    pub agent: String,
    /// Normalized workspace identifier.
    pub workspace: Option<String>,
    /// Workspace string exactly as observed.
    pub workspace_original: Option<String>,
    /// Absolute path of the source transcript.
    pub source_path: String,
    /// Zero-based message ordinal within its conversation.
    pub msg_idx: u64,
    /// Message timestamp, seconds since the Unix epoch.
    pub created_at: Option<i64>,
    /// Conversation title, when the transcript carries one.
    pub title: Option<String>,
    /// Message body.
    pub content: String,
    /// Durable source identifier.
    pub source_id: String,
    /// Origin classification (local, imported, federated, ...).
    pub origin_kind: String,
    /// Origin host for federated sources.
    pub origin_host: Option<String>,
    /// Conversation grouping key.
    pub conversation_id: Option<i64>,
}

/// One CASS message, borrowed.
///
/// Ingest paths take this form so a caller streaming from its own store never
/// has to clone message bodies just to hand them to the indexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassDocumentRef<'a> {
    /// Agent that produced the session.
    pub agent: &'a str,
    /// Normalized workspace identifier.
    pub workspace: Option<&'a str>,
    /// Workspace string exactly as observed.
    pub workspace_original: Option<&'a str>,
    /// Absolute path of the source transcript.
    pub source_path: &'a str,
    /// Zero-based message ordinal within its conversation.
    pub msg_idx: u64,
    /// Message timestamp, seconds since the Unix epoch.
    pub created_at: Option<i64>,
    /// Conversation title, when the transcript carries one.
    pub title: Option<&'a str>,
    /// Message body.
    pub content: &'a str,
    /// Durable source identifier.
    pub source_id: &'a str,
    /// Origin classification (local, imported, federated, ...).
    pub origin_kind: &'a str,
    /// Origin host for federated sources.
    pub origin_host: Option<&'a str>,
    /// Conversation grouping key.
    pub conversation_id: Option<i64>,
}

impl CassDocument {
    /// Borrow this document without copying any field.
    #[must_use]
    pub fn as_ref(&self) -> CassDocumentRef<'_> {
        CassDocumentRef {
            agent: &self.agent,
            workspace: self.workspace.as_deref(),
            workspace_original: self.workspace_original.as_deref(),
            source_path: &self.source_path,
            msg_idx: self.msg_idx,
            created_at: self.created_at,
            title: self.title.as_deref(),
            content: &self.content,
            source_id: &self.source_id,
            origin_kind: &self.origin_kind,
            origin_host: self.origin_host.as_deref(),
            conversation_id: self.conversation_id,
        }
    }
}

/// Canonical CASS document identity.
///
/// # Why this is conversation-scoped
///
/// The Tantivy-era identity was `"{source_id}#{msg_idx}"`. That is NOT unique:
/// `source_id` is the *source*, not the conversation — every locally-discovered
/// conversation shares `LOCAL_SOURCE_ID` — so message 0 of every local
/// conversation collides on `"local#0"`. Tantivy tolerated it because a
/// document id there is an ordinary field with no uniqueness constraint. Quill
/// treats the document id as a primary key and refuses duplicates, which is how
/// the collision surfaced at all.
///
/// The conversation discriminator is `conversation_id` when the store has
/// assigned one, and the transcript path otherwise — a conversation always has
/// one or the other, and both are stable across reindexing.
#[must_use]
pub fn cass_document_identity(
    source_id: &str,
    conversation: CassConversationKey<'_>,
    msg_idx: u64,
) -> String {
    let path = conversation.source_path;
    conversation.id.map_or_else(
        || format!("{source_id}#{path}#{msg_idx}"),
        |id| format!("{source_id}#{path}#{id}#{msg_idx}"),
    )
}

/// What discriminates one conversation from another within a source.
///
/// Both axes are carried rather than one: the transcript path alone collides
/// when a single file holds several conversations, and the assigned id alone
/// collides when a caller reuses one id across conversations. Neither is
/// individually guaranteed, so the identity uses the path always and the id
/// additionally whenever the store has assigned one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassConversationKey<'a> {
    /// Absolute transcript path, always present.
    pub source_path: &'a str,
    /// The store's assigned conversation id, when it has one.
    pub id: Option<i64>,
}

impl<'a> CassConversationKey<'a> {
    /// The discriminator for `document`.
    #[must_use]
    pub const fn for_document(document: CassDocumentRef<'a>) -> Self {
        Self {
            source_path: document.source_path,
            id: document.conversation_id,
        }
    }

    /// A discriminator from explicit parts.
    #[must_use]
    pub const fn new(source_path: &'a str, id: Option<i64>) -> Self {
        Self { source_path, id }
    }
}

/// The three columns computed from a document rather than supplied with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassDerivedColumns {
    /// Edge n-grams over the title, empty when the document has no title.
    pub title_prefix: String,
    /// Edge n-grams over a bounded content prefix.
    pub content_prefix: String,
    /// Character-bounded, ellipsized content preview.
    pub preview: String,
    /// `conversation_id` in the canonical encoding for its stored column.
    ///
    /// The field is `I64 { indexed: false, fast: false }` with `stored: true`.
    /// Scribe validates a stored numeric column as exactly eight little-endian
    /// bytes, so this is the required encoding rather than a chosen one: a
    /// friendlier base-10 rendering is refused at accumulation with
    /// `InvalidNumericBytes`. Owned here so the borrowed value slices have
    /// something to point at.
    pub conversation_id_bytes: Option<[u8; 8]>,
}

impl CassDerivedColumns {
    /// Compute every derived column for one document.
    #[must_use]
    pub fn derive(document: CassDocumentRef<'_>) -> Self {
        let (content_prefix, preview) = cass_build_content_prefix_and_preview(document.content);
        Self {
            title_prefix: document
                .title
                .map_or_else(String::new, cass_generate_edge_ngrams),
            content_prefix,
            preview,
            conversation_id_bytes: document.conversation_id.map(i64::to_le_bytes),
        }
    }
}

/// Field ordinals of [`crate::schema::CASS_SEMANTIC_SCHEMA`], by name.
///
/// The ordinals are fixed by the compiled descriptor, so the CASS profile needs
/// no runtime field-handle lookup: this replaces the incumbent `CassFields`
/// struct outright. They are named rather than inlined so a schema reordering
/// is a compile-time edit in one place instead of a silent column swap.
pub mod field {
    /// `agent`, keyword.
    pub const AGENT: u16 = 0;
    /// `workspace`, keyword.
    pub const WORKSPACE: u16 = 1;
    /// `workspace_original`, stored only.
    pub const WORKSPACE_ORIGINAL: u16 = 2;
    /// `source_path`, stored only.
    pub const SOURCE_PATH: u16 = 3;
    /// `msg_idx`, indexed unsigned.
    pub const MSG_IDX: u16 = 4;
    /// `created_at`, indexed and fast signed.
    pub const CREATED_AT: u16 = 5;
    /// `title`, hyphen-normalized text.
    pub const TITLE: u16 = 6;
    /// `content`, hyphen-normalized text.
    pub const CONTENT: u16 = 7;
    /// `title_prefix`, prefix-normalized text.
    pub const TITLE_PREFIX: u16 = 8;
    /// `content_prefix`, prefix-normalized text.
    pub const CONTENT_PREFIX: u16 = 9;
    /// `preview`, stored only.
    pub const PREVIEW: u16 = 10;
    /// `source_id`, keyword.
    pub const SOURCE_ID: u16 = 11;
    /// `origin_kind`, keyword.
    pub const ORIGIN_KIND: u16 = 12;
    /// `origin_host`, keyword.
    pub const ORIGIN_HOST: u16 = 13;
    /// `conversation_id`, stored signed.
    pub const CONVERSATION_ID: u16 = 14;
}

/// The complete column set for one CASS document, ready for accumulation.
///
/// This is the value-construction half of CASS ingest, deliberately separated
/// from the ingest pipeline so it can be built and asserted on without a live
/// index. Text and byte columns borrow — from the document for source columns
/// and from [`CassDerivedColumns`] for derived ones — so building these copies
/// nothing.
///
/// Absent optional columns are omitted rather than written as empty strings: an
/// empty keyword is a real, matchable term, so writing one would make
/// "no workspace recorded" indistinguishable from "workspace is the empty
/// string" at query time.
#[derive(Debug, Clone)]
pub struct CassFieldValues<'a> {
    /// Analyzed and keyword text columns.
    pub indexed: Vec<crate::scribe::IndexedFieldValue<'a>>,
    /// Numeric columns.
    pub numeric: Vec<crate::scribe::IndexedNumericValue>,
    /// Stored-only byte columns.
    pub stored: Vec<crate::scribe::StoredFieldValue<'a>>,
}

impl<'a> CassFieldValues<'a> {
    /// Build every column for `document`, borrowing derived text from `derived`.
    #[must_use]
    pub fn build(document: CassDocumentRef<'a>, derived: &'a CassDerivedColumns) -> Self {
        use crate::scribe::{IndexedFieldValue, IndexedNumericValue, StoredFieldValue};

        let mut indexed = Vec::with_capacity(10);
        let mut numeric = Vec::with_capacity(2);
        let mut stored = Vec::with_capacity(4);

        indexed.push(IndexedFieldValue::new(field::AGENT, document.agent));
        indexed.push(IndexedFieldValue::new(field::SOURCE_ID, document.source_id));
        indexed.push(IndexedFieldValue::new(
            field::ORIGIN_KIND,
            document.origin_kind,
        ));
        indexed.push(IndexedFieldValue::new(field::CONTENT, document.content));
        indexed.push(IndexedFieldValue::new(
            field::CONTENT_PREFIX,
            &derived.content_prefix,
        ));
        if let Some(workspace) = document.workspace {
            indexed.push(IndexedFieldValue::new(field::WORKSPACE, workspace));
        }
        if let Some(origin_host) = document.origin_host {
            indexed.push(IndexedFieldValue::new(field::ORIGIN_HOST, origin_host));
        }
        if let Some(title) = document.title {
            indexed.push(IndexedFieldValue::new(field::TITLE, title));
            indexed.push(IndexedFieldValue::new(
                field::TITLE_PREFIX,
                &derived.title_prefix,
            ));
        }

        numeric.push(IndexedNumericValue::u64(field::MSG_IDX, document.msg_idx));
        if let Some(created_at) = document.created_at {
            numeric.push(IndexedNumericValue::i64(field::CREATED_AT, created_at));
        }

        stored.push(StoredFieldValue::new(
            field::SOURCE_PATH,
            document.source_path.as_bytes(),
        ));
        stored.push(StoredFieldValue::new(
            field::PREVIEW,
            derived.preview.as_bytes(),
        ));
        if let Some(workspace_original) = document.workspace_original {
            stored.push(StoredFieldValue::new(
                field::WORKSPACE_ORIGINAL,
                workspace_original.as_bytes(),
            ));
        }
        if let Some(conversation_id) = derived.conversation_id_bytes.as_ref() {
            stored.push(StoredFieldValue::new(
                field::CONVERSATION_ID,
                conversation_id,
            ));
        }

        Self {
            indexed,
            numeric,
            stored,
        }
    }
}

impl CassDocument {
    /// Project onto the owned form Quill ingests.
    ///
    /// This derives the computed columns and copies every value, because
    /// [`SchemaDocument`] owns its columns while [`CassFieldValues`] borrows
    /// them. A caller that already holds its documents in memory for the whole
    /// batch can skip the copy by building [`CassFieldValues`] directly.
    #[must_use]
    pub fn to_schema_document(&self) -> SchemaDocument {
        let derived = CassDerivedColumns::derive(self.as_ref());
        let values = CassFieldValues::build(self.as_ref(), &derived);
        SchemaDocument {
            id: cass_document_identity(
                &self.source_id,
                CassConversationKey::for_document(self.as_ref()),
                self.msg_idx,
            ),
            indexed: values
                .indexed
                .iter()
                .map(|value| (value.field_ord, value.text.to_owned()))
                .collect(),
            numeric: values
                .numeric
                .iter()
                .map(|value| (value.field_ord, value.value))
                .collect(),
            stored: values
                .stored
                .iter()
                .map(|value| (value.field_ord, value.bytes.to_vec()))
                .collect(),
        }
    }
}

/// Build the edge-ngram content prefix and the stored preview together.
///
/// They are returned as a pair because both read the same content and must
/// agree about it; computing them apart is how a prefix column drifts from the
/// preview a user is shown.
#[must_use]
pub fn cass_build_content_prefix_and_preview(content: &str) -> (String, String) {
    (
        cass_generate_edge_ngrams(cass_prefix_source(content, CONTENT_PREFIX_MAX_BYTES)),
        cass_build_preview(content, PREVIEW_MAX_CHARS),
    )
}

/// The largest char-boundary prefix of `content` that is at most `max_bytes`.
///
/// UTF-8 scalars are at most four bytes, so the boundary at or below
/// `max_bytes` is at most three bytes below it: the backward walk terminates
/// in at most four steps regardless of input length.
#[must_use]
pub fn cass_prefix_source(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }
    let mut end = max_bytes;
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    &content[..end]
}

/// Character-bounded preview, ellipsized only when it actually truncates.
#[must_use]
pub fn cass_build_preview(content: &str, max_chars: usize) -> String {
    let mut cut = content.len();
    for (count, (byte_idx, _)) in content.char_indices().enumerate() {
        if count == max_chars {
            cut = byte_idx;
            break;
        }
    }
    let truncated = cut < content.len();
    let mut out = String::with_capacity(cut + if truncated { '…'.len_utf8() } else { 0 });
    out.push_str(&content[..cut]);
    if truncated {
        out.push('…');
    }
    out
}
