use std::path::{Path, PathBuf};
#[cfg(feature = "bench-internals")]
use std::sync::Arc;
use std::sync::RwLock;
#[cfg(feature = "bench-internals")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};

use asupersync::Cx;
use asupersync::sync::Mutex;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::types::{DocId, IndexableDocument};
use tantivy::schema::Schema;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, Searcher, TantivyDocument, Term};
use tracing::{debug, warn};

#[cfg(feature = "bench-internals")]
use crate::benchmark::{
    BenchmarkDirectorySyncState, BenchmarkWriterAttestation, BenchmarkWriterMode,
    BenchmarkWriterReceipt,
};
use crate::schema::{SchemaFields, TOKENIZER_NAME, build_schema};
use crate::tokenizer::build_tokenizer;

/// Default heap size for the Tantivy `IndexWriter` (50 MB).
pub const WRITER_HEAP_BYTES: usize = 50_000_000;

/// Maximum query length in characters. Queries exceeding this are truncated
/// with a warning log. Prevents pathological parsing of enormous inputs.
pub const MAX_QUERY_LENGTH: usize = 10_000;

/// Which writer construction the caller explicitly asked for.
///
/// The plan, not the thread count, decides whether a benchmark receipt exists.
/// Inferring "benchmark" from `writer_threads: None` made every ordinary
/// `create`/`open`/`in_memory` index carry a shipping-auto receipt it never
/// asked for, which would let ordinary constructions masquerade as screened
/// benchmark candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriterPlan {
    /// Ordinary shipping construction through Tantivy's own width selection.
    /// Never produces a benchmark receipt.
    Shipping,
    /// Read-only construction: no `IndexWriter`, no writer lock, no indexing
    /// thread pool. Only [`TantivyIndex::open_read_only`] produces this plan.
    ReadOnly,
    /// Ordinary construction at a pinned width. Never produces a benchmark
    /// receipt either.
    ///
    /// Width and receipt are independent axes. Collapsing this into
    /// `Shipping` silently moved the single-threaded oracle off
    /// `writer_with_num_threads(1, heap)` and onto Tantivy's auto selection —
    /// a real behaviour change bought only to express "no receipt".
    ///
    /// Gated with its only constructor, so the default build carries no
    /// variant that nothing can build.
    #[cfg(feature = "tantivy-oracle")]
    PinnedWidth(usize),
    /// Explicit benchmark construction through Tantivy's own width selection.
    #[cfg(feature = "bench-internals")]
    BenchmarkShippingAuto,
    /// Explicit benchmark construction at a pinned width.
    #[cfg(feature = "bench-internals")]
    BenchmarkFixed(usize),
}

#[cfg(feature = "bench-internals")]
impl WriterPlan {
    /// Width recorded on the benchmark accessor.
    ///
    /// Only a benchmark plan reports one: `PinnedWidth` pins a width without
    /// claiming any screening identity. Gated with the field it feeds, so the
    /// default build has no unreachable helper to warn about.
    pub const fn benchmark_threads(self) -> Option<usize> {
        match self {
            Self::BenchmarkFixed(threads) => Some(threads),
            Self::Shipping | Self::ReadOnly | Self::BenchmarkShippingAuto => None,
            #[cfg(feature = "tantivy-oracle")]
            Self::PinnedWidth(_) => None,
        }
    }
}

/// A Tantivy-backed full-text search index implementing [`LexicalSearch`].
///
/// Thread-safe for concurrent reads. Writes are serialized internally via
/// the Tantivy `IndexWriter` (which requires `&mut self` for `add_document`
/// but is wrapped here for the trait interface).
pub struct TantivyIndex {
    pub(super) index: Index,
    pub(super) fields: SchemaFields,
    pub(super) reader: IndexReader,
    /// `None` when the index was opened through [`Self::open_read_only`]: no
    /// `IndexWriter` is constructed, so Tantivy's exclusive
    /// `.tantivy-writer.lock` is never taken and concurrent read-only
    /// processes can all serve the same published generation. Every write
    /// entry point returns a typed `InvalidConfig` error instead of a writer
    /// in that state.
    pub(super) writer: Mutex<Option<IndexWriter>>,
    /// True when opened through [`Self::open_read_only`].
    pub(super) read_only: bool,
    pub(super) doc_count: AtomicUsize,
    /// Append-only `ordinal -> doc_id` table backing the fast id-materialization
    /// path. Index `i` holds the `doc_id` of the document assigned ordinal `i`.
    /// Grows by one per indexed document (including re-upserts); ordinals are
    /// monotonic and never reused, so it stays correct across deletes/merges
    /// (a deleted doc's ordinal simply never appears in search results).
    pub(super) ord_table: RwLock<Vec<DocId>>,
    pub(super) path: Option<PathBuf>,
    /// Exact writer-pool width accepted by Tantivy's benchmark constructor.
    #[cfg(feature = "bench-internals")]
    pub(super) benchmark_writer_threads: Option<usize>,
    /// Typed receipt for the benchmark writer construction, when this index
    /// was built through a benchmark seam.
    #[cfg(feature = "bench-internals")]
    pub(super) benchmark_writer_receipt: Option<BenchmarkWriterReceipt>,
    /// One-shot live attestation for the benchmark writer construction.
    #[cfg(feature = "bench-internals")]
    pub(super) benchmark_writer_attestation: Option<BenchmarkWriterAttestation>,
    /// Shared only with the benchmark-created on-disk directory wrapper.
    #[cfg(feature = "bench-internals")]
    pub(super) benchmark_directory_sync_state: Option<Arc<BenchmarkDirectorySyncState>>,
    /// Set only by the fresh-fixture QG-4 no-merge setup API.
    #[cfg(feature = "bench-internals")]
    pub(super) benchmark_qg4_no_merge_prepared: AtomicBool,
    /// Which Tantivy writer constructor this index actually invoked.
    ///
    /// Per-instance and test-only: a plan or receipt records what the caller
    /// *asked for*, so a test reading either proves nothing about the call that
    /// ran. This is written by the helper that performs the call itself. It is
    /// deliberately not a global counter — parallel tests would race one.
    #[cfg(test)]
    pub(super) observed_writer_call: WriterCall,
}

/// The Tantivy writer constructor a `TantivyIndex` actually invoked.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriterCall {
    /// `Index::writer(heap)` — Tantivy selected the width.
    Auto,
    /// `Index::writer_with_num_threads(threads, heap)`.
    ///
    /// Gated exactly as `call_fixed_writer` is: the only thing that can build
    /// this variant is that helper, so without those features it would be a
    /// variant nothing constructs.
    #[cfg(any(feature = "tantivy-oracle", feature = "bench-internals"))]
    Fixed(usize),
}

/// Perform the shipping-auto Tantivy call, recording that it was the one made.
fn call_auto_writer(
    index: &Index,
    writer_heap_bytes: usize,
    #[cfg(test)] observed: &mut WriterCall,
) -> tantivy::Result<IndexWriter> {
    #[cfg(test)]
    {
        *observed = WriterCall::Auto;
    }
    index.writer(writer_heap_bytes)
}

/// Perform the pinned-width Tantivy call, recording that it was the one made.
///
/// Gated to the features that actually reach it — the oracle's pinned plan and
/// the benchmark seams — so the default build has no helper without a caller.
#[cfg(any(feature = "tantivy-oracle", feature = "bench-internals"))]
pub fn call_fixed_writer(
    index: &Index,
    threads: usize,
    writer_heap_bytes: usize,
    #[cfg(test)] observed: &mut WriterCall,
) -> tantivy::Result<IndexWriter> {
    #[cfg(test)]
    {
        *observed = WriterCall::Fixed(threads);
    }
    index.writer_with_num_threads(threads, writer_heap_bytes)
}

impl std::fmt::Debug for TantivyIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TantivyIndex")
            .field(
                "staged_doc_count_hint",
                &self.doc_count.load(Ordering::Relaxed),
            )
            .field("path", &self.path)
            .field("read_only", &self.read_only)
            .finish_non_exhaustive()
    }
}

impl TantivyIndex {
    pub(super) fn checked_searcher_doc_count(searcher: &Searcher) -> SearchResult<usize> {
        usize::try_from(searcher.num_docs()).map_err(|_| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: "current Tantivy reader document count does not fit usize".into(),
        })
    }

    /// Typed error for a write attempted through a read-only handle.
    pub(super) fn read_only_writer_error(phase: &str) -> SearchError {
        SearchError::InvalidConfig {
            field: format!("tantivy.writer.{phase}"),
            value: "read_only".to_owned(),
            reason: "index was opened with TantivyIndex::open_read_only and holds no IndexWriter"
                .to_owned(),
        }
    }

    pub(super) fn map_writer_lock_error(
        phase: &str,
        error: asupersync::sync::LockError,
    ) -> SearchError {
        match error {
            asupersync::sync::LockError::Poisoned => SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(std::io::Error::other("writer mutex poisoned")),
            },
            asupersync::sync::LockError::Cancelled => SearchError::Cancelled {
                phase: phase.into(),
                reason: "writer lock cancelled".into(),
            },
            asupersync::sync::LockError::PolledAfterCompletion => SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(std::io::Error::other(format!(
                    "writer mutex future reused after completion during {phase}"
                ))),
            },
            asupersync::sync::LockError::TimedOut(deadline) => SearchError::Cancelled {
                phase: phase.into(),
                reason: format!("writer lock timed out at {deadline:?}"),
            },
        }
    }

    /// Create a new Tantivy index at the given directory path.
    ///
    /// If the directory does not exist, it will be created.
    /// If an index already exists at this path, it will be opened.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the Tantivy index cannot be
    /// created or opened.
    pub fn create(path: &Path) -> SearchResult<Self> {
        let (schema, fields) = build_schema();

        std::fs::create_dir_all(path).map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })?;

        let index = Index::create_in_dir(path, schema.clone())
            .or_else(|_| {
                // If creation fails (already exists), try opening instead.
                Index::open_in_dir(path)
            })
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(e),
            })?;

        Self::from_index(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            WRITER_HEAP_BYTES,
        )
    }

    /// Open an existing Tantivy index at the given directory path.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` if the path does not exist.
    /// Returns `SearchError::SubsystemError` if the index cannot be opened.
    pub fn open(path: &Path) -> SearchResult<Self> {
        if !path.exists() {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }

        let (schema, fields) = build_schema();
        let index = Index::open_in_dir(path).map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })?;

        Self::from_index(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            WRITER_HEAP_BYTES,
        )
    }

    /// Open an existing on-disk index for reading only.
    ///
    /// Unlike [`Self::open`], no [`IndexWriter`] is constructed, so this never
    /// takes Tantivy's exclusive `.tantivy-writer.lock`, never allocates the
    /// writer heap, and never spawns indexing threads. Any number of processes
    /// can open the same published generation this way while a writer holds
    /// the lock elsewhere. Write entry points on the returned index fail with
    /// a typed `InvalidConfig` error instead of mutating anything.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` if the path does not exist, or
    /// `SearchError::SubsystemError` if the index cannot be opened.
    pub fn open_read_only(path: &Path) -> SearchResult<Self> {
        if !path.exists() {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }

        let (schema, fields) = build_schema();
        let index = Index::open_in_dir(path).map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })?;

        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            Some(path.to_path_buf()),
            WRITER_HEAP_BYTES,
            WriterPlan::ReadOnly,
            #[cfg(feature = "bench-internals")]
            None,
        )
    }

    /// True when this handle was opened through [`Self::open_read_only`] and
    /// therefore holds no writer.
    #[must_use]
    pub const fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Create an in-memory Tantivy index (useful for testing).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the index cannot be created.
    pub fn in_memory() -> SearchResult<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema.clone());
        Self::from_index(index, schema, fields, None, WRITER_HEAP_BYTES)
    }

    /// Internal constructor shared by `create`, `open`, and `in_memory`.
    pub(super) fn from_index(
        index: Index,
        schema: Schema,
        fields: SchemaFields,
        path: Option<PathBuf>,
        writer_heap_bytes: usize,
    ) -> SearchResult<Self> {
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            path,
            writer_heap_bytes,
            WriterPlan::Shipping,
            #[cfg(feature = "bench-internals")]
            None,
        )
    }

    pub(super) fn from_index_with_writer_threads(
        index: Index,
        _schema: Schema,
        mut fields: SchemaFields,
        path: Option<PathBuf>,
        writer_heap_bytes: usize,
        plan: WriterPlan,
        #[cfg(feature = "bench-internals")] benchmark_directory_sync_state: Option<
            Arc<BenchmarkDirectorySyncState>,
        >,
    ) -> SearchResult<Self> {
        // Resolve the `ord` fast field against the *actual* index schema so
        // indexes created before this field existed (no `ord`) cleanly fall back
        // to docstore materialization instead of using a phantom field handle.
        fields.ord = index.schema().get_field("ord").ok();

        // Register our custom tokenizer.
        let tokenizer_manager = index.tokenizers().clone();
        tokenizer_manager.register(TOKENIZER_NAME, build_tokenizer());

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(e),
            })?;

        // The selection branch is the single source of both the writer and its
        // receipt seed. Nothing downstream can relabel a `writer_with_num_threads`
        // construction as shipping-auto, or vice versa, because the mode is
        // produced here by the arm that actually ran.
        #[cfg(feature = "bench-internals")]
        let mut benchmark_writer_receipt = None;
        #[cfg(feature = "bench-internals")]
        let mut benchmark_writer_attestation = None;
        // Default is overwritten by whichever helper actually runs below; the
        // helper is the only writer of this value.
        #[cfg(test)]
        let mut observed_writer_call = WriterCall::Auto;
        let writer = match plan {
            // Read-only handles never construct a writer: that is the whole
            // point of the plan. Tantivy's writer lock stays free for the
            // process that actually publishes generations.
            WriterPlan::ReadOnly => Ok(None),
            // Ordinary shipping construction reaches the same Tantivy call as a
            // benchmark shipping-auto construction, and deliberately produces no
            // receipt: only an explicit benchmark plan may claim one.
            WriterPlan::Shipping => call_auto_writer(
                &index,
                writer_heap_bytes,
                #[cfg(test)]
                &mut observed_writer_call,
            )
            .map(Some),
            // Pinned but unscreened: the same Tantivy call a benchmark fixed
            // plan makes, deliberately without a receipt.
            #[cfg(feature = "tantivy-oracle")]
            WriterPlan::PinnedWidth(thread_count) => call_fixed_writer(
                &index,
                thread_count,
                writer_heap_bytes,
                #[cfg(test)]
                &mut observed_writer_call,
            )
            .map(Some),
            #[cfg(feature = "bench-internals")]
            WriterPlan::BenchmarkShippingAuto => {
                let writer = call_auto_writer(
                    &index,
                    writer_heap_bytes,
                    #[cfg(test)]
                    &mut observed_writer_call,
                );
                if writer.is_ok() {
                    let receipt = BenchmarkWriterReceipt::seed(
                        BenchmarkWriterMode::ShippingAuto,
                        writer_heap_bytes,
                        &index,
                    )?;
                    benchmark_writer_attestation =
                        Some(BenchmarkWriterAttestation::mint(receipt.clone()));
                    benchmark_writer_receipt = Some(receipt);
                }
                writer.map(Some)
            }
            #[cfg(feature = "bench-internals")]
            WriterPlan::BenchmarkFixed(thread_count) => {
                let writer = call_fixed_writer(
                    &index,
                    thread_count,
                    writer_heap_bytes,
                    #[cfg(test)]
                    &mut observed_writer_call,
                );
                if writer.is_ok() {
                    let receipt = BenchmarkWriterReceipt::seed(
                        BenchmarkWriterMode::Fixed {
                            threads: thread_count,
                        },
                        writer_heap_bytes,
                        &index,
                    )?;
                    benchmark_writer_attestation =
                        Some(BenchmarkWriterAttestation::mint(receipt.clone()));
                    benchmark_writer_receipt = Some(receipt);
                }
                writer.map(Some)
            }
        }
        .map_err(|e| SearchError::SubsystemError {
            subsystem: "tantivy",
            source: Box::new(e),
        })?;
        let read_only = writer.is_none();

        // Count existing documents.
        let searcher = reader.searcher();
        let doc_count = usize::try_from(searcher.num_docs()).unwrap_or(usize::MAX);

        // Restore the ordinal→doc_id table for a reopened on-disk index from its
        // sidecar so id materialization keeps the fast path instead of falling
        // back to the docstore. The sidecar is best-effort: if it is absent,
        // corrupt, or stale the table is left short and those ordinals resolve
        // via the docstore (correct, just unaccelerated).
        let mut ord_table = match (path.as_deref(), fields.ord) {
            (Some(dir), Some(_)) => Self::load_ord_table_sidecar(dir).unwrap_or_default(),
            _ => Vec::new(),
        };
        if fields.ord.is_some() {
            // Guard against a stale-short sidecar (e.g. a persist that failed on
            // the last commit): `Column::max_value` is O(1) columnar metadata, so
            // padding the table to cover the highest existing ordinal keeps the
            // next assigned ordinal (`ord_table.len()`) collision-free, and the
            // padded empty slots fall back to the docstore on read.
            let mut max_ord = 0u64;
            let mut saw_ord = false;
            for sr in searcher.segment_readers() {
                if let Ok(col) = sr.fast_fields().u64("ord")
                    && col.num_docs() > 0
                {
                    max_ord = max_ord.max(col.max_value());
                    saw_ord = true;
                }
            }
            if saw_ord {
                let needed = usize::try_from(max_ord)
                    .unwrap_or(usize::MAX)
                    .saturating_add(1);
                if ord_table.len() < needed {
                    ord_table.resize(needed, DocId::default());
                }
            }
        }

        Ok(Self {
            index,
            fields,
            reader,
            writer: Mutex::new(writer),
            read_only,
            doc_count: AtomicUsize::new(doc_count),
            ord_table: RwLock::new(ord_table),
            path,
            #[cfg(feature = "bench-internals")]
            benchmark_writer_threads: plan.benchmark_threads(),
            // Seeded above by the selection branch itself, so the record and
            // the call it describes cannot disagree.
            //
            // This constructor is shared with the default build, so the field
            // needs the same gate as its neighbour above — without it the
            // no-feature build names a field that does not exist.
            #[cfg(feature = "bench-internals")]
            benchmark_writer_receipt,
            #[cfg(feature = "bench-internals")]
            benchmark_writer_attestation,
            #[cfg(feature = "bench-internals")]
            benchmark_directory_sync_state,
            #[cfg(feature = "bench-internals")]
            benchmark_qg4_no_merge_prepared: AtomicBool::new(false),
            #[cfg(test)]
            observed_writer_call,
        })
    }

    /// Load the persisted `ordinal → doc_id` table sidecar (`ord_table.json`)
    /// from an on-disk index directory. Returns `None` on any error or absence
    /// so the caller can fall back to docstore materialization.
    fn load_ord_table_sidecar(dir: &Path) -> Option<Vec<DocId>> {
        let file = std::fs::File::open(dir.join("ord_table.json")).ok()?;
        serde_json::from_reader(std::io::BufReader::new(file)).ok()
    }

    /// Persist the `ordinal → doc_id` table to the index directory sidecar so a
    /// later `open` can restore the fast id-materialization path. Best-effort:
    /// errors are logged and swallowed (the in-memory fast path is unaffected;
    /// only a future reopen would fall back to the docstore). Written atomically
    /// via a temp file + rename. No-op for in-memory indexes / pre-`ord` schemas.
    pub(super) fn persist_ord_table(&self) {
        let Some(dir) = self.path.as_ref() else {
            return;
        };
        if self.fields.ord.is_none() {
            return;
        }
        let Ok(table) = self.ord_table.read() else {
            return;
        };
        let tmp = dir.join("ord_table.json.tmp");
        let final_path = dir.join("ord_table.json");
        let write = std::fs::File::create(&tmp).and_then(|file| {
            let mut writer = std::io::BufWriter::new(file);
            serde_json::to_writer(&mut writer, &*table).map_err(std::io::Error::other)?;
            std::io::Write::flush(&mut writer)
        });
        match write {
            Ok(()) => {
                if let Err(e) = std::fs::rename(&tmp, &final_path) {
                    debug!(error = %e, "ord_table sidecar rename failed; reopen will use docstore");
                }
            }
            Err(e) => {
                debug!(error = %e, "ord_table sidecar write failed; reopen will use docstore");
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }

    /// Convert an `IndexableDocument` to a Tantivy document.
    pub(super) fn to_tantivy_doc(&self, doc: &IndexableDocument) -> TantivyDocument {
        let mut tantivy_doc = TantivyDocument::new();
        tantivy_doc.add_text(self.fields.id, &doc.id);
        tantivy_doc.add_text(self.fields.content, &doc.content);
        tantivy_doc.add_text(self.fields.title, doc.title.as_deref().unwrap_or(""));

        // Serialize metadata as JSON string.
        if !doc.metadata.is_empty()
            && let Ok(json) = serde_json::to_string(&doc.metadata)
        {
            tantivy_doc.add_text(self.fields.metadata_json, &json);
        }

        tantivy_doc
    }

    /// Assign the next dense ordinal to `tantivy_doc` and append `doc_id` to the
    /// `ord_table` so id materialization can skip the docstore decompress.
    ///
    /// Must be called while holding the writer lock: that serializes ordinal
    /// assignment with `add_document`, so ordinal `i` always corresponds to
    /// `ord_table[i]`. No-op when the schema has no `ord` field (pre-`ord`
    /// indexes) or when the table lock is poisoned (the doc is left without an
    /// ordinal and that hit falls back to docstore materialization).
    pub(super) fn assign_ord(&self, tantivy_doc: &mut TantivyDocument, doc_id: &str) {
        if let Some(ord_field) = self.fields.ord
            && let Ok(mut table) = self.ord_table.write()
        {
            let ord = u64::try_from(table.len()).unwrap_or(u64::MAX);
            table.push(DocId::from(doc_id));
            tantivy_doc.add_u64(ord_field, ord);
        }
    }

    /// Delete a document by its ID.
    ///
    /// A successful return commits the deletion and refreshes the readable
    /// generation before releasing the writer lock.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the writer lock is poisoned
    /// or cancelled.
    pub async fn delete_document(&self, cx: &Cx, doc_id: &str) -> SearchResult<()> {
        let term = Term::from_field_text(self.fields.id, doc_id);
        let mut writer_guard = self
            .writer
            .lock(cx)
            .await
            .map_err(|error| Self::map_writer_lock_error("tantivy.delete", error))?;
        let Some(writer) = writer_guard.as_mut() else {
            return Err(Self::read_only_writer_error("delete"));
        };
        writer.delete_term(term);
        writer
            .commit()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;

        self.reader
            .reload()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "tantivy",
                source: Box::new(error),
            })?;

        let actual = Self::checked_searcher_doc_count(&self.reader.searcher())?;
        self.doc_count.store(actual, Ordering::Relaxed);

        debug!(doc_count = actual, %doc_id, "tantivy delete committed and reloaded");
        Ok(())
    }

    /// Returns the directory path for this index, if on-disk.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Cloneable handle to the underlying Tantivy index.
    ///
    /// This is primarily used by durability wrappers that protect/verify
    /// segment artifacts outside the lexical crate.
    #[must_use]
    pub fn index_handle(&self) -> Index {
        self.index.clone()
    }

    /// Truncate an overlong query and log a warning.
    pub(super) fn truncate_query(query: &str) -> &str {
        // UTF-8 uses at least one byte per character, so this is a cheap
        // common-case proof that the query is within the character limit.
        if query.len() <= MAX_QUERY_LENGTH {
            return query;
        }

        // `char_indices().nth(MAX_QUERY_LENGTH)` points at the first character
        // outside the allowed prefix. If it is absent, the string exceeds the
        // byte count but still contains at most MAX_QUERY_LENGTH characters.
        let Some((end, _)) = query.char_indices().nth(MAX_QUERY_LENGTH) else {
            return query;
        };
        warn!(
            original_len_bytes = query.len(),
            max_chars = MAX_QUERY_LENGTH,
            "query truncated to MAX_QUERY_LENGTH"
        );
        &query[..end]
    }

    /// Render the shipping parser's tantivy query tree for one input.
    ///
    /// Diagnostic-only surface for differential structure archaeology
    /// (bd-1i4j4): the execution tree tantivy evaluates decides its f32
    /// accumulation association, so divergence work needs to SEE it.
    #[doc(hidden)]
    #[must_use]
    pub fn debug_shipping_parse(&self, query: &str) -> String {
        let query = Self::truncate_query(query);
        format!("{:?}", self.parse_query_shipping(query))
    }
}
