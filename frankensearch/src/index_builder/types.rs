use std::path::PathBuf;
use std::sync::Arc;

use frankensearch_core::traits::LexicalRead;
use frankensearch_embed::auto_detect::TwoTierAvailability;
use frankensearch_index::TwoTierIndex;

/// Per-arm byte accounting for a completed build (bd-8nqz.3).
///
/// Vector-only size reporting hid the lexical arm entirely; every arm that
/// wrote bytes must appear here so no aggregate can mask a missing arm.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexSizeBreakdown {
    /// Sum of all arms below.
    pub total: u64,
    /// Fast-tier FSVI bytes (dedicated or fallback filename).
    pub vector_fast: u64,
    /// Quality-tier FSVI bytes (0 when no quality index was built).
    pub vector_quality: u64,
    /// Recursive size of the lexical index directory (0 when absent).
    pub lexical: u64,
}

/// Receipt for the lexical indexing arm of a build (bd-8nqz.3).
///
/// Lexical admission is independent of embedding outcome: every valid source
/// document is attempted here even when its embeddings failed, so the
/// documents most in need of lexical fallback remain lexically searchable.
#[derive(Debug, Clone)]
pub struct LexicalArmReceipt {
    /// Active backend: `"quill"` or `"tantivy"`.
    pub backend: &'static str,
    /// Directory the lexical index was written to.
    pub path: PathBuf,
    /// Documents attempted (all valid source documents).
    pub attempted: usize,
    /// Documents successfully indexed.
    pub indexed: usize,
    /// Per-document lexical errors (`doc_id`, error message).
    pub errors: Vec<(String, String)>,
    /// Published manifest generation. Currently `None`: the keeper snapshot
    /// does not expose its generation publicly yet; the root-bound reader
    /// work (bd-8nqz.2) adds that accessor and fills this in.
    pub generation: Option<u64>,
    /// Whether the lexical index was published (bulk seal / commit reached).
    pub published: bool,
}

/// Statistics from a completed index build.
#[derive(Debug, Clone)]
pub struct IndexBuildStats {
    /// Total valid source documents submitted to the build.
    pub source_count: usize,
    /// Number of documents successfully indexed into the fast vector tier.
    pub doc_count: usize,
    /// Number of documents whose fast embedding failed (absent from the
    /// vector tiers; still admitted to the lexical arm when enabled).
    pub error_count: usize,
    /// Per-document fast-embedding errors (`doc_id`, error message).
    pub errors: Vec<(String, String)>,
    /// Documents successfully indexed into the quality vector tier.
    /// Zero when no quality embedder is configured.
    pub quality_indexed: usize,
    /// Per-document quality-embedding errors (`doc_id`, error message).
    /// These documents remain in the fast tier (fast-only degradation).
    pub quality_errors: Vec<(String, String)>,
    /// Receipt for the lexical arm. `None` when lexical indexing is compiled
    /// out, no backend is selected, or no document reached lexical staging.
    pub lexical: Option<LexicalArmReceipt>,
    /// Per-arm byte accounting (replaces vector-only size reporting).
    pub size_bytes: IndexSizeBreakdown,
    /// Total build time in milliseconds.
    pub total_ms: f64,
    /// Time spent on embedding in milliseconds.
    pub embed_ms: f64,
    /// Time spent building the lexical arm in milliseconds (0 when skipped).
    pub lexical_ms: f64,
    /// Whether a quality-tier index was built.
    pub has_quality_index: bool,
    /// Embedder availability this generation was actually built with.
    ///
    /// The index is written with the vector identity of whatever embedder was
    /// resolved, so a [`TwoTierAvailability::HashOnly`] build produces a
    /// permanently non-semantic generation: installing a model afterwards
    /// cannot repair it, because the stored vectors are hashes. Callers that
    /// care about semantic quality must inspect this rather than assume the
    /// build was semantic — a degraded build otherwise succeeds silently and
    /// returns plausible-looking hits at query time (`bd-a6zt`).
    pub embedder_availability: TwoTierAvailability,
}

impl IndexBuildStats {
    /// Whether this generation was built with a degraded embedder stack.
    ///
    /// `true` means the index content itself is degraded, not merely the
    /// current process configuration.
    #[must_use]
    pub const fn is_degraded_generation(&self) -> bool {
        self.embedder_availability.is_degraded()
    }
}

/// Progress update during index building.
#[derive(Debug, Clone)]
pub struct IndexProgress {
    /// Documents processed so far.
    pub completed: usize,
    /// Total documents to process.
    pub total: usize,
    /// Current phase description.
    pub phase: &'static str,
}

/// The lexical engine admitted by [`super::open_hybrid`].
///
/// [`Self::Quill`] is the default facade authority. [`Self::TantivyOracle`]
/// is an explicitly selected rollback/oracle reader; it never means that the
/// default `lexical` facade alias resolved to Tantivy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexicalReaderBackend {
    /// The default Quill lexical reader.
    Quill,
    /// The explicit Tantivy oracle or accepted rollback reader.
    TantivyOracle,
}

/// The opened arms of a hybrid index directory (bd-8nqz.3).
#[derive(Clone)]
pub struct HybridIndexParts {
    /// Two-tier vector index.
    pub vectors: Arc<TwoTierIndex>,
    /// Active lexical reader for `<dir>/lexical`, when one exists and a
    /// lexical backend is compiled in.
    pub lexical: Option<Arc<dyn LexicalRead>>,
    /// Actual lexical engine behind [`Self::lexical`].
    ///
    /// A caller that requires Quill's semantic behavior can fail closed when
    /// this is [`Some(LexicalReaderBackend::TantivyOracle)`]. It is `None`
    /// exactly when [`Self::lexical`] is absent.
    pub lexical_backend: Option<LexicalReaderBackend>,
}

impl std::fmt::Debug for HybridIndexParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridIndexParts")
            .field("has_lexical", &self.lexical.is_some())
            .field("lexical_backend", &self.lexical_backend)
            .finish_non_exhaustive()
    }
}
