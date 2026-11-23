use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use frankensearch_core::SearchResult;
use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::generation::ArtifactGenerationIdentityV1;
use hnsw_rs::prelude::{DistDot, Hnsw};
#[cfg(test)]
use hnsw_rs::prelude::{Neighbour, PointId};
use serde::{Deserialize, Serialize};

use crate::{SHA256_BYTES, VectorIndex};

/// Default HNSW `M` (max connections per node).
pub const HNSW_DEFAULT_M: usize = 16;
/// Default HNSW `ef_construction` (build-time beam width).
pub const HNSW_DEFAULT_EF_CONSTRUCTION: usize = 200;
/// Default HNSW `ef_search` (query-time beam width).
pub const HNSW_DEFAULT_EF_SEARCH: usize = 100;
/// Default HNSW max layer depth.
pub const HNSW_DEFAULT_MAX_LAYER: usize = 16;

// `hnsw_rs` documents parallel insertion as efficient only for batches of
// roughly 1,000 points per Rayon worker. Small batches also have no useful
// parallel speedup, so keep their construction deterministic and serial.
pub(super) const HNSW_PARALLEL_INSERT_MIN_POINTS_PER_THREAD: usize = 1_000;

/// ANN construction/runtime parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswConfig {
    /// HNSW `M` (max connections per node).
    pub m: usize,
    /// HNSW `ef_construction` (build-time beam width).
    pub ef_construction: usize,
    /// Default HNSW `ef_search` (query-time beam width).
    pub ef_search: usize,
    /// Maximum HNSW layer depth.
    pub max_layer: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: HNSW_DEFAULT_M,
            ef_construction: HNSW_DEFAULT_EF_CONSTRUCTION,
            ef_search: HNSW_DEFAULT_EF_SEARCH,
            max_layer: HNSW_DEFAULT_MAX_LAYER,
        }
    }
}

/// Current on-disk metadata format. v2 added the native graph sidecars
/// (`*.hnsw.graph` + `*.hnsw.data`) alongside the JSON metadata; v3 records
/// graphs built with the dimension-aware `DistDot` roundoff budget; v4 replaces
/// the sampled source fingerprint with a digest of every live vector; v5 records
/// the exact native sidecar generation and basename selected during publication;
/// v6 attests point/layer invariants after build and native load; v7 persists
/// graph-origin-to-source-row mappings for incremental insertion. Older native
/// graphs must be rebuilt under the current persistence contract.
pub const HNSW_META_FORMAT_CURRENT: u32 = 7;

pub(super) const HNSW_GENERATION_RECEIPT_VERSION: u32 = 1;
pub(super) const HNSW_GENERATION_RECEIPT_FILENAME: &str = ".frankensearch-hnsw-ready.json";
pub(super) const HNSW_GENERATION_RECEIPT_MAX_BYTES: usize = 64 * 1024;
pub(super) const HNSW_SAVE_LOCK_DIRECTORY: &str = ".frankensearch-hnsw-save-locks";

pub(super) type HnswMetadataPublisher = fn(&Path, &Path, &[u8]) -> SearchResult<()>;

// Keep the classical gamma_k floating-point error model in its well-conditioned
// region. With u = f32::EPSILON / 2 and k = 8n + 32, this cap is exactly the
// largest dimension for which k*u <= 1/4.
pub(super) const DIST_DOT_MAX_DIMENSION: usize = 524_284;

#[derive(Debug, Clone, Copy)]
pub(super) struct DistDotBudget {
    pub(super) radius_squared: f32,
    pub(super) score_tolerance: f32,
}

/// On-disk metadata for the HNSW index.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HnswMeta {
    /// Sidecar format. Absent in legacy v1 metadata (deserializes to 0).
    #[serde(default)]
    pub(super) format_version: u32,
    pub(super) doc_ids: Vec<String>,
    /// Physical source row for each graph origin id, in graph insertion order.
    #[serde(default)]
    pub(super) source_positions: Vec<u32>,
    /// Physical source extent, including retained tombstones.
    #[serde(default)]
    pub(super) source_record_count: usize,
    pub(super) config: HnswConfig,
    pub(super) dimension: usize,
    /// Deterministic fingerprint of the vectors the persisted graph was built
    /// from (FNV-1a 64 over every live f32 vector; see [`super::fingerprint::fingerprint_vectors`]).
    /// Lets native-graph load detect "doc IDs match
    /// but the underlying vectors were silently swapped" and fall back to a
    /// rebuild rather than serve stale ANN hits.
    ///
    /// Absent in legacy metadata (deserializes to 0). Legacy formats rebuild
    /// before the native fast path; current-format sidecars compare 0 like any
    /// other fingerprint and therefore cannot use omission to bypass validation.
    #[serde(default)]
    pub(super) vector_fingerprint: u64,
    /// Directory containing the native graph/data pair, relative to metadata.
    /// Every distinct graph state owns an atomically created generation so
    /// `file_dump` never truncates the pair referenced by installed metadata.
    #[serde(default)]
    pub(super) sidecar_generation: Option<String>,
    /// Basename shared by the native `.hnsw.graph` and `.hnsw.data` files.
    ///
    /// A loaded `hnsw_rs` graph refuses to overwrite an occupied dump and
    /// returns a randomized basename instead. Persisting that returned value
    /// makes the metadata commit point authoritative. Missing location fields
    /// invalidate current-format native loading and force a rebuild.
    #[serde(default)]
    pub(super) sidecar_basename: Option<String>,
    /// Exact identity of the FSVI generation this graph was built from
    /// (`bd-r65a`).
    ///
    /// Absent in every sidecar written before identity binding, and absent
    /// when the source index is a legacy v1 artifact that has no identity to
    /// bind. Absence is never treated as a match: see
    /// [`HnswSourceIdentityV1::admits`].
    #[serde(default)]
    pub(super) source_identity: Option<HnswSourceIdentityV1>,
}

/// The source-generation identity a persisted ANN sidecar is bound to.
///
/// Content equality is NOT identity equality. `vector_fingerprint` and the
/// doc-id sequence prove the graph indexes the same vectors in the same order;
/// they cannot prove those vectors came from the same published generation or
/// the same embedding space. Two FSVI generations can hold byte-identical live
/// content and still be different artifacts — a re-publication carries a new
/// generation nonce, and a model revision bump changes the space fingerprint
/// without necessarily changing a single vector. Serving a sidecar across that
/// boundary is silent reuse of a stale graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HnswSourceIdentityV1 {
    /// Full-width artifact generation of the source FSVI.
    pub(super) generation: ArtifactGenerationIdentityV1,
    /// SHA-256 of the canonical full-width generation identity.
    generation_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the complete embedding identity bundle.
    pub(super) identity_bundle_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the mathematical embedding space.
    pub(super) space_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the physical vector storage identity (format, quantization,
    /// endianness).
    pub(super) storage_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the ordered live document identifiers.
    ordered_live_docset_digest: [u8; SHA256_BYTES],
}

impl HnswSourceIdentityV1 {
    /// Capture the identity of `index`, or `None` for a legacy v1 source that
    /// carries no identity at all.
    pub(super) fn capture(index: &VectorIndex) -> Option<Self> {
        let identity = index.metadata().identity_v2.as_ref()?;
        Some(Self {
            generation: identity.generation,
            generation_fingerprint: identity.generation_fingerprint,
            identity_bundle_fingerprint: identity.identity_bundle_fingerprint,
            space_fingerprint: identity.space_fingerprint,
            storage_fingerprint: identity.storage_fingerprint,
            ordered_live_docset_digest: identity.ordered_live_docset_digest,
        })
    }

    /// Whether a sidecar bound to `persisted` may be served against `live`.
    ///
    /// Both directions of absence fail closed. A sidecar with no recorded
    /// identity cannot be proven to belong to an identity-bearing generation,
    /// and a sidecar that names a generation cannot be served against a source
    /// that cannot name one. The only admissible cases are "both absent"
    /// (legacy v1 on both sides, where there is nothing to bind and the
    /// content checks stand alone) and "both present and equal".
    pub(super) fn admits(persisted: Option<&Self>, live: Option<&Self>) -> bool {
        match (persisted, live) {
            (None, None) => true,
            (Some(persisted), Some(live)) => persisted == live,
            _ => false,
        }
    }
}

/// Durable proof that an immutable native generation finished writing before
/// metadata publication was attempted. A later save can validate and reuse the
/// generation after an interrupted or failed publication without deleting it
/// or dumping another complete copy.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HnswGenerationReceipt {
    pub(super) receipt_version: u32,
    pub(super) metadata_file_name: String,
    pub(super) format_version: u32,
    pub(super) generation: String,
    pub(super) sidecar_basename: String,
    pub(super) doc_count: usize,
    pub(super) doc_ids_fingerprint: u64,
    pub(super) vector_fingerprint: u64,
    pub(super) dimension: usize,
    pub(super) config: HnswConfig,
    /// Binds the origin-to-row map and physical extent used at publication.
    pub(super) source_map_fingerprint: u64,
    pub(super) graph: HnswSidecarDigest,
    pub(super) data: HnswSidecarDigest,
    /// Source FSVI generation identity this generation was dumped from
    /// (`bd-21zyj`). Absent in receipts written before identity binding, and
    /// absent for a legacy v1 source with no identity. Absence never matches
    /// a bound graph: see [`HnswSourceIdentityV1::admits`].
    #[serde(default)]
    pub(super) source_identity: Option<HnswSourceIdentityV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HnswSidecarDigest {
    pub(super) byte_len: u64,
    pub(super) fnv1a64: u64,
}

#[derive(Debug)]
pub(super) struct ValidatedHnswGeneration {
    pub(super) generation: String,
    pub(super) basename: String,
    pub(super) graph: PathBuf,
    /// Identity recorded in the generation's own receipt (`bd-21zyj`).
    pub(super) source_identity: Option<HnswSourceIdentityV1>,
}

/// How an HNSW load obtained its in-memory graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HnswLoadDisposition {
    /// The current native graph/data pair was deserialized from disk.
    Native,
    /// Metadata was readable, but the graph had to be rebuilt from the source index.
    Rebuilt,
}

/// Result of attempting to extend an existing graph without rebuilding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswAppendDisposition {
    /// The graph already belongs to the requested source generation.
    NoOp,
    /// New points were inserted into the existing graph.
    Appended {
        /// Number of new graph points inserted.
        inserted: usize,
    },
    /// The returned graph is unchanged and still belongs to the prior source.
    FullRebuildRequired {
        /// Stable diagnostic explaining why append was refused.
        reason: &'static str,
    },
}

/// Diagnostics for one ANN query.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnSearchStats {
    /// Number of vectors indexed.
    pub index_size: usize,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Effective ef used for this query.
    pub ef_search: usize,
    /// Requested `k`.
    pub k_requested: usize,
    /// Returned result count.
    pub k_returned: usize,
    /// Query latency in microseconds.
    pub search_time_us: u64,
    /// Whether this path is approximate ANN.
    pub is_approximate: bool,
    /// Why the query fell back to an exact scan, if it did.
    pub fallback_reason: Option<AnnFallbackReason>,
    /// Estimated recall@k from the ef/k ratio (see `estimate_recall`).
    ///
    /// This is a heuristic point estimate with NO guarantee. For a certified,
    /// distribution-free recall bound (the automated replacement for a human
    /// recall-budget sign-off), use
    /// [`crate::recall_certificate::conformal_recall_lower_bound`] over a measured
    /// calibration sample instead.
    pub estimated_recall: f64,
    /// Typed classification when the query produced zero hits.
    ///
    /// `Some(reason)` if and only if the returned hit list is empty. Present
    /// so the ANN lane classifies zero-signal states with the same
    /// [`ZeroSignalReason`] vocabulary as the exact lane (bd-tqhc).
    pub zero_signal: Option<ZeroSignalReason>,
}

/// Reason an ANN query returned exact rather than approximate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnFallbackReason {
    /// The native HNSW search returned fewer unique neighbors than exist for
    /// the requested `k`, so an exact scan repaired the result set.
    Underfilled,
    /// The native HNSW search returned zero candidates although the graph
    /// indexes points. Stronger anomaly signal than a partial underfill:
    /// the graph produced no signal at all and an exact scan repaired it.
    EmptyDespiteIndexedPoints,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HnswHitResolution {
    CanonicalPublic,
    RawPhysical,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct HnswTopologyPoint {
    pub(super) origin_id: usize,
    pub(super) point_id: PointId,
    pub(super) neighborhoods: Vec<Vec<Neighbour>>,
}

/// HNSW ANN index with a durable mapping from graph origins to `VectorIndex` rows.
pub struct HnswIndex {
    pub(super) hnsw: Hnsw<'static, f32, DistDot>,
    pub(super) doc_ids: Vec<String>,
    /// Maps compact HNSW origin ids to canonical persisted `VectorIndex` rows.
    pub(super) source_positions: Vec<u32>,
    /// Physical main-slab extent at construction/load time.
    ///
    /// Soft deletion leaves this unchanged and is supported between rebuilds.
    /// A different extent means the borrowed exact-repair source is no longer
    /// the immutable main slab this graph indexes.
    pub(super) source_record_count: usize,
    pub(super) dimension: usize,
    pub(super) config: HnswConfig,
    /// Fingerprint of the vectors the graph was built from. See
    /// [`HnswMeta::vector_fingerprint`].
    pub(super) vector_fingerprint: u64,
    /// Exact identity of the source FSVI generation, when it has one
    /// (`bd-r65a`). `None` for a legacy v1 source with no identity to bind.
    pub(super) source_identity: Option<HnswSourceIdentityV1>,
    /// Whether this graph instance has already warned about an underfill.
    ///
    /// Graph instances are per-generation (rebuilt on reload), so gating the
    /// warning here yields the required once-per-generation bound; repeat
    /// underfills log at debug (bd-tqhc no-warn-storm policy).
    pub(super) underfill_warned: AtomicBool,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("points", &self.hnsw.get_nb_point())
            .field("doc_ids", &self.doc_ids.len())
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
