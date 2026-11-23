//! Optional HNSW approximate nearest-neighbor index (`ann` feature).
//!
//! This module wraps `hnsw_rs` behind a frankensearch-native API.
//!
//! # Persistence
//!
//! The metadata sidecar (e.g. `vector.fast.hnsw`) stores `doc_ids`, config and
//! dimension as JSON. Since format v2 the native `hnsw_rs` graph is also
//! persisted in sidecars beside it, so `load()` deserializes the prebuilt graph
//! directly instead of rebuilding it from vectors. Format v4 fingerprints
//! every live source vector so a stale graph cannot survive an unsampled vector
//! change. Format v5 records the exact
//! generation directory and basename selected during atomic publication, so no
//! save truncates the pair named by installed metadata. A persistent advisory
//! save lock and durable in-generation READY receipt serialize writers, let
//! publication retries reuse complete generations, and reclaim generations
//! superseded by a successful metadata publication.
//! Format v6 attests the native graph's point and layer topology, invalidating
//! graphs produced by `hnsw_rs` versions that could misfile reverse edges.
//! Connectivity is attested as one weak base-layer component; directed
//! coverage from the search entry is measured and warned about but is not an
//! admission condition, because bounded neighbour pruning legitimately orphans
//! in-edges at scale (#32).
//! Legacy sidecars and any load failure fall back to the
//! rebuild-from-`VectorIndex` path.

mod engine;
mod fingerprint;
mod persistence;
mod search;
mod topology;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use persistence::{hnsw_save_lock_artifact_path, materialize_hnsw_save_lock_artifact};
pub(crate) use types::HNSW_META_FORMAT_CURRENT;
pub use types::{
    AnnFallbackReason, AnnSearchStats, HNSW_DEFAULT_EF_CONSTRUCTION, HNSW_DEFAULT_EF_SEARCH,
    HNSW_DEFAULT_M, HNSW_DEFAULT_MAX_LAYER, HnswAppendDisposition, HnswConfig, HnswIndex,
    HnswLoadDisposition,
};

impl HnswIndex {
    /// Return raw physical main-slab candidates for `TwoTierIndex`.
    ///
    /// Document-ID deduplication and WAL supersession are deliberately deferred
    /// until `TwoTierIndex` has ranked the main and resident-WAL physical
    /// candidates together. Exact underfill repair uses the same raw contract.
    pub(crate) fn knn_search_raw_with_stats_against(
        &self,
        source: &crate::VectorIndex,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> frankensearch_core::SearchResult<(Vec<frankensearch_core::VectorHit>, AnnSearchStats)>
    {
        self.knn_search_with_optional_source(
            Some(source),
            query,
            k,
            ef_search,
            types::HnswHitResolution::RawPhysical,
        )
    }
}
