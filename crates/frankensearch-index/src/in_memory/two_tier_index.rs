use std::path::Path;

use frankensearch_core::{SearchError, SearchResult, VectorHit};

use crate::ValidatedFsviBytes;
use crate::search::{ClassifiedHits, SearchParams};
use crate::simd::dot_product_f16_f32;

use super::InMemoryVectorIndex;

/// In-memory two-tier index wrapping fast and optional quality `InMemoryVectorIndex`.
///
/// Provides the same `search_fast()` / `quality_scores_for_hits()` API as
/// [`crate::TwoTierIndex`] but with fully-resident memory for deterministic latency.
#[derive(Debug, Clone)]
pub struct InMemoryTwoTierIndex {
    pub(super) fast_index: InMemoryVectorIndex,
    pub(super) quality_index: Option<InMemoryVectorIndex>,
}

impl InMemoryTwoTierIndex {
    /// Create from two pre-built in-memory indices.
    #[must_use]
    pub const fn new(
        fast_index: InMemoryVectorIndex,
        quality_index: Option<InMemoryVectorIndex>,
    ) -> Self {
        Self {
            fast_index,
            quality_index,
        }
    }

    /// Construct the shipping in-memory two-tier product from admitted v2
    /// owners and generation-keyed residual-cache directories. The fast tier
    /// and optional quality tier independently attach only their own exact
    /// sidecars; a cache failure leaves that tier's established flat search
    /// active without weakening FSVI admission or cross-tier identity.
    ///
    /// # Errors
    ///
    /// Returns errors while loading the admitted source vectors. Optional
    /// sidecar cache I/O is deliberately contained as per-tier flat fallback.
    pub fn from_admitted_v2_with_residual_sidecar_cache(
        fast_source: &ValidatedFsviBytes,
        fast_cache_dir: &Path,
        quality_source: Option<(&ValidatedFsviBytes, &Path)>,
    ) -> SearchResult<Self> {
        let fast_index = InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(
            fast_source,
            fast_cache_dir,
        )?;
        let quality_index = quality_source
            .map(|(source, cache_dir)| {
                InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(source, cache_dir)
            })
            .transpose()?;
        Ok(Self::new(fast_index, quality_index))
    }

    /// Load from an existing two-tier index directory, reading all data into memory.
    ///
    /// Looks for `vector.fast.idx` (required) and `vector.quality.idx` (optional).
    /// Falls back to `vector.idx` if the fast filename doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns errors from FSVI parsing or vector loading.
    pub fn from_dir(dir: &Path) -> SearchResult<Self> {
        let fast_path = dir.join(crate::two_tier::VECTOR_INDEX_FAST_FILENAME);
        let fast_path = if fast_path.exists() {
            fast_path
        } else {
            let fallback = dir.join(crate::two_tier::VECTOR_INDEX_FALLBACK_FILENAME);
            if !fallback.exists() {
                return Err(SearchError::IndexNotFound { path: fast_path });
            }
            fallback
        };
        let fast_index = InMemoryVectorIndex::from_fsvi(&fast_path)?;

        let quality_path = dir.join(crate::two_tier::VECTOR_INDEX_QUALITY_FILENAME);
        let quality_index = if quality_path.exists() {
            Some(InMemoryVectorIndex::from_fsvi(&quality_path)?)
        } else {
            None
        };

        Ok(Self {
            fast_index,
            quality_index,
        })
    }

    /// Search the fast tier.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`InMemoryVectorIndex::search_top_k`].
    pub fn search_fast(&self, query_vec: &[f32], k: usize) -> SearchResult<Vec<VectorHit>> {
        self.fast_index.search_top_k(query_vec, k, None)
    }

    /// Search the fast tier with typed zero-signal classification.
    ///
    /// Mirrors [`crate::TwoTierIndex::search_fast_classified`] so in-memory
    /// and persistent two-tier paths classify equivalent states identically
    /// (bd-tqhc). The in-memory variant performs no transition logging: its
    /// indexes are ephemeral, and the once-per-generation logging bound is
    /// owned by the persistent path.
    ///
    /// # Errors
    ///
    /// Everything [`Self::search_fast`] returns, plus
    /// [`SearchError::InvalidConfig`] for non-finite query vectors.
    pub fn search_fast_classified(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> SearchResult<ClassifiedHits> {
        self.fast_index.search_top_k_classified(query_vec, k, None)
    }

    /// Search the fast tier with configurable parallelism.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`InMemoryVectorIndex::search_top_k_with_params`].
    pub fn search_fast_with_params(
        &self,
        query_vec: &[f32],
        k: usize,
        params: Option<SearchParams>,
    ) -> SearchResult<Vec<VectorHit>> {
        let params = params.unwrap_or_default();
        self.fast_index
            .search_top_k_with_params(query_vec, k, None, params)
    }

    /// Compute quality-tier scores for fast-index hits.
    ///
    /// Missing quality entries produce `0.0`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if `query_vec` doesn't match
    /// the quality index dimensionality.
    pub fn quality_scores_for_hits(
        &self,
        query_vec: &[f32],
        hits: &[VectorHit],
    ) -> SearchResult<Vec<Option<f32>>> {
        let Some(quality) = &self.quality_index else {
            return Ok(vec![None; hits.len()]);
        };
        if query_vec.len() != quality.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: quality.dimension,
                found: query_vec.len(),
            });
        }
        let mut scores = Vec::with_capacity(hits.len());
        for hit in hits {
            let score = quality
                .index_of_doc_id(&hit.doc_id)
                .map(|idx| dot_product_f16_f32(quality.vector_slice(idx), query_vec))
                .transpose()?;
            scores.push(score);
        }
        Ok(scores)
    }

    /// Whether a quality index is loaded.
    #[must_use]
    pub const fn has_quality_index(&self) -> bool {
        self.quality_index.is_some()
    }

    /// Number of documents in the fast tier.
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.fast_index.record_count()
    }

    /// Iterate over all document IDs in fast-tier order.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = &str> {
        self.fast_index.iter_doc_ids()
    }

    /// Get a reference to the fast index.
    #[must_use]
    pub const fn fast_index(&self) -> &InMemoryVectorIndex {
        &self.fast_index
    }

    /// Get a reference to the quality index (if present).
    #[must_use]
    pub const fn quality_index(&self) -> Option<&InMemoryVectorIndex> {
        self.quality_index.as_ref()
    }

    /// Space fingerprint of the fast tier, when known
    /// (see [`InMemoryVectorIndex::space_fingerprint_hex`]).
    #[must_use]
    pub fn fast_space_fingerprint_hex(&self) -> Option<&str> {
        self.fast_index.space_fingerprint_hex()
    }

    /// Space fingerprint of the quality tier, when known.
    ///
    /// `None` both when no quality index is loaded and when the loaded one
    /// carries no identity; either way there is no quality-tier space to
    /// verify a query embedding against
    /// (see [`InMemoryVectorIndex::space_fingerprint_hex`]).
    #[must_use]
    pub fn quality_space_fingerprint_hex(&self) -> Option<&str> {
        self.quality_index
            .as_ref()
            .and_then(InMemoryVectorIndex::space_fingerprint_hex)
    }

    /// Whether the fast tier's space identity is FSVI-v2-HEADER-attested
    /// rather than declared
    /// (see [`InMemoryVectorIndex::space_identity_is_attested`];
    /// bd-9xuj T2-C4-write, guards 2+8).
    #[must_use]
    pub const fn fast_identity_is_attested(&self) -> bool {
        self.fast_index.space_identity_is_attested()
    }

    /// Whether the quality tier's space identity is FSVI-v2-HEADER-attested.
    ///
    /// `false` both when no quality index is loaded and when the loaded one's
    /// identity was declared rather than read from a validated v2 header
    /// (see [`InMemoryVectorIndex::space_identity_is_attested`];
    /// bd-9xuj T2-C4-write, guards 2+8).
    #[must_use]
    pub fn quality_identity_is_attested(&self) -> bool {
        self.quality_index
            .as_ref()
            .is_some_and(InMemoryVectorIndex::space_identity_is_attested)
    }

    /// Embedder id retained by the fast tier, when its source carried one
    /// (see [`InMemoryVectorIndex::embedder_id`]; bd-9xuj T2-C2).
    #[must_use]
    pub fn fast_embedder_id(&self) -> Option<&str> {
        self.fast_index.embedder_id()
    }

    /// Embedder revision retained by the fast tier, when its source carried
    /// one (see [`InMemoryVectorIndex::embedder_revision`]; bd-9xuj T2-C2).
    #[must_use]
    pub fn fast_embedder_revision(&self) -> Option<&str> {
        self.fast_index.embedder_revision()
    }

    /// Embedder id retained by the quality tier: `None` both when no quality
    /// index is loaded and when the loaded one's source carried no id
    /// (bd-9xuj T2-C2).
    #[must_use]
    pub fn quality_embedder_id(&self) -> Option<&str> {
        self.quality_index
            .as_ref()
            .and_then(InMemoryVectorIndex::embedder_id)
    }

    /// Embedder revision retained by the quality tier, under the same rules
    /// as [`Self::quality_embedder_id`] (bd-9xuj T2-C2).
    #[must_use]
    pub fn quality_embedder_revision(&self) -> Option<&str> {
        self.quality_index
            .as_ref()
            .and_then(InMemoryVectorIndex::embedder_revision)
    }
}
