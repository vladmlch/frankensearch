use std::collections::HashMap;
#[cfg(feature = "ann")]
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::Ordering as AtomicOrdering;

use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::{
    BoundQueryEmbedding, SearchError, SearchResult, TieredQueryEmbeddings, TwoTierConfig, VectorHit,
};
use tracing::{debug, info, warn};

use crate::{ClassifiedHits, SearchParams, ValidatedFsviBytes, VectorIndex, dot_product_f32_f32};

use super::activation::{ActivatedTierSearch, activate_tier};
use super::builder::SemanticVectorTier;
use super::{QualityAlignment, TierSource, TwoTierIndex, ZERO_SIGNAL_NONE, zero_signal_code};

impl TwoTierIndex {
    /// Activate typed owner-backed search for one query
    /// (bd-core-vector-space-search-guard-ctzo).
    ///
    /// Every identity join happens HERE, before a single vector byte is
    /// read: requested topology against the tiers actually retained, the
    /// artifact's own canonical identity against the caller's admitted
    /// binding, the complete bd-9xuj admission law (space fingerprints join
    /// AND the producer is identical — a certified-foreign producer is
    /// comparison-grade telemetry and is refused here), storage identity,
    /// dimension, and the generation and live-docset witnesses used for
    /// coverage. The returned [`ActivatedTierSearch`] is the only route to a
    /// vector read, so no search method can run ahead of its validation:
    /// there is nothing to instrument for "zero work before rejection"
    /// because the work is unreachable, not merely unreached.
    ///
    /// The expected identity bundle is the one this index RETAINS from its own
    /// admission ([`Self::fast_admitted_binding`]), never a bundle the caller
    /// supplies. The r1 revision of this method took the bindings as
    /// parameters and defended itself by comparing the caller's canonical
    /// bytes against the artifact's; retaining the binding at admission
    /// removes the parameter instead of checking it, so "the caller presented
    /// an identity this artifact does not carry" stops being a refusal and
    /// becomes unrepresentable. It also removes the plumbing that would
    /// otherwise force every search caller to carry a binding it has no way
    /// to obtain.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] naming the exact tier and
    /// contract field for a topology, identity, producer, dimension or
    /// generation mismatch, and for a tier the query requests that this
    /// index does not retain as an admitted v2 owner.
    pub fn activate_owner_backed_search<'index, 'query>(
        &'index self,
        embeddings: &'query TieredQueryEmbeddings,
    ) -> SearchResult<ActivatedTierSearch<'index, 'query>> {
        let topology = embeddings.supported_topology();
        let fast = match embeddings.fast() {
            Some(query) => Some(activate_tier(
                self.fast_source.admitted_pair(),
                query,
                "fast",
                self.native_fast_hnsw.as_ref().and_then(|policy| {
                    policy
                        .graph
                        .as_ref()
                        .map(|graph| (graph, policy.params.ef_search))
                }),
            )?),
            None => None,
        };
        let quality = match embeddings.quality() {
            Some(query) => Some(activate_tier(
                self.quality_source
                    .as_ref()
                    .and_then(TierSource::admitted_pair),
                query,
                "quality",
                self.native_quality_hnsw.as_ref().and_then(|policy| {
                    policy
                        .graph
                        .as_ref()
                        .map(|graph| (graph, policy.params.ef_search))
                }),
            )?),
            None => None,
        };
        if fast.is_none() && quality.is_none() {
            return Err(SearchError::InvalidConfig {
                field: "search_activation.topology".to_owned(),
                value: format!("{topology:?}"),
                reason: "no tier was requested; vector activation requires at least one \
                         owner-backed tier"
                    .to_owned(),
            });
        }
        Ok(ActivatedTierSearch {
            fast,
            quality,
            topology,
        })
    }

    /// Quality-tier counterpart of [`Self::fast_admitted_owner`].
    ///
    /// `None` both when no quality tier is loaded and when the loaded one is
    /// a plain v1 path-opened tier.
    #[must_use]
    pub fn quality_admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        self.quality_source
            .as_ref()
            .and_then(TierSource::admitted_owner)
    }

    /// Search the fast tier only.
    ///
    /// # Errors
    ///
    /// Propagates errors from source-aware HNSW candidate retrieval (when ANN
    /// is selected) or `VectorIndex::search_top_k` (brute-force fallback).
    pub fn search_fast(&self, query_vec: &[f32], k: usize) -> SearchResult<Vec<VectorHit>> {
        self.search_fast_with_params(query_vec, k, None)
    }

    /// Search the fast tier with optional brute-force parallelism overrides.
    ///
    /// When ANN is active for the fast tier, ANN continues to own candidate
    /// retrieval and `params` is ignored. When brute-force search is used,
    /// `params` controls the Rayon threshold/chunking path.
    ///
    /// # Errors
    ///
    /// Propagates errors from source-aware HNSW candidate retrieval (when ANN
    /// is selected) or `VectorIndex::search_top_k_with_params` / `search_top_k`.
    pub fn search_fast_with_params(
        &self,
        query_vec: &[f32],
        k: usize,
        params: Option<SearchParams>,
    ) -> SearchResult<Vec<VectorHit>> {
        #[cfg(feature = "ann")]
        if let Some(ann) = &self.fast_ann {
            // Source-aware HNSW filters tombstones and exact-repairs native
            // underfill. Request exactly k physical main candidates so
            // duplicate document IDs consume the same pre-resolution slots as
            // canonical VectorIndex search; overfetching would incorrectly
            // backfill lower-ranked public IDs.
            let (hits, stats) = ann.knn_search_raw_with_stats_against(
                self.fast_tier(),
                query_vec,
                k,
                self.config.hnsw_ef_search,
            )?;
            if let Some(reason) = stats.fallback_reason {
                let fallback_count = self
                    .ann_fallback_count
                    .fetch_add(1, AtomicOrdering::Relaxed)
                    .saturating_add(1);
                warn!(
                    ?reason,
                    fallback_count,
                    index_size = stats.index_size,
                    k_requested = stats.k_requested,
                    ef_search = stats.ef_search,
                    search_time_us = stats.search_time_us,
                    "fast-tier ANN degraded to exact search"
                );
            }

            return self.resolve_fast_ann_and_wal(hits, query_vec, k);
        }
        let mrl_config = crate::mrl::MrlConfig {
            search_dims: self.config.mrl_search_dims,
            rescore_dims: 0,
            rescore_top_k: self.config.mrl_rescore_top_k,
        };

        if mrl_config.search_dims > 0 && mrl_config.search_dims < self.fast_tier().dimension() {
            return self.fast_tier().mrl_search(query_vec, k, &mrl_config, None);
        }

        // Default (no explicit exact-scan params): the fast tier is a reranked candidate generator
        // (its hits feed RRF + graph + phase-1 corrections downstream), so use the lossless int8
        // two-pass rather than the exact f16 scan — matching the sync searcher's fast tier
        // (`sync_searcher::search_fast_hits`) and strictly MORE accurate than the ANN path this same
        // method uses when `ann` is enabled. Proven candidate-lossless (`int8_vs_f16_fast_ab`:
        // set-recall@10 = 1.0000, exact-order-match 32/32) and ~1.3-1.43x faster on the fast-tier
        // scan. int8 two-pass itself falls back to the exact scan for F32 slabs, WAL-dirty indexes,
        // or k == 0, so those paths stay bit-identical. Explicit `params` still honour the exact scan
        // + parallelism configuration.
        const FAST_TIER_MULT: usize = 3;
        params.map_or_else(
            || {
                self.fast_tier()
                    .search_top_k_int8_two_pass(query_vec, k, FAST_TIER_MULT)
            },
            |params| {
                self.fast_tier()
                    .search_top_k_with_params(query_vec, k, None, params)
            },
        )
    }

    /// Search the fast tier with typed zero-signal classification.
    ///
    /// Behaves like [`Self::search_fast`] with the fail-closed differences
    /// of the classified exact lane
    /// ([`VectorIndex::search_top_k_classified`]): non-finite queries are
    /// rejected instead of silently scoring garbage, and an empty result
    /// always carries a typed [`ZeroSignalReason`]. Availability transitions
    /// are logged once per state change, never per query.
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
        if query_vec.len() != self.fast_tier().dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.fast_tier().dimension(),
                found: query_vec.len(),
            });
        }
        if k == 0 {
            let classified = ClassifiedHits::empty(ZeroSignalReason::CallerRequestedZeroK);
            self.note_zero_signal(classified.zero_signal);
            return Ok(classified);
        }
        if query_vec.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "query".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "query vector must be finite".to_owned(),
            });
        }
        if query_vec.iter().all(|&value| value == 0.0) {
            let classified = ClassifiedHits::empty(ZeroSignalReason::ZeroNormQuery);
            self.note_zero_signal(classified.zero_signal);
            return Ok(classified);
        }
        let hits = self.search_fast(query_vec, k)?;
        let zero_signal = hits.is_empty().then(|| self.classify_fast_empty());
        self.note_zero_signal(zero_signal);
        Ok(ClassifiedHits { hits, zero_signal })
    }

    /// Classify why a well-formed fast-tier search returned nothing.
    ///
    /// Mirrors [`VectorIndex::classify_empty_result`], with one refinement:
    /// when ANN owns candidate retrieval and the census still shows usable
    /// live vectors, the empty result is the ANN-availability anomaly rather
    /// than a data problem.
    fn classify_fast_empty(&self) -> ZeroSignalReason {
        let state = self.fast_tier().zero_signal_state();
        if let Some(reason) = state.state_reason() {
            return reason;
        }
        if state.is_wal_only() {
            return ZeroSignalReason::WalOnlyNoLiveRecords;
        }
        #[cfg(feature = "ann")]
        if self.fast_ann.is_some() {
            return ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors;
        }
        ZeroSignalReason::NoUsableVectors
    }

    /// Record a zero-signal observation and log state transitions exactly
    /// once.
    ///
    /// Request-scoped reasons (k = 0, filters, query vector defects) are
    /// per-request events: they log at debug and never touch the state
    /// machine, so an interleaved k = 0 query cannot fabricate a recovery
    /// or a re-degradation. State-scoped reasons participate in the
    /// transition machine: availability failures warn once per transition,
    /// benign states log at info once per transition, and the first
    /// hit-producing search after any state-scoped emptiness logs recovery.
    fn note_zero_signal(&self, reason: Option<ZeroSignalReason>) {
        match reason {
            Some(request_scoped) if request_scoped.is_request_scoped() => {
                debug!(
                    reason_code = request_scoped.reason_code(),
                    "fast-tier search returned empty: request-scoped zero-signal"
                );
            }
            Some(state_scoped) => {
                let code = zero_signal_code(state_scoped);
                let previous = self.last_zero_signal.swap(code, AtomicOrdering::Relaxed);
                if previous == code {
                    return;
                }
                if state_scoped.is_availability_failure() {
                    warn!(
                        reason_code = state_scoped.reason_code(),
                        reason = %state_scoped,
                        "fast-tier semantic lane is unusable"
                    );
                } else {
                    info!(
                        reason_code = state_scoped.reason_code(),
                        reason = %state_scoped,
                        "fast-tier semantic lane has no signal"
                    );
                }
            }
            None => {
                let previous = self
                    .last_zero_signal
                    .swap(ZERO_SIGNAL_NONE, AtomicOrdering::Relaxed);
                if previous != ZERO_SIGNAL_NONE {
                    info!("fast-tier semantic lane recovered: search produced hits");
                }
            }
        }
    }

    #[cfg(feature = "ann")]
    fn resolve_fast_ann_and_wal(
        &self,
        ann_hits: Vec<VectorHit>,
        query_vec: &[f32],
        k: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        // HNSW maps compact graph ids to canonical physical VectorIndex rows.
        // Keep that identity: resolving through doc_id would collapse distinct
        // physical rows before the canonical result resolver gets to rank them.
        let base_index = self.fast_tier().record_count();
        let mut hits = Vec::with_capacity(
            ann_hits
                .len()
                .saturating_add(self.fast_tier().wal_entries.len()),
        );
        for hit in ann_hits {
            if let Ok(position) = usize::try_from(hit.index)
                && position < base_index
                && !self.fast_tier().is_deleted(position)
            {
                hits.push(hit);
            }
        }

        // Resident WAL entries are not in the native graph. Preserve canonical
        // `VectorIndex::resolve_sorted_entries` semantics exactly:
        //
        // 1. rank the physical main+WAL candidate pool;
        // 2. retain only the physical top-k;
        // 3. suppress a main hit when any resident WAL version of its doc_id
        //    exists (including a non-finite/corrupt version);
        // 4. deduplicate public doc_ids best-first.
        //
        // Suppression must happen after physical top-k selection. Doing it
        // earlier backfills a lower-ranked hit that canonical exact search
        // deliberately does not return.
        for (wal_index, entry) in self.fast_tier().wal_entries.iter().enumerate() {
            let score = dot_product_f32_f32(&entry.embedding, query_vec)?;
            if !score.is_finite() {
                continue;
            }
            let virtual_index =
                base_index
                    .checked_add(wal_index)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "index".to_owned(),
                        value: wal_index.to_string(),
                        reason: "WAL virtual index overflow".to_owned(),
                    })?;
            let index = u32::try_from(virtual_index).map_err(|_| SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: virtual_index.to_string(),
                reason: "WAL entry index exceeds u32 range".to_owned(),
            })?;
            hits.push(VectorHit {
                index,
                score,
                doc_id: entry.doc_id.as_str().into(),
            });
        }

        let by_score_index = |left: &VectorHit, right: &VectorHit| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        };
        const SELECT_NTH_MIN: usize = 256;
        if k < hits.len() && hits.len() >= SELECT_NTH_MIN {
            hits.select_nth_unstable_by(k, by_score_index);
            hits.truncate(k);
            hits.sort_unstable_by(by_score_index);
        } else {
            hits.sort_by(by_score_index);
            hits.truncate(k);
        }

        let wal_doc_ids: HashSet<&str> = self
            .fast_tier()
            .wal_entries
            .iter()
            .map(|entry| entry.doc_id.as_str())
            .collect();
        let mut seen_doc_ids = HashSet::with_capacity(hits.len());
        hits.retain(|hit| {
            let is_main = usize::try_from(hit.index).is_ok_and(|index| index < base_index);
            if is_main && wal_doc_ids.contains(hit.doc_id.as_str()) {
                return false;
            }
            seen_doc_ids.insert(hit.doc_id.clone())
        });
        Ok(hits)
    }

    /// Retrieve quality candidates from a v1 index bound to its producer.
    ///
    /// The existing revision header must contain the query producer's complete
    /// fingerprint, as written by [`TwoTierIndexBuilder::set_quality_identity`](super::TwoTierIndexBuilder::set_quality_identity).
    /// This compares provenance but does not mint an admitted v2 owner or coverage
    /// witness. Admitted v2 callers use [`Self::activate_owner_backed_search`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the quality tier is absent or
    /// its stored producer differs. Propagates dimension and vector read errors.
    pub fn search_quality_with_producer(
        &self,
        query: &BoundQueryEmbedding,
        k: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        let index = self
            .quality_tier()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "search_activation.quality".to_owned(),
                value: "absent".to_owned(),
                reason: "quality retrieval requires a quality index".to_owned(),
            })?;
        if index.embedder_revision() != query.identity_fingerprint() {
            return Err(SearchError::InvalidConfig {
                field: "search_activation.quality.producer_revision".to_owned(),
                value: index.embedder_revision().to_owned(),
                reason: "rebuild the quality index with the query's complete producer identity"
                    .to_owned(),
            });
        }
        index.search_top_k(query.vector(), k, None)
    }

    /// Compute quality-tier scores for fast-index document positions.
    ///
    /// Missing quality entries produce `None`, allowing downstream blending
    /// to use fast-only scores without penalizing documents that lack quality
    /// vectors.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if `query_vec` does not match
    /// the quality index dimensionality (when a quality index is present), and
    /// propagates decode/corruption errors from the quality index.
    pub fn quality_scores_for_hits(
        &self,
        query_vec: &[f32],
        hits: &[VectorHit],
    ) -> SearchResult<Vec<Option<f32>>> {
        let Some(quality_index) = self.quality_tier() else {
            return Ok(vec![None; hits.len()]);
        };

        if query_vec.len() != quality_index.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: quality_index.dimension(),
                found: query_vec.len(),
            });
        }

        // Precompute `doc_id → latest WAL entry index` once, instead of an O(W)
        // reverse linear scan of the quality WAL **per hit** (was O(hits·W); the
        // exact scan in `search.rs` already precomputes a WAL map). `&str` keys
        // borrow the WAL doc_ids (no clone); forward insert keeps the latest
        // (highest-index) entry, matching the prior rev-scan-first-match. Empty
        // WAL → non-allocating empty map, so the common (compacted) path is neutral.
        let wal_latest: HashMap<&str, usize> = {
            let mut m = HashMap::with_capacity(quality_index.wal_entries.len());
            for (i, entry) in quality_index.wal_entries.iter().enumerate() {
                m.insert(entry.doc_id.as_str(), i);
            }
            m
        };

        let mut scores = Vec::with_capacity(hits.len());
        for hit in hits {
            let mut found_score = if let Some(&i) = wal_latest.get(hit.doc_id.as_str()) {
                Some(dot_product_f32_f32(
                    &quality_index.wal_entries[i].embedding,
                    query_vec,
                )?)
            } else {
                None
            };

            if found_score.is_none() {
                let fast_idx = if hit.index == u32::MAX {
                    self.fast_tier().find_index_by_doc_id(&hit.doc_id)?
                } else if (hit.index as usize) < self.fast_tier().record_count() {
                    Some(hit.index as usize)
                } else {
                    None
                };

                if let Some(idx) = fast_idx {
                    found_score =
                        self.score_quality_for_fast_index(quality_index, query_vec, idx)?;
                }
            }

            if found_score.is_none() {
                if let Some(qual_idx) = quality_index.find_index_by_doc_id(&hit.doc_id)? {
                    found_score = Some(quality_index.dot_query_at(qual_idx, query_vec)?);
                }
            }

            scores.push(found_score);
        }
        Ok(scores)
    }

    /// Returns true when a quality index was loaded.
    #[must_use]
    pub const fn has_quality_index(&self) -> bool {
        self.quality_source.is_some()
    }

    /// Returns true when fast-tier ANN is loaded/enabled.
    #[cfg(feature = "ann")]
    #[must_use]
    pub const fn has_fast_ann(&self) -> bool {
        self.fast_ann.is_some()
    }

    /// Returns true when quality-tier ANN is loaded/enabled.
    #[cfg(feature = "ann")]
    #[must_use]
    pub const fn has_quality_ann(&self) -> bool {
        self.quality_ann.is_some()
    }

    /// Number of fast-tier ANN queries that degraded to an exact scan.
    ///
    /// This monotonic counter makes persistent graph underfill observable even
    /// through search APIs that return only hits.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn ann_fallback_count(&self) -> u64 {
        self.ann_fallback_count.load(AtomicOrdering::Relaxed)
    }

    /// Number of documents in the fast tier (canonical document count).
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.fast_tier().record_count()
    }

    /// Embedder identity recorded in the fast-tier index header.
    #[must_use]
    pub fn fast_embedder_id(&self) -> &str {
        self.fast_tier().embedder_id()
    }

    /// Embedder revision recorded in the fast-tier index header.
    #[must_use]
    pub fn fast_embedder_revision(&self) -> &str {
        self.fast_tier().embedder_revision()
    }

    /// Embedder identity recorded in the quality-tier index header, when loaded.
    #[must_use]
    pub fn quality_embedder_id(&self) -> Option<&str> {
        self.quality_tier().map(VectorIndex::embedder_id)
    }

    /// Embedder revision recorded in the quality-tier index header, when loaded.
    #[must_use]
    pub fn quality_embedder_revision(&self) -> Option<&str> {
        self.quality_tier().map(VectorIndex::embedder_revision)
    }

    /// Lowercase hex SHA-256 fingerprint of the fast tier's embedding space,
    /// when known (bd-9xuj T2-C2).
    ///
    /// This is the index-side join key for
    /// [`frankensearch_core::BoundQueryEmbedding::verify_space_identity`]:
    /// a bound query embedding is admissible against the fast tier exactly
    /// when its space fingerprint equals this value (necessary, never
    /// sufficient for a foreign producer — bundle-holding seams must apply
    /// [`frankensearch_core::BoundQueryEmbedding::verify_producer_conformance`]).
    ///
    /// `Some` when the tier's artifact carries a validated FSVI v2 identity
    /// header, or when this instance was returned by a
    /// [`TwoTierIndexBuilder`](super::TwoTierIndexBuilder) whose caller declared the producing identity
    /// ([`TwoTierIndexBuilder::set_fast_identity`](super::TwoTierIndexBuilder::set_fast_identity)). `None` is the typed
    /// legacy-unidentified state: v1 artifacts persist no space identity, so
    /// a reopen from disk of a v1-built index returns `None` and must be
    /// routed as `LegacyUnidentified` reindex — never admitted on the
    /// id/revision strings or dimension equality.
    #[must_use]
    pub fn fast_space_fingerprint_hex(&self) -> Option<&str> {
        self.fast_space_fingerprint_hex.as_deref()
    }

    /// Quality-tier counterpart of [`Self::fast_space_fingerprint_hex`].
    ///
    /// `None` both when no quality index is loaded and when the loaded one
    /// carries no identity; either way there is no quality-tier space to
    /// verify a query embedding against.
    #[must_use]
    pub fn quality_space_fingerprint_hex(&self) -> Option<&str> {
        self.quality_space_fingerprint_hex.as_deref()
    }

    /// Complete identity bundle of the embedder that produced the fast
    /// tier's vectors, when this instance was returned by a builder that
    /// declared it (bd-9xuj T2-C2; see
    /// [`TwoTierIndexBuilder::set_fast_identity`](super::TwoTierIndexBuilder::set_fast_identity)).
    ///
    /// This is the `expected` side for
    /// [`frankensearch_core::BoundQueryEmbedding::verify_producer_conformance`].
    /// Its storage component describes the producing embedder's output
    /// contract, not this index's persisted encoding. Process-local for v1
    /// artifacts: a reopen from disk has no bundle and returns `None`.
    #[must_use]
    pub const fn fast_declared_identity(&self) -> Option<&EmbeddingIdentityBundleV1> {
        self.fast_declared_identity.as_ref()
    }

    /// Quality-tier counterpart of [`Self::fast_declared_identity`].
    #[must_use]
    pub const fn quality_declared_identity(&self) -> Option<&EmbeddingIdentityBundleV1> {
        self.quality_declared_identity.as_ref()
    }

    /// Whether the fast tier's identity is FSVI-v2-HEADER-attested rather
    /// than builder-time declared (bd-9xuj T2-C4-write, admission guards
    /// 2+8).
    ///
    /// The attested bit derives from WHERE the identity came from, not from
    /// stored state that could drift: a tier's `VectorIndex` carries
    /// `identity_v2()` metadata exactly when its bytes were parsed as a
    /// validated FSVI v2 header inside exact admission
    /// ([`Self::open_admitted_v2_with_paths`] →
    /// [`VectorIndex::open_admitted_v2`]; plain [`VectorIndex::open`] is
    /// strictly v1 and can never produce it). A builder-declared identity
    /// ([`TwoTierIndexBuilder::set_fast_identity`](super::TwoTierIndexBuilder::set_fast_identity)) populates
    /// [`Self::fast_declared_identity`] and the space fingerprint, but never
    /// this bit: a declaration is a claim by the constructing process, an
    /// attestation is read out of the artifact's own bytes. Attested-only
    /// admission seams (the refresh identity-bound merge) must join against
    /// attested identity exclusively and route declared-only or v1 tiers to
    /// the typed legacy refusal.
    #[must_use]
    pub fn fast_identity_is_attested(&self) -> bool {
        self.fast_tier().identity_v2().is_some()
    }

    /// Quality-tier counterpart of [`Self::fast_identity_is_attested`].
    ///
    /// `false` both when no quality index is loaded and when the loaded one
    /// carries no validated v2 identity header.
    #[must_use]
    pub fn quality_identity_is_attested(&self) -> bool {
        self.quality_tier()
            .is_some_and(|index| index.identity_v2().is_some())
    }

    /// Space fingerprint of the tier a semantic vector was served from, when
    /// known (bd-9xuj T2-C2). The typed join between
    /// [`Self::semantic_vector_with_tier_for_doc_id`]'s provenance and the
    /// per-tier space accessors.
    #[must_use]
    pub fn space_fingerprint_hex_for_tier(&self, tier: SemanticVectorTier) -> Option<&str> {
        match tier {
            SemanticVectorTier::Fast => self.fast_space_fingerprint_hex(),
            SemanticVectorTier::Quality => self.quality_space_fingerprint_hex(),
        }
    }

    /// Filesystem path of the loaded fast-tier index artifact.
    ///
    /// For admitted v2 tiers this is the original absolute opener path,
    /// retained as provenance. Queries continue to use the sealed byte owner
    /// if that pathname is later moved or replaced.
    #[must_use]
    pub fn fast_index_path(&self) -> &Path {
        self.fast_source.artifact_path()
    }

    /// Filesystem path of the loaded quality-tier index artifact, when loaded.
    #[must_use]
    pub fn quality_index_path(&self) -> Option<&Path> {
        self.quality_source.as_ref().map(TierSource::artifact_path)
    }

    /// Iterate over all document IDs in fast-tier order.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = SearchResult<String>> + '_ {
        (0..self.doc_count()).map(|i| self.fast_tier().doc_id_at(i).map(ToOwned::to_owned))
    }

    /// Document ID at a given fast-tier index position.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the index is out of bounds or reading fails.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.fast_tier().doc_id_at(index)
    }

    /// Fast-tier index position for a given document id.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index reading fails.
    pub fn fast_index_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<usize>> {
        self.fast_tier().find_index_by_doc_id(doc_id)
    }

    /// Fast-tier vector for the given document id.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn fast_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        let hash = crate::fnv1a_hash(doc_id.as_bytes());
        for entry in self.fast_tier().wal_entries.iter().rev() {
            if entry.doc_id_hash == hash && entry.doc_id == doc_id {
                return Ok(Some(entry.embedding.clone()));
            }
        }

        if let Some(index) = self.fast_tier().find_index_by_doc_id(doc_id)? {
            return self.fast_tier().vector_at_f32(index).map(Some);
        }

        Ok(None)
    }

    /// Quality-tier vector for the given document id when available.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn quality_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        let Some(quality_index) = self.quality_tier() else {
            return Ok(None);
        };

        let hash = crate::fnv1a_hash(doc_id.as_bytes());
        for entry in quality_index.wal_entries.iter().rev() {
            if entry.doc_id_hash == hash && entry.doc_id == doc_id {
                return Ok(Some(entry.embedding.clone()));
            }
        }

        if let Some(fast_index) = self.fast_tier().find_index_by_doc_id(doc_id)? {
            if let Some(quality_index_pos) = self.quality_index_for_fast_index(fast_index) {
                return quality_index.vector_at_f32(quality_index_pos).map(Some);
            }
        }

        if let Some(qual_idx) = quality_index.find_index_by_doc_id(doc_id)? {
            return quality_index.vector_at_f32(qual_idx).map(Some);
        }

        Ok(None)
    }

    /// Semantic vector for the given document id, preferring quality tier.
    ///
    /// Falls back to the fast-tier vector when the quality tier is unavailable
    /// or missing for this document.
    ///
    /// The fallback is SILENT here: the returned vector may live in the fast
    /// tier's embedding space rather than the quality tier's, and this
    /// signature cannot say which. Callers that must know — any caller that
    /// compares the vector against other vectors — should use
    /// [`Self::semantic_vector_with_tier_for_doc_id`], which returns the
    /// serving tier alongside the vector (bd-9xuj T2-C2).
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn semantic_vector_for_doc_id(&self, doc_id: &str) -> SearchResult<Option<Vec<f32>>> {
        Ok(self
            .semantic_vector_with_tier_for_doc_id(doc_id)?
            .map(|(_, vector)| vector))
    }

    /// Semantic vector for the given document id with typed tier provenance
    /// (bd-9xuj T2-C2).
    ///
    /// Identical lookup order and results as
    /// [`Self::semantic_vector_for_doc_id`] — quality tier first, silent
    /// fast-tier fallback — but the returned [`SemanticVectorTier`] names the
    /// tier that served the vector, so a caller can resolve which embedding
    /// space it lives in via [`Self::space_fingerprint_hex_for_tier`] instead
    /// of comparing vectors across spaces unknowingly. Changing the fallback
    /// itself is out of scope here (C4); this accessor only makes it
    /// observable.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if index access fails.
    pub fn semantic_vector_with_tier_for_doc_id(
        &self,
        doc_id: &str,
    ) -> SearchResult<Option<(SemanticVectorTier, Vec<f32>)>> {
        if let Some(quality) = self.quality_vector_for_doc_id(doc_id)? {
            return Ok(Some((SemanticVectorTier::Quality, quality)));
        }
        Ok(self
            .fast_vector_for_doc_id(doc_id)?
            .map(|vector| (SemanticVectorTier::Fast, vector)))
    }

    /// Whether the fast-tier document at `index` has a quality-tier vector.
    #[must_use]
    pub fn has_quality_for_index(&self, index: usize) -> bool {
        if index >= self.doc_count() {
            return false;
        }
        match &self.quality_alignment {
            QualityAlignment::None => false,

            QualityAlignment::Aligned => true,

            QualityAlignment::Mapping(map) => map.get(index).copied().flatten().is_some(),
        }
    }

    /// Accessor for the configuration used to open this index.
    #[must_use]
    pub const fn config(&self) -> &TwoTierConfig {
        &self.config
    }

    fn score_quality_for_fast_index(
        &self,

        quality_index: &VectorIndex,

        query_vec: &[f32],

        fast_idx: usize,
    ) -> SearchResult<Option<f32>> {
        if fast_idx >= self.doc_count() {
            return Ok(None);
        }
        let quality_idx = match &self.quality_alignment {
            QualityAlignment::None => return Ok(None),

            QualityAlignment::Aligned => fast_idx,

            QualityAlignment::Mapping(map) => match map.get(fast_idx).copied().flatten() {
                Some(idx) => idx,

                None => return Ok(None),
            },
        };

        // Fused byte-based dot (no per-hit `Vec<f32>` decode), matching the
        // brute-force scan's scorer; bit-identical for `dim % 32 == 0`.
        quality_index.dot_query_at(quality_idx, query_vec).map(Some)
    }

    fn quality_index_for_fast_index(&self, fast_idx: usize) -> Option<usize> {
        match &self.quality_alignment {
            QualityAlignment::None => None,
            QualityAlignment::Aligned => Some(fast_idx),
            QualityAlignment::Mapping(map) => map.get(fast_idx).copied().flatten(),
        }
    }
}
