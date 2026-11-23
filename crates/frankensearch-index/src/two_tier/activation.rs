use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use frankensearch_core::{
    BoundQueryEmbedding, RetrievalTopology, SearchCoverageV1, SearchError, SearchResult,
    SpaceIdentityAdmission, TierQueryCoverageV1, VectorHit,
};
use tracing::warn;

use crate::native_hnsw::ValidatedNativeHnsw;
use crate::{
    FsviAdmissionError, FsviV2IdentityBinding, SearchParams, ValidatedFsviBytes, VectorIndex,
};

/// Admit one identity-complete FSVI v2 tier through the only legitimate v2
/// open path and hand back the SEALED OWNER, whole (bd-9xuj T2-C4-write r2).
///
/// [`VectorIndex::open_admitted_v2`] copies the artifact once into a sealed
/// byte owner, verifies the complete identity/content bindings against
/// `binding`, and rejects any WAL directory entry. The r1 revision of this
/// helper moved `validated.index` out and DROPPED the owner — discarding the
/// `ValidatedFsviBytes` capability, the complete witness, and the
/// publication state, contradicting the owner contract on
/// [`ValidatedFsviBytes`] ("no conversion into a mutable/path-opened
/// `VectorIndex`"). The r2 repair retains the owner in full; callers wrap it
/// in [`TierSource::AdmittedV2`](super::TierSource::AdmittedV2) and borrow the validated index from inside
/// it, so the parsed v2 identity metadata (`identity_v2()` stays `Some`)
/// still marks the tier ATTESTED on the assembled [`TwoTierIndex`](super::TwoTierIndex).
/// One tier that passed every identity join, paired with the query bound to
/// it (bd-ctzo). Holding this is the proof the join happened.
#[derive(Debug)]
pub struct ActivatedTier<'index, 'query> {
    owner: &'index ValidatedFsviBytes,
    query: &'query BoundQueryEmbedding,
    tier: &'static str,
    native_hnsw: Option<(&'index ValidatedNativeHnsw, usize)>,
}

impl ActivatedTier<'_, '_> {
    /// Exact top-k over the retained owner's own bytes.
    ///
    /// # Errors
    ///
    /// Propagates typed row/vector decode failures from the owner.
    pub fn search_top_k(&self, k: usize) -> SearchResult<Vec<VectorHit>> {
        self.owner.search_top_k(self.query.vector(), k, None)
    }

    /// Exact top-k with explicit parallelism settings over the retained owner.
    ///
    /// This bypasses all candidate generators, including native and legacy
    /// ANN and dimension truncation, while retaining the query's admission.
    ///
    /// # Errors
    /// Propagates typed row/vector decode failures from the owner.
    pub fn search_top_k_with_params(
        &self,
        k: usize,
        params: SearchParams,
    ) -> SearchResult<Vec<VectorHit>> {
        self.owner
            .index
            .search_top_k_with_params(self.query.vector(), k, None, params)
    }

    /// Retrieve with this activation's configured backend. Exact rescoring
    /// reads the same retained allocation and uses the exact lane's scorer.
    fn search_candidates(&self, k: usize) -> SearchResult<Vec<VectorHit>> {
        let Some((graph, ef_search)) = self.native_hnsw else {
            return self.search_top_k(k);
        };
        let target = k.min(self.owner.live_count());
        if target == 0 {
            return Ok(Vec::new());
        }
        let width = target.max(ef_search).min(self.owner.record_count());
        let candidates = graph.search(self.query.vector(), width, Some(width))?;
        let mut hits = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let index = candidate.physical_row();
            let row = usize::try_from(index).map_err(|_| SearchError::InvalidConfig {
                field: format!("search_activation.{}.physical_row", self.tier),
                value: index.to_string(),
                reason: "native graph row does not fit usize".to_owned(),
            })?;
            let score = self.owner.index.dot_query_at(row, self.query.vector())?;
            if !score.is_finite() {
                return Err(SearchError::InvalidConfig {
                    field: format!("search_activation.{}.score", self.tier),
                    value: "non-finite".to_owned(),
                    reason: "native candidate rescoring must produce a finite score".to_owned(),
                });
            }
            hits.push(VectorHit {
                index,
                score,
                doc_id: candidate.doc_id().into(),
            });
        }
        // The exact VectorIndex lane resolves equal scores by physical row,
        // whose FSVI order is doc-id hash order, not lexical doc-id order.
        hits.sort_unstable_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        });
        hits.truncate(target);
        Ok(hits)
    }

    /// The retained owner this tier serves from.
    #[must_use]
    pub const fn owner(&self) -> &ValidatedFsviBytes {
        self.owner
    }

    /// Generation sequence witnessed by the retained owner — coverage is
    /// reconstructed from this, never from a caller-supplied scalar.
    #[must_use]
    pub const fn generation_sequence(&self) -> u64 {
        self.owner.witness().generation.sequence
    }

    /// Live document count witnessed by the retained owner.
    #[must_use]
    pub const fn live_count(&self) -> u64 {
        self.owner.witness().live_count
    }
}

/// A validated activation over one or both tiers (bd-ctzo).
///
/// Every method here is reachable only after
/// [`TwoTierIndex::activate_owner_backed_search`](super::TwoTierIndex::activate_owner_backed_search) completed all joins, which
/// is what makes "no vector read before validation" a structural property
/// rather than an assertion.
#[derive(Debug)]
pub struct ActivatedTierSearch<'index, 'query> {
    pub(super) fast: Option<ActivatedTier<'index, 'query>>,
    pub(super) quality: Option<ActivatedTier<'index, 'query>>,
    pub(super) topology: RetrievalTopology,
}

impl<'index, 'query> ActivatedTierSearch<'index, 'query> {
    /// Topology the validated query bindings actually support.
    #[must_use]
    pub const fn topology(&self) -> RetrievalTopology {
        self.topology
    }

    /// The activated fast tier, when the query bound one.
    #[must_use]
    pub const fn fast(&self) -> Option<&ActivatedTier<'index, 'query>> {
        self.fast.as_ref()
    }

    /// The activated quality tier, when the query bound one.
    #[must_use]
    pub const fn quality(&self) -> Option<&ActivatedTier<'index, 'query>> {
        self.quality.as_ref()
    }

    /// Retrieve directly from the QUALITY owner with the quality-bound query.
    ///
    /// This is a first-class operation, not a rescoring of a fast-selected
    /// pool: a document absent from the fast tier's top-k is still reachable
    /// here, which is the behaviour the fast-pool-rescoring anti-pattern
    /// silently loses.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when no quality tier was
    /// activated, and propagates typed decode failures.
    pub fn search_quality(&self, k: usize) -> SearchResult<Vec<VectorHit>> {
        self.quality
            .as_ref()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "search_activation.quality".to_owned(),
                value: format!("{:?}", self.topology),
                reason: "quality retrieval requires an activated quality tier".to_owned(),
            })?
            .search_candidates(k)
    }

    /// Retrieve directly from the FAST owner with the fast-bound query.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when no fast tier was
    /// activated, and propagates typed decode failures.
    pub fn search_fast(&self, k: usize) -> SearchResult<Vec<VectorHit>> {
        self.fast
            .as_ref()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "search_activation.fast".to_owned(),
                value: format!("{:?}", self.topology),
                reason: "fast retrieval requires an activated fast tier".to_owned(),
            })?
            .search_candidates(k)
    }

    /// Reconstruct this query's per-tier coverage from the retained owner
    /// witnesses and the candidates actually returned (bd-ctzo C4).
    ///
    /// `returned` is the result set the caller is about to serve. Each tier's
    /// contribution is COUNTED against it by canonical `doc_id` — not predicted
    /// from the requested `k`, not taken from a caller-supplied scalar, and
    /// not inferred from which tiers were activated. A tier that was
    /// activated but whose documents all lost the blend contributes zero, and
    /// that zero is a measurement.
    ///
    /// A tier the query did not bind reports `NotRequested`; there is no
    /// variant that reports an absent tier as covering nothing, because
    /// "not asked" and "asked and found nothing" are different facts.
    ///
    /// # Errors
    ///
    /// Propagates typed decode failures from re-reading each activated tier's
    /// own top-`k` doc ids, which is how contribution is attributed.
    pub fn coverage(&self, returned: &[VectorHit], k: usize) -> SearchResult<SearchCoverageV1> {
        let fast = Self::tier_coverage(self.fast.as_ref(), returned, k)?;
        let quality = Self::tier_coverage(self.quality.as_ref(), returned, k)?;
        Ok(SearchCoverageV1::new(self.topology, fast, quality))
    }

    /// Coverage for one tier: the owner's own witness plus a counted
    /// contribution.
    fn tier_coverage(
        tier: Option<&ActivatedTier<'index, 'query>>,
        returned: &[VectorHit],
        k: usize,
    ) -> SearchResult<TierQueryCoverageV1> {
        let Some(tier) = tier else {
            return Ok(TierQueryCoverageV1::NotRequested);
        };
        let served: BTreeSet<String> = tier
            .search_candidates(k)?
            .into_iter()
            .map(|hit| hit.doc_id.to_string())
            .collect();
        let contributed = returned
            .iter()
            .filter(|hit| served.contains(hit.doc_id.as_str()))
            .count();
        Ok(TierQueryCoverageV1::Witnessed {
            generation_sequence: tier.generation_sequence(),
            live_count: tier.live_count(),
            contributed_candidates: contributed as u64,
        })
    }

    /// Independent per-tier retrieval unioned by canonical document identity
    /// (bd-ctzo C2 `FullProgressive`).
    ///
    /// Each tier is searched with ITS OWN bound query against ITS OWN owner;
    /// neither pool constrains the other. The union is deterministic:
    /// documents are keyed by canonical `doc_id`, the better score wins a
    /// duplicate, and ties break by `doc_id` so the order never depends on
    /// which tier happened to return first.
    ///
    /// # Errors
    ///
    /// Propagates typed decode failures from either tier.
    pub fn search_union(&self, k: usize) -> SearchResult<Vec<VectorHit>> {
        let mut merged: BTreeMap<String, VectorHit> = BTreeMap::new();
        for tier in [self.fast.as_ref(), self.quality.as_ref()]
            .into_iter()
            .flatten()
        {
            for hit in tier.search_candidates(k)? {
                merged
                    .entry(hit.doc_id.to_string())
                    .and_modify(|existing| {
                        if hit.score > existing.score {
                            *existing = hit.clone();
                        }
                    })
                    .or_insert(hit);
            }
        }
        let mut union: Vec<VectorHit> = merged.into_values().collect();
        union.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
        union.truncate(k);
        Ok(union)
    }
}

/// Perform every identity join for one tier before any vector read (bd-ctzo).
///
/// `admitted` is the tier's indivisible admission product — the retained
/// owner together with the binding it was admitted under. Admission already
/// proved those two agree byte-for-byte on canonical identity, storage and
/// generation ([`validate_expected_v2_binding`]), so this function does not
/// re-derive the artifact's own coherence. What it establishes is the join
/// that admission could not know about: whether THIS QUERY belongs to that
/// artifact's space and producer, and whether its width matches.
///
/// [`validate_expected_v2_binding`]: crate::VectorIndex::open_admitted_v2
pub(super) fn activate_tier<'index, 'query>(
    admitted: Option<(&'index ValidatedFsviBytes, &FsviV2IdentityBinding)>,
    query: &'query BoundQueryEmbedding,
    tier: &'static str,
    native_hnsw: Option<(&'index ValidatedNativeHnsw, usize)>,
) -> SearchResult<ActivatedTier<'index, 'query>> {
    let (owner, binding) = admitted.ok_or_else(|| SearchError::InvalidConfig {
        field: format!("search_activation.{tier}.owner"),
        value: "<absent>".to_owned(),
        reason: format!(
            "the query binds a {tier} embedding but this index retains no admitted FSVI v2 \
             {tier} owner; a legacy or path-opened tier is not searchable under typed \
             activation and requires a reindex"
        ),
    })?;
    // The complete admission law. A certified-foreign producer is
    // comparison-grade telemetry only, so it is refused rather than admitted.
    match query.verify_producer_conformance(&binding.frozen_identity().identity, tier)? {
        SpaceIdentityAdmission::SameProducer => {}
        // Anything that is not the same attested producer -- today only
        // ConformanceCompatibleProducer -- is comparison-grade telemetry, and
        // a future variant must default to refusal rather than inherit
        // admission by being added.
        other => {
            return Err(SearchError::InvalidConfig {
                field: format!("search_activation.{tier}.producer_conformance"),
                value: format!("{other:?}"),
                reason: format!(
                    "the query's producer is not the {tier} index's attested producer; a \
                     certified-compatible pairing is comparison-grade telemetry and never an \
                     admission basis"
                ),
            });
        }
    }
    // A redundant cross-check, kept deliberately and labelled as redundant:
    // the space join above already implies this width equality through
    // `space.dimension == storage.dimension` (bundle validation),
    // `storage.dimension == metadata.dimension` (admission), and
    // `vector.len() == space.dimension` (`BoundQueryEmbedding::new`). Unlike
    // the generation comparison removed below, this one guards the SHAPE OF
    // THE READ that happens two lines later, so it is worth an O(1) test
    // immediately before the bytes are touched rather than a proof by
    // transitivity across three modules.
    if query.vector().len() != owner.dimension() {
        return Err(SearchError::InvalidConfig {
            field: format!("search_activation.{tier}.dimension"),
            value: query.vector().len().to_string(),
            reason: format!(
                "query vector width does not match the {tier} index dimension {}",
                owner.dimension()
            ),
        });
    }
    // No generation join here, deliberately. `binding.generation()` and
    // `owner.witness().generation` are the same admission fact read twice:
    // admission refuses an artifact whose persisted generation and generation
    // fingerprint are not byte-equal to the binding's, and the witness copies
    // its generation straight out of the artifact's validated header. When the
    // binding was a caller parameter that comparison could fail and was worth
    // making; now that owner and binding are installed as one product it can
    // only ever be true, and an error arm no test can reach is not a guard.
    // The generation a caller CAN get wrong is the one it asks the index to
    // install, and `try_replace_admitted_v2` refuses that before any state
    // change. Coverage still reports the generation from the owner witness
    // (`ActivatedTier::generation_sequence`), never from a caller.
    Ok(ActivatedTier {
        owner,
        query,
        tier,
        native_hnsw,
    })
}

pub(super) fn admit_v2_tier(
    path: &Path,
    binding: &FsviV2IdentityBinding,
    tier: &str,
) -> SearchResult<ValidatedFsviBytes> {
    VectorIndex::open_admitted_v2(path, binding)
        .map_err(|error| admission_error_to_search_error(error, tier, path))
}

/// Map a typed [`FsviAdmissionError`] into the [`SearchError`] surface,
/// naming the tier so refusals stay attributable. I/O and corruption pass
/// through unchanged; reindex/upgrade/snapshot outcomes become typed
/// `InvalidConfig` refusals rather than being flattened into strings at the
/// caller.
fn admission_error_to_search_error(
    error: FsviAdmissionError,
    tier: &str,
    path: &Path,
) -> SearchError {
    match error {
        FsviAdmissionError::Index(error) => error,
        other => SearchError::InvalidConfig {
            field: format!("two_tier.{tier}_v2_admission"),
            value: path.display().to_string(),
            reason: other.to_string(),
        },
    }
}

/// Loudly surface v1 WAL replay at two-tier reopen (bd-9xuj C4-write r3,
/// fault test iv — a generation-collision sidecar must never be treated as
/// active rows SILENTLY).
///
/// In the v1 format a WAL whose generation equals `next_generation(main)`
/// is byte-indistinguishable from legitimate incremental appends — and that
/// includes a FOREIGN sidecar left beside a freshly republished main by a
/// pre-r3 crash: fresh legacy mains start at compaction generation 1, so a
/// leftover generation-2 sidecar collides exactly.
/// `TwoTierIndexBuilder::finish` now publishes each tier as an
/// authority-carrying successor generation (`publish_tier`, bd-zhjv8), so
/// the rename itself invalidates the destination's sidecar and finish can
/// no longer manufacture that state; this warning keeps
/// any residual legacy state (pre-r3 crash artifacts, external
/// manipulation) from being replayed silently. Read-side classification
/// cannot refuse here without breaking the legitimate v1
/// incremental-append reopen contract, which external `VectorIndex`
/// consumers rely on.
pub(super) fn warn_if_wal_rows_replayed(tier: &str, path: &Path, index: &VectorIndex) {
    let replayed = index.wal_record_count();
    if replayed > 0 {
        warn!(
            tier,
            path = %path.display(),
            replayed_wal_records = replayed,
            "two-tier v1 open replayed WAL sidecar rows; if this main was \
             just republished, the rows may belong to a replaced foreign \
             generation (generation counters wrap mod 255)"
        );
    }
}
