use std::fmt;

use serde::{Deserialize, Serialize};

use crate::SearchError;
use crate::error::SearchResult;
use crate::generation::EmbeddingIdentityBundleV1;
use crate::traits::IdentityBoundEmbedding;

use super::RetrievalTopology;
use super::coverage_scope::{CoverageUnknownReasonV1, TIER_COVERAGE_SCHEMA_VERSION_V1};

// ---------------------------------------------------------------------------
// Typed per-tier query embeddings (bd-9xuj)
// ---------------------------------------------------------------------------

/// A query embedding bound to the complete identity of the embedding space
/// that produced it.
///
/// Raw `&[f32]` crossing an API boundary is how one vector gets applied to
/// multiple spaces: same dimensions make the bug silently plausible, and
/// different dimensions merely error. A bound embedding carries its
/// [`EmbeddingIdentityBundleV1`], so every consumer can verify the vector
/// belongs to the space it is about to search — by fingerprint, not by
/// dimension coincidence.
#[derive(Clone)]
pub struct BoundQueryEmbedding {
    vector: Vec<f32>,
    pub(super) identity: EmbeddingIdentityBundleV1,
    identity_fingerprint: String,
    space_fingerprint: String,
}

impl fmt::Debug for BoundQueryEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundQueryEmbedding")
            .field("dimension", &self.vector.len())
            .field("identity_fingerprint", &self.identity_fingerprint)
            .field("space_fingerprint", &self.space_fingerprint)
            .finish_non_exhaustive()
    }
}

impl BoundQueryEmbedding {
    /// Bind a vector to the identity of the space that produced it.
    ///
    /// The identity bundle is validated before anything is bound
    /// ([`EmbeddingIdentityBundleV1::validate`]): a bound embedding is a
    /// claim every downstream verifier trusts, so an incoherent bundle —
    /// component fingerprints that do not bind each other, dimension
    /// contradictions — must be rejected here, not discovered at a seam.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the identity bundle is
    /// malformed or an in-process `Vec<f32>` is paired with a
    /// non-native/non-f32 storage identity. Returns
    /// [`SearchError::DimensionMismatch`] when the vector length disagrees
    /// with its validated mathematical space.
    pub fn new(vector: Vec<f32>, identity: EmbeddingIdentityBundleV1) -> SearchResult<Self> {
        identity.validate()?;
        let declared_dimension =
            usize::try_from(identity.space.dimension).map_err(|_| SearchError::InvalidConfig {
                field: "bound_query_embedding.dimension".to_owned(),
                value: identity.space.dimension.to_string(),
                reason: "dimension does not fit usize".to_owned(),
            })?;
        if vector.len() != declared_dimension {
            return Err(SearchError::DimensionMismatch {
                expected: declared_dimension,
                found: vector.len(),
            });
        }
        let validated = IdentityBoundEmbedding {
            values: vector,
            identity,
        };
        validated.validate()?;
        let IdentityBoundEmbedding {
            values: vector,
            identity,
        } = validated;
        let identity_fingerprint = identity.fingerprint();
        let space_fingerprint = identity.space.fingerprint();
        Ok(Self {
            vector,
            identity,
            identity_fingerprint,
            space_fingerprint,
        })
    }

    /// The query vector.
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// The complete identity of the producing space.
    #[must_use]
    pub const fn identity(&self) -> &EmbeddingIdentityBundleV1 {
        &self.identity
    }

    /// Lowercase SHA-256 fingerprint of the full identity bundle, computed
    /// once at bind time.
    #[must_use]
    pub fn identity_fingerprint(&self) -> &str {
        &self.identity_fingerprint
    }

    /// Lowercase SHA-256 fingerprint of the *space* component only
    /// ([`EmbeddingIdentityBundleV1::space`]), computed once at bind time.
    ///
    /// This is the join key at every index seam: a query-side bundle binds
    /// in-memory `f32` storage while an index-side bundle binds its
    /// persisted storage format (for example `fsvi-v2`), so their
    /// full-bundle fingerprints legitimately differ even when both were
    /// produced by the same model in the same mathematical space.
    #[must_use]
    pub fn space_fingerprint(&self) -> &str {
        &self.space_fingerprint
    }

    /// Verify this embedding's complete identity bundle — space, producer,
    /// input contract, *and* storage — matches what a consumer expects.
    ///
    /// This is an embedder-to-embedder comparison: use it only when both
    /// sides bind the same storage identity. At an index seam the storage
    /// components legitimately differ (query-side `in-memory-*` versus the
    /// index's persisted format), so full-bundle fingerprints can never
    /// match there — join on [`Self::verify_space_identity`] and admit via
    /// [`Self::verify_producer_conformance`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] naming `tier` when the
    /// fingerprints differ — including the same-dimension wrong-space case
    /// that raw vector APIs silently accept.
    pub fn verify_space(&self, expected_fingerprint: &str, tier: &str) -> SearchResult<()> {
        if self.identity_fingerprint == expected_fingerprint {
            return Ok(());
        }
        Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.identity"),
            value: self.identity_fingerprint.clone(),
            reason: format!(
                "query embedding was produced in a different embedding space than the \
                 {tier} index expects (expected identity fingerprint {expected_fingerprint})"
            ),
        })
    }

    /// Join this embedding against an index-side *space* fingerprint — the
    /// index-seam identity check (bd-9xuj T2-C1).
    ///
    /// At an index seam the two sides legitimately disagree on storage
    /// (see [`Self::verify_space`]), so the join key is the space
    /// fingerprint ([`EmbeddingSpaceIdentityV1::fingerprint`]), which binds
    /// model, revision, dimension, and the input contract via
    /// [`EmbeddingSpaceIdentityV1::input_contract_fingerprint`] — but not
    /// the physical storage.
    ///
    /// A matching space fingerprint is **necessary, not sufficient**, for
    /// admission: a bare fingerprint carries no producer attestation, so
    /// this check cannot certify that a *different producer's* vectors are
    /// interchangeable with the index's. Seams that retain the full
    /// expected identity bundle must call
    /// [`Self::verify_producer_conformance`], which applies the complete
    /// bd-9xuj admission law (same producer, or a conformance-certified
    /// foreign producer; anything else rejects).
    ///
    /// # The `LegacyUnidentified` boundary
    ///
    /// `expected_space_fingerprint` must come from an identity-bearing
    /// source: an FSVI v2 header written by `VectorIndex::create_v2`, or an
    /// explicitly supplied space identity. The only production writers that
    /// call `create_v2` today are the two staging call sites in the fusion
    /// crate's `RefreshWorker::stage_identity_bound_generation` (one per
    /// tier), whose output is staged and non-canonical until publication —
    /// `TwoTierIndexBuilder::finish` still routes through the legacy v1
    /// constructors with `identity_v2: None` — so every *canonical*
    /// production index on disk is v1 and has *no* space fingerprint to
    /// pass here. Those artifacts must never reach this verifier: the seam
    /// routes them as typed `LegacyUnidentified`
    /// (`FsviReindexReason::LegacyUnidentified` → `RecoveryPlan` reindex,
    /// then FSVI v2 activation) rather than fabricating a fingerprint,
    /// admitting on dimension equality, or warning and proceeding.
    ///
    /// [`EmbeddingSpaceIdentityV1::fingerprint`]: crate::generation::EmbeddingSpaceIdentityV1::fingerprint
    /// [`EmbeddingSpaceIdentityV1::input_contract_fingerprint`]: crate::generation::EmbeddingSpaceIdentityV1::input_contract_fingerprint
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] naming `tier` when the space
    /// fingerprints differ — including the same-dimension wrong-space case
    /// that raw vector APIs silently accept.
    pub fn verify_space_identity(
        &self,
        expected_space_fingerprint: &str,
        tier: &str,
    ) -> SearchResult<()> {
        if self.space_fingerprint == expected_space_fingerprint {
            return Ok(());
        }
        Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.space_identity"),
            value: self.space_fingerprint.clone(),
            reason: format!(
                "query embedding was produced in a different embedding space than the \
                 {tier} index expects (expected space fingerprint {expected_space_fingerprint})"
            ),
        })
    }

    /// Apply the complete bd-9xuj admission law against a fully known
    /// expected identity bundle.
    ///
    /// Admission requires the space fingerprints to join (exactly as
    /// [`Self::verify_space_identity`]) **and** producer conformance:
    ///
    /// - the same attested producer →
    ///   [`SpaceIdentityAdmission::SameProducer`];
    /// - a different producer → admitted only when both producers carry the
    ///   byte-identical pinned golden-vector certificate
    ///   ([`crate::generation::GoldenVectorCertificateV1`]), returning
    ///   [`SpaceIdentityAdmission::ConformanceCompatibleProducer`] — typed
    ///   telemetry the caller is expected to record;
    /// - otherwise the pairing is rejected. A matching space fingerprint
    ///   alone never admits a foreign producer.
    ///
    /// # `ConformanceCompatibleProducer` is comparison-grade, NOT trust
    ///
    /// This is a *pairwise* check: it establishes only that the two bundles
    /// AGREE with each other. Mutual agreement does not bind either producer
    /// to any trusted fixture — two bundles carrying the same wrong (or
    /// fabricated) certificate bytes still "agree". Verification against a
    /// pinned trusted corpus is the trunk's certificate flow
    /// ([`crate::generation::GoldenVectorCertificateV1::verify_exact_f32`]
    /// and the witness flow in `generation.rs`), which this method deliberately
    /// does not perform. Callers must therefore treat
    /// `ConformanceCompatibleProducer` as comparison-grade telemetry and
    /// never as an admission basis — the one production caller
    /// (`require_same_producer` in the fusion crate's refresh path) logs the
    /// outcome and refuses it.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`]: propagated from
    /// [`EmbeddingIdentityBundleV1::validate`] when `expected` is not itself
    /// a coherent bundle; naming `query_embedding.<tier>.space_identity`
    /// when the space fingerprints differ; naming
    /// `query_embedding.<tier>.producer_conformance` when the spaces join
    /// but the foreign producer carries no matching golden-vector
    /// certificate.
    pub fn verify_producer_conformance(
        &self,
        expected: &EmbeddingIdentityBundleV1,
        tier: &str,
    ) -> SearchResult<SpaceIdentityAdmission> {
        expected.validate()?;
        self.verify_space_identity(&expected.space.fingerprint(), tier)?;
        let query_producer_fingerprint = self.identity.producer.fingerprint();
        let expected_producer_fingerprint = expected.producer.fingerprint();
        if query_producer_fingerprint == expected_producer_fingerprint {
            return Ok(SpaceIdentityAdmission::SameProducer);
        }
        // Pairwise certified-conformance shape (the bd-9xuj r3
        // `is_conformance_compatible_with` law, inlined against the trunk's
        // generation API, which superseded the pairwise method with the
        // witness/certificate flow): both bundles validate (self at bind
        // time, `expected` above), the spaces join (established above), and
        // both producers carry the byte-identical pinned golden-vector
        // certificate.
        if self.identity.producer.golden_vectors == expected.producer.golden_vectors {
            return Ok(SpaceIdentityAdmission::ConformanceCompatibleProducer {
                query_producer_fingerprint,
                expected_producer_fingerprint,
            });
        }
        Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.producer_conformance"),
            value: query_producer_fingerprint,
            reason: format!(
                "query embedding's producer shares the {tier} index's embedding space but \
                 is not certified conformance-compatible with its producer (expected \
                 producer fingerprint {expected_producer_fingerprint}); a matching space \
                 fingerprint alone never admits a foreign producer — admission requires \
                 the identical pinned golden-vector certificate"
            ),
        })
    }
}

/// Typed outcome of the bd-9xuj admission law for a space-verified pairing.
///
/// Emitted by [`BoundQueryEmbedding::verify_producer_conformance`] so callers
/// can log and route the admission basis instead of collapsing it into a bare
/// `Ok`: a foreign producer admitted through its golden-vector certificate is
/// telemetry-visible by construction.
///
/// `#[non_exhaustive]` (review #8151): the different-corpus-digest case
/// today rejects only vacuously through the single reject arm — nothing
/// distinguishes "certificates pinned to different corpora, so conformance
/// is unattestable" from "same corpus, conformance failed". A future typed
/// `Unattestable` classification is expected as a third arm; marking the
/// enum non-exhaustive now means adding it will not be a semver break on
/// downstream matches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "admission")]
#[non_exhaustive]
pub enum SpaceIdentityAdmission {
    /// Same space and the same attested producer.
    SameProducer,
    /// Same space, different producer, admitted because both producers carry
    /// the identical pinned golden-vector certificate
    /// ([`crate::generation::GoldenVectorCertificateV1`]).
    ConformanceCompatibleProducer {
        /// Producer fingerprint of the query-side bundle.
        query_producer_fingerprint: String,
        /// Producer fingerprint the consumer expected.
        expected_producer_fingerprint: String,
    },
}

impl SpaceIdentityAdmission {
    /// Stable `snake_case` code for logs and machine payloads.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::SameProducer => "same_producer",
            Self::ConformanceCompatibleProducer { .. } => "conformance_compatible_producer",
        }
    }
}

/// Per-tier query embeddings, each independently bound to its own space.
///
/// Replaces raw cross-tier `&[f32]` parameters (bd-9xuj): the fast and
/// quality tiers are different embedding spaces, so a search that consults
/// both must carry one bound embedding per tier. At least one tier is
/// always present — a tier-less value is unrepresentable.
#[derive(Clone)]
pub struct TieredQueryEmbeddings {
    fast: Option<BoundQueryEmbedding>,
    quality: Option<BoundQueryEmbedding>,
}

impl fmt::Debug for TieredQueryEmbeddings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TieredQueryEmbeddings")
            .field("fast", &self.fast)
            .field("quality", &self.quality)
            .finish()
    }
}

impl TieredQueryEmbeddings {
    /// Embeddings for both tiers (full progressive search).
    #[must_use]
    pub const fn progressive(fast: BoundQueryEmbedding, quality: BoundQueryEmbedding) -> Self {
        Self {
            fast: Some(fast),
            quality: Some(quality),
        }
    }

    /// Fast-tier only (quality unavailable or deliberately skipped).
    #[must_use]
    pub const fn fast_only(fast: BoundQueryEmbedding) -> Self {
        Self {
            fast: Some(fast),
            quality: None,
        }
    }

    /// Quality-tier only: the quality index is the primary retrieval arm,
    /// never a rescoring pass over a fast-selected pool.
    #[must_use]
    pub const fn quality_only(quality: BoundQueryEmbedding) -> Self {
        Self {
            fast: None,
            quality: Some(quality),
        }
    }

    /// The fast-tier embedding, when present.
    #[must_use]
    pub const fn fast(&self) -> Option<&BoundQueryEmbedding> {
        self.fast.as_ref()
    }

    /// The quality-tier embedding, when present.
    #[must_use]
    pub const fn quality(&self) -> Option<&BoundQueryEmbedding> {
        self.quality.as_ref()
    }

    /// The topology these embeddings can support on their own (before
    /// index/coverage constraints narrow it).
    ///
    /// This is a *request-shape* helper derived from which tiers carry a
    /// bound embedding. It is deliberately NOT authority-bearing and must
    /// never be recorded as a realized topology: the only authority for that
    /// is [`super::TierCoveragePairV1::derive_topology`], which additionally
    /// validates bindings, per-tier coverage, and intent. Use this to answer
    /// "what could these embeddings ask for", never "what did the search do".
    ///
    /// # A hash-control space can never ask for a semantic topology
    ///
    /// If ANY bound tier was produced in an
    /// [`EmbeddingSpaceKindV1::HashControl`] space, the answer is
    /// [`RetrievalTopology::HashControl`] — never `FastOnly`, `QualityOnly`
    /// or `FullProgressive` (bd-ctzo C4). Those three names assert semantic
    /// retrieval; a deterministic control lane produces vectors whose
    /// neighbourhoods carry no learned meaning, so reporting one of them for
    /// a hash query is the "hash-as-semantic" claim this bead forbids. The
    /// r1 revision of this method looked only at which tiers were `Some`,
    /// which meant every hash-control fixture in the tree reported
    /// `FullProgressive` and looked like semantic availability.
    ///
    /// A MIXED pair — one semantic tier, one hash tier — also answers
    /// `HashControl`. It is the weaker of the two claims, and a topology is a
    /// statement about the whole retrieval, not about its best arm.
    ///
    /// [`EmbeddingSpaceKindV1::HashControl`]: crate::generation::EmbeddingSpaceKindV1::HashControl
    #[must_use]
    pub const fn supported_topology(&self) -> RetrievalTopology {
        if self.binds_hash_control() {
            return RetrievalTopology::HashControl;
        }
        match (&self.fast, &self.quality) {
            (Some(_), Some(_)) => RetrievalTopology::FullProgressive,
            (Some(_), None) => RetrievalTopology::FastOnly,
            (None, Some(_)) => RetrievalTopology::QualityOnly,
            // Unreachable by construction; lexical-only is the honest floor.
            (None, None) => RetrievalTopology::LexicalOnly,
        }
    }

    /// Whether any bound tier was produced in a non-semantic control space.
    #[must_use]
    pub const fn binds_hash_control(&self) -> bool {
        const fn is_hash(embedding: Option<&BoundQueryEmbedding>) -> bool {
            match embedding {
                Some(bound) => matches!(
                    bound.identity.space.kind,
                    crate::generation::EmbeddingSpaceKindV1::HashControl
                ),
                None => false,
            }
        }
        is_hash(self.fast.as_ref()) || is_hash(self.quality.as_ref())
    }
}

// ---------------------------------------------------------------------------
// Per-query coverage reconstructed from retained owner witnesses (bd-ctzo C4)
// ---------------------------------------------------------------------------

/// What one tier contributed to ONE query.
///
/// Every populated field is read from the retained owner's own witness or
/// counted off the candidates the search actually returned. There is no
/// constructor that accepts a coverage scalar, because a caller's claim about
/// how much of the corpus it searched is not evidence of anything.
///
/// # Unknown is not zero
///
/// The `Unknown` variant carries NO counts. That is the point: a legacy
/// artifact retains no witness, and reporting its coverage as `0` would be
/// indistinguishable from a live tier that genuinely matched nothing — the
/// exact conflation bd-ctzo C4 forbids. A consumer that wants a number must
/// first destructure `Witnessed`, which means it cannot accidentally average
/// an unknown into a total.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum TierQueryCoverageV1 {
    /// The query bound no embedding for this tier, so it was never consulted.
    NotRequested,
    /// The query bound this tier, but no owner-backed witness exists for it.
    Unknown {
        /// Closed reason no witness was available.
        reason: CoverageUnknownReasonV1,
    },
    /// The tier was served by a retained admitted owner.
    Witnessed {
        /// Generation sequence the owner witnesses. Not a caller's claim.
        generation_sequence: u64,
        /// Live document count the owner witnesses.
        live_count: u64,
        /// How many of the returned candidates this tier actually produced,
        /// counted against the results, not predicted from the request.
        contributed_candidates: u64,
    },
}

impl TierQueryCoverageV1 {
    /// Stable `snake_case` state code for logs and machine payloads.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Unknown { .. } => "unknown",
            Self::Witnessed { .. } => "witnessed",
        }
    }

    /// The witnessed live count, or `None` when this tier has no witness.
    ///
    /// Deliberately `Option<u64>` rather than `u64`: see the type-level note
    /// on why unknown must not collapse to zero.
    #[must_use]
    pub const fn witnessed_live_count(&self) -> Option<u64> {
        match self {
            Self::Witnessed { live_count, .. } => Some(*live_count),
            Self::NotRequested | Self::Unknown { .. } => None,
        }
    }
}

/// Coverage and contribution for one complete tiered query (bd-ctzo C4).
///
/// Built by the index layer from an activated search plus the candidates it
/// returned. Serialized form is stable and bounded: it carries counts,
/// generation sequences and closed enum codes only — never vectors, query
/// text, document ids, tokens, secrets or paths.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchCoverageV1 {
    /// Schema version of this receipt.
    pub schema_version: u16,
    /// Topology the bound embeddings could support — already hash-aware, so
    /// a control-lane query cannot be recorded here as semantic.
    pub topology: RetrievalTopology,
    /// Fast-tier coverage.
    pub fast: TierQueryCoverageV1,
    /// Quality-tier coverage.
    pub quality: TierQueryCoverageV1,
}

impl SearchCoverageV1 {
    /// Assemble a per-query coverage receipt.
    ///
    /// Intentionally has no `coverage_ratio`, `percent` or `total` parameter:
    /// everything here is either the topology the bindings support or a fact
    /// one of the two tier witnesses already established.
    #[must_use]
    pub const fn new(
        topology: RetrievalTopology,
        fast: TierQueryCoverageV1,
        quality: TierQueryCoverageV1,
    ) -> Self {
        Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            topology,
            fast,
            quality,
        }
    }

    /// Whether this query ran on a non-semantic control lane.
    #[must_use]
    pub const fn is_hash_control(&self) -> bool {
        matches!(self.topology, RetrievalTopology::HashControl)
    }

    /// A bounded one-line summary safe to put in a log.
    ///
    /// Contains only closed state codes, generation sequences and counts. No
    /// vector, query, document id, token, secret or path can reach it: the
    /// type holds none of those to begin with, which is a stronger guarantee
    /// than remembering to redact at each call site.
    #[must_use]
    pub fn redacted_summary(&self) -> String {
        fn tier(label: &str, coverage: &TierQueryCoverageV1) -> String {
            match coverage {
                TierQueryCoverageV1::Witnessed {
                    generation_sequence,
                    live_count,
                    contributed_candidates,
                } => format!(
                    "{label}=witnessed(gen={generation_sequence},live={live_count},\
                     contributed={contributed_candidates})"
                ),
                TierQueryCoverageV1::Unknown { reason } => {
                    format!("{label}=unknown({reason:?})")
                }
                TierQueryCoverageV1::NotRequested => format!("{label}=not_requested"),
            }
        }
        format!(
            "topology={:?} {} {}",
            self.topology,
            tier("fast", &self.fast),
            tier("quality", &self.quality)
        )
    }
}
