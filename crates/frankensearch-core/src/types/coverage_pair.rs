use serde::{Deserialize, Serialize};

use crate::error::SearchResult;
use crate::generation::EmbeddingSpaceKindV1;
use crate::recovery_plan::COMPLETE_COVERAGE_PPM;

use super::RetrievalTopology;
use super::coverage_codec::{
    CoverageEncoder, coverage_error, derive_lossless_semantic_topology, embedding_space_kind_code,
    encode_realized_tier_coverage, encode_retrieval_topology, retrieval_topology_fits_request,
    validate_binding_kinds, validate_coverage_schema, validate_coverage_sha256,
    validate_relation_counts, validate_tier_binding,
};
use super::coverage_scope::{
    CoverageBasisV1, CoverageCountsV1, CoverageRelationV1, CoverageTierV1, CoverageUnknownReasonV1,
    CoverageUnverifiedReasonV1, CoverageWitnessV1, TIER_COVERAGE_SCHEMA_VERSION_V1,
};
use super::coverage_tier::{
    TierCoverageRealizationV1, TierCoverageRequestV1, TierCoverageV1, TierCoverageWireV1,
    TrustedTierCoverageContextV1,
};
use super::query_embeddings::TieredQueryEmbeddings;

/// Typed proof that fast and quality requested the same exact member universe
/// under different generation/corpus scopes.
///
/// The proof cannot authorize differing membership: canonical document sets
/// must be equal, and ordered shard sequences must be byte-for-byte equal in
/// the same order. The independent authority fingerprint is supplied by the
/// retained owner context and is never trusted from a serialized payload.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CrossTierUniverseEvidenceV1 {
    schema_version: u16,
    fast_live_witness_fingerprint: String,
    quality_live_witness_fingerprint: String,
    fast_scope_fingerprint: String,
    quality_scope_fingerprint: String,
    basis: CoverageBasisV1,
    member_count: u64,
    membership_fingerprint: String,
    set_fingerprint: String,
    authority_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossTierUniverseEvidenceWireV1 {
    schema_version: u16,
    fast_live_witness_fingerprint: String,
    quality_live_witness_fingerprint: String,
    fast_scope_fingerprint: String,
    quality_scope_fingerprint: String,
    basis: CoverageBasisV1,
    member_count: u64,
    membership_fingerprint: String,
    set_fingerprint: String,
    authority_fingerprint: String,
}

impl CrossTierUniverseEvidenceWireV1 {
    fn into_candidate(self) -> SearchResult<CrossTierUniverseEvidenceV1> {
        let evidence = CrossTierUniverseEvidenceV1 {
            schema_version: self.schema_version,
            fast_live_witness_fingerprint: self.fast_live_witness_fingerprint,
            quality_live_witness_fingerprint: self.quality_live_witness_fingerprint,
            fast_scope_fingerprint: self.fast_scope_fingerprint,
            quality_scope_fingerprint: self.quality_scope_fingerprint,
            basis: self.basis,
            member_count: self.member_count,
            membership_fingerprint: self.membership_fingerprint,
            set_fingerprint: self.set_fingerprint,
            authority_fingerprint: self.authority_fingerprint,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl CrossTierUniverseEvidenceV1 {
    fn build(
        fast: &CoverageWitnessV1,
        quality: &CoverageWitnessV1,
        authority_fingerprint: String,
    ) -> SearchResult<Self> {
        validate_coverage_sha256(
            "coverage.cross_tier.authority_fingerprint",
            &authority_fingerprint,
        )?;
        if fast.tier != CoverageTierV1::Fast || quality.tier != CoverageTierV1::Quality {
            return Err(coverage_error(
                "coverage.cross_tier",
                "swapped-tier-roles",
                "cross-tier evidence requires fast then quality live witnesses",
            ));
        }
        if fast.scope == quality.scope {
            return Err(coverage_error(
                "coverage.cross_tier",
                "redundant-same-scope-evidence",
                "cross-tier evidence is valid only when generation or corpus scope differs",
            ));
        }
        if fast.basis != quality.basis
            || fast.member_count != quality.member_count
            || fast.membership_fingerprint != quality.membership_fingerprint
            || fast.set_fingerprint != quality.set_fingerprint
        {
            return Err(coverage_error(
                "coverage.cross_tier",
                "requested-universe-mismatch",
                "cross-tier evidence cannot authorize differing sets, bases, or ordered sequences",
            ));
        }
        let evidence = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            fast_live_witness_fingerprint: fast.fingerprint(),
            quality_live_witness_fingerprint: quality.fingerprint(),
            fast_scope_fingerprint: fast.scope.fingerprint(),
            quality_scope_fingerprint: quality.scope.fingerprint(),
            basis: fast.basis,
            member_count: fast.member_count,
            membership_fingerprint: fast.membership_fingerprint.clone(),
            set_fingerprint: fast.set_fingerprint.clone(),
            authority_fingerprint,
        };
        evidence.validate_bound(fast, quality)?;
        Ok(evidence)
    }

    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.cross_tier.schema_version", self.schema_version)?;
        for (field, value) in [
            (
                "fast_live_witness_fingerprint",
                &self.fast_live_witness_fingerprint,
            ),
            (
                "quality_live_witness_fingerprint",
                &self.quality_live_witness_fingerprint,
            ),
            ("fast_scope_fingerprint", &self.fast_scope_fingerprint),
            ("quality_scope_fingerprint", &self.quality_scope_fingerprint),
            ("membership_fingerprint", &self.membership_fingerprint),
            ("set_fingerprint", &self.set_fingerprint),
            ("authority_fingerprint", &self.authority_fingerprint),
        ] {
            validate_coverage_sha256(&format!("coverage.cross_tier.{field}"), value)?;
        }
        if self.fast_scope_fingerprint == self.quality_scope_fingerprint {
            return Err(coverage_error(
                "coverage.cross_tier",
                "redundant-same-scope-evidence",
                "cross-tier evidence requires different generation or corpus scope",
            ));
        }
        Ok(())
    }

    fn validate_bound(
        &self,
        fast: &CoverageWitnessV1,
        quality: &CoverageWitnessV1,
    ) -> SearchResult<()> {
        self.validate()?;
        if self.fast_live_witness_fingerprint != fast.fingerprint()
            || self.quality_live_witness_fingerprint != quality.fingerprint()
            || self.fast_scope_fingerprint != fast.scope.fingerprint()
            || self.quality_scope_fingerprint != quality.scope.fingerprint()
            || self.basis != fast.basis
            || self.basis != quality.basis
            || self.member_count != fast.member_count
            || self.member_count != quality.member_count
            || self.membership_fingerprint != fast.membership_fingerprint
            || self.membership_fingerprint != quality.membership_fingerprint
            || self.set_fingerprint != fast.set_fingerprint
            || self.set_fingerprint != quality.set_fingerprint
        {
            return Err(coverage_error(
                "coverage.cross_tier",
                "binding-mismatch",
                "cross-tier evidence does not bind both exact live witnesses",
            ));
        }
        Ok(())
    }

    /// Domain-separated fingerprint of the complete cross-tier proof.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.cross-tier-universe.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.fast_live_witness_fingerprint);
        encoder.text(&self.quality_live_witness_fingerprint);
        encoder.text(&self.fast_scope_fingerprint);
        encoder.text(&self.quality_scope_fingerprint);
        encoder.text(self.basis.code());
        encoder.u64(self.member_count);
        encoder.text(&self.membership_fingerprint);
        encoder.text(&self.set_fingerprint);
        encoder.text(&self.authority_fingerprint);
        encoder.fingerprint()
    }
}

/// Fast and quality coverage values retained as two independently validated authorities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TierCoveragePairV1 {
    schema_version: u16,
    fast: TierCoverageV1,
    quality: TierCoverageV1,
    cross_tier_universe_evidence: Option<Box<CrossTierUniverseEvidenceV1>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TierCoveragePairWireV1 {
    schema_version: u16,
    fast: TierCoverageWireV1,
    quality: TierCoverageWireV1,
    cross_tier_universe_evidence: Option<Box<CrossTierUniverseEvidenceWireV1>>,
}

impl TierCoveragePairWireV1 {
    fn into_candidate(self) -> SearchResult<TierCoveragePairV1> {
        let pair = TierCoveragePairV1 {
            schema_version: self.schema_version,
            fast: self.fast.into_candidate()?,
            quality: self.quality.into_candidate()?,
            cross_tier_universe_evidence: self
                .cross_tier_universe_evidence
                .map(|evidence| (*evidence).into_candidate().map(Box::new))
                .transpose()?,
        };
        pair.validate()?;
        Ok(pair)
    }
}

/// Syntactically checked fast/quality wire claim without owner authority.
#[derive(Deserialize)]
#[serde(transparent)]
pub struct UntrustedTierCoveragePairV1(TierCoveragePairWireV1);

impl UntrustedTierCoveragePairV1 {
    /// Promote only after recomputing both tiers and any cross-tier proof from
    /// independent owner authority.
    ///
    /// # Errors
    ///
    /// Rejects malformed wire data, invalid trusted contexts, missing or
    /// malformed independent authority, and any non-identical recomputation.
    pub fn validate_against(
        self,
        fast_context: &TrustedTierCoverageContextV1,
        quality_context: &TrustedTierCoverageContextV1,
        cross_tier_authority_fingerprint: Option<&str>,
    ) -> SearchResult<TierCoveragePairV1> {
        let candidate = self.0.into_candidate()?;
        let fast = fast_context.recompute()?;
        let quality = quality_context.recompute()?;
        let recomputed = match cross_tier_authority_fingerprint {
            Some(authority) => TierCoveragePairV1::new_with_cross_tier_authority(
                fast,
                quality,
                authority.to_owned(),
            )?,
            None => TierCoveragePairV1::new(fast, quality)?,
        };
        if candidate != recomputed {
            return Err(coverage_error(
                "coverage.pair.untrusted",
                "authority-mismatch",
                "serialized coverage pair does not equal owner-backed recomputation",
            ));
        }
        Ok(recomputed)
    }
}

impl TierCoveragePairV1 {
    /// Pair exactly one fast and one quality coverage authority.
    ///
    /// # Errors
    ///
    /// Rejects swapped roles or any invalid child coverage object.
    pub fn new(fast: TierCoverageV1, quality: TierCoverageV1) -> SearchResult<Self> {
        Self::from_parts(fast, quality, None)
    }

    /// Pair equal requested membership under differing generation/corpus
    /// scopes using an independently supplied cross-tier authority.
    ///
    /// The authority fingerprint must be obtained from a retained owner outside
    /// the serialized claim. Calling this constructor with a self-sourced wire
    /// digest does not promote or attest untrusted data.
    ///
    /// # Errors
    ///
    /// Rejects same-scope evidence, mismatched bases or membership, reordered
    /// ordered shards, malformed authority fingerprints, and unrequested arms.
    pub fn new_with_cross_tier_authority(
        fast: TierCoverageV1,
        quality: TierCoverageV1,
        authority_fingerprint: impl Into<String>,
    ) -> SearchResult<Self> {
        let fast_live = fast.live_witness().ok_or_else(|| {
            coverage_error(
                "coverage.cross_tier.fast",
                "not-requested",
                "cross-tier evidence requires a requested fast universe",
            )
        })?;
        let quality_live = quality.live_witness().ok_or_else(|| {
            coverage_error(
                "coverage.cross_tier.quality",
                "not-requested",
                "cross-tier evidence requires a requested quality universe",
            )
        })?;
        let evidence = CrossTierUniverseEvidenceV1::build(
            fast_live,
            quality_live,
            authority_fingerprint.into(),
        )?;
        Self::from_parts(fast, quality, Some(Box::new(evidence)))
    }

    fn from_parts(
        fast: TierCoverageV1,
        quality: TierCoverageV1,
        cross_tier_universe_evidence: Option<Box<CrossTierUniverseEvidenceV1>>,
    ) -> SearchResult<Self> {
        let pair = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            fast,
            quality,
            cross_tier_universe_evidence,
        };
        pair.validate()?;
        Ok(pair)
    }

    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.pair.schema_version", self.schema_version)?;
        self.fast.validate()?;
        self.quality.validate()?;
        if self.fast.tier != CoverageTierV1::Fast || self.quality.tier != CoverageTierV1::Quality {
            return Err(coverage_error(
                "coverage.pair",
                "swapped-tier-roles",
                "pair requires independent fast then quality coverage values",
            ));
        }
        match (self.fast.live_witness(), self.quality.live_witness()) {
            (None | Some(_), None) | (None, Some(_)) => {
                if self.cross_tier_universe_evidence.is_some() {
                    return Err(coverage_error(
                        "coverage.cross_tier",
                        "evidence-for-unrequested-tier",
                        "cross-tier evidence requires both requested universes",
                    ));
                }
            }
            (Some(fast), Some(quality)) => {
                if fast.basis != quality.basis
                    || fast.member_count != quality.member_count
                    || fast.membership_fingerprint != quality.membership_fingerprint
                    || fast.set_fingerprint != quality.set_fingerprint
                {
                    return Err(coverage_error(
                        "coverage.pair.requested_universe",
                        "mismatch",
                        "fast and quality must request the same canonical set or ordered shard sequence",
                    ));
                }
                match (
                    fast.scope == quality.scope,
                    self.cross_tier_universe_evidence.as_deref(),
                ) {
                    (true, None) => {}
                    (true, Some(_)) => {
                        return Err(coverage_error(
                            "coverage.cross_tier",
                            "redundant-same-scope-evidence",
                            "same-scope requested universes must not carry cross-tier evidence",
                        ));
                    }
                    (false, None) => {
                        return Err(coverage_error(
                            "coverage.cross_tier",
                            "missing",
                            "different requested generation/corpus scopes require typed cross-tier evidence",
                        ));
                    }
                    (false, Some(evidence)) => evidence.validate_bound(fast, quality)?,
                }
            }
        }
        Ok(())
    }

    /// Fast-tier coverage authority.
    #[must_use]
    pub const fn fast(&self) -> &TierCoverageV1 {
        &self.fast
    }

    /// Quality-tier coverage authority.
    #[must_use]
    pub const fn quality(&self) -> &TierCoverageV1 {
        &self.quality
    }

    /// Cross-tier universe proof, present only for equal exact membership
    /// under differing generation/corpus scopes.
    #[must_use]
    pub fn cross_tier_universe_evidence(&self) -> Option<&CrossTierUniverseEvidenceV1> {
        self.cross_tier_universe_evidence.as_deref()
    }

    /// Stable digest that preserves independent fast/quality witness roles.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.tier-coverage-pair.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.fast.fingerprint());
        encoder.text(&self.quality.fingerprint());
        encoder.option_text(
            self.cross_tier_universe_evidence
                .as_ref()
                .map(|evidence| evidence.fingerprint())
                .as_deref(),
        );
        encoder.fingerprint()
    }

    /// Derive requested and realized topology from validated bindings plus both coverages.
    ///
    /// This is the authority-bearing constructor for realized topology. It validates complete
    /// identity bundles, joins each binding to its independently named tier/space, rejects a
    /// duplicate or swapped binding, and refuses semantic intent for hash-control spaces.
    /// Unknown/unverified coverage never becomes availability, and a realized superset fails
    /// closed because stale/foreign indexed members must not be searched.
    ///
    /// # Errors
    ///
    /// Rejects missing/extra/swapped/duplicate bindings, malformed identities, intent/kind
    /// mismatch, unsafe supersets, and any coverage-pair invariant failure.
    pub fn derive_topology(
        &self,
        intent: RetrievalIntentV1,
        embeddings: Option<&TieredQueryEmbeddings>,
    ) -> SearchResult<DerivedRetrievalTopologyV1> {
        self.validate()?;
        let fast_embedding = embeddings.and_then(TieredQueryEmbeddings::fast);
        let quality_embedding = embeddings.and_then(TieredQueryEmbeddings::quality);
        validate_tier_binding(&self.fast, fast_embedding)?;
        validate_tier_binding(&self.quality, quality_embedding)?;

        if let (Some(fast), Some(quality)) = (fast_embedding, quality_embedding)
            && fast.space_fingerprint() == quality.space_fingerprint()
        {
            return Err(coverage_error(
                "retrieval.bindings",
                "duplicate-space-binding",
                "fast and quality roles require independent space bindings",
            ));
        }

        let fast_kind = fast_embedding.map(|embedding| embedding.identity.space.kind);
        let quality_kind = quality_embedding.map(|embedding| embedding.identity.space.kind);
        let requested_from_arms = match (self.fast.is_requested(), self.quality.is_requested()) {
            (false, false) => RetrievalTopology::LexicalOnly,
            (true, false) => RetrievalTopology::FastOnly,
            (false, true) => RetrievalTopology::QualityOnly,
            (true, true) => RetrievalTopology::FullProgressive,
        };
        let fast_realized = RealizedTierCoverageV1::from_coverage(&self.fast)?;
        let quality_realized = RealizedTierCoverageV1::from_coverage(&self.quality)?;

        let (requested, coarse_realized) = match intent {
            RetrievalIntentV1::LexicalOnly => {
                if embeddings.is_some() || requested_from_arms != RetrievalTopology::LexicalOnly {
                    return Err(coverage_error(
                        "retrieval.intent",
                        "lexical_only",
                        "lexical-only intent cannot retain semantic/hash bindings or coverage",
                    ));
                }
                (
                    RetrievalTopology::LexicalOnly,
                    Some(RetrievalTopology::LexicalOnly),
                )
            }
            RetrievalIntentV1::HashControl => {
                if requested_from_arms == RetrievalTopology::LexicalOnly {
                    return Err(coverage_error(
                        "retrieval.intent",
                        "hash_control",
                        "hash-control intent requires an explicit bound control tier",
                    ));
                }
                validate_binding_kinds(
                    intent,
                    fast_kind,
                    quality_kind,
                    EmbeddingSpaceKindV1::HashControl,
                )?;
                let realized = if fast_realized.contributes() || quality_realized.contributes() {
                    Some(RetrievalTopology::HashControl)
                } else {
                    None
                };
                (RetrievalTopology::HashControl, realized)
            }
            RetrievalIntentV1::Semantic => {
                if requested_from_arms == RetrievalTopology::LexicalOnly {
                    return Err(coverage_error(
                        "retrieval.intent",
                        "semantic",
                        "semantic intent requires at least one independently bound tier",
                    ));
                }
                validate_binding_kinds(
                    intent,
                    fast_kind,
                    quality_kind,
                    EmbeddingSpaceKindV1::Semantic,
                )?;
                (
                    requested_from_arms,
                    derive_lossless_semantic_topology(&fast_realized, &quality_realized),
                )
            }
        };

        let semantic_available = intent == RetrievalIntentV1::Semantic
            && (fast_realized.contributes() || quality_realized.contributes());
        let decision = DerivedRetrievalTopologyV1 {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            intent,
            requested,
            coarse_realized,
            fast_realized,
            quality_realized,
            fast_space_fingerprint: fast_embedding
                .map(|embedding| embedding.space_fingerprint().to_owned()),
            quality_space_fingerprint: quality_embedding
                .map(|embedding| embedding.space_fingerprint().to_owned()),
            fast_kind,
            quality_kind,
            coverage_pair_fingerprint: self.fingerprint(),
            semantic_available,
        };
        decision.validate()?;
        Ok(decision)
    }
}

/// Declared retrieval intent checked against the actual binding-space kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalIntentV1 {
    /// No vector arm may be present.
    LexicalOnly,
    /// Explicit deterministic non-semantic control/degradation lane.
    HashControl,
    /// Learned semantic vector retrieval.
    Semantic,
}

impl RetrievalIntentV1 {
    pub(super) const fn code(self) -> &'static str {
        match self {
            Self::LexicalOnly => "lexical_only",
            Self::HashControl => "hash_control",
            Self::Semantic => "semantic",
        }
    }
}

/// Exact realized state of one independently admitted retrieval tier.
///
/// Unlike the legacy topology vocabulary, this preserves missing, unknown,
/// unverified, zero-live, complete, and partial facts per tier. Availability
/// is derived from exact nonzero intersection counts, never rounded PPM.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum RealizedTierCoverageV1 {
    /// The tier was not requested.
    NotRequested,
    /// The tier was requested but no realized index exists.
    Missing,
    /// No immutable owner could establish realized facts.
    Unknown {
        /// Closed reason for the unknown state.
        reason: CoverageUnknownReasonV1,
    },
    /// Observations exist but lack owner-backed member verification.
    Unverified {
        /// Closed reason verification failed.
        reason: CoverageUnverifiedReasonV1,
        /// Optional diagnostic count; never grants availability.
        observed_indexed_count: Option<u64>,
    },
    /// Requested and indexed member sets are both empty.
    ZeroLive {
        /// Exact checked zero-live counts.
        counts: CoverageCountsV1,
    },
    /// Requested and indexed member sets are exactly complete and non-empty.
    Complete {
        /// Exact checked complete counts.
        counts: CoverageCountsV1,
    },
    /// Incomplete same/cross-scope coverage, including exact disjoint facts.
    Partial {
        /// Exact derived relation; never `Complete`, `ZeroLive`, or `Superset`.
        relation: CoverageRelationV1,
        /// Exact checked counts and diagnostic PPM.
        counts: CoverageCountsV1,
    },
}

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
enum RealizedTierCoverageWireV1 {
    NotRequested,
    Missing,
    Unknown {
        reason: CoverageUnknownReasonV1,
    },
    Unverified {
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    },
    ZeroLive {
        counts: CoverageCountsV1,
    },
    Complete {
        counts: CoverageCountsV1,
    },
    Partial {
        relation: CoverageRelationV1,
        counts: CoverageCountsV1,
    },
}

impl RealizedTierCoverageWireV1 {
    fn into_candidate(self) -> SearchResult<RealizedTierCoverageV1> {
        let realized = match self {
            Self::NotRequested => RealizedTierCoverageV1::NotRequested,
            Self::Missing => RealizedTierCoverageV1::Missing,
            Self::Unknown { reason } => RealizedTierCoverageV1::Unknown { reason },
            Self::Unverified {
                reason,
                observed_indexed_count,
            } => RealizedTierCoverageV1::Unverified {
                reason,
                observed_indexed_count,
            },
            Self::ZeroLive { counts } => RealizedTierCoverageV1::ZeroLive { counts },
            Self::Complete { counts } => RealizedTierCoverageV1::Complete { counts },
            Self::Partial { relation, counts } => {
                RealizedTierCoverageV1::Partial { relation, counts }
            }
        };
        realized.validate()?;
        Ok(realized)
    }
}

impl RealizedTierCoverageV1 {
    fn from_coverage(coverage: &TierCoverageV1) -> SearchResult<Self> {
        let realized = match (&coverage.requested, &coverage.realized) {
            (TierCoverageRequestV1::MissingTier, TierCoverageRealizationV1::MissingTier) => {
                Self::NotRequested
            }
            (TierCoverageRequestV1::Requested { .. }, TierCoverageRealizationV1::MissingTier) => {
                Self::Missing
            }
            (
                TierCoverageRequestV1::Requested { .. },
                TierCoverageRealizationV1::Unknown { reason },
            ) => Self::Unknown { reason: *reason },
            (
                TierCoverageRequestV1::Requested { .. },
                TierCoverageRealizationV1::Unverified {
                    reason,
                    observed_indexed_count,
                },
            ) => Self::Unverified {
                reason: *reason,
                observed_indexed_count: *observed_indexed_count,
            },
            (
                TierCoverageRequestV1::Requested { .. },
                TierCoverageRealizationV1::Verified {
                    relation, counts, ..
                },
            ) => match relation {
                CoverageRelationV1::ZeroLive => Self::ZeroLive { counts: *counts },
                CoverageRelationV1::Complete => Self::Complete { counts: *counts },
                CoverageRelationV1::Subset
                | CoverageRelationV1::Overlap
                | CoverageRelationV1::Disjoint => Self::Partial {
                    relation: *relation,
                    counts: *counts,
                },
                CoverageRelationV1::Superset => {
                    return Err(coverage_error(
                        "retrieval.coverage",
                        "unsafe-superset",
                        "indexed coverage contains stale or foreign members",
                    ));
                }
            },
            (TierCoverageRequestV1::MissingTier, _) => {
                return Err(coverage_error(
                    "retrieval.coverage",
                    "realized-unrequested-tier",
                    "unrequested tier cannot carry realized coverage",
                ));
            }
        };
        realized.validate()?;
        Ok(realized)
    }

    pub(super) fn validate(&self) -> SearchResult<()> {
        match self {
            Self::NotRequested | Self::Missing | Self::Unknown { .. } | Self::Unverified { .. } => {
                Ok(())
            }
            Self::ZeroLive { counts } => {
                validate_relation_counts(CoverageRelationV1::ZeroLive, *counts)
            }
            Self::Complete { counts } => {
                validate_relation_counts(CoverageRelationV1::Complete, *counts)
            }
            Self::Partial { relation, counts } => {
                if !matches!(
                    relation,
                    CoverageRelationV1::Subset
                        | CoverageRelationV1::Overlap
                        | CoverageRelationV1::Disjoint
                ) {
                    return Err(coverage_error(
                        "retrieval.realized_tier.relation",
                        relation.code(),
                        "partial realized state requires subset, overlap, or disjoint relation",
                    ));
                }
                validate_relation_counts(*relation, *counts)
            }
        }
    }

    /// True only when exact checked facts contain at least one live indexed member.
    #[must_use]
    pub const fn contributes(&self) -> bool {
        match self {
            Self::Complete { counts } | Self::Partial { counts, .. } => {
                counts.intersection_count > 0
            }
            Self::NotRequested
            | Self::Missing
            | Self::Unknown { .. }
            | Self::Unverified { .. }
            | Self::ZeroLive { .. } => false,
        }
    }

    /// Diagnostic PPM derived from exact counts, when coverage is verified.
    #[must_use]
    pub const fn coverage_ppm(&self) -> Option<u32> {
        match self {
            Self::ZeroLive { counts }
            | Self::Complete { counts }
            | Self::Partial { counts, .. } => Some(counts.coverage_ppm),
            Self::NotRequested | Self::Missing | Self::Unknown { .. } | Self::Unverified { .. } => {
                None
            }
        }
    }
}

/// Validated requested-versus-realized topology receipt.
///
/// This receipt contains bounded digests only. It records a successful derivation; it is not a
/// replacement for retaining the query embeddings and owner-backed coverage authorities used by
/// [`TierCoveragePairV1::derive_topology`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct DerivedRetrievalTopologyV1 {
    schema_version: u16,
    intent: RetrievalIntentV1,
    requested: RetrievalTopology,
    coarse_realized: Option<RetrievalTopology>,
    fast_realized: RealizedTierCoverageV1,
    quality_realized: RealizedTierCoverageV1,
    fast_space_fingerprint: Option<String>,
    quality_space_fingerprint: Option<String>,
    fast_kind: Option<EmbeddingSpaceKindV1>,
    quality_kind: Option<EmbeddingSpaceKindV1>,
    coverage_pair_fingerprint: String,
    semantic_available: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DerivedRetrievalTopologyWireV1 {
    schema_version: u16,
    intent: RetrievalIntentV1,
    requested: RetrievalTopology,
    coarse_realized: Option<RetrievalTopology>,
    fast_realized: RealizedTierCoverageWireV1,
    quality_realized: RealizedTierCoverageWireV1,
    fast_space_fingerprint: Option<String>,
    quality_space_fingerprint: Option<String>,
    fast_kind: Option<EmbeddingSpaceKindV1>,
    quality_kind: Option<EmbeddingSpaceKindV1>,
    coverage_pair_fingerprint: String,
    semantic_available: bool,
}

impl DerivedRetrievalTopologyWireV1 {
    fn into_candidate(self) -> SearchResult<DerivedRetrievalTopologyV1> {
        let decision = DerivedRetrievalTopologyV1 {
            schema_version: self.schema_version,
            intent: self.intent,
            requested: self.requested,
            coarse_realized: self.coarse_realized,
            fast_realized: self.fast_realized.into_candidate()?,
            quality_realized: self.quality_realized.into_candidate()?,
            fast_space_fingerprint: self.fast_space_fingerprint,
            quality_space_fingerprint: self.quality_space_fingerprint,
            fast_kind: self.fast_kind,
            quality_kind: self.quality_kind,
            coverage_pair_fingerprint: self.coverage_pair_fingerprint,
            semantic_available: self.semantic_available,
        };
        decision.validate()?;
        Ok(decision)
    }
}

/// Serialized derived-topology claim awaiting fresh coverage and binding authority.
#[derive(Deserialize)]
#[serde(transparent)]
pub struct UntrustedDerivedRetrievalTopologyV1(DerivedRetrievalTopologyWireV1);

impl UntrustedDerivedRetrievalTopologyV1 {
    /// Promote only by rerunning the complete topology derivation and comparing
    /// every receipt field with the wire claim.
    ///
    /// # Errors
    ///
    /// Rejects malformed wire data, invalid coverage or query bindings, and any
    /// receipt field that differs from fresh authoritative derivation.
    pub fn validate_against(
        self,
        pair: &TierCoveragePairV1,
        intent: RetrievalIntentV1,
        embeddings: Option<&TieredQueryEmbeddings>,
    ) -> SearchResult<DerivedRetrievalTopologyV1> {
        let candidate = self.0.into_candidate()?;
        let recomputed = pair.derive_topology(intent, embeddings)?;
        if candidate != recomputed {
            return Err(coverage_error(
                "retrieval.topology.untrusted",
                "authority-mismatch",
                "serialized topology does not equal a fresh authoritative derivation",
            ));
        }
        Ok(recomputed)
    }
}

impl DerivedRetrievalTopologyV1 {
    fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("retrieval.topology.schema_version", self.schema_version)?;
        self.fast_realized.validate()?;
        self.quality_realized.validate()?;
        validate_coverage_sha256(
            "retrieval.topology.coverage_pair_fingerprint",
            &self.coverage_pair_fingerprint,
        )?;
        for (field, fingerprint) in [
            (
                "retrieval.topology.fast_space_fingerprint",
                self.fast_space_fingerprint.as_deref(),
            ),
            (
                "retrieval.topology.quality_space_fingerprint",
                self.quality_space_fingerprint.as_deref(),
            ),
        ] {
            if let Some(fingerprint) = fingerprint {
                validate_coverage_sha256(field, fingerprint)?;
            }
        }
        if self.fast_space_fingerprint.is_some() != self.fast_kind.is_some()
            || self.quality_space_fingerprint.is_some() != self.quality_kind.is_some()
        {
            return Err(coverage_error(
                "retrieval.topology.bindings",
                "incomplete-kind-binding",
                "each retained space fingerprint must carry its validated space kind",
            ));
        }
        if self.fast_space_fingerprint.is_some()
            && self.fast_space_fingerprint == self.quality_space_fingerprint
        {
            return Err(coverage_error(
                "retrieval.topology.bindings",
                "duplicate-space-binding",
                "fast and quality roles require independent space bindings",
            ));
        }
        if self.fast_space_fingerprint.is_none()
            != matches!(self.fast_realized, RealizedTierCoverageV1::NotRequested)
            || self.quality_space_fingerprint.is_none()
                != matches!(self.quality_realized, RealizedTierCoverageV1::NotRequested)
        {
            return Err(coverage_error(
                "retrieval.topology.bindings",
                "request-state-mismatch",
                "each requested tier must have exactly one independently bound query embedding",
            ));
        }
        let binding_shape = match (
            self.fast_space_fingerprint.is_some(),
            self.quality_space_fingerprint.is_some(),
        ) {
            (false, false) => RetrievalTopology::LexicalOnly,
            (true, false) => RetrievalTopology::FastOnly,
            (false, true) => RetrievalTopology::QualityOnly,
            (true, true) => RetrievalTopology::FullProgressive,
        };
        match self.intent {
            RetrievalIntentV1::LexicalOnly => {
                if binding_shape != RetrievalTopology::LexicalOnly
                    || self.requested != RetrievalTopology::LexicalOnly
                    || self.coarse_realized != Some(RetrievalTopology::LexicalOnly)
                {
                    return Err(coverage_error(
                        "retrieval.topology",
                        "invalid-lexical-receipt",
                        "lexical-only receipt cannot carry vector bindings or topology",
                    ));
                }
            }
            RetrievalIntentV1::HashControl => {
                validate_binding_kinds(
                    self.intent,
                    self.fast_kind,
                    self.quality_kind,
                    EmbeddingSpaceKindV1::HashControl,
                )?;
                let expected =
                    if self.fast_realized.contributes() || self.quality_realized.contributes() {
                        Some(RetrievalTopology::HashControl)
                    } else {
                        None
                    };
                if self.requested != RetrievalTopology::HashControl
                    || self.coarse_realized != expected
                {
                    return Err(coverage_error(
                        "retrieval.topology",
                        "invalid-hash-control-receipt",
                        "hash control must remain explicit and non-semantic",
                    ));
                }
            }
            RetrievalIntentV1::Semantic => {
                validate_binding_kinds(
                    self.intent,
                    self.fast_kind,
                    self.quality_kind,
                    EmbeddingSpaceKindV1::Semantic,
                )?;
                if self.requested != binding_shape
                    || self.coarse_realized
                        != derive_lossless_semantic_topology(
                            &self.fast_realized,
                            &self.quality_realized,
                        )
                {
                    return Err(coverage_error(
                        "retrieval.topology",
                        "request-realization-mismatch",
                        "realized semantic topology exceeds or contradicts requested bindings",
                    ));
                }
            }
        }
        if let Some(RetrievalTopology::PartialQuality { coverage_ppm }) = self.coarse_realized
            && (coverage_ppm == 0 || coverage_ppm >= COMPLETE_COVERAGE_PPM)
        {
            return Err(coverage_error(
                "retrieval.topology.realized.coverage_ppm",
                &coverage_ppm.to_string(),
                "partial quality coverage must be strictly between zero and complete",
            ));
        }
        if let Some(realized) = self.coarse_realized
            && !retrieval_topology_fits_request(self.requested, realized)
        {
            return Err(coverage_error(
                "retrieval.topology",
                "request-realization-mismatch",
                "coarse realized topology exceeds or contradicts the requested topology",
            ));
        }
        let expected_semantic = self.intent == RetrievalIntentV1::Semantic
            && (self.fast_realized.contributes() || self.quality_realized.contributes());
        if self.semantic_available != expected_semantic {
            return Err(coverage_error(
                "retrieval.topology.semantic_available",
                &self.semantic_available.to_string(),
                "must be derived from semantic intent and realized semantic topology",
            ));
        }
        Ok(())
    }

    /// Validated request topology.
    #[must_use]
    pub const fn requested(&self) -> RetrievalTopology {
        self.requested
    }

    /// Lossless legacy coarse topology, absent when exact per-tier facts cannot
    /// be represented without lying.
    #[must_use]
    pub const fn coarse_realized(&self) -> Option<RetrievalTopology> {
        self.coarse_realized
    }

    /// Exact independently realized fast-tier state.
    #[must_use]
    pub const fn fast_realized(&self) -> &RealizedTierCoverageV1 {
        &self.fast_realized
    }

    /// Exact independently realized quality-tier state.
    #[must_use]
    pub const fn quality_realized(&self) -> &RealizedTierCoverageV1 {
        &self.quality_realized
    }

    /// Whether a learned semantic space is truthfully available.
    #[must_use]
    pub const fn semantic_available(&self) -> bool {
        self.semantic_available
    }

    /// Stable receipt digest; semantic mutations change it.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.derived-retrieval-topology.v1");
        encoder.u16(self.schema_version);
        encoder.text(self.intent.code());
        encode_retrieval_topology(&mut encoder, self.requested);
        match self.coarse_realized {
            Some(realized) => {
                encoder.u8(1);
                encode_retrieval_topology(&mut encoder, realized);
            }
            None => encoder.u8(0),
        }
        encode_realized_tier_coverage(&mut encoder, &self.fast_realized);
        encode_realized_tier_coverage(&mut encoder, &self.quality_realized);
        encoder.option_text(self.fast_space_fingerprint.as_deref());
        encoder.option_text(self.quality_space_fingerprint.as_deref());
        encoder.option_text(self.fast_kind.map(embedding_space_kind_code));
        encoder.option_text(self.quality_kind.map(embedding_space_kind_code));
        encoder.text(&self.coverage_pair_fingerprint);
        encoder.bool(self.semantic_available);
        encoder.fingerprint()
    }
}
