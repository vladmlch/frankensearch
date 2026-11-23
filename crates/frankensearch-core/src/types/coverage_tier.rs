use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::SearchResult;
use crate::generation::EmbeddingSpaceIdentityV1;

use super::coverage_codec::{
    CoverageEncoder, coverage_error, derive_coverage_relation, encode_coverage_realization,
    encode_coverage_request, validate_coverage_schema, validate_relation_counts,
};
use super::coverage_scope::{
    CoverageBasisV1, CoverageCountsV1, CoverageRelationV1, CoverageScopeV1, CoverageTierV1,
    CoverageUnknownReasonV1, CoverageUnverifiedReasonV1, CoverageWitnessV1, CoverageWitnessWireV1,
    CrossScopeCoverageEvidenceV1, CrossScopeCoverageEvidenceWireV1,
    TIER_COVERAGE_SCHEMA_VERSION_V1, prepare_coverage_members,
};

/// Requested side of one tier's independent coverage contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum TierCoverageRequestV1 {
    /// This tier was not requested; it carries no borrowed witness.
    MissingTier,
    /// The exact live member set requested for this tier.
    Requested {
        /// Tier-tagged live member witness.
        live: CoverageWitnessV1,
    },
}

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum TierCoverageRequestWireV1 {
    MissingTier,
    Requested { live: CoverageWitnessWireV1 },
}

impl TierCoverageRequestWireV1 {
    pub(super) fn into_candidate(self) -> SearchResult<TierCoverageRequestV1> {
        match self {
            Self::MissingTier => Ok(TierCoverageRequestV1::MissingTier),
            Self::Requested { live } => Ok(TierCoverageRequestV1::Requested {
                live: live.into_candidate()?,
            }),
        }
    }
}

/// Realized side of one tier's independent coverage contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum TierCoverageRealizationV1 {
    /// No realized tier/index exists.
    MissingTier,
    /// Coverage could not be observed from an immutable authority.
    Unknown {
        /// Closed unknown-state reason.
        reason: CoverageUnknownReasonV1,
    },
    /// A scalar observation exists, but no verified unique-member witness does.
    Unverified {
        /// Closed verification failure reason.
        reason: CoverageUnverifiedReasonV1,
        /// Optional observed count; never used to derive semantic availability.
        observed_indexed_count: Option<u64>,
    },
    /// Coverage was derived from two duplicate-free member sets.
    Verified {
        /// Tier-tagged indexed member witness.
        indexed: Box<CoverageWitnessV1>,
        /// Derived exact set relation.
        relation: CoverageRelationV1,
        /// Derived checked counts and fixed-point ratio.
        counts: CoverageCountsV1,
        /// Required and fully bound only when generation/corpus scopes differ.
        cross_scope_evidence: Option<Box<CrossScopeCoverageEvidenceV1>>,
    },
}

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum TierCoverageRealizationWireV1 {
    MissingTier,
    Unknown {
        reason: CoverageUnknownReasonV1,
    },
    Unverified {
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    },
    Verified {
        indexed: Box<CoverageWitnessWireV1>,
        relation: CoverageRelationV1,
        counts: CoverageCountsV1,
        cross_scope_evidence: Option<Box<CrossScopeCoverageEvidenceWireV1>>,
    },
}

impl TierCoverageRealizationWireV1 {
    pub(super) fn into_candidate(self) -> SearchResult<TierCoverageRealizationV1> {
        match self {
            Self::MissingTier => Ok(TierCoverageRealizationV1::MissingTier),
            Self::Unknown { reason } => Ok(TierCoverageRealizationV1::Unknown { reason }),
            Self::Unverified {
                reason,
                observed_indexed_count,
            } => Ok(TierCoverageRealizationV1::Unverified {
                reason,
                observed_indexed_count,
            }),
            Self::Verified {
                indexed,
                relation,
                counts,
                cross_scope_evidence,
            } => Ok(TierCoverageRealizationV1::Verified {
                indexed: Box::new((*indexed).into_candidate()?),
                relation,
                counts,
                cross_scope_evidence: cross_scope_evidence
                    .map(|evidence| (*evidence).into_candidate().map(Box::new))
                    .transpose()?,
            }),
        }
    }
}

/// Requested and realized coverage for exactly one tier.
///
/// Fields are private and the trusted type is Serialize-only. Wire claims deserialize only as
/// [`UntrustedTierCoverageV1`] and acquire authority solely by exact comparison with a fresh
/// [`TrustedTierCoverageContextV1`] recomputation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TierCoverageV1 {
    pub(super) schema_version: u16,
    pub(super) tier: CoverageTierV1,
    pub(super) requested: TierCoverageRequestV1,
    pub(super) realized: TierCoverageRealizationV1,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TierCoverageWireV1 {
    schema_version: u16,
    tier: CoverageTierV1,
    requested: TierCoverageRequestWireV1,
    realized: TierCoverageRealizationWireV1,
}

impl TierCoverageWireV1 {
    pub(super) fn into_candidate(self) -> SearchResult<TierCoverageV1> {
        let coverage = TierCoverageV1 {
            schema_version: self.schema_version,
            tier: self.tier,
            requested: self.requested.into_candidate()?,
            realized: self.realized.into_candidate()?,
        };
        coverage.validate()?;
        Ok(coverage)
    }
}

impl TierCoverageV1 {
    /// Construct an explicitly unrequested/missing tier.
    #[must_use]
    pub const fn not_requested(tier: CoverageTierV1) -> Self {
        Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            tier,
            requested: TierCoverageRequestV1::MissingTier,
            realized: TierCoverageRealizationV1::MissingTier,
        }
    }

    /// Construct a requested tier whose index is missing.
    ///
    /// # Errors
    ///
    /// Rejects malformed space/scope receipts, duplicate members, or unbounded member ids.
    pub fn requested_missing(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
    ) -> SearchResult<Self> {
        Self::requested_without_verified_realization(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TierCoverageRealizationV1::MissingTier,
        )
    }

    /// Construct a requested tier with explicitly unknown realized coverage.
    ///
    /// # Errors
    ///
    /// Rejects malformed space/scope receipts, duplicate members, or unbounded member ids.
    pub fn unknown(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        reason: CoverageUnknownReasonV1,
    ) -> SearchResult<Self> {
        Self::requested_without_verified_realization(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TierCoverageRealizationV1::Unknown { reason },
        )
    }

    /// Construct a requested tier with an unverified scalar observation.
    ///
    /// `observed_indexed_count` remains diagnostic only and never contributes to topology.
    ///
    /// # Errors
    ///
    /// Rejects malformed space/scope receipts, duplicate members, or unbounded member ids.
    pub fn unverified(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    ) -> SearchResult<Self> {
        Self::requested_without_verified_realization(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TierCoverageRealizationV1::Unverified {
                reason,
                observed_indexed_count,
            },
        )
    }

    fn requested_without_verified_realization(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        realized: TierCoverageRealizationV1,
    ) -> SearchResult<Self> {
        let live = prepare_coverage_members(tier, space, basis, live_scope, live_members)?.witness;
        let coverage = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            tier,
            requested: TierCoverageRequestV1::Requested { live },
            realized,
        };
        coverage.validate()?;
        Ok(coverage)
    }

    /// Derive a verified relation from requested live and realized indexed member collections.
    ///
    /// Canonical document membership is order-independent. Ordered shard membership preserves
    /// order in each witness while set comparison remains duplicate-free. When scope differs,
    /// `cross_scope_authority_fingerprint` is mandatory and is bound into exact set evidence.
    /// That fingerprint must come from retained independent owner authority: payload bytes and a
    /// digest computed by the claimant are not attestation and must never be passed here during
    /// wire promotion.
    /// A cross-scope exact-complete claim is rejected because generation/corpus equality cannot
    /// be inferred from equal member names.
    ///
    /// # Errors
    ///
    /// Rejects malformed receipts, duplicates, checked-count overflow, missing/redundant
    /// cross-scope authority, and impossible cross-scope complete claims.
    #[allow(clippy::too_many_arguments)]
    pub fn verified(
        tier: CoverageTierV1,
        space: &EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        indexed_scope: CoverageScopeV1,
        indexed_members: Vec<String>,
        cross_scope_authority_fingerprint: Option<String>,
    ) -> SearchResult<Self> {
        let live = prepare_coverage_members(tier, space, basis, live_scope, live_members)?;
        let indexed = prepare_coverage_members(tier, space, basis, indexed_scope, indexed_members)?;
        let counts = CoverageCountsV1::from_member_sets(&live.members, &indexed.members)?;
        let relation = derive_coverage_relation(counts);
        let scopes_match = live.witness.scope == indexed.witness.scope;
        let cross_scope_evidence = match (scopes_match, cross_scope_authority_fingerprint) {
            (true, None) => None,
            (true, Some(_)) => {
                return Err(coverage_error(
                    "coverage.cross_scope_authority_fingerprint",
                    "redundant-same-scope-authority",
                    "must be absent when generation and corpus scopes are identical",
                ));
            }
            (false, None) => {
                return Err(coverage_error(
                    "coverage.cross_scope_authority_fingerprint",
                    "missing",
                    "different generation/corpus scopes require explicit comparison evidence",
                ));
            }
            (false, Some(authority)) => {
                if relation == CoverageRelationV1::Complete {
                    return Err(coverage_error(
                        "coverage.relation",
                        relation.code(),
                        "complete coverage requires identical generation and corpus scopes",
                    ));
                }
                Some(Box::new(CrossScopeCoverageEvidenceV1::build(
                    &live, &indexed, authority,
                )?))
            }
        };
        let coverage = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            tier,
            requested: TierCoverageRequestV1::Requested { live: live.witness },
            realized: TierCoverageRealizationV1::Verified {
                indexed: Box::new(indexed.witness),
                relation,
                counts,
                cross_scope_evidence,
            },
        };
        coverage.validate()?;
        Ok(coverage)
    }

    pub(super) fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.schema_version", self.schema_version)?;
        match (&self.requested, &self.realized) {
            (TierCoverageRequestV1::MissingTier, TierCoverageRealizationV1::MissingTier) => {}
            (TierCoverageRequestV1::MissingTier, _) => {
                return Err(coverage_error(
                    "coverage.realized",
                    "present-for-unrequested-tier",
                    "realized coverage requires an independent requested-tier witness",
                ));
            }
            (TierCoverageRequestV1::Requested { live }, realized) => {
                live.validate()?;
                if live.tier != self.tier {
                    return Err(coverage_error(
                        "coverage.requested.live.tier",
                        live.tier.code(),
                        "a tier cannot borrow the other tier's live witness",
                    ));
                }
                if let TierCoverageRealizationV1::Verified {
                    indexed,
                    relation,
                    counts,
                    cross_scope_evidence,
                } = realized
                {
                    indexed.validate()?;
                    if indexed.tier != self.tier {
                        return Err(coverage_error(
                            "coverage.realized.indexed.tier",
                            indexed.tier.code(),
                            "a tier cannot borrow the other tier's indexed witness",
                        ));
                    }
                    if indexed.space_fingerprint != live.space_fingerprint
                        || indexed.space_kind != live.space_kind
                    {
                        return Err(coverage_error(
                            "coverage.realized.indexed.space",
                            "space-binding-mismatch",
                            "requested and realized witnesses must bind the same validated embedding space",
                        ));
                    }
                    if indexed.basis != live.basis {
                        return Err(coverage_error(
                            "coverage.realized.indexed.basis",
                            indexed.basis.code(),
                            "requested and realized witnesses must use the same member basis",
                        ));
                    }
                    if counts.live_count != live.member_count
                        || counts.indexed_count != indexed.member_count
                    {
                        return Err(coverage_error(
                            "coverage.counts",
                            "witness-count-mismatch",
                            "live/indexed counts must equal their independently derived witnesses",
                        ));
                    }
                    validate_relation_counts(*relation, *counts)?;
                    let scopes_match = live.scope == indexed.scope;
                    match (scopes_match, cross_scope_evidence) {
                        (true, None) => {}
                        (true, Some(_)) => {
                            return Err(coverage_error(
                                "coverage.cross_scope_evidence",
                                "present-for-same-scope",
                                "same-scope coverage must not carry cross-scope evidence",
                            ));
                        }
                        (false, None) => {
                            return Err(coverage_error(
                                "coverage.cross_scope_evidence",
                                "missing",
                                "different generation/corpus scopes are unverified without evidence",
                            ));
                        }
                        (false, Some(evidence)) => {
                            evidence.validate()?;
                            if *relation == CoverageRelationV1::Complete {
                                return Err(coverage_error(
                                    "coverage.relation",
                                    relation.code(),
                                    "complete coverage requires identical generation/corpus scopes",
                                ));
                            }
                            if evidence.live_scope_fingerprint != live.scope.fingerprint()
                                || evidence.indexed_scope_fingerprint != indexed.scope.fingerprint()
                                || evidence.live_set_fingerprint != live.set_fingerprint
                                || evidence.indexed_set_fingerprint != indexed.set_fingerprint
                                || evidence.live_witness_fingerprint != live.fingerprint()
                                || evidence.indexed_witness_fingerprint != indexed.fingerprint()
                            {
                                return Err(coverage_error(
                                    "coverage.cross_scope_evidence",
                                    "binding-mismatch",
                                    "evidence does not bind the exact live and indexed witnesses",
                                ));
                            }
                        }
                    }
                    if *relation == CoverageRelationV1::Complete
                        && (live.set_fingerprint != indexed.set_fingerprint
                            || live.membership_fingerprint != indexed.membership_fingerprint)
                    {
                        return Err(coverage_error(
                            "coverage.relation",
                            relation.code(),
                            "complete coverage requires exact same-scope membership witnesses",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Tier role of this independent coverage value.
    #[must_use]
    pub const fn tier(&self) -> CoverageTierV1 {
        self.tier
    }

    /// Expected query/index mathematical-space fingerprint, absent only when unrequested.
    #[must_use]
    pub fn expected_space_fingerprint(&self) -> Option<&str> {
        self.live_witness()
            .map(CoverageWitnessV1::space_fingerprint)
    }

    /// Requested coverage state.
    #[must_use]
    pub const fn requested(&self) -> &TierCoverageRequestV1 {
        &self.requested
    }

    /// Realized coverage state.
    #[must_use]
    pub const fn realized(&self) -> &TierCoverageRealizationV1 {
        &self.realized
    }

    /// Derived relation/counts when and only when realized coverage is verified.
    #[must_use]
    pub const fn verified_facts(&self) -> Option<(CoverageRelationV1, CoverageCountsV1)> {
        match self.realized {
            TierCoverageRealizationV1::Verified {
                relation, counts, ..
            } => Some((relation, counts)),
            TierCoverageRealizationV1::MissingTier
            | TierCoverageRealizationV1::Unknown { .. }
            | TierCoverageRealizationV1::Unverified { .. } => None,
        }
    }

    /// Requested live witness, absent only when the tier was not requested.
    #[must_use]
    pub const fn live_witness(&self) -> Option<&CoverageWitnessV1> {
        match &self.requested {
            TierCoverageRequestV1::Requested { live } => Some(live),
            TierCoverageRequestV1::MissingTier => None,
        }
    }

    /// Realized indexed witness, present only for verified coverage.
    #[must_use]
    pub fn indexed_witness(&self) -> Option<&CoverageWitnessV1> {
        match &self.realized {
            TierCoverageRealizationV1::Verified { indexed, .. } => Some(indexed.as_ref()),
            TierCoverageRealizationV1::MissingTier
            | TierCoverageRealizationV1::Unknown { .. }
            | TierCoverageRealizationV1::Unverified { .. } => None,
        }
    }

    /// Explicit cross-scope comparison evidence, when generation/corpus differs.
    #[must_use]
    pub fn cross_scope_evidence(&self) -> Option<&CrossScopeCoverageEvidenceV1> {
        match &self.realized {
            TierCoverageRealizationV1::Verified {
                cross_scope_evidence,
                ..
            } => cross_scope_evidence.as_deref(),
            TierCoverageRealizationV1::MissingTier
            | TierCoverageRealizationV1::Unknown { .. }
            | TierCoverageRealizationV1::Unverified { .. } => None,
        }
    }

    /// Stable domain-separated digest of all requested and realized facts.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.tier-coverage.v1");
        encoder.u16(self.schema_version);
        encoder.text(self.tier.code());
        encode_coverage_request(&mut encoder, &self.requested);
        encode_coverage_realization(&mut encoder, &self.realized);
        encoder.fingerprint()
    }

    pub(super) const fn is_requested(&self) -> bool {
        matches!(self.requested, TierCoverageRequestV1::Requested { .. })
    }
}

#[derive(Clone)]
enum TrustedTierCoverageRealizationContextV1 {
    Missing,
    Unknown(CoverageUnknownReasonV1),
    Unverified {
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    },
    Verified {
        indexed_scope: CoverageScopeV1,
        indexed_members: Vec<String>,
        cross_scope_authority_fingerprint: Option<String>,
    },
}

/// Retained owner authority used to validate a serialized tier-coverage claim.
///
/// The context owns the validated mathematical space and the exact member
/// collections needed to recompute every digest, relation, count, and evidence
/// binding. It deliberately implements neither serialization nor deserialization;
/// its custom debug output never exposes member identifiers or producer identity.
#[derive(Clone)]
pub struct TrustedTierCoverageContextV1 {
    tier: CoverageTierV1,
    space: Option<EmbeddingSpaceIdentityV1>,
    basis: Option<CoverageBasisV1>,
    live_scope: Option<CoverageScopeV1>,
    live_members: Vec<String>,
    realization: TrustedTierCoverageRealizationContextV1,
}

impl fmt::Debug for TrustedTierCoverageContextV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustedTierCoverageContextV1")
            .field("tier", &self.tier)
            .field(
                "space_fingerprint",
                &self
                    .space
                    .as_ref()
                    .map(EmbeddingSpaceIdentityV1::fingerprint),
            )
            .field("basis", &self.basis)
            .field("live_member_count", &self.live_members.len())
            .field(
                "realization",
                &match self.realization {
                    TrustedTierCoverageRealizationContextV1::Missing => "missing",
                    TrustedTierCoverageRealizationContextV1::Unknown(_) => "unknown",
                    TrustedTierCoverageRealizationContextV1::Unverified { .. } => "unverified",
                    TrustedTierCoverageRealizationContextV1::Verified { .. } => "verified",
                },
            )
            .finish_non_exhaustive()
    }
}

impl TrustedTierCoverageContextV1 {
    /// Retain authority for an explicitly unrequested tier.
    #[must_use]
    pub const fn not_requested(tier: CoverageTierV1) -> Self {
        Self {
            tier,
            space: None,
            basis: None,
            live_scope: None,
            live_members: Vec::new(),
            realization: TrustedTierCoverageRealizationContextV1::Missing,
        }
    }

    /// Retain authority for a requested tier whose index is missing.
    ///
    /// # Errors
    ///
    /// Rejects malformed space or scope identity and invalid live membership.
    pub fn requested_missing(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
    ) -> SearchResult<Self> {
        Self::requested(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TrustedTierCoverageRealizationContextV1::Missing,
        )
    }

    /// Retain authority for a requested tier with unknown realized coverage.
    ///
    /// # Errors
    ///
    /// Rejects malformed space or scope identity and invalid live membership.
    pub fn unknown(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        reason: CoverageUnknownReasonV1,
    ) -> SearchResult<Self> {
        Self::requested(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TrustedTierCoverageRealizationContextV1::Unknown(reason),
        )
    }

    /// Retain authority for a requested tier with an unverified observation.
    ///
    /// # Errors
    ///
    /// Rejects malformed space or scope identity, invalid live membership, and
    /// impossible observed counts.
    pub fn unverified(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        reason: CoverageUnverifiedReasonV1,
        observed_indexed_count: Option<u64>,
    ) -> SearchResult<Self> {
        Self::requested(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TrustedTierCoverageRealizationContextV1::Unverified {
                reason,
                observed_indexed_count,
            },
        )
    }

    /// Retain all authority needed to recompute verified coverage.
    ///
    /// A cross-scope fingerprint must be independently sourced from the owner
    /// that authorized comparison. Copying it from an untrusted receipt would
    /// make the caller self-attest and is not a valid promotion workflow.
    ///
    /// # Errors
    ///
    /// Rejects malformed identity, invalid membership, unauthorized cross-scope
    /// comparison, or impossible set algebra.
    #[allow(clippy::too_many_arguments)]
    pub fn verified(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        indexed_scope: CoverageScopeV1,
        indexed_members: Vec<String>,
        cross_scope_authority_fingerprint: Option<String>,
    ) -> SearchResult<Self> {
        Self::requested(
            tier,
            space,
            basis,
            live_scope,
            live_members,
            TrustedTierCoverageRealizationContextV1::Verified {
                indexed_scope,
                indexed_members,
                cross_scope_authority_fingerprint,
            },
        )
    }

    fn requested(
        tier: CoverageTierV1,
        space: EmbeddingSpaceIdentityV1,
        basis: CoverageBasisV1,
        live_scope: CoverageScopeV1,
        live_members: Vec<String>,
        realization: TrustedTierCoverageRealizationContextV1,
    ) -> SearchResult<Self> {
        let context = Self {
            tier,
            space: Some(space),
            basis: Some(basis),
            live_scope: Some(live_scope),
            live_members,
            realization,
        };
        context.recompute()?;
        Ok(context)
    }

    /// Recompute the trusted coverage value from retained authority.
    ///
    /// # Errors
    ///
    /// Rejects an internally inconsistent context or any identity, membership,
    /// evidence, or set-algebra invariant that fails fresh validation.
    pub fn recompute(&self) -> SearchResult<TierCoverageV1> {
        let Some(space) = self.space.as_ref() else {
            if self.basis.is_none()
                && self.live_scope.is_none()
                && self.live_members.is_empty()
                && matches!(
                    self.realization,
                    TrustedTierCoverageRealizationContextV1::Missing
                )
            {
                return Ok(TierCoverageV1::not_requested(self.tier));
            }
            return Err(coverage_error(
                "coverage.context",
                "incomplete-unrequested-context",
                "unrequested authority must not retain requested-tier facts",
            ));
        };
        let basis = self.basis.ok_or_else(|| {
            coverage_error(
                "coverage.context.basis",
                "missing",
                "requested authority requires a coverage basis",
            )
        })?;
        let live_scope = self.live_scope.clone().ok_or_else(|| {
            coverage_error(
                "coverage.context.live_scope",
                "missing",
                "requested authority requires a live scope",
            )
        })?;
        match &self.realization {
            TrustedTierCoverageRealizationContextV1::Missing => TierCoverageV1::requested_missing(
                self.tier,
                space,
                basis,
                live_scope,
                self.live_members.clone(),
            ),
            TrustedTierCoverageRealizationContextV1::Unknown(reason) => TierCoverageV1::unknown(
                self.tier,
                space,
                basis,
                live_scope,
                self.live_members.clone(),
                *reason,
            ),
            TrustedTierCoverageRealizationContextV1::Unverified {
                reason,
                observed_indexed_count,
            } => TierCoverageV1::unverified(
                self.tier,
                space,
                basis,
                live_scope,
                self.live_members.clone(),
                *reason,
                *observed_indexed_count,
            ),
            TrustedTierCoverageRealizationContextV1::Verified {
                indexed_scope,
                indexed_members,
                cross_scope_authority_fingerprint,
            } => TierCoverageV1::verified(
                self.tier,
                space,
                basis,
                live_scope,
                self.live_members.clone(),
                indexed_scope.clone(),
                indexed_members.clone(),
                cross_scope_authority_fingerprint.clone(),
            ),
        }
    }
}

/// Syntactically checked wire claim that has not yet acquired coverage authority.
#[derive(Deserialize)]
#[serde(transparent)]
pub struct UntrustedTierCoverageV1(TierCoverageWireV1);

impl UntrustedTierCoverageV1 {
    /// Promote only by exact comparison with a fresh owner-backed recomputation.
    ///
    /// # Errors
    ///
    /// Rejects malformed wire data and any claim that differs from the trusted
    /// owner context after exact recomputation.
    pub fn validate_against(
        self,
        context: &TrustedTierCoverageContextV1,
    ) -> SearchResult<TierCoverageV1> {
        let candidate = self.0.into_candidate()?;
        let recomputed = context.recompute()?;
        if candidate != recomputed {
            return Err(coverage_error(
                "coverage.untrusted",
                "authority-mismatch",
                "serialized coverage does not equal owner-backed recomputation",
            ));
        }
        Ok(recomputed)
    }
}
