use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::error::SearchResult;
use crate::generation::{EmbeddingSpaceIdentityV1, EmbeddingSpaceKindV1};
use crate::recovery_plan::COMPLETE_COVERAGE_PPM;

use super::coverage_codec::{
    CoverageEncoder, coverage_error, embedding_space_kind_code, fingerprint_member_set,
    validate_coverage_schema, validate_coverage_sha256,
};

// ---------------------------------------------------------------------------
// Independent requested/realized per-tier coverage (bd-9xuj C1 AC3/AC4)
// ---------------------------------------------------------------------------

/// Schema version for the closed per-tier coverage and topology receipts.
pub const TIER_COVERAGE_SCHEMA_VERSION_V1: u16 = 1;
const MAX_COVERAGE_MEMBER_BYTES: usize = 4_096;

/// Semantic role of one independently witnessed retrieval tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageTierV1 {
    /// Latency-oriented embedding/index tier.
    Fast,
    /// Quality-oriented embedding/index tier.
    Quality,
}

impl CoverageTierV1 {
    pub(super) const fn code(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Quality => "quality",
        }
    }
}

/// Canonical member representation used to witness one tier's coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageBasisV1 {
    /// An order-independent, duplicate-free canonical document-id set.
    CanonicalDocuments,
    /// A duplicate-free shard-id sequence whose order is identity material.
    OrderedShards,
}

impl CoverageBasisV1 {
    pub(super) const fn code(self) -> &'static str {
        match self {
            Self::CanonicalDocuments => "canonical_documents",
            Self::OrderedShards => "ordered_shards",
        }
    }
}

/// Exact set relationship between realized indexed members and requested live members.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageRelationV1 {
    /// Both the requested live set and realized indexed set are empty.
    ZeroLive,
    /// Realized and requested sets are exactly equal and non-empty.
    Complete,
    /// The realized set is a proper subset of the requested live set.
    Subset,
    /// The realized set is a proper superset and therefore contains stale/foreign members.
    Superset,
    /// Both sides have unique members and a non-empty proper intersection.
    Overlap,
    /// Both sides are non-empty and have no member in common.
    Disjoint,
}

impl CoverageRelationV1 {
    pub(super) const fn code(self) -> &'static str {
        match self {
            Self::ZeroLive => "zero_live",
            Self::Complete => "complete",
            Self::Subset => "subset",
            Self::Superset => "superset",
            Self::Overlap => "overlap",
            Self::Disjoint => "disjoint",
        }
    }
}

/// Closed reason why realized coverage is not known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageUnknownReasonV1 {
    /// No immutable owner is available to make a coverage statement.
    OwnerUnavailable,
    /// A legacy artifact has no admissible identity or coverage witness.
    LegacyUnidentified,
    /// The active generation cannot be resolved atomically.
    GenerationUnresolved,
    /// The canonical live corpus cannot be resolved.
    CorpusUnresolved,
}

/// Closed reason why observed coverage facts are present but unverified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageUnverifiedReasonV1 {
    /// No canonical member witness accompanies the observed count.
    WitnessMissing,
    /// A supplied witness digest did not verify against retained authority.
    WitnessDigestMismatch,
    /// Counts were observed through a source that cannot attest uniqueness.
    CountsUntrusted,
    /// Different generation/corpus scopes have no explicit comparison evidence.
    CrossScopeUnproven,
}

/// Immutable generation and corpus facts that scope one member witness.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CoverageScopeV1 {
    generation_fingerprint: String,
    corpus_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageScopeWireV1 {
    generation_fingerprint: String,
    corpus_fingerprint: String,
}

impl<'de> Deserialize<'de> for CoverageScopeV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = CoverageScopeWireV1::deserialize(deserializer)?;
        Self::new(wire.generation_fingerprint, wire.corpus_fingerprint)
            .map_err(serde::de::Error::custom)
    }
}

impl CoverageScopeV1 {
    /// Construct a scope from exact lowercase SHA-256 generation and corpus receipts.
    ///
    /// # Errors
    ///
    /// Rejects malformed or non-canonical fingerprints.
    pub fn new(
        generation_fingerprint: impl Into<String>,
        corpus_fingerprint: impl Into<String>,
    ) -> SearchResult<Self> {
        let scope = Self {
            generation_fingerprint: generation_fingerprint.into(),
            corpus_fingerprint: corpus_fingerprint.into(),
        };
        scope.validate()?;
        Ok(scope)
    }

    fn validate(&self) -> SearchResult<()> {
        validate_coverage_sha256(
            "coverage.scope.generation_fingerprint",
            &self.generation_fingerprint,
        )?;
        validate_coverage_sha256(
            "coverage.scope.corpus_fingerprint",
            &self.corpus_fingerprint,
        )
    }

    /// Exact generation receipt.
    #[must_use]
    pub fn generation_fingerprint(&self) -> &str {
        &self.generation_fingerprint
    }

    /// Exact corpus receipt.
    #[must_use]
    pub fn corpus_fingerprint(&self) -> &str {
        &self.corpus_fingerprint
    }

    /// Domain-separated stable fingerprint of both scope facts.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.coverage-scope.v1");
        encoder.text(&self.generation_fingerprint);
        encoder.text(&self.corpus_fingerprint);
        encoder.fingerprint()
    }
}

/// Bounded, tier-tagged witness of a canonical document set or ordered shard sequence.
///
/// Only digests and counts are retained: query/document text and raw member identifiers never
/// enter logs or serialized diagnostics. The tier role participates in [`Self::fingerprint`], so
/// a fast witness cannot be borrowed as quality evidence even when both cover the same members.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CoverageWitnessV1 {
    pub(super) schema_version: u16,
    pub(super) tier: CoverageTierV1,
    pub(super) space_fingerprint: String,
    pub(super) space_kind: EmbeddingSpaceKindV1,
    pub(super) basis: CoverageBasisV1,
    pub(super) scope: CoverageScopeV1,
    pub(super) member_count: u64,
    pub(super) membership_fingerprint: String,
    pub(super) set_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CoverageWitnessWireV1 {
    schema_version: u16,
    tier: CoverageTierV1,
    space_fingerprint: String,
    space_kind: EmbeddingSpaceKindV1,
    basis: CoverageBasisV1,
    scope: CoverageScopeV1,
    member_count: u64,
    membership_fingerprint: String,
    set_fingerprint: String,
}

impl CoverageWitnessWireV1 {
    pub(super) fn into_candidate(self) -> SearchResult<CoverageWitnessV1> {
        let witness = CoverageWitnessV1 {
            schema_version: self.schema_version,
            tier: self.tier,
            space_fingerprint: self.space_fingerprint,
            space_kind: self.space_kind,
            basis: self.basis,
            scope: self.scope,
            member_count: self.member_count,
            membership_fingerprint: self.membership_fingerprint,
            set_fingerprint: self.set_fingerprint,
        };
        witness.validate()?;
        Ok(witness)
    }
}

impl CoverageWitnessV1 {
    pub(super) fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.witness.schema_version", self.schema_version)?;
        self.scope.validate()?;
        validate_coverage_sha256(
            "coverage.witness.space_fingerprint",
            &self.space_fingerprint,
        )?;
        validate_coverage_sha256(
            "coverage.witness.membership_fingerprint",
            &self.membership_fingerprint,
        )?;
        validate_coverage_sha256("coverage.witness.set_fingerprint", &self.set_fingerprint)
    }

    /// Tier role cryptographically bound by this witness.
    #[must_use]
    pub const fn tier(&self) -> CoverageTierV1 {
        self.tier
    }

    /// Validated mathematical-space fingerprint that produced this tier.
    #[must_use]
    pub fn space_fingerprint(&self) -> &str {
        &self.space_fingerprint
    }

    /// Explicit semantic versus hash-control classification of that space.
    #[must_use]
    pub const fn space_kind(&self) -> EmbeddingSpaceKindV1 {
        self.space_kind
    }

    /// Member representation bound by this witness.
    #[must_use]
    pub const fn basis(&self) -> CoverageBasisV1 {
        self.basis
    }

    /// Immutable generation/corpus scope.
    #[must_use]
    pub const fn scope(&self) -> &CoverageScopeV1 {
        &self.scope
    }

    /// Duplicate-free member count derived at construction.
    #[must_use]
    pub const fn member_count(&self) -> u64 {
        self.member_count
    }

    /// Order-sensitive membership digest (document ids are canonically sorted first).
    #[must_use]
    pub fn membership_fingerprint(&self) -> &str {
        &self.membership_fingerprint
    }

    /// Order-independent set digest used for checked relation evidence.
    #[must_use]
    pub fn set_fingerprint(&self) -> &str {
        &self.set_fingerprint
    }

    /// Domain-separated digest of every semantic witness field, including tier role.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.coverage-witness.v1");
        encoder.u16(self.schema_version);
        encoder.text(self.tier.code());
        encoder.text(&self.space_fingerprint);
        encoder.text(embedding_space_kind_code(self.space_kind));
        encoder.text(self.basis.code());
        encoder.text(&self.scope.fingerprint());
        encoder.u64(self.member_count);
        encoder.text(&self.membership_fingerprint);
        encoder.text(&self.set_fingerprint);
        encoder.fingerprint()
    }
}

pub(super) struct PreparedCoverageMembers {
    pub(super) witness: CoverageWitnessV1,
    pub(super) members: BTreeSet<String>,
}

pub(super) fn prepare_coverage_members(
    tier: CoverageTierV1,
    space: &EmbeddingSpaceIdentityV1,
    basis: CoverageBasisV1,
    scope: CoverageScopeV1,
    members: Vec<String>,
) -> SearchResult<PreparedCoverageMembers> {
    space.validate()?;
    scope.validate()?;
    let member_count = u64::try_from(members.len()).map_err(|_| {
        coverage_error(
            "coverage.member_count",
            "redacted-overflow",
            "member count does not fit in u64",
        )
    })?;
    let mut membership_encoder = match basis {
        CoverageBasisV1::CanonicalDocuments => {
            CoverageEncoder::new(b"frankensearch.coverage-canonical-documents.v1")
        }
        CoverageBasisV1::OrderedShards => {
            CoverageEncoder::new(b"frankensearch.coverage-ordered-shards.v1")
        }
    };
    membership_encoder.u64(member_count);
    let mut unique = BTreeSet::new();
    for (index, member) in members.into_iter().enumerate() {
        if member.is_empty() || member.len() > MAX_COVERAGE_MEMBER_BYTES {
            return Err(coverage_error(
                "coverage.members",
                &format!("redacted-member-at-index-{index}"),
                "member identifiers must be non-empty and at most 4096 bytes",
            ));
        }
        if basis == CoverageBasisV1::OrderedShards {
            membership_encoder.text(&member);
        }
        if !unique.insert(member) {
            return Err(coverage_error(
                "coverage.members",
                &format!("redacted-duplicate-at-index-{index}"),
                "duplicate member identifiers are forbidden because they inflate coverage",
            ));
        }
    }

    let mut set_encoder = CoverageEncoder::new(b"frankensearch.coverage-member-set.v1");
    set_encoder.u64(member_count);
    for member in &unique {
        set_encoder.text(member);
    }
    let set_fingerprint = set_encoder.fingerprint();

    if basis == CoverageBasisV1::CanonicalDocuments {
        for member in &unique {
            membership_encoder.text(member);
        }
    }

    Ok(PreparedCoverageMembers {
        witness: CoverageWitnessV1 {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            tier,
            space_fingerprint: space.fingerprint(),
            space_kind: space.kind,
            basis,
            scope,
            member_count,
            membership_fingerprint: membership_encoder.fingerprint(),
            set_fingerprint,
        },
        members: unique,
    })
}

/// Checked scalar facts derived from two duplicate-free member sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct CoverageCountsV1 {
    pub(super) live_count: u64,
    pub(super) indexed_count: u64,
    pub(super) intersection_count: u64,
    pub(super) union_count: u64,
    pub(super) coverage_ppm: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageCountsWireV1 {
    live_count: u64,
    indexed_count: u64,
    intersection_count: u64,
    union_count: u64,
    coverage_ppm: u32,
}

impl<'de> Deserialize<'de> for CoverageCountsV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = CoverageCountsWireV1::deserialize(deserializer)?;
        Self::checked(
            wire.live_count,
            wire.indexed_count,
            wire.intersection_count,
            wire.union_count,
        )
        .and_then(|counts| {
            if counts.coverage_ppm == wire.coverage_ppm {
                Ok(counts)
            } else {
                Err(coverage_error(
                    "coverage.counts.coverage_ppm",
                    &wire.coverage_ppm.to_string(),
                    "must be derived from intersection_count/live_count",
                ))
            }
        })
        .map_err(serde::de::Error::custom)
    }
}

impl CoverageCountsV1 {
    pub(super) fn checked(
        live_count: u64,
        indexed_count: u64,
        intersection_count: u64,
        union_count: u64,
    ) -> SearchResult<Self> {
        if intersection_count > live_count || intersection_count > indexed_count {
            return Err(coverage_error(
                "coverage.counts.intersection_count",
                &intersection_count.to_string(),
                "cannot exceed either live_count or indexed_count",
            ));
        }
        let expected_union = u128::from(live_count)
            .checked_add(u128::from(indexed_count))
            .and_then(|sum| sum.checked_sub(u128::from(intersection_count)))
            .ok_or_else(|| {
                coverage_error(
                    "coverage.counts.union_count",
                    "redacted-overflow",
                    "checked set-union arithmetic overflowed",
                )
            })?;
        let expected_union = u64::try_from(expected_union).map_err(|_| {
            coverage_error(
                "coverage.counts.union_count",
                "redacted-overflow",
                "set union cannot be represented by u64",
            )
        })?;
        if union_count != expected_union {
            return Err(coverage_error(
                "coverage.counts.union_count",
                &union_count.to_string(),
                "must equal live_count + indexed_count - intersection_count",
            ));
        }
        let coverage_ppm = if live_count == 0 {
            0
        } else {
            let scaled = u128::from(intersection_count) * u128::from(COMPLETE_COVERAGE_PPM);
            u32::try_from(scaled / u128::from(live_count)).map_err(|_| {
                coverage_error(
                    "coverage.counts.coverage_ppm",
                    "redacted-overflow",
                    "derived fixed-point coverage does not fit in u32",
                )
            })?
        };
        Ok(Self {
            live_count,
            indexed_count,
            intersection_count,
            union_count,
            coverage_ppm,
        })
    }

    pub(super) fn from_member_sets(
        live: &BTreeSet<String>,
        indexed: &BTreeSet<String>,
    ) -> SearchResult<Self> {
        let live_count = u64::try_from(live.len()).map_err(|_| {
            coverage_error(
                "coverage.counts.live_count",
                "redacted-overflow",
                "live member count does not fit in u64",
            )
        })?;
        let indexed_count = u64::try_from(indexed.len()).map_err(|_| {
            coverage_error(
                "coverage.counts.indexed_count",
                "redacted-overflow",
                "indexed member count does not fit in u64",
            )
        })?;
        let intersection_count =
            u64::try_from(live.intersection(indexed).count()).map_err(|_| {
                coverage_error(
                    "coverage.counts.intersection_count",
                    "redacted-overflow",
                    "intersection member count does not fit in u64",
                )
            })?;
        let union_count = u64::try_from(live.union(indexed).count()).map_err(|_| {
            coverage_error(
                "coverage.counts.union_count",
                "redacted-overflow",
                "union member count does not fit in u64",
            )
        })?;
        Self::checked(live_count, indexed_count, intersection_count, union_count)
    }

    /// Canonical live-member count.
    #[must_use]
    pub const fn live_count(self) -> u64 {
        self.live_count
    }

    /// Canonical indexed-member count.
    #[must_use]
    pub const fn indexed_count(self) -> u64 {
        self.indexed_count
    }

    /// Checked intersection count.
    #[must_use]
    pub const fn intersection_count(self) -> u64 {
        self.intersection_count
    }

    /// Checked union count.
    #[must_use]
    pub const fn union_count(self) -> u64 {
        self.union_count
    }

    /// Floor of `intersection/live * 1_000_000`, derived with u128 intermediates.
    #[must_use]
    pub const fn coverage_ppm(self) -> u32 {
        self.coverage_ppm
    }
}

/// Explicit comparison evidence required when requested and realized scopes differ.
///
/// The evidence binds both scopes, both exact set digests, the computed intersection and union,
/// and a retained authority fingerprint. It is never inferred from counts or another tier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CrossScopeCoverageEvidenceV1 {
    pub(super) schema_version: u16,
    pub(super) live_witness_fingerprint: String,
    pub(super) indexed_witness_fingerprint: String,
    pub(super) live_scope_fingerprint: String,
    pub(super) indexed_scope_fingerprint: String,
    pub(super) live_set_fingerprint: String,
    pub(super) indexed_set_fingerprint: String,
    pub(super) intersection_set_fingerprint: String,
    pub(super) union_set_fingerprint: String,
    pub(super) authority_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CrossScopeCoverageEvidenceWireV1 {
    schema_version: u16,
    live_witness_fingerprint: String,
    indexed_witness_fingerprint: String,
    live_scope_fingerprint: String,
    indexed_scope_fingerprint: String,
    live_set_fingerprint: String,
    indexed_set_fingerprint: String,
    intersection_set_fingerprint: String,
    union_set_fingerprint: String,
    authority_fingerprint: String,
}

impl CrossScopeCoverageEvidenceWireV1 {
    pub(super) fn into_candidate(self) -> SearchResult<CrossScopeCoverageEvidenceV1> {
        let evidence = CrossScopeCoverageEvidenceV1 {
            schema_version: self.schema_version,
            live_witness_fingerprint: self.live_witness_fingerprint,
            indexed_witness_fingerprint: self.indexed_witness_fingerprint,
            live_scope_fingerprint: self.live_scope_fingerprint,
            indexed_scope_fingerprint: self.indexed_scope_fingerprint,
            live_set_fingerprint: self.live_set_fingerprint,
            indexed_set_fingerprint: self.indexed_set_fingerprint,
            intersection_set_fingerprint: self.intersection_set_fingerprint,
            union_set_fingerprint: self.union_set_fingerprint,
            authority_fingerprint: self.authority_fingerprint,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl CrossScopeCoverageEvidenceV1 {
    pub(super) fn build(
        live: &PreparedCoverageMembers,
        indexed: &PreparedCoverageMembers,
        authority_fingerprint: String,
    ) -> SearchResult<Self> {
        validate_coverage_sha256(
            "coverage.cross_scope.authority_fingerprint",
            &authority_fingerprint,
        )?;
        let intersection = live
            .members
            .intersection(&indexed.members)
            .cloned()
            .collect::<BTreeSet<_>>();
        let union = live
            .members
            .union(&indexed.members)
            .cloned()
            .collect::<BTreeSet<_>>();
        let evidence = Self {
            schema_version: TIER_COVERAGE_SCHEMA_VERSION_V1,
            live_witness_fingerprint: live.witness.fingerprint(),
            indexed_witness_fingerprint: indexed.witness.fingerprint(),
            live_scope_fingerprint: live.witness.scope.fingerprint(),
            indexed_scope_fingerprint: indexed.witness.scope.fingerprint(),
            live_set_fingerprint: live.witness.set_fingerprint.clone(),
            indexed_set_fingerprint: indexed.witness.set_fingerprint.clone(),
            intersection_set_fingerprint: fingerprint_member_set(
                b"frankensearch.coverage-intersection-set.v1",
                &intersection,
            )?,
            union_set_fingerprint: fingerprint_member_set(
                b"frankensearch.coverage-union-set.v1",
                &union,
            )?,
            authority_fingerprint,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub(super) fn validate(&self) -> SearchResult<()> {
        validate_coverage_schema("coverage.cross_scope.schema_version", self.schema_version)?;
        for (field, value) in [
            ("live_witness_fingerprint", &self.live_witness_fingerprint),
            (
                "indexed_witness_fingerprint",
                &self.indexed_witness_fingerprint,
            ),
            ("live_scope_fingerprint", &self.live_scope_fingerprint),
            ("indexed_scope_fingerprint", &self.indexed_scope_fingerprint),
            ("live_set_fingerprint", &self.live_set_fingerprint),
            ("indexed_set_fingerprint", &self.indexed_set_fingerprint),
            (
                "intersection_set_fingerprint",
                &self.intersection_set_fingerprint,
            ),
            ("union_set_fingerprint", &self.union_set_fingerprint),
            ("authority_fingerprint", &self.authority_fingerprint),
        ] {
            validate_coverage_sha256(&format!("coverage.cross_scope.{field}"), value)?;
        }
        if self.live_scope_fingerprint == self.indexed_scope_fingerprint {
            return Err(coverage_error(
                "coverage.cross_scope",
                "redundant-same-scope-evidence",
                "cross-scope evidence is valid only when generation or corpus differs",
            ));
        }
        Ok(())
    }

    /// Domain-separated fingerprint of the complete cross-scope comparison receipt.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut encoder = CoverageEncoder::new(b"frankensearch.cross-scope-coverage.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.live_witness_fingerprint);
        encoder.text(&self.indexed_witness_fingerprint);
        encoder.text(&self.live_scope_fingerprint);
        encoder.text(&self.indexed_scope_fingerprint);
        encoder.text(&self.live_set_fingerprint);
        encoder.text(&self.indexed_set_fingerprint);
        encoder.text(&self.intersection_set_fingerprint);
        encoder.text(&self.union_set_fingerprint);
        encoder.text(&self.authority_fingerprint);
        encoder.fingerprint()
    }
}
