use std::collections::BTreeSet;
use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::SearchError;
use crate::error::SearchResult;
use crate::generation::EmbeddingSpaceKindV1;
use crate::recovery_plan::COMPLETE_COVERAGE_PPM;

use super::RetrievalTopology;
use super::coverage_pair::{RealizedTierCoverageV1, RetrievalIntentV1};
use super::coverage_scope::{
    CoverageCountsV1, CoverageRelationV1, CoverageTierV1, CoverageUnknownReasonV1,
    CoverageUnverifiedReasonV1, TIER_COVERAGE_SCHEMA_VERSION_V1,
};
use super::coverage_tier::{TierCoverageRealizationV1, TierCoverageRequestV1, TierCoverageV1};
use super::query_embeddings::BoundQueryEmbedding;

pub(super) fn validate_tier_binding(
    coverage: &TierCoverageV1,
    embedding: Option<&BoundQueryEmbedding>,
) -> SearchResult<()> {
    match (coverage.is_requested(), embedding) {
        (false, None) => Ok(()),
        (false, Some(_)) => Err(coverage_error(
            "retrieval.bindings",
            coverage.tier.code(),
            "an unrequested tier cannot retain an extra query binding",
        )),
        (true, None) => Err(coverage_error(
            "retrieval.bindings",
            coverage.tier.code(),
            "a requested tier requires its own independently bound query embedding",
        )),
        (true, Some(embedding)) => {
            embedding.identity.validate()?;
            let expected = coverage.expected_space_fingerprint().ok_or_else(|| {
                coverage_error(
                    "retrieval.bindings",
                    coverage.tier.code(),
                    "requested coverage omitted its expected mathematical space",
                )
            })?;
            embedding.verify_space_identity(expected, coverage.tier.code())
        }
    }
}

pub(super) fn validate_binding_kinds(
    intent: RetrievalIntentV1,
    fast_kind: Option<EmbeddingSpaceKindV1>,
    quality_kind: Option<EmbeddingSpaceKindV1>,
    expected: EmbeddingSpaceKindV1,
) -> SearchResult<()> {
    if fast_kind.is_none() && quality_kind.is_none() {
        return Err(coverage_error(
            "retrieval.bindings",
            intent.code(),
            "vector retrieval intent requires at least one validated binding",
        ));
    }
    for (tier, kind) in [
        (CoverageTierV1::Fast, fast_kind),
        (CoverageTierV1::Quality, quality_kind),
    ] {
        if let Some(kind) = kind
            && kind != expected
        {
            return Err(coverage_error(
                &format!("retrieval.bindings.{}.space_kind", tier.code()),
                embedding_space_kind_code(kind),
                match expected {
                    EmbeddingSpaceKindV1::Semantic => {
                        "hash control cannot satisfy semantic availability"
                    }
                    EmbeddingSpaceKindV1::HashControl => {
                        "hash-control intent cannot relabel a learned semantic space"
                    }
                },
            ));
        }
    }
    Ok(())
}

pub(super) const fn embedding_space_kind_code(kind: EmbeddingSpaceKindV1) -> &'static str {
    match kind {
        EmbeddingSpaceKindV1::Semantic => "semantic",
        EmbeddingSpaceKindV1::HashControl => "hash_control",
    }
}

pub(super) fn derive_lossless_semantic_topology(
    fast: &RealizedTierCoverageV1,
    quality: &RealizedTierCoverageV1,
) -> Option<RetrievalTopology> {
    match (fast, quality) {
        (RealizedTierCoverageV1::Complete { .. }, RealizedTierCoverageV1::Complete { .. }) => {
            Some(RetrievalTopology::FullProgressive)
        }
        (
            RealizedTierCoverageV1::Complete { .. },
            RealizedTierCoverageV1::Partial { counts, .. },
        ) if counts.intersection_count > 0
            && counts.coverage_ppm > 0
            && counts.coverage_ppm < COMPLETE_COVERAGE_PPM =>
        {
            Some(RetrievalTopology::PartialQuality {
                coverage_ppm: counts.coverage_ppm,
            })
        }
        (RealizedTierCoverageV1::Complete { .. }, quality) if !quality.contributes() => {
            Some(RetrievalTopology::FastOnly)
        }
        (fast, RealizedTierCoverageV1::Complete { .. }) if !fast.contributes() => {
            Some(RetrievalTopology::QualityOnly)
        }
        (fast, quality) if !fast.contributes() && !quality.contributes() => {
            Some(RetrievalTopology::LexicalOnly)
        }
        _ => None,
    }
}

/// Canonical compatibility law shared by coverage derivation and recovery contracts.
#[must_use]
pub const fn retrieval_topology_fits_request(
    requested: RetrievalTopology,
    realized: RetrievalTopology,
) -> bool {
    match requested {
        RetrievalTopology::LexicalOnly => matches!(realized, RetrievalTopology::LexicalOnly),
        RetrievalTopology::FastOnly => matches!(
            realized,
            RetrievalTopology::FastOnly | RetrievalTopology::LexicalOnly
        ),
        RetrievalTopology::QualityOnly => matches!(
            realized,
            RetrievalTopology::QualityOnly
                | RetrievalTopology::PartialQuality { .. }
                | RetrievalTopology::LexicalOnly
        ),
        RetrievalTopology::FullProgressive => matches!(
            realized,
            RetrievalTopology::FullProgressive
                | RetrievalTopology::PartialQuality { .. }
                | RetrievalTopology::FastOnly
                | RetrievalTopology::QualityOnly
                | RetrievalTopology::LexicalOnly
        ),
        RetrievalTopology::HashControl => matches!(realized, RetrievalTopology::HashControl),
        RetrievalTopology::PartialQuality { .. } => false,
    }
}

pub(super) const fn derive_coverage_relation(counts: CoverageCountsV1) -> CoverageRelationV1 {
    if counts.live_count == 0 && counts.indexed_count == 0 {
        CoverageRelationV1::ZeroLive
    } else if counts.live_count == counts.indexed_count
        && counts.intersection_count == counts.live_count
    {
        CoverageRelationV1::Complete
    } else if counts.intersection_count == counts.indexed_count
        && counts.indexed_count < counts.live_count
    {
        CoverageRelationV1::Subset
    } else if counts.intersection_count == counts.live_count
        && counts.live_count < counts.indexed_count
    {
        CoverageRelationV1::Superset
    } else if counts.intersection_count == 0 {
        CoverageRelationV1::Disjoint
    } else {
        CoverageRelationV1::Overlap
    }
}

pub(super) fn validate_relation_counts(
    relation: CoverageRelationV1,
    counts: CoverageCountsV1,
) -> SearchResult<()> {
    let derived = derive_coverage_relation(counts);
    if relation != derived {
        return Err(coverage_error(
            "coverage.relation",
            relation.code(),
            &format!("checked counts derive relation {}", derived.code()),
        ));
    }
    if counts.coverage_ppm > COMPLETE_COVERAGE_PPM {
        return Err(coverage_error(
            "coverage.counts.coverage_ppm",
            &counts.coverage_ppm.to_string(),
            "fixed-point coverage cannot exceed one million PPM",
        ));
    }
    Ok(())
}

pub(super) fn fingerprint_member_set(
    domain: &[u8],
    members: &BTreeSet<String>,
) -> SearchResult<String> {
    let member_count = u64::try_from(members.len()).map_err(|_| {
        coverage_error(
            "coverage.member_count",
            "redacted-overflow",
            "member count does not fit in u64",
        )
    })?;
    let mut encoder = CoverageEncoder::new(domain);
    encoder.u64(member_count);
    for member in members {
        encoder.text(member);
    }
    Ok(encoder.fingerprint())
}

pub(super) fn encode_coverage_request(
    encoder: &mut CoverageEncoder,
    request: &TierCoverageRequestV1,
) {
    match request {
        TierCoverageRequestV1::MissingTier => encoder.u8(0),
        TierCoverageRequestV1::Requested { live } => {
            encoder.u8(1);
            encoder.text(&live.fingerprint());
        }
    }
}

pub(super) fn encode_coverage_realization(
    encoder: &mut CoverageEncoder,
    realization: &TierCoverageRealizationV1,
) {
    match realization {
        TierCoverageRealizationV1::MissingTier => encoder.u8(0),
        TierCoverageRealizationV1::Unknown { reason } => {
            encoder.u8(1);
            encoder.text(coverage_unknown_reason_code(*reason));
        }
        TierCoverageRealizationV1::Unverified {
            reason,
            observed_indexed_count,
        } => {
            encoder.u8(2);
            encoder.text(coverage_unverified_reason_code(*reason));
            encoder.option_u64(*observed_indexed_count);
        }
        TierCoverageRealizationV1::Verified {
            indexed,
            relation,
            counts,
            cross_scope_evidence,
        } => {
            encoder.u8(3);
            encoder.text(&indexed.fingerprint());
            encoder.text(relation.code());
            encoder.u64(counts.live_count);
            encoder.u64(counts.indexed_count);
            encoder.u64(counts.intersection_count);
            encoder.u64(counts.union_count);
            encoder.u32(counts.coverage_ppm);
            encoder.option_text(
                cross_scope_evidence
                    .as_ref()
                    .map(|evidence| evidence.fingerprint())
                    .as_deref(),
            );
        }
    }
}

pub(super) fn encode_realized_tier_coverage(
    encoder: &mut CoverageEncoder,
    realized: &RealizedTierCoverageV1,
) {
    match realized {
        RealizedTierCoverageV1::NotRequested => encoder.u8(0),
        RealizedTierCoverageV1::Missing => encoder.u8(1),
        RealizedTierCoverageV1::Unknown { reason } => {
            encoder.u8(2);
            encoder.text(coverage_unknown_reason_code(*reason));
        }
        RealizedTierCoverageV1::Unverified {
            reason,
            observed_indexed_count,
        } => {
            encoder.u8(3);
            encoder.text(coverage_unverified_reason_code(*reason));
            encoder.option_u64(*observed_indexed_count);
        }
        RealizedTierCoverageV1::ZeroLive { counts } => {
            encoder.u8(4);
            encode_coverage_counts(encoder, *counts);
        }
        RealizedTierCoverageV1::Complete { counts } => {
            encoder.u8(5);
            encode_coverage_counts(encoder, *counts);
        }
        RealizedTierCoverageV1::Partial { relation, counts } => {
            encoder.u8(6);
            encoder.text(relation.code());
            encode_coverage_counts(encoder, *counts);
        }
    }
}

pub(super) fn encode_coverage_counts(encoder: &mut CoverageEncoder, counts: CoverageCountsV1) {
    encoder.u64(counts.live_count);
    encoder.u64(counts.indexed_count);
    encoder.u64(counts.intersection_count);
    encoder.u64(counts.union_count);
    encoder.u32(counts.coverage_ppm);
}

pub(super) const fn coverage_unknown_reason_code(reason: CoverageUnknownReasonV1) -> &'static str {
    match reason {
        CoverageUnknownReasonV1::OwnerUnavailable => "owner_unavailable",
        CoverageUnknownReasonV1::LegacyUnidentified => "legacy_unidentified",
        CoverageUnknownReasonV1::GenerationUnresolved => "generation_unresolved",
        CoverageUnknownReasonV1::CorpusUnresolved => "corpus_unresolved",
    }
}

pub(super) const fn coverage_unverified_reason_code(
    reason: CoverageUnverifiedReasonV1,
) -> &'static str {
    match reason {
        CoverageUnverifiedReasonV1::WitnessMissing => "witness_missing",
        CoverageUnverifiedReasonV1::WitnessDigestMismatch => "witness_digest_mismatch",
        CoverageUnverifiedReasonV1::CountsUntrusted => "counts_untrusted",
        CoverageUnverifiedReasonV1::CrossScopeUnproven => "cross_scope_unproven",
    }
}

pub(super) fn encode_retrieval_topology(
    encoder: &mut CoverageEncoder,
    topology: RetrievalTopology,
) {
    encoder.text(topology.code());
    if let RetrievalTopology::PartialQuality { coverage_ppm } = topology {
        encoder.u32(coverage_ppm);
    }
}

pub(super) fn validate_coverage_schema(field: &str, schema_version: u16) -> SearchResult<()> {
    if schema_version == TIER_COVERAGE_SCHEMA_VERSION_V1 {
        return Ok(());
    }
    Err(coverage_error(
        field,
        &schema_version.to_string(),
        "unknown coverage schema version",
    ))
}

pub(super) fn validate_coverage_sha256(field: &str, value: &str) -> SearchResult<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    let redacted_value = format!("redacted-invalid-sha256-length-{}", value.len());
    Err(coverage_error(
        field,
        &redacted_value,
        "must be exactly 64 lowercase hexadecimal SHA-256 characters",
    ))
}

pub(super) fn coverage_error(field: &str, value: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

#[derive(Debug)]
pub(super) struct CoverageEncoder {
    bytes: Vec<u8>,
}

impl CoverageEncoder {
    pub(super) fn new(domain: &[u8]) -> Self {
        let mut encoder = Self { bytes: Vec::new() };
        encoder.bytes(domain);
        encoder
    }

    pub(super) fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    pub(super) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub(super) fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn bytes(&mut self, value: &[u8]) {
        self.u64(u64::try_from(value.len()).unwrap_or(u64::MAX));
        self.bytes.extend_from_slice(value);
    }

    pub(super) fn text(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    pub(super) fn option_text(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.text(value);
            }
            None => self.u8(0),
        }
    }

    pub(super) fn option_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.u64(value);
            }
            None => self.u8(0),
        }
    }

    pub(super) fn fingerprint(self) -> String {
        let digest = Sha256::digest(&self.bytes);
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            let _ = write!(&mut hex, "{byte:02x}");
        }
        hex
    }
}
