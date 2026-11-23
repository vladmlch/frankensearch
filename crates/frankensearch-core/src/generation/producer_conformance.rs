use serde::{Deserialize, Serialize};

use super::codec::{CanonicalEncoder, sha256_hex, validate_sha256};
use super::embedding_identity::{
    EmbeddingIdentityBundleV1, FOREIGN_PRODUCER_CONFORMANCE_CERTIFICATE_SCHEMA_V1,
    GoldenVectorCertificateV1,
};

/// Why producer compatibility could not be established.
///
/// Variants deliberately carry no caller-controlled text. Callers may log the
/// stable variant name together with bounded witness fingerprints without ever
/// exposing conformance inputs, vectors, model paths, or policy material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum ProducerCompatibilityErrorV1 {
    /// The reference bundle failed its complete identity validation.
    #[error("reference embedding identity is invalid")]
    InvalidReferenceIdentity,
    /// The candidate bundle failed its complete identity validation.
    #[error("candidate embedding identity is invalid")]
    InvalidCandidateIdentity,
    /// The bundles describe different mathematical embedding spaces.
    #[error("producer comparison crossed mathematical embedding spaces")]
    SpaceMismatch,
    /// Exact producer identity did not match, so certified evidence is required.
    #[error("foreign producer requires an explicit conformance certificate")]
    CertificateRequired,
    /// The certified path was invoked for an already-exact producer.
    #[error("foreign-producer certificate is forbidden for an exact producer")]
    CertificateForbiddenForExactProducer,
    /// The ordered fixture or one of its exact vector sets was malformed.
    #[error("golden conformance fixture is malformed")]
    GoldenFixtureInvalid,
    /// Reference and candidate outputs or their embedded certificates differ.
    #[error("golden conformance vectors do not match exactly")]
    GoldenVectorMismatch,
    /// The raw certificate has an unknown schema or malformed field.
    #[error("foreign-producer certificate is malformed")]
    CertificateMalformed,
    /// The canonical raw certificate does not match the independently pinned digest.
    #[error("foreign-producer certificate fingerprint is not trusted")]
    CertificateFingerprintMismatch,
    /// The certificate names a policy other than the independently pinned policy.
    #[error("foreign-producer certificate policy does not match trusted policy")]
    PolicyMismatch,
    /// One or both direction-sensitive producer bindings are wrong.
    #[error("foreign-producer certificate does not bind the exact producer pair")]
    ProducerBindingMismatch,
    /// The certificate does not bind the independently verified golden fixture.
    #[error("foreign-producer certificate does not bind the trusted fixture")]
    FixtureBindingMismatch,
    /// The certificate is not active yet at the trusted evaluation time.
    #[error("foreign-producer certificate is not yet valid")]
    CertificateNotYetValid,
    /// The certificate is expired at the trusted evaluation time.
    #[error("foreign-producer certificate is expired")]
    CertificateExpired,
    /// The certificate revision is outside the trusted policy window.
    #[error("foreign-producer certificate revision is outside trusted policy")]
    CertificateRevisionOutsidePolicy,
}

/// Authority-free fixture receipt derived from both producers' actual outputs.
///
/// This type is intentionally serializable but not deserializable. Untrusted
/// bytes can never regain the authority established by executing both producers
/// over the same ordered inputs and comparing their exact `f32` output bits.
///
/// ```compile_fail
/// use frankensearch_core::generation::VerifiedGoldenConformanceManifestV1;
/// let _: VerifiedGoldenConformanceManifestV1 = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct VerifiedGoldenConformanceManifestV1 {
    pub(super) certificate: GoldenVectorCertificateV1,
    pub(super) canonical_bytes: Vec<u8>,
    pub(super) fingerprint: String,
}

/// Raw, non-authoritative certificate claiming that two distinct producers are
/// bit-exact on one verified fixture.
///
/// Deserialization only recovers an untrusted claim. The sole authority-bearing
/// promotion path is
/// [`super::embedding_identity::EmbeddingIdentityBundleV1::verify_certified_foreign_producer_with`], which
/// compares every field with independently trusted identity, policy, time, and
/// fixture state.
///
/// Raw certificate bytes alone cannot become either trust context or witness:
///
/// ```compile_fail
/// use frankensearch_core::generation::{
///     ForeignProducerConformanceCertificateV1, ProducerCompatibilityWitnessV1,
///     TrustedProducerConformanceContextV1,
/// };
/// let raw: ForeignProducerConformanceCertificateV1 = serde_json::from_str("{}").unwrap();
/// let _: TrustedProducerConformanceContextV1<'_> = raw.clone().into();
/// let _: ProducerCompatibilityWitnessV1 = raw.into();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignProducerConformanceCertificateV1 {
    pub(super) schema_version: u16,
    pub(super) reference_producer_fingerprint: String,
    pub(super) candidate_producer_fingerprint: String,
    pub(super) space_fingerprint: String,
    pub(super) golden_fixture_fingerprint: String,
    pub(super) policy_fingerprint: String,
    pub(super) certificate_revision: u64,
    pub(super) not_before_unix_seconds: u64,
    pub(super) expires_at_unix_seconds: u64,
}

/// Independently sourced authority for validating one raw foreign-producer
/// certificate.
///
/// The pinned certificate and policy fingerprints, evaluation time, revision
/// window, and verified fixture must come from retained owners or sealed local
/// policy. Never populate this context from the certificate being validated.
/// This type is intentionally neither serializable nor deserializable.
///
/// ```compile_fail
/// use frankensearch_core::generation::TrustedProducerConformanceContextV1;
/// let _: TrustedProducerConformanceContextV1<'_> = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TrustedProducerConformanceContextV1<'a> {
    pub(super) policy_fingerprint: &'a str,
    pub(super) certificate_fingerprint: &'a str,
    pub(super) fixture: &'a VerifiedGoldenConformanceManifestV1,
    pub(super) evaluation_time_unix_seconds: u64,
    pub(super) minimum_certificate_revision: u64,
    pub(super) maximum_certificate_revision: u64,
}

/// How an opaque producer-compatibility witness was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducerCompatibilityKindV1 {
    /// Both validated bundles carry the exact same producer identity.
    Exact,
    /// A distinct producer was promoted through an independently trusted certificate.
    Certified,
}

/// Opaque proof that one candidate producer is admissible relative to one
/// reference producer.
///
/// Fields are private and the type is not deserializable, so a caller cannot
/// manufacture runtime authority from JSON or plausible digest strings.
///
/// ```compile_fail
/// use frankensearch_core::generation::ProducerCompatibilityWitnessV1;
/// let _: ProducerCompatibilityWitnessV1 = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct ProducerCompatibilityWitnessV1 {
    pub(super) kind: ProducerCompatibilityKindV1,
    pub(super) space_fingerprint: String,
    pub(super) reference_producer_fingerprint: String,
    pub(super) candidate_producer_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) certificate_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) policy_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) golden_fixture_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) certificate_revision: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expires_at_unix_seconds: Option<u64>,
}

impl VerifiedGoldenConformanceManifestV1 {
    /// Execute the fixture comparison boundary over already-produced exact vectors.
    ///
    /// # Errors
    ///
    /// Returns [`ProducerCompatibilityErrorV1::GoldenFixtureInvalid`] for malformed
    /// corpus/vector shapes, or [`ProducerCompatibilityErrorV1::GoldenVectorMismatch`]
    /// when the producers differ in any ordered input, shape, or exact `f32` bit.
    pub fn from_exact_pair_f32(
        texts: &[&str],
        reference_vectors: &[Vec<f32>],
        candidate_vectors: &[Vec<f32>],
    ) -> Result<Self, ProducerCompatibilityErrorV1> {
        let reference = GoldenVectorCertificateV1::from_exact_f32(texts, reference_vectors)
            .map_err(|_| ProducerCompatibilityErrorV1::GoldenFixtureInvalid)?;
        let candidate = GoldenVectorCertificateV1::from_exact_f32(texts, candidate_vectors)
            .map_err(|_| ProducerCompatibilityErrorV1::GoldenFixtureInvalid)?;
        if reference != candidate {
            return Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch);
        }
        let canonical_bytes = Self::canonical_bytes_for(&reference);
        Ok(Self {
            certificate: reference,
            fingerprint: sha256_hex(&canonical_bytes),
            canonical_bytes,
        })
    }

    /// Exact common golden-vector certificate produced by the fixture run.
    #[must_use]
    pub const fn certificate(&self) -> &GoldenVectorCertificateV1 {
        &self.certificate
    }

    /// Domain-separated SHA-256 of the frozen fixture manifest.
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Exact canonical bytes bound by [`Self::fingerprint`].
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    fn canonical_bytes_for(certificate: &GoldenVectorCertificateV1) -> Vec<u8> {
        let mut encoder =
            CanonicalEncoder::new(b"frankensearch.verified-golden-conformance-manifest.v1");
        certificate.encode(&mut encoder);
        encoder.finish()
    }

    pub(super) fn validate(&self) -> Result<(), ProducerCompatibilityErrorV1> {
        self.certificate
            .validate()
            .map_err(|_| ProducerCompatibilityErrorV1::GoldenFixtureInvalid)?;
        let canonical_bytes = Self::canonical_bytes_for(&self.certificate);
        // ubs:ignore — these canonical certificate bytes are public integrity evidence, not secrets.
        if self.canonical_bytes != canonical_bytes
            // ubs:ignore — this fingerprint is public certificate-integrity evidence, not a secret.
            || self.fingerprint != sha256_hex(&canonical_bytes)
        {
            return Err(ProducerCompatibilityErrorV1::GoldenFixtureInvalid);
        }
        Ok(())
    }
}

impl ForeignProducerConformanceCertificateV1 {
    /// Create an authority-free raw receipt for a verified producer pair.
    ///
    /// The returned value remains untrusted until promoted against a
    /// [`TrustedProducerConformanceContextV1`] that independently pins its
    /// canonical fingerprint. Computing that fingerprint from this returned
    /// value and immediately feeding it back as the supposed independent pin is
    /// self-assertion, not certification.
    ///
    /// # Errors
    ///
    /// Returns a typed producer-compatibility error for invalid identities,
    /// cross-space or exact-producer use, fixture disagreement, malformed policy,
    /// zero revision, or an empty validity interval.
    pub fn new_untrusted_receipt_from_verified_pair(
        reference: &EmbeddingIdentityBundleV1,
        candidate: &EmbeddingIdentityBundleV1,
        fixture: &VerifiedGoldenConformanceManifestV1,
        policy_fingerprint: &str,
        certificate_revision: u64,
        not_before_unix_seconds: u64,
        expires_at_unix_seconds: u64,
    ) -> Result<Self, ProducerCompatibilityErrorV1> {
        validate_reference_and_candidate(reference, candidate)?;
        // ubs:ignore — embedding-space fingerprints are public compatibility identities.
        if reference.space.fingerprint() != candidate.space.fingerprint() {
            return Err(ProducerCompatibilityErrorV1::SpaceMismatch);
        }
        let reference_producer_fingerprint = reference.producer.fingerprint();
        let candidate_producer_fingerprint = candidate.producer.fingerprint();
        if reference_producer_fingerprint == candidate_producer_fingerprint {
            return Err(ProducerCompatibilityErrorV1::CertificateForbiddenForExactProducer);
        }
        fixture.validate()?;
        if fixture.certificate != reference.producer.golden_vectors
            || fixture.certificate != candidate.producer.golden_vectors
        {
            return Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch);
        }
        if validate_sha256(
            "producer_conformance.policy_fingerprint",
            policy_fingerprint,
        )
        .is_err()
            || certificate_revision == 0
            || not_before_unix_seconds >= expires_at_unix_seconds
        {
            return Err(ProducerCompatibilityErrorV1::CertificateMalformed);
        }
        Ok(Self {
            schema_version: FOREIGN_PRODUCER_CONFORMANCE_CERTIFICATE_SCHEMA_V1,
            reference_producer_fingerprint,
            candidate_producer_fingerprint,
            space_fingerprint: reference.space.fingerprint(),
            golden_fixture_fingerprint: fixture.fingerprint.clone(),
            policy_fingerprint: policy_fingerprint.to_owned(),
            certificate_revision,
            not_before_unix_seconds,
            expires_at_unix_seconds,
        })
    }

    /// Domain-separated canonical bytes for policy pinning.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder =
            CanonicalEncoder::new(b"frankensearch.foreign-producer-conformance-certificate.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.reference_producer_fingerprint);
        encoder.text(&self.candidate_producer_fingerprint);
        encoder.text(&self.space_fingerprint);
        encoder.text(&self.golden_fixture_fingerprint);
        encoder.text(&self.policy_fingerprint);
        encoder.u64(self.certificate_revision);
        encoder.u64(self.not_before_unix_seconds);
        encoder.u64(self.expires_at_unix_seconds);
        encoder.finish()
    }

    /// Lowercase SHA-256 of [`Self::canonical_bytes`].
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    /// Certificate revision bound by the canonical fingerprint.
    #[must_use]
    pub const fn certificate_revision(&self) -> u64 {
        self.certificate_revision
    }

    /// Exclusive expiry bound in Unix seconds.
    #[must_use]
    pub const fn expires_at_unix_seconds(&self) -> u64 {
        self.expires_at_unix_seconds
    }

    pub(super) fn validate(&self) -> Result<(), ProducerCompatibilityErrorV1> {
        if self.schema_version != FOREIGN_PRODUCER_CONFORMANCE_CERTIFICATE_SCHEMA_V1
            || self.certificate_revision == 0
            || self.not_before_unix_seconds >= self.expires_at_unix_seconds
        {
            return Err(ProducerCompatibilityErrorV1::CertificateMalformed);
        }
        for (field, fingerprint) in [
            (
                "producer_conformance.reference_producer_fingerprint",
                &self.reference_producer_fingerprint,
            ),
            (
                "producer_conformance.candidate_producer_fingerprint",
                &self.candidate_producer_fingerprint,
            ),
            (
                "producer_conformance.space_fingerprint",
                &self.space_fingerprint,
            ),
            (
                "producer_conformance.golden_fixture_fingerprint",
                &self.golden_fixture_fingerprint,
            ),
            (
                "producer_conformance.policy_fingerprint",
                &self.policy_fingerprint,
            ),
        ] {
            if validate_sha256(field, fingerprint).is_err() {
                return Err(ProducerCompatibilityErrorV1::CertificateMalformed);
            }
        }
        Ok(())
    }
}

impl<'a> TrustedProducerConformanceContextV1<'a> {
    /// Bind independently trusted policy, certificate, fixture, time, and revision state.
    ///
    /// The C1 foundation deliberately provides no production policy registry or
    /// consumer. Applications must source these pins from an owner-controlled
    /// channel independent of the raw certificate; this constructor does not
    /// authenticate caller authority and must never be populated from fields or
    /// digests learned only from the certificate under review.
    ///
    /// # Errors
    ///
    /// Returns [`ProducerCompatibilityErrorV1::CertificateMalformed`] for malformed
    /// pinned digests or an invalid revision window, and propagates fixture defects.
    pub fn from_independent_policy(
        policy_fingerprint: &'a str,
        certificate_fingerprint: &'a str,
        fixture: &'a VerifiedGoldenConformanceManifestV1,
        evaluation_time_unix_seconds: u64,
        minimum_certificate_revision: u64,
        maximum_certificate_revision: u64,
    ) -> Result<Self, ProducerCompatibilityErrorV1> {
        if validate_sha256(
            "trusted_producer_conformance.policy_fingerprint",
            policy_fingerprint,
        )
        .is_err()
            || validate_sha256(
                "trusted_producer_conformance.certificate_fingerprint",
                certificate_fingerprint,
            )
            .is_err()
            || minimum_certificate_revision == 0
            || minimum_certificate_revision > maximum_certificate_revision
        {
            return Err(ProducerCompatibilityErrorV1::CertificateMalformed);
        }
        fixture.validate()?;
        Ok(Self {
            policy_fingerprint,
            certificate_fingerprint,
            fixture,
            evaluation_time_unix_seconds,
            minimum_certificate_revision,
            maximum_certificate_revision,
        })
    }
}

impl ProducerCompatibilityWitnessV1 {
    /// Exact versus explicitly certified producer compatibility.
    #[must_use]
    pub const fn kind(&self) -> ProducerCompatibilityKindV1 {
        self.kind
    }

    /// Mathematical space shared by the producer pair.
    #[must_use]
    pub fn space_fingerprint(&self) -> &str {
        &self.space_fingerprint
    }

    /// Reference producer bound into the witness.
    #[must_use]
    pub fn reference_producer_fingerprint(&self) -> &str {
        &self.reference_producer_fingerprint
    }

    /// Candidate producer bound into the witness.
    #[must_use]
    pub fn candidate_producer_fingerprint(&self) -> &str {
        &self.candidate_producer_fingerprint
    }

    /// Canonical certificate fingerprint for certified witnesses.
    #[must_use]
    pub fn certificate_fingerprint(&self) -> Option<&str> {
        self.certificate_fingerprint.as_deref()
    }
}

pub(super) fn validate_reference_and_candidate(
    reference: &EmbeddingIdentityBundleV1,
    candidate: &EmbeddingIdentityBundleV1,
) -> Result<(), ProducerCompatibilityErrorV1> {
    reference
        .validate()
        .map_err(|_| ProducerCompatibilityErrorV1::InvalidReferenceIdentity)?;
    candidate
        .validate()
        .map_err(|_| ProducerCompatibilityErrorV1::InvalidCandidateIdentity)
}
