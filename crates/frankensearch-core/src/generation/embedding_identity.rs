use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::SearchError;

use super::QuantizationFormat;
use super::codec::{
    CanonicalEncoder, identity_error, sha256_hex, validate_identity_text,
    validate_optional_identity_text, validate_schema, validate_sha256,
};
use super::producer_conformance::{
    ForeignProducerConformanceCertificateV1, ProducerCompatibilityErrorV1,
    ProducerCompatibilityKindV1, ProducerCompatibilityWitnessV1,
    TrustedProducerConformanceContextV1, validate_reference_and_candidate,
};

// ---------------------------------------------------------------------------
// Embedding identity contracts
// ---------------------------------------------------------------------------

/// Current schema for the mathematical embedding-space identity.
pub const EMBEDDING_SPACE_IDENTITY_SCHEMA_V1: u16 = 1;
/// Current schema for producer attestations.
pub const EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1: u16 = 1;
/// Current schema for outer embedding-input contracts.
pub const EMBEDDING_INPUT_CONTRACT_SCHEMA_V1: u16 = 1;
/// Current schema for vector-storage identities.
pub const VECTOR_STORAGE_IDENTITY_SCHEMA_V1: u16 = 1;
/// Current schema for explicit foreign-producer conformance certificates.
pub const FOREIGN_PRODUCER_CONFORMANCE_CERTIFICATE_SCHEMA_V1: u16 = 1;
/// Current schema for immutable artifact-generation identities.
pub const ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1: u16 = 1;

pub(super) const MAX_IDENTITY_FIELD_BYTES: usize = 4_096;

/// Immutable identity of one published artifact generation.
///
/// `sequence` is a full-width monotone counter within one publication lineage;
/// it replaces the wrap-prone `u8` compaction generation in FSVI v1. The
/// caller-supplied 128-bit nonce distinguishes independently built generations
/// that happen to use the same sequence number. The nonce is identity material,
/// not a credential.
///
/// Equality is exact. Neither the sequence nor the nonce alone is a sufficient
/// generation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactGenerationIdentityV1 {
    /// [`ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1`].
    pub schema_version: u16,
    /// Monotone generation sequence within the publisher's lineage.
    pub sequence: u64,
    /// Unique identity material for this build/publication attempt.
    pub nonce: [u8; 16],
}

impl ArtifactGenerationIdentityV1 {
    /// Construct and validate one artifact-generation identity.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the nonce is all zeroes.
    pub fn new(sequence: u64, nonce: [u8; 16]) -> Result<Self, SearchError> {
        let identity = Self {
            schema_version: ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1,
            sequence,
            nonce,
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Validate the schema and reject the reserved all-zero nonce.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an unknown schema or an
    /// all-zero nonce.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "artifact_generation.schema_version",
            self.schema_version,
            ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1,
        )?;
        // ubs:ignore — this serialized generation-uniqueness nonce is public, not an authenticator.
        if self.nonce == [0; 16] {
            return Err(identity_error(
                "artifact_generation.nonce",
                "redacted-zero-nonce",
                "must contain unique non-zero generation identity material",
            ));
        }
        Ok(())
    }

    /// Canonical domain-separated bytes used by persisted FSVI and WAL
    /// bindings.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.artifact-generation-identity.v1");
        encoder.u16(self.schema_version);
        encoder.u64(self.sequence);
        encoder.bytes(&self.nonce);
        encoder.finish()
    }

    /// Lowercase SHA-256 of [`Self::canonical_bytes`].
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

/// One immutable, role-tagged artifact participating in a vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingArtifactIdentityV1 {
    /// Stable semantic role such as `weights`, `tokenizer`, `vocabulary`, or `config`.
    pub role: String,
    /// Lowercase SHA-256 of the exact artifact bytes.
    pub sha256: String,
    /// Exact artifact size in bytes.
    pub size: u64,
}

/// Whether an embedding space is semantic or an explicit non-semantic control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingSpaceKindV1 {
    /// A learned vector space pinned by immutable model artifacts.
    Semantic,
    /// A deterministic test/control space that must never imply semantic availability.
    HashControl,
}

/// Complete algorithm contract for a deterministic hash control.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HashControlProfileV1 {
    /// Algorithm family, for example `fnv1a-feature-hash`.
    pub algorithm: String,
    /// Immutable algorithm/protocol revision.
    pub algorithm_revision: String,
    /// Seed or key identifier. This is configuration, not secret key material.
    pub seed: u64,
    /// Exact feature extraction rules.
    pub feature_rules: String,
    /// Exact tokenization rules.
    pub tokenization_rules: String,
    /// Exact signed-bucket rules.
    pub signing_rules: String,
    /// Exact output normalization rules.
    pub normalization_rules: String,
}

/// Structured Matryoshka/projection ancestry for a derived vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProjectionV1 {
    /// Fingerprint of the complete parent vector space.
    pub parent_space_fingerprint: String,
    /// Dimension before projection.
    pub source_dimension: u32,
    /// Dimension after projection.
    pub output_dimension: u32,
    /// Exact deterministic projection/truncation rule.
    pub projection_rule: String,
    /// Exact post-projection normalization rule.
    pub renormalization_rule: String,
}

/// Mathematical identity of the complete input-to-vector map.
///
/// Equality is exact. Human names, directory names, filenames, and dimension-only
/// matches never establish compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingSpaceIdentityV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Semantic model/control identifier, never a display label.
    pub logical_model_id: String,
    /// Immutable upstream model or algorithm revision.
    pub immutable_revision: String,
    /// Semantic versus explicit hash-control classification.
    pub kind: EmbeddingSpaceKindV1,
    /// Fingerprint of the verified artifact-space contract: immutable
    /// role-tagged artifacts plus all model semantics, excluding producer and
    /// distribution metadata.
    ///
    /// Hash controls use the canonical hash-profile fingerprint here.
    pub artifact_manifest_fingerprint: String,
    /// Artifact identities. Canonical encoding sorts them by role.
    pub artifacts: Vec<EmbeddingArtifactIdentityV1>,
    /// Exact tokenizer identity.
    pub tokenizer_fingerprint: String,
    /// Exact vocabulary identity.
    pub vocabulary_fingerprint: String,
    /// Exact model configuration identity.
    pub model_config_fingerprint: String,
    /// Model-internal preprocessing contract.
    pub model_preprocessing: String,
    /// Sequence length, truncation, and padding contract.
    pub sequence_policy: String,
    /// Query instruction applied inside the model contract.
    pub query_instruction: String,
    /// Document instruction applied inside the model contract.
    pub document_instruction: String,
    /// Pooling rule.
    pub pooling: String,
    /// Output normalization rule.
    pub output_normalization: String,
    /// Output dimension.
    pub dimension: u32,
    /// Fingerprint of the separately versioned outer input contract.
    pub input_contract_fingerprint: String,
    /// Required for hash controls and forbidden for semantic spaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_control: Option<HashControlProfileV1>,
    /// Structured ancestry for MRL or another deterministic projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<EmbeddingProjectionV1>,
}

impl EmbeddingSpaceIdentityV1 {
    /// Validate all required fields and cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for an unknown schema, malformed digest,
    /// duplicate role, incomplete semantic space, or inconsistent projection.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "embedding_space_identity.schema_version",
            self.schema_version,
            EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
        )?;
        validate_identity_text("logical_model_id", &self.logical_model_id)?;
        validate_identity_text("immutable_revision", &self.immutable_revision)?;
        validate_sha256(
            "artifact_manifest_fingerprint",
            &self.artifact_manifest_fingerprint,
        )?;
        validate_sha256("tokenizer_fingerprint", &self.tokenizer_fingerprint)?;
        validate_sha256("vocabulary_fingerprint", &self.vocabulary_fingerprint)?;
        validate_sha256("model_config_fingerprint", &self.model_config_fingerprint)?;
        validate_identity_text("model_preprocessing", &self.model_preprocessing)?;
        validate_identity_text("sequence_policy", &self.sequence_policy)?;
        validate_optional_identity_text("query_instruction", &self.query_instruction)?;
        validate_optional_identity_text("document_instruction", &self.document_instruction)?;
        validate_identity_text("pooling", &self.pooling)?;
        validate_identity_text("output_normalization", &self.output_normalization)?;
        validate_sha256(
            "input_contract_fingerprint",
            &self.input_contract_fingerprint,
        )?;
        if self.dimension == 0 {
            return Err(identity_error(
                "dimension",
                "0",
                "must be greater than zero",
            ));
        }

        let mut roles = BTreeSet::new();
        for artifact in &self.artifacts {
            validate_identity_text("artifacts[].role", &artifact.role)?;
            validate_sha256("artifacts[].sha256", &artifact.sha256)?;
            if artifact.size == 0 {
                return Err(identity_error(
                    "artifacts[].size",
                    "0",
                    "must be greater than zero",
                ));
            }
            if !roles.insert(artifact.role.as_str()) {
                return Err(identity_error(
                    "artifacts[].role",
                    &artifact.role,
                    "duplicate artifact role",
                ));
            }
        }

        match (self.kind, &self.hash_control) {
            (EmbeddingSpaceKindV1::Semantic, None) => {
                if self.artifacts.is_empty() {
                    return Err(identity_error(
                        "artifacts",
                        "[]",
                        "semantic spaces require immutable artifacts",
                    ));
                }
            }
            (EmbeddingSpaceKindV1::Semantic, Some(_)) => {
                return Err(identity_error(
                    "hash_control",
                    "present",
                    "semantic spaces cannot carry a hash-control profile",
                ));
            }
            (EmbeddingSpaceKindV1::HashControl, Some(profile)) => {
                profile.validate()?;
                // ubs:ignore — manifest fingerprints are public compatibility identities, not secrets.
                if self.artifact_manifest_fingerprint != profile.fingerprint() {
                    return Err(identity_error(
                        "artifact_manifest_fingerprint",
                        &self.artifact_manifest_fingerprint,
                        "must equal the canonical hash-control profile fingerprint",
                    ));
                }
                if !self.artifacts.is_empty() {
                    return Err(identity_error(
                        "artifacts",
                        "present",
                        "hash controls bind rules, not learned model artifacts",
                    ));
                }
            }
            (EmbeddingSpaceKindV1::HashControl, None) => {
                return Err(identity_error(
                    "hash_control",
                    "absent",
                    "hash-control spaces require the complete algorithm profile",
                ));
            }
        }

        if let Some(projection) = &self.projection {
            projection.validate()?;
            if projection.output_dimension != self.dimension {
                return Err(identity_error(
                    "projection.output_dimension",
                    &projection.output_dimension.to_string(),
                    "must equal the space output dimension",
                ));
            }
        }
        Ok(())
    }

    /// Canonical domain-separated, length-prefixed bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.embedding-space.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.logical_model_id);
        encoder.text(&self.immutable_revision);
        encoder.u8(match self.kind {
            EmbeddingSpaceKindV1::Semantic => 1,
            EmbeddingSpaceKindV1::HashControl => 2,
        });
        encoder.text(&self.artifact_manifest_fingerprint);

        let mut artifacts = self.artifacts.iter().collect::<Vec<_>>();
        artifacts.sort_by(|left, right| left.role.cmp(&right.role));
        encoder.usize(artifacts.len());
        for artifact in artifacts {
            encoder.text(&artifact.role);
            encoder.text(&artifact.sha256);
            encoder.u64(artifact.size);
        }

        encoder.text(&self.tokenizer_fingerprint);
        encoder.text(&self.vocabulary_fingerprint);
        encoder.text(&self.model_config_fingerprint);
        encoder.text(&self.model_preprocessing);
        encoder.text(&self.sequence_policy);
        encoder.text(&self.query_instruction);
        encoder.text(&self.document_instruction);
        encoder.text(&self.pooling);
        encoder.text(&self.output_normalization);
        encoder.u32(self.dimension);
        encoder.text(&self.input_contract_fingerprint);
        encoder.option(self.hash_control.as_ref(), HashControlProfileV1::encode);
        encoder.option(self.projection.as_ref(), EmbeddingProjectionV1::encode);
        encoder.finish()
    }

    /// Lowercase SHA-256 fingerprint of [`canonical_bytes`](Self::canonical_bytes).
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    /// Produce a structurally derived MRL/projection identity.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the target is not a strict non-zero reduction.
    pub fn derive_projection(
        &self,
        target_dimension: u32,
        projection_rule: &str,
        renormalization_rule: &str,
    ) -> Result<Self, SearchError> {
        self.validate()?;
        if target_dimension == 0 || target_dimension >= self.dimension {
            return Err(identity_error(
                "projection.output_dimension",
                &target_dimension.to_string(),
                "must be between 1 and parent dimension - 1",
            ));
        }
        validate_identity_text("projection_rule", projection_rule)?;
        validate_identity_text("renormalization_rule", renormalization_rule)?;

        let mut derived = self.clone();
        derived.dimension = target_dimension;
        renormalization_rule.clone_into(&mut derived.output_normalization);
        derived.projection = Some(EmbeddingProjectionV1 {
            parent_space_fingerprint: self.fingerprint(),
            source_dimension: self.dimension,
            output_dimension: target_dimension,
            projection_rule: projection_rule.to_owned(),
            renormalization_rule: renormalization_rule.to_owned(),
        });
        derived.validate()?;
        Ok(derived)
    }
}

impl HashControlProfileV1 {
    fn validate(&self) -> Result<(), SearchError> {
        validate_identity_text("hash_control.algorithm", &self.algorithm)?;
        validate_identity_text("hash_control.algorithm_revision", &self.algorithm_revision)?;
        validate_identity_text("hash_control.feature_rules", &self.feature_rules)?;
        validate_identity_text("hash_control.tokenization_rules", &self.tokenization_rules)?;
        validate_identity_text("hash_control.signing_rules", &self.signing_rules)?;
        validate_identity_text(
            "hash_control.normalization_rules",
            &self.normalization_rules,
        )
    }

    /// Canonical domain-separated, length-prefixed profile bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.hash-control-profile.v1");
        self.encode(&mut encoder);
        encoder.finish()
    }

    /// Lowercase SHA-256 of the complete deterministic hash-control profile.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    fn encode(&self, encoder: &mut CanonicalEncoder) {
        encoder.text(&self.algorithm);
        encoder.text(&self.algorithm_revision);
        encoder.u64(self.seed);
        encoder.text(&self.feature_rules);
        encoder.text(&self.tokenization_rules);
        encoder.text(&self.signing_rules);
        encoder.text(&self.normalization_rules);
    }
}

impl EmbeddingProjectionV1 {
    fn validate(&self) -> Result<(), SearchError> {
        validate_sha256(
            "projection.parent_space_fingerprint",
            &self.parent_space_fingerprint,
        )?;
        if self.source_dimension == 0
            || self.output_dimension == 0
            || self.output_dimension >= self.source_dimension
        {
            return Err(identity_error(
                "projection.dimension",
                &format!("{}->{}", self.source_dimension, self.output_dimension),
                "must be a strict non-zero reduction",
            ));
        }
        validate_identity_text("projection.projection_rule", &self.projection_rule)?;
        validate_identity_text(
            "projection.renormalization_rule",
            &self.renormalization_rule,
        )
    }

    fn encode(&self, encoder: &mut CanonicalEncoder) {
        encoder.text(&self.parent_space_fingerprint);
        encoder.u32(self.source_dimension);
        encoder.u32(self.output_dimension);
        encoder.text(&self.projection_rule);
        encoder.text(&self.renormalization_rule);
    }
}

/// Outer content-selection and canonicalization contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingInputContractV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Outer canonicalization before model-specific preprocessing.
    pub canonicalization: String,
    /// Which document/query content is selected.
    pub content_selection: String,
    /// Chunking and overlap semantics.
    pub chunking: String,
    /// Query instruction semantics.
    pub query_instruction: String,
    /// Document instruction semantics.
    pub document_instruction: String,
    /// Document-id and aggregation semantics.
    pub doc_id_semantics: String,
}

impl EmbeddingInputContractV1 {
    /// Validate the complete input contract.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for an unknown schema or an empty required field.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "embedding_input_contract.schema_version",
            self.schema_version,
            EMBEDDING_INPUT_CONTRACT_SCHEMA_V1,
        )?;
        validate_identity_text("canonicalization", &self.canonicalization)?;
        validate_identity_text("content_selection", &self.content_selection)?;
        validate_identity_text("chunking", &self.chunking)?;
        validate_optional_identity_text("query_instruction", &self.query_instruction)?;
        validate_optional_identity_text("document_instruction", &self.document_instruction)?;
        validate_identity_text("doc_id_semantics", &self.doc_id_semantics)
    }

    /// Canonical domain-separated, length-prefixed bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.embedding-input.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.canonicalization);
        encoder.text(&self.content_selection);
        encoder.text(&self.chunking);
        encoder.text(&self.query_instruction);
        encoder.text(&self.document_instruction);
        encoder.text(&self.doc_id_semantics);
        encoder.finish()
    }

    /// Lowercase SHA-256 of the canonical input-contract bytes.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

/// Golden-vector certificate pinning one implementation to a vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenVectorCertificateV1 {
    /// Digest of the ordered, redacted golden input corpus.
    pub corpus_sha256: String,
    /// Digest of exact output f32 bit patterns.
    pub vectors_sha256: String,
    /// Number of golden vectors.
    pub vector_count: u32,
    /// Dimension of every golden vector.
    pub dimension: u32,
}

impl GoldenVectorCertificateV1 {
    /// Fingerprint an ordered, non-empty conformance text corpus.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the corpus is empty.
    pub fn corpus_fingerprint(texts: &[&str]) -> Result<String, SearchError> {
        if texts.is_empty() {
            return Err(identity_error(
                "golden.corpus_shape",
                "0 texts",
                "the golden corpus must be non-empty",
            ));
        }
        let mut corpus = CanonicalEncoder::new(b"frankensearch.golden-corpus.v1");
        corpus.usize(texts.len());
        for text in texts {
            corpus.text(text);
        }
        Ok(sha256_hex(&corpus.finish()))
    }

    /// Build a certificate from an ordered text corpus and exact `f32` output bits.
    ///
    /// Text and vector bytes are encoded in separate domain-separated,
    /// length-prefixed transcripts. Floating-point values are recorded with
    /// [`f32::to_bits`], preserving signed zero and NaN payloads.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the corpus is empty, the text/vector counts
    /// differ, vector dimensions are inconsistent, or the shape exceeds `u32`.
    pub fn from_exact_f32(texts: &[&str], vectors: &[Vec<f32>]) -> Result<Self, SearchError> {
        if texts.is_empty() || texts.len() != vectors.len() {
            return Err(identity_error(
                "golden.corpus_shape",
                &format!("{} texts/{} vectors", texts.len(), vectors.len()),
                "the non-empty text and vector counts must match",
            ));
        }
        let dimension = vectors.first().map_or(0, Vec::len);
        if dimension == 0 || vectors.iter().any(|vector| vector.len() != dimension) {
            return Err(identity_error(
                "golden.vector_shape",
                &format!("{} vectors", vectors.len()),
                "all golden vectors must have one identical non-zero dimension",
            ));
        }
        let vector_count = u32::try_from(vectors.len()).map_err(|_| {
            identity_error(
                "golden.vector_count",
                "out-of-range",
                "golden vector count must fit in u32",
            )
        })?;
        let dimension = u32::try_from(dimension).map_err(|_| {
            identity_error(
                "golden.dimension",
                "out-of-range",
                "golden vector dimension must fit in u32",
            )
        })?;

        let mut outputs = CanonicalEncoder::new(b"frankensearch.golden-f32-vectors.v1");
        outputs.u32(vector_count);
        outputs.u32(dimension);
        for vector in vectors {
            for value in vector {
                outputs.u32(value.to_bits());
            }
        }

        Ok(Self {
            corpus_sha256: Self::corpus_fingerprint(texts)?,
            vectors_sha256: sha256_hex(&outputs.finish()),
            vector_count,
            dimension,
        })
    }

    /// Verify an ordered corpus and exact output bits against this certificate.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the certificate is malformed or any corpus,
    /// shape, or output bit differs. The error never includes input or vector data.
    pub fn verify_exact_f32(
        &self,
        texts: &[&str],
        vectors: &[Vec<f32>],
    ) -> Result<(), SearchError> {
        self.validate()?;
        let observed = Self::from_exact_f32(texts, vectors)?;
        if observed != *self {
            return Err(identity_error(
                "golden.conformance",
                "mismatch",
                "ordered corpus, vector shape, or exact f32 output bits drifted",
            ));
        }
        Ok(())
    }

    /// Validate digest shape and non-zero certificate dimensions.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for malformed hashes or an empty certificate.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_sha256("golden.corpus_sha256", &self.corpus_sha256)?;
        validate_sha256("golden.vectors_sha256", &self.vectors_sha256)?;
        if self.vector_count == 0 || self.dimension == 0 {
            return Err(identity_error(
                "golden.shape",
                &format!("{}x{}", self.vector_count, self.dimension),
                "vector count and dimension must be non-zero",
            ));
        }
        Ok(())
    }

    pub(super) fn encode(&self, encoder: &mut CanonicalEncoder) {
        encoder.text(&self.corpus_sha256);
        encoder.text(&self.vectors_sha256);
        encoder.u32(self.vector_count);
        encoder.u32(self.dimension);
    }
}

/// Attestation for the implementation that produced vectors in a space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProducerAttestationV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Backend family such as `model2vec-native`, `fastembed-onnx`, or `remote-api`.
    pub backend: String,
    /// Immutable backend implementation revision.
    pub implementation_revision: String,
    /// Wire/inference protocol revision.
    pub protocol_revision: String,
    /// Numeric execution profile, including dtype and deterministic tolerances.
    pub numeric_profile: String,
    /// Fingerprint of the complete provenance manifest. Local/native learned
    /// models bind the full frozen `ModelArtifactManifestV1` here, including
    /// provider, repository, license, distribution metadata, execution
    /// contract, and golden certificate. Hash controls bind their canonical
    /// algorithm profile.
    pub provenance_manifest_fingerprint: String,
    /// Fingerprint of the exact mathematical vector space.
    pub space_fingerprint: String,
    /// Pinned conformance-vector certificate.
    pub golden_vectors: GoldenVectorCertificateV1,
}

impl EmbeddingProducerAttestationV1 {
    /// Validate the complete producer attestation.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for an unknown schema, empty field, malformed
    /// fingerprint, or incomplete golden-vector certificate.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "embedding_producer_attestation.schema_version",
            self.schema_version,
            EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
        )?;
        validate_identity_text("backend", &self.backend)?;
        validate_identity_text("implementation_revision", &self.implementation_revision)?;
        validate_identity_text("protocol_revision", &self.protocol_revision)?;
        validate_identity_text("numeric_profile", &self.numeric_profile)?;
        validate_sha256(
            "provenance_manifest_fingerprint",
            &self.provenance_manifest_fingerprint,
        )?;
        validate_sha256("space_fingerprint", &self.space_fingerprint)?;
        self.golden_vectors.validate()
    }

    /// Canonical domain-separated, length-prefixed bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.embedding-producer.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.backend);
        encoder.text(&self.implementation_revision);
        encoder.text(&self.protocol_revision);
        encoder.text(&self.numeric_profile);
        encoder.text(&self.provenance_manifest_fingerprint);
        encoder.text(&self.space_fingerprint);
        self.golden_vectors.encode(&mut encoder);
        encoder.finish()
    }

    /// Lowercase SHA-256 of the canonical producer-attestation bytes.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

/// Physical vector encoding identity, deliberately separate from vector-space identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorStorageIdentityV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Storage format and format revision, for example `fsvi-v2`.
    pub format: String,
    /// Quantization/storage scalar format.
    pub quantization: QuantizationFormat,
    /// Byte order.
    pub endianness: String,
    /// Stored-vector normalization assumption.
    pub vector_normalization: String,
    /// Stored vector dimension.
    pub dimension: u32,
}

impl VectorStorageIdentityV1 {
    /// Validate the storage contract.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for an unknown schema, empty field, or zero dimension.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "vector_storage_identity.schema_version",
            self.schema_version,
            VECTOR_STORAGE_IDENTITY_SCHEMA_V1,
        )?;
        validate_identity_text("format", &self.format)?;
        validate_identity_text("endianness", &self.endianness)?;
        validate_identity_text("vector_normalization", &self.vector_normalization)?;
        if self.dimension == 0 {
            return Err(identity_error(
                "storage.dimension",
                "0",
                "must be greater than zero",
            ));
        }
        if self.format.starts_with("in-memory-") && self.quantization != QuantizationFormat::F32 {
            return Err(identity_error(
                "storage.quantization",
                &format!("{:?}", self.quantization),
                "in-memory Vec<f32> formats require F32 values",
            ));
        }
        if self.format.starts_with("in-memory-")
            && !matches!(
                self.endianness.as_str(),
                "native-f32-values" | "native-test-only"
            )
        {
            return Err(identity_error(
                "storage.endianness",
                &self.endianness,
                "in-memory formats require an explicit native-value contract",
            ));
        }
        if self.format.starts_with("fsvi-") && self.endianness != "little-endian" {
            return Err(identity_error(
                "storage.endianness",
                &self.endianness,
                "FSVI storage is canonically little-endian",
            ));
        }
        Ok(())
    }

    /// Canonical domain-separated, length-prefixed bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.vector-storage.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.format);
        encoder.u8(match self.quantization {
            QuantizationFormat::F32 => 1,
            QuantizationFormat::F16 => 2,
            QuantizationFormat::Int8 => 3,
            QuantizationFormat::Int4 => 4,
        });
        encoder.text(&self.endianness);
        encoder.text(&self.vector_normalization);
        encoder.u32(self.dimension);
        encoder.finish()
    }

    /// Lowercase SHA-256 of the canonical storage bytes.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

/// Complete, independently versioned embedding identity bundle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingIdentityBundleV1 {
    /// Mathematical input-to-vector map.
    pub space: EmbeddingSpaceIdentityV1,
    /// Implementation/protocol attestation.
    pub producer: EmbeddingProducerAttestationV1,
    /// Outer content-selection/canonicalization contract.
    pub input: EmbeddingInputContractV1,
    /// Physical vector encoding identity.
    pub storage: VectorStorageIdentityV1,
}

/// Persistable exact bytes and fingerprint for one complete identity bundle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenEmbeddingIdentityBundleV1 {
    /// Structured identity contracts.
    pub identity: EmbeddingIdentityBundleV1,
    /// Exact domain-separated canonical bytes.
    pub canonical_bytes: Vec<u8>,
    /// Lowercase SHA-256 of `canonical_bytes`.
    pub fingerprint: String,
}

impl EmbeddingIdentityBundleV1 {
    /// Validate every component and all cross-component bindings.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when any component is malformed or a fingerprint
    /// or dimension does not bind the other bundled contracts exactly.
    pub fn validate(&self) -> Result<(), SearchError> {
        self.space.validate()?;
        self.producer.validate()?;
        self.input.validate()?;
        self.storage.validate()?;
        let space_fingerprint = self.space.fingerprint();
        if self.producer.space_fingerprint != space_fingerprint {
            return Err(identity_error(
                "producer.space_fingerprint",
                &self.producer.space_fingerprint,
                "does not bind the bundled space identity",
            ));
        }
        if self.space.kind == EmbeddingSpaceKindV1::HashControl
            && self.producer.provenance_manifest_fingerprint
                != self.space.artifact_manifest_fingerprint
        {
            return Err(identity_error(
                "producer.provenance_manifest_fingerprint",
                &self.producer.provenance_manifest_fingerprint,
                "hash controls must bind the canonical hash-control profile",
            ));
        }
        let input_fingerprint = self.input.fingerprint();
        if self.space.input_contract_fingerprint != input_fingerprint {
            return Err(identity_error(
                "space.input_contract_fingerprint",
                &self.space.input_contract_fingerprint,
                "does not bind the bundled input contract",
            ));
        }
        if self.storage.dimension != self.space.dimension {
            return Err(identity_error(
                "storage.dimension",
                &self.storage.dimension.to_string(),
                "does not match the bundled space dimension",
            ));
        }
        if self.storage.vector_normalization != self.space.output_normalization {
            return Err(identity_error(
                "storage.vector_normalization",
                &self.storage.vector_normalization,
                "does not match the bundled space output normalization",
            ));
        }
        if self.producer.golden_vectors.dimension != self.space.dimension {
            return Err(identity_error(
                "producer.golden_vectors.dimension",
                &self.producer.golden_vectors.dimension.to_string(),
                "does not match the bundled space dimension",
            ));
        }
        Ok(())
    }

    /// Canonical domain-separated bytes containing the four component fingerprints.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.embedding-bundle.v1");
        encoder.text(&self.space.fingerprint());
        encoder.text(&self.producer.fingerprint());
        encoder.text(&self.input.fingerprint());
        encoder.text(&self.storage.fingerprint());
        encoder.finish()
    }

    /// Lowercase SHA-256 of the complete identity bundle.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    /// Prove that a candidate carries the exact same producer identity.
    ///
    /// Storage identity is deliberately excluded: the same producer may persist
    /// an identical mathematical space in a different physical encoding. A
    /// same-space producer with any implementation, protocol, numeric, provenance,
    /// or golden-certificate drift must use the explicitly certified path.
    ///
    /// # Errors
    ///
    /// Returns a closed [`ProducerCompatibilityErrorV1`] when either bundle is
    /// invalid, the mathematical spaces differ, or the producer fingerprints are
    /// not identical.
    pub fn verify_exact_producer_with(
        &self,
        candidate: &Self,
    ) -> Result<ProducerCompatibilityWitnessV1, ProducerCompatibilityErrorV1> {
        validate_reference_and_candidate(self, candidate)?;
        let space_fingerprint = self.space.fingerprint();
        // ubs:ignore — embedding-space fingerprints are public compatibility identities.
        if space_fingerprint != candidate.space.fingerprint() {
            return Err(ProducerCompatibilityErrorV1::SpaceMismatch);
        }
        let reference_producer_fingerprint = self.producer.fingerprint();
        let candidate_producer_fingerprint = candidate.producer.fingerprint();
        if reference_producer_fingerprint != candidate_producer_fingerprint {
            return Err(ProducerCompatibilityErrorV1::CertificateRequired);
        }
        Ok(ProducerCompatibilityWitnessV1 {
            kind: ProducerCompatibilityKindV1::Exact,
            space_fingerprint,
            reference_producer_fingerprint,
            candidate_producer_fingerprint,
            certificate_fingerprint: None,
            policy_fingerprint: None,
            golden_fixture_fingerprint: None,
            certificate_revision: None,
            expires_at_unix_seconds: None,
        })
    }

    /// Prove that a distinct producer is admissible under an explicitly trusted
    /// conformance certificate.
    ///
    /// The certificate is direction-sensitive and binds the exact producer pair,
    /// mathematical space, independently executed golden fixture, policy,
    /// revision, and validity interval. The trusted context must independently
    /// pin the certificate's canonical fingerprint; copying fields from the raw
    /// certificate into the context does not establish a trustworthy boundary.
    ///
    /// # Errors
    ///
    /// Returns a closed [`ProducerCompatibilityErrorV1`] for invalid identities,
    /// cross-space use, exact-producer misuse, malformed or untrusted certificate
    /// state, binding drift, fixture disagreement, expiry, or revision-policy
    /// violations.
    pub fn verify_certified_foreign_producer_with(
        &self,
        candidate: &Self,
        certificate: &ForeignProducerConformanceCertificateV1,
        trusted: TrustedProducerConformanceContextV1<'_>,
    ) -> Result<ProducerCompatibilityWitnessV1, ProducerCompatibilityErrorV1> {
        validate_reference_and_candidate(self, candidate)?;
        let space_fingerprint = self.space.fingerprint();
        // ubs:ignore — space fingerprints are public compatibility identities, not secrets.
        if space_fingerprint != candidate.space.fingerprint() {
            return Err(ProducerCompatibilityErrorV1::SpaceMismatch);
        }
        let reference_producer_fingerprint = self.producer.fingerprint();
        let candidate_producer_fingerprint = candidate.producer.fingerprint();
        if reference_producer_fingerprint == candidate_producer_fingerprint {
            return Err(ProducerCompatibilityErrorV1::CertificateForbiddenForExactProducer);
        }

        certificate.validate()?;
        trusted.fixture.validate()?;
        let certificate_fingerprint = certificate.fingerprint();
        // ubs:ignore — certificate fingerprints are public integrity identities, not authenticators.
        if certificate_fingerprint != trusted.certificate_fingerprint {
            return Err(ProducerCompatibilityErrorV1::CertificateFingerprintMismatch);
        }
        // ubs:ignore — policy fingerprints are public conformance identities, not secrets.
        if certificate.policy_fingerprint != trusted.policy_fingerprint {
            return Err(ProducerCompatibilityErrorV1::PolicyMismatch);
        }
        if certificate.reference_producer_fingerprint != reference_producer_fingerprint
            || certificate.candidate_producer_fingerprint != candidate_producer_fingerprint
            || certificate.space_fingerprint != space_fingerprint
        {
            return Err(ProducerCompatibilityErrorV1::ProducerBindingMismatch);
        }
        // ubs:ignore — fixture fingerprints are public conformance identities, not secrets.
        if certificate.golden_fixture_fingerprint != trusted.fixture.fingerprint
            // ubs:ignore — golden-vector certificates are public conformance evidence.
            || trusted.fixture.certificate != self.producer.golden_vectors
            // ubs:ignore — golden-vector certificates are public conformance evidence.
            || trusted.fixture.certificate != candidate.producer.golden_vectors
        {
            return Err(ProducerCompatibilityErrorV1::FixtureBindingMismatch);
        }
        if certificate.certificate_revision < trusted.minimum_certificate_revision
            || certificate.certificate_revision > trusted.maximum_certificate_revision
        {
            return Err(ProducerCompatibilityErrorV1::CertificateRevisionOutsidePolicy);
        }
        if trusted.evaluation_time_unix_seconds < certificate.not_before_unix_seconds {
            return Err(ProducerCompatibilityErrorV1::CertificateNotYetValid);
        }
        if trusted.evaluation_time_unix_seconds >= certificate.expires_at_unix_seconds {
            return Err(ProducerCompatibilityErrorV1::CertificateExpired);
        }

        Ok(ProducerCompatibilityWitnessV1 {
            kind: ProducerCompatibilityKindV1::Certified,
            space_fingerprint,
            reference_producer_fingerprint,
            candidate_producer_fingerprint,
            certificate_fingerprint: Some(certificate_fingerprint),
            policy_fingerprint: Some(certificate.policy_fingerprint.clone()),
            golden_fixture_fingerprint: Some(certificate.golden_fixture_fingerprint.clone()),
            certificate_revision: Some(certificate.certificate_revision),
            expires_at_unix_seconds: Some(certificate.expires_at_unix_seconds),
        })
    }

    /// Validate and freeze the exact bytes that persistence must store.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when any component or cross-binding is invalid.
    pub fn freeze(&self) -> Result<FrozenEmbeddingIdentityBundleV1, SearchError> {
        self.validate()?;
        let canonical_bytes = self.canonical_bytes();
        Ok(FrozenEmbeddingIdentityBundleV1 {
            identity: self.clone(),
            fingerprint: sha256_hex(&canonical_bytes),
            canonical_bytes,
        })
    }

    /// Derive a complete identity bundle for a deterministic MRL/projection wrapper.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the parent is invalid or the target is not a
    /// strict non-zero reduction.
    pub fn derive_projection(
        &self,
        target_dimension: u32,
        projection_rule: &str,
        renormalization_rule: &str,
    ) -> Result<Self, SearchError> {
        self.validate()?;
        let mut derived = self.clone();
        derived.space = self.space.derive_projection(
            target_dimension,
            projection_rule,
            renormalization_rule,
        )?;
        let parent_producer_fingerprint = self.producer.fingerprint();
        derived.producer.implementation_revision = format!(
            "frankensearch-identity-projection-wrapper-v1:parent={parent_producer_fingerprint}"
        );
        "deterministic-identity-projection-v1".clone_into(&mut derived.producer.protocol_revision);
        derived.producer.numeric_profile =
            format!("projection-and-renormalization-f32-v1:parent={parent_producer_fingerprint}");
        derived.producer.space_fingerprint = derived.space.fingerprint();
        derived.producer.golden_vectors.dimension = target_dimension;
        let mut certificate = CanonicalEncoder::new(b"frankensearch.projected-golden.v1");
        certificate.text(&self.producer.golden_vectors.vectors_sha256);
        certificate.u32(target_dimension);
        certificate.text(projection_rule);
        certificate.text(renormalization_rule);
        derived.producer.golden_vectors.vectors_sha256 = sha256_hex(&certificate.finish());
        derived.storage.dimension = target_dimension;
        renormalization_rule.clone_into(&mut derived.storage.vector_normalization);
        derived.validate()?;
        Ok(derived)
    }

    /// Construct an explicitly synthetic identity for tests and examples.
    ///
    /// The constructor is intentionally named and tagged as synthetic so it cannot
    /// be mistaken for verified semantic availability.
    #[must_use]
    pub fn explicit_test_model(model_id: &str, dimension: u32) -> Self {
        let seed_digest = sha256_hex(model_id.as_bytes());
        let input = EmbeddingInputContractV1 {
            schema_version: EMBEDDING_INPUT_CONTRACT_SCHEMA_V1,
            canonicalization: "explicit-test-identity-v1".to_owned(),
            content_selection: "caller-provided-test-text".to_owned(),
            chunking: "none".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            doc_id_semantics: "test-only-no-document-binding".to_owned(),
        };
        let profile = HashControlProfileV1 {
            algorithm: "explicit-test-vector-source".to_owned(),
            algorithm_revision: "v1".to_owned(),
            seed: 0,
            feature_rules: model_id.to_owned(),
            tokenization_rules: "test-defined".to_owned(),
            signing_rules: "test-defined".to_owned(),
            normalization_rules: "test-defined".to_owned(),
        };
        let profile_fingerprint = profile.fingerprint();
        let space = EmbeddingSpaceIdentityV1 {
            schema_version: EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
            logical_model_id: model_id.to_owned(),
            immutable_revision: "explicit-test-v1".to_owned(),
            kind: EmbeddingSpaceKindV1::HashControl,
            artifact_manifest_fingerprint: profile_fingerprint.clone(),
            artifacts: Vec::new(),
            tokenizer_fingerprint: seed_digest.clone(),
            vocabulary_fingerprint: seed_digest.clone(),
            model_config_fingerprint: seed_digest.clone(),
            model_preprocessing: "test-defined".to_owned(),
            sequence_policy: "test-defined".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            pooling: "test-defined".to_owned(),
            output_normalization: "test-defined".to_owned(),
            dimension,
            input_contract_fingerprint: input.fingerprint(),
            hash_control: Some(profile),
            projection: None,
        };
        let producer = EmbeddingProducerAttestationV1 {
            schema_version: EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
            backend: "explicit-test-backend".to_owned(),
            implementation_revision: "v1".to_owned(),
            protocol_revision: "in-process-v1".to_owned(),
            numeric_profile: "test-defined-f32".to_owned(),
            provenance_manifest_fingerprint: profile_fingerprint,
            space_fingerprint: space.fingerprint(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: seed_digest.clone(),
                vectors_sha256: seed_digest,
                vector_count: 1,
                dimension,
            },
        };
        Self {
            space,
            producer,
            input,
            storage: VectorStorageIdentityV1 {
                schema_version: VECTOR_STORAGE_IDENTITY_SCHEMA_V1,
                format: "in-memory-test-vector-v1".to_owned(),
                quantization: QuantizationFormat::F32,
                endianness: "native-test-only".to_owned(),
                vector_normalization: "test-defined".to_owned(),
                dimension,
            },
        }
    }
}

impl FrozenEmbeddingIdentityBundleV1 {
    #[cfg(test)]
    pub(crate) fn explicit_test_model(model_id: &str, dimension: u32) -> Self {
        EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension)
            .freeze()
            .expect("explicit test identity must be valid")
    }

    /// Reject unknown schemas, noncanonical bytes, and bytes/digest disagreement.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the structured identity, canonical bytes, or
    /// stored digest disagree.
    pub fn validate(&self) -> Result<(), SearchError> {
        self.identity.validate()?;
        validate_sha256("frozen_bundle.fingerprint", &self.fingerprint)?;
        let canonical_bytes = self.identity.canonical_bytes();
        // ubs:ignore — these canonical authority bytes are public integrity evidence.
        if self.canonical_bytes != canonical_bytes {
            return Err(identity_error(
                "frozen_bundle.canonical_bytes",
                "redacted",
                "stored bytes disagree with canonical structured identity",
            ));
        }
        let fingerprint = sha256_hex(&canonical_bytes);
        // ubs:ignore — this fingerprint is public authority-integrity evidence, not a secret.
        if self.fingerprint != fingerprint {
            return Err(identity_error(
                "frozen_bundle.fingerprint",
                &self.fingerprint,
                "stored digest disagrees with canonical bytes",
            ));
        }
        Ok(())
    }
}

/// Frozen complete identity stored by generation manifests.
///
/// The former display-name/weights/dimension-only structure is deliberately
/// absorbed into exact canonical bytes plus their digest so no ambiguous
/// parallel revision type remains.
pub type EmbedderRevision = FrozenEmbeddingIdentityBundleV1;
