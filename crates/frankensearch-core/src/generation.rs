//! Generation manifest schema and validator for Native Mode distributed search.
//!
//! A *generation* represents a complete, consistent snapshot of all search artifacts
//! (vector indices, lexical segments, embedder metadata) built from a contiguous window
//! of document commits. Replicas atomically activate a generation to serve queries,
//! ensuring no mixed-generation reads within a single request.
//!
//! The [`GenerationManifest`] captures everything needed to replicate, verify, and
//! activate a generation on any node. The `ManifestValidator` enforces structural
//! and semantic invariants before activation is permitted.

use serde::{Deserialize, Serialize};
// ---------------------------------------------------------------------------
// Commit range
// ---------------------------------------------------------------------------

/// Contiguous range of commit sequence numbers that produced this generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRange {
    /// First commit (inclusive) in the window.
    pub low: u64,
    /// Last commit (inclusive) in the window.
    pub high: u64,
}

impl CommitRange {
    /// Number of commits covered by this range.
    #[must_use]
    pub const fn len(&self) -> u64 {
        if self.high < self.low {
            return 0;
        }
        self.high - self.low + 1
    }

    /// Whether the range is empty (high < low after wrapping / invalid state).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.high < self.low
    }
}

/// Vector quantization format used in FSVI artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationFormat {
    /// IEEE 754 single-precision (32-bit).
    F32,
    /// IEEE 754 half-precision (16-bit). Default for frankensearch.
    F16,
    /// Signed 8-bit integer with per-vector scale factor.
    Int8,
    /// Signed 4-bit integer packed two per byte.
    Int4,
}

mod activation_manifest;
mod authority_core;
mod authority_local_floor;
mod codec;
mod embedding_identity;
mod manifest;
mod producer_conformance;
mod receipts;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_authority;
#[cfg(test)]
mod tests_identity;
#[cfg(test)]
mod tests_manifest;
#[cfg(test)]
mod tests_receipts;
#[cfg(test)]
mod tests_support;

pub use activation_manifest::{
    ActivationManifest, ActivationManifestV1, ActivationManifestV2, GenerationAuthorityActionV1,
    GenerationComponentReceiptV1, GenerationComponentReceiptsV1, GenerationComponentReceiptsV2,
    verify_authority_manifest_reference, verify_authority_manifest_reference_v1,
};
pub use authority_core::{
    ANTI_ROLLBACK_FLOOR_SCHEMA_V1, AntiRollbackFloorRecordV1, AuthorityFloorV1, AuthorityRefV1,
    AuthoritySlotV1, GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V1,
    GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2, GENERATION_ACTIVATION_MANIFEST_SCHEMA_V2,
    GENERATION_AUTHORITY_SCHEMA_V1, GENERATION_AUTHORITY_SLOT_BYTES_V1,
    GENERATION_LOCK_FRAME_BYTES_V1, GenerationAuthorityErrorV1, GenerationLockFrameKindV1,
    GenerationLockFrameV1, GenerationRootSecurityProfileV1, InMemoryAntiRollbackFloorStoreV1,
    resolve_authority_slot_frames_v1, resolve_authority_slots_at_floor_v1,
    resolve_authority_slots_v1, resolve_authority_slots_with_locks_v1,
    resolve_authority_slots_with_profile_v1,
};
pub use authority_local_floor::LocalAntiRollbackFloorStoreV1;
pub use embedding_identity::{
    ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1, ArtifactGenerationIdentityV1,
    EMBEDDING_INPUT_CONTRACT_SCHEMA_V1, EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
    EMBEDDING_SPACE_IDENTITY_SCHEMA_V1, EmbedderRevision, EmbeddingArtifactIdentityV1,
    EmbeddingIdentityBundleV1, EmbeddingInputContractV1, EmbeddingProducerAttestationV1,
    EmbeddingProjectionV1, EmbeddingSpaceIdentityV1, EmbeddingSpaceKindV1,
    FOREIGN_PRODUCER_CONFORMANCE_CERTIFICATE_SCHEMA_V1, FrozenEmbeddingIdentityBundleV1,
    GoldenVectorCertificateV1, HashControlProfileV1, VECTOR_STORAGE_IDENTITY_SCHEMA_V1,
    VectorStorageIdentityV1,
};
pub use manifest::{
    ActivationInvariant, EmbedderTierTag, FindingSeverity, GenerationManifest, InvariantKind,
    LexicalArtifact, MANIFEST_SCHEMA_VERSION, RepairDescriptor, ValidationFinding,
    ValidationResult, VectorArtifact, compute_manifest_hash, require_valid, validate_manifest,
};
pub use producer_conformance::{
    ForeignProducerConformanceCertificateV1, ProducerCompatibilityErrorV1,
    ProducerCompatibilityKindV1, ProducerCompatibilityWitnessV1,
    TrustedProducerConformanceContextV1, VerifiedGoldenConformanceManifestV1,
};
pub use receipts::{
    CANONICAL_DOCSET_DIGEST_DOMAIN_V1, CanonicalDocsetV1, ComponentJoinErrorV1,
    ExactComponentReceiptV1, ExactGenerationComponentsV1, GENERATION_SOURCE_CHECKPOINT_DOMAIN_V1,
    GenerationComponentRole, MetadataReceiptDriftV1, SourceCheckpointV1,
    generation_source_checkpoint_v1,
};
