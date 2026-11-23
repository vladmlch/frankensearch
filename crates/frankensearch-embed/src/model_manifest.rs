//! Model manifest definitions and verification helpers.
//!
//! This module is intentionally synchronous and runtime-agnostic:
//! it performs filesystem and hashing work only, and leaves transport/network
//! to higher-level download orchestration.

use serde::{Deserialize, Serialize};

mod catalog;
mod frozen;
mod lifecycle;
mod verification;

#[cfg(test)]
mod frozen_tests;
#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod verification_tests;

pub use catalog::{ModelFile, ModelManifest, ModelManifestCatalog};
pub use frozen::{
    FrozenModelArtifactManifestV1, ModelArtifactFileV1, ModelArtifactManifestV1,
    ModelArtifactRoleV1, ModelExecutionContractV1, VerifiedModelArtifactsV1,
};
pub use lifecycle::{
    ConsentSource, DownloadConsent, ModelLifecycle, ModelState, resolve_download_consent,
};
pub use verification::{
    FileVerificationState, VERIFICATION_MARKER_SCHEMA_VERSION, VerificationMarker,
    is_verification_cached, verify_dir_and_record, verify_dir_cached, verify_file_sha256,
};

/// Environment variable for explicit model-download consent.
pub const DOWNLOAD_CONSENT_ENV: &str = "FRANKENSEARCH_ALLOW_DOWNLOAD";

/// Placeholder checksum used until a model file is downloaded and verified.
pub const PLACEHOLDER_VERIFY_AFTER_DOWNLOAD: &str = "PLACEHOLDER_VERIFY_AFTER_DOWNLOAD";

/// Placeholder revision used by built-in manifests until pinned revisions are filled in.
pub const PLACEHOLDER_PINNED_REVISION: &str = "UNPINNED_VERIFY_AFTER_DOWNLOAD";

/// Schema version for the manifest catalog format.
///
/// Bump this when the manifest structure changes in a backwards-incompatible way.
/// Consumers compare the embedded schema version against the cached manifest to
/// detect model upgrades that require re-download.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

/// Which search tier a model serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Fast tier (~0.57ms per query), lower dimension.
    Fast,
    /// Quality tier (~128ms per query), higher dimension.
    Quality,
    /// Cross-encoder reranker, applied to top-K results.
    Reranker,
}

/// Schema for the frozen artifact-plus-execution manifest.
pub const MODEL_ARTIFACT_MANIFEST_SCHEMA_V1: u16 = 1;

pub(super) const MAX_FROZEN_MANIFEST_FIELD_BYTES: usize = 4_096;

/// Ordered synthetic corpus used by every registered embedding producer's
/// bit-exact conformance certificate.
///
/// These bounded strings contain no user data. Ordering and bytes are part of
/// the certificate contract.
pub const MODEL_CONFORMANCE_TEXTS_V1: [&str; 4] = [
    "hello world",
    "semantic search finds related ideas",
    "identifier fsvi_v2",
    "naive cafe Tokyo",
];

/// Exact truncation and padding policy of the `MiniLM` quality tier under the
/// pinned `FastEmbed` adapter. 256 tokens (bd-7651e) embedded code about 2x
/// faster than 512 with no measured BEIR or code-set loss.
pub(crate) const FASTEMBED_SEQUENCE_POLICY_256: &str =
    "max-length=256;longest-first;batch-longest-padding";
/// Exact truncation and padding policy of every other `FastEmbed` producer.
pub(crate) const FASTEMBED_SEQUENCE_POLICY_512: &str =
    "max-length=512;longest-first;batch-longest-padding";

/// Token budget the pinned `FastEmbed` 7.1.0 adapter passes for a registered
/// sequence policy; `None` for a policy it cannot execute.
#[cfg(feature = "fastembed")]
pub(crate) fn fastembed_max_length(sequence_policy: &str) -> Option<usize> {
    match sequence_policy {
        FASTEMBED_SEQUENCE_POLICY_256 => Some(256),
        FASTEMBED_SEQUENCE_POLICY_512 => Some(512),
        _ => None,
    }
}
/// Exact `FastEmbed` plus adapter-level output normalization pipeline.
pub(crate) const FASTEMBED_OUTPUT_NORMALIZATION_V1: &str =
    "fastembed-l2-eps-1e-12-then-l2-f32-zero-on-degenerate-v1";
/// Exact native `Model2Vec` input preparation and empty/OOV behavior.
pub(crate) const MODEL2VEC_PREPROCESSING_V1: &str =
    "encode-special-tokens=false;discard-oov=true;empty-or-all-oov=zero-vector";
/// V1 plus the reference `model2vec` rule for `WordPiece` models: the
/// tokenizer's unknown-token id is dropped before pooling, as the upstream
/// `StaticModel.encode` does. Used by the English potion-base models (GH #50);
/// the frozen multilingual contract keeps V1.
pub(crate) const MODEL2VEC_PREPROCESSING_V2: &str = "encode-special-tokens=false;discard-oov=true;discard-wordpiece-unk=true;empty-or-all-oov=zero-vector";
/// Exact sequence behavior of the frozen Potion tokenizer.
pub(crate) const MODEL2VEC_SEQUENCE_POLICY_V1: &str = "tokenizer-configured;no-padding";
/// Exact native `Model2Vec` pooling rule.
pub(crate) const MODEL2VEC_POOLING_V1: &str = "mean-in-vocabulary-token-rows-v1";
/// Exact native `Model2Vec` output normalization rule.
pub(crate) const MODEL2VEC_OUTPUT_NORMALIZATION_V1: &str = "l2-f32-zero-on-degenerate-v1";
