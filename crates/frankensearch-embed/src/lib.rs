//! Embedder implementations for the frankensearch hybrid search library.
//!
//! Provides three tiers of text embedding:
//! - **Hash** (`hash` feature, default): FNV-1a hash embedder, zero dependencies, always available.
//! - **`Model2Vec`** (`model2vec` feature): potion-128M static embedder, fast tier (~0.57ms).
//! - **`FastEmbed`** (`fastembed` feature): MiniLM-L6-v2 ONNX embedder, quality tier (~128ms).
//!
//! The `EmbedderStack` auto-detection probes for available models and configures
//! the best fast+quality pair automatically.

pub mod auto_detect;
pub mod batch_coalescer;
#[cfg(feature = "bundled-default-models")]
pub mod bundled_default_models;
pub mod cached_embedder;
pub mod model_cache;
pub mod model_manifest;
pub mod model_registry;
pub mod simd;
pub use auto_detect::{
    DetectOptions, DimReduceEmbedder, EmbedderStack, ModelAvailabilityDiagnostic, ModelStatus,
    TwoTierAvailability,
};
pub use batch_coalescer::{
    BatchCoalescer, CoalescedBatch, CoalescerConfig, CoalescerMetrics, Priority,
};
#[cfg(feature = "bundled-default-models")]
pub use bundled_default_models::{EmbeddedModelInstallSummary, ensure_default_semantic_models};
pub use simd::accumulate_f32_into;

// When bundled-default-models is disabled (lite build), provide a no-op
// `ensure_default_semantic_models` so downstream crates compile without
// feature-gating every call site.
#[cfg(not(feature = "bundled-default-models"))]
pub use lite_fallback::{EmbeddedModelInstallSummary, ensure_default_semantic_models};

#[cfg(not(feature = "bundled-default-models"))]
mod lite_fallback {
    use std::path::{Path, PathBuf};

    use frankensearch_core::error::SearchResult;

    /// Summary returned by the no-op lite-build materialization.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EmbeddedModelInstallSummary {
        /// Effective model root (inherited from caller or platform default).
        pub model_root: PathBuf,
        /// Always 0 in the lite build -- no embedded models to write.
        pub models_written: usize,
        /// Always 0 in the lite build.
        pub bytes_written: u64,
    }

    /// No-op: lite builds have no embedded models to materialize.
    ///
    /// Returns a summary with zero writes. Callers should check for models
    /// on disk at the standard location (`~/.local/share/frankensearch/models/`)
    /// and prompt the user to run `fsfs download-models` if they are missing.
    ///
    /// # Errors
    ///
    /// This lite fallback currently does not return an error; the `Result`
    /// mirrors the bundled-model API used when embedded models are enabled.
    pub fn ensure_default_semantic_models(
        model_root: Option<&Path>,
    ) -> SearchResult<EmbeddedModelInstallSummary> {
        let root = model_root.map(|p| p.to_path_buf()).unwrap_or_else(|| {
            crate::model_registry::ensure_model_storage_layout_checked()
                .unwrap_or_else(|_| PathBuf::from("models"))
        });
        Ok(EmbeddedModelInstallSummary {
            model_root: root,
            models_written: 0,
            bytes_written: 0,
        })
    }
}
pub use cached_embedder::{CacheStats, CachedEmbedder};
pub use model_cache::{
    ENV_DATA_DIR, ENV_MODEL_DIR, KnownModel, MODEL_CACHE_LAYOUT_VERSION, ModelCacheLayout,
    ModelDirEntry, ensure_cache_layout, ensure_default_cache, is_model_installed, known_models,
    model_file_path, resolve_cache_root,
};
pub use model_manifest::{
    ConsentSource, DOWNLOAD_CONSENT_ENV, DownloadConsent, FrozenModelArtifactManifestV1,
    MANIFEST_SCHEMA_VERSION, MODEL_ARTIFACT_MANIFEST_SCHEMA_V1, ModelArtifactFileV1,
    ModelArtifactManifestV1, ModelArtifactRoleV1, ModelExecutionContractV1, ModelFile,
    ModelLifecycle, ModelManifest, ModelManifestCatalog, ModelState, ModelTier,
    PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, VerificationMarker, VerifiedModelArtifactsV1,
    is_verification_cached, resolve_download_consent, verify_dir_and_record, verify_dir_cached,
    verify_file_sha256,
};
pub use model_registry::{
    BAKEOFF_CUTOFF_DATE, EmbedderRegistry, RegisteredEmbedder, RegisteredReranker,
    registered_embedders, registered_rerankers,
};

#[cfg(feature = "hash")]
pub mod hash_embedder;

#[cfg(feature = "hash")]
pub use hash_embedder::{
    HashAlgorithm, HashEmbedder, jl_accumulate_lanes, jl_accumulate_lanes_scalar,
    jl_accumulate_lanes8, jl_accumulate_lanes8_scalar,
};

#[cfg(feature = "model2vec")]
pub mod model2vec_embedder;

#[cfg(feature = "model2vec")]
pub use model2vec_embedder::{Model2VecEmbedder, RegisteredModel2Vec, find_model_dir};

#[cfg(feature = "fastembed")]
pub mod fastembed_embedder;

#[cfg(feature = "fastembed")]
pub use fastembed_embedder::{FastEmbedEmbedder, OnnxEmbedderConfig};

#[cfg(feature = "download")]
pub mod model_download;

#[cfg(feature = "download")]
pub use model_download::{
    DownloadConfig, DownloadProgress, MODEL_ACQUISITION_PROGRESS_SCHEMA_V1,
    MODEL_ACQUISITION_RECEIPT_SCHEMA_V1, MODEL_ACQUISITION_RECOVERY_SCHEMA_V1,
    ModelAcquisitionCacheReasonV1, ModelAcquisitionOutcomeV1, ModelAcquisitionProgressV1,
    ModelAcquisitionReceiptV1, ModelAcquisitionRecoveryV1, ModelAcquisitionRequest,
    ModelAcquisitionSource, ModelAcquisitionSourceKindV1, ModelAcquisitionStageV1,
    ModelAcquisitionVerificationResultV1, ModelDownloader, VerifiedModelStageV1,
    diagnose_model_acquisition,
};

#[cfg(feature = "api")]
pub mod api_provider;

#[cfg(feature = "api")]
pub mod api_embedder;

#[cfg(feature = "api")]
pub use api_embedder::{
    ApiEmbedder, ApiEmbedderConfig, AssumedRemoteApi, AssumedRemoteEmbeddingBatchV1,
    PinnedRemoteAttesterV1, RemoteApiTrustLevelV1,
};
#[cfg(feature = "api")]
pub use api_provider::{
    ApiProvider, GeminiProvider, MIN_REMOTE_ATTESTATION_KEY_BYTES, OpenAiProvider,
    REMOTE_EMBEDDING_ATTESTATION_SCHEMA_V1, REMOTE_EMBEDDING_CHALLENGE_SCHEMA_V1,
    RemoteEmbeddingAttestationV1, RemoteEmbeddingChallengeV1, remote_embedding_payload_sha256,
    remote_endpoint_fingerprint, remote_ordered_request_sha256,
};

#[cfg(test)]
mod build_policy_tests {
    use std::fs;
    use std::path::Path;

    const BUILD_SCRIPT: &str = include_str!("../build.rs");

    fn load_module_subtree(stem: &str) -> String {
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let root_file = src_dir.join(format!("{stem}.rs"));
        let mut combined = fs::read_to_string(&root_file)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", root_file.display()));

        let sub_dir = src_dir.join(stem);
        if sub_dir.is_dir() {
            let mut sub_files = Vec::new();
            let mut stack = vec![sub_dir];
            while let Some(dir) = stack.pop() {
                let entries = fs::read_dir(&dir)
                    .unwrap_or_else(|err| panic!("failed to read {}: {err}", dir.display()));
                for entry in entries {
                    let path = entry
                        .unwrap_or_else(|err| {
                            panic!("failed to inspect entry in {}: {err}", dir.display())
                        })
                        .path();
                    if path.is_dir() {
                        stack.push(path);
                    } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                        sub_files.push(path);
                    }
                }
            }
            sub_files.sort();
            for path in sub_files {
                let content = fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
                combined.push('\n');
                combined.push_str(&content);
            }
        }

        combined
    }

    #[test]
    fn build_script_is_strictly_network_free_and_non_destructive() {
        for forbidden in [
            "Command::new",
            "TcpStream",
            "curl",
            "http://",
            "https://",
            "remove_file",
            "remove_dir",
        ] {
            assert!(
                !BUILD_SCRIPT.contains(forbidden),
                "build.rs contains forbidden network/destructive token {forbidden}"
            );
        }
        assert!(BUILD_SCRIPT.contains("FRANKENSEARCH_BUNDLED_MODELS_SOURCE_DIR"));
        assert!(BUILD_SCRIPT.contains("build.rs is network-free"));
    }

    #[test]
    fn model_identity_logs_do_not_emit_raw_paths() {
        for stem in [
            "auto_detect",
            "fastembed_embedder",
            "model2vec_embedder",
            "model_download",
            "model_registry",
        ] {
            let source = load_module_subtree(stem);
            for forbidden in [
                "path = %",
                "model_dir = %",
                "destination = %",
                "checked_paths =",
                "url = %",
                "error = %",
                "reason = %error",
                "detail = %detail",
                "provider = other",
            ] {
                assert!(
                    !source.contains(forbidden),
                    "model logging source ({stem}) contains raw-path field {forbidden}"
                );
            }
        }
    }
}
