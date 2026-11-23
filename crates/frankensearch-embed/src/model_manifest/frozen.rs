//! Frozen artifact manifest schema, validation, canonical hashing, and identity derivation.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{
    EMBEDDING_INPUT_CONTRACT_SCHEMA_V1, EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
    EMBEDDING_SPACE_IDENTITY_SCHEMA_V1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
    EmbeddingInputContractV1, EmbeddingProducerAttestationV1, EmbeddingSpaceIdentityV1,
    EmbeddingSpaceKindV1, GoldenVectorCertificateV1, QuantizationFormat,
    VECTOR_STORAGE_IDENTITY_SCHEMA_V1, VectorStorageIdentityV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::catalog::ModelManifest;
use super::verification::{
    invalid_manifest_field, is_verification_cached, promote_atomically, resolve_model_file_path,
    sync_registered_artifacts, to_hex_lowercase, validate_model_file_name, verify_file_sha256,
};
use super::{
    FASTEMBED_OUTPUT_NORMALIZATION_V1, FASTEMBED_SEQUENCE_POLICY_256,
    FASTEMBED_SEQUENCE_POLICY_512, MAX_FROZEN_MANIFEST_FIELD_BYTES,
    MODEL_ARTIFACT_MANIFEST_SCHEMA_V1, MODEL_CONFORMANCE_TEXTS_V1,
    MODEL2VEC_OUTPUT_NORMALIZATION_V1, MODEL2VEC_POOLING_V1, MODEL2VEC_PREPROCESSING_V1,
    MODEL2VEC_PREPROCESSING_V2, MODEL2VEC_SEQUENCE_POLICY_V1,
};

/// Semantic role of one artifact in a frozen model bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArtifactRoleV1 {
    /// Learned weights or static embedding table.
    Weights,
    /// Tokenizer implementation/configuration.
    Tokenizer,
    /// Standalone vocabulary.
    Vocabulary,
    /// Model architecture/configuration.
    ModelConfig,
    /// Special-token mapping.
    SpecialTokens,
    /// Tokenizer runtime configuration.
    TokenizerConfig,
    /// Projection/MRL matrix or descriptor.
    Projection,
}

impl ModelArtifactRoleV1 {
    const fn tag(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::Tokenizer => "tokenizer",
            Self::Vocabulary => "vocabulary",
            Self::ModelConfig => "model_config",
            Self::SpecialTokens => "special_tokens",
            Self::TokenizerConfig => "tokenizer_config",
            Self::Projection => "projection",
        }
    }
}

/// One role-tagged immutable artifact in a frozen model manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactFileV1 {
    /// Semantic role. Roles are unique within a manifest.
    pub role: ModelArtifactRoleV1,
    /// Safe relative path inside the model directory.
    pub relative_path: String,
    /// Immutable upstream URL. Never emitted in structured runtime logs.
    pub upstream_url: String,
    /// Exact byte size.
    pub size: u64,
    /// Exact lowercase SHA-256.
    pub sha256: String,
}

/// Execution semantics that turn verified artifacts into one mathematical space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelExecutionContractV1 {
    /// Native/provider backend family.
    pub backend: String,
    /// Immutable implementation revision.
    pub implementation_revision: String,
    /// Inference or wire protocol revision.
    pub protocol_revision: String,
    /// Numeric execution profile.
    pub numeric_profile: String,
    /// Weights/container format.
    pub weights_format: String,
    /// Tokenizer implementation identity.
    pub tokenizer_family: String,
    /// Model-internal preprocessing.
    pub model_preprocessing: String,
    /// Sequence length, truncation, and padding.
    pub sequence_policy: String,
    /// Pooling rule.
    pub pooling: String,
    /// Output normalization.
    pub output_normalization: String,
    /// Query instruction.
    pub query_instruction: String,
    /// Document instruction.
    pub document_instruction: String,
    /// Outer content-selection/canonicalization contract.
    pub input_contract: EmbeddingInputContractV1,
    /// Pinned implementation conformance certificate.
    pub golden_vectors: GoldenVectorCertificateV1,
}

/// Complete frozen model artifact manifest shared by local/native backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactManifestV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Stable artifact provider.
    pub provider: String,
    /// Semantic model identifier.
    pub logical_model_id: String,
    /// Immutable upstream model revision.
    pub upstream_revision: String,
    /// Immutable upstream repository identity.
    pub upstream_repository: String,
    /// Unique role-tagged artifacts.
    pub artifacts: Vec<ModelArtifactFileV1>,
    /// SPDX license identifier.
    pub license_spdx: String,
    /// Digest of the canonical pinned license metadata assertion.
    pub license_metadata_sha256: String,
    /// Output dimension.
    pub dimension: u32,
    /// Complete execution semantics.
    pub execution: ModelExecutionContractV1,
}

/// Canonical bytes and digest proven to correspond to a validated manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenModelArtifactManifestV1 {
    /// Validated structured manifest.
    pub manifest: ModelArtifactManifestV1,
    /// Domain-separated canonical bytes.
    pub canonical_bytes: Vec<u8>,
    /// SHA-256 of `canonical_bytes`.
    pub fingerprint: String,
}

/// Proof that every file in a frozen manifest was verified in one directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedModelArtifactsV1 {
    frozen: FrozenModelArtifactManifestV1,
}

impl VerifiedModelArtifactsV1 {
    /// Validated frozen manifest.
    #[must_use]
    pub const fn frozen(&self) -> &FrozenModelArtifactManifestV1 {
        &self.frozen
    }

    /// Derive the complete runtime identity from verified artifacts.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if any cross-contract fingerprint is inconsistent.
    pub fn identity_bundle(
        &self,
        quantization: QuantizationFormat,
        storage_format: &str,
    ) -> SearchResult<EmbeddingIdentityBundleV1> {
        self.frozen
            .manifest
            .identity_bundle(quantization, storage_format)
    }
}

impl ModelArtifactManifestV1 {
    /// Frozen native `Model2Vec` contract for the built-in potion fast tier.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the built-in pinned download manifest drifts
    /// from the registered execution contract.
    pub fn potion_128m_native() -> SearchResult<Self> {
        Self::model2vec_native(
            &ModelManifest::potion_128m(),
            MODEL2VEC_PREPROCESSING_V1,
            "f7dabe71dbb62abf9271f9568d799accb558b982521d3a48337c6a760d7e6c74",
        )
    }

    /// Frozen native `Model2Vec` contract for the opt-in English
    /// `potion-base-8M` fast tier (GH #50).
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the pinned download manifest drifts from the
    /// registered execution contract.
    pub fn potion_base_8m_native() -> SearchResult<Self> {
        Self::model2vec_native(
            &ModelManifest::potion_base_8m(),
            MODEL2VEC_PREPROCESSING_V2,
            "b4a36ff36976bbfa263a13d6b9da4dbc853293400898f6d9cbd0acb6c622391f",
        )
    }

    /// Frozen native `Model2Vec` contract for the opt-in English
    /// `potion-base-32M` fast tier (GH #50).
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the pinned download manifest drifts from the
    /// registered execution contract.
    pub fn potion_base_32m_native() -> SearchResult<Self> {
        Self::model2vec_native(
            &ModelManifest::potion_base_32m(),
            MODEL2VEC_PREPROCESSING_V2,
            "d8c5ddc0d2f252035e5d9dd8b06337e43bae822c69291950f7865df0773d24c1",
        )
    }

    /// The one native `Model2Vec` execution contract: every registered potion
    /// model runs the same tokenizer/row-gather/mean/L2 backend, so only its
    /// pinned artifacts, its preprocessing rule and its exact output
    /// certificate differ.
    fn model2vec_native(
        download: &ModelManifest,
        model_preprocessing: &str,
        vectors_sha256: &str,
    ) -> SearchResult<Self> {
        let dimension = download.dimension.ok_or_else(|| {
            invalid_manifest_field(
                "dimension",
                &download.id,
                "embedding manifests require a fixed output dimension",
            )
        })?;
        let execution = ModelExecutionContractV1 {
            backend: "model2vec-native".to_owned(),
            implementation_revision: format!(
                "frankensearch-embed-{}+model2vec-native-v1",
                env!("CARGO_PKG_VERSION")
            ),
            protocol_revision: "tokenizers-0.23.2+safetensors-0.8.0-static-table-v1".to_owned(),
            numeric_profile: "f32-row-gather-mean-l2-v1".to_owned(),
            weights_format: "safetensors-f32-matrix-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
            model_preprocessing: model_preprocessing.to_owned(),
            sequence_policy: MODEL2VEC_SEQUENCE_POLICY_V1.to_owned(),
            pooling: MODEL2VEC_POOLING_V1.to_owned(),
            output_normalization: MODEL2VEC_OUTPUT_NORMALIZATION_V1.to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint()?,
                vectors_sha256: vectors_sha256.to_owned(),
                vector_count: 4,
                dimension,
            },
        };
        Self::from_download_manifest(download, "minishlab-huggingface", execution)
    }

    /// Frozen FastEmbed/ONNX contract for the built-in `MiniLM` quality tier.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the built-in pinned download manifest drifts
    /// from the registered execution contract.
    pub fn minilm_fastembed() -> SearchResult<Self> {
        let execution = ModelExecutionContractV1 {
            backend: "fastembed-onnx".to_owned(),
            implementation_revision: format!(
                "frankensearch-embed-{}+fastembed-7.1.0",
                env!("CARGO_PKG_VERSION")
            ),
            protocol_revision: "fastembed-7.1.0+ort-2.0.0-rc.13-user-defined-onnx-v1".to_owned(),
            numeric_profile: "ort-2.0.0-rc.13-cpu-f32-host-default-intra-threads-v1".to_owned(),
            weights_format: "onnx-opset-pinned-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
            model_preprocessing: "bert-tokenizer-special-tokens=true;empty-input=zero-vector"
                .to_owned(),
            sequence_policy: FASTEMBED_SEQUENCE_POLICY_256.to_owned(),
            pooling: "attention-mask-mean-pool-v1".to_owned(),
            output_normalization: FASTEMBED_OUTPUT_NORMALIZATION_V1.to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint()?,
                // GOLDEN-CHANGE bd-2ba5: ORT 1.28.0 (da9b5e3) replaces
                // 1.24.2 (058787c). Relinking the latter with the current Rust
                // adapter reproduces the historical certificate exactly.
                // The loader now verifies this certificate before attesting it.
                vectors_sha256: "67cec04aef931fb5b5db8be074e92370c4f62e6f89ec45bec5ecd52a2444d6c3"
                    .to_owned(),
                vector_count: 4,
                dimension: 384,
            },
        };
        Self::from_download_manifest(
            &ModelManifest::minilm_v2(),
            "sentence-transformers-huggingface",
            qualify_fastembed_platform(
                execution,
                "5693dd454b03d7c4ae3a96ea429eddbbaf60519e11ded361a26a1f221f996843",
                "26db12f2e6f084f991d61a29da7f6c66c1e70981b74f07cc73c7f235ee6c0142",
            ),
        )
    }

    /// Frozen pure-Rust frankentorch/safetensors contract for `MiniLM`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the immutable upstream artifact metadata or
    /// native execution contract is incomplete.
    pub fn minilm_native_frankentorch() -> SearchResult<Self> {
        let download_manifest = ModelManifest::minilm_v2_native_weights()?;
        let execution = ModelExecutionContractV1 {
            backend: "frankentorch-native-minilm".to_owned(),
            implementation_revision:
                "frankensearch-rerank-native-embedder-v1+frankentorch-c305306b251753099620ad5fe02e78c07c167cf6"
                    .to_owned(),
            protocol_revision:
                "tokenizers-0.23.2+frankentorch-bert-encoder-v1".to_owned(),
            numeric_profile: "f32-weights-int8-linear-f32-accumulate-v2".to_owned(),
            weights_format: "safetensors-f32-runtime-int8-linear-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
            model_preprocessing: "bert-special-tokens=true;token-type-ids=zero".to_owned(),
            // 256 tokens, as the FastEmbed MiniLM tier (bd-7651e).
            sequence_policy: "max-length=256;longest-first;no-padding".to_owned(),
            pooling: "mean-all-returned-tokens-including-specials-no-padding-v1".to_owned(),
            output_normalization: "l2-f32-if-norm-gt-zero-else-unchanged-v1".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint()?,
                vectors_sha256: "bed15455ed5910d6ebcf39b28a22e321d83a904f99c53e79724995b662e67c26"
                    .to_owned(),
                vector_count: 4,
                dimension: 384,
            },
        };
        Self::from_download_manifest(
            &download_manifest,
            "sentence-transformers-huggingface",
            execution,
        )
    }

    /// Explicit full-F32 native `MiniLM` producer. Its artifact bytes, tokenizer
    /// and pooling match the int8 model, but its numeric execution and output
    /// certificate identify a separate producer. Existing int8 indexes cannot
    /// silently accept these vectors.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the pinned artifact or execution metadata is invalid.
    pub fn minilm_native_frankentorch_f32() -> SearchResult<Self> {
        let mut manifest = Self::minilm_native_frankentorch()?;
        "frankentorch-native-minilm-f32".clone_into(&mut manifest.execution.backend);
        "frankensearch-rerank-native-embedder-f32-v1+frankentorch-0.1.0"
            .clone_into(&mut manifest.execution.implementation_revision);
        "f32-weights-f32-linear-f32-accumulate-v1"
            .clone_into(&mut manifest.execution.numeric_profile);
        "safetensors-f32-runtime-f32-linear-v1".clone_into(&mut manifest.execution.weights_format);
        // GOLDEN-CHANGE bd-2ba5: new explicit precision profile, measured from
        // the same four conformance texts. The int8 certificate above is retained.
        "aa230ec70ebb5c43bb0e08751c57e6f5bb6ccf9b314c292daf1bd979370fc2f1"
            .clone_into(&mut manifest.execution.golden_vectors.vectors_sha256);
        manifest.validate()?;
        Ok(manifest)
    }

    /// Frozen pure-Rust Frankentorch contract for the opt-in multilingual
    /// `paraphrase-multilingual-MiniLM-L12-v2` embedding space.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the immutable upstream artifact metadata or
    /// native execution contract is incomplete.
    pub fn multilingual_minilm_native_frankentorch() -> SearchResult<Self> {
        let execution = ModelExecutionContractV1 {
            backend: "frankentorch-native-multilingual-minilm".to_owned(),
            implementation_revision: "frankensearch-rerank-native-embedder-v2+frankentorch-0.1.0"
                .to_owned(),
            protocol_revision: "tokenizers-0.23.2-xlmr-unigram+frankentorch-bert-dynamic-layers-v2"
                .to_owned(),
            numeric_profile: "f32-weights-int8-linear-f32-accumulate-v2".to_owned(),
            weights_format: "safetensors-f32-runtime-int8-linear-v1".to_owned(),
            tokenizer_family: "huggingface-tokenizers-json-xlmr-unigram-v1".to_owned(),
            model_preprocessing: "xlmr-special-tokens=true;token-type-ids=zero".to_owned(),
            sequence_policy: "max-length=512;longest-first;no-padding".to_owned(),
            pooling: "mean-all-returned-tokens-including-specials-no-padding-v1".to_owned(),
            output_normalization: "l2-f32-if-norm-gt-zero-else-unchanged-v1".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            input_contract: default_plain_text_input_contract(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: conformance_corpus_fingerprint()?,
                vectors_sha256: "c7dcf38c4aff04846e5457e658da704dd3d8ac177182ec2104411c064be7ec7d"
                    .to_owned(),
                vector_count: 4,
                dimension: 384,
            },
        };
        Self::from_download_manifest(
            &ModelManifest::multilingual_minilm_l12_v2(),
            "sentence-transformers-huggingface",
            execution,
        )
    }

    /// Frozen `FastEmbed` contract for Snowflake Arctic Embed S.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the pinned manifest or contract is incomplete.
    pub fn snowflake_fastembed() -> SearchResult<Self> {
        let execution = fastembed_execution_contract(
            384,
            "snowflake-arctic-embed-s",
            // GOLDEN-CHANGE bd-2ba5: separately measured Snowflake output
            // under ORT 1.28.0; the historical 1.24.2 certificate is retained
            // as a real loader-refusal regression alongside MiniLM's.
            "8ab295190de5eb629ef7920e3aec6d989c1b7f695b4f75baebfb716fb81b7f6c",
        )?;
        Self::from_download_manifest(
            &ModelManifest::snowflake_arctic_s(),
            "snowflake-huggingface",
            qualify_fastembed_platform(
                execution,
                "f9f9b1071d82dd22614086a7a0e05bcdc785ca3b04158c4f914e678c75fd6c8d",
                "01e4e1fb6c2e5e4234080ce5644d13cb08ddece97e6e31fa39d49083cea37c37",
            ),
        )
    }

    /// Frozen `FastEmbed` contract for Nomic Embed Text v1.5.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the pinned manifest or contract is incomplete.
    pub fn nomic_fastembed() -> SearchResult<Self> {
        let execution = fastembed_execution_contract(
            768,
            "nomic-embed-text-v1.5",
            "dbb7e33fdb5ccb4864faf9ff425b35a83a2d9dcd4f8d736033d7f819e0c1e851",
        )?;
        Self::from_download_manifest(
            &ModelManifest::nomic_embed(),
            "nomic-ai-huggingface",
            qualify_fastembed_platform(
                execution,
                "7041b782516edfb91097d668443130d098bca7a035c8a85024150b5a09aebc67",
                "8b2d7421218d8ab9e6d0f9c3ea66b371be56324a5560c13e2b416c5064e51f05",
            ),
        )
    }

    /// Convert the existing pinned download manifest plus explicit execution
    /// semantics into the frozen identity contract.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for unknown/duplicate roles or incomplete metadata.
    pub fn from_download_manifest(
        manifest: &ModelManifest,
        provider: &str,
        execution: ModelExecutionContractV1,
    ) -> SearchResult<Self> {
        manifest.validate()?;
        if !manifest.is_production_ready() {
            return Err(invalid_manifest_field(
                "production_ready",
                &manifest.id,
                "frozen manifests require pinned revision, size, and SHA-256 for every artifact",
            ));
        }
        let dimension = manifest.dimension.ok_or_else(|| {
            invalid_manifest_field(
                "dimension",
                &manifest.id,
                "embedding manifests require a fixed output dimension",
            )
        })?;
        let artifacts = manifest
            .files
            .iter()
            .map(|file| {
                Ok(ModelArtifactFileV1 {
                    role: artifact_role_for_path(&file.name)?,
                    relative_path: file.name.clone(),
                    upstream_url: manifest.download_url(file),
                    size: file.size,
                    sha256: file.sha256.clone(),
                })
            })
            .collect::<SearchResult<Vec<_>>>()?;
        let license_metadata_sha256 = license_metadata_fingerprint(
            &manifest.license,
            provider,
            &manifest.repo,
            &manifest.revision,
        );
        let frozen = Self {
            schema_version: MODEL_ARTIFACT_MANIFEST_SCHEMA_V1,
            provider: provider.to_owned(),
            logical_model_id: manifest.id.clone(),
            upstream_revision: manifest.revision.clone(),
            upstream_repository: manifest.repo.clone(),
            artifacts,
            license_spdx: manifest.license.clone(),
            license_metadata_sha256,
            dimension,
            execution,
        };
        frozen.validate()?;
        Ok(frozen)
    }

    /// Validate schema, role uniqueness, hashes, sizes, license, and execution semantics.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for any incomplete or ambiguous contract.
    pub fn validate(&self) -> SearchResult<()> {
        if self.schema_version != MODEL_ARTIFACT_MANIFEST_SCHEMA_V1 {
            return Err(invalid_manifest_field(
                "artifact_manifest.schema_version",
                &self.schema_version.to_string(),
                "unsupported frozen artifact manifest schema",
            ));
        }
        for (field, value) in [
            ("provider", self.provider.as_str()),
            ("logical_model_id", self.logical_model_id.as_str()),
            ("upstream_repository", self.upstream_repository.as_str()),
            ("license_spdx", self.license_spdx.as_str()),
            ("execution.backend", self.execution.backend.as_str()),
            (
                "execution.implementation_revision",
                self.execution.implementation_revision.as_str(),
            ),
            (
                "execution.protocol_revision",
                self.execution.protocol_revision.as_str(),
            ),
            (
                "execution.numeric_profile",
                self.execution.numeric_profile.as_str(),
            ),
            (
                "execution.weights_format",
                self.execution.weights_format.as_str(),
            ),
            (
                "execution.tokenizer_family",
                self.execution.tokenizer_family.as_str(),
            ),
            (
                "execution.model_preprocessing",
                self.execution.model_preprocessing.as_str(),
            ),
            (
                "execution.sequence_policy",
                self.execution.sequence_policy.as_str(),
            ),
            ("execution.pooling", self.execution.pooling.as_str()),
            (
                "execution.output_normalization",
                self.execution.output_normalization.as_str(),
            ),
        ] {
            validate_frozen_manifest_text(field, value, false)?;
        }
        for (field, value) in [
            (
                "execution.query_instruction",
                self.execution.query_instruction.as_str(),
            ),
            (
                "execution.document_instruction",
                self.execution.document_instruction.as_str(),
            ),
        ] {
            validate_frozen_manifest_text(field, value, true)?;
        }
        if !matches!(self.upstream_revision.len(), 40 | 64)
            || !self
                .upstream_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid_manifest_field(
                "upstream_revision",
                &self.upstream_revision,
                "must be a complete lowercase 40- or 64-character immutable commit digest",
            ));
        }
        if self.dimension == 0 {
            return Err(invalid_manifest_field(
                "dimension",
                "0",
                "must be greater than zero",
            ));
        }
        validate_frozen_sha256("license_metadata_sha256", &self.license_metadata_sha256)?;
        let expected_license_metadata = license_metadata_fingerprint(
            &self.license_spdx,
            &self.provider,
            &self.upstream_repository,
            &self.upstream_revision,
        );
        if self.license_metadata_sha256 != expected_license_metadata {
            return Err(invalid_manifest_field(
                "license_metadata_sha256",
                &self.license_metadata_sha256,
                "must bind the canonical SPDX, provider, repository, and immutable revision assertion",
            ));
        }
        self.execution.input_contract.validate()?;
        self.execution.golden_vectors.validate().map_err(|error| {
            invalid_manifest_field(
                "execution.golden_vectors",
                &self.logical_model_id,
                &error.to_string(),
            )
        })?;
        if self.execution.golden_vectors.dimension != self.dimension {
            return Err(invalid_manifest_field(
                "execution.golden_vectors.dimension",
                &self.execution.golden_vectors.dimension.to_string(),
                "must equal manifest dimension",
            ));
        }

        let mut roles = std::collections::BTreeSet::new();
        let mut paths = std::collections::BTreeSet::new();
        for artifact in &self.artifacts {
            validate_frozen_manifest_text(
                "artifacts[].relative_path",
                &artifact.relative_path,
                false,
            )?;
            validate_model_file_name(&artifact.relative_path)?;
            validate_frozen_sha256("artifacts[].sha256", &artifact.sha256)?;
            if artifact.size == 0 {
                return Err(invalid_manifest_field(
                    "artifacts[].size",
                    "0",
                    "must be greater than zero",
                ));
            }
            validate_frozen_manifest_text(
                "artifacts[].upstream_url",
                &artifact.upstream_url,
                false,
            )?;
            if !artifact.upstream_url.starts_with("https://")
                || artifact
                    .upstream_url
                    .bytes()
                    .any(|byte| byte.is_ascii_whitespace())
                || artifact.upstream_url.contains('@')
                || artifact.upstream_url.contains('?')
                || artifact.upstream_url.contains('#')
            {
                return Err(invalid_manifest_field(
                    "artifacts[].upstream_url",
                    "redacted",
                    "must be credential-free HTTPS without userinfo, query, or fragment",
                ));
            }
            if !roles.insert(artifact.role) {
                return Err(invalid_manifest_field(
                    "artifacts[].role",
                    artifact.role.tag(),
                    "duplicate semantic artifact role",
                ));
            }
            if !paths.insert(artifact.relative_path.as_str()) {
                return Err(invalid_manifest_field(
                    "artifacts[].relative_path",
                    &artifact.relative_path,
                    "duplicate artifact path",
                ));
            }
        }
        if !roles.contains(&ModelArtifactRoleV1::Weights)
            || !roles.contains(&ModelArtifactRoleV1::Tokenizer)
        {
            return Err(invalid_manifest_field(
                "artifacts",
                &self.logical_model_id,
                "embedding manifests require unique weights and tokenizer roles",
            ));
        }
        Ok(())
    }

    /// Domain-separated canonical bytes. Artifact order is canonicalized by the
    /// frozen serialized role tag rather than enum declaration order.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_frozen_bytes(&mut bytes, b"frankensearch.model-artifact-manifest.v1");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        for value in [
            &self.provider,
            &self.logical_model_id,
            &self.upstream_revision,
            &self.upstream_repository,
            &self.license_spdx,
            &self.license_metadata_sha256,
        ] {
            append_frozen_bytes(&mut bytes, value.as_bytes());
        }
        bytes.extend_from_slice(&self.dimension.to_be_bytes());
        let mut artifacts = self.artifacts.iter().collect::<Vec<_>>();
        artifacts.sort_by_key(|artifact| artifact.role.tag());
        append_frozen_len(&mut bytes, artifacts.len());
        for artifact in artifacts {
            append_frozen_bytes(&mut bytes, artifact.role.tag().as_bytes());
            append_frozen_bytes(&mut bytes, artifact.relative_path.as_bytes());
            append_frozen_bytes(&mut bytes, artifact.upstream_url.as_bytes());
            bytes.extend_from_slice(&artifact.size.to_be_bytes());
            append_frozen_bytes(&mut bytes, artifact.sha256.as_bytes());
        }
        for value in [
            &self.execution.backend,
            &self.execution.implementation_revision,
            &self.execution.protocol_revision,
            &self.execution.numeric_profile,
            &self.execution.weights_format,
            &self.execution.tokenizer_family,
            &self.execution.model_preprocessing,
            &self.execution.sequence_policy,
            &self.execution.query_instruction,
            &self.execution.document_instruction,
            &self.execution.pooling,
            &self.execution.output_normalization,
        ] {
            append_frozen_bytes(&mut bytes, value.as_bytes());
        }
        append_frozen_bytes(&mut bytes, &self.execution.input_contract.canonical_bytes());
        append_frozen_bytes(
            &mut bytes,
            self.execution.golden_vectors.corpus_sha256.as_bytes(),
        );
        append_frozen_bytes(
            &mut bytes,
            self.execution.golden_vectors.vectors_sha256.as_bytes(),
        );
        bytes.extend_from_slice(&self.execution.golden_vectors.vector_count.to_be_bytes());
        bytes.extend_from_slice(&self.execution.golden_vectors.dimension.to_be_bytes());
        bytes
    }

    /// Fingerprint only the fields that define the mathematical embedding
    /// space, excluding producer/backend and distribution metadata.
    ///
    /// This deliberately omits provider, repository URL, license, backend,
    /// implementation/protocol revision, numeric execution profile, and golden
    /// certificate. Those remain covered by the full frozen manifest and the
    /// separately bound producer attestation. Consequently, two conformant
    /// implementations over the same artifacts and model semantics may share a
    /// space identity without pretending to be the same producer.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the complete manifest is invalid.
    pub fn space_contract_fingerprint(&self) -> SearchResult<String> {
        self.validate()?;
        Ok(sha256_hex_bytes(&self.space_contract_canonical_bytes()))
    }

    fn space_contract_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_frozen_bytes(
            &mut bytes,
            b"frankensearch.model-artifact-space-contract.v1",
        );
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        append_frozen_bytes(&mut bytes, self.logical_model_id.as_bytes());
        append_frozen_bytes(&mut bytes, self.upstream_revision.as_bytes());
        bytes.extend_from_slice(&self.dimension.to_be_bytes());

        let mut artifacts = self.artifacts.iter().collect::<Vec<_>>();
        artifacts.sort_by_key(|artifact| artifact.role.tag());
        append_frozen_len(&mut bytes, artifacts.len());
        for artifact in artifacts {
            append_frozen_bytes(&mut bytes, artifact.role.tag().as_bytes());
            bytes.extend_from_slice(&artifact.size.to_be_bytes());
            append_frozen_bytes(&mut bytes, artifact.sha256.as_bytes());
        }

        for value in [
            &self.execution.weights_format,
            &self.execution.tokenizer_family,
            &self.execution.model_preprocessing,
            &self.execution.sequence_policy,
            &self.execution.query_instruction,
            &self.execution.document_instruction,
            &self.execution.pooling,
            &self.execution.output_normalization,
        ] {
            append_frozen_bytes(&mut bytes, value.as_bytes());
        }
        append_frozen_bytes(&mut bytes, &self.execution.input_contract.canonical_bytes());
        bytes
    }

    /// Validate and freeze exact canonical bytes plus their fingerprint.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the manifest is incomplete.
    pub fn freeze(&self) -> SearchResult<FrozenModelArtifactManifestV1> {
        self.validate()?;
        let canonical_bytes = self.canonical_bytes();
        Ok(FrozenModelArtifactManifestV1 {
            manifest: self.clone(),
            fingerprint: sha256_hex_bytes(&canonical_bytes),
            canonical_bytes,
        })
    }

    /// Verify all registered files and reject partial layouts or unregistered
    /// files that collide with a registered semantic artifact role. Unrelated
    /// cache metadata such as notices may coexist with the artifact set.
    ///
    /// # Errors
    ///
    /// Returns a typed search error on missing, unexpected, size-drifted, or
    /// hash-drifted artifacts.
    pub fn verify_dir(&self, model_dir: &Path) -> SearchResult<VerifiedModelArtifactsV1> {
        let frozen = self.freeze()?;
        let mut failures = Vec::new();
        for artifact in &self.artifacts {
            let path = resolve_model_file_path(model_dir, &artifact.relative_path)?;
            match verify_file_sha256(&path, &artifact.sha256, artifact.size) {
                Ok(()) => {}
                Err(SearchError::ModelNotFound { .. }) => {
                    failures.push((artifact, "missing"));
                }
                Err(SearchError::HashMismatch { .. }) => {
                    failures.push((artifact, "sha256-or-size-mismatch"));
                }
                Err(SearchError::ModelLoadFailed { .. } | SearchError::Io(_)) => {
                    failures.push((artifact, "unreadable-or-not-regular-file"));
                }
                Err(error) => return Err(error),
            }
        }
        if !failures.is_empty() {
            let mut detail = String::new();
            for (artifact, reason) in &failures {
                if !detail.is_empty() {
                    detail.push_str("; ");
                }
                let _ = write!(
                    detail,
                    "{}:{}:{reason}:fetch={}",
                    artifact.role.tag(),
                    artifact.relative_path,
                    artifact.upstream_url
                );
            }
            // ubs:ignore — reason is a public diagnostic tag, not a secret.
            if failures.iter().all(|(_, reason)| *reason == "missing") {
                return Err(SearchError::ModelNotFound {
                    name: format!(
                        "{} (incomplete frozen artifact set: {detail})",
                        self.logical_model_id
                    ),
                });
            }
            return Err(SearchError::ModelLoadFailed {
                path: PathBuf::from("<redacted-model-dir>"),
                source: format!("frozen artifact set rejected: {detail}").into(),
            });
        }
        verify_no_extra_artifacts(model_dir, &self.artifacts)?;
        Ok(VerifiedModelArtifactsV1 { frozen })
    }

    /// Verify a model directory against this frozen manifest, reusing the
    /// download manifest's verification receipt when it is still valid.
    ///
    /// A native manifest is derived from its download manifest by
    /// [`Self::from_download_manifest`], so both describe one artifact set.
    /// When `download_manifest` carries a valid `.verified` receipt for
    /// `model_dir` (minted only by [`verify_dir_and_record`](super::verify_dir_and_record) after a full
    /// SHA-256 pass, and invalidated by any size, mtime, or identity change)
    /// and every artifact here matches that manifest's file entry on relative
    /// path, size, and SHA-256, the full hash pass is skipped. This is what
    /// keeps a 512 MB model from being re-hashed on every process start. Any
    /// mismatch, missing receipt, or stale receipt falls back to
    /// [`Self::verify_dir`]; nothing is minted here.
    /// The registered native `MiniLM` installation alias is also accepted,
    /// but only for its exact built-in download and producer manifests.
    /// Its distinct catalog id never changes the frozen producer identity.
    ///
    /// # Errors
    ///
    /// Same as [`Self::verify_dir`].
    pub fn verify_dir_cached(
        &self,
        download_manifest: &ModelManifest,
        model_dir: &Path,
    ) -> SearchResult<VerifiedModelArtifactsV1> {
        if (self.artifacts_match_download_manifest(download_manifest)
            || self.matches_registered_native_installation(download_manifest))
            && is_verification_cached(download_manifest, model_dir)
        {
            let frozen = self.freeze()?;
            verify_no_extra_artifacts(model_dir, &self.artifacts)?;
            // This is the receipt actually admitted above, not an observation
            // from status/doctor or an unchecked marker found on disk.
            let receipt_manifest = download_manifest.freeze_verification_manifest()?;
            tracing::info!(
                manifest_id = %download_manifest.id,
                receipt_manifest_fingerprint = %receipt_manifest.fingerprint,
                "model verification receipt accepted"
            );
            return Ok(VerifiedModelArtifactsV1 { frozen });
        }
        self.verify_dir(model_dir)
    }

    /// Bind the separate native installation catalog entry to the two already
    /// registered `MiniLM` producers. Arbitrary logical-id aliases, modified
    /// artifacts, and modified execution contracts cannot borrow this receipt.
    pub(super) fn matches_registered_native_installation(
        &self,
        download_manifest: &ModelManifest,
    ) -> bool {
        if download_manifest != &ModelManifest::minilm_v2_native() {
            return false;
        }
        Self::minilm_native_frankentorch().is_ok_and(|registered| self == &registered)
            || Self::minilm_native_frankentorch_f32().is_ok_and(|registered| self == &registered)
    }

    /// True when every artifact here is byte-for-byte the same file entry
    /// (relative path, size, SHA-256) as in `download_manifest`, and the two
    /// describe the same logical model with the same number of files.
    pub(super) fn artifacts_match_download_manifest(
        &self,
        download_manifest: &ModelManifest,
    ) -> bool {
        if self.logical_model_id != download_manifest.id
            || self.artifacts.len() != download_manifest.files.len()
        {
            return false;
        }
        self.artifacts.iter().all(|artifact| {
            download_manifest.files.iter().any(|file| {
                // A download-manifest file is keyed by `name`, a frozen
                // artifact by `relative_path`; clippy's operator-grouping
                // heuristic misreads the pair, so bind it first.
                let same_path = file.name == artifact.relative_path;
                same_path
                    && file.size == artifact.size
                    && file.sha256.eq_ignore_ascii_case(&artifact.sha256)
            })
        })
    }

    /// Promote a fully verified frozen artifact set into its final directory.
    ///
    /// Every registered file is synced before the sibling-directory rename.
    /// An existing generation is moved to a unique backup and is never deleted.
    ///
    /// # Errors
    ///
    /// Returns a typed search error when verification, syncing, or publication
    /// fails. If the final rename fails after moving the previous generation,
    /// the implementation attempts to restore that generation immediately.
    pub fn promote_verified_installation(
        &self,
        staged_dir: &Path,
        destination_dir: &Path,
    ) -> SearchResult<Option<PathBuf>> {
        self.verify_dir(staged_dir)?;
        sync_registered_artifacts(
            staged_dir,
            self.artifacts
                .iter()
                .map(|artifact| artifact.relative_path.as_str()),
        )?;
        promote_atomically(staged_dir, destination_dir)
    }

    /// Derive the identity declared by this frozen manifest without claiming
    /// that a local artifact directory has already been verified.
    ///
    /// This is intended for lazy embedders that must expose their immutable
    /// expected identity before first use. They still must verify the artifact
    /// directory and compare the loaded identity byte-for-byte before returning
    /// any vector.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the manifest or cross-contract bindings are
    /// incomplete.
    pub fn declared_identity_bundle(
        &self,
        quantization: QuantizationFormat,
        storage_format: &str,
    ) -> SearchResult<EmbeddingIdentityBundleV1> {
        self.identity_bundle(quantization, storage_format)
    }

    pub(super) fn identity_bundle(
        &self,
        quantization: QuantizationFormat,
        storage_format: &str,
    ) -> SearchResult<EmbeddingIdentityBundleV1> {
        let space_contract_fingerprint = self.space_contract_fingerprint()?;
        let provenance_manifest_fingerprint = self.freeze()?.fingerprint;
        let input = self.execution.input_contract.clone();
        let tokenizer_fingerprint = role_fingerprint(self, ModelArtifactRoleV1::Tokenizer)?;
        let vocabulary_fingerprint = role_fingerprint_optional(
            self,
            ModelArtifactRoleV1::Vocabulary,
            &tokenizer_fingerprint,
        );
        let model_config_fingerprint = role_fingerprint_optional(
            self,
            ModelArtifactRoleV1::ModelConfig,
            &space_contract_fingerprint,
        );
        let space = EmbeddingSpaceIdentityV1 {
            schema_version: EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
            logical_model_id: self.logical_model_id.clone(),
            immutable_revision: self.upstream_revision.clone(),
            kind: EmbeddingSpaceKindV1::Semantic,
            artifact_manifest_fingerprint: space_contract_fingerprint,
            artifacts: self
                .artifacts
                .iter()
                .map(|artifact| EmbeddingArtifactIdentityV1 {
                    role: artifact.role.tag().to_owned(),
                    sha256: artifact.sha256.clone(),
                    size: artifact.size,
                })
                .collect(),
            tokenizer_fingerprint,
            vocabulary_fingerprint,
            model_config_fingerprint,
            model_preprocessing: self.execution.model_preprocessing.clone(),
            sequence_policy: self.execution.sequence_policy.clone(),
            query_instruction: self.execution.query_instruction.clone(),
            document_instruction: self.execution.document_instruction.clone(),
            pooling: self.execution.pooling.clone(),
            output_normalization: self.execution.output_normalization.clone(),
            dimension: self.dimension,
            input_contract_fingerprint: input.fingerprint(),
            hash_control: None,
            projection: None,
        };
        let producer = EmbeddingProducerAttestationV1 {
            schema_version: EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
            backend: self.execution.backend.clone(),
            implementation_revision: self.execution.implementation_revision.clone(),
            protocol_revision: self.execution.protocol_revision.clone(),
            numeric_profile: self.execution.numeric_profile.clone(),
            provenance_manifest_fingerprint,
            space_fingerprint: space.fingerprint(),
            golden_vectors: self.execution.golden_vectors.clone(),
        };
        let bundle = EmbeddingIdentityBundleV1 {
            space,
            producer,
            input,
            storage: VectorStorageIdentityV1 {
                schema_version: VECTOR_STORAGE_IDENTITY_SCHEMA_V1,
                format: storage_format.to_owned(),
                quantization,
                endianness: storage_endianness(storage_format).to_owned(),
                vector_normalization: self.execution.output_normalization.clone(),
                dimension: self.dimension,
            },
        };
        bundle.validate()?;
        Ok(bundle)
    }
}

impl FrozenModelArtifactManifestV1 {
    /// Recompute canonical bytes and fingerprint, rejecting any disagreement.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if structured fields, bytes, or digest disagree.
    pub fn validate(&self) -> SearchResult<()> {
        self.manifest.validate()?;
        validate_frozen_sha256("fingerprint", &self.fingerprint)?;
        let canonical = self.manifest.canonical_bytes();
        if canonical != self.canonical_bytes {
            return Err(invalid_manifest_field(
                "canonical_bytes",
                &self.manifest.logical_model_id,
                "stored canonical bytes disagree with structured manifest",
            ));
        }
        let fingerprint = sha256_hex_bytes(&canonical);
        if fingerprint != self.fingerprint {
            return Err(invalid_manifest_field(
                "fingerprint",
                &self.fingerprint,
                "stored fingerprint disagrees with canonical bytes",
            ));
        }
        Ok(())
    }
}

pub(super) fn default_plain_text_input_contract() -> EmbeddingInputContractV1 {
    EmbeddingInputContractV1 {
        schema_version: EMBEDDING_INPUT_CONTRACT_SCHEMA_V1,
        canonicalization: "caller-utf8-as-is-v1".to_owned(),
        content_selection: "single-caller-supplied-text-v1".to_owned(),
        chunking: "none-at-embedder-boundary-v1".to_owned(),
        query_instruction: String::new(),
        document_instruction: String::new(),
        doc_id_semantics: "vector-independent-of-document-id-v1".to_owned(),
    }
}

fn storage_endianness(storage_format: &str) -> &'static str {
    if storage_format.starts_with("in-memory-") {
        "native-f32-values"
    } else {
        "little-endian"
    }
}

// GOLDEN-CHANGE bd-2ba5: the macOS ARM64 ORT 1.28.0 build (da9b5e3)
// produces different f32 bits from the Linux build of that same revision.
// Each certificate was measured in fresh processes using the production
// adapter's ordered first batch. The qualified platform has its own identity;
// load_from_manifest still executes and checks the exact certificate before
// returning an embedder. Other platforms retain the existing contract.
// GOLDEN-CHANGE bd-dsbym/bd-2ba5: Windows x86-64's build of the same ORT
// revision also has distinct, reproducible output bits. Fresh-process probes
// and the 355-text paired semantic review are recorded in UPGRADE_LOG.md.
// Register a separate producer; never admit its bits under a Linux identity.
fn qualify_fastembed_platform(
    mut execution: ModelExecutionContractV1,
    macos_arm64_vectors_sha256: &str,
    windows_x64_vectors_sha256: &str,
) -> ModelExecutionContractV1 {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "ort-1.28.0-da9b5e3-macos-aarch64-cpu-f32-host-default-intra-threads-v1"
            .clone_into(&mut execution.numeric_profile);
        macos_arm64_vectors_sha256.clone_into(&mut execution.golden_vectors.vectors_sha256);
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "ort-1.28.0-da9b5e3-windows-x86_64-cpu-f32-host-default-intra-threads-v1"
            .clone_into(&mut execution.numeric_profile);
        windows_x64_vectors_sha256.clone_into(&mut execution.golden_vectors.vectors_sha256);
    }
    execution
}

fn fastembed_execution_contract(
    dimension: u32,
    model_id: &str,
    vectors_sha256: &str,
) -> SearchResult<ModelExecutionContractV1> {
    Ok(ModelExecutionContractV1 {
        backend: "fastembed-onnx".to_owned(),
        implementation_revision: format!(
            "frankensearch-embed-{}+fastembed-7.1.0:{model_id}",
            env!("CARGO_PKG_VERSION")
        ),
        protocol_revision: "fastembed-7.1.0+ort-2.0.0-rc.13-user-defined-onnx-v1".to_owned(),
        numeric_profile: "ort-2.0.0-rc.13-cpu-f32-host-default-intra-threads-v1".to_owned(),
        weights_format: "onnx-opset-pinned-v1".to_owned(),
        tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
        model_preprocessing: "model-tokenizer-special-tokens=true;empty-input=zero-vector"
            .to_owned(),
        sequence_policy: FASTEMBED_SEQUENCE_POLICY_512.to_owned(),
        pooling: "attention-mask-mean-pool-v1".to_owned(),
        output_normalization: FASTEMBED_OUTPUT_NORMALIZATION_V1.to_owned(),
        query_instruction: String::new(),
        document_instruction: String::new(),
        input_contract: default_plain_text_input_contract(),
        golden_vectors: GoldenVectorCertificateV1 {
            corpus_sha256: conformance_corpus_fingerprint()?,
            vectors_sha256: vectors_sha256.to_owned(),
            vector_count: 4,
            dimension,
        },
    })
}

pub(super) fn conformance_corpus_fingerprint() -> SearchResult<String> {
    GoldenVectorCertificateV1::corpus_fingerprint(&MODEL_CONFORMANCE_TEXTS_V1)
}

fn artifact_role_for_path(path: &str) -> SearchResult<ModelArtifactRoleV1> {
    let file_name = path.rsplit('/').next().unwrap_or(path);
    match file_name {
        "model.safetensors" | "model_f32.safetensors" | "model.onnx" => {
            Ok(ModelArtifactRoleV1::Weights)
        }
        "tokenizer.json" => Ok(ModelArtifactRoleV1::Tokenizer),
        "vocab.json" | "vocab.txt" => Ok(ModelArtifactRoleV1::Vocabulary),
        "config.json" => Ok(ModelArtifactRoleV1::ModelConfig),
        "special_tokens_map.json" => Ok(ModelArtifactRoleV1::SpecialTokens),
        "tokenizer_config.json" => Ok(ModelArtifactRoleV1::TokenizerConfig),
        "projection.safetensors" | "projection.bin" => Ok(ModelArtifactRoleV1::Projection),
        _ => Err(invalid_manifest_field(
            "artifacts[].role",
            path,
            "artifact path has no registered semantic role",
        )),
    }
}

fn validate_frozen_sha256(field: &str, value: &str) -> SearchResult<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    Err(invalid_manifest_field(
        field,
        "redacted-invalid-sha256",
        "must be lowercase 64-character SHA-256",
    ))
}

pub(super) fn validate_frozen_manifest_text(
    field: &str,
    value: &str,
    allow_empty: bool,
) -> SearchResult<()> {
    if value.len() > MAX_FROZEN_MANIFEST_FIELD_BYTES {
        return Err(invalid_manifest_field(
            field,
            "redacted-oversized",
            "field exceeds the bounded frozen-manifest size",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(invalid_manifest_field(
            field,
            "redacted-control-character",
            "field must not contain control characters",
        ));
    }
    if !allow_empty && value.trim().is_empty() {
        return Err(invalid_manifest_field(field, value, "must not be empty"));
    }
    Ok(())
}

fn append_frozen_len(bytes: &mut Vec<u8>, value: usize) {
    let value = u64::try_from(value).unwrap_or(u64::MAX);
    bytes.extend_from_slice(&value.to_be_bytes());
}

pub(super) fn append_frozen_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    append_frozen_len(bytes, value.len());
    bytes.extend_from_slice(value);
}

pub(super) fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    to_hex_lowercase(&digest)
}

pub(super) fn license_metadata_fingerprint(
    license_spdx: &str,
    provider: &str,
    repository: &str,
    revision: &str,
) -> String {
    let mut bytes = Vec::new();
    append_frozen_bytes(&mut bytes, b"frankensearch.model-license-metadata.v1");
    for value in [license_spdx, provider, repository, revision] {
        append_frozen_bytes(&mut bytes, value.as_bytes());
    }
    sha256_hex_bytes(&bytes)
}

fn role_fingerprint(
    manifest: &ModelArtifactManifestV1,
    role: ModelArtifactRoleV1,
) -> SearchResult<String> {
    manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.role == role)
        .map(|artifact| artifact.sha256.clone())
        .ok_or_else(|| {
            invalid_manifest_field("artifacts[].role", role.tag(), "required role is absent")
        })
}

fn role_fingerprint_optional(
    manifest: &ModelArtifactManifestV1,
    role: ModelArtifactRoleV1,
    fallback: &str,
) -> String {
    manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.role == role)
        .map_or_else(|| fallback.to_owned(), |artifact| artifact.sha256.clone())
}

fn verify_no_extra_artifacts(
    model_dir: &Path,
    expected: &[ModelArtifactFileV1],
) -> SearchResult<()> {
    let expected_paths = expected
        .iter()
        .map(|artifact| artifact.relative_path.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut pending = vec![model_dir.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory).map_err(SearchError::Io)?;
        for entry in entries {
            let entry = entry.map_err(SearchError::Io)?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(SearchError::Io)?;
            let relative = path.strip_prefix(model_dir).map_err(|_| {
                invalid_manifest_field(
                    "model_dir",
                    "redacted",
                    "artifact escaped the verified model directory",
                )
            })?;
            let relative = relative.to_string_lossy().replace('\\', "/");
            if relative == ".verified" || expected_paths.contains(relative.as_str()) {
                continue;
            }
            if let Ok(role) = artifact_role_for_path(&relative) {
                return Err(invalid_manifest_field(
                    "artifacts[].role",
                    role.tag(),
                    "unregistered file collides with an expected semantic role",
                ));
            }
            if file_type.is_dir() {
                pending.push(path);
            }
        }
    }
    Ok(())
}
