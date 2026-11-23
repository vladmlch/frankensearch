//! Built-in model catalogs, JSON parsing, validation, and in-process registry accessors.

use std::path::{Path, PathBuf};

use frankensearch_core::error::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use super::frozen::{append_frozen_bytes, sha256_hex_bytes, validate_frozen_manifest_text};
use super::lifecycle::ModelState;
use super::verification::{
    invalid_manifest_field, is_valid_sha256_hex, manifest_registry, manifest_registry_lock_error,
    promote_atomically, resolve_model_file_path, sync_registered_artifacts,
    validate_model_file_name, verify_dir_and_record, verify_file_sha256,
};
use super::{
    MANIFEST_SCHEMA_VERSION, ModelTier, PLACEHOLDER_PINNED_REVISION,
    PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
};

/// One file required by a model manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    /// Relative path inside the model directory.
    pub name: String,
    /// Expected lowercase SHA256 hex digest.
    pub sha256: String,
    /// Expected size in bytes.
    pub size: u64,
    /// Explicit download URL. When `None`, the URL is derived from the parent
    /// manifest's `repo` + `revision` using the `HuggingFace` `/resolve/` path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl ModelFile {
    /// Returns true when the file still uses the placeholder checksum.
    #[must_use]
    pub fn uses_placeholder_checksum(&self) -> bool {
        self.sha256 == PLACEHOLDER_VERIFY_AFTER_DOWNLOAD
    }

    /// Returns true when checksum is usable for production verification.
    #[must_use]
    pub fn has_verified_checksum(&self) -> bool {
        is_valid_sha256_hex(&self.sha256) && !self.uses_placeholder_checksum()
    }

    /// Get the local filename (basename) for saving.
    ///
    /// For paths like `"onnx/model.onnx"`, returns `"model.onnx"`.
    /// This handles `HuggingFace` repos that restructure files into subdirectories.
    #[must_use]
    pub fn local_name(&self) -> &str {
        self.name.rsplit('/').next().unwrap_or(&self.name)
    }

    /// Return the download URL for this file, preferring the explicit `url`
    /// field and falling back to the standard `HuggingFace` `/resolve/` path.
    #[must_use]
    pub fn download_url(&self, repo: &str, revision: &str) -> String {
        self.url.as_ref().map_or_else(
            || {
                format!(
                    "https://huggingface.co/{repo}/resolve/{revision}/{}",
                    self.name
                )
            },
            Clone::clone,
        )
    }
}

/// Manifest for one downloadable model bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Stable model identifier.
    pub id: String,
    /// Human-readable version tag for manifest-managed model assets.
    #[serde(default)]
    pub version: String,
    /// Human-readable display name (e.g., "Potion Base 128M (fast tier)").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Optional longer description for CLI/help output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// `HuggingFace` repository slug.
    pub repo: String,
    /// Pinned revision (commit SHA).
    pub revision: String,
    /// Required files for this model.
    pub files: Vec<ModelFile>,
    /// SPDX-style license identifier.
    pub license: String,
    /// Output embedding dimension (e.g., 256 for potion, 384 for `MiniLM`).
    /// `None` for models that don't produce fixed-dim embeddings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    /// Which search tier this model serves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<ModelTier>,
    /// Optional precomputed aggregate download size in bytes.
    #[serde(
        default,
        rename = "total_size_bytes",
        skip_serializing_if = "is_zero_u64"
    )]
    pub download_size_bytes: u64,
}

/// Canonical identity used to bind a verification receipt to one production manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FrozenVerificationManifest {
    pub(super) fingerprint: String,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde requires &T signature
const fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

impl ModelManifest {
    /// Built-in manifest for MiniLM-L6-v2 (quality tier).
    #[must_use]
    pub fn minilm_v2() -> Self {
        const REVISION: &str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf";
        const REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
        Self {
            id: "all-minilm-l6-v2".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("All MiniLM L6 v2 (quality tier)".to_owned()),
            description: Some(
                "MiniLM-L6-v2 ONNX sentence embedding model for quality-tier semantic search"
                    .to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
                        .to_owned(),
                    size: 90_405_214,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
                        .to_owned(),
                    size: 466_247,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"
                        .to_owned(),
                    size: 612,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"
                        .to_owned(),
                    size: 112,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/special_tokens_map.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b"
                        .to_owned(),
                    size: 350,
                    url: Some(
                        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json"
                            .to_owned(),
                    ),
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(384),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 90_872_535,
        }
    }

    /// The native artifacts retain the original model id when constructing
    /// producer identities. Only the installation catalog uses a separate id.
    pub(super) fn minilm_v2_native_weights() -> SearchResult<Self> {
        let mut manifest = Self::minilm_v2();
        let weights = manifest
            .files
            .iter_mut()
            .find(|file| file.name == "onnx/model.onnx")
            .ok_or_else(|| {
                invalid_manifest_field(
                    "artifacts[].role",
                    "weights",
                    "MiniLM download manifest is missing its registered weights artifact",
                )
            })?;
        "model.safetensors".clone_into(&mut weights.name);
        weights.url = Some(
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/model.safetensors"
                .to_owned(),
        );
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
            .clone_into(&mut weights.sha256);
        weights.size = 90_868_376;
        manifest.download_size_bytes = manifest.files.iter().map(|file| file.size).sum();
        Ok(manifest)
    }

    /// Opt-in native `MiniLM` installation, separate from the ONNX directory.
    /// The files also back the existing frozen int8 and F32 producer contracts.
    #[must_use]
    pub fn minilm_v2_native() -> Self {
        let mut manifest = Self::minilm_v2_native_weights()
            .expect("built-in MiniLM manifest contains its registered weights");
        "all-minilm-l6-v2-native".clone_into(&mut manifest.id);
        manifest.display_name = Some("All MiniLM L6 v2 native (quality tier)".to_owned());
        manifest.description = Some(
            "MiniLM-L6-v2 safetensors for explicit pure-Rust quality-tier inference".to_owned(),
        );
        manifest
    }

    /// Opt-in manifest for multilingual `MiniLM` L12 sentence embeddings.
    ///
    /// This model is deliberately absent from [`Self::builtin_catalog`]: its
    /// 384-dimensional output is a distinct vector space from `all-MiniLM-L6-v2`,
    /// and its larger artifact must never be acquired or selected implicitly.
    #[must_use]
    pub fn multilingual_minilm_l12_v2() -> Self {
        const REVISION: &str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42";
        const REPO: &str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
        Self {
            id: "paraphrase-multilingual-minilm-l12-v2".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Paraphrase Multilingual MiniLM L12 v2 (opt-in)".to_owned()),
            description: Some(
                "Opt-in 384-dimensional multilingual sentence embedder for CJK and mixed-language retrieval"
                    .to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b"
                        .to_owned(),
                    size: 470_641_600,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"
                        .to_owned(),
                    size: 9_081_518,
                    url: None,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e"
                        .to_owned(),
                    size: 645,
                    url: None,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d"
                        .to_owned(),
                    size: 239,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b"
                        .to_owned(),
                    size: 526,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(384),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 479_724_528,
        }
    }

    /// Built-in manifest for potion-128M style `Model2Vec` assets (fast tier).
    #[must_use]
    pub fn potion_128m() -> Self {
        const REVISION: &str = "a28f4eebecd4dc585034f605e52d414878a0417c";
        const REPO: &str = "minishlab/potion-multilingual-128M";
        Self {
            id: "potion-multilingual-128m".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Potion Multilingual 128M (fast tier)".to_owned()),
            description: Some(
                "Model2Vec static embedding model for fast-tier multilingual retrieval".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1"
                        .to_owned(),
                    size: 18_616_131,
                    url: Some(
                        "https://huggingface.co/minishlab/potion-multilingual-128M/resolve/a28f4eebecd4dc585034f605e52d414878a0417c/tokenizer.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: "14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397"
                        .to_owned(),
                    size: 512_361_560,
                    url: Some(
                        "https://huggingface.co/minishlab/potion-multilingual-128M/resolve/a28f4eebecd4dc585034f605e52d414878a0417c/model.safetensors"
                            .to_owned(),
                    ),
                },
            ],
            license: "MIT".to_owned(),
            dimension: Some(256),
            tier: Some(ModelTier::Fast),
            download_size_bytes: 530_977_691,
        }
    }

    /// Opt-in manifest for the English `potion-base-8M` `Model2Vec` fast tier
    /// (GH #50).
    ///
    /// Deliberately absent from [`Self::builtin_catalog`]: it is a distinct
    /// 256-dimensional vector space from `potion-multilingual-128M`, so it is
    /// acquired and selected only explicitly. Its 30 MB matrix and 30k-piece
    /// `WordPiece` tokenizer load far faster than the 128M model's 512 MB
    /// matrix and 500k-piece Unigram tokenizer (GH #46). English-only.
    #[must_use]
    pub fn potion_base_8m() -> Self {
        const REVISION: &str = "bf8b056651a2c21b8d2565580b8569da283cab23";
        const REPO: &str = "minishlab/potion-base-8M";
        Self {
            id: "potion-base-8m".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Potion Base 8M (fast tier, English)".to_owned()),
            description: Some(
                "Compact English Model2Vec static embedding model for the fast tier".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50"
                        .to_owned(),
                    size: 683_666,
                    url: Some(
                        "https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056651a2c21b8d2565580b8569da283cab23/tokenizer.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2"
                        .to_owned(),
                    size: 30_236_760,
                    url: Some(
                        "https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056651a2c21b8d2565580b8569da283cab23/model.safetensors"
                            .to_owned(),
                    ),
                },
            ],
            license: "MIT".to_owned(),
            dimension: Some(256),
            tier: Some(ModelTier::Fast),
            download_size_bytes: 30_920_426,
        }
    }

    /// Opt-in manifest for the English `potion-base-32M` `Model2Vec` fast tier
    /// (GH #50). A distinct 512-dimensional space; selected only explicitly,
    /// like [`Self::potion_base_8m`]. English-only.
    #[must_use]
    pub fn potion_base_32m() -> Self {
        const REVISION: &str = "1e5a03f8eeb2c98b928fbbd846f22f816360919f";
        const REPO: &str = "minishlab/potion-base-32M";
        Self {
            id: "potion-base-32m".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Potion Base 32M (fast tier, English)".to_owned()),
            description: Some(
                "English Model2Vec static embedding model for the fast tier".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "7d75cbc54318138807c401b0f0c9721117c628b39de8e8e0edb6cb17e0ee7d18"
                        .to_owned(),
                    size: 1_493_150,
                    url: Some(
                        "https://huggingface.co/minishlab/potion-base-32M/resolve/1e5a03f8eeb2c98b928fbbd846f22f816360919f/tokenizer.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: "99f6c33204c9231a7391871b7a3c91409b532c8f587a9ea44fc282303d8dec28"
                        .to_owned(),
                    size: 129_210_456,
                    url: Some(
                        "https://huggingface.co/minishlab/potion-base-32M/resolve/1e5a03f8eeb2c98b928fbbd846f22f816360919f/model.safetensors"
                            .to_owned(),
                    ),
                },
            ],
            license: "MIT".to_owned(),
            dimension: Some(512),
            tier: Some(ModelTier::Fast),
            download_size_bytes: 130_703_606,
        }
    }

    /// Built-in manifest for flashrank-nano (cross-encoder reranker).
    #[must_use]
    pub fn flashrank_nano() -> Self {
        const REVISION: &str = PLACEHOLDER_PINNED_REVISION;
        const REPO: &str = "prithivida/flashrank-nano";
        Self {
            id: "flashrank-nano".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("FlashRank Nano (Reranker)".to_owned()),
            description: Some("FlashRank compact ONNX cross-encoder reranker model".to_owned()),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                    size: 0,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: None,
            tier: Some(ModelTier::Reranker),
            download_size_bytes: 0,
        }
    }

    /// Built-in manifest for MS MARCO `MiniLM` reranker (cross-encoder).
    #[must_use]
    pub fn ms_marco_reranker() -> Self {
        const REVISION: &str = "c5ee24cb16019beea0893ab7796b1df96625c6b8";
        const REPO: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";
        Self {
            id: "ms-marco-minilm-l-6-v2".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("MS MARCO MiniLM L-6 v2 (reranker)".to_owned()),
            description: Some(
                "MS MARCO cross-encoder reranker model for final relevance scoring".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                // The f32 safetensors export feeds the pure-Rust frankentorch
                // cross-encoder (`frankensearch-rerank` `native`), which is
                // the reranker fsfs ships; the ONNX export below stays for the
                // optional `fastembed-reranker` backend.
                ModelFile {
                    name: "model.safetensors".to_owned(),
                    sha256: "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
                        .to_owned(),
                    size: 90_870_598,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/model.safetensors"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a"
                        .to_owned(),
                    size: 91_011_230,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/onnx/model.onnx"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
                        .to_owned(),
                    size: 711_396,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/tokenizer.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc"
                        .to_owned(),
                    size: 794,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/config.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6"
                        .to_owned(),
                    size: 132,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/special_tokens_map.json"
                            .to_owned(),
                    ),
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8"
                        .to_owned(),
                    size: 1_330,
                    url: Some(
                        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/c5ee24cb16019beea0893ab7796b1df96625c6b8/tokenizer_config.json"
                            .to_owned(),
                    ),
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: None, // Cross-encoder produces scores, not embeddings
            tier: Some(ModelTier::Reranker),
            download_size_bytes: 182_595_480,
        }
    }

    // ==================== Bake-off Eligible Models ====================

    /// Snowflake Arctic Embed S manifest.
    ///
    /// Dimension: 384. Small, fast model with the same width as `MiniLM` but a
    /// distinct mathematical vector space.
    /// Verified checksums from `HuggingFace`.
    #[must_use]
    pub fn snowflake_arctic_s() -> Self {
        const REVISION: &str = "e596f507467533e48a2e17c007f0e1dacc837b33";
        const REPO: &str = "Snowflake/snowflake-arctic-embed-s";
        Self {
            id: "snowflake-arctic-embed-s".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Snowflake Arctic Embed S".to_owned()),
            description: Some(
                "Small, fast embedding model with MiniLM-compatible 384 dimensions".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "579c1f1778a0993eb0d2a1403340ffb491c769247fb46acc4f5cf8ac5b89c1e1"
                        .to_owned(),
                    size: 133_093_492,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854"
                        .to_owned(),
                    size: 711_649,
                    url: None,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "4e519aa92ec40943356032afe458c8829d70c5766b109e4a57490b82f72dcfb7"
                        .to_owned(),
                    size: 703,
                    url: None,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"
                        .to_owned(),
                    size: 695,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "9ca59277519f6e3692c8685e26b94d4afca2d5438deff66483db495e48735810"
                        .to_owned(),
                    size: 1_433,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(384),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 133_807_972,
        }
    }

    /// Nomic Embed Text v1.5 manifest.
    ///
    /// Dimension: 768. Long context support with Matryoshka embedding capability.
    /// Verified checksums from `HuggingFace`.
    #[must_use]
    pub fn nomic_embed() -> Self {
        const REVISION: &str = "e5cf08aadaa33385f5990def41f7a23405aec398";
        const REPO: &str = "nomic-ai/nomic-embed-text-v1.5";
        Self {
            id: "nomic-embed-text-v1.5".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Nomic Embed Text v1.5".to_owned()),
            description: Some(
                "Long context embedding model with Matryoshka capability (768 dims)".to_owned(),
            ),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965"
                        .to_owned(),
                    size: 547_310_275,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
                        .to_owned(),
                    size: 711_396,
                    url: None,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "0168e0883705b0bf8f2b381e10f45a9f3e1ef4b13869b43c160e4c8a70ddf442"
                        .to_owned(),
                    size: 2_331,
                    url: None,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"
                        .to_owned(),
                    size: 695,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "d7e0000bcc80134debd2222220427e6bf5fa20a669f40a0d0d1409cc18e0a9bc"
                        .to_owned(),
                    size: 1_191,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: Some(768),
            tier: Some(ModelTier::Quality),
            download_size_bytes: 548_025_888,
        }
    }

    /// Jina Reranker v1 Turbo EN manifest.
    ///
    /// Fast, optimized for English. Verified checksums from `HuggingFace`.
    #[must_use]
    pub fn jina_reranker_turbo() -> Self {
        const REVISION: &str = "b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2";
        const REPO: &str = "jinaai/jina-reranker-v1-turbo-en";
        Self {
            id: "jina-reranker-v1-turbo-en".to_owned(),
            version: "v1".to_owned(),
            display_name: Some("Jina Reranker v1 Turbo EN".to_owned()),
            description: Some("Fast cross-encoder reranker optimized for English".to_owned()),
            repo: REPO.to_owned(),
            revision: REVISION.to_owned(),
            files: vec![
                ModelFile {
                    name: "onnx/model.onnx".to_owned(),
                    sha256: "c1296c66c119de645fa9cdee536d8637740efe85224cfa270281e50f213aa565"
                        .to_owned(),
                    size: 151_296_975,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer.json".to_owned(),
                    sha256: "0046da43cc8c424b317f56b092b0512aaaa65c4f925d2f16af9d9eeb4d0ef902"
                        .to_owned(),
                    size: 2_030_772,
                    url: None,
                },
                ModelFile {
                    name: "config.json".to_owned(),
                    sha256: "e050ff6a15ae9295e84882fa0e98051bd8754856cd5201395ebf00ce9f2d609b"
                        .to_owned(),
                    size: 1_206,
                    url: None,
                },
                ModelFile {
                    name: "special_tokens_map.json".to_owned(),
                    sha256: "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f"
                        .to_owned(),
                    size: 280,
                    url: None,
                },
                ModelFile {
                    name: "tokenizer_config.json".to_owned(),
                    sha256: "d291c6652d96d56ffdbcf1ea19d9bae5ed79003f7648c627e725a619227ce8fa"
                        .to_owned(),
                    size: 1_215,
                    url: None,
                },
            ],
            license: "Apache-2.0".to_owned(),
            dimension: None, // Cross-encoder produces scores, not embeddings
            tier: Some(ModelTier::Reranker),
            download_size_bytes: 153_330_448,
        }
    }

    // ==================== Lookup & Listing Functions ====================

    /// Get manifest by embedder name.
    #[must_use]
    pub fn for_embedder(name: &str) -> Option<Self> {
        match name {
            "minilm" => Some(Self::minilm_v2()),
            "multilingual-minilm" | "paraphrase-multilingual-minilm-l12-v2" => {
                Some(Self::multilingual_minilm_l12_v2())
            }
            "snowflake-arctic-s" => Some(Self::snowflake_arctic_s()),
            "nomic-embed" => Some(Self::nomic_embed()),
            "potion-128m" => Some(Self::potion_128m()),
            _ => None,
        }
    }

    /// Get manifest by reranker name.
    #[must_use]
    pub fn for_reranker(name: &str) -> Option<Self> {
        match name {
            "ms-marco" => Some(Self::ms_marco_reranker()),
            "jina-reranker-turbo" => Some(Self::jina_reranker_turbo()),
            _ => None,
        }
    }

    /// Get all bake-off eligible embedder manifests.
    #[must_use]
    pub fn bakeoff_embedder_candidates() -> Vec<Self> {
        vec![Self::snowflake_arctic_s(), Self::nomic_embed()]
    }

    /// Get all bake-off eligible reranker manifests.
    #[must_use]
    pub fn bakeoff_reranker_candidates() -> Vec<Self> {
        vec![Self::jina_reranker_turbo()]
    }

    /// Get all bake-off eligible model manifests (embedders + rerankers).
    #[must_use]
    pub fn bakeoff_candidates() -> Vec<Self> {
        let mut candidates = Self::bakeoff_embedder_candidates();
        candidates.extend(Self::bakeoff_reranker_candidates());
        candidates
    }

    /// Return the compiled-in catalog of all built-in model manifests.
    ///
    /// This is the single source of truth for what models frankensearch needs.
    /// The binary always knows what models it requires without network access.
    #[must_use]
    pub fn builtin_catalog() -> ModelManifestCatalog {
        ModelManifestCatalog {
            schema_version: MANIFEST_SCHEMA_VERSION,
            models: vec![
                Self::potion_128m(),
                Self::minilm_v2(),
                Self::ms_marco_reranker(),
                Self::snowflake_arctic_s(),
                Self::nomic_embed(),
                Self::jina_reranker_turbo(),
                Self::flashrank_nano(),
            ],
        }
    }

    /// Models that are discoverable and downloadable only after an explicit
    /// model selection. They are excluded from default bulk acquisition.
    #[must_use]
    pub fn opt_in_catalog() -> ModelManifestCatalog {
        ModelManifestCatalog {
            schema_version: MANIFEST_SCHEMA_VERSION,
            models: vec![
                Self::multilingual_minilm_l12_v2(),
                Self::minilm_v2_native(),
                Self::potion_base_8m(),
                Self::potion_base_32m(),
            ],
        }
    }

    /// Parse a manifest from JSON and validate basic structure.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if JSON parsing or validation fails.
    pub fn from_json_str(raw: &str) -> SearchResult<Self> {
        let manifest =
            serde_json::from_str::<Self>(raw).map_err(|_source| SearchError::InvalidConfig {
                field: "manifest_json".to_owned(),
                value: "redacted-manifest-json".to_owned(),
                reason: "failed to parse manifest JSON; input was malformed or unsupported"
                    .to_owned(),
            })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serialize this manifest to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if serialization fails.
    pub fn to_pretty_json(&self) -> SearchResult<String> {
        serde_json::to_string_pretty(self).map_err(|source| SearchError::InvalidConfig {
            field: "manifest_json".to_owned(),
            value: self.id.clone(),
            reason: format!("failed to serialize manifest: {source}"),
        })
    }

    /// Returns true when all files have non-placeholder concrete checksums.
    #[must_use]
    pub fn has_verified_checksums(&self) -> bool {
        !self.files.is_empty() && self.files.iter().all(ModelFile::has_verified_checksum)
    }

    /// Returns true only for a complete lowercase commit digest.
    #[must_use]
    pub fn has_pinned_revision(&self) -> bool {
        let revision = self.revision.trim();
        matches!(revision.len(), 40 | 64)
            && revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    /// Returns true when this manifest is ready for production-grade verification.
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.has_verified_checksums() && self.has_pinned_revision()
    }

    /// Freeze the complete validated download manifest into one stable identity digest.
    ///
    /// The canonical payload includes every serialized manifest field, including the
    /// pinned revision and every file name, size, URL, and SHA-256. Any manifest evolution
    /// therefore invalidates an earlier verification receipt even when the logical ID is
    /// unchanged.
    pub(super) fn freeze_verification_manifest(&self) -> SearchResult<FrozenVerificationManifest> {
        self.validate()?;
        if !self.is_production_ready() {
            return Err(invalid_manifest_field(
                "production_ready",
                &self.id,
                "cached verification requires a pinned revision and SHA-256 for every file",
            ));
        }
        let serialized = serde_json::to_vec(self).map_err(|_| {
            invalid_manifest_field(
                "manifest_fingerprint",
                &self.id,
                "failed to serialize the validated manifest for canonical verification",
            )
        })?;
        let mut canonical = Vec::with_capacity(serialized.len().saturating_add(64));
        append_frozen_bytes(
            &mut canonical,
            b"frankensearch.model-download-verification-manifest.v1",
        );
        canonical.extend_from_slice(&MANIFEST_SCHEMA_VERSION.to_be_bytes());
        append_frozen_bytes(&mut canonical, &serialized);
        Ok(FrozenVerificationManifest {
            fingerprint: sha256_hex_bytes(&canonical),
        })
    }

    /// Sum of expected bytes for all files.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        if self.download_size_bytes > 0 {
            return self.download_size_bytes;
        }
        self.files.iter().map(|file| file.size).sum()
    }

    /// Alias for [`total_size_bytes`](Self::total_size_bytes).
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.total_size_bytes()
    }

    /// `HuggingFace` download URL for a specific file in this manifest.
    #[must_use]
    pub fn download_url(&self, file: &ModelFile) -> String {
        file.download_url(&self.repo, &self.revision)
    }

    /// Validate manifest fields for shape and checksum format.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for malformed fields.
    pub fn validate(&self) -> SearchResult<()> {
        for (field, value) in [
            ("id", self.id.as_str()),
            ("repo", self.repo.as_str()),
            ("revision", self.revision.as_str()),
            ("license", self.license.as_str()),
        ] {
            validate_frozen_manifest_text(field, value, false)?;
        }
        for (field, value) in [
            ("repo", self.repo.as_str()),
            ("revision", self.revision.as_str()),
        ] {
            if value.bytes().any(|byte| byte.is_ascii_whitespace())
                || value.contains('@')
                || value.contains('?')
                || value.contains('#')
            {
                return Err(invalid_manifest_field(
                    field,
                    "redacted",
                    "download coordinates must not contain whitespace, userinfo, query, or fragment",
                ));
            }
        }
        validate_frozen_manifest_text("version", &self.version, true)?;
        for (field, value) in [
            ("display_name", self.display_name.as_deref()),
            ("description", self.description.as_deref()),
        ] {
            if let Some(value) = value {
                validate_frozen_manifest_text(field, value, true)?;
            }
        }

        for file in &self.files {
            validate_model_file_name(&file.name)?;
            if file.uses_placeholder_checksum() {
            } else if !is_valid_sha256_hex(&file.sha256) {
                return Err(invalid_manifest_field(
                    "files[].sha256",
                    "redacted-invalid-sha256",
                    "must be lowercase 64-char SHA256 hex or placeholder",
                ));
            }
            if let Some(url) = &file.url {
                validate_frozen_manifest_text("files[].url", url, false)?;
                if !url.starts_with("https://")
                    || url.bytes().any(|byte| byte.is_ascii_whitespace())
                    || url.contains('@')
                    || url.contains('?')
                    || url.contains('#')
                {
                    return Err(invalid_manifest_field(
                        "files[].url",
                        "redacted",
                        "must be credential-free HTTPS without userinfo, query, or fragment",
                    ));
                }
            }
        }

        if self.download_size_bytes > 0 {
            let computed_size: u64 = self.files.iter().map(|file| file.size).sum();
            if computed_size != self.download_size_bytes {
                return Err(invalid_manifest_field(
                    "total_size_bytes",
                    &self.download_size_bytes.to_string(),
                    "must match the sum of files[].size",
                ));
            }
        }

        Ok(())
    }

    /// Enforce checksum policy; placeholder checksums are rejected in release mode.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if a release policy violation is detected.
    pub fn validate_checksum_policy(&self) -> SearchResult<()> {
        self.validate_checksum_policy_for(cfg!(not(debug_assertions)))
    }

    /// Enforce checksum policy with explicit release-mode toggle (useful for tests).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if release-mode requires concrete checksums.
    pub fn validate_checksum_policy_for(&self, release_mode: bool) -> SearchResult<()> {
        if release_mode && self.files.iter().any(ModelFile::uses_placeholder_checksum) {
            return Err(invalid_manifest_field(
                "files[].sha256",
                PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
                "placeholder checksums are forbidden in release mode",
            ));
        }
        Ok(())
    }

    /// Verify all manifest files in `model_dir` using streaming SHA256 checks.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when any file is missing or hash/size verification fails.
    pub fn verify_dir(&self, model_dir: &Path) -> SearchResult<()> {
        for file in &self.files {
            let path = resolve_model_file_path(model_dir, &file.name)?;
            verify_file_sha256(&path, &file.sha256, file.size)?;
        }
        Ok(())
    }

    /// Promote a staged model directory to final destination atomically after verification.
    ///
    /// Returns the backup path when an existing install was moved out of the way.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if verification or filesystem rename operations fail.
    pub fn promote_verified_installation(
        &self,
        staged_dir: &Path,
        destination_dir: &Path,
    ) -> SearchResult<Option<PathBuf>> {
        verify_dir_and_record(self, staged_dir)?;
        sync_registered_artifacts(staged_dir, self.files.iter().map(|file| file.name.as_str()))?;
        promote_atomically(staged_dir, destination_dir)
    }

    /// Return `UpdateAvailable` when installed revision differs from pinned revision.
    #[must_use]
    pub fn detect_update_state(&self, installed_revision: &str) -> Option<ModelState> {
        if !self.has_pinned_revision() {
            return None;
        }
        let current = installed_revision.trim();
        if current == self.revision {
            return None;
        }
        Some(ModelState::UpdateAvailable {
            current_revision: if current.is_empty() {
                "unknown".to_owned()
            } else {
                current.to_owned()
            },
            latest_revision: self.revision.clone(),
        })
    }

    /// Register this manifest in the in-process registry.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if validation fails or registry lock is poisoned.
    pub fn register(self) -> SearchResult<()> {
        self.validate()?;
        manifest_registry()
            .write()
            .map_err(|_| manifest_registry_lock_error("write"))?
            .insert(self.id.clone(), self);
        Ok(())
    }

    /// Look up a registered manifest by id.
    #[must_use]
    pub fn lookup(id: &str) -> Option<Self> {
        let guard = manifest_registry().read().unwrap_or_else(|poisoned| {
            tracing::warn!(
                "model manifest registry lock poisoned on read during lookup; using recovered state"
            );
            poisoned.into_inner()
        });
        guard.get(id).cloned()
    }

    /// Return all registered manifests in deterministic id order.
    #[must_use]
    pub fn registered() -> Vec<Self> {
        let guard = manifest_registry().read().unwrap_or_else(|poisoned| {
            tracing::warn!(
                "model manifest registry lock poisoned on read during listing; using recovered state"
            );
            poisoned.into_inner()
        });
        guard.values().cloned().collect()
    }
}

/// Model manifest catalog for bulk load/validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifestCatalog {
    /// Schema version for forward compatibility.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Manifests contained in this catalog.
    #[serde(default)]
    pub models: Vec<ModelManifest>,
}

const fn default_schema_version() -> u32 {
    MANIFEST_SCHEMA_VERSION
}

impl Default for ModelManifestCatalog {
    fn default() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            models: Vec::new(),
        }
    }
}

impl ModelManifestCatalog {
    /// Parse a catalog from JSON.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if parsing fails.
    pub fn from_json_str(raw: &str) -> SearchResult<Self> {
        serde_json::from_str::<Self>(raw).map_err(|_source| SearchError::InvalidConfig {
            field: "manifest_catalog_json".to_owned(),
            value: "redacted-manifest-catalog-json".to_owned(),
            reason: "failed to parse manifest catalog JSON; input was malformed or unsupported"
                .to_owned(),
        })
    }

    /// Validate every manifest in the catalog.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if any contained manifest is invalid.
    pub fn validate(&self) -> SearchResult<()> {
        for model in &self.models {
            model.validate()?;
        }
        Ok(())
    }
}
