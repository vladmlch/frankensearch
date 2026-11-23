//! Local backend discovery, remote-intent resolution, and candidate path helpers.

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use std::collections::BTreeSet;
#[cfg(not(any(feature = "model2vec", feature = "fastembed")))]
use std::path::Path;
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use tracing::debug;
#[cfg(any(feature = "model2vec", feature = "fastembed", feature = "api"))]
use tracing::info;
use tracing::warn;

use frankensearch_core::error::{SearchError, SearchResult};
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
#[cfg(feature = "api")]
use frankensearch_core::generation::FrozenEmbeddingIdentityBundleV1;
use frankensearch_core::traits::Embedder;

#[cfg(feature = "fastembed")]
use crate::fastembed_embedder::{
    DEFAULT_HF_ID as MINILM_HF_ID, DEFAULT_MODEL_NAME as MINILM_MODEL_NAME, FastEmbedEmbedder,
    find_model_dir_with_hf_id as find_fastembed_model_dir,
};
#[cfg(feature = "hash")]
use crate::hash_embedder::HashEmbedder;
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use crate::model_manifest::ModelManifest;
#[cfg(feature = "model2vec")]
use crate::model2vec_embedder::{
    Model2VecEmbedder, find_model_dir_with_hf_id as find_model2vec_model_dir,
};

#[cfg(feature = "model2vec")]
pub(super) const POTION_MODEL_NAME: &str = "potion-multilingual-128M";
#[cfg(feature = "model2vec")]
const POTION_HF_ID: &str = "minishlab/potion-multilingual-128M";
pub(super) const OFFLINE_ENV: &str = "FRANKENSEARCH_OFFLINE";

pub(super) fn parse_bool_flag(raw: &str) -> Option<bool> {
    let value = raw.trim();
    if value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
    {
        return Some(true);
    }
    if value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("no")
        || value.eq_ignore_ascii_case("off")
    {
        return Some(false);
    }
    None
}

#[cfg(feature = "model2vec")]
pub(super) fn detect_fast_embedder(model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    let manifest = ModelManifest::potion_128m();
    // When an explicit model_root is provided, skip system-wide discovery
    // to ensure test isolation and deterministic behavior.
    let discovered = if model_root.is_some() {
        None
    } else {
        find_model2vec_model_dir(POTION_MODEL_NAME, POTION_HF_ID)
    };
    let candidates = candidate_directories(model_root, POTION_MODEL_NAME, discovered.as_deref());
    let checked_candidates = candidates.len();

    for candidate in candidates {
        let missing = missing_manifest_files(&manifest, &candidate);
        if !missing.is_empty() {
            if candidate.is_dir() {
                debug!(
                    model = POTION_MODEL_NAME,
                    tier = "fast",
                    missing = ?missing,
                    "model directory exists but is incomplete, skipping candidate"
                );
            }
            continue;
        }
        if let Err(_error) = crate::model_manifest::verify_dir_cached(&manifest, &candidate) {
            warn!(
                model = POTION_MODEL_NAME,
                reason = "manifest_verification_failed",
                "model2vec manifest verification failed, skipping candidate"
            );
            continue;
        }

        // Shared load: a process that builds more than one embedder stack —
        // a daemon, or a runtime reconstructed per request — reuses the
        // resident tokenizer and matrix instead of paying seconds and half a
        // gigabyte again (GH #46).
        match Model2VecEmbedder::load_shared_with_name(&candidate, POTION_MODEL_NAME) {
            Ok(embedder) => {
                info!(
                    model = POTION_MODEL_NAME,
                    tier = "fast",
                    dimension = embedder.dimension(),
                    identity = embedder.identity().map_or_else(
                        |_| "unverifiable".to_owned(),
                        EmbeddingIdentityBundleV1::fingerprint
                    ),
                    "embedder detected"
                );
                return Some(embedder as Arc<dyn Embedder>);
            }
            Err(_error) => {
                warn!(
                    model = POTION_MODEL_NAME,
                    tier = "fast",
                    reason = "backend_load_failed",
                    "embedder unavailable"
                );
            }
        }
    }

    info!(
        model = POTION_MODEL_NAME,
        tier = "fast",
        checked_candidates,
        "embedder unavailable"
    );
    None
}

#[cfg(not(feature = "model2vec"))]
pub(super) fn detect_fast_embedder(_model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(feature = "fastembed")]
pub(super) fn detect_quality_embedder(model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    let manifest = ModelManifest::minilm_v2();
    // When an explicit model_root is provided, skip system-wide discovery
    // to ensure test isolation and deterministic behavior.
    let discovered = if model_root.is_some() {
        None
    } else {
        find_fastembed_model_dir(MINILM_MODEL_NAME, MINILM_HF_ID)
    };
    let candidates = candidate_directories(model_root, MINILM_MODEL_NAME, discovered.as_deref());
    let checked_candidates = candidates.len();

    for candidate in candidates {
        let missing = missing_manifest_files(&manifest, &candidate);
        if !missing.is_empty() {
            if candidate.is_dir() {
                debug!(
                    model = MINILM_MODEL_NAME,
                    tier = "quality",
                    missing = ?missing,
                    "model directory exists but is incomplete, skipping candidate"
                );
            }
            continue;
        }
        if let Err(_error) = crate::model_manifest::verify_dir_cached(&manifest, &candidate) {
            warn!(
                model = MINILM_MODEL_NAME,
                reason = "manifest_verification_failed",
                "quality manifest verification failed, skipping candidate"
            );
            continue;
        }

        match FastEmbedEmbedder::load_with_name(&candidate, MINILM_MODEL_NAME) {
            Ok(embedder) => {
                info!(
                    model = MINILM_MODEL_NAME,
                    tier = "quality",
                    dimension = embedder.dimension(),
                    identity = embedder.identity().map_or_else(
                        |_| "unverifiable".to_owned(),
                        EmbeddingIdentityBundleV1::fingerprint
                    ),
                    "embedder detected"
                );
                return Some(Arc::new(embedder));
            }
            Err(_error) => {
                warn!(
                    model = MINILM_MODEL_NAME,
                    tier = "quality",
                    reason = "backend_load_failed",
                    "embedder unavailable"
                );
            }
        }
    }

    info!(
        model = MINILM_MODEL_NAME,
        tier = "quality",
        checked_candidates,
        "embedder unavailable"
    );
    None
}

#[cfg(not(feature = "fastembed"))]
pub(super) fn detect_quality_embedder(_model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(feature = "hash")]
#[allow(clippy::unnecessary_wraps)]
pub(super) fn hash_fallback_embedder() -> Option<Arc<dyn Embedder>> {
    Some(Arc::new(HashEmbedder::default_256()))
}

#[cfg(not(feature = "hash"))]
pub(super) fn hash_fallback_embedder() -> Option<Arc<dyn Embedder>> {
    None
}

/// Detect API-based embedder from environment variables.
///
/// Checks for `OPENAI_API_KEY` or `GEMINI_API_KEY`, with optional
/// `FRANKENSEARCH_API_PROVIDER`, `FRANKENSEARCH_API_MODEL`, and
/// `FRANKENSEARCH_API_DIMENSION` overrides. Remote use additionally requires a
/// frozen `FrozenEmbeddingIdentityBundleV1` (structured identity, canonical
/// bytes, and digest) in `FRANKENSEARCH_API_IDENTITY_JSON`; a mutable provider
/// model name is never treated as compatibility evidence.
///
/// Returns a cached `ApiEmbedder` for the quality tier.
/// Caller-supplied detection policy (bd-p6z6.2).
///
/// Every `None` defers to the process environment; every `Some` overrides
/// it, so hosts with explicit configuration are immune to ambient
/// environment drift.
#[derive(Debug, Clone, Copy, Default)]
pub struct DetectOptions {
    /// `Some(true)`: never download models or construct a remote provider
    /// during this detection. Explicit remote intent then returns a typed
    /// unavailable error, while ambient provider credentials remain ignored.
    /// `Some(false)`: permit remote selection and downloads (still subject to
    /// consent policy) even if `FRANKENSEARCH_OFFLINE` is set. `None`: follow
    /// the environment.
    pub offline: Option<bool>,
}

/// Snapshot of the remote-embedding environment, taken once so the intent
/// resolver is a pure function (testable without process-global env
/// mutation, which this workspace forbids).
#[derive(Debug, Clone, Default)]
pub(super) struct RemoteIntentEnv {
    pub(super) offline: Option<String>,
    pub(super) provider: Option<String>,
    pub(super) model: Option<String>,
    pub(super) dimension: Option<String>,
    pub(super) identity_json: Option<String>,
    pub(super) openai_key: Option<String>,
    pub(super) gemini_key: Option<String>,
}

impl RemoteIntentEnv {
    pub(super) fn from_environment() -> Self {
        Self {
            offline: std::env::var(OFFLINE_ENV).ok(),
            provider: std::env::var("FRANKENSEARCH_API_PROVIDER").ok(),
            model: std::env::var("FRANKENSEARCH_API_MODEL").ok(),
            dimension: std::env::var("FRANKENSEARCH_API_DIMENSION").ok(),
            identity_json: std::env::var("FRANKENSEARCH_API_IDENTITY_JSON").ok(),
            openai_key: std::env::var("OPENAI_API_KEY").ok(),
            gemini_key: std::env::var("GEMINI_API_KEY").ok(),
        }
    }

    /// True when explicit frankensearch remote-embedding configuration is
    /// present. A bare `OPENAI_API_KEY`/`GEMINI_API_KEY` does NOT count:
    /// those are ambient credentials that unrelated tools export, and
    /// treating them as frankensearch intent would break every host that
    /// has one set.
    fn has_explicit_intent(&self) -> bool {
        self.provider.is_some()
            || self.model.is_some()
            || self.dimension.is_some()
            || self.identity_json.is_some()
    }
}

fn resolve_offline_policy(options: DetectOptions, env: &RemoteIntentEnv) -> SearchResult<bool> {
    if let Some(offline) = options.offline {
        return Ok(offline);
    }
    let Some(raw) = env.offline.as_deref() else {
        return Ok(false);
    };
    parse_bool_flag(raw).ok_or_else(|| SearchError::InvalidConfig {
        field: OFFLINE_ENV.to_owned(),
        value: raw.to_owned(),
        reason: "expected a boolean flag: 1/0, true/false, yes/no, or on/off".to_owned(),
    })
}

fn unverifiable_remote(reason: &str) -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: "environment".to_owned(),
        reason: reason.to_owned(),
    }
}

/// Resolve explicit remote-embedding intent to a verified embedder, a typed
/// failure, or nothing (bd-p6z6.2).
///
/// - `Ok((offline, None))`: no explicit frankensearch remote configuration
///   exists; `offline` is the single resolved policy snapshot shared with
///   model-download admission.
///   Ambient provider keys without any `FRANKENSEARCH_API_*` setting remain
///   ignored, including while offline.
/// - `Ok((false, Some(_)))`: explicit intent, producer-attested frozen
///   identity, verified construction.
/// - `Err(EmbedderUnavailable)`: explicit remote intent exists while the
///   effective detection policy is offline. This gate runs before provider
///   construction.
/// - `Err(UnverifiableRemoteSpace)`: explicit intent that is missing,
///   malformed, unverifiable, or producer-unattested. Detection never
///   converts explicit intent into `None` — a silent downgrade to
///   local/hash embedders would serve vectors from the wrong space.
pub(super) fn resolve_remote_intent(
    options: DetectOptions,
    env: &RemoteIntentEnv,
) -> SearchResult<(bool, Option<Arc<dyn Embedder>>)> {
    resolve_remote_intent_with(options, env, resolve_remote_intent_online)
}

pub(super) fn resolve_remote_intent_with<T>(
    options: DetectOptions,
    env: &RemoteIntentEnv,
    construct_online: impl FnOnce(&RemoteIntentEnv) -> SearchResult<Option<T>>,
) -> SearchResult<(bool, Option<T>)> {
    let offline = resolve_offline_policy(options, env)?;
    if offline {
        if env.has_explicit_intent() {
            return Err(SearchError::EmbedderUnavailable {
                model: "remote-tier".to_owned(),
                reason: format!(
                    "{OFFLINE_ENV}=1 or an explicit offline detection policy forbids remote \
                     provider construction"
                ),
            });
        }
        return Ok((true, None));
    }
    construct_online(env).map(|remote| (false, remote))
}

#[cfg(feature = "api")]
fn resolve_remote_intent_online(env: &RemoteIntentEnv) -> SearchResult<Option<Arc<dyn Embedder>>> {
    use crate::api_embedder::ApiEmbedder;
    use crate::api_provider::{GeminiProvider, OpenAiProvider};

    let explicit_dim: Option<usize> = env.dimension.as_deref().and_then(|s| s.parse().ok());
    let Some(identity_json) = env.identity_json.as_deref() else {
        if env.has_explicit_intent() {
            return Err(unverifiable_remote(
                "explicit remote configuration is present but \
                 FRANKENSEARCH_API_IDENTITY_JSON (the immutable space epoch) is not set",
            ));
        }
        if env.openai_key.is_some() || env.gemini_key.is_some() {
            warn!(
                field = "FRANKENSEARCH_API_IDENTITY_JSON",
                "ambient provider key ignored: no frankensearch remote configuration \
                 and no immutable space epoch was supplied"
            );
        }
        return Ok(None);
    };
    let identity: FrozenEmbeddingIdentityBundleV1 =
        serde_json::from_str(identity_json).map_err(|_| {
            unverifiable_remote("FRANKENSEARCH_API_IDENTITY_JSON is not valid identity JSON")
        })?;
    identity.validate().map_err(|_| {
        unverifiable_remote("FRANKENSEARCH_API_IDENTITY_JSON failed identity validation")
    })?;

    let provider: Box<dyn crate::api_provider::ApiProvider> = match env.provider.as_deref() {
        Some("gemini") => {
            let key = env.gemini_key.clone().ok_or_else(|| {
                unverifiable_remote("provider gemini is configured but GEMINI_API_KEY is not set")
            })?;
            match env.model.as_deref() {
                Some("embedding-001") => Box::new(GeminiProvider::embedding_001(key)),
                _ => Box::new(GeminiProvider::text_embedding_004(key)),
            }
        }
        Some("openai") => {
            let key = env.openai_key.clone().ok_or_else(|| {
                unverifiable_remote("provider openai is configured but OPENAI_API_KEY is not set")
            })?;
            match env.model.as_deref() {
                Some("text-embedding-3-large") => {
                    Box::new(OpenAiProvider::text_embedding_3_large(key, explicit_dim))
                }
                _ => Box::new(OpenAiProvider::text_embedding_3_small(key, explicit_dim)),
            }
        }
        None => {
            // Identity supplied without a provider name: prefer OpenAI if
            // its key is present, otherwise Gemini.
            if let Some(key) = env.openai_key.clone() {
                match env.model.as_deref() {
                    Some("text-embedding-3-large") => {
                        Box::new(OpenAiProvider::text_embedding_3_large(key, explicit_dim))
                    }
                    _ => Box::new(OpenAiProvider::text_embedding_3_small(key, explicit_dim)),
                }
            } else if let Some(key) = env.gemini_key.clone() {
                match env.model.as_deref() {
                    Some("embedding-001") => Box::new(GeminiProvider::embedding_001(key)),
                    _ => Box::new(GeminiProvider::text_embedding_004(key)),
                }
            } else {
                return Err(unverifiable_remote(
                    "an immutable space epoch is configured but no provider API key is set",
                ));
            }
        }
        Some(_other) => {
            return Err(unverifiable_remote(
                "FRANKENSEARCH_API_PROVIDER names an unknown provider",
            ));
        }
    };

    info!(
        provider = provider.provider_name(),
        dimension = provider.dimension(),
        identity = identity.fingerprint.as_str(),
        "detected API embedder from environment"
    );

    ApiEmbedder::with_defaults(provider, Some(identity))
        .map(|embedder| Some(Arc::new(embedder.cached_default()) as Arc<dyn Embedder>))
        .map_err(|_| {
            unverifiable_remote("the configured provider and the immutable space epoch disagree")
        })
}

/// Without the `api` feature the build cannot satisfy remote intent, so
/// explicit configuration is a typed failure rather than a silent ignore.
#[cfg(not(feature = "api"))]
fn resolve_remote_intent_online(env: &RemoteIntentEnv) -> SearchResult<Option<Arc<dyn Embedder>>> {
    if env.has_explicit_intent() {
        return Err(unverifiable_remote(
            "explicit remote configuration is present but this build lacks the `api` feature",
        ));
    }
    if env.openai_key.is_some() || env.gemini_key.is_some() {
        warn!(
            "ambient provider key ignored: no frankensearch remote configuration \
             was supplied and this build lacks the `api` feature"
        );
    }
    Ok(None)
}

/// Manifest-required files that are absent from `model_dir`.
///
/// Returns the names rather than a bool so a candidate that exists but is
/// incomplete — a half-finished download — can say which files it is short of.
/// That directory looks installed to a user and is silently skipped here, which
/// makes it one of the harder "why is semantic search off?" cases to diagnose.
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
fn missing_manifest_files<'a>(manifest: &'a ModelManifest, model_dir: &Path) -> Vec<&'a str> {
    manifest
        .files
        .iter()
        .filter(|file| !model_dir.join(&file.name).is_file())
        .map(|file| file.name.as_str())
        .collect()
}

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
fn candidate_directories(
    model_root: Option<&Path>,
    model_name: &str,
    discovered: Option<&Path>,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(root) = model_root {
        paths.push(root.join(model_name));
        paths.push(root.to_path_buf());
    }
    if let Some(path) = discovered {
        paths.push(path.to_path_buf());
    }

    let mut seen = BTreeSet::new();
    paths
        .into_iter()
        .filter(|path| seen.insert(path.clone()))
        .collect()
}
