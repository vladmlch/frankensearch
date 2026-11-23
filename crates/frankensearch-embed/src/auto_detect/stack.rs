//! Embedder stack assembly, readiness reporting, diagnostics, and MRL wrapping.

use std::fmt::{self, Write as _};
use std::path::Path;
use std::sync::Arc;

use asupersync::Cx;
use tracing::{info, warn};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{Embedder, SearchFuture};

use super::discovery::{
    DetectOptions, RemoteIntentEnv, detect_fast_embedder, detect_quality_embedder,
    hash_fallback_embedder, resolve_remote_intent,
};
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use super::lazy_download::{
    DownloadPolicy, download_policy_from_environment, maybe_lazy_fast_embedder,
    maybe_lazy_quality_embedder,
};

/// Availability classification for two-tier search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoTierAvailability {
    /// Fast + quality embedders available.
    Full,
    /// Semantic fast embedder available, no quality tier.
    FastOnly,
    /// Hash-only fallback path.
    HashOnly,
}

impl TwoTierAvailability {
    /// Whether this availability level represents a degraded state.
    #[must_use]
    pub const fn is_degraded(self) -> bool {
        matches!(self, Self::FastOnly | Self::HashOnly)
    }

    /// Human-readable summary of what is missing at this availability level.
    #[must_use]
    pub const fn degradation_summary(self) -> Option<&'static str> {
        match self {
            Self::Full => None,
            Self::FastOnly => Some(
                "Quality model unavailable: search will return fast-tier results only (no refinement phase).",
            ),
            Self::HashOnly => Some(
                "No semantic models available: hash control embeddings are not semantic search.",
            ),
        }
    }
}

impl fmt::Display for TwoTierAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full (fast + quality)"),
            Self::FastOnly => write!(f, "degraded (fast-only, no quality refinement)"),
            Self::HashOnly => write!(f, "minimal (hash-only, no semantic search)"),
        }
    }
}

/// Diagnostic information about model availability for user-facing messages.
///
/// Provides actionable guidance when models are missing, including cache paths,
/// download URLs, and environment variable hints.
#[derive(Debug, Clone)]
pub struct ModelAvailabilityDiagnostic {
    /// Current availability classification.
    pub availability: TwoTierAvailability,
    /// Resolved model cache directory path.
    pub cache_dir: std::path::PathBuf,
    /// Whether the system is in offline mode.
    pub offline: bool,
    /// Fast-tier model status.
    pub fast_status: ModelStatus,
    /// Quality-tier model status.
    pub quality_status: ModelStatus,
    /// Suggestions for the user to resolve degraded state.
    pub suggestions: Vec<String>,
}

/// Status of an individual model tier.
#[derive(Debug, Clone)]
pub enum ModelStatus {
    /// Model is loaded and ready.
    Ready {
        /// Model identifier.
        id: String,
    },
    /// Model files not found locally.
    NotFound {
        /// Model name that was searched for.
        model_name: String,
        /// `HuggingFace` repository URL for manual download.
        hf_repo_url: String,
        /// Paths that were searched.
        searched_paths: Vec<std::path::PathBuf>,
    },
    /// Download was blocked by policy (offline mode or consent denied).
    DownloadBlocked {
        /// Model name.
        model_name: String,
        /// Reason download was blocked.
        reason: String,
    },
    /// Feature not compiled in.
    FeatureDisabled {
        /// The feature flag that would enable this tier.
        feature_flag: String,
    },
    /// Hash control embedder is present; this is not a semantic model.
    HashFallback,
}

impl fmt::Display for ModelAvailabilityDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model availability: {}", self.availability)?;
        writeln!(f, "Cache directory: {}", self.cache_dir.display())?;
        if self.offline {
            writeln!(f, "Mode: OFFLINE (FRANKENSEARCH_OFFLINE=1)")?;
        }
        writeln!(f)?;
        writeln!(f, "Fast tier:    {}", self.fast_status)?;
        writeln!(f, "Quality tier: {}", self.quality_status)?;
        if !self.suggestions.is_empty() {
            writeln!(f)?;
            writeln!(f, "To resolve:")?;
            for suggestion in &self.suggestions {
                writeln!(f, "  - {suggestion}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready { id } => write!(f, "ready ({id})"),
            Self::NotFound {
                model_name,
                hf_repo_url,
                ..
            } => {
                write!(f, "NOT FOUND ({model_name}) — download from {hf_repo_url}")
            }
            Self::DownloadBlocked { model_name, reason } => {
                write!(f, "BLOCKED ({model_name}): {reason}")
            }
            Self::FeatureDisabled { feature_flag } => {
                write!(f, "DISABLED (compile with --features {feature_flag})")
            }
            Self::HashFallback => write!(f, "hash control (not a semantic model)"),
        }
    }
}

/// Resolved fast/quality embedder stack for progressive search.
#[derive(Clone)]
pub struct EmbedderStack {
    fast: Arc<dyn Embedder>,
    quality: Option<Arc<dyn Embedder>>,
    availability: TwoTierAvailability,
}

impl fmt::Debug for EmbedderStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fast_identity = self.fast.identity().map_or_else(
            |_| "unverifiable".to_owned(),
            EmbeddingIdentityBundleV1::fingerprint,
        );
        let quality_identity = self.quality.as_ref().map(|embedder| {
            embedder.identity().map_or_else(
                |_| "unverifiable".to_owned(),
                EmbeddingIdentityBundleV1::fingerprint,
            )
        });
        f.debug_struct("EmbedderStack")
            .field("availability", &self.availability)
            .field("fast_category", &self.fast.category())
            .field("fast_dim", &self.fast.dimension())
            .field("fast_identity", &fast_identity)
            .field(
                "quality_category",
                &self.quality.as_ref().map(|embedder| embedder.category()),
            )
            .field("quality_identity", &quality_identity)
            .finish()
    }
}

fn classify_stack_availability(
    fast: &dyn Embedder,
    quality: Option<&dyn Embedder>,
) -> TwoTierAvailability {
    if !fast.is_semantic() {
        TwoTierAvailability::HashOnly
    } else if quality.is_some_and(Embedder::is_semantic) {
        TwoTierAvailability::Full
    } else {
        TwoTierAvailability::FastOnly
    }
}

impl EmbedderStack {
    /// Build from explicit parts.
    ///
    /// Availability follows embedder identity, not slot occupancy: two hash
    /// control embedders are [`TwoTierAvailability::HashOnly`], not `Full`.
    #[must_use]
    pub fn from_parts(fast: Arc<dyn Embedder>, quality: Option<Arc<dyn Embedder>>) -> Self {
        let availability = classify_stack_availability(fast.as_ref(), quality.as_deref());
        Self {
            fast,
            quality,
            availability,
        }
    }

    /// Auto-detect best available embedders from default search paths.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbedderUnavailable` when no usable fast embedder is available.
    pub fn auto_detect() -> SearchResult<Self> {
        Self::auto_detect_with(None)
    }

    /// Auto-detect embedders with an optional explicit model root override.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbedderUnavailable` when no usable fast embedder is available.
    pub fn auto_detect_with(model_root: Option<&Path>) -> SearchResult<Self> {
        Self::auto_detect_with_options(model_root, &DetectOptions::default())
    }

    /// Auto-detect and refuse a hash-only stack.
    ///
    /// Use this when the caller asked for semantic search. [`Self::auto_detect`]
    /// still returns `HashOnly` for tests and explicit control policy.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::EmbedderUnavailable`] when no semantic model is
    /// present.
    pub fn auto_detect_semantic() -> SearchResult<Self> {
        Self::auto_detect()?.require_semantic()
    }

    /// Like [`Self::auto_detect_with`], then [`Self::require_semantic`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::EmbedderUnavailable`] when no semantic model is
    /// present.
    pub fn auto_detect_semantic_with(model_root: Option<&Path>) -> SearchResult<Self> {
        Self::auto_detect_with(model_root)?.require_semantic()
    }

    /// Like [`Self::auto_detect_with_options`], then [`Self::require_semantic`].
    ///
    /// Hosts that already have an explicit offline/remote policy should use
    /// this instead of detecting a hash-only stack and discarding it later.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::EmbedderUnavailable`] when no semantic model is
    /// present, plus the same policy errors as
    /// [`Self::auto_detect_with_options`].
    pub fn auto_detect_semantic_with_options(
        model_root: Option<&Path>,
        options: &DetectOptions,
    ) -> SearchResult<Self> {
        Self::auto_detect_with_options(model_root, options)?.require_semantic()
    }

    /// Detect only the fast tier and refuse a hash-only result.
    ///
    /// A host whose read path never consults a quality tier (the `fsfs` CLI
    /// serves one fast-tier vector generation) must not pay for opening the
    /// quality model on every process start: on the registered stack that is
    /// an ONNX session load worth hundreds of milliseconds per invocation.
    /// This resolves the same offline/remote policy as
    /// [`Self::auto_detect_with_options`], constructs the fast tier only, and
    /// applies [`Self::require_semantic`]. The returned stack reports
    /// [`TwoTierAvailability::FastOnly`] and has no quality embedder; callers
    /// that later need the quality tier resolve it separately.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::EmbedderUnavailable`] when no semantic fast
    /// model is present (hash control is refused), plus the same policy errors
    /// as [`Self::auto_detect_with_options`].
    pub fn auto_detect_fast_semantic_with_options(
        model_root: Option<&Path>,
        options: &DetectOptions,
    ) -> SearchResult<Self> {
        let remote_env = RemoteIntentEnv::from_environment();
        let (offline, _remote) = resolve_remote_intent(*options, &remote_env)?;

        #[cfg(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        ))]
        let fast = {
            let policy = download_policy_from_environment(offline);
            detect_fast_embedder(model_root)
                .or_else(|| maybe_lazy_fast_embedder(model_root, policy))
        };
        #[cfg(not(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        )))]
        let fast = {
            let _ = offline;
            detect_fast_embedder(model_root)
        };

        let fast = fast.or_else(hash_fallback_embedder).ok_or_else(|| {
            SearchError::EmbedderUnavailable {
                model: "fast-tier".to_owned(),
                reason: "no model2vec/hash embedder available in this build".to_owned(),
            }
        })?;
        let stack = Self::from_parts(fast, None);
        stack.report_readiness();
        stack.require_semantic()
    }

    /// Detect only the quality tier under caller-supplied policy.
    ///
    /// The complement of [`Self::auto_detect_fast_semantic_with_options`]:
    /// a host that already holds its fast embedder and now needs the quality
    /// model (to build or refine against a quality-tier generation) must not
    /// re-open the fast model to get it. Returns `None` when no quality
    /// model is present or admissible under the effective offline policy;
    /// a hash control embedder is never returned as a quality tier.
    ///
    /// # Errors
    ///
    /// Returns the same policy errors as [`Self::auto_detect_with_options`]
    /// (offline/remote intent resolution).
    pub fn auto_detect_quality_with_options(
        model_root: Option<&Path>,
        options: &DetectOptions,
    ) -> SearchResult<Option<Arc<dyn Embedder>>> {
        let remote_env = RemoteIntentEnv::from_environment();
        let (offline, remote) = resolve_remote_intent(*options, &remote_env)?;

        #[cfg(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        ))]
        let quality = {
            let policy = download_policy_from_environment(offline);
            detect_quality_embedder(model_root)
                .or_else(|| maybe_lazy_quality_embedder(model_root, policy))
                .or(remote)
        };
        #[cfg(not(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        )))]
        let quality = {
            let _ = offline;
            detect_quality_embedder(model_root).or(remote)
        };

        Ok(quality.filter(|embedder| embedder.is_semantic()))
    }

    /// Fail closed when this stack has no semantic fast embedder.
    ///
    /// Hash remains available via [`Self::auto_detect`] and
    /// [`Self::from_parts`]. This method is the typed refusal hosts should
    /// call instead of checking [`TwoTierAvailability::HashOnly`] themselves.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::EmbedderUnavailable`] for a hash-only stack.
    pub fn require_semantic(self) -> SearchResult<Self> {
        if matches!(self.availability, TwoTierAvailability::HashOnly) {
            return Err(SearchError::EmbedderUnavailable {
                model: "semantic".to_owned(),
                reason: self.degradation_message().unwrap_or_else(|| {
                    "no semantic model; hash-only stacks are not a working search engine".to_owned()
                }),
            });
        }
        Ok(self)
    }

    /// Auto-detect embedders under caller-supplied policy (bd-p6z6.2).
    ///
    /// Callers (fsfs config, library hosts) pass policy EXPLICITLY instead
    /// of relying on process environment: `options.offline = Some(true)`
    /// forbids model downloads and remote provider construction for this
    /// detection regardless of `FRANKENSEARCH_OFFLINE`, and `Some(false)`
    /// ignores the env variable entirely — environment cannot silently change
    /// library semantics when the caller states intent. `None` fields defer to
    /// the environment, preserving `auto_detect_with` behavior. Detection
    /// itself never prompts: interactive consent is a product-surface concern.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbedderUnavailable` when no usable fast
    /// embedder is available or explicit remote intent is prohibited by the
    /// effective offline policy, and `SearchError::UnverifiableRemoteSpace`
    /// when explicit remote configuration cannot be verified. Returns
    /// `SearchError::InvalidConfig` when a deferred
    /// `FRANKENSEARCH_OFFLINE` value is not a recognized boolean flag.
    pub fn auto_detect_with_options(
        model_root: Option<&Path>,
        options: &DetectOptions,
    ) -> SearchResult<Self> {
        let remote_env = RemoteIntentEnv::from_environment();
        let (offline, remote) = resolve_remote_intent(*options, &remote_env)?;

        #[cfg(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        ))]
        {
            let policy = download_policy_from_environment(offline);
            Self::auto_detect_with_policy(model_root, policy, remote)
        }
        #[cfg(not(all(
            feature = "download",
            any(feature = "model2vec", feature = "fastembed")
        )))]
        {
            let _ = offline;
            Self::auto_detect_with_policy(model_root, remote)
        }
    }

    #[cfg(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    ))]
    pub(super) fn auto_detect_with_policy(
        model_root: Option<&Path>,
        policy: DownloadPolicy,
        remote: Option<Arc<dyn Embedder>>,
    ) -> SearchResult<Self> {
        let quality = detect_quality_embedder(model_root)
            .or_else(|| maybe_lazy_quality_embedder(model_root, policy))
            .or(remote);
        let fast = detect_fast_embedder(model_root)
            .or_else(|| maybe_lazy_fast_embedder(model_root, policy))
            .or_else(hash_fallback_embedder)
            .ok_or_else(|| SearchError::EmbedderUnavailable {
                model: "fast-tier".to_owned(),
                reason: "no model2vec/hash embedder available in this build".to_owned(),
            })?;

        let stack = Self::from_parts(fast, quality);
        stack.report_readiness();
        Ok(stack)
    }

    #[cfg(not(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    )))]
    pub(super) fn auto_detect_with_policy(
        model_root: Option<&Path>,
        remote: Option<Arc<dyn Embedder>>,
    ) -> SearchResult<Self> {
        let quality = detect_quality_embedder(model_root).or(remote);
        let fast = detect_fast_embedder(model_root)
            .or_else(hash_fallback_embedder)
            .ok_or_else(|| SearchError::EmbedderUnavailable {
                model: "fast-tier".to_owned(),
                reason: "no model2vec/hash embedder available in this build".to_owned(),
            })?;

        let stack = Self::from_parts(fast, quality);
        stack.report_readiness();
        Ok(stack)
    }

    /// Report what auto-detection actually produced.
    ///
    /// A `HashOnly` stack is not a working semantic search engine. FNV-1a
    /// vectors carry no meaning, and [`TwoTierSearcher`] treats a shipped hash
    /// fast-embedder as a reason to skip the vector arm outright — so the
    /// caller receives an ordinary-looking `Ok` holding lexical-only results.
    /// Announcing that at `info!` is precisely how "semantic search never
    /// works" reaches users as silence, so it is escalated to `warn!` and
    /// carries a bounded reason code. The richer
    /// [`degradation_message`](Self::degradation_message) remains available to
    /// an explicit user-facing diagnostics surface because it includes the
    /// configured model-cache path and must not enter structured logs.
    ///
    /// [`TwoTierSearcher`]: https://docs.rs/frankensearch/latest/frankensearch/struct.TwoTierSearcher.html
    fn report_readiness(&self) {
        let fast_identity = self.fast.identity().map_or_else(
            |_| "unverifiable".to_owned(),
            EmbeddingIdentityBundleV1::fingerprint,
        );
        let quality_identity = self.quality.as_ref().map(|embedder| {
            embedder.identity().map_or_else(
                |_| "unverifiable".to_owned(),
                EmbeddingIdentityBundleV1::fingerprint,
            )
        });
        let quality_category = self.quality.as_ref().map(|embedder| embedder.category());
        if matches!(self.availability, TwoTierAvailability::HashOnly) {
            warn!(
                availability = ?self.availability,
                fast_category = ?self.fast.category(),
                fast_identity = %fast_identity,
                quality_category = ?quality_category,
                quality_identity = ?quality_identity,
                reason = "no-semantic-model",
                "SEMANTIC SEARCH UNAVAILABLE: no embedding model was found, so \
                 retrieval fell back to non-semantic hash vectors. Vector search \
                 will not return meaningful results until a model is installed."
            );
            return;
        }
        info!(
            availability = ?self.availability,
            fast_category = ?self.fast.category(),
            fast_identity = %fast_identity,
            quality_category = ?quality_category,
            quality_identity = ?quality_identity,
            "embedder stack ready"
        );
    }

    /// Apply MRL-style dimensionality reduction where supported.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when `target_dim` is zero.
    pub fn with_mrl_target_dim(mut self, target_dim: usize) -> SearchResult<Self> {
        if target_dim == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: "0".to_owned(),
                reason: "target dimension must be at least 1".to_owned(),
            });
        }

        self.fast = maybe_wrap_mrl(self.fast.clone(), target_dim)?;
        self.quality = self
            .quality
            .clone()
            .map(|embedder| maybe_wrap_mrl(embedder, target_dim))
            .transpose()?;
        self.availability =
            classify_stack_availability(self.fast.as_ref(), self.quality.as_deref());
        Ok(self)
    }

    /// Fast embedder reference.
    #[must_use]
    pub fn fast(&self) -> &dyn Embedder {
        self.fast.as_ref()
    }

    /// Fast embedder alias for API compatibility.
    #[must_use]
    pub fn fast_embedder(&self) -> &dyn Embedder {
        self.fast()
    }

    /// Cloned fast embedder handle.
    #[must_use]
    pub fn fast_arc(&self) -> Arc<dyn Embedder> {
        self.fast.clone()
    }

    /// Optional quality embedder reference.
    #[must_use]
    pub fn quality(&self) -> Option<&dyn Embedder> {
        self.quality.as_deref()
    }

    /// Optional quality embedder alias for API compatibility.
    #[must_use]
    pub fn quality_embedder(&self) -> Option<&dyn Embedder> {
        self.quality()
    }

    /// Cloned quality embedder handle.
    #[must_use]
    pub fn quality_arc(&self) -> Option<Arc<dyn Embedder>> {
        self.quality.clone()
    }

    /// Availability state.
    #[must_use]
    pub const fn availability(&self) -> TwoTierAvailability {
        self.availability
    }

    /// Generate a user-facing diagnostic report about model availability.
    ///
    /// Includes actionable suggestions for resolving degraded states:
    /// cache directory paths, manual download URLs, environment variable hints.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn diagnose(&self) -> ModelAvailabilityDiagnostic {
        let cache_dir = crate::model_cache::resolve_cache_root();
        let offline = std::env::var("FRANKENSEARCH_OFFLINE")
            .ok()
            .as_deref()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        let fast_status = if self.fast.is_semantic() {
            ModelStatus::Ready {
                id: self.fast.id().to_owned(),
            }
        } else {
            ModelStatus::HashFallback
        };

        #[allow(clippy::option_if_let_else)] // cfg blocks in else arm prevent map_or_else
        let quality_status = if let Some(ref quality) = self.quality {
            ModelStatus::Ready {
                id: quality.id().to_owned(),
            }
        } else {
            #[cfg(feature = "fastembed")]
            {
                if offline {
                    ModelStatus::DownloadBlocked {
                        model_name: "all-MiniLM-L6-v2".to_owned(),
                        reason: "FRANKENSEARCH_OFFLINE=1 disables auto-download".to_owned(),
                    }
                } else {
                    ModelStatus::NotFound {
                        model_name: "all-MiniLM-L6-v2".to_owned(),
                        hf_repo_url:
                            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
                                .to_owned(),
                        searched_paths: vec![cache_dir.join("all-MiniLM-L6-v2")],
                    }
                }
            }
            #[cfg(not(feature = "fastembed"))]
            {
                ModelStatus::FeatureDisabled {
                    feature_flag: "fastembed".to_owned(),
                }
            }
        };

        let mut suggestions = Vec::new();
        if self.availability.is_degraded() {
            #[cfg(not(feature = "bundled-default-models"))]
            if offline {
                suggestions.push(
                    "Unset FRANKENSEARCH_OFFLINE to allow automatic model downloads.".to_owned(),
                );
            }
            suggestions.push(format!(
                "Set FRANKENSEARCH_MODEL_DIR to point to a pre-populated model cache (current: {}).",
                cache_dir.display()
            ));

            if matches!(self.availability, TwoTierAvailability::HashOnly) {
                #[cfg(all(feature = "bundled-default-models", feature = "model2vec"))]
                suggestions.push(
                    "Default semantic models are bundled in fsfs. If still unavailable, ensure the model cache path is writable and run `fsfs status`."
                        .to_owned(),
                );
                #[cfg(feature = "model2vec")]
                #[cfg(not(feature = "bundled-default-models"))]
                suggestions.push(
                    "Download potion-multilingual-128M from https://huggingface.co/minishlab/potion-multilingual-128M and place in cache dir."
                        .to_owned(),
                );
                #[cfg(not(feature = "model2vec"))]
                suggestions.push(
                    "Compile with --features model2vec to enable the fast semantic tier."
                        .to_owned(),
                );
            }
            if self.quality.is_none() {
                #[cfg(all(feature = "bundled-default-models", feature = "fastembed"))]
                suggestions.push(
                    "Quality model should be bundled by default. Check cache permissions and verify `all-MiniLM-L6-v2` exists under the model directory."
                        .to_owned(),
                );
                #[cfg(feature = "fastembed")]
                #[cfg(not(feature = "bundled-default-models"))]
                suggestions.push(
                    "Download all-MiniLM-L6-v2 from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 and place in cache dir."
                        .to_owned(),
                );
                #[cfg(not(feature = "fastembed"))]
                suggestions.push(
                    "Compile with --features fastembed to enable the quality semantic tier."
                        .to_owned(),
                );
            }
            #[cfg(not(feature = "bundled-default-models"))]
            suggestions.push(
                "For air-gapped environments: run `fsfs download-models --output ./models/` on a networked machine, then copy to target."
                    .to_owned(),
            );
            #[cfg(feature = "bundled-default-models")]
            suggestions.push(
                "Optional: use `fsfs download-models` only when you want alternate semantic models beyond the bundled defaults."
                    .to_owned(),
            );
        }

        ModelAvailabilityDiagnostic {
            availability: self.availability,
            cache_dir,
            offline,
            fast_status,
            quality_status,
            suggestions,
        }
    }

    /// Returns a user-facing message if operating in a degraded mode, or `None` if fully available.
    #[must_use]
    pub fn degradation_message(&self) -> Option<String> {
        if !self.availability.is_degraded() {
            return None;
        }

        let diag = self.diagnose();
        let mut msg = String::new();
        if let Some(summary) = self.availability.degradation_summary() {
            msg.push_str(summary);
            msg.push('\n');
        }
        let _ = writeln!(msg, "Model cache: {}", diag.cache_dir.display());
        if diag.offline {
            msg.push_str("Offline mode: enabled (FRANKENSEARCH_OFFLINE=1)\n");
        }
        if !diag.suggestions.is_empty() {
            if matches!(self.availability, TwoTierAvailability::HashOnly) {
                msg.push_str(
                    "\nTo enable semantic search (installing a model cannot repair a hash-written index):\n",
                );
            } else {
                msg.push_str("\nTo improve search quality:\n");
            }
            for suggestion in &diag.suggestions {
                let _ = writeln!(msg, "  - {suggestion}");
            }
        }
        Some(msg)
    }
}

/// MRL dimension reduction wrapper.
pub struct DimReduceEmbedder {
    inner: Arc<dyn Embedder>,
    target_dim: usize,
    id: String,
    model_name: String,
    identity: EmbeddingIdentityBundleV1,
}

impl fmt::Debug for DimReduceEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DimReduceEmbedder")
            .field("target_dim", &self.target_dim)
            .field("identity", &self.identity.fingerprint())
            .finish_non_exhaustive()
    }
}

impl DimReduceEmbedder {
    /// Create a dimension-reduced view of an embedder.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when requested dimension is invalid.
    pub fn new(inner: Arc<dyn Embedder>, target_dim: usize) -> SearchResult<Self> {
        if target_dim == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: "0".to_owned(),
                reason: "target dimension must be at least 1".to_owned(),
            });
        }
        if target_dim > inner.dimension() {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: target_dim.to_string(),
                reason: format!(
                    "target dimension cannot exceed embedder dimension {}",
                    inner.dimension()
                ),
            });
        }
        if !inner.supports_mrl() {
            return Err(SearchError::InvalidConfig {
                field: "embedder.supports_mrl".to_owned(),
                value: inner.id().to_owned(),
                reason: "embedder does not support MRL truncation".to_owned(),
            });
        }
        let target_dimension =
            u32::try_from(target_dim).map_err(|_| SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: target_dim.to_string(),
                reason: "target dimension does not fit the identity schema".to_owned(),
            })?;
        let identity = inner.identity()?.derive_projection(
            target_dimension,
            "prefix-truncate-first-n-dimensions-v1",
            "l2-f32-zero-on-degenerate-v1",
        )?;

        Ok(Self {
            id: format!("{}-mrl-{target_dim}", inner.id()),
            model_name: format!("{} (MRL {target_dim})", inner.model_name()),
            identity,
            inner,
            target_dim,
        })
    }
}

impl Embedder for DimReduceEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let full = self.inner.embed(cx, text).await?;
            self.inner.truncate_embedding(&full, self.target_dim)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let full_batch = self.inner.embed_batch(cx, texts).await?;
            full_batch
                .iter()
                .map(|embedding| self.inner.truncate_embedding(embedding, self.target_dim))
                .collect()
        })
    }

    /// Both bound operations are the core defaults over this wrapper's own
    /// `embed`/`embed_batch`, so a bound batch binds what singles would.
    fn bound_batch_is_native(&self) -> bool {
        true
    }

    fn dimension(&self) -> usize {
        self.target_dim
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> frankensearch_core::traits::ModelCategory {
        self.inner.category()
    }

    fn tier(&self) -> frankensearch_core::traits::ModelTier {
        self.inner.tier()
    }

    fn supports_mrl(&self) -> bool {
        true
    }
}

fn maybe_wrap_mrl(
    embedder: Arc<dyn Embedder>,
    target_dim: usize,
) -> SearchResult<Arc<dyn Embedder>> {
    if target_dim >= embedder.dimension() || !embedder.supports_mrl() {
        return Ok(embedder);
    }
    Ok(Arc::new(DimReduceEmbedder::new(embedder, target_dim)?))
}
