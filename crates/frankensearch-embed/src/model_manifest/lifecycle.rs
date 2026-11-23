//! Model download consent resolution and runtime lifecycle state machine.

use frankensearch_core::error::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use super::DOWNLOAD_CONSENT_ENV;
use super::catalog::ModelManifest;
use super::verification::{invalid_state_transition, parse_bool_flag};

/// Runtime state of model availability and lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelState {
    NotInstalled,
    NeedsConsent,
    Downloading {
        progress_pct: u8,
        bytes_downloaded: u64,
        total_bytes: u64,
    },
    Verifying,
    /// Every staged byte is verified, but the generation is not published.
    StagedVerified,
    /// Artifacts are installed and load-tested, but no compatible index is selected yet.
    AcquiredNeedsReindex,
    Ready,
    Disabled {
        reason: String,
    },
    VerificationFailed {
        reason: String,
    },
    UpdateAvailable {
        current_revision: String,
        latest_revision: String,
    },
    Cancelled,
}

impl ModelState {
    /// Whether the model is ready for use.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Whether a download is in progress.
    #[must_use]
    pub fn is_downloading(&self) -> bool {
        matches!(self, Self::Downloading { .. })
    }

    /// Whether user consent is needed.
    #[must_use]
    pub fn needs_consent(&self) -> bool {
        matches!(self, Self::NeedsConsent)
    }

    /// Human-readable summary of the state.
    #[must_use]
    pub fn summary(&self) -> String {
        match self {
            Self::NotInstalled => "not installed".into(),
            Self::NeedsConsent => "needs consent".into(),
            Self::Downloading { progress_pct, .. } => {
                format!("downloading ({progress_pct}%)")
            }
            Self::Verifying => "verifying".into(),
            Self::StagedVerified => "staged and verified".into(),
            Self::AcquiredNeedsReindex => "acquired; compatible index required".into(),
            Self::Ready => "ready".into(),
            Self::Disabled { reason } => format!("disabled: {reason}"),
            Self::VerificationFailed { reason } => format!("verification failed: {reason}"),
            Self::UpdateAvailable {
                current_revision,
                latest_revision,
            } => {
                format!("update available: {current_revision} -> {latest_revision}")
            }
            Self::Cancelled => "cancelled".into(),
        }
    }
}

/// Where a consent decision came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentSource {
    Programmatic,
    Environment,
    Interactive,
    ConfigFile,
}

/// Resolved consent decision for model downloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadConsent {
    /// Whether downloads are allowed.
    pub granted: bool,
    /// Origin of the consent signal.
    pub source: Option<ConsentSource>,
}

impl DownloadConsent {
    /// Explicitly granted consent.
    #[must_use]
    pub const fn granted(source: ConsentSource) -> Self {
        Self {
            granted: true,
            source: Some(source),
        }
    }

    /// Explicitly denied consent.
    #[must_use]
    pub const fn denied(source: Option<ConsentSource>) -> Self {
        Self {
            granted: false,
            source,
        }
    }
}

/// Resolve download consent using priority:
/// programmatic > environment > interactive > config.
#[must_use]
pub fn resolve_download_consent(
    programmatic: Option<bool>,
    interactive: Option<bool>,
    config_file: Option<bool>,
) -> DownloadConsent {
    let env_value = std::env::var(DOWNLOAD_CONSENT_ENV).ok();
    resolve_download_consent_with_env(programmatic, env_value.as_deref(), interactive, config_file)
}

pub(super) fn resolve_download_consent_with_env(
    programmatic: Option<bool>,
    env_value: Option<&str>,
    interactive: Option<bool>,
    config_file: Option<bool>,
) -> DownloadConsent {
    if let Some(granted) = programmatic {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Programmatic),
        };
    }

    if let Some(raw) = env_value
        && let Some(granted) = parse_bool_flag(raw)
    {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Environment),
        };
    }

    if let Some(granted) = interactive {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::Interactive),
        };
    }

    if let Some(granted) = config_file {
        return DownloadConsent {
            granted,
            source: Some(ConsentSource::ConfigFile),
        };
    }

    DownloadConsent::denied(None)
}

/// Stateful lifecycle helper for model installation progress.
#[derive(Debug, Clone)]
pub struct ModelLifecycle {
    manifest: ModelManifest,
    state: ModelState,
    consent: DownloadConsent,
}

impl ModelLifecycle {
    /// Create lifecycle state for a manifest.
    #[must_use]
    pub const fn new(manifest: ModelManifest, consent: DownloadConsent) -> Self {
        let state = if consent.granted {
            ModelState::NotInstalled
        } else {
            ModelState::NeedsConsent
        };
        Self {
            manifest,
            state,
            consent,
        }
    }

    /// Current lifecycle state.
    #[must_use]
    pub const fn state(&self) -> &ModelState {
        &self.state
    }

    /// Underlying manifest for this lifecycle.
    #[must_use]
    pub const fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Mark consent as granted (e.g., after explicit user approval).
    pub fn approve_consent(&mut self, source: ConsentSource) {
        self.consent = DownloadConsent::granted(source);
        if matches!(self.state, ModelState::NeedsConsent) {
            self.state = ModelState::NotInstalled;
        }
    }

    /// Start the download state.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` on invalid transition or zero total bytes.
    pub fn begin_download(&mut self, total_bytes: u64) -> SearchResult<()> {
        if !self.consent.granted {
            self.state = ModelState::NeedsConsent;
            return Err(SearchError::EmbedderUnavailable {
                model: self.manifest.id.clone(),
                reason: "download consent required".to_owned(),
            });
        }
        if total_bytes == 0 {
            return Err(SearchError::InvalidConfig {
                field: "total_bytes".to_owned(),
                value: "0".to_owned(),
                reason: "must be greater than zero".to_owned(),
            });
        }

        match self.state {
            ModelState::NotInstalled
            | ModelState::Cancelled
            | ModelState::VerificationFailed { .. } => {
                self.state = ModelState::Downloading {
                    progress_pct: 0,
                    bytes_downloaded: 0,
                    total_bytes,
                };
                Ok(())
            }
            _ => Err(invalid_state_transition(
                &self.state,
                "begin_download",
                "expected NotInstalled/Cancelled/VerificationFailed",
            )),
        }
    }

    /// Update bytes downloaded and recompute bounded percent.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if not currently downloading.
    pub fn update_download_progress(&mut self, bytes_downloaded: u64) -> SearchResult<()> {
        let (progress_pct, total_bytes, bounded_bytes) = match self.state {
            ModelState::Downloading { total_bytes, .. } => {
                let bounded = bytes_downloaded.min(total_bytes);
                let pct_u64 = bounded.saturating_mul(100) / total_bytes;
                #[allow(clippy::cast_possible_truncation)]
                let pct = pct_u64 as u8;
                (pct.min(100), total_bytes, bounded)
            }
            _ => {
                return Err(invalid_state_transition(
                    &self.state,
                    "update_download_progress",
                    "expected Downloading",
                ));
            }
        };

        self.state = ModelState::Downloading {
            progress_pct,
            bytes_downloaded: bounded_bytes,
            total_bytes,
        };
        Ok(())
    }

    /// Move from downloading to verifying.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if not currently downloading.
    pub fn begin_verification(&mut self) -> SearchResult<()> {
        if matches!(self.state, ModelState::Downloading { .. }) {
            self.state = ModelState::Verifying;
            return Ok(());
        }
        Err(invalid_state_transition(
            &self.state,
            "begin_verification",
            "expected Downloading",
        ))
    }

    /// Mark the unpublished staging directory verified.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` unless verification is the active phase.
    pub fn mark_staged_verified(&mut self) -> SearchResult<()> {
        if matches!(self.state, ModelState::Verifying) {
            self.state = ModelState::StagedVerified;
            return Ok(());
        }
        Err(invalid_state_transition(
            &self.state,
            "mark_staged_verified",
            "expected Verifying",
        ))
    }

    /// Record a published, load-tested generation that still needs a compatible index.
    ///
    /// A verified warm cache may bypass consent and transport states. A newly
    /// acquired generation must arrive from `StagedVerified`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` while transport, verification, readiness, or a
    /// disabled/update state is active.
    pub fn mark_acquired_needs_reindex(&mut self) -> SearchResult<()> {
        if matches!(
            self.state,
            ModelState::NotInstalled
                | ModelState::NeedsConsent
                | ModelState::StagedVerified
                | ModelState::VerificationFailed { .. }
                | ModelState::Cancelled
        ) {
            self.state = ModelState::AcquiredNeedsReindex;
            return Ok(());
        }
        Err(invalid_state_transition(
            &self.state,
            "mark_acquired_needs_reindex",
            "expected a verified warm cache or StagedVerified generation",
        ))
    }

    /// Mark install ready.
    pub fn mark_ready(&mut self) {
        self.state = ModelState::Ready;
    }

    /// Mark install verification failed.
    pub fn fail_verification(&mut self, reason: impl Into<String>) {
        self.state = ModelState::VerificationFailed {
            reason: reason.into(),
        };
    }

    /// Mark model disabled.
    pub fn disable(&mut self, reason: impl Into<String>) {
        self.state = ModelState::Disabled {
            reason: reason.into(),
        };
    }

    /// Mark update available.
    pub fn mark_update_available(
        &mut self,
        current_revision: impl Into<String>,
        latest_revision: impl Into<String>,
    ) {
        self.state = ModelState::UpdateAvailable {
            current_revision: current_revision.into(),
            latest_revision: latest_revision.into(),
        };
    }

    /// Cancel current operation.
    pub fn cancel(&mut self) {
        self.state = ModelState::Cancelled;
    }

    /// Recover from cancelled state so a new download can start.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if current state is not `Cancelled`.
    pub fn recover_after_cancel(&mut self) -> SearchResult<()> {
        if !matches!(self.state, ModelState::Cancelled) {
            return Err(invalid_state_transition(
                &self.state,
                "recover_after_cancel",
                "expected Cancelled",
            ));
        }
        self.state = if self.consent.granted {
            ModelState::NotInstalled
        } else {
            ModelState::NeedsConsent
        };
        Ok(())
    }
}
