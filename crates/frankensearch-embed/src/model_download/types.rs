use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use frankensearch_core::error::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use crate::model_manifest::{FrozenModelArtifactManifestV1, ModelArtifactRoleV1};

use super::helpers::format_bytes;

/// Schema version for durable acquisition receipts.
pub const MODEL_ACQUISITION_RECEIPT_SCHEMA_V1: u16 = 1;
/// Schema version for path-free acquisition progress records.
pub const MODEL_ACQUISITION_PROGRESS_SCHEMA_V1: u16 = 1;
/// Schema version for path-free acquisition recovery diagnostics.
pub const MODEL_ACQUISITION_RECOVERY_SCHEMA_V1: u16 = 1;

/// Default per-artifact response body cap for model downloads.
///
/// The underlying HTTP codec defaults to a 16 MiB body limit, which is smaller
/// than several production-ready model artifacts in the built-in catalog (e.g.
/// the default Potion tokenizer is ~17.8 MiB and its model weights ~489 MiB).
/// Files are streamed to disk with bounded memory and verified against the
/// manifest's declared size and SHA-256, so this cap serves only as a finite
/// resource guard. 2 GiB comfortably covers the current catalog while keeping a
/// bounded ceiling; callers may override it via [`DownloadConfig`].
pub const DEFAULT_MAX_MODEL_ARTIFACT_BYTES: usize = 2 * 1024 * 1024 * 1024;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for model downloads.
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Maximum retries per file on transient failure.
    pub max_retries: u32,
    /// Base delay for exponential backoff between retries.
    pub retry_base_delay: Duration,
    /// User-Agent header value.
    pub user_agent: String,
    /// Maximum redirects to follow.
    pub max_redirects: u32,
    /// Maximum size, in bytes, of a single downloaded model artifact.
    ///
    /// Forwarded to the HTTP client's body-size limit. The codec default of
    /// 16 MiB is too small for real model artifacts, so this defaults to
    /// [`DEFAULT_MAX_MODEL_ARTIFACT_BYTES`]. Downloads are streamed to disk and
    /// verified against the manifest's declared size and SHA-256, so this bound
    /// only guards against an unexpectedly large response.
    pub max_response_bytes: usize,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_secs(1),
            user_agent: format!("frankensearch/{}", env!("CARGO_PKG_VERSION")),
            max_redirects: 5,
            max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
        }
    }
}

// ─── Progress ───────────────────────────────────────────────────────────────

/// Progress information for an in-flight model download.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Name of the file currently being downloaded.
    pub file_name: String,
    /// Bytes downloaded so far (current file).
    pub bytes_downloaded: u64,
    /// Total bytes expected (current file), if known.
    pub total_bytes: Option<u64>,
    /// Number of files completed so far.
    pub files_completed: usize,
    /// Total number of files to download.
    pub files_total: usize,
    /// Estimated download speed in bytes per second.
    pub speed_bytes_per_sec: f64,
    /// Estimated time remaining in seconds, if calculable.
    pub eta_seconds: Option<f64>,
}

/// Authorized byte source for one explicit acquisition.
pub enum ModelAcquisitionSource<'a> {
    /// Download the immutable URLs carried by the frozen manifest.
    Network,
    /// Copy the complete artifact tree rooted at this local directory.
    LocalBundle(&'a Path),
}

/// Inputs for one explicit frozen-model acquisition transaction.
pub struct ModelAcquisitionRequest<'a> {
    /// Frozen manifest that authorizes every accepted byte.
    pub frozen_manifest: &'a FrozenModelArtifactManifestV1,
    /// Explicit transport selected by the caller.
    pub source: ModelAcquisitionSource<'a>,
    /// Final generation directory.
    pub destination_dir: &'a Path,
}

impl fmt::Debug for ModelAcquisitionRequest<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelAcquisitionRequest")
            .field("manifest_fingerprint", &self.frozen_manifest.fingerprint)
            .field("source", &self.source)
            .field("destination_dir", &"<redacted-path>")
            .finish()
    }
}

impl fmt::Debug for ModelAcquisitionSource<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network => formatter.write_str("Network"),
            Self::LocalBundle(_) => formatter.write_str("LocalBundle(<redacted-path>)"),
        }
    }
}

/// Content source recorded in the path-free durable receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionSourceKindV1 {
    /// Immutable HTTPS artifacts from the manifest.
    Network,
    /// Explicit operator-supplied local bundle.
    LocalBundle,
    /// Existing destination was already complete and verified.
    WarmCache,
}

/// Publication result recorded in the durable receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionOutcomeV1 {
    /// A newly verified generation was published.
    Published,
    /// The existing generation was reverified and reused without transport.
    VerifiedWarmCache,
}

/// Why acquisition did or did not reuse the destination cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionCacheReasonV1 {
    /// No destination generation was present.
    DestinationMissing,
    /// A destination existed but failed frozen-manifest verification.
    DestinationRejected,
    /// The complete destination generation verified and was reused.
    FrozenGenerationVerified,
}

/// Current phase represented by an acquisition progress record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionStageV1 {
    /// Registered artifact bytes are being streamed.
    Transporting,
    /// The complete staged artifact set passed size and SHA-256 verification.
    StagedVerified,
    /// The verified artifact set passed the caller's load self-test.
    LoadSelfTestPassed,
    /// The verified generation was durably published.
    Published,
    /// The existing destination passed verification and load self-test.
    WarmCacheVerified,
    /// Acquisition failed without declaring semantic readiness.
    Failed,
    /// Structured cancellation was observed.
    Cancelled,
}

/// Verification result represented by an acquisition progress record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAcquisitionVerificationResultV1 {
    /// Streaming is still in progress.
    Pending,
    /// The phase's verification boundary passed.
    Passed,
    /// The transaction failed.
    Failed,
    /// The transaction observed structured cancellation.
    Cancelled,
}

/// Bounded, path-free progress for one frozen-model acquisition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAcquisitionProgressV1 {
    /// Progress schema version.
    pub schema_version: u16,
    /// Logical semantic model ID.
    pub logical_model_id: String,
    /// Immutable upstream revision.
    pub upstream_revision: String,
    /// Full frozen-manifest fingerprint.
    pub artifact_manifest_fingerprint: String,
    /// Actual or requested transport.
    pub source: ModelAcquisitionSourceKindV1,
    /// Credential-free source host for the current network artifact.
    pub source_host: Option<String>,
    /// Semantic role of the current artifact, if transport is active.
    pub artifact_role: Option<ModelArtifactRoleV1>,
    /// Exact registered SHA-256 for the current artifact.
    pub artifact_sha256: Option<String>,
    /// Transaction phase.
    pub stage: ModelAcquisitionStageV1,
    /// Bounded bytes processed for the current artifact or complete set.
    pub bytes_processed: u64,
    /// Exact declared bytes for the current artifact or complete set.
    pub total_bytes: u64,
    /// Completed artifact count.
    pub files_completed: usize,
    /// Registered artifact count.
    pub files_total: usize,
    /// Saturating duration since transaction start.
    pub duration_millis: u64,
    /// Cache disposition, once known.
    pub cache_reason: Option<ModelAcquisitionCacheReasonV1>,
    /// Verification outcome for this phase.
    pub verification_result: ModelAcquisitionVerificationResultV1,
}

/// Path-free inventory of retained acquisition recovery artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAcquisitionRecoveryV1 {
    /// Recovery diagnostic schema version.
    pub schema_version: u16,
    /// Whether the requested destination currently exists.
    pub destination_present: bool,
    /// Unique streaming stages retained after incomplete attempts.
    pub orphan_staging_generations: usize,
    /// Verified stages renamed into the publication lane but not committed.
    pub interrupted_installing_generations: usize,
    /// Prior generations preserved by replacement attempts.
    pub preserved_backup_generations: usize,
    /// Whether the persistent advisory lock file has been initialized.
    pub advisory_lock_present: bool,
}

/// Diagnose retained acquisition state without deleting or opening artifacts.
///
/// The result contains counts only, never paths or content. Garbage collection
/// is intentionally not implicit; a separate operator action must supply an
/// explicit retention policy.
///
/// # Errors
///
/// Returns a typed I/O or configuration error if the destination parent cannot
/// be inspected safely.
pub fn diagnose_model_acquisition(
    destination_dir: &Path,
) -> SearchResult<ModelAcquisitionRecoveryV1> {
    let parent = destination_dir
        .parent()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "model_acquisition.destination".to_owned(),
            value: "redacted".to_owned(),
            reason: "destination must have a parent directory".to_owned(),
        })?;
    let destination_name = destination_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "model_acquisition.destination".to_owned(),
            value: "redacted".to_owned(),
            reason: "destination must end in a valid UTF-8 directory name".to_owned(),
        })?;
    let staging_prefix = format!(".{destination_name}-download-");
    let installing_prefix = format!(".{destination_name}.installing.");
    let backup_prefix = format!("{destination_name}.backup.");
    let lock_name = format!(".{destination_name}.acquisition.lock");
    let mut recovery = ModelAcquisitionRecoveryV1 {
        schema_version: MODEL_ACQUISITION_RECOVERY_SCHEMA_V1,
        destination_present: destination_dir.exists(),
        orphan_staging_generations: 0,
        interrupted_installing_generations: 0,
        preserved_backup_generations: 0,
        advisory_lock_present: false,
    };
    if !parent.exists() {
        return Ok(recovery);
    }
    for entry in std::fs::read_dir(parent).map_err(SearchError::from)? {
        let entry = entry.map_err(SearchError::from)?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with(&staging_prefix) {
            recovery.orphan_staging_generations =
                recovery.orphan_staging_generations.saturating_add(1);
        } else if name.starts_with(&installing_prefix) {
            recovery.interrupted_installing_generations = recovery
                .interrupted_installing_generations
                .saturating_add(1);
        } else if name.starts_with(&backup_prefix) {
            recovery.preserved_backup_generations =
                recovery.preserved_backup_generations.saturating_add(1);
        } else if name == lock_name {
            recovery.advisory_lock_present = true;
        }
    }
    Ok(recovery)
}

/// Path-free evidence that a frozen model generation was acquired.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAcquisitionReceiptV1 {
    /// Receipt schema version.
    pub schema_version: u16,
    /// Logical semantic model ID.
    pub logical_model_id: String,
    /// Immutable upstream revision.
    pub upstream_revision: String,
    /// Pinned SPDX license assertion.
    pub license_spdx: String,
    /// Fingerprint of the full frozen artifact manifest.
    pub artifact_manifest_fingerprint: String,
    /// Mathematical embedding-space fingerprint.
    pub space_fingerprint: String,
    /// Local implementation/producer fingerprint.
    pub producer_fingerprint: String,
    /// Actual acquisition source.
    pub source: ModelAcquisitionSourceKindV1,
    /// Credential-free source hosts; empty for local/warm sources.
    pub source_hosts: Vec<String>,
    /// Publication or warm-cache outcome.
    pub outcome: ModelAcquisitionOutcomeV1,
    /// Exact registered artifact bytes verified.
    pub bytes_verified: u64,
    /// Bounded operation duration.
    pub duration_millis: u64,
    /// Whether an earlier destination generation was preserved as a backup.
    pub prior_generation_preserved: bool,
    /// Acquisition never implies compatibility with an existing index.
    pub requires_reindex: bool,
}

/// Verified but unpublished staging evidence.
pub struct VerifiedModelStageV1 {
    pub(super) staging_dir: PathBuf,
    pub(super) manifest_fingerprint: String,
    pub(super) source: ModelAcquisitionSourceKindV1,
    pub(super) source_hosts: Vec<String>,
    pub(super) bytes_verified: u64,
}

impl VerifiedModelStageV1 {
    /// Authorized staging directory used by the load self-test and publisher.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.staging_dir
    }
}

impl fmt::Debug for VerifiedModelStageV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedModelStageV1")
            .field("staging_dir", &"<redacted-path>")
            .field("manifest_fingerprint", &self.manifest_fingerprint)
            .field("source", &self.source)
            .field("source_hosts", &self.source_hosts)
            .field("bytes_verified", &self.bytes_verified)
            .finish()
    }
}

impl fmt::Display for DownloadProgress {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = self
            .total_bytes
            .filter(|&t| t > 0)
            .map(|t| self.bytes_downloaded as f64 / t as f64 * 100.0);

        if let Some(pct) = pct {
            write!(
                f,
                "[{}/{}] {} {:.0}% ({}/{})",
                self.files_completed + 1,
                self.files_total,
                self.file_name,
                pct,
                format_bytes(self.bytes_downloaded),
                format_bytes(self.total_bytes.unwrap_or(0)),
            )
        } else {
            write!(
                f,
                "[{}/{}] {} {}",
                self.files_completed + 1,
                self.files_total,
                self.file_name,
                format_bytes(self.bytes_downloaded),
            )
        }
    }
}
