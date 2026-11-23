use std::collections::BTreeSet;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{BufReader, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::http::h1::ClientError;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::QuantizationFormat;
use sha2::{Digest, Sha256};

use crate::model_manifest::{
    FrozenModelArtifactManifestV1, ModelArtifactFileV1, ModelFile, ModelManifest,
};

use super::types::{
    DownloadProgress, MODEL_ACQUISITION_PROGRESS_SCHEMA_V1, MODEL_ACQUISITION_RECEIPT_SCHEMA_V1,
    ModelAcquisitionCacheReasonV1, ModelAcquisitionOutcomeV1, ModelAcquisitionProgressV1,
    ModelAcquisitionReceiptV1, ModelAcquisitionSourceKindV1, ModelAcquisitionStageV1,
    ModelAcquisitionVerificationResultV1,
};

static STAGING_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);
const LOCAL_COPY_BUFFER_BYTES: usize = 1024 * 1024;
const IN_MEMORY_F32_STORAGE_FORMAT_V1: &str = "in-memory-f32-v1";

// ─── Helpers ────────────────────────────────────────────────────────────────

pub(super) struct AcquisitionLock {
    _file: File,
}

impl AcquisitionLock {
    pub(super) fn acquire(destination_dir: &Path) -> SearchResult<Self> {
        let parent = destination_dir
            .parent()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "model_acquisition.destination".to_owned(),
                value: "redacted".to_owned(),
                reason: "destination must have a parent directory".to_owned(),
            })?;
        std::fs::create_dir_all(parent).map_err(SearchError::from)?;
        let destination_name = destination_dir
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "model_acquisition.destination".to_owned(),
                value: "redacted".to_owned(),
                reason: "destination must end in a valid UTF-8 directory name".to_owned(),
            })?;
        let lock_path = parent.join(format!(".{destination_name}.acquisition.lock"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .map_err(SearchError::from)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(SearchError::InvalidConfig {
                field: "model_acquisition.lock".to_owned(),
                value: "busy".to_owned(),
                reason: "another process is acquiring this model generation".to_owned(),
            }),
            Err(TryLockError::Error(error)) => Err(SearchError::from(error)),
        }
    }
}

pub(super) fn validate_lifecycle_manifest(
    frozen: &FrozenModelArtifactManifestV1,
    lifecycle_manifest: &ModelManifest,
) -> SearchResult<()> {
    let expected = legacy_manifest_from_frozen(frozen)?;
    let same_files = expected.files.len() == lifecycle_manifest.files.len()
        && expected.files.iter().all(|expected_file| {
            lifecycle_manifest.files.iter().any(|actual_file| {
                actual_file.name == expected_file.name
                    && actual_file.size == expected_file.size
                    && actual_file.sha256 == expected_file.sha256
                    && lifecycle_manifest.download_url(actual_file)
                        == expected_file.url.as_deref().unwrap_or_default()
            })
        });
    if lifecycle_manifest.id != expected.id
        || lifecycle_manifest.repo != expected.repo
        || lifecycle_manifest.revision != expected.revision
        || lifecycle_manifest.license != expected.license
        || lifecycle_manifest.dimension != expected.dimension
        || !same_files
    {
        return Err(SearchError::InvalidConfig {
            field: "model_acquisition.lifecycle_manifest".to_owned(),
            value: lifecycle_manifest.id.clone(),
            reason: "lifecycle manifest must exactly bind the requested frozen artifact set"
                .to_owned(),
        });
    }
    Ok(())
}

pub(super) fn legacy_manifest_from_frozen(
    frozen: &FrozenModelArtifactManifestV1,
) -> SearchResult<ModelManifest> {
    frozen.validate()?;
    let total_bytes = total_frozen_bytes(frozen)?;
    Ok(ModelManifest {
        id: frozen.manifest.logical_model_id.clone(),
        version: frozen.manifest.upstream_revision.clone(),
        display_name: None,
        description: None,
        repo: frozen.manifest.upstream_repository.clone(),
        revision: frozen.manifest.upstream_revision.clone(),
        files: frozen
            .manifest
            .artifacts
            .iter()
            .map(|artifact| ModelFile {
                name: artifact.relative_path.clone(),
                sha256: artifact.sha256.clone(),
                size: artifact.size,
                url: Some(artifact.upstream_url.clone()),
            })
            .collect(),
        license: frozen.manifest.license_spdx.clone(),
        dimension: Some(frozen.manifest.dimension),
        tier: None,
        download_size_bytes: total_bytes,
    })
}

pub(super) fn total_frozen_bytes(frozen: &FrozenModelArtifactManifestV1) -> SearchResult<u64> {
    frozen
        .manifest
        .artifacts
        .iter()
        .try_fold(0_u64, |total, artifact| {
            total
                .checked_add(artifact.size)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "artifacts[].size".to_owned(),
                    value: "overflow".to_owned(),
                    reason: "total registered artifact bytes must fit in u64".to_owned(),
                })
        })
}

pub(super) fn check_acquisition_cancel(cx: &Cx, phase: &str) -> SearchResult<()> {
    if cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: phase.to_owned(),
            reason: "structured cancellation requested".to_owned(),
        });
    }
    Ok(())
}

pub(super) fn frozen_source_hosts(
    frozen: &FrozenModelArtifactManifestV1,
) -> SearchResult<Vec<String>> {
    let mut hosts = BTreeSet::new();
    for artifact in &frozen.manifest.artifacts {
        let authority = artifact
            .upstream_url
            .strip_prefix("https://")
            .and_then(|remainder| remainder.split('/').next())
            .filter(|authority| !authority.is_empty())
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "artifacts[].upstream_url".to_owned(),
                value: "redacted".to_owned(),
                reason: "expected a credential-free HTTPS authority".to_owned(),
            })?;
        let host = authority_host(authority)?;
        hosts.insert(host.to_owned());
    }
    Ok(hosts.into_iter().collect())
}

fn authority_host(authority: &str) -> SearchResult<&str> {
    let host = if let Some(rest) = authority.strip_prefix('[') {
        let closing = rest.find(']').ok_or_else(|| SearchError::InvalidConfig {
            field: "artifacts[].upstream_url".to_owned(),
            value: "redacted".to_owned(),
            reason: "malformed bracketed HTTPS host".to_owned(),
        })?;
        &rest[..closing]
    } else {
        authority.split(':').next().unwrap_or_default()
    };
    if host.is_empty()
        || !host.is_ascii()
        || host
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b':')))
    {
        return Err(SearchError::InvalidConfig {
            field: "artifacts[].upstream_url".to_owned(),
            value: "redacted".to_owned(),
            reason: "HTTPS source host contains invalid characters".to_owned(),
        });
    }
    Ok(host)
}

pub(super) fn diagnostic_source_host(url: &str) -> &str {
    url.split_once("://")
        .and_then(|(_, remainder)| remainder.split('/').next())
        .and_then(|authority| authority_host(authority).ok())
        .unwrap_or("<invalid-host>")
}

pub(super) fn acquisition_receipt(
    frozen: &FrozenModelArtifactManifestV1,
    source: ModelAcquisitionSourceKindV1,
    source_hosts: Vec<String>,
    outcome: ModelAcquisitionOutcomeV1,
    bytes_verified: u64,
    duration: Duration,
    prior_generation_preserved: bool,
) -> SearchResult<ModelAcquisitionReceiptV1> {
    let declared_bytes = total_frozen_bytes(frozen)?;
    if bytes_verified != declared_bytes {
        return Err(SearchError::InvalidConfig {
            field: "model_acquisition.bytes_verified".to_owned(),
            value: bytes_verified.to_string(),
            reason: "verified byte count must equal the frozen artifact total".to_owned(),
        });
    }
    let identity = frozen
        .manifest
        .declared_identity_bundle(QuantizationFormat::F32, IN_MEMORY_F32_STORAGE_FORMAT_V1)?;
    Ok(ModelAcquisitionReceiptV1 {
        schema_version: MODEL_ACQUISITION_RECEIPT_SCHEMA_V1,
        logical_model_id: frozen.manifest.logical_model_id.clone(),
        upstream_revision: frozen.manifest.upstream_revision.clone(),
        license_spdx: frozen.manifest.license_spdx.clone(),
        artifact_manifest_fingerprint: frozen.fingerprint.clone(),
        space_fingerprint: identity.space.fingerprint(),
        producer_fingerprint: identity.producer.fingerprint(),
        source,
        source_hosts,
        outcome,
        bytes_verified,
        duration_millis: saturating_duration_millis(duration),
        prior_generation_preserved,
        requires_reindex: true,
    })
}

pub(super) fn acquisition_transport_progress(
    frozen: &FrozenModelArtifactManifestV1,
    source: ModelAcquisitionSourceKindV1,
    progress: &DownloadProgress,
    duration: Duration,
    cache_reason: Option<ModelAcquisitionCacheReasonV1>,
) -> SearchResult<ModelAcquisitionProgressV1> {
    let artifact = frozen
        .manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.relative_path == progress.file_name)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "model_acquisition.progress_artifact".to_owned(),
            value: "unregistered".to_owned(),
            reason: "transport progress must reference a registered frozen artifact".to_owned(),
        })?;
    let source_host = if source == ModelAcquisitionSourceKindV1::Network {
        Some(
            artifact
                .upstream_url
                .strip_prefix("https://")
                .and_then(|remainder| remainder.split('/').next())
                .and_then(|authority| authority_host(authority).ok())
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "artifacts[].upstream_url".to_owned(),
                    value: "redacted".to_owned(),
                    reason: "network progress requires a valid credential-free HTTPS host"
                        .to_owned(),
                })?
                .to_owned(),
        )
    } else {
        None
    };
    Ok(ModelAcquisitionProgressV1 {
        schema_version: MODEL_ACQUISITION_PROGRESS_SCHEMA_V1,
        logical_model_id: frozen.manifest.logical_model_id.clone(),
        upstream_revision: frozen.manifest.upstream_revision.clone(),
        artifact_manifest_fingerprint: frozen.fingerprint.clone(),
        source,
        source_host,
        artifact_role: Some(artifact.role),
        artifact_sha256: Some(artifact.sha256.clone()),
        stage: ModelAcquisitionStageV1::Transporting,
        bytes_processed: progress.bytes_downloaded.min(artifact.size),
        total_bytes: artifact.size,
        files_completed: progress.files_completed.min(progress.files_total),
        files_total: frozen.manifest.artifacts.len(),
        duration_millis: saturating_duration_millis(duration),
        cache_reason,
        verification_result: ModelAcquisitionVerificationResultV1::Pending,
    })
}

pub(super) fn acquisition_phase_progress(
    frozen: &FrozenModelArtifactManifestV1,
    source: ModelAcquisitionSourceKindV1,
    stage: ModelAcquisitionStageV1,
    duration: Duration,
    cache_reason: Option<ModelAcquisitionCacheReasonV1>,
    verification_result: ModelAcquisitionVerificationResultV1,
) -> SearchResult<ModelAcquisitionProgressV1> {
    let total_bytes = total_frozen_bytes(frozen)?;
    let complete = matches!(
        stage,
        ModelAcquisitionStageV1::StagedVerified
            | ModelAcquisitionStageV1::LoadSelfTestPassed
            | ModelAcquisitionStageV1::Published
            | ModelAcquisitionStageV1::WarmCacheVerified
    );
    Ok(ModelAcquisitionProgressV1 {
        schema_version: MODEL_ACQUISITION_PROGRESS_SCHEMA_V1,
        logical_model_id: frozen.manifest.logical_model_id.clone(),
        upstream_revision: frozen.manifest.upstream_revision.clone(),
        artifact_manifest_fingerprint: frozen.fingerprint.clone(),
        source,
        source_host: None,
        artifact_role: None,
        artifact_sha256: None,
        stage,
        bytes_processed: if complete { total_bytes } else { 0 },
        total_bytes,
        files_completed: if complete {
            frozen.manifest.artifacts.len()
        } else {
            0
        },
        files_total: frozen.manifest.artifacts.len(),
        duration_millis: saturating_duration_millis(duration),
        cache_reason,
        verification_result,
    })
}

fn saturating_duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
thread_local! {
    static TEST_LOCAL_COPY_ERROR: std::cell::Cell<Option<ErrorKind>> =
        const { std::cell::Cell::new(None) };
}

#[allow(clippy::unnecessary_wraps)] // Test builds inject typed filesystem failures here.
fn local_copy_write_boundary() -> SearchResult<()> {
    #[cfg(test)]
    if let Some(kind) = TEST_LOCAL_COPY_ERROR.with(std::cell::Cell::get) {
        return Err(SearchError::from(std::io::Error::from(kind)));
    }
    Ok(())
}

#[cfg(test)]
pub(super) struct LocalCopyErrorGuard;

#[cfg(test)]
impl LocalCopyErrorGuard {
    pub(super) fn install(kind: ErrorKind) -> Self {
        TEST_LOCAL_COPY_ERROR.with(|fault| {
            assert!(
                fault.replace(Some(kind)).is_none(),
                "local-copy test error already installed"
            );
        });
        Self
    }
}

#[cfg(test)]
impl Drop for LocalCopyErrorGuard {
    fn drop(&mut self) {
        TEST_LOCAL_COPY_ERROR.with(|fault| fault.set(None));
    }
}

pub(super) fn copy_local_artifact(
    cx: &Cx,
    source_dir: &Path,
    staging_dir: &Path,
    artifact: &ModelArtifactFileV1,
    file_idx: usize,
    files_total: usize,
    on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
) -> SearchResult<()> {
    let source_path = source_dir.join(&artifact.relative_path);
    let metadata = std::fs::symlink_metadata(&source_path).map_err(SearchError::from)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(SearchError::InvalidConfig {
            field: "model_acquisition.local_artifact".to_owned(),
            value: "redacted".to_owned(),
            reason: "registered local artifacts must be regular files, never symlinks".to_owned(),
        });
    }

    let destination = staging_dir.join(&artifact.relative_path);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(SearchError::from)?;
    }
    let source = File::open(source_path).map_err(SearchError::from)?;
    let mut reader = BufReader::with_capacity(LOCAL_COPY_BUFFER_BYTES, source);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)
        .map_err(SearchError::from)?;
    let mut buffer = vec![0_u8; LOCAL_COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut bytes_copied = 0_u64;
    let started = Instant::now();
    on_progress(&DownloadProgress {
        file_name: artifact.relative_path.clone(),
        bytes_downloaded: 0,
        total_bytes: Some(artifact.size),
        files_completed: file_idx,
        files_total,
        speed_bytes_per_sec: 0.0,
        eta_seconds: None,
    });

    loop {
        check_acquisition_cancel(cx, "local-artifact-read")?;
        let count = reader.read(&mut buffer).map_err(SearchError::from)?;
        if count == 0 {
            break;
        }
        let count_u64 = u64::try_from(count).unwrap_or(u64::MAX);
        let next_size =
            bytes_copied
                .checked_add(count_u64)
                .ok_or_else(|| SearchError::HashMismatch {
                    path: PathBuf::from("<redacted-local-artifact>"),
                    expected: format!("size={}", artifact.size),
                    actual: "size-overflow".to_owned(),
                })?;
        if next_size > artifact.size {
            return Err(SearchError::HashMismatch {
                path: PathBuf::from("<redacted-local-artifact>"),
                expected: format!("size={}", artifact.size),
                actual: format!("size-at-least={next_size}"),
            });
        }
        local_copy_write_boundary()?;
        output
            .write_all(&buffer[..count])
            .map_err(SearchError::from)?;
        hasher.update(&buffer[..count]);
        bytes_copied = next_size;
        report_local_progress(
            artifact,
            file_idx,
            files_total,
            bytes_copied,
            started.elapsed(),
            on_progress,
        );
    }

    if bytes_copied != artifact.size {
        return Err(SearchError::HashMismatch {
            path: PathBuf::from("<redacted-local-artifact>"),
            expected: format!("size={}", artifact.size),
            actual: format!("size={bytes_copied}"),
        });
    }
    let actual_hash = sha256_digest_hex(hasher.finalize().as_slice());
    // ubs:ignore — pinned artifact digests are public integrity metadata.
    if actual_hash != artifact.sha256 {
        return Err(SearchError::HashMismatch {
            path: PathBuf::from("<redacted-local-artifact>"),
            expected: format!("sha256={},size={}", artifact.sha256, artifact.size),
            actual: format!("sha256={actual_hash},size={bytes_copied}"),
        });
    }
    check_acquisition_cancel(cx, "local-artifact-sync")?;
    output.flush().map_err(SearchError::from)?;
    output.sync_all().map_err(SearchError::from)?;
    report_local_progress(
        artifact,
        file_idx,
        files_total,
        bytes_copied,
        started.elapsed(),
        on_progress,
    );
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn report_local_progress(
    artifact: &ModelArtifactFileV1,
    file_idx: usize,
    files_total: usize,
    bytes_copied: u64,
    elapsed: Duration,
    on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
) {
    let speed = if elapsed.as_secs_f64() > 0.0 {
        bytes_copied as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    let eta_seconds = if speed <= f64::EPSILON || bytes_copied >= artifact.size {
        None
    } else {
        Some((artifact.size.saturating_sub(bytes_copied)) as f64 / speed)
    };
    on_progress(&DownloadProgress {
        file_name: artifact.relative_path.clone(),
        bytes_downloaded: bytes_copied,
        total_bytes: Some(artifact.size),
        files_completed: file_idx,
        files_total,
        speed_bytes_per_sec: speed,
        eta_seconds,
    });
}

/// Build a `HuggingFace` CDN URL for a model file.
#[cfg(test)]
pub(super) fn huggingface_url(repo: &str, revision: &str, file_name: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file_name}")
}

pub(super) fn create_unique_staging_dir(dest_dir: &Path) -> SearchResult<PathBuf> {
    // Prefer creating staging dir as a sibling to avoid dirtying the target dir.
    let (base_dir, prefix) = dest_dir.parent().map_or((dest_dir, "download"), |parent| {
        (
            parent,
            dest_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model"),
        )
    });

    std::fs::create_dir_all(base_dir).map_err(SearchError::from)?;

    let pid = std::process::id();
    for _ in 0..64 {
        let counter = STAGING_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = base_dir.join(format!(".{prefix}-download-{pid}-{counter:016x}"));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(err) if err.kind() == ErrorKind::AlreadyExists => {}
            Err(err) => return Err(SearchError::from(err)),
        }
    }

    Err(SearchError::ModelLoadFailed {
        path: PathBuf::from("<redacted-model-cache>"),
        source: "failed to allocate unique staging directory".into(),
    })
}

pub(super) fn response_content_length(headers: &[(String, String)]) -> Option<u64> {
    headers.iter().find_map(|(name, value)| {
        name.eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse::<u64>().ok())
            .flatten()
    })
}

pub(super) fn sha256_digest_hex(data: &[u8]) -> String {
    let mut out = String::with_capacity(64);
    for byte in data {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub(super) fn bounded_download_failure_reason(error: &SearchError) -> &'static str {
    match error {
        SearchError::HashMismatch { .. } => "artifact-integrity-mismatch",
        SearchError::ModelNotFound { .. } => "registered-artifact-missing",
        SearchError::ModelLoadFailed { .. } | SearchError::Io(_) => {
            "transport-or-filesystem-failure"
        }
        SearchError::Cancelled { .. } => "operation-cancelled",
        SearchError::InvalidConfig { .. } => "invalid-download-contract",
        _ => "model-download-failed",
    }
}

/// Compute lowercase hex SHA-256 of a byte slice.
#[cfg(test)]
pub(super) fn sha256_hex(data: &[u8]) -> String {
    sha256_digest_hex(Sha256::digest(data).as_slice())
}

pub(super) struct TempFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TempFileGuard {
    pub(super) const fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    pub(super) const fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.armed
            && let Err(e) = std::fs::remove_file(&self.path)
        {
            tracing::warn!(
                error_kind = ?e.kind(),
                "failed to clean up model download temp file"
            );
        }
    }
}

/// Format bytes as a human-readable string.
#[allow(clippy::cast_precision_loss)]
pub(super) fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Convert asupersync `ClientError` to `SearchError`.
pub(super) fn client_error_to_search(error: ClientError, url: &str) -> SearchError {
    if matches!(error, ClientError::Cancelled) {
        return SearchError::Cancelled {
            phase: "model-download-request".to_owned(),
            reason: "HTTP client observed structured cancellation".to_owned(),
        };
    }
    let detail = match error {
        ClientError::InvalidUrl(_) => "invalid-url".to_owned(),
        ClientError::DnsError(error)
        | ClientError::ConnectError(error)
        | ClientError::Io(error) => format!("io-{:?}", error.kind()),
        ClientError::TlsError(_) => "tls-handshake-failure".to_owned(),
        ClientError::HttpError(_) => "http-protocol-failure".to_owned(),
        ClientError::TooManyRedirects { count, max } => {
            format!("redirect-limit-{count}-of-{max}")
        }
        ClientError::DeadlineExceeded => "request-deadline-exceeded".to_owned(),
        ClientError::ConnectTunnelRefused { status, .. } => {
            format!("proxy-tunnel-http-{status}")
        }
        ClientError::InvalidConnectInput(_) => "invalid-proxy-connect-input".to_owned(),
        ClientError::ProxyError(_) => "proxy-failure".to_owned(),
        ClientError::PoolExhausted { port, .. } => {
            format!("connection-pool-exhausted-port-{port}")
        }
        ClientError::Cancelled => unreachable!("handled above"),
    };
    SearchError::ModelLoadFailed {
        path: PathBuf::from("<redacted-model-source>"),
        source: format!(
            "transport failure via host {} ({detail})",
            diagnostic_source_host(url)
        )
        .into(),
    }
}

pub(super) fn bounded_http_body_error(_error: &impl std::error::Error) -> &'static str {
    "http-body-failure"
}
