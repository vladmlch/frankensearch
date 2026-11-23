use std::fs::OpenOptions;
use std::future::poll_fn;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Instant;

use asupersync::Cx;
use asupersync::bytes::Buf;
use asupersync::http::body::{Body, Frame};
use asupersync::http::h1::{HttpClient, HttpClientConfig, Method, RedirectPolicy};
use frankensearch_core::error::{SearchError, SearchResult};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::model_manifest::{
    FrozenModelArtifactManifestV1, ModelFile, ModelLifecycle, ModelManifest,
};

use super::helpers::{
    AcquisitionLock, TempFileGuard, acquisition_phase_progress, acquisition_receipt,
    acquisition_transport_progress, bounded_download_failure_reason, bounded_http_body_error,
    check_acquisition_cancel, client_error_to_search, copy_local_artifact,
    create_unique_staging_dir, diagnostic_source_host, frozen_source_hosts,
    legacy_manifest_from_frozen, response_content_length, sha256_digest_hex, total_frozen_bytes,
    validate_lifecycle_manifest,
};
use super::types::{
    DownloadConfig, DownloadProgress, ModelAcquisitionCacheReasonV1, ModelAcquisitionOutcomeV1,
    ModelAcquisitionProgressV1, ModelAcquisitionReceiptV1, ModelAcquisitionRequest,
    ModelAcquisitionSource, ModelAcquisitionSourceKindV1, ModelAcquisitionStageV1,
    ModelAcquisitionVerificationResultV1, VerifiedModelStageV1,
};

// ─── Downloader ─────────────────────────────────────────────────────────────

/// Downloads model files from `HuggingFace` with verification and progress reporting.
pub struct ModelDownloader {
    config: DownloadConfig,
    client: HttpClient,
}

impl ModelDownloader {
    /// Create a new downloader with the given configuration.
    #[must_use]
    pub fn new(config: DownloadConfig) -> Self {
        let mut client_config = HttpClientConfig::default();
        client_config.redirect_policy = RedirectPolicy::Limited(config.max_redirects);
        client_config.user_agent = Some(config.user_agent.clone());
        // Lift the codec's 16 MiB default body cap; model artifacts routinely
        // exceed it. Streaming-to-disk keeps memory bounded regardless.
        client_config.max_body_size = Some(config.max_response_bytes);
        Self {
            config,
            client: HttpClient::with_config(client_config),
        }
    }

    /// Create a downloader with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DownloadConfig::default())
    }

    /// Acquire, verify, load-test, and publish one frozen model generation.
    ///
    /// The destination is protected by a persistent sibling lock file. A valid
    /// warm generation is load-tested and returned without consulting the
    /// requested transport. New bytes are streamed into a unique sibling stage,
    /// verified against the frozen manifest, load-tested, synced, and published.
    /// An earlier destination is preserved as a uniquely named backup.
    ///
    /// Acquisition deliberately ends in
    /// [`crate::model_manifest::ModelState::AcquiredNeedsReindex`]. Selecting or
    /// rebuilding a compatible vector index is a separate readiness decision.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid manifests, lock contention,
    /// cancellation, transport failures, filesystem failures, integrity drift,
    /// load-test failures, or publication failures.
    pub async fn acquire_frozen_model(
        &self,
        cx: &Cx,
        request: ModelAcquisitionRequest<'_>,
        lifecycle: &mut ModelLifecycle,
        on_progress: impl Fn(&ModelAcquisitionProgressV1) + Send + Sync,
        load_self_test: impl Fn(&Path, &FrozenModelArtifactManifestV1) -> SearchResult<()> + Send + Sync,
    ) -> SearchResult<ModelAcquisitionReceiptV1> {
        let started = Instant::now();
        let ModelAcquisitionRequest {
            frozen_manifest,
            source,
            destination_dir,
        } = request;
        let requested_source = match &source {
            ModelAcquisitionSource::Network => ModelAcquisitionSourceKindV1::Network,
            ModelAcquisitionSource::LocalBundle(_) => ModelAcquisitionSourceKindV1::LocalBundle,
        };
        let mut observed_cache_reason = None;
        let operation = async {
            frozen_manifest.validate()?;
            validate_lifecycle_manifest(frozen_manifest, lifecycle.manifest())?;
            check_acquisition_cancel(cx, "acquisition-start")?;

            let _lock = AcquisitionLock::acquire(destination_dir)?;
            check_acquisition_cancel(cx, "acquisition-lock")?;

            let cache_reason = if destination_dir.is_dir()
                && frozen_manifest.manifest.verify_dir(destination_dir).is_ok()
            {
                ModelAcquisitionCacheReasonV1::FrozenGenerationVerified
            } else if destination_dir.exists() {
                ModelAcquisitionCacheReasonV1::DestinationRejected
            } else {
                ModelAcquisitionCacheReasonV1::DestinationMissing
            };
            observed_cache_reason = Some(cache_reason);

            if cache_reason == ModelAcquisitionCacheReasonV1::FrozenGenerationVerified {
                check_acquisition_cancel(cx, "warm-cache-self-test")?;
                load_self_test(destination_dir, frozen_manifest)?;
                check_acquisition_cancel(cx, "warm-cache-complete")?;
                lifecycle.mark_acquired_needs_reindex()?;
                on_progress(&acquisition_phase_progress(
                    frozen_manifest,
                    ModelAcquisitionSourceKindV1::WarmCache,
                    ModelAcquisitionStageV1::WarmCacheVerified,
                    started.elapsed(),
                    Some(cache_reason),
                    ModelAcquisitionVerificationResultV1::Passed,
                )?);
                return acquisition_receipt(
                    frozen_manifest,
                    ModelAcquisitionSourceKindV1::WarmCache,
                    Vec::new(),
                    ModelAcquisitionOutcomeV1::VerifiedWarmCache,
                    total_frozen_bytes(frozen_manifest)?,
                    started.elapsed(),
                    false,
                );
            }

            let stage = match source {
                ModelAcquisitionSource::Network => {
                    let download_manifest = legacy_manifest_from_frozen(frozen_manifest)?;
                    let forward_progress = |progress: &DownloadProgress| {
                        if let Ok(progress) = acquisition_transport_progress(
                            frozen_manifest,
                            ModelAcquisitionSourceKindV1::Network,
                            progress,
                            started.elapsed(),
                            Some(cache_reason),
                        ) {
                            on_progress(&progress);
                        }
                    };
                    let staging_dir = self
                        .download_model(
                            cx,
                            &download_manifest,
                            destination_dir,
                            lifecycle,
                            forward_progress,
                        )
                        .await?;
                    VerifiedModelStageV1 {
                        staging_dir,
                        manifest_fingerprint: frozen_manifest.fingerprint.clone(),
                        source: ModelAcquisitionSourceKindV1::Network,
                        source_hosts: frozen_source_hosts(frozen_manifest)?,
                        bytes_verified: total_frozen_bytes(frozen_manifest)?,
                    }
                }
                ModelAcquisitionSource::LocalBundle(source_dir) => {
                    let forward_progress = |progress: &DownloadProgress| {
                        if let Ok(progress) = acquisition_transport_progress(
                            frozen_manifest,
                            ModelAcquisitionSourceKindV1::LocalBundle,
                            progress,
                            started.elapsed(),
                            Some(cache_reason),
                        ) {
                            on_progress(&progress);
                        }
                    };
                    Self::stage_local_bundle(
                        cx,
                        frozen_manifest,
                        source_dir,
                        destination_dir,
                        lifecycle,
                        &forward_progress,
                    )?
                }
            };

            if stage.manifest_fingerprint != frozen_manifest.fingerprint {
                return Err(SearchError::InvalidConfig {
                    field: "model_acquisition.stage_manifest".to_owned(),
                    value: "mismatch".to_owned(),
                    reason: "verified stage does not bind the requested frozen manifest".to_owned(),
                });
            }
            frozen_manifest.manifest.verify_dir(stage.path())?;
            on_progress(&acquisition_phase_progress(
                frozen_manifest,
                stage.source,
                ModelAcquisitionStageV1::StagedVerified,
                started.elapsed(),
                Some(cache_reason),
                ModelAcquisitionVerificationResultV1::Passed,
            )?);
            check_acquisition_cancel(cx, "load-self-test")?;
            load_self_test(stage.path(), frozen_manifest)?;
            on_progress(&acquisition_phase_progress(
                frozen_manifest,
                stage.source,
                ModelAcquisitionStageV1::LoadSelfTestPassed,
                started.elapsed(),
                Some(cache_reason),
                ModelAcquisitionVerificationResultV1::Passed,
            )?);
            check_acquisition_cancel(cx, "atomic-publication")?;
            let backup = frozen_manifest
                .manifest
                .promote_verified_installation(stage.path(), destination_dir)?;
            lifecycle.mark_acquired_needs_reindex()?;
            on_progress(&acquisition_phase_progress(
                frozen_manifest,
                stage.source,
                ModelAcquisitionStageV1::Published,
                started.elapsed(),
                Some(cache_reason),
                ModelAcquisitionVerificationResultV1::Passed,
            )?);

            acquisition_receipt(
                frozen_manifest,
                stage.source,
                stage.source_hosts,
                ModelAcquisitionOutcomeV1::Published,
                stage.bytes_verified,
                started.elapsed(),
                backup.is_some(),
            )
        }
        .await;

        if let Err(error) = &operation {
            let cancelled = matches!(error, SearchError::Cancelled { .. });
            if cancelled {
                lifecycle.cancel();
            } else {
                lifecycle.fail_verification(bounded_download_failure_reason(error));
            }
            if let Ok(progress) = acquisition_phase_progress(
                frozen_manifest,
                requested_source,
                if cancelled {
                    ModelAcquisitionStageV1::Cancelled
                } else {
                    ModelAcquisitionStageV1::Failed
                },
                started.elapsed(),
                observed_cache_reason,
                if cancelled {
                    ModelAcquisitionVerificationResultV1::Cancelled
                } else {
                    ModelAcquisitionVerificationResultV1::Failed
                },
            ) {
                on_progress(&progress);
            }
        }
        operation
    }

    fn stage_local_bundle(
        cx: &Cx,
        frozen: &FrozenModelArtifactManifestV1,
        source_dir: &Path,
        destination_dir: &Path,
        lifecycle: &mut ModelLifecycle,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<VerifiedModelStageV1> {
        let total_bytes = total_frozen_bytes(frozen)?;
        lifecycle.begin_download(total_bytes)?;
        check_acquisition_cancel(cx, "local-stage-start")?;
        let staging_dir = create_unique_staging_dir(destination_dir)?;
        let files_total = frozen.manifest.artifacts.len();
        let mut cumulative_bytes = 0_u64;

        for (file_idx, artifact) in frozen.manifest.artifacts.iter().enumerate() {
            check_acquisition_cancel(cx, "local-stage-copy")?;
            copy_local_artifact(
                cx,
                source_dir,
                &staging_dir,
                artifact,
                file_idx,
                files_total,
                on_progress,
            )?;
            cumulative_bytes = cumulative_bytes.checked_add(artifact.size).ok_or_else(|| {
                SearchError::InvalidConfig {
                    field: "artifacts[].size".to_owned(),
                    value: "overflow".to_owned(),
                    reason: "total registered artifact bytes must fit in u64".to_owned(),
                }
            })?;
            lifecycle.update_download_progress(cumulative_bytes)?;
        }

        check_acquisition_cancel(cx, "local-stage-verification")?;
        lifecycle.begin_verification()?;
        frozen.manifest.verify_dir(&staging_dir)?;
        lifecycle.mark_staged_verified()?;
        Ok(VerifiedModelStageV1 {
            staging_dir,
            manifest_fingerprint: frozen.fingerprint.clone(),
            source: ModelAcquisitionSourceKindV1::LocalBundle,
            source_hosts: Vec::new(),
            bytes_verified: total_bytes,
        })
    }

    /// Download all files for a model manifest into a staging directory.
    ///
    /// A unique staging directory is created under `{dest_dir}` for each call
    /// (for example, `.download-<pid>-<counter>`), and files are placed there
    /// during download. After all files are verified, the
    /// caller should use [`ModelManifest::promote_verified_installation`] to
    /// atomically install the model.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` on network failure, hash mismatch, or I/O error.
    pub async fn download_model(
        &self,
        cx: &Cx,
        manifest: &ModelManifest,
        dest_dir: &Path,
        lifecycle: &mut ModelLifecycle,
        on_progress: impl Fn(&DownloadProgress) + Send + Sync,
    ) -> SearchResult<PathBuf> {
        if let Err(error) = check_acquisition_cancel(cx, "download-start") {
            lifecycle.cancel();
            return Err(error);
        }
        let total_bytes = manifest.total_size_bytes();
        lifecycle.begin_download(total_bytes.max(1))?;

        if let Err(err) = manifest.validate() {
            lifecycle.fail_verification(format!(
                "manifest validation failed for '{}': {err}",
                manifest.id
            ));
            return Err(err);
        }

        if !manifest.is_production_ready() {
            let reason = format!(
                "manifest for '{}' must be production-ready (pinned revision + verified checksums) before download",
                manifest.id
            );
            lifecycle.fail_verification(reason.clone());
            return Err(SearchError::InvalidConfig {
                field: "manifest".to_owned(),
                value: manifest.id.clone(),
                reason,
            });
        }

        // Fail fast with an actionable error if any declared artifact exceeds the
        // configured response cap, rather than failing cryptically mid-stream with
        // a `BodyTooLarge` codec error after exhausting all retries.
        let cap = self.config.max_response_bytes as u64;
        if let Some(oversized) = manifest.files.iter().find(|file| file.size > cap) {
            let reason = format!(
                "model artifact '{}' declares {} bytes, which exceeds the configured \
                 download response cap of {cap} bytes; raise DownloadConfig.max_response_bytes",
                oversized.name, oversized.size
            );
            lifecycle.fail_verification(reason.clone());
            return Err(SearchError::InvalidConfig {
                field: "download.max_response_bytes".to_owned(),
                value: manifest.id.clone(),
                reason,
            });
        }

        let staging_dir = match create_unique_staging_dir(dest_dir) {
            Ok(path) => path,
            Err(err) => {
                lifecycle.fail_verification(format!(
                    "failed to create staging directory for '{}': {err}",
                    manifest.id
                ));
                return Err(err);
            }
        };

        let files_total = manifest.files.len();
        let mut cumulative_bytes: u64 = 0;

        for (idx, file) in manifest.files.iter().enumerate() {
            if let Err(error) = check_acquisition_cancel(cx, "download-file-start") {
                lifecycle.cancel();
                return Err(error);
            }
            let url = file.download_url(&manifest.repo, &manifest.revision);
            let file_dest = staging_dir.join(&file.name);

            // Create parent directories for nested paths (e.g., "onnx/model.onnx").
            if let Some(parent) = file_dest.parent()
                && let Err(err) = std::fs::create_dir_all(parent).map_err(SearchError::from)
            {
                lifecycle.fail_verification(format!(
                    "failed to create parent directory for '{}': {err}",
                    file.name
                ));
                return Err(err);
            }

            info!(
                file = %file.name,
                size = file.size,
                "downloading model file"
            );

            if let Err(err) = self
                .download_file_with_retry(
                    cx,
                    &url,
                    &file_dest,
                    file,
                    idx,
                    files_total,
                    &on_progress,
                )
                .await
            {
                if matches!(err, SearchError::Cancelled { .. }) {
                    lifecycle.cancel();
                } else {
                    lifecycle.fail_verification(format!(
                        "download failed for '{}': {}",
                        file.name,
                        bounded_download_failure_reason(&err)
                    ));
                }
                return Err(err);
            }

            cumulative_bytes = cumulative_bytes.saturating_add(file.size);
            if let Err(err) = lifecycle.update_download_progress(cumulative_bytes) {
                lifecycle.fail_verification(format!("failed to update download progress: {err}"));
                return Err(err);
            }
        }

        // Verify all files.
        if let Err(error) = check_acquisition_cancel(cx, "download-verification") {
            lifecycle.cancel();
            return Err(error);
        }
        if let Err(err) = lifecycle.begin_verification() {
            lifecycle.fail_verification(format!(
                "failed to transition model lifecycle into verification: {err}"
            ));
            return Err(err);
        }
        info!(model = %manifest.id, "verifying downloaded files");
        match manifest.verify_dir(&staging_dir) {
            Ok(()) => {
                lifecycle.mark_staged_verified()?;
                info!(model = %manifest.id, "model download stage complete and verified");
                Ok(staging_dir)
            }
            Err(e) => {
                lifecycle.fail_verification(bounded_download_failure_reason(&e));
                Err(e)
            }
        }
    }

    /// Download a single file with retry logic.
    pub(super) async fn download_file_with_retry(
        &self,
        cx: &Cx,
        url: &str,
        dest: &Path,
        file: &ModelFile,
        file_idx: usize,
        files_total: usize,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            check_acquisition_cancel(cx, "download-retry")?;
            if attempt > 0 {
                let delay = self.config.retry_base_delay * 2_u32.saturating_pow(attempt - 1);
                warn!(
                    file = %file.name,
                    attempt,
                    delay_ms = delay.as_millis(),
                    "retrying download after failure"
                );
                asupersync::time::sleep(cx.now(), delay).await;
                check_acquisition_cancel(cx, "download-backoff")?;
            }

            match self
                .download_single_file(cx, url, dest, file, file_idx, files_total, on_progress)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if matches!(e, SearchError::Cancelled { .. }) {
                        return Err(e);
                    }
                    warn!(
                        file = %file.name,
                        attempt,
                        reason = bounded_download_failure_reason(&e),
                        "download attempt failed"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| SearchError::ModelLoadFailed {
            path: dest.to_path_buf(),
            source: "download failed after all retries".into(),
        }))
    }

    /// Download a single file (one attempt).
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub(super) async fn download_single_file(
        &self,
        cx: &Cx,
        url: &str,
        dest: &Path,
        file: &ModelFile,
        file_idx: usize,
        files_total: usize,
        on_progress: &(impl Fn(&DownloadProgress) + Send + Sync),
    ) -> SearchResult<()> {
        let start = Instant::now();
        check_acquisition_cancel(cx, "download-request")?;

        // Report start.
        on_progress(&DownloadProgress {
            file_name: file.name.clone(),
            bytes_downloaded: 0,
            total_bytes: if file.size > 0 { Some(file.size) } else { None },
            files_completed: file_idx,
            files_total,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
        });

        // Stream directly into a temp file to keep memory bounded.
        let mut response = self
            .client
            .request_streaming(cx, Method::Get, url, Vec::new(), Vec::new())
            .await
            .map_err(|e| client_error_to_search(e, url))?;
        check_acquisition_cancel(cx, "download-response")?;

        // Check HTTP status.
        if response.head.status < 200 || response.head.status >= 300 {
            return Err(SearchError::ModelLoadFailed {
                path: PathBuf::from("<redacted-model-stage>"),
                source: format!(
                    "HTTP {} from host {}",
                    response.head.status,
                    diagnostic_source_host(url)
                )
                .into(),
            });
        }
        if let Some(content_length) = response_content_length(&response.head.headers)
            && file.size > 0
            && content_length != file.size
        {
            return Err(SearchError::HashMismatch {
                path: dest.to_path_buf(),
                expected: format!("content-length={}", file.size),
                actual: format!("content-length={content_length}"),
            });
        }

        let tmp_path = dest.with_extension("tmp");
        let mut tmp_guard = TempFileGuard::new(tmp_path.clone());
        let mut tmp_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(SearchError::from)?;

        let total_bytes = if file.size > 0 {
            Some(file.size)
        } else {
            response_content_length(&response.head.headers)
        };
        let mut hasher = Sha256::new();
        let mut bytes_downloaded: u64 = 0;

        while let Some(frame) = poll_fn(|cx| Pin::new(&mut response.body).poll_frame(cx)).await {
            check_acquisition_cancel(cx, "download-stream")?;
            let frame = frame.map_err(|e| SearchError::ModelLoadFailed {
                path: PathBuf::from("<redacted-model-stage>"),
                source: format!(
                    "stream read failed via host {} ({})",
                    diagnostic_source_host(url),
                    bounded_http_body_error(&e)
                )
                .into(),
            })?;
            if let Frame::Data(mut chunk) = frame {
                while chunk.has_remaining() {
                    check_acquisition_cancel(cx, "download-chunk")?;
                    let bytes = chunk.chunk();
                    if bytes.is_empty() {
                        break;
                    }
                    let byte_count = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                    let next_size = bytes_downloaded.checked_add(byte_count).ok_or_else(|| {
                        SearchError::HashMismatch {
                            path: dest.to_path_buf(),
                            expected: format!("size={}", file.size),
                            actual: "size-overflow".to_owned(),
                        }
                    })?;
                    if file.size > 0 && next_size > file.size {
                        return Err(SearchError::HashMismatch {
                            path: dest.to_path_buf(),
                            expected: format!("size={}", file.size),
                            actual: format!("size-at-least={next_size}"),
                        });
                    }
                    tmp_file.write_all(bytes).map_err(SearchError::from)?;
                    hasher.update(bytes);
                    bytes_downloaded = next_size;
                    chunk.advance(bytes.len());
                }

                let elapsed = start.elapsed();
                let speed = if elapsed.as_secs_f64() > 0.0 {
                    bytes_downloaded as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let eta_seconds = total_bytes.and_then(|total| {
                    if speed <= f64::EPSILON || bytes_downloaded >= total {
                        None
                    } else {
                        Some((total.saturating_sub(bytes_downloaded)) as f64 / speed)
                    }
                });
                on_progress(&DownloadProgress {
                    file_name: file.name.clone(),
                    bytes_downloaded,
                    total_bytes,
                    files_completed: file_idx,
                    files_total,
                    speed_bytes_per_sec: speed,
                    eta_seconds,
                });
            }
        }

        check_acquisition_cancel(cx, "download-file-sync")?;
        // ubs:ignore — artifact byte counts are public integrity metadata.
        if file.size > 0 && bytes_downloaded != file.size {
            return Err(SearchError::HashMismatch {
                path: dest.to_path_buf(),
                expected: format!("size={}", file.size),
                actual: format!("size={bytes_downloaded}"),
            });
        }

        // Verify SHA-256 only when the manifest provides a concrete checksum.
        if file.has_verified_checksum() {
            let actual_hash = sha256_digest_hex(hasher.finalize().as_slice());
            // ubs:ignore — pinned artifact digests are public integrity metadata.
            if actual_hash != file.sha256 {
                return Err(SearchError::HashMismatch {
                    path: dest.to_path_buf(),
                    expected: format!("sha256={},size={}", file.sha256, file.size),
                    actual: format!("sha256={actual_hash},size={bytes_downloaded}"),
                });
            }
        }

        tmp_file.flush().map_err(SearchError::from)?;
        tmp_file.sync_all().map_err(SearchError::from)?;
        drop(tmp_file);
        check_acquisition_cancel(cx, "download-file-publish")?;
        std::fs::rename(&tmp_path, dest).map_err(SearchError::from)?;
        tmp_guard.disarm();

        let elapsed = start.elapsed();
        let speed = if elapsed.as_secs_f64() > 0.0 {
            bytes_downloaded as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        on_progress(&DownloadProgress {
            file_name: file.name.clone(),
            bytes_downloaded,
            total_bytes: if file.size > 0 {
                Some(file.size)
            } else {
                total_bytes
            },
            files_completed: file_idx,
            files_total,
            speed_bytes_per_sec: speed,
            eta_seconds: Some(0.0),
        });

        info!(
            file = %file.name,
            bytes = bytes_downloaded,
            elapsed_ms = elapsed.as_millis(),
            "file saved"
        );

        Ok(())
    }
}
