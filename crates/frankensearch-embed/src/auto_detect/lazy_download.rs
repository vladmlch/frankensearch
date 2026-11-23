//! Lazy auto-download wrappers and progress reporting for embedders.

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use asupersync::Cx;
use asupersync::sync::OnceCell;
use tracing::{info, warn};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{Embedder, ModelCategory, ModelTier, SearchFuture};

#[cfg(feature = "fastembed")]
use crate::fastembed_embedder::{
    DEFAULT_DIMENSION as MINILM_DIMENSION, DEFAULT_MODEL_NAME as MINILM_MODEL_NAME,
    FastEmbedEmbedder,
};
use crate::model_download::{DownloadProgress, ModelDownloader};
use crate::model_manifest::{
    ConsentSource, DownloadConsent, ModelArtifactManifestV1, ModelLifecycle, ModelManifest,
    resolve_download_consent,
};
use crate::model_registry::ensure_model_storage_layout_checked;
#[cfg(feature = "model2vec")]
use crate::model2vec_embedder::Model2VecEmbedder;

use super::discovery::OFFLINE_ENV;
#[cfg(feature = "fastembed")]
use super::discovery::detect_quality_embedder;
#[cfg(feature = "model2vec")]
use super::discovery::{POTION_MODEL_NAME, detect_fast_embedder};

#[cfg(feature = "model2vec")]
const POTION_DIMENSION: usize = 256;
pub(super) const PROGRESS_BAR_WIDTH: usize = 30;

#[derive(Debug, Clone, Copy)]
pub(super) struct DownloadPolicy {
    consent: DownloadConsent,
    offline: bool,
    stderr_is_tty: bool,
}

impl DownloadPolicy {
    pub(super) const fn can_download(self) -> bool {
        self.consent.granted && !self.offline
    }

    pub(super) fn blocked_reason(self) -> String {
        if self.offline {
            return format!(
                "{OFFLINE_ENV}=1 or an explicit offline detection policy disables model \
                 auto-download"
            );
        }
        if !self.consent.granted {
            let source = self
                .consent
                .source
                .map_or_else(|| "unset".to_owned(), |s| format!("{s:?}"));
            return format!("download consent denied (source={source})");
        }
        "download policy blocked".to_owned()
    }

    #[cfg(test)]
    pub(super) const fn for_tests(
        consent: DownloadConsent,
        offline: bool,
        stderr_is_tty: bool,
    ) -> Self {
        Self {
            consent,
            offline,
            stderr_is_tty,
        }
    }
}

pub(super) fn download_policy_from_environment(offline: bool) -> DownloadPolicy {
    let consent = if offline {
        DownloadConsent::denied(Some(ConsentSource::Environment))
    } else {
        // Default to "denied" unless explicitly allowed via FRANKENSEARCH_ALLOW_DOWNLOAD=1.
        // This keeps first-run search offline-safe and avoids surprise network fetches.
        resolve_download_consent(None, None, Some(false))
    };
    DownloadPolicy {
        consent,
        offline,
        stderr_is_tty: io::stderr().is_terminal(),
    }
}

#[cfg(feature = "model2vec")]
pub(super) fn maybe_lazy_fast_embedder(
    model_root: Option<&Path>,
    policy: DownloadPolicy,
) -> Option<Arc<dyn Embedder>> {
    if !policy.can_download() {
        info!(
            model = POTION_MODEL_NAME,
            tier = "fast",
            reason = %policy.blocked_reason(),
            "auto-download disabled; fast tier falling back"
        );
        return None;
    }

    info!(
        model = POTION_MODEL_NAME,
        tier = "fast",
        "model not found locally; deferring download to first embed call"
    );
    match LazyModel2VecEmbedder::new(model_root.map(Path::to_path_buf), policy) {
        Ok(embedder) => Some(Arc::new(embedder)),
        Err(_error) => {
            warn!(
                model = POTION_MODEL_NAME,
                reason = "registered-identity-invalid",
                "registered lazy fast-tier identity is invalid"
            );
            None
        }
    }
}

#[cfg(not(feature = "model2vec"))]
pub(super) fn maybe_lazy_fast_embedder(
    _model_root: Option<&Path>,
    _policy: DownloadPolicy,
) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(feature = "fastembed")]
pub(super) fn maybe_lazy_quality_embedder(
    model_root: Option<&Path>,
    policy: DownloadPolicy,
) -> Option<Arc<dyn Embedder>> {
    if !policy.can_download() {
        info!(
            model = MINILM_MODEL_NAME,
            tier = "quality",
            reason = %policy.blocked_reason(),
            "auto-download disabled; quality tier unavailable"
        );
        return None;
    }

    info!(
        model = MINILM_MODEL_NAME,
        tier = "quality",
        "model not found locally; deferring download to first embed call"
    );
    match LazyFastEmbedEmbedder::new(model_root.map(Path::to_path_buf), policy) {
        Ok(embedder) => Some(Arc::new(embedder)),
        Err(_error) => {
            warn!(
                model = MINILM_MODEL_NAME,
                reason = "registered-identity-invalid",
                "registered lazy quality-tier identity is invalid"
            );
            None
        }
    }
}

#[cfg(not(feature = "fastembed"))]
pub(super) fn maybe_lazy_quality_embedder(
    _model_root: Option<&Path>,
    _policy: DownloadPolicy,
) -> Option<Arc<dyn Embedder>> {
    None
}

fn install_destination_dir(model_root: Option<&Path>, model_name: &str) -> SearchResult<PathBuf> {
    if let Some(root) = model_root {
        if root.ends_with(model_name) {
            return Ok(root.to_path_buf());
        }
        return Ok(root.join(model_name));
    }
    Ok(ensure_model_storage_layout_checked()?.join(model_name))
}

async fn download_and_install_manifest(
    cx: &Cx,
    manifest: &ModelManifest,
    destination_dir: &Path,
    policy: DownloadPolicy,
) -> SearchResult<()> {
    let start = Instant::now();
    let downloader = ModelDownloader::with_defaults();
    let mut lifecycle = ModelLifecycle::new(manifest.clone(), policy.consent);
    let staging_root = destination_dir
        .parent()
        .map_or_else(|| destination_dir.to_path_buf(), Path::to_path_buf);
    std::fs::create_dir_all(&staging_root)?;

    let reporter = Arc::new(DownloadProgressReporter::new(
        manifest.id.clone(),
        policy.stderr_is_tty,
    ));
    let reporter_for_cb = Arc::clone(&reporter);

    info!(
        model = %manifest.id,
        bytes = manifest.total_size_bytes(),
        "starting automatic model download"
    );

    let staged = match downloader
        .download_model(
            cx,
            manifest,
            &staging_root,
            &mut lifecycle,
            move |progress| {
                reporter_for_cb.report(progress);
            },
        )
        .await
    {
        Ok(staged) => staged,
        Err(error) => {
            reporter.finish_failed(start.elapsed(), &error);
            warn!(
                model = %manifest.id,
                duration_ms = start.elapsed().as_millis(),
                reason = "transport_or_verification_failed",
                "automatic model download failed"
            );
            return Err(error);
        }
    };

    match manifest.promote_verified_installation(&staged, destination_dir) {
        Ok(backup) => {
            reporter.finish_ok(start.elapsed(), manifest.total_size_bytes());
            info!(
                model = %manifest.id,
                backup_created = backup.is_some(),
                duration_ms = start.elapsed().as_millis(),
                bytes = manifest.total_size_bytes(),
                "automatic model download completed"
            );
            Ok(())
        }
        Err(error) => {
            reporter.finish_failed(start.elapsed(), &error);
            warn!(
                model = %manifest.id,
                duration_ms = start.elapsed().as_millis(),
                reason = "verified_promotion_failed",
                "automatic model promotion failed"
            );
            Err(error)
        }
    }
}

#[cfg(feature = "model2vec")]
pub(super) struct LazyModel2VecEmbedder {
    model_root: Option<PathBuf>,
    policy: DownloadPolicy,
    inner: OnceCell<Arc<dyn Embedder>>,
    identity: EmbeddingIdentityBundleV1,
}

#[cfg(feature = "model2vec")]
impl LazyModel2VecEmbedder {
    pub(super) fn new(model_root: Option<PathBuf>, policy: DownloadPolicy) -> SearchResult<Self> {
        let identity = ModelArtifactManifestV1::potion_128m_native()?
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
        Ok(Self {
            model_root,
            policy,
            inner: OnceCell::new(),
            identity,
        })
    }

    async fn ensure_loaded(&self, cx: &Cx) -> SearchResult<Arc<dyn Embedder>> {
        let embedder = self
            .inner
            .get_or_try_init(|| async { self.initialize(cx).await })
            .await?;
        Ok(Arc::clone(embedder))
    }

    async fn initialize(&self, cx: &Cx) -> SearchResult<Arc<dyn Embedder>> {
        if let Some(existing) = detect_fast_embedder(self.model_root.as_deref()) {
            let loaded_identity = existing.identity()?;
            // ubs:ignore — fingerprints are public compatibility IDs, not secrets.
            if loaded_identity.fingerprint() != self.identity.fingerprint() {
                return Err(SearchError::InvalidConfig {
                    field: "lazy_model2vec.identity".to_owned(),
                    value: loaded_identity.fingerprint(),
                    reason: "detected loaded identity disagrees with the registered lazy identity"
                        .to_owned(),
                });
            }
            return Ok(existing);
        }
        if !self.policy.can_download() {
            return Err(SearchError::EmbedderUnavailable {
                model: POTION_MODEL_NAME.to_owned(),
                reason: self.policy.blocked_reason(),
            });
        }

        let manifest = ModelManifest::potion_128m();
        let destination = install_destination_dir(self.model_root.as_deref(), POTION_MODEL_NAME)?;
        download_and_install_manifest(cx, &manifest, &destination, self.policy).await?;
        let embedder = Model2VecEmbedder::load_shared_with_name(&destination, POTION_MODEL_NAME)?;
        let loaded_identity = embedder.identity()?;
        // ubs:ignore — fingerprints are public compatibility IDs, not secrets.
        if loaded_identity.fingerprint() != self.identity.fingerprint() {
            return Err(SearchError::InvalidConfig {
                field: "lazy_model2vec.identity".to_owned(),
                value: loaded_identity.fingerprint(),
                reason: "verified loaded identity disagrees with the registered lazy identity"
                    .to_owned(),
            });
        }
        Ok(embedder as Arc<dyn Embedder>)
    }
}

#[cfg(feature = "model2vec")]
impl Embedder for LazyModel2VecEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let embedder = self.ensure_loaded(cx).await?;
            embedder.embed(cx, text).await
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let embedder = self.ensure_loaded(cx).await?;
            embedder.embed_batch(cx, texts).await
        })
    }

    /// Both bound operations are the core defaults over this wrapper's own
    /// `embed`/`embed_batch`, so a bound batch binds what singles would.
    fn bound_batch_is_native(&self) -> bool {
        true
    }

    fn dimension(&self) -> usize {
        POTION_DIMENSION
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        POTION_MODEL_NAME
    }

    fn model_name(&self) -> &str {
        POTION_MODEL_NAME
    }

    fn is_ready(&self) -> bool {
        self.inner.get().is_some_and(|embedder| embedder.is_ready())
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }

    fn tier(&self) -> ModelTier {
        ModelTier::Fast
    }
}

#[cfg(feature = "fastembed")]
struct LazyFastEmbedEmbedder {
    model_root: Option<PathBuf>,
    policy: DownloadPolicy,
    inner: OnceCell<Arc<dyn Embedder>>,
    identity: EmbeddingIdentityBundleV1,
}

#[cfg(feature = "fastembed")]
impl LazyFastEmbedEmbedder {
    fn new(model_root: Option<PathBuf>, policy: DownloadPolicy) -> SearchResult<Self> {
        let identity = ModelArtifactManifestV1::minilm_fastembed()?
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
        Ok(Self {
            model_root,
            policy,
            inner: OnceCell::new(),
            identity,
        })
    }

    async fn ensure_loaded(&self, cx: &Cx) -> SearchResult<Arc<dyn Embedder>> {
        let embedder = self
            .inner
            .get_or_try_init(|| async { self.initialize(cx).await })
            .await?;
        Ok(Arc::clone(embedder))
    }

    async fn initialize(&self, cx: &Cx) -> SearchResult<Arc<dyn Embedder>> {
        if let Some(existing) = detect_quality_embedder(self.model_root.as_deref()) {
            let loaded_identity = existing.identity()?;
            // ubs:ignore — fingerprints are public compatibility IDs, not secrets.
            if loaded_identity.fingerprint() != self.identity.fingerprint() {
                return Err(SearchError::InvalidConfig {
                    field: "lazy_fastembed.identity".to_owned(),
                    value: loaded_identity.fingerprint(),
                    reason: "detected loaded identity disagrees with the registered lazy identity"
                        .to_owned(),
                });
            }
            return Ok(existing);
        }
        if !self.policy.can_download() {
            return Err(SearchError::EmbedderUnavailable {
                model: MINILM_MODEL_NAME.to_owned(),
                reason: self.policy.blocked_reason(),
            });
        }

        let manifest = ModelManifest::minilm_v2();
        let destination = install_destination_dir(self.model_root.as_deref(), MINILM_MODEL_NAME)?;
        download_and_install_manifest(cx, &manifest, &destination, self.policy).await?;
        let embedder = FastEmbedEmbedder::load_with_name(&destination, MINILM_MODEL_NAME)?;
        let loaded_identity = embedder.identity()?;
        // ubs:ignore — fingerprints are public compatibility IDs, not secrets.
        if loaded_identity.fingerprint() != self.identity.fingerprint() {
            return Err(SearchError::InvalidConfig {
                field: "lazy_fastembed.identity".to_owned(),
                value: loaded_identity.fingerprint(),
                reason: "verified loaded identity disagrees with the registered lazy identity"
                    .to_owned(),
            });
        }
        Ok(Arc::new(embedder))
    }
}

#[cfg(feature = "fastembed")]
impl Embedder for LazyFastEmbedEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let embedder = self.ensure_loaded(cx).await?;
            embedder.embed(cx, text).await
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let embedder = self.ensure_loaded(cx).await?;
            embedder.embed_batch(cx, texts).await
        })
    }

    /// Both bound operations are the core defaults over this wrapper's own
    /// `embed`/`embed_batch`, so a bound batch binds what singles would.
    fn bound_batch_is_native(&self) -> bool {
        true
    }

    fn dimension(&self) -> usize {
        MINILM_DIMENSION
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        MINILM_MODEL_NAME
    }

    fn model_name(&self) -> &str {
        MINILM_MODEL_NAME
    }

    fn is_ready(&self) -> bool {
        self.inner.get().is_some_and(|embedder| embedder.is_ready())
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }

    fn tier(&self) -> ModelTier {
        ModelTier::Quality
    }
}

#[derive(Debug)]
struct DownloadProgressReporter {
    model_id: String,
    stderr_is_tty: bool,
    last_bucket: AtomicU8,
}

impl DownloadProgressReporter {
    const fn new(model_id: String, stderr_is_tty: bool) -> Self {
        Self {
            model_id,
            stderr_is_tty,
            last_bucket: AtomicU8::new(0),
        }
    }

    fn report(&self, progress: &DownloadProgress) {
        let progress_x100 = progress_percent_x100(progress);
        if self.stderr_is_tty {
            self.report_tty(progress, progress_x100);
        } else {
            self.report_non_tty(progress, progress_x100);
        }
    }

    fn finish_ok(&self, elapsed: std::time::Duration, total_bytes: u64) {
        if self.stderr_is_tty {
            eprintln!(
                "\rDownloaded {} in {:.1}s ({})",
                self.model_id,
                elapsed.as_secs_f64(),
                format_bytes(total_bytes),
            );
        } else {
            eprintln!(
                "Downloaded {} in {:.1}s ({})",
                self.model_id,
                elapsed.as_secs_f64(),
                format_bytes(total_bytes),
            );
        }
    }

    fn finish_failed(&self, elapsed: std::time::Duration, _error: &SearchError) {
        if self.stderr_is_tty {
            eprintln!(
                "\rDownload failed for {} after {:.1}s",
                self.model_id,
                elapsed.as_secs_f64(),
            );
        } else {
            eprintln!(
                "Download failed for {} after {:.1}s",
                self.model_id,
                elapsed.as_secs_f64(),
            );
        }
    }

    fn report_tty(&self, progress: &DownloadProgress, progress_x100: u64) {
        let pct_whole = progress_x100 / 100;
        let pct_frac = progress_x100 % 100;
        let bar = render_progress_bar(progress_x100);
        let total = progress
            .total_bytes
            .map_or_else(|| "?".to_owned(), format_bytes);
        eprint!(
            "\rDownloading {} [{}] {:>3}.{pct_frac:02}% {}/{} {} ETA {} ({}/{})",
            self.model_id,
            bar,
            pct_whole,
            format_bytes(progress.bytes_downloaded),
            total,
            format_speed(progress.speed_bytes_per_sec),
            format_eta(progress.eta_seconds),
            progress.files_completed + 1,
            progress.files_total.max(1),
        );
        let _ = io::stderr().flush();
    }

    fn report_non_tty(&self, progress: &DownloadProgress, progress_x100: u64) {
        let bucket = u8::try_from((progress_x100 / 1000).min(10)).unwrap_or(10);
        let previous = self.last_bucket.load(Ordering::Relaxed);
        if bucket <= previous {
            return;
        }
        if self
            .last_bucket
            .compare_exchange(previous, bucket, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            eprintln!(
                "Downloading {}... {}% ({}/{} {} ETA {})",
                self.model_id,
                bucket.saturating_mul(10),
                format_bytes(progress.bytes_downloaded),
                progress
                    .total_bytes
                    .map_or_else(|| "?".to_owned(), format_bytes),
                format_speed(progress.speed_bytes_per_sec),
                format_eta(progress.eta_seconds),
            );
        }
    }
}

pub(super) fn progress_percent_x100(progress: &DownloadProgress) -> u64 {
    let files_total = u64::try_from(progress.files_total).unwrap_or(1).max(1);
    let files_completed = u64::try_from(progress.files_completed)
        .unwrap_or(files_total)
        .min(files_total);
    let current_file_percent_x100 = progress
        .total_bytes
        .filter(|&total| total > 0)
        .map_or(0, |total| {
            progress.bytes_downloaded.min(total).saturating_mul(10_000) / total
        });
    files_completed
        .saturating_mul(10_000)
        .saturating_add(current_file_percent_x100)
        / files_total
}

pub(super) fn render_progress_bar(progress_x100: u64) -> String {
    let width = u64::try_from(PROGRESS_BAR_WIDTH).unwrap_or(30);
    let filled = usize::try_from(progress_x100.saturating_mul(width) / 10_000)
        .unwrap_or(PROGRESS_BAR_WIDTH)
        .min(PROGRESS_BAR_WIDTH);
    let mut bar = String::with_capacity(PROGRESS_BAR_WIDTH);
    bar.push_str(&"=".repeat(filled));
    bar.push_str(&" ".repeat(PROGRESS_BAR_WIDTH.saturating_sub(filled)));
    bar
}

pub(super) fn format_eta(seconds: Option<f64>) -> String {
    match seconds {
        Some(value) if value.is_finite() && value >= 0.0 => format!("{value:.1}s"),
        _ => "?".to_owned(),
    }
}

pub(super) fn format_speed(bytes_per_sec: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    if !bytes_per_sec.is_finite() || bytes_per_sec <= 0.0 {
        return "0 B/s".to_owned();
    }
    if bytes_per_sec >= GB {
        format!("{:.1} GB/s", bytes_per_sec / GB)
    } else if bytes_per_sec >= MB {
        format!("{:.1} MB/s", bytes_per_sec / MB)
    } else if bytes_per_sec >= KB {
        format!("{:.1} KB/s", bytes_per_sec / KB)
    } else {
        format!("{bytes_per_sec:.0} B/s")
    }
}

pub(super) fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        let whole = bytes / GB;
        let frac = bytes % GB * 10 / GB;
        format!("{whole}.{frac} GB")
    } else if bytes >= MB {
        let whole = bytes / MB;
        let frac = bytes % MB * 10 / MB;
        format!("{whole}.{frac} MB")
    } else if bytes >= KB {
        let whole = bytes / KB;
        let frac = bytes % KB * 10 / KB;
        format!("{whole}.{frac} KB")
    } else {
        format!("{bytes} B")
    }
}
