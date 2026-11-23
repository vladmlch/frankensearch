//! Streaming SHA-256 verification, `.verified` receipt caching, and atomic promotion.

use std::collections::BTreeMap;
#[cfg(unix)]
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::error::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::catalog::ModelManifest;
use super::lifecycle::ModelState;
use super::{MAX_FROZEN_MANIFEST_FIELD_BYTES, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD};

const HASH_BUFFER_SIZE: usize = 8 * 1024;

/// Verify file size + SHA256 using streaming read.
///
/// # Errors
///
/// Returns `SearchError` when file is missing, unreadable, or hash/size mismatch occurs.
pub fn verify_file_sha256(
    path: &Path,
    expected_sha256: &str,
    expected_size: u64,
) -> SearchResult<()> {
    record_verify_file_hash_call();
    if expected_sha256 == PLACEHOLDER_VERIFY_AFTER_DOWNLOAD {
        return Err(SearchError::InvalidConfig {
            field: "sha256".to_owned(),
            value: expected_sha256.to_owned(),
            reason: "placeholder checksum cannot be verified".to_owned(),
        });
    }
    if !is_valid_sha256_hex(expected_sha256) {
        return Err(SearchError::InvalidConfig {
            field: "sha256".to_owned(),
            value: "redacted-invalid-sha256".to_owned(),
            reason: "expected lowercase 64-char SHA256 hex".to_owned(),
        });
    }
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(SearchError::ModelNotFound {
                name: format!("missing model file: {}", path.display()),
            });
        }
        Err(source) => {
            return Err(SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: Box::new(source),
            });
        }
    };
    if !metadata.file_type().is_file() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "expected a regular file, not a symlink or special node".into(),
        });
    }

    let file = File::open(path).map_err(|source| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0_u8; HASH_BUFFER_SIZE];
    let mut hasher = Sha256::new();
    let mut bytes_read = 0_u64;

    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: Box::new(source),
            })?;
        if read == 0 {
            break;
        }
        let read_u64 = u64::try_from(read).map_err(|_| SearchError::InvalidConfig {
            field: "read_size".to_owned(),
            value: read.to_string(),
            reason: "read size does not fit u64".to_owned(),
        })?;
        bytes_read = bytes_read.saturating_add(read_u64);
        hasher.update(&buffer[..read]);
    }

    let actual_sha256 = to_hex_lowercase(&hasher.finalize());
    let expected_lower = expected_sha256.to_ascii_lowercase();
    // ubs:ignore — artifact sizes and SHA-256 digests are public integrity metadata.
    if bytes_read != expected_size || actual_sha256 != expected_lower {
        return Err(SearchError::HashMismatch {
            path: path.to_path_buf(),
            expected: format!("sha256={expected_lower},size={expected_size}"),
            actual: format!("sha256={actual_sha256},size={bytes_read}"),
        });
    }

    Ok(())
}

// Test-only count of full per-file hash passes on the calling thread.
//
// Mirrors the `TEST_AFTER_FULL_HASH_HOOK` seam: production builds compile the
// counter and its increment out, so behavior and layout are unchanged.
#[cfg(test)]
thread_local! {
    pub(super) static TEST_VERIFY_FILE_HASH_CALLS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

fn record_verify_file_hash_call() {
    #[cfg(test)]
    TEST_VERIFY_FILE_HASH_CALLS.with(|count| count.set(count.get() + 1));
}

// ─── Verification Cache ────────────────────────────────────────────────────

/// Name of the verification marker file within a model directory.
pub(super) const VERIFIED_MARKER_FILE: &str = ".verified";

/// Schema version for the verification receipt itself.
///
/// This is intentionally independent from [`MANIFEST_SCHEMA_VERSION`](super::MANIFEST_SCHEMA_VERSION). Older binaries
/// compare their marker field to the manifest schema, so using a distinct value also
/// makes a newly written fingerprint-bound receipt fail closed after a binary downgrade.
pub const VERIFICATION_MARKER_SCHEMA_VERSION: u32 = 1;

/// Lightweight filesystem fingerprint for one verified model file.
///
/// This is intentionally cheap to read and compare:
/// - file size (bytes)
/// - last-modified timestamp (unix nanos)
/// - creation timestamp when the platform exposes it
/// - platform file identity and change timestamp on Unix
///
/// These fields detect ordinary replacement and mutation without re-hashing large
/// model artifacts on every process start. They are not an authenticated defense
/// against an attacker who can rewrite both the model bytes and this user-owned
/// receipt; that threat requires an OS-backed trust root such as fs-verity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileVerificationState {
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last-modified timestamp since unix epoch (nanoseconds).
    pub modified_unix_nanos: u64,
    /// Creation timestamp since unix epoch (nanoseconds), when available.
    pub created_unix_nanos: Option<u64>,
    /// Stable platform file identity, when available.
    pub platform_file_id: Option<String>,
    /// Platform change timestamp, when available.
    pub platform_change_stamp: Option<String>,
}

impl FileVerificationState {
    /// Whether a freshly captured state still matches this recorded one.
    ///
    /// Every field must match, except a creation timestamp that only one side
    /// could observe: Rust's std reads Linux birth times through `statx` on
    /// glibc but not on musl, so a receipt minted by the static musl `fsfs`
    /// records none while a glibc build observes one for the same untouched
    /// file. Treating that as a mismatch made every glibc load re-hash the
    /// models forever. Size, mtime and, on Unix, inode and change stamp still
    /// bind the file.
    pub(super) fn still_matches(&self, current: &Self) -> bool {
        self.size_bytes == current.size_bytes
            && self.modified_unix_nanos == current.modified_unix_nanos
            && self.platform_file_id == current.platform_file_id
            && self.platform_change_stamp == current.platform_change_stamp
            && match (self.created_unix_nanos, current.created_unix_nanos) {
                (Some(recorded), Some(observed)) => recorded == observed,
                _ => true,
            }
    }
}

pub(super) fn capture_file_verification_state(path: &Path) -> Option<FileVerificationState> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if !metadata.file_type().is_file() {
        return None;
    }
    let modified = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let modified_unix_nanos = u64::try_from(modified).ok()?;
    let created_unix_nanos = metadata
        .created()
        .ok()
        .and_then(|created| created.duration_since(UNIX_EPOCH).ok())
        .and_then(|created| u64::try_from(created.as_nanos()).ok());
    #[cfg(unix)]
    let (platform_file_id, platform_change_stamp) = {
        use std::os::unix::fs::MetadataExt as _;
        (
            Some(format!(
                "unix-dev={};ino={}",
                metadata.dev(),
                metadata.ino()
            )),
            Some(format!(
                "unix-ctime={};nsec={}",
                metadata.ctime(),
                metadata.ctime_nsec()
            )),
        )
    };
    #[cfg(not(unix))]
    let (platform_file_id, platform_change_stamp) = (None, None);
    Some(FileVerificationState {
        size_bytes: metadata.len(),
        modified_unix_nanos,
        created_unix_nanos,
        platform_file_id,
        platform_change_stamp,
    })
}

pub(super) fn resolve_model_file_path(model_dir: &Path, file_name: &str) -> SearchResult<PathBuf> {
    validate_model_file_name(file_name)?;
    Ok(model_dir.join(file_name))
}

pub(super) fn validate_model_file_name(file_name: &str) -> SearchResult<()> {
    if file_name.len() > MAX_FROZEN_MANIFEST_FIELD_BYTES {
        return Err(invalid_manifest_field(
            "files[].name",
            "redacted-oversized",
            "must fit the bounded manifest field size",
        ));
    }
    if file_name.chars().any(char::is_control) {
        return Err(invalid_manifest_field(
            "files[].name",
            "redacted-control-character",
            "must not contain control characters",
        ));
    }
    if file_name.trim().is_empty() {
        return Err(invalid_manifest_field(
            "files[].name",
            file_name,
            "must not be empty",
        ));
    }
    for component in Path::new(file_name).components() {
        match component {
            std::path::Component::ParentDir => {
                return Err(invalid_manifest_field(
                    "files[].name",
                    "redacted-path-traversal",
                    "must not contain '..' path traversal",
                ));
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(invalid_manifest_field(
                    "files[].name",
                    "redacted-absolute-path",
                    "must be a relative path without root",
                ));
            }
            _ => {}
        }
    }
    if file_name.contains('\\')
        || file_name
            .bytes()
            .any(|byte| matches!(byte, b':' | b'<' | b'>' | b'"' | b'|' | b'?' | b'*'))
        || file_name.split('/').any(|component| {
            component.is_empty()
                || component == "."
                || component.trim() != component
                || component.ends_with('.')
                || is_windows_reserved_path_component(component)
        })
    {
        return Err(invalid_manifest_field(
            "files[].name",
            "redacted-nonportable-path",
            "must use a canonical portable relative path",
        ));
    }
    Ok(())
}

fn is_windows_reserved_path_component(component: &str) -> bool {
    let stem = component
        .split_once('.')
        .map_or(component, |(stem, _suffix)| stem);
    let upper = stem.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "CLOCK$"
            | "CONIN$"
            | "CONOUT$"
            | "COM¹"
            | "COM²"
            | "COM³"
            | "LPT¹"
            | "LPT²"
            | "LPT³"
    ) || upper
        .strip_prefix("COM")
        .or_else(|| upper.strip_prefix("LPT"))
        .is_some_and(|suffix| matches!(suffix.as_bytes(), &[b'1'..=b'9']))
}

/// Cached verification receipt stored as a small JSON file alongside model files.
///
/// When a model directory passes SHA-256 verification, a `.verified` marker is written
/// containing the exact frozen production-manifest fingerprint and lightweight file
/// fingerprints captured at verification time. Subsequent loads may skip re-hashing
/// only when both the manifest identity and every file state remain unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationMarker {
    /// Manifest identifier that was verified against.
    pub manifest_id: String,
    /// Manifest schema version at time of verification.
    pub schema_version: u32,
    /// SHA-256 of the complete canonical production manifest.
    pub manifest_fingerprint: String,
    /// Unix timestamp (seconds) when verification was performed.
    pub verified_at: u64,
    /// Per-file lightweight fingerprint at verification time, keyed by file name.
    pub file_states: BTreeMap<String, FileVerificationState>,
}

impl VerificationMarker {
    /// Create a receipt from file states captured around a successful full hash pass.
    fn from_verified_states(
        manifest: &ModelManifest,
        manifest_fingerprint: String,
        file_states: BTreeMap<String, FileVerificationState>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self {
            manifest_id: manifest.id.clone(),
            schema_version: VERIFICATION_MARKER_SCHEMA_VERSION,
            manifest_fingerprint,
            verified_at: now,
            file_states,
        }
    }

    /// Check whether this cached marker is still valid for the given manifest and directory.
    ///
    /// Returns `true` when:
    /// 1. The manifest is production-ready and its frozen fingerprint matches.
    /// 2. The manifest ID and schema version match.
    /// 3. No model file metadata fingerprint has changed since verification.
    fn is_valid_for(&self, manifest: &ModelManifest, model_dir: &Path) -> bool {
        let Ok(frozen) = manifest.freeze_verification_manifest() else {
            return false;
        };
        if self.manifest_id != manifest.id
            || self.schema_version != VERIFICATION_MARKER_SCHEMA_VERSION
            || self.manifest_fingerprint != frozen.fingerprint
            || self.file_states.len() != manifest.files.len()
        {
            return false;
        }

        for file in &manifest.files {
            let Some(expected_state) = self.file_states.get(&file.name) else {
                return false;
            };
            let Ok(path) = resolve_model_file_path(model_dir, &file.name) else {
                return false;
            };
            let Some(current_state) = capture_file_verification_state(&path) else {
                return false;
            };
            if !expected_state.still_matches(&current_state) {
                return false;
            }
        }

        true
    }
}

fn capture_manifest_file_states(
    manifest: &ModelManifest,
    model_dir: &Path,
) -> SearchResult<BTreeMap<String, FileVerificationState>> {
    let mut file_states = BTreeMap::new();
    for file in &manifest.files {
        let path = resolve_model_file_path(model_dir, &file.name)?;
        let state =
            capture_file_verification_state(&path).ok_or_else(|| SearchError::ModelNotFound {
                name: format!("{}:{}", manifest.id, file.name),
            })?;
        file_states.insert(file.name.clone(), state);
    }
    Ok(file_states)
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_AFTER_FULL_HASH_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

fn after_full_hash_boundary() {
    #[cfg(test)]
    TEST_AFTER_FULL_HASH_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

fn write_verification_marker_atomic(
    marker: &VerificationMarker,
    model_dir: &Path,
) -> SearchResult<()> {
    let encoded =
        serde_json::to_vec_pretty(marker).map_err(|source| SearchError::SubsystemError {
            subsystem: "model_verification_receipt",
            source: Box::new(source),
        })?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".verified.")
        .suffix(".tmp")
        .tempfile_in(model_dir)
        .map_err(SearchError::from)?;
    temporary.write_all(&encoded).map_err(SearchError::from)?;
    temporary.as_file().sync_all().map_err(SearchError::from)?;

    let current_states = capture_manifest_file_states_from_names(
        marker.file_states.keys().map(String::as_str),
        model_dir,
    )?;
    if current_states != marker.file_states {
        return Err(model_changed_during_verification(model_dir));
    }

    temporary
        .persist(model_dir.join(VERIFIED_MARKER_FILE))
        .map_err(|error| SearchError::from(error.error))?;
    #[cfg(unix)]
    sync_directory(model_dir)?;
    Ok(())
}

fn capture_manifest_file_states_from_names<'a>(
    file_names: impl IntoIterator<Item = &'a str>,
    model_dir: &Path,
) -> SearchResult<BTreeMap<String, FileVerificationState>> {
    let mut file_states = BTreeMap::new();
    for file_name in file_names {
        let path = resolve_model_file_path(model_dir, file_name)?;
        let state =
            capture_file_verification_state(&path).ok_or_else(|| SearchError::ModelNotFound {
                name: file_name.to_owned(),
            })?;
        file_states.insert(file_name.to_owned(), state);
    }
    Ok(file_states)
}

fn model_changed_during_verification(model_dir: &Path) -> SearchError {
    SearchError::HashMismatch {
        path: model_dir.to_path_buf(),
        expected: "stable file identity and metadata across full SHA-256 verification".to_owned(),
        actual: "model file state changed while verification was in progress".to_owned(),
    }
}

/// Check whether a valid verification marker exists for the given manifest and directory.
///
/// Returns `true` when a `.verified` file exists, parses as the current complete schema,
/// matches the exact frozen production manifest, and all recorded file states match.
///
/// The marker is a same-user performance receipt, not an authenticated artifact. Explicit
/// verification and every bundled write/promotion still perform full SHA-256 verification.
#[must_use]
pub fn is_verification_cached(manifest: &ModelManifest, model_dir: &Path) -> bool {
    let path = model_dir.join(VERIFIED_MARKER_FILE);
    let Ok(raw) = fs::read_to_string(&path) else {
        return false;
    };
    let Ok(marker) = serde_json::from_str::<VerificationMarker>(&raw) else {
        return false;
    };
    marker.is_valid_for(manifest, model_dir)
}

/// Verify a model directory, using a valid cached receipt when available.
///
/// If a valid `.verified` receipt exists (matching the exact frozen production manifest
/// and all file states), verification succeeds immediately without re-hashing. Otherwise,
/// full SHA-256 verification is performed via [`ModelManifest::verify_dir`] for this
/// invocation without mutating the cache. Non-production manifests are never admitted to
/// the cached path.
///
/// # Errors
///
/// Returns `SearchError` when the manifest is not production-ready or full verification
/// fails (hash mismatch, missing files, etc.).
pub fn verify_dir_cached(manifest: &ModelManifest, model_dir: &Path) -> SearchResult<()> {
    manifest.freeze_verification_manifest()?;

    if is_verification_cached(manifest, model_dir) {
        return Ok(());
    }

    manifest.verify_dir(model_dir)
}

/// Perform full SHA-256 verification and atomically record a cache receipt.
///
/// This is the only receipt-minting API. It captures every registered file's state before
/// hashing, performs full size-and-SHA verification, then refuses publication if any file
/// identity or metadata changed during verification. The receipt file is synced and
/// atomically renamed before its directory is synced.
///
/// External callers cannot bypass the hash pass with a raw marker writer:
///
/// ```compile_fail
/// use frankensearch_embed::write_verification_marker;
/// ```
///
/// # Errors
///
/// Returns `SearchError` when the manifest is not production-ready, any registered file
/// fails full verification, a file changes during verification, or durable receipt
/// publication fails.
pub fn verify_dir_and_record(manifest: &ModelManifest, model_dir: &Path) -> SearchResult<()> {
    let frozen = manifest.freeze_verification_manifest()?;
    let before = capture_manifest_file_states(manifest, model_dir)?;
    manifest.verify_dir(model_dir)?;
    after_full_hash_boundary();
    let after = capture_manifest_file_states(manifest, model_dir)?;
    if before != after {
        return Err(model_changed_during_verification(model_dir));
    }

    let marker = VerificationMarker::from_verified_states(manifest, frozen.fingerprint, after);
    write_verification_marker_atomic(&marker, model_dir)
}

pub(super) fn sync_registered_artifacts<'a>(
    staged_dir: &Path,
    relative_paths: impl IntoIterator<Item = &'a str>,
) -> SearchResult<()> {
    #[cfg(unix)]
    let mut parent_dirs = BTreeSet::<PathBuf>::new();
    for relative_path in relative_paths {
        let artifact_path = resolve_model_file_path(staged_dir, relative_path)?;
        // Windows FlushFileBuffers requires a handle with write access, even
        // when only flushing bytes already written to the staged artifact.
        File::options()
            .read(true)
            .write(cfg!(windows))
            .open(&artifact_path)
            .and_then(|file| file.sync_all())
            .map_err(SearchError::from)?;
        #[cfg(unix)]
        if let Some(parent) = artifact_path.parent() {
            parent_dirs.insert(parent.to_path_buf());
        }
    }
    // Windows has no portable directory fsync through std. Registered files
    // still receive sync_all before their atomic publication on every platform.
    #[cfg(unix)]
    {
        for parent in &parent_dirs {
            sync_directory(parent)?;
        }
        sync_directory(staged_dir)?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> SearchResult<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(SearchError::from)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PublicationBoundary {
    InstallingParentSync,
    BackupParentSync,
    PublishRename,
    PublishedParentSync,
}

#[cfg(test)]
thread_local! {
    static TEST_PUBLICATION_FAULT: std::cell::Cell<Option<PublicationBoundary>> =
        const { std::cell::Cell::new(None) };
}

#[allow(clippy::unnecessary_wraps)] // Test builds inject typed durability failures here.
fn publication_boundary(boundary: PublicationBoundary) -> SearchResult<()> {
    #[cfg(test)]
    if TEST_PUBLICATION_FAULT.with(|fault| fault.get() == Some(boundary)) {
        return Err(SearchError::from(std::io::Error::other(format!(
            "injected publication failure at {boundary:?}"
        ))));
    }
    #[cfg(not(test))]
    let _ = boundary;
    Ok(())
}

#[cfg(test)]
pub(super) struct PublicationFaultGuard;

#[cfg(test)]
impl PublicationFaultGuard {
    pub(super) fn install(boundary: PublicationBoundary) -> Self {
        TEST_PUBLICATION_FAULT.with(|fault| {
            assert!(
                fault.replace(Some(boundary)).is_none(),
                "publication test fault already installed"
            );
        });
        Self
    }
}

#[cfg(test)]
impl Drop for PublicationFaultGuard {
    fn drop(&mut self) {
        TEST_PUBLICATION_FAULT.with(|fault| fault.set(None));
    }
}

pub(super) fn promote_atomically(
    staged_dir: &Path,
    destination_dir: &Path,
) -> SearchResult<Option<PathBuf>> {
    let destination_parent =
        destination_dir
            .parent()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "destination_dir".to_owned(),
                value: "redacted".to_owned(),
                reason: "destination must have a parent directory".to_owned(),
            })?;
    fs::create_dir_all(destination_parent).map_err(SearchError::from)?;

    let stage_name = destination_dir.file_name().map_or_else(
        || "model".to_owned(),
        |part| part.to_string_lossy().into_owned(),
    );
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let pid = std::process::id();
    let stage_target =
        destination_parent.join(format!(".{stage_name}.installing.{timestamp}.{pid}"));
    fs::rename(staged_dir, &stage_target).map_err(SearchError::from)?;
    publication_boundary(PublicationBoundary::InstallingParentSync)?;
    #[cfg(unix)]
    sync_directory(destination_parent)?;

    let backup_path = if destination_dir.exists() {
        let backup = destination_parent.join(format!("{stage_name}.backup.{timestamp}.{pid}"));
        fs::rename(destination_dir, &backup).map_err(SearchError::from)?;
        let backup_sync = publication_boundary(PublicationBoundary::BackupParentSync);
        #[cfg(unix)]
        let backup_sync = backup_sync.and_then(|()| sync_directory(destination_parent));
        if let Err(sync_error) = backup_sync {
            if let Err(rollback_error) = fs::rename(&backup, destination_dir) {
                tracing::error!(
                    rollback_error_kind = ?rollback_error.kind(),
                    "model backup sync failed and the prior generation remains in its backup"
                );
            } else {
                #[cfg(unix)]
                let _ = sync_directory(destination_parent);
            }
            return Err(sync_error);
        }
        Some(backup)
    } else {
        None
    };

    let publish = publication_boundary(PublicationBoundary::PublishRename)
        .and_then(|()| fs::rename(&stage_target, destination_dir).map_err(SearchError::from));
    if let Err(publish_error) = publish {
        if let Some(backup) = &backup_path
            && let Err(rollback_error) = fs::rename(backup, destination_dir)
        {
            tracing::error!(
                rollback_error_kind = ?rollback_error.kind(),
                "model publication failed and prior generation remains in its backup"
            );
        }
        #[cfg(unix)]
        let _ = sync_directory(destination_parent);
        return Err(publish_error);
    }
    publication_boundary(PublicationBoundary::PublishedParentSync)?;
    #[cfg(unix)]
    sync_directory(destination_parent)?;
    Ok(backup_path)
}

pub(super) fn manifest_registry() -> &'static RwLock<BTreeMap<String, ModelManifest>> {
    static REGISTRY: OnceLock<RwLock<BTreeMap<String, ModelManifest>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let catalog = ModelManifest::builtin_catalog();
        let mut data = BTreeMap::new();
        for manifest in catalog.models {
            data.insert(manifest.id.clone(), manifest);
        }
        RwLock::new(data)
    })
}

pub(super) fn manifest_registry_lock_error(action: &str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "model_manifest",
        source: std::io::Error::other(format!("manifest registry {action} lock poisoned")).into(),
    }
}

pub(super) fn invalid_manifest_field(field: &str, value: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: truncate_for_error(value),
        reason: reason.to_owned(),
    }
}

pub(super) fn invalid_state_transition(
    state: &ModelState,
    operation: &str,
    reason: &str,
) -> SearchError {
    SearchError::InvalidConfig {
        field: "model_state".to_owned(),
        value: format!("{state:?}"),
        reason: format!("invalid transition for {operation}: {reason}"),
    }
}

pub(super) fn truncate_for_error(value: &str) -> String {
    const MAX: usize = 120;
    let mut chars = value.chars();
    let truncated: String = chars.by_ref().take(MAX).collect();
    if chars.next().is_none() {
        return truncated;
    }
    let mut out = truncated;
    out.push_str("...");
    out
}

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

pub(super) fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn to_hex_lowercase(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}
