use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io;
use std::path::{Component, Path, PathBuf};
#[cfg(test)]
use std::sync::OnceLock;
use std::sync::{Arc, Weak};
use std::time::SystemTime;

use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
use asupersync::sync::{LockError, Mutex, OwnedMutexGuard};

use crate::segment::PendingSegmentFile;

use super::admission::WriterAdmissionInner;
#[cfg(windows)]
use super::admission::open_windows_generation_claim;
#[cfg(feature = "bench-internals")]
use super::constants::BenchmarkQuillDirectorySyncState;
use super::constants::PUBLISH_LOCKS;
use super::errors::KeeperError;
use super::gc::canonical_segment_name;
use super::manifest::{LoadedManifest, Manifest};
#[cfg(test)]
use super::manifest_publish_choreography::observe_manifest_publish_checkpoint_for_test;
use super::manifest_publish_choreography::publish_manifest_choreography;
use super::recovered_segment::QuarantinedSegment;
#[cfg(not(windows))]
use super::slot_io::sync_directory;
use super::slot_io::{ManifestSlot, move_file_atomically, read_manifest_slot};
use super::writer::{PublicationAuthorityTransition, PublicationReadState};

/// Internal fault-injection points in immutable segment publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::keeper) enum SegmentPublishCheckpoint {
    /// The synced temp file was renamed to its canonical published name.
    SegmentRenamed,
    /// The index directory was fsynced after the segment rename.
    DirectorySynced,
    /// The complete-source repair sidecar was emitted and made durable.
    SidecarEmitted,
}

/// Publish one already-synced segment without durability sidecar I/O.
///
/// The first public accumulator-to-seal commit path activates this primitive
/// in the dependent G1a milestone. Until then it remains a tested Keeper
/// checkpoint building block.
#[allow(dead_code, reason = "activated by the dependent G1a commit milestone")]
pub(in crate::keeper) fn publish_pending_segment(
    pending: PendingSegmentFile,
) -> Result<PathBuf, KeeperError> {
    publish_pending_segment_with_observer(pending, |_, _| Ok(()), |_, _| Ok(None), |_, _| Ok(()))
}

fn publish_pending_segment_with_observer<B, A, O>(
    pending: PendingSegmentFile,
    mut before_rename: B,
    mut after_directory_sync: A,
    mut observe: O,
) -> Result<PathBuf, KeeperError>
where
    B: FnMut(&PendingSegmentFile, &Path) -> Result<(), KeeperError>,
    A: FnMut(&PendingSegmentFile, &Path) -> Result<Option<PathBuf>, KeeperError>,
    O: FnMut(SegmentPublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    let pending_path = pending.path().to_path_buf();
    let expected_temp = format!(".tmp-segment-{:016x}", pending.segment_id());
    if pending_path.file_name() != Some(OsStr::new(&expected_temp)) {
        return Err(KeeperError::Io {
            operation: "validate segment temp name",
            path: pending_path,
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "pending segment does not have its canonical temp name",
            ),
        });
    }
    let metadata = std::fs::symlink_metadata(&pending_path).map_err(|source| KeeperError::Io {
        operation: "inspect segment temp",
        path: pending_path.clone(),
        source,
    })?;
    if !metadata.file_type().is_file() || metadata.len() != pending.file_len() {
        return Err(KeeperError::Io {
            operation: "validate segment temp",
            path: pending_path.clone(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "pending segment is not a regular file with the encoded length",
            ),
        });
    }

    let directory = pending_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let published = directory.join(canonical_segment_name(pending.segment_id()));
    match std::fs::symlink_metadata(&published) {
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect published segment destination",
                path: published,
                source,
            });
        }
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "publish segment",
                path: published,
                source: io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "published segment destination already exists",
                ),
            });
        }
    }

    before_rename(&pending, &published)?;
    move_file_atomically(&pending_path, &published).map_err(|source| KeeperError::Io {
        operation: "rename segment temp to published",
        path: pending_path,
        source,
    })?;
    observe(SegmentPublishCheckpoint::SegmentRenamed, &published)?;
    #[cfg(not(windows))]
    sync_directory(&directory)?;
    observe(SegmentPublishCheckpoint::DirectorySynced, &directory)?;
    if let Some(sidecar) = after_directory_sync(&pending, &published)? {
        observe(SegmentPublishCheckpoint::SidecarEmitted, &sidecar)?;
    }
    // Publication consumes the temp-file capability: after rename, callers
    // must not be able to reuse the pending handle for a second publish.
    drop(pending);
    Ok(published)
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
pub(in crate::keeper) fn parse_segment_name(name: &str) -> Option<u64> {
    let hex = name.strip_prefix("seg-")?.strip_suffix(".fslx")?;
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

pub(in crate::keeper) fn parse_quarantined_segment_name(name: &str) -> Option<u64> {
    parse_segment_name(strip_quarantine_suffix(name)?)
}

pub(in crate::keeper) fn parse_quarantined_sidecar_segment_name(name: &str) -> Option<u64> {
    parse_segment_name(strip_quarantine_suffix(name)?.strip_suffix(".fec")?)
}

fn strip_quarantine_suffix(name: &str) -> Option<&str> {
    if let Some(base) = name.strip_suffix(".quarantine") {
        return Some(base);
    }
    let (base, probe) = name.rsplit_once('.')?;
    if probe.is_empty() || !probe.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    base.strip_suffix(".quarantine")
}

pub(in crate::keeper) fn discover_quarantined_segments(
    directory: &Path,
    selected: &Manifest,
) -> Result<Vec<QuarantinedSegment>, KeeperError> {
    let mut doc_counts = BTreeMap::new();
    for segment in &selected.segments {
        doc_counts.insert(segment.segment_id, u64::from(segment.doc_count));
    }
    for slot_name in ["MANIFEST", "MANIFEST.prev"] {
        let slot_path = directory.join(slot_name);
        if let ManifestSlot::Valid(manifest) = read_manifest_slot(&slot_path)? {
            for segment in manifest.segments {
                doc_counts
                    .entry(segment.segment_id)
                    .or_insert_with(|| u64::from(segment.doc_count));
            }
        }
    }

    let mut quarantined = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|source| KeeperError::Io {
        operation: "scan retained segment quarantines",
        path: directory.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| KeeperError::Io {
            operation: "read retained segment quarantine entry",
            path: directory.to_path_buf(),
            source,
        })?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(segment_id) = parse_quarantined_segment_name(&name) else {
            continue;
        };
        let file_type = entry.file_type().map_err(|source| KeeperError::Io {
            operation: "inspect retained segment quarantine",
            path: entry.path(),
            source,
        })?;
        if !file_type.is_file() {
            return Err(KeeperError::Io {
                operation: "inspect retained segment quarantine",
                path: entry.path(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "quarantined segment must be a regular file",
                ),
            });
        }
        quarantined.push(QuarantinedSegment {
            segment_id,
            path: entry.path(),
            estimated_missing_docs: doc_counts.get(&segment_id).copied(),
        });
    }
    quarantined.sort_unstable_by_key(|segment| (segment.segment_id, segment.path.clone()));
    Ok(quarantined)
}

pub(in crate::keeper) fn parse_claim_name(name: &OsStr) -> Option<u64> {
    let name = name.to_str()?;
    let digits = name.strip_prefix("gen-")?.strip_suffix(".claim")?;
    if digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    digits.parse().ok()
}

/// `rustix::fs::Stat::st_dev` carries the platform's native device type
/// (`i32` on macOS, `u64` on Linux) while `std::os::unix::fs::MetadataExt::
/// dev()` always yields the raw value widened to `u64` with a sign-extending
/// cast. Device-identity checks must apply the identical conversion or they
/// compare different bits on macOS.
#[cfg(unix)]
#[allow(clippy::unnecessary_cast, clippy::cast_sign_loss)]
pub(in crate::keeper) fn stat_dev_as_u64(stat: &rustix::fs::Stat) -> u64 {
    stat.st_dev as u64
}

pub(in crate::keeper) struct GenerationClaimGuard {
    admission: Arc<WriterAdmissionInner>,
    pub(in crate::keeper) name: OsString,
    claim_file: File,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(windows)]
    identity: same_file::Handle,
}

impl GenerationClaimGuard {
    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    pub(in crate::keeper) fn acquire(
        admission: Arc<WriterAdmissionInner>,
        generation: u64,
    ) -> Result<Self, KeeperError> {
        use rustix::fs::{AtFlags, Mode, OFlags, openat, statat};
        use std::os::unix::fs::MetadataExt;

        admission.ensure_directory_identity()?;
        let name = OsString::from(format!("gen-{generation}.claim"));
        let path = admission.directory.join(&name);
        let claim = match openat(
            &admission.directory_file,
            &name,
            OFlags::WRONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        ) {
            Ok(claim) => claim,
            Err(source) if source == rustix::io::Errno::EXIST => {
                let stat = statat(&admission.directory_file, &name, AtFlags::SYMLINK_NOFOLLOW)
                    .map_err(io::Error::from)
                    .map_err(|source| KeeperError::Io {
                        operation: "inspect conflicting generation claim",
                        path: path.clone(),
                        source,
                    })?;
                let regular = rustix::fs::FileType::from_raw_mode(stat.st_mode)
                    == rustix::fs::FileType::RegularFile;
                if !regular || stat.st_size != 0 {
                    return Err(KeeperError::InvalidClaimArtifact {
                        path,
                        detail: format!(
                            "claim must be a zero-length regular file, found type {:?} and length {}",
                            rustix::fs::FileType::from_raw_mode(stat.st_mode),
                            stat.st_size
                        ),
                    });
                }
                return Err(KeeperError::GenerationClaimConflict { path, generation });
            }
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "create O_EXCL generation claim",
                    path,
                    source: io::Error::from(source),
                });
            }
        };
        let claim_file = File::from(claim);
        let metadata = claim_file.metadata().map_err(|source| KeeperError::Io {
            operation: "inspect owned generation claim",
            path: path.clone(),
            source,
        })?;
        if !metadata.file_type().is_file() || metadata.len() != 0 {
            return Err(KeeperError::InvalidClaimArtifact {
                path,
                detail: "new claim is not a zero-length regular file".to_owned(),
            });
        }
        claim_file.sync_all().map_err(|source| KeeperError::Io {
            operation: "fsync generation claim",
            path: path.clone(),
            source,
        })?;
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync generation-claim directory",
                path: admission.directory.clone(),
                source,
            })?;
        admission.ensure_directory_identity()?;
        Ok(Self {
            admission,
            name,
            device: metadata.dev(),
            inode: metadata.ino(),
            claim_file,
        })
    }

    #[cfg(windows)]
    pub(in crate::keeper) fn acquire(
        admission: Arc<WriterAdmissionInner>,
        generation: u64,
    ) -> Result<Self, KeeperError> {
        admission.ensure_directory_identity()?;
        let name = OsString::from(format!("gen-{generation}.claim"));
        let path = admission.directory.join(&name);
        let (claim_file, identity) = match open_windows_generation_claim(&path, true) {
            Ok(claim) => claim,
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                match open_windows_generation_claim(&path, false) {
                    Ok(_) => {
                        return Err(KeeperError::GenerationClaimConflict { path, generation });
                    }
                    Err(source) => {
                        return Err(KeeperError::InvalidClaimArtifact {
                            path,
                            detail: source.to_string(),
                        });
                    }
                }
            }
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "create exclusive Windows generation claim",
                    path,
                    source,
                });
            }
        };
        claim_file.sync_all().map_err(|source| KeeperError::Io {
            operation: "persist Windows generation claim",
            path: path.clone(),
            source,
        })?;
        admission.ensure_directory_identity()?;
        Ok(Self {
            admission,
            name,
            claim_file,
            identity,
        })
    }

    #[cfg(not(any(
        windows,
        all(
            unix,
            not(any(
                target_os = "espidf",
                target_os = "horizon",
                target_os = "solaris",
                target_os = "vita",
                target_os = "wasi"
            ))
        )
    )))]
    pub(in crate::keeper) fn acquire(
        admission: Arc<WriterAdmissionInner>,
        _: u64,
    ) -> Result<Self, KeeperError> {
        Err(KeeperError::Io {
            operation: "create generation claim",
            path: admission.directory.clone(),
            source: io::Error::new(
                io::ErrorKind::Unsupported,
                "O_EXCL generation claims require Unix filesystem semantics",
            ),
        })
    }
}

impl Drop for GenerationClaimGuard {
    fn drop(&mut self) {
        release_generation_claim(self);
    }
}

#[cfg(unix)]
fn release_generation_claim(claim: &GenerationClaimGuard) {
    use rustix::fs::{AtFlags, statat, unlinkat};

    // Keep the owned descriptor alive through the identity check. If an
    // external actor replaced the pathname, leave that replacement untouched.
    let _ = claim.claim_file.metadata();
    let Ok(stat) = statat(
        &claim.admission.directory_file,
        &claim.name,
        AtFlags::SYMLINK_NOFOLLOW,
    ) else {
        return;
    };
    if stat_dev_as_u64(&stat) != claim.device || stat.st_ino != claim.inode || stat.st_size != 0 {
        return;
    }
    if unlinkat(
        &claim.admission.directory_file,
        Path::new(&claim.name),
        AtFlags::empty(),
    )
    .is_ok()
    {
        let _ = claim.admission.directory_file.sync_all();
    }
}

#[cfg(windows)]
fn release_generation_claim(claim: &GenerationClaimGuard) {
    let _ = claim.claim_file.metadata();
    if claim.admission.ensure_directory_identity().is_err() {
        return;
    }
    let path = claim.admission.directory.join(&claim.name);
    let Ok((_, observed)) = open_windows_generation_claim(&path, false) else {
        return;
    };
    if observed == claim.identity {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(not(any(unix, windows)))]
fn release_generation_claim(_: &GenerationClaimGuard) {}

/// In-process serializer for the two-slot MANIFEST publication protocol.
///
/// The owned asupersync guard moves into the blocking I/O closure. If the
/// awaiting task is cancelled after I/O starts, the guard remains held until
/// the atomic publication finishes, so a second writer cannot interleave the
/// two rename steps.
#[derive(Debug, Clone)]
pub(in crate::keeper) struct ManifestPublisher {
    pub(in crate::keeper) directory: PathBuf,
    #[cfg(test)]
    publish_lock_override: Option<Arc<Mutex<()>>>,
    publication_read_state: Option<PublicationReadState>,
    #[cfg(feature = "bench-internals")]
    benchmark_qg4_directory_sync_state: Option<Arc<BenchmarkQuillDirectorySyncState>>,
}

#[derive(Debug, Clone)]
enum ManifestProtection {
    Disabled,
}

impl ManifestPublisher {
    /// Bind a publisher to one Quill index directory.
    #[must_use]
    pub(in crate::keeper) fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            #[cfg(test)]
            publish_lock_override: None,
            publication_read_state: None,
            #[cfg(feature = "bench-internals")]
            benchmark_qg4_directory_sync_state: None,
        }
    }

    /// Bind this publisher to the reader-visible authority word for a public
    /// Quill writer. Private Keeper substrate tests intentionally leave this
    /// absent because they have no process-local reader `Arc` to coordinate.
    #[must_use]
    pub(in crate::keeper) fn with_publication_read_state(
        directory: impl Into<PathBuf>,
        publication_read_state: Option<PublicationReadState>,
    ) -> Self {
        Self {
            directory: directory.into(),
            #[cfg(test)]
            publish_lock_override: None,
            publication_read_state,
            #[cfg(feature = "bench-internals")]
            benchmark_qg4_directory_sync_state: None,
        }
    }

    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub(in crate::keeper) fn with_benchmark_qg4_directory_sync_state(
        mut self,
        state: Option<Arc<BenchmarkQuillDirectorySyncState>>,
    ) -> Self {
        self.benchmark_qg4_directory_sync_state = state;
        self
    }

    #[cfg(all(test, unix))]
    pub(in crate::keeper) fn with_publish_lock_for_test(
        directory: impl Into<PathBuf>,
        publish_lock: Arc<Mutex<()>>,
    ) -> Self {
        Self {
            directory: directory.into(),
            publish_lock_override: Some(publish_lock),
            publication_read_state: None,
            #[cfg(feature = "bench-internals")]
            benchmark_qg4_directory_sync_state: None,
        }
    }

    /// Publish one manifest using only the in-process serializer.
    ///
    /// This is a private substrate-test seam. Public mutation always goes
    /// through [`Self::publish_with_generation_claim`] while `KeeperWriter`
    /// retains `LOCK`. This ordinary path performs no sidecar I/O. If it follows
    /// a durable publication, an existing `MANIFEST.fec` may therefore be stale
    /// and must never be used as repair authority without matching the current
    /// source witness first.
    ///
    /// # Errors
    ///
    /// Returns typed validation, cancellation, transition, recovery, and I/O
    /// errors. A cancelled caller may observe an ambiguous result after the
    /// blocking phase starts; retrying is safe because generation validation
    /// reports whether the prior publication won.
    #[allow(dead_code, reason = "exercised by inline publication substrate tests")]
    pub(in crate::keeper) async fn publish(
        &self,
        cx: &Cx,
        manifest: &Manifest,
    ) -> Result<LoadedManifest, KeeperError> {
        self.publish_with_generation_claim(cx, manifest, |_, _| Ok(()))
            .await
    }

    /// Publish with a caller-supplied cross-process generation claim.
    ///
    /// `claim` runs after the temp file is durable and before either slot is
    /// renamed. Its returned guard stays alive through the directory fsync;
    /// `KeeperWriter` can therefore release its claim from `Drop`.
    ///
    /// # Errors
    ///
    /// In addition to [`Self::publish`] failures, returns any typed error from
    /// the claim callback.
    pub(in crate::keeper) async fn publish_with_generation_claim<C, F>(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        claim: F,
    ) -> Result<LoadedManifest, KeeperError>
    where
        C: Send + 'static,
        F: FnOnce(&Path, u64) -> Result<C, KeeperError> + Send + 'static,
    {
        self.publish_with_generation_claim_and_protection(
            cx,
            manifest,
            claim,
            ManifestProtection::Disabled,
        )
        .await
    }

    async fn publish_with_generation_claim_and_protection<C, F>(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        claim: F,
        protection: ManifestProtection,
    ) -> Result<LoadedManifest, KeeperError>
    where
        C: Send + 'static,
        F: FnOnce(&Path, u64) -> Result<C, KeeperError> + Send + 'static,
    {
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        // Resolve before choosing the directory-scoped lock, then carry the
        // exact resolved target through the blocking choreography. This keeps
        // a late-created symlink alias in the same serializer as its target.
        let unresolved_directory = self.directory.clone();
        let directory =
            spawn_blocking(move || normalize_publish_directory(unresolved_directory)).await;
        #[cfg(test)]
        let publish_lock = self.publish_lock(&directory);
        #[cfg(not(test))]
        let publish_lock = publish_lock_for_normalized_directory(directory.clone());
        let publication_read_state = self.publication_read_state.clone();
        #[cfg(feature = "bench-internals")]
        let benchmark_qg4_directory_sync_state = self.benchmark_qg4_directory_sync_state.clone();
        let guard = OwnedMutexGuard::lock(publish_lock, cx)
            .await
            .map_err(|source| KeeperError::PublishLock { source })?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        // Encode only after admission to the same-directory critical section.
        // Otherwise every waiter could retain its own 64 MiB output buffer.
        //
        // The publisher owns the freshness witness: a caller that left
        // `last_publish_unix_s` zero gets stamped with the current wall clock,
        // so every keeper-written generation carries the cross-process
        // visibility timestamp. An explicit caller stamp is preserved, which
        // keeps deterministic fixtures byte-stable.
        let mut manifest = manifest.clone();
        if manifest.last_publish_unix_s == 0 {
            manifest.last_publish_unix_s = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .ok()
                .and_then(|elapsed| i64::try_from(elapsed.as_secs()).ok())
                .ok_or_else(|| KeeperError::InvalidTransition {
                    detail: "current wall clock is outside the MANIFEST timestamp range".to_owned(),
                })?;
        }
        let bytes = manifest
            .to_bytes()
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        #[cfg(feature = "bench-internals")]
        let manifest_generation = manifest.generation;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        spawn_blocking(move || {
            // Use the exact target that selected the in-process serializer
            // for inspection, the generation claim, renames, and directory
            // fsync; resolving it again could split a late alias from its
            // lock after the claim boundary.
            match protection {
                ManifestProtection::Disabled => publish_manifest_locked(
                    directory,
                    &bytes,
                    guard,
                    claim,
                    publication_read_state,
                    #[cfg(feature = "bench-internals")]
                    benchmark_qg4_directory_sync_state.as_ref(),
                    #[cfg(feature = "bench-internals")]
                    manifest_generation,
                ),
            }
        })
        .await
    }

    #[cfg(all(test, unix))]
    pub(in crate::keeper) fn publish_lock_for_test(&self) -> Arc<Mutex<()>> {
        if let Some(publish_lock) = &self.publish_lock_override {
            return Arc::clone(publish_lock);
        }
        publish_lock_for_directory(&self.directory)
    }

    #[cfg(test)]
    fn publish_lock(&self, directory: &Path) -> Arc<Mutex<()>> {
        if let Some(publish_lock) = &self.publish_lock_override {
            return Arc::clone(publish_lock);
        }
        publish_lock_for_normalized_directory(directory.to_path_buf())
    }
}

#[cfg(all(test, unix))]
fn publish_lock_for_directory(directory: &Path) -> Arc<Mutex<()>> {
    let directory = normalize_publish_directory(directory.to_path_buf());
    publish_lock_for_normalized_directory(directory)
}

pub(in crate::keeper) fn publish_lock_for_normalized_directory(
    directory: PathBuf,
) -> Arc<Mutex<()>> {
    let locks = PUBLISH_LOCKS.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()));
    let mut locks = locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    locks.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = locks.get(&directory).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::with_name("quill.manifest_publish", ()));
    locks.insert(directory, Arc::downgrade(&lock));
    lock
}

pub(in crate::keeper) fn normalize_publish_directory(directory: PathBuf) -> PathBuf {
    let absolute = if directory.is_absolute() {
        directory
    } else if let Ok(current_directory) = std::env::current_dir() {
        current_directory.join(directory)
    } else {
        directory
    };
    if let Ok(canonical) = std::fs::canonicalize(&absolute) {
        return canonical;
    }

    // Canonicalize the longest existing prefix so aliases through an existing
    // symlink resolve to one stable I/O target even when the final index
    // directory has not been created yet. Normalize the missing suffix
    // lexically afterwards.
    for ancestor in absolute.ancestors().skip(1) {
        let Ok(mut canonical) = std::fs::canonicalize(ancestor) else {
            continue;
        };
        let Ok(suffix) = absolute.strip_prefix(ancestor) else {
            continue;
        };
        canonical.push(suffix);
        return lexical_normalize(&canonical);
    }
    lexical_normalize(&absolute)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if normalized
                    .file_name()
                    .is_some_and(|part| part != std::ffi::OsStr::new(".."))
                {
                    let _ = normalized.pop();
                } else if !path.is_absolute() {
                    normalized.push("..");
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

#[cfg(feature = "bench-internals")]
fn observe_benchmark_qg4_directory_sync(
    state: Option<&Arc<BenchmarkQuillDirectorySyncState>>,
    checkpoint: PublishCheckpoint,
    generation: u64,
) {
    if checkpoint == PublishCheckpoint::DirectorySynced {
        if let Some(state) = state {
            state.observe_successful_manifest_sync(generation);
        }
    }
}

fn publish_manifest_locked<C, F>(
    directory: PathBuf,
    bytes: &[u8],
    _guard: OwnedMutexGuard<()>,
    claim: F,
    publication_read_state: Option<PublicationReadState>,
    #[cfg(feature = "bench-internals")] benchmark_qg4_directory_sync_state: Option<
        &Arc<BenchmarkQuillDirectorySyncState>,
    >,
    #[cfg(feature = "bench-internals")] manifest_generation: u64,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
{
    let transition = publication_read_state
        .map(|state| {
            let proposed = Manifest::from_bytes(bytes)
                .map_err(|source| KeeperError::InvalidManifest { source })?;
            PublicationAuthorityTransition::begin(state, proposed.generation)
        })
        .transpose()?;
    #[cfg(test)]
    {
        let observed_directory = directory.clone();
        let result = publish_manifest_choreography(directory, bytes, claim, |checkpoint, _| {
            if let Some(transition) = transition.as_ref() {
                match checkpoint {
                    PublishCheckpoint::GenerationClaimed => transition.mark_indeterminate(),
                    PublishCheckpoint::TempMovedToCurrent => transition.mark_durable_ahead(),
                    _ => {}
                }
            }
            #[cfg(feature = "bench-internals")]
            observe_benchmark_qg4_directory_sync(
                benchmark_qg4_directory_sync_state,
                checkpoint,
                manifest_generation,
            );
            observe_manifest_publish_checkpoint_for_test(&observed_directory, checkpoint)
        });
        if result.is_err() {
            if let Some(transition) = transition.as_ref() {
                transition.restore_known_pre_rename_failure();
            }
        }
        result
    }
    #[cfg(not(test))]
    {
        let result = publish_manifest_choreography(directory, bytes, claim, |checkpoint, _| {
            if let Some(transition) = transition.as_ref() {
                match checkpoint {
                    PublishCheckpoint::GenerationClaimed => transition.mark_indeterminate(),
                    PublishCheckpoint::TempMovedToCurrent => transition.mark_durable_ahead(),
                    _ => {}
                }
            }
            #[cfg(feature = "bench-internals")]
            observe_benchmark_qg4_directory_sync(
                benchmark_qg4_directory_sync_state,
                checkpoint,
                manifest_generation,
            );
            Ok(())
        });
        if result.is_err() {
            if let Some(transition) = transition.as_ref() {
                transition.restore_known_pre_rename_failure();
            }
        }
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishCheckpoint {
    TempWritten,
    TempSynced,
    GenerationClaimed,
    CurrentMovedToPrevious,
    TempMovedToCurrent,
    DirectorySynced,
}

#[cfg(test)]
#[derive(Clone)]
pub(in crate::keeper) struct ManifestPublishPause {
    pub(in crate::keeper) checkpoint: PublishCheckpoint,
    pub(in crate::keeper) event: Arc<std::sync::Mutex<ManifestPublishCheckpointEvent>>,
    pub(in crate::keeper) released: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

/// One injected, typed pre-rename publisher failure for a real public-path
/// test. This is deliberately separate from the pause map: failure returns to
/// the blocking closure instead of parking it.
#[cfg(test)]
#[derive(Clone)]
pub(in crate::keeper) struct ManifestPublishFailure {
    pub(in crate::keeper) checkpoint: PublishCheckpoint,
}

/// Test-only checkpoint state shared by the blocking publisher and its async
/// observer. The observer retains a task waker so a blocking filesystem phase
/// can resume the test executor without a scheduler-yield assumption.
#[cfg(test)]
#[derive(Default)]
pub(in crate::keeper) struct ManifestPublishCheckpointEvent {
    pub(in crate::keeper) reached: bool,
    pub(in crate::keeper) waker: Option<std::task::Waker>,
}

/// One-shot control for pausing real MANIFEST choreography in cancellation
/// tests after a named filesystem checkpoint has completed.
#[cfg(test)]
pub struct ManifestPublishPauseControl {
    pub(in crate::keeper) directory: PathBuf,
    pub(in crate::keeper) checkpoint: PublishCheckpoint,
    pub(in crate::keeper) event: Arc<std::sync::Mutex<ManifestPublishCheckpointEvent>>,
    pub(in crate::keeper) released: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

/// Removes one test-only fault injection when dropped.
#[cfg(test)]
pub struct ManifestPublishFailureControl {
    pub(in crate::keeper) directory: PathBuf,
}

#[cfg(test)]
static MANIFEST_PUBLISH_PAUSES: OnceLock<
    std::sync::Mutex<BTreeMap<PathBuf, ManifestPublishPause>>,
> = OnceLock::new();

#[cfg(test)]
pub(in crate::keeper) static MANIFEST_PUBLISH_FAILURES: OnceLock<
    std::sync::Mutex<BTreeMap<PathBuf, ManifestPublishFailure>>,
> = OnceLock::new();

/// Serializes tests that arm the global MANIFEST publisher pause. The
/// production publisher itself is process-serialized; this test-only guard
/// prevents independently scheduled test runtimes from waiting behind a
/// different paused publisher before they can reach their own checkpoint.
#[cfg(test)]
pub(in crate::keeper) static MANIFEST_PUBLISH_PAUSE_TEST_SERIAL: OnceLock<std::sync::Mutex<()>> =
    OnceLock::new();

#[cfg(test)]
pub(in crate::keeper) fn manifest_publish_pauses()
-> &'static std::sync::Mutex<BTreeMap<PathBuf, ManifestPublishPause>> {
    MANIFEST_PUBLISH_PAUSES.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()))
}
