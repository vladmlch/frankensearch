use std::collections::BTreeSet;
#[cfg(unix)]
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsStr;
#[cfg(unix)]
use std::ffi::OsString;
use std::fs::File;
#[cfg(unix)]
use std::io;
#[cfg(unix)]
use std::io::Read;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::schema::SchemaDescriptor;

use super::constants::DEFAULT_GARBAGE_GRACE;
use super::errors::KeeperError;
use super::manifest::Manifest;
#[cfg(unix)]
use super::manifest::ManifestSource;
use super::publish::{parse_claim_name, parse_segment_name};
#[cfg(not(unix))]
use super::slot_io::ensure_atomic_publish_supported;
#[cfg(unix)]
use super::slot_io::read_manifest_slot_at;
#[cfg(not(windows))]
use super::slot_io::sync_directory;
use super::slot_io::{ManifestSlot, read_manifest_slot, replace_file_atomically};
#[cfg(unix)]
use super::snapshot::KeeperSnapshot;

/// Writer-owned garbage collection configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GarbageCollectionOptions {
    /// Minimum artifact age required before deletion.
    pub grace_period: Duration,
}

impl Default for GarbageCollectionOptions {
    fn default() -> Self {
        Self {
            grace_period: DEFAULT_GARBAGE_GRACE,
        }
    }
}

/// Deterministic report from one writer-locked garbage sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GarbageCollectionReport {
    /// Removed direct-child names, sorted bytewise by platform string order.
    pub removed: Vec<PathBuf>,
}

impl GarbageCollectionReport {
    /// Whether the sweep made no filesystem changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.removed.is_empty()
    }
}
/// Remove unreachable Quill artifacts while the crate-internal caller holds
/// the writer lock.
///
/// This seam stays crate-private because callers must already hold the
/// `KeeperWriter` admission that owns the directory handle. The function first
/// opens and validates the selected snapshot, so a failed recovery never
/// performs garbage collection. Reader [`super::snapshot::KeeperSnapshot::open`] never calls it.
/// Reachability is the union of every individually valid MANIFEST slot.
///
/// # Errors
///
/// Returns before deletion when recovery, directory enumeration, metadata, or
/// path-safety validation fails. An unlink or final directory fsync failure is
/// reported with the exact affected path.
pub(in crate::keeper) fn collect_writer_garbage_under_lock(
    directory: impl AsRef<Path>,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_writer_garbage_at(directory.as_ref(), schema, options, SystemTime::now())
}

/// [`collect_writer_garbage_under_lock`] with an injectable observation
/// clock, so crate tests can prove grace behaviour without waiting it out.
pub fn collect_writer_garbage_at(
    directory: &Path,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_writer_garbage_at_platform(directory, schema, options, now)
}

#[cfg(not(unix))]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_writer_garbage_at_platform(
    directory: &Path,
    _: SchemaDescriptor,
    _: GarbageCollectionOptions,
    _: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    ensure_atomic_publish_supported(directory)?;
    unreachable!("unsupported-platform guard always returns an error")
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_writer_garbage_at_platform(
    directory: &Path,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let snapshot = KeeperSnapshot::open(directory, schema)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    sweep_writer_garbage(&directory_file, directory, &snapshot, options, now)
}

/// [`collect_writer_garbage_under_lock`] judged against a snapshot the caller
/// opened under the same writer admission, instead of reopening one, which
/// verifies every segment again. The sweep reads both MANIFEST slots itself
/// and refuses to run unless the selected one still equals the snapshot's.
#[cfg(unix)]
pub(in crate::keeper) fn collect_writer_garbage_with_snapshot(
    directory: &Path,
    snapshot: &KeeperSnapshot,
    options: GarbageCollectionOptions,
) -> Result<GarbageCollectionReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    sweep_writer_garbage(
        &directory_file,
        directory,
        snapshot,
        options,
        SystemTime::now(),
    )
}

#[cfg(unix)]
fn sweep_writer_garbage(
    directory_file: &File,
    directory: &Path,
    snapshot: &KeeperSnapshot,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    let mut reachability = live_segment_names_at(directory_file, directory, snapshot)?;
    let current_generation = snapshot.loaded_manifest().manifest.generation;
    reachability.unreachable_since =
        segment_unreachability_floor_at(directory_file, directory, snapshot, now)?;
    sweep_garbage_directory(
        directory_file,
        directory,
        &reachability,
        current_generation,
        options,
        now,
    )
}

/// Durable record of one orphan-witnessing pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::keeper) struct OrphanWitnessReport {
    /// Segments that gained a first-sighting receipt, sorted bytewise.
    pub(in crate::keeper) witnessed: Vec<PathBuf>,
}

/// Stamp first-sighting receipts for segments no publication ever retired.
///
/// A segment that was staged into its canonical name and then abandoned by a
/// crash was never referenced by a published manifest, so no publication ever
/// observed it becoming unreachable and no retirement receipt exists. Past the
/// supersession threshold the sweep refuses such a segment forever, because by
/// inspection it is indistinguishable from a retired one: the durable state
/// carries no field that separates them (see the rejected routes recorded on
/// bd-u1efe). This pass supplies the missing witness -- an admitted writer
/// observing, at a durable instant, that no slot references the segment -- so
/// the grace period still runs from an observation rather than from the
/// segment's own mtime. Collection remains the sweep's decision alone; this
/// pass never unlinks anything.
///
/// # Preconditions
///
/// The caller must hold the writer admission **and** have no segment staged
/// but not yet published. A freshly staged segment is byte-for-byte
/// indistinguishable from an orphan, so witnessing while one is in flight
/// would stamp a segment that is about to become live. `KeeperWriter::open`
/// satisfies this by construction: it has published nothing of its own yet.
/// A later retirement of the same segment overwrites the sighting with the
/// stronger witness, so a stale sighting cannot shorten a real retirement's
/// grace.
///
/// # Errors
///
/// Returns before writing anything when recovery, directory identity, or
/// enumeration fails.
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
pub(in crate::keeper) fn witness_orphaned_segments_under_lock(
    directory: impl AsRef<Path>,
    schema: SchemaDescriptor,
) -> Result<OrphanWitnessReport, KeeperError> {
    witness_orphaned_segments_at(directory.as_ref(), schema, SystemTime::now())
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
pub(in crate::keeper) fn witness_orphaned_segments_at(
    directory: &Path,
    schema: SchemaDescriptor,
    now: SystemTime,
) -> Result<OrphanWitnessReport, KeeperError> {
    witness_orphaned_segments_at_platform(directory, schema, now)
}

#[cfg(not(unix))]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn witness_orphaned_segments_at_platform(
    directory: &Path,
    _: SchemaDescriptor,
    _: SystemTime,
) -> Result<OrphanWitnessReport, KeeperError> {
    ensure_atomic_publish_supported(directory)?;
    unreachable!("unsupported-platform guard always returns an error")
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn witness_orphaned_segments_at_platform(
    directory: &Path,
    schema: SchemaDescriptor,
    now: SystemTime,
) -> Result<OrphanWitnessReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let snapshot = KeeperSnapshot::open(directory, schema)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    witness_orphans_against(&directory_file, directory, &snapshot, now)
}

/// [`witness_orphaned_segments_under_lock`] judged against a snapshot the
/// caller opened under the same writer admission, instead of reopening one,
/// which verifies every segment again. It reads both MANIFEST slots itself
/// and refuses unless the selected one still equals the snapshot's.
#[cfg(unix)]
pub(in crate::keeper) fn witness_orphaned_segments_with_snapshot(
    directory: &Path,
    snapshot: &KeeperSnapshot,
) -> Result<OrphanWitnessReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    witness_orphans_against(&directory_file, directory, snapshot, SystemTime::now())
}

#[cfg(unix)]
fn witness_orphans_against(
    directory_file: &File,
    directory: &Path,
    snapshot: &KeeperSnapshot,
    now: SystemTime,
) -> Result<OrphanWitnessReport, KeeperError> {
    use rustix::fs::{AtFlags, Dir, FileType, statat};
    use std::os::unix::ffi::OsStringExt;

    let reachability = live_segment_names_at(directory_file, directory, snapshot)?;
    // Below the supersession threshold an unreferenced segment is provably
    // never-published and the sweep already reclaims it on file age, so a
    // witness would be pure churn: it would create a receipt only to delete
    // it again with the segment.
    if !reachability.retirement_provenance_required {
        return Ok(OrphanWitnessReport {
            witnessed: Vec::new(),
        });
    }
    let generation = snapshot.loaded_manifest().manifest.generation;
    let witnessed_unix_s = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .and_then(|elapsed| i64::try_from(elapsed.as_secs()).ok())
        .ok_or_else(|| KeeperError::InvalidTransition {
            detail: "observation time is outside the sighting receipt range".to_owned(),
        })?;

    let mut unreferenced = BTreeSet::<u64>::new();
    let mut covered = BTreeSet::<u64>::new();
    let mut entries = Dir::read_from(directory_file)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "scan orphan candidates",
            path: directory.to_path_buf(),
            source,
        })?;
    while let Some(entry) = entries.read() {
        let entry = entry
            .map_err(io::Error::from)
            .map_err(|source| KeeperError::Io {
                operation: "read orphan candidate",
                path: directory.to_path_buf(),
                source,
            })?;
        let name = OsString::from_vec(entry.file_name().to_bytes().to_vec());
        let Some(candidate) = classify_garbage_candidate(&name) else {
            continue;
        };
        if !matches!(
            candidate,
            GarbageCandidate::Segment | GarbageCandidate::RetirementReceipt { .. }
        ) {
            continue;
        }
        let candidate_path = directory.join(&name);
        let stat = match statat(directory_file, entry.file_name(), AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => stat,
            Err(source) if source == rustix::io::Errno::NOENT => continue,
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "inspect orphan candidate",
                    path: candidate_path,
                    source: io::Error::from(source),
                });
            }
        };
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
            continue;
        }
        match candidate {
            GarbageCandidate::Segment => {
                if !reachability.live.contains(&name)
                    && let Some(segment_id) = name.to_str().and_then(parse_segment_name)
                {
                    unreferenced.insert(segment_id);
                }
            }
            GarbageCandidate::RetirementReceipt { segment_id, .. } => {
                // Only a decodable, correctly bound receipt counts as
                // covering its segment. An unusable one leaves the segment
                // witness-less, and this pass supplies the witness the sweep
                // will then require.
                if let Some(receipt) =
                    read_retirement_receipt_at(directory_file, &name, &candidate_path, &stat)?
                    && receipt.segment_id == segment_id
                {
                    covered.insert(segment_id);
                }
            }
            GarbageCandidate::Temporary
            | GarbageCandidate::Sidecar { .. }
            | GarbageCandidate::Claim { .. } => {}
        }
    }

    let mut witnessed = Vec::new();
    for segment_id in unreferenced.difference(&covered) {
        write_unreachability_receipt(
            directory,
            SegmentRetirementReceipt {
                segment_id: *segment_id,
                retired_unix_s: witnessed_unix_s,
                retired_generation: generation,
                witness: UnreachabilityWitness::FirstSightedByWriter,
            },
        )?;
        witnessed.try_reserve(1).map_err(|error| KeeperError::Io {
            operation: "allocate orphan witness report",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
        witnessed.push(PathBuf::from(retirement_receipt_name(*segment_id)));
    }
    if !witnessed.is_empty() {
        sync_gc_directory(directory_file, directory)?;
    }
    witnessed.sort_unstable();
    Ok(OrphanWitnessReport { witnessed })
}

/// Remove abandoned genesis artifacts when both durable slots are absent.
///
/// This is a separate writer-locked state because ordinary snapshot recovery
/// correctly returns [`KeeperError::IndexNotFound`] when neither slot exists.
/// Any canonical generation claim blocks deletion for writer-open recovery to
/// resolve. This helper remains reserved for a future writer-only doctor path.
#[allow(dead_code, reason = "reserved for a writer-only doctor path")]
pub(in crate::keeper) fn collect_abandoned_genesis_garbage_under_lock(
    directory: impl AsRef<Path>,
    options: GarbageCollectionOptions,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_abandoned_genesis_garbage_at(directory.as_ref(), options, SystemTime::now())
}

#[allow(dead_code, reason = "reserved for a writer-only doctor path")]
pub(in crate::keeper) fn collect_abandoned_genesis_garbage_at(
    directory: &Path,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_abandoned_genesis_garbage_at_platform(directory, options, now)
}

#[cfg(not(unix))]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_abandoned_genesis_garbage_at_platform(
    directory: &Path,
    _: GarbageCollectionOptions,
    _: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    ensure_atomic_publish_supported(directory)?;
    unreachable!("unsupported-platform guard always returns an error")
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_abandoned_genesis_garbage_at_platform(
    directory: &Path,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    for slot_name in ["MANIFEST", "MANIFEST.prev"] {
        let path = directory.join(slot_name);
        match read_manifest_slot_at(&directory_file, OsStr::new(slot_name), &path)? {
            ManifestSlot::Missing => {}
            ManifestSlot::Valid(_) => return Err(KeeperError::RecoveryRequired { path }),
            ManifestSlot::Invalid(source) => {
                return Err(KeeperError::ManifestCorrupted { path, source });
            }
        }
    }
    ensure_gc_directory_identity(directory, &directory_file)?;
    // Genesis has no slot at all, so no publication ever retired anything and
    // no receipt can exist: everything present is abandoned in-flight work.
    let reachability = SegmentReachability {
        live: HashSet::new(),
        unreachable_since: None,
        retirement_provenance_required: false,
    };
    sweep_garbage_directory(&directory_file, directory, &reachability, 0, options, now)
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sweep_garbage_directory(
    directory_file: &File,
    directory: &Path,
    reachability: &SegmentReachability,
    current_generation: u64,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    use rustix::fs::{AtFlags, Dir, FileType, statat, unlinkat};
    use std::os::unix::ffi::OsStringExt;

    let live_segments = &reachability.live;
    let mut receipts = BTreeMap::<u64, SegmentRetirementReceipt>::new();
    let mut present_segments = HashSet::<OsString>::new();
    let mut candidates = Vec::<(OsString, GarbageCandidate, rustix::fs::Stat)>::new();
    let mut entries = Dir::read_from(directory_file)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "scan garbage candidates",
            path: directory.to_path_buf(),
            source,
        })?;
    while let Some(entry) = entries.read() {
        let entry = entry
            .map_err(io::Error::from)
            .map_err(|source| KeeperError::Io {
                operation: "read garbage candidate",
                path: directory.to_path_buf(),
                source,
            })?;
        let name = OsString::from_vec(entry.file_name().to_bytes().to_vec());
        let Some(candidate) = classify_garbage_candidate(&name) else {
            continue;
        };
        let candidate_path = directory.join(&name);
        let stat = match statat(directory_file, entry.file_name(), AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => stat,
            Err(source) if source == rustix::io::Errno::NOENT => continue,
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "inspect garbage candidate",
                    path: candidate_path,
                    source: io::Error::from(source),
                });
            }
        };
        let is_regular = FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile;
        if let GarbageCandidate::Claim { generation } = &candidate {
            if !is_regular || stat.st_size != 0 {
                return Err(KeeperError::InvalidClaimArtifact {
                    path: candidate_path,
                    detail: format!(
                        "claim must be a zero-length regular file, found type {:?} and length {}",
                        FileType::from_raw_mode(stat.st_mode),
                        stat.st_size
                    ),
                });
            }
            if *generation > current_generation {
                return Err(KeeperError::ClaimedGenerationPending {
                    path: candidate_path,
                    current: current_generation,
                    claimed: *generation,
                });
            }
        }
        if is_regular {
            if let GarbageCandidate::RetirementReceipt { segment_id, .. } = &candidate
                && let Some(receipt) =
                    read_retirement_receipt_at(directory_file, &name, &candidate_path, &stat)?
                && receipt.segment_id == *segment_id
            {
                receipts.insert(receipt.segment_id, receipt);
            }
            if matches!(candidate, GarbageCandidate::Segment) {
                present_segments
                    .try_reserve(1)
                    .map_err(|error| KeeperError::Io {
                        operation: "allocate present segment inventory",
                        path: directory.to_path_buf(),
                        source: io::Error::other(error.to_string()),
                    })?;
                present_segments.insert(name.clone());
            }
            candidates.try_reserve(1).map_err(|error| KeeperError::Io {
                operation: "allocate garbage candidate inventory",
                path: directory.to_path_buf(),
                source: io::Error::other(error.to_string()),
            })?;
            candidates.push((name, candidate, stat));
        }
    }

    let mut removable_segments = HashSet::<OsString>::new();
    removable_segments
        .try_reserve(candidates.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate removable segment inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for (name, candidate, stat) in &candidates {
        if matches!(candidate, GarbageCandidate::Segment)
            && !live_segments.contains(name)
            && segment_grace_served(
                reachability,
                name,
                stat,
                &receipts,
                now,
                options.grace_period,
            )
        {
            removable_segments.insert(name.clone());
        }
    }

    let mut removals = Vec::<OsString>::new();
    removals
        .try_reserve_exact(candidates.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate garbage removal inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for (name, candidate, stat) in candidates {
        let old_enough = stat_old_enough(&stat, now, options.grace_period);
        let remove = match candidate {
            GarbageCandidate::Temporary => old_enough,
            GarbageCandidate::Segment => removable_segments.contains(&name),
            GarbageCandidate::Sidecar { base } => {
                old_enough
                    && sidecar_is_orphan_at(directory_file, directory, &removable_segments, &base)?
            }
            GarbageCandidate::Claim { .. } => old_enough,
            // A receipt is reclaimed with the segment it authorized, and
            // otherwise only once it is proven stale: an aborted publication
            // can leave one behind for a segment that stayed reachable, and a
            // partially completed sweep can leave one behind with no segment
            // at all. A receipt that does not decode is left in place as
            // corruption evidence; it authorizes nothing either way.
            GarbageCandidate::RetirementReceipt { base, segment_id } => {
                removable_segments.contains(&base)
                    || (old_enough
                        && receipts.contains_key(&segment_id)
                        && (live_segments.contains(&base) || !present_segments.contains(&base)))
            }
        };
        if remove {
            removals.push(name);
        }
    }
    // Bytewise order removes `seg-<id>.fslx` before `seg-<id>.fslx.retired`,
    // so an interrupted sweep can only ever lose the receipt after its
    // segment, never the segment while its authorization survives.
    removals.sort_unstable();

    let mut removed = Vec::new();
    removed
        .try_reserve_exact(removals.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate garbage report",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for name in removals {
        let relative = Path::new(&name);
        let path = safe_direct_child(directory, relative)?;
        match unlinkat(directory_file, relative, AtFlags::empty()) {
            Ok(()) => removed.push(relative.to_path_buf()),
            Err(source) if source == rustix::io::Errno::NOENT => {}
            Err(source) => {
                if !removed.is_empty() {
                    sync_gc_directory(directory_file, directory)?;
                }
                return Err(KeeperError::Io {
                    operation: "remove garbage candidate",
                    path,
                    source: io::Error::from(source),
                });
            }
        }
    }
    if !removed.is_empty() {
        sync_gc_directory(directory_file, directory)?;
    }
    Ok(GarbageCollectionReport { removed })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
pub(in crate::keeper) fn segment_unreachability_floor_at(
    directory_file: &File,
    directory: &Path,
    snapshot: &KeeperSnapshot,
    _observation_time: SystemTime,
) -> Result<Option<SystemTime>, KeeperError> {
    use rustix::fs::{AtFlags, FileType, fstat, statat};

    let source = snapshot.loaded_manifest().source;
    match source {
        ManifestSource::PreviousAfterCorruptCurrent => {
            // An invalid current inode does not reveal when latent or silent
            // content corruption was first observed. Its ctime/mtime may be
            // arbitrarily older than the moment its segments became
            // unreachable to recovery, so no timestamp-derived grace floor is
            // admissible. Fail before the sweep until writer recovery
            // durably republishes a valid current slot.
            return Err(KeeperError::RecoveryRequired {
                path: directory.join("MANIFEST"),
            });
        }
        ManifestSource::InMemory => {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
        ManifestSource::PreviousAfterMissingCurrent | ManifestSource::Current => {}
    }

    if source != ManifestSource::Current {
        // The namespace mutation that removed or replaced MANIFEST is the
        // durable recovery-window witness. Unlike the caller's observation
        // time, it remains stable across repeated GC attempts. Include the
        // selected previous inode so the inferred floor cannot predate its
        // publication. Unrelated directory mutations may postpone
        // reclamation, but cannot accelerate it.
        let directory_stat = fstat(directory_file)
            .map_err(io::Error::from)
            .map_err(|source| KeeperError::Io {
                operation: "inspect recovery GC directory witness",
                path: directory.to_path_buf(),
                source,
            })?;
        if FileType::from_raw_mode(directory_stat.st_mode) != FileType::Directory {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
        let mut changed = required_gc_witness_time(
            &directory_stat,
            directory,
            "decode recovery GC directory timestamp",
        )?;

        let previous_path = directory.join("MANIFEST.prev");
        let previous_stat = statat(directory_file, "MANIFEST.prev", AtFlags::SYMLINK_NOFOLLOW)
            .map_err(io::Error::from)
            .map_err(|source| KeeperError::Io {
                operation: "inspect recovery MANIFEST.prev witness",
                path: previous_path.clone(),
                source,
            })?;
        if FileType::from_raw_mode(previous_stat.st_mode) != FileType::RegularFile {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
        changed = std::cmp::max(
            changed,
            required_gc_witness_time(
                &previous_stat,
                &previous_path,
                "decode recovery MANIFEST.prev timestamp",
            )?,
        );
        return Ok(Some(changed));
    }

    let path = directory.join("MANIFEST");
    let stat = statat(directory_file, "MANIFEST", AtFlags::SYMLINK_NOFOLLOW)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "inspect MANIFEST unreachability witness",
            path: path.clone(),
            source,
        })?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(KeeperError::GarbageDirectoryChanged {
            directory: directory.to_path_buf(),
        });
    }

    // Renaming the freshly synced current MANIFEST advances its inode ctime,
    // even when publication reuses an older temp whose mtime and embedded
    // informational timestamp predate the rename. The embedded witness may be
    // later (for example after a wall-clock step), so use the later boundary.
    let changed = required_gc_witness_time(&stat, &path, "decode MANIFEST GC timestamp")?;
    let manifest = &snapshot.loaded_manifest().manifest;
    let published = if manifest.last_publish_unix_s > 0 {
        Some(
            manifest_publish_time(manifest).ok_or_else(|| KeeperError::Io {
                operation: "decode MANIFEST publish timestamp",
                path,
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "MANIFEST publish timestamp is outside the SystemTime domain",
                ),
            })?,
        )
    } else {
        None
    };
    Ok(Some(published.map_or(changed, |published| {
        std::cmp::max(changed, published)
    })))
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
pub(in crate::keeper) fn required_gc_witness_time(
    stat: &rustix::fs::Stat,
    path: &Path,
    operation: &'static str,
) -> Result<SystemTime, KeeperError> {
    let changed = stat_change_time(stat).ok_or_else(|| KeeperError::Io {
        operation,
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::InvalidData,
            "filesystem change timestamp is outside the SystemTime domain",
        ),
    })?;
    let modified = stat_modified_time(stat).ok_or_else(|| KeeperError::Io {
        operation,
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::InvalidData,
            "filesystem modification timestamp is outside the SystemTime domain",
        ),
    })?;
    Ok(std::cmp::max(changed, modified))
}

/// What the durable slot pair proves about segment reachability for one sweep.
#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
#[derive(Debug)]
struct SegmentReachability {
    /// Canonical names referenced by any individually valid slot.
    live: HashSet<OsString>,
    /// Conservative manifest-level floor for the recovery window, when the
    /// selected slot supplies one: the latest durable slot mutation, which
    /// dates every unreachability transition from above.
    ///
    /// It gates only segments below the supersession threshold, where no
    /// retirement receipt can exist. Once `retirement_provenance_required`
    /// holds, each segment's receipt is the exact witness of its own
    /// transition and the floor is not consulted: a later publication cannot
    /// re-reference a retired segment, so it must not restart the segment's
    /// grace (see [`segment_grace_served`]).
    unreachable_since: Option<SystemTime>,
    /// Whether any generation could already have been superseded, and with it
    /// whether an unreferenced segment may be a *retired* segment rather than
    /// a never-published crash artifact.
    ///
    /// Generations run from 1 and the durable window holds only the distinct
    /// generations its valid slots name, so a directory whose selected
    /// generation does not exceed that count has never dropped a published
    /// manifest: nothing it fails to reference was ever reachable, and no
    /// first-unreachable receipt can exist to be demanded. Past that point an
    /// unreferenced segment is indistinguishable from a retired one by
    /// inspection alone, so deletion requires the durable receipt the
    /// retiring publication wrote. Counting *distinct* generations matters:
    /// two slots naming the same generation cover one, not two.
    retirement_provenance_required: bool,
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn live_segment_names_at(
    directory_file: &File,
    directory: &Path,
    snapshot: &KeeperSnapshot,
) -> Result<SegmentReachability, KeeperError> {
    let mut live = HashSet::new();
    let mut covered_generations = BTreeSet::new();
    for slot_name in ["MANIFEST", "MANIFEST.prev"] {
        let path = directory.join(slot_name);
        if let ManifestSlot::Valid(manifest) =
            read_manifest_slot_at(directory_file, OsStr::new(slot_name), &path)?
        {
            covered_generations.insert(manifest.generation);
            live.try_reserve(manifest.segments.len())
                .map_err(|error| KeeperError::Io {
                    operation: "allocate live segment set",
                    path: directory.to_path_buf(),
                    source: io::Error::other(error.to_string()),
                })?;
            live.extend(
                manifest
                    .segments
                    .iter()
                    .map(|segment| OsString::from(canonical_segment_name(segment.segment_id))),
            );
        }
    }
    let selected_name = match snapshot.loaded_manifest().source {
        ManifestSource::Current => "MANIFEST",
        ManifestSource::PreviousAfterMissingCurrent
        | ManifestSource::PreviousAfterCorruptCurrent => "MANIFEST.prev",
        ManifestSource::InMemory => {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
    };
    let selected_path = directory.join(selected_name);
    match read_manifest_slot_at(directory_file, OsStr::new(selected_name), &selected_path)? {
        ManifestSlot::Valid(manifest) if manifest == snapshot.loaded_manifest().manifest => {}
        ManifestSlot::Missing | ManifestSlot::Invalid(_) | ManifestSlot::Valid(_) => {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
    }
    let covered = u64::try_from(covered_generations.len()).map_err(|error| KeeperError::Io {
        operation: "count covered manifest generations",
        path: directory.to_path_buf(),
        source: io::Error::other(error.to_string()),
    })?;
    Ok(SegmentReachability {
        live,
        unreachable_since: None,
        retirement_provenance_required: snapshot.loaded_manifest().manifest.generation > covered,
    })
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
#[derive(Debug)]
pub(in crate::keeper) enum GarbageCandidate {
    Temporary,
    Segment,
    Sidecar {
        base: OsString,
    },
    Claim {
        generation: u64,
    },
    /// One segment's first-unreachable receipt. It authorizes deletion of its
    /// own segment and nothing else, and is itself reclaimed with that
    /// segment or once it is proven stale.
    RetirementReceipt {
        base: OsString,
        segment_id: u64,
    },
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
pub(in crate::keeper) fn classify_garbage_candidate(name: &OsStr) -> Option<GarbageCandidate> {
    let name = name.to_str()?;
    if name.starts_with(".tmp-") {
        return Some(GarbageCandidate::Temporary);
    }
    if let Some(base) = name.strip_suffix(".fec.tmp")
        && (parse_segment_name(base).is_some() || matches!(base, "MANIFEST" | "MANIFEST.prev"))
    {
        return Some(GarbageCandidate::Temporary);
    }
    if parse_segment_name(name).is_some() {
        return Some(GarbageCandidate::Segment);
    }
    if let Some(base) = name.strip_suffix(".fec")
        && (parse_segment_name(base).is_some() || matches!(base, "MANIFEST" | "MANIFEST.prev"))
    {
        return Some(GarbageCandidate::Sidecar {
            base: OsString::from(base),
        });
    }
    if let Some(base) = name.strip_suffix(".retired")
        && let Some(segment_id) = parse_segment_name(base)
    {
        return Some(GarbageCandidate::RetirementReceipt {
            base: OsString::from(base),
            segment_id,
        });
    }
    parse_claim_name(OsStr::new(name)).map(|generation| GarbageCandidate::Claim { generation })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sidecar_is_orphan_at(
    directory_file: &File,
    directory: &Path,
    removable_segments: &HashSet<OsString>,
    base: &OsStr,
) -> Result<bool, KeeperError> {
    use rustix::fs::{AtFlags, statat};

    let base_text = base.to_str();
    if base_text.is_some_and(|name| parse_segment_name(name).is_some())
        && removable_segments.contains(base)
    {
        return Ok(true);
    }
    match statat(directory_file, base, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(_) => Ok(false),
        Err(source) if source == rustix::io::Errno::NOENT => Ok(true),
        Err(source) => Err(KeeperError::Io {
            operation: "inspect sidecar base",
            path: directory.join(base),
            source: io::Error::from(source),
        }),
    }
}

/// Whether an unreferenced segment has served its full grace period.
///
/// Reader-safety rule: a reader can hold a MANIFEST that references `name`
/// only if it read that MANIFEST before the slot pair stopped referencing the
/// segment. Unlinking the file no earlier than one grace period after that
/// transition therefore leaves every such reader the whole grace period to
/// finish opening its segments, and a segment a reader has already mapped
/// survives its unlink. Which durable witness dates the transition depends on
/// whether any generation may already have been superseded:
///
/// * Past the supersession threshold the retiring publication's receipt is
///   the witness, and the only admissible one: a missing, unreadable, or
///   foreign-bound receipt proves nothing, and proving nothing fails closed
///   (inferring the transition from the segment's own mtime is exactly the
///   fail-open this gate exists to remove). The receipt is made durable
///   *before* the slot renames that complete the retirement, so it strictly
///   precedes the last instant a reader could have obtained a referencing
///   MANIFEST through either slot. Grace runs from the receipt alone. A later
///   publication never references a retired segment, so it must not postpone
///   that segment's reclamation: measuring from the latest publication
///   instead would mean an embedder that publishes more often than the grace
///   period never reclaims anything (cass#453).
/// * Below the threshold nothing was ever retired, so no receipt can exist and
///   the manifest-level floor -- the latest durable slot mutation -- dates the
///   transition conservatively.
///
/// In both cases the file's own mtime must also predate the grace window: a
/// segment rewritten after its witness is not the segment the witness dated.
#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn segment_grace_served(
    reachability: &SegmentReachability,
    name: &OsStr,
    stat: &rustix::fs::Stat,
    receipts: &BTreeMap<u64, SegmentRetirementReceipt>,
    now: SystemTime,
    grace_period: Duration,
) -> bool {
    if !stat_old_enough(stat, now, grace_period) {
        return false;
    }
    if !reachability.retirement_provenance_required {
        return reachability
            .unreachable_since
            .is_none_or(|unreachable_since| {
                now.duration_since(unreachable_since)
                    .is_ok_and(|age| age >= grace_period)
            });
    }
    let Some(segment_id) = name.to_str().and_then(parse_segment_name) else {
        return false;
    };
    receipts.get(&segment_id).is_some_and(|receipt| {
        receipt.retired_at().is_some_and(|retired_at| {
            now.duration_since(retired_at)
                .is_ok_and(|age| age >= grace_period)
        })
    })
}

/// Read one candidate receipt through the sweep's own directory handle.
///
/// A receipt of any other length, an unreadable image, or a failed seal is
/// reported as absent so the segment stays uncollectable. Only a vanished
/// entry and a genuine I/O fault are distinguished: the latter aborts the
/// sweep before anything is unlinked.
#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn read_retirement_receipt_at(
    directory_file: &File,
    name: &OsStr,
    path: &Path,
    stat: &rustix::fs::Stat,
) -> Result<Option<SegmentRetirementReceipt>, KeeperError> {
    use rustix::fs::{Mode, OFlags, openat};

    if usize::try_from(stat.st_size).ok() != Some(RETIREMENT_RECEIPT_BYTES) {
        return Ok(None);
    }
    let opened = match openat(
        directory_file,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    ) {
        Ok(opened) => opened,
        Err(source) if source == rustix::io::Errno::NOENT => return Ok(None),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "open retirement receipt",
                path: path.to_path_buf(),
                source: io::Error::from(source),
            });
        }
    };
    let mut file = File::from(opened);
    let mut bytes = [0_u8; RETIREMENT_RECEIPT_BYTES];
    match file.read_exact(&mut bytes) {
        Ok(()) => {}
        Err(source) if source.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "read retirement receipt",
                path: path.to_path_buf(),
                source,
            });
        }
    }
    Ok(SegmentRetirementReceipt::decode(&bytes))
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn stat_old_enough(stat: &rustix::fs::Stat, now: SystemTime, grace_period: Duration) -> bool {
    stat_modified_time(stat).is_some_and(|modified| {
        now.duration_since(modified)
            .is_ok_and(|age| age >= grace_period)
    })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn stat_modified_time(stat: &rustix::fs::Stat) -> Option<SystemTime> {
    unix_system_time(stat.st_mtime, u32::try_from(stat.st_mtime_nsec).ok()?)
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn stat_change_time(stat: &rustix::fs::Stat) -> Option<SystemTime> {
    unix_system_time(stat.st_ctime, u32::try_from(stat.st_ctime_nsec).ok()?)
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn manifest_publish_time(manifest: &Manifest) -> Option<SystemTime> {
    (manifest.last_publish_unix_s > 0)
        .then_some(manifest.last_publish_unix_s)
        .and_then(|seconds| unix_system_time(seconds, 0))
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn unix_system_time(seconds: i64, nanoseconds: u32) -> Option<SystemTime> {
    if nanoseconds >= 1_000_000_000 {
        return None;
    }
    if seconds >= 0 {
        return SystemTime::UNIX_EPOCH
            .checked_add(Duration::new(u64::try_from(seconds).ok()?, nanoseconds));
    }
    let magnitude = seconds.unsigned_abs();
    let before_epoch = if nanoseconds == 0 {
        Duration::from_secs(magnitude)
    } else {
        Duration::new(magnitude.checked_sub(1)?, 1_000_000_000 - nanoseconds)
    };
    SystemTime::UNIX_EPOCH.checked_sub(before_epoch)
}

#[cfg(unix)]
pub(in crate::keeper) fn open_gc_directory(directory: &Path) -> Result<File, KeeperError> {
    use rustix::fs::{Mode, OFlags, open};

    open(
        directory,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map(File::from)
    .map_err(|source| KeeperError::Io {
        operation: "open no-follow garbage-collection directory",
        path: directory.to_path_buf(),
        source: io::Error::from(source),
    })
}

#[cfg(unix)]
pub(in crate::keeper) fn ensure_gc_directory_identity(
    directory: &Path,
    opened: &File,
) -> Result<(), KeeperError> {
    use std::os::unix::fs::MetadataExt;

    let path_metadata = std::fs::metadata(directory).map_err(|source| KeeperError::Io {
        operation: "verify garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })?;
    let opened_metadata = opened.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect opened garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })?;
    if path_metadata.dev() != opened_metadata.dev()
        || path_metadata.ino() != opened_metadata.ino()
        || !opened_metadata.is_dir()
    {
        return Err(KeeperError::GarbageDirectoryChanged {
            directory: directory.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sync_gc_directory(opened: &File, directory: &Path) -> Result<(), KeeperError> {
    opened.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
pub(in crate::keeper) fn safe_direct_child(
    directory: &Path,
    relative: &Path,
) -> Result<PathBuf, KeeperError> {
    let mut components = relative.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(KeeperError::UnsafeGarbagePath {
            path: relative.to_path_buf(),
        });
    }
    Ok(directory.join(relative))
}

pub(in crate::keeper) fn canonical_segment_name(segment_id: u64) -> String {
    format!("seg-{segment_id:016x}.fslx")
}

/// Four-byte first-unreachable receipt magic, mirroring the MANIFEST style.
const RETIREMENT_RECEIPT_MAGIC: [u8; 4] = *b"QRTR";
/// Current durable first-unreachable receipt version.
const RETIREMENT_RECEIPT_VERSION: u8 = 1;
/// Exact byte length of a v1 receipt, including its trailing xxh3 seal.
pub(in crate::keeper) const RETIREMENT_RECEIPT_BYTES: usize = 40;

/// Canonical name of one segment's first-unreachable receipt.
pub(in crate::keeper) fn retirement_receipt_name(segment_id: u64) -> String {
    format!("{}.retired", canonical_segment_name(segment_id))
}

/// Which component durably witnessed a segment's first unreachability.
///
/// Both kinds carry the same authority -- an instant, after which the grace
/// period runs -- and differ only in who observed it. The distinction is kept
/// on disk because the two are produced by different code paths under
/// different preconditions, and an operator reading a directory by hand must
/// be able to tell a retirement from a crash-orphan sighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    windows,
    allow(
        dead_code,
        reason = "descriptor-relative Windows garbage collection remains explicitly unsupported"
    )
)]
pub(in crate::keeper) enum UnreachabilityWitness {
    /// A publication rotated the segment out of the durable slot pair.
    RetiredByPublication,
    /// An admitted writer with no staged work of its own observed a segment
    /// that no slot references and no receipt covers.
    FirstSightedByWriter,
}

#[cfg_attr(
    windows,
    allow(
        dead_code,
        reason = "descriptor-relative Windows garbage collection remains explicitly unsupported"
    )
)]
impl UnreachabilityWitness {
    const RETIRED_BY_PUBLICATION: u8 = 1;
    const FIRST_SIGHTED_BY_WRITER: u8 = 2;

    pub(in crate::keeper) const fn encode(self) -> u8 {
        match self {
            Self::RetiredByPublication => Self::RETIRED_BY_PUBLICATION,
            Self::FirstSightedByWriter => Self::FIRST_SIGHTED_BY_WRITER,
        }
    }

    pub(in crate::keeper) const fn decode(byte: u8) -> Option<Self> {
        match byte {
            Self::RETIRED_BY_PUBLICATION => Some(Self::RetiredByPublication),
            Self::FIRST_SIGHTED_BY_WRITER => Some(Self::FirstSightedByWriter),
            _ => None,
        }
    }
}

/// Durable witness that one segment is no longer reachable through any
/// MANIFEST slot.
///
/// Garbage collection must not infer a retired segment's unreachability age
/// from the segment's own mtime: a segment sealed long before it was retired
/// would then be deletable the instant it fell out of the slot pair, inside
/// any reader's drain interval. The publisher that performs the retirement is
/// the only component that observes that transition, so it stamps this
/// receipt and makes it durable *before* the slot renames make the transition
/// itself durable. A receipt may therefore exist for a still-reachable
/// segment (an aborted publish), but a retired segment can never lack one.
///
/// A segment that no publication ever referenced has no retirement to
/// witness, so an admitted writer stamps the same receipt with
/// [`UnreachabilityWitness::FirstSightedByWriter`] when it first observes the
/// orphan (see [`witness_orphaned_segments_at`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::keeper) struct SegmentRetirementReceipt {
    /// Segment this receipt is bound to; a mismatch against the file name
    /// makes the receipt inert, so one cannot be transplanted onto another.
    pub(in crate::keeper) segment_id: u64,
    /// Wall clock at which the transition was durably witnessed.
    pub(in crate::keeper) retired_unix_s: i64,
    /// Generation current when the witness was taken. For a retirement this
    /// is the publication that performed it.
    pub(in crate::keeper) retired_generation: u64,
    /// Which component observed the transition.
    pub(in crate::keeper) witness: UnreachabilityWitness,
}

#[cfg_attr(
    windows,
    allow(
        dead_code,
        reason = "descriptor-relative Windows garbage collection remains explicitly unsupported"
    )
)]
impl SegmentRetirementReceipt {
    pub(in crate::keeper) fn encode(&self) -> [u8; RETIREMENT_RECEIPT_BYTES] {
        let mut bytes = [0_u8; RETIREMENT_RECEIPT_BYTES];
        bytes[0..4].copy_from_slice(&RETIREMENT_RECEIPT_MAGIC);
        bytes[4] = RETIREMENT_RECEIPT_VERSION;
        bytes[5] = self.witness.encode();
        bytes[8..16].copy_from_slice(&self.segment_id.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.retired_unix_s.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.retired_generation.to_le_bytes());
        let seal = xxhash_rust::xxh3::xxh3_64(&bytes[0..32]);
        bytes[32..40].copy_from_slice(&seal.to_le_bytes());
        bytes
    }

    /// Decode one receipt image, rejecting every malformed or unsealed input.
    ///
    /// Returns `None` rather than an error: an unreadable receipt proves
    /// nothing, and proving nothing must leave the segment uncollectable
    /// instead of failing the whole sweep.
    pub(in crate::keeper) fn decode(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; RETIREMENT_RECEIPT_BYTES] = bytes.try_into().ok()?;
        if bytes[0..4] != RETIREMENT_RECEIPT_MAGIC[..]
            || bytes[4] != RETIREMENT_RECEIPT_VERSION
            || bytes[6..8] != [0_u8; 2][..]
        {
            return None;
        }
        let witness = UnreachabilityWitness::decode(bytes[5])?;
        let seal = u64::from_le_bytes(bytes[32..40].try_into().ok()?);
        if seal != xxhash_rust::xxh3::xxh3_64(&bytes[0..32]) {
            return None;
        }
        Some(Self {
            segment_id: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
            retired_unix_s: i64::from_le_bytes(bytes[16..24].try_into().ok()?),
            retired_generation: u64::from_le_bytes(bytes[24..32].try_into().ok()?),
            witness,
        })
    }

    /// The witnessed retirement instant, when it is representable here.
    #[cfg(unix)]
    pub(in crate::keeper) fn retired_at(&self) -> Option<SystemTime> {
        unix_system_time(self.retired_unix_s, 0)
    }
}

/// Stamp first-unreachable receipts for every segment this publication
/// retires, and make them durable before the caller performs the slot renames.
///
/// Each receipt is written to a `.tmp-` sibling, fsynced, atomically renamed
/// into place, and sealed with one directory fsync. Rewriting an existing
/// receipt is deliberate: an earlier aborted publication left a witness for a
/// transition that did not happen, and the surviving reader drain must be
/// measured from the retirement that does.
pub(in crate::keeper) fn write_segment_retirement_receipts(
    directory: &Path,
    retired: &BTreeSet<u64>,
    generation: u64,
) -> Result<(), KeeperError> {
    if retired.is_empty() {
        return Ok(());
    }
    let retired_unix_s = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .and_then(|elapsed| i64::try_from(elapsed.as_secs()).ok())
        .ok_or_else(|| KeeperError::InvalidTransition {
            detail: "current wall clock is outside the retirement receipt range".to_owned(),
        })?;
    for segment_id in retired {
        write_unreachability_receipt(
            directory,
            SegmentRetirementReceipt {
                segment_id: *segment_id,
                retired_unix_s,
                retired_generation: generation,
                witness: UnreachabilityWitness::RetiredByPublication,
            },
        )?;
    }
    #[cfg(not(windows))]
    sync_directory(directory)?;
    Ok(())
}

/// Install one receipt atomically, leaving the caller to fsync the directory.
///
/// An existing receipt is replaced rather than updated in place, so a crash
/// can only leave the old image or the new one, never a torn one.
pub(in crate::keeper) fn write_unreachability_receipt(
    directory: &Path,
    receipt: SegmentRetirementReceipt,
) -> Result<(), KeeperError> {
    let name = retirement_receipt_name(receipt.segment_id);
    let temp_path = directory.join(format!(".tmp-{name}"));
    let receipt_path = directory.join(&name);
    let mut temp = File::create(&temp_path).map_err(|source| KeeperError::Io {
        operation: "create unreachability receipt temp",
        path: temp_path.clone(),
        source,
    })?;
    temp.write_all(&receipt.encode())
        .and_then(|()| temp.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "persist unreachability receipt temp",
            path: temp_path.clone(),
            source,
        })?;
    drop(temp);
    replace_file_atomically(&temp_path, &receipt_path).map_err(|source| KeeperError::Io {
        operation: "rename unreachability receipt into place",
        path: receipt_path,
        source,
    })
}

/// Segments the pending publication removes from the durable slot pair.
///
/// After the renames the reachable set is the proposed manifest unioned with
/// the outgoing current slot. Anything the outgoing *previous* slot referenced
/// and neither of those retains has just become unreachable, and this is the
/// only moment any component observes that transition.
pub(in crate::keeper) fn retired_segment_ids(
    directory: &Path,
    outgoing_current: Option<&Manifest>,
    proposed: &Manifest,
) -> Result<BTreeSet<u64>, KeeperError> {
    let previous_path = directory.join("MANIFEST.prev");
    let ManifestSlot::Valid(outgoing_previous) = read_manifest_slot(&previous_path)? else {
        return Ok(BTreeSet::new());
    };
    let mut retained = BTreeSet::new();
    retained.extend(proposed.segments.iter().map(|segment| segment.segment_id));
    if let Some(current) = outgoing_current {
        retained.extend(current.segments.iter().map(|segment| segment.segment_id));
    }
    Ok(outgoing_previous
        .segments
        .iter()
        .map(|segment| segment.segment_id)
        .filter(|segment_id| !retained.contains(segment_id))
        .collect())
}
