use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use frankensearch_index::mapped_file::ReadOnlyMappedFile;

use crate::schema::SchemaDescriptor;
use crate::segment::SegmentReader;

use super::errors::{KeeperError, LiveDocumentFloor};
use super::gc::canonical_segment_name;
use super::manifest::{LoadedManifest, Manifest, ManifestSegment, ManifestSource};
use super::manifest_validation::validate_segment_transitions;
use super::publish::parse_claim_name;
use super::recovered_segment::{AuthenticatedFileWitness, RecoveredSegment};
use super::slot_io::{ManifestSlot, read_manifest_slot};
use super::writer::WriterProtection;

/// Load the current MANIFEST, falling back to `MANIFEST.prev` without writing.
///
/// A valid current slot wins. When both slots are valid, the previous
/// generation must equal the current generation (idempotent republish) or be
/// exactly one less. Missing current plus valid previous is the expected
/// between-renames crash window and is not `IndexNotFound`.
///
/// # Errors
///
/// Returns a typed not-found, corruption, invalid-pair, or I/O error.
pub fn load_manifest_pair(directory: impl AsRef<Path>) -> Result<LoadedManifest, KeeperError> {
    let directory = directory.as_ref();
    let current_path = directory.join("MANIFEST");
    let previous_path = directory.join("MANIFEST.prev");
    let current = match read_manifest_slot(&current_path)? {
        ManifestSlot::Valid(current) => {
            return load_valid_current(directory, &current_path, &previous_path, current);
        }
        current => current,
    };

    let previous = read_manifest_slot(&previous_path)?;
    match (current, previous) {
        (ManifestSlot::Missing, ManifestSlot::Valid(previous)) => Ok(LoadedManifest {
            manifest: previous,
            source: ManifestSource::PreviousAfterMissingCurrent,
        }),
        (ManifestSlot::Invalid(_), ManifestSlot::Valid(previous)) => Ok(LoadedManifest {
            manifest: previous,
            source: ManifestSource::PreviousAfterCorruptCurrent,
        }),
        (ManifestSlot::Missing, ManifestSlot::Missing) => Err(KeeperError::IndexNotFound {
            directory: directory.to_path_buf(),
        }),
        (ManifestSlot::Invalid(source), ManifestSlot::Missing) => {
            Err(KeeperError::ManifestCorrupted {
                path: current_path,
                source,
            })
        }
        (ManifestSlot::Missing, ManifestSlot::Invalid(source)) => {
            Err(KeeperError::ManifestCorrupted {
                path: previous_path,
                source,
            })
        }
        (ManifestSlot::Invalid(current), ManifestSlot::Invalid(previous)) => {
            Err(KeeperError::NoValidManifest {
                directory: directory.to_path_buf(),
                current: current.to_string(),
                previous: previous.to_string(),
            })
        }
        (ManifestSlot::Valid(_), _) => unreachable!("valid current returned above"),
    }
}

fn load_valid_current(
    directory: &Path,
    current_path: &Path,
    previous_path: &Path,
    current: Manifest,
) -> Result<LoadedManifest, KeeperError> {
    let previous = match read_manifest_slot(previous_path) {
        Ok(ManifestSlot::Valid(previous)) => previous,
        Ok(ManifestSlot::Missing | ManifestSlot::Invalid(_)) | Err(_) => {
            return Ok(LoadedManifest {
                manifest: current,
                source: ManifestSource::Current,
            });
        }
    };

    // Readers intentionally take no writer lock. Re-read the primary after the
    // secondary so a commit that advanced both slots cannot manufacture a
    // false pair error from two different publication instants.
    let stable_current = match read_manifest_slot(current_path) {
        Ok(ManifestSlot::Valid(stable_current)) if stable_current == current => stable_current,
        Ok(ManifestSlot::Valid(newer_current)) => {
            return Ok(LoadedManifest {
                manifest: newer_current,
                source: ManifestSource::Current,
            });
        }
        Ok(ManifestSlot::Missing | ManifestSlot::Invalid(_)) | Err(_) => {
            return Ok(LoadedManifest {
                manifest: current,
                source: ManifestSource::Current,
            });
        }
    };

    validate_manifest_pair(directory, stable_current, &previous)
}

pub(in crate::keeper) fn validate_manifest_pair(
    directory: &Path,
    current: Manifest,
    previous: &Manifest,
) -> Result<LoadedManifest, KeeperError> {
    let adjacent = previous.generation == current.generation
        || previous
            .generation
            .checked_add(1)
            .is_some_and(|generation| generation == current.generation);
    if !adjacent {
        return Err(KeeperError::InvalidGenerationPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
        });
    }
    if previous.generation == current.generation && previous != &current {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: "equal generations have different canonical contents".to_owned(),
        });
    }
    if previous.schema_id != current.schema_id {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: format!(
                "schema_id changed from {:#018x} to {:#018x}",
                previous.schema_id, current.schema_id
            ),
        });
    }
    if current.docid_high_watermark < previous.docid_high_watermark {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: format!(
                "docid_high_watermark rolled back from {} to {}",
                previous.docid_high_watermark, current.docid_high_watermark
            ),
        });
    }
    if previous.generation != current.generation {
        // Both generations are already published; this is a shape check on
        // the pair, not an admission decision, so the floor is waived — a
        // legitimately published `delete_all` must reopen.
        validate_segment_transitions(previous, &current, LiveDocumentFloor::Waived).map_err(
            |error| KeeperError::InvalidManifestPair {
                directory: directory.to_path_buf(),
                current: current.generation,
                previous: previous.generation,
                detail: error.to_string(),
            },
        )?;
    }
    Ok(LoadedManifest {
        manifest: current,
        source: ManifestSource::Current,
    })
}

pub(in crate::keeper) fn validate_loaded_schema(
    directory: &Path,
    expected: u64,
    loaded: &LoadedManifest,
) -> Result<(), KeeperError> {
    if loaded.manifest.schema_id == expected {
        return Ok(());
    }
    let slot = match loaded.source {
        ManifestSource::Current => "MANIFEST",
        ManifestSource::PreviousAfterMissingCurrent
        | ManifestSource::PreviousAfterCorruptCurrent => "MANIFEST.prev",
        ManifestSource::InMemory => "<in-memory>",
    };
    Err(KeeperError::SchemaMismatch {
        path: directory.join(slot),
        expected,
        found: loaded.manifest.schema_id,
    })
}

pub(in crate::keeper) fn validate_recovery_claims(
    directory: &Path,
    loaded: &LoadedManifest,
) -> Result<(), KeeperError> {
    if loaded.source == ManifestSource::Current || loaded.source == ManifestSource::InMemory {
        return Ok(());
    }
    let admitted_claim = loaded.manifest.generation.checked_add(1);
    for (path, claimed) in scan_generation_claims(directory)? {
        if claimed <= loaded.manifest.generation || admitted_claim == Some(claimed) {
            continue;
        }
        return Err(KeeperError::InvalidRecoveryClaim {
            path,
            recovered: loaded.manifest.generation,
            claimed,
        });
    }
    Ok(())
}

pub(in crate::keeper) fn scan_generation_claims(
    directory: &Path,
) -> Result<Vec<(PathBuf, u64)>, KeeperError> {
    let mut claims = Vec::new();
    let entries = std::fs::read_dir(directory).map_err(|source| KeeperError::Io {
        operation: "scan recovery claims",
        path: directory.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| KeeperError::Io {
            operation: "read recovery claim entry",
            path: directory.to_path_buf(),
            source,
        })?;
        let Some(claimed) = parse_claim_name(&entry.file_name()) else {
            continue;
        };
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(&path).map_err(|source| KeeperError::Io {
            operation: "inspect generation claim",
            path: path.clone(),
            source,
        })?;
        if !metadata.file_type().is_file() || metadata.len() != 0 {
            return Err(KeeperError::InvalidClaimArtifact {
                path,
                detail: format!(
                    "claim must be a zero-length regular file, found type {:?} and length {}",
                    metadata.file_type(),
                    metadata.len()
                ),
            });
        }
        claims.try_reserve(1).map_err(|error| KeeperError::Io {
            operation: "allocate recovery claim inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
        claims.push((path, claimed));
    }
    claims.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    Ok(claims)
}

/// A proposal segment the publication preflight verified and bound, with the
/// file it was read through, so that the writer's snapshot after a successful
/// publication need not verify the same bytes again.
pub(in crate::keeper) struct VerifiedProposalSegment {
    pub(in crate::keeper) segment: RecoveredSegment,
    pub(in crate::keeper) identity: Option<FileIdentity>,
}

/// Device and inode of a file; `None` where the platform reports neither, so
/// a published snapshot reopens every segment there.
pub(in crate::keeper) type FileIdentity = (u64, u64);

#[cfg(unix)]
#[allow(
    clippy::unnecessary_wraps,
    reason = "platforms without inode numbers report None"
)]
fn file_identity(metadata: &std::fs::Metadata) -> Option<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    Some((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn file_identity(_: &std::fs::Metadata) -> Option<FileIdentity> {
    None
}

pub(in crate::keeper) fn validate_proposed_manifest_segments(
    directory: &Path,
    manifest: &Manifest,
    schema: SchemaDescriptor,
    protection: &WriterProtection,
) -> Result<Vec<VerifiedProposalSegment>, KeeperError> {
    let mut verified = Vec::new();
    verified
        .try_reserve_exact(manifest.segments.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate verified proposal segments",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for manifest_segment in &manifest.segments {
        let path = directory.join(canonical_segment_name(manifest_segment.segment_id));
        let (reader, identity) = SegmentReader::open_published_checked(
            &path,
            schema,
            crate::segment::SegmentLimits::default(),
            |_, file| Ok(file_identity(&file.metadata()?)),
        )
        .map_err(|source| KeeperError::SegmentOpen {
            path: path.clone(),
            source,
        })?;
        let file_xxh3 = fully_verified_file_witness(&path, &reader)?;
        validate_segment_witnesses(&path, manifest_segment, &reader, || Ok(file_xxh3))?;
        #[cfg(test)]
        let full_prefix_hash_count = Arc::new(AtomicU64::new(0));
        #[cfg(test)]
        full_prefix_hash_count.fetch_add(1, AtomicOrdering::Relaxed);
        #[cfg(test)]
        let authenticated_file_witness =
            AuthenticatedFileWitness::mint_after_full_prefix_validation(
                file_xxh3,
                full_prefix_hash_count,
            );
        #[cfg(not(test))]
        let authenticated_file_witness =
            AuthenticatedFileWitness::mint_after_full_prefix_validation(file_xxh3);
        // A segment too large for one RaptorQ source block has no sidecar by
        // design; its bytes were just verified against the MANIFEST witness.
        let _ = protection;
        let segment = RecoveredSegment::bind(
            path,
            manifest_segment.clone(),
            reader,
            schema,
            authenticated_file_witness,
        )?;
        verified.push(VerifiedProposalSegment { segment, identity });
    }
    Ok(verified)
}

/// The MANIFEST segments of a snapshot, reusing a segment the publication
/// preflight verified when the MANIFEST binds it unchanged and its path still
/// names the file the preflight read; any other segment is opened and
/// verified as [`open_manifest_segments`] does.
pub(in crate::keeper) fn reuse_verified_segments(
    directory: &Path,
    manifest_segments: &[ManifestSegment],
    schema: SchemaDescriptor,
    mut verified: Vec<VerifiedProposalSegment>,
) -> Result<Vec<RecoveredSegment>, KeeperError> {
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(manifest_segments.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate recovered segment table",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for manifest_segment in manifest_segments {
        let path = directory.join(canonical_segment_name(manifest_segment.segment_id));
        let current = match std::fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() => file_identity(&metadata),
            _ => None,
        };
        let reusable = verified.iter().position(|candidate| {
            candidate.segment.manifest == *manifest_segment
                && candidate.identity.is_some()
                && candidate.identity == current
        });
        segments.push(match reusable {
            Some(index) => verified.swap_remove(index).segment,
            None => open_manifest_segment(directory, manifest_segment, schema)?,
        });
    }
    Ok(segments)
}

/// Upper bound on threads for one snapshot open, so a single open over a
/// many-segment index does not claim every core of a large host.
const MAX_SEGMENT_OPEN_THREADS: usize = 8;

/// Open, authenticate, and bind one MANIFEST segment for a read-only snapshot.
fn open_manifest_segment(
    directory: &Path,
    manifest_segment: &ManifestSegment,
    schema: SchemaDescriptor,
) -> Result<RecoveredSegment, KeeperError> {
    let path = directory.join(canonical_segment_name(manifest_segment.segment_id));
    let (reader, authenticated_file_witness) = SegmentReader::open_published_checked(
        &path,
        schema,
        crate::segment::SegmentLimits::default(),
        |reader, file| {
            Ok(authenticate_segment_witness(
                &path,
                manifest_segment,
                reader,
                file,
            ))
        },
    )
    .map_err(|source| KeeperError::SegmentOpen {
        path: path.clone(),
        source,
    })?;
    let authenticated_file_witness = authenticated_file_witness?;
    RecoveredSegment::bind(
        path,
        manifest_segment.clone(),
        reader,
        schema,
        authenticated_file_witness,
    )
}

/// Open every MANIFEST segment, verifying independent segments concurrently.
///
/// Every segment still gets the full same-descriptor prefix hash and witness
/// checks; only the wall-clock cost changes from the sum over segments to
/// roughly the largest one. Scoped OS threads are used rather than the rayon
/// pool so a caller holding a lock (writer open, recovery) never waits on, or
/// steals, unrelated pool work. Results are consumed in MANIFEST order, so the
/// reported error is the first failing segment in that order, exactly as the
/// sequential open reported it. If a helper thread cannot be spawned, the
/// calling thread does the remaining work itself.
pub(in crate::keeper) fn open_manifest_segments(
    directory: &Path,
    manifest_segments: &[ManifestSegment],
    schema: SchemaDescriptor,
) -> Result<Vec<RecoveredSegment>, KeeperError> {
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(manifest_segments.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate recovered segment table",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(MAX_SEGMENT_OPEN_THREADS)
        .min(manifest_segments.len());
    if workers <= 1 {
        for manifest_segment in manifest_segments {
            segments.push(open_manifest_segment(directory, manifest_segment, schema)?);
        }
        return Ok(segments);
    }

    let slots: Vec<OnceLock<Result<RecoveredSegment, KeeperError>>> =
        manifest_segments.iter().map(|_| OnceLock::new()).collect();
    let next = AtomicUsize::new(0);
    // Lowest MANIFEST index that failed so far; later segments are skipped
    // because their result can no longer be reported.
    let first_failure = AtomicUsize::new(usize::MAX);
    let work = || {
        loop {
            let index = next.fetch_add(1, AtomicOrdering::Relaxed);
            let Some(manifest_segment) = manifest_segments.get(index) else {
                break;
            };
            if index > first_failure.load(AtomicOrdering::Relaxed) {
                continue;
            }
            let result = open_manifest_segment(directory, manifest_segment, schema);
            if result.is_err() {
                first_failure.fetch_min(index, AtomicOrdering::Relaxed);
            }
            let _ = slots[index].set(result);
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..workers {
            if std::thread::Builder::new()
                .name("quill-segment-open".to_owned())
                .spawn_scoped(scope, work)
                .is_err()
            {
                break;
            }
        }
        work();
    });

    for (index, slot) in slots.into_iter().enumerate() {
        match slot.into_inner() {
            Some(Ok(segment)) => segments.push(segment),
            Some(Err(error)) => return Err(error),
            None => {
                return Err(KeeperError::InvalidTransition {
                    detail: format!(
                        "segment open skipped MANIFEST entry {index} without an earlier failure"
                    ),
                });
            }
        }
    }
    Ok(segments)
}

/// Authenticate a MANIFEST segment binding against the backing FSLX bytes.
///
/// `validate_segment_witnesses` retains the cheap header and length checks
/// before invoking the same-descriptor streamed hash closure. A successful
/// return mints an [`AuthenticatedFileWitness`] for a recovered segment.
fn authenticate_segment_witness(
    path: &Path,
    manifest: &ManifestSegment,
    reader: &SegmentReader<ReadOnlyMappedFile>,
    file: &mut File,
) -> Result<AuthenticatedFileWitness, KeeperError> {
    let mut verified_file_xxh3 = None;
    #[cfg(test)]
    let full_prefix_hash_count = Arc::new(AtomicU64::new(0));
    validate_segment_witnesses(path, manifest, reader, || {
        #[cfg(test)]
        full_prefix_hash_count.fetch_add(1, AtomicOrdering::Relaxed);
        let file_xxh3 = reader
            .verify_streamed_file_witness(file)
            .map_err(|source| KeeperError::SegmentOpen {
                path: path.to_path_buf(),
                source,
            })?;
        verified_file_xxh3 = Some(file_xxh3);
        Ok(file_xxh3)
    })?;
    let file_xxh3 = verified_file_xxh3.ok_or_else(|| KeeperError::InvalidTransition {
        detail: format!(
            "segment {} completed witness validation without a full-prefix receipt",
            path.display()
        ),
    })?;
    #[cfg(test)]
    let authenticated_file_witness = AuthenticatedFileWitness::mint_after_full_prefix_validation(
        file_xxh3,
        full_prefix_hash_count,
    );
    #[cfg(not(test))]
    let authenticated_file_witness =
        AuthenticatedFileWitness::mint_after_full_prefix_validation(file_xxh3);
    Ok(authenticated_file_witness)
}

/// Fully validate a segment and retain the one freshly recomputed prefix
/// witness for manifest binding. Eager recovery, stats, and proposed-manifest
/// preflight use this instead of rehashing the same full prefix afterward.
pub(in crate::keeper) fn fully_verified_file_witness(
    path: &Path,
    reader: &SegmentReader<impl AsRef<[u8]>>,
) -> Result<u64, KeeperError> {
    reader
        .verify_with_file_witness()
        .map_err(|source| KeeperError::SegmentOpen {
            path: path.to_path_buf(),
            source,
        })
}

pub(in crate::keeper) fn validate_segment_witnesses<F>(
    path: &Path,
    manifest: &ManifestSegment,
    reader: &SegmentReader<impl AsRef<[u8]>>,
    retained_file_witness: F,
) -> Result<(), KeeperError>
where
    F: FnOnce() -> Result<u64, KeeperError>,
{
    let header = reader.header();
    if header.segment_id != manifest.segment_id {
        return Err(KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: format!(
                "header segment_id {:#018x} != manifest {:#018x}",
                header.segment_id, manifest.segment_id
            ),
        });
    }
    if reader.file_len() != manifest.file_len {
        return Err(KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail: format!(
                "file length {} != manifest {}",
                reader.file_len(),
                manifest.file_len
            ),
        });
    }
    let actual_file_xxh3 = retained_file_witness()?;
    let mismatch = if actual_file_xxh3 != manifest.file_xxh3 {
        Some(format!(
            "actual file_xxh3 {actual_file_xxh3:#018x} != manifest {:#018x}",
            manifest.file_xxh3
        ))
    } else if header.docid_lo != manifest.docid_lo || header.docid_hi != manifest.docid_hi {
        Some(format!(
            "header range [{}, {}) != manifest [{}, {})",
            header.docid_lo, header.docid_hi, manifest.docid_lo, manifest.docid_hi
        ))
    } else if header.doc_count != manifest.doc_count {
        Some(format!(
            "header doc_count {} != manifest {}",
            header.doc_count, manifest.doc_count
        ))
    } else {
        None
    };
    if let Some(detail) = mismatch {
        return Err(KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail,
        });
    }
    Ok(())
}
