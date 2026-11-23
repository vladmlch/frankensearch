#[cfg(test)]
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;

#[cfg(windows)]
use super::admission::open_windows_regular_file_no_reparse;
use super::errors::{KeeperError, LiveDocumentFloor};
use super::gc::{retired_segment_ids, write_segment_retirement_receipts};
use super::manifest::{LoadedManifest, Manifest, ManifestSource};
use super::manifest_load::load_manifest_pair;
use super::manifest_validation::{usize_to_u64, validate_segment_transitions};
use super::publish::PublishCheckpoint;
#[cfg(test)]
use super::publish::{
    MANIFEST_PUBLISH_FAILURES, MANIFEST_PUBLISH_PAUSE_TEST_SERIAL, ManifestPublishCheckpointEvent,
    ManifestPublishFailure, ManifestPublishFailureControl, ManifestPublishPause,
    ManifestPublishPauseControl, manifest_publish_pauses, normalize_publish_directory,
};
#[cfg(not(any(unix, windows)))]
use super::slot_io::ensure_atomic_publish_supported;
use super::slot_io::replace_file_atomically;
#[cfg(not(windows))]
use super::slot_io::sync_directory;

#[cfg(test)]
fn manifest_publish_failures()
-> &'static std::sync::Mutex<BTreeMap<PathBuf, ManifestPublishFailure>> {
    MANIFEST_PUBLISH_FAILURES.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()))
}

/// Hold the shared test-only MANIFEST pause serialization guard.
///
/// Callers must acquire this before arming a pause and keep it until after the
/// publisher is released, so every paused public-operation test shares one
/// deterministic rendezvous lane.
#[cfg(test)]
pub fn manifest_publish_pause_test_serial_guard() -> std::sync::MutexGuard<'static, ()> {
    MANIFEST_PUBLISH_PAUSE_TEST_SERIAL
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Arm one real MANIFEST publisher checkpoint for a single directory.
#[cfg(test)]
pub fn pause_manifest_publish_at_checkpoint_for_test(
    directory: &Path,
    checkpoint: PublishCheckpoint,
) -> ManifestPublishPauseControl {
    let directory = normalize_publish_directory(directory.to_path_buf());
    let event = Arc::new(std::sync::Mutex::new(
        ManifestPublishCheckpointEvent::default(),
    ));
    let released = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let pause = ManifestPublishPause {
        checkpoint,
        event: Arc::clone(&event),
        released: Arc::clone(&released),
    };
    let mut pauses = manifest_publish_pauses()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match pauses.entry(directory.clone()) {
        std::collections::btree_map::Entry::Vacant(slot) => {
            slot.insert(pause);
        }
        std::collections::btree_map::Entry::Occupied(_) => {
            panic!("only one MANIFEST checkpoint pause may be armed per directory");
        }
    }
    drop(pauses);
    ManifestPublishPauseControl {
        directory,
        checkpoint,
        event,
        released,
    }
}

/// Inject one typed publisher failure after the named real choreography
/// checkpoint. Tests use `TempWritten` to prove the known-pre-rename path
/// restores `Stable` without manufacturing a filesystem error.
#[cfg(test)]
pub fn fail_manifest_publish_at_checkpoint_for_test(
    directory: &Path,
    checkpoint: PublishCheckpoint,
) -> ManifestPublishFailureControl {
    let directory = normalize_publish_directory(directory.to_path_buf());
    let mut failures = manifest_publish_failures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match failures.entry(directory.clone()) {
        std::collections::btree_map::Entry::Vacant(slot) => {
            slot.insert(ManifestPublishFailure { checkpoint });
        }
        std::collections::btree_map::Entry::Occupied(_) => {
            panic!("only one MANIFEST checkpoint failure may be armed per directory");
        }
    }
    drop(failures);
    ManifestPublishFailureControl { directory }
}

#[cfg(test)]
impl ManifestPublishPauseControl {
    /// Poll the checkpoint event, retaining this task's waker until the real
    /// blocking choreography completes the checkpoint.
    fn poll_reached(&self, task_cx: &std::task::Context<'_>) -> std::task::Poll<()> {
        let mut event = self
            .event
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if event.reached {
            return std::task::Poll::Ready(());
        }
        if !event
            .waker
            .as_ref()
            .is_some_and(|waker| waker.will_wake(task_cx.waker()))
        {
            event.waker = Some(task_cx.waker().clone());
        }
        std::task::Poll::Pending
    }

    /// Release the blocked choreography. Calling this more than once is safe.
    pub fn release(&self) {
        manifest_publish_pauses()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.directory);
        let (released, wake) = self.released.as_ref();
        *released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        wake.notify_all();
    }
}

#[cfg(test)]
impl Drop for ManifestPublishPauseControl {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
impl Drop for ManifestPublishFailureControl {
    fn drop(&mut self) {
        manifest_publish_failures()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.directory);
    }
}

/// Co-poll a real public publication with its armed filesystem checkpoint.
///
/// Publication can await earlier blocking preflight work before the pauseable
/// choreography. This helper therefore keeps polling the whole future while
/// also waiting on the checkpoint's registered waker. The timeout is an
/// asupersync timer diagnostic rather than a synchronous wait that would park
/// the current-thread executor and suppress the preflight wake-up.
#[cfg(test)]
pub async fn drive_manifest_publish_to_checkpoint_for_test<F: std::future::Future>(
    publish: &mut std::pin::Pin<Box<F>>,
    pause: &ManifestPublishPauseControl,
    timeout: Duration,
) -> Result<(), String> {
    let rendezvous = std::future::poll_fn(|task_cx| {
        if pause.poll_reached(task_cx).is_ready() {
            return std::task::Poll::Ready(Ok(()));
        }

        match publish.as_mut().poll(task_cx) {
            std::task::Poll::Ready(_) => std::task::Poll::Ready(Err(format!(
                "publication completed before the armed {:?} checkpoint",
                pause.checkpoint
            ))),
            std::task::Poll::Pending => {
                if pause.poll_reached(task_cx).is_ready() {
                    std::task::Poll::Ready(Ok(()))
                } else {
                    std::task::Poll::Pending
                }
            }
        }
    });

    asupersync::time::timeout(asupersync::time::wall_now(), timeout, rendezvous)
        .await
        .unwrap_or_else(|_| {
            Err(format!(
                "publication did not reach {:?} within {timeout:?}",
                pause.checkpoint
            ))
        })
}

#[cfg(test)]
pub(in crate::keeper) fn observe_manifest_publish_checkpoint_for_test(
    directory: &Path,
    checkpoint: PublishCheckpoint,
) -> Result<(), KeeperError> {
    let failure = manifest_publish_failures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(directory)
        .filter(|failure| failure.checkpoint == checkpoint)
        .cloned();
    if failure.is_some() {
        return Err(KeeperError::InvalidTransition {
            detail: format!("test-only MANIFEST publisher failure at {:?}", checkpoint),
        });
    }
    let pause = manifest_publish_pauses()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(directory)
        .filter(|pause| pause.checkpoint == checkpoint)
        .cloned();
    let Some(pause) = pause else {
        return Ok(());
    };
    let task_waker = {
        let mut event = pause
            .event
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        event.reached = true;
        event.waker.take()
    };
    if let Some(task_waker) = task_waker {
        task_waker.wake();
    }
    let (released, wake) = pause.released.as_ref();
    let mut released = released
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while !*released {
        released = wake
            .wait(released)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    drop(released);
    Ok(())
}

pub(in crate::keeper) fn publish_manifest_choreography<C, F, O>(
    directory: PathBuf,
    bytes: &[u8],
    claim: F,
    observe: O,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    publish_manifest_choreography_with_hooks(
        directory,
        bytes,
        claim,
        |_, _, _| Ok(()),
        |_, _, _, _| Ok(()),
        |_, _| Ok(()),
        observe,
    )
}

fn publish_manifest_choreography_with_hooks<C, F, B, M, A, O>(
    directory: PathBuf,
    bytes: &[u8],
    claim: F,
    mut before_slot_renames: B,
    mut after_current_move: M,
    mut before_directory_sync: A,
    mut observe: O,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
    B: FnMut(&Path, bool, u64) -> Result<(), KeeperError>,
    M: FnMut(&Path, &Path, &Path, u64) -> Result<(), KeeperError>,
    A: FnMut(&Path, &Path) -> Result<(), KeeperError>,
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    #[cfg(not(any(unix, windows)))]
    ensure_atomic_publish_supported(&directory)?;
    let proposed =
        Manifest::from_bytes(bytes).map_err(|source| KeeperError::InvalidManifest { source })?;
    let metadata = std::fs::metadata(&directory).map_err(|source| KeeperError::Io {
        operation: "inspect directory",
        path: directory.clone(),
        source,
    })?;
    if !metadata.is_dir() {
        return Err(KeeperError::Io {
            operation: "inspect directory",
            path: directory,
            source: io::Error::new(
                io::ErrorKind::NotADirectory,
                "index path is not a directory",
            ),
        });
    }

    let (expected_generation, rename_current, previous_manifest) =
        match load_manifest_pair(&directory) {
            Ok(loaded) => {
                if loaded.source == ManifestSource::PreviousAfterCorruptCurrent {
                    return Err(KeeperError::RecoveryRequired {
                        path: directory.join("MANIFEST"),
                    });
                }
                let expected = loaded.manifest.generation.checked_add(1).ok_or(
                    KeeperError::GenerationExhausted {
                        current: loaded.manifest.generation,
                    },
                )?;
                (
                    expected,
                    loaded.source == ManifestSource::Current,
                    Some(loaded.manifest),
                )
            }
            Err(KeeperError::IndexNotFound { .. }) => (1, false, None),
            Err(error) => return Err(error),
        };
    if proposed.generation != expected_generation {
        return Err(KeeperError::GenerationConflict {
            expected: expected_generation,
            proposed: proposed.generation,
        });
    }
    if let Some(previous) = &previous_manifest {
        if proposed.schema_id != previous.schema_id {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "schema_id changed from {:#018x} to {:#018x}",
                    previous.schema_id, proposed.schema_id
                ),
            });
        }
        if proposed.docid_high_watermark < previous.docid_high_watermark {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "docid_high_watermark rolled back from {} to {}",
                    previous.docid_high_watermark, proposed.docid_high_watermark
                ),
            });
        }
        // Shape re-check against the on-disk predecessor under the writer
        // LOCK. The live-document floor was already decided against the same
        // predecessor by `KeeperWriter::publish_with_intent`, which is the
        // only path into this choreography that carries a caller intent.
        validate_segment_transitions(previous, &proposed, LiveDocumentFloor::Waived)?;
    }

    let temp_path = directory.join(format!(".tmp-manifest-{}", proposed.generation));
    let current_path = directory.join("MANIFEST");
    let previous_path = directory.join("MANIFEST.prev");
    prepare_manifest_temp(&temp_path, bytes, &mut observe)?;

    let _claim_guard = claim(&directory, proposed.generation)?;
    observe(PublishCheckpoint::GenerationClaimed, &directory)?;
    // Only a rotation retires anything: without it the previous slot survives
    // this publication untouched, so the reachable set never shrinks. The
    // receipts are made durable here, ahead of the renames, so a crash can
    // only ever leave a witness for a retirement that did not complete --
    // never a completed retirement without its witness.
    if rename_current {
        let retired = retired_segment_ids(&directory, previous_manifest.as_ref(), &proposed)?;
        write_segment_retirement_receipts(&directory, &retired, proposed.generation)?;
    }
    before_slot_renames(&directory, rename_current, proposed.generation)?;
    if rename_current {
        replace_file_atomically(&current_path, &previous_path).map_err(|source| {
            KeeperError::Io {
                operation: "rename current to previous",
                path: current_path.clone(),
                source,
            }
        })?;
        after_current_move(
            &directory,
            &current_path,
            &previous_path,
            proposed.generation,
        )?;
        observe(PublishCheckpoint::CurrentMovedToPrevious, &previous_path)?;
    }
    replace_file_atomically(&temp_path, &current_path).map_err(|source| KeeperError::Io {
        operation: "rename temp to current",
        path: temp_path,
        source,
    })?;
    observe(PublishCheckpoint::TempMovedToCurrent, &current_path)?;
    before_directory_sync(&directory, &current_path)?;
    #[cfg(not(windows))]
    sync_directory(&directory)?;
    observe(PublishCheckpoint::DirectorySynced, &directory)?;

    Ok(LoadedManifest {
        manifest: proposed,
        source: ManifestSource::Current,
    })
}

fn prepare_manifest_temp<O>(path: &Path, bytes: &[u8], observe: &mut O) -> Result<(), KeeperError>
where
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    let mut temp_file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            verify_reusable_manifest_temp(path, bytes)?;
            observe(PublishCheckpoint::TempSynced, path)?;
            return Ok(());
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "create temp",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    temp_file
        .write_all(bytes)
        .map_err(|source| KeeperError::Io {
            operation: "write temp",
            path: path.to_path_buf(),
            source,
        })?;
    observe(PublishCheckpoint::TempWritten, path)?;
    temp_file.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync temp",
        path: path.to_path_buf(),
        source,
    })?;
    observe(PublishCheckpoint::TempSynced, path)
}

fn verify_reusable_manifest_temp(path: &Path, expected: &[u8]) -> Result<(), KeeperError> {
    let mut temp_file = open_existing_manifest_temp(path).map_err(|source| KeeperError::Io {
        operation: "open existing temp",
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = temp_file.metadata().map_err(|source| KeeperError::Io {
        operation: "stat existing temp",
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() || metadata.len() != usize_to_u64(expected.len()) {
        return Err(KeeperError::TempConflict {
            path: path.to_path_buf(),
        });
    }

    let mut existing = Vec::new();
    existing
        .try_reserve_exact(expected.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate existing temp buffer",
            path: path.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    Read::by_ref(&mut temp_file)
        .take(usize_to_u64(expected.len()).saturating_add(1))
        .read_to_end(&mut existing)
        .map_err(|source| KeeperError::Io {
            operation: "read existing temp",
            path: path.to_path_buf(),
            source,
        })?;
    if existing.as_slice() != expected {
        return Err(KeeperError::TempConflict {
            path: path.to_path_buf(),
        });
    }
    temp_file.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync existing temp",
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(unix)]
fn open_existing_manifest_temp(path: &Path) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, openat};

    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "temp path has no parent"))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "temp path has no file name"))?;
    let directory = File::open(parent)?;
    let temp_file = openat(
        &directory,
        file_name,
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)?;
    Ok(File::from(temp_file))
}

#[cfg(windows)]
fn open_existing_manifest_temp(path: &Path) -> io::Result<File> {
    open_windows_regular_file_no_reparse(path, true, false)
}

#[cfg(not(any(unix, windows)))]
fn open_existing_manifest_temp(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).open(path)
}
