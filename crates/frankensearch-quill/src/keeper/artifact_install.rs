use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use super::admission::WriterAdmissionInner;
#[cfg(windows)]
use super::admission::open_windows_generation_claim;
use super::errors::KeeperError;
use super::gc::safe_direct_child;
#[cfg(unix)]
use super::publish::stat_dev_as_u64;
use super::slot_io::replace_file_atomically;

/// Atomically create a fresh collision-free child file inside the admitted
/// directory with `create_new` (bd-b188): each attempt either claims the
/// name or proves it occupied, so no occupied staging name is ever
/// overwritten. Returns the claimed path and its open file.
#[allow(dead_code)] // called only on Tier-2 targets and by host tests
fn create_probed_file(
    admission: &WriterAdmissionInner,
    base: &str,
) -> Result<(PathBuf, File), KeeperError> {
    for collision in 0..=u16::MAX {
        let name = if collision == 0 {
            base.to_owned()
        } else {
            format!("{base}.{collision}")
        };
        safe_direct_child(&admission.directory, Path::new(&name))?;
        let candidate = admission.directory.join(&name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "create collision-safe staging file",
                    path: candidate,
                    source: error,
                });
            }
        }
    }
    Err(KeeperError::Io {
        operation: "allocate collision-free staging name",
        path: admission.directory.join(base),
        source: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "staging name suffix space is exhausted",
        ),
    })
}

pub(in crate::keeper) fn append_path_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

pub(in crate::keeper) fn retire_regular_artifact(
    admission: &WriterAdmissionInner,
    source: &Path,
    destination: &Path,
) -> Result<bool, KeeperError> {
    admission.ensure_directory_identity()?;
    if source.parent() != Some(admission.directory.as_path())
        || destination.parent() != Some(admission.directory.as_path())
    {
        return Err(KeeperError::Io {
            operation: "validate retirement paths",
            path: source.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "retirement paths must be direct children of the admitted directory",
            ),
        });
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use rustix::fs::{AtFlags, FileType, RenameFlags, renameat_with, statat};

        let source_name = source
            .file_name()
            .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                path: source.to_path_buf(),
            })?;
        let source_stat = match statat(
            &admission.directory_file,
            source_name,
            AtFlags::SYMLINK_NOFOLLOW,
        ) {
            Ok(stat) => stat,
            Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "inspect retirement source",
                    path: source.to_path_buf(),
                    source: io::Error::from(error),
                });
            }
        };
        if FileType::from_raw_mode(source_stat.st_mode) != FileType::RegularFile {
            return Err(KeeperError::Io {
                operation: "retire writer artifact",
                path: source.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "retirement source is not a regular file",
                ),
            });
        }

        let destination_name =
            destination
                .file_name()
                .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                    path: destination.to_path_buf(),
                })?;
        let mut collision = 0_u64;
        loop {
            let candidate = if collision == 0 {
                destination_name.to_os_string()
            } else {
                let mut name = destination_name.to_os_string();
                name.push(format!(".{collision}"));
                name
            };
            match renameat_with(
                &admission.directory_file,
                source_name,
                &admission.directory_file,
                &candidate,
                RenameFlags::NOREPLACE,
            ) {
                Ok(()) => return Ok(true),
                Err(error) if error == rustix::io::Errno::EXIST => {
                    collision = collision.checked_add(1).ok_or_else(|| KeeperError::Io {
                        operation: "allocate retirement destination",
                        path: destination.to_path_buf(),
                        source: io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "retirement destination suffix space is exhausted",
                        ),
                    })?;
                }
                Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
                Err(error) => {
                    return Err(KeeperError::Io {
                        operation: "atomically retire writer artifact",
                        path: admission.directory.join(candidate),
                        source: io::Error::from(error),
                    });
                }
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    retire_regular_artifact_portable(admission, source, destination)
}

/// Tier-2 no-replace retirement for platforms without `renameat2(NOREPLACE)`
/// (bd-b188), using only `std::fs`. Always compiled so host tests exercise
/// the same logic the cfg-selected fallback delegates to.
///
/// The destination suffix is claimed atomically with `create_new` (never an
/// occupied quarantine name), then the source is renamed over OUR OWN probe
/// file — portable across POSIX and Windows, and collision-safe by
/// construction. Returns `Ok(false)` when the source is absent.
#[allow(dead_code)] // called only on Tier-2 targets and by host tests
pub(in crate::keeper) fn retire_regular_artifact_portable(
    admission: &WriterAdmissionInner,
    source: &Path,
    destination: &Path,
) -> Result<bool, KeeperError> {
    admission.ensure_directory_identity()?;
    if source.parent() != Some(admission.directory.as_path())
        || destination.parent() != Some(admission.directory.as_path())
    {
        return Err(KeeperError::Io {
            operation: "validate retirement paths",
            path: source.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "retirement paths must be direct children of the admitted directory",
            ),
        });
    }
    match std::fs::symlink_metadata(source) {
        Ok(metadata) if metadata.file_type().is_file() => {}
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "retire writer artifact",
                path: source.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "retirement source is not a regular file",
                ),
            });
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(KeeperError::Io {
                operation: "inspect retirement source",
                path: source.to_path_buf(),
                source: error,
            });
        }
    }

    let base = destination
        .file_name()
        .ok_or_else(|| KeeperError::UnsafeGarbagePath {
            path: destination.to_path_buf(),
        })?
        .to_string_lossy()
        .into_owned();
    let (probe, probe_file) = create_probed_file(admission, &base)?;
    drop(probe_file);
    replace_file_atomically(source, &probe).map_err(|source_error| KeeperError::Io {
        operation: "retire writer artifact over claimed probe",
        path: probe,
        source: source_error,
    })?;
    Ok(true)
}

#[cfg(unix)]
pub(in crate::keeper) fn remove_stale_generation_claim(
    admission: &WriterAdmissionInner,
    path: &Path,
) -> Result<(), KeeperError> {
    use rustix::fs::{AtFlags, Mode, OFlags, openat, statat, unlinkat};
    use std::os::unix::fs::MetadataExt;

    admission.ensure_directory_identity()?;
    let name = path
        .file_name()
        .ok_or_else(|| KeeperError::UnsafeGarbagePath {
            path: path.to_path_buf(),
        })?;
    let claim = openat(
        &admission.directory_file,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(io::Error::from)
    .map_err(|source| KeeperError::Io {
        operation: "open stale generation claim",
        path: path.to_path_buf(),
        source,
    })?;
    let claim_file = File::from(claim);
    let metadata = claim_file.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect stale generation claim",
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_file() || metadata.len() != 0 {
        return Err(KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: "claim must be a zero-length regular file".to_owned(),
        });
    }
    let stat = statat(&admission.directory_file, name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "revalidate stale generation claim",
            path: path.to_path_buf(),
            source,
        })?;
    if stat_dev_as_u64(&stat) != metadata.dev()
        || stat.st_ino != metadata.ino()
        || stat.st_size != 0
    {
        return Err(KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: "claim pathname changed during stale recovery".to_owned(),
        });
    }
    unlinkat(&admission.directory_file, Path::new(name), AtFlags::empty())
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "remove stale generation claim",
            path: path.to_path_buf(),
            source,
        })?;
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync stale-claim removal",
            path: admission.directory.clone(),
            source,
        })
}

#[cfg(windows)]
pub(in crate::keeper) fn remove_stale_generation_claim(
    admission: &WriterAdmissionInner,
    path: &Path,
) -> Result<(), KeeperError> {
    admission.ensure_directory_identity()?;
    let (_, observed) = open_windows_generation_claim(path, false).map_err(|source| {
        KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: source.to_string(),
        }
    })?;
    let (_, revalidated) = open_windows_generation_claim(path, false).map_err(|source| {
        KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: source.to_string(),
        }
    })?;
    if observed != revalidated {
        return Err(KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: "claim pathname changed during stale recovery".to_owned(),
        });
    }
    std::fs::remove_file(path).map_err(|source| KeeperError::Io {
        operation: "remove stale Windows generation claim",
        path: path.to_path_buf(),
        source,
    })?;
    admission.ensure_directory_identity()
}

#[cfg(not(any(unix, windows)))]
pub(in crate::keeper) fn remove_stale_generation_claim(
    admission: &WriterAdmissionInner,
    path: &Path,
) -> Result<(), KeeperError> {
    Err(KeeperError::Io {
        operation: "remove stale generation claim",
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "stale claim recovery is unsupported for {}",
                admission.directory.display()
            ),
        ),
    })
}
