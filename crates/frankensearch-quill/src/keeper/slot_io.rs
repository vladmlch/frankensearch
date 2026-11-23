#[cfg(unix)]
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

#[cfg(windows)]
use super::admission::open_windows_regular_file_no_reparse;
use super::constants::MAX_MANIFEST_BYTES;
use super::errors::{KeeperError, ManifestCodecError};
use super::manifest::Manifest;
use super::manifest_validation::{non_canonical, usize_to_u64};

pub(in crate::keeper) enum ManifestSlot {
    Missing,
    Valid(Manifest),
    Invalid(ManifestCodecError),
}

pub(in crate::keeper) fn read_manifest_slot(path: &Path) -> Result<ManifestSlot, KeeperError> {
    let path_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(ManifestSlot::Missing);
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect slot path",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if !path_metadata.file_type().is_file() {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    let file = match open_manifest_slot(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(ManifestSlot::Missing),
        #[cfg(unix)]
        Err(error) if error.raw_os_error() == Some(rustix::io::Errno::LOOP.raw_os_error()) => {
            return Ok(ManifestSlot::Invalid(non_canonical(
                "manifest slot became a symlink while opening",
            )));
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "open",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    read_manifest_file(path, file)
}

pub(in crate::keeper) fn read_manifest_file(
    path: &Path,
    file: File,
) -> Result<ManifestSlot, KeeperError> {
    let file_length = file.metadata().map_err(|source| KeeperError::Io {
        operation: "stat",
        path: path.to_path_buf(),
        source,
    })?;
    if !file_length.file_type().is_file() {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    if file_length.len() > usize_to_u64(MAX_MANIFEST_BYTES) {
        return Ok(ManifestSlot::Invalid(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: file_length.len(),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        }));
    }
    let reserve = usize::try_from(file_length.len()).unwrap_or(MAX_MANIFEST_BYTES);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(reserve)
        .map_err(|error| KeeperError::Io {
            operation: "allocate read buffer",
            path: path.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    let read_limit = usize_to_u64(MAX_MANIFEST_BYTES).saturating_add(1);
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|source| KeeperError::Io {
            operation: "read",
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Ok(ManifestSlot::Invalid(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: usize_to_u64(bytes.len()),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        }));
    }
    Ok(match Manifest::from_bytes(&bytes) {
        Ok(manifest) => ManifestSlot::Valid(manifest),
        Err(error) => ManifestSlot::Invalid(error),
    })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
pub(in crate::keeper) fn read_manifest_slot_at(
    directory: &File,
    name: &OsStr,
    path: &Path,
) -> Result<ManifestSlot, KeeperError> {
    use rustix::fs::{AtFlags, FileType, Mode, OFlags, openat, statat};

    let stat = match statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(source) if source == rustix::io::Errno::NOENT => return Ok(ManifestSlot::Missing),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect slot path",
                path: path.to_path_buf(),
                source: io::Error::from(source),
            });
        }
    };
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    let file = match openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    ) {
        Ok(file) => File::from(file),
        Err(source) if source == rustix::io::Errno::NOENT => return Ok(ManifestSlot::Missing),
        Err(source) if source == rustix::io::Errno::LOOP => {
            return Ok(ManifestSlot::Invalid(non_canonical(
                "manifest slot became a symlink while opening",
            )));
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "open",
                path: path.to_path_buf(),
                source: io::Error::from(source),
            });
        }
    };
    read_manifest_file(path, file)
}

#[cfg(unix)]
pub(in crate::keeper) fn open_manifest_slot(path: &Path) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, openat};

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "slot path has no file name"))?;
    let directory = File::open(parent)?;
    let slot = openat(
        &directory,
        file_name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    Ok(File::from(slot))
}

#[cfg(windows)]
pub(in crate::keeper) fn open_manifest_slot(path: &Path) -> io::Result<File> {
    open_windows_regular_file_no_reparse(path, false, false)
}

#[cfg(not(any(unix, windows)))]
pub(in crate::keeper) fn open_manifest_slot(path: &Path) -> io::Result<File> {
    File::open(path)
}

#[cfg(windows)]
pub(in crate::keeper) fn replace_file_atomically(
    source: &Path,
    destination: &Path,
) -> io::Result<()> {
    atomicwrites::replace_atomic(source, destination)
}

#[cfg(not(windows))]
pub(in crate::keeper) fn replace_file_atomically(
    source: &Path,
    destination: &Path,
) -> io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(windows)]
pub(in crate::keeper) fn move_file_atomically(source: &Path, destination: &Path) -> io::Result<()> {
    atomicwrites::move_atomic(source, destination)
}

#[cfg(not(windows))]
pub(in crate::keeper) fn move_file_atomically(source: &Path, destination: &Path) -> io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(windows)]
pub(in crate::keeper) fn ensure_atomic_publish_supported(
    directory: &Path,
) -> Result<(), KeeperError> {
    Err(KeeperError::Io {
        operation: "verify descriptor-relative writer maintenance support",
        path: directory.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "descriptor-relative Quill garbage collection is not yet available on Windows",
        ),
    })
}

#[cfg(not(any(unix, windows)))]
pub(in crate::keeper) fn ensure_atomic_publish_supported(
    directory: &Path,
) -> Result<(), KeeperError> {
    Err(KeeperError::Io {
        operation: "verify atomic publish support",
        path: directory.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "MANIFEST replacement and directory fsync require Unix semantics",
        ),
    })
}

// Windows publication renames use `atomicwrites` with MOVEFILE_WRITE_THROUGH;
// callers omit the separate directory fsync there.
#[cfg(unix)]
pub(in crate::keeper) fn sync_directory(directory: &Path) -> Result<(), KeeperError> {
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "fsync directory",
            path: directory.to_path_buf(),
            source,
        })
}

#[cfg(not(any(unix, windows)))]
pub(in crate::keeper) fn sync_directory(directory: &Path) -> Result<(), KeeperError> {
    ensure_atomic_publish_supported(directory)
}
