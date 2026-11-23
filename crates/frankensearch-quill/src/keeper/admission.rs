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
use std::ffi::OsString;
use std::fs::File;
#[cfg(windows)]
use std::fs::OpenOptions;
use std::io::{self, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use super::constants::{WRITER_LOCK_FORMAT_VERSION, WRITER_LOCK_MAGIC, WRITER_LOCK_RECORD_BYTES};
use super::errors::KeeperError;
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
use super::gc::safe_direct_child;
#[cfg(unix)]
use super::gc::{ensure_gc_directory_identity, open_gc_directory};
use super::manifest_validation::usize_to_u64;
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
use super::publish::stat_dev_as_u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::keeper) struct WriterLockRecord {
    pub(in crate::keeper) pid: u32,
    pub(in crate::keeper) pid_start_nonce: u64,
    pub(in crate::keeper) acquired_unix_s: i64,
}

impl WriterLockRecord {
    pub(in crate::keeper) fn current(path: &Path) -> Result<Self, KeeperError> {
        let pid = std::process::id();
        if i32::try_from(pid).is_err() || pid == 0 {
            return Err(KeeperError::WriterLockCorrupted {
                path: path.to_path_buf(),
                detail: format!("current process id {pid} is outside the POSIX pid range"),
            });
        }
        let acquired_unix_s = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .and_then(|elapsed| i64::try_from(elapsed.as_secs()).ok())
            .ok_or_else(|| KeeperError::WriterLockCorrupted {
                path: path.to_path_buf(),
                detail: "current wall clock is outside the v1 lock-record range".to_owned(),
            })?;
        Ok(Self {
            pid,
            pid_start_nonce: writer_pid_start_nonce(pid, acquired_unix_s),
            acquired_unix_s,
        })
    }

    pub(in crate::keeper) fn to_bytes(self) -> [u8; WRITER_LOCK_RECORD_BYTES] {
        let mut bytes = [0_u8; WRITER_LOCK_RECORD_BYTES];
        bytes[..8].copy_from_slice(&WRITER_LOCK_MAGIC);
        bytes[8..12].copy_from_slice(&WRITER_LOCK_FORMAT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.pid.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.pid_start_nonce.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.acquired_unix_s.to_le_bytes());
        let checksum = crc32fast::hash(&bytes[..32]);
        bytes[32..36].copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    pub(in crate::keeper) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != WRITER_LOCK_RECORD_BYTES {
            return Err(format!(
                "v1 record must be exactly {WRITER_LOCK_RECORD_BYTES} bytes, found {}",
                bytes.len()
            ));
        }
        if bytes[..8] != WRITER_LOCK_MAGIC {
            return Err("invalid writer-lock magic".to_owned());
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != WRITER_LOCK_FORMAT_VERSION {
            return Err(format!("unsupported writer-lock version {version}"));
        }
        let stored_checksum = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        let computed_checksum = crc32fast::hash(&bytes[..32]);
        if stored_checksum != computed_checksum {
            return Err(format!(
                "writer-lock CRC mismatch: stored {stored_checksum:#010x}, computed {computed_checksum:#010x}"
            ));
        }
        let pid = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if pid == 0 {
            return Err(format!(
                "writer-lock pid {pid} is not a valid process identifier"
            ));
        }
        #[cfg(unix)]
        if i32::try_from(pid).is_err() {
            return Err(format!(
                "writer-lock pid {pid} is outside the POSIX pid range"
            ));
        }
        let pid_start_nonce = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let acquired_unix_s = i64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        if acquired_unix_s < 0 {
            return Err(format!(
                "writer-lock acquisition timestamp {acquired_unix_s} predates the Unix epoch"
            ));
        }
        Ok(Self {
            pid,
            pid_start_nonce,
            acquired_unix_s,
        })
    }
}

fn writer_pid_start_nonce(pid: u32, acquired_unix_s: i64) -> u64 {
    #[cfg(target_os = "linux")]
    if let Some(start_time) = linux_process_start_time(pid) {
        return writer_pid_start_nonce_from_linux_start_time(pid, start_time);
    }

    let mut identity = [0_u8; 12];
    identity[..4].copy_from_slice(&pid.to_le_bytes());
    identity[4..].copy_from_slice(&acquired_unix_s.to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&identity)
}

#[cfg(target_os = "linux")]
pub(in crate::keeper) fn writer_pid_start_nonce_from_linux_start_time(
    pid: u32,
    start_time: u64,
) -> u64 {
    let mut identity = [0_u8; 12];
    identity[..4].copy_from_slice(&pid.to_le_bytes());
    identity[4..].copy_from_slice(&start_time.to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&identity)
}

#[cfg(target_os = "linux")]
pub(in crate::keeper) fn linux_process_start_time(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_command = stat.get(stat.rfind(')')?.checked_add(1)?..)?;
    // The first token after the command is field 3 (`state`); starttime is
    // field 22, hence zero-based token 19 in this suffix.
    after_command.split_whitespace().nth(19)?.parse().ok()
}

pub(in crate::keeper) struct WriterAdmissionInner {
    pub(in crate::keeper) directory: PathBuf,
    pub(in crate::keeper) directory_file: File,
    #[cfg(windows)]
    pub(in crate::keeper) directory_identity: same_file::Handle,
    pub(in crate::keeper) lock_path: PathBuf,
    pub(in crate::keeper) lock_file: File,
    #[cfg(unix)]
    pub(in crate::keeper) lock_device: u64,
    #[cfg(unix)]
    pub(in crate::keeper) lock_inode: u64,
    #[cfg(windows)]
    pub(in crate::keeper) lock_identity: same_file::Handle,
    pub(in crate::keeper) record: WriterLockRecord,
}

impl WriterAdmissionInner {
    #[cfg(unix)]
    pub(in crate::keeper) fn ensure_directory_identity(&self) -> Result<(), KeeperError> {
        ensure_gc_directory_identity(&self.directory, &self.directory_file)?;
        ensure_writer_lock_identity(self)
    }

    #[cfg(windows)]
    pub(in crate::keeper) fn ensure_directory_identity(&self) -> Result<(), KeeperError> {
        ensure_windows_directory_identity(self)?;
        ensure_windows_writer_lock_identity(self)
    }

    #[cfg(not(any(unix, windows)))]
    pub(in crate::keeper) fn ensure_directory_identity(&self) -> Result<(), KeeperError> {
        Err(KeeperError::Io {
            operation: "verify writer-lock directory identity",
            path: self.directory.clone(),
            source: io::Error::new(
                io::ErrorKind::Unsupported,
                "cross-process Quill writer admission requires Unix directory identity",
            ),
        })
    }
}

#[cfg(windows)]
fn windows_path_is_reparse_point(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;

    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(windows)]
fn open_windows_admission_directory(
    directory: &Path,
) -> Result<(File, same_file::Handle), KeeperError> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };

    let file = OpenOptions::new()
        .read(true)
        // Deliberately omit FILE_SHARE_DELETE. Keeping this handle alive makes
        // directory rename/replacement fail while admission is retained.
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(directory)
        .map_err(|source| KeeperError::Io {
            operation: "open no-follow writer directory",
            path: directory.to_path_buf(),
            source,
        })?;
    let metadata = file.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect writer directory",
        path: directory.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_dir() || windows_path_is_reparse_point(&metadata) {
        return Err(KeeperError::WriterLockCorrupted {
            path: directory.to_path_buf(),
            detail: "writer directory must be a no-follow, non-reparse directory".to_owned(),
        });
    }
    let identity =
        same_file::Handle::from_file(file.try_clone().map_err(|source| KeeperError::Io {
            operation: "clone writer-directory handle",
            path: directory.to_path_buf(),
            source,
        })?)
        .map_err(|source| KeeperError::Io {
            operation: "identify writer directory",
            path: directory.to_path_buf(),
            source,
        })?;
    Ok((file, identity))
}

#[cfg(windows)]
fn open_windows_writer_lock(
    path: &Path,
    create: bool,
) -> Result<(File, same_file::Handle), KeeperError> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_FLAG_WRITE_THROUGH, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };

    match std::fs::symlink_metadata(path) {
        Ok(metadata)
            if !metadata.file_type().is_file() || windows_path_is_reparse_point(&metadata) =>
        {
            return Err(KeeperError::WriterLockCorrupted {
                path: path.to_path_buf(),
                detail: "LOCK must be a no-follow, non-reparse regular file".to_owned(),
            });
        }
        Ok(_) => {}
        Err(source) if create && source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect writer-lock pathname",
                path: path.to_path_buf(),
                source,
            });
        }
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(create)
        // The lock record remains readable by a contender, but omitting
        // FILE_SHARE_DELETE prevents pathname replacement while this handle
        // carries the authoritative byte-range lock.
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_WRITE_THROUGH)
        .open(path)
        .map_err(|source| KeeperError::Io {
            operation: "open no-follow writer lock",
            path: path.to_path_buf(),
            source,
        })?;
    let metadata = file.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect writer lock",
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_file() || windows_path_is_reparse_point(&metadata) {
        return Err(KeeperError::WriterLockCorrupted {
            path: path.to_path_buf(),
            detail: "LOCK must be a no-follow, non-reparse regular file".to_owned(),
        });
    }
    let identity =
        same_file::Handle::from_file(file.try_clone().map_err(|source| KeeperError::Io {
            operation: "clone writer-lock handle",
            path: path.to_path_buf(),
            source,
        })?)
        .map_err(|source| KeeperError::Io {
            operation: "identify writer lock",
            path: path.to_path_buf(),
            source,
        })?;
    Ok((file, identity))
}

#[cfg(windows)]
pub(in crate::keeper) fn open_windows_generation_claim(
    path: &Path,
    create_new: bool,
) -> io::Result<(File, same_file::Handle)> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_FLAG_WRITE_THROUGH, FILE_SHARE_DELETE, FILE_SHARE_READ,
        FILE_SHARE_WRITE,
    };

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(create_new)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_WRITE_THROUGH)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file()
        || windows_path_is_reparse_point(&metadata)
        || metadata.len() != 0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "generation claim must be a no-follow, non-reparse, zero-length regular file",
        ));
    }
    let identity = same_file::Handle::from_file(file.try_clone()?)?;
    Ok((file, identity))
}

#[cfg(windows)]
pub(in crate::keeper) fn open_windows_regular_file_no_reparse(
    path: &Path,
    write: bool,
    create_new: bool,
) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };

    let file = OpenOptions::new()
        .read(true)
        .write(write)
        .create_new(create_new)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() || windows_path_is_reparse_point(&metadata) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "path must be a no-follow, non-reparse regular file",
        ));
    }
    Ok(file)
}

#[cfg(windows)]
fn ensure_windows_directory_identity(admission: &WriterAdmissionInner) -> Result<(), KeeperError> {
    let (_, observed) = open_windows_admission_directory(&admission.directory)?;
    if observed != admission.directory_identity {
        return Err(KeeperError::WriterLockCorrupted {
            path: admission.directory.clone(),
            detail: "writer directory pathname no longer resolves to the admitted handle"
                .to_owned(),
        });
    }
    Ok(())
}

#[cfg(windows)]
fn ensure_windows_writer_lock_identity(
    admission: &WriterAdmissionInner,
) -> Result<(), KeeperError> {
    let (_, observed) = open_windows_writer_lock(&admission.lock_path, false)?;
    if observed != admission.lock_identity {
        return Err(KeeperError::WriterLockCorrupted {
            path: admission.lock_path.clone(),
            detail: "LOCK pathname no longer resolves to the locked handle".to_owned(),
        });
    }
    Ok(())
}

impl Drop for WriterAdmissionInner {
    fn drop(&mut self) {
        release_writer_admission(self);
    }
}

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
pub(in crate::keeper) fn acquire_writer_admission(
    directory: &Path,
) -> Result<Arc<WriterAdmissionInner>, KeeperError> {
    use rustix::fs::{FlockOperation, Mode, OFlags, flock, openat};

    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let lock_path = directory.join("LOCK");
    let lock = openat(
        &directory_file,
        "LOCK",
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)
    .map_err(|source| KeeperError::Io {
        operation: "open no-follow writer lock",
        path: lock_path.clone(),
        source,
    })?;
    let mut previous_lock_file = File::from(lock);
    let metadata = previous_lock_file
        .metadata()
        .map_err(|source| KeeperError::Io {
            operation: "inspect writer lock",
            path: lock_path.clone(),
            source,
        })?;
    if !metadata.file_type().is_file() {
        return Err(KeeperError::WriterLockCorrupted {
            path: lock_path,
            detail: "LOCK must be a no-follow regular file".to_owned(),
        });
    }
    use std::os::unix::fs::MetadataExt;
    let previous_lock_device = metadata.dev();
    let previous_lock_inode = metadata.ino();
    if let Err(source) = flock(
        &previous_lock_file,
        FlockOperation::NonBlockingLockExclusive,
    ) {
        if source == rustix::io::Errno::AGAIN {
            let owner_pid = read_writer_lock_record(&lock_path, &mut previous_lock_file)
                .ok()
                .flatten()
                .map(|record| record.pid);
            return Err(KeeperError::WriterBusy {
                path: lock_path,
                owner_pid,
            });
        }
        return Err(KeeperError::Io {
            operation: "acquire writer flock",
            path: lock_path,
            source: io::Error::from(source),
        });
    }

    if !writer_lock_path_matches(&lock_path, previous_lock_device, previous_lock_inode)? {
        return Err(KeeperError::WriterLockCorrupted {
            path: lock_path,
            detail: "LOCK pathname changed before writer admission completed".to_owned(),
        });
    }

    // A won flock is the authoritative proof that no cooperating writer owns
    // this inode. The record remains a conservative cross-check for a live
    // process on platforms where the recorded start identity can be observed.
    // A corrupt record cannot strand the directory forever: replace it
    // atomically while the authoritative flock is held.
    let previous = match read_writer_lock_record(&lock_path, &mut previous_lock_file) {
        Ok(record) => record,
        Err(KeeperError::WriterLockCorrupted { .. }) => None,
        Err(error) => return Err(error),
    };
    if let Some(previous) = previous
        && previous.pid != std::process::id()
        && writer_lock_record_names_live_owner(previous)
    {
        return Err(KeeperError::WriterBusy {
            path: lock_path,
            owner_pid: Some(previous.pid),
        });
    }

    let record = WriterLockRecord::current(&lock_path)?;
    let (lock_file, lock_device, lock_inode) =
        publish_writer_lock_record(&directory_file, directory, &lock_path, record)?;
    drop(previous_lock_file);
    let admission = Arc::new(WriterAdmissionInner {
        directory: directory.to_path_buf(),
        directory_file,
        lock_path,
        lock_file,
        lock_device,
        lock_inode,
        record,
    });
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync writer-lock directory",
            path: directory.to_path_buf(),
            source,
        })?;
    admission.ensure_directory_identity()?;
    Ok(admission)
}

#[cfg(windows)]
pub(in crate::keeper) fn acquire_writer_admission(
    directory: &Path,
) -> Result<Arc<WriterAdmissionInner>, KeeperError> {
    use std::fs::TryLockError;

    let (directory_file, directory_identity) = open_windows_admission_directory(directory)?;
    let lock_path = directory.join("LOCK");
    let (mut lock_file, lock_identity) = open_windows_writer_lock(&lock_path, true)?;
    match lock_file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            let owner_pid = read_writer_lock_record(&lock_path, &mut lock_file)
                .ok()
                .flatten()
                .map(|record| record.pid);
            return Err(KeeperError::WriterBusy {
                path: lock_path,
                owner_pid,
            });
        }
        Err(TryLockError::Error(source)) => {
            return Err(KeeperError::Io {
                operation: "acquire Windows writer lock",
                path: lock_path,
                source,
            });
        }
    }

    let record = WriterLockRecord::current(&lock_path)?;
    let bytes = record.to_bytes();
    lock_file
        .set_len(0)
        .and_then(|()| lock_file.seek(SeekFrom::Start(0)).map(|_| ()))
        .and_then(|()| lock_file.write_all(&bytes))
        .and_then(|()| lock_file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "persist Windows writer-lock record",
            path: lock_path.clone(),
            source,
        })?;
    let admission = Arc::new(WriterAdmissionInner {
        directory: directory.to_path_buf(),
        directory_file,
        directory_identity,
        lock_path,
        lock_file,
        lock_identity,
        record,
    });
    admission.ensure_directory_identity()?;
    if read_writer_lock_record(
        &admission.lock_path,
        &mut admission
            .lock_file
            .try_clone()
            .map_err(|source| KeeperError::Io {
                operation: "clone Windows writer-lock record handle",
                path: admission.lock_path.clone(),
                source,
            })?,
    )? != Some(record)
    {
        return Err(KeeperError::WriterLockCorrupted {
            path: admission.lock_path.clone(),
            detail: "persisted Windows writer-lock record did not round-trip".to_owned(),
        });
    }
    Ok(admission)
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
pub(in crate::keeper) fn acquire_writer_admission(
    directory: &Path,
) -> Result<Arc<WriterAdmissionInner>, KeeperError> {
    Err(KeeperError::Io {
        operation: "verify writer-lock support",
        path: directory.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "cross-process Quill writer admission is unsupported on this target",
        ),
    })
}

pub(in crate::keeper) fn read_writer_lock_record(
    path: &Path,
    file: &mut File,
) -> Result<Option<WriterLockRecord>, KeeperError> {
    let length = file
        .metadata()
        .map_err(|source| KeeperError::Io {
            operation: "inspect writer-lock record",
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if length == 0 {
        return Ok(None);
    }
    if length != usize_to_u64(WRITER_LOCK_RECORD_BYTES) {
        return Err(KeeperError::WriterLockCorrupted {
            path: path.to_path_buf(),
            detail: format!(
                "v1 record must be exactly {WRITER_LOCK_RECORD_BYTES} bytes, found {length}"
            ),
        });
    }
    let mut bytes = [0_u8; WRITER_LOCK_RECORD_BYTES];
    file.seek(SeekFrom::Start(0))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(|source| KeeperError::Io {
            operation: "read writer-lock record",
            path: path.to_path_buf(),
            source,
        })?;
    WriterLockRecord::from_bytes(&bytes)
        .map(Some)
        .map_err(|detail| KeeperError::WriterLockCorrupted {
            path: path.to_path_buf(),
            detail,
        })
}

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
fn publish_writer_lock_record(
    directory_file: &File,
    directory: &Path,
    path: &Path,
    record: WriterLockRecord,
) -> Result<(File, u64, u64), KeeperError> {
    use rustix::fs::{
        AtFlags, FileType, FlockOperation, Mode, OFlags, flock, openat, renameat, statat,
    };
    use std::os::unix::fs::MetadataExt;

    let bytes = record.to_bytes();
    let base = format!(".tmp-lock-{:016x}", record.pid_start_nonce);
    let mut collision = 0_u64;
    let (temporary_name, mut file, metadata) = loop {
        let name = if collision == 0 {
            OsString::from(&base)
        } else {
            OsString::from(format!("{base}.{collision}"))
        };
        safe_direct_child(directory, Path::new(&name))?;
        match openat(
            directory_file,
            &name,
            OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        ) {
            Ok(file) => {
                let file = File::from(file);
                let metadata = file.metadata().map_err(|source| KeeperError::Io {
                    operation: "inspect writer-lock temp",
                    path: directory.join(&name),
                    source,
                })?;
                break (name, file, metadata);
            }
            Err(source) if source == rustix::io::Errno::EXIST => {
                collision = collision.checked_add(1).ok_or_else(|| KeeperError::Io {
                    operation: "allocate writer-lock temp",
                    path: directory.join(&base),
                    source: io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "writer-lock temp suffix space is exhausted",
                    ),
                })?;
            }
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "create writer-lock temp",
                    path: directory.join(&name),
                    source: io::Error::from(source),
                });
            }
        }
    };
    let temporary_path = directory.join(&temporary_name);
    if !metadata.file_type().is_file() || metadata.len() != 0 {
        return Err(KeeperError::WriterLockCorrupted {
            path: temporary_path,
            detail: "new writer-lock temp is not an empty regular file".to_owned(),
        });
    }
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "persist writer-lock temp",
            path: temporary_path.clone(),
            source,
        })?;
    flock(&file, FlockOperation::NonBlockingLockExclusive)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "lock writer-lock temp",
            path: temporary_path.clone(),
            source,
        })?;
    let temp_stat = statat(directory_file, &temporary_name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "verify writer-lock temp",
            path: temporary_path.clone(),
            source,
        })?;
    if FileType::from_raw_mode(temp_stat.st_mode) != FileType::RegularFile
        || stat_dev_as_u64(&temp_stat) != metadata.dev()
        || temp_stat.st_ino != metadata.ino()
        || temp_stat.st_size != 36
    {
        return Err(KeeperError::WriterLockCorrupted {
            path: temporary_path,
            detail: "writer-lock temp pathname no longer names the synced locked inode".to_owned(),
        });
    }
    renameat(directory_file, &temporary_name, directory_file, "LOCK")
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "atomically publish writer-lock record",
            path: path.to_path_buf(),
            source,
        })?;
    directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync writer-lock directory",
            path: directory.to_path_buf(),
            source,
        })?;
    if !writer_lock_path_matches(path, metadata.dev(), metadata.ino())? {
        return Err(KeeperError::WriterLockCorrupted {
            path: path.to_path_buf(),
            detail: "published LOCK pathname does not name the synced locked inode".to_owned(),
        });
    }
    Ok((file, metadata.dev(), metadata.ino()))
}

pub(in crate::keeper) fn writer_lock_record_names_live_owner(record: WriterLockRecord) -> bool {
    if writer_pid_is_dead(record.pid) {
        return false;
    }

    // A live pid whose start identity cannot be inspected remains a
    // conservative live-owner result. Only a positively observed nonce
    // mismatch proves PID reuse.
    #[cfg(target_os = "linux")]
    {
        let Some(start_time) = linux_process_start_time(record.pid) else {
            return true;
        };
        writer_pid_start_nonce_from_linux_start_time(record.pid, start_time)
            == record.pid_start_nonce
    }

    #[cfg(not(target_os = "linux"))]
    true
}

#[cfg(unix)]
fn writer_pid_is_dead(pid: u32) -> bool {
    let Ok(raw_pid) = i32::try_from(pid) else {
        return false;
    };
    let Some(pid) = rustix::process::Pid::from_raw(raw_pid) else {
        return false;
    };
    matches!(
        rustix::process::test_kill_process(pid),
        Err(source) if source == rustix::io::Errno::SRCH
    )
}

#[cfg(not(unix))]
fn writer_pid_is_dead(_: u32) -> bool {
    false
}

#[cfg(unix)]
fn writer_lock_path_matches(path: &Path, device: u64, inode: u64) -> Result<bool, KeeperError> {
    use std::os::unix::fs::MetadataExt;

    let metadata = std::fs::symlink_metadata(path).map_err(|source| KeeperError::Io {
        operation: "verify writer-lock pathname",
        path: path.to_path_buf(),
        source,
    })?;
    Ok(metadata.file_type().is_file() && metadata.dev() == device && metadata.ino() == inode)
}

#[cfg(unix)]
fn ensure_writer_lock_identity(admission: &WriterAdmissionInner) -> Result<(), KeeperError> {
    if !writer_lock_path_matches(
        &admission.lock_path,
        admission.lock_device,
        admission.lock_inode,
    )? {
        return Err(KeeperError::WriterLockCorrupted {
            path: admission.lock_path.clone(),
            detail: "LOCK pathname no longer resolves to the flocked inode".to_owned(),
        });
    }
    Ok(())
}

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
fn release_writer_admission(admission: &mut WriterAdmissionInner) {
    use rustix::fs::{FlockOperation, flock};

    if read_writer_lock_record(&admission.lock_path, &mut admission.lock_file)
        .is_ok_and(|record| record == Some(admission.record))
    {
        let _ = admission.lock_file.set_len(0);
        let _ = admission.lock_file.sync_all();
    }
    let _ = flock(&admission.lock_file, FlockOperation::Unlock);
}

#[cfg(windows)]
fn release_writer_admission(admission: &mut WriterAdmissionInner) {
    if admission.ensure_directory_identity().is_ok()
        && read_writer_lock_record(&admission.lock_path, &mut admission.lock_file)
            .is_ok_and(|record| record == Some(admission.record))
    {
        let _ = admission.lock_file.set_len(0);
        let _ = admission.lock_file.sync_all();
    }
    let _ = admission.lock_file.unlock();
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
fn release_writer_admission(_: &mut WriterAdmissionInner) {}
