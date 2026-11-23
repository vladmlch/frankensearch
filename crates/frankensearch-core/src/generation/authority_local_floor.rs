use std::collections::BTreeMap;
use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use super::authority_core::{
    AUTHORITY_REF_BYTES_V1, AntiRollbackFloorRecordV1, AuthorityFloorV1, AuthorityRefV1,
    GenerationAuthorityErrorV1, constant_time_fingerprint_eq, predecessor_matches_authority,
};
use super::codec::CanonicalDecoder;

/// Magic prefix of one durable anti-rollback floor version file.
const LOCAL_FLOOR_FILE_MAGIC_V1: [u8; 8] = *b"FSARFLR1";
/// Exact byte length of one durable floor version file (bd-ynwdi).
///
/// `magic` 8 | `schema` 2 | `root_id` 16 | `authority` 108 | `cas_version` 8 |
/// `record_sha256` 32 | `idempotency_key` 16 | `expected_present` 1 |
/// `expected_record_sha256` 32 | `file_sha256` 32.
pub(super) const LOCAL_FLOOR_FILE_BYTES_V1: usize =
    8 + 2 + 16 + AUTHORITY_REF_BYTES_V1 + 8 + 32 + 16 + 1 + 32 + 32;
pub(super) const LOCAL_FLOOR_FILE_SUFFIX_V1: &str = ".floor";

/// Crash-durable local reference anti-rollback floor provider (bd-ynwdi).
///
/// This is the `CooperativeLocal` profile's owner-controlled floor. It lives
/// in a directory the caller chooses **outside** any replaceable generation
/// root, so replaying an older root cannot replay the floor with it. It
/// detects crashes and ordinary stale roots; it does not defend against a
/// hostile whole-store rollback that also rewinds this directory, which is
/// exactly why the profile is named cooperative rather than external.
///
/// Semantics are identical to [`super::InMemoryAntiRollbackFloorStoreV1`]:
/// linearizable per root, monotone sequence, exact predecessor linkage,
/// idempotent replay, and exactly one same-base winner. Durability and
/// linearizability come from the filesystem rather than a mutex:
///
/// * a persistent sibling `<root_hex>.lock` serializes readers and publishers
///   through scan, write, and durability completion. Each operation opens its
///   own descriptor; the kernel releases its lock on close or process exit;
/// * every CAS version is its own immutable file `<root_hex>/v<cas>.floor`,
///   created with `create_new`. The lock prevents a competing scan from
///   mistaking an in-progress write for an abandoned torn record;
/// * each file ends in a SHA-256 of its own bytes, so a torn or partial write
///   never reads as a floor: a highest-numbered file that fails to verify is
///   reported as [`GenerationAuthorityErrorV1::UnresolvedAttempt`], which
///   fails every further advance closed until an operator reconciles it —
///   files are never deleted or rewritten by this store;
/// * the file is `fsync`ed and then its directory is `fsync`ed before the
///   receipt is returned, so an acknowledged advance survives power loss.
///
/// The idempotency journal is the record file itself: the request's key and
/// expected receipt digest are stored beside the result, so a replay finds
/// its original receipt after a restart and a rebinding attempt conflicts.
#[derive(Debug, Clone)]
pub struct LocalAntiRollbackFloorStoreV1 {
    directory: std::path::PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LocalFloorFileV1 {
    pub(super) record: AntiRollbackFloorRecordV1,
    pub(super) idempotency_key: [u8; 16],
    pub(super) expected_record_sha256: Option<[u8; 32]>,
}

impl LocalFloorFileV1 {
    pub(super) fn encode(&self) -> [u8; LOCAL_FLOOR_FILE_BYTES_V1] {
        let mut bytes = [0_u8; LOCAL_FLOOR_FILE_BYTES_V1];
        let mut at = 0;
        let mut put = |chunk: &[u8]| {
            bytes[at..at + chunk.len()].copy_from_slice(chunk);
            at += chunk.len();
        };
        put(&LOCAL_FLOOR_FILE_MAGIC_V1);
        put(&self.record.schema_version.to_be_bytes());
        put(&self.record.root_id);
        put(&self.record.authority.canonical_bytes());
        put(&self.record.cas_version.to_be_bytes());
        put(&self.record.record_sha256);
        put(&self.idempotency_key);
        put(&[u8::from(self.expected_record_sha256.is_some())]);
        put(&self.expected_record_sha256.unwrap_or([0; 32]));
        let body_len = LOCAL_FLOOR_FILE_BYTES_V1 - 32;
        let digest: [u8; 32] = Sha256::digest(&bytes[..body_len]).into();
        bytes[body_len..].copy_from_slice(&digest);
        bytes
    }

    pub(super) fn decode(bytes: &[u8]) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() != LOCAL_FLOOR_FILE_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "local_floor_file.len",
            });
        }
        let body_len = LOCAL_FLOOR_FILE_BYTES_V1 - 32;
        let digest: [u8; 32] = Sha256::digest(&bytes[..body_len]).into();
        let stored: [u8; 32] =
            bytes[body_len..]
                .try_into()
                .map_err(|_| GenerationAuthorityErrorV1::InvalidField {
                    field: "local_floor_file.sha256",
                })?;
        if !constant_time_fingerprint_eq(&digest, &stored) {
            return Err(GenerationAuthorityErrorV1::ChecksumMismatch);
        }
        if bytes[..8] != LOCAL_FLOOR_FILE_MAGIC_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "local_floor_file.magic",
            });
        }
        let mut decoder = CanonicalDecoder::new(&bytes[8..body_len]);
        let schema_version = decoder.u16("local_floor_file.schema_version")?;
        let root_id: [u8; 16] = decoder
            .take(16, "local_floor_file.root_id")?
            .try_into()
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField {
                field: "local_floor_file.root_id",
            })?;
        let authority = AuthorityRefV1::from_canonical_bytes(
            decoder.take(AUTHORITY_REF_BYTES_V1, "local_floor_file.authority")?,
        )?;
        let cas_version = decoder.u64("local_floor_file.cas_version")?;
        let record_sha256: [u8; 32] = decoder
            .take(32, "local_floor_file.record_sha256")?
            .try_into()
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField {
                field: "local_floor_file.record_sha256",
            })?;
        let idempotency_key: [u8; 16] = decoder
            .take(16, "local_floor_file.idempotency_key")?
            .try_into()
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField {
                field: "local_floor_file.idempotency_key",
            })?;
        let expected_present = decoder.take(1, "local_floor_file.expected_present")?[0];
        let expected_bytes: [u8; 32] = decoder
            .take(32, "local_floor_file.expected_record_sha256")?
            .try_into()
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField {
                field: "local_floor_file.expected_record_sha256",
            })?;
        decoder.finish()?;
        let expected_record_sha256 = match expected_present {
            0 if expected_bytes == [0; 32] => None,
            0 => return Err(GenerationAuthorityErrorV1::NonCanonicalPadding),
            1 => Some(expected_bytes),
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "local_floor_file.expected_present",
                });
            }
        };
        let record = AntiRollbackFloorRecordV1 {
            schema_version,
            root_id,
            authority,
            cas_version,
            record_sha256,
        };
        record.validate()?;
        Ok(Self {
            record,
            idempotency_key,
            expected_record_sha256,
        })
    }
}

fn floor_io_unavailable(_: std::io::Error) -> GenerationAuthorityErrorV1 {
    GenerationAuthorityErrorV1::FloorStoreUnavailable
}

impl LocalAntiRollbackFloorStoreV1 {
    /// Bind a store to `directory`, creating it if absent.
    ///
    /// The directory must not live inside a generation root.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationAuthorityErrorV1::FloorStoreUnavailable`] when the
    /// directory cannot be created.
    pub fn open(
        directory: impl Into<std::path::PathBuf>,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let directory = directory.into();
        std::fs::create_dir_all(&directory).map_err(floor_io_unavailable)?;
        Ok(Self { directory })
    }

    /// Directory this store persists into.
    #[must_use]
    pub fn directory(&self) -> &std::path::Path {
        &self.directory
    }

    pub(super) fn root_directory(&self, root_id: [u8; 16]) -> std::path::PathBuf {
        let mut name = String::with_capacity(32);
        for byte in root_id {
            let _ = write!(name, "{byte:02x}");
        }
        self.directory.join(name)
    }

    pub(super) fn version_path(
        root_directory: &std::path::Path,
        cas_version: u64,
    ) -> std::path::PathBuf {
        root_directory.join(format!("v{cas_version:020}{LOCAL_FLOOR_FILE_SUFFIX_V1}"))
    }

    pub(super) fn lock_root(
        &self,
        root_id: [u8; 16],
    ) -> Result<std::fs::File, GenerationAuthorityErrorV1> {
        // Never unlink or replace this inode. Separate opens, rather than
        // cloned handles, also exclude callers sharing the same store instance.
        // This cooperative store assumes its owner preserves the directory.
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.root_directory(root_id).with_extension("lock"))
            .map_err(floor_io_unavailable)?;
        lock.lock().map_err(floor_io_unavailable)?;
        Ok(lock)
    }

    pub(super) fn sync_directories(
        &self,
        root_directory: &std::path::Path,
    ) -> Result<(), GenerationAuthorityErrorV1> {
        std::fs::File::open(root_directory)
            .and_then(|directory| directory.sync_all())
            .map_err(floor_io_unavailable)?;
        std::fs::File::open(&self.directory)
            .and_then(|directory| directory.sync_all())
            .map_err(floor_io_unavailable)
    }

    fn sync_record(
        &self,
        record: AntiRollbackFloorRecordV1,
    ) -> Result<(), GenerationAuthorityErrorV1> {
        let root_directory = self.root_directory(record.root_id);
        std::fs::File::open(Self::version_path(&root_directory, record.cas_version))
            .and_then(|file| file.sync_all())
            .map_err(floor_io_unavailable)?;
        self.sync_directories(&root_directory)
    }

    /// Every version file for `root_id`, keyed by the CAS version in its name,
    /// with the decode outcome of its bytes. A missing directory is no floor.
    #[allow(clippy::type_complexity)]
    fn scan(
        &self,
        root_id: [u8; 16],
    ) -> Result<
        BTreeMap<u64, Result<LocalFloorFileV1, GenerationAuthorityErrorV1>>,
        GenerationAuthorityErrorV1,
    > {
        let root_directory = self.root_directory(root_id);
        let entries = match std::fs::read_dir(&root_directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(BTreeMap::new());
            }
            Err(error) => return Err(floor_io_unavailable(error)),
        };
        let mut files = BTreeMap::new();
        for entry in entries {
            let entry = entry.map_err(floor_io_unavailable)?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(digits) = name
                .strip_prefix('v')
                .and_then(|rest| rest.strip_suffix(LOCAL_FLOOR_FILE_SUFFIX_V1))
            else {
                continue;
            };
            let Ok(cas_version) = digits.parse::<u64>() else {
                continue;
            };
            let bytes = std::fs::read(entry.path()).map_err(floor_io_unavailable)?;
            let decoded = LocalFloorFileV1::decode(&bytes).and_then(|file| {
                if file.record.root_id == root_id && file.record.cas_version == cas_version {
                    Ok(file)
                } else {
                    Err(GenerationAuthorityErrorV1::RootMismatch)
                }
            });
            files.insert(cas_version, decoded);
        }
        Ok(files)
    }

    /// Current floor for `root_id`, or `None` before any advance.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationAuthorityErrorV1::UnresolvedAttempt`] when the
    /// highest-numbered version file does not verify (torn, tampered, or
    /// foreign): the store then refuses every advance until it is reconciled
    /// out of band, because that file may be an acknowledged publication.
    pub fn load(
        &self,
        root_id: [u8; 16],
    ) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1> {
        if root_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.root_id",
            });
        }
        let _lock = self.lock_root(root_id)?;
        let files = self.scan(root_id)?;
        match files.into_iter().next_back() {
            None => Ok(None),
            Some((_, Ok(file))) => {
                // A previous writer may have completed its bytes but failed
                // the durability barrier. Reconcile before reporting a floor.
                self.sync_record(file.record)?;
                Ok(Some(file.record))
            }
            Some((_, Err(_))) => Err(GenerationAuthorityErrorV1::UnresolvedAttempt),
        }
    }

    /// Durable compare-and-advance with the same contract as
    /// [`super::InMemoryAntiRollbackFloorStoreV1::compare_and_advance`].
    ///
    /// # Errors
    ///
    /// Mirrors the in-memory provider's typed states, plus
    /// [`GenerationAuthorityErrorV1::UnresolvedAttempt`] for an unverifiable
    /// highest version file and
    /// [`GenerationAuthorityErrorV1::FloorStoreUnavailable`] for I/O failure.
    /// A lost `create_new` race is reported as
    /// [`GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict`]. A write
    /// that fails after the file was created is left in place as a torn file:
    /// it reads back as an unresolved attempt, never as a floor.
    pub fn compare_and_advance(
        &self,
        expected: Option<AntiRollbackFloorRecordV1>,
        next: AuthorityFloorV1,
        idempotency_key: [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1> {
        next.validate()?;
        if idempotency_key == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.idempotency_key",
            });
        }
        if let Some(expected) = expected {
            expected.validate()?;
            if expected.root_id != next.root_id {
                return Err(GenerationAuthorityErrorV1::RootMismatch);
            }
        }
        let expected_record_sha256 = expected.map(|record| record.record_sha256);

        let _lock = self.lock_root(next.root_id)?;
        let files = self.scan(next.root_id)?;
        for file in files.values().flatten() {
            if file.idempotency_key == idempotency_key {
                if file.expected_record_sha256 == expected_record_sha256
                    && file.record.authority == next.authority
                {
                    self.sync_record(file.record)?;
                    return Ok(file.record);
                }
                return Err(GenerationAuthorityErrorV1::FloorIdempotencyConflict);
            }
        }
        let current = match files.into_iter().next_back() {
            None => None,
            Some((_, Ok(file))) => Some(file.record),
            Some((_, Err(_))) => return Err(GenerationAuthorityErrorV1::UnresolvedAttempt),
        };
        if current != expected {
            return Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict);
        }
        match current {
            Some(record) => {
                let expected_sequence = record.authority.next_sequence()?;
                if next.authority.sequence <= record.authority.sequence {
                    return Err(GenerationAuthorityErrorV1::FloorSequenceRegression);
                }
                if next.authority.sequence != expected_sequence
                    || !predecessor_matches_authority(next.authority.predecessor, record.authority)
                {
                    return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
                }
            }
            None if next.authority.sequence != 1 || next.authority.predecessor.is_some() => {
                return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
            }
            None => {}
        }
        let cas_version = current
            .map(|record| record.cas_version)
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(GenerationAuthorityErrorV1::FloorVersionExhausted)?;
        let record = AntiRollbackFloorRecordV1::new(next, cas_version)?;
        let file = LocalFloorFileV1 {
            record,
            idempotency_key,
            expected_record_sha256,
        };

        let root_directory = self.root_directory(next.root_id);
        std::fs::create_dir_all(&root_directory).map_err(floor_io_unavailable)?;
        let path = Self::version_path(&root_directory, cas_version);
        // Keep immutable creation as a second fence. The root lock remains
        // held until both the record and its directory entries are durable.
        let mut handle = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(handle) => handle,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict);
            }
            Err(error) => return Err(floor_io_unavailable(error)),
        };
        std::io::Write::write_all(&mut handle, &file.encode())
            .and_then(|()| handle.sync_all())
            .map_err(floor_io_unavailable)?;
        drop(handle);
        // The first publication also creates root_directory itself; syncing
        // only that new directory does not persist its entry in the store.
        self.sync_directories(&root_directory)?;
        Ok(record)
    }
}
