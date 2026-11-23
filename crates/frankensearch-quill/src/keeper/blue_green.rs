use std::fs::OpenOptions;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

use crate::error::QuillError;

#[cfg(not(windows))]
use super::errors::KeeperError;
use super::manifest_validation::{put_u16, put_u32, usize_to_u64};
#[cfg(not(windows))]
use super::slot_io::sync_directory;
use super::slot_io::{open_manifest_slot, replace_file_atomically};
use super::stats_and_cursor::ByteCursor;

// ============================================================================
// Blue-green engine directories and the CURRENT pointer (registry §7.3, bead
// bd-quill-duel-blue-green-vwf7).
//
// Versioned sibling engine directories (`quill-v1/`, `tantivy/`) sit one level
// below a lexical root; a tiny `CURRENT` pointer file in that root names the
// active engine directory. Publication reuses the MANIFEST temp+rename+dir-
// fsync discipline one level up, so the G3 flip and every rebuild are single
// atomic pointer swaps with rollback being the same swap in reverse. The
// retired engine directory is preserved untouched until a human-approved
// retirement sweep (e9.3), keeping RULE-1 intact by construction.
// ============================================================================

/// Eight-byte CURRENT magic, including its trailing NUL.
pub const CURRENT_MAGIC: [u8; 8] = *b"FSLXCUR\0";
/// Current durable CURRENT-pointer format version.
pub const CURRENT_FORMAT_VERSION: u32 = 1;
/// Canonical CURRENT pointer file name in the lexical root.
pub const CURRENT_FILE_NAME: &str = "CURRENT";
/// Index format version recorded for pre-FSLX (tantivy) engine directories.
pub const TANTIVY_INDEX_FORMAT_VERSION: u32 = 0;
/// Encoded length excluding the variable directory-name bytes.
pub(in crate::keeper) const CURRENT_FIXED_BYTES: usize = 8 + 4 + 1 + 2 + 4 + 4;
/// Maximum accepted CURRENT pointer length (name bounded by its `u16` field).
const MAX_CURRENT_BYTES: usize = CURRENT_FIXED_BYTES + 65_535;
/// Bound on temp-name collisions tolerated from one publisher.
const CURRENT_TEMP_COLLISION_LIMIT: u32 = 128;

/// Engine family recorded in a CURRENT pointer (registry §7.3 kind codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueGreenEngine {
    /// Quill FSLX engine directory (kind code 1).
    Quill,
    /// Legacy tantivy engine directory (kind code 2).
    Tantivy,
}

impl BlueGreenEngine {
    /// Stable human label used in errors and telemetry.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Quill => "quill",
            Self::Tantivy => "tantivy",
        }
    }

    /// Registry §7.3 wire code.
    #[must_use]
    pub const fn kind_code(self) -> u8 {
        match self {
            Self::Quill => 1,
            Self::Tantivy => 2,
        }
    }

    fn from_kind_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Quill),
            2 => Some(Self::Tantivy),
            _ => None,
        }
    }

    /// Marker file whose presence identifies an engine directory of this kind
    /// during migration-adoption scans.
    const fn adoption_marker(self) -> &'static str {
        match self {
            Self::Quill => "MANIFEST",
            Self::Tantivy => "meta.json",
        }
    }

    /// Index format version recorded when adopting a directory of this kind.
    const fn adopted_format_version(self) -> u32 {
        match self {
            Self::Quill => crate::segment::FSLX_FORMAT_VERSION,
            Self::Tantivy => TANTIVY_INDEX_FORMAT_VERSION,
        }
    }
}

/// Decoded CURRENT pointer payload (registry §7.3).
///
/// Instances are always validated: construction and decoding reject empty,
/// over-long, non-UTF-8, or path-unsafe directory names, so a pointer can
/// never name anything but a plain direct child of its lexical root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentPointer {
    engine: BlueGreenEngine,
    dir_name: String,
    index_format_version: u32,
}

impl CurrentPointer {
    /// Construct a validated pointer to `dir_name` under a lexical root.
    ///
    /// # Errors
    ///
    /// Returns [`CurrentPointerError`] for an empty, over-long, or path-unsafe
    /// directory name.
    pub fn new(
        engine: BlueGreenEngine,
        dir_name: impl Into<String>,
        index_format_version: u32,
    ) -> Result<Self, CurrentPointerError> {
        let dir_name = dir_name.into();
        validate_current_dir_name(&dir_name)?;
        Ok(Self {
            engine,
            dir_name,
            index_format_version,
        })
    }

    /// Engine family of the active directory.
    #[must_use]
    pub const fn engine(&self) -> BlueGreenEngine {
        self.engine
    }

    /// Plain directory name (never a path) of the active engine directory.
    #[must_use]
    pub fn dir_name(&self) -> &str {
        &self.dir_name
    }

    /// Format version of the named engine directory's index format.
    #[must_use]
    pub const fn index_format_version(&self) -> u32 {
        self.index_format_version
    }

    /// Absolute engine directory for a lexical root.
    #[must_use]
    pub fn engine_dir(&self, lexical_root: &Path) -> PathBuf {
        lexical_root.join(&self.dir_name)
    }

    /// Encode with the registry §7.3 layout: magic, format version, engine
    /// kind, `u16` name length, name bytes, index format version, CRC32.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.dir_name.as_bytes();
        let mut bytes = Vec::with_capacity(CURRENT_FIXED_BYTES + name_bytes.len());
        bytes.extend_from_slice(&CURRENT_MAGIC);
        put_u32(&mut bytes, CURRENT_FORMAT_VERSION);
        bytes.push(self.engine.kind_code());
        #[allow(clippy::cast_possible_truncation)]
        put_u16(&mut bytes, name_bytes.len() as u16);
        bytes.extend_from_slice(name_bytes);
        put_u32(&mut bytes, self.index_format_version);
        let checksum = crc32fast::hash(&bytes);
        put_u32(&mut bytes, checksum);
        bytes
    }

    /// Decode and fully validate one CURRENT pointer image.
    ///
    /// The CRC covers every byte before the trailer; content fields are only
    /// interpreted after the checksum verifies, so corruption fails closed.
    ///
    /// # Errors
    ///
    /// Returns [`CurrentPointerError`] for truncation, bad magic, checksum
    /// mismatch, unknown versions or kind codes, trailing bytes, or an
    /// invalid directory name.
    pub fn decode(bytes: &[u8]) -> Result<Self, CurrentPointerError> {
        if bytes.len() < CURRENT_FIXED_BYTES {
            return Err(CurrentPointerError::Truncated {
                actual: bytes.len(),
                minimum: CURRENT_FIXED_BYTES,
            });
        }
        if bytes.len() > MAX_CURRENT_BYTES {
            return Err(CurrentPointerError::LengthMismatch {
                detail: format!(
                    "CURRENT pointer length {} exceeds {MAX_CURRENT_BYTES}",
                    bytes.len()
                ),
            });
        }
        if bytes[..CURRENT_MAGIC.len()] != CURRENT_MAGIC {
            return Err(CurrentPointerError::BadMagic);
        }
        let body_len = bytes.len() - 4;
        let expected_crc = crc32fast::hash(&bytes[..body_len]);
        let actual_crc = u32::from_le_bytes(bytes[body_len..].try_into().map_err(|_| {
            CurrentPointerError::Truncated {
                actual: bytes.len(),
                minimum: CURRENT_FIXED_BYTES,
            }
        })?);
        if actual_crc != expected_crc {
            return Err(CurrentPointerError::CrcMismatch {
                expected_crc,
                actual_crc,
            });
        }
        let mut cursor = ByteCursor::new(&bytes[CURRENT_MAGIC.len()..body_len]);
        let layout_error = |_| CurrentPointerError::LengthMismatch {
            detail: "CURRENT pointer body shorter than its declared layout".to_owned(),
        };
        let format_version = cursor.u32().map_err(layout_error)?;
        if format_version != CURRENT_FORMAT_VERSION {
            return Err(CurrentPointerError::UnsupportedFormatVersion(
                format_version,
            ));
        }
        let kind_code = cursor.u8().map_err(layout_error)?;
        let engine = BlueGreenEngine::from_kind_code(kind_code)
            .ok_or(CurrentPointerError::UnknownEngineKind(kind_code))?;
        let name_len = usize::from(cursor.u16().map_err(layout_error)?);
        if cursor.remaining() != name_len + 4 {
            return Err(CurrentPointerError::LengthMismatch {
                detail: format!(
                    "declared name length {name_len} leaves {} bytes, expected name plus index format version",
                    cursor.remaining()
                ),
            });
        }
        let name_bytes = cursor.take(name_len).map_err(layout_error)?;
        let dir_name = std::str::from_utf8(name_bytes)
            .map_err(|_| CurrentPointerError::DirNameNotUtf8)?
            .to_owned();
        validate_current_dir_name(&dir_name)?;
        let index_format_version = cursor.u32().map_err(layout_error)?;
        Ok(Self {
            engine,
            dir_name,
            index_format_version,
        })
    }
}

/// Validate that `dir_name` names exactly one plain directory entry.
fn validate_current_dir_name(dir_name: &str) -> Result<(), CurrentPointerError> {
    if dir_name.is_empty() {
        return Err(CurrentPointerError::EmptyDirName);
    }
    if dir_name.len() > 65_535 {
        return Err(CurrentPointerError::DirNameTooLong {
            actual: dir_name.len(),
        });
    }
    if dir_name.contains(['\\', '\0']) {
        return Err(CurrentPointerError::UnsafeDirName {
            dir_name: dir_name.to_owned(),
        });
    }
    let mut components = Path::new(dir_name).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(CurrentPointerError::UnsafeDirName {
            dir_name: dir_name.to_owned(),
        });
    }
    Ok(())
}

/// Typed CURRENT pointer failures (codec, publication, and resolution).
#[derive(Debug, Error)]
pub enum CurrentPointerError {
    /// Fewer bytes than the fixed pointer layout requires.
    #[error("CURRENT pointer is truncated: {actual} bytes, minimum {minimum}")]
    Truncated {
        /// Bytes actually present.
        actual: usize,
        /// Fixed-layout minimum.
        minimum: usize,
    },
    /// The eight-byte magic did not match `FSLXCUR\0`.
    #[error("CURRENT pointer magic mismatch")]
    BadMagic,
    /// Reader predates the pointer's format version.
    #[error(
        "CURRENT pointer format version {0} is unsupported (reader knows {CURRENT_FORMAT_VERSION})"
    )]
    UnsupportedFormatVersion(u32),
    /// Engine kind byte matches no registry entry.
    #[error("CURRENT pointer engine kind code {0} is unknown")]
    UnknownEngineKind(u8),
    /// Declared and actual field lengths disagree.
    #[error("CURRENT pointer length mismatch: {detail}")]
    LengthMismatch {
        /// Stable mismatch description.
        detail: String,
    },
    /// CRC32 over the body did not match the trailer.
    #[error(
        "CURRENT pointer CRC mismatch: computed {expected_crc:#010x}, stored {actual_crc:#010x}"
    )]
    CrcMismatch {
        /// CRC32 computed over the received body.
        expected_crc: u32,
        /// CRC32 stored in the trailer.
        actual_crc: u32,
    },
    /// Directory name is zero-length.
    #[error("CURRENT pointer directory name is empty")]
    EmptyDirName,
    /// Directory name exceeds its `u16` length field.
    #[error("CURRENT pointer directory name is {actual} bytes, exceeding u16")]
    DirNameTooLong {
        /// Actual name byte length.
        actual: usize,
    },
    /// Directory name bytes are not UTF-8.
    #[error("CURRENT pointer directory name is not valid UTF-8")]
    DirNameNotUtf8,
    /// Directory name is not one plain relative path component.
    #[error("CURRENT pointer directory name {dir_name:?} is not a plain direct child name")]
    UnsafeDirName {
        /// Rejected name.
        dir_name: String,
    },
    /// CURRENT names an engine directory that does not exist.
    #[error(
        "CURRENT pointer names {dir_name:?} under {root}, but that {engine} engine directory is missing; run fsfs doctor"
    )]
    MissingEngineDir {
        /// Lexical root containing the pointer.
        root: PathBuf,
        /// Named engine directory.
        dir_name: String,
        /// Engine family the pointer recorded.
        engine: &'static str,
    },
    /// CURRENT names an entry that is not a real directory beneath its root.
    #[error(
        "CURRENT pointer names {dir_name:?} under {root}, but that {engine} engine entry is not a direct directory"
    )]
    UnsafeEngineDir {
        /// Lexical root containing the pointer.
        root: PathBuf,
        /// Named engine entry.
        dir_name: String,
        /// Engine family the pointer recorded.
        engine: &'static str,
    },
    /// No CURRENT and several candidate engine directories: refusing to guess.
    #[error(
        "lexical root {root} has no CURRENT pointer and multiple engine directories {candidates:?}; run fsfs doctor to choose one"
    )]
    AmbiguousEngineDirs {
        /// Lexical root scanned.
        root: PathBuf,
        /// Sorted candidate directory names.
        candidates: Vec<String>,
    },
    /// Underlying filesystem failure.
    #[error("CURRENT pointer I/O during {operation} at {path}: {source}")]
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Path being operated on.
        path: PathBuf,
        /// Original error.
        source: io::Error,
    },
}

impl From<CurrentPointerError> for QuillError {
    fn from(error: CurrentPointerError) -> Self {
        match error {
            CurrentPointerError::Io { source, .. } => Self::Io(source),
            other => Self::Invariant {
                detail: other.to_string(),
            },
        }
    }
}

/// Publish `pointer` as the lexical root's CURRENT with the MANIFEST
/// temp+rename+dir-fsync discipline (registry §7.3, §6.2).
///
/// A unique `.tmp-current-<pid>-<n>` sibling is created `O_EXCL`, fully
/// written and fsynced, atomically renamed over `CURRENT`, and sealed with a
/// directory fsync. A crash anywhere before the rename leaves only an inert
/// temp that resolution ignores; the rename itself is the single atomic
/// transition, so flip and rollback are both one swap. Republishing identical
/// content is safe. Stale temps are intentionally left in place: they are
/// tiny, self-describing crash witnesses, and their retirement rides with the
/// flip orchestration layer's human-approved sweep.
///
/// # Errors
///
/// Returns [`CurrentPointerError::Io`] for temp creation, write, fsync,
/// rename, or directory-fsync failures. Windows uses a write-through atomic
/// replacement; unsupported targets retain the fail-closed directory gate.
pub fn publish_current(
    lexical_root: &Path,
    pointer: &CurrentPointer,
) -> Result<(), CurrentPointerError> {
    let encoded = pointer.encode();
    let current_path = lexical_root.join(CURRENT_FILE_NAME);
    let pid = std::process::id();
    let mut attempt = 0_u32;
    loop {
        let temp_path = lexical_root.join(format!(".tmp-current-{pid}-{attempt}"));
        let mut temp_file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => file,
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                attempt += 1;
                if attempt >= CURRENT_TEMP_COLLISION_LIMIT {
                    return Err(CurrentPointerError::Io {
                        operation: "allocate CURRENT temp",
                        path: temp_path,
                        source: io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "CURRENT temp collision limit exceeded",
                        ),
                    });
                }
                continue;
            }
            Err(source) => {
                return Err(CurrentPointerError::Io {
                    operation: "create CURRENT temp",
                    path: temp_path,
                    source,
                });
            }
        };
        temp_file
            .write_all(&encoded)
            .and_then(|()| temp_file.sync_all())
            .map_err(|source| CurrentPointerError::Io {
                operation: "persist CURRENT temp",
                path: temp_path.clone(),
                source,
            })?;
        drop(temp_file);
        replace_file_atomically(&temp_path, &current_path).map_err(|source| {
            CurrentPointerError::Io {
                operation: "rename CURRENT into place",
                path: current_path.clone(),
                source,
            }
        })?;
        #[cfg(not(windows))]
        sync_directory(lexical_root).map_err(|source| CurrentPointerError::Io {
            operation: "fsync CURRENT directory",
            path: lexical_root.to_path_buf(),
            source: match source {
                KeeperError::Io { source, .. } => source,
                other => io::Error::other(other.to_string()),
            },
        })?;
        return Ok(());
    }
}

/// Outcome of resolving a lexical root's active engine directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedCurrent {
    /// A valid CURRENT pointer was present.
    Pointer(CurrentPointer),
    /// No CURRENT existed; exactly one engine directory was found and adopted
    /// by writing a CURRENT pointer for it (migration bootstrap, registry
    /// §7.3).
    Adopted(CurrentPointer),
    /// No CURRENT and no engine directories: a fresh root.
    Empty,
}

impl ResolvedCurrent {
    /// The active pointer when one exists (present or adopted).
    #[must_use]
    pub fn pointer(&self) -> Option<&CurrentPointer> {
        match self {
            Self::Pointer(pointer) | Self::Adopted(pointer) => Some(pointer),
            Self::Empty => None,
        }
    }
}

/// Resolve the active engine directory under `lexical_root` (registry §7.3).
///
/// Order of decision: a present CURRENT is decoded, checksum-verified, and
/// its named directory must exist; an absent CURRENT triggers the adoption
/// scan — directories containing `MANIFEST` (quill) or `meta.json` (tantivy)
/// are engine candidates, exactly one candidate is adopted by publishing a
/// pointer for it, zero candidates yield [`ResolvedCurrent::Empty`], and
/// several candidates fail closed demanding doctor. Anything that is not a
/// real directory bearing an engine marker — user files, nested data,
/// `.tmp-*` crash witnesses — is ignored and never modified.
///
/// # Errors
///
/// Returns [`CurrentPointerError`] for corrupt or unreadable CURRENT bytes,
/// a pointer naming a missing directory, ambiguous engine directories, or
/// underlying I/O failure.
pub fn resolve_current(lexical_root: &Path) -> Result<ResolvedCurrent, CurrentPointerError> {
    let current_path = lexical_root.join(CURRENT_FILE_NAME);
    match read_current_file(&current_path)? {
        Some(bytes) => {
            let pointer = CurrentPointer::decode(&bytes)?;
            let engine_dir = pointer.engine_dir(lexical_root);
            match engine_directory_kind(&engine_dir)? {
                EngineDirectoryKind::Directory => {}
                EngineDirectoryKind::Missing => {
                    return Err(CurrentPointerError::MissingEngineDir {
                        root: lexical_root.to_path_buf(),
                        dir_name: pointer.dir_name().to_owned(),
                        engine: pointer.engine().label(),
                    });
                }
                EngineDirectoryKind::Unsafe => {
                    return Err(CurrentPointerError::UnsafeEngineDir {
                        root: lexical_root.to_path_buf(),
                        dir_name: pointer.dir_name().to_owned(),
                        engine: pointer.engine().label(),
                    });
                }
            }
            Ok(ResolvedCurrent::Pointer(pointer))
        }
        None => adopt_or_report_empty(lexical_root),
    }
}

/// Classification of a CURRENT-selected child without following symlinks.
///
/// A `CURRENT` name is syntactically one component, but that alone is not a
/// root boundary: `Path::is_dir` would follow a symlinked child outside the
/// lexical root. Readers must reject that shape before opening an engine.
enum EngineDirectoryKind {
    Missing,
    Directory,
    Unsafe,
}

fn engine_directory_kind(path: &Path) -> Result<EngineDirectoryKind, CurrentPointerError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => Ok(EngineDirectoryKind::Directory),
        Ok(_) => Ok(EngineDirectoryKind::Unsafe),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(EngineDirectoryKind::Missing),
        Err(source) => Err(CurrentPointerError::Io {
            operation: "stat CURRENT engine directory",
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Read a CURRENT file with a no-follow open and a bounded buffer.
fn read_current_file(current_path: &Path) -> Result<Option<Vec<u8>>, CurrentPointerError> {
    let mut file = match open_manifest_slot(current_path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(CurrentPointerError::Io {
                operation: "open CURRENT",
                path: current_path.to_path_buf(),
                source,
            });
        }
    };
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(usize_to_u64(MAX_CURRENT_BYTES).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| CurrentPointerError::Io {
            operation: "read CURRENT",
            path: current_path.to_path_buf(),
            source,
        })?;
    Ok(Some(bytes))
}

/// Scan `lexical_root` for adoptable engine directories (name-sorted).
///
/// Shared, strictly read-only policy used by both the mutating adoption rule
/// ([`adopt_or_report_empty`]) and the non-mutating inspector
/// ([`inspect_lexical_layout`]), so the two can never disagree about what
/// counts as an engine directory.
fn scan_adoption_candidates(
    lexical_root: &Path,
) -> Result<Vec<(String, BlueGreenEngine)>, CurrentPointerError> {
    let entries = std::fs::read_dir(lexical_root).map_err(|source| CurrentPointerError::Io {
        operation: "scan lexical root for engine directories",
        path: lexical_root.to_path_buf(),
        source,
    })?;
    let mut candidates: Vec<(String, BlueGreenEngine)> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| CurrentPointerError::Io {
            operation: "read lexical root entry",
            path: lexical_root.to_path_buf(),
            source,
        })?;
        let file_type = entry
            .file_type()
            .map_err(|source| CurrentPointerError::Io {
                operation: "stat lexical root entry",
                path: entry.path(),
                source,
            })?;
        if !file_type.is_dir() {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        // MANIFEST wins when both markers exist: quill is the forward format.
        let engine = if entry
            .path()
            .join(BlueGreenEngine::Quill.adoption_marker())
            .is_file()
        {
            Some(BlueGreenEngine::Quill)
        } else if entry
            .path()
            .join(BlueGreenEngine::Tantivy.adoption_marker())
            .is_file()
        {
            Some(BlueGreenEngine::Tantivy)
        } else {
            None
        };
        if let Some(engine) = engine {
            candidates.push((name, engine));
        }
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(candidates)
}

/// Scan `lexical_root` for engine directories and apply the adoption rule.
fn adopt_or_report_empty(lexical_root: &Path) -> Result<ResolvedCurrent, CurrentPointerError> {
    let mut candidates = scan_adoption_candidates(lexical_root)?;
    match candidates.len() {
        0 => Ok(ResolvedCurrent::Empty),
        1 => {
            let (dir_name, engine) = candidates.remove(0);
            let pointer = CurrentPointer::new(engine, dir_name, engine.adopted_format_version())?;
            publish_current(lexical_root, &pointer)?;
            Ok(ResolvedCurrent::Adopted(pointer))
        }
        _ => Err(CurrentPointerError::AmbiguousEngineDirs {
            root: lexical_root.to_path_buf(),
            candidates: candidates.into_iter().map(|(name, _)| name).collect(),
        }),
    }
}

/// Non-mutating classification of a lexical root's on-disk layout
/// (bd-8nqz.2).
///
/// Unlike [`resolve_current`], inspection NEVER adopts a child, writes
/// CURRENT, acquires a writer lease, or overwrites bytes — and it never
/// infers "empty" from MANIFEST absence alone: foreign and damaged layouts
/// are reported as themselves so no caller can mistake a tantivy index or a
/// half-destroyed blue-green root for free space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexicalLayout {
    /// No lexical content: the root is missing, or holds no engine markers,
    /// no CURRENT, and no adoptable children.
    Empty,
    /// A Quill index lives directly at the root (MANIFEST, no CURRENT).
    DirectQuill,
    /// A tantivy index lives directly at the root (meta.json, no CURRENT).
    DirectTantivy,
    /// Blue-green root with exactly one coherent active candidate.
    ///
    /// `pointer_on_disk == true` means a published CURRENT names a coherent
    /// engine directory. `false` means no CURRENT exists but exactly one
    /// adoptable child does; the pointer is the computed adoption candidate,
    /// which inspection reports but never publishes (the mutating
    /// [`resolve_current`] is the only adoption path).
    BlueGreen {
        /// Active (or computed adoption-candidate) pointer.
        pointer: CurrentPointer,
        /// Whether CURRENT actually exists on disk.
        pointer_on_disk: bool,
    },
    /// Quill markers present but the layout is damaged.
    CorruptQuill {
        /// Human-readable description of the damage.
        detail: String,
    },
    /// Tantivy markers present but the layout is damaged.
    CorruptTantivy {
        /// Human-readable description of the damage.
        detail: String,
    },
    /// Conflicting engine markers that no reader may silently pick from.
    Mixed {
        /// Human-readable description of the conflict.
        detail: String,
    },
    /// Layout cannot be attributed to one engine safely.
    Ambiguous {
        /// Human-readable description of why attribution failed.
        detail: String,
    },
}

impl LexicalLayout {
    /// Whether an index-builder-style writer may initialize a fresh Quill
    /// index at this root without destroying or shadowing existing content.
    #[must_use]
    pub const fn is_safe_for_quill_init(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Short stable label for errors and telemetry.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::DirectQuill => "direct-quill",
            Self::DirectTantivy => "direct-tantivy",
            Self::BlueGreen { .. } => "blue-green",
            Self::CorruptQuill { .. } => "corrupt-quill",
            Self::CorruptTantivy { .. } => "corrupt-tantivy",
            Self::Mixed { .. } => "mixed",
            Self::Ambiguous { .. } => "ambiguous",
        }
    }
}

/// Classify the layout at `lexical_root` without mutating anything.
///
/// # Errors
///
/// Returns [`CurrentPointerError`] only for I/O failures (permissions, a
/// symlinked CURRENT rejected by the no-follow open, unreadable directory
/// entries). Every attributable on-disk shape — including an undecodable
/// CURRENT or conflicting markers — is a successful classification, not an
/// error, so callers can act on the shape instead of guessing from error
/// strings.
pub fn inspect_lexical_layout(lexical_root: &Path) -> Result<LexicalLayout, CurrentPointerError> {
    if !lexical_root.is_dir() {
        return Ok(LexicalLayout::Empty);
    }
    let root_quill = lexical_root
        .join(BlueGreenEngine::Quill.adoption_marker())
        .is_file();
    let root_tantivy = lexical_root
        .join(BlueGreenEngine::Tantivy.adoption_marker())
        .is_file();
    let current_path = lexical_root.join(CURRENT_FILE_NAME);

    if let Some(bytes) = read_current_file(&current_path)? {
        if root_quill || root_tantivy {
            return Ok(LexicalLayout::Mixed {
                detail: format!(
                    "CURRENT coexists with direct engine markers at the root \
                     (MANIFEST: {root_quill}, meta.json: {root_tantivy})"
                ),
            });
        }
        let pointer = match CurrentPointer::decode(&bytes) {
            Ok(pointer) => pointer,
            Err(error) => {
                return Ok(LexicalLayout::Ambiguous {
                    detail: format!("CURRENT does not decode: {error}"),
                });
            }
        };
        let engine_dir = pointer.engine_dir(lexical_root);
        let marker = pointer.engine().adoption_marker();
        let corrupt = |detail: String| match pointer.engine() {
            BlueGreenEngine::Quill => LexicalLayout::CorruptQuill { detail },
            BlueGreenEngine::Tantivy => LexicalLayout::CorruptTantivy { detail },
        };
        match engine_directory_kind(&engine_dir)? {
            EngineDirectoryKind::Directory => {}
            EngineDirectoryKind::Missing => {
                return Ok(corrupt(format!(
                    "CURRENT names missing engine dir {:?}",
                    pointer.dir_name()
                )));
            }
            EngineDirectoryKind::Unsafe => {
                return Ok(corrupt(format!(
                    "CURRENT names non-directory or symlink engine entry {:?}",
                    pointer.dir_name()
                )));
            }
        }
        if !engine_dir.join(marker).is_file() {
            return Ok(corrupt(format!(
                "engine dir {:?} lacks its {marker} marker",
                pointer.dir_name()
            )));
        }
        return Ok(LexicalLayout::BlueGreen {
            pointer,
            pointer_on_disk: true,
        });
    }

    match (root_quill, root_tantivy) {
        (true, true) => {
            return Ok(LexicalLayout::Mixed {
                detail: "MANIFEST and meta.json coexist at the root".to_owned(),
            });
        }
        (true, false) => return Ok(LexicalLayout::DirectQuill),
        (false, true) => return Ok(LexicalLayout::DirectTantivy),
        (false, false) => {}
    }

    // No CURRENT and no direct markers: consult the shared adoption scan,
    // but never publish anything.
    let mut candidates = scan_adoption_candidates(lexical_root)?;
    match candidates.len() {
        0 => Ok(LexicalLayout::Empty),
        1 => {
            let (dir_name, engine) = candidates.remove(0);
            match CurrentPointer::new(engine, dir_name, engine.adopted_format_version()) {
                Ok(pointer) => Ok(LexicalLayout::BlueGreen {
                    pointer,
                    pointer_on_disk: false,
                }),
                Err(error) => Ok(LexicalLayout::Ambiguous {
                    detail: format!("sole engine dir cannot be pointed at: {error}"),
                }),
            }
        }
        _ => Ok(LexicalLayout::Ambiguous {
            detail: format!(
                "multiple adoptable engine dirs: {:?}",
                candidates
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
            ),
        }),
    }
}
