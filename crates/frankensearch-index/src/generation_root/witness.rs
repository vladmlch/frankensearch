use std::fmt;
use std::path::Path;

use super::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GENERATION_ROOT_MAX_CONTROL_FILE_BYTES,
    GENERATION_ROOT_MAX_FILE_BYTES, GenerationRootError, GenerationRootErrorKind,
    GenerationRootResult, GenerationRootStage, platform,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use super::{GENERATION_ROOT_CONTROL_FILE_MODE, GENERATION_ROOT_IMMUTABLE_FILE_MODE};

/// Filesystem profile admitted for generation-root durability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QualifiedFilesystem {
    /// Linux ext4.
    LinuxExt4,
    /// Linux Btrfs.
    LinuxBtrfs,
    /// Local writable APFS on Apple Silicon.
    AppleApfs,
}

/// Stable metadata snapshot retained for one admitted filesystem object.
///
/// This is an object/metadata witness, not a content authority. Derived
/// equality and hashing compare the raw snapshot, including access time;
/// regular-file admission uses a narrower mutation-stable comparison that
/// deliberately excludes access time on every admitted platform. Root and
/// directory route revalidation is narrower again: it attests only
/// device/inode, mode, owner/group, mount identity, and filesystem profile,
/// because contained publication may legitimately change directory contents,
/// link count, size, and timestamps. Use the owning qualified
/// file/control/lock capability's SHA-256 accessor when content identity is
/// required; never interpret a root/directory revalidation as an attestation
/// of directory contents or timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationRootObjectWitness {
    pub(in crate::generation_root) device: u64,
    pub(in crate::generation_root) inode: u64,
    pub(in crate::generation_root) mode: u32,
    pub(in crate::generation_root) hard_links: u64,
    pub(in crate::generation_root) uid: u32,
    pub(in crate::generation_root) gid: u32,
    pub(in crate::generation_root) byte_len: u64,
    pub(in crate::generation_root) modified_seconds: i64,
    pub(in crate::generation_root) modified_nanoseconds: i64,
    pub(in crate::generation_root) changed_seconds: i64,
    pub(in crate::generation_root) changed_nanoseconds: i64,
    pub(in crate::generation_root) accessed_seconds: i64,
    pub(in crate::generation_root) accessed_nanoseconds: i64,
    pub(in crate::generation_root) mount_identity: [u8; 32],
    pub(in crate::generation_root) filesystem: QualifiedFilesystem,
}

impl GenerationRootObjectWitness {
    /// Device identifier.
    #[must_use]
    pub const fn device(&self) -> u64 {
        self.device
    }

    /// Inode identifier.
    #[must_use]
    pub const fn inode(&self) -> u64 {
        self.inode
    }

    /// Full persisted mode word.
    #[must_use]
    pub const fn mode(&self) -> u32 {
        self.mode
    }

    /// Hard-link count.
    #[must_use]
    pub const fn hard_links(&self) -> u64 {
        self.hard_links
    }

    /// Owner user id.
    #[must_use]
    pub const fn uid(&self) -> u32 {
        self.uid
    }

    /// Owner group id.
    #[must_use]
    pub const fn gid(&self) -> u32 {
        self.gid
    }

    /// Exact byte length at admission.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Opaque, descriptor-derived filesystem namespace identity.
    ///
    /// On Btrfs this binds both the VFS mount and the containing subvolume.
    #[must_use]
    pub const fn mount_identity(&self) -> [u8; 32] {
        self.mount_identity
    }

    /// Qualified filesystem profile.
    #[must_use]
    pub const fn filesystem(&self) -> QualifiedFilesystem {
        self.filesystem
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(in crate::generation_root) fn same_object(self, other: Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(in crate::generation_root) fn same_directory_security_identity(self, other: Self) -> bool {
        self.same_object(other)
            && self.mode == other.mode
            && self.uid == other.uid
            && self.gid == other.gid
            && self.mount_identity == other.mount_identity
            && self.filesystem == other.filesystem
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(in crate::generation_root) fn same_file_security_identity(self, other: Self) -> bool {
        self.same_directory_security_identity(other) && self.hard_links == other.hard_links
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(in crate::generation_root) fn same_control_anchor_identity(self, other: Self) -> bool {
        self.same_file_security_identity(other) && self.byte_len == other.byte_len
    }

    #[cfg(target_os = "linux")]
    pub(in crate::generation_root) fn mutation_stable_eq(self, other: Self) -> bool {
        self.same_file_security_identity(other)
            && self.byte_len == other.byte_len
            && self.modified_seconds == other.modified_seconds
            && self.modified_nanoseconds == other.modified_nanoseconds
            && self.changed_seconds == other.changed_seconds
            && self.changed_nanoseconds == other.changed_nanoseconds
    }

    #[cfg(target_os = "macos")]
    pub(in crate::generation_root) fn mutation_stable_eq(self, other: Self) -> bool {
        self.same_file_security_identity(other)
            && self.byte_len == other.byte_len
            && self.modified_seconds == other.modified_seconds
            && self.modified_nanoseconds == other.modified_nanoseconds
            && self.changed_seconds == other.changed_seconds
            && self.changed_nanoseconds == other.changed_nanoseconds
    }
}

/// Fixed physical layout of the named generation-root control anchors.
///
/// The authority length is frozen by the generation-root protocol. The LOCK
/// length is supplied by the separately versioned owner/attempt-frame codec;
/// this platform layer neither parses nor authenticates those frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationRootAnchorLayout {
    lock_byte_len: u64,
}

impl GenerationRootAnchorLayout {
    /// Construct a bounded, non-empty named-anchor layout.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ResourceLimit`] when the LOCK image
    /// is empty or exceeds [`GENERATION_ROOT_MAX_CONTROL_FILE_BYTES`].
    pub fn new(lock_byte_len: u64) -> GenerationRootResult<Self> {
        if lock_byte_len == 0 || lock_byte_len > GENERATION_ROOT_MAX_CONTROL_FILE_BYTES {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ResourceLimit,
                GenerationRootStage::ReadRegularFile,
            )
            .with_counts(GENERATION_ROOT_MAX_CONTROL_FILE_BYTES, lock_byte_len));
        }
        Ok(Self { lock_byte_len })
    }

    /// Exact LOCK byte length supplied by the owner/attempt-frame codec.
    #[must_use]
    pub const fn lock_byte_len(&self) -> u64 {
        self.lock_byte_len
    }

    /// Frozen physical AUTHORITY byte length.
    #[must_use]
    pub const fn authority_byte_len(&self) -> u64 {
        GENERATION_ROOT_AUTHORITY_FILE_BYTES
    }
}

/// One descriptor-derived root/LOCK/AUTHORITY identity tuple.
///
/// This attests the retained physical objects, not the semantic validity of
/// either mutable control image. Content authority is exposed only through a
/// live read or writer guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationRootAnchorWitness {
    root: GenerationRootObjectWitness,
    lock: GenerationRootObjectWitness,
    authority: GenerationRootObjectWitness,
}

impl GenerationRootAnchorWitness {
    pub(in crate::generation_root) const fn new(
        root: GenerationRootObjectWitness,
        lock: GenerationRootObjectWitness,
        authority: GenerationRootObjectWitness,
    ) -> Self {
        Self {
            root,
            lock,
            authority,
        }
    }

    /// Qualified generation-root directory witness.
    #[must_use]
    pub const fn root(&self) -> GenerationRootObjectWitness {
        self.root
    }

    /// Retained named LOCK witness.
    #[must_use]
    pub const fn lock(&self) -> GenerationRootObjectWitness {
        self.lock
    }

    /// Retained named AUTHORITY witness.
    #[must_use]
    pub const fn authority(&self) -> GenerationRootObjectWitness {
        self.authority
    }
}

/// Sealed role for an admitted generation file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationFileRole {
    /// Immutable artifact, sealed read-only for its owner.
    ImmutableArtifact,
    /// Fixed-size control anchor whose future in-place writes require a
    /// publisher API that advances the admitted witness while holding the
    /// kernel lock.
    ///
    /// This qualification substrate exposes no write operation and rejects
    /// ambient mutation fail-closed.
    MutableControl,
}

/// Exact expected generation-file identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationFileExpectation {
    pub(in crate::generation_root) byte_len: u64,
    sha256: Option<[u8; 32]>,
    pub(in crate::generation_root) role: GenerationFileRole,
}

impl GenerationFileExpectation {
    /// Expect an exact length and SHA-256, subject to the global ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ResourceLimit`] when `byte_len`
    /// exceeds [`GENERATION_ROOT_MAX_FILE_BYTES`].
    pub fn immutable(byte_len: u64, sha256: [u8; 32]) -> GenerationRootResult<Self> {
        Self::new(
            byte_len,
            Some(sha256),
            GenerationFileRole::ImmutableArtifact,
        )
    }

    /// Expect the current exact control-file length without precommitting its
    /// content.
    ///
    /// This substrate admits and locks the unchanged image only. A later
    /// publisher must use a witness-advancing mutation API rather than changing
    /// the file through an ambient descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ResourceLimit`] when `byte_len`
    /// exceeds [`GENERATION_ROOT_MAX_CONTROL_FILE_BYTES`].
    pub fn control(byte_len: u64) -> GenerationRootResult<Self> {
        Self::new(byte_len, None, GenerationFileRole::MutableControl)
    }

    /// Expect an exact control-file length and SHA-256 at lock acquisition.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ResourceLimit`] when `byte_len`
    /// exceeds [`GENERATION_ROOT_MAX_CONTROL_FILE_BYTES`].
    pub fn control_with_sha256(byte_len: u64, sha256: [u8; 32]) -> GenerationRootResult<Self> {
        Self::new(byte_len, Some(sha256), GenerationFileRole::MutableControl)
    }

    fn new(
        byte_len: u64,
        sha256: Option<[u8; 32]>,
        role: GenerationFileRole,
    ) -> GenerationRootResult<Self> {
        let max_byte_len = match role {
            GenerationFileRole::ImmutableArtifact => GENERATION_ROOT_MAX_FILE_BYTES,
            GenerationFileRole::MutableControl => GENERATION_ROOT_MAX_CONTROL_FILE_BYTES,
        };
        if byte_len > max_byte_len {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ResourceLimit,
                GenerationRootStage::ReadRegularFile,
            )
            .with_counts(max_byte_len, byte_len));
        }
        Ok(Self {
            byte_len,
            sha256,
            role,
        })
    }

    /// Exact expected byte length.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Optional expected SHA-256.
    #[must_use]
    pub const fn sha256(&self) -> Option<[u8; 32]> {
        self.sha256
    }

    /// Sealed file role.
    #[must_use]
    pub const fn role(&self) -> GenerationFileRole {
        self.role
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(in crate::generation_root) const fn max_byte_len(self) -> u64 {
        match self.role {
            GenerationFileRole::ImmutableArtifact => GENERATION_ROOT_MAX_FILE_BYTES,
            GenerationFileRole::MutableControl => GENERATION_ROOT_MAX_CONTROL_FILE_BYTES,
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(in crate::generation_root) const fn expected_mode(self) -> u32 {
        match self.role {
            GenerationFileRole::ImmutableArtifact => GENERATION_ROOT_IMMUTABLE_FILE_MODE,
            GenerationFileRole::MutableControl => GENERATION_ROOT_CONTROL_FILE_MODE,
        }
    }
}

/// A bounded, strictly relative path under a qualified generation root.
#[derive(Clone)]
pub struct ConfinedGenerationPath {
    pub(in crate::generation_root) inner: platform::RelativePath,
}

impl ConfinedGenerationPath {
    /// Parse a non-empty relative path without dot, dot-dot, root, prefix,
    /// empty, oversized, or NUL-containing components.
    ///
    /// Unsupported targets return
    /// [`GenerationRootErrorKind::UnsupportedPlatform`] before filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns a bounded route or platform error.
    pub fn parse(path: &Path) -> GenerationRootResult<Self> {
        platform::parse_relative(path).map(|inner| Self { inner })
    }

    /// Number of relative components.
    #[must_use]
    pub fn component_count(&self) -> usize {
        platform::relative_component_count(&self.inner)
    }
}

impl fmt::Debug for ConfinedGenerationPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConfinedGenerationPath")
            .field("component_count", &self.component_count())
            .finish_non_exhaustive()
    }
}

/// Object type of one retained-root directory entry, as reported by a
/// no-follow `statat` on the retained root descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationRootEntryKind {
    /// Regular file.
    RegularFile,
    /// Directory.
    Directory,
    /// Symbolic link (never followed).
    Symlink,
    /// Any other object (device, socket, FIFO…).
    Other,
}

/// One entry directly inside the retained root, identified without
/// following links and without adopting its pathname.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenerationRootEntry {
    pub(in crate::generation_root) name: std::ffi::OsString,
    pub(in crate::generation_root) kind: GenerationRootEntryKind,
    pub(in crate::generation_root) device: u64,
    pub(in crate::generation_root) inode: u64,
    pub(in crate::generation_root) hard_links: u64,
    pub(in crate::generation_root) byte_len: u64,
    pub(in crate::generation_root) mode: u32,
}

impl GenerationRootEntry {
    /// Entry name (one component, never `.` or `..`).
    #[must_use]
    pub fn name(&self) -> &std::ffi::OsStr {
        &self.name
    }

    /// Object type.
    #[must_use]
    pub const fn kind(&self) -> GenerationRootEntryKind {
        self.kind
    }

    /// Device the object lives on.
    #[must_use]
    pub const fn device(&self) -> u64 {
        self.device
    }

    /// Inode number.
    #[must_use]
    pub const fn inode(&self) -> u64 {
        self.inode
    }

    /// Hard-link count.
    #[must_use]
    pub const fn hard_links(&self) -> u64 {
        self.hard_links
    }

    /// Byte length (for a symlink, the target length).
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Full `st_mode` (type bits included).
    #[must_use]
    pub const fn mode(&self) -> u32 {
        self.mode
    }
}

/// Bounded, name-sorted inventory of a retained root directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRootInventory {
    pub(in crate::generation_root) entries: Vec<GenerationRootEntry>,
}

impl GenerationRootInventory {
    /// Entries sorted by name.
    #[must_use]
    pub fn entries(&self) -> &[GenerationRootEntry] {
        &self.entries
    }

    /// Look one entry up by exact name.
    #[must_use]
    pub fn entry(&self, name: &std::ffi::OsStr) -> Option<&GenerationRootEntry> {
        self.entries
            .binary_search_by(|entry| entry.name.as_os_str().cmp(name))
            .ok()
            .map(|index| &self.entries[index])
    }

    /// Entries whose names are not in `declared`: the undeclared residue a
    /// declared-role closure must refuse, named rather than generic.
    #[must_use]
    pub fn undeclared<'a>(&'a self, declared: &[&std::ffi::OsStr]) -> Vec<&'a GenerationRootEntry> {
        self.entries
            .iter()
            .filter(|entry| !declared.contains(&entry.name.as_os_str()))
            .collect()
    }

    /// Groups of two or more names that resolve to one `(device, inode)`:
    /// aliases inside the root. Empty when every name is its own object.
    #[must_use]
    pub fn aliases(&self) -> Vec<Vec<&GenerationRootEntry>> {
        let mut by_identity: Vec<((u64, u64), Vec<&GenerationRootEntry>)> = Vec::new();
        for entry in &self.entries {
            let identity = (entry.device, entry.inode);
            match by_identity.iter_mut().find(|(key, _)| *key == identity) {
                Some((_, group)) => group.push(entry),
                None => by_identity.push((identity, vec![entry])),
            }
        }
        by_identity
            .into_iter()
            .filter_map(|(_, group)| (group.len() > 1).then_some(group))
            .collect()
    }

    /// Entries whose device differs from `device` (the root's device):
    /// cross-device objects a closure must refuse.
    #[must_use]
    pub fn foreign_device(&self, device: u64) -> Vec<&GenerationRootEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.device != device)
            .collect()
    }
}
