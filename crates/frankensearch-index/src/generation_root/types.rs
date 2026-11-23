use std::fmt;

/// Maximum accepted absolute or relative route length.
pub const GENERATION_ROOT_MAX_ROUTE_BYTES: usize = 4096;
/// Maximum accepted component length.
pub const GENERATION_ROOT_MAX_COMPONENT_BYTES: usize = 255;
/// Maximum accepted component count.
pub const GENERATION_ROOT_MAX_COMPONENTS: usize = 64;
/// Hard ceiling for one owned immutable artifact image.
pub const GENERATION_ROOT_MAX_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;
/// Hard ceiling for one repeatedly hashed mutable control anchor.
pub const GENERATION_ROOT_MAX_CONTROL_FILE_BYTES: u64 = 1024 * 1024;
/// Maximum names inspected while proving one exact directory entry.
pub const GENERATION_ROOT_MAX_DIRECTORY_ENTRIES: usize = 1_000_000;
/// Maximum aggregate name bytes inspected during one exact-name proof.
pub const GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES: usize = 64 * 1024 * 1024;
/// Maximum bytes read from Linux task-scoped procfs `mountinfo` for one qualification.
pub const GENERATION_ROOT_MAX_MOUNTINFO_BYTES: usize = 4 * 1024 * 1024;
/// Required mode for the qualified generation root and its directories.
pub const GENERATION_ROOT_DIRECTORY_MODE: u32 = 0o700;
/// Required sealed mode for immutable generation artifacts.
pub const GENERATION_ROOT_IMMUTABLE_FILE_MODE: u32 = 0o400;
/// Required mode for mutable generation control files.
pub const GENERATION_ROOT_CONTROL_FILE_MODE: u32 = 0o600;
/// Exact top-level name of the persistent generation-root lock anchor.
pub const GENERATION_ROOT_LOCK_FILE_NAME: &str = "LOCK";
/// Exact top-level name of the persistent generation-root authority anchor.
pub const GENERATION_ROOT_AUTHORITY_FILE_NAME: &str = "AUTHORITY";
/// Frozen physical length of the two-slot AUTHORITY inode.
///
/// Slot interpretation belongs to the authority codec. The platform layer
/// enforces only this immutable physical-layout fact.
pub const GENERATION_ROOT_AUTHORITY_FILE_BYTES: u64 = 8192;

/// Operation stage attached to every bounded generation-root failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GenerationRootStage {
    /// Reject an unsupported target before any filesystem access.
    PlatformGate,
    /// Parse and bound an absolute generation-root route.
    ParseRootRoute,
    /// Parse and bound a confined relative route.
    ParseRelativeRoute,
    /// Open the absolute route one component at a time.
    OpenRootRoute,
    /// Qualify root ownership and permissions.
    QualifyRootSecurity,
    /// Qualify the root filesystem and mount.
    QualifyFilesystem,
    /// Inspect an extended ACL through a retained descriptor.
    InspectAcl,
    /// Re-resolve and fence the absolute route.
    RevalidateRootRoute,
    /// Open and qualify a relative directory component.
    OpenRelativeDirectory,
    /// Open and qualify the final regular file.
    OpenRegularFile,
    /// Read one regular file into a bounded owned image.
    ReadRegularFile,
    /// Re-open and fence a relative file route.
    RevalidateRegularFile,
    /// Execute the platform's file durability barrier.
    SyncRegularFile,
    /// Execute the platform's directory durability barrier.
    SyncDirectory,
    /// Acquire a non-blocking kernel file lock.
    AcquireLock,
    /// Explicitly release a kernel file lock.
    ReleaseLock,
    /// Positional write of an owner/attempt frame into the LOCK anchor by the
    /// exclusive lock holder; `expected` carries the failing phase (1 = pre-write
    /// revalidation, 2 = write, 3 = barrier, 4 = witness re-stat, 5 = identity,
    /// 6 = exact reread, 7 = post-write revalidation) and `observed` the offset.
    WriteLockFrame,
    /// Positional write of one authenticated slot into the AUTHORITY anchor;
    /// phases as for [`Self::WriteLockFrame`].
    WriteAuthoritySlot,
    /// Bounded no-follow enumeration of the retained root directory.
    InventoryDirectory,
}

impl GenerationRootStage {
    const fn code(self) -> &'static str {
        match self {
            Self::WriteLockFrame => "write_lock_frame",
            Self::WriteAuthoritySlot => "write_authority_slot",
            Self::InventoryDirectory => "inventory_directory",
            Self::PlatformGate => "platform_gate",
            Self::ParseRootRoute => "parse_root_route",
            Self::ParseRelativeRoute => "parse_relative_route",
            Self::OpenRootRoute => "open_root_route",
            Self::QualifyRootSecurity => "qualify_root_security",
            Self::QualifyFilesystem => "qualify_filesystem",
            Self::InspectAcl => "inspect_acl",
            Self::RevalidateRootRoute => "revalidate_root_route",
            Self::OpenRelativeDirectory => "open_relative_directory",
            Self::OpenRegularFile => "open_regular_file",
            Self::ReadRegularFile => "read_regular_file",
            Self::RevalidateRegularFile => "revalidate_regular_file",
            Self::SyncRegularFile => "sync_regular_file",
            Self::SyncDirectory => "sync_directory",
            Self::AcquireLock => "acquire_lock",
            Self::ReleaseLock => "release_lock",
        }
    }
}

/// Stable failure class for generation-root admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GenerationRootErrorKind {
    /// This architecture/OS combination is intentionally unsupported.
    UnsupportedPlatform,
    /// A path is empty, ambiguous, unbounded, absolute where relative was
    /// required, or contains a forbidden component.
    InvalidRoute,
    /// A required kernel primitive is unavailable or filtered.
    UnsupportedKernelFeature,
    /// Apple Silicon final-file admission requires a caller/provider-owned
    /// already-open descriptor so this library never invokes a vnode open
    /// callback before proving the object type.
    PreopenedDescriptorRequired,
    /// A symbolic or magic link was encountered.
    SymbolicLink,
    /// An ancestor or requested root is not a directory.
    NotDirectory,
    /// The final object is not a regular file.
    NotRegularFile,
    /// A regular file has more than one hard link.
    HardLinked,
    /// An object is not owned by the effective user.
    WrongOwner,
    /// An object's permission mode is not the exact private mode.
    WrongMode,
    /// A provider-owned descriptor is not opened with the exact data access
    /// mode required by the API, or carries a metadata/event-only access flag.
    InvalidDescriptorAccess,
    /// A provider-owned descriptor is missing close-on-exec.
    CloseOnExecRequired,
    /// Per-object immutable, append-only, or other write-restricting flags
    /// make the object incompatible with the qualified writable-root profile.
    WriteRestrictedObject,
    /// Apple object flags are nonzero and therefore outside the deliberately
    /// narrow zero-flags APFS admission profile.
    UnsupportedObjectFlags,
    /// A platform extended ACL is present or cannot be inspected exactly
    /// through the retained descriptor.
    AclRejected,
    /// A descendant crossed the qualified root's mount.
    CrossDevice,
    /// The filesystem is remote, synthetic, unknown, or not allowlisted.
    UnsupportedFilesystem,
    /// The filesystem is read-only or ignores ownership.
    ReadOnlyFilesystem,
    /// A retained object or its canonical route changed during admission.
    ObjectChanged,
    /// A file length differs from its exact expected value.
    SizeMismatch,
    /// An immutable-artifact API received a control-file expectation or the
    /// reverse.
    WrongRole,
    /// A caller or on-disk value exceeds a resource ceiling.
    ResourceLimit,
    /// The exact owned bytes do not match the expected SHA-256.
    HashMismatch,
    /// The required durability primitive failed; no weaker barrier was used.
    DurabilityUnavailable,
    /// A non-blocking shared or exclusive kernel lock is contended.
    LockContended,
    /// Admission or a lock operation crossed a `fork()` process-identity
    /// boundary and cannot continue in the child process.
    ForkedProcess,
    /// A bounded operating-system error not covered by a more specific class.
    Io,
}

impl GenerationRootErrorKind {
    /// Stable, machine-readable reason code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnsupportedPlatform => "unsupported_platform",
            Self::InvalidRoute => "invalid_route",
            Self::UnsupportedKernelFeature => "unsupported_kernel_feature",
            Self::PreopenedDescriptorRequired => "preopened_descriptor_required",
            Self::SymbolicLink => "symbolic_link",
            Self::NotDirectory => "not_directory",
            Self::NotRegularFile => "not_regular_file",
            Self::HardLinked => "hard_linked",
            Self::WrongOwner => "wrong_owner",
            Self::WrongMode => "wrong_mode",
            Self::InvalidDescriptorAccess => "invalid_descriptor_access",
            Self::CloseOnExecRequired => "close_on_exec_required",
            Self::WriteRestrictedObject => "write_restricted_object",
            Self::UnsupportedObjectFlags => "unsupported_object_flags",
            Self::AclRejected => "acl_rejected",
            Self::CrossDevice => "cross_device",
            Self::UnsupportedFilesystem => "unsupported_filesystem",
            Self::ReadOnlyFilesystem => "read_only_filesystem",
            Self::ObjectChanged => "object_changed",
            Self::SizeMismatch => "size_mismatch",
            Self::WrongRole => "wrong_role",
            Self::ResourceLimit => "resource_limit",
            Self::HashMismatch => "hash_mismatch",
            Self::DurabilityUnavailable => "durability_unavailable",
            Self::LockContended => "lock_contended",
            Self::ForkedProcess => "forked_process",
            Self::Io => "io",
        }
    }
}

/// Bounded, path-redacted generation-root failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationRootError {
    kind: GenerationRootErrorKind,
    stage: GenerationRootStage,
    component_index: Option<u16>,
    raw_os_error: Option<i32>,
    expected: Option<u64>,
    observed: Option<u64>,
}

impl GenerationRootError {
    pub(in crate::generation_root) const fn new(
        kind: GenerationRootErrorKind,
        stage: GenerationRootStage,
    ) -> Self {
        Self {
            kind,
            stage,
            component_index: None,
            raw_os_error: None,
            expected: None,
            observed: None,
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(in crate::generation_root) fn at_component(mut self, component_index: usize) -> Self {
        self.component_index = Some(u16::try_from(component_index).unwrap_or(u16::MAX));
        self
    }

    // Not gated to the native platforms: the platform-independent publisher
    // records raw OS errors too, and `io::Error::raw_os_error` exists on every
    // target.
    pub(in crate::generation_root) const fn with_raw_os_error(mut self, raw_os_error: i32) -> Self {
        self.raw_os_error = Some(raw_os_error);
        self
    }

    pub(in crate::generation_root) const fn with_counts(
        mut self,
        expected: u64,
        observed: u64,
    ) -> Self {
        self.expected = Some(expected);
        self.observed = Some(observed);
        self
    }

    /// Stable failure class.
    #[must_use]
    pub const fn kind(&self) -> GenerationRootErrorKind {
        self.kind
    }

    /// Operation stage at which the failure was observed.
    #[must_use]
    pub const fn stage(&self) -> GenerationRootStage {
        self.stage
    }

    /// Zero-based path component index, when applicable.
    #[must_use]
    pub const fn component_index(&self) -> Option<u16> {
        self.component_index
    }

    /// Raw OS error number, when retaining it is useful and bounded.
    #[must_use]
    pub const fn raw_os_error(&self) -> Option<i32> {
        self.raw_os_error
    }

    /// Expected bounded numeric value, ceiling, count, or bitmask, when
    /// applicable.
    #[must_use]
    pub const fn expected(&self) -> Option<u64> {
        self.expected
    }

    /// Observed bounded numeric value, count, or bitmask, when applicable.
    #[must_use]
    pub const fn observed(&self) -> Option<u64> {
        self.observed
    }
}

impl fmt::Display for GenerationRootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "generation-root {} failed: {}",
            self.stage.code(),
            self.kind.code()
        )?;
        if let Some(component_index) = self.component_index {
            write!(formatter, " (component {component_index})")?;
        }
        if let Some(raw_os_error) = self.raw_os_error {
            write!(formatter, " (os error {raw_os_error})")?;
        }
        if let (Some(expected), Some(observed)) = (self.expected, self.observed) {
            write!(formatter, " (expected {expected}, observed {observed})")?;
        }
        if self.kind == GenerationRootErrorKind::AclRejected {
            write!(
                formatter,
                " (only the descriptor-bound absence of a platform extended ACL \
                 is admissible; presence or inspection uncertainty fails closed)"
            )?;
        }
        if self.kind == GenerationRootErrorKind::PreopenedDescriptorRequired {
            write!(
                formatter,
                " (Apple Silicon requires a trusted external provider to supply an owned \
                 already-open regular-file descriptor; use the preopened admission/lock API)"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for GenerationRootError {}

/// Result alias for this platform substrate.
pub type GenerationRootResult<T> = Result<T, GenerationRootError>;
