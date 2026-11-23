use std::fmt;
use std::path::Path;
use std::sync::Arc;

use super::handles::{GenerationRootLockMode, QualifiedGenerationControlFile};
use super::{
    ConfinedGenerationPath, GENERATION_ROOT_AUTHORITY_FILE_NAME, GENERATION_ROOT_LOCK_FILE_NAME,
    GenerationFileExpectation, GenerationFileRole, GenerationRootAnchorLayout,
    GenerationRootAnchorWitness, GenerationRootError, GenerationRootErrorKind,
    GenerationRootInventory, GenerationRootObjectWitness, GenerationRootResult,
    GenerationRootStage, QualifiedGenerationFile, platform,
};

/// Internal qualified directory capability used while atomically binding the
/// named LOCK and AUTHORITY anchors.
pub(in crate::generation_root) struct QualifiedGenerationDirectory {
    inner: platform::RootHandle,
}

impl QualifiedGenerationDirectory {
    /// Admit an existing absolute private generation-root route.
    ///
    /// The route is walked component-by-component without following links,
    /// then independently re-resolved before this function returns.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any route, security, mount,
    /// filesystem, ACL, kernel-feature, or identity uncertainty. Unsupported
    /// targets fail with [`GenerationRootErrorKind::UnsupportedPlatform`]
    /// before route parsing or filesystem I/O.
    pub(in crate::generation_root) fn admit(path: &Path) -> GenerationRootResult<Self> {
        platform::admit_root(path).map(|inner| Self { inner })
    }

    /// Root descriptor witness.
    #[must_use]
    pub(in crate::generation_root) fn witness(&self) -> GenerationRootObjectWitness {
        platform::root_witness(&self.inner)
    }

    /// Re-resolve the canonical route and require it to name the retained
    /// private root.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ObjectChanged`] or the precise
    /// qualification failure when the route no longer resolves identically.
    #[cfg(all(test, target_os = "linux"))]
    pub(in crate::generation_root) fn revalidate_route(&self) -> GenerationRootResult<()> {
        platform::revalidate_root(&self.inner)
    }

    /// Admit one exact regular-file image below this root.
    ///
    /// The returned owner retains the qualified data descriptor and one immutable
    /// `Arc<[u8]>`; header validation and search consumers can therefore use
    /// the same allocation after route replacement. On Linux the data
    /// descriptor is derived from a retained, fully qualified `O_PATH`
    /// capability through the verified `/proc/thread-self/fd` bridge
    /// documented at module level; it is not opened from the generation route
    /// before the final object is proven regular.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any route, mount, type, owner,
    /// mode, link-count, length, content, or before/after identity mismatch.
    /// On Apple Silicon this path-only API always returns
    /// [`GenerationRootErrorKind::PreopenedDescriptorRequired`]; use
    /// [`Self::admit_preopened_file`].
    #[cfg_attr(target_os = "macos", allow(clippy::unused_self))]
    pub(in crate::generation_root) fn admit_file(
        &self,
        path: &ConfinedGenerationPath,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationFile> {
        if expectation.role != GenerationFileRole::ImmutableArtifact {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::WrongRole,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let _ = path;
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::OpenRegularFile,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let admission = platform::admit_file(&self.inner, &path.inner, expectation)?;
            Ok(QualifiedGenerationFile {
                handle: admission.handle,
                bytes: admission.bytes,
                witness: admission.witness,
                sha256: admission.sha256,
            })
        }
    }

    /// Admit an exact Apple Silicon generation artifact from a descriptor
    /// already opened by a trusted external provider.
    ///
    /// The provider owns the proof that descriptor acquisition itself had no
    /// vnode/device side effects and attests that the descriptor is fresh,
    /// solely transferred, and has no retained duplicate or separately opened
    /// writable alias; POSIX descriptor inspection cannot verify those facts.
    /// This library first rejects non-regular descriptors and proves exact
    /// access status plus `FD_CLOEXEC`; it then qualifies the retained
    /// descriptor's mount, owner, permissions, link count, flags, ACL, and
    /// size before binding the canonical route and reading the exact content.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any descriptor, route, security,
    /// identity, content, or resource uncertainty.
    #[cfg(target_os = "macos")]
    pub(in crate::generation_root) fn admit_preopened_file(
        &self,
        path: &ConfinedGenerationPath,
        descriptor: std::os::fd::OwnedFd,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationFile> {
        if expectation.role != GenerationFileRole::ImmutableArtifact {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::WrongRole,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        let admission =
            platform::admit_preopened_file(&self.inner, &path.inner, descriptor, expectation)?;
        Ok(QualifiedGenerationFile {
            handle: admission.handle,
            bytes: admission.bytes,
            witness: admission.witness,
            sha256: admission.sha256,
        })
    }

    /// Apply the platform's strongest required directory durability barrier.
    ///
    /// Linux uses `fsync`; Apple Silicon uses `F_FULLFSYNC` and never
    /// downgrades to `fsync`.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::DurabilityUnavailable`] when the
    /// required primitive fails.
    pub(in crate::generation_root) fn sync_directory_durable(&self) -> GenerationRootResult<()> {
        platform::sync_root_directory(&self.inner)
    }

    pub(in crate::generation_root) fn inventory(
        &self,
    ) -> GenerationRootResult<GenerationRootInventory> {
        platform::inventory(&self.inner).map(|entries| GenerationRootInventory { entries })
    }

    /// Admit an existing fixed-size private control anchor as a persistent
    /// descriptor-owned capability.
    ///
    /// This foundation API deliberately exposes no mutation operation.
    /// Ambient content or metadata changes invalidate the capability. The
    /// publisher layer must add a lock-owned, witness-advancing write protocol
    /// for AUTHORITY/LOCK frames.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any route, mount, type, owner,
    /// mode, link-count, length, or identity mismatch. On Apple Silicon this
    /// path-only API always returns
    /// [`GenerationRootErrorKind::PreopenedDescriptorRequired`]; use
    /// [`Self::admit_preopened_control_file`].
    #[cfg_attr(target_os = "macos", allow(clippy::unused_self))]
    pub(in crate::generation_root) fn admit_control_file(
        &self,
        path: &ConfinedGenerationPath,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationControlFile> {
        if expectation.role != GenerationFileRole::MutableControl {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::WrongRole,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let _ = path;
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::OpenRegularFile,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            platform::admit_control_file(&self.inner, &path.inner, expectation)
                .map(|inner| QualifiedGenerationControlFile { inner })
        }
    }

    /// Admit an Apple Silicon control anchor from a provider-owned descriptor.
    ///
    /// The provider owns no-side-effect acquisition provenance. The supplied
    /// descriptor must be a fresh read-write, close-on-exec open description;
    /// future lock acquisition requires another independently opened
    /// descriptor through [`QualifiedGenerationControlFile::try_lock_preopened`].
    /// Freshness and sole ownership are trusted provider attestations: POSIX
    /// descriptor inspection cannot detect an unseen `dup`.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any descriptor, route, security,
    /// identity, content, or resource uncertainty.
    #[cfg(target_os = "macos")]
    pub(in crate::generation_root) fn admit_preopened_control_file(
        &self,
        path: &ConfinedGenerationPath,
        descriptor: std::os::fd::OwnedFd,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationControlFile> {
        if expectation.role != GenerationFileRole::MutableControl {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::WrongRole,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        platform::admit_preopened_control_file(&self.inner, &path.inner, descriptor, expectation)
            .map(|inner| QualifiedGenerationControlFile { inner })
    }
}

/// Provider-owned Apple Silicon descriptors for the two fixed control
/// anchors.
///
/// The trusted provider must transfer fresh, solely owned, read-write,
/// close-on-exec descriptors for the exact top-level LOCK and AUTHORITY files.
#[cfg(target_os = "macos")]
pub struct PreopenedGenerationRootAnchors {
    lock: std::os::fd::OwnedFd,
    authority: std::os::fd::OwnedFd,
}

#[cfg(target_os = "macos")]
impl PreopenedGenerationRootAnchors {
    /// Bind provider-labelled LOCK and AUTHORITY descriptors without exposing
    /// an order-dependent tuple at the root-admission call site.
    #[must_use]
    pub fn new(lock: std::os::fd::OwnedFd, authority: std::os::fd::OwnedFd) -> Self {
        Self { lock, authority }
    }
}

/// Qualified generation root with retained exact-name LOCK and AUTHORITY
/// descriptors.
///
/// Construction returns only after the root and both anchors have passed a
/// combined anti-mix fence. Mutable control content is deliberately not frozen
/// into this long-lived capability: each shared or exclusive guard captures a
/// fresh exact owned image while holding the named LOCK flock.
pub struct QualifiedGenerationRoot {
    pub(in crate::generation_root) directory: QualifiedGenerationDirectory,
    pub(in crate::generation_root) lock: QualifiedGenerationControlFile,
    pub(in crate::generation_root) authority: QualifiedGenerationControlFile,
    pub(in crate::generation_root) layout: GenerationRootAnchorLayout,
}

impl QualifiedGenerationRoot {
    /// Admit an existing private root and its exact top-level LOCK and
    /// AUTHORITY anchors.
    ///
    /// Linux opens both anchors descriptor-relatively. Apple Silicon returns
    /// [`GenerationRootErrorKind::PreopenedDescriptorRequired`]; use
    /// [`Self::admit_preopened`]. Unsupported targets fail before path parsing
    /// or filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed platform, route, security, filesystem,
    /// anchor-identity, length, or content-admission error.
    pub fn admit(path: &Path, layout: GenerationRootAnchorLayout) -> GenerationRootResult<Self> {
        let directory = QualifiedGenerationDirectory::admit(path)?;
        let lock_path = ConfinedGenerationPath::parse(Path::new(GENERATION_ROOT_LOCK_FILE_NAME))?;
        let authority_path =
            ConfinedGenerationPath::parse(Path::new(GENERATION_ROOT_AUTHORITY_FILE_NAME))?;
        let lock = directory.admit_control_file(
            &lock_path,
            GenerationFileExpectation::control(layout.lock_byte_len())?,
        )?;
        let authority = directory.admit_control_file(
            &authority_path,
            GenerationFileExpectation::control(layout.authority_byte_len())?,
        )?;
        let admitted = Self {
            directory,
            lock,
            authority,
            layout,
        };
        admitted.revalidate()?;
        Ok(admitted)
    }

    /// Admit Apple Silicon named anchors from trusted provider-owned
    /// descriptors.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed descriptor, route, security, filesystem,
    /// anchor-identity, length, or content-admission error.
    #[cfg(target_os = "macos")]
    pub fn admit_preopened(
        path: &Path,
        layout: GenerationRootAnchorLayout,
        anchors: PreopenedGenerationRootAnchors,
    ) -> GenerationRootResult<Self> {
        let directory = QualifiedGenerationDirectory::admit(path)?;
        let lock_path = ConfinedGenerationPath::parse(Path::new(GENERATION_ROOT_LOCK_FILE_NAME))?;
        let authority_path =
            ConfinedGenerationPath::parse(Path::new(GENERATION_ROOT_AUTHORITY_FILE_NAME))?;
        let lock = directory.admit_preopened_control_file(
            &lock_path,
            anchors.lock,
            GenerationFileExpectation::control(layout.lock_byte_len())?,
        )?;
        let authority = directory.admit_preopened_control_file(
            &authority_path,
            anchors.authority,
            GenerationFileExpectation::control(layout.authority_byte_len())?,
        )?;
        let admitted = Self {
            directory,
            lock,
            authority,
            layout,
        };
        admitted.revalidate()?;
        Ok(admitted)
    }

    /// Root descriptor witness.
    #[must_use]
    pub fn witness(&self) -> GenerationRootObjectWitness {
        self.directory.witness()
    }

    /// Descriptor-derived identity tuple retained by this root capability.
    #[must_use]
    pub fn anchor_witnesses(&self) -> GenerationRootAnchorWitness {
        GenerationRootAnchorWitness::new(
            self.directory.witness(),
            self.lock.witness(),
            self.authority.witness(),
        )
    }

    /// Frozen physical named-anchor layout.
    #[must_use]
    pub const fn anchor_layout(&self) -> GenerationRootAnchorLayout {
        self.layout
    }

    /// Revalidate the canonical root plus both retained named anchors as one
    /// anti-mix tuple.
    ///
    /// Mutable bytes, mtime, and ctime may advance between guards; object,
    /// route, mount, security, link, and fixed-length identity may not.
    ///
    /// # Errors
    ///
    /// Returns the precise fail-closed route or identity error.
    pub fn revalidate(&self) -> GenerationRootResult<()> {
        platform::revalidate_anchor_set(
            &self.directory.inner,
            &self.lock.inner,
            &self.authority.inner,
        )
    }

    /// Admit one exact immutable regular-file image below this root.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as descriptor-owned file admission.
    pub fn admit_file(
        &self,
        path: &ConfinedGenerationPath,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationFile> {
        self.directory.admit_file(path, expectation)
    }

    /// Admit one exact Apple Silicon immutable artifact from a provider-owned
    /// descriptor.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as preopened file admission.
    #[cfg(target_os = "macos")]
    pub fn admit_preopened_file(
        &self,
        path: &ConfinedGenerationPath,
        descriptor: std::os::fd::OwnedFd,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationFile> {
        self.directory
            .admit_preopened_file(path, descriptor, expectation)
    }

    /// Apply the platform's strongest required directory durability barrier.
    ///
    /// # Errors
    ///
    /// Returns a typed route, identity, or durability error.
    pub fn sync_directory_durable(&self) -> GenerationRootResult<()> {
        self.directory.sync_directory_durable()
    }

    /// Enumerate every entry directly inside the retained root directory.
    ///
    /// The scan runs on the retained root descriptor (never a pathname
    /// re-walk), identifies each entry with a no-follow `statat`, and is
    /// bounded by [`GENERATION_ROOT_MAX_DIRECTORY_ENTRIES`](super::GENERATION_ROOT_MAX_DIRECTORY_ENTRIES) and
    /// [`GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES`](super::GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES). The root is revalidated
    /// before and after the scan. Entries are returned sorted by name so two
    /// inventories of one unchanged directory are equal.
    ///
    /// This is the closure primitive: a consumer that knows the declared
    /// component set of a generation can now prove that nothing undeclared,
    /// no alias, and no foreign-device object sits beside it (bd-v3hgo),
    /// without adopting any pathname.
    ///
    /// # Errors
    ///
    /// Returns a typed route, identity, resource-limit, or platform error.
    pub fn inventory(&self) -> GenerationRootResult<GenerationRootInventory> {
        self.directory.inventory()
    }

    /// Acquire a non-blocking shared guard on the exact named LOCK anchor and
    /// capture fresh stable LOCK and AUTHORITY images.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::LockContended`] instead of blocking,
    /// or the precise tuple/content failure. Apple Silicon requires
    /// [`Self::read_guard_preopened`].
    pub fn read_guard(&self) -> GenerationRootResult<GenerationRootReadGuard<'_>> {
        #[cfg(target_os = "macos")]
        {
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::AcquireLock,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let flock =
                platform::try_anchor_lock(&self.lock.inner, GenerationRootLockMode::Shared)?;
            self.finish_read_guard(flock)
        }
    }

    /// Acquire an Apple Silicon shared guard using a fresh provider-owned LOCK
    /// open description.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed descriptor, contention, tuple, or content
    /// error.
    #[cfg(target_os = "macos")]
    pub fn read_guard_preopened(
        &self,
        fresh_lock_descriptor: std::os::fd::OwnedFd,
    ) -> GenerationRootResult<GenerationRootReadGuard<'_>> {
        let flock = platform::try_anchor_lock_preopened(
            &self.lock.inner,
            fresh_lock_descriptor,
            GenerationRootLockMode::Shared,
        )?;
        self.finish_read_guard(flock)
    }

    fn finish_read_guard(
        &self,
        flock: platform::LockHandle,
    ) -> GenerationRootResult<GenerationRootReadGuard<'_>> {
        let (lock_image, authority_image) = self.capture_stable_anchor_images()?;
        Ok(GenerationRootReadGuard {
            root: self,
            flock,
            lock_image,
            authority_image,
        })
    }

    fn capture_stable_anchor_images(
        &self,
    ) -> GenerationRootResult<(platform::AnchorImage, platform::AnchorImage)> {
        self.revalidate()?;
        let first_lock = platform::capture_anchor_image(&self.lock.inner)?;
        let first_authority = platform::capture_anchor_image(&self.authority.inner)?;
        self.revalidate()?;
        let second_lock = platform::capture_anchor_image(&self.lock.inner)?;
        let second_authority = platform::capture_anchor_image(&self.authority.inner)?;
        self.revalidate()?;
        if !platform::anchor_images_match(&first_lock, &second_lock)
            || !platform::anchor_images_match(&first_authority, &second_authority)
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        Ok((second_lock, second_authority))
    }

    // This deliberately lands one bead before the writer-lease consumer.
    #[cfg_attr(target_os = "macos", allow(clippy::unused_self))]
    pub(in crate::generation_root) fn try_exclusive_anchor_guard_impl(
        &self,
    ) -> GenerationRootResult<GenerationRootExclusiveAnchorGuard<'_>> {
        #[cfg(target_os = "macos")]
        {
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::AcquireLock,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let flock =
                platform::try_anchor_lock(&self.lock.inner, GenerationRootLockMode::Exclusive)?;
            let (lock_image, authority_image) = self.capture_stable_anchor_images()?;
            Ok(GenerationRootExclusiveAnchorGuard {
                root: self,
                flock,
                lock_image,
                authority_image,
            })
        }
    }

    #[cfg(target_os = "macos")]
    pub(in crate::generation_root) fn try_exclusive_anchor_guard_preopened_impl(
        &self,
        fresh_lock_descriptor: std::os::fd::OwnedFd,
    ) -> GenerationRootResult<GenerationRootExclusiveAnchorGuard<'_>> {
        let flock = platform::try_anchor_lock_preopened(
            &self.lock.inner,
            fresh_lock_descriptor,
            GenerationRootLockMode::Exclusive,
        )?;
        let (lock_image, authority_image) = self.capture_stable_anchor_images()?;
        Ok(GenerationRootExclusiveAnchorGuard {
            root: self,
            flock,
            lock_image,
            authority_image,
        })
    }
}

impl fmt::Debug for QualifiedGenerationRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedGenerationRoot")
            .field("anchors", &self.anchor_witnesses())
            .finish_non_exhaustive()
    }
}

/// Shared named-anchor guard carrying fresh exact control images.
///
/// The guard cannot outlive the qualified root that owns the retained anchor
/// descriptors:
///
/// ```compile_fail
/// use frankensearch_index::generation_root::{
///     GenerationRootReadGuard, QualifiedGenerationRoot,
/// };
///
/// fn impossible(root: &QualifiedGenerationRoot) -> GenerationRootReadGuard<'static> {
///     root.read_guard().expect("type-checking must reject this lifetime")
/// }
/// ```
#[must_use = "dropping the guard releases the shared generation-root lock"]
pub struct GenerationRootReadGuard<'root> {
    root: &'root QualifiedGenerationRoot,
    flock: platform::LockHandle,
    lock_image: platform::AnchorImage,
    authority_image: platform::AnchorImage,
}

impl GenerationRootReadGuard<'_> {
    /// Root and fresh LOCK/AUTHORITY witnesses captured under the shared lock.
    #[must_use]
    pub fn witnesses(&self) -> GenerationRootAnchorWitness {
        GenerationRootAnchorWitness::new(
            self.root.witness(),
            self.lock_image.witness,
            self.authority_image.witness,
        )
    }

    /// Bounded no-follow inventory of the retained root, taken while the
    /// shared anchor lock is held.
    ///
    /// # Errors
    ///
    /// Same as [`QualifiedGenerationRoot::inventory`].
    pub fn inventory(&self) -> GenerationRootResult<GenerationRootInventory> {
        self.root.inventory()
    }

    /// Borrow the exact named LOCK image.
    #[must_use]
    pub fn lock_bytes(&self) -> &[u8] {
        &self.lock_image.bytes
    }

    /// Clone the exact named LOCK allocation.
    #[must_use]
    pub fn lock_bytes_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.lock_image.bytes)
    }

    /// SHA-256 of the exact named LOCK image.
    #[must_use]
    pub const fn lock_sha256(&self) -> [u8; 32] {
        self.lock_image.sha256
    }

    /// Borrow the exact named AUTHORITY image.
    #[must_use]
    pub fn authority_bytes(&self) -> &[u8] {
        &self.authority_image.bytes
    }

    /// Clone the exact named AUTHORITY allocation.
    #[must_use]
    pub fn authority_bytes_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.authority_image.bytes)
    }

    /// SHA-256 of the exact named AUTHORITY image.
    #[must_use]
    pub const fn authority_sha256(&self) -> [u8; 32] {
        self.authority_image.sha256
    }

    /// Validate both captured images and explicitly release the shared flock.
    ///
    /// Unlock is attempted even when validation detects ambient mutation.
    ///
    /// # Errors
    ///
    /// Returns a typed content, route, process, or unlock error.
    pub fn release(self) -> GenerationRootResult<()> {
        let validation = platform::validate_anchor_image(&self.root.lock.inner, &self.lock_image)
            .and_then(|()| {
                platform::validate_anchor_image(&self.root.authority.inner, &self.authority_image)
            });
        let unlock = platform::unlock(self.flock);
        validation.and(unlock)
    }
}

impl fmt::Debug for GenerationRootReadGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRootReadGuard")
            .field("witnesses", &self.witnesses())
            .finish_non_exhaustive()
    }
}

/// Crate-private kernel primitive consumed by the later shared-index writer
/// lease. This is not itself a lease and exposes no mutation operation.
#[must_use = "dropping the guard releases the exclusive generation-root lock"]
#[allow(dead_code)]
pub struct GenerationRootExclusiveAnchorGuard<'root> {
    pub(in crate::generation_root) root: &'root QualifiedGenerationRoot,
    pub(in crate::generation_root) flock: platform::LockHandle,
    pub(in crate::generation_root) lock_image: platform::AnchorImage,
    pub(in crate::generation_root) authority_image: platform::AnchorImage,
}

impl GenerationRootExclusiveAnchorGuard<'_> {
    #[allow(dead_code)]
    pub fn release(self) -> GenerationRootResult<()> {
        platform::unlock(self.flock)
    }
}

impl fmt::Debug for GenerationRootExclusiveAnchorGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRootExclusiveAnchorGuard")
            .field("root", &self.root.anchor_witnesses())
            .field("lock_sha256", &self.lock_image.sha256)
            .field("authority_sha256", &self.authority_image.sha256)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for QualifiedGenerationDirectory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedGenerationDirectory")
            .field("witness", &self.witness())
            .finish_non_exhaustive()
    }
}
