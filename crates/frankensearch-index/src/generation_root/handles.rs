use std::fmt;
use std::sync::Arc;

#[cfg(all(test, target_os = "macos"))]
use super::{GenerationRootError, GenerationRootErrorKind, GenerationRootStage};
use super::{GenerationRootObjectWitness, GenerationRootResult, platform};

/// Retained descriptor plus the exact bytes admitted from it.
pub struct QualifiedGenerationFile {
    pub(in crate::generation_root) handle: platform::FileHandle,
    pub(in crate::generation_root) bytes: Arc<[u8]>,
    pub(in crate::generation_root) witness: GenerationRootObjectWitness,
    pub(in crate::generation_root) sha256: [u8; 32],
}

impl QualifiedGenerationFile {
    /// Borrow the exact owned byte image.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Clone the exact owned byte allocation for a descriptor-independent
    /// consumer such as `ValidatedFsviBytes::from_arc`.
    #[must_use]
    pub fn bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    /// Descriptor witness captured after complete admission.
    #[must_use]
    pub const fn witness(&self) -> GenerationRootObjectWitness {
        self.witness
    }

    /// SHA-256 of the exact owned byte image.
    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// Apply the platform's strongest required regular-file durability
    /// barrier to the retained descriptor.
    ///
    /// The descriptor is read and SHA-256 checked against the admitted exact
    /// image both before and after the barrier. This explicit operation
    /// therefore performs two full-file reads.
    ///
    /// # Errors
    ///
    /// Returns a typed route, identity, or durability error. A required
    /// durability primitive never falls back to a weaker barrier.
    pub fn sync_durable(&self) -> GenerationRootResult<()> {
        platform::sync_file(&self.handle, self.witness)
    }
}

impl fmt::Debug for QualifiedGenerationFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedGenerationFile")
            .field("witness", &self.witness)
            .finish_non_exhaustive()
    }
}

/// Internal qualified capability for one mutable private control anchor.
pub(in crate::generation_root) struct QualifiedGenerationControlFile {
    pub(in crate::generation_root) inner: platform::ControlHandle,
}

impl QualifiedGenerationControlFile {
    /// Object/metadata witness captured when this control file was admitted.
    ///
    /// This does not attest mutable content; the fixed named-anchor guards
    /// capture a fresh exact image while holding the kernel flock.
    #[must_use]
    pub(in crate::generation_root) fn witness(&self) -> GenerationRootObjectWitness {
        platform::control_witness(&self.inner)
    }

    /// Clone the exact descriptor-admitted control-file image.
    #[must_use]
    #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
    pub(in crate::generation_root) fn bytes(&self) -> Arc<[u8]> {
        platform::control_bytes(&self.inner)
    }

    /// SHA-256 of the exact descriptor-admitted control-file image.
    #[must_use]
    #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
    pub(in crate::generation_root) fn sha256(&self) -> [u8; 32] {
        platform::control_sha256(&self.inner)
    }

    /// Attempt a shared or exclusive non-blocking kernel lock on this exact
    /// qualified control-file identity.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::LockContended`] rather than
    /// blocking. Route replacement, process-identity drift, and all normal
    /// file-admission errors remain fail-closed.
    #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
    #[cfg_attr(target_os = "macos", allow(clippy::unused_self))]
    pub(in crate::generation_root) fn try_lock(
        &self,
        mode: GenerationRootLockMode,
    ) -> GenerationRootResult<GenerationRootLock> {
        #[cfg(target_os = "macos")]
        {
            let _ = mode;
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::AcquireLock,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            platform::try_lock(&self.inner, mode).map(|inner| GenerationRootLock { inner })
        }
    }

    /// Attempt an Apple Silicon lock using a fresh provider-owned open
    /// description for the exact admitted control object.
    ///
    /// A duplicated descriptor is not sufficient: `flock` state is attached
    /// to the open description. The external provider owns acquisition
    /// provenance; this method verifies descriptor type, access, route, and
    /// security before the non-blocking flock, then verifies the candidate's
    /// bound content only after flock succeeds. Exact Darwin status validation
    /// also rejects the persistent `FWASLOCKED` history bit left by a prior
    /// successful flock, but that defense does not prove the absence of unseen
    /// aliases. The provider must transfer the sole live reference and retain
    /// no duplicate. A surviving alias could convert or unlock the flock while
    /// this guard is live; creator-process drop and explicit unlock
    /// nevertheless issue `LOCK_UN` so an inherited alias cannot prolong the
    /// guard after release.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed descriptor/admission error or
    /// [`GenerationRootErrorKind::LockContended`].
    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    pub(in crate::generation_root) fn try_lock_preopened(
        &self,
        descriptor: std::os::fd::OwnedFd,
        mode: GenerationRootLockMode,
    ) -> GenerationRootResult<GenerationRootLock> {
        platform::try_lock_preopened(&self.inner, descriptor, mode)
            .map(|inner| GenerationRootLock { inner })
    }
}

impl fmt::Debug for QualifiedGenerationControlFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedGenerationControlFile")
            .field("witness", &self.witness())
            .finish_non_exhaustive()
    }
}

/// Internal requested kernel lock mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::generation_root) enum GenerationRootLockMode {
    /// Reader/shared lock.
    Shared,
    /// Writer/exclusive lock.
    #[allow(dead_code)]
    Exclusive,
}

/// Owned non-blocking kernel lock guard.
///
/// In the creator process, dropping the guard issues a best-effort explicit
/// `LOCK_UN` before closing the descriptor. This releases the open-description
/// lock even when a forked child still retains an inherited descriptor.
/// A child-side drop never unlocks the creator's guard. [`Self::unlock`]
/// reports explicit validation or unlock failures.
///
/// Applications must not continue normal multithreaded library execution
/// across `fork()`: the child may only close inherited capabilities or `exec`.
/// Operations that revalidate, synchronize, or unlock in a child fail with
/// [`GenerationRootErrorKind::ForkedProcess`].
#[must_use = "holding this guard keeps the kernel lock acquired"]
#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
pub(in crate::generation_root) struct GenerationRootLock {
    inner: platform::LockHandle,
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
impl GenerationRootLock {
    /// Object/metadata witness bound to this lock.
    ///
    /// This does not attest content. Use [`Self::sha256`] or [`Self::bytes`]
    /// for the exact locked image.
    #[must_use]
    pub(in crate::generation_root) fn witness(&self) -> GenerationRootObjectWitness {
        platform::lock_witness(&self.inner)
    }

    /// Clone the exact control-file image validated while the lock was held.
    #[must_use]
    pub(in crate::generation_root) fn bytes(&self) -> Arc<[u8]> {
        platform::lock_bytes(&self.inner)
    }

    /// SHA-256 of the exact control-file image validated while the lock was
    /// held.
    #[must_use]
    pub(in crate::generation_root) fn sha256(&self) -> [u8; 32] {
        platform::lock_sha256(&self.inner)
    }

    /// Lock mode bound to this guard.
    #[must_use]
    pub(in crate::generation_root) fn mode(&self) -> GenerationRootLockMode {
        platform::lock_mode(&self.inner)
    }

    /// Explicitly release this lock.
    ///
    /// # Errors
    ///
    /// Returns a typed process, content, route, or OS error. Explicit
    /// creator-process `LOCK_UN` is the primary release boundary; creator-side
    /// drop retries it best-effort after a reported failure. Closing the owned
    /// descriptor is only a fallback when no inherited or duplicated
    /// open-description reference survives.
    pub(in crate::generation_root) fn unlock(self) -> GenerationRootResult<()> {
        platform::unlock(self.inner)
    }

    /// Apply the required file durability barrier to the locked control file.
    ///
    /// This synchronizes the unchanged admitted image. It neither authorizes
    /// nor absorbs an ambient mutation.
    ///
    /// # Errors
    ///
    /// Returns a typed process, content, route, identity, or durability error.
    /// A required durability primitive never falls back to a weaker barrier.
    pub(in crate::generation_root) fn sync_durable(&self) -> GenerationRootResult<()> {
        platform::sync_lock_file(&self.inner)
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
impl fmt::Debug for GenerationRootLock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRootLock")
            .field("held", &true)
            .finish_non_exhaustive()
    }
}
