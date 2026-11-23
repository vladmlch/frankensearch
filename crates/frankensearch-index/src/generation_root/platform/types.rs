use super::super::{GenerationFileExpectation, GenerationRootObjectWitness};
#[cfg(test)]
use super::super::{GenerationRootLockMode, GenerationRootResult};
use rustix::fd::OwnedFd;
#[cfg(target_os = "macos")]
use rustix::fs::OFlags;
use std::ffi::OsString;
use std::fmt;
use std::sync::Arc;

pub(super) const MODE_PERMISSIONS_MASK: u32 = 0o7777;
#[cfg(target_os = "linux")]
pub(super) const LINUX_STATX_SUBVOLUME_MASK: u32 = 0x0000_8000;

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::generation_root) enum ProcSelfBindingPhase {
    RootQualification,
    MountInfoRead,
    BeforeDataOpen,
    AfterDataOpen,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::generation_root) enum ProcBridgeProbePhase {
    BeforeDataOpen,
    AfterDataOpen,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::generation_root) enum TestBoundary {
    AfterFirstRootQualification,
    BeforeAbsoluteAncestorAclRead {
        index: Option<usize>,
    },
    AfterAbsoluteAncestorAclRead {
        index: Option<usize>,
    },
    BeforeRootComponentOpen {
        index: usize,
    },
    AfterRootComponentOpen {
        index: usize,
    },
    BeforeRelativeDirectoryOpen {
        index: usize,
    },
    AfterRelativeDirectoryOpen {
        index: usize,
    },
    #[cfg(target_os = "linux")]
    BeforeRegularFileOpen {
        index: usize,
    },
    #[cfg(target_os = "linux")]
    AfterRegularFileOpen {
        index: usize,
        cloexec: bool,
    },
    BeforeRootDescriptorDuplicate,
    AfterRootDescriptorDuplicate,
    #[cfg(target_os = "linux")]
    AfterQualifiedFileOpen,
    BeforeRead {
        offset: u64,
    },
    AfterRead {
        offset: u64,
        byte_count: usize,
    },
    BeforeTrailingByteProbe,
    AfterTrailingByteProbe {
        byte_count: usize,
    },
    AfterExactRead,
    #[cfg(target_os = "linux")]
    BeforeFileRouteReopen,
    #[cfg(target_os = "linux")]
    AfterFileRouteReopen,
    #[cfg(target_os = "linux")]
    BeforeFinalRootRevalidation,
    BeforeFilesystemQualification,
    AfterFilesystemQualification,
    BeforeRouteMountIdentity,
    AfterRouteMountIdentity,
    BeforeObjectWitness,
    AfterObjectWitness,
    #[cfg(target_os = "linux")]
    BeforeMountInfoOpen,
    #[cfg(target_os = "linux")]
    AfterMountInfoOpen,
    #[cfg(target_os = "linux")]
    BeforeMountInfoRead,
    #[cfg(target_os = "linux")]
    AfterMountInfoRead {
        byte_count: usize,
    },
    #[cfg(target_os = "linux")]
    BeforeFinalProbeOpen {
        index: usize,
    },
    #[cfg(target_os = "linux")]
    AfterFinalProbeOpen {
        index: usize,
    },
    #[cfg(target_os = "linux")]
    AfterProbeQualified {
        index: usize,
    },
    #[cfg(target_os = "linux")]
    BeforeProcCapabilityRootOpen,
    #[cfg(target_os = "linux")]
    AfterProcCapabilityRootOpen,
    #[cfg(target_os = "linux")]
    BeforeProcSelfReadlink {
        phase: ProcSelfBindingPhase,
    },
    #[cfg(target_os = "linux")]
    AfterProcSelfReadlink {
        phase: ProcSelfBindingPhase,
    },
    #[cfg(target_os = "linux")]
    BeforeProcThreadSelfReadlink {
        phase: ProcSelfBindingPhase,
    },
    #[cfg(target_os = "linux")]
    AfterProcThreadSelfReadlink {
        phase: ProcSelfBindingPhase,
    },
    #[cfg(target_os = "linux")]
    BeforeProcBridgeProbe {
        index: usize,
        phase: ProcBridgeProbePhase,
    },
    #[cfg(target_os = "linux")]
    AfterProcBridgeProbe {
        index: usize,
        phase: ProcBridgeProbePhase,
    },
    #[cfg(target_os = "linux")]
    BeforeProcFdReopen {
        index: usize,
    },
    #[cfg(target_os = "linux")]
    AfterProcFdReopen {
        index: usize,
    },
    #[cfg(target_os = "linux")]
    BeforeFinalRouteProbe {
        index: usize,
    },
    #[cfg(target_os = "linux")]
    AfterFinalRouteProbe {
        index: usize,
    },
    BeforeLock,
    AfterLock,
    BeforeUnlock,
    AfterUnlock,
    BeforeFileSync,
    AfterFileSync,
    BeforeDirectorySync,
    AfterDirectorySync,
    BeforeExactNameEnumeration,
    AfterExactNameEnumeration,
    BeforeAclRead,
    AfterAclRead,
    BeforeSnapshotCellCompareExchange,
    #[cfg(target_os = "macos")]
    BeforePreopenedQualification,
    #[cfg(target_os = "macos")]
    AfterPreopenedQualification,
    #[cfg(target_os = "macos")]
    BeforeFinalRouteStat,
    #[cfg(target_os = "macos")]
    AfterFinalRouteStat,
    // Publisher crash points (bd-ycng6): a hook error here models a crash
    // or fault exactly between two durable steps of one publication.
    BeforePublishAttemptFrame,
    AfterPublishAttemptFrame,
    BeforePublishSlotWrite,
    /// After each bounded chunk of the slot write; `written` bytes are on
    /// disk, the rest are not: a fault here is a genuine short write.
    DuringPublishSlotWrite {
        written: usize,
    },
    AfterPublishSlotWrite,
    BeforePublishSlotReread,
    BeforePublishFloorAdvance,
    AfterPublishFloorAdvance,
    BeforePublishOwnerFrame,
}

/// Publisher-visible crash point (test builds only).
#[cfg(test)]
pub(in crate::generation_root) fn crash_point(boundary: TestBoundary) -> GenerationRootResult<()> {
    test_boundary(boundary)
}

#[cfg(test)]
type TestHook = Box<dyn FnMut(TestBoundary) -> GenerationRootResult<()>>;

#[cfg(test)]
std::thread_local! {
    static TEST_HOOK: std::cell::RefCell<Option<TestHook>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(in crate::generation_root) struct TestHookGuard;

#[cfg(test)]
impl Drop for TestHookGuard {
    fn drop(&mut self) {
        TEST_HOOK.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

#[cfg(test)]
pub(in crate::generation_root) fn install_test_hook(
    hook: impl FnMut(TestBoundary) -> GenerationRootResult<()> + 'static,
) -> TestHookGuard {
    TEST_HOOK.with(|slot| {
        let replaced = slot.borrow_mut().replace(Box::new(hook));
        assert!(
            replaced.is_none(),
            "only one test hook may be active per thread"
        );
    });
    TestHookGuard
}

#[cfg(test)]
pub(in crate::generation_root) fn test_boundary(
    boundary: TestBoundary,
) -> GenerationRootResult<()> {
    TEST_HOOK.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.as_mut().map_or(Ok(()), |hook| hook(boundary))
    })
}

#[derive(Clone)]
pub(in crate::generation_root) struct RelativePath {
    pub(super) components: Arc<[OsString]>,
}

#[derive(Clone)]
pub(in crate::generation_root) struct RootHandle {
    pub(super) state: Arc<RootState>,
}

pub(super) struct RootState {
    pub(super) route: AbsoluteRoute,
    pub(super) descriptor: OwnedFd,
    pub(super) witness: GenerationRootObjectWitness,
    pub(super) route_identities: Arc<[BasicObjectIdentity]>,
    pub(super) creator_process_id: u32,
}

pub(in crate::generation_root) struct FileHandle {
    pub(super) descriptor: OwnedFd,
    pub(super) root: RootHandle,
    pub(super) path: RelativePath,
    pub(super) expectation: GenerationFileExpectation,
    pub(super) content_witness: ExactContentWitness,
    pub(super) component_identities: Arc<[BasicObjectIdentity]>,
}

pub(in crate::generation_root) struct FileAdmission {
    pub(in crate::generation_root) handle: FileHandle,
    pub(in crate::generation_root) bytes: Arc<[u8]>,
    pub(in crate::generation_root) witness: GenerationRootObjectWitness,
    pub(in crate::generation_root) sha256: [u8; 32],
}

pub(in crate::generation_root) struct ControlHandle {
    pub(super) descriptor: OwnedFd,
    #[cfg(test)]
    pub(super) bytes: Arc<[u8]>,
    pub(super) root: RootHandle,
    pub(super) path: RelativePath,
    pub(super) expectation: GenerationFileExpectation,
    pub(super) witness: GenerationRootObjectWitness,
    #[cfg(test)]
    pub(super) content_witness: ExactContentWitness,
    pub(super) component_identities: Arc<[BasicObjectIdentity]>,
}

pub(in crate::generation_root) struct AnchorImage {
    pub(in crate::generation_root) bytes: Arc<[u8]>,
    pub(in crate::generation_root) witness: GenerationRootObjectWitness,
    pub(in crate::generation_root) sha256: [u8; 32],
}

pub(in crate::generation_root) struct LockHandle {
    pub(super) descriptor: OwnedFd,
    #[cfg(test)]
    pub(super) bytes: Arc<[u8]>,
    pub(super) witness: GenerationRootObjectWitness,
    pub(super) content_witness: ExactContentWitness,
    pub(super) root: RootHandle,
    pub(super) path: RelativePath,
    pub(super) expectation: GenerationFileExpectation,
    pub(super) component_identities: Arc<[BasicObjectIdentity]>,
    pub(super) creator_process_id: u32,
    pub(super) lock_held: bool,
    #[cfg(target_os = "macos")]
    pub(super) descriptor_status_flags: OFlags,
    #[cfg(test)]
    pub(super) mode: GenerationRootLockMode,
}

pub(super) struct AcquiredFlockDescriptor {
    descriptor: Option<OwnedFd>,
    pub(super) creator_process_id: u32,
}

impl AcquiredFlockDescriptor {
    pub(super) fn new(descriptor: OwnedFd, creator_process_id: u32) -> Self {
        Self {
            descriptor: Some(descriptor),
            creator_process_id,
        }
    }

    pub(super) fn descriptor(&self) -> &OwnedFd {
        let Some(descriptor) = self.descriptor.as_ref() else {
            unreachable!("an armed flock descriptor always owns its descriptor")
        };
        descriptor
    }

    pub(super) fn into_descriptor(mut self) -> OwnedFd {
        let Some(descriptor) = self.descriptor.take() else {
            unreachable!("an armed flock descriptor transfers exactly once")
        };
        descriptor
    }
}

impl Drop for AcquiredFlockDescriptor {
    fn drop(&mut self) {
        if self.creator_process_id == std::process::id()
            && let Some(descriptor) = self.descriptor.as_ref()
        {
            let _ = rustix::fs::flock(descriptor, rustix::fs::FlockOperation::Unlock);
        }
    }
}

#[derive(Clone)]
pub(super) struct AbsoluteRoute {
    pub(super) components: Arc<[OsString]>,
}

pub(super) struct OpenedRoute {
    pub(super) descriptor: OwnedFd,
    pub(super) route_identities: Vec<BasicObjectIdentity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BasicObjectIdentity {
    pub(super) device: u64,
    pub(super) inode: u64,
    pub(super) mode: u32,
    pub(super) uid: u32,
    pub(super) gid: u32,
    pub(super) mount_identity: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AbsoluteAncestorSecurityWitness {
    pub(super) identity: BasicObjectIdentity,
    pub(super) changed_seconds: i64,
    pub(super) changed_nanoseconds: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ExactContentWitness {
    pub(super) sha256: [u8; 32],
}

impl fmt::Debug for RootHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RootHandle")
            .field("witness", &self.state.witness)
            .finish_non_exhaustive()
    }
}
