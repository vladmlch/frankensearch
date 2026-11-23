use super::super::{
    GENERATION_ROOT_MAX_MOUNTINFO_BYTES, GenerationFileExpectation, GenerationRootError,
    GenerationRootErrorKind, GenerationRootObjectWitness, GenerationRootResult,
    GenerationRootStage, QualifiedFilesystem,
};
use super::ops::validate_process_identity;
use super::types::{
    AbsoluteRoute, LINUX_STATX_SUBVOLUME_MASK, OpenedRoute, ProcBridgeProbePhase,
    ProcSelfBindingPhase,
};
#[cfg(test)]
use super::types::{TestBoundary, test_boundary};
use super::unix_common::{
    basic_identity, os_error, qualify_absolute_ancestor, stat_inode_as_u64, stat_link_count_as_u64,
    stat_mode_as_u32, stat_seconds_as_i64, usize_as_u64, validate_private_directory,
    validate_private_regular_file, validate_private_regular_file_metadata,
    verify_exact_directory_entry,
};
use rustix::fd::{AsFd, OwnedFd};
use rustix::fs::{FileType, Mode, OFlags, fstat, fstatfs, fstatvfs};
use rustix::io::Errno;
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

#[cfg(target_os = "linux")]
pub(super) fn open_absolute_route(
    route: &AbsoluteRoute,
    stage: GenerationRootStage,
) -> GenerationRootResult<OpenedRoute> {
    use rustix::fs::{CWD, ResolveFlags, openat2};

    let flags =
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    let mut descriptor = openat2(
        CWD,
        Path::new("/"),
        flags,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )
    .map_err(|error| linux_open_error(stage, error))?;
    let mut route_identities = Vec::new();
    route_identities
        .try_reserve(route.components.len() + 1)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
    route_identities.push(qualify_absolute_ancestor(&descriptor, stage, None)?);
    for (index, component) in route.components.iter().enumerate() {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRootComponentOpen { index })?;
        verify_exact_directory_entry(&descriptor, component, stage, index)?;
        let next = openat2(
            &descriptor,
            component,
            flags,
            Mode::empty(),
            ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
        )
        .map_err(|error| linux_component_open_error(&descriptor, component, stage, index, error))?;
        verify_exact_directory_entry(&descriptor, component, stage, index)?;
        descriptor = next;
        let stat = fstat(&descriptor).map_err(|error| os_error(stage, error))?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::NotDirectory, stage)
                    .at_component(index),
            );
        }
        let identity = if index + 1 == route.components.len() {
            basic_identity(&descriptor, stage)?
        } else {
            qualify_absolute_ancestor(&descriptor, stage, Some(index))?
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterRootComponentOpen { index })?;
        route_identities.push(identity);
    }
    Ok(OpenedRoute {
        descriptor,
        route_identities,
    })
}

#[cfg(target_os = "linux")]
pub(super) fn open_relative_directory_component(
    parent: &OwnedFd,
    component: &OsStr,
    index: usize,
    root: GenerationRootObjectWitness,
    stage: GenerationRootStage,
) -> GenerationRootResult<OwnedFd> {
    use rustix::fs::{ResolveFlags, openat2};

    #[cfg(test)]
    test_boundary(TestBoundary::BeforeRelativeDirectoryOpen { index })?;
    verify_exact_directory_entry(parent, component, stage, index)?;
    let descriptor = openat2(
        parent,
        component,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
        ResolveFlags::BENEATH
            | ResolveFlags::NO_SYMLINKS
            | ResolveFlags::NO_MAGICLINKS
            | ResolveFlags::NO_XDEV,
    )
    .map_err(|error| linux_component_open_error(parent, component, stage, index, error))?;
    verify_exact_directory_entry(parent, component, stage, index)?;
    let witness = object_witness(&descriptor, root.filesystem, root.mount_identity, stage)?;
    if witness.mount_identity != root.mount_identity {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::CrossDevice, stage)
                .at_component(index),
        );
    }
    validate_private_directory(&descriptor, witness, stage)?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterRelativeDirectoryOpen { index })?;
    Ok(descriptor)
}

#[cfg(target_os = "linux")]
pub(super) fn open_regular_component(
    parent: &OwnedFd,
    component: &OsStr,
    index: usize,
    root: GenerationRootObjectWitness,
    expectation: GenerationFileExpectation,
    writable: bool,
    creator_process_id: u32,
    stage: GenerationRootStage,
) -> GenerationRootResult<OwnedFd> {
    use rustix::fs::openat;

    let access = if writable {
        OFlags::RDWR
    } else {
        OFlags::RDONLY
    };
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeRegularFileOpen { index })?;
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFinalProbeOpen { index })?;
    let (probe, probe_witness) =
        probe_regular_component(parent, component, index, root, expectation, stage)?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterProbeQualified { index })?;
    validate_process_identity(creator_process_id, stage)?;
    let (proc_root, proc_mount_id) = open_verified_proc_root(creator_process_id, stage)?;
    let probe_route = proc_thread_fd_route(probe.as_raw_fd());
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeProcFdReopen { index })?;
    validate_proc_bridge_entry(
        &proc_root,
        &probe_route,
        proc_mount_id,
        index,
        ProcBridgeProbePhase::BeforeDataOpen,
        stage,
    )?;
    validate_proc_self_binding(
        &proc_root,
        creator_process_id,
        ProcSelfBindingPhase::BeforeDataOpen,
        stage,
    )?;
    let descriptor = loop {
        match openat(
            &proc_root,
            &probe_route,
            access | OFlags::CLOEXEC | OFlags::NONBLOCK | OFlags::NOATIME,
            Mode::empty(),
        ) {
            Ok(descriptor) => break descriptor,
            Err(error) if error == Errno::INTR => {}
            Err(error) if error == Errno::PERM || error == Errno::ACCESS => {
                validate_process_identity(creator_process_id, stage)
                    .map_err(|process_error| process_error.at_component(index))?;
                let retained_probe =
                    object_witness(&probe, root.filesystem, root.mount_identity, stage)
                        .map_err(|witness_error| witness_error.at_component(index))?;
                if !probe_witness.mutation_stable_eq(retained_probe) {
                    return Err(GenerationRootError::new(
                        GenerationRootErrorKind::ObjectChanged,
                        stage,
                    )
                    .at_component(index));
                }
                return Err(proc_bridge_error(stage, error).at_component(index));
            }
            Err(error) => {
                return Err(proc_bridge_error(stage, error).at_component(index));
            }
        }
    };
    validate_proc_self_binding(
        &proc_root,
        creator_process_id,
        ProcSelfBindingPhase::AfterDataOpen,
        stage,
    )
    .map_err(|error| error.at_component(index))?;
    validate_proc_bridge_entry(
        &proc_root,
        &probe_route,
        proc_mount_id,
        index,
        ProcBridgeProbePhase::AfterDataOpen,
        stage,
    )?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterProcFdReopen { index })?;
    let reopened_witness =
        object_witness(&descriptor, root.filesystem, root.mount_identity, stage)?;
    validate_private_regular_file(&descriptor, reopened_witness, expectation, stage)?;
    if !probe_witness.mutation_stable_eq(reopened_witness) {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index),
        );
    }
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFinalRouteProbe { index })?;
    let (_final_probe, final_probe_witness) =
        probe_regular_component(parent, component, index, root, expectation, stage)?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFinalRouteProbe { index })?;
    if !probe_witness.mutation_stable_eq(final_probe_witness)
        || !reopened_witness.mutation_stable_eq(final_probe_witness)
    {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index),
        );
    }
    #[cfg(test)]
    {
        let cloexec = rustix::io::fcntl_getfd(&descriptor)
            .map_err(|error| os_error(stage, error).at_component(index))?
            .contains(rustix::io::FdFlags::CLOEXEC);
        test_boundary(TestBoundary::AfterRegularFileOpen { index, cloexec })?;
    }
    validate_process_identity(creator_process_id, stage)?;
    Ok(descriptor)
}

#[cfg(target_os = "linux")]
fn proc_thread_fd_route(raw_fd: RawFd) -> String {
    format!("thread-self/fd/{raw_fd}")
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn proc_thread_fd_route_for_test(raw_fd: RawFd) -> String {
    proc_thread_fd_route(raw_fd)
}

#[cfg(target_os = "linux")]
pub(super) fn probe_regular_component(
    parent: &OwnedFd,
    component: &OsStr,
    index: usize,
    root: GenerationRootObjectWitness,
    expectation: GenerationFileExpectation,
    stage: GenerationRootStage,
) -> GenerationRootResult<(OwnedFd, GenerationRootObjectWitness)> {
    use rustix::fs::{ResolveFlags, openat2};

    verify_exact_directory_entry(parent, component, stage, index)?;
    let probe = openat2(
        parent,
        component,
        OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
        ResolveFlags::BENEATH
            | ResolveFlags::NO_SYMLINKS
            | ResolveFlags::NO_MAGICLINKS
            | ResolveFlags::NO_XDEV,
    )
    .map_err(|error| linux_open_error(stage, error).at_component(index))?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFinalProbeOpen { index })?;
    verify_exact_directory_entry(parent, component, stage, index)?;
    validate_regular_descriptor_type(&probe, stage, index)?;
    let witness = object_witness(&probe, root.filesystem, root.mount_identity, stage)?;
    if witness.mount_identity != root.mount_identity {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::CrossDevice, stage)
                .at_component(index),
        );
    }
    validate_private_regular_file_metadata(&probe, witness, expectation, stage)?;
    Ok((probe, witness))
}

#[cfg(target_os = "linux")]
pub(in crate::generation_root) fn validate_regular_descriptor_type(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
    index: usize,
) -> GenerationRootResult<()> {
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error).at_component(index))?;
    match FileType::from_raw_mode(stat.st_mode) {
        FileType::RegularFile => Ok(()),
        FileType::Symlink => Err(GenerationRootError::new(
            GenerationRootErrorKind::SymbolicLink,
            stage,
        )
        .at_component(index)),
        _ => Err(
            GenerationRootError::new(GenerationRootErrorKind::NotRegularFile, stage)
                .at_component(index),
        ),
    }
}

#[cfg(target_os = "linux")]
fn open_verified_proc_root(
    process_id: u32,
    stage: GenerationRootStage,
) -> GenerationRootResult<(OwnedFd, u64)> {
    use rustix::fs::{CWD, ResolveFlags, openat2};

    validate_process_identity(process_id, stage)?;
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeProcCapabilityRootOpen)?;
    let descriptor = loop {
        match openat2(
            CWD,
            Path::new("/proc"),
            OFlags::PATH | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
            ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
        ) {
            Ok(descriptor) => break descriptor,
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_root_error(stage, error));
            }
        }
    };
    let filesystem = loop {
        match fstatfs(&descriptor) {
            Ok(filesystem) => break filesystem,
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_root_error(stage, error));
            }
        }
    };
    if i128::from(filesystem.f_type) != i128::from(libc::PROC_SUPER_MAGIC) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedKernelFeature,
            stage,
        ));
    }
    let stat = loop {
        match fstat(&descriptor) {
            Ok(stat) => break stat,
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_root_error(stage, error));
            }
        }
    };
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedKernelFeature,
            stage,
        ));
    }
    let proc_mount_id = proc_mount_id(&descriptor, stage)?;
    validate_proc_self_binding(
        &descriptor,
        process_id,
        ProcSelfBindingPhase::RootQualification,
        stage,
    )?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterProcCapabilityRootOpen)?;
    validate_process_identity(process_id, stage)?;
    Ok((descriptor, proc_mount_id))
}

#[cfg(target_os = "linux")]
fn validate_proc_self_binding(
    proc_root: &OwnedFd,
    process_id: u32,
    phase: ProcSelfBindingPhase,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    use rustix::fs::readlinkat;

    let _ = phase;

    if process_id != std::process::id() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ForkedProcess,
            stage,
        ));
    }
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeProcSelfReadlink { phase })?;
    let target = loop {
        match readlinkat(proc_root, "self", Vec::new()) {
            Ok(target) => break target,
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_bridge_error(stage, error));
            }
        }
    };
    #[cfg(test)]
    test_boundary(TestBoundary::AfterProcSelfReadlink { phase })?;
    if process_id != std::process::id() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ForkedProcess,
            stage,
        ));
    }
    if target.as_bytes() != process_id.to_string().as_bytes() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ));
    }
    let thread_id = rustix::thread::gettid();
    let expected_thread_target = format!("{process_id}/task/{thread_id}");
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeProcThreadSelfReadlink { phase })?;
    let thread_target = loop {
        match readlinkat(proc_root, "thread-self", Vec::new()) {
            Ok(target) => break target,
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_bridge_error(stage, error));
            }
        }
    };
    #[cfg(test)]
    test_boundary(TestBoundary::AfterProcThreadSelfReadlink { phase })?;
    if process_id != std::process::id() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ForkedProcess,
            stage,
        ));
    }
    if thread_target.as_bytes() != expected_thread_target.as_bytes() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn validate_proc_bridge_entry(
    proc_root: &OwnedFd,
    probe_route: &str,
    expected_proc_mount_id: u64,
    index: usize,
    phase: ProcBridgeProbePhase,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    use rustix::fs::{ResolveFlags, openat2};

    let _ = phase;

    #[cfg(test)]
    test_boundary(TestBoundary::BeforeProcBridgeProbe { index, phase })?;
    let descriptor = loop {
        match openat2(
            proc_root,
            probe_route,
            OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
            ResolveFlags::NO_XDEV,
        ) {
            Ok(descriptor) => break descriptor,
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_bridge_error(stage, error).at_component(index));
            }
        }
    };
    let stat = loop {
        match fstat(&descriptor) {
            Ok(stat) => break stat,
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_bridge_error(stage, error).at_component(index));
            }
        }
    };
    if FileType::from_raw_mode(stat.st_mode) != FileType::Symlink {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index),
        );
    }
    let filesystem = loop {
        match fstatfs(&descriptor) {
            Ok(filesystem) => break filesystem,
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_bridge_error(stage, error).at_component(index));
            }
        }
    };
    if i128::from(filesystem.f_type) != i128::from(libc::PROC_SUPER_MAGIC) {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index),
        );
    }
    let observed_mount_id =
        proc_mount_id(&descriptor, stage).map_err(|error| error.at_component(index))?;
    if observed_mount_id != expected_proc_mount_id {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index)
                .with_counts(expected_proc_mount_id, observed_mount_id),
        );
    }
    #[cfg(test)]
    test_boundary(TestBoundary::AfterProcBridgeProbe { index, phase })?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn proc_mount_id(descriptor: &OwnedFd, stage: GenerationRootStage) -> GenerationRootResult<u64> {
    let extended = proc_statx(descriptor, stage)?;
    validate_proc_mount_id_mask(
        rustix::fs::StatxFlags::from_bits_retain(extended.stx_mask),
        extended.stx_mnt_id,
        stage,
    )
}

#[cfg(target_os = "linux")]
fn proc_statx(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
) -> GenerationRootResult<rustix::fs::Statx> {
    use rustix::fs::{AtFlags, StatxFlags, statx};

    loop {
        match statx(
            descriptor,
            "",
            AtFlags::EMPTY_PATH | AtFlags::NO_AUTOMOUNT,
            StatxFlags::BASIC_STATS | StatxFlags::MNT_ID,
        ) {
            Ok(extended) => return Ok(extended),
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(proc_bridge_error(stage, error));
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn validate_proc_mount_id_mask(
    returned_mask: rustix::fs::StatxFlags,
    mount_id: u64,
    stage: GenerationRootStage,
) -> GenerationRootResult<u64> {
    if returned_mask.contains(rustix::fs::StatxFlags::MNT_ID) {
        Ok(mount_id)
    } else {
        Err(
            GenerationRootError::new(GenerationRootErrorKind::UnsupportedKernelFeature, stage)
                .with_counts(
                    u64::from(rustix::fs::StatxFlags::MNT_ID.bits()),
                    u64::from(returned_mask.bits() & rustix::fs::StatxFlags::MNT_ID.bits()),
                ),
        )
    }
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn validate_proc_mount_id_mask_for_test(
    returned_mask: rustix::fs::StatxFlags,
    mount_id: u64,
) -> GenerationRootResult<u64> {
    validate_proc_mount_id_mask(
        returned_mask,
        mount_id,
        GenerationRootStage::OpenRegularFile,
    )
}

#[cfg(target_os = "linux")]
fn proc_bridge_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
    let kind = if error == Errno::MFILE || error == Errno::NFILE || error == Errno::NOMEM {
        GenerationRootErrorKind::ResourceLimit
    } else if error == Errno::NOSYS || error == Errno::PERM || error == Errno::ACCESS {
        GenerationRootErrorKind::UnsupportedKernelFeature
    } else if error == Errno::NOENT
        || error == Errno::NOTDIR
        || error == Errno::LOOP
        || error == Errno::XDEV
        || error == Errno::STALE
    {
        GenerationRootErrorKind::ObjectChanged
    } else {
        GenerationRootErrorKind::Io
    };
    GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
}

#[cfg(target_os = "linux")]
fn proc_root_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
    let kind = if error == Errno::MFILE || error == Errno::NFILE || error == Errno::NOMEM {
        GenerationRootErrorKind::ResourceLimit
    } else if error == Errno::NOSYS
        || error == Errno::PERM
        || error == Errno::ACCESS
        || error == Errno::NOENT
        || error == Errno::NOTDIR
        || error == Errno::LOOP
        || error == Errno::XDEV
    {
        GenerationRootErrorKind::UnsupportedKernelFeature
    } else {
        GenerationRootErrorKind::Io
    };
    GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn proc_bridge_error_for_test(error: Errno) -> GenerationRootError {
    proc_bridge_error(GenerationRootStage::OpenRegularFile, error)
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn proc_root_error_for_test(error: Errno) -> GenerationRootError {
    proc_root_error(GenerationRootStage::OpenRegularFile, error)
}

#[cfg(target_os = "linux")]
pub(super) fn route_mount_identity(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
) -> GenerationRootResult<[u8; 32]> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeRouteMountIdentity)?;
    let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
    let extended = linux_statx(
        descriptor,
        stage,
        linux_filesystem_is_btrfs(filesystem_stat.f_type),
    )?;
    let identity = linux_filesystem_namespace_identity(filesystem_stat.f_type, &extended, stage)?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterRouteMountIdentity)?;
    Ok(identity)
}

#[cfg(target_os = "linux")]
pub(super) fn qualify_filesystem(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
) -> GenerationRootResult<(QualifiedFilesystem, [u8; 32])> {
    use rustix::fs::StatVfsMountFlags;

    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFilesystemQualification)?;
    let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
    let mount_stat = fstatvfs(descriptor).map_err(|error| os_error(stage, error))?;
    if mount_stat.f_flag.contains(StatVfsMountFlags::RDONLY) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ReadOnlyFilesystem,
            stage,
        ));
    }
    let extended = linux_statx(
        descriptor,
        stage,
        linux_filesystem_is_btrfs(filesystem_stat.f_type),
    )?;
    let filesystem =
        linux_filesystem_profile_at_stage(filesystem_stat.f_type, extended.stx_mnt_id, stage)?;
    let mount_identity =
        linux_filesystem_namespace_identity(filesystem_stat.f_type, &extended, stage)?;
    let qualified = (filesystem, mount_identity);
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFilesystemQualification)?;
    Ok(qualified)
}

#[cfg(target_os = "linux")]
pub(super) fn object_witness(
    descriptor: &OwnedFd,
    expected_filesystem: QualifiedFilesystem,
    expected_mount_identity: [u8; 32],
    stage: GenerationRootStage,
) -> GenerationRootResult<GenerationRootObjectWitness> {
    use rustix::fs::StatxFlags;

    #[cfg(test)]
    test_boundary(TestBoundary::BeforeObjectWitness)?;
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
    let extended = linux_statx(
        descriptor,
        stage,
        linux_filesystem_is_btrfs(filesystem_stat.f_type),
    )?;
    let mask = StatxFlags::from_bits_retain(extended.stx_mask);
    if !mask.contains(StatxFlags::BASIC_STATS) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedKernelFeature,
            stage,
        ));
    }
    validate_linux_writable_attributes(
        extended.stx_attributes,
        extended.stx_attributes_mask,
        stage,
    )?;
    if !linux_filesystem_type_matches(expected_filesystem, filesystem_stat.f_type) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::CrossDevice,
            stage,
        ));
    }
    let mount_identity =
        linux_filesystem_namespace_identity(filesystem_stat.f_type, &extended, stage)?;
    if mount_identity != expected_mount_identity {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::CrossDevice,
            stage,
        ));
    }
    let size = u64::try_from(stat.st_size)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
    let mode = stat_mode_as_u32(stat.st_mode);
    let inode = stat_inode_as_u64(stat.st_ino);
    let hard_links = stat_link_count_as_u64(stat.st_nlink);
    let modified_seconds = stat_seconds_as_i64(stat.st_mtime);
    let changed_seconds = stat_seconds_as_i64(stat.st_ctime);
    let accessed_seconds = stat_seconds_as_i64(stat.st_atime);
    let modified_nanoseconds = i64::try_from(stat.st_mtime_nsec)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
    let changed_nanoseconds = i64::try_from(stat.st_ctime_nsec)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
    let accessed_nanoseconds = i64::try_from(stat.st_atime_nsec)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
    let stat_device_major = rustix::fs::major(stat.st_dev);
    let stat_device_minor = rustix::fs::minor(stat.st_dev);
    if !linux_device_components_match(
        stat_device_major,
        stat_device_minor,
        extended.stx_dev_major,
        extended.stx_dev_minor,
    ) || u32::from(extended.stx_mode) != mode
        || u64::from(extended.stx_nlink) != hard_links
        || extended.stx_uid != stat.st_uid
        || extended.stx_gid != stat.st_gid
        || extended.stx_ino != inode
        || extended.stx_size != size
        || extended.stx_mtime.tv_sec != modified_seconds
        || i64::from(extended.stx_mtime.tv_nsec) != modified_nanoseconds
        || extended.stx_ctime.tv_sec != changed_seconds
        || i64::from(extended.stx_ctime.tv_nsec) != changed_nanoseconds
        || extended.stx_atime.tv_sec != accessed_seconds
        || i64::from(extended.stx_atime.tv_nsec) != accessed_nanoseconds
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ));
    }
    let witness = GenerationRootObjectWitness {
        device: stat_device_as_u64(stat.st_dev),
        inode,
        mode,
        hard_links,
        uid: stat.st_uid,
        gid: stat.st_gid,
        byte_len: size,
        modified_seconds,
        modified_nanoseconds,
        changed_seconds,
        changed_nanoseconds,
        accessed_seconds,
        accessed_nanoseconds,
        mount_identity,
        filesystem: expected_filesystem,
    };
    #[cfg(test)]
    test_boundary(TestBoundary::AfterObjectWitness)?;
    Ok(witness)
}

#[cfg(target_os = "linux")]
const fn linux_device_components_match(
    legacy_device_major: u32,
    legacy_device_minor: u32,
    extended_device_major: u32,
    extended_device_minor: u32,
) -> bool {
    legacy_device_major == extended_device_major && legacy_device_minor == extended_device_minor
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) const fn linux_device_components_match_for_test(
    legacy_device_major: u32,
    legacy_device_minor: u32,
    extended_device_major: u32,
    extended_device_minor: u32,
) -> bool {
    linux_device_components_match(
        legacy_device_major,
        legacy_device_minor,
        extended_device_major,
        extended_device_minor,
    )
}

#[cfg(target_os = "linux")]
fn validate_linux_writable_attributes(
    attributes: rustix::fs::StatxAttributes,
    attributes_mask: rustix::fs::StatxAttributes,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    use rustix::fs::StatxAttributes;

    let required = StatxAttributes::IMMUTABLE | StatxAttributes::APPEND;
    if !attributes_mask.contains(required) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedKernelFeature,
            stage,
        )
        .with_counts(required.bits(), attributes_mask.bits() & required.bits()));
    }
    let observed = attributes & required;
    if observed.is_empty() {
        Ok(())
    } else {
        Err(
            GenerationRootError::new(GenerationRootErrorKind::WriteRestrictedObject, stage)
                .with_counts(0, observed.bits()),
        )
    }
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn validate_linux_writable_attributes_for_test(
    attributes: rustix::fs::StatxAttributes,
    attributes_mask: rustix::fs::StatxAttributes,
) -> GenerationRootResult<()> {
    validate_linux_writable_attributes(
        attributes,
        attributes_mask,
        GenerationRootStage::QualifyFilesystem,
    )
}

#[cfg(target_os = "linux")]
fn linux_statx(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
    request_subvolume: bool,
) -> GenerationRootResult<rustix::fs::Statx> {
    use rustix::fs::{AtFlags, StatxFlags, statx};

    let mut requested = StatxFlags::BASIC_STATS | StatxFlags::MNT_ID;
    if request_subvolume {
        requested |= StatxFlags::from_bits_retain(LINUX_STATX_SUBVOLUME_MASK);
    }
    loop {
        match statx(
            descriptor,
            "",
            AtFlags::EMPTY_PATH | AtFlags::NO_AUTOMOUNT,
            requested,
        ) {
            Ok(extended) => return Ok(extended),
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(
                    if error == Errno::NOSYS
                        || error == Errno::PERM
                        || (request_subvolume && error == Errno::INVAL)
                    {
                        GenerationRootError::new(
                            GenerationRootErrorKind::UnsupportedKernelFeature,
                            stage,
                        )
                        .with_raw_os_error(error.raw_os_error())
                    } else if error == Errno::MFILE
                        || error == Errno::NFILE
                        || error == Errno::NOMEM
                    {
                        GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                            .with_raw_os_error(error.raw_os_error())
                    } else {
                        os_error(stage, error)
                    },
                );
            }
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn linux_filesystem_profile(
    raw_type: rustix::fs::FsWord,
    mount_id: u64,
) -> GenerationRootResult<QualifiedFilesystem> {
    linux_filesystem_profile_at_stage(raw_type, mount_id, GenerationRootStage::QualifyFilesystem)
}

#[cfg(target_os = "linux")]
fn linux_filesystem_profile_at_stage(
    raw_type: rustix::fs::FsWord,
    mount_id: u64,
    stage: GenerationRootStage,
) -> GenerationRootResult<QualifiedFilesystem> {
    let raw = i128::from(raw_type);
    let (expected_name, profile) = if raw == i128::from(libc::EXT4_SUPER_MAGIC) {
        (b"ext4".as_slice(), QualifiedFilesystem::LinuxExt4)
    } else if raw == i128::from(libc::BTRFS_SUPER_MAGIC) {
        (b"btrfs".as_slice(), QualifiedFilesystem::LinuxBtrfs)
    } else {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedFilesystem,
            stage,
        ));
    };
    let observed_name = linux_mountinfo_filesystem(mount_id, stage)?;
    if observed_name.as_slice() != expected_name {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedFilesystem,
            stage,
        ));
    }
    Ok(profile)
}

#[cfg(target_os = "linux")]
fn linux_filesystem_type_matches(
    expected: QualifiedFilesystem,
    raw_type: rustix::fs::FsWord,
) -> bool {
    let raw = i128::from(raw_type);
    match expected {
        QualifiedFilesystem::LinuxExt4 => raw == i128::from(libc::EXT4_SUPER_MAGIC),
        QualifiedFilesystem::LinuxBtrfs => raw == i128::from(libc::BTRFS_SUPER_MAGIC),
        QualifiedFilesystem::AppleApfs => false,
    }
}

#[cfg(target_os = "linux")]
fn linux_filesystem_is_btrfs(raw_type: rustix::fs::FsWord) -> bool {
    i128::from(raw_type) == i128::from(libc::BTRFS_SUPER_MAGIC)
}

#[cfg(target_os = "linux")]
fn linux_filesystem_namespace_identity(
    raw_type: rustix::fs::FsWord,
    extended: &rustix::fs::Statx,
    stage: GenerationRootStage,
) -> GenerationRootResult<[u8; 32]> {
    use rustix::fs::StatxFlags;

    let mask = StatxFlags::from_bits_retain(extended.stx_mask);
    if !mask.contains(StatxFlags::MNT_ID) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedKernelFeature,
            stage,
        ));
    }
    let subvolume_id = if linux_filesystem_is_btrfs(raw_type) {
        let subvolume_mask = StatxFlags::from_bits_retain(LINUX_STATX_SUBVOLUME_MASK);
        if !mask.contains(subvolume_mask) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedKernelFeature,
                stage,
            ));
        }
        // This is the kernel-authenticated value for the already-open
        // descriptor (`AT_EMPTY_PATH`), not a pathname re-resolution.
        // Keeping it as the sole source of truth also avoids imposing
        // consumer-side bindgen and libclang prerequisites for Btrfs.
        Some(extended.stx_subvol)
    } else {
        None
    };
    Ok(linux_filesystem_namespace_digest(
        raw_type,
        extended.stx_mnt_id,
        subvolume_id,
    ))
}

#[cfg(target_os = "linux")]
pub(super) fn validate_root_writable(
    descriptor: &OwnedFd,
    filesystem: QualifiedFilesystem,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    use rustix::fs::{Access, AtFlags, accessat};

    if filesystem != QualifiedFilesystem::LinuxBtrfs {
        return Ok(());
    }
    match accessat(
        descriptor,
        ".",
        Access::WRITE_OK,
        AtFlags::EACCESS | AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(()) => Ok(()),
        Err(error) if error == Errno::ROFS => Err(GenerationRootError::new(
            GenerationRootErrorKind::ReadOnlyFilesystem,
            stage,
        )
        .with_raw_os_error(error.raw_os_error())),
        Err(error) if error == Errno::NOSYS || error == Errno::PERM => Err(
            GenerationRootError::new(GenerationRootErrorKind::UnsupportedKernelFeature, stage)
                .with_raw_os_error(error.raw_os_error()),
        ),
        Err(error) => Err(os_error(stage, error)),
    }
}

#[cfg(target_os = "linux")]
fn linux_mountinfo_filesystem(
    mount_id: u64,
    stage: GenerationRootStage,
) -> GenerationRootResult<Vec<u8>> {
    use rustix::fs::{ResolveFlags, openat2};
    use rustix::io::read;

    let process_id = std::process::id();
    let thread_id = rustix::thread::gettid();
    let path = linux_task_mountinfo_route(process_id, thread_id.as_raw_pid());
    validate_process_identity(process_id, stage)?;
    let (proc_root, _proc_mount_id) = open_verified_proc_root(process_id, stage)?;
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeMountInfoOpen)?;
    let descriptor = openat2(
        &proc_root,
        &path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
        ResolveFlags::BENEATH
            | ResolveFlags::NO_SYMLINKS
            | ResolveFlags::NO_MAGICLINKS
            | ResolveFlags::NO_XDEV,
    )
    .map_err(|error| linux_mountinfo_error(stage, error))?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterMountInfoOpen)?;
    validate_process_identity(process_id, stage)?;
    if rustix::thread::gettid() != thread_id {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ));
    }
    let filesystem_stat =
        fstatfs(&descriptor).map_err(|error| linux_mountinfo_error(stage, error))?;
    if i128::from(filesystem_stat.f_type) != i128::from(libc::PROC_SUPER_MAGIC) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedFilesystem,
            stage,
        ));
    }
    let stat = fstat(&descriptor).map_err(|error| linux_mountinfo_error(stage, error))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedFilesystem,
            stage,
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve(64 * 1024)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
    let mut chunk = [0_u8; 8192];
    loop {
        let requested = mountinfo_read_request(bytes.len());
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeMountInfoRead)?;
        let count = match read(&descriptor, &mut chunk[..requested]) {
            Ok(count) => count,
            Err(error) if error == Errno::INTR => continue,
            Err(error) => return Err(linux_mountinfo_error(stage, error)),
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterMountInfoRead { byte_count: count })?;
        validate_process_identity(process_id, stage)?;
        if rustix::thread::gettid() != thread_id {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ));
        }
        if count == 0 {
            break;
        }
        validate_mountinfo_append(bytes.len(), count, stage)?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    validate_proc_self_binding(
        &proc_root,
        process_id,
        ProcSelfBindingPhase::MountInfoRead,
        stage,
    )?;
    parse_mountinfo_filesystem_at_stage(&bytes, mount_id, stage)
}

#[cfg(target_os = "linux")]
fn linux_mountinfo_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
    let kind = if error == Errno::MFILE || error == Errno::NFILE || error == Errno::NOMEM {
        GenerationRootErrorKind::ResourceLimit
    } else {
        GenerationRootErrorKind::Io
    };
    GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn linux_mountinfo_error_for_test(
    error: Errno,
) -> GenerationRootError {
    linux_mountinfo_error(GenerationRootStage::QualifyFilesystem, error)
}

#[cfg(target_os = "linux")]
fn linux_task_mountinfo_route(process_id: u32, thread_id: i32) -> std::path::PathBuf {
    std::path::PathBuf::from(process_id.to_string())
        .join("task")
        .join(thread_id.to_string())
        .join("mountinfo")
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn linux_task_mountinfo_route_for_test(
    process_id: u32,
    thread_id: i32,
) -> std::path::PathBuf {
    linux_task_mountinfo_route(process_id, thread_id)
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn linux_mountinfo_filesystem_for_test(
    mount_id: u64,
) -> GenerationRootResult<Vec<u8>> {
    linux_mountinfo_filesystem(mount_id, GenerationRootStage::QualifyFilesystem)
}

#[cfg(target_os = "linux")]
fn mountinfo_read_request(stored_bytes: usize) -> usize {
    GENERATION_ROOT_MAX_MOUNTINFO_BYTES
        .saturating_sub(stored_bytes)
        .saturating_add(1)
        .min(8192)
}

#[cfg(target_os = "linux")]
fn validate_mountinfo_append(
    stored_bytes: usize,
    incoming_bytes: usize,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let observed = stored_bytes.saturating_add(incoming_bytes);
    if observed > GENERATION_ROOT_MAX_MOUNTINFO_BYTES {
        Err(
            GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage).with_counts(
                usize_as_u64(GENERATION_ROOT_MAX_MOUNTINFO_BYTES),
                usize_as_u64(observed),
            ),
        )
    } else {
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn validate_mountinfo_bound_for_test(
    stored_bytes: usize,
    incoming_bytes: usize,
) -> (usize, GenerationRootResult<()>) {
    (
        mountinfo_read_request(stored_bytes),
        validate_mountinfo_append(
            stored_bytes,
            incoming_bytes,
            GenerationRootStage::QualifyFilesystem,
        ),
    )
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn parse_mountinfo_filesystem(
    bytes: &[u8],
    mount_id: u64,
) -> GenerationRootResult<Vec<u8>> {
    parse_mountinfo_filesystem_at_stage(bytes, mount_id, GenerationRootStage::QualifyFilesystem)
}

#[cfg(target_os = "linux")]
fn parse_mountinfo_filesystem_at_stage(
    bytes: &[u8],
    mount_id: u64,
    stage: GenerationRootStage,
) -> GenerationRootResult<Vec<u8>> {
    let mut matched = None;
    for line in bytes.split(|byte| *byte == b'\n') {
        let mut fields = line.split(|byte| *byte == b' ');
        let Some(observed_mount_id) = fields.next() else {
            continue;
        };
        if parse_ascii_u64(observed_mount_id) != Some(mount_id) {
            continue;
        }
        let filesystem = loop {
            let field = fields.next().ok_or_else(|| {
                GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
            })?;
            if field == b"-" {
                break fields.next().ok_or_else(|| {
                    GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
                })?;
            }
        };
        let mount_source = fields.next().ok_or_else(|| {
            GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
        })?;
        let super_options = fields.next().ok_or_else(|| {
            GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
        })?;
        if filesystem.is_empty() || mount_source.is_empty() || super_options.is_empty() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedFilesystem,
                stage,
            ));
        }
        let mut owned_filesystem = Vec::new();
        owned_filesystem
            .try_reserve_exact(filesystem.len())
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
        owned_filesystem.extend_from_slice(filesystem);
        if matched.replace(owned_filesystem).is_some() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedFilesystem,
                stage,
            ));
        }
    }
    matched.ok_or_else(|| {
        GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
    })
}

#[cfg(target_os = "linux")]
fn parse_ascii_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.is_empty() {
        return None;
    }
    let mut value = 0_u64;
    for byte in bytes {
        let digit = byte.checked_sub(b'0')?;
        if digit > 9 {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(u64::from(digit))?;
    }
    Some(value)
}

#[cfg(target_os = "linux")]
pub(in crate::generation_root) fn linux_filesystem_namespace_digest(
    raw_type: rustix::fs::FsWord,
    mount_id: u64,
    subvolume_id: Option<u64>,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"frankensearch.generation-root.linux-namespace.v2");
    digest.update(i128::from(raw_type).to_le_bytes());
    digest.update(mount_id.to_le_bytes());
    match subvolume_id {
        Some(subvolume_id) => {
            digest.update([1]);
            digest.update(subvolume_id.to_le_bytes());
        }
        None => digest.update([0]),
    }
    digest.finalize().into()
}

#[cfg(target_os = "linux")]
pub(super) const fn stat_device_as_u64(device: u64) -> u64 {
    device
}

#[cfg(target_os = "linux")]
fn linux_component_open_error(
    parent: &OwnedFd,
    component: &OsStr,
    stage: GenerationRootStage,
    index: usize,
    error: Errno,
) -> GenerationRootError {
    use rustix::fs::{AtFlags, statat};

    if error == Errno::NOTDIR
        && statat(parent, component, AtFlags::SYMLINK_NOFOLLOW)
            .is_ok_and(|stat| FileType::from_raw_mode(stat.st_mode) == FileType::Symlink)
    {
        return GenerationRootError::new(GenerationRootErrorKind::SymbolicLink, stage)
            .at_component(index)
            .with_raw_os_error(error.raw_os_error());
    }
    linux_open_error(stage, error).at_component(index)
}

#[cfg(target_os = "linux")]
fn linux_open_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
    let kind = if error == Errno::LOOP {
        GenerationRootErrorKind::SymbolicLink
    } else if error == Errno::NOTDIR {
        GenerationRootErrorKind::NotDirectory
    } else if error == Errno::XDEV {
        GenerationRootErrorKind::CrossDevice
    } else if error == Errno::NOSYS || error == Errno::PERM {
        GenerationRootErrorKind::UnsupportedKernelFeature
    } else if error == Errno::MFILE || error == Errno::NFILE || error == Errno::NOMEM {
        GenerationRootErrorKind::ResourceLimit
    } else {
        GenerationRootErrorKind::Io
    };
    GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn linux_open_error_for_test(error: Errno) -> GenerationRootError {
    linux_open_error(GenerationRootStage::OpenRegularFile, error)
}

#[cfg(target_os = "linux")]
pub(super) fn inspect_acl_if_required(
    descriptor: &OwnedFd,
    witness: GenerationRootObjectWitness,
    _stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let before = object_witness(
        descriptor,
        witness.filesystem,
        witness.mount_identity,
        GenerationRootStage::InspectAcl,
    )?;
    if !before.mutation_stable_eq(witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::InspectAcl,
        ));
    }
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeAclRead)?;
    let presence = crate::fd_acl::extended_acl_presence(descriptor.as_fd()).map_err(|error| {
        GenerationRootError::new(
            GenerationRootErrorKind::AclRejected,
            GenerationRootStage::InspectAcl,
        )
        .with_raw_os_error(error.raw_os_error().unwrap_or(libc::EIO))
    })?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterAclRead)?;
    match presence {
        crate::fd_acl::ExtendedAclPresence::Absent => {}
        crate::fd_acl::ExtendedAclPresence::Present => {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::AclRejected,
                GenerationRootStage::InspectAcl,
            ));
        }
    }
    let after = object_witness(
        descriptor,
        witness.filesystem,
        witness.mount_identity,
        GenerationRootStage::InspectAcl,
    )?;
    if !witness.mutation_stable_eq(after) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::InspectAcl,
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub(super) fn sync_file_descriptor(descriptor: &OwnedFd) -> GenerationRootResult<()> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFileSync)?;
    rustix::fs::fsync(descriptor).map_err(|error| {
        GenerationRootError::new(
            GenerationRootErrorKind::DurabilityUnavailable,
            GenerationRootStage::SyncRegularFile,
        )
        .with_raw_os_error(error.raw_os_error())
    })?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFileSync)?;
    Ok(())
}

#[cfg(target_os = "linux")]
pub(super) fn sync_directory_descriptor(descriptor: &OwnedFd) -> GenerationRootResult<()> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeDirectorySync)?;
    rustix::fs::fsync(descriptor).map_err(|error| {
        GenerationRootError::new(
            GenerationRootErrorKind::DurabilityUnavailable,
            GenerationRootStage::SyncDirectory,
        )
        .with_raw_os_error(error.raw_os_error())
    })?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterDirectorySync)?;
    Ok(())
}
