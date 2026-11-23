use super::super::{
    GenerationRootError, GenerationRootErrorKind, GenerationRootObjectWitness,
    GenerationRootResult, GenerationRootStage, QualifiedFilesystem,
};
use super::types::{AbsoluteRoute, OpenedRoute};
#[cfg(test)]
use super::types::{TestBoundary, test_boundary};
use super::unix_common::{
    basic_identity, os_error, qualify_absolute_ancestor, stat_inode_as_u64, stat_link_count_as_u64,
    stat_mode_as_u32, stat_seconds_as_i64, validate_private_directory,
    verify_exact_directory_entry,
};
use rustix::fd::{AsFd, OwnedFd};
use rustix::fs::{FileType, Mode, OFlags, fstat, fstatfs, fstatvfs, openat};
use rustix::io::Errno;
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::path::Path;

#[cfg(target_os = "macos")]
pub(super) fn open_absolute_route(
    route: &AbsoluteRoute,
    stage: GenerationRootStage,
) -> GenerationRootResult<OpenedRoute> {
    use rustix::fs::CWD;

    let flags =
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    let mut descriptor = openat(CWD, Path::new("/"), flags, Mode::empty())
        .map_err(|error| macos_open_error(stage, error))?;
    let mut route_identities = Vec::new();
    route_identities
        .try_reserve(route.components.len() + 1)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
    route_identities.push(qualify_absolute_ancestor(&descriptor, stage, None)?);
    for (index, component) in route.components.iter().enumerate() {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRootComponentOpen { index })?;
        verify_exact_directory_entry(&descriptor, component, stage, index)?;
        let next = openat(&descriptor, component, flags, Mode::empty()).map_err(|error| {
            macos_component_open_error(&descriptor, component, stage, index, error)
        })?;
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

#[cfg(target_os = "macos")]
pub(super) fn open_relative_directory_component(
    parent: &OwnedFd,
    component: &OsStr,
    index: usize,
    root: GenerationRootObjectWitness,
    stage: GenerationRootStage,
) -> GenerationRootResult<OwnedFd> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeRelativeDirectoryOpen { index })?;
    verify_exact_directory_entry(parent, component, stage, index)?;
    let descriptor = openat(
        parent,
        component,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(|error| macos_component_open_error(parent, component, stage, index, error))?;
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

#[cfg(target_os = "macos")]
pub(super) fn route_mount_identity(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
) -> GenerationRootResult<[u8; 32]> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeRouteMountIdentity)?;
    let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    let identity = macos_mount_identity(&filesystem_stat, stat_device_as_u64(stat.st_dev));
    #[cfg(test)]
    test_boundary(TestBoundary::AfterRouteMountIdentity)?;
    Ok(identity)
}

#[cfg(target_os = "macos")]
pub(super) fn qualify_filesystem(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
) -> GenerationRootResult<(QualifiedFilesystem, [u8; 32])> {
    use rustix::fs::StatVfsMountFlags;

    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFilesystemQualification)?;
    let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
    if macos_fixed_name(&filesystem_stat.f_fstypename) != b"apfs" {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedFilesystem,
            stage,
        ));
    }
    let flags = i64::from(filesystem_stat.f_flags);
    if flags & i64::from(libc::MNT_LOCAL) == 0 {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedFilesystem,
            stage,
        ));
    }
    if flags & i64::from(libc::MNT_RDONLY) != 0
        || flags & i64::from(libc::MNT_IGNORE_OWNERSHIP) != 0
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ReadOnlyFilesystem,
            stage,
        ));
    }
    let mount_stat = fstatvfs(descriptor).map_err(|error| os_error(stage, error))?;
    if mount_stat.f_flag.contains(StatVfsMountFlags::RDONLY) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ReadOnlyFilesystem,
            stage,
        ));
    }
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    let qualified = (
        QualifiedFilesystem::AppleApfs,
        macos_mount_identity(&filesystem_stat, stat_device_as_u64(stat.st_dev)),
    );
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFilesystemQualification)?;
    Ok(qualified)
}

#[cfg(target_os = "macos")]
pub(super) fn validate_root_writable(
    descriptor: &OwnedFd,
    filesystem: QualifiedFilesystem,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let (observed, _) = qualify_filesystem(descriptor, stage)?;
    if observed != filesystem {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::CrossDevice,
            stage,
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn object_witness(
    descriptor: &OwnedFd,
    expected_filesystem: QualifiedFilesystem,
    expected_mount_identity: [u8; 32],
    stage: GenerationRootStage,
) -> GenerationRootResult<GenerationRootObjectWitness> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeObjectWitness)?;
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    let (filesystem, mount_identity) = qualify_filesystem(descriptor, stage)?;
    if filesystem != expected_filesystem || mount_identity != expected_mount_identity {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::CrossDevice,
            stage,
        ));
    }
    validate_macos_object_flags(stat.st_flags, stage)?;
    let size = u64::try_from(stat.st_size)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
    let witness = GenerationRootObjectWitness {
        device: stat_device_as_u64(stat.st_dev),
        inode: stat_inode_as_u64(stat.st_ino),
        mode: stat_mode_as_u32(stat.st_mode),
        hard_links: stat_link_count_as_u64(stat.st_nlink),
        uid: stat.st_uid,
        gid: stat.st_gid,
        byte_len: size,
        modified_seconds: stat_seconds_as_i64(stat.st_mtime),
        modified_nanoseconds: stat.st_mtime_nsec,
        changed_seconds: stat_seconds_as_i64(stat.st_ctime),
        changed_nanoseconds: stat.st_ctime_nsec,
        accessed_seconds: stat_seconds_as_i64(stat.st_atime),
        accessed_nanoseconds: stat.st_atime_nsec,
        mount_identity,
        filesystem,
    };
    #[cfg(test)]
    test_boundary(TestBoundary::AfterObjectWitness)?;
    Ok(witness)
}

#[cfg(target_os = "macos")]
fn validate_macos_object_flags(flags: u32, stage: GenerationRootStage) -> GenerationRootResult<()> {
    if flags == 0 {
        Ok(())
    } else {
        Err(
            GenerationRootError::new(GenerationRootErrorKind::UnsupportedObjectFlags, stage)
                .with_counts(0, u64::from(flags)),
        )
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
pub(in crate::generation_root) fn validate_macos_object_flags_for_test(
    flags: u32,
) -> GenerationRootResult<()> {
    validate_macos_object_flags(flags, GenerationRootStage::OpenRegularFile)
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn macos_fixed_name<const N: usize>(value: &[libc::c_char; N]) -> Vec<u8> {
    value
        .iter()
        .map(|character| character.to_ne_bytes()[0])
        .take_while(|byte| *byte != 0)
        .collect()
}

#[cfg(target_os = "macos")]
fn macos_mount_identity(stat: &rustix::fs::StatFs, device: u64) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"frankensearch.generation-root.macos-mount.v1");
    digest.update(device.to_le_bytes());
    for character in &stat.f_fstypename {
        digest.update(character.to_ne_bytes());
    }
    for character in &stat.f_mntonname {
        digest.update(character.to_ne_bytes());
    }
    for character in &stat.f_mntfromname {
        digest.update(character.to_ne_bytes());
    }
    digest.finalize().into()
}

#[cfg(target_os = "macos")]
pub(super) fn stat_device_as_u64(device: i32) -> u64 {
    u64::from_ne_bytes(i64::from(device).to_ne_bytes())
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
pub(in crate::generation_root) fn stat_device_as_u64_for_test(device: i32) -> u64 {
    stat_device_as_u64(device)
}

#[cfg(target_os = "macos")]
pub(super) fn macos_open_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
    let kind = if error == Errno::LOOP {
        GenerationRootErrorKind::SymbolicLink
    } else if error == Errno::NOTDIR {
        GenerationRootErrorKind::NotDirectory
    } else {
        GenerationRootErrorKind::Io
    };
    GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
}

#[cfg(target_os = "macos")]
fn macos_component_open_error(
    parent: &OwnedFd,
    component: &OsStr,
    stage: GenerationRootStage,
    index: usize,
    error: Errno,
) -> GenerationRootError {
    use rustix::fs::{AtFlags, statat};

    if matches!(error, Errno::NOTDIR)
        && statat(parent, component, AtFlags::SYMLINK_NOFOLLOW)
            .is_ok_and(|stat| FileType::from_raw_mode(stat.st_mode) == FileType::Symlink)
    {
        return GenerationRootError::new(GenerationRootErrorKind::SymbolicLink, stage)
            .at_component(index)
            .with_raw_os_error(error.raw_os_error());
    }
    macos_open_error(stage, error).at_component(index)
}

#[cfg(target_os = "macos")]
pub(super) fn sync_file_descriptor(descriptor: &OwnedFd) -> GenerationRootResult<()> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFileSync)?;
    rustix::fs::fcntl_fullfsync(descriptor).map_err(|error| {
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

#[cfg(target_os = "macos")]
pub(super) fn sync_directory_descriptor(descriptor: &OwnedFd) -> GenerationRootResult<()> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeDirectorySync)?;
    rustix::fs::fcntl_fullfsync(descriptor).map_err(|error| {
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
