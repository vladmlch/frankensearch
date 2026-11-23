use super::super::{
    GENERATION_ROOT_DIRECTORY_MODE, GENERATION_ROOT_MAX_COMPONENT_BYTES,
    GENERATION_ROOT_MAX_COMPONENTS, GENERATION_ROOT_MAX_DIRECTORY_ENTRIES,
    GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES, GENERATION_ROOT_MAX_ROUTE_BYTES,
    GenerationFileExpectation, GenerationRootEntry, GenerationRootEntryKind, GenerationRootError,
    GenerationRootErrorKind, GenerationRootObjectWitness, GenerationRootResult,
    GenerationRootStage,
};
#[cfg(target_os = "linux")]
use super::linux::{
    inspect_acl_if_required, object_witness, open_regular_component,
    open_relative_directory_component, probe_regular_component, route_mount_identity,
    stat_device_as_u64,
};
#[cfg(target_os = "macos")]
use super::macos::{
    inspect_acl_if_required, macos_open_error, object_witness, open_relative_directory_component,
    route_mount_identity, stat_device_as_u64,
};
use super::ops::{revalidate_root, validate_lock_process};
use super::types::{
    AbsoluteAncestorSecurityWitness, BasicObjectIdentity, ControlHandle, ExactContentWitness,
    FileHandle, LockHandle, MODE_PERMISSIONS_MASK, RelativePath, RootHandle,
};
#[cfg(test)]
use super::types::{TestBoundary, test_boundary};
use rustix::fd::{AsFd, OwnedFd};
#[cfg(target_os = "macos")]
use rustix::fs::OFlags;
use rustix::fs::{FileType, fstat};
use rustix::io::{Errno, pread};
use sha2::{Digest, Sha256};
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::sync::Arc;

#[cfg(target_os = "linux")]
pub(super) fn revalidate_anchor_route(control: &ControlHandle) -> GenerationRootResult<()> {
    revalidate_root(&control.root)?;
    let retained = object_witness(
        &control.descriptor,
        control.witness.filesystem,
        control.witness.mount_identity,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    validate_private_regular_file(
        &control.descriptor,
        retained,
        control.expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !retained.same_control_anchor_identity(control.witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    let reopened = open_relative_probe(
        &control.root,
        &control.path,
        control.expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !reopened
        .witness
        .same_control_anchor_identity(control.witness)
        || reopened.component_identities != control.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    revalidate_root(&control.root)
}

#[cfg(target_os = "macos")]
pub(super) fn revalidate_anchor_route(control: &ControlHandle) -> GenerationRootResult<()> {
    revalidate_root(&control.root)?;
    validate_preopened_regular_descriptor_status(
        &control.descriptor,
        OFlags::RDWR,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    let retained = object_witness(
        &control.descriptor,
        control.witness.filesystem,
        control.witness.mount_identity,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    validate_private_regular_file(
        &control.descriptor,
        retained,
        control.expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !retained.same_control_anchor_identity(control.witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    let observed_components = bind_preopened_route(
        &control.root,
        &control.path,
        retained,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if observed_components.as_ref() != control.component_identities.as_ref() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    revalidate_root(&control.root)
}

#[cfg(target_os = "linux")]
#[cfg(test)]
pub(super) fn revalidate_control_route(control: &ControlHandle) -> GenerationRootResult<()> {
    revalidate_root(&control.root)?;
    let retained = object_witness(
        &control.descriptor,
        control.witness.filesystem,
        control.witness.mount_identity,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !retained.mutation_stable_eq(control.witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    let reopened = open_relative_probe(
        &control.root,
        &control.path,
        control.expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !reopened.witness.mutation_stable_eq(control.witness)
        || reopened.component_identities != control.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    revalidate_root(&control.root)
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
pub(super) fn revalidate_control_route(control: &ControlHandle) -> GenerationRootResult<()> {
    revalidate_preopened_file_route(
        &control.root,
        &control.path,
        control.expectation,
        &control.descriptor,
        control.witness,
        &control.component_identities,
        OFlags::RDWR,
        GenerationRootStage::RevalidateRegularFile,
    )
}

#[cfg(target_os = "linux")]
pub(super) fn revalidate_lock_route(lock: &LockHandle) -> GenerationRootResult<()> {
    validate_lock_process(lock, GenerationRootStage::RevalidateRegularFile)?;
    revalidate_root(&lock.root)?;
    let retained = object_witness(
        &lock.descriptor,
        lock.witness.filesystem,
        lock.witness.mount_identity,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !retained.mutation_stable_eq(lock.witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    let reopened = open_relative_probe(
        &lock.root,
        &lock.path,
        lock.expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !reopened.witness.mutation_stable_eq(lock.witness)
        || reopened.component_identities != lock.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    revalidate_root(&lock.root)
}

#[cfg(target_os = "macos")]
pub(super) fn revalidate_lock_route(lock: &LockHandle) -> GenerationRootResult<()> {
    validate_lock_process(lock, GenerationRootStage::RevalidateRegularFile)?;
    revalidate_preopened_file_route(
        &lock.root,
        &lock.path,
        lock.expectation,
        &lock.descriptor,
        lock.witness,
        &lock.component_identities,
        lock.descriptor_status_flags,
        GenerationRootStage::RevalidateRegularFile,
    )
}

pub(super) fn validate_lock_content(
    lock: &LockHandle,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let (observed, _) = read_exact_sha256(&lock.descriptor, lock.witness, lock.expectation, stage)?;
    validate_bound_content(observed, lock.content_witness, stage)
}

pub(super) struct OpenedFile {
    pub(super) descriptor: OwnedFd,
    pub(super) witness: GenerationRootObjectWitness,
    pub(super) component_identities: Arc<[BasicObjectIdentity]>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) enum PreopenedAccess {
    ReadOnly,
    ReadWrite,
}

#[cfg(target_os = "macos")]
pub(super) fn qualify_preopened_file(
    root: &RootHandle,
    path: &RelativePath,
    descriptor: OwnedFd,
    expectation: GenerationFileExpectation,
    access: PreopenedAccess,
    stage: GenerationRootStage,
) -> GenerationRootResult<OpenedFile> {
    #[cfg(test)]
    test_boundary(TestBoundary::BeforePreopenedQualification)?;
    validate_preopened_regular_descriptor(&descriptor, access, stage)?;
    let witness = object_witness(
        &descriptor,
        root.state.witness.filesystem,
        root.state.witness.mount_identity,
        stage,
    )?;
    validate_private_regular_file(&descriptor, witness, expectation, stage)?;
    let component_identities = bind_preopened_route(root, path, witness, stage)?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterPreopenedQualification)?;
    Ok(OpenedFile {
        descriptor,
        witness,
        component_identities,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn validate_preopened_regular_descriptor(
    descriptor: &OwnedFd,
    expected_access: PreopenedAccess,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let expected_status_flags = match expected_access {
        PreopenedAccess::ReadOnly => OFlags::RDONLY,
        PreopenedAccess::ReadWrite => OFlags::RDWR,
    };
    validate_preopened_regular_descriptor_status(descriptor, expected_status_flags, stage)
}

#[cfg(target_os = "macos")]
fn validate_preopened_regular_descriptor_status(
    descriptor: &OwnedFd,
    expected_status_flags: OFlags,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::NotRegularFile,
            stage,
        ));
    }
    let observed_flags =
        rustix::fs::fcntl_getfl(descriptor).map_err(|error| os_error(stage, error))?;
    if observed_flags != expected_status_flags {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::InvalidDescriptorAccess,
            stage,
        ));
    }
    let descriptor_flags =
        rustix::io::fcntl_getfd(descriptor).map_err(|error| os_error(stage, error))?;
    if !descriptor_flags.contains(rustix::io::FdFlags::CLOEXEC) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::CloseOnExecRequired,
            stage,
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn bind_preopened_route(
    root: &RootHandle,
    path: &RelativePath,
    witness: GenerationRootObjectWitness,
    stage: GenerationRootStage,
) -> GenerationRootResult<Arc<[BasicObjectIdentity]>> {
    use rustix::fs::{AtFlags, statat};

    let (final_name, ancestors) = path.components.split_last().ok_or_else(|| {
        GenerationRootError::new(
            GenerationRootErrorKind::InvalidRoute,
            GenerationRootStage::ParseRelativeRoute,
        )
    })?;
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeRootDescriptorDuplicate)?;
    let mut parent = rustix::io::fcntl_dupfd_cloexec(&root.state.descriptor, 0)
        .map_err(|error| os_error(stage, error))?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterRootDescriptorDuplicate)?;
    let mut component_identities = Vec::new();
    component_identities
        .try_reserve(path.components.len())
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
    for (index, component) in ancestors.iter().enumerate() {
        parent = open_relative_directory_component(
            &parent,
            component,
            index,
            root.state.witness,
            stage,
        )?;
        component_identities.push(basic_identity(&parent, stage)?);
    }
    let index = ancestors.len();
    verify_exact_directory_entry(&parent, final_name, stage, index)?;
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFinalRouteStat)?;
    let route_stat = statat(&parent, final_name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| macos_open_error(stage, error).at_component(index))?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFinalRouteStat)?;
    verify_exact_directory_entry(&parent, final_name, stage, index)?;
    let route_file_type = FileType::from_raw_mode(route_stat.st_mode);
    if route_file_type == FileType::Symlink {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::SymbolicLink, stage)
                .at_component(index),
        );
    }
    if route_file_type != FileType::RegularFile {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::NotRegularFile, stage)
                .at_component(index),
        );
    }
    if !macos_route_stat_matches_witness(&route_stat, witness) {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index),
        );
    }
    component_identities.push(BasicObjectIdentity {
        device: witness.device,
        inode: witness.inode,
        mode: witness.mode,
        uid: witness.uid,
        gid: witness.gid,
        mount_identity: witness.mount_identity,
    });
    Ok(Arc::from(component_identities))
}

#[cfg(target_os = "macos")]
fn macos_route_stat_matches_witness(
    stat: &rustix::fs::Stat,
    witness: GenerationRootObjectWitness,
) -> bool {
    let Ok(byte_len) = u64::try_from(stat.st_size) else {
        return false;
    };
    stat.st_flags == 0
        && stat_device_as_u64(stat.st_dev) == witness.device
        && stat_inode_as_u64(stat.st_ino) == witness.inode
        && stat_mode_as_u32(stat.st_mode) == witness.mode
        && stat_link_count_as_u64(stat.st_nlink) == witness.hard_links
        && stat.st_uid == witness.uid
        && stat.st_gid == witness.gid
        && byte_len == witness.byte_len
        && stat_seconds_as_i64(stat.st_mtime) == witness.modified_seconds
        && stat.st_mtime_nsec == witness.modified_nanoseconds
        && stat_seconds_as_i64(stat.st_ctime) == witness.changed_seconds
        && stat.st_ctime_nsec == witness.changed_nanoseconds
}

#[cfg(target_os = "macos")]
pub(super) fn revalidate_preopened_file_route(
    root: &RootHandle,
    path: &RelativePath,
    expectation: GenerationFileExpectation,
    descriptor: &OwnedFd,
    expected: GenerationRootObjectWitness,
    component_identities: &[BasicObjectIdentity],
    expected_status_flags: OFlags,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    revalidate_root(root)?;
    validate_preopened_regular_descriptor_status(descriptor, expected_status_flags, stage)?;
    let retained = object_witness(
        descriptor,
        expected.filesystem,
        expected.mount_identity,
        stage,
    )?;
    validate_private_regular_file(descriptor, retained, expectation, stage)?;
    if !retained.mutation_stable_eq(expected) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ));
    }
    let observed_components = bind_preopened_route(root, path, retained, stage)?;
    if observed_components.as_ref() != component_identities {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ));
    }
    revalidate_root(root)
}

#[cfg(target_os = "linux")]
pub(super) fn revalidate_file_route(
    file: &FileHandle,
    expected: GenerationRootObjectWitness,
) -> GenerationRootResult<()> {
    revalidate_root(&file.root)?;
    let retained = object_witness(
        &file.descriptor,
        expected.filesystem,
        expected.mount_identity,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !retained.mutation_stable_eq(expected) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    let reopened = open_relative_probe(
        &file.root,
        &file.path,
        file.expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !reopened.witness.mutation_stable_eq(expected)
        || reopened.component_identities != file.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    revalidate_root(&file.root)
}

#[cfg(target_os = "macos")]
pub(super) fn revalidate_file_route(
    file: &FileHandle,
    expected: GenerationRootObjectWitness,
) -> GenerationRootResult<()> {
    revalidate_preopened_file_route(
        &file.root,
        &file.path,
        file.expectation,
        &file.descriptor,
        expected,
        &file.component_identities,
        OFlags::RDONLY,
        GenerationRootStage::RevalidateRegularFile,
    )
}

#[cfg(target_os = "linux")]
pub(super) fn open_relative_file(
    root: &RootHandle,
    path: &RelativePath,
    expectation: GenerationFileExpectation,
    writable: bool,
    directory_stage: GenerationRootStage,
    regular_stage: GenerationRootStage,
) -> GenerationRootResult<OpenedFile> {
    let (final_name, ancestors) = path.components.split_last().ok_or_else(|| {
        GenerationRootError::new(
            GenerationRootErrorKind::InvalidRoute,
            GenerationRootStage::ParseRelativeRoute,
        )
    })?;
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeRootDescriptorDuplicate)?;
    let mut parent = rustix::io::fcntl_dupfd_cloexec(&root.state.descriptor, 0)
        .map_err(|error| os_error(directory_stage, error))?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterRootDescriptorDuplicate)?;
    let mut component_identities = Vec::new();
    component_identities
        .try_reserve(path.components.len())
        .map_err(|_| {
            GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, directory_stage)
        })?;
    for (index, component) in ancestors.iter().enumerate() {
        parent = open_relative_directory_component(
            &parent,
            component,
            index,
            root.state.witness,
            directory_stage,
        )?;
        component_identities.push(basic_identity(&parent, directory_stage)?);
    }
    let descriptor = open_regular_component(
        &parent,
        final_name,
        ancestors.len(),
        root.state.witness,
        expectation,
        writable,
        root.state.creator_process_id,
        regular_stage,
    )?;
    let witness = object_witness(
        &descriptor,
        root.state.witness.filesystem,
        root.state.witness.mount_identity,
        regular_stage,
    )?;
    validate_private_regular_file(&descriptor, witness, expectation, regular_stage)?;
    component_identities.push(basic_identity(&descriptor, regular_stage)?);
    #[cfg(test)]
    test_boundary(TestBoundary::AfterQualifiedFileOpen)?;
    Ok(OpenedFile {
        descriptor,
        witness,
        component_identities: Arc::from(component_identities),
    })
}

#[cfg(target_os = "linux")]
pub(super) fn open_relative_probe(
    root: &RootHandle,
    path: &RelativePath,
    expectation: GenerationFileExpectation,
    stage: GenerationRootStage,
) -> GenerationRootResult<OpenedFile> {
    let (final_name, ancestors) = path.components.split_last().ok_or_else(|| {
        GenerationRootError::new(
            GenerationRootErrorKind::InvalidRoute,
            GenerationRootStage::ParseRelativeRoute,
        )
    })?;
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeRootDescriptorDuplicate)?;
    let mut parent = rustix::io::fcntl_dupfd_cloexec(&root.state.descriptor, 0)
        .map_err(|error| os_error(stage, error))?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterRootDescriptorDuplicate)?;
    let mut component_identities = Vec::new();
    component_identities
        .try_reserve(path.components.len())
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
    for (index, component) in ancestors.iter().enumerate() {
        parent = open_relative_directory_component(
            &parent,
            component,
            index,
            root.state.witness,
            stage,
        )?;
        component_identities.push(basic_identity(&parent, stage)?);
    }
    let index = ancestors.len();
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFinalRouteProbe { index })?;
    let (descriptor, witness) = probe_regular_component(
        &parent,
        final_name,
        index,
        root.state.witness,
        expectation,
        stage,
    )?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFinalRouteProbe { index })?;
    component_identities.push(basic_identity(&descriptor, stage)?);
    Ok(OpenedFile {
        descriptor,
        witness,
        component_identities: Arc::from(component_identities),
    })
}

pub(super) fn read_exact_owned(
    descriptor: &OwnedFd,
    before: GenerationRootObjectWitness,
    expectation: GenerationFileExpectation,
) -> GenerationRootResult<(Vec<u8>, GenerationRootObjectWitness)> {
    let max_byte_len = expectation.max_byte_len();
    if expectation.byte_len > max_byte_len {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ResourceLimit,
            GenerationRootStage::ReadRegularFile,
        )
        .with_counts(max_byte_len, expectation.byte_len));
    }
    let byte_len = usize::try_from(expectation.byte_len).map_err(|_| {
        GenerationRootError::new(
            GenerationRootErrorKind::ResourceLimit,
            GenerationRootStage::ReadRegularFile,
        )
        .with_counts(usize_as_u64(usize::MAX), expectation.byte_len)
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(byte_len).map_err(|_| {
        GenerationRootError::new(
            GenerationRootErrorKind::ResourceLimit,
            GenerationRootStage::ReadRegularFile,
        )
        .with_counts(max_byte_len, expectation.byte_len)
    })?;
    bytes.resize(byte_len, 0);
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let file_offset = usize_as_u64(offset);
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRead {
            offset: file_offset,
        })?;
        match pread(descriptor, &mut bytes[offset..], file_offset) {
            Ok(0) => {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::SizeMismatch,
                    GenerationRootStage::ReadRegularFile,
                )
                .with_counts(expectation.byte_len, file_offset));
            }
            Ok(read) => {
                #[cfg(test)]
                test_boundary(TestBoundary::AfterRead {
                    offset: file_offset,
                    byte_count: read,
                })?;
                offset += read;
            }
            Err(error) if error == Errno::INTR => {}
            Err(error) => {
                return Err(os_error(GenerationRootStage::ReadRegularFile, error));
            }
        }
    }
    let mut probe = [0_u8; 1];
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeTrailingByteProbe)?;
    let probe_read = pread(descriptor, &mut probe, expectation.byte_len)
        .map_err(|error| os_error(GenerationRootStage::ReadRegularFile, error))?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterTrailingByteProbe {
        byte_count: probe_read,
    })?;
    if probe_read != 0 {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::SizeMismatch,
            GenerationRootStage::ReadRegularFile,
        )
        .with_counts(expectation.byte_len, expectation.byte_len.saturating_add(1)));
    }
    #[cfg(test)]
    test_boundary(TestBoundary::AfterExactRead)?;
    let after = object_witness(
        descriptor,
        before.filesystem,
        before.mount_identity,
        GenerationRootStage::ReadRegularFile,
    )?;
    if !before.mutation_stable_eq(after) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::ReadRegularFile,
        ));
    }
    Ok((bytes, after))
}

pub(super) fn read_exact_sha256(
    descriptor: &OwnedFd,
    before: GenerationRootObjectWitness,
    expectation: GenerationFileExpectation,
    stage: GenerationRootStage,
) -> GenerationRootResult<(ExactContentWitness, GenerationRootObjectWitness)> {
    const HASH_BUFFER_BYTES: usize = 64 * 1024;

    let max_byte_len = expectation.max_byte_len();
    if expectation.byte_len > max_byte_len {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                .with_counts(max_byte_len, expectation.byte_len),
        );
    }
    let mut digest = Sha256::new();
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(HASH_BUFFER_BYTES)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
    buffer.resize(HASH_BUFFER_BYTES, 0_u8);
    let mut offset = 0_u64;
    while offset < expectation.byte_len {
        let remaining = expectation.byte_len.saturating_sub(offset);
        let requested = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRead { offset })?;
        match pread(descriptor, &mut buffer[..requested], offset) {
            Ok(0) => {
                return Err(
                    GenerationRootError::new(GenerationRootErrorKind::SizeMismatch, stage)
                        .with_counts(expectation.byte_len, offset),
                );
            }
            Ok(read) => {
                #[cfg(test)]
                test_boundary(TestBoundary::AfterRead {
                    offset,
                    byte_count: read,
                })?;
                digest.update(&buffer[..read]);
                offset = offset.saturating_add(usize_as_u64(read));
            }
            Err(error) if error == Errno::INTR => {}
            Err(error) => return Err(os_error(stage, error)),
        }
    }
    let mut probe = [0_u8; 1];
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeTrailingByteProbe)?;
    let probe_read = pread(descriptor, &mut probe, expectation.byte_len)
        .map_err(|error| os_error(stage, error))?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterTrailingByteProbe {
        byte_count: probe_read,
    })?;
    if probe_read != 0 {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::SizeMismatch, stage)
                .with_counts(expectation.byte_len, expectation.byte_len.saturating_add(1)),
        );
    }
    #[cfg(test)]
    test_boundary(TestBoundary::AfterExactRead)?;
    let after = object_witness(descriptor, before.filesystem, before.mount_identity, stage)?;
    if !before.mutation_stable_eq(after) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ));
    }
    Ok((
        ExactContentWitness {
            sha256: digest.finalize().into(),
        },
        after,
    ))
}

pub(super) fn validate_bound_content(
    observed: ExactContentWitness,
    expected: ExactContentWitness,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    if observed == expected {
        Ok(())
    } else {
        Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ))
    }
}

pub(super) fn validate_expected_sha256(
    observed: ExactContentWitness,
    expectation: GenerationFileExpectation,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    if expectation
        .sha256
        .is_some_and(|expected| expected != observed.sha256)
    {
        Err(GenerationRootError::new(
            GenerationRootErrorKind::HashMismatch,
            stage,
        ))
    } else {
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn validate_bound_control_content_for_test(
    observed: [u8; 32],
    expected: [u8; 32],
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    validate_bound_content(
        ExactContentWitness { sha256: observed },
        ExactContentWitness { sha256: expected },
        stage,
    )
}

pub(super) fn parse_route_bytes(
    value: &OsStr,
    absolute: bool,
    stage: GenerationRootStage,
) -> GenerationRootResult<Vec<OsString>> {
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > GENERATION_ROOT_MAX_ROUTE_BYTES || bytes.contains(&0) {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::InvalidRoute, stage).with_counts(
                usize_as_u64(GENERATION_ROOT_MAX_ROUTE_BYTES),
                usize_as_u64(bytes.len()),
            ),
        );
    }
    if absolute != bytes.starts_with(b"/") {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::InvalidRoute,
            stage,
        ));
    }
    let body = if absolute { &bytes[1..] } else { bytes };
    if body.is_empty() || body.starts_with(b"/") || body.ends_with(b"/") {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::InvalidRoute,
            stage,
        ));
    }
    let mut components = Vec::new();
    components
        .try_reserve(GENERATION_ROOT_MAX_COMPONENTS)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
    for (index, component) in body.split(|byte| *byte == b'/').enumerate() {
        if component.is_empty()
            || component == b"."
            || component == b".."
            || component.len() > GENERATION_ROOT_MAX_COMPONENT_BYTES
        {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::InvalidRoute, stage)
                    .at_component(index)
                    .with_counts(
                        usize_as_u64(GENERATION_ROOT_MAX_COMPONENT_BYTES),
                        usize_as_u64(component.len()),
                    ),
            );
        }
        if index >= GENERATION_ROOT_MAX_COMPONENTS {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                    .at_component(index)
                    .with_counts(
                        usize_as_u64(GENERATION_ROOT_MAX_COMPONENTS),
                        usize_as_u64(index.saturating_add(1)),
                    ),
            );
        }
        components.push(OsStr::from_bytes(component).to_os_string());
    }
    Ok(components)
}

pub(super) fn basic_identity(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
) -> GenerationRootResult<BasicObjectIdentity> {
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    let mount_identity = route_mount_identity(descriptor, stage)?;
    Ok(BasicObjectIdentity {
        device: stat_device_as_u64(stat.st_dev),
        inode: stat_inode_as_u64(stat.st_ino),
        mode: stat_mode_as_u32(stat.st_mode),
        uid: stat.st_uid,
        gid: stat.st_gid,
        mount_identity,
    })
}

fn absolute_ancestor_security_witness(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
) -> GenerationRootResult<AbsoluteAncestorSecurityWitness> {
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    #[cfg(target_os = "linux")]
    let changed_nanoseconds = i64::try_from(stat.st_ctime_nsec)
        .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
    #[cfg(target_os = "macos")]
    let changed_nanoseconds = stat.st_ctime_nsec;
    Ok(AbsoluteAncestorSecurityWitness {
        identity: basic_identity(descriptor, stage)?,
        changed_seconds: stat_seconds_as_i64(stat.st_ctime),
        changed_nanoseconds,
    })
}

fn validate_absolute_ancestor_owner_mode(
    uid: u32,
    mode: u32,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let effective_uid = rustix::process::geteuid().as_raw();
    if uid != effective_uid && uid != 0 {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::WrongOwner, stage)
                .with_counts(u64::from(effective_uid), u64::from(uid)),
        );
    }
    let writable_by_non_owner = mode & 0o022;
    if writable_by_non_owner != 0 {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::WrongMode, stage)
                .with_counts(0, u64::from(writable_by_non_owner)),
        );
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn validate_absolute_ancestor_owner_mode_for_test(
    uid: u32,
    mode: u32,
) -> GenerationRootResult<()> {
    validate_absolute_ancestor_owner_mode(uid, mode, GenerationRootStage::QualifyRootSecurity)
}

fn at_absolute_component(error: GenerationRootError, index: Option<usize>) -> GenerationRootError {
    index.map_or(error, |index| error.at_component(index))
}

pub(super) fn qualify_absolute_ancestor(
    descriptor: &OwnedFd,
    stage: GenerationRootStage,
    index: Option<usize>,
) -> GenerationRootResult<BasicObjectIdentity> {
    // The ACL read is bracketed by two ctime witnesses so an owner, mode,
    // or ACL change racing the read cannot slip through. A directory's
    // ctime also moves when any entry is created or removed beside the
    // route, which is ordinary neighbour churn rather than a security
    // change — so a moved witness re-qualifies the ancestor from scratch
    // (owner, mode, and ACL are all re-read against the fresh witness)
    // instead of refusing, up to a small bound. Only an ancestor whose
    // ctime never settles across every attempt is reported as changed.
    const ANCESTOR_QUALIFICATION_ATTEMPTS: usize = 4;
    let result = (|| {
        for _ in 0..ANCESTOR_QUALIFICATION_ATTEMPTS {
            let before = absolute_ancestor_security_witness(descriptor, stage)?;
            #[cfg(target_os = "linux")]
            let raw_mode = before.identity.mode;
            #[cfg(target_os = "macos")]
            let raw_mode = u16::try_from(before.identity.mode).map_err(|_| {
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
            })?;
            if FileType::from_raw_mode(raw_mode) != FileType::Directory {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::NotDirectory,
                    stage,
                ));
            }
            validate_absolute_ancestor_owner_mode(
                before.identity.uid,
                before.identity.mode,
                stage,
            )?;
            #[cfg(test)]
            test_boundary(TestBoundary::BeforeAbsoluteAncestorAclRead { index })?;
            let acl_presence =
                crate::fd_acl::extended_acl_presence(descriptor.as_fd()).map_err(|error| {
                    GenerationRootError::new(GenerationRootErrorKind::AclRejected, stage)
                        .with_raw_os_error(error.raw_os_error().unwrap_or(libc::EIO))
                })?;
            #[cfg(test)]
            test_boundary(TestBoundary::AfterAbsoluteAncestorAclRead { index })?;
            match acl_presence {
                crate::fd_acl::ExtendedAclPresence::Absent => {}
                crate::fd_acl::ExtendedAclPresence::Present => {
                    return Err(GenerationRootError::new(
                        GenerationRootErrorKind::AclRejected,
                        stage,
                    ));
                }
            }
            let after = absolute_ancestor_security_witness(descriptor, stage)?;
            if before == after {
                return Ok(after.identity);
            }
        }
        Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            stage,
        ))
    })();
    result.map_err(|error| at_absolute_component(error, index))
}

pub(super) fn validate_private_directory(
    descriptor: &OwnedFd,
    witness: GenerationRootObjectWitness,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::NotDirectory,
            stage,
        ));
    }
    validate_owner_mode(witness, GENERATION_ROOT_DIRECTORY_MODE, stage)?;
    inspect_acl_if_required(descriptor, witness, stage)
}

pub(super) fn validate_private_regular_file(
    descriptor: &OwnedFd,
    witness: GenerationRootObjectWitness,
    expectation: GenerationFileExpectation,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    validate_private_regular_file_metadata(descriptor, witness, expectation, stage)?;
    inspect_acl_if_required(descriptor, witness, stage)
}

/// Validate only metadata available through either a data descriptor or a
/// Linux `O_PATH` probe. ACL inspection is deliberately excluded: Linux
/// `fgetxattr` rejects `O_PATH` descriptors with `EBADF`, so callers must
/// invoke [`validate_private_regular_file`] on the proc-derived RDONLY or
/// RDWR descriptor before content I/O.
pub(super) fn validate_private_regular_file_metadata(
    descriptor: &OwnedFd,
    witness: GenerationRootObjectWitness,
    expectation: GenerationFileExpectation,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::NotRegularFile,
            stage,
        ));
    }
    if witness.hard_links != 1 {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::HardLinked, stage)
                .with_counts(1, witness.hard_links),
        );
    }
    validate_owner_mode(witness, expectation.expected_mode(), stage)?;
    if witness.byte_len != expectation.byte_len {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::SizeMismatch, stage)
                .with_counts(expectation.byte_len, witness.byte_len),
        );
    }
    Ok(())
}

fn validate_owner_mode(
    witness: GenerationRootObjectWitness,
    expected_mode: u32,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    let effective_uid = rustix::process::geteuid().as_raw();
    if witness.uid != effective_uid {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::WrongOwner, stage)
                .with_counts(u64::from(effective_uid), u64::from(witness.uid)),
        );
    }
    let actual_mode = witness.mode & MODE_PERMISSIONS_MASK;
    if actual_mode != expected_mode {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::WrongMode, stage)
                .with_counts(u64::from(expected_mode), u64::from(actual_mode)),
        );
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn validate_owner_mode_for_test(
    witness: GenerationRootObjectWitness,
    expected_mode: u32,
) -> GenerationRootResult<()> {
    validate_owner_mode(
        witness,
        expected_mode,
        GenerationRootStage::QualifyRootSecurity,
    )
}

pub(super) fn os_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
    GenerationRootError::new(GenerationRootErrorKind::Io, stage)
        .with_raw_os_error(error.raw_os_error())
}

pub(super) fn usize_as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub(in crate::generation_root) fn stat_mode_as_u32<T>(mode: T) -> u32
where
    u32: From<T>,
{
    u32::from(mode)
}

pub(in crate::generation_root) fn stat_inode_as_u64<T>(inode: T) -> u64
where
    u64: From<T>,
{
    u64::from(inode)
}

pub(in crate::generation_root) fn stat_link_count_as_u64<T>(link_count: T) -> u64
where
    u64: From<T>,
{
    u64::from(link_count)
}

pub(in crate::generation_root) fn stat_seconds_as_i64<T>(seconds: T) -> i64
where
    i64: From<T>,
{
    i64::from(seconds)
}

pub(in crate::generation_root) fn verify_exact_directory_entry(
    parent: &OwnedFd,
    component: &OsStr,
    stage: GenerationRootStage,
    index: usize,
) -> GenerationRootResult<()> {
    use rustix::fs::Dir;

    #[cfg(test)]
    test_boundary(TestBoundary::BeforeExactNameEnumeration)?;
    let mut directory = Dir::read_from(parent).map_err(|error| {
        GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
            .at_component(index)
            .with_raw_os_error(error.raw_os_error())
    })?;
    let expected = component.as_bytes();
    let mut entry_count = 0_usize;
    let mut name_bytes = 0_usize;
    let mut exact_matches = 0_u8;
    while let Some(entry) = directory.read() {
        let entry = entry.map_err(|error| {
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index)
                .with_raw_os_error(error.raw_os_error())
        })?;
        entry_count = entry_count.saturating_add(1);
        let name = entry.file_name().to_bytes();
        name_bytes = name_bytes.saturating_add(name.len());
        validate_directory_scan_counts(entry_count, name_bytes, stage, index)?;
        if name == expected {
            exact_matches = exact_matches.saturating_add(1);
        }
    }
    #[cfg(test)]
    test_boundary(TestBoundary::AfterExactNameEnumeration)?;
    if exact_matches != 1 {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index)
                .with_counts(1, u64::from(exact_matches)),
        );
    }
    Ok(())
}

/// Bounded no-follow inventory of the retained root directory.
pub(in crate::generation_root) fn inventory(
    root: &RootHandle,
) -> GenerationRootResult<Vec<GenerationRootEntry>> {
    use rustix::fs::{AtFlags, Dir, FileType, statat};

    let stage = GenerationRootStage::InventoryDirectory;
    revalidate_root(root)?;
    let mut directory =
        Dir::read_from(&root.state.descriptor).map_err(|error| os_error(stage, error))?;
    let mut entries = Vec::new();
    let mut entry_count = 0_usize;
    let mut name_bytes = 0_usize;
    while let Some(entry) = directory.read() {
        let entry = entry.map_err(|error| os_error(stage, error))?;
        let name = entry.file_name();
        let raw = name.to_bytes();
        if raw == b"." || raw == b".." {
            continue;
        }
        entry_count = entry_count.saturating_add(1);
        name_bytes = name_bytes.saturating_add(raw.len());
        validate_directory_scan_counts(entry_count, name_bytes, stage, 0)?;
        let stat = statat(&root.state.descriptor, name, AtFlags::SYMLINK_NOFOLLOW)
            .map_err(|error| os_error(stage, error))?;
        let kind = match FileType::from_raw_mode(stat.st_mode) {
            FileType::RegularFile => GenerationRootEntryKind::RegularFile,
            FileType::Directory => GenerationRootEntryKind::Directory,
            FileType::Symlink => GenerationRootEntryKind::Symlink,
            _ => GenerationRootEntryKind::Other,
        };
        let byte_len = u64::try_from(stat.st_size)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
        entries.push(GenerationRootEntry {
            name: OsStr::from_bytes(raw).to_os_string(),
            kind,
            device: stat_device_as_u64(stat.st_dev),
            inode: stat_inode_as_u64(stat.st_ino),
            hard_links: stat_link_count_as_u64(stat.st_nlink),
            byte_len,
            mode: stat_mode_as_u32(stat.st_mode),
        });
    }
    revalidate_root(root)?;
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(entries)
}
fn validate_directory_scan_counts(
    entry_count: usize,
    name_bytes: usize,
    stage: GenerationRootStage,
    index: usize,
) -> GenerationRootResult<()> {
    if entry_count > GENERATION_ROOT_MAX_DIRECTORY_ENTRIES {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                .at_component(index)
                .with_counts(
                    usize_as_u64(GENERATION_ROOT_MAX_DIRECTORY_ENTRIES),
                    usize_as_u64(entry_count),
                ),
        );
    }
    if name_bytes > GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                .at_component(index)
                .with_counts(
                    usize_as_u64(GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES),
                    usize_as_u64(name_bytes),
                ),
        );
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn validate_directory_scan_counts_for_test(
    entry_count: usize,
    name_bytes: usize,
) -> GenerationRootResult<()> {
    validate_directory_scan_counts(
        entry_count,
        name_bytes,
        GenerationRootStage::OpenRelativeDirectory,
        7,
    )
}
