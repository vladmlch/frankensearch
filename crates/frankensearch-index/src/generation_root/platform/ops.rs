use super::super::{
    GENERATION_ROOT_AUTHORITY_FILE_NAME, GENERATION_ROOT_LOCK_FILE_NAME, GenerationFileExpectation,
    GenerationRootError, GenerationRootErrorKind, GenerationRootLockMode,
    GenerationRootObjectWitness, GenerationRootResult, GenerationRootStage,
};
#[cfg(target_os = "linux")]
use super::linux::{
    object_witness, open_absolute_route, qualify_filesystem, sync_directory_descriptor,
    sync_file_descriptor, validate_root_writable,
};
#[cfg(target_os = "macos")]
use super::macos::{
    object_witness, open_absolute_route, qualify_filesystem, sync_directory_descriptor,
    sync_file_descriptor, validate_root_writable,
};
use super::types::{
    AbsoluteRoute, AcquiredFlockDescriptor, AnchorImage, ControlHandle, ExactContentWitness,
    FileAdmission, FileHandle, LockHandle, RelativePath, RootHandle, RootState,
};
#[cfg(test)]
use super::types::{TestBoundary, test_boundary};
#[cfg(any(
    all(test, target_os = "linux"),
    all(test, target_os = "macos", target_arch = "aarch64")
))]
use super::unix_common::revalidate_control_route;
#[cfg(target_os = "macos")]
use super::unix_common::{
    PreopenedAccess, qualify_preopened_file, revalidate_preopened_file_route,
    validate_preopened_regular_descriptor,
};
#[cfg(target_os = "linux")]
use super::unix_common::{open_relative_file, open_relative_probe};
use super::unix_common::{
    os_error, parse_route_bytes, read_exact_owned, read_exact_sha256, revalidate_anchor_route,
    revalidate_file_route, revalidate_lock_route, usize_as_u64, validate_bound_content,
    validate_expected_sha256, validate_lock_content, validate_private_directory,
};
#[cfg(target_os = "macos")]
use rustix::fs::OFlags;
use rustix::io::Errno;
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
#[cfg(any(target_os = "macos", all(test, target_os = "linux")))]
use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::Arc;

pub(in crate::generation_root) fn parse_relative(
    path: &Path,
) -> GenerationRootResult<RelativePath> {
    let components = parse_route_bytes(
        path.as_os_str(),
        false,
        GenerationRootStage::ParseRelativeRoute,
    )?;
    Ok(RelativePath {
        components: Arc::from(components),
    })
}

pub(in crate::generation_root) fn relative_component_count(path: &RelativePath) -> usize {
    path.components.len()
}

pub(in crate::generation_root) fn admit_root(path: &Path) -> GenerationRootResult<RootHandle> {
    let creator_process_id = std::process::id();
    let route = AbsoluteRoute {
        components: Arc::from(parse_route_bytes(
            path.as_os_str(),
            true,
            GenerationRootStage::ParseRootRoute,
        )?),
    };
    let first = open_absolute_route(&route, GenerationRootStage::OpenRootRoute)?;
    let (filesystem, mount_identity) =
        qualify_filesystem(&first.descriptor, GenerationRootStage::QualifyFilesystem)?;
    let first_witness = object_witness(
        &first.descriptor,
        filesystem,
        mount_identity,
        GenerationRootStage::QualifyRootSecurity,
    )?;
    validate_private_directory(
        &first.descriptor,
        first_witness,
        GenerationRootStage::QualifyRootSecurity,
    )?;
    validate_root_writable(
        &first.descriptor,
        first_witness.filesystem,
        GenerationRootStage::QualifyFilesystem,
    )?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFirstRootQualification)?;

    let second = open_absolute_route(&route, GenerationRootStage::RevalidateRootRoute)?;
    if first.route_identities != second.route_identities {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRootRoute,
        ));
    }
    let (second_filesystem, second_mount_identity) =
        qualify_filesystem(&second.descriptor, GenerationRootStage::RevalidateRootRoute)?;
    let second_witness = object_witness(
        &second.descriptor,
        second_filesystem,
        second_mount_identity,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    validate_private_directory(
        &second.descriptor,
        second_witness,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    validate_root_writable(
        &second.descriptor,
        second_witness.filesystem,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    if !first_witness.same_directory_security_identity(second_witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRootRoute,
        ));
    }
    validate_process_identity(creator_process_id, GenerationRootStage::RevalidateRootRoute)?;

    Ok(RootHandle {
        state: Arc::new(RootState {
            route,
            descriptor: first.descriptor,
            witness: first_witness,
            route_identities: Arc::from(first.route_identities),
            creator_process_id,
        }),
    })
}

pub(in crate::generation_root) fn root_witness(root: &RootHandle) -> GenerationRootObjectWitness {
    root.state.witness
}

pub(in crate::generation_root) fn revalidate_root(root: &RootHandle) -> GenerationRootResult<()> {
    validate_process_identity(
        root.state.creator_process_id,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    let retained = object_witness(
        &root.state.descriptor,
        root.state.witness.filesystem,
        root.state.witness.mount_identity,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    if !retained.same_directory_security_identity(root.state.witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRootRoute,
        ));
    }
    validate_private_directory(
        &root.state.descriptor,
        retained,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    validate_root_writable(
        &root.state.descriptor,
        retained.filesystem,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    let reopened =
        open_absolute_route(&root.state.route, GenerationRootStage::RevalidateRootRoute)?;
    if reopened.route_identities.as_slice() != root.state.route_identities.as_ref() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRootRoute,
        ));
    }
    let (filesystem, mount_identity) = qualify_filesystem(
        &reopened.descriptor,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    let reopened_witness = object_witness(
        &reopened.descriptor,
        filesystem,
        mount_identity,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    validate_private_directory(
        &reopened.descriptor,
        reopened_witness,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    validate_root_writable(
        &reopened.descriptor,
        reopened_witness.filesystem,
        GenerationRootStage::RevalidateRootRoute,
    )?;
    if !retained.same_directory_security_identity(reopened_witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRootRoute,
        ));
    }
    validate_process_identity(
        root.state.creator_process_id,
        GenerationRootStage::RevalidateRootRoute,
    )
}

#[cfg(target_os = "linux")]
pub(in crate::generation_root) fn admit_file(
    root: &RootHandle,
    path: &RelativePath,
    expectation: GenerationFileExpectation,
) -> GenerationRootResult<FileAdmission> {
    revalidate_root(root)?;
    let opened = open_relative_file(
        root,
        path,
        expectation,
        false,
        GenerationRootStage::OpenRelativeDirectory,
        GenerationRootStage::OpenRegularFile,
    )?;
    let (bytes, after_read) = read_exact_owned(&opened.descriptor, opened.witness, expectation)?;
    let sha256: [u8; 32] = Sha256::digest(&bytes).into();
    if let Some(expected_sha256) = expectation.sha256 {
        if sha256 != expected_sha256 {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::HashMismatch,
                GenerationRootStage::ReadRegularFile,
            ));
        }
    }
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFileRouteReopen)?;
    let reopened = open_relative_probe(
        root,
        path,
        expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    #[cfg(test)]
    test_boundary(TestBoundary::AfterFileRouteReopen)?;
    if !after_read.mutation_stable_eq(reopened.witness)
        || opened.component_identities != reopened.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeFinalRootRevalidation)?;
    revalidate_root(root)?;
    Ok(FileAdmission {
        handle: FileHandle {
            descriptor: opened.descriptor,
            root: root.clone(),
            path: path.clone(),
            expectation,
            content_witness: ExactContentWitness { sha256 },
            component_identities: opened.component_identities,
        },
        bytes: Arc::from(bytes),
        witness: after_read,
        sha256,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::generation_root) fn admit_preopened_file(
    root: &RootHandle,
    path: &RelativePath,
    descriptor: OwnedFd,
    expectation: GenerationFileExpectation,
) -> GenerationRootResult<FileAdmission> {
    validate_process_identity(
        root.state.creator_process_id,
        GenerationRootStage::OpenRegularFile,
    )?;
    validate_preopened_regular_descriptor(
        &descriptor,
        PreopenedAccess::ReadOnly,
        GenerationRootStage::OpenRegularFile,
    )?;
    revalidate_root(root)?;
    let opened = qualify_preopened_file(
        root,
        path,
        descriptor,
        expectation,
        PreopenedAccess::ReadOnly,
        GenerationRootStage::OpenRegularFile,
    )?;
    let (bytes, after_read) = read_exact_owned(&opened.descriptor, opened.witness, expectation)?;
    let sha256: [u8; 32] = Sha256::digest(&bytes).into();
    if expectation
        .sha256
        .is_some_and(|expected_sha256| sha256 != expected_sha256)
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::HashMismatch,
            GenerationRootStage::ReadRegularFile,
        ));
    }
    revalidate_preopened_file_route(
        root,
        path,
        expectation,
        &opened.descriptor,
        after_read,
        &opened.component_identities,
        OFlags::RDONLY,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    revalidate_root(root)?;
    Ok(FileAdmission {
        handle: FileHandle {
            descriptor: opened.descriptor,
            root: root.clone(),
            path: path.clone(),
            expectation,
            content_witness: ExactContentWitness { sha256 },
            component_identities: opened.component_identities,
        },
        bytes: Arc::from(bytes),
        witness: after_read,
        sha256,
    })
}

pub(in crate::generation_root) fn sync_root_directory(
    root: &RootHandle,
) -> GenerationRootResult<()> {
    revalidate_root(root)?;
    sync_directory_descriptor(&root.state.descriptor)?;
    revalidate_root(root)
}

pub(in crate::generation_root) fn sync_file(
    file: &FileHandle,
    expected: GenerationRootObjectWitness,
) -> GenerationRootResult<()> {
    revalidate_file_route(file, expected)?;
    let (before_content, before) = read_exact_sha256(
        &file.descriptor,
        expected,
        file.expectation,
        GenerationRootStage::SyncRegularFile,
    )?;
    validate_bound_content(
        before_content,
        file.content_witness,
        GenerationRootStage::SyncRegularFile,
    )?;
    sync_file_descriptor(&file.descriptor)?;
    let (after_content, _) = read_exact_sha256(
        &file.descriptor,
        before,
        file.expectation,
        GenerationRootStage::SyncRegularFile,
    )?;
    validate_bound_content(
        after_content,
        file.content_witness,
        GenerationRootStage::SyncRegularFile,
    )?;
    revalidate_file_route(file, expected)
}

#[cfg(target_os = "linux")]
pub(in crate::generation_root) fn admit_control_file(
    root: &RootHandle,
    path: &RelativePath,
    expectation: GenerationFileExpectation,
) -> GenerationRootResult<ControlHandle> {
    revalidate_root(root)?;
    let opened = open_relative_file(
        root,
        path,
        expectation,
        true,
        GenerationRootStage::OpenRelativeDirectory,
        GenerationRootStage::OpenRegularFile,
    )?;
    let (bytes, after_read) = read_exact_owned(&opened.descriptor, opened.witness, expectation)?;
    let content_witness = ExactContentWitness {
        sha256: Sha256::digest(&bytes).into(),
    };
    let reopened = open_relative_probe(
        root,
        path,
        expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    if !after_read.mutation_stable_eq(reopened.witness)
        || opened.component_identities != reopened.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    let (revalidated_content, revalidated_after) = read_exact_sha256(
        &opened.descriptor,
        after_read,
        expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    validate_bound_content(
        revalidated_content,
        content_witness,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    revalidate_root(root)?;
    Ok(ControlHandle {
        descriptor: opened.descriptor,
        #[cfg(test)]
        bytes: Arc::from(bytes),
        root: root.clone(),
        path: path.clone(),
        expectation,
        witness: revalidated_after,
        #[cfg(test)]
        content_witness: revalidated_content,
        component_identities: opened.component_identities,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::generation_root) fn admit_preopened_control_file(
    root: &RootHandle,
    path: &RelativePath,
    descriptor: OwnedFd,
    expectation: GenerationFileExpectation,
) -> GenerationRootResult<ControlHandle> {
    validate_process_identity(
        root.state.creator_process_id,
        GenerationRootStage::OpenRegularFile,
    )?;
    validate_preopened_regular_descriptor(
        &descriptor,
        PreopenedAccess::ReadWrite,
        GenerationRootStage::OpenRegularFile,
    )?;
    revalidate_root(root)?;
    let opened = qualify_preopened_file(
        root,
        path,
        descriptor,
        expectation,
        PreopenedAccess::ReadWrite,
        GenerationRootStage::OpenRegularFile,
    )?;
    let (bytes, after_read) = read_exact_owned(&opened.descriptor, opened.witness, expectation)?;
    let content_witness = ExactContentWitness {
        sha256: Sha256::digest(&bytes).into(),
    };
    revalidate_preopened_file_route(
        root,
        path,
        expectation,
        &opened.descriptor,
        after_read,
        &opened.component_identities,
        OFlags::RDWR,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    let (revalidated_content, revalidated_after) = read_exact_sha256(
        &opened.descriptor,
        after_read,
        expectation,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    validate_bound_content(
        revalidated_content,
        content_witness,
        GenerationRootStage::RevalidateRegularFile,
    )?;
    revalidate_root(root)?;
    Ok(ControlHandle {
        descriptor: opened.descriptor,
        #[cfg(test)]
        bytes: Arc::from(bytes),
        root: root.clone(),
        path: path.clone(),
        expectation,
        witness: revalidated_after,
        #[cfg(test)]
        content_witness: revalidated_content,
        component_identities: opened.component_identities,
    })
}

pub(in crate::generation_root) fn control_witness(
    control: &ControlHandle,
) -> GenerationRootObjectWitness {
    control.witness
}

#[cfg(test)]
pub(in crate::generation_root) fn control_bytes(control: &ControlHandle) -> Arc<[u8]> {
    Arc::clone(&control.bytes)
}

#[cfg(test)]
pub(in crate::generation_root) fn control_sha256(control: &ControlHandle) -> [u8; 32] {
    control.content_witness.sha256
}

pub(in crate::generation_root) fn revalidate_anchor_set(
    root: &RootHandle,
    lock: &ControlHandle,
    authority: &ControlHandle,
) -> GenerationRootResult<()> {
    revalidate_root(root)?;
    if !Arc::ptr_eq(&root.state, &lock.root.state)
        || !Arc::ptr_eq(&root.state, &authority.root.state)
        || !relative_path_is_exact_name(&lock.path, GENERATION_ROOT_LOCK_FILE_NAME)
        || !relative_path_is_exact_name(&authority.path, GENERATION_ROOT_AUTHORITY_FILE_NAME)
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    revalidate_anchor_route(lock)?;
    revalidate_anchor_route(authority)?;
    let root_witness = root.state.witness;
    if lock.witness.same_object(authority.witness)
        || lock.witness.device != root_witness.device
        || authority.witness.device != root_witness.device
        || lock.witness.mount_identity != root_witness.mount_identity
        || authority.witness.mount_identity != root_witness.mount_identity
        || lock.witness.filesystem != root_witness.filesystem
        || authority.witness.filesystem != root_witness.filesystem
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::RevalidateRegularFile,
        ));
    }
    revalidate_anchor_route(lock)?;
    revalidate_anchor_route(authority)?;
    revalidate_root(root)
}

fn relative_path_is_exact_name(path: &RelativePath, expected: &str) -> bool {
    path.components.len() == 1 && path.components[0].as_os_str() == OsStr::new(expected)
}

pub(in crate::generation_root) fn capture_anchor_image(
    control: &ControlHandle,
) -> GenerationRootResult<AnchorImage> {
    revalidate_anchor_route(control)?;
    let before = object_witness(
        &control.descriptor,
        control.witness.filesystem,
        control.witness.mount_identity,
        GenerationRootStage::AcquireLock,
    )?;
    if !before.same_control_anchor_identity(control.witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::AcquireLock,
        ));
    }
    let (bytes, witness) = read_exact_owned(&control.descriptor, before, control.expectation)?;
    let sha256: [u8; 32] = Sha256::digest(&bytes).into();
    if control
        .expectation
        .sha256
        .is_some_and(|expected| expected != sha256)
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::HashMismatch,
            GenerationRootStage::AcquireLock,
        ));
    }
    revalidate_anchor_route(control)?;
    Ok(AnchorImage {
        bytes: Arc::from(bytes),
        witness,
        sha256,
    })
}

pub(in crate::generation_root) fn anchor_images_match(
    left: &AnchorImage,
    right: &AnchorImage,
) -> bool {
    left.sha256 == right.sha256
        && left.bytes == right.bytes
        && left.witness.mutation_stable_eq(right.witness)
}

pub(in crate::generation_root) fn validate_anchor_image(
    control: &ControlHandle,
    expected: &AnchorImage,
) -> GenerationRootResult<()> {
    let observed = capture_anchor_image(control)?;
    if anchor_images_match(&observed, expected) {
        Ok(())
    } else {
        Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::ReleaseLock,
        ))
    }
}

#[cfg(target_os = "linux")]
pub(in crate::generation_root) fn try_anchor_lock(
    control: &ControlHandle,
    mode: GenerationRootLockMode,
) -> GenerationRootResult<LockHandle> {
    let creator_process_id = std::process::id();
    revalidate_anchor_route(control)?;
    let opened = open_relative_file(
        &control.root,
        &control.path,
        control.expectation,
        true,
        GenerationRootStage::AcquireLock,
        GenerationRootStage::AcquireLock,
    )?;
    if !opened.witness.same_control_anchor_identity(control.witness)
        || opened.component_identities != control.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::AcquireLock,
        ));
    }
    let operation = match mode {
        GenerationRootLockMode::Shared => rustix::fs::FlockOperation::NonBlockingLockShared,
        GenerationRootLockMode::Exclusive => rustix::fs::FlockOperation::NonBlockingLockExclusive,
    };
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeLock)?;
    let acquired = match rustix::fs::flock(&opened.descriptor, operation) {
        Ok(()) => AcquiredFlockDescriptor::new(opened.descriptor, creator_process_id),
        Err(error) if error == Errno::WOULDBLOCK => {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::LockContended,
                GenerationRootStage::AcquireLock,
            ));
        }
        Err(error) => return Err(os_error(GenerationRootStage::AcquireLock, error)),
    };
    #[cfg(test)]
    test_boundary(TestBoundary::AfterLock)?;
    validate_process_identity(creator_process_id, GenerationRootStage::AcquireLock)?;

    let (first_bytes, first_after) =
        read_exact_owned(acquired.descriptor(), opened.witness, control.expectation)?;
    let first_content = ExactContentWitness {
        sha256: Sha256::digest(&first_bytes).into(),
    };
    validate_expected_sha256(
        first_content,
        control.expectation,
        GenerationRootStage::AcquireLock,
    )?;
    revalidate_anchor_route(control)?;
    let (second_bytes, second_after) =
        read_exact_owned(acquired.descriptor(), first_after, control.expectation)?;
    let second_content = ExactContentWitness {
        sha256: Sha256::digest(&second_bytes).into(),
    };
    if first_bytes != second_bytes || first_content != second_content {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::AcquireLock,
        ));
    }
    revalidate_anchor_route(control)?;
    let descriptor = acquired.into_descriptor();
    Ok(LockHandle {
        descriptor,
        #[cfg(test)]
        bytes: Arc::from(second_bytes),
        witness: second_after,
        content_witness: second_content,
        root: control.root.clone(),
        path: control.path.clone(),
        expectation: control.expectation,
        component_identities: opened.component_identities,
        creator_process_id,
        lock_held: true,
        #[cfg(test)]
        mode,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::generation_root) fn try_anchor_lock_preopened(
    control: &ControlHandle,
    descriptor: OwnedFd,
    mode: GenerationRootLockMode,
) -> GenerationRootResult<LockHandle> {
    let creator_process_id = std::process::id();
    validate_process_identity(
        control.root.state.creator_process_id,
        GenerationRootStage::AcquireLock,
    )?;
    validate_preopened_regular_descriptor(
        &descriptor,
        PreopenedAccess::ReadWrite,
        GenerationRootStage::AcquireLock,
    )?;
    revalidate_anchor_route(control)?;
    let opened = qualify_preopened_file(
        &control.root,
        &control.path,
        descriptor,
        control.expectation,
        PreopenedAccess::ReadWrite,
        GenerationRootStage::AcquireLock,
    )?;
    if !opened.witness.same_control_anchor_identity(control.witness)
        || opened.component_identities != control.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::AcquireLock,
        ));
    }
    let operation = match mode {
        GenerationRootLockMode::Shared => rustix::fs::FlockOperation::NonBlockingLockShared,
        GenerationRootLockMode::Exclusive => rustix::fs::FlockOperation::NonBlockingLockExclusive,
    };
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeLock)?;
    let acquired = match rustix::fs::flock(&opened.descriptor, operation) {
        Ok(()) => AcquiredFlockDescriptor::new(opened.descriptor, creator_process_id),
        Err(error) if error == Errno::WOULDBLOCK => {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::LockContended,
                GenerationRootStage::AcquireLock,
            ));
        }
        Err(error) => return Err(os_error(GenerationRootStage::AcquireLock, error)),
    };
    #[cfg(test)]
    test_boundary(TestBoundary::AfterLock)?;
    validate_process_identity(creator_process_id, GenerationRootStage::AcquireLock)?;
    let descriptor_status_flags = rustix::fs::fcntl_getfl(acquired.descriptor())
        .map_err(|error| os_error(GenerationRootStage::AcquireLock, error))?;

    let (first_bytes, first_after) =
        read_exact_owned(acquired.descriptor(), opened.witness, control.expectation)?;
    let first_content = ExactContentWitness {
        sha256: Sha256::digest(&first_bytes).into(),
    };
    validate_expected_sha256(
        first_content,
        control.expectation,
        GenerationRootStage::AcquireLock,
    )?;
    revalidate_anchor_route(control)?;
    let (second_bytes, second_after) =
        read_exact_owned(acquired.descriptor(), first_after, control.expectation)?;
    let second_content = ExactContentWitness {
        sha256: Sha256::digest(&second_bytes).into(),
    };
    if first_bytes != second_bytes || first_content != second_content {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::AcquireLock,
        ));
    }
    revalidate_anchor_route(control)?;
    let descriptor = acquired.into_descriptor();
    Ok(LockHandle {
        descriptor,
        #[cfg(test)]
        bytes: Arc::from(second_bytes),
        witness: second_after,
        content_witness: second_content,
        root: control.root.clone(),
        path: control.path.clone(),
        expectation: control.expectation,
        component_identities: opened.component_identities,
        creator_process_id,
        lock_held: true,
        descriptor_status_flags,
        #[cfg(test)]
        mode,
    })
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn try_lock(
    control: &ControlHandle,
    mode: GenerationRootLockMode,
) -> GenerationRootResult<LockHandle> {
    let creator_process_id = std::process::id();
    revalidate_control_route(control)?;
    let opened = open_relative_file(
        &control.root,
        &control.path,
        control.expectation,
        true,
        GenerationRootStage::AcquireLock,
        GenerationRootStage::AcquireLock,
    )?;
    if !opened.witness.mutation_stable_eq(control.witness) {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::AcquireLock,
        ));
    }
    if opened.component_identities != control.component_identities {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::AcquireLock,
        ));
    }
    let operation = match mode {
        GenerationRootLockMode::Shared => rustix::fs::FlockOperation::NonBlockingLockShared,
        GenerationRootLockMode::Exclusive => rustix::fs::FlockOperation::NonBlockingLockExclusive,
    };
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeLock)?;
    let acquired = match rustix::fs::flock(&opened.descriptor, operation) {
        Ok(()) => AcquiredFlockDescriptor::new(opened.descriptor, creator_process_id),
        Err(error) if error == Errno::WOULDBLOCK => {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::LockContended,
                GenerationRootStage::AcquireLock,
            ));
        }
        Err(error) => return Err(os_error(GenerationRootStage::AcquireLock, error)),
    };
    #[cfg(test)]
    test_boundary(TestBoundary::AfterLock)?;
    if creator_process_id != std::process::id() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ForkedProcess,
            GenerationRootStage::AcquireLock,
        ));
    }
    let (observed_content, after) = read_exact_sha256(
        acquired.descriptor(),
        opened.witness,
        control.expectation,
        GenerationRootStage::AcquireLock,
    )?;
    validate_bound_content(
        observed_content,
        control.content_witness,
        GenerationRootStage::AcquireLock,
    )?;
    validate_expected_sha256(
        observed_content,
        control.expectation,
        GenerationRootStage::AcquireLock,
    )?;
    revalidate_control_route(control)?;
    let (revalidated_content, revalidated_after) = read_exact_sha256(
        acquired.descriptor(),
        after,
        control.expectation,
        GenerationRootStage::AcquireLock,
    )?;
    validate_bound_content(
        revalidated_content,
        control.content_witness,
        GenerationRootStage::AcquireLock,
    )?;
    let descriptor = acquired.into_descriptor();
    Ok(LockHandle {
        descriptor,
        #[cfg(test)]
        bytes: Arc::clone(&control.bytes),
        witness: revalidated_after,
        content_witness: revalidated_content,
        root: control.root.clone(),
        path: control.path.clone(),
        expectation: control.expectation,
        component_identities: opened.component_identities,
        creator_process_id,
        lock_held: true,
        #[cfg(test)]
        mode,
    })
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
pub(in crate::generation_root) fn try_lock_preopened(
    control: &ControlHandle,
    descriptor: OwnedFd,
    mode: GenerationRootLockMode,
) -> GenerationRootResult<LockHandle> {
    let creator_process_id = std::process::id();
    validate_process_identity(
        control.root.state.creator_process_id,
        GenerationRootStage::AcquireLock,
    )?;
    validate_preopened_regular_descriptor(
        &descriptor,
        PreopenedAccess::ReadWrite,
        GenerationRootStage::AcquireLock,
    )?;
    revalidate_control_route(control)?;
    let opened = qualify_preopened_file(
        &control.root,
        &control.path,
        descriptor,
        control.expectation,
        PreopenedAccess::ReadWrite,
        GenerationRootStage::AcquireLock,
    )?;
    if !opened.witness.mutation_stable_eq(control.witness)
        || opened.component_identities != control.component_identities
    {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ObjectChanged,
            GenerationRootStage::AcquireLock,
        ));
    }
    let operation = match mode {
        GenerationRootLockMode::Shared => rustix::fs::FlockOperation::NonBlockingLockShared,
        GenerationRootLockMode::Exclusive => rustix::fs::FlockOperation::NonBlockingLockExclusive,
    };
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeLock)?;
    let acquired = match rustix::fs::flock(&opened.descriptor, operation) {
        Ok(()) => AcquiredFlockDescriptor::new(opened.descriptor, creator_process_id),
        Err(error) if error == Errno::WOULDBLOCK => {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::LockContended,
                GenerationRootStage::AcquireLock,
            ));
        }
        Err(error) => return Err(os_error(GenerationRootStage::AcquireLock, error)),
    };
    #[cfg(test)]
    test_boundary(TestBoundary::AfterLock)?;
    if creator_process_id != std::process::id() {
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ForkedProcess,
            GenerationRootStage::AcquireLock,
        ));
    }
    let descriptor_status_flags = rustix::fs::fcntl_getfl(acquired.descriptor())
        .map_err(|error| os_error(GenerationRootStage::AcquireLock, error))?;
    let (locked_content, locked_after) = read_exact_sha256(
        acquired.descriptor(),
        opened.witness,
        control.expectation,
        GenerationRootStage::AcquireLock,
    )?;
    validate_bound_content(
        locked_content,
        control.content_witness,
        GenerationRootStage::AcquireLock,
    )?;
    validate_expected_sha256(
        locked_content,
        control.expectation,
        GenerationRootStage::AcquireLock,
    )?;
    revalidate_control_route(control)?;
    let (revalidated_content, revalidated_after) = read_exact_sha256(
        acquired.descriptor(),
        locked_after,
        control.expectation,
        GenerationRootStage::AcquireLock,
    )?;
    validate_bound_content(
        revalidated_content,
        control.content_witness,
        GenerationRootStage::AcquireLock,
    )?;
    let descriptor = acquired.into_descriptor();
    Ok(LockHandle {
        descriptor,
        #[cfg(test)]
        bytes: Arc::clone(&control.bytes),
        witness: revalidated_after,
        content_witness: revalidated_content,
        root: control.root.clone(),
        path: control.path.clone(),
        expectation: control.expectation,
        component_identities: opened.component_identities,
        creator_process_id,
        lock_held: true,
        descriptor_status_flags,
        #[cfg(test)]
        mode,
    })
}

#[cfg(test)]
pub(in crate::generation_root) fn lock_witness(lock: &LockHandle) -> GenerationRootObjectWitness {
    lock.witness
}

#[cfg(test)]
pub(in crate::generation_root) fn lock_bytes(lock: &LockHandle) -> Arc<[u8]> {
    Arc::clone(&lock.bytes)
}

#[cfg(test)]
pub(in crate::generation_root) fn lock_sha256(lock: &LockHandle) -> [u8; 32] {
    lock.content_witness.sha256
}

#[cfg(test)]
pub(in crate::generation_root) fn lock_mode(lock: &LockHandle) -> GenerationRootLockMode {
    lock.mode
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
pub(in crate::generation_root) fn set_root_creator_process_id_for_test(
    root: &mut RootHandle,
    process_id: u32,
) {
    Arc::get_mut(&mut root.state)
        .expect("test root must have sole state ownership")
        .creator_process_id = process_id;
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
pub(in crate::generation_root) fn set_control_root_creator_process_id_for_test(
    control: &mut ControlHandle,
    process_id: u32,
) {
    Arc::get_mut(&mut control.root.state)
        .expect("test control root must have sole state ownership")
        .creator_process_id = process_id;
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn set_lock_creator_process_id_for_test(
    lock: &mut LockHandle,
    process_id: u32,
) {
    lock.creator_process_id = process_id;
}

#[cfg(all(test, target_os = "linux"))]
pub(in crate::generation_root) fn duplicate_lock_descriptor_for_inheritance_for_test(
    lock: &LockHandle,
) -> GenerationRootResult<OwnedFd> {
    rustix::io::dup(&lock.descriptor)
        .map_err(|error| os_error(GenerationRootStage::AcquireLock, error))
}

pub(in crate::generation_root) fn unlock(mut lock: LockHandle) -> GenerationRootResult<()> {
    if lock.creator_process_id != std::process::id() {
        drop(lock);
        return Err(GenerationRootError::new(
            GenerationRootErrorKind::ForkedProcess,
            GenerationRootStage::ReleaseLock,
        ));
    }
    let validation_result = revalidate_lock_route(&lock)
        .and_then(|()| validate_lock_content(&lock, GenerationRootStage::ReleaseLock));
    #[cfg(test)]
    test_boundary(TestBoundary::BeforeUnlock)?;
    let unlock_result = rustix::fs::flock(&lock.descriptor, rustix::fs::FlockOperation::Unlock)
        .map_err(|error| os_error(GenerationRootStage::ReleaseLock, error));
    if unlock_result.is_ok() {
        lock.lock_held = false;
    }
    #[cfg(test)]
    let unlock_result = unlock_result.and_then(|()| test_boundary(TestBoundary::AfterUnlock));
    drop(lock);
    validation_result.and(unlock_result)
}

impl Drop for LockHandle {
    fn drop(&mut self) {
        if self.lock_held && self.creator_process_id == std::process::id() {
            let _ = rustix::fs::flock(&self.descriptor, rustix::fs::FlockOperation::Unlock);
            self.lock_held = false;
        }
    }
}

/// Positional, bounded overwrite of one range inside a control anchor,
/// followed by the platform durability barrier on that descriptor.
///
/// The range must stay inside the anchor's frozen physical length: the
/// AUTHORITY and LOCK inodes are never extended, truncated, renamed, or
/// unlinked. Route and identity are revalidated before and after the
/// write so a swapped anchor cannot receive slot bytes.
pub(in crate::generation_root) fn write_control_range(
    control: &ControlHandle,
    offset: u64,
    bytes: &[u8],
) -> GenerationRootResult<()> {
    revalidate_anchor_route(control)?;
    let end = offset
        .checked_add(usize_as_u64(bytes.len()))
        .filter(|end| *end <= control.expectation.byte_len)
        .ok_or_else(|| {
            GenerationRootError::new(
                GenerationRootErrorKind::ResourceLimit,
                GenerationRootStage::SyncRegularFile,
            )
            .with_counts(control.expectation.byte_len, offset)
        })?;
    debug_assert!(end <= control.expectation.byte_len);
    let mut written = 0_usize;
    while written < bytes.len() {
        // Test builds write in bounded chunks so a fault between two
        // chunks models a genuine short write (a torn slot produced by
        // the real write path, not by a test poking the file).
        #[cfg(test)]
        let chunk = &bytes[written..bytes.len().min(written + 1024)];
        #[cfg(not(test))]
        let chunk = &bytes[written..];
        let count = rustix::io::pwrite(&control.descriptor, chunk, offset + usize_as_u64(written))
            .map_err(|error| os_error(GenerationRootStage::SyncRegularFile, error))?;
        if count == 0 {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::Io,
                GenerationRootStage::SyncRegularFile,
            ));
        }
        written += count;
        #[cfg(test)]
        test_boundary(TestBoundary::DuringPublishSlotWrite { written })?;
    }
    sync_file_descriptor(&control.descriptor)?;
    revalidate_anchor_route(control)
}

/// Same contract as [`write_control_range`] for the LOCK anchor held by an
/// exclusive flock: frames are only ever written by the lock holder, so the
/// creator-process check runs first.
pub(in crate::generation_root) fn write_lock_range(
    lock: &mut LockHandle,
    offset: u64,
    bytes: &[u8],
) -> GenerationRootResult<()> {
    const STAGE: GenerationRootStage = GenerationRootStage::WriteLockFrame;
    let at = |phase: u64| {
        move |error: GenerationRootError| {
            GenerationRootError::new(error.kind(), STAGE).with_counts(phase, offset)
        }
    };
    validate_lock_process(lock, STAGE)?;
    revalidate_lock_route(lock).map_err(at(1))?;
    let end = offset
        .checked_add(usize_as_u64(bytes.len()))
        .filter(|end| *end <= lock.expectation.byte_len)
        .ok_or_else(|| {
            GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, STAGE)
                .with_counts(lock.expectation.byte_len, offset)
        })?;
    debug_assert!(end <= lock.expectation.byte_len);
    let mut written = 0_usize;
    while written < bytes.len() {
        let count = rustix::io::pwrite(
            &lock.descriptor,
            &bytes[written..],
            offset + usize_as_u64(written),
        )
        .map_err(|error| os_error(STAGE, error).with_counts(2, offset))?;
        if count == 0 {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::Io, STAGE).with_counts(2, offset)
            );
        }
        written += count;
    }
    sync_file_descriptor(&lock.descriptor).map_err(at(3))?;
    // The retained witness is mutation-stable (mtime/ctime); the lock
    // holder is the one legitimate mutator, so it advances its own witness
    // to the exact post-write state before revalidating the route.
    let after = object_witness(
        &lock.descriptor,
        lock.witness.filesystem,
        lock.witness.mount_identity,
        STAGE,
    )
    .map_err(at(4))?;
    if !after.same_file_security_identity(lock.witness) || after.byte_len != lock.witness.byte_len {
        return Err(
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, STAGE)
                .with_counts(5, offset),
        );
    }
    let (content, after) =
        read_exact_owned(&lock.descriptor, after, lock.expectation).map_err(at(6))?;
    lock.content_witness = ExactContentWitness {
        sha256: Sha256::digest(&content).into(),
    };
    lock.witness = after;
    #[cfg(test)]
    {
        lock.bytes = Arc::from(content);
    }
    revalidate_lock_route(lock).map_err(at(7))
}

/// Exact current bytes of the LOCK anchor through the held lock descriptor.
pub(in crate::generation_root) fn read_lock_image(
    lock: &LockHandle,
) -> GenerationRootResult<Vec<u8>> {
    validate_lock_process(lock, GenerationRootStage::ReadRegularFile)?;
    revalidate_lock_route(lock)?;
    let (bytes, _) = read_exact_owned(&lock.descriptor, lock.witness, lock.expectation)?;
    Ok(bytes)
}

#[cfg(test)]
pub(in crate::generation_root) fn sync_lock_file(lock: &LockHandle) -> GenerationRootResult<()> {
    validate_lock_process(lock, GenerationRootStage::SyncRegularFile)?;
    revalidate_lock_route(lock)?;
    let (before_content, before) = read_exact_sha256(
        &lock.descriptor,
        lock.witness,
        lock.expectation,
        GenerationRootStage::SyncRegularFile,
    )?;
    validate_bound_content(
        before_content,
        lock.content_witness,
        GenerationRootStage::SyncRegularFile,
    )?;
    sync_file_descriptor(&lock.descriptor)?;
    let (after_content, after) = read_exact_sha256(
        &lock.descriptor,
        before,
        lock.expectation,
        GenerationRootStage::SyncRegularFile,
    )?;
    validate_bound_content(
        after_content,
        lock.content_witness,
        GenerationRootStage::SyncRegularFile,
    )?;
    revalidate_lock_route(lock)?;
    let (revalidated_content, _) = read_exact_sha256(
        &lock.descriptor,
        after,
        lock.expectation,
        GenerationRootStage::SyncRegularFile,
    )?;
    validate_bound_content(
        revalidated_content,
        lock.content_witness,
        GenerationRootStage::SyncRegularFile,
    )
}

pub(super) fn validate_lock_process(
    lock: &LockHandle,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    validate_process_identity(lock.creator_process_id, stage)
}

pub(super) fn validate_process_identity(
    expected_process_id: u32,
    stage: GenerationRootStage,
) -> GenerationRootResult<()> {
    if expected_process_id == std::process::id() {
        Ok(())
    } else {
        Err(GenerationRootError::new(
            GenerationRootErrorKind::ForkedProcess,
            stage,
        ))
    }
}
