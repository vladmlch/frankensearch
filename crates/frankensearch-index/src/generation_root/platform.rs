#![cfg_attr(not(any(target_os = "linux", target_os = "macos")), allow(dead_code))]

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod ops;
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod types;
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix_common;

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(super) use ops::{
    admit_root, anchor_images_match, capture_anchor_image, control_witness, parse_relative,
    read_lock_image, relative_component_count, revalidate_anchor_set, root_witness, sync_file,
    sync_root_directory, unlock, validate_anchor_image, write_control_range, write_lock_range,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(super) use types::{
    AnchorImage, ControlHandle, FileHandle, LockHandle, RelativePath, RootHandle,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(super) use unix_common::inventory;

#[cfg(target_os = "linux")]
pub(super) use ops::{admit_control_file, admit_file, try_anchor_lock};

#[cfg(target_os = "macos")]
pub(super) use ops::{
    admit_preopened_control_file, admit_preopened_file, try_anchor_lock_preopened,
};

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
pub(super) use ops::{
    control_bytes, control_sha256, lock_bytes, lock_mode, lock_sha256, lock_witness,
    set_root_creator_process_id_for_test, sync_lock_file,
};
#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
pub(super) use types::{TestBoundary, crash_point, install_test_hook, test_boundary};

#[cfg(all(test, target_os = "linux"))]
pub(super) use linux::{
    linux_device_components_match_for_test, linux_filesystem_namespace_digest,
    linux_filesystem_profile, linux_mountinfo_error_for_test, linux_mountinfo_filesystem_for_test,
    linux_open_error_for_test, linux_task_mountinfo_route_for_test, parse_mountinfo_filesystem,
    proc_bridge_error_for_test, proc_root_error_for_test, proc_thread_fd_route_for_test,
    validate_linux_writable_attributes_for_test, validate_mountinfo_bound_for_test,
    validate_proc_mount_id_mask_for_test, validate_regular_descriptor_type,
};
#[cfg(all(test, target_os = "linux"))]
pub(super) use ops::{
    duplicate_lock_descriptor_for_inheritance_for_test, revalidate_root,
    set_lock_creator_process_id_for_test, try_lock,
};
#[cfg(all(test, target_os = "linux"))]
pub(super) use types::{ProcBridgeProbePhase, ProcSelfBindingPhase};
#[cfg(all(test, target_os = "linux"))]
pub(super) use unix_common::{
    stat_inode_as_u64, stat_link_count_as_u64, stat_mode_as_u32, stat_seconds_as_i64,
    validate_absolute_ancestor_owner_mode_for_test, validate_bound_control_content_for_test,
    validate_directory_scan_counts_for_test, validate_owner_mode_for_test,
    verify_exact_directory_entry,
};

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
pub(super) use macos::{stat_device_as_u64_for_test, validate_macos_object_flags_for_test};
#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
pub(super) use ops::{set_control_root_creator_process_id_for_test, try_lock_preopened};

// Typed zero-I/O fallback stubs for unsupported targets.

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
use super::{
    GenerationFileExpectation, GenerationRootEntry, GenerationRootError, GenerationRootErrorKind,
    GenerationRootLockMode, GenerationRootObjectWitness, GenerationRootResult, GenerationRootStage,
};
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
use std::path::Path;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[derive(Clone)]
pub(super) struct RelativePath {
    component_count: usize,
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) struct RootHandle {
    witness: GenerationRootObjectWitness,
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) struct FileHandle {
    _private: (),
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) struct ControlHandle {
    witness: GenerationRootObjectWitness,
    bytes: std::sync::Arc<[u8]>,
    sha256: [u8; 32],
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) struct LockHandle {
    witness: GenerationRootObjectWitness,
    bytes: std::sync::Arc<[u8]>,
    sha256: [u8; 32],
    mode: GenerationRootLockMode,
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) struct AnchorImage {
    pub(super) bytes: std::sync::Arc<[u8]>,
    pub(super) witness: GenerationRootObjectWitness,
    pub(super) sha256: [u8; 32],
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) struct FileAdmission {
    pub(super) handle: FileHandle,
    pub(super) bytes: std::sync::Arc<[u8]>,
    pub(super) witness: GenerationRootObjectWitness,
    pub(super) sha256: [u8; 32],
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
const fn unsupported<T>() -> GenerationRootResult<T> {
    Err(GenerationRootError::new(
        GenerationRootErrorKind::UnsupportedPlatform,
        GenerationRootStage::PlatformGate,
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn parse_relative(_path: &Path) -> GenerationRootResult<RelativePath> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) const fn relative_component_count(path: &RelativePath) -> usize {
    path.component_count
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn admit_root(_path: &Path) -> GenerationRootResult<RootHandle> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) const fn root_witness(root: &RootHandle) -> GenerationRootObjectWitness {
    root.witness
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn revalidate_root(_root: &RootHandle) -> GenerationRootResult<()> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn admit_file(
    _root: &RootHandle,
    _path: &RelativePath,
    _expectation: GenerationFileExpectation,
) -> GenerationRootResult<FileAdmission> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn sync_root_directory(_root: &RootHandle) -> GenerationRootResult<()> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn inventory(_root: &RootHandle) -> GenerationRootResult<Vec<GenerationRootEntry>> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn sync_file(
    _file: &FileHandle,
    _expected: GenerationRootObjectWitness,
) -> GenerationRootResult<()> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn admit_control_file(
    _root: &RootHandle,
    _path: &RelativePath,
    _expectation: GenerationFileExpectation,
) -> GenerationRootResult<ControlHandle> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) const fn control_witness(control: &ControlHandle) -> GenerationRootObjectWitness {
    control.witness
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn control_bytes(control: &ControlHandle) -> std::sync::Arc<[u8]> {
    std::sync::Arc::clone(&control.bytes)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) const fn control_sha256(control: &ControlHandle) -> [u8; 32] {
    control.sha256
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn revalidate_anchor_set(
    _root: &RootHandle,
    _lock: &ControlHandle,
    _authority: &ControlHandle,
) -> GenerationRootResult<()> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn capture_anchor_image(_control: &ControlHandle) -> GenerationRootResult<AnchorImage> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn anchor_images_match(left: &AnchorImage, right: &AnchorImage) -> bool {
    left.sha256 == right.sha256 && left.bytes == right.bytes && left.witness == right.witness
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn validate_anchor_image(
    _control: &ControlHandle,
    _expected: &AnchorImage,
) -> GenerationRootResult<()> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn try_anchor_lock(
    _control: &ControlHandle,
    _mode: GenerationRootLockMode,
) -> GenerationRootResult<LockHandle> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn try_lock(
    _control: &ControlHandle,
    _mode: GenerationRootLockMode,
) -> GenerationRootResult<LockHandle> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) const fn lock_witness(lock: &LockHandle) -> GenerationRootObjectWitness {
    lock.witness
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn lock_bytes(lock: &LockHandle) -> std::sync::Arc<[u8]> {
    std::sync::Arc::clone(&lock.bytes)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) const fn lock_sha256(lock: &LockHandle) -> [u8; 32] {
    lock.sha256
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) const fn lock_mode(lock: &LockHandle) -> GenerationRootLockMode {
    lock.mode
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn unlock(lock: LockHandle) -> GenerationRootResult<()> {
    drop(lock.bytes);
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn sync_lock_file(_lock: &LockHandle) -> GenerationRootResult<()> {
    unsupported()
}

// The authority publisher is platform-independent code, so the fallback
// module must carry every entry point it names, not only the ones the
// root-admission path uses. Each is unreachable — a `RootHandle` can only
// come from `admit_root`, which fails closed with `UnsupportedPlatform` —
// and all handle fields are private to this module. The inhabited layouts
// keep borrowed accessors valid without manufacturing an admitted handle.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn write_control_range(
    _control: &ControlHandle,
    _offset: u64,
    _bytes: &[u8],
) -> GenerationRootResult<()> {
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[expect(
    clippy::needless_pass_by_ref_mut,
    reason = "Matches the native mutable lock contract; unsupported platforms never admit a lock"
)]
pub(super) fn write_lock_range(
    lock: &mut LockHandle,
    _offset: u64,
    _bytes: &[u8],
) -> GenerationRootResult<()> {
    let _ = lock;
    unsupported()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn read_lock_image(_lock: &LockHandle) -> GenerationRootResult<Vec<u8>> {
    unsupported()
}
