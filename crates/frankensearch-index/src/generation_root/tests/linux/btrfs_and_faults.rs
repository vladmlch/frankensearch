use super::super::super::platform::{
    ProcBridgeProbePhase, ProcSelfBindingPhase, TestBoundary, install_test_hook,
    linux_filesystem_namespace_digest, parse_mountinfo_filesystem, stat_inode_as_u64,
    stat_link_count_as_u64, stat_mode_as_u32, stat_seconds_as_i64,
    validate_directory_scan_counts_for_test, validate_mountinfo_bound_for_test,
};
use super::super::super::{
    GENERATION_ROOT_MAX_DIRECTORY_ENTRIES, GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES,
    GENERATION_ROOT_MAX_MOUNTINFO_BYTES, GenerationFileExpectation, GenerationRootAnchorLayout,
    GenerationRootErrorKind, GenerationRootLockMode, GenerationRootStage,
    QualifiedGenerationDirectory as QualifiedGenerationRoot,
    QualifiedGenerationRoot as NamedGenerationRoot,
};
use super::*;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

#[test]
#[ignore = "requires FRANKENSEARCH_LINUX_CHATTR_TEST_ROOT on native ext4/Btrfs, chattr, and CAP_LINUX_IMMUTABLE"]
fn physical_statx_immutable_and_append_attributes_fail_before_data_open_or_lock() {
    use rustix::fs::{AtFlags, StatxAttributes, StatxFlags, statx};

    let base = std::env::var_os("FRANKENSEARCH_LINUX_CHATTR_TEST_ROOT")
        .map(PathBuf::from)
        .expect("FRANKENSEARCH_LINUX_CHATTR_TEST_ROOT must identify native ext4 or Btrfs");
    assert!(
        base.is_absolute(),
        "the physical attribute fixture route must be absolute"
    );
    let filesystem = rustix::fs::statfs(&base).expect("physical attribute mount must be statable");
    assert!(
        i128::from(filesystem.f_type) == i128::from(libc::EXT4_SUPER_MAGIC)
            || i128::from(filesystem.f_type) == i128::from(libc::BTRFS_SUPER_MAGIC),
        "the receipt must execute on native ext4 or Btrfs"
    );

    let root_path = fixture_root_under(&base, "physical-statx-attributes");
    let immutable_contents = b"immutable physical attribute";
    let immutable_path = private_file(&root_path, "immutable", immutable_contents);
    let append_path = control_file(&root_path, "append-control", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path)
        .expect("the writable native root must qualify before object attributes are set");

    run_chattr_fixture_command("+i", &immutable_path);
    let mut immutable_reset = ChattrReset::new("-i", &immutable_path);
    run_chattr_fixture_command("+a", &append_path);
    let mut append_reset = ChattrReset::new("-a", &append_path);

    for (path, attribute) in [
        (&immutable_path, StatxAttributes::IMMUTABLE),
        (&append_path, StatxAttributes::APPEND),
    ] {
        let descriptor =
            File::open(path).expect("attributed fixture must remain readable for proof");
        let observed = statx(
            &descriptor,
            "",
            AtFlags::EMPTY_PATH | AtFlags::NO_AUTOMOUNT,
            StatxFlags::BASIC_STATS,
        )
        .expect("physical attributed descriptor must support statx");
        assert!(
            observed.stx_attributes_mask.contains(attribute),
            "the kernel must report support for the tested physical attribute"
        );
        assert!(
            observed.stx_attributes.contains(attribute),
            "the physical attribute must be set before the rejection receipt"
        );
    }

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let immutable_error = root
        .admit_file(
            &confined("immutable"),
            immutable_expectation(immutable_contents),
        )
        .expect_err("a physically immutable artifact must fail writable qualification");
    assert_eq!(
        immutable_error.kind(),
        GenerationRootErrorKind::WriteRestrictedObject
    );
    assert_eq!(
        immutable_error.stage(),
        GenerationRootStage::OpenRegularFile
    );
    let append_error = root
        .admit_control_file(
            &confined("append-control"),
            GenerationFileExpectation::control(8)
                .expect("small control expectation must be accepted"),
        )
        .expect_err("a physically append-only control file must fail qualification");
    assert_eq!(
        append_error.kind(),
        GenerationRootErrorKind::WriteRestrictedObject
    );
    assert_eq!(append_error.stage(), GenerationRootStage::OpenRegularFile);
    let observed = boundaries.lock().expect("boundary log should lock");
    // Root revalidation legitimately opens the verified procfs root
    // to bind task-scoped mountinfo before inspecting the attributed
    // component. `BeforeProcFdReopen` is the distinct boundary that
    // would derive a data descriptor from the qualified `O_PATH`
    // probe.
    assert!(
        !observed.iter().any(|boundary| matches!(
            boundary,
            TestBoundary::BeforeProcFdReopen { .. }
                | TestBoundary::BeforeRead { .. }
                | TestBoundary::BeforeLock
        )),
        "physical immutable/append rejection must precede data-fd derivation, reads, and \
         flock; observed={observed:?}"
    );
    drop(observed);
    drop(guard);

    immutable_reset.clear();
    append_reset.clear();
    root.admit_file(
        &confined("immutable"),
        immutable_expectation(immutable_contents),
    )
    .expect("the same artifact must qualify after the immutable attribute is cleared");
    root.admit_control_file(
        &confined("append-control"),
        GenerationFileExpectation::control(8).expect("small control expectation must be accepted"),
    )
    .expect("the same control must qualify after the append-only attribute is cleared")
    .try_lock(GenerationRootLockMode::Exclusive)
    .expect("the cleared control file must reach flock")
    .unlock()
    .expect("the cleared control lock must release");
}

#[test]
fn stat_field_normalization_is_lossless_across_32_and_64_bit_abi_widths() {
    assert_eq!(stat_mode_as_u32(u16::MAX), u32::from(u16::MAX));
    assert_eq!(stat_mode_as_u32(u32::MAX), u32::MAX);
    assert_eq!(stat_inode_as_u64(u32::MAX), u64::from(u32::MAX));
    assert_eq!(stat_inode_as_u64(u64::MAX), u64::MAX);
    assert_eq!(stat_link_count_as_u64(u32::MAX), u64::from(u32::MAX));
    assert_eq!(stat_link_count_as_u64(u64::MAX), u64::MAX);
    assert_eq!(stat_seconds_as_i64(i32::MIN), i64::from(i32::MIN));
    assert_eq!(stat_seconds_as_i64(i32::MAX), i64::from(i32::MAX));
    assert_eq!(stat_seconds_as_i64(i64::MIN), i64::MIN);
    assert_eq!(stat_seconds_as_i64(i64::MAX), i64::MAX);
}

#[test]
fn btrfs_namespace_identity_binds_mount_and_subvolume() {
    let raw_type: rustix::fs::FsWord = libc::BTRFS_SUPER_MAGIC;
    let top_level = linux_filesystem_namespace_digest(raw_type, 41, Some(5));
    let nested = linux_filesystem_namespace_digest(raw_type, 41, Some(256));
    let other_mount = linux_filesystem_namespace_digest(raw_type, 42, Some(5));

    assert_ne!(
        top_level, nested,
        "one VFS mount must not collapse distinct Btrfs subvolumes"
    );
    assert_ne!(
        top_level, other_mount,
        "the namespace identity must continue to bind the VFS mount"
    );
    assert_eq!(
        top_level,
        linux_filesystem_namespace_digest(raw_type, 41, Some(5)),
        "the namespace identity must be deterministic"
    );
}

#[test]
#[ignore = "requires FRANKENSEARCH_LINUX_ABSOLUTE_MOUNT_TEST_ROOT plus mount, umount, and CAP_SYS_ADMIN"]
fn physical_absolute_route_accepts_bind_transition_and_rejects_same_source_remount() {
    let base = std::env::var_os("FRANKENSEARCH_LINUX_ABSOLUTE_MOUNT_TEST_ROOT")
        .map(PathBuf::from)
        .expect(
            "FRANKENSEARCH_LINUX_ABSOLUTE_MOUNT_TEST_ROOT must identify a secure native ext4/Btrfs ancestor",
        );
    assert!(
        base.is_absolute(),
        "the physical absolute-route receipt must use an absolute trusted ancestor"
    );

    let receipt_root = fixture_root_under(&base, "physical-absolute-bind-transition");
    let source = private_dir(&receipt_root, "source");
    let source_root = private_dir(&source, "qualified");
    let lock_bytes = b"lock-v1!";
    control_file(&source_root, "LOCK", lock_bytes);
    control_file(&source_root, "AUTHORITY", &authority_image(0x5a));
    let layout = GenerationRootAnchorLayout::new(
        u64::try_from(lock_bytes.len()).expect("the lock fixture length fits u64"),
    )
    .expect("the named lock fixture length is bounded");
    let target = private_dir(&receipt_root, "mountpoint");
    let bind_mount = BindMountFixture::new(&source, &target);
    let mounted_root = target.join("qualified");

    let direct = QualifiedGenerationRoot::admit(&source_root)
        .expect("the source root should independently qualify");
    let through_bind = QualifiedGenerationRoot::admit(&mounted_root)
        .expect("an absolute route may cross into a qualified adjacent bind mount");
    assert_eq!(
        direct.witness().inode(),
        through_bind.witness().inode(),
        "the bind route must resolve the same underlying final directory"
    );
    assert_ne!(
        direct.witness().mount_identity(),
        through_bind.witness().mount_identity(),
        "the route witness must distinguish the adjacent bind mount"
    );
    let named = NamedGenerationRoot::admit(&mounted_root, layout)
        .expect("fixed anchors should qualify through the adjacent bind mount");
    named
        .read_guard()
        .expect("the named LOCK should acquire through the qualified mount transition")
        .release()
        .expect("the unchanged mounted guard should release");
    drop(named);
    drop(through_bind);
    drop(direct);

    let swapped = Arc::new(Mutex::new(false));
    let hook_swapped = Arc::clone(&swapped);
    let mut hook_mount = bind_mount;
    let hook = install_test_hook(move |boundary| {
        if !*hook_swapped.lock().expect("mount-swap state should lock")
            && boundary == TestBoundary::AfterFirstRootQualification
        {
            hook_mount.remount_same_source();
            *hook_swapped.lock().expect("mount-swap state should lock") = true;
        }
        Ok(())
    });
    let error = NamedGenerationRoot::admit(&mounted_root, layout)
        .expect_err("a new mount identity between the two absolute walks must fail");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::RevalidateRootRoute);
    assert!(
        *swapped.lock().expect("mount-swap state should lock"),
        "the physical receipt must execute the remount boundary"
    );
    drop(hook);
}

#[test]
#[ignore = "requires FRANKENSEARCH_LINUX_NETWORK_TEST_ROOT on a writable native NFS mount"]
fn physical_linux_network_mount_fails_before_data_open_or_mutation() {
    let network_root = std::env::var_os("FRANKENSEARCH_LINUX_NETWORK_TEST_ROOT")
        .map(PathBuf::from)
        .expect("FRANKENSEARCH_LINUX_NETWORK_TEST_ROOT must identify the native NFS mount");
    assert!(
        network_root.is_absolute(),
        "the physical network receipt must use an absolute provider route"
    );
    let filesystem =
        rustix::fs::statfs(&network_root).expect("the physical network mount must be statable");
    assert_eq!(
        i128::from(filesystem.f_type),
        i128::from(libc::NFS_SUPER_MAGIC),
        "the provider route must be a real NFS mount"
    );
    let mount = rustix::fs::statvfs(&network_root)
        .expect("the physical network mount flags must be readable");
    assert!(
        !mount.f_flag.contains(rustix::fs::StatVfsMountFlags::RDONLY),
        "the provider NFS mount must be writable so this receipt isolates filesystem-type rejection"
    );

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let _hook = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("network boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let error = QualifiedGenerationRoot::admit(&network_root)
        .expect_err("a network filesystem must fail qualification");
    assert_eq!(error.kind(), GenerationRootErrorKind::UnsupportedFilesystem);
    assert_eq!(error.stage(), GenerationRootStage::QualifyFilesystem);

    let observed = boundaries.lock().expect("network boundary log should lock");
    assert!(
        observed.contains(&TestBoundary::BeforeFilesystemQualification),
        "the receipt must reach the descriptor-bound filesystem classifier"
    );
    assert!(
        !observed.contains(&TestBoundary::AfterFilesystemQualification),
        "an NFS root must not cross the qualification boundary"
    );
    assert!(
        !observed.iter().any(|boundary| matches!(
            boundary,
            TestBoundary::BeforeRegularFileOpen { .. }
                | TestBoundary::BeforeRead { .. }
                | TestBoundary::BeforeLock
                | TestBoundary::BeforeFileSync
                | TestBoundary::BeforeDirectorySync
        )),
        "NFS rejection must precede every regular-file, content, lock, and durability boundary"
    );
    drop(observed);
}

#[test]
#[ignore = "requires FRANKENSEARCH_BTRFS_TEST_ROOT on a writable native Btrfs mount"]
fn physical_btrfs_retained_files_locks_and_durability_succeed() {
    let base = std::env::var_os("FRANKENSEARCH_BTRFS_TEST_ROOT")
        .map(PathBuf::from)
        .expect("FRANKENSEARCH_BTRFS_TEST_ROOT must identify the native Btrfs mount");
    assert!(
        base.is_absolute(),
        "the physical Btrfs mount route must be absolute"
    );
    let filesystem = rustix::fs::statfs(&base).expect("physical Btrfs mount must be statable");
    assert_eq!(
        i128::from(filesystem.f_type),
        i128::from(libc::BTRFS_SUPER_MAGIC),
        "the receipt must execute on a real Btrfs mount"
    );

    let root_path = fixture_root_under(&base, "physical-btrfs");
    let generation = private_dir(&root_path, "generation");
    let contents = b"physical Btrfs generation";
    private_file(&generation, "vector.fsvi", contents);
    control_file(&root_path, "LOCK", b"lock-v1!");

    let root =
        QualifiedGenerationRoot::admit(&root_path).expect("writable Btrfs root must qualify");
    assert_eq!(
        root.witness().filesystem(),
        crate::generation_root::QualifiedFilesystem::LinuxBtrfs
    );
    let admitted = root
        .admit_file(
            &confined("generation/vector.fsvi"),
            immutable_expectation(contents),
        )
        .expect("exact immutable Btrfs file must qualify");
    assert_eq!(admitted.as_bytes(), contents);
    admitted
        .sync_durable()
        .expect("retained Btrfs regular-file fsync must succeed");

    let control = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8)
                .expect("small control expectation must be accepted"),
        )
        .expect("Btrfs control file must qualify");
    let lock = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("Btrfs control-file flock must succeed");
    lock.sync_durable()
        .expect("locked Btrfs control-file fsync must succeed");
    lock.unlock()
        .expect("Btrfs control-file unlock must succeed");
    root.sync_directory_durable()
        .expect("retained Btrfs directory fsync must succeed");
}

#[test]
#[ignore = "requires FRANKENSEARCH_BTRFS_TEST_ROOT, btrfs-progs, and CAP_SYS_ADMIN"]
fn physical_btrfs_nested_subvolume_is_a_confined_namespace_boundary() {
    let base = std::env::var_os("FRANKENSEARCH_BTRFS_TEST_ROOT")
        .map(PathBuf::from)
        .expect("FRANKENSEARCH_BTRFS_TEST_ROOT must identify the native Btrfs mount");
    let filesystem = rustix::fs::statfs(&base).expect("physical Btrfs mount must be statable");
    assert_eq!(
        i128::from(filesystem.f_type),
        i128::from(libc::BTRFS_SUPER_MAGIC),
        "the receipt must execute on a real Btrfs mount"
    );

    let root_path = fixture_root_under(&base, "physical-btrfs-nested-subvolume");
    let nested_path = root_path.join("nested");
    run_btrfs_fixture_command(&[
        OsStr::new("subvolume"),
        OsStr::new("create"),
        nested_path.as_os_str(),
    ]);
    fs::set_permissions(&nested_path, fs::Permissions::from_mode(0o700))
        .expect("nested Btrfs subvolume mode must be private");
    let contents = b"nested subvolume must not cross the root namespace";
    private_file(&nested_path, "vector.fsvi", contents);

    let root = QualifiedGenerationRoot::admit(&root_path)
        .expect("the containing writable Btrfs root must qualify");
    assert_eq!(
        root.admit_file(
            &confined("nested/vector.fsvi"),
            immutable_expectation(contents),
        )
        .expect_err("a nested Btrfs subvolume must be a confinement boundary")
        .kind(),
        GenerationRootErrorKind::CrossDevice
    );
}

#[test]
#[ignore = "requires FRANKENSEARCH_BTRFS_TEST_ROOT, btrfs-progs, and CAP_SYS_ADMIN"]
fn physical_btrfs_snapshot_substitution_invalidates_the_retained_route() {
    let base = std::env::var_os("FRANKENSEARCH_BTRFS_TEST_ROOT")
        .map(PathBuf::from)
        .expect("FRANKENSEARCH_BTRFS_TEST_ROOT must identify the native Btrfs mount");
    let filesystem = rustix::fs::statfs(&base).expect("physical Btrfs mount must be statable");
    assert_eq!(
        i128::from(filesystem.f_type),
        i128::from(libc::BTRFS_SUPER_MAGIC),
        "the receipt must execute on a real Btrfs mount"
    );

    let parent_path = fixture_root_under(&base, "physical-btrfs-snapshot-substitution");
    let live_path = parent_path.join("live");
    run_btrfs_fixture_command(&[
        OsStr::new("subvolume"),
        OsStr::new("create"),
        live_path.as_os_str(),
    ]);
    fs::set_permissions(&live_path, fs::Permissions::from_mode(0o700))
        .expect("live Btrfs subvolume mode must be private");
    private_file(&live_path, "manifest", b"sealed generation");

    let retained = QualifiedGenerationRoot::admit(&live_path)
        .expect("the original Btrfs subvolume must qualify");
    let retained_path = parent_path.join("retained");
    fs::rename(&live_path, &retained_path)
        .expect("the original subvolume must move to an unused persistent route");
    run_btrfs_fixture_command(&[
        OsStr::new("subvolume"),
        OsStr::new("snapshot"),
        retained_path.as_os_str(),
        live_path.as_os_str(),
    ]);

    let replacement = QualifiedGenerationRoot::admit(&live_path)
        .expect("the replacement snapshot is independently well-formed");
    assert_ne!(
        retained.witness().mount_identity(),
        replacement.witness().mount_identity(),
        "snapshot and source must have distinct namespace identities"
    );
    assert_eq!(
        retained
            .revalidate_route()
            .expect_err("snapshot substitution must invalidate the retained route")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
#[ignore = "requires FRANKENSEARCH_BTRFS_TEST_ROOT, btrfs-progs, and CAP_SYS_ADMIN"]
fn physical_read_only_btrfs_subvolume_fails_on_a_writable_mount() {
    use rustix::fs::StatVfsMountFlags;

    let base = std::env::var_os("FRANKENSEARCH_BTRFS_TEST_ROOT")
        .map(PathBuf::from)
        .expect("FRANKENSEARCH_BTRFS_TEST_ROOT must identify the native Btrfs mount");
    let filesystem = rustix::fs::statfs(&base).expect("physical Btrfs mount must be statable");
    assert_eq!(
        i128::from(filesystem.f_type),
        i128::from(libc::BTRFS_SUPER_MAGIC),
        "the receipt must execute on a real Btrfs mount"
    );

    let parent_path = fixture_root_under(&base, "physical-btrfs-read-only-subvolume");
    let readonly_path = parent_path.join("readonly");
    run_btrfs_fixture_command(&[
        OsStr::new("subvolume"),
        OsStr::new("create"),
        readonly_path.as_os_str(),
    ]);
    fs::set_permissions(&readonly_path, fs::Permissions::from_mode(0o700))
        .expect("read-only fixture mode must be private before sealing");
    run_btrfs_fixture_command(&[
        OsStr::new("property"),
        OsStr::new("set"),
        OsStr::new("-t"),
        OsStr::new("subvol"),
        readonly_path.as_os_str(),
        OsStr::new("ro"),
        OsStr::new("true"),
    ]);
    let readonly_descriptor =
        File::open(&readonly_path).expect("read-only fixture must be openable");
    let write_error = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(readonly_path.join("writability-probe"))
        .expect_err("the receipt requires a genuinely read-only Btrfs subvolume");
    assert_eq!(
        write_error.raw_os_error(),
        Some(libc::EROFS),
        "the read-only proof must be the kernel EROFS result"
    );
    let mount =
        rustix::fs::fstatvfs(&readonly_descriptor).expect("Btrfs mount flags must be readable");
    assert!(
        !mount.f_flag.contains(StatVfsMountFlags::RDONLY),
        "the enclosing VFS mount must remain writable so this proves subvolume handling"
    );
    assert_eq!(
        QualifiedGenerationRoot::admit(&readonly_path)
            .expect_err("read-only Btrfs subvolume must fail admission")
            .kind(),
        GenerationRootErrorKind::ReadOnlyFilesystem
    );
}

#[test]
#[ignore = "requires FRANKENSEARCH_BTRFS_READ_ONLY_TEST_ROOT on a native read-only Btrfs mount"]
fn physical_read_only_btrfs_fails_before_security_admission() {
    use rustix::fs::StatVfsMountFlags;

    let base = std::env::var_os("FRANKENSEARCH_BTRFS_READ_ONLY_TEST_ROOT")
        .map(PathBuf::from)
        .expect("FRANKENSEARCH_BTRFS_READ_ONLY_TEST_ROOT must identify the native Btrfs mount");
    assert!(
        base.is_absolute(),
        "the physical read-only Btrfs mount route must be absolute"
    );
    let filesystem =
        rustix::fs::statfs(&base).expect("physical read-only Btrfs mount must be statable");
    assert_eq!(
        i128::from(filesystem.f_type),
        i128::from(libc::BTRFS_SUPER_MAGIC),
        "the receipt must execute on a real Btrfs mount"
    );
    let descriptor = File::open(&base).expect("physical read-only Btrfs root must be openable");
    let mount = rustix::fs::fstatvfs(&descriptor)
        .expect("physical read-only Btrfs mount flags must be readable");
    assert!(
        mount.f_flag.contains(StatVfsMountFlags::RDONLY),
        "the receipt requires a genuinely read-only mount"
    );
    assert_eq!(
        QualifiedGenerationRoot::admit(&base)
            .expect_err("read-only Btrfs must fail before owner or mode admission")
            .kind(),
        GenerationRootErrorKind::ReadOnlyFilesystem
    );
}

#[test]
fn mountinfo_parser_requires_one_exact_well_formed_mount_record() {
    let fixture = b"41 1 8:1 / /other rw,relatime - xfs /dev/other rw\n\
                    42 1 8:2 / /target rw,relatime shared:7 - ext4 /dev/root rw\n";
    assert_eq!(
        parse_mountinfo_filesystem(fixture, 42).expect("one exact mount-id record should qualify"),
        b"ext4"
    );

    for malformed in [
        b"42 1 8:2 / /target rw,relatime ext4 /dev/root rw\n".as_slice(),
        b"42 1 8:2 / /target rw,relatime -  /dev/root rw\n".as_slice(),
        b"42 1 8:2 / /target rw,relatime - ext4  rw\n".as_slice(),
        b"42 1 8:2 / /target rw,relatime - ext4 /dev/root \n".as_slice(),
    ] {
        assert_eq!(
            parse_mountinfo_filesystem(malformed, 42)
                .expect_err("malformed matching mountinfo record must fail")
                .kind(),
            GenerationRootErrorKind::UnsupportedFilesystem
        );
    }

    let duplicate = b"42 1 8:2 / /a rw - ext4 /dev/a rw\n\
                      42 1 8:2 / /b rw - ext4 /dev/b rw\n";
    assert_eq!(
        parse_mountinfo_filesystem(duplicate, 42)
            .expect_err("duplicate mount-id records must fail")
            .kind(),
        GenerationRootErrorKind::UnsupportedFilesystem
    );
    assert_eq!(
        parse_mountinfo_filesystem(
            b"18446744073709551616 1 8:2 / /target rw - ext4 /dev/root rw\n",
            u64::MAX,
        )
        .expect_err("overflowing mount ids must never alias a valid u64")
        .kind(),
        GenerationRootErrorKind::UnsupportedFilesystem
    );
    assert_eq!(
        parse_mountinfo_filesystem(fixture, 99)
            .expect_err("missing mount-id record must fail")
            .kind(),
        GenerationRootErrorKind::UnsupportedFilesystem
    );
}

#[test]
fn directory_and_mountinfo_resource_boundaries_report_exact_numeric_counts() {
    validate_directory_scan_counts_for_test(
        GENERATION_ROOT_MAX_DIRECTORY_ENTRIES,
        GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES,
    )
    .expect("both exact directory-scan ceilings should be accepted");

    let entry_overflow = validate_directory_scan_counts_for_test(
        GENERATION_ROOT_MAX_DIRECTORY_ENTRIES.saturating_add(1),
        0,
    )
    .expect_err("one overflow entry must fail");
    assert_eq!(
        entry_overflow.kind(),
        GenerationRootErrorKind::ResourceLimit
    );
    assert_eq!(
        entry_overflow.expected(),
        Some(u64::try_from(GENERATION_ROOT_MAX_DIRECTORY_ENTRIES).expect("entry ceiling fits u64"))
    );
    assert_eq!(
        entry_overflow.observed(),
        Some(
            u64::try_from(GENERATION_ROOT_MAX_DIRECTORY_ENTRIES.saturating_add(1))
                .expect("entry sentinel fits u64")
        )
    );

    let name_overflow = validate_directory_scan_counts_for_test(
        1,
        GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES.saturating_add(1),
    )
    .expect_err("one overflow name byte must fail");
    assert_eq!(name_overflow.kind(), GenerationRootErrorKind::ResourceLimit);
    assert_eq!(
        name_overflow.expected(),
        Some(
            u64::try_from(GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES)
                .expect("name-byte ceiling fits u64")
        )
    );
    assert_eq!(
        name_overflow.observed(),
        Some(
            u64::try_from(GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES.saturating_add(1))
                .expect("name-byte sentinel fits u64")
        )
    );

    let (last_request, accepted) =
        validate_mountinfo_bound_for_test(GENERATION_ROOT_MAX_MOUNTINFO_BYTES - 1, 1);
    assert_eq!(last_request, 2, "one accepted byte plus one sentinel");
    accepted.expect("the exact mountinfo storage ceiling should be accepted");
    let (sentinel_request, overflow) =
        validate_mountinfo_bound_for_test(GENERATION_ROOT_MAX_MOUNTINFO_BYTES, 1);
    assert_eq!(sentinel_request, 1, "only an overflow sentinel may be read");
    let overflow = overflow.expect_err("the mountinfo sentinel byte must fail");
    assert_eq!(overflow.kind(), GenerationRootErrorKind::ResourceLimit);
    assert_eq!(
        overflow.expected(),
        Some(
            u64::try_from(GENERATION_ROOT_MAX_MOUNTINFO_BYTES).expect("mountinfo ceiling fits u64")
        )
    );
    assert_eq!(
        overflow.observed(),
        Some(
            u64::try_from(GENERATION_ROOT_MAX_MOUNTINFO_BYTES.saturating_add(1))
                .expect("mountinfo sentinel fits u64")
        )
    );
}

#[test]
fn every_linux_open_read_lock_and_sync_boundary_is_fault_injectable() {
    for target in [
        TestBoundary::BeforeRootComponentOpen { index: 0 },
        TestBoundary::AfterRootComponentOpen { index: 0 },
        TestBoundary::BeforeExactNameEnumeration,
        TestBoundary::AfterExactNameEnumeration,
        TestBoundary::BeforeRouteMountIdentity,
        TestBoundary::AfterRouteMountIdentity,
        TestBoundary::BeforeFilesystemQualification,
        TestBoundary::AfterFilesystemQualification,
        TestBoundary::BeforeObjectWitness,
        TestBoundary::AfterObjectWitness,
        TestBoundary::BeforeProcCapabilityRootOpen,
        TestBoundary::AfterProcCapabilityRootOpen,
        TestBoundary::BeforeProcSelfReadlink {
            phase: ProcSelfBindingPhase::RootQualification,
        },
        TestBoundary::AfterProcSelfReadlink {
            phase: ProcSelfBindingPhase::RootQualification,
        },
        TestBoundary::BeforeProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::RootQualification,
        },
        TestBoundary::AfterProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::RootQualification,
        },
        TestBoundary::BeforeMountInfoOpen,
        TestBoundary::AfterMountInfoOpen,
        TestBoundary::BeforeMountInfoRead,
        TestBoundary::BeforeProcSelfReadlink {
            phase: ProcSelfBindingPhase::MountInfoRead,
        },
        TestBoundary::AfterProcSelfReadlink {
            phase: ProcSelfBindingPhase::MountInfoRead,
        },
        TestBoundary::BeforeProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::MountInfoRead,
        },
        TestBoundary::AfterProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::MountInfoRead,
        },
        TestBoundary::AfterFirstRootQualification,
        TestBoundary::BeforeAclRead,
        TestBoundary::AfterAclRead,
    ] {
        let root_path = fixture_root("root-boundary");
        let _guard = install_test_hook(move |boundary| {
            if boundary == target {
                return Err(injected(GenerationRootStage::OpenRootRoute));
            }
            Ok(())
        });
        let error = QualifiedGenerationRoot::admit(&root_path)
            .expect_err("injected root-open boundary must fail");
        assert_eq!(error.kind(), GenerationRootErrorKind::Io);
    }

    let contents = b"boundary";
    for target in [
        TestBoundary::BeforeRootDescriptorDuplicate,
        TestBoundary::AfterRootDescriptorDuplicate,
        TestBoundary::BeforeRelativeDirectoryOpen { index: 0 },
        TestBoundary::AfterRelativeDirectoryOpen { index: 0 },
        TestBoundary::BeforeRegularFileOpen { index: 1 },
        TestBoundary::BeforeFinalProbeOpen { index: 1 },
        TestBoundary::AfterFinalProbeOpen { index: 1 },
        TestBoundary::AfterProbeQualified { index: 1 },
        TestBoundary::BeforeProcCapabilityRootOpen,
        TestBoundary::AfterProcCapabilityRootOpen,
        TestBoundary::BeforeProcSelfReadlink {
            phase: ProcSelfBindingPhase::RootQualification,
        },
        TestBoundary::AfterProcSelfReadlink {
            phase: ProcSelfBindingPhase::RootQualification,
        },
        TestBoundary::BeforeProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::RootQualification,
        },
        TestBoundary::AfterProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::RootQualification,
        },
        TestBoundary::BeforeProcBridgeProbe {
            index: 1,
            phase: ProcBridgeProbePhase::BeforeDataOpen,
        },
        TestBoundary::AfterProcBridgeProbe {
            index: 1,
            phase: ProcBridgeProbePhase::BeforeDataOpen,
        },
        TestBoundary::BeforeProcSelfReadlink {
            phase: ProcSelfBindingPhase::BeforeDataOpen,
        },
        TestBoundary::AfterProcSelfReadlink {
            phase: ProcSelfBindingPhase::BeforeDataOpen,
        },
        TestBoundary::BeforeProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::BeforeDataOpen,
        },
        TestBoundary::AfterProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::BeforeDataOpen,
        },
        TestBoundary::BeforeProcFdReopen { index: 1 },
        TestBoundary::BeforeProcSelfReadlink {
            phase: ProcSelfBindingPhase::AfterDataOpen,
        },
        TestBoundary::AfterProcSelfReadlink {
            phase: ProcSelfBindingPhase::AfterDataOpen,
        },
        TestBoundary::BeforeProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::AfterDataOpen,
        },
        TestBoundary::AfterProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase::AfterDataOpen,
        },
        TestBoundary::BeforeProcBridgeProbe {
            index: 1,
            phase: ProcBridgeProbePhase::AfterDataOpen,
        },
        TestBoundary::AfterProcBridgeProbe {
            index: 1,
            phase: ProcBridgeProbePhase::AfterDataOpen,
        },
        TestBoundary::AfterProcFdReopen { index: 1 },
        TestBoundary::BeforeFinalRouteProbe { index: 1 },
        TestBoundary::AfterFinalRouteProbe { index: 1 },
        TestBoundary::AfterRegularFileOpen {
            index: 1,
            cloexec: true,
        },
        TestBoundary::AfterQualifiedFileOpen,
        TestBoundary::BeforeAclRead,
        TestBoundary::AfterAclRead,
        TestBoundary::BeforeRead { offset: 0 },
        TestBoundary::AfterRead {
            offset: 0,
            byte_count: contents.len(),
        },
        TestBoundary::BeforeTrailingByteProbe,
        TestBoundary::AfterTrailingByteProbe { byte_count: 0 },
        TestBoundary::AfterExactRead,
        TestBoundary::BeforeFileRouteReopen,
        TestBoundary::AfterFileRouteReopen,
        TestBoundary::BeforeFinalRootRevalidation,
    ] {
        let root_path = fixture_root("file-boundary");
        let generation = private_dir(&root_path, "generation");
        private_file(&generation, "artifact", contents);
        let root = QualifiedGenerationRoot::admit(&root_path)
            .expect("private root should qualify before injection");
        let _guard = install_test_hook(move |boundary| {
            if boundary == target {
                return Err(injected(GenerationRootStage::ReadRegularFile));
            }
            Ok(())
        });
        let error = root
            .admit_file(
                &confined("generation/artifact"),
                immutable_expectation(contents),
            )
            .expect_err("injected file boundary must fail");
        assert_eq!(error.kind(), GenerationRootErrorKind::Io);
    }

    {
        let root_path = fixture_root("mountinfo-read-complete-boundary");
        let _guard = install_test_hook(|boundary| {
            if matches!(boundary, TestBoundary::AfterMountInfoRead { .. }) {
                return Err(injected(GenerationRootStage::QualifyFilesystem));
            }
            Ok(())
        });
        assert_eq!(
            QualifiedGenerationRoot::admit(&root_path)
                .expect_err("completed mountinfo read boundary must be injectable")
                .kind(),
            GenerationRootErrorKind::Io
        );
    }

    for target in [TestBoundary::BeforeLock, TestBoundary::AfterLock] {
        let root_path = fixture_root("lock-boundary");
        control_file(&root_path, "LOCK", b"lock-v1!");
        let root = QualifiedGenerationRoot::admit(&root_path)
            .expect("private root should qualify before injection");
        let control = root
            .admit_control_file(
                &confined("LOCK"),
                GenerationFileExpectation::control(8)
                    .expect("small expectation should be accepted"),
            )
            .expect("control file should qualify before lock injection");
        let _guard = install_test_hook(move |boundary| {
            if boundary == target {
                return Err(injected(GenerationRootStage::AcquireLock));
            }
            Ok(())
        });
        let error = control
            .try_lock(GenerationRootLockMode::Exclusive)
            .expect_err("injected lock boundary must fail");
        assert_eq!(error.kind(), GenerationRootErrorKind::Io);
    }

    for target in [TestBoundary::BeforeUnlock, TestBoundary::AfterUnlock] {
        let root_path = fixture_root("unlock-boundary");
        control_file(&root_path, "LOCK", b"lock-v1!");
        let root = QualifiedGenerationRoot::admit(&root_path)
            .expect("private root should qualify before injection");
        let control = root
            .admit_control_file(
                &confined("LOCK"),
                GenerationFileExpectation::control(8)
                    .expect("small expectation should be accepted"),
            )
            .expect("control file should qualify before unlock injection");
        let lock = control
            .try_lock(GenerationRootLockMode::Exclusive)
            .expect("exclusive lock should succeed before unlock injection");
        let _guard = install_test_hook(move |boundary| {
            if boundary == target {
                return Err(injected(GenerationRootStage::ReleaseLock));
            }
            Ok(())
        });
        assert_eq!(
            lock.unlock()
                .expect_err("injected unlock boundary must fail")
                .kind(),
            GenerationRootErrorKind::Io
        );
    }

    for target in [TestBoundary::BeforeFileSync, TestBoundary::AfterFileSync] {
        let root_path = fixture_root("file-sync-boundary");
        private_file(&root_path, "artifact", contents);
        let root = QualifiedGenerationRoot::admit(&root_path)
            .expect("private root should qualify before injection");
        let file = root
            .admit_file(&confined("artifact"), immutable_expectation(contents))
            .expect("file should qualify before sync injection");
        let _guard = install_test_hook(move |boundary| {
            if boundary == target {
                return Err(injected(GenerationRootStage::SyncRegularFile));
            }
            Ok(())
        });
        assert_eq!(
            file.sync_durable()
                .expect_err("injected file-sync boundary must fail")
                .kind(),
            GenerationRootErrorKind::Io
        );
    }

    for target in [
        TestBoundary::BeforeDirectorySync,
        TestBoundary::AfterDirectorySync,
    ] {
        let root_path = fixture_root("directory-sync-boundary");
        let root = QualifiedGenerationRoot::admit(&root_path)
            .expect("private root should qualify before injection");
        let _guard = install_test_hook(move |boundary| {
            if boundary == target {
                return Err(injected(GenerationRootStage::SyncDirectory));
            }
            Ok(())
        });
        assert_eq!(
            root.sync_directory_durable()
                .expect_err("injected directory-sync boundary must fail")
                .kind(),
            GenerationRootErrorKind::Io
        );
    }
}

#[test]
fn exact_path_reopen_detects_same_size_inode_substitution() {
    let root_path = fixture_root("inode-substitution");
    let original = private_file(&root_path, "artifact", b"original");
    let replacement_source = private_file(&root_path, "replacement", b"attacker");
    let retained = root_path.join("retained-original");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let observed = Arc::new(Mutex::new(false));
    let hook_observed = Arc::clone(&observed);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeFileRouteReopen {
            fs::rename(&original, &retained)
                .expect("original should move to a fresh retained path");
            fs::rename(&replacement_source, &original)
                .expect("replacement should move into the original path");
            *hook_observed.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    assert_eq!(
        root.admit_file(&confined("artifact"), immutable_expectation(b"original"))
            .expect_err("same-size inode substitution must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
    assert!(*observed.lock().expect("hook state should lock"));
}

#[test]
fn root_open_fault_does_not_modify_decoy_tree() {
    let root_path = fixture_root("root-fault-decoy");
    let decoy_path = private_file(&root_path, "decoy", b"unchanged");
    let _guard = install_test_hook(|boundary| {
        if boundary == (TestBoundary::BeforeRootComponentOpen { index: 0 }) {
            return Err(injected(GenerationRootStage::OpenRootRoute));
        }
        Ok(())
    });
    assert!(QualifiedGenerationRoot::admit(&root_path).is_err());
    assert_eq!(
        fs::read(decoy_path).expect("decoy should remain readable"),
        b"unchanged"
    );
}

#[test]
fn existing_file_descriptor_is_never_replaced_by_ambient_path_changes() {
    let root_path = fixture_root("retained-descriptor");
    let contents = b"descriptor-owned";
    let artifact = private_file(&root_path, "artifact", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(&confined("artifact"), immutable_expectation(contents))
        .expect("file should qualify");
    let moved = root_path.join("artifact-moved");
    fs::rename(&artifact, &moved).expect("artifact should move to a fresh path");
    private_file(&root_path, "artifact", b"ambient-changed!");
    assert_eq!(admitted.as_bytes(), contents);
    assert_eq!(admitted.sha256(), digest(contents));
    assert_eq!(
        admitted
            .sync_durable()
            .expect_err("same-size route substitution must fence retained-file sync")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn regular_file_open_rejects_missing_path_without_path_disclosure() {
    let root_path = fixture_root("missing");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let error = root
        .admit_file(
            &confined("sensitive-missing-name"),
            GenerationFileExpectation::immutable(0, digest(b""))
                .expect("zero-length expectation should be accepted"),
        )
        .expect_err("missing file must fail");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert!(!error.to_string().contains("sensitive-missing-name"));
    assert!(!format!("{error:?}").contains("sensitive-missing-name"));
}

#[test]
fn retained_descriptor_sync_detects_post_admission_content_change() {
    let root_path = fixture_root("sync-drift");
    let artifact = private_file(&root_path, "artifact", b"before!!");
    fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
        .expect("fixture should become writable for the retained test fd");
    let mut mutation_file = OpenOptions::new()
        .write(true)
        .open(&artifact)
        .expect("retained mutation fd should open");
    fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
        .expect("fixture should be resealed before admission");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(&confined("artifact"), immutable_expectation(b"before!!"))
        .expect("file should qualify");
    mutation_file
        .write_all(b"after!!!")
        .expect("same-size mutation should succeed");
    mutation_file
        .sync_all()
        .expect("mutation should be durable");
    assert_eq!(
        admitted
            .sync_durable()
            .expect_err("post-admission mutation must fence durability")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn linux_atime_only_drift_is_excluded_without_restoring_metadata() {
    let root_path = fixture_root("atime-drift");
    // The fixture below proves that an ambient (non-O_NOATIME) read advances atime and
    // that admission neither trips on it nor rewinds it. On a `noatime` mount the kernel
    // never advances atime for anyone, so the precondition cannot be staged and the
    // drift under test does not exist; report the host as not applicable rather than
    // asserting a property of the developer's mount options. Everything else about the
    // filesystem is left to the assertions below.
    let mount = rustix::fs::statvfs(&root_path).expect("fixture mount flags should be readable");
    if mount
        .f_flag
        .contains(rustix::fs::StatVfsMountFlags::NOATIME)
    {
        eprintln!(
            "skipping linux_atime_only_drift_is_excluded_without_restoring_metadata: \
             the fixture root is on a noatime mount, so an ambient read cannot \
             advance atime and atime-only drift cannot be staged here"
        );
        return;
    }
    let artifact = private_file(&root_path, "artifact", b"atime");
    let timestamp_owner = File::open(&artifact).expect("fixture should open for timestamp");
    let modified = timestamp_owner
        .metadata()
        .and_then(|metadata| metadata.modified())
        .expect("fixture mtime should be readable");
    timestamp_owner
        .set_times(
            std::fs::FileTimes::new()
                .set_accessed(UNIX_EPOCH)
                .set_modified(modified),
        )
        .expect("fixture atime should become observably old before admission");

    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(&confined("artifact"), immutable_expectation(b"atime"))
        .expect("file should qualify while O_NOATIME preserves the old atime");
    let before = fs::metadata(&artifact)
        .expect("fixture metadata should load")
        .accessed()
        .expect("fixture atime should be readable");
    let mut ambient_reader = File::open(&artifact).expect("ambient reader should open");
    let mut byte = [0_u8; 1];
    ambient_reader
        .read_exact(&mut byte)
        .expect("ambient read should advance atime");
    let after = fs::metadata(&artifact)
        .expect("fixture metadata should reload")
        .accessed()
        .expect("advanced atime should be readable");
    assert!(
        after > before,
        "fixture must prove an atime-only advancement"
    );
    admitted
        .sync_durable()
        .expect("Linux atime-only drift is not a content mutation");
    assert_eq!(
        fs::metadata(&artifact)
            .expect("fixture metadata should remain readable")
            .accessed()
            .expect("fixture atime should remain readable"),
        after,
        "validation must never restore attacker-observed atime"
    );
}
