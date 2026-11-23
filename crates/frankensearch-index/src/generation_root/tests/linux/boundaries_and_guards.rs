use super::super::super::platform::{
    TestBoundary, install_test_hook, validate_absolute_ancestor_owner_mode_for_test,
    validate_owner_mode_for_test,
};
use super::super::super::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationFileExpectation, GenerationRootAnchorLayout,
    GenerationRootError, GenerationRootErrorKind, GenerationRootLockMode, GenerationRootStage,
    QualifiedGenerationDirectory as QualifiedGenerationRoot,
    QualifiedGenerationRoot as NamedGenerationRoot,
};
use super::*;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt, symlink};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[test]
fn open_control_file_requires_write_access_and_private_regular_identity() {
    let root_path = fixture_root("lock-mode");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o400))
        .expect("fixture lock mode should change");
    let error = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect_err("non-private writable control file must fail");
    assert!(
        matches!(
            error.kind(),
            GenerationRootErrorKind::WrongMode | GenerationRootErrorKind::Io
        ),
        "write denial must fail before lock ownership"
    );
}

#[test]
fn sealed_file_role_mismatch_precedes_disk_access() {
    let root_path = fixture_root("role-mismatch");
    private_file(&root_path, "artifact", b"01234567");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    assert_eq!(
        root.admit_file(
            &confined("artifact"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect_err("control role must not enter immutable admission")
        .kind(),
        GenerationRootErrorKind::WrongRole
    );
    assert_eq!(
        root.admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::immutable(8, digest(b"lock-v1!"))
                .expect("small expectation should be accepted"),
        )
        .expect_err("immutable role must not enter control admission")
        .kind(),
        GenerationRootErrorKind::WrongRole
    );
}

#[test]
fn file_sync_failure_never_falls_back_to_a_weaker_barrier() {
    let root_path = fixture_root("no-sync-fallback");
    private_file(&root_path, "artifact", b"barrier");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(&confined("artifact"), immutable_expectation(b"barrier"))
        .expect("file should qualify");
    let calls = Arc::new(Mutex::new(Vec::new()));
    let hook_calls = Arc::clone(&calls);
    let _guard = install_test_hook(move |boundary| {
        if matches!(
            boundary,
            TestBoundary::BeforeFileSync | TestBoundary::AfterFileSync
        ) {
            hook_calls
                .lock()
                .expect("hook calls should lock")
                .push(boundary);
        }
        if boundary == TestBoundary::BeforeFileSync {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::DurabilityUnavailable,
                GenerationRootStage::SyncRegularFile,
            ));
        }
        Ok(())
    });
    assert_eq!(
        admitted
            .sync_durable()
            .expect_err("required barrier failure must propagate")
            .kind(),
        GenerationRootErrorKind::DurabilityUnavailable
    );
    assert_eq!(
        calls.lock().expect("hook calls should lock").as_slice(),
        &[TestBoundary::BeforeFileSync]
    );
}

#[test]
fn fixture_files_use_single_link_private_modes() {
    let root_path = fixture_root("fixture-self-check");
    let file_path = private_file(&root_path, "artifact", b"x");
    let root_metadata = fs::metadata(&root_path).expect("root metadata should load");
    let file_metadata = fs::metadata(file_path).expect("file metadata should load");
    assert_eq!(root_metadata.permissions().mode() & 0o7777, 0o700);
    assert_eq!(file_metadata.permissions().mode() & 0o7777, 0o400);
}

#[test]
fn root_admission_rejects_non_directory_final_component() {
    let container = fixture_root("root-not-directory");
    let file_path = private_file(&container, "not-a-root", b"x");
    let error = QualifiedGenerationRoot::admit(&file_path)
        .expect_err("regular file cannot be a generation root");
    assert!(
        matches!(
            error.kind(),
            GenerationRootErrorKind::NotDirectory | GenerationRootErrorKind::Io
        ),
        "non-directory final route must fail closed"
    );
}

#[test]
fn root_revalidation_detects_security_identity_drift() {
    let root_path = fixture_root("root-security-drift");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    fs::set_permissions(&root_path, fs::Permissions::from_mode(0o750))
        .expect("fixture mode should change");
    assert_eq!(
        root.revalidate_route()
            .expect_err("root mode drift must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn caller_control_hash_is_checked_only_after_lock_acquisition() {
    let root_path = fixture_root("lock-hash-order");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let control = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect("control file should qualify");
    let first = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("first exclusive lock should succeed");
    let wrong_hash = GenerationFileExpectation::control_with_sha256(8, [0x11; 32])
        .expect("small expectation should be accepted");
    let wrong_control = root
        .admit_control_file(&confined("LOCK"), wrong_hash)
        .expect("digest-qualified control file should qualify structurally");
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    assert_eq!(
        wrong_control
            .try_lock(GenerationRootLockMode::Exclusive)
            .expect_err("content must not be inspected through a contended lock")
            .kind(),
        GenerationRootErrorKind::LockContended
    );
    let observed = boundaries.lock().expect("boundary log should lock");
    assert!(observed.contains(&TestBoundary::BeforeLock));
    assert!(
        !observed
            .iter()
            .any(|boundary| matches!(boundary, TestBoundary::BeforeRead { .. })),
        "a contended lock must return before post-admission content or caller-digest reads"
    );
    drop(observed);
    drop(guard);
    first.unlock().expect("exclusive unlock should succeed");
    assert_eq!(
        wrong_control
            .try_lock(GenerationRootLockMode::Exclusive)
            .expect_err("wrong hash must fail after lock acquisition")
            .kind(),
        GenerationRootErrorKind::HashMismatch
    );
}

#[test]
fn explicit_file_creation_does_not_replace_existing_data() {
    let root_path = fixture_root("create-new");
    let path = private_file(&root_path, "artifact", b"first");
    let result = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&path);
    assert!(
        result.is_err(),
        "fixture and production doctrine require create-new semantics"
    );
    assert_eq!(
        fs::read(path).expect("original bytes should remain readable"),
        b"first"
    );
}

#[test]
fn retained_root_directory_sync_revalidates_before_and_after_barrier() {
    let root_path = fixture_root("root-sync-fence");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let moved = root_path.with_extension("moved-before-sync");
    fs::rename(&root_path, &moved).expect("root should move to a fresh path");
    DirBuilder::new()
        .mode(0o700)
        .create(&root_path)
        .expect("replacement root should be creatable");
    fs::set_permissions(&root_path, fs::Permissions::from_mode(0o700))
        .expect("replacement root mode should be settable");
    let error = root
        .sync_directory_durable()
        .expect_err("root route replacement must fail before sync");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::RevalidateRootRoute);
}

#[test]
fn root_component_replacement_during_walk_is_fault_detectable() {
    let root_path = fixture_root("ancestor-walk-race");
    let components = root_path
        .components()
        .filter(|component| !matches!(component, std::path::Component::RootDir))
        .count();
    assert!(components > 0);
    let observed = Arc::new(Mutex::new(false));
    let hook_observed = Arc::clone(&observed);
    let last_index = components.saturating_sub(1);
    let _guard = install_test_hook(move |boundary| {
        if boundary == (TestBoundary::AfterRootComponentOpen { index: last_index }) {
            *hook_observed.lock().expect("hook state should lock") = true;
            return Err(injected(GenerationRootStage::OpenRootRoute));
        }
        Ok(())
    });
    assert!(QualifiedGenerationRoot::admit(&root_path).is_err());
    assert!(*observed.lock().expect("hook state should lock"));
}

#[test]
fn read_fault_is_bounded_and_does_not_return_partial_bytes() {
    let root_path = fixture_root("read-fault");
    let contents = b"no partial publication";
    private_file(&root_path, "artifact", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let _guard = install_test_hook(|boundary| {
        if boundary == (TestBoundary::BeforeRead { offset: 0 }) {
            return Err(injected(GenerationRootStage::ReadRegularFile));
        }
        Ok(())
    });
    let error = root
        .admit_file(&confined("artifact"), immutable_expectation(contents))
        .expect_err("injected read failure must return no owner");
    assert_eq!(error.kind(), GenerationRootErrorKind::Io);
    assert_eq!(error.stage(), GenerationRootStage::ReadRegularFile);
}

#[test]
fn file_length_probe_rejects_growth_beyond_expected_end() {
    let root_path = fixture_root("end-probe");
    let artifact = private_file(&root_path, "artifact", b"12345678");
    fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
        .expect("fixture should become writable for the retained test fd");
    let mut mutation_file = OpenOptions::new()
        .write(true)
        .open(&artifact)
        .expect("retained mutation fd should open");
    fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
        .expect("fixture should be resealed before admission");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let mutated = Arc::new(Mutex::new(false));
    let hook_mutated = Arc::clone(&mutated);
    let _guard = install_test_hook(move |boundary| {
        if boundary
            == (TestBoundary::AfterRead {
                offset: 0,
                byte_count: 8,
            })
            && !*hook_mutated.lock().expect("hook state should lock")
        {
            mutation_file
                .seek(SeekFrom::End(0))
                .expect("fixture should be seekable");
            mutation_file.write_all(b"9").expect("fixture should grow");
            mutation_file.sync_all().expect("growth should be durable");
            *hook_mutated.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    assert_eq!(
        root.admit_file(&confined("artifact"), immutable_expectation(b"12345678"),)
            .expect_err("one-byte end growth must fail")
            .kind(),
        GenerationRootErrorKind::SizeMismatch
    );
}

#[test]
fn wrong_owner_witness_is_rejected_without_requiring_privileged_chown() {
    let root_path = fixture_root("owner-contract");
    let root =
        QualifiedGenerationRoot::admit(&root_path).expect("current-user root should qualify");
    let mut wrong_owner = root.witness();
    wrong_owner.uid = rustix::process::geteuid().as_raw().wrapping_add(1);
    assert_eq!(
        validate_owner_mode_for_test(
            wrong_owner,
            crate::generation_root::GENERATION_ROOT_DIRECTORY_MODE,
        )
        .expect_err("a non-effective owner witness must fail")
        .kind(),
        GenerationRootErrorKind::WrongOwner
    );
}

#[test]
fn opened_file_descriptor_witness_is_stable_across_arc_clones() {
    let root_path = fixture_root("arc-owner");
    let contents = b"one allocation";
    private_file(&root_path, "artifact", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(&confined("artifact"), immutable_expectation(contents))
        .expect("file should qualify");
    let first = admitted.bytes();
    let second = admitted.bytes();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.as_ref(), contents);
}

#[test]
fn file_mode_change_after_admission_fences_sync() {
    let root_path = fixture_root("file-mode-drift");
    let artifact = private_file(&root_path, "artifact", b"mode");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(&confined("artifact"), immutable_expectation(b"mode"))
        .expect("file should qualify");
    fs::set_permissions(&artifact, fs::Permissions::from_mode(0o640))
        .expect("fixture mode should change");
    assert_eq!(
        admitted
            .sync_durable()
            .expect_err("file mode drift must fence sync")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn open_file_missing_write_permission_never_acquires_lock() {
    let root_path = fixture_root("lock-permission");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o400))
        .expect("fixture lock mode should change");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let observed = Arc::new(Mutex::new(false));
    let hook_observed = Arc::clone(&observed);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeLock {
            *hook_observed.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    assert!(
        root.admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .is_err()
    );
    assert!(
        !*observed.lock().expect("hook state should lock"),
        "mode validation must precede the flock syscall"
    );
}

#[test]
fn root_filesystem_profile_exposes_only_qualified_local_type() {
    let root_path = fixture_root("filesystem-profile");
    // Derive the expected profile from the kernel's own statfs magic rather than
    // hard-coding one filesystem: the witness must report the qualified local type the
    // fixture root actually lives on, whichever of the qualified Linux filesystems that
    // is. Anything else is not a qualified root and the test cannot run there.
    let filesystem = rustix::fs::statfs(&root_path).expect("fixture root should be statable");
    let raw_type = i128::from(filesystem.f_type);
    let expected = if raw_type == i128::from(libc::EXT4_SUPER_MAGIC) {
        crate::generation_root::QualifiedFilesystem::LinuxExt4
    } else if raw_type == i128::from(libc::BTRFS_SUPER_MAGIC) {
        crate::generation_root::QualifiedFilesystem::LinuxBtrfs
    } else {
        panic!(
            "fixture root must live on ext4 or Btrfs (the qualified Linux filesystems); \
             statfs reported f_type {raw_type:#x}"
        );
    };
    let root =
        QualifiedGenerationRoot::admit(&root_path).expect("qualified local root should qualify");
    assert_eq!(root.witness().filesystem(), expected);
    assert_ne!(root.witness().mount_identity(), [0; 32]);
}

#[test]
fn explicit_private_file_is_not_mutated_by_failed_hash_admission() {
    let root_path = fixture_root("hash-decoy");
    let artifact = private_file(&root_path, "artifact", b"unchanged");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let expectation = GenerationFileExpectation::immutable(9, [0xff; 32])
        .expect("small expectation should be accepted");
    assert_eq!(
        root.admit_file(&confined("artifact"), expectation)
            .expect_err("wrong digest must fail")
            .kind(),
        GenerationRootErrorKind::HashMismatch
    );
    assert_eq!(
        fs::read(artifact).expect("artifact should remain readable"),
        b"unchanged"
    );
}

#[test]
fn opened_root_requires_absolute_route() {
    let error =
        QualifiedGenerationRoot::admit(Path::new("relative")).expect_err("relative root must fail");
    assert_eq!(error.kind(), GenerationRootErrorKind::InvalidRoute);
    assert_eq!(error.stage(), GenerationRootStage::ParseRootRoute);
}

#[test]
fn root_path_with_repeated_separator_is_rejected_before_open() {
    let error = QualifiedGenerationRoot::admit(Path::new("/data//forbidden"))
        .expect_err("repeated separator must fail");
    assert_eq!(error.kind(), GenerationRootErrorKind::InvalidRoute);
    assert_eq!(error.stage(), GenerationRootStage::ParseRootRoute);
}

#[test]
fn lock_file_hardlink_is_rejected_before_flock() {
    let root_path = fixture_root("lock-hardlink");
    let lock = control_file(&root_path, "LOCK", b"lock-v1!");
    fs::hard_link(&lock, root_path.join("LOCK-alias"))
        .expect("fixture hardlink should be creatable");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let observed = Arc::new(Mutex::new(false));
    let hook_observed = Arc::clone(&observed);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeLock {
            *hook_observed.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    assert_eq!(
        root.admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect_err("hardlinked lock file must fail")
        .kind(),
        GenerationRootErrorKind::HardLinked
    );
    assert!(!*observed.lock().expect("hook state should lock"));
}

#[test]
fn lock_sync_rechecks_identity_after_barrier() {
    let root_path = fixture_root("lock-sync-fence");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let control = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect("control file should qualify");
    let lock = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("exclusive lock should succeed");
    let _guard = install_test_hook(|boundary| {
        if boundary == TestBoundary::AfterFileSync {
            return Err(injected(GenerationRootStage::SyncRegularFile));
        }
        Ok(())
    });
    assert_eq!(
        lock.sync_durable()
            .expect_err("post-barrier fault must propagate")
            .kind(),
        GenerationRootErrorKind::Io
    );
    lock.unlock()
        .expect("lock should still be explicitly releasable");
}

#[test]
fn persistent_test_fixtures_never_use_automatic_cleanup() {
    let root_path = fixture_root("persistent");
    private_file(&root_path, "evidence", b"retained");
    assert!(root_path.exists());
    assert_eq!(
        fs::read(root_path.join("evidence")).expect("evidence should persist"),
        b"retained"
    );
}

#[test]
fn root_qualification_rejects_symlink_final_component() {
    let container = fixture_root("root-symlink-container");
    let target = fixture_root("root-symlink-target");
    let link = container.join("root-link");
    symlink(&target, &link).expect("root symlink should be creatable");
    assert_eq!(
        QualifiedGenerationRoot::admit(&link)
            .expect_err("root symlink must fail")
            .kind(),
        GenerationRootErrorKind::SymbolicLink
    );
}

/// Linux half of the cross-platform symlink error-kind parity contract
/// (bd-54h2l LOW B).
///
/// The macOS arm
/// `macos_absolute_root_route_rejects_ancestor_and_final_symlinks`
/// asserts [`GenerationRootErrorKind::SymbolicLink`] for an absolute
/// route whose ANCESTOR is a symlink and for one whose FINAL component
/// is, but it lives in a `cfg(all(target_os = "macos", target_arch =
/// "aarch64"))` module that never compiles on a Linux host. Linux
/// previously covered only the final component at root admission
/// (`root_qualification_rejects_symlink_final_component`) and the
/// confined-route case
/// (`ancestor_and_final_symlinks_fail_without_touching_decoys`), so the
/// ancestor-of-an-absolute-route shape had no Linux twin and the parity
/// claim rested on nothing measurable here.
///
/// This pins the Linux side of that claim on the identical input shape.
/// The macOS side remains an ENCODED EXPECTATION, not a live proof:
/// confirming both platforms agree still needs an arm64 macOS run, and
/// that residual is named on bd-54h2l.
#[test]
fn absolute_root_route_rejects_ancestor_and_final_symlinks() {
    let container = fixture_root("absolute-symlink-container");
    let target_parent = fixture_root("absolute-symlink-target-parent");
    let target_root = private_dir(&target_parent, "qualified-root");
    let ancestor_link = container.join("ancestor-link");
    let final_link = container.join("final-link");
    symlink(&target_parent, &ancestor_link)
        .expect("absolute-route ancestor symlink should be creatable");
    symlink(&target_root, &final_link).expect("absolute-route final symlink should be creatable");

    assert_eq!(
        QualifiedGenerationRoot::admit(&ancestor_link.join("qualified-root"))
            .expect_err("an absolute-route ancestor symlink must fail")
            .kind(),
        GenerationRootErrorKind::SymbolicLink
    );
    assert_eq!(
        QualifiedGenerationRoot::admit(&final_link)
            .expect_err("an absolute-route final symlink must fail")
            .kind(),
        GenerationRootErrorKind::SymbolicLink
    );
}

#[test]
fn file_reopen_fault_occurs_after_exact_hashing() {
    let root_path = fixture_root("reopen-order");
    let contents = b"hash-before-reopen";
    private_file(&root_path, "artifact", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        if boundary == TestBoundary::BeforeFileRouteReopen {
            return Err(injected(GenerationRootStage::RevalidateRegularFile));
        }
        Ok(())
    });
    assert!(
        root.admit_file(&confined("artifact"), immutable_expectation(contents))
            .is_err()
    );
    let observed = boundaries.lock().expect("boundary log should lock");
    let exact_read = observed
        .iter()
        .position(|boundary| *boundary == TestBoundary::AfterExactRead)
        .expect("exact read boundary should be observed");
    let reopen = observed
        .iter()
        .position(|boundary| *boundary == TestBoundary::BeforeFileRouteReopen)
        .expect("reopen boundary should be observed");
    drop(observed);
    assert!(exact_read < reopen);
}

#[test]
fn zero_length_file_is_admitted_without_partial_read_state() {
    let root_path = fixture_root("zero-length");
    private_file(&root_path, "empty", b"");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(&confined("empty"), immutable_expectation(b""))
        .expect("exact zero-length file should qualify");
    assert!(admitted.as_bytes().is_empty());
    assert_eq!(admitted.sha256(), digest(b""));
}

#[test]
fn lock_guard_debug_never_exposes_descriptor_or_path() {
    let root_path = fixture_root("lock-debug");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let control = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect("control file should qualify");
    let lock = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("exclusive lock should succeed");
    let debug = format!("{lock:?}");
    assert_eq!(debug, "GenerationRootLock { held: true, .. }");
    assert!(!debug.contains("LOCK"));
    assert!(!debug.contains(root_path.to_string_lossy().as_ref()));
    lock.unlock().expect("lock should release");
}

#[test]
fn file_admission_revalidates_root_after_file_reopen() {
    let root_path = fixture_root("final-root-fence");
    let contents = b"root fence";
    private_file(&root_path, "artifact", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let moved = root_path.with_extension("moved-at-final-fence");
    let swap_path = root_path.clone();
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeFinalRootRevalidation {
            fs::rename(&swap_path, &moved).expect("root should move to a fresh path");
            DirBuilder::new()
                .mode(0o700)
                .create(&swap_path)
                .expect("replacement root should be creatable");
            fs::set_permissions(&swap_path, fs::Permissions::from_mode(0o700))
                .expect("replacement root mode should be settable");
        }
        Ok(())
    });
    assert_eq!(
        root.admit_file(&confined("artifact"), immutable_expectation(contents))
            .expect_err("final root replacement must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn regular_file_descriptor_is_opened_cloexec() {
    let root_path = fixture_root("cloexec");
    private_file(&root_path, "artifact", b"cloexec");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let observed = Arc::new(Mutex::new(None));
    let hook_observed = Arc::clone(&observed);
    let _guard = install_test_hook(move |boundary| {
        if let TestBoundary::AfterRegularFileOpen { index: 0, cloexec } = boundary {
            *hook_observed.lock().expect("hook state should lock") = Some(cloexec);
        }
        Ok(())
    });
    root.admit_file(&confined("artifact"), immutable_expectation(b"cloexec"))
        .expect("file should qualify");
    assert_eq!(
        *observed.lock().expect("hook state should lock"),
        Some(true),
        "the retained descriptor must carry FD_CLOEXEC"
    );
}

#[test]
fn fixture_file_open_is_create_new_not_truncating() {
    let root_path = fixture_root("fixture-create-new");
    let path = private_file(&root_path, "artifact", b"original");
    let error = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&path)
        .expect_err("create-new should reject an existing fixture");
    assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(
        fs::read(path).expect("original should remain readable"),
        b"original"
    );
}

#[test]
fn error_counts_are_bounded_numeric_not_content_bearing() {
    let observed = crate::generation_root::GENERATION_ROOT_MAX_CONTROL_FILE_BYTES.saturating_add(1);
    let error =
        GenerationFileExpectation::control(observed).expect_err("control-file ceiling must fail");
    assert_eq!(
        error.expected(),
        Some(crate::generation_root::GENERATION_ROOT_MAX_CONTROL_FILE_BYTES)
    );
    assert_eq!(error.observed(), Some(observed));
    assert!(!error.to_string().contains("artifact"));
}

#[test]
fn file_admission_does_not_reopen_through_symlink_substitution() {
    let root_path = fixture_root("symlink-substitution");
    let contents = b"original";
    let original = private_file(&root_path, "artifact", contents);
    let decoy = private_file(&root_path, "decoy", contents);
    let retained = root_path.join("retained");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeFileRouteReopen {
            fs::rename(&original, &retained).expect("original should move to a fresh path");
            symlink(&decoy, &original).expect("attacker symlink should be creatable");
        }
        Ok(())
    });
    assert_eq!(
        root.admit_file(&confined("artifact"), immutable_expectation(contents))
            .expect_err("symlink substitution must fail")
            .kind(),
        GenerationRootErrorKind::SymbolicLink
    );
}

#[test]
fn direct_root_and_file_operations_leave_existing_siblings_unchanged() {
    let root_path = fixture_root("siblings");
    let sibling = private_file(&root_path, "sibling", b"sibling bytes");
    private_file(&root_path, "artifact", b"artifact bytes");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    root.admit_file(
        &confined("artifact"),
        immutable_expectation(b"artifact bytes"),
    )
    .expect("artifact should qualify");
    assert_eq!(
        fs::read(sibling).expect("sibling should remain readable"),
        b"sibling bytes"
    );
}

#[test]
fn opened_root_has_single_effective_owner() {
    let root_path = fixture_root("owner");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    assert_eq!(root.witness().uid(), rustix::process::geteuid().as_raw());
    assert_eq!(root.witness().gid(), rustix::process::getegid().as_raw());
}

#[test]
fn exact_control_file_size_is_checked_before_flock() {
    let root_path = fixture_root("lock-size");
    control_file(&root_path, "LOCK", b"short");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let observed = Arc::new(Mutex::new(false));
    let hook_observed = Arc::clone(&observed);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeLock {
            *hook_observed.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    assert_eq!(
        root.admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect_err("wrong lock size must fail")
        .kind(),
        GenerationRootErrorKind::SizeMismatch
    );
    assert!(!*observed.lock().expect("hook state should lock"));
}

#[test]
fn lock_candidate_open_faults_are_reported_as_acquire_lock() {
    let root_path = fixture_root("lock-candidate-stage");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let control = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("control should qualify before the injected drift");
    let reached_flock = Arc::new(Mutex::new(false));
    let hook_reached_flock = Arc::clone(&reached_flock);
    let mut changed = false;
    let _guard = install_test_hook(move |boundary| {
        if !changed && boundary == (TestBoundary::BeforeRegularFileOpen { index: 0 }) {
            fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o644))
                .expect("lock candidate mode drift should be injectable");
            changed = true;
        }
        if boundary == TestBoundary::BeforeLock {
            *hook_reached_flock.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    let error = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect_err("the drifted lock candidate must fail before flock");
    assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
    assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
    assert!(
        !*reached_flock.lock().expect("hook state should lock"),
        "candidate qualification must precede flock"
    );
}

#[test]
fn root_security_error_identifies_stage_without_path() {
    let root_path = fixture_root("root-stage");
    fs::set_permissions(&root_path, fs::Permissions::from_mode(0o755))
        .expect("fixture mode should change");
    let error = QualifiedGenerationRoot::admit(&root_path).expect_err("public root mode must fail");
    assert_eq!(error.stage(), GenerationRootStage::QualifyRootSecurity);
    assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
    assert!(
        !error
            .to_string()
            .contains(root_path.to_string_lossy().as_ref())
    );
}

#[test]
fn absolute_ancestor_owner_and_mode_policy_is_fail_closed() {
    let effective_uid = rustix::process::geteuid().as_raw();
    for owner in [0, effective_uid] {
        for mode in [0o700, 0o750, 0o755, 0o500, 0o555] {
            validate_absolute_ancestor_owner_mode_for_test(owner, mode)
                .expect("root-owned or euid-owned non-writable ancestors are trusted");
        }
    }
    for mode in [0o720, 0o702, 0o770, 0o707, 0o1777] {
        let error = validate_absolute_ancestor_owner_mode_for_test(effective_uid, mode)
            .expect_err("sticky or non-sticky group/world writable ancestors must fail");
        assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
    }
    let wrong_owner = if effective_uid == 0 {
        1
    } else {
        effective_uid + 1
    };
    assert_eq!(
        validate_absolute_ancestor_owner_mode_for_test(wrong_owner, 0o755)
            .expect_err("an unrelated owner must fail")
            .kind(),
        GenerationRootErrorKind::WrongOwner
    );
}

#[test]
fn absolute_route_qualifies_slash_and_every_non_final_component() {
    let root_path = fixture_root("absolute-ancestor-boundaries");
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    QualifiedGenerationRoot::admit(&root_path)
        .expect("the secure retained fixture route should qualify");
    let observed = boundaries.lock().expect("boundary log should lock");
    assert!(observed.contains(&TestBoundary::BeforeAbsoluteAncestorAclRead { index: None }));
    assert!(observed.contains(&TestBoundary::AfterAbsoluteAncestorAclRead { index: None }));
    let final_index = root_path.components().count().saturating_sub(2);
    assert!(observed.contains(&TestBoundary::BeforeRootComponentOpen { index: final_index }));
    assert!(
        !observed.contains(&TestBoundary::BeforeAbsoluteAncestorAclRead {
            index: Some(final_index),
        }),
        "the final root uses the stricter private-root qualifier, not ancestor policy"
    );
    drop(observed);
}

#[test]
fn group_world_and_sticky_writable_absolute_ancestors_are_rejected() {
    for (label, mode) in [("group", 0o770), ("world", 0o707), ("sticky", 0o1777)] {
        let ancestor = fixture_root(&format!("writable-ancestor-{label}"));
        let root_path = private_dir(&ancestor, "root");
        fs::set_permissions(&ancestor, fs::Permissions::from_mode(mode))
            .expect("hostile ancestor mode should be settable");
        let error = QualifiedGenerationRoot::admit(&root_path)
            .expect_err("an attacker-writable absolute ancestor must fail");
        assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
        assert!(error.component_index().is_some());
    }
}

#[test]
fn access_and_default_acls_on_absolute_ancestors_are_rejected() {
    for (label, xattr_name) in [
        ("access", "system.posix_acl_access"),
        ("default", "system.posix_acl_default"),
    ] {
        let ancestor = fixture_root(&format!("acl-ancestor-{label}"));
        let root_path = private_dir(&ancestor, "root");
        let retained = File::open(&ancestor).expect("ancestor descriptor should open");
        install_linux_acl_xattr(&retained, xattr_name, 0o7, 0o4, 0, 0o4, 0);
        assert_eq!(
            QualifiedGenerationRoot::admit(&root_path)
                .expect_err("an ancestor extended ACL must fail")
                .kind(),
            GenerationRootErrorKind::AclRejected
        );
    }
}

#[test]
fn ancestor_mutation_inside_the_acl_sandwich_is_rejected() {
    // A real security mutation (group/other-writable) slipped in
    // between the owner/mode check and the ACL read moves the ctime
    // witness; the re-qualification re-reads the mode and names it.
    let ancestor = fixture_root("ancestor-acl-sandwich");
    let root_path = private_dir(&ancestor, "root");
    let ancestor_index = ancestor.components().count().saturating_sub(2);
    let mut mutated = false;
    let hook_ancestor = ancestor.clone();
    let _guard = install_test_hook(move |boundary| {
        if !mutated
            && boundary
                == (TestBoundary::BeforeAbsoluteAncestorAclRead {
                    index: Some(ancestor_index),
                })
        {
            fs::set_permissions(&hook_ancestor, fs::Permissions::from_mode(0o722))
                .expect("the ancestor security mutation should be injectable");
            mutated = true;
        }
        Ok(())
    });
    let error = QualifiedGenerationRoot::admit(&root_path)
        .expect_err("a security mutation inside the ACL sandwich must fail");
    assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
    assert_eq!(
        error.component_index(),
        Some(u16::try_from(ancestor_index).expect("fixture depth fits u16"))
    );
}

#[test]
fn sibling_churn_inside_the_acl_sandwich_is_requalified_not_rejected() {
    // A neighbour created beside the route moves the ancestor's ctime
    // exactly like a security change would; the ancestor is
    // re-qualified against the fresh witness and admitted.
    let ancestor = fixture_root("ancestor-acl-sandwich-churn");
    let root_path = private_dir(&ancestor, "root");
    let ancestor_index = ancestor.components().count().saturating_sub(2);
    let mut churned = 0_u32;
    let hook_ancestor = ancestor.clone();
    let _guard = install_test_hook(move |boundary| {
        if churned < 2
            && boundary
                == (TestBoundary::BeforeAbsoluteAncestorAclRead {
                    index: Some(ancestor_index),
                })
        {
            private_dir(&hook_ancestor, &format!("neighbour-{churned}"));
            churned += 1;
        }
        Ok(())
    });
    let root = QualifiedGenerationRoot::admit(&root_path)
        .expect("sibling churn inside the sandwich is not a security change");
    assert_eq!(root.witness().mode() & 0o022, 0);
}

#[test]
fn an_ancestor_whose_ctime_never_settles_is_rejected() {
    let ancestor = fixture_root("ancestor-acl-sandwich-unsettled");
    let root_path = private_dir(&ancestor, "root");
    let ancestor_index = ancestor.components().count().saturating_sub(2);
    let mut churned = 0_u32;
    let hook_ancestor = ancestor.clone();
    let _guard = install_test_hook(move |boundary| {
        if boundary
            == (TestBoundary::BeforeAbsoluteAncestorAclRead {
                index: Some(ancestor_index),
            })
        {
            // Every attempt sees a fresh neighbour: the witness never
            // settles within the bound, so the ancestor is refused.
            private_dir(&hook_ancestor, &format!("neighbour-{churned}"));
            churned += 1;
        }
        Ok(())
    });
    let error = QualifiedGenerationRoot::admit(&root_path)
        .expect_err("an ancestor that never settles must fail closed");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(
        error.component_index(),
        Some(u16::try_from(ancestor_index).expect("fixture depth fits u16"))
    );
}

#[test]
fn ancestor_acl_added_between_full_route_walks_is_rejected() {
    let ancestor = fixture_root("ancestor-acl-between-walks");
    let root_path = private_dir(&ancestor, "root");
    let hook_ancestor = ancestor.clone();
    let mut mutated = false;
    let _guard = install_test_hook(move |boundary| {
        if !mutated && boundary == TestBoundary::AfterFirstRootQualification {
            let retained = File::open(&hook_ancestor).expect("ancestor descriptor should reopen");
            install_linux_acl_xattr(&retained, "system.posix_acl_default", 0o7, 0o4, 0, 0o4, 0);
            mutated = true;
        }
        Ok(())
    });
    assert_eq!(
        QualifiedGenerationRoot::admit(&root_path)
            .expect_err("an ACL added before the second route walk must fail")
            .kind(),
        GenerationRootErrorKind::AclRejected
    );
}

#[test]
fn absolute_ancestor_substitution_is_rejected_but_sibling_churn_is_allowed() {
    let parent = fixture_root("ancestor-substitution-parent");
    let live = private_dir(&parent, "live");
    let live_root = private_dir(&live, "root");
    let decoy = private_dir(&parent, "decoy");
    private_dir(&decoy, "root");
    let retained = parent.join("retained-live");
    let hook_live = live.clone();
    let hook_decoy = decoy.clone();
    let mut swapped = false;
    let swap_guard = install_test_hook(move |boundary| {
        if !swapped && boundary == TestBoundary::AfterFirstRootQualification {
            fs::rename(&hook_live, &retained)
                .expect("the original ancestor should move to a fresh retained name");
            fs::rename(&hook_decoy, &hook_live)
                .expect("the decoy ancestor should move into the live route");
            swapped = true;
        }
        Ok(())
    });
    assert_eq!(
        QualifiedGenerationRoot::admit(&live_root)
            .expect_err("an absolute ancestor substitution must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
    drop(swap_guard);

    let churn_parent = fixture_root("ancestor-sibling-churn");
    let churn_root = private_dir(&churn_parent, "root");
    let hook_parent = churn_parent.clone();
    let mut churned = false;
    let _guard = install_test_hook(move |boundary| {
        if !churned && boundary == TestBoundary::AfterFirstRootQualification {
            control_file(&hook_parent, "unrelated-sibling", b"retained");
            churned = true;
        }
        Ok(())
    });
    QualifiedGenerationRoot::admit(&churn_root)
        .expect("unrelated sibling churn must not poison persisted route identity");
}

#[test]
fn named_anchor_guard_binds_exact_tuple_and_images() {
    let lock_bytes = b"lock-v1!";
    let (root_path, _authority_path, layout) =
        named_root_fixture("named-anchor-happy", lock_bytes, 0xa5);
    let root = NamedGenerationRoot::admit(&root_path, layout)
        .expect("the exact named anchor tuple should qualify");
    let witnesses = root.anchor_witnesses();
    assert_eq!(witnesses.root(), root.witness());
    assert_ne!(witnesses.root().inode(), witnesses.lock().inode());
    assert_ne!(witnesses.root().inode(), witnesses.authority().inode());
    assert_ne!(witnesses.lock().inode(), witnesses.authority().inode());
    assert_eq!(
        witnesses.lock().byte_len(),
        u64::try_from(lock_bytes.len()).expect("fixture length fits u64")
    );
    assert_eq!(
        witnesses.authority().byte_len(),
        GENERATION_ROOT_AUTHORITY_FILE_BYTES
    );
    let guard = root
        .read_guard()
        .expect("the shared read guard should acquire");
    assert_eq!(guard.lock_bytes(), lock_bytes);
    assert!(guard.authority_bytes().iter().all(|byte| *byte == 0xa5));
    assert_eq!(guard.lock_sha256(), digest(lock_bytes));
    assert_eq!(guard.authority_sha256(), digest(guard.authority_bytes()));
    let first_lock_arc = guard.lock_bytes_arc();
    let second_lock_arc = guard.lock_bytes_arc();
    assert!(Arc::ptr_eq(&first_lock_arc, &second_lock_arc));
    guard
        .release()
        .expect("the unchanged shared guard should release");
}

#[test]
fn named_authority_legitimate_in_place_rewrite_is_visible_to_the_next_guard() {
    let (root_path, authority_path, layout) =
        named_root_fixture("named-anchor-rewrite", b"lock-v1!", 0x11);
    let root = NamedGenerationRoot::admit(&root_path, layout)
        .expect("the original named anchors should qualify");
    let first = root.read_guard().expect("the first guard should acquire");
    assert!(first.authority_bytes().iter().all(|byte| *byte == 0x11));
    first.release().expect("the first guard should release");

    let replacement = authority_image(0x22);
    rewrite_control_file(&authority_path, &replacement);
    root.revalidate()
        .expect("same-inode fixed-length authority advancement is legitimate");
    let second = root.read_guard().expect("the next guard should acquire");
    assert_eq!(second.authority_bytes(), replacement);
    second.release().expect("the second guard should release");
}

#[test]
fn shared_guards_coexist_and_exclude_the_exclusive_anchor_guard() {
    let (root_path, _authority_path, layout) =
        named_root_fixture("named-anchor-contention", b"lock-v1!", 0x33);
    let root =
        NamedGenerationRoot::admit(&root_path, layout).expect("the named anchors should qualify");
    let first = root.read_guard().expect("the first reader should acquire");
    let second = root.read_guard().expect("a second reader should coexist");
    assert_eq!(
        root.try_exclusive_anchor_guard()
            .expect_err("an exclusive guard must not bypass active readers")
            .kind(),
        GenerationRootErrorKind::LockContended
    );
    second.release().expect("the second reader should release");
    first.release().expect("the first reader should release");
    root.try_exclusive_anchor_guard()
        .expect("the exclusive guard should acquire after both readers release")
        .release()
        .expect("the exclusive guard should release");
}

#[test]
fn captured_anchor_arc_is_immutable_and_release_detects_ambient_mutation() {
    let (root_path, authority_path, layout) =
        named_root_fixture("named-anchor-ambient-mutation", b"lock-v1!", 0x44);
    let root =
        NamedGenerationRoot::admit(&root_path, layout).expect("the named anchors should qualify");
    let guard = root.read_guard().expect("the shared guard should acquire");
    let captured = guard.authority_bytes_arc();
    let replacement = authority_image(0x55);
    rewrite_control_file(&authority_path, &replacement);
    assert!(captured.iter().all(|byte| *byte == 0x44));
    let error = guard
        .release()
        .expect_err("ambient mutation must invalidate explicit guard release");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::ReleaseLock);
    root.try_exclusive_anchor_guard()
        .expect("failed validation must still unlock the kernel flock")
        .release()
        .expect("the post-failure exclusive guard should release");
}

#[test]
fn drift_between_double_anchor_captures_fails_before_returning_a_guard() {
    let (root_path, authority_path, layout) =
        named_root_fixture("named-anchor-double-capture", b"lock-v1!", 0x66);
    let root =
        NamedGenerationRoot::admit(&root_path, layout).expect("the named anchors should qualify");
    let replacement = authority_image(0x77);
    let mut exact_reads = 0_u8;
    let mut injected = false;
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::AfterExactRead {
            exact_reads = exact_reads.saturating_add(1);
        } else if exact_reads >= 4
            && !injected
            && matches!(boundary, TestBoundary::BeforeRootComponentOpen { .. })
        {
            rewrite_control_file(&authority_path, &replacement);
            injected = true;
        }
        Ok(())
    });
    let error = root
        .read_guard()
        .expect_err("authority drift between the two captures must fail");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(
        error.stage(),
        GenerationRootStage::AcquireLock,
        "the completed first image and later image must disagree at the guard boundary"
    );
}

#[test]
fn named_anchor_alias_missing_name_and_route_replacement_fail_closed() {
    let alias_root = fixture_root("named-anchor-hardlink-alias");
    let shared = authority_image(0x88);
    let lock_path = control_file(&alias_root, "LOCK", &shared);
    fs::hard_link(&lock_path, alias_root.join("AUTHORITY"))
        .expect("the hostile hard-link alias should be creatable");
    let layout = GenerationRootAnchorLayout::new(GENERATION_ROOT_AUTHORITY_FILE_BYTES)
        .expect("the alias layout is bounded");
    assert_eq!(
        NamedGenerationRoot::admit(&alias_root, layout)
            .expect_err("LOCK and AUTHORITY must never alias one inode")
            .kind(),
        GenerationRootErrorKind::HardLinked
    );

    let missing_root = fixture_root("named-anchor-missing");
    control_file(&missing_root, "lock", b"lock-v1!");
    control_file(&missing_root, "AUTHORITY", &authority_image(0x99));
    assert!(
        NamedGenerationRoot::admit(
            &missing_root,
            GenerationRootAnchorLayout::new(8).expect("layout is bounded"),
        )
        .is_err(),
        "the lower-case decoy must not satisfy the exact LOCK route"
    );

    let (root_path, authority_path, layout) =
        named_root_fixture("named-anchor-route-swap", b"lock-v1!", 0xaa);
    let root = NamedGenerationRoot::admit(&root_path, layout)
        .expect("the original named tuple should qualify");
    let retained = root_path.join("AUTHORITY-retained");
    let hook_root = root_path.clone();
    let mut exact_reads = 0_u8;
    let mut injected = false;
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::AfterExactRead {
            exact_reads = exact_reads.saturating_add(1);
        } else if exact_reads >= 4
            && !injected
            && matches!(boundary, TestBoundary::BeforeRootComponentOpen { .. })
        {
            fs::rename(&authority_path, &retained)
                .expect("the original authority should move to a retained name");
            control_file(&hook_root, "AUTHORITY", &authority_image(0xaa));
            injected = true;
        }
        Ok(())
    });
    assert_eq!(
        root.read_guard()
            .expect_err("route replacement during tuple capture must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn named_root_and_read_guard_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NamedGenerationRoot>();
    assert_send_sync::<crate::generation_root::GenerationRootReadGuard<'static>>();
}

#[test]
fn private_file_helper_never_opens_read_write_by_default() {
    let root_path = fixture_root("helper-access");
    let path = private_file(&root_path, "artifact", b"readable");
    let mut file = File::open(path).expect("fixture should be readable");
    let position = file
        .stream_position()
        .expect("fixture stream position should be available");
    assert_eq!(position, 0);
}
