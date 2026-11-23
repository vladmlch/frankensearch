use super::super::super::platform::{
    TestBoundary, install_test_hook, set_control_root_creator_process_id_for_test,
    set_root_creator_process_id_for_test,
};
use super::super::super::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationFileExpectation, GenerationRootAnchorLayout,
    GenerationRootErrorKind, GenerationRootLockMode, GenerationRootStage,
    PreopenedGenerationRootAnchors, QualifiedFilesystem,
    QualifiedGenerationDirectory as QualifiedGenerationRoot,
    QualifiedGenerationRoot as NamedGenerationRoot,
};
use super::*;
use std::fs::{self, File, OpenOptions};
use std::os::fd::OwnedFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt, symlink};
use std::process::Command;
use std::sync::{Arc, Mutex};

#[test]
fn named_anchor_preopened_guard_binds_exact_tuple_and_advances_authority() {
    let lock_bytes = b"lock-v1!";
    let (root_path, lock_path, authority_path, layout) =
        named_root_fixture("named-anchor-preopened", lock_bytes, 0x31);
    let root = admit_named_root(&root_path, &lock_path, &authority_path, layout);
    assert_eq!(
        root.read_guard()
            .expect_err("Apple Silicon must require a fresh provider-owned LOCK open")
            .kind(),
        GenerationRootErrorKind::PreopenedDescriptorRequired
    );
    let first = root
        .read_guard_preopened(preopened_control(&lock_path))
        .expect("the first provider-owned shared guard should acquire");
    assert_eq!(first.lock_bytes(), lock_bytes);
    assert!(first.authority_bytes().iter().all(|byte| *byte == 0x31));
    first.release().expect("the unchanged guard should release");

    let replacement = authority_image(0x42);
    rewrite_control_file(&authority_path, &replacement);
    root.revalidate()
        .expect("same-inode fixed-length AUTHORITY advancement should remain admissible");
    let second = root
        .read_guard_preopened(preopened_control(&lock_path))
        .expect("the next provider-owned shared guard should acquire");
    assert_eq!(second.authority_bytes(), replacement);
    second.release().expect("the advanced guard should release");
}

#[test]
fn preopened_shared_guards_coexist_and_exclude_a_fresh_exclusive_open() {
    let (root_path, lock_path, authority_path, layout) =
        named_root_fixture("named-anchor-preopened-contention", b"lock-v1!", 0x53);
    let root = admit_named_root(&root_path, &lock_path, &authority_path, layout);
    let first = root
        .read_guard_preopened(preopened_control(&lock_path))
        .expect("the first provider-owned reader should acquire");
    let second = root
        .read_guard_preopened(preopened_control(&lock_path))
        .expect("the second provider-owned reader should coexist");
    assert_eq!(
        root.try_exclusive_anchor_guard_preopened(preopened_control(&lock_path))
            .expect_err("a fresh exclusive open must contend with active readers")
            .kind(),
        GenerationRootErrorKind::LockContended
    );
    second.release().expect("the second reader should release");
    first.release().expect("the first reader should release");
    root.try_exclusive_anchor_guard_preopened(preopened_control(&lock_path))
        .expect("the exclusive guard should acquire after both readers release")
        .release()
        .expect("the exclusive guard should release");
}

#[test]
fn preopened_named_anchor_hardlink_alias_and_ancestor_acls_fail_closed() {
    let alias_root = fixture_root("named-anchor-preopened-alias");
    let shared = authority_image(0x64);
    let lock_path = control_file(&alias_root, "LOCK", &shared);
    let authority_path = alias_root.join("AUTHORITY");
    fs::hard_link(&lock_path, &authority_path)
        .expect("the hostile APFS hard-link alias should be creatable");
    set_empty_acl(&authority_path);
    let error = NamedGenerationRoot::admit_preopened(
        &alias_root,
        GenerationRootAnchorLayout::new(GENERATION_ROOT_AUTHORITY_FILE_BYTES)
            .expect("the alias layout is bounded"),
        PreopenedGenerationRootAnchors::new(
            preopened_control(&lock_path),
            preopened_control(&authority_path),
        ),
    )
    .expect_err("LOCK and AUTHORITY must never alias one inode");
    assert_eq!(error.kind(), GenerationRootErrorKind::HardLinked);

    for (label, allow) in [("allow", true), ("deny", false)] {
        let ancestor = fixture_root(&format!("absolute-ancestor-acl-{label}"));
        let root_path = private_dir(&ancestor, "root");
        set_non_empty_acl(&ancestor, allow);
        assert_eq!(
            QualifiedGenerationRoot::admit(&root_path)
                .expect_err("every non-empty APFS ancestor ACL must fail")
                .kind(),
            GenerationRootErrorKind::AclRejected
        );
    }
}

#[test]
fn preopened_guard_release_detects_live_authority_mutation_and_unlocks() {
    let (root_path, lock_path, authority_path, layout) =
        named_root_fixture("named-anchor-preopened-live-mutation", b"lock-v1!", 0x71);
    let root = admit_named_root(&root_path, &lock_path, &authority_path, layout);
    let guard = root
        .read_guard_preopened(preopened_control(&lock_path))
        .expect("the provider-owned shared guard should acquire");
    let captured = guard.authority_bytes_arc();
    rewrite_control_file(&authority_path, &authority_image(0x72));
    assert!(
        captured.iter().all(|byte| *byte == 0x71),
        "the live guard must retain its immutable APFS authority image"
    );
    let error = guard
        .release()
        .expect_err("ambient APFS authority mutation must invalidate release");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::ReleaseLock);
    root.try_exclusive_anchor_guard_preopened(preopened_control(&lock_path))
        .expect("failed validation must still release the provider-owned flock")
        .release()
        .expect("the post-failure exclusive guard should release");
}

#[test]
fn preopened_double_capture_rejects_content_drift_and_route_replacement() {
    let (drift_root, drift_lock, drift_authority, drift_layout) =
        named_root_fixture("named-anchor-preopened-double-drift", b"lock-v1!", 0x81);
    let root = admit_named_root(&drift_root, &drift_lock, &drift_authority, drift_layout);
    let replacement = authority_image(0x82);
    let mut exact_reads = 0_u8;
    let mut injected = false;
    let drift_hook = install_test_hook(move |boundary| {
        if boundary == TestBoundary::AfterExactRead {
            exact_reads = exact_reads.saturating_add(1);
        } else if exact_reads >= 4
            && !injected
            && matches!(boundary, TestBoundary::BeforeRootComponentOpen { .. })
        {
            rewrite_control_file(&drift_authority, &replacement);
            injected = true;
        }
        Ok(())
    });
    let drift_error = root
        .read_guard_preopened(preopened_control(&drift_lock))
        .expect_err("APFS authority drift between captures must fail");
    assert_eq!(drift_error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(
        drift_error.stage(),
        GenerationRootStage::AcquireLock,
        "the completed first image and later image must disagree at the guard boundary"
    );
    drop(drift_hook);

    let (route_root, route_lock, route_authority, route_layout) =
        named_root_fixture("named-anchor-preopened-route-swap", b"lock-v1!", 0x91);
    let route = admit_named_root(&route_root, &route_lock, &route_authority, route_layout);
    let retained = route_root.join("AUTHORITY-retained");
    let hook_root = route_root.clone();
    let mut route_reads = 0_u8;
    let mut injected = false;
    let route_hook = install_test_hook(move |boundary| {
        if boundary == TestBoundary::AfterExactRead {
            route_reads = route_reads.saturating_add(1);
        } else if route_reads >= 4
            && !injected
            && matches!(boundary, TestBoundary::BeforeRootComponentOpen { .. })
        {
            fs::rename(&route_authority, &retained)
                .expect("the original APFS authority should move aside");
            control_file(&hook_root, "AUTHORITY", &authority_image(0x92));
            injected = true;
        }
        Ok(())
    });
    assert_eq!(
        route
            .read_guard_preopened(preopened_control(&route_lock))
            .expect_err("APFS authority route replacement between captures must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
    drop(route_hook);
}

#[test]
fn absolute_ancestor_acl_mutation_inside_darwin_ctime_sandwich_is_rejected() {
    let ancestor = fixture_root("absolute-ancestor-acl-sandwich");
    let root_path = private_dir(&ancestor, "root");
    let ancestor_index = ancestor.components().count().saturating_sub(2);
    let hook_ancestor = ancestor.clone();
    let mut mutated = false;
    let hook = install_test_hook(move |boundary| {
        if !mutated
            && boundary
                == (TestBoundary::AfterAbsoluteAncestorAclRead {
                    index: Some(ancestor_index),
                })
        {
            set_non_empty_acl(&hook_ancestor, true);
            mutated = true;
        }
        Ok(())
    });
    // The moved ctime witness re-qualifies the ancestor, and the
    // re-read now sees the inserted ACL: the verdict names the ACL
    // itself rather than the fence that caught it.
    let error = QualifiedGenerationRoot::admit(&root_path)
        .expect_err("an ACL inserted after the Darwin probe must be caught on re-read");
    assert_eq!(error.kind(), GenerationRootErrorKind::AclRejected);
    assert_eq!(
        error.component_index(),
        Some(u16::try_from(ancestor_index).expect("fixture depth fits u16"))
    );
    drop(hook);
}

#[test]
fn physical_apfs_empty_acl_exact_names_locks_and_fullsync_succeed() {
    let root_path = fixture_root("physical-success");
    let generation = private_dir(&root_path, "Generation");
    let contents = b"physical APFS generation";
    let artifact_path = private_file(&generation, "Vector.fsvi", contents);
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");

    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    assert_eq!(root.witness().filesystem(), QualifiedFilesystem::AppleApfs);
    let admitted = root
        .admit_preopened_file(
            &confined("Generation/Vector.fsvi"),
            preopened_immutable(&artifact_path),
            immutable_expectation(contents),
        )
        .expect("empty-ACL exact-case immutable APFS file should qualify");
    assert_eq!(admitted.as_bytes(), contents);
    admitted
        .sync_durable()
        .expect("retained APFS regular file must support F_FULLFSYNC");

    let control = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("empty-ACL APFS control file should qualify");
    assert_eq!(control.bytes().as_ref(), b"lock-v1!");
    assert_eq!(control.sha256(), digest(b"lock-v1!"));
    let lock = control
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect("APFS control-file flock should succeed");
    assert_eq!(lock.witness(), control.witness());
    assert_eq!(lock.bytes().as_ref(), b"lock-v1!");
    assert_eq!(lock.sha256(), control.sha256());
    assert_eq!(lock.mode(), GenerationRootLockMode::Exclusive);
    lock.sync_durable()
        .expect("locked APFS control file must support F_FULLFSYNC");
    lock.unlock()
        .expect("APFS control-file unlock should succeed");
    root.sync_directory_durable()
        .expect("retained APFS directory must support F_FULLFSYNC");

    assert_eq!(
        root.admit_preopened_file(
            &confined("generation/Vector.fsvi"),
            preopened_immutable(&artifact_path),
            immutable_expectation(contents),
        )
        .expect_err("case-folded ancestor spelling must fail even on insensitive APFS")
        .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
    assert_eq!(
        root.admit_preopened_file(
            &confined("Generation/vector.fsvi"),
            preopened_immutable(&artifact_path),
            immutable_expectation(contents),
        )
        .expect_err("case-folded final spelling must fail even on insensitive APFS")
        .kind(),
        GenerationRootErrorKind::ObjectChanged
    );

    let decomposed_name = "e\u{301}.fsvi";
    let decomposed_path = private_file(&generation, decomposed_name, b"unicode");
    assert_eq!(
        root.admit_preopened_file(
            &confined("Generation/\u{e9}.fsvi"),
            preopened_immutable(&decomposed_path),
            immutable_expectation(b"unicode"),
        )
        .expect_err("canonically equivalent but byte-distinct spelling must fail")
        .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn macos_independent_lock_contention_and_creator_release_survive_aliases_and_failures() {
    let root_path = fixture_root("lock-lifecycle");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let control = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("empty-ACL APFS control should qualify");

    let shared_one = control
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Shared,
        )
        .expect("first independent shared lock should succeed");
    let shared_two = control
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Shared,
        )
        .expect("second independent shared lock should succeed");
    let contention_boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&contention_boundaries);
    let contention_hook = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let contention_error = control
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect_err("an independent exclusive lock must contend with shared locks");
    assert_eq!(
        contention_error.kind(),
        GenerationRootErrorKind::LockContended
    );
    let observed = contention_boundaries
        .lock()
        .expect("boundary log should lock");
    assert!(observed.contains(&TestBoundary::BeforeLock));
    assert!(
        !observed
            .iter()
            .any(|boundary| matches!(boundary, TestBoundary::BeforeRead { .. })),
        "a contended APFS lock must return before descriptor content is inspected"
    );
    drop(observed);
    drop(contention_hook);
    drop(shared_one);
    shared_two
        .unlock()
        .expect("the second independent shared lock should release");

    let drop_candidate = preopened_control(&lock_path);
    let surviving_drop_alias = rustix::io::fcntl_dupfd_cloexec(&drop_candidate, 0)
        .expect("the regression fixture should retain a known duplicate");
    let dropped = control
        .try_lock_preopened(drop_candidate, GenerationRootLockMode::Exclusive)
        .expect("the aliased candidate should acquire under the trusted-provider contract");
    drop(dropped);
    control
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect("creator Drop must explicitly unlock despite a surviving alias")
        .unlock()
        .expect("fresh contender after Drop should release");
    drop(surviving_drop_alias);

    let explicit_candidate = preopened_control(&lock_path);
    let surviving_explicit_alias = rustix::io::fcntl_dupfd_cloexec(&explicit_candidate, 0)
        .expect("the explicit-unlock fixture should retain a known duplicate");
    control
        .try_lock_preopened(explicit_candidate, GenerationRootLockMode::Exclusive)
        .expect("the explicit-unlock fixture should acquire")
        .unlock()
        .expect("explicit unlock should disarm creator Drop deterministically");
    control
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect("explicit LOCK_UN must release despite a surviving alias")
        .unlock()
        .expect("fresh contender after explicit unlock should release");
    drop(surviving_explicit_alias);

    let failed_candidate = preopened_control(&lock_path);
    let surviving_failure_alias = rustix::io::fcntl_dupfd_cloexec(&failed_candidate, 0)
        .expect("the post-flock failure fixture should retain a known duplicate");
    let hook = install_test_hook(|boundary| {
        if boundary == TestBoundary::AfterLock {
            return Err(injected(GenerationRootStage::AcquireLock));
        }
        Ok(())
    });
    assert_eq!(
        control
            .try_lock_preopened(failed_candidate, GenerationRootLockMode::Exclusive)
            .expect_err("injected post-flock failure must fail")
            .kind(),
        GenerationRootErrorKind::Io
    );
    drop(hook);
    control
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect("provisional RAII must unlock after post-flock failure despite an alias")
        .unlock()
        .expect("fresh contender after injected failure should release");
    drop(surviving_failure_alias);
}

#[test]
fn macos_path_final_admission_requires_preopened_capabilities_before_any_io_boundary() {
    let root_path = fixture_root("path-requires-preopened");
    let artifact_path = private_file(&root_path, "artifact", b"sealed");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let control = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("trusted preopened control should qualify");

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    for error in [
        root.admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
            .expect_err("path artifact admission must require external provenance"),
        root.admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect_err("path control admission must require external provenance"),
    ] {
        assert_eq!(
            error.kind(),
            GenerationRootErrorKind::PreopenedDescriptorRequired
        );
        assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
    }
    let lock_error = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect_err("path lock acquisition must require a fresh provider descriptor");
    assert_eq!(
        lock_error.kind(),
        GenerationRootErrorKind::PreopenedDescriptorRequired
    );
    assert_eq!(lock_error.stage(), GenerationRootStage::AcquireLock);
    assert!(
        boundaries
            .lock()
            .expect("boundary log should lock")
            .is_empty(),
        "typed path rejection must precede revalidation, open, read, flock, ACL, and sync"
    );
    assert!(
        artifact_path.exists(),
        "the known regular fixture remains intact"
    );
}

#[test]
fn macos_forked_capabilities_reject_before_provider_descriptor_inspection() {
    let root_path = fixture_root("forked-preopened");
    let artifact_path = private_file(&root_path, "artifact", b"sealed");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let mut root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    set_root_creator_process_id_for_test(&mut root.inner, std::process::id().wrapping_add(1));
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let artifact_error = root
        .admit_preopened_file(
            &confined("artifact"),
            File::open(&root_path)
                .expect("the precedence fixture directory should open")
                .into(),
            immutable_expectation(b"sealed"),
        )
        .expect_err("ForkedProcess must outrank the supplied directory's NotRegularFile error");
    assert_eq!(
        artifact_error.kind(),
        GenerationRootErrorKind::ForkedProcess
    );
    assert_eq!(artifact_error.stage(), GenerationRootStage::OpenRegularFile);
    let control_error = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_immutable(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect_err("ForkedProcess must outrank the read-only fd's InvalidDescriptorAccess error");
    assert_eq!(control_error.kind(), GenerationRootErrorKind::ForkedProcess);
    assert_eq!(control_error.stage(), GenerationRootStage::OpenRegularFile);
    assert!(
        boundaries
            .lock()
            .expect("boundary log should lock")
            .is_empty(),
        "creator-process rejection must precede provider-fd inspection and every I/O \
         boundary"
    );
    drop(guard);

    let root = QualifiedGenerationRoot::admit(&root_path).expect("fresh APFS root should qualify");
    let mut control = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("fresh control should qualify");
    drop(root);
    set_control_root_creator_process_id_for_test(
        &mut control.inner,
        std::process::id().wrapping_add(1),
    );
    let lock_boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_lock_boundaries = Arc::clone(&lock_boundaries);
    let _lock_guard = install_test_hook(move |boundary| {
        hook_lock_boundaries
            .lock()
            .expect("lock boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let lock_error = control
        .try_lock_preopened(
            preopened_immutable(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect_err("ForkedProcess must outrank the lock fd's InvalidDescriptorAccess error");
    assert_eq!(lock_error.kind(), GenerationRootErrorKind::ForkedProcess);
    assert_eq!(lock_error.stage(), GenerationRootStage::AcquireLock);
    assert!(
        lock_boundaries
            .lock()
            .expect("lock boundary log should lock")
            .is_empty(),
        "creator-process rejection must precede lock-candidate inspection, ACL, content, \
         and flock"
    );
    assert!(
        artifact_path.exists(),
        "the known regular fixture must remain intact"
    );
}

#[test]
fn macos_preopened_type_access_and_cloexec_gates_precede_content_and_lock_operations() {
    let root_path = fixture_root("preopened-gates");
    let artifact_path = private_file(&root_path, "artifact", b"sealed");
    let execute_path = private_file(&root_path, "execute-artifact", b"sealed");
    fs::set_permissions(&execute_path, fs::Permissions::from_mode(0o500))
        .expect("O_EXEC fixture must be owner-executable");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let fifo_path = root_path.join("fifo");
    let fifo_status = Command::new("/usr/bin/mkfifo")
        .arg("-m")
        .arg("600")
        .arg(&fifo_path)
        .status()
        .expect("macOS mkfifo should start for the special-file fixture");
    assert!(
        fifo_status.success(),
        "macOS mkfifo should create the special-file fixture"
    );
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });

    let fifo: OwnedFd = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&fifo_path)
        .expect("trusted provider should supply the FIFO descriptor for rejection")
        .into();
    let fifo_error = root
        .admit_preopened_control_file(
            &confined("fifo"),
            fifo,
            GenerationFileExpectation::control(0)
                .expect("zero-length control expectation should be accepted"),
        )
        .expect_err("a supplied FIFO must fail at the first fstat type gate");
    assert_eq!(fifo_error.kind(), GenerationRootErrorKind::NotRegularFile);

    let device: OwnedFd = File::open("/dev/null")
        .expect("trusted provider should supply a real device descriptor for rejection")
        .into();
    let device_error = root
        .admit_preopened_file(
            &confined("artifact"),
            device,
            immutable_expectation(b"sealed"),
        )
        .expect_err("a supplied device must fail at the first fstat type gate");
    assert_eq!(device_error.kind(), GenerationRootErrorKind::NotRegularFile);

    fs::set_permissions(&artifact_path, fs::Permissions::from_mode(0o600))
        .expect("fixture should become writable before the provider opens the wrong mode");
    let wrong_artifact_access: OwnedFd = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&artifact_path)
        .expect("trusted provider should supply a deliberately wrong access mode")
        .into();
    fs::set_permissions(&artifact_path, fs::Permissions::from_mode(0o400))
        .expect("fixture should be resealed before library admission");
    let access_error = root
        .admit_preopened_file(
            &confined("artifact"),
            wrong_artifact_access,
            immutable_expectation(b"sealed"),
        )
        .expect_err("immutable admission requires an exactly read-only descriptor");
    assert_eq!(
        access_error.kind(),
        GenerationRootErrorKind::InvalidDescriptorAccess
    );

    let wrong_control_access = preopened_immutable(&lock_path);
    let control_access_error = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            wrong_control_access,
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect_err("control admission requires an exactly read-write descriptor");
    assert_eq!(
        control_access_error.kind(),
        GenerationRootErrorKind::InvalidDescriptorAccess
    );

    let missing_cloexec = preopened_immutable(&artifact_path);
    rustix::io::fcntl_setfd(&missing_cloexec, rustix::io::FdFlags::empty())
        .expect("test provider should clear close-on-exec");
    let cloexec_error = root
        .admit_preopened_file(
            &confined("artifact"),
            missing_cloexec,
            immutable_expectation(b"sealed"),
        )
        .expect_err("missing FD_CLOEXEC must fail before route or content work");
    assert_eq!(
        cloexec_error.kind(),
        GenerationRootErrorKind::CloseOnExecRequired
    );

    let event_only_error = root
        .admit_preopened_file(
            &confined("artifact"),
            preopened_with_status_flag(&artifact_path, libc::O_EVTONLY),
            immutable_expectation(b"sealed"),
        )
        .expect_err("O_EVTONLY is not a data-readable descriptor");
    assert_eq!(
        event_only_error.kind(),
        GenerationRootErrorKind::InvalidDescriptorAccess
    );

    let execute_only_error = root
        .admit_preopened_file(
            &confined("execute-artifact"),
            preopened_with_status_flag(&execute_path, libc::O_EXEC),
            immutable_expectation(b"sealed"),
        )
        .expect_err("O_EXEC is not a data-readable descriptor");
    assert_eq!(
        execute_only_error.kind(),
        GenerationRootErrorKind::InvalidDescriptorAccess
    );

    let observed = boundaries.lock().expect("boundary log should lock");
    assert!(
        !observed.iter().any(|boundary| matches!(
            boundary,
            TestBoundary::BeforeRead { .. }
                | TestBoundary::BeforeLock
                | TestBoundary::BeforeFileSync
                | TestBoundary::BeforeAclRead
                | TestBoundary::BeforeFinalRouteStat
        )),
        "type/access/status/CLOEXEC rejection must precede ACL, route, content, flock, and sync"
    );
    drop(observed);
    drop(guard);
}

#[test]
fn macos_preopened_lock_candidate_rejects_persistent_fwaslocked_history_before_io() {
    const DARWIN_FWASLOCKED: u32 = 0x0000_4000;

    let root_path = fixture_root("preopened-fwaslocked-history");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let control = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("fresh control descriptor should qualify");
    let candidate = preopened_control(&lock_path);
    rustix::fs::flock(
        &candidate,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    )
    .expect("the fresh candidate should take an uncontended kernel flock");
    rustix::fs::flock(&candidate, rustix::fs::FlockOperation::Unlock)
        .expect("the fixture should explicitly unlock before admission");
    let observed_status =
        rustix::fs::fcntl_getfl(&candidate).expect("candidate status flags should load");
    assert!(
        observed_status.contains(rustix::fs::OFlags::from_bits_retain(DARWIN_FWASLOCKED)),
        "Darwin must expose the kernel-private FWASLOCKED history bit after successful \
         flock plus unlock"
    );

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let error = control
        .try_lock_preopened(candidate, GenerationRootLockMode::Exclusive)
        .expect_err("a candidate with prior flock history is not fresh");
    assert_eq!(
        error.kind(),
        GenerationRootErrorKind::InvalidDescriptorAccess
    );
    assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
    assert!(
        boundaries
            .lock()
            .expect("boundary log should lock")
            .is_empty(),
        "exact status validation must reject before preopened qualification, ACL, route, \
         content, or flock boundaries"
    );
}

#[test]
fn macos_preopened_route_binding_rejects_siblings_and_post_admission_substitution() {
    let root_path = fixture_root("preopened-route-binding");
    let artifact_path = private_file(&root_path, "artifact", b"artifact");
    let sibling_path = private_file(&root_path, "sibling", b"sibling!");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");

    let sibling_error = root
        .admit_preopened_file(
            &confined("artifact"),
            preopened_immutable(&sibling_path),
            immutable_expectation(b"sibling!"),
        )
        .expect_err("a valid sibling descriptor must not satisfy another route");
    assert_eq!(sibling_error.kind(), GenerationRootErrorKind::ObjectChanged);

    let admitted = root
        .admit_preopened_file(
            &confined("artifact"),
            preopened_immutable(&artifact_path),
            immutable_expectation(b"artifact"),
        )
        .expect("the exact descriptor and route should qualify");
    fs::rename(&artifact_path, root_path.join("artifact-original"))
        .expect("the admitted artifact should move to a retained route");
    private_file(&root_path, "artifact", b"artifact");
    let file_error = admitted
        .sync_durable()
        .expect_err("route substitution must fence retained-descriptor sync");
    assert_eq!(file_error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(
        file_error.stage(),
        GenerationRootStage::RevalidateRegularFile
    );

    let control = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("the exact control descriptor and route should qualify");
    let fresh_lock_descriptor = preopened_control(&lock_path);
    fs::rename(&lock_path, root_path.join("LOCK-original"))
        .expect("the admitted control should move to a retained route");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let lock_reached = Arc::new(Mutex::new(false));
    let hook_lock_reached = Arc::clone(&lock_reached);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeLock {
            *hook_lock_reached.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    let error = control
        .try_lock_preopened(fresh_lock_descriptor, GenerationRootLockMode::Exclusive)
        .expect_err("route substitution must fence the fresh lock descriptor");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::RevalidateRegularFile);
    assert!(
        !*lock_reached.lock().expect("hook state should lock"),
        "route revalidation must reject before flock"
    );
}

#[test]
fn macos_absolute_root_route_rejects_ancestor_and_final_symlinks() {
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
fn macos_preopened_rejects_ancestor_and_final_symlinks_without_touching_decoys() {
    let root_path = fixture_root("preopened-symlink-routes");
    let decoy_directory = private_dir(&root_path, "decoy-directory");
    let ancestor_target = private_file(&decoy_directory, "artifact", b"ancestor decoy stays exact");
    let final_target = private_file(&root_path, "final-target", b"final decoy stays exact");
    symlink(&decoy_directory, root_path.join("ancestor-link"))
        .expect("ancestor symlink should be creatable");
    symlink(&final_target, root_path.join("final-link"))
        .expect("final symlink should be creatable");

    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let _hook = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("symlink boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let ancestor_error = root
        .admit_preopened_file(
            &confined("ancestor-link/artifact"),
            preopened_immutable(&ancestor_target),
            immutable_expectation(b"ancestor decoy stays exact"),
        )
        .expect_err("an ancestor symlink must not bind a trusted descriptor");
    assert_eq!(ancestor_error.kind(), GenerationRootErrorKind::SymbolicLink);
    assert_eq!(ancestor_error.component_index(), Some(0));

    let final_error = root
        .admit_preopened_file(
            &confined("final-link"),
            preopened_immutable(&final_target),
            immutable_expectation(b"final decoy stays exact"),
        )
        .expect_err("a final symlink must not bind a trusted descriptor");
    assert_eq!(final_error.kind(), GenerationRootErrorKind::SymbolicLink);
    assert_eq!(final_error.component_index(), Some(0));

    let observed = boundaries.lock().expect("symlink boundary log should lock");
    assert!(
        !observed.iter().any(|boundary| matches!(
            boundary,
            TestBoundary::BeforeRead { .. }
                | TestBoundary::BeforeTrailingByteProbe
                | TestBoundary::AfterExactRead
        )),
        "route-bound symlink rejection must precede every content-read boundary"
    );
    drop(observed);

    assert_eq!(
        fs::read(&ancestor_target).expect("ancestor decoy should remain readable"),
        b"ancestor decoy stays exact"
    );
    assert_eq!(
        fs::read(&final_target).expect("final decoy should remain readable"),
        b"final decoy stays exact"
    );
}
