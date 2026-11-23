use super::super::super::platform::{
    TestBoundary, install_test_hook, validate_regular_descriptor_type, verify_exact_directory_entry,
};
use super::super::super::{
    GenerationFileExpectation, GenerationRootError, GenerationRootErrorKind,
    GenerationRootLockMode, GenerationRootStage,
    QualifiedGenerationDirectory as QualifiedGenerationRoot,
};
use super::*;
use std::ffi::OsStr;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt, symlink};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[test]
fn exact_file_admission_retains_bytes_identity_and_durability() {
    let root_path = fixture_root("happy");
    let generation = private_dir(&root_path, "generation");
    let contents = b"immutable generation image";
    private_file(&generation, "vector.fsvi", contents);

    let root = QualifiedGenerationRoot::admit(&root_path).expect("ext4 root should qualify");
    assert_eq!(root.witness().mode() & 0o7777, 0o700);
    let admitted = root
        .admit_file(
            &confined("generation/vector.fsvi"),
            immutable_expectation(contents),
        )
        .expect("exact immutable file should be admitted");
    assert_eq!(admitted.as_bytes(), contents);
    assert_eq!(admitted.bytes().as_ref(), contents);
    assert_eq!(admitted.sha256(), digest(contents));
    assert_eq!(admitted.witness().byte_len(), contents.len() as u64);
    assert_eq!(admitted.witness().hard_links(), 1);
    assert_eq!(admitted.witness().mode() & 0o7777, 0o400);
    admitted
        .sync_durable()
        .expect("retained regular-file fsync should succeed");
    root.sync_directory_durable()
        .expect("retained root fsync should succeed");

    let debug_root = format!("{root:?}");
    let debug_file = format!("{admitted:?}");
    assert!(!debug_root.contains(root_path.to_string_lossy().as_ref()));
    assert!(!debug_file.contains("vector.fsvi"));
    assert!(!debug_file.contains("immutable generation image"));
}

#[test]
fn route_and_error_diagnostics_never_disclose_path_bytes() {
    let root_path = fixture_root("redaction");
    let secret_component = "secret-customer-generation";
    let path = confined(secret_component);
    let error = QualifiedGenerationRoot::admit(Path::new("relative-root"))
        .expect_err("relative root route must fail");
    let display = error.to_string();
    let debug = format!("{error:?}");
    let path_debug = format!("{path:?}");
    assert!(!display.contains("relative-root"));
    assert!(!debug.contains("relative-root"));
    assert!(!path_debug.contains(secret_component));
    assert!(!path_debug.contains(root_path.to_string_lossy().as_ref()));
}

#[test]
fn exact_directory_enumeration_rejects_case_distinct_component_bytes() {
    use rustix::fs::{Mode, OFlags, open};

    let root_path = fixture_root("exact-component-bytes");
    private_file(&root_path, "ExactName", b"x");
    let descriptor = open(
        &root_path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .expect("fixture directory should open");
    verify_exact_directory_entry(
        &descriptor,
        OsStr::new("ExactName"),
        GenerationRootStage::OpenRegularFile,
        0,
    )
    .expect("the exact component bytes should be present once");
    assert_eq!(
        verify_exact_directory_entry(
            &descriptor,
            OsStr::new("exactname"),
            GenerationRootStage::OpenRegularFile,
            0,
        )
        .expect_err("case-distinct bytes must never count as the exact route")
        .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn root_and_descendant_modes_are_exactly_private() {
    let bad_root = fixture_root("bad-root-mode");
    fs::set_permissions(&bad_root, fs::Permissions::from_mode(0o750))
        .expect("fixture mode should change");
    assert_eq!(
        QualifiedGenerationRoot::admit(&bad_root)
            .expect_err("non-private root must fail")
            .kind(),
        GenerationRootErrorKind::WrongMode
    );

    let root_path = fixture_root("bad-descendant-mode");
    let directory = private_dir(&root_path, "generation");
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o750))
        .expect("fixture mode should change");
    private_file(&directory, "artifact", b"x");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    assert_eq!(
        root.admit_file(
            &confined("generation/artifact"),
            immutable_expectation(b"x")
        )
        .expect_err("non-private descendant directory must fail")
        .kind(),
        GenerationRootErrorKind::WrongMode
    );

    let root_path = fixture_root("bad-file-mode");
    let artifact = private_file(&root_path, "artifact", b"x");
    fs::set_permissions(&artifact, fs::Permissions::from_mode(0o640))
        .expect("fixture mode should change");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    assert_eq!(
        root.admit_file(&confined("artifact"), immutable_expectation(b"x"))
            .expect_err("non-private regular file must fail")
            .kind(),
        GenerationRootErrorKind::WrongMode
    );
}

#[test]
fn linux_rejects_a_granting_default_acl_on_the_root() {
    let root_path = fixture_root("root-default-acl");
    let retained = File::open(&root_path).expect("private root should open");
    install_linux_acl_xattr(&retained, "system.posix_acl_default", 0o7, 0o4, 0, 0o4, 0);
    assert_eq!(
        crate::fd_acl::extended_acl_presence(retained.as_fd())
            .expect("descriptor-bound default ACL probe should succeed"),
        crate::fd_acl::ExtendedAclPresence::Present
    );
    let error = QualifiedGenerationRoot::admit(&root_path)
        .expect_err("a granting default ACL must fail root admission");
    assert_eq!(error.kind(), GenerationRootErrorKind::AclRejected);
    assert_eq!(error.stage(), GenerationRootStage::InspectAcl);
}

#[test]
fn linux_rejects_acl_bearing_artifact_control_and_lock_descriptors_before_content_io() {
    let artifact_root = fixture_root("artifact-access-acl");
    let artifact_path = private_file(&artifact_root, "artifact", b"sealed");
    let artifact_acl =
        File::open(&artifact_path).expect("immutable artifact should open for ACL setup");
    install_linux_acl_xattr(&artifact_acl, "system.posix_acl_access", 0o4, 0, 0, 0, 0);
    let artifact_generation =
        QualifiedGenerationRoot::admit(&artifact_root).expect("the ACL-free root should qualify");
    let artifact_boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_artifact_boundaries = Arc::clone(&artifact_boundaries);
    let artifact_hook = install_test_hook(move |boundary| {
        hook_artifact_boundaries
            .lock()
            .expect("artifact boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let artifact_error = artifact_generation
        .admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
        .expect_err("an ACL-bearing artifact must fail");
    assert_eq!(artifact_error.kind(), GenerationRootErrorKind::AclRejected);
    assert_eq!(artifact_error.stage(), GenerationRootStage::InspectAcl);
    let artifact_observed = artifact_boundaries
        .lock()
        .expect("artifact boundary log should lock");
    assert!(artifact_observed.contains(&TestBoundary::BeforeAclRead));
    let after_data_reopen = artifact_observed
        .iter()
        .position(|boundary| *boundary == (TestBoundary::AfterProcFdReopen { index: 0 }))
        .expect("Linux ACL inspection requires the real proc-derived data descriptor");
    let before_acl = artifact_observed
        .iter()
        .enumerate()
        .skip(after_data_reopen.saturating_add(1))
        .find_map(|(index, boundary)| (*boundary == TestBoundary::BeforeAclRead).then_some(index))
        .expect("ACL inspection boundary should be recorded");
    assert!(
        before_acl > after_data_reopen,
        "the O_PATH probe must complete without fgetxattr; ACL inspection belongs only \
         to the real proc-derived data descriptor"
    );
    assert!(
        !artifact_observed
            .iter()
            .any(|boundary| matches!(boundary, TestBoundary::BeforeRead { .. })),
        "ACL rejection must precede the first content read"
    );
    drop(artifact_observed);
    drop(artifact_hook);

    let control_root = fixture_root("control-access-acl");
    let control_path = control_file(&control_root, "LOCK", b"lock-v1!");
    let control_acl = File::open(&control_path).expect("control file should open for ACL setup");
    install_linux_acl_xattr(&control_acl, "system.posix_acl_access", 0o6, 0, 0, 0, 0);
    let control_generation =
        QualifiedGenerationRoot::admit(&control_root).expect("the ACL-free root should qualify");
    let control_error = control_generation
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect_err("an ACL-bearing control descriptor must fail");
    assert_eq!(control_error.kind(), GenerationRootErrorKind::AclRejected);
    assert_eq!(control_error.stage(), GenerationRootStage::InspectAcl);

    let lock_root = fixture_root("lock-access-acl");
    let lock_path = control_file(&lock_root, "LOCK", b"lock-v1!");
    let lock_generation =
        QualifiedGenerationRoot::admit(&lock_root).expect("the ACL-free root should qualify");
    let control = lock_generation
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("the ACL-free control should qualify before target mutation");
    let reached_flock = Arc::new(Mutex::new(false));
    let hook_reached_flock = Arc::clone(&reached_flock);
    let acl_mutated = Arc::new(Mutex::new(false));
    let hook_acl_mutated = Arc::clone(&acl_mutated);
    let hook_lock_path = lock_path.clone();
    let mut after_data_reopen = false;
    let _hook = install_test_hook(move |boundary| {
        if boundary == (TestBoundary::AfterProcFdReopen { index: 0 }) {
            after_data_reopen = true;
        }
        if after_data_reopen
            && boundary == TestBoundary::BeforeAclRead
            && !*hook_acl_mutated.lock().expect("hook state should lock")
        {
            let descriptor = File::open(&hook_lock_path)
                .expect("lock target should open for injected ACL mutation");
            install_linux_acl_xattr(&descriptor, "system.posix_acl_access", 0o6, 0, 0, 0, 0);
            *hook_acl_mutated.lock().expect("hook state should lock") = true;
        }
        if boundary == TestBoundary::BeforeLock {
            *hook_reached_flock.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    let lock_error = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect_err("an ACL-bearing lock descriptor must fail before flock");
    assert_eq!(lock_error.kind(), GenerationRootErrorKind::AclRejected);
    assert_eq!(lock_error.stage(), GenerationRootStage::InspectAcl);
    assert!(
        *acl_mutated.lock().expect("hook state should lock"),
        "the lock candidate ACL must be injected after the real data reopen"
    );
    assert!(
        !*reached_flock.lock().expect("hook state should lock"),
        "ACL rejection must precede flock"
    );
}

#[test]
fn ancestor_and_final_symlinks_fail_without_touching_decoys() {
    let root_path = fixture_root("symlink");
    let decoy = fixture_root("decoy");
    let decoy_file = private_file(&decoy, "artifact", b"decoy stays exact");
    symlink(&decoy, root_path.join("jump")).expect("ancestor symlink should be creatable");
    symlink(&decoy_file, root_path.join("final")).expect("final symlink should be creatable");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");

    for route in ["jump/artifact", "final"] {
        let error = root
            .admit_file(
                &confined(route),
                GenerationFileExpectation::immutable(17, [0; 32])
                    .expect("small expectation should be accepted"),
            )
            .expect_err("symlink route must fail");
        assert_eq!(error.kind(), GenerationRootErrorKind::SymbolicLink);
    }
    assert_eq!(
        fs::read(&decoy_file).expect("decoy should remain readable"),
        b"decoy stays exact"
    );
}

#[test]
fn hardlinks_and_special_files_are_never_admitted() {
    use std::os::unix::net::UnixListener;

    let root_path = fixture_root("special");
    let linked = private_file(&root_path, "linked", b"linked");
    fs::hard_link(&linked, root_path.join("linked-alias"))
        .expect("fixture hardlink should be creatable");
    private_dir(&root_path, "directory-final");
    rustix::fs::mkfifoat(
        rustix::fs::CWD,
        root_path.join("fifo-final"),
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    )
    .expect("fixture FIFO should be creatable");
    let root_descriptor = File::open(&root_path).expect("fixture root descriptor should open");
    let short_socket_route = format!("/proc/self/fd/{}/socket-final", root_descriptor.as_raw_fd());
    let _socket = UnixListener::bind(&short_socket_route)
        .expect("fixture socket should be creatable through a short retained-fd route");

    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    assert_eq!(
        root.admit_file(&confined("linked"), immutable_expectation(b"linked"))
            .expect_err("hardlinked file must fail")
            .kind(),
        GenerationRootErrorKind::HardLinked
    );
    assert_eq!(
        root.admit_file(
            &confined("directory-final"),
            GenerationFileExpectation::immutable(0, digest(b""))
                .expect("zero-length expectation should be accepted"),
        )
        .expect_err("directory must not be admitted as a regular file")
        .kind(),
        GenerationRootErrorKind::NotRegularFile
    );
    for route in ["fifo-final", "socket-final"] {
        assert!(
            root.admit_file(
                &confined(route),
                GenerationFileExpectation::immutable(0, digest(b""))
                    .expect("zero-length expectation should be accepted"),
            )
            .is_err(),
            "special file {route} must fail closed"
        );
    }
}

#[test]
fn linux_final_probe_rejects_fifo_before_endpoint_io_procfd_reopen_or_read() {
    let root_path = fixture_root("fifo-preflight");
    let fifo_path = root_path.join("fifo-final");
    rustix::fs::mkfifoat(
        rustix::fs::CWD,
        &fifo_path,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    )
    .expect("fixture FIFO should be creatable");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let hook_fifo = fifo_path.clone();
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        if boundary == (TestBoundary::AfterFinalProbeOpen { index: 0 }) {
            let error = OpenOptions::new()
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&hook_fifo)
                .expect_err("O_PATH preflight must not establish a FIFO reader endpoint");
            assert_eq!(
                error.raw_os_error(),
                Some(libc::ENXIO),
                "a nonblocking FIFO writer must still observe no reader endpoint"
            );
        }
        Ok(())
    });

    for error in [
        root.admit_file(
            &confined("fifo-final"),
            GenerationFileExpectation::immutable(0, digest(b""))
                .expect("zero-length expectation should be accepted"),
        )
        .expect_err("immutable FIFO must fail at type preflight"),
        root.admit_control_file(
            &confined("fifo-final"),
            GenerationFileExpectation::control(0)
                .expect("zero-length control expectation should be accepted"),
        )
        .expect_err("control FIFO must fail at type preflight"),
    ] {
        assert_eq!(error.kind(), GenerationRootErrorKind::NotRegularFile);
        assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
    }
    let observed = boundaries.lock().expect("boundary log should lock");
    assert!(
        observed.contains(&TestBoundary::AfterFinalProbeOpen { index: 0 }),
        "the no-I/O O_PATH probe should be observable"
    );
    // Root revalidation legitimately opens the verified procfs root
    // to bind task-scoped mountinfo before inspecting the final
    // component. `BeforeProcFdReopen` is the distinct boundary that
    // would reopen this probed FIFO as a data descriptor.
    assert!(
        !observed.iter().any(|boundary| matches!(
            boundary,
            TestBoundary::BeforeProcFdReopen { .. }
                | TestBoundary::BeforeRead { .. }
                | TestBoundary::BeforeLock
                | TestBoundary::BeforeFileSync
        )),
        "special nodes must fail before final-capability reopen or any content operation"
    );
    drop(observed);
}

#[test]
fn linux_final_probe_rejects_a_real_device_descriptor_without_io_reopen() {
    use rustix::fs::{CWD, ResolveFlags, openat2};

    let descriptor = openat2(
        CWD,
        Path::new("/dev/null"),
        rustix::fs::OFlags::PATH | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )
    .expect("the physical device should support a side-effect-free O_PATH reference");
    let error =
        validate_regular_descriptor_type(&descriptor, GenerationRootStage::OpenRegularFile, 0)
            .expect_err("the real device must fail the production type gate");
    assert_eq!(error.kind(), GenerationRootErrorKind::NotRegularFile);
    assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
}

#[test]
fn linux_regular_to_fifo_swap_after_probe_never_touches_the_replacement() {
    let root_path = fixture_root("regular-to-fifo-after-probe");
    private_file(&root_path, "artifact", b"sealed");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let writer_probe = Arc::new(Mutex::new(None));
    let hook_writer_probe = Arc::clone(&writer_probe);
    let hook_root = root_path.clone();
    let mut swapped = false;
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        if !swapped && boundary == (TestBoundary::AfterProbeQualified { index: 0 }) {
            fs::rename(
                hook_root.join("artifact"),
                hook_root.join("artifact-original"),
            )
            .expect("the probed regular file should move without invalidating its fd");
            rustix::fs::mkfifoat(
                rustix::fs::CWD,
                hook_root.join("artifact"),
                rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            )
            .expect("the canonical route should be replaceable with a FIFO fixture");
            swapped = true;
        }
        if boundary == (TestBoundary::AfterProcFdReopen { index: 0 }) {
            let observed_errno = match OpenOptions::new()
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(hook_root.join("artifact"))
            {
                Ok(writer) => {
                    drop(writer);
                    None
                }
                Err(error) => error.raw_os_error(),
            };
            *hook_writer_probe
                .lock()
                .expect("writer probe state should lock") = Some(observed_errno);
        }
        Ok(())
    });
    let error = root
        .admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
        .expect_err("the final route reprobe must reject a special-node substitution");
    assert!(
        matches!(
            error.kind(),
            GenerationRootErrorKind::NotRegularFile | GenerationRootErrorKind::ObjectChanged
        ),
        "the substitution must fail as a type or identity change"
    );
    assert_eq!(
        *writer_probe.lock().expect("writer probe state should lock"),
        Some(Some(libc::ENXIO)),
        "after the proc-derived data reopen, a nonblocking FIFO writer must still observe \
         no reader; otherwise the retained descriptor touched the replacement FIFO"
    );
    assert!(
        !boundaries
            .lock()
            .expect("boundary log should lock")
            .iter()
            .any(|boundary| matches!(boundary, TestBoundary::BeforeRead { .. })),
        "the replacement FIFO must never reach the content reader"
    );
}

#[test]
fn linux_filtered_procfd_capability_reopen_is_typed_unsupported() {
    let root_path = fixture_root("filtered-procfd");
    private_file(&root_path, "artifact", b"sealed");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let _guard = install_test_hook(|boundary| {
        if boundary == TestBoundary::BeforeProcCapabilityRootOpen {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedKernelFeature,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        Ok(())
    });
    let error = root
        .admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
        .expect_err("a filtered procfd capability bridge must fail closed");
    assert_eq!(
        error.kind(),
        GenerationRootErrorKind::UnsupportedKernelFeature
    );
    assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
}

#[test]
fn size_hash_grow_and_shrink_fail_closed() {
    let root_path = fixture_root("content");
    let contents = b"abcdefgh";
    let artifact = private_file(&root_path, "artifact", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");

    let wrong_size = GenerationFileExpectation::immutable(7, digest(b"abcdefg"))
        .expect("small expectation should be accepted");
    assert_eq!(
        root.admit_file(&confined("artifact"), wrong_size)
            .expect_err("wrong exact size must fail")
            .kind(),
        GenerationRootErrorKind::SizeMismatch
    );
    let wrong_hash = GenerationFileExpectation::immutable(8, [0x31; 32])
        .expect("small expectation should be accepted");
    assert_eq!(
        root.admit_file(&confined("artifact"), wrong_hash)
            .expect_err("wrong exact digest must fail")
            .kind(),
        GenerationRootErrorKind::HashMismatch
    );

    for grow in [true, false] {
        fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
            .expect("fixture should become writable for the retained test fd");
        let mut mutation_file = OpenOptions::new()
            .write(true)
            .open(&artifact)
            .expect("retained mutation fd should open");
        fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
            .expect("fixture should be resealed before admission");
        let guard = install_test_hook(move |boundary| {
            if boundary == (TestBoundary::BeforeRead { offset: 0 }) {
                if grow {
                    mutation_file
                        .seek(SeekFrom::End(0))
                        .expect("fixture should be seekable");
                    mutation_file.write_all(b"+").expect("fixture should grow");
                } else {
                    mutation_file.set_len(0).expect("fixture should shrink");
                }
                mutation_file
                    .sync_all()
                    .expect("size mutation should be durable");
            }
            Ok(())
        });
        assert!(
            root.admit_file(&confined("artifact"), immutable_expectation(contents))
                .is_err(),
            "mid-read size drift must fail closed"
        );
        drop(guard);
        fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
            .expect("fixture should become writable for reset");
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&artifact)
            .expect("fixture should be resettable");
        file.write_all(contents).expect("fixture should reset");
        file.sync_all().expect("reset should be durable");
        fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
            .expect("fixture should be resealed after reset");
    }
}

#[test]
fn same_size_rewrite_and_hardlink_race_fail_after_initial_validation() {
    let contents = b"original";

    let root_path = fixture_root("rewrite-race");
    let artifact = private_file(&root_path, "artifact", contents);
    fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
        .expect("fixture should become writable for the retained test fd");
    let mut rewrite_file = OpenOptions::new()
        .write(true)
        .open(&artifact)
        .expect("retained rewrite fd should open");
    fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
        .expect("fixture should be resealed before admission");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeFileRouteReopen {
            rewrite_file
                .seek(SeekFrom::Start(0))
                .expect("fixture should be seekable");
            rewrite_file
                .write_all(b"rewrite!")
                .expect("same-size rewrite should succeed");
            rewrite_file.sync_all().expect("rewrite should be durable");
        }
        Ok(())
    });
    assert_eq!(
        root.admit_file(&confined("artifact"), immutable_expectation(contents))
            .expect_err("same-size post-read rewrite must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
    drop(guard);

    let root_path = fixture_root("nlink-race");
    let artifact = private_file(&root_path, "artifact", contents);
    let alias = root_path.join("late-alias");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let _guard = install_test_hook(move |boundary| {
        if boundary == (TestBoundary::BeforeRead { offset: 0 }) {
            fs::hard_link(&artifact, &alias).expect("late hardlink should be creatable");
        }
        Ok(())
    });
    assert_eq!(
        root.admit_file(&confined("artifact"), immutable_expectation(contents))
            .expect_err("post-open hardlink must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn file_and_root_path_swaps_are_detected_while_retained_bytes_survive() {
    let contents = b"retained exact bytes";
    let root_path = fixture_root("retained");
    let artifact = private_file(&root_path, "artifact", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(&confined("artifact"), immutable_expectation(contents))
        .expect("exact file should be admitted");

    let retained_artifact = root_path.join("artifact-retained");
    fs::rename(&artifact, &retained_artifact)
        .expect("fixture file should move to a fresh retained name");
    private_file(&root_path, "artifact", b"x");
    assert_eq!(admitted.as_bytes(), contents);
    assert_eq!(
        root.admit_file(&confined("artifact"), immutable_expectation(contents))
            .expect_err("replacement file must not satisfy original digest")
            .kind(),
        GenerationRootErrorKind::SizeMismatch
    );

    let retained_root = root_path.with_extension("retained-root");
    fs::rename(&root_path, &retained_root)
        .expect("fixture root should move to a fresh retained name");
    DirBuilder::new()
        .mode(0o700)
        .create(&root_path)
        .expect("attacker root should be creatable");
    fs::set_permissions(&root_path, fs::Permissions::from_mode(0o700))
        .expect("attacker root mode should be settable");
    assert_eq!(
        root.revalidate_route()
            .expect_err("root pathname replacement must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
    assert_eq!(admitted.as_bytes(), contents);
}

#[test]
fn root_swap_during_initial_double_resolution_is_rejected() {
    let root_path = fixture_root("root-admission-race");
    let retained_root = root_path.with_extension("first");
    let swap_source = root_path.clone();
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::AfterFirstRootQualification {
            fs::rename(&swap_source, &retained_root)
                .expect("first root should move to a fresh name");
            DirBuilder::new()
                .mode(0o700)
                .create(&swap_source)
                .expect("replacement root should be creatable");
            fs::set_permissions(&swap_source, fs::Permissions::from_mode(0o700))
                .expect("replacement root mode should be settable");
        }
        Ok(())
    });
    assert_eq!(
        QualifiedGenerationRoot::admit(&root_path)
            .expect_err("mid-admission root swap must fail")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn intermediate_ancestor_substitution_is_detected_when_final_root_inode_is_unchanged() {
    let base = fixture_root("ancestor-chain");
    let ancestor = private_dir(&base, "ancestor");
    let root_path = private_dir(&ancestor, "root");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted_root_witness = root.witness();

    let displaced_ancestor = base.join("displaced-ancestor");
    fs::rename(&ancestor, &displaced_ancestor)
        .expect("original ancestor should move to a fresh retained name");
    let replacement_ancestor = private_dir(&base, "ancestor");
    fs::rename(
        displaced_ancestor.join("root"),
        replacement_ancestor.join("root"),
    )
    .expect("the same admitted root inode should move under the replacement ancestor");

    assert_eq!(
        fs::metadata(&root_path)
            .expect("canonical root route should resolve again")
            .ino(),
        admitted_root_witness.inode()
    );
    assert_eq!(
        root.revalidate_route()
            .expect_err("whole-route validation must reject a replaced intermediate ancestor")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn relative_ancestor_substitution_is_detected_when_final_file_inode_is_unchanged() {
    let root_path = fixture_root("relative-ancestor-chain");
    let ancestor = private_dir(&root_path, "ancestor");
    let nested = private_dir(&ancestor, "nested");
    let contents = b"same retained file";
    private_file(&nested, "artifact", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(
            &confined("ancestor/nested/artifact"),
            immutable_expectation(contents),
        )
        .expect("immutable file should qualify");
    let admitted_inode = admitted.witness().inode();

    let displaced_ancestor = root_path.join("displaced-ancestor");
    fs::rename(&ancestor, &displaced_ancestor)
        .expect("original relative ancestor should move to a fresh retained name");
    let replacement_ancestor = private_dir(&root_path, "ancestor");
    fs::rename(
        displaced_ancestor.join("nested"),
        replacement_ancestor.join("nested"),
    )
    .expect("the original final subtree should move below the replacement ancestor");

    assert_eq!(
        fs::metadata(root_path.join("ancestor/nested/artifact"))
            .expect("canonical final file route should resolve again")
            .ino(),
        admitted_inode
    );
    assert_eq!(
        admitted
            .sync_durable()
            .expect_err("the full relative identity chain must reject a replaced ancestor")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
}

#[test]
fn control_ancestor_substitution_fails_before_flock_when_final_inode_is_unchanged() {
    let root_path = fixture_root("control-ancestor-chain");
    let ancestor = private_dir(&root_path, "ancestor");
    let nested = private_dir(&ancestor, "nested");
    control_file(&nested, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let control = root
        .admit_control_file(
            &confined("ancestor/nested/LOCK"),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("control file should qualify");
    let admitted_inode = control.witness().inode();

    let displaced_ancestor = root_path.join("displaced-ancestor");
    fs::rename(&ancestor, &displaced_ancestor)
        .expect("original relative ancestor should move to a fresh retained name");
    let replacement_ancestor = private_dir(&root_path, "ancestor");
    fs::rename(
        displaced_ancestor.join("nested"),
        replacement_ancestor.join("nested"),
    )
    .expect("the original control subtree should move below the replacement ancestor");

    assert_eq!(
        fs::metadata(root_path.join("ancestor/nested/LOCK"))
            .expect("canonical final control route should resolve again")
            .ino(),
        admitted_inode
    );
    let lock_boundary_reached = Arc::new(Mutex::new(false));
    let hook_lock_boundary_reached = Arc::clone(&lock_boundary_reached);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeLock {
            *hook_lock_boundary_reached
                .lock()
                .expect("hook state should lock") = true;
        }
        Ok(())
    });
    assert_eq!(
        control
            .try_lock(GenerationRootLockMode::Exclusive)
            .expect_err("relative ancestor replacement must fail before flock")
            .kind(),
        GenerationRootErrorKind::ObjectChanged
    );
    assert!(
        !*lock_boundary_reached
            .lock()
            .expect("hook state should lock"),
        "relative route validation must precede flock"
    );
}
