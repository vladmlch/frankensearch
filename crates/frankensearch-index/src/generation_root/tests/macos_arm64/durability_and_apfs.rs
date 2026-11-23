use super::super::super::platform::{TestBoundary, install_test_hook};
use super::super::super::{
    GenerationFileExpectation, GenerationRootError, GenerationRootErrorKind,
    GenerationRootLockMode, GenerationRootStage,
    QualifiedGenerationDirectory as QualifiedGenerationRoot,
};
use super::*;
use std::fs::{self, DirBuilder, File};
use std::io::Read;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[test]
fn macos_immutable_content_drift_fences_sync_before_durability() {
    let root_path = fixture_root("mapped-immutable-before-sync");
    let (artifact_path, mapping) = mapped_immutable_file(&root_path, "artifact");
    let admitted_bytes =
        fs::read(&artifact_path).expect("mapped APFS immutable fixture must be readable");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let admitted = root
        .admit_preopened_file(
            &confined("artifact"),
            preopened_immutable(&artifact_path),
            immutable_expectation(&admitted_bytes),
        )
        .expect("mapped APFS immutable fixture should qualify before ambient mutation");

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let mut mapping = Some(mapping);
    let mut mutated = false;
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        if !mutated && boundary == (TestBoundary::BeforeRead { offset: 0 }) {
            mutate_mapping(
                mapping
                    .as_mut()
                    .expect("shared APFS immutable mapping must remain retained"),
            );
            mutated = true;
        }
        Ok(())
    });
    let error = admitted
        .sync_durable()
        .expect_err("pre-durability APFS immutable content drift must fail closed");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
    assert!(
        !boundaries
            .lock()
            .expect("boundary log should lock")
            .contains(&TestBoundary::BeforeFileSync),
        "APFS immutable validation must fail before F_FULLFSYNC"
    );
}

#[test]
fn macos_immutable_content_drift_during_durability_is_detected_after_barrier() {
    let root_path = fixture_root("mapped-immutable-during-sync");
    let (artifact_path, mapping) = mapped_immutable_file(&root_path, "artifact");
    let admitted_bytes =
        fs::read(&artifact_path).expect("mapped APFS immutable fixture must be readable");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let admitted = root
        .admit_preopened_file(
            &confined("artifact"),
            preopened_immutable(&artifact_path),
            immutable_expectation(&admitted_bytes),
        )
        .expect("mapped APFS immutable fixture should qualify before ambient mutation");

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let mut mapping = Some(mapping);
    let mut mutated = false;
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        if !mutated && boundary == TestBoundary::BeforeFileSync {
            mutate_mapping(
                mapping
                    .as_mut()
                    .expect("shared APFS immutable mapping must remain retained"),
            );
            mutated = true;
        }
        Ok(())
    });
    let error = admitted
        .sync_durable()
        .expect_err("APFS immutable mutation at durability must fail after the barrier");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
    let observed = boundaries.lock().expect("boundary log should lock");
    let after_sync = observed
        .iter()
        .position(|boundary| *boundary == TestBoundary::AfterFileSync)
        .expect("F_FULLFSYNC should complete before post-check failure");
    assert!(
        observed.iter().enumerate().any(|(index, boundary)| {
            index > after_sync && *boundary == (TestBoundary::BeforeRead { offset: 0 })
        }),
        "a second exact APFS immutable read must follow F_FULLFSYNC"
    );
    drop(observed);
}

#[test]
fn macos_control_without_caller_digest_rejects_preheld_mapping_mutation_after_flock() {
    let root_path = fixture_root("mapped-control-after-flock");
    let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
    let byte_len = fs::metadata(&lock_path)
        .expect("mapped APFS control fixture metadata should load")
        .len();
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let expectation = GenerationFileExpectation::control(byte_len)
        .expect("mapped APFS control fixture length should be accepted");
    let control = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            expectation,
        )
        .expect("mapped APFS control fixture should qualify before ambient mutation");

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let mut mapping = Some(mapping);
    let mut locked = false;
    let mut mutated = false;
    let guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        if boundary == TestBoundary::AfterLock {
            locked = true;
        }
        if locked && !mutated && boundary == (TestBoundary::BeforeRead { offset: 0 }) {
            mutate_mapping(
                mapping
                    .as_mut()
                    .expect("shared APFS mapping should remain retained by the hook"),
            );
            mutated = true;
        }
        Ok(())
    });
    let error = control
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect_err("post-flock APFS mapped mutation must invalidate the bound image");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
    let observed = boundaries.lock().expect("boundary log should lock");
    let before_lock = observed
        .iter()
        .position(|boundary| *boundary == TestBoundary::BeforeLock)
        .expect("flock attempt boundary should be observed");
    let after_lock = observed
        .iter()
        .position(|boundary| *boundary == TestBoundary::AfterLock)
        .expect("successful flock boundary should be observed");
    let first_locked_read = observed
        .iter()
        .enumerate()
        .find_map(|(index, boundary)| {
            (index > after_lock && *boundary == (TestBoundary::BeforeRead { offset: 0 }))
                .then_some(index)
        })
        .expect("post-flock content validation should read the descriptor");
    assert!(before_lock < after_lock && after_lock < first_locked_read);
    drop(observed);
    drop(guard);

    let fresh = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            expectation,
        )
        .expect("the mutated APFS image should support a fresh bound admission");
    fresh
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect("failed validation must not leak the APFS kernel flock")
        .unlock()
        .expect("fresh APFS lock should release cleanly");
}

#[test]
fn macos_control_content_drift_fences_sync_before_durability() {
    let root_path = fixture_root("mapped-control-before-sync");
    let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
    let byte_len = fs::metadata(&lock_path)
        .expect("mapped APFS control fixture metadata should load")
        .len();
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let lock = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(byte_len)
                .expect("mapped APFS control fixture length should be accepted"),
        )
        .expect("mapped APFS control fixture should qualify")
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect("mapped APFS control fixture should lock");

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let mut mapping = Some(mapping);
    let mut mutated = false;
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        if !mutated && boundary == (TestBoundary::BeforeRead { offset: 0 }) {
            mutate_mapping(
                mapping
                    .as_mut()
                    .expect("shared APFS mapping should remain retained by the hook"),
            );
            mutated = true;
        }
        Ok(())
    });
    let error = lock
        .sync_durable()
        .expect_err("pre-durability APFS content drift must fail closed");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
    assert!(
        !boundaries
            .lock()
            .expect("boundary log should lock")
            .contains(&TestBoundary::BeforeFileSync),
        "APFS content validation must fail before the durability syscall"
    );
}

#[test]
fn macos_control_content_drift_during_durability_is_detected_after_barrier() {
    let root_path = fixture_root("mapped-control-during-sync");
    let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
    let byte_len = fs::metadata(&lock_path)
        .expect("mapped APFS control fixture metadata should load")
        .len();
    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let lock = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(byte_len)
                .expect("mapped APFS control fixture length should be accepted"),
        )
        .expect("mapped APFS control fixture should qualify")
        .try_lock_preopened(
            preopened_control(&lock_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect("mapped APFS control fixture should lock");

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let mut mapping = Some(mapping);
    let mut mutated = false;
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        if !mutated && boundary == TestBoundary::BeforeFileSync {
            mutate_mapping(
                mapping
                    .as_mut()
                    .expect("shared APFS mapping should remain retained by the hook"),
            );
            mutated = true;
        }
        Ok(())
    });
    let error = lock
        .sync_durable()
        .expect_err("APFS mutation at durability must fail after the barrier");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
    let observed = boundaries.lock().expect("boundary log should lock");
    let after_sync = observed
        .iter()
        .position(|boundary| *boundary == TestBoundary::AfterFileSync)
        .expect("the APFS durability barrier should complete before post-check failure");
    assert!(
        observed.iter().enumerate().any(|(index, boundary)| {
            index > after_sync && *boundary == (TestBoundary::BeforeRead { offset: 0 })
        }),
        "a second exact APFS content read must follow the durability barrier"
    );
    drop(observed);
}

#[test]
fn physical_apfs_rejects_every_non_empty_allow_or_deny_acl() {
    for (label, allow) in [("allow-acl", true), ("deny-acl", false)] {
        let root_path = fixture_root(label);
        set_non_empty_acl(&root_path, allow);

        let error = QualifiedGenerationRoot::admit(&root_path)
            .expect_err("every non-empty APFS ACL must fail closed");
        assert_eq!(error.kind(), GenerationRootErrorKind::AclRejected);
        assert_eq!(error.stage(), GenerationRootStage::InspectAcl);
        assert!(
            error.to_string().contains("descriptor-bound absence"),
            "operator diagnostic must explain the descriptor-bound fail-closed policy"
        );
    }
}

#[test]
fn physical_apfs_acl_add_remove_cycle_rejects_then_admits_same_root() {
    let root_path = fixture_root("acl-add-remove-cycle");
    set_non_empty_acl(&root_path, true);

    let error = QualifiedGenerationRoot::admit(&root_path)
        .expect_err("the ACL-bearing APFS root must fail closed");
    assert_eq!(error.kind(), GenerationRootErrorKind::AclRejected);
    assert_eq!(error.stage(), GenerationRootStage::InspectAcl);

    set_empty_acl(&root_path);
    QualifiedGenerationRoot::admit(&root_path)
        .expect("the same APFS root must qualify after its ACL is removed");
}

#[test]
fn physical_apfs_rejects_acl_bearing_artifact_control_and_lock_targets() {
    let artifact_root = fixture_root("artifact-target-acl");
    let artifact_path = private_file(&artifact_root, "artifact", b"sealed");
    set_non_empty_acl(&artifact_path, true);
    let artifact_generation = QualifiedGenerationRoot::admit(&artifact_root)
        .expect("the empty-ACL APFS root should qualify");
    assert_eq!(
        artifact_generation
            .admit_preopened_file(
                &confined("artifact"),
                preopened_immutable(&artifact_path),
                immutable_expectation(b"sealed"),
            )
            .expect_err("an ACL-bearing artifact descriptor must fail")
            .kind(),
        GenerationRootErrorKind::AclRejected
    );

    let control_root = fixture_root("control-target-acl");
    let control_path = control_file(&control_root, "LOCK", b"lock-v1!");
    set_non_empty_acl(&control_path, true);
    let control_generation = QualifiedGenerationRoot::admit(&control_root)
        .expect("the empty-ACL APFS root should qualify");
    assert_eq!(
        control_generation
            .admit_preopened_control_file(
                &confined("LOCK"),
                preopened_control(&control_path),
                GenerationFileExpectation::control(8)
                    .expect("small control expectation should be accepted"),
            )
            .expect_err("an ACL-bearing control descriptor must fail")
            .kind(),
        GenerationRootErrorKind::AclRejected
    );

    let lock_root = fixture_root("lock-target-acl");
    let lock_path = control_file(&lock_root, "LOCK", b"lock-v1!");
    let lock_generation =
        QualifiedGenerationRoot::admit(&lock_root).expect("the empty-ACL APFS root should qualify");
    let control = lock_generation
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("the empty-ACL control should qualify before target mutation");
    let lock_candidate = preopened_control(&lock_path);
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let hook_lock_path = lock_path.clone();
    let mut mutated = false;
    let _hook = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("hook state should lock")
            .push(boundary);
        if !mutated && boundary == TestBoundary::BeforePreopenedQualification {
            set_non_empty_acl(&hook_lock_path, true);
            mutated = true;
        }
        Ok(())
    });
    assert_eq!(
        control
            .try_lock_preopened(lock_candidate, GenerationRootLockMode::Exclusive,)
            .expect_err("the ACL-bearing fresh lock candidate must fail before flock")
            .kind(),
        GenerationRootErrorKind::AclRejected
    );
    let observed = boundaries.lock().expect("hook state should lock");
    assert!(
        observed.contains(&TestBoundary::BeforePreopenedQualification),
        "the retained control route must revalidate before candidate mutation"
    );
    assert!(
        observed.contains(&TestBoundary::BeforeAclRead),
        "the fresh candidate must reach the descriptor-bound ACL probe"
    );
    assert!(
        !observed.contains(&TestBoundary::BeforeLock),
        "fresh-candidate ACL rejection must precede flock"
    );
    drop(observed);
}

#[test]
fn physical_apfs_rejects_nonzero_flags_on_artifact_control_and_lock_targets() {
    for (label, flag) in [
        ("artifact-immutable-flag", "uchg"),
        ("artifact-nodump-flag", "nodump"),
    ] {
        let root_path = fixture_root(label);
        let artifact_path = private_file(&root_path, "artifact", b"sealed");
        let root = QualifiedGenerationRoot::admit(&root_path)
            .expect("the zero-flags APFS root should qualify");
        run_chflags(flag, &artifact_path);
        let observed_flags = fs::metadata(&artifact_path)
            .expect("flagged APFS artifact metadata should load")
            .st_flags();
        assert_ne!(observed_flags, 0, "chflags must establish the fixture");
        let error = root
            .admit_preopened_file(
                &confined("artifact"),
                preopened_immutable(&artifact_path),
                immutable_expectation(b"sealed"),
            )
            .expect_err("a flagged artifact must be outside the zero-flags profile");
        assert_eq!(
            error.kind(),
            GenerationRootErrorKind::UnsupportedObjectFlags
        );
        assert_eq!(error.expected(), Some(0));
        assert_eq!(error.observed(), Some(u64::from(observed_flags)));
    }

    let control_root = fixture_root("control-nodump-flag");
    let control_path = control_file(&control_root, "LOCK", b"lock-v1!");
    let control_generation = QualifiedGenerationRoot::admit(&control_root)
        .expect("the zero-flags APFS root should qualify");
    run_chflags("nodump", &control_path);
    assert_ne!(
        fs::metadata(&control_path)
            .expect("flagged APFS control metadata should load")
            .st_flags(),
        0
    );
    assert_eq!(
        control_generation
            .admit_preopened_control_file(
                &confined("LOCK"),
                preopened_control(&control_path),
                GenerationFileExpectation::control(8)
                    .expect("small control expectation should be accepted"),
            )
            .expect_err("a flagged control target must fail")
            .kind(),
        GenerationRootErrorKind::UnsupportedObjectFlags
    );

    let lock_root = fixture_root("lock-nodump-flag");
    let lock_path = control_file(&lock_root, "LOCK", b"lock-v1!");
    let lock_generation = QualifiedGenerationRoot::admit(&lock_root)
        .expect("the zero-flags APFS root should qualify");
    let control = lock_generation
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&lock_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("the zero-flags control should qualify before target mutation");
    let lock_candidate = preopened_control(&lock_path);
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let hook_lock_path = lock_path.clone();
    let mut mutated = false;
    let _hook = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("hook state should lock")
            .push(boundary);
        if !mutated && boundary == TestBoundary::BeforePreopenedQualification {
            run_chflags("nodump", &hook_lock_path);
            mutated = true;
        }
        Ok(())
    });
    let error = control
        .try_lock_preopened(lock_candidate, GenerationRootLockMode::Exclusive)
        .expect_err("the flagged fresh lock candidate must fail before flock");
    assert_eq!(
        error.kind(),
        GenerationRootErrorKind::UnsupportedObjectFlags
    );
    let observed = boundaries.lock().expect("hook state should lock");
    assert!(
        observed.contains(&TestBoundary::BeforePreopenedQualification),
        "the retained control route must revalidate before candidate mutation"
    );
    assert!(
        !observed.contains(&TestBoundary::BeforeLock),
        "fresh-candidate object-flag rejection must precede flock"
    );
    drop(observed);
}

#[test]
fn physical_apfs_acl_bearing_retained_descriptor_rejects_empty_replacement() {
    let root_path = fixture_root("retained-acl");
    set_non_empty_acl(&root_path, true);
    let displaced = root_path.with_extension("retained-acl-original");
    let canonical = root_path.clone();
    let substituted = Arc::new(Mutex::new(false));
    let hook_substituted = Arc::clone(&substituted);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeAclRead
            && !*hook_substituted.lock().expect("hook state should lock")
        {
            fs::rename(&canonical, &displaced)
                .expect("ACL-bearing root should move to a retained route");
            DirBuilder::new()
                .mode(0o700)
                .create(&canonical)
                .expect("empty-ACL replacement root should be creatable");
            fs::set_permissions(&canonical, fs::Permissions::from_mode(0o700))
                .expect("replacement root mode should be settable");
            set_empty_acl(&canonical);
            *hook_substituted.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    let error = QualifiedGenerationRoot::admit(&root_path)
        .expect_err("the ACL-bearing retained descriptor must reject before route fallback");
    assert_eq!(error.kind(), GenerationRootErrorKind::AclRejected);
    assert_eq!(error.stage(), GenerationRootStage::InspectAcl);
    assert!(*substituted.lock().expect("hook state should lock"));
}

#[test]
fn physical_apfs_empty_retained_descriptor_ignores_acl_bearing_replacement() {
    let root_path = fixture_root("retained-empty-acl");
    let displaced = root_path.with_extension("retained-empty-acl-original");
    let canonical = root_path.clone();
    let substituted = Arc::new(Mutex::new(false));
    let hook_substituted = Arc::clone(&substituted);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeAclRead
            && !*hook_substituted.lock().expect("hook state should lock")
        {
            fs::rename(&canonical, &displaced)
                .expect("empty-ACL root should move to a retained route");
            DirBuilder::new()
                .mode(0o700)
                .create(&canonical)
                .expect("replacement root should be creatable");
            fs::set_permissions(&canonical, fs::Permissions::from_mode(0o700))
                .expect("replacement root mode should be settable");
            set_non_empty_acl(&canonical, true);
            *hook_substituted.lock().expect("hook state should lock") = true;
        }
        Ok(())
    });
    let error = QualifiedGenerationRoot::admit(&root_path)
        .expect_err("canonical route replacement must fail after retained ACL inspection");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert!(*substituted.lock().expect("hook state should lock"));
}

#[test]
fn physical_apfs_excludes_regular_file_atime_only_drift_and_never_restores_it() {
    let root_path = fixture_root("regular-file-atime-exclusion");
    let artifact_path = private_file(&root_path, "artifact", b"atime");
    let control_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let future_modified = SystemTime::now() + Duration::from_secs(24 * 60 * 60);
    for path in [&artifact_path, &control_path] {
        let timestamp_owner = File::open(path).expect("fixture should open for timestamp setup");
        timestamp_owner
            .set_times(
                std::fs::FileTimes::new()
                    .set_accessed(UNIX_EPOCH)
                    .set_modified(future_modified),
            )
            .expect(
                "fixture atime must be old and mtime future-dated before admission so \
                 each physical read remains relatime-eligible",
            );
    }

    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let artifact = root
        .admit_preopened_file(
            &confined("artifact"),
            preopened_immutable(&artifact_path),
            immutable_expectation(b"atime"),
        )
        .expect("APFS artifact should qualify");
    let control = root
        .admit_preopened_control_file(
            &confined("LOCK"),
            preopened_control(&control_path),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("APFS control should qualify");
    let artifact_before = fs::metadata(&artifact_path)
        .expect("artifact metadata should load")
        .accessed()
        .expect("artifact atime should be readable");
    let control_before = fs::metadata(&control_path)
        .expect("control metadata should load")
        .accessed()
        .expect("control atime should be readable");

    std::thread::sleep(Duration::from_millis(20));
    for path in [&artifact_path, &control_path] {
        let mut reader = File::open(path).expect("ambient reader should open");
        let mut byte = [0_u8; 1];
        reader
            .read_exact(&mut byte)
            .expect("ambient read should advance atime");
    }
    let artifact_after = fs::metadata(&artifact_path)
        .expect("artifact metadata should reload")
        .accessed()
        .expect("advanced artifact atime should be readable");
    let control_after = fs::metadata(&control_path)
        .expect("control metadata should reload")
        .accessed()
        .expect("advanced control atime should be readable");
    assert!(
        artifact_after > artifact_before,
        "fixture must prove post-admission artifact-atime advancement"
    );
    assert!(
        control_after > control_before,
        "fixture must prove post-admission control-atime advancement"
    );

    artifact
        .sync_durable()
        .expect("APFS artifact atime alone is excluded from the mutation witness");
    control
        .try_lock_preopened(
            preopened_control(&control_path),
            GenerationRootLockMode::Exclusive,
        )
        .expect("APFS control atime alone must not fence locking")
        .unlock()
        .expect("atime-only APFS lock should release");
    assert!(
        fs::metadata(&artifact_path)
            .expect("artifact metadata should remain readable")
            .accessed()
            .expect("artifact atime should remain readable")
            >= artifact_after,
        "validation must never restore the artifact atime"
    );
    assert!(
        fs::metadata(&control_path)
            .expect("control metadata should remain readable")
            .accessed()
            .expect("control atime should remain readable")
            >= control_after,
        "validation must never restore the control atime"
    );
}

#[test]
fn physical_apfs_excludes_directory_atime_only_drift_and_never_restores_it() {
    let root_path = fixture_root("atime-exclusion");
    private_file(&root_path, "directory-entry", b"atime");
    let timestamp_owner =
        File::open(&root_path).expect("fixture directory should open for timestamp");
    let future_modified = SystemTime::now() + Duration::from_secs(24 * 60 * 60);
    timestamp_owner
        .set_times(
            std::fs::FileTimes::new()
                .set_accessed(UNIX_EPOCH)
                .set_modified(future_modified),
        )
        .expect(
            "directory atime must be old and mtime future-dated before admission so \
             each physical readdir remains relatime-eligible",
        );
    drop(timestamp_owner);

    let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
    let before = fs::metadata(&root_path)
        .expect("directory metadata should load")
        .accessed()
        .expect("directory atime should be readable");
    let observed_entry = fs::read_dir(&root_path)
        .expect("ambient directory reader should open")
        .next()
        .transpose()
        .expect("ambient directory entry should be readable");
    assert!(
        observed_entry.is_some(),
        "fixture must contain an entry so readdir performs observable work"
    );
    let after = fs::metadata(&root_path)
        .expect("directory metadata should reload")
        .accessed()
        .expect("advanced directory atime should be readable");
    assert!(
        after > before,
        "fixture must prove a post-admission directory-atime advancement"
    );
    root.sync_directory_durable()
        .expect("APFS directory atime alone is excluded from the mutation witness");
    assert_eq!(
        fs::metadata(&root_path)
            .expect("directory metadata should remain readable")
            .accessed()
            .expect("directory atime should remain readable"),
        after,
        "validation must never restore attacker-observed atime"
    );
}

#[test]
fn every_macos_specific_open_acl_and_fullsync_boundary_is_fault_injectable() {
    for target in [
        TestBoundary::BeforeExactNameEnumeration,
        TestBoundary::AfterExactNameEnumeration,
        TestBoundary::BeforeAclRead,
        TestBoundary::AfterAclRead,
    ] {
        let root_path = fixture_root("macos-boundary");
        let _guard = install_test_hook(move |boundary| {
            if boundary == target {
                return Err(injected(GenerationRootStage::InspectAcl));
            }
            Ok(())
        });
        assert_eq!(
            QualifiedGenerationRoot::admit(&root_path)
                .expect_err("injected macOS boundary must fail")
                .kind(),
            GenerationRootErrorKind::Io
        );
    }

    for target in [
        TestBoundary::BeforePreopenedQualification,
        TestBoundary::AfterPreopenedQualification,
        TestBoundary::BeforeFinalRouteStat,
        TestBoundary::AfterFinalRouteStat,
    ] {
        let root_path = fixture_root("preopened-boundary");
        let artifact_path = private_file(&root_path, "artifact", b"sealed");
        let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
        let _guard = install_test_hook(move |boundary| {
            if boundary == target {
                return Err(injected(GenerationRootStage::OpenRegularFile));
            }
            Ok(())
        });
        assert_eq!(
            root.admit_preopened_file(
                &confined("artifact"),
                preopened_immutable(&artifact_path),
                immutable_expectation(b"sealed"),
            )
            .expect_err("injected preopened qualification boundary must fail")
            .kind(),
            GenerationRootErrorKind::Io
        );
    }

    for target in [TestBoundary::BeforeFileSync, TestBoundary::AfterFileSync] {
        let root_path = fixture_root("file-fullsync-failure");
        let artifact_path = private_file(&root_path, "artifact", b"barrier");
        let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
        let admitted = root
            .admit_preopened_file(
                &confined("artifact"),
                preopened_immutable(&artifact_path),
                immutable_expectation(b"barrier"),
            )
            .expect("empty-ACL immutable APFS file should qualify");
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
            if boundary == target {
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
                .expect_err("required file F_FULLFSYNC boundary failure must propagate")
                .kind(),
            GenerationRootErrorKind::DurabilityUnavailable
        );
        let expected = if target == TestBoundary::BeforeFileSync {
            &[TestBoundary::BeforeFileSync][..]
        } else {
            &[TestBoundary::BeforeFileSync, TestBoundary::AfterFileSync][..]
        };
        assert_eq!(
            calls.lock().expect("hook calls should lock").as_slice(),
            expected,
            "file F_FULLFSYNC failure must not retry or fall back to a weaker barrier"
        );
    }

    for target in [
        TestBoundary::BeforeDirectorySync,
        TestBoundary::AfterDirectorySync,
    ] {
        let root_path = fixture_root("directory-fullsync-failure");
        let root = QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let hook_calls = Arc::clone(&calls);
        let _guard = install_test_hook(move |boundary| {
            if matches!(
                boundary,
                TestBoundary::BeforeDirectorySync | TestBoundary::AfterDirectorySync
            ) {
                hook_calls
                    .lock()
                    .expect("hook calls should lock")
                    .push(boundary);
            }
            if boundary == target {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::DurabilityUnavailable,
                    GenerationRootStage::SyncDirectory,
                ));
            }
            Ok(())
        });
        assert_eq!(
            root.sync_directory_durable()
                .expect_err("required directory F_FULLFSYNC boundary failure must propagate")
                .kind(),
            GenerationRootErrorKind::DurabilityUnavailable
        );
        let expected = if target == TestBoundary::BeforeDirectorySync {
            &[TestBoundary::BeforeDirectorySync][..]
        } else {
            &[
                TestBoundary::BeforeDirectorySync,
                TestBoundary::AfterDirectorySync,
            ][..]
        };
        assert_eq!(
            calls.lock().expect("hook calls should lock").as_slice(),
            expected,
            "directory F_FULLFSYNC failure must not retry or fall back to a weaker barrier"
        );
    }
}

#[test]
#[ignore = "physical M4 mount matrix requires explicit host paths"]
fn physical_m4_external_noowners_and_network_mounts_fail_before_mutation() {
    let external = std::env::var_os("FRANKENSEARCH_M4_EXTERNAL_NOOWNERS_ROOT")
        .expect("set FRANKENSEARCH_M4_EXTERNAL_NOOWNERS_ROOT for the physical receipt");
    let network = std::env::var_os("FRANKENSEARCH_M4_NETWORK_ROOT")
        .expect("set FRANKENSEARCH_M4_NETWORK_ROOT for the physical receipt");

    let external_error = QualifiedGenerationRoot::admit(Path::new(&external))
        .expect_err("APFS with ownership disabled must fail qualification");
    assert!(matches!(
        external_error.kind(),
        GenerationRootErrorKind::ReadOnlyFilesystem
            | GenerationRootErrorKind::UnsupportedFilesystem
    ));
    assert_eq!(
        QualifiedGenerationRoot::admit(Path::new(&network))
            .expect_err("network filesystem must fail qualification")
            .kind(),
        GenerationRootErrorKind::UnsupportedFilesystem
    );
}

#[test]
#[ignore = "physical M4 case-sensitive and case-insensitive APFS bases are explicit"]
fn physical_m4_case_sensitive_and_insensitive_apfs_both_require_exact_bytes() {
    let insensitive_base = std::env::var_os("FRANKENSEARCH_M4_CASE_INSENSITIVE_APFS_BASE")
        .expect("set the case-insensitive APFS base for the physical receipt");
    let sensitive_base = std::env::var_os("FRANKENSEARCH_M4_CASE_SENSITIVE_APFS_BASE")
        .expect("set the case-sensitive APFS base for the physical receipt");

    for (base, expected_sensitive, label) in [
        (insensitive_base, false, "case-insensitive"),
        (sensitive_base, true, "case-sensitive"),
    ] {
        let root_path = fixture_root_under(Path::new(&base), label);
        assert_eq!(
            observe_case_sensitive(&root_path),
            expected_sensitive,
            "fixture volume sensitivity must match its declared physical lane"
        );
        let artifact_path = private_file(&root_path, "CaseArtifact", b"case");
        let root = QualifiedGenerationRoot::admit(&root_path)
            .expect("local ownership-aware APFS fixture should qualify");
        root.admit_preopened_file(
            &confined("CaseArtifact"),
            preopened_immutable(&artifact_path),
            immutable_expectation(b"case"),
        )
        .expect("the exact-case route must qualify on each APFS personality");
        assert_eq!(
            root.admit_preopened_file(
                &confined("caseartifact"),
                preopened_immutable(&artifact_path),
                immutable_expectation(b"case"),
            )
            .expect_err("byte-inexact spelling must fail on every APFS personality")
            .kind(),
            GenerationRootErrorKind::ObjectChanged
        );
    }
}
