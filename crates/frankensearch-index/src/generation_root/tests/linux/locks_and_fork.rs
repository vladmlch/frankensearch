use super::super::super::platform::{
    TestBoundary, duplicate_lock_descriptor_for_inheritance_for_test, install_test_hook,
    linux_device_components_match_for_test, linux_filesystem_profile,
    linux_mountinfo_error_for_test, linux_mountinfo_filesystem_for_test, linux_open_error_for_test,
    linux_task_mountinfo_route_for_test, parse_mountinfo_filesystem, proc_bridge_error_for_test,
    proc_root_error_for_test, proc_thread_fd_route_for_test, set_lock_creator_process_id_for_test,
    set_root_creator_process_id_for_test, validate_bound_control_content_for_test,
    validate_linux_writable_attributes_for_test, validate_proc_mount_id_mask_for_test,
};
use super::super::super::{
    GenerationFileExpectation, GenerationRootErrorKind, GenerationRootLockMode,
    GenerationRootStage, QualifiedGenerationDirectory as QualifiedGenerationRoot,
};
use super::*;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

#[test]
fn lock_modes_contend_release_sync_and_validate_optional_digest() {
    let root_path = fixture_root("locks");
    let contents = b"lock-v1!";
    control_file(&root_path, "LOCK", contents);
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let route = confined("LOCK");
    let expectation = GenerationFileExpectation::control(8)
        .expect("small control-file expectation should be accepted");
    let control = root
        .admit_control_file(&route, expectation)
        .expect("control file should be admitted once");
    assert_eq!(control.bytes().as_ref(), contents);
    assert_eq!(control.sha256(), digest(contents));

    let shared_one = control
        .try_lock(GenerationRootLockMode::Shared)
        .expect("first shared lock should succeed");
    let shared_two = control
        .try_lock(GenerationRootLockMode::Shared)
        .expect("second shared lock should succeed");
    assert_eq!(
        control
            .try_lock(GenerationRootLockMode::Exclusive)
            .expect_err("exclusive lock must contend with shared locks")
            .kind(),
        GenerationRootErrorKind::LockContended
    );
    shared_one.unlock().expect("shared unlock should succeed");
    shared_two.unlock().expect("shared unlock should succeed");

    let exclusive = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("exclusive lock should succeed after release");
    assert_eq!(exclusive.witness(), control.witness());
    assert_eq!(exclusive.bytes().as_ref(), contents);
    assert_eq!(exclusive.sha256(), control.sha256());
    assert_eq!(exclusive.mode(), GenerationRootLockMode::Exclusive);
    assert_eq!(
        control
            .try_lock(GenerationRootLockMode::Shared)
            .expect_err("shared lock must contend with exclusive lock")
            .kind(),
        GenerationRootErrorKind::LockContended
    );
    exclusive
        .sync_durable()
        .expect("locked control-file fsync should succeed");
    exclusive.unlock().expect("exclusive unlock should succeed");

    let exact_control = root
        .admit_control_file(
            &route,
            GenerationFileExpectation::control_with_sha256(8, digest(contents))
                .expect("small expectation should be accepted"),
        )
        .expect("digest-qualified control capability should be admitted");
    let exact = exact_control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("digest-qualified lock should succeed");
    exact
        .unlock()
        .expect("digest-qualified unlock should succeed");
    let wrong = GenerationFileExpectation::control_with_sha256(8, [0xa5; 32])
        .expect("small expectation should be accepted");
    assert_eq!(
        root.admit_control_file(&route, wrong)
            .expect("wrong digest is checked only while acquiring the lock")
            .try_lock(GenerationRootLockMode::Exclusive)
            .expect_err("wrong lock-file digest must fail after acquisition")
            .kind(),
        GenerationRootErrorKind::HashMismatch
    );
}

#[test]
fn immutable_content_drift_fences_sync_before_durability() {
    let root_path = fixture_root("mapped-immutable-before-sync");
    let (artifact_path, mapping) = mapped_immutable_file(&root_path, "artifact");
    let admitted_bytes =
        fs::read(&artifact_path).expect("mapped immutable fixture must be readable");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(
            &confined("artifact"),
            immutable_expectation(&admitted_bytes),
        )
        .expect("mapped immutable fixture should qualify before ambient mutation");

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
                    .expect("shared immutable mapping should remain retained by the hook"),
            );
            mutated = true;
        }
        Ok(())
    });
    let error = admitted
        .sync_durable()
        .expect_err("pre-durability immutable content drift must fail closed");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
    assert!(
        !boundaries
            .lock()
            .expect("boundary log should lock")
            .contains(&TestBoundary::BeforeFileSync),
        "immutable content validation must fail before the durability syscall"
    );
}

#[test]
fn immutable_content_drift_during_durability_is_detected_after_barrier() {
    let root_path = fixture_root("mapped-immutable-during-sync");
    let (artifact_path, mapping) = mapped_immutable_file(&root_path, "artifact");
    let admitted_bytes =
        fs::read(&artifact_path).expect("mapped immutable fixture must be readable");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let admitted = root
        .admit_file(
            &confined("artifact"),
            immutable_expectation(&admitted_bytes),
        )
        .expect("mapped immutable fixture should qualify before ambient mutation");

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
                    .expect("shared immutable mapping should remain retained by the hook"),
            );
            mutated = true;
        }
        Ok(())
    });
    let error = admitted
        .sync_durable()
        .expect_err("immutable mutation at durability must fail after the barrier");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
    let observed = boundaries.lock().expect("boundary log should lock");
    let after_sync = observed
        .iter()
        .position(|boundary| *boundary == TestBoundary::AfterFileSync)
        .expect("the durability barrier should complete before post-check failure");
    assert!(
        observed.iter().enumerate().any(|(index, boundary)| {
            index > after_sync && *boundary == (TestBoundary::BeforeRead { offset: 0 })
        }),
        "a second exact immutable content read must follow the durability barrier"
    );
    drop(observed);
}

#[test]
fn control_writer_mapping_is_lock_contended() {
    let root_path = fixture_root("mapped-control-after-flock");
    let (lock_path, _mapping) = mapped_control_file(&root_path, "LOCK");
    let byte_len = fs::metadata(&lock_path)
        .expect("mapped control fixture metadata should load")
        .len();
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let expectation = GenerationFileExpectation::control(byte_len)
        .expect("mapped control fixture length should be accepted");
    let control = root
        .admit_control_file(&confined("LOCK"), expectation)
        .expect("mapped control fixture should qualify before lock acquisition");
    let error = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect_err("retained FSVI writer mapping must block a second exclusive lock");
    assert_eq!(error.kind(), GenerationRootErrorKind::LockContended);
    assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
}

#[test]
fn control_content_drift_fences_sync_before_durability() {
    let root_path = fixture_root("mapped-control-before-sync");
    let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
    let byte_len = fs::metadata(&lock_path)
        .expect("mapped control fixture metadata should load")
        .len();
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let error = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(byte_len)
                .expect("mapped control fixture length should be accepted"),
        )
        .expect("mapped control fixture should qualify")
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect_err("retained FSVI writer mapping must block the durability lock");
    assert_eq!(error.kind(), GenerationRootErrorKind::LockContended);
    assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
    drop(mapping);
}

#[test]
fn control_content_drift_during_durability_is_detected_after_barrier() {
    let root_path = fixture_root("mapped-control-during-sync");
    let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
    let byte_len = fs::metadata(&lock_path)
        .expect("mapped control fixture metadata should load")
        .len();
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let error = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(byte_len)
                .expect("mapped control fixture length should be accepted"),
        )
        .expect("mapped control fixture should qualify")
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect_err("retained FSVI writer mapping must block the durability lock");
    assert_eq!(error.kind(), GenerationRootErrorKind::LockContended);
    assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
    drop(mapping);
}

#[test]
fn bound_content_mismatch_is_object_changed_even_when_metadata_is_not_consulted() {
    let error = validate_bound_control_content_for_test(
        [0x11; 32],
        [0x22; 32],
        GenerationRootStage::AcquireLock,
    )
    .expect_err("different content witnesses must fail independently of metadata");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
    validate_bound_control_content_for_test(
        [0x33; 32],
        [0x33; 32],
        GenerationRootStage::SyncRegularFile,
    )
    .expect("identical content witnesses should validate");
}

#[test]
fn inherited_lock_guard_never_issues_unlock_from_a_different_process_identity() {
    let root_path = fixture_root("forked-lock");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let control = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect("control file should qualify");
    let mut inherited = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("exclusive lock should succeed");
    set_lock_creator_process_id_for_test(&mut inherited.inner, std::process::id().wrapping_add(1));
    let unlock_boundary_reached = Arc::new(Mutex::new(false));
    let hook_boundary_reached = Arc::clone(&unlock_boundary_reached);
    let guard = install_test_hook(move |boundary| {
        if matches!(
            boundary,
            TestBoundary::BeforeUnlock | TestBoundary::AfterUnlock
        ) {
            *hook_boundary_reached
                .lock()
                .expect("hook state should lock") = true;
        }
        Ok(())
    });
    assert_eq!(
        inherited
            .unlock()
            .expect_err("simulated fork child must refuse explicit unlock")
            .kind(),
        GenerationRootErrorKind::ForkedProcess
    );
    assert!(
        !*unlock_boundary_reached
            .lock()
            .expect("hook state should lock"),
        "process identity must be checked before the explicit LOCK_UN path"
    );
    drop(guard);

    control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("descriptor close should leave the control file lockable")
        .unlock()
        .expect("fresh same-process lock should unlock");
}

#[test]
fn inherited_root_capability_fails_before_proc_bridge_and_regular_file_io() {
    let root_path = fixture_root("forked-root");
    private_file(&root_path, "artifact", b"fork");
    let mut root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    set_root_creator_process_id_for_test(&mut root.inner, std::process::id().wrapping_add(1));
    let boundary_reached = Arc::new(Mutex::new(false));
    let hook_boundary_reached = Arc::clone(&boundary_reached);
    let _guard = install_test_hook(move |_| {
        *hook_boundary_reached
            .lock()
            .expect("hook state should lock") = true;
        Ok(())
    });

    let file_error = root
        .admit_file(&confined("artifact"), immutable_expectation(b"fork"))
        .expect_err("simulated fork child must reject before the proc-fd bridge");
    assert_eq!(file_error.kind(), GenerationRootErrorKind::ForkedProcess);
    assert_eq!(file_error.stage(), GenerationRootStage::RevalidateRootRoute);
    let route_error = root
        .revalidate_route()
        .expect_err("simulated fork child must reject the retained root");
    assert_eq!(route_error.kind(), GenerationRootErrorKind::ForkedProcess);
    assert_eq!(
        route_error.stage(),
        GenerationRootStage::RevalidateRootRoute
    );
    assert!(
        !*boundary_reached.lock().expect("hook state should lock"),
        "process identity must be checked before proc-root qualification, proc-fd \
         resolution, reads, flock, sync, or route work"
    );
}

#[test]
#[ignore = "requires native Linux, python3, fork, and procfs"]
fn physical_linux_proc_bridge_fork_receipt_separates_child_self_from_parent_fd_number() {
    let root_path = fixture_root("physical-fork-proc-bridge");
    let inherited_path = private_file(&root_path, "inherited", b"inherited");
    let parent_decoy_path = private_file(&root_path, "parent-decoy", b"parent-decoy");
    let script = r#"
import json
import os
import sys

inherited_path = os.path.realpath(sys.argv[1])
parent_decoy_path = os.path.realpath(sys.argv[2])
parent_pid = os.getpid()
probe = os.open(inherited_path, os.O_PATH | os.O_CLOEXEC)
probe_number = probe
ready_read, ready_write = os.pipe()
continue_read, continue_write = os.pipe()
child = os.fork()
if child == 0:
    os.close(ready_read)
    os.close(continue_write)
    os.write(ready_write, b"r")
    os.close(ready_write)
    if os.read(continue_read, 1) != b"c":
        os._exit(3)
    os.close(continue_read)
    child_self_route = os.readlink(f"/proc/self/fd/{probe_number}")
    parent_same_number_route = os.readlink(
        f"/proc/{parent_pid}/fd/{probe_number}"
    )
    child_ok = (
        child_self_route == inherited_path
        and parent_same_number_route == parent_decoy_path
    )
    print(json.dumps({
        "role": "child",
        "probe_number": probe_number,
        "self_route": child_self_route,
        "parent_same_number_route": parent_same_number_route,
        "self_is_inherited": child_self_route == inherited_path,
        "parent_same_number_is_decoy": (
            parent_same_number_route == parent_decoy_path
        ),
    }), flush=True)
    os._exit(0 if child_ok else 4)

os.close(ready_write)
os.close(continue_read)
if os.read(ready_read, 1) != b"r":
    raise SystemExit(5)
os.close(ready_read)
os.close(probe)
decoy = os.open(parent_decoy_path, os.O_PATH | os.O_CLOEXEC)
if decoy != probe_number:
    os.dup2(decoy, probe_number, inheritable=False)
    os.close(decoy)
parent_self_route = os.readlink(f"/proc/self/fd/{probe_number}")
os.write(continue_write, b"c")
os.close(continue_write)
_, status = os.waitpid(child, 0)
child_exit = os.waitstatus_to_exitcode(status)
print(json.dumps({
    "role": "parent",
    "probe_number": probe_number,
    "self_route": parent_self_route,
    "self_is_decoy": parent_self_route == parent_decoy_path,
    "child_exit": child_exit,
}), flush=True)
if parent_self_route != parent_decoy_path or child_exit != 0:
    raise SystemExit(6)
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&inherited_path)
        .arg(&parent_decoy_path)
        .output()
        .expect("native proc-bridge fork receipt requires python3");
    assert!(
        output.status.success(),
        "proc-bridge fork receipt failed: status={:?}, stdout={}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("fork receipt JSON must be UTF-8");
    let mut records = stdout.lines().map(|line| {
        serde_json::from_str::<serde_json::Value>(line)
            .expect("each fork receipt line must be JSON")
    });
    let child = records.next().expect("child receipt must be present");
    let parent = records.next().expect("parent receipt must be present");
    assert!(
        records.next().is_none(),
        "fork receipt must contain exactly child and parent records"
    );
    assert_eq!(child["role"], "child");
    assert_eq!(child["self_is_inherited"], true);
    assert_eq!(child["parent_same_number_is_decoy"], true);
    assert_eq!(parent["role"], "parent");
    assert_eq!(parent["self_is_decoy"], true);
    assert_eq!(parent["child_exit"], 0);
    assert_eq!(
        child["probe_number"], parent["probe_number"],
        "the receipt must compare the exact same numeric descriptor in both processes"
    );
    assert_ne!(
        child["self_route"], child["parent_same_number_route"],
        "the child's self route must remain bound to its inherited probe while the \
         parent's same-number route names the decoy"
    );
}

#[test]
#[ignore = "requires native Linux, python3, and fork/flock process support"]
fn physical_linux_fork_receipt_proves_close_vs_explicit_unlock_with_inherited_ofd() {
    let root_path = fixture_root("physical-fork-ofd");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let script = r#"
import fcntl
import json
import os
import sys

path = sys.argv[1]

def open_lock():
    return os.open(path, os.O_RDWR | os.O_CLOEXEC)

held = open_lock()
fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
read_end, write_end = os.pipe()
child = os.fork()
if child == 0:
    os.close(write_end)
    os.read(read_end, 1)
    os.close(held)
    os._exit(0)
os.close(read_end)
os.close(held)
probe = open_lock()
close_only_contended = False
try:
    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    close_only_contended = True
os.write(write_end, b"x")
os.close(write_end)
os.waitpid(child, 0)
fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
fcntl.flock(probe, fcntl.LOCK_UN)
os.close(probe)

held = open_lock()
fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
read_end, write_end = os.pipe()
child = os.fork()
if child == 0:
    os.close(write_end)
    os.read(read_end, 1)
    os.close(held)
    os._exit(0)
os.close(read_end)
fcntl.flock(held, fcntl.LOCK_UN)
os.close(held)
probe = open_lock()
fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
explicit_unlock_released = True
fcntl.flock(probe, fcntl.LOCK_UN)
os.close(probe)
os.write(write_end, b"x")
os.close(write_end)
os.waitpid(child, 0)

print(json.dumps({
    "close_only_contended_while_child_retained_ofd": close_only_contended,
    "explicit_unlock_released_while_child_retained_ofd": explicit_unlock_released,
}))
if not close_only_contended or not explicit_unlock_released:
    raise SystemExit(2)
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&lock_path)
        .output()
        .expect("native fork receipt requires python3");
    assert!(
        output.status.success(),
        "native fork/OFD receipt failed: status={:?}, stdout={}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout =
        String::from_utf8(output.stdout).expect("native fork/OFD receipt must emit UTF-8 JSON");
    assert!(stdout.contains("\"close_only_contended_while_child_retained_ofd\": true"));
    assert!(stdout.contains("\"explicit_unlock_released_while_child_retained_ofd\": true"));
}

#[test]
#[ignore = "requires FRANKENSEARCH_LINUX_LOCK_FORK_TEST_ROOT on native ext4/Btrfs plus python3"]
fn physical_generation_lock_drop_and_unlock_release_while_child_retains_same_ofd() {
    struct FdHolder {
        child: Option<std::process::Child>,
    }

    impl FdHolder {
        fn assert_alive(&mut self) {
            assert!(
                self.child
                    .as_mut()
                    .expect("fd-holder child must be present")
                    .try_wait()
                    .expect("fd-holder liveness should be observable")
                    .is_none(),
                "child must still retain the guard OFD"
            );
        }

        fn release(mut self) {
            let mut child = self.child.take().expect("fd-holder child must be present");
            child
                .stdin
                .take()
                .expect("fd-holder stdin must be piped")
                .write_all(b"x")
                .expect("fd-holder release byte must be writable");
            let output = child
                .wait_with_output()
                .expect("fd-holder process must be waitable");
            assert!(
                output.status.success(),
                "fd-holder failed: status={:?}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    impl Drop for FdHolder {
        fn drop(&mut self) {
            let Some(mut child) = self.child.take() else {
                return;
            };
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(b"x");
            }
            match child.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }
    }

    fn spawn_inherited_fd_holder(alias: &rustix::fd::OwnedFd) -> FdHolder {
        let mut child = Command::new("python3")
            .arg("-c")
            .arg(
                "import os,sys; fd=int(sys.argv[1]); os.fstat(fd); \
                 print('READY', flush=True); sys.stdin.buffer.read(1)",
            )
            .arg(alias.as_raw_fd().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("native inherited-OFD receipt requires python3");
        let stdout = child.stdout.take().expect("fd-holder stdout must be piped");
        let mut ready = String::new();
        BufReader::new(stdout)
            .read_line(&mut ready)
            .expect("fd-holder readiness must be readable");
        assert_eq!(ready.trim(), "READY", "child must retain the exact alias");
        FdHolder { child: Some(child) }
    }

    let base = std::env::var_os("FRANKENSEARCH_LINUX_LOCK_FORK_TEST_ROOT")
        .map(PathBuf::from)
        .expect("set FRANKENSEARCH_LINUX_LOCK_FORK_TEST_ROOT to native ext4/Btrfs");
    let filesystem =
        rustix::fs::statfs(&base).expect("physical lock fixture base must be statable");
    assert!(
        i128::from(filesystem.f_type) == i128::from(libc::EXT4_SUPER_MAGIC)
            || i128::from(filesystem.f_type) == i128::from(libc::BTRFS_SUPER_MAGIC),
        "physical inherited-OFD receipt requires ext4 or Btrfs"
    );
    let root_path = fixture_root_under(&base, "physical-generation-lock-fork");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let control = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8)
                .expect("small control expectation should be accepted"),
        )
        .expect("control file should qualify");

    let dropped = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("Rust guard should acquire for Drop receipt");
    let drop_alias = duplicate_lock_descriptor_for_inheritance_for_test(&dropped.inner)
        .expect("test should duplicate the private guard descriptor");
    let mut drop_child = spawn_inherited_fd_holder(&drop_alias);
    drop(drop_alias);
    drop_child.assert_alive();
    drop(dropped);
    control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("creator Drop must LOCK_UN while the child alias survives")
        .unlock()
        .expect("fresh lock after creator Drop should release");
    drop_child.release();

    let explicit = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("Rust guard should acquire for explicit-unlock receipt");
    let explicit_alias = duplicate_lock_descriptor_for_inheritance_for_test(&explicit.inner)
        .expect("test should duplicate the explicit guard descriptor");
    let mut explicit_child = spawn_inherited_fd_holder(&explicit_alias);
    drop(explicit_alias);
    explicit_child.assert_alive();
    explicit
        .unlock()
        .expect("creator explicit LOCK_UN should succeed");
    control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect("explicit LOCK_UN must release while the child alias survives")
        .unlock()
        .expect("fresh lock after explicit unlock should release");
    explicit_child.release();
}

#[test]
fn control_route_replacement_fences_lock_acquisition_and_locked_sync() {
    let root_path = fixture_root("control-route-replacement-before-lock");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let control = root
        .admit_control_file(
            &confined("LOCK"),
            GenerationFileExpectation::control(8).expect("small expectation should be accepted"),
        )
        .expect("control file should qualify");
    fs::rename(&lock_path, root_path.join("retained-LOCK"))
        .expect("original control file should move to a fresh retained route");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let before_lock_reached = Arc::new(Mutex::new(false));
    let hook_before_lock_reached = Arc::clone(&before_lock_reached);
    let guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeLock {
            *hook_before_lock_reached
                .lock()
                .expect("hook state should lock") = true;
        }
        Ok(())
    });
    let lock_error = control
        .try_lock(GenerationRootLockMode::Exclusive)
        .expect_err("replacement route must fence lock acquisition");
    assert_eq!(lock_error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(
        lock_error.stage(),
        GenerationRootStage::RevalidateRegularFile
    );
    assert!(
        !*before_lock_reached.lock().expect("hook state should lock"),
        "route identity must be checked before flock"
    );
    drop(guard);

    let root_path = fixture_root("control-route-replacement-after-lock");
    let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
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
    let retained_lock_path = root_path.join("retained-LOCK");
    fs::rename(&lock_path, &retained_lock_path)
        .expect("locked file should move to a fresh retained route");
    control_file(&root_path, "LOCK", b"lock-v1!");
    let sync_boundary_reached = Arc::new(Mutex::new(false));
    let hook_sync_boundary_reached = Arc::clone(&sync_boundary_reached);
    let unlock_boundary_reached = Arc::new(Mutex::new(false));
    let hook_unlock_boundary_reached = Arc::clone(&unlock_boundary_reached);
    let _guard = install_test_hook(move |boundary| {
        if boundary == TestBoundary::BeforeFileSync {
            *hook_sync_boundary_reached
                .lock()
                .expect("hook state should lock") = true;
        }
        if boundary == TestBoundary::BeforeUnlock {
            *hook_unlock_boundary_reached
                .lock()
                .expect("hook state should lock") = true;
        }
        Ok(())
    });
    let sync_error = lock
        .sync_durable()
        .expect_err("replacement route must fence locked sync");
    assert_eq!(sync_error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(
        sync_error.stage(),
        GenerationRootStage::RevalidateRegularFile
    );
    assert!(
        !*sync_boundary_reached
            .lock()
            .expect("hook state should lock"),
        "route identity must be checked before the durability syscall"
    );
    let unlock_error = lock
        .unlock()
        .expect_err("route replacement remains visible during explicit unlock");
    assert_eq!(unlock_error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(
        unlock_error.stage(),
        GenerationRootStage::RevalidateRegularFile
    );
    assert!(
        *unlock_boundary_reached
            .lock()
            .expect("hook state should lock"),
        "route-validation failure must not skip the explicit LOCK_UN path"
    );
    let retained_lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&retained_lock_path)
        .expect("retained lock inode should reopen after failed validation");
    rustix::fs::flock(
        &retained_lock,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    )
    .expect("failed validation must still release the retained inode lock");
    rustix::fs::flock(&retained_lock, rustix::fs::FlockOperation::Unlock)
        .expect("test lock cleanup should succeed");
}

#[test]
fn unsupported_linux_filesystems_fail_before_admission() {
    let tmpfs_base = Path::new("/dev/shm");
    let tmpfs_stat =
        rustix::fs::statfs(tmpfs_base).expect("physical tmpfs fixture must be statable");
    assert_eq!(
        i128::from(tmpfs_stat.f_type),
        i128::from(libc::TMPFS_MAGIC),
        "unsupported-filesystem receipt requires a real tmpfs mount"
    );
    assert_eq!(
        QualifiedGenerationRoot::admit(tmpfs_base)
            .expect_err("the tmpfs mount itself must fail the physical allowlist")
            .kind(),
        GenerationRootErrorKind::UnsupportedFilesystem
    );
    assert_eq!(
        QualifiedGenerationRoot::admit(Path::new("/proc"))
            .expect_err("procfs root must fail the physical filesystem allowlist")
            .kind(),
        GenerationRootErrorKind::UnsupportedFilesystem
    );
}

#[test]
fn xfs_remains_fail_closed_until_its_native_receipt_bead_closes() {
    let raw_type: rustix::fs::FsWord = libc::XFS_SUPER_MAGIC;
    assert_eq!(
        linux_filesystem_profile(raw_type, 1)
            .expect_err("an unreceipted XFS profile must not reach mountinfo admission")
            .kind(),
        GenerationRootErrorKind::UnsupportedFilesystem
    );
}

#[test]
fn statx_writable_truth_requires_and_rejects_immutable_and_append_attributes() {
    use rustix::fs::StatxAttributes;

    let required = StatxAttributes::IMMUTABLE | StatxAttributes::APPEND;
    validate_linux_writable_attributes_for_test(StatxAttributes::empty(), required)
        .expect("a reported writable object should pass");

    let missing_mask = validate_linux_writable_attributes_for_test(
        StatxAttributes::empty(),
        StatxAttributes::IMMUTABLE,
    )
    .expect_err("an unreported append attribute must fail closed");
    assert_eq!(
        missing_mask.kind(),
        GenerationRootErrorKind::UnsupportedKernelFeature
    );
    assert_eq!(missing_mask.stage(), GenerationRootStage::QualifyFilesystem);

    for attribute in [StatxAttributes::IMMUTABLE, StatxAttributes::APPEND] {
        let error = validate_linux_writable_attributes_for_test(attribute, required)
            .expect_err("immutable or append-only objects are not writable truth");
        assert_eq!(error.kind(), GenerationRootErrorKind::WriteRestrictedObject);
        assert_eq!(error.stage(), GenerationRootStage::QualifyFilesystem);
    }
}

#[test]
fn stat_and_statx_device_components_must_match_exactly() {
    assert!(linux_device_components_match_for_test(8, 1, 8, 1));
    assert!(!linux_device_components_match_for_test(8, 1, 9, 1));
    assert!(!linux_device_components_match_for_test(8, 1, 8, 2));
}

#[test]
fn proc_bridge_errno_classes_distinguish_limits_policy_and_identity_drift() {
    for error in [
        rustix::io::Errno::MFILE,
        rustix::io::Errno::NFILE,
        rustix::io::Errno::NOMEM,
    ] {
        assert_eq!(
            proc_bridge_error_for_test(error).kind(),
            GenerationRootErrorKind::ResourceLimit
        );
        assert_eq!(
            linux_open_error_for_test(error).kind(),
            GenerationRootErrorKind::ResourceLimit
        );
        let mountinfo_error = linux_mountinfo_error_for_test(error);
        assert_eq!(
            mountinfo_error.kind(),
            GenerationRootErrorKind::ResourceLimit
        );
        assert_eq!(mountinfo_error.raw_os_error(), Some(error.raw_os_error()));
    }
    let mountinfo_io = linux_mountinfo_error_for_test(rustix::io::Errno::IO);
    assert_eq!(mountinfo_io.kind(), GenerationRootErrorKind::Io);
    assert_eq!(
        mountinfo_io.raw_os_error(),
        Some(rustix::io::Errno::IO.raw_os_error())
    );
    for error in [rustix::io::Errno::PERM, rustix::io::Errno::ACCESS] {
        assert_eq!(
            proc_bridge_error_for_test(error).kind(),
            GenerationRootErrorKind::UnsupportedKernelFeature
        );
    }
    for error in [
        rustix::io::Errno::NOENT,
        rustix::io::Errno::NOTDIR,
        rustix::io::Errno::LOOP,
        rustix::io::Errno::XDEV,
        rustix::io::Errno::STALE,
    ] {
        assert_eq!(
            proc_bridge_error_for_test(error).kind(),
            GenerationRootErrorKind::ObjectChanged
        );
    }
    for error in [
        rustix::io::Errno::NOENT,
        rustix::io::Errno::NOTDIR,
        rustix::io::Errno::LOOP,
        rustix::io::Errno::XDEV,
    ] {
        assert_eq!(
            proc_root_error_for_test(error).kind(),
            GenerationRootErrorKind::UnsupportedKernelFeature
        );
    }
}

#[test]
fn proc_bridge_requires_kernel_attestation_of_the_mount_id_field() {
    use rustix::fs::StatxFlags;

    assert_eq!(
        validate_proc_mount_id_mask_for_test(StatxFlags::MNT_ID, 47)
            .expect("reported mount identity should pass"),
        47
    );
    let error = validate_proc_mount_id_mask_for_test(StatxFlags::BASIC_STATS, 47)
        .expect_err("an unreported mount-id value must never be trusted");
    assert_eq!(
        error.kind(),
        GenerationRootErrorKind::UnsupportedKernelFeature
    );
    assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
    assert_eq!(error.expected(), Some(u64::from(StatxFlags::MNT_ID.bits())));
    assert_eq!(error.observed(), Some(0));
}

#[test]
fn proc_thread_self_binding_and_bridge_route_name_the_current_thread_table() {
    let process_id = std::process::id();
    let thread_id = rustix::thread::gettid();
    assert_eq!(
        fs::read_link("/proc/thread-self")
            .expect("the kernel thread-self binding must be readable"),
        PathBuf::from(format!("{process_id}/task/{thread_id}"))
    );
    assert_eq!(
        proc_thread_fd_route_for_test(47),
        "thread-self/fd/47",
        "the production bridge must never route through the leader's self/fd table"
    );
    assert_eq!(
        linux_task_mountinfo_route_for_test(process_id, thread_id.as_raw_pid()),
        PathBuf::from(format!("{process_id}/task/{thread_id}/mountinfo")),
        "the verified proc-root-relative route must name the accessing thread namespace"
    );
}

#[test]
#[ignore = "physical Linux CLONE_FILES regression must run as an exact isolated test"]
#[allow(deprecated)]
fn physical_linux_thread_self_bridge_survives_an_unshared_fd_table_collision() {
    use rustix::thread::UnshareFlags;
    use std::sync::mpsc;

    let root_path = fixture_root("thread-self-unshared-files");
    let artifact_path = private_file(&root_path, "artifact", b"thread-local capability");
    let (descriptor_sender, descriptor_receiver) = mpsc::sync_channel(0);
    let (continue_sender, continue_receiver) = mpsc::sync_channel(0);
    let worker = std::thread::spawn(move || {
        // rustix 1.1.4 retains this deprecated safe wrapper. This
        // worker owns every descriptor it opens after unsharing and
        // terminates immediately after the receipt, satisfying the
        // API's no-cross-table-descriptor-use invariant.
        rustix::thread::unshare(UnshareFlags::FILES)
            .expect("the worker file table must unshare without privilege");
        let retained = File::open(&artifact_path).expect("the worker-local artifact must open");
        let raw_fd = retained.as_raw_fd();
        descriptor_sender
            .send(raw_fd)
            .expect("the leader must receive the collision descriptor number");
        continue_receiver
            .recv()
            .expect("the leader must establish its decoy at the same fd number");

        assert_eq!(
            fs::read_link(format!("/proc/self/fd/{raw_fd}"))
                .expect("the leader-table proc route must resolve"),
            PathBuf::from("/dev/null"),
            "self/fd must demonstrate the unrelated leader-table collision"
        );
        assert_eq!(
            fs::read_link(format!("/proc/thread-self/fd/{raw_fd}"))
                .expect("the worker-table proc route must resolve"),
            artifact_path,
            "thread-self/fd must remain bound to the worker-local capability"
        );

        let root = QualifiedGenerationRoot::admit(&root_path)
            .expect("the private root must qualify after CLONE_FILES");
        let admitted = root
            .admit_file(
                &confined("artifact"),
                immutable_expectation(b"thread-local capability"),
            )
            .expect("production admission must reopen through the worker descriptor table");
        assert_eq!(
            admitted.bytes().as_ref(),
            b"thread-local capability",
            "the admitted bytes must come from the worker-local retained capability"
        );
    });

    let worker_fd = descriptor_receiver
        .recv()
        .expect("the worker must report its retained fd number");
    let leader_decoy = File::open("/dev/null").expect("the leader decoy must open");
    assert_eq!(
        leader_decoy.as_raw_fd(),
        worker_fd,
        "the isolated receipt requires an exact leader/worker fd-number collision"
    );
    continue_sender
        .send(())
        .expect("the worker must be released after collision setup");
    worker
        .join()
        .expect("the unshared-table worker must complete without panic");
}

#[test]
#[ignore = "physical Linux CLONE_NEWNS receipt requires root and exact isolated execution"]
#[allow(deprecated)]
fn physical_linux_task_mountinfo_observes_the_calling_threads_private_mount_namespace() {
    use rustix::fs::{AtFlags, CWD, StatxFlags, statx};
    use rustix::thread::UnshareFlags;

    assert!(
        rustix::process::geteuid().is_root(),
        "the task-scoped mountinfo receipt requires CAP_SYS_ADMIN"
    );
    let mountpoint = private_dir(
        &fixture_root("task-mountinfo-private-namespace"),
        "bind-mount",
    );
    let worker = std::thread::spawn(move || {
        // NEWNS does not split the descriptor table. This isolated
        // worker performs all namespace-sensitive work itself and
        // exits immediately, so the private mounts disappear with
        // the last task holding the namespace.
        rustix::thread::unshare(UnshareFlags::NEWNS)
            .expect("the worker must acquire a private mount namespace");
        let private_output = Command::new("mount")
            .args(["--make-rprivate", "/"])
            .output()
            .expect("the physical receipt requires util-linux mount");
        assert!(
            private_output.status.success(),
            "making the worker namespace private failed: status={:?}, stdout={}, stderr={}",
            private_output.status.code(),
            String::from_utf8_lossy(&private_output.stdout),
            String::from_utf8_lossy(&private_output.stderr)
        );
        let bind_output = Command::new("mount")
            .arg("--bind")
            .arg(&mountpoint)
            .arg(&mountpoint)
            .output()
            .expect("the physical receipt requires util-linux mount");
        assert!(
            bind_output.status.success(),
            "worker-private bind mount failed: status={:?}, stdout={}, stderr={}",
            bind_output.status.code(),
            String::from_utf8_lossy(&bind_output.stdout),
            String::from_utf8_lossy(&bind_output.stderr)
        );

        let extended = statx(
            CWD,
            &mountpoint,
            AtFlags::NO_AUTOMOUNT,
            StatxFlags::BASIC_STATS | StatxFlags::MNT_ID,
        )
        .expect("the worker-private bind mount must expose statx identity");
        assert!(
            StatxFlags::from_bits_retain(extended.stx_mask).contains(StatxFlags::MNT_ID),
            "the kernel must attest the worker-private mount id"
        );
        let filesystem = rustix::fs::statfs(&mountpoint)
            .expect("the worker-private bind mount must be statable");
        let expected = if i128::from(filesystem.f_type) == i128::from(libc::EXT4_SUPER_MAGIC) {
            b"ext4".as_slice()
        } else if i128::from(filesystem.f_type) == i128::from(libc::BTRFS_SUPER_MAGIC) {
            b"btrfs".as_slice()
        } else {
            panic!("the physical receipt requires ext4 or Btrfs");
        };

        let leader_bytes =
            fs::read("/proc/self/mountinfo").expect("the leader mount table must remain readable");
        assert!(
            parse_mountinfo_filesystem(&leader_bytes, extended.stx_mnt_id).is_err(),
            "the leader table must not contain the worker-private bind mount"
        );
        let task_path = linux_task_mountinfo_route_for_test(
            std::process::id(),
            rustix::thread::gettid().as_raw_pid(),
        );
        let task_bytes = fs::read(Path::new("/proc").join(&task_path))
            .expect("the worker task mount table must be readable");
        assert_eq!(
            parse_mountinfo_filesystem(&task_bytes, extended.stx_mnt_id)
                .expect("the worker task table must contain its private bind mount"),
            expected
        );
        assert_eq!(
            linux_mountinfo_filesystem_for_test(extended.stx_mnt_id)
                .expect("production mountinfo lookup must use the worker task table"),
            expected,
            "the production route must attest the accessing thread's mount namespace"
        );
    });
    worker
        .join()
        .expect("the private-mount-namespace worker must complete without panic");
}
