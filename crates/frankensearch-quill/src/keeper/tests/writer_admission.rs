use super::*;

#[cfg(any(unix, windows))]
struct WriterChild {
    child: Option<std::process::Child>,
}

#[cfg(any(unix, windows))]
impl WriterChild {
    fn child_mut(&mut self) -> &mut std::process::Child {
        self.child.as_mut().expect("child is still owned")
    }

    fn wait_success(mut self) -> TestResult {
        let status = self.child.take().expect("child is still owned").wait()?;
        if !status.success() {
            return Err(format!("writer child exited with {status}").into());
        }
        Ok(())
    }

    #[cfg(unix)]
    fn kill_and_reap(mut self) -> TestResult {
        let mut child = self.child.take().expect("child is still owned");
        let kill_result = child.kill();
        let wait_result = child.wait();
        kill_result?;
        let status = wait_result?;
        if status.success() {
            return Err("killed writer child unexpectedly exited successfully".into());
        }
        Ok(())
    }
}

#[cfg(any(unix, windows))]
impl Drop for WriterChild {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(any(unix, windows))]
fn spawn_writer_child(
    role: &str,
    directory: &Path,
    control: &Path,
    identifier: &str,
) -> Result<WriterChild, Box<dyn std::error::Error>> {
    let child = std::process::Command::new(std::env::current_exe()?)
        .arg("--exact")
        .arg("keeper::tests::writer_lock_child_dispatch")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("QUILL_WRITER_LOCK_ROLE", role)
        .env("QUILL_WRITER_LOCK_DIRECTORY", directory)
        .env("QUILL_WRITER_LOCK_CONTROL", control)
        .env("QUILL_WRITER_LOCK_ID", identifier)
        .spawn()?;
    Ok(WriterChild { child: Some(child) })
}

#[cfg(any(unix, windows))]
fn wait_for_child_marker(child: &mut WriterChild, marker: &Path, label: &str) -> TestResult {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if marker.exists() {
            return Ok(());
        }
        if let Some(status) = child.child_mut().try_wait()? {
            return Err(format!("{label} exited before its marker: {status}").into());
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!("timed out waiting for {label}").into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn writer_lock_wire_golden_and_corruption_matrix() {
    let record = WriterLockRecord {
        pid: 42,
        pid_start_nonce: 0x1122_3344_5566_7788,
        acquired_unix_s: 1_700_000_000,
    };
    let golden = hex_bytes(
        "46534c584c434b00010000002a0000008877665544332211\
         00f153650000000044f89139",
    );
    assert_eq!(record.to_bytes().as_slice(), golden);
    assert_eq!(WriterLockRecord::from_bytes(&golden), Ok(record));
    for length in 0..WRITER_LOCK_RECORD_BYTES {
        assert!(WriterLockRecord::from_bytes(&golden[..length]).is_err());
    }
    let mut bad_magic = golden.clone();
    bad_magic[0] ^= 0xff;
    assert!(WriterLockRecord::from_bytes(&bad_magic).is_err());
    let mut bad_version = golden.clone();
    bad_version[8] = 2;
    let checksum = crc32fast::hash(&bad_version[..32]);
    bad_version[32..].copy_from_slice(&checksum.to_le_bytes());
    assert!(WriterLockRecord::from_bytes(&bad_version).is_err());
    let mut zero_pid = golden.clone();
    zero_pid[12..16].copy_from_slice(&0_u32.to_le_bytes());
    let checksum = crc32fast::hash(&zero_pid[..32]);
    zero_pid[32..].copy_from_slice(&checksum.to_le_bytes());
    assert!(WriterLockRecord::from_bytes(&zero_pid).is_err());
    let mut bad_crc = golden;
    bad_crc[20] ^= 1;
    assert!(WriterLockRecord::from_bytes(&bad_crc).is_err());
}

#[cfg(windows)]
#[test]
fn windows_writer_admission_is_cross_process_exclusive_and_reusable() -> TestResult {
    let index = tempdir()?;
    create_test_index(index.path()).map_err(io::Error::other)?;

    let holder_control = tempdir()?;
    let mut holder = spawn_writer_child("hold", index.path(), holder_control.path(), "first")?;
    wait_for_child_marker(
        &mut holder,
        &holder_control.path().join("ready-first"),
        "first Windows writer",
    )?;

    let contender_control = tempdir()?;
    let mut contender =
        spawn_writer_child("contend", index.path(), contender_control.path(), "blocked")?;
    std::fs::write(contender_control.path().join("START"), [])?;
    wait_for_child_marker(
        &mut contender,
        &contender_control.path().join("busy-blocked"),
        "blocked Windows writer",
    )?;
    contender.wait_success()?;

    std::fs::write(holder_control.path().join("RELEASE"), [])?;
    holder.wait_success()?;

    let successor_control = tempdir()?;
    let mut successor =
        spawn_writer_child("hold", index.path(), successor_control.path(), "second")?;
    wait_for_child_marker(
        &mut successor,
        &successor_control.path().join("ready-second"),
        "successor Windows writer",
    )?;
    std::fs::write(successor_control.path().join("RELEASE"), [])?;
    successor.wait_success()?;
    Ok(())
}

#[cfg(windows)]
#[test]
fn windows_writer_admission_prevents_replacement_and_rejects_non_file_lock() -> TestResult {
    let directory = tempdir()?;
    let admission = acquire_writer_admission(directory.path())?;
    let lock_path = directory.path().join("LOCK");
    let displaced = directory.path().join("LOCK.displaced");
    assert!(
        std::fs::rename(&lock_path, &displaced).is_err(),
        "the retained Windows handle must deny LOCK replacement"
    );
    assert!(matches!(
        acquire_writer_admission(directory.path()),
        Err(KeeperError::WriterBusy { .. })
    ));
    drop(admission);

    std::fs::rename(&lock_path, &displaced)?;
    std::fs::create_dir(&lock_path)?;
    assert!(matches!(
        acquire_writer_admission(directory.path()),
        Err(KeeperError::WriterLockCorrupted { .. })
    ));
    Ok(())
}

#[cfg(windows)]
#[test]
fn windows_writer_create_publish_and_reopen_committed_generation() -> TestResult {
    let index = tempdir()?;
    let directory = index.path().to_path_buf();
    let published: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let schema_id = DEFAULT_SCHEMA
            .schema_id()
            .map_err(|error| error.to_string())?;
        writer
            .publish(&cx, &Manifest::empty(2, schema_id, 0))
            .await
            .map_err(|error| error.to_string())?;
        assert!(!directory.join("gen-2.claim").exists());
        assert_eq!(
            writer
                .retained_snapshot_for_bookkeeping()
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        drop(writer);

        let reopened =
            KeeperSnapshot::open(&directory, DEFAULT_SCHEMA).map_err(|error| error.to_string())?;
        assert_eq!(reopened.loaded_manifest().manifest.generation, 2);
        Ok(())
    });
    published.map_err(io::Error::other)?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn writer_admission_atomically_replaces_and_recovers_a_torn_record() -> TestResult {
    use std::os::unix::fs::MetadataExt;

    let directory = tempdir()?;
    let lock_path = directory.path().join("LOCK");
    std::fs::write(&lock_path, [])?;
    let empty_inode = std::fs::metadata(&lock_path)?.ino();
    let first = acquire_writer_admission(directory.path())?;
    let active_inode = std::fs::metadata(&lock_path)?.ino();
    assert_ne!(
        active_inode, empty_inode,
        "record publication must replace rather than overwrite LOCK"
    );
    let active_bytes = std::fs::read(&lock_path)?;
    assert_eq!(active_bytes.len(), WRITER_LOCK_RECORD_BYTES);
    assert!(matches!(
        acquire_writer_admission(directory.path()),
        Err(KeeperError::WriterBusy {
            owner_pid: Some(pid),
            ..
        }) if pid == std::process::id()
    ));
    assert_eq!(std::fs::read(&lock_path)?, active_bytes);
    drop(first);
    assert_eq!(std::fs::metadata(&lock_path)?.len(), 0);

    let second = acquire_writer_admission(directory.path())?;
    drop(second);
    std::fs::write(&lock_path, b"truncated")?;
    let torn_inode = std::fs::metadata(&lock_path)?.ino();
    let recovered = acquire_writer_admission(directory.path())?;
    assert_ne!(
        std::fs::metadata(&lock_path)?.ino(),
        torn_inode,
        "torn record recovery must install a complete replacement inode"
    );
    let recovered_bytes = std::fs::read(&lock_path)?;
    assert_eq!(
        WriterLockRecord::from_bytes(&recovered_bytes),
        Ok(recovered.record)
    );
    assert!(matches!(
        acquire_writer_admission(directory.path()),
        Err(KeeperError::WriterBusy { .. })
    ));
    drop(recovered);
    assert_eq!(std::fs::metadata(lock_path)?.len(), 0);
    Ok(())
}

#[cfg(unix)]
#[test]
fn writer_admission_is_exclusive_reusable_and_repairs_torn_records() -> TestResult {
    let directory = tempdir()?;
    let first = acquire_writer_admission(directory.path())?;
    let lock_path = directory.path().join("LOCK");
    let active_bytes = std::fs::read(&lock_path)?;
    assert_eq!(active_bytes.len(), WRITER_LOCK_RECORD_BYTES);
    assert!(matches!(
        acquire_writer_admission(directory.path()),
        Err(KeeperError::WriterBusy {
            owner_pid: Some(pid),
            ..
        }) if pid == std::process::id()
    ));
    assert_eq!(std::fs::read(&lock_path)?, active_bytes);
    drop(first);
    assert_eq!(std::fs::metadata(&lock_path)?.len(), 0);

    let second = acquire_writer_admission(directory.path())?;
    drop(second);
    std::fs::write(&lock_path, b"truncated")?;
    let repaired = acquire_writer_admission(directory.path())?;
    assert_eq!(
        std::fs::metadata(&lock_path)?.len(),
        usize_to_u64(WRITER_LOCK_RECORD_BYTES)
    );
    drop(repaired);
    assert_eq!(std::fs::metadata(lock_path)?.len(), 0);
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn live_reused_pid_with_a_different_start_nonce_does_not_block_takeover() -> TestResult {
    let directory = tempdir()?;
    let control = tempdir()?;
    let ready = control.path().join("ready-reused");
    let mut child = spawn_writer_child("live_pid", directory.path(), control.path(), "reused")?;
    wait_for_child_marker(&mut child, &ready, "live pid fixture")?;

    let pid = child.child_mut().id();
    let acquired_unix_s = 1_700_000_000;
    let observed_nonce = linux_process_start_time(pid)
        .map(|start_time| writer_pid_start_nonce_from_linux_start_time(pid, start_time))
        .ok_or_else(|| io::Error::other("live child start identity is observable"))?;
    let stale = WriterLockRecord {
        pid,
        pid_start_nonce: observed_nonce ^ 1,
        acquired_unix_s,
    };
    std::fs::write(directory.path().join("LOCK"), stale.to_bytes())?;
    assert!(
        !writer_lock_record_names_live_owner(stale),
        "same numeric pid with a different process start must be stale"
    );

    let admission = acquire_writer_admission(directory.path())?;
    drop(admission);

    let matching = WriterLockRecord {
        pid,
        pid_start_nonce: observed_nonce,
        acquired_unix_s,
    };
    std::fs::write(directory.path().join("LOCK"), matching.to_bytes())?;
    assert!(writer_lock_record_names_live_owner(matching));
    assert!(matches!(
        acquire_writer_admission(directory.path()),
        Err(KeeperError::WriterBusy {
            owner_pid: Some(owner),
            ..
        }) if owner == pid
    ));

    std::fs::write(control.path().join("RELEASE"), [])?;
    child.wait_success()?;
    let after_exit = acquire_writer_admission(directory.path())?;
    drop(after_exit);
    Ok(())
}

#[cfg(unix)]
#[test]
fn writer_admission_replaces_a_live_pid_record_with_a_stale_start_nonce() -> TestResult {
    let directory = tempdir()?;
    let admission = acquire_writer_admission(directory.path())?;
    let lock_path = directory.path().join("LOCK");
    drop(admission);

    let current = WriterLockRecord::current(&lock_path)?;
    let stale = WriterLockRecord {
        pid_start_nonce: current.pid_start_nonce ^ 1,
        ..current
    };
    std::fs::write(&lock_path, stale.to_bytes())?;

    let replacement = acquire_writer_admission(directory.path())?;
    assert_ne!(replacement.record.pid_start_nonce, stale.pid_start_nonce);
    drop(replacement);
    assert_eq!(std::fs::metadata(lock_path)?.len(), 0);
    Ok(())
}

#[cfg(unix)]
#[test]
fn stale_writer_start_nonce_is_not_reported_as_live() -> TestResult {
    let directory = tempdir()?;
    let lock_path = directory.path().join("LOCK");
    let current = WriterLockRecord::current(&lock_path)?;
    let stale = WriterLockRecord {
        pid_start_nonce: current.pid_start_nonce ^ 1,
        ..current
    };
    std::fs::write(&lock_path, stale.to_bytes())?;

    assert!(!detect_live_writer(directory.path()));
    Ok(())
}

#[cfg(unix)]
#[test]
fn old_guards_preserve_replacement_lock_and_claim_names() -> TestResult {
    let directory = tempdir()?;
    let admission = acquire_writer_admission(directory.path())?;
    let original_lock = directory.path().join("LOCK");
    let moved_lock = directory.path().join("LOCK.old");
    std::fs::rename(&original_lock, &moved_lock)?;
    let replacement = WriterLockRecord {
        pid: std::process::id(),
        pid_start_nonce: 0xfeed_face,
        acquired_unix_s: 1_700_000_001,
    }
    .to_bytes();
    std::fs::write(&original_lock, replacement)?;
    assert!(matches!(
        admission.ensure_directory_identity(),
        Err(KeeperError::WriterLockCorrupted { .. })
    ));
    drop(admission);
    assert_eq!(std::fs::read(&original_lock)?, replacement);
    assert_eq!(std::fs::metadata(&moved_lock)?.len(), 0);

    let claims = tempdir()?;
    let admission = acquire_writer_admission(claims.path())?;
    let claim = GenerationClaimGuard::acquire(Arc::clone(&admission), 1)?;
    assert!(matches!(
        GenerationClaimGuard::acquire(Arc::clone(&admission), 1),
        Err(KeeperError::GenerationClaimConflict { generation: 1, .. })
    ));
    let claim_path = claims.path().join("gen-1.claim");
    let moved_claim = claims.path().join("gen-1.claim.old");
    std::fs::rename(&claim_path, moved_claim)?;
    std::fs::write(&claim_path, [])?;
    drop(claim);
    assert!(
        claim_path.exists(),
        "old guard must preserve replacement claim"
    );
    drop(admission);
    Ok(())
}

#[cfg(unix)]
#[test]
fn sigstop_writer_is_never_reaped_and_readers_remain_touchless() -> TestResult {
    let index = tempdir()?;
    create_test_index(index.path()).map_err(io::Error::other)?;
    let control = tempdir()?;
    let ready = control.path().join("ready-holder");
    let mut child = spawn_writer_child("hold_unpublished", index.path(), control.path(), "holder")?;
    wait_for_child_marker(&mut child, &ready, "writer holder")?;
    assert!(index.path().join("seg-feedfacefeedface.fslx").exists());

    let lock_path = index.path().join("LOCK");
    let lock_file = OpenOptions::new().write(true).open(&lock_path)?;
    lock_file.set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;
    let before = directory_bytes(index.path())?;
    rustix::process::kill_process(
        rustix::process::Pid::from_child(child.child_mut()),
        rustix::process::Signal::STOP,
    )?;

    let contender_path = index.path().to_path_buf();
    let blocked: Result<(), String> = run_with_test_cx(move |cx| async move {
        match KeeperWriter::open(&cx, contender_path, DEFAULT_SCHEMA).await {
            Err(KeeperError::WriterBusy { .. }) => Ok(()),
            Ok(writer) => {
                drop(writer);
                Err("SIGSTOP holder was incorrectly taken over".to_owned())
            }
            Err(error) => Err(format!("unexpected contender failure: {error}")),
        }
    });
    blocked.map_err(io::Error::other)?;
    KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?;
    KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?;
    assert_eq!(directory_bytes(index.path())?, before);

    rustix::process::kill_process(
        rustix::process::Pid::from_child(child.child_mut()),
        rustix::process::Signal::CONT,
    )?;
    std::fs::write(control.path().join("RELEASE"), [])?;
    child.wait_success()?;
    let reopen_path = index.path().to_path_buf();
    let reopened: Result<(), String> = run_with_test_cx(move |cx| async move {
        KeeperWriter::open(&cx, reopen_path, DEFAULT_SCHEMA)
            .await
            .map(drop)
            .map_err(|error| error.to_string())
    });
    reopened.map_err(io::Error::other)?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn two_processes_racing_dead_owner_takeover_admit_exactly_one() -> TestResult {
    let index = tempdir()?;
    create_test_index(index.path()).map_err(io::Error::other)?;
    let dead_control = tempdir()?;
    let mut dead = spawn_writer_child("hold", index.path(), dead_control.path(), "dead")?;
    wait_for_child_marker(
        &mut dead,
        &dead_control.path().join("ready-dead"),
        "dead-owner fixture",
    )?;
    dead.kill_and_reap()?;
    assert_eq!(
        std::fs::metadata(index.path().join("LOCK"))?.len(),
        usize_to_u64(WRITER_LOCK_RECORD_BYTES)
    );

    let control = tempdir()?;
    let mut first = spawn_writer_child("contend", index.path(), control.path(), "a")?;
    let mut second = spawn_writer_child("contend", index.path(), control.path(), "b")?;
    std::fs::write(control.path().join("START"), [])?;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let first_done =
            control.path().join("winner-a").exists() || control.path().join("busy-a").exists();
        let second_done =
            control.path().join("winner-b").exists() || control.path().join("busy-b").exists();
        if first_done && second_done {
            break;
        }
        if std::time::Instant::now() >= deadline {
            return Err("timed out waiting for takeover contenders".into());
        }
        if let Some(status) = first.child_mut().try_wait()?
            && !first_done
        {
            return Err(format!("first contender exited early: {status}").into());
        }
        if let Some(status) = second.child_mut().try_wait()?
            && !second_done
        {
            return Err(format!("second contender exited early: {status}").into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let winners = usize::from(control.path().join("winner-a").exists())
        + usize::from(control.path().join("winner-b").exists());
    let busy = usize::from(control.path().join("busy-a").exists())
        + usize::from(control.path().join("busy-b").exists());
    assert_eq!(winners, 1);
    assert_eq!(busy, 1);
    std::fs::write(control.path().join("RELEASE"), [])?;
    first.wait_success()?;
    second.wait_success()?;
    assert_eq!(std::fs::metadata(index.path().join("LOCK"))?.len(), 0);
    Ok(())
}

#[cfg(unix)]
#[test]
fn keeper_writer_claims_generations_and_abandons_interrupted_temp() -> TestResult {
    let index = tempdir()?;
    let directory = index.path().to_path_buf();
    let first: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(
            std::fs::metadata(directory.join("LOCK"))
                .map_err(|error| error.to_string())?
                .len(),
            usize_to_u64(WRITER_LOCK_RECORD_BYTES)
        );
        let schema_id = DEFAULT_SCHEMA
            .schema_id()
            .map_err(|error| error.to_string())?;
        writer
            .publish(&cx, &Manifest::empty(2, schema_id, 0))
            .await
            .map_err(|error| error.to_string())?;
        assert!(!directory.join("gen-2.claim").exists());
        assert_eq!(
            writer
                .retained_snapshot_for_bookkeeping()
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        drop(writer);
        Ok(())
    });
    first.map_err(io::Error::other)?;
    assert_eq!(std::fs::metadata(index.path().join("LOCK"))?.len(), 0);

    let schema_id = DEFAULT_SCHEMA.schema_id()?;
    let interrupted = Manifest::empty(3, schema_id, 0).to_bytes()?;
    std::fs::write(index.path().join(".tmp-manifest-3"), interrupted)?;
    std::fs::write(index.path().join("gen-3.claim"), [])?;
    let directory = index.path().to_path_buf();
    let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(
            writer
                .retained_snapshot_for_bookkeeping()
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        assert!(!directory.join(".tmp-manifest-3").exists());
        assert!(!directory.join("gen-3.claim").exists());
        writer
            .publish(&cx, &Manifest::empty(3, schema_id, 0))
            .await
            .map_err(|error| error.to_string())?;
        assert_eq!(
            writer
                .retained_snapshot_for_bookkeeping()
                .loaded_manifest()
                .manifest
                .generation,
            3
        );
        Ok(())
    });
    recovered.map_err(io::Error::other)?;
    assert!(std::fs::read_dir(index.path())?.any(|entry| {
        entry
            .ok()
            .and_then(|entry| entry.file_name().into_string().ok())
            .is_some_and(|name| name.starts_with(".tmp-abandoned-manifest-3-"))
    }));
    Ok(())
}
