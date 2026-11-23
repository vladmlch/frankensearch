use super::*;

#[test]
fn real_manifest_choreography_recovers_at_every_checkpoint() -> TestResult {
    const CHECKPOINTS: [PublishCheckpoint; 6] = [
        PublishCheckpoint::TempWritten,
        PublishCheckpoint::TempSynced,
        PublishCheckpoint::GenerationClaimed,
        PublishCheckpoint::CurrentMovedToPrevious,
        PublishCheckpoint::TempMovedToCurrent,
        PublishCheckpoint::DirectorySynced,
    ];

    let success_directory = tempdir()?;
    let schema_id = DEFAULT_SCHEMA.schema_id()?;
    write_manifest(
        &success_directory.path().join("MANIFEST"),
        &Manifest::empty(1, schema_id, 0),
    )?;
    let proposed = Manifest::empty(2, schema_id, 0);
    let proposed_bytes = proposed.to_bytes()?;
    let mut observed = Vec::new();
    publish_manifest_choreography(
        success_directory.path().to_path_buf(),
        &proposed_bytes,
        |_, _| Ok(()),
        |checkpoint, _| {
            observed.push(checkpoint);
            Ok(())
        },
    )?;
    assert_eq!(observed, CHECKPOINTS);

    for fault in CHECKPOINTS {
        let directory = tempdir()?;
        let initial = Manifest::empty(1, schema_id, 0);
        write_manifest(&directory.path().join("MANIFEST"), &initial)?;
        let result = publish_manifest_choreography(
            directory.path().to_path_buf(),
            &proposed_bytes,
            |_, _| Ok(()),
            |checkpoint, path| {
                if checkpoint == fault {
                    return Err(KeeperError::Io {
                        operation: "inject manifest publish crash",
                        path: path.to_path_buf(),
                        source: io::Error::other(format!("injected crash at {checkpoint:?}")),
                    });
                }
                Ok(())
            },
        );
        assert!(
            matches!(
                result,
                Err(KeeperError::Io {
                    operation: "inject manifest publish crash",
                    ..
                })
            ),
            "{fault:?}"
        );

        let before_open = directory_bytes(directory.path())?;
        let first = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        let second = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(directory_bytes(directory.path())?, before_open, "{fault:?}");
        assert_eq!(
            first.loaded_manifest(),
            second.loaded_manifest(),
            "{fault:?}"
        );
        let expected_generation = match fault {
            PublishCheckpoint::TempWritten
            | PublishCheckpoint::TempSynced
            | PublishCheckpoint::GenerationClaimed
            | PublishCheckpoint::CurrentMovedToPrevious => 1,
            PublishCheckpoint::TempMovedToCurrent | PublishCheckpoint::DirectorySynced => 2,
        };
        assert_eq!(
            first.loaded_manifest().manifest.generation,
            expected_generation,
            "{fault:?}"
        );
        let expected_source = if fault == PublishCheckpoint::CurrentMovedToPrevious {
            ManifestSource::PreviousAfterMissingCurrent
        } else {
            ManifestSource::Current
        };
        assert_eq!(first.loaded_manifest().source, expected_source, "{fault:?}");

        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        let report = collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?;
        let temp_survived_fault = matches!(
            fault,
            PublishCheckpoint::TempWritten
                | PublishCheckpoint::TempSynced
                | PublishCheckpoint::GenerationClaimed
                | PublishCheckpoint::CurrentMovedToPrevious
        );
        assert_eq!(
            report.removed,
            if temp_survived_fault {
                vec![PathBuf::from(".tmp-manifest-2")]
            } else {
                Vec::new()
            },
            "{fault:?}"
        );
        let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(
            reopened.loaded_manifest(),
            first.loaded_manifest(),
            "{fault:?}"
        );
    }
    Ok(())
}

#[test]
fn retry_reuses_byte_identical_temp_after_claim_failure() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        publisher
            .publish(&cx, &sample_manifest(1))
            .await
            .expect("publish genesis");
        let proposed = sample_manifest(2);
        let expected_bytes = proposed.to_bytes().expect("encode proposal");

        let first_result = publisher
            .publish_with_generation_claim::<(), _>(&cx, &proposed, |path, generation| {
                Err(KeeperError::Io {
                    operation: "test generation claim",
                    path: path.join(format!("gen-{generation}.claim")),
                    source: io::Error::other("injected claim failure"),
                })
            })
            .await;
        assert!(matches!(
            first_result,
            Err(KeeperError::Io {
                operation: "test generation claim",
                ..
            })
        ));
        let temp_path = directory.path().join(".tmp-manifest-2");
        assert_eq!(
            std::fs::read(&temp_path).expect("read durable temp"),
            expected_bytes
        );

        publisher
            .publish(&cx, &proposed)
            .await
            .expect("retry byte-identical durable temp");
        assert_eq!(
            load_manifest_pair(directory.path())
                .expect("reopen retried publish")
                .manifest,
            proposed
        );
    });
}

#[test]
fn mismatched_existing_temp_fails_closed_without_overwrite() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        let temp_path = directory.path().join(".tmp-manifest-1");
        let stale_bytes = b"stale temp from another proposal";
        std::fs::write(&temp_path, stale_bytes).expect("write stale temp fixture");

        assert!(matches!(
            publisher.publish(&cx, &sample_manifest(1)).await,
            Err(KeeperError::TempConflict { .. })
        ));
        assert_eq!(
            std::fs::read(&temp_path).expect("read preserved stale temp"),
            stale_bytes
        );
        assert!(!directory.path().join("MANIFEST").exists());
    });
}

#[cfg(unix)]
#[test]
fn existing_temp_symlink_fails_closed_without_publication() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        let expected_bytes = sample_manifest(1).to_bytes().expect("encode proposal");
        let symlink_target = directory.path().join("byte-identical-target");
        std::fs::write(&symlink_target, expected_bytes).expect("write symlink target");
        let temp_path = directory.path().join(".tmp-manifest-1");
        std::os::unix::fs::symlink(&symlink_target, &temp_path)
            .expect("create temp symlink fixture");

        assert!(matches!(
            publisher.publish(&cx, &sample_manifest(1)).await,
            Err(KeeperError::Io {
                operation: "open existing temp",
                ..
            })
        ));
        assert!(
            std::fs::symlink_metadata(&temp_path)
                .expect("stat preserved temp symlink")
                .file_type()
                .is_symlink()
        );
        assert!(!directory.path().join("MANIFEST").exists());
    });
}

#[cfg(unix)]
#[test]
fn labruntime_serializes_concurrent_publishers_across_a_late_symlink_alias() {
    let root = tempdir().expect("temp directory");
    let directory = root.path().join("target-index");
    let alias = root.path().join("index-alias");
    std::fs::create_dir(&directory).expect("create target index directory");
    let production_first_publisher = ManifestPublisher::new(&alias);
    let production_second_publisher = ManifestPublisher::new(&directory);
    let test_publish_lock = Arc::new(Mutex::with_name(
        "quill.manifest_publish.late_symlink_alias_test",
        (),
    ));
    let first_publisher =
        ManifestPublisher::with_publish_lock_for_test(&alias, Arc::clone(&test_publish_lock));
    let second_publisher =
        ManifestPublisher::with_publish_lock_for_test(&directory, Arc::clone(&test_publish_lock));
    std::os::unix::fs::symlink(&directory, &alias).expect("create alias after first publisher");
    let production_lock = production_first_publisher.publish_lock_for_test();
    assert!(Arc::ptr_eq(
        &production_lock,
        &production_second_publisher.publish_lock_for_test()
    ));
    let lock = first_publisher.publish_lock_for_test();
    assert!(Arc::ptr_eq(
        &lock,
        &second_publisher.publish_lock_for_test()
    ));
    assert!(!Arc::ptr_eq(&lock, &production_lock));
    let saw_two_waiters = Arc::new(AtomicBool::new(false));
    let successes = Arc::new(AtomicUsize::new(0));
    let conflicts = Arc::new(AtomicUsize::new(0));

    let mut lab = LabRuntime::new(LabConfig::new(0x51a1).max_steps(100_000));
    let region = lab.state.create_root_region(Budget::INFINITE);
    let holder_lock = Arc::clone(&lock);
    let holder_witness = Arc::clone(&saw_two_waiters);
    let (holder, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            let cx = Cx::for_testing();
            let guard = OwnedMutexGuard::lock(Arc::clone(&holder_lock), &cx)
                .await
                .expect("holder lock");
            while holder_lock.waiters() < 2 {
                yield_now().await;
            }
            holder_witness.store(true, Ordering::SeqCst);
            drop(guard);
        })
        .expect("create holder task");

    let first_successes = Arc::clone(&successes);
    let first_conflicts = Arc::clone(&conflicts);
    let (first_task, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            let cx = Cx::for_testing();
            match first_publisher.publish(&cx, &sample_manifest(1)).await {
                Ok(_) => {
                    first_successes.fetch_add(1, Ordering::SeqCst);
                }
                Err(KeeperError::GenerationConflict {
                    expected: 2,
                    proposed: 1,
                }) => {
                    first_conflicts.fetch_add(1, Ordering::SeqCst);
                }
                Err(error) => panic!("unexpected first publish result: {error}"),
            }
        })
        .expect("create first publisher");

    let second_successes = Arc::clone(&successes);
    let second_conflicts = Arc::clone(&conflicts);
    let (second_task, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            let cx = Cx::for_testing();
            match second_publisher.publish(&cx, &sample_manifest(1)).await {
                Ok(_) => {
                    second_successes.fetch_add(1, Ordering::SeqCst);
                }
                Err(KeeperError::GenerationConflict {
                    expected: 2,
                    proposed: 1,
                }) => {
                    second_conflicts.fetch_add(1, Ordering::SeqCst);
                }
                Err(error) => panic!("unexpected second publish result: {error}"),
            }
        })
        .expect("create second publisher");

    lab.scheduler.lock().schedule(holder, 0);
    lab.step_for_test();
    lab.scheduler.lock().schedule(first_task, 0);
    lab.scheduler.lock().schedule(second_task, 0);
    let report = lab.run_until_quiescent_with_report();

    assert!(saw_two_waiters.load(Ordering::SeqCst));
    assert_eq!(successes.load(Ordering::SeqCst), 1);
    assert_eq!(conflicts.load(Ordering::SeqCst), 1);
    assert!(report.quiescent, "LabRuntime must reach quiescence");
    assert!(report.oracle_report.all_passed(), "oracles must pass");
    assert!(report.invariant_violations.is_empty());
    assert_eq!(
        load_manifest_pair(&directory)
            .expect("reopen after concurrent publishes")
            .manifest
            .generation,
        1
    );
}

#[cfg(unix)]
#[test]
fn labruntime_writer_admission_blocks_a_second_writer_but_not_a_reader() -> TestResult {
    let index = tempdir()?;
    create_test_index(index.path()).map_err(io::Error::other)?;
    let directory = index.path().to_path_buf();
    let holder_ready = Arc::new(AtomicBool::new(false));
    let contender_was_busy = Arc::new(AtomicBool::new(false));
    let reader_was_touchless = Arc::new(AtomicBool::new(false));
    let observers_done = Arc::new(AtomicUsize::new(0));

    let mut lab = LabRuntime::new(LabConfig::new(0x10c6).max_steps(100_000));
    let region = lab.state.create_root_region(Budget::INFINITE);

    let holder_directory = directory.clone();
    let holder_ready_task = Arc::clone(&holder_ready);
    let holder_done = Arc::clone(&observers_done);
    let (holder, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            let cx = Cx::for_testing();
            let writer = KeeperWriter::open(&cx, holder_directory, DEFAULT_SCHEMA)
                .await
                .expect("first writer admission");
            holder_ready_task.store(true, Ordering::SeqCst);
            while holder_done.load(Ordering::SeqCst) < 2 {
                yield_now().await;
            }
            drop(writer);
        })
        .expect("create holder task");

    let contender_directory = directory.clone();
    let contender_ready = Arc::clone(&holder_ready);
    let contender_result = Arc::clone(&contender_was_busy);
    let contender_done = Arc::clone(&observers_done);
    let (contender, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            while !contender_ready.load(Ordering::SeqCst) {
                yield_now().await;
            }
            let cx = Cx::for_testing();
            match KeeperWriter::open(&cx, contender_directory, DEFAULT_SCHEMA).await {
                Err(KeeperError::WriterBusy { .. }) => {
                    contender_result.store(true, Ordering::SeqCst);
                }
                Ok(writer) => {
                    drop(writer);
                    panic!("second writer was admitted while the first held LOCK");
                }
                Err(error) => panic!("unexpected second-writer result: {error}"),
            }
            contender_done.fetch_add(1, Ordering::SeqCst);
        })
        .expect("create contender task");

    let reader_directory = directory.clone();
    let reader_ready = Arc::clone(&holder_ready);
    let reader_result = Arc::clone(&reader_was_touchless);
    let reader_done = Arc::clone(&observers_done);
    let (reader, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            while !reader_ready.load(Ordering::SeqCst) {
                yield_now().await;
            }
            let before = directory_bytes(&reader_directory).expect("reader inventory before");
            let snapshot = KeeperSnapshot::open(&reader_directory, DEFAULT_SCHEMA)
                .expect("reader opens while writer is live");
            assert_eq!(snapshot.loaded_manifest().manifest.generation, 1);
            let after = directory_bytes(&reader_directory).expect("reader inventory after");
            assert_eq!(after, before, "reader open must not touch the directory");
            reader_result.store(true, Ordering::SeqCst);
            reader_done.fetch_add(1, Ordering::SeqCst);
        })
        .expect("create reader task");

    lab.scheduler.lock().schedule(holder, 0);
    lab.step_for_test();
    assert!(holder_ready.load(Ordering::SeqCst));
    lab.scheduler.lock().schedule(contender, 0);
    lab.scheduler.lock().schedule(reader, 0);
    let report = lab.run_until_quiescent_with_report();

    assert!(contender_was_busy.load(Ordering::SeqCst));
    assert!(reader_was_touchless.load(Ordering::SeqCst));
    assert!(report.quiescent, "LabRuntime must reach quiescence");
    assert!(report.oracle_report.all_passed(), "oracles must pass");
    assert!(report.invariant_violations.is_empty());
    assert_eq!(std::fs::metadata(directory.join("LOCK"))?.len(), 0);
    Ok(())
}

#[test]
fn keeper_create_is_create_or_open_and_in_memory_creates_no_files() {
    let memory = KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("in-memory genesis");
    assert_eq!(memory.directory(), None);
    assert_eq!(memory.loaded_manifest().source, ManifestSource::InMemory);
    assert_eq!(memory.loaded_manifest().manifest.generation, 1);
    assert!(memory.segments().is_empty());

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let root = tempdir().expect("temp directory");
        let directory = root.path().join("quill-index");
        let created = KeeperSnapshot::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .expect("create genesis");
        assert_eq!(created.directory(), Some(directory.as_path()));
        assert_eq!(created.loaded_manifest().manifest.generation, 1);
        let manifest_bytes = std::fs::read(directory.join("MANIFEST")).expect("manifest");

        let reopened = KeeperSnapshot::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .expect("create-or-open existing index");
        assert_eq!(reopened.loaded_manifest().manifest.generation, 1);
        assert_eq!(
            std::fs::read(directory.join("MANIFEST")).expect("manifest after reopen"),
            manifest_bytes
        );
        assert!(matches!(
            KeeperSnapshot::create(&cx, &directory, FSFS_CHUNK_SCHEMA).await,
            Err(KeeperError::SchemaMismatch { .. })
        ));
    });
}
