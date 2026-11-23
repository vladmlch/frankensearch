use super::*;

fn run_with_blocking_cx<F, Fut>(operation: F)
where
    F: FnOnce(Cx) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let cx = Cx::for_testing();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("build test runtime with a blocking pool");
    runtime.block_on(operation(cx));
}

/// A wall-clock bound for the test-only durable-publish rendezvous.
///
/// The production publisher reaches `DirectorySynced` in its dedicated
/// blocking lane. The test waits for that lane's checkpoint event rather
/// than assuming a fixed number of async scheduler yields is enough for
/// an OS directory fsync. The bound only has to catch a hang, never to
/// race the scheduler: at 2s it flaked under the full parallel suite when
/// the blocking pool was saturated by ~660 neighbours (bd-zam2i).
const DURABLE_PUBLISH_CHECKPOINT_TIMEOUT: Duration = Duration::from_secs(30);

#[test]
fn dropped_publish_future_after_durable_install_advances_the_next_publication() -> TestResult {
    let _pause_serial = manifest_publish_pause_test_serial_guard();
    let index = tempdir()?;
    let directory = index.path().to_path_buf();
    run_with_blocking_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .expect("create genesis index");
        let schema_id = DEFAULT_SCHEMA.schema_id().expect("default schema identity");
        assert_eq!(
            writer
                .snapshot()
                .expect("fresh writer snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        assert!(!writer.publication_awaits_reconciliation());

        // Use the pre-existing production choreography checkpoint: after
        // its directory fsync generation 2 is durable, but the future has
        // not yet resumed to install its refreshed retained snapshot.
        let successor = Manifest::empty(2, schema_id, 0);
        let pause = pause_manifest_publish_at_checkpoint_for_test(
            &directory,
            PublishCheckpoint::DirectorySynced,
        );
        let mut publish = Box::pin(writer.publish(&cx, &successor));
        drive_manifest_publish_to_checkpoint_for_test(
            &mut publish,
            &pause,
            DURABLE_PUBLISH_CHECKPOINT_TIMEOUT,
        )
        .await
        .expect("reach the bounded durable-publish rendezvous");
        drop(publish);
        pause.release();

        let installed = Manifest::from_bytes(
            &std::fs::read(directory.join("MANIFEST")).expect("read installed MANIFEST"),
        )
        .expect("decode installed MANIFEST");
        assert_eq!(
            installed.generation, 2,
            "the real publisher must have made generation N+1 durable before the drop"
        );
        assert!(matches!(
            writer.snapshot(),
            Err(KeeperError::PublicationReconciliationRequired {
                retained_generation: 1,
                proposed_generation: 2,
            })
        ));
        let stats_error = SegmentStatsProvider::segment_stats(&writer)
            .expect_err("writer stats must fail closed with its authority snapshot");
        assert!(
            matches!(
                &stats_error,
                SearchError::SubsystemError { subsystem, source }
                    if *subsystem == "quill"
                        && matches!(
                            source.downcast_ref::<KeeperError>(),
                            Some(KeeperError::PublicationReconciliationRequired {
                                retained_generation: 1,
                                proposed_generation: 2,
                            })
                        )
            ),
            "writer stats must retain the typed reconciliation failure: {stats_error:?}"
        );
        assert!(
            writer.publication_awaits_reconciliation(),
            "marker observation alone is not authority; the checked reader must fail closed"
        );

        // The checked generation-1 snapshot assertion immediately above
        // is parent-red. This generation-3 call verifies the corrected
        // reconciliation behavior after that fail-closed observation.
        let advanced = writer
            .publish(&cx, &Manifest::empty(3, schema_id, 0))
            .await
            .expect("the next publication reconciles the abandoned generation")
            .loaded_manifest()
            .manifest
            .generation;
        assert_eq!(advanced, 3);
        assert!(!writer.publication_awaits_reconciliation());
        assert_eq!(
            SegmentStatsProvider::segment_stats(&writer)
                .expect("reconciled writer stats are authoritative")
                .published_generation,
            3
        );

        let reopened =
            KeeperSnapshot::open(&directory, DEFAULT_SCHEMA).expect("reopen published index");
        assert_eq!(
            writer
                .snapshot()
                .expect("reconciled writer snapshot is authoritative")
                .loaded_manifest()
                .manifest,
            reopened.loaded_manifest().manifest,
            "the retained handle and a fresh reopen must agree exactly"
        );
        let previous = Manifest::from_bytes(
            &std::fs::read(directory.join("MANIFEST.prev")).expect("read previous MANIFEST"),
        )
        .expect("decode previous MANIFEST");
        assert_eq!(
            previous.generation, 2,
            "the abandoned generation is retained as the previous slot, never rolled back"
        );
    });
    Ok(())
}

#[test]
fn abandoned_publication_is_adopted_by_its_identical_retry_and_by_reconciliation() -> TestResult {
    let _pause_serial = manifest_publish_pause_test_serial_guard();
    let index = tempdir()?;
    let directory = index.path().to_path_buf();
    run_with_blocking_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .expect("create genesis index");
        let schema_id = DEFAULT_SCHEMA.schema_id().expect("default schema identity");

        // An owner that retries the exact abandoned proposal adopts the
        // durable generation instead of publishing a second successor.
        let second = Manifest::empty(2, schema_id, 0);
        let pause = pause_manifest_publish_at_checkpoint_for_test(
            &directory,
            PublishCheckpoint::DirectorySynced,
        );
        let mut publish = Box::pin(writer.publish(&cx, &second));
        drive_manifest_publish_to_checkpoint_for_test(
            &mut publish,
            &pause,
            DURABLE_PUBLISH_CHECKPOINT_TIMEOUT,
        )
        .await
        .expect("reach the bounded durable-publish rendezvous");
        drop(publish);
        pause.release();

        let adopted = writer
            .publish(&cx, &second)
            .await
            .expect("the identical retry adopts the durable generation")
            .loaded_manifest()
            .manifest
            .generation;
        assert_eq!(
            adopted, 2,
            "an identical retry must reconcile generation N+1, not publish N+2"
        );
        assert!(!writer.publication_awaits_reconciliation());
        let previous = Manifest::from_bytes(
            &std::fs::read(directory.join("MANIFEST.prev")).expect("read previous MANIFEST"),
        )
        .expect("decode previous MANIFEST");
        assert_eq!(
            previous.generation, 1,
            "adoption must not publish a second successor over the abandoned one"
        );

        // An owner that never retries still gets an honest view: explicit
        // reconciliation proves the retained snapshot against disk.
        let third = Manifest::empty(3, schema_id, 0);
        let pause = pause_manifest_publish_at_checkpoint_for_test(
            &directory,
            PublishCheckpoint::DirectorySynced,
        );
        let mut publish = Box::pin(writer.publish(&cx, &third));
        drive_manifest_publish_to_checkpoint_for_test(
            &mut publish,
            &pause,
            DURABLE_PUBLISH_CHECKPOINT_TIMEOUT,
        )
        .await
        .expect("reach the bounded durable-publish rendezvous");
        drop(publish);
        pause.release();

        assert!(matches!(
            writer.snapshot(),
            Err(KeeperError::PublicationReconciliationRequired {
                retained_generation: 2,
                proposed_generation: 3,
            })
        ));
        let reconciled = writer
            .reconcile_publication(&cx)
            .await
            .expect("reconcile the abandoned publication")
            .loaded_manifest()
            .manifest
            .clone();
        assert_eq!(reconciled.generation, 3);
        assert!(!writer.publication_awaits_reconciliation());
        let reopened =
            KeeperSnapshot::open(&directory, DEFAULT_SCHEMA).expect("reopen published index");
        assert_eq!(
            reconciled,
            reopened.loaded_manifest().manifest,
            "reconciliation must install exactly what durable authority holds"
        );
        // Reconciling again is inert, and publication continues monotonically.
        writer
            .reconcile_publication(&cx)
            .await
            .expect("reconciliation without an abandoned publication is a no-op");
        let advanced = writer
            .publish(&cx, &Manifest::empty(4, schema_id, 0))
            .await
            .expect("publish the reconciled successor")
            .loaded_manifest()
            .manifest
            .generation;
        assert_eq!(advanced, 4);
    });
    Ok(())
}

#[cfg(unix)]
#[test]
fn keeper_writer_adopts_exact_published_segment_and_preserves_retry_temp() -> TestResult {
    let directory = tempdir()?.keep();
    let assertion_directory = directory.clone();
    let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let encoded = encoded_test_segment(0xa11d, 0, 2, 1).map_err(|error| error.to_string())?;
        let canonical = directory.join(canonical_segment_name(encoded.header().segment_id));
        std::fs::write(&canonical, encoded.as_bytes()).map_err(|error| error.to_string())?;
        let pending = encoded
            .write_temp(&directory)
            .map_err(|error| error.to_string())?;
        let temp_path = pending.path().to_path_buf();

        let published = writer
            .publish_segment(&cx, pending)
            .await
            .map_err(|error| error.to_string())?;

        if published != canonical {
            return Err("reconciled segment returned the wrong canonical path".to_owned());
        }
        if std::fs::read(&canonical).map_err(|error| error.to_string())? != encoded.as_bytes() {
            return Err("reconciled canonical segment bytes changed".to_owned());
        }
        if std::fs::read(&temp_path).map_err(|error| error.to_string())? != encoded.as_bytes() {
            return Err("reconciled retry temp was not preserved exactly".to_owned());
        }
        Ok(())
    });
    outcome.map_err(io::Error::other)?;
    assert!(assertion_directory.is_dir());
    Ok(())
}

#[cfg(unix)]
#[test]
fn keeper_writer_rejects_differing_published_segment_without_overwrite() -> TestResult {
    let directory = tempdir()?.keep();
    let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let encoded = encoded_test_segment(0xc011, 0, 2, 1).map_err(|error| error.to_string())?;
        let canonical = directory.join(canonical_segment_name(encoded.header().segment_id));
        let mut differing = encoded.as_bytes().to_vec();
        let changed = differing
            .last_mut()
            .ok_or_else(|| "encoded segment fixture must not be empty".to_owned())?;
        *changed ^= 0x40;
        std::fs::write(&canonical, &differing).map_err(|error| error.to_string())?;
        let pending = encoded
            .write_temp(&directory)
            .map_err(|error| error.to_string())?;
        let temp_path = pending.path().to_path_buf();

        match writer.publish_segment(&cx, pending).await {
            Err(KeeperError::Io {
                operation: "reconcile published segment",
                source,
                ..
            }) if source.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(format!(
                    "unexpected segment reconciliation failure: {error}"
                ));
            }
            Ok(_) => return Err("differing canonical segment was adopted".to_owned()),
        }
        if std::fs::read(&canonical).map_err(|error| error.to_string())? != differing {
            return Err("differing canonical segment was overwritten".to_owned());
        }
        if std::fs::read(&temp_path).map_err(|error| error.to_string())? != encoded.as_bytes() {
            return Err("retry temp changed after canonical conflict".to_owned());
        }
        Ok(())
    });
    outcome.map_err(io::Error::other)?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn keeper_writer_reconciles_exact_installed_manifest_from_stale_snapshot() -> TestResult {
    let directory = tempdir()?.keep();
    let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let schema_id = DEFAULT_SCHEMA
            .schema_id()
            .map_err(|error| error.to_string())?;
        let proposed = Manifest::empty(2, schema_id, 0);
        ManifestPublisher::new(&directory)
            .publish(&cx, &proposed)
            .await
            .map_err(|error| error.to_string())?;
        if writer
            .retained_snapshot_for_bookkeeping()
            .loaded_manifest()
            .manifest
            .generation
            != 1
        {
            return Err("writer snapshot unexpectedly advanced before retry".to_owned());
        }

        writer
            .publish(&cx, &proposed)
            .await
            .map_err(|error| error.to_string())?;

        let installed = &writer
            .retained_snapshot_for_bookkeeping()
            .loaded_manifest()
            .manifest;
        if installed.generation != 2 {
            return Err("exact installed proposal was not reconciled".to_owned());
        }
        if !manifest_matches_proposal(installed, &proposed) {
            return Err("reconciled MANIFEST differs from the proposal".to_owned());
        }
        Ok(())
    });
    outcome.map_err(io::Error::other)?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn keeper_writer_rejects_differing_installed_manifest_from_stale_snapshot() -> TestResult {
    let directory = tempdir()?.keep();
    let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let schema_id = DEFAULT_SCHEMA
            .schema_id()
            .map_err(|error| error.to_string())?;
        let mut installed = Manifest::empty(2, schema_id, 0);
        installed.last_publish_unix_s = 1_700_000_000;
        let mut differing = installed.clone();
        differing.last_publish_unix_s = 1_700_000_001;
        ManifestPublisher::new(&directory)
            .publish(&cx, &installed)
            .await
            .map_err(|error| error.to_string())?;

        match writer.publish(&cx, &differing).await {
            Err(KeeperError::GenerationConflict {
                expected: 3,
                proposed: 2,
            }) => {}
            Err(error) => {
                return Err(format!("unexpected differing-proposal failure: {error}"));
            }
            Ok(_) => return Err("differing installed MANIFEST was reconciled".to_owned()),
        }
        if writer
            .retained_snapshot_for_bookkeeping()
            .loaded_manifest()
            .manifest
            .generation
            != 1
        {
            return Err("writer snapshot advanced after differing proposal".to_owned());
        }
        let on_disk = load_manifest_pair(&directory)
            .map_err(|error| error.to_string())?
            .manifest;
        if on_disk != installed {
            return Err("differing retry changed the installed MANIFEST".to_owned());
        }
        Ok(())
    });
    outcome.map_err(io::Error::other)?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn writer_preflights_missing_segment_before_manifest_publication() -> TestResult {
    let index = tempdir()?;
    let encoded = encoded_test_segment(0xdead, 0, 1, 1)?;
    let proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
    let directory = index.path().to_path_buf();
    let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let manifest_path = directory.join("MANIFEST");
        let before = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
        match writer.publish(&cx, &proposed).await {
            Err(KeeperError::SegmentOpen { .. }) => {}
            Err(error) => return Err(format!("unexpected publish failure: {error}")),
            Ok(_) => return Err("missing segment was published as authority".to_owned()),
        }
        let after = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
        if after != before {
            return Err("MANIFEST changed before segment preflight failed".to_owned());
        }
        if writer
            .retained_snapshot_for_bookkeeping()
            .loaded_manifest()
            .manifest
            .generation
            != 1
        {
            return Err("writer snapshot advanced after rejected publication".to_owned());
        }
        Ok(())
    });
    rejected.map_err(io::Error::other)?;
    assert_eq!(
        KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
            .loaded_manifest()
            .manifest
            .generation,
        1
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn writer_validates_manifest_before_segment_preflight() -> TestResult {
    let index = tempdir()?;
    let encoded = encoded_test_segment(0xbad, 0, 1, 1)?;
    let mut proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
    proposed.generation = 0;
    let directory = index.path().to_path_buf();
    let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        match writer.publish(&cx, &proposed).await {
            Err(KeeperError::InvalidManifest { .. }) => Ok(()),
            Err(error) => Err(format!("unexpected publish failure: {error}")),
            Ok(_) => Err("invalid manifest reached segment preflight".to_owned()),
        }
    });
    rejected.map_err(io::Error::other)?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn writer_preflights_corrupt_segment_before_manifest_publication() -> TestResult {
    let index = tempdir()?;
    let encoded = encoded_test_segment(0xcafe, 0, 1, 1)?;
    let section_offset = usize::try_from(encoded.section_entries()[0].offset)?;
    let proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
    let directory = index.path().to_path_buf();
    let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let pending = encoded
            .write_temp(&directory)
            .map_err(|error| error.to_string())?;
        let published = writer
            .publish_segment(&cx, pending)
            .await
            .map_err(|error| error.to_string())?;
        let mut bytes = std::fs::read(&published).map_err(|error| error.to_string())?;
        bytes[section_offset] ^= 0x80;
        std::fs::write(&published, bytes).map_err(|error| error.to_string())?;

        let manifest_path = directory.join("MANIFEST");
        let before = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
        match writer.publish(&cx, &proposed).await {
            Err(KeeperError::SegmentOpen { .. }) => {}
            Err(error) => return Err(format!("unexpected publish failure: {error}")),
            Ok(_) => return Err("corrupt segment was published as authority".to_owned()),
        }
        let after = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
        if after != before {
            return Err("MANIFEST changed before segment preflight failed".to_owned());
        }
        if writer
            .retained_snapshot_for_bookkeeping()
            .loaded_manifest()
            .manifest
            .generation
            != 1
        {
            return Err("writer snapshot advanced after rejected publication".to_owned());
        }
        Ok(())
    });
    rejected.map_err(io::Error::other)?;
    assert_eq!(
        KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
            .loaded_manifest()
            .manifest
            .generation,
        1
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn portable_retire_is_collision_safe_and_never_replaces() -> TestResult {
    let index = tempdir()?;
    let admission = acquire_writer_admission(index.path())?;
    let source = index.path().join("corrupt.bin");
    std::fs::write(&source, b"corrupt payload")?;
    let destination = index.path().join("quarantine.bin");

    // Absent destination: retires onto the base name.
    assert!(retire_regular_artifact_portable(
        &admission,
        &source,
        &destination
    )?);
    assert_eq!(std::fs::read(&destination)?, b"corrupt payload");
    assert!(!source.exists());

    // Occupied destination: the next retry claims a fresh suffixed probe
    // and the occupied quarantine keeps its bytes.
    std::fs::write(&source, b"second corrupt payload")?;
    assert!(retire_regular_artifact_portable(
        &admission,
        &source,
        &destination
    )?);
    assert_eq!(std::fs::read(&destination)?, b"corrupt payload");
    let siblings: Vec<_> = std::fs::read_dir(index.path())?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.starts_with("quarantine.bin."))
        .collect();
    assert_eq!(siblings.len(), 1, "expected one probed sibling quarantine");
    assert_eq!(
        std::fs::read(index.path().join(&siblings[0]))?,
        b"second corrupt payload"
    );

    // Absent source reports no work.
    assert!(!retire_regular_artifact_portable(
        &admission,
        &source,
        &destination
    )?);
    Ok(())
}

#[cfg(unix)]
#[test]
fn writer_retires_corrupt_primary_and_recovers_from_previous() -> TestResult {
    let index = tempdir()?;
    let directory = index.path().to_path_buf();
    let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
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
        Ok(())
    });
    setup.map_err(io::Error::other)?;
    std::fs::write(index.path().join("MANIFEST"), b"corrupt primary")?;
    let before_reader = directory_bytes(index.path())?;
    assert_eq!(
        KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
            .loaded_manifest()
            .manifest
            .generation,
        1
    );
    assert_eq!(directory_bytes(index.path())?, before_reader);

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
            1
        );
        let schema_id = DEFAULT_SCHEMA
            .schema_id()
            .map_err(|error| error.to_string())?;
        writer
            .publish(&cx, &Manifest::empty(2, schema_id, 0))
            .await
            .map_err(|error| error.to_string())?;
        Ok(())
    });
    recovered.map_err(io::Error::other)?;
    assert_eq!(
        KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
            .loaded_manifest()
            .manifest
            .generation,
        2
    );
    assert!(std::fs::read_dir(index.path())?.any(|entry| {
        entry
            .ok()
            .and_then(|entry| entry.file_name().into_string().ok())
            .is_some_and(|name| name.starts_with(".tmp-corrupt-manifest-"))
    }));
    Ok(())
}
