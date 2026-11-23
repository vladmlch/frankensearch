use super::*;

#[test]
fn deterministic_publish_crash_states_recover_read_only_and_gc_exactly() -> TestResult {
    const CASES: [&str; 10] = [
        "segment temp written",
        "segment temp fsynced",
        "segment renamed",
        "segment directory fsynced",
        "uncommitted sidecar emitted",
        "manifest temp written",
        "manifest temp fsynced",
        "between manifest renames",
        "new manifest current",
        "committed sidecar emitted",
    ];

    for (case_index, label) in CASES.into_iter().enumerate() {
        let directory = tempdir()?;
        let segment_a = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
        let segment_b_name = canonical_segment_name(0xb);
        let segment_b = if case_index >= 2 {
            Some(write_test_segment(directory.path(), 0xb, 2, 2, 4)?)
        } else {
            let encoded = encoded_test_segment(0xb, 2, 4, 1)?;
            let temp_path = directory.path().join(".tmp-segment-000000000000000b");
            std::fs::write(&temp_path, encoded.as_bytes())?;
            if case_index == 1 {
                File::open(&temp_path)?.sync_all()?;
            }
            None
        };

        let previous = durable_test_manifest(1, vec![segment_a.clone()]);
        if case_index == 7 {
            write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        } else if case_index >= 8 {
            let current = durable_test_manifest(
                2,
                vec![segment_a.clone(), segment_b.clone().expect("segment B")],
            );
            write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
            write_manifest(&directory.path().join("MANIFEST"), &current)?;
        } else {
            write_manifest(&directory.path().join("MANIFEST"), &previous)?;
        }

        if matches!(case_index, 5..=7) {
            let next = durable_test_manifest(
                2,
                vec![segment_a.clone(), segment_b.clone().expect("segment B")],
            );
            let temp_manifest = directory.path().join(".tmp-manifest-2");
            write_manifest(&temp_manifest, &next)?;
            if case_index == 6 {
                File::open(&temp_manifest)?.sync_all()?;
            }
        }
        #[cfg(not(windows))]
        if case_index == 3 {
            sync_directory(directory.path())?;
        }
        if case_index == 4 || case_index == 9 {
            std::fs::write(
                directory.path().join(format!("{segment_b_name}.fec")),
                b"sidecar",
            )?;
        }

        let before = directory_bytes(directory.path())?;
        let first = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
            .unwrap_or_else(|error| panic!("{label}: first recovery failed: {error}"));
        let second = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
            .unwrap_or_else(|error| panic!("{label}: second recovery failed: {error}"));
        assert_eq!(first.loaded_manifest(), second.loaded_manifest(), "{label}");
        assert_eq!(directory_bytes(directory.path())?, before, "{label}");
        let expected_committed = if case_index >= 8 { 2 } else { 1 };
        assert_eq!(first.segments().len(), expected_committed, "{label}");
        for segment in first.segments() {
            assert_eq!(
                segment.section(SectionKind::TERMDICT)?,
                Some(EMPTY_TERM_DICTIONARY.as_slice()),
                "{label}"
            );
        }

        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        let report = collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?;
        let mut expected_removed = match case_index {
            0 | 1 => vec![PathBuf::from(".tmp-segment-000000000000000b")],
            2 | 3 => vec![PathBuf::from(&segment_b_name)],
            4 => vec![
                PathBuf::from(&segment_b_name),
                PathBuf::from(format!("{segment_b_name}.fec")),
            ],
            5..=7 => vec![
                PathBuf::from(".tmp-manifest-2"),
                PathBuf::from(&segment_b_name),
            ],
            8 | 9 => Vec::new(),
            _ => unreachable!(),
        };
        expected_removed.sort_unstable();
        assert_eq!(report.removed, expected_removed, "{label}");

        let mut expected_inventory = before;
        expected_inventory.retain(|(name, _)| {
            !report
                .removed
                .iter()
                .any(|removed| removed.as_os_str() == name)
        });
        assert_eq!(
            directory_bytes(directory.path())?,
            expected_inventory,
            "{label}"
        );
        let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
            .unwrap_or_else(|error| panic!("{label}: post-GC recovery failed: {error}"));
        assert_eq!(
            reopened.loaded_manifest(),
            first.loaded_manifest(),
            "{label}"
        );
        assert_eq!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?,
            GarbageCollectionReport {
                removed: Vec::new()
            },
            "{label}"
        );
    }
    Ok(())
}

#[test]
fn writer_gc_unions_both_slots_honors_grace_and_is_idempotent() -> TestResult {
    let directory = tempdir()?;
    let segment_a = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
    let segment_b = write_test_segment(directory.path(), 0xb, 3, 2, 4)?;
    let segment_c = write_test_segment(directory.path(), 0xc, 2, 4, 6)?;
    let segment_e = write_test_segment(directory.path(), 0xe, 4, 6, 8)?;
    let segment_f = write_test_segment(directory.path(), 0x10, 5, 8, 10)?;
    let previous = durable_test_manifest(1, vec![segment_a.clone(), segment_c.clone()]);
    let mut current = durable_test_manifest(2, vec![segment_a, segment_b]);
    current.docid_high_watermark = previous.docid_high_watermark;
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    write_manifest(&directory.path().join("MANIFEST"), &current)?;

    std::fs::write(directory.path().join(".tmp-manifest-3"), b"uncommitted")?;
    std::fs::write(
        directory.path().join(format!(
            "{}.fec",
            canonical_segment_name(segment_e.segment_id)
        )),
        b"orphan sidecar",
    )?;
    let fresh_base = directory
        .path()
        .join(canonical_segment_name(segment_f.segment_id));
    let old_sidecar_with_fresh_base = directory.path().join(format!(
        "{}.fec",
        canonical_segment_name(segment_f.segment_id)
    ));
    std::fs::write(&old_sidecar_with_fresh_base, b"sidecar older than base")?;
    File::options().write(true).open(&fresh_base)?.set_times(
        std::fs::FileTimes::new().set_modified(
            SystemTime::now()
                .checked_add(Duration::from_secs(3_600))
                .expect("test clock remains representable"),
        ),
    )?;
    std::fs::write(
        directory.path().join(format!(
            "{}.fec",
            canonical_segment_name(segment_c.segment_id)
        )),
        b"previous sidecar",
    )?;
    std::fs::write(
        directory.path().join("MANIFEST.fec"),
        b"live manifest sidecar",
    )?;
    std::fs::write(directory.path().join("notes.fec"), b"user data")?;
    std::fs::write(directory.path().join(".quarantine"), b"operator data")?;
    let quarantine_name = format!("{}.quarantine", canonical_segment_name(0xf));
    std::fs::write(
        directory.path().join(&quarantine_name),
        b"quarantined operator data",
    )?;
    std::fs::write(
        directory.path().join("SEG-000000000000000f.fslx"),
        b"user data",
    )?;
    std::fs::create_dir(directory.path().join(".tmp-directory"))?;

    #[cfg(unix)]
    let outside_fixture = {
        use std::os::unix::ffi::OsStringExt;

        let outside = tempdir()?;
        let target = outside.path().join("outside-data");
        std::fs::write(&target, b"outside")?;
        std::os::unix::fs::symlink(&target, directory.path().join("seg-000000000000000f.fslx"))?;
        std::os::unix::fs::symlink(&target, directory.path().join(".tmp-link"))?;
        std::fs::write(
            directory
                .path()
                .join(OsString::from_vec(b".tmp-nonutf8-\xff".to_vec())),
            b"non-UTF-8 user data",
        )?;

        let before_open = directory_bytes(directory.path())?;
        let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(directory_bytes(directory.path())?, before_open);
        assert_eq!(snapshot.segments().len(), 2);
        assert_eq!(std::fs::read(&target)?, b"outside");
        (outside, target)
    };

    let fresh = collect_writer_garbage_under_lock(
        directory.path(),
        DEFAULT_SCHEMA,
        GarbageCollectionOptions::default(),
    )?;
    assert!(
        fresh.is_empty(),
        "fresh garbage must survive the grace window"
    );
    assert!(
        directory
            .path()
            .join(canonical_segment_name(segment_e.segment_id))
            .exists()
    );

    std::fs::write(directory.path().join("gen-2.claim"), [])?;
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .expect("test clock remains representable");
    let report = collect_writer_garbage_at(
        directory.path(),
        DEFAULT_SCHEMA,
        GarbageCollectionOptions::default(),
        now,
    )?;
    let orphan_name = canonical_segment_name(segment_e.segment_id);
    assert_eq!(
        report.removed,
        vec![
            PathBuf::from(".tmp-manifest-3"),
            PathBuf::from("gen-2.claim"),
            PathBuf::from(&orphan_name),
            PathBuf::from(format!("{orphan_name}.fec")),
        ]
    );
    assert!(directory.path().join(canonical_segment_name(0xc)).exists());
    assert!(fresh_base.exists());
    assert!(old_sidecar_with_fresh_base.exists());
    assert!(
        directory
            .path()
            .join(format!("{}.fec", canonical_segment_name(0xc)))
            .exists()
    );
    assert!(directory.path().join("notes.fec").exists());
    assert!(directory.path().join(".quarantine").exists());
    assert!(directory.path().join(&quarantine_name).exists());
    assert!(directory.path().join("SEG-000000000000000f.fslx").exists());
    assert!(directory.path().join(".tmp-directory").is_dir());
    #[cfg(unix)]
    {
        assert!(directory.path().join(".tmp-link").is_symlink());
        assert!(
            directory
                .path()
                .join("seg-000000000000000f.fslx")
                .is_symlink()
        );
        use std::os::unix::ffi::OsStringExt;
        assert!(
            directory
                .path()
                .join(OsString::from_vec(b".tmp-nonutf8-\xff".to_vec()))
                .exists()
        );
        assert_eq!(std::fs::read(&outside_fixture.1)?, b"outside");
    }
    assert!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?
        .is_empty()
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn old_segment_gets_a_full_grace_window_after_becoming_unreachable() -> TestResult {
    let directory = tempdir()?;
    let unreachable = write_test_segment(directory.path(), 0xdead, 1, 0, 2)?;
    let unreachable_path = directory
        .path()
        .join(canonical_segment_name(unreachable.segment_id));
    File::options()
        .write(true)
        .open(&unreachable_path)?
        .set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;

    // The segment was reachable before generation 2 and has only just
    // fallen out of the two-slot union at generation 3. Its ancient file
    // mtime must not consume the grace period that starts at that
    // unreachability transition. Generation 3 can have superseded a
    // published manifest, so this models a real retirement: the receipt
    // the generation-3 publication would have stamped is part of the
    // fixture, and without it the sweep correctly refuses to act (see
    // gc_refuses_a_backdated_unreferenced_segment_without_a_retirement_receipt).
    // The receipt carries whole seconds, so date the fixture's transition
    // at a whole second: the sweep measures grace from exactly this
    // instant.
    let retired_at = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs(),
        ))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    write_test_retirement_receipt(directory.path(), unreachable.segment_id, 3, retired_at)?;
    write_manifest(
        &directory.path().join("MANIFEST.prev"),
        &durable_test_manifest(2, Vec::new()),
    )?;
    write_manifest(
        &directory.path().join("MANIFEST"),
        &durable_test_manifest(3, Vec::new()),
    )?;
    let unreachable_since = retired_at;
    let options = GarbageCollectionOptions {
        grace_period: Duration::from_secs(60),
    };

    let before_grace = unreachable_since
        .checked_add(Duration::from_secs(59))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert!(
        collect_writer_garbage_at(directory.path(), DEFAULT_SCHEMA, options, before_grace,)?
            .is_empty(),
        "an old segment must survive until it has been unreachable for the full grace period"
    );
    assert!(unreachable_path.exists());

    let after_grace = unreachable_since
        .checked_add(Duration::from_secs(60))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert_eq!(
        collect_writer_garbage_at(directory.path(), DEFAULT_SCHEMA, options, after_grace,)?.removed,
        vec![
            PathBuf::from(canonical_segment_name(unreachable.segment_id)),
            PathBuf::from(retirement_receipt_name(unreachable.segment_id)),
        ]
    );
    assert!(!unreachable_path.exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn recovery_gc_floor_is_stable_across_repeated_attempts() -> TestResult {
    let directory = tempdir()?;
    let unreachable = write_test_segment(directory.path(), 0xbeef, 1, 0, 2)?;
    let segment_name = canonical_segment_name(unreachable.segment_id);
    let segment_path = directory.path().join(&segment_name);
    File::options()
        .write(true)
        .open(&segment_path)?
        .set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;

    write_manifest(
        &directory.path().join("MANIFEST.prev"),
        &durable_test_manifest(1, Vec::new()),
    )?;
    write_manifest(
        &directory.path().join(".tmp-manifest-2"),
        &durable_test_manifest(2, vec![unreachable]),
    )?;

    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(
        snapshot.loaded_manifest().source,
        ManifestSource::PreviousAfterMissingCurrent
    );
    let directory_file = open_gc_directory(directory.path())?;
    let first_observation = SystemTime::now();
    let first_floor = segment_unreachability_floor_at(
        &directory_file,
        directory.path(),
        &snapshot,
        first_observation,
    )?
    .ok_or_else(|| io::Error::other("recovery must supply a GC floor"))?;
    let later_observation = first_observation
        .checked_add(Duration::from_secs(3_600))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    let second_floor = segment_unreachability_floor_at(
        &directory_file,
        directory.path(),
        &snapshot,
        later_observation,
    )?
    .ok_or_else(|| io::Error::other("recovery must supply a stable GC floor"))?;
    assert_eq!(
        second_floor, first_floor,
        "a later observation must not restart the recovery grace window"
    );
    // `std::fs::write` stamps the directory (the floor's witness) when it
    // creates the temp manifest and the temp's own mtime when it writes;
    // a clock tick between the two leaves the temp younger than the floor
    // (bd-yzyvr). Temporaries age from their own mtime, so pin it to the
    // floor to make both grace boundaries below exact.
    File::options()
        .write(true)
        .open(directory.path().join(".tmp-manifest-2"))?
        .set_times(std::fs::FileTimes::new().set_modified(first_floor))?;

    let options = GarbageCollectionOptions {
        grace_period: Duration::from_secs(60),
    };
    let before_grace = first_floor
        .checked_add(Duration::from_secs(59))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert!(
        collect_writer_garbage_at(directory.path(), DEFAULT_SCHEMA, options, before_grace)?
            .is_empty(),
        "the first recovery attempt must preserve the segment for the full grace"
    );
    assert!(segment_path.exists());

    let after_grace = first_floor
        .checked_add(options.grace_period)
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    let mut expected_removed = vec![
        PathBuf::from(".tmp-manifest-2"),
        PathBuf::from(segment_name),
    ];
    expected_removed.sort_unstable();
    assert_eq!(
        collect_writer_garbage_at(directory.path(), DEFAULT_SCHEMA, options, after_grace)?.removed,
        expected_removed,
        "the second recovery attempt must not reset the grace to its call time"
    );
    assert!(!segment_path.exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn corrupt_current_blocks_gc_until_valid_current_is_republished() -> TestResult {
    use rustix::fs::{AtFlags, statat};

    let directory = tempdir()?;
    let unreachable = write_test_segment(directory.path(), 0x00c0_ffee, 1, 0, 2)?;
    let segment_name = canonical_segment_name(unreachable.segment_id);
    let segment_path = directory.path().join(&segment_name);
    File::options()
        .write(true)
        .open(&segment_path)?
        .set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;

    write_manifest(
        &directory.path().join("MANIFEST.prev"),
        &durable_test_manifest(1, Vec::new()),
    )?;
    let current_path = directory.path().join("MANIFEST");
    std::fs::write(&current_path, b"old latent corruption")?;
    File::options()
        .write(true)
        .open(&current_path)?
        .set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;

    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(
        snapshot.loaded_manifest().source,
        ManifestSource::PreviousAfterCorruptCurrent
    );
    let directory_file = open_gc_directory(directory.path())?;
    let current_stat = statat(&directory_file, "MANIFEST", AtFlags::SYMLINK_NOFOLLOW)?;
    let stale_inode_time = required_gc_witness_time(
        &current_stat,
        &current_path,
        "decode test corrupt-current timestamp",
    )?;
    let first_observation = stale_inode_time
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    let second_observation = first_observation
        .checked_add(Duration::from_secs(3_600))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    let before = directory_bytes(directory.path())?;

    assert!(matches!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            first_observation,
        ),
        Err(KeeperError::RecoveryRequired { .. })
    ));
    assert_eq!(directory_bytes(directory.path())?, before);
    assert!(segment_path.exists());

    assert!(matches!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            second_observation,
        ),
        Err(KeeperError::RecoveryRequired { .. })
    ));
    assert_eq!(directory_bytes(directory.path())?, before);
    assert!(segment_path.exists());

    write_manifest(&current_path, &durable_test_manifest(2, Vec::new()))?;
    File::open(&current_path)?.sync_all()?;
    #[cfg(not(windows))]
    sync_directory(directory.path())?;
    let recovered = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(recovered.loaded_manifest().source, ManifestSource::Current);
    let directory_file = open_gc_directory(directory.path())?;
    let republish_floor = segment_unreachability_floor_at(
        &directory_file,
        directory.path(),
        &recovered,
        second_observation,
    )?
    .ok_or_else(|| io::Error::other("valid republish supplies a GC floor"))?;
    let after_republish_grace = republish_floor
        .checked_add(DEFAULT_GARBAGE_GRACE)
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert_eq!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            after_republish_grace,
        )?
        .removed,
        vec![PathBuf::from(segment_name)]
    );
    assert!(!segment_path.exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn later_publications_do_not_postpone_a_receipted_retirement() -> TestResult {
    fn valid_manifest(path: &Path) -> Result<Manifest, String> {
        match read_manifest_slot(path).map_err(|error| error.to_string())? {
            ManifestSlot::Valid(manifest) => Ok(manifest),
            ManifestSlot::Missing => Err(format!("{} is missing", path.display())),
            ManifestSlot::Invalid(error) => Err(format!("{} is invalid: {error}", path.display())),
        }
    }

    let index = tempdir()?;
    let segment = write_test_segment(index.path(), 0xd00d, 1, 0, 2)?;
    let segment_id = segment.segment_id;
    let segment_path = index.path().join(canonical_segment_name(segment_id));
    File::options()
        .write(true)
        .open(&segment_path)?
        .set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;
    write_manifest(
        &index.path().join("MANIFEST"),
        &durable_test_manifest(1, vec![segment]),
    )?;

    let directory = index.path().to_path_buf();
    let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        if !writer
            .retained_snapshot_for_bookkeeping()
            .loaded_manifest()
            .manifest
            .segments
            .iter()
            .any(|segment| segment.segment_id == segment_id)
        {
            return Err("generation 1 does not reference the segment".to_owned());
        }

        let mut publish_two = writer
            .retained_snapshot_for_bookkeeping()
            .next_manifest()
            .map_err(|error| error.to_string())?;
        writer
            .retained_snapshot_for_bookkeeping()
            .delete_all(&mut publish_two)
            .map_err(|error| error.to_string())?;
        publish_two.last_publish_unix_s = 0;
        writer
            .publish(&cx, &publish_two)
            .await
            .map_err(|error| error.to_string())?;

        let current = valid_manifest(&directory.join("MANIFEST"))?;
        let previous = valid_manifest(&directory.join("MANIFEST.prev"))?;
        if current.generation != 2
            || current
                .segments
                .iter()
                .any(|segment| segment.segment_id == segment_id)
            || previous.generation != 1
            || !previous
                .segments
                .iter()
                .any(|segment| segment.segment_id == segment_id)
        {
            return Err(
                "generation 2 must retain the segment only through MANIFEST.prev".to_owned(),
            );
        }
        let options = GarbageCollectionOptions {
            grace_period: Duration::from_secs(60),
        };
        let while_previous_is_live = SystemTime::now()
            .checked_add(Duration::from_secs(120))
            .ok_or_else(|| "test clock remains representable".to_owned())?;
        if !collect_writer_garbage_at(&directory, DEFAULT_SCHEMA, options, while_previous_is_live)
            .map_err(|error| error.to_string())?
            .is_empty()
            || !segment_path.exists()
        {
            return Err("MANIFEST.prev reachability did not protect the segment".to_owned());
        }

        let mut publish_three = writer
            .retained_snapshot_for_bookkeeping()
            .next_manifest()
            .map_err(|error| error.to_string())?;
        publish_three.last_publish_unix_s = 0;
        writer
            .publish(&cx, &publish_three)
            .await
            .map_err(|error| error.to_string())?;
        let current = valid_manifest(&directory.join("MANIFEST"))?;
        let previous = valid_manifest(&directory.join("MANIFEST.prev"))?;
        if current.generation != 3
            || previous.generation != 2
            || current
                .segments
                .iter()
                .chain(&previous.segments)
                .any(|segment| segment.segment_id == segment_id)
        {
            return Err("generation 3 did not make the segment unreachable".to_owned());
        }

        // The generation-3 publication is the one that retired the
        // segment, so its receipt is the durable witness of the
        // transition and the clock the grace period runs on.
        let receipt_bytes = std::fs::read(directory.join(retirement_receipt_name(segment_id)))
            .map_err(|error| error.to_string())?;
        let receipt = SegmentRetirementReceipt::decode(&receipt_bytes)
            .ok_or_else(|| "generation 3 did not stamp a decodable receipt".to_owned())?;
        if receipt.retired_generation != 3
            || receipt.witness != UnreachabilityWitness::RetiredByPublication
        {
            return Err(format!("unexpected retirement receipt: {receipt:?}"));
        }
        let retired_at = receipt
            .retired_at()
            .ok_or_else(|| "receipt instant is representable".to_owned())?;
        let before_first_grace = retired_at
            .checked_add(Duration::from_secs(59))
            .ok_or_else(|| "test clock remains representable".to_owned())?;
        if !collect_writer_garbage_at(&directory, DEFAULT_SCHEMA, options, before_first_grace)
            .map_err(|error| error.to_string())?
            .is_empty()
            || !segment_path.exists()
        {
            return Err("newly unreachable segment did not receive its full grace".to_owned());
        }

        // Two more back-to-back publications, the shape of an embedder
        // that runs incremental passes more often than the grace period
        // (cass#453). Each advances the manifest-level floor, but neither
        // references the retired segment, so neither may restart its
        // grace.
        let directory_file = open_gc_directory(&directory).map_err(|error| error.to_string())?;
        let third_publish_floor = segment_unreachability_floor_at(
            &directory_file,
            &directory,
            writer.retained_snapshot_for_bookkeeping(),
            SystemTime::now(),
        )
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "generation 3 did not supply an unreachability floor".to_owned())?;
        let mut latest_floor = third_publish_floor;
        for generation in 4..=5_u64 {
            std::thread::sleep(Duration::from_millis(10));
            let mut publish = writer
                .retained_snapshot_for_bookkeeping()
                .next_manifest()
                .map_err(|error| error.to_string())?;
            publish.last_publish_unix_s = 0;
            writer
                .publish(&cx, &publish)
                .await
                .map_err(|error| error.to_string())?;
            let directory_file =
                open_gc_directory(&directory).map_err(|error| error.to_string())?;
            let floor = segment_unreachability_floor_at(
                &directory_file,
                &directory,
                writer.retained_snapshot_for_bookkeeping(),
                SystemTime::now(),
            )
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                format!("generation {generation} did not supply an unreachability floor")
            })?;
            if floor <= latest_floor {
                return Err(format!(
                    "generation {generation} did not advance the manifest-level floor"
                ));
            }
            latest_floor = floor;
        }
        let receipt_after = std::fs::read(directory.join(retirement_receipt_name(segment_id)))
            .map_err(|error| error.to_string())?;
        if receipt_after != receipt_bytes {
            return Err("later publications rewrote a receipt they did not stamp".to_owned());
        }

        // The receipt's own deadline is strictly earlier than the deadline
        // the reset floor would impose, which is what makes this assertion
        // discriminate: under a floor-reset rule the segment would survive
        // here.
        let receipt_grace_deadline = retired_at
            .checked_add(options.grace_period)
            .ok_or_else(|| "test clock remains representable".to_owned())?;
        let floor_grace_deadline = latest_floor
            .checked_add(options.grace_period)
            .ok_or_else(|| "test clock remains representable".to_owned())?;
        if receipt_grace_deadline >= floor_grace_deadline {
            return Err("fixture did not separate the receipt clock from the floor".to_owned());
        }
        let report =
            collect_writer_garbage_at(&directory, DEFAULT_SCHEMA, options, receipt_grace_deadline)
                .map_err(|error| error.to_string())?;
        // The receipt authorized this deletion, so it is reclaimed with
        // the segment it authorized.
        if report.removed
            != vec![
                PathBuf::from(canonical_segment_name(segment_id)),
                PathBuf::from(retirement_receipt_name(segment_id)),
            ]
            || segment_path.exists()
        {
            return Err(format!(
                "segment was not reclaimed at its receipt's grace deadline: {:?}",
                report.removed
            ));
        }

        // Every segment the later publications still reference is intact
        // and the writer's own view reopens cleanly after the sweep.
        KeeperSnapshot::open(&directory, DEFAULT_SCHEMA).map_err(|error| error.to_string())?;
        Ok(())
    });
    outcome.map_err(io::Error::other)?;
    Ok(())
}

#[test]
fn gc_refuses_to_delete_a_newly_unreachable_segment_without_provenance() -> TestResult {
    let directory = tempdir()?;
    let live = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
    let newly_unreachable = write_test_segment(directory.path(), 0xb, 2, 2, 4)?;
    let previous = durable_test_manifest(2, vec![live.clone()]);
    let mut current = durable_test_manifest(3, vec![live]);
    current.docid_high_watermark = previous.docid_high_watermark;
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    write_manifest(&directory.path().join("MANIFEST"), &current)?;

    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .expect("test clock remains representable");
    let report = collect_writer_garbage_at(
        directory.path(),
        DEFAULT_SCHEMA,
        GarbageCollectionOptions::default(),
        now,
    )?;

    assert!(report.is_empty());
    assert!(
        directory
            .path()
            .join(canonical_segment_name(newly_unreachable.segment_id))
            .exists(),
        "a missing first-unreachable receipt must fail closed rather than infer age from mtime"
    );
    Ok(())
}
