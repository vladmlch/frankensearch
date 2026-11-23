use super::*;

// ==== Blue-green CURRENT pointer (bd-quill-duel-blue-green-vwf7) ====

fn quill_v1_pointer() -> CurrentPointer {
    CurrentPointer::new(
        BlueGreenEngine::Quill,
        "quill-v1",
        crate::segment::FSLX_FORMAT_VERSION,
    )
    .expect("valid quill-v1 pointer")
}

#[test]
fn current_pointer_codec_roundtrips_both_engines_and_golden_bytes() {
    for (engine, dir_name, version) in [
        (
            BlueGreenEngine::Quill,
            "quill-v1",
            crate::segment::FSLX_FORMAT_VERSION,
        ),
        (
            BlueGreenEngine::Tantivy,
            "tantivy",
            TANTIVY_INDEX_FORMAT_VERSION,
        ),
        (BlueGreenEngine::Quill, "q", 0),
        (BlueGreenEngine::Quill, &"x".repeat(255), u32::MAX),
        (BlueGreenEngine::Tantivy, &"y".repeat(65_535), 7),
    ] {
        let pointer = CurrentPointer::new(engine, dir_name, version).expect("valid pointer");
        let encoded = pointer.encode();
        assert_eq!(encoded.len(), CURRENT_FIXED_BYTES + dir_name.len());
        let decoded = CurrentPointer::decode(&encoded).expect("roundtrip decode");
        assert_eq!(decoded, pointer);
    }

    // Golden image pins the exact registry §7.3 wire layout.
    let encoded = quill_v1_pointer().encode();
    let expected_body: &[u8] = b"FSLXCUR\0";
    assert_eq!(&encoded[..8], expected_body);
    assert_eq!(&encoded[8..12], &1_u32.to_le_bytes());
    assert_eq!(encoded[12], 1);
    assert_eq!(&encoded[13..15], &8_u16.to_le_bytes());
    assert_eq!(&encoded[15..23], b"quill-v1");
    assert_eq!(
        &encoded[23..27],
        &crate::segment::FSLX_FORMAT_VERSION.to_le_bytes()
    );
    let crc = u32::from_le_bytes(encoded[27..31].try_into().expect("crc bytes"));
    assert_eq!(crc, crc32fast::hash(&encoded[..27]));
    assert_eq!(encoded.len(), 31);
}

#[test]
fn current_pointer_decode_fails_closed_on_every_corruption_class() {
    let valid = quill_v1_pointer().encode();

    // Truncation at every byte boundary: typed error, never a panic.
    for cut in 0..valid.len() {
        assert!(
            CurrentPointer::decode(&valid[..cut]).is_err(),
            "truncation at {cut} must fail"
        );
    }
    // Trailing bytes.
    let mut trailed = valid.clone();
    trailed.push(0);
    assert!(matches!(
        CurrentPointer::decode(&trailed),
        Err(CurrentPointerError::CrcMismatch { .. } | CurrentPointerError::LengthMismatch { .. })
    ));
    // Single-byte flips across every byte class must fail (CRC or a typed
    // structural rejection), never silently decode or panic.
    for offset in 0..valid.len() {
        let mut flipped = valid.clone();
        flipped[offset] ^= 0x5A;
        assert!(
            CurrentPointer::decode(&flipped).is_err(),
            "flip at {offset} must fail"
        );
    }
    // Bad magic with a repaired CRC reports the magic, not the checksum.
    let mut bad_magic = valid.clone();
    bad_magic[..8].copy_from_slice(b"FSLXNOPE");
    rewrite_crc(&mut bad_magic);
    assert!(matches!(
        CurrentPointer::decode(&bad_magic),
        Err(CurrentPointerError::BadMagic)
    ));
    // Unsupported format version with a repaired CRC.
    let mut future = valid.clone();
    future[8..12].copy_from_slice(&2_u32.to_le_bytes());
    rewrite_crc(&mut future);
    assert!(matches!(
        CurrentPointer::decode(&future),
        Err(CurrentPointerError::UnsupportedFormatVersion(2))
    ));
    // Unknown engine kind with a repaired CRC.
    let mut alien = valid.clone();
    alien[12] = 9;
    rewrite_crc(&mut alien);
    assert!(matches!(
        CurrentPointer::decode(&alien),
        Err(CurrentPointerError::UnknownEngineKind(9))
    ));
    // Non-UTF-8 directory name with a repaired CRC.
    let mut non_utf8 = valid.clone();
    non_utf8[15] = 0xFF;
    rewrite_crc(&mut non_utf8);
    assert!(matches!(
        CurrentPointer::decode(&non_utf8),
        Err(CurrentPointerError::DirNameNotUtf8)
    ));
    // Path-unsafe names with repaired CRCs.
    for hostile in ["..", "a/b", "/abs", "a\\b", "a\0b", ""] {
        assert!(CurrentPointer::new(BlueGreenEngine::Quill, hostile, 1).is_err());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&CURRENT_MAGIC);
        put_u32(&mut bytes, CURRENT_FORMAT_VERSION);
        bytes.push(1);
        put_u16(
            &mut bytes,
            u16::try_from(hostile.len()).expect("hostile name fits u16"),
        );
        bytes.extend_from_slice(hostile.as_bytes());
        put_u32(&mut bytes, 1);
        let crc = crc32fast::hash(&bytes);
        put_u32(&mut bytes, crc);
        assert!(
            CurrentPointer::decode(&bytes).is_err(),
            "hostile name {hostile:?} must fail"
        );
    }
}

/// bd-8nqz.2: every layout state classifies without a single write. The
/// decisive case is the single-adoptable-child root: `resolve_current`
/// ADOPTS it (publishes CURRENT); `inspect_lexical_layout` must report it
/// and leave the disk byte-identical.
#[test]
fn inspect_lexical_layout_classifies_every_state_without_mutating() -> TestResult {
    // Missing root → Empty, and inspection must not create it.
    let missing = tempdir()?;
    let missing_root = missing.path().join("absent");
    assert_eq!(inspect_lexical_layout(&missing_root)?, LexicalLayout::Empty);
    assert!(
        !missing_root.exists(),
        "inspection must not create the root"
    );

    // Bare dir → Empty, no CURRENT appears.
    let bare = tempdir()?;
    assert_eq!(inspect_lexical_layout(bare.path())?, LexicalLayout::Empty);
    assert!(!bare.path().join(CURRENT_FILE_NAME).exists());

    // Direct Quill: MANIFEST at the root, no CURRENT.
    let direct_quill = tempdir()?;
    std::fs::write(direct_quill.path().join("MANIFEST"), b"m")?;
    assert_eq!(
        inspect_lexical_layout(direct_quill.path())?,
        LexicalLayout::DirectQuill
    );

    // Direct tantivy: meta.json at the root, no CURRENT.
    let direct_tantivy = tempdir()?;
    std::fs::write(direct_tantivy.path().join("meta.json"), b"{}")?;
    assert_eq!(
        inspect_lexical_layout(direct_tantivy.path())?,
        LexicalLayout::DirectTantivy
    );

    // Mixed: both markers at the root — never silently pick one.
    let mixed = tempdir()?;
    std::fs::write(mixed.path().join("MANIFEST"), b"m")?;
    std::fs::write(mixed.path().join("meta.json"), b"{}")?;
    assert!(matches!(
        inspect_lexical_layout(mixed.path())?,
        LexicalLayout::Mixed { .. }
    ));

    // Valid blue-green: published CURRENT naming a coherent engine dir.
    let published = tempdir()?;
    let quill_dir = published.path().join("quill-v1");
    std::fs::create_dir(&quill_dir)?;
    std::fs::write(quill_dir.join("MANIFEST"), b"m")?;
    publish_current(published.path(), &quill_v1_pointer())?;
    assert_eq!(
        inspect_lexical_layout(published.path())?,
        LexicalLayout::BlueGreen {
            pointer: quill_v1_pointer(),
            pointer_on_disk: true,
        }
    );

    // Mixed: CURRENT coexists with a direct root marker.
    let current_plus_direct = tempdir()?;
    let cpd_dir = current_plus_direct.path().join("quill-v1");
    std::fs::create_dir(&cpd_dir)?;
    std::fs::write(cpd_dir.join("MANIFEST"), b"m")?;
    publish_current(current_plus_direct.path(), &quill_v1_pointer())?;
    std::fs::write(current_plus_direct.path().join("MANIFEST"), b"squatter")?;
    assert!(matches!(
        inspect_lexical_layout(current_plus_direct.path())?,
        LexicalLayout::Mixed { .. }
    ));

    // THE decisive non-mutation case: single adoptable child, no CURRENT.
    let adoptable = tempdir()?;
    let child = adoptable.path().join("quill-v1");
    std::fs::create_dir(&child)?;
    std::fs::write(child.join("MANIFEST"), b"m")?;
    assert_eq!(
        inspect_lexical_layout(adoptable.path())?,
        LexicalLayout::BlueGreen {
            pointer: quill_v1_pointer(),
            pointer_on_disk: false,
        }
    );
    assert!(
        !adoptable.path().join(CURRENT_FILE_NAME).exists(),
        "inspection must never publish CURRENT; adoption belongs to resolve_current alone"
    );
    // And the same root still adopts through the mutating path afterwards.
    assert_eq!(
        resolve_current(adoptable.path())?,
        ResolvedCurrent::Adopted(quill_v1_pointer())
    );

    // Corrupt Quill: CURRENT names a dir that does not exist.
    let corrupt_missing = tempdir()?;
    let ghost = CurrentPointer::new(
        BlueGreenEngine::Quill,
        "quill-v2",
        crate::segment::FSLX_FORMAT_VERSION,
    )?;
    std::fs::write(
        corrupt_missing.path().join(CURRENT_FILE_NAME),
        ghost.encode(),
    )?;
    assert!(matches!(
        inspect_lexical_layout(corrupt_missing.path())?,
        LexicalLayout::CorruptQuill { .. }
    ));

    // Corrupt Quill: engine dir exists but lacks its MANIFEST marker.
    let corrupt_marker = tempdir()?;
    let bald = corrupt_marker.path().join("quill-v1");
    std::fs::create_dir(&bald)?;
    std::fs::write(
        corrupt_marker.path().join(CURRENT_FILE_NAME),
        quill_v1_pointer().encode(),
    )?;
    assert!(matches!(
        inspect_lexical_layout(corrupt_marker.path())?,
        LexicalLayout::CorruptQuill { .. }
    ));

    // Corrupt tantivy: CURRENT names a tantivy dir lacking meta.json.
    let corrupt_tantivy = tempdir()?;
    let tantivy_dir = corrupt_tantivy.path().join("tantivy");
    std::fs::create_dir(&tantivy_dir)?;
    let tantivy_pointer = CurrentPointer::new(
        BlueGreenEngine::Tantivy,
        "tantivy",
        TANTIVY_INDEX_FORMAT_VERSION,
    )?;
    std::fs::write(
        corrupt_tantivy.path().join(CURRENT_FILE_NAME),
        tantivy_pointer.encode(),
    )?;
    assert!(matches!(
        inspect_lexical_layout(corrupt_tantivy.path())?,
        LexicalLayout::CorruptTantivy { .. }
    ));

    // Ambiguous: CURRENT bytes do not decode.
    let undecodable = tempdir()?;
    std::fs::write(undecodable.path().join(CURRENT_FILE_NAME), b"garbage")?;
    assert!(matches!(
        inspect_lexical_layout(undecodable.path())?,
        LexicalLayout::Ambiguous { .. }
    ));

    // Ambiguous: multiple adoptable children, no CURRENT.
    let multiple = tempdir()?;
    for (name, marker) in [("quill-v1", "MANIFEST"), ("tantivy", "meta.json")] {
        let dir = multiple.path().join(name);
        std::fs::create_dir(&dir)?;
        std::fs::write(dir.join(marker), b"m")?;
    }
    assert!(matches!(
        inspect_lexical_layout(multiple.path())?,
        LexicalLayout::Ambiguous { .. }
    ));
    assert!(
        !multiple.path().join(CURRENT_FILE_NAME).exists(),
        "ambiguity must not trigger any tie-breaking write"
    );

    Ok(())
}

/// Symlink boundary: a symlinked CURRENT is rejected by the no-follow
/// open instead of being silently followed out of the root.
#[cfg(unix)]
#[test]
fn inspect_lexical_layout_rejects_symlinked_current() -> TestResult {
    let root = tempdir()?;
    std::fs::write(root.path().join("target"), b"elsewhere")?;
    std::os::unix::fs::symlink(
        root.path().join("target"),
        root.path().join(CURRENT_FILE_NAME),
    )?;
    assert!(
        inspect_lexical_layout(root.path()).is_err(),
        "no-follow open must refuse a symlinked CURRENT"
    );
    Ok(())
}

/// A syntactically safe CURRENT child name must still remain below the
/// lexical root. `Path::is_dir` follows symlinks, so this pins the
/// no-follow directory classification used by both inspection and the
/// mutating resolver.
#[cfg(unix)]
#[test]
fn current_selected_symlink_child_is_corrupt_and_never_resolved() -> TestResult {
    let root = tempdir()?;
    let outside = tempdir()?;
    std::fs::write(outside.path().join("MANIFEST"), b"outside-root")?;
    std::os::unix::fs::symlink(outside.path(), root.path().join("quill-v1"))?;
    std::fs::write(
        root.path().join(CURRENT_FILE_NAME),
        quill_v1_pointer().encode(),
    )?;

    assert!(matches!(
        inspect_lexical_layout(root.path())?,
        LexicalLayout::CorruptQuill { .. }
    ));
    assert!(matches!(
        resolve_current(root.path()),
        Err(CurrentPointerError::UnsafeEngineDir { .. })
    ));
    Ok(())
}

#[test]
fn publish_current_swaps_flip_and_rollback_leave_stale_temps_inert() -> TestResult {
    let root = tempdir()?;
    let quill_dir = root.path().join("quill-v1");
    let tantivy_dir = root.path().join("tantivy");
    std::fs::create_dir(&quill_dir)?;
    std::fs::create_dir(&tantivy_dir)?;

    // A stale crash witness from an interrupted earlier attempt is ignored
    // by resolution and never disturbed by publication.
    let stale_temp = root.path().join(".tmp-current-1-0");
    std::fs::write(&stale_temp, b"abandoned crash witness")?;

    let quill_pointer = quill_v1_pointer();
    publish_current(root.path(), &quill_pointer)?;
    assert_eq!(
        resolve_current(root.path())?,
        ResolvedCurrent::Pointer(quill_pointer.clone())
    );
    assert_eq!(std::fs::read(&stale_temp)?, b"abandoned crash witness");

    // Flip: quill -> tantivy in one pointer swap; old directory retained.
    let tantivy_pointer = CurrentPointer::new(
        BlueGreenEngine::Tantivy,
        "tantivy",
        TANTIVY_INDEX_FORMAT_VERSION,
    )?;
    publish_current(root.path(), &tantivy_pointer)?;
    assert_eq!(
        resolve_current(root.path())?,
        ResolvedCurrent::Pointer(tantivy_pointer.clone())
    );
    assert!(quill_dir.is_dir(), "retired quill directory is retained");

    // Rollback is the identical swap in reverse, zero rebuild.
    publish_current(root.path(), &quill_pointer)?;
    let resolved = resolve_current(root.path())?;
    assert_eq!(resolved, ResolvedCurrent::Pointer(quill_pointer.clone()));
    assert_eq!(
        resolved.pointer().expect("pointer").engine_dir(root.path()),
        quill_dir
    );
    assert!(
        tantivy_dir.is_dir(),
        "retired tantivy directory is retained"
    );
    Ok(())
}

#[test]
fn resolve_current_adopts_single_engine_dir_and_bootstraps_pointer() -> TestResult {
    // Quill directory adopted via its MANIFEST marker.
    let quill_root = tempdir()?;
    let quill_dir = quill_root.path().join("quill-v1");
    std::fs::create_dir(&quill_dir)?;
    std::fs::write(quill_dir.join("MANIFEST"), b"genesis placeholder")?;
    let adopted = resolve_current(quill_root.path())?;
    let expected = CurrentPointer::new(
        BlueGreenEngine::Quill,
        "quill-v1",
        crate::segment::FSLX_FORMAT_VERSION,
    )?;
    assert_eq!(adopted, ResolvedCurrent::Adopted(expected.clone()));
    // Adoption wrote a durable CURRENT; the next resolve reads it.
    assert_eq!(
        resolve_current(quill_root.path())?,
        ResolvedCurrent::Pointer(expected)
    );
    assert_eq!(
        CurrentPointer::decode(&std::fs::read(quill_root.path().join(CURRENT_FILE_NAME))?)?,
        quill_v1_pointer()
    );

    // Tantivy directory adopted via its meta.json marker.
    let tantivy_root = tempdir()?;
    let tantivy_dir = tantivy_root.path().join("tantivy");
    std::fs::create_dir(&tantivy_dir)?;
    std::fs::write(tantivy_dir.join("meta.json"), b"{}")?;
    assert_eq!(
        resolve_current(tantivy_root.path())?,
        ResolvedCurrent::Adopted(CurrentPointer::new(
            BlueGreenEngine::Tantivy,
            "tantivy",
            TANTIVY_INDEX_FORMAT_VERSION,
        )?)
    );

    // A directory bearing both markers adopts as quill (forward format wins).
    let mixed_root = tempdir()?;
    let mixed_dir = mixed_root.path().join("engine");
    std::fs::create_dir(&mixed_dir)?;
    std::fs::write(mixed_dir.join("MANIFEST"), b"x")?;
    std::fs::write(mixed_dir.join("meta.json"), b"{}")?;
    let resolved = resolve_current(mixed_root.path())?;
    assert_eq!(
        resolved.pointer().expect("adopted pointer").engine(),
        BlueGreenEngine::Quill
    );
    Ok(())
}

#[test]
fn resolve_current_reports_empty_ambiguous_and_missing_closed() -> TestResult {
    // Fresh root: Empty, and nothing is created.
    let empty_root = tempdir()?;
    assert_eq!(resolve_current(empty_root.path())?, ResolvedCurrent::Empty);
    assert!(std::fs::read_dir(empty_root.path())?.next().is_none());

    // Multiple engine directories and no CURRENT: fail closed for doctor.
    let ambiguous_root = tempdir()?;
    for name in ["quill-v1", "tantivy"] {
        let dir = ambiguous_root.path().join(name);
        std::fs::create_dir(&dir)?;
        std::fs::write(dir.join("MANIFEST"), b"x")?;
    }
    let error = resolve_current(ambiguous_root.path()).expect_err("must demand doctor");
    match error {
        CurrentPointerError::AmbiguousEngineDirs { candidates, .. } => {
            assert_eq!(
                candidates,
                vec!["quill-v1".to_owned(), "tantivy".to_owned()]
            );
        }
        other => panic!("expected AmbiguousEngineDirs, got {other:?}"),
    }
    assert!(
        !ambiguous_root.path().join(CURRENT_FILE_NAME).exists(),
        "ambiguous resolution must not publish a guess"
    );

    // CURRENT names a directory that does not exist: fail closed.
    let missing_root = tempdir()?;
    publish_current(
        missing_root.path(),
        &CurrentPointer::new(BlueGreenEngine::Quill, "quill-v9", 1)?,
    )?;
    assert!(matches!(
        resolve_current(missing_root.path()),
        Err(CurrentPointerError::MissingEngineDir { .. })
    ));

    // Corrupt CURRENT bytes: typed decode failure, never a guess.
    let corrupt_root = tempdir()?;
    std::fs::write(corrupt_root.path().join(CURRENT_FILE_NAME), b"garbage")?;
    assert!(resolve_current(corrupt_root.path()).is_err());
    Ok(())
}

#[test]
fn resolve_current_ignores_user_files_and_tmp_witnesses() -> TestResult {
    let root = tempdir()?;
    std::fs::write(root.path().join("notes.txt"), b"user data")?;
    std::fs::write(root.path().join(".tmp-current-9-3"), b"crash witness")?;
    let random_dir = root.path().join("random");
    std::fs::create_dir(&random_dir)?;
    std::fs::write(random_dir.join("data.bin"), b"not an engine")?;
    assert_eq!(resolve_current(root.path())?, ResolvedCurrent::Empty);
    // Nothing was adopted, published, or modified.
    assert!(!root.path().join(CURRENT_FILE_NAME).exists());
    assert_eq!(std::fs::read(root.path().join("notes.txt"))?, b"user data");
    assert_eq!(
        std::fs::read(root.path().join(".tmp-current-9-3"))?,
        b"crash witness"
    );
    Ok(())
}

// ==== Visibility contract (bd-quill-duel-visibility-contract-9rk3) ====

/// Build a v1 MANIFEST image by removing the v2 timestamp word from a v2
/// image and re-versioning/re-checksumming it.
fn downgrade_to_v1_image(v2: &[u8]) -> Vec<u8> {
    // v2 layout: magic(8) version(4) generation(8) watermark(8) schema(8)
    // engine(4) flags(4) timestamp(8) rest.. crc(4)
    assert_eq!(&v2[8..12], &MANIFEST_FORMAT_VERSION.to_le_bytes());
    let mut v1 = Vec::with_capacity(v2.len() - 8);
    v1.extend_from_slice(&v2[..8]);
    v1.extend_from_slice(&MANIFEST_FORMAT_VERSION_V1.to_le_bytes());
    v1.extend_from_slice(&v2[12..44]); // generation..flags
    v1.extend_from_slice(&v2[52..]); // segments/stats..crc
    let body_len = v1.len() - 4;
    let crc = crc32fast::hash(&v1[..body_len]);
    v1[body_len..].copy_from_slice(&crc.to_le_bytes());
    v1
}

#[test]
fn manifest_v1_images_read_with_unknown_timestamp_and_rewrite_as_v2() -> TestResult {
    let v2 = sample_manifest(5).to_bytes()?;
    let v1 = downgrade_to_v1_image(&v2);
    let decoded = Manifest::from_bytes(&v1)?;
    assert_eq!(decoded.last_publish_unix_s, 0, "v1 images carry no witness");
    let mut expected = sample_manifest(5);
    expected.last_publish_unix_s = 0;
    assert_eq!(decoded, expected);
    // Rewriting upgrades the image to v2 while preserving the zero field.
    let rewritten = decoded.to_bytes()?;
    assert_eq!(&rewritten[8..12], &MANIFEST_FORMAT_VERSION.to_le_bytes());
    assert_eq!(Manifest::from_bytes(&rewritten)?, decoded);
    // A v1 image with hostile trailing content still fails closed.
    let mut trailed = v1.clone();
    trailed.push(0);
    assert!(Manifest::from_bytes(&trailed).is_err());
    Ok(())
}

#[test]
fn publisher_stamps_zero_timestamp_and_preserves_explicit_witnesses() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        let mut first = sample_manifest(1);
        first.last_publish_unix_s = 0;
        let before = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("wall clock")
            .as_secs();
        let loaded = publisher
            .publish(&cx, &first)
            .await
            .expect("publish zero-timestamp proposal");
        let stamped = loaded.manifest.last_publish_unix_s;
        assert!(
            u64::try_from(stamped).expect("stamped witness is positive")
                >= before.saturating_sub(2),
            "zero witness is stamped with the publish wall clock"
        );

        // An explicit witness is preserved verbatim (deterministic fixtures).
        let mut second = sample_manifest(2);
        second.last_publish_unix_s = 1_700_000_042;
        let loaded = publisher
            .publish(&cx, &second)
            .await
            .expect("publish explicit-timestamp proposal");
        assert_eq!(loaded.manifest.last_publish_unix_s, 1_700_000_042);

        // Successors built from a loaded manifest keep the stamped chain.
        let mut third = loaded.manifest.clone();
        third.generation = 3;
        let loaded = publisher
            .publish(&cx, &third)
            .await
            .expect("publish successor built from loaded manifest");
        assert_eq!(loaded.manifest.last_publish_unix_s, 1_700_000_042);
    });
}

#[test]
fn segment_stats_surfaces_freshness_fields_and_live_writer() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let writer = KeeperWriter::create(&cx, directory.path(), DEFAULT_SCHEMA)
            .await
            .expect("create index");
        let writer_stats =
            SegmentStatsProvider::segment_stats(&writer).expect("read live-writer segment stats");
        assert_eq!(writer_stats.published_generation, 1);
        assert_eq!(writer_stats.sealed_segments, 0);
        assert_eq!(writer_stats.live_docs, 0);
        assert!(
            writer_stats.last_publish_unix.is_some(),
            "genesis is stamped"
        );
        assert!(
            writer_stats.live_writer,
            "writer holds LOCK by construction"
        );
        assert!(
            writer_stats.managed_disk_bytes > 0,
            "MANIFEST and LOCK are managed bytes"
        );
        assert_eq!(writer_stats.delta_segments, 0);

        // A concurrent read-only snapshot sees the live writer through the
        // D1 LOCK record (same pid, demonstrably alive).
        let snapshot =
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA).expect("open snapshot");
        let snapshot_stats = SegmentStatsProvider::segment_stats(&snapshot)
            .expect("read concurrent snapshot segment stats");
        assert_eq!(snapshot_stats.published_generation, 1);
        assert!(snapshot_stats.last_publish_unix.is_some());
        assert!(snapshot_stats.live_writer, "LOCK record names a live pid");

        // After the writer releases admission, the truncated LOCK record
        // reports no live writer.
        drop(writer);
        let snapshot =
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA).expect("reopen snapshot");
        let released = SegmentStatsProvider::segment_stats(&snapshot)
            .expect("read released snapshot segment stats");
        assert!(
            !released.live_writer,
            "released LOCK reports no live writer"
        );
    });
}

#[test]
fn visibility_barrier_truth_table_matches_the_contract() {
    let base = SegmentStats {
        last_publish_unix: Some(1_700_000_000),
        ..SegmentStats::default()
    };
    let one_second = Duration::from_millis(1_000);
    // No pending changes: never due, even with no witness at all.
    assert!(!base.visibility_barrier_due(one_second, false, 1_700_100_000));
    assert!(!SegmentStats::default().visibility_barrier_due(one_second, false, 1_700_100_000));
    // Pending changes with no durable witness: immediately due.
    assert!(SegmentStats::default().visibility_barrier_due(one_second, true, 1_700_100_000));
    // Within the bound: not due; at or beyond: due.
    assert!(!base.visibility_barrier_due(one_second, true, 1_700_000_000));
    assert!(base.visibility_barrier_due(one_second, true, 1_700_000_001));
    assert!(base.visibility_barrier_due(Duration::from_millis(500), true, 1_700_000_001));
    assert!(base.visibility_barrier_due(one_second, true, 1_700_000_100));
    // Clock skew reads as zero lag, never as a negative duration.
    assert!(!base.visibility_barrier_due(one_second, true, 1_699_999_000));
}

#[cfg(unix)]
#[test]
fn grace_expired_gc_removes_every_repair_orphan_without_touching_authority() -> TestResult {
    let index = tempdir()?;
    let schema_id = DEFAULT_SCHEMA.schema_id()?;
    let canonical = Manifest::empty(1, schema_id, 0).to_bytes()?;
    std::fs::write(index.path().join("MANIFEST"), &canonical)?;
    let orphans = [
        ".tmp-repair-MANIFEST-00000000000000aa",
        ".tmp-repair-MANIFEST-00000000000000aa.1",
        ".tmp-ready-repair-MANIFEST-00000000000000aa",
        ".tmp-corrupt-MANIFEST-00000000000000aa",
        ".tmp-corrupt-manifest-fec-00000000000000aa",
        ".tmp-corrupt-manifest-00000000000000aa",
    ];
    for (offset, name) in orphans.iter().enumerate() {
        std::fs::write(index.path().join(name), format!("orphan {offset}"))?;
    }
    let report = collect_writer_garbage_under_lock(
        index.path(),
        DEFAULT_SCHEMA,
        GarbageCollectionOptions {
            grace_period: Duration::ZERO,
        },
    )?;
    assert_eq!(
        report.removed.len(),
        orphans.len(),
        "every orphan class swept"
    );
    assert_eq!(
        std::fs::read(index.path().join("MANIFEST"))?,
        canonical,
        "canonical authority bytes are unchanged"
    );
    let second = collect_writer_garbage_under_lock(
        index.path(),
        DEFAULT_SCHEMA,
        GarbageCollectionOptions {
            grace_period: Duration::ZERO,
        },
    )?;
    assert!(second.removed.is_empty(), "second sweep is empty");
    Ok(())
}
