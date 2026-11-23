use super::*;

#[test]
fn empty_manifest_has_stable_wire_golden() -> TestResult {
    let manifest = Manifest::empty(1, 0x1122_3344_5566_7788, 0);
    let bytes = manifest.to_bytes()?;
    // GOLDEN-CHANGE: the 0.4.0 package release (the public API breaks:
    // the CJK analyzer variant and the docid lease reports) advances
    // producer metadata, not the wire format. Bytes 36..40 are
    // `CURRENT_ENGINE_VERSION` (0.4.0 => `00 00 04 00`); the trailing CRC32
    // (zlib, little-endian) covers all prior bytes, the method that
    // reproduces every retained image below. The 0.3.5 image, written by
    // main-built binaries and never published, joins the still-readable
    // list.
    let expected = hex_bytes(
        "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
         00000400000000000000000000000000000000000000000020d1b7d3",
    );
    assert_eq!(bytes, expected);
    assert_eq!(Manifest::from_bytes(&bytes)?, manifest);

    // Keep previous producers' exact wire images readable and writable;
    // only their engine-version word and the covering CRC differ.
    for (version, wire_hex) in [
        (
            pack_engine_version(0, 3, 5),
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             05000300000000000000000000000000000000000000000018144d5e",
        ),
        (
            pack_engine_version(0, 3, 4),
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             0400030000000000000000000000000000000000000000003f7168df",
        ),
        (
            pack_engine_version(0, 3, 3),
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             030003000000000000000000000000000000000000000000484670ee",
        ),
        (
            pack_engine_version(0, 3, 2),
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             0200030000000000000000000000000000000000000000006f23556f",
        ),
        (
            pack_engine_version(0, 3, 1),
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             010003000000000000000000000000000000000000000000478a4b37",
        ),
        (
            pack_engine_version(0, 3, 0),
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             00000300000000000000000000000000000000000000000060ef6eb6",
        ),
        (
            pack_engine_version(0, 2, 4),
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             040002000000000000000000000000000000000000000000ff1540c8",
        ),
        (
            pack_engine_version(0, 2, 3),
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             030002000000000000000000000000000000000000000000882258f9",
        ),
    ] {
        let previous_bytes = hex_bytes(wire_hex);
        let mut previous = Manifest::from_bytes(&previous_bytes)?;
        assert_eq!(previous.engine_version, version);
        assert_eq!(previous.to_bytes()?, previous_bytes);
        previous.engine_version = CURRENT_ENGINE_VERSION;
        assert_eq!(previous, manifest);
    }
    Ok(())
}

#[test]
fn zero_schema_hash_is_a_valid_roundtrip() -> TestResult {
    let manifest = Manifest::empty(1, 0, 0);
    assert_eq!(Manifest::from_bytes(&manifest.to_bytes()?)?, manifest);
    Ok(())
}

#[test]
fn nonempty_manifest_has_stable_wire_golden() -> TestResult {
    let manifest = Manifest {
        generation: 0x0102_0304_0506_0708,
        docid_high_watermark: 20,
        schema_id: 0x1122_3344_5566_7788,
        engine_version: 0x0102_0304,
        flags: MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
        last_publish_unix_s: 1_700_000_000,
        segments: vec![ManifestSegment {
            segment_id: 0x1020_3040_5060_7080,
            seal_seq: 0x1112_1314_1516_1718,
            file_len: 0x2122_2324_2526_2728,
            file_xxh3: 0x3132_3334_3536_3738,
            docid_lo: 10,
            docid_hi: 20,
            doc_count: 2,
            tombstones: array_tombstones(0, &[12, 18]),
        }],
        field_stats: vec![ManifestFieldStats {
            field_ord: 3,
            total_tokens: 0x4142_4344_4546_4748,
            doc_count: 2,
        }],
    };
    let bytes = manifest.to_bytes()?;
    // v2 golden: version 2, `last_publish_unix_s` (1_700_000_000) after
    // flags, then the v1 segment/tombstone/stats layout unchanged.
    let expected = hex_bytes(
        "46534c584d414e0002000000080706050403020114000000000000008877665544332211\
         040302010100000000f15365000000000100000080706050403020101817161514131211\
         282726252423222138373635343332310a00000000000000140000000000000002000000\
         0100000000000002000c001200010000000300484746454443424102000000122be130",
    );
    assert_eq!(
        bytes, expected,
        "nonempty wire golden pins segment, tombstone, and stats field order"
    );
    assert_eq!(Manifest::from_bytes(&bytes)?, manifest);
    Ok(())
}

#[test]
fn nonempty_manifest_roundtrips_exactly() -> TestResult {
    let manifest = sample_manifest(9);
    let bytes = manifest.to_bytes()?;
    assert_eq!(Manifest::from_bytes(&bytes)?, manifest);
    Ok(())
}

#[test]
fn every_truncation_and_checksum_flip_is_rejected_without_panic() -> TestResult {
    let bytes = sample_manifest(3).to_bytes()?;
    for length in 0..bytes.len() {
        let parsed = std::panic::catch_unwind(|| Manifest::from_bytes(&bytes[..length]));
        assert!(parsed.is_ok(), "parser panicked at truncation {length}");
        assert!(parsed.expect("catch_unwind result").is_err());
    }
    let body = &bytes[..bytes.len() - 4];
    for length in 0..body.len() {
        let mut valid_crc_prefix = body[..length].to_vec();
        put_u32(&mut valid_crc_prefix, crc32fast::hash(&body[..length]));
        let parsed = std::panic::catch_unwind(|| Manifest::from_bytes(&valid_crc_prefix));
        assert!(
            parsed.is_ok(),
            "parser panicked at valid-CRC body truncation {length}"
        );
        assert!(parsed.expect("catch_unwind result").is_err());
    }
    for index in [0, 8, 44, bytes.len() / 2, bytes.len() - 1] {
        let mut corrupt = bytes.clone();
        corrupt[index] ^= 0x80;
        assert!(matches!(
            Manifest::from_bytes(&corrupt),
            Err(ManifestCodecError::ChecksumMismatch { .. })
        ));
    }
    Ok(())
}

#[test]
fn valid_crc_cannot_hide_hostile_counts_or_unknown_flags() -> TestResult {
    let mut bytes = Manifest::empty(1, 7, 0).to_bytes()?;
    bytes[40..44].copy_from_slice(&0x8000_0000_u32.to_le_bytes());
    rewrite_crc(&mut bytes);
    assert!(matches!(
        Manifest::from_bytes(&bytes),
        Err(ManifestCodecError::NonCanonical { .. })
    ));

    let mut bytes = Manifest::empty(1, 7, 0).to_bytes()?;
    let hostile_segment_count =
        u32::try_from(MAX_MANIFEST_SEGMENTS + 1).expect("segment cap fits u32");
    bytes[52..56].copy_from_slice(&hostile_segment_count.to_le_bytes());
    rewrite_crc(&mut bytes);
    assert!(matches!(
        Manifest::from_bytes(&bytes),
        Err(ManifestCodecError::ResourceLimit {
            resource: "segment count",
            ..
        })
    ));
    Ok(())
}

#[test]
fn q1_validation_rejects_overlap_duplicates_and_watermark_rollback() {
    let mut manifest = sample_manifest(1);
    manifest.segments[1].docid_lo = 150;
    assert!(manifest.validate().is_err());

    let mut manifest = sample_manifest(1);
    manifest.segments[1].segment_id = manifest.segments[0].segment_id;
    assert!(manifest.validate().is_err());

    let mut manifest = sample_manifest(1);
    manifest.segments[1].seal_seq = manifest.segments[0].seal_seq;
    assert!(manifest.validate().is_err());

    let mut manifest = sample_manifest(1);
    manifest.docid_high_watermark = manifest.segments[1].docid_hi - 1;
    assert!(manifest.validate().is_err());
}

#[test]
fn tombstones_must_be_canonical_and_inside_the_segment_range() {
    let mut manifest = sample_manifest(1);
    manifest.segments[0].tombstones = array_tombstones(0, &[99]);
    assert!(manifest.validate().is_err());

    assert!(TombstoneSet::from_bytes(&array_tombstone_bytes(0, &[150, 150])).is_err());

    let mut bitmap = vec![1, 0, 0, 0, 0, 0, 1, 1, 0];
    bitmap.extend_from_slice(&[0; 8_192]);
    bitmap[9] = 1;
    assert!(TombstoneSet::from_bytes(&bitmap).is_err());
}

#[test]
fn tombstone_container_thresholds_and_chunk_count_are_bounded() {
    let oversized_lows = (0..=TOMBSTONE_ARRAY_MAX_CARDINALITY).collect::<Vec<_>>();
    assert!(matches!(
        validate_tombstone_bytes(&array_tombstone_bytes(0, &oversized_lows), None),
        Err(ManifestCodecError::NonCanonical { .. })
    ));

    assert!(matches!(
        validate_tombstone_bytes(&bitmap_tombstone_bytes(0, &[7]), None),
        Err(ManifestCodecError::NonCanonical { .. })
    ));

    let dense_lows = (0..u16::try_from(TOMBSTONE_BITMAP_MIN_CARDINALITY)
        .expect("bitmap threshold fits u16"))
        .collect::<Vec<_>>();
    let dense = bitmap_tombstones(0, &dense_lows);
    assert_eq!(
        validate_tombstone_bytes(dense.as_bytes(), Some((0, 65_536)))
            .expect("dense bitmap is canonical"),
        TOMBSTONE_BITMAP_MIN_CARDINALITY
    );
    assert!(validate_tombstone_bytes(dense.as_bytes(), Some((100, 65_536))).is_err());

    let mut too_many_chunks = Vec::new();
    put_u32(
        &mut too_many_chunks,
        u32::try_from(MAX_TOMBSTONE_CHUNKS + 1).expect("chunk cap fits u32"),
    );
    assert!(matches!(
        validate_tombstone_bytes(&too_many_chunks, None),
        Err(ManifestCodecError::ResourceLimit {
            resource: "tombstone chunk count",
            ..
        })
    ));

    let sparse = array_tombstones(0, &[1, 100]);
    let dense_lows = (0..=TOMBSTONE_ARRAY_MAX_CARDINALITY).collect::<Vec<_>>();
    let dense = bitmap_tombstones(0, &dense_lows);
    assert!(
        dense
            .is_monotone_superset_of(&sparse)
            .expect("compare array to bitmap")
    );
    let missing = array_tombstones(0, &[1, 5_000]);
    assert!(
        !dense
            .is_monotone_superset_of(&missing)
            .expect("detect missing bitmap bit")
    );
}

#[test]
fn tombstone_set_mutation_preserves_hysteresis_and_exact_roundtrip() -> TestResult {
    let sparse_lows = (0..4_095_u16).collect::<Vec<_>>();
    let mut set = array_tombstones(0, &sparse_lows);
    assert_eq!(set.cardinality(), 4_095);
    assert_eq!(set.as_bytes()[6], 0);

    assert!(set.insert(4_095)?);
    assert_eq!(set.cardinality(), 4_096);
    assert_eq!(set.as_bytes()[6], 0, "4,096 remains ARRAY");
    let array_wire = set.as_bytes().to_vec();
    assert_eq!(
        TombstoneSet::from_bytes(&array_wire)?.as_bytes(),
        array_wire
    );

    assert!(set.insert(4_096)?);
    assert_eq!(set.cardinality(), 4_097);
    assert_eq!(set.as_bytes()[6], 1, "4,097 promotes to BITMAP");
    assert!((0..=4_096_u32).all(|docid| set.contains(docid)));
    let bitmap_wire = set.as_bytes().to_vec();
    assert_eq!(
        TombstoneSet::from_bytes(&bitmap_wire)?.as_bytes(),
        bitmap_wire
    );

    let dense_lows = (0..3_585_u16).collect::<Vec<_>>();
    let mut dense = bitmap_tombstones(0, &dense_lows);
    assert!(dense.remove(3_584)?);
    assert_eq!(dense.cardinality(), 3_584);
    assert_eq!(dense.as_bytes()[6], 1, "3,584 retains BITMAP");
    assert!(dense.remove(3_583)?);
    assert_eq!(dense.cardinality(), 3_583);
    assert_eq!(dense.as_bytes()[6], 0, "3,583 demotes to ARRAY");
    Ok(())
}

#[test]
fn tombstone_set_handles_chunk_edges_duplicates_and_full_bitmap() -> TestResult {
    let mut set = TombstoneSet::new();
    for docid in [u32::MAX, 0, 65_536 + 7, 65_535] {
        assert!(set.insert(docid)?);
    }
    assert!(!set.insert(u32::MAX)?);
    assert_eq!(set.cardinality(), 4);
    assert!(
        [u32::MAX, 0, 65_536 + 7, 65_535]
            .into_iter()
            .all(|docid| set.contains(docid))
    );
    assert_eq!(tombstone_chunk_count(set.as_bytes())?, 3);
    assert_eq!(TombstoneSet::from_bytes(set.as_bytes())?, set);

    let all_lows = (0..=u16::MAX).collect::<Vec<_>>();
    let mut full = bitmap_tombstones(u16::MAX, &all_lows);
    assert_eq!(full.cardinality(), 65_536);
    assert_eq!(&full.as_bytes()[7..9], &[0, 0]);
    assert!(full.remove(u32::MAX)?);
    assert_eq!(&full.as_bytes()[7..9], &u16::MAX.to_le_bytes());
    assert!(full.insert(u32::MAX)?);
    assert_eq!(&full.as_bytes()[7..9], &[0, 0]);
    Ok(())
}

#[test]
fn tombstone_successors_preserve_representation_state() -> TestResult {
    let array_4_096 = array_tombstones(0, &(0..4_096_u16).collect::<Vec<_>>());
    let bitmap_4_097 = bitmap_tombstones(0, &(0..4_097_u16).collect::<Vec<_>>());
    assert!(bitmap_4_097.is_monotone_superset_of(&array_4_096)?);

    let bitmap_3_584 = bitmap_tombstones(0, &(0..3_584_u16).collect::<Vec<_>>());
    let array_3_585 = array_tombstones(0, &(0..3_585_u16).collect::<Vec<_>>());
    assert!(
        !array_3_585.is_monotone_superset_of(&bitmap_3_584)?,
        "retained BITMAP cannot demote in a growth-only generation"
    );

    let array_overlap = array_tombstones(0, &(0..4_096_u16).collect::<Vec<_>>());
    let bitmap_overlap = bitmap_tombstones(0, &(0..4_096_u16).collect::<Vec<_>>());
    assert!(
        !bitmap_overlap.is_monotone_superset_of(&array_overlap)?,
        "ARRAY cannot promote before its 4,097th member"
    );
    assert!(
        !bitmap_3_584.is_monotone_superset_of(&TombstoneSet::new())?,
        "a new chunk cannot begin as BITMAP inside the hysteresis overlap"
    );
    assert_eq!(
        TombstoneSet::from_bytes(array_overlap.as_bytes())?.as_bytes(),
        array_overlap.as_bytes()
    );
    assert_eq!(
        TombstoneSet::from_bytes(bitmap_overlap.as_bytes())?.as_bytes(),
        bitmap_overlap.as_bytes()
    );
    Ok(())
}

#[test]
fn stats_must_be_ordered_and_bounded_by_snapshot_docs() {
    let mut manifest = sample_manifest(1);
    manifest.field_stats.swap(0, 1);
    assert!(manifest.validate().is_err());

    let mut manifest = sample_manifest(1);
    manifest.field_stats[0].doc_count = 6;
    assert!(manifest.validate().is_err());
}

#[test]
fn stats_denominator_must_equal_the_canonical_at_seal_count() {
    // Every field row uses the same global at-seal denominator
    // (bd-quill-e3-keeper-ndtk.10); tombstones do not change it. Reject
    // deviations in either direction; exact equality round-trips.
    let mut too_large = sample_manifest(1);
    too_large.field_stats[0].doc_count = 6;
    let error = too_large.validate().expect_err("over-count rejected");
    assert!(
        error
            .to_string()
            .contains("differs from canonical at-seal denominator")
    );

    let mut too_small = sample_manifest(1);
    too_small.field_stats[1].doc_count = 4;
    let error = too_small.validate().expect_err("under-count rejected");
    assert!(
        error
            .to_string()
            .contains("differs from canonical at-seal denominator")
    );

    let canonical = sample_manifest(1);
    canonical.validate().expect("exact denominator passes");
    let bytes = canonical.to_bytes().expect("encode canonical");
    assert_eq!(
        Manifest::from_bytes(&bytes).expect("decode canonical"),
        canonical,
        "canonical denominator round-trips"
    );

    // Tombstones never change the denominator: removing the tombstone
    // set from a segment does not relax the equality rule.
    let mut no_tombstones = canonical;
    no_tombstones.segments[0].tombstones = TombstoneSet::new();
    no_tombstones
        .validate()
        .expect("tombstone-free keeps denominator");
    no_tombstones.field_stats[0].doc_count = 2;
    assert!(no_tombstones.validate().is_err());
}

#[test]
fn arbitrary_bytes_never_panic() {
    let mut state = 0x5e31_9a72_d0c4_b611_u64;
    for case in 0..1_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let length = usize::try_from(state % 257).expect("small generated length");
        let mut bytes = Vec::with_capacity(length);
        for _ in 0..length {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.push(state.to_le_bytes()[0]);
        }
        assert!(
            std::panic::catch_unwind(|| Manifest::from_bytes(&bytes)).is_ok(),
            "parser panic in case {case}"
        );
    }
}

#[test]
fn pair_loader_distinguishes_not_found_and_both_fallback_windows() -> TestResult {
    let directory = tempdir()?;
    assert!(matches!(
        load_manifest_pair(directory.path()),
        Err(KeeperError::IndexNotFound { .. })
    ));

    let previous = sample_manifest(4);
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    let loaded = load_manifest_pair(directory.path())?;
    assert_eq!(loaded.manifest, previous);
    assert_eq!(loaded.source, ManifestSource::PreviousAfterMissingCurrent);

    std::fs::write(directory.path().join("MANIFEST"), b"corrupt")?;
    let loaded = load_manifest_pair(directory.path())?;
    assert_eq!(loaded.manifest, previous);
    assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);
    Ok(())
}

#[cfg(unix)]
#[test]
fn pair_loader_never_follows_manifest_symlinks_or_special_entries() -> TestResult {
    use std::os::unix::fs::symlink;

    let directory = tempdir()?;
    let previous = sample_manifest(4);
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    let symlink_target = directory.path().join("manifest-target");
    write_manifest(&symlink_target, &sample_manifest(5))?;
    symlink(&symlink_target, directory.path().join("MANIFEST"))?;

    let loaded = load_manifest_pair(directory.path())?;
    assert_eq!(loaded.manifest, previous);
    assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);

    let directory = tempdir()?;
    let previous = sample_manifest(7);
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    std::fs::create_dir(directory.path().join("MANIFEST"))?;
    let loaded = load_manifest_pair(directory.path())?;
    assert_eq!(loaded.manifest, previous);
    assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);

    let directory = tempdir()?;
    let current = sample_manifest(9);
    write_manifest(&directory.path().join("MANIFEST"), &current)?;
    let previous_target = directory.path().join("previous-target");
    write_manifest(&previous_target, &sample_manifest(8))?;
    symlink(&previous_target, directory.path().join("MANIFEST.prev"))?;
    let loaded = load_manifest_pair(directory.path())?;
    assert_eq!(loaded.manifest, current);
    assert_eq!(loaded.source, ManifestSource::Current);
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn pair_loader_treats_fifo_manifest_as_invalid_without_blocking() -> TestResult {
    use rustix::fs::{Mode, mkfifoat};

    let directory = tempdir()?;
    let previous = sample_manifest(4);
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    let directory_file = File::open(directory.path())?;
    mkfifoat(&directory_file, "MANIFEST", Mode::RUSR | Mode::WUSR)?;

    let loaded = load_manifest_pair(directory.path())?;
    assert_eq!(loaded.manifest, previous);
    assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);
    Ok(())
}

#[test]
fn pair_loader_rejects_nonadjacent_valid_generations() -> TestResult {
    let directory = tempdir()?;
    write_manifest(&directory.path().join("MANIFEST"), &sample_manifest(9))?;
    write_manifest(&directory.path().join("MANIFEST.prev"), &sample_manifest(3))?;
    assert!(matches!(
        load_manifest_pair(directory.path()),
        Err(KeeperError::InvalidGenerationPair {
            current: 9,
            previous: 3,
            ..
        })
    ));
    Ok(())
}

#[test]
fn pair_loader_rejects_stable_identity_and_watermark_disagreement() -> TestResult {
    let directory = tempdir()?;
    let previous = sample_manifest(8);

    let mut equal_but_different = previous.clone();
    equal_but_different.flags = MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
    write_manifest(&directory.path().join("MANIFEST"), &equal_but_different)?;
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    assert!(matches!(
        load_manifest_pair(directory.path()),
        Err(KeeperError::InvalidManifestPair {
            current: 8,
            previous: 8,
            ..
        })
    ));

    let mut schema_mismatch = sample_manifest(9);
    schema_mismatch.schema_id ^= 1;
    write_manifest(&directory.path().join("MANIFEST"), &schema_mismatch)?;
    assert!(matches!(
        load_manifest_pair(directory.path()),
        Err(KeeperError::InvalidManifestPair {
            current: 9,
            previous: 8,
            ..
        })
    ));

    let mut watermark_rollback = sample_manifest(9);
    watermark_rollback.docid_high_watermark = previous.docid_high_watermark - 1;
    write_manifest(&directory.path().join("MANIFEST"), &watermark_rollback)?;
    assert!(matches!(
        load_manifest_pair(directory.path()),
        Err(KeeperError::InvalidManifestPair {
            current: 9,
            previous: 8,
            ..
        })
    ));

    let mut resurrected = sample_manifest(9);
    resurrected.segments[0].tombstones = array_tombstones(0, &[101]);
    write_manifest(&directory.path().join("MANIFEST"), &resurrected)?;
    assert!(matches!(
        load_manifest_pair(directory.path()),
        Err(KeeperError::InvalidManifestPair {
            current: 9,
            previous: 8,
            ..
        })
    ));

    let mut changed_stats = sample_manifest(9);
    changed_stats.field_stats[0].total_tokens += 1;
    write_manifest(&directory.path().join("MANIFEST"), &changed_stats)?;
    assert!(matches!(
        load_manifest_pair(directory.path()),
        Err(KeeperError::InvalidManifestPair {
            current: 9,
            previous: 8,
            ..
        })
    ));
    Ok(())
}

#[cfg(unix)]
#[test]
fn valid_current_wins_when_previous_cannot_be_read_as_a_file() -> TestResult {
    let directory = tempdir()?;
    let current = sample_manifest(1);
    write_manifest(&directory.path().join("MANIFEST"), &current)?;
    std::fs::create_dir(directory.path().join("MANIFEST.prev"))?;

    let loaded = load_manifest_pair(directory.path())?;
    assert_eq!(loaded.manifest, current);
    assert_eq!(loaded.source, ManifestSource::Current);
    Ok(())
}

#[test]
fn publish_reopens_monotone_current_and_previous_generations() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        let first = sample_manifest(1);
        publisher
            .publish(&cx, &first)
            .await
            .expect("publish genesis");

        let mut second = sample_manifest(2);
        second.docid_high_watermark = first.docid_high_watermark + 100;
        publisher
            .publish(&cx, &second)
            .await
            .expect("publish second");

        let loaded = load_manifest_pair(directory.path()).expect("reopen pair");
        assert_eq!(loaded.manifest, second);
        assert_eq!(loaded.source, ManifestSource::Current);
        let previous_bytes =
            std::fs::read(directory.path().join("MANIFEST.prev")).expect("read previous");
        assert_eq!(
            Manifest::from_bytes(&previous_bytes).expect("parse previous"),
            first
        );
    });
}

#[test]
fn publish_rejects_generation_gap_and_watermark_rollback() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        let first = sample_manifest(1);
        publisher
            .publish(&cx, &first)
            .await
            .expect("publish genesis");

        assert!(matches!(
            publisher.publish(&cx, &sample_manifest(3)).await,
            Err(KeeperError::GenerationConflict {
                expected: 2,
                proposed: 3
            })
        ));
        let mut rollback = sample_manifest(2);
        rollback.docid_high_watermark = first.docid_high_watermark - 1;
        assert!(matches!(
            publisher.publish(&cx, &rollback).await,
            Err(KeeperError::InvalidTransition { .. })
        ));
    });
}

#[test]
fn publish_preserves_retained_segment_identity_and_monotone_tombstones() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        let first = sample_manifest(1);
        publisher
            .publish(&cx, &first)
            .await
            .expect("publish genesis");

        let mut mutated = sample_manifest(2);
        mutated.segments[0].file_xxh3 ^= 1;
        assert!(matches!(
            publisher.publish(&cx, &mutated).await,
            Err(KeeperError::InvalidTransition { .. })
        ));

        let mut resurrected = sample_manifest(2);
        resurrected.segments[0].tombstones = array_tombstones(0, &[101]);
        assert!(matches!(
            publisher.publish(&cx, &resurrected).await,
            Err(KeeperError::InvalidTransition { .. })
        ));

        let mut stale_seal = sample_manifest(2);
        stale_seal.segments.push(ManifestSegment {
            segment_id: 0x1003,
            seal_seq: 6,
            file_len: 512,
            file_xxh3: 0xc1c2_c3c4_c5c6_c7c8,
            docid_lo: 70_000,
            docid_hi: 70_001,
            doc_count: 1,
            tombstones: TombstoneSet::new(),
        });
        // Keep the canonical STATS denominator consistent with the new
        // aggregate so the stale seal_seq is the transition being tested.
        for stats in &mut stale_seal.field_stats {
            stats.doc_count = 6;
        }
        assert!(matches!(
            publisher.publish(&cx, &stale_seal).await,
            Err(KeeperError::InvalidTransition { .. })
        ));

        let mut changed_stats = sample_manifest(2);
        changed_stats.field_stats[0].total_tokens += 1;
        assert!(matches!(
            publisher.publish(&cx, &changed_stats).await,
            Err(KeeperError::InvalidTransition { .. })
        ));

        let mut monotone = sample_manifest(2);
        monotone.segments[0].tombstones = array_tombstones(0, &[101, 150, 175]);
        publisher
            .publish(&cx, &monotone)
            .await
            .expect("monotone tombstone growth publishes");
        assert_eq!(
            load_manifest_pair(directory.path())
                .expect("reopen monotone publish")
                .manifest,
            monotone
        );
    });
}

#[test]
fn empty_manifest_starts_a_new_seal_sequence_epoch() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        let first = sample_manifest(1);
        publisher
            .publish(&cx, &first)
            .await
            .expect("publish genesis");

        let mut empty = Manifest::empty(
            2,
            first.schema_id,
            first.docid_high_watermark.saturating_add(1),
        );
        empty.last_publish_unix_s = 1_700_000_000;
        publisher
            .publish(&cx, &empty)
            .await
            .expect("publish explicit empty generation");

        let mut restarted = Manifest::empty(3, first.schema_id, empty.docid_high_watermark);
        restarted.last_publish_unix_s = 1_700_000_000;
        let mut segment = first.segments[0].clone();
        segment.segment_id = 0x2001;
        segment.seal_seq = 1;
        restarted.segments.push(segment);
        restarted.field_stats = vec![
            ManifestFieldStats {
                field_ord: 1,
                total_tokens: 12,
                doc_count: 3,
            },
            ManifestFieldStats {
                field_ord: 3,
                total_tokens: 42,
                doc_count: 3,
            },
        ];
        publisher
            .publish(&cx, &restarted)
            .await
            .expect("restart seal sequence after empty epoch boundary");
        assert_eq!(
            load_manifest_pair(directory.path())
                .expect("reopen restarted seal epoch")
                .manifest,
            restarted
        );
    });
}

#[test]
fn invalid_or_cancelled_proposal_creates_no_manifest_slot() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        assert!(matches!(
            publisher.publish(&cx, &Manifest::empty(0, 7, 0)).await,
            Err(KeeperError::InvalidManifest { .. })
        ));

        cx.set_cancel_requested(true);
        assert!(matches!(
            publisher.publish(&cx, &Manifest::empty(1, 7, 0)).await,
            Err(KeeperError::PublishLock {
                source: LockError::Cancelled
            })
        ));
        assert!(!directory.path().join("MANIFEST").exists());
        assert!(!directory.path().join("MANIFEST.prev").exists());
    });
}

#[test]
fn generation_claim_runs_after_temp_fsync_and_before_renames() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        publisher
            .publish(&cx, &sample_manifest(1))
            .await
            .expect("publish genesis");
        let saw_expected_window = Arc::new(AtomicBool::new(false));
        let witness = Arc::clone(&saw_expected_window);
        publisher
            .publish_with_generation_claim(&cx, &sample_manifest(2), move |path, generation| {
                let temp = path.join(format!(".tmp-manifest-{generation}"));
                let current = path.join("MANIFEST");
                let previous = path.join("MANIFEST.prev");
                witness.store(
                    temp.is_file() && current.is_file() && !previous.exists(),
                    Ordering::SeqCst,
                );
                Ok(())
            })
            .await
            .expect("claimed publish");
        assert!(saw_expected_window.load(Ordering::SeqCst));
    });
}

#[test]
fn ordinary_segment_publish_performs_zero_sidecar_io() -> TestResult {
    let directory = tempdir()?;
    let encoded = encoded_test_segment(0x50, 0, 2, 1)?;
    let pending = encoded.write_temp(directory.path())?;
    let published = directory.path().join(canonical_segment_name(0x50));
    let mut sidecar_name = published.as_os_str().to_os_string();
    sidecar_name.push(".fec");
    let sidecar = PathBuf::from(sidecar_name);
    let sentinel = b"opaque feature-off segment sidecar";
    std::fs::write(&sidecar, sentinel)?;

    assert_eq!(publish_pending_segment(pending)?, published);
    assert_eq!(std::fs::read(&published)?, encoded.as_bytes());
    assert_eq!(std::fs::read(sidecar)?, sentinel);
    Ok(())
}

#[test]
fn ordinary_manifest_publish_performs_zero_sidecar_io() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let publisher = ManifestPublisher::new(directory.path());
        publisher
            .publish(&cx, &sample_manifest(1))
            .await
            .expect("publish genesis");
        let sidecar = directory.path().join("MANIFEST.fec");
        std::fs::write(&sidecar, b"opaque feature-off artifact").expect("write sentinel");

        publisher
            .publish(&cx, &sample_manifest(2))
            .await
            .expect("publish without durability");

        assert_eq!(
            std::fs::read(&sidecar).expect("sentinel remains readable"),
            b"opaque feature-off artifact"
        );
        assert!(!directory.path().join("MANIFEST.prev.fec").exists());
    });
}
