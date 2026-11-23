use super::*;

#[cfg(unix)]
#[test]
fn gc_refuses_a_backdated_unreferenced_segment_without_a_retirement_receipt() -> TestResult {
    let (directory, stranded) = stranded_segment_fixture()?;
    let stranded_path = directory
        .path()
        .join(canonical_segment_name(stranded.segment_id));
    // A year past every timestamp in the fixture: nothing here is young,
    // so age is the only thing a fail-open collector could be waiting on.
    let now = SystemTime::now()
        .checked_add(Duration::from_secs(365 * 24 * 60 * 60))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;

    let report = collect_writer_garbage_at(
        directory.path(),
        DEFAULT_SCHEMA,
        GarbageCollectionOptions::default(),
        now,
    )?;
    assert!(report.is_empty());
    assert!(
        stranded_path.exists(),
        "a missing first-unreachable receipt must fail closed however old the segment file is"
    );

    // Paired positive control: same directory, same clock, same call --
    // only the durable witness is added. A collector that is merely dead
    // cannot pass both halves of this test.
    write_test_retirement_receipt(directory.path(), stranded.segment_id, 3, SystemTime::now())?;
    let report = collect_writer_garbage_at(
        directory.path(),
        DEFAULT_SCHEMA,
        GarbageCollectionOptions::default(),
        now,
    )?;
    assert_eq!(
        report.removed,
        vec![
            PathBuf::from(canonical_segment_name(stranded.segment_id)),
            PathBuf::from(retirement_receipt_name(stranded.segment_id)),
        ],
        "a witnessed retirement past its grace is still collected exactly once"
    );
    assert!(!stranded_path.exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn gc_counts_distinct_slot_generations_before_trusting_mtime() -> TestResult {
    // A slot pair covering generations 1 and 2 cannot have dropped a
    // published manifest, so an unreferenced segment there is provably a
    // never-published crash artifact and its file age still governs.
    let adjacent = tempdir()?;
    let live = write_test_segment(adjacent.path(), 0xa, 1, 0, 2)?;
    write_test_segment(adjacent.path(), 0xb, 2, 2, 4)?;
    let previous = durable_test_manifest(1, vec![live.clone()]);
    let mut current = durable_test_manifest(2, vec![live.clone()]);
    current.docid_high_watermark = previous.docid_high_watermark;
    write_manifest(&adjacent.path().join("MANIFEST.prev"), &previous)?;
    write_manifest(&adjacent.path().join("MANIFEST"), &current)?;
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert_eq!(
        collect_writer_garbage_at(
            adjacent.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?
        .removed,
        vec![PathBuf::from(canonical_segment_name(0xb))],
        "generations 1 and 2 leave no room for a retirement to have happened"
    );

    // Two slots naming the *same* generation cover one generation, not
    // two, so generation 1 has already been dropped and an unreferenced
    // segment may be a retired one. Counting slots instead of distinct
    // generations would reopen mtime inference here.
    let duplicated = tempdir()?;
    let live = write_test_segment(duplicated.path(), 0xa, 1, 0, 2)?;
    let stranded = write_test_segment(duplicated.path(), 0xb, 2, 2, 4)?;
    let repeated = durable_test_manifest(2, vec![live]);
    write_manifest(&duplicated.path().join("MANIFEST.prev"), &repeated)?;
    write_manifest(&duplicated.path().join("MANIFEST"), &repeated)?;
    assert!(
        collect_writer_garbage_at(
            duplicated.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?
        .is_empty(),
        "a repeated slot generation must not be counted twice"
    );
    assert!(
        duplicated
            .path()
            .join(canonical_segment_name(stranded.segment_id))
            .exists()
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn gc_rejects_forged_truncated_and_foreign_retirement_receipts() -> TestResult {
    let valid = SegmentRetirementReceipt {
        segment_id: 0xb,
        retired_unix_s: 1_700_000_000,
        retired_generation: 3,
        witness: UnreachabilityWitness::RetiredByPublication,
    };
    let mut forged = valid.encode().to_vec();
    forged[16] ^= 0xff;
    let mut wrong_magic = valid.encode().to_vec();
    wrong_magic[0] = b'X';
    let mut wrong_version = valid.encode().to_vec();
    wrong_version[4] = 2;
    let mut unknown_witness = valid.encode().to_vec();
    unknown_witness[5] = 9;
    let mut dirty_reserved = valid.encode().to_vec();
    dirty_reserved[6] = 1;
    let truncated = valid.encode()[..RETIREMENT_RECEIPT_BYTES - 1].to_vec();
    let mut padded = valid.encode().to_vec();
    padded.push(0);
    let foreign = SegmentRetirementReceipt {
        segment_id: 0xc,
        ..valid
    }
    .encode()
    .to_vec();
    let future = SegmentRetirementReceipt {
        retired_unix_s: i64::MAX,
        ..valid
    }
    .encode()
    .to_vec();

    for (label, image) in [
        ("seal broken by a flipped timestamp byte", forged),
        ("foreign magic", wrong_magic),
        ("unknown version", wrong_version),
        ("unknown witness kind", unknown_witness),
        ("non-zero reserved bytes", dirty_reserved),
        ("truncated image", truncated),
        ("over-long image", padded),
        ("receipt bound to another segment", foreign),
        ("retirement stamped beyond the clock", future),
    ] {
        let (directory, stranded) = stranded_segment_fixture()?;
        let stranded_path = directory
            .path()
            .join(canonical_segment_name(stranded.segment_id));
        let receipt_path = directory
            .path()
            .join(retirement_receipt_name(stranded.segment_id));
        std::fs::write(&receipt_path, &image)?;
        let now = SystemTime::now()
            .checked_add(Duration::from_secs(365 * 24 * 60 * 60))
            .ok_or_else(|| io::Error::other("test clock remains representable"))?;

        let report = collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?;
        assert!(report.is_empty(), "{label}");
        assert!(stranded_path.exists(), "{label}: segment survives");
        assert_eq!(
            std::fs::read(&receipt_path)?,
            image,
            "{label}: an unusable receipt is preserved as evidence"
        );
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn gc_refuses_a_receipt_that_has_not_served_its_full_grace() -> TestResult {
    let (directory, stranded) = stranded_segment_fixture()?;
    let stranded_path = directory
        .path()
        .join(canonical_segment_name(stranded.segment_id));
    let retired_at = SystemTime::now();
    write_test_retirement_receipt(directory.path(), stranded.segment_id, 3, retired_at)?;
    let options = GarbageCollectionOptions {
        grace_period: Duration::from_secs(600),
    };

    // The manifest-level floor has long since expired here; only the
    // receipt's own witness still protects the segment.
    let before_grace = retired_at
        .checked_add(Duration::from_secs(599))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert!(
        collect_writer_garbage_at(directory.path(), DEFAULT_SCHEMA, options, before_grace)?
            .is_empty(),
        "grace is measured from the witnessed transition"
    );
    assert!(stranded_path.exists());

    let after_grace = retired_at
        .checked_add(Duration::from_secs(601))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert_eq!(
        collect_writer_garbage_at(directory.path(), DEFAULT_SCHEMA, options, after_grace)?.removed,
        vec![
            PathBuf::from(canonical_segment_name(stranded.segment_id)),
            PathBuf::from(retirement_receipt_name(stranded.segment_id)),
        ]
    );
    assert!(!stranded_path.exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn gc_retires_a_receipt_whose_segment_is_live_again_or_already_gone() -> TestResult {
    // The margin over the grace period only has to make receipts stamped
    // DURING this test look expired at `now`; at +1s the stamps landed
    // after `now - grace` when the parallel suite delayed the writes
    // below by more than a second (bd-zam2i).
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(60))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;

    // An aborted publication can stamp a receipt and never perform the
    // rotation, leaving a witness for a segment that is still reachable.
    let directory = tempdir()?;
    let live = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
    let previous = durable_test_manifest(2, vec![live.clone()]);
    let current = durable_test_manifest(3, vec![live]);
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    write_manifest(&directory.path().join("MANIFEST"), &current)?;
    write_test_retirement_receipt(directory.path(), 0xa, 3, SystemTime::now())?;
    assert_eq!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?
        .removed,
        vec![PathBuf::from(retirement_receipt_name(0xa))],
        "a stale receipt is reclaimed and the live segment is untouched"
    );
    assert!(directory.path().join(canonical_segment_name(0xa)).exists());

    // An interrupted sweep can unlink the segment and leave its receipt.
    let orphaned = tempdir()?;
    let live = write_test_segment(orphaned.path(), 0xa, 1, 0, 2)?;
    let previous = durable_test_manifest(2, vec![live.clone()]);
    let current = durable_test_manifest(3, vec![live]);
    write_manifest(&orphaned.path().join("MANIFEST.prev"), &previous)?;
    write_manifest(&orphaned.path().join("MANIFEST"), &current)?;
    write_test_retirement_receipt(orphaned.path(), 0xb, 3, SystemTime::now())?;
    assert_eq!(
        collect_writer_garbage_at(
            orphaned.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?
        .removed,
        vec![PathBuf::from(retirement_receipt_name(0xb))],
        "a receipt with no segment left to authorize is reclaimed"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn witness_stamps_orphans_and_never_a_reachable_or_already_witnessed_segment() -> TestResult {
    let directory = tempdir()?;
    let current_live = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
    let previous_only = write_test_segment(directory.path(), 0xc, 2, 4, 6)?;
    let orphan = write_test_segment(directory.path(), 0xb, 3, 2, 4)?;
    let retired = write_test_segment(directory.path(), 0xd, 4, 6, 8)?;
    // Generation 4 over slots naming 3 and 4: a generation has been
    // superseded, so this directory is past the threshold.
    let previous = durable_test_manifest(3, vec![current_live.clone(), previous_only.clone()]);
    let mut current = durable_test_manifest(4, vec![current_live]);
    current.docid_high_watermark = previous.docid_high_watermark;
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    write_manifest(&directory.path().join("MANIFEST"), &current)?;

    // A segment already covered by a retirement receipt keeps that
    // receipt verbatim: re-witnessing it would restart its grace on every
    // writer open and it would never be reclaimed at all.
    let retired_at = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_700_000_000))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    write_test_retirement_receipt(directory.path(), retired.segment_id, 4, retired_at)?;
    let retired_receipt_bytes = std::fs::read(
        directory
            .path()
            .join(retirement_receipt_name(retired.segment_id)),
    )?;

    let report = witness_orphaned_segments_at(directory.path(), DEFAULT_SCHEMA, retired_at)?;
    assert_eq!(
        report.witnessed,
        vec![PathBuf::from(retirement_receipt_name(orphan.segment_id))],
        "only the unreferenced, unwitnessed segment is sighted"
    );
    assert!(
        !directory.path().join(retirement_receipt_name(0xa)).exists(),
        "a segment live in MANIFEST is never sighted"
    );
    assert!(
        !directory
            .path()
            .join(retirement_receipt_name(previous_only.segment_id))
            .exists(),
        "a published segment still reachable through MANIFEST.prev is never sighted"
    );
    assert_eq!(
        std::fs::read(
            directory
                .path()
                .join(retirement_receipt_name(retired.segment_id))
        )?,
        retired_receipt_bytes,
        "an existing retirement receipt is never overwritten"
    );

    let sighting = SegmentRetirementReceipt::decode(&std::fs::read(
        directory
            .path()
            .join(retirement_receipt_name(orphan.segment_id)),
    )?)
    .ok_or_else(|| io::Error::other("the sighting decodes"))?;
    assert_eq!(sighting.segment_id, orphan.segment_id);
    assert_eq!(
        sighting.witness,
        UnreachabilityWitness::FirstSightedByWriter
    );
    assert_eq!(sighting.retired_generation, 4);

    // Idempotent: a second pass adds nothing and rewrites nothing.
    let orphan_bytes = std::fs::read(
        directory
            .path()
            .join(retirement_receipt_name(orphan.segment_id)),
    )?;
    let repeat = witness_orphaned_segments_at(
        directory.path(),
        DEFAULT_SCHEMA,
        retired_at
            .checked_add(Duration::from_secs(3_600))
            .ok_or_else(|| io::Error::other("test clock remains representable"))?,
    )?;
    assert!(repeat.witnessed.is_empty());
    assert_eq!(
        std::fs::read(
            directory
                .path()
                .join(retirement_receipt_name(orphan.segment_id))
        )?,
        orphan_bytes,
        "a repeated pass must not restart the sighting's grace"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn witnessed_orphan_is_reclaimed_only_after_a_full_grace_from_its_sighting() -> TestResult {
    let (directory, orphan) = stranded_segment_fixture()?;
    let orphan_path = directory
        .path()
        .join(canonical_segment_name(orphan.segment_id));
    let sighted_at = SystemTime::now();

    // The fail-closed baseline this bead started from: past the
    // threshold, an orphan with no witness is never reclaimed however old
    // it is (its mtime is UNIX_EPOCH here).
    let long_after = sighted_at
        .checked_add(Duration::from_secs(365 * 24 * 60 * 60))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            long_after,
        )?
        .is_empty(),
        "an unwitnessed orphan is not reclaimable on file age"
    );
    assert!(orphan_path.exists());

    assert_eq!(
        witness_orphaned_segments_at(directory.path(), DEFAULT_SCHEMA, sighted_at)?.witnessed,
        vec![PathBuf::from(retirement_receipt_name(orphan.segment_id))]
    );

    // Grace runs from the sighting, not from the segment's mtime.
    let before_grace = sighted_at
        .checked_add(
            DEFAULT_GARBAGE_GRACE
                .checked_sub(Duration::from_secs(2))
                .ok_or_else(|| io::Error::other("grace exceeds two seconds"))?,
        )
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            before_grace,
        )?
        .is_empty(),
        "a fresh sighting owes the full grace period"
    );
    assert!(orphan_path.exists());

    let after_grace = sighted_at
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(2))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert_eq!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            after_grace,
        )?
        .removed,
        vec![
            PathBuf::from(canonical_segment_name(orphan.segment_id)),
            PathBuf::from(retirement_receipt_name(orphan.segment_id)),
        ],
        "a witnessed orphan is reclaimed with its receipt"
    );
    assert!(!orphan_path.exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn writer_open_witnesses_orphans_before_it_sweeps() -> TestResult {
    // The production wiring: a crash orphan in a past-threshold directory
    // gains its witness at the next writer admission, without that open
    // deleting anything.
    let (directory, orphan) = stranded_segment_fixture()?;
    let orphan_path = directory
        .path()
        .join(canonical_segment_name(orphan.segment_id));
    let receipt_path = directory
        .path()
        .join(retirement_receipt_name(orphan.segment_id));
    assert!(!receipt_path.exists());

    let path = directory.path().to_path_buf();
    let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
        KeeperWriter::open(&cx, &path, DEFAULT_SCHEMA)
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    });
    outcome.map_err(io::Error::other)?;

    let receipt = SegmentRetirementReceipt::decode(&std::fs::read(&receipt_path)?)
        .ok_or_else(|| io::Error::other("writer open wrote a decodable sighting"))?;
    assert_eq!(receipt.segment_id, orphan.segment_id);
    assert_eq!(receipt.witness, UnreachabilityWitness::FirstSightedByWriter);
    assert!(
        orphan_path.exists(),
        "witnessing must not delete anything by itself"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn witness_does_nothing_below_the_supersession_threshold() -> TestResult {
    // The crash-state contract: one slot at generation 1 leaves no room
    // for a retirement, the sweep still reclaims on file age in a single
    // pass, and no receipt is manufactured for it.
    let directory = tempdir()?;
    let live = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
    let orphan = write_test_segment(directory.path(), 0xb, 2, 2, 4)?;
    write_manifest(
        &directory.path().join("MANIFEST"),
        &durable_test_manifest(1, vec![live]),
    )?;

    assert!(
        witness_orphaned_segments_at(directory.path(), DEFAULT_SCHEMA, SystemTime::now())?
            .witnessed
            .is_empty(),
        "below the threshold there is no missing witness to supply"
    );
    assert!(
        !directory
            .path()
            .join(retirement_receipt_name(orphan.segment_id))
            .exists()
    );
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert_eq!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?
        .removed,
        vec![PathBuf::from(canonical_segment_name(orphan.segment_id))],
        "single-pass reclamation below the threshold is unchanged"
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn a_retirement_overwrites_an_earlier_sighting_of_the_same_segment() -> TestResult {
    // A segment staged while a witness pass ran can be sighted and then
    // legitimately published and later retired. The retirement is the
    // stronger, later witness and must replace the sighting, or the
    // segment would be reclaimed on a clock that predates its own
    // reachability.
    let directory = tempdir()?;
    let segment = write_test_segment(directory.path(), 0xa11, 1, 0, 2)?;
    let ancient = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(1_000_000))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    write_unreachability_receipt(
        directory.path(),
        SegmentRetirementReceipt {
            segment_id: segment.segment_id,
            retired_unix_s: i64::try_from(
                ancient.duration_since(SystemTime::UNIX_EPOCH)?.as_secs(),
            )?,
            retired_generation: 1,
            witness: UnreachabilityWitness::FirstSightedByWriter,
        },
    )?;

    let generation_one = durable_test_manifest(1, vec![segment.clone()]);
    publish_manifest_choreography(
        directory.path().to_path_buf(),
        &generation_one.to_bytes()?,
        |_, _| Ok(()),
        |_, _| Ok(()),
    )?;
    let mut generation_two = durable_test_manifest(2, Vec::new());
    generation_two.docid_high_watermark = generation_one.docid_high_watermark;
    publish_manifest_choreography(
        directory.path().to_path_buf(),
        &generation_two.to_bytes()?,
        |_, _| Ok(()),
        |_, _| Ok(()),
    )?;
    let mut generation_three = durable_test_manifest(3, Vec::new());
    generation_three.docid_high_watermark = generation_one.docid_high_watermark;
    publish_manifest_choreography(
        directory.path().to_path_buf(),
        &generation_three.to_bytes()?,
        |_, _| Ok(()),
        |_, _| Ok(()),
    )?;

    let receipt = SegmentRetirementReceipt::decode(&std::fs::read(
        directory
            .path()
            .join(retirement_receipt_name(segment.segment_id)),
    )?)
    .ok_or_else(|| io::Error::other("the receipt decodes"))?;
    assert_eq!(receipt.witness, UnreachabilityWitness::RetiredByPublication);
    assert_eq!(receipt.retired_generation, 3);
    let retired_at = receipt
        .retired_at()
        .ok_or_else(|| io::Error::other("receipt timestamp is representable"))?;
    assert!(
        retired_at > ancient,
        "the retirement must supersede the stale sighting's clock"
    );

    // The stale sighting would have authorized deletion decades ago.
    assert!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            ancient
                .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
                .ok_or_else(|| io::Error::other("test clock remains representable"))?,
        )?
        .is_empty()
    );
    assert!(
        directory
            .path()
            .join(canonical_segment_name(segment.segment_id))
            .exists()
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn publication_stamps_a_first_unreachable_receipt_for_every_segment_it_retires() -> TestResult {
    let directory = tempdir()?;
    let retired = write_test_segment(directory.path(), 0xa11, 1, 0, 2)?;
    let retained = write_test_segment(directory.path(), 0xb22, 2, 2, 4)?;
    let receipt_path = directory.path().join(retirement_receipt_name(0xa11));

    let generation_one = durable_test_manifest(1, vec![retired.clone()]);
    publish_manifest_choreography(
        directory.path().to_path_buf(),
        &generation_one.to_bytes()?,
        |_, _| Ok(()),
        |_, _| Ok(()),
    )?;

    // Generation 2 drops the segment from the current slot, but the
    // rotation moves generation 1 into MANIFEST.prev, so it is still
    // reachable and nothing may be witnessed yet.
    let generation_two = durable_test_manifest(2, vec![retained.clone()]);
    publish_manifest_choreography(
        directory.path().to_path_buf(),
        &generation_two.to_bytes()?,
        |_, _| Ok(()),
        |_, _| Ok(()),
    )?;
    assert!(
        !receipt_path.exists(),
        "a segment retained by MANIFEST.prev has not been retired"
    );

    // Generation 3 pushes generation 1 out of the window. This is the
    // only moment any component observes the transition.
    let generation_three = durable_test_manifest(3, vec![retained]);
    publish_manifest_choreography(
        directory.path().to_path_buf(),
        &generation_three.to_bytes()?,
        |_, _| Ok(()),
        |_, _| Ok(()),
    )?;

    let receipt = SegmentRetirementReceipt::decode(&std::fs::read(&receipt_path)?)
        .ok_or_else(|| io::Error::other("publication wrote a decodable receipt"))?;
    assert_eq!(receipt.segment_id, 0xa11);
    assert_eq!(receipt.retired_generation, 3);
    let retired_at = receipt
        .retired_at()
        .ok_or_else(|| io::Error::other("receipt timestamp is representable"))?;
    assert!(retired_at <= SystemTime::now());

    let segment_path = directory.path().join(canonical_segment_name(0xa11));
    assert!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            retired_at
                .checked_add(
                    DEFAULT_GARBAGE_GRACE
                        .checked_sub(Duration::from_secs(1))
                        .ok_or_else(|| io::Error::other("grace exceeds one second"))?,
                )
                .ok_or_else(|| io::Error::other("test clock remains representable"))?,
        )?
        .is_empty(),
        "the witnessed retirement still owes its grace period"
    );
    assert!(segment_path.exists());

    let after_grace = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .ok_or_else(|| io::Error::other("test clock remains representable"))?;
    assert_eq!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            after_grace,
        )?
        .removed,
        vec![
            PathBuf::from(canonical_segment_name(0xa11)),
            PathBuf::from(retirement_receipt_name(0xa11)),
        ]
    );
    assert!(!segment_path.exists());
    assert!(!receipt_path.exists());
    Ok(())
}

#[test]
fn future_claim_blocks_gc_before_any_deletion() -> TestResult {
    let directory = tempdir()?;
    write_manifest(
        &directory.path().join("MANIFEST"),
        &durable_test_manifest(2, Vec::new()),
    )?;
    std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
    std::fs::write(directory.path().join("gen-3.claim"), [])?;
    let before = directory_bytes(directory.path())?;
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .expect("test clock remains representable");

    assert!(matches!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        ),
        Err(KeeperError::ClaimedGenerationPending {
            current: 2,
            claimed: 3,
            ..
        })
    ));
    assert_eq!(directory_bytes(directory.path())?, before);
    Ok(())
}

#[test]
fn stale_claim_still_honors_the_gc_grace_period() -> TestResult {
    let directory = tempdir()?;
    write_manifest(
        &directory.path().join("MANIFEST"),
        &durable_test_manifest(2, Vec::new()),
    )?;
    std::fs::write(directory.path().join("gen-2.claim"), [])?;

    assert!(
        collect_writer_garbage_under_lock(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
        )?
        .is_empty()
    );
    assert!(directory.path().join("gen-2.claim").exists());

    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .expect("test clock remains representable");
    assert_eq!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?
        .removed,
        vec![PathBuf::from("gen-2.claim")]
    );
    Ok(())
}

#[test]
fn malformed_claim_blocks_gc_before_any_deletion() -> TestResult {
    let directory = tempdir()?;
    write_manifest(
        &directory.path().join("MANIFEST"),
        &durable_test_manifest(2, Vec::new()),
    )?;
    std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
    std::fs::create_dir(directory.path().join("gen-2.claim"))?;
    let before = directory_bytes(directory.path())?;
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .expect("test clock remains representable");

    assert!(matches!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        ),
        Err(KeeperError::InvalidClaimArtifact { .. })
    ));
    assert_eq!(directory_bytes(directory.path())?, before);
    assert!(directory.path().join("gen-2.claim").is_dir());

    let nonzero = tempdir()?;
    write_manifest(
        &nonzero.path().join("MANIFEST"),
        &durable_test_manifest(2, Vec::new()),
    )?;
    std::fs::write(nonzero.path().join(".tmp-old"), b"must survive")?;
    std::fs::write(nonzero.path().join("gen-2.claim"), b"not empty")?;
    let before = directory_bytes(nonzero.path())?;
    assert!(matches!(
        collect_writer_garbage_at(
            nonzero.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        ),
        Err(KeeperError::InvalidClaimArtifact { .. })
    ));
    assert_eq!(directory_bytes(nonzero.path())?, before);

    #[cfg(unix)]
    {
        let symlinked = tempdir()?;
        write_manifest(
            &symlinked.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(symlinked.path().join(".tmp-old"), b"must survive")?;
        let target = symlinked.path().join("claim-target");
        std::fs::write(&target, [])?;
        std::os::unix::fs::symlink(&target, symlinked.path().join("gen-2.claim"))?;
        let before = directory_bytes(symlinked.path())?;
        assert!(matches!(
            collect_writer_garbage_at(
                symlinked.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::InvalidClaimArtifact { .. })
        ));
        assert_eq!(directory_bytes(symlinked.path())?, before);
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn gc_rejects_symlinked_directory_alias_without_touching_target() -> TestResult {
    let target = tempdir()?;
    std::fs::write(target.path().join(".tmp-old"), b"must survive")?;
    let alias_root = tempdir()?;
    let alias = alias_root.path().join("index-alias");
    std::os::unix::fs::symlink(target.path(), &alias)?;
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .expect("test clock remains representable");

    assert!(matches!(
        collect_abandoned_genesis_garbage_at(&alias, GarbageCollectionOptions::default(), now,),
        Err(KeeperError::Io {
            operation: "open no-follow garbage-collection directory",
            ..
        })
    ));
    assert_eq!(
        std::fs::read(target.path().join(".tmp-old"))?,
        b"must survive"
    );
    Ok(())
}

#[test]
fn genesis_gc_requires_absent_slots_and_no_pending_claim() -> TestResult {
    let directory = tempdir()?;
    let segment = write_test_segment(directory.path(), 0x44, 1, 0, 2)?;
    write_manifest(
        &directory.path().join(".tmp-manifest-1"),
        &durable_test_manifest(1, vec![segment.clone()]),
    )?;
    std::fs::write(directory.path().join(".tmp-write-buffer"), b"uncommitted")?;
    std::fs::write(directory.path().join("MANIFEST.fec"), b"orphan sidecar")?;
    std::fs::write(
        directory.path().join("MANIFEST.fec.tmp"),
        b"interrupted manifest sidecar",
    )?;
    let segment_sidecar_temp = directory.path().join(format!(
        "{}.fec.tmp",
        canonical_segment_name(segment.segment_id)
    ));
    std::fs::write(&segment_sidecar_temp, b"interrupted segment sidecar")?;
    std::fs::write(directory.path().join("notes"), b"operator data")?;
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .expect("test clock remains representable");

    let report = collect_abandoned_genesis_garbage_at(
        directory.path(),
        GarbageCollectionOptions::default(),
        now,
    )?;
    let segment_name = canonical_segment_name(segment.segment_id);
    assert_eq!(
        report.removed,
        vec![
            PathBuf::from(".tmp-manifest-1"),
            PathBuf::from(".tmp-write-buffer"),
            PathBuf::from("MANIFEST.fec"),
            PathBuf::from("MANIFEST.fec.tmp"),
            PathBuf::from(segment_name),
            segment_sidecar_temp
                .file_name()
                .expect("sidecar temp has a file name")
                .into(),
        ]
    );
    assert_eq!(
        std::fs::read(directory.path().join("notes"))?,
        b"operator data"
    );
    assert!(
        collect_abandoned_genesis_garbage_at(
            directory.path(),
            GarbageCollectionOptions::default(),
            now,
        )?
        .is_empty()
    );

    let claimed = tempdir()?;
    std::fs::write(claimed.path().join(".tmp-manifest-1"), b"must survive")?;
    std::fs::write(claimed.path().join("gen-1.claim"), [])?;
    let before = directory_bytes(claimed.path())?;
    assert!(matches!(
        collect_abandoned_genesis_garbage_at(
            claimed.path(),
            GarbageCollectionOptions::default(),
            now,
        ),
        Err(KeeperError::ClaimedGenerationPending {
            current: 0,
            claimed: 1,
            ..
        })
    ));
    assert_eq!(directory_bytes(claimed.path())?, before);

    let corrupt = tempdir()?;
    std::fs::write(corrupt.path().join("MANIFEST"), b"corrupt")?;
    std::fs::write(corrupt.path().join(".tmp-old"), b"must survive")?;
    let before = directory_bytes(corrupt.path())?;
    assert!(matches!(
        collect_abandoned_genesis_garbage_at(
            corrupt.path(),
            GarbageCollectionOptions::default(),
            now,
        ),
        Err(KeeperError::ManifestCorrupted { .. })
    ));
    assert_eq!(directory_bytes(corrupt.path())?, before);
    Ok(())
}

#[test]
fn failed_writer_recovery_never_collects_and_path_guard_rejects_escape() -> TestResult {
    let directory = tempdir()?;
    let missing = ManifestSegment {
        segment_id: 0xdead,
        seal_seq: 1,
        file_len: 100,
        file_xxh3: 7,
        docid_lo: 0,
        docid_hi: 2,
        doc_count: 1,
        tombstones: TombstoneSet::new(),
    };
    write_manifest(
        &directory.path().join("MANIFEST"),
        &durable_test_manifest(1, vec![missing]),
    )?;
    std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
    let before = directory_bytes(directory.path())?;
    let now = SystemTime::now()
        .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
        .expect("test clock remains representable");
    assert!(matches!(
        collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        ),
        Err(KeeperError::SegmentOpen { .. })
    ));
    assert_eq!(directory_bytes(directory.path())?, before);

    for unsafe_path in [
        Path::new("../outside"),
        Path::new("nested/file"),
        Path::new("/absolute"),
    ] {
        assert!(matches!(
            safe_direct_child(directory.path(), unsafe_path),
            Err(KeeperError::UnsafeGarbagePath { .. })
        ));
    }
    assert_eq!(
        safe_direct_child(directory.path(), Path::new(".tmp-owned"))?,
        directory.path().join(".tmp-owned")
    );
    assert!(classify_garbage_candidate(OsStr::new("notes.fec")).is_none());
    assert!(classify_garbage_candidate(OsStr::new("notes.fec.tmp")).is_none());
    assert!(classify_garbage_candidate(OsStr::new("SEG-0000000000000001.fslx")).is_none());
    assert!(matches!(
        classify_garbage_candidate(OsStr::new("seg-0000000000000001.fslx.fec.tmp")),
        Some(GarbageCandidate::Temporary)
    ));
    assert!(matches!(
        classify_garbage_candidate(OsStr::new("MANIFEST.fec.tmp")),
        Some(GarbageCandidate::Temporary)
    ));
    Ok(())
}
