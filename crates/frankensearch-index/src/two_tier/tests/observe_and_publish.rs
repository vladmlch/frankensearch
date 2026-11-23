use super::*;

/// Byte-for-byte fixture snapshot of one artifact and (optionally) its
/// WAL sidecar, for observation-invariance assertions.
fn tier_snapshot(path: &Path) -> (Vec<u8>, Option<Vec<u8>>) {
    let main = fs::read(path).expect("read main artifact");
    let wal_path = crate::wal::wal_path_for(path);
    let wal = wal_path
        .exists()
        .then(|| fs::read(&wal_path).expect("read wal"));
    (main, wal)
}

#[test]
fn observe_tier_classifies_v1_seed_live_and_v2_without_mutation() {
    let dir = temp_index_dir("observe-basic");
    fs::create_dir_all(&dir).expect("create temp dir");

    // Empty v1 seed: no content retained.
    let seed_path = dir.join("seed.idx");
    VectorIndex::create(&seed_path, "seed-model", 4)
        .expect("create seed")
        .finish()
        .expect("finish seed");
    let seed_snapshot = tier_snapshot(&seed_path);
    let seed_observation = observe_tier(&seed_path).expect("observe seed");
    assert!(
        matches!(seed_observation, FsviTierObservation::V1(_)),
        "v1 seed must observe as V1, got {seed_observation:?}"
    );
    let FsviTierObservation::V1(seed) = seed_observation else {
        return;
    };
    assert_eq!(seed.record_count, 0);
    assert_eq!(seed.active_wal_records, 0);
    assert!(!seed.wal_sidecar_present);
    assert!(!seed.retains_content());
    assert_eq!(tier_snapshot(&seed_path), seed_snapshot);

    // Live v1 with a fresh WAL append: content retained via both signals.
    let live_path = dir.join("live.idx");
    write_index_file(&live_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write live v1");
    {
        let mut live = VectorIndex::open(&live_path).expect("open live v1");
        live.append("doc-wal", &[0.0, 1.0, 0.0, 0.0])
            .expect("append WAL resident");
    }
    let live_snapshot = tier_snapshot(&live_path);
    assert!(live_snapshot.1.is_some(), "fixture must have a WAL sidecar");
    let live_observation = observe_tier(&live_path).expect("observe live");
    assert!(
        matches!(live_observation, FsviTierObservation::V1(_)),
        "live v1 must observe as V1, got {live_observation:?}"
    );
    let FsviTierObservation::V1(live) = live_observation else {
        return;
    };
    assert_eq!(live.record_count, 1);
    assert_eq!(live.active_wal_records, 1);
    assert!(live.wal_sidecar_present);
    assert!(live.retains_content());
    assert_eq!(
        tier_snapshot(&live_path),
        live_snapshot,
        "observation must not rewrite the artifact or its WAL"
    );

    // v2: header-only recognition.
    let v2_path = dir.join("observed.v2.idx");
    let (binding, _) = fsvi_v2_binding("observe-v2-model", 4, 2);
    write_v2_tier(&v2_path, &binding, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]);
    let v2_snapshot = tier_snapshot(&v2_path);
    let v2_observation = observe_tier(&v2_path).expect("observe v2");
    assert!(
        matches!(v2_observation, FsviTierObservation::V2IdentityComplete(_)),
        "v2 tier must observe as V2IdentityComplete, got {v2_observation:?}"
    );
    let FsviTierObservation::V2IdentityComplete(metadata) = v2_observation else {
        return;
    };
    assert_eq!(metadata.record_count, 1);
    assert!(metadata.identity_v2.is_some());
    assert_eq!(tier_snapshot(&v2_path), v2_snapshot);

    let _ = fs::remove_dir_all(&dir);
}

/// NO-GO item 2 repair, stale-WAL half: `VectorIndex::open` DELETES a
/// stale WAL sidecar; read-only observation must classify it as inactive
/// while leaving the file byte-identical.
#[test]
fn observe_tier_never_deletes_a_stale_wal_sidecar() {
    let dir = temp_index_dir("observe-stale-wal");
    fs::create_dir_all(&dir).expect("create temp dir");

    // Donor: advance the compaction generation once, then append, so the
    // donor WAL header carries generation 2 (= next(1)).
    let donor_path = dir.join("donor.idx");
    write_index_file(&donor_path, &[("donor-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write donor");
    {
        let mut donor = VectorIndex::open(&donor_path).expect("open donor");
        donor
            .append("donor-wal-1", &[0.0, 1.0, 0.0, 0.0])
            .expect("append pre-compaction");
        donor.compact().expect("compact to bump generation");
        donor
            .append("donor-wal-2", &[0.0, 0.0, 1.0, 0.0])
            .expect("append post-compaction");
    }
    let donor_wal = crate::wal::wal_path_for(&donor_path);
    assert!(donor_wal.exists(), "donor WAL must exist");

    // Target: generation-0 main slab with a live row; the donor WAL's
    // generation cannot match next(0), so it reads as STALE here.
    let target_path = dir.join("target.idx");
    write_index_file(&target_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write target");
    let target_wal = crate::wal::wal_path_for(&target_path);
    fs::copy(&donor_wal, &target_wal).expect("transplant stale WAL");

    // Precondition: the transplanted WAL really is stale for the target.
    let (entries, wal_gen, valid_len) =
        crate::wal::read_wal(&target_wal, 4, Quantization::F16).expect("read fixture WAL");
    assert!(
        !entries.is_empty() && valid_len > 0,
        "fixture WAL has entries"
    );
    assert_ne!(
        wal_gen,
        crate::next_generation(0),
        "fixture WAL generation must mismatch the target main slab"
    );

    let snapshot = tier_snapshot(&target_path);
    let target_observation = observe_tier(&target_path).expect("observe target");
    assert!(
        matches!(target_observation, FsviTierObservation::V1(_)),
        "target must observe as V1, got {target_observation:?}"
    );
    let FsviTierObservation::V1(observation) = target_observation else {
        return;
    };
    assert_eq!(
        observation.active_wal_records, 0,
        "stale WAL entries belong to a dead generation"
    );
    assert!(observation.wal_sidecar_present);
    assert!(
        observation.retains_content(),
        "the live main row retains content regardless of WAL staleness"
    );
    assert!(
        target_wal.exists(),
        "read-only observation must NEVER delete a stale WAL (VectorIndex::open does)"
    );
    assert_eq!(
        tier_snapshot(&target_path),
        snapshot,
        "artifact and WAL must be byte-identical after observation"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// NO-GO item 2 repair, corrupt-trailer half: `VectorIndex::open`
/// TRUNCATES a corrupt WAL trailer; read-only observation must count the
/// valid prefix and leave the trailer bytes in place.
#[test]
fn observe_tier_never_truncates_a_corrupt_wal_trailer() {
    let dir = temp_index_dir("observe-corrupt-trailer");
    fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("trailer.idx");
    write_index_file(&path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write v1");
    {
        let mut index = VectorIndex::open(&path).expect("open v1");
        index
            .append("doc-wal", &[0.0, 1.0, 0.0, 0.0])
            .expect("append valid WAL entry");
    }
    let wal_path = crate::wal::wal_path_for(&path);
    let clean_len = fs::metadata(&wal_path).expect("stat wal").len();
    {
        use std::io::Write as _;
        let mut wal_file = fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .expect("open wal for corruption");
        wal_file
            .write_all(&[0xAB; 32])
            .expect("append corrupt trailer");
    }
    let corrupted_len = fs::metadata(&wal_path).expect("stat wal").len();
    assert_eq!(corrupted_len, clean_len + 32);

    let snapshot = tier_snapshot(&path);
    let trailer_observation = observe_tier(&path).expect("observe");
    assert!(
        matches!(trailer_observation, FsviTierObservation::V1(_)),
        "must observe as V1, got {trailer_observation:?}"
    );
    let FsviTierObservation::V1(observation) = trailer_observation else {
        return;
    };
    assert_eq!(
        observation.active_wal_records, 1,
        "the valid prefix still counts"
    );
    assert_eq!(
        fs::metadata(&wal_path).expect("re-stat wal").len(),
        corrupted_len,
        "read-only observation must NEVER truncate a corrupt trailer \
         (VectorIndex::open does)"
    );
    assert_eq!(tier_snapshot(&path), snapshot);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn observe_tier_reports_upgrade_required_for_future_schema() {
    let dir = temp_index_dir("observe-upgrade");
    fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("future.idx");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&crate::FSVI_MAGIC);
    bytes.extend_from_slice(&3_u16.to_le_bytes());
    bytes.extend_from_slice(&[0_u8; 16]);
    fs::write(&path, &bytes).expect("write future-schema fixture");
    let observation = observe_tier(&path).expect("observe future schema");
    assert!(
        matches!(
            observation,
            FsviTierObservation::UpgradeRequired(FsviUpgradeRequired {
                found_version: 3,
                supported_version: crate::FSVI_V2_VERSION,
            })
        ),
        "got {observation:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Write-side WAL hygiene: a builder rewrite of a tier removes the
/// now-dead adjacent WAL sidecar so it can never resurrect foreign rows
/// into the new generation (read-only classification deliberately leaves
/// stale sidecars in place; the WRITE path owns their cleanup).
#[test]
fn builder_finish_removes_adjacent_wal_sidecars() {
    let dir = temp_index_dir("builder-wal-hygiene");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("old-doc", &[1.0, 0.0, 0.0, 0.0])])
        .expect("write prior generation");
    {
        let mut prior = VectorIndex::open(&fast_path).expect("open prior");
        prior
            .append("wal-resident", &[0.0, 1.0, 0.0, 0.0])
            .expect("append WAL resident");
    }
    let wal_path = crate::wal::wal_path_for(&fast_path);
    assert!(wal_path.exists(), "fixture WAL must exist before rebuild");

    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_fast_record("new-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("add rebuild row");
    let rebuilt = builder.finish().expect("finish rebuild");

    assert!(
        !wal_path.exists(),
        "the rewritten tier's dead WAL sidecar must be removed by the write path"
    );
    assert_eq!(rebuilt.doc_count(), 1);
    let ids: Vec<String> = rebuilt.iter_doc_ids().filter_map(Result::ok).collect();
    assert_eq!(ids, vec!["new-doc".to_owned()]);
    assert!(
        !ids.iter().any(|id| id == "wal-resident" || id == "old-doc"),
        "no row from the replaced generation may leak into the rebuild: {ids:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// Directory entries under `dir` whose names carry the sibling staging
/// marker (`.tmp.<pid>.<nanos>`), for asserting staged replacements
/// never leak.
fn staging_leftovers(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .expect("read dir")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp."))
        .collect();
    names.sort();
    names
}

/// r3 fault tests (ii) + (iv) — the crash-after-main-rename state under
/// the OLD r2 ordering, and the generation-2 collision reopen
/// (bd-9xuj C4-write r3; audits #8366/#8367).
///
/// Constructs on disk exactly what a crash between r2's
/// `writer.finish()` rename and its best-effort WAL removal left:
/// a fresh generation-1 main REPLACED over the canonical name with the
/// old generation's generation-2 WAL still adjacent. Pins, exactly:
/// - the observer reports the sidecar PRESENT and its rows ACTIVE
///   (`next_generation(1) == 2` — byte-indistinguishable from
///   legitimate incremental appends), so `retains_content()` holds and
///   observation-driven admission seams fail closed on this state
///   instead of consuming it (the refresh-side refusal is pinned in
///   `frankensearch-fusion/src/refresh.rs`);
/// - the plain v1 reopen REPLAYS the foreign row (the frozen
///   `VectorIndex::open` v1 contract; `lib.rs` is not editable in this
///   train) — this resurrection is precisely why the r3 protocol makes
///   the state unconstructable, and the two-tier open path now surfaces
///   the replay with a warning rather than silence;
/// - `finish()` over ANY WAL-bearing directory exits with
///   {new main, no WAL}: `publish_tier` installs an authority-carrying
///   successor generation (bd-zhjv8) whose rename atomically invalidates
///   the destination's sidecar, so the hazard state {new main + ADOPTABLE
///   old WAL} is unreachable through `finish()` at every interruption
///   point.
#[test]
fn crash_after_rename_state_is_pinned_and_unconstructable_through_finish() {
    let dir = temp_index_dir("r3-crash-after-rename");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);

    // Old generation: fresh legacy main (compaction generation 1) plus
    // one incremental append, which creates a generation-2 WAL sidecar.
    write_index_file(&fast_path, &[("old-doc", &[1.0, 0.0, 0.0, 0.0])])
        .expect("write old generation");
    {
        let mut old = VectorIndex::open(&fast_path).expect("open old generation");
        old.append("foreign-doc", &[0.0, 1.0, 0.0, 0.0])
            .expect("append WAL resident");
    }
    let wal_path = crate::wal::wal_path_for(&fast_path);
    let (entries, wal_gen, valid_len) =
        crate::wal::read_wal(&wal_path, 4, Quantization::F16).expect("read fixture WAL");
    assert_eq!(entries.len(), 1, "fixture WAL carries the foreign row");
    assert!(valid_len > 0);
    assert_eq!(
        wal_gen,
        crate::next_generation(1),
        "fixture WAL generation must be 2 = next(1): the exact collision"
    );

    // Simulate r2's crash point: a replacement generation-1 main is
    // renamed over the canonical name; the WAL removal never ran.
    let scratch = dir.join("replacement.scratch");
    write_index_file(&scratch, &[("new-doc", &[0.0, 0.0, 1.0, 0.0])])
        .expect("write replacement main");
    fs::rename(&scratch, &fast_path).expect("simulate crash after rename");
    assert!(wal_path.exists(), "crash state: gen-2 WAL still adjacent");

    // Pin (iv), observer half: the sidecar is REPORTED, its rows count
    // as ACTIVE against the fresh generation-1 main, and the state
    // therefore reads as content-retaining — the conservative signal
    // admission seams refuse on. Nothing is silently dropped and
    // nothing on disk is touched.
    let observation = observe_tier(&fast_path).expect("observe crash state");
    assert!(
        matches!(observation, FsviTierObservation::V1(_)),
        "crash state must observe as V1, got {observation:?}"
    );
    let FsviTierObservation::V1(observed) = observation else {
        return;
    };
    assert_eq!(observed.record_count, 1);
    assert_eq!(
        observed.active_wal_records, 1,
        "a generation-2 WAL is byte-indistinguishable from legitimate \
         incremental appends against a generation-1 main"
    );
    assert!(observed.wal_sidecar_present);
    assert!(observed.retains_content());
    assert!(wal_path.exists(), "observation never deletes the sidecar");

    // Pin (iv), reopen half: the plain v1 open REPLAYS the foreign row
    // (frozen `VectorIndex::open` contract). This is the resurrection
    // the r3 ordering exists to make unmanufacturable; the two-tier
    // reopen surfaces it via `warn_if_wal_rows_replayed` rather than
    // silence.
    {
        let reopened = VectorIndex::open(&fast_path).expect("v1 reopen of crash state");
        assert_eq!(reopened.record_count(), 1, "main slab rows");
        assert_eq!(
            reopened.wal_record_count(),
            1,
            "the foreign WAL row IS replayed by the frozen v1 open path"
        );
        let wal_ids: Vec<&str> = reopened.wal_records().map(|(id, _)| id).collect();
        assert_eq!(
            wal_ids,
            vec!["foreign-doc"],
            "the replayed resident row is exactly the foreign append"
        );
    }
    let two_tier = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("two-tier reopen");
    assert_eq!(
        two_tier.doc_count(),
        1,
        "the doc-id surface counts the main slab only"
    );
    let hits = two_tier
        .search_fast(&[0.0, 0.8, 0.6, 0.0], 4)
        .expect("search crash state");
    let hit_ids: Vec<&str> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
    assert_eq!(
        hit_ids,
        vec!["foreign-doc", "new-doc"],
        "pinned: the legacy crash state resurrects the foreign row \
         through the SEARCH surface on plain reopen — unreachable \
         through finish() after r3"
    );
    drop(two_tier);

    // Pin (ii): the NEW protocol cannot leave this state. A full
    // rebuild over the WAL-bearing directory publishes a successor
    // generation whose rename invalidates the sidecar, retires it after
    // the authority switch, and exits with {new main, no WAL}.
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .add_fast_record("rebuilt-doc", &[0.5, 0.5, 0.0, 0.0])
        .expect("add rebuild row");
    let rebuilt = builder.finish().expect("rebuild over crash state");
    assert!(
        !wal_path.exists(),
        "finish() must never exit with a published main and a live WAL"
    );
    let ids: Vec<String> = rebuilt.iter_doc_ids().filter_map(Result::ok).collect();
    assert_eq!(ids, vec!["rebuilt-doc".to_owned()]);
    assert_eq!(
        staging_leftovers(&dir),
        Vec::<String>::new(),
        "no staged replacement survives a successful finish()"
    );

    let _ = fs::remove_dir_all(&dir);
}
