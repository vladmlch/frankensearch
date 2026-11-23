use super::*;

#[test]
fn publication_authority_phase_decodes_in_const_context() {
    const PHASES: [PublicationAuthorityPhase; 4] = [
        PublicationAuthorityPhase::from_bits(0),
        PublicationAuthorityPhase::from_bits(1),
        PublicationAuthorityPhase::from_bits(2),
        PublicationAuthorityPhase::from_bits(3),
    ];

    assert_eq!(
        PHASES,
        [
            PublicationAuthorityPhase::Stable,
            PublicationAuthorityPhase::Preparing,
            PublicationAuthorityPhase::DurableAhead,
            PublicationAuthorityPhase::Indeterminate,
        ]
    );
}

#[test]
fn publication_transition_unwind_marks_preparing_authority_indeterminate() {
    let state = PublicationReadState::new(7).expect("pack initial authority generation");
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let state = state.clone();
        move || {
            let _transition = PublicationAuthorityTransition::begin(state, 8)
                .expect("begin publication transition");
            panic!("inject pre-rename publisher unwind");
        }
    }));
    assert!(unwind.is_err(), "the injected closure must unwind");
    let observed = state.load();
    assert_eq!(observed.generation(), 7);
    assert_eq!(observed.phase(), PublicationAuthorityPhase::Indeterminate);
    assert!(
        !observed.is_readable(),
        "an unwound pre-rename closure must never leave its predecessor authoritative"
    );
}

#[test]
fn publication_transition_rejects_an_unrepresentable_successor_before_cas() {
    let generation = PublicationReadState::MAX_GENERATION;
    let state = PublicationReadState::new(generation).expect("pack maximum authority generation");
    let Err(error) = PublicationAuthorityTransition::begin(state.clone(), generation + 1) else {
        panic!("the packed successor generation must be rejected before CAS");
    };
    assert!(matches!(error, KeeperError::InvalidTransition { .. }));
    let observed = state.load();
    assert_eq!(observed.generation(), generation);
    assert_eq!(observed.phase(), PublicationAuthorityPhase::Stable);
}

#[test]
fn publication_stabilize_never_regresses_a_newer_generation() {
    let state = PublicationReadState::new(7).expect("pack initial authority generation");
    state
        .stabilize(9)
        .expect("install the newer process-local successor");

    state
        .stabilize(8)
        .expect("a stale stabilizer is superseded without lowering authority");
    let observed = state.load();
    assert_eq!(observed.generation(), 9);
    assert_eq!(observed.phase(), PublicationAuthorityPhase::Stable);

    let transition = PublicationAuthorityTransition::begin(state.clone(), 10)
        .expect("begin the next publication");
    state
        .stabilize(8)
        .expect("a stale stabilizer must also preserve a newer active publication");
    let observed = state.load();
    assert_eq!(observed.generation(), 9);
    assert_eq!(observed.phase(), PublicationAuthorityPhase::Preparing);
    transition.restore_known_pre_rename_failure();
}

#[test]
fn generic_stabilize_never_clears_same_generation_preparing() {
    let state = PublicationReadState::new(7).expect("pack initial authority generation");
    let transition = PublicationAuthorityTransition::begin(state.clone(), 8)
        .expect("begin publication transition");

    assert!(matches!(
        state.stabilize(7),
        Err(KeeperError::InvalidTransition { .. })
    ));
    assert!(matches!(
        state.stabilize(8),
        Err(KeeperError::InvalidTransition { .. })
    ));
    let observed = state.load();
    assert_eq!(observed.generation(), 7);
    assert_eq!(observed.phase(), PublicationAuthorityPhase::Preparing);
    transition.restore_known_pre_rename_failure();
}

#[test]
fn only_reconciliation_clears_same_generation_indeterminate() {
    let state = PublicationReadState::new(7).expect("pack initial authority generation");
    let transition = PublicationAuthorityTransition::begin(state.clone(), 8)
        .expect("begin publication transition");
    transition.mark_indeterminate();
    drop(transition);

    assert!(matches!(
        state.stabilize(7),
        Err(KeeperError::InvalidTransition { .. })
    ));
    state
        .stabilize_reconciled_same_generation(7)
        .expect("proven reconciliation restores the same generation");
    let observed = state.load();
    assert_eq!(observed.generation(), 7);
    assert_eq!(observed.phase(), PublicationAuthorityPhase::Stable);
}

#[test]
fn reconciliation_never_clears_same_generation_durable_ahead() {
    let state = PublicationReadState::new(7).expect("pack initial authority generation");
    let transition = PublicationAuthorityTransition::begin(state.clone(), 8)
        .expect("begin publication transition");
    transition.mark_durable_ahead();
    drop(transition);

    assert!(matches!(
        state.stabilize(7),
        Err(KeeperError::InvalidTransition { .. })
    ));
    assert!(matches!(
        state.stabilize_reconciled_same_generation(7),
        Err(KeeperError::InvalidTransition { .. })
    ));
    let observed = state.load();
    assert_eq!(observed.generation(), 7);
    assert_eq!(observed.phase(), PublicationAuthorityPhase::DurableAhead);
}

fn tier_test_segment(segment_id: u64, docid_lo: u64, docid_hi: u64) -> ManifestSegment {
    ManifestSegment {
        segment_id,
        seal_seq: segment_id,
        file_len: 1,
        file_xxh3: segment_id,
        docid_lo,
        docid_hi,
        doc_count: u32::try_from(docid_hi - docid_lo).expect("test segment width fits u32"),
        tombstones: TombstoneSet::default(),
    }
}

fn apply_tier_test_plan(
    segments: &mut Vec<ManifestSegment>,
    plan: &TierMergePlan,
    output_segment_id: u64,
) {
    let start = segments
        .iter()
        .position(|segment| segment.segment_id == plan.source_segment_ids[0])
        .expect("planned first source remains live");
    let end = start + plan.source_segment_ids.len();
    assert_eq!(
        segments[start..end]
            .iter()
            .map(|segment| segment.segment_id)
            .collect::<Vec<_>>(),
        plan.source_segment_ids,
        "policy plan must name one uninterrupted manifest slice",
    );
    let doc_count = segments[start..end]
        .iter()
        .map(|segment| segment.doc_count)
        .sum();
    segments.splice(
        start..end,
        [ManifestSegment {
            segment_id: output_segment_id,
            seal_seq: output_segment_id,
            file_len: 1,
            file_xxh3: output_segment_id,
            docid_lo: plan.docid_lo,
            docid_hi: plan.docid_hi,
            doc_count,
            tombstones: TombstoneSet::default(),
        }],
    );
}

fn assert_pairwise_disjoint(segments: &[ManifestSegment]) {
    for pair in segments.windows(2) {
        assert!(
            pair[0].docid_hi <= pair[1].docid_lo,
            "covering intervals overlap: {:?} then {:?}",
            pair[0],
            pair[1],
        );
    }
}

#[test]
fn tier_planner_classifies_widths_and_selects_only_bound_consecutive_runs() {
    let policy = TierMergePolicy {
        fanout: 3,
        small_max_docid_width: 16,
        medium_max_docid_width: 64,
        max_hole_ratio: 0.5,
    };
    assert_eq!(policy.classify_width(16), SegmentSizeTier::Small);
    assert_eq!(policy.classify_width(17), SegmentSizeTier::Medium);
    assert_eq!(policy.classify_width(65), SegmentSizeTier::Large);

    let segments = vec![
        tier_test_segment(1, 0, 4),
        tier_test_segment(2, 6, 10),
        tier_test_segment(3, 12, 16),
        tier_test_segment(4, 80, 112),
    ];
    let plan = plan_tier_merge(&segments, policy)
        .expect("valid policy")
        .expect("three small consecutive sources merge");
    assert_eq!(plan.source_segment_ids, [1, 2, 3]);
    assert_eq!((plan.docid_lo, plan.docid_hi), (0, 16));
    assert!((plan.hole_ratio - 0.25).abs() < f64::EPSILON);

    let rejected = plan_tier_merge(
        &segments,
        TierMergePolicy {
            max_hole_ratio: 0.24,
            ..policy
        },
    )
    .expect("valid restrictive policy");
    assert!(rejected.is_none(), "hole gate must decline the run");

    let mut empty = tier_test_segment(9, 20, 21);
    empty.docid_hi = empty.docid_lo;
    assert!(matches!(
        plan_tier_merge(&[empty], policy),
        Err(TierPolicyError::InvalidSegmentRange { segment_id: 9, .. })
    ));
    assert!(matches!(
        plan_tier_merge(&[segments[1].clone(), segments[0].clone()], policy),
        Err(TierPolicyError::InvalidSegmentOrder {
            left_segment_id: 2,
            right_segment_id: 1,
        })
    ));
}

#[test]
fn tier_policy_bounds_segment_count_under_ten_thousand_watch_batches() {
    let policy = TierMergePolicy {
        fanout: 8,
        small_max_docid_width: 64,
        medium_max_docid_width: 512,
        max_hole_ratio: 0.5,
    };
    let mut segments = Vec::new();
    let mut next_output_id = 10_001_u64;
    for batch in 0..10_000_u64 {
        segments.push(tier_test_segment(batch + 1, batch, batch + 1));
        while let Some(plan) = plan_tier_merge(&segments, policy).expect("valid watch policy") {
            apply_tier_test_plan(&mut segments, &plan, next_output_id);
            next_output_id += 1;
        }
    }
    let tier_count_bound = 3 * (policy.fanout - 1);
    assert!(
        segments.len() <= tier_count_bound,
        "{} live segments exceeded the S/M/L bound {tier_count_bound}",
        segments.len(),
    );
    assert!(
        plan_tier_merge(&segments, policy)
            .expect("valid final policy")
            .is_none(),
        "watch simulation must drain every eligible same-tier run",
    );
    assert_pairwise_disjoint(&segments);
}

#[test]
fn interleaved_multi_shard_random_merge_schedule_preserves_q1_intervals() {
    let policy = TierMergePolicy {
        fanout: 4,
        small_max_docid_width: 65_536,
        medium_max_docid_width: 8 * 65_536,
        max_hole_ratio: 1.0,
    };
    let mut rng = 0x51a7_d15c_0de5_1234_u64;
    let mut next_ord = [0_u64; 4];
    let mut lease_base = [None; 4];
    let mut next_block_base = 0_u64;
    let mut segments = Vec::new();
    let mut next_id = 1_u64;
    for _ in 0..2_000 {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let random = rng.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let shard = usize::try_from(random % 4).expect("shard fits usize");
        let width = 1 + (random >> 8) % 17;
        let base = *lease_base[shard].get_or_insert_with(|| {
            let granted = next_block_base;
            next_block_base += 65_536;
            next_ord[shard] = 0;
            granted
        });
        let lo = base + next_ord[shard];
        let hi = lo + width;
        next_ord[shard] += width;
        segments.push(tier_test_segment(next_id, lo, hi));
        next_id += 1;
        segments.sort_unstable_by_key(|segment| segment.docid_lo);
        assert_pairwise_disjoint(&segments);

        let merge_budget = usize::try_from((random >> 16) % 5).expect("budget fits usize");
        let mut retired_for_merge = false;
        for _ in 0..merge_budget {
            let Some(plan) = plan_tier_merge(&segments, policy).expect("valid random policy")
            else {
                break;
            };
            apply_tier_test_plan(&mut segments, &plan, next_id);
            next_id += 1;
            if !retired_for_merge {
                // Production concat retires every live shard session so a
                // later append cannot land inside the new union hull.
                lease_base.fill(None);
                retired_for_merge = true;
            }
            assert_pairwise_disjoint(&segments);
        }
    }
    while let Some(plan) = plan_tier_merge(&segments, policy).expect("valid drain policy") {
        apply_tier_test_plan(&mut segments, &plan, next_id);
        next_id += 1;
        assert_pairwise_disjoint(&segments);
    }
}

#[test]
fn rank_pruning_cache_caps_unique_terms_and_rejects_dictionary_drift() {
    let cache = RankPruningCache::new();
    let term_metadata =
        TermMetadata::without_positions(1, ByteSpan::new(0, 1), ByteSpan::new(0, 1));
    for term_ord in
        0..u32::try_from(MAX_RANK_PRUNING_CACHE_TERMS + 16).expect("cache test count fits u32")
    {
        cache
            .insert(
                term_ord,
                term_metadata,
                Arc::new(ValidatedTermPruningMetadata::empty_for_cache_test()),
            )
            .expect("bounded cache admission");
    }
    assert_eq!(cache.len(), MAX_RANK_PRUNING_CACHE_TERMS);
    assert!(
        cache
            .get(
                u32::try_from(MAX_RANK_PRUNING_CACHE_TERMS).expect("cache limit fits u32"),
                term_metadata,
            )
            .expect("bounded cache lookup")
            .is_none(),
        "terms beyond the cap must remain query-local"
    );

    let drifted = TermMetadata::without_positions(1, ByteSpan::new(1, 1), ByteSpan::new(0, 1));
    assert!(cache.get(0, drifted).is_err());
}

#[test]
fn exact_lexical_snapshot_receipt_binds_docset_checkpoint_and_descriptor_reopen() -> TestResult {
    let index = tempdir()?;
    let directory = index.path().to_path_buf();
    let encoded = encoded_identity_test_segment(
        0x0e1a_c7a1,
        0,
        &[Some("receipt-a"), None, Some("receipt-b")],
    )?;
    let manifest = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
    let publish_directory = directory.clone();
    run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &publish_directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let pending = encoded
            .write_temp(&publish_directory)
            .map_err(|error| error.to_string())?;
        writer
            .publish_segment(&cx, pending)
            .await
            .map_err(|error| error.to_string())?;
        writer
            .publish(&cx, &manifest)
            .await
            .map_err(|error| error.to_string())?;
        Ok::<(), String>(())
    })
    .map_err(io::Error::other)?;

    let snapshot = KeeperSnapshot::open(&directory, DEFAULT_SCHEMA)?;
    let descriptor =
        snapshot.exact_lexical_snapshot_descriptor(SourceCheckpointV1::derive(&CommitRange {
            low: 1,
            high: 7,
        }))?;
    let receipt = descriptor.receipt();
    assert_eq!(receipt.role, GenerationComponentRole::Lexical);
    assert_eq!(receipt.live_document_count, 2);
    assert_eq!(
        receipt.source_checkpoint,
        SourceCheckpointV1::derive(&CommitRange { low: 1, high: 7 }).to_bytes()
    );
    let expected_docset =
        CanonicalDocsetV1::from_ordered_live_documents(["receipt-a", "receipt-b"])?;
    assert_eq!(receipt.docset_digest, expected_docset.digest());
    assert_ne!(receipt.bytes.sha256, [0; 32]);

    let reopened = descriptor.reopen()?;
    assert_eq!(
        reopened.exact_lexical_component_receipt(SourceCheckpointV1::derive(&CommitRange {
            low: 1,
            high: 7
        }))?,
        *receipt,
        "paired control: descriptor-backed reopen must reproduce every receipt field"
    );
    let other_checkpoint =
        snapshot.exact_lexical_component_receipt(SourceCheckpointV1::derive(&CommitRange {
            low: 1,
            high: 8,
        }))?;
    assert_eq!(other_checkpoint.docset_digest, receipt.docset_digest);
    assert_eq!(other_checkpoint.bytes, receipt.bytes);
    assert_ne!(
        other_checkpoint.source_checkpoint,
        receipt.source_checkpoint
    );

    let advance_directory = directory.clone();
    run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::open(&cx, &advance_directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let next = writer
            .retained_snapshot_for_bookkeeping()
            .next_manifest()
            .map_err(|error| error.to_string())?;
        writer
            .publish(&cx, &next)
            .await
            .map_err(|error| error.to_string())?;
        Ok::<(), String>(())
    })
    .map_err(io::Error::other)?;
    assert!(matches!(
        descriptor.reopen(),
        Err(KeeperError::ExactSnapshotReceipt { .. })
    ));
    Ok(())
}

#[cfg(unix)]
#[test]
fn a_published_snapshot_reuses_only_segments_its_preflight_verified_in_place() -> TestResult {
    let index = tempdir()?;
    let directory = index.path().to_path_buf();
    let manifest = publish_fixture_segment(&directory, 0x0e1a_5e01)?;
    let preflight = || {
        validate_proposed_manifest_segments(
            &directory,
            &manifest,
            DEFAULT_SCHEMA,
            &WriterProtection::Disabled,
        )
    };

    // The same binding over the same file: the preflight's bound segment
    // becomes the snapshot's, and its bytes are not hashed again.
    let verified = preflight()?;
    let backing = Arc::clone(&verified[0].segment.reader);
    let snapshot = KeeperSnapshot::open_once_reusing(&directory, DEFAULT_SCHEMA, verified)?;
    assert!(Arc::ptr_eq(&snapshot.segments()[0].reader, &backing));

    // A MANIFEST that binds the segment differently is verified again.
    let mut verified = preflight()?;
    verified[0].segment.manifest.seal_seq += 1;
    let backing = Arc::clone(&verified[0].segment.reader);
    let snapshot = KeeperSnapshot::open_once_reusing(&directory, DEFAULT_SCHEMA, verified)?;
    assert!(!Arc::ptr_eq(&snapshot.segments()[0].reader, &backing));
    assert_eq!(snapshot.segments()[0].manifest(), &manifest.segments[0]);

    // The same bytes behind a new inode are verified again, never trusted
    // through the preflight's mapping of the file they replaced.
    let verified = preflight()?;
    let backing = Arc::clone(&verified[0].segment.reader);
    let path = directory.join(canonical_segment_name(manifest.segments[0].segment_id));
    let replacement = directory.join(".tmp-replacement");
    std::fs::copy(&path, &replacement)?;
    std::fs::rename(&replacement, &path)?;
    let snapshot = KeeperSnapshot::open_once_reusing(&directory, DEFAULT_SCHEMA, verified)?;
    assert!(!Arc::ptr_eq(&snapshot.segments()[0].reader, &backing));
    Ok(())
}

#[cfg(unix)]
#[test]
fn a_retained_snapshot_that_no_longer_matches_disk_refuses_the_sweep_and_the_witness() -> TestResult
{
    let index = tempdir()?;
    let directory = index.path().to_path_buf();
    publish_fixture_segment(&directory, 0x0e1a_5e02)?;
    let stale = KeeperSnapshot::open(&directory, DEFAULT_SCHEMA)?;
    let advance_directory = directory.clone();
    run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::open(&cx, &advance_directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let next = writer
            .retained_snapshot_for_bookkeeping()
            .next_manifest()
            .map_err(|error| error.to_string())?;
        writer
            .publish(&cx, &next)
            .await
            .map_err(|error| error.to_string())?;
        Ok::<(), String>(())
    })
    .map_err(io::Error::other)?;

    let before = directory_bytes(&directory)?;
    let options = GarbageCollectionOptions {
        grace_period: Duration::ZERO,
    };
    assert!(matches!(
        collect_writer_garbage_with_snapshot(&directory, &stale, options),
        Err(KeeperError::GarbageDirectoryChanged { .. })
    ));
    assert!(matches!(
        witness_orphaned_segments_with_snapshot(&directory, &stale),
        Err(KeeperError::GarbageDirectoryChanged { .. })
    ));
    assert_eq!(directory_bytes(&directory)?, before);
    let current = KeeperSnapshot::open(&directory, DEFAULT_SCHEMA)?;
    collect_writer_garbage_with_snapshot(&directory, &current, options)?;
    Ok(())
}

#[test]
fn referenced_segment_unknown_schema_maps_to_public_corruption() {
    let path = PathBuf::from("index/seg-0000000000000001.fslx");
    let error: SearchError = KeeperError::SegmentOpen {
        path: path.clone(),
        source: QuillError::UnknownSchema { schema_id: 0x55 },
    }
    .into();
    assert!(matches!(
        error,
        SearchError::IndexCorrupted { path: actual, detail }
            if actual == path && detail.contains("unknown schema")
    ));
}
