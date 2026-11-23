use super::*;

#[test]
fn composite_snapshot_uses_keeper_at_seal_and_live_delta_statistics() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let sealed_documents = [
            IndexableDocument::new("sealed-live", "alpha"),
            IndexableDocument::new("sealed-deleted", "alpha"),
        ];
        index
            .index_documents(&cx, &sealed_documents)
            .await
            .expect("accumulate sealed fixtures");
        index.commit(&cx).await.expect("commit sealed fixtures");

        let committed = index
            .snapshot()
            .expect("sealed upsert snapshot is authoritative")
            .clone();
        let mut tombstoned_manifest = committed.next_manifest().expect("next manifest");
        assert!(
            committed
                .delete_document(&mut tombstoned_manifest, "sealed-deleted")
                .expect("stage sealed delete")
        );
        let tombstoned = committed
            .publish_owned_segments(&tombstoned_manifest, Vec::new())
            .expect("publish tombstone-only successor");
        assert_eq!(tombstoned.at_seal_doc_count(), 2);
        assert_eq!(tombstoned.doc_count(), 1);
        assert_eq!(
            snapshot_doc_freq(&tombstoned, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
                .expect("sealed alpha df"),
            2,
            "sealed tombstones remain in BM25 df until compaction"
        );

        let generation = tombstoned.loaded_manifest().manifest.generation;
        let first_base = u64::from(DOC_ORDS_PER_LEASE);
        let second_base = first_base * 2;
        let mut first =
            DeltaSegment::new(DEFAULT_SCHEMA, first_base, usize::MAX).expect("first Delta shard");
        apply_alpha_delta(
            &mut first,
            u32::try_from(first_base).expect("first Delta docid"),
            "delta-a",
            2,
        );
        let mut second =
            DeltaSegment::new(DEFAULT_SCHEMA, second_base, usize::MAX).expect("second Delta shard");
        apply_alpha_delta(
            &mut second,
            u32::try_from(second_base).expect("second Delta docid"),
            "delta-b",
            3,
        );

        let snapshot = QuillSearchSnapshot::compose(
            11,
            Arc::new(tombstoned),
            vec![
                Arc::new(first.freeze(generation)),
                Arc::new(second.freeze(generation)),
            ],
        )
        .expect("compose Keeper plus Delta snapshot");
        assert_eq!(snapshot.snapshot_epoch(), 11);
        assert_eq!(snapshot.bm25_doc_count(), 4);
        assert_eq!(snapshot.live_doc_count(), 3);
        assert_eq!(
            snapshot
                .bm25_field_stats(CONTENT_FIELD)
                .expect("content stats"),
            SnapshotFieldStats {
                field_ord: CONTENT_FIELD,
                total_tokens: 7,
                doc_count: 4,
            }
        );
        assert_eq!(
            snapshot
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("composite alpha df"),
            4
        );
    });
}

#[test]
fn three_delta_upserts_contribute_one_live_bm25_row() {
    run_with_cx(|cx| async move {
        let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        apply_alpha_delta(&mut delta, 0, "same-id", 1);
        apply_alpha_delta(&mut delta, 1, "same-id", 2);
        apply_alpha_delta(&mut delta, 2, "same-id", 3);
        assert_eq!(delta.physical_document_count(), 3);
        assert_eq!(delta.live_document_count(), 1);
        assert_eq!(
            delta
                .find_term(CONTENT_FIELD, b"alpha")
                .expect("alpha Delta term")
                .live_doc_freq(),
            1
        );

        let frozen = Arc::new(delta.freeze(generation));
        let snapshot = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
            .expect("compose repeated-upsert snapshot");
        assert_eq!(alpha_snapshot_tuple(&snapshot), (0, 1, 1, 3, 1));
        let pre_stats = snapshot
            .bm25_field_stats(CONTENT_FIELD)
            .expect("pre-seal content stats");
        let pre_df = snapshot
            .bm25_doc_freq(CONTENT_FIELD, b"alpha")
            .expect("pre-seal alpha df");
        let mut pre_scorer = lower_delta_term(&frozen, &snapshot, CONTENT_FIELD, b"alpha", 1.0)
            .expect("pre-seal Delta term scorer");
        assert_eq!(pre_scorer.doc(), Some(2));

        let sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
        sealed
            .index_documents(
                &cx,
                &[IndexableDocument::new("same-id", "alpha alpha alpha")],
            )
            .await
            .expect("accumulate final logical upsert row");
        sealed
            .commit(&cx)
            .await
            .expect("seal final logical upsert row");
        let keeper = sealed
            .snapshot()
            .expect("sealed upsert snapshot is authoritative");
        let post_stats = snapshot_field(&keeper, CONTENT_FIELD).expect("post-seal stats");
        let post_df = snapshot_doc_freq(&keeper, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
            .expect("post-seal alpha df");
        assert_eq!(pre_stats, post_stats);
        assert_eq!(pre_df, post_df);

        let mut post_scorer = lower_term(
            &keeper.segments()[0],
            &keeper,
            DEFAULT_SCHEMA,
            CONTENT_FIELD,
            b"alpha",
            1.0,
        )
        .expect("post-seal term scorer");
        assert_eq!(
            pre_scorer.score().expect("pre-seal score").to_bits(),
            post_scorer.score().expect("post-seal score").to_bits(),
            "the final live upsert must keep bit-identical BM25 across seal"
        );
    });
}

#[test]
fn delta_cursor_matches_sealed_advance_positions_and_safe_impact_bound() {
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
    apply_alpha_delta(&mut delta, 0, "replaced", 300);
    apply_alpha_delta(&mut delta, 1, "kept", 2);
    apply_alpha_delta(&mut delta, 2, "replaced", 3);
    apply_alpha_delta(&mut delta, 3, "deleted", 4);
    assert_eq!(delta.delete_delta_id("deleted"), Some(3));
    let frozen = delta.freeze(0);

    let postings = [Posting::new(1, 2), Posting::new(2, 3)];
    let encoded_postings = EncodedPostingList::encode(&postings).expect("sealed postings");
    let posting_list = encoded_postings
        .posting_list()
        .expect("sealed posting list");
    let encoded_positions =
        EncodedPositionList::encode(&postings, &[0, 1, 0, 1, 2]).expect("sealed positions");
    let position_list = encoded_positions
        .position_list(&posting_list)
        .expect("sealed position list");
    let mut sealed = crate::argus::SealedPostingCursor::with_positions(&position_list, 2)
        .expect("sealed cursor");
    let mut live =
        DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"alpha").expect("live Delta cursor");
    let live_norms = DeltaFieldNorms::new(&frozen, CONTENT_FIELD);

    assert_eq!(live.size_hint(), 2);
    assert_eq!(live.cost(), 2);
    assert_eq!(live.segment_num_docs(), 2);
    assert_eq!(live_norms.fieldnorm_id(0), None);
    assert_eq!(live_norms.fieldnorm_id(1), Some(fieldnorm_to_id(2)));
    assert_eq!(live.doc(), sealed.doc());
    assert_eq!(live.freq(), sealed.freq());
    let mut delta_positions = Vec::new();
    let mut sealed_positions = Vec::new();
    live.positions_handle()
        .expect("Delta positions")
        .decode_into(&mut delta_positions)
        .expect("decode Delta positions");
    sealed
        .positions_handle()
        .expect("sealed positions")
        .decode_into(&mut sealed_positions)
        .expect("decode sealed positions");
    assert_eq!(delta_positions, sealed_positions);

    for target in [0, 1, 2, 2] {
        assert_eq!(
            live.advance(target).expect("advance Delta cursor"),
            sealed.advance(target).expect("advance sealed cursor")
        );
        assert_eq!(live.doc(), sealed.doc());
        assert_eq!(live.freq(), sealed.freq());
    }
    live.positions_handle()
        .expect("advanced Delta positions")
        .decode_into(&mut delta_positions)
        .expect("decode advanced Delta positions");
    sealed
        .positions_handle()
        .expect("advanced sealed positions")
        .decode_into(&mut sealed_positions)
        .expect("decode advanced sealed positions");
    assert_eq!(delta_positions, sealed_positions);
    assert_eq!(
        live.next().expect("exhaust Delta cursor"),
        sealed.next().expect("exhaust sealed cursor")
    );
    assert_eq!(live.next().expect("fused Delta cursor"), None);
    assert_eq!(
        live.advance(u32::MAX)
            .expect("advance exhausted Delta cursor"),
        None
    );

    let mut missing = DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"missing")
        .expect("missing-term Delta cursor");
    assert_eq!(missing.doc(), None);
    assert_eq!(missing.freq(), None);
    assert!(missing.positions_handle().is_none());
    assert_eq!(missing.size_hint(), 0);
    assert_eq!(missing.cost(), 0);
    assert_eq!(missing.segment_num_docs(), 2);
    assert_eq!(missing.next().expect("fused missing-term cursor"), None);

    let mut all_dead =
        DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("all-dead Delta shard");
    apply_alpha_delta(&mut all_dead, 0, "dead", 1);
    assert_eq!(all_dead.delete_delta_id("dead"), Some(0));
    let all_dead = all_dead.freeze(0);
    let mut all_dead_cursor =
        DeltaPostingCursor::new(&all_dead, CONTENT_FIELD, b"alpha").expect("all-dead term cursor");
    assert_eq!(all_dead_cursor.doc(), None);
    assert_eq!(all_dead_cursor.size_hint(), 0);
    assert_eq!(all_dead_cursor.cost(), 0);
    assert_eq!(all_dead_cursor.next().expect("fused all-dead cursor"), None);
    assert!(
        all_dead
            .find_term(CONTENT_FIELD, b"alpha")
            .expect("physical all-dead term")
            .block_max()
            .is_none()
    );

    let impact = frozen
        .find_term(CONTENT_FIELD, b"alpha")
        .expect("alpha term")
        .block_max()
        .expect("live term impact bound");
    assert_eq!(impact.max_frequency(), u32::MAX);
    for average in [0.25_f32, 2.5, 1_000.0] {
        let bound = impact
            .score_upper_bound(average, 1.0)
            .expect("finite non-negative score bound");
        let cache = crate::contract::compute_tf_cache(average);
        for (frequency, fieldnorm_id) in [(2_u32, fieldnorm_to_id(2)), (3_u32, fieldnorm_to_id(3))]
        {
            let frequency = frequency as f32;
            let score = frequency / (frequency + cache[usize::from(fieldnorm_id)]);
            assert!(
                bound >= score,
                "avgdl={average} bound={bound} live_score={score}"
            );
        }
    }
    assert_eq!(impact.score_upper_bound(2.5, -1.0), None);
}

#[test]
fn delta_advance_matches_sealed_cursor_across_multiblock_tombstone_boundaries() {
    const TOMBSTONED: [u32; 10] = [0, 1, 2, 63, 64, 127, 128, 129, 258, 259];
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
    let mut sealed_postings = Vec::new();
    let mut sealed_positions = Vec::new();
    for docid in 0_u32..260 {
        let frequency = docid % 5 + 1;
        let document_id = format!("doc-{docid}");
        apply_alpha_delta(&mut delta, docid, &document_id, frequency);
        if !TOMBSTONED.contains(&docid) {
            sealed_postings.push(Posting::new(docid, frequency));
            sealed_positions.extend(0..frequency);
        }
    }
    for docid in TOMBSTONED {
        assert_eq!(delta.delete_delta_id(&format!("doc-{docid}")), Some(docid));
    }
    let frozen = delta.freeze(0);
    let encoded_postings = EncodedPostingList::encode(&sealed_postings).expect("sealed postings");
    let posting_list = encoded_postings
        .posting_list()
        .expect("sealed posting list");
    let encoded_positions =
        EncodedPositionList::encode(&sealed_postings, &sealed_positions).expect("sealed positions");
    let position_list = encoded_positions
        .position_list(&posting_list)
        .expect("sealed position list");
    let segment_num_docs = u32::try_from(sealed_postings.len()).expect("fixture fits u32");

    for target in (0_u32..=260).chain(std::iter::once(u32::MAX)) {
        let mut live =
            DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"alpha").expect("Delta cursor");
        let mut sealed =
            crate::argus::SealedPostingCursor::with_positions(&position_list, segment_num_docs)
                .expect("sealed cursor");
        assert_eq!(
            live.advance(target).expect("advance Delta cursor"),
            sealed.advance(target).expect("advance sealed cursor"),
            "target={target}"
        );
        assert_eq!(live.freq(), sealed.freq(), "target={target}");
        if live.doc().is_some() {
            let mut live_positions = Vec::new();
            let mut oracle_positions = Vec::new();
            live.positions_handle()
                .expect("Delta position handle")
                .decode_into(&mut live_positions)
                .expect("decode Delta positions");
            sealed
                .positions_handle()
                .expect("sealed position handle")
                .decode_into(&mut oracle_positions)
                .expect("decode sealed positions");
            assert_eq!(live_positions, oracle_positions, "target={target}");
        }
    }

    let mut live = DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"alpha").expect("Delta cursor");
    let mut sealed =
        crate::argus::SealedPostingCursor::with_positions(&position_list, segment_num_docs)
            .expect("sealed cursor");
    loop {
        assert_eq!(live.doc(), sealed.doc());
        assert_eq!(live.freq(), sealed.freq());
        if live.doc().is_none() {
            break;
        }
        assert_eq!(
            live.next().expect("next Delta cursor"),
            sealed.next().expect("next sealed cursor")
        );
    }
}

#[test]
fn snapshot_composition_rejects_stale_schema_and_lease_drift() {
    let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
    let generation = keeper.loaded_manifest().manifest.generation;

    let stale = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
        .expect("stale Delta")
        .freeze(generation + 1);
    let Err(error) = QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::new(stale)])
    else {
        // ubs:ignore -- Accepting stale state must fail this negative test.
        panic!("stale Delta generation was accepted");
    };
    assert!(matches!(
        error,
        SnapshotError::KeeperGenerationMismatch { .. }
    ));

    let wrong_schema = DeltaSegment::new(FSFS_CHUNK_SCHEMA, 0, usize::MAX)
        .expect("wrong-schema Delta")
        .freeze(generation);
    let Err(error) =
        QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::new(wrong_schema)])
    else {
        // ubs:ignore -- Accepting the wrong schema must fail this negative test.
        panic!("wrong-schema Delta was accepted");
    };
    assert!(matches!(error, SnapshotError::SchemaMismatch { .. }));

    let first = Arc::new(
        DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
            .expect("first overlapping Delta")
            .freeze(generation),
    );
    let second = Arc::new(
        DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
            .expect("second overlapping Delta")
            .freeze(generation),
    );
    let Err(error) =
        QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::clone(&first), second])
    else {
        // ubs:ignore -- Accepting overlapping leases must fail this negative test.
        panic!("overlapping Delta leases were accepted");
    };
    assert!(matches!(
        error,
        SnapshotError::OverlappingDeltaLeases { .. }
    ));

    let publisher = SnapshotPublisher::new(keeper, vec![first]).expect("snapshot publisher");
    let replacement = Arc::new(
        DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
            .expect("replacement Delta")
            .freeze(generation),
    );
    assert!(matches!(
        publisher.publish_delta(1, replacement),
        Err(SnapshotError::ShardOutOfBounds {
            shard: 1,
            shard_count: 1
        })
    ));
}

#[test]
fn snapshot_delta_lease_lookup_selects_only_the_owner_without_reordering_deltas() {
    let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
    let generation = keeper.loaded_manifest().manifest.generation;

    let mut first =
        DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("first aligned Delta lease");
    let first_docid = u32::try_from(first.lease_base()).expect("first docid fits u32");
    apply_sealable_delta_document(&mut first, first_docid, "lease-first", "first-value", 1);

    let mut tombstoned = DeltaSegment::new(DEFAULT_SCHEMA, first.lease_end(), usize::MAX)
        .expect("successor aligned Delta lease");
    let tombstoned_docid =
        u32::try_from(tombstoned.lease_base()).expect("tombstoned docid fits u32");
    apply_sealable_delta_document(
        &mut tombstoned,
        tombstoned_docid,
        "lease-tombstone",
        "tombstone-value",
        1,
    );
    assert_eq!(
        tombstoned.delete_delta_id("lease-tombstone"),
        Some(tombstoned_docid)
    );

    let gap = DeltaSegment::new(DEFAULT_SCHEMA, tombstoned.lease_end(), usize::MAX)
        .expect("aligned unused gap lease");
    let mut third = DeltaSegment::new(DEFAULT_SCHEMA, gap.lease_end(), usize::MAX)
        .expect("third aligned Delta lease");
    let third_docid = u32::try_from(third.lease_base()).expect("third docid fits u32");
    apply_sealable_delta_document(&mut third, third_docid, "lease-third", "third-value", 1);

    let first = Arc::new(first.freeze(generation));
    let tombstoned = Arc::new(tombstoned.freeze(generation));
    let third = Arc::new(third.freeze(generation));
    reset_delta_lease_overlap_comparisons();
    let snapshot = QuillSearchSnapshot::compose(
        0,
        keeper,
        vec![
            Arc::clone(&third),
            Arc::clone(&first),
            Arc::clone(&tombstoned),
        ],
    )
    .expect("compose shuffled nonoverlapping Delta leases");
    assert_eq!(
        delta_lease_overlap_comparisons(),
        snapshot.delta_count() - 1,
        "sorted Delta lease validation compares each adjacent pair exactly once"
    );

    for (actual, expected) in snapshot
        .delta_snapshots()
        .iter()
        .zip([&third, &first, &tombstoned])
    {
        assert!(
            Arc::ptr_eq(actual, expected),
            "lease lookup must not reorder the primary Delta table"
        );
    }

    let reset_attempts = || {
        third.reset_materialization_attempts();
        first.reset_materialization_attempts();
        tombstoned.reset_materialization_attempts();
    };
    let attempts = || {
        [
            third.materialization_attempts(),
            first.materialization_attempts(),
            tombstoned.materialization_attempts(),
        ]
    };

    reset_attempts();
    assert_eq!(
        snapshot.materialize_document_id(first_docid).as_deref(),
        Some("lease-first")
    );
    assert_eq!(
        attempts(),
        [0, 1, 0],
        "only the lease owner may materialize an ID hit"
    );

    reset_attempts();
    assert_eq!(
        snapshot
            .materialize_stored_value(CONTENT_FIELD, first_docid)
            .expect("materialize selected stored value"),
        Some(b"first-value".to_vec())
    );
    assert_eq!(
        attempts(),
        [0, 1, 0],
        "only the lease owner may materialize stored bytes"
    );

    reset_attempts();
    assert_eq!(snapshot.materialize_document_id(tombstoned_docid), None);
    assert_eq!(
        attempts(),
        [0, 0, 1],
        "a tombstoned selected row must not scan nonowners"
    );

    reset_attempts();
    assert_eq!(
        snapshot
            .materialize_stored_value(CONTENT_FIELD, tombstoned_docid)
            .expect("materialize tombstoned selected row"),
        None
    );
    assert_eq!(
        attempts(),
        [0, 0, 1],
        "stored tombstone validation must remain within the selected Delta"
    );

    let gap_docid = u32::try_from(gap.lease_base()).expect("gap docid fits u32");
    reset_attempts();
    assert_eq!(snapshot.materialize_document_id(gap_docid), None);
    assert_eq!(
        attempts(),
        [0, 0, 0],
        "a lease gap must not probe any Delta"
    );
}

#[test]
fn snapshot_composition_rejects_keeper_delta_occupied_range_overlap() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        index
            .index_documents(&cx, &[IndexableDocument::new("sealed", "alpha")])
            .await
            .expect("accumulate sealed row");
        index.commit(&cx).await.expect("commit sealed row");

        let keeper = index
            .snapshot()
            .expect("overlap fixture snapshot is authoritative");
        let generation = keeper.loaded_manifest().manifest.generation;
        let segment = &keeper.loaded_manifest().manifest.segments[0];
        assert_eq!((segment.docid_lo, segment.docid_hi), (0, 1));

        let mut overlapping =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("overlapping Delta");
        apply_alpha_delta(&mut overlapping, 0, "duplicate-docid", 1);
        let Err(error) = QuillSearchSnapshot::compose(
            0,
            Arc::clone(&keeper),
            vec![Arc::new(overlapping.freeze(generation))],
        ) else {
            panic!("occupied Keeper/Delta docid overlap was accepted"); // ubs:ignore -- Test rejects overlapping document IDs.
        };
        assert_eq!(
            error,
            SnapshotError::KeeperDeltaDocidOverlap {
                delta_lo: 0,
                delta_hi: 1,
                segment_id: segment.segment_id,
                keeper_lo: 0,
                keeper_hi: 1,
            }
        );

        let mut continuation =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("same-lease Delta");
        apply_alpha_delta(&mut continuation, 1, "nonoverlapping-docid", 1);
        QuillSearchSnapshot::compose(0, keeper, vec![Arc::new(continuation.freeze(generation))])
            .expect("actual same-lease continuation range does not overlap Keeper");
    });
}

#[test]
fn failed_complete_publications_leave_the_current_epoch_unchanged() {
    run_with_cx(|cx| async move {
        let genesis =
            Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"));
        let genesis_generation = genesis.loaded_manifest().manifest.generation;
        let snapshot_source =
            SnapshotPublisher::new(Arc::clone(&genesis), Vec::new()).expect("genesis publisher");

        let first = QuillIndex::in_memory(deterministic_config()).expect("first index");
        first
            .index_documents(&cx, &[IndexableDocument::new("first", "alpha")])
            .await
            .expect("accumulate first successor");
        first.commit(&cx).await.expect("publish first successor");
        let successor = first
            .snapshot()
            .expect("first successor snapshot is authoritative");

        let before = snapshot_source.load();
        let stale_delta = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, DOC_ORDS_PER_LEASE.into(), usize::MAX)
                .expect("stale Delta")
                .freeze(genesis_generation),
        );
        assert!(matches!(
            snapshot_source.publish_complete(Arc::clone(&successor), vec![stale_delta]),
            Err(SnapshotError::KeeperGenerationMismatch { .. })
        ));
        assert!(Arc::ptr_eq(&before, &snapshot_source.load()));
        assert_eq!(alpha_snapshot_tuple(&before), (0, 0, 0, 0, 0));

        let accepted = snapshot_source
            .publish_complete(Arc::clone(&successor), Vec::new())
            .expect("publish valid Keeper successor");
        assert_eq!(alpha_snapshot_tuple(&accepted), (1, 1, 1, 1, 1));

        let Err(rollback) = snapshot_source.publish_complete(genesis, Vec::new()) else {
            panic!("stale Keeper rollback was accepted"); // ubs:ignore -- Test rejects rollback.
        };
        assert_eq!(
            rollback,
            SnapshotError::KeeperGenerationRegression {
                current: successor.loaded_manifest().manifest.generation,
                proposed: genesis_generation,
            }
        );
        assert!(Arc::ptr_eq(&accepted, &snapshot_source.load()));

        let second = QuillIndex::in_memory(deterministic_config()).expect("second index");
        second
            .index_documents(&cx, &[IndexableDocument::new("second", "beta")])
            .await
            .expect("accumulate colliding successor");
        second
            .commit(&cx)
            .await
            .expect("publish colliding successor fixture");
        let collision = second
            .snapshot()
            .expect("colliding successor snapshot is authoritative");
        let Err(collision) = snapshot_source.publish_complete(collision, Vec::new()) else {
            panic!("same-generation divergent MANIFEST was accepted"); // ubs:ignore -- Test rejects a divergent manifest.
        };
        assert_eq!(
            collision,
            SnapshotError::KeeperGenerationCollision {
                generation: successor.loaded_manifest().manifest.generation,
            }
        );
        assert!(Arc::ptr_eq(&accepted, &snapshot_source.load()));
    });
}

#[test]
fn scalar_and_delta_writer_modes_cannot_discard_each_others_pending_state() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        index
            .index_documents(&cx, &[IndexableDocument::new("scalar", "sealed first")])
            .await
            .expect("accumulate scalar row");
        let generation = index
            .snapshot()
            .expect("scalar fixture snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let empty_delta = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, u64::from(DOC_ORDS_PER_LEASE), usize::MAX)
                .expect("empty Delta")
                .freeze(generation),
        );
        let Err(error) = index.publish_delta_table(vec![empty_delta]) else {
            panic!("Delta publication overtook scalar pending state"); // ubs:ignore -- Test rejects publication order violation.
        };
        assert!(error.to_string().contains("fully committed"));
        index.commit(&cx).await.expect("commit scalar row");

        // The single-shard writer's watermark is its exact frontier; a
        // Delta lease must start at the next block boundary above it.
        let lease_base = index
            .snapshot()
            .expect("sealed scalar snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .docid_high_watermark
            .next_multiple_of(u64::from(DOC_ORDS_PER_LEASE));
        let global_docid = u32::try_from(lease_base).expect("Delta docid");
        let generation = index
            .snapshot()
            .expect("sealed scalar snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let mut delta =
            DeltaSegment::new(DEFAULT_SCHEMA, lease_base, usize::MAX).expect("live Delta");
        apply_alpha_delta(&mut delta, global_docid, "delta", 1);
        index
            .publish_delta_table(vec![Arc::new(delta.freeze(generation))])
            .expect("publish live Delta");
        let before = index
            .search_snapshot()
            .expect("published Delta snapshot is authoritative");

        let error = index
            .index_documents(
                &cx,
                &[IndexableDocument::new("forbidden", "scalar overlap")],
            )
            .await
            .expect_err("scalar ingest must reject an active Delta epoch");
        assert!(error.to_string().contains("Delta epochs are active"));
        let Err(error) = index.commit(&cx).await else {
            panic!("scalar commit accepted an active Delta epoch"); // ubs:ignore -- Test rejects conflicting scalar publication.
        };
        assert!(error.to_string().contains("Delta epochs are active"));

        let after = index
            .search_snapshot()
            .expect("post-rejection Delta snapshot is authoritative");
        assert!(Arc::ptr_eq(&before, &after));
        assert_eq!(
            after.materialize_document_id(global_docid).as_deref(),
            Some("delta")
        );
        assert_eq!(after.bm25_doc_freq(CONTENT_FIELD, b"alpha").unwrap(), 1);
    });
}

#[test]
fn publisher_retains_a_held_epoch_until_its_last_reader_drops() {
    let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
    let generation = keeper.loaded_manifest().manifest.generation;
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
    let publisher = SnapshotPublisher::new(
        Arc::clone(&keeper),
        vec![Arc::new(delta.freeze(generation))],
    )
    .expect("snapshot publisher");
    let held = publisher.load();
    let weak = Arc::downgrade(&held);

    apply_alpha_delta(&mut delta, 0, "new", 1);
    let next_epoch = publisher
        .publish_delta(0, Arc::new(delta.freeze(generation)))
        .expect("publish next Delta epoch");
    assert_eq!(alpha_snapshot_tuple(&held), (0, 0, 0, 0, 0));
    assert_eq!(alpha_snapshot_tuple(&next_epoch), (1, 1, 1, 1, 1));
    assert_eq!(alpha_snapshot_tuple(&publisher.load()), (1, 1, 1, 1, 1));
    assert!(weak.upgrade().is_some());

    drop(held);
    assert!(
        weak.upgrade().is_none(),
        "the old composite must retire after its final reader Arc drops"
    );
}

#[test]
fn labruntime_readers_observe_only_complete_composite_epochs() {
    let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
    let generation = keeper.loaded_manifest().manifest.generation;
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
    let publisher = Arc::new(
        SnapshotPublisher::new(
            Arc::clone(&keeper),
            vec![Arc::new(delta.freeze(generation))],
        )
        .expect("snapshot publisher"),
    );
    apply_alpha_delta(&mut delta, 0, "new", 1);
    let next_delta = Arc::new(delta.freeze(generation));
    let reader_saw_old = Arc::new(AtomicBool::new(false));
    let writer_done = Arc::new(AtomicBool::new(false));

    let mut lab = LabRuntime::new(LabConfig::new(0xe5_2002).max_steps(100_000));
    let region = lab.state.create_root_region(Budget::INFINITE);

    let reader_publisher = Arc::clone(&publisher);
    let reader_started = Arc::clone(&reader_saw_old);
    let reader_done = Arc::clone(&writer_done);
    let (reader, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            assert_eq!(
                alpha_snapshot_tuple(&reader_publisher.load()),
                (0, 0, 0, 0, 0)
            );
            reader_started.store(true, Ordering::SeqCst);
            while !reader_done.load(Ordering::SeqCst) {
                let observed = alpha_snapshot_tuple(&reader_publisher.load());
                assert!(
                    observed == (0, 0, 0, 0, 0) || observed == (1, 1, 1, 1, 1),
                    "reader observed a torn composite tuple: {observed:?}"
                );
                yield_now().await;
            }
            assert_eq!(
                alpha_snapshot_tuple(&reader_publisher.load()),
                (1, 1, 1, 1, 1)
            );
        })
        .expect("create snapshot reader task");

    let writer_publisher = Arc::clone(&publisher);
    let writer_started = Arc::clone(&reader_saw_old);
    let writer_finished = Arc::clone(&writer_done);
    let (writer, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            while !writer_started.load(Ordering::SeqCst) {
                yield_now().await;
            }
            writer_publisher
                .publish_delta(0, next_delta)
                .expect("publish complete Delta epoch");
            writer_finished.store(true, Ordering::SeqCst);
        })
        .expect("create snapshot writer task");

    lab.scheduler.lock().schedule(reader, 0);
    lab.step_for_test();
    assert!(reader_saw_old.load(Ordering::SeqCst));
    lab.scheduler.lock().schedule(writer, 0);
    let report = lab.run_until_quiescent_with_report();

    assert!(report.quiescent, "LabRuntime must reach quiescence");
    assert!(report.oracle_report.all_passed(), "oracles must pass");
    assert!(report.invariant_violations.is_empty());
}
