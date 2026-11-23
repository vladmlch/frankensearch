use super::*;

#[test]
fn tombstone_folded_delta_seal_is_byte_identical_to_direct_scribe_flush() {
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta lease");
    apply_sealable_delta_document(&mut delta, 0, "dead-a", "ghost", 1);
    apply_sealable_delta_document(&mut delta, 1, "live-a", "alpha", 2);
    apply_sealable_delta_document(&mut delta, 2, "dead-b", "ghost", 2);
    apply_sealable_delta_document(&mut delta, 3, "live-b", "beta", 1);
    assert_eq!(delta.delete_delta_id("dead-a"), Some(0));
    assert_eq!(delta.delete_delta_id("dead-b"), Some(2));
    let frozen = delta.freeze(1);

    let metadata = b"{}";
    let ordinal_a = 1_u64.to_le_bytes();
    let ordinal_b = 3_u64.to_le_bytes();
    let mut direct = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("direct accumulator");
    direct
        .add_document_with_values(
            1,
            &[
                IndexedFieldValue::new(ID_FIELD, "live-a"),
                IndexedFieldValue::new(CONTENT_FIELD, "alpha alpha"),
                IndexedFieldValue::new(TITLE_FIELD, ""),
            ],
            &[],
            &[
                StoredFieldValue::new(METADATA_FIELD, metadata),
                StoredFieldValue::new(ORD_FIELD, &ordinal_a),
            ],
        )
        .expect("accumulate direct live-a");
    direct
        .add_document_with_values(
            3,
            &[
                IndexedFieldValue::new(ID_FIELD, "live-b"),
                IndexedFieldValue::new(CONTENT_FIELD, "beta"),
                IndexedFieldValue::new(TITLE_FIELD, ""),
            ],
            &[],
            &[
                StoredFieldValue::new(METADATA_FIELD, metadata),
                StoredFieldValue::new(ORD_FIELD, &ordinal_b),
            ],
        )
        .expect("accumulate direct live-b");
    let documents = [
        FlushDocumentInput::new(1, "live-a", shipping_content_hash("live-a", "alpha alpha")),
        FlushDocumentInput::new(3, "live-b", shipping_content_hash("live-b", "beta")),
    ];
    let metadata = DeltaFlushInput {
        segment_id: 0xe5_4000,
        created_unix_s: 1_700_000_000,
        engine_version: CURRENT_ENGINE_VERSION,
    };
    let direct_encoded = flush_accumulator_with_mode(
        &direct,
        FlushSegmentInput {
            segment_id: metadata.segment_id,
            lease_docid_base: 0,
            created_unix_s: metadata.created_unix_s,
            engine_version: metadata.engine_version,
            documents: &documents,
        },
        FlushMode::Scalar,
    )
    .expect("flush direct Scribe fixture");
    let delta_encoded = flush_delta_snapshot(&frozen, metadata)
        .expect("flush tombstone-folded Delta")
        .expect("live Delta emits a segment");
    assert_eq!(delta_encoded.as_bytes(), direct_encoded.as_bytes());

    let mut all_dead = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("all-dead Delta");
    apply_sealable_delta_document(&mut all_dead, 0, "dead-only", "ghost", 1);
    assert_eq!(all_dead.delete_delta_id("dead-only"), Some(0));
    assert!(
        flush_delta_snapshot(&all_dead.freeze(1), metadata)
            .expect("all-dead seal result")
            .is_none(),
        "an all-tombstoned Delta must not emit an empty FSLX"
    );
}

#[test]
fn convenience_delta_row_is_searchable_but_not_sealable_without_content_hash() {
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta lease");
    let positions = [0];
    let fieldnorms = [
        DeltaFieldNorm {
            field_ord: ID_FIELD,
            raw_length: 1,
            fieldnorm_id: fieldnorm_to_id(1),
        },
        DeltaFieldNorm {
            field_ord: CONTENT_FIELD,
            raw_length: 1,
            fieldnorm_id: fieldnorm_to_id(1),
        },
        DeltaFieldNorm {
            field_ord: TITLE_FIELD,
            raw_length: 0,
            fieldnorm_id: fieldnorm_to_id(0),
        },
    ];
    let postings = [
        DeltaTermPosting {
            field_ord: ID_FIELD,
            term: b"convenience",
            frequency: 1,
            positions: None,
        },
        DeltaTermPosting {
            field_ord: CONTENT_FIELD,
            term: b"alpha",
            frequency: 1,
            positions: Some(&positions),
        },
    ];
    delta
        .apply_document(0, DocId::from("convenience"), &fieldnorms, &postings)
        .expect("apply process-local convenience row");
    let frozen = delta.freeze(1);
    assert_eq!(frozen.content_hash(0), None);
    assert_eq!(
        DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"alpha")
            .expect("query convenience row")
            .doc(),
        Some(0)
    );
    assert!(matches!(
        flush_delta_snapshot(
            &frozen,
            DeltaFlushInput {
                segment_id: 0xe5_4007,
                created_unix_s: 1_700_000_007,
                engine_version: CURRENT_ENGINE_VERSION,
            },
        ),
        Err(FlushError::MissingDeltaContentHash { global_docid: 0 })
    ));
}

#[test]
fn delta_seal_numeric_and_stored_sidecars_are_byte_identical_to_direct_scribe_flush() {
    let mut delta =
        DeltaSegment::new(DELTA_PARITY_SCHEMA, 0, usize::MAX).expect("Delta parity lease");
    {
        let mut add_delta =
            |global_docid: u32, document_id: &str, frequency: u32, rank: u64, opaque: &[u8]| {
                let content = std::iter::repeat_n(
                    "alpha",
                    usize::try_from(frequency).expect("fixture frequency fits usize"),
                )
                .collect::<Vec<_>>()
                .join(" ");
                let positions = (0..frequency).collect::<Vec<_>>();
                let fieldnorms = [
                    DeltaFieldNorm {
                        field_ord: 0,
                        raw_length: 1,
                        fieldnorm_id: fieldnorm_to_id(1),
                    },
                    DeltaFieldNorm {
                        field_ord: 1,
                        raw_length: frequency,
                        fieldnorm_id: fieldnorm_to_id(frequency),
                    },
                    DeltaFieldNorm {
                        field_ord: 4,
                        raw_length: 0,
                        fieldnorm_id: fieldnorm_to_id(0),
                    },
                ];
                let postings = [
                    DeltaTermPosting {
                        field_ord: 0,
                        term: document_id.as_bytes(),
                        frequency: 1,
                        positions: None,
                    },
                    DeltaTermPosting {
                        field_ord: 1,
                        term: b"alpha",
                        frequency,
                        positions: Some(&positions),
                    },
                ];
                delta
                    .apply_document_with_values(
                        global_docid,
                        DocId::from(document_id),
                        shipping_content_hash(document_id, &content),
                        &fieldnorms,
                        &postings,
                        &[DeltaNumericValue::u64(2, rank)],
                        &[
                            DeltaStoredValue::new(0, document_id.as_bytes()),
                            DeltaStoredValue::new(1, content.as_bytes()),
                            DeltaStoredValue::new(3, opaque),
                        ],
                    )
                    .expect("apply typed Delta parity document");
            };
        add_delta(0, "dead", 1, 5, b"discarded");
        add_delta(1, "live-a", 2, 11, b"opaque-a");
        add_delta(3, "live-b", 1, u64::MAX, b"");
    }
    assert_eq!(delta.delete_delta_id("dead"), Some(0));
    let frozen = delta.freeze(1);

    let mut direct =
        ColumnarAccumulator::new(DELTA_PARITY_SCHEMA).expect("direct typed accumulator");
    for (doc_ord, document_id, content, rank, opaque) in [
        (1, "live-a", "alpha alpha", 11, b"opaque-a".as_slice()),
        (3, "live-b", "alpha", u64::MAX, b"".as_slice()),
    ] {
        direct
            .add_document_with_values(
                doc_ord,
                &[
                    IndexedFieldValue::new(0, document_id),
                    IndexedFieldValue::new(1, content),
                ],
                &[IndexedNumericValue::u64(2, rank)],
                &[StoredFieldValue::new(3, opaque)],
            )
            .expect("accumulate direct typed parity document");
    }
    let documents = [
        FlushDocumentInput::new(1, "live-a", shipping_content_hash("live-a", "alpha alpha")),
        FlushDocumentInput::new(3, "live-b", shipping_content_hash("live-b", "alpha")),
    ];
    let metadata = DeltaFlushInput {
        segment_id: 0xe5_4008,
        created_unix_s: 1_700_000_008,
        engine_version: CURRENT_ENGINE_VERSION,
    };
    let direct_encoded = flush_accumulator_with_mode(
        &direct,
        FlushSegmentInput {
            segment_id: metadata.segment_id,
            lease_docid_base: 0,
            created_unix_s: metadata.created_unix_s,
            engine_version: metadata.engine_version,
            documents: &documents,
        },
        FlushMode::Scalar,
    )
    .expect("flush direct typed parity fixture");
    let delta_encoded = flush_delta_snapshot(&frozen, metadata)
        .expect("flush typed Delta parity fixture")
        .expect("live typed Delta emits FSLX");
    assert_eq!(delta_encoded.as_bytes(), direct_encoded.as_bytes());
}

#[test]
fn delta_seal_transaction_has_no_gap_and_never_crosses_a_lease() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let first_generation = index
            .snapshot()
            .expect("first lease snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let first_docid = DOC_ORDS_PER_LEASE - 1;
        let mut first_delta =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("first Delta lease");
        let dead_docid = DOC_ORDS_PER_LEASE - 3;
        apply_sealable_delta_document(&mut first_delta, dead_docid, "dead", "ghost", 1);
        apply_alpha_delta(&mut first_delta, first_docid, "lease-zero", 1);
        assert_eq!(first_delta.delete_delta_id("dead"), Some(dead_docid));
        let first_sealed = Arc::new(first_delta.freeze(first_generation));
        index
            .publish_delta_table(vec![Arc::clone(&first_sealed)])
            .expect("publish first Delta epoch");
        let held_first = index
            .search_snapshot()
            .expect("first Delta snapshot is authoritative");
        let first_pre_search = index
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("search first Delta epoch");
        let first_pre_all = index
            .search_paginated(&cx, "*", 10, 0, true)
            .expect("match all first Delta epoch");
        let first_pre_docids = index
            .collect_docids(&cx, "alpha")
            .expect("collect first Delta epoch");
        assert_eq!(first_pre_search.hits[0].document_id, "lease-zero");
        assert_eq!(first_pre_all.hits.len(), 1);
        assert_eq!(first_pre_all.total_count, Some(1));
        assert_eq!(first_pre_docids, vec![first_docid]);

        let second_lease = u64::from(DOC_ORDS_PER_LEASE);
        let empty_second = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, second_lease, usize::MAX)
                .expect("second empty Delta")
                .freeze(first_generation + 1),
        );
        let installed_first = index
            .seal_delta_snapshot(
                &cx,
                Arc::clone(&first_sealed),
                vec![empty_second],
                DeltaFlushInput {
                    segment_id: 0xe5_4001,
                    created_unix_s: 1_700_000_001,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .await
            .expect("seal first Delta lease");

        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("search sealed first epoch"),
            first_pre_search,
            "public ranking, score bits, ids, and counts must cross the first seal unchanged"
        );
        assert_eq!(
            index
                .search_paginated(&cx, "*", 10, 0, true)
                .expect("match all sealed first epoch"),
            first_pre_all,
            "Delta tombstones and lease holes must remain hidden after sealing"
        );
        assert_eq!(
            index
                .collect_docids(&cx, "alpha")
                .expect("collect sealed first epoch"),
            first_pre_docids
        );

        assert_eq!(
            held_first.bm25_doc_freq(CONTENT_FIELD, b"alpha").unwrap(),
            1
        );
        assert_eq!(
            installed_first
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .unwrap(),
            1
        );
        assert_eq!(held_first.keeper_snapshot().segments().len(), 0);
        assert_eq!(held_first.delta_snapshots()[0].live_document_count(), 1);
        assert_eq!(installed_first.keeper_snapshot().segments().len(), 1);
        assert_eq!(
            installed_first.delta_snapshots()[0].live_document_count(),
            0
        );
        assert_eq!(
            installed_first
                .keeper_snapshot()
                .materialize_document_id(first_docid)
                .as_deref(),
            Some("lease-zero")
        );

        let second_generation = index
            .snapshot()
            .expect("sealed first lease snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let second_docid = DOC_ORDS_PER_LEASE;
        let mut second_delta = DeltaSegment::new(DEFAULT_SCHEMA, second_lease, usize::MAX)
            .expect("second Delta lease");
        apply_alpha_delta(&mut second_delta, second_docid, "lease-one", 1);
        let second_sealed = Arc::new(second_delta.freeze(second_generation));
        index
            .publish_delta_table(vec![Arc::clone(&second_sealed)])
            .expect("publish second Delta epoch");
        let held_second = index
            .search_snapshot()
            .expect("second Delta snapshot is authoritative");
        let second_pre_search = index
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("search second Delta epoch");
        let second_pre_docids = index
            .collect_docids(&cx, "alpha")
            .expect("collect second Delta epoch");

        let third_lease = second_lease + u64::from(DOC_ORDS_PER_LEASE);
        let empty_third = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, third_lease, usize::MAX)
                .expect("third empty Delta")
                .freeze(second_generation + 1),
        );
        let installed_second = index
            .seal_delta_snapshot(
                &cx,
                Arc::clone(&second_sealed),
                vec![empty_third],
                DeltaFlushInput {
                    segment_id: 0xe5_4002,
                    created_unix_s: 1_700_000_002,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .await
            .expect("seal second Delta lease");

        assert_eq!(
            index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("search sealed second epoch"),
            second_pre_search,
            "global public ranking must cross a mixed Keeper/Delta seal unchanged"
        );
        assert_eq!(
            index
                .collect_docids(&cx, "alpha")
                .expect("collect sealed second epoch"),
            second_pre_docids
        );

        assert_eq!(
            held_second.bm25_doc_freq(CONTENT_FIELD, b"alpha").unwrap(),
            2
        );
        assert_eq!(
            installed_second
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .unwrap(),
            2
        );
        let segments = &installed_second
            .keeper_snapshot()
            .loaded_manifest()
            .manifest
            .segments;
        assert_eq!(segments.len(), 2);
        assert_eq!(
            (segments[0].docid_lo, segments[0].docid_hi),
            (65_535, 65_536)
        );
        assert_eq!(
            (segments[1].docid_lo, segments[1].docid_hi),
            (65_536, 65_537)
        );
        for segment in segments {
            assert_eq!(
                segment.docid_lo / u64::from(DOC_ORDS_PER_LEASE),
                (segment.docid_hi - 1) / u64::from(DOC_ORDS_PER_LEASE),
                "one sealed segment must stay inside one Q1 lease"
            );
        }

        let post_survivor_watermark = third_lease + u64::from(DOC_ORDS_PER_LEASE);
        assert_eq!(
            installed_second
                .keeper_snapshot()
                .loaded_manifest()
                .manifest
                .docid_high_watermark,
            post_survivor_watermark,
            "the durable watermark must cover every allocated replacement lease"
        );
        let mut reopened = QuillIndex::from_backend(
            IndexBackend::Memory(
                index
                    .snapshot()
                    .expect("sealed second lease snapshot is authoritative")
                    .as_ref()
                    .clone(),
            ),
            DEFAULT_SCHEMA,
            deterministic_config(),
        )
        .expect("reopen sealed Keeper snapshot");
        assert_eq!(
            reopened.writer_mut().next_lease_base,
            post_survivor_watermark
        );
        reopened
            .index_documents(
                &cx,
                &[IndexableDocument::new("after-reopen", "fresh allocation")],
            )
            .await
            .expect("index after reopen");
        reopened.commit(&cx).await.expect("commit after reopen");
        assert_eq!(
            reopened
                .snapshot()
                .expect("reopened lease snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .segments
                .last()
                .expect("post-reopen segment")
                .docid_lo,
            post_survivor_watermark,
            "reopen must not reuse any Delta-owned lease"
        );
    });
}

#[test]
fn delta_seal_rejects_an_omitted_surviving_shard_epoch() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let generation = index
            .snapshot()
            .expect("omitted-survivor snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let mut source =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("source Delta shard");
        apply_sealable_delta_document(&mut source, 0, "source", "alpha", 1);
        let source = Arc::new(source.freeze(generation));
        let survivor_base = u64::from(DOC_ORDS_PER_LEASE);
        let survivor_docid = DOC_ORDS_PER_LEASE;
        let mut survivor = DeltaSegment::new(DEFAULT_SCHEMA, survivor_base, usize::MAX)
            .expect("surviving Delta shard");
        apply_sealable_delta_document(&mut survivor, survivor_docid, "survivor", "beta", 1);
        let survivor = Arc::new(survivor.freeze(generation));
        index
            .publish_delta_table(vec![Arc::clone(&source), Arc::clone(&survivor)])
            .expect("publish two-shard Delta table");
        let before = index
            .search_paginated(&cx, "alpha OR beta", 10, 0, true)
            .expect("query both Delta shards");

        let Err(error) = index
            .seal_delta_snapshot(
                &cx,
                Arc::clone(&source),
                Vec::new(),
                DeltaFlushInput {
                    segment_id: 0xe5_4003,
                    created_unix_s: 1_700_000_003,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .await
        else {
            panic!("seal must not discard an independently published shard"); // ubs:ignore -- Test rejects loss of a published shard.
        };
        assert!(matches!(
            error,
            QuillIndexError::Snapshot(SnapshotError::MissingDeltaRebind {
                lease_base,
                lease_end,
            }) if lease_base == survivor_base
                && lease_end == survivor_base + u64::from(DOC_ORDS_PER_LEASE)
        ));
        assert!(index.writer_mut().pending_delta_seal.is_none());
        assert_eq!(
            index
                .snapshot()
                .expect("failed seal snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha OR beta", 10, 0, true)
                .expect("failed seal leaves complete table visible"),
            before
        );

        let rebound = Arc::new(survivor.rebind_keeper_generation(generation + 1));
        index
            .seal_delta_snapshot(
                &cx,
                source,
                vec![rebound],
                DeltaFlushInput {
                    segment_id: 0xe5_4003,
                    created_unix_s: 1_700_000_003,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .await
            .expect("complete successor table seals source shard");
        assert_eq!(
            index
                .search_paginated(&cx, "alpha OR beta", 10, 0, true)
                .expect("successful seal retains surviving shard"),
            before
        );
    });
}

#[test]
fn delta_seal_rejects_a_stale_earlier_freeze_of_a_surviving_shard() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let generation = index
            .snapshot()
            .expect("stale-survivor snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let mut source =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("source Delta shard");
        apply_sealable_delta_document(&mut source, 0, "source", "alpha", 1);
        let source = Arc::new(source.freeze(generation));

        let survivor_base = u64::from(DOC_ORDS_PER_LEASE);
        let mut survivor = DeltaSegment::new(DEFAULT_SCHEMA, survivor_base, usize::MAX)
            .expect("surviving Delta shard");
        apply_sealable_delta_document(
            &mut survivor,
            DOC_ORDS_PER_LEASE,
            "survivor-first",
            "beta",
            1,
        );
        let stale_survivor = Arc::new(survivor.freeze(generation));
        apply_sealable_delta_document(
            &mut survivor,
            DOC_ORDS_PER_LEASE + 1,
            "survivor-latest",
            "gamma",
            1,
        );
        let latest_survivor = Arc::new(survivor.freeze(generation));
        assert_ne!(
            stale_survivor.publication_lineage(),
            latest_survivor.publication_lineage(),
            "successive freezes must name distinct immutable epochs"
        );

        index
            .publish_delta_table(vec![Arc::clone(&source), Arc::clone(&latest_survivor)])
            .expect("publish source and latest surviving epoch");
        let before = index
            .search_paginated(&cx, "alpha OR beta OR gamma", 10, 0, true)
            .expect("query complete pre-seal view");
        assert_eq!(before.total_count, Some(3));

        let stale_rebound = Arc::new(stale_survivor.rebind_keeper_generation(generation + 1));
        let Err(error) = index
            .seal_delta_snapshot(
                &cx,
                Arc::clone(&source),
                vec![stale_rebound],
                DeltaFlushInput {
                    segment_id: 0xe5_4004,
                    created_unix_s: 1_700_000_004,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .await
        else {
            panic!("seal must reject an older freeze of the surviving shard"); // ubs:ignore -- Test rejects a stale shard freeze.
        };
        assert!(matches!(
            error,
            QuillIndexError::Snapshot(SnapshotError::MissingDeltaRebind {
                lease_base,
                lease_end,
            }) if lease_base == survivor_base
                && lease_end == survivor_base + u64::from(DOC_ORDS_PER_LEASE)
        ));
        assert!(index.writer_mut().pending_delta_seal.is_none());
        assert_eq!(
            index
                .snapshot()
                .expect("failed stale seal snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation,
            generation
        );
        assert_eq!(
            index
                .search_paginated(&cx, "alpha OR beta OR gamma", 10, 0, true)
                .expect("rejected stale replacement leaves complete view visible"),
            before
        );

        let latest_rebound = Arc::new(latest_survivor.rebind_keeper_generation(generation + 1));
        index
            .seal_delta_snapshot(
                &cx,
                source,
                vec![latest_rebound],
                DeltaFlushInput {
                    segment_id: 0xe5_4004,
                    created_unix_s: 1_700_000_004,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .await
            .expect("exact surviving epoch rebind seals source shard");
        assert_eq!(
            index
                .search_paginated(&cx, "alpha OR beta OR gamma", 10, 0, true)
                .expect("successful seal retains latest surviving epoch"),
            before
        );
    });
}

#[test]
fn labruntime_delta_to_keeper_swap_has_no_visibility_gap() {
    let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
    let generation = keeper.loaded_manifest().manifest.generation;
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta lease");
    apply_alpha_delta(&mut delta, 0, "continuous", 1);
    let sealed = Arc::new(delta.freeze(generation));
    let replacement_base = u64::from(DOC_ORDS_PER_LEASE);
    let replacement_docid = u32::try_from(replacement_base).expect("replacement docid");
    let mut current_survivor = DeltaSegment::new(DEFAULT_SCHEMA, replacement_base, usize::MAX)
        .expect("current surviving Delta");
    apply_sealable_delta_document(
        &mut current_survivor,
        replacement_docid,
        "surviving",
        "beta",
        2,
    );
    let current_survivor = Arc::new(current_survivor.freeze(generation));
    let publisher = SnapshotPublisher::new(
        Arc::clone(&keeper),
        vec![Arc::clone(&sealed), Arc::clone(&current_survivor)],
    )
    .expect("snapshot publisher");
    let encoded = flush_delta_snapshot(
        &sealed,
        DeltaFlushInput {
            segment_id: 0xe5_4010,
            created_unix_s: 1_700_000_010,
            engine_version: CURRENT_ENGINE_VERSION,
        },
    )
    .expect("build Delta seal")
    .expect("live Delta emits a segment");
    let mut manifest = keeper.next_manifest().expect("successor manifest");
    manifest.segments.push(manifest_segment(&encoded, 1));
    manifest.docid_high_watermark = sealed.lease_end();
    let mut pending = BTreeMap::new();
    for field in DEFAULT_SCHEMA
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
    {
        pending.insert(
            field.id,
            (
                sealed
                    .live_total_tokens(field.id)
                    .expect("Delta field stats"),
                1,
            ),
        );
    }
    manifest.field_stats =
        merge_field_stats(&manifest.field_stats, &pending).expect("merge Delta field stats");
    let successor = Arc::new(
        keeper
            .publish_owned_segments(&manifest, vec![encoded])
            .expect("publish owned successor"),
    );
    let replacement = Arc::new(current_survivor.rebind_keeper_generation(manifest.generation));
    let prepared = publisher
        .prepare_sealed_manifest_with_deltas(DEFAULT_SCHEMA, &manifest, vec![replacement])
        .expect("prepare complete Delta seal publication");
    let publisher = Arc::new(publisher);
    let reader_started = Arc::new(AtomicBool::new(false));
    let writer_done = Arc::new(AtomicBool::new(false));

    let mut lab = LabRuntime::new(LabConfig::new(0xe5_4004).max_steps(100_000));
    let region = lab.state.create_root_region(Budget::INFINITE);
    let reader_publisher = Arc::clone(&publisher);
    let reader_started_flag = Arc::clone(&reader_started);
    let reader_done = Arc::clone(&writer_done);
    let (reader, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            reader_started_flag.store(true, Ordering::SeqCst);
            while !reader_done.load(Ordering::SeqCst) {
                let observed = reader_publisher.load();
                assert_eq!(
                    observed
                        .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                        .expect("alpha frequency"),
                    1
                );
                assert_eq!(
                    observed
                        .bm25_doc_freq(CONTENT_FIELD, b"beta")
                        .expect("beta frequency"),
                    1
                );
                let residency = (
                    observed.keeper_snapshot().segments().len(),
                    observed
                        .delta_snapshots()
                        .iter()
                        .map(|delta| delta.live_document_count())
                        .sum::<usize>(),
                );
                assert!(
                    residency == (0, 2) || residency == (1, 1),
                    "reader observed a visibility gap or duplicate: {residency:?}"
                );
                yield_now().await;
            }
            let observed = reader_publisher.load();
            assert_eq!(observed.keeper_snapshot().segments().len(), 1);
            assert_eq!(observed.delta_snapshots()[0].live_document_count(), 1);
        })
        .expect("create seal reader");

    let writer_publisher = Arc::clone(&publisher);
    let writer_started = Arc::clone(&reader_started);
    let writer_finished = Arc::clone(&writer_done);
    let (writer, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            while !writer_started.load(Ordering::SeqCst) {
                yield_now().await;
            }
            SnapshotPublisher::validate_prepared_sealed(successor.as_ref(), &prepared)
                .expect("prepared seal successor remains valid");
            writer_publisher.install_validated_prepared_sealed(successor, prepared);
            writer_finished.store(true, Ordering::SeqCst);
        })
        .expect("create seal writer");

    lab.scheduler.lock().schedule(reader, 0);
    lab.step_for_test();
    assert!(reader_started.load(Ordering::SeqCst));
    lab.scheduler.lock().schedule(writer, 0);
    let report = lab.run_until_quiescent_with_report();
    assert!(report.quiescent);
    assert!(report.oracle_report.all_passed());
    assert!(report.invariant_violations.is_empty());
}
