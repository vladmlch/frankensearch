use super::*;

#[test]
fn compaction_preserves_q1_docids_and_live_query_identity() {
    run_with_blocking_cx(|cx| async move {
        let deleted = [1_u32, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
        let index = q1_ob4_tombstoned_index(40, &deleted);
        let source = index
            .snapshot()
            .expect("compaction source snapshot is authoritative")
            .clone();
        let source_manifest = source.loaded_manifest().manifest.clone();
        let source_segment = &source.segments()[0];
        let source_segment_id = source_segment.manifest().segment_id;
        let query_before = q1_ob4_query_evidence(&index, &cx);
        let live_docids_before = index
            .collect_docids(&cx, "shared")
            .expect("Q1-OB4 live docids before compaction");
        assert_eq!(source_segment.manifest().tombstones.cardinality(), 12);
        assert_eq!(source_segment.manifest().doc_count, 40);

        let report = index
            .compact(&cx, CompactionPolicy::default())
            .await
            .expect("Q1-OB4 in-memory compaction");
        assert_eq!(report.generation_before, source_manifest.generation);
        assert_eq!(report.generation_after, source_manifest.generation + 1);
        assert_eq!(report.examined_segments, 1);
        assert_eq!(report.compacted_segments, 1);
        assert_eq!(report.removed_segments, 0);
        assert_eq!(report.dropped_documents, 12);
        assert_eq!(report.input_bytes, source_segment.manifest().file_len);
        assert!(report.output_bytes > 0);

        let replacement_snapshot = index
            .snapshot()
            .expect("compaction replacement snapshot is authoritative");
        let replacement = &replacement_snapshot.segments()[0];
        assert_ne!(replacement.manifest().segment_id, source_segment_id);
        assert_eq!(replacement.manifest().docid_lo, 0);
        assert_eq!(replacement.manifest().docid_hi, 40);
        assert_eq!(replacement.manifest().doc_count, 28);
        assert_eq!(replacement.manifest().tombstones.cardinality(), 0);
        replacement.verify().expect("verify compacted Q1-OB4 FSLX");
        assert_eq!(
            index
                .collect_docids(&cx, "shared")
                .expect("Q1-OB4 live docids after compaction"),
            live_docids_before,
            "surviving global docids must not be renumbered",
        );
        assert_eq!(
            q1_ob4_query_evidence(&index, &cx),
            query_before,
            "ranked identities, order, global docids, counts, phrase matches, and fast-column ranges must be query-identical",
        );
        for global_docid in 0_u32..40 {
            let should_live = !deleted.contains(&global_docid);
            assert_eq!(
                index
                    .snapshot()
                    .expect("compaction liveness snapshot is authoritative")
                    .is_live(global_docid),
                should_live
            );
            assert_eq!(
                index
                    .snapshot()
                    .expect("compaction materialization snapshot is authoritative")
                    .materialize_document_id(global_docid)
                    .is_some(),
                should_live,
                "IDMAP/IDHASH hole mismatch at global docid {global_docid}",
            );
        }
        for kind in KNOWN_SECTION_KINDS {
            assert_eq!(
                source_segment
                    .section(kind)
                    .expect("read Q1-OB4 source section")
                    .is_some(),
                replacement
                    .section(kind)
                    .expect("read Q1-OB4 replacement section")
                    .is_some(),
                "compaction changed section presence for kind {}",
                kind.raw(),
            );
        }
        assert!(
            index
                .snapshot()
                .expect("compaction statistics snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats
                .iter()
                .all(|row| row.doc_count == 28),
            "post-compaction statistics must be re-derived over retained documents",
        );
    });
}

#[test]
fn compaction_rederives_exact_token_totals_from_retained_postings() {
    run_with_blocking_cx(|cx| async move {
        let retained_content = std::iter::repeat_n("retained", 4_096)
            .collect::<Vec<_>>()
            .join(" ");
        let deleted_content = std::iter::repeat_n("deleted", 3_001)
            .collect::<Vec<_>>()
            .join(" ");
        let documents = vec![
            IndexableDocument::new("long-retained", retained_content),
            IndexableDocument::new("long-deleted", deleted_content),
        ];
        let sealed = seal_q1_ob2a_documents(&documents, 0, 0x0b40_1001);
        let field_stats =
            merge_field_stats(&[], &sealed.field_stats).expect("long source statistics");
        let committed = q1_ob2a_owned_index(vec![sealed.encoded], field_stats);
        let mut tombstoned = committed
            .snapshot()
            .expect("long compaction source snapshot is authoritative")
            .next_manifest()
            .expect("long tombstone MANIFEST");
        assert!(
            committed
                .snapshot()
                .expect("long compaction source snapshot is authoritative")
                .delete_document(&mut tombstoned, "long-deleted")
                .expect("delete long document")
        );
        let tombstoned = committed
            .snapshot()
            .expect("long compaction source snapshot is authoritative")
            .publish_owned_segments(&tombstoned, Vec::new())
            .expect("publish long tombstones");
        let compacted = QuillIndex::from_backend(
            IndexBackend::Memory(tombstoned),
            DEFAULT_SCHEMA,
            deterministic_config(),
        )
        .expect("bind long compaction fixture");

        let canonical = seal_q1_ob2a_documents(&documents[..1], 0, 0x0b40_1002);
        let canonical_stats = canonical.field_stats;
        let report = compacted
            .compact(&cx, CompactionPolicy::default())
            .await
            .expect("compact long fieldnorm fixture");
        assert!(report.changed());
        assert_eq!(
            compacted
                .snapshot()
                .expect("long compaction result snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats
                .iter()
                .map(|row| (row.field_ord, (row.total_tokens, row.doc_count)))
                .collect::<BTreeMap<_, _>>(),
            canonical_stats,
            "STATS must use exact retained posting frequencies, not quantized fieldnorm lengths",
        );
    });
}

#[test]
fn compaction_density_boundary_is_strict_and_successor_is_idempotent() {
    run_with_blocking_cx(|cx| async move {
        let boundary_deleted = [0_u32, 1, 2, 3];
        let boundary = q1_ob4_tombstoned_index(20, &boundary_deleted);
        let boundary_generation = boundary
            .snapshot()
            .expect("boundary snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let boundary_segment_id = boundary
            .snapshot()
            .expect("boundary snapshot is authoritative")
            .segments()[0]
            .manifest()
            .segment_id;
        let report = boundary
            .compact(&cx, CompactionPolicy::default())
            .await
            .expect("20-percent boundary pass");
        assert!(!report.changed(), "density equality must not compact");
        assert_eq!(report.generation_after, boundary_generation);
        assert_eq!(
            boundary
                .snapshot()
                .expect("boundary successor snapshot is authoritative")
                .segments()[0]
                .manifest()
                .segment_id,
            boundary_segment_id,
        );

        let eligible_deleted = [0_u32, 1, 2, 3, 4];
        let eligible = q1_ob4_tombstoned_index(20, &eligible_deleted);
        let first = eligible
            .compact(&cx, CompactionPolicy::default())
            .await
            .expect("25-percent compaction");
        assert!(first.changed());
        let compacted_generation = first.generation_after;
        let compacted_segment_id = eligible
            .snapshot()
            .expect("eligible successor snapshot is authoritative")
            .segments()[0]
            .manifest()
            .segment_id;
        let second = eligible
            .compact(&cx, CompactionPolicy::default())
            .await
            .expect("idempotent successor pass");
        assert!(!second.changed());
        assert_eq!(second.generation_before, compacted_generation);
        assert_eq!(second.generation_after, compacted_generation);
        assert_eq!(
            eligible
                .snapshot()
                .expect("idempotent successor snapshot is authoritative")
                .segments()[0]
                .manifest()
                .segment_id,
            compacted_segment_id,
            "a tombstone-free replacement must not thrash",
        );
    });
}

#[test]
fn compaction_removes_fully_deleted_segments_and_rejects_invalid_policy() {
    run_with_blocking_cx(|cx| async move {
        let deleted = (0_u32..20).collect::<Vec<_>>();
        let index = q1_ob4_tombstoned_index(20, &deleted);
        let generation = index
            .snapshot()
            .expect("fully-deleted snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .generation;
        let report = index
            .compact(&cx, CompactionPolicy::default())
            .await
            .expect("fully deleted compaction");
        assert_eq!(report.generation_after, generation + 1);
        assert_eq!(report.compacted_segments, 1);
        assert_eq!(report.removed_segments, 1);
        assert_eq!(report.dropped_documents, 20);
        assert_eq!(report.output_bytes, 0);
        assert!(
            index
                .snapshot()
                .expect("fully-deleted successor snapshot is authoritative")
                .segments()
                .is_empty()
        );
        assert_eq!(
            index
                .snapshot()
                .expect("fully-deleted successor snapshot is authoritative")
                .doc_count(),
            0
        );
        assert!(
            index
                .snapshot()
                .expect("fully-deleted statistics snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .field_stats
                .iter()
                .all(|row| row.doc_count == 0 && row.total_tokens == 0),
        );

        for density in [0.0, -0.1, 1.1, f64::INFINITY, f64::NAN] {
            let Err(error) = index
                .snapshot()
                .expect("invalid-policy snapshot is authoritative")
                .compact_owned(CompactionPolicy::new(density), 0)
            else {
                panic!("invalid density {density:?} unexpectedly compacted"); // ubs:ignore -- Test rejects invalid density.
            };
            assert!(matches!(
                error,
                KeeperError::Compaction {
                    source: CompactionError::InvalidDensity { .. }
                }
            ));
        }
    });
}

#[test]
fn durable_compaction_crash_boundary_keeps_old_manifest_authoritative() {
    let _pause_serial = crate::keeper::manifest_publish_pause_test_serial_guard();
    run_with_blocking_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("temporary Q1-OB4 Keeper directory");
        let documents = q1_ob2a_documents();
        let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable Q1-OB4 index");
        index
            .index_documents(&cx, &documents[..20])
            .await
            .expect("accumulate durable Q1-OB4 documents");
        index
            .commit(&cx)
            .await
            .expect("commit durable Q1-OB4 segment");
        let source_segment_id = index
            .snapshot()
            .expect("durable compaction source snapshot is authoritative")
            .segments()[0]
            .manifest()
            .segment_id;
        let source_path = directory
            .path()
            .join(format!("seg-{source_segment_id:016x}.fslx"));
        let mut tombstoned = index
            .snapshot()
            .expect("durable compaction source snapshot is authoritative")
            .next_manifest()
            .expect("durable Q1-OB4 tombstone MANIFEST");
        for global_docid in 0_u32..5 {
            assert!(
                index
                    .snapshot()
                    .expect("durable compaction source snapshot is authoritative")
                    .delete_document(&mut tombstoned, &format!("q1-ob2a-{global_docid:03}"),)
                    .expect("stage durable Q1-OB4 tombstone"),
            );
        }
        match &mut index.writer_mut().backend {
            IndexBackend::Durable(writer) => {
                writer
                    .publish(&cx, &tombstoned)
                    .await
                    .expect("publish durable Q1-OB4 tombstones");
            }
            IndexBackend::Memory(_) => panic!("durable fixture must own KeeperWriter"), // ubs:ignore -- Test fixture requires a durable backend.
        }
        drop(index);
        let index = QuillIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("reopen durable tombstone generation");
        let old_manifest = index
            .snapshot()
            .expect("durable tombstone snapshot is authoritative")
            .loaded_manifest()
            .manifest
            .clone();
        let query_before = q1_ob4_query_evidence(&index, &cx);
        let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
            directory.path(),
            crate::keeper::PublishCheckpoint::TempWritten,
        );
        let mut compact = Box::pin(index.compact(&cx, CompactionPolicy::default()));
        crate::keeper::drive_manifest_publish_to_checkpoint_for_test(
            &mut compact,
            &pause,
            PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT,
        )
        .await
        .expect("reach bounded durable compaction crash rendezvous");

        let on_disk = Manifest::from_bytes(
            &std::fs::read(directory.path().join("MANIFEST"))
                .expect("read authoritative pre-compaction MANIFEST"),
        )
        .expect("decode authoritative pre-compaction MANIFEST");
        assert_eq!(on_disk, old_manifest);
        assert!(
            source_path.exists(),
            "old immutable segment must remain intact"
        );
        let installed_segment_count = std::fs::read_dir(directory.path())
            .expect("inspect compaction crash directory")
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("seg-") && name.ends_with(".fslx")
            })
            .count();
        assert_eq!(
            installed_segment_count, 2,
            "replacement output must be installed but still unreferenced",
        );
        let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
            .expect("reopen the old authoritative crash-boundary snapshot");
        assert_eq!(reopened.loaded_manifest().manifest, old_manifest);
        assert_eq!(
            reopened.segments()[0].manifest().segment_id,
            source_segment_id
        );
        assert_eq!(
            reopened.segments()[0].manifest().tombstones.cardinality(),
            5
        );

        pause.release();
        let report = compact.await.expect("finish released durable compaction");
        assert!(report.changed());
        assert_eq!(q1_ob4_query_evidence(&index, &cx), query_before);
        assert!(
            source_path.exists(),
            "compaction does not delete old files inline"
        );
    });
}

#[test]
fn adversarial_glob_and_phrase_queries_exhaust_fuel_deterministically() {
    run_with_cx(|cx| async move {
        // The glob leg pins the boundary from both sides. The bulk glob
        // collector stopped charging the per-expanded-term
        // doc-frequency probe the drained union paid for BM25 figures
        // the constant score then discarded, so the fixture glob's
        // boundary moved from budget 4 (old tuple `(4, 4, 1, 3, 0, 0)`
        // refused) down to budget 3: the refusal consumes the segment
        // touch, the expansion walk and the single term open (two
        // dictionary blocks), and the completion leg below proves one
        // more unit of budget finishes the same ranked query — so a
        // future change that re-inflates glob admissions fails the
        // completion leg instead of quietly shifting the boundary.
        let queries = [
            (
                3,
                (3, 3, 1, 2, 0, 0),
                Query::glob(Vec::from([CONTENT_FIELD]), "rust*".to_owned()),
            ),
            (
                7,
                (7, 7, 1, 4, 2, 0),
                Query::phrase(
                    Vec::from([crate::query::QueryField::new(CONTENT_FIELD, 1.0)]),
                    Vec::from([
                        crate::query::PositionedTerm::new(0, "rust"),
                        crate::query::PositionedTerm::new(1, "ownership"),
                    ]),
                    0,
                    false,
                ),
            ),
        ];
        for (budget, expected, query) in &queries {
            let limited = QuillIndex::in_memory(QuillConfig {
                query_fuel_budget: *budget,
                deterministic_ingest: true,
                ..QuillConfig::default()
            })
            .expect("memory index with bounded query fuel");
            limited
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("accumulate fixture");
            limited.commit(&cx).await.expect("publish fixture");
            let limited_snapshot = limited
                .search_snapshot()
                .expect("bounded-fuel snapshot is authoritative");
            let first = limited
                .execute_ranked_query(&cx, query, &limited_snapshot, 10, 0, true, Vec::new())
                .expect_err("adversarial query must exhaust its boundary budget");
            let second = limited
                .execute_ranked_query(&cx, query, &limited_snapshot, 10, 0, true, Vec::new())
                .expect_err("repeated adversarial query must exhaust identically");
            let first = fuel_diagnostics(&first);
            assert_eq!(fuel_diagnostics(&second), first, "query={query:?}");
            assert_eq!(first, *expected, "query={query:?}");
        }

        // Completion leg: one unit of budget past the glob boundary
        // (budget 3 refuses above) must rank the whole fixture result
        // with the arm's constant score.
        let boundary = QuillIndex::in_memory(QuillConfig {
            query_fuel_budget: 4,
            deterministic_ingest: true,
            ..QuillConfig::default()
        })
        .expect("memory index at the glob fuel boundary");
        boundary
            .index_documents(&cx, &fixture_documents())
            .await
            .expect("accumulate boundary fixture");
        boundary
            .commit(&cx)
            .await
            .expect("publish boundary fixture");
        let boundary_snapshot = boundary
            .search_snapshot()
            .expect("boundary snapshot is authoritative");
        let glob_query = &queries[0].2;
        let completed = boundary
            .execute_ranked_query(&cx, glob_query, &boundary_snapshot, 10, 0, true, Vec::new())
            .expect("one unit past the boundary must complete the glob");
        assert_eq!(completed.total_count, Some(2));
        assert!(
            completed
                .hits
                .iter()
                .all(|hit| hit.score.to_bits() == 1.0_f32.to_bits())
        );

        let default = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        default
            .index_documents(&cx, &fixture_documents())
            .await
            .expect("accumulate default-budget fixture");
        default
            .commit(&cx)
            .await
            .expect("publish default-budget fixture");
        let default_snapshot = default
            .search_snapshot()
            .expect("default-fuel snapshot is authoritative");
        for (_, _, query) in &queries {
            default
                .execute_ranked_query(&cx, query, &default_snapshot, 10, 0, true, Vec::new())
                .expect("default fuel budget must cover fixture corpus");
            default
                .execute_docid_query(&cx, query, &default_snapshot)
                .expect("default fuel budget must cover fixture docset collection");
        }
    });
}

#[test]
fn delta_string_range_preflights_the_complete_ordered_view() {
    let cx = Cx::for_testing();
    let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
    let generation = keeper.loaded_manifest().manifest.generation;
    let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("range Delta");
    let content = "t00 t01 t02 t03 t04 t05 t06 t07 t08 t09 t10 t11 t12 t13 t14 t15";
    apply_tokenized_delta_document(&mut delta, 0, "range-doc", content);
    assert_eq!(delta.term_count(), 17, "ID plus sixteen content terms");
    let ordered_view_blocks = delta_ordered_view_blocks(delta.term_count());
    assert_eq!(ordered_view_blocks, 2);
    let snapshot =
        QuillSearchSnapshot::compose(0, keeper, vec![Arc::new(delta.freeze(generation))])
            .expect("compose Delta-only range snapshot");
    assert_eq!(snapshot.keeper_snapshot().segments().len(), 0);

    let budget = ordered_view_blocks - 1;
    let limited: QueryCheckpointHandle<'_> =
        QueryCheckpoint::new(&cx, "range_ordered_view", budget, ordered_view_blocks);
    let refused = snapshot_string_range_terms(
        &limited,
        &snapshot,
        DEFAULT_SCHEMA,
        CONTENT_FIELD,
        Bound::Unbounded,
        Bound::Unbounded,
    )
    .expect_err("the complete ordered view must be admitted before it is built");
    assert_eq!(
        fuel_diagnostics(&refused),
        (budget, 0, 0, 0, 0, 0),
        "an under-budget range must refuse before any ordered-view work is recorded"
    );
}

#[test]
fn public_string_ranges_use_snapshot_term_ceiling_for_fuel_admission() {
    run_with_cx(|cx| async move {
        const FUEL_BUDGET: u64 = 6;
        let documents = ["atlas", "boreal", "cinder", "dynamo"]
            .into_iter()
            .map(|term| IndexableDocument::new(format!("range-{term}"), term))
            .collect::<Vec<_>>();
        let public_ranges = ["content:[a TO *]", "content:[a TO z]"];

        let exact =
            QuillIndex::in_memory(deterministic_config()).expect("create exact string-range index");
        exact
            .index_documents(&cx, &documents)
            .await
            .expect("accumulate exact string-range fixture");
        exact
            .commit(&cx)
            .await
            .expect("publish exact string-range fixture");
        for raw_query in public_ranges {
            let result = exact
                .search_paginated(&cx, raw_query, 10, 0, true)
                .expect("range results must not be truncated by the fuel correction");
            assert_eq!(result.total_count, Some(4), "query={raw_query}");
            assert_eq!(result.hits.len(), 4, "query={raw_query}");
        }

        let limited = QuillIndex::in_memory(QuillConfig {
            deterministic_ingest: true,
            glob_expansion_limit: 1,
            query_fuel_budget: FUEL_BUDGET,
            ..QuillConfig::default()
        })
        .expect("create low-fuel string-range index");
        limited
            .index_documents(&cx, &documents)
            .await
            .expect("accumulate low-fuel string-range fixture");
        limited
            .commit(&cx)
            .await
            .expect("publish low-fuel string-range fixture");
        let snapshot = limited
            .search_snapshot()
            .expect("string-range fixture snapshot is authoritative");

        for raw_query in public_ranges {
            let mut parsed = limited
                .reader
                .default_parser()
                .expect("public string-range parser")
                .parse_lenient(raw_query);
            let _ = canonicalize_query(&mut parsed.query);
            let upper_bound = query_work_upper_bound(
                &parsed.query,
                &snapshot,
                limited.reader.schema,
                limited.reader.config.glob_expansion_limit,
            )
            .expect("bound public string-range query");
            assert!(
                upper_bound > FUEL_BUDGET,
                "the finite snapshot term ceiling must enable metering for {raw_query}"
            );

            let first = limited
                .search_paginated(&cx, raw_query, 10, 0, true)
                .expect_err("low-fuel public string range must exhaust rather than truncate");
            let second = limited
                .search_paginated(&cx, raw_query, 10, 0, true)
                .expect_err("repeated low-fuel public string range must exhaust identically");
            assert_eq!(fuel_diagnostics(&first), fuel_diagnostics(&second));
            assert!(matches!(first, QuillIndexError::QueryFuelExhausted { .. }));
        }

        // The language parser intentionally drops a fully-unbounded range.
        // Exercise the same production lower-and-collect path for a
        // preparsed tree: neither endpoint can proxy its cardinality, so
        // the snapshot ceiling is required.
        let fully_unbounded = Query::range(CONTENT_FIELD, Bound::Unbounded, Bound::Unbounded);
        let upper_bound = query_work_upper_bound(
            &fully_unbounded,
            &snapshot,
            limited.reader.schema,
            limited.reader.config.glob_expansion_limit,
        )
        .expect("bound fully-unbounded preparsed string range");
        assert!(upper_bound > FUEL_BUDGET);
        let first = limited
            .execute_ranked_query(&cx, &fully_unbounded, &snapshot, 10, 0, true, Vec::new())
            .expect_err("fully-unbounded range must also exhaust low fuel");
        let second = limited
            .execute_ranked_query(&cx, &fully_unbounded, &snapshot, 10, 0, true, Vec::new())
            .expect_err("fully-unbounded range must exhaust identically");
        assert_eq!(fuel_diagnostics(&first), fuel_diagnostics(&second));
        assert!(matches!(first, QuillIndexError::QueryFuelExhausted { .. }));
    });
}

#[test]
fn owned_phrase_cursor_streams_positions_across_independent_block_seams() {
    run_with_cx(|cx| async move {
        const DOCUMENT_COUNT: usize = 270;
        const TERM_FREQUENCY: usize = 24;

        let content = std::iter::repeat_n("anchor", TERM_FREQUENCY)
            .collect::<Vec<_>>()
            .join(" ");
        let documents = (0..DOCUMENT_COUNT)
            .map(|ordinal| {
                IndexableDocument::new(format!("phrase-seam-{ordinal:03}"), content.clone())
            })
            .collect::<Vec<_>>();
        let index = QuillIndex::in_memory(QuillConfig {
            deterministic_ingest: true,
            max_ingest_shards: 1,
            ..QuillConfig::default()
        })
        .expect("construct phrase-seam index");
        // Seed the deterministic accumulator below the internal bulk-parallel
        // threshold. The following call must then stay on the scalar wrapper
        // route, giving this cursor test one segment without coupling its
        // seam oracle to the independent four-segment bulk-ingest policy.
        index
            .index_documents(&cx, &documents[..1])
            .await
            .expect("seed phrase-seam corpus");
        index
            .index_documents(&cx, &documents[1..])
            .await
            .expect("index remaining phrase-seam corpus");
        index.commit(&cx).await.expect("commit phrase-seam corpus");

        let snapshot = index
            .snapshot()
            .expect("phrase-seam snapshot is authoritative");
        assert_eq!(snapshot.segments().len(), 1);
        let segment = &snapshot.segments()[0];
        let dictionary =
            open_dictionary(segment, DEFAULT_SCHEMA).expect("open phrase-seam term dictionary");
        let found = dictionary
            .lookup(CONTENT_FIELD, b"anchor")
            .expect("look up phrase-seam term")
            .expect("phrase-seam term is present");
        assert_eq!(
            usize::try_from(found.metadata.doc_freq).expect("document frequency fits usize"),
            DOCUMENT_COUNT,
        );

        let postings_section =
            required_section(segment, SectionKind::POSTINGS).expect("POSTINGS section");
        let postings_bytes = span(
            postings_section,
            found.metadata.postings,
            "phrase-seam POSTINGS",
        )
        .expect("slice phrase-seam postings");
        let postings = PostingList::parse(postings_bytes, found.metadata.doc_freq)
            .expect("parse phrase-seam postings");
        let position_span = found.metadata.positions.expect("positioned anchor term");
        let position_section =
            required_section(segment, SectionKind::POSITIONS).expect("POSITIONS section");
        let position_bytes = span(position_section, position_span, "phrase-seam POSITIONS")
            .expect("slice phrase-seam positions");
        let position_list =
            PositionList::parse(position_bytes, &postings).expect("parse phrase-seam positions");
        assert!(postings.blocks().len() >= 2, "fixture needs posting seams");
        assert!(
            position_list.block_count() >= 2,
            "fixture needs position seams"
        );
        assert_ne!(
            postings.blocks()[1].base_posting_ordinal,
            position_list.blocks()[1].base_posting_ordinal(),
            "the first posting and position seams must be independent",
        );
        let legacy_position_rows = (0..found.metadata.doc_freq)
            .map(|ordinal| {
                position_list
                    .positions_for_ordinal(ordinal)?
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, PositionCodecError>>()
            .expect("materialize legacy ordinal position rows");

        crate::quiver::reset_position_run_work_for_test();
        PositionList::parse(position_bytes, &postings)
            .expect("measure the mandatory position-validation pass");
        let (validation_decodes, validation_consumes) = crate::quiver::position_run_work_for_test();
        assert_eq!(validation_decodes, 0);
        assert_eq!(validation_consumes, DOCUMENT_COUNT);

        crate::quiver::reset_position_run_work_for_test();
        let owned = open_owned_cursor(
            segment,
            DEFAULT_SCHEMA,
            CONTENT_FIELD,
            b"anchor",
            true,
            None,
        )
        .expect("materialize production phrase cursor");
        let (owned_decodes, scan_consumes) = crate::quiver::position_run_work_for_test();
        assert_eq!(owned_decodes, DOCUMENT_COUNT);
        assert_eq!(
            scan_consumes, validation_consumes,
            "materialization may validate each run once but must not rescan preceding rows",
        );
        assert_eq!(owned.postings.len(), DOCUMENT_COUNT);
        let position_rows = owned.positions.expect("owned position rows");
        assert_eq!(position_rows.len(), DOCUMENT_COUNT);
        assert_eq!(position_rows, legacy_position_rows);
        let expected_frequency = u32::try_from(TERM_FREQUENCY).expect("term frequency fits u32");
        let expected_positions = (0..expected_frequency).collect::<Vec<_>>();
        for (ordinal, (posting, positions)) in owned.postings.iter().zip(&position_rows).enumerate()
        {
            assert_eq!(posting.freq, expected_frequency);
            assert_eq!(positions, &expected_positions, "posting ordinal {ordinal}");
        }

        reset_owned_fieldnorm_bridge_calls();
        let phrase = index
            .search_paginated(&cx, "\"anchor anchor\"", DOCUMENT_COUNT, 0, true)
            .expect("execute checkpointed phrase query across both seam families");
        assert_eq!(
            phrase.total_count,
            Some(u64::try_from(DOCUMENT_COUNT).expect("document count fits u64")),
        );
        assert_eq!(phrase.hits.len(), DOCUMENT_COUNT);
        assert_eq!(
            owned_fieldnorm_bridge_calls(),
            0,
            "shipping sealed phrase lowering must borrow the validated DOCLEN field",
        );
    });
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn public_phrase_slop_matches_tantivy_ids_order_and_score_bits() {
    use frankensearch_lexical::tantivy_crate::{
        Index, TantivyDocument,
        collector::TopDocs,
        doc,
        query::QueryParser,
        schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions},
    };

    run_with_cx(|cx| async move {
        let corpus = [
            ("exact", "a b c"),
            ("one-gap", "a x b c"),
            ("two-gaps", "a x b x c"),
            ("left-swap", "b a c"),
            ("right-swap", "a c b"),
            ("twice", "a b c a b c"),
        ];

        let quill = QuillIndex::in_memory(deterministic_config()).expect("construct Quill");
        let documents = corpus
            .iter()
            .map(|(id, content)| IndexableDocument::new(*id, *content))
            .collect::<Vec<_>>();
        quill
            .index_documents(&cx, &documents)
            .await
            .expect("index public Quill slop corpus");
        quill.commit(&cx).await.expect("commit Quill slop corpus");

        let mut schema_builder = Schema::builder();
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let content = schema_builder.add_text_field(
            "content",
            TextOptions::default().set_indexing_options(indexing),
        );
        let tantivy = Index::create_in_ram(schema_builder.build());
        let mut writer = tantivy
            .writer_with_num_threads::<TantivyDocument>(1, 15_000_000)
            .expect("construct Tantivy writer");
        for (_, text) in corpus {
            writer
                .add_document(doc!(content => text))
                .expect("index Tantivy slop document");
        }
        writer.commit().expect("commit Tantivy slop corpus");
        drop(writer);
        let reader = tantivy.reader().expect("open Tantivy slop reader");
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&tantivy, vec![content]);

        for raw_query in ["content:\"a b c\"~1", "content:\"a b c\"~2"] {
            let quill_ranked = quill
                .search_paginated(&cx, raw_query, corpus.len(), 0, true)
                .expect("execute public Quill phrase-slop query")
                .hits
                .iter()
                .map(|hit| (hit.document_id.clone(), hit.score.to_bits()))
                .collect::<Vec<_>>();
            let parsed = parser
                .parse_query(raw_query)
                .expect("parse Tantivy phrase-slop query");
            let tantivy_ranked = searcher
                .search(
                    parsed.as_ref(),
                    &TopDocs::with_limit(corpus.len()).order_by_score(),
                )
                .expect("execute Tantivy phrase-slop query")
                .into_iter()
                .map(|(score, address)| {
                    assert_eq!(address.segment_ord, 0);
                    (
                        corpus[usize::try_from(address.doc_id).expect("doc id fits")]
                            .0
                            .to_owned(),
                        score.to_bits(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(quill_ranked, tantivy_ranked, "query={raw_query}");
        }

        #[cfg(feature = "bench-internals")]
        {
            let zero_slop_prefix = DefaultQueryParser::new(DEFAULT_SCHEMA)
                .expect("construct phrase-prefix parser")
                .parse("content:\"a b c\"*")
                .query;
            let ignored_slop_prefix = Query::phrase(
                Vec::from([crate::query::QueryField::new(CONTENT_FIELD, 1.0)]),
                Vec::from([
                    crate::query::PositionedTerm::new(0, "a"),
                    crate::query::PositionedTerm::new(1, "b"),
                    crate::query::PositionedTerm::new(2, "c"),
                ]),
                u32::MAX,
                true,
            );
            assert!(matches!(
                ignored_slop_prefix.root(),
                QueryNode::Phrase { prefix: true, .. }
            ));

            let zero_slop = quill
                .search_preparsed_paginated(&cx, &zero_slop_prefix, corpus.len(), 0, true)
                .expect("execute zero-slop phrase prefix");
            let ignored_slop = quill
                .search_preparsed_paginated(&cx, &ignored_slop_prefix, corpus.len(), 0, true)
                .expect("execute phrase prefix carrying ignored slop");
            assert!(!zero_slop.hits.is_empty());
            assert!(zero_slop.hits.len() < corpus.len());
            assert_eq!(ignored_slop, zero_slop);
        }
    });
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn public_fast_field_exists_matches_tantivy_ids_and_score_bits() {
    use frankensearch_lexical::tantivy_crate::{
        Index, TantivyDocument,
        collector::TopDocs,
        doc,
        query::ExistsQuery,
        schema::{FAST, Schema},
    };

    run_with_cx(|cx| async move {
        let corpus = [
            ("exists-c", "gamma"),
            ("exists-a", "alpha"),
            ("exists-b", "beta"),
        ];

        let quill = QuillIndex::in_memory(deterministic_config()).expect("construct Quill");
        let documents = corpus
            .iter()
            .map(|(id, content)| IndexableDocument::new(*id, *content))
            .collect::<Vec<_>>();
        quill
            .index_documents(&cx, &documents)
            .await
            .expect("index public Quill existence corpus");
        quill
            .commit(&cx)
            .await
            .expect("commit Quill existence corpus");

        let mut schema_builder = Schema::builder();
        let ord = schema_builder.add_u64_field("ord", FAST);
        let tantivy = Index::create_in_ram(schema_builder.build());
        let mut writer = tantivy
            .writer_with_num_threads::<TantivyDocument>(1, 15_000_000)
            .expect("construct Tantivy writer");
        for (ordinal, _) in corpus.iter().enumerate() {
            writer
                .add_document(doc!(
                    ord => u64::try_from(ordinal).expect("fixture ordinal fits u64")
                ))
                .expect("index Tantivy existence document");
        }
        writer.commit().expect("commit Tantivy existence corpus");
        drop(writer);
        let reader = tantivy.reader().expect("open Tantivy existence reader");
        let searcher = reader.searcher();

        let mut quill_hits = quill
            .search_doc_ids(&cx, "ord:*", corpus.len())
            .expect("execute public Quill fast-field existence query")
            .iter()
            .map(|hit| (hit.document_id.clone(), hit.score.to_bits()))
            .collect::<Vec<_>>();
        // Tantivy 0.26's QueryParser parses `ord:*` to an existence leaf
        // and then refuses it as UnsupportedQuery ("Range query need to
        // target a specific field"), so the reference is the existence
        // query that leaf denotes, built directly.
        let existence = ExistsQuery::new("ord".to_owned(), false);
        let mut tantivy_hits = searcher
            .search(
                &existence,
                &TopDocs::with_limit(corpus.len()).order_by_score(),
            )
            .expect("execute Tantivy fast-field existence query")
            .into_iter()
            .map(|(score, address)| {
                assert_eq!(address.segment_ord, 0);
                (
                    corpus[usize::try_from(address.doc_id).expect("doc id fits")]
                        .0
                        .to_owned(),
                    score.to_bits(),
                )
            })
            .collect::<Vec<_>>();
        quill_hits.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        tantivy_hits.sort_unstable_by(|left, right| left.0.cmp(&right.0));

        assert_eq!(quill_hits, tantivy_hits);
        assert_eq!(
            quill_hits,
            ["exists-a", "exists-b", "exists-c"]
                .into_iter()
                .map(|id| (id.to_owned(), 1.0_f32.to_bits()))
                .collect::<Vec<_>>()
        );
    });
}

#[test]
fn scalar_memory_commit_is_visibility_boundary_and_queries_end_to_end() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        index
            .index_documents(&cx, &fixture_documents())
            .await
            .expect("accumulate fixture");

        let before = index
            .search_paginated(&cx, "rust", 10, 0, true)
            .expect("search old snapshot");
        assert!(before.hits.is_empty(), "uncommitted docs must be invisible");
        assert_eq!(before.total_count, Some(0));
        assert_eq!(before.doc_count, 0);

        index.commit(&cx).await.expect("publish owned snapshot");
        let term = index
            .search_paginated(&cx, "rust", 10, 0, true)
            .expect("term query");
        assert_eq!(term.total_count, Some(2));
        assert_eq!(term.doc_count, 3);
        assert_eq!(term.hits.len(), 2);
        assert!(
            term.hits
                .iter()
                .all(|hit| hit.document_id.starts_with("rust-"))
        );

        let phrase = index
            .search_paginated(&cx, "\"rust ownership\"", 10, 0, true)
            .expect("phrase query");
        assert_eq!(phrase.total_count, Some(1));
        assert_eq!(phrase.hits[0].document_id, "rust-1");

        let phrase_prefix = index
            .search_paginated(&cx, "\"rust own\"*", 10, 0, true)
            .expect("phrase-prefix query");
        assert_eq!(phrase_prefix.total_count, Some(1));
        assert_eq!(phrase_prefix.hits[0].document_id, "rust-1");

        let boolean = index
            .search_paginated(&cx, "rust AND systems", 10, 0, true)
            .expect("Boolean query");
        assert_eq!(boolean.total_count, Some(1));
        assert_eq!(boolean.hits[0].document_id, "rust-2");

        let page = index
            .search_paginated(&cx, "rust", 1, 1, true)
            .expect("paginated query");
        assert_eq!(page.total_count, Some(2));
        assert_eq!(page.hits.len(), 1);
    });
}

#[test]
fn production_writer_routes_batches_across_resolved_shards_with_disjoint_leases() {
    run_with_cx(|cx| async move {
        let config = QuillConfig {
            max_ingest_shards: 4,
            ..QuillConfig::default()
        };
        let detected = std::thread::available_parallelism().map_or(1, usize::from);
        let expected_shards = config.resolved_ingest_shards(detected);
        let mut index = QuillIndex::in_memory(config).expect("multi-shard memory index");
        assert_eq!(
            index.writer_mut().shard_router.shard_count(),
            expected_shards
        );

        for batch in 0..expected_shards.max(2) {
            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        format!("shard-{batch}"),
                        "cross shard tier policy fixture",
                    )],
                )
                .await
                .expect("route one production batch");
        }
        index.commit(&cx).await.expect("publish routed batches");
        let snapshot = index
            .snapshot()
            .expect("routed-shard snapshot is authoritative");
        assert_pairwise_disjoint_manifest(snapshot.loaded_manifest().manifest.segments.as_slice());
        if expected_shards > 1 {
            let ranges = snapshot
                .segments()
                .iter()
                .map(|segment| (segment.manifest().docid_lo, segment.manifest().docid_hi))
                .collect::<Vec<_>>();
            assert!(
                ranges
                    .iter()
                    .any(|(lo, _)| *lo >= u64::from(DOC_ORDS_PER_LEASE)),
                "a resolved multi-shard writer must consume independent Q1 leases",
            );
        }
    });
}
