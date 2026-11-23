use super::*;

// ─── Schema tests ───────────────────────────────────────────────────

#[test]
fn schema_has_required_fields() {
    let (schema, fields) = build_schema();
    assert!(schema.get_field_entry(fields.id).is_stored());
    assert!(schema.get_field_entry(fields.content).is_stored());
    assert!(schema.get_field_entry(fields.title).is_stored());
    assert!(schema.get_field_entry(fields.metadata_json).is_stored());
}

// ─── Index lifecycle tests ──────────────────────────────────────────

#[test]
fn create_in_memory() {
    let idx = TantivyIndex::in_memory().expect("create");
    assert_eq!(idx.doc_count().expect("document count"), 0);
}

#[test]
fn create_on_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let idx = TantivyIndex::create(dir.path()).expect("create");
    assert_eq!(idx.doc_count().expect("document count"), 0);
    assert_eq!(idx.path(), Some(dir.path()));
}

#[test]
fn open_nonexistent_returns_error() {
    let result = TantivyIndex::open(Path::new("/nonexistent/tantivy_index_xyz"));
    assert!(result.is_err());
}

/// Pins the ORACLE's behavior for stacked unary prefixes (`NOT -x`,
/// `NOT NOT x`), which sit OUTSIDE the contract grammar's single-prefix
/// fragment rule (`docs/contracts/quill-language-contract.md` §"fragment").
/// The pinned Tantivy 0.26.1 parser is the normative behavior for such
/// sequences; Quill's lowering must reproduce whatever this test observes
/// (bd-251nt — found by the bd-bsjw structure-aware campaign).
#[test]
fn stacked_negation_prefixes_pin_oracle_semantics() {
    run_with_cx(|cx| async move {
        let docs = vec![
            IndexableDocument::new("doc-a", "alpha need"),
            IndexableDocument::new("doc-b", "alpha other"),
            IndexableDocument::new("doc-c", "need only"),
            IndexableDocument::new("doc-d", "plain filler"),
        ];
        let idx = TantivyIndex::in_memory().expect("create");
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let observe = |query: &str| {
            let mut ids = idx
                .search_doc_ids(&cx, query, 10)
                .expect("search")
                .into_iter()
                .map(|hit| hit.doc_id)
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ids
        };

        assert_eq!(
            observe("NOT -need"),
            vec!["doc-b".to_owned(), "doc-d".to_owned()],
            "oracle collapses NOT + '-' to a single exclusion of `need`"
        );
        assert_eq!(
            observe("NOT NOT need"),
            vec![
                "doc-a".to_owned(),
                "doc-b".to_owned(),
                "doc-c".to_owned(),
                "doc-d".to_owned(),
            ],
            "oracle treats NOT NOT as a lenient-parse match-all, NOT as \
             logical double negation"
        );
        assert_eq!(
            observe("alpha NOT -need"),
            vec!["doc-b".to_owned()],
            "oracle: alpha-docs minus need-docs"
        );

        // In-GROUP stacked prefixes (bd-bsjw finding 4). `observe` here is
        // the SHIPPING surface (`search_doc_ids`), which bd-f20ye repaired,
        // so this now pins the repaired behaviour: a boost on the group no
        // longer changes membership.
        //
        // The original assertion recorded the opposite and explained it as
        // "the lenient fallback DROPS the negations". Measurement under
        // bd-f20ye falsified that account — nothing is dropped, the
        // `MustNot` survives parsing, and the lenient parser reports no
        // errors. The defect was that a boosted group re-nests its negation
        // as a positive alternative. The ORACLE-side pin of the unrepaired
        // behaviour, which is what this test's name is about, now lives in
        // `boosting_a_group_that_negates_changes_its_meaning_in_the_pinned_oracle`,
        // where it is observed through `oracle_observe_*` rather than
        // through a shipping method.
        assert_eq!(
            observe("(alpha NOT -need)"),
            vec!["doc-b".to_owned()],
            "unboosted group keeps the stacked-prefix collapse"
        );
        assert_eq!(
            observe("(alpha NOT -need)^2"),
            observe("(alpha NOT -need)"),
            "bd-f20ye: in the shipping path a boost must not change which \
             documents match"
        );
        assert_eq!(
            observe("(alpha NOT need)^2"),
            observe("(alpha NOT need)"),
            "bd-f20ye: plain NOT under a group boost is repaired in the \
             shipping path too, not just the stacked-prefix form"
        );
    });
}

/// Tantivy 0.26.1's `PhraseScorer` panics on an illegal post-termination
/// seek when a NEGATED PHRASE rides beside a positive term — found by the
/// bd-bsjw structure-aware campaign (`generic NOT "indexes Parser or
/// minimal"`). Quill executes the same shape without incident. The
/// shipping execution boundary converts the upstream panic into a typed
/// degradation instead of aborting the host process, and the index stays
/// fully usable afterwards (bd-nqeb4). When the pinned oracle is upgraded
/// past the upstream defect, the typed-error assertion here fails —
/// making the fix visible instead of silently shifting the target.
#[test]
fn oracle_negated_phrase_beside_term_degrades_typed_instead_of_panicking() {
    run_with_cx(|cx| async move {
        let docs = vec![
            IndexableDocument::new("doc-a", "alpha need only"),
            IndexableDocument::new("doc-b", "alpha other"),
            IndexableDocument::new("doc-c", "need only"),
        ];
        let idx = TantivyIndex::in_memory().expect("create");
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");
        // The phrase's terms exist but the SEQUENCE does not, so the
        // phrase docset terminates immediately — the precondition for the
        // upstream illegal seek.
        let error = idx
            .search_doc_ids(&cx, "alpha NOT \"only need\"", 10)
            .expect_err("upstream PhraseScorer defect must surface as a typed error");
        assert!(
            error
                .to_string()
                .contains("panicked during query execution"),
            "degradation must name the panic boundary: {error}"
        );

        // The availability property: the same index keeps serving ordinary
        // queries after the guarded failure.
        let survivors = idx
            .search_doc_ids(&cx, "alpha", 10)
            .expect("index must remain fully usable after a guarded panic");
        assert_eq!(survivors.len(), 2);
    });
}

#[test]
fn reopen_on_disk_restores_fast_id_materialization() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    run_with_cx(|cx| async move {
        let docs: Vec<IndexableDocument> = (0..20)
            .map(|i| {
                IndexableDocument::new(
                    format!("doc-{i:03}"),
                    format!("alpha beta document number {i} searchable content"),
                )
            })
            .collect();

        let ids_before = {
            let idx = TantivyIndex::create(&path).expect("create");
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");
            idx.search_doc_ids(&cx, "document", 20)
                .expect("search")
                .into_iter()
                .map(|h| h.doc_id)
                .collect::<Vec<_>>()
        }; // original index (and its writer lock) dropped here

        // Commit persisted the ordinal→doc_id sidecar.
        assert!(
            path.join("ord_table.json").exists(),
            "ord_table sidecar should be written on commit"
        );

        // Reopen: the sidecar restores the fast materialization path, and
        // results must be byte-identical to the pre-close ranking.
        let reopened = TantivyIndex::open(&path).expect("open");
        let ids_after = reopened
            .search_doc_ids(&cx, "document", 20)
            .expect("search")
            .into_iter()
            .map(|h| h.doc_id)
            .collect::<Vec<_>>();

        assert!(!ids_after.is_empty(), "reopened index should return hits");
        assert_eq!(
            ids_before, ids_after,
            "reopened index must return identical ranked doc_ids"
        );
    });
}

/// A read-only open must succeed while another handle still holds
/// Tantivy's exclusive writer lock, must rank identically, and must refuse
/// writes with a typed error instead of a panic or a silent no-op.
///
/// Before `open_read_only` existed, every reader went through `open`,
/// which constructs an `IndexWriter` and therefore takes
/// `.tantivy-writer.lock`; a second concurrent reader process failed the
/// open and callers silently dropped the lexical arm.
#[test]
fn open_read_only_serves_while_a_writer_holds_the_lock() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    run_with_cx(|cx| async move {
        let docs: Vec<IndexableDocument> = (0..12)
            .map(|i| {
                IndexableDocument::new(
                    format!("doc-{i:03}"),
                    format!("gamma delta document number {i} searchable content"),
                )
            })
            .collect();

        // The writer-holding handle stays alive for the whole test.
        let writer_index = TantivyIndex::create(&path).expect("create");
        writer_index
            .index_documents(&cx, &docs)
            .await
            .expect("index");
        writer_index.commit(&cx).await.expect("commit");
        let ids_via_writer = writer_index
            .search_doc_ids(&cx, "document", 20)
            .expect("search")
            .into_iter()
            .map(|h| h.doc_id)
            .collect::<Vec<_>>();
        assert!(
            path.join(".tantivy-writer.lock").exists(),
            "the writer handle must hold Tantivy's writer lock for this test to mean anything"
        );

        // A second `open` would contend for the writer lock; the read-only
        // open must not, so it succeeds while `writer_index` is alive.
        let reader = TantivyIndex::open_read_only(&path)
            .expect("read-only open must not need the writer lock");
        assert!(reader.is_read_only());
        assert!(!writer_index.is_read_only());

        let ids_via_reader = reader
            .search_doc_ids(&cx, "document", 20)
            .expect("search")
            .into_iter()
            .map(|h| h.doc_id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids_via_writer, ids_via_reader,
            "read-only handle must rank the published generation identically"
        );

        // Writes through the read-only handle fail closed with a typed error.
        let extra = IndexableDocument::new("doc-extra".to_owned(), "gamma extra".to_owned());
        let index_err = reader
            .index_document(&cx, &extra)
            .await
            .expect_err("read-only index must refuse index_document");
        assert!(
            matches!(&index_err, SearchError::InvalidConfig { field, value, .. }
                if field == "tantivy.writer.index" && value == "read_only"),
            "unexpected error for read-only index_document: {index_err}"
        );
        let commit_err = reader
            .commit(&cx)
            .await
            .expect_err("read-only index must refuse commit");
        assert!(
            matches!(&commit_err, SearchError::InvalidConfig { field, .. }
                if field == "tantivy.writer.commit"),
            "unexpected error for read-only commit: {commit_err}"
        );
        let delete_err = reader
            .delete_document(&cx, "doc-000")
            .await
            .expect_err("read-only index must refuse delete_document");
        assert!(
            matches!(&delete_err, SearchError::InvalidConfig { field, .. }
                if field == "tantivy.writer.delete"),
            "unexpected error for read-only delete: {delete_err}"
        );

        // The refused writes changed nothing visible to either handle.
        let ids_after = reader
            .search_doc_ids(&cx, "document", 20)
            .expect("search")
            .into_iter()
            .map(|h| h.doc_id)
            .collect::<Vec<_>>();
        assert_eq!(ids_via_writer, ids_after);
    });
}

// ─── Indexing tests ─────────────────────────────────────────────────

#[test]
fn map_writer_lock_error_polled_after_completion_maps_to_subsystem_error() {
    let err = TantivyIndex::map_writer_lock_error(
        "tantivy.index",
        asupersync::sync::LockError::PolledAfterCompletion,
    );
    assert!(
        matches!(err, SearchError::SubsystemError { .. }),
        "expected subsystem error, got {err:?}"
    );
    if let SearchError::SubsystemError { source, .. } = err {
        assert!(
            source
                .to_string()
                .contains("writer mutex future reused after completion during tantivy.index")
        );
    }
}

#[test]
fn index_single_document() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "Hello world");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");
        assert_eq!(idx.doc_count().expect("document count"), 1);
    });
}

#[test]
fn index_batch_documents() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("batch index");
        idx.commit(&cx).await.expect("commit");
        assert_eq!(idx.doc_count().expect("document count"), 5);
    });
}

#[test]
fn upsert_replaces_existing_document() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc_v1 = IndexableDocument::new("doc-1", "Version one content");
        idx.index_document(&cx, &doc_v1).await.expect("index v1");
        idx.commit(&cx).await.expect("commit v1");

        let doc_v2 = IndexableDocument::new("doc-1", "Version two content updated");
        idx.index_document(&cx, &doc_v2).await.expect("index v2");
        idx.commit(&cx).await.expect("commit v2");

        let results = idx.search(&cx, "updated", 10).await.expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "doc-1");
    });
}

// ─── Search tests ───────────────────────────────────────────────────

#[cfg(feature = "bench-internals")]
#[test]
fn benchmark_schema_without_positions_uses_presence_only_postings() {
    let idx = TantivyIndex::in_memory_with_benchmark_config(50_000_000, 1, false)
        .expect("create position-free benchmark oracle");
    run_with_cx(|cx| async move {
        idx.index_documents(
            &cx,
            &[
                IndexableDocument::new(
                    "repeated",
                    "term00001 term00001 term00001 term00001 term00001 qgpreflight",
                ),
                IndexableDocument::new("single", "term00001 qgpreflight"),
                IndexableDocument::new("decoy", "term00002 qgpreflight"),
            ],
        )
        .await
        .expect("index position-free oracle fixture");
        idx.commit(&cx)
            .await
            .expect("commit position-free oracle fixture");

        let hits = idx
            .search(&cx, "term00001", 2)
            .await
            .expect("search position-free oracle fixture");
        let ids: Vec<_> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
        assert_eq!(ids, ["single", "repeated"]);
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn benchmark_writer_join_rearms_without_changing_searchable_state() {
    let idx = TantivyIndex::in_memory_with_benchmark_config(50_000_000, 1, true)
        .expect("create benchmark oracle");
    run_with_cx(|cx| async move {
        idx.index_document(
            &cx,
            &IndexableDocument::new("before-join", "lifecycle fence"),
        )
        .await
        .expect("index pre-fence document");
        idx.commit(&cx).await.expect("commit pre-fence document");

        let (idx, receipt) = idx
            .benchmark_join_workers_and_rearm(50_000_000, 1)
            .expect("join workers and rearm writer");
        assert!(receipt.searchable_segments_before > 0);
        assert!(receipt.searchable_segments_after > 0);
        assert!(receipt.writer_rearmed);
        assert_eq!(
            idx.search(&cx, "lifecycle", 10)
                .await
                .expect("search pre-fence document")
                .len(),
            1
        );

        idx.index_document(
            &cx,
            &IndexableDocument::new("after-join", "writer remains usable"),
        )
        .await
        .expect("index post-fence document");
        idx.commit(&cx).await.expect("commit post-fence document");
        assert_eq!(
            idx.search(&cx, "writer", 10)
                .await
                .expect("search post-fence document")
                .len(),
            1
        );
    });
}

#[cfg(feature = "bench-internals")]
#[test]
fn benchmark_writer_join_terminal_retains_searchable_reader_without_rearm() {
    let idx = TantivyIndex::in_memory_with_benchmark_config(50_000_000, 1, true)
        .expect("create benchmark oracle");
    run_with_cx(|cx| async move {
        idx.index_document(
            &cx,
            &IndexableDocument::new("terminal-join", "one shot bulk fixture"),
        )
        .await
        .expect("index terminal-fence document");
        idx.commit(&cx)
            .await
            .expect("commit terminal-fence document");

        let (reader, receipt) = idx
            .benchmark_join_workers_retaining_reader()
            .expect("join workers while retaining the terminal reader");
        assert!(receipt.searchable_segments_before > 0);
        assert!(receipt.searchable_segments_after > 0);
        assert!(!receipt.writer_rearmed);
        let tail_ids = reader
            .benchmark_search_exact_id("terminal-join")
            .expect("search the prepared terminal tail after worker join");
        assert_eq!(
            tail_ids.iter().map(DocId::as_str).collect::<Vec<_>>(),
            ["terminal-join"],
            "a counted segment is not accepted in place of a post-join tail search"
        );
        assert!(
            reader
                .benchmark_search_exact_id("missing-terminal-tail")
                .expect("search missing terminal tail after worker join")
                .is_empty(),
            "planted negative: the retained reader must not fabricate tail visibility"
        );
    });
}

#[test]
fn search_empty_query_returns_empty() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "", 10).await.expect("search");
        assert!(results.is_empty());

        let results = idx.search(&cx, "   ", 10).await.expect("search whitespace");
        assert!(results.is_empty());
    });
}

#[test]
fn deferred_fusion_candidates_restore_exact_metadata() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let full = idx.search(&cx, "Rust", 10).await.expect("full search");
        let batch = idx
            .search_candidates(&cx, "Rust", 10)
            .await
            .expect("fusion candidates");

        // `is_deferred` is the split-trait replacement for the retired
        // `fusion_metadata_is_deferred` capability flag.
        assert!(batch.is_deferred());
        let (mut candidates, pin) = batch.into_parts();
        assert_eq!(candidates.len(), full.len());
        assert!(candidates.iter().all(|result| result.metadata.is_none()));
        for (candidate, expected) in candidates.iter().zip(&full) {
            assert_eq!(candidate.doc_id, expected.doc_id);
            assert_eq!(candidate.score.to_bits(), expected.score.to_bits());
            assert_eq!(
                candidate.lexical_score.map(f32::to_bits),
                expected.lexical_score.map(f32::to_bits)
            );
        }

        idx.hydrate_candidates(&cx, pin.as_ref(), &mut candidates)
            .await
            .expect("hydrate winners");
        for (candidate, expected) in candidates.iter().zip(&full) {
            assert_eq!(candidate.metadata, expected.metadata);
        }
    });
}

#[test]
fn search_returns_relevant_results() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "Rust", 10).await.expect("search");
        assert!(!results.is_empty(), "should find documents mentioning Rust");
        let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(ids.contains(&"doc-1"), "should find doc-1");
        assert!(ids.contains(&"doc-3"), "should find doc-3");
    });
}

#[test]
fn search_respects_limit() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx
            .search(&cx, "machine learning", 1)
            .await
            .expect("search");
        assert_eq!(results.len(), 1);
    });
}

#[test]
fn search_results_have_lexical_source() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "Rust", 5).await.expect("search");
        for r in &results {
            assert_eq!(r.source, ScoreSource::Lexical);
            assert!(r.lexical_score.is_some());
            assert!(r.lexical_score.unwrap() > 0.0);
            assert!(r.fast_score.is_none());
            assert!(r.quality_score.is_none());
            assert!(r.rerank_score.is_none());
            assert!(r.explanation.is_none());
        }
    });
}

#[test]
fn search_scores_are_descending() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "language", 10).await.expect("search");
        if results.len() > 1 {
            for pair in results.windows(2) {
                assert!(
                    pair[0].score >= pair[1].score,
                    "scores should be descending: {} >= {}",
                    pair[0].score,
                    pair[1].score
                );
            }
        }
    });
}

#[test]
fn title_boost_affects_ranking() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc_a = IndexableDocument::new("doc-a", "consensus algorithm for distributed systems");
        let doc_b = IndexableDocument::new("doc-b", "some distributed system design")
            .with_title("Consensus Protocol");

        idx.index_document(&cx, &doc_a).await.expect("index a");
        idx.index_document(&cx, &doc_b).await.expect("index b");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "consensus", 2).await.expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].doc_id, "doc-b",
            "title-boosted document should rank first"
        );
    });
}

#[test]
fn metadata_preserved_in_results() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "test content")
            .with_metadata("source", "unit_test")
            .with_metadata("lang", "en");

        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "test", 1).await.expect("search");
        assert_eq!(results.len(), 1);

        let meta = results[0].metadata.as_ref().expect("metadata present");
        assert_eq!(meta["source"], "unit_test");
        assert_eq!(meta["lang"], "en");
    });
}

#[test]
fn no_results_for_unmatched_query() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "xylophone", 10).await.expect("search");
        assert!(results.is_empty(), "no documents mention xylophone");
    });
}

#[test]
fn zero_limit_returns_empty_without_collector_panic() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        idx.index_document(&cx, &IndexableDocument::new("doc-1", "searchable text"))
            .await
            .expect("index");
        idx.commit(&cx).await.expect("commit");

        let parsed = idx.parse_query_lenient("searchable");
        let searcher = idx.reader.searcher();
        let counted = execute_query_with_offset(&searcher, &*parsed, 0, 0)
            .expect("counted zero-limit search");
        assert_eq!(counted.hits, [] as [LexicalDocHit; 0]);
        assert_eq!(counted.total_count, 1);
        assert_eq!(
            execute_top_k(&searcher, &*parsed, 0, 0).expect("count-free zero-limit search"),
            [] as [LexicalDocHit; 0]
        );

        assert!(
            idx.search(&cx, "searchable", 0)
                .await
                .expect("search")
                .is_empty()
        );
        assert_eq!(
            idx.search_doc_ids(&cx, "searchable", 0).expect("id search"),
            [] as [LexicalIdHit; 0]
        );
        assert!(
            idx.search_with_snippets(&cx, "searchable", 0, &SnippetConfig::default())
                .expect("snippet search")
                .is_empty()
        );
    });
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn oracle_tie_expansion_preserves_the_shipping_top_k() {
    let idx = TantivyIndex::in_memory_single_threaded_oracle().expect("create oracle");
    run_with_cx(|cx| async move {
        idx.writer
            .lock(&cx)
            .await
            .expect("lock writer")
            .as_mut()
            .expect("an in-memory index carries a writer")
            .set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));

        const DOC_COUNT: usize = 256;
        const SEGMENT_SIZE: usize = 32;
        const LIMIT: usize = 100;
        const QUERY: &str = "term00001 term00007 generated record";
        for segment_start in (0..DOC_COUNT).step_by(SEGMENT_SIZE) {
            let docs = (segment_start..segment_start + SEGMENT_SIZE)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("tie-{ordinal:03}"),
                        "term00001 term00007 generated record",
                    )
                })
                .collect::<Vec<_>>();
            idx.index_documents(&cx, &docs)
                .await
                .expect("index tie segment");
            idx.commit(&cx).await.expect("commit tie segment");
        }

        let native = idx
            .search_doc_ids(&cx, QUERY, LIMIT)
            .expect("shipping top-k");
        let observed = idx
            .oracle_observe_query(
                &cx,
                QUERY,
                LIMIT,
                DOC_COUNT,
                &SnippetConfig {
                    max_chars: 0,
                    ..SnippetConfig::default()
                },
            )
            .expect("expanded oracle observation");
        let native_rows = native
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.bm25_score.to_bits()))
            .collect::<Vec<_>>();
        let observed_rows = observed
            .hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
            .collect::<Vec<_>>();

        assert_eq!(
            observed_rows, native_rows,
            "expanded tie evidence must not redefine the shipping top-k"
        );
        assert_eq!(observed.total_count, DOC_COUNT);
        assert!(observed.cutoff_tie_complete);
        assert_eq!(observed.cutoff_tie_group.len(), DOC_COUNT);
    });
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn oracle_counted_pages_keep_exact_totals_and_absolute_ranks() {
    let idx = TantivyIndex::in_memory_single_threaded_oracle().expect("create oracle");
    run_with_cx(|cx| async move {
        let docs = (0..9)
            .map(|ordinal| {
                IndexableDocument::new(
                    format!("doc-{ordinal:02}"),
                    format!("shared counted-page term document {ordinal}"),
                )
            })
            .collect::<Vec<_>>();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        idx.index_document(
            &cx,
            &IndexableDocument::new("staged-doc", "shared counted-page term document staged"),
        )
        .await
        .expect("stage uncommitted document");

        let count_only = idx
            .oracle_observe_page(&cx, "counted-page", 0, 0)
            .expect("count-only page");
        let first = idx
            .oracle_observe_page(&cx, "counted-page", 3, 0)
            .expect("first page");
        let second = idx
            .oracle_observe_page(&cx, "counted-page", 3, 3)
            .expect("second page");
        let prefix = idx
            .oracle_observe_page(&cx, "counted-page", 6, 0)
            .expect("combined prefix");
        let past_end = idx
            .oracle_observe_page(&cx, "counted-page", 3, 100)
            .expect("past-end page");
        let ranked = idx
            .oracle_observe_query(&cx, "counted-page", 3, 16, &SnippetConfig::default())
            .expect("ranked observation");

        for page in [&count_only, &first, &second, &prefix, &past_end] {
            assert_eq!(page.total_count, 9, "exact Count must ignore k and offset");
            assert_eq!(
                page.doc_count, 9,
                "oracle document count must share the page searcher's generation"
            );
        }
        assert_eq!(ranked.total_count, 9);
        assert_eq!(
            ranked.doc_count, 9,
            "ranked oracle document count must ignore staged writer state"
        );
        assert!(
            count_only.hits.is_empty(),
            "limit zero must retain Count only"
        );
        assert!(past_end.hits.is_empty(), "past-end offset must be empty");
        assert_eq!(first.hits.len(), 3);
        assert_eq!(second.hits.len(), 3);
        assert_eq!(prefix.hits.len(), 6);

        for (page_rank, hit) in first.hits.iter().enumerate() {
            assert_eq!(hit.page_rank, page_rank);
            assert_eq!(hit.absolute_rank, page_rank);
        }
        for (page_rank, hit) in second.hits.iter().enumerate() {
            assert_eq!(hit.page_rank, page_rank);
            assert_eq!(hit.absolute_rank, page_rank + 3);
        }
        let combined = first
            .hits
            .iter()
            .chain(&second.hits)
            .map(|hit| (&hit.doc_id, hit.score_bits))
            .collect::<Vec<_>>();
        let prefix = prefix
            .hits
            .iter()
            .map(|hit| (&hit.doc_id, hit.score_bits))
            .collect::<Vec<_>>();
        assert_eq!(combined, prefix, "offset pages must compose exactly");
    });
}

#[test]
fn tantivy_indexing_enforces_max_token_len_boundary() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let max = tantivy::tokenizer::MAX_TOKEN_LEN;
        let kept = "k".repeat(max);
        let dropped = "d".repeat(max + 1);
        idx.index_document(&cx, &IndexableDocument::new("kept", &kept))
            .await
            .expect("index boundary token");
        idx.index_document(&cx, &IndexableDocument::new("dropped", &dropped))
            .await
            .expect("index oversized token document");
        idx.commit(&cx).await.expect("commit");

        let searcher = idx.reader.searcher();
        let kept_query = TermQuery::new(
            Term::from_field_text(idx.fields.content, &kept),
            IndexRecordOption::WithFreqsAndPositions,
        );
        let dropped_query = TermQuery::new(
            Term::from_field_text(idx.fields.content, &dropped),
            IndexRecordOption::WithFreqsAndPositions,
        );
        assert_eq!(searcher.search(&kept_query, &Count).expect("search"), 1);
        assert_eq!(searcher.search(&dropped_query, &Count).expect("search"), 0);
    });
}

// ─── Delete tests ───────────────────────────────────────────────────

#[test]
fn delete_document_removes_from_index() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");
        assert_eq!(idx.doc_count().expect("document count"), 5);

        idx.delete_document(&cx, "doc-1").await.expect("delete");
        idx.commit(&cx).await.expect("commit after delete");

        let results = idx.search(&cx, "Rust systems", 10).await.expect("search");
        assert!(
            !results.iter().any(|r| r.doc_id == "doc-1"),
            "deleted document should not appear"
        );
    });
}

#[test]
fn delete_document_publishes_to_public_lexical_readers_before_return() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let retained = IndexableDocument::new("retained", "delete visibility sentinel");
        let deleted = IndexableDocument::new("deleted", "delete visibility sentinel");
        LexicalWrite::index_documents(&idx, &cx, &[retained, deleted])
            .await
            .expect("index documents through the public write trait");
        LexicalWrite::commit(&idx, &cx)
            .await
            .expect("publish documents through the public write trait");

        assert_eq!(
            LexicalRead::doc_count(&idx).expect("count published documents"),
            2
        );

        idx.delete_document(&cx, "deleted")
            .await
            .expect("delete must publish before returning");

        let hits = LexicalRead::search(&idx, &cx, "delete visibility sentinel", 10)
            .await
            .expect("public lexical reader must see the committed deletion");
        assert_eq!(
            hits.iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            vec!["retained"]
        );
        assert_eq!(
            LexicalRead::doc_count(&idx).expect("count refreshed readable generation"),
            1
        );
    });
}

// ─── Edge case tests ────────────────────────────────────────────────

#[test]
fn search_with_special_characters() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "Error code ERR-404: page not found");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "ERR-404", 10).await.expect("search");
        assert!(!results.is_empty(), "should find hyphenated term");
    });
}

#[test]
fn case_insensitive_search() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "Rust Programming Language");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "rust", 10).await.expect("search lowercase");
        assert!(!results.is_empty());

        let results = idx.search(&cx, "RUST", 10).await.expect("search uppercase");
        assert!(!results.is_empty());
    });
}

#[test]
fn empty_metadata_not_stored() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "no metadata here");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search(&cx, "metadata", 1).await.expect("search");
        assert_eq!(results.len(), 1);
        assert!(results[0].metadata.is_none());
    });
}

#[test]
fn doc_count_accurate_after_operations() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        assert_eq!(idx.doc_count().expect("document count"), 0);

        let doc = IndexableDocument::new("doc-1", "first");
        idx.index_document(&cx, &doc).await.expect("index");
        assert_eq!(
            idx.doc_count().expect("pre-commit document count"),
            0,
            "the count must describe the current searcher, not staged writer state"
        );
        idx.commit(&cx).await.expect("commit");
        assert_eq!(idx.doc_count().expect("document count"), 1);

        let doc = IndexableDocument::new("doc-2", "second");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");
        assert_eq!(idx.doc_count().expect("document count"), 2);

        idx.delete_document(&cx, "doc-1").await.expect("delete");
        idx.commit(&cx).await.expect("commit delete");
        assert_eq!(idx.doc_count().expect("document count"), 1);
    });
}

// ─── Trait object safety ────────────────────────────────────────────

#[test]
fn tantivy_index_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TantivyIndex>();
}

// ─── On-disk persistence tests ──────────────────────────────────────

#[test]
fn reopen_preserves_documents() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Phase 1: Create and populate.
    {
        let idx = TantivyIndex::create(dir.path()).expect("create");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let doc = IndexableDocument::new("doc-1", "persistent content");
            idx.index_document(&cx, &doc).await.expect("index");
            idx.commit(&cx).await.expect("commit");
        });
    }

    // Phase 2: Reopen and verify.
    {
        let idx = TantivyIndex::open(dir.path()).expect("open");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let results = idx.search(&cx, "persistent", 10).await.expect("search");
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc-1");
        });
    }
}

/// Default-feature coverage for the writer-call observation.
///
/// Without this the observation is written on every build but only read behind
/// `bench-internals`, so a default `--all-targets` run would carry a field that
/// is never read. The fix is a real reader, not an allow: the ordinary
/// in-memory constructor genuinely must reach `Index::writer`, and that is
/// worth asserting on its own.
#[test]
fn the_ordinary_in_memory_constructor_invokes_the_auto_writer() {
    let index = TantivyIndex::in_memory().expect("ordinary in-memory index");
    assert_eq!(
        index.observed_writer_call,
        WriterCall::Auto,
        "the ordinary constructor must reach Index::writer, not a pinned width"
    );
}
