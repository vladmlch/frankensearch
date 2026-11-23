use super::*;

// ─── Query explanation tests (bd-3un.18) ─────────────────────────────

#[test]
fn classify_empty_query() {
    assert_eq!(classify_query(""), QueryExplanation::Empty);
    assert_eq!(classify_query("   "), QueryExplanation::Empty);
}

#[test]
fn classify_simple_query() {
    assert_eq!(classify_query("rust"), QueryExplanation::Simple);
    assert_eq!(
        classify_query("  authentication  "),
        QueryExplanation::Simple
    );
}

#[test]
fn classify_phrase_query() {
    assert_eq!(
        classify_query("\"error handling\""),
        QueryExplanation::Phrase
    );
    assert_eq!(classify_query("'single quotes'"), QueryExplanation::Phrase);
}

#[test]
fn classify_boolean_query() {
    assert_eq!(classify_query("rust async"), QueryExplanation::Boolean);
    assert_eq!(
        classify_query("distributed consensus algorithm"),
        QueryExplanation::Boolean
    );
}

#[test]
fn query_explanation_display() {
    assert_eq!(QueryExplanation::Empty.to_string(), "empty");
    assert_eq!(QueryExplanation::Simple.to_string(), "simple");
    assert_eq!(QueryExplanation::Phrase.to_string(), "phrase");
    assert_eq!(QueryExplanation::Boolean.to_string(), "boolean");
}

#[test]
fn query_explanation_serde_roundtrip() {
    let json = serde_json::to_string(&QueryExplanation::Phrase).unwrap();
    let decoded: QueryExplanation = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, QueryExplanation::Phrase);
}

// ─── Snippet config tests ────────────────────────────────────────────

#[test]
fn snippet_config_default() {
    let config = SnippetConfig::default();
    assert_eq!(config.max_chars, DEFAULT_SNIPPET_MAX_CHARS);
    assert_eq!(config.highlight_prefix, "<b>");
    assert_eq!(config.highlight_postfix, "</b>");
}

#[test]
fn snippet_unicode_window_uses_tantivy_byte_offsets() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        idx.index_document(&cx, &IndexableDocument::new("unicode", "éé alpha"))
            .await
            .expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig {
            max_chars: 6,
            ..SnippetConfig::default()
        };
        let results = idx
            .search_with_snippets(&cx, "éé", 1, &config)
            .expect("snippet search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet.as_deref(), Some("<b>éé</b>"));
    });
}

// ─── search_with_snippets tests (bd-3un.18) ─────────────────────────

#[test]
fn search_with_snippets_returns_results() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig::default();
        let results = idx
            .search_with_snippets(&cx, "Rust", 10, &config)
            .expect("search");

        assert!(!results.is_empty());
        assert_eq!(results[0].rank, 0);
        assert_eq!(results[0].query_type, QueryExplanation::Simple);
        assert!(results[0].bm25_score > 0.0);
    });
}

#[test]
fn search_doc_ids_returns_ranked_identifiers() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx.search_doc_ids(&cx, "Rust", 10).expect("search");
        assert_ne!(results, [] as [LexicalIdHit; 0]);
        for (expected_rank, hit) in results.iter().enumerate() {
            assert_eq!(hit.rank, expected_rank, "rank should be sequential");
            assert!(!hit.doc_id.is_empty());
            assert!(hit.bm25_score.is_finite());
        }
    });
}

#[test]
fn all_doc_ids_enumerates_only_live_committed_documents() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = [
            IndexableDocument::new("doc-c", "gamma"),
            IndexableDocument::new("doc-a", "alpha"),
            IndexableDocument::new("doc-b", "beta"),
        ];
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");
        idx.delete_document(&cx, "doc-b").await.expect("delete");
        idx.commit(&cx).await.expect("commit delete");

        assert_eq!(
            idx.all_doc_ids().expect("enumerate ids"),
            vec![DocId::from("doc-a"), DocId::from("doc-c")]
        );
        let exported = idx.all_documents().expect("export documents");
        assert_eq!(
            exported
                .iter()
                .map(|document| document.id.as_str())
                .collect::<Vec<_>>(),
            vec!["doc-a", "doc-c"]
        );
        assert_eq!(exported[0].content, "alpha");
        assert_eq!(exported[1].content, "gamma");
    });
}

#[test]
fn every_live_tantivy_search_surface_has_one_total_order_under_score_ties() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        idx.writer
            .lock(&cx)
            .await
            .expect("lock writer")
            .as_mut()
            .expect("an in-memory index carries a writer")
            .set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));

        const DOC_COUNT: usize = 120;
        const SEGMENT_SIZE: usize = 30;
        for segment_start in (0..DOC_COUNT).step_by(SEGMENT_SIZE) {
            let docs = (segment_start..segment_start + SEGMENT_SIZE)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("tie-{ordinal:03}"),
                        "alpha beta gamma delta shared exact score",
                    )
                    .with_title("alpha beta heading")
                    .with_metadata("ordinal", ordinal.to_string())
                })
                .collect::<Vec<_>>();
            idx.index_documents(&cx, &docs)
                .await
                .expect("index tie segment");
            idx.commit(&cx).await.expect("commit tie segment");
        }

        let query_shapes = [
            "alpha",
            "alpha beta",
            "alpha beta gamma",
            "\"alpha beta\"",
            "alpha AND beta",
            "title:alpha",
        ];
        for query in query_shapes {
            for limit in [0, 1, DOC_COUNT - 1, DOC_COUNT, DOC_COUNT + 1] {
                assert_live_lexical_contract(&idx, &cx, query, limit, DOC_COUNT).await;
            }
        }
        assert_live_lexical_contract(&idx, &cx, "", 20, 0).await;
        assert_live_lexical_contract(&idx, &cx, "term-not-present", 20, 0).await;
    });
}

#[test]
fn harvested_natural_language_query_keeps_count_off_the_public_candidate_path() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let documents = shared_full120_docs();
        for batch in documents.chunks(5) {
            idx.index_documents(&cx, batch)
                .await
                .expect("index Full120 campaign batch");
        }
        idx.commit(&cx).await.expect("commit Full120");

        const QUERY: &str = "how to sear a steak properly";
        const LIMIT: usize = 20;
        let full = idx.search(&cx, QUERY, LIMIT).await.expect("full search");
        reset_collector_invocations();
        let ids = idx
            .search_doc_ids(&cx, QUERY, LIMIT)
            .expect("count-free ID search");
        let candidates = idx
            .search_candidates(&cx, QUERY, LIMIT)
            .await
            .expect("fusion candidates")
            .into_parts()
            .0;
        assert_eq!(
            collector_invocations(),
            (0, 2),
            "both public candidate surfaces must construct TopDocs-only and never Count"
        );
        let counted = idx
            .search_doc_ids_counted(&cx, QUERY, LIMIT)
            .expect("counted benchmark control");
        assert_eq!(
            collector_invocations(),
            (1, 2),
            "only the explicit benchmark control may construct (TopDocs, Count)"
        );

        assert_eq!(full.len(), LIMIT);
        assert_eq!(ids.len(), full.len());
        assert_eq!(candidates.len(), full.len());
        assert_eq!(counted.len(), LIMIT);
        for (rank, ((expected, id_hit), candidate)) in
            full.iter().zip(&ids).zip(&candidates).enumerate()
        {
            let context = format!("harvested-14 rank={rank}");
            assert_eq!(id_hit.rank, rank, "{context}");
            assert_eq!(id_hit.doc_id, expected.doc_id, "{context}");
            assert_eq!(
                id_hit.bm25_score.to_bits(),
                expected.score.to_bits(),
                "{context}"
            );
            assert_eq!(candidate.doc_id, expected.doc_id, "{context}");
            assert_eq!(
                candidate.score.to_bits(),
                expected.score.to_bits(),
                "{context}"
            );
        }

        let parsed = idx.parse_query_lenient(QUERY);
        let counted_page = execute_query_with_offset(&idx.reader.searcher(), &*parsed, LIMIT, 0)
            .expect("exact counted API");
        assert_eq!(
            counted_page.total_count, 110,
            "the pinned Full120 harvested-14 exact count drifted"
        );
    });
}

#[test]
fn search_with_snippets_empty_query() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig::default();
        let results = idx
            .search_with_snippets(&cx, "", 10, &config)
            .expect("search");
        assert!(results.is_empty());
    });
}

#[test]
fn search_with_snippets_has_highlighted_content() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new(
            "doc-1",
            "The Rust programming language is fast and memory-safe",
        );
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig::default();
        let results = idx
            .search_with_snippets(&cx, "Rust", 1, &config)
            .expect("search");

        assert_eq!(results.len(), 1);
        if let Some(snippet) = &results[0].snippet {
            assert!(
                snippet.contains("<b>"),
                "snippet should have highlight tags: {snippet}"
            );
        }
    });
}

#[test]
fn search_with_snippets_custom_highlight_tags() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "Rust is awesome for systems programming");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig {
            max_chars: 200,
            highlight_prefix: "<em>".to_owned(),
            highlight_postfix: "</em>".to_owned(),
        };
        let results = idx
            .search_with_snippets(&cx, "Rust", 1, &config)
            .expect("search");

        assert_eq!(results.len(), 1);
        if let Some(snippet) = &results[0].snippet {
            assert!(
                snippet.contains("<em>"),
                "snippet should use custom highlight: {snippet}"
            );
            assert!(
                !snippet.contains("<b>"),
                "should NOT use default highlight: {snippet}"
            );
        }
    });
}

#[test]
fn search_with_snippets_ranks_are_sequential() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig::default();
        let results = idx
            .search_with_snippets(&cx, "language", 10, &config)
            .expect("search");

        for (i, hit) in results.iter().enumerate() {
            assert_eq!(hit.rank, i, "rank should be sequential");
        }
    });
}

#[test]
fn search_with_snippets_metadata_preserved() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc =
            IndexableDocument::new("doc-1", "metadata test content").with_metadata("key", "value");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig::default();
        let results = idx
            .search_with_snippets(&cx, "metadata", 1, &config)
            .expect("search");

        assert_eq!(results.len(), 1);
        let meta = results[0].metadata.as_ref().expect("metadata");
        assert_eq!(meta["key"], "value");
    });
}

#[test]
fn search_with_snippets_phrase_query() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "error handling in Rust is explicit");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig::default();
        let results = idx
            .search_with_snippets(&cx, "\"error handling\"", 10, &config)
            .expect("search");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].query_type, QueryExplanation::Phrase);
    });
}

#[test]
fn search_with_snippets_boolean_query() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let docs = sample_docs();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let config = SnippetConfig::default();
        let results = idx
            .search_with_snippets(&cx, "machine learning", 10, &config)
            .expect("search");

        assert!(!results.is_empty());
        assert_eq!(results[0].query_type, QueryExplanation::Boolean);
    });
}

// ─── Query truncation tests ──────────────────────────────────────────

#[test]
fn truncate_query_short_passthrough() {
    let q = "hello world";
    assert_eq!(TantivyIndex::truncate_query(q), q);
}

#[test]
fn truncate_query_at_limit() {
    let q = "a".repeat(MAX_QUERY_LENGTH);
    assert_eq!(TantivyIndex::truncate_query(&q), q.as_str());
}

#[test]
fn truncate_query_over_limit() {
    let q = "a".repeat(MAX_QUERY_LENGTH + 100);
    let truncated = TantivyIndex::truncate_query(&q);
    assert_eq!(truncated.len(), MAX_QUERY_LENGTH);
    assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
}

#[test]
fn truncate_query_counts_multibyte_characters() {
    let over = "\u{00E9}".repeat(MAX_QUERY_LENGTH + 3);
    let truncated = TantivyIndex::truncate_query(&over);
    assert!(truncated.is_char_boundary(truncated.len()));
    assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
    assert_eq!(truncated.len(), MAX_QUERY_LENGTH * '\u{00E9}'.len_utf8());
}

#[test]
fn truncate_query_preserves_multibyte_query_within_character_limit() {
    let query = "\u{00E9}".repeat(MAX_QUERY_LENGTH / 2 + 1);
    assert!(query.len() > MAX_QUERY_LENGTH);
    assert_eq!(TantivyIndex::truncate_query(&query), query);
}

#[test]
fn overlong_query_still_searches() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "findable content");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        // Build a query with the search term followed by lots of padding.
        let mut long_query = "findable ".to_owned();
        long_query.push_str(&"padding ".repeat(2000));
        assert!(long_query.len() > MAX_QUERY_LENGTH);

        let results = idx
            .search(&cx, &long_query, 10)
            .await
            .expect("should not error");
        assert!(
            !results.is_empty(),
            "truncated query should still find docs"
        );
    });
}

// ─── LexicalHit serde test ───────────────────────────────────────────

#[test]
fn lexical_hit_serde_roundtrip() {
    let hit = LexicalHit {
        doc_id: "doc-42".into(),
        bm25_score: 2.75,
        rank: 0,
        snippet: Some("<b>Rust</b> is great".into()),
        query_type: QueryExplanation::Simple,
        metadata: Some(serde_json::json!({"lang": "en"})),
    };
    let json = serde_json::to_string(&hit).expect("serialize");
    let roundtripped: LexicalHit = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(roundtripped.doc_id, "doc-42");
    assert!((roundtripped.bm25_score - 2.75).abs() < f32::EPSILON);
    assert_eq!(roundtripped.rank, 0);
    assert_eq!(
        roundtripped.snippet.as_deref(),
        Some("<b>Rust</b> is great")
    );
    assert_eq!(roundtripped.query_type, QueryExplanation::Simple);
}

// ─── Special character robustness tests ──────────────────────────────

#[test]
fn search_with_special_chars_no_crash() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "some content");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        // These queries contain special characters that could trip up a parser.
        // In lenient mode, Tantivy should handle them gracefully.
        for query in &["@user", "#hashtag", "foo:bar", "a+b", "hello!"] {
            let result = idx.search(&cx, query, 10).await;
            assert!(result.is_ok(), "query '{query}' should not error");
        }
    });
}

#[test]
fn search_no_results_returns_empty_not_error() {
    let idx = TantivyIndex::in_memory().expect("create");
    run_with_cx(|cx| async move {
        let doc = IndexableDocument::new("doc-1", "hello world");
        idx.index_document(&cx, &doc).await.expect("index");
        idx.commit(&cx).await.expect("commit");

        let results = idx
            .search(&cx, "nonexistentterm", 10)
            .await
            .expect("no error");
        assert!(results.is_empty());
    });
}

/// Tantivy candidates hydrate from the generation that scored them.
///
/// The same contract Quill already proves, now that `TantivyIndex`
/// overrides `search_candidates`/`hydrate_candidates` instead of falling
/// back to the eager default (`bd-8nqz.1`). Scores come from generation
/// N; a commit publishes N+1; the retained batch must still hydrate N's
/// metadata while a fresh search observes N+1.
#[test]
fn tantivy_candidates_hydrate_from_the_scoring_generation() {
    use frankensearch_core::traits::LexicalRead;

    run_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("tempdir");
        let idx = TantivyIndex::create(dir.path()).expect("create");

        let v1 = IndexableDocument::new("doc-a", "alpha pinned content").with_metadata("rev", "v1");
        LexicalWrite::index_document(&idx, &cx, &v1)
            .await
            .expect("index N");
        LexicalWrite::commit(&idx, &cx).await.expect("publish N");

        // Score on N. Candidates defer metadata and carry the searcher pin.
        let batch = LexicalRead::search_candidates(&idx, &cx, "pinned", 10)
            .await
            .expect("candidates on N");
        assert!(
            batch.is_deferred(),
            "Tantivy candidates must carry a searcher pin, not fall back to eager"
        );
        assert_eq!(batch.results().len(), 1);
        assert!(
            batch.results()[0].metadata.is_none(),
            "deferred candidates skip metadata until hydration"
        );

        // Publish N+1 with different metadata for the same document.
        let v2 = IndexableDocument::new("doc-a", "alpha pinned content").with_metadata("rev", "v2");
        LexicalWrite::index_document(&idx, &cx, &v2)
            .await
            .expect("upsert N+1");
        LexicalWrite::commit(&idx, &cx).await.expect("publish N+1");

        let rev_of = |result: &ScoredResult| -> Option<String> {
            result.metadata.as_ref().and_then(|metadata| {
                metadata
                    .get("rev")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
            })
        };

        // A fresh search observes N+1.
        let fresh = LexicalRead::search(&idx, &cx, "pinned", 10)
            .await
            .expect("fresh search");
        assert_eq!(rev_of(&fresh[0]).as_deref(), Some("v2"));

        // The retained batch hydrates from its PINNED generation N.
        let (mut winners, context) = batch.into_parts();
        LexicalRead::hydrate_candidates(&idx, &cx, context.as_ref(), &mut winners)
            .await
            .expect("pinned hydration");
        assert_eq!(
            rev_of(&winners[0]).as_deref(),
            Some("v1"),
            "hydration must read the scoring searcher, not the newest generation"
        );
    });
}

/// A foreign or missing context is refused rather than silently reading
/// the current generation.
#[test]
fn tantivy_hydration_rejects_a_foreign_or_absent_context() {
    use frankensearch_core::traits::{LexicalHydrationContext, LexicalRead};

    run_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("tempdir");
        let idx = TantivyIndex::create(dir.path()).expect("create");
        let doc = IndexableDocument::new("doc-a", "alpha").with_metadata("rev", "v1");
        LexicalWrite::index_document(&idx, &cx, &doc)
            .await
            .expect("index");
        LexicalWrite::commit(&idx, &cx).await.expect("commit");

        let mut winners = vec![ScoredResult {
            doc_id: DocId::from("doc-a"),
            score: 1.0,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(1.0),
            rerank_score: None,
            explanation: None,
            metadata: None,
        }];

        let absent = LexicalRead::hydrate_candidates(&idx, &cx, None, &mut winners)
            .await
            .expect_err("deferred winners without their pin must not hydrate silently");
        assert!(matches!(
            absent,
            SearchError::SubsystemError {
                subsystem: "tantivy.hydration",
                ..
            }
        ));

        let foreign = LexicalHydrationContext::new("quill", Box::new(7_u64));
        let error = LexicalRead::hydrate_candidates(&idx, &cx, Some(&foreign), &mut winners)
            .await
            .expect_err("a foreign pin must be refused");
        match error {
            SearchError::SubsystemError { subsystem, source } => {
                assert_eq!(subsystem, "tantivy.hydration");
                assert!(
                    source.to_string().contains("quill"),
                    "the rejection must name the foreign backend"
                );
            }
            other => panic!("expected a typed subsystem error, got {other:?}"),
        }
    });
}
