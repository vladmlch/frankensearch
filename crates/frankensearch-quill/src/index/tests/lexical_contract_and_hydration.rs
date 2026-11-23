use super::*;

#[test]
fn stored_value_hydration_does_not_rematerialize_candidate_identity() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(QuillConfig {
            deterministic_ingest: true,
            tier_fanout: usize::MAX,
            ..QuillConfig::default()
        })
        .expect("create multi-segment stored-value index");
        for document in [
            IndexableDocument::new("segment-one", "alpha filler"),
            IndexableDocument::new("segment-two", "beta filler"),
            IndexableDocument::new("stored-target", "needle exact stored bytes")
                .with_metadata("path", "src/target.rs"),
        ] {
            LexicalWrite::index_document(&index, &cx, &document)
                .await
                .expect("index one segment fixture document");
            LexicalWrite::commit(&index, &cx)
                .await
                .expect("seal one fixture segment");
        }

        let snapshot = index.search_snapshot().expect("pin fixture snapshot");
        assert_eq!(snapshot.keeper.segments().len(), 3);
        let materialization_call_count = || {
            snapshot
                .keeper
                .segments()
                .iter()
                .map(RecoveredSegment::document_id_materialization_call_count)
                .sum::<u64>()
        };
        assert_eq!(materialization_call_count(), 0);

        let batch = LexicalRead::search_candidates(&index, &cx, "needle", 10)
            .await
            .expect("search deferred candidate");
        let (mut candidates, pin) = batch.into_parts();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].doc_id, "stored-target");
        assert_eq!(materialization_call_count(), 1);

        LexicalRead::hydrate_candidates(&index, &cx, pin.as_ref(), &mut candidates)
            .await
            .expect("hydrate deferred candidate metadata");
        assert_eq!(
            candidates[0].metadata.as_deref(),
            Some(&serde_json::json!({"path": "src/target.rs"}))
        );

        let global_docid = snapshot
            .resolve_document_id("stored-target")
            .expect("resolve stored target")
            .expect("stored target is live");
        assert_eq!(
            snapshot
                .materialize_stored_value(CONTENT_FIELD, global_docid)
                .expect("materialize stored content"),
            Some(b"needle exact stored bytes".to_vec())
        );
        assert_eq!(materialization_call_count(), 1);

        assert!(
            index
                .delete_document(&cx, "stored-target")
                .await
                .expect("tombstone stored target")
        );
        let tombstoned = index
            .search_snapshot()
            .expect("pin tombstoned successor snapshot");
        assert_eq!(
            tombstoned
                .materialize_stored_value(CONTENT_FIELD, global_docid)
                .expect("read tombstoned stored content"),
            None
        );
        assert_eq!(
            tombstoned
                .keeper
                .segments()
                .iter()
                .map(RecoveredSegment::document_id_materialization_call_count)
                .sum::<u64>(),
            1
        );
    });
}

#[test]
fn stored_value_hydration_parent_red_reuses_one_plan_for_two_sealed_winners() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(QuillConfig {
            deterministic_ingest: true,
            tier_fanout: usize::MAX,
            ..QuillConfig::default()
        })
        .expect("create same-segment stored-value index");
        for document in [
            IndexableDocument::new("stored-first", "needle first payload")
                .with_metadata("path", "src/first.rs"),
            IndexableDocument::new("stored-second", "needle second payload")
                .with_metadata("path", "src/second.rs"),
        ] {
            LexicalWrite::index_document(&index, &cx, &document)
                .await
                .expect("stage stored-value winner");
        }
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("seal both stored-value winners together");

        let snapshot = index.search_snapshot().expect("pin sealed winner snapshot");
        assert_eq!(snapshot.keeper.segments().len(), 1);
        let segment = &snapshot.keeper.segments()[0];
        let exact_segment_bytes = segment.source_bytes().to_vec();
        assert_eq!(segment.stored_meta_lookup_cache_counts(), (0, 0));

        let first_docid = snapshot
            .resolve_document_id("stored-first")
            .expect("resolve first winner")
            .expect("first winner is live");
        let second_docid = snapshot
            .resolve_document_id("stored-second")
            .expect("resolve second winner")
            .expect("second winner is live");

        let batch = LexicalRead::search_candidates(&index, &cx, "needle", 10)
            .await
            .expect("select two deferred stored-value winners");
        let (mut winners, pin) = batch.into_parts();
        assert_eq!(
            winners.len(),
            2,
            "fixture must hydrate two same-segment winners"
        );
        assert!(winners.iter().all(|winner| winner.metadata.is_none()));

        LexicalRead::hydrate_candidates(&index, &cx, pin.as_ref(), &mut winners)
            .await
            .expect("hydrate both stored-value winners");
        for winner in &winners {
            match winner.doc_id.as_str() {
                "stored-first" => assert_eq!(
                    winner.metadata.as_deref(),
                    Some(&serde_json::json!({"path": "src/first.rs"}))
                ),
                "stored-second" => assert_eq!(
                    winner.metadata.as_deref(),
                    Some(&serde_json::json!({"path": "src/second.rs"}))
                ),
                other => panic!("unexpected stored-value winner {other}"), // ubs:ignore -- Test rejects an unexpected stored-value winner.
            }
        }
        assert_eq!(segment.source_bytes(), exact_segment_bytes.as_slice());
        let plan_counts = segment.stored_meta_lookup_cache_counts();
        assert_eq!(plan_counts, (2, 1));

        assert_eq!(
            snapshot
                .materialize_stored_value(u16::MAX, first_docid)
                .expect("read absent stored field"),
            None,
            "an absent stored value must remain absent"
        );
        assert!(
            index
                .delete_document(&cx, "stored-second")
                .await
                .expect("tombstone second winner")
        );
        let tombstoned = index
            .search_snapshot()
            .expect("pin tombstoned same-backing successor");
        assert_eq!(
            tombstoned
                .materialize_stored_value(CONTENT_FIELD, second_docid)
                .expect("read tombstoned stored content"),
            None,
            "a tombstoned row must not materialize stored bytes"
        );
        assert_eq!(
            tombstoned.keeper.segments()[0].stored_meta_lookup_cache_counts(),
            (3, 1),
            "tombstone-only rebinds share the one plan only for the exact immutable backing"
        );
    });
}

/// A document's explanation reproduces its search score, and its per
/// clause parts add up to that score: field boosts (the title's 2x) and
/// query boosts (`^3`) are carried into each part, a term the document
/// lacks scores zero, prohibited clauses are not itemized, and a document
/// the query does not match has no explanation.
#[test]
fn explain_document_parts_add_up_to_the_search_score() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("create index");
        for document in [
            IndexableDocument::new("guide", "ownership ownership borrow rules")
                .with_title("Ownership guide"),
            IndexableDocument::new("elision", "the lifetime elision rule"),
            IndexableDocument::new("both", "ownership and lifetime together"),
            IndexableDocument::new("garden", "an unrelated line about gardening"),
        ] {
            LexicalWrite::index_document(&index, &cx, &document)
                .await
                .expect("index document");
        }
        LexicalWrite::commit(&index, &cx).await.expect("commit");

        for query in [
            "ownership lifetime",
            "ownership^3 lifetime",
            "\"lifetime elision\" ownership",
        ] {
            let hits = index
                .search_paginated(&cx, query, 10, 0, false)
                .expect("search")
                .hits;
            assert!(hits.len() >= 2, "{query}: {hits:?}");
            for hit in hits.iter() {
                let explanation = index
                    .explain_document(&cx, query, &hit.document_id)
                    .expect("explain")
                    .expect("a ranked document has an explanation");
                assert_eq!(explanation.score, hit.score, "{query} {}", hit.document_id);
                let parts = explanation
                    .terms
                    .iter()
                    .map(|term| f64::from(term.score))
                    .sum::<f64>();
                assert!(
                    (parts - f64::from(hit.score)).abs() <= 1e-4 * f64::from(hit.score),
                    "{query} {}: parts {parts} vs score {} in {:?}",
                    hit.document_id,
                    hit.score,
                    explanation.terms
                );
            }
        }

        let guide = index
            .explain_document(&cx, "ownership lifetime", "guide")
            .expect("explain")
            .expect("guide matches");
        let part = |text: &str, field: &str| {
            guide
                .terms
                .iter()
                .find(|term| term.text == text && term.field == field)
                .unwrap_or_else(|| panic!("no {text}/{field} in {:?}", guide.terms))
        };
        assert!(part("ownership", "title").score > 0.0);
        assert!(part("ownership", "content").score > 0.0);
        assert!(part("lifetime", "content").score.abs() < f32::EPSILON);
        assert!(part("lifetime", "title").score.abs() < f32::EPSILON);
        let ownership = part("ownership", "content");
        assert_eq!(ownership.kind, QuillClauseKind::Term);
        assert_eq!(ownership.doc_freq, Some(2));
        assert_eq!(ownership.idf, Some(crate::contract::idf(2, 4)));

        let phrase = index
            .explain_document(&cx, "\"lifetime elision\"", "elision")
            .expect("explain")
            .expect("the phrase matches");
        assert!(
            phrase
                .terms
                .iter()
                .any(|term| term.kind == QuillClauseKind::Phrase
                    && term.text == "lifetime elision"
                    && term.field == "content"
                    && term.score > 0.0)
        );

        let prohibited = index
            .explain_document(&cx, "ownership -gardening", "both")
            .expect("explain")
            .expect("both matches");
        assert!(prohibited.terms.iter().all(|term| term.text == "ownership"));

        assert_eq!(
            index
                .explain_document(&cx, "ownership", "garden")
                .expect("explain"),
            None
        );
        assert_eq!(
            index
                .explain_document(&cx, "ownership", "missing")
                .expect("explain"),
            None
        );
    });
}

/// Snippets for chosen documents equal the ranked snippet tail's for what
/// it returns (so the same terms and document-frequency weights pick the
/// window), fall back to the opening without query terms, stop at
/// `max_source_bytes`, and skip documents that are not live.
#[test]
fn snippets_for_documents_match_the_ranked_snippets() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("create index");
        let filler = "filler ".repeat(40);
        let mixed = format!("ownership {filler}lifetime");
        let deep = format!("opening words {filler}ownership");
        for (id, text) in [
            ("common", "ownership ownership borrow rules"),
            ("rare", "the lifetime elision rule"),
            ("none", "an unrelated opening line about gardening"),
            ("mixed", mixed.as_str()),
            ("deep", deep.as_str()),
        ] {
            LexicalWrite::index_document(&index, &cx, &IndexableDocument::new(id, text))
                .await
                .expect("index document");
        }
        LexicalWrite::commit(&index, &cx).await.expect("commit");

        let config = crate::SnippetConfig::default();
        let query = "ownership lifetime";
        let ranked = index
            .search_with_snippets(&cx, query, 10, &config)
            .expect("ranked snippets");
        assert!(ranked.iter().any(|hit| hit.document_id == "mixed"));
        let ids = ranked
            .iter()
            .map(|hit| hit.document_id.as_str())
            .collect::<Vec<_>>();
        let chosen = index
            .snippets_for_documents(&cx, query, &ids, &config, usize::MAX)
            .expect("chosen snippets");
        for (hit, snippet) in ranked.iter().zip(&chosen) {
            assert_eq!(snippet, &hit.snippet, "{}", hit.document_id);
        }

        let extra = index
            .snippets_for_documents(&cx, query, &["none", "missing", "deep"], &config, 40)
            .expect("snippets beyond the ranked tail");
        assert!(
            extra[0]
                .as_deref()
                .is_some_and(|snippet| snippet.starts_with("an unrelated opening line")),
            "{:?}",
            extra[0]
        );
        assert_eq!(extra[1], None);
        assert!(
            extra[2]
                .as_deref()
                .is_some_and(|snippet| snippet.starts_with("opening words")
                    && !snippet.contains("ownership")),
            "{:?}",
            extra[2]
        );
    });
}

#[test]
fn quill_lexical_contract_is_immediate_hydratable_cancel_safe_and_upserts() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("create lexical backend");
        assert!(index.path().is_none());

        let original = IndexableDocument::new("quill-doc", "native quill ownership backend")
            .with_metadata("path", "src/lib.rs")
            .with_metadata("lang", "rust");
        LexicalWrite::index_document(&index, &cx, &original)
            .await
            .expect("index through lexical trait");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit through lexical trait");
        assert_eq!(
            LexicalRead::doc_count(&index).expect("read lexical document count"),
            1
        );

        let full = LexicalRead::search(&index, &cx, "ownership", 10)
            .await
            .expect("full lexical search");
        assert_eq!(full.len(), 1);
        let expected_metadata = serde_json::json!({
            "lang": "rust",
            "path": "src/lib.rs",
        });
        assert_eq!(full[0].metadata.as_deref(), Some(&expected_metadata));

        let mut candidate_future = LexicalRead::search_candidates(&index, &cx, "ownership", 10);
        let batch = {
            let waker = Waker::noop();
            let mut task_cx = Context::from_waker(waker);
            match candidate_future.as_mut().poll(&mut task_cx) {
                Poll::Ready(result) => result.expect("fusion candidates"),
                Poll::Pending => panic!("Quill fusion candidates must be first-poll ready"), // ubs:ignore -- Test requires first-poll readiness.
            }
        };
        drop(candidate_future);
        // `is_deferred` replaces the retired `fusion_metadata_is_deferred`
        // capability flag: the batch itself now reports the shape.
        assert!(batch.is_deferred());
        let (candidates, pin) = batch.into_parts();
        assert_eq!(candidates.len(), full.len());
        assert_eq!(candidates[0].doc_id, full[0].doc_id);
        assert_eq!(candidates[0].score.to_bits(), full[0].score.to_bits());
        assert!(candidates[0].metadata.is_none());

        let mut hydrated = candidates;
        LexicalRead::hydrate_candidates(&index, &cx, pin.as_ref(), &mut hydrated)
            .await
            .expect("hydrate fusion winners");
        assert_eq!(hydrated[0].metadata, full[0].metadata);

        let ids = index
            .search_doc_ids(&cx, "ownership", 10)
            .expect("identifier-only public search");
        assert_eq!(ids[0].document_id, "quill-doc");
        let enriched = index
            .search_with_snippets(&cx, "ownership", 10, &crate::SnippetConfig::default())
            .expect("enriched public search");
        assert_eq!(enriched[0].metadata.as_deref(), Some(&expected_metadata));
        assert!(
            enriched[0]
                .snippet
                .as_deref()
                .is_some_and(|snippet| snippet.contains("<b>ownership</b>"))
        );

        let replacement = IndexableDocument::new("quill-doc", "python replacement backend")
            .with_metadata("lang", "python");
        LexicalWrite::index_document(&index, &cx, &replacement)
            .await
            .expect("upsert replacement through lexical trait");
        assert!(
            LexicalRead::search(&index, &cx, "ownership", 10)
                .await
                .expect("search removed content before explicit commit")
                .is_empty()
        );
        assert_eq!(
            LexicalRead::search(&index, &cx, "replacement", 10)
                .await
                .expect("search replacement before explicit commit")
                .len(),
            1
        );
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("no-op commit after atomic replacement");
        assert!(
            LexicalRead::search(&index, &cx, "ownership", 10)
                .await
                .expect("search removed content")
                .is_empty()
        );
        let replacement_hits = LexicalRead::search(&index, &cx, "replacement", 10)
            .await
            .expect("search replacement content");
        assert_eq!(replacement_hits.len(), 1);
        assert_eq!(
            replacement_hits[0].metadata.as_deref(),
            Some(&serde_json::json!({"lang": "python"}))
        );
        assert_eq!(
            index.segment_stats().expect("read segment stats").live_docs,
            1
        );

        assert!(
            index
                .delete_document(&cx, "quill-doc")
                .await
                .expect("delete live document")
        );
        assert!(
            !index
                .delete_document(&cx, "quill-doc")
                .await
                .expect("repeat delete")
        );
        assert_eq!(
            index
                .doc_count()
                .expect("deleted document count is authoritative"),
            0
        );

        LexicalWrite::index_documents(
            &index,
            &cx,
            &[
                IndexableDocument::new("first", "alpha"),
                IndexableDocument::new("second", "beta"),
                IndexableDocument::new("third", "gamma"),
            ],
        )
        .await
        .expect("repopulate through lexical trait");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit repopulated backend");
        assert_eq!(
            index
                .delete_documents(&cx, &["first", "missing", "second"])
                .await
                .expect("delete bounded document batch"),
            2
        );
        assert_eq!(
            index
                .doc_count()
                .expect("batch-delete count is authoritative"),
            1
        );
        assert_eq!(
            LexicalRead::search(&index, &cx, "gamma", 10)
                .await
                .expect("search batch-delete survivor")
                .len(),
            1
        );
        index.delete_all(&cx).await.expect("delete all documents");
        assert_eq!(
            index
                .doc_count()
                .expect("delete-all count is authoritative"),
            0
        );

        let cancelled = Cx::for_testing();
        cancelled.set_cancel_requested(true);
        assert!(matches!(
            LexicalRead::search(&index, &cancelled, "alpha", 10).await,
            Err(SearchError::Cancelled { ref phase, .. }) if phase == "search"
        ));
        assert!(matches!(
            LexicalWrite::index_document(
                &index,
                &cancelled,
                &IndexableDocument::new("cancelled", "never indexed"),
            )
            .await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(
            index
                .doc_count()
                .expect("cancelled mutation count is authoritative"),
            0
        );
    });
}

#[test]
fn lexical_trait_batch_upsert_publishes_one_atomic_generation() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let original = [
            IndexableDocument::new("first", "old alpha"),
            IndexableDocument::new("second", "old beta"),
            IndexableDocument::new("third", "old gamma"),
        ];
        LexicalWrite::index_documents(&index, &cx, &original)
            .await
            .expect("seed lexical documents");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("publish seed generation");
        let seed_generation = index
            .segment_stats()
            .expect("read seed segment stats")
            .published_generation;

        let replacements = [
            IndexableDocument::new("first", "new alpha"),
            IndexableDocument::new("second", "new beta"),
            IndexableDocument::new("third", "new gamma"),
        ];
        LexicalWrite::index_documents(&index, &cx, &replacements)
            .await
            .expect("publish replacement batch");
        assert_eq!(
            index
                .segment_stats()
                .expect("read replacement segment stats")
                .published_generation,
            seed_generation.saturating_add(1),
            "one upsert batch must publish tombstones and replacements together"
        );
        assert!(!index.has_uncommitted_changes());
        assert_eq!(
            index
                .search_doc_ids(&cx, "new", 10)
                .expect("replacement batch is immediately searchable")
                .len(),
            3
        );

        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit after replacement is a no-op");
        assert_eq!(
            index
                .segment_stats()
                .expect("read committed segment stats")
                .published_generation,
            seed_generation.saturating_add(1)
        );
        assert_eq!(
            index
                .doc_count()
                .expect("atomic upsert count is authoritative"),
            3
        );
    });
}

/// A replacement issued while an unrelated batch is still accumulating
/// publishes that batch first instead of refusing the call.
///
/// The engine-neutral `LexicalWrite` surface must not make an ordinary
/// interleaving succeed on one backend and fail on another, so the three
/// outcomes pinned here are: the call is accepted, the pending batch is not
/// lost, and the replacement actually replaces.
#[test]
fn lexical_trait_replacement_publishes_a_pending_batch_instead_of_refusing() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("seeded", "old alpha")],
        )
        .await
        .expect("seed published document");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("publish seed generation");

        LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("pending", "fresh beta")],
        )
        .await
        .expect("stage an unrelated pending batch");
        assert!(
            index.has_uncommitted_changes(),
            "the pending batch must still be unpublished before the replacement"
        );

        LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("seeded", "new gamma")],
        )
        .await
        .expect("replacement must be accepted while a batch is pending");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit after replacement");

        assert_eq!(
            index
                .search_doc_ids(&cx, "gamma", 10)
                .expect("replacement is searchable")
                .len(),
            1
        );
        assert!(
            index
                .search_doc_ids(&cx, "alpha", 10)
                .expect("superseded revision is retired")
                .is_empty(),
            "the replaced revision must not remain live"
        );
        assert_eq!(
            index
                .search_doc_ids(&cx, "beta", 10)
                .expect("pending batch survives")
                .len(),
            1,
            "the batch that was pending must not be lost"
        );
        assert_eq!(
            index.doc_count().expect("published count is authoritative"),
            2
        );
    });
}

/// A rejected replacement batch must publish NOTHING, even though
/// accepting it would have published pending work first.
///
/// Publication is a side effect of accepting the call, so the publish-first
/// path reopens `bd-quill-rejected-ingest-publishes-partial-batch-aihri`
/// unless malformed input is refused before the commit: otherwise
/// `[live replacement, invalid]` publishes an unrelated pending document
/// and only then fails. Each assertion below fails differently — the
/// generation catches the publish, the search catches the leaked document,
/// and the uncommitted flag catches the drained writer state.
#[test]
fn lexical_trait_rejected_replacement_batch_publishes_nothing() {
    run_with_cx(|cx| async move {
        // Only the empty identifier is reachable here. A repeated id is NOT
        // an error on this surface: the facade collapses a batch to the
        // final occurrence of each id before the writer sees it, which is
        // the pinned last-write-wins behavior. The preflight still checks
        // for repeats because the writer is also reached without that
        // collapse; that arm is covered directly by
        // [`commit_invariant_batch_shape_refuses_what_publishing_cannot_fix`].
        for (label, batch) in [(
            "empty id",
            vec![
                IndexableDocument::new("seeded", "new gamma"),
                IndexableDocument::new("", "invalid"),
            ],
        )] {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            LexicalWrite::index_documents(
                &index,
                &cx,
                &[IndexableDocument::new("seeded", "old alpha")],
            )
            .await
            .expect("seed published document");
            LexicalWrite::commit(&index, &cx)
                .await
                .expect("publish seed generation");
            let published = index
                .segment_stats()
                .expect("read seed segment stats")
                .published_generation;

            LexicalWrite::index_documents(
                &index,
                &cx,
                &[IndexableDocument::new("pending", "fresh beta")],
            )
            .await
            .expect("stage an unrelated pending batch");

            let rejected = LexicalWrite::index_documents(&index, &cx, &batch).await;
            assert!(rejected.is_err(), "{label} must be refused");
            assert_eq!(
                index
                    .segment_stats()
                    .expect("read segment stats")
                    .published_generation,
                published,
                "{label} must not advance the published generation"
            );
            assert!(
                index
                    .search_doc_ids(&cx, "beta", 10)
                    .expect("pending batch visibility")
                    .is_empty(),
                "{label} must not publish the pending document"
            );
            assert_eq!(
                index
                    .search_doc_ids(&cx, "alpha", 10)
                    .expect("seeded revision is untouched")
                    .len(),
                1,
                "{label} must leave the published revision in place"
            );
            assert!(
                index.has_uncommitted_changes(),
                "{label} must leave the pending batch staged"
            );
        }
    });
}

/// An armed retry guard with NOTHING staged must still refuse, and must not
/// leave a deletion behind.
///
/// This is the flag-only shape, and it is the dangerous one precisely
/// because it looks clean: `has_uncommitted_changes` does not consult
/// `ingest_retry_required`, so a cancelled ingest that accumulated no
/// documents leaves a writer with no shard rows, no staged flush and no
/// retained proposal. Installing the replacement MANIFEST on that state
/// stages a TOMBSTONE for the live document and then fails, and the
/// explicit retry commit the guard demands publishes that deletion with no
/// replacement rows — the document is destroyed by a call that returned an
/// error.
///
/// The third assertion is the one that catches it: refusing is not enough
/// if a proposal survives the refusal.
#[test]
fn lexical_trait_armed_retry_guard_alone_refuses_without_staging_a_deletion() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("seeded", "old alpha")],
        )
        .await
        .expect("seed published document");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("publish seed generation");
        let published = index
            .segment_stats()
            .expect("read seed segment stats")
            .published_generation;

        // Flag only: no accumulated documents, no staged flush, no retained
        // proposal. This is what a cancelled ingest can leave behind.
        index.writer_mut().ingest_retry_required = true;
        assert!(
            !index.has_uncommitted_changes(),
            "the writer must look clean to has_uncommitted_changes for this to be the flag-only shape"
        );

        let refused = LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("seeded", "new gamma")],
        )
        .await;
        assert!(
            refused.is_err(),
            "an armed retry guard must refuse a replacement, got {refused:?}"
        );
        assert!(
            index.writer_mut().pending_replacement_manifest.is_none(),
            "a refused replacement must not retain a deletion-only MANIFEST proposal"
        );

        // The retry the guard asks for must publish nothing.
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("explicit retry commit");
        assert_eq!(
            index
                .search_doc_ids(&cx, "alpha", 10)
                .expect("published revision after retry")
                .len(),
            1,
            "the retry commit must not destroy the live document"
        );
        assert!(
            index
                .search_doc_ids(&cx, "gamma", 10)
                .expect("refused replacement after retry")
                .is_empty(),
            "the refused replacement must not appear"
        );
        assert_eq!(
            index.doc_count().expect("document count after retry"),
            1,
            "the retry commit must not publish a deletion"
        );
        assert_eq!(
            index
                .segment_stats()
                .expect("read segment stats after retry")
                .published_generation,
            published,
            "a refused replacement leaves nothing for the retry to publish"
        );
    });
}

/// Seed three committed documents that all match `alpha` and each match
/// their own ordinal word, so a hollowed generation shows up both as a
/// count and as missing hits (gh#45 fixtures). Returns the published
/// generation.
async fn seed_three_alpha_documents(cx: &Cx, index: &QuillIndex) -> u64 {
    LexicalWrite::index_documents(
        index,
        cx,
        &[
            IndexableDocument::new("gh45-1", "alpha one"),
            IndexableDocument::new("gh45-2", "alpha two"),
            IndexableDocument::new("gh45-3", "alpha three"),
        ],
    )
    .await
    .expect("seed three documents");
    LexicalWrite::commit(index, cx)
        .await
        .expect("publish the seed generation");
    assert_eq!(index.doc_count().expect("seed doc count"), 3);
    index
        .segment_stats()
        .expect("seed segment stats")
        .published_generation
}

fn hit_count(index: &QuillIndex, cx: &Cx, query: &str) -> usize {
    index
        .search_doc_ids(cx, query, 10)
        // ubs:ignore -- Search failure must fail the test helper rather than produce a count.
        .unwrap_or_else(|error| panic!("search {query:?}: {error}"))
        .len()
}

/// gh#45 reproduction, admission shape: a replacement batch that is REFUSED
/// before any row is accumulated must not leave its tombstones behind.
///
/// The writer stages a tombstone for every identity the batch replaces,
/// retains that proposal, and only then runs batch admission. Before the
/// fix a refused batch returned its error with the proposal still
/// retained, and the next `commit()` published it: a valid MANIFEST whose
/// only change was the deletion of the document the caller tried to
/// update.
#[test]
fn refused_replacement_batch_leaves_no_deletion_for_commit_to_publish() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let seeded = seed_three_alpha_documents(&cx, &index).await;

        // "gh45-1" is live, so this batch is a replacement; the empty id
        // fails admission after the tombstone for gh45-1 is staged.
        let refused = LexicalWrite::index_documents(
            &index,
            &cx,
            &[
                IndexableDocument::new("gh45-1", "omega one"),
                IndexableDocument::new("", "nameless"),
            ],
        )
        .await;
        assert!(
            refused.is_err(),
            "an empty id must be refused, got {refused:?}"
        );
        {
            let writer = index.writer_mut();
            assert!(
                writer.pending_replacement_manifest.is_none(),
                "a refused replacement must not retain its tombstoned proposal"
            );
            assert!(
                !writer.ingest_retry_required,
                "nothing was mutated, so nothing needs a retry"
            );
        }
        assert!(
            !index.has_uncommitted_changes(),
            "the writer must be exactly as clean as before the refused batch"
        );

        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit after a refused batch");
        assert_eq!(
            index.doc_count().expect("doc count"),
            3,
            "commit after a refused replacement must not delete anything"
        );
        assert_eq!(hit_count(&index, &cx, "alpha"), 3);
        assert_eq!(
            hit_count(&index, &cx, "one"),
            1,
            "the document the refused batch tried to replace is still served"
        );
        assert_eq!(hit_count(&index, &cx, "omega"), 0);
        assert_eq!(
            published_generation(&index),
            seeded,
            "there was nothing to publish, so no generation advanced"
        );

        // A well-formed retry of the same replacement lands normally.
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("gh45-1", "omega one")],
        )
        .await
        .expect("retry the replacement");
        assert_eq!(index.doc_count().expect("doc count after retry"), 3);
        assert_eq!(hit_count(&index, &cx, "omega"), 1);
        assert_eq!(hit_count(&index, &cx, "alpha"), 2);
    });
}

/// gh#45 reproduction, partial shape: a replacement batch that fails after
/// SOME of its rows were accumulated must not leave either half behind.
///
/// Cancellation lands once one of three replacement rows sits in the shard
/// accumulator. Before the fix the writer kept the tombstoned proposal AND
/// the accumulated row, armed the retry guard, and the retry commit it
/// asked for published a generation serving one document where three had
/// been live. Publishing the row without the proposal would be no better:
/// two live rows for one identity.
#[test]
fn replacement_cancelled_midway_keeps_the_previous_generation_authoritative() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let seeded = seed_three_alpha_documents(&cx, &index).await;
        let replacements = [
            IndexableDocument::new("gh45-1", "omega one"),
            IndexableDocument::new("gh45-2", "omega two"),
            IndexableDocument::new("gh45-3", "omega three"),
        ];

        index.writer_mut().cancel_ingest_after_documents = Some(1);
        let cancelled = LexicalWrite::index_documents(&index, &cx, &replacements).await;
        index.writer_mut().cancel_ingest_after_documents = None;
        cx.set_cancel_requested(false);
        assert!(
            matches!(
                &cancelled,
                Err(frankensearch_core::SearchError::Cancelled { .. })
            ),
            "the batch must fail with the typed cancellation, got {cancelled:?}"
        );
        {
            let writer = index.writer_mut();
            assert!(
                writer.pending_replacement_manifest.is_none(),
                "a failed replacement must not retain its tombstoned proposal"
            );
            assert!(
                !writer.ingest_retry_required,
                "the discarded transaction leaves nothing to retry"
            );
            assert!(
                writer
                    .shards
                    .iter()
                    .all(|shard| shard.accumulator.document_count() == 0),
                "partially accumulated replacement rows must be discarded with the proposal"
            );
            assert!(writer.uncommitted_ids.is_empty());
        }
        assert!(!index.has_uncommitted_changes());

        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit after a cancelled batch");
        assert_eq!(
            index.doc_count().expect("doc count"),
            3,
            "commit after a cancelled replacement must serve every previously live document"
        );
        assert_eq!(hit_count(&index, &cx, "alpha"), 3);
        assert_eq!(
            hit_count(&index, &cx, "omega"),
            0,
            "no replacement row may leak out of a failed batch"
        );
        assert_eq!(published_generation(&index), seeded);

        // The same batch, retried whole, replaces all three.
        LexicalWrite::index_documents(&index, &cx, &replacements)
            .await
            .expect("retry the full replacement");
        assert_eq!(index.doc_count().expect("doc count after retry"), 3);
        assert_eq!(hit_count(&index, &cx, "omega"), 3);
        assert_eq!(hit_count(&index, &cx, "alpha"), 0);
    });
}

/// The commit path refuses a proposal that would hollow the generation,
/// independently of how it got there (gh#45, second layer).
///
/// The writer is put directly into the shape the pre-fix code left behind
/// — a retained tombstone-only replacement proposal with no rows — so this
/// keeps guarding `commit()` even if some path other than `upsert` ever
/// manages to retain such a proposal.
#[test]
fn commit_refuses_a_retained_tombstone_only_replacement_proposal() {
    run_with_cx(|cx| async move {
        let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let seeded = seed_three_alpha_documents(&cx, &index).await;
        let committed = index.snapshot().expect("authoritative snapshot");
        let mut hollow = committed.next_manifest().expect("next manifest");
        assert!(
            committed
                .delete_document(&mut hollow, "gh45-2")
                .expect("stage the tombstone")
        );
        index.writer_mut().pending_replacement_manifest = Some(hollow);

        match index.commit(&cx).await {
            Err(QuillIndexError::Keeper(KeeperError::LiveDocumentFloor {
                previous_live_docs,
                proposed_live_docs,
                ..
            })) => {
                assert_eq!((previous_live_docs, proposed_live_docs), (3, 2));
            }
            Err(other) => {
                panic!("commit must refuse with the live-document floor, got {other:?}") // ubs:ignore -- Test requires the exact live-document refusal.
            }
            Ok(_) => panic!("commit must refuse a proposal that deletes without replacing"), // ubs:ignore -- Test rejects a destructive proposal.
        }
        assert_eq!(index.doc_count().expect("doc count"), 3);
        assert_eq!(hit_count(&index, &cx, "alpha"), 3);
        assert_eq!(published_generation(&index), seeded);
        assert!(
            !index.has_uncommitted_changes(),
            "an unpublishable proposal is discarded, not retained for a retry that can only fail again"
        );

        // The writer stays usable: a no-op commit and a real replacement
        // both succeed.
        index
            .commit(&cx)
            .await
            .expect("commit after the refusal has nothing to publish");
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("gh45-2", "omega two")],
        )
        .await
        .expect("replacement after the refusal");
        assert_eq!(index.doc_count().expect("doc count"), 3);
        assert_eq!(hit_count(&index, &cx, "omega"), 1);
    });
}

/// Explicit deletions are the opt-in the floor exists to preserve: they
/// publish under `PublishIntent::ExplicitDeletion` and keep working,
/// including a full replace with FEWER documents through `delete_all`
/// (gh#45).
#[test]
fn explicit_deletions_still_publish_smaller_generations() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let seeded = seed_three_alpha_documents(&cx, &index).await;

        assert_eq!(
            index
                .delete_documents(&cx, &["gh45-1", "gh45-3"])
                .await
                .expect("delete two of three"),
            2
        );
        assert_eq!(index.doc_count().expect("doc count"), 1);
        assert_eq!(hit_count(&index, &cx, "alpha"), 1);
        assert_eq!(hit_count(&index, &cx, "two"), 1);
        assert_eq!(published_generation(&index), seeded + 1);

        index.delete_all(&cx).await.expect("delete all");
        assert_eq!(index.doc_count().expect("doc count"), 0);
        assert_eq!(hit_count(&index, &cx, "alpha"), 0);
        assert_eq!(published_generation(&index), seeded + 2);

        // Explicit full replace with fewer documents: an emptied corpus
        // followed by a smaller one.
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[
                IndexableDocument::new("gh45-1", "omega one"),
                IndexableDocument::new("gh45-2", "omega two"),
            ],
        )
        .await
        .expect("index the smaller corpus");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("publish the smaller corpus");
        assert_eq!(index.doc_count().expect("doc count"), 2);
        assert_eq!(hit_count(&index, &cx, "omega"), 2);
        assert_eq!(hit_count(&index, &cx, "alpha"), 0);
    });
}

/// An ordinary partial replacement — one of three identities superseded,
/// its row landing — is a one-for-one swap and passes the floor unchanged
/// (gh#45 control).
#[test]
fn successful_partial_replacement_is_unaffected_by_the_floor() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        let seeded = seed_three_alpha_documents(&cx, &index).await;

        LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("gh45-2", "omega two")],
        )
        .await
        .expect("replace one of three");
        assert_eq!(index.doc_count().expect("doc count"), 3);
        assert_eq!(hit_count(&index, &cx, "omega"), 1);
        assert_eq!(hit_count(&index, &cx, "alpha"), 2);
        assert_eq!(
            hit_count(&index, &cx, "two"),
            1,
            "the superseded row is gone and its replacement is served"
        );
        assert!(published_generation(&index) > seeded);
        assert!(!index.has_uncommitted_changes());
    });
}

/// A retained scalar retry state keeps its refusal; the publish-first path
/// must not resolve it as a side effect.
///
/// These sentinels do not mean "there is work to publish" — they mean a
/// prior mutation ended ambiguously and must be reconciled by an EXPLICIT
/// commit. `commit_with_trigger` publishes pending segments and then clears
/// `ingest_retry_required`, so an implicit commit issued from an unrelated
/// replacement would publish a possibly-partial ingest and disarm the guard
/// that exists to prevent precisely that.
///
/// Each assertion fails differently: the generation catches the publish,
/// the searches catch the leaked and the clobbered document, and the
/// sentinel check catches the disarmed guard.
#[test]
fn lexical_trait_retained_scalar_retry_states_still_refuse_a_replacement() {
    run_with_cx(|cx| async move {
        for label in ["ingest_retry_required", "staged_flush"] {
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            LexicalWrite::index_documents(
                &index,
                &cx,
                &[IndexableDocument::new("seeded", "old alpha")],
            )
            .await
            .expect("seed published document");
            LexicalWrite::commit(&index, &cx)
                .await
                .expect("publish seed generation");
            let published = index
                .segment_stats()
                .expect("read seed segment stats")
                .published_generation;

            // Ambiguous pending work: staged, unpublished, and about to be
            // marked unreconciled.
            LexicalWrite::index_documents(
                &index,
                &cx,
                &[IndexableDocument::new("pending", "fresh beta")],
            )
            .await
            .expect("stage pending scalar work");

            match label {
                "ingest_retry_required" => index.writer_mut().ingest_retry_required = true,
                _ => index
                    .writer_mut()
                    .prepare_shard_flush(0)
                    .expect("stage a seal without installing it"),
            }

            let refused = LexicalWrite::index_documents(
                &index,
                &cx,
                &[IndexableDocument::new("seeded", "new gamma")],
            )
            .await;
            assert!(
                refused.is_err(),
                "{label} must keep refusing a replacement, got {refused:?}"
            );
            assert_eq!(
                index
                    .segment_stats()
                    .expect("read segment stats")
                    .published_generation,
                published,
                "{label} must not advance the published generation"
            );
            assert!(
                index
                    .search_doc_ids(&cx, "beta", 10)
                    .expect("pending visibility")
                    .is_empty(),
                "{label} must not publish the ambiguous pending work"
            );
            assert_eq!(
                index
                    .search_doc_ids(&cx, "alpha", 10)
                    .expect("seeded revision")
                    .len(),
                1,
                "{label} must leave the published revision in place"
            );
            assert!(
                index
                    .search_doc_ids(&cx, "gamma", 10)
                    .expect("replacement visibility")
                    .is_empty(),
                "{label} must not apply the refused replacement"
            );
            assert!(
                index.has_uncommitted_changes(),
                "{label} must leave the writer state staged"
            );
            let writer = index.writer_mut();
            match label {
                "ingest_retry_required" => assert!(
                    writer.ingest_retry_required,
                    "the commit-retry guard must stay armed"
                ),
                _ => assert!(
                    writer.staged_flush.is_some(),
                    "the staged seal must stay retained"
                ),
            }
        }
    });
}

/// The preflight refuses exactly the conditions a publication cannot fix,
/// and nothing else.
///
/// The negative half matters as much as the positive: an id that is live in
/// the published snapshot, or one already staged in `uncommitted_ids`, must
/// NOT be refused here, because publishing the pending work is precisely
/// what makes those legal. Refusing them would turn the publish-first path
/// back into the refusal it replaced.
#[test]
fn commit_invariant_batch_shape_refuses_what_publishing_cannot_fix() {
    run_with_cx(|cx| async move {
        let refused = |documents: &[IndexableDocument]| {
            QuillWriterState::validate_commit_invariant_batch_shape(
                &cx,
                IngestBatch::Default(documents),
            )
            .expect_err("must refuse")
            .to_string()
        };
        assert!(
            refused(&[IndexableDocument::new("", "body")]).contains("nonempty"),
            "an empty id is invalid against every published state"
        );
        assert!(
            refused(&[
                IndexableDocument::new("repeated", "one"),
                IndexableDocument::new("repeated", "two"),
            ])
            .contains("duplicate live document id"),
            "an in-batch repeat is invalid against every published state"
        );
        QuillWriterState::validate_commit_invariant_batch_shape(
            &cx,
            IngestBatch::Default(&[
                IndexableDocument::new("alpha", "one"),
                IndexableDocument::new("beta", "two"),
            ]),
        )
        .expect("distinct nonempty ids are not this check's business");
    });
}

/// A durable index can be created and reopened on the CASS descriptor.
///
/// This is the seam the cass migration needs: the schema-parameterized
/// constructors were previously in-memory only and bench-gated, so a
/// CASS-schema index could not survive a process boundary at all. The
/// reopen is the real assertion — it proves the descriptor was persisted
/// and re-validated, not merely accepted at create time.
#[test]
fn cass_schema_index_is_creatable_and_reopenable_on_disk() {
    run_with_cx(|cx| async move {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("cass-quill");
        drop(
            QuillIndex::create_with_schema(
                &cx,
                &root,
                crate::schema::CASS_SEMANTIC_SCHEMA,
                deterministic_config(),
            )
            .await
            .expect("create durable CASS-schema index"),
        );
        let reopened = QuillIndex::open_with_schema(
            &cx,
            &root,
            crate::schema::CASS_SEMANTIC_SCHEMA,
            deterministic_config(),
        )
        .await
        .expect("reopen durable CASS-schema index");
        assert_eq!(reopened.doc_count().expect("count"), 0);

        // The descriptor is verified, not inferred: opening the same
        // directory with the shipping schema must fail rather than
        // silently reinterpret CASS segments.
        assert!(
            QuillIndex::open_with_schema(&cx, &root, DEFAULT_SCHEMA, deterministic_config())
                .await
                .is_err(),
            "a mismatched descriptor must be a typed failure, not a silent misread"
        );
    });
}

/// The publish-first path is gated on an actual replacement, so a batch
/// that replaces nothing still accumulates rather than publishing early.
#[test]
fn lexical_trait_pure_append_still_accumulates_after_a_published_generation() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[IndexableDocument::new("seeded", "old alpha")],
        )
        .await
        .expect("seed published document");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("publish seed generation");
        let published = index
            .segment_stats()
            .expect("read seed segment stats")
            .published_generation;

        for id in ["fresh-one", "fresh-two"] {
            LexicalWrite::index_documents(&index, &cx, &[IndexableDocument::new(id, "beta")])
                .await
                .expect("stage a pure-append batch");
        }
        assert!(index.has_uncommitted_changes());
        assert_eq!(
            index
                .segment_stats()
                .expect("read segment stats")
                .published_generation,
            published,
            "a batch that replaces nothing must not publish a generation"
        );
    });
}

#[test]
fn lexical_trait_disjoint_batches_accumulate_before_commit() {
    run_with_cx(|cx| async move {
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[
                IndexableDocument::new("first", "alpha"),
                IndexableDocument::new("second", "beta"),
            ],
        )
        .await
        .expect("stage first disjoint batch");
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[
                IndexableDocument::new("third", "gamma"),
                IndexableDocument::new("fourth", "delta"),
            ],
        )
        .await
        .expect("stage second disjoint batch");

        LexicalWrite::commit(&index, &cx)
            .await
            .expect("publish both disjoint batches");
        assert_eq!(
            index
                .doc_count()
                .expect("disjoint batch count is authoritative"),
            4
        );
    });
}
