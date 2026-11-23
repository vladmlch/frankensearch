use super::*;

/// Helper: run async test code with a `Cx`.
pub fn run_with_cx<F, Fut>(f: F)
where
    F: FnOnce(Cx) -> Fut,
    Fut: Future<Output = ()>,
{
    asupersync::test_utils::run_test_with_cx(f);
}

pub fn sample_docs() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("doc-1", "Rust is a systems programming language")
            .with_title("Rust Overview")
            .with_metadata("lang", "en"),
        IndexableDocument::new(
            "doc-2",
            "Python is great for data science and machine learning",
        )
        .with_title("Python for ML"),
        IndexableDocument::new("doc-3", "The Rust borrow checker prevents data races")
            .with_title("Rust Safety"),
        IndexableDocument::new(
            "doc-4",
            "Distributed consensus algorithms like Raft and Paxos",
        )
        .with_title("Consensus Algorithms"),
        IndexableDocument::new(
            "doc-5",
            "Machine learning models for natural language processing",
        )
        .with_title("NLP Models"),
    ]
}

pub fn shared_full120_docs() -> Vec<IndexableDocument> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../../tests/fixtures/corpus.json"))
            .expect("parse committed Full120 corpus");
    let documents = fixture["documents"]
        .as_array()
        .expect("fixture documents array");
    assert_eq!(documents.len(), 120, "Full120 fixture count drifted");
    documents
        .iter()
        .enumerate()
        .map(|(ordinal, document)| {
            let mut metadata = document["metadata"]
                .as_object()
                .expect("fixture metadata")
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        value
                            .as_str()
                            .map_or_else(|| value.to_string(), str::to_owned),
                    )
                })
                .collect::<std::collections::HashMap<_, _>>();
            metadata.insert(
                "created_at".to_owned(),
                document["created_at"]
                    .as_str()
                    .expect("fixture created_at")
                    .to_owned(),
            );
            metadata.insert(
                "doc_type".to_owned(),
                document["doc_type"]
                    .as_str()
                    .expect("fixture doc_type")
                    .to_owned(),
            );
            metadata.insert("created_at_ms".to_owned(), ordinal.to_string());
            IndexableDocument {
                id: document["doc_id"]
                    .as_str()
                    .expect("fixture doc_id")
                    .to_owned(),
                content: document["content"]
                    .as_str()
                    .expect("fixture content")
                    .to_owned(),
                title: Some(
                    document["title"]
                        .as_str()
                        .expect("fixture title")
                        .to_owned(),
                ),
                metadata,
            }
        })
        .collect()
}

fn optional_score_bits(score: Option<f32>) -> Option<u32> {
    score.map(f32::to_bits)
}

fn assert_scored_result_contract(actual: &ScoredResult, expected: &ScoredResult, context: &str) {
    assert_eq!(actual.doc_id, expected.doc_id, "doc_id: {context}");
    assert_eq!(
        actual.score.to_bits(),
        expected.score.to_bits(),
        "score: {context}"
    );
    assert_eq!(actual.source, expected.source, "source: {context}");
    assert_eq!(actual.index, expected.index, "index: {context}");
    assert_eq!(
        optional_score_bits(actual.fast_score),
        optional_score_bits(expected.fast_score),
        "fast score: {context}"
    );
    assert_eq!(
        optional_score_bits(actual.quality_score),
        optional_score_bits(expected.quality_score),
        "quality score: {context}"
    );
    assert_eq!(
        optional_score_bits(actual.lexical_score),
        optional_score_bits(expected.lexical_score),
        "lexical score: {context}"
    );
    assert_eq!(
        optional_score_bits(actual.rerank_score),
        optional_score_bits(expected.rerank_score),
        "rerank score: {context}"
    );
    assert_eq!(
        serde_json::to_value(&actual.explanation).expect("serialize actual explanation"),
        serde_json::to_value(&expected.explanation).expect("serialize expected explanation"),
        "explanation: {context}"
    );
    assert_eq!(
        actual.metadata.as_deref(),
        expected.metadata.as_deref(),
        "metadata: {context}"
    );
}

pub async fn assert_live_lexical_contract(
    idx: &TantivyIndex,
    cx: &Cx,
    query: &str,
    limit: usize,
    expected_total_count: usize,
) {
    let context = format!("query={query:?}, limit={limit}");
    let full = idx.search(cx, query, limit).await.expect("full search");
    let id_hits = idx
        .search_doc_ids(cx, query, limit)
        .expect("ord-table ID search");
    let docstore_hits = idx
        .search_doc_ids_via_docstore(cx, query, limit)
        .expect("docstore ID search");
    let candidate_batch = idx
        .search_candidates(cx, query, limit)
        .await
        .expect("fusion candidate search");
    let (mut candidates, candidate_pin) = candidate_batch.into_parts();

    assert_eq!(
        id_hits, docstore_hits,
        "ordinal table must independently resolve the same IDs as stored documents: {context}"
    );
    assert_eq!(full.len(), id_hits.len(), "full/id length: {context}");
    assert_eq!(
        full.len(),
        candidates.len(),
        "full/candidate length: {context}"
    );
    assert_eq!(
        full.len(),
        limit.min(expected_total_count),
        "result count: {context}"
    );

    for (rank, ((expected, id_hit), candidate)) in
        full.iter().zip(&id_hits).zip(&candidates).enumerate()
    {
        let row_context = format!("{context}, rank={rank}");
        assert_eq!(id_hit.rank, rank, "ID rank: {row_context}");
        assert_eq!(id_hit.doc_id, expected.doc_id, "ID doc_id: {row_context}");
        assert_eq!(
            id_hit.bm25_score.to_bits(),
            expected.score.to_bits(),
            "ID score: {row_context}"
        );
        assert_eq!(
            candidate.doc_id, expected.doc_id,
            "candidate doc_id: {row_context}"
        );
        assert_eq!(
            candidate.score.to_bits(),
            expected.score.to_bits(),
            "candidate score: {row_context}"
        );
        assert_eq!(
            optional_score_bits(candidate.lexical_score),
            optional_score_bits(expected.lexical_score),
            "candidate lexical score: {row_context}"
        );
        assert_eq!(candidate.source, expected.source, "source: {row_context}");
        assert_eq!(candidate.index, expected.index, "index: {row_context}");
        assert_eq!(
            optional_score_bits(candidate.fast_score),
            optional_score_bits(expected.fast_score),
            "fast score: {row_context}"
        );
        assert_eq!(
            optional_score_bits(candidate.quality_score),
            optional_score_bits(expected.quality_score),
            "quality score: {row_context}"
        );
        assert_eq!(
            optional_score_bits(candidate.rerank_score),
            optional_score_bits(expected.rerank_score),
            "rerank score: {row_context}"
        );
        assert_eq!(
            serde_json::to_value(&candidate.explanation).expect("serialize candidate explanation"),
            serde_json::to_value(&expected.explanation).expect("serialize expected explanation"),
            "explanation: {row_context}"
        );
        assert!(
            candidate.metadata.is_none(),
            "candidate metadata must be deferred: {row_context}"
        );
    }

    idx.hydrate_candidates(cx, candidate_pin.as_ref(), &mut candidates)
        .await
        .expect("hydrate fusion metadata");
    for (rank, (candidate, expected)) in candidates.iter().zip(&full).enumerate() {
        assert_scored_result_contract(
            candidate,
            expected,
            &format!("{context}, hydrated rank={rank}"),
        );
    }

    let parsed = idx.parse_query_lenient(query);
    let counted = execute_query_with_offset(&idx.reader.searcher(), &*parsed, limit, 0)
        .expect("exact counted search");
    assert_eq!(
        counted.total_count, expected_total_count,
        "Count must remain exact and independent of k: {context}"
    );
}
