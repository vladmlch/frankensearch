use super::tests_support::vhit;
use super::*;

/// `cmp_rank` must be a total order: score descending, ties broken on `doc_id`.
#[test]
fn cmp_rank_breaks_score_ties_on_doc_id() {
    let mut hits = [
        vhit(0, 0.5, "b"),
        vhit(1, 0.9, "z"),
        vhit(2, 0.5, "a"),
        vhit(3, 0.5, "c"),
    ];
    hits.sort_unstable_by(VectorHit::cmp_rank);
    let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(ids, ["z", "a", "b", "c"]);
}

/// NaN sorts LAST, inheriting `cmp_by_score`. A bare descending `total_cmp` would do the
/// opposite — IEEE 754 `totalOrder` ranks `+NaN` above `+inf`, putting a corrupted score at
/// rank 0. Phase-1 score corrections re-sort with this comparator, so the distinction is load
/// bearing.
#[test]
fn cmp_rank_sorts_nan_last_unlike_bare_total_cmp() {
    let mut hits = [
        vhit(0, f32::NAN, "nan"),
        vhit(1, 0.1, "lo"),
        vhit(2, 0.9, "hi"),
    ];
    hits.sort_unstable_by(VectorHit::cmp_rank);
    let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(ids, ["hi", "lo", "nan"], "NaN must sort last");

    // The trap this guards against.
    let mut bare = [vhit(0, f32::NAN, "nan"), vhit(1, 0.9, "hi")];
    bare.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    assert_eq!(
        bare[0].doc_id.as_str(),
        "nan",
        "bare descending total_cmp puts +NaN at rank 0 — the reason cmp_rank exists"
    );
}

#[test]
fn indexable_document_builder() {
    let doc = IndexableDocument::new("doc-1", "Hello world")
        .with_title("Greeting")
        .with_metadata("source", "test");

    assert_eq!(doc.id, "doc-1");
    assert_eq!(doc.content, "Hello world");
    assert_eq!(doc.title.as_deref(), Some("Greeting"));
    assert_eq!(doc.metadata.get("source").map(String::as_str), Some("test"));
}

#[test]
fn indexable_document_minimal() {
    let doc = IndexableDocument::new("id", "text");
    assert!(doc.title.is_none());
    assert!(doc.metadata.is_empty());
}

#[test]
fn vector_hit_nan_safe_ordering() {
    let hit_a = VectorHit {
        index: 0,
        score: 0.9,
        doc_id: "a".into(),
    };
    let hit_nan = VectorHit {
        index: 1,
        score: f32::NAN,
        doc_id: "b".into(),
    };
    // NaN should sort below real values (hit_a should come first).
    assert_eq!(
        hit_a.cmp_by_score(&hit_nan),
        std::cmp::Ordering::Less // a comes first (better score)
    );
}

#[test]
fn fused_hit_tie_breaking() {
    let hit_both = FusedHit {
        doc_id: "a".into(),
        rrf_score: 0.02,
        lexical_rank: Some(1),
        semantic_rank: Some(3),
        hash_rank: None,
        semantic_index: Some(3),
        lexical_score: Some(5.0),
        semantic_score: Some(0.8),
        hash_score: None,
        in_both_sources: true,
    };
    let hit_semantic_only = FusedHit {
        doc_id: "b".into(),
        rrf_score: 0.02, // Same RRF score
        lexical_rank: None,
        semantic_rank: Some(2),
        hash_rank: None,
        semantic_index: Some(2),
        lexical_score: None,
        semantic_score: Some(0.9),
        hash_score: None,
        in_both_sources: false,
    };
    // Same RRF -> in_both_sources wins
    assert_eq!(
        hit_both.cmp_for_ranking(&hit_semantic_only),
        std::cmp::Ordering::Less // hit_both ranks first
    );
}

#[test]
fn fused_hit_rrf_score_dominates() {
    let high = FusedHit {
        doc_id: "a".into(),
        rrf_score: 0.03,
        lexical_rank: None,
        semantic_rank: Some(1),
        hash_rank: None,
        semantic_index: Some(1),
        lexical_score: None,
        semantic_score: Some(0.9),
        hash_score: None,
        in_both_sources: false,
    };
    let low = FusedHit {
        doc_id: "b".into(),
        rrf_score: 0.01,
        lexical_rank: Some(1),
        semantic_rank: Some(1),
        hash_rank: None,
        semantic_index: Some(1),
        lexical_score: Some(10.0),
        semantic_score: Some(0.99),
        hash_score: None,
        in_both_sources: true,
    };
    // Higher RRF always wins regardless of other fields.
    assert_eq!(
        high.cmp_for_ranking(&low),
        std::cmp::Ordering::Less // high ranks first
    );
}

#[test]
fn fused_hit_deterministic_doc_id_tiebreak() {
    let a = FusedHit {
        doc_id: "alpha".into(),
        rrf_score: 0.02,
        lexical_rank: None,
        semantic_rank: None,
        hash_rank: None,
        semantic_index: None,
        lexical_score: None,
        semantic_score: None,
        hash_score: None,
        in_both_sources: false,
    };
    let b = FusedHit {
        doc_id: "beta".into(),
        rrf_score: 0.02,
        lexical_rank: None,
        semantic_rank: None,
        hash_rank: None,
        semantic_index: None,
        lexical_score: None,
        semantic_score: None,
        hash_score: None,
        in_both_sources: false,
    };
    // All else equal -> lexicographic doc_id ascending.
    assert_eq!(a.cmp_for_ranking(&b), std::cmp::Ordering::Less);
}

#[test]
fn scored_result_serde_roundtrip() {
    let result = ScoredResult {
        doc_id: "doc-42".into(),
        score: 0.85,
        source: ScoreSource::Hybrid,
        index: None,
        fast_score: Some(0.7),
        quality_score: Some(0.9),
        lexical_score: Some(12.5),
        rerank_score: None,
        explanation: None,
        metadata: Some(std::sync::Arc::new(
            serde_json::json!({"tags": ["rust", "search"]}),
        )),
    };
    let json = serde_json::to_string(&result).expect("serialize");
    let roundtripped: ScoredResult = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(roundtripped.doc_id, "doc-42");
    assert!((roundtripped.score - 0.85).abs() < f32::EPSILON);
    assert_eq!(roundtripped.source, ScoreSource::Hybrid);
    assert!(roundtripped.metadata.is_some());
}

#[test]
fn rank_changes_total() {
    let changes = RankChanges {
        promoted: 3,
        demoted: 2,
        stable: 5,
    };
    assert_eq!(changes.total(), 10);
}

#[test]
fn search_phase_initial_construction() {
    let phase = SearchPhase::Initial {
        results: vec![],
        latency: Duration::from_millis(12),
        metrics: PhaseMetrics {
            embedder_id: "potion-128M".into(),
            vectors_searched: 1000,
            lexical_candidates: 50,
            fused_count: 10,
            skip_reason: None,
            hash_control_candidates: 0,
        },
    };
    if let SearchPhase::Initial { latency, .. } = phase {
        assert_eq!(latency, Duration::from_millis(12));
    }
}

#[test]
fn search_phase_refinement_failed_carries_results() {
    let initial = vec![ScoredResult {
        doc_id: "doc-1".into(),
        score: 0.5,
        source: ScoreSource::Hybrid,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: None,
        rerank_score: None,
        explanation: None,
        metadata: None,
    }];
    let phase = SearchPhase::RefinementFailed {
        initial_results: initial,
        error: SearchError::SearchTimeout {
            elapsed_ms: 500,
            budget_ms: 300,
        },
        latency: Duration::from_millis(500),
    };
    if let SearchPhase::RefinementFailed {
        initial_results, ..
    } = phase
    {
        assert_eq!(initial_results.len(), 1);
        assert_eq!(initial_results[0].doc_id, "doc-1");
    }
}

#[test]
fn indexable_document_serde_roundtrip() {
    let doc = IndexableDocument::new("id-1", "Hello world")
        .with_title("Test")
        .with_metadata("lang", "en");
    let json = serde_json::to_string(&doc).expect("serialize");
    let rt: IndexableDocument = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt.id, "id-1");
    assert_eq!(rt.title.as_deref(), Some("Test"));
    assert_eq!(rt.metadata.get("lang").map(String::as_str), Some("en"));
}

#[test]
fn search_metrics_serde_roundtrip() {
    let metrics = SearchMetrics {
        mode: SearchMode::Hybrid,
        query_class: Some(QueryClass::NaturalLanguage),
        total_latency_ms: 12.5,
        phase1_latency_ms: Some(5.4),
        phase2_latency_ms: Some(7.1),
        result_count: 10,
        lexical_candidates: 60,
        semantic_candidates: 45,
        hash_control_candidates: 0,
        refined: true,
    };

    let json = serde_json::to_string(&metrics).expect("serialize");
    let decoded: SearchMetrics = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded.mode, SearchMode::Hybrid);
    assert_eq!(decoded.query_class, Some(QueryClass::NaturalLanguage));
    assert!((decoded.total_latency_ms - 12.5).abs() < f64::EPSILON);
    assert_eq!(decoded.result_count, 10);
    assert_eq!(decoded.semantic_candidates, 45);
    assert_eq!(decoded.hash_control_candidates, 0);
    assert!(decoded.refined);
    assert!(
        !json.contains("hash_control_candidates"),
        "zero hash-control count should omit the key: {json}"
    );

    let legacy = serde_json::from_str::<SearchMetrics>(
        r#"{"mode":"Hybrid","query_class":null,"total_latency_ms":1.0,"phase1_latency_ms":null,"phase2_latency_ms":null,"result_count":0,"lexical_candidates":0,"semantic_candidates":3,"refined":false}"#,
    )
    .expect("legacy SearchMetrics without hash_control_candidates");
    assert_eq!(legacy.semantic_candidates, 3);
    assert_eq!(legacy.hash_control_candidates, 0);
}

#[test]
fn embedding_metrics_serde_roundtrip() {
    let metrics = EmbeddingMetrics {
        embedder_id: "potion-multilingual-128M".into(),
        batch_size: 32,
        duration_ms: 1.9,
        dimension: 256,
        is_semantic: true,
    };

    let json = serde_json::to_string(&metrics).expect("serialize");
    let decoded: EmbeddingMetrics = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded.embedder_id, "potion-multilingual-128M");
    assert_eq!(decoded.batch_size, 32);
    assert_eq!(decoded.dimension, 256);
    assert!(decoded.is_semantic);
}

#[test]
fn index_metrics_serde_roundtrip() {
    let metrics = IndexMetrics {
        doc_count: 1_000,
        index_size_bytes: 12_345_678,
        updated_docs: 11,
        staleness_detected: false,
    };

    let json = serde_json::to_string(&metrics).expect("serialize");
    let decoded: IndexMetrics = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded.doc_count, 1_000);
    assert_eq!(decoded.index_size_bytes, 12_345_678);
    assert_eq!(decoded.updated_docs, 11);
    assert!(!decoded.staleness_detected);
}

// ─── bd-ta55 tests begin ───

#[test]
fn indexable_document_multiple_metadata() {
    let doc = IndexableDocument::new("d1", "content")
        .with_metadata("a", "1")
        .with_metadata("b", "2")
        .with_metadata("c", "3");
    assert_eq!(doc.metadata.len(), 3);
    assert_eq!(doc.metadata["a"], "1");
    assert_eq!(doc.metadata["b"], "2");
    assert_eq!(doc.metadata["c"], "3");
}

#[test]
fn indexable_document_metadata_overwrite() {
    let doc = IndexableDocument::new("d1", "content")
        .with_metadata("key", "old")
        .with_metadata("key", "new");
    assert_eq!(doc.metadata.len(), 1);
    assert_eq!(doc.metadata["key"], "new");
}

#[test]
fn indexable_document_clone_debug() {
    let doc = IndexableDocument::new("d1", "text").with_title("T");
    let cloned = doc.clone();
    assert_eq!(cloned.id, "d1");
    assert_eq!(cloned.title.as_deref(), Some("T"));
    let dbg = format!("{doc:?}");
    assert!(dbg.contains("IndexableDocument"));
    assert!(dbg.contains("d1"));
}

#[test]
fn vector_hit_partial_eq() {
    let a = VectorHit {
        index: 0,
        score: 0.5,
        doc_id: "a".into(),
    };
    let b = VectorHit {
        index: 0,
        score: 0.5,
        doc_id: "a".into(),
    };
    assert_eq!(a, b);

    let c = VectorHit {
        index: 1,
        score: 0.5,
        doc_id: "a".into(),
    };
    assert_ne!(a, c);
}

#[test]
fn vector_hit_serde_roundtrip() {
    let hit = VectorHit {
        index: 42,
        score: 0.95,
        doc_id: "doc-x".into(),
    };
    let json = serde_json::to_string(&hit).unwrap();
    let decoded: VectorHit = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, hit);
}

#[test]
fn vector_hit_both_nan() {
    let a = VectorHit {
        index: 0,
        score: f32::NAN,
        doc_id: "a".into(),
    };
    let b = VectorHit {
        index: 1,
        score: f32::NAN,
        doc_id: "b".into(),
    };
    // Both NaN -> treated as equal (both map to NEG_INFINITY)
    assert_eq!(a.cmp_by_score(&b), std::cmp::Ordering::Equal);
}

#[test]
fn vector_hit_equal_scores() {
    let a = VectorHit {
        index: 0,
        score: 0.75,
        doc_id: "a".into(),
    };
    let b = VectorHit {
        index: 1,
        score: 0.75,
        doc_id: "b".into(),
    };
    assert_eq!(a.cmp_by_score(&b), std::cmp::Ordering::Equal);
}

#[test]
fn vector_hit_descending_order() {
    let high = VectorHit {
        index: 0,
        score: 0.9,
        doc_id: "h".into(),
    };
    let low = VectorHit {
        index: 1,
        score: 0.1,
        doc_id: "l".into(),
    };
    // Descending: high comes first (Less means "before")
    assert_eq!(high.cmp_by_score(&low), std::cmp::Ordering::Less);
    assert_eq!(low.cmp_by_score(&high), std::cmp::Ordering::Greater);
}

#[test]
fn fused_hit_serde_roundtrip() {
    let hit = FusedHit {
        doc_id: "fused-1".into(),
        rrf_score: 0.025,
        lexical_rank: Some(3),
        semantic_rank: Some(7),
        hash_rank: None,
        semantic_index: Some(7),
        lexical_score: Some(8.5),
        semantic_score: Some(0.72),
        hash_score: None,
        in_both_sources: true,
    };
    let json = serde_json::to_string(&hit).unwrap();
    let decoded: FusedHit = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.doc_id, "fused-1");
    assert!((decoded.rrf_score - 0.025).abs() < f64::EPSILON);
    assert!(decoded.in_both_sources);
    assert_eq!(decoded.lexical_rank, Some(3));
    assert_eq!(decoded.hash_rank, None);
    assert!(
        !json.contains("hash_rank") && !json.contains("hash_score"),
        "zero hash fields should omit the keys: {json}"
    );

    let mut hash_hit = hit;
    hash_hit.remap_hash_control_ranks();
    assert_eq!(hash_hit.hash_rank, Some(7));
    assert_eq!(hash_hit.semantic_rank, None);
    assert_eq!(hash_hit.hash_score, Some(0.72));
    assert_eq!(hash_hit.semantic_score, None);
    let hash_json = serde_json::to_string(&hash_hit).unwrap();
    let hash_decoded: FusedHit = serde_json::from_str(&hash_json).unwrap();
    assert_eq!(hash_decoded.hash_rank, Some(7));
    assert_eq!(hash_decoded.semantic_rank, None);
    assert_eq!(hash_decoded.hash_score, Some(0.72));
    assert_eq!(hash_decoded.semantic_score, None);
    assert!(
        hash_json.contains("hash_rank") && hash_json.contains("hash_score"),
        "remapped hash hit must serialize hash fields: {hash_json}"
    );
    assert!(
        !hash_json.contains("semantic_rank") && !hash_json.contains("semantic_score"),
        "remapped hash hit must not publish semantic keys: {hash_json}"
    );
}

#[test]
fn fused_hit_lexical_score_tiebreak() {
    // Same RRF, same in_both_sources -> lexical_score descending
    let high_lex = FusedHit {
        doc_id: "z".into(), // worse doc_id
        rrf_score: 0.02,
        lexical_rank: Some(1),
        semantic_rank: None,
        hash_rank: None,
        semantic_index: None,
        lexical_score: Some(15.0),
        semantic_score: None,
        hash_score: None,
        in_both_sources: false,
    };
    let low_lex = FusedHit {
        doc_id: "a".into(), // better doc_id
        rrf_score: 0.02,
        lexical_rank: Some(5),
        semantic_rank: None,
        hash_rank: None,
        semantic_index: None,
        lexical_score: Some(3.0),
        semantic_score: None,
        hash_score: None,
        in_both_sources: false,
    };
    // Higher lexical_score wins (level 3)
    assert_eq!(high_lex.cmp_for_ranking(&low_lex), std::cmp::Ordering::Less);
}

#[test]
fn fused_hit_clone_debug() {
    let hit = FusedHit {
        doc_id: "test".into(),
        rrf_score: 0.01,
        lexical_rank: None,
        semantic_rank: Some(5),
        hash_rank: None,
        semantic_index: Some(5),
        lexical_score: None,
        semantic_score: Some(0.6),
        hash_score: None,
        in_both_sources: false,
    };
    let cloned = hit.clone();
    assert_eq!(cloned.doc_id, "test");
    let dbg = format!("{hit:?}");
    assert!(dbg.contains("FusedHit"));
}

#[test]
fn score_source_all_variants_serde() {
    let variants = [
        ScoreSource::Lexical,
        ScoreSource::SemanticFast,
        ScoreSource::SemanticQuality,
        ScoreSource::Hybrid,
        ScoreSource::Reranked,
        ScoreSource::HashControl,
    ];
    for variant in &variants {
        let json = serde_json::to_string(variant).unwrap();
        let decoded: ScoreSource = serde_json::from_str(&json).unwrap();
        assert_eq!(&decoded, variant);
    }
}

#[test]
fn is_hash_generation_id_matches_control_families() {
    assert!(super::is_hash_generation_id("hash"));
    assert!(super::is_hash_generation_id("HASH"));
    assert!(super::is_hash_generation_id("hash-fnv1a-256"));
    assert!(super::is_hash_generation_id("fnv1a-256"));
    assert!(super::is_hash_generation_id("jl-128"));
    assert!(super::is_hash_generation_id("hash/fnv1a"));
    assert!(super::is_hash_generation_id("HASH/FNV1A"));
    assert!(!super::is_hash_generation_id("minilm-l6-v2"));
    assert!(!super::is_hash_generation_id("potion-128m"));
    assert!(!super::is_hash_generation_id("stub-384"));
    assert!(!super::is_hash_generation_id("hashed-minilm"));
}

#[test]
fn score_source_clone_copy_eq() {
    let a = ScoreSource::Hybrid;
    let b = a; // Copy
    let c = a; // Copy (ScoreSource is Copy)
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_ne!(ScoreSource::Lexical, ScoreSource::Hybrid);
}

#[test]
fn scored_result_all_none_optionals() {
    let result = ScoredResult {
        doc_id: "min".into(),
        score: 0.1,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: None,
        rerank_score: None,
        explanation: None,
        metadata: None,
    };
    let json = serde_json::to_string(&result).unwrap();
    let decoded: ScoredResult = serde_json::from_str(&json).unwrap();
    assert!(decoded.fast_score.is_none());
    assert!(decoded.quality_score.is_none());
    assert!(decoded.rerank_score.is_none());
    assert!(decoded.metadata.is_none());
}

#[test]
fn scored_result_all_some_optionals() {
    let result = ScoredResult {
        doc_id: "max".into(),
        score: 0.99,
        source: ScoreSource::Reranked,
        index: Some(3),
        fast_score: Some(0.7),
        quality_score: Some(0.9),
        lexical_score: Some(15.0),
        rerank_score: Some(0.95),
        explanation: None,
        metadata: Some(std::sync::Arc::new(serde_json::json!({"key": "value"}))),
    };
    let json = serde_json::to_string(&result).unwrap();
    let decoded: ScoredResult = serde_json::from_str(&json).unwrap();
    assert!(decoded.fast_score.is_some());
    assert!(decoded.quality_score.is_some());
    assert!(decoded.lexical_score.is_some());
    assert!(decoded.rerank_score.is_some());
    assert!(decoded.metadata.is_some());
}

#[test]
fn scored_result_clone_debug() {
    let result = ScoredResult {
        doc_id: "d".into(),
        score: 0.5,
        source: ScoreSource::SemanticFast,
        index: Some(0),
        fast_score: Some(0.5),
        quality_score: None,
        lexical_score: None,
        rerank_score: None,
        explanation: None,
        metadata: None,
    };
    let cloned = result.clone();
    assert_eq!(cloned.doc_id, "d");
    let dbg = format!("{result:?}");
    assert!(dbg.contains("ScoredResult"));
    assert!(dbg.contains("SemanticFast"));
}

#[test]
fn search_mode_all_variants_serde() {
    let variants = [
        SearchMode::Lexical,
        SearchMode::Semantic,
        SearchMode::Hybrid,
        SearchMode::TwoTier,
    ];
    for variant in &variants {
        let json = serde_json::to_string(variant).unwrap();
        let decoded: SearchMode = serde_json::from_str(&json).unwrap();
        assert_eq!(&decoded, variant);
    }
}

#[test]
fn phase_metrics_serde_clone_debug() {
    let metrics = PhaseMetrics {
        embedder_id: "test-embed".into(),
        vectors_searched: 500,
        lexical_candidates: 30,
        fused_count: 10,
        skip_reason: None,
        hash_control_candidates: 0,
    };
    let json = serde_json::to_string(&metrics).unwrap();
    let decoded: PhaseMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.embedder_id, "test-embed");
    assert_eq!(decoded.vectors_searched, 500);
    assert_eq!(decoded.hash_control_candidates, 0);
    assert!(
        !json.contains("hash_control_candidates"),
        "zero hash-control count should omit the key: {json}"
    );

    let cloned = metrics.clone();
    assert_eq!(cloned.fused_count, 10);

    let dbg = format!("{metrics:?}");
    assert!(dbg.contains("PhaseMetrics"));

    let hash_metrics = PhaseMetrics {
        embedder_id: "fnv1a-256".into(),
        vectors_searched: 500,
        lexical_candidates: 0,
        fused_count: 4,
        skip_reason: Some("non_semantic_fast_embedder_vector_control".into()),
        hash_control_candidates: 4,
    };
    let hash_json = serde_json::to_string(&hash_metrics).unwrap();
    let hash_decoded: PhaseMetrics = serde_json::from_str(&hash_json).unwrap();
    assert_eq!(hash_decoded.hash_control_candidates, 4);
    assert!(
        hash_json.contains("hash_control_candidates"),
        "non-zero hash-control count must be serialized: {hash_json}"
    );

    let legacy = serde_json::from_str::<PhaseMetrics>(
        r#"{"embedder_id":"potion-128M","vectors_searched":10,"lexical_candidates":3,"fused_count":3}"#,
    )
    .expect("legacy PhaseMetrics without hash_control_candidates");
    assert_eq!(legacy.hash_control_candidates, 0);
}

#[test]
fn rank_changes_default_and_zero_total() {
    let changes = RankChanges::default();
    assert_eq!(changes.promoted, 0);
    assert_eq!(changes.demoted, 0);
    assert_eq!(changes.stable, 0);
    assert_eq!(changes.total(), 0);
}

#[test]
fn rank_changes_serde_roundtrip() {
    let changes = RankChanges {
        promoted: 5,
        demoted: 3,
        stable: 12,
    };
    let json = serde_json::to_string(&changes).unwrap();
    let decoded: RankChanges = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.promoted, 5);
    assert_eq!(decoded.demoted, 3);
    assert_eq!(decoded.stable, 12);
    assert_eq!(decoded.total(), 20);
}

#[test]
fn rank_changes_total_saturates_on_overflow() {
    let changes = RankChanges {
        promoted: usize::MAX,
        demoted: 1,
        stable: 1,
    };
    assert_eq!(changes.total(), usize::MAX);
}

#[test]
fn search_phase_refined_construction() {
    let phase = SearchPhase::Refined {
        results: vec![ScoredResult {
            doc_id: "r1".into(),
            score: 0.9,
            source: ScoreSource::Hybrid,
            index: None,
            fast_score: Some(0.7),
            quality_score: Some(0.9),
            lexical_score: None,
            rerank_score: None,
            explanation: None,
            metadata: None,
        }],
        latency: Duration::from_millis(120),
        metrics: PhaseMetrics {
            embedder_id: "minilm".into(),
            vectors_searched: 2000,
            lexical_candidates: 100,
            fused_count: 20,
            skip_reason: None,
            hash_control_candidates: 0,
        },
        rank_changes: RankChanges {
            promoted: 4,
            demoted: 2,
            stable: 14,
        },
    };
    assert!(matches!(phase, SearchPhase::Refined { .. }));
    let SearchPhase::Refined {
        results,
        latency,
        rank_changes,
        ..
    } = phase
    else {
        return;
    };
    assert_eq!(results.len(), 1);
    assert_eq!(latency, Duration::from_millis(120));
    assert_eq!(rank_changes.total(), 20);
}

#[test]
fn search_phase_reranked_construction() {
    let phase = SearchPhase::Reranked {
        results: vec![ScoredResult {
            doc_id: "r1".into(),
            score: 0.95,
            source: ScoreSource::Reranked,
            index: None,
            fast_score: Some(0.7),
            quality_score: Some(0.9),
            lexical_score: None,
            rerank_score: Some(0.95),
            explanation: None,
            metadata: None,
        }],
        latency: Duration::from_millis(450),
        metrics: PhaseMetrics {
            embedder_id: "flashrank".into(),
            vectors_searched: 2000,
            lexical_candidates: 100,
            fused_count: 20,
            skip_reason: None,
            hash_control_candidates: 0,
        },
    };
    assert!(matches!(phase, SearchPhase::Reranked { .. }));
    let SearchPhase::Reranked {
        results, latency, ..
    } = phase
    else {
        return;
    };
    assert_eq!(results.len(), 1);
    assert_eq!(latency, Duration::from_millis(450));
}

#[test]
fn search_phase_debug() {
    let phase = SearchPhase::Initial {
        results: vec![],
        latency: Duration::from_millis(5),
        metrics: PhaseMetrics {
            embedder_id: "e".into(),
            vectors_searched: 0,
            lexical_candidates: 0,
            fused_count: 0,
            skip_reason: None,
            hash_control_candidates: 0,
        },
    };
    let dbg = format!("{phase:?}");
    assert!(dbg.contains("Initial"));
}

#[test]
fn vector_hit_nan_sorts_below_real() {
    let real = VectorHit {
        index: 0,
        score: -100.0, // very low but real
        doc_id: "real".into(),
    };
    let nan = VectorHit {
        index: 1,
        score: f32::NAN,
        doc_id: "nan".into(),
    };
    // NaN maps to NEG_INFINITY, so even -100.0 beats it
    assert_eq!(real.cmp_by_score(&nan), std::cmp::Ordering::Less);
    assert_eq!(nan.cmp_by_score(&real), std::cmp::Ordering::Greater);
}

#[test]
fn vector_hit_negative_scores_descending() {
    let a = VectorHit {
        index: 0,
        score: -0.1,
        doc_id: "a".into(),
    };
    let b = VectorHit {
        index: 1,
        score: -0.9,
        doc_id: "b".into(),
    };
    // -0.1 > -0.9, so a comes first (Less in descending order)
    assert_eq!(a.cmp_by_score(&b), std::cmp::Ordering::Less);
}

#[test]
fn fused_hit_in_both_sources_tiebreak() {
    let both = FusedHit {
        doc_id: "z".into(), // worse doc_id
        rrf_score: 0.02,
        lexical_rank: Some(3),
        semantic_rank: Some(5),
        hash_rank: None,
        semantic_index: Some(5),
        lexical_score: None,
        semantic_score: None,
        hash_score: None,
        in_both_sources: true,
    };
    let single = FusedHit {
        doc_id: "a".into(), // better doc_id
        rrf_score: 0.02,
        lexical_rank: Some(1),
        semantic_rank: None,
        hash_rank: None,
        semantic_index: None,
        lexical_score: None,
        semantic_score: None,
        hash_score: None,
        in_both_sources: false,
    };
    // Same RRF -> in_both_sources=true wins (level 2)
    assert_eq!(both.cmp_for_ranking(&single), std::cmp::Ordering::Less);
}

// ─── bd-ta55 tests end ───
