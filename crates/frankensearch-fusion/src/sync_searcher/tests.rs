use super::*;
use frankensearch_core::ScoreSource;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Bind one synthetic vector to an explicitly synthetic identity.
///
/// `explicit_test_model` is the sanctioned constructor for exactly this:
/// its own docs say it is "intentionally named and tagged as synthetic so
/// it cannot be mistaken for verified semantic availability". These
/// fixtures build vectors by hand, so a synthetic identity is the true
/// one — the alternative would be fabricating a model name, which is what
/// this whole surface exists to prevent.
fn bound(model: &str, vector: Vec<f32>) -> BoundQueryEmbedding {
    let dimension = u32::try_from(vector.len()).expect("fixture dimension fits u32");
    BoundQueryEmbedding::new(
        vector,
        EmbeddingIdentityBundleV1::explicit_test_model(model, dimension),
    )
    .expect("synthetic fixture query binds")
}

/// The fixture indexes below build both tiers from vectors of the same
/// synthetic space, so one bundle legitimately describes both arms. A
/// production pair (potion fast + `MiniLM` quality) would bind two DIFFERENT
/// bundles here, and `admit` would join each against its own tier.
fn tiered(vector: Vec<f32>) -> TieredQueryEmbeddings {
    TieredQueryEmbeddings::progressive(
        bound("sync-fixture", vector.clone()),
        bound("sync-fixture", vector),
    )
}

fn fast_only_query(vector: Vec<f32>) -> TieredQueryEmbeddings {
    TieredQueryEmbeddings::fast_only(bound("sync-fixture", vector))
}

/// An in-memory tier loaded through exact FSVI v2 admission, so it carries
/// an ATTESTED space identity read out of the artifact's own header.
///
/// This is the only way to reach the attested branch: `from_vectors` is
/// legacy-unidentified by construction, and fabricating a fingerprint on
/// it would make the identity join pass without proving anything.
fn attested_tier(
    dir: &std::path::Path,
    file: &str,
    model: &str,
    sequence: u64,
    rows: &[(&str, &[f32])],
) -> (InMemoryVectorIndex, EmbeddingIdentityBundleV1) {
    let dimension = u32::try_from(rows[0].1.len()).expect("fixture dimension fits u32");
    let mut artifact = EmbeddingIdentityBundleV1::explicit_test_model(model, dimension);
    "fsvi-v2".clone_into(&mut artifact.storage.format);
    artifact.storage.quantization = frankensearch_core::generation::QuantizationFormat::F16;
    "little-endian".clone_into(&mut artifact.storage.endianness);
    let binding = frankensearch_index::FsviV2IdentityBinding::new(
        frankensearch_core::generation::ArtifactGenerationIdentityV1::new(sequence, [0x7a; 16])
            .expect("test generation"),
        artifact.freeze().expect("freeze artifact identity"),
    )
    .expect("valid FSVI v2 binding");

    let path = dir.join(file);
    let mut writer = frankensearch_index::VectorIndex::create_v2(&path, binding.clone())
        .expect("create_v2 fixture");
    for (doc_id, vector) in rows {
        writer.write_record(doc_id, vector).expect("write v2 row");
    }
    writer.finish().expect("finish v2 fixture");

    let owner = frankensearch_index::VectorIndex::open_admitted_v2(&path, &binding)
        .expect("admit the v2 fixture");
    let tier = InMemoryVectorIndex::from_admitted_v2(&owner).expect("load admitted tier");
    assert!(
        tier.space_identity_is_attested(),
        "the fixture is only meaningful while the tier's identity is header-attested"
    );
    // The query-side bundle of the SAME space: in-process f32 storage,
    // which can never be byte-equal to the persisted f16 one. That is why
    // the join is on the space, not the whole bundle.
    (
        tier,
        EmbeddingIdentityBundleV1::explicit_test_model(model, dimension),
    )
}

#[cfg(target_os = "linux")]
#[test]
fn admitted_v2_default_product_preserves_exact_result_across_cache_states() {
    // The first record fixes the global int8 scale at one. The remaining
    // tiny orthogonal rows and the final 0.003 f16 row then all quantize to the same
    // int8 pass-1 score for query [0, 1]. The approximate candidate route
    // cannot retain the final exact winner. The default product scan must
    // return that winner both with an attached sidecar and with a hostile
    // unavailable cache: the sidecar may accelerate the exact route, but
    // cache state must not select a different algorithm. This is a route
    // proof by an observable result, not a private-field assertion.
    const CORPUS_ROWS: usize = 10_002;
    let dir = temp_dir("residual-default-product-route");
    let source_path = dir.join("source.fsvi");
    let cache_dir = dir.join("residual-cache");
    let unavailable_cache = dir.join("residual-cache-is-a-file");
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&unavailable_cache)
        .expect("create a non-directory cache obstacle without replacement");

    let mut artifact = EmbeddingIdentityBundleV1::explicit_test_model("residual-route", 2);
    "fsvi-v2".clone_into(&mut artifact.storage.format);
    artifact.storage.quantization = frankensearch_core::generation::QuantizationFormat::F16;
    "little-endian".clone_into(&mut artifact.storage.endianness);
    let binding = frankensearch_index::FsviV2IdentityBinding::new(
        frankensearch_core::generation::ArtifactGenerationIdentityV1::new(73, [0x5e; 16])
            .expect("test generation"),
        artifact.freeze().expect("freeze test identity"),
    )
    .expect("valid FSVI-v2 binding");
    let mut writer = frankensearch_index::VectorIndex::create_v2(&source_path, binding.clone())
        .expect("create admitted-v2 source");
    writer
        .write_record("scale", &[1.0, 0.0])
        .expect("write global scale row");
    for row in 0..CORPUS_ROWS - 2 {
        writer
            .write_record(&format!("zero-{row:05}"), &[0.003, 0.0])
            .expect("write orthogonal candidate");
    }
    writer
        .write_record("exact-winner", &[0.0, 0.003])
        .expect("write exact-only winner");
    writer.finish().expect("seal admitted-v2 source");
    let owner = frankensearch_index::VectorIndex::open_admitted_v2(&source_path, &binding)
        .expect("admit immutable source");
    let query_vector = vec![0.0, 1.0];

    let approximate = InMemoryVectorIndex::from_admitted_v2(&owner)
        .expect("load independently for negative control")
        .search_top_k_int8_two_pass(&query_vector, 1, 3)
        .expect("run approximate negative control");
    assert_ne!(
        approximate[0].doc_id, "exact-winner",
        "the all-tied int8 candidate pass must demonstrate why this route needs exact selection"
    );

    let config = TwoTierConfig {
        fast_only: true,
        ..TwoTierConfig::default()
    };
    let query_identity = EmbeddingIdentityBundleV1::explicit_test_model("residual-route", 2);
    assert!(
        CORPUS_ROWS >= SearchParams::default().parallel_threshold,
        "the product fixture must remain in the default parallel target scale"
    );
    let unavailable = SyncTwoTierSearcher::from_admitted_v2_with_residual_sidecar_cache(
        &owner,
        &unavailable_cache,
        None,
        config.clone(),
    )
    .expect("an unavailable cache preserves the shipping exact product");
    assert!(
        !unavailable.index.fast_index().has_exact_residual_sidecar(),
        "the regular-file cache obstacle must leave no sidecar attached"
    );
    let unavailable_query = TieredQueryEmbeddings::fast_only(
        BoundQueryEmbedding::new(query_vector.clone(), query_identity.clone())
            .expect("bind exact unavailable-cache query identity"),
    );
    let (unavailable_results, _) = unavailable
        .search_collect(&unavailable_query, 1)
        .expect("unavailable-cache default product search");
    assert_eq!(
        unavailable_results[0].doc_id, "exact-winner",
        "cache failure must retain the same exact winner rather than selecting int8 candidates"
    );

    std::fs::create_dir(&cache_dir).expect("create fresh immutable cache directory");
    let searcher = SyncTwoTierSearcher::from_admitted_v2_with_residual_sidecar_cache(
        &owner,
        &cache_dir,
        None,
        config.clone(),
    )
    .expect("construct shipping sync product through the sidecar cache route");
    assert!(
        searcher.search_params.is_none(),
        "the regression exercises the default product search selection"
    );
    let query = TieredQueryEmbeddings::fast_only(
        BoundQueryEmbedding::new(query_vector.clone(), query_identity.clone())
            .expect("bind exact query identity"),
    );
    let (results, _) = searcher
        .search_collect(&query, 1)
        .expect("default product search");
    assert_eq!(
        results[0].doc_id, "exact-winner",
        "an attached sidecar preserves the same shipping exact result"
    );
    let entry_count_after_publish = std::fs::read_dir(&cache_dir)
        .expect("read immutable cache directory")
        .flatten()
        .count();
    assert_eq!(
        entry_count_after_publish, 1,
        "exactly one sidecar was published"
    );

    let reopened = SyncTwoTierSearcher::from_admitted_v2_with_residual_sidecar_cache(
        &owner, &cache_dir, None, config,
    )
    .expect("reopen shipping sync product through the existing cache entry");
    let reopened_query = TieredQueryEmbeddings::fast_only(
        BoundQueryEmbedding::new(query_vector, query_identity).expect("bind reopened query"),
    );
    let (reopened_results, _) = reopened
        .search_collect(&reopened_query, 1)
        .expect("reopened default product search");
    assert_eq!(reopened_results[0].doc_id, "exact-winner");
    assert_eq!(
        std::fs::read_dir(&cache_dir)
            .expect("read reused immutable cache directory")
            .flatten()
            .count(),
        entry_count_after_publish,
        "reopening selects the generation-matched artifact without overwriting or publishing another"
    );
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);
    for _ in 0..1024 {
        let nonce = TEMP_NONCE.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-sync-dbp10-{label}-{}-{}-{nonce}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        match std::fs::create_dir(&dir) {
            Ok(()) => return dir,
            // Falling out of the match is the next attempt: this arm is the
            // last statement in the loop body, so an explicit `continue`
            // says nothing the control flow does not already say.
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => panic!("create unique temp dir: {error}"),
        }
    }
    panic!("exhausted unique temp directory names")
}

fn make_index() -> Arc<InMemoryTwoTierIndex> {
    let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
    let fast_vectors = vec![vec![1.0, 0.0], vec![0.7, 0.3], vec![0.0, 1.0]];
    let quality_vectors = vec![vec![0.2, 0.8], vec![1.0, 0.0], vec![0.0, 1.0]];
    let fast = InMemoryVectorIndex::from_vectors(doc_ids.clone(), fast_vectors, 2).unwrap();
    let quality = InMemoryVectorIndex::from_vectors(doc_ids, quality_vectors, 2).unwrap();
    Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)))
}

fn lexical_result(doc_id: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.into(),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

struct StaticLexical {
    hits: Vec<ScoredResult>,
}

impl SyncLexicalSearch for StaticLexical {
    fn search_sync(&self, _query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
        Ok(self.hits.iter().take(limit).cloned().collect())
    }
}

struct ExcludeB;

impl SearchFilter for ExcludeB {
    fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
        doc_id != "b"
    }

    fn name(&self) -> &'static str {
        "exclude-b"
    }
}

fn vector_hit(doc_id: &str, index: u32, score: f32) -> VectorHit {
    VectorHit {
        doc_id: doc_id.into(),
        index,
        score,
    }
}

/// The union conversion carries the blended score and the no-index
/// sentinel through, and fabricates NO per-tier evidence.
///
/// This replaces `aligned_numeric_score_lookup_matches_doc_id_maps`,
/// whose subject — a numeric lookup keyed by fast-tier row ordinal — was
/// only sound while the quality tier was a re-scored subset of the fast
/// pool. Under independent retrieval a quality-only document has no fast
/// ordinal, and the old helper's `.unwrap_or((hit.score, None))` fallback
/// would have reported the BLENDED score as that document's fast score.
#[test]
fn union_conversion_reports_no_index_and_no_fabricated_evidence() {
    let blended = vec![
        vector_hit("quality-only", u32::MAX, 0.91),
        vector_hit("a", 10, 0.88),
    ];
    let actual =
        unique_vector_hits_to_scored_results_owned(blended, 2, ScoreSource::SemanticQuality);
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].doc_id, "quality-only");
    assert_eq!(
        actual[0].index, None,
        "a document with no fast-tier row must not report one"
    );
    assert_eq!(actual[1].index, Some(10));
    for result in &actual {
        assert_eq!(
            (result.fast_score, result.quality_score),
            (None, None),
            "per-tier evidence is attached from the pools, never invented here"
        );
        assert_eq!(result.source, ScoreSource::SemanticQuality);
    }
}

/// bd-dbp10 acceptance 4, THE PLANTED NEGATIVE: the quality tier must be
/// scored with the QUALITY-bound query, never the fast one.
///
/// The fixture makes the two tiers disagree on purpose. The fast tier puts
/// `a` first for the fast query; in the quality tier the fast query's
/// direction points at `a` too, but the QUALITY query points at `c`. So
/// the refined winner is `c` if and only if the quality tier saw its own
/// query. Feeding it the fast vector — which is exactly what this code did
/// before — returns `a`, and the assertion below is what says so.
///
/// Note the two vectors have the SAME WIDTH. That is the whole reason this
/// bug could live in production: every dimension check passes.
#[test]
fn the_quality_tier_is_scored_with_the_quality_query_not_the_fast_one() {
    let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
    // Fast space: the fast query [1,0] ranks a > b > c.
    let fast = InMemoryVectorIndex::from_vectors(
        doc_ids.clone(),
        vec![vec![1.0, 0.0], vec![0.7, 0.7], vec![0.0, 1.0]],
        2,
    )
    .unwrap();
    // Quality space: the fast query [1,0] would rank a first here too,
    // but the quality query [0,1] ranks c first.
    let quality = InMemoryVectorIndex::from_vectors(
        doc_ids,
        vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]],
        2,
    )
    .unwrap();
    let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)));
    let searcher = SyncTwoTierSearcher::new(
        index,
        TwoTierConfig {
            // Quality-dominant blend so the quality tier's opinion decides
            // the refined order rather than merely nudging it.
            quality_weight: 1.0,
            ..TwoTierConfig::default()
        },
    );

    let query = TieredQueryEmbeddings::progressive(
        bound("sync-fixture", vec![1.0, 0.0]),
        bound("sync-fixture", vec![0.0, 1.0]),
    );
    let (results, _) = searcher.search_collect(&query, 3).unwrap();
    assert_eq!(
        results[0].doc_id,
        "c",
        "the refined winner must come from the QUALITY query's ranking; got {:?}",
        results
            .iter()
            .map(|r| r.doc_id.as_str())
            .collect::<Vec<_>>()
    );
}

/// bd-dbp10 acceptance 2: on an attested index every identity join
/// completes before any vector read, and a query from another space is
/// refused by field name.
#[test]
fn an_attested_tier_refuses_a_foreign_space_query() {
    let dir = temp_dir("attested-refusal");
    let (fast, fast_identity) = attested_tier(
        &dir,
        "vector.fast.idx",
        "dbp10-fast",
        11,
        &[("a", &[1.0, 0.0]), ("b", &[0.0, 1.0])],
    );
    let index = Arc::new(InMemoryTwoTierIndex::new(fast, None));
    let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());

    // Positive control: the space this tier was written in.
    let (results, _) = searcher
        .search_collect(
            &TieredQueryEmbeddings::fast_only(
                BoundQueryEmbedding::new(vec![1.0, 0.0], fast_identity).expect("bind"),
            ),
            2,
        )
        .expect("the producing space is admitted");
    assert_eq!(results[0].doc_id, "a");

    // Same width, different space: refused, by the same contract field the
    // async searcher and the owner-backed activation use.
    let error = searcher
        .search_collect(&fast_only_query(vec![1.0, 0.0]), 2)
        .expect_err("a foreign embedding space must be refused");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. }
                if field == "query_embedding.fast.space_identity"
        ),
        "got {error:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// bd-ctzo C2 on the sync path: with an ATTESTED quality tier the refined
/// phase retrieves independently, so a document the fast tier does not
/// contain at all becomes reachable.
#[test]
fn an_attested_quality_tier_reaches_a_document_outside_the_fast_pool() {
    let dir = temp_dir("attested-union");
    let (fast, fast_identity) = attested_tier(
        &dir,
        "vector.fast.idx",
        "dbp10-union-fast",
        21,
        &[
            ("doc-near", &[1.0, 0.0, 0.0]),
            ("doc-far", &[0.0, 0.0, 1.0]),
        ],
    );
    // The quality tier holds a document the FAST TIER DOES NOT CONTAIN.
    // No rescoring of a fast-selected pool can produce it.
    let (quality, quality_identity) = attested_tier(
        &dir,
        "vector.quality.idx",
        "dbp10-union-quality",
        21,
        &[
            ("doc-near", &[0.0, 0.0, 1.0]),
            ("doc-far", &[0.0, 0.5, 0.0]),
            ("doc-quality-only", &[0.0, 1.0, 0.0]),
        ],
    );
    let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)));
    let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());

    let query = TieredQueryEmbeddings::progressive(
        BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], fast_identity).expect("bind fast"),
        BoundQueryEmbedding::new(vec![0.0, 1.0, 0.0], quality_identity).expect("bind quality"),
    );
    let (results, _) = searcher.search_collect(&query, 3).unwrap();
    let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
    assert!(
        ids.contains(&"doc-quality-only"),
        "independent quality retrieval must reach a document the fast tier lacks; got {ids:?}"
    );
    let quality_only = results
        .iter()
        .find(|r| r.doc_id == "doc-quality-only")
        .expect("present");
    assert_eq!(
        quality_only.index, None,
        "a document with no fast-tier row must not report one"
    );
    assert_eq!(
        quality_only.fast_score, None,
        "the fast tier never saw it, so it has no fast evidence"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// bd-dbp10 acceptance 3: a legacy tier carries no identity to join
/// against, and must stay searchable rather than becoming a refusal.
#[test]
fn a_legacy_unidentified_index_stays_searchable() {
    let index = make_index();
    assert!(
        index.fast_index().space_fingerprint_hex().is_none(),
        "the fixture is only meaningful while the index is legacy-unidentified"
    );
    let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
    let (results, metrics) = searcher
        .search_collect(&tiered(vec![1.0, 0.0]), 3)
        .expect("a legacy index stays searchable");
    assert!(
        !results.is_empty(),
        "the legacy lane must still return hits"
    );
    assert_eq!(
        metrics.skip_reason, None,
        "the legacy lane still refines; it is unidentified, not degraded"
    );
}

/// A quality tier the caller bound no quality embedding for is SKIPPED
/// with a typed reason — never served by reusing the fast embedding.
#[test]
fn a_fast_only_query_skips_the_quality_tier_instead_of_reusing_the_fast_vector() {
    let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
    let (_, metrics) = searcher
        .search_collect(&fast_only_query(vec![1.0, 0.0]), 3)
        .unwrap();
    assert_eq!(
        metrics.skip_reason.as_deref(),
        Some("quality_query_embedding_absent"),
        "the index HAS a quality tier; the query simply bound nothing for it"
    );
    assert_eq!(metrics.phase2_vectors_searched, 0);
}

#[test]
fn hash_control_skips_quality_and_does_not_become_semantic_quality() {
    let hash_bundle = EmbeddingIdentityBundleV1::explicit_test_model("fnv1a-256", 2);
    let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
    let fast = InMemoryVectorIndex::from_vectors_with_identity(
        doc_ids.clone(),
        vec![vec![1.0, 0.0], vec![0.7, 0.3], vec![0.0, 1.0]],
        2,
        &hash_bundle.space,
    )
    .expect("hash-control fast index");
    let quality = InMemoryVectorIndex::from_vectors(
        doc_ids,
        vec![vec![0.2, 0.8], vec![1.0, 0.0], vec![0.0, 1.0]],
        2,
    )
    .expect("quality index");
    let searcher = SyncTwoTierSearcher::new(
        Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality))),
        TwoTierConfig::default(),
    );
    let query = TieredQueryEmbeddings::progressive(
        BoundQueryEmbedding::new(vec![1.0, 0.0], hash_bundle).expect("hash-control query binds"),
        BoundQueryEmbedding::new(
            vec![1.0, 0.0],
            EmbeddingIdentityBundleV1::explicit_test_model("sync-fixture", 2),
        )
        .expect("quality query binds"),
    );
    let (results, metrics) = searcher
        .search_collect(&query, 2)
        .expect("hash-control search still returns hits");
    assert!(!results.is_empty());
    assert!(
        results
            .iter()
            .all(|result| result.source == ScoreSource::HashControl),
        "hash-control hits must not be relabeled SemanticQuality: {:?}",
        results
            .iter()
            .map(|result| result.source)
            .collect::<Vec<_>>()
    );
    assert_eq!(metrics.phase2_vectors_searched, 0);
    assert_eq!(
        metrics.skip_reason.as_deref(),
        Some("non_semantic_fast_embedder_vector_control")
    );
    assert!(metrics.hash_control_candidates > 0);
    assert_eq!(metrics.semantic_candidates, 0);
    let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
    assert!(
        phases
            .iter()
            .any(|phase| matches!(phase, SearchPhase::Initial { .. }))
    );
    assert!(
        !phases
            .iter()
            .any(|phase| matches!(phase, SearchPhase::Refined { .. })),
        "hash-control Initial must not schedule a Refined phase"
    );
    assert!(
        matches!(phases.first(), Some(SearchPhase::Initial { .. })),
        "first hash-control phase must be Initial, got {:?}",
        phases.first()
    );
    if let Some(SearchPhase::Initial {
        metrics: phase_metrics,
        ..
    }) = phases.first()
    {
        assert_eq!(
            phase_metrics.hash_control_candidates,
            metrics.hash_control_candidates
        );
        assert!(phase_metrics.hash_control_candidates > 0);
    }
}

#[test]
fn search_collect_returns_refined_results() {
    let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
    let (results, metrics) = searcher.search_collect(&tiered(vec![1.0, 0.0]), 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].source, ScoreSource::SemanticQuality);
    assert!(metrics.phase1_total_ms >= 0.0);
    assert!(metrics.phase2_total_ms >= 0.0);
    // The semantic lane produced hits, so it must not be classified.
    assert_eq!(metrics.zero_signal, None);
}

#[test]
fn empty_index_search_carries_typed_zero_signal() {
    let fast = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 2).unwrap();
    let index = Arc::new(InMemoryTwoTierIndex::new(fast, None));
    let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
    let (results, metrics) = searcher.search_collect(&tiered(vec![1.0, 0.0]), 3).unwrap();
    assert!(results.is_empty());
    assert_eq!(
        metrics.zero_signal,
        Some(ZeroSignalReason::NewlyCreatedEmpty),
        "an empty semantic lane must say why, not just return nothing"
    );
}

#[test]
fn filter_eliminating_every_candidate_classifies_as_filter() {
    struct ExcludeAll;
    impl SearchFilter for ExcludeAll {
        fn matches(&self, _doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
            false
        }
        fn name(&self) -> &'static str {
            "exclude-all"
        }
    }
    let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
    let (results, metrics) = searcher
        .search_collect_with_filter(&tiered(vec![1.0, 0.0]), 3, Some(&ExcludeAll))
        .unwrap();
    assert!(results.is_empty());
    assert_eq!(
        metrics.zero_signal,
        Some(ZeroSignalReason::FilterEliminatedAll)
    );
}

#[test]
fn zero_norm_query_over_empty_index_prefers_request_scoped_reason() {
    // Request-scoped pre-scan classification outranks index state:
    // a zero-norm query is the caller's defect even on an empty index.
    let fast = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 2).unwrap();
    let index = Arc::new(InMemoryTwoTierIndex::new(fast, None));
    let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
    let (results, metrics) = searcher
        .search_collect(&fast_only_query(vec![0.0, 0.0]), 3)
        .unwrap();
    assert!(results.is_empty());
    assert_eq!(metrics.zero_signal, Some(ZeroSignalReason::ZeroNormQuery));
}

#[test]
fn rrf_weights_flow_through_searcher_to_fusion() {
    // Lexical favors "c" (then "b"); the quality/semantic tier favors a different doc
    // for query [1,0]. Extreme opposite tier weights must therefore flip the top result,
    // proving `with_rrf_weights` / `with_rrf_tiebreak` reach the fusion `RrfConfig`.
    let make_lex = || {
        Arc::new(StaticLexical {
            hits: vec![lexical_result("c", 10.0), lexical_result("b", 5.0)],
        })
    };
    let q = tiered(vec![1.0_f32, 0.0]);

    let sem_heavy = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_lexical(make_lex())
        .with_rrf_weights(0.01, 100.0)
        .with_rrf_tiebreak(crate::rrf::RrfTiebreak::Hash);
    let lex_heavy = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_lexical(make_lex())
        .with_rrf_weights(100.0, 0.01);

    let (sem_res, _) = sem_heavy.search_collect(&q, 3).unwrap();
    let (lex_res, _) = lex_heavy.search_collect(&q, 3).unwrap();
    assert!(!sem_res.is_empty() && !lex_res.is_empty());
    assert_ne!(
        sem_res[0].doc_id, lex_res[0].doc_id,
        "opposite tier weights must change the fused top result (weights reach fusion)"
    );
}

#[test]
fn nqc_dense_downweight_empty_sketch_is_bit_identical_to_base_weight() {
    let hits = [lexical_result("a", 10.0), lexical_result("b", 1.0)];
    let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_rrf_weights(1.0, 1.3)
        .with_nqc_dense_downweight(0.5, 0.1, NqcDenseWeight::new());

    assert_eq!(
        searcher.effective_semantic_weight(&hits).to_bits(),
        searcher.rrf_semantic_weight.to_bits(),
    );
}

#[test]
fn nqc_dense_downweight_flows_through_searcher_to_fusion() {
    // Both searchers up-weight the dense tier (semantic 5×) so it dominates by default;
    // enabling the NQC dense down-weight with a sample below the query's NQC drives the
    // dense weight to 0, so lexical (favoring "c") dominates instead. Different top =>
    // the opt-in down-weight reaches the fusion RrfConfig.
    let make_lex = || {
        Arc::new(StaticLexical {
            hits: vec![lexical_result("c", 10.0), lexical_result("b", 5.0)],
        })
    };
    let q = tiered(vec![1.0_f32, 0.0]);

    let neutral = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_lexical(make_lex())
        .with_rrf_weights(1.0, 5.0);
    // Query NQC (cv of lexical scores [10, 5] ≈ 0.333) is above every sampled value, so
    // its percentile is 1.0 and dense_weight(beta=1, w_min=0.05) = clip(1 - 1·1, 0.05, 1)
    // = 0.05 → effective semantic weight 5·0.05 = 0.25 (< lexical 1.0), still > 0 so it is
    // not neutralized by the tier-weight sanitizer.
    let sample = NqcDenseWeight::from_sample(&[0.1, 0.2, 0.3]);
    let downweighted = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_lexical(make_lex())
        .with_rrf_weights(1.0, 5.0)
        .with_nqc_dense_downweight(1.0, 0.05, sample);

    let (neutral_res, _) = neutral.search_collect(&q, 3).unwrap();
    let (down_res, _) = downweighted.search_collect(&q, 3).unwrap();
    assert!(!neutral_res.is_empty() && !down_res.is_empty());
    assert_eq!(
        down_res[0].doc_id, "c",
        "zeroing the dense tier lets lexical dominate"
    );
    assert_ne!(
        neutral_res[0].doc_id, down_res[0].doc_id,
        "the NQC dense down-weight must change the fused top (it reaches fusion)"
    );
}

#[test]
fn nqc_adaptive_cold_start_is_bit_identical_to_base_weight() {
    // A freshly-enabled adaptive down-weight has an empty online sketch, so a cold-start
    // query's effective semantic weight is bit-identical to the base (safe to default-on).
    let hits = [lexical_result("a", 10.0), lexical_result("b", 1.0)];
    let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_rrf_weights(1.0, 1.3)
        .with_nqc_dense_downweight_adaptive(0.5, 0.1, 128, 8, 4);
    assert_eq!(
        searcher.effective_semantic_weight(&hits).to_bits(),
        searcher.rrf_semantic_weight.to_bits(),
        "cold-start (empty online sketch) leaves fusion byte-identical"
    );
}

#[test]
fn nqc_adaptive_warms_up_then_down_weights() {
    // Warm the online sketch with observed queries, then a high-NQC query lands at a high
    // percentile and is down-weighted below the base semantic weight (reaches fusion online).
    let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_rrf_weights(1.0, 1.0)
        .with_nqc_dense_downweight_adaptive(0.5, 0.05, 128, 8, 4);
    let warm = [
        lexical_result("a", 5.0),
        lexical_result("b", 5.0),
        lexical_result("c", 5.0),
    ];
    for _ in 0..12 {
        let _ = searcher.effective_semantic_weight(&warm);
    }
    let peaked = [
        lexical_result("a", 100.0),
        lexical_result("b", 1.0),
        lexical_result("c", 0.5),
    ];
    let weight = searcher.effective_semantic_weight(&peaked);
    assert!(
        weight < searcher.rrf_semantic_weight,
        "a high-NQC query is down-weighted after the online sketch warms up, got {weight}"
    );
}

#[test]
fn nqc_adaptive_defaults_enable_neutral_cold_start_and_disabled_clears_it() {
    let hits = [lexical_result("a", 10.0), lexical_result("b", 1.0)];
    // Blessed defaults: enabled, but cold-start (empty sketch) is byte-identical to base.
    let defaults = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_rrf_weights(1.0, 1.3)
        .with_nqc_dense_downweight_adaptive_defaults();
    assert_eq!(
        defaults.effective_semantic_weight(&hits).to_bits(),
        defaults.rrf_semantic_weight.to_bits(),
        "production-default cold start is neutral"
    );
    // Disabling clears both the adaptive and static paths -> byte-identical, no observation.
    let disabled = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
        .with_rrf_weights(1.0, 1.3)
        .with_nqc_dense_downweight_adaptive_defaults()
        .with_nqc_dense_downweight_disabled();
    assert_eq!(
        disabled.effective_semantic_weight(&hits).to_bits(),
        disabled.rrf_semantic_weight.to_bits(),
        "disabled escape hatch is byte-identical"
    );
}

#[test]
fn search_iter_yields_initial_then_refined() {
    let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
    let phases = searcher
        .search_iter(&tiered(vec![1.0, 0.0]), 2)
        .collect::<Vec<_>>();
    assert_eq!(phases.len(), 2);
    assert!(matches!(phases[0], SearchPhase::Initial { .. }));
    assert!(matches!(phases[1], SearchPhase::Refined { .. }));
}

#[test]
fn fast_only_mode_skips_phase_two() {
    let config = TwoTierConfig {
        fast_only: true,
        ..TwoTierConfig::default()
    };
    let searcher = SyncTwoTierSearcher::new(make_index(), config);
    let phases = searcher
        .search_iter(&tiered(vec![1.0, 0.0]), 2)
        .collect::<Vec<_>>();
    assert_eq!(phases.len(), 1);
    assert!(matches!(phases[0], SearchPhase::Initial { .. }));
}

#[test]
fn filter_is_applied_to_fast_and_refined_results() {
    let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
    let (results, _) = searcher
        .search_collect_with_filter(&tiered(vec![1.0, 0.0]), 3, Some(&ExcludeB))
        .unwrap();
    assert!(results.iter().all(|result| result.doc_id != "b"));
}

/// An empty query vector is now refused at BIND time, not at search time.
///
/// This test previously called `search_collect(&[], 3)` and asserted the
/// searcher returned `DimensionMismatch`. It had been failing on trunk:
/// the empty slice reached the scan and came back `Ok` with a
/// `ZeroNormQuery` classification instead. The typed query surface makes
/// the question moot — a zero-length vector cannot be bound to a
/// two-dimensional space at all, so the refusal now happens strictly
/// earlier and cannot be reached by a search. Same defect, moved to where
/// it is provable.
#[test]
fn an_empty_query_vector_cannot_even_be_bound() {
    let error = BoundQueryEmbedding::new(
        Vec::new(),
        EmbeddingIdentityBundleV1::explicit_test_model("sync-fixture", 2),
    )
    .expect_err("a zero-length vector is not a member of a 2-dimensional space");
    assert!(
        matches!(error, SearchError::DimensionMismatch { .. }),
        "got {error:?}"
    );
}

#[test]
fn lexical_fusion_can_introduce_lexical_only_hits() {
    let lexical = Arc::new(StaticLexical {
        hits: vec![lexical_result("lex-only", 10.0), lexical_result("a", 9.0)],
    });
    let searcher =
        SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default()).with_lexical(lexical);
    let (results, _) = searcher.search_collect(&tiered(vec![1.0, 0.0]), 3).unwrap();
    assert!(results.iter().any(|result| result.doc_id == "lex-only"));
    assert_eq!(
        results
            .iter()
            .find(|result| result.doc_id == "lex-only")
            .map(|result| result.source),
        Some(ScoreSource::Lexical),
        "a fused lexical-only hit is not hybrid semantic search"
    );
}

#[cfg(feature = "quill")]
#[test]
fn quill_sync_adapter_is_read_only_and_maps_cancellation() {
    use frankensearch_core::IndexableDocument;
    use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchIndex};

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QuillSyncLexicalSearch>();

        let directory = tempfile::tempdir().expect("Quill fixture directory");
        let config = QuillConfig {
            deterministic_ingest: true,
            ..QuillConfig::default()
        };
        let writer = QuillIndex::create(&cx, directory.path(), config.clone())
            .await
            .expect("create durable Quill writer");
        writer
            .index_documents(
                &cx,
                &[IndexableDocument::new(
                    "quill-only",
                    "native quill lexical result",
                )],
            )
            .await
            .expect("index Quill fixture");
        writer.commit(&cx).await.expect("commit Quill fixture");
        drop(writer);

        let quill = Arc::new(
            QuillSearchIndex::open(&cx, directory.path(), config.clone())
                .await
                .expect("open read-only Quill search handle"),
        );
        let reopened_writer = QuillIndex::open(&cx, directory.path(), config)
            .await
            .expect("read-only search handle must not retain the writer lease");
        drop(reopened_writer);

        let lexical = Arc::new(QuillSyncLexicalSearch::new(
            Arc::clone(&quill),
            cx.clone(),
            "native quill",
        ));
        assert_eq!(lexical.query(), "native quill");
        let hits = lexical.search_sync(&[f32::NAN], 1).expect("sync search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "quill-only");
        assert_eq!(hits[0].source, ScoreSource::Lexical);
        assert_eq!(hits[0].lexical_score, Some(hits[0].score));

        let searcher =
            SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default()).with_lexical(lexical);
        let (fused, _) = searcher
            .search_collect(&tiered(vec![1.0, 0.0]), 3)
            .expect("hybrid sync search");
        assert!(fused.iter().any(|hit| hit.doc_id == "quill-only"));

        let cancelled = cx.clone();
        cancelled.set_cancel_requested(true);
        let adapter = QuillSyncLexicalSearch::new(quill, cancelled, "native");
        assert!(matches!(
            adapter.search_sync(&[], 1),
            Err(SearchError::Cancelled { ref phase, .. }) if phase == "search"
        ));
    });
}
