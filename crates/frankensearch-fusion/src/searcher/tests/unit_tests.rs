use super::*;

#[test]
fn scaled_budget_clamps_positive_budget_to_at_least_one() {
    assert_eq!(scaled_budget(1, 0.5), 1);
    assert_eq!(scaled_budget(1, 0.25), 1);
    assert_eq!(scaled_budget(2, 0.5), 1);
    assert_eq!(scaled_budget(0, 0.5), 0);
    assert_eq!(scaled_budget(4, 0.0), 0);
}

fn phase2_lookup_checksum_linear(
    blended: &[VectorHit],
    fast_hits: &[VectorHit],
    quality_hits: &[VectorHit],
) -> f32 {
    blended.iter().fold(0.0_f32, |acc, hit| {
        let fast_score = fast_hits
            .iter()
            .find(|h| h.doc_id == hit.doc_id)
            .map_or(0.0_f32, |h| h.score);
        let quality_score = quality_hits
            .iter()
            .find(|h| h.doc_id == hit.doc_id)
            .map_or(0.0_f32, |h| h.score);
        acc + fast_score + quality_score + hit.score
    })
}

fn phase2_lookup_checksum_mapped(
    blended: &[VectorHit],
    fast_scores_by_doc: &AHashMap<&str, f32>,
    quality_scores_by_doc: &AHashMap<&str, f32>,
) -> f32 {
    blended.iter().fold(0.0_f32, |acc, hit| {
        let fast_score = fast_scores_by_doc
            .get(hit.doc_id.as_str())
            .copied()
            .unwrap_or(0.0_f32);
        let quality_score = quality_scores_by_doc
            .get(hit.doc_id.as_str())
            .copied()
            .unwrap_or(0.0_f32);
        acc + fast_score + quality_score + hit.score
    })
}

fn build_phase2_lookup_fixture(
    doc_count: usize,
) -> (Vec<VectorHit>, Vec<VectorHit>, Vec<VectorHit>) {
    let fast_hits: Vec<VectorHit> = (0..doc_count)
        .map(|idx| VectorHit {
            index: u32::try_from(idx).expect("idx fits in u32 for test fixture"),
            score: idx as f32 * 0.001_f32,
            doc_id: format!("doc-{idx}").into(),
        })
        .collect();
    let quality_hits: Vec<VectorHit> = (0..doc_count)
        .step_by(3)
        .map(|idx| VectorHit {
            index: u32::try_from(idx).expect("idx fits in u32 for test fixture"),
            score: idx as f32 * 0.002_f32,
            doc_id: format!("doc-{idx}").into(),
        })
        .collect();
    let blended: Vec<VectorHit> = (0..doc_count)
        .map(|idx| VectorHit {
            index: u32::try_from(idx).expect("idx fits in u32 for test fixture"),
            doc_id: format!("doc-{idx}").into(),
            score: idx as f32 * 0.0005_f32,
        })
        .collect();
    (fast_hits, quality_hits, blended)
}

#[test]
fn phase2_lookup_maps_match_linear_scan_oracle() {
    let (fast_hits, quality_hits, blended) = build_phase2_lookup_fixture(10_000);
    let fast_scores_by_doc: AHashMap<&str, f32> = fast_hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score))
        .collect();
    let quality_scores_by_doc: AHashMap<&str, f32> = quality_hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score))
        .collect();

    let linear = phase2_lookup_checksum_linear(&blended, &fast_hits, &quality_hits);
    let mapped =
        phase2_lookup_checksum_mapped(&blended, &fast_scores_by_doc, &quality_scores_by_doc);

    let diff = (linear - mapped).abs();
    assert!(
        diff <= 0.0001_f32,
        "mapped lookup diverged from linear oracle: diff={diff}"
    );
}

// ─── bd-3a7q tests begin ───

#[test]
fn scaled_budget_negative_multiplier_returns_zero() {
    assert_eq!(scaled_budget(10, -1.0), 0);
    assert_eq!(scaled_budget(100, -0.5), 0);
}

#[test]
fn scaled_budget_exact_one_multiplier() {
    assert_eq!(scaled_budget(7, 1.0), 7);
    assert_eq!(scaled_budget(1, 1.0), 1);
}

#[test]
fn scaled_budget_large_values_do_not_panic() {
    let result = scaled_budget(usize::MAX / 2, 2.0);
    assert!(result >= 1);
}

#[test]
fn scaled_budget_nan_multiplier_returns_zero() {
    assert_eq!(scaled_budget(10, f32::NAN), 0);
}

#[test]
fn scaled_budget_infinity_multiplier_returns_zero() {
    assert_eq!(scaled_budget(10, f32::INFINITY), 0);
    assert_eq!(scaled_budget(10, f32::NEG_INFINITY), 0);
}

#[test]
fn scaled_budget_fractional_rounds_up() {
    // 3 * 0.4 = 1.2 → ceil = 2
    assert_eq!(scaled_budget(3, 0.4), 2);
    // 5 * 0.3 = 1.5 → ceil = 2
    assert_eq!(scaled_budget(5, 0.3), 2);
}

#[test]
fn embedder_tier_for_stage_quality_always_returns_quality() {
    assert_eq!(
        embedder_tier_for_stage(EmbeddingStage::Quality, ModelCategory::HashEmbedder),
        EmbedderTier::Quality
    );
    assert_eq!(
        embedder_tier_for_stage(EmbeddingStage::Quality, ModelCategory::StaticEmbedder),
        EmbedderTier::Quality
    );
    assert_eq!(
        embedder_tier_for_stage(EmbeddingStage::Quality, ModelCategory::TransformerEmbedder),
        EmbedderTier::Quality
    );
}

#[test]
fn embedder_tier_for_stage_fast_maps_category() {
    assert_eq!(
        embedder_tier_for_stage(EmbeddingStage::Fast, ModelCategory::HashEmbedder),
        EmbedderTier::Hash
    );
    assert_eq!(
        embedder_tier_for_stage(EmbeddingStage::Fast, ModelCategory::StaticEmbedder),
        EmbedderTier::Fast
    );
    assert_eq!(
        embedder_tier_for_stage(EmbeddingStage::Fast, ModelCategory::TransformerEmbedder),
        EmbedderTier::Quality
    );
}

#[test]
fn embedder_tier_for_stage_background_maps_same_as_fast() {
    assert_eq!(
        embedder_tier_for_stage(EmbeddingStage::Background, ModelCategory::HashEmbedder),
        EmbedderTier::Hash
    );
    assert_eq!(
        embedder_tier_for_stage(EmbeddingStage::Background, ModelCategory::StaticEmbedder),
        EmbedderTier::Fast
    );
    assert_eq!(
        embedder_tier_for_stage(
            EmbeddingStage::Background,
            ModelCategory::TransformerEmbedder
        ),
        EmbedderTier::Quality
    );
}

#[test]
fn next_telemetry_identifier_is_ulid_like_and_unique() {
    let id1 = next_telemetry_identifier("root");
    let id2 = next_telemetry_identifier("root");
    assert!(is_valid_ulid_like(&id1));
    assert!(is_valid_ulid_like(&id2));
    assert_ne!(id1, id2);
}

#[test]
fn next_telemetry_identifier_is_uppercase_crockford_base32() {
    let id = next_telemetry_identifier("evt");
    // "evt_" prefix (4 chars) + 26 ULID chars = 30
    assert_eq!(
        id.len(),
        30,
        "prefixed ULID: 4-char prefix + 26-char ULID body"
    );
    let body = &id["evt_".len()..];
    assert!(
        body.chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit()),
        "ULID body should use uppercase Crockford base32"
    );
}

#[test]
fn telemetry_timestamp_now_is_valid_rfc3339() {
    let ts = telemetry_timestamp_now();
    assert!(!ts.is_empty());
    assert!(
        frankensearch_core::rfc3339::parse_rfc3339_to_unix_nanos(&ts).is_ok(),
        "should be RFC3339"
    );
}

#[test]
fn telemetry_timestamp_fallback_constant_is_valid_rfc3339() {
    assert!(
        frankensearch_core::rfc3339::parse_rfc3339_to_unix_nanos(
            TELEMETRY_TIMESTAMP_FALLBACK_RFC3339
        )
        .is_ok(),
        "fallback timestamp must remain RFC3339"
    );
}

#[test]
fn cpu_pct_from_jiffies_first_sample_returns_zero() {
    let current = CpuJiffiesSnapshot {
        process_jiffies: 500,
        total_jiffies: 10_000,
    };
    assert!(
        (cpu_pct_from_jiffies(None, current) - 0.0).abs() < f64::EPSILON,
        "first sample should not report synthetic CPU utilization"
    );
}

#[test]
fn cpu_pct_from_jiffies_clamps_to_conformance_range() {
    let previous = CpuJiffiesSnapshot {
        process_jiffies: 100,
        total_jiffies: 1_000,
    };
    let current = CpuJiffiesSnapshot {
        process_jiffies: 200,
        total_jiffies: 1_100,
    };
    let cpu_pct = cpu_pct_from_jiffies(Some(previous), current);
    assert!(
        (cpu_pct - 100.0).abs() < f64::EPSILON,
        "high deltas should clamp at 100%"
    );
}

#[test]
fn parse_proc_total_jiffies_extracts_aggregate_sum() {
    let fixture = "cpu  10 20 30 40 50 60 70 80 90 100\ncpu0 1 2 3 4\n";
    assert_eq!(parse_proc_total_jiffies(fixture), Some(550));
}

#[test]
fn parse_proc_process_jiffies_handles_command_with_spaces() {
    let fixture = "1234 (two tier worker) S 1 2 3 4 5 6 7 8 9 10 200 300 0 0 0 0 0 0 0";
    assert_eq!(parse_proc_process_jiffies(fixture), Some(500));
}

#[test]
fn parse_proc_status_rss_bytes_extracts_vm_rss() {
    let fixture = "Name:\tproc\nVmRSS:\t   4096 kB\nThreads:\t8\n";
    assert_eq!(parse_proc_status_rss_bytes(fixture), Some(4_194_304));
}

#[test]
fn parse_proc_io_bytes_extracts_read_and_write_bytes() {
    let fixture = "rchar: 1\nwchar: 2\nread_bytes: 333\nwrite_bytes: 444\n";
    assert_eq!(parse_proc_io_bytes(fixture), Some((333, 444)));
}

#[test]
fn parse_proc_load_avg_1m_extracts_first_value() {
    let fixture = "1.42 0.99 0.55 2/321 9999\n";
    assert_eq!(parse_proc_load_avg_1m(fixture), Some(1.42));
}

#[test]
fn vector_hits_to_scored_results_truncates_to_k() {
    let hits: Vec<VectorHit> = (0..10)
        .map(|i| VectorHit {
            index: i,
            score: (10 - i) as f32,
            doc_id: format!("doc-{i}").into(),
        })
        .collect();
    let config = TwoTierConfig::default();
    let results = vector_hits_to_scored_results(&hits, 3, &config, "test-fast");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].doc_id, "doc-0");
    assert_eq!(results[2].doc_id, "doc-2");
}

#[test]
fn vector_hits_to_scored_results_empty_hits() {
    let config = TwoTierConfig::default();
    let results = vector_hits_to_scored_results(&[], 5, &config, "test-fast");
    assert!(results.is_empty());
}

#[test]
fn vector_hits_to_scored_results_sets_correct_fields() {
    let hits = vec![VectorHit {
        index: 0,
        score: 0.95,
        doc_id: "my-doc".into(),
    }];
    let config = TwoTierConfig::default();
    let results = vector_hits_to_scored_results(&hits, 10, &config, "test-fast");
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert_eq!(r.doc_id, "my-doc");
    assert!((r.score - 0.95).abs() < f32::EPSILON);
    assert_eq!(r.source, ScoreSource::SemanticFast);
    assert!(
        r.fast_score
            .is_some_and(|s| (s - 0.95).abs() < f32::EPSILON)
    );
    assert!(r.quality_score.is_none());
    assert!(r.lexical_score.is_none());
    assert!(r.rerank_score.is_none());
    assert!(r.metadata.is_none());
}

#[test]
fn fused_hits_hybrid_source_classification() {
    let fused = vec![frankensearch_core::types::FusedHit {
        doc_id: "hybrid-doc".into(),
        rrf_score: 2.0,
        lexical_rank: Some(0),
        semantic_rank: Some(1),
        hash_rank: None,
        semantic_index: Some(1),
        lexical_score: Some(3.0),
        semantic_score: Some(0.8),
        hash_score: None,
        in_both_sources: true,
    }];
    let results = fused_hits_to_scored_results(&fused, &[], false, "fast-test", 60.0);
    assert_eq!(results[0].source, ScoreSource::Hybrid);
    assert_eq!(results[0].fast_score, Some(0.8));
    assert_eq!(results[0].lexical_score, Some(3.0));
}

#[test]
fn fused_hits_semantic_only_source() {
    let fused = vec![frankensearch_core::types::FusedHit {
        doc_id: "sem-only".into(),
        rrf_score: 1.0,
        lexical_rank: None,
        semantic_rank: Some(0),
        hash_rank: None,
        semantic_index: Some(0),
        lexical_score: None,
        semantic_score: Some(0.9),
        hash_score: None,
        in_both_sources: false,
    }];
    let results = fused_hits_to_scored_results(&fused, &[], false, "fast-test", 60.0);
    assert_eq!(results[0].source, ScoreSource::SemanticFast);
}

#[test]
fn hash_control_does_not_promote_to_semantic_quality() {
    assert!(!source_can_promote_to_quality(ScoreSource::HashControl));
    assert!(!source_can_promote_to_quality(ScoreSource::Lexical));
    assert!(!source_can_promote_to_quality(ScoreSource::Hybrid));
    assert!(source_can_promote_to_quality(ScoreSource::SemanticFast));
    assert!(source_can_promote_to_quality(ScoreSource::SemanticQuality));
    assert!(source_can_promote_to_quality(ScoreSource::Reranked));

    assert_eq!(
        promoted_quality_source(ScoreSource::HashControl, true),
        ScoreSource::HashControl
    );
    assert_eq!(
        promoted_quality_source(ScoreSource::Lexical, true),
        ScoreSource::Lexical
    );
    assert_eq!(
        promoted_quality_source(ScoreSource::Hybrid, true),
        ScoreSource::Hybrid
    );
    assert_eq!(
        promoted_quality_source(ScoreSource::SemanticFast, true),
        ScoreSource::SemanticQuality
    );
    assert_eq!(
        promoted_quality_source(ScoreSource::SemanticFast, false),
        ScoreSource::SemanticFast
    );
    assert_eq!(
        promoted_quality_source(ScoreSource::HashControl, false),
        ScoreSource::HashControl
    );
}

#[test]
fn fused_hits_hash_control_is_not_semantic_fast() {
    let fused = vec![frankensearch_core::types::FusedHit {
        doc_id: "hash-only".into(),
        rrf_score: 1.0,
        lexical_rank: None,
        semantic_rank: Some(0),
        hash_rank: None,
        semantic_index: Some(0),
        lexical_score: None,
        semantic_score: Some(0.9),
        hash_score: None,
        in_both_sources: false,
    }];
    let results = fused_hits_to_scored_results(&fused, &[], true, "fnv1a-256", 60.0);
    assert_eq!(results[0].source, ScoreSource::HashControl);
    assert!(
        matches!(
            results[0].explanation.as_ref().and_then(|explanation| {
                explanation.components.first().map(|component| &component.source)
            }),
            Some(ExplainedSource::HashControl { embedder, .. }) if embedder == "fnv1a-256"
        ),
        "hash explanation must not use SemanticFast: {:?}",
        results[0].explanation
    );
    let semantic = fused_hits_to_scored_results(&fused, &[], false, "minilm-l6-v2", 60.0);
    assert_eq!(semantic[0].source, ScoreSource::SemanticFast);

    let remapped = vec![frankensearch_core::types::FusedHit {
        doc_id: "hash-only".into(),
        rrf_score: 1.0,
        lexical_rank: None,
        semantic_rank: None,
        hash_rank: Some(0),
        semantic_index: Some(0),
        lexical_score: None,
        semantic_score: None,
        hash_score: Some(0.9),
        in_both_sources: false,
    }];
    let remapped_results = fused_hits_to_scored_results(&remapped, &[], true, "fnv1a-256", 60.0);
    assert_eq!(remapped_results[0].source, ScoreSource::HashControl);
    assert_eq!(remapped_results[0].fast_score, Some(0.9));
}

#[test]
fn fused_hits_lexical_only_source() {
    let fused = vec![frankensearch_core::types::FusedHit {
        doc_id: "lex-only".into(),
        rrf_score: 1.0,
        lexical_rank: Some(0),
        semantic_rank: None,
        hash_rank: None,
        semantic_index: None,
        lexical_score: Some(2.5),
        semantic_score: None,
        hash_score: None,
        in_both_sources: false,
    }];
    let results = fused_hits_to_scored_results(&fused, &[], false, "fast-test", 60.0);
    assert_eq!(results[0].source, ScoreSource::Lexical);
}

#[test]
fn fused_hits_graph_only_source_defaults_to_hybrid() {
    let fused = vec![frankensearch_core::types::FusedHit {
        doc_id: "graph-only".into(),
        rrf_score: 1.0,
        lexical_rank: None,
        semantic_rank: None,
        hash_rank: None,
        semantic_index: None,
        lexical_score: None,
        semantic_score: None,
        hash_score: None,
        in_both_sources: false,
    }];
    let results = fused_hits_to_scored_results(&fused, &[], false, "fast-test", 60.0);
    assert_eq!(results[0].source, ScoreSource::Hybrid);
}

#[test]
fn fused_hits_include_explanation_when_enabled() {
    let fused = vec![frankensearch_core::types::FusedHit {
        doc_id: "explained".into(),
        rrf_score: 1.25,
        lexical_rank: Some(1),
        semantic_rank: Some(3),
        hash_rank: None,
        semantic_index: Some(3),
        lexical_score: Some(2.0),
        semantic_score: Some(0.5),
        hash_score: None,
        in_both_sources: true,
    }];

    let results = fused_hits_to_scored_results(&fused, &[], true, "fast-test", 60.0);
    let explanation = results[0]
        .explanation
        .as_ref()
        .expect("explain=true should populate explanation");
    assert_eq!(explanation.phase, ExplanationPhase::Initial);
    assert_eq!(explanation.components.len(), 2);
    assert!(explanation.components[0].rrf_contribution > 0.0);
    assert!(explanation.components[1].rrf_contribution > 0.0);
}

#[test]
fn normalize_for_negation_match_lowercases() {
    assert_eq!(normalize_for_negation_match("HELLO"), "hello");
    assert_eq!(normalize_for_negation_match("MiXeD"), "mixed");
}

#[test]
fn normalize_for_negation_match_nfc_composing() {
    // e + combining acute = NFC café
    let decomposed = "caf\u{0065}\u{0301}";
    let result = normalize_for_negation_match(decomposed);
    assert_eq!(result, "café");
}

#[test]
fn term_is_word_like_alphanumeric_and_underscore() {
    assert!(term_is_word_like("hello"));
    assert!(term_is_word_like("hello_world"));
    assert!(term_is_word_like("abc123"));
    assert!(term_is_word_like("_"));
}

#[test]
fn term_is_word_like_false_for_special_chars() {
    assert!(!term_is_word_like("hello.world"));
    assert!(!term_is_word_like("src/main"));
    assert!(!term_is_word_like("a-b"));
    assert!(!term_is_word_like("foo bar"));
}

#[test]
fn term_is_word_like_empty_is_true() {
    // all chars satisfy predicate vacuously
    assert!(term_is_word_like(""));
}

#[test]
fn is_word_char_boundaries() {
    assert!(is_word_char('a'));
    assert!(is_word_char('Z'));
    assert!(is_word_char('5'));
    assert!(is_word_char('_'));
    assert!(!is_word_char(' '));
    assert!(!is_word_char('.'));
    assert!(!is_word_char('-'));
    assert!(!is_word_char('/'));
}

#[test]
fn contains_term_with_word_boundaries_exact_match() {
    assert!(contains_term_with_word_boundaries("hello world", "hello"));
    assert!(contains_term_with_word_boundaries("hello world", "world"));
}

#[test]
fn contains_term_with_word_boundaries_rejects_substring() {
    assert!(!contains_term_with_word_boundaries("theorem", "he"));
    assert!(!contains_term_with_word_boundaries("unhelpful", "help"));
}

#[test]
fn contains_term_with_word_boundaries_punctuation_boundary() {
    assert!(contains_term_with_word_boundaries("(hello) world", "hello"));
    assert!(contains_term_with_word_boundaries("say hello!", "hello"));
}

#[test]
fn contains_term_with_word_boundaries_start_and_end() {
    assert!(contains_term_with_word_boundaries("he", "he"));
    assert!(contains_term_with_word_boundaries("he said", "he"));
    assert!(contains_term_with_word_boundaries("said he", "he"));
}

#[test]
fn contains_negative_term_word_like_uses_boundaries() {
    assert!(!contains_negative_term("the theorem proves it", "he"));
    assert!(contains_negative_term("he went home", "he"));
}

#[test]
fn contains_negative_term_non_word_uses_substring() {
    assert!(contains_negative_term(
        "path=/workspace/src/main.rs",
        "src/main.rs"
    ));
}

#[test]
fn find_negative_match_empty_term_skipped() {
    let mut parsed = ParsedQuery::parse("query");
    parsed.negative_terms.push(String::new());
    let exclusions = to_exclusions(&parsed);
    let result = find_negative_match("any document text", &exclusions);
    assert!(result.is_none());
}

#[test]
fn find_negative_match_phrase_substring() {
    let parsed = ParsedQuery::parse(r#"query NOT "danger zone""#);
    let exclusions = to_exclusions(&parsed);
    let matched = find_negative_match("entering the danger zone now", &exclusions);
    assert_eq!(matched, Some("danger zone".to_owned()));
}

#[test]
fn find_negative_match_phrase_not_found() {
    let parsed = ParsedQuery::parse(r#"query NOT "exact phrase""#);
    let exclusions = to_exclusions(&parsed);
    let matched = find_negative_match("different text entirely", &exclusions);
    assert!(matched.is_none());
}

#[test]
fn should_exclude_document_returns_false_when_text_fn_returns_none() {
    let parsed = ParsedQuery::parse("query -unsafe");
    let exclusions = to_exclusions(&parsed);
    let result = should_exclude_document("doc-1", &exclusions, &|_| None, "test");
    assert!(!result);
}

#[test]
fn should_exclude_document_returns_true_when_text_matches_negation() {
    let parsed = ParsedQuery::parse("query -unsafe");
    let exclusions = to_exclusions(&parsed);
    let result = should_exclude_document(
        "doc-1",
        &exclusions,
        &|_| Some("unsafe code".to_owned()),
        "test",
    );
    assert!(result);
}

#[test]
fn should_exclude_document_returns_false_when_text_doesnt_match() {
    let parsed = ParsedQuery::parse("query -unsafe");
    let exclusions = to_exclusions(&parsed);
    let result = should_exclude_document(
        "doc-1",
        &exclusions,
        &|_| Some("safe code".to_owned()),
        "test",
    );
    assert!(!result);
}

#[test]
fn filter_scored_results_by_negations_empty_input() {
    let parsed = ParsedQuery::parse("query -foo");
    let exclusions = to_exclusions(&parsed);
    let results = filter_scored_results_by_negations(vec![], &exclusions, &|_| None, "test");
    assert!(results.is_empty());
}

#[test]
fn filter_vector_hits_by_negations_empty_input() {
    let parsed = ParsedQuery::parse("query -foo");
    let exclusions = to_exclusions(&parsed);
    let results = filter_vector_hits_by_negations(vec![], &exclusions, &|_| None, "test");
    assert!(results.is_empty());
}

#[test]
fn vector_exclusion_source_is_hash_control_for_hash_embedder() {
    let hash = NonSemanticEmbedder::new("fnv1a-test", 4);
    let semantic = StubEmbedder::new("fast", 4);
    assert_eq!(vector_exclusion_source(&hash), "hash_control");
    assert_eq!(vector_exclusion_source(&semantic), "semantic");
    assert!(!vector_exclusion_source(&hash).contains("semantic"));
}

#[test]
fn builder_with_lexical_sets_lexical() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
        .with_lexical(Arc::new(StubLexical));
    let debug = format!("{searcher:?}");
    assert!(debug.contains("has_lexical: true"));
}

#[test]
fn builder_nqc_dense_downweight_scales_dense_weight_when_enabled() {
    use frankensearch_core::ScoreSource;
    let lex_hit = |id: &str, score: f32| ScoredResult {
        doc_id: id.into(),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    };
    let hits = [lex_hit("a", 10.0), lex_hit("b", 1.0)]; // peaked → high NQC (cv ≈ 0.818)

    // Default OFF: fields neutral, effective weight == base.
    let off = TwoTierSearcher::new(
        build_test_index(4),
        Arc::new(StubEmbedder::new("f", 4)),
        TwoTierConfig::default(),
    );
    assert_eq!(off.nqc_downweight_beta.to_bits(), 0);
    assert!((off.effective_semantic_weight(&hits) - off.rrf_semantic_weight).abs() < 1e-12);

    // Enabled but not warmed up: an empty sketch is neutral with no score scan.
    let empty = TwoTierSearcher::new(
        build_test_index(4),
        Arc::new(StubEmbedder::new("f", 4)),
        TwoTierConfig::default(),
    )
    .with_rrf_weights(1.0, 1.3)
    .with_nqc_dense_downweight(0.5, 0.1, NqcDenseWeight::new());
    assert_eq!(
        empty.effective_semantic_weight(&hits).to_bits(),
        empty.rrf_semantic_weight.to_bits(),
    );

    // Enabled: the query NQC is above every sampled value (percentile 1.0), so
    // dense_weight(beta=0.5) = clip(1 − 0.5·1) = 0.5 → the dense weight is halved.
    let weight = NqcDenseWeight::from_sample(&[0.1, 0.2, 0.3]);
    let on = TwoTierSearcher::new(
        build_test_index(4),
        Arc::new(StubEmbedder::new("f", 4)),
        TwoTierConfig::default(),
    )
    .with_nqc_dense_downweight(0.5, 0.0, weight);
    assert!((on.nqc_downweight_beta - 0.5).abs() < 1e-6);
    assert_eq!(on.nqc_dense_weight.len(), 3);
    let eff = on.effective_semantic_weight(&hits);
    assert!(
        (eff - 0.5 * on.rrf_semantic_weight).abs() < 1e-6,
        "eff={eff}"
    );
    assert!(eff < on.rrf_semantic_weight);
}

#[test]
fn builder_nqc_dense_downweight_adaptive_cold_start_then_warms() {
    use frankensearch_core::ScoreSource;
    let lex_hit = |id: &str, score: f32| ScoredResult {
        doc_id: id.into(),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    };
    let searcher = TwoTierSearcher::new(
        build_test_index(4),
        Arc::new(StubEmbedder::new("f", 4)),
        TwoTierConfig::default(),
    )
    .with_rrf_weights(1.0, 1.0)
    .with_nqc_dense_downweight_adaptive(0.5, 0.05, 128, 8, 4);

    // Cold start: the online sketch is empty, so fusion is byte-identical to the base.
    let flat = [lex_hit("a", 5.0), lex_hit("b", 5.0), lex_hit("c", 5.0)];
    assert_eq!(
        searcher.effective_semantic_weight(&flat).to_bits(),
        searcher.rrf_semantic_weight.to_bits(),
        "cold-start online sketch is neutral"
    );
    // Warm the sketch, then a high-NQC query is down-weighted below the base weight.
    for _ in 0..12 {
        let _ = searcher.effective_semantic_weight(&flat);
    }
    let peaked = [lex_hit("a", 100.0), lex_hit("b", 1.0), lex_hit("c", 0.5)];
    let weight = searcher.effective_semantic_weight(&peaked);
    assert!(
        weight < searcher.rrf_semantic_weight,
        "high-NQC query down-weighted after the online sketch warms up, got {weight}"
    );
}

#[test]
fn builder_without_lexical() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
    let debug = format!("{searcher:?}");
    assert!(debug.contains("has_lexical: false"));
}

#[test]
fn builder_with_host_adapter_shows_in_debug() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let adapter = Arc::new(RecordingHostAdapter::new("test-project"));
    let searcher =
        TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_host_adapter(adapter);
    let debug = format!("{searcher:?}");
    assert!(debug.contains("has_host_adapter: true"));
}

#[test]
fn builder_with_reranker_shows_in_debug() {
    struct DummyReranker;
    impl Reranker for DummyReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            _docs: &'a [frankensearch_core::traits::RerankDocument],
        ) -> SearchFuture<'a, Vec<frankensearch_core::traits::RerankScore>> {
            Box::pin(async { Ok(vec![]) })
        }
        fn id(&self) -> &str {
            "dummy"
        }
        fn model_name(&self) -> &str {
            "dummy-reranker"
        }
    }
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
        .with_reranker(Arc::new(DummyReranker));
    let debug = format!("{searcher:?}");
    assert!(debug.contains("has_reranker: true"));
}

#[test]
fn should_run_quality_false_when_fast_only() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let quality = Arc::new(StubEmbedder::new("quality", 4));
    let config = TwoTierConfig {
        fast_only: true,
        ..TwoTierConfig::default()
    };
    let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);
    assert!(!searcher.should_run_quality());
}

#[test]
fn should_run_quality_false_when_no_quality_embedder() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
    assert!(!searcher.should_run_quality());
}

#[test]
fn should_run_quality_false_when_quality_index_is_unavailable() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let quality = Arc::new(StubEmbedder::new("quality", 4));
    let searcher =
        TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_quality_embedder(quality);
    assert!(!searcher.should_run_quality());
}

#[test]
fn should_run_quality_true_when_quality_embedder_and_not_fast_only() {
    let index = build_test_index_with_quality(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let quality = Arc::new(StubEmbedder::new("quality", 4));
    let searcher =
        TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_quality_embedder(quality);
    assert!(searcher.should_run_quality());
}

#[test]
fn should_run_quality_false_when_fast_embedder_is_hash_control() {
    let index = build_test_index_with_quality(4);
    let fast = Arc::new(NonSemanticEmbedder::new("fnv1a-test", 4));
    let quality = Arc::new(StubEmbedder::new("quality", 4));
    let searcher =
        TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_quality_embedder(quality);
    assert!(!searcher.should_run_quality());
}

#[test]
fn live_search_stream_initially_empty() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
    let health = searcher.live_search_stream_health();
    assert_eq!(health.emitted_total, 0);
    assert_eq!(health.buffered, 0);
    let drained = searcher.drain_live_search_stream(10);
    assert!(drained.is_empty());
}

#[test]
fn debug_shows_quality_embedder_id() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let quality = Arc::new(StubEmbedder::new("my-quality", 4));
    let searcher =
        TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_quality_embedder(quality);
    let debug = format!("{searcher:?}");
    assert!(debug.contains("my-quality"));
}

#[test]
fn debug_shows_none_quality_when_not_set() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
    let debug = format!("{searcher:?}");
    assert!(debug.contains("quality_embedder: None"));
}

#[test]
fn search_collect_with_text_returns_refined_when_quality_available() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = build_test_index_with_quality(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);

        let (results, metrics) = searcher
            .search_collect_with_text(&cx, "test", 5, |_| None)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(metrics.quality_embed_ms > 0.0);
        // When quality refinement succeeds, results should contain quality scores
        assert!(results.iter().any(|r| r.quality_score.is_some()));
    });
}

#[test]
fn with_runtime_metrics_collector_replaces_default() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let custom_collector = Arc::new(RuntimeMetricsCollector::default());
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
        .with_runtime_metrics_collector(custom_collector.clone());

    // Verify via Arc pointer equality
    assert!(Arc::ptr_eq(
        &searcher.runtime_metrics_collector,
        &custom_collector
    ));
}

#[test]
fn with_live_search_stream_emitter_replaces_default() {
    let index = build_test_index(4);
    let fast = Arc::new(StubEmbedder::new("fast", 4));
    let custom_emitter = Arc::new(LiveSearchStreamEmitter::default());
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
        .with_live_search_stream_emitter(custom_emitter.clone());

    assert!(Arc::ptr_eq(
        &searcher.live_search_stream_emitter,
        &custom_emitter
    ));
}

// ─── bd-3a7q tests end ───

#[test]
#[ignore = "performance probe"]
fn perf_probe_phase2_lookup_map_vs_linear_scan() {
    let doc_count = std::env::var("PHASE2_LOOKUP_PERF_DOCS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(30_000);
    let iterations = std::env::var("PHASE2_LOOKUP_PERF_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(80);

    let (fast_hits, quality_hits, blended) = build_phase2_lookup_fixture(doc_count);
    let fast_scores_by_doc: AHashMap<&str, f32> = fast_hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score))
        .collect();
    let quality_scores_by_doc: AHashMap<&str, f32> = quality_hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score))
        .collect();

    let mut linear_checksum = 0.0_f32;
    let linear_start = std::time::Instant::now();
    for _ in 0..iterations {
        linear_checksum += std::hint::black_box(phase2_lookup_checksum_linear(
            &blended,
            &fast_hits,
            &quality_hits,
        ));
    }
    let linear_ms = linear_start.elapsed().as_secs_f64() * 1000.0;

    let mut mapped_checksum = 0.0_f32;
    let mapped_start = std::time::Instant::now();
    for _ in 0..iterations {
        mapped_checksum += std::hint::black_box(phase2_lookup_checksum_mapped(
            &blended,
            &fast_scores_by_doc,
            &quality_scores_by_doc,
        ));
    }
    let mapped_ms = mapped_start.elapsed().as_secs_f64() * 1000.0;

    let checksum_diff = (linear_checksum - mapped_checksum).abs();
    assert!(
        checksum_diff <= 0.01_f32,
        "lookup checksum mismatch: linear={linear_checksum} mapped={mapped_checksum}"
    );
    println!(
        "PHASE2_LOOKUP_PERF map_ms={mapped_ms:.3} linear_ms={linear_ms:.3} speedup={:.3} doc_count={doc_count} iterations={iterations}",
        if mapped_ms > 0.0 {
            linear_ms / mapped_ms
        } else {
            f64::INFINITY
        }
    );
}
