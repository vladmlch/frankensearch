use tantivy::tokenizer::RegexTokenizer;

use super::*;

#[test]
fn is_cjk_detects_chinese_characters() {
    assert!(is_cjk('\u{4E00}')); // CJK Unified start
    assert!(is_cjk('\u{641C}')); // 搜
    assert!(is_cjk('\u{7D22}')); // 索
    assert!(!is_cjk('a'));
    assert!(!is_cjk('1'));
}

#[test]
fn is_cjk_detects_japanese_hiragana_katakana() {
    assert!(is_cjk('\u{3042}')); // あ (hiragana)
    assert!(is_cjk('\u{30A2}')); // ア (katakana)
}

#[test]
fn is_cjk_detects_korean_hangul() {
    assert!(is_cjk('\u{AC00}')); // 가 (hangul start)
    assert!(is_cjk('\u{D558}')); // 하
}

#[test]
fn cass_cjk_collect_fast_matches_slow() {
    // Byte-identity proof for the one-pass fusion: fast == slow on every
    // reject path and every decompose path.
    let cases = [
        "",               // empty -> None
        "a",              // single ASCII -> None
        "hello world",    // ASCII with space -> None (space non-CJK)
        "搜",             // single CJK -> None (unigram, no bigrams)
        "搜索",           // 2 CJK -> Some
        "搜索引擎",       // 4 CJK -> Some
        "検索とうきょう", // mixed CJK scripts -> Some
        "한국어",         // hangul -> Some
        "搜a",            // CJK then ASCII -> None (not all-CJK)
        "a搜",            // ASCII lead -> None
        "搜 索",          // CJK, space, CJK -> None (space breaks all-CJK)
        "𠀀𠀁",           // Extension B (4-byte) CJK -> Some
    ];
    for c in cases {
        assert_eq!(
            cass_cjk_collect_fast(c),
            cass_cjk_collect_slow(c),
            "fast/slow parity for {c:?}"
        );
    }
}

#[test]
fn cjk_bigram_decompose_produces_bigrams() {
    // Build a minimal tokenizer pipeline with just CJK bigrams.
    let regex_tok = RegexTokenizer::new(
        r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u3100-\u312F\u3300-\u33FF\uF900-\uFAFF]+"
    ).unwrap();
    let mut analyzer = TextAnalyzer::builder(regex_tok)
        .filter(CjkBigramDecompose)
        .build();

    // Chinese: 搜索引擎 -> bigrams: 搜索, 索引, 引擎
    let mut stream = analyzer.token_stream("搜索引擎");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert_eq!(tokens, vec!["搜索", "索引", "引擎"]);
}

#[test]
fn cjk_bigram_single_char_emits_unigram() {
    let regex_tok = RegexTokenizer::new(r"[\u4E00-\u9FFF]+").unwrap();
    let mut analyzer = TextAnalyzer::builder(regex_tok)
        .filter(CjkBigramDecompose)
        .build();

    let mut stream = analyzer.token_stream("搜");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert_eq!(tokens, vec!["搜"]);
}

#[test]
fn cjk_bigram_passes_through_ascii() {
    // ASCII tokens should pass through unchanged.
    let regex_tok =
        RegexTokenizer::new(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|[\u4E00-\u9FFF]+").unwrap();
    let mut analyzer = TextAnalyzer::builder(regex_tok)
        .filter(CjkBigramDecompose)
        .build();

    let mut stream = analyzer.token_stream("hello");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert_eq!(tokens, vec!["hello"]);
}

#[test]
fn cjk_mixed_text_tokenizes_both() {
    // Mixed CJK and ASCII text should produce tokens for both.
    let regex_tok = RegexTokenizer::new(
        r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]+",
    )
    .unwrap();
    let mut analyzer = TextAnalyzer::builder(regex_tok)
        .filter(CjkBigramDecompose)
        .filter(CassNormalizeAndLimit)
        .build();

    // "Hello搜索World" -> hello, 搜索, world
    let mut stream = analyzer.token_stream("Hello搜索World");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert_eq!(tokens, vec!["hello", "搜索", "world"]);
}

#[test]
fn cass_tokenizer_matches_legacy_regex_boundaries() {
    let regex_pattern = r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u3100-\u312F\u3300-\u33FF\uF900-\uFAFF\U00020000-\U0002A6DF]+";
    let mut regex = RegexTokenizer::new(regex_pattern).unwrap();
    let mut custom = CassTokenizer::default();

    for text in [
        "",
        "Hello, happy tax payer!",
        "bd-q3fy foo_bar baz-qux",
        "abc-123 -- def",
        "Hello搜索World",
        "foo搜索-barあいう123",
        "caf\u{00E9} 𠀀 token",
        "multi---dash and trailing- hyphen",
    ] {
        let mut regex_stream = regex.token_stream(text);
        let mut regex_tokens = Vec::new();
        while regex_stream.advance() {
            let token = regex_stream.token();
            regex_tokens.push((
                token.text.clone(),
                token.offset_from,
                token.offset_to,
                token.position,
            ));
        }

        let mut custom_stream = custom.token_stream(text);
        let mut custom_tokens = Vec::new();
        while custom_stream.advance() {
            let token = custom_stream.token();
            custom_tokens.push((
                token.text.clone(),
                token.offset_from,
                token.offset_to,
                token.position,
            ));
        }

        assert_eq!(custom_tokens, regex_tokens, "token mismatch for {text:?}");
    }
}

#[test]
fn cass_tokenizer_keeps_extension_b_cjk() {
    let mut tokenizer = CassTokenizer::default();
    let mut stream = tokenizer.token_stream("𠀀搜索 test");
    let mut tokens = Vec::new();
    while stream.advance() {
        let token = stream.token();
        tokens.push(token.text.clone());
    }

    assert_eq!(tokens, vec!["𠀀搜索".to_string(), "test".to_string()]);
}

#[test]
fn cass_normalize_and_limit_matches_legacy_pipeline() {
    for text in [
        "",
        "Hello, happy tax payer!",
        "BD-Q3FY foo_bar BAZ-QUX",
        "abc-123 -- def",
        "Hello搜索World",
        "foo搜索-barあいう123",
        "caf\u{00E9} 𠀀 token",
        "multi---dash and trailing- hyphen",
        &format!("{} keep", "X".repeat(255)),
        &format!("{} keep", "X".repeat(257)),
        &format!("{} keep", "X".repeat(300)),
    ] {
        let mut legacy = TextAnalyzer::builder(CassTokenizer::default())
            .filter(HyphenDecompose)
            .filter(CjkBigramDecompose)
            .filter(tantivy::tokenizer::LowerCaser)
            .filter(tantivy::tokenizer::RemoveLongFilter::limit(256))
            .build();
        let mut optimized = TextAnalyzer::builder(CassTokenizer::default())
            .filter(HyphenDecompose)
            .filter(CjkBigramDecompose)
            .filter(CassNormalizeAndLimit)
            .build();

        let mut legacy_stream = legacy.token_stream(text);
        let mut legacy_tokens = Vec::new();
        while legacy_stream.advance() {
            let token = legacy_stream.token();
            legacy_tokens.push((
                token.text.clone(),
                token.offset_from,
                token.offset_to,
                token.position,
                token.position_length,
            ));
        }

        let mut optimized_stream = optimized.token_stream(text);
        let mut optimized_tokens = Vec::new();
        while optimized_stream.advance() {
            let token = optimized_stream.token();
            optimized_tokens.push((
                token.text.clone(),
                token.offset_from,
                token.offset_to,
                token.position,
                token.position_length,
            ));
        }

        assert_eq!(
            optimized_tokens, legacy_tokens,
            "normalized analyzer mismatch for {text:?}"
        );
    }
}

#[test]
fn cass_normalize_and_limit_pins_inclusive_shipping_boundary() {
    let shipping_tokens = |text: &str| {
        let mut analyzer = TextAnalyzer::builder(CassTokenizer::default())
            .filter(HyphenDecompose)
            .filter(CjkBigramDecompose)
            .filter(CassNormalizeAndLimit)
            .build();
        let mut stream = analyzer.token_stream(text);
        let mut tokens = Vec::new();
        while stream.advance() {
            let token = stream.token();
            tokens.push((
                token.text.clone(),
                token.offset_from,
                token.offset_to,
                token.position,
                token.position_length,
            ));
        }
        tokens
    };
    let tantivy_tokens = |text: &str| {
        let mut analyzer = TextAnalyzer::builder(CassTokenizer::default())
            .filter(HyphenDecompose)
            .filter(CjkBigramDecompose)
            .filter(tantivy::tokenizer::LowerCaser)
            .filter(tantivy::tokenizer::RemoveLongFilter::limit(256))
            .build();
        let mut stream = analyzer.token_stream(text);
        let mut tokens = Vec::new();
        while stream.advance() {
            let token = stream.token();
            tokens.push((
                token.text.clone(),
                token.offset_from,
                token.offset_to,
                token.position,
                token.position_length,
            ));
        }
        tokens
    };

    for length in [255, 256, 257] {
        let input = format!("{} keep", "X".repeat(length));
        let kept = vec![
            ("x".repeat(length), 0, length, 0, 1),
            ("keep".to_owned(), length + 1, length + 5, 1, 1),
        ];
        let dropped = vec![("keep".to_owned(), length + 1, length + 5, 1, 1)];
        let shipping_actual = shipping_tokens(&input);
        let tantivy_actual = tantivy_tokens(&input);

        assert_eq!(
            shipping_actual.as_slice(),
            if length <= 256 {
                kept.as_slice()
            } else {
                dropped.as_slice()
            },
            "shipping CASS boundary at {length} bytes"
        );
        assert_eq!(
            tantivy_actual.as_slice(),
            if length < 256 {
                kept.as_slice()
            } else {
                dropped.as_slice()
            },
            "Tantivy RemoveLongFilter boundary at {length} bytes"
        );
    }
}

#[test]
fn cjk_sanitize_query_preserves_chinese() {
    let out = cass_sanitize_query("搜索引擎 test");
    assert!(out.contains("搜索引擎"));
    assert!(out.contains("test"));
}

#[test]
fn cjk_index_and_search_roundtrip() {
    // End-to-end: index a Chinese document, search for a substring.
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

    let doc = CassDocument {
        agent: "claude".to_string(),
        workspace: None,
        workspace_original: None,
        source_path: "/tmp/test".to_string(),
        msg_idx: 0,
        created_at: Some(1_700_000_000_000),
        title: Some("搜索引擎测试".to_string()),
        content: "这是一个搜索引擎的测试".to_string(),
        conversation_id: None,
        source_id: "local".to_string(),
        origin_kind: "local".to_string(),
        origin_host: None,
    };
    idx.add_cass_documents(&[doc]).expect("add");
    idx.commit().expect("commit");

    let reader = idx.reader().expect("reader");
    reader.reload().expect("reload");
    let searcher = reader.searcher();

    let fields = idx.fields();
    let query = cass_build_tantivy_query("搜索", &CassQueryFilters::default(), &fields);

    let results = searcher
        .search(
            &query,
            &tantivy::collector::TopDocs::with_limit(10).order_by_score(),
        )
        .expect("search");
    assert!(
        !results.is_empty(),
        "CJK search for '搜索' should find the indexed Chinese document"
    );
}

#[test]
fn japanese_index_and_search_roundtrip() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

    let doc = CassDocument {
        agent: "claude".to_string(),
        workspace: None,
        workspace_original: None,
        source_path: "/tmp/test".to_string(),
        msg_idx: 0,
        created_at: Some(1_700_000_000_000),
        title: Some("テスト".to_string()),
        content: "これはテストです".to_string(),
        conversation_id: None,
        source_id: "local".to_string(),
        origin_kind: "local".to_string(),
        origin_host: None,
    };
    idx.add_cass_documents(&[doc]).expect("add");
    idx.commit().expect("commit");

    let reader = idx.reader().expect("reader");
    reader.reload().expect("reload");
    let searcher = reader.searcher();

    let fields = idx.fields();
    let query = cass_build_tantivy_query("テスト", &CassQueryFilters::default(), &fields);

    let results = searcher
        .search(
            &query,
            &tantivy::collector::TopDocs::with_limit(10).order_by_score(),
        )
        .expect("search");
    assert!(
        !results.is_empty(),
        "CJK search for 'テスト' should find the indexed Japanese document"
    );
}

#[test]
fn korean_index_and_search_roundtrip() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

    let doc = CassDocument {
        agent: "claude".to_string(),
        workspace: None,
        workspace_original: None,
        source_path: "/tmp/test".to_string(),
        msg_idx: 0,
        created_at: Some(1_700_000_000_000),
        title: Some("검색엔진".to_string()),
        content: "한국어 검색엔진 테스트".to_string(),
        conversation_id: None,
        source_id: "local".to_string(),
        origin_kind: "local".to_string(),
        origin_host: None,
    };
    idx.add_cass_documents(&[doc]).expect("add");
    idx.commit().expect("commit");

    let reader = idx.reader().expect("reader");
    reader.reload().expect("reload");
    let searcher = reader.searcher();

    let fields = idx.fields();
    let query = cass_build_tantivy_query("검색", &CassQueryFilters::default(), &fields);

    let results = searcher
        .search(
            &query,
            &tantivy::collector::TopDocs::with_limit(10).order_by_score(),
        )
        .expect("search");
    assert!(
        !results.is_empty(),
        "CJK search for '검색' should find the indexed Korean document"
    );
}

#[test]
fn force_merge_bounded_consolidates_segments() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

    // Suppress auto-merge during setup so each commit leaves its own
    // segment; otherwise the default LogMergePolicy could collapse them and
    // we'd never exercise the bounded multi-pass path.
    idx.writer.set_merge_policy(Box::new(NoMergePolicy));

    let make_doc = |i: u64| CassDocument {
        agent: "claude".to_string(),
        workspace: None,
        workspace_original: None,
        source_path: format!("/tmp/test/{i}"),
        msg_idx: i,
        created_at: Some(1_700_000_000_000_i64.saturating_add(i.cast_signed())),
        title: Some(format!("doc {i}")),
        content: format!("bounded merge segment number {i}"),
        source_id: "local".to_string(),
        origin_kind: "local".to_string(),
        origin_host: None,
        conversation_id: None,
    };

    // Five separate commits => five segments (auto-merge suppressed).
    for i in 0..5u64 {
        idx.add_cass_documents(&[make_doc(i)]).expect("add");
        idx.commit().expect("commit");
    }
    let before = idx.index.searchable_segment_ids().expect("segments").len();
    assert!(before > 1, "expected multiple segments, got {before}");

    // Bounded merge in batches of 2 must consolidate to a single segment
    // and restore the default merge policy on the way out.
    let merges = idx.force_merge_bounded(2).expect("bounded merge");
    assert!(merges >= 1, "expected at least one bounded merge pass");

    let after = idx.index.searchable_segment_ids().expect("segments").len();
    assert!(
        after <= 1,
        "bounded merge should consolidate to <=1 segment, got {after}"
    );

    // All five documents survive the consolidation.
    let reader = idx.reader().expect("reader");
    reader.reload().expect("reload");
    let searcher = reader.searcher();
    let fields = idx.fields();
    let query = cass_build_tantivy_query("bounded", &CassQueryFilters::default(), &fields);
    let results = searcher
        .search(
            &query,
            &tantivy::collector::TopDocs::with_limit(10).order_by_score(),
        )
        .expect("search");
    assert_eq!(
        results.len(),
        5,
        "all documents should remain searchable after bounded merge"
    );
}

/// 100-segment regression: exercises the bounded-merge progression at a
/// scale closer to the failure-mode described in #26 (many small segments,
/// auto-merge suppressed). With `batch_size = 8` the inner loop must do
/// at least `ceil(100 / 8) = 13` initial passes; in practice each pass
/// collapses the leading 8 segments to 1, so the total iteration count is
/// bounded by `O(N / (B - 1))`. We assert the operation completes and
/// leaves at most one segment regardless of exact iteration count.
#[test]
fn force_merge_bounded_progressively_collapses_many_segments() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");
    idx.writer.set_merge_policy(Box::new(NoMergePolicy));

    // 100 single-doc commits => 100 segments.
    for i in 0..100u64 {
        idx.add_cass_documents(&[CassDocument {
            agent: "claude".to_string(),
            workspace: None,
            workspace_original: None,
            source_path: format!("/tmp/test100/{i}"),
            msg_idx: i,
            created_at: Some(1_700_000_000_000_i64.saturating_add(i.cast_signed())),
            title: Some(format!("doc {i}")),
            content: format!("scale test segment {i}"),
            source_id: "local".to_string(),
            origin_kind: "local".to_string(),
            origin_host: None,
            conversation_id: None,
        }])
        .expect("add");
        idx.commit().expect("commit");
    }
    let before = idx.index.searchable_segment_ids().expect("segments").len();
    assert!(
        before >= 50,
        "expected many segments before bounded merge, got {before}"
    );

    let merges = idx.force_merge_bounded(8).expect("bounded merge");
    // ceil(100/8) = 13 is the first-pass lower bound; subsequent passes
    // collapse the residual. We just assert progress was made and the
    // final segment count is the target.
    assert!(
        merges >= 13,
        "expected at least 13 bounded passes for 100 segments / batch 8, got {merges}"
    );

    let after = idx.index.searchable_segment_ids().expect("segments").len();
    assert!(
        after <= 1,
        "bounded merge should collapse 100 segments to <=1, got {after}"
    );
}

/// Regression: the previously-configured merge policy must be *preserved*
/// across `force_merge_bounded`, not silently reset to
/// `LogMergePolicy::default()`. This is the central correctness gain over
/// the obvious "install `NoMergePolicy`, then install `LogMergePolicy` on
/// exit" pattern: a caller that has tuned the policy (e.g. via
/// [`CassTantivyIndex::configure_bulk_load_merge_policy`]) keeps that
/// tuning after recovery instead of having it silently dropped.
#[test]
fn force_merge_bounded_restores_prior_merge_policy() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut idx = CassTantivyIndex::open_or_create(dir.path()).expect("create");

    // Add a couple of segments so the bounded loop actually runs.
    idx.writer.set_merge_policy(Box::new(NoMergePolicy));
    for i in 0..3u64 {
        idx.add_cass_documents(&[CassDocument {
            agent: "claude".to_string(),
            workspace: None,
            workspace_original: None,
            source_path: format!("/tmp/restore/{i}"),
            msg_idx: i,
            created_at: Some(1_700_000_000_000_i64.saturating_add(i.cast_signed())),
            title: Some(format!("restore {i}")),
            content: format!("merge policy restore probe {i}"),
            source_id: "local".to_string(),
            origin_kind: "local".to_string(),
            origin_host: None,
            conversation_id: None,
        }])
        .expect("add");
        idx.commit().expect("commit");
    }
    // Reinstall the bulk-load policy as the "prior" snapshot that
    // force_merge_bounded should preserve. The bulk-load policy is a
    // LogMergePolicy with `min_num_segments` set to
    // `CASS_BULK_LOAD_MIN_SEGMENTS_PER_MERGE` (= 256), which is the
    // distinctive fingerprint we look for after the bounded merge.
    idx.configure_bulk_load_merge_policy();
    let prior_dbg = format!("{:?}", idx.writer.get_merge_policy());
    assert!(
        prior_dbg.contains("min_num_segments: 256"),
        "sanity: prior policy should be the bulk-load LogMergePolicy, got {prior_dbg}"
    );

    let _ = idx.force_merge_bounded(2).expect("bounded merge");

    let after_dbg = format!("{:?}", idx.writer.get_merge_policy());
    // The restored policy is wrapped in `ArcMergePolicy(...)` because
    // `set_merge_policy` requires a `Box` while `get_merge_policy` only
    // exposes an `Arc`; the wrapper delegates `compute_merge_candidates`
    // to the inner policy, so steady-state auto-merge behavior is
    // preserved. The wrapped fingerprint must still carry the
    // distinctive bulk-load `min_num_segments: 256` — i.e. the caller's
    // tuning was NOT silently replaced with `LogMergePolicy::default()`.
    assert!(
        after_dbg.contains("min_num_segments: 256"),
        "force_merge_bounded dropped the bulk-load tuning: after = {after_dbg}"
    );
    assert!(
        !after_dbg.contains("NoMergePolicy"),
        "force_merge_bounded left NoMergePolicy in place: after = {after_dbg}"
    );
}
