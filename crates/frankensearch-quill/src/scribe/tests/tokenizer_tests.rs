use super::*;

use crate::schema::Analyzer as AnalyzerKind;
#[cfg(feature = "tantivy-oracle")]
use frankensearch_lexical::tantivy_crate::tokenizer::TokenStream;
#[cfg(feature = "tantivy-oracle")]
use serde_json::Value;

#[test]
fn default_cjk_bigrams_split_unspaced_runs() {
    let mut analyzer = FrankensearchTokenizer::default();
    let mut terms = Vec::new();
    analyzer.analyze(
        AnalyzerKind::FrankensearchCjkBigrams,
        "Rust 日本語 搜索引擎",
        &mut |t| terms.push((t.text.clone(), t.position)),
    );
    assert_eq!(
        terms,
        vec![
            ("rust".into(), 0),
            ("日本".into(), 1),
            ("本語".into(), 2),
            ("搜索".into(), 3),
            ("索引".into(), 4),
            ("引擎".into(), 5)
        ]
    );
}

#[test]
fn borrowed_default_reuses_normalized_input_and_scratches_changed_tokens() {
    let text = "lowercase123 MIXED ÜBER";
    let input_start = text.as_ptr() as usize;
    let input_end = input_start + text.len();
    let mut analyzer = FrankensearchTokenizer::default();
    let mut observed = Vec::new();
    analyzer.analyze_borrowed(AnalyzerKind::FrankensearchDefault, text, &mut |token| {
        let token_start = token.text.as_ptr() as usize;
        observed.push((
            token.text.to_owned(),
            token.offset_from,
            token.offset_to,
            (input_start..input_end).contains(&token_start),
        ));
    });

    assert_eq!(
        observed,
        vec![
            ("lowercase123".to_owned(), 0, 12, true),
            ("mixed".to_owned(), 13, 18, false),
            ("über".to_owned(), 19, 24, false),
        ]
    );
}

fn boundary_mask_tokens(text: &str) -> Vec<AnalyzedToken> {
    let mut analyzer = BoundaryMaskTokenizer::default();
    let mut tokens = Vec::new();
    analyzer.analyze(AnalyzerKind::FrankensearchDefault, text, &mut |token| {
        tokens.push(token.clone());
    });
    tokens
}

#[cfg(feature = "tantivy-oracle")]
fn incumbent_tokens(text: &str) -> Vec<AnalyzedToken> {
    let mut analyzer = frankensearch_lexical::default_tokenizer_for_bench();
    let mut stream = analyzer.token_stream(text);
    let mut tokens = Vec::new();
    while stream.advance() {
        let token = stream.token();
        tokens.push(AnalyzedToken {
            text: token.text.clone(),
            position: u32::try_from(token.position)
                .expect("shipping tokenizer position fits Quill u32 contract"),
            offset_from: token.offset_from,
            offset_to: token.offset_to,
            position_length: token.position_length,
        });
    }
    tokens
}

#[cfg(feature = "tantivy-oracle")]
fn cass_tokens(analyzer_kind: AnalyzerKind, text: &str) -> Vec<AnalyzedToken> {
    let mut analyzer = CassAnalyzer::default();
    let mut tokens = Vec::new();
    analyzer.analyze(analyzer_kind, text, &mut |token| tokens.push(token.clone()));
    tokens
}

#[cfg(feature = "tantivy-oracle")]
fn incumbent_cass_tokens(analyzer_kind: AnalyzerKind, text: &str) -> Vec<AnalyzedToken> {
    let mut index = frankensearch_lexical::tantivy_crate::Index::create_in_ram(
        frankensearch_lexical::cass_compat::cass_build_schema(),
    );
    frankensearch_lexical::cass_compat::cass_ensure_tokenizer(&mut index);
    let tokenizer_name = match analyzer_kind {
        AnalyzerKind::CassHyphenNormalize => "hyphen_normalize",
        AnalyzerKind::CassPrefixNormalize => "prefix_normalize",
        AnalyzerKind::FrankensearchDefault => {
            unreachable!("CASS incumbent helper requires a CASS analyzer")
        }
    };
    let mut analyzer = index
        .tokenizers()
        .get(tokenizer_name)
        .expect("shipping CASS tokenizer is registered");
    let mut stream = analyzer.token_stream(text);
    let mut tokens = Vec::new();
    while stream.advance() {
        let token = stream.token();
        tokens.push(AnalyzedToken {
            text: token.text.clone(),
            position: u32::try_from(token.position)
                .expect("shipping CASS tokenizer position fits Quill u32 contract"),
            offset_from: token.offset_from,
            offset_to: token.offset_to,
            position_length: token.position_length,
        });
    }
    tokens
}

#[cfg(feature = "tantivy-oracle")]
fn fixture_input(case: &Value) -> String {
    if let Some(input) = case.get("input").and_then(Value::as_str) {
        return input.to_owned();
    }
    let generated = case
        .get("generated_input")
        .and_then(Value::as_object)
        .expect("analyzer fixture must contain input or generated_input");
    let repeat = generated
        .get("repeat")
        .and_then(Value::as_str)
        .expect("generated analyzer input must name repeat text");
    let count = ["count", "count_bytes", "count_chars"]
        .into_iter()
        .find_map(|key| generated.get(key).and_then(Value::as_u64))
        .expect("generated analyzer input must contain a repeat count");
    repeat.repeat(usize::try_from(count).expect("fixture repeat count fits usize"))
}

#[cfg(feature = "tantivy-oracle")]
fn expected_fixture_tokens(case: &Value) -> Option<Vec<AnalyzedToken>> {
    case.get("expected_tokens")?.as_array().map(|tokens| {
        tokens
            .iter()
            .map(|token| AnalyzedToken {
                text: token["text"]
                    .as_str()
                    .expect("fixture token text")
                    .to_owned(),
                position: token["position"]
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .expect("fixture token position fits u32"),
                offset_from: token["offset_from"]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .expect("fixture token start fits usize"),
                offset_to: token["offset_to"]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .expect("fixture token end fits usize"),
                position_length: token["position_length"]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .expect("fixture token position length fits usize"),
            })
            .collect()
    })
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn scalar_reference_executes_fixture_and_matches_shipping_incumbent() {
    let fixture: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
        .expect("language contract fixture is valid JSON");
    let cases = fixture["analyzer_cases"]
        .as_array()
        .expect("language contract analyzer_cases is an array");
    let mut executed = 0_u32;
    for case in cases {
        if case["analyzer"] != "frankensearch_default" {
            continue;
        }
        executed += 1;
        let case_id = case["id"].as_str().expect("analyzer case has an id");
        let input = fixture_input(case);
        let actual = analyzed_tokens(&input);
        assert_eq!(
            actual,
            incumbent_tokens(&input),
            "Quill token bytes diverged from the shipping incumbent for {case_id}"
        );
        if let Some(expected) = expected_fixture_tokens(case) {
            assert_eq!(actual, expected, "fixture golden diverged for {case_id}");
        }

        if let Some(admission) = case.get("token_admission").and_then(Value::as_str) {
            let mut analyzer = FrankensearchTokenizer::default();
            let mut admitted = Vec::new();
            let report = analyze_admitted(
                &mut analyzer,
                AnalyzerKind::FrankensearchDefault,
                &input,
                &mut |token| admitted.push(token.clone()),
            )
            .expect("default family supports the fixture analyzer");
            let kept = admission == "kept";
            assert!(kept || admission == "dropped", "{case_id}: {admission}");
            if kept {
                assert_eq!(report.admitted_tokens, 1, "{case_id}");
                assert_eq!(report.oversized_tokens, 0, "{case_id}");
                assert_eq!(admitted.len(), 1, "{case_id}");
            } else {
                assert_eq!(report.admitted_tokens, 0, "{case_id}");
                assert_eq!(report.oversized_tokens, 1, "{case_id}");
                assert!(admitted.is_empty(), "{case_id}");
            }
        }
    }
    assert_eq!(executed, 6, "all default-analyzer fixtures must execute");
}

fn scalar_reference_tokens(text: &str) -> Vec<AnalyzedToken> {
    let mut tokens = Vec::new();
    analyze_default_scalar_reference(text, &mut |token| tokens.push(token.clone()));
    tokens
}

#[test]
fn swar_ascii_alnum_mark_matches_scalar_for_every_ascii_byte_in_every_lane() {
    // The garbage-marker caveat only covers bytes >= 0x80; every ASCII lane
    // must classify exactly like `u8::is_ascii_alphanumeric`, with a
    // non-alnum filler in the other lanes to catch cross-lane borrow bleed.
    for lane in 0..SWAR_LANES {
        for byte in 0_u8..=0x7F {
            let mut lanes = [b'.'; SWAR_LANES];
            lanes[lane] = byte;
            let word = u64::from_le_bytes(lanes);
            let marked = (swar_ascii_alnum_mark(word) >> (lane * 8)) & 0x80 != 0;
            assert_eq!(
                marked,
                byte.is_ascii_alphanumeric(),
                "lane {lane} byte {byte:#04x} misclassified"
            );
        }
    }
}

#[test]
fn swar_ascii_alnum_mark_matches_scalar_on_random_ascii_words() {
    // Every lane of a fully random ASCII word must agree with the scalar
    // classifier — this is the cross-lane contamination guard.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..8192 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let word = state & 0x7F7F_7F7F_7F7F_7F7F; // force ASCII lanes
        let mark = swar_ascii_alnum_mark(word);
        for lane in 0..SWAR_LANES {
            let byte = u8::try_from((word >> (lane * 8)) & 0xFF).expect("masked to a byte");
            let marked = (mark >> (lane * 8)) & 0x80 != 0;
            assert_eq!(
                marked,
                byte.is_ascii_alphanumeric(),
                "byte {byte:#04x} in lane {lane} misclassified"
            );
        }
    }
}

/// Inputs engineered so tokens and non-ASCII bytes land at every offset
/// relative to the 8-byte SWAR window: token runs of length 6..=17 straddle
/// the first/second word boundary, and multi-byte scalar values appear at
/// lane 7 (window edge), lane 0 of the next window, and mid-window.
const LANE_EDGE_CASES: &[&str] = &[
    "",
    "a",
    "aa",
    "aaaaaaa",           // 7-byte token: entirely inside the tail path
    "aaaaaaaa",          // exactly one full window
    "aaaaaaaaa",         // one full window + tail
    "aaaaaaaaaaaaaaa",   // 15
    "aaaaaaaaaaaaaaaa",  // 16 (two full windows)
    "aaaaaaaaaaaaaaaaa", // 17
    "       a",          // 7 separators then alnum at lane 7
    "        a",         // 8 separators then alnum at lane 0 of the next window
    "aaaaaaa a",         // token ends exactly at the window edge
    "POL-358",
    "camelCase snake_case_name",
    "café",
    "aaaaaaaé",  // 7 ASCII alnum + a 2-byte scalar starting at lane 7
    "aaaaaaaaé", // 8 ASCII alnum + a 2-byte scalar at lane 0 of the next window
    "aaaaaaa😀", // 4-byte scalar (separator) straddling the window edge
    "aaaaaa中b", // 3-byte alnum scalar mid-window continues the token
    "Ω123",
    "搜索引擎",
    "a搜b索c",
    "———",
    "\u{0}\u{1}mix\u{7f}end",
];

#[test]
fn swar_default_matches_scalar_reference_on_lane_edge_cases() {
    for &case in LANE_EDGE_CASES {
        let shipping = analyzed_tokens(case);
        assert_eq!(shipping, scalar_reference_tokens(case));
        assert_eq!(
            boundary_mask_tokens(case),
            shipping,
            "boundary-mask candidate diverged from shipping SWAR for {case:?}"
        );
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn swar_default_matches_shipping_incumbent_on_lane_edge_cases() {
    for &case in LANE_EDGE_CASES {
        assert_eq!(
            analyzed_tokens(case),
            incumbent_tokens(case),
            "SWAR analyzer diverged from the shipping tokenizer for {case:?}"
        );
    }
}

#[test]
fn swar_default_matches_scalar_reference_on_random_corpus() {
    // Deterministic xorshift corpus mixing ASCII alnum/separators with
    // 2/3/4-byte scalar values (both alphanumeric and separator classes) at
    // random lengths so tokens straddle SWAR lane edges at every offset.
    const ALPHABET: &[char] = &[
        'a', 'z', 'A', 'Z', '0', '9', '_', ' ', '-', '.', '/', '\n', '\t', 'é', 'Ω', 'ß', 'İ',
        '中', '搜', '索', 'テ', '검', '—', '，', '😀', '𠀀',
    ];
    let alphabet_len = u64::try_from(ALPHABET.len()).expect("alphabet length fits u64");
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..4000 {
        let raw_len = usize::try_from(next() % 48).expect("length fits usize");
        let mut input = String::new();
        for _ in 0..raw_len {
            let idx = usize::try_from(next() % alphabet_len).expect("index fits usize");
            input.push(ALPHABET[idx]);
        }
        assert_eq!(
            boundary_mask_tokens(&input),
            analyzed_tokens(&input),
            "boundary-mask candidate diverged from shipping SWAR for {input:?}"
        );
        assert_eq!(
            analyzed_tokens(&input),
            scalar_reference_tokens(&input),
            "boundary-mask analyzer diverged from the scalar char-walk reference for {input:?}"
        );
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn native_cass_executes_fixture_and_matches_shipping_incumbent() {
    let fixture: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
        .expect("language contract fixture is valid JSON");
    let cases = fixture["analyzer_cases"]
        .as_array()
        .expect("language contract analyzer_cases is an array");
    let mut executed = 0_u32;
    for case in cases {
        let analyzer_kind = match case["analyzer"].as_str() {
            Some("hyphen_normalize") => AnalyzerKind::CassHyphenNormalize,
            Some("prefix_normalize") => AnalyzerKind::CassPrefixNormalize,
            Some("frankensearch_default") => continue,
            Some(other) => panic!("unhandled language-contract analyzer {other}"),
            None => panic!("analyzer fixture has no analyzer name"),
        };
        executed += 1;
        let case_id = case["id"].as_str().expect("analyzer case has an id");
        let input = fixture_input(case);
        let actual = cass_tokens(analyzer_kind, &input);
        assert_eq!(
            actual,
            incumbent_cass_tokens(analyzer_kind, &input),
            "native CASS stream diverged from the shipping incumbent for {case_id}"
        );
        if let Some(expected) = expected_fixture_tokens(case) {
            assert_eq!(actual, expected, "fixture golden diverged for {case_id}");
        }
        if let Some(expected_count) = case.get("expected_token_count").and_then(Value::as_u64) {
            assert_eq!(
                actual.len(),
                usize::try_from(expected_count).expect("fixture token count fits usize"),
                "{case_id}"
            );
        }
        if let Some(expected_bytes) = case.get("expected_token_bytes").and_then(Value::as_u64) {
            assert_eq!(
                actual.first().map(|token| token.text.len()),
                Some(usize::try_from(expected_bytes).expect("fixture byte count fits usize")),
                "{case_id}"
            );
        }
        if let Some(repeat) = case.get("expected_token_repeat").and_then(Value::as_str) {
            let expected_byte = repeat.as_bytes().first().copied();
            assert!(
                repeat.len() == 1
                    && actual.first().is_some_and(|token| {
                        token
                            .text
                            .as_bytes()
                            .iter()
                            .all(|byte| Some(*byte) == expected_byte)
                    }),
                "{case_id}"
            );
        }
    }
    assert_eq!(executed, 8, "all CASS analyzer fixtures must execute");
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn native_cass_matches_incumbent_at_token_boundaries_and_script_edges() {
    for analyzer_kind in [
        AnalyzerKind::CassHyphenNormalize,
        AnalyzerKind::CassPrefixNormalize,
    ] {
        for input in [
            "",
            "-abc abc- a--b a-b a1-b2",
            "ASCII_123/next",
            "搜",
            "搜索引擎",
            "かなカナ",
            "한글검색",
            "𠀀𠀁",
            "東京かな한글",
            "A搜索B",
            "éclair",
        ] {
            assert_eq!(
                cass_tokens(analyzer_kind, input),
                incumbent_cass_tokens(analyzer_kind, input),
                "analyzer={analyzer_kind:?} input={input:?}"
            );
        }
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn generated_prefix_pipeline_matches_incumbent_composition() {
    for input in [
        "",
        "Hello, happy tax payer!",
        "bd-q3fy foo_bar baz-qux",
        "abc-123 -- def",
        "Hello搜索World",
        "foo搜索-barあいう123",
        "café 𠀀 token",
        "multi---dash and trailing- hyphen",
    ] {
        let generated = cass_generate_edge_ngrams(input);
        let prefix_tokens = cass_tokens(AnalyzerKind::CassPrefixNormalize, &generated);
        assert_eq!(
            prefix_tokens,
            incumbent_cass_tokens(AnalyzerKind::CassPrefixNormalize, &generated),
            "generated prefix stream diverged for input={input:?}"
        );
        assert_eq!(
            prefix_tokens,
            cass_tokens(AnalyzerKind::CassHyphenNormalize, &generated),
            "generated prefixes must not require hyphen decomposition for input={input:?}"
        );
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn native_cass_matches_incumbent_across_shared_fixture_corpus() {
    let fixture: Value =
        serde_json::from_str(SHARED_CORPUS_FIXTURE).expect("shared corpus fixture is valid JSON");
    let documents = fixture["documents"]
        .as_array()
        .expect("shared corpus documents are an array");
    assert_eq!(
        documents.len(),
        120,
        "the complete pinned corpus must execute"
    );
    for document in documents {
        let doc_id = document["doc_id"].as_str().expect("fixture document id");
        for field_name in ["title", "content"] {
            let text = document[field_name].as_str().expect("fixture text field");
            assert_eq!(
                cass_tokens(AnalyzerKind::CassHyphenNormalize, text),
                incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, text),
                "doc={doc_id} field={field_name}"
            );

            let generated = cass_generate_edge_ngrams(text);
            assert_eq!(
                cass_tokens(AnalyzerKind::CassPrefixNormalize, &generated),
                incumbent_cass_tokens(AnalyzerKind::CassPrefixNormalize, &generated),
                "generated prefix doc={doc_id} field={field_name}"
            );
            assert_eq!(
                cass_build_preview(text, 400),
                frankensearch_lexical::cass_compat::cass_build_preview_slow(text, 400),
                "preview doc={doc_id} field={field_name}"
            );
        }
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn seeded_cass_differential_matches_incumbent_and_named_slow_oracles() {
    const SEED: u64 = 0xE102_CA55_5C12_1BE5;
    const ATOMS: [&str; 15] = [
        "A", "z9", "-", "--", " ", "_", "/", "搜", "索", "かな", "한글", "𠀀", "é", "🙂", "123",
    ];
    let mut rng = DeterministicRng(SEED);
    for case in 0..128_u32 {
        let mut input = String::new();
        for _ in 0..rng.choose(48) {
            input.push_str(ATOMS[rng.choose(ATOMS.len())]);
        }

        let mut offset = 0;
        let mut native_walk = 0_u64;
        while let Some((ch, next)) = tokenizer_next_char(&input, offset) {
            native_walk = native_walk.wrapping_add(u64::from(ch as u32));
            offset = next;
        }
        assert_eq!(
            native_walk,
            frankensearch_lexical::cass_compat::cass_char_walk_slow(&input),
            "seed={SEED:#x} case={case} input={input:?}"
        );

        let native_cjk = if input.is_empty() || !input.chars().all(is_cass_cjk) {
            None
        } else {
            let chars: Vec<char> = input.chars().collect();
            (chars.len() >= 2).then_some(chars)
        };
        assert_eq!(
            native_cjk,
            frankensearch_lexical::cass_compat::cass_cjk_collect_slow(&input),
            "seed={SEED:#x} case={case} input={input:?}"
        );

        for analyzer_kind in [
            AnalyzerKind::CassHyphenNormalize,
            AnalyzerKind::CassPrefixNormalize,
        ] {
            assert_eq!(
                cass_tokens(analyzer_kind, &input),
                incumbent_cass_tokens(analyzer_kind, &input),
                "seed={SEED:#x} case={case} analyzer={analyzer_kind:?} input={input:?}"
            );
        }
        assert_eq!(
            cass_generate_edge_ngrams(&input),
            frankensearch_lexical::cass_compat::cass_generate_edge_ngrams_slow(&input),
            "seed={SEED:#x} case={case} input={input:?}"
        );
        let preview_bound = rng.choose(32);
        assert_eq!(
            cass_build_preview(&input, preview_bound),
            frankensearch_lexical::cass_compat::cass_build_preview_slow(&input, preview_bound),
            "seed={SEED:#x} case={case} max_chars={preview_bound} input={input:?}"
        );
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_cjk_range_endpoints_and_neighbors_match_slow_oracle() {
    const RANGES: [(u32, u32); 9] = [
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x3040, 0x309F),
        (0x30A0, 0x30FF),
        (0xAC00, 0xD7AF),
        (0x3100, 0x312F),
        (0x3300, 0x33FF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF),
    ];
    for (start, end) in RANGES {
        assert!(is_cass_cjk(
            char::from_u32(start).expect("valid range start")
        ));
        assert!(is_cass_cjk(char::from_u32(end).expect("valid range end")));
        for codepoint in [start - 1, start, end, end + 1] {
            let Some(ch) = char::from_u32(codepoint) else {
                continue;
            };
            let input = ch.to_string().repeat(2);
            assert_eq!(
                is_cass_cjk(ch),
                frankensearch_lexical::cass_compat::cass_cjk_collect_slow(&input).is_some(),
                "codepoint=U+{codepoint:04X}"
            );
            assert_eq!(
                cass_tokens(AnalyzerKind::CassHyphenNormalize, &input),
                incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &input),
                "codepoint=U+{codepoint:04X}"
            );
        }
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_filter_order_preserves_parts_bigrams_and_position_gaps() {
    let oversized_compound = format!("{}-{}", "A".repeat(130), "B".repeat(130));
    let compound_tokens = cass_tokens(AnalyzerKind::CassHyphenNormalize, &oversized_compound);
    assert_eq!(
        compound_tokens,
        incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &oversized_compound)
    );
    assert_eq!(compound_tokens.len(), 2);
    assert!(compound_tokens.iter().all(|token| token.position == 0));
    assert_eq!(compound_tokens[0].text, "a".repeat(130));
    assert_eq!(compound_tokens[1].text, "b".repeat(130));

    let long_cjk = "搜".repeat(100);
    let cjk_tokens = cass_tokens(AnalyzerKind::CassHyphenNormalize, &long_cjk);
    assert_eq!(
        cjk_tokens,
        incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &long_cjk)
    );
    assert_eq!(cjk_tokens.len(), 99);
    assert!(cjk_tokens.iter().all(|token| token.text == "搜搜"));

    let exact_limit = "X".repeat(CASS_MAX_TOKEN_BYTES);
    let limit_tokens = cass_tokens(AnalyzerKind::CassHyphenNormalize, &exact_limit);
    assert_eq!(
        limit_tokens,
        incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &exact_limit)
    );
    assert_eq!(limit_tokens.len(), 1);
    assert_eq!(limit_tokens[0].text, "x".repeat(CASS_MAX_TOKEN_BYTES));

    let dropped_then_kept = format!("{} ok", "X".repeat(CASS_MAX_TOKEN_BYTES + 1));
    let gap_tokens = cass_tokens(AnalyzerKind::CassHyphenNormalize, &dropped_then_kept);
    assert_eq!(
        gap_tokens,
        incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &dropped_then_kept)
    );
    assert_eq!(gap_tokens.len(), 1);
    assert_eq!(gap_tokens[0].text, "ok");
    assert_eq!(gap_tokens[0].position, 1);
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_helpers_execute_contract_fixture_and_match_slow_oracles() {
    let fixture: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
        .expect("language contract fixture is valid JSON");
    let cases = fixture["helper_cases"]
        .as_array()
        .expect("language contract helper_cases is an array");
    let mut executed = 0_u32;
    for case in cases {
        let case_id = case["id"].as_str().expect("helper case has an id");
        let input = fixture_input(case);
        match case["helper"].as_str() {
            Some("cass_generate_edge_ngrams") => {
                executed += 1;
                let actual = cass_generate_edge_ngrams(&input);
                assert_eq!(
                    actual,
                    frankensearch_lexical::cass_compat::cass_generate_edge_ngrams_slow(&input),
                    "{case_id}"
                );
                if let Some(expected) = case.get("expected").and_then(Value::as_str) {
                    assert_eq!(actual, expected, "{case_id}");
                }
                if let Some(expected_last) =
                    case.get("last_expected_prefix").and_then(Value::as_str)
                {
                    assert_eq!(actual.split_whitespace().next_back(), Some(expected_last));
                }
                if let Some(expected_count) =
                    case.get("expected_prefix_count").and_then(Value::as_u64)
                {
                    assert_eq!(
                        actual.split_whitespace().count(),
                        usize::try_from(expected_count).expect("fixture prefix count fits usize"),
                        "{case_id}"
                    );
                }
            }
            Some("cass_build_preview") => {
                executed += 1;
                let max_chars = case["max_chars"]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .expect("preview fixture bound fits usize");
                let actual = cass_build_preview(&input, max_chars);
                assert_eq!(
                    actual,
                    frankensearch_lexical::cass_compat::cass_build_preview_slow(&input, max_chars),
                    "{case_id}"
                );
                assert_eq!(
                    actual,
                    case["expected"].as_str().expect("preview expected value"),
                    "{case_id}"
                );
            }
            Some("truncate_query") => {}
            Some(other) => panic!("unhandled language-contract helper {other} in {case_id}"),
            None => panic!("helper fixture {case_id} has no helper name"),
        }
    }
    assert_eq!(executed, 5, "all native CASS helper fixtures must execute");
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_helper_boundaries_match_slow_oracles() {
    let inputs = [
        String::new(),
        "a".to_owned(),
        "é".to_owned(),
        "é".repeat(20),
        "é".repeat(21),
        "hello,世界/한글".to_owned(),
    ];
    for input in &inputs {
        assert_eq!(
            cass_generate_edge_ngrams(input),
            frankensearch_lexical::cass_compat::cass_generate_edge_ngrams_slow(input),
            "input={input:?}"
        );
        for max_chars in [0, 1, 2, 19, 20, 21, 64] {
            assert_eq!(
                cass_build_preview(input, max_chars),
                frankensearch_lexical::cass_compat::cass_build_preview_slow(input, max_chars),
                "input={input:?} max_chars={max_chars}"
            );
        }
    }
}

#[test]
fn cass_analyzer_reset_retains_reusable_scratch() {
    let mut analyzer = CassAnalyzer::default();
    analyzer.analyze(
        AnalyzerKind::CassHyphenNormalize,
        &"A".repeat(CASS_MAX_TOKEN_BYTES),
        &mut |_| {},
    );
    let retained = analyzer.bytes_reserved();
    assert!(retained >= CASS_MAX_TOKEN_BYTES);
    analyzer.reset();
    assert_eq!(analyzer.bytes_reserved(), retained);
    analyzer.analyze(AnalyzerKind::CassPrefixNormalize, "short", &mut |_| {});
    assert_eq!(analyzer.bytes_reserved(), retained);
}

// ── E1.8: Scribe unit/property tests (tokenizer/CASS parity, accumulator
//          invariants, radix-flush determinism). E2E ingest round-trip and
//          the LabRuntime cancellation leak-oracle live in
//          `crates/frankensearch-quill/tests/scribe_e2e.rs`.

/// First token-stream divergence as `(source byte offset, lane index mod 32)`
/// so a failing lane-edge sweep case pinpoints the straddling boundary.
#[cfg(feature = "tantivy-oracle")]
fn first_divergent_offset(a: &[AnalyzedToken], b: &[AnalyzedToken]) -> Option<(usize, usize)> {
    for (ta, tb) in a.iter().zip(b.iter()) {
        if ta != tb {
            return Some((ta.offset_from, ta.offset_from % 32));
        }
    }
    if a.len() == b.len() {
        return None;
    }
    let shared = a.len().min(b.len());
    let off = a
        .get(shared)
        .or_else(|| b.get(shared))
        .map_or(0, |token| token.offset_from);
    Some((off, off % 32))
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn e18_tokenizer_lane_edge_sweep_1_to_129_bytes_matches_scalar_and_incumbent() {
    // Sweep a separator (hence a token start AND end) across every byte
    // offset for inputs sized 1..=129 bytes, so a token boundary lands at
    // every 8/16/32-byte SWAR lane edge. Parity vs the scalar char-walk
    // oracle on every case; vs the shipping incumbent on the lane-edge
    // subset to bound tokenizer-construction cost.
    for len in 1..=129usize {
        for sep_pos in 0..len {
            let input: String = (0..len)
                .map(|i| if i == sep_pos { ' ' } else { 'a' })
                .collect();
            let swar = analyzed_tokens(&input);
            let scalar = scalar_reference_tokens(&input);
            if let Some((off, lane)) = first_divergent_offset(&swar, &scalar) {
                panic!(
                    "len={len} sep_pos={sep_pos}: SWAR diverged from the scalar reference at byte offset {off} (lane {lane}) for {input:?}"
                );
            }
            if sep_pos % 8 == 0 {
                let incumbent = incumbent_tokens(&input);
                if let Some((off, lane)) = first_divergent_offset(&swar, &incumbent) {
                    panic!(
                        "len={len} sep_pos={sep_pos}: SWAR diverged from the shipping incumbent at byte offset {off} (lane {lane}) for {input:?}"
                    );
                }
            }
        }
    }

    // Multi-byte scalars (2/3/4 bytes) straddling every lane edge.
    for mb in ['é', '中', '𠀀'] {
        for pad in 0..40usize {
            let mut input = "a".repeat(pad);
            input.push(mb);
            input.push_str("bb");
            let swar = analyzed_tokens(&input);
            let scalar = scalar_reference_tokens(&input);
            if let Some((off, lane)) = first_divergent_offset(&swar, &scalar) {
                panic!(
                    "mb={mb:?} pad={pad}: SWAR diverged from the scalar reference at byte offset {off} (lane {lane}) for {input:?}"
                );
            }
        }
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn e18_cass_family_parity_including_cjk_extension_b() {
    // CJK Extension-B (U+20000..=U+2A6DF, 4-byte UTF-8) is inside
    // `is_cass_cjk`, so the native CASS families must bigram-decompose it
    // exactly like the shipping incumbent. Mix Ext-B with ASCII, hyphens,
    // and BMP CJK so hyphen-normalize and prefix-normalize both exercise it.
    let cases = [
        "\u{20000}\u{20001}\u{2A6DF}",
        "alpha-\u{20000}\u{20001}",
        "\u{20000}beta\u{4E00}\u{20002}",
        "POL-358 \u{20000}\u{20001} gamma",
        "\u{20000}-\u{20001}-code",
        "mix\u{20000}東京\u{2A700}tail",
    ];
    for kind in [
        AnalyzerKind::CassHyphenNormalize,
        AnalyzerKind::CassPrefixNormalize,
    ] {
        for case in cases {
            assert_eq!(
                cass_tokens(kind, case),
                incumbent_cass_tokens(kind, case),
                "native CASS {kind:?} diverged from the shipping incumbent for {case:?}"
            );
        }
    }
}
