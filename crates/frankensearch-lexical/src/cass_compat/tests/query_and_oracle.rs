use super::*;

/// The CASS oracle observation producer must expose exactly the evidence the
/// gauntlet comparator consumes, under the CASS profile contract
/// (`document_identity=source_id#msg_idx`, `snippets=disabled`).
#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_oracle_observation_carries_contract_identity_and_full_tie_evidence() {
    let (_directory, index) = cass_result_fixture();
    let filters = CassQueryFilters::default();

    // CASS bare terms are PREFIX-matched (the schema carries dedicated
    // `*_prefix` fields), so `alpha` retrieves `alpha active`,
    // `alpha deprecated` AND `alphabeta retained` — but not `xalpha
    // archived`, which only contains the term as an infix. That asymmetry
    // is the load-bearing part of this expectation: a subject adapter that
    // treats CASS terms as exact would pass a naive 2-hit assertion and
    // still be wrong.
    let observation = index
        .cass_oracle_observe_query("alpha", &filters, 10, 8)
        .expect("observe CASS term query");
    assert_eq!(
        observation
            .hits
            .iter()
            .map(|hit| hit.doc_id.as_str())
            .collect::<std::collections::BTreeSet<_>>(),
        ["local-a#0", "local-b#1", "local-d#5"]
            .into_iter()
            .collect(),
        "identity must be the contract's source_id#msg_idx rendering, \
         over the prefix-expanded match set",
    );
    assert_eq!(observation.total_count, 3);
    assert_eq!(observation.doc_count, 6, "live document count");
    assert!(
        observation.hits.iter().all(|hit| hit.snippet.is_none()),
        "the CASS activation profile is snippet-free",
    );
    assert!(
        observation
            .hits
            .iter()
            .enumerate()
            .all(|(index, hit)| hit.rank == index),
        "ranks must be the native fetched order",
    );

    let profiled_filters = CassQueryFilters {
        agents: vec!["claude".to_owned()],
        source_filter: CassSourceFilter::Local,
        ..CassQueryFilters::default()
    };
    let profile = index
        .cass_oracle_observe_query_profile(
            "c++ \"alpha active\" OR -deprecated",
            &profiled_filters,
            10,
            8,
        )
        .expect("bounded profile observation");
    assert_eq!(
        profile.sanitized_query,
        "c   \"alpha active\" OR -deprecated"
    );
    assert_eq!(
        profile.tokens,
        vec![
            CassQueryToken::Term("c++".to_owned()),
            CassQueryToken::Phrase("alpha active".to_owned()),
            CassQueryToken::Or,
            CassQueryToken::Not,
            CassQueryToken::Term("deprecated".to_owned()),
        ]
    );
    assert!(profile.has_boolean_operators);
    assert_eq!(profile.filters, profiled_filters);
    let profile_retrieval = match profile.outcome {
        CassOracleProfileOutcome::Success(retrieval) => retrieval,
        CassOracleProfileOutcome::Error(error) => {
            panic!("fallible CASS profile unexpectedly failed: {error:?}")
        }
    };
    assert_eq!(
        profile_retrieval,
        index
            .cass_oracle_observe_query(
                "c++ \"alpha active\" OR -deprecated",
                &profiled_filters,
                10,
                8,
            )
            .expect("legacy CASS observation for parity")
    );

    // An empty CASS query is a filter-only browse, NOT the default
    // profile's empty short-circuit: it must still match every document.
    let browse = index
        .cass_oracle_observe_query("", &filters, 10, 8)
        .expect("observe CASS filter-only browse");
    assert_eq!(
        browse.hits.len(),
        6,
        "empty CASS query must browse, not short-circuit: {:?}",
        browse
            .hits
            .iter()
            .map(|hit| &hit.doc_id)
            .collect::<Vec<_>>(),
    );
    assert_eq!(browse.total_count, 6);

    // Structured filters narrow the browse through the same producer.
    let filtered = index
        .cass_oracle_observe_query(
            "",
            &CassQueryFilters {
                source_filter: CassSourceFilter::SourceId("local-a".to_owned()),
                ..CassQueryFilters::default()
            },
            10,
            8,
        )
        .expect("observe CASS structured filter");
    assert_eq!(
        filtered
            .hits
            .iter()
            .map(|hit| hit.doc_id.as_str())
            .collect::<Vec<_>>(),
        vec!["local-a#0"],
    );

    // Truncating to k must not discard the cutoff tie evidence: the group is
    // computed over the expanded fetch, then `hits` is cut to k.
    let truncated = index
        .cass_oracle_observe_query("alpha", &filters, 1, 8)
        .expect("observe CASS truncated query");
    assert_eq!(truncated.hits.len(), 1, "hits truncate to the requested k");
    assert_eq!(
        truncated.total_count, 3,
        "total_count stays independent of k",
    );
    assert!(
        truncated.cutoff_tie_complete,
        "the whole match set was fetched, so the tie group is complete",
    );
    assert!(
        truncated
            .cutoff_tie_group
            .iter()
            .all(|hit| hit.score_bits == truncated.hits[0].score_bits),
        "the tie group holds exactly the cutoff-score rows",
    );
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_oracle_profile_rejects_hostile_bounds_before_clone_or_search() {
    let (_directory, index) = cass_result_fixture();
    let oversized_raw = "x".repeat(MAX_CASS_ORACLE_RAW_QUERY_BYTES + 1);
    assert!(matches!(
        index.cass_oracle_observe_query_profile(
            &oversized_raw,
            &CassQueryFilters::default(),
            10,
            8,
        ),
        Err(SearchError::InvalidConfig { field, .. })
            if field == "cass_oracle.raw_query_bytes"
    ));

    let too_many_tokens = "x ".repeat(MAX_CASS_ORACLE_TOKENS + 1);
    assert!(matches!(
        index.cass_oracle_observe_query_profile(
            &too_many_tokens,
            &CassQueryFilters::default(),
            10,
            8,
        ),
        Err(SearchError::InvalidConfig { field, .. })
            if field == "cass_oracle.token_count"
    ));

    let aggregate_filters = CassQueryFilters {
        agents: vec![
            "x".repeat(MAX_CASS_ORACLE_FILTER_VALUE_BYTES);
            MAX_CASS_ORACLE_FILTER_BYTES / MAX_CASS_ORACLE_FILTER_VALUE_BYTES + 1
        ],
        ..CassQueryFilters::default()
    };
    assert!(matches!(
        index.cass_oracle_observe_query_profile("x", &aggregate_filters, 10, 8),
        Err(SearchError::InvalidConfig { field, .. })
            if field == "cass_oracle.aggregate_filter_bytes"
    ));

    assert!(matches!(
        index.cass_oracle_observe_query_profile(
            "x",
            &CassQueryFilters::default(),
            MAX_CASS_ORACLE_FETCH_HITS,
            1,
        ),
        Err(SearchError::InvalidConfig { field, .. })
            if field == "cass_oracle.expanded_fetch_hits"
    ));
}

#[test]
fn cass_sanitize_query_preserves_wildcards_quotes_and_hyphens() {
    let out = cass_sanitize_query("c++ \"hello-world\" *config*");
    assert!(out.contains('"'));
    assert!(out.contains('*'));
    // Hyphens are now preserved so hyphenated identifiers stay intact.
    assert!(out.contains("hello-world"));
}

#[test]
fn cass_boolean_operator_detection_matches_parsed_tokens() {
    for (query, expected) in [
        ("", false),
        ("auth token cache", false),
        ("auth AND token", true),
        ("auth && token", true),
        ("auth OR token", true),
        ("auth || token", true),
        ("NOT legacy", true),
        ("-legacy", true),
        ("\"exact phrase\"", true),
        ("search 搜索", false),
    ] {
        let tokens = cass_parse_boolean_query(query);
        assert_eq!(
            cass_tokens_have_boolean_operators(&tokens),
            expected,
            "token-slice detection mismatch for {query:?}"
        );
        assert_eq!(
            cass_has_boolean_operators(query),
            expected,
            "public detection mismatch for {query:?}"
        );
    }
}

#[test]
fn cass_query_builder_single_parse_matches_operator_reparse() {
    let fields = fields();
    let filter_sets = [
        CassQueryFilters::default(),
        CassQueryFilters {
            agents: vec!["claude".to_string(), "codex".to_string()],
            workspaces: vec!["/data/projects/frankensearch".to_string()],
            created_from: Some(1_700_000_000_000),
            created_to: Some(1_800_000_000_000),
            source_filter: CassSourceFilter::Remote,
        },
    ];
    let queries = [
        "",
        "auth token cache",
        "auth AND token",
        "auth && token OR cache",
        "\"exact phrase\" OR cache",
        "NOT legacy",
        "-legacy identifier",
        "AND",
        "auth OR",
        "\"unterminated phrase",
        "search 搜索 NOT stale",
    ];

    for filters in &filter_sets {
        for query in queries {
            let legacy = cass_build_tantivy_query(query, filters, &fields);
            let single_parse = cass_build_tantivy_query_single_parse(query, filters, &fields);
            assert_eq!(
                format!("{legacy:?}"),
                format!("{single_parse:?}"),
                "query tree changed for {query:?} with filters {filters:?}"
            );
        }
    }
}

#[test]
fn cass_writer_config_scales_with_parallelism() {
    // Single-core: 1 thread, heap clamped up to the 256 MB floor.
    assert_eq!(
        cass_writer_config_for_parallelism(1),
        CassWriterConfig {
            num_threads: 1,
            heap_size_bytes: 256 * 1024 * 1024,
        }
    );
    // Dual-core: 2 threads × 128 MB = 256 MB hits the floor exactly.
    assert_eq!(
        cass_writer_config_for_parallelism(2),
        CassWriterConfig {
            num_threads: 2,
            heap_size_bytes: 256 * 1024 * 1024,
        }
    );
    // Quad-core: 4 threads × 128 MB = 512 MB (above the floor).
    assert_eq!(
        cass_writer_config_for_parallelism(4),
        CassWriterConfig {
            num_threads: 4,
            heap_size_bytes: 512 * 1024 * 1024,
        }
    );
    // Eight cores stay below the `CASS_MAX_WRITER_THREADS` cap.
    assert_eq!(
        cass_writer_config_for_parallelism(8),
        CassWriterConfig {
            num_threads: 8,
            heap_size_bytes: 1024 * 1024 * 1024,
        }
    );
    // At the new cap: 32 threads × 128 MB = 4 GB.
    assert_eq!(
        cass_writer_config_for_parallelism(32),
        CassWriterConfig {
            num_threads: 32,
            heap_size_bytes: 4 * 1024 * 1024 * 1024,
        }
    );
    // Above the cap (e.g. 64-core workstation): still clamped to 32 threads.
    assert_eq!(
        cass_writer_config_for_parallelism(64),
        CassWriterConfig {
            num_threads: 32,
            heap_size_bytes: 4 * 1024 * 1024 * 1024,
        }
    );
}

#[test]
fn cass_parallel_add_plan_enables_parallel_submission_for_large_batches() {
    assert_eq!(
        cass_parallel_add_plan(4_096, 8, 512, 2_048, 64),
        CassAddPlan {
            parallel_batches: true,
            batch_docs: 512,
            batch_count: 8,
        }
    );
    assert_eq!(
        cass_parallel_add_plan(200_000, 32, 512, 2_048, 64),
        CassAddPlan {
            parallel_batches: true,
            batch_docs: 3_125,
            batch_count: 64,
        }
    );
    assert_eq!(
        cass_parallel_add_plan(1_024, 8, 512, 2_048, 64),
        CassAddPlan {
            parallel_batches: false,
            batch_docs: 512,
            batch_count: 2,
        }
    );
}

#[test]
fn next_char_from_ascii_matches_decode() {
    // The ASCII fast-path must produce the byte-for-byte same (char, next_offset) sequence as
    // the UTF-8 decode on every input — ASCII, multi-byte, and mixed — since it feeds the
    // tokenizer whose token boundaries must not shift.
    for text in [
        "",
        "hello world",
        "bd-q3fy ID_42",
        "éclair café",
        "日本語 mixed 한국어 text",
        "a\u{300}b\u{1F600}c", // combining mark + emoji interleaved with ascii
        "\u{4E00}\u{4E01}xyz",
    ] {
        let mut off = 0usize;
        loop {
            let fast = next_char_from(text, off);
            let slow = next_char_from_slow(text, off);
            assert_eq!(fast, slow, "mismatch in {text:?} at offset {off}");
            match fast {
                Some((_, next)) => off = next,
                None => break,
            }
        }
    }
    // And the full-walk harnesses must agree.
    for text in ["hello world code_id-42", "日本語 mixed", ""] {
        assert_eq!(
            cass_char_walk_fast(text),
            cass_char_walk_slow(text),
            "{text:?}"
        );
    }
}

#[test]
fn cass_generate_edge_ngrams_emits_expected_prefixes() {
    assert_eq!(
        cass_generate_edge_ngrams("hello world"),
        "he hel hell hello wo wor worl world"
    );
    assert_eq!(
        cass_generate_edge_ngrams("éclair"),
        "éc écl écla éclai éclair"
    );
    assert_eq!(cass_generate_edge_ngrams("x"), "");
}

#[test]
fn cass_generate_edge_ngrams_caps_prefixes_at_twenty_chars() {
    assert_eq!(
        cass_generate_edge_ngrams("abcdefghijklmnopqrstuvwxy"),
        "ab abc abcd abcde abcdef abcdefg abcdefgh abcdefghi abcdefghij abcdefghijk abcdefghijkl abcdefghijklm abcdefghijklmn abcdefghijklmno abcdefghijklmnop abcdefghijklmnopq abcdefghijklmnopqr abcdefghijklmnopqrs abcdefghijklmnopqrst"
    );
}

#[test]
fn cass_build_preview_preserves_existing_behavior() {
    assert_eq!(cass_build_preview("", 0), "");
    assert_eq!(cass_build_preview("hello", 0), "…");
    assert_eq!(cass_build_preview("hello", 10), "hello");
    assert_eq!(cass_build_preview("hello world", 5), "hello…");
    assert_eq!(cass_build_preview("éclair", 3), "écl…");
}

/// PARITY GATE: the ASCII-fast `cass_generate_edge_ngrams` must equal the `char_indices` original
/// for ASCII words (fast path), non-ASCII words (fallback), and mixed text — including the
/// 20-char cap, short words (<3 chars → nothing), and words straddling the ASCII/Unicode split.
#[test]
fn cass_generate_edge_ngrams_matches_slow() {
    for text in [
        "",
        "x",
        "hi",
        "hello world",
        "bd-q3fy ID_42 snake_case",
        "abcdefghijklmnopqrstuvwxyz longwordovercap",
        "éclair café naïve",
        "日本語 hello 한국어 world",
        "aB3 café x9 déjà vu 12345678901234567890123",
    ] {
        assert_eq!(
            cass_generate_edge_ngrams(text),
            cass_generate_edge_ngrams_slow(text),
            "mismatch for {text:?}"
        );
    }
}

/// PARITY GATE: the O(1) `cass_prefix_source` must return the byte-identical prefix as the
/// forward-scan original for every `content/max_bytes` combination — including cuts that land
/// mid-multibyte-char (must round down to the boundary), at a boundary, at 0, and past the end.
#[test]
fn cass_prefix_source_matches_slow() {
    let ascii = "abcdefghij".repeat(50); // 500 bytes
    let uni = "aéb日cé".repeat(50); // multibyte, varied widths
    for content in ["", "x", "abc", ascii.as_str(), uni.as_str()] {
        for max_bytes in [0usize, 1, 2, 3, 4, 5, 7, 100, 499, 500, 501, 10_000] {
            let fast = cass_prefix_source(content, max_bytes);
            let slow = cass_prefix_source_slow(content, max_bytes);
            assert_eq!(
                fast,
                slow,
                "content.len()={} max_bytes={max_bytes}",
                content.len()
            );
        }
    }
}

/// PARITY GATE: the bulk-copy `cass_build_preview` must equal the char-by-char original for
/// every content length / `max_chars` / ASCII-vs-Unicode combination (boundary at, before, and
/// after the cut; empty; all-multibyte).
#[test]
fn cass_build_preview_matches_slow() {
    let ascii = "the quick brown fox jumps over the lazy dog ".repeat(20);
    let uni = "café 日本語 éclair 한국어 naïve ".repeat(20);
    for content in ["", "x", "hello world", ascii.as_str(), uni.as_str()] {
        for max_chars in [0usize, 1, 3, 4, 5, 10, 50, 400, 100_000] {
            assert_eq!(
                cass_build_preview(content, max_chars),
                cass_build_preview_slow(content, max_chars),
                "content.len()={} max_chars={max_chars}",
                content.len()
            );
        }
    }
}

#[test]
fn cass_prefix_fields_store_basic_without_freqs_or_positions() {
    let schema = cass_build_schema();

    for field_name in ["title_prefix", "content_prefix"] {
        let field = schema.get_field(field_name).unwrap();
        let field_entry = schema.get_field_entry(field);
        assert_eq!(
            field_entry.field_type().get_index_record_option(),
            Some(IndexRecordOption::Basic),
            "unexpected index record option for {field_name}"
        );
        let tantivy::schema::FieldType::Str(text_options) = field_entry.field_type() else {
            panic!("{field_name} should be a text field");
        };
        assert_eq!(
            text_options
                .get_indexing_options()
                .expect("prefix field indexing options")
                .tokenizer(),
            "prefix_normalize",
            "prefix field should use the cheaper generated-prefix analyzer"
        );
    }
}

#[test]
fn cass_prefix_analyzer_matches_full_analyzer_for_generated_prefix_terms() {
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
        let prefix_terms = cass_generate_edge_ngrams(text);
        let mut full = TextAnalyzer::builder(CassTokenizer::default())
            .filter(HyphenDecompose)
            .filter(CjkBigramDecompose)
            .filter(CassNormalizeAndLimit)
            .build();
        let mut prefix = TextAnalyzer::builder(CassTokenizer::default())
            .filter(CjkBigramDecompose)
            .filter(CassNormalizeAndLimit)
            .build();

        let mut full_stream = full.token_stream(&prefix_terms);
        let mut full_tokens = Vec::new();
        while full_stream.advance() {
            let token = full_stream.token();
            full_tokens.push((
                token.text.clone(),
                token.offset_from,
                token.offset_to,
                token.position,
                token.position_length,
            ));
        }

        let mut prefix_stream = prefix.token_stream(&prefix_terms);
        let mut prefix_tokens = Vec::new();
        while prefix_stream.advance() {
            let token = prefix_stream.token();
            prefix_tokens.push((
                token.text.clone(),
                token.offset_from,
                token.offset_to,
                token.position,
                token.position_length,
            ));
        }

        assert_eq!(
            prefix_tokens, full_tokens,
            "prefix analyzer changed generated edge-ngram tokens for {text:?}"
        );
    }
}

#[test]
fn cass_preview_field_is_stored_only() {
    let schema = cass_build_schema();
    let field = schema.get_field("preview").unwrap();
    let field_entry = schema.get_field_entry(field);

    assert!(field_entry.is_stored(), "preview should stay stored");
    assert_eq!(
        field_entry.field_type().get_index_record_option(),
        None,
        "preview should not be indexed"
    );
}

#[test]
fn cass_content_field_is_indexed_not_stored() {
    let schema = cass_build_schema();
    let field = schema.get_field("content").unwrap();
    let field_entry = schema.get_field_entry(field);

    assert!(field_entry.is_indexed(), "content must stay indexed");
    assert!(
        !field_entry.is_stored(),
        "content should hydrate from canonical storage instead of Tantivy stored fields"
    );
    assert_eq!(
        field_entry.field_type().get_index_record_option(),
        Some(IndexRecordOption::WithFreqsAndPositions),
        "content should keep full positional indexing"
    );
}

#[test]
fn cass_build_content_prefix_and_preview_matches_existing_helpers() {
    let samples = [
        "",
        "hello world",
        "éclair crème brûlée",
        "foo_bar baz-qux 12345",
        "你好 世界 from cass",
        &"alpha beta gamma ".repeat(64),
    ];

    for sample in samples {
        let (prefix, preview) = cass_build_content_prefix_and_preview(sample);
        assert_eq!(prefix, cass_generate_edge_ngrams(sample));
        assert_eq!(preview, cass_build_preview(sample, 400));
    }
}

#[test]
fn cass_build_content_prefix_caps_long_message_bodies() {
    let in_prefix_window = "alpha ".repeat(900);
    let after_prefix_window = "omega ".repeat(900);
    let content = format!("{in_prefix_window}{after_prefix_window}");

    let (prefix, preview) = cass_build_content_prefix_and_preview(&content);

    assert_eq!(preview, cass_build_preview(&content, 400));
    assert!(prefix.contains("alpha"));
    assert!(
        !prefix.contains("omega"),
        "content_prefix should not edge-ngram the whole message body"
    );
    assert_eq!(
        prefix,
        cass_generate_edge_ngrams(cass_prefix_source(&content, 4 * 1024))
    );
}

#[test]
fn cass_sanitize_query_splits_on_underscores() {
    // Underscores are NOT alphanumeric, so the sanitizer replaces them
    // with spaces — matching the tokenizer which uses [a-zA-Z0-9] (not \w).
    let out = cass_sanitize_query("hello_world");
    assert_eq!(out, "hello world");
}

#[test]
fn cass_build_query_empty_returns_allquery() {
    let f = fields();
    let q = cass_build_tantivy_query("", &CassQueryFilters::default(), &f);
    assert!(format!("{q:?}").to_ascii_lowercase().contains("allquery"));
}

#[test]
fn cass_build_query_applies_agent_filter() {
    let f = fields();
    let filters = CassQueryFilters {
        agents: vec!["claude".to_string(), "codex".to_string()],
        ..CassQueryFilters::default()
    };
    let q = cass_build_tantivy_query("auth", &filters, &f);
    let dbg = format!("{q:?}");
    assert!(
        dbg.contains("BooleanQuery"),
        "expected boolean query: {dbg}"
    );
}

/// cass#52 grammar (GH #56): NOT > AND (explicit or implied) > OR,
/// parentheses group, and malformed groups recover. Each mixed case
/// separates the standard reading from the legacy OR-binds-tighter one.
#[test]
fn cass_boolean_grammar_uses_standard_precedence_and_parentheses() {
    use CassQueryToken::{LParen, Not, Or, RParen, Term};
    let term = |text: &str| Term(text.to_owned());
    // `(` inside a word and `)` with no open group are term characters.
    assert_eq!(
        cass_parse_boolean_query("foo(bar) x)"),
        [term("foo(bar)"), term("x)")]
    );
    assert_eq!(
        cass_parse_boolean_query("-(a b)"),
        [Not, LParen, term("a"), term("b"), RParen]
    );
    assert_eq!(
        cass_parse_boolean_query("(a OR)"),
        [LParen, term("a"), Or, RParen]
    );

    let (_directory, index) = cass_result_fixture();
    let reader = index.reader().expect("open CASS Boolean reader");
    reader.reload().expect("reload CASS Boolean reader");
    let searcher = reader.searcher();
    let fields = index.fields();
    let matched = |raw_query: &str| {
        scored_cass_results(&searcher, &fields, raw_query, &CassQueryFilters::default())
            .into_keys()
            .collect::<Vec<_>>()
    };
    for (raw_query, expected) in [
        // deprecated ∪ (gamma ∩ active); legacy (deprecated ∪ gamma) ∩ active = {3}
        ("deprecated OR gamma AND active", vec![1, 3]),
        // (active ∩ beta) ∪ deprecated; legacy active ∩ (beta ∪ deprecated) = {2}
        ("active AND beta OR deprecated", vec![1, 2]),
        ("active beta OR deprecated", vec![1, 2]),
        ("active && beta || deprecated", vec![1, 2]),
        ("(deprecated OR gamma) AND active", vec![3]),
        ("active AND (beta OR deprecated)", vec![2]),
        ("-(beta OR gamma)", vec![0, 1, 4, 5]),
        ("alpha -(active OR deprecated)", vec![5]),
        ("deprecated OR -(alpha OR active)", vec![1, 4]),
        ("((gamma))", vec![3]),
        // Recovery: an unclosed group closes at the end; an empty group
        // and a dangling operator are skipped.
        ("(deprecated OR gamma", vec![1, 3]),
        ("gamma ()", vec![3]),
        ("gamma AND", vec![3]),
        ("OR gamma", vec![3]),
    ] {
        assert_eq!(matched(raw_query), expected, "{raw_query:?}");
    }

    // A group around a query both grammars read alike changes no score.
    let flat = scored_cass_results(
        &searcher,
        &fields,
        "alpha AND NOT deprecated",
        &CassQueryFilters::default(),
    );
    for grouped in ["(alpha AND NOT deprecated)", "((alpha) AND -(deprecated))"] {
        let actual = scored_cass_results(&searcher, &fields, grouped, &CassQueryFilters::default());
        assert_eq!(
            actual
                .iter()
                .map(|(msg_idx, score)| (*msg_idx, score.to_bits()))
                .collect::<Vec<_>>(),
            flat.iter()
                .map(|(msg_idx, score)| (*msg_idx, score.to_bits()))
                .collect::<Vec<_>>(),
            "{grouped:?}"
        );
    }
}

#[test]
fn cass_standalone_negation_matches_complement_and_is_score_neutral() {
    let (_directory, index) = cass_result_fixture();
    let reader = index.reader().expect("open CASS Boolean reader");
    reader.reload().expect("reload CASS Boolean reader");
    let searcher = reader.searcher();
    let fields = index.fields();

    let all_scores = scored_cass_results(&searcher, &fields, "", &CassQueryFilters::default());
    // Negation is parity-based (cass#52 grammar): two NOTs cancel.
    assert_eq!(
        scored_cass_results(
            &searcher,
            &fields,
            "NOT NOT deprecated",
            &CassQueryFilters::default()
        ),
        scored_cass_results(
            &searcher,
            &fields,
            "deprecated",
            &CassQueryFilters::default()
        ),
    );
    for raw_query in ["NOT deprecated", "-deprecated", "NOT NOT NOT deprecated"] {
        let actual =
            scored_cass_results(&searcher, &fields, raw_query, &CassQueryFilters::default());
        assert_eq!(
            actual.keys().copied().collect::<Vec<_>>(),
            vec![0, 2, 3, 4, 5],
            "standalone complement result set for {raw_query:?}"
        );
        for (msg_idx, score) in actual {
            assert_eq!(
                score.to_bits(),
                all_scores[&msg_idx].to_bits(),
                "MustNot changed the AllQuery score for {raw_query:?} document {msg_idx}"
            );
        }
    }

    let claude_filter = CassQueryFilters {
        agents: vec!["claude".to_owned()],
        ..CassQueryFilters::default()
    };
    let filtered_all_scores = scored_cass_results(&searcher, &fields, "", &claude_filter);
    for raw_query in ["NOT deprecated", "-deprecated"] {
        let actual = scored_cass_results(&searcher, &fields, raw_query, &claude_filter);
        assert_eq!(
            actual.keys().copied().collect::<Vec<_>>(),
            vec![0, 3],
            "filtered standalone complement result set for {raw_query:?}"
        );
        for (msg_idx, score) in actual {
            assert_eq!(
                score.to_bits(),
                filtered_all_scores[&msg_idx].to_bits(),
                "MustNot changed the filtered score for {raw_query:?} document {msg_idx}"
            );
        }
    }

    let positive_scores =
        scored_cass_results(&searcher, &fields, "alpha", &CassQueryFilters::default());
    let mixed = scored_cass_results(
        &searcher,
        &fields,
        "alpha AND NOT deprecated",
        &CassQueryFilters::default(),
    );
    assert_eq!(mixed.keys().copied().collect::<Vec<_>>(), vec![0, 5]);
    for (msg_idx, score) in mixed {
        assert_eq!(
            score.to_bits(),
            positive_scores[&msg_idx].to_bits(),
            "positive AND NOT gained a score-producing AllQuery anchor"
        );
    }

    let all_negative = scored_cass_results(
        &searcher,
        &fields,
        "-deprecated -beta",
        &CassQueryFilters::default(),
    );
    assert_eq!(
        all_negative.keys().copied().collect::<Vec<_>>(),
        vec![0, 3, 4, 5]
    );
    for (msg_idx, score) in all_negative {
        assert_eq!(score.to_bits(), all_scores[&msg_idx].to_bits());
    }
}

#[test]
fn cass_regex_globs_match_whole_terms_and_fail_closed() {
    assert_eq!(
        CassWildcardPattern::Suffix("legacy".to_owned())
            .to_regex()
            .as_deref(),
        Some(".*legacy")
    );
    assert_eq!(
        CassWildcardPattern::Complex("a*ha".to_owned())
            .to_regex()
            .as_deref(),
        Some("a.*ha")
    );
    assert_eq!(
        CassWildcardPattern::Complex("*urr*nt".to_owned())
            .to_regex()
            .as_deref(),
        Some(".*urr.*nt")
    );

    let (_directory, index) = cass_result_fixture();
    let reader = index.reader().expect("open CASS regex reader");
    reader.reload().expect("reload CASS regex reader");
    let searcher = reader.searcher();
    let fields = index.fields();
    let no_filters = CassQueryFilters::default();

    for (raw_query, expected) in [
        ("*legacy", vec![1]),
        ("*deprecated", vec![1]),
        ("*alpha", vec![0, 1, 4]),
        ("*alpha*", vec![0, 1, 4, 5]),
        ("a*ha", vec![0, 1]),
        ("*urr*nt", vec![0, 2, 3]),
        ("*pha*b*", vec![5]),
        ("*legacy *deprecated", vec![1]),
    ] {
        assert_eq!(
            cass_result_ids(&searcher, &fields, raw_query, &no_filters),
            expected,
            "regex-backed CASS result set for {raw_query:?}"
        );
    }

    let claude_filter = CassQueryFilters {
        agents: vec!["claude".to_owned()],
        ..CassQueryFilters::default()
    };
    assert_eq!(
        cass_result_ids(&searcher, &fields, "*active", &claude_filter),
        vec![0, 3]
    );

    fn force_regex_construction_failure(field: Field, _pattern: &str) -> SearchResult<RegexQuery> {
        cass_regex_query_cached(field, "(")
    }

    let raw_query = "NOT *legacy";
    let failed_build = cass_try_build_tantivy_query_from_tokens(
        cass_parse_boolean_query(raw_query),
        true,
        &CassQueryFilters::default(),
        &fields,
        force_regex_construction_failure,
    );
    assert!(
        failed_build.is_err(),
        "the forced invalid regex must reach the root builder"
    );
    let fail_closed = cass_fail_closed_tantivy_query(raw_query, failed_build);
    let hits = searcher
        .search(&*fail_closed, &tantivy::collector::DocSetCollector)
        .expect("search explicit match-none fallback");
    assert!(
        hits.is_empty(),
        "a negated regex construction failure must not widen to AllQuery"
    );

    #[cfg(feature = "tantivy-oracle")]
    {
        let profile = index
            .cass_oracle_observe_query_profile_with_regex_factory(
                raw_query,
                &CassQueryFilters::default(),
                10,
                8,
                force_regex_construction_failure,
            )
            .expect("bounded forced-failure profile");
        assert_eq!(
            profile.tokens,
            vec![
                CassQueryToken::Not,
                CassQueryToken::Term("*legacy".to_owned())
            ]
        );
        assert!(matches!(
            profile.outcome,
            CassOracleProfileOutcome::Error(_)
        ));
    }
}
