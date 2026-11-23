use std::ops::Bound;

#[cfg(feature = "tantivy-oracle")]
use frankensearch_core::traits::LexicalWrite;
#[cfg(feature = "tantivy-oracle")]
use frankensearch_core::types::IndexableDocument;
#[cfg(feature = "tantivy-oracle")]
use frankensearch_lexical::TantivyIndex;
use serde_json::Value;

use super::*;

#[test]
fn escape_unknown_field_colons_keeps_identifiers_and_urls_literal() {
    for (query, expected) in [
        ("HashMap::new", r"HashMap\:\:new"),
        ("std::io::Result", r"std\:\:io\:\:Result"),
        ("https://sqlite.org", r"https\://sqlite.org"),
        ("TODO: flaky", r"TODO\: flaky"),
        ("lang:rust", r"lang\:rust"),
        // A real field keeps its prefix, also after a group or negation.
        ("title:rust", "title:rust"),
        ("(title:rust) -content:beta", "(title:rust) -content:beta"),
        // `title::x` is not a field prefix; a quoted phrase is literal.
        ("title::x", r"title\:\:x"),
        ("\"std::io\" error", "\"std::io\" error"),
        // An escape the caller already wrote is kept, so it is idempotent.
        (r"a\:b", r"a\:b"),
        ("plain words", "plain words"),
    ] {
        let escaped = escape_unknown_field_colons(query, crate::DEFAULT_SCHEMA);
        assert_eq!(escaped, expected, "{query}");
        assert_eq!(
            escape_unknown_field_colons(&escaped, crate::DEFAULT_SCHEMA),
            expected,
            "{query} twice"
        );
    }
    // Field names come from the schema at hand.
    assert_eq!(
        escape_unknown_field_colons("agent:claude HashMap::new", crate::CASS_SEMANTIC_SCHEMA),
        r"agent:claude HashMap\:\:new"
    );
    assert_eq!(
        escape_unknown_field_colons("agent:claude", crate::DEFAULT_SCHEMA),
        r"agent\:claude"
    );
    // The parser drops the raw identifier and keeps the escaped one.
    let parser = DefaultQueryParser::new(crate::DEFAULT_SCHEMA).expect("parser");
    assert!(parser.parse_lenient("HashMap::new").query.is_empty());
    let escaped = escape_unknown_field_colons("HashMap::new", crate::DEFAULT_SCHEMA);
    assert!(!parser.parse_lenient(&escaped).query.is_empty());
}

#[test]
fn default_parser_matches_pinned_tantivy_fixture_trees() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../tests/fixtures/quill_language_contract.json"
    ))
    .expect("language fixture parses");
    let cases = fixture["parse_tree_cases"]
        .as_array()
        .expect("parse tree case array");
    let expected = cases
        .iter()
        .filter(|case| case["parser"] == "default_lenient")
        .count();
    let mut executed = 0;
    for case in cases {
        if case["parser"] != "default_lenient" {
            continue;
        }
        let id = case["id"].as_str().expect("fixture id");
        let input = case["input"].as_str().expect("fixture input");
        let parsed = parser().parse(input);
        assert_eq!(
            fixture_json(&parsed.query, None),
            case["expected_ast"],
            "fixture {id}"
        );
        if let Some(expected) = case.get("expected_diagnostic").and_then(Value::as_str) {
            assert!(
                parsed
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message.contains(expected)),
                "fixture {id} expected diagnostic {expected:?}, got {:?}",
                parsed.diagnostics
            );
        } else {
            assert!(
                parsed.diagnostics.is_empty(),
                "fixture {id} unexpectedly diagnosed {:?}",
                parsed.diagnostics
            );
        }
        executed += 1;
    }
    assert_eq!(executed, expected, "every pinned default parser case ran");
}

#[test]
fn harvested_query_corpus_never_fails_or_panics() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../../../tests/fixtures/queries.json"))
            .expect("query corpus parses");
    for row in fixture.as_array().expect("query rows") {
        let query = row["query"].as_str().expect("query string");
        let _ = parser().parse(query);
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn parser_result_sets_match_the_pinned_tantivy_adapter() {
    const DOCS: [EvalDoc; 10] = [
        EvalDoc {
            id: "doc-a",
            content: "rust ownership error handling",
            title: "Rust",
        },
        EvalDoc {
            id: "doc-b",
            content: "rust deprecated",
            title: "Legacy Rust",
        },
        EvalDoc {
            id: "doc-c",
            content: "deprecated ownership",
            title: "Old",
        },
        EvalDoc {
            id: "doc-d",
            content: "and",
            title: "Conjunction",
        },
        EvalDoc {
            id: "doc-e",
            content: "or",
            title: "Disjunction",
        },
        EvalDoc {
            id: "doc-f",
            content: "transformers text embeddings",
            title: "Semantic Search",
        },
        EvalDoc {
            id: "doc-g",
            content: "search index new",
            title: "SearchIndex",
        },
        EvalDoc {
            id: "doc-z",
            content: "unrelated",
            title: "Nothing",
        },
        EvalDoc {
            id: "foo bar",
            content: "space sentinel",
            title: "Escaped",
        },
        EvalDoc {
            id: "foo:bar",
            content: "colon sentinel",
            title: "Escaped",
        },
    ];

    let fixture: Value =
        serde_json::from_str(include_str!("../../../../../tests/fixtures/queries.json"))
            .expect("query corpus parses");
    let mut queries = fixture
        .as_array()
        .expect("query rows")
        .iter()
        .map(|row| row["query"].as_str().expect("query string").to_owned())
        .collect::<Vec<_>>();
    queries.extend(
        [
            "rust OR -deprecated",
            "-rust OR deprecated",
            "-deprecated rust OR ownership",
            "rust -deprecated OR ownership",
            "rust OR -deprecated AND ownership",
            "rust OR (-deprecated)",
            "rust AND NOT deprecated",
            "rust AND -deprecated",
            "rust NOT deprecated",
            "rust NOT",
            "NOT\tdeprecated",
            "rust NOT\tdeprecated",
            "NOT NOT rust",
            "NOT NOT",
            "rust AND",
            "rust OR",
            "rust AND\townership",
            "rust\nAND ownership",
            "rust OR OR ownership",
            "rust AND AND ownership",
            "(OR rust)",
            "rust ) ownership",
            "(rust OR ownership)^2",
            "rust^2^3",
            "id:doc-a^2^3",
            "\"rust\"^2^3",
            "\"rust ownership\"~deprecated",
            "\"rust ownership\"~4294967296",
            "title:(rust OR \"error handling\")",
            "title : ownership",
            "title: ownership",
            "+title:Rust",
            "-title:Rust",
            "foo*:bar",
            r"id:foo\ bar",
            r"id:foo\:bar",
            r"id:foo\q",
            r"SearchIndex\:\:new",
            "/rust/",
            "rust OR /ownership/",
            "-/rust/ rust",
            "rust AND /ownership/ deprecated",
            "id:[doc-a TO doc-z}",
            "id:[doc-a doc-c]",
            "id:[doc-a TO",
            "id:[doc-a",
            "id: IN [doc-a doc-c]",
            "id:([doc-a TO doc-c} OR IN [doc-z])",
        ]
        .into_iter()
        .map(str::to_owned),
    );

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let index = TantivyIndex::in_memory().expect("create Tantivy oracle");
        let documents = DOCS
            .iter()
            .map(|doc| IndexableDocument::new(doc.id, doc.content).with_title(doc.title.to_owned()))
            .collect::<Vec<_>>();
        index
            .index_documents(&cx, &documents)
            .await
            .expect("index oracle documents");
        index.commit(&cx).await.expect("commit oracle documents");

        let mut mismatches = Vec::new();
        for raw in queries {
            let mut oracle = index
                .search_doc_ids(&cx, &raw, DOCS.len())
                .expect("execute Tantivy oracle")
                .into_iter()
                .map(|hit| hit.doc_id.to_string())
                .collect::<Vec<_>>();
            oracle.sort_unstable();
            let parsed = parser().parse(&raw);
            let mut quill = DOCS
                .iter()
                .copied()
                .filter(|doc| ast_matches(&parsed.query, *doc))
                .map(|doc| doc.id.to_owned())
                .collect::<Vec<_>>();
            quill.sort_unstable();
            if quill != oracle {
                mismatches.push(format!(
                    "{raw:?}: quill={quill:?}, oracle={oracle:?}, ast={:?}",
                    parsed.query
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "result-set differential mismatches:\n{}",
            mismatches.join("\n")
        );
    });
}

#[test]
fn classification_preserves_incumbent_surface() {
    assert_eq!(classify_query(""), QueryExplanation::Empty);
    assert_eq!(classify_query("   "), QueryExplanation::Empty);
    assert_eq!(classify_query("rust"), QueryExplanation::Simple);
    assert_eq!(
        classify_query("  authentication  "),
        QueryExplanation::Simple
    );
    assert_eq!(
        classify_query("\"error handling\""),
        QueryExplanation::Phrase
    );
    assert_eq!(classify_query("'single quotes'"), QueryExplanation::Phrase);
    assert_eq!(classify_query("rust async"), QueryExplanation::Boolean);
    assert_eq!(QueryExplanation::Empty.to_string(), "empty");
    assert_eq!(QueryExplanation::Simple.to_string(), "simple");
    assert_eq!(QueryExplanation::Phrase.to_string(), "phrase");
    assert_eq!(QueryExplanation::Boolean.to_string(), "boolean");

    let json =
        serde_json::to_string(&QueryExplanation::Phrase).expect("query explanation serializes");
    let decoded: QueryExplanation =
        serde_json::from_str(&json).expect("query explanation deserializes");
    assert_eq!(decoded, QueryExplanation::Phrase);
}

#[test]
fn empty_whitespace_and_lone_signs_lower_to_match_none() {
    for query in ["", "   ", "-", "+"] {
        assert!(
            parser().parse(query).query.is_empty(),
            "{query:?} must not become a match-all query"
        );
    }
}

#[test]
fn truncation_is_character_counted_and_boundary_safe() {
    let short = "hello world";
    assert_eq!(truncate_query(short), short);

    let at_limit = "a".repeat(MAX_QUERY_LENGTH);
    assert_eq!(truncate_query(&at_limit), at_limit);

    let over = "a".repeat(MAX_QUERY_LENGTH + 3);
    assert_eq!(truncate_query(&over).len(), MAX_QUERY_LENGTH);

    let multibyte = "é".repeat(MAX_QUERY_LENGTH + 3);
    let truncated = truncate_query(&multibyte);
    assert!(truncated.is_char_boundary(truncated.len()));
    assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
    assert_eq!(truncated.len(), MAX_QUERY_LENGTH * 'é'.len_utf8());

    let byte_long_but_char_short = "é".repeat(MAX_QUERY_LENGTH / 2 + 1);
    assert_eq!(
        truncate_query(&byte_long_but_char_short),
        byte_long_but_char_short
    );
}

#[test]
fn parse_reports_truncation_once_and_uses_the_prefix() {
    let query = format!("needle {}", "padding ".repeat(2_000));
    let parsed = parser().parse(&query);
    assert!(parsed.was_truncated);
    assert_eq!(
        parsed
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.kind == QueryDiagnosticKind::Truncated)
            .count(),
        1
    );
}

#[test]
fn lenient_corners_drop_or_recover_without_error() {
    let cases = [
        "-", "+", "NOT", "AND", "OR", "rust AND", "OR rust", "@user", "#hashtag", "foo:bar", "a+b",
        "hello!", "((rust)", "rust ))",
    ];
    for case in cases {
        let _ = parser().parse(case);
    }
}

#[test]
fn standalone_and_leading_operator_recovery_is_pinned() {
    for (raw, normalized) in [("AND", "and"), ("OR", "or"), ("NOT", "not")] {
        let query = parser().parse(raw).query;
        assert!(
            is_default_term(&query, normalized),
            "standalone {raw} must be retained as a literal"
        );
    }
    for raw in ["AND rust", "OR rust"] {
        let parsed = parser().parse(raw);
        assert!(is_default_term(&parsed.query, "rust"));
        assert!(
            parsed
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery)
        );
    }
}

#[test]
fn explicit_grouping_and_phrase_suffixes_are_retained() {
    let parsed = parser().parse("(rust OR ownership) AND \"error handling\"~2");
    assert!(matches!(
        parsed.query.root(),
        QueryNode::Boolean {
            operator: Some(BooleanOperator::And),
            ..
        }
    ));
    let QueryNode::Boolean {
        clauses,
        operator: Some(BooleanOperator::And),
    } = parsed.query.root()
    else {
        return;
    };
    assert_eq!(clauses.len(), 2);
    assert!(matches!(
        parsed.query.node(clauses[1].query),
        QueryNode::Phrase {
            slop: 2,
            prefix: false,
            ..
        }
    ));

    let recovered = parser().parse("\"error handling\"~2*");
    assert!(matches!(
        recovered.query.root(),
        QueryNode::Phrase {
            slop: 2,
            prefix: false,
            ..
        }
    ));
    assert!(recovered.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
            && diagnostic.message.contains("phrase suffix")
    }));
}

#[test]
fn analyzed_phrase_terms_retain_positions() {
    let query = parser().parse("src/main.rs").query;
    assert!(matches!(query.root(), QueryNode::Phrase { .. }));
    let QueryNode::Phrase { terms, .. } = query.root() else {
        return;
    };
    assert_eq!(
        terms.as_slice(),
        [
            PositionedTerm::new(0, "src"),
            PositionedTerm::new(1, "main"),
            PositionedTerm::new(2, "rs"),
        ]
    );
}

/// A trailing operand that duplicates an explicit-OR-run member must
/// SURVIVE: the pinned grammar nests the OR run one level below the
/// implicit disjunction, so duplicate retention never crosses that
/// boundary (bd-htcun) — and the nesting then FLATTENS for execution so
/// score accumulation associates exactly like the oracle's. A pure
/// implicit chain still dedups.
#[test]
fn mixed_chain_duplicate_survives_or_run_nesting() {
    let mixed = parser().parse("alpha OR beta AND gamma alpha");
    let QueryNode::Boolean { clauses, .. } = mixed.query.root() else {
        panic!("mixed chain must lower to a boolean: {:?}", mixed.query);
    };
    // Structural mirror (bd-5o5z8): the OR-chain keeps its own group and
    // each unfielded term keeps its two-field expansion group, exactly
    // the nesting tantivy evaluates; only the trailing raw duplicate
    // survives beside it.
    assert_eq!(
        clauses.len(),
        2,
        "mirrored execution shape: the OR/AND chain group and the \
         surviving trailing duplicate group: {clauses:?}"
    );
    let QueryNode::Boolean { clauses: chain, .. } = mixed.query.node(clauses[0].query) else {
        panic!("the leading OR chain keeps its group: {clauses:?}");
    };
    assert_eq!(chain.len(), 2);
    assert!(
        is_expansion_group(&mixed.query, chain[0].query, "alpha"),
        "leading operand keeps its expansion group: {chain:?}"
    );
    let QueryNode::Boolean {
        clauses: and_arm, ..
    } = mixed.query.node(chain[1].query)
    else {
        panic!("the AND arm keeps its group: {chain:?}");
    };
    assert_eq!(and_arm.len(), 2);
    assert!(and_arm.iter().all(|clause| clause.occur == Occur::Must));
    assert!(is_expansion_group(&mixed.query, and_arm[0].query, "beta"));
    assert!(is_expansion_group(&mixed.query, and_arm[1].query, "gamma"));
    assert!(
        is_expansion_group(&mixed.query, clauses[1].query, "alpha"),
        "the trailing raw duplicate survives as its own group: {clauses:?}"
    );

    let pure = parser().parse("alpha alpha");
    let QueryNode::Boolean {
        clauses: pure_clauses,
        ..
    } = pure.query.root()
    else {
        panic!(
            "two-operand implicit chain lowers to a boolean: {:?}",
            pure.query
        );
    };
    assert_eq!(
        pure_clauses.len(),
        2,
        "a pure implicit chain retains one syntax term expanded over its two fields: \
         {pure_clauses:?}"
    );
}

#[test]
fn field_ids_not_backend_handles_are_stored_in_leaves() {
    let parsed = parser().parse("title:Rust");
    assert!(matches!(
        parsed.query.root(),
        QueryNode::Term { fields, text }
            if fields.as_slice() == [QueryField::new(2, TITLE_BOOST)] && text == "rust"
    ));

    let id = parser().parse("id:Case-Sensitive");
    assert!(matches!(
        id.query.root(),
        QueryNode::Term { fields, text }
            if fields.as_slice() == [QueryField::new(0, 1.0)] && text == "Case-Sensitive"
    ));
}

#[test]
fn default_terms_expand_to_per_term_should_groups() {
    let single = parser().parse("content:release").query;
    assert!(
        matches!(
            single.root(),
            QueryNode::Term { fields, text }
                if fields.as_slice() == [QueryField::new(1, 1.0)] && text == "release"
        ),
        "an explicitly single-field term must remain one leaf"
    );

    // Structural mirror (bd-5o5z8): the parenthesised pair keeps its
    // group and each unfielded term keeps its per-term expansion group —
    // tantivy's exact evaluation tree, whose nesting decides the f32
    // association the gauntlet pins bit-for-bit.
    let parsed = parser().parse("(release OR require) OR return");
    let QueryNode::Boolean { clauses, operator } = parsed.query.root() else {
        panic!("unfielded three-term OR must lower to an ordered boolean");
    };
    assert_eq!(*operator, Some(BooleanOperator::Or));
    assert_eq!(clauses.len(), 2, "paren group plus trailing term group");
    assert!(clauses.iter().all(|clause| clause.occur == Occur::Should));
    let QueryNode::Boolean { clauses: paren, .. } = parsed.query.node(clauses[0].query) else {
        panic!("the parenthesised pair keeps its group");
    };
    assert_eq!(paren.len(), 2);
    assert!(is_expansion_group(&parsed.query, paren[0].query, "release"));
    assert!(is_expansion_group(&parsed.query, paren[1].query, "require"));
    assert!(is_expansion_group(
        &parsed.query,
        clauses[1].query,
        "return"
    ));
}

#[test]
fn boosts_groups_and_scopes_canonicalize_before_dedup() {
    // The pinned Tantivy grammar removes identical syntactic children
    // before field expansion. Parentheses disappear at leaf parsing,
    // identity boosts disappear before the rewrite, and boost equality is
    // based on parsed f64 values.
    for raw in [
        "rust rust",
        "(rust) rust",
        "rust rust^1",
        "rust^2 rust^2.0",
        "rust OR rust",
    ] {
        let query = parser().parse(raw).query;
        assert_eq!(
            term_leaf_count(&query, "rust"),
            2,
            "{raw:?} must retain one syntax term expanded over two fields: {query:?}"
        );
    }

    // Syntax identity precedes analysis: case and quote delimiter remain
    // distinct even when the analyzer produces the same lowercase term.
    for raw in ["Rust rust", "\"rust\" rust", "'rust' \"rust\""] {
        let query = parser().parse(raw).query;
        assert_eq!(
            term_leaf_count(&query, "rust"),
            4,
            "{raw:?} must retain both syntax terms and both field expansions: {query:?}"
        );
    }
    // Occurrence is part of the syntax key, and duplicate exclusions merge
    // exactly as duplicate positive clauses do.
    let merged = parser().parse("-rust -rust").query;
    let count_must_not = |query: &Query| match query.root() {
        QueryNode::Boolean { clauses, .. } => clauses
            .iter()
            .filter(|clause| clause.occur == Occur::MustNot)
            .count(),
        _ => 0,
    };
    assert_eq!(count_must_not(&merged), 1, "duplicate exclusions merge");

    for raw in ["rust^2^3", "(rust)^2^3", "\"rust\"^2^3"] {
        let recovered = parser().parse(raw);
        assert!(matches!(
            recovered.query.root(),
            QueryNode::Boost { factor, .. } if factor.to_bits() == 2.0_f32.to_bits()
        ));
        assert!(recovered.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic.message.contains("trailing boost")
        }));
    }
    let leading_decimal = parser().parse("rust^.5");
    assert!(is_default_term(&leading_decimal.query, "rust"));
    assert!(
        leading_decimal
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.kind == QueryDiagnosticKind::InvalidBoost })
    );
    let trailing_decimal = parser().parse("rust^2.");
    assert!(matches!(
        trailing_decimal.query.root(),
        QueryNode::Boost { factor, .. } if factor.to_bits() == 2.0_f32.to_bits()
    ));
    assert!(trailing_decimal.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
            && diagnostic.message.contains("trailing boost")
    }));

    let invalid = parser().parse("rust rust^scientific1e2");
    // The invalid boost is rejected and erased from the syntax tree, so
    // the surviving atom deduplicates with its bare sibling.
    assert_eq!(term_leaf_count(&invalid.query, "rust"), 2);
    assert!(
        invalid
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::InvalidBoost)
    );

    let scoped = parser().parse("title:(rust) content:(rust)").query;
    let QueryNode::Boolean { clauses, .. } = scoped.root() else {
        return;
    };
    assert_eq!(clauses.len(), 2);
    assert!(!query_subtree_eq(
        &scoped,
        clauses[0].query,
        clauses[1].query
    ));

    let recovered = parser().parse("unknown:(bad title:good)");
    assert!(
        ast_matches(
            &recovered.query,
            EvalDoc {
                id: "explicit-child",
                content: "bad",
                title: "good",
            },
        ),
        "the valid explicit inner field was not preserved: {:?}",
        recovered.query
    );
}

#[test]
fn parser_syntax_rewrite_stops_at_non_identity_boosts() {
    let boosted = parser().parse("(rust rust)^2").query;
    let QueryNode::Boost { query, factor } = boosted.root() else {
        panic!("the group boost must survive");
    };
    assert_eq!(factor.to_bits(), 2.0_f32.to_bits());
    let QueryNode::Boolean { clauses, .. } = boosted.node(*query) else {
        panic!("the boosted group must retain its Boolean syntax boundary");
    };
    assert_eq!(clauses.len(), 2, "rewrite_ast does not enter Boost");
    assert!(query_subtree_eq(
        &boosted,
        clauses[0].query,
        clauses[1].query
    ));

    let siblings = parser().parse("(rust rust)^2 rust^2").query;
    let QueryNode::Boolean { clauses, .. } = siblings.root() else {
        panic!("the distinct boosted siblings must both survive");
    };
    assert_eq!(clauses.len(), 2);
    assert!(!query_subtree_eq(
        &siblings,
        clauses[0].query,
        clauses[1].query
    ));
}

#[test]
fn parser_syntax_rewrite_flattens_singletons_and_transfers_occurrence() {
    // Structural mirror (bd-5o5z8): singleton wrappers still erase and
    // occurrences still transfer, but each unfielded term keeps its
    // two-field expansion group — the erased wrapper leaves the GROUP as
    // the surviving node, never a re-flattened leaf list.
    let unary = parser().parse("+a").query;
    assert!(is_expansion_group(&unary, unary.root_id(), "a"));
    let grouped = parser().parse("(+a)").query;
    assert!(is_expansion_group(&grouped, grouped.root_id(), "a"));
    let rewritten = parser().parse("(+a) a").query;
    assert!(
        is_expansion_group(&rewritten, rewritten.root_id(), "a"),
        "the raw-deduplicated singleton unwraps to the expansion group"
    );

    let boosted_empty = parser().parse("(+unknown:x^2) a").query;
    let QueryNode::Boolean { clauses, .. } = boosted_empty.root() else {
        panic!("the optional boosted-empty branch and term remain Boolean");
    };
    assert_eq!(clauses.len(), 2);
    assert!(clauses.iter().all(|clause| clause.occur == Occur::Should));
    assert!(matches!(
        boosted_empty.node(clauses[0].query),
        QueryNode::Boost { .. }
    ));
    assert!(is_expansion_group(&boosted_empty, clauses[1].query, "a"));

    let negative = parser().parse("(-a) b").query;
    let QueryNode::Boolean { clauses, .. } = negative.root() else {
        panic!("negative group plus optional term must remain Boolean");
    };
    assert_eq!(clauses.len(), 2);
    assert_eq!(clauses[0].occur, Occur::MustNot);
    assert!(is_expansion_group(&negative, clauses[0].query, "a"));
    assert_eq!(clauses[1].occur, Occur::Should);
    assert!(is_expansion_group(&negative, clauses[1].query, "b"));

    let required = parser().parse("(a AND a) b").query;
    let QueryNode::Boolean { clauses, .. } = required.root() else {
        panic!("required group plus optional term must remain Boolean");
    };
    assert_eq!(clauses.len(), 2);
    assert_eq!(clauses[0].occur, Occur::Must);
    assert!(is_expansion_group(&required, clauses[0].query, "a"));
    assert_eq!(clauses[1].occur, Occur::Should);
    assert!(is_expansion_group(&required, clauses[1].query, "b"));

    let equal_groups = parser().parse("((a AND a) b) (+a b)").query;
    let QueryNode::Boolean { clauses, .. } = equal_groups.root() else {
        panic!("recursively equal groups deduplicate and unwrap to the pair");
    };
    assert_eq!(clauses.len(), 2, "the deduplicated singleton unwraps");
    assert_eq!(clauses[0].occur, Occur::Must);
    assert!(is_expansion_group(&equal_groups, clauses[0].query, "a"));
    assert_eq!(clauses[1].occur, Occur::Should);
    assert!(is_expansion_group(&equal_groups, clauses[1].query, "b"));

    let flattened = parser().parse("(a AND a) +a").query;
    let QueryNode::Boolean { clauses, .. } = flattened.root() else {
        panic!("dedup-before-unwrap must leave the root Boolean");
    };
    assert_eq!(clauses.len(), 2, "the rewrite does not dedup a second time");
    assert!(clauses.iter().all(|clause| clause.occur == Occur::Must));
    assert!(query_subtree_eq(
        &flattened,
        clauses[0].query,
        clauses[1].query
    ));
}

#[test]
fn parser_syntax_rewrite_retains_lowered_away_raw_identity_until_trim() {
    let distinct_regex = parser().parse("(a /x/) (a /y/)").query;
    let QueryNode::Boolean { clauses, .. } = distinct_regex.root() else {
        panic!("raw-distinct groups must both survive");
    };
    assert_eq!(clauses.len(), 2);
    // Raw-identity dedup ran on the NESTED syntax (distinct /x/ vs /y/
    // prevent it); under the structural mirror each survivor keeps its
    // expansion group instead of splicing to leaves (bd-5o5z8).
    assert!(
        clauses
            .iter()
            .all(|clause| { is_expansion_group(&distinct_regex, clause.query, "a") })
    );

    let duplicate_regex = parser().parse("(a /x/) (a /x/)").query;
    assert!(
        is_expansion_group(&duplicate_regex, duplicate_regex.root_id(), "a"),
        "raw-identical groups deduplicate and the singleton unwraps to the group"
    );

    let distinct_unknown = parser().parse("(a unknown:x) (a unknown:y)").query;
    let QueryNode::Boolean { clauses, .. } = distinct_unknown.root() else {
        panic!("distinct unknown-field syntax must prevent parent dedup");
    };
    assert_eq!(clauses.len(), 2);
    assert!(
        clauses
            .iter()
            .all(|clause| { is_expansion_group(&distinct_unknown, clause.query, "a") })
    );

    let boosted = parser().parse("(a AND unknown:x)^2").query;
    let QueryNode::Boost { query, .. } = boosted.root() else {
        panic!("a valid boost remains a lowering boundary");
    };
    let QueryNode::Boolean { clauses, .. } = boosted.node(*query) else {
        panic!("the boosted conjunction remains Boolean");
    };
    assert_eq!(clauses.len(), 1, "semantic drops trim inside Boost");
    assert_eq!(clauses[0].occur, Occur::Must);

    let f64_only = parser().parse("a unknown:x^1.00000001").query;
    assert!(
        is_expansion_group(&f64_only, f64_only.root_id(), "a"),
        "an f64-only boost boundary erases and the survivor keeps its group"
    );
}

#[test]
fn overflowing_boost_recovers_without_nonfinite_score() {
    for overflowing in [format!("4{}", "0".repeat(38)), "9".repeat(400)] {
        let parsed = parser().parse(&format!("rust rust^{overflowing}"));
        assert!(
            parsed
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.kind == QueryDiagnosticKind::InvalidBoost })
        );
        let QueryNode::Boolean { clauses, .. } = parsed.query.root() else {
            panic!("the recovered duplicate keeps the rewritten root clause");
        };
        assert_eq!(
            clauses.len(),
            2,
            "the invalid boost is absent before the surviving term expands by field"
        );

        let group = parser().parse(&format!("(rust rust)^{overflowing}")).query;
        let QueryNode::Boolean { clauses, .. } = group.root() else {
            panic!("the recovered group keeps its rewritten root Boolean");
        };
        assert_eq!(
            clauses.len(),
            2,
            "invalid boost cannot block child rewrite or ordered field expansion"
        );
    }
}

#[test]
fn fast_numeric_field_exists_lowers_to_an_unbounded_range() {
    let typed_parser = DefaultQueryParser::new(TYPED_SCHEMA).expect("typed test schema is valid");
    let exists = typed_parser.parse("unsigned:*");
    assert_eq!(
        exists.query,
        Query::range(4, Bound::Unbounded, Bound::Unbounded)
    );
    assert!(exists.diagnostics.is_empty());

    let non_fast = typed_parser.parse("signed:*");
    assert_eq!(non_fast.query, Query::empty());
    assert!(non_fast.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::UnsupportedField
            && diagnostic
                .message
                .contains("does not support fast-field existence queries")
    }));
}

#[test]
fn typed_ranges_sets_comparisons_and_partial_recovery_are_pinned() {
    let typed_parser = DefaultQueryParser::new(TYPED_SCHEMA).expect("typed test schema is valid");
    assert_eq!(
        typed_parser.parse("signed:[-5 TO 10}").query,
        Query::range(
            3,
            Bound::Included(QueryValue::I64(-5)),
            Bound::Excluded(QueryValue::I64(10)),
        )
    );
    assert_eq!(
        typed_parser
            .parse("unsigned: >= 18446744073709551615")
            .query,
        Query::range(
            4,
            Bound::Included(QueryValue::U64(u64::MAX)),
            Bound::Unbounded,
        )
    );

    for raw in [
        "signed:[* TO 5] signed:{* TO 5]",
        "signed:[-5 TO *] signed:[-5 TO *}",
        "signed:<5 signed:[* TO 5}",
        "signed:>=5 signed:[5 TO *]",
    ] {
        let parsed = typed_parser.parse(raw);
        let QueryNode::Boolean { clauses, .. } = parsed.query.root() else {
            panic!("equivalent unbounded ranges must retain one root clause: {raw}");
        };
        assert_eq!(
            clauses.len(),
            1,
            "unbounded delimiters normalize before parser dedup: {raw}"
        );
    }

    let partial = typed_parser.parse("unsigned:[bad TO 3]");
    assert_eq!(
        partial.query,
        Query::range(4, Bound::Unbounded, Bound::Included(QueryValue::U64(3)),)
    );
    assert!(
        partial
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::InvalidTypedValue)
    );

    let set = typed_parser.parse("unsigned: IN [1 bad 2 1]");
    assert_eq!(
        set.query,
        Query::set(4, Vec::from([QueryValue::U64(1), QueryValue::U64(2)]),)
    );
    assert_eq!(
        typed_parser.parse("unsigned: IN []").query,
        Query::set(4, Vec::new())
    );
    assert_eq!(
        parser().parse(r#"id: IN [foo,bar ""]"#).query,
        Query::set(
            0,
            Vec::from([
                QueryValue::Str("foo,bar".to_owned()),
                QueryValue::Str(String::new()),
            ]),
        )
    );
    assert_eq!(
        parser().parse(r#"id: IN ["foo]bar"]"#).query,
        Query::set(0, Vec::from([QueryValue::Str("foo]bar".to_owned())]))
    );
    assert_eq!(
        parser().parse(r#"id: IN ["foo\q"]"#).query,
        Query::set(0, Vec::from([QueryValue::Str("fooq".to_owned())]))
    );
    let unterminated_set = parser().parse("id: IN [doc-a doc-b");
    assert_eq!(
        unterminated_set.query,
        Query::set(
            0,
            Vec::from([
                QueryValue::Str("doc-a".to_owned()),
                QueryValue::Str("doc-b".to_owned()),
            ]),
        )
    );
    assert!(unterminated_set.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
            && diagnostic.message.contains("unterminated typed set")
    }));
    assert_eq!(
        parser().parse(r"id:[foo\ bar TO z]").query,
        Query::range(
            0,
            Bound::Included(QueryValue::Str("foo bar".to_owned())),
            Bound::Included(QueryValue::Str("z".to_owned())),
        )
    );
    assert_eq!(
        parser().parse(r"id:[foo\\:bar TO z]").query,
        Query::range(
            0,
            Bound::Included(QueryValue::Str(r"foo\:bar".to_owned())),
            Bound::Included(QueryValue::Str("z".to_owned())),
        )
    );
    let unterminated = parser().parse("id:[a TO z");
    assert_eq!(
        unterminated.query,
        Query::range(
            0,
            Bound::Included(QueryValue::Str("a".to_owned())),
            Bound::Included(QueryValue::Str("z".to_owned())),
        )
    );
    assert!(unterminated.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
            && diagnostic.message.contains("unterminated typed range")
    }));
    let missing_to = parser().parse("id:[a z]");
    assert_eq!(
        missing_to.query,
        Query::range(
            0,
            Bound::Included(QueryValue::Str("a".to_owned())),
            Bound::Included(QueryValue::Str("z".to_owned())),
        )
    );
    assert!(missing_to.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
            && diagnostic.message.contains("missing keyword `TO`")
    }));
    for raw in ["id:[a TO", "id:[a"] {
        assert_eq!(
            parser().parse(raw).query,
            Query::range(
                0,
                Bound::Included(QueryValue::Str("a".to_owned())),
                Bound::Unbounded,
            ),
            "{raw}"
        );
    }
    assert!(typed_parser.parse("stored:[a TO b]").query.is_empty());
    let positionless_phrase = typed_parser.parse("summary:\"two words\"");
    assert!(matches!(
        positionless_phrase.query.root(),
        QueryNode::Phrase { terms, .. } if terms.len() == 2
    ));
    assert!(matches!(
        validate_index_capabilities(&positionless_phrase.query, TYPED_SCHEMA),
        Err(QueryCapabilityError::PositionsRequired {
            schema: "query-parser-typed-test",
            ref field,
            operator: QueryExplanation::Phrase,
            capability: IndexCapability::Positions,
        }) if field == "summary"
    ));
    assert!(typed_parser.parse("signed:[* TO *]").query.is_empty());

    assert_eq!(
        parser().parse("ord:[0 TO 1]").query,
        Query::range(
            4,
            Bound::Included(QueryValue::U64(0)),
            Bound::Included(QueryValue::U64(1)),
        )
    );
    assert_eq!(
        parser().parse("ord: IN [0 1]").query,
        Query::set(4, Vec::from([QueryValue::U64(0), QueryValue::U64(1)]),)
    );
}

#[test]
fn recovery_retains_valid_siblings_and_repairs_recursive_negatives() {
    for raw in ["(OR rust)", "(AND rust)"] {
        let parsed = parser().parse(raw);
        assert!(is_default_term(&parsed.query, "rust"));
        assert!(parsed.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
                && diagnostic.byte_offset == Some(1)
        }));
    }

    let unmatched = parser().parse("rust ) ownership");
    assert!(is_default_term(&unmatched.query, "rust"));
    assert!(unmatched.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery && diagnostic.byte_offset == Some(5)
    }));
    assert_eq!(
        parser()
            .parse("rust ))")
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery)
            .count(),
        1
    );

    let all_negative = parser().parse("(-alpha) OR (-beta)");
    let QueryNode::Boolean { clauses, .. } = all_negative.query.root() else {
        return;
    };
    assert!(
        matches!(clauses.last(), Some(QueryClause { occur: Occur::Should, query })
        if matches!(all_negative.query.node(*query), QueryNode::All))
    );
    assert!(
        all_negative
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::AllNegativeRepair)
    );

    let boosted = parser().parse("(-alpha)^2");
    let QueryNode::Boost { query, .. } = boosted.query.root() else {
        panic!("boost must remain outermost");
    };
    let QueryNode::Boolean { clauses, .. } = boosted.query.node(*query) else {
        panic!("negative repair must remain inside the boost");
    };
    assert!(
        matches!(clauses.last(), Some(QueryClause { occur: Occur::Should, query })
        if matches!(boosted.query.node(*query), QueryNode::All))
    );
}

#[test]
fn quoted_escapes_single_quotes_and_phrase_prefix_errors_are_pinned() {
    assert!(matches!(
        parser().parse("title:'error handling'").query.root(),
        QueryNode::Phrase { terms, .. } if terms.len() == 2
    ));
    assert!(matches!(
        parser().parse(r#"title:"say \"hello\"""#).query.root(),
        QueryNode::Phrase { terms, .. }
            if terms.iter().map(|term| term.text.as_str()).eq(["say", "hello"])
    ));
    let prefix = parser().parse("\"word\"*");
    assert!(prefix.query.is_empty());
    assert!(
        prefix
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::DroppedFragment)
    );

    let malformed_slop = parser().parse("\"rust ownership\"~deprecated");
    let QueryNode::Boolean { clauses, .. } = malformed_slop.query.root() else {
        panic!("malformed slop must retain phrase and trailing fragments");
    };
    assert_eq!(clauses.len(), 2);
    assert!(matches!(
        malformed_slop.query.node(clauses[0].query),
        QueryNode::Phrase { slop: 0, .. }
    ));
    assert!(is_expansion_group(
        &malformed_slop.query,
        clauses[1].query,
        "deprecated",
    ));
    assert!(malformed_slop.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::SyntaxRecovery
            && diagnostic
                .message
                .contains("retained as a separate fragment")
    }));

    let overflowing_slop = parser().parse("\"rust ownership\"~4294967296");
    let QueryNode::Boolean { clauses, .. } = overflowing_slop.query.root() else {
        panic!("overflowing slop must retain phrase and trailing fragments");
    };
    assert_eq!(clauses.len(), 2);
    assert!(matches!(
        overflowing_slop.query.node(clauses[0].query),
        QueryNode::Phrase { slop: 0, .. }
    ));
    assert!(is_expansion_group(
        &overflowing_slop.query,
        clauses[1].query,
        "4294967296",
    ));
}

#[test]
fn unquoted_escapes_and_unsupported_regex_recover_without_retargeting() {
    assert_eq!(
        parser().parse(r"id:foo\ bar").query,
        Query::term(Vec::from([QueryField::new(0, 1.0)]), "foo bar".to_owned())
    );
    assert_eq!(
        parser().parse(r"id:foo\:bar").query,
        Query::term(Vec::from([QueryField::new(0, 1.0)]), "foo:bar".to_owned())
    );
    assert_eq!(
        parser().parse(r"id:foo\q").query,
        Query::term(Vec::from([QueryField::new(0, 1.0)]), r"foo\q".to_owned())
    );

    let regex = parser().parse("/rust/");
    assert!(regex.query.is_empty());
    assert!(regex.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::DroppedFragment
            && diagnostic.message.contains("regular-expression")
    }));
    let recovered = parser().parse("-/rust/ rust").query;
    let QueryNode::Boolean { clauses, .. } = recovered.root() else {
        panic!("surviving term must retain default-field expansion");
    };
    assert_eq!(clauses.len(), 2);
    assert!(is_single_field_term(
        &recovered,
        clauses[0].query,
        "rust",
        QueryField::new(1, 1.0),
    ));
    assert!(is_single_field_term(
        &recovered,
        clauses[1].query,
        "rust",
        QueryField::new(2, TITLE_BOOST),
    ));
}

#[test]
fn parser_dedup_is_raw_and_preserves_distinct_normalized_fragments() {
    let distinct = parser().parse("Rust rust").query;
    assert!(matches!(distinct.root(), QueryNode::Boolean { .. }));
    assert_eq!(term_leaf_count(&distinct, "rust"), 4);

    // Idempotent MustNot clauses DO merge, even nested: the a-group and
    // one merged negation wrapper survive (bd-5o5z8 nesting).
    let recursive = parser().parse("a OR -b OR (-b)").query;
    assert!(matches!(
        recursive.root(),
        QueryNode::Boolean { clauses, .. } if clauses.len() == 2
    ));
}
