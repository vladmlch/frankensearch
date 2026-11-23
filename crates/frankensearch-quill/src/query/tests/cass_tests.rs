#[cfg(feature = "tantivy-oracle")]
use std::collections::BTreeSet;

use serde_json::{Value, json};

use super::*;
use crate::schema::{CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA};

fn cass_parser() -> CassQueryParser {
    CassQueryParser::new(CASS_SEMANTIC_SCHEMA).expect("CASS schema supports its parser")
}

#[test]
fn cass_parser_observation_uses_the_real_lexer_and_admitted_prefix() {
    let parser = cass_parser();
    let raw = "c++ \"exact phrase\" OR -deprecated";
    let filters = CassQueryFilters::default();
    let observed = parser
        .parse_observed(raw, &filters)
        .expect("bounded observation");

    assert_eq!(observed.parsed, parser.parse(raw, &filters));
    assert_eq!(observed.admitted_query, raw);
    assert_eq!(
        observed.sanitized_query,
        "c   \"exact phrase\" OR -deprecated"
    );
    assert_eq!(
        observed.tokens,
        vec![
            CassQueryToken::Term {
                text: "c++".to_owned(),
                byte_offset: 0,
            },
            CassQueryToken::Phrase {
                text: "exact phrase".to_owned(),
                byte_offset: 4,
            },
            CassQueryToken::Or { byte_offset: 19 },
            CassQueryToken::Not { byte_offset: 22 },
            CassQueryToken::Term {
                text: "deprecated".to_owned(),
                byte_offset: 23,
            },
        ]
    );

    let mut oversized = "auth".to_owned();
    oversized.push_str(&" ".repeat(MAX_QUERY_LENGTH - oversized.chars().count()));
    oversized.push_str("private-tail");
    let truncated = parser
        .parse_observed(&oversized, &filters)
        .expect("bounded truncated observation");
    assert!(truncated.parsed.was_truncated);
    assert_eq!(truncated.admitted_query.chars().count(), MAX_QUERY_LENGTH);
    assert!(!truncated.sanitized_query.contains("private-tail"));
    assert!(truncated.tokens.iter().all(|token| {
        !matches!(
            token,
            CassQueryToken::Term { text, .. } if text.contains("private-tail")
        )
    }));
}

#[test]
fn cass_parser_observation_uses_utf8_byte_offsets_within_the_admitted_prefix() {
    let parser = cass_parser();
    let filters = CassQueryFilters::default();
    let raw = "搜索 \"éclair\" OR 尾巴";
    let observed = parser
        .parse_observed(raw, &filters)
        .expect("multibyte observation is bounded");

    assert_eq!(observed.admitted_query, raw);
    assert_eq!(
        observed.tokens,
        vec![
            CassQueryToken::Term {
                text: "搜索".to_owned(),
                byte_offset: 0,
            },
            CassQueryToken::Phrase {
                text: "éclair".to_owned(),
                byte_offset: "搜索 ".len(),
            },
            CassQueryToken::Or {
                byte_offset: "搜索 \"éclair\" ".len(),
            },
            CassQueryToken::Term {
                text: "尾巴".to_owned(),
                byte_offset: "搜索 \"éclair\" OR ".len(),
            },
        ]
    );
    assert!(observed.tokens.iter().all(|token| {
        let offset = match token {
            CassQueryToken::Term { byte_offset, .. }
            | CassQueryToken::Phrase { byte_offset, .. }
            | CassQueryToken::And { byte_offset }
            | CassQueryToken::Or { byte_offset }
            | CassQueryToken::Not { byte_offset }
            | CassQueryToken::LParen { byte_offset }
            | CassQueryToken::RParen { byte_offset } => *byte_offset,
        };
        offset < observed.admitted_query.len() && observed.admitted_query.is_char_boundary(offset)
    }));
}

#[test]
fn cass_parser_observation_rejects_hostile_bounds_before_witness_allocation() {
    let parser = cass_parser();
    let oversized_raw = "x".repeat(MAX_CASS_OBSERVATION_RAW_QUERY_BYTES + 1);
    assert_eq!(
        parser.parse_observed(&oversized_raw, &CassQueryFilters::default()),
        Err(CassQueryObservationError::BoundExceeded {
            field: "raw_query_bytes",
            actual: MAX_CASS_OBSERVATION_RAW_QUERY_BYTES + 1,
            limit: MAX_CASS_OBSERVATION_RAW_QUERY_BYTES,
        })
    );

    let oversized_filter = "x".repeat(MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES + 1);
    let filters = CassQueryFilters {
        agents: vec![oversized_filter],
        ..CassQueryFilters::default()
    };
    assert_eq!(
        parser.parse_observed("bounded", &filters),
        Err(CassQueryObservationError::BoundExceeded {
            field: "filter_value_bytes",
            actual: MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES + 1,
            limit: MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES,
        })
    );

    let aggregate_filters = CassQueryFilters {
        agents: vec![
            "x".repeat(MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES);
            MAX_CASS_OBSERVATION_FILTER_BYTES / MAX_CASS_OBSERVATION_FILTER_VALUE_BYTES + 1
        ],
        ..CassQueryFilters::default()
    };
    assert!(matches!(
        parser.parse_observed("bounded", &aggregate_filters),
        Err(CassQueryObservationError::BoundExceeded {
            field: "aggregate_filter_bytes",
            ..
        })
    ));

    let filters = CassQueryFilters {
        agents: vec!["x".to_owned(); MAX_CASS_OBSERVATION_FILTER_VALUES + 1],
        ..CassQueryFilters::default()
    };
    assert_eq!(
        parser.parse_observed("bounded", &filters),
        Err(CassQueryObservationError::BoundExceeded {
            field: "filter_value_count",
            actual: MAX_CASS_OBSERVATION_FILTER_VALUES + 1,
            limit: MAX_CASS_OBSERVATION_FILTER_VALUES,
        })
    );
}

fn cass_filters_from_fixture(case: &Value) -> CassQueryFilters {
    let Some(filters) = case.get("filters") else {
        return CassQueryFilters::default();
    };
    let strings = |name: &str| {
        filters[name]
            .as_array()
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().expect("string filter").to_owned())
                    .collect()
            })
            .unwrap_or_default()
    };
    let source_filter = match filters["source_filter"].as_str() {
        Some("local") => CassSourceFilter::Local,
        Some("remote") => CassSourceFilter::Remote,
        Some("source_id") => CassSourceFilter::SourceId(
            filters["source_id"]
                .as_str()
                .expect("source_id filter value")
                .to_owned(),
        ),
        Some(unexpected) => panic!("unexpected CASS source fixture {unexpected:?}"),
        None => CassSourceFilter::All,
    };
    CassQueryFilters {
        agents: strings("agents"),
        workspaces: strings("workspaces"),
        created_from: filters["created_from"].as_i64(),
        created_to: filters["created_to"].as_i64(),
        source_filter,
    }
}

fn cass_wildcard_fixture(input: &str) -> Value {
    let pattern = CassWildcardPattern::parse(input);
    let searchable = ["title", "content", "title_prefix", "content_prefix"];
    let regex_fields = ["content", "title"];
    match pattern {
        CassWildcardPattern::Exact(_) => json!({
            "type": "Glob",
            "pattern": input,
            "class": "Exact",
            "strategy": "TermQuery",
            "fields": searchable,
        }),
        CassWildcardPattern::Prefix(term) => json!({
            "type": "Glob",
            "pattern": input,
            "normalized_term": term,
            "class": "Prefix",
            "strategy": "TermQuery",
            "fields": searchable,
        }),
        CassWildcardPattern::Suffix(term) => json!({
            "type": "Glob",
            "pattern": input,
            "class": "Suffix",
            "strategy": "RegexQuery",
            "regex": CassWildcardPattern::Suffix(term).to_regex(),
            "fields": regex_fields,
        }),
        CassWildcardPattern::Substring(term) => json!({
            "type": "Glob",
            "pattern": input,
            "class": "Substring",
            "strategy": "RegexQuery",
            "regex": CassWildcardPattern::Substring(term).to_regex(),
            "fields": regex_fields,
        }),
        CassWildcardPattern::Complex(term) => json!({
            "type": "Glob",
            "pattern": input,
            "class": "Complex",
            "strategy": "RegexQuery",
            "regex": CassWildcardPattern::Complex(term).to_regex(),
            "fields": regex_fields,
            "question_mark_operator": false,
        }),
    }
}

#[test]
fn cass_parser_matches_pinned_tantivy_fixture_trees() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../tests/fixtures/quill_language_contract.json"
    ))
    .expect("language fixture parses");
    let cases = fixture["parse_tree_cases"]
        .as_array()
        .expect("parse tree case array");
    let expected = cases.iter().filter(|case| case["parser"] == "cass").count();
    let mut executed = 0_usize;
    for case in cases {
        if case["parser"] != "cass" {
            continue;
        }
        let id = case["id"].as_str().expect("fixture id");
        let input = case["input"].as_str().expect("fixture input");
        let filters = cass_filters_from_fixture(case);
        let parsed = cass_parser().parse(input, &filters);
        let expected_ast =
            case["expected_ast"]["ref"].as_str().map_or_else(
                || &case["expected_ast"],
                |reference| {
                    &cases
                        .iter()
                        .find(|candidate| candidate["id"] == reference)
                        .unwrap_or_else(|| {
                            panic!("fixture {id} references missing {reference}")
                        })["expected_ast"]
                },
            );
        if case["query_class"] == "glob" {
            assert_eq!(&cass_wildcard_fixture(input), expected_ast, "fixture {id}");
            match (CassWildcardPattern::parse(input), parsed.query.root()) {
                (
                    CassWildcardPattern::Exact(expected) | CassWildcardPattern::Prefix(expected),
                    QueryNode::Boolean { clauses, operator },
                ) => {
                    // CASS fork (cass-prefix-expansion): exact and prefix
                    // literals lower to two OR'd arms - the incumbent
                    // four-field term query plus an unwindowed `content`
                    // glob - so a matching token is found past the 4 KiB
                    // `content_prefix` window. The fixture row pinned above
                    // still documents the shipping adapter's TermQuery
                    // strategy; this arm pins the fork's native shape.
                    assert_eq!(
                        operator,
                        &Some(BooleanOperator::Or),
                        "native parser glob arm operator for {id}"
                    );
                    assert_eq!(clauses.len(), 2, "native parser glob arms for {id}");
                    match (
                        parsed.query.node(clauses[0].query),
                        parsed.query.node(clauses[1].query),
                    ) {
                        (
                            QueryNode::Term { fields, text },
                            QueryNode::Glob { field_ids, pattern },
                        ) => {
                            assert_eq!(text, &expected, "native parser glob text for {id}");
                            assert_eq!(
                                fields,
                                &cass_parser().fields.searchable(),
                                "native parser glob fields for {id}"
                            );
                            assert_eq!(
                                pattern,
                                &format!("{}*", expected),
                                "native parser glob prefix-arm pattern for {id}"
                            );
                            assert_eq!(
                                field_ids.as_slice(),
                                &[cass_parser().fields.content][..],
                                "native parser glob prefix-arm field for {id}"
                            );
                        }
                        (first, second) => {
                            panic!(
                                "native parser two-arm mismatch for {id}: {first:?} + {second:?}"
                            )
                        }
                    }
                }
                (
                    CassWildcardPattern::Suffix(_)
                    | CassWildcardPattern::Substring(_)
                    | CassWildcardPattern::Complex(_),
                    QueryNode::Glob { field_ids, pattern },
                ) => {
                    assert_eq!(pattern, &input.to_lowercase(), "native glob for {id}");
                    assert_eq!(
                        field_ids,
                        &cass_parser().fields.regex_fields(),
                        "native parser regex fields for {id}"
                    );
                }
                (wildcard, query) => {
                    panic!("native parser glob mismatch for {id}: {wildcard:?} -> {query:?}")
                }
            }
        } else {
            assert_eq!(
                &fixture_json_for_schema(CASS_SEMANTIC_SCHEMA, &parsed.query, None),
                expected_ast,
                "fixture {id}"
            );
        }
        if let Some(expected_diagnostic) = case["expected_diagnostic"].as_str() {
            assert!(
                parsed
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message.contains(expected_diagnostic)),
                "fixture {id} lacked diagnostic {expected_diagnostic:?}: {:?}",
                parsed.diagnostics,
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
    assert_eq!(executed, expected, "every pinned CASS parser case ran");
    assert!(executed >= 39, "the full harvested CASS contract slice ran");
}

/// Fork (cass-prefix-expansion): non-CJK exact and prefix literals
/// lower to two OR'd arms — the incumbent four-field term query plus a
/// trailing-star glob over the unwindowed `content` field — so a
/// matching token is found past the 4 KiB `content_prefix` window
/// instead of only inside it.
#[test]
fn cass_exact_and_prefix_terms_lower_to_two_or_arms() {
    let filters = CassQueryFilters::default();
    for raw in ["docker", "docker*"] {
        let parsed = cass_parser().parse(raw, &filters);
        let QueryNode::Boolean { clauses, operator } = parsed.query.root() else {
            panic!("{raw} must lower to a two-arm Boolean root");
        };
        assert_eq!(
            operator,
            &Some(BooleanOperator::Or),
            "{raw}'s arms must combine with OR"
        );
        assert_eq!(clauses.len(), 2, "{raw} must have exactly two arms");
        match (
            parsed.query.node(clauses[0].query),
            parsed.query.node(clauses[1].query),
        ) {
            (QueryNode::Term { fields, text }, QueryNode::Glob { field_ids, pattern }) => {
                assert_eq!(text, "docker", "the exact arm carries the term itself");
                assert_eq!(
                    fields,
                    &cass_parser().fields.searchable(),
                    "the exact arm keeps the incumbent four-field expansion"
                );
                assert_eq!(
                    pattern, "docker*",
                    "the glob arm carries the trailing-star pattern"
                );
                assert_eq!(
                    field_ids,
                    &Vec::from([cass_parser().fields.content]),
                    "the glob arm expands only the unwindowed content field"
                );
            }
            (first, second) => panic!(
                "{raw} must lower to [term over searchable, content glob], got {first:?} + {second:?}"
            ),
        }
    }
}

/// Fork (cass-prefix-expansion): the CJK bigram decomposition and the
/// regex glob classes keep their incumbent lowerings — CJK literals
/// gain no glob arm, and a suffix literal keeps its regex glob root.
#[test]
fn cass_cjk_literals_and_regex_globs_keep_their_lowerings() {
    let filters = CassQueryFilters::default();
    let parsed = cass_parser().parse("搜索引擎", &filters);
    match parsed.query.root() {
        QueryNode::Boolean { clauses, operator } => {
            assert_eq!(
                operator,
                &Some(BooleanOperator::And),
                "CJK bigrams stay Must-joined"
            );
            assert_eq!(
                clauses.len(),
                3,
                "four CJK chars decompose to three bigrams"
            );
            for clause in clauses {
                assert_eq!(clause.occur, Occur::Must, "every bigram stays required");
                let QueryNode::Term { fields, .. } = parsed.query.node(clause.query) else {
                    panic!("each CJK bigram must stay an exact term, not gain a glob arm");
                };
                assert_eq!(
                    fields,
                    &cass_parser().fields.searchable(),
                    "each bigram keeps the four-field expansion"
                );
            }
        }
        other => panic!("CJK input must keep the bigram Boolean root, got {other:?}"),
    }
    let single = cass_parser().parse("搜", &filters);
    assert!(
        matches!(single.query.root(), QueryNode::Term { .. }),
        "a single CJK character stays a unigram term over the searchable fields"
    );
    let suffix = cass_parser().parse("*ocker", &filters);
    match suffix.query.root() {
        QueryNode::Glob { field_ids, pattern } => {
            assert_eq!(pattern, "*ocker", "the suffix literal keeps its pattern");
            assert_eq!(
                field_ids,
                &cass_parser().fields.regex_fields(),
                "the suffix glob keeps the regex fields [content, title]"
            );
        }
        other => panic!("a suffix literal must keep its regex glob root, got {other:?}"),
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_helpers_match_the_shipping_adapter() {
    use frankensearch_lexical::{
        CassQueryToken as OracleToken, CassWildcardPattern as OracleWildcard,
        cass_parse_boolean_query as oracle_parse, cass_sanitize_query as oracle_sanitize,
    };

    for input in [
        "",
        "auth token",
        "auth AND token",
        "auth&&token OR cache",
        "NOT -deprecated",
        "\"exact phrase\"",
        "c++ hello_world bd-q3fy",
        "検索 abc搜索def",
        "auth\rOR\rcache",
        "NOT AND cache",
        "auth OR NOT AND deprecated",
        "(auth OR token) cache",
        "-(auth token)",
        "foo(bar) x)",
        "((auth)",
        "(auth OR)",
        "auth ()",
    ] {
        assert_eq!(
            cass_sanitize_query(input),
            oracle_sanitize(input),
            "{input:?}"
        );
        let native = cass_lex(input, &mut Vec::new());
        let oracle = oracle_parse(input);
        let normalized_native = native
            .iter()
            .map(|token| match token {
                CassLexToken::Term { text, .. } => ("term", text.as_str()),
                CassLexToken::Phrase { text, .. } => ("phrase", text.as_str()),
                CassLexToken::And { .. } => ("and", ""),
                CassLexToken::Or { .. } => ("or", ""),
                CassLexToken::Not { .. } => ("not", ""),
                CassLexToken::LParen { .. } => ("(", ""),
                CassLexToken::RParen { .. } => (")", ""),
            })
            .collect::<Vec<_>>();
        let normalized_oracle = oracle
            .iter()
            .map(|token| match token {
                OracleToken::Term(text) => ("term", text.as_str()),
                OracleToken::Phrase(text) => ("phrase", text.as_str()),
                OracleToken::And => ("and", ""),
                OracleToken::Or => ("or", ""),
                OracleToken::Not => ("not", ""),
                OracleToken::LParen => ("(", ""),
                OracleToken::RParen => (")", ""),
            })
            .collect::<Vec<_>>();
        assert_eq!(normalized_native, normalized_oracle, "{input:?}");
    }

    for input in ["exact", "prefix*", "*suffix", "*middle*", "a*b*c", "f.o*"] {
        let native = CassWildcardPattern::parse(input);
        let oracle = OracleWildcard::parse(input);
        assert_eq!(format!("{native:?}"), format!("{oracle:?}"), "{input:?}");
        assert_eq!(native.to_regex(), oracle.to_regex(), "{input:?}");
    }
}

/// Fork (cass-prefix-expansion): assert `query`'s root is the two-arm
/// Exact lowering of `expected` — [term over the searchable fields,
/// content glob `{expected}*`] under an OR — which is what every
/// recovered single-token parse now lowers to.
fn assert_two_arm_exact_root(query: &Query, expected: &str) {
    let QueryNode::Boolean { clauses, operator } = query.root() else {
        panic!("{expected:?} must recover as the two-arm Boolean root");
    };
    assert_eq!(
        operator,
        &Some(BooleanOperator::Or),
        "{expected:?}'s arms must combine with OR"
    );
    assert_eq!(clauses.len(), 2, "{expected:?} must have exactly two arms");
    match (query.node(clauses[0].query), query.node(clauses[1].query)) {
        (QueryNode::Term { fields, text }, QueryNode::Glob { field_ids, pattern }) => {
            assert_eq!(text, expected, "the exact arm carries the term itself");
            assert_eq!(
                fields,
                &cass_parser().fields.searchable(),
                "the exact arm keeps the incumbent four-field expansion"
            );
            assert_eq!(
                pattern,
                &format!("{expected}*"),
                "the glob arm carries the trailing-star pattern"
            );
            assert_eq!(
                field_ids,
                &Vec::from([cass_parser().fields.content]),
                "the glob arm expands only the unwindowed content field"
            );
        }
        (first, second) => panic!(
            "{expected:?} must recover to [term over searchable, content glob], got \
             {first:?} + {second:?}"
        ),
    }
}

#[test]
fn cass_parser_validates_configuration_and_recovers_hostile_input() {
    assert!(matches!(
        CassQueryParser::new(DEFAULT_SCHEMA),
        Err(CassQueryParserConfigError::MissingField { field: "agent", .. })
    ));

    let parser = cass_parser();
    let mut oversized = "auth".to_owned();
    oversized.push_str(&" ".repeat(MAX_QUERY_LENGTH - oversized.chars().count()));
    oversized.push_str("cache");
    let truncated = parser.parse(&oversized, &CassQueryFilters::default());
    assert!(truncated.was_truncated);
    assert!(truncated.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == QueryDiagnosticKind::Truncated
            && diagnostic.message.contains("10000 Unicode scalar values")
    }));
    assert_two_arm_exact_root(&truncated.query, "auth");

    for raw in [
        "AND auth",
        "auth OR",
        "NOT",
        "NOT NOT",
        "auth NOT",
        "* auth",
        "auth AND OR cache",
        "auth OR NOT AND deprecated",
        "\"***\" auth",
        "\0\0\0",
    ] {
        let parsed = parser.parse(raw, &CassQueryFilters::default());
        assert!(
            !parsed.diagnostics.is_empty(),
            "hostile input {raw:?} must explain its recovery"
        );
    }

    let recovered = parser.parse("NOT AND cache", &CassQueryFilters::default());
    assert_two_arm_exact_root(&recovered.query, "cache");
    // The operand-less NOT and the operand-less AND are both dropped with
    // a diagnostic, leaving `auth OR deprecated`.
    let recovered = parser.parse("auth OR NOT AND deprecated", &CassQueryFilters::default());
    assert!(matches!(
        recovered.query.root(),
        QueryNode::Boolean {
            operator: Some(BooleanOperator::Or),
            ..
        }
    ));
    assert_eq!(
        recovered.query,
        parser
            .parse("auth OR deprecated", &CassQueryFilters::default())
            .query
    );
}

/// The `msg_idx` result set of each `(query, filters)` case in the pinned
/// Tantivy CASS builder, over an index of `docs`.
#[cfg(feature = "tantivy-oracle")]
fn cass_oracle_result_sets(
    docs: &[CassEvalDoc],
    cases: &[(&str, CassQueryFilters)],
) -> Vec<BTreeSet<u64>> {
    use frankensearch_lexical::tantivy_crate::collector::DocSetCollector;
    use frankensearch_lexical::{
        CassDocumentRef, CassQueryFilters as OracleFilters, CassSourceFilter as OracleSource,
        CassTantivyIndex, TantivyDocument, Value as TantivyValue, cass_build_tantivy_query,
    };

    let directory = tempfile::tempdir().expect("temporary CASS index directory");
    let mut index = CassTantivyIndex::open_or_create(directory.path()).expect("CASS index");
    let oracle_documents = docs
        .iter()
        .map(|doc| CassDocumentRef {
            agent: doc.agent,
            workspace: doc.workspace,
            workspace_original: doc.workspace,
            source_path: "/fixture/session.jsonl",
            msg_idx: doc.msg_idx,
            created_at: doc.created_at,
            title: Some(doc.title),
            content: doc.content,
            source_id: doc.source_id,
            origin_kind: doc.origin_kind,
            origin_host: None,
            conversation_id: None,
        })
        .collect::<Vec<_>>();
    index
        .add_cass_document_refs(&oracle_documents)
        .expect("index CASS oracle documents");
    index.commit().expect("commit CASS oracle documents");
    let reader = index.reader().expect("open CASS oracle reader");
    reader.reload().expect("reload committed CASS oracle");
    let searcher = reader.searcher();
    let fields = index.fields();
    cases
        .iter()
        .map(|(raw, filters)| {
            let oracle_filters = OracleFilters {
                agents: filters.agents.clone(),
                workspaces: filters.workspaces.clone(),
                created_from: filters.created_from,
                created_to: filters.created_to,
                source_filter: match &filters.source_filter {
                    CassSourceFilter::All => OracleSource::All,
                    CassSourceFilter::Local => OracleSource::Local,
                    CassSourceFilter::Remote => OracleSource::Remote,
                    CassSourceFilter::SourceId(source_id) => {
                        OracleSource::SourceId(source_id.clone())
                    }
                },
            };
            let oracle_query = cass_build_tantivy_query(raw, &oracle_filters, &fields);
            searcher
                .search(&*oracle_query, &DocSetCollector)
                .expect("search shipping CASS builder")
                .into_iter()
                .map(|address| {
                    let document: TantivyDocument =
                        searcher.doc(address).expect("load CASS oracle document");
                    document
                        .get_first(fields.msg_idx)
                        .and_then(|value| value.as_u64())
                        .expect("stored CASS msg_idx")
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_parser_result_sets_match_the_shipping_tantivy_builder() {
    const DOCS: [CassEvalDoc; 8] = [
        CassEvalDoc {
            msg_idx: 0,
            agent: "claude",
            workspace: Some("/alpha"),
            created_at: Some(100),
            title: "Authentication Cache",
            content: "auth token cache active",
            source_id: "local-a",
            origin_kind: "local",
        },
        CassEvalDoc {
            msg_idx: 1,
            agent: "codex",
            workspace: Some("/alpha"),
            created_at: Some(200),
            title: "Legacy Authentication",
            content: "auth deprecated token",
            source_id: "remote-a",
            origin_kind: "ssh",
        },
        CassEvalDoc {
            msg_idx: 2,
            agent: "claude",
            workspace: Some("/beta"),
            created_at: Some(300),
            title: "Cache Search",
            content: "cache search engine",
            source_id: "archive-7",
            origin_kind: "local",
        },
        CassEvalDoc {
            msg_idx: 3,
            agent: "gemini",
            workspace: None,
            created_at: Some(400),
            title: "搜索引擎",
            content: "这是一个搜索引擎测试",
            source_id: "remote-b",
            origin_kind: "ssh",
        },
        CassEvalDoc {
            msg_idx: 4,
            agent: "codex",
            workspace: Some("/beta"),
            created_at: Some(500),
            title: "Error Handling",
            content: "error handling guide auth cache",
            source_id: "archive-7",
            origin_kind: "local",
        },
        CassEvalDoc {
            msg_idx: 5,
            agent: "claude",
            workspace: Some("/alpha"),
            created_at: Some(600),
            title: "Foobar Service",
            content: "foobarbaz middleware",
            source_id: "remote-c",
            origin_kind: "ssh",
        },
        CassEvalDoc {
            msg_idx: 6,
            agent: "codex",
            workspace: Some("/gamma"),
            created_at: None,
            title: "Unrelated",
            content: "nothing applicable",
            source_id: "local-b",
            origin_kind: "local",
        },
        CassEvalDoc {
            msg_idx: 7,
            agent: "claude",
            workspace: Some("/alpha"),
            created_at: Some(700),
            title: "Token Store",
            content: "token cache deprecated",
            source_id: "remote-d",
            origin_kind: "ssh",
        },
    ];

    let cases = vec![
        ("auth", CassQueryFilters::default()),
        ("auth token", CassQueryFilters::default()),
        ("auth OR token OR cache", CassQueryFilters::default()),
        // Both engines read the cass#52 grammar (GH #56); on this corpus
        // the legacy OR-binds-tighter reading differs from these.
        ("auth OR token AND cache", CassQueryFilters::default()),
        ("auth AND token OR cache", CassQueryFilters::default()),
        ("auth token OR search", CassQueryFilters::default()),
        (
            "(auth OR token) AND deprecated",
            CassQueryFilters::default(),
        ),
        (
            "auth AND (search OR deprecated)",
            CassQueryFilters::default(),
        ),
        ("NOT NOT deprecated", CassQueryFilters::default()),
        ("-(auth OR search)", CassQueryFilters::default()),
        ("cache -(token OR search)", CassQueryFilters::default()),
        (
            "deprecated OR -(auth OR cache)",
            CassQueryFilters::default(),
        ),
        ("((auth))", CassQueryFilters::default()),
        ("(auth OR cache", CassQueryFilters::default()),
        ("auth () OR search", CassQueryFilters::default()),
        ("auth && cache", CassQueryFilters::default()),
        ("auth || search", CassQueryFilters::default()),
        ("\"error handling\"", CassQueryFilters::default()),
        ("foo*", CassQueryFilters::default()),
        ("*bar", CassQueryFilters::default()),
        ("*baz", CassQueryFilters::default()),
        ("*bar*", CassQueryFilters::default()),
        ("f*bar", CassQueryFilters::default()),
        ("*foo*baz", CassQueryFilters::default()),
        ("*bar *baz", CassQueryFilters::default()),
        ("f*o", CassQueryFilters::default()),
        ("搜索", CassQueryFilters::default()),
        ("auth AND NOT deprecated", CassQueryFilters::default()),
        ("auth OR NOT deprecated", CassQueryFilters::default()),
        ("NOT deprecated", CassQueryFilters::default()),
        ("-deprecated", CassQueryFilters::default()),
        (
            "cache",
            CassQueryFilters {
                agents: vec!["claude".to_owned(), "codex".to_owned()],
                workspaces: vec!["/alpha".to_owned(), "/beta".to_owned()],
                ..CassQueryFilters::default()
            },
        ),
        (
            "*cache",
            CassQueryFilters {
                agents: vec!["claude".to_owned()],
                workspaces: vec!["/alpha".to_owned()],
                source_filter: CassSourceFilter::Remote,
                ..CassQueryFilters::default()
            },
        ),
        (
            "",
            CassQueryFilters {
                created_from: Some(200),
                created_to: Some(500),
                ..CassQueryFilters::default()
            },
        ),
        (
            "",
            CassQueryFilters {
                source_filter: CassSourceFilter::Local,
                ..CassQueryFilters::default()
            },
        ),
        (
            "",
            CassQueryFilters {
                source_filter: CassSourceFilter::Remote,
                ..CassQueryFilters::default()
            },
        ),
        (
            "",
            CassQueryFilters {
                source_filter: CassSourceFilter::SourceId("archive-7".to_owned()),
                ..CassQueryFilters::default()
            },
        ),
    ];

    let oracle_sets = cass_oracle_result_sets(&DOCS, &cases);
    for ((raw, filters), oracle) in cases.iter().zip(oracle_sets) {
        let parsed = cass_parser().parse(raw, filters);
        let native = DOCS
            .iter()
            .copied()
            .filter(|doc| cass_ast_matches(&parsed.query, *doc))
            .map(|doc| doc.msg_idx)
            .collect::<BTreeSet<_>>();
        assert_eq!(native, oracle, "result-set differential for {raw:?}");
    }
}

/// cass#52 grammar: result sets follow standard Boolean algebra (NOT >
/// AND > OR, parentheses group, NOT NOT cancels) in Quill and in the
/// pinned Tantivy oracle (GH #56). The corpus separates the standard
/// reading from the legacy OR-binds-tighter one on every mixed case, so
/// the legacy grammar fails each of them in either engine.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn cass_parser_precedence_matches_set_algebra_over_the_oracle() {
    let doc = |msg_idx: u64, content: &'static str| CassEvalDoc {
        msg_idx,
        agent: "claude",
        workspace: None,
        created_at: None,
        title: "",
        content,
        source_id: "local",
        origin_kind: "local",
    };
    let docs = [
        doc(0, "auth"),
        doc(1, "token cache"),
        doc(2, "token"),
        doc(3, "cache"),
        doc(4, "auth cache"),
    ];
    let cases: [(&str, Vec<u64>); 9] = [
        // auth ∪ (token ∩ cache); legacy: (auth ∪ token) ∩ cache = {1, 4}
        ("auth OR token AND cache", vec![0, 1, 4]),
        // (auth ∩ token) ∪ cache; legacy: auth ∩ (token ∪ cache) = {4}
        ("auth AND token OR cache", vec![1, 3, 4]),
        ("auth token OR cache", vec![1, 3, 4]),
        ("(auth OR token) AND cache", vec![1, 4]),
        ("auth AND (token OR cache)", vec![4]),
        ("NOT NOT auth", vec![0, 4]),
        ("-(token OR cache)", vec![0]),
        ("auth -(token OR cache)", vec![0]),
        ("((auth))", vec![0, 4]),
    ];
    let oracle_sets = cass_oracle_result_sets(
        &docs,
        &cases
            .iter()
            .map(|(raw, _)| (*raw, CassQueryFilters::default()))
            .collect::<Vec<_>>(),
    );
    for ((raw, expected), oracle) in cases.into_iter().zip(oracle_sets) {
        assert_eq!(
            oracle.into_iter().collect::<Vec<_>>(),
            expected,
            "oracle {raw:?}"
        );
        let parsed = cass_parser().parse(raw, &CassQueryFilters::default());
        let matched = docs
            .iter()
            .copied()
            .filter(|doc| cass_ast_matches(&parsed.query, *doc))
            .map(|doc| doc.msg_idx)
            .collect::<Vec<_>>();
        assert_eq!(matched, expected, "{raw:?} -> {:?}", parsed.query);
        assert!(
            parsed.diagnostics.is_empty(),
            "{raw:?}: {:?}",
            parsed.diagnostics
        );
    }
}

/// cass#52 grammar, feature-independent: equivalent spellings lower to
/// the same tree, the two groupings of a mixed query differ, code
/// parentheses stay inside terms, and malformed groups are recovered.
#[test]
fn cass_parser_groups_with_parentheses_and_standard_precedence() {
    let parser = cass_parser();
    let filters = CassQueryFilters::default();
    let tree = |raw: &str| parser.parse(raw, &filters).query;

    assert_eq!(
        tree("auth OR token AND cache"),
        tree("auth OR (token AND cache)")
    );
    assert_ne!(
        tree("auth OR token AND cache"),
        tree("(auth OR token) AND cache"),
        "negative: the legacy grammar read the first as the second"
    );
    assert_eq!(
        tree("auth token OR cache"),
        tree("(auth AND token) OR cache")
    );
    assert_eq!(tree("NOT NOT auth"), tree("auth"));
    assert_eq!(tree("-(auth OR token)"), tree("NOT (auth OR token)"));
    assert_eq!(tree("((auth))"), tree("auth"));

    // `(` inside a word and `)` with no open group are term characters.
    let observed = parser
        .parse_observed("foo(bar) x)", &filters)
        .expect("bounded observation");
    assert_eq!(
        observed.tokens,
        vec![
            CassQueryToken::Term {
                text: "foo(bar)".to_owned(),
                byte_offset: 0,
            },
            CassQueryToken::Term {
                text: "x)".to_owned(),
                byte_offset: 9,
            },
        ]
    );
    let grouped = parser
        .parse_observed("-(a b)", &filters)
        .expect("bounded observation");
    assert_eq!(
        grouped.tokens,
        vec![
            CassQueryToken::Not { byte_offset: 0 },
            CassQueryToken::LParen { byte_offset: 1 },
            CassQueryToken::Term {
                text: "a".to_owned(),
                byte_offset: 2,
            },
            CassQueryToken::Term {
                text: "b".to_owned(),
                byte_offset: 4,
            },
            CassQueryToken::RParen { byte_offset: 5 },
        ]
    );

    let unclosed = parser.parse("(auth OR token", &filters);
    assert_eq!(unclosed.query, tree("auth OR token"));
    assert!(
        unclosed
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("unclosed '('")),
        "{:?}",
        unclosed.diagnostics
    );
    let empty = parser.parse("auth ()", &filters);
    assert_eq!(empty.query, tree("auth"));
    assert!(
        empty
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("empty group")),
        "{:?}",
        empty.diagnostics
    );
}
