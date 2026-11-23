use std::io::{self, Write};
use std::ops::Bound;
use std::sync::{Arc, Mutex};

use super::*;
use crate::schema::{
    Analyzer, DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA, FieldDescriptor, FieldKind, SchemaDescriptor,
};

#[test]
fn all_negative_repair_is_root_only() {
    let query = parser().parse("rust OR (-deprecated)").query;
    assert!(matches!(query.root(), QueryNode::Boolean { .. }));
    let QueryNode::Boolean { clauses, .. } = query.root() else {
        return;
    };
    assert_eq!(clauses.len(), 2, "rust group plus the negation wrapper");
    assert!(matches!(
        query.node(clauses[1].query),
        QueryNode::Boolean { .. }
    ));
    let QueryNode::Boolean {
        clauses: negative, ..
    } = query.node(clauses[1].query)
    else {
        return;
    };
    assert_eq!(negative.len(), 1);
    assert_eq!(negative[0].occur, Occur::MustNot);
}

/// A position-free text schema is a valid, smaller index configuration.
/// Before `bd-vmpb` it was unqueryable end to end: `DefaultQueryParser::new`
/// refused to construct, so the QG `positions_off` fixture panicked and
/// aborted every QG-1 matrix run before any evidence could be written.
const POSITIONLESS_FIELDS: [FieldDescriptor; 2] = [
    FieldDescriptor {
        id: 0,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "title",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
];
const POSITIONLESS_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "query-parser-positionless-test",
    fields: &POSITIONLESS_FIELDS,
};

#[test]
fn positionless_default_fields_are_accepted_and_parse_position_independent_queries() {
    let parser = DefaultQueryParser::new(POSITIONLESS_SCHEMA)
        .expect("a positionless default-analyzed schema must bind a parser (bd-vmpb)");

    // Every position-independent operator must parse AND clear the
    // capability gate on a positionless index.
    for query in [
        "alpha",
        "alpha beta",
        "alpha OR beta",
        "alpha AND beta",
        "-alpha beta",
        "content:alpha",
        "alph*",
    ] {
        let parsed = parser.parse(query);
        validate_index_capabilities(&parsed.query, POSITIONLESS_SCHEMA).unwrap_or_else(|error| {
            panic!("{query:?} is position-independent and must be servable: {error}")
        });
    }
}

/// The capability gate rejects a phrase regardless of whether it came
/// from the string parser or an already-built AST.
#[test]
fn capability_gate_rejects_a_constructed_phrase_on_positionless_fields() {
    let phrase = Query::phrase(
        Vec::from([QueryField::new(0, 1.0)]),
        Vec::from([
            PositionedTerm {
                position: 0,
                text: "alpha".to_owned(),
            },
            PositionedTerm {
                position: 1,
                text: "beta".to_owned(),
            },
        ]),
        0,
        false,
    );

    let error = validate_index_capabilities(&phrase, POSITIONLESS_SCHEMA)
        .expect_err("a multi-term phrase reads positions and cannot be served");
    let QueryCapabilityError::PositionsRequired {
        schema,
        field,
        operator,
        capability,
    } = error
    else {
        panic!("constructed phrase must fail on missing positions");
    };
    assert_eq!(schema, "query-parser-positionless-test");
    // The error names the offending FIELD, not a bare "unsupported query".
    assert_eq!(field, "content");
    assert_eq!(operator, QueryExplanation::Phrase);
    assert_eq!(capability, IndexCapability::Positions);

    // Nesting must not launder it.
    let nested = Query::boolean(Vec::from([BooleanClause::new(Occur::Should, phrase)]), None);
    assert!(validate_index_capabilities(&nested, POSITIONLESS_SCHEMA).is_err());
}

#[test]
fn positionless_parser_preserves_phrases_for_typed_capability_rejection() {
    let parser = DefaultQueryParser::new(POSITIONLESS_SCHEMA).expect("bind positionless parser");
    let parsed = parser.parse("\"alpha beta\"");

    assert_ne!(
        parsed.query,
        Query::empty(),
        "a capability mismatch must not silently erase the parsed branch"
    );
    let error = validate_index_capabilities(&parsed.query, POSITIONLESS_SCHEMA)
        .expect_err("the preserved phrase must fail at the typed capability boundary");
    assert!(matches!(
        error,
        QueryCapabilityError::PositionsRequired {
            schema: "query-parser-positionless-test",
            ref field,
            operator: QueryExplanation::Phrase,
            capability: IndexCapability::Positions,
        } if field == "content"
    ));
}

#[test]
fn single_term_phrase_needs_no_positions() {
    let parser = DefaultQueryParser::new(POSITIONLESS_SCHEMA).expect("bind positionless parser");
    let parsed = parser.parse("\"alpha\"");
    // A one-term phrase degenerates to a term match and reads no position
    // list, which is the same exemption lowering already applies.
    validate_index_capabilities(&parsed.query, POSITIONLESS_SCHEMA)
        .expect("a single-term phrase reads no positions");
}

#[test]
fn positioned_schema_still_serves_phrases() {
    let parser = DefaultQueryParser::new(DEFAULT_SCHEMA).expect("bind shipping parser");
    let parsed = parser.parse("\"alpha beta\"");
    validate_index_capabilities(&parsed.query, DEFAULT_SCHEMA)
        .expect("positionful behaviour must be unchanged");
}

#[test]
fn schemas_without_both_default_fields_are_rejected() {
    assert!(matches!(
        DefaultQueryParser::new(FSFS_CHUNK_SCHEMA),
        Err(QueryParserConfigError::MissingDefaultField { field: "title", .. })
    ));

    // A default field with the WRONG ANALYZER is still rejected: query-time
    // term text would not match the text indexing produced, so the schema
    // genuinely cannot be served. Positions are a separate axis — see
    // `positionless_default_fields_are_accepted` (`bd-vmpb`).
    const WRONG_ANALYZER_FIELDS: [FieldDescriptor; 2] = [
        FieldDescriptor {
            id: 0,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::CassHyphenNormalize,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
    ];
    const WRONG_ANALYZER_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "query-parser-wrong-analyzer-test",
        fields: &WRONG_ANALYZER_FIELDS,
    };
    assert!(matches!(
        DefaultQueryParser::new(WRONG_ANALYZER_SCHEMA),
        Err(QueryParserConfigError::InvalidDefaultField {
            field: "content",
            ..
        })
    ));

    const GAPPED_FIELDS: [FieldDescriptor; 2] = [
        FieldDescriptor {
            id: 0,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 2,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
    ];
    const GAPPED_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "query-parser-gapped-test",
        fields: &GAPPED_FIELDS,
    };
    assert!(matches!(
        DefaultQueryParser::new(GAPPED_SCHEMA),
        Err(QueryParserConfigError::InvalidSchema { .. })
    ));
}

#[test]
fn range_and_glob_ast_variants_use_stable_field_ids() {
    let range = Query::range(
        5,
        Bound::Included(QueryValue::I64(10)),
        Bound::Excluded(QueryValue::I64(20)),
    );
    let set = Query::set(5, Vec::from([QueryValue::I64(10), QueryValue::I64(20)]));
    let glob = Query::glob(Vec::from([6, 7]), "cache*".to_owned());
    assert!(matches!(range.root(), QueryNode::Range { field_id: 5, .. }));
    assert!(matches!(set.root(), QueryNode::Set { field_id: 5, .. }));
    assert!(
        matches!(glob.root(), QueryNode::Glob { field_ids, .. } if field_ids.as_slice() == [6, 7])
    );
}

#[test]
fn parser_depth_is_independent_of_the_native_stack() {
    let mut query = "needle".to_owned();
    for _ in 0..MAX_QUERY_DEPTH + 20 {
        query = format!("needle OR ({query})");
    }
    let parsed = parser().parse(&query);
    assert!(!parsed.query.is_empty());
    assert!(
        !parsed
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::DepthLimit)
    );
}

#[test]
fn redundant_outer_groups_do_not_consume_the_depth_budget() {
    let query = format!(
        "{}needle{}",
        "(".repeat(MAX_QUERY_DEPTH + 20),
        ")".repeat(MAX_QUERY_DEPTH + 20)
    );
    let parsed = parser().parse(&query);
    let unwrapped = parser().parse("needle");
    assert_eq!(parsed.query, unwrapped.query);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.kind != QueryDiagnosticKind::DepthLimit)
    );
}

#[test]
fn redundant_same_field_outer_scopes_do_not_consume_the_depth_budget() {
    let query = format!(
        "{}needle{}",
        "content:(".repeat(MAX_QUERY_DEPTH + 20),
        ")".repeat(MAX_QUERY_DEPTH + 20)
    );
    let parsed = parser().parse(&query);
    let scoped = parser().parse("content:(needle)");
    assert_eq!(parsed.query, scoped.query);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.kind != QueryDiagnosticKind::DepthLimit)
    );
}

/// A chain that interleaves unfielded groups with one repeated field scope
/// is inert for both reasons at once, so neither pure normalization claimed
/// it and every group was charged against the depth limit.
#[test]
fn redundant_mixed_scope_outer_groups_do_not_consume_the_depth_budget() {
    let scoped = parser().parse("content:(needle)");
    for (label, query) in [
        (
            "unfielded innermost",
            format!(
                "{}(needle{}",
                "content:(".repeat(MAX_QUERY_DEPTH + 20),
                ")".repeat(MAX_QUERY_DEPTH + 21)
            ),
        ),
        (
            "unfielded outermost",
            format!(
                "({}needle{}",
                "content:(".repeat(MAX_QUERY_DEPTH + 20),
                ")".repeat(MAX_QUERY_DEPTH + 21)
            ),
        ),
    ] {
        let parsed = parser().parse(&query);
        assert_eq!(parsed.query, scoped.query, "{label}");
        assert!(
            parsed
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.kind != QueryDiagnosticKind::DepthLimit),
            "{label} must not charge the depth budget"
        );
    }
}

/// The retained wrapper must be the FIELDED one.
///
/// Keeping whichever opener happened to be innermost would drop the scope
/// whenever that opener is an unfielded group, quietly widening a
/// `content`-scoped query to the default fields. That is a wrong answer,
/// not a refused one, so it is pinned separately from the depth property.
#[test]
fn mixed_scope_peeling_retains_the_field_scope() {
    let peeled = parser().parse("content:((needle))");
    assert_eq!(peeled.query, parser().parse("content:(needle)").query);
    assert_ne!(
        peeled.query,
        parser().parse("(needle)").query,
        "peeling must not widen a scoped query to the default fields"
    );
}

#[test]
fn same_field_outer_scope_peeling_refuses_non_pure_wrappers() {
    for raw in [
        "content:(title:(needle))",
        "content:((title:(needle)))",
        "content:(content:(needle)^2)",
        "content:(content:(needle))^2",
        "content:(content:(needle) title)",
        "content:(content:(needle)) title",
        "+content:(content:(needle))",
        "content:(content:(needle)",
    ] {
        let mut diagnostics = Vec::new();
        let mut tokens = lex(raw, &mut diagnostics);
        let original = tokens.clone();
        peel_redundant_outer_groups(&mut tokens);
        assert_eq!(tokens, original, "{raw:?} must not be normalized");
    }
}

#[test]
fn deterministic_hostile_strings_do_not_panic() {
    let alphabet = b"abc XYZ0123\"'():^+-*/._@#\t\n";
    let alphabet_len = u64::try_from(alphabet.len()).expect("alphabet length fits u64");
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    for length in 0..512 {
        let mut input = String::with_capacity(length);
        for _ in 0..length {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let index = usize::try_from(state % alphabet_len).expect("alphabet index fits usize");
            input.push(char::from(alphabet[index]));
        }
        let _ = parser().parse(&input);
    }
}

#[derive(Clone, Debug)]
struct TestLogWriter {
    buffer: Arc<Mutex<Vec<u8>>>,
}

impl Write for TestLogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.buffer
            .lock()
            .expect("test log buffer lock is not poisoned")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn every_lenient_diagnostic_is_also_logged_at_warn() {
    let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let writer_buffer = Arc::clone(&buffer);
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_max_level(tracing::Level::WARN)
        .with_writer(move || TestLogWriter {
            buffer: Arc::clone(&writer_buffer),
        })
        .finish();
    let capture_buffer = Arc::clone(&buffer);
    let parsed = tracing::subscriber::with_default(subscriber, || {
        let mut parsed = parser().parse("unknown_field:value \"unterminated");
        for _ in 0..256 {
            let captured_bytes = capture_buffer
                .lock()
                .expect("test log buffer lock is not poisoned");
            let captured = String::from_utf8_lossy(&captured_bytes).lines().count();
            if captured >= parsed.diagnostics.len() {
                break;
            }
            drop(captured_bytes);
            tracing::callsite::rebuild_interest_cache();
            std::thread::sleep(std::time::Duration::from_millis(1));
            parsed = parser().parse("unknown_field:value \"unterminated");
        }
        parsed
    });
    assert!(!parsed.diagnostics.is_empty());
    let logs = String::from_utf8(
        buffer
            .lock()
            .expect("test log buffer lock is not poisoned")
            .clone(),
    )
    .expect("captured tracing output is UTF-8");
    let diagnostic_lines = logs
        .lines()
        .filter(|line| line.contains("lenient query parse diagnostic"))
        .collect::<Vec<_>>();
    assert!(
        diagnostic_lines.len() >= parsed.diagnostics.len(),
        "captured {} warning events for {} diagnostics: {logs:?}",
        diagnostic_lines.len(),
        parsed.diagnostics.len()
    );
    assert!(diagnostic_lines.iter().all(|line| {
        line.trim_start().starts_with("WARN ") && line.contains(crate::tracing_conventions::TARGET)
    }));
    assert!(!logs.contains("unknown_field"));
    assert!(!logs.contains("unterminated"));
    assert!(!logs.contains("value"));
}

/// Parse through the real Grammar while bypassing the public 10,000-byte
/// truncation, so the oversized-token admission branch is exercisable.
fn parse_untruncated(parser: &DefaultQueryParser, input: &str) -> ParsedQuery {
    let mut diagnostics = Vec::new();
    let tokens = lex(input, &mut diagnostics);
    let mut grammar = Grammar {
        parser: *parser,
        tokens,
        cursor: 0,
        diagnostics,
        lowered_atoms: Vec::new(),
        field_scopes: Vec::new(),
        allocation_failure: None,
    };
    grammar.lower_primary_tokens();
    grammar.collapse_groups();
    grammar.recover_leading_binary_operators();
    let mut parsed = grammar.parse_expression();
    grammar.recover_trailing_tokens();
    if let Some(requested) = parsed
        .as_ref()
        .and_then(|node| node.dedup_key.allocation_failure())
    {
        grammar.allocation_failure("query syntax arena", requested);
    }
    if grammar.allocation_failure.is_none() {
        if let Some(node) = parsed.as_mut() {
            rewrite_parser_syntax(node);
            trim_parser_dropped(&mut node.query, &node.dedup_key);
            flatten_should_of_should(&mut node.query);
        }
        repair_root_all_negative(&mut parsed, &mut grammar);
    }
    let arena_query = grammar.allocation_failure.map_or_else(
        || parsed.map_or_else(Query::empty, |node| node.query),
        Query::arena_allocation_failure,
    );
    ParsedQuery {
        query: arena_query,
        explanation: classify_query(input),
        diagnostics: grammar.diagnostics,
        was_truncated: false,
    }
}

/// One ASCII token of exactly `bytes` length plus its quoted variant.
fn oversized_atom() -> String {
    "x".repeat(crate::grimoire::MAX_TERM_BYTES + 1)
}

#[test]
fn public_query_strings_cannot_carry_oversized_tokens() {
    let parser = parser();
    // A 70,000-byte single token exceeds the 65,530-byte admission bound,
    // but the public string cap (10,000 bytes) truncates long before the
    // admission rule can ever see it.
    let query = "x".repeat(70_000);
    let parsed = parser.parse(&query);
    assert!(parsed.was_truncated);
    assert!(
        parsed
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == QueryDiagnosticKind::Truncated)
    );
    fn max_term_bytes(query: &Query) -> usize {
        let mut maximum = 0;
        let mut stack = Vec::from([query.root_id()]);
        while let Some(node_id) = stack.pop() {
            match query.node(node_id) {
                QueryNode::Term { text, .. } => maximum = maximum.max(text.len()),
                QueryNode::Phrase { terms, .. } => {
                    maximum =
                        maximum.max(terms.iter().map(|term| term.text.len()).max().unwrap_or(0));
                }
                QueryNode::Boolean { clauses, .. } => {
                    stack.extend(clauses.iter().map(|clause| clause.query));
                }
                QueryNode::Boost { query, .. } => stack.push(*query),
                QueryNode::Empty
                | QueryNode::AllocationFailure { .. }
                | QueryNode::All
                | QueryNode::Range { .. }
                | QueryNode::Set { .. }
                | QueryNode::Glob { .. } => {}
            }
        }
        maximum
    }
    assert!(
        max_term_bytes(&parsed.query) <= MAX_QUERY_LENGTH,
        "public parser must never surface a token beyond the 10,000-byte cap"
    );
}

#[test]
fn oversized_standalone_and_phrase_atoms_lower_to_match_none() {
    let parser = parser();
    // Standalone oversized token: MatchNone.
    let standalone = parse_untruncated(&parser, &oversized_atom());
    assert!(
        standalone.query.is_empty(),
        "oversized standalone atom must lower to Query::Empty, got {:?}",
        standalone.query
    );
    // Oversized member of a quoted phrase: the whole phrase is MatchNone
    // (positions of surviving terms are retained; the atom still cannot
    // match because one required position is unsatisfiable).
    let phrase = parse_untruncated(&parser, &format!("\"cache {}\"", oversized_atom()));
    assert!(
        phrase.query.is_empty(),
        "phrase containing an oversized term must lower to Query::Empty, got {:?}",
        phrase.query
    );
}

/// The match-nothing lowering announces itself in the diagnostics stream.
///
/// Until `TermLengthLimit` existed, this branch returned `Query::Empty`
/// silently, so no consumer could tell a query that MATCHED nothing from
/// one that was LOWERED to match nothing. The control parses an admissible
/// term through the same path and asserts the diagnostic is absent, so a
/// version of this test that passed by finding a diagnostic on every query
/// would fail.
#[test]
fn an_oversized_term_lowering_announces_itself() {
    let parser = parser();
    let leak_probe = "x".repeat(64);

    for (label, query) in [
        ("standalone", oversized_atom()),
        ("phrase member", format!("\"cache {}\"", oversized_atom())),
    ] {
        let parsed = parse_untruncated(&parser, &query);
        assert!(
            parsed.query.is_empty(),
            "{label} must still lower to match-nothing"
        );
        let diagnostic = parsed
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.kind == QueryDiagnosticKind::TermLengthLimit)
            .unwrap_or_else(|| {
                panic!(
                    "{label} lowering must announce itself, got {:?}",
                    parsed
                        .diagnostics
                        .iter()
                        .map(|diagnostic| diagnostic.kind)
                        .collect::<Vec<_>>()
                )
            });
        assert!(
            diagnostic.byte_offset.is_some(),
            "{label} diagnostic must locate the atom it lowered"
        );
        assert!(
            diagnostic.fragment.is_none() && !diagnostic.message.contains(&leak_probe),
            "{label} diagnostic must not carry the oversized term as a payload"
        );
    }

    // Control: an admissible term takes the same analyze_field path and
    // emits no length diagnostic, so the assertions above are attributable
    // to the oversized term rather than to the path being noisy.
    let ordinary = parse_untruncated(&parser, "cache");
    assert!(!ordinary.query.is_empty(), "the control must still match");
    assert!(
        ordinary
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.kind != QueryDiagnosticKind::TermLengthLimit),
        "an admissible term must not report a length limit"
    );
}

#[test]
fn oversized_clauses_keep_boolean_sibling_semantics() {
    let parser = parser();
    // Required conjunction: the parser retains the Empty clause so the
    // scorer shorts the whole conjunction to MatchNone (argus proof).
    let conjunction = parse_untruncated(&parser, &format!("cache AND {}", oversized_atom()));
    let QueryNode::Boolean { clauses, .. } = conjunction.query.root() else {
        panic!(
            "conjunction must remain a Boolean, got {:?}",
            conjunction.query
        );
    };
    assert!(
        clauses.iter().any(|clause| clause.occur == Occur::Must
            && matches!(conjunction.query.node(clause.query), QueryNode::Empty)),
        "conjunction must retain Must(Empty) for the oversized operand: {clauses:?}"
    );
    assert!(
        clauses.iter().any(|clause| clause.occur == Occur::Must
            && !matches!(conjunction.query.node(clause.query), QueryNode::Empty)),
        "conjunction must retain the matchable sibling: {clauses:?}"
    );

    // Optional disjunction: the oversized operand lowers to an Empty
    // Should clause, which the scorer drops while the matchable sibling
    // determines results (argus proof).
    let disjunction = parse_untruncated(&parser, &format!("cache OR {}", oversized_atom()));
    let QueryNode::Boolean { clauses, .. } = disjunction.query.root() else {
        panic!(
            "disjunction must remain a Boolean, got {:?}",
            disjunction.query
        );
    };
    assert!(
        clauses.iter().any(|clause| clause.occur == Occur::Should
            && matches!(disjunction.query.node(clause.query), QueryNode::Empty)),
        "disjunction must retain Should(Empty) for the oversized operand: {clauses:?}"
    );

    // Negated oversized operand: the exclusion is vacuous (an oversized
    // token matches nothing), and the parser's all-negative repair
    // inserts the All sibling that preserves complement semantics.
    let negation = parse_untruncated(&parser, &format!("-{}", oversized_atom()));
    let QueryNode::Boolean { clauses, .. } = negation.query.root() else {
        panic!(
            "repaired negation must be a Boolean, got {:?}",
            negation.query
        );
    };
    assert!(
        clauses
            .iter()
            .any(|clause| matches!(negation.query.node(clause.query), QueryNode::All)),
        "negated oversized query must gain the All sibling: {clauses:?}"
    );
    assert!(
        clauses.iter().any(|clause| clause.occur == Occur::MustNot
            && matches!(negation.query.node(clause.query), QueryNode::Empty)),
        "negated oversized operand lowers to MustNot(Empty): {clauses:?}"
    );
}

// ==== AST canonicalization (bd-quill-duel-ast-dedup-7hsu) ====

fn term_clause(occur: Occur, field_id: u16, text: &str) -> BooleanClause {
    BooleanClause::new(
        occur,
        Query::term(Vec::from([QueryField::new(field_id, 1.0)]), text.to_owned()),
    )
}

fn range_clause(occur: Occur, field_id: u16, low: i64, high: i64) -> BooleanClause {
    BooleanClause::new(
        occur,
        Query::range(
            field_id,
            Bound::Included(QueryValue::I64(low)),
            Bound::Included(QueryValue::I64(high)),
        ),
    )
}

#[test]
fn arena_append_preserves_all_and_empty_sentinels() {
    let child = Query::boolean(
        Vec::from([
            BooleanClause::new(Occur::Should, Query::all()),
            BooleanClause::new(Occur::Should, Query::empty()),
            term_clause(Occur::Should, 1, "child"),
        ]),
        None,
    );
    let query = Query::boolean(
        Vec::from([
            term_clause(Occur::Should, 1, "root"),
            BooleanClause::new(Occur::Should, child),
        ]),
        Some(BooleanOperator::Or),
    );
    let QueryNode::Boolean { clauses, .. } = query.root() else {
        panic!("outer query stays Boolean")
    };
    let QueryNode::Boolean {
        clauses: child_clauses,
        ..
    } = query.node(clauses[1].query)
    else {
        panic!("nested query stays Boolean")
    };
    assert!(matches!(query.node(child_clauses[0].query), QueryNode::All));
    assert!(matches!(
        query.node(child_clauses[1].query),
        QueryNode::Empty
    ));
}

#[test]
fn canonicalize_dedups_must_not_and_preserves_distinct_exclusions() {
    let mut query = Query::boolean(
        Vec::from([
            term_clause(Occur::MustNot, 1, "alpha"),
            term_clause(Occur::MustNot, 1, "beta"),
            term_clause(Occur::MustNot, 1, "alpha"),
            term_clause(Occur::Must, 1, "gamma"),
        ]),
        None,
    );
    let report = canonicalize_query(&mut query);
    assert_eq!(report.must_not_duplicates_removed, 1);
    let QueryNode::Boolean { clauses, .. } = query.root() else {
        panic!("stays Boolean")
    };
    let must_not_terms: Vec<_> = clauses
        .iter()
        .filter(|clause| clause.occur == Occur::MustNot)
        .filter_map(|clause| match query.node(clause.query) {
            QueryNode::Term { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(must_not_terms, ["alpha", "beta"]);
}

#[test]
fn canonicalize_never_dedups_scoring_clauses_for_oracle_conformance() {
    // THE CAVEAT: (A OR A) scores A twice in the oracle's sum-union, so
    // scoring duplicates MUST survive canonicalization.
    let mut should_query = Query::boolean(
        Vec::from([
            term_clause(Occur::Should, 1, "alpha"),
            term_clause(Occur::Should, 1, "alpha"),
            term_clause(Occur::Should, 1, "alpha"),
        ]),
        Some(BooleanOperator::Or),
    );
    let report = canonicalize_query(&mut should_query);
    assert_eq!(report.must_not_duplicates_removed, 0);
    assert_eq!(report.filter_duplicates_removed, 0);
    let QueryNode::Boolean { clauses, .. } = should_query.root() else {
        panic!("stays Boolean")
    };
    assert_eq!(clauses.len(), 3, "every scoring duplicate is retained");

    // Scoring Must (Term) duplicates are retained for the same reason.
    let mut must_query = Query::boolean(
        Vec::from([
            term_clause(Occur::Must, 1, "alpha"),
            term_clause(Occur::Must, 1, "alpha"),
        ]),
        Some(BooleanOperator::And),
    );
    let _ = canonicalize_query(&mut must_query);
    let QueryNode::Boolean { clauses, .. } = must_query.root() else {
        panic!("stays Boolean")
    };
    assert_eq!(clauses.len(), 2, "scoring Must duplicates are retained");
}

#[test]
fn canonicalize_retains_all_positive_scorers() {
    let child = Query::boolean(
        Vec::from([
            range_clause(Occur::Must, 7, 10, 99),
            term_clause(Occur::Must, 1, "alpha"),
            range_clause(Occur::Must, 7, 10, 99),
            BooleanClause::new(Occur::Must, Query::all()),
            BooleanClause::new(Occur::Must, Query::all()),
        ]),
        None,
    );
    let mut query = Query::boost(child, 2.0);
    let report = canonicalize_query(&mut query);
    assert_eq!(report.filter_duplicates_removed, 0);
    let QueryNode::Boost {
        query: child,
        factor,
    } = query.root()
    else {
        panic!("stays boosted")
    };
    assert_eq!(factor.to_bits(), 2.0_f32.to_bits());
    let QueryNode::Boolean { clauses, .. } = query.node(*child) else {
        panic!("boosted child stays Boolean")
    };
    assert_eq!(clauses.len(), 5);
    assert_eq!(
        clauses
            .iter()
            .filter(|clause| matches!(query.node(clause.query), QueryNode::Range { .. }))
            .count(),
        2,
        "duplicate constant scorers retain both score contributions",
    );
    assert!(clauses.iter().any(
        |clause| matches!(query.node(clause.query), QueryNode::Term { text, .. } if text == "alpha")
    ));
    assert_eq!(
        clauses
            .iter()
            .filter(|clause| matches!(query.node(clause.query), QueryNode::All))
            .count(),
        2,
        "ancestor boosts can make duplicate All children score independently",
    );
}

#[test]
fn canonicalize_must_not_key_frames_distinct_string_bounds() {
    let first = Query::range(
        7,
        Bound::Included(QueryValue::Str("L".to_owned())),
        Bound::Included(QueryValue::Str("p\0\u{2}q".to_owned())),
    );
    let second = Query::range(
        7,
        Bound::Included(QueryValue::Str("L\0\u{2}p".to_owned())),
        Bound::Included(QueryValue::Str("q".to_owned())),
    );
    assert_ne!(canonical_sort_key(&first), canonical_sort_key(&second));

    let mut query = Query::boolean(
        Vec::from([
            BooleanClause::new(Occur::Must, Query::all()),
            BooleanClause::new(Occur::MustNot, first),
            BooleanClause::new(Occur::MustNot, second),
        ]),
        None,
    );
    let report = canonicalize_query(&mut query);
    assert_eq!(report.must_not_duplicates_removed, 0);
    let QueryNode::Boolean { clauses, .. } = query.root() else {
        panic!("stays Boolean")
    };
    assert_eq!(clauses.len(), 3);
}

#[test]
fn canonicalize_groups_occurrences_without_reordering_scoring_clauses() {
    let mut query = Query::boolean(
        Vec::from([
            term_clause(Occur::MustNot, 1, "zzz"),
            term_clause(Occur::Must, 9, "beta"),
            BooleanClause::new(
                Occur::Should,
                Query::boost(term_clause(Occur::Should, 3, "alpha").query, 16_777_216.0),
            ),
            term_clause(Occur::Must, 1, "alpha"),
            BooleanClause::new(
                Occur::Should,
                Query::boost(term_clause(Occur::Should, 3, "gamma").query, 1.0),
            ),
            BooleanClause::new(
                Occur::Should,
                Query::boost(term_clause(Occur::Should, 3, "beta").query, 0.5),
            ),
        ]),
        None,
    );
    let report = canonicalize_query(&mut query);
    assert_eq!(report.boolean_levels_sorted, 1);
    let QueryNode::Boolean { clauses, .. } = query.root() else {
        panic!("stays Boolean")
    };
    let occurs: Vec<_> = clauses.iter().map(|clause| clause.occur).collect();
    assert_eq!(
        occurs,
        [
            Occur::Must,
            Occur::Must,
            Occur::Should,
            Occur::Should,
            Occur::Should,
            Occur::MustNot,
        ]
    );
    let should_factors = clauses
        .iter()
        .filter(|clause| clause.occur == Occur::Should)
        .filter_map(|clause| match query.node(clause.query) {
            QueryNode::Boost { factor, .. } => Some(*factor),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        should_factors,
        [16_777_216.0, 1.0, 0.5],
        "score-bearing Should construction order is bit-observable",
    );
}

#[test]
fn canonicalize_recurses_and_is_idempotent() {
    let nested = Query::boolean(
        Vec::from([
            term_clause(Occur::MustNot, 2, "dup"),
            term_clause(Occur::MustNot, 2, "dup"),
            term_clause(Occur::Must, 2, "b"),
            term_clause(Occur::Must, 2, "a"),
        ]),
        None,
    );
    let mut query = Query::boolean(
        Vec::from([
            BooleanClause::new(Occur::Must, nested),
            term_clause(Occur::Must, 1, "outer"),
        ]),
        None,
    );
    let first = canonicalize_query(&mut query);
    assert_eq!(first.must_not_duplicates_removed, 1);
    assert!(first.boolean_levels_sorted >= 1);
    let after_first = query.clone();
    let second = canonicalize_query(&mut query);
    assert_eq!(query, after_first, "idempotent");
    assert_eq!(second.must_not_duplicates_removed, 0);
    assert_eq!(second.filter_duplicates_removed, 0);
    assert_eq!(second.boolean_levels_sorted, 0);
}

#[test]
fn canonicalize_sorts_glob_fields_and_preserves_scoring_expansions() {
    let mut glob = Query::glob(Vec::from([3, 1, 3, 2, 1]), "alp*".to_owned());
    let report = canonicalize_query(&mut glob);
    assert_eq!(report.glob_fields_canonicalized, 1);
    let QueryNode::Glob { field_ids, .. } = glob.root() else {
        panic!("stays Glob")
    };
    assert_eq!(field_ids, &[1, 1, 2, 3, 3]);

    let mut term = Query::term(
        Vec::from([
            QueryField::new(1, 1.0),
            QueryField::new(2, 2.0),
            QueryField::new(1, 1.0),
            QueryField::new(1, 2.0), // same id, different boost: kept
        ]),
        "alpha".to_owned(),
    );
    let report = canonicalize_query(&mut term);
    assert_eq!(report.duplicate_fields_removed, 0);
    let QueryNode::Term { fields, .. } = term.root() else {
        panic!("stays Term")
    };
    assert_eq!(fields.len(), 4);
    assert_eq!(fields[0], QueryField::new(1, 1.0));
    assert_eq!(fields[1], QueryField::new(2, 2.0));
    assert_eq!(fields[2], QueryField::new(1, 1.0));
    assert_eq!(fields[3], QueryField::new(1, 2.0));
}

#[test]
fn canonicalize_leaves_parse_fixtures_exact_when_already_canonical() {
    // A query that is already canonical is byte-stable (no report, no
    // reorder), so differential-pinned parser fixtures never drift.
    let mut query = Query::boolean(
        Vec::from([
            term_clause(Occur::Must, 1, "alpha"),
            term_clause(Occur::Should, 3, "beta"),
        ]),
        Some(BooleanOperator::And),
    );
    let original = query.clone();
    let report = canonicalize_query(&mut query);
    assert_eq!(query, original);
    assert_eq!(report.boolean_levels_sorted, 0);
}
