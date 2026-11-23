#![allow(clippy::wildcard_imports)]

use std::ops::Bound;

use serde_json::{Value, json};

pub(super) use super::*;
use crate::schema::{
    Analyzer, Analyzer as AnalyzerKind, CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, FieldDescriptor,
    FieldKind, SchemaDescriptor,
};
#[cfg(feature = "tantivy-oracle")]
use crate::scribe::{CassAnalyzer, cass_generate_edge_ngrams};
use crate::scribe::{FrankensearchTokenizer, analyze_admitted};

mod cass_tests;
mod limits_and_canonical_tests;
mod parser_oracle_tests;

pub(super) const TYPED_FIELDS: [FieldDescriptor; 7] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
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
    FieldDescriptor {
        id: 3,
        name: "signed",
        kind: FieldKind::I64 {
            indexed: true,
            fast: false,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 4,
        name: "unsigned",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 5,
        name: "stored",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 6,
        name: "summary",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: false,
    },
];

pub(super) const TYPED_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "query-parser-typed-test",
    fields: &TYPED_FIELDS,
};

pub(super) fn parser() -> DefaultQueryParser {
    DefaultQueryParser::new(DEFAULT_SCHEMA).expect("default schema supports its parser")
}

/// Whether `node` is the default two-field expansion group of `text`:
/// `Boolean[Should content-term, Should title-term(^2)]` — the shape an
/// unfielded term lowers to under the structural mirror (bd-5o5z8),
/// matching tantivy's nested `(content OR title^2)` evaluation unit.
pub(super) fn is_expansion_group(query: &Query, node: QueryNodeId, text: &str) -> bool {
    let QueryNode::Boolean { clauses, .. } = query.node(node) else {
        return false;
    };
    clauses.len() == 2
        && clauses.iter().all(|clause| clause.occur == Occur::Should)
        && is_single_field_term(query, clauses[0].query, text, QueryField::new(1, 1.0))
        && is_single_field_term(
            query,
            clauses[1].query,
            text,
            QueryField::new(2, TITLE_BOOST),
        )
}

pub(super) fn is_single_field_term(
    query: &Query,
    node: QueryNodeId,
    text: &str,
    field: QueryField,
) -> bool {
    matches!(
        query.node(node),
        QueryNode::Term { fields, text: candidate }
            if candidate == text && fields.as_slice() == [field]
    )
}

pub(super) fn is_default_term(query: &Query, text: &str) -> bool {
    let QueryNode::Boolean {
        clauses,
        operator: None,
    } = query.root()
    else {
        return false;
    };
    clauses.len() == 2
        && clauses.iter().all(|clause| clause.occur == Occur::Should)
        && is_single_field_term(query, clauses[0].query, text, QueryField::new(1, 1.0))
        && is_single_field_term(
            query,
            clauses[1].query,
            text,
            QueryField::new(2, TITLE_BOOST),
        )
}

pub(super) fn term_leaf_count(query: &Query, text: &str) -> usize {
    let mut pending = Vec::new();
    assert!(pending.try_reserve_exact(query.node_count()).is_ok());
    pending.push(query.root_id());
    let mut count = 0;
    while let Some(node) = pending.pop() {
        match query.node(node) {
            QueryNode::Term {
                text: candidate, ..
            } => count += usize::from(candidate == text),
            QueryNode::Boolean { clauses, .. } => {
                pending.extend(clauses.iter().rev().map(|clause| clause.query));
            }
            QueryNode::Boost { query: child, .. } => pending.push(*child),
            QueryNode::Empty
            | QueryNode::AllocationFailure { .. }
            | QueryNode::All
            | QueryNode::Phrase { .. }
            | QueryNode::Range { .. }
            | QueryNode::Set { .. }
            | QueryNode::Glob { .. } => {}
        }
    }
    count
}

pub(super) fn query_subtree_eq(query: &Query, left: QueryNodeId, right: QueryNodeId) -> bool {
    let mut keys = Vec::new();
    assert!(keys.try_reserve_exact(query.nodes.len()).is_ok());
    for node in &query.nodes {
        keys.push(canonical_sort_key_for_node(node, &keys));
    }
    canonical_child_key(&keys, left) == canonical_child_key(&keys, right)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct EvalDoc {
    pub(super) id: &'static str,
    pub(super) content: &'static str,
    pub(super) title: &'static str,
}

#[cfg(feature = "tantivy-oracle")]
#[derive(Debug, Clone, Copy)]
pub(super) struct CassEvalDoc {
    pub(super) msg_idx: u64,
    pub(super) agent: &'static str,
    pub(super) workspace: Option<&'static str>,
    pub(super) created_at: Option<i64>,
    pub(super) title: &'static str,
    pub(super) content: &'static str,
    pub(super) source_id: &'static str,
    pub(super) origin_kind: &'static str,
}

fn analyzed_terms(text: &str) -> Vec<String> {
    let mut analyzer = FrankensearchTokenizer::default();
    let mut terms = Vec::new();
    let result = analyze_admitted(
        &mut analyzer,
        AnalyzerKind::FrankensearchDefault,
        text,
        &mut |token| terms.push(token.text.clone()),
    );
    assert!(result.is_ok(), "default analyzer is registered");
    terms
}

#[cfg(feature = "tantivy-oracle")]
fn cass_analyzed_terms(text: &str, analyzer_kind: AnalyzerKind) -> Vec<PositionedTerm> {
    let mut analyzer = CassAnalyzer::default();
    let mut terms = Vec::new();
    let result = analyze_admitted(&mut analyzer, analyzer_kind, text, &mut |token| {
        terms.push(PositionedTerm::new(token.position, token.text.clone()));
    });
    assert!(result.is_ok(), "CASS analyzer is registered");
    terms
}

#[cfg(feature = "tantivy-oracle")]
fn cass_field_text(doc: CassEvalDoc, field_id: u16) -> Option<&'static str> {
    match CASS_SEMANTIC_SCHEMA.fields.get(usize::from(field_id))?.name {
        "agent" => Some(doc.agent),
        "workspace" => doc.workspace,
        "title" => Some(doc.title),
        "content" => Some(doc.content),
        "source_id" => Some(doc.source_id),
        "origin_kind" => Some(doc.origin_kind),
        _ => None,
    }
}

#[cfg(feature = "tantivy-oracle")]
fn cass_field_terms(doc: CassEvalDoc, field_id: u16) -> Vec<PositionedTerm> {
    let Some(field) = CASS_SEMANTIC_SCHEMA.fields.get(usize::from(field_id)) else {
        return Vec::new();
    };
    match field.name {
        "title" => cass_analyzed_terms(doc.title, AnalyzerKind::CassHyphenNormalize),
        "content" => cass_analyzed_terms(doc.content, AnalyzerKind::CassHyphenNormalize),
        "title_prefix" => cass_analyzed_terms(
            &cass_generate_edge_ngrams(doc.title),
            AnalyzerKind::CassPrefixNormalize,
        ),
        "content_prefix" => cass_analyzed_terms(
            &cass_generate_edge_ngrams(doc.content),
            AnalyzerKind::CassPrefixNormalize,
        ),
        _ => cass_field_text(doc, field_id)
            .map_or_else(Vec::new, |text| vec![PositionedTerm::new(0, text)]),
    }
}

#[cfg(feature = "tantivy-oracle")]
fn cass_term_matches(doc: CassEvalDoc, field: QueryField, needle: &str) -> bool {
    cass_field_terms(doc, field.field_id)
        .iter()
        .any(|term| term.text == needle)
}

#[cfg(feature = "tantivy-oracle")]
fn cass_phrase_matches(doc: CassEvalDoc, field: QueryField, phrase: &[PositionedTerm]) -> bool {
    let haystack = cass_field_terms(doc, field.field_id);
    let Some(first) = phrase.first() else {
        return false;
    };
    haystack.iter().any(|candidate| {
        candidate.text == first.text
            && phrase.iter().all(|needle| {
                let position = candidate
                    .position
                    .saturating_add(needle.position.saturating_sub(first.position));
                haystack
                    .iter()
                    .any(|term| term.position == position && term.text == needle.text)
            })
    })
}

#[cfg(feature = "tantivy-oracle")]
fn cass_bound_matches(value: i64, lower: &Bound<QueryValue>, upper: &Bound<QueryValue>) -> bool {
    let lower_matches = match lower {
        Bound::Included(QueryValue::I64(lower)) => value >= *lower,
        Bound::Excluded(QueryValue::I64(lower)) => value > *lower,
        Bound::Unbounded => true,
        Bound::Included(_) | Bound::Excluded(_) => false,
    };
    let upper_matches = match upper {
        Bound::Included(QueryValue::I64(upper)) => value <= *upper,
        Bound::Excluded(QueryValue::I64(upper)) => value < *upper,
        Bound::Unbounded => true,
        Bound::Included(_) | Bound::Excluded(_) => false,
    };
    lower_matches && upper_matches
}

#[cfg(feature = "tantivy-oracle")]
fn star_glob_matches(pattern: &str, text: &str) -> bool {
    let pattern = pattern.chars().collect::<Vec<_>>();
    let text = text.chars().collect::<Vec<_>>();
    let mut pattern_index = 0_usize;
    let mut text_index = 0_usize;
    let mut star_index = None;
    let mut star_match = 0_usize;
    while text_index < text.len() {
        if pattern.get(pattern_index) == text.get(text_index) {
            pattern_index += 1;
            text_index += 1;
        } else if pattern.get(pattern_index) == Some(&'*') {
            star_index = Some(pattern_index);
            pattern_index += 1;
            star_match = text_index;
        } else if let Some(star) = star_index {
            star_match += 1;
            text_index = star_match;
            pattern_index = star + 1;
        } else {
            return false;
        }
    }
    pattern[pattern_index..].iter().all(|ch| *ch == '*')
}

pub(super) fn evaluate_query(
    query: &Query,
    mut evaluate_leaf: impl FnMut(&QueryNode) -> bool,
) -> bool {
    enum EvalTask {
        Visit(QueryNodeId),
        Boolean(QueryNodeId),
    }

    let node_count = query.node_count();
    let mut pending = Vec::new();
    let mut values = Vec::new();
    assert!(
        pending
            .try_reserve_exact(node_count.saturating_mul(2))
            .is_ok()
    );
    assert!(values.try_reserve_exact(node_count).is_ok());
    pending.push(EvalTask::Visit(query.root_id()));
    while let Some(task) = pending.pop() {
        match task {
            EvalTask::Visit(node) => match query.node(node) {
                QueryNode::Boolean { clauses, .. } => {
                    pending.push(EvalTask::Boolean(node));
                    pending.extend(
                        clauses
                            .iter()
                            .rev()
                            .map(|clause| EvalTask::Visit(clause.query)),
                    );
                }
                QueryNode::Boost { query: child, .. } => {
                    pending.push(EvalTask::Visit(*child));
                }
                leaf => values.push(evaluate_leaf(leaf)),
            },
            EvalTask::Boolean(node) => {
                let QueryNode::Boolean { clauses, .. } = query.node(node) else {
                    return false;
                };
                let mut has_must = false;
                let mut should_count = 0_usize;
                let mut should_match = false;
                let mut rejected = false;
                for clause in clauses.iter().rev() {
                    let Some(child_matches) = values.pop() else {
                        return false;
                    };
                    match clause.occur {
                        Occur::Must => {
                            has_must = true;
                            rejected |= !child_matches;
                        }
                        Occur::Should => {
                            should_count += 1;
                            should_match |= child_matches;
                        }
                        Occur::MustNot => rejected |= child_matches,
                    }
                }
                values.push(!rejected && (has_must || (should_count != 0 && should_match)));
            }
        }
    }
    values.pop().unwrap_or(false)
}

#[cfg(feature = "tantivy-oracle")]
pub(super) fn cass_ast_matches(query: &Query, doc: CassEvalDoc) -> bool {
    evaluate_query(query, |node| match node {
        QueryNode::Empty | QueryNode::AllocationFailure { .. } => false,
        QueryNode::All => true,
        QueryNode::Term { fields, text } => fields
            .iter()
            .copied()
            .any(|field| cass_term_matches(doc, field, text)),
        QueryNode::Phrase { fields, terms, .. } => fields
            .iter()
            .copied()
            .any(|field| cass_phrase_matches(doc, field, terms)),
        QueryNode::Range {
            field_id,
            lower,
            upper,
        } if CASS_SEMANTIC_SCHEMA.fields[usize::from(*field_id)].name == "created_at" => doc
            .created_at
            .is_some_and(|value| cass_bound_matches(value, lower, upper)),
        QueryNode::Range { .. } => false,
        QueryNode::Set { field_id, values } => {
            cass_field_text(doc, *field_id).is_some_and(|actual| {
                values
                    .iter()
                    .any(|value| value == &QueryValue::Str(actual.to_owned()))
            })
        }
        QueryNode::Glob { field_ids, pattern } => field_ids.iter().copied().any(|field_id| {
            cass_field_terms(doc, field_id)
                .iter()
                .any(|term| star_glob_matches(pattern, &term.text))
        }),
        QueryNode::Boolean { .. } | QueryNode::Boost { .. } => false,
    })
}

fn field_text(doc: EvalDoc, field_id: u16) -> Option<&'static str> {
    match DEFAULT_SCHEMA.fields.get(usize::from(field_id))?.name {
        "id" => Some(doc.id),
        "content" => Some(doc.content),
        "title" => Some(doc.title),
        _ => None,
    }
}

fn field_term_matches(doc: EvalDoc, field: QueryField, needle: &str) -> bool {
    let Some(text) = field_text(doc, field.field_id) else {
        return false;
    };
    if DEFAULT_SCHEMA.fields[usize::from(field.field_id)].kind == FieldKind::Keyword {
        text == needle
    } else {
        analyzed_terms(text).iter().any(|term| term == needle)
    }
}

fn field_phrase_matches(
    doc: EvalDoc,
    field: QueryField,
    phrase: &[PositionedTerm],
    prefix: bool,
) -> bool {
    let Some(text) = field_text(doc, field.field_id) else {
        return false;
    };
    let haystack = analyzed_terms(text);
    let Some(first_position) = phrase.first().map(|term| term.position) else {
        return false;
    };
    haystack.iter().enumerate().any(|(start, _)| {
        phrase.iter().enumerate().all(|(index, term)| {
            let relative = term.position.saturating_sub(first_position);
            let Ok(relative) = usize::try_from(relative) else {
                return false;
            };
            let Some(candidate) = haystack.get(start.saturating_add(relative)) else {
                return false;
            };
            if prefix && index + 1 == phrase.len() {
                candidate.starts_with(&term.text)
            } else {
                candidate == &term.text
            }
        })
    })
}

fn string_bound_matches(value: &str, lower: &Bound<QueryValue>, upper: &Bound<QueryValue>) -> bool {
    let lower_matches = match lower {
        Bound::Included(QueryValue::Str(lower)) => value >= lower.as_str(),
        Bound::Excluded(QueryValue::Str(lower)) => value > lower.as_str(),
        Bound::Unbounded => true,
        Bound::Included(_) | Bound::Excluded(_) => false,
    };
    let upper_matches = match upper {
        Bound::Included(QueryValue::Str(upper)) => value <= upper.as_str(),
        Bound::Excluded(QueryValue::Str(upper)) => value < upper.as_str(),
        Bound::Unbounded => true,
        Bound::Included(_) | Bound::Excluded(_) => false,
    };
    lower_matches && upper_matches
}

pub(super) fn ast_matches(query: &Query, doc: EvalDoc) -> bool {
    evaluate_query(query, |node| match node {
        QueryNode::Empty | QueryNode::AllocationFailure { .. } => false,
        QueryNode::All => true,
        QueryNode::Term { fields, text } => fields
            .iter()
            .copied()
            .any(|field| field_term_matches(doc, field, text)),
        QueryNode::Phrase {
            fields,
            terms,
            prefix,
            ..
        } => fields
            .iter()
            .copied()
            .any(|field| field_phrase_matches(doc, field, terms, *prefix)),
        QueryNode::Range {
            field_id,
            lower,
            upper,
        } => field_text(doc, *field_id)
            .is_some_and(|value| string_bound_matches(value, lower, upper)),
        QueryNode::Set { field_id, values } => field_text(doc, *field_id).is_some_and(|value| {
            values
                .iter()
                .any(|candidate| candidate == &QueryValue::Str(value.to_owned()))
        }),
        QueryNode::Glob { field_ids, pattern } => field_ids
            .iter()
            .copied()
            .any(|field_id| field_text(doc, field_id).is_some_and(|value| value == pattern)),
        QueryNode::Boolean { .. } | QueryNode::Boost { .. } => false,
    })
}

fn field_name(schema: SchemaDescriptor, field_id: u16) -> &'static str {
    schema.fields[usize::from(field_id)].name
}

fn value_json(value: &QueryValue) -> Value {
    match value {
        QueryValue::I64(value) => json!(value),
        QueryValue::U64(value) => json!(value),
        QueryValue::Str(value) => json!(value),
    }
}

fn value_tag(value: &QueryValue) -> &'static str {
    match value {
        QueryValue::I64(_) => "I64",
        QueryValue::U64(_) => "U64",
        QueryValue::Str(_) => "Str",
    }
}

fn bound_json(bound: &Bound<QueryValue>) -> Value {
    match bound {
        Bound::Included(value) => {
            json!({ "bound": "Included", "value": value_json(value) })
        }
        Bound::Excluded(value) => {
            json!({ "bound": "Excluded", "value": value_json(value) })
        }
        Bound::Unbounded => json!({ "bound": "Unbounded" }),
    }
}

fn typed_tag<'a>(prefix: &'a str, values: impl IntoIterator<Item = &'a QueryValue>) -> String {
    let suffix = values.into_iter().next().map_or("Str", value_tag);
    format!("{prefix}{suffix}")
}

pub(super) fn fixture_json(query: &Query, occur: Option<Occur>) -> Value {
    fixture_json_for_schema(DEFAULT_SCHEMA, query, occur)
}

pub(super) fn fixture_json_for_schema(
    schema: SchemaDescriptor,
    query: &Query,
    occur: Option<Occur>,
) -> Value {
    enum JsonTask {
        Visit(QueryNodeId, Option<Occur>),
        BuildBoolean(QueryNodeId),
        BuildBoost(f32),
    }

    let mut pending = Vec::new();
    let mut values = Vec::new();
    assert!(
        pending
            .try_reserve_exact(query.node_count().saturating_mul(2))
            .is_ok()
    );
    assert!(values.try_reserve_exact(query.node_count()).is_ok());
    pending.push(JsonTask::Visit(query.root_id(), occur));
    while let Some(task) = pending.pop() {
        let JsonTask::Visit(node_id, occur) = task else {
            let JsonTask::BuildBoolean(node_id) = task else {
                let JsonTask::BuildBoost(factor) = task else {
                    return Value::Null;
                };
                let Some(child) = values.pop() else {
                    return Value::Null;
                };
                values.push(json!({ "type": "Boost", "factor": factor, "query": child }));
                continue;
            };
            let QueryNode::Boolean { clauses, operator } = query.node(node_id) else {
                return Value::Null;
            };
            let mut children = Vec::new();
            assert!(children.try_reserve_exact(clauses.len()).is_ok());
            for clause in clauses.iter().rev() {
                let Some(child) = values.pop() else {
                    return Value::Null;
                };
                children.push(json!({
                    "occur": match clause.occur {
                        Occur::Must => "Must",
                        Occur::Should => "Should",
                        Occur::MustNot => "MustNot",
                    },
                    "query": child,
                }));
            }
            children.reverse();
            let mut value = json!({ "type": "Boolean", "children": children });
            if let Some(operator) = operator {
                value["operator"] = json!(match operator {
                    BooleanOperator::And => "AND",
                    BooleanOperator::Or => "OR",
                });
            }
            values.push(value);
            continue;
        };
        let value = match query.node(node_id) {
            QueryNode::Empty => json!({ "type": "Empty" }),
            QueryNode::AllocationFailure { requested } => {
                json!({ "type": "AllocationFailure", "requested": requested })
            }
            QueryNode::All => json!({ "type": "All" }),
            QueryNode::Term { fields, text } => {
                let fields = fields
                    .iter()
                    .map(|field| {
                        json!({
                            "name": field_name(schema, field.field_id),
                            "boost": field.boost,
                        })
                    })
                    .collect::<Vec<_>>();
                let mut value = json!({ "type": "Term", "text": text, "fields": fields });
                if occur == Some(Occur::MustNot) {
                    value["score"] = json!(0.0);
                }
                value
            }
            QueryNode::Phrase {
                fields,
                terms,
                slop,
                prefix,
            } => {
                let fields = fields
                    .iter()
                    .map(|field| {
                        json!({
                            "name": field_name(schema, field.field_id),
                            "boost": field.boost,
                        })
                    })
                    .collect::<Vec<_>>();
                let terms = terms
                    .iter()
                    .map(|term| term.text.as_str())
                    .collect::<Vec<_>>();
                let mut value = json!({
                    "type": if *prefix { "PhrasePrefix" } else { "Phrase" },
                    "terms": terms,
                    "slop": slop,
                    "fields": fields,
                });
                if occur == Some(Occur::MustNot) {
                    value["score"] = json!(0.0);
                }
                value
            }
            QueryNode::Boolean { clauses, .. } => {
                pending.push(JsonTask::BuildBoolean(node_id));
                pending.extend(
                    clauses
                        .iter()
                        .rev()
                        .map(|clause| JsonTask::Visit(clause.query, Some(clause.occur))),
                );
                continue;
            }
            QueryNode::Range {
                field_id,
                lower,
                upper,
            } => {
                let values = [lower, upper].into_iter().filter_map(|bound| match bound {
                    Bound::Included(value) | Bound::Excluded(value) => Some(value),
                    Bound::Unbounded => None,
                });
                let mut value = json!({
                    "type": typed_tag("Range", values),
                    "field": field_name(schema, *field_id),
                    "lower": bound_json(lower),
                    "upper": bound_json(upper),
                });
                if schema == CASS_SEMANTIC_SCHEMA {
                    value["matched_score"] = json!(1.0);
                }
                value
            }
            QueryNode::Set { field_id, values } => json!({
                "type": typed_tag("Set", values),
                "field": field_name(schema, *field_id),
                "values": values.iter().map(value_json).collect::<Vec<_>>(),
            }),
            QueryNode::Glob { field_ids, pattern } => json!({
                "type": "Glob",
                "pattern": pattern,
                "fields": field_ids
                    .iter()
                    .map(|field_id| field_name(schema, *field_id))
                    .collect::<Vec<_>>(),
            }),
            QueryNode::Boost {
                query: child,
                factor,
            } => {
                pending.push(JsonTask::BuildBoost(*factor));
                pending.push(JsonTask::Visit(*child, occur));
                continue;
            }
        };
        values.push(value);
    }
    values.pop().unwrap_or(Value::Null)
}
