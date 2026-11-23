use std::ops::Bound;

use crate::schema::{Analyzer as AnalyzerKind, FieldDescriptor, FieldKind};
use crate::scribe::{FrankensearchTokenizer, analyze_admitted};

use super::ast::{BooleanClause, Occur, PositionedTerm, Query, QueryField, QueryValue};
use super::canonical::{QueryDiagnostic, QueryDiagnosticKind};
use super::lexer::{LexToken, operator_atom, trim_typed_literal};
use super::rewrite::{
    combine_and, combine_or, merge_field_queries, negative_boolean_dedup_key,
    wrap_direct_negative_or_operand,
};
use super::syntax::{
    AtomToken, DefaultQueryParser, RangeToken, SetToken, SyntaxKey, SyntaxNode, atom_cache_key,
    atom_dedup_key, boost_syntax_key, emit_diagnostic, empty_query_for_syntax,
    mark_syntax_lowered_away, parse_boost_factor, range_dedup_key, set_dedup_key, take_syntax_key,
};
use super::{TITLE_BOOST, TITLE_FIELD_NAME};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ParsedNode {
    pub(super) query: Query,
    pub(super) occur: Option<Occur>,
    pub(super) from_not: bool,
    pub(super) dedup_key: SyntaxKey,
}

pub(super) struct Grammar {
    pub(super) parser: DefaultQueryParser,
    pub(super) tokens: Vec<LexToken>,
    pub(super) cursor: usize,
    pub(super) diagnostics: Vec<QueryDiagnostic>,
    pub(super) lowered_atoms: Vec<(String, Option<Query>)>,
    pub(super) field_scopes: Vec<String>,
    pub(super) allocation_failure: Option<usize>,
}

impl Grammar {
    fn allocation_node(&mut self, resource: &'static str, count: usize) -> ParsedNode {
        self.allocation_failure(resource, count);
        ParsedNode {
            query: Query::arena_allocation_failure(count),
            occur: None,
            from_not: false,
            dedup_key: SyntaxKey::empty_boolean(),
        }
    }

    pub(super) fn allocation_failure(&mut self, resource: &'static str, count: usize) {
        if self.allocation_failure.is_none() {
            self.allocation_failure = Some(count);
        }
        self.drop_with_diagnostic(
            QueryDiagnosticKind::AllocationFailure,
            &format!("could not reserve {count} entries for {resource}"),
            self.peek().map_or(0, LexToken::byte_offset),
            None,
        );
    }

    pub(super) fn lower_primary_tokens(&mut self) {
        let token_count = self.tokens.len();
        let group_count = self
            .tokens
            .iter()
            .filter(|token| matches!(token, LexToken::LeftParen { .. }))
            .count();
        let mut scoped_groups = Vec::new();
        if scoped_groups.try_reserve_exact(group_count).is_err()
            || self.field_scopes.try_reserve_exact(group_count).is_err()
            || self.lowered_atoms.try_reserve_exact(token_count).is_err()
        {
            self.allocation_failure("query parser scratch", token_count.max(group_count));
            self.tokens.clear();
            return;
        }

        self.cursor = 0;
        while let Some(token) = self.peek().cloned() {
            match token {
                LexToken::LeftParen { field, .. } => {
                    let scoped = field.is_some();
                    if let Some(field) = field {
                        self.field_scopes.push(field);
                    }
                    scoped_groups.push(scoped);
                    self.cursor += 1;
                }
                LexToken::RightParen { .. } => {
                    if scoped_groups.pop() == Some(true) {
                        let _ = self.field_scopes.pop();
                    }
                    self.cursor += 1;
                }
                LexToken::Atom(_)
                | LexToken::Range(_)
                | LexToken::Set(_)
                | LexToken::UnsupportedRegex { .. }
                | LexToken::Dropped(_) => {
                    let token_index = self.cursor;
                    let byte_offset = token.byte_offset();
                    let lowered = self.parse_primary();
                    self.tokens[token_index] =
                        lowered.map_or(LexToken::Dropped(byte_offset), |node| LexToken::Lowered {
                            node,
                            byte_offset,
                        });
                }
                LexToken::And(_)
                | LexToken::Or(_)
                | LexToken::Not(_)
                | LexToken::Plus(_)
                | LexToken::Minus(_)
                | LexToken::Lowered { .. } => self.cursor += 1,
            }
        }
        self.cursor = 0;
        self.field_scopes.clear();
    }

    fn push_group_token(&mut self, tokens: &mut Vec<LexToken>, token: LexToken) -> bool {
        if tokens.len() == tokens.capacity() && tokens.try_reserve(1).is_err() {
            self.allocation_failure("query group tokens", tokens.len().saturating_add(1));
            return false;
        }
        tokens.push(token);
        true
    }

    fn parse_flat_tokens(&mut self, tokens: Vec<LexToken>) -> Option<ParsedNode> {
        let saved_tokens = std::mem::replace(&mut self.tokens, tokens);
        let saved_cursor = std::mem::replace(&mut self.cursor, 0);
        self.recover_leading_binary_operators();
        let parsed = self.parse_expression();
        self.recover_trailing_tokens();
        self.tokens = saved_tokens;
        self.cursor = saved_cursor;
        parsed
    }

    fn collapsed_group_token(
        &mut self,
        tokens: Vec<LexToken>,
        byte_offset: usize,
        boost: Option<&str>,
        close_offset: usize,
    ) -> LexToken {
        let Some(mut node) = self.parse_flat_tokens(tokens) else {
            return LexToken::Dropped(byte_offset);
        };
        node.query = self.apply_raw_boost(node.query, boost, close_offset, None);
        node.dedup_key = boost_syntax_key(node.dedup_key, boost);
        LexToken::Lowered { node, byte_offset }
    }

    pub(super) fn collapse_groups(&mut self) {
        struct GroupFrame {
            tokens: Vec<LexToken>,
            byte_offset: usize,
        }

        let input = std::mem::take(&mut self.tokens);
        let group_count = input
            .iter()
            .filter(|token| matches!(token, LexToken::LeftParen { .. }))
            .count();
        let mut frames = Vec::new();
        if frames.try_reserve_exact(group_count).is_err() {
            self.allocation_failure("query group frames", group_count);
            return;
        }
        let mut current = Vec::new();
        for token in input {
            match token {
                LexToken::LeftParen { byte_offset, .. } => {
                    frames.push(GroupFrame {
                        tokens: current,
                        byte_offset,
                    });
                    current = Vec::new();
                }
                LexToken::RightParen { byte_offset, boost } => {
                    let Some(frame) = frames.pop() else {
                        if !self.push_group_token(
                            &mut current,
                            LexToken::RightParen { byte_offset, boost },
                        ) {
                            self.tokens.clear();
                            return;
                        }
                        continue;
                    };
                    let token = self.collapsed_group_token(
                        current,
                        frame.byte_offset,
                        boost.as_deref(),
                        byte_offset,
                    );
                    current = frame.tokens;
                    if !self.push_group_token(&mut current, token) {
                        self.tokens.clear();
                        return;
                    }
                }
                token => {
                    if !self.push_group_token(&mut current, token) {
                        self.tokens.clear();
                        return;
                    }
                }
            }
        }
        while let Some(frame) = frames.pop() {
            self.syntax_diagnostic_at(
                "syntax recovery: missing closing parenthesis",
                frame.byte_offset,
            );
            let token =
                self.collapsed_group_token(current, frame.byte_offset, None, frame.byte_offset);
            current = frame.tokens;
            if !self.push_group_token(&mut current, token) {
                self.tokens.clear();
                return;
            }
        }
        self.tokens = current;
        self.cursor = 0;
    }

    pub(super) fn recover_leading_binary_operators(&mut self) {
        while matches!(self.peek(), Some(LexToken::And(_) | LexToken::Or(_))) {
            let offset = self.peek().map_or(0, LexToken::byte_offset);
            self.cursor += 1;
            self.syntax_diagnostic_at("syntax recovery: leading Boolean operator dropped", offset);
        }
    }

    fn recover_operator_as_operand(&mut self) -> bool {
        let Some((raw, offset)) = self.peek().and_then(|token| match token {
            LexToken::And(offset) => Some(("AND", *offset)),
            LexToken::Or(offset) => Some(("OR", *offset)),
            _ => None,
        }) else {
            return false;
        };
        self.tokens[self.cursor] = operator_atom(raw, offset);
        self.syntax_diagnostic_at(
            "syntax recovery: repeated Boolean operator retained as a literal",
            offset,
        );
        true
    }

    pub(super) fn parse_expression(&mut self) -> Option<ParsedNode> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Option<ParsedNode> {
        let mut top_nodes = Vec::new();
        let mut top_operands = 0_usize;
        let mut run_nodes = Vec::new();
        let mut run_operands = 0_usize;
        let mut run_explicit = false;
        let mut joined_from_explicit_or = false;
        let node_bound = self.tokens.len();
        if top_nodes.try_reserve_exact(node_bound).is_err()
            || run_nodes.try_reserve_exact(node_bound).is_err()
        {
            return Some(self.allocation_node("Boolean parser nodes", node_bound));
        }

        loop {
            if !self.peek().is_some_and(LexToken::can_start_operand) {
                break;
            }
            run_operands += 1;
            let mut node = self.parse_and();
            let joined_to_explicit_or = matches!(self.peek(), Some(LexToken::Or(_)));
            if let Some(node) = node.as_mut()
                && node.occur == Some(Occur::MustNot)
                && (joined_from_explicit_or || joined_to_explicit_or)
            {
                wrap_direct_negative_or_operand(node);
            }
            if let Some(node) = node {
                run_nodes.push(node);
            }
            if let Some(LexToken::Or(offset)) = self.peek() {
                let offset = *offset;
                self.cursor += 1;
                if !self.peek().is_some_and(LexToken::can_start_operand)
                    && !self.recover_operator_as_operand()
                {
                    self.syntax_diagnostic_at("syntax recovery: OR has no right operand", offset);
                    break;
                }
                run_explicit = true;
                joined_from_explicit_or = true;
            } else if !self.peek().is_some_and(LexToken::can_start_operand) {
                break;
            } else {
                joined_from_explicit_or = false;
                Self::flush_or_run(
                    &mut top_nodes,
                    &mut top_operands,
                    &mut run_nodes,
                    &mut run_operands,
                    &mut run_explicit,
                );
                if run_nodes.try_reserve_exact(node_bound).is_err() {
                    return Some(self.allocation_node("Boolean parser run", node_bound));
                }
            }
        }

        if top_operands == 0 {
            // The whole chain is one run — a pure explicit-OR chain, a pure
            // implicit chain, or a single operand. Keep the historical flat
            // combination, including its same-level duplicate-retention rule.
            return combine_or(run_nodes, run_operands, run_explicit);
        }
        Self::flush_or_run(
            &mut top_nodes,
            &mut top_operands,
            &mut run_nodes,
            &mut run_operands,
            &mut run_explicit,
        );
        combine_or(top_nodes, top_operands, false)
    }

    /// Close one explicit-`OR` run at an implicit-adjacency boundary.
    ///
    /// The pinned grammar nests an explicit `OR` expression as its own node
    /// beneath the implicit disjunction, so its operands live one level below
    /// operands joined by plain adjacency — and the duplicate-retention
    /// rewrite, which only compares children of the SAME boolean level, never
    /// dedups across that boundary. Flattening the whole mixed chain instead
    /// made Quill drop a trailing operand that duplicates an OR-run member,
    /// scoring exactly one clause short of the oracle (bd-htcun, found by the
    /// bd-bsjw campaign as a clean 2.0x score deficit on
    /// `slices OR title:Nested AND reused slices`).
    fn flush_or_run(
        top_nodes: &mut Vec<ParsedNode>,
        top_operands: &mut usize,
        run_nodes: &mut Vec<ParsedNode>,
        run_operands: &mut usize,
        run_explicit: &mut bool,
    ) {
        if *run_operands == 0 && run_nodes.is_empty() {
            return;
        }
        *top_operands += 1;
        let nodes = std::mem::take(run_nodes);
        let operands = std::mem::replace(run_operands, 0);
        let explicit = std::mem::replace(run_explicit, false);
        if explicit {
            if let Some(node) = combine_or(nodes, operands, true) {
                top_nodes.push(node);
            }
        } else {
            // An adjacency boundary closes a run immediately, so a run
            // without an explicit OR is a single implicit operand.
            top_nodes.extend(nodes);
        }
    }

    fn parse_and(&mut self) -> Option<ParsedNode> {
        let mut nodes = Vec::new();
        let node_bound = self.tokens.len();
        if nodes.try_reserve_exact(node_bound).is_err() {
            return Some(self.allocation_node("conjunction parser nodes", node_bound));
        }
        let mut syntactic_operands = 1;
        // bd-quill-shipping-conformance-parse-split-w7bsu: a `NOT` operand of
        // an explicit `AND` keeps the `MustNot` occurrence `parse_unary` gave
        // it, so `NOT A AND B` lowers to `[(MustNot, A), (Must, B)]` and means
        // "B but not A". It previously went through `wrap_not_for_and`, which
        // re-wrapped it as a POSITIVE clause holding only a `MustNot` — a
        // clause with no positive term, which matches nothing and emptied the
        // whole conjunction. See the repair note on `parse_and` below.
        if let Some(node) = self.parse_unary() {
            nodes.push(node);
        }
        let mut explicit_and = false;
        while let Some(LexToken::And(offset)) = self.peek() {
            let offset = *offset;
            self.cursor += 1;
            if !self.peek().is_some_and(LexToken::can_start_operand)
                && !self.recover_operator_as_operand()
            {
                self.syntax_diagnostic_at("syntax recovery: AND has no right operand", offset);
                break;
            }
            explicit_and = true;
            syntactic_operands += 1;
            if let Some(node) = self.parse_unary() {
                nodes.push(node);
            }
        }
        if explicit_and {
            combine_and(nodes, syntactic_operands)
        } else {
            nodes.pop()
        }
    }

    fn parse_unary(&mut self) -> Option<ParsedNode> {
        let mut occur = None;
        let mut unary_offset = None;
        let mut not_count = 0_usize;
        while let Some(token) = self.peek() {
            match token {
                LexToken::Plus(offset) => {
                    unary_offset = Some(*offset);
                    occur = Some(Occur::Must);
                    self.cursor += 1;
                }
                LexToken::Minus(offset) => {
                    unary_offset = Some(*offset);
                    occur = Some(Occur::MustNot);
                    self.cursor += 1;
                }
                LexToken::Not(offset) => {
                    unary_offset = Some(*offset);
                    not_count += 1;
                    self.cursor += 1;
                }
                _ => break,
            }
        }

        let Some(mut node) = self.parse_primary() else {
            if let Some(offset) = unary_offset {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::DroppedFragment,
                    "unparseable unary fragment dropped",
                    offset,
                    None,
                );
            }
            return None;
        };
        for _ in 1..not_count {
            let dedup_key = take_syntax_key(&mut node.dedup_key);
            node.query = Query::boolean(
                Vec::from([BooleanClause::new(Occur::MustNot, node.query)]),
                None,
            );
            node.dedup_key = negative_boolean_dedup_key(dedup_key);
        }
        if not_count != 0 {
            // A `NOT` stacked with an explicit `+`/`-` prefix (`NOT -x`) sits
            // outside the contract grammar's single-prefix fragment rule; the
            // pinned oracle collapses the stack to ONE exclusion rather than
            // composing a double negation (bd-251nt, found by the bd-bsjw
            // structure-aware campaign; oracle behavior pinned by
            // stacked_negation_prefixes_pin_oracle_semantics in the lexical
            // crate). `NOT +x` is unprobed on the oracle and takes the same
            // collapse; revisit with a probe before relying on it.
            node.occur = Some(Occur::MustNot);
            node.from_not = true;
        } else if let Some(occur) = occur {
            node.occur = Some(occur);
        }
        Some(node)
    }

    fn parse_primary(&mut self) -> Option<ParsedNode> {
        let token = self.tokens.get(self.cursor)?.clone();
        let token_offset = token.byte_offset();
        match token {
            LexToken::Lowered { node, .. } => {
                self.cursor += 1;
                Some(node)
            }
            LexToken::Atom(mut atom) => {
                self.cursor += 1;
                if atom.field.is_none() {
                    atom.field = self.field_scopes.last().cloned();
                }
                let mut dedup_key = atom_dedup_key(&atom);
                let cache_key = atom_cache_key(&atom);
                if let Some(cache_key) = cache_key.as_ref() {
                    if let Some(index) = self
                        .lowered_atoms
                        .iter()
                        .position(|(candidate, _)| candidate == cache_key)
                    {
                        let cached = self.lowered_atoms[index].1.clone();
                        if cached.is_none() {
                            dedup_key = mark_syntax_lowered_away(dedup_key);
                        }
                        let query = cached.unwrap_or_else(|| empty_query_for_syntax(&dedup_key));
                        return Some(ParsedNode {
                            query,
                            occur: None,
                            from_not: false,
                            dedup_key,
                        });
                    }
                }
                let query = self.lower_atom(&atom);
                if let Some(cache_key) = cache_key {
                    self.lowered_atoms.push((cache_key, query.clone()));
                }
                if query.is_none() {
                    dedup_key = mark_syntax_lowered_away(dedup_key);
                }
                let query = query.unwrap_or_else(|| empty_query_for_syntax(&dedup_key));
                Some(ParsedNode {
                    query,
                    occur: None,
                    from_not: false,
                    dedup_key,
                })
            }
            LexToken::Range(mut token) => {
                self.cursor += 1;
                if token.field.is_none() {
                    token.field = self.field_scopes.last().cloned();
                }
                let mut dedup_key = range_dedup_key(&token);
                let lowered = self.lower_range(&token);
                if lowered.is_none() {
                    dedup_key = mark_syntax_lowered_away(dedup_key);
                }
                let query = lowered.unwrap_or_else(|| empty_query_for_syntax(&dedup_key));
                Some(ParsedNode {
                    query,
                    occur: None,
                    from_not: false,
                    dedup_key,
                })
            }
            LexToken::Set(mut token) => {
                self.cursor += 1;
                if token.field.is_none() {
                    token.field = self.field_scopes.last().cloned();
                }
                let mut dedup_key = set_dedup_key(&token);
                let lowered = self.lower_set(&token);
                if lowered.is_none() {
                    dedup_key = mark_syntax_lowered_away(dedup_key);
                }
                let query = lowered.unwrap_or_else(|| empty_query_for_syntax(&dedup_key));
                Some(ParsedNode {
                    query,
                    occur: None,
                    from_not: false,
                    dedup_key,
                })
            }
            LexToken::UnsupportedRegex {
                mut field,
                pattern,
                boost,
                ..
            } => {
                self.cursor += 1;
                if field.is_none() {
                    field = self.field_scopes.last().cloned();
                }
                let dedup_key = mark_syntax_lowered_away(boost_syntax_key(
                    SyntaxKey::leaf(SyntaxNode::Regex { field, pattern }),
                    boost.as_deref(),
                ));
                Some(ParsedNode {
                    query: empty_query_for_syntax(&dedup_key),
                    occur: None,
                    from_not: false,
                    dedup_key,
                })
            }
            LexToken::Dropped(_) => {
                self.cursor += 1;
                None
            }
            LexToken::LeftParen { byte_offset, .. } => {
                self.cursor += 1;
                self.syntax_diagnostic_at(
                    "syntax recovery: unresolved opening parenthesis dropped",
                    byte_offset,
                );
                None
            }
            LexToken::RightParen { .. } | LexToken::And(_) | LexToken::Or(_) => None,
            LexToken::Not(_) | LexToken::Plus(_) | LexToken::Minus(_) => {
                self.syntax_diagnostic_at(
                    "syntax recovery: unexpected unary token dropped",
                    token_offset,
                );
                None
            }
        }
    }

    fn lower_atom(&mut self, atom: &AtomToken) -> Option<Query> {
        if !atom.delimiter.is_quoted() && atom.raw == "*" {
            let Some(field_name) = atom.field.as_deref() else {
                return Some(self.apply_boost(Query::all(), atom));
            };
            let Some(descriptor) = self
                .parser
                .schema
                .fields
                .iter()
                .find(|field| field.name == field_name)
                .copied()
            else {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::UnknownField,
                    &format!("unknown field {field_name}; unsupported field fragment dropped"),
                    atom.byte_offset,
                    Some(atom.raw.clone()),
                );
                return None;
            };
            if matches!(
                descriptor.kind,
                FieldKind::I64 { fast: true, .. } | FieldKind::U64 { fast: true, .. }
            ) {
                return Some(self.apply_boost(
                    Query::range(descriptor.id, Bound::Unbounded, Bound::Unbounded),
                    atom,
                ));
            }
            self.drop_with_diagnostic(
                QueryDiagnosticKind::UnsupportedField,
                &format!(
                    "field {} does not support fast-field existence queries",
                    descriptor.name
                ),
                atom.byte_offset,
                None,
            );
            return None;
        }

        let mut fields = Vec::new();
        let requested_fields = if atom.field.is_some() {
            1
        } else {
            self.parser.default_fields.len()
        };
        if fields.try_reserve_exact(requested_fields).is_err() {
            self.allocation_failure("query atom fields", requested_fields);
            return Some(Query::arena_allocation_failure(requested_fields));
        }
        if let Some(field_name) = atom.field.as_deref() {
            let Some(descriptor) = self
                .parser
                .schema
                .fields
                .iter()
                .find(|field| field.name == field_name)
                .copied()
            else {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::UnknownField,
                    &format!("unknown field {field_name}; unsupported field fragment dropped"),
                    atom.byte_offset,
                    Some(atom.raw.clone()),
                );
                return None;
            };
            fields.push(QueryField::new(
                descriptor.id,
                if descriptor.name == TITLE_FIELD_NAME {
                    TITLE_BOOST
                } else {
                    1.0
                },
            ));
        } else {
            fields.extend_from_slice(&self.parser.default_fields);
        }

        let mut field_queries = Vec::new();
        if field_queries.try_reserve_exact(fields.len()).is_err() {
            self.allocation_failure("lowered field queries", fields.len());
            return Some(Query::arena_allocation_failure(fields.len()));
        }
        for field in fields {
            let Some(descriptor) = self.field_by_id(field.field_id) else {
                continue;
            };
            let Some(query) = self.analyze_field(descriptor, field, atom) else {
                continue;
            };
            field_queries.push(query);
        }
        let query = merge_field_queries(field_queries)?;
        Some(self.apply_boost(query, atom))
    }

    fn lower_range(&mut self, token: &RangeToken) -> Option<Query> {
        let descriptor = self.resolve_typed_field(token.field.as_deref(), token.byte_offset)?;
        let (lower, lower_invalid) = self.lower_bound(
            descriptor,
            &token.lower,
            token.lower_inclusive,
            token.byte_offset,
        );
        let (upper, upper_invalid) = self.lower_bound(
            descriptor,
            &token.upper,
            token.upper_inclusive,
            token.byte_offset,
        );
        if (lower_invalid && upper_invalid)
            || matches!((&lower, &upper), (Bound::Unbounded, Bound::Unbounded))
        {
            self.push_diagnostic(QueryDiagnostic {
                kind: QueryDiagnosticKind::DroppedFragment,
                message: "typed range without a usable bound dropped".to_owned(),
                byte_offset: Some(token.byte_offset),
                fragment: None,
            });
            return None;
        }
        let query = Query::range(descriptor.id, lower, upper);
        Some(self.apply_raw_boost(query, token.boost.as_deref(), token.byte_offset, None))
    }

    fn lower_set(&mut self, token: &SetToken) -> Option<Query> {
        let descriptor = self.resolve_typed_field(token.field.as_deref(), token.byte_offset)?;
        let mut values = Vec::new();
        if values.try_reserve_exact(token.values.len()).is_err() {
            self.allocation_failure("typed set values", token.values.len());
            return Some(Query::arena_allocation_failure(token.values.len()));
        }
        for raw in &token.values {
            match Self::lower_typed_value(descriptor, raw) {
                Some(value) if !values.contains(&value) => values.push(value),
                Some(_) => {}
                None => self.push_diagnostic(QueryDiagnostic {
                    kind: QueryDiagnosticKind::InvalidTypedValue,
                    message: format!(
                        "invalid typed set member for field {}; member dropped",
                        descriptor.name
                    ),
                    byte_offset: Some(token.byte_offset),
                    fragment: None,
                }),
            }
        }
        let query = Query::set(descriptor.id, values);
        Some(self.apply_raw_boost(query, token.boost.as_deref(), token.byte_offset, None))
    }

    fn resolve_typed_field(
        &mut self,
        field_name: Option<&str>,
        byte_offset: usize,
    ) -> Option<FieldDescriptor> {
        let Some(field_name) = field_name else {
            self.drop_with_diagnostic(
                QueryDiagnosticKind::DroppedFragment,
                "typed range or set requires a field scope",
                byte_offset,
                None,
            );
            return None;
        };
        let Some(descriptor) = self
            .parser
            .schema
            .fields
            .iter()
            .find(|field| field.name == field_name)
            .copied()
        else {
            self.drop_with_diagnostic(
                QueryDiagnosticKind::UnknownField,
                &format!("unknown field {field_name}; typed fragment dropped"),
                byte_offset,
                None,
            );
            return None;
        };
        let queryable = match descriptor.kind {
            FieldKind::Keyword => true,
            FieldKind::Text {
                analyzer: AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams,
                ..
            } => true,
            FieldKind::I64 { indexed, fast } | FieldKind::U64 { indexed, fast } => indexed || fast,
            FieldKind::StoredOnly | FieldKind::Text { .. } => false,
        };
        if !queryable {
            self.drop_with_diagnostic(
                QueryDiagnosticKind::UnsupportedField,
                &format!("field {} does not support typed queries", descriptor.name),
                byte_offset,
                None,
            );
            return None;
        }
        Some(descriptor)
    }

    fn lower_bound(
        &mut self,
        descriptor: FieldDescriptor,
        raw: &str,
        inclusive: bool,
        byte_offset: usize,
    ) -> (Bound<QueryValue>, bool) {
        if raw == "*" {
            return (Bound::Unbounded, false);
        }
        let Some(value) = Self::lower_typed_value(descriptor, raw) else {
            self.push_diagnostic(QueryDiagnostic {
                kind: QueryDiagnosticKind::InvalidTypedValue,
                message: format!(
                    "invalid typed range bound for field {}; bound made unbounded",
                    descriptor.name
                ),
                byte_offset: Some(byte_offset),
                fragment: None,
            });
            return (Bound::Unbounded, true);
        };
        if inclusive {
            (Bound::Included(value), false)
        } else {
            (Bound::Excluded(value), false)
        }
    }

    fn lower_typed_value(descriptor: FieldDescriptor, raw: &str) -> Option<QueryValue> {
        let raw = trim_typed_literal(raw);
        match descriptor.kind {
            FieldKind::I64 { indexed, fast } if indexed || fast => {
                raw.parse().ok().map(QueryValue::I64)
            }
            FieldKind::U64 { indexed, fast } if indexed || fast => {
                raw.parse().ok().map(QueryValue::U64)
            }
            FieldKind::Keyword => Some(QueryValue::Str(raw.to_owned())),
            FieldKind::Text {
                analyzer: AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams,
                ..
            } => {
                let FieldKind::Text {
                    analyzer: analyzer_kind,
                    ..
                } = descriptor.kind
                else {
                    return None;
                };
                let mut analyzer = FrankensearchTokenizer::default();
                let mut terms = Vec::new();
                let report = analyze_admitted(&mut analyzer, analyzer_kind, raw, &mut |token| {
                    terms.push(token.text.clone());
                })
                .ok()?;
                if report.oversized_tokens == 0 && terms.len() == 1 {
                    terms.pop().map(QueryValue::Str)
                } else {
                    None
                }
            }
            FieldKind::StoredOnly
            | FieldKind::Text { .. }
            | FieldKind::I64 { .. }
            | FieldKind::U64 { .. } => None,
        }
    }

    fn field_by_id(&self, field_id: u16) -> Option<FieldDescriptor> {
        self.parser
            .schema
            .fields
            .get(usize::from(field_id))
            .copied()
            .filter(|field| field.id == field_id)
    }

    fn analyze_field(
        &mut self,
        descriptor: FieldDescriptor,
        field: QueryField,
        atom: &AtomToken,
    ) -> Option<Query> {
        match descriptor.kind {
            FieldKind::Text {
                analyzer: AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams,
                ..
            } => {
                let FieldKind::Text {
                    analyzer: analyzer_kind,
                    ..
                } = descriptor.kind
                else {
                    return None;
                };
                let mut analyzer = FrankensearchTokenizer::default();
                let mut terms = Vec::new();
                let Ok(report) =
                    analyze_admitted(&mut analyzer, analyzer_kind, &atom.raw, &mut |token| {
                        terms.push(PositionedTerm::new(token.position, token.text.clone()));
                    })
                else {
                    self.drop_with_diagnostic(
                        QueryDiagnosticKind::UnsupportedField,
                        "default analyzer implementation is unavailable",
                        atom.byte_offset,
                        None,
                    );
                    return None;
                };
                if report.oversized_tokens != 0 {
                    // The fragment is deliberately omitted: it WOULD be the
                    // oversized term itself, which by definition exceeds the
                    // bound the fragment field is documented to respect, so
                    // carrying it would push a >MAX_TERM_BYTES payload into the
                    // diagnostics stream. The byte offset still locates it.
                    self.push_diagnostic(QueryDiagnostic {
                        kind: QueryDiagnosticKind::TermLengthLimit,
                        message:
                            "analyzed term exceeds the admitted term-length bound; branch lowered to match-nothing"
                                .to_owned(),
                        byte_offset: Some(atom.byte_offset),
                        fragment: None,
                    });
                    return Some(Query::empty());
                }
                match terms.len() {
                    0 => None,
                    1 if atom.prefix => {
                        self.push_diagnostic(QueryDiagnostic {
                            kind: QueryDiagnosticKind::DroppedFragment,
                            message: "phrase prefix requires at least two analyzed terms; field branch dropped"
                                .to_owned(),
                            byte_offset: Some(atom.byte_offset),
                            fragment: None,
                        });
                        None
                    }
                    1 => terms
                        .pop()
                        .map(|term| Query::term(Vec::from([field]), term.text)),
                    _ => Some(Query::phrase(
                        Vec::from([field]),
                        terms,
                        atom.slop,
                        atom.prefix,
                    )),
                }
            }
            FieldKind::Keyword if atom.prefix => {
                self.push_diagnostic(QueryDiagnostic {
                    kind: QueryDiagnosticKind::DroppedFragment,
                    message:
                        "phrase prefix requires at least two analyzed terms; field branch dropped"
                            .to_owned(),
                    byte_offset: Some(atom.byte_offset),
                    fragment: None,
                });
                None
            }
            FieldKind::Keyword => {
                (!atom.raw.is_empty()).then(|| Query::term(Vec::from([field]), atom.raw.clone()))
            }
            FieldKind::Text { .. } => {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::UnsupportedField,
                    &format!(
                        "field {} uses an analyzer unsupported by the default parser",
                        descriptor.name
                    ),
                    atom.byte_offset,
                    Some(atom.raw.clone()),
                );
                None
            }
            FieldKind::StoredOnly
            | FieldKind::I64 { indexed: false, .. }
            | FieldKind::U64 { indexed: false, .. }
            | FieldKind::I64 { indexed: true, .. }
            | FieldKind::U64 { indexed: true, .. } => {
                self.drop_with_diagnostic(
                    QueryDiagnosticKind::UnsupportedField,
                    &format!(
                        "field {} is not a text field supported by this parser",
                        descriptor.name
                    ),
                    atom.byte_offset,
                    Some(atom.raw.clone()),
                );
                None
            }
        }
    }

    fn apply_boost(&mut self, query: Query, atom: &AtomToken) -> Query {
        self.apply_raw_boost(
            query,
            atom.boost.as_deref(),
            atom.byte_offset,
            Some(atom.raw.clone()),
        )
    }

    fn apply_raw_boost(
        &mut self,
        query: Query,
        raw_boost: Option<&str>,
        byte_offset: usize,
        fragment: Option<String>,
    ) -> Query {
        let Some(raw_boost) = raw_boost else {
            return query;
        };
        match parse_boost_factor(raw_boost) {
            // Bit-exact for the same reason as the boost-chain lowering above:
            // an identity boost is dropped, and every other factor is kept.
            Some(factor) if factor.to_bits() == 1.0_f32.to_bits() => query,
            Some(factor) => Query::boost(query, factor),
            _ => {
                self.push_diagnostic(QueryDiagnostic {
                    kind: QueryDiagnosticKind::InvalidBoost,
                    message: format!("invalid boost {raw_boost:?}; recovered branch without boost"),
                    byte_offset: Some(byte_offset),
                    fragment,
                });
                query
            }
        }
    }

    fn peek(&self) -> Option<&LexToken> {
        self.tokens.get(self.cursor)
    }

    pub(super) fn push_diagnostic(&mut self, diagnostic: QueryDiagnostic) {
        emit_diagnostic(&mut self.diagnostics, diagnostic);
    }

    fn drop_with_diagnostic(
        &mut self,
        kind: QueryDiagnosticKind,
        message: &str,
        byte_offset: usize,
        fragment: Option<String>,
    ) {
        self.push_diagnostic(QueryDiagnostic {
            kind,
            message: message.to_owned(),
            byte_offset: Some(byte_offset),
            fragment,
        });
    }

    fn syntax_diagnostic_at(&mut self, message: &str, byte_offset: usize) {
        self.push_diagnostic(QueryDiagnostic {
            kind: QueryDiagnosticKind::SyntaxRecovery,
            message: message.to_owned(),
            byte_offset: Some(byte_offset),
            fragment: None,
        });
    }

    pub(super) fn recover_trailing_tokens(&mut self) {
        while let Some(token) = self.tokens.get(self.cursor) {
            let offset = token.byte_offset();
            self.cursor += 1;
            self.drop_with_diagnostic(
                QueryDiagnosticKind::DroppedFragment,
                "unparseable trailing fragment dropped",
                offset,
                None,
            );
        }
    }
}
