use super::ast::{
    BooleanClause, BooleanOperator, Occur, Query, QueryClause, QueryNode, QueryNodeId,
};
use super::canonical::{QueryDiagnostic, QueryDiagnosticKind};
use super::grammar::{Grammar, ParsedNode};
use super::syntax::{SyntaxKey, SyntaxNode, SyntaxNodeId, take_syntax_key};

pub(super) fn wrap_direct_negative_or_operand(node: &mut ParsedNode) {
    let dedup_key = take_syntax_key(&mut node.dedup_key);
    let query = std::mem::replace(&mut node.query, Query::empty());
    node.query = Query::boolean(Vec::from([BooleanClause::new(Occur::MustNot, query)]), None);
    node.occur = None;
    node.from_not = false;
    node.dedup_key = negative_boolean_dedup_key(dedup_key);
}

// `wrap_not_for_and` was REMOVED by
// bd-quill-shipping-conformance-parse-split-w7bsu. It used to re-wrap a `NOT`
// operand of an explicit `AND` into a positive clause holding only a `MustNot`,
// mirroring tantivy 0.26.1's lowering of `A AND NOT B`. That lowering is the
// defect bd-eeq0q measured: a clause with no positive term matches nothing, so
// the entire conjunction came back empty. `combine_boolean` already honours
// `node.occur`, and `parse_unary` already sets `MustNot` for a `NOT` operand,
// so deleting the wrapper is the whole repair.
//
// `wrap_direct_negative_or_operand` is deliberately NOT removed: it handles the
// `OR` side, where a bare negative operand really does need wrapping so it
// cannot become a positive alternative.

pub(super) fn negative_boolean_dedup_key(child: SyntaxKey) -> SyntaxKey {
    SyntaxKey::boolean(Vec::from([(Some(Occur::MustNot), child)]))
}

/// Mirror the pinned grammar's lowering: an optional (`Should`) clause whose
/// child is an all-optional boolean splices its children into the parent.
///
/// Duplicate retention already ran on the NESTED syntax (where the pinned
/// grammar keeps explicit-`OR` runs and parenthesised groups one level below
/// the implicit disjunction — bd-htcun), but the oracle then FLATTENS
/// pure-disjunctive nesting for execution, so score accumulation associates
/// left-to-right over one flat clause list. Executing the nested shape
/// instead diverges by ULPs on boosted mixes (campaign finding: 2-ULP
/// `RankMismatch` on `((small^3.0 OR when AND this reused) OR tracing NOT
/// generic)`). Flattening after dedup reproduces the oracle's association
/// exactly. A non-`Should` occurrence or a mixed-occurrence child is a
/// boundary, as is any non-boolean node (boosts included).
/// Collapse parser-drop residue without disturbing live Should nesting
/// (bd-1i4j4).
///
/// Tantivy preserves nested `Should` groups and evaluates each level with its
/// generic union, so the f32 accumulation association follows the parse tree;
/// merging that nesting away broke score-bit parity on every unfielded
/// multi-term shape. What must still collapse is DROP RESIDUE: a fragment the
/// parser deleted leaves an empty group behind, and the typed-refusal
/// surfaces key on the query collapsing to `Empty`. Post-order: an
/// `Occur::Should` clause whose child is `Empty` or an empty `Boolean` is
/// dropped, and a `Boolean` left with no clauses becomes `Empty`. `Must` and
/// `MustNot` clauses are never touched, and singleton groups are kept —
/// lowering's `scorer_union` pops a one-child union, so the wrapper never
/// constructs and cannot change association.
pub(super) fn collapse_degenerate_groups(query: &mut Query) {
    let mut pending = Vec::new();
    if pending.try_reserve_exact(query.node_count()).is_err() {
        return;
    }
    pending.push((query.root_id(), false));
    while let Some((node_id, visited)) = pending.pop() {
        let QueryNode::Boolean { clauses, .. } = query.node(node_id) else {
            continue;
        };
        if !visited {
            pending.push((node_id, true));
            pending.extend(clauses.iter().rev().map(|clause| (clause.query, false)));
            continue;
        }
        let is_degenerate = |query: &Query, node| match query.node(node) {
            QueryNode::Empty => true,
            QueryNode::Boolean { clauses, .. } => clauses.is_empty(),
            _ => false,
        };
        let retained_len = clauses.iter().fold(0_usize, |len, clause| {
            if clause.occur == Occur::Should && is_degenerate(query, clause.query) {
                len
            } else {
                len.saturating_add(1)
            }
        });
        if retained_len == clauses.len() {
            continue;
        }
        let mut retained = Vec::new();
        if retained.try_reserve_exact(retained_len).is_err() {
            continue;
        }
        for clause in clauses {
            if clause.occur == Occur::Should && is_degenerate(query, clause.query) {
                continue;
            }
            retained.push(*clause);
        }
        if let QueryNode::Boolean { clauses, .. } = query.node_mut(node_id) {
            *clauses = retained;
        }
    }
    // The flatten's OTHER retained duty (bd-htcun): a post-trim SINGLETON
    // `Should` group splices into its parent. Execution-neutral either way —
    // lowering's `scorer_union` pops a one-child union — but the parser-AST
    // hygiene laws (raw-identity dedup, diagnostics siblings) are pinned over
    // the spliced shape, and tantivy's own grammar rewrite performs the same
    // singleton transfer. Multi-child groups are NEVER spliced: their nesting
    // is the association the structural mirror exists to preserve.
    let mut pending = Vec::new();
    if pending.try_reserve_exact(query.node_count()).is_err() {
        return;
    }
    pending.push(query.root_id());
    while let Some(node_id) = pending.pop() {
        let QueryNode::Boolean { clauses, .. } = query.node(node_id) else {
            continue;
        };
        pending.extend(clauses.iter().map(|clause| clause.query));
        let mut spliced = Vec::new();
        if spliced.try_reserve_exact(clauses.len()).is_err() {
            continue;
        }
        let mut changed = false;
        for clause in clauses {
            if clause.occur == Occur::Should
                && let QueryNode::Boolean {
                    clauses: children, ..
                } = query.node(clause.query)
                && children.len() == 1
                && children[0].occur == Occur::Should
            {
                spliced.push(children[0]);
                changed = true;
            } else {
                spliced.push(*clause);
            }
        }
        if changed && let QueryNode::Boolean { clauses, .. } = query.node_mut(node_id) {
            *clauses = spliced;
        }
        // A Boolean holding exactly one `Should` clause is a pure wrapper:
        // a one-child union IS its child at lowering (`scorer_union` pops
        // it), and the flat-era flatten always erased it. Unwrapping keeps
        // the pinned root/sibling AST laws intact without ever touching
        // multi-child nesting.
        let QueryNode::Boolean { clauses, .. } = query.node(node_id) else {
            continue;
        };
        if clauses.len() == 1
            && clauses[0].occur == Occur::Should
            && matches!(query.node(clauses[0].query), QueryNode::Boolean { .. })
        {
            // Only a BOOLEAN child unwraps: the flat-era flatten spliced
            // Should-of-Should exclusively, so `B[Should=Range]`-style
            // wrappers around leaf nodes are pinned shapes and stay.
            let child = clauses[0].query;
            let replacement = query.node(child).clone();
            *query.node_mut(node_id) = replacement;
        }
    }
    if let QueryNode::Boolean { clauses, .. } = query.node(query.root_id())
        && clauses.is_empty()
    {
        *query = Query::empty();
    }
}

/// RETIRED from the scored parse (bd-5o5z8): tantivy preserves Should
/// nesting and evaluates it with generic unions, so the shipping pipeline
/// now runs `collapse_degenerate_groups` instead. Retained solely for the
/// tests-module legacy-pipeline replica that pins what the flat era did.
#[cfg(test)]
pub(super) fn flatten_should_of_should(query: &mut Query) {
    let mut pending = Vec::new();
    if pending.try_reserve_exact(query.node_count()).is_err() {
        return;
    }
    pending.push((query.root_id(), false));
    while let Some((node_id, visited)) = pending.pop() {
        let QueryNode::Boolean { clauses, .. } = query.node(node_id) else {
            continue;
        };
        if !visited {
            pending.push((node_id, true));
            pending.extend(clauses.iter().rev().map(|clause| (clause.query, false)));
            continue;
        }
        let flattened_len = clauses.iter().fold(0_usize, |len, clause| {
            if clause.occur == Occur::Should
                && let QueryNode::Boolean {
                    clauses: children, ..
                } = query.node(clause.query)
                && children.iter().all(|child| child.occur == Occur::Should)
            {
                len.saturating_add(children.len())
            } else {
                len.saturating_add(1)
            }
        });
        if flattened_len == clauses.len() {
            continue;
        }
        let mut flattened = Vec::new();
        if flattened.try_reserve_exact(flattened_len).is_err() {
            continue;
        }
        for clause in clauses {
            if clause.occur == Occur::Should
                && let QueryNode::Boolean {
                    clauses: children, ..
                } = query.node(clause.query)
                && children.iter().all(|child| child.occur == Occur::Should)
            {
                flattened.extend_from_slice(children);
            } else {
                flattened.push(*clause);
            }
        }
        if let QueryNode::Boolean { clauses, .. } = query.node_mut(node_id) {
            *clauses = flattened;
        }
    }
}

pub(super) fn combine_and(nodes: Vec<ParsedNode>, syntactic_operands: usize) -> Option<ParsedNode> {
    combine_boolean(
        nodes,
        Occur::Must,
        Some(BooleanOperator::And),
        syntactic_operands,
    )
}

pub(super) fn combine_or(
    nodes: Vec<ParsedNode>,
    syntactic_operands: usize,
    explicit_or: bool,
) -> Option<ParsedNode> {
    combine_boolean(
        nodes,
        Occur::Should,
        explicit_or.then_some(BooleanOperator::Or),
        syntactic_operands,
    )
}

fn combine_boolean(
    nodes: Vec<ParsedNode>,
    default_occur: Occur,
    operator: Option<BooleanOperator>,
    syntactic_operands: usize,
) -> Option<ParsedNode> {
    let requested = nodes.len();
    let mut clauses = Vec::new();
    let mut keys = Vec::new();
    if clauses.try_reserve_exact(requested).is_err() || keys.try_reserve_exact(requested).is_err() {
        return Some(ParsedNode {
            query: Query::arena_allocation_failure(requested),
            occur: None,
            from_not: false,
            dedup_key: SyntaxKey::empty_boolean(),
        });
    }
    for node in nodes {
        let syntax_occur = node.occur.or(match operator {
            Some(BooleanOperator::And) => Some(Occur::Must),
            Some(BooleanOperator::Or) => Some(Occur::Should),
            None => None,
        });
        let occur = node.occur.unwrap_or(default_occur);
        keys.push((syntax_occur, node.dedup_key));
        clauses.push(BooleanClause::new(occur, node.query));
    }
    if clauses.is_empty() {
        return None;
    }

    if clauses.len() == 1 && syntactic_operands == 1 && clauses[0].occur != Occur::MustNot {
        let clause = clauses.pop()?;
        let (_, dedup_key) = keys.pop()?;
        return Some(ParsedNode {
            query: clause.query,
            occur: None,
            from_not: false,
            dedup_key,
        });
    }

    let dedup_key = SyntaxKey::boolean(keys);
    let query = dedup_key.allocation_failure().map_or_else(
        || Query::boolean(clauses, operator),
        Query::arena_allocation_failure,
    );
    Some(ParsedNode {
        query,
        occur: None,
        from_not: false,
        dedup_key,
    })
}

/// Mirror `tantivy-query-grammar`'s parser-AST rewrite while the raw syntax
/// identity is still paired with the lowered query tree.
///
/// The pinned grammar recurses only through Boolean clauses, so duplicates
/// beneath a non-identity boost deliberately survive. At each visited Boolean
/// it retains the first exact `(Option<Occur>, child)` pair and then flattens a
/// singleton child whose outer occurrence is implicit, transferring the
/// child's explicit occurrence to the parent clause.
pub(super) fn rewrite_parser_syntax(node: &mut ParsedNode) {
    let query = &mut node.query;
    let syntax = &mut node.dedup_key;
    let mut pending = Vec::new();
    let task_count = query
        .node_count()
        .saturating_add(syntax.nodes.len().saturating_add(1));
    if pending.try_reserve_exact(task_count).is_err() {
        return;
    }
    pending.push((query.root_id(), SyntaxKey::root_id(), false));
    while let Some((query_id, syntax_id, visited)) = pending.pop() {
        if !visited {
            let (QueryNode::Boolean { clauses, .. }, SyntaxNode::Boolean(keys)) =
                (query.node(query_id), syntax.node(syntax_id))
            else {
                continue;
            };
            if clauses.len() != keys.len() {
                debug_assert_eq!(clauses.len(), keys.len());
                continue;
            }
            pending.push((query_id, syntax_id, true));
            pending.extend(
                clauses
                    .iter()
                    .zip(keys)
                    .rev()
                    .map(|(clause, (_, key))| (clause.query, *key, false)),
            );
            continue;
        }

        let (QueryNode::Boolean { clauses, .. }, SyntaxNode::Boolean(keys)) =
            (query.node(query_id), syntax.node(syntax_id))
        else {
            continue;
        };
        let mut keep = Vec::new();
        if keep.try_reserve_exact(keys.len()).is_err() {
            continue;
        }
        for (index, (occur, child)) in keys.iter().enumerate() {
            let duplicate = keys[..index].iter().any(|(prior_occur, prior_child)| {
                prior_occur == occur && syntax_subtree_eq(syntax, *prior_child, *child)
            });
            keep.push(!duplicate);
        }
        let retained = keep.iter().filter(|keep| **keep).count();
        if retained != keep.len() {
            let mut retained_keys = Vec::new();
            let mut retained_clauses = Vec::new();
            if retained_keys.try_reserve_exact(retained).is_err()
                || retained_clauses.try_reserve_exact(retained).is_err()
            {
                continue;
            }
            retained_keys.extend(
                keys.iter()
                    .zip(&keep)
                    .filter(|(_, keep)| **keep)
                    .map(|(key, _)| *key),
            );
            retained_clauses.extend(
                clauses
                    .iter()
                    .zip(&keep)
                    .filter(|(_, keep)| **keep)
                    .map(|(clause, _)| *clause),
            );
            if let SyntaxNode::Boolean(keys) = syntax.node_mut(syntax_id) {
                *keys = retained_keys;
            }
            if let QueryNode::Boolean { clauses, .. } = query.node_mut(query_id) {
                *clauses = retained_clauses;
            }
        }

        let (QueryNode::Boolean { clauses, .. }, SyntaxNode::Boolean(keys)) =
            (query.node(query_id), syntax.node(syntax_id))
        else {
            continue;
        };
        let mut replacements = Vec::new();
        if replacements.try_reserve_exact(keys.len()).is_err() {
            continue;
        }
        for ((outer_occur, child_key), outer_clause) in keys.iter().zip(clauses) {
            let replacement = if outer_occur.is_none()
                && let SyntaxNode::Boolean(inner_keys) = syntax.node(*child_key)
                && let QueryNode::Boolean {
                    clauses: inner_clauses,
                    ..
                } = query.node(outer_clause.query)
                && inner_keys.len() == 1
                && inner_clauses.len() == 1
            {
                Some((inner_keys[0], inner_clauses[0]))
            } else {
                None
            };
            replacements.push(replacement);
        }
        if let SyntaxNode::Boolean(keys) = syntax.node_mut(syntax_id) {
            for (key, replacement) in keys.iter_mut().zip(&replacements) {
                if let Some((inner_key, _)) = replacement {
                    *key = *inner_key;
                }
            }
        }
        if let QueryNode::Boolean { clauses, .. } = query.node_mut(query_id) {
            for (clause, replacement) in clauses.iter_mut().zip(replacements) {
                if let Some((_, inner_clause)) = replacement {
                    *clause = inner_clause;
                }
            }
        }
    }
}

fn syntax_subtree_eq(syntax: &SyntaxKey, left: SyntaxNodeId, right: SyntaxNodeId) -> bool {
    let mut pending = Vec::new();
    if pending
        .try_reserve_exact(syntax.nodes.len().saturating_add(1))
        .is_err()
    {
        return false;
    }
    pending.push((left, right));
    while let Some((left, right)) = pending.pop() {
        match (syntax.node(left), syntax.node(right)) {
            (SyntaxNode::LoweredAway(left), SyntaxNode::LoweredAway(right)) => {
                pending.push((*left, *right));
            }
            (SyntaxNode::Boolean(left), SyntaxNode::Boolean(right)) => {
                if left.len() != right.len()
                    || left
                        .iter()
                        .zip(right)
                        .any(|((left, _), (right, _))| left != right)
                {
                    return false;
                }
                pending.extend(
                    left.iter()
                        .zip(right)
                        .rev()
                        .map(|((_, left), (_, right))| (*left, *right)),
                );
            }
            (
                SyntaxNode::Boost {
                    query: left,
                    factor_bits: left_bits,
                },
                SyntaxNode::Boost {
                    query: right,
                    factor_bits: right_bits,
                },
            ) if left_bits == right_bits => pending.push((*left, *right)),
            (left, right) if left == right => {}
            _ => return false,
        }
    }
    true
}

/// Mirror Tantivy's post-lowering `trim_ast` for parser leaves that failed
/// semantic lowering. Intentional match-none leaves remain executable; only
/// syntax explicitly tagged `LoweredAway` disappears. A boost is a
/// boundary even when its child failed lowering.
pub(super) fn trim_parser_dropped(query: &mut Query, syntax: &SyntaxKey) -> bool {
    let result_len = syntax.nodes.len().saturating_add(1);
    let mut results = Vec::new();
    if results.try_reserve_exact(result_len).is_err() {
        return false;
    }
    results.resize(result_len, false);
    let mut pending = Vec::new();
    if pending
        .try_reserve_exact(query.node_count().saturating_add(result_len))
        .is_err()
    {
        return false;
    }
    pending.push((query.root_id(), SyntaxKey::root_id(), false));
    while let Some((query_id, syntax_id, visited)) = pending.pop() {
        let result_index = if syntax_id == SyntaxNodeId::ROOT {
            syntax.nodes.len()
        } else {
            syntax_id.index()
        };
        if !visited {
            match syntax.node(syntax_id) {
                SyntaxNode::LoweredAway(_) => results[result_index] = true,
                SyntaxNode::Boost { query: child, .. } => {
                    let query_child = match query.node(query_id) {
                        QueryNode::Boost { query, .. } => *query,
                        QueryNode::Empty
                        | QueryNode::All
                        | QueryNode::AllocationFailure { .. }
                        | QueryNode::Term { .. }
                        | QueryNode::Phrase { .. }
                        | QueryNode::Boolean { .. }
                        | QueryNode::Range { .. }
                        | QueryNode::Set { .. }
                        | QueryNode::Glob { .. } => query_id,
                    };
                    pending.push((query_id, syntax_id, true));
                    pending.push((query_child, *child, false));
                }
                SyntaxNode::Boolean(keys) => {
                    let QueryNode::Boolean { clauses, .. } = query.node(query_id) else {
                        continue;
                    };
                    if clauses.len() != keys.len() {
                        debug_assert_eq!(clauses.len(), keys.len());
                        continue;
                    }
                    pending.push((query_id, syntax_id, true));
                    pending.extend(
                        clauses
                            .iter()
                            .zip(keys)
                            .rev()
                            .map(|(clause, (_, key))| (clause.query, *key, false)),
                    );
                }
                SyntaxNode::Atom { .. }
                | SyntaxNode::Range { .. }
                | SyntaxNode::Set { .. }
                | SyntaxNode::Regex { .. } => {}
            }
            continue;
        }

        match syntax.node(syntax_id) {
            SyntaxNode::Boost { query: child, .. } => {
                results[result_index] = if matches!(query.node(query_id), QueryNode::Boost { .. }) {
                    false
                } else {
                    results[child.index()]
                };
            }
            SyntaxNode::Boolean(keys) => {
                let QueryNode::Boolean { clauses, .. } = query.node(query_id) else {
                    continue;
                };
                let retained = clauses
                    .iter()
                    .zip(keys)
                    .filter(|(_, (_, key))| !results[key.index()])
                    .count();
                if retained == 0 {
                    *query.node_mut(query_id) = QueryNode::Empty;
                    results[result_index] = true;
                    continue;
                }
                if retained != clauses.len() {
                    let mut kept = Vec::new();
                    if kept.try_reserve_exact(retained).is_err() {
                        continue;
                    }
                    kept.extend(
                        clauses
                            .iter()
                            .zip(keys)
                            .filter(|(_, (_, key))| !results[key.index()])
                            .map(|(clause, _)| *clause),
                    );
                    if let QueryNode::Boolean { clauses, .. } = query.node_mut(query_id) {
                        *clauses = kept;
                    }
                }
            }
            SyntaxNode::LoweredAway(_)
            | SyntaxNode::Atom { .. }
            | SyntaxNode::Range { .. }
            | SyntaxNode::Set { .. }
            | SyntaxNode::Regex { .. } => {}
        }
    }
    results[syntax.nodes.len()]
}

pub(super) fn repair_root_all_negative(node: &mut Option<ParsedNode>, grammar: &mut Grammar) {
    let Some(parsed) = node else {
        return;
    };
    if !query_is_all_negative(&parsed.query) {
        return;
    }
    match add_all_to_negative_root(&mut parsed.query) {
        Ok(false) => return,
        Ok(true) => {}
        Err(requested) => {
            parsed.query = Query::arena_allocation_failure(requested);
            grammar.allocation_failure("all-negative repair clauses", requested);
            return;
        }
    }
    grammar.push_diagnostic(QueryDiagnostic {
        kind: QueryDiagnosticKind::AllNegativeRepair,
        message: "all-negative query repaired with All".to_owned(),
        byte_offset: None,
        fragment: None,
    });
}

fn query_is_all_negative(query: &Query) -> bool {
    let mut results = Vec::new();
    if results.try_reserve_exact(query.nodes.len()).is_err() {
        return false;
    }
    for node in &query.nodes {
        let all_negative = match node {
            QueryNode::Boolean { clauses, .. } => {
                !clauses.is_empty()
                    && clauses.iter().all(|clause| {
                        clause.occur == Occur::MustNot
                            || (!matches!(clause.query, QueryNodeId::ALL | QueryNodeId::EMPTY)
                                && results[clause.query.index()])
                    })
            }
            QueryNode::Boost { query: child, .. } => {
                !matches!(*child, QueryNodeId::ALL | QueryNodeId::EMPTY) && results[child.index()]
            }
            QueryNode::Empty
            | QueryNode::All
            | QueryNode::AllocationFailure { .. }
            | QueryNode::Term { .. }
            | QueryNode::Phrase { .. }
            | QueryNode::Range { .. }
            | QueryNode::Set { .. }
            | QueryNode::Glob { .. } => false,
        };
        results.push(all_negative);
    }
    match query.root() {
        QueryNode::Boolean { clauses, .. } => {
            !clauses.is_empty()
                && clauses.iter().all(|clause| {
                    clause.occur == Occur::MustNot
                        || (!matches!(clause.query, QueryNodeId::ALL | QueryNodeId::EMPTY)
                            && results[clause.query.index()])
                })
        }
        QueryNode::Boost { query: child, .. } => {
            !matches!(*child, QueryNodeId::ALL | QueryNodeId::EMPTY) && results[child.index()]
        }
        QueryNode::Empty
        | QueryNode::All
        | QueryNode::AllocationFailure { .. }
        | QueryNode::Term { .. }
        | QueryNode::Phrase { .. }
        | QueryNode::Range { .. }
        | QueryNode::Set { .. }
        | QueryNode::Glob { .. } => false,
    }
}

fn add_all_to_negative_root(query: &mut Query) -> Result<bool, usize> {
    let mut current = query.root_id();
    while let QueryNode::Boost { query: child, .. } = query.node(current) {
        current = *child;
    }
    let QueryNode::Boolean { clauses, .. } = query.node_mut(current) else {
        return Ok(false);
    };
    if clauses.try_reserve_exact(1).is_err() {
        return Err(clauses.len().saturating_add(1));
    }
    clauses.push(QueryClause {
        occur: Occur::Should,
        query: QueryNodeId::ALL,
    });
    Ok(true)
}

pub(super) fn merge_field_queries(queries: Vec<Query>) -> Option<Query> {
    let first = queries.first()?.clone();
    match first.root() {
        QueryNode::Term { text, .. }
            if queries.iter().all(
                |query| matches!(query.root(), QueryNode::Term { text: candidate, .. } if candidate == text),
            ) =>
        {
            if queries.len() == 1 {
                return queries.into_iter().next();
            }
            let requested = queries.len();
            let mut clauses = Vec::new();
            if clauses.try_reserve_exact(requested).is_err() {
                return Some(Query::arena_allocation_failure(requested));
            }
            for query in queries {
                clauses.push(BooleanClause::new(Occur::Should, query));
            }
            Some(Query::boolean(clauses, None))
        }
        QueryNode::Phrase {
            terms,
            slop,
            prefix,
            ..
        } if queries.iter().all(|query| {
            matches!(
                query.root(),
                QueryNode::Phrase {
                    terms: candidate_terms,
                    slop: candidate_slop,
                    prefix: candidate_prefix,
                    ..
                } if candidate_terms == terms
                    && candidate_slop == slop
                    && candidate_prefix == prefix
            )
        }) =>
        {
            let requested = queries.iter().fold(0_usize, |count, query| {
                let QueryNode::Phrase { fields, .. } = query.root() else {
                    return count;
                };
                count.saturating_add(fields.len())
            });
            let mut fields = Vec::new();
            if fields.try_reserve_exact(requested).is_err() {
                return Some(Query::arena_allocation_failure(requested));
            }
            for query in queries {
                let QueryNode::Phrase {
                    fields: query_fields,
                    ..
                } = query.root
                else {
                    return None;
                };
                fields.extend(query_fields);
            }
            Some(Query::phrase(fields, terms.clone(), *slop, *prefix))
        }
        QueryNode::Empty if queries.iter().all(Query::is_empty) => Some(Query::empty()),
        QueryNode::Empty
        | QueryNode::AllocationFailure { .. }
        | QueryNode::All
        | QueryNode::Term { .. }
        | QueryNode::Phrase { .. }
        | QueryNode::Boolean { .. }
        | QueryNode::Range { .. }
        | QueryNode::Set { .. }
        | QueryNode::Glob { .. }
        | QueryNode::Boost { .. } => {
            let requested = queries.len();
            let mut clauses = Vec::new();
            if clauses.try_reserve_exact(requested).is_err() {
                return Some(Query::arena_allocation_failure(requested));
            }
            for query in queries {
                clauses.push(BooleanClause::new(Occur::Should, query));
            }
            Some(Query::boolean(clauses, None))
        }
    }
}

#[cfg(test)]
mod pttxk_slop_probe {
    use crate::query::DefaultQueryParser;
    #[test]
    #[ignore = "bd-1i4j4 measurement probe; run explicitly"]
    fn probe_slop_parse() {
        let parser = DefaultQueryParser::new(crate::DEFAULT_SCHEMA).expect("parser");
        let parsed = parser.parse_lenient("\"alpha beta\"~2");
        println!("PROBE diagnostics = {:?}", parsed.diagnostics);
        println!("PROBE query root = {:?}", parsed.query.root());
        println!("PROBE nodes = {}", parsed.query.node_count());
    }
}
