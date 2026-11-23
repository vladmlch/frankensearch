use std::borrow::Cow;

/// Strip the boost from any parenthesized group that contains a negation
/// (bd-f20ye), for the SHIPPING search path only.
///
/// # Why a boost has to be dropped rather than moved
///
/// Tantivy 0.26.1 lowers `(A NOT B)` as a `MustNot` clause OF the enclosing
/// boolean, which excludes, but lowers `(A NOT B)^2` by nesting the negation in
/// its own `BooleanQuery { [(MustNot, …)], msm: 0 }` and attaching THAT as a
/// clause of the outer boolean. A matcher meaning "every document except B"
/// then becomes a positive alternative, so the group stops excluding:
/// `(A NOT B)^2` returns the documents B was supposed to remove, and
/// `(A AND NOT B)^2` fails the other way and returns nothing.
///
/// The repair cannot be done on the parsed tree: `BoostQuery`'s fields are
/// private and it exposes no accessors, so the boolean inside a boost is
/// unreachable. It therefore happens on the query text, before parsing.
///
/// Dropping the boost is exact for MEMBERSHIP, which is the property at stake,
/// and lossy for SCORING: a `MustNot` clause contributes no score, so the boost
/// only ever scaled the positive operands, and those keep their relative order
/// without it. Redistributing the factor onto each positive operand would
/// preserve scores too, but requires splitting arbitrary group syntax by hand —
/// more ways to be wrong, for an ordering nicety, on a shape Tantivy cannot
/// currently express correctly at all.
///
/// Quoted phrases are respected, so a literal `"a NOT b"` never triggers it.
/// Rewrite `A AND NOT B` to `A NOT B` for the SHIPPING path only (bd-eeq0q).
///
/// THE DEFECT, measured on tantivy 0.26.1 with p1="alpha beta", p2="alpha gamma":
///
/// ```text
///   alpha NOT beta            [p2]   correct
///   alpha -beta               [p2]   correct
///   alpha AND NOT beta        []     WRONG — p2 has alpha and no beta
///   (alpha AND NOT beta)      []     WRONG
///   (alpha AND NOT beta)^2    []     WRONG, and untouched by the bd-f20ye repair
/// ```
///
/// `A AND NOT B` lowers to
/// `Bool{[(Must, A), (Must, Bool{[(MustNot, B)], msm: 0})], msm: 0}`. The second
/// `Must` operand is a boolean with only a `MustNot` clause and no positive
/// clause, so it matches nothing and the whole conjunction is empty. The same
/// engine returns the right answer for `A NOT B` and `A -B`, so this is an
/// inconsistency inside one query language rather than a defensible reading.
///
/// The repair deletes the redundant `AND` in front of a `NOT`, which is exactly
/// the spelling the engine already handles. It is textual for the same reason
/// [`repair_boosted_group_negation`] is: the lowering happens inside tantivy's
/// parser, and the resulting `BooleanQuery` cannot be rewritten from outside.
///
/// SCOPE, deliberately narrow. Only the exact token sequence `AND` `NOT` is
/// touched, both uppercase, both standalone. Quoted text is skipped, so a
/// literal `"a AND NOT b"` phrase query is never rewritten. Lowercase `and not`
/// is a pair of ordinary terms in this grammar and is left alone.
///
/// NOT A CONFORMANCE CHANGE. This runs only from
/// [`crate::TantivyIndex::parse_query_shipping`]; every `oracle_observe_*` caller
/// keeps using [`crate::TantivyIndex::parse_query_lenient`] and still reproduces the
/// defect, so the pinned comparator does not move. Quill was MEASURED to share
/// this behaviour (it returns nothing for `A AND NOT B` too, because
/// `wrap_not_for_and` in its parser mirrors the same lowering), which is why
/// repairing this side alone changes no Quill-versus-oracle comparison.
pub fn repair_and_not(query: &str) -> Cow<'_, str> {
    // Cheap reject: the token pair cannot be present.
    if !query.contains("AND") {
        return Cow::Borrowed(query);
    }

    let bytes = query.as_bytes();
    let mut drop_ranges: Vec<(usize, usize)> = Vec::new();
    let mut in_quotes = false;
    let mut index = 0_usize;

    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'"' {
            in_quotes = !in_quotes;
            index += 1;
            continue;
        }
        if in_quotes {
            index += 1;
            continue;
        }
        if byte == b'A' && query[index..].starts_with("AND") {
            let starts_token = index.checked_sub(1).is_none_or(|previous| {
                bytes[previous].is_ascii_whitespace() || bytes[previous] == b'('
            });
            // `AND` must be followed by whitespace, then `NOT` as its own token.
            let mut cursor = index + 3;
            let space_start = cursor;
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let had_space = cursor > space_start;
            let followed_by_not = had_space
                && query[cursor..].starts_with("NOT")
                && bytes
                    .get(cursor + 3)
                    .is_some_and(|next| next.is_ascii_whitespace() || *next == b'(');
            if starts_token && followed_by_not {
                // Drop `AND` and the whitespace that separated it from `NOT`,
                // leaving the preceding separator intact: `a AND NOT b`
                // becomes `a NOT b`, never `aNOT b`.
                drop_ranges.push((index, cursor));
                index = cursor;
                continue;
            }
        }
        index += 1;
    }

    if drop_ranges.is_empty() {
        return Cow::Borrowed(query);
    }
    let mut repaired = String::with_capacity(query.len());
    let mut copied = 0_usize;
    for (start, end) in drop_ranges {
        repaired.push_str(&query[copied..start]);
        copied = end;
    }
    repaired.push_str(&query[copied..]);
    Cow::Owned(repaired)
}

pub fn repair_boosted_group_negation(query: &str) -> Cow<'_, str> {
    // Cheap reject: no group boost, nothing to repair.
    if !query.contains(")^") {
        return Cow::Borrowed(query);
    }

    let bytes = query.as_bytes();
    // One flag per open group: did THIS group contain a negation directly?
    let mut group_negates: Vec<bool> = Vec::new();
    // Byte ranges of `^<number>` suffixes to remove.
    let mut drop_ranges: Vec<(usize, usize)> = Vec::new();
    let mut in_quotes = false;
    let mut index = 0_usize;

    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'"' {
            in_quotes = !in_quotes;
            index += 1;
            continue;
        }
        if in_quotes {
            index += 1;
            continue;
        }
        match byte {
            b'(' => group_negates.push(false),
            b')' => {
                let negates = group_negates.pop().unwrap_or(false);
                // A boost suffix is `^` followed by a number, possibly decimal.
                let mut cursor = index + 1;
                if negates && cursor < bytes.len() && bytes[cursor] == b'^' {
                    let start = cursor;
                    cursor += 1;
                    let digits_start = cursor;
                    while cursor < bytes.len()
                        && (bytes[cursor].is_ascii_digit() || bytes[cursor] == b'.')
                    {
                        cursor += 1;
                    }
                    if cursor > digits_start {
                        drop_ranges.push((start, cursor));
                    }
                }
            }
            b'N' => {
                // `NOT` as a standalone token, at this group's own level.
                let is_token = query[index..].starts_with("NOT")
                    && index.checked_sub(1).is_none_or(|previous| {
                        bytes[previous].is_ascii_whitespace() || bytes[previous] == b'('
                    })
                    && bytes
                        .get(index + 3)
                        .is_none_or(|next| next.is_ascii_whitespace() || *next == b'(');
                if is_token && let Some(current) = group_negates.last_mut() {
                    *current = true;
                }
            }
            b'-' => {
                // Prefix exclusion: `-term`, not a hyphen inside a word.
                let is_prefix = index.checked_sub(1).is_none_or(|previous| {
                    bytes[previous].is_ascii_whitespace() || bytes[previous] == b'('
                }) && bytes
                    .get(index + 1)
                    .is_some_and(|next| !next.is_ascii_whitespace());
                if is_prefix && let Some(current) = group_negates.last_mut() {
                    *current = true;
                }
            }
            _ => {}
        }
        index += 1;
    }

    if drop_ranges.is_empty() {
        return Cow::Borrowed(query);
    }
    let mut repaired = String::with_capacity(query.len());
    let mut cursor = 0_usize;
    for (start, end) in drop_ranges {
        repaired.push_str(&query[cursor..start]);
        cursor = end;
    }
    repaired.push_str(&query[cursor..]);
    Cow::Owned(repaired)
}

/// One operand of a boolean level, with its occurrence prefix decoded.
struct LevelOperand {
    negated: bool,
    required: bool,
    text: String,
}

/// A parsed element of one boolean level.
enum LevelItem {
    Operand(LevelOperand),
    And,
    Or,
}

/// Repair a CONJUNCTION that contains a negated operand (bd-8a2a8, **DIV-010**).
///
/// THE DEFECT. The pinned grammar declares `default join := OR; explicit AND has
/// precedence over OR` (`docs/contracts/quill-language-contract.md`), so
/// `a NOT b AND c` reads as `a OR (c AND NOT b)`. Tantivy 0.26.1 lowers the
/// `AND` conjunct to `Bool{[(Must, Bool{[(MustNot, b)], msm: 0}), (Must, c)]}`,
/// whose first operand has no positive clause and therefore matches nothing, so
/// the whole conjunct drops and only `a` survives. MEASURED on the shared
/// Core100 fixture: `release NOT bounds AND small` returns 21 documents
/// (`|release|`) where the declared reading is 41 (`|release ∪ small|`), which
/// is what Quill returns and what this engine ITSELF returns when the same
/// grouping is written out as `release ((small) NOT bounds)`. Two independent
/// implementations agreeing on the explicit form is what attributes the defect
/// to the implicit lowering rather than to either engine's semantics.
///
/// THE REPAIR normalises a whole `AND` chain that contains a negation into the
/// occurrence spelling this engine already lowers correctly: every positive
/// operand becomes `+operand` and every negative one `-operand`, in the operand
/// order the user wrote, so `a NOT b AND c` becomes `a (-b +c)`. That is exact
/// rather than approximate —
/// `+`/`-` ARE the `Must`/`MustNot` the chain means — which matters because the
/// obvious narrow alternative, deleting the `AND`, was measured to break queries
/// that were already correct (`explains NOT bounds AND refactors` answers 60 and
/// dropping the `AND` answers 59). It is textual for the same reason
/// [`repair_and_not`] and [`repair_boosted_group_negation`] are: the lowering
/// happens inside tantivy's parser and the resulting `BooleanQuery` cannot be
/// rewritten from outside.
///
/// SCOPE, deliberately narrow. A chain is rewritten only when it holds more than
/// one operand and at least one of them is negated; a pure-positive `a AND b`
/// keeps its spelling. Anything the scanner cannot decode — an unbalanced quote,
/// a dangling `AND`, an empty operand — returns the query untouched rather than
/// guessing. Quoted text is skipped, so a literal `"a AND NOT b"` phrase is
/// never rewritten, and a `field:(a b)` operand is carried through opaquely.
///
/// A CHAIN THAT SPANS ITS WHOLE LEVEL IS EMITTED WITHOUT PARENTHESES, which is
/// load-bearing rather than cosmetic: [`repair_boosted_group_negation`] records
/// negation against the INNERMOST open group, so wrapping `(a AND NOT b)^2` into
/// `((+a -b))^2` would hide the negation from it and silently regress DIV-009.
///
/// NOT A CONFORMANCE CHANGE. This runs only from
/// [`crate::TantivyIndex::parse_query_shipping`]; every `oracle_observe_*` caller keeps
/// using [`crate::TantivyIndex::parse_query_lenient`] and still reproduces the defect,
/// so the pinned comparator does not move.
pub fn repair_negated_conjunction(query: &str) -> Cow<'_, str> {
    // Cheap reject: without an explicit `AND` there is no chain to normalise.
    if !query.contains("AND") {
        return Cow::Borrowed(query);
    }
    rewrite_conjunction_level(query).map_or(Cow::Borrowed(query), Cow::Owned)
}

/// Is `token` present at `index` as a standalone token at this level?
fn is_level_token(level: &str, index: usize, token: &str) -> bool {
    let bytes = level.as_bytes();
    level[index..].starts_with(token)
        && index
            .checked_sub(1)
            .is_none_or(|previous| bytes[previous].is_ascii_whitespace() || bytes[previous] == b'(')
        && bytes
            .get(index + token.len())
            .is_none_or(|next| next.is_ascii_whitespace() || *next == b'(')
}

/// Find the `)` matching the `(` at `open`, ignoring parentheses inside quotes.
fn matching_close_paren(level: &str, open: usize) -> Option<usize> {
    let bytes = level.as_bytes();
    let mut depth = 0_usize;
    let mut in_quotes = false;
    for (offset, byte) in bytes[open..].iter().enumerate() {
        let cursor = open + offset;
        match byte {
            b'"' => in_quotes = !in_quotes,
            b'(' if !in_quotes => depth += 1,
            b')' if !in_quotes => {
                depth -= 1;
                if depth == 0 {
                    return Some(cursor);
                }
            }
            _ => {}
        }
    }
    None
}

/// Read one operand body starting at `start`, returning its text, the index just
/// past it, and whether a nested level inside it was rewritten.
fn parse_operand_body(level: &str, start: usize) -> Option<(String, usize, bool)> {
    let bytes = level.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    if bytes[start] == b'(' {
        let close = matching_close_paren(level, start)?;
        let inner = &level[start + 1..close];
        let rewritten = rewrite_conjunction_level(inner);
        let inner_changed = rewritten.is_some();
        let inner_text = rewritten.unwrap_or_else(|| inner.to_string());
        // Trailing modifiers stay attached to the group: `^boost`, `~slop`.
        let mut cursor = close + 1;
        while cursor < bytes.len()
            && !bytes[cursor].is_ascii_whitespace()
            && bytes[cursor] != b')'
            && bytes[cursor] != b'('
        {
            cursor += 1;
        }
        let suffix = &level[close + 1..cursor];
        return Some((format!("({inner_text}){suffix}"), cursor, inner_changed));
    }

    let mut in_quotes = false;
    let mut cursor = start;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if byte == b'"' {
            in_quotes = !in_quotes;
            cursor += 1;
            continue;
        }
        if in_quotes {
            cursor += 1;
            continue;
        }
        // A `(` inside a bare operand is a fielded group like `field:(a b)`:
        // carry it through opaquely rather than splitting the operand in two.
        if byte == b'(' {
            cursor = matching_close_paren(level, cursor)? + 1;
            continue;
        }
        if byte.is_ascii_whitespace() || byte == b')' {
            break;
        }
        cursor += 1;
    }
    if in_quotes || cursor == start {
        return None;
    }
    Some((level[start..cursor].to_string(), cursor, false))
}

/// Rewrite every negated `AND` chain at one boolean level, recursing into
/// parenthesised operands. `None` means "nothing to change, or not decodable".
fn rewrite_conjunction_level(level: &str) -> Option<String> {
    let bytes = level.as_bytes();
    let mut items: Vec<LevelItem> = Vec::new();
    let mut nested_changed = false;
    let mut index = 0_usize;

    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if is_level_token(level, index, "AND") {
            items.push(LevelItem::And);
            index += 3;
            continue;
        }
        if is_level_token(level, index, "OR") {
            items.push(LevelItem::Or);
            index += 2;
            continue;
        }
        // Occurrence prefixes: any run of negators, and an explicit `+`.
        let mut negated = false;
        let mut required = false;
        loop {
            let byte = *bytes.get(index)?;
            if byte == b'-' || byte == b'+' {
                // A bare `-` or `+` with nothing attached is not an occurrence
                // prefix, and this scanner declines rather than guessing.
                if bytes
                    .get(index + 1)
                    .is_none_or(|next| next.is_ascii_whitespace())
                {
                    return None;
                }
                if byte == b'-' {
                    negated = true;
                } else {
                    required = true;
                }
                index += 1;
                continue;
            }
            if is_level_token(level, index, "NOT") {
                negated = true;
                index += 3;
                while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                    index += 1;
                }
                continue;
            }
            break;
        }
        let (text, next, inner_changed) = parse_operand_body(level, index)?;
        nested_changed |= inner_changed;
        items.push(LevelItem::Operand(LevelOperand {
            negated,
            required,
            text,
        }));
        index = next;
    }

    // Group the level into `AND` chains. `AND` binds tighter than the default
    // join, so a maximal chain is exactly the unit whose lowering is defective.
    let mut chains: Vec<Vec<LevelOperand>> = Vec::new();
    let mut joins: Vec<&'static str> = Vec::new();
    let mut expect_operand = false;
    let mut next_join = " ";
    for item in items {
        match item {
            LevelItem::And => {
                if chains.is_empty() || expect_operand {
                    return None;
                }
                expect_operand = true;
            }
            LevelItem::Or => {
                if chains.is_empty() || expect_operand {
                    return None;
                }
                next_join = " OR ";
            }
            LevelItem::Operand(operand) => {
                if expect_operand {
                    chains.last_mut()?.push(operand);
                    expect_operand = false;
                } else {
                    if !chains.is_empty() {
                        joins.push(next_join);
                    }
                    next_join = " ";
                    chains.push(vec![operand]);
                }
            }
        }
    }
    if expect_operand || chains.is_empty() {
        return None;
    }

    let single_chain = chains.len() == 1;
    let mut rewrote = false;
    let mut rendered: Vec<String> = Vec::with_capacity(chains.len());
    for chain in &chains {
        if chain.len() > 1 && chain.iter().any(|operand| operand.negated) {
            rewrote = true;
            let mut body = String::new();
            for operand in chain {
                if !body.is_empty() {
                    body.push(' ');
                }
                body.push(if operand.negated { '-' } else { '+' });
                body.push_str(&operand.text);
            }
            // Bare when it IS the level, so DIV-009 still sees the negation.
            rendered.push(if single_chain {
                body
            } else {
                format!("({body})")
            });
        } else if chain.len() > 1 {
            rendered.push(
                chain
                    .iter()
                    .map(render_operand)
                    .collect::<Vec<_>>()
                    .join(" AND "),
            );
        } else {
            rendered.push(render_operand(&chain[0]));
        }
    }

    if !rewrote && !nested_changed {
        return None;
    }
    let mut output = rendered.first()?.clone();
    for (position, piece) in rendered.iter().enumerate().skip(1) {
        output.push_str(joins.get(position - 1)?);
        output.push_str(piece);
    }
    Some(output)
}

/// Re-emit an operand outside a rewritten chain, preserving its occurrence.
fn render_operand(operand: &LevelOperand) -> String {
    if operand.negated {
        format!("-{}", operand.text)
    } else if operand.required {
        format!("+{}", operand.text)
    } else {
        operand.text.clone()
    }
}
