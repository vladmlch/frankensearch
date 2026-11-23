use super::canonical::{QueryDiagnostic, QueryDiagnosticKind};
use super::grammar::ParsedNode;
use super::syntax::{AtomDelimiter, AtomToken, RangeToken, SetToken, emit_diagnostic};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum LexToken {
    Atom(AtomToken),
    Range(RangeToken),
    Set(SetToken),
    UnsupportedRegex {
        byte_offset: usize,
        field: Option<String>,
        pattern: String,
        boost: Option<String>,
    },
    Dropped(usize),
    And(usize),
    Or(usize),
    Not(usize),
    Plus(usize),
    Minus(usize),
    Lowered {
        node: ParsedNode,
        byte_offset: usize,
    },
    LeftParen {
        byte_offset: usize,
        field: Option<String>,
    },
    RightParen {
        byte_offset: usize,
        boost: Option<String>,
    },
}

/// Normalize depth-inert wrappers around the whole token stream before
/// precedence parsing.
///
/// Balanced unfielded, unboosted wrappers are removed. A pure chain of
/// identical unboosted field scopes keeps one innermost wrapper so it still
/// supplies the field to otherwise-unqualified leaves. Both rules avoid
/// charging semantically redundant depth against the parser limit without
/// weakening recovery for genuinely nested Boolean structure.
pub(super) fn peel_redundant_outer_groups(tokens: &mut Vec<LexToken>) {
    peel_redundant_same_field_outer_scopes(tokens);

    let mut peel_count = tokens
        .iter()
        .take_while(|token| matches!(token, LexToken::LeftParen { field: None, .. }))
        .count();
    if peel_count == 0 {
        return;
    }

    let token_count = tokens.len();
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate() {
        match token {
            LexToken::LeftParen { .. } => depth += 1,
            LexToken::RightParen { boost, .. } => {
                if depth == 0 {
                    return;
                }
                if depth <= peel_count && (index != token_count - depth || boost.is_some()) {
                    peel_count = depth - 1;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    if depth != 0 || peel_count == 0 {
        return;
    }

    tokens.drain(..peel_count);
    tokens.truncate(tokens.len() - peel_count);
}

/// Collapse a pure whole-stream chain of one field scope, including any
/// unfielded groups interleaved with it.
///
/// One fielded wrapper is retained: it still supplies the field scope to
/// otherwise-unqualified leaves. The interleaved unfielded groups need no
/// replacement because a group inside a field scope already inherits it, so
/// `content:((x))`, `(content:(x))` and `content:(content:(x))` all denote the
/// same query as `content:(x)` — which is exactly why charging them against the
/// depth limit turned a valid query into match-none.
///
/// A chain mixing two DIFFERENT field scopes is not inert and is refused, as is
/// an unbalanced nested body, a boost on the closing suffix, an unmatched
/// closure, or material outside that suffix; those leave the stream untouched
/// for ordinary parser recovery.
fn peel_redundant_same_field_outer_scopes(tokens: &mut Vec<LexToken>) {
    if !matches!(tokens.first(), Some(LexToken::LeftParen { .. })) {
        return;
    }
    let opening_count = tokens
        .iter()
        .take_while(|token| matches!(token, LexToken::LeftParen { .. }))
        .count();
    if opening_count < 2 {
        return;
    }
    let opening = &tokens[..opening_count];

    // The retained wrapper must be a FIELDED one. Keeping whichever opener
    // happened to be innermost would silently drop the scope when that opener
    // is an unfielded group, turning a field-scoped query into a default-field
    // one — a wrong answer rather than a refused one.
    let Some(scope_index) = opening
        .iter()
        .position(|token| matches!(token, LexToken::LeftParen { field: Some(_), .. }))
    else {
        // A wholly unfielded run is the other normalization's job.
        return;
    };
    let LexToken::LeftParen {
        field: Some(field), ..
    } = &opening[scope_index]
    else {
        return;
    };
    let uniform_scope = opening.iter().all(|token| match token {
        LexToken::LeftParen { field: None, .. } => true,
        LexToken::LeftParen {
            field: Some(candidate),
            ..
        } => candidate == field,
        _ => false,
    });
    if !uniform_scope {
        return;
    }

    let Some(body_end) = tokens.len().checked_sub(opening_count) else {
        return;
    };
    if body_end <= opening_count
        || tokens[body_end..]
            .iter()
            .any(|token| !matches!(token, LexToken::RightParen { boost: None, .. }))
    {
        return;
    }

    // Nested Boolean structure inside the repeated scopes does not make those
    // scopes meaningful: retaining one identical fielded wrapper preserves the
    // field inherited by every unqualified leaf. Prove that the body is
    // independently balanced before removing any outer pair so malformed input
    // still reaches the parser's ordinary recovery unchanged.
    let mut body_depth = 0_usize;
    for token in &tokens[opening_count..body_end] {
        match token {
            LexToken::LeftParen { .. } => body_depth += 1,
            LexToken::RightParen { .. } => {
                if body_depth == 0 {
                    return;
                }
                body_depth -= 1;
            }
            _ => {}
        }
    }
    if body_depth != 0 {
        return;
    }

    let retained = tokens[scope_index].clone();
    let peel_count = opening_count - 1;
    tokens.drain(..opening_count);
    tokens.insert(0, retained);
    tokens.truncate(tokens.len() - peel_count);
}

impl LexToken {
    pub(super) const fn byte_offset(&self) -> usize {
        match self {
            Self::Atom(atom) => atom.byte_offset,
            Self::Range(token) => token.byte_offset,
            Self::Set(token) => token.byte_offset,
            Self::UnsupportedRegex {
                byte_offset: offset,
                ..
            }
            | Self::Dropped(offset)
            | Self::And(offset)
            | Self::Or(offset)
            | Self::Not(offset)
            | Self::Plus(offset)
            | Self::Minus(offset) => *offset,
            Self::Lowered { byte_offset, .. } => *byte_offset,
            Self::LeftParen { byte_offset, .. } | Self::RightParen { byte_offset, .. } => {
                *byte_offset
            }
        }
    }

    pub(super) const fn can_start_operand(&self) -> bool {
        matches!(
            self,
            Self::Atom(_)
                | Self::Range(_)
                | Self::Set(_)
                | Self::UnsupportedRegex { .. }
                | Self::Dropped(_)
                | Self::Not(_)
                | Self::Plus(_)
                | Self::Minus(_)
                | Self::Lowered { .. }
                | Self::LeftParen { .. }
        )
    }
}

pub(super) fn lex(query: &str, diagnostics: &mut Vec<QueryDiagnostic>) -> Vec<LexToken> {
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut open_groups = 0_usize;

    while cursor < query.len() {
        let Some(ch) = query[cursor..].chars().next() else {
            break;
        };
        if ch.is_whitespace() {
            cursor += ch.len_utf8();
            continue;
        }

        if let Some((field, value_start)) = field_prefix(query, cursor) {
            let value_start = skip_whitespace(query, value_start);
            if query[value_start..].starts_with('(') {
                tokens.push(LexToken::LeftParen {
                    byte_offset: value_start,
                    field: Some(unescape_field_name(field)),
                });
                open_groups += 1;
                cursor = value_start + 1;
                continue;
            }
            if let Some(delimiter) = query[value_start..]
                .chars()
                .next()
                .filter(|candidate| matches!(candidate, '"' | '\''))
            {
                let (atom, next) = lex_quoted(
                    query,
                    value_start,
                    delimiter,
                    Some(unescape_field_name(field)),
                    diagnostics,
                );
                tokens.push(LexToken::Atom(atom));
                cursor = next;
                continue;
            }
            if matches!(query[value_start..].chars().next(), Some('[' | '{')) {
                let (token, next) =
                    lex_typed_range(query, cursor, Some(field), value_start, diagnostics);
                if let Some(token) = token {
                    tokens.push(token);
                } else {
                    tokens.push(LexToken::Dropped(cursor));
                }
                cursor = next;
                continue;
            }
            if let Some(open_offset) = set_open_offset(query, value_start) {
                let (token, next) =
                    lex_typed_set(query, cursor, Some(field), open_offset, diagnostics);
                if let Some(token) = token {
                    tokens.push(token);
                } else {
                    tokens.push(LexToken::Dropped(cursor));
                }
                cursor = next;
                continue;
            }
            if comparison_operator(query, value_start).is_some() {
                let (token, next) =
                    lex_comparison_range(query, cursor, Some(field), value_start, diagnostics);
                if let Some(token) = token {
                    tokens.push(token);
                } else {
                    tokens.push(LexToken::Dropped(cursor));
                }
                cursor = next;
                continue;
            }

            let end = scan_fragment_end(query, value_start);
            let fragment = &query[value_start..end];
            let (raw, boost, trailing_boost) = split_boost(fragment);
            if let Some(relative) = trailing_boost {
                emit_trailing_boost_diagnostic(diagnostics, value_start + relative);
            }
            let raw = unescape_fragment(raw).0;
            if is_regex_fragment(&raw) {
                emit_regex_diagnostic(diagnostics, cursor);
                tokens.push(unsupported_regex_token(
                    cursor,
                    Some(unescape_field_name(field)),
                    &raw,
                    boost,
                ));
            } else {
                tokens.push(LexToken::Atom(AtomToken {
                    raw,
                    field: Some(unescape_field_name(field)),
                    delimiter: AtomDelimiter::None,
                    slop: 0,
                    prefix: false,
                    boost: boost.map(str::to_owned),
                    byte_offset: cursor,
                }));
            }
            cursor = end;
            continue;
        }

        if matches!(ch, '[' | '{') {
            let (token, next) = lex_typed_range(query, cursor, None, cursor, diagnostics);
            if let Some(token) = token {
                tokens.push(token);
            } else {
                tokens.push(LexToken::Dropped(cursor));
            }
            cursor = next;
            continue;
        }
        if let Some(open_offset) = set_open_offset(query, cursor) {
            let (token, next) = lex_typed_set(query, cursor, None, open_offset, diagnostics);
            if let Some(token) = token {
                tokens.push(token);
            } else {
                tokens.push(LexToken::Dropped(cursor));
            }
            cursor = next;
            continue;
        }
        if comparison_operator(query, cursor).is_some() {
            let (token, next) = lex_comparison_range(query, cursor, None, cursor, diagnostics);
            if let Some(token) = token {
                tokens.push(token);
            } else {
                tokens.push(LexToken::Dropped(cursor));
            }
            cursor = next;
            continue;
        }

        match ch {
            '(' => {
                tokens.push(LexToken::LeftParen {
                    byte_offset: cursor,
                    field: None,
                });
                open_groups += 1;
                cursor += 1;
                continue;
            }
            ')' => {
                let (boost, next) = lex_boost_suffix(query, cursor + 1, diagnostics);
                if open_groups == 0 {
                    emit_diagnostic(
                        diagnostics,
                        QueryDiagnostic {
                            kind: QueryDiagnosticKind::SyntaxRecovery,
                            message: "syntax recovery: unmatched closing parenthesis dropped"
                                .to_owned(),
                            byte_offset: Some(cursor),
                            fragment: None,
                        },
                    );
                    tokens.push(LexToken::Dropped(cursor));
                    break;
                }
                tokens.push(LexToken::RightParen {
                    byte_offset: cursor,
                    boost,
                });
                open_groups -= 1;
                cursor = next;
                continue;
            }
            '+' => {
                tokens.push(LexToken::Plus(cursor));
                cursor += 1;
                continue;
            }
            '-' => {
                tokens.push(LexToken::Minus(cursor));
                cursor += 1;
                continue;
            }
            '"' | '\'' => {
                let (atom, next) = lex_quoted(query, cursor, ch, None, diagnostics);
                tokens.push(LexToken::Atom(atom));
                cursor = next;
                continue;
            }
            _ => {}
        }

        let end = scan_fragment_end(query, cursor);
        let fragment = &query[cursor..end];
        let field: Option<String> = None;
        let value = fragment;
        let value_offset = cursor;
        let (raw, boost, trailing_boost) = split_boost(value);
        if let Some(relative) = trailing_boost {
            emit_trailing_boost_diagnostic(diagnostics, value_offset + relative);
        }
        let (raw, raw_was_escaped) = unescape_fragment(raw);
        if is_regex_fragment(&raw) {
            emit_regex_diagnostic(diagnostics, cursor);
            tokens.push(unsupported_regex_token(cursor, None, &raw, boost));
            cursor = end;
            continue;
        }
        let operator_has_ascii_space = query.as_bytes().get(end) == Some(&b' ');
        let token = if field.is_none() && boost.is_none() && !raw_was_escaped {
            match raw.as_str() {
                "AND" if operator_has_ascii_space => LexToken::And(cursor),
                "OR" if operator_has_ascii_space => LexToken::Or(cursor),
                "NOT" if operator_has_ascii_space => LexToken::Not(cursor),
                _ => LexToken::Atom(AtomToken {
                    raw,
                    field,
                    delimiter: AtomDelimiter::None,
                    slop: 0,
                    prefix: false,
                    boost: None,
                    byte_offset: cursor,
                }),
            }
        } else {
            LexToken::Atom(AtomToken {
                raw,
                field,
                delimiter: AtomDelimiter::None,
                slop: 0,
                prefix: false,
                boost: boost.map(str::to_owned),
                byte_offset: cursor,
            })
        };
        tokens.push(token);
        cursor = end;
    }

    if tokens.len() == 1 {
        let replacement = match &tokens[0] {
            LexToken::And(offset) => Some(operator_atom("AND", *offset)),
            LexToken::Or(offset) => Some(operator_atom("OR", *offset)),
            LexToken::Not(offset) => {
                emit_diagnostic(
                    diagnostics,
                    QueryDiagnostic {
                        kind: QueryDiagnosticKind::SyntaxRecovery,
                        message: "syntax recovery: standalone NOT retained as a literal".to_owned(),
                        byte_offset: Some(*offset),
                        fragment: None,
                    },
                );
                Some(operator_atom("NOT", *offset))
            }
            _ => None,
        };
        if let Some(replacement) = replacement {
            tokens[0] = replacement;
        }
    }

    tokens
}

fn field_prefix(query: &str, start: usize) -> Option<(&str, usize)> {
    let first = query[start..].chars().next()?;
    if first == '-' || (first != '\\' && is_field_special(first)) {
        return None;
    }
    let mut escaped = false;
    let mut field_end = None;
    for (relative, ch) in query[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == ':' {
            let colon = start + relative;
            let end = field_end.map_or(colon, |end| start + end);
            return (end > start).then_some((&query[start..end], colon + 1));
        }
        if is_field_special(ch) && ch != ' ' {
            return None;
        }
        if ch.is_whitespace() {
            field_end.get_or_insert(relative);
            continue;
        }
        if field_end.is_some() {
            break;
        }
    }
    None
}

fn is_field_special(ch: char) -> bool {
    matches!(
        ch,
        '+' | '^'
            | '`'
            | ':'
            | '{'
            | '}'
            | '"'
            | '\''
            | '['
            | ']'
            | '('
            | ')'
            | '!'
            | '\\'
            | '*'
            | ' '
    )
}

fn unescape_field_name(field: &str) -> String {
    let mut value = String::with_capacity(field.len());
    let mut escaped = false;
    for ch in field.chars() {
        if escaped {
            if !is_field_special(ch) {
                value.push('\\');
            }
            value.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            value.push(ch);
        }
    }
    if escaped {
        value.push('\\');
    }
    value
}

fn skip_whitespace(query: &str, mut cursor: usize) -> usize {
    while let Some(ch) = query[cursor..].chars().next() {
        if !ch.is_whitespace() {
            break;
        }
        cursor += ch.len_utf8();
    }
    cursor
}

fn set_open_offset(query: &str, start: usize) -> Option<usize> {
    let rest = query.get(start..)?.strip_prefix("IN")?;
    let after_in = start + 2;
    if !rest.chars().next().is_some_and(char::is_whitespace) {
        return None;
    }
    let open = skip_whitespace(query, after_in);
    query[open..].starts_with('[').then_some(open)
}

fn lex_typed_range(
    query: &str,
    byte_offset: usize,
    field: Option<&str>,
    open_offset: usize,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (Option<LexToken>, usize) {
    let lower_inclusive = query.as_bytes().get(open_offset) == Some(&b'[');
    let (close_offset, upper_inclusive, missing_close) = typed_close(query, open_offset).map_or(
        (query.len(), lower_inclusive, true),
        |(offset, inclusive)| (offset, inclusive, false),
    );
    if missing_close {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: unterminated typed range retained to end of query",
        );
    }
    let interior = query[open_offset + 1..close_offset].trim();
    let Some(parts) = split_set_values(interior) else {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: malformed typed range dropped",
        );
        return (
            None,
            close_offset.saturating_add(usize::from(!missing_close)),
        );
    };
    let (lower, upper) = match parts.as_slice() {
        [lower, separator, upper] if separator == "TO" => (lower.clone(), upper.clone()),
        [lower, separator] if separator == "TO" => {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                "syntax recovery: typed range missing upper bound",
            );
            (lower.clone(), "*".to_owned())
        }
        [lower, upper] => {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                "syntax recovery: typed range missing keyword `TO`",
            );
            (lower.clone(), upper.clone())
        }
        [lower] => {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                "syntax recovery: typed range missing keyword `TO` and upper bound",
            );
            (lower.clone(), "*".to_owned())
        }
        _ => {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                "syntax recovery: typed range requires `lower TO upper`",
            );
            return (
                None,
                close_offset.saturating_add(usize::from(!missing_close)),
            );
        }
    };
    let after_close = close_offset.saturating_add(usize::from(!missing_close));
    let (boost, next) = lex_boost_suffix(query, after_close, diagnostics);
    (
        Some(LexToken::Range(RangeToken {
            field: field.map(unescape_field_name),
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
            boost,
            byte_offset,
        })),
        next,
    )
}

fn lex_typed_set(
    query: &str,
    byte_offset: usize,
    field: Option<&str>,
    open_offset: usize,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (Option<LexToken>, usize) {
    let (close_offset, closed_with_bracket, missing_close) = typed_close(query, open_offset)
        .map_or((query.len(), true, true), |(offset, bracket)| {
            (offset, bracket, false)
        });
    if missing_close {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: unterminated typed set retained to end of query",
        );
    }
    if !missing_close && !closed_with_bracket {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: typed set requires square brackets",
        );
        return (None, close_offset + 1);
    }
    let after_close = close_offset.saturating_add(usize::from(!missing_close));
    let Some(values) = split_set_values(&query[open_offset + 1..close_offset]) else {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: malformed typed set dropped",
        );
        return (None, after_close);
    };
    let (boost, next) = lex_boost_suffix(query, after_close, diagnostics);
    (
        Some(LexToken::Set(SetToken {
            field: field.map(unescape_field_name),
            values,
            boost,
            byte_offset,
        })),
        next,
    )
}

fn split_set_values(interior: &str) -> Option<Vec<String>> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut delimiter = None;
    let mut escaped = false;
    let mut token_started = false;
    for ch in interior.chars() {
        if escaped {
            if delimiter.is_none() && !is_query_escape(ch) {
                current.push('\\');
            }
            current.push(ch);
            escaped = false;
            token_started = true;
        } else if ch == '\\' {
            escaped = true;
            token_started = true;
        } else if let Some(quote) = delimiter {
            if ch == quote {
                delimiter = None;
            } else {
                current.push(ch);
            }
        } else if matches!(ch, '"' | '\'') {
            delimiter = Some(ch);
            token_started = true;
        } else if ch == ',' {
            current.push(ch);
            token_started = true;
        } else if ch.is_whitespace() {
            if token_started {
                values.push(std::mem::take(&mut current));
                token_started = false;
            }
        } else {
            current.push(ch);
            token_started = true;
        }
    }
    if escaped || delimiter.is_some() {
        return None;
    }
    if token_started {
        values.push(current);
    }
    Some(values)
}

fn comparison_operator(query: &str, start: usize) -> Option<(&'static str, usize)> {
    for operator in [">=", "<=", ">", "<"] {
        if query[start..].starts_with(operator) {
            return Some((operator, operator.len()));
        }
    }
    None
}

fn lex_comparison_range(
    query: &str,
    byte_offset: usize,
    field: Option<&str>,
    operator_offset: usize,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (Option<LexToken>, usize) {
    let Some((operator, operator_len)) = comparison_operator(query, operator_offset) else {
        return (None, operator_offset);
    };
    let value_start = skip_whitespace(query, operator_offset + operator_len);
    let end = scan_fragment_end(query, value_start);
    let (value, boost, trailing_boost) = split_boost(&query[value_start..end]);
    if let Some(relative) = trailing_boost {
        emit_trailing_boost_diagnostic(diagnostics, value_start + relative);
    }
    if value.is_empty() {
        emit_typed_syntax_diagnostic(
            diagnostics,
            byte_offset,
            "syntax recovery: comparison range requires a bound",
        );
        return (None, end);
    }
    let value = unescape_fragment(value).0;
    let (lower, upper, lower_inclusive, upper_inclusive) = match operator {
        ">" => (value, "*".to_owned(), false, false),
        ">=" => (value, "*".to_owned(), true, false),
        "<" => ("*".to_owned(), value, false, false),
        "<=" => ("*".to_owned(), value, false, true),
        _ => return (None, end),
    };
    (
        Some(LexToken::Range(RangeToken {
            field: field.map(unescape_field_name),
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
            boost: boost.map(str::to_owned),
            byte_offset,
        })),
        end,
    )
}

fn typed_close(query: &str, open_offset: usize) -> Option<(usize, bool)> {
    let mut delimiter = None;
    let mut escaped = false;
    for (relative, ch) in query[open_offset + 1..].char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if let Some(quote) = delimiter {
            if ch == quote {
                delimiter = None;
            }
        } else if matches!(ch, '"' | '\'') {
            delimiter = Some(ch);
        } else if matches!(ch, ']' | '}') {
            return Some((open_offset + 1 + relative, ch == ']'));
        }
    }
    None
}

fn emit_typed_syntax_diagnostic(
    diagnostics: &mut Vec<QueryDiagnostic>,
    byte_offset: usize,
    message: &str,
) {
    emit_diagnostic(
        diagnostics,
        QueryDiagnostic {
            kind: QueryDiagnosticKind::SyntaxRecovery,
            message: message.to_owned(),
            byte_offset: Some(byte_offset),
            fragment: None,
        },
    );
}

fn lex_boost_suffix(
    query: &str,
    start: usize,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (Option<String>, usize) {
    if query.as_bytes().get(start) != Some(&b'^') {
        return (None, start);
    }
    let end = scan_fragment_end(query, start);
    let suffix = &query[start + 1..end];
    let (boost, trailing) = split_boost_component(suffix);
    if let Some(relative) = trailing {
        emit_trailing_boost_diagnostic(diagnostics, start + 1 + relative);
    }
    (Some(boost.to_owned()), end)
}

fn emit_trailing_boost_diagnostic(diagnostics: &mut Vec<QueryDiagnostic>, byte_offset: usize) {
    emit_diagnostic(
        diagnostics,
        QueryDiagnostic {
            kind: QueryDiagnosticKind::SyntaxRecovery,
            message: "syntax recovery: trailing boost suffix dropped".to_owned(),
            byte_offset: Some(byte_offset),
            fragment: None,
        },
    );
}

fn is_regex_fragment(raw: &str) -> bool {
    raw.starts_with('/')
}

fn unsupported_regex_token(
    byte_offset: usize,
    field: Option<String>,
    raw: &str,
    boost: Option<&str>,
) -> LexToken {
    let pattern = raw.strip_prefix('/').unwrap_or(raw);
    let pattern = pattern.strip_suffix('/').unwrap_or(pattern);
    LexToken::UnsupportedRegex {
        byte_offset,
        field,
        pattern: pattern.replace("\\/", "/"),
        boost: boost.map(str::to_owned),
    }
}

fn emit_regex_diagnostic(diagnostics: &mut Vec<QueryDiagnostic>, byte_offset: usize) {
    emit_diagnostic(
        diagnostics,
        QueryDiagnostic {
            kind: QueryDiagnosticKind::DroppedFragment,
            message: "unsupported regular-expression query dropped".to_owned(),
            byte_offset: Some(byte_offset),
            fragment: None,
        },
    );
}

pub(super) fn operator_atom(raw: &str, byte_offset: usize) -> LexToken {
    LexToken::Atom(AtomToken {
        raw: raw.to_owned(),
        field: None,
        delimiter: AtomDelimiter::None,
        slop: 0,
        prefix: false,
        boost: None,
        byte_offset,
    })
}

fn scan_fragment_end(query: &str, start: usize) -> usize {
    let mut escaped = false;
    for (relative, ch) in query[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch.is_whitespace() || matches!(ch, '(' | ')') {
            return start + relative;
        }
    }
    query.len()
}

fn unescape_fragment(fragment: &str) -> (String, bool) {
    let mut value = String::with_capacity(fragment.len());
    let mut escaped = false;
    let mut had_escape = false;
    for ch in fragment.chars() {
        if escaped {
            if !is_query_escape(ch) {
                value.push('\\');
            }
            value.push(ch);
            escaped = false;
            had_escape = true;
        } else if ch == '\\' {
            escaped = true;
        } else {
            value.push(ch);
        }
    }
    had_escape |= escaped;
    (value, had_escape)
}

fn is_query_escape(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '^' | '`' | ':' | '[' | ']' | '{' | '}' | '"' | '\'' | '(' | ')' | '\\' | '-'
        )
}

fn lex_quoted(
    query: &str,
    quote_offset: usize,
    delimiter: char,
    field: Option<String>,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> (AtomToken, usize) {
    let content_start = quote_offset + delimiter.len_utf8();
    let mut raw = String::new();
    let mut cursor = content_start;
    let mut escaped = false;
    let mut terminated = false;
    while let Some(ch) = query[cursor..].chars().next() {
        cursor += ch.len_utf8();
        if escaped {
            raw.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == delimiter {
            terminated = true;
            break;
        } else {
            raw.push(ch);
        }
    }
    if escaped {
        raw.push('\\');
    }
    if !terminated {
        emit_diagnostic(
            diagnostics,
            QueryDiagnostic {
                kind: QueryDiagnosticKind::SyntaxRecovery,
                message: "syntax recovery: unterminated quoted fragment".to_owned(),
                byte_offset: Some(quote_offset),
                fragment: Some(raw.clone()),
            },
        );
    }

    let suffix_start = cursor;
    let suffix_end = scan_fragment_end(query, suffix_start);
    let suffix = &query[suffix_start..suffix_end];
    let parsed_suffix = parse_phrase_suffix(suffix);
    if let Some(relative) = parsed_suffix.recovery_offset {
        let byte_offset = suffix_start + relative;
        if parsed_suffix.trailing_boost {
            emit_trailing_boost_diagnostic(diagnostics, byte_offset);
        } else {
            emit_typed_syntax_diagnostic(
                diagnostics,
                byte_offset,
                if parsed_suffix.relex_suffix {
                    "syntax recovery: unsupported phrase suffix retained as a separate fragment"
                } else {
                    "syntax recovery: unsupported phrase suffix dropped"
                },
            );
        }
    }
    let next = if parsed_suffix.relex_suffix {
        suffix_start
    } else {
        suffix_end
    };
    (
        AtomToken {
            raw,
            field,
            delimiter: AtomDelimiter::from_quote(delimiter),
            slop: parsed_suffix.slop,
            prefix: parsed_suffix.prefix,
            boost: parsed_suffix.boost.map(str::to_owned),
            byte_offset: quote_offset,
        },
        next,
    )
}

struct ParsedPhraseSuffix<'a> {
    slop: u32,
    prefix: bool,
    boost: Option<&'a str>,
    recovery_offset: Option<usize>,
    trailing_boost: bool,
    relex_suffix: bool,
}

fn parse_phrase_suffix(suffix: &str) -> ParsedPhraseSuffix<'_> {
    let mut slop = 0;
    let mut prefix = false;
    let mut consumed = 0;
    if suffix.starts_with('*') {
        prefix = true;
        consumed = 1;
    } else if let Some(rest) = suffix.strip_prefix('~') {
        let digit_count = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digit_count != 0 {
            let Ok(parsed_slop) = rest[..digit_count].parse() else {
                return ParsedPhraseSuffix {
                    slop,
                    prefix,
                    boost: None,
                    recovery_offset: Some(0),
                    trailing_boost: false,
                    relex_suffix: true,
                };
            };
            slop = parsed_slop;
            consumed = 1 + digit_count;
        } else {
            return ParsedPhraseSuffix {
                slop,
                prefix,
                boost: None,
                recovery_offset: Some(0),
                trailing_boost: false,
                relex_suffix: true,
            };
        }
    }
    let rest = &suffix[consumed..];
    if let Some(boost_suffix) = rest.strip_prefix('^') {
        let (boost, trailing) = split_boost_component(boost_suffix);
        if let Some(relative) = trailing {
            return ParsedPhraseSuffix {
                slop,
                prefix,
                boost: Some(boost),
                recovery_offset: Some(consumed + 1 + relative),
                trailing_boost: true,
                relex_suffix: false,
            };
        }
        return ParsedPhraseSuffix {
            slop,
            prefix,
            boost: Some(boost_suffix),
            recovery_offset: None,
            trailing_boost: false,
            relex_suffix: false,
        };
    }
    ParsedPhraseSuffix {
        slop,
        prefix,
        boost: None,
        recovery_offset: (!rest.is_empty()).then_some(consumed),
        trailing_boost: false,
        relex_suffix: false,
    }
}

fn find_unescaped(fragment: &str, needle: char) -> Option<usize> {
    let mut escaped = false;
    for (offset, ch) in fragment.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == needle {
            return Some(offset);
        }
    }
    None
}

fn split_boost(fragment: &str) -> (&str, Option<&str>, Option<usize>) {
    let Some(first) = find_unescaped(fragment, '^').filter(|offset| *offset != 0) else {
        return (fragment, None, None);
    };
    let suffix = &fragment[first + 1..];
    let (boost, trailing) = split_boost_component(suffix);
    (
        &fragment[..first],
        Some(boost),
        trailing.map(|offset| first + 1 + offset),
    )
}

fn split_boost_component(suffix: &str) -> (&str, Option<usize>) {
    let syntax_end = find_unescaped(suffix, '^').unwrap_or(suffix.len());
    let candidate = &suffix[..syntax_end];
    let integer_digits = candidate.bytes().take_while(u8::is_ascii_digit).count();
    if integer_digits == 0 {
        return (
            candidate,
            (syntax_end != suffix.len()).then_some(syntax_end),
        );
    }

    let mut valid_end = integer_digits;
    if candidate.as_bytes().get(integer_digits) == Some(&b'.') {
        let fraction_digits = candidate[integer_digits + 1..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
        if fraction_digits != 0 {
            valid_end = integer_digits + 1 + fraction_digits;
        }
    }
    let trailing = if valid_end != candidate.len() {
        Some(valid_end)
    } else {
        (syntax_end != suffix.len()).then_some(syntax_end)
    };
    (&candidate[..valid_end], trailing)
}

pub(super) fn trim_typed_literal(raw: &str) -> &str {
    if raw.len() >= 2 {
        let bytes = raw.as_bytes();
        if matches!(
            (bytes.first(), bytes.last()),
            (Some(&b'"'), Some(&b'"')) | (Some(&b'\''), Some(&b'\''))
        ) {
            return &raw[1..raw.len() - 1];
        }
    }
    raw
}
