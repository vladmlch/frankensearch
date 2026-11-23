use tantivy::Index;
use tantivy::tokenizer::{TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer};

/// Specialized tokenizer for cass lexical fields.
///
/// This preserves the current regex tokenizer semantics for:
/// - ASCII alphanumeric runs
/// - hyphen-joined ASCII alphanumeric runs
/// - contiguous runs in the exact CJK ranges covered by the old regex
///
/// A dedicated scanner avoids the regex DFA hot path that dominates rebuild
/// time on large corpora.
#[derive(Clone, Default)]
pub(super) struct CassTokenizer {
    token: Token,
}

pub(super) struct CassTokenStream<'a> {
    text: &'a str,
    cursor: usize,
    token: &'a mut Token,
}

#[inline]
fn is_cass_tokenizer_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{3040}'..='\u{309F}'
            | '\u{30A0}'..='\u{30FF}'
            | '\u{AC00}'..='\u{D7AF}'
            | '\u{3100}'..='\u{312F}'
            | '\u{3300}'..='\u{33FF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{20000}'..='\u{2A6DF}'
    )
}

#[inline]
pub(super) fn next_char_from(text: &str, offset: usize) -> Option<(char, usize)> {
    // ASCII fast-path: the tokenizer scans char-by-char, and its inputs (English, code, IDs) are
    // overwhelmingly ASCII. `offset` is always a char boundary (the tokenizer only advances by whole
    // chars), so a leading byte `< 128` is a complete single-byte UTF-8 char — return it directly,
    // skipping the `chars().next()` UTF-8 decode + `len_utf8()` recompute. Bit-identical to the slow
    // path for every input (asserted in `next_char_from_ascii_matches_decode`).
    let b = *text.as_bytes().get(offset)?;
    if b < 128 {
        return Some((b as char, offset + 1));
    }
    let ch = text[offset..].chars().next()?;
    Some((ch, offset + ch.len_utf8()))
}

/// Pre-fast-path [`next_char_from`], retained doc-hidden for the same-binary A/B and the parity
/// test. Not a shipping path.
#[doc(hidden)]
#[must_use]
pub fn next_char_from_slow(text: &str, offset: usize) -> Option<(char, usize)> {
    let ch = text[offset..].chars().next()?;
    Some((ch, offset + ch.len_utf8()))
}

/// Doc-hidden bench harness: walk `text` char-by-char via the shipped (ASCII-fast) `next_char_from`,
/// folding char values so the loop cannot be optimized away. Mirrors the tokenizer's scan cost.
#[doc(hidden)]
#[must_use]
pub fn cass_char_walk_fast(text: &str) -> u64 {
    let mut acc = 0u64;
    let mut off = 0usize;
    while let Some((ch, next)) = next_char_from(text, off) {
        acc = acc.wrapping_add(ch as u64);
        off = next;
    }
    acc
}

/// Doc-hidden bench harness: the same walk via [`next_char_from_slow`] (the pre-fast-path decode).
#[doc(hidden)]
#[must_use]
pub fn cass_char_walk_slow(text: &str) -> u64 {
    let mut acc = 0u64;
    let mut off = 0usize;
    while let Some((ch, next)) = next_char_from_slow(text, off) {
        acc = acc.wrapping_add(ch as u64);
        off = next;
    }
    acc
}

impl Tokenizer for CassTokenizer {
    type TokenStream<'a> = CassTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.token.reset();
        CassTokenStream {
            text,
            cursor: 0,
            token: &mut self.token,
        }
    }
}

impl CassTokenStream<'_> {
    fn scan_ascii_token(&self, mut cursor: usize) -> usize {
        let mut end = cursor;
        let mut last_was_ascii_alnum = false;

        while let Some((ch, next_cursor)) = next_char_from(self.text, cursor) {
            if ch.is_ascii_alphanumeric() {
                end = next_cursor;
                cursor = next_cursor;
                last_was_ascii_alnum = true;
                continue;
            }

            if ch == '-'
                && last_was_ascii_alnum
                && let Some((next_ch, _)) = next_char_from(self.text, next_cursor)
                && next_ch.is_ascii_alphanumeric()
            {
                end = next_cursor;
                cursor = next_cursor;
                last_was_ascii_alnum = false;
                continue;
            }

            break;
        }

        end
    }

    fn scan_cjk_token(&self, mut cursor: usize) -> usize {
        let mut end = cursor;
        while let Some((ch, next_cursor)) = next_char_from(self.text, cursor) {
            if !is_cass_tokenizer_cjk(ch) {
                break;
            }
            end = next_cursor;
            cursor = next_cursor;
        }
        end
    }
}

impl TokenStream for CassTokenStream<'_> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        self.token.position = self.token.position.wrapping_add(1);

        while let Some((ch, next_cursor)) = next_char_from(self.text, self.cursor) {
            if ch.is_ascii_alphanumeric() {
                let offset_from = self.cursor;
                let offset_to = self.scan_ascii_token(self.cursor);
                self.token.offset_from = offset_from;
                self.token.offset_to = offset_to;
                self.token.text.push_str(&self.text[offset_from..offset_to]);
                self.cursor = offset_to;
                return true;
            }

            if is_cass_tokenizer_cjk(ch) {
                let offset_from = self.cursor;
                let offset_to = self.scan_cjk_token(next_cursor);
                self.token.offset_from = offset_from;
                self.token.offset_to = offset_to;
                self.token.text.push_str(&self.text[offset_from..offset_to]);
                self.cursor = offset_to;
                return true;
            }

            self.cursor = next_cursor;
        }

        false
    }

    fn token(&self) -> &Token {
        self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        self.token
    }
}

// ─── HyphenDecompose token filter ────────────────────────────────────────────
//
// When a token contains an interior hyphen (e.g. `bd-q3fy`), the filter emits
// the compound form **and** the individual sub-parts so that both exact and
// partial matches work.  Tokens without hyphens pass through unchanged.

/// A [`TokenFilter`] that decomposes hyphenated tokens into their compound form
/// plus each hyphen-delimited part.
///
/// Given the token `bd-q3fy`, the stream will yield:
///   1. `bd-q3fy`   (compound, same position)
///   2. `bd`         (part, same position)
///   3. `q3fy`       (part, same position)
#[derive(Clone)]
pub struct HyphenDecompose;

impl TokenFilter for HyphenDecompose {
    type Tokenizer<T: Tokenizer> = HyphenDecomposeFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> HyphenDecomposeFilter<T> {
        HyphenDecomposeFilter {
            inner: tokenizer,
            pending: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct HyphenDecomposeFilter<T> {
    inner: T,
    pending: Vec<Token>,
}

impl<T: Tokenizer> Tokenizer for HyphenDecomposeFilter<T> {
    type TokenStream<'a> = HyphenDecomposeStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.pending.clear();
        HyphenDecomposeStream {
            tail: self.inner.token_stream(text),
            pending: &mut self.pending,
        }
    }
}

pub struct HyphenDecomposeStream<'a, T> {
    tail: T,
    pending: &'a mut Vec<Token>,
}

impl<T: TokenStream> HyphenDecomposeStream<'_, T> {
    /// If the current upstream token contains hyphens, push the compound form
    /// and sub-parts (in reverse order so `.pop()` yields them in order).
    fn decompose(&mut self) {
        let token = self.tail.token();
        if !token.text.contains('-') {
            return;
        }
        let parts: Vec<&str> = token.text.split('-').filter(|s| !s.is_empty()).collect();
        if parts.len() < 2 {
            return;
        }
        // Push sub-parts in reverse so pop() yields them left-to-right.
        for &part in parts.iter().rev() {
            self.pending.push(Token {
                text: part.to_owned(),
                position: token.position,
                offset_from: token.offset_from,
                offset_to: token.offset_to,
                position_length: token.position_length,
            });
        }
        // Push the compound form last so it is popped first.
        self.pending.push(token.clone());
    }
}

impl<T: TokenStream> TokenStream for HyphenDecomposeStream<'_, T> {
    fn advance(&mut self) -> bool {
        // Drain any buffered tokens from a previous decomposition first.
        self.pending.pop();
        if !self.pending.is_empty() {
            return true;
        }

        if !self.tail.advance() {
            return false;
        }

        self.decompose();
        // If decompose produced tokens, the first will come from pending.
        // Otherwise the plain upstream token is returned via `token()`.
        true
    }

    fn token(&self) -> &Token {
        self.pending.last().unwrap_or_else(|| self.tail.token())
    }

    fn token_mut(&mut self) -> &mut Token {
        self.pending
            .last_mut()
            .unwrap_or_else(|| self.tail.token_mut())
    }
}

// ─── CJK Bigram token filter ────────────────────────────────────────────────
//
// CJK scripts (Chinese, Japanese kanji, Korean hangul) are written without
// whitespace between words.  A single CJK token such as "搜索引擎" must be
// decomposed into overlapping character bigrams ("搜索", "索引", "引擎") so
// that sub-string queries can match.  Non-CJK tokens pass through unchanged.

/// Returns `true` when `c` falls in a CJK-relevant Unicode range.
#[inline]
pub(super) fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
        | '\u{3100}'..='\u{312F}' // Bopomofo
        | '\u{3300}'..='\u{33FF}' // CJK Compatibility
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{20000}'..='\u{2A6DF}' // CJK Extension B
    )
}

/// One-pass CJK-token collection for the bigram decomposer: decode `text`
/// **once**, validating that every char is CJK while collecting them. Returns
/// `Some(chars)` iff the token is entirely CJK with ≥2 chars (the only case
/// that decomposes into bigrams); `None` otherwise. A non-CJK token bails on
/// its first char *before* allocating, so the common ASCII/Latin token pays
/// nothing.
///
/// This fuses the two full UTF-8 decode passes the naive form makes — the
/// guard scan `chars().all(is_cjk)` followed by `chars().collect()` — into a
/// single pass. Byte-identical to [`cass_cjk_collect_slow`]: `decompose_cjk`
/// only emits bigrams when all chars are CJK and the count is ≥2, exactly the
/// `Some` case here (empty, non-CJK, and single-CJK all map to `None`).
#[doc(hidden)]
#[must_use]
pub fn cass_cjk_collect_fast(text: &str) -> Option<Vec<char>> {
    let mut chars = text.chars();
    // First char decides the common case: a non-CJK lead (ASCII/Latin/space)
    // rejects immediately with no allocation.
    let first = chars.next()?;
    if !is_cjk(first) {
        return None;
    }
    let mut buf: Vec<char> = Vec::with_capacity(text.len() / 3 + 1);
    buf.push(first);
    for c in chars {
        if !is_cjk(c) {
            return None;
        }
        buf.push(c);
    }
    if buf.len() < 2 {
        // Single CJK character: unchanged (emitted as its own unigram).
        return None;
    }
    Some(buf)
}

/// The pre-fusion two-pass form, retained as the bench oracle for
/// [`cass_cjk_collect_fast`]: decode `text` once to validate (`all`) and again
/// to `collect`.
#[doc(hidden)]
#[must_use]
pub fn cass_cjk_collect_slow(text: &str) -> Option<Vec<char>> {
    if text.is_empty() || !text.chars().all(is_cjk) {
        return None;
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 {
        return None;
    }
    Some(chars)
}

/// Pre-optimization CJK bigram materialization retained for same-binary benchmarks.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub fn cass_cjk_bigrams_staged(text: &str, token: &Token, pending: &mut Vec<Token>) {
    let Some(chars) = cass_cjk_collect_fast(text) else {
        return;
    };
    let mut bigrams: Vec<Token> = Vec::with_capacity(chars.len());
    for i in (0..chars.len() - 1).rev() {
        let mut bigram = String::with_capacity(8);
        bigram.push(chars[i]);
        bigram.push(chars[i + 1]);
        bigrams.push(Token {
            text: bigram,
            position: token.position,
            offset_from: token.offset_from,
            offset_to: token.offset_to,
            position_length: token.position_length,
        });
    }
    pending.extend(bigrams);
}

/// Direct-to-pending CJK bigram materialization candidate for same-binary benchmarks.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub fn cass_cjk_bigrams_direct(text: &str, token: &Token, pending: &mut Vec<Token>) {
    let Some(chars) = cass_cjk_collect_fast(text) else {
        return;
    };
    pending.reserve(chars.len() - 1);
    for i in (0..chars.len() - 1).rev() {
        let mut bigram = String::with_capacity(8);
        bigram.push(chars[i]);
        bigram.push(chars[i + 1]);
        pending.push(Token {
            text: bigram,
            position: token.position,
            offset_from: token.offset_from,
            offset_to: token.offset_to,
            position_length: token.position_length,
        });
    }
}

/// A [`TokenFilter`] that decomposes tokens consisting entirely of CJK
/// characters into overlapping character bigrams.
///
/// Given the token `搜索引擎`, the stream will yield:
///   `搜索`, `索引`, `引擎`
///
/// A single CJK character emits itself as a unigram.
/// Tokens that contain no CJK characters pass through unchanged.
#[derive(Clone)]
pub struct CjkBigramDecompose;

impl TokenFilter for CjkBigramDecompose {
    type Tokenizer<T: Tokenizer> = CjkBigramDecomposeFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> CjkBigramDecomposeFilter<T> {
        CjkBigramDecomposeFilter {
            inner: tokenizer,
            pending: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct CjkBigramDecomposeFilter<T> {
    inner: T,
    pending: Vec<Token>,
}

impl<T: Tokenizer> Tokenizer for CjkBigramDecomposeFilter<T> {
    type TokenStream<'a> = CjkBigramDecomposeStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.pending.clear();
        CjkBigramDecomposeStream {
            tail: self.inner.token_stream(text),
            pending: &mut self.pending,
        }
    }
}

pub struct CjkBigramDecomposeStream<'a, T> {
    tail: T,
    pending: &'a mut Vec<Token>,
}

impl<T: TokenStream> CjkBigramDecomposeStream<'_, T> {
    fn decompose_cjk(&mut self) {
        let token = self.tail.token();

        // Only decompose tokens that are entirely CJK with ≥2 chars. One fused
        // pass validates + collects (most cass tokens are ASCII and reject on
        // their first char with no allocation); empty / non-CJK / single-CJK
        // tokens all yield `None` and are left unchanged.
        let Some(chars) = cass_cjk_collect_fast(&token.text) else {
            return;
        };

        // Build bigrams in reverse so `.pop()` yields them left-to-right.
        // We replace the original token entirely with bigrams.
        self.pending.reserve(chars.len() - 1);
        for i in (0..chars.len() - 1).rev() {
            let mut bigram = String::with_capacity(8);
            bigram.push(chars[i]);
            bigram.push(chars[i + 1]);
            self.pending.push(Token {
                text: bigram,
                position: token.position,
                offset_from: token.offset_from,
                offset_to: token.offset_to,
                position_length: token.position_length,
            });
        }

        // The first bigram is popped first (it's at the end of the vec).
    }
}

impl<T: TokenStream> TokenStream for CjkBigramDecomposeStream<'_, T> {
    fn advance(&mut self) -> bool {
        self.pending.pop();
        if !self.pending.is_empty() {
            return true;
        }

        if !self.tail.advance() {
            return false;
        }

        self.decompose_cjk();
        true
    }

    fn token(&self) -> &Token {
        self.pending.last().unwrap_or_else(|| self.tail.token())
    }

    fn token_mut(&mut self) -> &mut Token {
        self.pending
            .last_mut()
            .unwrap_or_else(|| self.tail.token_mut())
    }
}

/// A cass-specific normalization filter that preserves the shipping CASS
/// analyzer's inclusive 256-byte admission contract.
///
/// This intentionally differs at exactly 256 bytes from Tantivy's
/// `RemoveLongFilter::limit(256)`, whose predicate retains only tokens shorter
/// than the configured limit. CASS has historically retained 256-byte tokens
/// and dropped tokens starting at 257 bytes, so that behavior is the oracle.
///
/// `CassTokenizer` only emits ASCII alphanumeric runs (optionally hyphenated)
/// plus CJK runs, so ASCII-only in-place lowercasing is behaviorally identical
/// to Tantivy's generic lowercaser here while avoiding its broader Unicode path.
#[derive(Clone, Copy, Default)]
pub struct CassNormalizeAndLimit;

impl TokenFilter for CassNormalizeAndLimit {
    type Tokenizer<T: Tokenizer> = CassNormalizeAndLimitFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> CassNormalizeAndLimitFilter<T> {
        CassNormalizeAndLimitFilter { inner: tokenizer }
    }
}

#[derive(Clone)]
pub struct CassNormalizeAndLimitFilter<T> {
    inner: T,
}

impl<T: Tokenizer> Tokenizer for CassNormalizeAndLimitFilter<T> {
    type TokenStream<'a> = CassNormalizeAndLimitStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CassNormalizeAndLimitStream {
            tail: self.inner.token_stream(text),
        }
    }
}

pub struct CassNormalizeAndLimitStream<T> {
    tail: T,
}

impl<T: TokenStream> TokenStream for CassNormalizeAndLimitStream<T> {
    fn advance(&mut self) -> bool {
        while self.tail.advance() {
            let token = self.tail.token_mut();
            if token.text.len() > 256 {
                continue;
            }
            token.text.make_ascii_lowercase();
            return true;
        }

        false
    }

    fn token(&self) -> &Token {
        self.tail.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.tail.token_mut()
    }
}

/// Register the tokenizers used by cass-compatible lexical fields.
///
/// `hyphen_normalize` pipeline:
///   1. `CassTokenizer` — matches ASCII alphanumeric runs (with hyphens)
///      **and** CJK character runs as separate tokens.
///   2. `HyphenDecompose` — for each hyphenated token, emits the compound
///      form *and* the individual sub-parts (all at the same position) so
///      both exact ID searches and partial-word searches match.
///   3. `CjkBigramDecompose` — decomposes CJK-only tokens into overlapping
///      character bigrams so that sub-string queries work for Chinese,
///      Japanese, and Korean text.
///   4. `CassNormalizeAndLimit` — applies ASCII lowercase plus the shipping
///      CASS inclusive 256-byte limit (not Tantivy's strict `< 256` predicate).
///
/// `prefix_normalize` is used only for generated edge-ngram fields. Those
/// values are whitespace-separated prefix terms produced by
/// `cass_generate_edge_ngrams`, so they cannot contain hyphens. Keeping the CJK
/// bigram and normalization filters preserves prefix-field query behavior while
/// avoiding the per-token `HyphenDecompose` layer on the highest-volume basic
/// postings fields.
pub fn cass_ensure_tokenizer(index: &mut Index) {
    let analyzer = TextAnalyzer::builder(CassTokenizer::default())
        .filter(HyphenDecompose)
        .filter(CjkBigramDecompose)
        .filter(CassNormalizeAndLimit)
        .build();
    index.tokenizers().register("hyphen_normalize", analyzer);
    let prefix_analyzer = TextAnalyzer::builder(CassTokenizer::default())
        .filter(CjkBigramDecompose)
        .filter(CassNormalizeAndLimit)
        .build();
    index
        .tokenizers()
        .register("prefix_normalize", prefix_analyzer);
}
