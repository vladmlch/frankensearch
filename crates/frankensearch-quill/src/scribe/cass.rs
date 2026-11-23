use crate::schema::Analyzer as AnalyzerKind;

use super::analyzer::{AnalyzedToken, AnalyzedTokenRef, TokenAnalyzer, sealed};
use super::tokenizer::{FrankensearchTokenizer, next_token_position, tokenizer_next_char};

/// Maximum token length retained by the native CASS analyzer pipeline.
///
/// The shipping CASS analyzer retains exactly 256 UTF-8 bytes and drops 257.
/// This inclusive boundary intentionally differs from Tantivy's strict
/// `RemoveLongFilter::limit(256)` predicate and is much smaller than Quill's
/// global [`MAX_TERM_BYTES`](crate::grimoire::MAX_TERM_BYTES) admission ceiling.
pub const CASS_MAX_TOKEN_BYTES: usize = 256;

/// Maximum Unicode-scalar prefix length generated for CASS prefix fields.
pub const CASS_MAX_EDGE_NGRAM_CHARS: usize = 20;

/// Native implementation of the two CASS analyzer pipelines.
///
/// [`AnalyzerKind::CassHyphenNormalize`] applies the shipping
/// `CassTokenizer -> HyphenDecompose -> CjkBigramDecompose ->
/// CassNormalizeAndLimit` stages. [`AnalyzerKind::CassPrefixNormalize`] omits
/// only hyphen decomposition. One scratch token is reused for the complete
/// stream; compound alternatives, parts, and CJK bigrams are sent directly to
/// the callback without a staging allocation.
#[derive(Debug, Clone, Default)]
pub struct CassAnalyzer {
    token: AnalyzedToken,
}

impl sealed::Sealed for CassAnalyzer {}

/// Every analyzer pipeline compiled into this build, in one family.
///
/// `ColumnarAccumulator` is generic over its analyzer but monomorphizes to a
/// single type per accumulator, so a family that answers `supports` for only
/// one kind cannot ingest a schema that mixes pipelines —
/// [`crate::schema::CASS_SEMANTIC_SCHEMA`] analyzes `title`/`content` with the
/// CASS hyphen pipeline and the `_prefix` columns with the CASS prefix
/// pipeline. This family is the default ingest analyzer so every compiled
/// descriptor is indexable.
///
/// Dispatch is per analyzed FIELD VALUE, not per token: the kind is a
/// parameter of `analyze`/`analyze_borrowed`, which each tokenize a whole
/// field. `analyze_borrowed` is forwarded rather than inherited so the default
/// pipeline keeps its borrowed fast path into the term interner — inheriting
/// the trait's owned-token bridge would silently regress shipping ingest.
#[derive(Debug, Clone, Default)]
pub struct CompiledAnalyzerFamily {
    default: FrankensearchTokenizer,
    cass: CassAnalyzer,
}

impl sealed::Sealed for CompiledAnalyzerFamily {}

impl TokenAnalyzer for CompiledAnalyzerFamily {
    fn supports(&self, analyzer: AnalyzerKind) -> bool {
        match analyzer {
            AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams => {
                self.default.supports(analyzer)
            }
            AnalyzerKind::CassHyphenNormalize | AnalyzerKind::CassPrefixNormalize => {
                self.cass.supports(analyzer)
            }
        }
    }

    fn analyze(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        match analyzer {
            AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams => {
                self.default.analyze(analyzer, text, sink);
            }
            AnalyzerKind::CassHyphenNormalize | AnalyzerKind::CassPrefixNormalize => {
                self.cass.analyze(analyzer, text, sink);
            }
        }
    }

    fn analyze_borrowed(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn for<'token> FnMut(AnalyzedTokenRef<'token>),
    ) {
        match analyzer {
            AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams => {
                self.default.analyze_borrowed(analyzer, text, sink);
            }
            AnalyzerKind::CassHyphenNormalize | AnalyzerKind::CassPrefixNormalize => {
                self.cass.analyze_borrowed(analyzer, text, sink);
            }
        }
    }

    fn bytes_reserved(&self) -> usize {
        self.default
            .bytes_reserved()
            .saturating_add(self.cass.bytes_reserved())
    }

    fn reset(&mut self) {
        self.default.reset();
        self.cass.reset();
    }
}

/// Exact CJK ranges recognized by the incumbent CASS tokenizer.
///
/// Keep this predicate shared inside Quill: broadening it to later Unicode
/// extensions would change durable term bytes and therefore requires an
/// explicit language-contract revision.
#[inline]
pub fn is_cass_cjk(ch: char) -> bool {
    matches!(
        ch,
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

fn cass_ascii_token_end(text: &str, mut cursor: usize) -> usize {
    let mut end = cursor;
    let mut last_was_ascii_alphanumeric = false;

    while let Some((ch, next_cursor)) = tokenizer_next_char(text, cursor) {
        if ch.is_ascii_alphanumeric() {
            end = next_cursor;
            cursor = next_cursor;
            last_was_ascii_alphanumeric = true;
            continue;
        }
        if ch == '-'
            && last_was_ascii_alphanumeric
            && let Some((next_ch, _)) = tokenizer_next_char(text, next_cursor)
            && next_ch.is_ascii_alphanumeric()
        {
            end = next_cursor;
            cursor = next_cursor;
            last_was_ascii_alphanumeric = false;
            continue;
        }
        break;
    }
    end
}

fn cass_cjk_token_end(text: &str, mut cursor: usize) -> usize {
    let mut end = cursor;
    while let Some((ch, next_cursor)) = tokenizer_next_char(text, cursor) {
        if !is_cass_cjk(ch) {
            break;
        }
        end = next_cursor;
        cursor = next_cursor;
    }
    end
}

impl CassAnalyzer {
    fn emit_normalized(
        &mut self,
        source: &str,
        position: u32,
        offset_from: usize,
        offset_to: usize,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        if source.len() > CASS_MAX_TOKEN_BYTES {
            return;
        }
        self.token.text.clear();
        self.token.text.push_str(source);
        self.token.text.make_ascii_lowercase();
        self.token.position = position;
        self.token.offset_from = offset_from;
        self.token.offset_to = offset_to;
        self.token.position_length = 1;
        sink(&self.token);
    }

    fn emit_cjk(
        &mut self,
        source: &str,
        position: u32,
        offset_from: usize,
        offset_to: usize,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        let mut chars = source.chars();
        let Some(mut left) = chars.next() else {
            return;
        };
        let Some(mut right) = chars.next() else {
            self.emit_normalized(source, position, offset_from, offset_to, sink);
            return;
        };

        loop {
            self.token.text.clear();
            self.token.text.push(left);
            self.token.text.push(right);
            self.token.position = position;
            self.token.offset_from = offset_from;
            self.token.offset_to = offset_to;
            self.token.position_length = 1;
            sink(&self.token);

            left = right;
            let Some(next) = chars.next() else {
                break;
            };
            right = next;
        }
    }

    fn emit_ascii(
        &mut self,
        analyzer: AnalyzerKind,
        source: &str,
        position: u32,
        offset_from: usize,
        offset_to: usize,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        self.emit_normalized(source, position, offset_from, offset_to, sink);
        if analyzer != AnalyzerKind::CassHyphenNormalize || !source.contains('-') {
            return;
        }
        for part in source.split('-').filter(|part| !part.is_empty()) {
            self.emit_normalized(part, position, offset_from, offset_to, sink);
        }
    }
}

impl TokenAnalyzer for CassAnalyzer {
    fn supports(&self, analyzer: AnalyzerKind) -> bool {
        matches!(
            analyzer,
            AnalyzerKind::CassHyphenNormalize | AnalyzerKind::CassPrefixNormalize
        )
    }

    fn analyze(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        debug_assert!(self.supports(analyzer));
        let mut cursor = 0;
        let mut position = 0_u32;

        while let Some((ch, next_cursor)) = tokenizer_next_char(text, cursor) {
            let (offset_to, is_cjk) = if ch.is_ascii_alphanumeric() {
                (cass_ascii_token_end(text, cursor), false)
            } else if is_cass_cjk(ch) {
                (cass_cjk_token_end(text, next_cursor), true)
            } else {
                cursor = next_cursor;
                continue;
            };
            let source = &text[cursor..offset_to];
            if is_cjk {
                self.emit_cjk(source, position, cursor, offset_to, sink);
            } else {
                self.emit_ascii(analyzer, source, position, cursor, offset_to, sink);
            }
            position = next_token_position(position);
            cursor = offset_to;
        }
    }

    fn bytes_reserved(&self) -> usize {
        self.token.text.capacity()
    }

    fn reset(&mut self) {
        self.token.text.clear();
        self.token.position = 0;
        self.token.offset_from = 0;
        self.token.offset_to = 0;
        self.token.position_length = 0;
    }
}

fn push_cass_prefix(out: &mut String, prefix: &str) {
    if !out.is_empty() {
        out.push(' ');
    }
    out.push_str(prefix);
}

/// Generate the shipping CASS edge-prefix field value.
///
/// Every alphanumeric word contributes prefixes of 2 through 20 Unicode
/// scalar values, in word and prefix-length order, separated by one ASCII
/// space. Source case is preserved; the prefix analyzer lowercases later.
#[must_use]
pub fn cass_generate_edge_ngrams(text: &str) -> String {
    const MAX_BOUNDARIES: usize = CASS_MAX_EDGE_NGRAM_CHARS + 1;
    let mut prefixes = String::with_capacity(text.len().saturating_mul(2));
    for word in text.split(|ch: char| !ch.is_alphanumeric()) {
        if word.is_ascii() {
            let upper = word.len().min(CASS_MAX_EDGE_NGRAM_CHARS);
            for end in 2..=upper {
                push_cass_prefix(&mut prefixes, &word[..end]);
            }
            continue;
        }

        let mut boundaries = [0_usize; MAX_BOUNDARIES];
        let mut boundary_count = 0;
        for (byte_index, _) in word.char_indices() {
            if boundary_count == MAX_BOUNDARIES {
                break;
            }
            boundaries[boundary_count] = byte_index;
            boundary_count += 1;
        }
        if boundary_count < MAX_BOUNDARIES {
            boundaries[boundary_count] = word.len();
            boundary_count += 1;
        }
        if boundary_count < 3 {
            continue;
        }
        for &end in &boundaries[2..boundary_count] {
            push_cass_prefix(&mut prefixes, &word[..end]);
        }
    }
    prefixes
}

/// Return a Unicode-scalar-bounded CASS preview.
///
/// The first `max_chars` scalar values are copied byte-for-byte and `…` is
/// appended exactly when additional input remains.
#[must_use]
pub fn cass_build_preview(content: &str, max_chars: usize) -> String {
    let mut cut = content.len();
    for (count, (byte_index, _)) in content.char_indices().enumerate() {
        if count == max_chars {
            cut = byte_index;
            break;
        }
    }
    let truncated = cut < content.len();
    let mut preview = String::with_capacity(cut + if truncated { '…'.len_utf8() } else { 0 });
    preview.push_str(&content[..cut]);
    if truncated {
        preview.push('…');
    }
    preview
}
