use tantivy::tokenizer::{TextAnalyzer, Token, TokenStream, Tokenizer};

/// Fused equivalent of Tantivy's `SimpleTokenizer` followed by `LowerCaser`.
///
/// ASCII characters are classified directly from their byte and each token is
/// lowercased without the generic lowercaser's extra `is_ascii` scan. Non-ASCII
/// characters retain the exact `char::is_alphanumeric` and `char::to_lowercase`
/// behavior of the former two-stage analyzer.
#[derive(Clone, Default)]
struct FrankensearchTokenizer {
    token: Token,
}

struct FrankensearchTokenStream<'a> {
    text: &'a str,
    cursor: usize,
    token: &'a mut Token,
}

#[inline]
fn tokenizer_next_char(text: &str, offset: usize) -> Option<(char, usize)> {
    let remaining = text.get(offset..)?;
    let first = *remaining.as_bytes().first()?;
    if first.is_ascii() {
        Some((char::from(first), offset + 1))
    } else {
        let ch = remaining.chars().next()?;
        Some((ch, offset + ch.len_utf8()))
    }
}

#[inline]
fn tokenizer_is_alphanumeric(ch: char) -> bool {
    if ch.is_ascii() {
        ch.is_ascii_alphanumeric()
    } else {
        ch.is_alphanumeric()
    }
}

impl Tokenizer for FrankensearchTokenizer {
    type TokenStream<'a> = FrankensearchTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.token.reset();
        FrankensearchTokenStream {
            text,
            cursor: 0,
            token: &mut self.token,
        }
    }
}

impl TokenStream for FrankensearchTokenStream<'_> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        self.token.position = self.token.position.wrapping_add(1);

        while let Some((ch, next_cursor)) = tokenizer_next_char(self.text, self.cursor) {
            if !tokenizer_is_alphanumeric(ch) {
                self.cursor = next_cursor;
                continue;
            }

            let offset_from = self.cursor;
            let mut offset_to = next_cursor;
            let mut resume_at = next_cursor;
            let mut all_ascii = ch.is_ascii();
            while let Some((next_ch, after_next)) = tokenizer_next_char(self.text, resume_at) {
                if !tokenizer_is_alphanumeric(next_ch) {
                    resume_at = after_next;
                    break;
                }
                all_ascii &= next_ch.is_ascii();
                offset_to = after_next;
                resume_at = after_next;
            }

            self.token.offset_from = offset_from;
            self.token.offset_to = offset_to;
            let source = &self.text[offset_from..offset_to];
            if all_ascii {
                self.token.text.push_str(source);
                self.token.text.make_ascii_lowercase();
            } else {
                for source_char in source.chars() {
                    self.token.text.extend(source_char.to_lowercase());
                }
            }
            self.cursor = resume_at;
            return true;
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

/// Build and register the custom tokenizer.
///
/// Semantics match `SimpleTokenizer` (split on non-alphanumeric characters)
/// followed by `LowerCaser`. `POL-358` therefore becomes `pol`, `358`. For
/// hyphen-preserving tokenization see `cass_ensure_tokenizer`.
pub fn build_tokenizer() -> TextAnalyzer {
    TextAnalyzer::from(FrankensearchTokenizer::default())
}

/// Return the production analyzer for a same-binary benchmark comparator.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn default_tokenizer_for_bench() -> TextAnalyzer {
    build_tokenizer()
}
