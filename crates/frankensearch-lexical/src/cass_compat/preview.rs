fn cass_push_prefix_term(out: &mut String, term: &str) {
    if !out.is_empty() {
        out.push(' ');
    }
    out.push_str(term);
}

/// Generate edge n-grams from text for prefix search acceleration.
///
/// For each word, emits its length-2..N prefixes (capped at 20 chars) space-separated. The original
/// re-decoded every word with `char_indices` to find prefix boundaries — a **second** UTF-8 decode
/// on top of the `split` that already scanned the text. For an **ASCII word** (the common case for
/// English/code/IDs), char boundaries are byte positions, so the prefixes are `word[..2..=min(len,
/// 20)]` directly — no `char_indices` decode. Non-ASCII words take the original boundary-collecting
/// path. Byte-for-byte identical output (`cass_generate_edge_ngrams_matches_slow`).
#[must_use]
pub fn cass_generate_edge_ngrams(text: &str) -> String {
    const MAX_NGRAM_INDICES: usize = 21;
    let mut ngrams = String::with_capacity(text.len() * 2);
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        // ASCII fast-path: byte positions ARE char boundaries, so skip the `char_indices` re-decode
        // and slice prefixes directly. `2..=upper` is empty (no prefixes) when `upper < 2`.
        if word.is_ascii() {
            let upper = word.len().min(MAX_NGRAM_INDICES - 1);
            for end in 2..=upper {
                cass_push_prefix_term(&mut ngrams, &word[..end]);
            }
            continue;
        }

        let mut indices = [0usize; MAX_NGRAM_INDICES];
        let mut index_count = 0usize;

        for (i, _) in word.char_indices() {
            if index_count == MAX_NGRAM_INDICES {
                break;
            }
            indices[index_count] = i;
            index_count += 1;
        }

        if index_count < MAX_NGRAM_INDICES {
            indices[index_count] = word.len();
            index_count += 1;
        }

        if index_count < 3 {
            continue;
        }
        for &end_idx in &indices[2..index_count] {
            cass_push_prefix_term(&mut ngrams, &word[..end_idx]);
        }
    }
    ngrams
}

/// Pre-ASCII-fast-path [`cass_generate_edge_ngrams`], retained doc-hidden for the same-binary A/B +
/// parity test.
#[doc(hidden)]
#[must_use]
pub fn cass_generate_edge_ngrams_slow(text: &str) -> String {
    const MAX_NGRAM_INDICES: usize = 21;
    let mut ngrams = String::with_capacity(text.len() * 2);
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let mut indices = [0usize; MAX_NGRAM_INDICES];
        let mut index_count = 0usize;
        for (i, _) in word.char_indices() {
            if index_count == MAX_NGRAM_INDICES {
                break;
            }
            indices[index_count] = i;
            index_count += 1;
        }
        if index_count < MAX_NGRAM_INDICES {
            indices[index_count] = word.len();
            index_count += 1;
        }
        if index_count < 3 {
            continue;
        }
        for &end_idx in &indices[2..index_count] {
            cass_push_prefix_term(&mut ngrams, &word[..end_idx]);
        }
    }
    ngrams
}

/// Build a bounded-length preview from message content.
///
/// Takes the first `max_chars` characters and appends `…` if the content is longer. One
/// `char_indices` scan finds the cut byte-offset, then a single `push_str` bulk-copies the prefix
/// into a pre-sized buffer — versus pushing `max_chars` chars individually (each re-encoding the
/// char and growing an unallocated `String` through repeated reallocations). Byte-for-byte identical
/// output (`cass_build_preview_matches_slow`).
#[must_use]
pub fn cass_build_preview(content: &str, max_chars: usize) -> String {
    // Byte offset just past the `max_chars`-th char, or `content.len()` if there are fewer.
    let mut cut = content.len();
    let mut count = 0usize;
    // Pinned-oracle scan kept in its shipped shape (single `char_indices` walk).
    #[allow(clippy::explicit_counter_loop)]
    for (byte_idx, _) in content.char_indices() {
        if count == max_chars {
            cut = byte_idx;
            break;
        }
        count += 1;
    }
    let truncated = cut < content.len();
    let mut out = String::with_capacity(cut + if truncated { '…'.len_utf8() } else { 0 });
    out.push_str(&content[..cut]);
    if truncated {
        out.push('…');
    }
    out
}

/// Pre-bulk-copy [`cass_build_preview`], retained doc-hidden for the same-binary A/B + parity test.
#[doc(hidden)]
#[must_use]
pub fn cass_build_preview_slow(content: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut chars = content.chars();
    for _ in 0..max_chars {
        if let Some(ch) = chars.next() {
            out.push(ch);
        } else {
            return out;
        }
    }
    if chars.next().is_some() {
        out.push('…');
    }
    out
}

#[must_use]
pub(super) fn cass_build_content_prefix_and_preview(content: &str) -> (String, String) {
    const PREVIEW_MAX_CHARS: usize = 400;
    const CONTENT_PREFIX_MAX_BYTES: usize = 4 * 1024;

    let prefix_source = cass_prefix_source(content, CONTENT_PREFIX_MAX_BYTES);
    (
        cass_generate_edge_ngrams(prefix_source),
        cass_build_preview(content, PREVIEW_MAX_CHARS),
    )
}

/// The largest char-boundary prefix of `content` that is `≤ max_bytes` bytes.
///
/// The original walked `char_indices` forward from byte 0 — `O(max_bytes)` char decodes just to locate
/// the boundary near `max_bytes`. Since UTF-8 chars are ≤ 4 bytes, the largest boundary `≤ max_bytes`
/// is at most 3 bytes below `max_bytes`, so a backward `is_char_boundary` walk finds it in **≤ 4
/// steps** — O(1). Byte-for-byte identical result (`cass_prefix_source_matches_slow`).
pub(super) fn cass_prefix_source(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }
    // `max_bytes < content.len()` here, so it is a valid index; `is_char_boundary(0)` is always true,
    // so the walk terminates within one UTF-8 char width.
    let mut end = max_bytes;
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    &content[..end]
}

/// Pre-O(1) [`cass_prefix_source`], retained doc-hidden for the same-binary A/B + parity test.
#[doc(hidden)]
#[must_use]
pub fn cass_prefix_source_slow(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }
    let mut end = 0usize;
    for (byte_idx, _) in content.char_indices() {
        if byte_idx > max_bytes {
            break;
        }
        end = byte_idx;
    }
    &content[..end]
}

/// Doc-hidden bench wrapper (the shipped `cass_prefix_source` is private).
#[doc(hidden)]
#[must_use]
pub fn cass_prefix_source_fast_bench(content: &str, max_bytes: usize) -> usize {
    cass_prefix_source(content, max_bytes).len()
}
