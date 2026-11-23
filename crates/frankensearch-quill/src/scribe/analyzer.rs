use thiserror::Error;

use crate::grimoire::MAX_TERM_BYTES;
use crate::schema::Analyzer as AnalyzerKind;

/// One token emitted by a Quill analyzer.
///
/// Offsets are half-open byte offsets into the original input. Positions are
/// `u32` end to end, matching the on-disk postings contract and making
/// truncation impossible at the analyzer/accumulator boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnalyzedToken {
    /// Normalized token text.
    pub text: String,
    /// Logical token position, starting at zero.
    pub position: u32,
    /// Source byte offset of the first token byte.
    pub offset_from: usize,
    /// Source byte offset immediately after the token.
    pub offset_to: usize,
    /// Position span. The default analyzer always emits one.
    pub position_length: usize,
}

/// Borrowed normalized token emitted directly into an ingest consumer.
///
/// The default tokenizer uses the original input slice when an ASCII token is
/// already lowercase, and its retained scratch only when normalization changes
/// bytes. The view is valid only for the duration of the sink call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalyzedTokenRef<'a> {
    /// Normalized token text.
    pub text: &'a str,
    /// Logical token position, starting at zero.
    pub position: u32,
    /// Source byte offset of the first token byte.
    pub offset_from: usize,
    /// Source byte offset immediately after the token.
    pub offset_to: usize,
    /// Position span. The default analyzer always emits one.
    pub position_length: usize,
}

impl<'a> From<&'a AnalyzedToken> for AnalyzedTokenRef<'a> {
    fn from(token: &'a AnalyzedToken) -> Self {
        Self {
            text: token.text.as_str(),
            position: token.position,
            offset_from: token.offset_from,
            offset_to: token.offset_to,
            position_length: token.position_length,
        }
    }
}

/// Allocation-free callback surface shared by scalar and future SIMD/CASS
/// analyzer families.
///
/// The token reference is valid only for the duration of the callback. An
/// implementation may reuse one scratch token for the complete stream. The
/// trait is sealed because Quill relies on every implementation preserving
/// the `u32` token-count and stable-support invariants while appending without
/// a staging allocation.
pub trait TokenAnalyzer: sealed::Sealed {
    /// Whether this family implements the requested schema pipeline.
    ///
    /// This answer must remain stable for the lifetime of the implementation.
    /// A family may support several kinds so one accumulator can serve mixed
    /// schemas such as [`crate::schema::CASS_SEMANTIC_SCHEMA`].
    fn supports(&self, analyzer: AnalyzerKind) -> bool;

    /// Analyze `text` in source order and call `sink` once per emitted token.
    ///
    /// Callers invoke this only for kinds accepted by [`Self::supports`].
    fn analyze(&mut self, analyzer: AnalyzerKind, text: &str, sink: &mut dyn FnMut(&AnalyzedToken));

    /// Analyze directly into a consumer that does not require owned token text.
    ///
    /// The default bridge preserves custom analyzer behavior. Analyzer
    /// families with a borrowed fast path override this method so unchanged
    /// input bytes can flow straight into the term interner.
    fn analyze_borrowed(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn for<'token> FnMut(AnalyzedTokenRef<'token>),
    ) {
        self.analyze(analyzer, text, &mut |token| sink(token.into()));
    }

    /// Retained analyzer scratch included in RSS/reuse diagnostics.
    fn bytes_reserved(&self) -> usize {
        0
    }

    /// Reset logical analyzer state while retaining reusable scratch.
    fn reset(&mut self) {}
}

pub(super) mod sealed {
    pub trait Sealed {}
}

/// Token-admission accounting returned by [`analyze_admitted`].
///
/// Query lowering needs the oversized count to distinguish genuinely empty
/// analysis from an unmatchable required/phrase clause. Document ingest uses
/// `admitted_tokens` for fieldnorms and statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnalysisReport {
    /// Tokens emitted by the analyzer before global length admission.
    pub raw_tokens: usize,
    /// Tokens whose normalized text was at most [`MAX_TERM_BYTES`].
    pub admitted_tokens: usize,
    /// Tokens rejected solely for exceeding [`MAX_TERM_BYTES`].
    pub oversized_tokens: usize,
}

/// A requested pipeline is absent from an injected analyzer family.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
#[error("analyzer family does not implement {analyzer:?}")]
pub struct UnsupportedAnalysis {
    /// Requested schema analyzer.
    pub analyzer: AnalyzerKind,
}

/// Analyze and apply Quill's global 65,530-byte term admission rule.
///
/// Filtering happens after analysis, so retained tokens keep their original
/// positions even when an oversized predecessor is dropped.
///
/// # Errors
///
/// Returns [`UnsupportedAnalysis`] when the family does not implement
/// `analyzer_kind`; the sink is not called in that case.
pub fn analyze_admitted<A: TokenAnalyzer + ?Sized>(
    analyzer: &mut A,
    analyzer_kind: AnalyzerKind,
    text: &str,
    sink: &mut dyn FnMut(&AnalyzedToken),
) -> Result<AnalysisReport, UnsupportedAnalysis> {
    if !analyzer.supports(analyzer_kind) {
        return Err(UnsupportedAnalysis {
            analyzer: analyzer_kind,
        });
    }
    let mut report = AnalysisReport::default();
    analyzer.analyze(analyzer_kind, text, &mut |token| {
        report.raw_tokens += 1;
        if token.text.len() > MAX_TERM_BYTES {
            report.oversized_tokens += 1;
            tracing::warn!(
                token_bytes = token.text.len(),
                max_token_bytes = MAX_TERM_BYTES,
                position = token.position,
                "Quill dropped an oversized analyzed token"
            );
            return;
        }
        report.admitted_tokens += 1;
        sink(token);
    });
    Ok(report)
}

/// Analyze directly into a borrowed consumer and apply the global term limit.
///
/// This is the ingest form of [`analyze_admitted`]. It preserves the same
/// report and position-gap semantics while allowing unchanged normalized text
/// to bypass analyzer-owned string construction.
pub(super) fn analyze_admitted_borrowed<A: TokenAnalyzer + ?Sized>(
    analyzer: &mut A,
    analyzer_kind: AnalyzerKind,
    text: &str,
    sink: &mut dyn for<'token> FnMut(AnalyzedTokenRef<'token>),
) -> Result<AnalysisReport, UnsupportedAnalysis> {
    if !analyzer.supports(analyzer_kind) {
        return Err(UnsupportedAnalysis {
            analyzer: analyzer_kind,
        });
    }
    let mut report = AnalysisReport::default();
    analyzer.analyze_borrowed(analyzer_kind, text, &mut |token| {
        report.raw_tokens += 1;
        if token.text.len() > MAX_TERM_BYTES {
            report.oversized_tokens += 1;
            tracing::warn!(
                token_bytes = token.text.len(),
                max_token_bytes = MAX_TERM_BYTES,
                position = token.position,
                "Quill dropped an oversized analyzed token"
            );
            return;
        }
        report.admitted_tokens += 1;
        sink(token);
    });
    Ok(report)
}
