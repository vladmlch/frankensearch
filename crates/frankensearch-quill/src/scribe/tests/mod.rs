#![allow(clippy::wildcard_imports)]

mod accumulator_tests;
mod flush_tests;
mod interner_tests;
mod lease_tests;
mod tokenizer_tests;

pub(super) use super::*;

use std::hash::{BuildHasher, BuildHasherDefault, Hasher};

use frankensearch_core::DocId;

use crate::contract::fieldnorm_to_id;
use crate::delta::{DeltaFieldNorm, DeltaSegment, DeltaSnapshot, DeltaTermPosting};
use crate::schema::{Analyzer as AnalyzerKind, FieldDescriptor, FieldKind, SchemaDescriptor};

#[cfg(feature = "tantivy-oracle")]
pub(super) const LANGUAGE_CONTRACT_FIXTURE: &str =
    include_str!("../../../../../tests/fixtures/quill_language_contract.json");
#[cfg(feature = "tantivy-oracle")]
pub(super) const SHARED_CORPUS_FIXTURE: &str =
    include_str!("../../../../../tests/fixtures/corpus.json");

/// Degenerate hasher: every key hashes to 0, forcing every intern through
/// the `Many` collision-verification path.
#[derive(Default)]
pub(super) struct ConstHasher;
impl Hasher for ConstHasher {
    fn finish(&self) -> u64 {
        0
    }
    fn write(&mut self, _bytes: &[u8]) {}
}
pub(super) type ConstBuild = BuildHasherDefault<ConstHasher>;

pub(super) fn rescanned_term_interner_bytes_reserved<S: BuildHasher>(
    interner: &TermInterner<S>,
) -> usize {
    let collision_ids = interner
        .buckets
        .values()
        .map(|bucket| match bucket {
            Bucket::One(_) => 0,
            Bucket::Many(entries) => entries.capacity() * std::mem::size_of::<BucketEntry>(),
        })
        .sum::<usize>();
    interner
        .arena
        .bytes_reserved()
        .saturating_add(
            interner
                .spans
                .capacity()
                .saturating_mul(std::mem::size_of::<ArenaSpan>()),
        )
        .saturating_add(
            interner
                .buckets
                .capacity()
                .saturating_mul(TERM_BUCKET_BYTES_ESTIMATE),
        )
        .saturating_add(interner.key_scratch.capacity())
        .saturating_add(collision_ids)
}

const MIXED_POSITION_FIELDS: [FieldDescriptor; 4] = [
    FieldDescriptor {
        id: 0,
        name: "key",
        kind: FieldKind::Keyword,
        stored: false,
    },
    FieldDescriptor {
        id: 1,
        name: "with_positions",
        kind: FieldKind::Text {
            analyzer: AnalyzerKind::FrankensearchDefault,
            positions: true,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 2,
        name: "without_positions",
        kind: FieldKind::Text {
            analyzer: AnalyzerKind::FrankensearchDefault,
            positions: false,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 3,
        name: "stored",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
];
pub(super) const MIXED_POSITION_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "scribe-mixed-positions-v1",
    fields: &MIXED_POSITION_FIELDS,
};

const DELTA_STREAMING_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "term",
    kind: FieldKind::Keyword,
    stored: false,
}];
const DELTA_STREAMING_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "scribe-delta-streaming-v1",
    fields: &DELTA_STREAMING_FIELDS,
};

pub(super) fn delta_streaming_snapshot() -> DeltaSnapshot {
    let mut delta = DeltaSegment::new(DELTA_STREAMING_SCHEMA, 0, usize::MAX)
        .expect("valid Delta streaming fixture");
    let first_norms = [DeltaFieldNorm {
        field_ord: 0,
        raw_length: 2,
        fieldnorm_id: fieldnorm_to_id(2),
    }];
    let first_postings = [
        DeltaTermPosting {
            field_ord: 0,
            term: b"zeta",
            frequency: 1,
            positions: None,
        },
        DeltaTermPosting {
            field_ord: 0,
            term: b"alpha",
            frequency: 1,
            positions: None,
        },
    ];
    delta
        .apply_document_with_values(
            0,
            DocId::from("delta-first"),
            0x11,
            &first_norms,
            &first_postings,
            &[],
            &[],
        )
        .expect("first Delta fixture document");

    let second_norms = [DeltaFieldNorm {
        field_ord: 0,
        raw_length: 1,
        fieldnorm_id: fieldnorm_to_id(1),
    }];
    let second_postings = [DeltaTermPosting {
        field_ord: 0,
        term: b"beta",
        frequency: 1,
        positions: None,
    }];
    delta
        .apply_document_with_values(
            1,
            DocId::from("delta-second"),
            0x22,
            &second_norms,
            &second_postings,
            &[],
            &[],
        )
        .expect("second Delta fixture document");
    delta.freeze(7)
}

const UNSUPPORTED_ANALYZER_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "cass_text",
    kind: FieldKind::Text {
        analyzer: AnalyzerKind::CassHyphenNormalize,
        positions: true,
    },
    stored: false,
}];
pub(super) const UNSUPPORTED_ANALYZER_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "scribe-unsupported-analyzer-v1",
    fields: &UNSUPPORTED_ANALYZER_FIELDS,
};

const INDEXED_NUMERIC_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "sequence",
    kind: FieldKind::U64 {
        indexed: true,
        fast: false,
    },
    stored: true,
}];
pub(super) const INDEXED_NUMERIC_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "scribe-indexed-numeric-v1",
    fields: &INDEXED_NUMERIC_FIELDS,
};

const FAST_ONLY_NUMERIC_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "fast_sequence",
    kind: FieldKind::U64 {
        indexed: false,
        fast: true,
    },
    stored: false,
}];
pub(super) const FAST_ONLY_NUMERIC_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "scribe-fast-only-numeric-v1",
    fields: &FAST_ONLY_NUMERIC_FIELDS,
};

const SIGNED_UNSIGNED_NUMERIC_FIELDS: [FieldDescriptor; 2] = [
    FieldDescriptor {
        id: 0,
        name: "created_at",
        kind: FieldKind::I64 {
            indexed: true,
            fast: false,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 1,
        name: "sequence",
        kind: FieldKind::U64 {
            indexed: true,
            fast: false,
        },
        stored: false,
    },
];
pub(super) const SIGNED_UNSIGNED_NUMERIC_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "scribe-signed-unsigned-numeric-v1",
    fields: &SIGNED_UNSIGNED_NUMERIC_FIELDS,
};

#[derive(Debug, Default)]
pub(super) struct SamePositionAnalyzer;

impl sealed::Sealed for SamePositionAnalyzer {}

impl TokenAnalyzer for SamePositionAnalyzer {
    // Covers both Frankensearch analyzers so it can stand in for the
    // shipping schema whichever of them its text fields select.
    fn supports(&self, analyzer: AnalyzerKind) -> bool {
        matches!(
            analyzer,
            AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams
        )
    }

    fn analyze(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        assert!(self.supports(analyzer), "{analyzer:?}");
        for token_text in ["compound", "part"] {
            sink(&AnalyzedToken {
                text: token_text.to_owned(),
                position: 0,
                offset_from: 0,
                offset_to: text.len(),
                position_length: 1,
            });
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct MixedCassAnalyzer;

impl sealed::Sealed for MixedCassAnalyzer {}

impl TokenAnalyzer for MixedCassAnalyzer {
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
        let (token_text, position) = match analyzer {
            AnalyzerKind::CassHyphenNormalize => ("hyphen-dispatch", 7),
            AnalyzerKind::CassPrefixNormalize => ("prefix-dispatch", 11),
            AnalyzerKind::FrankensearchDefault | AnalyzerKind::FrankensearchCjkBigrams => {
                unreachable!("fixture does not advertise the default analyzers")
            }
        };
        sink(&AnalyzedToken {
            text: token_text.to_owned(),
            position,
            offset_from: 0,
            offset_to: text.len(),
            position_length: 1,
        });
    }
}

#[derive(Debug)]
pub(super) struct DeterministicRng(pub(super) u64);

impl DeterministicRng {
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(super) fn choose(&mut self, upper: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(upper).expect("small choice bound"))
            .expect("bounded random choice fits usize")
    }
}

pub(super) fn randomized_text(rng: &mut DeterministicRng) -> String {
    const WORDS: [&str; 8] = [
        "Alpha", "BETA", "Rust2024", "ÉCLAIR", "東京", "needle", "tail", "z9",
    ];
    const SEPARATORS: [&str; 5] = [" ", "-", "/", "...", "_"];
    let count = rng.choose(12);
    let mut text = String::new();
    for index in 0..count {
        if index != 0 {
            text.push_str(SEPARATORS[rng.choose(SEPARATORS.len())]);
        }
        text.push_str(WORDS[rng.choose(WORDS.len())]);
    }
    text
}

pub(super) fn phrase_matches(
    field: &FieldTokenColumns,
    doc_ord: u32,
    phrase_term_ids: &[u32],
) -> bool {
    let Some(positions) = field.positions() else {
        return false;
    };
    let Some(first_term) = phrase_term_ids.first() else {
        return false;
    };
    field
        .term_ids()
        .iter()
        .zip(field.doc_ords())
        .zip(positions)
        .filter(|((term_id, row_doc), _)| {
            u32::eq(*term_id, first_term) && u32::eq(*row_doc, &doc_ord)
        })
        .any(|(_, start_position)| {
            phrase_term_ids
                .iter()
                .enumerate()
                .skip(1)
                .all(|(offset, term_id)| {
                    let expected_position = start_position.saturating_add(
                        u32::try_from(offset).expect("test phrase length fits u32"),
                    );
                    field
                        .term_ids()
                        .iter()
                        .zip(field.doc_ords())
                        .zip(positions)
                        .any(|((row_term, row_doc), row_position)| {
                            u32::eq(row_term, term_id)
                                && u32::eq(row_doc, &doc_ord)
                                && u32::eq(row_position, &expected_position)
                        })
                })
        })
}

pub(super) fn analyzed_tokens(text: &str) -> Vec<AnalyzedToken> {
    let mut analyzer = FrankensearchTokenizer::default();
    let mut tokens = Vec::new();
    analyzer.analyze(AnalyzerKind::FrankensearchDefault, text, &mut |token| {
        tokens.push(token.clone());
    });
    tokens
}
