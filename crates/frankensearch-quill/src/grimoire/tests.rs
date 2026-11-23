use std::borrow::Cow;
use std::error::Error;
use std::ops::Bound;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::schema::{Analyzer, FieldDescriptor, FieldKind, SchemaDescriptor};

use super::codec::{SliceReader, decode_entry, field_has_positions, vint_len, write_vint};
use super::cursor::contains_subslice;
use super::*;
type TestResult = Result<(), Box<dyn Error>>;

fn init_replay_tracing() {
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
}

const KEYWORD_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
    id: 0,
    name: "keyword",
    kind: FieldKind::Keyword,
    stored: false,
}];
const KEYWORD_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "grimoire-keyword-tests",
    fields: &KEYWORD_FIELDS,
};

const MIXED_FIELDS: [FieldDescriptor; 3] = [
    FieldDescriptor {
        id: 0,
        name: "keyword",
        kind: FieldKind::Keyword,
        stored: false,
    },
    FieldDescriptor {
        id: 1,
        name: "text_without_positions",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: false,
    },
    FieldDescriptor {
        id: 2,
        name: "text_with_positions",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: false,
    },
];
const MIXED_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "grimoire-mixed-tests",
    fields: &MIXED_FIELDS,
};

fn fixture_inputs(
    schema: SchemaDescriptor,
    keys: &[(u16, Vec<u8>)],
) -> (Vec<TermInput<'_>>, TermSectionLengths) {
    let positions_present = schema.fields.iter().any(|field| {
        matches!(
            field.kind,
            FieldKind::Text {
                positions: true,
                ..
            }
        )
    });
    let mut postings_offset = 0_u64;
    let mut positions_offset = 0_u64;
    let mut blockmax_offset = 0_u64;
    let mut inputs = Vec::with_capacity(keys.len());
    for (index, (field_ord, term)) in keys.iter().enumerate() {
        let postings = ByteSpan::new(postings_offset, 1);
        postings_offset += 1;
        let blockmax = ByteSpan::new(blockmax_offset, 1);
        blockmax_offset += 1;
        let doc_freq = u32::try_from(index % 7 + 1).expect("bounded fixture doc_freq");
        let metadata = if field_has_positions(schema, *field_ord)
            .expect("fixture uses a term-indexed field")
        {
            let positions = ByteSpan::new(positions_offset, 1);
            positions_offset += 1;
            TermMetadata::with_positions(doc_freq, postings, positions, blockmax)
        } else {
            TermMetadata::without_positions(doc_freq, postings, blockmax)
        };
        inputs.push(TermInput::new(*field_ord, term, metadata));
    }
    (
        inputs,
        TermSectionLengths {
            postings: postings_offset,
            positions: positions_present.then_some(positions_offset),
            blockmax: blockmax_offset,
        },
    )
}

fn encode_fixture(
    schema: SchemaDescriptor,
    keys: &[(u16, Vec<u8>)],
) -> Result<
    (
        EncodedTermDictionary,
        Vec<TermInput<'_>>,
        TermSectionLengths,
    ),
    TermDictionaryError,
> {
    let (inputs, sections) = fixture_inputs(schema, keys);
    let encoded = EncodedTermDictionary::encode_sorted(schema, sections, &inputs)?;
    Ok((encoded, inputs, sections))
}

fn terms_from_cursor(cursor: TermCursor<'_, '_>) -> Result<Vec<OwnedTerm>, TermDictionaryError> {
    cursor.collect_bounded(usize::MAX)
}

fn sorted_numbered_keys(count: usize) -> Vec<(u16, Vec<u8>)> {
    (0..count)
        .map(|index| (0, format!("term-{index:05}").into_bytes()))
        .collect()
}

fn expanded_glob_terms(
    dictionary: &TermDictionary<'_>,
    field_ord: u16,
    pattern: &[u8],
    limit: usize,
) -> Result<Vec<Vec<u8>>, TermDictionaryError> {
    Ok(dictionary
        .expand_glob(field_ord, pattern, limit)?
        .into_iter()
        .map(|row| row.term)
        .collect())
}

fn brute_force_glob_matches(pattern: &[u8], term: &[u8]) -> bool {
    let Some((&head, tail)) = pattern.split_first() else {
        return term.is_empty();
    };
    if head == b'*' {
        let mut remaining_pattern = tail;
        while remaining_pattern.first() == Some(&b'*') {
            remaining_pattern = &remaining_pattern[1..];
        }
        return (0..=term.len()).any(|matched_bytes| {
            brute_force_glob_matches(remaining_pattern, &term[matched_bytes..])
        });
    }
    let Some((&term_head, term_tail)) = term.split_first() else {
        return false;
    };
    head == term_head && brute_force_glob_matches(tail, term_tail)
}

fn next_glob_random(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

#[test]
fn empty_and_single_entry_have_golden_wire_bytes() -> TestResult {
    let empty_sections = TermSectionLengths {
        postings: 0,
        positions: None,
        blockmax: 0,
    };
    let empty = EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, empty_sections, &[])?;
    assert_eq!(empty.as_bytes(), &[0, 0, 0, 0]);
    assert_eq!(empty.term_count(), 0);
    assert_eq!(empty.block_count(), 0);
    assert!(
        empty
            .dictionary(KEYWORD_SCHEMA, empty_sections)?
            .cursor()?
            .current()
            .is_none()
    );

    let metadata = TermMetadata::without_positions(1, ByteSpan::new(0, 1), ByteSpan::new(0, 1));
    let input = [TermInput::new(0, b"a", metadata)];
    let sections = TermSectionLengths {
        postings: 1,
        positions: None,
        blockmax: 1,
    };
    let single = EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &input)?;
    assert_eq!(
        single.as_bytes(),
        &[
            1, 0, 0, 0, // block_count
            3, 0, 0, b'a', 0, // index key and blocks-area-relative offset
            1, 0, // entry_count
            3, 0, 0, b'a', // restart key
            1, 0, 1, 0, 1, // doc_freq and section spans
        ]
    );
    assert_eq!(
        single
            .dictionary(KEYWORD_SCHEMA, sections)?
            .lookup(0, b"a")?,
        Some(TermMatch {
            term_ord: 0,
            metadata,
        })
    );

    let mixed_inputs = [
        TermInput::new(
            1,
            b"a",
            TermMetadata::without_positions(1, ByteSpan::new(0, 1), ByteSpan::new(0, 1)),
        ),
        TermInput::new(
            2,
            b"a",
            TermMetadata::with_positions(
                1,
                ByteSpan::new(1, 1),
                ByteSpan::new(0, 1),
                ByteSpan::new(1, 1),
            ),
        ),
    ];
    let mixed_sections = TermSectionLengths {
        postings: 2,
        positions: Some(1),
        blockmax: 2,
    };
    let mixed = EncodedTermDictionary::encode_sorted(MIXED_SCHEMA, mixed_sections, &mixed_inputs)?;
    assert_eq!(
        mixed.as_bytes(),
        &[
            1, 0, 0, 0, // block_count
            3, 0, 1, b'a', 0, // index
            2, 0, // entry_count
            3, 0, 1, b'a', 1, 0, 1, 0, 1, // non-positional restart entry
            1, 2, 2, b'a', 1, 1, 1, 0, 1, 1, 1, // positional compressed entry
        ]
    );
    Ok(())
}

#[test]
fn round_trip_preserves_raw_terms_fields_ordinals_and_metadata() -> TestResult {
    let keys = vec![
        (0, Vec::new()),
        (0, vec![0]),
        (0, b"alpha".to_vec()),
        (0, vec![0xfe, 0xff]),
        (1, b"alpha".to_vec()),
        (1, b"omega".to_vec()),
        (2, b"alpha".to_vec()),
        (2, vec![0xff]),
    ];
    let (encoded, inputs, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
    let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;
    let rows = terms_from_cursor(dictionary.cursor()?)?;
    assert_eq!(rows.len(), keys.len());
    for (ordinal, row) in rows.iter().enumerate() {
        assert_eq!(row.term_ord, u32::try_from(ordinal)?);
        assert_eq!(row.field_ord, keys[ordinal].0);
        assert_eq!(row.term, keys[ordinal].1);
        assert_eq!(row.metadata, inputs[ordinal].metadata);
    }
    assert_eq!(
        encoded.as_bytes(),
        EncodedTermDictionary::encode_sorted(MIXED_SCHEMA, sections, &inputs,)?.as_bytes()
    );
    Ok(())
}

#[test]
fn validated_metadata_builds_borrowed_views_only_for_its_exact_byte_source() -> TestResult {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ValidatedTermDictionaryMetadata>();

    let keys = sorted_numbered_keys(33);
    let (encoded, inputs, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
    let metadata = TermDictionary::validate_metadata(encoded.as_bytes(), KEYWORD_SCHEMA, sections)?;

    let first = TermDictionary::from_validated_metadata(encoded.as_bytes(), &metadata)?;
    let second = TermDictionary::from_validated_metadata(encoded.as_bytes(), &metadata)?;
    assert!(matches!(&first.blocks, Cow::Borrowed(_)));
    assert!(matches!(&first.restarts, Cow::Borrowed(_)));
    assert_eq!(first.term_count(), 33);
    assert_eq!(first.block_count(), second.block_count());
    assert_eq!(first.restart_count(), second.restart_count());
    assert_eq!(
        first.lookup(0, b"term-00016")?,
        Some(TermMatch {
            term_ord: 16,
            metadata: inputs[16].metadata,
        })
    );
    assert_eq!(
        terms_from_cursor(second.cursor()?)?
            .into_iter()
            .map(|term| term.term)
            .collect::<Vec<_>>(),
        keys.into_iter().map(|(_, term)| term).collect::<Vec<_>>()
    );

    let copied_bytes = encoded.as_bytes().to_vec();
    assert!(matches!(
        TermDictionary::from_validated_metadata(&copied_bytes, &metadata),
        Err(TermDictionaryError::InvalidSchema { detail })
            if detail.contains("different byte source")
    ));
    Ok(())
}

#[test]
fn restart_boundaries_support_hits_and_neighbor_misses() -> TestResult {
    for (count, expected_restarts) in [(15, 1), (16, 1), (17, 2)] {
        let keys = sorted_numbered_keys(count);
        let (encoded, inputs, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(KEYWORD_SCHEMA, sections)?;
        assert_eq!(dictionary.block_count(), 1);
        assert_eq!(dictionary.restart_count(), expected_restarts);
        let mut scratch = TermScratch::new();
        for (ordinal, (_, term)) in keys.iter().enumerate() {
            assert_eq!(
                dictionary.lookup_with_scratch(0, term, &mut scratch)?,
                Some(TermMatch {
                    term_ord: u32::try_from(ordinal)?,
                    metadata: inputs[ordinal].metadata,
                })
            );
        }
        assert_eq!(dictionary.lookup(0, b"term-00000-")?, None);
        assert_eq!(dictionary.lookup(0, b"term-99999")?, None);
    }
    Ok(())
}

#[test]
fn greedy_blocks_and_oversized_singletons_reopen_canonically() -> TestResult {
    let keys = sorted_numbered_keys(900);
    let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
    let dictionary = encoded.dictionary(KEYWORD_SCHEMA, sections)?;
    assert!(dictionary.block_count() > 1);
    for block in dictionary.blocks.iter() {
        let byte_len = block.byte_range.end - block.byte_range.start;
        assert!(block.entry_count == 1 || byte_len <= TERM_BLOCK_TARGET_BYTES);
    }
    assert_eq!(terms_from_cursor(dictionary.cursor()?)?.len(), keys.len());

    let maximum_term = vec![b'x'; MAX_TERM_BYTES];
    let maximum_key = vec![(0, maximum_term.clone())];
    let (oversized, _, oversized_sections) = encode_fixture(KEYWORD_SCHEMA, &maximum_key)?;
    let oversized_dictionary = oversized.dictionary(KEYWORD_SCHEMA, oversized_sections)?;
    assert_eq!(oversized_dictionary.block_count(), 1);
    assert!(
        oversized_dictionary.blocks[0].byte_range.end
            - oversized_dictionary.blocks[0].byte_range.start
            > TERM_BLOCK_TARGET_BYTES
    );
    assert!(oversized_dictionary.lookup(0, &maximum_term)?.is_some());

    let too_long = vec![b'x'; MAX_TERM_BYTES + 1];
    let too_long_key = vec![(0, too_long)];
    let (inputs, sections) = fixture_inputs(KEYWORD_SCHEMA, &too_long_key);
    assert!(matches!(
        EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &inputs),
        Err(TermDictionaryError::TermTooLong { length, .. })
            if length == MAX_TERM_BYTES + 1
    ));
    Ok(())
}

#[test]
fn field_prefix_and_range_cursors_match_naive_filtering() -> TestResult {
    let keys = vec![
        (0, Vec::new()),
        (0, b"a".to_vec()),
        (0, b"aa".to_vec()),
        (0, b"ab".to_vec()),
        (0, b"b".to_vec()),
        (0, vec![0xff]),
        (0, vec![0xff, 0]),
        (1, b"a".to_vec()),
        (1, b"z".to_vec()),
        (2, b"a".to_vec()),
        (2, b"b".to_vec()),
    ];
    let (encoded, _, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
    let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;

    let field_zero = terms_from_cursor(dictionary.field_cursor(0)?)?;
    let expected_field_zero: Vec<&[u8]> = keys
        .iter()
        .filter(|(field, _)| *field == 0)
        .map(|(_, term)| term.as_slice())
        .collect();
    assert_eq!(
        field_zero
            .iter()
            .map(|row| row.term.as_slice())
            .collect::<Vec<_>>(),
        expected_field_zero
    );

    let a_prefix = terms_from_cursor(dictionary.prefix_cursor(0, b"a")?)?;
    assert_eq!(
        a_prefix
            .iter()
            .map(|row| row.term.as_slice())
            .collect::<Vec<_>>(),
        vec![b"a".as_slice(), b"aa".as_slice(), b"ab".as_slice()]
    );
    let high_prefix = terms_from_cursor(dictionary.prefix_cursor(0, &[0xff])?)?;
    assert_eq!(
        high_prefix
            .iter()
            .map(|row| row.term.as_slice())
            .collect::<Vec<_>>(),
        vec![&[0xff][..], &[0xff, 0][..]]
    );
    assert_eq!(
        terms_from_cursor(dictionary.prefix_cursor(0, b"")?)?.len(),
        expected_field_zero.len()
    );

    let range = terms_from_cursor(dictionary.range_cursor(
        0,
        Bound::Included(b"aa"),
        Bound::Excluded(&[0xff]),
    )?)?;
    assert_eq!(
        range
            .iter()
            .map(|row| row.term.as_slice())
            .collect::<Vec<_>>(),
        vec![b"aa".as_slice(), b"ab".as_slice(), b"b".as_slice()]
    );
    assert!(
        terms_from_cursor(dictionary.range_cursor(
            0,
            Bound::Included(b"a"),
            Bound::Excluded(b"a"),
        )?)?
        .is_empty()
    );
    assert!(matches!(
        dictionary.range_cursor(0, Bound::Included(b"z"), Bound::Included(b"a")),
        Err(TermDictionaryError::InvalidRange)
    ));
    Ok(())
}

#[test]
fn glob_expansion_covers_every_pattern_class_and_field_scope() -> TestResult {
    let mut keys = vec![
        (0, Vec::new()),
        (0, b"alpha".to_vec()),
        (0, b"alphabet".to_vec()),
        (0, b"beta".to_vec()),
        (0, b"delta".to_vec()),
        (0, b"omega".to_vec()),
        (0, b"pre-middle-suf".to_vec()),
        (0, b"prefix".to_vec()),
        (0, b"quest?mark".to_vec()),
        (0, "résumé".as_bytes().to_vec()),
        (0, "世界-end".as_bytes().to_vec()),
        (0, vec![0xff, 0]),
        (1, b"alpha".to_vec()),
        (1, b"omega".to_vec()),
        (1, b"prefix".to_vec()),
    ];
    keys.sort();
    let (encoded, _, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
    let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;
    let generous_limit = keys.len();

    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"", generous_limit)?,
        vec![Vec::<u8>::new()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"alpha", generous_limit)?,
        vec![b"alpha".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"alp*", generous_limit)?,
        vec![b"alpha".to_vec(), b"alphabet".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"alp***", generous_limit)?,
        vec![b"alpha".to_vec(), b"alphabet".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"*ta", generous_limit)?,
        vec![b"beta".to_vec(), b"delta".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"*middle*", generous_limit)?,
        vec![b"pre-middle-suf".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"p*e*suf", generous_limit)?,
        vec![b"pre-middle-suf".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"p**e***suf", generous_limit)?,
        vec![b"pre-middle-suf".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"quest?mark", generous_limit)?,
        vec![b"quest?mark".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"quest?*", generous_limit)?,
        vec![b"quest?mark".to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, "ré*".as_bytes(), generous_limit)?,
        vec!["résumé".as_bytes().to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, "*界*".as_bytes(), generous_limit)?,
        vec!["世界-end".as_bytes().to_vec()]
    );
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, &[b'*', 0xff, b'*'], generous_limit)?,
        vec![vec![0xff, 0]]
    );

    let expected_field_zero = keys
        .iter()
        .filter(|(field_ord, _)| *field_ord == 0)
        .map(|(_, term)| term.clone())
        .collect::<Vec<_>>();
    let all_once = expanded_glob_terms(&dictionary, 0, b"*", generous_limit)?;
    let all_twice = expanded_glob_terms(&dictionary, 0, b"**", generous_limit)?;
    assert_eq!(all_once, expected_field_zero);
    assert_eq!(all_twice, expected_field_zero);
    assert_eq!(
        expanded_glob_terms(&dictionary, 1, b"*", generous_limit)?,
        vec![b"alpha".to_vec(), b"omega".to_vec(), b"prefix".to_vec()]
    );
    assert!(expanded_glob_terms(&dictionary, 2, b"*", generous_limit)?.is_empty());
    assert!(contains_subslice(b"1234567needle", b"needle"));
    assert!(contains_subslice(b"12345678needle", b"needle"));
    assert!(!contains_subslice(b"12345678needlf", b"needle"));
    Ok(())
}

#[test]
fn glob_expansion_limit_fails_on_exactly_one_additional_match() -> TestResult {
    let keys = vec![
        (0, b"a".to_vec()),
        (0, b"aa".to_vec()),
        (0, b"ab".to_vec()),
        (0, b"b".to_vec()),
    ];
    let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
    let dictionary = encoded.dictionary(KEYWORD_SCHEMA, sections)?;

    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"a*", 3)?,
        vec![b"a".to_vec(), b"aa".to_vec(), b"ab".to_vec()]
    );
    assert!(matches!(
        dictionary.expand_glob(0, b"a*", 2),
        Err(TermDictionaryError::GlobExpansionLimitExceeded {
            pattern,
            field_ord: 0,
            limit: 2,
            actual: 3,
        }) if pattern == "a*"
    ));
    assert!(matches!(
        dictionary.expand_glob(0, b"*a*", 2),
        Err(TermDictionaryError::GlobExpansionLimitExceeded {
            pattern,
            field_ord: 0,
            limit: 2,
            actual: 3,
        }) if pattern == "*a*"
    ));
    assert!(matches!(
        dictionary.expand_glob(0, b"a", 0),
        Err(TermDictionaryError::GlobExpansionLimitExceeded {
            pattern,
            field_ord: 0,
            limit: 0,
            actual: 1,
        }) if pattern == "a"
    ));
    assert_eq!(
        expanded_glob_terms(&dictionary, 0, b"a", 1)?,
        vec![b"a".to_vec()]
    );
    assert!(expanded_glob_terms(&dictionary, 0, b"missing", 0)?.is_empty());
    Ok(())
}

#[test]
fn seeded_glob_expansion_matches_independent_brute_force_filter() -> TestResult {
    const BASE_SEED: u64 = 0x5eed_fade_cafe_babe;
    const TERM_ALPHABET: [u8; 8] = [b'a', b'b', b'c', b'-', b'?', 0, 0x80, 0xff];
    const PATTERN_ALPHABET: [u8; 9] = [b'a', b'b', b'c', b'-', b'?', 0, 0x80, 0xff, b'*'];

    init_replay_tracing();
    for seed in 0..16_u64 {
        let case_seed = BASE_SEED ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut state = case_seed;
        let mut keys = Vec::new();
        for field_ord in 0..=1_u16 {
            for _ in 0..96 {
                let term_len = usize::try_from(next_glob_random(&mut state) % 13)?;
                let mut term = Vec::with_capacity(term_len);
                for _ in 0..term_len {
                    let alphabet_len = u64::try_from(TERM_ALPHABET.len())?;
                    let index = usize::try_from(next_glob_random(&mut state) % alphabet_len)?;
                    term.push(TERM_ALPHABET[index]);
                }
                keys.push((field_ord, term));
            }
        }
        keys.sort();
        keys.dedup();

        let (encoded, _, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;
        for pattern_case in 0..48 {
            let pattern_len = usize::try_from(next_glob_random(&mut state) % 9)?;
            let mut pattern = Vec::with_capacity(pattern_len.saturating_add(2));
            for _ in 0..pattern_len {
                let alphabet_len = u64::try_from(PATTERN_ALPHABET.len())?;
                let index = usize::try_from(next_glob_random(&mut state) % alphabet_len)?;
                pattern.push(PATTERN_ALPHABET[index]);
            }
            if pattern_case % 7 == 0 {
                pattern.extend_from_slice(b"**");
            }

            for field_ord in 0..=1_u16 {
                let expected = keys
                    .iter()
                    .filter(|(candidate_field, term)| {
                        *candidate_field == field_ord && brute_force_glob_matches(&pattern, term)
                    })
                    .map(|(_, term)| term.clone())
                    .collect::<Vec<_>>();
                let actual = expanded_glob_terms(&dictionary, field_ord, &pattern, keys.len())?;
                assert_eq!(
                    actual, expected,
                    "seed={case_seed:#x} pattern_case={pattern_case} field={field_ord} pattern={pattern:?}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn fixed_seed_term_sets_match_naive_lookup_iteration_and_prefixes() -> TestResult {
    init_replay_tracing();
    for seed in 0..24_u64 {
        let case_seed = seed.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        tracing::info!(
            codec = "TERMDICT",
            seed = case_seed,
            case = seed,
            replay = "fixed_seed_term_sets_match_naive_lookup_iteration_and_prefixes",
            "starting randomized dictionary case"
        );
        let mut state = case_seed;
        let mut keys = Vec::new();
        keys.push((0, Vec::new()));
        keys.push((0, vec![u8::try_from(seed)?; 256]));
        for index in 0..192_usize {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let random = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            let field_ord = u16::try_from(index % MIXED_FIELDS.len())?;
            let mut term = format!("crate::module::{:02}::symbol::", index % 12).into_bytes();
            term.extend_from_slice(&random.to_be_bytes());
            keys.push((field_ord, term));
        }
        keys.sort_by(|(left_field, left_term), (right_field, right_term)| {
            left_field
                .cmp(right_field)
                .then_with(|| left_term.cmp(right_term))
        });
        keys.dedup();

        let (encoded, inputs, sections) = encode_fixture(MIXED_SCHEMA, &keys)?;
        let dictionary = encoded.dictionary(MIXED_SCHEMA, sections)?;
        let rows = terms_from_cursor(dictionary.cursor()?)?;
        assert_eq!(rows.len(), keys.len());
        for (ordinal, row) in rows.iter().enumerate() {
            assert_eq!(row.field_ord, keys[ordinal].0);
            assert_eq!(row.term, keys[ordinal].1);
            assert_eq!(row.metadata, inputs[ordinal].metadata);
            if ordinal.is_multiple_of(11) {
                assert_eq!(
                    dictionary.lookup(row.field_ord, &row.term)?,
                    Some(TermMatch {
                        term_ord: u32::try_from(ordinal)?,
                        metadata: row.metadata,
                    })
                );
            }
        }

        for field_ord in 0..u16::try_from(MIXED_FIELDS.len())? {
            let sample = keys
                .iter()
                .find(|(field, term)| *field == field_ord && term.len() >= 12)
                .expect("generated field has a long term");
            let prefix = &sample.1[..12];
            let expected: Vec<&[u8]> = keys
                .iter()
                .filter(|(field, term)| *field == field_ord && term.starts_with(prefix))
                .map(|(_, term)| term.as_slice())
                .collect();
            let actual = terms_from_cursor(dictionary.prefix_cursor(field_ord, prefix)?)?;
            assert_eq!(
                actual
                    .iter()
                    .map(|row| row.term.as_slice())
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
    Ok(())
}

#[test]
fn encoder_rejects_order_schema_and_reference_violations() {
    let sections = TermSectionLengths {
        postings: 2,
        positions: None,
        blockmax: 2,
    };
    let first = TermMetadata::without_positions(1, ByteSpan::new(0, 1), ByteSpan::new(0, 1));
    let second = TermMetadata::without_positions(1, ByteSpan::new(1, 1), ByteSpan::new(1, 1));
    let descending = [
        TermInput::new(0, b"b", first),
        TermInput::new(0, b"a", second),
    ];
    assert!(matches!(
        EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &descending),
        Err(TermDictionaryError::NonAscendingInput { index: 1 })
    ));
    let duplicate = [
        TermInput::new(0, b"a", first),
        TermInput::new(0, b"a", second),
    ];
    assert!(matches!(
        EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &duplicate),
        Err(TermDictionaryError::NonAscendingInput { index: 1 })
    ));

    let zero = [TermInput::new(
        0,
        b"a",
        TermMetadata::without_positions(0, ByteSpan::new(0, 1), ByteSpan::new(0, 1)),
    )];
    assert!(matches!(
        EncodedTermDictionary::encode_sorted(
            KEYWORD_SCHEMA,
            TermSectionLengths {
                postings: 1,
                positions: None,
                blockmax: 1,
            },
            &zero,
        ),
        Err(TermDictionaryError::ZeroDocFrequency { .. })
    ));

    let unknown = [TermInput::new(9, b"a", first)];
    assert!(matches!(
        EncodedTermDictionary::encode_sorted(
            KEYWORD_SCHEMA,
            TermSectionLengths {
                postings: 1,
                positions: None,
                blockmax: 1,
            },
            &unknown,
        ),
        Err(TermDictionaryError::UnknownField { field_ord: 9 })
    ));

    let gap = [TermInput::new(
        0,
        b"a",
        TermMetadata::without_positions(1, ByteSpan::new(1, 1), ByteSpan::new(0, 1)),
    )];
    assert!(matches!(
        EncodedTermDictionary::encode_sorted(KEYWORD_SCHEMA, sections, &gap),
        Err(TermDictionaryError::NonContiguousReference {
            section: "POSTINGS",
            ..
        })
    ));

    let positions_on_keyword = [TermInput::new(
        0,
        b"a",
        TermMetadata::with_positions(
            1,
            ByteSpan::new(0, 1),
            ByteSpan::new(0, 1),
            ByteSpan::new(0, 1),
        ),
    )];
    assert!(matches!(
        EncodedTermDictionary::encode_sorted(
            KEYWORD_SCHEMA,
            TermSectionLengths {
                postings: 1,
                positions: None,
                blockmax: 1,
            },
            &positions_on_keyword,
        ),
        Err(TermDictionaryError::PositionsPresenceMismatch { .. })
    ));
}

#[test]
fn section_presence_and_exact_consumption_are_enforced() {
    let empty = [0, 0, 0, 0];
    assert!(
        TermDictionary::parse(
            &empty,
            MIXED_SCHEMA,
            TermSectionLengths {
                postings: 0,
                positions: Some(0),
                blockmax: 0,
            },
        )
        .is_ok()
    );
    assert!(matches!(
        TermDictionary::parse(
            &empty,
            MIXED_SCHEMA,
            TermSectionLengths {
                postings: 0,
                positions: None,
                blockmax: 0,
            },
        ),
        Err(TermDictionaryError::PositionsSectionMismatch { .. })
    ));
    assert!(matches!(
        TermDictionary::parse(
            &empty,
            KEYWORD_SCHEMA,
            TermSectionLengths {
                postings: 1,
                positions: None,
                blockmax: 0,
            },
        ),
        Err(TermDictionaryError::SectionLengthMismatch {
            section: "POSTINGS",
            ..
        })
    ));
}

#[test]
fn parser_rejects_structural_mutations_and_every_truncation() -> TestResult {
    let keys = vec![(0, b"a".to_vec())];
    let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
    let bytes = encoded.as_bytes();
    for end in 0..bytes.len() {
        assert!(TermDictionary::parse(&bytes[..end], KEYWORD_SCHEMA, sections).is_err());
    }

    let mut index_mismatch = bytes.to_vec();
    index_mismatch[7] = b'b';
    assert!(matches!(
        TermDictionary::parse(&index_mismatch, KEYWORD_SCHEMA, sections),
        Err(TermDictionaryError::IndexKeyMismatch { block_index: 0 })
    ));

    let mut bad_offset = bytes.to_vec();
    bad_offset[8] = 1;
    assert!(matches!(
        TermDictionary::parse(&bad_offset, KEYWORD_SCHEMA, sections),
        Err(TermDictionaryError::InvalidBlockOffset { block_index: 0, .. })
    ));

    let mut empty_block = bytes.to_vec();
    empty_block[9..11].copy_from_slice(&0_u16.to_le_bytes());
    assert!(matches!(
        TermDictionary::parse(&empty_block, KEYWORD_SCHEMA, sections),
        Err(TermDictionaryError::EmptyBlock { block_index: 0 })
    ));

    let mut trailing = bytes.to_vec();
    trailing.push(0);
    assert!(matches!(
        TermDictionary::parse(&trailing, KEYWORD_SCHEMA, sections),
        Err(TermDictionaryError::TrailingBlockBytes { block_index: 0, .. })
    ));

    let mut bad_key = vec![0, 0, b'a'];
    let mut invalid_prefix_reader = SliceReader::new(&[4, 1, b'b']);
    assert!(matches!(
        decode_entry(
            &mut invalid_prefix_reader,
            1,
            &mut bad_key,
            KEYWORD_SCHEMA,
            sections,
            1,
        ),
        Err(TermDictionaryError::InvalidPrefix { .. })
    ));
    let mut noncanonical_key = vec![0, 0, b'a'];
    let mut noncanonical_reader = SliceReader::new(&[1, 2, 0, b'b']);
    assert!(matches!(
        decode_entry(
            &mut noncanonical_reader,
            1,
            &mut noncanonical_key,
            KEYWORD_SCHEMA,
            sections,
            1,
        ),
        Err(TermDictionaryError::NonCanonicalPrefix { .. })
    ));
    Ok(())
}

#[test]
fn parser_rejects_split_order_size_vint_and_reference_corruption() -> TestResult {
    fn singleton_block(term: u8, postings_offset: u64, blockmax_offset: u64) -> Vec<u8> {
        let mut block = 1_u16.to_le_bytes().to_vec();
        write_vint(3, &mut block);
        block.extend_from_slice(&[0, 0, term]);
        for value in [1, postings_offset, 1, blockmax_offset, 1] {
            write_vint(value, &mut block);
        }
        block
    }

    fn indexed_dictionary(first_keys: &[(u8, u64)], blocks: &[Vec<u8>]) -> Vec<u8> {
        let mut bytes = u32::try_from(first_keys.len())
            .expect("bounded fixture block count")
            .to_le_bytes()
            .to_vec();
        for (term, relative_offset) in first_keys {
            write_vint(3, &mut bytes);
            bytes.extend_from_slice(&[0, 0, *term]);
            write_vint(*relative_offset, &mut bytes);
        }
        for block in blocks {
            bytes.extend_from_slice(block);
        }
        bytes
    }

    let first = singleton_block(b'a', 0, 0);
    let second = singleton_block(b'b', 1, 1);
    let premature = indexed_dictionary(
        &[(b'a', 0), (b'b', u64::try_from(first.len())?)],
        &[first.clone(), second],
    );
    let two_sections = TermSectionLengths {
        postings: 2,
        positions: None,
        blockmax: 2,
    };
    assert!(matches!(
        TermDictionary::parse(&premature, KEYWORD_SCHEMA, two_sections),
        Err(TermDictionaryError::PrematureBlockSplit { block_index: 0 })
    ));

    let mut descending_block = 2_u16.to_le_bytes().to_vec();
    write_vint(3, &mut descending_block);
    descending_block.extend_from_slice(&[0, 0, b'a']);
    for value in [1, 0, 1, 0, 1] {
        write_vint(value, &mut descending_block);
    }
    write_vint(2, &mut descending_block);
    write_vint(0, &mut descending_block);
    for value in [1, 1, 1, 1, 1] {
        write_vint(value, &mut descending_block);
    }
    let descending = indexed_dictionary(&[(b'a', 0)], &[descending_block]);
    assert!(matches!(
        TermDictionary::parse(&descending, KEYWORD_SCHEMA, two_sections),
        Err(TermDictionaryError::NonAscendingKey { term_ordinal: 1 })
    ));

    let mut oversized_block = 2_u16.to_le_bytes().to_vec();
    oversized_block.resize(TERM_BLOCK_TARGET_BYTES + 1, 0);
    let oversized = indexed_dictionary(&[(b'a', 0)], &[oversized_block]);
    assert!(matches!(
        TermDictionary::parse(&oversized, KEYWORD_SCHEMA, two_sections),
        Err(TermDictionaryError::OversizedBlock {
            block_index: 0,
            entry_count: 2,
            ..
        })
    ));

    let valid = indexed_dictionary(&[(b'a', 0)], &[first]);
    let one_section = TermSectionLengths {
        postings: 1,
        positions: None,
        blockmax: 1,
    };
    let mut out_of_bounds = valid.clone();
    out_of_bounds[16] = 2;
    assert!(matches!(
        TermDictionary::parse(&out_of_bounds, KEYWORD_SCHEMA, one_section),
        Err(TermDictionaryError::ReferenceOutOfBounds {
            section: "POSTINGS",
            ..
        })
    ));
    assert!(matches!(
        TermDictionary::parse(
            &valid,
            KEYWORD_SCHEMA,
            TermSectionLengths {
                postings: 2,
                ..one_section
            },
        ),
        Err(TermDictionaryError::SectionLengthMismatch {
            section: "POSTINGS",
            ..
        })
    ));

    let mut overflow_block = 1_u16.to_le_bytes().to_vec();
    write_vint(3, &mut overflow_block);
    overflow_block.extend_from_slice(&[0, 0, b'a']);
    for value in [1, u64::MAX, 1, 0, 1] {
        write_vint(value, &mut overflow_block);
    }
    let overflow = indexed_dictionary(&[(b'a', 0)], &[overflow_block]);
    assert!(matches!(
        TermDictionary::parse(
            &overflow,
            KEYWORD_SCHEMA,
            TermSectionLengths {
                postings: u64::MAX,
                positions: None,
                blockmax: 1,
            },
        ),
        Err(TermDictionaryError::ReferenceOverflow {
            section: "POSTINGS",
            ..
        })
    ));

    let mut noncanonical_vint = valid;
    noncanonical_vint.splice(8..9, [0x80, 0]);
    assert!(matches!(
        TermDictionary::parse(&noncanonical_vint, KEYWORD_SCHEMA, one_section),
        Err(TermDictionaryError::NonCanonicalVint { offset: 8 })
    ));

    let hostile_header = 2_u32.to_le_bytes();
    assert!(matches!(
        TermDictionary::parse_with_limits(
            &hostile_header,
            KEYWORD_SCHEMA,
            TermSectionLengths {
                postings: 0,
                positions: None,
                blockmax: 0,
            },
            TermDictionaryLimits {
                max_bytes: hostile_header.len(),
                max_blocks: 2,
                max_terms: 1,
                max_restarts: 2,
            },
        ),
        Err(TermDictionaryError::TermBudgetExceeded {
            limit: 1,
            actual: 2,
        })
    ));
    Ok(())
}

#[test]
fn vint_boundaries_are_canonical_and_overflow_safe() -> TestResult {
    for value in [0, 127, 128, 16_383, 16_384, u64::from(u32::MAX), u64::MAX] {
        let mut bytes = Vec::new();
        write_vint(value, &mut bytes);
        assert_eq!(bytes.len(), vint_len(value));
        let mut reader = SliceReader::new(&bytes);
        assert_eq!(reader.read_vint()?, value);
        assert_eq!(reader.position(), bytes.len());
    }
    assert!(matches!(
        SliceReader::new(&[0x80, 0]).read_vint(),
        Err(TermDictionaryError::NonCanonicalVint { offset: 0 })
    ));
    assert!(matches!(
        SliceReader::new(&[0xff; 10]).read_vint(),
        Err(TermDictionaryError::VintOverflow { offset: 0 })
    ));
    assert!(matches!(
        SliceReader::new(&[0x80]).read_vint(),
        Err(TermDictionaryError::Truncated { .. })
    ));
    Ok(())
}

#[test]
fn explicit_reader_and_writer_budgets_fail_closed() -> TestResult {
    let keys = sorted_numbered_keys(17);
    let (encoded, inputs, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
    let bytes = encoded.as_bytes();
    let base = TermDictionaryLimits {
        max_bytes: bytes.len(),
        max_blocks: usize::try_from(encoded.block_count())?,
        max_terms: keys.len(),
        max_restarts: 2,
    };
    assert!(TermDictionary::parse_with_limits(bytes, KEYWORD_SCHEMA, sections, base).is_ok());
    assert!(matches!(
        TermDictionary::parse_with_limits(
            bytes,
            KEYWORD_SCHEMA,
            sections,
            TermDictionaryLimits {
                max_bytes: bytes.len() - 1,
                ..base
            },
        ),
        Err(TermDictionaryError::ByteBudgetExceeded { .. })
    ));
    assert!(matches!(
        TermDictionary::parse_with_limits(
            bytes,
            KEYWORD_SCHEMA,
            sections,
            TermDictionaryLimits {
                max_blocks: 0,
                ..base
            },
        ),
        Err(TermDictionaryError::BlockBudgetExceeded { .. })
    ));
    assert!(matches!(
        TermDictionary::parse_with_limits(
            bytes,
            KEYWORD_SCHEMA,
            sections,
            TermDictionaryLimits {
                max_terms: 16,
                ..base
            },
        ),
        Err(TermDictionaryError::TermBudgetExceeded { .. })
    ));
    assert!(matches!(
        TermDictionary::parse_with_limits(
            bytes,
            KEYWORD_SCHEMA,
            sections,
            TermDictionaryLimits {
                max_restarts: 1,
                ..base
            },
        ),
        Err(TermDictionaryError::RestartBudgetExceeded { .. })
    ));
    assert!(matches!(
        EncodedTermDictionary::encode_sorted_with_limits(
            KEYWORD_SCHEMA,
            sections,
            &inputs,
            TermDictionaryLimits {
                max_restarts: 1,
                ..TermDictionaryLimits::default()
            },
        ),
        Err(TermDictionaryError::RestartBudgetExceeded {
            limit: 1,
            actual: 2,
        })
    ));
    assert!(matches!(
        EncodedTermDictionary::encode_sorted_with_limits(
            KEYWORD_SCHEMA,
            sections,
            &inputs,
            TermDictionaryLimits {
                max_bytes: bytes.len() - 1,
                ..TermDictionaryLimits::default()
            },
        ),
        Err(TermDictionaryError::ByteBudgetExceeded { .. })
    ));
    assert!(matches!(
        EncodedTermDictionary::encode_sorted_with_limits(
            KEYWORD_SCHEMA,
            sections,
            &inputs,
            TermDictionaryLimits {
                max_blocks: 0,
                ..TermDictionaryLimits::default()
            },
        ),
        Err(TermDictionaryError::BlockBudgetExceeded {
            limit: 0,
            actual: 1,
        })
    ));
    assert!(matches!(
        EncodedTermDictionary::encode_sorted_with_limits(
            KEYWORD_SCHEMA,
            sections,
            &inputs,
            TermDictionaryLimits {
                max_terms: 16,
                ..TermDictionaryLimits::default()
            },
        ),
        Err(TermDictionaryError::TermBudgetExceeded {
            limit: 16,
            actual: 17,
        })
    ));
    Ok(())
}

#[test]
fn bounded_materialization_and_cursor_fusion_are_explicit() -> TestResult {
    let keys = sorted_numbered_keys(3);
    let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &keys)?;
    let dictionary = encoded.dictionary(KEYWORD_SCHEMA, sections)?;
    assert!(matches!(
        dictionary.cursor()?.collect_bounded(2),
        Err(TermDictionaryError::MaterializationLimitExceeded { limit: 2 })
    ));
    let mut cursor = dictionary.cursor()?;
    assert!(cursor.current().is_some());
    assert!(cursor.next()?.is_some());
    assert!(cursor.next()?.is_some());
    assert!(cursor.next()?.is_none());
    assert!(cursor.next()?.is_none());
    Ok(())
}

#[test]
fn arbitrary_and_unaligned_bytes_never_panic() -> TestResult {
    const BASE_SEED: u64 = 0x4d59_5df4_d0f3_3173;

    init_replay_tracing();
    let valid_keys = sorted_numbered_keys(17);
    let (encoded, _, sections) = encode_fixture(KEYWORD_SCHEMA, &valid_keys)?;
    for shift in 0..32 {
        let mut padded = vec![0xa5; shift];
        padded.extend_from_slice(encoded.as_bytes());
        let dictionary = TermDictionary::parse(&padded[shift..], KEYWORD_SCHEMA, sections)?;
        assert_eq!(dictionary.term_count(), 17);
    }

    let mut state = BASE_SEED;
    for length in 0..=512 {
        tracing::info!(
            codec = "TERMDICT",
            seed = BASE_SEED,
            case = length,
            replay = "arbitrary_and_unaligned_bytes_never_panic",
            "starting hostile-byte case"
        );
        let mut bytes = Vec::with_capacity(length);
        for _ in 0..length {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            bytes.push(state.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes()[0]);
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = TermDictionary::parse(
                &bytes,
                MIXED_SCHEMA,
                TermSectionLengths {
                    postings: 64,
                    positions: Some(64),
                    blockmax: 64,
                },
            );
        }));
        assert!(
            result.is_ok(),
            "seed={BASE_SEED:#x} parser panicked for {length} hostile bytes"
        );
    }
    Ok(())
}
