use std::path::Path;

use frankensearch_core::error::SearchError;
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::Embedder;

use crate::simd::{Model2VecAccumulationRoute, last_model2vec_accumulation_route_for_test};

use super::super::{
    DEFAULT_MODEL_NAME, MODEL2VEC_PREPROCESSING_V1, MODEL2VEC_PREPROCESSING_V2, Model2VecEmbedder,
    PARALLEL_BATCH_MIN, finish_mean_pool_and_normalize, finish_mean_pool_and_normalize_former,
};
use super::{
    assert_f32_bits_eq, assert_fast_ids_and_former_vector_bits, create_test_model,
    create_tokenizer_parity_model, create_wordpiece_test_model, former_embed_sync,
    former_embed_token_ids, former_token_ids,
};

/// GH #50: the V2 contract drops the `WordPiece` unknown token before
/// pooling, as reference `model2vec` does, so an unknown word leaves the
/// vector bit-identical; V1 pools its row, so the vector moves. V2 on a
/// tokenizer with no `WordPiece` unknown token is refused at load.
#[test]
fn v2_preprocessing_drops_wordpiece_unknown_tokens_and_v1_keeps_them() {
    let dir = tempfile::tempdir().unwrap();
    create_wordpiece_test_model(dir.path(), 11, 8);
    let load = |preprocessing: &str| {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("wordpiece", 1);
        preprocessing.clone_into(&mut identity.space.model_preprocessing);
        Model2VecEmbedder::load_preverified(dir.path(), "wordpiece", identity)
    };

    let v2 = load(MODEL2VEC_PREPROCESSING_V2).expect("V2 WordPiece load");
    assert_eq!(v2.dropped_token_id, Some(0));
    let known = v2.embed_sync("hello world").unwrap();
    assert!(known.iter().any(|value| *value != 0.0));
    assert_f32_bits_eq(
        &v2.embed_sync("hello zzzz world").unwrap(),
        &known,
        "V2 must pool exactly the known tokens",
    );
    assert!(
        v2.embed_sync("zzzz qqqq")
            .unwrap()
            .iter()
            .all(|value| *value == 0.0),
        "an all-unknown text is the zero vector under V2"
    );

    let v1 = load(MODEL2VEC_PREPROCESSING_V1).expect("V1 WordPiece load");
    assert_eq!(v1.dropped_token_id, None);
    assert_ne!(
        v1.embed_sync("hello zzzz world").unwrap(),
        v1.embed_sync("hello world").unwrap(),
        "V1 pools the unknown token's row"
    );

    let word_level = tempfile::tempdir().unwrap();
    create_test_model(word_level.path(), 11, 8);
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("word-level", 1);
    MODEL2VEC_PREPROCESSING_V2.clone_into(&mut identity.space.model_preprocessing);
    assert!(
        matches!(
            Model2VecEmbedder::load_preverified(word_level.path(), "word-level", identity),
            Err(SearchError::ModelLoadFailed { .. })
        ),
        "V2 requires a WordPiece tokenizer"
    );
}

#[test]
fn embed_batch_sync_matches_serial_across_parallel_boundary() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);
    let embedder = Model2VecEmbedder::load_with_name(dir.path(), "test-model").unwrap();

    // Straddle PARALLEL_BATCH_MIN so both the serial and Rayon paths are exercised.
    for &batch_size in &[0_usize, 1, PARALLEL_BATCH_MIN - 1, PARALLEL_BATCH_MIN, 17] {
        let docs: Vec<String> = (0..batch_size)
            .map(|i| format!("hello world test rust search {i}"))
            .collect();
        let texts: Vec<&str> = docs.iter().map(String::as_str).collect();

        let serial: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| former_embed_sync(&embedder, t))
            .collect();
        let batched = embedder.embed_batch_sync(&texts).unwrap();

        assert_eq!(batched.len(), serial.len(), "len at n={batch_size}");
        for (index, (batched, former)) in batched.iter().zip(&serial).enumerate() {
            assert_f32_bits_eq(
                batched,
                former,
                &format!("batch order or output diverged at n={batch_size}, index={index}"),
            );
        }
    }
}

#[test]
fn embed_sync_observes_offset_free_tokenizer_route() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);
    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();

    let former = embedder.tokenizer.encode("hello world", false).unwrap();
    assert!(
        former.get_tokens().iter().any(|token| !token.is_empty()),
        "the former encode route must materialize token text for this planted route witness"
    );

    embedder.embed_sync("hello world").unwrap();
    assert!(
        embedder.last_tokenizer_route_was_offset_free(),
        "shipping Model2Vec must retain the encode_fast offset-free tokenizer route"
    );
}

#[test]
fn encode_fast_token_ids_and_vectors_match_former_oracle() {
    let dir = tempfile::tempdir().unwrap();
    create_tokenizer_parity_model(dir.path());
    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();

    let mut inputs = crate::model_manifest::MODEL_CONFORMANCE_TEXTS_V1
        .iter()
        .map(|text| (*text).to_owned())
        .collect::<Vec<_>>();
    inputs.extend([
        String::new(),
        "HELLO world".to_owned(),
        "CAFÉ cafe\u{301}".to_owned(),
        "hello 東京 world".to_owned(),
        "definitely-oov-token".to_owned(),
        "hello <ADDED> [SPECIAL] world".to_owned(),
    ]);

    for tokens in [511_usize, 512, 513] {
        inputs.push(
            std::iter::repeat_n("hello", tokens)
                .collect::<Vec<_>>()
                .join(" "),
        );
    }

    for input in &inputs {
        assert_fast_ids_and_former_vector_bits(&embedder, input, input);
    }

    let special_ids = former_token_ids(&embedder, "hello <ADDED> [SPECIAL] world");
    assert!(
        special_ids.contains(&12) && special_ids.contains(&13),
        "fixture must exercise both the normalized added token and the added special token"
    );
    for tokens in [511_usize, 512, 513] {
        let input = std::iter::repeat_n("hello", tokens)
            .collect::<Vec<_>>()
            .join(" ");
        let expected_len = tokens.min(512);
        let former_ids = former_token_ids(&embedder, &input);
        let fast_ids = embedder
            .tokenizer
            .encode_fast(input.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(
            former_ids.len(),
            expected_len,
            "former tokenizer truncation at {tokens} input tokens"
        );
        assert_eq!(
            fast_ids.len(),
            expected_len,
            "fast tokenizer truncation at {tokens} input tokens"
        );
    }

    let texts = inputs.iter().map(String::as_str).collect::<Vec<_>>();
    let former = texts
        .iter()
        .map(|text| former_embed_sync(&embedder, text))
        .collect::<Vec<_>>();
    let batched = embedder.embed_batch_sync(&texts).unwrap();
    assert_eq!(batched.len(), former.len());
    for (index, (actual, expected)) in batched.iter().zip(&former).enumerate() {
        assert_f32_bits_eq(
            actual,
            expected,
            &format!("batch order or vector bits changed at index={index}"),
        );
    }
}

fn expected_native_256_route(token_count: usize) -> Model2VecAccumulationRoute {
    #[cfg(target_arch = "x86_64")]
    {
        if token_count < 512 {
            if std::is_x86_feature_detected!("avx2") {
                Model2VecAccumulationRoute::Native256ShortAvx2
            } else {
                Model2VecAccumulationRoute::Base
            }
        } else {
            Model2VecAccumulationRoute::Prefetched
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = token_count;
        Model2VecAccumulationRoute::Base
    }
}

#[test]
fn native_256_embed_sync_matches_former_pool_and_finish_bits() {
    const DIMENSIONS: usize = 256;
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, DIMENSIONS);
    let mut embedder = Model2VecEmbedder::load(dir.path()).unwrap();

    for &tokens in &[0_usize, 1, 2, 3, 4, 8, 16, 32, 64, 511, 512, 513] {
        let text = (0..tokens)
            .map(|position| match position % 4 {
                0 => "hello",
                1 => "world",
                2 => "missing-token",
                _ => "hello",
            })
            .collect::<Vec<_>>()
            .join(" ");
        let expected = former_embed_sync(&embedder, &text);
        let actual = embedder.embed_sync(&text).unwrap();
        assert_f32_bits_eq(&actual, &expected, &format!("tokens={tokens}"));
        assert_eq!(
            embedder
                .tokenizer
                .encode_fast(text.as_str(), false)
                .unwrap()
                .get_ids(),
            former_token_ids(&embedder, &text),
            "token IDs at native-256 boundary tokens={tokens}"
        );
        if !text.is_empty() {
            let token_count = embedder
                .tokenizer
                .encode(text.as_str(), false)
                .unwrap()
                .len();
            assert_eq!(
                last_model2vec_accumulation_route_for_test(),
                expected_native_256_route(token_count),
                "shipping embed_sync route for {tokens} input words ({token_count} token IDs)"
            );
        }
    }

    for text in [
        "hello caf\u{e9} world",
        "hello \u{6771}\u{4eac} hello",
        "HELLO hello HELLO",
    ] {
        let expected = former_embed_sync(&embedder, text);
        let actual = embedder.embed_sync(text).unwrap();
        assert_f32_bits_eq(&actual, &expected, text);
    }

    let hello = DIMENSIONS..DIMENSIONS * 2;
    embedder.embeddings[hello.clone()].fill(-0.0);
    let expected = former_embed_sync(&embedder, "hello hello");
    let actual = embedder.embed_sync("hello hello").unwrap();
    assert_f32_bits_eq(&actual, &expected, "signed-zero row");

    embedder.embeddings[hello.clone()].fill(1.0e-20);
    let expected = former_embed_sync(&embedder, "hello");
    let actual = embedder.embed_sync("hello").unwrap();
    assert_f32_bits_eq(&actual, &expected, "below normalization guard");

    embedder.embeddings[hello.clone()].fill(1.0e-4);
    let expected = former_embed_sync(&embedder, "hello");
    let actual = embedder.embed_sync("hello").unwrap();
    assert_f32_bits_eq(&actual, &expected, "above normalization guard");

    embedder.embeddings[hello].fill(f32::NAN);
    let expected = former_embed_sync(&embedder, "hello world hello");
    let actual = embedder.embed_sync("hello world hello").unwrap();
    assert_f32_bits_eq(&actual, &expected, "non-finite pooled row");
}

#[test]
fn fused_mean_and_ordered_norm_finish_matches_former_bits() {
    let arbitrary_initial_sum = (0_u32..256)
        .map(|index| {
            let value = f32::from_bits(0x3f80_0000 + index);
            if index % 2 == 0 { value } else { -value }
        })
        .collect::<Vec<_>>();
    let cases = vec![
        ("arbitrary finite sum", arbitrary_initial_sum),
        (
            "normalization guard boundary",
            vec![f32::MIN_POSITIVE, f32::EPSILON.sqrt(), -f32::EPSILON.sqrt()],
        ),
        ("signed zero", vec![-0.0, 0.0, -0.0, 0.0]),
        (
            "non-finite values",
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX],
        ),
    ];

    for (label, initial_sum) in cases {
        for &count in &[1_usize, 2, 3, 4, 511, 512, 513] {
            let mut former = initial_sum.clone();
            let mut fused = initial_sum.clone();
            finish_mean_pool_and_normalize_former(&mut former, count);
            finish_mean_pool_and_normalize(&mut fused, count);
            assert_f32_bits_eq(&fused, &former, &format!("{label}, count={count}"));
        }
    }
}

#[test]
fn full_embed_sync_fused_finish_matches_former_bits_across_shapes_and_values() {
    for &dimensions in &[1_usize, 255, 256, 257] {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, dimensions);
        let mut embedder = Model2VecEmbedder::load(dir.path()).unwrap();

        let mut full_embed_sync_corpus =
            vec![String::new(), "hello definitely-oov-token hello".to_owned()];
        for token_count in [1_usize, 2, 3, 4, 511, 512, 513] {
            full_embed_sync_corpus.push(
                std::iter::repeat_n("hello", token_count)
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }

        let oov_grouping_ids = former_token_ids(&embedder, &full_embed_sync_corpus[1]);
        assert!(
            oov_grouping_ids.contains(&0),
            "the full embed_sync OOV grouping corpus must emit the [UNK] token at dim={dimensions}"
        );

        let hello = dimensions..dimensions * 2;
        for (lane, value) in embedder.embeddings[hello.clone()].iter_mut().enumerate() {
            let lane = u32::try_from(lane).expect("fixture dimension fits u32");
            let magnitude = f32::from_bits(0x3f80_0000 + lane);
            *value = if lane % 2 == 0 { magnitude } else { -magnitude };
        }
        for text in &full_embed_sync_corpus {
            assert_fast_ids_and_former_vector_bits(
                &embedder,
                text,
                &format!("finite full embed_sync corpus dim={dimensions}"),
            );
        }

        #[allow(clippy::cast_precision_loss)]
        let guard_center = (f32::EPSILON / dimensions as f32).sqrt();
        let below_guard = f32::from_bits(guard_center.to_bits() - 1);
        let above_guard = f32::from_bits(guard_center.to_bits() + 1);
        for (scenario, value) in [
            ("subnormal", f32::from_bits(1)),
            ("below guard", below_guard),
            ("above guard", above_guard),
            ("signed zero", -0.0),
            ("NaN", f32::from_bits(0x7fc0_0001)),
            ("infinity", f32::INFINITY),
        ] {
            embedder.embeddings[hello.clone()].fill(value);
            for text in &full_embed_sync_corpus {
                assert_fast_ids_and_former_vector_bits(
                    &embedder,
                    text,
                    &format!("{scenario} full embed_sync corpus dim={dimensions}"),
                );
            }
        }

        let invalid_token_ids = [
            1_u32,
            u32::try_from(embedder.vocab_size).expect("fixture vocabulary fits u32"),
            u32::MAX,
            2_u32,
        ];
        let former = former_embed_token_ids(&embedder, &invalid_token_ids);
        let fused = embedder.embed_token_ids(&invalid_token_ids);
        assert_f32_bits_eq(
            &fused,
            &former,
            &format!("mixed valid/OOV token IDs dim={dimensions}"),
        );

        let all_oov_token_ids = [
            u32::try_from(embedder.vocab_size).expect("fixture vocabulary fits u32"),
            u32::MAX,
        ];
        let former = former_embed_token_ids(&embedder, &all_oov_token_ids);
        let fused = embedder.embed_token_ids(&all_oov_token_ids);
        assert_f32_bits_eq(
            &fused,
            &former,
            &format!("all OOV token IDs dim={dimensions}"),
        );
    }
}

/// Bit-exact producer conformance proof for the frozen potion manifest.
#[test]
#[ignore = "requires a verified potion model dir via POTION_FIXTURE_DIR"]
fn conformance_certificate_matches_fixture() {
    let dir = std::env::var("POTION_FIXTURE_DIR")
        .expect("set POTION_FIXTURE_DIR to a potion-multilingual-128M directory");
    let manifest = crate::model_manifest::ModelArtifactManifestV1::potion_128m_native()
        .expect("registered potion manifest");
    let verified = manifest
        .verify_dir(Path::new(&dir))
        .expect("verify frozen potion artifacts");
    let expected_identity = verified
        .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .expect("derive verified potion identity");
    let embedder = Model2VecEmbedder::load_preverified(
        Path::new(&dir),
        DEFAULT_MODEL_NAME,
        expected_identity.clone(),
    )
    .expect("load verified potion embedder");
    assert_eq!(embedder.identity().unwrap(), &expected_identity);

    let added_vocabulary = embedder.tokenizer.get_added_vocabulary().get_vocab();
    let pad_id = *added_vocabulary
        .get("[PAD]")
        .expect("verified Potion tokenizer must retain [PAD] as an added special token");
    let unk_id = *added_vocabulary
        .get("[UNK]")
        .expect("verified Potion tokenizer must retain [UNK] as an added special token");
    let added_special_text = "hello [PAD] [UNK] world";
    let long_over_512_text = std::iter::repeat_n("hello", 1024)
        .collect::<Vec<_>>()
        .join(" ");
    let mut parity_texts = crate::model_manifest::MODEL_CONFORMANCE_TEXTS_V1
        .iter()
        .map(|text| (*text).to_owned())
        .collect::<Vec<_>>();
    parity_texts.extend([
        "Caf\u{e9} na\u{ef}ve \u{2014} \u{6771}\u{4eac} \u{1f980}".to_owned(),
        added_special_text.to_owned(),
        "\tmetaspace  boundaries\tand unseen-oov-\u{10ffff}".to_owned(),
        long_over_512_text.clone(),
    ]);
    for text in &parity_texts {
        assert_fast_ids_and_former_vector_bits(
            &embedder,
            text,
            "verified Potion tokenizer parity input",
        );
    }

    let added_special_ids = former_token_ids(&embedder, added_special_text);
    assert!(
        added_special_ids.contains(&pad_id) && added_special_ids.contains(&unk_id),
        "verified Potion tokenizer must emit both literal added special-token IDs"
    );
    let former_long = embedder
        .tokenizer
        .encode(long_over_512_text.as_str(), false)
        .expect("encode long verified Potion input");
    let fast_long = embedder
        .tokenizer
        .encode_fast(long_over_512_text.as_str(), false)
        .expect("encode_fast long verified Potion input");
    assert!(
        embedder.tokenizer.get_truncation().is_none(),
        "registered Potion tokenizer must preserve its configured no-truncation policy"
    );
    assert!(
        former_long.len() > 512,
        "long verified Potion input must exceed the former 512-token synthetic boundary"
    );
    assert_eq!(
        fast_long.len(),
        former_long.len(),
        "verified Potion long-input token count diverged"
    );
    assert_eq!(
        fast_long.get_ids(),
        former_long.get_ids(),
        "verified Potion long-input token IDs diverged"
    );
    assert!(
        former_long.get_overflowing().is_empty() && fast_long.get_overflowing().is_empty(),
        "configured no-truncation policy must not emit overflow encodings"
    );

    let texts = &crate::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
    let former_vectors = texts
        .iter()
        .map(|text| {
            assert_fast_ids_and_former_vector_bits(
                &embedder,
                text,
                "verified Potion conformance input",
            );
            former_embed_sync(&embedder, text)
        })
        .collect::<Vec<_>>();
    let vectors = embedder
        .embed_batch_sync(texts)
        .expect("embed bounded conformance corpus");
    assert_eq!(vectors.len(), former_vectors.len());
    for (index, (actual, expected)) in vectors.iter().zip(&former_vectors).enumerate() {
        assert_f32_bits_eq(
            actual,
            expected,
            &format!("verified Potion batch order or vector bits changed at index={index}"),
        );
    }
    let observed =
        frankensearch_core::generation::GoldenVectorCertificateV1::from_exact_f32(texts, &vectors)
            .expect("compute exact conformance certificate");
    let expected = manifest.execution.golden_vectors;
    assert_eq!(
        observed, expected,
        "Model2Vec output bits drifted from the registered producer certificate"
    );
}
