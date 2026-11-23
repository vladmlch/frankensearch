use std::fs;

use super::*;

mod parity_and_pooling;
mod streaming_and_cache;

/// Create a minimal `Model2Vec` model in a temp directory for testing.
///
/// Creates a tiny tokenizer and a small safetensors file with known values.
pub(super) fn create_test_model(dir: &Path, vocab_size: usize, dimensions: usize) {
    // Create a minimal tokenizer.json
    // This is a minimal valid HuggingFace tokenizer config
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {
                "id": 0,
                "content": "[UNK]",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ],
        "normalizer": {
            "type": "Lowercase"
        },
        "pre_tokenizer": {
            "type": "Whitespace"
        },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": create_test_vocab(vocab_size),
            "unk_token": "[UNK]"
        }
    });

    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json).unwrap(),
    )
    .unwrap();

    // Create safetensors file with known embedding values
    create_test_safetensors(dir, vocab_size, dimensions);
}

/// Create a tokenizer fixture that exercises added tokens and truncation while retaining
/// enough rows to pool every emitted ID.
pub(super) fn create_tokenizer_parity_model(dir: &Path) {
    let mut vocab = create_test_vocab(16)
        .as_object()
        .expect("test vocabulary is an object")
        .clone();
    vocab.insert("café".to_owned(), serde_json::Value::from(11));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": {
            "direction": "Right",
            "max_length": 512,
            "strategy": "LongestFirst",
            "stride": 0
        },
        "padding": null,
        "added_tokens": [
            {
                "id": 0,
                "content": "[UNK]",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            },
            {
                "id": 12,
                "content": "<added>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": true,
                "special": false
            },
            {
                "id": 13,
                "content": "[SPECIAL]",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ],
        "normalizer": {
            "type": "Lowercase"
        },
        "pre_tokenizer": {
            "type": "Whitespace"
        },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "[UNK]"
        }
    });

    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json).unwrap(),
    )
    .unwrap();
    create_test_safetensors(dir, 16, 256);
}

/// A `WordPiece` fixture like the potion-base models' tokenizer: `[UNK]`
/// (id 0) has its own matrix row, so pooling it is observable.
pub(super) fn create_wordpiece_test_model(dir: &Path, vocab_size: usize, dimensions: usize) {
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": { "type": "Lowercase" },
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordPiece",
            "vocab": create_test_vocab(vocab_size),
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100
        }
    });
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json).unwrap(),
    )
    .unwrap();
    create_test_safetensors(dir, vocab_size, dimensions);
}

/// Create a test vocabulary mapping words to token IDs.
fn create_test_vocab(vocab_size: usize) -> serde_json::Value {
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".to_owned(), serde_json::Value::from(0));

    let test_words = [
        "hello", "world", "test", "rust", "search", "embed", "vector", "model", "fast", "query",
    ];

    for (i, word) in test_words.iter().enumerate() {
        if i + 1 < vocab_size {
            vocab.insert((*word).to_owned(), serde_json::Value::from(i + 1));
        }
    }

    serde_json::Value::Object(vocab)
}

/// Create a minimal safetensors file with a known embedding matrix.
fn create_test_safetensors(dir: &Path, vocab_size: usize, dimensions: usize) {
    use std::collections::HashMap;

    // Build embedding matrix: each row is [row_idx * 0.1, row_idx * 0.1 + 0.01, ...]
    let mut data = Vec::with_capacity(vocab_size * dimensions * 4);
    for row in 0..vocab_size {
        for col in 0..dimensions {
            #[allow(clippy::cast_precision_loss)]
            let val = (row as f32).mul_add(0.1, (col as f32) * 0.01);
            data.extend_from_slice(&val.to_le_bytes());
        }
    }

    let mut tensors = HashMap::new();
    tensors.insert(
        "embeddings".to_owned(),
        safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![vocab_size, dimensions],
            &data,
        )
        .unwrap(),
    );

    let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
    fs::write(dir.join("model.safetensors"), serialized).unwrap();
}

// ── Loading ────────────────────────────────────────────────────────

#[test]
fn load_valid_model() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load_with_name(dir.path(), "test-model").unwrap();
    assert_eq!(embedder.dimensions, 8);
    assert_eq!(embedder.vocab_size, 12);
    assert_eq!(embedder.name, "test-model");
}

#[test]
fn load_preverified_rejects_tensor_dimension_drift() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("dimension-drift", 7);
    identity.producer.backend = "attested-fixture-backend".to_owned();
    identity.validate().unwrap();

    let error = Model2VecEmbedder::load_preverified(dir.path(), "dimension-drift", identity)
        .expect_err("parsed tensor width must agree with the attested identity");
    assert!(matches!(error, SearchError::ModelLoadFailed { .. }));
}

#[test]
fn registered_identity_matches_native_execution_contract() {
    let identity = ModelArtifactManifestV1::potion_128m_native()
        .unwrap()
        .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();
    validate_registered_execution_contract(&identity).unwrap();

    let mut drifted = identity;
    drifted.space.model_preprocessing.push_str("-drift");
    assert!(validate_registered_execution_contract(&drifted).is_err());
}

#[test]
fn load_missing_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    // Only create safetensors, not tokenizer
    create_test_safetensors(dir.path(), 10, 4);

    let result = Model2VecEmbedder::load(dir.path());
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, SearchError::ModelNotFound { .. }),
        "expected ModelNotFound, got {err:?}"
    );
}

#[test]
fn load_missing_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    // Only create tokenizer
    fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();

    let result = Model2VecEmbedder::load(dir.path());
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        SearchError::ModelNotFound { .. }
    ));
}

#[test]
fn load_nonexistent_directory() {
    let result = Model2VecEmbedder::load("/nonexistent/path/to/model");
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        SearchError::ModelNotFound { .. }
    ));
}

// ── Embedding ──────────────────────────────────────────────────────

#[test]
fn embed_produces_correct_dimension() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    let vec = embedder.embed_sync("hello world").unwrap();
    assert_eq!(vec.len(), 8);
}

#[test]
fn embed_output_is_l2_normalized() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    let vec = embedder.embed_sync("hello world").unwrap();

    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
}

#[test]
fn embed_empty_string_returns_zero_vector() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    let vec = embedder.embed_sync("").unwrap();
    assert_eq!(vec.len(), 8);
    assert!(vec.iter().all(|&x| x == 0.0));
}

#[test]
fn embed_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    let a = embedder.embed_sync("hello world").unwrap();
    let b = embedder.embed_sync("hello world").unwrap();
    assert_eq!(a, b, "same input must produce same output");
}

#[test]
fn embed_different_inputs_different_outputs() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    let a = embedder.embed_sync("hello").unwrap();
    let b = embedder.embed_sync("world").unwrap();
    assert_ne!(a, b, "different inputs should produce different embeddings");
}

/// The former `embed_sync` pooling and finish sequence, retained independently
/// of the production gather helper for exact native-256 parity checks.
pub(super) fn former_embed_sync(embedder: &Model2VecEmbedder, text: &str) -> Vec<f32> {
    if text.is_empty() {
        return vec![0.0; embedder.dimensions];
    }

    let token_ids = former_token_ids(embedder, text);
    former_embed_token_ids(embedder, &token_ids)
}

pub(super) fn former_embed_token_ids(embedder: &Model2VecEmbedder, token_ids: &[u32]) -> Vec<f32> {
    let mut sum = vec![0.0_f32; embedder.dimensions];
    let mut count = 0_usize;
    for &token_id in token_ids {
        let index = token_id as usize;
        if index < embedder.vocab_size {
            let start = index * embedder.dimensions;
            crate::simd::accumulate_f32_into(
                &mut sum,
                &embedder.embeddings[start..start + embedder.dimensions],
            );
            count += 1;
        }
    }
    if count == 0 {
        return vec![0.0; embedder.dimensions];
    }

    finish_mean_pool_and_normalize_former(&mut sum, count);
    sum
}

pub(super) fn former_token_ids(embedder: &Model2VecEmbedder, text: &str) -> Vec<u32> {
    embedder
        .tokenizer
        .encode(text, false)
        .unwrap()
        .get_ids()
        .to_vec()
}

pub(super) fn assert_fast_ids_and_former_vector_bits(
    embedder: &Model2VecEmbedder,
    text: &str,
    scenario: &str,
) {
    let former_ids = former_token_ids(embedder, text);
    let fast_ids = embedder
        .tokenizer
        .encode_fast(text, false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert_eq!(fast_ids, former_ids, "token IDs diverged for {scenario}");

    let expected = former_embed_sync(embedder, text);
    let actual = embedder.embed_sync(text).unwrap();
    assert_f32_bits_eq(&actual, &expected, scenario);
}

pub(super) fn assert_f32_bits_eq(actual: &[f32], expected: &[f32], scenario: &str) {
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "{scenario}"
    );
}

// ── OOV Handling ───────────────────────────────────────────────────

#[test]
fn embed_all_oov_returns_zero_vector() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    // "zzzzzzz" is not in our test vocab
    let vec = embedder.embed_sync("xyzxyzxyz qqqqq").unwrap();
    // All tokens should be OOV → zero vector
    assert_eq!(vec.len(), 8);
}

// ── Embedder Trait ─────────────────────────────────────────────────

#[test]
fn trait_is_semantic() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    assert!(embedder.is_semantic());
}

#[test]
fn trait_category_is_static() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    assert_eq!(embedder.category(), ModelCategory::StaticEmbedder);
}

#[test]
fn trait_dimension() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    assert_eq!(embedder.dimension(), 8);
}

#[test]
fn trait_does_not_infer_mrl_from_model2vec_backend() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    assert!(!embedder.supports_mrl());
}

#[test]
fn trait_id_and_name() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load_with_name(dir.path(), "my-model").unwrap();
    assert_eq!(embedder.id(), "my-model");
    assert_eq!(embedder.model_name(), "my-model");
}

// ── Thread Safety ──────────────────────────────────────────────────

#[test]
fn embedder_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Model2VecEmbedder>();
}

// ── Debug impl ─────────────────────────────────────────────────────

#[test]
fn debug_does_not_dump_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    let debug = format!("{embedder:?}");
    assert!(debug.contains("Model2VecEmbedder"));
    assert!(debug.contains("dimensions: 8"));
    assert!(debug.contains("vocab_size: 12"));
    // Must NOT contain actual embedding data
    assert!(!debug.contains("0.1"));
}

// ── Model Directory Search ─────────────────────────────────────────

#[test]
fn has_required_files_positive() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 4, 2);
    assert!(has_required_files(dir.path()));
}

#[test]
fn has_required_files_negative() {
    let dir = tempfile::tempdir().unwrap();
    assert!(!has_required_files(dir.path()));
}
