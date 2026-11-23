use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use tokenizers::Tokenizer;

use super::super::safetensors_stream::{
    MATRIX_READ_CHUNK_BYTES, discover_tensor_name, parse_f32_matrix,
};
use super::super::{
    DEFAULT_MODEL_NAME, Model2VecEmbedder, Model2VecTokenizer, model2vec_full_loads_for,
};
use super::{assert_f32_bits_eq, create_test_model};

// ── Tensor Discovery ───────────────────────────────────────────────

#[test]
fn tensor_discovery_finds_standard_name() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 4, 2);

    // The test model uses "embeddings" as tensor name
    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    assert_eq!(embedder.vocab_size, 4);
}

#[test]
fn tensor_discovery_single_tensor_fallback() {
    let dir = tempfile::tempdir().unwrap();

    // Create tokenizer
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "added_tokens": [],
        "model": {
            "type": "WordLevel",
            "vocab": {"hello": 0, "world": 1},
            "unk_token": "hello"
        }
    });
    fs::write(
        dir.path().join("tokenizer.json"),
        serde_json::to_string(&tokenizer_json).unwrap(),
    )
    .unwrap();

    // Create safetensors with a non-standard tensor name
    let mut data = vec![0u8; 2 * 3 * 4]; // 2 rows × 3 dims × 4 bytes
    for (i, chunk) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let val = i as f32;
        chunk.copy_from_slice(&val.to_le_bytes());
    }

    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "my_custom_tensor_name".to_owned(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 3], &data).unwrap(),
    );

    let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
    fs::write(dir.path().join("model.safetensors"), serialized).unwrap();

    // Should fall back to the single tensor
    let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
    assert_eq!(embedder.vocab_size, 2);
    assert_eq!(embedder.dimensions, 3);
}

// ── Parse Matrix ───────────────────────────────────────────────────

#[test]
fn parse_f32_matrix_correct() {
    // 2 rows × 2 dims = 16 bytes
    let data: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let matrix = parse_f32_matrix(&data, 2, 2).unwrap();
    assert_eq!(matrix.len(), 4);
    assert_eq!(&matrix[0..2], &[1.0, 2.0]);
    assert_eq!(&matrix[2..4], &[3.0, 4.0]);
}

#[test]
fn parse_f32_matrix_too_short() {
    let data = vec![0u8; 4]; // Only 1 float, need more
    let result = parse_f32_matrix(&data, 2, 2);
    assert!(result.is_err());
}

#[test]
fn parse_f32_matrix_too_long() {
    // 16 bytes expected for [2 x 2] f32, plus one trailing garbage byte
    let mut data: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    data.push(0xAA);

    let result = parse_f32_matrix(&data, 2, 2);
    assert!(result.is_err());
}

#[test]
fn parse_f32_matrix_size_overflow() {
    let result = parse_f32_matrix(&[], usize::MAX, 2);
    assert!(result.is_err());
}

// ── GH #46: streamed matrix load and the process-wide instance cache ──

/// Serialises the real-model tests. They all point at the SAME directory
/// (`POTION_FIXTURE_DIR`), so the per-directory load counter is only
/// deterministic while one of them runs at a time.
fn real_model_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Decode a safetensors file the way the loader did BEFORE #46: read the
/// whole file, deserialize over that buffer, then materialise the matrix.
/// This is the oracle every bit-identity assertion below compares against.
fn whole_buffer_decode(model_dir: &Path) -> (usize, usize, Vec<f32>) {
    let bytes = fs::read(model_dir.join("model.safetensors")).unwrap();
    let safetensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let names = safetensors.names();
    let tensor_name = discover_tensor_name(&names).expect("oracle: no embedding tensor");
    let tensor = safetensors.tensor(&tensor_name).unwrap();
    let shape = tensor.shape();
    assert_eq!(shape.len(), 2, "oracle: expected a 2D tensor");
    let (vocab_size, dimensions) = (shape[0], shape[1]);
    let values = parse_f32_matrix(tensor.data(), vocab_size, dimensions).unwrap();
    (vocab_size, dimensions, values)
}

#[test]
fn streamed_matrix_is_bit_identical_to_whole_buffer_decode() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let embedder = Model2VecEmbedder::load_with_name(dir.path(), "test-model").unwrap();
    let (vocab_size, dimensions, expected) = whole_buffer_decode(dir.path());

    assert_eq!(embedder.vocab_size, vocab_size);
    assert_eq!(embedder.dimensions, dimensions);
    assert_f32_bits_eq(
        &embedder.embeddings,
        &expected,
        "streamed matrix must decode to the same bits as the pre-#46 whole-buffer path",
    );
}

#[test]
fn streamed_matrix_survives_a_chunk_boundary() {
    // MATRIX_READ_CHUNK_BYTES is 1 MiB; make the matrix straddle it so the
    // multi-chunk path, not just the single-read path, is exercised.
    let dimensions = 256;
    let vocab_size = (MATRIX_READ_CHUNK_BYTES / (dimensions * 4)) * 2 + 3;
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), vocab_size, dimensions);

    let embedder = Model2VecEmbedder::load_with_name(dir.path(), "chunked").unwrap();
    let (_, _, expected) = whole_buffer_decode(dir.path());
    assert!(
        expected.len() * 4 > MATRIX_READ_CHUNK_BYTES,
        "fixture must be larger than one read chunk"
    );
    assert_f32_bits_eq(
        &embedder.embeddings,
        &expected,
        "a matrix spanning several read chunks must decode identically",
    );
}

#[test]
fn shared_load_builds_the_model_once_per_process() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);
    let key = std::fs::canonicalize(dir.path()).unwrap();

    assert_eq!(model2vec_full_loads_for(&key), 0);

    let first = Model2VecEmbedder::load_shared_with_name(dir.path(), "shared").unwrap();
    assert_eq!(
        model2vec_full_loads_for(&key),
        1,
        "the first shared load must build the model"
    );

    let second = Model2VecEmbedder::load_shared_with_name(dir.path(), "shared").unwrap();
    assert_eq!(
        model2vec_full_loads_for(&key),
        1,
        "the second load in the same process must not re-read the matrix or rebuild the tokenizer"
    );
    assert!(
        Arc::ptr_eq(&first, &second),
        "both callers must share one instance"
    );

    // And a third, to prove the hit is not a one-off.
    let third = Model2VecEmbedder::load_shared_with_name(dir.path(), "shared").unwrap();
    assert_eq!(model2vec_full_loads_for(&key), 1);
    assert!(Arc::ptr_eq(&first, &third));
}

#[test]
fn shared_load_is_bit_identical_to_a_direct_load() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let direct = Model2VecEmbedder::load_with_name(dir.path(), "shared-parity").unwrap();
    let shared = Model2VecEmbedder::load_shared_with_name(dir.path(), "shared-parity").unwrap();

    assert_f32_bits_eq(
        &shared.embeddings,
        &direct.embeddings,
        "a shared load must carry the same matrix bits as a direct one",
    );
    for text in ["hello world", "rust search", "", "totally-unseen-token"] {
        assert_f32_bits_eq(
            &shared.embed_sync(text).unwrap(),
            &direct.embed_sync(text).unwrap(),
            "shared and direct embedders must produce identical embeddings",
        );
    }
}

#[test]
fn shared_load_reloads_after_the_last_reference_drops() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);
    let key = std::fs::canonicalize(dir.path()).unwrap();

    {
        let held = Model2VecEmbedder::load_shared_with_name(dir.path(), "weak").unwrap();
        assert_eq!(model2vec_full_loads_for(&key), 1);
        drop(held);
    }

    // The cache holds only a Weak, so the 512 MB matrix is not pinned for
    // the life of the process once nobody wants it.
    let reloaded = Model2VecEmbedder::load_shared_with_name(dir.path(), "weak").unwrap();
    assert_eq!(
        model2vec_full_loads_for(&key),
        2,
        "a released model must be rebuilt on the next request, not resurrected"
    );
    assert_eq!(reloaded.vocab_size, 12);
}

#[test]
fn shared_load_keeps_distinct_model_directories_apart() {
    let first_dir = tempfile::tempdir().unwrap();
    let second_dir = tempfile::tempdir().unwrap();
    create_test_model(first_dir.path(), 12, 8);
    create_test_model(second_dir.path(), 10, 4);

    let first = Model2VecEmbedder::load_shared_with_name(first_dir.path(), "shared").unwrap();
    let second = Model2VecEmbedder::load_shared_with_name(second_dir.path(), "shared").unwrap();

    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first.dimensions, 8);
    assert_eq!(second.dimensions, 4);
}

#[test]
fn shared_load_keeps_distinct_display_names_apart() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    // The display name is part of the cache key on purpose: returning an
    // instance whose `name` disagrees with the caller's request would be
    // worse than loading twice. Callers that want to share agree on it.
    let first = Model2VecEmbedder::load_shared_with_name(dir.path(), "alpha").unwrap();
    let second = Model2VecEmbedder::load_shared_with_name(dir.path(), "beta").unwrap();

    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first.name, "alpha");
    assert_eq!(second.name, "beta");
}

#[test]
fn load_rejects_a_truncated_safetensors_file() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let path = dir.path().join("model.safetensors");
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();

    let error = Model2VecEmbedder::load_with_name(dir.path(), "truncated").unwrap_err();
    assert!(
        error.to_string().contains("length mismatch"),
        "a truncated artifact must be refused by the completeness check, got: {error}"
    );
}

#[test]
fn load_rejects_a_trailing_garbage_safetensors_file() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let path = dir.path().join("model.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    bytes.extend_from_slice(&[0xAA; 16]);
    fs::write(&path, &bytes).unwrap();

    let error = Model2VecEmbedder::load_with_name(dir.path(), "overlong").unwrap_err();
    assert!(
        error.to_string().contains("length mismatch"),
        "an over-long artifact must be refused too, got: {error}"
    );
}

#[test]
fn load_rejects_an_absurd_header_length_prefix() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    let path = dir.path().join("model.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    fs::write(&path, &bytes).unwrap();

    let error = Model2VecEmbedder::load_with_name(dir.path(), "huge-header").unwrap_err();
    assert!(
        error.to_string().contains("exceeds"),
        "a corrupt length prefix must be bounded before it becomes an allocation, got: {error}"
    );
}

#[test]
fn load_rejects_a_non_f32_embedding_tensor() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    // Same name and rank, different dtype. The pre-#46 path caught this
    // only as a byte-count mismatch; the dtype is now checked outright.
    let data = vec![0_u8; 12 * 8 * 4];
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "embeddings".to_owned(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::I32, vec![12, 8], &data).unwrap(),
    );
    let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
    fs::write(dir.path().join("model.safetensors"), serialized).unwrap();

    let error = Model2VecEmbedder::load_with_name(dir.path(), "int-tensor").unwrap_err();
    assert!(
        error.to_string().contains("F32"),
        "a non-F32 tensor must be named as such, got: {error}"
    );
}

/// Bit-identity on the REAL artifact. A synthetic fixture cannot prove
/// this: it has one small tensor, no added/special tokens and no
/// multi-chunk read. Run with:
///
/// ```text
/// POTION_FIXTURE_DIR=$HOME/.local/share/frankensearch/models/potion-multilingual-128M \
///   cargo test -p frankensearch-embed --features model2vec --lib -- --ignored real_model
/// ```
#[test]
#[ignore = "requires a real potion model dir via POTION_FIXTURE_DIR"]
fn real_model_streamed_matrix_is_bit_identical_to_whole_buffer_decode() {
    let _serialized = real_model_test_lock()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let dir = std::env::var("POTION_FIXTURE_DIR")
        .expect("set POTION_FIXTURE_DIR to a potion-multilingual-128M directory");
    let dir = Path::new(&dir);

    let embedder = Model2VecEmbedder::load_with_name(dir, DEFAULT_MODEL_NAME).unwrap();
    let (vocab_size, dimensions, expected) = whole_buffer_decode(dir);

    assert_eq!(embedder.vocab_size, vocab_size);
    assert_eq!(embedder.dimensions, dimensions);
    assert_eq!(embedder.embeddings.len(), expected.len());
    assert_f32_bits_eq(
        &embedder.embeddings,
        &expected,
        "the real 512 MB matrix must stream to exactly the bits the whole-buffer path produced",
    );

    // And the embeddings the matrix feeds, over the registered
    // conformance corpus plus the multilingual/emoji cases.
    let texts = [
        "hello world",
        "semantic search finds related ideas",
        "identifier fsvi_v2",
        "naive cafe Tokyo",
        "Café naïve — 東京 🦀",
        "",
    ];
    let reference = Model2VecEmbedder {
        // The library tokenizer, so this reference also cross-checks the
        // compact Unigram model the embedder under test loads.
        tokenizer: Model2VecTokenizer::Library(
            Tokenizer::from_file(dir.join("tokenizer.json")).unwrap(),
        ),
        embeddings: expected,
        dimensions,
        vocab_size,
        name: DEFAULT_MODEL_NAME.to_owned(),
        model_dir: dir.to_owned(),
        identity: embedder.identity.clone(),
        dropped_token_id: embedder.dropped_token_id,
        last_tokenizer_route_was_offset_free: AtomicBool::new(false),
    };
    for text in texts {
        assert_f32_bits_eq(
            &embedder.embed_sync(text).unwrap(),
            &reference.embed_sync(text).unwrap(),
            "embeddings must be bit-identical before and after the streamed load",
        );
    }
}

/// Reuse on the REAL artifact: the second load in a process must not
/// re-read the 512 MB matrix or rebuild the 500k-piece tokenizer. Same
/// env gate as above.
#[test]
#[ignore = "requires a real potion model dir via POTION_FIXTURE_DIR"]
fn real_model_shared_load_is_built_once_per_process() {
    let _serialized = real_model_test_lock()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let dir = std::env::var("POTION_FIXTURE_DIR")
        .expect("set POTION_FIXTURE_DIR to a potion-multilingual-128M directory");
    let dir = Path::new(&dir);
    let key = std::fs::canonicalize(dir).unwrap();

    let before = model2vec_full_loads_for(&key);
    let first = Model2VecEmbedder::load_shared_with_name(dir, DEFAULT_MODEL_NAME).unwrap();
    assert_eq!(model2vec_full_loads_for(&key), before + 1);

    let second = Model2VecEmbedder::load_shared_with_name(dir, DEFAULT_MODEL_NAME).unwrap();

    // The counter and the pointer identity ARE the proof. A wall-clock
    // threshold would add nothing but a flake on a loaded build host.
    assert_eq!(
        model2vec_full_loads_for(&key),
        before + 1,
        "the second real-model load must be served from the process cache"
    );
    assert!(Arc::ptr_eq(&first, &second));
}

/// Offset validation must survive the move off `SafeTensors::deserialize`.
///
/// `locate` parses the header by deserializing `safetensors::tensor::
/// Metadata` directly, which reaches `Metadata::new` -> `validate()` and
/// therefore enforces contiguity from offset 0. That is an implementation
/// detail of the upstream crate — `Deserialize for Metadata` does not
/// document it — so pin it here: if a future `safetensors` routes
/// deserialization around `validate()`, this loader would silently stop
/// checking offsets and still compile.
#[test]
fn load_rejects_non_contiguous_tensor_offsets() {
    let dir = tempfile::tempdir().unwrap();
    create_test_model(dir.path(), 12, 8);

    // A header whose single tensor does not start at offset 0. Written by
    // hand because the safetensors serializer cannot emit one.
    let payload = vec![0_u8; 16 + 12 * 8 * 4];
    let header =
        br#"{"embeddings":{"dtype":"F32","shape":[12,8],"data_offsets":[16,400]}}"#.to_vec();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload[..400]);
    fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

    let error = Model2VecEmbedder::load_with_name(dir.path(), "gapped").unwrap_err();
    let text = error.to_string();
    assert!(
        text.contains("failed to parse safetensors"),
        "a tensor that does not start at offset 0 must be refused by the header validation, got: {text}"
    );
}
