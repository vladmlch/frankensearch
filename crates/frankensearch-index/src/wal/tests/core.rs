use std::fs;

use frankensearch_core::SearchError;
use half::f16;

use super::super::io::{decode_vector, wal_entry_count};
use super::super::*;
use super::{make_entry, temp_wal_path};

#[test]
fn decode_vector_f16_simd_matches_scalar_bit_exact() {
    // Exercise both the SIMD 16-byte block path and the scalar tail: dim
    // 13 = one 8-lane block + a 5-element remainder. Values span normal,
    // subnormal, signed, and non-finite f16 encodings so any magic-factor
    // widen mismatch surfaces as a bit difference (bd-y3hf).
    let f16_values: [f16; 13] = [
        f16::from_f32(0.0),
        f16::from_f32(-0.0),
        f16::from_f32(1.5),
        f16::from_f32(-2.25),
        f16::from_f32(65504.0),
        f16::from_f32(-65504.0),
        f16::from_bits(0x0001),
        f16::from_bits(0x8001),
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::NAN,
        f16::from_f32(0.333_251),
        f16::from_f32(-123.4),
    ];
    let mut bytes = Vec::with_capacity(f16_values.len() * 2);
    for value in f16_values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let simd = decode_vector(&bytes, f16_values.len(), Quantization::F16);
    let scalar: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .take(f16_values.len())
        .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
        .collect();

    assert_eq!(simd.len(), f16_values.len());
    assert_eq!(scalar.len(), f16_values.len());
    for (index, (lhs, rhs)) in simd.iter().zip(scalar.iter()).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "SIMD-widen f16 decode diverged from scalar at lane {index}",
        );
    }
}

#[test]
fn roundtrip_single_batch() {
    let path = temp_wal_path("roundtrip-single");
    let dim = 4;
    let entries = vec![make_entry("doc-0", 1.0, dim), make_entry("doc-1", 2.0, dim)];

    append_wal_batch(&path, &entries, dim, Quantization::F16, 0, false).unwrap();

    let (loaded, generation, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(generation, 0);
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].doc_id, "doc-0");
    assert_eq!(loaded[1].doc_id, "doc-1");
    assert!((loaded[0].embedding[0] - 1.0).abs() < 0.01);
    assert!((loaded[1].embedding[0] - 2.0).abs() < 0.01);

    std::fs::remove_file(&path).ok();
}

#[test]
fn roundtrip_multiple_batches() {
    let path = temp_wal_path("roundtrip-multi");
    let dim = 4;

    append_wal_batch(
        &path,
        &[make_entry("doc-0", 1.0, dim)],
        dim,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();
    append_wal_batch(
        &path,
        &[make_entry("doc-1", 2.0, dim), make_entry("doc-2", 3.0, dim)],
        dim,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    let (loaded, generation, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(generation, 0);
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded[0].doc_id, "doc-0");
    assert_eq!(loaded[1].doc_id, "doc-1");
    assert_eq!(loaded[2].doc_id, "doc-2");

    std::fs::remove_file(&path).ok();
}

#[test]
fn truncated_batch_is_discarded() {
    let path = temp_wal_path("truncated");
    let dim = 4;

    append_wal_batch(
        &path,
        &[make_entry("doc-good", 1.0, dim)],
        dim,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    // Append garbage to simulate a partial batch write.
    let mut data = std::fs::read(&path).unwrap();
    data.extend_from_slice(&BATCH_MAGIC);
    data.extend_from_slice(&1_u32.to_le_bytes()); // claims 1 entry
    data.extend_from_slice(&[0xFF; 3]); // truncated entry
    std::fs::write(&path, &data).unwrap();

    let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(loaded.len(), 1, "only the good batch should survive");
    assert_eq!(loaded[0].doc_id, "doc-good");

    std::fs::remove_file(&path).ok();
}

#[test]
fn crc_corrupted_batch_is_discarded() {
    let path = temp_wal_path("crc-corrupt");
    let dim = 4;

    append_wal_batch(
        &path,
        &[make_entry("doc-good", 1.0, dim)],
        dim,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();
    append_wal_batch(
        &path,
        &[make_entry("doc-bad", 2.0, dim)],
        dim,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    // Corrupt the CRC of the second batch.
    let mut data = std::fs::read(&path).unwrap();
    let last_byte = data.len() - 1;
    data[last_byte] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(
        loaded.len(),
        1,
        "corrupted second batch should be discarded"
    );
    assert_eq!(loaded[0].doc_id, "doc-good");

    std::fs::remove_file(&path).ok();
}

#[test]
fn nonexistent_wal_returns_empty() {
    let path = temp_wal_path("nonexistent");
    let (loaded, _, _) = read_wal(&path, 4, Quantization::F16).unwrap();
    assert!(loaded.is_empty());
}

#[test]
fn dimension_mismatch_is_error() {
    let path = temp_wal_path("dim-mismatch");
    append_wal_batch(
        &path,
        &[make_entry("doc-0", 1.0, 4)],
        4,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    let result = read_wal(&path, 8, Quantization::F16);
    assert!(result.is_err());

    std::fs::remove_file(&path).ok();
}

#[test]
fn f32_quantization_roundtrip() {
    let path = temp_wal_path("f32-quant");
    let dim = 4;
    let entries = vec![make_entry("doc-0", 0.123_456, dim)];

    append_wal_batch(&path, &entries, dim, Quantization::F32, 0, false).unwrap();

    let (loaded, _, _) = read_wal(&path, dim, Quantization::F32).unwrap();
    assert_eq!(loaded.len(), 1);
    assert!(
        (loaded[0].embedding[0] - 0.123_456).abs() < f32::EPSILON,
        "f32 should round-trip exactly"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn wal_path_derivation() {
    let fsvi = Path::new("/data/index.fsvi");
    let wal = wal_path_for(fsvi);
    assert_eq!(wal.to_str().unwrap(), "/data/index.fsvi.wal");
}

#[test]
fn remove_wal_is_idempotent() {
    let path = temp_wal_path("remove-idem");
    remove_wal(&path).unwrap(); // doesn't exist — ok
    std::fs::write(&path, b"dummy").unwrap();
    remove_wal(&path).unwrap(); // exists — removed
    assert!(!path.exists());
    remove_wal(&path).unwrap(); // gone again — ok
}

#[test]
fn index_tagging_roundtrip() {
    let pos = 42;
    let tagged = to_wal_index(pos);
    assert!(is_wal_index(tagged));
    assert!(!is_wal_index(pos));
    assert_eq!(from_wal_index(tagged), pos);
}

// ─── bd-vbzm tests begin ───

#[test]
fn wal_config_default_values() {
    let config = WalConfig::default();
    assert_eq!(config.compaction_threshold, 1000);
    assert!((config.compaction_ratio - 0.10).abs() < f64::EPSILON);
    assert!(config.fsync_on_write);
}

#[test]
fn wal_config_debug_clone() {
    let config = WalConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.compaction_threshold, config.compaction_threshold);
    let debug = format!("{config:?}");
    assert!(debug.contains("WalConfig"));
}

#[test]
fn compaction_stats_debug_clone() {
    let stats = CompactionStats {
        main_records_before: 100,
        wal_records: 10,
        total_records_after: 110,
        elapsed_ms: 42.5,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.main_records_before, 100);
    assert_eq!(cloned.wal_records, 10);
    assert_eq!(cloned.total_records_after, 110);
    assert!((cloned.elapsed_ms - 42.5).abs() < f64::EPSILON);
    let debug = format!("{stats:?}");
    assert!(debug.contains("CompactionStats"));
}

#[test]
fn wal_too_small_returns_empty() {
    let path = temp_wal_path("too-small");
    std::fs::write(&path, [0u8; 10]).unwrap();
    let (loaded, _, _) = read_wal(&path, 4, Quantization::F16).unwrap();
    assert!(loaded.is_empty());
    std::fs::remove_file(&path).ok();
}

#[test]
fn bad_magic_bytes_is_error() {
    let path = temp_wal_path("bad-magic");
    let mut header = vec![0xDE, 0xAD, 0xBE, 0xEF]; // wrong magic
    header.extend_from_slice(&WAL_VERSION.to_le_bytes());
    header.extend_from_slice(&4_u32.to_le_bytes());
    header.push(Quantization::F16 as u8);
    header.push(0); // compaction_gen
    header.extend_from_slice(&[0u8; 4]);
    let crc = crc32_of(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    std::fs::write(&path, &header).unwrap();

    let result = read_wal(&path, 4, Quantization::F16);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("magic"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn version_mismatch_is_error() {
    let path = temp_wal_path("version-mismatch");
    let mut header = Vec::new();
    header.extend_from_slice(&WAL_MAGIC);
    header.extend_from_slice(&99_u16.to_le_bytes()); // wrong version
    header.extend_from_slice(&4_u32.to_le_bytes());
    header.push(Quantization::F16 as u8);
    header.push(0); // compaction_gen
    header.extend_from_slice(&[0u8; 4]);
    let crc = crc32_of(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    std::fs::write(&path, &header).unwrap();

    let result = read_wal(&path, 4, Quantization::F16);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("version"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn quantization_mismatch_is_error() {
    let path = temp_wal_path("quant-mismatch");
    append_wal_batch(
        &path,
        &[make_entry("doc-0", 1.0, 4)],
        4,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    let result = read_wal(&path, 4, Quantization::F32);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("quantization"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn header_crc_mismatch_is_error() {
    let path = temp_wal_path("header-crc");
    append_wal_batch(
        &path,
        &[make_entry("doc-0", 1.0, 4)],
        4,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    let mut data = std::fs::read(&path).unwrap();
    // Corrupt header CRC (bytes 16-19)
    data[16] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    let result = read_wal(&path, 4, Quantization::F16);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("CRC"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn empty_batch_roundtrip() {
    let path = temp_wal_path("empty-batch");
    let dim = 4;
    append_wal_batch(&path, &[], dim, Quantization::F16, 0, false).unwrap();
    let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert!(loaded.is_empty());
    std::fs::remove_file(&path).ok();
}

#[test]
fn wal_entry_doc_id_hash_matches_fnv1a() {
    let entry = make_entry("hello", 1.0, 4);
    assert_eq!(entry.doc_id_hash, crate::fnv1a_hash(b"hello"));
}

#[test]
fn wal_entry_clone() {
    let entry = make_entry("test-doc", 0.5, 8);
    #[allow(clippy::redundant_clone)]
    let cloned = entry.clone();
    assert_eq!(cloned.doc_id, "test-doc");
    assert_eq!(cloned.embedding.len(), 8);
    assert!((cloned.embedding[0] - 0.5).abs() < f32::EPSILON);
}

#[test]
fn index_tagging_zero() {
    let tagged = to_wal_index(0);
    assert!(is_wal_index(tagged));
    assert_eq!(from_wal_index(tagged), 0);
}

#[test]
fn index_tagging_large_value() {
    let pos = (1_usize << (usize::BITS - 2)) - 1;
    let tagged = to_wal_index(pos);
    assert!(is_wal_index(tagged));
    assert_eq!(from_wal_index(tagged), pos);
}

#[test]
fn is_wal_index_false_for_normal_values() {
    assert!(!is_wal_index(0));
    assert!(!is_wal_index(1));
    assert!(!is_wal_index(usize::MAX >> 1));
}

#[test]
fn crc32_of_empty() {
    let crc = crc32_of(&[]);
    assert_eq!(crc, 0); // CRC32 of empty is 0
}

#[test]
fn crc32_of_known_data() {
    // Fixed IEEE CRC-32 values, independently checked with zlib, detect
    // polynomial/initialization drift that repeat-call equality cannot.
    assert_eq!(crc32_of(b""), 0);
    assert_eq!(crc32_of(b"hello"), 0x3610_a686);
    assert_eq!(crc32_of(b"123456789"), 0xcbf4_3926);
    let all_bytes: Vec<u8> = (u8::MIN..=u8::MAX).collect();
    assert_eq!(crc32_of(&all_bytes), 0x2905_8c73);
}

#[test]
fn oversized_doc_id_returns_error() {
    let path = temp_wal_path("oversized-docid");
    let dim = 4;
    // Create a doc_id that exceeds u16::MAX (65535) bytes.
    let big_id = "x".repeat(70_000);
    let entries = vec![WalEntry {
        doc_id: big_id,
        doc_id_hash: 0,
        embedding: vec![1.0; dim],
    }];
    let result = append_wal_batch(&path, &entries, dim, Quantization::F16, 0, false);
    assert!(result.is_err(), "should reject doc_id exceeding u16 length");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("doc_id_length"),
        "error should mention doc_id_length: {err}"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn oversized_batch_entry_count_returns_error() {
    let result = wal_entry_count(usize::MAX);
    assert!(result.is_err(), "should reject entry count exceeding u32");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("wal_batch_entries"),
        "error should mention wal_batch_entries: {err}"
    );
}

#[test]
fn oversized_dimension_returns_error() {
    let path = temp_wal_path("oversized-dimension");
    let result = append_wal_batch(&path, &[], usize::MAX, Quantization::F16, 0, false);
    assert!(result.is_err(), "should reject dimension exceeding u32");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("dimension"),
        "error should mention dimension: {err}"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn wal_corrupted_produces_index_corrupted_error() {
    let err = wal_corrupted(Path::new("/test/path.fsvi.wal"), "test detail");
    match err {
        SearchError::IndexCorrupted { path, detail } => {
            assert_eq!(path, Path::new("/test/path.fsvi.wal"));
            assert_eq!(detail, "test detail");
        }
        other => panic!("expected IndexCorrupted, got {other:?}"),
    }
}

#[test]
fn wal_path_for_various_inputs() {
    let wal = wal_path_for(Path::new("index.fsvi"));
    assert_eq!(wal, PathBuf::from("index.fsvi.wal"));

    let wal = wal_path_for(Path::new("/abs/path/data.fsvi"));
    assert_eq!(wal, PathBuf::from("/abs/path/data.fsvi.wal"));
}

#[test]
fn append_with_fsync_succeeds() {
    let path = temp_wal_path("fsync-test");
    let dim = 4;
    append_wal_batch(
        &path,
        &[make_entry("doc-sync", 1.0, dim)],
        dim,
        Quantization::F16,
        0,
        true,
    )
    .unwrap();

    let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].doc_id, "doc-sync");
    std::fs::remove_file(&path).ok();
}

#[test]
fn append_preserves_existing_header() {
    let path = temp_wal_path("preserve-header");
    let dim = 4;
    append_wal_batch(
        &path,
        &[make_entry("doc-0", 1.0, dim)],
        dim,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    let data_after_first = std::fs::read(&path).unwrap();
    let header_bytes = &data_after_first[..WAL_HEADER_SIZE];

    append_wal_batch(
        &path,
        &[make_entry("doc-1", 2.0, dim)],
        dim,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    let data_after_second = std::fs::read(&path).unwrap();
    // Header should be unchanged
    assert_eq!(&data_after_second[..WAL_HEADER_SIZE], header_bytes);
    // File should be larger
    assert!(data_after_second.len() > data_after_first.len());

    let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(loaded.len(), 2);
    std::fs::remove_file(&path).ok();
}

#[test]
fn bad_batch_magic_discards_batch() {
    let path = temp_wal_path("bad-batch-magic");
    let dim = 4;
    append_wal_batch(
        &path,
        &[make_entry("doc-good", 1.0, dim)],
        dim,
        Quantization::F16,
        0,
        false,
    )
    .unwrap();

    let mut data = std::fs::read(&path).unwrap();
    // Append a batch with wrong magic
    data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]); // bad batch magic
    data.extend_from_slice(&1_u32.to_le_bytes());
    std::fs::write(&path, &data).unwrap();

    let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].doc_id, "doc-good");
    std::fs::remove_file(&path).ok();
}

#[test]
fn f16_quantization_has_limited_precision() {
    let path = temp_wal_path("f16-precision");
    let dim = 2;
    let entries = vec![WalEntry {
        doc_id: "precise".into(),
        doc_id_hash: crate::fnv1a_hash(b"precise"),
        embedding: vec![0.123_456, -0.987_654],
    }];

    append_wal_batch(&path, &entries, dim, Quantization::F16, 0, false).unwrap();
    let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(loaded.len(), 1);
    // F16 has ~3 decimal digits of precision
    assert!((loaded[0].embedding[0] - 0.123_456).abs() < 0.001);
    assert!((loaded[0].embedding[1] + 0.987_654).abs() < 0.001);
    std::fs::remove_file(&path).ok();
}

#[test]
fn f32_quantization_exact_roundtrip() {
    let path = temp_wal_path("f32-exact");
    let dim = 3;
    let val = std::f32::consts::PI;
    let entries = vec![WalEntry {
        doc_id: "pi-doc".into(),
        doc_id_hash: crate::fnv1a_hash(b"pi-doc"),
        embedding: vec![val, -val, 0.0],
    }];

    append_wal_batch(&path, &entries, dim, Quantization::F32, 0, false).unwrap();
    let (loaded, _, _) = read_wal(&path, dim, Quantization::F32).unwrap();
    assert!((loaded[0].embedding[0] - val).abs() < f32::EPSILON);
    assert!((loaded[0].embedding[1] + val).abs() < f32::EPSILON);
    assert!(loaded[0].embedding[2].abs() < f32::EPSILON);
    std::fs::remove_file(&path).ok();
}

#[test]
fn append_to_preexisting_empty_file_writes_header() {
    let path = temp_wal_path("pre-empty");
    let dim = 3;

    // Create an empty file before calling append_wal_batch.
    // This simulates the race window the TOCTOU fix addresses:
    // another process might create the file between our existence
    // check and our open call.
    std::fs::File::create(&path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);

    let entries = vec![WalEntry {
        doc_id: "doc-0".into(),
        doc_id_hash: crate::fnv1a_hash(b"doc-0"),
        embedding: vec![1.0, 2.0, 3.0],
    }];
    append_wal_batch(&path, &entries, dim, Quantization::F16, 0, false).unwrap();

    let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].doc_id, "doc-0");
    std::fs::remove_file(&path).ok();
}

// ─── bd-vbzm tests end ───
