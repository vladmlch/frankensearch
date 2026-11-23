#![allow(clippy::wildcard_imports)]

use super::*;

// ─── Quantization edge cases ────────────────────────────────────────

#[test]
fn quantization_bytes_per_element() {
    assert_eq!(Quantization::F32.bytes_per_element(), 4);
    assert_eq!(Quantization::F16.bytes_per_element(), 2);
}

#[test]
fn quantization_from_wire_valid() {
    let path = Path::new("test.fsvi");
    assert_eq!(Quantization::from_wire(0, path).unwrap(), Quantization::F32);
    assert_eq!(Quantization::from_wire(1, path).unwrap(), Quantization::F16);
}

#[test]
fn quantization_from_wire_invalid() {
    let path = Path::new("test.fsvi");
    assert!(Quantization::from_wire(2, path).is_err());
    assert!(Quantization::from_wire(255, path).is_err());
}

// ─── align_up edge cases ────────────────────────────────────────────

#[test]
fn align_up_zero_alignment() {
    assert_eq!(align_up(42, 0).unwrap(), 42);
}

#[test]
fn align_up_already_aligned() {
    assert_eq!(align_up(128, 64).unwrap(), 128);
}

#[test]
fn align_up_zero_value() {
    assert_eq!(align_up(0, 64).unwrap(), 0);
}

#[test]
fn align_up_one_over() {
    assert_eq!(align_up(65, 64).unwrap(), 128);
}

// ─── fnv1a_hash edge cases ──────────────────────────────────────────

#[test]
fn fnv1a_hash_empty_input() {
    let hash = fnv1a_hash(b"");
    assert_eq!(hash, 0xcbf2_9ce4_8422_2325);
}

#[test]
fn fnv1a_hash_deterministic() {
    let h1 = fnv1a_hash(b"hello");
    let h2 = fnv1a_hash(b"hello");
    assert_eq!(h1, h2);
}

#[test]
fn fnv1a_hash_different_inputs_differ() {
    let h1 = fnv1a_hash(b"doc-a");
    let h2 = fnv1a_hash(b"doc-b");
    assert_ne!(h1, h2);
}

// ─── is_tombstoned_flags ────────────────────────────────────────────

#[test]
fn tombstone_flag_logic() {
    assert!(!is_tombstoned_flags(0x0000));
    assert!(is_tombstoned_flags(RECORD_FLAG_TOMBSTONE));
    assert!(is_tombstoned_flags(0x0003)); // tombstone + custom
    assert!(!is_tombstoned_flags(0x0002)); // only custom
}

// ─── validate_header_string ─────────────────────────────────────────

#[test]
fn validate_header_string_empty_embedder_id_rejected() {
    let result = validate_header_string("", "embedder_id");
    assert!(result.is_err());
}

#[test]
fn validate_header_string_empty_embedder_revision_ok() {
    let result = validate_header_string("", "embedder_revision");
    assert!(result.is_ok());
}

#[test]
fn validate_header_string_normal_ok() {
    let result = validate_header_string("potion-128M", "embedder_id");
    assert!(result.is_ok());
}

// ─── VectorMetadata clone/eq ────────────────────────────────────────

#[test]
fn vector_metadata_clone_eq() {
    let meta = VectorMetadata {
        fsvi_version: FSVI_VERSION,
        embedder_id: "test".to_owned(),
        embedder_revision: "v1".to_owned(),
        dimension: 256,
        quantization: Quantization::F16,
        compaction_gen: 0,
        publication_nonce: 0,
        record_count: 100,
        vectors_offset: 1024,
        identity_v2: None,
    };
    let cloned = meta.clone();
    assert_eq!(meta, cloned);
}

// ─── VectorIndex::create validation ─────────────────────────────────

#[test]
fn create_zero_dimension_rejected() {
    let path = temp_index_path("zero-dim");
    let result = VectorIndex::create(&path, "test", 0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        SearchError::InvalidConfig { .. }
    ));
}

#[test]
fn create_empty_embedder_id_rejected() {
    let path = temp_index_path("empty-embedder");
    let result = VectorIndex::create(&path, "", 4);
    assert!(result.is_err());
}

#[test]
fn create_with_revision_empty_revision_ok() {
    let path = temp_index_path("empty-rev");
    let writer =
        VectorIndex::create_with_revision(&path, "test", "", 4, Quantization::F16).unwrap();
    writer.finish().unwrap();
    let index = VectorIndex::open(&path).unwrap();
    assert_eq!(index.embedder_revision(), "");
    std::fs::remove_file(&path).ok();
}

// ─── VectorIndexWriter rejection cases ──────────────────────────────

#[test]
fn write_record_nan_embedding_rejected() {
    let path = temp_index_path("nan-embed");
    let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
    let result = writer.write_record("doc", &[1.0, f32::NAN, 0.0]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("non-finite"),
        "expected non-finite error, got: {err}"
    );
}

#[test]
fn write_record_inf_embedding_rejected() {
    let path = temp_index_path("inf-embed");
    let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
    let result = writer.write_record("doc", &[1.0, f32::INFINITY, 0.0]);
    assert!(result.is_err());
}

#[test]
fn write_record_zero_norm_embedding_rejected() {
    // bd-tqhc: an all-zero vector can never match any query, so writers
    // reject it instead of planting a permanently unusable record.
    let path = temp_index_path("zero-norm-embed");
    let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
    let result = writer.write_record("doc", &[0.0, 0.0, 0.0]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("non-zero"),
        "expected zero-norm rejection, got: {err}"
    );
}

#[test]
fn append_zero_norm_embedding_rejected() {
    let path = temp_index_path("zero-norm-append");
    let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
    writer.write_record("doc-live", &[1.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let result = index.append("doc-zero", &[0.0, 0.0, 0.0]);
    assert!(result.is_err());
    // The rejection must not have disturbed the live record.
    assert_eq!(index.record_count(), 1);
    assert_eq!(index.wal_record_count(), 0);
}

#[test]
fn zero_signal_census_counts_are_consistent() {
    let path = temp_index_path("zero-signal-census");
    let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
    writer.write_record("doc-a", &[1.0, 0.0, 0.0]).unwrap();
    writer.write_record("doc-b", &[0.0, 1.0, 0.0]).unwrap();
    writer.write_record("doc-c", &[0.0, 0.0, 1.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let healthy = index.zero_signal_state();
    assert_eq!(healthy.record_count, 3);
    assert_eq!(healthy.live_count, 3);
    assert_eq!(healthy.tombstone_count, 0);
    assert_eq!(healthy.wal_count, 0);
    assert_eq!(healthy.usable_vector_count, 3);
    assert_eq!(healthy.state_reason(), None);

    index.soft_delete("doc-b").unwrap();
    let after_delete = index.zero_signal_state();
    assert_eq!(after_delete.live_count, 2);
    assert_eq!(after_delete.tombstone_count, 1);
    assert_eq!(after_delete.usable_vector_count, 2);
    assert_eq!(index.live_count(), 2);
}

// ─── VectorIndex::open edge cases ───────────────────────────────────

#[test]
fn open_nonexistent_file_returns_index_not_found() {
    let path = temp_index_path("nonexistent-open");
    let result = VectorIndex::open(&path);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        SearchError::IndexNotFound { .. }
    ));
}

#[test]
fn open_truncated_file_detected() {
    let path = temp_index_path("truncated-open");
    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.write_record("doc-0", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let data = std::fs::read(&path).unwrap();
    std::fs::write(&path, &data[..data.len() - 4]).unwrap();

    let result = VectorIndex::open(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("truncated") || err.contains("too small") || err.contains("extends"),
        "expected truncation error, got: {err}"
    );

    std::fs::remove_file(&path).ok();
}

// ─── FSVI constants ─────────────────────────────────────────────────

#[test]
fn fsvi_magic_is_four_bytes() {
    assert_eq!(FSVI_MAGIC.len(), 4);
    assert_eq!(&FSVI_MAGIC, b"FSVI");
}

#[test]
fn fsvi_version_is_one() {
    assert_eq!(FSVI_VERSION, 1);
}

#[test]
fn record_size_is_sixteen() {
    assert_eq!(RECORD_SIZE_BYTES, 16);
}

// ─── vector_at_f16 on f16 index ─────────────────────────────────────

fn assert_f16_destination_allocation_error(error: SearchError, dimension: usize) {
    assert!(matches!(
        error,
        SearchError::InvalidConfig {
            field,
            value,
            reason,
        } if field == "vectors.f16_destination"
            && value == dimension.to_string()
            && reason == "destination allocation failed"
    ));
}

#[test]
fn f16_destination_reservation_rejects_maximum_shape_without_oom() {
    let mut destination = Vec::new();
    let error = reserve_f16_destination(&mut destination, usize::MAX)
        .expect_err("a maximum f16 vector shape must fail before allocation");

    assert_f16_destination_allocation_error(error, usize::MAX);
    assert!(
        destination.is_empty(),
        "a failed reservation must not append"
    );
}

#[test]
fn vector_at_f16_roundtrip() {
    let path = temp_index_path("f16-at-roundtrip");
    let mut writer =
        VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F16).unwrap();
    writer.write_record("doc", &[0.5, -0.5, 1.0]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    fail_next_f16_destination_reservation();
    let allocation_error = index
        .vector_at_f16(0)
        .expect_err("f16 decode allocation failure must be typed");
    assert_f16_destination_allocation_error(allocation_error, 3);
    assert!(
        !f16_destination_reservation_failure_is_pending(),
        "the f16 decode branch must consume the reservation failure"
    );
    let f16_vec = index.vector_at_f16(0).unwrap();
    assert_eq!(f16_vec.len(), 3);
    assert!((f16_vec[0].to_f32() - 0.5).abs() < 0.01);
    assert!((f16_vec[1].to_f32() - (-0.5)).abs() < 0.01);
    assert!((f16_vec[2].to_f32() - 1.0).abs() < 0.01);

    let mut direct = Vec::new();
    index.extend_vector_at_f16(0, &mut direct).unwrap();
    assert_eq!(direct, f16_vec);

    std::fs::remove_file(&path).ok();
}

// ─── vector_at_f16 on f32 index (converts) ─────────────────────────

#[test]
fn vector_at_f16_from_f32_index() {
    let path = temp_index_path("f16-from-f32");
    let mut writer =
        VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F32).unwrap();
    writer.write_record("doc", &[0.25, -0.75, 1.0]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    fail_next_f16_destination_reservation();
    let allocation_error = index
        .vector_at_f16(0)
        .expect_err("f32 decode allocation failure must be typed");
    assert_f16_destination_allocation_error(allocation_error, 3);
    assert!(
        !f16_destination_reservation_failure_is_pending(),
        "the f32 decode branch must consume the reservation failure"
    );
    let f16_vec = index.vector_at_f16(0).unwrap();
    assert_eq!(f16_vec.len(), 3);
    assert!((f16_vec[0].to_f32() - 0.25).abs() < 0.01);

    let mut direct = Vec::new();
    index.extend_vector_at_f16(0, &mut direct).unwrap();
    assert_eq!(direct, f16_vec);

    std::fs::remove_file(&path).ok();
}

// ─── metadata accessor ──────────────────────────────────────────────

#[test]
fn metadata_accessor_returns_consistent_data() {
    let path = temp_index_path("metadata-accessor");
    let mut writer =
        VectorIndex::create_with_revision(&path, "emb-1", "rev-9", 16, Quantization::F32).unwrap();
    writer.write_record("d", &[0.5; 16]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let meta = index.metadata();
    assert_eq!(meta.embedder_id, "emb-1");
    assert_eq!(meta.embedder_revision, "rev-9");
    assert_eq!(meta.dimension, 16);
    assert_eq!(meta.quantization, Quantization::F32);
    assert_eq!(meta.record_count, 1);
    assert_eq!(meta.vectors_offset % 64, 0);

    std::fs::remove_file(&path).ok();
}

// ─── is_deleted accessor ────────────────────────────────────────────

#[test]
fn is_deleted_false_for_live_record() {
    let path = temp_index_path("is-deleted-live");
    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    assert!(!index.is_deleted(0));

    std::fs::remove_file(&path).ok();
}

// ─── tombstone_ratio empty index ────────────────────────────────────

#[test]
fn tombstone_ratio_empty_index_is_zero() {
    let path = temp_index_path("tomb-ratio-empty");
    let writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    assert!(index.tombstone_ratio().abs() < f64::EPSILON);
    assert!(!index.needs_vacuum());

    std::fs::remove_file(&path).ok();
}

// ─── WalConfig default ──────────────────────────────────────────────

#[test]
fn wal_config_default_values() {
    let cfg = WalConfig::default();
    assert!(cfg.compaction_threshold > 0);
    assert!(cfg.compaction_ratio > 0.0);
}

// ─── F32 roundtrip with explicit revision ───────────────────────────

#[test]
fn f32_roundtrip_with_revision() {
    let path = temp_index_path("f32-rev-roundtrip");
    let original = vec![std::f32::consts::PI, std::f32::consts::E, 0.0, -1.0];
    let mut writer =
        VectorIndex::create_with_revision(&path, "f32-emb", "rev-42", 4, Quantization::F32)
            .unwrap();
    writer.write_record("doc", &original).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let recovered = index.vector_at_f32(0).unwrap();
    assert_eq!(recovered, original, "f32 must roundtrip exactly");
    assert_eq!(index.embedder_revision(), "rev-42");

    std::fs::remove_file(&path).ok();
}

// ─── Header CRC corruption by flipping data byte ────────────────────

#[test]
fn header_crc_detects_embedder_id_corruption() {
    let path = temp_index_path("crc-embedder-corrupt");
    let mut writer = VectorIndex::create(&path, "test-embedder-long", 4).unwrap();
    writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let mut data = std::fs::read(&path).unwrap();
    // Flip a byte in the embedder_id region (after magic+version+id_len = 8 bytes)
    data[10] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    let result = VectorIndex::open(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("CRC") || err.contains("crc"),
        "expected CRC error, got: {err}"
    );

    std::fs::remove_file(&path).ok();
}

// ─── bd-1fh4 tests begin ──────────────────────────────────────────

#[test]
fn vacuum_stats_debug_clone_partial_eq() {
    let stats = VacuumStats {
        records_before: 10,
        records_after: 8,
        tombstones_removed: 2,
        bytes_reclaimed: 1024,
        duration: Duration::from_millis(5),
    };
    let debug = format!("{stats:?}");
    assert!(debug.contains("VacuumStats"));
    assert!(debug.contains("records_before: 10"));

    let cloned = stats.clone();
    assert_eq!(stats, cloned);
}

#[test]
fn quantization_debug_clone_copy_eq() {
    let f16 = Quantization::F16;
    let f32q = Quantization::F32;

    let debug_f16 = format!("{f16:?}");
    assert!(debug_f16.contains("F16"));
    let debug_f32 = format!("{f32q:?}");
    assert!(debug_f32.contains("F32"));

    let f16_copy = f16;
    assert_eq!(f16, f16_copy);
    let f32_copy = f32q;
    assert_eq!(f32q, f32_copy);
    assert_ne!(f16, f32q);
}

#[test]
fn vector_index_debug_includes_path() {
    let path = temp_index_path("debug-fmt");
    let writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let debug = format!("{index:?}");
    assert!(debug.contains("VectorIndex"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn set_wal_config_overrides_defaults() {
    let path = temp_index_path("wal-cfg-override");
    let dim = 4;
    let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
    for i in 0..100 {
        writer
            .write_record(&format!("d{i}"), &sample_vector(0.1, dim))
            .unwrap();
    }
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    // With 100 main records and default config, 1 WAL entry should not trigger.
    index.append("wal-1", &sample_vector(0.5, dim)).unwrap();
    assert!(!index.needs_compaction());

    // Set a low threshold to trigger compaction.
    index.set_wal_config(WalConfig {
        compaction_threshold: 1,
        compaction_ratio: 0.001,
        fsync_on_write: false,
    });
    assert!(index.needs_compaction());

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn find_index_by_doc_hash_empty_index_none() {
    let path = temp_index_path("hash-empty");
    let writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    assert!(index.find_index_by_doc_hash(0xDEAD_BEEF).is_none());
    assert!(index.find_index_by_doc_hash(0).is_none());

    std::fs::remove_file(&path).ok();
}

#[test]
fn get_embeddings_mixed_hit_miss() {
    let path = temp_index_path("emb-mixed");
    let mut writer =
        VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F16).unwrap();
    writer.write_record("alpha", &[1.0, 0.0, 0.0]).unwrap();
    writer.write_record("beta", &[0.0, 1.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let index = VectorIndex::open(&path).unwrap();
    let alpha_hash = fnv1a_hash(b"alpha");
    let beta_hash = fnv1a_hash(b"beta");
    let missing_hash = fnv1a_hash(b"gamma");

    let results = index.get_embeddings(&[alpha_hash, missing_hash, beta_hash]);
    assert_eq!(results.len(), 3);
    assert!(results[0].is_some(), "alpha should be found");
    assert!(results[1].is_none(), "gamma should be missing");
    assert!(results[2].is_some(), "beta should be found");

    std::fs::remove_file(&path).ok();
}

#[test]
fn append_batch_empty_is_noop() {
    let path = temp_index_path("append-empty-batch");
    let writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    index.append_batch(&[]).unwrap();
    assert_eq!(index.wal_record_count(), 0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn append_nan_embedding_rejected() {
    let path = temp_index_path("append-nan");
    let writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let result = index.append("doc", &[1.0, f32::NAN, 0.0, 0.0]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("finite"), "expected finite error, got: {err}");
}

#[test]
fn append_inf_embedding_rejected() {
    let path = temp_index_path("append-inf");
    let writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let result = index.append("doc", &[1.0, 0.0, f32::INFINITY, 0.0]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("finite"), "expected finite error, got: {err}");
}

#[test]
fn soft_delete_already_deleted_returns_false() {
    let path = temp_index_path("double-delete");
    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    assert!(index.soft_delete("doc").unwrap(), "first delete");
    assert!(!index.soft_delete("doc").unwrap(), "second delete");
    assert!(!index.soft_delete("doc").unwrap(), "third delete");

    std::fs::remove_file(&path).ok();
}

#[test]
fn published_mapping_refuses_mutation_without_the_writer_lock() {
    let path = temp_index_path("reader-refuses-published-mutation");
    let mut writer = VectorIndex::create(&path, "test", 4).expect("create fixture");
    writer
        .write_record("doc", &[1.0, 0.0, 0.0, 0.0])
        .expect("write fixture row");
    writer.finish().expect("publish fixture");

    let mut reader = VectorIndex::open_read_only(&path).expect("shared reader opens");
    let error = reader
        .soft_delete("doc")
        .expect_err("shared mapping must never mutate a published FSVI");
    assert!(matches!(
        error,
        SearchError::InvalidConfig { field, value, .. }
            if field == "fsvi.writer_lock" && value == "soft_delete_batch"
    ));
    assert!(
        !reader.is_deleted(0),
        "the rejected mutation must leave the reader's mapped bytes unchanged"
    );

    let contended = VectorIndex::open_writer(&path)
        .expect_err("a live shared mapping must prevent a writable map");
    assert!(matches!(
        contended,
        SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
    ));

    drop(reader);
    let mut locked_writer =
        VectorIndex::open(&path).expect("writer opens after shared reader drops");
    assert!(
        locked_writer
            .soft_delete("doc")
            .expect("writer lock permits mutation"),
        "exclusive writer mutates only after acquiring the artifact lock"
    );
}

/// Real duplicated descriptors exercise open-description lock lifetime;
/// this does not claim to execute fork or use a mapping in a child.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn map_lock_shared_release_with_duplicate_still_excludes_a_live_reader() {
    let path = temp_index_path("map-lock-shared-retained-duplicate");
    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer.write_record("kept", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let first = VectorIndex::open_read_only(&path).unwrap();
    let second = VectorIndex::open_read_only(&path).unwrap();
    let retained = match first.map_lock.as_ref().unwrap() {
        FsviMapLock::Shared { file, .. } => file.try_clone().unwrap(),
        FsviMapLock::Exclusive { .. } => panic!("reader acquired a writer lock"),
    };
    let refused = VectorIndex::open_writer(&path).unwrap_err();
    assert!(matches!(
        refused,
        SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
    ));
    drop(first);
    let refused = VectorIndex::open_writer(&path).unwrap_err();
    assert!(matches!(
        refused,
        SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
    ));
    let hits = second
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "kept");

    drop(second);
    let mut writer = VectorIndex::open_writer(&path)
        .expect("last real reader releases despite the first reader's retained descriptor");
    assert!(retained.metadata().unwrap().len() > 0);
    writer.append("added", &[0.0, 1.0, 0.0, 0.0]).unwrap();
    drop(writer);
    let reader = VectorIndex::open_read_only(&path).unwrap();
    let mut ids = reader
        .search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None)
        .unwrap()
        .into_iter()
        .map(|hit| hit.doc_id)
        .collect::<Vec<_>>();
    ids.sort();
    assert_eq!(ids, vec!["added", "kept"]);
    assert!(retained.metadata().unwrap().len() > 0);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn map_lock_writer_release_with_duplicate_preserves_tombstone_and_wal() {
    let path = temp_index_path("map-lock-writer-retained-duplicate");
    let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
    writer
        .write_record("deleted", &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    writer.write_record("kept", &[0.0, 1.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let mut writer = VectorIndex::open_writer(&path).unwrap();
    let retained = match writer.map_lock.as_ref().unwrap() {
        FsviMapLock::Exclusive { file, .. } => file.try_clone().unwrap(),
        FsviMapLock::Shared { .. } => panic!("writer acquired a reader lock"),
    };
    let refused = VectorIndex::open_read_only(&path).unwrap_err();
    assert!(matches!(
        refused,
        SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
    ));
    let refused = VectorIndex::open_writer(&path).unwrap_err();
    assert!(matches!(
        refused,
        SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
    ));
    writer.append("added", &[0.0, 0.0, 1.0, 0.0]).unwrap();
    assert!(writer.soft_delete("deleted").unwrap());
    drop(writer);

    let reader = VectorIndex::open_read_only(&path)
        .expect("completed writer releases while its descriptor duplicate remains open");
    assert_eq!(reader.tombstone_count(), 1);
    assert_eq!(reader.wal_record_count(), 1);
    let mut ids = reader
        .search_top_k(&[1.0, 1.0, 1.0, 0.0], 10, None)
        .unwrap()
        .into_iter()
        .map(|hit| hit.doc_id)
        .collect::<Vec<_>>();
    ids.sort();
    assert_eq!(ids, vec!["added", "kept"]);
    assert_eq!(
        retained.metadata().unwrap().len(),
        fs::metadata(&path).unwrap().len()
    );
}

/// Simulate the noncreator PID branch with a genuine retained descriptor.
/// The creator's live lock must survive either inherited guard variant.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn map_lock_noncreator_drop_never_unlocks_the_creators_description() {
    for exclusive in [false, true] {
        let path = temp_index_path(if exclusive {
            "map-lock-noncreator-writer"
        } else {
            "map-lock-noncreator-reader"
        });
        let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.write_record("kept", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();
        let mut child_copy = VectorIndex::open_with_lock(&path, exclusive).unwrap();
        let (file, creator_pid) = match child_copy.map_lock.as_mut().unwrap() {
            FsviMapLock::Shared {
                file, creator_pid, ..
            }
            | FsviMapLock::Exclusive {
                file, creator_pid, ..
            } => (file, creator_pid),
        };
        let creator = file.try_clone().unwrap();
        *creator_pid = std::process::id().wrapping_add(1);
        drop(child_copy);

        let refused = VectorIndex::open_writer(&path).unwrap_err();
        assert!(matches!(
            refused,
            SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
        ));
        if exclusive {
            let refused = VectorIndex::open_read_only(&path).unwrap_err();
            assert!(matches!(
                refused,
                SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
            ));
        }
        creator
            .unlock()
            .expect("only creator releases its retained lock");
        let writer = VectorIndex::open_writer(&path).unwrap();
        let hits = writer
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "kept");
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn map_lock_rewrite_keeps_the_published_inode_exclusive_until_owner_drop() {
    use std::os::unix::fs::MetadataExt as _;

    for compact in [false, true] {
        let path = temp_index_path(if compact {
            "map-lock-reload-compaction"
        } else {
            "map-lock-reload-vacuum"
        });
        let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer
            .write_record("deleted", &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        writer.write_record("kept", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();
        let mut writer = VectorIndex::open_writer(&path).unwrap();
        let old_inode = match writer.map_lock.as_ref().unwrap() {
            FsviMapLock::Exclusive { file, .. } => file.try_clone().unwrap(),
            FsviMapLock::Shared { .. } => panic!("writer acquired a reader lock"),
        };
        writer.append("added", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert!(writer.soft_delete("deleted").unwrap());
        if compact {
            writer.compact().unwrap();
            assert_eq!(writer.record_count(), 2);
            assert_eq!(writer.wal_record_count(), 0);
        } else {
            writer.vacuum().unwrap();
            assert_eq!(writer.record_count(), 1);
            assert_eq!(writer.wal_record_count(), 1);
        }
        assert_eq!(writer.tombstone_count(), 0);
        let new_inode = match writer.map_lock.as_ref().unwrap() {
            FsviMapLock::Exclusive { file, .. } => file.try_clone().unwrap(),
            FsviMapLock::Shared { .. } => panic!("reload lost its writer lock"),
        };
        let old_metadata = old_inode.metadata().unwrap();
        let new_metadata = new_inode.metadata().unwrap();
        let published_metadata = fs::metadata(&path).unwrap();
        assert_ne!(
            (old_metadata.dev(), old_metadata.ino()),
            (new_metadata.dev(), new_metadata.ino()),
            "rewrite must replace the old inode"
        );
        assert_eq!(
            (new_metadata.dev(), new_metadata.ino()),
            (published_metadata.dev(), published_metadata.ino()),
            "the retained lock must belong to the mapped published inode"
        );
        let refused = VectorIndex::open_read_only(&path).unwrap_err();
        assert!(matches!(
            refused,
            SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
        ));
        let refused = VectorIndex::open_writer(&path).unwrap_err();
        assert!(matches!(
            refused,
            SearchError::InvalidConfig { field, .. } if field == "fsvi.map_lock"
        ));

        drop(writer);
        let reader = VectorIndex::open_read_only(&path)
            .expect("reloaded owner releases despite retained old and new descriptors");
        let mut ids = reader
            .search_top_k(&[1.0, 1.0, 1.0, 0.0], 10, None)
            .unwrap()
            .into_iter()
            .map(|hit| hit.doc_id)
            .collect::<Vec<_>>();
        ids.sort();
        assert_eq!(ids, vec!["added", "kept"]);
        assert!(old_inode.metadata().unwrap().len() > 0);
        assert!(new_inode.metadata().unwrap().len() > 0);
    }
}

#[test]
fn compact_preserves_wal_config() {
    let path = temp_index_path("compact-cfg");
    let dim = 4;
    let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
    for i in 0..20 {
        writer
            .write_record(&format!("d{i}"), &sample_vector(0.1, dim))
            .unwrap();
    }
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    let custom = WalConfig {
        compaction_threshold: 99,
        compaction_ratio: 0.90,
        fsync_on_write: false,
    };
    index.set_wal_config(custom);
    index.append("wal-1", &sample_vector(0.5, dim)).unwrap();
    index.compact().unwrap();

    // After compaction, the custom config should be preserved.
    assert_eq!(index.wal_record_count(), 0);
    // Verify config persists: threshold=99 and ratio=0.90,
    // with 21 main records, 1 WAL entry → ratio ~0.048 < 0.90.
    index.append("wal-2", &sample_vector(0.3, dim)).unwrap();
    assert!(!index.needs_compaction());

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

#[test]
fn soft_delete_wal_restores_state_on_rewrite_failure() {
    let path = temp_index_path("wal-delete-restore");
    let dim = 4;

    let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
    writer
        .write_record("main-0", &sample_vector(1.0, dim))
        .unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    index.append("wal-a", &[0.0, 1.0, 0.0, 0.0]).unwrap();
    index.append("wal-b", &[0.0, 0.0, 1.0, 0.0]).unwrap();
    assert_eq!(index.wal_record_count(), 2);

    // Make the WAL parent directory read-only to force a rewrite failure.
    let wal_file = wal::wal_path_for(&path);
    let wal_dir = wal_file.parent().unwrap();
    let original_perms = fs::metadata(wal_dir).unwrap().permissions();
    let mut readonly = original_perms.clone();
    readonly.set_readonly(true);
    if fs::set_permissions(wal_dir, readonly).is_err() {
        // Sandboxed environments may not allow permission changes; skip.
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
        return;
    }

    let result = index.soft_delete("wal-a");

    // Restore directory permissions before any assertions so cleanup works.
    fs::set_permissions(wal_dir, original_perms).unwrap();

    // Precondition check: a read-only directory only blocks writes for an
    // unprivileged process. On root / `CAP_DAC_OVERRIDE` environments (some CI
    // and rch remote workers run as root), the rewrite is NOT blocked and
    // `soft_delete` succeeds — the forced-failure scenario this test exercises
    // cannot occur there. Skip gracefully instead of asserting a failure that
    // is physically impossible on that worker (the source of intermittent,
    // worker-dependent flakiness in this test).
    if result.is_ok() {
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
        return;
    }

    // The delete should have failed.
    assert!(result.is_err(), "expected error from read-only directory");

    // In-memory WAL entries must be fully restored.
    assert_eq!(
        index.wal_record_count(),
        2,
        "WAL entries should be restored after rewrite failure"
    );

    // Both entries should still be searchable.
    let hits = index.search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None).unwrap();
    assert!(hits.iter().any(|h| h.doc_id == "wal-a"));
    assert!(hits.iter().any(|h| h.doc_id == "wal-b"));

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(wal::wal_path_for(&path)).ok();
}

// ─── Regression: Duplicate entries on compaction crash ──────────────

#[test]
fn repro_duplicate_entries_on_compaction_crash() {
    let path = temp_index_path("compaction-crash");
    let dim = 4;

    // 1. Create initial index with 1 document
    let mut writer =
        VectorIndex::create_with_revision(&path, "test", "v1", dim, Quantization::F16).unwrap();
    writer.write_record("doc-A", &[1.0, 0.0, 0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();

    // 2. Append a document to WAL
    index.append("doc-B", &[0.0, 1.0, 0.0, 0.0]).unwrap();

    // Check state before "compaction"
    let hits = index.search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None).unwrap();
    assert_eq!(hits.len(), 2);

    // 3. Simulate compaction crash:
    // We want to create a state where "doc-B" is in Main Index AND in WAL.
    // We can do this by running `compact` but preventing the WAL deletion.
    // Since we can't easily interrupt `compact`, we'll simulate the filesystem state.

    // Close index to flush everything
    drop(index);

    // Manually create the "post-compaction" main index that includes both
    // A and B. It is built at a side path and renamed over `path` exactly
    // the way compact() publishes — writer::finish() itself now refuses to
    // publish beside a live WAL sidecar (the bd-cnby1 guard), so the
    // crash state must be constructed at the filesystem level, which is
    // also where the real crash window lives (between compact()'s rename
    // and its WAL removal).
    let staged_path = path.with_extension("compacted");
    let mut compact_writer =
        VectorIndex::create_with_revision(&staged_path, "test", "v1", dim, Quantization::F16)
            .unwrap()
            .with_generation(2); // Simulate correct compaction increment
    compact_writer
        .write_record("doc-A", &[1.0, 0.0, 0.0, 0.0])
        .unwrap();
    compact_writer
        .write_record("doc-B", &[0.0, 1.0, 0.0, 0.0])
        .unwrap();
    compact_writer.finish().unwrap();
    fs::rename(&staged_path, &path).unwrap(); // Publish over `path`; the old WAL survives untouched.

    // Restore the WAL file (because `finish` doesn't touch it, but we need to ensure it exists and has doc-B)
    // Actually, `finish` overwrites `path`. The WAL file is at `path.wal`.
    // We didn't delete `path.wal`. So `path.wal` still contains "doc-B".

    // 4. Re-open index. It should load Main (A, B) and WAL (B).
    let index_reopened = VectorIndex::open(&path).unwrap();

    // 5. Search. If bug exists, we'll see "doc-B" twice.
    let hits = index_reopened
        .search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None)
        .unwrap();

    // Debug output
    for hit in &hits {
        println!("Hit: {} score={}", hit.doc_id, hit.score);
    }

    // Clean up
    let _ = fs::remove_file(&path);
    let _ = wal::remove_wal(&wal::wal_path_for(&path));

    // Assert failure
    let hit_count = hits.len();
    assert_eq!(
        hit_count, 2,
        "Should have exactly 2 hits (A and B), found {hit_count}"
    );
    let b_count = hits.iter().filter(|h| h.doc_id == "doc-B").count();
    assert_eq!(b_count, 1, "Should have exactly 1 'doc-B', found {b_count}");
}

// ─── bd-1fh4 tests end ────────────────────────────────────────────
