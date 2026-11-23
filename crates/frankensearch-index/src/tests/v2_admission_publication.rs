#![allow(clippy::wildcard_imports)]

use super::*;

#[test]
fn fsvi_v2_inspection_distinguishes_reindex_upgrade_and_corruption() {
    let legacy_path = temp_index_path("v2-inspect-legacy");
    VectorIndex::create(&legacy_path, "legacy-display-name", 4)
        .expect("legacy writer")
        .finish()
        .expect("legacy finish");
    assert!(matches!(
        VectorIndex::inspect(&legacy_path),
        Ok(FsviInspection::ReindexRequired(FsviReindexRequired {
            reason: FsviReindexReason::LegacyUnidentified,
            found_version: FSVI_VERSION,
            ..
        }))
    ));
    let expected = fsvi_v2_binding("v2-inspect", 4, Quantization::F16, 1, 0x22);
    assert!(matches!(
        admit_owned_v2_fixture(&legacy_path, &expected),
        Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
            reason: FsviReindexReason::LegacyUnidentified,
            ..
        }))
    ));

    let future_path = temp_index_path("v2-inspect-future");
    let mut future = Vec::from(FSVI_MAGIC);
    future.extend_from_slice(&(FSVI_V2_VERSION + 1).to_le_bytes());
    fs::write(&future_path, future).expect("write future prefix");
    assert!(matches!(
        VectorIndex::inspect(&future_path),
        Ok(FsviInspection::UpgradeRequired(FsviUpgradeRequired {
            found_version,
            supported_version: FSVI_V2_VERSION,
        })) if found_version == FSVI_V2_VERSION + 1
    ));
    assert!(matches!(
        admit_owned_v2_fixture(&future_path, &expected),
        Err(FsviAdmissionError::UpgradeRequired(FsviUpgradeRequired {
            found_version,
            ..
        })) if found_version == FSVI_V2_VERSION + 1
    ));

    let corrupt_path = temp_index_path("v2-inspect-corrupt");
    fs::write(&corrupt_path, b"FSV").expect("write truncated prefix");
    assert_inspection_corrupted(&corrupt_path);
}

#[test]
fn fsvi_v2_admission_requires_exact_generation_storage_and_identity() {
    let binding = fsvi_v2_binding("v2-exact", 4, Quantization::F16, 7, 0x33);
    let (path, expected) = write_v2_fixture("v2-exact", binding);

    let wrong_generation = fsvi_v2_binding("v2-exact", 4, Quantization::F16, 8, 0x34);
    assert!(matches!(
        admit_owned_v2_fixture(&path, &wrong_generation),
        Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
            reason: FsviReindexReason::GenerationMismatch,
            ..
        }))
    ));

    let wrong_storage = fsvi_v2_binding("v2-exact", 4, Quantization::F32, 7, 0x33);
    assert!(matches!(
        admit_owned_v2_fixture(&path, &wrong_storage),
        Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
            reason: FsviReindexReason::StorageMismatch,
            ..
        }))
    ));

    let wrong_identity = fsvi_v2_binding(
        "same-dimension-different-space",
        4,
        Quantization::F16,
        7,
        0x33,
    );
    assert!(matches!(
        admit_owned_v2_fixture(&path, &wrong_identity),
        Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
            reason: FsviReindexReason::IdentityMismatch,
            ..
        }))
    ));

    admit_owned_v2_fixture(&path, &expected).expect("control admission");
}

#[test]
fn fsvi_v2_header_mutation_matrix_fails_closed() {
    let binding = fsvi_v2_binding("v2-header-matrix", 4, Quantization::F16, 9, 0x44);
    let (source_path, _) = write_v2_fixture("v2-header-matrix-source", binding);
    let source = fs::read(&source_path).expect("read source v2");
    let header_size = usize::try_from(u32::from_le_bytes(
        source[6..10].try_into().expect("header size"),
    ))
    .expect("header size fits");
    assert!(header_size > FSVI_V2_FIXED_PREFIX_BYTES + 4);

    let fixed_mutations = [
        ("binding-schema", 10usize),
        ("quantization", 12),
        ("header-flags", 13),
        ("dimension", 16),
        ("generation-schema", 36),
        ("generation-reserved", 38),
        ("generation-sequence", 40),
        ("generation-nonce", 48),
        ("bundle-fingerprint", FSVI_V2_BUNDLE_FINGERPRINT_OFFSET),
        ("space-fingerprint", FSVI_V2_SPACE_FINGERPRINT_OFFSET),
        ("producer-fingerprint", FSVI_V2_PRODUCER_FINGERPRINT_OFFSET),
        ("input-fingerprint", FSVI_V2_INPUT_FINGERPRINT_OFFSET),
        ("storage-fingerprint", FSVI_V2_STORAGE_FINGERPRINT_OFFSET),
        (
            "generation-fingerprint",
            FSVI_V2_GENERATION_FINGERPRINT_OFFSET,
        ),
    ];
    for (name, offset) in fixed_mutations {
        let path = temp_index_path(&format!("v2-header-{name}"));
        let mut mutated = source.clone();
        mutated[offset] ^= 0x01;
        refresh_v2_header_crc(&mut mutated);
        fs::write(&path, mutated).expect("write header mutation");
        assert_inspection_corrupted(&path);
    }

    let canonical_path = temp_index_path("v2-header-canonical");
    let mut canonical_mutation = source;
    canonical_mutation[FSVI_V2_FIXED_PREFIX_BYTES] ^= 0x01;
    refresh_v2_header_crc(&mut canonical_mutation);
    fs::write(&canonical_path, canonical_mutation).expect("write canonical mutation");
    assert_inspection_corrupted(&canonical_path);
}

#[test]
fn fsvi_v2_magic_version_and_crc_mutations_fail_closed() {
    let binding = fsvi_v2_binding("v2-magic-version-crc", 4, Quantization::F16, 55, 0xaf);
    let (source_path, expected) = write_v2_fixture("v2-magic-version-crc-source", binding);
    let source = fs::read(&source_path).expect("read source v2");
    let header_size = usize::try_from(u32::from_le_bytes(
        source[6..10].try_into().expect("header size"),
    ))
    .expect("header size fits");

    let magic_path = temp_index_path("v2-magic-corruption");
    let mut magic = source.clone();
    magic[0] ^= 0x01;
    fs::write(&magic_path, magic).expect("write magic mutation");
    assert_owned_admission_corrupted(&magic_path, &expected);

    let version_path = temp_index_path("v2-version-upgrade");
    let mut version = source.clone();
    version[4..6].copy_from_slice(&(FSVI_V2_VERSION + 1).to_le_bytes());
    fs::write(&version_path, version).expect("write future-version mutation");
    assert!(matches!(
        admit_owned_v2_fixture(&version_path, &expected),
        Err(FsviAdmissionError::UpgradeRequired(FsviUpgradeRequired {
            found_version,
            supported_version: FSVI_V2_VERSION,
        })) if found_version == FSVI_V2_VERSION + 1
    ));

    let crc_path = temp_index_path("v2-header-crc-corruption");
    let mut crc = source;
    crc[header_size - 1] ^= 0x01;
    fs::write(&crc_path, crc).expect("write header CRC mutation");
    assert_owned_admission_corrupted(&crc_path, &expected);
}

#[test]
fn fsvi_v2_complete_space_parser_rejects_suffix_and_dimension_drift() {
    let binding = fsvi_v2_binding("v2-complete-space", 4, Quantization::F16, 9, 0x45);
    let path = temp_index_path("v2-complete-space");

    let mut suffixed = binding.space_canonical_bytes.clone();
    suffixed.push(0);
    assert!(matches!(
        parse_complete_space_identity(&path, &suffixed, 4, binding.input_fingerprint,),
        Err(SearchError::IndexCorrupted { .. })
    ));

    let mut wrong_dimension = binding.frozen_identity.identity.space.clone();
    wrong_dimension.dimension = 5;
    assert!(matches!(
        parse_complete_space_identity(
            &path,
            &wrong_dimension.canonical_bytes(),
            4,
            binding.input_fingerprint,
        ),
        Err(SearchError::IndexCorrupted { .. })
    ));
}

#[test]
fn fsvi_v2_complete_space_parser_round_trips_semantic_and_projection_branches() {
    let path = temp_index_path("v2-complete-space-semantic");
    let semantic = semantic_fsvi_v2_binding("semantic-space", 8, 9, 0x46);
    let parsed = parse_complete_space_identity(
        &path,
        &semantic.space_canonical_bytes,
        8,
        semantic.input_fingerprint,
    )
    .expect("complete semantic identity parses");
    assert_eq!(parsed.0, "semantic-space");
    assert_eq!(parsed.1, "explicit-test-v1");
    assert_eq!(
        parsed.2,
        semantic.frozen_identity.identity.space.output_normalization
    );

    let projected_identity = semantic
        .frozen_identity
        .identity
        .derive_projection(4, "first-four-components", "l2-after-projection")
        .expect("derive semantic projection");
    let projected = FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(10, [0x47; 16]).expect("valid projected generation"),
        projected_identity
            .freeze()
            .expect("freeze projected identity"),
    )
    .expect("valid projected FSVI v2 binding");
    let parsed = parse_complete_space_identity(
        &path,
        &projected.space_canonical_bytes,
        4,
        projected.input_fingerprint,
    )
    .expect("complete projected identity parses");
    assert_eq!(parsed.0, "semantic-space");
    assert_eq!(parsed.2, "l2-after-projection");
}

#[test]
fn fsvi_v2_content_mutation_matrix_fails_admission() {
    let binding = fsvi_v2_binding("v2-content-matrix", 4, Quantization::F16, 10, 0x55);
    let (source_path, expected) = write_v2_fixture("v2-content-matrix-source", binding);
    let source = fs::read(&source_path).expect("read source v2");
    let inspected = VectorIndex::inspect(&source_path).expect("inspect source");
    assert!(matches!(&inspected, FsviInspection::V2IdentityComplete(_)));
    let FsviInspection::V2IdentityComplete(metadata) = inspected else {
        return;
    };
    let identity = metadata.identity_v2.as_ref().expect("v2 metadata");
    let header_size = identity.header_size;
    let vectors_offset =
        usize::try_from(metadata.vectors_offset).expect("vector offset fits usize");
    let strings_offset = header_size + metadata.record_count * RECORD_SIZE_BYTES;
    let string_bytes = "doc-alpha".len() + "doc-beta".len();
    let padding_offset = strings_offset + string_bytes;
    assert!(
        padding_offset < vectors_offset,
        "fixture must include padding"
    );

    let content_mutations = [
        ("record-flags", header_size + 14),
        ("document-id", strings_offset),
        ("alignment-padding", padding_offset),
        ("vector-slab", vectors_offset),
    ];
    for (name, offset) in content_mutations {
        let path = temp_index_path(&format!("v2-content-{name}"));
        let mut mutated = source.clone();
        mutated[offset] ^= 0x01;
        fs::write(&path, mutated).expect("write content mutation");
        assert_owned_admission_corrupted(&path, &expected);
    }

    for (name, offset) in [
        ("docset-digest", FSVI_V2_DOCSET_DIGEST_OFFSET),
        ("vector-digest", FSVI_V2_VECTOR_DIGEST_OFFSET),
    ] {
        let path = temp_index_path(&format!("v2-content-{name}"));
        let mut mutated = source.clone();
        mutated[offset] ^= 0x01;
        refresh_v2_header_crc(&mut mutated);
        fs::write(&path, mutated).expect("write digest mutation");
        assert_owned_admission_corrupted(&path, &expected);
    }

    let trailing_path = temp_index_path("v2-content-trailing");
    let mut trailing = source.clone();
    trailing.push(0);
    fs::write(&trailing_path, trailing).expect("write trailing byte");
    assert_owned_admission_corrupted(&trailing_path, &expected);

    let truncated_path = temp_index_path("v2-content-truncated");
    fs::write(&truncated_path, &source[..source.len() - 1]).expect("write truncated vector");
    assert_owned_admission_corrupted(&truncated_path, &expected);

    let unaligned_path = temp_index_path("v2-content-unaligned-vector-slab");
    let mut unaligned = source.clone();
    let unaligned_vectors_offset = vectors_offset
        .checked_sub(1)
        .expect("fixture vector offset has alignment padding");
    assert!(
        unaligned_vectors_offset >= padding_offset,
        "fixture must retain a valid string table when one padding byte is removed"
    );
    unaligned.remove(unaligned_vectors_offset);
    unaligned[28..36].copy_from_slice(
        &u64::try_from(unaligned_vectors_offset)
            .expect("unaligned vector offset fits u64")
            .to_le_bytes(),
    );
    refresh_v2_header_crc(&mut unaligned);
    fs::write(&unaligned_path, unaligned).expect("write unaligned vector slab");
    assert_owned_admission_corrupted(&unaligned_path, &expected);

    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let wal_path = wal::wal_path_for(&source_path);
        fs::write(&wal_path, b"live-sidecar").expect("write live WAL sidecar");
        let sidecar_before = fs::read(&wal_path).expect("read sidecar before rejection");
        assert_snapshot_rejection(
            VectorIndex::open_admitted_v2(&source_path, &expected),
            FsviSnapshotRejectionReason::PublishedWalPresent,
        );
        assert_eq!(
            fs::read(&wal_path).expect("read preserved sidecar"),
            sidecar_before
        );
    }
}

#[test]
fn fsvi_v2_owner_rejects_invalid_utf8_and_resealed_duplicate_document_ids() {
    let binding = fsvi_v2_binding(
        "v2-record-identity-corruption",
        4,
        Quantization::F16,
        11,
        0x66,
    );
    let path = temp_index_path("v2-record-identity-corruption");
    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("create v2 fixture");
    writer
        .write_record("aa-owner", &[1.0, 0.0, 0.0, 0.0])
        .expect("write first equal-length id");
    writer
        .write_record("bb-owner", &[0.0, 1.0, 0.0, 0.0])
        .expect("write second equal-length id");
    writer.finish().expect("finish v2 fixture");

    let source = fs::read(&path).expect("read fixture bytes");
    let header_size = usize::try_from(u32::from_le_bytes(
        source[6..10].try_into().expect("header size"),
    ))
    .expect("header size fits");
    let strings_offset = header_size + (2 * RECORD_SIZE_BYTES);
    let first_record = header_size;
    let second_record = first_record + RECORD_SIZE_BYTES;
    let first_id_offset = usize::try_from(u32::from_le_bytes(
        source[first_record + 8..first_record + 12]
            .try_into()
            .expect("first id offset"),
    ))
    .expect("first id offset fits");
    let second_id_offset = usize::try_from(u32::from_le_bytes(
        source[second_record + 8..second_record + 12]
            .try_into()
            .expect("second id offset"),
    ))
    .expect("second id offset fits");
    let first_id_len = usize::from(u16::from_le_bytes(
        source[first_record + 12..first_record + 14]
            .try_into()
            .expect("first id length"),
    ));
    let second_id_len = usize::from(u16::from_le_bytes(
        source[second_record + 12..second_record + 14]
            .try_into()
            .expect("second id length"),
    ));
    assert_eq!(
        first_id_len, second_id_len,
        "fixture must make a duplicate-ID mutation preserve the string layout"
    );

    let mut invalid_utf8 = source.clone();
    invalid_utf8[strings_offset + first_id_offset] = 0xff;
    assert!(matches!(
        ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(invalid_utf8), &binding),
        Err(FsviAdmissionError::Index(
            SearchError::IndexCorrupted { .. }
        ))
    ));

    let mut duplicate = source;
    let first_id = duplicate
        [strings_offset + first_id_offset..strings_offset + first_id_offset + first_id_len]
        .to_vec();
    duplicate[strings_offset + second_id_offset..strings_offset + second_id_offset + second_id_len]
        .copy_from_slice(&first_id);
    duplicate[second_record..second_record + 8]
        .copy_from_slice(&fnv1a_hash(&first_id).to_le_bytes());

    let mut docset_hasher = Sha256::new();
    update_digest_domain(&mut docset_hasher, ORDERED_DOCSET_DIGEST_DOMAIN);
    docset_hasher.update(2_u64.to_be_bytes());
    for record_offset in [first_record, second_record] {
        let id_offset = usize::try_from(u32::from_le_bytes(
            duplicate[record_offset + 8..record_offset + 12]
                .try_into()
                .expect("duplicate id offset"),
        ))
        .expect("duplicate id offset fits");
        let id_len = usize::from(u16::from_le_bytes(
            duplicate[record_offset + 12..record_offset + 14]
                .try_into()
                .expect("duplicate id length"),
        ));
        docset_hasher.update(u64::try_from(id_len).expect("id length fits").to_be_bytes());
        docset_hasher
            .update(&duplicate[strings_offset + id_offset..strings_offset + id_offset + id_len]);
    }
    let duplicate_docset_digest: [u8; SHA256_BYTES] = docset_hasher.finalize().into();
    duplicate[FSVI_V2_DOCSET_DIGEST_OFFSET..FSVI_V2_DOCSET_DIGEST_OFFSET + SHA256_BYTES]
        .copy_from_slice(&duplicate_docset_digest);
    refresh_v2_header_crc(&mut duplicate);
    assert_eq!(
        &duplicate[FSVI_V2_DOCSET_DIGEST_OFFSET..FSVI_V2_DOCSET_DIGEST_OFFSET + SHA256_BYTES],
        duplicate_docset_digest.as_slice(),
        "fixture must preserve both the ordered docset digest and header CRC"
    );

    assert!(matches!(
        ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(duplicate), &binding),
        Err(FsviAdmissionError::Index(
            SearchError::IndexCorrupted { .. }
        ))
    ));
}

#[test]
fn fsvi_v2_owner_rejects_resealed_noncanonical_document_id_order() {
    let binding = fsvi_v2_binding("v2-record-order-corruption", 4, Quantization::F16, 12, 0x67);
    let path = temp_index_path("v2-record-order-corruption");
    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("create v2 fixture");
    writer
        .write_record("aa-order", &[1.0, 0.0, 0.0, 0.0])
        .expect("write first equal-length id");
    writer
        .write_record("bb-order", &[0.0, 1.0, 0.0, 0.0])
        .expect("write second equal-length id");
    writer.finish().expect("finish v2 fixture");

    let mut bytes = fs::read(&path).expect("read fixture bytes");
    let header_size = usize::try_from(u32::from_le_bytes(
        bytes[6..10].try_into().expect("header size"),
    ))
    .expect("header size fits");
    let records_offset = header_size;
    let strings_offset = header_size + (2 * RECORD_SIZE_BYTES);
    let first_record = records_offset;
    let second_record = first_record + RECORD_SIZE_BYTES;
    let first_id_offset = usize::try_from(u32::from_le_bytes(
        bytes[first_record + 8..first_record + 12]
            .try_into()
            .expect("first id offset"),
    ))
    .expect("first id offset fits");
    let second_id_offset = usize::try_from(u32::from_le_bytes(
        bytes[second_record + 8..second_record + 12]
            .try_into()
            .expect("second id offset"),
    ))
    .expect("second id offset fits");
    let first_id_len = usize::from(u16::from_le_bytes(
        bytes[first_record + 12..first_record + 14]
            .try_into()
            .expect("first id length"),
    ));
    let second_id_len = usize::from(u16::from_le_bytes(
        bytes[second_record + 12..second_record + 14]
            .try_into()
            .expect("second id length"),
    ));
    assert_eq!(
        first_id_len, second_id_len,
        "fixture needs equal-length ids"
    );

    let first_id = bytes
        [strings_offset + first_id_offset..strings_offset + first_id_offset + first_id_len]
        .to_vec();
    let second_id = bytes
        [strings_offset + second_id_offset..strings_offset + second_id_offset + second_id_len]
        .to_vec();
    bytes[strings_offset + first_id_offset..strings_offset + first_id_offset + first_id_len]
        .copy_from_slice(&second_id);
    bytes[strings_offset + second_id_offset..strings_offset + second_id_offset + second_id_len]
        .copy_from_slice(&first_id);
    bytes[first_record..first_record + 8].copy_from_slice(&fnv1a_hash(&second_id).to_le_bytes());
    bytes[second_record..second_record + 8].copy_from_slice(&fnv1a_hash(&first_id).to_le_bytes());

    let mut docset_hasher = Sha256::new();
    update_digest_domain(&mut docset_hasher, ORDERED_DOCSET_DIGEST_DOMAIN);
    docset_hasher.update(2_u64.to_be_bytes());
    for (id_offset, id_len) in [
        (first_id_offset, first_id_len),
        (second_id_offset, second_id_len),
    ] {
        docset_hasher.update(u64::try_from(id_len).expect("id length fits").to_be_bytes());
        docset_hasher
            .update(&bytes[strings_offset + id_offset..strings_offset + id_offset + id_len]);
    }
    let docset_digest: [u8; SHA256_BYTES] = docset_hasher.finalize().into();
    bytes[FSVI_V2_DOCSET_DIGEST_OFFSET..FSVI_V2_DOCSET_DIGEST_OFFSET + SHA256_BYTES]
        .copy_from_slice(&docset_digest);
    refresh_v2_header_crc(&mut bytes);

    assert!(matches!(
        ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), &binding),
        Err(FsviAdmissionError::Index(
            SearchError::IndexCorrupted { .. }
        ))
    ));
}

#[test]
fn fsvi_v2_writer_refuses_to_publish_beside_an_existing_wal() {
    let path = temp_index_path("v2-writer-existing-wal");
    let binding = fsvi_v2_binding("v2-writer-existing-wal", 4, Quantization::F16, 11, 0x65);
    let mut writer = VectorIndex::create_v2(&path, binding).expect("create v2 writer");
    writer
        .write_record("doc", &[1.0, 0.0, 0.0, 0.0])
        .expect("write record");
    let wal_path = wal::wal_path_for(&path);
    fs::write(&wal_path, b"stale-wal").expect("write stale WAL");

    let error = writer
        .finish()
        .expect_err("v2 publication must refuse every existing WAL sidecar");
    assert!(matches!(
        error,
        SearchError::InvalidConfig { ref field, .. } if field == "fsvi_v2.wal_sidecar"
    ));
    assert!(
        !path.exists(),
        "a rejected v2 generation must not publish its target"
    );
    assert!(
        wal_path.exists(),
        "refusal must preserve the pre-existing WAL for operator recovery"
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn completed_sibling_publication_uses_sealed_bytes_and_preserves_evidence() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let destination = directory.path().join("current.fsvi");
    let completed_sibling = directory.path().join("completed.fsvi");
    let binding = fsvi_v2_binding(
        "completed-sibling-publication",
        4,
        Quantization::F16,
        31,
        0x91,
    );

    let mut old_writer =
        VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
    old_writer
        .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("old record");
    old_writer.finish().expect("finish old generation");
    let destination_wal = wal::wal_path_for(&destination);
    fs::write(&destination_wal, b"stale destination WAL").expect("write stale WAL");

    let mut completed_writer =
        VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
    completed_writer
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("write alpha");
    completed_writer
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("write beta");
    completed_writer.finish().expect("finish completed sibling");
    let completed_bytes = fs::read(&completed_sibling).expect("read completed evidence");
    let completed_owner = ValidatedFsviBytes::open_published(&completed_sibling, &binding)
        .expect("admit completed evidence");
    let expected_witness = completed_owner.witness().clone();

    let published =
        ValidatedFsviBytes::publish_completed_sibling(&destination, &completed_sibling, &binding)
            .expect("publish exact completed bytes");

    assert_eq!(published.witness(), &expected_witness);
    assert!(published.published_wal_absent());
    assert_eq!(
        fs::read(&destination).expect("read published generation"),
        completed_bytes
    );
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains readable"),
        completed_bytes
    );
    assert!(
        !destination_wal.exists(),
        "stale destination WAL must be removed only after main publication"
    );
    let hits = published
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 1, None)
        .expect("search exact published owner");
    assert_eq!(hits[0].doc_id, "doc-alpha");
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn completed_sibling_alias_path_is_rejected_without_touching_evidence() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let completed_sibling = directory.path().join("completed.fsvi");
    let destination_alias = directory.path().join(".").join("completed.fsvi");
    let binding = fsvi_v2_binding("completed-sibling-alias", 4, Quantization::F16, 36, 0x96);

    let mut writer =
        VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
    writer
        .write_record("doc", &[1.0, 0.0, 0.0, 0.0])
        .expect("write record");
    writer.finish().expect("finish completed sibling");
    let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed evidence");
    let entries_before = directory_entry_names(directory.path());

    let error = ValidatedFsviBytes::publish_completed_sibling(
        &destination_alias,
        &completed_sibling,
        &binding,
    )
    .expect_err("same-inode alias must be rejected");
    assert!(matches!(
        error,
        FsviAdmissionError::Index(SearchError::InvalidConfig {
            ref field,
            ref reason,
            ..
        }) if field == "fsvi_v2.completed_sibling"
            && reason.contains("same file")
    ));
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains"),
        completed_bytes
    );
    assert_eq!(directory_entry_names(directory.path()), entries_before);
    let reopened = ValidatedFsviBytes::open_published(&completed_sibling, &binding)
        .expect("completed evidence remains admissible");
    let expected_digest: [u8; SHA256_BYTES] = Sha256::digest(&completed_bytes).into();
    assert_eq!(reopened.witness().whole_image_sha256, expected_digest);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn completed_sibling_cannot_alias_destination_wal_path() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let destination = directory.path().join("current.fsvi");
    let completed_sibling = wal::wal_path_for(&destination);
    let binding = fsvi_v2_binding(
        "completed-sibling-wal-alias",
        4,
        Quantization::F16,
        38,
        0x98,
    );

    let mut old_writer =
        VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
    old_writer
        .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("old record");
    old_writer.finish().expect("finish old generation");

    let mut writer =
        VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
    writer
        .write_record("new-doc", &[1.0, 0.0, 0.0, 0.0])
        .expect("write record");
    writer.finish().expect("finish completed sibling");
    let old_main_bytes = fs::read(&destination).expect("snapshot old main");
    let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed evidence");
    let entries_before = directory_entry_names(directory.path());

    let error =
        ValidatedFsviBytes::publish_completed_sibling(&destination, &completed_sibling, &binding)
            .expect_err("completed evidence at the WAL pathname must be rejected");
    assert!(matches!(
        error,
        FsviAdmissionError::Index(SearchError::InvalidConfig {
            ref field,
            ref reason,
            ..
        }) if field == "fsvi_v2.completed_sibling"
            && reason.contains("destination WAL")
    ));
    assert_eq!(
        fs::read(&destination).expect("old main remains"),
        old_main_bytes
    );
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains"),
        completed_bytes
    );
    assert_eq!(directory_entry_names(directory.path()), entries_before);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn rejected_completed_sibling_leaves_old_main_and_wal_untouched() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let destination = directory.path().join("current.fsvi");
    let completed_sibling = directory.path().join("completed.fsvi");
    let actual_binding =
        fsvi_v2_binding_with_input_variant("same-model", 4, 41, 0xa1, "actual-chunking");
    let expected_binding =
        fsvi_v2_binding_with_input_variant("same-model", 4, 41, 0xa1, "different-chunking");

    let mut old_writer =
        VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
    old_writer
        .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("old record");
    old_writer.finish().expect("finish old generation");
    let destination_wal = wal::wal_path_for(&destination);
    fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
    let old_main_bytes = fs::read(&destination).expect("snapshot old main");
    let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

    VectorIndex::create_v2(&completed_sibling, actual_binding)
        .expect("v2 writer")
        .finish()
        .expect("finish completed sibling");
    let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");
    let entries_before = directory_entry_names(directory.path());
    let destination_before = stable_file_identity(
        &fs::symlink_metadata(&destination).expect("destination metadata before rejection"),
    );
    let destination_wal_before = stable_file_identity(
        &fs::symlink_metadata(&destination_wal).expect("WAL metadata before rejection"),
    );
    let completed_before = stable_file_identity(
        &fs::symlink_metadata(&completed_sibling)
            .expect("completed evidence metadata before rejection"),
    );
    let parent_before = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("parent metadata before rejection"),
    );

    assert!(matches!(
        ValidatedFsviBytes::publish_completed_sibling(
            &destination,
            &completed_sibling,
            &expected_binding,
        ),
        Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
            reason: FsviReindexReason::IdentityMismatch,
            ..
        }))
    ));
    let destination_after = stable_file_identity(
        &fs::symlink_metadata(&destination).expect("destination metadata after rejection"),
    );
    let destination_wal_after = stable_file_identity(
        &fs::symlink_metadata(&destination_wal).expect("WAL metadata after rejection"),
    );
    let completed_after = stable_file_identity(
        &fs::symlink_metadata(&completed_sibling)
            .expect("completed evidence metadata after rejection"),
    );
    let parent_after = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("parent metadata after rejection"),
    );
    assert_eq!(destination_after, destination_before);
    assert_eq!(destination_wal_after, destination_wal_before);
    assert_eq!(completed_after, completed_before);
    assert_eq!(parent_after, parent_before);
    assert_eq!(directory_entry_names(directory.path()), entries_before);
    assert_eq!(
        fs::read(&destination).expect("old main remains"),
        old_main_bytes
    );
    assert_eq!(
        fs::read(&destination_wal).expect("old WAL remains"),
        old_wal_bytes
    );
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains"),
        completed_bytes
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn pre_replace_failure_leaves_old_generation_and_completed_evidence_untouched() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let destination = directory.path().join("current.fsvi");
    let completed_sibling = directory.path().join("completed.fsvi");
    let binding = fsvi_v2_binding("pre-replace-failure", 4, Quantization::F16, 44, 0xa4);

    let mut old_writer =
        VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
    old_writer
        .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("old record");
    old_writer.finish().expect("finish old generation");
    let destination_wal = wal::wal_path_for(&destination);
    fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
    let old_main_bytes = fs::read(&destination).expect("snapshot old main");
    let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

    VectorIndex::create_v2(&completed_sibling, binding.clone())
        .expect("v2 writer")
        .finish()
        .expect("finish completed sibling");
    let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");

    let error = ValidatedFsviBytes::publish_completed_sibling_with_hooks(
        &destination,
        &completed_sibling,
        &binding,
        |_| Ok(()),
        || {
            Err(SearchError::Cancelled {
                phase: "before-fsvi-main-replace".to_owned(),
                reason: "injected publication interruption".to_owned(),
            })
        },
        || Ok(()),
        || Ok(()),
    )
    .expect_err("injected boundary must stop before destination replacement");
    assert!(matches!(
        error,
        FsviAdmissionError::Index(SearchError::Cancelled { ref phase, .. })
            if phase == "before-fsvi-main-replace"
    ));
    assert_eq!(
        fs::read(&destination).expect("old main remains"),
        old_main_bytes
    );
    assert_eq!(
        fs::read(&destination_wal).expect("old WAL remains"),
        old_wal_bytes
    );
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains"),
        completed_bytes
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn corrupt_durable_temp_is_rejected_before_old_generation_is_touched() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let destination = directory.path().join("current.fsvi");
    let completed_sibling = directory.path().join("completed.fsvi");
    let binding = fsvi_v2_binding(
        "pre-publish-temp-validation",
        4,
        Quantization::F16,
        46,
        0xa6,
    );

    let mut old_writer =
        VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
    old_writer
        .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("old record");
    old_writer.finish().expect("finish old generation");
    let destination_wal = wal::wal_path_for(&destination);
    fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
    let old_main_bytes = fs::read(&destination).expect("snapshot old main");
    let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

    VectorIndex::create_v2(&completed_sibling, binding.clone())
        .expect("v2 writer")
        .finish()
        .expect("finish completed sibling");
    let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");

    assert!(matches!(
        ValidatedFsviBytes::publish_completed_sibling_with_hooks(
            &destination,
            &completed_sibling,
            &binding,
            |temporary_path| {
                let mut corrupted = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(temporary_path)?;
                corrupted.write_all(&FSVI_MAGIC)?;
                corrupted.sync_all()?;
                Ok(())
            },
            || Ok(()),
            || Ok(()),
            || Ok(()),
        ),
        Err(FsviAdmissionError::Index(
            SearchError::IndexCorrupted { .. }
        ))
    ));
    assert_eq!(
        fs::read(&destination).expect("old main remains"),
        old_main_bytes
    );
    assert_eq!(
        fs::read(&destination_wal).expect("old WAL remains"),
        old_wal_bytes
    );
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains"),
        completed_bytes
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn post_main_sync_failure_leaves_v2_plus_wal_fail_closed() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let destination = directory.path().join("current.fsvi");
    let completed_sibling = directory.path().join("completed.fsvi");
    let binding = fsvi_v2_binding("post-main-sync-failure", 4, Quantization::F16, 51, 0xb1);

    let mut old_writer =
        VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
    old_writer
        .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("old record");
    old_writer.finish().expect("finish old generation");
    let destination_wal = wal::wal_path_for(&destination);
    fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
    let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

    let mut completed_writer =
        VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
    completed_writer
        .write_record("new-doc", &[1.0, 0.0, 0.0, 0.0])
        .expect("new record");
    completed_writer.finish().expect("finish completed sibling");
    let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");

    let error = ValidatedFsviBytes::publish_completed_sibling_with_hooks(
        &destination,
        &completed_sibling,
        &binding,
        |_| Ok(()),
        || Ok(()),
        || {
            Err(SearchError::Cancelled {
                phase: "after-fsvi-main-sync".to_owned(),
                reason: "injected publication interruption".to_owned(),
            })
        },
        || Ok(()),
    )
    .expect_err("injected boundary must stop before WAL removal");
    assert!(matches!(
        error,
        FsviAdmissionError::Index(SearchError::Cancelled { ref phase, .. })
            if phase == "after-fsvi-main-sync"
    ));
    assert_eq!(
        fs::read(&destination).expect("new v2 main remains durable"),
        completed_bytes
    );
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains"),
        completed_bytes
    );
    assert_eq!(
        fs::read(&destination_wal).expect("old WAL remains fail-closed"),
        old_wal_bytes
    );
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&destination, &binding),
        FsviSnapshotRejectionReason::PublishedWalPresent,
    );
    assert!(matches!(
        VectorIndex::inspect(&destination),
        Ok(FsviInspection::V2IdentityComplete(_))
    ));
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn post_wal_sync_failure_leaves_exact_admissible_v2_generation() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let destination = directory.path().join("current.fsvi");
    let completed_sibling = directory.path().join("completed.fsvi");
    let binding = fsvi_v2_binding("post-wal-sync-failure", 4, Quantization::F16, 56, 0xb6);

    let mut old_writer =
        VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
    old_writer
        .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("old record");
    old_writer.finish().expect("finish old generation");
    let destination_wal = wal::wal_path_for(&destination);
    fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");

    let mut completed_writer =
        VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
    completed_writer
        .write_record("new-doc", &[1.0, 0.0, 0.0, 0.0])
        .expect("new record");
    completed_writer.finish().expect("finish completed sibling");
    let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");
    let completed_owner = ValidatedFsviBytes::open_published(&completed_sibling, &binding)
        .expect("admit completed evidence");
    let completed_witness = completed_owner.witness().clone();

    let error = ValidatedFsviBytes::publish_completed_sibling_with_hooks(
        &destination,
        &completed_sibling,
        &binding,
        |_| Ok(()),
        || Ok(()),
        || Ok(()),
        || {
            Err(SearchError::Cancelled {
                phase: "before-fsvi-final-reopen".to_owned(),
                reason: "injected publication interruption".to_owned(),
            })
        },
    )
    .expect_err("injected boundary must stop before final exact reopen");
    assert!(matches!(
        error,
        FsviAdmissionError::Index(SearchError::Cancelled { ref phase, .. })
            if phase == "before-fsvi-final-reopen"
    ));
    assert_eq!(
        fs::read(&destination).expect("new v2 main remains durable"),
        completed_bytes
    );
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains"),
        completed_bytes
    );
    assert!(
        !destination_wal.exists(),
        "WAL absence must already be durable before the final reopen boundary"
    );
    let reopened = ValidatedFsviBytes::reopen_exact(&destination, &binding, &completed_witness)
        .expect("post-WAL crash state is exact and admissible");
    assert_eq!(reopened.witness(), &completed_witness);
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[test]
fn completed_sibling_publication_is_typed_and_non_mutating_without_noatime() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let destination = directory.path().join("current.fsvi");
    let completed_sibling = directory.path().join("completed.fsvi");
    let binding = fsvi_v2_binding(
        "unsupported-noatime-publication",
        4,
        Quantization::F16,
        61,
        0xc1,
    );

    let mut old_writer =
        VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
    old_writer
        .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
        .expect("old record");
    old_writer.finish().expect("finish old generation");
    let destination_wal = wal::wal_path_for(&destination);
    fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
    let old_main_bytes = fs::read(&destination).expect("snapshot old main");
    let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

    VectorIndex::create_v2(&completed_sibling, binding.clone())
        .expect("v2 writer")
        .finish()
        .expect("finish completed sibling");
    let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");
    let entries_before = directory_entry_names(directory.path());
    let destination_before = stable_file_identity(
        &fs::symlink_metadata(&destination).expect("destination metadata before rejection"),
    );
    let destination_wal_before = stable_file_identity(
        &fs::symlink_metadata(&destination_wal).expect("WAL metadata before rejection"),
    );
    let completed_before = stable_file_identity(
        &fs::symlink_metadata(&completed_sibling)
            .expect("completed evidence metadata before rejection"),
    );
    let parent_before = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("parent metadata before rejection"),
    );

    assert_snapshot_rejection(
        ValidatedFsviBytes::publish_completed_sibling(&destination, &completed_sibling, &binding),
        FsviSnapshotRejectionReason::NoAtimeUnsupported,
    );
    let destination_after = stable_file_identity(
        &fs::symlink_metadata(&destination).expect("destination metadata after rejection"),
    );
    let destination_wal_after = stable_file_identity(
        &fs::symlink_metadata(&destination_wal).expect("WAL metadata after rejection"),
    );
    let completed_after = stable_file_identity(
        &fs::symlink_metadata(&completed_sibling)
            .expect("completed evidence metadata after rejection"),
    );
    let parent_after = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("parent metadata after rejection"),
    );
    assert_eq!(destination_after, destination_before);
    assert_eq!(destination_wal_after, destination_wal_before);
    assert_eq!(completed_after, completed_before);
    assert_eq!(parent_after, parent_before);
    assert_eq!(directory_entry_names(directory.path()), entries_before);
    assert_eq!(
        fs::read(&destination).expect("old main remains"),
        old_main_bytes
    );
    assert_eq!(
        fs::read(&destination_wal).expect("old WAL remains"),
        old_wal_bytes
    );
    assert_eq!(
        fs::read(&completed_sibling).expect("completed evidence remains"),
        completed_bytes
    );
}

#[test]
fn fsvi_v2_writer_rejects_ambiguous_records_and_invalid_bindings() {
    let duplicate_path = temp_index_path("v2-duplicate");
    let binding = fsvi_v2_binding("v2-writer-guard", 4, Quantization::F16, 11, 0x66);
    let mut duplicate = VectorIndex::create_v2(&duplicate_path, binding).expect("duplicate writer");
    duplicate
        .write_record("same", &[1.0, 0.0, 0.0, 0.0])
        .expect("first duplicate");
    duplicate
        .write_record("same", &[0.0, 1.0, 0.0, 0.0])
        .expect("second duplicate");
    assert!(matches!(
        duplicate.finish(),
        Err(SearchError::InvalidConfig { field, .. }) if field == "doc_id"
    ));

    let empty_path = temp_index_path("v2-empty-doc-id");
    let binding = fsvi_v2_binding("v2-writer-guard", 4, Quantization::F16, 12, 0x67);
    let mut empty = VectorIndex::create_v2(&empty_path, binding).expect("empty-id writer");
    empty
        .write_record("", &[1.0, 0.0, 0.0, 0.0])
        .expect("record is rejected atomically at finish");
    assert!(matches!(
        empty.finish(),
        Err(SearchError::InvalidConfig { field, .. }) if field == "doc_id"
    ));

    let mut wrong_format =
        frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
            "wrong-format",
            4,
        );
    wrong_format.storage.format = "fsvi-v1".to_owned();
    wrong_format.storage.quantization = QuantizationFormat::F16;
    wrong_format.storage.endianness = "little-endian".to_owned();
    let generation = ArtifactGenerationIdentityV1::new(1, [0x70; 16]).expect("valid generation");
    assert!(
        FsviV2IdentityBinding::new(
            generation,
            wrong_format
                .freeze()
                .expect("valid non-v2 storage identity")
        )
        .is_err()
    );

    let mut unsupported =
        frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
            "unsupported-quantization",
            4,
        );
    unsupported.storage.format = "fsvi-v2".to_owned();
    unsupported.storage.quantization = QuantizationFormat::Int8;
    unsupported.storage.endianness = "little-endian".to_owned();
    assert!(
        FsviV2IdentityBinding::new(
            generation,
            unsupported
                .freeze()
                .expect("valid but unsupported storage identity")
        )
        .is_err()
    );

    let mut frozen = frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
        "noncanonical",
        4,
    );
    frozen.storage.format = "fsvi-v2".to_owned();
    frozen.storage.quantization = QuantizationFormat::F16;
    frozen.storage.endianness = "little-endian".to_owned();
    let mut frozen = frozen.freeze().expect("valid frozen identity");
    frozen.canonical_bytes.push(0);
    assert!(FsviV2IdentityBinding::new(generation, frozen).is_err());

    let invalid_generation = ArtifactGenerationIdentityV1 {
        schema_version: 1,
        sequence: 1,
        nonce: [0; 16],
    };
    let valid_binding = fsvi_v2_binding("v2-generation-guard", 4, Quantization::F16, 1, 0x77);
    assert!(
        FsviV2IdentityBinding::new(invalid_generation, valid_binding.frozen_identity().clone())
            .is_err()
    );
}
