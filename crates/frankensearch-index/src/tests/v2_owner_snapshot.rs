#![allow(clippy::wildcard_imports)]

use super::*;

#[test]
fn validated_fsvi_bytes_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<ValidatedFsviBytes>();
}

#[test]
fn snapshot_parent_maps_only_missing_or_empty_parents_to_current_directory() {
    assert_eq!(
        snapshot_parent_or_current(Path::new("index.fsvi")),
        Path::new(".")
    );
    assert_eq!(
        snapshot_parent_or_current(Path::new("./index.fsvi")),
        Path::new(".")
    );
    assert_eq!(
        snapshot_parent_or_current(Path::new("nested/index.fsvi")),
        Path::new("nested")
    );
}

#[test]
fn fsvi_v2_round_trip_inspects_and_admits_exact_identity() {
    let binding = fsvi_v2_binding("v2-round-trip", 4, Quantization::F16, u64::MAX, 0xa5);
    let (path, expected) = write_v2_fixture("v2-round-trip", binding);

    let inspected = VectorIndex::inspect(&path).expect("inspect v2");
    assert!(matches!(&inspected, FsviInspection::V2IdentityComplete(_)));
    let FsviInspection::V2IdentityComplete(metadata) = inspected else {
        return;
    };
    assert_eq!(metadata.fsvi_version, FSVI_V2_VERSION);
    assert_eq!(metadata.dimension, 4);
    assert_eq!(metadata.quantization, Quantization::F16);
    assert_eq!(metadata.record_count, 2);
    assert_eq!(
        metadata
            .identity_v2
            .as_ref()
            .expect("v2 identity metadata")
            .generation,
        expected.generation()
    );
    assert_eq!(expected.generation().sequence, u64::MAX);
    assert_eq!(metadata.vectors_offset % VECTOR_ALIGN_BYTES, 0);

    let mut index =
        admit_owned_v2_fixture(&path, &expected).expect("exact owned admission succeeds");
    assert!(index.is_identity_admitted_v2());
    assert!(!index.published_wal_absent());
    assert!(index.owner_and_search_share_allocation());
    assert_eq!(index.record_count(), 2);
    assert_eq!(index.embedder_id(), "v2-round-trip");
    assert_eq!(index.embedder_revision(), "explicit-test-v1");
    let identity = index.identity_v2();
    assert_ne!(identity.identity_bundle_fingerprint, [0; SHA256_BYTES]);
    assert_ne!(identity.ordered_live_docset_digest, [0; SHA256_BYTES]);
    assert_ne!(identity.vector_content_digest, [0; SHA256_BYTES]);

    let alpha = index
        .find_index_by_doc_hash(fnv1a_hash(b"doc-alpha"))
        .expect("find alpha");
    assert_eq!(index.doc_id_at(alpha).expect("alpha id"), "doc-alpha");
    let vector = index.vector_at_f32(alpha).expect("alpha vector");
    assert!((vector[0] - 1.0).abs() < 0.002);

    let query = [1.0, 0.0, 0.0, 0.0];
    let owner_hits = index
        .search_top_k(&query, 2, None)
        .expect("owner exact search");
    let normal_hits = index
        .index
        .search_top_k(&query, 2, None)
        .expect("normal exact search over same allocation");
    assert_eq!(owner_hits.len(), normal_hits.len());
    for (owner_hit, normal_hit) in owner_hits.iter().zip(&normal_hits) {
        assert_eq!(owner_hit.index, normal_hit.index);
        assert_eq!(owner_hit.doc_id, normal_hit.doc_id);
        assert_eq!(owner_hit.score.to_bits(), normal_hit.score.to_bits());
    }

    let serialized = serde_json::to_vec(index.witness()).expect("serialize witness");
    let round_trip: FsviV2Witness =
        serde_json::from_slice(&serialized).expect("deserialize witness");
    assert_eq!(&round_trip, index.witness());

    let rows = index.row_source();
    assert_eq!(rows.witness(), index.witness());
    assert_eq!(
        rows.identity_bundle_fingerprint(),
        &identity.identity_bundle_fingerprint
    );
    assert_eq!(rows.space_fingerprint(), &identity.space_fingerprint);
    assert_eq!(rows.producer_fingerprint(), &identity.producer_fingerprint);
    assert_eq!(rows.input_fingerprint(), &identity.input_fingerprint);
    assert_eq!(rows.storage_fingerprint(), &identity.storage_fingerprint);
    assert_eq!(
        rows.generation_fingerprint(),
        &identity.generation_fingerprint
    );
    assert_eq!(
        rows.row(alpha).expect("owner row").vector_bytes(),
        index.index.vector_bytes(alpha).expect("normal row bytes")
    );

    let before_bytes = index.bytes.to_vec();
    let before_witness = index.witness().clone();
    let mutation = index
        .index
        .set_record_flags(alpha, FsviRecordFlags::TOMBSTONE.bits());
    assert!(matches!(
        mutation,
        Err(SearchError::InvalidConfig { field, .. }) if field == "fsvi_v2.mutation"
    ));
    assert_eq!(index.bytes.as_ref(), before_bytes.as_slice());
    assert_eq!(index.witness(), &before_witness);

    assert_eq!(index.ann_admission(), FsviAnnAdmission::Enabled);
    assert!(!wal::wal_path_for(&path).exists());
}

#[test]
fn fsvi_v2_empty_artifact_is_identity_bound_and_admissible() {
    let path = temp_index_path("v2-empty");
    let binding = fsvi_v2_binding("v2-empty", 4, Quantization::F32, 0, 0x11);
    VectorIndex::create_v2(&path, binding.clone())
        .expect("create empty v2")
        .finish()
        .expect("finish empty v2");

    let index = admit_owned_v2_fixture(&path, &binding).expect("admit empty owned v2 artifact");
    assert_eq!(index.record_count(), 0);
    let identity = index.identity_v2();
    assert_ne!(identity.ordered_live_docset_digest, [0; SHA256_BYTES]);
    assert_ne!(identity.vector_content_digest, [0; SHA256_BYTES]);
    assert_eq!(
        usize::try_from(index.metadata().vectors_offset).expect("offset fits"),
        usize::try_from(fs::metadata(&path).expect("empty artifact metadata").len())
            .expect("file length fits")
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn sealed_owner_survives_path_replacement_and_exact_reopen_rejects_new_bytes() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let path = directory.path().join("current.fsvi");
    let replacement_path = directory.path().join("replacement.fsvi");
    let retained_path = directory.path().join("retained-original.fsvi");
    let binding = fsvi_v2_binding(
        "same-display-and-generation",
        4,
        Quantization::F16,
        21,
        0x81,
    );

    let mut original = VectorIndex::create_v2(&path, binding.clone()).expect("original writer");
    original
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("original alpha");
    original
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("original beta");
    original.finish().expect("finish original");
    let owner = ValidatedFsviBytes::open_published(&path, &binding).expect("admit original owner");
    let expected_witness = owner.witness().clone();
    let expected_hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search original");
    let expected_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("original row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();

    let mut replacement =
        VectorIndex::create_v2(&replacement_path, binding.clone()).expect("replacement writer");
    replacement
        .write_record("doc-alpha", &[0.0, 1.0, 0.0, 0.0])
        .expect("replacement alpha");
    replacement
        .write_record("doc-beta", &[1.0, 0.0, 0.0, 0.0])
        .expect("replacement beta");
    replacement.finish().expect("finish replacement");
    fs::rename(&path, &retained_path).expect("retain original inode");
    fs::rename(&replacement_path, &path).expect("publish replacement inode");

    assert_eq!(owner.witness(), &expected_witness);
    let observed_hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search retained owner");
    assert_eq!(observed_hits.len(), expected_hits.len());
    for (observed, expected) in observed_hits.iter().zip(&expected_hits) {
        assert_eq!(observed.index, expected.index);
        assert_eq!(observed.doc_id, expected.doc_id);
        assert_eq!(observed.score.to_bits(), expected.score.to_bits());
    }
    let observed_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("retained owner row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();
    assert_eq!(observed_rows, expected_rows);

    assert!(matches!(
        ValidatedFsviBytes::reopen_exact(&path, &binding, &expected_witness),
        Err(FsviAdmissionError::SnapshotRejected(FsviSnapshotRejected {
            reason: FsviSnapshotRejectionReason::WitnessMismatch,
            ..
        }))
    ));
    let replacement_owner =
        ValidatedFsviBytes::open_published(&path, &binding).expect("admit replacement");
    assert_eq!(
        replacement_owner.witness().ordered_live_docset_digest,
        expected_witness.ordered_live_docset_digest
    );
    assert_ne!(
        replacement_owner.witness().vector_content_digest,
        expected_witness.vector_content_digest
    );
    assert_ne!(
        replacement_owner.witness().whole_image_sha256,
        expected_witness.whole_image_sha256
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn sealed_owner_survives_in_place_same_size_path_rewrite() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let path = directory.path().join("current.fsvi");
    let replacement_path = directory.path().join("replacement.fsvi");
    let binding = fsvi_v2_binding("same-size-rewrite", 4, Quantization::F16, 22, 0x82);

    let mut original = VectorIndex::create_v2(&path, binding.clone()).expect("original writer");
    original
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("original alpha");
    original
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("original beta");
    original.finish().expect("finish original");
    let owner = ValidatedFsviBytes::open_published(&path, &binding).expect("admit original owner");
    let expected_witness = owner.witness().clone();
    let expected_hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search original");

    let mut replacement =
        VectorIndex::create_v2(&replacement_path, binding.clone()).expect("replacement writer");
    replacement
        .write_record("doc-alpha", &[0.0, 1.0, 0.0, 0.0])
        .expect("replacement alpha");
    replacement
        .write_record("doc-beta", &[1.0, 0.0, 0.0, 0.0])
        .expect("replacement beta");
    replacement.finish().expect("finish replacement");

    let replacement_bytes = fs::read(&replacement_path).expect("read replacement bytes");
    let original_identity =
        stable_file_identity(&fs::symlink_metadata(&path).expect("original metadata"));
    assert_eq!(
        usize::try_from(original_identity.size).expect("original length fits"),
        replacement_bytes.len(),
        "fixture must exercise an in-place rewrite at the exact old length"
    );
    fs::write(&path, &replacement_bytes).expect("rewrite original pathname in place");
    assert_eq!(
        stable_file_identity(&fs::symlink_metadata(&path).expect("rewritten metadata")).inode,
        original_identity.inode,
        "same-size rewrite must retain the pathname inode rather than rename a replacement"
    );

    assert_eq!(owner.witness(), &expected_witness);
    let observed_hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search retained owner");
    assert_eq!(observed_hits.len(), expected_hits.len());
    for (observed, expected) in observed_hits.iter().zip(&expected_hits) {
        assert_eq!(observed.index, expected.index);
        assert_eq!(observed.doc_id, expected.doc_id);
        assert_eq!(observed.score.to_bits(), expected.score.to_bits());
    }

    assert!(matches!(
        ValidatedFsviBytes::reopen_exact(&path, &binding, &expected_witness),
        Err(FsviAdmissionError::SnapshotRejected(FsviSnapshotRejected {
            reason: FsviSnapshotRejectionReason::WitnessMismatch,
            ..
        }))
    ));
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn sealed_owner_survives_post_admission_hardlink_and_fresh_open_rejects_alias() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let path = directory.path().join("current.fsvi");
    let hardlink_path = directory.path().join("post-admission-alias.fsvi");
    let binding = fsvi_v2_binding("post-admission-hardlink", 4, Quantization::F16, 24, 0x84);

    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("original writer");
    writer
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("original alpha");
    writer
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("original beta");
    writer.finish().expect("finish original");

    let owner = ValidatedFsviBytes::open_published(&path, &binding).expect("admit original owner");
    let expected_witness = owner.witness().clone();
    let expected_hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search original");
    let expected_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("original row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();

    fs::hard_link(&path, &hardlink_path).expect("create post-admission hardlink alias");
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&path, &binding),
        FsviSnapshotRejectionReason::HardLinked,
    );
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&hardlink_path, &binding),
        FsviSnapshotRejectionReason::HardLinked,
    );

    assert_eq!(owner.witness(), &expected_witness);
    assert_eq!(
        owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search retained owner after hardlink alias"),
        expected_hits
    );
    let observed_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("retained owner row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();
    assert_eq!(observed_rows, expected_rows);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn sealed_owner_survives_post_admission_wal_sidecar_and_fresh_open_rejects_it() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let path = directory.path().join("current.fsvi");
    let binding = fsvi_v2_binding("post-admission-wal", 4, Quantization::F16, 25, 0x85);

    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("original writer");
    writer
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("original alpha");
    writer
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("original beta");
    writer.finish().expect("finish original");

    let owner = ValidatedFsviBytes::open_published(&path, &binding).expect("admit original owner");
    let expected_witness = owner.witness().clone();
    let expected_hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search original");
    let expected_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("original row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();

    let wal_path = wal::wal_path_for(&path);
    fs::write(&wal_path, []).expect("introduce post-admission empty WAL sidecar");
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&path, &binding),
        FsviSnapshotRejectionReason::PublishedWalPresent,
    );

    assert_eq!(owner.witness(), &expected_witness);
    assert_eq!(
        owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search retained owner after WAL sidecar"),
        expected_hits
    );
    let observed_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("retained owner row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();
    assert_eq!(observed_rows, expected_rows);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn sealed_owner_survives_post_admission_symlink_substitution() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("private publication directory");
    let path = directory.path().join("current.fsvi");
    let retained_path = directory.path().join("retained-original.fsvi");
    let binding = fsvi_v2_binding("post-admission-symlink", 4, Quantization::F16, 26, 0x86);

    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("original writer");
    writer
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("original alpha");
    writer
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("original beta");
    writer.finish().expect("finish original");

    let owner = ValidatedFsviBytes::open_published(&path, &binding).expect("admit original owner");
    let expected_witness = owner.witness().clone();
    let expected_hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search original");
    let expected_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("original row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();

    fs::rename(&path, &retained_path).expect("retain admitted pathname target");
    symlink(&retained_path, &path).expect("substitute admitted pathname with symlink");
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&path, &binding),
        FsviSnapshotRejectionReason::SymbolicLink,
    );

    assert_eq!(owner.witness(), &expected_witness);
    assert_eq!(
        owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search retained owner after symlink substitution"),
        expected_hits
    );
    let observed_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("retained owner row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();
    assert_eq!(observed_rows, expected_rows);
}

#[test]
fn sealed_owner_survives_path_truncation_and_extension_after_admission() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let path = directory.path().join("current.fsvi");
    let binding = fsvi_v2_binding("length-mutation", 4, Quantization::F16, 23, 0x83);

    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("original writer");
    writer
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("original alpha");
    writer
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("original beta");
    writer.finish().expect("finish original");

    let owner = ValidatedFsviBytes::open_published(&path, &binding).expect("admit original owner");
    let expected_witness = owner.witness().clone();
    let expected_hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
        .expect("search original");
    let expected_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("original row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();
    let original_bytes = owner.bytes.to_vec();

    OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open path for truncation")
        .set_len(0)
        .expect("truncate admitted pathname");
    assert!(
        ValidatedFsviBytes::open_published(&path, &binding).is_err(),
        "a truncated pathname must never be admitted as the retained owner"
    );
    assert_eq!(owner.witness(), &expected_witness);
    assert_eq!(
        owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search retained owner after truncation"),
        expected_hits
    );

    fs::write(&path, &original_bytes).expect("restore original pathname bytes");
    let mut extended = OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open path for extension");
    extended
        .write_all(b"unexpected-trailing-bytes")
        .expect("extend admitted pathname");
    extended.sync_all().expect("sync extension");
    assert!(
        ValidatedFsviBytes::open_published(&path, &binding).is_err(),
        "a length-extended pathname must never be admitted as the retained owner"
    );
    assert_eq!(owner.witness(), &expected_witness);
    let observed_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
        .map(|index| {
            let row = owner.row(index).expect("retained owner row");
            (
                row.doc_id().to_owned(),
                row.vector_bytes().to_vec(),
                row.flags(),
            )
        })
        .collect();
    assert_eq!(observed_rows, expected_rows);
    assert_eq!(
        owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search retained owner after extension"),
        expected_hits
    );
}

#[test]
fn same_display_strings_and_dimension_cannot_cross_open_distinct_identity_bundles() {
    let path_a = temp_index_path("v2-same-display-identity-a");
    let path_b = temp_index_path("v2-same-display-identity-b");
    let binding_a = fsvi_v2_binding_with_input_variant("same-display", 4, 22, 0x82, "none-v1");
    let binding_b = fsvi_v2_binding_with_input_variant("same-display", 4, 22, 0x82, "none-v2");
    for (path, binding) in [(&path_a, &binding_a), (&path_b, &binding_b)] {
        let mut writer = VectorIndex::create_v2(path, binding.clone()).expect("identity writer");
        writer
            .write_record("same-doc", &[1.0, 0.0, 0.0, 0.0])
            .expect("identity record");
        writer.finish().expect("identity finish");
    }

    let owner_a = admit_owned_v2_fixture(&path_a, &binding_a).expect("owner a");
    let owner_b = admit_owned_v2_fixture(&path_b, &binding_b).expect("owner b");
    assert_eq!(owner_a.embedder_id(), owner_b.embedder_id());
    assert_eq!(owner_a.embedder_revision(), owner_b.embedder_revision());
    assert_eq!(owner_a.dimension(), owner_b.dimension());
    assert_ne!(
        owner_a.witness().identity_bundle_fingerprint,
        owner_b.witness().identity_bundle_fingerprint
    );
    assert_ne!(owner_a.witness(), owner_b.witness());
    assert!(matches!(
        admit_owned_v2_fixture(&path_a, &binding_b),
        Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
            reason: FsviReindexReason::IdentityMismatch,
            ..
        }))
    ));
    assert!(matches!(
        admit_owned_v2_fixture(&path_b, &binding_a),
        Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
            reason: FsviReindexReason::IdentityMismatch,
            ..
        }))
    ));
}

#[test]
fn tombstones_are_physical_but_absent_from_live_digest_and_exact_search() {
    let path = temp_index_path("v2-tombstone-owner");
    let binding = fsvi_v2_binding("v2-tombstone-owner", 4, Quantization::F32, 23, 0x83);
    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("tombstone writer");
    writer
        .write_tombstone_record("best-but-dead", &[1.0, 0.0, 0.0, 0.0])
        .expect("tombstone row");
    writer
        .write_record("live-result", &[0.0, 1.0, 0.0, 0.0])
        .expect("live row");
    writer.finish().expect("finish tombstone fixture");

    let owner = admit_owned_v2_fixture(&path, &binding).expect("tombstone owner");
    assert_eq!(owner.record_count(), 2);
    assert_eq!(owner.live_count(), 1);
    assert_eq!(owner.tombstone_count(), 1);
    let rows = owner.row_source();
    let states: Vec<(String, FsviRecordFlags)> = (0..rows.record_count())
        .map(|index| {
            let row = rows.row(index).expect("validated row");
            (row.doc_id().to_owned(), row.flags())
        })
        .collect();
    assert!(states.contains(&("best-but-dead".to_owned(), FsviRecordFlags::TOMBSTONE)));
    assert!(states.contains(&("live-result".to_owned(), FsviRecordFlags::LIVE)));

    let hits = owner
        .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
        .expect("exact tombstone search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "live-result");

    let mut expected_docset = Sha256::new();
    update_digest_domain(&mut expected_docset, ORDERED_DOCSET_DIGEST_DOMAIN);
    expected_docset.update(1_u64.to_be_bytes());
    expected_docset.update(
        u64::try_from("live-result".len())
            .expect("live id length")
            .to_be_bytes(),
    );
    expected_docset.update(b"live-result");
    let expected_docset: [u8; SHA256_BYTES] = expected_docset.finalize().into();
    assert_eq!(owner.witness().ordered_live_docset_digest, expected_docset);
    assert_eq!(owner.ann_admission(), FsviAnnAdmission::Enabled);
}

#[test]
fn all_live_docset_digest_remains_byte_compatible_with_xomn_formula() {
    let binding = fsvi_v2_binding("v2-all-live-compatible", 4, Quantization::F16, 24, 0x84);
    let (path, expected) = write_v2_fixture("v2-all-live-compatible", binding);
    let owner = admit_owned_v2_fixture(&path, &expected).expect("all-live owner");
    let mut prior_formula = Sha256::new();
    update_digest_domain(&mut prior_formula, ORDERED_DOCSET_DIGEST_DOMAIN);
    prior_formula.update(
        u64::try_from(owner.record_count())
            .expect("record count fits")
            .to_be_bytes(),
    );
    for index in 0..owner.record_count() {
        let doc_id = owner.doc_id_at(index).expect("all-live id");
        prior_formula.update(
            u64::try_from(doc_id.len())
                .expect("doc id length fits")
                .to_be_bytes(),
        );
        prior_formula.update(doc_id.as_bytes());
    }
    let prior_formula: [u8; SHA256_BYTES] = prior_formula.finalize().into();
    assert_eq!(owner.witness().ordered_live_docset_digest, prior_formula);
    assert_eq!(owner.live_count(), owner.record_count());
    assert_eq!(owner.tombstone_count(), 0);
}

#[test]
fn v2_owner_accepts_unicode_and_maximum_u16_document_ids() {
    let path = temp_index_path("v2-unicode-boundary-ids");
    let binding = fsvi_v2_binding("v2-unicode-boundary-ids", 4, Quantization::F16, 25, 0x85);
    let boundary_id = "x".repeat(usize::from(u16::MAX));
    let unicode_id = "航海図-🏴‍☠️-café";
    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("boundary writer");
    writer
        .write_record(&boundary_id, &[1.0, 0.0, 0.0, 0.0])
        .expect("maximum id");
    writer
        .write_record(unicode_id, &[0.0, 1.0, 0.0, 0.0])
        .expect("unicode id");
    writer.finish().expect("finish boundary fixture");

    let owner = admit_owned_v2_fixture(&path, &binding).expect("boundary owner");
    let ids: std::collections::HashSet<String> = (0..owner.record_count())
        .map(|index| owner.doc_id_at(index).expect("boundary id").to_owned())
        .collect();
    assert!(ids.contains(&boundary_id));
    assert!(ids.contains(unicode_id));
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[test]
fn published_open_fails_closed_when_noatime_is_unsupported() {
    let binding = fsvi_v2_binding("v2-noatime-unsupported", 4, Quantization::F16, 26, 0x86);
    let (path, expected) = write_v2_fixture("v2-noatime-unsupported", binding);

    assert_snapshot_rejection(
        VectorIndex::open_admitted_v2(&path, &expected),
        FsviSnapshotRejectionReason::NoAtimeUnsupported,
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn published_open_preserves_literal_bytes_metadata_directory_and_wal_absence() {
    let directory = tempfile::tempdir().expect("private publication directory");
    let path = directory.path().join("generation.fsvi");
    let binding = fsvi_v2_binding("v2-side-effect-free", 4, Quantization::F16, 26, 0x86);
    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("side-effect writer");
    writer
        .write_record("doc", &[1.0, 0.0, 0.0, 0.0])
        .expect("side-effect row");
    writer.finish().expect("finish side-effect fixture");

    let expected_bytes = fs::read(&path).expect("read expected image");
    let timestamp = UNIX_EPOCH + Duration::from_secs(1_600_000_123);
    let file = OpenOptions::new()
        .read(true)
        .open(&path)
        .expect("open fixture to freeze timestamps");
    file.set_times(
        std::fs::FileTimes::new()
            .set_accessed(timestamp)
            .set_modified(timestamp),
    )
    .expect("freeze fixture timestamps");
    drop(file);

    let wal_path = wal::wal_path_for(&path);
    assert!(!wal_path.exists());
    let entries_before = directory_entry_names(directory.path());
    let file_before = stable_file_identity(&fs::symlink_metadata(&path).expect("metadata before"));
    let parent_before = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("parent metadata before"),
    );

    let owner = ValidatedFsviBytes::open_published(&path, &binding).expect("side-effect-free open");

    let file_after = stable_file_identity(&fs::symlink_metadata(&path).expect("metadata after"));
    let parent_after = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("parent metadata after"),
    );
    assert_eq!(file_after, file_before);
    assert_eq!(parent_after, parent_before);
    assert_eq!(directory_entry_names(directory.path()), entries_before);
    assert!(!wal_path.exists());
    assert_eq!(owner.owned_byte_len(), expected_bytes.len());
    assert_eq!(owner.bytes.as_ref(), expected_bytes.as_slice());
    assert!(owner.published_wal_absent());
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn published_open_rejects_symlink_hardlink_and_nonregular_paths() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("private topology directory");
    let target = directory.path().join("target.fsvi");
    let symlink_path = directory.path().join("symlink.fsvi");
    let hardlink_path = directory.path().join("hardlink.fsvi");
    let nonregular_path = directory.path().join("directory.fsvi");
    let binding = fsvi_v2_binding("v2-path-topology", 4, Quantization::F16, 27, 0x87);
    VectorIndex::create_v2(&target, binding.clone())
        .expect("topology writer")
        .finish()
        .expect("finish topology fixture");

    symlink(&target, &symlink_path).expect("create final-component symlink");
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&symlink_path, &binding),
        FsviSnapshotRejectionReason::SymbolicLink,
    );

    fs::hard_link(&target, &hardlink_path).expect("create hardlink alias");
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&target, &binding),
        FsviSnapshotRejectionReason::HardLinked,
    );
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&hardlink_path, &binding),
        FsviSnapshotRejectionReason::HardLinked,
    );

    fs::create_dir(&nonregular_path).expect("create nonregular final path");
    assert_snapshot_rejection(
        ValidatedFsviBytes::open_published(&nonregular_path, &binding),
        FsviSnapshotRejectionReason::NotRegularFile,
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn stable_snapshot_detects_path_and_parent_replacement_after_read() {
    let path_directory = tempfile::tempdir().expect("private path-race directory");
    let current = path_directory.path().join("current.fsvi");
    let replacement = path_directory.path().join("replacement.fsvi");
    let retained = path_directory.path().join("retained.fsvi");
    let binding = fsvi_v2_binding("v2-path-race", 4, Quantization::F16, 28, 0x88);
    for path in [&current, &replacement] {
        VectorIndex::create_v2(path, binding.clone())
            .expect("path-race writer")
            .finish()
            .expect("finish path-race fixture");
    }
    let path_snapshot =
        PublishedFsviPathSnapshot::read(&current).expect("take stable path snapshot");
    fs::rename(&current, &retained).expect("retain read inode");
    fs::rename(&replacement, &current).expect("replace publication pathname");
    assert_snapshot_rejection(
        path_snapshot.verify(),
        FsviSnapshotRejectionReason::PathChangedDuringRead,
    );

    let parent_directory = tempfile::tempdir().expect("private parent-race directory");
    let parent_path = parent_directory.path().join("current.fsvi");
    VectorIndex::create_v2(&parent_path, binding)
        .expect("parent-race writer")
        .finish()
        .expect("finish parent-race fixture");
    let parent_snapshot =
        PublishedFsviPathSnapshot::read(&parent_path).expect("take stable parent snapshot");
    fs::write(
        parent_directory.path().join("publisher-marker"),
        b"published",
    )
    .expect("mutate containing directory");
    assert_snapshot_rejection(
        parent_snapshot.verify(),
        FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
    );
}

#[test]
fn reserved_record_flag_matrix_is_not_constructible_or_admissible() {
    assert_eq!(
        FsviRecordFlags::from_bits(FsviRecordFlags::LIVE.bits()).expect("live flags"),
        FsviRecordFlags::LIVE
    );
    assert_eq!(
        FsviRecordFlags::from_bits(FsviRecordFlags::TOMBSTONE.bits()).expect("tombstone flags"),
        FsviRecordFlags::TOMBSTONE
    );
    for bits in [0x0002_u16, 0x0003, u16::MAX] {
        assert!(matches!(
            FsviRecordFlags::from_bits(bits),
            Err(SearchError::IndexCorrupted { .. })
        ));
    }

    let binding = fsvi_v2_binding("v2-reserved-flags", 4, Quantization::F16, 29, 0x89);
    let (path, expected) = write_v2_fixture("v2-reserved-flags", binding);
    let source = fs::read(path).expect("read flag fixture");
    let header_size = usize::try_from(u32::from_le_bytes(
        source[6..10].try_into().expect("header size"),
    ))
    .expect("header size fits");
    for bits in [0x0002_u16, 0x0003, u16::MAX] {
        let mut mutated = source.clone();
        mutated[header_size + 14..header_size + 16].copy_from_slice(&bits.to_le_bytes());
        assert!(matches!(
            ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(mutated), &expected),
            Err(FsviAdmissionError::Index(
                SearchError::IndexCorrupted { .. }
            ))
        ));
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn published_open_rejects_every_wal_shape_without_mutating_it() {
    for shape in ["empty", "valid", "corrupt", "truncated"] {
        let directory = tempfile::tempdir().expect("private WAL publication directory");
        let path = directory.path().join(format!("{shape}.fsvi"));
        let binding = fsvi_v2_binding(
            &format!("v2-published-wal-{shape}"),
            4,
            Quantization::F16,
            30,
            0x8a,
        );
        VectorIndex::create_v2(&path, binding.clone())
            .expect("WAL publication writer")
            .finish()
            .expect("finish WAL publication fixture");
        let wal_path = wal::wal_path_for(&path);
        match shape {
            "empty" => fs::write(&wal_path, []).expect("write empty sidecar"),
            "valid" => wal::append_wal_batch(
                &wal_path,
                &[wal::WalEntry {
                    doc_id: "wal-doc".to_owned(),
                    doc_id_hash: fnv1a_hash(b"wal-doc"),
                    embedding: vec![1.0, 0.0, 0.0, 0.0],
                }],
                4,
                Quantization::F16,
                0,
                false,
            )
            .expect("write valid sidecar"),
            "corrupt" => fs::write(&wal_path, b"not-a-wal").expect("write corrupt sidecar"),
            "truncated" => {
                wal::append_wal_batch(
                    &wal_path,
                    &[wal::WalEntry {
                        doc_id: "wal-doc".to_owned(),
                        doc_id_hash: fnv1a_hash(b"wal-doc"),
                        embedding: vec![1.0, 0.0, 0.0, 0.0],
                    }],
                    4,
                    Quantization::F16,
                    0,
                    false,
                )
                .expect("write complete sidecar before truncation");
                let mut truncated = fs::read(&wal_path).expect("read complete sidecar");
                truncated.pop();
                fs::write(&wal_path, truncated).expect("write truncated sidecar");
            }
            other => panic!("unhandled WAL shape {other}"),
        }

        let wal_bytes_before = fs::read(&wal_path).expect("read sidecar before rejection");
        let index_before =
            stable_file_identity(&fs::symlink_metadata(&path).expect("index metadata before"));
        let wal_before =
            stable_file_identity(&fs::symlink_metadata(&wal_path).expect("WAL metadata before"));
        let entries_before = directory_entry_names(directory.path());
        let parent_before = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("parent metadata before"),
        );

        assert_snapshot_rejection(
            ValidatedFsviBytes::open_published(&path, &binding),
            FsviSnapshotRejectionReason::PublishedWalPresent,
        );

        assert_eq!(
            stable_file_identity(&fs::symlink_metadata(&path).expect("index metadata after")),
            index_before
        );
        assert_eq!(
            stable_file_identity(&fs::symlink_metadata(&wal_path).expect("WAL metadata after")),
            wal_before
        );
        assert_eq!(
            stable_file_identity(
                &fs::symlink_metadata(directory.path()).expect("parent metadata after")
            ),
            parent_before
        );
        assert_eq!(directory_entry_names(directory.path()), entries_before);
        assert_eq!(
            fs::read(&wal_path).expect("read sidecar after rejection"),
            wal_bytes_before
        );
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn published_open_rejects_aliased_wal_entries_without_mutating_targets() {
    use std::os::unix::fs::symlink;

    for (alias_kind, symbolic_alias) in [("symlink", true), ("hardlink", false)] {
        let directory = tempfile::tempdir().expect("private aliased WAL publication directory");
        let path = directory.path().join(format!("{alias_kind}.fsvi"));
        let backing_path = directory.path().join(format!("{alias_kind}-wal-backing"));
        let binding = fsvi_v2_binding(
            &format!("v2-published-{alias_kind}-wal"),
            4,
            Quantization::F16,
            54,
            0xae,
        );
        VectorIndex::create_v2(&path, binding.clone())
            .expect("aliased WAL publication writer")
            .finish()
            .expect("finish aliased WAL publication fixture");
        fs::write(&backing_path, b"aliased-wal-bytes").expect("write alias backing bytes");
        let wal_path = wal::wal_path_for(&path);
        if symbolic_alias {
            symlink(&backing_path, &wal_path).expect("create WAL symlink");
        } else {
            fs::hard_link(&backing_path, &wal_path).expect("create WAL hardlink");
        }

        let index_before = stable_file_identity(
            &fs::symlink_metadata(&path).expect("index metadata before rejection"),
        );
        let wal_bytes_before = fs::read(&wal_path).expect("read aliased WAL before rejection");
        let backing_bytes_before =
            fs::read(&backing_path).expect("read alias backing before rejection");
        let wal_before = stable_file_identity(
            &fs::symlink_metadata(&wal_path).expect("WAL metadata before rejection"),
        );
        let backing_before = stable_file_identity(
            &fs::symlink_metadata(&backing_path).expect("backing metadata before rejection"),
        );
        let entries_before = directory_entry_names(directory.path());
        let parent_before = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("parent metadata before rejection"),
        );

        assert_snapshot_rejection(
            ValidatedFsviBytes::open_published(&path, &binding),
            FsviSnapshotRejectionReason::PublishedWalPresent,
        );

        assert_eq!(
            stable_file_identity(
                &fs::symlink_metadata(&path).expect("index metadata after rejection"),
            ),
            index_before
        );
        assert_eq!(
            stable_file_identity(
                &fs::symlink_metadata(&wal_path).expect("WAL metadata after rejection"),
            ),
            wal_before
        );
        assert_eq!(
            stable_file_identity(
                &fs::symlink_metadata(&backing_path).expect("backing metadata after rejection"),
            ),
            backing_before
        );
        // Inspect metadata before our own read_dir can update directory
        // access time on filesystems that record every directory read.
        assert_eq!(
            stable_file_identity(
                &fs::symlink_metadata(directory.path()).expect("parent metadata after rejection"),
            ),
            parent_before
        );
        assert_eq!(directory_entry_names(directory.path()), entries_before);
        assert_eq!(
            fs::read(&wal_path).expect("read aliased WAL after rejection"),
            wal_bytes_before
        );
        assert_eq!(
            fs::read(&backing_path).expect("read alias backing after rejection"),
            backing_bytes_before
        );
    }
}
