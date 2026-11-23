use std::sync::Arc;

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::ValidatedFsviBytes;
use crate::{FsviRecordFlags, VectorIndex, fnv1a_hash};
use frankensearch_core::error::SearchError;
use frankensearch_core::generation::QuantizationFormat;

use super::super::NATIVE_HNSW_FORMAT_VERSION;
use super::super::binding::NativeHnswGenerationBindingV2;
use super::super::codec::NativeHnsw;
use super::super::io_helpers::{encode_lower_hex, sha256_hex};
use super::super::receipt::{NativeHnswGenerationReceiptV2, native_hnsw_generation_receipt_path};
use super::super::types::{
    GraphDefect, NativeHnswParamsIdentityV1, ValidatedNativeHnsw, ValidatedNativeHnswHit,
    widen_search_width,
};
use super::{
    FSVI_V2_PUBLICATION_NONCE_OFFSET, TestStore, admitted_fsvi_owner,
    admitted_fsvi_owner_with_flags, admitted_fsvi_owner_with_publication_nonce, fsvi_rows,
    fsvi_v2_binding, fsvi_v2_binding_with_quantization, fsvi_v2_header_size,
    fsvi_v2_publication_nonce, generation_owner, params, sampled_entry_for_count,
    seed_with_tombstoned_entry, tempfile, topology_snapshot,
};

#[test]
fn validated_handle_retains_build_owner_after_caller_and_path_scope_end() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ValidatedNativeHnsw>();

    // `generation_owner` returns only owned admitted bytes; its temporary
    // source directory has already left scope before this call returns.
    let owner = generation_owner(80, 0x80, "retained-build-owner", 4, 16);
    let weak_owner = Arc::downgrade(&owner);
    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 80)
        .expect("build retains the admitted owner");
    assert_eq!(Arc::strong_count(&owner), 2);

    drop(owner);
    assert_eq!(weak_owner.strong_count(), 1);
    let hits = bound
        .search(&[1.0, 0.0, 0.0, 0.0], 4, None)
        .expect("retained build owner remains searchable");
    assert_eq!(hits.len(), 4);
    assert!(hits.iter().all(|hit| hit.flags().is_live()));
    assert!(hits.iter().all(|hit| hit.doc_id().starts_with("doc-")));

    drop(hits);
    drop(bound);
    assert!(
        weak_owner.upgrade().is_none(),
        "the handle must be the final strong owner after the caller drops its Arc"
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn owner_bound_ann_remains_on_admitted_bytes_after_pathname_replacement() {
    let directory = tempfile::tempdir().expect("temporary FSVI owner directory");
    let current_path = directory.path().join("current.fsvi");
    let replacement_path = directory.path().join("replacement.fsvi");
    let retained_path = directory.path().join("retained-original.fsvi");
    let binding = fsvi_v2_binding(82, 0x82, "ann-path-replacement", 4);

    let mut original =
        VectorIndex::create_v2(&current_path, binding.clone()).expect("create original FSVI");
    original
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("write original alpha");
    original
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("write original beta");
    original.finish().expect("finish original FSVI");

    let owner = Arc::new(
        ValidatedFsviBytes::open_published(&current_path, &binding)
            .expect("admit original published FSVI"),
    );
    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 82)
        .expect("build graph from admitted owner");
    assert_eq!(
        bound
            .search(&[1.0, 0.0, 0.0, 0.0], 1, Some(owner.record_count()))
            .expect("search original owner through ANN")[0]
            .doc_id(),
        "doc-alpha"
    );

    let mut replacement = VectorIndex::create_v2(&replacement_path, binding.clone())
        .expect("create replacement FSVI");
    replacement
        .write_record("doc-alpha", &[0.0, 1.0, 0.0, 0.0])
        .expect("write replacement alpha");
    replacement
        .write_record("doc-beta", &[1.0, 0.0, 0.0, 0.0])
        .expect("write replacement beta");
    replacement.finish().expect("finish replacement FSVI");
    std::fs::rename(&current_path, &retained_path).expect("retain original pathname target");
    std::fs::rename(&replacement_path, &current_path)
        .expect("publish semantically different replacement");

    let fresh_owner = ValidatedFsviBytes::open_published(&current_path, &binding)
        .expect("fresh admission observes replacement bytes");
    assert_eq!(
        fresh_owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 1, None)
            .expect("exact search replacement owner")[0]
            .doc_id,
        "doc-beta",
        "the substituted pathname must expose replacement semantics to fresh admission"
    );
    assert_eq!(
        bound
            .search(&[1.0, 0.0, 0.0, 0.0], 1, Some(owner.record_count()))
            .expect("search retained owner-bound ANN")[0]
            .doc_id(),
        "doc-alpha",
        "the owner-built graph must resolve through its retained admitted bytes"
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn owner_bound_ann_survives_same_size_in_place_path_rewrite() {
    use std::os::unix::fs::MetadataExt;

    let directory = tempfile::tempdir().expect("temporary FSVI owner directory");
    let current_path = directory.path().join("current.fsvi");
    let replacement_path = directory.path().join("replacement.fsvi");
    let binding = fsvi_v2_binding(83, 0x83, "ann-in-place-rewrite", 4);

    let mut original =
        VectorIndex::create_v2(&current_path, binding.clone()).expect("create original FSVI");
    original
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("write original alpha");
    original
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("write original beta");
    original.finish().expect("finish original FSVI");

    let owner = Arc::new(
        ValidatedFsviBytes::open_published(&current_path, &binding)
            .expect("admit original published FSVI"),
    );
    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 83)
        .expect("build graph from admitted owner");
    assert_eq!(
        bound
            .search(&[1.0, 0.0, 0.0, 0.0], 1, Some(owner.record_count()))
            .expect("search original owner through ANN")[0]
            .doc_id(),
        "doc-alpha"
    );

    let mut replacement = VectorIndex::create_v2(&replacement_path, binding.clone())
        .expect("create replacement FSVI");
    replacement
        .write_record("doc-alpha", &[0.0, 1.0, 0.0, 0.0])
        .expect("write replacement alpha");
    replacement
        .write_record("doc-beta", &[1.0, 0.0, 0.0, 0.0])
        .expect("write replacement beta");
    replacement.finish().expect("finish replacement FSVI");
    let replacement_bytes = std::fs::read(&replacement_path).expect("read replacement FSVI");
    let original_metadata =
        std::fs::symlink_metadata(&current_path).expect("read original metadata");
    assert_eq!(
        usize::try_from(original_metadata.len()).expect("original length fits usize"),
        replacement_bytes.len(),
        "fixture must preserve pathname length for the in-place rewrite"
    );
    std::fs::write(&current_path, replacement_bytes).expect("rewrite admitted pathname in place");
    assert_eq!(
        std::fs::symlink_metadata(&current_path)
            .expect("read rewritten metadata")
            .ino(),
        original_metadata.ino(),
        "fixture must retain the original inode rather than rename a replacement"
    );

    let fresh_owner = ValidatedFsviBytes::open_published(&current_path, &binding)
        .expect("fresh admission observes rewritten bytes");
    assert_eq!(
        fresh_owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 1, None)
            .expect("exact search rewritten owner")[0]
            .doc_id,
        "doc-beta",
        "fresh admission must observe the rewritten pathname semantics"
    );
    assert_eq!(
        bound
            .search(&[1.0, 0.0, 0.0, 0.0], 1, Some(owner.record_count()))
            .expect("search retained owner-bound ANN")[0]
            .doc_id(),
        "doc-alpha",
        "the owner-built graph must continue to resolve through admitted bytes"
    );
}

#[test]
fn owner_bound_ann_full_beam_matches_exact_owner_ranking() {
    let directory = tempfile::tempdir().expect("temporary FSVI owner directory");
    let binding = fsvi_v2_binding(84, 0x84, "ann-exact-owner-parity", 4);
    let rows = [
        (
            "dead-best".to_owned(),
            vec![1.0, 0.0, 0.0, 0.0],
            FsviRecordFlags::TOMBSTONE,
        ),
        (
            "live-alpha".to_owned(),
            vec![0.9, 0.1, 0.0, 0.0],
            FsviRecordFlags::LIVE,
        ),
        (
            "live-beta".to_owned(),
            vec![0.6, 0.4, 0.0, 0.0],
            FsviRecordFlags::LIVE,
        ),
        (
            "live-gamma".to_owned(),
            vec![0.1, 0.9, 0.0, 0.0],
            FsviRecordFlags::LIVE,
        ),
    ];
    let owner =
        admitted_fsvi_owner_with_flags(directory.path(), "owner-parity.fsvi", &binding, &rows);
    let query = [1.0, 0.0, 0.0, 0.0];
    let exact = owner
        .search_top_k(&query, owner.live_count(), None)
        .expect("exact same-owner search");
    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 84)
        .expect("build graph from exact-search owner");
    let ann = bound
        .search(&query, owner.live_count(), Some(owner.record_count()))
        .expect("full-beam owner-bound ANN search");

    assert_eq!(
        ann.iter()
            .map(ValidatedNativeHnswHit::doc_id)
            .collect::<Vec<_>>(),
        exact
            .iter()
            .map(|hit| hit.doc_id.as_str())
            .collect::<Vec<_>>(),
        "full-beam ANN must preserve the exact ranking from its retained owner"
    );
    assert_eq!(
        ann.iter()
            .map(ValidatedNativeHnswHit::physical_row)
            .collect::<Vec<_>>(),
        exact.iter().map(|hit| hit.index).collect::<Vec<_>>(),
        "ANN hits must resolve the exact physical rows owned by the same image"
    );
    assert!(ann.iter().all(|hit| hit.flags().is_live()));
    assert!(ann.iter().all(|hit| hit.doc_id() != "dead-best"));
}

#[test]
fn validated_handle_retains_loaded_owner_after_caller_scope_end() {
    let owner = generation_owner(81, 0x81, "retained-load-owner", 4, 16);
    let weak_owner = Arc::downgrade(&owner);
    let loaded = {
        let directory = tempfile::tempdir().expect("temporary graph directory");
        let graph_path = directory.path().join("retained-owner.fshnsw");
        let built = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 81)
            .expect("build owner-bound graph");
        built.save(&graph_path).expect("save owner-bound graph");
        drop(built);
        assert_eq!(Arc::strong_count(&owner), 1);

        let (loaded, _) = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
            .expect("load retains the exact admitted owner");
        loaded
    };
    // Both the original FSVI source and the persisted graph path are gone;
    // only the loaded graph and its retained owner allocation remain.
    assert_eq!(Arc::strong_count(&owner), 2);

    drop(owner);
    assert_eq!(weak_owner.strong_count(), 1);
    let hits = loaded
        .search(&[0.0, 1.0, 0.0, 0.0], 4, None)
        .expect("retained loaded owner remains searchable");
    assert_eq!(hits.len(), 4);
    assert!(hits.iter().all(|hit| hit.flags().is_live()));

    drop(hits);
    drop(loaded);
    assert!(weak_owner.upgrade().is_none());
}

// ─── FSVI generation receipts ──────────────────────────────────────

#[test]
fn generation_receipt_round_trip_binds_every_identity_layer_and_is_deterministic() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let graph_path = directory.path().join("graph.fshnsw");
    let receipt_path = directory.path().join("graph.fshnsw.receipt");
    let owner = generation_owner(u64::MAX, 0xa5, "receipt-round-trip", 8, 48);
    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 0x51de_cafe)
        .expect("owner-bound build");
    let binding = &bound.binding;

    assert_eq!(
        native_hnsw_generation_receipt_path(&graph_path).expect("canonical receipt path"),
        receipt_path
    );
    let receipt = bound.save(&graph_path).expect("save owner-bound graph");
    receipt.validate().expect("receipt validates");
    let persisted = std::fs::read(&receipt_path).expect("receipt bytes");
    let decoded =
        NativeHnswGenerationReceiptV2::from_bytes(&persisted, &receipt_path).expect("decode");
    assert_eq!(decoded, receipt);
    assert_eq!(decoded.artifact_generation, binding.artifact_generation);
    assert_eq!(
        decoded.embedding_identity_fingerprint,
        binding.embedding_identity_fingerprint
    );
    assert_eq!(
        decoded.graph_byte_len,
        std::fs::metadata(&graph_path)
            .expect("graph metadata")
            .len()
    );
    assert_eq!(
        decoded.graph_sha256,
        sha256_hex(&std::fs::read(&graph_path).unwrap())
    );
    assert_eq!(decoded.native_format_version, NATIVE_HNSW_FORMAT_VERSION);
    assert_eq!(
        decoded.params,
        NativeHnswParamsIdentityV1::from_params(params()).unwrap()
    );
    assert_eq!(decoded.point_count, 48);
    assert_eq!(decoded.fsvi_physical_row_count, 48);
    assert_eq!(decoded.vector_content_digest, binding.vector_content_digest);
    assert_eq!(
        decoded.ordered_live_docset_digest,
        binding.ordered_live_docset_digest
    );
    assert_eq!(
        decoded.fsvi_whole_image_sha256,
        binding.fsvi_whole_image_sha256
    );
    assert_eq!(
        decoded.fsvi_whole_image_sha256,
        encode_lower_hex(owner.witness().whole_image_sha256)
    );
    assert_eq!(
        decoded.topology_sha256,
        bound.graph.topology_sha256().unwrap()
    );

    let (loaded, observed) =
        ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path).expect("owner-bound load");
    assert_eq!(observed, receipt);
    assert_eq!(
        topology_snapshot(&loaded.graph),
        topology_snapshot(&bound.graph)
    );
    let hits = loaded
        .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8, None)
        .expect("owner-bound search");
    assert!(!hits.is_empty());
    for hit in hits {
        let physical_index = usize::try_from(hit.physical_row()).expect("physical row fits usize");
        let owner_row = owner.row(physical_index).expect("same-owner row");
        assert_eq!(hit.doc_id(), owner_row.doc_id());
        assert_eq!(hit.flags(), owner_row.flags());
    }

    bound.save(&graph_path).expect("repeat owner-bound save");
    assert_eq!(
        std::fs::read(&receipt_path).expect("repeat receipt"),
        persisted,
        "same graph/generation/identity must have one stable receipt encoding"
    );

    let mut serde_value = serde_json::to_value(&receipt).expect("serialize receipt");
    serde_value["future_field"] = serde_json::json!(true);
    assert!(
        serde_json::from_value::<NativeHnswGenerationReceiptV2>(serde_value).is_err(),
        "the public receipt schema must reject unknown fields"
    );
}

#[test]
fn generation_binding_is_derived_from_the_admitted_fsvi_owner() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(7, 7, "admitted-owner", 8);
    let rows = fsvi_rows(9, 8);
    let owner = admitted_fsvi_owner(directory.path(), "owner.fsvi", &fsvi_binding, &rows);
    let binding = NativeHnswGenerationBindingV2::from_validated_fsvi(&owner)
        .expect("derive binding from admitted owner");
    let identity = owner.identity_v2();

    assert_eq!(binding.artifact_generation, identity.generation);
    assert_eq!(
        binding.embedding_identity_fingerprint,
        encode_lower_hex(identity.identity_bundle_fingerprint)
    );
    assert_eq!(
        binding.vector_content_digest,
        encode_lower_hex(identity.vector_content_digest)
    );
    assert_eq!(
        binding.ordered_live_docset_digest,
        encode_lower_hex(identity.ordered_live_docset_digest)
    );
    assert_eq!(
        binding.fsvi_whole_image_sha256,
        encode_lower_hex(owner.witness().whole_image_sha256)
    );
    assert_eq!(binding.fsvi_physical_row_count, 9);
}

#[test]
fn bound_load_rejects_same_identity_and_count_with_different_vectors_before_graph_read() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(31, 0x31, "vector-content-binding", 4);
    let rows_a = vec![
        ("doc-alpha".to_owned(), vec![1.0, 0.0, 0.0, 0.0]),
        ("doc-beta".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
        ("doc-gamma".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
    ];
    let rows_b = vec![
        ("doc-alpha".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
        ("doc-beta".to_owned(), vec![1.0, 0.0, 0.0, 0.0]),
        ("doc-gamma".to_owned(), vec![0.0, 0.0, 0.0, 1.0]),
    ];
    let owner_a = admitted_fsvi_owner(directory.path(), "vectors-a.fsvi", &fsvi_binding, &rows_a);
    let owner_b = admitted_fsvi_owner(directory.path(), "vectors-b.fsvi", &fsvi_binding, &rows_b);
    let bound_a =
        ValidatedNativeHnsw::build(Arc::clone(&owner_a), params(), 31).expect("owner-bound graph");
    let binding_a = &bound_a.binding;
    let binding_b =
        NativeHnswGenerationBindingV2::from_validated_fsvi(&owner_b).expect("binding b");

    assert_eq!(binding_a.artifact_generation, binding_b.artifact_generation);
    assert_eq!(
        binding_a.embedding_identity_fingerprint,
        binding_b.embedding_identity_fingerprint
    );
    assert_eq!(
        binding_a.ordered_live_docset_digest,
        binding_b.ordered_live_docset_digest
    );
    assert_eq!(
        binding_a.fsvi_physical_row_count,
        binding_b.fsvi_physical_row_count
    );
    assert_ne!(
        binding_a.vector_content_digest,
        binding_b.vector_content_digest
    );

    let graph_path = directory.path().join("vectors.fshnsw");
    bound_a.save(&graph_path).expect("owner-bound save");
    std::fs::write(&graph_path, b"graph bytes must not be read")
        .expect("poison graph after receipt publication");

    let error = ValidatedNativeHnsw::load(Arc::clone(&owner_b), &graph_path)
        .expect_err("different vector content must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("receipt vector content digest mismatch")),
        "vector mismatch did not fail at receipt identity: {error:?}"
    );
}

#[test]
fn bound_load_rejects_same_vectors_and_identity_with_different_ordered_docset_before_graph_read() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(32, 0x32, "ordered-docset-binding", 4);
    let shared_vector = vec![1.0, 0.0, 0.0, 0.0];
    let rows_a = vec![
        ("doc-alpha".to_owned(), shared_vector.clone()),
        ("doc-beta".to_owned(), shared_vector.clone()),
        ("doc-gamma".to_owned(), shared_vector.clone()),
    ];
    let rows_b = vec![
        ("doc-alpha".to_owned(), shared_vector.clone()),
        ("doc-delta".to_owned(), shared_vector.clone()),
        ("doc-gamma".to_owned(), shared_vector),
    ];
    let owner_a = admitted_fsvi_owner(directory.path(), "docset-a.fsvi", &fsvi_binding, &rows_a);
    let owner_b = admitted_fsvi_owner(directory.path(), "docset-b.fsvi", &fsvi_binding, &rows_b);
    let bound_a =
        ValidatedNativeHnsw::build(Arc::clone(&owner_a), params(), 32).expect("owner-bound graph");
    let binding_a = &bound_a.binding;
    let binding_b =
        NativeHnswGenerationBindingV2::from_validated_fsvi(&owner_b).expect("binding b");

    assert_eq!(binding_a.artifact_generation, binding_b.artifact_generation);
    assert_eq!(
        binding_a.embedding_identity_fingerprint,
        binding_b.embedding_identity_fingerprint
    );
    assert_eq!(
        binding_a.vector_content_digest,
        binding_b.vector_content_digest
    );
    assert_eq!(
        binding_a.fsvi_physical_row_count,
        binding_b.fsvi_physical_row_count
    );
    assert_ne!(
        binding_a.ordered_live_docset_digest,
        binding_b.ordered_live_docset_digest
    );

    let graph_path = directory.path().join("docset.fshnsw");
    bound_a.save(&graph_path).expect("owner-bound save");
    std::fs::write(&graph_path, b"graph bytes must not be read")
        .expect("poison graph after receipt publication");

    let error = ValidatedNativeHnsw::load(Arc::clone(&owner_b), &graph_path)
        .expect_err("different ordered docset must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("receipt ordered live-docset digest mismatch")),
        "docset mismatch did not fail at receipt identity: {error:?}"
    );
}

#[test]
fn bound_load_rejects_same_vectors_and_live_docset_with_different_tombstone_layout() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(35, 0x35, "tombstone-layout-binding", 4);
    let live_ids = ["live-alpha", "live-beta"];
    let smallest_live_hash = live_ids
        .iter()
        .map(|doc_id| fnv1a_hash(doc_id.as_bytes()))
        .min()
        .expect("live id");
    let largest_live_hash = live_ids
        .iter()
        .map(|doc_id| fnv1a_hash(doc_id.as_bytes()))
        .max()
        .expect("live id");
    let tombstone_before = (0..100_000_u32)
        .map(|candidate| format!("dead-before-{candidate}"))
        .find(|doc_id| fnv1a_hash(doc_id.as_bytes()) < smallest_live_hash)
        .expect("find tombstone id sorting before the live rows");
    let tombstone_after = (0..100_000_u32)
        .map(|candidate| format!("dead-after-{candidate}"))
        .find(|doc_id| fnv1a_hash(doc_id.as_bytes()) > largest_live_hash)
        .expect("find tombstone id sorting after the live rows");
    let shared_vector = vec![1.0, 0.0, 0.0, 0.0];
    let rows_a = vec![
        (
            live_ids[0].to_owned(),
            shared_vector.clone(),
            FsviRecordFlags::LIVE,
        ),
        (
            live_ids[1].to_owned(),
            shared_vector.clone(),
            FsviRecordFlags::LIVE,
        ),
        (
            tombstone_before,
            shared_vector.clone(),
            FsviRecordFlags::TOMBSTONE,
        ),
    ];
    let rows_b = vec![
        (
            live_ids[0].to_owned(),
            shared_vector.clone(),
            FsviRecordFlags::LIVE,
        ),
        (
            live_ids[1].to_owned(),
            shared_vector.clone(),
            FsviRecordFlags::LIVE,
        ),
        (tombstone_after, shared_vector, FsviRecordFlags::TOMBSTONE),
    ];
    let owner_a =
        admitted_fsvi_owner_with_flags(directory.path(), "layout-a.fsvi", &fsvi_binding, &rows_a);
    let owner_b =
        admitted_fsvi_owner_with_flags(directory.path(), "layout-b.fsvi", &fsvi_binding, &rows_b);
    let binding_a =
        NativeHnswGenerationBindingV2::from_validated_fsvi(&owner_a).expect("binding a");
    let binding_b =
        NativeHnswGenerationBindingV2::from_validated_fsvi(&owner_b).expect("binding b");

    assert_eq!(binding_a.artifact_generation, binding_b.artifact_generation);
    assert_eq!(
        binding_a.embedding_identity_fingerprint,
        binding_b.embedding_identity_fingerprint
    );
    assert_eq!(
        binding_a.vector_content_digest,
        binding_b.vector_content_digest
    );
    assert_eq!(
        binding_a.ordered_live_docset_digest,
        binding_b.ordered_live_docset_digest
    );
    assert_eq!(
        binding_a.fsvi_physical_row_count,
        binding_b.fsvi_physical_row_count
    );
    assert_ne!(
        binding_a.fsvi_whole_image_sha256,
        binding_b.fsvi_whole_image_sha256
    );
    let layout_a: Vec<(String, FsviRecordFlags)> = (0..owner_a.record_count())
        .map(|index| {
            let row = owner_a.row(index).expect("owner-a row");
            (row.doc_id().to_owned(), row.flags())
        })
        .collect();
    let layout_b: Vec<(String, FsviRecordFlags)> = (0..owner_b.record_count())
        .map(|index| {
            let row = owner_b.row(index).expect("owner-b row");
            (row.doc_id().to_owned(), row.flags())
        })
        .collect();
    assert_eq!(layout_a[0].1, FsviRecordFlags::TOMBSTONE);
    assert_eq!(layout_b[layout_b.len() - 1].1, FsviRecordFlags::TOMBSTONE);
    assert_ne!(layout_a, layout_b);

    let bound_a = ValidatedNativeHnsw::build(Arc::clone(&owner_a), params(), 35)
        .expect("public tombstone-aware graph fixture");
    let graph_path = directory.path().join("layout.fshnsw");
    bound_a
        .save(&graph_path)
        .expect("public receipted graph fixture");
    std::fs::write(&graph_path, b"graph bytes must not be read")
        .expect("poison graph after receipt publication");

    let error = ValidatedNativeHnsw::load(Arc::clone(&owner_b), &graph_path)
        .expect_err("different physical tombstone layout must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("receipt FSVI whole-image SHA-256 mismatch")),
        "tombstone-layout mismatch did not fail at receipt identity: {error:?}"
    );
}

/// Owner-bound ANN over an F32 owner must rank at the owner's declared
/// storage precision, not at F16.
///
/// Planted negative: the two closest live rows differ by less than one F16
/// step at 1.0, so their F16 encodings are bit-identical (asserted), and
/// the physically FIRST of the pair carries the WORSE vector (asserted).
/// An implementation that resolved graph distances through F16 — by
/// assuming the F16 storage every other owner fixture uses, or by
/// quantizing on the way in — cannot separate the pair at all and would
/// fall back to physical-row order, returning the worse row first.
#[test]
fn owner_bound_ann_ranks_an_f32_owner_below_one_f16_step() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding_with_quantization(
        44,
        0x44,
        "ann-f32-precision",
        4,
        QuantizationFormat::F32,
    );
    // Both components collapse onto the same F16 code point, so F16 storage
    // could not represent the pair as distinct rows at all.
    let better_component = 1.0_f32;
    let worse_component = 0.9999_f32;
    assert!(worse_component < better_component);
    assert_eq!(
        half::f16::from_f32(better_component),
        half::f16::from_f32(worse_component),
        "the fixture pair must be indistinguishable under F16 storage"
    );

    let one = "ann-f32-one";
    let two = "ann-f32-two";
    let (physically_first, physically_second) =
        if fnv1a_hash(one.as_bytes()) < fnv1a_hash(two.as_bytes()) {
            (one, two)
        } else {
            (two, one)
        };
    let rows = vec![
        (
            physically_first.to_owned(),
            vec![worse_component, 0.0, 0.0, 0.0],
        ),
        (
            physically_second.to_owned(),
            vec![better_component, 0.0, 0.0, 0.0],
        ),
        ("ann-f32-far".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
    ];
    let owner = admitted_fsvi_owner(directory.path(), "f32-owner.fsvi", &fsvi_binding, &rows);

    let dimension = owner.dimension();
    assert_eq!(
        owner
            .row(0)
            .expect("first physical row")
            .vector_bytes()
            .len(),
        dimension * 4,
        "the owner must physically store four bytes per F32 component"
    );
    assert_eq!(
        owner.row(0).expect("first physical row").doc_id(),
        physically_first,
        "the worse vector must occupy the lower physical row"
    );

    let query = [1.0, 0.0, 0.0, 0.0];
    let exact = owner
        .search_top_k(&query, owner.live_count(), None)
        .expect("exact F32 owner search");
    assert_eq!(
        exact
            .iter()
            .map(|hit| hit.doc_id.as_str())
            .collect::<Vec<_>>(),
        vec![physically_second, physically_first, "ann-f32-far"],
        "exact F32 search must separate the sub-F16-step pair"
    );

    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 44)
        .expect("build graph from F32 owner");
    let ann = bound
        .search(&query, owner.live_count(), Some(owner.record_count()))
        .expect("full-beam owner-bound ANN search over F32 storage");
    assert_eq!(
        ann.iter()
            .map(ValidatedNativeHnswHit::doc_id)
            .collect::<Vec<_>>(),
        exact
            .iter()
            .map(|hit| hit.doc_id.as_str())
            .collect::<Vec<_>>(),
        "F32 owner-bound ANN must reproduce the exact F32 ranking"
    );
    assert_eq!(
        ann.iter()
            .map(ValidatedNativeHnswHit::physical_row)
            .collect::<Vec<_>>(),
        exact.iter().map(|hit| hit.index).collect::<Vec<_>>(),
        "F32 ANN hits must resolve the exact physical rows of the same owner"
    );
}

/// A stale sidecar from a content-identical predecessor generation must be
/// refused by the successor owner.
///
/// Planted negative: the two published artifacts agree on every
/// content-derived binding field — embedding identity, storage, vector
/// content digest, ordered live-docset digest, physical row count — and
/// their entire post-header content region is byte-identical. An
/// implementation that bound an ANN graph to content alone would adopt the
/// predecessor's graph for the successor generation.
#[test]
fn bound_load_rejects_a_content_identical_owner_from_a_different_generation() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let predecessor_binding = fsvi_v2_binding(41, 0x41, "generation-rebind", 4);
    let successor_binding = fsvi_v2_binding(42, 0x42, "generation-rebind", 4);
    let rows = vec![
        ("doc-alpha".to_owned(), vec![1.0, 0.0, 0.0, 0.0]),
        ("doc-beta".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
        ("doc-gamma".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
    ];
    let predecessor_path = directory.path().join("generation-predecessor.fsvi");
    let successor_path = directory.path().join("generation-successor.fsvi");
    let predecessor = admitted_fsvi_owner(
        directory.path(),
        "generation-predecessor.fsvi",
        &predecessor_binding,
        &rows,
    );
    let successor = admitted_fsvi_owner(
        directory.path(),
        "generation-successor.fsvi",
        &successor_binding,
        &rows,
    );

    let predecessor_bytes = std::fs::read(&predecessor_path).expect("read predecessor image");
    let successor_bytes = std::fs::read(&successor_path).expect("read successor image");
    let header_size = fsvi_v2_header_size(&predecessor_bytes);
    assert_eq!(
        header_size,
        fsvi_v2_header_size(&successor_bytes),
        "a pure generation change must not resize the header"
    );
    assert_eq!(
        predecessor_bytes[header_size..],
        successor_bytes[header_size..],
        "the fixture must differ only in header identity, never in content"
    );

    let predecessor_binding_v2 = NativeHnswGenerationBindingV2::from_validated_fsvi(&predecessor)
        .expect("predecessor binding");
    let successor_binding_v2 =
        NativeHnswGenerationBindingV2::from_validated_fsvi(&successor).expect("successor binding");
    assert_eq!(
        predecessor_binding_v2.embedding_identity_fingerprint,
        successor_binding_v2.embedding_identity_fingerprint
    );
    assert_eq!(
        predecessor_binding_v2.vector_storage_fingerprint,
        successor_binding_v2.vector_storage_fingerprint
    );
    assert_eq!(
        predecessor_binding_v2.vector_content_digest,
        successor_binding_v2.vector_content_digest
    );
    assert_eq!(
        predecessor_binding_v2.ordered_live_docset_digest,
        successor_binding_v2.ordered_live_docset_digest
    );
    assert_eq!(
        predecessor_binding_v2.fsvi_physical_row_count,
        successor_binding_v2.fsvi_physical_row_count
    );
    assert_ne!(
        predecessor_binding_v2.artifact_generation,
        successor_binding_v2.artifact_generation
    );
    assert_ne!(
        predecessor_binding_v2.fsvi_whole_image_sha256,
        successor_binding_v2.fsvi_whole_image_sha256
    );

    let bound = ValidatedNativeHnsw::build(Arc::clone(&predecessor), params(), 41)
        .expect("predecessor owner-bound graph");
    let graph_path = directory.path().join("generation.fshnsw");
    bound.save(&graph_path).expect("publish receipted graph");
    ValidatedNativeHnsw::load(Arc::clone(&predecessor), &graph_path)
        .expect("the sidecar must load against the generation that built it");

    std::fs::write(&graph_path, b"graph bytes must not be read")
        .expect("poison graph after receipt publication");
    let error = ValidatedNativeHnsw::load(Arc::clone(&successor), &graph_path)
        .expect_err("a predecessor generation's sidecar must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("receipt artifact generation mismatch")),
        "generation drift did not fail at receipt identity: {error:?}"
    );
}

/// Two tiers of distinct publications whose FSVI images differ only in the
/// bd-miio8 publication nonce must not share one ANN sidecar.
///
/// Planted negative: this is the strongest same-content forgery the format
/// admits. Every receipt field except the whole-image SHA-256 agrees —
/// including the artifact generation and its fingerprint, which separate
/// the content-only cases — and the two files differ in exactly the two
/// nonce bytes plus the header CRC that covers them. Only the exact
/// whole-image witness distinguishes them, so a binder that checked
/// identity, generation, and content digests would accept a graph built
/// over the other publication's bytes.
#[test]
fn bound_load_rejects_a_publication_nonce_twin_whose_every_other_receipt_field_matches() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(43, 0x43, "publication-nonce-twin", 4);
    let rows = vec![
        ("doc-alpha".to_owned(), vec![1.0, 0.0, 0.0, 0.0]),
        ("doc-beta".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
        ("doc-gamma".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
    ];
    let first_path = directory.path().join("nonce-first.fsvi");
    let second_path = directory.path().join("nonce-second.fsvi");
    let first = admitted_fsvi_owner_with_publication_nonce(
        directory.path(),
        "nonce-first.fsvi",
        &fsvi_binding,
        &rows,
        0x0101,
    );
    let second = admitted_fsvi_owner_with_publication_nonce(
        directory.path(),
        "nonce-second.fsvi",
        &fsvi_binding,
        &rows,
        0x0202,
    );
    let first_bytes = std::fs::read(&first_path).expect("read first image");
    let second_bytes = std::fs::read(&second_path).expect("read second image");
    assert_eq!(fsvi_v2_publication_nonce(&first_bytes), 0x0101);
    assert_eq!(fsvi_v2_publication_nonce(&second_bytes), 0x0202);
    let header_size = fsvi_v2_header_size(&first_bytes);
    assert_eq!(header_size, fsvi_v2_header_size(&second_bytes));
    assert_eq!(first_bytes.len(), second_bytes.len());
    let crc_offset = header_size - 4;
    assert_eq!(
        first_bytes[..FSVI_V2_PUBLICATION_NONCE_OFFSET],
        second_bytes[..FSVI_V2_PUBLICATION_NONCE_OFFSET],
        "the fixture must not disturb the header prefix"
    );
    assert_ne!(
        first_bytes[FSVI_V2_PUBLICATION_NONCE_OFFSET..FSVI_V2_PUBLICATION_NONCE_OFFSET + 2],
        second_bytes[FSVI_V2_PUBLICATION_NONCE_OFFSET..FSVI_V2_PUBLICATION_NONCE_OFFSET + 2],
        "the fixture must actually carry two different publication nonces"
    );
    assert_eq!(
        first_bytes[FSVI_V2_PUBLICATION_NONCE_OFFSET + 2..crc_offset],
        second_bytes[FSVI_V2_PUBLICATION_NONCE_OFFSET + 2..crc_offset],
        "no header identity field other than the nonce may differ"
    );
    assert_eq!(
        first_bytes[header_size..],
        second_bytes[header_size..],
        "record table, string table, padding, and vector slab must be byte-identical"
    );
    for index in 0..first.record_count() {
        let first_row = first.row(index).expect("first row");
        let second_row = second.row(index).expect("second row");
        assert_eq!(first_row.doc_id(), second_row.doc_id());
        assert_eq!(first_row.flags(), second_row.flags());
        assert_eq!(first_row.vector_bytes(), second_row.vector_bytes());
    }

    let first_binding = NativeHnswGenerationBindingV2::from_validated_fsvi(&first)
        .expect("first publication binding");
    let second_binding = NativeHnswGenerationBindingV2::from_validated_fsvi(&second)
        .expect("second publication binding");
    assert_eq!(
        first_binding.artifact_generation,
        second_binding.artifact_generation
    );
    assert_eq!(
        first_binding.artifact_generation_fingerprint,
        second_binding.artifact_generation_fingerprint
    );
    assert_eq!(
        first_binding.embedding_identity_fingerprint,
        second_binding.embedding_identity_fingerprint
    );
    assert_eq!(
        first_binding.embedding_space_fingerprint,
        second_binding.embedding_space_fingerprint
    );
    assert_eq!(
        first_binding.embedding_producer_fingerprint,
        second_binding.embedding_producer_fingerprint
    );
    assert_eq!(
        first_binding.embedding_input_fingerprint,
        second_binding.embedding_input_fingerprint
    );
    assert_eq!(
        first_binding.vector_storage_fingerprint,
        second_binding.vector_storage_fingerprint
    );
    assert_eq!(
        first_binding.vector_content_digest,
        second_binding.vector_content_digest
    );
    assert_eq!(
        first_binding.ordered_live_docset_digest,
        second_binding.ordered_live_docset_digest
    );
    assert_eq!(
        first_binding.fsvi_physical_row_count,
        second_binding.fsvi_physical_row_count
    );
    assert_ne!(
        first_binding.fsvi_whole_image_sha256, second_binding.fsvi_whole_image_sha256,
        "the whole-image witness is the only field that separates the twins"
    );

    let bound = ValidatedNativeHnsw::build(Arc::clone(&first), params(), 43)
        .expect("first publication owner-bound graph");
    let graph_path = directory.path().join("nonce.fshnsw");
    bound.save(&graph_path).expect("publish receipted graph");
    ValidatedNativeHnsw::load(Arc::clone(&first), &graph_path)
        .expect("the sidecar must load against the publication that built it");

    std::fs::write(&graph_path, b"graph bytes must not be read")
        .expect("poison graph after receipt publication");
    let error = ValidatedNativeHnsw::load(Arc::clone(&second), &graph_path)
        .expect_err("a different publication's sidecar must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("receipt FSVI whole-image SHA-256 mismatch")),
        "publication-nonce drift did not fail at receipt identity: {error:?}"
    );
}

#[test]
fn validated_owner_bound_api_routes_through_tombstones_and_round_trips() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(36, 0x36, "tombstone-admission", 4);
    let rows = vec![
        (
            "live".to_owned(),
            vec![1.0, 0.0, 0.0, 0.0],
            FsviRecordFlags::LIVE,
        ),
        (
            "dead".to_owned(),
            vec![0.0, 1.0, 0.0, 0.0],
            FsviRecordFlags::TOMBSTONE,
        ),
    ];
    let owner =
        admitted_fsvi_owner_with_flags(directory.path(), "tombstoned.fsvi", &fsvi_binding, &rows);

    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 36)
        .expect("tombstoned owner builds a public ANN handle");
    assert_eq!(bound.len(), 2, "both physical rows remain routing nodes");
    assert_eq!(owner.live_count(), 1);
    assert_eq!(owner.tombstone_count(), 1);

    let tombstone_index = (0..owner.record_count())
        .find(|&index| {
            owner
                .row(index)
                .expect("admitted owner row")
                .flags()
                .is_tombstone()
        })
        .expect("fixture tombstone");
    // ubs:ignore -- a public test query must not taint unrelated comparisons.
    let query = owner
        .vector_at_f32(tombstone_index)
        .expect("query equal to the nearest tombstoned vector");
    let hits = bound
        .search(&query, 8, Some(1))
        .expect("bounded search widens past its tombstoned nearest candidate");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id(), "live");
    assert!(hits[0].flags().is_live());
    assert_ne!(
        usize::try_from(hits[0].physical_row()).expect("physical row fits usize"),
        tombstone_index
    );

    let graph_path = directory.path().join("tombstoned.fshnsw");
    // ubs:ignore -- a public integrity receipt is not authentication material.
    let receipt = bound
        .save(&graph_path)
        .expect("save tombstone-aware receipt");
    assert_eq!(receipt.fsvi_physical_row_count, 2);
    assert_eq!(
        receipt.fsvi_whole_image_sha256,
        encode_lower_hex(owner.witness().whole_image_sha256)
    );

    let (loaded, loaded_receipt) = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
        .expect("load tombstone-aware graph");
    assert_eq!(loaded_receipt, receipt);
    assert_eq!(
        topology_snapshot(&loaded.graph),
        topology_snapshot(&bound.graph)
    );
    assert_eq!(
        loaded
            .search(&query, 8, Some(1))
            .expect("loaded bounded widening"),
        hits
    );
}

#[test]
fn ninety_percent_tombstones_widen_to_exact_live_cardinality_in_stable_order() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(40, 0x40, "ninety-percent-tombstones", 8);
    let corpus = TestStore::synthetic(20, 8);
    let rows: Vec<(String, Vec<f32>, FsviRecordFlags)> = corpus
        .vectors
        .into_iter()
        .enumerate()
        .map(|(row, vector)| {
            (
                format!("ninety-percent-{row:02}"),
                vector,
                if row < 2 {
                    FsviRecordFlags::LIVE
                } else {
                    FsviRecordFlags::TOMBSTONE
                },
            )
        })
        .collect();
    let owner = admitted_fsvi_owner_with_flags(
        directory.path(),
        "ninety-percent.fsvi",
        &fsvi_binding,
        &rows,
    );
    assert_eq!(owner.record_count(), 20);
    assert_eq!(owner.live_count(), 2);
    assert_eq!(owner.tombstone_count(), 18);

    let (seed, bound) = (0..10_000_u64)
        .find_map(|seed| {
            let entry = sampled_entry_for_count(owner.record_count(), seed)?;
            if owner
                .row(usize::try_from(entry).ok()?)
                .ok()?
                .flags()
                .is_live()
            {
                return None;
            }
            ValidatedNativeHnsw::build(Arc::clone(&owner), params(), seed)
                .ok()
                .map(|bound| (seed, bound))
        })
        .expect("find a sound graph with a tombstoned entry point");
    let entry = bound.graph.entry_point().expect("nonempty graph entry");
    assert!(
        owner
            .row(usize::try_from(entry).expect("entry fits usize"))
            .expect("entry row")
            .flags()
            .is_tombstone(),
        "the graph entry itself must exercise tombstone routing"
    );

    // ubs:ignore -- a public test query must not taint unrelated comparisons.
    let query = owner
        .vector_at_f32(usize::try_from(entry).expect("entry fits usize"))
        .expect("query equal to the tombstoned entry vector");
    let first = bound
        .search(&query, usize::MAX, Some(1))
        .expect("bounded widening reaches every required live row");
    let second = bound
        .search(&query, usize::MAX, Some(1))
        .expect("repeat bounded widening is deterministic");
    assert_eq!(first, second);
    assert_eq!(first.len(), owner.live_count());
    assert!(first.iter().all(|hit| hit.flags().is_live()));
    assert!(
        first
            .iter()
            .all(|hit| hit.doc_id().starts_with("ninety-percent-0")),
        "only the two live document ids may cross the boundary: {first:?}"
    );

    let expected_live_rows: Vec<u32> = (0..owner.record_count())
        .filter(|&index| owner.row(index).expect("owner row").flags().is_live())
        .map(|index| u32::try_from(index).expect("physical row fits u32"))
        .collect();
    let mut returned_rows: Vec<u32> = first
        .iter()
        .map(ValidatedNativeHnswHit::physical_row)
        .collect();
    returned_rows.sort_unstable();
    assert_eq!(
        returned_rows, expected_live_rows,
        "every live physical row must be returned exactly once"
    );
    assert_eq!(bound.graph.seed(), seed);

    let mut widths = vec![owner.live_count()];
    while *widths.last().expect("initial width") < owner.record_count() {
        widths.push(widen_search_width(
            *widths.last().expect("prior width"),
            owner.record_count(),
        ));
    }
    assert_eq!(widths, vec![2, 4, 8, 16, 20]);
    assert_eq!(
        widen_search_width(owner.record_count(), owner.record_count()),
        owner.record_count(),
        "the physical-row bound is a fixed point"
    );
}

#[test]
fn all_tombstoned_owner_and_zero_k_return_truthful_empty_results() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(41, 0x41, "all-tombstoned", 4);
    let rows: Vec<(String, Vec<f32>, FsviRecordFlags)> = (0..8)
        .map(|row| {
            let mut vector = vec![0.0; 4];
            vector[row % 4] = 1.0;
            (
                format!("all-dead-{row}"),
                vector,
                FsviRecordFlags::TOMBSTONE,
            )
        })
        .collect();
    let owner = admitted_fsvi_owner_with_flags(
        directory.path(),
        "all-tombstoned.fsvi",
        &fsvi_binding,
        &rows,
    );
    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 41)
        .expect("all physical tombstones remain valid routing nodes");

    assert_eq!(bound.len(), 8);
    assert_eq!(owner.live_count(), 0);
    assert_eq!(owner.tombstone_count(), 8);
    assert!(
        bound
            .search(&[1.0, 0.0, 0.0, 0.0], 8, Some(1))
            .expect("all-tombstoned search")
            .is_empty()
    );
    assert!(
        bound
            .search(&[1.0, 0.0, 0.0, 0.0], 0, Some(1))
            .expect("zero-k search")
            .is_empty()
    );
}

#[test]
fn physical_row_selection_never_deduplicates_equal_document_labels() {
    let store = TestStore::new(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let graph = NativeHnsw::build(params(), 42, &store).expect("build raw graph");
    let tombstoned = [false, true, false];
    let duplicate_labels = ["duplicate", "duplicate", "duplicate"];
    let hits = graph
        .search(&[0.0, 1.0, 0.0], 3, Some(3), &store)
        .expect("full-width physical-row search");
    let live_resolved: Vec<(u32, &str)> = hits
        .into_iter()
        .filter(|(physical_row, _)| {
            let physical_index = usize::try_from(*physical_row).expect("physical row fits usize");
            !tombstoned[physical_index]
        })
        .map(|(physical_row, _)| {
            let physical_index = usize::try_from(physical_row).expect("physical row fits usize");
            (physical_row, duplicate_labels[physical_index])
        })
        .collect();

    assert_eq!(live_resolved.len(), 2);
    assert_ne!(live_resolved[0].0, live_resolved[1].0);
    assert!(
        live_resolved
            .iter()
            // ubs:ignore -- this is a public document-label regression fixture.
            .all(|(_, doc_id)| *doc_id == "duplicate"),
        "equal document labels must not collapse distinct physical rows"
    );

    // Identity-complete FSVI v2 currently requires unique document ids,
    // so an owner-bound public duplicate-id state cannot be constructed.
    // Pin that upstream admission boundary explicitly instead of
    // weakening it merely to manufacture an HNSW fixture.
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(42, 0x42, "duplicate-doc-ids", 3);
    let fsvi_path = directory.path().join("duplicate-doc-ids.fsvi");
    let mut writer =
        VectorIndex::create_v2(&fsvi_path, fsvi_binding).expect("create duplicate-id fixture");
    writer
        .write_record("duplicate", &[1.0, 0.0, 0.0])
        .expect("first duplicate row");
    writer
        .write_record("duplicate", &[0.0, 1.0, 0.0])
        .expect("second duplicate row is buffered before canonical admission");
    let error = writer
        .finish()
        .expect_err("FSVI v2 must reject duplicate document ids before HNSW admission");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig {
                ref field,
                ref value,
                ref reason,
            // ubs:ignore -- public schema field name, not secret material.
            } if field == "doc_id"
                // ubs:ignore -- public duplicate marker, not secret material.
                && value == "<duplicate>"
                && reason.contains("unique physical row per document id")
        ),
        "unexpected duplicate-id owner rejection: {error:?}"
    );
}

#[test]
fn full_width_live_underfill_is_a_typed_graph_defect() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fsvi_binding = fsvi_v2_binding(43, 0x43, "typed-live-underfill", 4);
    let rows: Vec<(String, Vec<f32>, FsviRecordFlags)> = (0..8)
        .map(|row| {
            // ubs:ignore -- deterministic fixture row index, not secret material.
            let live = row == 0;
            let mut vector = vec![0.0; 4];
            vector[row % 4] = 1.0;
            (
                format!("typed-underfill-{row}"),
                vector,
                if live {
                    FsviRecordFlags::LIVE
                } else {
                    FsviRecordFlags::TOMBSTONE
                },
            )
        })
        .collect();
    let owner = admitted_fsvi_owner_with_flags(
        directory.path(),
        "typed-live-underfill.fsvi",
        &fsvi_binding,
        &rows,
    );
    let seed = seed_with_tombstoned_entry(&owner);
    let mut bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), seed)
        .expect("build structurally valid owner-bound graph");
    let live_row = (0..owner.record_count())
        .find(|&index| owner.row(index).expect("owner row").flags().is_live())
        .expect("one live row");
    let live_id = u32::try_from(live_row).expect("live row fits u32");
    assert_ne!(bound.graph.entry_point(), Some(live_id));

    // Forge an impossible post-admission reachability defect. Public
    // build/load cannot create this state because both structurally
    // attest first; the mutation proves search reports a typed defect if
    // the invariant is ever violated rather than returning zero hits.
    for point in &mut bound.graph.adjacency {
        for neighbours in &mut point.layers {
            // ubs:ignore -- public graph row id, not secret material.
            neighbours.retain(|&candidate| candidate != live_id);
        }
    }
    for neighbours in &mut bound.graph.adjacency[live_row].layers {
        neighbours.clear();
    }

    // ubs:ignore -- a public test query must not taint unrelated comparisons.
    let query = owner.vector_at_f32(live_row).expect("live query vector");
    let error = bound
        .search(&query, 1, Some(1))
        .expect_err("full-width underfill must fail closed");
    assert!(
        matches!(&error, SearchError::SubsystemError { .. }),
        "expected typed native-hnsw subsystem defect, got {error:?}"
    );
    let SearchError::SubsystemError { subsystem, source } = error else {
        return;
    };
    assert_eq!(subsystem, "native-hnsw");
    assert_eq!(
        source.downcast_ref::<GraphDefect>(),
        Some(&GraphDefect::LiveResultUnderfill {
            requested_k: 1,
            expected_live_hits: 1,
            returned_live_hits: 0,
            physical_rows: 8,
        })
    );
}

#[test]
fn validated_owner_bound_search_rejects_invalid_queries_before_zero_result_paths() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let nonempty_binding = fsvi_v2_binding(37, 0x37, "query-validation-nonempty", 4);
    let nonempty_rows = fsvi_rows(8, 4);
    let nonempty_owner = admitted_fsvi_owner(
        directory.path(),
        "query-validation-nonempty.fsvi",
        &nonempty_binding,
        &nonempty_rows,
    );
    let nonempty = ValidatedNativeHnsw::build(Arc::clone(&nonempty_owner), params(), 37)
        .expect("build nonempty owner-bound graph");

    let wrong_dimension = nonempty
        .search(&[1.0, 0.0, 0.0], 0, None)
        .expect_err("zero-k must not bypass query-dimension validation");
    assert!(matches!(
        wrong_dimension,
        SearchError::DimensionMismatch {
            expected: 4,
            found: 3
        }
    ));
    for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = nonempty
            .search(&[non_finite, 0.0, 0.0, 0.0], 0, None)
            .expect_err("zero-k must not bypass non-finite query validation");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig {
                    ref field,
                    ref value,
                    ref reason,
                } if field == "query"
                    && value == "non-finite"
                    && reason == "all query vector values must be finite"
            ),
            "unexpected non-finite query rejection: {error:?}"
        );
    }
    assert!(
        nonempty
            .search(&[1.0, 0.0, 0.0, 0.0], 0, None)
            .expect("valid zero-k query")
            .is_empty()
    );

    let empty_binding = fsvi_v2_binding(38, 0x38, "query-validation-empty", 4);
    let empty_owner = admitted_fsvi_owner(
        directory.path(),
        "query-validation-empty.fsvi",
        &empty_binding,
        &[],
    );
    let empty = ValidatedNativeHnsw::build(Arc::clone(&empty_owner), params(), 38)
        .expect("build empty owner-bound graph");

    let wrong_dimension = empty
        .search(&[1.0, 0.0, 0.0], 1, None)
        .expect_err("empty graph must not bypass query-dimension validation");
    assert!(matches!(
        wrong_dimension,
        SearchError::DimensionMismatch {
            expected: 4,
            found: 3
        }
    ));
    for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = empty
            .search(&[non_finite, 0.0, 0.0, 0.0], 1, None)
            .expect_err("empty graph must not bypass non-finite query validation");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig {
                    ref field,
                    ref value,
                    ref reason,
                } if field == "query"
                    && value == "non-finite"
                    && reason == "all query vector values must be finite"
            ),
            "unexpected non-finite empty-graph rejection: {error:?}"
        );
    }
    assert!(
        empty
            .search(&[1.0, 0.0, 0.0, 0.0], 1, None)
            .expect("valid empty-graph query")
            .is_empty()
    );
}
