use std::path::Path;

use crate::VectorIndex;

use super::super::fingerprint::{
    fingerprint_doc_ids, fingerprint_live_vector_index, fingerprint_source_map,
};
use super::super::types::{
    HNSW_GENERATION_RECEIPT_FILENAME, HnswAppendDisposition, HnswConfig, HnswGenerationReceipt,
    HnswIndex, HnswLoadDisposition, HnswMeta, HnswSourceIdentityV1,
};
use super::{
    normalized_vector, ready_generation_paths, reject_hnsw_metadata_publish, temp_path, write_index,
};

fn append_test_source(
    path: &Path,
    sequence: u64,
    model: &str,
    rows: &[(String, Vec<f32>)],
) -> crate::ValidatedFsviBytes {
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model, 32);
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = QuantizationFormat::F32;
    identity.storage.endianness = "little-endian".to_owned();
    let binding = crate::FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(sequence, [0xb1; 16]).expect("generation"),
        identity.freeze().expect("identity"),
    )
    .expect("binding");
    let mut writer = VectorIndex::create_v2(path, binding.clone()).expect("create source");
    for (id, vector) in rows {
        writer.write_record(id, vector).expect("write row");
    }
    writer.finish().expect("finish source");
    crate::ValidatedFsviBytes::from_arc(
        std::sync::Arc::<[u8]>::from(std::fs::read(path).expect("source bytes")),
        &binding,
    )
    .expect("validated source")
}

#[test]
fn incremental_append_loaded_graph_preserves_nodes_and_searches_after_reload() {
    let dir = tempfile::tempdir().expect("test directory");
    let mut rows: Vec<_> = (0..33)
        .map(|row| (format!("doc-{row:04}"), normalized_vector(row + 1, 32)))
        .collect();
    rows.sort_by_key(|(id, _)| crate::fnv1a_hash(id.as_bytes()));
    // FSVI sorts by hash. Insert before every old physical row, forcing
    // remapping while all existing graph origin IDs remain unchanged.
    let (new_id, query) = rows.remove(0);
    let prior = append_test_source(&dir.path().join("prior.fsvi"), 1, "append-model", &rows);
    let ann_path = dir.path().join("ann.json");
    HnswIndex::build_from_vector_index(&prior.index, HnswConfig::default())
        .expect("build prior graph")
        .save(&ann_path)
        .expect("save prior graph");
    let (loaded, load) =
        HnswIndex::load_with_disposition(&ann_path, &prior.index).expect("native prior load");
    assert_eq!(load, HnswLoadDisposition::Native);
    let old_points: Vec<_> = loaded.hnsw.get_point_indexation().into_iter().collect();
    let (loaded, noop) = loaded
        .append_from_vector_index(&prior.index, &prior.index)
        .expect("no-change append");
    assert_eq!(noop, HnswAppendDisposition::NoOp);
    rows.push((new_id.clone(), query.clone()));
    let current = append_test_source(&dir.path().join("current.fsvi"), 2, "append-model", &rows);
    let (appended, disposition) = loaded
        .append_from_vector_index(&prior.index, &current.index)
        .expect("append to loaded graph");
    assert_eq!(disposition, HnswAppendDisposition::Appended { inserted: 1 });
    assert_eq!(appended.source_positions[0], 1);
    assert_eq!(appended.source_positions[32], 0);
    for old in &old_points {
        let retained = appended
            .hnsw
            .get_point_indexation()
            .into_iter()
            .find(|point| point.get_origin_id() == old.get_origin_id())
            .expect("retained original point");
        assert!(
            std::sync::Arc::ptr_eq(old, &retained),
            "must reuse old graph nodes"
        );
    }
    assert!(
        appended
            .matches_vector_index(&current.index)
            .expect("current match")
    );
    assert!(
        !appended
            .matches_vector_index(&prior.index)
            .expect("prior mismatch")
    );
    let (hits, stats) = appended
        .knn_search_with_stats(&query, 1, 128)
        .expect("ANN query");
    assert_eq!(hits[0].doc_id, new_id);
    assert!(stats.is_approximate);
    assert_eq!(stats.fallback_reason, None);
    let (source_hits, source_stats) = appended
        .knn_search_with_stats_against(&current.index, &query, 1, 128)
        .expect("mapped source ANN");
    assert_eq!(source_hits[0].index, 0);
    assert!(source_stats.is_approximate);
    appended.save(&ann_path).expect("publish appended graph");
    let (reloaded, load) =
        HnswIndex::load_with_disposition(&ann_path, &current.index).expect("reload appended graph");
    assert_eq!(load, HnswLoadDisposition::Native);
    assert_eq!(reloaded.source_positions, appended.source_positions);
    let (hits, stats) = reloaded
        .knn_search_with_stats(&query, 1, 128)
        .expect("reloaded ANN");
    assert_eq!(hits[0].doc_id, new_id);
    assert!(stats.is_approximate);
    assert_eq!(stats.fallback_reason, None);
    assert!(
        HnswIndex::try_load_native(&ann_path, &prior.index)
            .expect("old admission")
            .is_none()
    );
    // Repeat from a natively loaded, already non-identity row map. A
    // one-shot remapping implementation must not fail on the next delta.
    let retained_points: Vec<_> = reloaded.hnsw.get_point_indexation().into_iter().collect();
    let (reloaded, disposition) = reloaded
        .append_from_vector_index(&current.index, &current.index)
        .expect("mapped no-op");
    assert_eq!(disposition, HnswAppendDisposition::NoOp);
    let second_query = normalized_vector(101, 32);
    rows.push(("second-new-doc".to_owned(), second_query.clone()));
    let next = append_test_source(&dir.path().join("next.fsvi"), 3, "append-model", &rows);
    let (next_graph, disposition) = reloaded
        .append_from_vector_index(&current.index, &next.index)
        .expect("second mapped append");
    assert_eq!(disposition, HnswAppendDisposition::Appended { inserted: 1 });
    for point in retained_points {
        assert!(
            next_graph
                .hnsw
                .get_point_indexation()
                .into_iter()
                .any(|p| std::sync::Arc::ptr_eq(&point, &p))
        );
    }
    next_graph.save(&ann_path).expect("save second append");
    let (next_loaded, disposition) =
        HnswIndex::load_with_disposition(&ann_path, &next.index).expect("load second append");
    assert_eq!(disposition, HnswLoadDisposition::Native);
    let (hits, stats) = next_loaded
        .knn_search_with_stats_against(&next.index, &second_query, 1, 128)
        .expect("second appended ANN");
    assert_eq!(hits[0].doc_id, "second-new-doc");
    assert_eq!(
        next.index
            .doc_id_at(hits[0].index as usize)
            .expect("physical row"),
        "second-new-doc"
    );
    assert!(stats.is_approximate);
    assert_eq!(stats.fallback_reason, None);
}

#[test]
fn incremental_append_refuses_changes_and_preserves_prior_graph() {
    let dir = tempfile::tempdir().expect("test directory");
    let mut rows: Vec<_> = (0..9)
        .map(|row| (format!("doc-{row:04}"), normalized_vector(row + 1, 32)))
        .collect();
    rows.sort_by_key(|(id, _)| crate::fnv1a_hash(id.as_bytes()));
    let suffix = rows.pop().expect("suffix row");
    let prior = append_test_source(&dir.path().join("prior.fsvi"), 1, "append-model", &rows);
    let ann_path = dir.path().join("ann.json");
    HnswIndex::build_from_vector_index(&prior.index, HnswConfig::default())
        .expect("build graph")
        .save(&ann_path)
        .expect("save graph");
    let persisted = std::fs::read(&ann_path).expect("prior metadata");
    for case in ["changed", "deleted", "replaced", "model", "generation"] {
        let mut changed = rows.clone();
        changed.push(suffix.clone());
        match case {
            "changed" => changed[2].1 = normalized_vector(102, 32),
            "deleted" => {
                changed.remove(2);
                changed.push(("extra-new-doc".to_owned(), normalized_vector(103, 32)));
            }
            "replaced" => changed[2].0 = "replacement-id".to_owned(),
            _ => {}
        }
        let current = append_test_source(
            &dir.path().join(format!("{case}.fsvi")),
            if case == "generation" { 1 } else { 2 },
            if case == "model" {
                "other-model"
            } else {
                "append-model"
            },
            &changed,
        );
        let (loaded, disposition) =
            HnswIndex::load_with_disposition(&ann_path, &prior.index).expect("load original graph");
        assert_eq!(disposition, HnswLoadDisposition::Native);
        let old_points: Vec<_> = loaded.hnsw.get_point_indexation().into_iter().collect();
        let (unchanged, disposition) = loaded
            .append_from_vector_index(&prior.index, &current.index)
            .expect("refuse delta");
        let expected_reason = match case {
            "deleted" => "old_document_missing",
            "model" => "embedding_identity_changed",
            "generation" => "generation_not_advanced",
            "changed" => "old_vector_changed",
            _ => "old_document_missing",
        };
        assert_eq!(
            disposition,
            HnswAppendDisposition::FullRebuildRequired {
                reason: expected_reason
            },
            "{case}"
        );
        assert!(
            unchanged
                .matches_vector_index(&prior.index)
                .expect("prior preserved")
        );
        assert!(
            !unchanged
                .matches_vector_index(&current.index)
                .expect("current rejected")
        );
        assert_eq!(unchanged.hnsw.get_nb_point(), rows.len());
        for old in &old_points {
            assert!(
                unchanged
                    .hnsw
                    .get_point_indexation()
                    .into_iter()
                    .any(|p| std::sync::Arc::ptr_eq(old, &p))
            );
        }
        assert_eq!(
            std::fs::read(&ann_path).expect("unchanged metadata"),
            persisted
        );
    }
}

#[test]
fn incremental_append_native_load_rejects_invalid_source_maps() {
    let dir = tempfile::tempdir().expect("test directory");
    let rows: Vec<_> = (0..8)
        .map(|row| (format!("doc-{row:04}"), normalized_vector(row + 1, 32)))
        .collect();
    let source = append_test_source(&dir.path().join("source.fsvi"), 1, "append-model", &rows);
    let ann_path = dir.path().join("ann.json");
    HnswIndex::build_from_vector_index(&source.index, HnswConfig::default())
        .expect("build graph")
        .save(&ann_path)
        .expect("save graph");
    let original_meta = std::fs::read(&ann_path).expect("metadata");
    let pristine: HnswMeta = serde_json::from_slice(&original_meta).expect("parse metadata");
    let receipt_path = dir
        .path()
        .join(pristine.sidecar_generation.as_ref().expect("generation"))
        .join(HNSW_GENERATION_RECEIPT_FILENAME);
    let original_receipt = std::fs::read(&receipt_path).expect("receipt");
    for case in [
        "reordered",
        "duplicate",
        "outside",
        "missing",
        "extent",
        "v6",
    ] {
        let mut meta: HnswMeta = serde_json::from_slice(&original_meta).expect("metadata");
        match case {
            "reordered" => meta.source_positions.swap(0, 1),
            "duplicate" => meta.source_positions[1] = meta.source_positions[0],
            "outside" => meta.source_positions[0] = u32::MAX,
            "missing" => {
                meta.source_positions.pop();
            }
            "v6" => meta.format_version = 6,
            _ => meta.source_record_count += 1,
        }
        std::fs::write(&ann_path, serde_json::to_vec(&meta).expect("encode map"))
            .expect("plant invalid map");
        // First reject metadata that diverges from its unchanged receipt.
        std::fs::write(&receipt_path, &original_receipt).expect("restore receipt");
        assert!(
            HnswIndex::try_load_native(&ann_path, &source.index)
                .expect("native admission")
                .is_none(),
            "{case}"
        );
        // Even a matching map digest cannot authorize invalid coverage or
        // wrong row identities against the real source generation.
        let mut receipt: HnswGenerationReceipt =
            serde_json::from_slice(&original_receipt).expect("parse receipt");
        receipt.source_map_fingerprint =
            fingerprint_source_map(&meta.source_positions, meta.source_record_count);
        std::fs::write(
            &receipt_path,
            serde_json::to_vec(&receipt).expect("encode receipt"),
        )
        .expect("plant matching digest");
        assert!(
            HnswIndex::try_load_native(&ann_path, &source.index)
                .expect("source admission")
                .is_none(),
            "{case}"
        );
    }
}

// ─── bd-r65a: ANN sidecars are bound to the source FSVI generation ───

/// Two v2 generations holding byte-identical live content, differing only
/// in their published generation identity.
fn identity_bound_pair() -> (crate::ValidatedFsviBytes, crate::ValidatedFsviBytes) {
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use std::sync::Arc;

    let make = |sequence: u64, nonce: u8, tag: &str| {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("r65a-ann-source", 8);
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F32;
        identity.storage.endianness = "little-endian".to_owned();
        let binding = crate::FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(sequence, [nonce; 16])
                .expect("valid test generation"),
            identity.freeze().expect("valid frozen identity"),
        )
        .expect("valid FSVI v2 binding");
        let path = temp_path(tag, "fsvi");
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("create v2 source");
        for row in 0..6_usize {
            writer
                .write_record(&format!("doc-{row:04}"), &normalized_vector(row + 1, 8))
                .expect("write v2 row");
        }
        writer.finish().expect("finish v2 source");
        let bytes = std::fs::read(&path).expect("read v2 source");
        let _ = std::fs::remove_file(&path);
        crate::ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), &binding)
            .expect("admit v2 source")
    };

    (
        make(41, 0xa1, "r65a-generation-a"),
        make(42, 0xa2, "r65a-generation-b"),
    )
}

/// Planted negative: a stale sidecar whose CONTENT matches perfectly.
///
/// Same path shape, same dimension, same doc ids in the same order, and —
/// asserted below — the same vector fingerprint, so every content check
/// the loader performs says "this graph belongs to this index". Only the
/// published generation identity differs. An implementation that bound
/// sidecars to content alone would serve the stale graph.
#[test]
fn a_sidecar_is_refused_against_a_content_identical_different_generation() {
    let (owner_a, owner_b) = identity_bound_pair();
    let index_a = &owner_a.index;
    let index_b = &owner_b.index;

    // The content checks cannot tell these apart.
    assert_eq!(index_a.dimension(), index_b.dimension());
    assert_eq!(index_a.record_count(), index_b.record_count());
    let doc_ids_a: Vec<String> = (0..index_a.record_count())
        .map(|row| index_a.doc_id_at(row).expect("doc id").to_owned())
        .collect();
    let doc_ids_b: Vec<String> = (0..index_b.record_count())
        .map(|row| index_b.doc_id_at(row).expect("doc id").to_owned())
        .collect();
    assert_eq!(doc_ids_a, doc_ids_b, "the fixture must share its doc set");
    let fingerprint_a = fingerprint_live_vector_index(index_a, doc_ids_a.len(), 8)
        .expect("fingerprint generation a");
    let fingerprint_b = fingerprint_live_vector_index(index_b, doc_ids_b.len(), 8)
        .expect("fingerprint generation b");
    assert_eq!(
        fingerprint_a, fingerprint_b,
        "the planted negative requires an identical vector fingerprint; \
         otherwise the content guard alone would catch it and prove nothing"
    );

    // Only the identity differs.
    assert_ne!(
        owner_a.identity_v2().generation,
        owner_b.identity_v2().generation
    );

    let ann = HnswIndex::build_from_vector_index(index_a, HnswConfig::default())
        .expect("build from generation a");
    assert!(
        ann.source_identity.is_some(),
        "a graph built from a v2 source must record its identity"
    );
    assert!(
        ann.matches_vector_index(index_a).expect("match against a"),
        "the graph must still match the generation it was built from"
    );
    assert!(
        !ann.matches_vector_index(index_b).expect("match against b"),
        "a graph built from generation a must NOT match generation b"
    );
}

/// The same refusal at the persisted-load boundary: the sidecar must be
/// rebuilt rather than served natively.
#[test]
fn a_persisted_sidecar_rebuilds_against_a_different_generation() {
    let (owner_a, owner_b) = identity_bound_pair();
    let metadata_path = temp_path("r65a-sidecar", "hnsw");
    let ann = HnswIndex::build_from_vector_index(&owner_a.index, HnswConfig::default())
        .expect("build from generation a");
    ann.save(&metadata_path).expect("persist sidecar");

    let (_, native) = HnswIndex::load_with_disposition(&metadata_path, &owner_a.index)
        .expect("load against its own generation");
    assert_eq!(
        native,
        HnswLoadDisposition::Native,
        "the sidecar must load natively against the generation that built it"
    );

    let (rebuilt, disposition) = HnswIndex::load_with_disposition(&metadata_path, &owner_b.index)
        .expect("load against a different generation");
    assert_eq!(
        disposition,
        HnswLoadDisposition::Rebuilt,
        "a different generation must force a rebuild, never a native serve"
    );
    assert_eq!(rebuilt.len(), owner_b.index.record_count());

    // The strict read-only API must refuse it outright rather than
    // reporting a native load.
    assert!(
        HnswIndex::try_load_native(&metadata_path, &owner_b.index)
            .expect("strict load call")
            .is_none(),
        "the strict native API must not report a cross-generation sidecar as native"
    );
}

/// Absence is not a match, in either direction: a sidecar with no recorded
/// identity cannot be proven to belong to an identity-bearing generation,
/// and vice versa. Without this, every pre-binding sidecar would be served
/// against any v2 source forever.
#[test]
fn unbound_and_bound_sidecars_never_admit_each_other() {
    let (owner_a, _) = identity_bound_pair();
    let bound = HnswSourceIdentityV1::capture(&owner_a.index).expect("v2 source is bound");

    assert!(HnswSourceIdentityV1::admits(None, None));
    assert!(HnswSourceIdentityV1::admits(Some(&bound), Some(&bound)));
    assert!(!HnswSourceIdentityV1::admits(None, Some(&bound)));
    assert!(!HnswSourceIdentityV1::admits(Some(&bound), None));
}

/// A legacy v1 source has no identity to bind, so the content checks stand
/// alone and existing behavior is preserved.
#[test]
fn a_legacy_v1_source_still_loads_natively() {
    let source_path = temp_path("r65a-legacy-source", "fsvi");
    let source = write_index(
        &source_path,
        &[normalized_vector(1, 8), normalized_vector(2, 8)],
    )
    .expect("legacy source");
    assert!(
        HnswSourceIdentityV1::capture(&source).is_none(),
        "a v1 source carries no identity"
    );
    let ann = HnswIndex::build_from_vector_index(&source, HnswConfig::default())
        .expect("build from legacy source");
    let metadata_path = temp_path("r65a-legacy-sidecar", "hnsw");
    ann.save(&metadata_path).expect("persist legacy sidecar");
    let (_, disposition) =
        HnswIndex::load_with_disposition(&metadata_path, &source).expect("load legacy sidecar");
    assert_eq!(disposition, HnswLoadDisposition::Native);
}

/// Publication-time identity re-validation (bd-21zyj).
///
/// Planted negative: a READY generation left on disk by a DIFFERENT
/// published FSVI generation whose content is indistinguishable from this
/// graph's. The test asserts the receipt agrees on doc-id fingerprint,
/// vector fingerprint, dimension and params — every field the reuse
/// validator checks — so the content guard provably cannot separate them.
/// Only the source generation identity differs. Without the identity gate
/// `save()` adopts that directory and publishes metadata naming THIS
/// graph's identity while pointing at the other generation's bytes.
#[test]
fn save_refuses_to_reuse_a_ready_generation_from_another_source_generation() {
    let (owner_a, owner_b) = identity_bound_pair();

    // Generation B publishes first and is then abandoned mid-publication,
    // leaving a durable READY generation on disk.
    let metadata_path = temp_path("bd21zyj-sidecar", "hnsw");
    let ann_b = HnswIndex::build_from_vector_index(&owner_b.index, HnswConfig::default())
        .expect("build from generation b");
    ann_b
        .save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
        .expect_err("retain an unpublished READY generation from generation b");
    let leftover = ready_generation_paths(&metadata_path, ann_b.vector_fingerprint);
    assert_eq!(
        leftover.len(),
        1,
        "generation b must leave exactly one READY generation"
    );
    let leftover_name = leftover[0]
        .file_name()
        .and_then(|name| name.to_str())
        .expect("leftover generation name")
        .to_owned();

    // Generation A's graph. Every CONTENT field the reuse validator checks
    // is identical, which is what makes this a planted negative rather
    // than an ordinary mismatch.
    let ann_a = HnswIndex::build_from_vector_index(&owner_a.index, HnswConfig::default())
        .expect("build from generation a");
    assert_eq!(
        ann_a.vector_fingerprint, ann_b.vector_fingerprint,
        "the planted negative requires an identical vector fingerprint"
    );
    assert_eq!(
        ann_a.doc_ids, ann_b.doc_ids,
        "and an identical ordered doc set"
    );
    assert_eq!(ann_a.dimension, ann_b.dimension);
    assert_eq!(ann_a.config, ann_b.config);
    let receipt: HnswGenerationReceipt = serde_json::from_slice(
        &std::fs::read(leftover[0].join(HNSW_GENERATION_RECEIPT_FILENAME))
            .expect("read leftover receipt"),
    )
    .expect("parse leftover receipt");
    assert_eq!(
        receipt.doc_ids_fingerprint,
        fingerprint_doc_ids(&ann_a.doc_ids)
    );
    assert_eq!(receipt.vector_fingerprint, ann_a.vector_fingerprint);
    assert_eq!(receipt.dimension, ann_a.dimension);
    assert_eq!(receipt.config, ann_a.config);
    // ...and only the identity differs.
    assert_ne!(receipt.source_identity, ann_a.source_identity);
    assert!(receipt.source_identity.is_some() && ann_a.source_identity.is_some());

    ann_a.save(&metadata_path).expect("publish generation a");
    let published: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read published metadata"))
            .expect("parse published metadata");
    assert_ne!(
        published.sidecar_generation.as_deref(),
        Some(leftover_name.as_str()),
        "publication must not adopt a READY generation dumped from another source generation"
    );
    assert_eq!(
        published.source_identity, ann_a.source_identity,
        "the published metadata must name the identity its bytes actually came from"
    );
}

/// Control: a leftover READY generation from the SAME source generation is
/// still reused, so the gate refuses foreign generations rather than
/// disabling reuse altogether.
#[test]
fn save_still_reuses_a_ready_generation_from_the_same_source_generation() {
    let (owner_a, _) = identity_bound_pair();
    let metadata_path = temp_path("bd21zyj-control", "hnsw");
    let ann = HnswIndex::build_from_vector_index(&owner_a.index, HnswConfig::default())
        .expect("build from generation a");
    ann.save_with_metadata_publisher(&metadata_path, reject_hnsw_metadata_publish)
        .expect_err("retain an unpublished READY generation");
    let leftover = ready_generation_paths(&metadata_path, ann.vector_fingerprint);
    assert_eq!(leftover.len(), 1);
    let leftover_name = leftover[0]
        .file_name()
        .and_then(|name| name.to_str())
        .expect("generation name")
        .to_owned();

    ann.save(&metadata_path)
        .expect("publish reusing the retained generation");
    let published: HnswMeta =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("read published metadata"))
            .expect("parse published metadata");
    assert_eq!(
        published.sidecar_generation.as_deref(),
        Some(leftover_name.as_str()),
        "a same-generation READY directory must still be reused"
    );
}
