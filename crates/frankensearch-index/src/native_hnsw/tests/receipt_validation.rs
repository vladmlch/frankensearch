use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;
use std::sync::Arc;

use frankensearch_core::error::SearchError;

use super::super::SHA256_BYTES;
use super::super::codec::NativeHnsw;
use super::super::io_helpers::put_u32;
use super::super::receipt::{NativeHnswGenerationReceiptV2, native_hnsw_generation_receipt_path};
use super::super::types::{GraphDefect, ValidatedNativeHnsw};
use super::{
    admitted_fsvi_owner, first_persisted_neighbour, fsvi_rows, fsvi_v2_binding, generation_owner,
    params, reseal_persisted_checksums, tempfile, write_resealed_receipt,
};

#[test]
fn validated_owner_bound_constructor_attests_graph_before_handle_escape() {
    fn assert_graph_defect(error: SearchError, expected: &GraphDefect) {
        assert!(
            matches!(&error, SearchError::SubsystemError { .. }),
            "expected native-hnsw subsystem error, got {error:?}"
        );
        if let SearchError::SubsystemError { subsystem, source } = error {
            assert_eq!(subsystem, "native-hnsw");
            assert_eq!(
                source.downcast_ref::<GraphDefect>(),
                Some(expected),
                "verified constructor must preserve the exact graph defect"
            );
        }
    }

    let owner = generation_owner(39, 0x39, "post-build-attestation", 4, 32);
    let binding =
        ValidatedNativeHnsw::binding_for_owner(&owner).expect("derive exact owner binding");
    let other_directory = tempfile::tempdir().expect("temporary other-owner directory");
    let shared_identity = fsvi_v2_binding(39, 0x39, "post-build-attestation", 4);
    let mut other_rows = fsvi_rows(32, 4);
    other_rows[0].1 = vec![0.5, 0.5, 0.0, 0.0];
    let other_owner = admitted_fsvi_owner(
        other_directory.path(),
        "other-owner.fsvi",
        &shared_identity,
        &other_rows,
    );
    let other_binding = ValidatedNativeHnsw::binding_for_owner(&other_owner)
        .expect("derive same-cardinality mismatched binding");
    assert_eq!(
        binding.artifact_generation,
        other_binding.artifact_generation
    );
    assert_eq!(
        binding.embedding_identity_fingerprint,
        other_binding.embedding_identity_fingerprint
    );
    assert_eq!(
        binding.embedding_space_fingerprint,
        other_binding.embedding_space_fingerprint
    );
    assert_eq!(
        binding.ordered_live_docset_digest,
        other_binding.ordered_live_docset_digest
    );
    assert_eq!(
        binding.fsvi_physical_row_count,
        other_binding.fsvi_physical_row_count
    );
    assert_ne!(
        binding.vector_content_digest,
        other_binding.vector_content_digest
    );

    let mut binding_mismatch_graph =
        NativeHnsw::build(params(), 39, owner.as_ref()).expect("build binding canary graph");
    binding_mismatch_graph
        .adjacency
        .pop()
        .expect("binding canary graph has a final row");
    let binding_error = ValidatedNativeHnsw::from_verified_graph(
        Arc::clone(&owner),
        other_binding,
        binding_mismatch_graph,
    )
    .expect_err("another owner's binding must not escape with this owner and graph");
    assert!(
        matches!(
            binding_error,
            SearchError::InvalidConfig {
                ref field,
                ref value,
                reason: _,
            } if field == "native_hnsw_receipt.generation_binding"
                && value == "redacted-mismatch"
        ),
        "unexpected mismatched-binding rejection: {binding_error:?}"
    );

    let sound = NativeHnsw::build(params(), 39, owner.as_ref()).expect("build sound graph");
    ValidatedNativeHnsw::from_verified_graph(Arc::clone(&owner), binding.clone(), sound)
        .expect("sound graph passes public-handle attestation");

    let mut one_way =
        NativeHnsw::build(params(), 39, owner.as_ref()).expect("build graph to corrupt");
    let (id, layer, neighbour) = one_way
        .adjacency
        .iter()
        .enumerate()
        .find_map(|(id, point)| {
            point
                .layers
                .iter()
                .enumerate()
                .find_map(|(layer, neighbours)| {
                    neighbours
                        .first()
                        .copied()
                        .map(|neighbour| (id, layer, neighbour))
                })
        })
        .expect("nontrivial graph has an edge");
    let id = u32::try_from(id).expect("test graph length fits u32");
    // ubs:ignore — graph point IDs are public topology data, not authenticators.
    one_way.adjacency[neighbour as usize].layers[layer].retain(|&candidate| candidate != id);
    let error =
        ValidatedNativeHnsw::from_verified_graph(Arc::clone(&owner), binding.clone(), one_way)
            .expect_err("one-way graph must not escape as a public handle");
    assert_graph_defect(
        error,
        &GraphDefect::MissingReciprocalEdge {
            id,
            layer,
            neighbour,
        },
    );

    let mut truncated =
        NativeHnsw::build(params(), 39, owner.as_ref()).expect("build graph to truncate");
    truncated.adjacency.pop().expect("graph has a final row");
    let error = ValidatedNativeHnsw::from_verified_graph(Arc::clone(&owner), binding, truncated)
        .expect_err("cardinality-mismatched graph must not escape as a public handle");
    assert_graph_defect(
        error,
        &GraphDefect::StoreCardinalityMismatch {
            graph_points: 31,
            store_rows: 32,
        },
    );
}

#[test]
fn legacy_or_missing_fsvi_identity_is_rejected_before_graph_bytes_are_read() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let graph_path = directory.path().join("legacy.fshnsw");
    let receipt_path = native_hnsw_generation_receipt_path(&graph_path).expect("receipt path");
    let owner = generation_owner(33, 0x33, "legacy-rejection", 4, 4);
    let bound =
        ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 33).expect("owner-bound graph");
    bound.save(&graph_path).expect("owner-bound save");
    let canonical = std::fs::read(&receipt_path).expect("canonical receipt");
    std::fs::write(&graph_path, b"graph bytes must not be read")
        .expect("poison graph after receipt publication");

    let mut legacy = canonical.clone();
    legacy[8..10].copy_from_slice(&1u16.to_be_bytes());
    std::fs::write(&receipt_path, legacy).expect("write schema-v1 receipt");
    let legacy_error = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
        .expect_err("schema-v1 receipt must fail");
    assert!(
        matches!(legacy_error, SearchError::IndexCorrupted { ref path, ref detail }
            if path == &receipt_path && detail.contains("schema 1")),
        "legacy receipt did not fail before graph access: {legacy_error:?}"
    );

    let identity_prefix_len = 8 + 2 + 2 + 8 + 16 + (6 * SHA256_BYTES);
    std::fs::write(&receipt_path, &canonical[..identity_prefix_len])
        .expect("write receipt missing FSVI content identity");
    let missing_error = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
        .expect_err("missing FSVI content identity must fail");
    assert!(
        matches!(missing_error, SearchError::IndexCorrupted { ref path, ref detail }
            if path == &receipt_path && detail.contains("vector content digest")),
        "missing identity did not fail before graph access: {missing_error:?}"
    );

    let whole_image_prefix_len = identity_prefix_len + (2 * SHA256_BYTES);
    std::fs::write(&receipt_path, &canonical[..whole_image_prefix_len])
        .expect("write receipt missing whole-image witness");
    let missing_whole_image = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
        .expect_err("missing whole-image witness must fail");
    assert!(
        matches!(missing_whole_image, SearchError::IndexCorrupted { ref path, ref detail }
            if path == &receipt_path && detail.contains("whole-image")),
        "missing whole-image witness did not fail before graph access: {missing_whole_image:?}"
    );
}

#[test]
fn save_rejects_physical_row_count_mismatch_before_graph_publication() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let graph_path = directory.path().join("count-mismatch.fshnsw");
    let receipt_path = native_hnsw_generation_receipt_path(&graph_path).expect("receipt path");
    let owner = generation_owner(34, 0x34, "count-mismatch", 4, 5);
    let mut bound =
        ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 34).expect("owner-bound graph");
    bound.graph.adjacency.pop().expect("test graph row");

    let error = bound
        .save(&graph_path)
        .expect_err("physical row count mismatch must fail");
    assert!(
        matches!(error, SearchError::InvalidConfig { ref field, .. }
            // ubs:ignore — this public error-field schema label is not secret material.
            if field == "native_hnsw_receipt.graph.point_count"),
        "unexpected count mismatch error: {error:?}"
    );
    assert!(!graph_path.exists());
    assert!(!receipt_path.exists());
}

#[test]
fn bound_load_rejects_missing_truncated_tampered_and_noncanonical_receipts() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let graph_path = directory.path().join("graph.fshnsw");
    let receipt_path = native_hnsw_generation_receipt_path(&graph_path).expect("receipt path");
    let owner = generation_owner(9, 9, "receipt-corruption", 6, 32);
    let bound =
        ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 0x99).expect("owner-bound graph");

    bound.graph.save(&graph_path).expect("unbound graph save");
    let missing = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
        .expect_err("missing receipt must fail");
    assert!(
        // ubs:ignore — receipt paths are public filesystem identities, not credentials.
        matches!(missing, SearchError::IndexNotFound { ref path } if path == &receipt_path)
    );

    bound.save(&graph_path).expect("owner-bound save");
    let canonical = std::fs::read(&receipt_path).expect("canonical receipt");

    std::fs::write(&receipt_path, &canonical[..canonical.len() - 1]).expect("truncate receipt");
    assert!(matches!(
        ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
            .expect_err("truncated receipt must fail"),
        SearchError::IndexCorrupted { .. }
    ));

    let mut body_tamper = canonical.clone();
    body_tamper[20] ^= 1;
    std::fs::write(&receipt_path, body_tamper).expect("tamper receipt body");
    assert!(
        ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path).is_err(),
        "body tamper without a new SHA-256 seal must fail"
    );

    let mut seal_tamper = canonical.clone();
    *seal_tamper.last_mut().expect("receipt seal byte") ^= 1;
    std::fs::write(&receipt_path, seal_tamper).expect("tamper receipt seal");
    assert!(
        ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path).is_err(),
        "receipt SHA-256 tamper must fail"
    );

    let mut trailing = canonical;
    trailing.push(0);
    std::fs::write(&receipt_path, trailing).expect("append noncanonical byte");
    let error = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
        .expect_err("trailing receipt bytes must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("trailing bytes") || detail.contains("canonically encoded")),
        "unexpected noncanonical receipt error: {error:?}"
    );
}

#[test]
fn bound_load_rejects_swapped_stale_resealed_and_replaced_graphs() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let first_path = directory.path().join("first.fshnsw");
    let second_path = directory.path().join("second.fshnsw");
    let first_receipt = native_hnsw_generation_receipt_path(&first_path).expect("first receipt");
    let second_receipt = native_hnsw_generation_receipt_path(&second_path).expect("second receipt");
    let owner = generation_owner(11, 0x11, "swap-stale", 6, 40);
    let first = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 11)
        .expect("first owner-bound graph");
    let second = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 29)
        .expect("second owner-bound graph");

    first.save(&first_path).expect("save first");
    second.save(&second_path).expect("save second");
    let first_receipt_bytes = std::fs::read(&first_receipt).expect("first receipt bytes");
    let second_receipt_bytes = std::fs::read(&second_receipt).expect("second receipt bytes");
    std::fs::write(&first_receipt, &second_receipt_bytes).expect("swap second into first");
    std::fs::write(&second_receipt, &first_receipt_bytes).expect("swap first into second");
    assert!(
        ValidatedNativeHnsw::load(Arc::clone(&owner), &first_path).is_err(),
        "receipt from a different basename/artifact must fail"
    );
    assert!(
        ValidatedNativeHnsw::load(Arc::clone(&owner), &second_path).is_err(),
        "the reciprocal receipt swap must also fail"
    );

    std::fs::write(&first_receipt, &first_receipt_bytes).expect("restore first receipt");
    second
        .graph
        .save(&first_path)
        .expect("replace graph without receipt");
    let stale = ValidatedNativeHnsw::load(Arc::clone(&owner), &first_path)
        .expect_err("stale receipt must fail");
    assert!(
        matches!(stale, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("SHA-256") || detail.contains("receipt")),
        "unexpected stale-receipt error: {stale:?}"
    );

    first.save(&first_path).expect("restore bound first");
    let mut forged_graph = std::fs::read(&first_path).expect("graph bytes");
    let (neighbour_offset, edge_owner) =
        first_persisted_neighbour(&forged_graph).expect("graph edge");
    put_u32(&mut forged_graph, neighbour_offset, edge_owner);
    reseal_persisted_checksums(&mut forged_graph);
    std::fs::write(&first_path, forged_graph).expect("write resealed graph forgery");
    let resealed = ValidatedNativeHnsw::load(Arc::clone(&owner), &first_path)
        .expect_err("resealed graph replacement must fail receipt SHA-256");
    assert!(
        matches!(resealed, SearchError::IndexCorrupted { ref detail, .. }
            if detail.contains("SHA-256")),
        "unexpected resealed graph error: {resealed:?}"
    );
}

#[test]
fn bound_load_rejects_replacement_between_hash_and_parse() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let graph_path = directory.path().join("race.fshnsw");
    let owner = generation_owner(3, 3, "replacement-race", 6, 48);
    let original = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 3)
        .expect("original owner-bound graph");
    let replacement = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 5)
        .expect("replacement owner-bound graph");
    original.save(&graph_path).expect("save original binding");

    let error = ValidatedNativeHnsw::load_with_after_first_observation(
        Arc::clone(&owner),
        &graph_path,
        || replacement.graph.save(&graph_path).map(|_| ()),
    )
    .expect_err("replacement during verification must fail");
    assert!(
        matches!(error, SearchError::IndexCorrupted { .. }),
        "replacement race was not rejected: {error:?}"
    );
}

#[test]
fn every_resealed_receipt_field_is_checked_against_trust_or_graph_state() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let graph_path = directory.path().join("all-fields.fshnsw");
    let receipt_path = native_hnsw_generation_receipt_path(&graph_path).expect("receipt path");
    let owner = generation_owner(77, 0x77, "every-receipt-field", 8, 48);
    let bound = ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 0xfeed)
        .expect("owner-bound graph");
    let original = bound.save(&graph_path).expect("owner-bound save");

    macro_rules! rejects_resealed {
        ($label:literal, $change:expr) => {{
            let mut changed = original.clone();
            ($change)(&mut changed);
            write_resealed_receipt(&receipt_path, &mut changed);
            let error = ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path)
                .expect_err(concat!($label, " drift must fail"));
            assert!(
                matches!(
                    error,
                    SearchError::IndexCorrupted { .. } | SearchError::InvalidConfig { .. }
                ),
                "{} drift returned an unexpected error: {error:?}",
                $label
            );
        }};
    }

    rejects_resealed!("receipt schema", |r: &mut NativeHnswGenerationReceiptV2| {
        r.schema_version += 1;
    });
    rejects_resealed!(
        "generation schema",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.artifact_generation.schema_version += 1;
            r.artifact_generation_fingerprint = r.artifact_generation.fingerprint();
        }
    );
    rejects_resealed!(
        "generation sequence",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.artifact_generation.sequence += 1;
            r.artifact_generation_fingerprint = r.artifact_generation.fingerprint();
        }
    );
    rejects_resealed!(
        "generation nonce",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.artifact_generation.nonce[0] ^= 1;
            r.artifact_generation_fingerprint = r.artifact_generation.fingerprint();
        }
    );
    rejects_resealed!(
        "generation fingerprint",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.artifact_generation_fingerprint = "0".repeat(64);
        }
    );
    rejects_resealed!(
        "embedding identity fingerprint",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.embedding_identity_fingerprint = "1".repeat(64);
        }
    );
    rejects_resealed!(
        "embedding space fingerprint",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.embedding_space_fingerprint = "2".repeat(64);
        }
    );
    rejects_resealed!(
        "embedding producer fingerprint",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.embedding_producer_fingerprint = "3".repeat(64);
        }
    );
    rejects_resealed!(
        "embedding input fingerprint",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.embedding_input_fingerprint = "4".repeat(64);
        }
    );
    rejects_resealed!(
        "vector storage fingerprint",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.vector_storage_fingerprint = "5".repeat(64);
        }
    );
    rejects_resealed!(
        "vector content digest",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.vector_content_digest = "8".repeat(64);
        }
    );
    rejects_resealed!(
        "ordered live-docset digest",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.ordered_live_docset_digest = "9".repeat(64);
        }
    );
    rejects_resealed!(
        "FSVI whole-image SHA-256",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.fsvi_whole_image_sha256 = "a".repeat(64);
        }
    );
    rejects_resealed!(
        "FSVI physical row count",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.fsvi_physical_row_count += 1;
        }
    );
    rejects_resealed!("graph basename", |r: &mut NativeHnswGenerationReceiptV2| {
        r.graph_basename = "other.fshnsw".to_owned();
    });
    rejects_resealed!(
        "graph byte length",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.graph_byte_len += 1;
        }
    );
    rejects_resealed!("graph SHA-256", |r: &mut NativeHnswGenerationReceiptV2| {
        r.graph_sha256 = "6".repeat(64);
    });
    rejects_resealed!(
        "native format version",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.native_format_version += 1;
        }
    );
    rejects_resealed!("parameter m", |r: &mut NativeHnswGenerationReceiptV2| {
        r.params.m += 1;
    });
    rejects_resealed!("parameter m0", |r: &mut NativeHnswGenerationReceiptV2| {
        r.params.m0 += 1;
    });
    rejects_resealed!(
        "parameter ef_construction",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.params.ef_construction += 1;
        }
    );
    rejects_resealed!(
        "parameter ef_search",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.params.ef_search += 1;
        }
    );
    rejects_resealed!("seed", |r: &mut NativeHnswGenerationReceiptV2| {
        r.seed ^= 1;
    });
    rejects_resealed!("point count", |r: &mut NativeHnswGenerationReceiptV2| {
        r.point_count += 1;
    });
    rejects_resealed!("entry point", |r: &mut NativeHnswGenerationReceiptV2| {
        let count = u32::try_from(r.point_count).expect("test count");
        r.entry_point = Some((r.entry_point.expect("non-empty graph") + 1) % count);
    });
    rejects_resealed!("maximum level", |r: &mut NativeHnswGenerationReceiptV2| {
        r.max_level = u64::from(r.max_level == 0);
    });
    rejects_resealed!("payload CRC", |r: &mut NativeHnswGenerationReceiptV2| {
        r.payload_crc32 ^= 1;
    });
    rejects_resealed!("header CRC", |r: &mut NativeHnswGenerationReceiptV2| {
        r.header_crc32 ^= 1;
    });
    rejects_resealed!(
        "topology SHA-256",
        |r: &mut NativeHnswGenerationReceiptV2| {
            r.topology_sha256 = "7".repeat(64);
        }
    );

    let mut receipt_seal_tamper = original.to_bytes().expect("canonical receipt");
    *receipt_seal_tamper.last_mut().expect("receipt seal byte") ^= 1;
    std::fs::write(&receipt_path, receipt_seal_tamper).expect("receipt seal tamper");
    assert!(
        ValidatedNativeHnsw::load(Arc::clone(&owner), &graph_path).is_err(),
        "receipt SHA-256 is itself a bound field"
    );

    std::fs::write(
        &receipt_path,
        original.to_bytes().expect("restore original receipt"),
    )
    .expect("restore receipt");
    let wrong_generation = generation_owner(78, 0x78, "every-receipt-field", 8, 48);
    assert!(
        ValidatedNativeHnsw::load(Arc::clone(&wrong_generation), &graph_path).is_err(),
        "caller-held generation drift must fail"
    );
    let wrong_identity = generation_owner(77, 0x77, "different-identity", 8, 48);
    assert!(
        ValidatedNativeHnsw::load(Arc::clone(&wrong_identity), &graph_path).is_err(),
        "caller-held frozen identity drift must fail"
    );
}

#[test]
fn receipt_paths_fail_closed_on_malformed_or_nonlocal_names() {
    for malformed in [
        Path::new(""),
        Path::new("./graph.fshnsw"),
        Path::new("nested/../graph.fshnsw"),
        Path::new("graph.bin"),
        Path::new("graph\nforged.fshnsw"),
    ] {
        assert!(
            native_hnsw_generation_receipt_path(malformed).is_err(),
            "malformed graph path was admitted: {malformed:?}"
        );
    }

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;

        let invalid = PathBuf::from(std::ffi::OsString::from_vec(
            b"invalid-\xff.fshnsw".to_vec(),
        ));
        assert!(
            native_hnsw_generation_receipt_path(&invalid).is_err(),
            "non-UTF-8 basename must not enter the receipt"
        );
    }
}

#[cfg(unix)]
#[test]
fn receipt_save_and_load_reject_symlinks_special_files_and_symlinked_parents() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("temporary directory");
    let owner = generation_owner(19, 0x19, "path-hardening", 4, 16);
    let bound =
        ValidatedNativeHnsw::build(Arc::clone(&owner), params(), 19).expect("owner-bound graph");

    let preflight_graph = directory.path().join("preflight.fshnsw");
    let preflight_receipt =
        native_hnsw_generation_receipt_path(&preflight_graph).expect("receipt path");
    let receipt_target = directory.path().join("receipt-target");
    std::fs::write(&receipt_target, b"sentinel").expect("receipt target");
    symlink(&receipt_target, &preflight_receipt).expect("receipt symlink");
    assert!(
        matches!(
            bound
                .save(&preflight_graph)
                .expect_err("receipt symlink must fail before graph save"),
            SearchError::Io(_)
        ),
        "receipt symlink returned wrong error"
    );
    assert!(
        !preflight_graph.exists(),
        "receipt preflight failure must not publish the graph"
    );

    let source_graph = directory.path().join("source.fshnsw");
    bound.save(&source_graph).expect("source binding");
    let source_receipt =
        native_hnsw_generation_receipt_path(&source_graph).expect("source receipt");
    let alias_graph = directory.path().join("alias.fshnsw");
    bound.graph.save(&alias_graph).expect("alias graph bytes");
    let alias_receipt = native_hnsw_generation_receipt_path(&alias_graph).expect("alias receipt");
    symlink(&source_receipt, &alias_receipt).expect("load receipt symlink");
    assert!(matches!(
        ValidatedNativeHnsw::load(Arc::clone(&owner), &alias_graph)
            .expect_err("load must reject receipt symlink"),
        SearchError::Io(_)
    ));

    let special_graph = directory.path().join("special.fshnsw");
    let special_receipt =
        native_hnsw_generation_receipt_path(&special_graph).expect("special receipt");
    std::fs::create_dir(&special_receipt).expect("special receipt directory");
    assert!(matches!(
        bound
            .save(&special_graph)
            .expect_err("special receipt must fail before graph save"),
        SearchError::Io(_)
    ));
    assert!(!special_graph.exists());

    let real_parent = directory.path().join("real-parent");
    std::fs::create_dir(&real_parent).expect("real parent");
    let alias_parent = directory.path().join("alias-parent");
    symlink(&real_parent, &alias_parent).expect("symlink parent");
    assert!(matches!(
        bound
            .save(&alias_parent.join("graph.fshnsw"))
            .expect_err("symlinked parent must fail"),
        SearchError::Io(_)
    ));
}
