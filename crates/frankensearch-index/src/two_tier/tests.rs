#[cfg(feature = "ann")]
use std::collections::HashSet;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::generation::{ArtifactGenerationIdentityV1, QuantizationFormat};
use frankensearch_core::{
    BoundQueryEmbedding, RetrievalTopology, SearchError, SearchResult, TierQueryCoverageV1,
    TieredQueryEmbeddings, VectorHit,
};

#[cfg(feature = "ann")]
use crate::hnsw::{HNSW_META_FORMAT_CURRENT, HnswLoadDisposition};
#[cfg(unix)]
use crate::native_hnsw::native_hnsw_generation_receipt_path;
use crate::{FsviUpgradeRequired, Quantization, SearchParams};

#[cfg(feature = "ann")]
use super::ann::maybe_load_or_build_ann_with_save;
#[cfg(feature = "ann")]
use super::paths::{
    validate_ann_persistence_paths, validate_ann_save_lock_identities, validate_index_paths,
};
use super::*;

#[cfg(feature = "ann")]
mod ann_sidecars;
mod core_and_builder;
mod explicit_paths_and_aliases;
mod native_hnsw_and_activation;
mod observe_and_publish;
mod v2_admission;

fn temp_index_dir(label: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "frankensearch-two-tier-{label}-{}-{timestamp}",
        std::process::id()
    ))
}

fn write_index_file(path: &Path, rows: &[(&str, &[f32])]) -> SearchResult<()> {
    let dimension = rows
        .first()
        .map(|(_, vector)| vector.len())
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "rows".to_owned(),
            value: "[]".to_owned(),
            reason: "rows must not be empty".to_owned(),
        })?;
    let mut writer = VectorIndex::create(path, "test", dimension)?;
    for (doc_id, vector) in rows {
        writer.write_record(doc_id, vector)?;
    }
    writer.finish()
}

#[cfg(feature = "ann")]
fn load_native_ann_sidecar(path: &Path, source_index: &VectorIndex) -> HnswIndex {
    let (ann, disposition) =
        HnswIndex::load_with_disposition(path, source_index).expect("load persisted ANN sidecar");
    assert_eq!(
        disposition,
        HnswLoadDisposition::Native,
        "the repaired sidecar must use the native graph path on its next load"
    );

    let metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(path).expect("read ANN metadata"))
            .expect("parse ANN metadata");
    assert_eq!(
        metadata["format_version"].as_u64(),
        Some(u64::from(HNSW_META_FORMAT_CURRENT)),
        "repaired ANN metadata must use the current format"
    );
    let generation = metadata["sidecar_generation"]
        .as_str()
        .expect("current metadata generation");
    let basename = metadata["sidecar_basename"]
        .as_str()
        .expect("current metadata basename");
    let sidecar_parent = path.parent().expect("ANN metadata parent").join(generation);
    assert!(
        sidecar_parent
            .join(format!("{basename}.hnsw.graph"))
            .is_file(),
        "repaired native graph sibling must exist"
    );
    assert!(
        sidecar_parent
            .join(format!("{basename}.hnsw.data"))
            .is_file(),
        "repaired native data sibling must exist"
    );
    ann
}

/// Identity binding for `model_id` with canonical FSVI v2 storage, plus
/// the in-memory-storage sibling bundle a producing embedder would hold.
fn fsvi_v2_binding(
    model_id: &str,
    dimension: u32,
    sequence: u64,
) -> (FsviV2IdentityBinding, EmbeddingIdentityBundleV1) {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension);
    "fsvi-v2".clone_into(&mut identity.storage.format);
    identity.storage.quantization = QuantizationFormat::F16;
    "little-endian".clone_into(&mut identity.storage.endianness);
    let generation =
        ArtifactGenerationIdentityV1::new(sequence, [0x4d; 16]).expect("valid test generation");
    let binding = FsviV2IdentityBinding::new(
        generation,
        identity.freeze().expect("freeze artifact identity"),
    )
    .expect("valid FSVI v2 identity binding");
    (binding, identity)
}

fn write_v2_tier(path: &Path, binding: &FsviV2IdentityBinding, rows: &[(&str, &[f32])]) {
    let mut writer =
        VectorIndex::create_v2(path, binding.clone()).expect("create_v2 writer for fixture");
    for (doc_id, vector) in rows {
        writer.write_record(doc_id, vector).expect("write v2 row");
    }
    writer.finish().expect("finish v2 fixture");
}

#[rustfmt::skip]
#[cfg(feature = "ann")]
#[test]
fn ann_wal_shadowed_top_hit_stays_suppressed_through_post_build_tombstone() {
    use crate::wal::WalEntry;

    let dir = temp_index_dir("ann-wal-shadowed-top");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(
        &fast_path,
        &[
            ("doc-a", &[0.9, 0.435_889_9]),
            ("doc-b", &[0.8, -0.6]),
            ("doc-tombstone", &[0.0, 1.0]),
        ],
    )
    .expect("write fast index");
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 32,
        ..TwoTierConfig::default()
    };
    let mut index = TwoTierIndex::open(&dir, config.clone()).expect("open ANN index");
    let wal_entry = WalEntry {
        doc_id: "doc-a".into(),
        doc_id_hash: crate::fnv1a_hash(b"doc-a"),
        embedding: vec![-1.0, 0.0],
    };
    let fast_tier = index
        .fast_tier_mut_for_test()
        .expect("path-opened fast tier");
    fast_tier.wal_entries.push(wal_entry.clone());

    let normal_query = [1.0_f32, 0.0];
    assert!(
        index
            .fast_tier()
            .search_top_k(&normal_query, 1, None)
            .expect("canonical normal search")
            .is_empty(),
        "the physical main winner is shadowed after top-k selection"
    );
    assert!(
        index
            .search_fast(&normal_query, 1)
            .expect("normal ANN search")
            .is_empty(),
        "normal ANN must not backfill after suppressing a WAL-shadowed winner"
    );
    assert_eq!(index.ann_fallback_count(), 0);

    drop(index);
    let mut writer =
        VectorIndex::open_writer(&fast_path).expect("writer opens after reader drops");
    assert!(
        writer
            .soft_delete("doc-tombstone")
            .expect("post-build tombstone")
    );
    drop(writer);
    let mut index = TwoTierIndex::open(&dir, config).expect("reopen ANN index after tombstone");
    index
        .fast_tier_mut_for_test()
        .expect("reopened path-opened fast tier")
        .wal_entries
        .push(wal_entry);
    let tombstone_query = [0.0_f32, 1.0];
    assert!(
        index
            .fast_tier()
            .search_top_k(&tombstone_query, 1, None)
            .expect("canonical tombstone search")
            .is_empty(),
        "the exact main winner is still shadowed by the WAL"
    );
    assert!(
        index
            .search_fast(&tombstone_query, 1)
            .expect("refreshed ANN search")
            .is_empty(),
        "the refreshed ANN path must preserve WAL suppression after tombstoning"
    );
    assert_eq!(
        index.ann_fallback_count(),
        0,
        "a sidecar refreshed for the tombstone must not report stale-graph fallback"
    );
}

#[cfg(feature = "ann")]
#[test]
fn missing_ann_role_aliases_follow_the_mounted_filesystem_not_the_os_name() {
    let dir = temp_index_dir("ann-volume-collation");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join("fast.fsvi");
    let quality_path = dir.join("quality.fsvi");
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");
    write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0])]).expect("write quality index");

    for (label, probe_name, alternate_probe_name, left_name, right_name) in [
        (
            "ascii",
            "ascii-Probe",
            "ASCII-pROBE",
            "shared-ascii.hnsw",
            "SHARED-ASCII.HNSW",
        ),
        (
            "unicode-full-case-fold",
            "unicode-Straße",
            "unicode-STRASSE",
            "shared-Straße.hnsw",
            "shared-STRASSE.hnsw",
        ),
        (
            "unicode-one-to-one-case",
            "unicode-É-probe",
            "unicode-é-probe",
            "shared-É.hnsw",
            "shared-é.hnsw",
        ),
        (
            "unicode-normalization",
            "unicode-é-normalization-probe",
            "unicode-e\u{301}-normalization-probe",
            "shared-é-normalization.hnsw",
            "shared-e\u{301}-normalization.hnsw",
        ),
    ] {
        let probe = dir.join(probe_name);
        fs::write(&probe, b"filesystem collation probe").expect("write collation probe");
        let alternate_probe = dir.join(alternate_probe_name);
        let volume_aliases = crate::file_identity::is_same_file(&probe, &alternate_probe)
            .unwrap_or_else(|error| {
                assert_eq!(
                    error.kind(),
                    std::io::ErrorKind::NotFound,
                    "unexpected collation probe error"
                );
                false
            });
        eprintln!(
            "bd-07os filesystem-alias probe: label={label} primary={} alternate={} \
             aliases={volume_aliases}",
            probe.display(),
            alternate_probe.display()
        );

        let paths = TwoTierIndexPaths::new(&fast_path)
            .with_quality_index(&quality_path)
            .with_fast_ann(dir.join(left_name))
            .with_quality_ann(dir.join(right_name));
        validate_index_paths(&paths).expect("missing leaves are initially distinct");
        let validation = validate_ann_save_lock_identities(&paths, true, true);
        if volume_aliases {
            assert!(
                matches!(
                    validation,
                    Err(SearchError::InvalidConfig { ref field, .. })
                        if field == "index_paths"
                ),
                "mounted filesystem must reject {label} ANN aliases: {validation:?}"
            );
        } else {
            validation.unwrap_or_else(|error| {
                panic!(
                    "mounted filesystem must preserve distinct {label} ANN roles: \
                     {error:?}"
                )
            });
        }
    }
}
