use std::collections::BTreeMap;

use crate::SearchError;

use super::tests_support::{refresh_manifest_hash, sample_vector_artifact, valid_manifest};
use super::*;

#[test]
fn valid_manifest_passes() {
    let m = valid_manifest();
    let r = validate_manifest(&m);
    assert!(r.is_valid(), "findings: {:#?}", r.findings);
    assert!(r.errors().is_empty());
}

#[test]
fn legacy_schema_version_is_error() {
    let mut m = valid_manifest();
    m.schema_version = MANIFEST_SCHEMA_VERSION - 1;
    refresh_manifest_hash(&mut m);
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "schema_version"));
}

#[test]
fn future_schema_version_is_error() {
    let mut m = valid_manifest();
    m.schema_version = MANIFEST_SCHEMA_VERSION + 1;
    refresh_manifest_hash(&mut m);
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "schema_version"));
}

#[test]
fn empty_generation_id_is_error() {
    let mut m = valid_manifest();
    m.generation_id = String::new();
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "generation_id"));
}

#[test]
fn empty_manifest_hash_is_error() {
    let mut m = valid_manifest();
    m.manifest_hash.clear();
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "manifest_hash"));
}

#[test]
fn malformed_manifest_hash_is_error() {
    let mut m = valid_manifest();
    m.manifest_hash = "not-a-sha256".into();
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "manifest_hash"));
}

#[test]
fn mismatched_manifest_hash_is_error() {
    let mut m = valid_manifest();
    m.manifest_hash = "0".repeat(64);
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(
        r.errors()
            .iter()
            .any(|f| f.check == "manifest_hash" && f.message.contains("does not match canonical"))
    );
}

#[test]
fn manifest_hash_match_is_case_insensitive() {
    let mut m = valid_manifest();
    m.manifest_hash = m.manifest_hash.to_uppercase();
    let r = validate_manifest(&m);
    assert!(r.is_valid());
}

#[test]
fn invalid_commit_range_is_error() {
    let mut m = valid_manifest();
    m.commit_range = CommitRange { low: 50, high: 10 };
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "commit_range"));
}

#[test]
fn zero_timestamps_are_errors() {
    let mut m = valid_manifest();
    m.build_started_at = 0;
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "build_started_at"));
}

#[test]
fn completed_before_started_is_error() {
    let mut m = valid_manifest();
    m.build_completed_at = m.build_started_at - 1;
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "build_timestamps"));
}

#[test]
fn no_embedders_is_error() {
    let mut m = valid_manifest();
    m.embedders.clear();
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "embedders"));
}

#[test]
fn embedder_empty_fields_are_errors() {
    let mut m = valid_manifest();
    let mut invalid = EmbedderRevision::explicit_test_model("bad", 256);
    invalid.identity.space.logical_model_id.clear();
    invalid.identity.space.dimension = 0;
    m.embedders.insert("bad".into(), invalid);
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    let errors = r.errors();
    assert!(errors.iter().any(|f| f.check == "embedder_identity"));
}

#[test]
fn duplicate_vector_artifact_paths_is_error() {
    let mut m = valid_manifest();
    m.vector_artifacts
        .push(sample_vector_artifact("vectors/shard_0.fsvi", 100));
    m.total_documents = 200;
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(
        r.errors()
            .iter()
            .any(|f| f.check == "vector_artifact_duplicate")
    );
}

#[test]
fn empty_artifact_path_is_error() {
    let mut m = valid_manifest();
    m.vector_artifacts.push(sample_vector_artifact("", 10));
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "vector_artifact_path"));
}

#[test]
fn empty_artifact_checksum_is_error() {
    let mut m = valid_manifest();
    m.vector_artifacts[0].checksum = String::new();
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(
        r.errors()
            .iter()
            .any(|f| f.check == "vector_artifact_checksum")
    );
}

#[test]
fn repair_descriptor_unknown_artifact_is_error() {
    let mut m = valid_manifest();
    m.repair_descriptors.push(RepairDescriptor {
        protected_artifact: "nonexistent.fsvi".into(),
        sidecar_path: "nonexistent.fsvi.fec".into(),
        source_symbols: 10,
        repair_symbols: 2,
        overhead_ratio: 0.2,
    });
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(
        r.errors()
            .iter()
            .any(|f| f.check == "repair_descriptor_target")
    );
}

#[test]
fn repair_descriptor_zero_source_symbols_is_error() {
    let mut m = valid_manifest();
    m.repair_descriptors[0].source_symbols = 0;
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(
        r.errors()
            .iter()
            .any(|f| f.check == "repair_descriptor_symbols")
    );
}

#[test]
fn extreme_repair_overhead_is_warning() {
    let mut m = valid_manifest();
    m.repair_descriptors[0].overhead_ratio = 15.0;
    refresh_manifest_hash(&mut m);
    let r = validate_manifest(&m);
    assert!(r.is_valid());
    assert!(
        r.warnings()
            .iter()
            .any(|f| f.check == "repair_descriptor_overhead")
    );
}

#[test]
fn duplicate_invariant_id_is_error() {
    let mut m = valid_manifest();
    m.activation_invariants
        .push(m.activation_invariants[0].clone());
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "invariant_duplicate"));
}

#[test]
fn zero_total_documents_with_artifacts_is_error() {
    let mut m = valid_manifest();
    m.total_documents = 0;
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    assert!(r.errors().iter().any(|f| f.check == "total_documents"));
}

#[test]
fn lexical_count_mismatch_is_warning() {
    let mut m = valid_manifest();
    m.lexical_artifacts[0].document_count = 50; // != total_documents (100)
    refresh_manifest_hash(&mut m);
    let r = validate_manifest(&m);
    assert!(r.is_valid());
    assert!(
        r.warnings()
            .iter()
            .any(|f| f.check == "lexical_document_count")
    );
}

#[test]
fn two_tier_vector_count_accepted() {
    let mut m = valid_manifest();
    // Fast tier: 100 vectors + Quality tier: 100 vectors = 200 total, 100 docs
    m.vector_artifacts = vec![
        sample_vector_artifact("vectors/fast.fsvi", 100),
        sample_vector_artifact("vectors/quality.fsvi", 100),
    ];
    m.vector_artifacts[1].embedder_tier = EmbedderTierTag::Quality;
    // Update repair descriptor to reference the new fast tier artifact.
    m.repair_descriptors[0].protected_artifact = "vectors/fast.fsvi".into();
    refresh_manifest_hash(&mut m);
    let r = validate_manifest(&m);
    assert!(r.is_valid(), "findings: {:#?}", r.findings);
}

#[test]
fn serde_roundtrip() {
    let m = valid_manifest();
    let json = serde_json::to_string_pretty(&m).expect("serialize");
    let deserialized: GenerationManifest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(m, deserialized);
}

#[test]
fn commit_range_len_and_empty() {
    let range = CommitRange { low: 5, high: 10 };
    assert_eq!(range.len(), 6);
    assert!(!range.is_empty());

    let empty = CommitRange { low: 10, high: 5 };
    assert!(empty.is_empty());
}

#[test]
fn single_commit_range() {
    let range = CommitRange { low: 42, high: 42 };
    assert_eq!(range.len(), 1);
    assert!(!range.is_empty());
}

#[test]
fn require_valid_passes_for_valid_manifest() {
    let m = valid_manifest();
    let r = validate_manifest(&m);
    assert!(require_valid(&r).is_ok());
}

#[test]
fn require_valid_fails_for_invalid_manifest() {
    let mut m = valid_manifest();
    m.generation_id = String::new();
    let r = validate_manifest(&m);
    let err = require_valid(&r).unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn empty_manifest_collects_multiple_errors() {
    let m = GenerationManifest {
        schema_version: 0,
        generation_id: String::new(),
        manifest_hash: String::new(),
        commit_range: CommitRange { low: 10, high: 5 },
        build_started_at: 0,
        build_completed_at: 0,
        embedders: BTreeMap::new(),
        vector_artifacts: vec![],
        lexical_artifacts: vec![],
        repair_descriptors: vec![],
        activation_invariants: vec![],
        total_documents: 0,
        metadata: BTreeMap::new(),
    };
    let r = validate_manifest(&m);
    assert!(!r.is_valid());
    // Should find at least: schema_version, generation_id, commit_range,
    // build_started_at, build_completed_at, embedders
    assert!(r.errors().len() >= 5, "found {} errors", r.errors().len());
}

#[test]
fn metadata_is_preserved() {
    let mut m = valid_manifest();
    m.metadata.insert("build_host".into(), "node-7".into());
    m.metadata.insert("deployment".into(), "production".into());
    let json = serde_json::to_string(&m).expect("serialize");
    let deserialized: GenerationManifest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.metadata.get("build_host").unwrap(), "node-7");
}

#[test]
fn invariant_kinds_serialize() {
    let kinds = vec![
        InvariantKind::AllArtifactsVerified,
        InvariantKind::EmbedderRevisionMatch,
        InvariantKind::VectorCountConsistency {
            expected_total: 500,
        },
        InvariantKind::CommitContinuity { previous_high: 99 },
        InvariantKind::Custom {
            check_name: "custom_check".into(),
        },
    ];
    for kind in &kinds {
        let json = serde_json::to_string(kind).expect("serialize");
        let back: InvariantKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(kind, &back);
    }
}

#[test]
fn quantization_format_serialize() {
    for fmt in &[
        QuantizationFormat::F32,
        QuantizationFormat::F16,
        QuantizationFormat::Int8,
        QuantizationFormat::Int4,
    ] {
        let json = serde_json::to_string(fmt).expect("serialize");
        let back: QuantizationFormat = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(fmt, &back);
    }
}
