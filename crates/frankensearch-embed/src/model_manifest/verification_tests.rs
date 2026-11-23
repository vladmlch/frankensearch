//! Unit tests for model manifest enrichment, native receipts, and verification caching.

use std::fs;
use std::path::PathBuf;

use frankensearch_core::error::SearchError;
use frankensearch_core::generation::{GoldenVectorCertificateV1, QuantizationFormat};

use super::catalog::{ModelFile, ModelManifest, ModelManifestCatalog};
use super::frozen::{
    ModelArtifactManifestV1, ModelExecutionContractV1, conformance_corpus_fingerprint,
    default_plain_text_input_contract,
};
use super::frozen_tests::{before_dependency_refresh, before_minilm_256, write_temp_file};
use super::verification::{
    FileVerificationState, TEST_AFTER_FULL_HASH_HOOK, TEST_VERIFY_FILE_HASH_CALLS,
    VERIFICATION_MARKER_SCHEMA_VERSION, VERIFIED_MARKER_FILE, VerificationMarker,
    capture_file_verification_state, is_verification_cached, to_hex_lowercase,
    verify_dir_and_record, verify_dir_cached,
};
use super::{
    MANIFEST_SCHEMA_VERSION, MODEL2VEC_OUTPUT_NORMALIZATION_V1, MODEL2VEC_POOLING_V1,
    MODEL2VEC_PREPROCESSING_V1, MODEL2VEC_SEQUENCE_POLICY_V1, ModelTier,
    PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
};

// ── bd-2w7x.5: Model manifest enrichment tests ─────────────────────

#[test]
fn manifest_schema_version_is_two() {
    assert_eq!(MANIFEST_SCHEMA_VERSION, 2);
}

#[test]
fn model_tier_serde_roundtrip() {
    for tier in &[ModelTier::Fast, ModelTier::Quality, ModelTier::Reranker] {
        let json = serde_json::to_string(tier).unwrap();
        let decoded: ModelTier = serde_json::from_str(&json).unwrap();
        assert_eq!(&decoded, tier);
    }
}

#[test]
fn model_tier_serde_uses_snake_case() {
    assert_eq!(serde_json::to_string(&ModelTier::Fast).unwrap(), "\"fast\"");
    assert_eq!(
        serde_json::to_string(&ModelTier::Quality).unwrap(),
        "\"quality\""
    );
    assert_eq!(
        serde_json::to_string(&ModelTier::Reranker).unwrap(),
        "\"reranker\""
    );
}

#[test]
fn builtin_potion_has_correct_metadata() {
    let m = ModelManifest::potion_128m();
    assert_eq!(m.dimension, Some(256));
    assert_eq!(m.tier, Some(ModelTier::Fast));
    assert_eq!(m.license, "MIT");
    assert!(m.display_name.is_some());
    assert!(m.display_name.as_deref().unwrap().contains("fast"));
}

#[test]
fn builtin_minilm_has_correct_metadata() {
    let m = ModelManifest::minilm_v2();
    assert_eq!(m.dimension, Some(384));
    assert_eq!(m.tier, Some(ModelTier::Quality));
    assert!(m.display_name.is_some());
    assert!(m.display_name.as_deref().unwrap().contains("quality"));
}

#[test]
fn native_minilm_download_preserves_frozen_producer_artifacts() {
    let download = ModelManifest::minilm_v2_native();
    download.validate().unwrap();
    assert!(download.is_production_ready());
    assert!(
        !ModelManifest::builtin_catalog()
            .models
            .iter()
            .any(|m| m.id == download.id)
    );
    let frozen = ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap();
    assert_eq!(download.files.len(), frozen.artifacts.len());
    for artifact in &frozen.artifacts {
        let file = download
            .files
            .iter()
            .find(|file| file.name == artifact.relative_path)
            .unwrap();
        assert_eq!(file.sha256, artifact.sha256);
        assert_eq!(file.size, artifact.size);
    }
}

#[test]
fn native_installation_receipt_requires_the_exact_registered_pair() {
    let download = ModelManifest::minilm_v2_native();
    for native in [
        ModelArtifactManifestV1::minilm_native_frankentorch().unwrap(),
        ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap(),
    ] {
        assert!(native.matches_registered_native_installation(&download));
        assert!(
            !native.artifacts_match_download_manifest(&download),
            "the generic logical-id boundary must remain strict"
        );
        let mut foreign_download = download.clone();
        foreign_download.id.push_str("-foreign");
        assert!(!native.matches_registered_native_installation(&foreign_download));
        foreign_download = download.clone();
        foreign_download.revision = "a".repeat(40);
        assert!(!native.matches_registered_native_installation(&foreign_download));
        foreign_download = download.clone();
        foreign_download.files[0].sha256 = "0".repeat(64);
        assert!(!native.matches_registered_native_installation(&foreign_download));
        foreign_download = download.clone();
        foreign_download.files[0].url = Some("https://example.invalid/model".to_owned());
        assert!(!native.matches_registered_native_installation(&foreign_download));

        let mut foreign_producer = native.clone();
        foreign_producer.logical_model_id.push_str("-foreign");
        assert!(!foreign_producer.matches_registered_native_installation(&download));
        foreign_producer = native.clone();
        foreign_producer.execution.golden_vectors.vectors_sha256 = "0".repeat(64);
        assert!(!foreign_producer.matches_registered_native_installation(&download));
        foreign_producer = native.clone();
        foreign_producer.artifacts[0].sha256 = "0".repeat(64);
        assert!(!foreign_producer.matches_registered_native_installation(&download));
        assert!(!native.matches_registered_native_installation(&ModelManifest::minilm_v2()));
    }
    assert!(
        !ModelArtifactManifestV1::multilingual_minilm_native_frankentorch()
            .unwrap()
            .matches_registered_native_installation(&download)
    );
}

#[test]
fn native_f32_tokenizer_refresh_preserves_historical_identity_and_requires_admission() {
    use frankensearch_core::generation::ProducerCompatibilityErrorV1;

    let current = ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap();
    // GOLDEN-CHANGE bd-7651e: the tokenizer refresh is checked from the
    // 512-token producer that preceded the move to 256 tokens; both of its
    // fingerprints below are the retained pre-change values.
    let before_256 = before_minilm_256(current.clone());
    let historical = before_dependency_refresh(current.clone());
    let mut restored = historical.clone();
    before_256
        .execution
        .protocol_revision
        .clone_into(&mut restored.execution.protocol_revision);
    assert_eq!(
        restored, before_256,
        "only tokenizer protocol provenance changed"
    );
    let current_identity = current
        .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();
    let before_256_identity = before_256
        .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();
    let historical_identity = historical
        .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();
    assert_eq!(
        historical_identity.fingerprint(),
        "35d0a014b4ec6224eb42552ea1099b24bead0c9c51906b6035207b6105e8af01"
    );
    assert_eq!(
        before_256_identity.fingerprint(),
        "aa25d24b07a2d233445cb6605c95d33601ea36a044d3a4b2bed65e7590386109"
    );
    // GOLDEN-CHANGE bd-7651e: the 256-token space; only sequence_policy moved
    // (minilm_256_changes_only_the_space_sequence_policy proves that).
    assert_eq!(
        current_identity.fingerprint(),
        "8c37f4cf6f63cdbc937ff7b03369606555e0eb165779c0019b1591fcadaf0860"
    );
    assert_eq!(before_256_identity.space, historical_identity.space);
    assert_ne!(current_identity.space, before_256_identity.space);
    assert_eq!(
        current.execution.golden_vectors,
        historical.execution.golden_vectors
    );
    assert_eq!(
        before_256_identity.verify_exact_producer_with(&historical_identity),
        Err(ProducerCompatibilityErrorV1::CertificateRequired)
    );
}

#[test]
#[ignore = "requires real native MiniLM files via MINILM_FIXTURE_DIR"]
fn native_installation_receipt_verifies_real_files_and_preserves_identity() {
    use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

    let source = PathBuf::from(std::env::var_os("MINILM_FIXTURE_DIR").unwrap());
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("installed");
    fs::create_dir(&dir).unwrap();
    let download = ModelManifest::minilm_v2_native();
    let native = ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap();
    for file in &download.files {
        let destination = dir.join(&file.name);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(source.join(&file.name), destination).unwrap();
    }
    let verify = || {
        TEST_VERIFY_FILE_HASH_CALLS.with(|count| count.set(0));
        let verified = native.verify_dir_cached(&download, &dir).unwrap();
        assert_eq!(
            verified
                .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
                .unwrap()
                .fingerprint(),
            // GOLDEN-CHANGE bd-7651e: the current 256-token F32 identity.
            "8c37f4cf6f63cdbc937ff7b03369606555e0eb165779c0019b1591fcadaf0860"
        );
        TEST_VERIFY_FILE_HASH_CALLS.with(std::cell::Cell::get)
    };

    assert_eq!(
        verify(),
        download.files.len(),
        "absent receipt hashes all files"
    );
    assert!(!dir.join(VERIFIED_MARKER_FILE).exists());
    verify_dir_and_record(&download, &dir).unwrap();
    assert_eq!(verify(), 0, "a current registered receipt avoids rehashing");
    fs::write(dir.join(VERIFIED_MARKER_FILE), b"malformed receipt").unwrap();
    assert_eq!(verify(), download.files.len(), "malformed receipt rehashes");
    verify_dir_and_record(&download, &dir).unwrap();

    let mut foreign = native.clone();
    foreign.artifacts[0].sha256 = "0".repeat(64);
    assert!(foreign.verify_dir_cached(&download, &dir).is_err());

    let weights_path = dir.join("model.safetensors");
    let mut weights = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&weights_path)
        .unwrap();
    #[cfg(unix)]
    let original_modified = weights.metadata().unwrap().modified().unwrap();
    weights.seek(SeekFrom::End(-1)).unwrap();
    let mut byte = [0];
    weights.read_exact(&mut byte).unwrap();
    weights.seek(SeekFrom::End(-1)).unwrap();
    weights.write_all(&[byte[0] ^ 1]).unwrap();
    weights.sync_all().unwrap();
    // Restoring mtime cannot hide a same-size Unix mutation from the bound ctime.
    #[cfg(unix)]
    weights.set_modified(original_modified).unwrap();
    drop(weights);
    assert!(!is_verification_cached(&download, &dir));
    assert!(native.verify_dir_cached(&download, &dir).is_err());

    // Preserve the changed inode outside the model directory, then replace
    // it with correct bytes. The old receipt must still miss on replacement.
    fs::rename(&weights_path, tmp.path().join("changed-weights")).unwrap();
    fs::copy(source.join("model.safetensors"), &weights_path).unwrap();
    assert!(!is_verification_cached(&download, &dir));
    assert_eq!(
        verify(),
        download.files.len(),
        "replacement requires rehashing"
    );
    verify_dir_and_record(&download, &dir).unwrap();
    assert_eq!(verify(), 0);
}

#[test]
fn multilingual_minilm_is_production_ready_but_not_in_default_catalog() {
    let manifest = ModelManifest::multilingual_minilm_l12_v2();
    manifest.validate().unwrap();
    assert!(manifest.is_production_ready());
    assert_eq!(manifest.dimension, Some(384));
    assert_eq!(manifest.tier, Some(ModelTier::Quality));
    assert_eq!(manifest.total_size_bytes(), 479_724_528);
    assert!(
        !ModelManifest::builtin_catalog()
            .models
            .iter()
            .any(|candidate| candidate.id == manifest.id)
    );
    let opt_in = ModelManifest::opt_in_catalog();
    assert_eq!(
        opt_in.models,
        vec![
            manifest,
            ModelManifest::minilm_v2_native(),
            ModelManifest::potion_base_8m(),
            ModelManifest::potion_base_32m(),
        ]
    );
    opt_in.validate().unwrap();
}

#[test]
fn builtin_reranker_has_correct_metadata() {
    let m = ModelManifest::ms_marco_reranker();
    assert_eq!(m.id, "ms-marco-minilm-l-6-v2");
    assert_eq!(m.dimension, None); // Cross-encoder, no embedding dim
    assert_eq!(m.tier, Some(ModelTier::Reranker));
    assert!(m.display_name.is_some());
    assert!(m.display_name.as_deref().unwrap().contains("reranker"));
    m.validate().unwrap();
}

#[test]
fn builtin_catalog_contains_all_models() {
    let catalog = ModelManifest::builtin_catalog();
    assert_eq!(catalog.schema_version, MANIFEST_SCHEMA_VERSION);
    assert_eq!(catalog.models.len(), 7);

    let ids: Vec<&str> = catalog.models.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"potion-multilingual-128m"));
    assert!(ids.contains(&"all-minilm-l6-v2"));
    assert!(ids.contains(&"ms-marco-minilm-l-6-v2"));
    assert!(ids.contains(&"snowflake-arctic-embed-s"));
    assert!(ids.contains(&"nomic-embed-text-v1.5"));
    assert!(ids.contains(&"jina-reranker-v1-turbo-en"));
    assert!(ids.contains(&"flashrank-nano"));

    catalog.validate().unwrap();
}

#[test]
fn builtin_catalog_covers_all_tiers() {
    let catalog = ModelManifest::builtin_catalog();
    let tiers: Vec<Option<ModelTier>> = catalog.models.iter().map(|m| m.tier).collect();
    assert!(tiers.contains(&Some(ModelTier::Fast)));
    assert!(tiers.contains(&Some(ModelTier::Quality)));
    assert!(tiers.contains(&Some(ModelTier::Reranker)));
}

#[test]
fn builtin_manifests_include_version_description_and_size_metadata() {
    let manifests = [
        ModelManifest::potion_128m(),
        ModelManifest::minilm_v2(),
        ModelManifest::ms_marco_reranker(),
        ModelManifest::snowflake_arctic_s(),
        ModelManifest::nomic_embed(),
        ModelManifest::jina_reranker_turbo(),
        ModelManifest::flashrank_nano(),
    ];

    for manifest in manifests {
        assert!(!manifest.version.is_empty());
        assert!(manifest.description.is_some());
        // Some built-in manifests intentionally use placeholder metadata
        // (sha256=PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, size=0) so that file
        // sizes are confirmed at runtime during the first download rather
        // than baked into source code. For those, we only assert the
        // size-sum invariant, not a non-zero total.
        let all_placeholder = manifest
            .files
            .iter()
            .all(|file| file.sha256 == PLACEHOLDER_VERIFY_AFTER_DOWNLOAD);
        if !all_placeholder {
            assert!(manifest.download_size_bytes > 0);
        }
        let summed_size: u64 = manifest.files.iter().map(|file| file.size).sum();
        assert_eq!(manifest.download_size_bytes, summed_size);
    }
}

#[test]
fn model_file_download_url_uses_explicit_when_present() {
    let file = ModelFile {
        name: "model.onnx".to_owned(),
        sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
        size: 0,
        url: Some("https://mirror.example.com/model.onnx".to_owned()),
    };
    let url = file.download_url("owner/repo", "abc123");
    assert_eq!(url, "https://mirror.example.com/model.onnx");
}

#[test]
fn model_file_download_url_derives_from_repo_when_none() {
    let file = ModelFile {
        name: "onnx/model.onnx".to_owned(),
        sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
        size: 0,
        url: None,
    };
    let url = file.download_url("sentence-transformers/all-MiniLM-L6-v2", "abc123");
    assert_eq!(
        url,
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/abc123/onnx/model.onnx"
    );
}

#[test]
fn model_file_url_field_is_optional_in_json() {
    // URL absent: should deserialize with url=None
    let json = r#"{"name":"f.bin","sha256":"PLACEHOLDER_VERIFY_AFTER_DOWNLOAD","size":0}"#;
    let file: ModelFile = serde_json::from_str(json).unwrap();
    assert!(file.url.is_none());

    // URL present: should deserialize
    let json = r#"{"name":"f.bin","sha256":"PLACEHOLDER_VERIFY_AFTER_DOWNLOAD","size":0,"url":"https://example.com/f.bin"}"#;
    let file: ModelFile = serde_json::from_str(json).unwrap();
    assert_eq!(file.url.as_deref(), Some("https://example.com/f.bin"));
}

#[test]
fn model_file_url_skipped_in_serialization_when_none() {
    let file = ModelFile {
        name: "f.bin".to_owned(),
        sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
        size: 0,
        url: None,
    };
    let json = serde_json::to_string(&file).unwrap();
    assert!(!json.contains("url"));
}

#[test]
fn manifest_display_name_optional_in_json() {
    let json = r#"{
        "id": "test", "repo": "r", "revision": "v",
        "files": [], "license": "MIT"
    }"#;
    let m: ModelManifest = serde_json::from_str(json).unwrap();
    assert!(m.display_name.is_none());
    assert!(m.dimension.is_none());
    assert!(m.tier.is_none());
}

#[test]
fn manifest_with_all_new_fields_roundtrips() {
    let m = ModelManifest {
        id: "test".to_owned(),
        version: "test-v1".to_owned(),
        display_name: Some("Test Model".to_owned()),
        description: Some("test manifest".to_owned()),
        repo: "owner/repo".to_owned(),
        revision: "abc123".to_owned(),
        files: vec![ModelFile {
            name: "model.onnx".to_owned(),
            sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
            size: 0,
            url: Some("https://example.com/model.onnx".to_owned()),
        }],
        license: "MIT".to_owned(),
        dimension: Some(384),
        tier: Some(ModelTier::Quality),
        download_size_bytes: 0,
    };
    let json = m.to_pretty_json().unwrap();
    let restored = ModelManifest::from_json_str(&json).unwrap();
    assert_eq!(restored.display_name, m.display_name);
    assert_eq!(restored.dimension, m.dimension);
    assert_eq!(restored.tier, m.tier);
    assert_eq!(restored.files[0].url, m.files[0].url);
}

#[test]
fn catalog_schema_version_defaults_on_missing() {
    let json = r#"{"models":[]}"#;
    let catalog = ModelManifestCatalog::from_json_str(json).unwrap();
    assert_eq!(catalog.schema_version, MANIFEST_SCHEMA_VERSION);
}

#[test]
fn catalog_schema_version_preserved_from_json() {
    let json = r#"{"schema_version": 42, "models":[]}"#;
    let catalog = ModelManifestCatalog::from_json_str(json).unwrap();
    assert_eq!(catalog.schema_version, 42);
}

#[test]
fn builtin_catalog_json_roundtrip() {
    let catalog = ModelManifest::builtin_catalog();
    let json = serde_json::to_string_pretty(&catalog).unwrap();
    let restored = ModelManifestCatalog::from_json_str(&json).unwrap();
    assert_eq!(restored.schema_version, catalog.schema_version);
    assert_eq!(restored.models.len(), catalog.models.len());
    for (orig, rest) in catalog.models.iter().zip(restored.models.iter()) {
        assert_eq!(orig.id, rest.id);
        assert_eq!(orig.dimension, rest.dimension);
        assert_eq!(orig.tier, rest.tier);
    }
}

#[test]
fn registry_includes_reranker() {
    let all = ModelManifest::registered();
    let ids: Vec<&str> = all.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains(&"ms-marco-minilm-l-6-v2"),
        "registry should contain ms-marco reranker, got: {ids:?}"
    );
}

// ─── Verification Cache Tests ──────────────────────────────────────

fn make_test_manifest(file_name: &str, content: &[u8]) -> ModelManifest {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content);
    let sha = to_hex_lowercase(&hasher.finalize());
    ModelManifest {
        id: "test-model".to_owned(),
        repo: "test/repo".to_owned(),
        revision: "a".repeat(40),
        files: vec![ModelFile {
            name: file_name.to_owned(),
            sha256: sha,
            size: u64::try_from(content.len()).unwrap(),
            url: None,
        }],
        license: "MIT".to_owned(),
        tier: None,
        dimension: None,
        display_name: None,
        version: String::new(),
        description: None,
        download_size_bytes: u64::try_from(content.len()).unwrap(),
    }
}

#[test]
fn verification_marker_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"hello model";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    let json = std::fs::read_to_string(tmp.path().join(VERIFIED_MARKER_FILE)).unwrap();
    let restored: VerificationMarker = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.manifest_id, "test-model");
    assert_eq!(restored.schema_version, VERIFICATION_MARKER_SCHEMA_VERSION);
    assert_eq!(restored.manifest_fingerprint.len(), 64);
    let state = restored.file_states.get("model.bin").unwrap();
    assert_eq!(state.size_bytes, u64::try_from(content.len()).unwrap());
    assert!(state.modified_unix_nanos > 0);
}

#[test]
fn verification_cache_hit_when_files_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"model data";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    assert!(!is_verification_cached(&manifest, tmp.path()));
    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    assert!(is_verification_cached(&manifest, tmp.path()));
}

#[test]
fn verification_cache_miss_when_manifest_id_changes() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"model data";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    let mut changed = manifest;
    changed.id = "different-model".to_owned();
    assert!(!is_verification_cached(&changed, tmp.path()));
}

#[test]
fn verification_cache_miss_when_file_state_differs() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"model data";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    // Write marker, then tamper with the recorded file state.
    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    assert!(is_verification_cached(&manifest, tmp.path()));

    let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
    let raw = std::fs::read_to_string(&marker_path).unwrap();
    let mut marker: VerificationMarker = serde_json::from_str(&raw).unwrap();
    // Change recorded metadata so it no longer matches the actual file.
    marker.file_states.insert(
        "model.bin".to_owned(),
        FileVerificationState {
            size_bytes: 1,
            modified_unix_nanos: 1,
            created_unix_nanos: None,
            platform_file_id: None,
            platform_change_stamp: None,
        },
    );
    let tampered = serde_json::to_string_pretty(&marker).unwrap();
    std::fs::write(&marker_path, tampered).unwrap();

    assert!(!is_verification_cached(&manifest, tmp.path()));
}

#[test]
fn verify_dir_cached_is_observational_on_uncached_success() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"model data for verify";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    assert!(!tmp.path().join(VERIFIED_MARKER_FILE).exists());
    verify_dir_cached(&manifest, tmp.path()).unwrap();
    assert!(
        !tmp.path().join(VERIFIED_MARKER_FILE).exists(),
        "observational verification must not mint a receipt"
    );
}

#[test]
fn verify_dir_cached_skips_rehash_on_cached_hit() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"model data cached";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    // The authority-bearing operation performs the full hash and mints the receipt.
    verify_dir_and_record(&manifest, tmp.path()).unwrap();

    // The consumer operation may then admit the unchanged receipt.
    verify_dir_cached(&manifest, tmp.path()).unwrap();
}

#[test]
fn file_state_birth_time_is_compared_only_when_both_sides_observe_it() {
    let recorded = FileVerificationState {
        size_bytes: 10,
        modified_unix_nanos: 5,
        created_unix_nanos: Some(3),
        platform_file_id: Some("unix-dev=1;ino=2".to_owned()),
        platform_change_stamp: Some("unix-ctime=4;nsec=0".to_owned()),
    };
    let unobserved = FileVerificationState {
        created_unix_nanos: None,
        ..recorded.clone()
    };
    assert!(recorded.still_matches(&recorded));
    assert!(recorded.still_matches(&unobserved));
    assert!(unobserved.still_matches(&recorded));
    assert!(!recorded.still_matches(&FileVerificationState {
        created_unix_nanos: Some(4),
        ..recorded.clone()
    }));
    for changed in [
        FileVerificationState {
            size_bytes: 11,
            ..unobserved.clone()
        },
        FileVerificationState {
            modified_unix_nanos: 6,
            ..unobserved.clone()
        },
        FileVerificationState {
            platform_file_id: Some("unix-dev=1;ino=3".to_owned()),
            ..unobserved.clone()
        },
        FileVerificationState {
            platform_change_stamp: Some("unix-ctime=5;nsec=0".to_owned()),
            ..unobserved.clone()
        },
    ] {
        assert!(!recorded.still_matches(&changed), "{changed:?}");
        assert!(!unobserved.still_matches(&changed), "{changed:?}");
    }
}

#[test]
fn receipt_minted_without_birth_times_is_admitted_and_still_binds_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"model data minted by a musl build";
    let manifest = make_test_manifest("model.bin", content);
    let model = tmp.path().join("model.bin");
    write_temp_file(&model, content);
    verify_dir_and_record(&manifest, tmp.path()).unwrap();

    let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
    let minted: VerificationMarker =
        serde_json::from_str(&std::fs::read_to_string(&marker_path).unwrap()).unwrap();
    let rewrite = |created: Option<u64>| {
        let mut marker = minted.clone();
        for state in marker.file_states.values_mut() {
            state.created_unix_nanos = created;
        }
        std::fs::write(&marker_path, serde_json::to_string_pretty(&marker).unwrap()).unwrap();
    };

    // A static musl build cannot read birth times, so its receipt has none.
    rewrite(None);
    assert!(is_verification_cached(&manifest, tmp.path()));

    // When both sides observe a birth time, it must still agree.
    if let Some(observed) = minted.file_states["model.bin"].created_unix_nanos {
        rewrite(Some(observed ^ 1));
        assert!(!is_verification_cached(&manifest, tmp.path()));
    }

    // Without a birth time, a same-size rewrite is still caught.
    rewrite(None);
    write_temp_file(&model, b"model data minted by a musl buile");
    assert!(!is_verification_cached(&manifest, tmp.path()));
}

/// Build a two-file download manifest plus the native manifest derived
/// from it, exactly the way `potion_128m_native` derives from
/// `ModelManifest::potion_128m`, but over small fixture bytes.
fn native_fixture(tokenizer: &[u8], weights: &[u8]) -> (ModelManifest, ModelArtifactManifestV1) {
    use sha2::{Digest, Sha256};
    let sha = |bytes: &[u8]| {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        to_hex_lowercase(&hasher.finalize())
    };
    let mut manifest = make_test_manifest("tokenizer.json", tokenizer);
    manifest.files.push(ModelFile {
        name: "model.safetensors".to_owned(),
        sha256: sha(weights),
        size: u64::try_from(weights.len()).unwrap(),
        url: None,
    });
    manifest.dimension = Some(4);
    manifest.download_size_bytes = u64::try_from(tokenizer.len() + weights.len()).unwrap();
    let execution = ModelExecutionContractV1 {
        backend: "test-native".to_owned(),
        implementation_revision: "test-impl-v1".to_owned(),
        protocol_revision: "test-proto-v1".to_owned(),
        numeric_profile: "f32-test-v1".to_owned(),
        weights_format: "safetensors-f32-matrix-v1".to_owned(),
        tokenizer_family: "huggingface-tokenizers-json-v1".to_owned(),
        model_preprocessing: MODEL2VEC_PREPROCESSING_V1.to_owned(),
        sequence_policy: MODEL2VEC_SEQUENCE_POLICY_V1.to_owned(),
        pooling: MODEL2VEC_POOLING_V1.to_owned(),
        output_normalization: MODEL2VEC_OUTPUT_NORMALIZATION_V1.to_owned(),
        query_instruction: String::new(),
        document_instruction: String::new(),
        input_contract: default_plain_text_input_contract(),
        golden_vectors: GoldenVectorCertificateV1 {
            corpus_sha256: conformance_corpus_fingerprint().unwrap(),
            vectors_sha256: "f".repeat(64),
            vector_count: 4,
            dimension: 4,
        },
    };
    let native =
        ModelArtifactManifestV1::from_download_manifest(&manifest, "test-provider", execution)
            .expect("fixture native manifest must validate");
    (manifest, native)
}

#[test]
fn native_verify_dir_cached_falls_back_to_full_hash_without_a_receipt() {
    let tmp = tempfile::tempdir().unwrap();
    let (manifest, native) = native_fixture(b"{\"tokenizer\":1}", b"weights-for-test");
    write_temp_file(&tmp.path().join("tokenizer.json"), b"{\"tokenizer\":1}");
    write_temp_file(&tmp.path().join("model.safetensors"), b"weights-for-test");

    // No receipt yet: the cached entry point must behave exactly like the
    // full pass, succeed on matching bytes, and mint nothing.
    native.verify_dir_cached(&manifest, tmp.path()).unwrap();
    assert!(!tmp.path().join(VERIFIED_MARKER_FILE).exists());

    // Drifted bytes with no receipt are still caught by the full pass.
    write_temp_file(&tmp.path().join("model.safetensors"), b"weights-for-tesT");
    assert!(
        native.verify_dir_cached(&manifest, tmp.path()).is_err(),
        "without a receipt the cached path must still hash and reject drift"
    );
}

/// Prove the receipt path does not re-hash without resting on an
/// unreadable-file trick: a valid receipt must satisfy the cached path
/// with zero full hash passes (counted per thread by
/// `TEST_VERIFY_FILE_HASH_CALLS`), and a native manifest whose artifact set
/// no longer matches the download manifest must fall through to the real
/// hash pass. Unlike a chmod premise this holds in every environment:
/// `chmod 000` bumps the file's ctime, which the receipt binds
/// (`platform_change_stamp`), so the receipt is refused by design and the
/// cached path would re-hash — no environment exists where chmod proves
/// a skip.
#[test]
fn native_verify_dir_cached_skips_the_hash_pass_on_a_valid_receipt() {
    let tmp = tempfile::tempdir().unwrap();
    let (manifest, native) = native_fixture(b"{\"tokenizer\":2}", b"weights-cached");
    write_temp_file(&tmp.path().join("tokenizer.json"), b"{\"tokenizer\":2}");
    write_temp_file(&tmp.path().join("model.safetensors"), b"weights-cached");

    // Only the download-manifest authority mints the receipt.
    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    assert!(is_verification_cached(&manifest, tmp.path()));
    TEST_VERIFY_FILE_HASH_CALLS.with(|count| count.set(0));

    // A valid receipt must satisfy the native manifest outright, without
    // opening or hashing a single registered artifact.
    native
        .verify_dir_cached(&manifest, tmp.path())
        .expect("a valid receipt must satisfy the derived native manifest");
    TEST_VERIFY_FILE_HASH_CALLS.with(|count| {
        assert_eq!(count.get(), 0, "a valid receipt must skip the hash pass");
    });

    // A mismatched manifest must never borrow the receipt: the call
    // re-hashes every registered artifact (correct bytes here, so it
    // legitimately succeeds — the count is what proves the fall-through).
    let mut drifted = manifest.clone();
    let weights_entry = drifted
        .files
        .iter_mut()
        .find(|file| file.name == "model.safetensors")
        .expect("fixture must carry a weights entry");
    weights_entry.sha256 = "0".repeat(64);
    native
        .verify_dir_cached(&drifted, tmp.path())
        .expect("full verification of correct bytes must still succeed");
    TEST_VERIFY_FILE_HASH_CALLS.with(|count| {
        assert!(
            count.get() >= manifest.files.len(),
            "a mismatched manifest must fall through to the full hash pass"
        );
    });
}

#[test]
fn verification_cache_miss_when_same_id_revision_changes() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"revision-bound model data";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    assert!(is_verification_cached(&manifest, tmp.path()));

    let mut changed = manifest;
    changed.revision = "b".repeat(40);
    assert!(
        !is_verification_cached(&changed, tmp.path()),
        "same-ID revision evolution must invalidate the old receipt"
    );
}

#[test]
fn verification_cache_miss_when_same_id_sha_changes() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"checksum-bound model data";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    assert!(is_verification_cached(&manifest, tmp.path()));

    let mut changed = manifest;
    changed.files[0].sha256 = "b".repeat(64);
    assert!(
        !is_verification_cached(&changed, tmp.path()),
        "same-ID checksum evolution must invalidate the old receipt"
    );
}

#[cfg(unix)]
#[test]
fn verification_cache_rejects_metadata_mutation_that_bumps_the_change_stamp() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let content = b"change-stamp model data";
    let manifest = make_test_manifest("model.bin", content);
    let path = tmp.path().join("model.bin");
    write_temp_file(&path, content);

    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    assert!(is_verification_cached(&manifest, tmp.path()));

    // chmod updates the file's ctime; the receipt binds the platform
    // change stamp, so a metadata-only mutation (size, mtime, and file
    // identity unchanged) must still be refused. Pure stat observation:
    // identical for root and non-root, no readability premise.
    let original = std::fs::metadata(&path).unwrap().permissions();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let refused = !is_verification_cached(&manifest, tmp.path());
    std::fs::set_permissions(&path, original).unwrap();

    assert!(
        refused,
        "a ctime-bumping metadata mutation must invalidate the receipt"
    );
}

#[test]
fn verification_cache_rejects_legacy_marker_without_manifest_fingerprint() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"legacy marker model data";
    let manifest = make_test_manifest("model.bin", content);
    let path = tmp.path().join("model.bin");
    write_temp_file(&path, content);
    let state = capture_file_verification_state(&path).unwrap();
    let legacy = serde_json::json!({
        "manifest_id": manifest.id,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "verified_at": 1,
        "file_states": {
            "model.bin": {
                "size_bytes": state.size_bytes,
                "modified_unix_nanos": state.modified_unix_nanos
            }
        }
    });
    std::fs::write(
        tmp.path().join(VERIFIED_MARKER_FILE),
        serde_json::to_vec_pretty(&legacy).unwrap(),
    )
    .unwrap();

    assert!(
        !is_verification_cached(&manifest, tmp.path()),
        "pre-fingerprint markers must fail closed"
    );
}

#[test]
fn verification_cache_rejects_partial_current_marker() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"partial marker model data";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
    let raw = std::fs::read_to_string(&marker_path).unwrap();
    let mut marker: VerificationMarker = serde_json::from_str(&raw).unwrap();
    marker.file_states.clear();
    std::fs::write(&marker_path, serde_json::to_vec_pretty(&marker).unwrap()).unwrap();

    assert!(
        !is_verification_cached(&manifest, tmp.path()),
        "a current-schema receipt missing any manifest file state must fail closed"
    );
}

#[test]
fn verification_cache_rejects_unknown_marker_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"unknown marker field model data";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&tmp.path().join("model.bin"), content);

    verify_dir_and_record(&manifest, tmp.path()).unwrap();
    let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
    let raw = std::fs::read_to_string(&marker_path).unwrap();
    let mut marker: serde_json::Value = serde_json::from_str(&raw).unwrap();
    marker.as_object_mut().unwrap().insert(
        "unrecognized_trust_claim".to_owned(),
        serde_json::json!(true),
    );
    std::fs::write(&marker_path, serde_json::to_vec_pretty(&marker).unwrap()).unwrap();

    assert!(
        !is_verification_cached(&manifest, tmp.path()),
        "receipts with unknown trust claims must fail closed"
    );
}

#[test]
fn full_verify_and_record_rejects_corrupt_bytes_without_minting() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest = make_test_manifest("model.bin", b"registered bytes");
    write_temp_file(&tmp.path().join("model.bin"), b"corrupted bytes!");

    let error = verify_dir_and_record(&manifest, tmp.path()).unwrap_err();
    assert!(matches!(error, SearchError::HashMismatch { .. }));
    assert!(
        !tmp.path().join(VERIFIED_MARKER_FILE).exists(),
        "failed full verification must never mint a receipt"
    );
}

#[test]
fn full_verify_and_record_rejects_mutation_after_hash_before_receipt() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"stable registered bytes";
    let manifest = make_test_manifest("model.bin", content);
    let model_path = tmp.path().join("model.bin");
    write_temp_file(&model_path, content);

    TEST_AFTER_FULL_HASH_HOOK.with(|slot| {
        assert!(slot.borrow().is_none());
        slot.replace(Some(Box::new(move || {
            std::fs::write(model_path, b"changed after the full hash completed").unwrap();
        })));
    });

    let error = verify_dir_and_record(&manifest, tmp.path()).unwrap_err();
    assert!(matches!(error, SearchError::HashMismatch { .. }));
    assert!(
        !tmp.path().join(VERIFIED_MARKER_FILE).exists(),
        "a hash-vs-metadata race must fail before receipt publication"
    );
}

#[test]
fn verified_promotion_publishes_model_and_receipt_together() {
    let parent = tempfile::tempdir().unwrap();
    let staged = tempfile::tempdir_in(parent.path()).unwrap();
    let content = b"atomically promoted model bytes";
    let manifest = make_test_manifest("model.bin", content);
    write_temp_file(&staged.path().join("model.bin"), content);
    let destination = parent.path().join("published-model");

    manifest
        .promote_verified_installation(staged.path(), &destination)
        .unwrap();

    assert_eq!(
        std::fs::read(destination.join("model.bin")).unwrap(),
        content
    );
    assert!(destination.join(VERIFIED_MARKER_FILE).is_file());
    assert!(is_verification_cached(&manifest, &destination));
}

#[test]
fn partial_receipt_never_prevents_full_rehash() {
    let tmp = tempfile::tempdir().unwrap();
    let content = b"partial receipt bytes";
    let manifest = make_test_manifest("model.bin", content);
    let model_path = tmp.path().join("model.bin");
    write_temp_file(&model_path, content);
    verify_dir_and_record(&manifest, tmp.path()).unwrap();

    let marker_path = tmp.path().join(VERIFIED_MARKER_FILE);
    let raw = std::fs::read_to_string(&marker_path).unwrap();
    let mut marker: VerificationMarker = serde_json::from_str(&raw).unwrap();
    marker.file_states.clear();
    std::fs::write(&marker_path, serde_json::to_vec_pretty(&marker).unwrap()).unwrap();
    std::fs::write(model_path, b"corrupt partial bytes").unwrap();

    assert!(
        matches!(
            verify_dir_cached(&manifest, tmp.path()),
            Err(SearchError::HashMismatch { .. })
        ),
        "a partial receipt must fall through to full SHA verification"
    );
}

#[test]
fn built_in_verification_manifest_fingerprints_are_golden() {
    let potion = ModelManifest::potion_128m()
        .freeze_verification_manifest()
        .unwrap()
        .fingerprint;
    let minilm = ModelManifest::minilm_v2()
        .freeze_verification_manifest()
        .unwrap()
        .fingerprint;

    assert_eq!(
        potion,
        "32b58266fc633cf0e95c05d80bcb9c8f943786bb9e83aed6676eba8311b779e9"
    );
    assert_eq!(
        minilm,
        "5a563a081f9d3febe93302339766bb9ef314e170c8579a33c500f18fdcbe3f8e"
    );
}

#[test]
fn verify_dir_cached_rejects_non_production_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest = ModelManifest {
        id: "test".to_owned(),
        repo: "r".to_owned(),
        revision: "v".to_owned(),
        files: vec![ModelFile {
            name: "f.bin".to_owned(),
            sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
            size: 0,
            url: None,
        }],
        license: "MIT".to_owned(),
        tier: None,
        dimension: None,
        display_name: None,
        version: String::new(),
        description: None,
        download_size_bytes: 0,
    };
    let error = verify_dir_cached(&manifest, tmp.path()).unwrap_err();
    assert!(
        matches!(error, SearchError::InvalidConfig { .. }),
        "placeholder manifests must never receive cached admission: {error}"
    );
    assert!(!tmp.path().join(VERIFIED_MARKER_FILE).exists());
}
