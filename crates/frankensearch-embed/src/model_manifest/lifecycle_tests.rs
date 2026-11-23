//! Unit tests for manifest JSON parsing, lifecycle state machine, consent, and atomic promotion.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::error::SearchError;
use sha2::{Digest, Sha256};

use super::catalog::{ModelFile, ModelManifest, ModelManifestCatalog};
use super::frozen_tests::write_temp_file;
use super::lifecycle::{
    ConsentSource, DownloadConsent, ModelLifecycle, ModelState, resolve_download_consent_with_env,
};
#[cfg(unix)]
use super::verification::capture_file_verification_state;
use super::verification::{
    PublicationBoundary, PublicationFaultGuard, is_valid_sha256_hex, to_hex_lowercase,
    truncate_for_error, validate_model_file_name, verify_file_sha256,
};
use super::{PLACEHOLDER_PINNED_REVISION, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD};

#[test]
fn invalid_manifest_json_returns_clear_error() {
    let err = ModelManifest::from_json_str("{\"credential\":\"secret-token\",not-valid-json}")
        .unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
    assert!(err.to_string().contains("manifest JSON"));
    assert!(!err.to_string().contains("secret-token"));
}

#[test]
fn valid_manifest_json_round_trips_expected_fields() {
    let manifest = ModelManifest::from_json_str(
        r#"{
            "id":"test-model",
            "repo":"acme/test-model",
            "revision":"0123456789abcdef0123456789abcdef01234567",
            "files":[
                {
                    "name":"model.bin",
                    "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "size":42
                }
            ],
            "license":"MIT"
        }"#,
    )
    .unwrap();

    assert_eq!(manifest.id, "test-model");
    assert_eq!(manifest.repo, "acme/test-model");
    assert_eq!(manifest.total_size_bytes(), 42);
    assert!(manifest.has_verified_checksums());
    assert!(manifest.has_pinned_revision());
    assert!(manifest.is_production_ready());
}

#[test]
fn downloadable_manifest_rejects_log_control_characters_redacted() {
    let mut manifest = ModelManifest::potion_128m();
    manifest.id = "model\nforged-log-line".to_owned();
    let error = manifest.validate().unwrap_err();
    assert!(error.to_string().contains("control characters"));
    assert!(!error.to_string().contains("forged-log-line"));

    let mut manifest = ModelManifest::potion_128m();
    manifest.files[0].name = "tokenizer.json\nforged-log-line".to_owned();
    let error = manifest.validate().unwrap_err();
    assert!(error.to_string().contains("control characters"));
    assert!(!error.to_string().contains("forged-log-line"));

    let mut manifest = ModelManifest::potion_128m();
    manifest.files[0].sha256 = "digest\nforged-log-line".to_owned();
    let error = manifest.validate().unwrap_err();
    assert!(error.to_string().contains("redacted-invalid-sha256"));
    assert!(!error.to_string().contains("forged-log-line"));

    let mut manifest = ModelManifest::potion_128m();
    manifest.files[0].url =
        Some("https://models.example.invalid/tokenizer.json?token=secret".to_owned());
    let error = manifest.validate().unwrap_err();
    assert!(error.to_string().contains("credential-free HTTPS"));
    assert!(!error.to_string().contains("secret"));
}

#[test]
fn missing_required_manifest_field_is_rejected_with_redacted_input() {
    let err = ModelManifest::from_json_str(
        r#"{
            "id":"test-model",
            "repo":"acme/test-model",
            "revision":"0123456789abcdef0123456789abcdef01234567",
            "files":[]
        }"#,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        SearchError::InvalidConfig {
            ref field,
            ref value,
            ..
        // ubs:ignore — this is a public fixed redaction sentinel, not a secret.
        } if field == "manifest_json" && value == "redacted-manifest-json"
    ));
    assert!(err.to_string().contains("malformed or unsupported"));
    assert!(!err.to_string().contains("acme/test-model"));
}

#[test]
fn verify_file_sha256_success_wrong_hash_and_truncated() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.bin");
    let bytes = b"model-bytes";
    write_temp_file(&path, bytes);

    let expected_hash = to_hex_lowercase(&Sha256::digest(bytes));
    let expected_size = u64::try_from(bytes.len()).unwrap();
    verify_file_sha256(&path, &expected_hash, expected_size).unwrap();

    let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
    let err = verify_file_sha256(&path, wrong_hash, expected_size).unwrap_err();
    assert!(matches!(err, SearchError::HashMismatch { .. }));

    let err = verify_file_sha256(&path, &expected_hash, expected_size + 1).unwrap_err();
    assert!(matches!(err, SearchError::HashMismatch { .. }));
}

#[test]
fn verify_file_sha256_rejects_placeholder_invalid_hash_and_missing_file() {
    let temp = tempfile::tempdir().unwrap();
    let missing_path = temp.path().join("missing.bin");

    let err = verify_file_sha256(&missing_path, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, 1).unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));

    let err = verify_file_sha256(&missing_path, "digest\nforged-log-line", 1).unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
    assert!(err.to_string().contains("redacted-invalid-sha256"));
    assert!(!err.to_string().contains("forged-log-line"));

    let valid_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let err = verify_file_sha256(&missing_path, valid_hash, 1).unwrap_err();
    assert!(matches!(err, SearchError::ModelNotFound { .. }));
}

#[cfg(unix)]
#[test]
fn verify_file_sha256_rejects_symlinked_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("target.bin");
    let link = temp.path().join("model.bin");
    write_temp_file(&target, b"model-bytes");
    std::os::unix::fs::symlink(&target, &link).unwrap();

    let expected_hash = to_hex_lowercase(&Sha256::digest(b"model-bytes"));
    let error = verify_file_sha256(&link, &expected_hash, 11).unwrap_err();
    assert!(matches!(&error, SearchError::ModelLoadFailed { .. }));
    assert!(error.to_string().contains("symlink"));
    assert!(
        capture_file_verification_state(&link).is_none(),
        "verification cache metadata must not follow symlinks"
    );
}

#[test]
fn catalog_validate_reports_invalid_nested_manifest() {
    let catalog = ModelManifestCatalog::from_json_str(
        r#"{
            "models":[
                {
                    "id":"bad-model",
                    "repo":"acme/bad-model",
                    "revision":"0123456789abcdef0123456789abcdef01234567",
                    "files":[
                        {"name":"model.bin","sha256":"bad-hash","size":10}
                    ],
                    "license":"MIT"
                }
            ]
        }"#,
    )
    .unwrap();

    let err = catalog.validate().unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn lifecycle_state_machine_success_path() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );

    assert_eq!(lifecycle.state(), &ModelState::NotInstalled);

    lifecycle.begin_download(100).unwrap();
    lifecycle.update_download_progress(40).unwrap();
    lifecycle.begin_verification().unwrap();
    lifecycle.mark_staged_verified().unwrap();
    assert_eq!(lifecycle.state(), &ModelState::StagedVerified);
    lifecycle.mark_acquired_needs_reindex().unwrap();

    assert_eq!(lifecycle.state(), &ModelState::AcquiredNeedsReindex);
}

#[test]
fn lifecycle_state_machine_failure_path() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );

    lifecycle.begin_download(100).unwrap();
    lifecycle.fail_verification("checksum mismatch");
    assert!(matches!(
        lifecycle.state(),
        ModelState::VerificationFailed { .. }
    ));
}

#[test]
fn download_progress_percent_is_bounded_to_100() {
    let manifest = ModelManifest::minilm_v2();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );
    lifecycle.begin_download(10).unwrap();
    lifecycle.update_download_progress(10_000).unwrap();

    let progress_pct = match lifecycle.state() {
        ModelState::Downloading { progress_pct, .. } => *progress_pct,
        _ => 0,
    };
    assert!(progress_pct <= 100);
    assert_eq!(progress_pct, 100);
}

#[test]
fn placeholder_checksums_are_rejected_in_release_policy_mode() {
    let mut manifest = ModelManifest::minilm_v2();
    manifest.files[0].sha256 = PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned();
    manifest.files[0].size = 0;
    manifest.files[0].url = None;
    let err = manifest.validate_checksum_policy_for(true).unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn cancelled_state_can_recover() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );
    lifecycle.begin_download(10).unwrap();
    lifecycle.cancel();
    lifecycle.recover_after_cancel().unwrap();
    assert_eq!(lifecycle.state(), &ModelState::NotInstalled);
}

#[test]
fn empty_manifest_catalog_is_valid() {
    let catalog = ModelManifestCatalog::from_json_str(r#"{"models":[]}"#).unwrap();
    assert!(catalog.models.is_empty());
    catalog.validate().unwrap();
}

#[test]
fn unreadable_model_file_returns_clear_error() {
    let temp = tempfile::tempdir().unwrap();
    let model_root = temp.path();
    let bogus_path = model_root.join("tokenizer.json");
    fs::create_dir_all(&bogus_path).unwrap();

    let manifest = ModelManifest {
        id: "test".to_owned(),
        version: "test-v1".to_owned(),
        display_name: None,
        description: None,
        repo: "owner/repo".to_owned(),
        revision: "abcdef1".to_owned(),
        files: vec![ModelFile {
            name: "tokenizer.json".to_owned(),
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            size: 1,
            url: None,
        }],
        license: "MIT".to_owned(),
        dimension: None,
        tier: None,
        download_size_bytes: 0,
    };

    let err = manifest.verify_dir(model_root).unwrap_err();
    assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
    assert!(err.to_string().contains("regular file"));
}

#[test]
fn verify_dir_rejects_traversal_file_names_without_needing_validate_call() {
    let temp = tempfile::tempdir().expect("tempdir");
    let manifest = ModelManifest {
        id: "test".to_owned(),
        version: "test-v1".to_owned(),
        display_name: None,
        description: None,
        repo: "owner/repo".to_owned(),
        revision: "abcdef1".to_owned(),
        files: vec![ModelFile {
            name: "../escape.bin".to_owned(),
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            size: 0,
            url: None,
        }],
        license: "MIT".to_owned(),
        dimension: None,
        tier: None,
        download_size_bytes: 0,
    };

    let err = manifest
        .verify_dir(temp.path())
        .expect_err("must reject traversal");
    assert!(matches!(
        err,
        SearchError::InvalidConfig { ref field, .. } if field == "files[].name"
    ));
    assert!(err.to_string().contains("path traversal"));
}

#[test]
fn model_file_names_reject_noncanonical_portable_encodings() {
    for name in [
        "./model.safetensors",
        "C:/model.safetensors",
        "model.safetensors:alternate-stream",
        "onnx//model.onnx",
        "onnx\\model.onnx",
        "tokenizer.json/",
        "onnx/model?.onnx",
        "weights*/model.onnx",
        "NUL",
        "com1.bin",
        "COM¹.txt",
        "LPT9.config",
        "clock$",
        "CONOUT$/model.bin",
        "model.onnx.",
        " model.onnx",
        "model.onnx ",
    ] {
        let error = validate_model_file_name(name).unwrap_err();
        // Windows recognizes the drive prefix before the portable-name
        // check; other hosts reject its colon in the portable-name check.
        let reason = if cfg!(windows) && name == "C:/model.safetensors" {
            "relative path without root"
        } else {
            "canonical portable"
        };
        assert!(error.to_string().contains(reason), "{name}: {error}");
        assert!(!error.to_string().contains(name));
    }
}

#[test]
fn can_register_and_lookup_custom_manifest() {
    let unique_id = format!(
        "custom-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let manifest = ModelManifest {
        id: unique_id.clone(),
        version: "test-v1".to_owned(),
        display_name: None,
        description: None,
        repo: "acme/custom".to_owned(),
        revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        files: vec![ModelFile {
            name: "weights.bin".to_owned(),
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            size: 42,
            url: None,
        }],
        license: "MIT".to_owned(),
        dimension: None,
        tier: None,
        download_size_bytes: 0,
    };

    manifest.clone().register().unwrap();
    let loaded = ModelManifest::lookup(&unique_id).unwrap();
    assert_eq!(loaded, manifest);
}

#[test]
fn resolve_download_consent_priority_order() {
    let consent = resolve_download_consent_with_env(Some(false), Some("1"), Some(true), Some(true));
    assert_eq!(consent.source, Some(ConsentSource::Programmatic));
    assert!(!consent.granted);

    let consent = resolve_download_consent_with_env(None, Some("1"), Some(false), Some(true));
    assert_eq!(consent.source, Some(ConsentSource::Environment));
    assert!(consent.granted);

    let consent = resolve_download_consent_with_env(None, None, Some(false), Some(true));
    assert_eq!(consent.source, Some(ConsentSource::Interactive));
    assert!(!consent.granted);
}

// ── bd-3un.51: Additional coverage ───────────────────────────────

#[test]
fn valid_manifest_parses_all_fields() {
    let json = r#"{
        "id": "test-model",
        "repo": "owner/test-model",
        "revision": "abc123def456",
        "files": [
            {"name": "model.onnx", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 1024},
            {"name": "tokenizer.json", "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "size": 512}
        ],
        "license": "Apache-2.0"
    }"#;
    let manifest = ModelManifest::from_json_str(json).unwrap();
    assert_eq!(manifest.id, "test-model");
    assert_eq!(manifest.repo, "owner/test-model");
    assert_eq!(manifest.revision, "abc123def456");
    assert_eq!(manifest.files.len(), 2);
    assert_eq!(manifest.files[0].name, "model.onnx");
    assert_eq!(manifest.files[1].size, 512);
    assert_eq!(manifest.license, "Apache-2.0");
}

#[test]
fn missing_id_field_returns_clear_error() {
    let json = r#"{"id": "", "repo": "r", "revision": "v", "files": [], "license": "MIT"}"#;
    let err = ModelManifest::from_json_str(json).unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn missing_repo_field_returns_clear_error() {
    let json = r#"{"id": "m", "repo": " ", "revision": "v", "files": [], "license": "MIT"}"#;
    let err = ModelManifest::from_json_str(json).unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn missing_revision_field_returns_clear_error() {
    let json = r#"{"id": "m", "repo": "r", "revision": "", "files": [], "license": "MIT"}"#;
    let err = ModelManifest::from_json_str(json).unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn missing_license_field_returns_clear_error() {
    let json = r#"{"id": "m", "repo": "r", "revision": "v", "files": [], "license": ""}"#;
    let err = ModelManifest::from_json_str(json).unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn invalid_sha256_format_rejected() {
    let json = r#"{
        "id": "m", "repo": "r", "revision": "v", "license": "MIT",
        "files": [{"name": "f.bin", "sha256": "not-a-valid-hash", "size": 1}]
    }"#;
    let err = ModelManifest::from_json_str(json).unwrap_err();
    assert!(err.to_string().contains("SHA256 hex"));
}

#[test]
fn file_with_zero_size_and_valid_hash_accepted() {
    let json = r#"{
        "id": "m", "repo": "r", "revision": "v", "license": "MIT",
        "files": [{"name": "f.bin", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 0}]
    }"#;
    let manifest = ModelManifest::from_json_str(json).unwrap();
    assert_eq!(manifest.files[0].size, 0);
}

#[test]
fn empty_file_name_rejected() {
    let json = r#"{
        "id": "m", "repo": "r", "revision": "v", "license": "MIT",
        "files": [{"name": "", "sha256": "PLACEHOLDER_VERIFY_AFTER_DOWNLOAD", "size": 0}]
    }"#;
    let err = ModelManifest::from_json_str(json).unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn verify_missing_file_returns_model_not_found() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("does_not_exist.bin");
    let hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let err = verify_file_sha256(&path, hash, 100).unwrap_err();
    assert!(matches!(err, SearchError::ModelNotFound { .. }));
}

#[test]
fn verify_placeholder_checksum_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("file.bin");
    write_temp_file(&path, b"data");
    let err = verify_file_sha256(&path, PLACEHOLDER_VERIFY_AFTER_DOWNLOAD, 4).unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
    assert!(err.to_string().contains("placeholder"));
}

#[test]
fn verify_zero_expected_size_accepts_empty_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("empty.bin");
    write_temp_file(&path, b"");
    let hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    verify_file_sha256(&path, hash, 0).unwrap();
}

#[test]
fn verify_zero_expected_size_still_rejects_non_empty_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("file.bin");
    write_temp_file(&path, b"data");
    let hash = to_hex_lowercase(&Sha256::digest(b"data"));
    let err = verify_file_sha256(&path, &hash, 0).unwrap_err();
    assert!(matches!(err, SearchError::HashMismatch { .. }));
}

#[test]
fn to_pretty_json_roundtrip() {
    let manifest = ModelManifest::potion_128m();
    let json = manifest.to_pretty_json().unwrap();
    let restored = ModelManifest::from_json_str(&json).unwrap();
    assert_eq!(restored.id, manifest.id);
    assert_eq!(restored.files.len(), manifest.files.len());
}

#[test]
fn builtin_manifests_validate() {
    ModelManifest::minilm_v2().validate().unwrap();
    ModelManifest::potion_128m().validate().unwrap();
}

#[test]
fn builtin_manifests_are_production_ready() {
    assert!(ModelManifest::minilm_v2().is_production_ready());
    assert!(ModelManifest::potion_128m().is_production_ready());
    assert!(ModelManifest::ms_marco_reranker().is_production_ready());
}

#[test]
fn has_pinned_revision_rejects_every_non_digest_alias() {
    for alias in &[
        "main",
        "latest",
        "HEAD",
        "legacy-default-branch",
        PLACEHOLDER_PINNED_REVISION,
    ] {
        let m = ModelManifest {
            revision: alias.to_string(),
            ..ModelManifest::potion_128m()
        };
        assert!(
            !m.has_pinned_revision(),
            "'{alias}' should not be considered pinned"
        );
    }
}

#[test]
fn has_pinned_revision_accepts_commit_sha() {
    let m = ModelManifest {
        revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        ..ModelManifest::potion_128m()
    };
    assert!(m.has_pinned_revision());
}

#[test]
fn total_size_bytes_sums_all_files() {
    let m = ModelManifest {
        files: vec![
            ModelFile {
                name: "a".to_owned(),
                sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                size: 100,
                url: None,
            },
            ModelFile {
                name: "b".to_owned(),
                sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
                size: 200,
                url: None,
            },
        ],
        download_size_bytes: 0,
        ..ModelManifest::potion_128m()
    };
    assert_eq!(m.total_size_bytes(), 300);
}

#[test]
fn model_state_serde_roundtrip() {
    let states = vec![
        ModelState::NotInstalled,
        ModelState::NeedsConsent,
        ModelState::Downloading {
            progress_pct: 50,
            bytes_downloaded: 1000,
            total_bytes: 2000,
        },
        ModelState::Verifying,
        ModelState::StagedVerified,
        ModelState::AcquiredNeedsReindex,
        ModelState::Ready,
        ModelState::Disabled {
            reason: "out of disk".to_owned(),
        },
        ModelState::VerificationFailed {
            reason: "hash mismatch".to_owned(),
        },
        ModelState::UpdateAvailable {
            current_revision: "old".to_owned(),
            latest_revision: "new".to_owned(),
        },
        ModelState::Cancelled,
    ];
    for state in &states {
        let json = serde_json::to_string(state).unwrap();
        let decoded: ModelState = serde_json::from_str(&json).unwrap();
        assert_eq!(&decoded, state);
    }
}

#[test]
fn consent_source_serde_roundtrip() {
    for source in &[
        ConsentSource::Programmatic,
        ConsentSource::Environment,
        ConsentSource::Interactive,
        ConsentSource::ConfigFile,
    ] {
        let json = serde_json::to_string(source).unwrap();
        let decoded: ConsentSource = serde_json::from_str(&json).unwrap();
        assert_eq!(&decoded, source);
    }
}

#[test]
fn lifecycle_needs_consent_when_not_granted() {
    let manifest = ModelManifest::potion_128m();
    let lifecycle = ModelLifecycle::new(manifest, DownloadConsent::denied(None));
    assert_eq!(lifecycle.state(), &ModelState::NeedsConsent);
}

#[test]
fn lifecycle_begin_download_without_consent_fails() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(manifest, DownloadConsent::denied(None));
    let err = lifecycle.begin_download(100).unwrap_err();
    assert!(matches!(err, SearchError::EmbedderUnavailable { .. }));
}

#[test]
fn lifecycle_begin_download_zero_bytes_fails() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );
    let err = lifecycle.begin_download(0).unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn lifecycle_approve_consent_transitions() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(manifest, DownloadConsent::denied(None));
    assert_eq!(lifecycle.state(), &ModelState::NeedsConsent);

    lifecycle.approve_consent(ConsentSource::Interactive);
    assert_eq!(lifecycle.state(), &ModelState::NotInstalled);
}

#[test]
fn lifecycle_disable_and_update() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );

    lifecycle.disable("maintenance");
    assert!(matches!(lifecycle.state(), ModelState::Disabled { .. }));

    lifecycle.mark_update_available("v1", "v2");
    assert!(matches!(
        lifecycle.state(),
        ModelState::UpdateAvailable { .. }
    ));
}

#[test]
fn lifecycle_recovery_from_non_cancelled_fails() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );
    let err = lifecycle.recover_after_cancel().unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn lifecycle_begin_verification_from_not_downloading_fails() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );
    let err = lifecycle.begin_verification().unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn lifecycle_update_progress_from_not_downloading_fails() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );
    let err = lifecycle.update_download_progress(50).unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn detect_update_state_same_revision_returns_none() {
    let m = ModelManifest {
        revision: "abc123".to_owned(),
        ..ModelManifest::potion_128m()
    };
    assert!(m.detect_update_state("abc123").is_none());
}

#[test]
fn detect_update_state_different_revision_returns_update() {
    let latest_revision = "a".repeat(40);
    let installed_revision = "b".repeat(40);
    let m = ModelManifest {
        revision: latest_revision,
        ..ModelManifest::potion_128m()
    };
    let state = m.detect_update_state(&installed_revision).unwrap();
    assert!(matches!(state, ModelState::UpdateAvailable { .. }));
}

#[test]
fn detect_update_state_unpinned_returns_none() {
    let manifest = ModelManifest {
        revision: PLACEHOLDER_PINNED_REVISION.to_owned(),
        ..ModelManifest::potion_128m()
    };
    assert!(manifest.detect_update_state("anything").is_none());
}

#[test]
fn resolve_consent_config_file_path() {
    let consent = resolve_download_consent_with_env(None, None, None, Some(true));
    assert_eq!(consent.source, Some(ConsentSource::ConfigFile));
    assert!(consent.granted);
}

#[test]
fn resolve_consent_no_source_denies() {
    let consent = resolve_download_consent_with_env(None, None, None, None);
    assert!(!consent.granted);
    assert!(consent.source.is_none());
}

#[test]
fn resolve_consent_env_values() {
    for (val, expected) in &[
        ("1", true),
        ("true", true),
        ("yes", true),
        ("on", true),
        ("0", false),
        ("false", false),
        ("no", false),
        ("off", false),
    ] {
        let consent = resolve_download_consent_with_env(None, Some(val), None, None);
        assert_eq!(consent.granted, *expected, "env={val}");
    }
}

#[test]
fn resolve_consent_invalid_env_skipped() {
    let consent = resolve_download_consent_with_env(None, Some("maybe"), Some(true), None);
    assert_eq!(consent.source, Some(ConsentSource::Interactive));
    assert!(consent.granted);
}

#[test]
fn model_file_placeholder_detection() {
    let file = ModelFile {
        name: "f.bin".to_owned(),
        sha256: PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned(),
        size: 0,
        url: None,
    };
    assert!(file.uses_placeholder_checksum());
    assert!(!file.has_verified_checksum());
}

#[test]
fn model_file_verified_checksum_detection() {
    let file = ModelFile {
        name: "f.bin".to_owned(),
        sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        size: 42,
        url: None,
    };
    assert!(!file.uses_placeholder_checksum());
    assert!(file.has_verified_checksum());
}

#[test]
fn model_file_zero_byte_verified_checksum_detection() {
    let file = ModelFile {
        name: "empty.bin".to_owned(),
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_owned(),
        size: 0,
        url: None,
    };
    assert!(!file.uses_placeholder_checksum());
    assert!(file.has_verified_checksum());
}

#[test]
fn promote_verified_installation_success() {
    let temp = tempfile::tempdir().unwrap();
    let staged = temp.path().join("staged");
    let dest = temp.path().join("final");
    fs::create_dir_all(&staged).unwrap();

    let data = b"model data";
    write_temp_file(&staged.join("model.bin"), data);
    let hash = to_hex_lowercase(&Sha256::digest(data));
    let size = u64::try_from(data.len()).unwrap();

    let manifest = ModelManifest {
        id: "test".to_owned(),
        version: "test-v1".to_owned(),
        display_name: None,
        description: None,
        repo: "owner/repo".to_owned(),
        revision: "a".repeat(40),
        files: vec![ModelFile {
            name: "model.bin".to_owned(),
            sha256: hash,
            size,
            url: None,
        }],
        license: "MIT".to_owned(),
        dimension: None,
        tier: None,
        download_size_bytes: 0,
    };

    let backup = manifest
        .promote_verified_installation(&staged, &dest)
        .unwrap();
    assert!(backup.is_none());
    assert!(dest.join("model.bin").exists());
}

#[test]
fn promote_verified_creates_backup_of_existing() {
    let temp = tempfile::tempdir().unwrap();
    let staged = temp.path().join("staged");
    let dest = temp.path().join("final");
    fs::create_dir_all(&staged).unwrap();
    fs::create_dir_all(&dest).unwrap();
    write_temp_file(&dest.join("old.bin"), b"old");

    let data = b"new model";
    write_temp_file(&staged.join("model.bin"), data);
    let hash = to_hex_lowercase(&Sha256::digest(data));
    let size = u64::try_from(data.len()).unwrap();

    let manifest = ModelManifest {
        id: "test".to_owned(),
        version: "test-v1".to_owned(),
        display_name: None,
        description: None,
        repo: "owner/repo".to_owned(),
        revision: "a".repeat(40),
        files: vec![ModelFile {
            name: "model.bin".to_owned(),
            sha256: hash,
            size,
            url: None,
        }],
        license: "MIT".to_owned(),
        dimension: None,
        tier: None,
        download_size_bytes: 0,
    };

    let backup = manifest
        .promote_verified_installation(&staged, &dest)
        .unwrap();
    assert!(backup.is_some());
    assert!(dest.join("model.bin").exists());
}

fn publication_failure_fixture() -> (tempfile::TempDir, ModelManifest, PathBuf, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let staged = temp.path().join("staged");
    let destination = temp.path().join("final");
    fs::create_dir_all(&staged).unwrap();
    fs::create_dir_all(&destination).unwrap();
    write_temp_file(&staged.join("model.bin"), b"new model");
    write_temp_file(&destination.join("prior.bin"), b"prior model");
    let manifest = ModelManifest {
        id: "publication-fault-fixture".to_owned(),
        version: "test-v1".to_owned(),
        display_name: None,
        description: None,
        repo: "owner/repo".to_owned(),
        revision: "a".repeat(40),
        files: vec![ModelFile {
            name: "model.bin".to_owned(),
            sha256: to_hex_lowercase(&Sha256::digest(b"new model")),
            size: 9,
            url: None,
        }],
        license: "MIT".to_owned(),
        dimension: None,
        tier: None,
        download_size_bytes: 9,
    };
    (temp, manifest, staged, destination)
}

fn sibling_names_with_prefix(parent: &Path, prefix: &str) -> Vec<String> {
    fs::read_dir(parent)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with(prefix))
        .collect()
}

#[test]
fn interrupted_installing_parent_fsync_preserves_prior_generation() {
    let (temp, manifest, staged, destination) = publication_failure_fixture();
    let _fault = PublicationFaultGuard::install(PublicationBoundary::InstallingParentSync);

    let error = manifest
        .promote_verified_installation(&staged, &destination)
        .unwrap_err();

    assert!(matches!(error, SearchError::Io(_)));
    assert_eq!(
        fs::read(destination.join("prior.bin")).unwrap(),
        b"prior model"
    );
    assert_eq!(
        sibling_names_with_prefix(temp.path(), ".final.installing.").len(),
        1
    );
}

#[test]
fn interrupted_backup_parent_fsync_rolls_prior_generation_back() {
    let (temp, manifest, staged, destination) = publication_failure_fixture();
    let _fault = PublicationFaultGuard::install(PublicationBoundary::BackupParentSync);

    let error = manifest
        .promote_verified_installation(&staged, &destination)
        .unwrap_err();

    assert!(matches!(error, SearchError::Io(_)));
    assert_eq!(
        fs::read(destination.join("prior.bin")).unwrap(),
        b"prior model"
    );
    assert_eq!(
        sibling_names_with_prefix(temp.path(), ".final.installing.").len(),
        1
    );
}

#[test]
fn interrupted_publish_rename_rolls_prior_generation_back() {
    let (temp, manifest, staged, destination) = publication_failure_fixture();
    let _fault = PublicationFaultGuard::install(PublicationBoundary::PublishRename);

    let error = manifest
        .promote_verified_installation(&staged, &destination)
        .unwrap_err();

    assert!(matches!(error, SearchError::Io(_)));
    assert_eq!(
        fs::read(destination.join("prior.bin")).unwrap(),
        b"prior model"
    );
    assert_eq!(
        sibling_names_with_prefix(temp.path(), ".final.installing.").len(),
        1
    );
}

#[test]
fn interrupted_published_parent_fsync_preserves_prior_backup() {
    let (temp, manifest, staged, destination) = publication_failure_fixture();
    let _fault = PublicationFaultGuard::install(PublicationBoundary::PublishedParentSync);

    let error = manifest
        .promote_verified_installation(&staged, &destination)
        .unwrap_err();

    assert!(matches!(error, SearchError::Io(_)));
    assert_eq!(
        fs::read(destination.join("model.bin")).unwrap(),
        b"new model"
    );
    let backups = sibling_names_with_prefix(temp.path(), "final.backup.");
    assert_eq!(backups.len(), 1);
    assert_eq!(
        fs::read(temp.path().join(&backups[0]).join("prior.bin")).unwrap(),
        b"prior model"
    );
}

#[test]
fn manifest_catalog_with_multiple_models() {
    let json = r#"{"models": [
        {"id": "m1", "repo": "r1", "revision": "v1", "files": [], "license": "MIT"},
        {"id": "m2", "repo": "r2", "revision": "v2", "files": [], "license": "Apache-2.0"}
    ]}"#;
    let catalog = ModelManifestCatalog::from_json_str(json).unwrap();
    assert_eq!(catalog.models.len(), 2);
    catalog.validate().unwrap();
}

#[test]
fn manifest_catalog_invalid_model_fails_validation() {
    let json = r#"{"models": [
        {"id": "", "repo": "r", "revision": "v", "files": [], "license": "MIT"}
    ]}"#;
    let catalog = ModelManifestCatalog::from_json_str(json).unwrap();
    let err = catalog.validate().unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn is_valid_sha256_hex_checks() {
    assert!(is_valid_sha256_hex(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ));
    assert!(!is_valid_sha256_hex("short"));
    // Uppercase rejected.
    assert!(!is_valid_sha256_hex(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    ));
    // Invalid hex chars rejected.
    assert!(!is_valid_sha256_hex(
        "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg"
    ));
}

#[test]
fn download_consent_constructors() {
    let granted = DownloadConsent::granted(ConsentSource::Programmatic);
    assert!(granted.granted);
    assert_eq!(granted.source, Some(ConsentSource::Programmatic));

    let denied = DownloadConsent::denied(Some(ConsentSource::Environment));
    assert!(!denied.granted);
    assert_eq!(denied.source, Some(ConsentSource::Environment));

    let denied_none = DownloadConsent::denied(None);
    assert!(!denied_none.granted);
    assert!(denied_none.source.is_none());
}

#[test]
fn lifecycle_can_restart_after_verification_failure() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );

    lifecycle.begin_download(100).unwrap();
    lifecycle.fail_verification("bad hash");
    assert!(matches!(
        lifecycle.state(),
        ModelState::VerificationFailed { .. }
    ));

    lifecycle.begin_download(100).unwrap();
    assert!(matches!(lifecycle.state(), ModelState::Downloading { .. }));
}

#[test]
fn lifecycle_double_begin_download_from_ready_fails() {
    let manifest = ModelManifest::potion_128m();
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );

    lifecycle.begin_download(100).unwrap();
    lifecycle.begin_verification().unwrap();
    lifecycle.mark_ready();

    let err = lifecycle.begin_download(200).unwrap_err();
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn truncate_for_error_short_passthrough() {
    let short = "hello world";
    assert_eq!(truncate_for_error(short), "hello world");
}

#[test]
fn truncate_for_error_long_truncated() {
    let long = "x".repeat(200);
    let result = truncate_for_error(&long);
    assert!(result.ends_with("..."));
    assert!(result.len() < 200);
}
