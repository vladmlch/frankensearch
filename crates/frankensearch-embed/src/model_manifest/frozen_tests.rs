//! Unit tests for frozen artifact manifests and producer identity compatibility.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use frankensearch_core::error::SearchError;
use frankensearch_core::generation::{
    ForeignProducerConformanceCertificateV1, GoldenVectorCertificateV1,
    ProducerCompatibilityErrorV1, ProducerCompatibilityKindV1, QuantizationFormat,
    TrustedProducerConformanceContextV1, VerifiedGoldenConformanceManifestV1,
};
use sha2::{Digest, Sha256};

use super::catalog::ModelManifest;
use super::frozen::{
    ModelArtifactFileV1, ModelArtifactManifestV1, ModelArtifactRoleV1, ModelExecutionContractV1,
    default_plain_text_input_contract, license_metadata_fingerprint, sha256_hex_bytes,
};
use super::verification::to_hex_lowercase;
use super::{
    FASTEMBED_SEQUENCE_POLICY_256, FASTEMBED_SEQUENCE_POLICY_512, MAX_FROZEN_MANIFEST_FIELD_BYTES,
    MODEL_ARTIFACT_MANIFEST_SCHEMA_V1, MODEL_CONFORMANCE_TEXTS_V1,
};

pub(super) fn write_temp_file(path: &Path, bytes: &[u8]) {
    let mut file = File::create(path).unwrap();
    file.write_all(bytes).unwrap();
    file.flush().unwrap();
}

fn sample_artifact_manifest() -> ModelArtifactManifestV1 {
    let input_contract = default_plain_text_input_contract();
    ModelArtifactManifestV1 {
        schema_version: MODEL_ARTIFACT_MANIFEST_SCHEMA_V1,
        provider: "fixture-provider".to_owned(),
        logical_model_id: "fixture-semantic-model".to_owned(),
        upstream_revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        upstream_repository: "fixture/model".to_owned(),
        artifacts: vec![
            ModelArtifactFileV1 {
                role: ModelArtifactRoleV1::Weights,
                relative_path: "model.safetensors".to_owned(),
                upstream_url: "https://models.example.invalid/immutable/model.safetensors"
                    .to_owned(),
                size: 7,
                sha256: to_hex_lowercase(&Sha256::digest(b"weights")),
            },
            ModelArtifactFileV1 {
                role: ModelArtifactRoleV1::Tokenizer,
                relative_path: "tokenizer.json".to_owned(),
                upstream_url: "https://models.example.invalid/immutable/tokenizer.json".to_owned(),
                size: 9,
                sha256: to_hex_lowercase(&Sha256::digest(b"tokenizer")),
            },
        ],
        license_spdx: "Apache-2.0".to_owned(),
        license_metadata_sha256: license_metadata_fingerprint(
            "Apache-2.0",
            "fixture-provider",
            "fixture/model",
            "0123456789abcdef0123456789abcdef01234567",
        ),
        dimension: 3,
        execution: ModelExecutionContractV1 {
            backend: "fixture-native".to_owned(),
            implementation_revision: "fixture-implementation-v1".to_owned(),
            protocol_revision: "fixture-protocol-v1".to_owned(),
            numeric_profile: "f32-fixture-v1".to_owned(),
            weights_format: "safetensors-f32-v1".to_owned(),
            tokenizer_family: "fixture-tokenizer-v1".to_owned(),
            model_preprocessing: "fixture-preprocessing-v1".to_owned(),
            sequence_policy: "max-length=8;truncate-right;no-padding".to_owned(),
            pooling: "mean-v1".to_owned(),
            output_normalization: "l2-f32-v1".to_owned(),
            query_instruction: "query: ".to_owned(),
            document_instruction: "document: ".to_owned(),
            input_contract,
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: "1".repeat(64),
                vectors_sha256: "2".repeat(64),
                vector_count: 2,
                dimension: 3,
            },
        },
    }
}

#[test]
fn frozen_artifact_manifest_is_role_order_canonical_and_exact() {
    let left = sample_artifact_manifest();
    left.validate().unwrap();
    let left_frozen = left.freeze().unwrap();
    left_frozen.validate().unwrap();

    let mut right = left;
    right.artifacts.reverse();
    right.validate().unwrap();
    let right_frozen = right.freeze().unwrap();
    assert_eq!(left_frozen.canonical_bytes, right_frozen.canonical_bytes);
    assert_eq!(left_frozen.fingerprint, right_frozen.fingerprint);
}

#[test]
fn foreign_producers_require_explicit_trusted_conformance_certificates() {
    let conformance_texts = ["query: alpha", "document: beta"];
    let conformance_vectors = vec![vec![0.0, -0.0, 1.0], vec![0.5, -0.5, 0.25]];
    let fixture = VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(
        &conformance_texts,
        &conformance_vectors,
        &conformance_vectors,
    )
    .unwrap();
    let policy_fingerprint = "9".repeat(64);

    let mut left = sample_artifact_manifest();
    left.execution.golden_vectors = fixture.certificate().clone();
    let left_manifest_fingerprint = left.freeze().unwrap().fingerprint;
    let left_identity = left
        .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();

    let mut right = left.clone();
    right.execution.backend = "fixture-alternate-backend".to_owned();
    right.execution.implementation_revision = "fixture-implementation-v2".to_owned();
    right.execution.protocol_revision = "fixture-protocol-v2".to_owned();
    right.execution.numeric_profile = "f32-fixture-alternate-kernel-v1".to_owned();
    let right_manifest_fingerprint = right.freeze().unwrap().fingerprint;
    let right_identity = right
        .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();

    assert_ne!(left_manifest_fingerprint, right_manifest_fingerprint);
    assert_eq!(
        left_identity.producer.provenance_manifest_fingerprint,
        left_manifest_fingerprint
    );
    assert_eq!(
        right_identity.producer.provenance_manifest_fingerprint,
        right_manifest_fingerprint
    );
    assert_eq!(
        left_identity.space.fingerprint(),
        right_identity.space.fingerprint()
    );
    assert_eq!(
        left_identity.verify_exact_producer_with(&right_identity),
        Err(ProducerCompatibilityErrorV1::CertificateRequired)
    );
    let right_certificate =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &left_identity,
            &right_identity,
            &fixture,
            &policy_fingerprint,
            1,
            100,
            200,
        )
        .unwrap();
    let right_certificate_fingerprint = right_certificate.fingerprint();
    let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
        &policy_fingerprint,
        &right_certificate_fingerprint,
        &fixture,
        150,
        1,
        1,
    )
    .unwrap();
    assert_eq!(
        left_identity
            .verify_certified_foreign_producer_with(&right_identity, &right_certificate, trusted,)
            .unwrap()
            .kind(),
        ProducerCompatibilityKindV1::Certified
    );
    assert_ne!(
        left_identity.producer.fingerprint(),
        right_identity.producer.fingerprint()
    );
    assert_ne!(left_identity.fingerprint(), right_identity.fingerprint());

    let mut redistributed = left.clone();
    redistributed.provider = "fixture-mirror-provider".to_owned();
    redistributed.upstream_repository = "fixture/model-mirror".to_owned();
    redistributed.license_spdx = "MIT".to_owned();
    redistributed.license_metadata_sha256 = license_metadata_fingerprint(
        &redistributed.license_spdx,
        &redistributed.provider,
        &redistributed.upstream_repository,
        &redistributed.upstream_revision,
    );
    for artifact in &mut redistributed.artifacts {
        artifact.upstream_url = format!(
            "https://mirror.example.invalid/immutable/{}",
            artifact.relative_path
        );
    }
    let redistributed_identity = redistributed
        .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();
    assert_eq!(
        left_identity.space.fingerprint(),
        redistributed_identity.space.fingerprint()
    );
    assert_ne!(
        left_identity.producer.fingerprint(),
        redistributed_identity.producer.fingerprint()
    );
    assert_eq!(
        left_identity.verify_exact_producer_with(&redistributed_identity),
        Err(ProducerCompatibilityErrorV1::CertificateRequired)
    );
    let redistributed_certificate =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &left_identity,
            &redistributed_identity,
            &fixture,
            &policy_fingerprint,
            2,
            100,
            200,
        )
        .unwrap();
    let redistributed_certificate_fingerprint = redistributed_certificate.fingerprint();
    let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
        &policy_fingerprint,
        &redistributed_certificate_fingerprint,
        &fixture,
        150,
        2,
        2,
    )
    .unwrap();
    assert_eq!(
        left_identity
            .verify_certified_foreign_producer_with(
                &redistributed_identity,
                &redistributed_certificate,
                trusted,
            )
            .unwrap()
            .kind(),
        ProducerCompatibilityKindV1::Certified,
        "distribution metadata may differ only with explicit trusted evidence"
    );

    let mut nonconformant = right.clone();
    nonconformant.execution.golden_vectors.vectors_sha256 = "8".repeat(64);
    let nonconformant_identity = nonconformant
        .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();
    assert_eq!(
        left_identity.space.fingerprint(),
        nonconformant_identity.space.fingerprint()
    );
    assert_eq!(
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &left_identity,
            &nonconformant_identity,
            &fixture,
            &policy_fingerprint,
            3,
            100,
            200,
        ),
        Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch)
    );

    let mut changed_semantics = right;
    changed_semantics.execution.pooling.push_str("-drift");
    assert_ne!(
        left_identity.space.fingerprint(),
        changed_semantics
            .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap()
            .space
            .fingerprint()
    );
}

#[test]
fn every_artifact_manifest_field_participates_in_the_fingerprint() {
    let base = sample_artifact_manifest();
    let base_fingerprint = sha256_hex_bytes(&base.canonical_bytes());
    macro_rules! changed {
        ($label:literal, $mutate:expr) => {{
            let mut candidate = base.clone();
            $mutate(&mut candidate);
            assert_ne!(
                base_fingerprint,
                sha256_hex_bytes(&candidate.canonical_bytes()),
                $label
            );
        }};
    }

    changed!("schema", |m: &mut ModelArtifactManifestV1| m
        .schema_version +=
        1);
    changed!("provider", |m: &mut ModelArtifactManifestV1| m
        .provider
        .push('x'));
    changed!("model", |m: &mut ModelArtifactManifestV1| m
        .logical_model_id
        .push('x'));
    changed!("revision", |m: &mut ModelArtifactManifestV1| m
        .upstream_revision
        .push('x'));
    changed!("repository", |m: &mut ModelArtifactManifestV1| m
        .upstream_repository
        .push('x'));
    changed!("role", |m: &mut ModelArtifactManifestV1| m.artifacts[0]
        .role =
        ModelArtifactRoleV1::Projection);
    changed!("path", |m: &mut ModelArtifactManifestV1| m.artifacts[0]
        .relative_path
        .push('x'));
    changed!("url", |m: &mut ModelArtifactManifestV1| m.artifacts[0]
        .upstream_url
        .push('x'));
    changed!("size", |m: &mut ModelArtifactManifestV1| m.artifacts[0]
        .size += 1);
    changed!("artifact digest", |m: &mut ModelArtifactManifestV1| m
        .artifacts[0]
        .sha256 =
        "3".repeat(64));
    changed!("license", |m: &mut ModelArtifactManifestV1| m
        .license_spdx
        .push('x'));
    changed!("license digest", |m: &mut ModelArtifactManifestV1| m
        .license_metadata_sha256 =
        "4".repeat(64));
    changed!("dimension", |m: &mut ModelArtifactManifestV1| m
        .dimension +=
        1);
    changed!("backend", |m: &mut ModelArtifactManifestV1| m
        .execution
        .backend
        .push('x'));
    changed!("implementation", |m: &mut ModelArtifactManifestV1| m
        .execution
        .implementation_revision
        .push('x'));
    changed!("protocol", |m: &mut ModelArtifactManifestV1| m
        .execution
        .protocol_revision
        .push('x'));
    changed!("numeric", |m: &mut ModelArtifactManifestV1| m
        .execution
        .numeric_profile
        .push('x'));
    changed!("weights format", |m: &mut ModelArtifactManifestV1| m
        .execution
        .weights_format
        .push('x'));
    changed!("tokenizer family", |m: &mut ModelArtifactManifestV1| m
        .execution
        .tokenizer_family
        .push('x'));
    changed!("preprocessing", |m: &mut ModelArtifactManifestV1| m
        .execution
        .model_preprocessing
        .push('x'));
    changed!("sequence", |m: &mut ModelArtifactManifestV1| m
        .execution
        .sequence_policy
        .push('x'));
    changed!("pooling", |m: &mut ModelArtifactManifestV1| m
        .execution
        .pooling
        .push('x'));
    changed!("normalization", |m: &mut ModelArtifactManifestV1| m
        .execution
        .output_normalization
        .push('x'));
    changed!("query instruction", |m: &mut ModelArtifactManifestV1| m
        .execution
        .query_instruction
        .push('x'));
    changed!("document instruction", |m: &mut ModelArtifactManifestV1| m
        .execution
        .document_instruction
        .push('x'));
    changed!("input contract", |m: &mut ModelArtifactManifestV1| m
        .execution
        .input_contract
        .canonicalization
        .push('x'));
    changed!("golden corpus", |m: &mut ModelArtifactManifestV1| m
        .execution
        .golden_vectors
        .corpus_sha256 =
        "5".repeat(64));
    changed!("golden vectors", |m: &mut ModelArtifactManifestV1| m
        .execution
        .golden_vectors
        .vectors_sha256 =
        "6".repeat(64));
    changed!("golden count", |m: &mut ModelArtifactManifestV1| m
        .execution
        .golden_vectors
        .vector_count +=
        1);
    changed!("golden dimension", |m: &mut ModelArtifactManifestV1| m
        .execution
        .golden_vectors
        .dimension +=
        1);
}

#[test]
fn frozen_manifest_rejects_schema_roles_bytes_and_digest_disagreement() {
    let manifest = sample_artifact_manifest();
    let frozen = manifest.freeze().unwrap();

    let mut unknown_schema = manifest.clone();
    unknown_schema.schema_version += 1;
    assert!(unknown_schema.validate().is_err());

    let mut floating_revision = manifest.clone();
    floating_revision.upstream_revision = "main".to_owned();
    assert!(floating_revision.validate().is_err());

    let mut duplicate_role = manifest.clone();
    duplicate_role.artifacts[1].role = ModelArtifactRoleV1::Weights;
    assert!(duplicate_role.validate().is_err());

    let mut stale_license_assertion = manifest.clone();
    stale_license_assertion.license_spdx = "MIT".to_owned();
    assert!(stale_license_assertion.validate().is_err());

    let mut oversized_instruction = manifest.clone();
    oversized_instruction.execution.query_instruction =
        "x".repeat(MAX_FROZEN_MANIFEST_FIELD_BYTES + 1);
    assert!(oversized_instruction.validate().is_err());

    let mut missing_role = manifest;
    missing_role
        .artifacts
        .retain(|artifact| artifact.role != ModelArtifactRoleV1::Tokenizer);
    assert!(missing_role.validate().is_err());

    let mut bad_bytes = frozen.clone();
    bad_bytes.canonical_bytes.push(0);
    assert!(bad_bytes.validate().is_err());

    let mut bad_digest = frozen.clone();
    bad_digest.fingerprint = "0".repeat(64);
    assert!(bad_digest.validate().is_err());

    let mut injected_digest = frozen;
    injected_digest.fingerprint = "digest\nforged-log-line".to_owned();
    let error = injected_digest.validate().unwrap_err();
    assert!(error.to_string().contains("redacted-invalid-sha256"));
    assert!(!error.to_string().contains("forged-log-line"));

    let mut unknown_field = serde_json::to_value(sample_artifact_manifest()).unwrap();
    unknown_field["execution"]["future_unregistered_field"] = serde_json::json!(true);
    assert!(
        serde_json::from_value::<ModelArtifactManifestV1>(unknown_field).is_err(),
        "versioned manifests must reject unknown fields instead of silently dropping them"
    );

    let mut credentialed_url = sample_artifact_manifest();
    credentialed_url.artifacts[0].upstream_url =
        "https://models.example.invalid/model.safetensors?token=secret".to_owned();
    let error = credentialed_url.freeze().unwrap_err();
    assert!(error.to_string().contains("credential-free HTTPS"));
    assert!(!error.to_string().contains("secret"));

    let mut log_injection = sample_artifact_manifest();
    log_injection.logical_model_id = "fixture\nforged-log-line".to_owned();
    let error = log_injection.freeze().unwrap_err();
    assert!(error.to_string().contains("control characters"));
    assert!(!error.to_string().contains("forged-log-line"));

    let mut digest_injection = sample_artifact_manifest();
    digest_injection.artifacts[0].sha256 = "digest\nforged-log-line".to_owned();
    let error = digest_injection.freeze().unwrap_err();
    assert!(error.to_string().contains("redacted-invalid-sha256"));
    assert!(!error.to_string().contains("forged-log-line"));
}

#[test]
fn streaming_artifact_verification_rejects_size_hash_missing_and_role_collision() {
    let manifest = sample_artifact_manifest();
    let exact = tempfile::tempdir().unwrap();
    write_temp_file(&exact.path().join("model.safetensors"), b"weights");
    write_temp_file(&exact.path().join("tokenizer.json"), b"tokenizer");
    write_temp_file(&exact.path().join("NOTICE"), b"fixture notice");
    let verified = manifest.verify_dir(exact.path()).unwrap();
    verified
        .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap()
        .validate()
        .unwrap();

    let wrong_size = tempfile::tempdir().unwrap();
    write_temp_file(&wrong_size.path().join("model.safetensors"), b"weight");
    write_temp_file(&wrong_size.path().join("tokenizer.json"), b"tokenizer");
    assert!(manifest.verify_dir(wrong_size.path()).is_err());

    let wrong_hash = tempfile::tempdir().unwrap();
    write_temp_file(&wrong_hash.path().join("model.safetensors"), b"WeightS");
    write_temp_file(&wrong_hash.path().join("tokenizer.json"), b"tokenizer");
    assert!(manifest.verify_dir(wrong_hash.path()).is_err());

    let missing = tempfile::tempdir().unwrap();
    write_temp_file(&missing.path().join("model.safetensors"), b"weights");
    assert!(manifest.verify_dir(missing.path()).is_err());

    let all_missing = tempfile::tempdir().unwrap();
    let error = manifest.verify_dir(all_missing.path()).unwrap_err();
    assert!(matches!(&error, SearchError::ModelNotFound { .. }));
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("model.safetensors:missing"));
    assert!(diagnostic.contains("tokenizer.json:missing"));
    assert!(diagnostic.contains("https://models.example.invalid/immutable/model.safetensors"));
    assert!(diagnostic.contains("https://models.example.invalid/immutable/tokenizer.json"));
    assert!(
        !diagnostic.contains(&all_missing.path().display().to_string()),
        "artifact diagnostics must not expose the raw model-directory path"
    );

    let mixed_failure = tempfile::tempdir().unwrap();
    write_temp_file(&mixed_failure.path().join("model.safetensors"), b"WeightS");
    let error = manifest.verify_dir(mixed_failure.path()).unwrap_err();
    assert!(matches!(&error, SearchError::ModelLoadFailed { .. }));
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("model.safetensors:sha256-or-size-mismatch"));
    assert!(diagnostic.contains("tokenizer.json:missing"));
    assert!(diagnostic.contains("<redacted-model-dir>"));
    assert!(
        !diagnostic.contains(&mixed_failure.path().display().to_string()),
        "mixed artifact diagnostics must not expose the raw model-directory path"
    );

    let collision = tempfile::tempdir().unwrap();
    write_temp_file(&collision.path().join("model.safetensors"), b"weights");
    write_temp_file(&collision.path().join("tokenizer.json"), b"tokenizer");
    fs::create_dir_all(collision.path().join("unregistered")).unwrap();
    write_temp_file(
        &collision.path().join("unregistered/model.safetensors"),
        b"collision",
    );
    assert!(manifest.verify_dir(collision.path()).is_err());

    let unexpected_role = tempfile::tempdir().unwrap();
    write_temp_file(
        &unexpected_role.path().join("model.safetensors"),
        b"weights",
    );
    write_temp_file(&unexpected_role.path().join("tokenizer.json"), b"tokenizer");
    write_temp_file(&unexpected_role.path().join("config.json"), b"{}");
    assert!(manifest.verify_dir(unexpected_role.path()).is_err());
}

#[cfg(unix)]
#[test]
fn artifact_verification_rejects_symlinked_extra_role_collision() {
    let manifest = sample_artifact_manifest();
    let collision = tempfile::tempdir().unwrap();
    write_temp_file(&collision.path().join("model.safetensors"), b"weights");
    write_temp_file(&collision.path().join("tokenizer.json"), b"tokenizer");
    fs::create_dir_all(collision.path().join("unregistered")).unwrap();
    let target = collision.path().join("unrelated.bin");
    write_temp_file(&target, b"unrelated");
    std::os::unix::fs::symlink(&target, collision.path().join("unregistered/model.onnx")).unwrap();

    let error = manifest.verify_dir(collision.path()).unwrap_err();
    assert!(error.to_string().contains("semantic role"));
}

#[test]
fn every_registered_local_backend_has_a_frozen_manifest() {
    for manifest in [
        ModelArtifactManifestV1::potion_128m_native().unwrap(),
        ModelArtifactManifestV1::minilm_fastembed().unwrap(),
        ModelArtifactManifestV1::minilm_native_frankentorch().unwrap(),
        ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap(),
        ModelArtifactManifestV1::multilingual_minilm_native_frankentorch().unwrap(),
        ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
        ModelArtifactManifestV1::nomic_fastembed().unwrap(),
    ] {
        let frozen = manifest.freeze().unwrap();
        frozen.validate().unwrap();
        assert_eq!(
            frozen.manifest.dimension,
            frozen.manifest.execution.golden_vectors.dimension
        );
        assert_eq!(
            frozen.manifest.execution.golden_vectors.corpus_sha256,
            GoldenVectorCertificateV1::corpus_fingerprint(&MODEL_CONFORMANCE_TEXTS_V1).unwrap()
        );
        let identity = frozen
            .manifest
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_eq!(identity.storage.endianness, "native-f32-values");
    }
}

#[test]
fn persisted_storage_identity_remains_little_endian() {
    let identity = ModelArtifactManifestV1::potion_128m_native()
        .unwrap()
        .declared_identity_bundle(QuantizationFormat::F16, "fsvi-v2")
        .unwrap();
    assert_eq!(identity.storage.endianness, "little-endian");
}

// Rewrite the adapter package version of the Potion and FastEmbed
// producers; native frankentorch producers use independently pinned
// revisions and are returned unchanged.
fn with_adapter_package(
    mut manifest: ModelArtifactManifestV1,
    from: &str,
    to: &str,
) -> ModelArtifactManifestV1 {
    if matches!(
        manifest.execution.backend.as_str(),
        "model2vec-native" | "fastembed-onnx"
    ) {
        let adapter = manifest
            .execution
            .implementation_revision
            .strip_prefix(&format!("frankensearch-embed-{from}+"))
            .unwrap_or_else(|| panic!("adapter must identify the actual {from} package"))
            .to_owned();
        manifest.execution.implementation_revision = format!("frankensearch-embed-{to}+{adapter}");
    }
    manifest
}

// Package 0.4.0 changes only these adapters' implementation provenance.
// Reconstruct the 0.3.1 revision that wrote every index of the 0.3 line.
fn before_embed_0_4(manifest: ModelArtifactManifestV1) -> ModelArtifactManifestV1 {
    with_adapter_package(manifest, "0.4.0", "0.3.1")
}

// Package 0.3.1 changed only these adapters' implementation provenance.
// Reconstruct the historical 0.2.7 revision before checking its frozen
// fixtures.
fn before_adapter_release(manifest: ModelArtifactManifestV1) -> ModelArtifactManifestV1 {
    with_adapter_package(manifest, "0.3.1", "0.2.7")
}

#[test]
fn adapter_release_changes_producer_identity_without_changing_space_or_vectors() {
    use frankensearch_core::generation::ProducerCompatibilityErrorV1;

    assert_eq!(env!("CARGO_PKG_VERSION"), "0.4.0");
    for current in [
        ModelArtifactManifestV1::potion_128m_native().unwrap(),
        ModelArtifactManifestV1::minilm_fastembed().unwrap(),
        ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
        ModelArtifactManifestV1::nomic_fastembed().unwrap(),
    ] {
        let historical = before_adapter_release(before_embed_0_4(current.clone()));
        assert_eq!(
            current.space_contract_fingerprint().unwrap(),
            historical.space_contract_fingerprint().unwrap()
        );
        assert_eq!(
            current.execution.golden_vectors,
            historical.execution.golden_vectors
        );
        let current_identity = current
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        let historical_identity = historical
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_ne!(current_identity.producer, historical_identity.producer);
        assert_ne!(
            current_identity.fingerprint(),
            historical_identity.fingerprint()
        );
        current_identity
            .verify_exact_producer_with(&current_identity)
            .expect("the current producer must admit itself");
        assert_eq!(
            current_identity.verify_exact_producer_with(&historical_identity),
            Err(ProducerCompatibilityErrorV1::CertificateRequired),
            "matching spaces and vectors must not admit a historical producer as current"
        );
        assert_ne!(
            current.freeze().unwrap().fingerprint,
            historical.freeze().unwrap().fingerprint
        );
        // Preserve admission refusal for the 0.3 line's adapters too,
        // without changing their model bytes, space, or certificate.
        for previous_package in ["0.3.1", "0.3.0"] {
            let previous = with_adapter_package(current.clone(), "0.4.0", previous_package);
            assert_ne!(previous, current);
            assert_eq!(
                current.space_contract_fingerprint().unwrap(),
                previous.space_contract_fingerprint().unwrap()
            );
            assert_eq!(
                current.execution.golden_vectors,
                previous.execution.golden_vectors
            );
            let previous_identity = previous
                .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
                .unwrap();
            assert_eq!(
                current_identity.verify_exact_producer_with(&previous_identity),
                Err(ProducerCompatibilityErrorV1::CertificateRequired),
                "a {previous_package} adapter must not be admitted as current"
            );
            assert_ne!(
                current.freeze().unwrap().fingerprint,
                previous.freeze().unwrap().fingerprint
            );
        }
    }
}

#[test]
fn potion_safetensors_refresh_requires_distinct_producer_admission() {
    use frankensearch_core::generation::ProducerCompatibilityErrorV1;

    let current = ModelArtifactManifestV1::potion_128m_native().unwrap();
    assert_eq!(
        current.execution.protocol_revision,
        "tokenizers-0.23.2+safetensors-0.8.0-static-table-v1"
    );
    let mut previous = current.clone();
    previous.execution.protocol_revision =
        "tokenizers-0.23.2+safetensors-0.7.0-static-table-v1".to_owned();
    assert_eq!(
        current.execution.implementation_revision, previous.execution.implementation_revision,
        "this negative must isolate protocol drift from the adapter bump"
    );
    assert_eq!(
        current.execution.golden_vectors,
        previous.execution.golden_vectors
    );
    assert_eq!(
        current.space_contract_fingerprint().unwrap(),
        previous.space_contract_fingerprint().unwrap()
    );
    let current_identity = current
        .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();
    let previous_identity = previous
        .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .unwrap();
    current_identity
        .verify_exact_producer_with(&current_identity)
        .expect("current Potion producer admits itself");
    assert_eq!(
        current_identity.verify_exact_producer_with(&previous_identity),
        Err(ProducerCompatibilityErrorV1::CertificateRequired)
    );
    assert_ne!(
        current.freeze().unwrap().fingerprint,
        previous.freeze().unwrap().fingerprint
    );
}

/// Reconstruct an English `MiniLM` producer before bd-7651e moved it from
/// 512 to 256 tokens. Every other producer is returned unchanged.
pub(super) fn before_minilm_256(mut manifest: ModelArtifactManifestV1) -> ModelArtifactManifestV1 {
    let historical = match (
        manifest.execution.backend.as_str(),
        manifest.logical_model_id.as_str(),
    ) {
        ("fastembed-onnx", "all-minilm-l6-v2") => {
            Some((FASTEMBED_SEQUENCE_POLICY_256, FASTEMBED_SEQUENCE_POLICY_512))
        }
        ("frankentorch-native-minilm" | "frankentorch-native-minilm-f32", _) => Some((
            "max-length=256;longest-first;no-padding",
            "max-length=512;longest-first;no-padding",
        )),
        _ => None,
    };
    if let Some((current, previous)) = historical {
        assert_eq!(manifest.execution.sequence_policy, current);
        previous.clone_into(&mut manifest.execution.sequence_policy);
    }
    manifest
}

/// bd-7651e: only the sequence policy moves, so the embedding space and
/// every producer fingerprint change, but the conformance texts are far
/// shorter than either budget and their certified vectors stay identical.
#[test]
fn minilm_256_changes_only_the_space_sequence_policy() {
    for current in [
        ModelArtifactManifestV1::minilm_fastembed().unwrap(),
        ModelArtifactManifestV1::minilm_native_frankentorch().unwrap(),
        ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap(),
    ] {
        let previous = before_minilm_256(current.clone());
        assert_ne!(
            current.space_contract_fingerprint().unwrap(),
            previous.space_contract_fingerprint().unwrap()
        );
        assert_eq!(
            current.execution.golden_vectors,
            previous.execution.golden_vectors
        );
        let current_identity = current
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        let previous_identity = previous
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_ne!(current_identity.space, previous_identity.space);
        assert!(
            current_identity
                .verify_exact_producer_with(&previous_identity)
                .is_err(),
            "a 512-token index must never admit the 256-token producer"
        );
        let mut restored = current.clone();
        previous
            .execution
            .sequence_policy
            .clone_into(&mut restored.execution.sequence_policy);
        assert_eq!(restored, previous, "no other manifest field may drift");
    }
    for unchanged in [
        ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
        ModelArtifactManifestV1::nomic_fastembed().unwrap(),
        ModelArtifactManifestV1::multilingual_minilm_native_frankentorch().unwrap(),
    ] {
        assert_eq!(before_minilm_256(unchanged.clone()), unchanged);
        assert!(
            unchanged
                .execution
                .sequence_policy
                .starts_with("max-length=512;")
        );
    }
}

// Keep the 7.0.1 producer independently of older dependency history.
fn before_fastembed_7_1(mut manifest: ModelArtifactManifestV1) -> ModelArtifactManifestV1 {
    if manifest.execution.backend == "fastembed-onnx" {
        assert!(
            manifest
                .execution
                .implementation_revision
                .contains("+fastembed-7.1.0")
        );
        manifest.execution.implementation_revision = manifest
            .execution
            .implementation_revision
            .replacen("+fastembed-7.1.0", "+fastembed-7.0.1", 1);
        assert_eq!(
            manifest.execution.protocol_revision,
            "fastembed-7.1.0+ort-2.0.0-rc.13-user-defined-onnx-v1"
        );
        manifest.execution.protocol_revision =
            "fastembed-7.0.1+ort-2.0.0-rc.13-user-defined-onnx-v1".to_owned();
    }
    manifest
}

#[test]
fn fastembed_7_1_requires_new_producer_admission_with_unchanged_certificates() {
    use frankensearch_core::generation::ProducerCompatibilityErrorV1;

    for current in [
        ModelArtifactManifestV1::minilm_fastembed().unwrap(),
        ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
        ModelArtifactManifestV1::nomic_fastembed().unwrap(),
    ] {
        let previous = before_fastembed_7_1(current.clone());
        assert_eq!(current.artifacts, previous.artifacts);
        assert_eq!(
            current.execution.golden_vectors,
            previous.execution.golden_vectors
        );
        assert_eq!(
            current.space_contract_fingerprint().unwrap(),
            previous.space_contract_fingerprint().unwrap()
        );
        let current_identity = current
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        let previous_identity = previous
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_ne!(current_identity.producer, previous_identity.producer);
        assert_eq!(
            current_identity.verify_exact_producer_with(&previous_identity),
            Err(ProducerCompatibilityErrorV1::CertificateRequired)
        );
        let mut restored = previous;
        restored.execution.implementation_revision = restored
            .execution
            .implementation_revision
            .replacen("+fastembed-7.0.1", "+fastembed-7.1.0", 1);
        restored.execution.protocol_revision =
            "fastembed-7.1.0+ort-2.0.0-rc.13-user-defined-onnx-v1".to_owned();
        assert_eq!(
            restored, current,
            "only the dependency producer identity changes"
        );
    }
}

// Preserve the immediately preceding dependency identity independently of
// adapter package version and every platform's numerical certificate.
fn before_fastembed_7(mut manifest: ModelArtifactManifestV1) -> ModelArtifactManifestV1 {
    if manifest.execution.backend == "fastembed-onnx" {
        assert!(
            manifest
                .execution
                .implementation_revision
                .contains("+fastembed-7.0.1")
        );
        manifest.execution.implementation_revision = manifest
            .execution
            .implementation_revision
            .replacen("+fastembed-7.0.1", "+fastembed-6.0.3", 1);
        assert_eq!(
            manifest.execution.protocol_revision,
            "fastembed-7.0.1+ort-2.0.0-rc.13-user-defined-onnx-v1"
        );
        manifest.execution.protocol_revision =
            "fastembed-6.0.3+ort-2.0.0-rc.13-user-defined-onnx-v1".to_owned();
    }
    manifest
}

#[test]
fn fastembed_7_changes_producer_identity_without_changing_space_or_vectors() {
    use frankensearch_core::generation::ProducerCompatibilityErrorV1;

    for current in [
        ModelArtifactManifestV1::minilm_fastembed().unwrap(),
        ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
        ModelArtifactManifestV1::nomic_fastembed().unwrap(),
    ]
    .map(before_fastembed_7_1)
    {
        // Only the FastEmbed segment moves; the adapter package stays.
        let previous = before_fastembed_7(current.clone());
        assert!(
            previous
                .execution
                .implementation_revision
                .starts_with(&format!(
                    "frankensearch-embed-{}+fastembed-6.0.3",
                    env!("CARGO_PKG_VERSION")
                ))
        );
        assert_eq!(current.artifacts, previous.artifacts);
        assert_eq!(
            current.execution.input_contract,
            previous.execution.input_contract
        );
        assert_eq!(
            current.execution.golden_vectors,
            previous.execution.golden_vectors
        );
        assert_eq!(
            current.space_contract_fingerprint().unwrap(),
            previous.space_contract_fingerprint().unwrap()
        );
        let current_identity = current
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        let previous_identity = previous
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_ne!(current_identity.producer, previous_identity.producer);
        assert_ne!(
            current_identity.fingerprint(),
            previous_identity.fingerprint()
        );
        current_identity
            .verify_exact_producer_with(&current_identity)
            .unwrap();
        assert_eq!(
            current_identity.verify_exact_producer_with(&previous_identity),
            Err(ProducerCompatibilityErrorV1::CertificateRequired)
        );
        assert_ne!(
            current.freeze().unwrap().fingerprint,
            previous.freeze().unwrap().fingerprint
        );
        let mut restored = previous;
        restored.execution.implementation_revision = restored
            .execution
            .implementation_revision
            .replacen("+fastembed-6.0.3", "+fastembed-7.0.1", 1);
        restored.execution.protocol_revision =
            "fastembed-7.0.1+ort-2.0.0-rc.13-user-defined-onnx-v1".to_owned();
        assert_eq!(restored, current, "no other manifest field may drift");
    }
}

// Reconstruct the pre-refresh producer without changing any artifact or
// preprocessing. Windows previously declared the Linux numeric contract;
// retain those historical hashes rather than rewriting them to the newly
// qualified Windows output. Current owning-loader tests check the new bits.
pub(super) fn before_dependency_refresh(
    mut manifest: ModelArtifactManifestV1,
) -> ModelArtifactManifestV1 {
    let current = manifest.freeze().unwrap().fingerprint;
    manifest = before_embed_0_4(manifest);
    manifest = before_minilm_256(manifest);
    manifest = before_fastembed_7_1(manifest);
    manifest = before_fastembed_7(manifest);
    manifest = before_adapter_release(manifest);
    let (current_protocol, previous_protocol) = match manifest.execution.backend.as_str() {
        "model2vec-native" => (
            "tokenizers-0.23.2+safetensors-0.8.0-static-table-v1",
            "tokenizers-0.23.1+safetensors-0.7.0-static-table-v1",
        ),
        "fastembed-onnx" => {
            if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
                let (windows, linux) = match manifest.logical_model_id.as_str() {
                    "all-minilm-l6-v2" => (
                        "26db12f2e6f084f991d61a29da7f6c66c1e70981b74f07cc73c7f235ee6c0142",
                        "67cec04aef931fb5b5db8be074e92370c4f62e6f89ec45bec5ecd52a2444d6c3",
                    ),
                    "snowflake-arctic-embed-s" => (
                        "01e4e1fb6c2e5e4234080ce5644d13cb08ddece97e6e31fa39d49083cea37c37",
                        "8ab295190de5eb629ef7920e3aec6d989c1b7f695b4f75baebfb716fb81b7f6c",
                    ),
                    "nomic-embed-text-v1.5" => (
                        "8b2d7421218d8ab9e6d0f9c3ea66b371be56324a5560c13e2b416c5064e51f05",
                        "dbb7e33fdb5ccb4864faf9ff425b35a83a2d9dcd4f8d736033d7f819e0c1e851",
                    ),
                    model => panic!("unclassified Windows ONNX producer: {model}"),
                };
                assert_eq!(manifest.execution.golden_vectors.vectors_sha256, windows);
                assert_eq!(
                    manifest.execution.numeric_profile,
                    "ort-1.28.0-da9b5e3-windows-x86_64-cpu-f32-host-default-intra-threads-v1"
                );
                linux.clone_into(&mut manifest.execution.golden_vectors.vectors_sha256);
                "ort-2.0.0-rc.13-cpu-f32-host-default-intra-threads-v1"
                    .clone_into(&mut manifest.execution.numeric_profile);
            }
            assert!(
                manifest
                    .execution
                    .implementation_revision
                    .contains("+fastembed-6.0.3")
            );
            manifest.execution.implementation_revision = manifest
                .execution
                .implementation_revision
                .replacen("+fastembed-6.0.3", "+fastembed-6.0.0", 1);
            (
                "fastembed-6.0.3+ort-2.0.0-rc.13-user-defined-onnx-v1",
                "fastembed-6.0.0+ort-2.0.0-rc.13-user-defined-onnx-v1",
            )
        }
        "frankentorch-native-minilm" | "frankentorch-native-minilm-f32" => (
            "tokenizers-0.23.2+frankentorch-bert-encoder-v1",
            "tokenizers-0.23.1+frankentorch-bert-encoder-v1",
        ),
        "frankentorch-native-multilingual-minilm" => (
            "tokenizers-0.23.2-xlmr-unigram+frankentorch-bert-dynamic-layers-v2",
            "tokenizers-0.23.1-xlmr-unigram+frankentorch-bert-dynamic-layers-v2",
        ),
        backend => panic!("unclassified registered producer: {backend}"),
    };
    assert_eq!(manifest.execution.protocol_revision, current_protocol);
    previous_protocol.clone_into(&mut manifest.execution.protocol_revision);
    assert_ne!(manifest.freeze().unwrap().fingerprint, current);
    manifest
}

/// GH #50: each registered potion model is its own vector space. A naive
/// implementation that reused the 128M contract for the smaller models
/// (same logical id or space) would let an index mix them; it must not.
#[test]
fn registered_potion_models_have_distinct_pinned_identities() {
    let models = [
        (
            ModelManifest::potion_128m(),
            ModelArtifactManifestV1::potion_128m_native().unwrap(),
            256,
        ),
        (
            ModelManifest::potion_base_8m(),
            ModelArtifactManifestV1::potion_base_8m_native().unwrap(),
            256,
        ),
        (
            ModelManifest::potion_base_32m(),
            ModelArtifactManifestV1::potion_base_32m_native().unwrap(),
            512,
        ),
    ];
    let mut logical_ids = std::collections::BTreeSet::new();
    let mut spaces = std::collections::BTreeSet::new();
    let mut producers = std::collections::BTreeSet::new();
    for (download, native, dimension) in &models {
        assert!(
            download.is_production_ready(),
            "{} must be pinned",
            download.id
        );
        assert_eq!(download.revision.len(), 40, "{} revision", download.id);
        assert_eq!(
            download.download_size_bytes,
            download.files.iter().map(|file| file.size).sum::<u64>(),
            "{} download size is the sum of its files",
            download.id
        );
        let identity = native
            .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_eq!(identity.space.dimension, *dimension, "{}", download.id);
        assert_eq!(native.execution.golden_vectors.dimension, *dimension);
        logical_ids.insert(identity.space.logical_model_id.clone());
        spaces.insert(identity.space.fingerprint());
        producers.insert(identity.fingerprint());
    }
    assert_eq!(
        logical_ids.len(),
        3,
        "logical model ids must differ: {logical_ids:?}"
    );
    assert_eq!(spaces.len(), 3, "vector spaces must differ");
    assert_eq!(producers.len(), 3, "producer identities must differ");

    // Opt-in only: never part of default bulk acquisition.
    let default_ids = ModelManifest::builtin_catalog()
        .models
        .into_iter()
        .map(|manifest| manifest.id)
        .collect::<Vec<_>>();
    let opt_in_ids = ModelManifest::opt_in_catalog()
        .models
        .into_iter()
        .map(|manifest| manifest.id)
        .collect::<Vec<_>>();
    for id in ["potion-base-8m", "potion-base-32m"] {
        assert!(
            !default_ids.iter().any(|known| known == id),
            "{id} is opt-in"
        );
        assert!(
            opt_in_ids.iter().any(|known| known == id),
            "{id} is registered"
        );
    }
}

#[test]
fn registered_manifest_fingerprints_are_exact_fixtures() {
    // GOLDEN-CHANGE bd-2ba5: the explicit F32 producer and dependency
    // metadata correction are followed by the measured ORT 1.24.2 ->
    // 1.28.0 MiniLM/Snowflake certificate change. Nomic's output is unchanged.
    // Historical certificates must fail the actual loader; exact current
    // conformance is checked both at load and in the real-model fixtures.
    // GOLDEN-CHANGE release 0.2.7 (GH #46): four adapter manifests include
    // the crate version in implementation_revision. Only that provenance
    // field moves from 0.2.6 to 0.2.7; reconstruction retains both 0.2.5
    // and 0.2.6 hashes and proves that no other manifest field or output
    // certificate drifted. The space identity is built from
    // space_contract_fingerprint, which carries no crate version. The
    // producer and bundle fingerprints still change: fsfs revision checks
    // and strict library search activation require rebuilding an index
    // produced by 0.2.6, even when its space and output certificate agree.
    // GOLDEN-CHANGE macOS ARM64 qualification: only the three ONNX
    // numeric profiles and output certificates differ on that platform.
    // Retain exact Linux fixtures and reconstruct them below before the
    // previous-version check; all artifact and input fields stay frozen.
    // bd-dsbym: retain every prior fixture verbatim after correcting the
    // Tokenizers/FastEmbed dependency identity. No vector golden changes.
    // GOLDEN-CHANGE releases 0.3.0/0.3.1/0.4.0: reconstruct the prior 0.2.7
    // adapter (through 0.3.1) before checking these exact historical
    // hashes. Production manifests keep their actual package revision; no
    // historical hash is replaced.
    // GOLDEN-CHANGE Safetensors 0.8.0: Potion's producer protocol now names
    // its actual dependency. The reconstruction restores 0.7.0 alongside
    // the old Tokenizers protocol; artifacts and vector certificates stay exact.
    let observed = [
        ModelArtifactManifestV1::potion_128m_native().unwrap(),
        ModelArtifactManifestV1::minilm_fastembed().unwrap(),
        ModelArtifactManifestV1::minilm_native_frankentorch().unwrap(),
        ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap(),
        ModelArtifactManifestV1::multilingual_minilm_native_frankentorch().unwrap(),
        ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
        ModelArtifactManifestV1::nomic_fastembed().unwrap(),
    ]
    .map(|manifest| {
        let frozen = before_dependency_refresh(manifest).freeze().unwrap();
        (frozen.manifest.execution.backend, frozen.fingerprint)
    });
    let expected = [
        (
            "model2vec-native".to_owned(),
            "2d2e46f3db95cd70a990d78b45ad3485f3fefda9e727e498047cec151605da73".to_owned(),
        ),
        (
            "fastembed-onnx".to_owned(),
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "16feb0917d546ef8b2cd438a28f25e1d9481c76ffcc18a270e3e4c566d35dd0e"
            } else {
                "77c7b38ba81262047b7e9f14f89dfc37f20a6d1924ec1baee1c6f372dcd84ed1"
            }
            .to_owned(),
        ),
        (
            "frankentorch-native-minilm".to_owned(),
            "726d6dde1d25946d1c7287a26f92518c158ee6d009b8a76f52748275f063e72c".to_owned(),
        ),
        (
            "frankentorch-native-minilm-f32".to_owned(),
            "6fc7b062661b115c8f82f323f9db5802cab96e93a80d3adfe6c5a5c008c1af87".to_owned(),
        ),
        (
            "frankentorch-native-multilingual-minilm".to_owned(),
            "59160d9e43d396d05b4139c99f9feb7922da14868587fca7e33d379821a41405".to_owned(),
        ),
        (
            "fastembed-onnx".to_owned(),
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "95dc7bc56fe6bed21d82cfc5c4cfc4a2544b3fc571dd5b2aa626cb514e33cdd7"
            } else {
                "b5cd1821b46db09d6e902c691167543a698ef3a8674d568750269b4048ae4186"
            }
            .to_owned(),
        ),
        (
            "fastembed-onnx".to_owned(),
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "e423b6557eb53f57ef08fecd96703c8a8e108b8af0aace18c623d8ceb0329a8e"
            } else {
                "9adb5a60c0e20624c9ce1db786a8a548c68ba6605b97ba3bab787e2baaeddaa9"
            }
            .to_owned(),
        ),
    ];
    assert_eq!(observed, expected);

    for (manifest, previous_fingerprint, older_fingerprint, linux_certificate) in [
        (
            ModelArtifactManifestV1::potion_128m_native().unwrap(),
            "4e997c1e42f4078b723a0043a3ba941ff63bd52e65e8157c5230a2d7a0bc063b",
            "3783488bae83c3fd07e9913df11430f6be4e7c70da94f7c5d75ee78dc0ebfadf",
            None,
        ),
        (
            ModelArtifactManifestV1::minilm_fastembed().unwrap(),
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "327feba7d864c417b214b32eb49b85b04d6b7fe73c93350e65cac0a7d5ca9c6b"
            } else {
                "df15494ac05894c47aef7924b06dee0b937847964ef4c322343e5dad6efcbc97"
            },
            "7c2debb9bf81b9d6f37a4cab092af12a74fdc563d9ffab9cfa20af7ecb8302ca",
            Some("67cec04aef931fb5b5db8be074e92370c4f62e6f89ec45bec5ecd52a2444d6c3"),
        ),
        (
            ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "0857407535835be7313cdc767be24e5c11bf37a0a0b62a46672ba7b0078b5654"
            } else {
                "3c1a5cc06410ad93869ea522779e496cffb4a4b7742a40be86365444d18c9899"
            },
            "07436d59a036ba2917bd6fa3a6e859101fcd9d91a706f39dd723df0daa15b008",
            Some("8ab295190de5eb629ef7920e3aec6d989c1b7f695b4f75baebfb716fb81b7f6c"),
        ),
        (
            ModelArtifactManifestV1::nomic_fastembed().unwrap(),
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "7ad71e245b9cc2c7f38be14d205104eae4c8d6de75e2635d19dcba73bf5a4085"
            } else {
                "9559ef790c74d3335ded4dc2aaf8ee2c4818de250ebc44bd3ea020b959513077"
            },
            "f20cac45eb8310e793362e25fa3bcbe943dc2af78536840d93da58a1d31fa40a",
            Some("dbb7e33fdb5ccb4864faf9ff425b35a83a2d9dcd4f8d736033d7f819e0c1e851"),
        ),
    ] {
        let mut manifest = before_dependency_refresh(manifest);
        let adapter = manifest
            .execution
            .implementation_revision
            .strip_prefix("frankensearch-embed-0.2.7+")
            .expect("reconstructed fixture must name the historical 0.2.7 adapter")
            .to_owned();
        manifest.execution.implementation_revision = format!("frankensearch-embed-0.2.6+{adapter}");
        assert_eq!(
            manifest.freeze().unwrap().fingerprint,
            previous_fingerprint,
            "the release bump must preserve the exact previous platform manifest"
        );
        if cfg!(all(target_os = "macos", target_arch = "aarch64"))
            && let Some(certificate) = linux_certificate
        {
            "ort-2.0.0-rc.13-cpu-f32-host-default-intra-threads-v1"
                .clone_into(&mut manifest.execution.numeric_profile);
            certificate.clone_into(&mut manifest.execution.golden_vectors.vectors_sha256);
        }
        manifest.execution.implementation_revision = format!("frankensearch-embed-0.2.5+{adapter}");
        assert_eq!(
            manifest.freeze().unwrap().fingerprint,
            older_fingerprint,
            "the release bump must change only adapter version provenance"
        );
    }
}
