use super::embedding_identity::MAX_IDENTITY_FIELD_BYTES;
use super::tests_support::{
    sample_foreign_producer_pair, sample_semantic_identity, valid_manifest,
};
use super::*;

#[test]
fn artifact_generation_identity_is_full_width_and_round_trips() {
    let identity =
        ArtifactGenerationIdentityV1::new(u64::MAX, [0xa5; 16]).expect("valid generation");
    identity.validate().expect("generation validates");

    let encoded = serde_json::to_vec(&identity).expect("serialize generation");
    let decoded: ArtifactGenerationIdentityV1 =
        serde_json::from_slice(&encoded).expect("deserialize generation");

    assert_eq!(decoded, identity);
    assert_eq!(decoded.sequence, u64::MAX);
    assert_eq!(decoded.fingerprint().len(), 64);
    assert_eq!(decoded.fingerprint(), identity.fingerprint());
}

#[test]
fn artifact_generation_identity_rejects_reserved_or_unknown_values() {
    assert!(ArtifactGenerationIdentityV1::new(0, [0; 16]).is_err());

    let mut unknown_schema =
        ArtifactGenerationIdentityV1::new(0, [1; 16]).expect("valid generation");
    unknown_schema.schema_version = ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1 + 1;
    assert!(unknown_schema.validate().is_err());

    let injected = serde_json::json!({
        "schema_version": ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1,
        "sequence": 7,
        "nonce": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "unexpected": "field"
    });
    assert!(
        serde_json::from_value::<ArtifactGenerationIdentityV1>(injected).is_err(),
        "unknown generation fields must fail closed"
    );
}

#[test]
fn every_artifact_generation_field_changes_the_fingerprint() {
    let base = ArtifactGenerationIdentityV1::new(7, [1; 16]).expect("valid base generation");

    let mut changed_sequence = base;
    changed_sequence.sequence += 1;
    assert_ne!(base.fingerprint(), changed_sequence.fingerprint());

    let mut changed_nonce = base;
    changed_nonce.nonce[15] ^= 1;
    assert_ne!(base.fingerprint(), changed_nonce.fingerprint());

    let mut changed_schema = base;
    changed_schema.schema_version += 1;
    assert_ne!(base.fingerprint(), changed_schema.fingerprint());
    assert!(changed_schema.validate().is_err());
}

#[test]
fn golden_certificate_binds_order_shape_and_exact_f32_bits() {
    let texts = ["", "Unicode café", "signed zero"];
    let vectors = vec![
        vec![0.0, -0.0],
        vec![1.0, f32::from_bits(0x7fc0_0042)],
        vec![-3.5, f32::INFINITY],
    ];
    let certificate = GoldenVectorCertificateV1::from_exact_f32(&texts, &vectors).unwrap();
    certificate.verify_exact_f32(&texts, &vectors).unwrap();

    let mut reordered_texts = texts;
    reordered_texts.swap(0, 1);
    assert!(
        certificate
            .verify_exact_f32(&reordered_texts, &vectors)
            .is_err()
    );

    let mut changed_bits = vectors.clone();
    changed_bits[0][0] = -0.0;
    assert!(certificate.verify_exact_f32(&texts, &changed_bits).is_err());

    let inconsistent = vec![vec![1.0, 2.0], vec![3.0]];
    assert!(GoldenVectorCertificateV1::from_exact_f32(&texts[..2], &inconsistent).is_err());
}

#[test]
fn exact_producer_witness_allows_storage_drift_but_rejects_foreign_producers() {
    let (reference, foreign, _) = sample_foreign_producer_pair();
    assert_eq!(
        reference.verify_exact_producer_with(&foreign),
        Err(ProducerCompatibilityErrorV1::CertificateRequired)
    );

    let mut alternate_storage = reference.clone();
    alternate_storage.storage.format = "fsvi-v2".to_owned();
    alternate_storage.storage.quantization = QuantizationFormat::F16;
    alternate_storage.storage.endianness = "little-endian".to_owned();
    alternate_storage
        .validate()
        .expect("valid alternate storage");
    assert_ne!(reference.fingerprint(), alternate_storage.fingerprint());

    let witness = reference
        .verify_exact_producer_with(&alternate_storage)
        .expect("exact producer remains compatible across storage encodings");
    assert_eq!(witness.kind(), ProducerCompatibilityKindV1::Exact);
    assert_eq!(
        witness.reference_producer_fingerprint(),
        witness.candidate_producer_fingerprint()
    );
    assert!(witness.certificate_fingerprint().is_none());
}

#[test]
fn certified_foreign_producer_requires_pinned_directional_evidence() {
    let (reference, candidate, fixture) = sample_foreign_producer_pair();
    let policy_fingerprint = "9".repeat(64);
    let certificate =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &candidate,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        )
        .expect("certificate receipt");
    let certificate_fingerprint = certificate.fingerprint();
    let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
        &policy_fingerprint,
        &certificate_fingerprint,
        &fixture,
        150,
        7,
        7,
    )
    .expect("trusted context");

    let witness = reference
        .verify_certified_foreign_producer_with(&candidate, &certificate, trusted)
        .expect("trusted foreign producer");
    assert_eq!(witness.kind(), ProducerCompatibilityKindV1::Certified);
    assert_eq!(
        witness.certificate_fingerprint(),
        Some(certificate_fingerprint.as_str())
    );
    assert_eq!(
        witness.reference_producer_fingerprint(),
        reference.producer.fingerprint()
    );
    assert_eq!(
        witness.candidate_producer_fingerprint(),
        candidate.producer.fingerprint()
    );

    let serialized = serde_json::to_string(&witness).expect("serialize opaque witness");
    assert!(!serialized.contains("alternate-native-kernel"));
    assert!(!serialized.contains("query: alpha"));
    assert!(!serialized.contains("document: beta"));
    assert!(!serialized.contains("canonical_bytes"));
}

#[test]
fn every_foreign_certificate_field_is_bound_by_the_pinned_fingerprint() {
    let (reference, candidate, fixture) = sample_foreign_producer_pair();
    let policy_fingerprint = "9".repeat(64);
    let certificate =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &candidate,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        )
        .unwrap();
    let pinned_fingerprint = certificate.fingerprint();

    let mut unknown_schema = certificate.clone();
    unknown_schema.schema_version += 1;
    let mut changed_reference = certificate.clone();
    changed_reference.reference_producer_fingerprint = "a".repeat(64);
    let mut changed_candidate = certificate.clone();
    changed_candidate.candidate_producer_fingerprint = "b".repeat(64);
    let mut changed_space = certificate.clone();
    changed_space.space_fingerprint = "c".repeat(64);
    let mut changed_fixture = certificate.clone();
    changed_fixture.golden_fixture_fingerprint = "d".repeat(64);
    let mut changed_policy = certificate.clone();
    changed_policy.policy_fingerprint = "e".repeat(64);
    let mut changed_revision = certificate.clone();
    changed_revision.certificate_revision += 1;
    let mut changed_start = certificate.clone();
    changed_start.not_before_unix_seconds += 1;
    let mut changed_expiry = certificate.clone();
    changed_expiry.expires_at_unix_seconds -= 1;

    let malformed_context = TrustedProducerConformanceContextV1::from_independent_policy(
        &policy_fingerprint,
        &pinned_fingerprint,
        &fixture,
        150,
        1,
        u64::MAX,
    )
    .unwrap();
    assert_eq!(
        reference.verify_certified_foreign_producer_with(
            &candidate,
            &unknown_schema,
            malformed_context,
        ),
        Err(ProducerCompatibilityErrorV1::CertificateMalformed)
    );

    for (label, mutated) in [
        ("reference producer", changed_reference),
        ("candidate producer", changed_candidate),
        ("space", changed_space),
        ("fixture", changed_fixture),
        ("policy", changed_policy),
        ("revision", changed_revision),
        ("not-before", changed_start),
        ("expiry", changed_expiry),
    ] {
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &pinned_fingerprint,
            &fixture,
            150,
            1,
            u64::MAX,
        )
        .unwrap();
        assert_eq!(
            reference.verify_certified_foreign_producer_with(&candidate, &mutated, trusted,),
            Err(ProducerCompatibilityErrorV1::CertificateFingerprintMismatch),
            "mutated {label} must invalidate the pinned certificate"
        );
    }
}

#[test]
fn foreign_certificate_rejects_policy_pair_fixture_and_bundle_substitution() {
    let (reference, candidate, fixture) = sample_foreign_producer_pair();
    let policy_fingerprint = "9".repeat(64);
    let certificate =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &candidate,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        )
        .unwrap();

    let mut wrong_policy = certificate.clone();
    wrong_policy.policy_fingerprint = "8".repeat(64);
    let wrong_policy_fingerprint = wrong_policy.fingerprint();
    let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
        &policy_fingerprint,
        &wrong_policy_fingerprint,
        &fixture,
        150,
        7,
        7,
    )
    .unwrap();
    assert_eq!(
        reference.verify_certified_foreign_producer_with(&candidate, &wrong_policy, trusted,),
        Err(ProducerCompatibilityErrorV1::PolicyMismatch)
    );

    let swapped =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &candidate,
            &reference,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        )
        .unwrap();
    let swapped_fingerprint = swapped.fingerprint();
    let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
        &policy_fingerprint,
        &swapped_fingerprint,
        &fixture,
        150,
        7,
        7,
    )
    .unwrap();
    assert_eq!(
        reference.verify_certified_foreign_producer_with(&candidate, &swapped, trusted),
        Err(ProducerCompatibilityErrorV1::ProducerBindingMismatch)
    );

    for (label, reference_slot, substituted_fingerprint) in [
        (
            "reference storage fingerprint",
            true,
            reference.storage.fingerprint(),
        ),
        (
            "candidate storage fingerprint",
            false,
            candidate.storage.fingerprint(),
        ),
        (
            "reference full-bundle fingerprint",
            true,
            reference.fingerprint(),
        ),
        (
            "candidate full-bundle fingerprint",
            false,
            candidate.fingerprint(),
        ),
    ] {
        let mut substitution = certificate.clone();
        if reference_slot {
            substitution.reference_producer_fingerprint = substituted_fingerprint;
        } else {
            substitution.candidate_producer_fingerprint = substituted_fingerprint;
        }
        let substitution_fingerprint = substitution.fingerprint();
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &substitution_fingerprint,
            &fixture,
            150,
            7,
            7,
        )
        .unwrap();
        assert_eq!(
            reference.verify_certified_foreign_producer_with(&candidate, &substitution, trusted,),
            Err(ProducerCompatibilityErrorV1::ProducerBindingMismatch),
            "{label} must never stand in for a producer fingerprint"
        );
    }

    let mut fixture_substitution = certificate.clone();
    fixture_substitution.golden_fixture_fingerprint = "a".repeat(64);
    let fixture_substitution_fingerprint = fixture_substitution.fingerprint();
    let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
        &policy_fingerprint,
        &fixture_substitution_fingerprint,
        &fixture,
        150,
        7,
        7,
    )
    .unwrap();
    assert_eq!(
        reference.verify_certified_foreign_producer_with(
            &candidate,
            &fixture_substitution,
            trusted,
        ),
        Err(ProducerCompatibilityErrorV1::FixtureBindingMismatch)
    );

    let mut fabricated_candidate = candidate.clone();
    fabricated_candidate.producer.golden_vectors.vectors_sha256 = "f".repeat(64);
    fabricated_candidate.validate().unwrap();
    assert_eq!(
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &fabricated_candidate,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        ),
        Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch)
    );

    let mut fabricated_reference = reference.clone();
    let mut fabricated_candidate = candidate;
    fabricated_reference.producer.golden_vectors.vectors_sha256 = "e".repeat(64);
    fabricated_candidate.producer.golden_vectors.vectors_sha256 = "e".repeat(64);
    fabricated_reference.validate().unwrap();
    fabricated_candidate.validate().unwrap();
    assert_eq!(
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &fabricated_reference,
            &fabricated_candidate,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        ),
        Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch),
        "two matching self-asserted summaries cannot replace the independently executed fixture"
    );
}

#[test]
fn foreign_certificate_enforces_time_revision_and_full_width_boundaries() {
    let (reference, candidate, fixture) = sample_foreign_producer_pair();
    let policy_fingerprint = "9".repeat(64);
    let certificate =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &candidate,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        )
        .unwrap();
    let certificate_fingerprint = certificate.fingerprint();

    for (time, expected) in [
        (
            99,
            Err(ProducerCompatibilityErrorV1::CertificateNotYetValid),
        ),
        (100, Ok(ProducerCompatibilityKindV1::Certified)),
        (199, Ok(ProducerCompatibilityKindV1::Certified)),
        (200, Err(ProducerCompatibilityErrorV1::CertificateExpired)),
    ] {
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &certificate_fingerprint,
            &fixture,
            time,
            7,
            7,
        )
        .unwrap();
        let observed = reference
            .verify_certified_foreign_producer_with(&candidate, &certificate, trusted)
            .map(|witness| witness.kind());
        assert_eq!(observed, expected, "evaluation time {time}");
    }

    for (minimum_revision, maximum_revision) in [(8, 8), (1, 6)] {
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &certificate_fingerprint,
            &fixture,
            150,
            minimum_revision,
            maximum_revision,
        )
        .unwrap();
        assert_eq!(
            reference.verify_certified_foreign_producer_with(&candidate, &certificate, trusted,),
            Err(ProducerCompatibilityErrorV1::CertificateRevisionOutsidePolicy)
        );
    }

    let full_width =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &candidate,
            &fixture,
            &policy_fingerprint,
            u64::MAX,
            u64::MAX - 2,
            u64::MAX,
        )
        .unwrap();
    let full_width_fingerprint = full_width.fingerprint();
    let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
        &policy_fingerprint,
        &full_width_fingerprint,
        &fixture,
        u64::MAX - 1,
        u64::MAX,
        u64::MAX,
    )
    .unwrap();
    assert_eq!(
        reference
            .verify_certified_foreign_producer_with(&candidate, &full_width, trusted)
            .unwrap()
            .kind(),
        ProducerCompatibilityKindV1::Certified
    );
}

#[test]
fn producer_compatibility_rejects_input_tokenizer_and_invalid_identity_drift() {
    let (reference, candidate, fixture) = sample_foreign_producer_pair();
    let policy_fingerprint = "9".repeat(64);

    let mut changed_input = candidate.clone();
    changed_input.input.canonicalization.push_str("-drift");
    changed_input.space.input_contract_fingerprint = changed_input.input.fingerprint();
    changed_input.producer.space_fingerprint = changed_input.space.fingerprint();
    changed_input.validate().unwrap();
    assert_eq!(
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &changed_input,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        ),
        Err(ProducerCompatibilityErrorV1::SpaceMismatch)
    );

    let mut changed_tokenizer = candidate.clone();
    changed_tokenizer.space.tokenizer_fingerprint = "7".repeat(64);
    changed_tokenizer.producer.space_fingerprint = changed_tokenizer.space.fingerprint();
    changed_tokenizer.validate().unwrap();
    assert_eq!(
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &changed_tokenizer,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        ),
        Err(ProducerCompatibilityErrorV1::SpaceMismatch)
    );

    let mut invalid_reference = reference.clone();
    invalid_reference.storage.dimension = 0;
    assert_eq!(
        invalid_reference.verify_exact_producer_with(&reference),
        Err(ProducerCompatibilityErrorV1::InvalidReferenceIdentity)
    );
    let mut invalid_candidate = candidate;
    invalid_candidate.storage.dimension = 0;
    assert_eq!(
        reference.verify_exact_producer_with(&invalid_candidate),
        Err(ProducerCompatibilityErrorV1::InvalidCandidateIdentity)
    );
}

#[test]
fn foreign_certificate_and_verified_fixture_fail_closed_on_untrusted_bytes() {
    let (reference, candidate, fixture) = sample_foreign_producer_pair();
    let policy_fingerprint = "9".repeat(64);
    let certificate =
        ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
            &reference,
            &candidate,
            &fixture,
            &policy_fingerprint,
            7,
            100,
            200,
        )
        .unwrap();

    let mut unknown = serde_json::to_value(&certificate).unwrap();
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(true));
    assert!(serde_json::from_value::<ForeignProducerConformanceCertificateV1>(unknown).is_err());

    let raw = serde_json::to_string(&certificate).unwrap();
    let duplicate_schema = raw.replacen('{', "{\"schema_version\":1,", 1);
    assert!(
        serde_json::from_str::<ForeignProducerConformanceCertificateV1>(&duplicate_schema).is_err()
    );

    let texts = ["query: alpha", "document: beta"];
    let reference_vectors = vec![
        vec![0.0, -0.0, 1.0, -1.0, 0.25, 0.5, 0.75, 1.25],
        vec![2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5],
    ];
    let mut changed_bits = reference_vectors.clone();
    changed_bits[0][0] = -0.0;
    assert_eq!(
        VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(
            &texts,
            &reference_vectors,
            &changed_bits,
        ),
        Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch)
    );

    let nan_texts = ["NaN payload"];
    let reference_nan = vec![vec![f32::from_bits(0x7fc0_0041)]];
    let candidate_nan = vec![vec![f32::from_bits(0x7fc0_0042)]];
    assert_eq!(
        VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(
            &nan_texts,
            &reference_nan,
            &candidate_nan,
        ),
        Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch),
        "NaN payload bits are part of exact producer conformance"
    );
    let malformed = vec![vec![1.0], vec![]];
    assert_eq!(
        VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(
            &texts,
            &reference_vectors,
            &malformed,
        ),
        Err(ProducerCompatibilityErrorV1::GoldenFixtureInvalid)
    );
}

#[test]
fn every_identity_field_participates_in_the_bundle_fingerprint() {
    let base = EmbeddingIdentityBundleV1::explicit_test_model("mutation-matrix", 256);
    let base_fingerprint = base.fingerprint();
    macro_rules! changed {
        ($label:literal, $mutate:expr) => {{
            let mut candidate = base.clone();
            $mutate(&mut candidate);
            assert_ne!(base_fingerprint, candidate.fingerprint(), $label);
        }};
    }

    changed!("space schema", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .schema_version +=
        1);
    changed!("logical model", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .logical_model_id
        .push('x'));
    changed!("immutable revision", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .immutable_revision
        .push('x'));
    changed!("space kind", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .kind =
        EmbeddingSpaceKindV1::Semantic);
    changed!(
        "manifest fingerprint",
        |v: &mut EmbeddingIdentityBundleV1| v.space.artifact_manifest_fingerprint = "a".repeat(64)
    );
    changed!("artifact", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .artifacts
        .push(EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 7,
        }));
    changed!("tokenizer", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .tokenizer_fingerprint =
        "a".repeat(64));
    changed!("vocabulary", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .vocabulary_fingerprint =
        "b".repeat(64));
    changed!("model config", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .model_config_fingerprint =
        "c".repeat(64));
    changed!(
        "model preprocessing",
        |v: &mut EmbeddingIdentityBundleV1| v.space.model_preprocessing.push('x')
    );
    changed!("sequence policy", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .sequence_policy
        .push('x'));
    changed!("query instruction", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .query_instruction
        .push('x'));
    changed!(
        "document instruction",
        |v: &mut EmbeddingIdentityBundleV1| v.space.document_instruction.push('x')
    );
    changed!("pooling", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .pooling
        .push('x'));
    changed!("normalization", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .output_normalization
        .push('x'));
    changed!("space dimension", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .dimension +=
        1);
    changed!("input binding", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .input_contract_fingerprint =
        "d".repeat(64));
    changed!("hash algorithm", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .hash_control
        .as_mut()
        .unwrap()
        .algorithm
        .push('x'));
    changed!("hash revision", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .hash_control
        .as_mut()
        .unwrap()
        .algorithm_revision
        .push('x'));
    changed!("hash seed", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .hash_control
        .as_mut()
        .unwrap()
        .seed +=
        1);
    changed!("hash features", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .hash_control
        .as_mut()
        .unwrap()
        .feature_rules
        .push('x'));
    changed!("hash tokenization", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .hash_control
        .as_mut()
        .unwrap()
        .tokenization_rules
        .push('x'));
    changed!("hash signing", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .hash_control
        .as_mut()
        .unwrap()
        .signing_rules
        .push('x'));
    changed!("hash normalization", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .hash_control
        .as_mut()
        .unwrap()
        .normalization_rules
        .push('x'));
    changed!("projection", |v: &mut EmbeddingIdentityBundleV1| v
        .space
        .projection =
        Some(EmbeddingProjectionV1 {
            parent_space_fingerprint: "e".repeat(64),
            source_dimension: 512,
            output_dimension: 256,
            projection_rule: "prefix".to_owned(),
            renormalization_rule: "l2".to_owned(),
        }));

    let artifact_base = sample_semantic_identity();
    let artifact_fingerprint = artifact_base.fingerprint();
    macro_rules! changed_artifact_field {
        ($label:literal, $mutate:expr) => {{
            let mut candidate = artifact_base.clone();
            $mutate(&mut candidate.space.artifacts[0]);
            assert_ne!(artifact_fingerprint, candidate.fingerprint(), $label);
        }};
    }
    changed_artifact_field!(
        "artifact role",
        |artifact: &mut EmbeddingArtifactIdentityV1| {
            artifact.role.push('x');
        }
    );
    changed_artifact_field!(
        "artifact digest",
        |artifact: &mut EmbeddingArtifactIdentityV1| {
            artifact.sha256 = "a".repeat(64);
        }
    );
    changed_artifact_field!(
        "artifact size",
        |artifact: &mut EmbeddingArtifactIdentityV1| {
            artifact.size += 1;
        }
    );

    let projection_base = sample_semantic_identity()
        .derive_projection(4, "prefix-truncate-v1", "l2-f32-after-prefix-v1")
        .unwrap();
    let projection_fingerprint = projection_base.fingerprint();
    macro_rules! changed_projection_field {
        ($label:literal, $mutate:expr) => {{
            let mut candidate = projection_base.clone();
            $mutate(candidate.space.projection.as_mut().unwrap());
            assert_ne!(projection_fingerprint, candidate.fingerprint(), $label);
        }};
    }
    changed_projection_field!(
        "projection parent",
        |projection: &mut EmbeddingProjectionV1| {
            projection.parent_space_fingerprint = "a".repeat(64);
        }
    );
    changed_projection_field!(
        "projection source dimension",
        |projection: &mut EmbeddingProjectionV1| {
            projection.source_dimension += 1;
        }
    );
    changed_projection_field!(
        "projection output dimension",
        |projection: &mut EmbeddingProjectionV1| {
            projection.output_dimension += 1;
        }
    );
    changed_projection_field!(
        "projection rule",
        |projection: &mut EmbeddingProjectionV1| {
            projection.projection_rule.push('x');
        }
    );
    changed_projection_field!(
        "projection renormalization",
        |projection: &mut EmbeddingProjectionV1| {
            projection.renormalization_rule.push('x');
        }
    );

    changed!("producer schema", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .schema_version +=
        1);
    changed!("backend", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .backend
        .push('x'));
    changed!("implementation", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .implementation_revision
        .push('x'));
    changed!("protocol", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .protocol_revision
        .push('x'));
    changed!("numeric profile", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .numeric_profile
        .push('x'));
    changed!(
        "producer provenance manifest",
        |v: &mut EmbeddingIdentityBundleV1| v.producer.provenance_manifest_fingerprint =
            "c".repeat(64)
    );
    changed!(
        "producer space binding",
        |v: &mut EmbeddingIdentityBundleV1| v.producer.space_fingerprint = "f".repeat(64)
    );
    changed!("golden corpus", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .golden_vectors
        .corpus_sha256 =
        "a".repeat(64));
    changed!("golden vectors", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .golden_vectors
        .vectors_sha256 =
        "b".repeat(64));
    changed!("golden count", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .golden_vectors
        .vector_count +=
        1);
    changed!("golden dimension", |v: &mut EmbeddingIdentityBundleV1| v
        .producer
        .golden_vectors
        .dimension +=
        1);

    changed!("input schema", |v: &mut EmbeddingIdentityBundleV1| v
        .input
        .schema_version +=
        1);
    changed!("canonicalization", |v: &mut EmbeddingIdentityBundleV1| v
        .input
        .canonicalization
        .push('x'));
    changed!("content selection", |v: &mut EmbeddingIdentityBundleV1| v
        .input
        .content_selection
        .push('x'));
    changed!("chunking", |v: &mut EmbeddingIdentityBundleV1| v
        .input
        .chunking
        .push('x'));
    changed!(
        "outer query instruction",
        |v: &mut EmbeddingIdentityBundleV1| v.input.query_instruction.push('x')
    );
    changed!(
        "outer document instruction",
        |v: &mut EmbeddingIdentityBundleV1| v.input.document_instruction.push('x')
    );
    changed!(
        "document id semantics",
        |v: &mut EmbeddingIdentityBundleV1| v.input.doc_id_semantics.push('x')
    );

    changed!("storage schema", |v: &mut EmbeddingIdentityBundleV1| v
        .storage
        .schema_version +=
        1);
    changed!("storage format", |v: &mut EmbeddingIdentityBundleV1| v
        .storage
        .format
        .push('x'));
    changed!("quantization", |v: &mut EmbeddingIdentityBundleV1| v
        .storage
        .quantization =
        QuantizationFormat::Int8);
    changed!("endianness", |v: &mut EmbeddingIdentityBundleV1| v
        .storage
        .endianness
        .push('x'));
    changed!(
        "storage normalization",
        |v: &mut EmbeddingIdentityBundleV1| v.storage.vector_normalization.push('x')
    );
    changed!("storage dimension", |v: &mut EmbeddingIdentityBundleV1| {
        v.storage.dimension += 1;
    });
}

#[test]
fn artifact_role_order_is_canonical_but_duplicate_roles_fail() {
    let left = sample_semantic_identity();
    let mut right = left.clone();
    right.space.artifacts.reverse();
    right.producer.space_fingerprint = right.space.fingerprint();
    right.validate().unwrap();
    assert_eq!(left.space.fingerprint(), right.space.fingerprint());
    assert_eq!(left.fingerprint(), right.fingerprint());
    assert_eq!(
        left.verify_exact_producer_with(&right).unwrap().kind(),
        ProducerCompatibilityKindV1::Exact
    );

    let mut duplicate = left;
    duplicate
        .space
        .artifacts
        .push(duplicate.space.artifacts[0].clone());
    assert!(duplicate.validate().is_err());
}

#[test]
fn hash_control_requires_its_canonical_profile_fingerprint() {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("hash-profile-binding", 32);
    identity.space.artifact_manifest_fingerprint = "0".repeat(64);
    identity.producer.space_fingerprint = identity.space.fingerprint();
    assert!(identity.validate().is_err());

    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("hash-producer-binding", 32);
    identity.producer.provenance_manifest_fingerprint = "0".repeat(64);
    assert!(identity.validate().is_err());
}

#[test]
fn storage_identity_rejects_cross_field_encoding_contradictions() {
    let mut quantized_memory =
        EmbeddingIdentityBundleV1::explicit_test_model("quantized-memory", 32);
    quantized_memory.storage.quantization = QuantizationFormat::F16;
    assert!(quantized_memory.validate().is_err());

    let mut byte_ordered_memory =
        EmbeddingIdentityBundleV1::explicit_test_model("byte-ordered-memory", 32);
    byte_ordered_memory.storage.endianness = "little-endian".to_owned();
    assert!(byte_ordered_memory.validate().is_err());

    let mut native_fsvi = EmbeddingIdentityBundleV1::explicit_test_model("native-fsvi", 32);
    native_fsvi.storage.format = "fsvi-v2".to_owned();
    native_fsvi.storage.endianness = "native-f32-values".to_owned();
    assert!(native_fsvi.validate().is_err());
}

#[test]
fn frozen_bundle_rejects_noncanonical_bytes_digest_drift_and_unknown_schema() {
    let identity = sample_semantic_identity();
    let frozen = identity.freeze().unwrap();
    frozen.validate().unwrap();

    let mut bad_bytes = frozen.clone();
    bad_bytes.canonical_bytes.push(0);
    assert!(bad_bytes.validate().is_err());

    let mut bad_digest = frozen.clone();
    bad_digest.fingerprint = "0".repeat(64);
    assert!(bad_digest.validate().is_err());

    let mut injected_digest = frozen.clone();
    injected_digest.fingerprint = "digest\nforged-log-line".to_owned();
    let error = injected_digest.validate().unwrap_err();
    assert!(error.to_string().contains("redacted-invalid-sha256"));
    assert!(!error.to_string().contains("forged-log-line"));

    let mut unknown_space = identity.clone();
    unknown_space.space.schema_version += 1;
    assert!(unknown_space.validate().is_err());
    let mut unknown_producer = identity.clone();
    unknown_producer.producer.schema_version += 1;
    assert!(unknown_producer.validate().is_err());
    let mut unknown_input = identity.clone();
    unknown_input.input.schema_version += 1;
    assert!(unknown_input.validate().is_err());
    let mut unknown_storage = identity;
    unknown_storage.storage.schema_version += 1;
    assert!(unknown_storage.validate().is_err());

    let mut normalization_drift = sample_semantic_identity();
    normalization_drift
        .storage
        .vector_normalization
        .push_str("-drift");
    assert!(normalization_drift.validate().is_err());

    let mut oversized_instruction = sample_semantic_identity();
    oversized_instruction.space.query_instruction = "x".repeat(MAX_IDENTITY_FIELD_BYTES + 1);
    assert!(oversized_instruction.validate().is_err());

    let mut unknown_field = serde_json::to_value(sample_semantic_identity()).unwrap();
    unknown_field["space"]["future_unregistered_field"] = serde_json::json!(true);
    assert!(
        serde_json::from_value::<EmbeddingIdentityBundleV1>(unknown_field).is_err(),
        "versioned identities must reject unknown fields instead of silently dropping them"
    );

    let mut log_injection = EmbeddingIdentityBundleV1::explicit_test_model("control-character", 32);
    log_injection.space.logical_model_id = "safe\nforged-log-line".to_owned();
    let error = log_injection.validate().unwrap_err();
    assert!(error.to_string().contains("control characters"));
    assert!(!error.to_string().contains("forged-log-line"));

    let mut digest_injection =
        EmbeddingIdentityBundleV1::explicit_test_model("digest-control-character", 32);
    digest_injection.space.tokenizer_fingerprint = "digest\nforged-log-line".to_owned();
    let error = digest_injection.validate().unwrap_err();
    assert!(error.to_string().contains("redacted-invalid-sha256"));
    assert!(!error.to_string().contains("forged-log-line"));
}

#[test]
fn generation_manifest_persists_and_revalidates_identity_bytes_and_digest() {
    let manifest = valid_manifest();
    let encoded = serde_json::to_value(&manifest).unwrap();
    let persisted = &encoded["embedders"]["fast"];
    assert!(
        persisted["canonical_bytes"]
            .as_array()
            .is_some_and(|v| !v.is_empty())
    );
    assert_eq!(persisted["fingerprint"].as_str().map(str::len), Some(64));

    let mut tampered = manifest;
    tampered
        .embedders
        .get_mut("fast")
        .unwrap()
        .canonical_bytes
        .push(0);
    assert!(!validate_manifest(&tampered).is_valid());
}

#[test]
fn mrl_identity_is_structurally_derived_and_cross_bound() {
    let parent = sample_semantic_identity();
    let child = parent
        .derive_projection(4, "prefix-truncate-v1", "l2-f32-after-prefix-v1")
        .unwrap();
    child.validate().unwrap();
    let projection = child.space.projection.as_ref().unwrap();
    assert_eq!(
        projection.parent_space_fingerprint,
        parent.space.fingerprint()
    );
    assert_eq!(projection.source_dimension, 8);
    assert_eq!(projection.output_dimension, 4);
    assert_eq!(child.storage.dimension, 4);
    assert_eq!(child.space.output_normalization, "l2-f32-after-prefix-v1");
    assert_eq!(child.storage.vector_normalization, "l2-f32-after-prefix-v1");
    assert_eq!(child.producer.space_fingerprint, child.space.fingerprint());
    assert!(
        child
            .producer
            .implementation_revision
            .starts_with("frankensearch-identity-projection-wrapper-v1:parent=")
    );
    assert_eq!(
        child.producer.protocol_revision,
        "deterministic-identity-projection-v1"
    );
    assert_ne!(parent.fingerprint(), child.fingerprint());
}
