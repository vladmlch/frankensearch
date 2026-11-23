use std::collections::BTreeMap;

use super::*;

pub(super) fn authority_reference(sequence: u64, predecessor: Option<[u8; 32]>) -> AuthorityRefV1 {
    AuthorityRefV1::new(
        sequence,
        [u8::try_from(sequence).expect("small test sequence"); 16],
        4_096 + sequence,
        [u8::try_from(sequence + 16).expect("small test digest"); 32],
        predecessor,
    )
    .expect("valid test authority reference")
}

pub(super) fn authority_slot(slot_index: u8, authority: AuthorityRefV1) -> AuthoritySlotV1 {
    AuthoritySlotV1 {
        slot_index,
        root_id: [0x5a; 16],
        authority,
    }
}

pub(super) fn lock_frame(kind: GenerationLockFrameKindV1) -> GenerationLockFrameV1 {
    GenerationLockFrameV1::new(kind, [0x71; 16], [0x72; 16], [0x73; 16], 1, [0x74; 32])
        .expect("valid test lock frame")
}

pub(super) fn authority_lock(
    kind: GenerationLockFrameKindV1,
    authority: AuthorityRefV1,
) -> GenerationLockFrameV1 {
    GenerationLockFrameV1::new(
        kind,
        [0x5a; 16],
        [0x72; 16],
        [0x73; 16],
        1,
        authority.fingerprint(),
    )
    .expect("valid authority-bound test lock frame")
}

pub(super) fn component_receipts() -> GenerationComponentReceiptsV1 {
    GenerationComponentReceiptsV1 {
        vector: GenerationComponentReceiptV1 {
            byte_len: 101,
            sha256: [0x11; 32],
        },
        lexical: GenerationComponentReceiptV1 {
            byte_len: 102,
            sha256: [0x12; 32],
        },
        ann: GenerationComponentReceiptV1 {
            byte_len: 103,
            sha256: [0x13; 32],
        },
        metadata: GenerationComponentReceiptV1 {
            byte_len: 104,
            sha256: [0x14; 32],
        },
    }
}

pub(super) fn activation_manifest(
    authority_sequence: u64,
    predecessor: Option<AuthorityRefV1>,
) -> ActivationManifestV1 {
    ActivationManifestV1::new(
        authority_sequence,
        predecessor,
        GenerationAuthorityActionV1::Activate,
        ArtifactGenerationIdentityV1::new(7, [0x21; 16]).expect("test generation"),
        [0x31; 32],
        [0x32; 32],
        [0x33; 32],
        component_receipts(),
    )
    .expect("valid activation manifest")
}

pub(super) fn sample_embedder() -> EmbedderRevision {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("potion-128M", 256);
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = QuantizationFormat::F16;
    identity.storage.endianness = "little-endian".to_owned();
    identity.freeze().unwrap()
}

pub(super) fn sample_semantic_identity() -> EmbeddingIdentityBundleV1 {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("semantic-test", 8);
    identity.space.kind = EmbeddingSpaceKindV1::Semantic;
    identity.space.hash_control = None;
    identity.space.artifact_manifest_fingerprint = "1".repeat(64);
    identity.space.artifacts = vec![
        EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "2".repeat(64),
            size: 10,
        },
        EmbeddingArtifactIdentityV1 {
            role: "tokenizer".to_owned(),
            sha256: "3".repeat(64),
            size: 20,
        },
    ];
    identity.space.tokenizer_fingerprint = "3".repeat(64);
    identity.space.vocabulary_fingerprint = "4".repeat(64);
    identity.space.model_config_fingerprint = "5".repeat(64);
    identity.producer.space_fingerprint = identity.space.fingerprint();
    identity.validate().expect("sample semantic identity");
    identity
}

pub(super) fn sample_foreign_producer_pair() -> (
    EmbeddingIdentityBundleV1,
    EmbeddingIdentityBundleV1,
    VerifiedGoldenConformanceManifestV1,
) {
    let texts = ["query: alpha", "document: beta"];
    let vectors = vec![
        vec![0.0, -0.0, 1.0, -1.0, 0.25, 0.5, 0.75, 1.25],
        vec![2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5],
    ];
    let fixture =
        VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(&texts, &vectors, &vectors)
            .expect("exact fixture");

    let mut reference = sample_semantic_identity();
    reference.producer.golden_vectors = fixture.certificate().clone();
    reference.validate().expect("reference identity");

    let mut candidate = reference.clone();
    candidate.producer.backend = "alternate-native-kernel".to_owned();
    candidate.producer.implementation_revision = "alternate-implementation-v2".to_owned();
    candidate.producer.protocol_revision = "alternate-protocol-v3".to_owned();
    candidate.producer.numeric_profile = "deterministic-f32-alternate-v1".to_owned();
    candidate.producer.provenance_manifest_fingerprint = "6".repeat(64);
    candidate.validate().expect("candidate identity");
    assert_ne!(
        reference.producer.fingerprint(),
        candidate.producer.fingerprint()
    );

    (reference, candidate, fixture)
}

pub(super) fn sample_vector_artifact(path: &str, count: u64) -> VectorArtifact {
    VectorArtifact {
        path: path.into(),
        size_bytes: 1024,
        checksum: "deadbeef".into(),
        vector_count: count,
        dimension: 256,
        embedder_tier: EmbedderTierTag::Fast,
    }
}

pub(super) fn sample_lexical_artifact(path: &str, count: u64) -> LexicalArtifact {
    LexicalArtifact {
        path: path.into(),
        size_bytes: 2048,
        checksum: "cafebabe".into(),
        document_count: count,
    }
}

pub(super) fn valid_manifest() -> GenerationManifest {
    let mut embedders = BTreeMap::new();
    embedders.insert("fast".into(), sample_embedder());

    let mut manifest = GenerationManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        generation_id: "gen-001".into(),
        manifest_hash: String::new(),
        commit_range: CommitRange { low: 1, high: 100 },
        build_started_at: 1_700_000_000_000,
        build_completed_at: 1_700_000_060_000,
        embedders,
        vector_artifacts: vec![sample_vector_artifact("vectors/shard_0.fsvi", 100)],
        lexical_artifacts: vec![sample_lexical_artifact("lexical/segment_0", 100)],
        repair_descriptors: vec![RepairDescriptor {
            protected_artifact: "vectors/shard_0.fsvi".into(),
            sidecar_path: "vectors/shard_0.fsvi.fec".into(),
            source_symbols: 64,
            repair_symbols: 13,
            overhead_ratio: 0.2,
        }],
        activation_invariants: vec![
            ActivationInvariant {
                id: "all_artifacts".into(),
                description: "All artifacts verified".into(),
                kind: InvariantKind::AllArtifactsVerified,
            },
            ActivationInvariant {
                id: "embedder_match".into(),
                description: "Embedder revision matches runtime".into(),
                kind: InvariantKind::EmbedderRevisionMatch,
            },
        ],
        total_documents: 100,
        metadata: BTreeMap::new(),
    };
    manifest.manifest_hash = compute_manifest_hash(&manifest).expect("hash");
    manifest
}

pub(super) fn refresh_manifest_hash(manifest: &mut GenerationManifest) {
    manifest.manifest_hash = compute_manifest_hash(manifest).expect("hash");
}

pub(super) fn docset(documents: &[&str]) -> CanonicalDocsetV1 {
    CanonicalDocsetV1::from_ordered_live_documents(documents.iter().copied())
        .expect("valid canonical docset")
}

pub(super) const CONTROL_DOCS: [&str; 4] = ["doc-a", "doc-b", "doc-c", "doc-d"];

pub(super) fn component(
    role: GenerationComponentRole,
    docset_digest: [u8; 32],
    checkpoint: [u8; 32],
) -> ExactComponentReceiptV1 {
    ExactComponentReceiptV1 {
        role,
        bytes: GenerationComponentReceiptV1 {
            byte_len: 4096 + u64::from(role.wire()),
            sha256: [role.wire(); 32],
        },
        docset_digest,
        live_document_count: 4,
        source_checkpoint: checkpoint,
    }
}

/// All four roles agreeing. Every mutation test below is this quartet with
/// exactly one field moved, so a failure is attributable to that field.
pub(super) fn control_quartet() -> (
    ExactComponentReceiptV1,
    ExactComponentReceiptV1,
    ExactComponentReceiptV1,
    ExactComponentReceiptV1,
) {
    let digest = docset(&CONTROL_DOCS).digest();
    let checkpoint = [0x5c; 32];
    (
        component(GenerationComponentRole::Vector, digest, checkpoint),
        component(GenerationComponentRole::Lexical, digest, checkpoint),
        component(GenerationComponentRole::Ann, digest, checkpoint),
        component(GenerationComponentRole::Metadata, digest, checkpoint),
    )
}
