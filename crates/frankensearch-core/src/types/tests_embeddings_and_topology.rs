use super::tests_support::{identity, semantic_identity, test_digest};
use super::*;

#[test]
fn bound_embedding_rejects_dimension_mismatch_at_bind_time() {
    let error = BoundQueryEmbedding::new(vec![0.5; 7], identity("fast-model", 8))
        .expect_err("7-dim vector cannot bind an 8-dim space");
    assert!(matches!(
        error,
        SearchError::DimensionMismatch {
            expected: 8,
            found: 7
        }
    ));
}

#[test]
fn bound_embedding_matches_identity_bound_validation_and_redacts_debug() {
    let mut malformed = identity("malformed-bundle", 8);
    malformed.producer.space_fingerprint = test_digest('f');
    assert!(BoundQueryEmbedding::new(vec![0.0; 8], malformed).is_err());

    assert!(BoundQueryEmbedding::new(Vec::new(), identity("zero-dimension", 0)).is_err());

    let mut persisted = identity("persisted-storage", 8);
    persisted.storage.format = "fsvi-v2".to_owned();
    persisted.storage.endianness = "little-endian".to_owned();
    persisted
        .validate()
        .expect("persisted bundle is valid identity, but not an in-memory vector contract");
    assert!(BoundQueryEmbedding::new(vec![0.0; 8], persisted).is_err());

    let bound = BoundQueryEmbedding::new(
        vec![0.123_456, 0.654_321, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        identity("debug-private-model", 8),
    )
    .expect("valid in-memory binding");
    let diagnostic = format!("{bound:?}");
    assert!(!diagnostic.contains("0.123456"));
    assert!(!diagnostic.contains("0.654321"));
    assert!(!diagnostic.contains("debug-private-model"));
    assert!(diagnostic.contains("dimension: 8"));
}

#[test]
fn same_dimension_wrong_space_is_rejected_by_fingerprint() {
    // The defining bd-9xuj case: identical dimensions, different models.
    // A raw &[f32] API cannot tell these apart; the bound embedding must.
    let fast =
        BoundQueryEmbedding::new(vec![0.5; 8], identity("fast-model", 8)).expect("bind fast");
    let quality_space = identity("quality-model", 8);
    assert_ne!(
        fast.identity_fingerprint(),
        quality_space.fingerprint(),
        "distinct models must have distinct fingerprints"
    );
    let error = fast
        .verify_space(&quality_space.fingerprint(), "quality")
        .expect_err("fast vector must not enter the quality space");
    assert!(
        matches!(&error, SearchError::InvalidConfig { .. }),
        "expected InvalidConfig, got {error:?}"
    );
    if let SearchError::InvalidConfig { field, reason, .. } = &error {
        assert_eq!(field, "query_embedding.quality.identity");
        assert!(reason.contains("different embedding space"));
    }
}

#[test]
fn matching_space_verifies() {
    let bundle = identity("fast-model", 8);
    let expected = bundle.fingerprint();
    let bound = BoundQueryEmbedding::new(vec![0.5; 8], bundle).expect("bind");
    bound
        .verify_space(&expected, "fast")
        .expect("same space must verify");
    assert_eq!(bound.vector().len(), 8);
}

// ─── bd-9xuj T2-C1 (rebuild): bind-time validation + corrected admission law ───

#[test]
fn bind_time_validation_rejects_incoherent_identity_bundle() -> Result<(), String> {
    // The bundle must prove itself before it can prove anything to a
    // consumer: this producer's space_fingerprint no longer binds the
    // bundled space, so binding must surface the bundle's own typed
    // validation error instead of accepting the incoherent claim.
    let mut broken = identity("bind-validate-model", 8);
    broken.producer.space_fingerprint = broken.producer.golden_vectors.corpus_sha256.clone();
    let error = BoundQueryEmbedding::new(vec![0.5; 8], broken)
        .expect_err("an incoherent identity bundle must not bind");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig { field, .. } = error else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "embedding_identity.producer.space_fingerprint");
    Ok(())
}

/// Pins the T2-C1 impedance mismatch (readiness map §0.1): a query-side
/// bundle binds `in-memory-*` storage while an index-side bundle binds
/// `fsvi-v2`, so for the SAME mathematical space the full-bundle
/// fingerprints can never match — `verify_space` fails closed on
/// legitimate traffic at every index seam, while the space-scoped
/// `verify_space_identity` joins on the space fingerprint and admits it.
#[test]
fn space_scoped_verify_joins_across_storage_formats() {
    // Query side: explicit test identity, in-memory f32 storage.
    let query_side = identity("shared-model", 8);
    // Index side: the SAME space, persisted as fsvi-v2 little-endian.
    let mut index_side = identity("shared-model", 8);
    index_side.storage.format = "fsvi-v2".to_owned();
    index_side.storage.endianness = "little-endian".to_owned();
    index_side
        .validate()
        .expect("index-side bundle must be a legitimate, validating bundle");

    assert_eq!(
        query_side.space.fingerprint(),
        index_side.space.fingerprint(),
        "same model + dimension is the same mathematical space"
    );
    assert_ne!(
        query_side.fingerprint(),
        index_side.fingerprint(),
        "storage difference must alter the full-bundle fingerprint"
    );

    let bound = BoundQueryEmbedding::new(vec![0.5; 8], query_side).expect("bind");
    assert_eq!(
        bound.space_fingerprint(),
        bound.identity().space.fingerprint(),
        "bind-time space fingerprint must match the bundled space"
    );

    // Full-bundle verify: fails closed on this legitimate pairing (§0.1).
    bound
        .verify_space(&index_side.fingerprint(), "quality")
        .expect_err("full-bundle fingerprints never match across storage formats");
    // Space-scoped verify: the correct join key at the index seam.
    bound
        .verify_space_identity(&index_side.space.fingerprint(), "quality")
        .expect("same space must verify space-scoped across storage formats");
}

/// Pins the T2-C1 impedance mismatch (readiness map §0.1) under the
/// corrected admission law: same space, same producer, different
/// storage. The full-bundle fingerprints can never match across an
/// index seam, the space fingerprint joins, and admission is granted as
/// typed [`SpaceIdentityAdmission::SameProducer`] — not because "same
/// space is sufficient".
#[test]
fn same_producer_admits_across_storage_formats() {
    // Query side: explicit test identity, in-memory f32 storage.
    let query_side = identity("shared-model", 8);
    // Index side: the SAME space and producer, persisted as fsvi-v2.
    let mut index_side = identity("shared-model", 8);
    index_side.storage.format = "fsvi-v2".to_owned();
    index_side.storage.endianness = "little-endian".to_owned();
    index_side
        .validate()
        .expect("index-side bundle must be a legitimate, validating bundle");

    assert_eq!(
        query_side.space.fingerprint(),
        index_side.space.fingerprint(),
        "same model + dimension is the same mathematical space"
    );
    assert_ne!(
        query_side.fingerprint(),
        index_side.fingerprint(),
        "storage difference must alter the full-bundle fingerprint"
    );

    let bound = BoundQueryEmbedding::new(vec![0.5; 8], query_side).expect("bind");
    assert_eq!(
        bound.space_fingerprint(),
        bound.identity().space.fingerprint(),
        "bind-time space fingerprint must match the bundled space"
    );

    // Full-bundle verify: fails closed on this legitimate pairing (§0.1)...
    bound
        .verify_space(&index_side.fingerprint(), "quality")
        .expect_err("full-bundle fingerprints never match across storage formats");
    // ...the space fingerprint joins at the seam...
    bound
        .verify_space_identity(&index_side.space.fingerprint(), "quality")
        .expect("same space must join space-scoped across storage formats");
    // ...and the admission law grants the pairing as the SAME producer.
    let admission = bound
        .verify_producer_conformance(&index_side, "quality")
        .expect("same producer must be admitted");
    assert_eq!(admission, SpaceIdentityAdmission::SameProducer);
    assert_eq!(admission.code(), "same_producer");
}

#[test]
fn same_dimension_wrong_space_is_rejected_by_space_fingerprint() -> Result<(), String> {
    // The defining bd-9xuj case through the seam verifier: identical
    // dimensions, different models. Dimension checks cannot tell these
    // apart; the space fingerprint must.
    let bound =
        BoundQueryEmbedding::new(vec![0.5; 8], identity("fast-model", 8)).expect("bind fast");
    let quality_space = identity("quality-model", 8);
    let error = bound
        .verify_space_identity(&quality_space.space.fingerprint(), "quality")
        .expect_err("fast vector must not enter the quality space");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig {
        field,
        value,
        reason,
    } = error
    else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "query_embedding.quality.space_identity");
    assert_eq!(value, bound.space_fingerprint());
    assert!(reason.contains("different embedding space"));
    assert!(reason.contains(&quality_space.space.fingerprint()));

    // The bundle-level admission law rejects the same pairing at the
    // space join, before producer conformance is even considered.
    let error = bound
        .verify_producer_conformance(&quality_space, "quality")
        .expect_err("wrong space must reject at the bundle level too");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig { field, .. } = error else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "query_embedding.quality.space_identity");
    Ok(())
}

/// The corrected admission law (bd-9xuj map, post-review supersession
/// notice): a foreign producer in the same space is admitted ONLY when
/// both producers carry the byte-identical pinned golden-vector
/// certificate, with typed
/// [`SpaceIdentityAdmission::ConformanceCompatibleProducer`] telemetry;
/// otherwise it is rejected. Matching space fingerprints alone never
/// admit a pairing.
#[test]
fn foreign_producer_admission_requires_conformance_certificate() -> Result<(), String> {
    let base = identity("conformance-model", 8);
    let bound = BoundQueryEmbedding::new(vec![0.5; 8], base.clone()).expect("bind");

    // Same space, different producer, SAME golden-vector certificate:
    // certified conformance-compatible → admitted, telemetry-typed.
    let mut certified = base.clone();
    certified.producer.backend = "alternate-conformant-backend".to_owned();
    certified
        .validate()
        .expect("certified sibling must validate");
    assert_ne!(
        certified.producer.fingerprint(),
        base.producer.fingerprint(),
        "a different backend attestation must alter the producer fingerprint"
    );
    let admission = bound
        .verify_producer_conformance(&certified, "fast")
        .expect("certified foreign producer must be admitted");
    assert_eq!(admission.code(), "conformance_compatible_producer");
    let rendered = format!("{admission:?}");
    let SpaceIdentityAdmission::ConformanceCompatibleProducer {
        query_producer_fingerprint,
        expected_producer_fingerprint,
    } = admission
    else {
        return Err(format!(
            "expected ConformanceCompatibleProducer, got {rendered}"
        ));
    };
    assert_eq!(query_producer_fingerprint, base.producer.fingerprint());
    assert_eq!(
        expected_producer_fingerprint,
        certified.producer.fingerprint()
    );

    // Same space, different producer, DIFFERENT golden-vector
    // certificate: same-space is not sufficient — typed rejection.
    let mut uncertified = base.clone();
    uncertified.producer.backend = "alternate-uncertified-backend".to_owned();
    uncertified.producer.golden_vectors.vectors_sha256 = base.space.fingerprint();
    uncertified
        .validate()
        .expect("uncertified sibling must still be a coherent bundle");
    let error = bound
        .verify_producer_conformance(&uncertified, "fast")
        .expect_err("uncertified foreign producer must be rejected");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig {
        field,
        value,
        reason,
    } = error
    else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "query_embedding.fast.producer_conformance");
    assert_eq!(value, base.producer.fingerprint());
    assert!(reason.contains("golden-vector certificate"));
    Ok(())
}

/// Pins the KNOWN GAP the review (#8151) named, so it is documented
/// rather than hidden: two producers whose golden-vector certificates
/// differ ONLY in the corpus digest were certified against different
/// golden corpora, so their conformance is not comparable at all —
/// yet today the admission law rejects them through the same single
/// reject arm as a same-corpus conformance failure. The rejection is
/// correct but VACUOUS: no typed `Unattestable` classification exists
/// yet to distinguish "cannot be attested" from "attested and failed".
/// When that third [`SpaceIdentityAdmission`] arm lands (the enum is
/// `#[non_exhaustive]` for exactly that addition), this test pins the
/// behavior it replaces.
#[test]
fn different_corpus_digest_certificates_reject_vacuously_today() -> Result<(), String> {
    let base = identity("corpus-digest-model", 8);
    let bound = BoundQueryEmbedding::new(vec![0.5; 8], base.clone()).expect("bind");

    // Expected side differs from the query side ONLY in the golden
    // corpus digest: a certificate genuinely pinned to a different
    // (equally valid) golden corpus.
    let mut foreign_corpus = base.clone();
    foreign_corpus.producer.golden_vectors.corpus_sha256 =
        crate::generation::GoldenVectorCertificateV1::corpus_fingerprint(&[
            "a different golden corpus",
        ])
        .expect("corpus fingerprint");
    foreign_corpus
        .validate()
        .expect("a certificate pinned to another corpus is still a coherent bundle");
    assert_ne!(
        foreign_corpus.producer.golden_vectors.corpus_sha256,
        base.producer.golden_vectors.corpus_sha256,
        "the corpus digests must differ"
    );
    assert_eq!(
        foreign_corpus.producer.golden_vectors.vectors_sha256,
        base.producer.golden_vectors.vectors_sha256,
        "everything but the corpus digest must be identical"
    );
    assert_eq!(
        foreign_corpus.producer.golden_vectors.vector_count,
        base.producer.golden_vectors.vector_count
    );
    assert_eq!(
        foreign_corpus.producer.golden_vectors.dimension,
        base.producer.golden_vectors.dimension
    );

    // Today: rejected through the single producer-conformance reject
    // arm — indistinguishable from a same-corpus conformance failure.
    let error = bound
        .verify_producer_conformance(&foreign_corpus, "fast")
        .expect_err("a different-corpus certificate must not be admitted");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig { field, .. } = error else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "query_embedding.fast.producer_conformance");
    Ok(())
}

#[test]
fn distinct_models_at_equal_dimension_always_reject_space_scoped() {
    // Readiness-map §3.3(10) property at C1 scope: explicit_test_model(a, d)
    // vs (b, d) with a != b must always reject, for every tier label —
    // at the space join and under the full admission law alike.
    let dims = [8_u32, 384];
    let models = ["model-a", "model-b", "model-c"];
    for dim in dims {
        for (i, a) in models.iter().enumerate() {
            for b in &models[i + 1..] {
                let bound = BoundQueryEmbedding::new(vec![0.25; dim as usize], identity(a, dim))
                    .expect("bind");
                let other = identity(b, dim);
                for tier in ["fast", "quality"] {
                    bound
                        .verify_space_identity(&other.space.fingerprint(), tier)
                        .expect_err("distinct models must never share a space");
                    bound
                        .verify_producer_conformance(&other, tier)
                        .expect_err("distinct models must never be admitted");
                }
                // Reflexive: a model always verifies against its own space.
                bound
                    .verify_space_identity(&identity(a, dim).space.fingerprint(), "fast")
                    .expect("own space must verify");
            }
        }
    }
}

#[test]
fn tiered_constructors_retain_independent_bindings() {
    let fast = BoundQueryEmbedding::new(vec![0.1; 8], identity("fast-model", 8)).unwrap();
    let quality = BoundQueryEmbedding::new(vec![0.2; 16], identity("quality-model", 16)).unwrap();

    let progressive = TieredQueryEmbeddings::progressive(fast.clone(), quality.clone());
    assert!(progressive.fast().is_some() && progressive.quality().is_some());

    let fast_only = TieredQueryEmbeddings::fast_only(fast);
    assert!(fast_only.fast().is_some() && fast_only.quality().is_none());
    let quality_only = TieredQueryEmbeddings::quality_only(quality);
    assert!(quality_only.fast().is_none() && quality_only.quality().is_some());
}

#[test]
fn space_identity_admission_codes_and_serde_are_stable() {
    let same = SpaceIdentityAdmission::SameProducer;
    assert_eq!(same.code(), "same_producer");
    let json = serde_json::to_string(&same).expect("serialize");
    assert!(json.contains("\"admission\":\"same_producer\""));

    let compat = SpaceIdentityAdmission::ConformanceCompatibleProducer {
        query_producer_fingerprint: "a".repeat(64),
        expected_producer_fingerprint: "b".repeat(64),
    };
    assert_eq!(compat.code(), "conformance_compatible_producer");
    let json = serde_json::to_string(&compat).expect("serialize");
    assert!(json.contains("\"admission\":\"conformance_compatible_producer\""));
    let back: SpaceIdentityAdmission = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, compat);
}

#[test]
fn tiered_constructors_report_supported_topology() {
    let fast = BoundQueryEmbedding::new(vec![0.1; 8], semantic_identity("fast-model", 8)).unwrap();
    let quality =
        BoundQueryEmbedding::new(vec![0.2; 16], semantic_identity("quality-model", 16)).unwrap();

    let progressive = TieredQueryEmbeddings::progressive(fast.clone(), quality.clone());
    assert_eq!(
        progressive.supported_topology(),
        RetrievalTopology::FullProgressive
    );

    assert_eq!(
        TieredQueryEmbeddings::fast_only(fast).supported_topology(),
        RetrievalTopology::FastOnly
    );
    assert_eq!(
        TieredQueryEmbeddings::quality_only(quality).supported_topology(),
        RetrievalTopology::QualityOnly
    );
}

/// bd-ctzo C4: a hash-control space can never report a semantic topology.
///
/// This test previously did not exist, and the one above asserted
/// `FullProgressive` for two `explicit_test_model` bundles — which are
/// `HashControl` spaces. Every fixture in the tree was therefore claiming
/// semantic availability it did not have.
#[test]
fn a_hash_control_space_can_never_report_a_semantic_topology() {
    let hash_fast = BoundQueryEmbedding::new(vec![0.1; 8], identity("hash-fast", 8)).unwrap();
    let hash_quality = BoundQueryEmbedding::new(vec![0.2; 8], identity("hash-quality", 8)).unwrap();
    let semantic_fast =
        BoundQueryEmbedding::new(vec![0.1; 8], semantic_identity("real-fast", 8)).unwrap();
    let semantic_quality =
        BoundQueryEmbedding::new(vec![0.2; 8], semantic_identity("real-quality", 8)).unwrap();

    for (label, embeddings) in [
        (
            "both hash",
            TieredQueryEmbeddings::progressive(hash_fast.clone(), hash_quality.clone()),
        ),
        (
            "fast hash only",
            TieredQueryEmbeddings::fast_only(hash_fast.clone()),
        ),
        (
            "quality hash only",
            TieredQueryEmbeddings::quality_only(hash_quality.clone()),
        ),
        // A MIXED pair is still not semantic: a topology describes the
        // whole retrieval, not its strongest arm.
        (
            "semantic fast + hash quality",
            TieredQueryEmbeddings::progressive(semantic_fast.clone(), hash_quality),
        ),
        (
            "hash fast + semantic quality",
            TieredQueryEmbeddings::progressive(hash_fast, semantic_quality.clone()),
        ),
    ] {
        assert_eq!(
            embeddings.supported_topology(),
            RetrievalTopology::HashControl,
            "{label} must not claim a semantic topology"
        );
        assert!(embeddings.binds_hash_control(), "{label}");
    }

    // Control: two learned spaces DO report a semantic topology, so the
    // guard above is not simply answering HashControl for everything.
    let semantic = TieredQueryEmbeddings::progressive(semantic_fast, semantic_quality);
    assert_eq!(
        semantic.supported_topology(),
        RetrievalTopology::FullProgressive
    );
    assert!(!semantic.binds_hash_control());
}

/// bd-ctzo C4: unknown coverage has no number, so it cannot be reported
/// as zero.
#[test]
fn unknown_tier_coverage_cannot_be_read_as_a_zero_count() {
    let unknown = TierQueryCoverageV1::Unknown {
        reason: CoverageUnknownReasonV1::LegacyUnidentified,
    };
    let empty_but_live = TierQueryCoverageV1::Witnessed {
        generation_sequence: 7,
        live_count: 0,
        contributed_candidates: 0,
    };
    assert_eq!(unknown.witnessed_live_count(), None);
    assert_eq!(empty_but_live.witnessed_live_count(), Some(0));
    assert_ne!(
        unknown, empty_but_live,
        "an unwitnessed tier and a genuinely empty one must not compare equal"
    );

    let coverage = SearchCoverageV1::new(
        RetrievalTopology::FastOnly,
        unknown,
        TierQueryCoverageV1::NotRequested,
    );
    let summary = coverage.redacted_summary();
    assert!(summary.contains("fast=unknown"), "{summary}");
    assert!(summary.contains("quality=not_requested"), "{summary}");
    // The summary is built from closed codes and integers only; there is
    // no field on the type through which a doc id or vector could reach it.
    assert!(!summary.contains("doc"), "{summary}");
}

#[test]
fn topology_codes_are_stable_and_semantic_partition_is_correct() {
    let all = [
        RetrievalTopology::LexicalOnly,
        RetrievalTopology::HashControl,
        RetrievalTopology::FastOnly,
        RetrievalTopology::QualityOnly,
        RetrievalTopology::FullProgressive,
        RetrievalTopology::PartialQuality {
            coverage_ppm: 750_000,
        },
    ];
    let codes: Vec<&str> = all.iter().map(RetrievalTopology::code).collect();
    assert_eq!(
        codes,
        vec![
            "lexical_only",
            "hash_control",
            "fast_only",
            "quality_only",
            "full_progressive",
            "partial_quality",
        ]
    );
    for topology in all {
        let semantic = topology.is_semantic();
        match topology {
            RetrievalTopology::LexicalOnly | RetrievalTopology::HashControl => {
                assert!(!semantic, "{topology:?} must not claim semantic");
            }
            _ => assert!(semantic, "{topology:?} is semantic"),
        }
    }
}

#[test]
fn topology_serde_roundtrips_with_coverage() {
    let topology = RetrievalTopology::PartialQuality {
        coverage_ppm: 333_333,
    };
    let json = serde_json::to_string(&topology).expect("serialize");
    assert!(json.contains("partial_quality"));
    assert!(json.contains("333333"));
    let decoded: RetrievalTopology = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded, topology);
}
