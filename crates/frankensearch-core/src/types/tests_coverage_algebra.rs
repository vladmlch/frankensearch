use std::collections::BTreeSet;

use proptest::prelude::*;

use super::coverage_codec::{derive_coverage_relation, derive_lossless_semantic_topology};
use super::tests_support::{
    bound, identity, members, semantic_identity, test_digest, test_scope, verified_documents,
    verified_documents_context,
};
use super::*;
use crate::generation::EmbeddingSpaceKindV1;
use crate::recovery_plan::COMPLETE_COVERAGE_PPM;

#[test]
fn coverage_relations_and_checked_counts_are_derived_from_unique_members() {
    let bundle = semantic_identity("coverage-relations", 8);
    let cases = [
        (
            members(&[]),
            members(&[]),
            CoverageRelationV1::ZeroLive,
            [0, 0, 0, 0, 0],
        ),
        (
            members(&["a", "b"]),
            members(&["b", "a"]),
            CoverageRelationV1::Complete,
            [2, 2, 2, 2, u64::from(COMPLETE_COVERAGE_PPM)],
        ),
        (
            members(&["a", "b", "c"]),
            members(&["a", "c"]),
            CoverageRelationV1::Subset,
            [3, 2, 2, 3, 666_666],
        ),
        (
            members(&["a", "b"]),
            members(&["a", "b", "c"]),
            CoverageRelationV1::Superset,
            [2, 3, 2, 3, u64::from(COMPLETE_COVERAGE_PPM)],
        ),
        (
            members(&["a", "b"]),
            members(&["b", "c"]),
            CoverageRelationV1::Overlap,
            [2, 2, 1, 3, 500_000],
        ),
        (
            members(&["a"]),
            members(&["b"]),
            CoverageRelationV1::Disjoint,
            [1, 1, 0, 2, 0],
        ),
        (
            members(&["a"]),
            members(&[]),
            CoverageRelationV1::Subset,
            [1, 0, 0, 1, 0],
        ),
    ];

    for (live, indexed, expected_relation, expected_counts) in cases {
        let coverage = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            live,
            test_scope('1', '2'),
            indexed,
            None,
        )
        .expect("set relation derives");
        let facts = coverage.verified_facts();
        assert!(facts.is_some());
        if let Some((relation, counts)) = facts {
            assert_eq!(relation, expected_relation);
            assert_eq!(counts.live_count(), expected_counts[0]);
            assert_eq!(counts.indexed_count(), expected_counts[1]);
            assert_eq!(counts.intersection_count(), expected_counts[2]);
            assert_eq!(counts.union_count(), expected_counts[3]);
            assert_eq!(u64::from(counts.coverage_ppm()), expected_counts[4]);
        }
    }
}

#[test]
fn canonical_documents_ignore_order_but_ordered_shards_bind_order() {
    let bundle = semantic_identity("witness-order", 8);
    let documents_a = verified_documents(
        CoverageTierV1::Fast,
        &bundle,
        &["doc-c", "doc-a", "doc-b"],
        &["doc-a"],
    );
    let documents_b = verified_documents(
        CoverageTierV1::Fast,
        &bundle,
        &["doc-b", "doc-c", "doc-a"],
        &["doc-a"],
    );
    assert_eq!(documents_a.fingerprint(), documents_b.fingerprint());

    let shards_a = TierCoverageV1::verified(
        CoverageTierV1::Fast,
        &bundle.space,
        CoverageBasisV1::OrderedShards,
        test_scope('1', '2'),
        members(&["shard-a", "shard-b"]),
        test_scope('1', '2'),
        members(&["shard-a"]),
        None,
    )
    .expect("ordered shard coverage validates");
    let shards_b = TierCoverageV1::verified(
        CoverageTierV1::Fast,
        &bundle.space,
        CoverageBasisV1::OrderedShards,
        test_scope('1', '2'),
        members(&["shard-b", "shard-a"]),
        test_scope('1', '2'),
        members(&["shard-a"]),
        None,
    )
    .expect("reordered shard coverage validates");
    assert_ne!(shards_a.fingerprint(), shards_b.fingerprint());
    assert_eq!(
        shards_a
            .live_witness()
            .map(CoverageWitnessV1::set_fingerprint),
        shards_b
            .live_witness()
            .map(CoverageWitnessV1::set_fingerprint),
        "set facts stay comparable while ordered membership identity changes"
    );
    assert_ne!(
        shards_a
            .live_witness()
            .map(CoverageWitnessV1::membership_fingerprint),
        shards_b
            .live_witness()
            .map(CoverageWitnessV1::membership_fingerprint)
    );
}

#[test]
fn duplicate_members_never_inflate_document_or_shard_coverage() {
    let bundle = semantic_identity("duplicate-rejection", 8);
    for basis in [
        CoverageBasisV1::CanonicalDocuments,
        CoverageBasisV1::OrderedShards,
    ] {
        let error = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &bundle.space,
            basis,
            test_scope('1', '2'),
            members(&["same", "same"]),
            test_scope('1', '2'),
            members(&["same"]),
            None,
        )
        .expect_err("duplicate members fail closed");
        assert!(error.to_string().contains("duplicate member"));
    }
}

#[test]
fn empty_and_oversized_member_diagnostics_are_bounded_and_redacted() {
    let bundle = semantic_identity("bounded-diagnostics", 8);
    for bad_member in [String::new(), "sensitive-query-document".repeat(300)] {
        let error = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            vec![bad_member.clone()],
            test_scope('1', '2'),
            Vec::new(),
            None,
        )
        .expect_err("invalid member fails closed");
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("redacted-member-at-index-0"));
        if !bad_member.is_empty() {
            assert!(!diagnostic.contains(&bad_member));
        }
        assert!(diagnostic.len() < 512);
    }
}

#[test]
fn malformed_digest_diagnostics_never_echo_caller_material() {
    let caller_material = "caller-controlled-model-path-fragment".repeat(400);
    let error = CoverageScopeV1::new(caller_material.clone(), test_digest('2'))
        .expect_err("malformed generation digest fails closed");
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("redacted-invalid-sha256-length"));
    assert!(!diagnostic.contains(&caller_material));
    assert!(diagnostic.len() < 512);
}

#[test]
fn maximum_count_and_fixed_point_boundaries_are_overflow_safe() {
    let complete = CoverageCountsV1::checked(u64::MAX, u64::MAX, u64::MAX, u64::MAX)
        .expect("maximum complete counts fit");
    assert_eq!(complete.coverage_ppm(), COMPLETE_COVERAGE_PPM);

    let one_of_three =
        CoverageCountsV1::checked(3, 1, 1, 3).expect("one-third counts are algebraically valid");
    assert_eq!(one_of_three.coverage_ppm(), 333_333);

    let floor_boundary = CoverageCountsV1::checked(u64::MAX, 1, 1, u64::MAX)
        .expect("large denominator uses a safe u128 intermediate");
    assert_eq!(floor_boundary.coverage_ppm(), 0);

    assert!(CoverageCountsV1::checked(u64::MAX, u64::MAX, 0, u64::MAX).is_err());
    assert!(CoverageCountsV1::checked(1, 1, 2, 0).is_err());
    assert!(CoverageCountsV1::checked(2, 2, 1, 2).is_err());
}

#[test]
fn missing_unknown_unverified_and_zero_live_are_distinct_closed_states() {
    let bundle = semantic_identity("closed-states", 8);
    let missing = TierCoverageV1::not_requested(CoverageTierV1::Fast);
    let requested_missing = TierCoverageV1::requested_missing(
        CoverageTierV1::Fast,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a"]),
    )
    .expect("requested missing tier validates");
    let unknown = TierCoverageV1::unknown(
        CoverageTierV1::Fast,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a"]),
        CoverageUnknownReasonV1::OwnerUnavailable,
    )
    .expect("unknown tier validates");
    let unverified = TierCoverageV1::unverified(
        CoverageTierV1::Fast,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a"]),
        CoverageUnverifiedReasonV1::CountsUntrusted,
        Some(9),
    )
    .expect("unverified tier validates");
    let zero_live = verified_documents(CoverageTierV1::Fast, &bundle, &[], &[]);

    assert!(matches!(
        missing.requested(),
        TierCoverageRequestV1::MissingTier
    ));
    assert!(matches!(
        requested_missing.realized(),
        TierCoverageRealizationV1::MissingTier
    ));
    assert!(matches!(
        unknown.realized(),
        TierCoverageRealizationV1::Unknown { .. }
    ));
    assert!(matches!(
        unverified.realized(),
        TierCoverageRealizationV1::Unverified { .. }
    ));
    assert_eq!(
        zero_live.verified_facts().map(|facts| facts.0),
        Some(CoverageRelationV1::ZeroLive)
    );

    let fingerprints = [
        missing.fingerprint(),
        requested_missing.fingerprint(),
        unknown.fingerprint(),
        unverified.fingerprint(),
        zero_live.fingerprint(),
    ];
    assert_eq!(fingerprints.iter().collect::<BTreeSet<_>>().len(), 5);
}

#[test]
fn cross_scope_comparison_requires_exact_evidence_and_never_claims_complete() {
    let bundle = semantic_identity("cross-scope", 8);
    let no_evidence = TierCoverageV1::verified(
        CoverageTierV1::Quality,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('3', '2'),
        members(&["a"]),
        None,
    );
    assert!(no_evidence.is_err());

    let redundant_evidence = TierCoverageV1::verified(
        CoverageTierV1::Quality,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('1', '2'),
        members(&["a"]),
        Some(test_digest('9')),
    );
    assert!(redundant_evidence.is_err());

    let cross_scope_complete = TierCoverageV1::verified(
        CoverageTierV1::Quality,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('3', '2'),
        members(&["b", "a"]),
        Some(test_digest('9')),
    );
    assert!(cross_scope_complete.is_err());

    let partial = TierCoverageV1::verified(
        CoverageTierV1::Quality,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('3', '4'),
        members(&["a"]),
        Some(test_digest('9')),
    )
    .expect("explicit evidence admits a checked cross-scope partial relation");
    assert_eq!(
        partial.verified_facts().map(|facts| facts.0),
        Some(CoverageRelationV1::Subset)
    );
    assert!(partial.cross_scope_evidence().is_some());
}

#[test]
fn cross_scope_evidence_mutation_and_witness_borrowing_fail_closed() {
    let bundle = semantic_identity("evidence-mutation", 8);
    let partial = TierCoverageV1::verified(
        CoverageTierV1::Quality,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('3', '2'),
        members(&["a"]),
        Some(test_digest('9')),
    )
    .expect("cross-scope partial validates");
    let mut mutated = serde_json::to_value(&partial).expect("serialize coverage");
    mutated["realized"]["cross_scope_evidence"]["live_set_fingerprint"] =
        serde_json::Value::String(test_digest('f'));
    let untrusted: UntrustedTierCoverageV1 =
        serde_json::from_value(mutated).expect("mutated claim remains syntactic");
    let context = TrustedTierCoverageContextV1::verified(
        CoverageTierV1::Quality,
        bundle.space.clone(),
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('3', '2'),
        members(&["a"]),
        Some(test_digest('9')),
    )
    .expect("trusted cross-scope context validates");
    assert!(untrusted.validate_against(&context).is_err());

    let fast = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
    let mut borrowed = serde_json::to_value(&fast).expect("serialize fast coverage");
    borrowed["tier"] = serde_json::Value::String("quality".to_owned());
    let untrusted: UntrustedTierCoverageV1 =
        serde_json::from_value(borrowed).expect("borrowed claim remains syntactic");
    let fast_context =
        verified_documents_context(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
    assert!(untrusted.validate_against(&fast_context).is_err());
}

#[test]
fn pair_rejects_swapped_roles_and_tier_domains_make_equal_sets_distinct() {
    let bundle = semantic_identity("tier-domain", 8);
    let fast = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a", "b"]);
    let quality = verified_documents(CoverageTierV1::Quality, &bundle, &["a", "b"], &["a", "b"]);
    assert_ne!(
        fast.live_witness().map(CoverageWitnessV1::fingerprint),
        quality.live_witness().map(CoverageWitnessV1::fingerprint),
        "tier role is identity material even for equal document sets"
    );
    assert!(TierCoveragePairV1::new(quality.clone(), fast.clone()).is_err());
    assert!(TierCoveragePairV1::new(fast, quality).is_ok());
}

#[test]
fn complete_independent_semantic_tiers_derive_full_progressive_topology() {
    let fast_bundle = semantic_identity("topology-fast", 8);
    let quality_bundle = semantic_identity("topology-quality", 16);
    let coverage = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b"], &["a", "b"]),
        verified_documents(
            CoverageTierV1::Quality,
            &quality_bundle,
            &["a", "b"],
            &["b", "a"],
        ),
    )
    .expect("independent coverage pair validates");
    let embeddings = TieredQueryEmbeddings::progressive(bound(fast_bundle), bound(quality_bundle));
    let decision = coverage
        .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
        .expect("validated complete bindings derive topology");
    assert_eq!(decision.requested(), RetrievalTopology::FullProgressive);
    assert_eq!(
        decision.coarse_realized(),
        Some(RetrievalTopology::FullProgressive)
    );
    assert!(decision.semantic_available());
}

#[test]
fn partial_quality_is_derived_from_quality_intersection_not_a_scalar_claim() {
    let fast_bundle = semantic_identity("partial-fast", 8);
    let quality_bundle = semantic_identity("partial-quality", 16);
    let coverage = TierCoveragePairV1::new(
        verified_documents(
            CoverageTierV1::Fast,
            &fast_bundle,
            &["a", "b", "c"],
            &["a", "b", "c"],
        ),
        verified_documents(
            CoverageTierV1::Quality,
            &quality_bundle,
            &["a", "b", "c"],
            &["a", "c"],
        ),
    )
    .expect("partial quality pair validates");
    let embeddings = TieredQueryEmbeddings::progressive(bound(fast_bundle), bound(quality_bundle));
    let decision = coverage
        .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
        .expect("checked quality intersection derives partial topology");
    assert_eq!(decision.requested(), RetrievalTopology::FullProgressive);
    assert_eq!(
        decision.coarse_realized(),
        Some(RetrievalTopology::PartialQuality {
            coverage_ppm: 666_666
        })
    );
    assert!(decision.semantic_available());
}

#[test]
fn complete_quality_only_derives_direct_quality_retrieval() {
    let quality_bundle = semantic_identity("direct-quality", 16);
    let pair = TierCoveragePairV1::new(
        TierCoverageV1::not_requested(CoverageTierV1::Fast),
        verified_documents(
            CoverageTierV1::Quality,
            &quality_bundle,
            &["a", "b"],
            &["a", "b"],
        ),
    )
    .expect("quality-only pair validates");
    let embeddings = TieredQueryEmbeddings::quality_only(bound(quality_bundle));
    let decision = pair
        .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
        .expect("direct quality binding derives quality-only retrieval");
    assert_eq!(decision.requested(), RetrievalTopology::QualityOnly);
    assert_eq!(
        decision.coarse_realized(),
        Some(RetrievalTopology::QualityOnly)
    );
    assert!(decision.semantic_available());
}

#[test]
fn exact_realization_preserves_partial_fast_multi_partial_and_quality_only_partial() {
    let fast_bundle = semantic_identity("exact-partial-fast", 8);
    let quality_bundle = semantic_identity("exact-partial-quality", 16);

    let fast_only_pair = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b", "c"], &["a"]),
        TierCoverageV1::not_requested(CoverageTierV1::Quality),
    )
    .expect("partial fast-only pair validates");
    let fast_only = fast_only_pair
        .derive_topology(
            RetrievalIntentV1::Semantic,
            Some(&TieredQueryEmbeddings::fast_only(bound(
                fast_bundle.clone(),
            ))),
        )
        .expect("partial fast remains an exact available tier");
    assert!(fast_only.semantic_available());
    assert!(fast_only.coarse_realized().is_none());
    assert!(matches!(
        fast_only.fast_realized(),
        RealizedTierCoverageV1::Partial { counts, .. }
            if counts.intersection_count() == 1
    ));
    assert!(matches!(
        fast_only.quality_realized(),
        RealizedTierCoverageV1::NotRequested
    ));

    let multi_pair = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b", "c"], &["a"]),
        verified_documents(
            CoverageTierV1::Quality,
            &quality_bundle,
            &["a", "b", "c"],
            &["b"],
        ),
    )
    .expect("two independently partial tiers validate");
    let multi = multi_pair
        .derive_topology(
            RetrievalIntentV1::Semantic,
            Some(&TieredQueryEmbeddings::progressive(
                bound(fast_bundle.clone()),
                bound(quality_bundle.clone()),
            )),
        )
        .expect("multi-partial exact realization derives");
    assert!(multi.semantic_available());
    assert!(multi.coarse_realized().is_none());
    assert!(matches!(
        multi.fast_realized(),
        RealizedTierCoverageV1::Partial { .. }
    ));
    assert!(matches!(
        multi.quality_realized(),
        RealizedTierCoverageV1::Partial { .. }
    ));

    let quality_only_pair = TierCoveragePairV1::new(
        TierCoverageV1::not_requested(CoverageTierV1::Fast),
        verified_documents(
            CoverageTierV1::Quality,
            &quality_bundle,
            &["a", "b", "c"],
            &["c"],
        ),
    )
    .expect("partial quality-only pair validates");
    let quality_only = quality_only_pair
        .derive_topology(
            RetrievalIntentV1::Semantic,
            Some(&TieredQueryEmbeddings::quality_only(bound(quality_bundle))),
        )
        .expect("partial quality-only exact realization derives");
    assert!(quality_only.semantic_available());
    assert!(quality_only.coarse_realized().is_none());
    assert!(matches!(
        quality_only.quality_realized(),
        RealizedTierCoverageV1::Partial { .. }
    ));
}

#[test]
fn positive_intersection_contributes_even_when_ppm_rounds_to_zero() {
    let counts = CoverageCountsV1::checked(u64::MAX, 1, 1, u64::MAX)
        .expect("extreme exact subset counts validate");
    assert_eq!(counts.coverage_ppm(), 0);
    let realized = RealizedTierCoverageV1::Partial {
        relation: CoverageRelationV1::Subset,
        counts,
    };
    realized.validate().expect("exact partial state validates");
    assert!(realized.contributes());
    assert!(
        derive_lossless_semantic_topology(&realized, &RealizedTierCoverageV1::NotRequested)
            .is_none()
    );
}

#[test]
fn pair_requires_one_exact_requested_universe_and_typed_cross_scope_proof() {
    let fast_bundle = semantic_identity("universe-fast", 8);
    let quality_bundle = semantic_identity("universe-quality", 16);
    let fast = verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b"], &["a", "b"]);
    let different_members = verified_documents(
        CoverageTierV1::Quality,
        &quality_bundle,
        &["a", "c"],
        &["a", "c"],
    );
    assert!(TierCoveragePairV1::new(fast.clone(), different_members).is_err());

    let ordered_quality = TierCoverageV1::verified(
        CoverageTierV1::Quality,
        &quality_bundle.space,
        CoverageBasisV1::OrderedShards,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('1', '2'),
        members(&["a", "b"]),
        None,
    )
    .expect("ordered quality coverage validates independently");
    assert!(TierCoveragePairV1::new(fast.clone(), ordered_quality).is_err());

    let fast_cross = TierCoverageV1::verified(
        CoverageTierV1::Fast,
        &fast_bundle.space,
        CoverageBasisV1::OrderedShards,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('1', '2'),
        members(&["a", "b"]),
        None,
    )
    .expect("fast cross-tier scope validates");
    let quality_cross = TierCoverageV1::verified(
        CoverageTierV1::Quality,
        &quality_bundle.space,
        CoverageBasisV1::OrderedShards,
        test_scope('3', '2'),
        members(&["a", "b"]),
        test_scope('3', '2'),
        members(&["a", "b"]),
        None,
    )
    .expect("quality cross-tier scope validates");
    assert!(TierCoveragePairV1::new(fast_cross.clone(), quality_cross.clone()).is_err());
    assert!(
        TierCoveragePairV1::new_with_cross_tier_authority(
            fast_cross.clone(),
            quality_cross,
            test_digest('8'),
        )
        .is_ok()
    );

    let reordered_quality = TierCoverageV1::verified(
        CoverageTierV1::Quality,
        &quality_bundle.space,
        CoverageBasisV1::OrderedShards,
        test_scope('3', '2'),
        members(&["b", "a"]),
        test_scope('3', '2'),
        members(&["b", "a"]),
        None,
    )
    .expect("reordered quality is independently valid");
    assert!(
        TierCoveragePairV1::new_with_cross_tier_authority(
            fast_cross,
            reordered_quality,
            test_digest('8'),
        )
        .is_err()
    );
}

#[test]
fn unknown_unverified_and_zero_live_never_become_semantic_availability() {
    let fast_bundle = semantic_identity("unavailable-fast", 8);
    let quality_bundle = semantic_identity("unavailable-quality", 16);
    let live = members(&["a"]);
    let variants = [
        TierCoverageV1::requested_missing(
            CoverageTierV1::Quality,
            &quality_bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            live.clone(),
        )
        .expect("missing quality validates"),
        TierCoverageV1::unknown(
            CoverageTierV1::Quality,
            &quality_bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            live.clone(),
            CoverageUnknownReasonV1::OwnerUnavailable,
        )
        .expect("unknown quality validates"),
        TierCoverageV1::unverified(
            CoverageTierV1::Quality,
            &quality_bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            live,
            CoverageUnverifiedReasonV1::CountsUntrusted,
            Some(1),
        )
        .expect("unverified quality validates"),
    ];
    for quality in variants {
        let pair = TierCoveragePairV1::new(
            verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a"], &["a"]),
            quality,
        )
        .expect("coverage pair validates");
        let embeddings = TieredQueryEmbeddings::progressive(
            bound(fast_bundle.clone()),
            bound(quality_bundle.clone()),
        );
        let decision = pair
            .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .expect("unavailable quality conservatively drops to fast");
        assert_eq!(
            decision.coarse_realized(),
            Some(RetrievalTopology::FastOnly)
        );
    }

    let zero_pair = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &fast_bundle, &[], &[]),
        verified_documents(CoverageTierV1::Quality, &quality_bundle, &[], &[]),
    )
    .expect("zero-live pair validates");
    let zero_embeddings =
        TieredQueryEmbeddings::progressive(bound(fast_bundle), bound(quality_bundle));
    let zero_decision = zero_pair
        .derive_topology(RetrievalIntentV1::Semantic, Some(&zero_embeddings))
        .expect("zero live degrades without inventing availability");
    assert_eq!(
        zero_decision.coarse_realized(),
        Some(RetrievalTopology::LexicalOnly)
    );
    assert!(!zero_decision.semantic_available());
}

#[test]
fn hash_control_is_explicit_and_can_never_satisfy_semantic_intent() {
    let hash_bundle = identity("explicit-hash-control", 8);
    assert_eq!(hash_bundle.space.kind, EmbeddingSpaceKindV1::HashControl);
    let coverage = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &hash_bundle, &["a"], &["a"]),
        TierCoverageV1::not_requested(CoverageTierV1::Quality),
    )
    .expect("hash coverage pair validates");
    let embeddings = TieredQueryEmbeddings::fast_only(bound(hash_bundle));
    let control = coverage
        .derive_topology(RetrievalIntentV1::HashControl, Some(&embeddings))
        .expect("explicit hash-control intent validates");
    assert_eq!(control.requested(), RetrievalTopology::HashControl);
    assert_eq!(
        control.coarse_realized(),
        Some(RetrievalTopology::HashControl)
    );
    assert!(!control.semantic_available());
    assert!(
        coverage
            .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .is_err()
    );
}

#[test]
fn swapped_duplicate_missing_and_extra_bindings_are_rejected() {
    let fast_bundle = semantic_identity("binding-fast", 8);
    let quality_bundle = semantic_identity("binding-quality", 16);
    let pair = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a"], &["a"]),
        verified_documents(CoverageTierV1::Quality, &quality_bundle, &["a"], &["a"]),
    )
    .expect("coverage pair validates");
    let swapped = TieredQueryEmbeddings::progressive(
        bound(quality_bundle.clone()),
        bound(fast_bundle.clone()),
    );
    assert!(
        pair.derive_topology(RetrievalIntentV1::Semantic, Some(&swapped))
            .is_err()
    );
    let only_fast = TieredQueryEmbeddings::fast_only(bound(fast_bundle));
    assert!(
        pair.derive_topology(RetrievalIntentV1::Semantic, Some(&only_fast))
            .is_err()
    );

    let unrequested_quality = TierCoveragePairV1::new(
        pair.fast().clone(),
        TierCoverageV1::not_requested(CoverageTierV1::Quality),
    )
    .expect("fast-only coverage validates");
    let extra_quality = TieredQueryEmbeddings::progressive(
        bound(semantic_identity("binding-fast", 8)),
        bound(quality_bundle),
    );
    assert!(
        unrequested_quality
            .derive_topology(RetrievalIntentV1::Semantic, Some(&extra_quality))
            .is_err()
    );

    let shared_bundle = semantic_identity("duplicate-binding", 8);
    let duplicate_pair = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &shared_bundle, &["a"], &["a"]),
        verified_documents(CoverageTierV1::Quality, &shared_bundle, &["a"], &["a"]),
    )
    .expect("coverage roles remain independent before binding");
    let duplicate_embeddings =
        TieredQueryEmbeddings::progressive(bound(shared_bundle.clone()), bound(shared_bundle));
    assert!(
        duplicate_pair
            .derive_topology(RetrievalIntentV1::Semantic, Some(&duplicate_embeddings))
            .is_err()
    );
}

#[test]
fn realized_superset_fails_topology_derivation_closed() {
    let bundle = semantic_identity("unsafe-superset", 8);
    let pair = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &bundle, &["live"], &["live", "stale"]),
        TierCoverageV1::not_requested(CoverageTierV1::Quality),
    )
    .expect("superset remains representable as evidence");
    let embeddings = TieredQueryEmbeddings::fast_only(bound(bundle));
    assert!(
        pair.derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
            .is_err()
    );
}

#[test]
fn lexical_only_requires_no_requested_tiers_or_bindings() {
    let pair = TierCoveragePairV1::new(
        TierCoverageV1::not_requested(CoverageTierV1::Fast),
        TierCoverageV1::not_requested(CoverageTierV1::Quality),
    )
    .expect("missing pair validates");
    let decision = pair
        .derive_topology(RetrievalIntentV1::LexicalOnly, None)
        .expect("no vector authority derives lexical only");
    assert_eq!(decision.requested(), RetrievalTopology::LexicalOnly);
    assert_eq!(
        decision.coarse_realized(),
        Some(RetrievalTopology::LexicalOnly)
    );
    assert!(!decision.semantic_available());
}

#[test]
fn coverage_pair_and_derived_topology_serde_roundtrip_canonically() {
    let fast_bundle = semantic_identity("serde-fast", 8);
    let quality_bundle = semantic_identity("serde-quality", 16);
    let fast_context = verified_documents_context(
        CoverageTierV1::Fast,
        &fast_bundle,
        &["a", "b", "c"],
        &["a", "b", "c"],
    );
    let quality_context = verified_documents_context(
        CoverageTierV1::Quality,
        &quality_bundle,
        &["a", "b", "c"],
        &["a", "c"],
    );
    let pair = TierCoveragePairV1::new(
        fast_context.recompute().expect("recompute fast"),
        quality_context.recompute().expect("recompute quality"),
    )
    .expect("serde pair validates");
    let embeddings = TieredQueryEmbeddings::progressive(
        bound(fast_bundle.clone()),
        bound(quality_bundle.clone()),
    );
    let decision = pair
        .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
        .expect("serde decision derives");

    let pair_json = serde_json::to_string(&pair).expect("serialize pair");
    let pair_claim: UntrustedTierCoveragePairV1 =
        serde_json::from_str(&pair_json).expect("deserialize pair");
    let pair_roundtrip = pair_claim
        .validate_against(&fast_context, &quality_context, None)
        .expect("owner-backed pair promotion succeeds");
    assert_eq!(pair_roundtrip, pair);
    assert_eq!(pair_roundtrip.fingerprint(), pair.fingerprint());

    let decision_json = serde_json::to_string(&decision).expect("serialize decision");
    let decision_claim: UntrustedDerivedRetrievalTopologyV1 =
        serde_json::from_str(&decision_json).expect("deserialize decision");
    let decision_roundtrip = decision_claim
        .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
        .expect("fresh derivation promotes decision");
    assert_eq!(decision_roundtrip, decision);
    assert_eq!(decision_roundtrip.fingerprint(), decision.fingerprint());
    assert!(!pair_json.contains("doc-a"));
    assert!(!decision_json.contains("query"));
}

#[test]
fn coherent_wire_claims_cannot_self_attest_coverage_or_cross_tier_authority() {
    let fast_bundle = semantic_identity("wire-authority-fast", 8);
    let quality_bundle = semantic_identity("wire-authority-quality", 16);
    let trusted_fast =
        verified_documents_context(CoverageTierV1::Fast, &fast_bundle, &["a", "b"], &["a"]);

    let coherent_other =
        verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "c"], &["a"]);
    let claim: UntrustedTierCoverageV1 = serde_json::from_value(
        serde_json::to_value(coherent_other).expect("serialize coherent alternate claim"),
    )
    .expect("coherent alternate claim parses");
    assert!(claim.validate_against(&trusted_fast).is_err());

    let transplanted = verified_documents(
        CoverageTierV1::Fast,
        &semantic_identity("wire-authority-other-space", 8),
        &["a", "b"],
        &["a"],
    );
    let claim: UntrustedTierCoverageV1 = serde_json::from_value(
        serde_json::to_value(transplanted).expect("serialize space transplant"),
    )
    .expect("space transplant claim parses");
    assert!(claim.validate_against(&trusted_fast).is_err());

    let fast_context = TrustedTierCoverageContextV1::verified(
        CoverageTierV1::Fast,
        fast_bundle.space.clone(),
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('1', '2'),
        members(&["a", "b"]),
        None,
    )
    .expect("fast owner context validates");
    let quality_context = TrustedTierCoverageContextV1::verified(
        CoverageTierV1::Quality,
        quality_bundle.space.clone(),
        CoverageBasisV1::CanonicalDocuments,
        test_scope('3', '2'),
        members(&["a", "b"]),
        test_scope('3', '2'),
        members(&["a", "b"]),
        None,
    )
    .expect("quality owner context validates");
    let pair = TierCoveragePairV1::new_with_cross_tier_authority(
        fast_context.recompute().expect("recompute fast"),
        quality_context.recompute().expect("recompute quality"),
        test_digest('8'),
    )
    .expect("trusted cross-tier proof builds");
    let mut forged = serde_json::to_value(pair).expect("serialize pair");
    forged["cross_tier_universe_evidence"]["authority_fingerprint"] =
        serde_json::Value::String(test_digest('9'));
    let claim: UntrustedTierCoveragePairV1 =
        serde_json::from_value(forged).expect("self-consistent authority claim parses");
    let independent_authority = test_digest('8');
    assert!(
        claim
            .validate_against(
                &fast_context,
                &quality_context,
                Some(independent_authority.as_str())
            )
            .is_err()
    );
}

#[test]
fn derived_wire_claim_cannot_choose_pair_hash_or_swap_or_duplicate_spaces() {
    let fast_bundle = semantic_identity("derived-wire-fast", 8);
    let quality_bundle = semantic_identity("derived-wire-quality", 16);
    let pair = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a"], &["a"]),
        verified_documents(CoverageTierV1::Quality, &quality_bundle, &["a"], &["a"]),
    )
    .expect("pair validates");
    let embeddings = TieredQueryEmbeddings::progressive(
        bound(fast_bundle.clone()),
        bound(quality_bundle.clone()),
    );
    let decision = pair
        .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
        .expect("decision derives");
    let base = serde_json::to_value(decision).expect("serialize decision");

    let mut arbitrary_pair = base.clone();
    arbitrary_pair["coverage_pair_fingerprint"] = serde_json::Value::String(test_digest('f'));
    let claim: UntrustedDerivedRetrievalTopologyV1 =
        serde_json::from_value(arbitrary_pair).expect("arbitrary pair hash parses");
    assert!(
        claim
            .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
            .is_err()
    );

    let mut swapped = base.clone();
    swapped["fast_space_fingerprint"] =
        serde_json::Value::String(quality_bundle.space.fingerprint());
    swapped["quality_space_fingerprint"] =
        serde_json::Value::String(fast_bundle.space.fingerprint());
    let claim: UntrustedDerivedRetrievalTopologyV1 =
        serde_json::from_value(swapped).expect("swapped space claims parse");
    assert!(
        claim
            .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
            .is_err()
    );

    let mut duplicate = base;
    duplicate["quality_space_fingerprint"] =
        serde_json::Value::String(fast_bundle.space.fingerprint());
    let claim: UntrustedDerivedRetrievalTopologyV1 =
        serde_json::from_value(duplicate).expect("duplicate space claim parses");
    assert!(
        claim
            .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
            .is_err()
    );
}

#[test]
fn coverage_serde_denies_unknown_duplicate_and_impossible_fields() {
    let bundle = semantic_identity("serde-deny", 8);
    let coverage = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
    let mut unknown = serde_json::to_value(&coverage).expect("serialize coverage");
    unknown["borrowed_quality_coverage_ppm"] = serde_json::Value::from(500_000);
    assert!(serde_json::from_value::<UntrustedTierCoverageV1>(unknown).is_err());

    let generation = test_digest('1');
    let corpus = test_digest('2');
    let duplicate_scope = format!(
        "{{\"generation_fingerprint\":\"{generation}\",\"generation_fingerprint\":\"{generation}\",\"corpus_fingerprint\":\"{corpus}\"}}"
    );
    assert!(serde_json::from_str::<CoverageScopeV1>(&duplicate_scope).is_err());

    let mut impossible = serde_json::to_value(&coverage).expect("serialize coverage");
    impossible["realized"]["relation"] = serde_json::Value::String("complete".to_owned());
    let claim: UntrustedTierCoverageV1 =
        serde_json::from_value(impossible).expect("impossible relation remains syntactic");
    let context = verified_documents_context(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
    assert!(claim.validate_against(&context).is_err());

    let topology_with_unknown = r#"{"topology":"fast_only","coverage_ppm":1000000}"#;
    assert!(serde_json::from_str::<RetrievalTopology>(topology_with_unknown).is_err());
    assert!(
        serde_json::from_str::<RetrievalTopology>(r#"{"topology":"partial_quality"}"#).is_err()
    );
    assert!(
        serde_json::from_str::<RetrievalTopology>(
            r#"{"topology":"partial_quality","coverage_ppm":null}"#
        )
        .is_err()
    );
    assert!(
        serde_json::from_str::<RetrievalTopology>(
            r#"{"topology":"fast_only","coverage_ppm":null}"#
        )
        .is_err()
    );
    assert!(
        serde_json::from_str::<RetrievalTopology>(r#"{"topology":"fast_only","extra":"field"}"#)
            .is_err()
    );
}

#[test]
fn retrieval_topology_compatibility_law_is_exhaustive() {
    let variants = [
        RetrievalTopology::LexicalOnly,
        RetrievalTopology::HashControl,
        RetrievalTopology::FastOnly,
        RetrievalTopology::QualityOnly,
        RetrievalTopology::FullProgressive,
        RetrievalTopology::PartialQuality {
            coverage_ppm: 500_000,
        },
    ];
    for requested in variants {
        for realized in variants {
            let expected = match requested {
                RetrievalTopology::LexicalOnly => {
                    matches!(realized, RetrievalTopology::LexicalOnly)
                }
                RetrievalTopology::HashControl => {
                    matches!(realized, RetrievalTopology::HashControl)
                }
                RetrievalTopology::FastOnly => matches!(
                    realized,
                    RetrievalTopology::FastOnly | RetrievalTopology::LexicalOnly
                ),
                RetrievalTopology::QualityOnly => matches!(
                    realized,
                    RetrievalTopology::QualityOnly
                        | RetrievalTopology::PartialQuality { .. }
                        | RetrievalTopology::LexicalOnly
                ),
                RetrievalTopology::FullProgressive => matches!(
                    realized,
                    RetrievalTopology::FullProgressive
                        | RetrievalTopology::PartialQuality { .. }
                        | RetrievalTopology::FastOnly
                        | RetrievalTopology::QualityOnly
                        | RetrievalTopology::LexicalOnly
                ),
                RetrievalTopology::PartialQuality { .. } => false,
            };
            assert_eq!(
                retrieval_topology_fits_request(requested, realized),
                expected,
                "compatibility mismatch for {requested:?} -> {realized:?}"
            );
        }
    }
}

#[test]
fn topology_receipt_mutations_fail_validation() {
    let fast_bundle = semantic_identity("receipt-fast", 8);
    let quality_bundle = semantic_identity("receipt-quality", 16);
    let pair = TierCoveragePairV1::new(
        verified_documents(CoverageTierV1::Fast, &fast_bundle, &["a", "b"], &["a", "b"]),
        verified_documents(
            CoverageTierV1::Quality,
            &quality_bundle,
            &["a", "b"],
            &["a"],
        ),
    )
    .expect("receipt pair validates");
    let embeddings = TieredQueryEmbeddings::progressive(bound(fast_bundle), bound(quality_bundle));
    let decision = pair
        .derive_topology(RetrievalIntentV1::Semantic, Some(&embeddings))
        .expect("receipt derives");

    let base = serde_json::to_value(&decision).expect("serialize decision");
    let mut semantic_lie = base.clone();
    semantic_lie["semantic_available"] = serde_json::Value::Bool(false);
    let claim: UntrustedDerivedRetrievalTopologyV1 =
        serde_json::from_value(semantic_lie).expect("semantic lie remains syntactic");
    assert!(
        claim
            .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
            .is_err()
    );

    let mut complete_partial = base.clone();
    complete_partial["coarse_realized"]["coverage_ppm"] =
        serde_json::Value::from(COMPLETE_COVERAGE_PPM);
    let claim: UntrustedDerivedRetrievalTopologyV1 =
        serde_json::from_value(complete_partial).expect("boundary claim remains syntactic");
    assert!(
        claim
            .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
            .is_err()
    );

    let mut hash_relabel = base;
    hash_relabel["fast_kind"] = serde_json::Value::String("hash_control".to_owned());
    let claim: UntrustedDerivedRetrievalTopologyV1 =
        serde_json::from_value(hash_relabel).expect("kind relabel remains syntactic");
    assert!(
        claim
            .validate_against(&pair, RetrievalIntentV1::Semantic, Some(&embeddings))
            .is_err()
    );
}

#[test]
fn every_coverage_semantic_mutation_changes_the_stable_fingerprint() {
    let bundle = semantic_identity("fingerprint-base", 8);
    let other_bundle = semantic_identity("fingerprint-other", 8);
    let base = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "b"], &["a"]);
    let reordered = verified_documents(CoverageTierV1::Fast, &bundle, &["b", "a"], &["a"]);
    assert_eq!(base.fingerprint(), reordered.fingerprint());

    let changed_member = verified_documents(CoverageTierV1::Fast, &bundle, &["a", "c"], &["a"]);
    let changed_space =
        verified_documents(CoverageTierV1::Fast, &other_bundle, &["a", "b"], &["a"]);
    let changed_scope = TierCoverageV1::verified(
        CoverageTierV1::Fast,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('3', '2'),
        members(&["a", "b"]),
        test_scope('3', '2'),
        members(&["a"]),
        None,
    )
    .expect("changed scope validates");
    let changed_basis = TierCoverageV1::verified(
        CoverageTierV1::Fast,
        &bundle.space,
        CoverageBasisV1::OrderedShards,
        test_scope('1', '2'),
        members(&["a", "b"]),
        test_scope('1', '2'),
        members(&["a"]),
        None,
    )
    .expect("changed basis validates");
    let changed_state = TierCoverageV1::unknown(
        CoverageTierV1::Fast,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(&["a", "b"]),
        CoverageUnknownReasonV1::OwnerUnavailable,
    )
    .expect("changed state validates");
    let fingerprints = [
        base.fingerprint(),
        changed_member.fingerprint(),
        changed_space.fingerprint(),
        changed_scope.fingerprint(),
        changed_basis.fingerprint(),
        changed_state.fingerprint(),
    ];
    assert_eq!(fingerprints.iter().collect::<BTreeSet<_>>().len(), 6);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn coverage_set_algebra_matches_u16_bitsets_and_canonical_order(
        live_mask in any::<u16>(),
        indexed_mask in any::<u16>(),
    ) {
        let bundle = semantic_identity("property-coverage", 8);
        let mut live = (0..16)
            .filter(|bit| live_mask & (1_u16 << bit) != 0)
            .map(|bit| format!("doc-{bit:02}"))
            .collect::<Vec<_>>();
        let indexed = (0..16)
            .filter(|bit| indexed_mask & (1_u16 << bit) != 0)
            .map(|bit| format!("doc-{bit:02}"))
            .collect::<Vec<_>>();
        let coverage_result = TierCoverageV1::verified(
            CoverageTierV1::Fast,
            &bundle.space,
            CoverageBasisV1::CanonicalDocuments,
            test_scope('1', '2'),
            live.clone(),
            test_scope('1', '2'),
            indexed.clone(),
            None,
        );
        prop_assert!(coverage_result.is_ok());
        if let Ok(coverage) = coverage_result {
            let intersection = live_mask & indexed_mask;
            let union = live_mask | indexed_mask;
            let expected_counts = CoverageCountsV1::checked(
                u64::from(live_mask.count_ones()),
                u64::from(indexed_mask.count_ones()),
                u64::from(intersection.count_ones()),
                u64::from(union.count_ones()),
            );
            prop_assert!(expected_counts.is_ok());
            if let Ok(expected_counts) = expected_counts {
                prop_assert_eq!(coverage.verified_facts(), Some((derive_coverage_relation(expected_counts), expected_counts)));
            }
            live.reverse();
            let reordered = TierCoverageV1::verified(
                CoverageTierV1::Fast,
                &bundle.space,
                CoverageBasisV1::CanonicalDocuments,
                test_scope('1', '2'),
                live,
                test_scope('1', '2'),
                indexed,
                None,
            );
            prop_assert!(reordered.is_ok());
            if let Ok(reordered) = reordered {
                prop_assert_eq!(coverage.fingerprint(), reordered.fingerprint());
            }
        }
    }
}
