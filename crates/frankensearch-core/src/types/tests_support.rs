use crate::generation::{
    EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
};

use super::{
    BoundQueryEmbedding, CoverageBasisV1, CoverageScopeV1, CoverageTierV1, TierCoverageV1,
    TrustedTierCoverageContextV1, VectorHit,
};

pub(super) fn vhit(index: u32, score: f32, doc_id: &str) -> VectorHit {
    VectorHit {
        index,
        score,
        doc_id: doc_id.into(),
    }
}

pub(super) fn identity(name: &str, dim: u32) -> EmbeddingIdentityBundleV1 {
    EmbeddingIdentityBundleV1::explicit_test_model(name, dim)
}

pub(super) fn test_digest(nibble: char) -> String {
    assert!(nibble.is_ascii_hexdigit() && !nibble.is_ascii_uppercase());
    nibble.to_string().repeat(64)
}

pub(super) fn semantic_identity(name: &str, dim: u32) -> EmbeddingIdentityBundleV1 {
    let mut bundle = identity(name, dim);
    bundle.space.kind = EmbeddingSpaceKindV1::Semantic;
    bundle.space.hash_control = None;
    bundle.space.artifact_manifest_fingerprint = test_digest('a');
    bundle.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
        role: "weights".to_owned(),
        sha256: test_digest('b'),
        size: 1,
    }];
    bundle.producer.space_fingerprint = bundle.space.fingerprint();
    bundle.validate().expect("semantic test identity validates");
    bundle
}

pub(super) fn test_scope(generation: char, corpus: char) -> CoverageScopeV1 {
    CoverageScopeV1::new(test_digest(generation), test_digest(corpus))
        .expect("test scope validates")
}

pub(super) fn members(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

pub(super) fn verified_documents(
    tier: CoverageTierV1,
    bundle: &EmbeddingIdentityBundleV1,
    live: &[&str],
    indexed: &[&str],
) -> TierCoverageV1 {
    TierCoverageV1::verified(
        tier,
        &bundle.space,
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(live),
        test_scope('1', '2'),
        members(indexed),
        None,
    )
    .expect("same-scope document coverage validates")
}

pub(super) fn verified_documents_context(
    tier: CoverageTierV1,
    bundle: &EmbeddingIdentityBundleV1,
    live: &[&str],
    indexed: &[&str],
) -> TrustedTierCoverageContextV1 {
    TrustedTierCoverageContextV1::verified(
        tier,
        bundle.space.clone(),
        CoverageBasisV1::CanonicalDocuments,
        test_scope('1', '2'),
        members(live),
        test_scope('1', '2'),
        members(indexed),
        None,
    )
    .expect("same-scope document context validates")
}

pub(super) fn bound(bundle: EmbeddingIdentityBundleV1) -> BoundQueryEmbedding {
    BoundQueryEmbedding::new(vec![0.25; bundle.space.dimension as usize], bundle)
        .expect("test query binding validates")
}
