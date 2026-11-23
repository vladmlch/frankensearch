pub(super) use super::*;
pub(super) use crate::{
    config::ZeroSignalReason,
    generation::{
        EMBEDDING_SPACE_IDENTITY_SCHEMA_V1, EmbeddingArtifactIdentityV1, EmbeddingSpaceIdentityV1,
        EmbeddingSpaceKindV1,
    },
    traits::ModelTier,
    types::RetrievalTopology,
};

#[path = "tests_authorization.rs"]
mod authorization;
#[path = "tests_planning.rs"]
mod planning;
#[path = "tests_wire.rs"]
mod wire;

pub(super) const TEST_NOW_UNIX_SECONDS: u64 = 2_000_000_000;
pub(super) const TEST_AUTHORIZATION_ISSUED_AT_UNIX_SECONDS: u64 = TEST_NOW_UNIX_SECONDS - 60;
pub(super) const TEST_AUTHORIZATION_EXPIRES_AT_UNIX_SECONDS: u64 = TEST_NOW_UNIX_SECONDS + 600;
pub(super) const TEST_AUTHORIZATION_NONCE: &str = "0123456789abcdef0123456789abcdef";

pub(super) fn semantic_space() -> EmbeddingSpaceIdentityV1 {
    EmbeddingSpaceIdentityV1 {
        schema_version: EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
        logical_model_id: "fixture-semantic-model".to_owned(),
        immutable_revision: "revision-0123456789abcdef".to_owned(),
        kind: EmbeddingSpaceKindV1::Semantic,
        artifact_manifest_fingerprint: "a".repeat(64),
        artifacts: vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "c".repeat(64),
            size: 42_000_000,
        }],
        tokenizer_fingerprint: "d".repeat(64),
        vocabulary_fingerprint: "e".repeat(64),
        model_config_fingerprint: "f".repeat(64),
        model_preprocessing: "nfc-v1".to_owned(),
        sequence_policy: "truncate-256-v1".to_owned(),
        query_instruction: String::new(),
        document_instruction: String::new(),
        pooling: "mean-v1".to_owned(),
        output_normalization: "l2-v1".to_owned(),
        dimension: 384,
        input_contract_fingerprint: "1".repeat(64),
        hash_control: None,
        projection: None,
    }
}

pub(super) fn target_for(model_tier: ModelTier) -> ModelAcquisitionTarget {
    ModelAcquisitionTarget {
        model_id: "fixture-semantic-model".to_owned(),
        model_tier,
        embedding_space: semantic_space(),
        manifest_fingerprint: "a".repeat(64),
        upstream_revision: "revision-0123456789abcdef".to_owned(),
        license_spdx: "Apache-2.0".to_owned(),
        network_source_hosts: vec!["models.example.test".to_owned()],
        byte_budget: 42_000_000,
        destination_class: ModelDestinationClass::ManagedCache,
        destination_fingerprint: "b".repeat(64),
        document_count: 12_345,
        estimated_reindex_duration_ms: 98_765,
        issued_at_unix_seconds: TEST_AUTHORIZATION_ISSUED_AT_UNIX_SECONDS,
        expires_at_unix_seconds: TEST_AUTHORIZATION_EXPIRES_AT_UNIX_SECONDS,
        nonce: TEST_AUTHORIZATION_NONCE.to_owned(),
    }
}

pub(super) fn target() -> ModelAcquisitionTarget {
    target_for(ModelTier::Quality)
}

pub(super) const fn missing(tier: ModelTier) -> SemanticReadiness {
    SemanticReadiness::ModelMissing { tier }
}

pub(super) const fn unloadable(tier: ModelTier) -> SemanticReadiness {
    SemanticReadiness::ModelUnloadable { tier }
}

pub(super) const fn explicit(requested_topology: RetrievalTopology) -> RecoveryRequest {
    RecoveryRequest {
        mode: RequestMode::ExplicitSemantic,
        requested_topology,
    }
}

pub(super) const fn hybrid(requested_topology: RetrievalTopology) -> RecoveryRequest {
    RecoveryRequest {
        mode: RequestMode::Hybrid,
        requested_topology,
    }
}

pub(super) const fn hash_control() -> RecoveryRequest {
    RecoveryRequest {
        mode: RequestMode::HashControl,
        requested_topology: RetrievalTopology::HashControl,
    }
}

pub(super) fn semantic_requests() -> [RecoveryRequest; 6] {
    [
        explicit(RetrievalTopology::FastOnly),
        explicit(RetrievalTopology::QualityOnly),
        explicit(RetrievalTopology::FullProgressive),
        hybrid(RetrievalTopology::FastOnly),
        hybrid(RetrievalTopology::QualityOnly),
        hybrid(RetrievalTopology::FullProgressive),
    ]
}

pub(super) fn representative_states() -> Vec<SemanticReadiness> {
    vec![
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Remote,
        },
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Daemon,
        },
        missing(ModelTier::Quality),
        unloadable(ModelTier::Quality),
        SemanticReadiness::IndexAbsent,
        SemanticReadiness::IdentityMismatch,
        SemanticReadiness::DaemonMismatch,
        SemanticReadiness::IndexEmpty(ZeroSignalReason::NewlyCreatedEmpty),
        SemanticReadiness::IndexEmpty(ZeroSignalReason::AllTombstoned),
        SemanticReadiness::IndexEmpty(ZeroSignalReason::WalOnlyNoLiveRecords),
        SemanticReadiness::IndexEmpty(ZeroSignalReason::CallerRequestedZeroK),
        SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors),
        SemanticReadiness::IndexEmpty(ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors),
        SemanticReadiness::ManifestUnsafe,
        SemanticReadiness::AnnStale,
        SemanticReadiness::GenerationIncomplete,
        SemanticReadiness::PartialQualityCoverage {
            provenance: VerifiedSemanticProvenance::Local,
            coverage_ppm: 750_000,
        },
        SemanticReadiness::PartialQualityCoverage {
            provenance: VerifiedSemanticProvenance::Remote,
            coverage_ppm: 500_000,
        },
        SemanticReadiness::RemoteUnverified,
        SemanticReadiness::HashControl,
    ]
}

pub(super) fn unready_states() -> Vec<SemanticReadiness> {
    representative_states()
        .into_iter()
        .filter(|state| {
            !state.semantic_available() && !matches!(state, SemanticReadiness::HashControl)
        })
        .collect()
}

pub(super) fn all_policies() -> Vec<RecoveryPolicy> {
    let mut out = Vec::new();
    for interaction in [
        InteractionPolicy::Interactive,
        InteractionPolicy::NonInteractive,
    ] {
        for network in [NetworkPolicy::Allowed, NetworkPolicy::Offline] {
            out.push(RecoveryPolicy {
                interaction,
                network,
                acquisition_authorization: None,
            });
        }
    }
    out
}

pub(super) fn permissive() -> RecoveryPolicy {
    RecoveryPolicy {
        interaction: InteractionPolicy::Interactive,
        network: NetworkPolicy::Allowed,
        acquisition_authorization: None,
    }
}

// These test builders deliberately own state and policy so call sites can
// pass temporary fixtures without introducing local bindings solely for
// borrow lifetimes.
#[allow(clippy::needless_pass_by_value)]
pub(super) fn plan(
    state: SemanticReadiness,
    request: RecoveryRequest,
    policy: RecoveryPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<RecoveryPlan, RecoveryContractError> {
    super::plan(TrustedRecoveryContext::new(
        &state,
        request,
        &policy,
        acquisition_target,
        TEST_NOW_UNIX_SECONDS,
    ))
}

#[allow(clippy::needless_pass_by_value)]
pub(super) fn planned(
    state: SemanticReadiness,
    request: RecoveryRequest,
    policy: RecoveryPolicy,
) -> RecoveryPlan {
    plan(state, request, policy, Some(&target())).expect("valid recovery plan")
}

pub(super) fn tier_for_request(request: RecoveryRequest) -> ModelTier {
    match request.requested_topology {
        RetrievalTopology::FastOnly => ModelTier::Fast,
        RetrievalTopology::QualityOnly | RetrievalTopology::FullProgressive => ModelTier::Quality,
        other @ (RetrievalTopology::LexicalOnly
        | RetrievalTopology::PartialQuality { .. }
        | RetrievalTopology::HashControl) => {
            assert!(
                matches!(
                    other,
                    RetrievalTopology::FastOnly
                        | RetrievalTopology::QualityOnly
                        | RetrievalTopology::FullProgressive
                ),
                "semantic recovery helper received a non-semantic topology: {other:?}"
            );
            ModelTier::Quality
        }
    }
}

pub(super) fn state_for_request(
    state: &SemanticReadiness,
    request: RecoveryRequest,
) -> SemanticReadiness {
    match state {
        SemanticReadiness::ModelMissing { .. } => missing(tier_for_request(request)),
        SemanticReadiness::ModelUnloadable { .. } => unloadable(tier_for_request(request)),
        _ => state.clone(),
    }
}

pub(super) fn decode_and_validate(
    value: serde_json::Value,
    state: &SemanticReadiness,
    request: RecoveryRequest,
    policy: &RecoveryPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<RecoveryPlan, String> {
    let untrusted: UntrustedRecoveryPlan =
        serde_json::from_value(value).map_err(|error| error.to_string())?;
    untrusted
        .validate_against(TrustedRecoveryContext::new(
            state,
            request,
            policy,
            acquisition_target,
            TEST_NOW_UNIX_SECONDS,
        ))
        .map_err(|error| error.to_string())
}
