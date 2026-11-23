use super::COMPLETE_COVERAGE_PPM;
use super::authorization::{ModelAcquisitionTarget, validate_authorization_binding};
use super::contract::{
    RecoveryAction, RecoveryPlan, Retryability, SemanticResponseContract, TrustedRecoveryContext,
};
use super::types::{
    InteractionPolicy, NetworkPolicy, RecoveryContractError, RecoveryPlanSchemaVersion,
    RecoveryRequest, RequestMode, SemanticReadiness,
};
use crate::{config::ZeroSignalReason, traits::ModelTier, types::RetrievalTopology};

/// Compute the truthful plan for a readiness state under a request and
/// policy. Pure and deterministic: identical inputs yield identical plans.
///
/// # Errors
///
/// Rejects ambiguous request topology, invalid partial coverage, hash
/// requests without hash-control readiness, and malformed acquisition
/// targets before returning executable recovery metadata. Trusted evaluation
/// time is mandatory and never read from the serialized plan or ambient
/// process state.
pub fn plan(trusted: TrustedRecoveryContext<'_>) -> Result<RecoveryPlan, RecoveryContractError> {
    let TrustedRecoveryContext {
        state,
        request,
        policy,
        acquisition_target,
        evaluation_time_unix_seconds,
    } = trusted;
    let state = state.clone();
    let policy = policy.clone();
    let request = request.validate()?;
    if let Some(authorization) = &policy.acquisition_authorization {
        authorization.validate()?;
    }
    validate_hash_mode_state(request.mode, &state)?;
    validate_readiness(&state)?;
    validate_acquisition_tier(&state, request.requested_topology, acquisition_target)?;

    let action = action_for(
        &state,
        request.requested_topology,
        policy.network,
        acquisition_target,
    )?;
    validate_authorization_binding(
        action
            .as_ref()
            .and_then(RecoveryAction::required_authorization),
        policy.acquisition_authorization.as_ref(),
        evaluation_time_unix_seconds,
    )?;
    let (action, retryability) = match action {
        None => (
            None,
            if state.semantic_available()
                || matches!(
                    (&state, request.mode),
                    (SemanticReadiness::HashControl, RequestMode::HashControl)
                )
            {
                Retryability::NotNeeded
            } else {
                Retryability::AfterRequestChange
            },
        ),
        Some(mut action) => {
            let network_blocked =
                action.network_required && matches!(policy.network, NetworkPolicy::Offline);
            if network_blocked {
                push_prerequisite(&mut action, "recovery.policy.allow_network");
            }
            let authorization_satisfied =
                action
                    .required_authorization
                    .as_ref()
                    .is_some_and(|required| {
                        // ubs:ignore — this public policy contract is not an authenticator.
                        policy.acquisition_authorization.as_ref() == Some(required)
                    });
            let binding_missing = matches!(
                action.code.as_str(),
                "recovery.action.acquire_model" | "recovery.action.reacquire_model"
            ) && action.required_authorization.is_none();
            if binding_missing {
                push_prerequisite(&mut action, "recovery.policy.bind_model");
            }
            let consent_blocked = action.consent_required
                && matches!(policy.interaction, InteractionPolicy::NonInteractive)
                && !authorization_satisfied;
            if consent_blocked {
                push_prerequisite(&mut action, "recovery.policy.grant_consent");
            }
            let capability_blocked = action
                .prerequisites
                .iter()
                .any(|code| code.starts_with("recovery.capability."));
            let retryability = if capability_blocked {
                Retryability::BlockedByCapability
            } else if network_blocked
                || binding_missing
                || consent_blocked
                || !action.prerequisites.is_empty()
            {
                Retryability::BlockedByPolicy
            } else {
                Retryability::AfterAction
            };
            (Some(action), retryability)
        }
    };

    let semantic_available = state.semantic_available();
    let response_contract = response_contract_for(&state, request)?;
    let state_code = state.state_code().to_owned();
    let provenance = state.provenance();

    Ok(RecoveryPlan {
        schema_version: RecoveryPlanSchemaVersion::V4,
        state,
        state_code,
        provenance,
        mode: request.mode,
        requested_topology: request.requested_topology,
        policy,
        semantic_available,
        retryability,
        action,
        response_contract,
    })
}

fn validate_hash_mode_state(
    mode: RequestMode,
    state: &SemanticReadiness,
) -> Result<(), RecoveryContractError> {
    let hash_request = matches!(mode, RequestMode::HashControl);
    let hash_state = matches!(state, SemanticReadiness::HashControl);
    if hash_request == hash_state {
        Ok(())
    } else {
        Err(RecoveryContractError::HashControlModeReadinessMismatch)
    }
}

fn validate_readiness(state: &SemanticReadiness) -> Result<(), RecoveryContractError> {
    if let SemanticReadiness::PartialQualityCoverage { coverage_ppm, .. } = state
        && !(1..COMPLETE_COVERAGE_PPM).contains(coverage_ppm)
    {
        return Err(RecoveryContractError::InvalidCoverage {
            topology: RetrievalTopology::PartialQuality {
                coverage_ppm: *coverage_ppm,
            },
            coverage_ppm: *coverage_ppm,
        });
    }
    Ok(())
}

fn validate_acquisition_tier(
    state: &SemanticReadiness,
    requested_topology: RetrievalTopology,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<(), RecoveryContractError> {
    let readiness_tier = match state {
        SemanticReadiness::ModelMissing { tier } | SemanticReadiness::ModelUnloadable { tier } => {
            *tier
        }
        _ => return Ok(()),
    };
    let topology_matches = match requested_topology {
        // ubs:ignore — model tiers are public recovery-routing facts, not secrets.
        RetrievalTopology::FastOnly => readiness_tier == ModelTier::Fast,
        // ubs:ignore — model tiers are public recovery-routing facts, not secrets.
        RetrievalTopology::QualityOnly => readiness_tier == ModelTier::Quality,
        RetrievalTopology::FullProgressive => true,
        RetrievalTopology::LexicalOnly
        | RetrievalTopology::PartialQuality { .. }
        | RetrievalTopology::HashControl => false,
    };
    if !topology_matches {
        return Err(RecoveryContractError::UnavailableTierTopologyMismatch {
            tier: readiness_tier,
            requested_topology,
        });
    }
    if let Some(target) = acquisition_target
        // ubs:ignore — model tiers are public recovery-routing facts, not secrets.
        && target.model_tier != readiness_tier
    {
        return Err(RecoveryContractError::AcquisitionTargetTierMismatch {
            readiness_tier,
            target_tier: target.model_tier,
        });
    }
    Ok(())
}

fn response_contract_for(
    state: &SemanticReadiness,
    request: RecoveryRequest,
) -> Result<Option<SemanticResponseContract>, RecoveryContractError> {
    if matches!(request.mode, RequestMode::HashControl) {
        return SemanticResponseContract::new(
            RetrievalTopology::HashControl,
            RetrievalTopology::HashControl,
            0,
            0,
            None,
        )
        .map(Some);
    }

    if state.semantic_available() {
        let (realized_topology, coverage_ppm) = match (state, request.requested_topology) {
            (
                SemanticReadiness::PartialQualityCoverage { coverage_ppm, .. },
                RetrievalTopology::QualityOnly | RetrievalTopology::FullProgressive,
            ) => (
                RetrievalTopology::PartialQuality {
                    coverage_ppm: *coverage_ppm,
                },
                *coverage_ppm,
            ),
            _ => (request.requested_topology, COMPLETE_COVERAGE_PPM),
        };
        return SemanticResponseContract::new(
            request.requested_topology,
            realized_topology,
            coverage_ppm,
            0,
            None,
        )
        .map(Some);
    }

    if matches!(request.mode, RequestMode::Hybrid) {
        return SemanticResponseContract::new(
            request.requested_topology,
            RetrievalTopology::LexicalOnly,
            0,
            0,
            Some(state.state_code().to_owned()),
        )
        .map(Some);
    }

    Ok(None)
}

fn push_prerequisite(action: &mut RecoveryAction, code: &str) {
    // ubs:ignore — prerequisite codes are public recovery facts, not authenticators.
    if !action.prerequisites.iter().any(|existing| existing == code) {
        action.prerequisites.push(code.to_owned());
    }
}

// The parameters mirror RecoveryAction's schema-mandated boolean fields
// one-to-one; an intermediate flags type would only restate the struct.
#[allow(clippy::fn_params_excessive_bools)]
fn simple_action(
    code: &str,
    explanation: &str,
    argv: &[&str],
    network_required: bool,
    consent_required: bool,
    preserves_old_data: bool,
    potentially_destructive: bool,
    expected_postcondition: &str,
) -> RecoveryAction {
    RecoveryAction {
        code: code.to_owned(),
        explanation: explanation.to_owned(),
        argv: argv.iter().map(|&a| a.to_owned()).collect(),
        network_required,
        consent_required,
        preserves_old_data,
        potentially_destructive,
        prerequisites: Vec::new(),
        expected_postcondition: expected_postcondition.to_owned(),
        required_authorization: None,
    }
}

fn block_unbound_semantic_index_action(
    mut action: RecoveryAction,
    requested_topology: RetrievalTopology,
    quality_capability: &str,
) -> RecoveryAction {
    action.argv.clear();
    let capability = if matches!(requested_topology, RetrievalTopology::FastOnly) {
        "recovery.capability.execute_bound_semantic_index"
    } else {
        quality_capability
    };
    push_prerequisite(&mut action, capability);
    action
}

/// The exhaustive state → action table. No wildcard arm: adding a
/// readiness state without planning for it is a compile error.
fn action_for(
    state: &SemanticReadiness,
    requested_topology: RetrievalTopology,
    network: NetworkPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<Option<RecoveryAction>, RecoveryContractError> {
    match state {
        SemanticReadiness::Ready { .. } | SemanticReadiness::HashControl => Ok(None),
        SemanticReadiness::ModelMissing { .. } => {
            model_acquisition_action(false, network, acquisition_target).map(Some)
        }
        SemanticReadiness::ModelUnloadable { .. } => {
            model_acquisition_action(true, network, acquisition_target).map(Some)
        }
        SemanticReadiness::IndexAbsent => Ok(Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.build_index",
                "A verified model is present but no vector index exists. Current generic fsfs \
                 indexing can still fall through to a non-semantic producer, so recovery needs \
                 an executor bound to the attested semantic space and requested topology.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.index_built",
            ),
            requested_topology,
            "recovery.capability.build_quality_tier",
        ))),
        SemanticReadiness::IdentityMismatch => Ok(Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.reindex_full",
                "The index was built in a different embedding space than the configured model. \
                 Rebuild only if the identity change is intentional, through an executor bound \
                 to the attested semantic space and requested topology; the existing index is \
                 replaced.",
                &[],
                false,
                true,
                false,
                true,
                "recovery.post.index_rebuilt",
            ),
            requested_topology,
            "recovery.capability.reindex_quality_tier",
        ))),
        SemanticReadiness::DaemonMismatch => {
            let mut action = simple_action(
                "recovery.action.restart_daemon",
                "The embedding daemon serves a different space than the local configuration, \
                 but fsfs currently has no parser-executable restart operation. A bound daemon \
                 lifecycle capability must land before recovery can run.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.daemon_aligned",
            );
            push_prerequisite(&mut action, "recovery.capability.restart_daemon");
            Ok(Some(action))
        }
        SemanticReadiness::IndexEmpty(reason) => Ok(plan_for_empty(*reason, requested_topology)),
        SemanticReadiness::ManifestUnsafe => Ok(Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.reindex_full",
                "The manifest failed safety validation and its artifacts must not be trusted; \
                 rebuild from source content only through an executor bound to the attested \
                 semantic space and requested topology.",
                &[],
                false,
                true,
                false,
                true,
                "recovery.post.index_rebuilt",
            ),
            requested_topology,
            "recovery.capability.reindex_quality_tier",
        ))),
        SemanticReadiness::AnnStale => {
            let mut action = simple_action(
                "recovery.action.rebuild_ann",
                "The ANN sidecar belongs to an older index generation, but generic indexing \
                 does not bind the exact ANN generation to rebuild. Exact search remains \
                 correct while the dedicated generation-aware capability is unavailable.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.ann_rebuilt",
            );
            push_prerequisite(&mut action, "recovery.capability.rebuild_ann_generation");
            Ok(Some(action))
        }
        SemanticReadiness::GenerationIncomplete => Ok(Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.resume_index",
                "An index generation was interrupted before publication. Completing it requires \
                 an executor bound to the attested semantic space, topology, and generation; \
                 published data remains untouched.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.generation_completed",
            ),
            requested_topology,
            "recovery.capability.resume_quality_generation",
        ))),
        SemanticReadiness::PartialQualityCoverage { .. }
            if matches!(requested_topology, RetrievalTopology::FastOnly) =>
        {
            Ok(None)
        }
        SemanticReadiness::PartialQualityCoverage { .. } => {
            let mut action = simple_action(
                "recovery.action.backfill_quality",
                "Some records lack quality-tier embeddings, but generic indexing does not \
                 express a quality-only backfill. Search remains available while the dedicated \
                 tier-aware capability is unavailable.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.coverage_completed",
            );
            push_prerequisite(&mut action, "recovery.capability.backfill_quality_tier");
            Ok(Some(action))
        }
        SemanticReadiness::RemoteUnverified => {
            let mut action = simple_action(
                "recovery.action.provide_attestation",
                "Explicit remote intent cannot be admitted because its producer space is not \
                 attested. Supply a pinned producer attester through the caller-owned \
                 configuration boundary; there is no safe generic command that can invent \
                 this trust root.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.remote_attested",
            );
            push_prerequisite(&mut action, "recovery.policy.provide_attestation");
            Ok(Some(action))
        }
    }
}

fn model_acquisition_action(
    reacquire: bool,
    network: NetworkPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<RecoveryAction, RecoveryContractError> {
    let code = if reacquire {
        "recovery.action.reacquire_model"
    } else {
        "recovery.action.acquire_model"
    };
    let required_authorization = acquisition_target
        .map(|target| target.authorization_for(network))
        .transpose()?;
    let explanation = match (network, reacquire) {
        (NetworkPolicy::Allowed, false) => {
            "The configured semantic model must be acquired under the exact frozen \
             authorization, but current fsfs download syntax cannot bind every authorized \
             identity, byte-budget, destination, corpus, and reindex field to execution. This \
             action is deliberately non-executable until a bound executor exists."
        }
        (NetworkPolicy::Allowed, true) => {
            "The cached model failed verification or load self-test, but current fsfs download \
             syntax cannot bind the complete frozen re-acquisition authorization to execution. \
             This action is deliberately non-executable until a bound executor exists; index \
             data is untouched."
        }
        (NetworkPolicy::Offline, false) => {
            "A complete local-bundle import is required, but the current fsfs parser has no \
             offline importer. This action is deliberately non-executable until that capability \
             exists; no network access is permitted."
        }
        (NetworkPolicy::Offline, true) => {
            "Replacing the unloadable cache from a complete local bundle requires an importer \
             that the current fsfs parser does not provide. This action is deliberately \
             non-executable until that capability exists; index data is untouched."
        }
    };
    let mut action = simple_action(
        code,
        explanation,
        &[],
        matches!(network, NetworkPolicy::Allowed),
        true,
        !reacquire,
        reacquire,
        "recovery.post.model_acquired_unverified",
    );
    action.required_authorization = required_authorization;
    match network {
        NetworkPolicy::Allowed => push_prerequisite(
            &mut action,
            "recovery.capability.execute_bound_model_acquisition",
        ),
        NetworkPolicy::Offline => {
            push_prerequisite(&mut action, "recovery.capability.import_model_bundle");
        }
    }
    Ok(action)
}

/// Empty-index planning follows the zero-signal classification: benign
/// state emptiness wants ingestion, availability failures want rebuilds,
/// the ANN anomaly wants a sidecar rebuild, and request-scoped reasons
/// need no system action at all.
fn plan_for_empty(
    reason: ZeroSignalReason,
    requested_topology: RetrievalTopology,
) -> Option<RecoveryAction> {
    match reason {
        ZeroSignalReason::NewlyCreatedEmpty
        | ZeroSignalReason::AllTombstoned
        | ZeroSignalReason::WalOnlyNoLiveRecords => Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.ingest_content",
                "The index holds no live records. Populate it only through an executor bound to \
                 the attested semantic space and requested topology.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.index_populated",
            ),
            requested_topology,
            "recovery.capability.ingest_quality_tier",
        )),
        ZeroSignalReason::NoUsableVectors => Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.reindex_full",
                "Live records exist but none of their stored vectors is usable (zero-norm or \
                 corrupt); rebuild only through an executor bound to the attested semantic space \
                 and requested topology.",
                &[],
                false,
                true,
                false,
                true,
                "recovery.post.index_rebuilt",
            ),
            requested_topology,
            "recovery.capability.reindex_quality_tier",
        )),
        ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors => {
            let mut action = simple_action(
                "recovery.action.rebuild_ann",
                "The ANN graph returned no candidates although usable live vectors exist, but \
                 generic indexing does not bind the exact ANN generation to rebuild. The vector \
                 index remains untouched.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.ann_rebuilt",
            );
            push_prerequisite(&mut action, "recovery.capability.rebuild_ann_generation");
            Some(action)
        }
        ZeroSignalReason::CallerRequestedZeroK
        | ZeroSignalReason::FilterEliminatedAll
        | ZeroSignalReason::NonFiniteQuery
        | ZeroSignalReason::ZeroNormQuery => None,
    }
}
