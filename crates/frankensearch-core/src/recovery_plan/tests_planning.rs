use super::*;
use crate::decision_plane::ReasonCode;
use crate::recovery_plan::contract::SemanticResponseContractWire;

#[test]
fn every_stable_code_is_valid_and_unique() {
    let mut codes = Vec::new();
    let states = representative_states();
    for state in &states {
        codes.push(state.state_code().to_owned());
        if !matches!(state, SemanticReadiness::HashControl) {
            for request in semantic_requests() {
                for policy in all_policies() {
                    let tier = tier_for_request(request);
                    let state = state_for_request(state, request);
                    let plan = plan(state, request, policy, Some(&target_for(tier)))
                        .expect("representative plan");
                    if let Some(action) = plan.action {
                        codes.push(action.code.clone());
                        codes.push(action.expected_postcondition.clone());
                        codes.extend(action.prerequisites);
                    }
                }
            }
        }
    }
    let hash_plan = planned(SemanticReadiness::HashControl, hash_control(), permissive());
    codes.push(hash_plan.state_code);
    let unbound = plan(
        missing(ModelTier::Fast),
        explicit(RetrievalTopology::FastOnly),
        permissive(),
        None,
    )
    .expect("unbound acquisition still returns a blocked plan");
    codes.extend(unbound.action.expect("model action").prerequisites);
    codes.push("recovery.policy.allow_network".to_owned());
    for code in &codes {
        assert!(
            ReasonCode::new(code.as_str()).is_valid(),
            "invalid stable code format: {code}"
        );
    }
    // Distinct states never share a code with distinct actions.
    let state_codes: std::collections::HashSet<_> = states.iter().map(|s| s.state_code()).collect();
    assert_eq!(state_codes.len(), 13, "one code per state variant");

    let emitted: std::collections::HashSet<_> = codes.iter().map(String::as_str).collect();
    let v1_codes = [
        "recovery.state.ready",
        "recovery.state.model_missing",
        "recovery.state.model_unloadable",
        "recovery.state.index_absent",
        "recovery.state.identity_mismatch",
        "recovery.state.daemon_mismatch",
        "recovery.state.index_empty",
        "recovery.state.manifest_unsafe",
        "recovery.state.ann_stale",
        "recovery.state.generation_incomplete",
        "recovery.state.partial_quality_coverage",
        "recovery.action.acquire_model",
        "recovery.action.reacquire_model",
        "recovery.action.build_index",
        "recovery.action.reindex_full",
        "recovery.action.restart_daemon",
        "recovery.action.ingest_content",
        "recovery.action.rebuild_ann",
        "recovery.action.resume_index",
        "recovery.action.backfill_quality",
        "recovery.post.model_acquired_unverified",
        "recovery.post.index_built",
        "recovery.post.index_rebuilt",
        "recovery.post.daemon_aligned",
        "recovery.post.index_populated",
        "recovery.post.ann_rebuilt",
        "recovery.post.generation_completed",
        "recovery.post.coverage_completed",
        "recovery.policy.allow_network",
        "recovery.policy.grant_consent",
    ];
    for old_code in v1_codes {
        assert!(
            emitted.contains(old_code),
            "v1 code disappeared: {old_code}"
        );
    }
    for appended in [
        "recovery.state.remote_unverified",
        "recovery.state.hash_control",
        "recovery.action.provide_attestation",
        "recovery.post.remote_attested",
        "recovery.policy.provide_attestation",
        "recovery.policy.bind_model",
        "recovery.capability.backfill_quality_tier",
        "recovery.capability.build_quality_tier",
        "recovery.capability.execute_bound_model_acquisition",
        "recovery.capability.execute_bound_semantic_index",
        "recovery.capability.ingest_quality_tier",
        "recovery.capability.import_model_bundle",
        "recovery.capability.rebuild_ann_generation",
        "recovery.capability.reindex_quality_tier",
        "recovery.capability.restart_daemon",
        "recovery.capability.resume_quality_generation",
    ] {
        assert!(
            emitted.contains(appended),
            "appended code absent: {appended}"
        );
    }
}

#[test]
fn explicit_semantic_fails_closed_for_every_unready_state() {
    for state in unready_states() {
        let plan = planned(
            state.clone(),
            explicit(RetrievalTopology::FullProgressive),
            permissive(),
        );
        assert!(!plan.semantic_available);
        assert!(
            plan.response_contract.is_none(),
            "explicit semantic never degrades to lexical: {state:?}"
        );
    }
}

#[test]
fn hybrid_degrades_with_metadata_exactly_when_unavailable() {
    for state in unready_states() {
        let plan = planned(
            state.clone(),
            hybrid(RetrievalTopology::FullProgressive),
            permissive(),
        );
        let response = plan
            .response_contract
            .expect("unavailable hybrid must carry response contract");
        assert_eq!(
            response.requested_topology(),
            RetrievalTopology::FullProgressive
        );
        assert_eq!(response.realized_topology(), RetrievalTopology::LexicalOnly);
        assert_eq!(response.coverage_ppm(), 0);
        assert_eq!(response.admitted_semantic_scores(), 0);
        assert_eq!(response.degradation_reason_code(), Some(state.state_code()));
    }
}

#[test]
fn acquisition_never_claims_readiness() {
    for state in [missing(ModelTier::Quality), unloadable(ModelTier::Quality)] {
        let reacquire = matches!(state, SemanticReadiness::ModelUnloadable { .. });
        let plan = planned(
            state,
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
        );
        assert_eq!(plan.retryability, Retryability::BlockedByCapability);
        let action = plan.action.expect("acquisition state has an action");
        assert_eq!(
            action.expected_postcondition,
            "recovery.post.model_acquired_unverified"
        );
        assert_ne!(action.expected_postcondition, "recovery.state.ready");
        assert!(action.network_required);
        assert!(action.consent_required);
        assert_eq!(action.preserves_old_data, !reacquire);
        assert_eq!(action.potentially_destructive, reacquire);
        assert_eq!(
            action.code,
            if reacquire {
                "recovery.action.reacquire_model"
            } else {
                "recovery.action.acquire_model"
            }
        );
        assert!(
            action.argv.is_empty(),
            "partial download syntax must not masquerade as exact bound authorization"
        );
        assert_eq!(
            action.prerequisites,
            ["recovery.capability.execute_bound_model_acquisition"]
        );
        let authorization = action
            .required_authorization
            .expect("acquisition binds exact authorization");
        assert_eq!(authorization.model_id, "fixture-semantic-model");
        assert_eq!(authorization.model_tier, ModelTier::Quality);
        assert_eq!(authorization.embedding_space, semantic_space());
        assert_eq!(authorization.manifest_fingerprint, "a".repeat(64));
        assert_eq!(authorization.upstream_revision, "revision-0123456789abcdef");
        assert_eq!(authorization.license_spdx, "Apache-2.0");
        assert_eq!(authorization.byte_budget, 42_000_000);
        assert_eq!(
            authorization.destination_class,
            ModelDestinationClass::ManagedCache
        );
        assert_eq!(authorization.destination_fingerprint, "b".repeat(64));
        assert_eq!(authorization.document_count, 12_345);
        assert_eq!(authorization.estimated_reindex_duration_ms, 98_765);
        assert_eq!(
            authorization.issued_at_unix_seconds,
            TEST_AUTHORIZATION_ISSUED_AT_UNIX_SECONDS
        );
        assert_eq!(
            authorization.expires_at_unix_seconds,
            TEST_AUTHORIZATION_EXPIRES_AT_UNIX_SECONDS
        );
        assert_eq!(authorization.nonce, TEST_AUTHORIZATION_NONCE);
        assert!(matches!(
            authorization.source,
            ModelAcquisitionSource::Network { .. }
        ));
    }
}

#[test]
fn offline_policy_reports_missing_import_capability_without_fictional_argv() {
    let policy = RecoveryPolicy {
        interaction: InteractionPolicy::Interactive,
        network: NetworkPolicy::Offline,
        acquisition_authorization: None,
    };
    for state in [missing(ModelTier::Quality), unloadable(ModelTier::Quality)] {
        let plan = planned(
            state,
            hybrid(RetrievalTopology::FullProgressive),
            policy.clone(),
        );
        assert_eq!(plan.retryability, Retryability::BlockedByCapability);
        let action = plan.action.expect("action still recommended");
        assert!(
            action.argv.is_empty(),
            "offline recovery must not publish argv that fsfs cannot parse"
        );
        assert!(!action.network_required);
        assert!(action.consent_required);
        assert_eq!(
            action.prerequisites,
            ["recovery.capability.import_model_bundle"]
        );
        assert!(matches!(
            action.required_authorization.expect("offline scope").source,
            ModelAcquisitionSource::LocalBundle
        ));
    }
}

#[test]
fn planned_semantic_mutations_never_publish_unbound_argv() {
    for state in unready_states() {
        let full_recovery = planned(
            state.clone(),
            explicit(RetrievalTopology::FullProgressive),
            permissive(),
        );
        if let Some(action) = full_recovery.action {
            assert!(
                action.argv.is_empty(),
                "quality/full recovery must not claim generic indexing realizes the \
                 requested topology for {state:?}: {:?}",
                action.argv
            );
        }

        let request = explicit(RetrievalTopology::FastOnly);
        let fast_state = state_for_request(&state, request);
        let fast_recovery = plan(
            fast_state,
            request,
            permissive(),
            Some(&target_for(ModelTier::Fast)),
        )
        .expect("valid fast-tier recovery plan");
        let Some(action) = fast_recovery.action else {
            continue;
        };
        assert!(
            action.argv.is_empty(),
            "fast-only recovery must not execute until semantic producer identity and \
             generation are bound for {state:?}: {:?}",
            action.argv
        );
    }

    for state in [
        SemanticReadiness::IndexAbsent,
        SemanticReadiness::IdentityMismatch,
        SemanticReadiness::ManifestUnsafe,
        SemanticReadiness::GenerationIncomplete,
        SemanticReadiness::IndexEmpty(ZeroSignalReason::NewlyCreatedEmpty),
        SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors),
    ] {
        let recovery = plan(
            state.clone(),
            explicit(RetrievalTopology::FastOnly),
            permissive(),
            Some(&target_for(ModelTier::Fast)),
        )
        .expect("fast-tier semantic mutation plan");
        let action = recovery.action.expect("semantic mutation action");
        assert!(action.argv.is_empty(), "{state:?}");
        assert!(
            action
                .prerequisites
                .contains(&"recovery.capability.execute_bound_semantic_index".to_owned()),
            "{state:?}"
        );
        assert_eq!(
            recovery.retryability,
            Retryability::BlockedByCapability,
            "{state:?}"
        );
    }

    let daemon = planned(
        SemanticReadiness::DaemonMismatch,
        explicit(RetrievalTopology::FullProgressive),
        permissive(),
    );
    let daemon_action = daemon.action.expect("daemon recovery");
    assert!(daemon_action.argv.is_empty());
    assert_eq!(
        daemon_action.prerequisites,
        ["recovery.capability.restart_daemon"]
    );
    assert_eq!(daemon.retryability, Retryability::BlockedByCapability);
}

#[test]
fn noninteractive_policy_blocks_consent_actions_with_prerequisite() {
    let policy = RecoveryPolicy {
        interaction: InteractionPolicy::NonInteractive,
        network: NetworkPolicy::Allowed,
        acquisition_authorization: None,
    };
    for state in [
        SemanticReadiness::IdentityMismatch,
        SemanticReadiness::ManifestUnsafe,
        SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors),
    ] {
        let plan = planned(
            state.clone(),
            explicit(RetrievalTopology::FullProgressive),
            policy.clone(),
        );
        assert_eq!(
            plan.retryability,
            Retryability::BlockedByCapability,
            "{state:?}"
        );
        let action = plan.action.expect("destructive states have actions");
        assert!(action.consent_required);
        assert!(action.potentially_destructive);
        assert!(!action.preserves_old_data);
        assert!(
            action
                .prerequisites
                .contains(&"recovery.policy.grant_consent".to_owned())
        );
        assert!(
            action
                .prerequisites
                .contains(&"recovery.capability.reindex_quality_tier".to_owned())
        );
    }
}

#[test]
fn request_scoped_emptiness_needs_no_system_action() {
    for reason in [
        ZeroSignalReason::CallerRequestedZeroK,
        ZeroSignalReason::FilterEliminatedAll,
        ZeroSignalReason::NonFiniteQuery,
        ZeroSignalReason::ZeroNormQuery,
    ] {
        let plan = planned(
            SemanticReadiness::IndexEmpty(reason),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
        );
        assert!(plan.action.is_none(), "{reason:?}");
        assert_eq!(plan.retryability, Retryability::AfterRequestChange);
        assert!(!plan.semantic_available);
    }
}

#[test]
fn partial_coverage_is_available_with_a_backfill_action() {
    let plan = planned(
        SemanticReadiness::PartialQualityCoverage {
            provenance: VerifiedSemanticProvenance::Remote,
            coverage_ppm: 625_000,
        },
        hybrid(RetrievalTopology::FullProgressive),
        permissive(),
    );
    assert!(plan.semantic_available);
    assert_eq!(plan.provenance, SemanticProvenance::VerifiedRemote);
    let response = plan.response_contract.as_ref().expect("response contract");
    assert_eq!(
        response.realized_topology(),
        RetrievalTopology::PartialQuality {
            coverage_ppm: 625_000
        }
    );
    assert_eq!(response.coverage_ppm(), 625_000);
    assert_eq!(response.admitted_semantic_scores(), 0);
    assert!(response.degradation_reason_code().is_none());
    let action = plan.action.expect("backfill recommended");
    assert_eq!(action.code, "recovery.action.backfill_quality");
    assert!(action.preserves_old_data);
    // The lane serves partial results, but no current command can express
    // the required quality-only backfill.
    assert_eq!(plan.retryability, Retryability::BlockedByCapability);
    assert_eq!(
        action.prerequisites,
        ["recovery.capability.backfill_quality_tier"]
    );
}

#[test]
fn partial_quality_coverage_only_backfills_topologies_that_request_quality() {
    let state = SemanticReadiness::PartialQualityCoverage {
        provenance: VerifiedSemanticProvenance::Remote,
        coverage_ppm: 625_000,
    };
    for mode in [RequestMode::ExplicitSemantic, RequestMode::Hybrid] {
        let fast = planned(
            state.clone(),
            RecoveryRequest {
                mode,
                requested_topology: RetrievalTopology::FastOnly,
            },
            permissive(),
        );
        assert!(fast.semantic_available);
        assert_eq!(fast.retryability, Retryability::NotNeeded);
        assert!(
            fast.action.is_none(),
            "complete fast-tier coverage needs no quality backfill"
        );
        let response = fast.response_contract.expect("fast response contract");
        assert_eq!(response.realized_topology(), RetrievalTopology::FastOnly);
        assert_eq!(response.coverage_ppm(), COMPLETE_COVERAGE_PPM);

        for requested_topology in [
            RetrievalTopology::QualityOnly,
            RetrievalTopology::FullProgressive,
        ] {
            let quality = planned(
                state.clone(),
                RecoveryRequest {
                    mode,
                    requested_topology,
                },
                permissive(),
            );
            assert!(quality.semantic_available);
            assert_eq!(quality.retryability, Retryability::BlockedByCapability);
            let action = quality.action.expect("quality request needs backfill");
            assert_eq!(action.code, "recovery.action.backfill_quality");
            assert!(action.argv.is_empty());
            assert_eq!(
                action.prerequisites,
                ["recovery.capability.backfill_quality_tier"]
            );
            let response = quality
                .response_contract
                .expect("quality response contract");
            assert_eq!(
                response.realized_topology(),
                RetrievalTopology::PartialQuality {
                    coverage_ppm: 625_000
                }
            );
            assert_eq!(response.coverage_ppm(), 625_000);
        }
    }
}

#[test]
fn request_topology_matrix_is_explicit_and_hash_isolated() {
    for request in semantic_requests().into_iter().chain([hash_control()]) {
        assert_eq!(request.validate(), Ok(request));
        let json = serde_json::to_string(&request).expect("serialize request");
        let decoded: RecoveryRequest =
            serde_json::from_str(&json).expect("deserialize valid request");
        assert_eq!(decoded, request);
    }

    let invalid = [
        explicit(RetrievalTopology::LexicalOnly),
        explicit(RetrievalTopology::HashControl),
        explicit(RetrievalTopology::PartialQuality {
            coverage_ppm: 500_000,
        }),
        hybrid(RetrievalTopology::LexicalOnly),
        hybrid(RetrievalTopology::HashControl),
        hybrid(RetrievalTopology::PartialQuality {
            coverage_ppm: 500_000,
        }),
        RecoveryRequest {
            mode: RequestMode::HashControl,
            requested_topology: RetrievalTopology::LexicalOnly,
        },
        RecoveryRequest {
            mode: RequestMode::HashControl,
            requested_topology: RetrievalTopology::FastOnly,
        },
        RecoveryRequest {
            mode: RequestMode::HashControl,
            requested_topology: RetrievalTopology::QualityOnly,
        },
        RecoveryRequest {
            mode: RequestMode::HashControl,
            requested_topology: RetrievalTopology::FullProgressive,
        },
        RecoveryRequest {
            mode: RequestMode::HashControl,
            requested_topology: RetrievalTopology::PartialQuality {
                coverage_ppm: 500_000,
            },
        },
    ];
    for request in invalid {
        assert!(matches!(
            request.validate(),
            Err(RecoveryContractError::InvalidRequestTopology { .. })
        ));
        let json = serde_json::to_string(&request).expect("serialize invalid request");
        assert!(
            serde_json::from_str::<RecoveryRequest>(&json).is_err(),
            "serde must not bypass request validation: {request:?}"
        );
    }

    for state in [
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
        missing(ModelTier::Quality),
        SemanticReadiness::RemoteUnverified,
    ] {
        assert_eq!(
            plan(state, hash_control(), permissive(), Some(&target())),
            Err(RecoveryContractError::HashControlModeReadinessMismatch)
        );
    }
    for request in semantic_requests() {
        assert_eq!(
            plan(
                SemanticReadiness::HashControl,
                request,
                permissive(),
                Some(&target()),
            ),
            Err(RecoveryContractError::HashControlModeReadinessMismatch)
        );
    }
}

#[test]
fn provenance_matrix_never_promotes_unverified_or_hash_producers() {
    for (producer, expected) in [
        (
            VerifiedSemanticProvenance::Local,
            SemanticProvenance::VerifiedLocal,
        ),
        (
            VerifiedSemanticProvenance::Remote,
            SemanticProvenance::VerifiedRemote,
        ),
        (
            VerifiedSemanticProvenance::Daemon,
            SemanticProvenance::VerifiedDaemon,
        ),
    ] {
        let plan = planned(
            SemanticReadiness::Ready {
                provenance: producer,
            },
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
        );
        assert!(plan.semantic_available);
        assert_eq!(plan.provenance, expected);
        assert_eq!(
            plan.response_contract
                .expect("verified producer may respond")
                .realized_topology(),
            RetrievalTopology::QualityOnly
        );
    }

    let remote = planned(
        SemanticReadiness::RemoteUnverified,
        explicit(RetrievalTopology::QualityOnly),
        permissive(),
    );
    assert_eq!(remote.provenance, SemanticProvenance::UnverifiedRemote);
    assert!(!remote.semantic_available);
    assert!(remote.response_contract.is_none());
    let action = remote.action.expect("attestation recovery");
    assert_eq!(action.code, "recovery.action.provide_attestation");
    assert!(action.argv.is_empty());
    assert_eq!(
        action.prerequisites,
        ["recovery.policy.provide_attestation"]
    );

    let hash = planned(SemanticReadiness::HashControl, hash_control(), permissive());
    assert_eq!(hash.provenance, SemanticProvenance::HashControl);
    assert!(!hash.semantic_available);
    assert_eq!(
        hash.response_contract
            .expect("explicit hash control response")
            .realized_topology(),
        RetrievalTopology::HashControl
    );
}

#[test]
fn response_contract_accepts_only_truthful_topology_and_coverage_pairs() {
    for (requested, realized, coverage) in [
        (
            RetrievalTopology::FastOnly,
            RetrievalTopology::FastOnly,
            COMPLETE_COVERAGE_PPM,
        ),
        (
            RetrievalTopology::QualityOnly,
            RetrievalTopology::QualityOnly,
            COMPLETE_COVERAGE_PPM,
        ),
        (
            RetrievalTopology::FullProgressive,
            RetrievalTopology::FullProgressive,
            COMPLETE_COVERAGE_PPM,
        ),
        (
            RetrievalTopology::FullProgressive,
            RetrievalTopology::FastOnly,
            COMPLETE_COVERAGE_PPM,
        ),
        (
            RetrievalTopology::FullProgressive,
            RetrievalTopology::QualityOnly,
            COMPLETE_COVERAGE_PPM,
        ),
        (
            RetrievalTopology::QualityOnly,
            RetrievalTopology::PartialQuality {
                coverage_ppm: 250_000,
            },
            250_000,
        ),
        (
            RetrievalTopology::FullProgressive,
            RetrievalTopology::PartialQuality {
                coverage_ppm: 750_000,
            },
            750_000,
        ),
    ] {
        let contract = SemanticResponseContract::new(requested, realized, coverage, 7, None)
            .expect("truthful semantic response");
        assert_eq!(contract.admitted_semantic_scores(), 7);
        let json = serde_json::to_string(&contract).expect("serialize response");
        assert!(
            serde_json::from_str::<SemanticResponseContractWire>(&json).is_ok(),
            "wire shape remains decodable only into the private raw type"
        );
    }

    for requested in [
        RetrievalTopology::FastOnly,
        RetrievalTopology::QualityOnly,
        RetrievalTopology::FullProgressive,
    ] {
        SemanticResponseContract::new(
            requested,
            RetrievalTopology::LexicalOnly,
            0,
            0,
            Some("recovery.state.model_missing".to_owned()),
        )
        .expect("typed lexical degradation");
    }
    SemanticResponseContract::new(
        RetrievalTopology::HashControl,
        RetrievalTopology::HashControl,
        0,
        0,
        None,
    )
    .expect("explicit hash control");
}

#[test]
fn response_contract_rejects_silent_or_impossible_contribution_claims() {
    assert_eq!(
        SemanticResponseContract::new(
            RetrievalTopology::LexicalOnly,
            RetrievalTopology::LexicalOnly,
            0,
            0,
            Some("recovery.state.model_missing".to_owned()),
        ),
        Err(RecoveryContractError::UnexpectedDegradationReason)
    );
    SemanticResponseContract::new(
        RetrievalTopology::LexicalOnly,
        RetrievalTopology::LexicalOnly,
        0,
        0,
        None,
    )
    .expect("an explicitly lexical request is not a degradation");
    assert!(matches!(
        SemanticResponseContract::new(
            RetrievalTopology::PartialQuality {
                coverage_ppm: 500_000,
            },
            RetrievalTopology::LexicalOnly,
            0,
            0,
            Some("recovery.state.model_missing".to_owned()),
        ),
        Err(RecoveryContractError::IncompatibleResponseTopology { .. })
    ));
    for (requested, realized) in [
        (RetrievalTopology::FastOnly, RetrievalTopology::QualityOnly),
        (
            RetrievalTopology::QualityOnly,
            RetrievalTopology::FullProgressive,
        ),
        (
            RetrievalTopology::HashControl,
            RetrievalTopology::LexicalOnly,
        ),
    ] {
        assert!(matches!(
            SemanticResponseContract::new(requested, realized, COMPLETE_COVERAGE_PPM, 0, None,),
            Err(RecoveryContractError::IncompatibleResponseTopology { .. })
        ));
    }

    for (realized, coverage) in [
        (RetrievalTopology::FastOnly, 0),
        (RetrievalTopology::QualityOnly, COMPLETE_COVERAGE_PPM - 1),
        (
            RetrievalTopology::FullProgressive,
            COMPLETE_COVERAGE_PPM + 1,
        ),
        (RetrievalTopology::LexicalOnly, 1),
        (RetrievalTopology::HashControl, 1),
        (RetrievalTopology::PartialQuality { coverage_ppm: 0 }, 0),
        (
            RetrievalTopology::PartialQuality {
                coverage_ppm: COMPLETE_COVERAGE_PPM,
            },
            COMPLETE_COVERAGE_PPM,
        ),
        (
            RetrievalTopology::PartialQuality {
                coverage_ppm: 250_000,
            },
            500_000,
        ),
    ] {
        let requested = match realized {
            RetrievalTopology::HashControl => RetrievalTopology::HashControl,
            RetrievalTopology::LexicalOnly | RetrievalTopology::FastOnly => {
                RetrievalTopology::FastOnly
            }
            RetrievalTopology::QualityOnly | RetrievalTopology::PartialQuality { .. } => {
                RetrievalTopology::QualityOnly
            }
            RetrievalTopology::FullProgressive => RetrievalTopology::FullProgressive,
        };
        let reason = matches!(realized, RetrievalTopology::LexicalOnly)
            .then(|| "recovery.state.model_missing".to_owned());
        assert!(matches!(
            SemanticResponseContract::new(requested, realized, coverage, 0, reason),
            Err(RecoveryContractError::InvalidCoverage { .. })
        ));
    }

    for topology in [
        RetrievalTopology::LexicalOnly,
        RetrievalTopology::HashControl,
    ] {
        let requested = if matches!(topology, RetrievalTopology::HashControl) {
            RetrievalTopology::HashControl
        } else {
            RetrievalTopology::FastOnly
        };
        let reason = matches!(topology, RetrievalTopology::LexicalOnly)
            .then(|| "recovery.state.model_missing".to_owned());
        assert!(matches!(
            SemanticResponseContract::new(requested, topology, 0, 1, reason),
            Err(RecoveryContractError::NonSemanticScoresAdmitted { .. })
        ));
    }

    assert_eq!(
        SemanticResponseContract::new(
            RetrievalTopology::FastOnly,
            RetrievalTopology::LexicalOnly,
            0,
            0,
            None,
        ),
        Err(RecoveryContractError::MissingDegradationReason)
    );
    assert_eq!(
        SemanticResponseContract::new(
            RetrievalTopology::FastOnly,
            RetrievalTopology::FastOnly,
            COMPLETE_COVERAGE_PPM,
            1,
            Some("recovery.state.model_missing".to_owned()),
        ),
        Err(RecoveryContractError::UnexpectedDegradationReason)
    );
}

#[test]
fn shell_rendering_quotes_unsafe_arguments() {
    let action = RecoveryAction {
        code: "recovery.action.build_index".to_owned(),
        explanation: String::new(),
        argv: vec![
            "fsfs".to_owned(),
            "index".to_owned(),
            "--index-dir".to_owned(),
            "/data/My Projects/it's here".to_owned(),
            ARG_SOURCE_DIR.to_owned(),
        ],
        network_required: false,
        consent_required: false,
        preserves_old_data: true,
        potentially_destructive: false,
        prerequisites: Vec::new(),
        expected_postcondition: "recovery.post.index_built".to_owned(),
        required_authorization: None,
    };
    assert_eq!(
        action.shell_command(),
        "fsfs index --index-dir '/data/My Projects/it'\\''s here' <source-dir>"
    );
}

#[test]
fn serialization_roundtrips_and_locks_schema_version() {
    for state in representative_states() {
        if matches!(state, SemanticReadiness::HashControl) {
            continue;
        }
        for request in semantic_requests() {
            for policy in all_policies() {
                let tier = tier_for_request(request);
                let state = state_for_request(&state, request);
                let acquisition_target = target_for(tier);
                let original = plan(
                    state.clone(),
                    request,
                    policy.clone(),
                    Some(&acquisition_target),
                )
                .expect("valid representative plan");
                let json = serde_json::to_value(&original).expect("serialize plan");
                assert_eq!(
                    json["schema_version"],
                    serde_json::Value::String(RECOVERY_PLAN_SCHEMA_VERSION.to_owned())
                );
                let decoded =
                    decode_and_validate(json, &state, request, &policy, Some(&acquisition_target))
                        .expect("validate untrusted plan against authoritative context");
                assert_eq!(decoded, original);
            }
        }
    }
    let hash = planned(SemanticReadiness::HashControl, hash_control(), permissive());
    let json = serde_json::to_value(&hash).expect("serialize hash plan");
    let decoded = decode_and_validate(
        json,
        &SemanticReadiness::HashControl,
        hash_control(),
        &permissive(),
        None,
    )
    .expect("validate hash plan");
    assert_eq!(decoded, hash);
}

#[test]
fn golden_plan_json_for_model_missing_offline_hybrid() {
    let plan = planned(
        missing(ModelTier::Quality),
        hybrid(RetrievalTopology::FullProgressive),
        RecoveryPolicy {
            interaction: InteractionPolicy::NonInteractive,
            network: NetworkPolicy::Offline,
            acquisition_authorization: None,
        },
    );
    let json = serde_json::to_value(&plan).expect("serialize plan");
    assert_eq!(json["schema_version"], "frankensearch.recovery_plan.v4");
    assert_eq!(json["state"]["state"], "model_missing");
    assert_eq!(json["state"]["detail"]["tier"], "quality");
    assert_eq!(json["state_code"], "recovery.state.model_missing");
    assert_eq!(json["provenance"], "unavailable");
    assert_eq!(json["mode"], "hybrid");
    assert_eq!(json["requested_topology"]["topology"], "full_progressive");
    assert_eq!(json["policy"]["interaction"], "non_interactive");
    assert_eq!(json["policy"]["network"], "offline");
    assert!(json["policy"]["acquisition_authorization"].is_null());
    assert_eq!(json["semantic_available"], false);
    assert_eq!(json["retryability"], "blocked_by_capability");
    assert_eq!(json["action"]["code"], "recovery.action.acquire_model");
    assert_eq!(json["action"]["argv"], serde_json::json!([]));
    assert_eq!(json["action"]["network_required"], false);
    assert_eq!(json["action"]["consent_required"], true);
    assert_eq!(
        json["action"]["prerequisites"][0],
        "recovery.capability.import_model_bundle"
    );
    assert_eq!(
        json["action"]["prerequisites"][1],
        "recovery.policy.grant_consent"
    );
    assert_eq!(
        json["action"]["required_authorization"]["schema_version"],
        "frankensearch.model_acquisition_authorization.v3"
    );
    assert_eq!(
        json["action"]["required_authorization"]["model_id"],
        "fixture-semantic-model"
    );
    assert_eq!(
        json["action"]["required_authorization"]["model_tier"],
        "quality"
    );
    assert_eq!(
        json["action"]["required_authorization"]["embedding_space"]["logical_model_id"],
        "fixture-semantic-model"
    );
    assert_eq!(
        json["action"]["required_authorization"]["embedding_space"]["dimension"],
        384
    );
    assert_eq!(
        json["action"]["required_authorization"]["source"]["kind"],
        "local_bundle"
    );
    assert_eq!(
        json["action"]["required_authorization"]["byte_budget"],
        42_000_000
    );
    assert_eq!(
        json["action"]["required_authorization"]["document_count"],
        12_345
    );
    assert_eq!(
        json["action"]["required_authorization"]["estimated_reindex_duration_ms"],
        98_765
    );
    assert_eq!(
        json["action"]["required_authorization"]["issued_at_unix_seconds"],
        TEST_AUTHORIZATION_ISSUED_AT_UNIX_SECONDS
    );
    assert_eq!(
        json["action"]["required_authorization"]["expires_at_unix_seconds"],
        TEST_AUTHORIZATION_EXPIRES_AT_UNIX_SECONDS
    );
    assert_eq!(
        json["action"]["required_authorization"]["nonce"],
        TEST_AUTHORIZATION_NONCE
    );
    assert_eq!(
        json["response_contract"]["requested_topology"]["topology"],
        "full_progressive"
    );
    assert_eq!(
        json["response_contract"]["realized_topology"]["topology"],
        "lexical_only"
    );
    assert_eq!(json["response_contract"]["coverage_ppm"], 0);
    assert_eq!(json["response_contract"]["admitted_semantic_scores"], 0);
    assert_eq!(
        json["response_contract"]["degradation_reason_code"],
        "recovery.state.model_missing"
    );
}

#[test]
fn golden_transition_table_is_stable() {
    // One compact row per unique state code and semantic mode under the
    // permissive policy, plus the explicit hash-control row.
    let states = vec![
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
        missing(ModelTier::Quality),
        unloadable(ModelTier::Quality),
        SemanticReadiness::IndexAbsent,
        SemanticReadiness::IdentityMismatch,
        SemanticReadiness::DaemonMismatch,
        SemanticReadiness::IndexEmpty(ZeroSignalReason::NewlyCreatedEmpty),
        SemanticReadiness::ManifestUnsafe,
        SemanticReadiness::AnnStale,
        SemanticReadiness::GenerationIncomplete,
        SemanticReadiness::PartialQualityCoverage {
            provenance: VerifiedSemanticProvenance::Local,
            coverage_ppm: 750_000,
        },
        SemanticReadiness::RemoteUnverified,
    ];
    let mut rows = Vec::new();
    for state in &states {
        for request in [
            explicit(RetrievalTopology::FullProgressive),
            hybrid(RetrievalTopology::FullProgressive),
        ] {
            let plan = planned(state.clone(), request, permissive());
            let mode_tag = match request.mode {
                RequestMode::ExplicitSemantic => "semantic",
                RequestMode::Hybrid => "hybrid",
                RequestMode::HashControl => unreachable!("semantic request array"),
            };
            let action_tag = plan.action.as_ref().map_or_else(
                || "none,false,false".to_owned(),
                |a| format!("{},{},{}", a.code, a.network_required, a.consent_required),
            );
            let retry_tag = match plan.retryability {
                Retryability::NotNeeded => "not_needed",
                Retryability::AfterAction => "after_action",
                Retryability::AfterRequestChange => "after_request_change",
                Retryability::BlockedByPolicy => "blocked",
                Retryability::BlockedByCapability => "capability_blocked",
            };
            let (realized, coverage) = plan
                .response_contract
                .as_ref()
                .map_or(("none", 0), |response| {
                    (response.realized_topology().code(), response.coverage_ppm())
                });
            rows.push(format!(
                "{}|{} => {},{},{},{}",
                plan.state_code, mode_tag, action_tag, retry_tag, realized, coverage
            ));
        }
    }
    let hash = planned(SemanticReadiness::HashControl, hash_control(), permissive());
    let hash_response = hash.response_contract.expect("hash response");
    rows.push(format!(
        "{}|hash_control => none,false,false,not_needed,{},{}",
        hash.state_code,
        hash_response.realized_topology().code(),
        hash_response.coverage_ppm()
    ));
    let expected: Vec<&str> = vec![
        "recovery.state.ready|semantic => none,false,false,not_needed,full_progressive,1000000",
        "recovery.state.ready|hybrid => none,false,false,not_needed,full_progressive,1000000",
        "recovery.state.model_missing|semantic => recovery.action.acquire_model,true,true,capability_blocked,none,0",
        "recovery.state.model_missing|hybrid => recovery.action.acquire_model,true,true,capability_blocked,lexical_only,0",
        "recovery.state.model_unloadable|semantic => recovery.action.reacquire_model,true,true,capability_blocked,none,0",
        "recovery.state.model_unloadable|hybrid => recovery.action.reacquire_model,true,true,capability_blocked,lexical_only,0",
        "recovery.state.index_absent|semantic => recovery.action.build_index,false,false,capability_blocked,none,0",
        "recovery.state.index_absent|hybrid => recovery.action.build_index,false,false,capability_blocked,lexical_only,0",
        "recovery.state.identity_mismatch|semantic => recovery.action.reindex_full,false,true,capability_blocked,none,0",
        "recovery.state.identity_mismatch|hybrid => recovery.action.reindex_full,false,true,capability_blocked,lexical_only,0",
        "recovery.state.daemon_mismatch|semantic => recovery.action.restart_daemon,false,false,capability_blocked,none,0",
        "recovery.state.daemon_mismatch|hybrid => recovery.action.restart_daemon,false,false,capability_blocked,lexical_only,0",
        "recovery.state.index_empty|semantic => recovery.action.ingest_content,false,false,capability_blocked,none,0",
        "recovery.state.index_empty|hybrid => recovery.action.ingest_content,false,false,capability_blocked,lexical_only,0",
        "recovery.state.manifest_unsafe|semantic => recovery.action.reindex_full,false,true,capability_blocked,none,0",
        "recovery.state.manifest_unsafe|hybrid => recovery.action.reindex_full,false,true,capability_blocked,lexical_only,0",
        "recovery.state.ann_stale|semantic => recovery.action.rebuild_ann,false,false,capability_blocked,none,0",
        "recovery.state.ann_stale|hybrid => recovery.action.rebuild_ann,false,false,capability_blocked,lexical_only,0",
        "recovery.state.generation_incomplete|semantic => recovery.action.resume_index,false,false,capability_blocked,none,0",
        "recovery.state.generation_incomplete|hybrid => recovery.action.resume_index,false,false,capability_blocked,lexical_only,0",
        "recovery.state.partial_quality_coverage|semantic => recovery.action.backfill_quality,false,false,capability_blocked,partial_quality,750000",
        "recovery.state.partial_quality_coverage|hybrid => recovery.action.backfill_quality,false,false,capability_blocked,partial_quality,750000",
        "recovery.state.remote_unverified|semantic => recovery.action.provide_attestation,false,false,blocked,none,0",
        "recovery.state.remote_unverified|hybrid => recovery.action.provide_attestation,false,false,blocked,lexical_only,0",
        "recovery.state.hash_control|hash_control => none,false,false,not_needed,hash_control,0",
    ];
    assert_eq!(
        rows, expected,
        "transition table changed: contract review required"
    );
}
