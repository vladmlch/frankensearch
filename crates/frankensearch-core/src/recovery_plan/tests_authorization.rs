use super::*;
use crate::generation::EmbeddingIdentityBundleV1;
use crate::recovery_plan::authorization::validate_authorization_binding;

#[test]
fn noninteractive_acquisition_requires_exact_scoped_authorization() {
    for network in [NetworkPolicy::Allowed, NetworkPolicy::Offline] {
        let required = planned(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            RecoveryPolicy {
                interaction: InteractionPolicy::Interactive,
                network,
                acquisition_authorization: None,
            },
        )
        .action
        .expect("acquisition action")
        .required_authorization
        .expect("scoped authorization");

        let exact = planned(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            RecoveryPolicy {
                interaction: InteractionPolicy::NonInteractive,
                network,
                acquisition_authorization: Some(required.clone()),
            },
        );
        let exact_action = exact.action.expect("acquisition action");
        assert_eq!(exact.retryability, Retryability::BlockedByCapability);
        assert_eq!(
            exact_action.prerequisites,
            [match network {
                NetworkPolicy::Allowed => {
                    "recovery.capability.execute_bound_model_acquisition"
                }
                NetworkPolicy::Offline => "recovery.capability.import_model_bundle",
            }]
        );

        let mut mismatches = Vec::new();
        let mut authorization = required.clone();
        authorization.model_id.push_str("-different");
        authorization.embedding_space.logical_model_id = authorization.model_id.clone();
        mismatches.push(("model_id", authorization));
        let mut authorization = required.clone();
        authorization.model_tier = match required.model_tier {
            ModelTier::Fast => ModelTier::Quality,
            ModelTier::Quality => ModelTier::Fast,
        };
        mismatches.push(("model_tier", authorization));
        let mut authorization = required.clone();
        authorization.embedding_space.dimension += 1;
        mismatches.push(("embedding_space", authorization));
        let mut authorization = required.clone();
        authorization.manifest_fingerprint = "c".repeat(64);
        mismatches.push(("manifest_fingerprint", authorization));
        let mut authorization = required.clone();
        authorization.upstream_revision.push_str("-different");
        authorization.embedding_space.immutable_revision = authorization.upstream_revision.clone();
        mismatches.push(("upstream_revision", authorization));
        let mut authorization = required.clone();
        authorization.license_spdx = "MIT".to_owned();
        mismatches.push(("license_spdx", authorization));
        let mut authorization = required.clone();
        authorization.source = match network {
            NetworkPolicy::Allowed => ModelAcquisitionSource::LocalBundle,
            NetworkPolicy::Offline => ModelAcquisitionSource::Network {
                source_hosts: vec!["models.example.test".to_owned()],
            },
        };
        mismatches.push(("source", authorization));
        let mut authorization = required.clone();
        authorization.byte_budget += 1;
        mismatches.push(("byte_budget", authorization));
        let mut authorization = required.clone();
        authorization.destination_class = ModelDestinationClass::ExplicitDirectory;
        mismatches.push(("destination_class", authorization));
        let mut authorization = required.clone();
        authorization.destination_fingerprint = "d".repeat(64);
        mismatches.push(("destination_fingerprint", authorization));
        let mut authorization = required.clone();
        authorization.document_count += 1;
        mismatches.push(("document_count", authorization));
        let mut authorization = required.clone();
        authorization.estimated_reindex_duration_ms += 1;
        mismatches.push(("estimated_reindex_duration_ms", authorization));
        let mut authorization = required.clone();
        authorization.issued_at_unix_seconds -= 1;
        mismatches.push(("issued_at_unix_seconds", authorization));
        let mut authorization = required.clone();
        authorization.expires_at_unix_seconds += 1;
        mismatches.push(("expires_at_unix_seconds", authorization));
        let mut authorization = required.clone();
        authorization.nonce = "fedcba9876543210fedcba9876543210".to_owned();
        mismatches.push(("nonce", authorization));

        for (field, authorization) in mismatches {
            let result = plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                RecoveryPolicy {
                    interaction: InteractionPolicy::NonInteractive,
                    network,
                    acquisition_authorization: Some(authorization),
                },
                Some(&target()),
            );
            assert_eq!(
                result,
                Err(RecoveryContractError::MismatchedAcquisitionAuthorization { field }),
                "scope mismatch must fail closed for {field}"
            );
        }
    }
}

#[test]
fn acquisition_authorization_enforces_window_and_nonce_boundaries() {
    let authorization = target()
        .authorization_for(NetworkPolicy::Allowed)
        .expect("valid authorization fixture");

    for (issued_at_unix_seconds, expires_at_unix_seconds) in [(100, 100), (101, 100)] {
        let mut invalid = authorization.clone();
        invalid.issued_at_unix_seconds = issued_at_unix_seconds;
        invalid.expires_at_unix_seconds = expires_at_unix_seconds;
        assert_eq!(
            invalid.validate(),
            Err(
                RecoveryContractError::InvalidAcquisitionAuthorizationWindow {
                    issued_at_unix_seconds,
                    expires_at_unix_seconds,
                }
            )
        );
    }

    let mut maximum_lifetime = authorization.clone();
    maximum_lifetime.issued_at_unix_seconds = 1_000;
    maximum_lifetime.expires_at_unix_seconds =
        1_000 + MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS;
    maximum_lifetime
        .validate()
        .expect("the maximum authorization lifetime is inclusive");

    let mut excessive_lifetime = maximum_lifetime.clone();
    excessive_lifetime.expires_at_unix_seconds =
        excessive_lifetime.expires_at_unix_seconds.saturating_add(1);
    assert_eq!(
        excessive_lifetime.validate(),
        Err(
            RecoveryContractError::AcquisitionAuthorizationLifetimeExceeded {
                lifetime_seconds: MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS + 1,
                max_lifetime_seconds: MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS,
            }
        )
    );

    for nonce in [
        "0123456789abcdef0123456789abcde",
        "0123456789abcdef0123456789abcdef0",
        "0123456789ABCDEF0123456789ABCDEF",
        "0123456789abcdef0123456789abcdeg",
        "00000000000000000000000000000000",
    ] {
        let mut invalid = authorization.clone();
        invalid.nonce = nonce.to_owned();
        assert_eq!(
            invalid.validate(),
            Err(RecoveryContractError::InvalidAcquisitionAuthorizationNonce),
            "invalid nonce unexpectedly admitted: {nonce}"
        );
    }

    for nonce in [
        "00000000000000000000000000000001",
        "ffffffffffffffffffffffffffffffff",
    ] {
        let mut valid = authorization.clone();
        valid.nonce = nonce.to_owned();
        let result = valid.validate();
        assert!(result.is_ok(), "valid nonce {nonce} rejected: {result:?}");
    }
}

#[test]
fn acquisition_authorization_uses_trusted_time_and_exclusive_expiry() {
    let authorization = target()
        .authorization_for(NetworkPolicy::Allowed)
        .expect("valid authorization fixture");
    let before_issuance = authorization.issued_at_unix_seconds.saturating_sub(1);

    assert_eq!(
        authorization.validate_at(before_issuance),
        Err(RecoveryContractError::AcquisitionAuthorizationNotYetValid {
            issued_at_unix_seconds: authorization.issued_at_unix_seconds,
            evaluation_time_unix_seconds: before_issuance,
        })
    );
    authorization
        .validate_at(authorization.issued_at_unix_seconds)
        .expect("authorization is valid at its inclusive issuance boundary");
    authorization
        .validate_at(authorization.expires_at_unix_seconds - 1)
        .expect("authorization is valid immediately before expiry");
    assert_eq!(
        authorization.validate_at(authorization.expires_at_unix_seconds),
        Err(RecoveryContractError::AcquisitionAuthorizationExpired {
            expires_at_unix_seconds: authorization.expires_at_unix_seconds,
            evaluation_time_unix_seconds: authorization.expires_at_unix_seconds,
        })
    );
}

#[test]
fn authorization_binding_rejects_surplus_and_stale_scopes() {
    let required = target()
        .authorization_for(NetworkPolicy::Allowed)
        .expect("valid authorization fixture");
    assert_eq!(
        validate_authorization_binding(None, Some(&required), TEST_NOW_UNIX_SECONDS),
        Err(RecoveryContractError::SurplusAcquisitionAuthorization)
    );

    assert_eq!(
        validate_authorization_binding(Some(&required), None, required.expires_at_unix_seconds,),
        Err(RecoveryContractError::AcquisitionAuthorizationExpired {
            expires_at_unix_seconds: required.expires_at_unix_seconds,
            evaluation_time_unix_seconds: required.expires_at_unix_seconds,
        })
    );

    let mut stale_supplied = required.clone();
    stale_supplied.expires_at_unix_seconds = TEST_NOW_UNIX_SECONDS;
    assert_eq!(
        validate_authorization_binding(
            Some(&required),
            Some(&stale_supplied),
            TEST_NOW_UNIX_SECONDS,
        ),
        Err(RecoveryContractError::AcquisitionAuthorizationExpired {
            expires_at_unix_seconds: TEST_NOW_UNIX_SECONDS,
            evaluation_time_unix_seconds: TEST_NOW_UNIX_SECONDS,
        })
    );
}

#[test]
fn planner_rejects_surplus_authorization_when_no_action_requires_it() {
    let surplus = target()
        .authorization_for(NetworkPolicy::Allowed)
        .expect("valid authorization fixture");
    let result = plan(
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
        explicit(RetrievalTopology::FastOnly),
        RecoveryPolicy {
            interaction: InteractionPolicy::NonInteractive,
            network: NetworkPolicy::Allowed,
            acquisition_authorization: Some(surplus),
        },
        None,
    );
    assert_eq!(
        result,
        Err(RecoveryContractError::SurplusAcquisitionAuthorization)
    );
}

#[test]
fn promotion_and_execution_each_recheck_authorization_expiry() {
    let state = missing(ModelTier::Quality);
    let request = explicit(RetrievalTopology::QualityOnly);
    let acquisition_target = target();
    let presentation_policy = RecoveryPolicy {
        interaction: InteractionPolicy::Interactive,
        network: NetworkPolicy::Allowed,
        acquisition_authorization: None,
    };
    let presentation_plan = crate::recovery_plan::plan(TrustedRecoveryContext::new(
        &state,
        request,
        &presentation_policy,
        Some(&acquisition_target),
        acquisition_target.expires_at_unix_seconds - 1,
    ))
    .expect("interactive presentation plan before authorization");
    assert_eq!(
        presentation_plan.validate_for_execution_at(acquisition_target.expires_at_unix_seconds - 1),
        Err(RecoveryContractError::MissingAcquisitionAuthorization),
        "a required but ungranted authorization must never pass the execution gate"
    );

    let exact_authorization = acquisition_target
        .authorization_for(NetworkPolicy::Allowed)
        .expect("exact authorization");
    let policy = RecoveryPolicy {
        interaction: InteractionPolicy::NonInteractive,
        network: NetworkPolicy::Allowed,
        acquisition_authorization: Some(exact_authorization),
    };
    let canonical = crate::recovery_plan::plan(TrustedRecoveryContext::new(
        &state,
        request,
        &policy,
        Some(&acquisition_target),
        TEST_NOW_UNIX_SECONDS,
    ))
    .expect("serialize a currently valid recovery plan");
    let serialized = serde_json::to_value(canonical).expect("serialize recovery plan");
    let promoted = serde_json::from_value::<UntrustedRecoveryPlan>(serialized.clone())
        .expect("decode untrusted recovery plan")
        .validate_against(TrustedRecoveryContext::new(
            &state,
            request,
            &policy,
            Some(&acquisition_target),
            acquisition_target.expires_at_unix_seconds - 1,
        ))
        .expect("promotion succeeds immediately before expiry");
    promoted
        .validate_for_execution_at(acquisition_target.expires_at_unix_seconds - 1)
        .expect("exact supplied authorization executes immediately before expiry");
    assert_eq!(
        promoted.validate_for_execution_at(acquisition_target.expires_at_unix_seconds),
        Err(RecoveryContractError::AcquisitionAuthorizationExpired {
            expires_at_unix_seconds: acquisition_target.expires_at_unix_seconds,
            evaluation_time_unix_seconds: acquisition_target.expires_at_unix_seconds,
        }),
        "a previously promoted plan is not a timeless execution capability"
    );

    let untrusted: UntrustedRecoveryPlan =
        serde_json::from_value(serialized).expect("decode second untrusted recovery plan");
    assert_eq!(
        untrusted.validate_against(TrustedRecoveryContext::new(
            &state,
            request,
            &policy,
            Some(&acquisition_target),
            acquisition_target.expires_at_unix_seconds,
        )),
        Err(RecoveryContractError::AcquisitionAuthorizationExpired {
            expires_at_unix_seconds: acquisition_target.expires_at_unix_seconds,
            evaluation_time_unix_seconds: acquisition_target.expires_at_unix_seconds,
        })
    );
}

#[test]
fn planner_rejects_programmatically_constructed_invalid_policy_authorization() {
    let mut invalid = target()
        .authorization_for(NetworkPolicy::Allowed)
        .expect("valid authorization fixture");
    invalid.byte_budget = 0;
    let result = plan(
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
        explicit(RetrievalTopology::FastOnly),
        RecoveryPolicy {
            interaction: InteractionPolicy::NonInteractive,
            network: NetworkPolicy::Allowed,
            acquisition_authorization: Some(invalid),
        },
        None,
    );
    assert_eq!(
        result,
        Err(RecoveryContractError::ZeroAcquisitionByteBudget),
        "planner must not serialize invalid authorization even when the ready action ignores it"
    );
}

#[test]
fn acquisition_target_must_be_bound_and_well_formed() {
    let unbound = plan(
        missing(ModelTier::Fast),
        explicit(RetrievalTopology::FastOnly),
        permissive(),
        None,
    )
    .expect("missing binding yields a non-executable plan");
    assert_eq!(unbound.retryability, Retryability::BlockedByCapability);
    let action = unbound.action.expect("acquisition action");
    assert!(action.required_authorization.is_none());
    assert!(
        action.argv.is_empty(),
        "an unbound target cannot name an exact model and must not be executable"
    );
    assert_eq!(
        action.prerequisites,
        [
            "recovery.capability.execute_bound_model_acquisition",
            "recovery.policy.bind_model",
        ]
    );

    let mut malformed = target();
    malformed.model_id = " ".to_owned();
    assert!(matches!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&malformed),
        ),
        Err(RecoveryContractError::InvalidAcquisitionScopeField { field: "model_id" })
    ));

    let mut malformed = target();
    malformed.manifest_fingerprint = " ".to_owned();
    assert!(matches!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&malformed),
        ),
        Err(RecoveryContractError::InvalidAcquisitionScopeField {
            field: "manifest_fingerprint"
        })
    ));

    let mut malformed = target();
    malformed.upstream_revision = "revision\ninjected".to_owned();
    assert!(matches!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&malformed),
        ),
        Err(RecoveryContractError::InvalidAcquisitionScopeField {
            field: "upstream_revision"
        })
    ));

    let mut malformed = target();
    malformed.license_spdx.clear();
    assert!(matches!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&malformed),
        ),
        Err(RecoveryContractError::InvalidAcquisitionScopeField {
            field: "license_spdx"
        })
    ));

    let mut malformed = target();
    malformed.destination_fingerprint.clear();
    assert!(matches!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&malformed),
        ),
        Err(RecoveryContractError::InvalidAcquisitionScopeField {
            field: "destination_fingerprint"
        })
    ));

    let mut malformed = target();
    malformed.byte_budget = 0;
    assert_eq!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&malformed),
        ),
        Err(RecoveryContractError::ZeroAcquisitionByteBudget)
    );

    let mut missing_host = target();
    missing_host.network_source_hosts.clear();
    assert_eq!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&missing_host),
        ),
        Err(RecoveryContractError::MissingNetworkSourceHosts)
    );
    let offline = plan(
        missing(ModelTier::Quality),
        explicit(RetrievalTopology::QualityOnly),
        RecoveryPolicy {
            interaction: InteractionPolicy::Interactive,
            network: NetworkPolicy::Offline,
            acquisition_authorization: None,
        },
        Some(&missing_host),
    )
    .expect("local bundle does not depend on network hosts");
    assert!(matches!(
        offline
            .action
            .expect("offline action")
            .required_authorization
            .expect("offline authorization")
            .source,
        ModelAcquisitionSource::LocalBundle
    ));
}

#[test]
fn acquisition_tier_and_embedding_space_are_anchored_to_trusted_inputs() {
    assert!(matches!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::FastOnly),
            permissive(),
            Some(&target_for(ModelTier::Quality)),
        ),
        Err(RecoveryContractError::UnavailableTierTopologyMismatch {
            tier: ModelTier::Quality,
            ..
        })
    ));
    assert!(matches!(
        plan(
            missing(ModelTier::Fast),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&target_for(ModelTier::Fast)),
        ),
        Err(RecoveryContractError::UnavailableTierTopologyMismatch {
            tier: ModelTier::Fast,
            ..
        })
    ));
    assert_eq!(
        plan(
            missing(ModelTier::Fast),
            explicit(RetrievalTopology::FullProgressive),
            permissive(),
            Some(&target_for(ModelTier::Quality)),
        ),
        Err(RecoveryContractError::AcquisitionTargetTierMismatch {
            readiness_tier: ModelTier::Fast,
            target_tier: ModelTier::Quality,
        })
    );
    plan(
        missing(ModelTier::Fast),
        explicit(RetrievalTopology::FullProgressive),
        permissive(),
        Some(&target_for(ModelTier::Fast)),
    )
    .expect("full progressive may recover the exact unavailable fast tier");
    plan(
        missing(ModelTier::Quality),
        explicit(RetrievalTopology::FullProgressive),
        permissive(),
        Some(&target_for(ModelTier::Quality)),
    )
    .expect("full progressive may recover the exact unavailable quality tier");

    let mut malformed = target();
    malformed.embedding_space.dimension = 0;
    assert!(matches!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&malformed),
        ),
        Err(RecoveryContractError::InvalidAcquisitionSpaceIdentity { .. })
    ));

    let mut hash_target = target();
    hash_target.embedding_space =
        EmbeddingIdentityBundleV1::explicit_test_model("hash-control", 384).space;
    hash_target.model_id = hash_target.embedding_space.logical_model_id.clone();
    hash_target.upstream_revision = hash_target.embedding_space.immutable_revision.clone();
    assert_eq!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&hash_target),
        ),
        Err(RecoveryContractError::NonSemanticAcquisitionSpace)
    );

    let mut inconsistent = target();
    inconsistent.model_id = "other-semantic-model".to_owned();
    assert_eq!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&inconsistent),
        ),
        Err(RecoveryContractError::InconsistentAcquisitionIdentity { field: "model_id" })
    );

    let mut inconsistent = target();
    inconsistent.upstream_revision = "different-immutable-revision".to_owned();
    assert_eq!(
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&inconsistent),
        ),
        Err(RecoveryContractError::InconsistentAcquisitionIdentity {
            field: "upstream_revision"
        })
    );
}

#[test]
fn network_source_hosts_are_path_free_credential_free_authorities() {
    for host in [
        "models.example.test",
        "MODELS.EXAMPLE.TEST:443",
        "localhost",
        "127.0.0.1",
        "127.0.0.1:8443",
        "[2001:db8::1]",
        "[2001:db8::1]:443",
    ] {
        let mut valid = target();
        valid.network_source_hosts = vec![host.to_owned()];
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
            Some(&valid),
        )
        .expect("credential-free host authority is valid");
    }

    for host in [
        "",
        " ",
        "https://models.example.test",
        "user:secret@models.example.test",
        "models.example.test/path",
        "models.example.test?variant=quality",
        "models.example.test#fragment",
        "models.example.test\\path",
        "models .example.test",
        "models.example.test\n",
        "models_example.test",
        "-models.example.test",
        "models-.example.test",
        "models..example.test",
        "models.example.test.",
        "999.999.999.999",
        "models.example.test:",
        "models.example.test:0",
        "models.example.test:+443",
        "models.example.test:65536",
        "models.example.test:https",
        "2001:db8::1",
        "[2001:db8::1",
        "[2001:db8::1]suffix",
        "[2001:db8::1]:0",
        "[2001:db8::1]:65536",
        "[not-ipv6]",
    ] {
        let mut malformed = target();
        malformed.network_source_hosts = vec![host.to_owned()];
        assert_eq!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::InvalidNetworkSourceHost),
            "unexpectedly accepted source host {host:?}"
        );
    }
}

#[test]
fn invalid_partial_readiness_never_enters_a_plan() {
    for coverage_ppm in [0, COMPLETE_COVERAGE_PPM, COMPLETE_COVERAGE_PPM + 1] {
        assert!(matches!(
            plan(
                SemanticReadiness::PartialQualityCoverage {
                    provenance: VerifiedSemanticProvenance::Local,
                    coverage_ppm,
                },
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&target()),
            ),
            Err(RecoveryContractError::InvalidCoverage { .. })
        ));
    }
}
