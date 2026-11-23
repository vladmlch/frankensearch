use serde::Deserialize;

use super::*;

#[test]
fn serde_rejects_unknown_versions_fields_and_mismatched_scopes() {
    let state = missing(ModelTier::Quality);
    let request = hybrid(RetrievalTopology::FullProgressive);
    let policy = permissive();
    let acquisition_target = target();
    let canonical = plan(
        state.clone(),
        request,
        policy.clone(),
        Some(&acquisition_target),
    )
    .expect("canonical plan");
    let plan_json = serde_json::to_value(canonical).expect("serialize plan");
    let rejects = |value| {
        decode_and_validate(value, &state, request, &policy, Some(&acquisition_target)).is_err()
    };

    for legacy_version in [
        "frankensearch.recovery_plan.v1",
        "frankensearch.recovery_plan.v2",
        "frankensearch.recovery_plan.v3",
    ] {
        let mut changed = plan_json.clone();
        changed["schema_version"] = serde_json::Value::String(legacy_version.to_owned());
        assert!(
            rejects(changed),
            "legacy plan schema unexpectedly decoded: {legacy_version}"
        );
    }

    let mut changed = plan_json.clone();
    changed["unknown_contract_field"] = serde_json::Value::Bool(true);
    assert!(rejects(changed));

    let mut changed = plan_json.clone();
    changed["state_code"] = serde_json::Value::String("recovery.state.ready".to_owned());
    assert!(rejects(changed));

    let mut changed = plan_json.clone();
    changed["provenance"] = serde_json::Value::String("verified_local".to_owned());
    assert!(rejects(changed));

    let mut changed = plan_json.clone();
    changed["semantic_available"] = serde_json::Value::Bool(true);
    assert!(rejects(changed));

    let mut changed = plan_json.clone();
    changed["retryability"] = serde_json::Value::String("not_needed".to_owned());
    assert!(rejects(changed));

    let action_mutations = [
        ("code", serde_json::json!("recovery.action.reacquire_model")),
        ("explanation", serde_json::json!("trust the payload")),
        (
            "argv",
            serde_json::json!(["fsfs", "download-models", "--model", "different-model"]),
        ),
        ("network_required", serde_json::json!(false)),
        ("consent_required", serde_json::json!(false)),
        ("preserves_old_data", serde_json::json!(false)),
        ("potentially_destructive", serde_json::json!(true)),
        (
            "prerequisites",
            serde_json::json!(["recovery.policy.allow_network"]),
        ),
        (
            "expected_postcondition",
            serde_json::json!("recovery.state.ready"),
        ),
    ];
    for (field, value) in action_mutations {
        let mut changed = plan_json.clone();
        changed["action"][field] = value;
        assert!(rejects(changed), "trusted wire action field {field}");
    }

    let mut changed = plan_json.clone();
    changed["action"]["required_authorization"]["byte_budget"] =
        serde_json::Value::from(42_000_001_u64);
    assert!(rejects(changed), "trusted wire authorization substitution");

    let response_mutations = [
        (
            "requested_topology",
            serde_json::json!({"topology": "quality_only"}),
        ),
        (
            "realized_topology",
            serde_json::json!({"topology": "fast_only"}),
        ),
        ("coverage_ppm", serde_json::json!(1)),
        ("admitted_semantic_scores", serde_json::json!(1)),
        (
            "degradation_reason_code",
            serde_json::json!("recovery.state.model_unloadable"),
        ),
    ];
    for (field, value) in response_mutations {
        let mut changed = plan_json.clone();
        changed["response_contract"][field] = value;
        assert!(rejects(changed), "trusted wire response field {field}");
    }

    let authorization = plan_json["action"]["required_authorization"].clone();
    for legacy_version in [
        "frankensearch.model_acquisition_authorization.v1",
        "frankensearch.model_acquisition_authorization.v2",
    ] {
        let mut changed = authorization.clone();
        changed["schema_version"] = serde_json::Value::String(legacy_version.to_owned());
        assert!(
            serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
            "legacy authorization schema unexpectedly decoded: {legacy_version}"
        );
    }

    let mut changed = authorization.clone();
    changed["model_id"] = serde_json::Value::String(" ".to_owned());
    assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

    let mut changed = authorization.clone();
    changed["model_tier"] = serde_json::Value::String("Quality".to_owned());
    assert!(
        serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
        "v4 recovery authorization must not inherit ModelTier's Rust casing"
    );

    for required_field in [
        "model_id",
        "model_tier",
        "embedding_space",
        "document_count",
        "estimated_reindex_duration_ms",
        "issued_at_unix_seconds",
        "expires_at_unix_seconds",
        "nonce",
    ] {
        let mut changed = authorization.clone();
        changed
            .as_object_mut()
            .expect("authorization JSON object")
            .remove(required_field);
        assert!(
            serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
            "missing required consent field unexpectedly decoded: {required_field}"
        );
    }

    for (field, invalid_value) in [
        ("issued_at_unix_seconds", serde_json::json!("now")),
        ("expires_at_unix_seconds", serde_json::Value::Null),
        ("nonce", serde_json::json!(123_u64)),
    ] {
        let mut changed = authorization.clone();
        changed[field] = invalid_value;
        assert!(
            serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
            "wrongly typed authorization field unexpectedly decoded: {field}"
        );
    }

    let mut changed = authorization.clone();
    changed["source"]["kind"] = serde_json::Value::String("ambient".to_owned());
    assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

    let mut changed = authorization.clone();
    changed["destination_class"] = serde_json::Value::String("unbounded_path".to_owned());
    assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

    let mut changed = authorization.clone();
    changed["byte_budget"] = serde_json::Value::from(0);
    assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

    let mut changed = authorization.clone();
    changed["embedding_space"]["dimension"] = serde_json::Value::from(0);
    assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

    let mut changed = authorization.clone();
    changed["embedding_space"]["logical_model_id"] =
        serde_json::Value::String("different-model".to_owned());
    assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

    for host in [
        "https://models.example.test",
        "user:secret@models.example.test",
        "models.example.test/path",
        "models.example.test?variant=quality",
        "models.example.test#fragment",
        "models example.test",
        "models.example.test:0",
        "models.example.test:+443",
        "2001:db8::1",
        "[not-ipv6]",
    ] {
        let mut changed = authorization.clone();
        changed["source"]["source_hosts"] = serde_json::json!([host]);
        assert!(
            serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
            "serde unexpectedly accepted source host {host:?}"
        );
    }

    let mut changed = authorization;
    changed["manifest_fingerprint"] = serde_json::Value::String(" ".to_owned());
    assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

    let mut changed = plan_json.clone();
    changed["action"]["argv"] =
        serde_json::json!(["fsfs", "download-models", "--model", "different-model"]);
    assert!(rejects(changed), "argv must remain bound to trusted target");

    let mut changed = plan_json;
    changed["action"]["required_authorization"]["source"] = serde_json::json!({
        "kind": "local_bundle",
        "source_hosts": ["must-not-be-ignored.example.test"]
    });
    assert!(rejects(changed));
}

#[test]
fn untrusted_wire_rejects_unknown_fields_at_every_nested_object_boundary() {
    let state = missing(ModelTier::Quality);
    let request = hybrid(RetrievalTopology::FullProgressive);
    let policy = permissive();
    let acquisition_target = target();
    let canonical = plan(state, request, policy, Some(&acquisition_target))
        .expect("canonical nested-wire fixture");
    let plan_json = serde_json::to_value(canonical).expect("serialize nested-wire fixture");
    let rejects_plan = |value| serde_json::from_value::<UntrustedRecoveryPlan>(value).is_err();

    let mut changed = plan_json.clone();
    changed["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "plan envelope");

    let mut changed = plan_json.clone();
    changed["state"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "readiness envelope");

    let mut changed = plan_json.clone();
    changed["state"]["detail"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "readiness detail");

    let mut changed = plan_json.clone();
    changed["requested_topology"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "requested topology");

    let mut changed = plan_json.clone();
    changed["policy"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "policy");

    let mut changed = plan_json.clone();
    changed["action"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "action");

    let mut changed = plan_json.clone();
    changed["action"]["required_authorization"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "acquisition authorization");

    let mut changed = plan_json.clone();
    changed["action"]["required_authorization"]["embedding_space"]["unexpected"] =
        serde_json::json!(true);
    assert!(rejects_plan(changed), "embedding-space identity");

    let mut changed = plan_json.clone();
    changed["action"]["required_authorization"]["embedding_space"]["artifacts"][0]["unexpected"] =
        serde_json::json!(true);
    assert!(rejects_plan(changed), "embedding artifact identity");

    let mut changed = plan_json.clone();
    changed["action"]["required_authorization"]["source"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "acquisition source");

    let mut changed = plan_json.clone();
    changed["response_contract"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "response contract");

    let mut changed = plan_json.clone();
    changed["response_contract"]["requested_topology"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "response requested topology");

    let mut changed = plan_json;
    changed["response_contract"]["realized_topology"]["unexpected"] = serde_json::json!(true);
    assert!(rejects_plan(changed), "response realized topology");

    for state_json in [
        serde_json::json!({
            "state": "ready",
            "detail": {"provenance": "local", "unexpected": true}
        }),
        serde_json::json!({
            "state": "model_missing",
            "detail": {"tier": "fast", "unexpected": true}
        }),
        serde_json::json!({
            "state": "model_unloadable",
            "detail": {"tier": "quality", "unexpected": true}
        }),
        serde_json::json!({
            "state": "partial_quality_coverage",
            "detail": {
                "provenance": "daemon",
                "coverage_ppm": 625_000,
                "unexpected": true
            }
        }),
    ] {
        assert!(
            serde_json::from_value::<SemanticReadiness>(state_json).is_err(),
            "object-valued readiness detail discarded an unknown field"
        );
    }

    assert!(
        serde_json::from_value::<SemanticReadiness>(serde_json::json!({
            "state": "index_absent",
            "unexpected": true
        }))
        .is_err(),
        "unit readiness envelope discarded an unknown field"
    );

    assert!(
        serde_json::from_value::<SemanticReadiness>(serde_json::json!({
            "state": "model_missing",
            "detail": {"tier": "Quality"}
        }))
        .is_err(),
        "v3 readiness tier must use the recovery-local lowercase spelling"
    );

    assert!(
        serde_json::from_value::<RecoveryRequest>(serde_json::json!({
            "mode": "hybrid",
            "requested_topology": {
                "topology": "partial_quality",
                "coverage_ppm": 625_000,
                "unexpected": true
            }
        }))
        .is_err(),
        "structured topology discarded an unknown field"
    );

    for invalid_topology in [
        serde_json::json!({"topology": "fast_only", "coverage_ppm": 625_000}),
        serde_json::json!({"topology": "partial_quality"}),
        serde_json::json!({"topology": "partial_quality", "coverage_ppm": null}),
    ] {
        assert!(
            serde_json::from_value::<RecoveryRequest>(serde_json::json!({
                "mode": "hybrid",
                "requested_topology": invalid_topology
            }))
            .is_err(),
            "topology-specific field presence was not enforced"
        );
    }
}

#[test]
fn tagged_wire_envelopes_enforce_variant_specific_fields() {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TopologyHarness {
        value: RetrievalTopology,
    }

    for expected in [
        RetrievalTopology::LexicalOnly,
        RetrievalTopology::HashControl,
        RetrievalTopology::FastOnly,
        RetrievalTopology::QualityOnly,
        RetrievalTopology::FullProgressive,
    ] {
        let topology_json = serde_json::to_value(expected).expect("serialize topology");
        let decoded: TopologyHarness =
            serde_json::from_value(serde_json::json!({"value": topology_json.clone()}))
                .expect("decode exact unit topology");
        assert_eq!(decoded.value, expected);

        let mut unknown = topology_json.clone();
        unknown["unexpected"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<TopologyHarness>(serde_json::json!({"value": unknown}))
                .is_err(),
            "unit topology accepted an unknown field: {expected:?}"
        );

        let mut forbidden = topology_json;
        forbidden["coverage_ppm"] = serde_json::json!(625_000);
        assert!(
            serde_json::from_value::<TopologyHarness>(serde_json::json!({"value": forbidden}))
                .is_err(),
            "unit topology accepted partial-only coverage: {expected:?}"
        );
        for forbidden_value in [serde_json::Value::Null, serde_json::json!({})] {
            let mut forbidden = serde_json::to_value(expected).expect("serialize topology");
            forbidden["coverage_ppm"] = forbidden_value;
            assert!(
                serde_json::from_value::<TopologyHarness>(serde_json::json!({"value": forbidden}))
                    .is_err(),
                "unit topology accepted null or structured coverage: {expected:?}"
            );
        }
    }

    let partial = RetrievalTopology::PartialQuality {
        coverage_ppm: 625_000,
    };
    let partial_json = serde_json::to_value(partial).expect("serialize partial topology");
    let decoded: TopologyHarness =
        serde_json::from_value(serde_json::json!({"value": partial_json.clone()}))
            .expect("decode exact partial topology");
    assert_eq!(decoded.value, partial);
    for invalid in [
        serde_json::json!({"topology": "partial_quality"}),
        serde_json::json!({"topology": "partial_quality", "coverage_ppm": null}),
        serde_json::json!({
            "topology": "partial_quality",
            "coverage_ppm": 625_000,
            "unexpected": true
        }),
        serde_json::json!({}),
        serde_json::json!({"topology": null}),
    ] {
        assert!(
            serde_json::from_value::<TopologyHarness>(serde_json::json!({"value": invalid}))
                .is_err(),
            "topology envelope accepted a missing, null, or forbidden field"
        );
    }

    let unit_states = [
        SemanticReadiness::IndexAbsent,
        SemanticReadiness::IdentityMismatch,
        SemanticReadiness::DaemonMismatch,
        SemanticReadiness::ManifestUnsafe,
        SemanticReadiness::AnnStale,
        SemanticReadiness::GenerationIncomplete,
        SemanticReadiness::RemoteUnverified,
        SemanticReadiness::HashControl,
    ];
    for expected in unit_states {
        let state_json = serde_json::to_value(&expected).expect("serialize unit readiness");
        assert_eq!(
            serde_json::from_value::<SemanticReadiness>(state_json.clone())
                .expect("decode exact unit readiness"),
            expected
        );

        let mut unknown = state_json.clone();
        unknown["unexpected"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<SemanticReadiness>(unknown).is_err(),
            "unit readiness accepted an unknown field: {expected:?}"
        );

        for forbidden_detail in [
            serde_json::json!({}),
            serde_json::json!("caller_requested_zero_k"),
            serde_json::json!({"tier": "fast"}),
        ] {
            let mut forbidden = state_json.clone();
            forbidden["detail"] = forbidden_detail;
            assert!(
                serde_json::from_value::<SemanticReadiness>(forbidden).is_err(),
                "unit readiness accepted scalar or structured detail: {expected:?}"
            );
        }

        let mut null_detail = state_json;
        null_detail["detail"] = serde_json::Value::Null;
        assert!(
            serde_json::from_value::<SemanticReadiness>(null_detail).is_err(),
            "unit readiness accepted null detail: {expected:?}"
        );
    }

    for expected in [
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
        missing(ModelTier::Fast),
        unloadable(ModelTier::Quality),
        SemanticReadiness::IndexEmpty(ZeroSignalReason::CallerRequestedZeroK),
        SemanticReadiness::PartialQualityCoverage {
            provenance: VerifiedSemanticProvenance::Daemon,
            coverage_ppm: 625_000,
        },
    ] {
        let state_json = serde_json::to_value(&expected).expect("serialize detailed readiness");
        assert_eq!(
            serde_json::from_value::<SemanticReadiness>(state_json.clone())
                .expect("decode exact detailed readiness"),
            expected
        );

        let mut missing_detail = state_json.clone();
        missing_detail
            .as_object_mut()
            .expect("readiness object")
            .remove("detail");
        assert!(
            serde_json::from_value::<SemanticReadiness>(missing_detail).is_err(),
            "detailed readiness accepted missing detail: {expected:?}"
        );

        let mut null_detail = state_json.clone();
        null_detail["detail"] = serde_json::Value::Null;
        assert!(
            serde_json::from_value::<SemanticReadiness>(null_detail).is_err(),
            "detailed readiness accepted null detail: {expected:?}"
        );

        let mut forbidden_detail = state_json.clone();
        if let Some(detail) = forbidden_detail["detail"].as_object_mut() {
            detail.insert("unexpected".to_owned(), serde_json::json!(true));
        } else {
            forbidden_detail["detail"] = serde_json::json!({"unexpected": true});
        }
        assert!(
            serde_json::from_value::<SemanticReadiness>(forbidden_detail).is_err(),
            "detailed readiness discarded a forbidden detail field: {expected:?}"
        );

        let mut unknown = state_json;
        unknown["unexpected"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<SemanticReadiness>(unknown).is_err(),
            "detailed readiness accepted an unknown envelope field: {expected:?}"
        );
    }
    assert!(serde_json::from_value::<SemanticReadiness>(serde_json::json!({})).is_err());
    assert!(
        serde_json::from_value::<SemanticReadiness>(serde_json::json!({"state": null})).is_err()
    );
    for cross_variant in [
        serde_json::json!({
            "state": "ready",
            "detail": {"tier": "fast"}
        }),
        serde_json::json!({
            "state": "model_missing",
            "detail": {"provenance": "local"}
        }),
        serde_json::json!({
            "state": "model_unloadable",
            "detail": {"provenance": "daemon", "coverage_ppm": 625_000}
        }),
        serde_json::json!({
            "state": "index_empty",
            "detail": {"provenance": "local", "coverage_ppm": 625_000}
        }),
        serde_json::json!({
            "state": "partial_quality_coverage",
            "detail": "caller_requested_zero_k"
        }),
    ] {
        assert!(
            serde_json::from_value::<SemanticReadiness>(cross_variant).is_err(),
            "readiness tag accepted another variant's otherwise-valid detail"
        );
    }

    let local_bundle =
        serde_json::to_value(ModelAcquisitionSource::LocalBundle).expect("serialize local bundle");
    assert_eq!(
        serde_json::from_value::<ModelAcquisitionSource>(local_bundle.clone())
            .expect("decode exact local bundle"),
        ModelAcquisitionSource::LocalBundle
    );
    for invalid in [
        serde_json::json!({"kind": "local_bundle", "unexpected": true}),
        serde_json::json!({"kind": "local_bundle", "source_hosts": []}),
        serde_json::json!({
            "kind": "local_bundle",
            "source_hosts": ["models.example.test"]
        }),
        serde_json::json!({"kind": "local_bundle", "source_hosts": null}),
        serde_json::json!({}),
        serde_json::json!({"kind": null}),
    ] {
        assert!(
            serde_json::from_value::<ModelAcquisitionSource>(invalid).is_err(),
            "local-bundle source accepted a missing, null, or forbidden field"
        );
    }

    let network = ModelAcquisitionSource::Network {
        source_hosts: vec!["models.example.test".to_owned()],
    };
    let network_json = serde_json::to_value(&network).expect("serialize network source");
    assert_eq!(
        serde_json::from_value::<ModelAcquisitionSource>(network_json.clone())
            .expect("decode exact network source"),
        network
    );
    for invalid in [
        serde_json::json!({"kind": "network"}),
        serde_json::json!({"kind": "network", "source_hosts": null}),
        serde_json::json!({
            "kind": "network",
            "source_hosts": ["models.example.test"],
            "unexpected": true
        }),
    ] {
        assert!(
            serde_json::from_value::<ModelAcquisitionSource>(invalid).is_err(),
            "network source accepted a missing, null, or forbidden field"
        );
    }
}

#[test]
fn recovery_v4_requires_every_canonical_null_field_to_be_present() {
    let ready = planned(
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
        explicit(RetrievalTopology::FastOnly),
        permissive(),
    );
    let ready_json = serde_json::to_value(ready).expect("serialize ready plan");
    assert!(ready_json["policy"]["acquisition_authorization"].is_null());
    assert!(ready_json["action"].is_null());
    assert!(ready_json["response_contract"]["degradation_reason_code"].is_null());
    serde_json::from_value::<UntrustedRecoveryPlan>(ready_json.clone())
        .expect("explicit canonical nulls decode");

    let mut missing = ready_json.clone();
    missing["policy"]
        .as_object_mut()
        .expect("policy object")
        .remove("acquisition_authorization");
    assert!(
        serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
        "missing policy.acquisition_authorization was treated as explicit null"
    );

    let mut missing = ready_json.clone();
    missing
        .as_object_mut()
        .expect("plan object")
        .remove("action");
    assert!(
        serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
        "missing plan.action was treated as explicit null"
    );

    let mut missing = ready_json;
    missing["response_contract"]
        .as_object_mut()
        .expect("response object")
        .remove("degradation_reason_code");
    assert!(
        serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
        "missing response.degradation_reason_code was treated as explicit null"
    );

    let unavailable = planned(
        SemanticReadiness::IndexAbsent,
        explicit(RetrievalTopology::FastOnly),
        permissive(),
    );
    let unavailable_json = serde_json::to_value(unavailable).expect("serialize unavailable plan");
    assert!(unavailable_json["action"]["required_authorization"].is_null());
    assert!(unavailable_json["response_contract"].is_null());
    serde_json::from_value::<UntrustedRecoveryPlan>(unavailable_json.clone())
        .expect("explicit action/response nulls decode");

    let mut missing = unavailable_json.clone();
    missing["action"]
        .as_object_mut()
        .expect("action object")
        .remove("required_authorization");
    assert!(
        serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
        "missing action.required_authorization was treated as explicit null"
    );

    let mut missing = unavailable_json;
    missing
        .as_object_mut()
        .expect("plan object")
        .remove("response_contract");
    assert!(
        serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
        "missing plan.response_contract was treated as explicit null"
    );
}

#[test]
fn trusted_context_rejects_coherent_plan_and_target_substitution() {
    let state = missing(ModelTier::Quality);
    let request = hybrid(RetrievalTopology::FullProgressive);
    let policy = permissive();
    let acquisition_target = target();

    let ready_forgery = plan(
        SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
        request,
        policy.clone(),
        Some(&acquisition_target),
    )
    .expect("internally coherent ready plan");
    assert!(
        decode_and_validate(
            serde_json::to_value(ready_forgery).expect("serialize forgery"),
            &state,
            request,
            &policy,
            Some(&acquisition_target),
        )
        .is_err(),
        "a coherent state, response, and retryability forgery must not replace probe state"
    );

    let request_forgery = plan(
        state.clone(),
        hybrid(RetrievalTopology::QualityOnly),
        policy.clone(),
        Some(&acquisition_target),
    )
    .expect("internally coherent alternate-request plan");
    assert!(
        decode_and_validate(
            serde_json::to_value(request_forgery).expect("serialize forgery"),
            &state,
            request,
            &policy,
            Some(&acquisition_target),
        )
        .is_err(),
        "a coherent topology and response forgery must not replace the caller request"
    );

    let offline_policy = RecoveryPolicy {
        interaction: InteractionPolicy::Interactive,
        network: NetworkPolicy::Offline,
        acquisition_authorization: None,
    };
    let policy_forgery = plan(
        state.clone(),
        request,
        offline_policy,
        Some(&acquisition_target),
    )
    .expect("internally coherent alternate-policy plan");
    assert!(
        decode_and_validate(
            serde_json::to_value(policy_forgery).expect("serialize forgery"),
            &state,
            request,
            &policy,
            Some(&acquisition_target),
        )
        .is_err(),
        "wire policy must not grant or remove execution capabilities"
    );

    let trusted_coverage = SemanticReadiness::PartialQualityCoverage {
        provenance: VerifiedSemanticProvenance::Local,
        coverage_ppm: 625_000,
    };
    let forged_coverage = SemanticReadiness::PartialQualityCoverage {
        provenance: VerifiedSemanticProvenance::Local,
        coverage_ppm: 700_000,
    };
    let coverage_forgery = plan(forged_coverage, request, policy.clone(), None)
        .expect("internally coherent alternate-coverage plan");
    assert!(
        decode_and_validate(
            serde_json::to_value(coverage_forgery).expect("serialize forgery"),
            &trusted_coverage,
            request,
            &policy,
            None,
        )
        .is_err(),
        "matching state and response coverage substitutions must not replace the census"
    );

    let mut alternate_target = target();
    alternate_target.model_id = "alternate-semantic-model".to_owned();
    alternate_target.embedding_space.logical_model_id = alternate_target.model_id.clone();
    alternate_target.upstream_revision = "alternate-immutable-revision".to_owned();
    alternate_target.embedding_space.immutable_revision =
        alternate_target.upstream_revision.clone();
    let target_forgery = plan(
        state.clone(),
        request,
        policy.clone(),
        Some(&alternate_target),
    )
    .expect("internally coherent alternate-target plan");
    assert!(
        decode_and_validate(
            serde_json::to_value(target_forgery).expect("serialize forgery"),
            &state,
            request,
            &policy,
            Some(&acquisition_target),
        )
        .is_err(),
        "a self-consistent action and authorization cannot substitute another frozen target"
    );
}
