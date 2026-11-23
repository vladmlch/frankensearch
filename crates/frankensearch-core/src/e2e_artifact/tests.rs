use super::*;

fn make_valid_manifest() -> ManifestBody {
    ManifestBody {
        suite: Suite::Interaction,
        determinism_tier: DeterminismTier::BitExact,
        seed: 42,
        config_hash: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_owned(),
        index_version: Some("fsvi-v3".to_owned()),
        model_versions: vec![ModelVersion {
            name: "potion-128M".to_owned(),
            revision: "abc123".to_owned(),
            digest: None,
        }],
        platform: Platform {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            rustc: "nightly-2026-02-01".to_owned(),
        },
        clock_mode: ClockMode::Simulated,
        tie_break_policy: "doc_id_lexical".to_owned(),
        artifacts: vec![
            ArtifactEntry {
                file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL.to_owned(),
                checksum: "sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
                    .to_owned(),
                line_count: Some(147),
            },
            ArtifactEntry {
                file: E2E_ARTIFACT_ENV_JSON.to_owned(),
                checksum: "sha256:1111111111111111111111111111111111111111111111111111111111111111"
                    .to_owned(),
                line_count: None,
            },
            ArtifactEntry {
                file: E2E_ARTIFACT_REPRO_LOCK.to_owned(),
                checksum: "sha256:2222222222222222222222222222222222222222222222222222222222222222"
                    .to_owned(),
                line_count: None,
            },
        ],
        duration_ms: 1234,
        exit_status: ExitStatus::Pass,
    }
}

fn make_valid_event() -> EventBody {
    EventBody {
        event_type: E2eEventType::OracleCheck,
        correlation: Correlation {
            event_id: "01HQXG5M7QABCDEF12345678AB".to_owned(),
            root_request_id: "01HQXG5M7P3KZFV9N2RSTW6YAB".to_owned(),
            parent_event_id: None,
        },
        severity: E2eSeverity::Info,
        lane_id: Some("baseline".to_owned()),
        oracle_id: Some("ORACLE_NO_DUPLICATES".to_owned()),
        outcome: Some(E2eOutcome::Pass),
        reason_code: None,
        context: Some("0 duplicates in 10 results".to_owned()),
        metrics: None,
    }
}

#[test]
fn manifest_roundtrip() {
    let manifest = make_valid_manifest();

    let envelope = E2eEnvelope::new(
        "e2e-manifest-v1",
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        "2026-02-14T12:00:00Z",
        manifest,
    );

    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: E2eEnvelope<ManifestBody> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, envelope);
    assert_eq!(decoded.v, 1);
    assert_eq!(decoded.schema, "e2e-manifest-v1");
}

#[test]
fn event_roundtrip() {
    let event = make_valid_event();

    let envelope = E2eEnvelope::new(
        "e2e-event-v1",
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        "2026-02-14T12:00:01Z",
        event,
    );

    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: E2eEnvelope<EventBody> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, envelope);
}

#[test]
fn oracle_report_roundtrip() {
    let report = OracleReportBody {
        lanes: vec![LaneReport {
            lane_id: "baseline".to_owned(),
            seed: 3_405_643_776,
            query_count: 5,
            verdicts: vec![
                OracleVerdictRecord {
                    oracle_id: "ORACLE_NO_DUPLICATES".to_owned(),
                    outcome: E2eOutcome::Pass,
                    context: None,
                },
                OracleVerdictRecord {
                    oracle_id: "ORACLE_EXPLAIN_PRESENT".to_owned(),
                    outcome: E2eOutcome::Skip,
                    context: Some("explain disabled".to_owned()),
                },
            ],
            pass_count: 1,
            fail_count: 0,
            skip_count: 1,
            all_passed: true,
        }],
        totals: ReportTotals {
            lanes_run: 1,
            lanes_passed: 1,
            oracles_pass: 1,
            oracles_fail: 0,
            oracles_skip: 1,
            all_passed: true,
        },
    };

    let envelope = E2eEnvelope::new(
        "e2e-oracle-report-v1",
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        "2026-02-14T12:00:02Z",
        report,
    );

    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: E2eEnvelope<OracleReportBody> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, envelope);
}

#[test]
fn replay_roundtrip() {
    let replay = ReplayBody {
        replay_type: ReplayEventType::Query,
        offset_ms: 0,
        seq: 0,
        payload: serde_json::json!({"query_text": "test", "k": 10}),
    };

    let envelope = E2eEnvelope::new(
        "e2e-replay-v1",
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        "2026-02-14T12:00:00Z",
        replay,
    );

    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: E2eEnvelope<ReplayBody> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, envelope);
}

#[test]
fn snapshot_diff_roundtrip() {
    let diff = SnapshotDiffBody {
        comparison_mode: DeterminismTier::Statistical,
        baseline_run_id: "01HQXG4K6N2JYFV8M1QRSV5XZA".to_owned(),
        diffs: vec![DiffEntry {
            field_path: "results[0].score".to_owned(),
            baseline: "0.8765".to_owned(),
            current: "0.8764".to_owned(),
            delta: Some("0.0001".to_owned()),
            within_tolerance: true,
            tolerance: Some("0.001".to_owned()),
        }],
        pass: true,
        mismatch_count: 0,
    };

    let envelope = E2eEnvelope::new(
        "e2e-snapshot-diff-v1",
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        "2026-02-14T12:00:03Z",
        diff,
    );

    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: E2eEnvelope<SnapshotDiffBody> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, envelope);
}

#[test]
fn reason_codes_follow_namespace_pattern() {
    let codes = [
        reason_codes::ORACLE_PASS,
        reason_codes::ORACLE_ORDERING_VIOLATED,
        reason_codes::ORACLE_DUPLICATES_FOUND,
        reason_codes::ORACLE_PHASE_MISMATCH,
        reason_codes::ORACLE_SCORE_NON_MONOTONIC,
        reason_codes::ORACLE_SKIP_FEATURE_DISABLED,
        reason_codes::ORACLE_SKIP_STUB_BACKEND,
        reason_codes::RUN_SETUP_FAILED,
        reason_codes::RUN_TIMEOUT,
        reason_codes::REPLAY_SEED_MISMATCH,
        reason_codes::DIFF_TOLERANCE_EXCEEDED,
        reason_codes::DIFF_FIELD_MISSING,
    ];

    for code in &codes {
        let parts: Vec<&str> = code.split('.').collect();
        assert_eq!(
            parts.len(),
            3,
            "reason code '{code}' must have exactly 3 dot-separated parts"
        );
        assert_eq!(
            parts[0], "e2e",
            "reason code '{code}' must start with 'e2e'"
        );
    }
}

#[test]
fn event_type_serde_matches_schema() {
    let event = EventBody {
        event_type: E2eEventType::LaneStart,
        correlation: Correlation {
            event_id: "01HQXG5M7QABCDEF12345678AB".to_owned(),
            root_request_id: "01HQXG5M7P3KZFV9N2RSTW6YAB".to_owned(),
            parent_event_id: None,
        },
        severity: E2eSeverity::Info,
        lane_id: Some("baseline".to_owned()),
        oracle_id: None,
        outcome: None,
        reason_code: None,
        context: None,
        metrics: None,
    };

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"type\":\"lane_start\""));
    assert!(json.contains("\"severity\":\"info\""));
}

#[test]
fn optional_fields_omitted_when_none() {
    let event = EventBody {
        event_type: E2eEventType::E2eStart,
        correlation: Correlation {
            event_id: "01HQXG5M7QABCDEF12345678AB".to_owned(),
            root_request_id: "01HQXG5M7P3KZFV9N2RSTW6YAB".to_owned(),
            parent_event_id: None,
        },
        severity: E2eSeverity::Info,
        lane_id: None,
        oracle_id: None,
        outcome: None,
        reason_code: None,
        context: None,
        metrics: None,
    };

    let json = serde_json::to_string(&event).unwrap();
    assert!(!json.contains("lane_id"));
    assert!(!json.contains("oracle_id"));
    assert!(!json.contains("outcome"));
    assert!(!json.contains("reason_code"));
    assert!(!json.contains("context"));
    assert!(!json.contains("metrics"));
}

#[test]
fn exit_status_variants_roundtrip() {
    for status in [ExitStatus::Pass, ExitStatus::Fail, ExitStatus::Error] {
        let json = serde_json::to_string(&status).unwrap();
        let decoded: ExitStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, status);
    }
}

#[test]
fn suite_variants_roundtrip() {
    for suite in [Suite::Core, Suite::Fsfs, Suite::Ops, Suite::Interaction] {
        let json = serde_json::to_string(&suite).unwrap();
        let decoded: Suite = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, suite);
    }
}

#[test]
fn manifest_validator_accepts_valid_manifest() {
    assert!(validate_manifest_body(&make_valid_manifest()).is_ok());
}

#[test]
fn manifest_validator_requires_events_stream_artifact() {
    let mut manifest = make_valid_manifest();
    manifest.artifacts = vec![ArtifactEntry {
        file: "oracle-report.json".to_owned(),
        checksum: "sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            .to_owned(),
        line_count: None,
    }];

    let err = validate_manifest_body(&manifest).unwrap_err();
    assert_eq!(
        err,
        E2eArtifactValidationError::MissingRequiredArtifact {
            required_file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
        }
    );
}

#[test]
fn manifest_validator_requires_env_json_artifact() {
    let mut manifest = make_valid_manifest();
    manifest
        .artifacts
        .retain(|artifact| artifact.file != E2E_ARTIFACT_ENV_JSON);

    let err = validate_manifest_body(&manifest).unwrap_err();
    assert_eq!(
        err,
        E2eArtifactValidationError::MissingRequiredArtifact {
            required_file: E2E_ARTIFACT_ENV_JSON,
        }
    );
}

#[test]
fn manifest_validator_requires_repro_lock_artifact() {
    let mut manifest = make_valid_manifest();
    manifest
        .artifacts
        .retain(|artifact| artifact.file != E2E_ARTIFACT_REPRO_LOCK);

    let err = validate_manifest_body(&manifest).unwrap_err();
    assert_eq!(
        err,
        E2eArtifactValidationError::MissingRequiredArtifact {
            required_file: E2E_ARTIFACT_REPRO_LOCK,
        }
    );
}

#[test]
fn manifest_validator_rejects_duplicate_artifact_entries() {
    let mut manifest = make_valid_manifest();
    manifest.artifacts.push(manifest.artifacts[0].clone());

    let err = validate_manifest_body(&manifest).unwrap_err();
    assert!(matches!(
        err,
        E2eArtifactValidationError::DuplicateArtifactEntry { file }
        if file == E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL
    ));
}

#[test]
fn manifest_validator_rejects_line_count_on_non_jsonl() {
    let mut manifest = make_valid_manifest();
    manifest.artifacts.push(ArtifactEntry {
        file: "oracle-report.json".to_owned(),
        checksum: "sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            .to_owned(),
        line_count: Some(1),
    });

    let err = validate_manifest_body(&manifest).unwrap_err();
    assert!(matches!(
        err,
        E2eArtifactValidationError::UnexpectedLineCountForNonJsonl { file }
        if file == "oracle-report.json"
    ));
}

#[test]
fn manifest_validator_requires_failure_artifacts_for_failed_runs() {
    let mut manifest = make_valid_manifest();
    manifest.exit_status = ExitStatus::Fail;

    let err = validate_manifest_body(&manifest).unwrap_err();
    assert_eq!(
        err,
        E2eArtifactValidationError::MissingRequiredArtifact {
            required_file: E2E_ARTIFACT_ARTIFACTS_INDEX_JSON,
        }
    );

    manifest.artifacts.push(ArtifactEntry {
        file: E2E_ARTIFACT_ARTIFACTS_INDEX_JSON.to_owned(),
        checksum: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_owned(),
        line_count: None,
    });
    let err = validate_manifest_body(&manifest).unwrap_err();
    assert_eq!(
        err,
        E2eArtifactValidationError::MissingRequiredArtifact {
            required_file: E2E_ARTIFACT_REPLAY_COMMAND_TXT,
        }
    );

    manifest.artifacts.push(ArtifactEntry {
        file: E2E_ARTIFACT_REPLAY_COMMAND_TXT.to_owned(),
        checksum: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_owned(),
        line_count: None,
    });
    assert!(validate_manifest_body(&manifest).is_ok());
}

#[test]
fn manifest_validator_requires_transcript_for_failed_ops_runs() {
    let mut manifest = make_valid_manifest();
    manifest.suite = Suite::Ops;
    manifest.exit_status = ExitStatus::Error;
    manifest.artifacts.push(ArtifactEntry {
        file: E2E_ARTIFACT_ARTIFACTS_INDEX_JSON.to_owned(),
        checksum: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            .to_owned(),
        line_count: None,
    });
    manifest.artifacts.push(ArtifactEntry {
        file: E2E_ARTIFACT_REPLAY_COMMAND_TXT.to_owned(),
        checksum: "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
            .to_owned(),
        line_count: None,
    });

    let err = validate_manifest_body(&manifest).unwrap_err();
    assert_eq!(
        err,
        E2eArtifactValidationError::MissingRequiredArtifact {
            required_file: E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT,
        }
    );

    manifest.artifacts.push(ArtifactEntry {
        file: E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT.to_owned(),
        checksum: "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
            .to_owned(),
        line_count: None,
    });
    assert!(validate_manifest_body(&manifest).is_ok());
}

#[test]
fn event_validator_accepts_pass_without_reason_code() {
    assert!(validate_event_body(&make_valid_event()).is_ok());
}

#[test]
fn event_validator_requires_reason_code_for_failures() {
    let mut event = make_valid_event();
    event.outcome = Some(E2eOutcome::Fail);
    event.reason_code = None;

    let err = validate_event_body(&event).unwrap_err();
    assert_eq!(
        err,
        E2eArtifactValidationError::MissingReasonCode {
            outcome: E2eOutcome::Fail,
        }
    );
}

#[test]
fn event_validator_rejects_invalid_reason_code_pattern() {
    let mut event = make_valid_event();
    event.outcome = Some(E2eOutcome::Skip);
    event.reason_code = Some("E2E.INVALID.CODE".to_owned());

    let err = validate_event_body(&event).unwrap_err();
    assert!(matches!(
        err,
        E2eArtifactValidationError::InvalidReasonCode { reason_code }
        if reason_code == "E2E.INVALID.CODE"
    ));
}

#[test]
fn envelope_validator_rejects_invalid_run_id() {
    let envelope = E2eEnvelope::new(
        E2E_SCHEMA_MANIFEST,
        "not-a-ulid",
        "2026-02-14T12:00:00Z",
        make_valid_manifest(),
    );
    let err = validate_manifest_envelope(&envelope).unwrap_err();
    assert!(matches!(
        err,
        E2eArtifactValidationError::InvalidRunId { run_id } if run_id == "not-a-ulid"
    ));
}

#[test]
fn envelope_validator_rejects_schema_tag_mismatch() {
    let envelope = E2eEnvelope::new(
        E2E_SCHEMA_EVENT,
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        "2026-02-14T12:00:00Z",
        make_valid_manifest(),
    );
    let err = validate_manifest_envelope(&envelope).unwrap_err();
    assert!(matches!(
        err,
        E2eArtifactValidationError::SchemaTagMismatch { expected, found }
        if expected == E2E_SCHEMA_MANIFEST && found == E2E_SCHEMA_EVENT
    ));
}

#[test]
fn envelope_validator_accepts_canonical_rfc3339_utc_and_round_trips() {
    let ts = "2026-02-14T12:00:00Z";
    parse_canonical_rfc3339_utc(ts).expect("canonical UTC instant");
    let envelope = E2eEnvelope::new(
        E2E_SCHEMA_MANIFEST,
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        ts,
        make_valid_manifest(),
    );
    validate_manifest_envelope(&envelope).expect("canonical ts must admit");
    assert_eq!(envelope.ts, ts);
    assert!(
        parse_canonical_rfc3339_utc("2024-02-29T00:00:00Z").is_ok(),
        "Gregorian leap day must admit"
    );
    assert!(
        parse_canonical_rfc3339_utc("2000-02-29T23:59:59Z").is_ok(),
        "century leap year must admit"
    );
}

#[test]
fn envelope_validator_rejects_hostile_timestamps() {
    let cases = [
        "2026-02-14T12:00:00+00:00",
        "2026-02-14T12:00:00-00:00",
        "2026-02-14T12:00:00.000Z",
        "2026-02-14T12:00:00.0Z",
        "2026-02-14T12:00:00z",
        "2026-02-14 12:00:00Z",
        "2026-02-14T12:00:00Z ",
        " 2026-02-14T12:00:00Z",
        "2026-02-14T12:00:00Ztrail",
        "2026-02-30T12:00:00Z",
        "2026-13-01T12:00:00Z",
        "2026-00-01T12:00:00Z",
        "2026-02-14T24:00:00Z",
        "2026-02-14T12:60:00Z",
        "2026-02-14T12:00:60Z",
        "2025-02-29T12:00:00Z",
        "1900-02-29T12:00:00Z",
        "2026-04-31T12:00:00Z",
        "2026-02-14T12:00:00",
        "",
    ];
    for ts in cases {
        let envelope = E2eEnvelope::new(
            E2E_SCHEMA_MANIFEST,
            "01HQXG5M7P3KZFV9N2RSTW6YAB",
            ts,
            make_valid_manifest(),
        );
        let err = validate_manifest_envelope(&envelope).unwrap_err();
        assert!(
            matches!(
                err,
                E2eArtifactValidationError::InvalidTimestamp { ts: ref rejected }
                if rejected == ts
            ),
            "ts {ts:?} must fail closed, got {err:?}"
        );
        assert!(
            err.to_string().contains("ts"),
            "error must name ts, got {err}"
        );
    }
}

#[test]
fn manifest_validator_rejects_hostile_config_hashes() {
    let cases = [
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256:E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85",
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855ff",
        "sha256:",
        "",
        "privacy-policy:1",
        "tui.replay.sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "md5:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85g",
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n",
    ];
    for config_hash in cases {
        let mut manifest = make_valid_manifest();
        manifest.config_hash = config_hash.to_owned();
        let err = validate_manifest_body(&manifest).unwrap_err();
        assert!(
            matches!(
                err,
                E2eArtifactValidationError::InvalidConfigHash {
                    config_hash: ref rejected
                } if rejected == config_hash
            ),
            "config_hash {config_hash:?} must fail closed, got {err:?}"
        );
        assert!(
            err.to_string().contains("config_hash"),
            "error must name config_hash, got {err}"
        );
    }
}

#[test]
fn normalize_artifact_file_name_maps_legacy_aliases() {
    assert_eq!(
        normalize_artifact_file_name("legacy/run_manifest.json"),
        E2E_ARTIFACT_MANIFEST_JSON
    );
    assert_eq!(
        normalize_artifact_file_name("events.jsonl"),
        E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL
    );
    assert_eq!(
        normalize_artifact_file_name("artifacts-index.json"),
        E2E_ARTIFACT_ARTIFACTS_INDEX_JSON
    );
    assert_eq!(
        normalize_artifact_file_name("replay.txt"),
        E2E_ARTIFACT_REPLAY_COMMAND_TXT
    );
}

#[test]
fn normalize_replay_command_collapses_outer_whitespace_only() {
    let normalized = normalize_replay_command(
        "  cargo   test   -p frankensearch-fsfs -- --exact \"scenario  with  spaces\"  ",
    );
    assert_eq!(
        normalized,
        "cargo test -p frankensearch-fsfs -- --exact \"scenario  with  spaces\""
    );
}

#[test]
fn build_artifact_entries_is_sorted_and_uses_sha256_format() {
    let entries = build_artifact_entries([
        ArtifactEmissionInput {
            file: "replay.txt",
            bytes: b"cargo test -p frankensearch-fsfs -- --exact scenario_cli_degrade_path\n",
            line_count: None,
        },
        ArtifactEmissionInput {
            file: "events.jsonl",
            bytes: b"{\"line\":1}\n{\"line\":2}\n",
            line_count: Some(2),
        },
        ArtifactEmissionInput {
            file: "artifacts-index.json",
            bytes: b"[]",
            line_count: None,
        },
    ])
    .expect("entries should build");

    let files: Vec<&str> = entries.iter().map(|entry| entry.file.as_str()).collect();
    assert_eq!(
        files,
        vec![
            E2E_ARTIFACT_ARTIFACTS_INDEX_JSON,
            E2E_ARTIFACT_REPLAY_COMMAND_TXT,
            E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
        ]
    );
    assert!(
        entries
            .iter()
            .all(|entry| entry.checksum.starts_with("sha256:") && entry.checksum.len() == 71)
    );
}

#[test]
fn build_artifact_entries_rejects_duplicates_after_name_normalization() {
    let err = build_artifact_entries([
        ArtifactEmissionInput {
            file: "events.jsonl",
            bytes: b"{}\n",
            line_count: Some(1),
        },
        ArtifactEmissionInput {
            file: "structured_events.jsonl",
            bytes: b"{}\n",
            line_count: Some(1),
        },
    ])
    .expect_err("duplicate canonical name must fail");
    assert!(matches!(
        err,
        E2eArtifactEmitterError::DuplicateNormalizedArtifactFile { file }
            if file == E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL
    ));
}

#[test]
fn build_artifact_entries_enforces_jsonl_line_count_contract() {
    let missing_line_count = build_artifact_entries([ArtifactEmissionInput {
        file: "events.jsonl",
        bytes: b"{}\n",
        line_count: None,
    }])
    .expect_err("jsonl without line_count must fail");
    assert!(matches!(
        missing_line_count,
        E2eArtifactEmitterError::MissingLineCountForJsonl { file }
            if file == E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL
    ));

    let unexpected_line_count = build_artifact_entries([ArtifactEmissionInput {
        file: "replay_command.txt",
        bytes: b"cargo test -p frankensearch-fsfs\n",
        line_count: Some(1),
    }])
    .expect_err("non-jsonl with line_count must fail");
    assert!(matches!(
        unexpected_line_count,
        E2eArtifactEmitterError::UnexpectedLineCountForNonJsonl { file }
            if file == E2E_ARTIFACT_REPLAY_COMMAND_TXT
    ));
}

#[test]
fn render_artifacts_index_is_deterministically_sorted() {
    let entries = vec![
        ArtifactEntry {
            file: E2E_ARTIFACT_REPLAY_COMMAND_TXT.to_owned(),
            checksum: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                .to_owned(),
            line_count: None,
        },
        ArtifactEntry {
            file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL.to_owned(),
            checksum: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .to_owned(),
            line_count: Some(2),
        },
    ];
    let rendered = render_artifacts_index(&entries).expect("render must succeed");
    let replay_pos = rendered
        .find(E2E_ARTIFACT_REPLAY_COMMAND_TXT)
        .expect("replay file present");
    let events_pos = rendered
        .find(E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL)
        .expect("events file present");
    assert!(replay_pos < events_pos);
}

#[test]
fn core_manifest_artifacts_for_failed_run_are_v1_compliant() {
    let events = vec![
        E2eEnvelope::new(
            E2E_SCHEMA_EVENT,
            "01HQXG5M7P3KZFV9N2RSTW6YAB",
            "2026-02-14T12:00:01Z",
            EventBody {
                event_type: E2eEventType::E2eStart,
                correlation: Correlation {
                    event_id: "evt-start".to_owned(),
                    root_request_id: "root-1".to_owned(),
                    parent_event_id: None,
                },
                severity: E2eSeverity::Info,
                lane_id: None,
                oracle_id: None,
                outcome: None,
                reason_code: Some(reason_codes::RUN_SETUP_FAILED.to_owned()),
                context: Some("starting core lane".to_owned()),
                metrics: None,
            },
        ),
        E2eEnvelope::new(
            E2E_SCHEMA_EVENT,
            "01HQXG5M7P3KZFV9N2RSTW6YAB",
            "2026-02-14T12:00:02Z",
            EventBody {
                event_type: E2eEventType::E2eEnd,
                correlation: Correlation {
                    event_id: "evt-end".to_owned(),
                    root_request_id: "root-1".to_owned(),
                    parent_event_id: Some("evt-start".to_owned()),
                },
                severity: E2eSeverity::Warn,
                lane_id: None,
                oracle_id: None,
                outcome: Some(E2eOutcome::Fail),
                reason_code: Some(reason_codes::RUN_SETUP_FAILED.to_owned()),
                context: Some("core lane failed".to_owned()),
                metrics: None,
            },
        ),
    ];

    let mut events_jsonl = String::new();
    for event in &events {
        events_jsonl.push_str(&serde_json::to_string(event).expect("serialize event"));
        events_jsonl.push('\n');
    }

    let (artifacts, artifacts_index_json) = build_core_manifest_artifacts(
        events_jsonl.as_bytes(),
        u64::try_from(events.len()).expect("line count within u64"),
        ExitStatus::Fail,
        Some("  cargo   run --example validate_full_pipeline  "),
    )
    .expect("core failed-run artifacts should build");

    let artifact_files: Vec<&str> = artifacts.iter().map(|entry| entry.file.as_str()).collect();
    assert!(artifact_files.contains(&E2E_ARTIFACT_ENV_JSON));
    assert!(artifact_files.contains(&E2E_ARTIFACT_REPRO_LOCK));
    assert!(artifact_files.contains(&E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL));
    assert!(artifact_files.contains(&E2E_ARTIFACT_REPLAY_COMMAND_TXT));
    assert!(artifact_files.contains(&E2E_ARTIFACT_ARTIFACTS_INDEX_JSON));
    assert!(artifacts_index_json.is_some());

    let mut manifest = make_valid_manifest();
    manifest.suite = Suite::Core;
    manifest.exit_status = ExitStatus::Fail;
    manifest.artifacts = artifacts;
    let manifest_envelope = E2eEnvelope::new(
        E2E_SCHEMA_MANIFEST,
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        "2026-02-14T12:00:03Z",
        manifest,
    );

    assert!(validate_manifest_envelope(&manifest_envelope).is_ok());
    for event in &events {
        assert!(validate_event_envelope(event).is_ok());
    }
}

#[test]
fn core_manifest_artifacts_for_pass_run_omit_failure_only_files() {
    let structured_events_jsonl = b"{\"v\":1}\n";
    let (artifacts, artifacts_index_json) =
        build_core_manifest_artifacts(structured_events_jsonl, 1, ExitStatus::Pass, None)
            .expect("pass-run artifacts should build");

    assert!(artifacts_index_json.is_none());
    let artifact_files: Vec<&str> = artifacts.iter().map(|entry| entry.file.as_str()).collect();
    assert_eq!(
        artifact_files,
        vec![
            E2E_ARTIFACT_ENV_JSON,
            E2E_ARTIFACT_REPRO_LOCK,
            E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
        ]
    );
}

#[test]
fn core_manifest_artifacts_require_replay_command_for_failure() {
    let err = build_core_manifest_artifacts(b"{}\n", 1, ExitStatus::Fail, None)
        .expect_err("failure artifacts without replay command must fail");
    assert_eq!(err, E2eArtifactEmitterError::MissingReplayCommandForFailure);
}

// ─── bd-1o5g tests begin ───

#[test]
fn schema_constants_have_expected_values() {
    assert_eq!(E2E_SCHEMA_VERSION, 1);
    assert_eq!(E2E_SCHEMA_MANIFEST, "e2e-manifest-v1");
    assert_eq!(E2E_SCHEMA_EVENT, "e2e-event-v1");
    assert_eq!(E2E_SCHEMA_ORACLE_REPORT, "e2e-oracle-report-v1");
    assert_eq!(E2E_SCHEMA_REPLAY, "e2e-replay-v1");
    assert_eq!(E2E_SCHEMA_SNAPSHOT_DIFF, "e2e-snapshot-diff-v1");
    assert_eq!(E2E_ARTIFACT_REPLAY_JSONL, "replay.jsonl");
}

#[test]
fn validation_error_display_all_variants() {
    let errors = vec![
        E2eArtifactValidationError::UnsupportedSchemaVersion {
            expected: 1,
            found: 2,
        },
        E2eArtifactValidationError::SchemaTagMismatch {
            expected: "e2e-manifest-v1",
            found: "wrong".into(),
        },
        E2eArtifactValidationError::InvalidRunId {
            run_id: "bad".into(),
        },
        E2eArtifactValidationError::InvalidTimestamp {
            ts: "not-a-timestamp".into(),
        },
        E2eArtifactValidationError::InvalidConfigHash {
            config_hash: "bare-digest".into(),
        },
        E2eArtifactValidationError::MissingRequiredArtifact {
            required_file: "test.json",
        },
        E2eArtifactValidationError::DuplicateArtifactEntry {
            file: "dup.json".into(),
        },
        E2eArtifactValidationError::MissingLineCountForJsonl {
            file: "test.jsonl".into(),
        },
        E2eArtifactValidationError::UnexpectedLineCountForNonJsonl {
            file: "test.json".into(),
        },
        E2eArtifactValidationError::MissingLaneId {
            event_type: E2eEventType::LaneStart,
        },
        E2eArtifactValidationError::MissingOracleFields,
        E2eArtifactValidationError::MissingReasonCode {
            outcome: E2eOutcome::Fail,
        },
        E2eArtifactValidationError::InvalidReasonCode {
            reason_code: "bad".into(),
        },
    ];
    for err in &errors {
        let display = err.to_string();
        assert!(!display.is_empty());
        let cloned = err.clone();
        assert_eq!(err, &cloned);
    }
}

#[test]
fn emitter_error_display_all_variants() {
    let errors = vec![
        E2eArtifactEmitterError::DuplicateNormalizedArtifactFile {
            file: "dup.json".into(),
        },
        E2eArtifactEmitterError::MissingLineCountForJsonl {
            file: "test.jsonl".into(),
        },
        E2eArtifactEmitterError::UnexpectedLineCountForNonJsonl {
            file: "test.json".into(),
        },
        E2eArtifactEmitterError::ArtifactsIndexRender {
            detail: "bad json".into(),
        },
        E2eArtifactEmitterError::MissingReplayCommandForFailure,
    ];
    for err in &errors {
        let display = err.to_string();
        assert!(!display.is_empty());
        let cloned = err.clone();
        assert_eq!(err, &cloned);
    }
}

#[test]
fn artifact_emission_input_traits() {
    let input = ArtifactEmissionInput {
        file: "test.json",
        bytes: b"hello",
        line_count: None,
    };
    let copied = input; // Copy
    assert_eq!(input, copied);
    let dbg = format!("{input:?}");
    assert!(dbg.contains("ArtifactEmissionInput"));
}

#[test]
fn envelope_validator_rejects_unsupported_version() {
    let mut envelope = E2eEnvelope::new(
        E2E_SCHEMA_MANIFEST,
        "01HQXG5M7P3KZFV9N2RSTW6YAB",
        "2026-02-14T12:00:00Z",
        make_valid_manifest(),
    );
    envelope.v = 99;
    let err = validate_manifest_envelope(&envelope).unwrap_err();
    assert!(matches!(
        err,
        E2eArtifactValidationError::UnsupportedSchemaVersion {
            expected: 1,
            found: 99,
        }
    ));
}

#[test]
fn determinism_tier_all_variants_serde() {
    for tier in [
        DeterminismTier::BitExact,
        DeterminismTier::Semantic,
        DeterminismTier::Statistical,
    ] {
        let json = serde_json::to_string(&tier).unwrap();
        let decoded: DeterminismTier = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, tier);
    }
}

#[test]
fn clock_mode_all_variants_serde() {
    for mode in [ClockMode::Simulated, ClockMode::Frozen, ClockMode::Realtime] {
        let json = serde_json::to_string(&mode).unwrap();
        let decoded: ClockMode = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, mode);
    }
}

#[test]
fn replay_event_type_all_variants_serde() {
    for t in [
        ReplayEventType::Query,
        ReplayEventType::ConfigChange,
        ReplayEventType::ClockAdvance,
        ReplayEventType::Signal,
    ] {
        let json = serde_json::to_string(&t).unwrap();
        let decoded: ReplayEventType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, t);
    }
}

#[test]
fn e2e_event_type_all_variants_serde() {
    for t in [
        E2eEventType::E2eStart,
        E2eEventType::E2eEnd,
        E2eEventType::LaneStart,
        E2eEventType::LaneEnd,
        E2eEventType::OracleCheck,
        E2eEventType::PhaseTransition,
        E2eEventType::Assertion,
    ] {
        let json = serde_json::to_string(&t).unwrap();
        let decoded: E2eEventType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, t);
    }
}

#[test]
fn e2e_severity_all_variants_serde() {
    for s in [E2eSeverity::Info, E2eSeverity::Warn, E2eSeverity::Error] {
        let json = serde_json::to_string(&s).unwrap();
        let decoded: E2eSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, s);
    }
}

#[test]
fn e2e_outcome_all_variants_serde() {
    for o in [E2eOutcome::Pass, E2eOutcome::Fail, E2eOutcome::Skip] {
        let json = serde_json::to_string(&o).unwrap();
        let decoded: E2eOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, o);
    }
}

#[test]
fn normalize_artifact_file_name_passthrough_unknown() {
    assert_eq!(
        normalize_artifact_file_name("custom_report.json"),
        "custom_report.json"
    );
    assert_eq!(
        normalize_artifact_file_name("  some/path/to/data.bin  "),
        "data.bin"
    );
}

#[test]
fn normalize_replay_command_strips_backticks() {
    let result = normalize_replay_command("`cargo test --release`");
    assert_eq!(result, "cargo test --release");
}

#[test]
fn normalize_replay_command_preserves_single_quoted_whitespace() {
    let result = normalize_replay_command("cargo test 'with   internal   spaces' --flag");
    assert_eq!(result, "cargo test 'with   internal   spaces' --flag");
}

#[test]
fn sha256_checksum_deterministic_and_format() {
    let hash1 = sha256_checksum(b"hello world");
    let hash2 = sha256_checksum(b"hello world");
    assert_eq!(hash1, hash2);
    assert!(hash1.starts_with("sha256:"));
    assert_eq!(hash1.len(), 71); // "sha256:" (7) + 64 hex chars

    let empty_hash = sha256_checksum(b"");
    assert!(empty_hash.starts_with("sha256:"));
    assert_ne!(hash1, empty_hash);
}

#[test]
fn event_validator_requires_lane_id_for_lane_start() {
    let mut event = make_valid_event();
    event.event_type = E2eEventType::LaneStart;
    event.lane_id = None;
    event.oracle_id = None;
    event.outcome = None;
    let err = validate_event_body(&event).unwrap_err();
    assert!(matches!(
        err,
        E2eArtifactValidationError::MissingLaneId {
            event_type: E2eEventType::LaneStart
        }
    ));
}

#[test]
fn event_validator_requires_lane_id_for_lane_end() {
    let mut event = make_valid_event();
    event.event_type = E2eEventType::LaneEnd;
    event.lane_id = Some(String::new()); // empty string
    event.oracle_id = None;
    event.outcome = None;
    let err = validate_event_body(&event).unwrap_err();
    assert!(matches!(
        err,
        E2eArtifactValidationError::MissingLaneId {
            event_type: E2eEventType::LaneEnd
        }
    ));
}

#[test]
fn event_validator_requires_oracle_fields() {
    let mut event = make_valid_event();
    event.event_type = E2eEventType::OracleCheck;
    event.oracle_id = None; // missing oracle_id
    let err = validate_event_body(&event).unwrap_err();
    assert_eq!(err, E2eArtifactValidationError::MissingOracleFields);

    let mut event2 = make_valid_event();
    event2.event_type = E2eEventType::OracleCheck;
    event2.oracle_id = Some("oracle-1".into());
    event2.outcome = None; // missing outcome
    let err2 = validate_event_body(&event2).unwrap_err();
    assert_eq!(err2, E2eArtifactValidationError::MissingOracleFields);
}

#[test]
fn event_validator_requires_reason_code_for_skip() {
    let mut event = make_valid_event();
    event.outcome = Some(E2eOutcome::Skip);
    event.reason_code = None;
    let err = validate_event_body(&event).unwrap_err();
    assert_eq!(
        err,
        E2eArtifactValidationError::MissingReasonCode {
            outcome: E2eOutcome::Skip,
        }
    );
}

#[test]
fn event_validator_accepts_valid_reason_code() {
    let mut event = make_valid_event();
    event.outcome = Some(E2eOutcome::Fail);
    event.reason_code = Some("e2e.oracle.ordering_violated".to_owned());
    assert!(validate_event_body(&event).is_ok());
}

#[test]
fn reason_code_validation_edge_cases() {
    // Too few segments
    assert!(!is_valid_reason_code("e2e.oracle"));
    // Too many segments
    assert!(!is_valid_reason_code("e2e.oracle.pass.extra"));
    // Empty segment
    assert!(!is_valid_reason_code("e2e..pass"));
    // Uppercase
    assert!(!is_valid_reason_code("E2E.oracle.pass"));
    // Valid
    assert!(is_valid_reason_code("e2e.oracle.pass"));
    // Underscores allowed in category and code
    assert!(is_valid_reason_code("e2e.oracle_check.ordering_violated"));
    // Underscore NOT allowed in namespace
    assert!(!is_valid_reason_code("e2e_ns.oracle.pass"));
}

// ─── bd-1o5g tests end ───
