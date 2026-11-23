use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::*;
use crate::collectors::{
    EmbedderTier, EmbeddingStage, EmbeddingStatus, IndexInventory, IndexOperation, IndexStatus,
    LifecycleSeverity, LifecycleState, QuantizationMode, TELEMETRY_SCHEMA_VERSION,
    TelemetryCorrelation, TelemetryEmbedderInfo, TelemetryEmbeddingJob, TelemetryEnvelope,
    TelemetryEvent, TelemetryInstance, TelemetryResourceSample,
};
use crate::error::{SearchError, SearchResult};

#[derive(Debug)]
struct MemoryHostAdapter {
    identity: AdapterIdentity,
    emitted: Arc<Mutex<Vec<TelemetryEnvelope>>>,
    lifecycle: Arc<Mutex<Vec<AdapterLifecycleEvent>>>,
    fail_emit: bool,
    fail_lifecycle: bool,
}

impl MemoryHostAdapter {
    fn new(identity: AdapterIdentity) -> Self {
        Self {
            identity,
            emitted: Arc::new(Mutex::new(Vec::new())),
            lifecycle: Arc::new(Mutex::new(Vec::new())),
            fail_emit: false,
            fail_lifecycle: false,
        }
    }
}

impl HostAdapter for MemoryHostAdapter {
    fn identity(&self) -> AdapterIdentity {
        self.identity.clone()
    }

    fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
        if self.fail_emit {
            return Err(SearchError::InvalidConfig {
                field: "adapter.emit".to_owned(),
                value: "forced_failure".to_owned(),
                reason: "emit failure requested in test".to_owned(),
            });
        }
        self.emitted.lock().unwrap().push(envelope.clone());
        Ok(())
    }

    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
        if self.fail_lifecycle {
            return Err(SearchError::InvalidConfig {
                field: "adapter.lifecycle".to_owned(),
                value: "forced_failure".to_owned(),
                reason: "lifecycle failure requested in test".to_owned(),
            });
        }
        self.lifecycle.lock().unwrap().push(event.clone());
        Ok(())
    }
}

#[derive(Debug, Default)]
struct RecordingSink {
    emitted: Arc<Mutex<Vec<TelemetryEnvelope>>>,
    lifecycle: Arc<Mutex<Vec<AdapterLifecycleEvent>>>,
}

impl AdapterSink for RecordingSink {
    fn emit(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
        self.emitted.lock().unwrap().push(envelope.clone());
        Ok(())
    }

    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
        self.lifecycle.lock().unwrap().push(event.clone());
        Ok(())
    }
}

fn default_identity() -> AdapterIdentity {
    AdapterIdentity {
        adapter_id: "sample-host-adapter".to_owned(),
        adapter_version: "0.1.0".to_owned(),
        host_project: "sample-host".to_owned(),
        runtime_role: Some("query".to_owned()),
        instance_uuid: Some("sample-instance-uuid".to_owned()),
        telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
        redaction_policy_version: "v1".to_owned(),
    }
}

fn load_fixture(name: &str) -> TelemetryEnvelope {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/fixtures")
        .join(name);
    let raw = fs::read_to_string(path).unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn assert_golden_json<T: serde::Serialize>(name: &str, value: &T) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let golden_path = dir.join(format!("{name}.golden.json"));
    let actual = serde_json::to_string_pretty(value).expect("serialize golden json");

    if std::env::var("UPDATE_GOLDENS").is_ok() {
        fs::create_dir_all(&dir).expect("create golden directory");
        fs::write(&golden_path, actual.as_bytes()).expect("write golden file");
        return;
    }

    let expected = fs::read_to_string(&golden_path).unwrap_or_else(|_| {
        panic!(
            "Golden file not found: {}\nSet UPDATE_GOLDENS=1 to create it.",
            golden_path.display()
        );
    });

    let actual_trimmed = actual.trim_end_matches(['\n', '\r']);
    let expected_trimmed = expected.trim_end_matches(['\n', '\r']);

    if actual_trimmed != expected_trimmed {
        let actual_path = golden_path.with_extension("actual.json");
        fs::write(&actual_path, actual_trimmed.as_bytes()).expect("write actual file");
        panic!(
            "GOLDEN MISMATCH: {name}\nexpected: {}\nactual: {}",
            golden_path.display(),
            actual_path.display()
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ViolationSnapshot {
    code: String,
    field: String,
    message: String,
}

fn snapshot_violations(violations: &[ConformanceViolation]) -> Vec<ViolationSnapshot> {
    let mut snapshots: Vec<ViolationSnapshot> = violations
        .iter()
        .map(|violation| ViolationSnapshot {
            code: violation.code.clone(),
            field: violation.field.clone(),
            message: violation.message.clone(),
        })
        .collect();
    snapshots.sort_by(|a, b| {
        a.code
            .cmp(&b.code)
            .then_with(|| a.field.cmp(&b.field))
            .then_with(|| a.message.cmp(&b.message))
    });
    snapshots
}

#[test]
fn sample_host_adapter_passes_conformance_with_golden_fixtures() {
    let harness = ConformanceHarness::default();
    let adapter = MemoryHostAdapter::new(default_identity());
    let fixtures = vec![
        load_fixture("telemetry-search-v1.json"),
        load_fixture("telemetry-embedding-v1.json"),
        load_fixture("telemetry-index-v1.json"),
        load_fixture("telemetry-resource-v1.json"),
        load_fixture("telemetry-lifecycle-v1.json"),
    ];

    let report = harness.run(&adapter, &fixtures);
    assert!(report.passed, "{:?}", report.violations);
    assert_eq!(report.fixtures_checked, fixtures.len());
    assert_eq!(report.emitted_events, fixtures.len());
    assert!(report.lifecycle_hooks_checked >= 2);
    assert!(report.violations.is_empty());
}

#[test]
fn conformance_report_matches_golden_output() {
    let harness = ConformanceHarness::default();
    let adapter = MemoryHostAdapter::new(default_identity());
    let fixtures = vec![
        load_fixture("telemetry-search-v1.json"),
        load_fixture("telemetry-embedding-v1.json"),
        load_fixture("telemetry-index-v1.json"),
        load_fixture("telemetry-resource-v1.json"),
        load_fixture("telemetry-lifecycle-v1.json"),
    ];

    let report = harness.run(&adapter, &fixtures);
    assert_golden_json("adapter_conformance_report_v1", &report);
}

#[test]
fn telemetry_fixtures_pass_envelope_conformance() {
    let harness = ConformanceHarness::default();
    let fixtures = [
        "telemetry-search-v1.json",
        "telemetry-embedding-v1.json",
        "telemetry-index-v1.json",
        "telemetry-resource-v1.json",
        "telemetry-lifecycle-v1.json",
    ];

    for fixture in fixtures {
        let envelope = load_fixture(fixture);
        let violations = harness.validate_envelope(&envelope);
        assert!(
            violations.is_empty(),
            "fixture {fixture} should pass conformance, got {violations:?}"
        );
    }
}

#[test]
fn conformance_violations_match_golden_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");
    envelope.v = 99;
    envelope.ts = String::new();

    if let TelemetryEvent::Search {
        instance,
        correlation,
        query,
        results,
        metrics,
        ..
    } = &mut envelope.event
    {
        instance.instance_id = "not-a-ulid".to_owned();
        instance.project_key = " ".to_owned();
        instance.host_name = String::new();
        instance.pid = Some(1234);

        correlation.event_id = "bad-event".to_owned();
        correlation.root_request_id = "bad-root".to_owned();
        correlation.parent_event_id = Some("bad-parent".to_owned());

        query.text = format!("BEGIN PRIVATE KEY{}", "a".repeat(600));
        metrics.latency_us = 0;
        results.result_count = 7;
        results.lexical_count = 0;
        results.semantic_count = 0;
    } else {
        panic!("search fixture shape changed");
    }

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_search_invalid_v1",
        &snapshot,
    );
}

#[test]
fn conformance_violations_embedding_invalid_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    envelope.event = TelemetryEvent::Embedding {
        instance: TelemetryInstance {
            instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            project_key: "frankensearch".to_owned(),
            host_name: "test-host".to_owned(),
            pid: Some(1234),
        },
        correlation: TelemetryCorrelation {
            event_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            root_request_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            parent_event_id: None,
        },
        job: TelemetryEmbeddingJob {
            job_id: " ".to_owned(), // invalid
            queue_depth: 0,
            doc_count: 0, // invalid
            stage: EmbeddingStage::Fast,
        },
        embedder: TelemetryEmbedderInfo {
            id: String::new(), // invalid
            tier: EmbedderTier::Fast,
            dimension: 0, // invalid
        },
        status: EmbeddingStatus::Completed,
        duration_ms: 0, // invalid
    };

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_embedding_invalid_v1",
        &snapshot,
    );
}

#[test]
fn conformance_violations_index_invalid_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    envelope.event = TelemetryEvent::Index {
        instance: TelemetryInstance {
            instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            project_key: "frankensearch".to_owned(),
            host_name: "test-host".to_owned(),
            pid: Some(1234),
        },
        correlation: TelemetryCorrelation {
            event_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            root_request_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            parent_event_id: None,
        },
        operation: IndexOperation::Build,
        inventory: IndexInventory {
            words: 0,
            tokens: 0,
            lines: 0,
            bytes: 0,
            docs: 0,
        },
        dimension: 0, // invalid
        quantization: QuantizationMode::F16,
        status: IndexStatus::Completed,
        duration_ms: 0, // invalid
    };

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json("adapter_conformance_violations_index_invalid_v1", &snapshot);
}

#[test]
fn conformance_violations_resource_invalid_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    envelope.event = TelemetryEvent::Resource {
        instance: TelemetryInstance {
            instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            project_key: "frankensearch".to_owned(),
            host_name: "test-host".to_owned(),
            pid: Some(1234),
        },
        correlation: TelemetryCorrelation {
            event_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            root_request_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            parent_event_id: None,
        },
        sample: TelemetryResourceSample {
            cpu_pct: 150.0, // invalid
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            interval_ms: 0,          // invalid
            load_avg_1m: Some(-1.0), // invalid
            pressure_profile: None,
        },
    };

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_resource_invalid_v1",
        &snapshot,
    );
}

#[test]
fn conformance_violations_lifecycle_invalid_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    envelope.event = TelemetryEvent::Lifecycle {
        instance: TelemetryInstance {
            instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            project_key: "frankensearch".to_owned(),
            host_name: "test-host".to_owned(),
            pid: Some(1234),
        },
        correlation: TelemetryCorrelation {
            event_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            root_request_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
            parent_event_id: None,
        },
        state: LifecycleState::Started,
        severity: LifecycleSeverity::Info,
        reason: Some(" ".to_owned()), // invalid
        uptime_ms: Some(0),           // invalid
    };

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_lifecycle_invalid_v1",
        &snapshot,
    );
}

#[test]
fn conformance_violations_search_empty_query_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
        query.text = "   ".to_owned();
    } else {
        panic!("search fixture shape changed");
    }

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_search_empty_query_v1",
        &snapshot,
    );
}

#[test]
fn conformance_violations_search_missing_source_counts_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    if let TelemetryEvent::Search { results, .. } = &mut envelope.event {
        results.result_count = 1;
        results.lexical_count = 0;
        results.semantic_count = 0;
    } else {
        panic!("search fixture shape changed");
    }

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_search_missing_source_counts_v1",
        &snapshot,
    );
}

#[test]
fn hash_control_source_counts_satisfy_search_telemetry() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    if let TelemetryEvent::Search { results, .. } = &mut envelope.event {
        results.result_count = 3;
        results.lexical_count = 0;
        results.semantic_count = 0;
        results.hash_control_count = 5;
    } else {
        panic!("search fixture shape changed");
    }

    let violations = harness.validate_envelope(&envelope);
    assert!(
        violations.iter().all(|violation| {
            violation.code != "adapter.event.search.missing_source_counts"
                && violation.code != "adapter.event.search.result_count_exceeds_sources"
        }),
        "hash-control-only results must be a valid source count: {violations:?}"
    );
}

#[test]
fn conformance_violations_privacy_redaction_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
        // query contains a "forbidden" pattern (mocked secret)
        query.text = "find documents with key=sk-1234567890abcdef".to_owned();
    } else {
        panic!("search fixture shape changed");
    }

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_privacy_redaction_v1",
        &snapshot,
    );
}

#[test]
fn conformance_violations_search_filter_fidelity_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    if let TelemetryEvent::Search { results, .. } = &mut envelope.event {
        // Simulate complex results with varied source metadata
        results.result_count = 3;
        results.lexical_count = 1;
        results.semantic_count = 2;
    } else {
        panic!("search fixture shape changed");
    }

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_filter_fidelity_v1",
        &snapshot,
    );
}

#[test]
fn conformance_violations_metadata_security_redaction_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
        // query contains multiple forbidden patterns
        query.text = "find docs with key=sk-123 and token=Bearer abcdef123456".to_owned();
    } else {
        panic!("search fixture shape changed");
    }

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_metadata_redaction_v1",
        &snapshot,
    );
}

#[test]
fn conformance_violations_search_result_count_exceeds_sources_snapshot() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    if let TelemetryEvent::Search { results, .. } = &mut envelope.event {
        results.lexical_count = 1;
        results.semantic_count = 1;
        results.result_count = 5;
    } else {
        panic!("search fixture shape changed");
    }

    let violations = harness.validate_envelope(&envelope);
    let snapshot = snapshot_violations(&violations);
    assert_golden_json(
        "adapter_conformance_violations_search_result_count_exceeds_sources_v1",
        &snapshot,
    );
}

#[test]
fn canonical_host_profiles_emit_expected_identity_handshake() {
    let cases = [
        (
            CanonicalHostProject::CodingAgentSessionSearch,
            "cass-host-adapter",
            "coding_agent_session_search",
        ),
        (CanonicalHostProject::Xf, "xf-host-adapter", "xf"),
        (
            CanonicalHostProject::McpAgentMailRust,
            "mcp-agent-mail-host-adapter",
            "mcp_agent_mail_rust",
        ),
        (
            CanonicalHostProject::Frankenterm,
            "frankenterm-host-adapter",
            "frankenterm",
        ),
    ];

    for (host, expected_adapter_id, expected_project) in cases {
        let adapter =
            ForwardingHostAdapter::for_host(host, "1.2.3", None, Some("instance-xyz".to_owned()));
        let identity = adapter.identity();
        assert_eq!(identity.adapter_id, expected_adapter_id);
        assert_eq!(identity.host_project, expected_project);
        assert_eq!(identity.adapter_version, "1.2.3");
        assert_eq!(identity.telemetry_schema_version, TELEMETRY_SCHEMA_VERSION);
        assert_eq!(identity.redaction_policy_version, "v1");
        assert_eq!(
            identity.runtime_role.as_deref(),
            Some(host.default_runtime_role())
        );
    }
}

#[test]
fn host_project_attribution_matches_alias_boundaries() {
    let cases = [
        (
            None,
            Some("foo-cass-bar"),
            None,
            "coding_agent_session_search",
            85,
            "attribution.telemetry_project_key",
            false,
        ),
        (
            None,
            Some("cassandra"),
            Some("mystery-box"),
            "unknown",
            20,
            "attribution.unknown",
            false,
        ),
        (
            None,
            Some("mcp-agent-mail-runner"),
            None,
            "mcp_agent_mail_rust",
            85,
            "attribution.telemetry_project_key",
            false,
        ),
        (
            None,
            Some("prefixxf"),
            Some("worker"),
            "unknown",
            20,
            "attribution.unknown",
            false,
        ),
        (
            Some("xf"),
            Some("cass-host"),
            Some("coding-agent-session-search"),
            "xf",
            70,
            "attribution.collision",
            true,
        ),
    ];

    for (
        identity,
        telemetry,
        host,
        expected_project,
        expected_confidence,
        expected_reason,
        expected_collision,
    ) in cases
    {
        let attribution = resolve_host_project_attribution(identity, telemetry, host);
        assert_eq!(attribution.resolved_project_key, expected_project);
        assert_eq!(attribution.confidence_score, expected_confidence);
        assert_eq!(attribution.reason_code, expected_reason);
        assert_eq!(attribution.collision, expected_collision);
    }
}

#[test]
fn forwarding_host_adapter_routes_events_to_sink() {
    let sink = Arc::new(RecordingSink::default());
    let sink_trait: Arc<dyn AdapterSink> = sink.clone();
    let adapter = ForwardingHostAdapter::for_cass("0.9.0", None).with_sink(sink_trait);

    let fixture = load_fixture("telemetry-search-v1.json");
    adapter.emit_telemetry(&fixture).unwrap();
    adapter
        .on_lifecycle_event(&AdapterLifecycleEvent::SessionStart {
            ts: "now".to_owned(),
        })
        .unwrap();
    adapter
        .on_lifecycle_event(&AdapterLifecycleEvent::HealthTick {
            ts: "tick".to_owned(),
        })
        .unwrap();
    adapter
        .on_lifecycle_event(&AdapterLifecycleEvent::SessionStop {
            ts: "later".to_owned(),
        })
        .unwrap();

    {
        let emitted = sink.emitted.lock().unwrap();
        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0], fixture);
        drop(emitted);
    }

    {
        let lifecycle = sink.lifecycle.lock().unwrap();
        assert_eq!(lifecycle.len(), 3);
        assert!(matches!(
            lifecycle[0],
            AdapterLifecycleEvent::SessionStart { .. }
        ));
        assert!(matches!(
            lifecycle[1],
            AdapterLifecycleEvent::HealthTick { .. }
        ));
        assert!(matches!(
            lifecycle[2],
            AdapterLifecycleEvent::SessionStop { .. }
        ));
        drop(lifecycle);
    }
}

#[test]
fn forwarding_host_adapter_conformance_passes_for_all_canonical_hosts() {
    let harness = ConformanceHarness::default();
    let fixtures = vec![
        load_fixture("telemetry-search-v1.json"),
        load_fixture("telemetry-embedding-v1.json"),
        load_fixture("telemetry-index-v1.json"),
        load_fixture("telemetry-resource-v1.json"),
        load_fixture("telemetry-lifecycle-v1.json"),
    ];

    for host in CanonicalHostProject::ALL {
        let adapter = ForwardingHostAdapter::for_host(host, "2.0.0", None, None);
        let report = harness.run(&adapter, &fixtures);
        assert!(
            report.passed,
            "host {:?} failed conformance: {:?}",
            host, report.violations
        );
        assert_eq!(report.fixtures_checked, fixtures.len());
        assert_eq!(report.emitted_events, fixtures.len());
        assert!(report.lifecycle_hooks_checked >= 2);
    }
}

#[test]
fn identity_schema_version_mismatch_is_reported() {
    let harness = ConformanceHarness::default();
    let mut identity = default_identity();
    identity.telemetry_schema_version = 9;

    let violations = harness.validate_identity(&identity);
    assert!(violations.iter().any(|violation| {
        violation.code == "adapter.identity.schema_version_mismatch"
            && violation.field == "identity.telemetry_schema_version"
    }));
}

#[test]
fn identity_optional_handshake_fields_require_nonempty_values_when_present() {
    let harness = ConformanceHarness::default();
    let mut identity = default_identity();
    identity.runtime_role = Some("   ".to_owned());
    identity.instance_uuid = Some(String::new());

    let violations = harness.validate_identity(&identity);
    assert!(violations.iter().any(|violation| {
        violation.code == "adapter.identity.empty_runtime_role"
            && violation.field == "identity.runtime_role"
    }));
    assert!(violations.iter().any(|violation| {
        violation.code == "adapter.identity.empty_instance_uuid"
            && violation.field == "identity.instance_uuid"
    }));
}

#[test]
fn redaction_forbidden_pattern_is_detected() {
    let harness = ConformanceHarness::default();
    let mut envelope = load_fixture("telemetry-search-v1.json");

    assert!(
        matches!(envelope.event, TelemetryEvent::Search { .. }),
        "fixture shape changed"
    );

    if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
        query.text = "BEGIN PRIVATE KEY".to_owned();
    }

    let violations = harness.validate_envelope(&envelope);
    assert!(
        violations
            .iter()
            .any(|violation| violation.code == "adapter.redaction.forbidden_pattern")
    );
}
