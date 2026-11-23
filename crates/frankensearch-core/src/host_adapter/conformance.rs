use serde::{Deserialize, Serialize};

use super::DEFAULT_REDACTION_FORBIDDEN_PATTERNS;
use super::adapter::{AdapterLifecycleEvent, HostAdapter};
use super::identity::{AdapterIdentity, CanonicalHostProject};
use crate::collectors::{
    TELEMETRY_SCHEMA_VERSION, TelemetryCorrelation, TelemetryEmbedderInfo, TelemetryEmbeddingJob,
    TelemetryEnvelope, TelemetryEvent, TelemetryInstance, TelemetrySearchMetrics,
    TelemetrySearchResults,
};
use crate::error::SearchError;

/// One machine-readable conformance violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceViolation {
    /// Stable violation code.
    pub code: String,
    /// Field/path associated with the violation.
    pub field: String,
    /// Human-readable diagnostic.
    pub message: String,
}

/// Summary result for an adapter conformance run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceReport {
    /// Whether all checks passed.
    pub passed: bool,
    /// Number of fixture events validated.
    pub fixtures_checked: usize,
    /// Number of successful adapter `emit_telemetry` calls.
    pub emitted_events: usize,
    /// Number of successful lifecycle hook invocations.
    pub lifecycle_hooks_checked: usize,
    /// Validation failures.
    pub violations: Vec<ConformanceViolation>,
}

impl ConformanceReport {
    const fn with_violations(
        fixtures_checked: usize,
        emitted_events: usize,
        lifecycle_hooks_checked: usize,
        violations: Vec<ConformanceViolation>,
    ) -> Self {
        Self {
            passed: violations.is_empty(),
            fixtures_checked,
            emitted_events,
            lifecycle_hooks_checked,
            violations,
        }
    }
}

/// Configuration for adapter conformance checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceConfig {
    /// Required telemetry schema version.
    pub expected_schema_version: u8,
    /// Required redaction policy version label.
    pub required_redaction_policy_version: String,
    /// Forbidden substrings that must not appear in serialized telemetry payloads.
    pub forbidden_substrings: Vec<String>,
}

impl Default for ConformanceConfig {
    fn default() -> Self {
        Self {
            expected_schema_version: TELEMETRY_SCHEMA_VERSION,
            required_redaction_policy_version: "v1".to_owned(),
            forbidden_substrings: DEFAULT_REDACTION_FORBIDDEN_PATTERNS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }
}

/// Conformance harness for host adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceHarness {
    config: ConformanceConfig,
}

impl Default for ConformanceHarness {
    fn default() -> Self {
        Self::new(ConformanceConfig::default())
    }
}

impl ConformanceHarness {
    /// Construct a harness with explicit configuration.
    #[must_use]
    pub const fn new(config: ConformanceConfig) -> Self {
        Self { config }
    }

    /// Access harness configuration.
    #[must_use]
    pub const fn config(&self) -> &ConformanceConfig {
        &self.config
    }

    /// Validate adapter identity handshake.
    #[must_use]
    pub fn validate_identity(&self, identity: &AdapterIdentity) -> Vec<ConformanceViolation> {
        let mut violations = Vec::new();

        if identity.adapter_id.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_adapter_id",
                "identity.adapter_id",
                "adapter_id must be non-empty",
            ));
        }
        if identity.adapter_version.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_adapter_version",
                "identity.adapter_version",
                "adapter_version must be non-empty",
            ));
        }
        if identity.host_project.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_host_project",
                "identity.host_project",
                "host_project must be non-empty",
            ));
        }
        if identity
            .runtime_role
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            violations.push(violation(
                "adapter.identity.empty_runtime_role",
                "identity.runtime_role",
                "runtime_role must be non-empty when provided",
            ));
        }
        if identity
            .instance_uuid
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            violations.push(violation(
                "adapter.identity.empty_instance_uuid",
                "identity.instance_uuid",
                "instance_uuid must be non-empty when provided",
            ));
        }

        if identity.telemetry_schema_version != self.config.expected_schema_version {
            violations.push(ConformanceViolation {
                code: "adapter.identity.schema_version_mismatch".to_owned(),
                field: "identity.telemetry_schema_version".to_owned(),
                message: format!(
                    "expected schema v{}, got v{}",
                    self.config.expected_schema_version, identity.telemetry_schema_version
                ),
            });
        }

        if identity.redaction_policy_version != self.config.required_redaction_policy_version {
            violations.push(ConformanceViolation {
                code: "adapter.identity.redaction_policy_mismatch".to_owned(),
                field: "identity.redaction_policy_version".to_owned(),
                message: format!(
                    "expected redaction policy {}, got {}",
                    self.config.required_redaction_policy_version,
                    identity.redaction_policy_version
                ),
            });
        }

        if let Some(violation) =
            canonical_identity_pair_violation(&identity.adapter_id, &identity.host_project)
        {
            violations.push(violation);
        }

        violations
    }

    /// Validate one telemetry envelope for schema/version and redaction compliance.
    #[must_use]
    pub fn validate_envelope(&self, envelope: &TelemetryEnvelope) -> Vec<ConformanceViolation> {
        let mut violations = Vec::new();

        if envelope.v != self.config.expected_schema_version {
            violations.push(ConformanceViolation {
                code: "adapter.envelope.schema_version_mismatch".to_owned(),
                field: "envelope.v".to_owned(),
                message: format!(
                    "expected schema v{}, got v{}",
                    self.config.expected_schema_version, envelope.v
                ),
            });
        }

        if envelope.ts.trim().is_empty() {
            violations.push(violation(
                "adapter.envelope.missing_timestamp",
                "envelope.ts",
                "timestamp must be non-empty",
            ));
        }

        match &envelope.event {
            TelemetryEvent::Search {
                instance,
                correlation,
                query,
                results,
                metrics,
                ..
            } => validate_search_event(
                instance,
                correlation,
                query.text.as_str(),
                results,
                metrics,
                &mut violations,
            ),
            TelemetryEvent::Embedding {
                instance,
                correlation,
                job,
                embedder,
                duration_ms,
                ..
            } => validate_embedding_event(
                instance,
                correlation,
                job,
                embedder,
                *duration_ms,
                &mut violations,
            ),
            TelemetryEvent::Index {
                instance,
                correlation,
                dimension,
                duration_ms,
                ..
            } => validate_index_event(
                instance,
                correlation,
                *dimension,
                *duration_ms,
                &mut violations,
            ),
            TelemetryEvent::Resource {
                instance,
                correlation,
                sample,
            } => {
                validate_resource_event(
                    instance,
                    correlation,
                    sample.cpu_pct,
                    sample.interval_ms,
                    sample.load_avg_1m,
                    &mut violations,
                );
            }
            TelemetryEvent::Lifecycle {
                instance,
                correlation,
                reason,
                uptime_ms,
                ..
            } => validate_lifecycle_event(
                instance,
                correlation,
                reason.as_deref(),
                *uptime_ms,
                &mut violations,
            ),
        }

        violations.extend(validate_redaction_forbidden_substrings(
            envelope,
            &self.config.forbidden_substrings,
        ));

        violations
    }

    /// Execute a full conformance run over identity, lifecycle hooks, and fixture envelopes.
    ///
    /// Lifecycle order:
    /// 1. `SessionStart`
    /// 2. per-envelope `HealthTick`
    /// 3. `SessionStop`
    #[must_use]
    pub fn run(
        &self,
        adapter: &dyn HostAdapter,
        fixtures: &[TelemetryEnvelope],
    ) -> ConformanceReport {
        let identity = adapter.identity();
        let mut violations = self.validate_identity(&identity);
        let mut lifecycle_hooks_checked = 0usize;
        let mut emitted_events = 0usize;

        let start_event = AdapterLifecycleEvent::SessionStart {
            ts: "conformance-start".to_owned(),
        };
        match adapter.on_lifecycle_event(&start_event) {
            Ok(()) => lifecycle_hooks_checked += 1,
            Err(err) => violations.push(adapter_error_violation("lifecycle_start", &err)),
        }

        for (idx, envelope) in fixtures.iter().enumerate() {
            let event_violations = self.validate_envelope(envelope);
            if !event_violations.is_empty() {
                for violation in event_violations {
                    violations.push(ConformanceViolation {
                        code: violation.code,
                        field: format!("fixtures[{idx}].{}", violation.field),
                        message: violation.message,
                    });
                }
            }

            match adapter.emit_telemetry(envelope) {
                Ok(()) => emitted_events += 1,
                Err(err) => violations.push(adapter_error_violation("emit_telemetry", &err)),
            }

            let tick_event = AdapterLifecycleEvent::HealthTick {
                ts: format!("conformance-tick-{idx}"),
            };
            match adapter.on_lifecycle_event(&tick_event) {
                Ok(()) => lifecycle_hooks_checked += 1,
                Err(err) => violations.push(adapter_error_violation("lifecycle_tick", &err)),
            }
        }

        let stop_event = AdapterLifecycleEvent::SessionStop {
            ts: "conformance-stop".to_owned(),
        };
        match adapter.on_lifecycle_event(&stop_event) {
            Ok(()) => lifecycle_hooks_checked += 1,
            Err(err) => violations.push(adapter_error_violation("lifecycle_stop", &err)),
        }

        ConformanceReport::with_violations(
            fixtures.len(),
            emitted_events,
            lifecycle_hooks_checked,
            violations,
        )
    }
}

fn validate_instance(
    prefix: &str,
    instance: &TelemetryInstance,
    violations: &mut Vec<ConformanceViolation>,
) {
    if !is_valid_ulid(&instance.instance_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.invalid_instance_id".to_owned(),
            field: format!("{prefix}.instance_id"),
            message: "instance_id must be a 26-char ULID".to_owned(),
        });
    }
    if instance.project_key.trim().is_empty() {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.missing_project_key".to_owned(),
            field: format!("{prefix}.project_key"),
            message: "project_key must be non-empty".to_owned(),
        });
    }
    if instance.host_name.trim().is_empty() {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.missing_host_name".to_owned(),
            field: format!("{prefix}.host_name"),
            message: "host_name must be non-empty".to_owned(),
        });
    }
}

fn validate_search_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    query_text: &str,
    results: &TelemetrySearchResults,
    metrics: &TelemetrySearchMetrics,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("search.instance", instance, violations);
    validate_correlation("search.correlation", correlation, violations);
    if query_text.trim().is_empty() {
        violations.push(violation(
            "adapter.event.search.empty_query",
            "search.query.text",
            "query text must be non-empty",
        ));
    }
    let char_count = query_text.chars().count();
    if char_count > 500 {
        violations.push(ConformanceViolation {
            code: "adapter.event.search.query_too_long".to_owned(),
            field: "search.query.text".to_owned(),
            message: format!("query length {char_count} exceeds 500"),
        });
    }
    if metrics.latency_us == 0 {
        violations.push(violation(
            "adapter.event.search.zero_latency",
            "search.metrics.latency_us",
            "latency_us should be > 0 for completed search telemetry",
        ));
    }
    let source_total = results
        .lexical_count
        .saturating_add(results.semantic_count)
        .saturating_add(results.hash_control_count);
    if results.result_count > 0 && source_total == 0 {
        violations.push(violation(
            "adapter.event.search.missing_source_counts",
            "search.results",
            "result_count > 0 requires lexical_count, semantic_count, or hash_control_count to be non-zero",
        ));
    }
    if source_total > 0 && results.result_count > source_total {
        violations.push(ConformanceViolation {
            code: "adapter.event.search.result_count_exceeds_sources".to_owned(),
            field: "search.results.result_count".to_owned(),
            message: format!(
                "result_count {} exceeds lexical_count + semantic_count + hash_control_count ({source_total})",
                results.result_count
            ),
        });
    }
}

fn validate_duration_event(
    prefix: &str,
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    duration_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance(&format!("{prefix}.instance"), instance, violations);
    validate_correlation(&format!("{prefix}.correlation"), correlation, violations);
    if duration_ms == 0 {
        violations.push(ConformanceViolation {
            code: format!("adapter.event.{prefix}.zero_duration"),
            field: format!("{prefix}.duration_ms"),
            message: "duration_ms should be > 0 for completed telemetry".to_owned(),
        });
    }
}

fn validate_embedding_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    job: &TelemetryEmbeddingJob,
    embedder: &TelemetryEmbedderInfo,
    duration_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_duration_event("embedding", instance, correlation, duration_ms, violations);
    if job.job_id.trim().is_empty() {
        violations.push(violation(
            "adapter.event.embedding.missing_job_id",
            "embedding.job.job_id",
            "job_id must be non-empty",
        ));
    }
    if job.doc_count == 0 {
        violations.push(violation(
            "adapter.event.embedding.zero_doc_count",
            "embedding.job.doc_count",
            "doc_count should be > 0",
        ));
    }
    if embedder.id.trim().is_empty() {
        violations.push(violation(
            "adapter.event.embedding.missing_embedder_id",
            "embedding.embedder.id",
            "embedder id must be non-empty",
        ));
    }
    if embedder.dimension == 0 {
        violations.push(violation(
            "adapter.event.embedding.zero_dimension",
            "embedding.embedder.dimension",
            "dimension should be > 0",
        ));
    }
}

fn validate_index_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    dimension: usize,
    duration_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_duration_event("index", instance, correlation, duration_ms, violations);
    if dimension == 0 {
        violations.push(violation(
            "adapter.event.index.zero_dimension",
            "index.dimension",
            "dimension should be > 0",
        ));
    }
}

fn validate_resource_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    cpu_pct: f64,
    interval_ms: u64,
    load_avg_1m: Option<f64>,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("resource.instance", instance, violations);
    validate_correlation("resource.correlation", correlation, violations);
    if interval_ms == 0 {
        violations.push(violation(
            "adapter.event.resource.zero_interval",
            "resource.sample.interval_ms",
            "interval_ms must be greater than zero",
        ));
    }
    if !cpu_pct.is_finite() {
        violations.push(violation(
            "adapter.event.resource.invalid_cpu_pct",
            "resource.sample.cpu_pct",
            "cpu_pct must be finite",
        ));
    } else if !(0.0..=100.0).contains(&cpu_pct) {
        violations.push(ConformanceViolation {
            code: "adapter.event.resource.cpu_pct_out_of_range".to_owned(),
            field: "resource.sample.cpu_pct".to_owned(),
            message: format!("cpu_pct {cpu_pct} must be in [0, 100]"),
        });
    }
    if let Some(load_avg_1m) = load_avg_1m
        && (!load_avg_1m.is_finite() || load_avg_1m < 0.0)
    {
        violations.push(ConformanceViolation {
            code: "adapter.event.resource.invalid_load_avg_1m".to_owned(),
            field: "resource.sample.load_avg_1m".to_owned(),
            message: format!("load_avg_1m {load_avg_1m} must be finite and >= 0"),
        });
    }
}

fn validate_lifecycle_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    reason: Option<&str>,
    uptime_ms: Option<u64>,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("lifecycle.instance", instance, violations);
    validate_correlation("lifecycle.correlation", correlation, violations);
    if reason.is_some_and(|text| text.trim().is_empty()) {
        violations.push(violation(
            "adapter.event.lifecycle.empty_reason",
            "lifecycle.reason",
            "reason must be non-empty when provided",
        ));
    }
    if uptime_ms.is_some_and(|uptime| uptime == 0) {
        violations.push(violation(
            "adapter.event.lifecycle.zero_uptime",
            "lifecycle.uptime_ms",
            "uptime_ms should be > 0 when provided",
        ));
    }
}

fn validate_correlation(
    prefix: &str,
    correlation: &TelemetryCorrelation,
    violations: &mut Vec<ConformanceViolation>,
) {
    if !is_valid_ulid(&correlation.event_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_event_id".to_owned(),
            field: format!("{prefix}.event_id"),
            message: "event_id must be a 26-char ULID".to_owned(),
        });
    }
    if !is_valid_ulid(&correlation.root_request_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_root_request_id".to_owned(),
            field: format!("{prefix}.root_request_id"),
            message: "root_request_id must be a 26-char ULID".to_owned(),
        });
    }
    if let Some(parent_event_id) = &correlation.parent_event_id
        && !is_valid_ulid(parent_event_id)
    {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_parent_event_id".to_owned(),
            field: format!("{prefix}.parent_event_id"),
            message: "parent_event_id must be a 26-char ULID when provided".to_owned(),
        });
    }
}

fn validate_redaction_forbidden_substrings(
    envelope: &TelemetryEnvelope,
    forbidden_substrings: &[String],
) -> Vec<ConformanceViolation> {
    let mut violations = Vec::new();
    let payload = match serde_json::to_string(envelope) {
        Ok(payload) => payload,
        Err(source) => {
            violations.push(ConformanceViolation {
                code: "adapter.redaction.serialization_failed".to_owned(),
                field: "envelope".to_owned(),
                message: format!("failed to serialize envelope for redaction check: {source}"),
            });
            return violations;
        }
    };

    let payload_lower = payload.to_ascii_lowercase();
    for pattern in forbidden_substrings {
        if pattern.is_empty() {
            continue;
        }
        let pattern_lower = pattern.to_ascii_lowercase();
        if payload_lower.contains(&pattern_lower) {
            violations.push(ConformanceViolation {
                code: "adapter.redaction.forbidden_pattern".to_owned(),
                field: "envelope".to_owned(),
                message: format!("payload contains forbidden pattern '{pattern}'"),
            });
        }
    }

    violations
}

fn is_valid_ulid(candidate: &str) -> bool {
    if candidate.len() != 26 {
        return false;
    }
    candidate.bytes().all(|byte| {
        matches!(
            byte,
            b'0'..=b'9'
                | b'A'..=b'H'
                | b'J'..=b'K'
                | b'M'..=b'N'
                | b'P'..=b'T'
                | b'V'..=b'Z'
                | b'a'..=b'h'
                | b'j'..=b'k'
                | b'm'..=b'n'
                | b'p'..=b't'
                | b'v'..=b'z'
        )
    })
}

fn violation(code: &str, field: &str, message: &str) -> ConformanceViolation {
    ConformanceViolation {
        code: code.to_owned(),
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

fn adapter_error_violation(context: &str, error: &SearchError) -> ConformanceViolation {
    ConformanceViolation {
        code: "adapter.hook.error".to_owned(),
        field: context.to_owned(),
        message: error.to_string(),
    }
}

fn canonical_identity_pair_violation(
    adapter_id: &str,
    host_project: &str,
) -> Option<ConformanceViolation> {
    let adapter_id = adapter_id.trim();
    let host_project = host_project.trim();
    if adapter_id.is_empty() || host_project.is_empty() {
        return None;
    }

    let canonical_for_host = CanonicalHostProject::ALL
        .iter()
        .copied()
        .find(|host| host.host_project_key() == host_project);
    if let Some(host) = canonical_for_host
        && host.adapter_id() != adapter_id
    {
        return Some(ConformanceViolation {
            code: "adapter.identity.canonical_pair_mismatch".to_owned(),
            field: "identity.adapter_id".to_owned(),
            message: format!(
                "host_project '{}' expects adapter_id '{}', got '{}'",
                host_project,
                host.adapter_id(),
                adapter_id
            ),
        });
    }

    let canonical_for_adapter = CanonicalHostProject::ALL
        .iter()
        .copied()
        .find(|host| host.adapter_id() == adapter_id);
    if let Some(host) = canonical_for_adapter
        && host.host_project_key() != host_project
    {
        return Some(ConformanceViolation {
            code: "adapter.identity.canonical_pair_mismatch".to_owned(),
            field: "identity.host_project".to_owned(),
            message: format!(
                "adapter_id '{}' expects host_project '{}', got '{}'",
                adapter_id,
                host.host_project_key(),
                host_project
            ),
        });
    }

    None
}
