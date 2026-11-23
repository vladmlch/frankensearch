use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Number;

use super::E2E_SCHEMA_VERSION;

// ─── Envelope ────────────────────────────────────────────────────────────────

/// Top-level envelope wrapping every e2e artifact (JSON object or JSONL line).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct E2eEnvelope<B> {
    /// Schema version (currently 1).
    pub v: u32,
    /// Artifact type discriminant (e.g. `e2e-manifest-v1`).
    pub schema: String,
    /// Unique run identifier (ULID, 26-char Crockford base32).
    pub run_id: String,
    /// RFC 3339 timestamp of artifact creation.
    pub ts: String,
    /// Type-specific body payload.
    pub body: B,
}

impl<B> E2eEnvelope<B> {
    pub fn new(schema: &str, run_id: &str, ts: &str, body: B) -> Self {
        Self {
            v: E2E_SCHEMA_VERSION,
            schema: schema.to_owned(),
            run_id: run_id.to_owned(),
            ts: ts.to_owned(),
            body,
        }
    }
}

// ─── Manifest ────────────────────────────────────────────────────────────────

/// Body of the `e2e-manifest-v1` artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestBody {
    pub suite: Suite,
    pub determinism_tier: DeterminismTier,
    pub seed: u64,
    pub config_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_version: Option<String>,
    pub model_versions: Vec<ModelVersion>,
    pub platform: Platform,
    pub clock_mode: ClockMode,
    pub tie_break_policy: String,
    pub artifacts: Vec<ArtifactEntry>,
    pub duration_ms: u64,
    pub exit_status: ExitStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Suite {
    Core,
    Fsfs,
    Ops,
    Interaction,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismTier {
    BitExact,
    Semantic,
    Statistical,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClockMode {
    Simulated,
    Frozen,
    Realtime,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExitStatus {
    Pass,
    Fail,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelVersion {
    pub name: String,
    pub revision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Platform {
    pub os: String,
    pub arch: String,
    pub rustc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactEntry {
    pub file: String,
    pub checksum: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_count: Option<u64>,
}

// ─── Events ──────────────────────────────────────────────────────────────────

/// Body of the `e2e-event-v1` artifact (one per JSONL line).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventBody {
    #[serde(rename = "type")]
    pub event_type: E2eEventType,
    pub correlation: Correlation,
    pub severity: E2eSeverity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<E2eOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<BTreeMap<String, Number>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum E2eEventType {
    E2eStart,
    E2eEnd,
    LaneStart,
    LaneEnd,
    OracleCheck,
    PhaseTransition,
    Assertion,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum E2eSeverity {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum E2eOutcome {
    Pass,
    Fail,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Correlation {
    pub event_id: String,
    pub root_request_id: String,
    pub parent_event_id: Option<String>,
}

// ─── Oracle Report ───────────────────────────────────────────────────────────

/// Body of the `e2e-oracle-report-v1` artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OracleReportBody {
    pub lanes: Vec<LaneReport>,
    pub totals: ReportTotals,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LaneReport {
    pub lane_id: String,
    pub seed: u64,
    pub query_count: u32,
    pub verdicts: Vec<OracleVerdictRecord>,
    pub pass_count: u32,
    pub fail_count: u32,
    pub skip_count: u32,
    pub all_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OracleVerdictRecord {
    pub oracle_id: String,
    pub outcome: E2eOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReportTotals {
    pub lanes_run: u32,
    pub lanes_passed: u32,
    pub oracles_pass: u32,
    pub oracles_fail: u32,
    pub oracles_skip: u32,
    pub all_passed: bool,
}

// ─── Replay ──────────────────────────────────────────────────────────────────

/// Body of the `e2e-replay-v1` artifact (one per JSONL line).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayBody {
    #[serde(rename = "type")]
    pub replay_type: ReplayEventType,
    pub offset_ms: u64,
    pub seq: u64,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayEventType {
    Query,
    ConfigChange,
    ClockAdvance,
    Signal,
}

// ─── Snapshot Diff ───────────────────────────────────────────────────────────

/// Body of the `e2e-snapshot-diff-v1` artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotDiffBody {
    pub comparison_mode: DeterminismTier,
    pub baseline_run_id: String,
    pub diffs: Vec<DiffEntry>,
    pub pass: bool,
    pub mismatch_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiffEntry {
    pub field_path: String,
    pub baseline: String,
    pub current: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
    pub within_tolerance: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<String>,
}

// ─── Reason Code Constants ───────────────────────────────────────────────────

/// E2e-specific reason code constants.
pub mod reason_codes {
    pub const ORACLE_PASS: &str = "e2e.oracle.pass";
    pub const ORACLE_ORDERING_VIOLATED: &str = "e2e.oracle.ordering_violated";
    pub const ORACLE_DUPLICATES_FOUND: &str = "e2e.oracle.duplicates_found";
    pub const ORACLE_PHASE_MISMATCH: &str = "e2e.oracle.phase_mismatch";
    pub const ORACLE_SCORE_NON_MONOTONIC: &str = "e2e.oracle.score_non_monotonic";
    pub const ORACLE_SKIP_FEATURE_DISABLED: &str = "e2e.oracle.skip_feature_disabled";
    pub const ORACLE_SKIP_STUB_BACKEND: &str = "e2e.oracle.skip_stub_backend";
    pub const RUN_SETUP_FAILED: &str = "e2e.run.setup_failed";
    pub const RUN_TIMEOUT: &str = "e2e.run.timeout";
    pub const REPLAY_SEED_MISMATCH: &str = "e2e.replay.seed_mismatch";
    pub const DIFF_TOLERANCE_EXCEEDED: &str = "e2e.diff.tolerance_exceeded";
    pub const DIFF_FIELD_MISSING: &str = "e2e.diff.field_missing";
}
