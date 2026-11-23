//! Unified E2E artifact schema types (bd-2hz.10.11.1).
//!
//! Defines the canonical envelope and body types for all e2e test artifacts:
//! manifest, events, oracle reports, replay recordings, and snapshot diffs.
//! All types are `Serialize`/`Deserialize` for JSON/JSONL emission.

mod emitter;
mod schema;
mod validation;

pub use emitter::{
    ArtifactEmissionInput, E2eArtifactEmitterError, build_artifact_entries,
    build_core_manifest_artifacts, normalize_artifact_file_name, normalize_replay_command,
    render_artifacts_index, sha256_checksum,
};
pub use schema::{
    ArtifactEntry, ClockMode, Correlation, DeterminismTier, DiffEntry, E2eEnvelope, E2eEventType,
    E2eOutcome, E2eSeverity, EventBody, ExitStatus, LaneReport, ManifestBody, ModelVersion,
    OracleReportBody, OracleVerdictRecord, Platform, ReplayBody, ReplayEventType, ReportTotals,
    SnapshotDiffBody, Suite, reason_codes,
};
pub use validation::{
    E2eArtifactValidationError, validate_envelope, validate_event_body, validate_event_envelope,
    validate_manifest_body, validate_manifest_envelope,
};

/// Schema version for the e2e artifact envelope.
pub const E2E_SCHEMA_VERSION: u32 = 1;
/// Schema discriminator for manifest artifacts.
pub const E2E_SCHEMA_MANIFEST: &str = "e2e-manifest-v1";
/// Schema discriminator for event artifacts.
pub const E2E_SCHEMA_EVENT: &str = "e2e-event-v1";
/// Schema discriminator for oracle report artifacts.
pub const E2E_SCHEMA_ORACLE_REPORT: &str = "e2e-oracle-report-v1";
/// Schema discriminator for replay artifacts.
pub const E2E_SCHEMA_REPLAY: &str = "e2e-replay-v1";
/// Schema discriminator for snapshot diff artifacts.
pub const E2E_SCHEMA_SNAPSHOT_DIFF: &str = "e2e-snapshot-diff-v1";

/// Mandatory structured events stream for unified e2e packs.
pub const E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL: &str = "structured_events.jsonl";
/// Canonical manifest file name inside an e2e artifact bundle.
pub const E2E_ARTIFACT_MANIFEST_JSON: &str = "manifest.json";
/// Mandatory environment capture for replay-first reproducibility.
pub const E2E_ARTIFACT_ENV_JSON: &str = "env.json";
/// Mandatory deterministic replay lockfile for reproducibility packs.
pub const E2E_ARTIFACT_REPRO_LOCK: &str = "repro.lock";
/// Mandatory artifact index for failed runs.
pub const E2E_ARTIFACT_ARTIFACTS_INDEX_JSON: &str = "artifacts_index.json";
/// Mandatory replay command pointer for failed runs.
pub const E2E_ARTIFACT_REPLAY_COMMAND_TXT: &str = "replay_command.txt";
/// Canonical replay stream filename for deterministic e2e packs.
pub const E2E_ARTIFACT_REPLAY_JSONL: &str = "replay.jsonl";
/// Mandatory terminal transcript for failed ops/ui lanes.
pub const E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT: &str = "terminal_transcript.txt";

#[cfg(test)]
use validation::{is_valid_reason_code, parse_canonical_rfc3339_utc};

#[cfg(test)]
mod tests;
