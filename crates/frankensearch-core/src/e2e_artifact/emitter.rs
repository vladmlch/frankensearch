use std::collections::BTreeMap;
use std::fmt::Write as _;

use sha2::{Digest, Sha256};
use thiserror::Error;

use super::schema::{ArtifactEntry, ExitStatus};
use super::{
    E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_ENV_JSON, E2E_ARTIFACT_MANIFEST_JSON,
    E2E_ARTIFACT_REPLAY_COMMAND_TXT, E2E_ARTIFACT_REPRO_LOCK, E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
};

/// One artifact payload candidate used by shared emitters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactEmissionInput<'a> {
    pub file: &'a str,
    pub bytes: &'a [u8],
    pub line_count: Option<u64>,
}

/// Errors surfaced by shared artifact-entry emitter helpers.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum E2eArtifactEmitterError {
    #[error("duplicate artifact file after normalization: '{file}'")]
    DuplicateNormalizedArtifactFile { file: String },
    #[error("artifact '{file}' is JSONL and must include line_count")]
    MissingLineCountForJsonl { file: String },
    #[error("artifact '{file}' is not JSONL and must not include line_count")]
    UnexpectedLineCountForNonJsonl { file: String },
    #[error("failed to render artifacts_index.json: {detail}")]
    ArtifactsIndexRender { detail: String },
    #[error("failed run requires replay command payload")]
    MissingReplayCommandForFailure,
}

/// Normalize artifact filenames to canonical v1 names.
///
/// Legacy names are mapped according to `docs/e2e-artifact-contract.md` and
/// path prefixes are stripped so all suites emit one stable filename grammar.
#[must_use]
pub fn normalize_artifact_file_name(file: &str) -> String {
    let trimmed = file.trim();
    let basename = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
    match basename {
        "run_manifest.json" => E2E_ARTIFACT_MANIFEST_JSON.to_owned(),
        "events.jsonl" => E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL.to_owned(),
        "artifacts-index.json" => E2E_ARTIFACT_ARTIFACTS_INDEX_JSON.to_owned(),
        "replay.txt" => E2E_ARTIFACT_REPLAY_COMMAND_TXT.to_owned(),
        _ => basename.to_owned(),
    }
}

/// Normalize replay commands so all suites emit equivalent, copy/paste-safe
/// command strings regardless of input whitespace formatting.
#[must_use]
pub fn normalize_replay_command(command: &str) -> String {
    let trimmed = command.trim();
    let raw = trimmed
        .strip_prefix('`')
        .and_then(|s| s.strip_suffix('`'))
        .unwrap_or(trimmed);

    let mut normalized = String::with_capacity(raw.len());
    let mut in_single_quotes = false;
    let mut in_double_quotes = false;
    let mut pending_space = false;

    for ch in raw.chars() {
        match ch {
            '\'' if !in_double_quotes => {
                in_single_quotes = !in_single_quotes;
                if pending_space && !normalized.is_empty() {
                    normalized.push(' ');
                    pending_space = false;
                }
                normalized.push(ch);
            }
            '"' if !in_single_quotes => {
                in_double_quotes = !in_double_quotes;
                if pending_space && !normalized.is_empty() {
                    normalized.push(' ');
                    pending_space = false;
                }
                normalized.push(ch);
            }
            c if c.is_whitespace() && !in_single_quotes && !in_double_quotes => {
                pending_space = !normalized.is_empty();
            }
            c => {
                if pending_space && !normalized.is_empty() {
                    normalized.push(' ');
                    pending_space = false;
                }
                normalized.push(c);
            }
        }
    }

    normalized.trim().to_owned()
}

/// Compute a canonical SHA-256 checksum string for artifact payload bytes.
#[must_use]
pub fn sha256_checksum(bytes: &[u8]) -> String {
    format!("sha256:{}", lower_hex(Sha256::digest(bytes)))
}

fn lower_hex(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

/// Build deterministic manifest artifact entries from raw artifact payloads.
///
/// Output is stably sorted by canonical filename and rejects duplicate names
/// after legacy-name normalization.
///
/// # Errors
///
/// Returns an error if canonicalized names collide or line-count contracts are
/// violated (`*.jsonl` requires `line_count`; non-JSONL forbids it).
pub fn build_artifact_entries<'a, I>(
    inputs: I,
) -> Result<Vec<ArtifactEntry>, E2eArtifactEmitterError>
where
    I: IntoIterator<Item = ArtifactEmissionInput<'a>>,
{
    let mut ordered_entries = BTreeMap::<String, ArtifactEntry>::new();

    for input in inputs {
        let file = normalize_artifact_file_name(input.file);
        let is_jsonl = file
            .rsplit_once('.')
            .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("jsonl"));

        if is_jsonl && input.line_count.is_none() {
            return Err(E2eArtifactEmitterError::MissingLineCountForJsonl { file });
        }
        if !is_jsonl && input.line_count.is_some() {
            return Err(E2eArtifactEmitterError::UnexpectedLineCountForNonJsonl { file });
        }

        let entry = ArtifactEntry {
            file: file.clone(),
            checksum: sha256_checksum(input.bytes),
            line_count: input.line_count,
        };

        if ordered_entries.insert(file.clone(), entry).is_some() {
            return Err(E2eArtifactEmitterError::DuplicateNormalizedArtifactFile { file });
        }
    }

    Ok(ordered_entries.into_values().collect())
}

/// Render a stable `artifacts_index.json` payload from manifest artifact entries.
///
/// # Errors
///
/// Returns an error if serialization fails.
pub fn render_artifacts_index(entries: &[ArtifactEntry]) -> Result<String, serde_json::Error> {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|left, right| left.file.cmp(&right.file));
    serde_json::to_string_pretty(&sorted)
}

/// Build canonical manifest artifact entries for core e2e lanes.
///
/// For all runs this emits the reproducibility core:
/// `structured_events.jsonl`, `env.json`, and `repro.lock`.
/// Failed/error runs additionally emit a normalized `replay_command.txt` entry
/// and materialize `artifacts_index.json` content.
///
/// Returns `(manifest_entries, artifacts_index_json)` where the second item is
/// `Some(...)` only for failed/error runs.
///
/// # Errors
///
/// Returns an error if payload contracts are invalid or artifact index
/// serialization fails.
pub fn build_core_manifest_artifacts(
    structured_events_jsonl: &[u8],
    structured_event_line_count: u64,
    exit_status: ExitStatus,
    replay_command: Option<&str>,
) -> Result<(Vec<ArtifactEntry>, Option<String>), E2eArtifactEmitterError> {
    let env_json_payload = r#"{"schema":"frankensearch.e2e.env.v1","captured_env":[]}"#;
    let repro_lock_payload = format!(
        "schema=frankensearch.e2e.repro-lock.v1\nstructured_events_line_count={structured_event_line_count}\nexit_status={}\n",
        match exit_status {
            ExitStatus::Pass => "pass",
            ExitStatus::Fail => "fail",
            ExitStatus::Error => "error",
        }
    );

    let mut emission_inputs = vec![ArtifactEmissionInput {
        file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
        bytes: structured_events_jsonl,
        line_count: Some(structured_event_line_count),
    }];
    emission_inputs.push(ArtifactEmissionInput {
        file: E2E_ARTIFACT_ENV_JSON,
        bytes: env_json_payload.as_bytes(),
        line_count: None,
    });
    emission_inputs.push(ArtifactEmissionInput {
        file: E2E_ARTIFACT_REPRO_LOCK,
        bytes: repro_lock_payload.as_bytes(),
        line_count: None,
    });

    let normalized_replay_command = if matches!(exit_status, ExitStatus::Fail | ExitStatus::Error) {
        let raw_replay_command =
            replay_command.ok_or(E2eArtifactEmitterError::MissingReplayCommandForFailure)?;
        Some(normalize_replay_command(raw_replay_command))
    } else {
        None
    };

    if let Some(replay_command_payload) = normalized_replay_command.as_deref() {
        emission_inputs.push(ArtifactEmissionInput {
            file: E2E_ARTIFACT_REPLAY_COMMAND_TXT,
            bytes: replay_command_payload.as_bytes(),
            line_count: None,
        });
    }

    let mut entries = build_artifact_entries(emission_inputs)?;
    let artifacts_index_json = if matches!(exit_status, ExitStatus::Fail | ExitStatus::Error) {
        let index_payload = render_artifacts_index(&entries).map_err(|err| {
            E2eArtifactEmitterError::ArtifactsIndexRender {
                detail: err.to_string(),
            }
        })?;
        entries.push(ArtifactEntry {
            file: E2E_ARTIFACT_ARTIFACTS_INDEX_JSON.to_owned(),
            checksum: sha256_checksum(index_payload.as_bytes()),
            line_count: None,
        });
        entries.sort_by(|left, right| left.file.cmp(&right.file));
        Some(index_payload)
    } else {
        None
    };

    Ok((entries, artifacts_index_json))
}
