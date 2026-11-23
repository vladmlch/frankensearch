use std::collections::BTreeSet;

use thiserror::Error;

use super::schema::{
    E2eEnvelope, E2eEventType, E2eOutcome, EventBody, ExitStatus, ManifestBody, Suite,
};
use super::{
    E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_ENV_JSON, E2E_ARTIFACT_REPLAY_COMMAND_TXT,
    E2E_ARTIFACT_REPRO_LOCK, E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
    E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT, E2E_SCHEMA_EVENT, E2E_SCHEMA_MANIFEST,
    E2E_SCHEMA_VERSION,
};

/// Validation errors for unified e2e artifact bundles.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum E2eArtifactValidationError {
    #[error("unsupported e2e schema version: expected {expected}, found {found}")]
    UnsupportedSchemaVersion { expected: u32, found: u32 },
    #[error("schema tag mismatch: expected '{expected}', found '{found}'")]
    SchemaTagMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("invalid run_id '{run_id}': expected 26-char Crockford ULID")]
    InvalidRunId { run_id: String },
    #[error("invalid ts '{ts}': expected canonical RFC3339 UTC YYYY-MM-DDTHH:MM:SSZ")]
    InvalidTimestamp { ts: String },
    #[error(
        "invalid config_hash '{config_hash}': expected sha256: plus exactly 64 lowercase hexadecimal digits"
    )]
    InvalidConfigHash { config_hash: String },
    #[error("manifest missing required artifact entry '{required_file}'")]
    MissingRequiredArtifact { required_file: &'static str },
    #[error("manifest contains duplicate artifact entry '{file}'")]
    DuplicateArtifactEntry { file: String },
    #[error("artifact '{file}' is JSONL and must include line_count")]
    MissingLineCountForJsonl { file: String },
    #[error("artifact '{file}' is not JSONL and must not include line_count")]
    UnexpectedLineCountForNonJsonl { file: String },
    #[error("event '{event_type:?}' requires lane_id")]
    MissingLaneId { event_type: E2eEventType },
    #[error("oracle_check event requires oracle_id and outcome")]
    MissingOracleFields,
    #[error("event outcome '{outcome:?}' requires reason_code")]
    MissingReasonCode { outcome: E2eOutcome },
    #[error("reason_code '{reason_code}' does not match e2e namespace pattern")]
    InvalidReasonCode { reason_code: String },
}

/// Validate common envelope invariants shared by all e2e artifacts.
///
/// # Errors
///
/// Returns an error if the schema version, schema tag, or run ID is invalid.
pub fn validate_envelope<B>(
    envelope: &E2eEnvelope<B>,
    expected_schema: &'static str,
) -> Result<(), E2eArtifactValidationError> {
    if envelope.v != E2E_SCHEMA_VERSION {
        return Err(E2eArtifactValidationError::UnsupportedSchemaVersion {
            expected: E2E_SCHEMA_VERSION,
            found: envelope.v,
        });
    }
    if envelope.schema != expected_schema {
        return Err(E2eArtifactValidationError::SchemaTagMismatch {
            expected: expected_schema,
            found: envelope.schema.clone(),
        });
    }
    if !is_valid_ulid(&envelope.run_id) {
        return Err(E2eArtifactValidationError::InvalidRunId {
            run_id: envelope.run_id.clone(),
        });
    }
    parse_canonical_rfc3339_utc(&envelope.ts)?;
    Ok(())
}

/// Validate a manifest envelope and its body contract.
///
/// # Errors
///
/// Returns an error if envelope invariants or manifest body constraints are violated.
pub fn validate_manifest_envelope(
    envelope: &E2eEnvelope<ManifestBody>,
) -> Result<(), E2eArtifactValidationError> {
    validate_envelope(envelope, E2E_SCHEMA_MANIFEST)?;
    validate_manifest_body(&envelope.body)
}

/// Validate an event envelope and its body contract.
///
/// # Errors
///
/// Returns an error if envelope invariants or event body constraints are violated.
pub fn validate_event_envelope(
    envelope: &E2eEnvelope<EventBody>,
) -> Result<(), E2eArtifactValidationError> {
    validate_envelope(envelope, E2E_SCHEMA_EVENT)?;
    validate_event_body(&envelope.body)
}

/// Validate manifest body invariants that JSON shape checks cannot guarantee.
///
/// # Errors
///
/// Returns an error if required artifacts are missing, duplicated, or have invalid line count.
pub fn validate_manifest_body(body: &ManifestBody) -> Result<(), E2eArtifactValidationError> {
    if !is_canonical_config_hash(&body.config_hash) {
        return Err(E2eArtifactValidationError::InvalidConfigHash {
            config_hash: body.config_hash.clone(),
        });
    }
    let mut seen_files = BTreeSet::new();

    for artifact in &body.artifacts {
        if !seen_files.insert(artifact.file.as_str()) {
            return Err(E2eArtifactValidationError::DuplicateArtifactEntry {
                file: artifact.file.clone(),
            });
        }

        let is_jsonl = artifact
            .file
            .rsplit_once('.')
            .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("jsonl"));
        if is_jsonl && artifact.line_count.is_none() {
            return Err(E2eArtifactValidationError::MissingLineCountForJsonl {
                file: artifact.file.clone(),
            });
        }
        if !is_jsonl && artifact.line_count.is_some() {
            return Err(E2eArtifactValidationError::UnexpectedLineCountForNonJsonl {
                file: artifact.file.clone(),
            });
        }
    }

    if !seen_files.contains(E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL) {
        return Err(E2eArtifactValidationError::MissingRequiredArtifact {
            required_file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
        });
    }

    for required_file in [E2E_ARTIFACT_ENV_JSON, E2E_ARTIFACT_REPRO_LOCK] {
        if !seen_files.contains(required_file) {
            return Err(E2eArtifactValidationError::MissingRequiredArtifact { required_file });
        }
    }

    if matches!(body.exit_status, ExitStatus::Fail | ExitStatus::Error) {
        for required_file in [
            E2E_ARTIFACT_ARTIFACTS_INDEX_JSON,
            E2E_ARTIFACT_REPLAY_COMMAND_TXT,
        ] {
            if !seen_files.contains(required_file) {
                return Err(E2eArtifactValidationError::MissingRequiredArtifact { required_file });
            }
        }

        if body.suite == Suite::Ops && !seen_files.contains(E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT) {
            return Err(E2eArtifactValidationError::MissingRequiredArtifact {
                required_file: E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT,
            });
        }
    }

    Ok(())
}

/// Validate event body invariants for lane/oracle semantics.
///
/// # Errors
///
/// Returns an error if required fields for the event type are missing or invalid.
pub fn validate_event_body(event: &EventBody) -> Result<(), E2eArtifactValidationError> {
    match event.event_type {
        E2eEventType::LaneStart | E2eEventType::LaneEnd => {
            if !has_non_empty_text(event.lane_id.as_deref()) {
                return Err(E2eArtifactValidationError::MissingLaneId {
                    event_type: event.event_type,
                });
            }
        }
        E2eEventType::OracleCheck => {
            if !has_non_empty_text(event.oracle_id.as_deref()) || event.outcome.is_none() {
                return Err(E2eArtifactValidationError::MissingOracleFields);
            }
        }
        E2eEventType::E2eStart
        | E2eEventType::E2eEnd
        | E2eEventType::PhaseTransition
        | E2eEventType::Assertion => {}
    }

    if let Some(outcome) = event.outcome
        && matches!(outcome, E2eOutcome::Fail | E2eOutcome::Skip)
        && !has_non_empty_text(event.reason_code.as_deref())
    {
        return Err(E2eArtifactValidationError::MissingReasonCode { outcome });
    }

    if let Some(reason_code) = event.reason_code.as_deref()
        && !is_valid_reason_code(reason_code)
    {
        return Err(E2eArtifactValidationError::InvalidReasonCode {
            reason_code: reason_code.to_owned(),
        });
    }

    Ok(())
}

fn has_non_empty_text(value: Option<&str>) -> bool {
    value.is_some_and(|text| !text.is_empty())
}

fn is_valid_ulid(value: &str) -> bool {
    value.len() == 26 && value.bytes().all(is_valid_ulid_byte)
}

/// Admit only `YYYY-MM-DDTHH:MM:SSZ`. Numeric offsets, lowercase `z`,
/// fractions, whitespace, trailing data, and impossible calendar dates fail.
pub(super) fn parse_canonical_rfc3339_utc(ts: &str) -> Result<(), E2eArtifactValidationError> {
    let reject = || Err(E2eArtifactValidationError::InvalidTimestamp { ts: ts.to_owned() });
    let bytes = ts.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return reject();
    }
    let digit = |index: usize| -> Option<u32> {
        let byte = *bytes.get(index)?;
        byte.is_ascii_digit().then_some(u32::from(byte - b'0'))
    };
    let pair = |index: usize| -> Option<u32> { Some(digit(index)? * 10 + digit(index + 1)?) };
    let mut year_digits = [0_u32; 4];
    for (slot, index) in year_digits.iter_mut().zip(0_usize..4) {
        let Some(value) = digit(index) else {
            return reject();
        };
        *slot = value;
    }
    let year = year_digits[0] * 1000 + year_digits[1] * 100 + year_digits[2] * 10 + year_digits[3];
    let Some(month) = pair(5) else {
        return reject();
    };
    let Some(day) = pair(8) else {
        return reject();
    };
    let Some(hour) = pair(11) else {
        return reject();
    };
    let Some(minute) = pair(14) else {
        return reject();
    };
    let Some(second) = pair(17) else {
        return reject();
    };
    if !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 59 {
        return reject();
    }
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_gregorian_leap_year(year) => 29,
        2 => 28,
        _ => return reject(),
    };
    if day == 0 || day > days_in_month {
        return reject();
    }
    let reconstructed = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z");
    if reconstructed != ts {
        return reject();
    }
    Ok(())
}

const fn is_gregorian_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

fn is_canonical_config_hash(value: &str) -> bool {
    let Some(digest) = value.strip_prefix("sha256:") else {
        return false;
    };
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

const fn is_valid_ulid_byte(byte: u8) -> bool {
    matches!(
        byte,
        b'0'..=b'9' | b'A'..=b'H' | b'J'..=b'N' | b'P'..=b'T' | b'V'..=b'Z'
    )
}

pub(super) fn is_valid_reason_code(value: &str) -> bool {
    let mut parts = value.split('.');
    let Some(namespace) = parts.next() else {
        return false;
    };
    let Some(category) = parts.next() else {
        return false;
    };
    let Some(code) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    is_reason_segment(namespace, false)
        && is_reason_segment(category, true)
        && is_reason_segment(code, true)
}

fn is_reason_segment(segment: &str, allow_underscore: bool) -> bool {
    !segment.is_empty()
        && segment.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || (allow_underscore && byte == b'_')
        })
}
