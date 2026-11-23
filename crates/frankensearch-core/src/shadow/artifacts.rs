use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use serde::Serialize;

use super::types::{
    ShadowDegradationKind, ShadowDegradationRecord, ShadowDivergenceRecord, ShadowObservationRecord,
};
use super::{
    SHADOW_ARTIFACT_DIRECTORY, SHADOW_DEGRADATION_SCHEMA_VERSION, SHADOW_DEGRADATIONS_FILE,
    SHADOW_DIVERGENCES_FILE, SHADOW_OBSERVATION_SCHEMA_VERSION, SHADOW_OBSERVATIONS_FILE,
};

/// Cross-process evidence summary consumed by doctor/status surfaces.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShadowArtifactSummary {
    /// Number of completed, schema-valid observations.
    pub observation_count: u64,
    /// Number of schema-valid classified divergence records.
    pub divergence_count: u64,
    /// Number of schema-valid typed degradation records.
    pub degradation_count: u64,
    /// Number of malformed JSONL lines across all three streams.
    pub malformed_line_count: u64,
    /// Highest generation found in either stream.
    pub latest_generation: Option<u64>,
}

impl ShadowArtifactSummary {
    /// Read evidence below `<index_root>/.quill-shadow`.
    ///
    /// Missing streams are treated as empty. Other I/O failures are surfaced.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when an existing stream cannot be read.
    pub fn read_index_root(index_root: impl AsRef<Path>) -> io::Result<Self> {
        Self::read_artifact_directory(index_root.as_ref().join(SHADOW_ARTIFACT_DIRECTORY))
    }

    /// Read evidence directly from an artifact directory.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when an existing stream cannot be read.
    pub fn read_artifact_directory(directory: impl AsRef<Path>) -> io::Result<Self> {
        let directory = directory.as_ref();
        let mut summary = Self::default();
        read_jsonl_if_present(&directory.join(SHADOW_OBSERVATIONS_FILE), |line| {
            match serde_json::from_str::<ShadowObservationRecord>(line) {
                Ok(record) if record.schema_version == SHADOW_OBSERVATION_SCHEMA_VERSION => {
                    summary.observation_count = summary.observation_count.saturating_add(1);
                    summary.latest_generation = Some(
                        summary
                            .latest_generation
                            .map_or(record.manifest_generation, |generation| {
                                generation.max(record.manifest_generation)
                            }),
                    );
                }
                _ => {
                    summary.malformed_line_count = summary.malformed_line_count.saturating_add(1);
                }
            }
        })?;
        read_jsonl_if_present(&directory.join(SHADOW_DIVERGENCES_FILE), |line| {
            match serde_json::from_str::<ShadowDivergenceRecord>(line) {
                Ok(record) if record.validate().is_ok() => {
                    summary.divergence_count = summary.divergence_count.saturating_add(1);
                    summary.latest_generation = Some(
                        summary
                            .latest_generation
                            .map_or(record.manifest_generation, |generation| {
                                generation.max(record.manifest_generation)
                            }),
                    );
                }
                _ => {
                    summary.malformed_line_count = summary.malformed_line_count.saturating_add(1);
                }
            }
        })?;
        read_jsonl_if_present(&directory.join(SHADOW_DEGRADATIONS_FILE), |line| {
            match serde_json::from_str::<ShadowDegradationRecord>(line) {
                Ok(record) if record.schema_version == SHADOW_DEGRADATION_SCHEMA_VERSION => {
                    summary.degradation_count = summary.degradation_count.saturating_add(1);
                    summary.latest_generation = Some(
                        summary
                            .latest_generation
                            .map_or(record.manifest_generation, |generation| {
                                generation.max(record.manifest_generation)
                            }),
                    );
                }
                _ => {
                    summary.malformed_line_count = summary.malformed_line_count.saturating_add(1);
                }
            }
        })?;
        Ok(summary)
    }
}

fn read_jsonl_if_present(path: &Path, mut visit: impl FnMut(&str)) -> io::Result<()> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for line in BufReader::new(file).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            visit(&line);
        }
    }
    Ok(())
}

pub(super) struct ShadowArtifactSink {
    directory: PathBuf,
    append_lock: Mutex<()>,
}

impl ShadowArtifactSink {
    pub(super) fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            append_lock: Mutex::new(()),
        }
    }

    pub(super) fn append_divergence(&self, record: &ShadowDivergenceRecord) -> io::Result<()> {
        record.validate().map_err(io::Error::other)?;
        self.append_json(SHADOW_DIVERGENCES_FILE, record)
    }

    pub(super) fn append_degradation(&self, record: &ShadowDegradationRecord) -> io::Result<()> {
        self.append_json(SHADOW_DEGRADATIONS_FILE, record)
    }

    pub(super) fn append_observation(&self, record: &ShadowObservationRecord) -> io::Result<()> {
        self.append_json(SHADOW_OBSERVATIONS_FILE, record)
    }

    fn append_json(&self, name: &str, value: &impl Serialize) -> io::Result<()> {
        let _guard = self
            .append_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        fs::create_dir_all(&self.directory)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.directory.join(name))?;
        serde_json::to_writer(&mut file, value).map_err(io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_data()
    }
}

/// Append one typed preparation or lifecycle degradation.
///
/// This is used by serving adapters before a [`super::ShadowLexical`] or
/// [`super::ShadowLexicalObserver`] can be constructed. The directory is the
/// artifact directory itself, normally `<index>/.quill-shadow`.
///
/// # Errors
///
/// Returns an I/O error when the append-only JSONL record cannot be persisted.
pub fn append_shadow_degradation(
    artifact_directory: impl Into<PathBuf>,
    manifest_generation: u64,
    kind: ShadowDegradationKind,
    detail: impl Into<String>,
) -> io::Result<()> {
    ShadowArtifactSink::new(artifact_directory.into()).append_degradation(
        &ShadowDegradationRecord {
            schema_version: SHADOW_DEGRADATION_SCHEMA_VERSION,
            manifest_generation,
            kind,
            detail: detail.into(),
        },
    )
}
